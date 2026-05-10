//! 4-corner perspective transform with bilinear sampling.
//!
//! Given a source quadrilateral (`src_corners`) and a destination
//! quadrilateral (`dst_corners`), compute the unique 3×3 homography
//! matrix `H` such that for each `(sx, sy)` in the source quad:
//!
//! ```text
//!   [ dx · w ]       [ sx ]
//!   [ dy · w ] = H · [ sy ]
//!   [   w    ]       [  1 ]
//! ```
//!
//! Then walk every output pixel and inverse-map it through `H⁻¹` to
//! find a source coordinate. A bilinear sample at that source coord
//! becomes the output pixel. Output pixels whose inverse maps fall
//! outside the source rectangle are filled with the background colour
//! (default opaque black).
//!
//! Corners are listed in **clockwise order from top-left**:
//! `[top_left, top_right, bottom_right, bottom_left]`. Both arrays
//! must use this convention or the resulting transform will fold the
//! image.
//!
//! ```text
//! src: 0──────1     dst: 0──────────1
//!      │      │           \         │
//!      │      │            \        │
//!      3──────2             3──────2
//! ```
//!
//! Operates on packed `Rgb24` / `Rgba`. By default output dimensions
//! match the input (no canvas grow); pixels falling outside the
//! destination quad keep the background colour. Use
//! [`Perspective::with_background`] to override the background, and
//! [`Perspective::with_output_size`] to emit a larger / smaller canvas
//! than the input (useful when the destination quad escapes the
//! source rectangle, e.g. mapping onto a virtual screen wider than
//! the source).
//!
//! Solving the homography uses an 8×8 linear system: the eight
//! unknowns are the first eight entries of `H` (we fix `H[2][2] = 1`).
//! Each (src, dst) corner pair contributes two rows to the system —
//! one for `dx` and one for `dy`. Gaussian elimination with partial
//! pivoting solves the system in O(8³) time. See
//! `solve_homography` below.
//!
//! Returns [`Error::invalid`] when the source corners are degenerate
//! (collinear) — the linear system is singular and there's no unique
//! mapping.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Four-corner perspective warp.
#[derive(Clone, Debug)]
pub struct Perspective {
    pub src_corners: [(f32, f32); 4],
    pub dst_corners: [(f32, f32); 4],
    pub background: [u8; 4],
    /// Optional explicit output canvas size in pixels. `None` (the
    /// default) means "use the input frame's `(width, height)`". When
    /// `Some((w, h))` the filter emits a `w × h` frame regardless of
    /// the input dimensions — handy when the destination quad escapes
    /// the source rectangle.
    pub output_size: Option<(u32, u32)>,
}

impl Perspective {
    /// Construct a perspective filter mapping `src_corners` → `dst_corners`.
    pub fn new(src_corners: [(f32, f32); 4], dst_corners: [(f32, f32); 4]) -> Self {
        Self {
            src_corners,
            dst_corners,
            background: [0, 0, 0, 255],
            output_size: None,
        }
    }

    /// Override the background fill (RGBA — alpha is used on Rgba
    /// outputs and ignored on Rgb24).
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }

    /// Override the output canvas size. The destination quad can sit
    /// anywhere in this `[0, w-1] × [0, h-1]` rectangle — pixels
    /// outside the quad's footprint receive the background fill.
    /// Pass `(0, 0)` to disable (matches `output_size = None`).
    pub fn with_output_size(mut self, w: u32, h: u32) -> Self {
        self.output_size = if w == 0 || h == 0 { None } else { Some((w, h)) };
        self
    }
}

impl ImageFilter for Perspective {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Perspective requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let src_w = params.width as usize;
        let src_h = params.height as usize;
        // Destination canvas size — defaults to the input shape, can be
        // overridden by `with_output_size` to emit a wider / taller
        // canvas when the dst quad escapes the source rectangle.
        let (dst_w, dst_h) = self
            .output_size
            .map(|(w, h)| (w as usize, h as usize))
            .unwrap_or((src_w, src_h));

        let src = &input.planes[0];
        let row_bytes = dst_w * bpp;
        let mut out = vec![0u8; row_bytes * dst_h];

        if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        // Solve the forward homography src → dst, then invert it for
        // per-pixel back-mapping.
        let forward = solve_homography(&self.src_corners, &self.dst_corners)
            .ok_or_else(|| Error::invalid("Perspective: source corners are degenerate"))?;
        let inverse = invert_3x3(&forward)
            .ok_or_else(|| Error::invalid("Perspective: forward homography is non-invertible"))?;

        for py in 0..dst_h {
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..dst_w {
                let (sx, sy, w_h) = apply_homography(&inverse, px as f32, py as f32);
                let base = px * bpp;
                if w_h.abs() < 1e-9 {
                    write_background(&mut dst_row[base..base + bpp], &self.background, bpp);
                    continue;
                }
                let sx = sx / w_h;
                let sy = sy / w_h;
                // A tiny epsilon absorbs floating-point round-off at
                // the source-rectangle boundary so e.g. an sx that
                // computes to `15.0000005` for a 16-pixel-wide source
                // still hits the inside branch (and clamps to 15.0
                // inside `bilinear_sample_into`) instead of falling
                // through to the background fill.
                const EPS: f32 = 1e-4;
                if sx < -EPS
                    || sx > (src_w as f32 - 1.0) + EPS
                    || sy < -EPS
                    || sy > (src_h as f32 - 1.0) + EPS
                {
                    write_background(&mut dst_row[base..base + bpp], &self.background, bpp);
                    continue;
                }
                bilinear_sample_into(
                    &src.data,
                    src.stride,
                    src_w,
                    src_h,
                    bpp,
                    sx,
                    sy,
                    &mut dst_row[base..base + bpp],
                );
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_bytes,
                data: out,
            }],
        })
    }
}

fn write_background(dst: &mut [u8], bg: &[u8; 4], bpp: usize) {
    for (i, slot) in dst.iter_mut().enumerate().take(bpp) {
        *slot = bg[i.min(3)];
    }
}

/// Apply a 3×3 homography `H` to `(x, y, 1)`. Returns `(x', y', w')`
/// where `(x'/w', y'/w')` is the destination point.
fn apply_homography(h: &[[f32; 3]; 3], x: f32, y: f32) -> (f32, f32, f32) {
    (
        h[0][0] * x + h[0][1] * y + h[0][2],
        h[1][0] * x + h[1][1] * y + h[1][2],
        h[2][0] * x + h[2][1] * y + h[2][2],
    )
}

/// Solve for the 3×3 homography matrix `H` mapping `src[i]` → `dst[i]`
/// for `i in 0..4`. Returns `None` when the source quad is degenerate
/// (the linear system is singular). `H[2][2]` is normalised to `1.0`
/// — the standard convention since the matrix is defined up to scale.
///
/// Each (src, dst) pair contributes two equations:
///
/// ```text
///   dx · (h31·sx + h32·sy + 1) = h11·sx + h12·sy + h13
///   dy · (h31·sx + h32·sy + 1) = h21·sx + h22·sy + h23
/// ```
///
/// Rearranged into the canonical
/// `[ sx, sy, 1, 0, 0, 0, -dx·sx, -dx·sy ] · H̃ = dx`
/// `[ 0,  0,  0, sx, sy, 1, -dy·sx, -dy·sy ] · H̃ = dy`
/// form, where `H̃ = [h11, h12, h13, h21, h22, h23, h31, h32]`.
pub(crate) fn solve_homography(
    src: &[(f32, f32); 4],
    dst: &[(f32, f32); 4],
) -> Option<[[f32; 3]; 3]> {
    // 8x8 augmented matrix: 8 unknowns, 8 equations, 1 RHS column.
    let mut a = [[0f32; 9]; 8];
    for i in 0..4 {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];
        let row = i * 2;
        a[row] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, dx];
        a[row + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy, dy];
    }

    // Gauss-Jordan elimination with partial pivoting on rows.
    for col in 0..8 {
        // Find pivot row (largest absolute value in this column at or
        // below the diagonal).
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            let v = row[col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-9 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
        }
        // Normalise pivot row.
        let pv = a[col][col];
        for v in &mut a[col] {
            *v /= pv;
        }
        // Eliminate this column in every other row. We split the
        // matrix at `col` so the pivot row borrows distinctly from the
        // rows being modified, which keeps the borrow-checker happy
        // without a per-element index loop.
        let (head, tail) = a.split_at_mut(col);
        let (pivot_row_slice, rest) = tail.split_first_mut().expect("col < 8");
        for r in head.iter_mut().chain(rest.iter_mut()) {
            let factor = r[col];
            if factor == 0.0 {
                continue;
            }
            for (rv, pv) in r.iter_mut().zip(pivot_row_slice.iter()) {
                *rv -= factor * *pv;
            }
        }
    }

    let h = [
        [a[0][8], a[1][8], a[2][8]],
        [a[3][8], a[4][8], a[5][8]],
        [a[6][8], a[7][8], 1.0],
    ];
    Some(h)
}

/// Invert a 3×3 matrix via classical adjugate / determinant. Returns
/// `None` if the matrix is singular (`|det| < 1e-9`).
pub(crate) fn invert_3x3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let m_inv = [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ];
    Some(m_inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = f(x, y);
                data.extend_from_slice(&[r, g, b]);
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn identity_corners_round_trip_image() {
        // src == dst ⇒ the homography is the identity ⇒ output should
        // match the input byte-for-byte.
        let corners = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let f = Perspective::new(corners, corners);
        let input = rgb(16, 16, |x, y| (x as u8 * 16, y as u8 * 16, 50));
        let out = f.apply(&input, params_rgb(16, 16)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn translation_shifts_image() {
        // Translate the entire image one pixel to the right by mapping
        // src corners to (1, 0) - (16, 0) - (16, 15) - (1, 15).
        let src = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let dst = [(1.0, 0.0), (16.0, 0.0), (16.0, 15.0), (1.0, 15.0)];
        let f = Perspective::new(src, dst).with_background([10, 20, 30, 255]);
        let input = rgb(16, 16, |x, _| (x as u8 * 16, 0, 0));
        let out = f.apply(&input, params_rgb(16, 16)).unwrap();
        // Column 0 of output should now be background (input column -1
        // is out-of-bounds).
        assert_eq!(&out.planes[0].data[0..3], &[10, 20, 30]);
        // Column 1 of output should be input column 0 (R = 0).
        assert_eq!(out.planes[0].data[3], 0);
    }

    #[test]
    fn solve_homography_matches_identity() {
        let corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let h = solve_homography(&corners, &corners).unwrap();
        // Identity within float tolerance.
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (r, exp_row) in h.iter().zip(expected.iter()) {
            for (got, exp) in r.iter().zip(exp_row.iter()) {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "got {got}, expected {exp} (matrix {h:?})"
                );
            }
        }
    }

    #[test]
    fn solve_homography_rejects_degenerate_quad() {
        // All four corners on the same line ⇒ singular system.
        let collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let dst = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        assert!(solve_homography(&collinear, &dst).is_none());
    }

    #[test]
    fn invert_3x3_round_trips_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&id).unwrap();
        for (row, exp_row) in inv.iter().zip(id.iter()) {
            for (got, exp) in row.iter().zip(exp_row.iter()) {
                assert!((got - exp).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let corners = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)];
        let err = Perspective::new(corners, corners)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Perspective"));
    }

    #[test]
    fn output_size_grows_canvas_and_preserves_quad_mapping() {
        // r9: emit a 32×32 canvas from a 16×16 input. The src quad is
        // the full input rect; the dst quad maps the input into the
        // top-left 16×16 corner of the output. Pixels outside that
        // sub-rectangle must carry the background colour, and the
        // top-left 16×16 region must be a near-byte-copy of the input.
        let src = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let dst = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let f = Perspective::new(src, dst)
            .with_output_size(32, 32)
            .with_background([7, 8, 9, 255]);
        let input = rgb(16, 16, |x, y| (x as u8 * 16, y as u8 * 16, 50));
        let out = f.apply(&input, params_rgb(16, 16)).unwrap();

        // Output frame is 32×32 ⇒ stride = 32*3 = 96.
        assert_eq!(out.planes[0].data.len(), 32 * 32 * 3);
        assert_eq!(out.planes[0].stride, 96);

        // (20, 20) is outside the 16×16 dst quad ⇒ background.
        let off = (20 * 32 + 20) * 3;
        assert_eq!(&out.planes[0].data[off..off + 3], &[7, 8, 9]);

        // (5, 7) is inside the dst quad ⇒ should match the input pixel
        // at (5, 7) modulo bilinear-sampling rounding (identity quad ⇒
        // exact integer back-map ⇒ exact match).
        let in_off = (7 * 16 + 5) * 3;
        let out_off = (7 * 32 + 5) * 3;
        assert_eq!(
            &out.planes[0].data[out_off..out_off + 3],
            &input.planes[0].data[in_off..in_off + 3]
        );
    }

    #[test]
    fn output_size_shrink_canvas_clamps_to_quad_footprint() {
        // Emit an 8×8 canvas from a 16×16 input by pinning the dst
        // quad to (0, 0)..(7, 7). The whole 8×8 canvas should be
        // covered by the quad (no background pixels visible).
        let src = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let dst = [(0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)];
        let f = Perspective::new(src, dst)
            .with_output_size(8, 8)
            .with_background([222, 222, 222, 255]);
        let input = rgb(16, 16, |_, _| (40, 60, 80));
        let out = f.apply(&input, params_rgb(16, 16)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
        // Every pixel should sample the flat input colour, NOT the
        // background.
        for y in 0..8 {
            for x in 0..8 {
                let off = (y * 8 + x) * 3;
                assert_eq!(&out.planes[0].data[off..off + 3], &[40, 60, 80]);
            }
        }
    }

    #[test]
    fn perspective_keystone_warps_and_keeps_background() {
        // Pinch the top edge inward by 4 pixels on each side — the
        // top of the output becomes a smaller patch from the top of
        // the input. Outside that smaller patch the output should
        // carry the background colour.
        let src = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let dst = [(4.0, 0.0), (11.0, 0.0), (15.0, 15.0), (0.0, 15.0)];
        let f = Perspective::new(src, dst).with_background([200, 200, 200, 255]);
        let input = rgb(16, 16, |_, _| (50, 50, 50));
        let out = f.apply(&input, params_rgb(16, 16)).unwrap();
        // Top-left output pixel (0, 0) is outside the new top edge
        // ([4, 11] x 0) ⇒ should carry background.
        assert_eq!(&out.planes[0].data[0..3], &[200, 200, 200]);
        // Bottom-row output (still mapped to source bottom row) should
        // be input colour.
        assert_eq!(&out.planes[0].data[15 * 48..15 * 48 + 3], &[50, 50, 50]);
    }
}
