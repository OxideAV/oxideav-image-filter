//! 2-D affine transform (`-distort Affine 'sx,ry,rx,sy,tx,ty'`).
//!
//! Applies an arbitrary 2×3 affine map (six coefficients) to the input
//! image with bilinear resampling. The forward transform takes a source
//! coordinate `(sx, sy)` to a destination coordinate `(dx, dy)` via
//!
//! ```text
//!   [ dx ]   [ sx_coef  ry_coef  tx ] [ sx ]
//!   [ dy ] = [ rx_coef  sy_coef  ty ] [ sy ]
//!                                     [  1 ]
//! ```
//!
//! At apply time we compute the inverse 2×3 matrix and inverse-map every
//! output pixel back to the source. Coordinates outside the source
//! rectangle take the configured background colour.
//!
//! Output canvas size defaults to the input size; use
//! [`Affine::with_output_size`] to override (handy when the forward
//! transform translates / scales the image off-canvas).
//!
//! # Pixel formats
//!
//! Packed `Rgb24` and `Rgba` only — geometric warps make no sense on
//! sub-sampled chroma without a full pipeline-aware re-rasteriser.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// 2-D affine warp.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    /// Six matrix coefficients in `(sx, ry, rx, sy, tx, ty)` order
    /// matching ImageMagick's `-distort Affine` argument convention:
    ///
    /// ```text
    ///   dx = sx * sx_in + ry * sy_in + tx
    ///   dy = rx * sx_in + sy * sy_in + ty
    /// ```
    ///
    /// Identity is `(1, 0, 0, 1, 0, 0)`.
    pub matrix: [f32; 6],
    /// Background colour for out-of-source pixels (RGBA).
    pub background: [u8; 4],
    /// Optional explicit output canvas size. `None` ⇒ use the input
    /// frame's `(width, height)`.
    pub output_size: Option<(u32, u32)>,
}

impl Affine {
    /// Build an affine warp with `(sx, ry, rx, sy, tx, ty)` coefficients.
    pub fn new(matrix: [f32; 6]) -> Self {
        Self {
            matrix,
            background: [0, 0, 0, 255],
            output_size: None,
        }
    }

    /// Identity affine warp (output identical to input).
    pub fn identity() -> Self {
        Self::new([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    }

    /// Override the background fill (RGBA).
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }

    /// Override the output canvas size in pixels. Pass `(0, 0)` to
    /// disable (matches `output_size = None`).
    pub fn with_output_size(mut self, w: u32, h: u32) -> Self {
        self.output_size = if w == 0 || h == 0 { None } else { Some((w, h)) };
        self
    }

    /// 3×3 form of the affine matrix (last row implicit `(0, 0, 1)`).
    /// Useful when composing with a perspective transform.
    pub fn as_3x3(&self) -> [[f32; 3]; 3] {
        let m = self.matrix;
        [[m[0], m[1], m[4]], [m[2], m[3], m[5]], [0.0, 0.0, 1.0]]
    }
}

impl ImageFilter for Affine {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Affine requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let src_w = params.width as usize;
        let src_h = params.height as usize;
        let (dst_w, dst_h) = self
            .output_size
            .map(|(w, h)| (w as usize, h as usize))
            .unwrap_or((src_w, src_h));
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

        let inv = invert_affine(self.matrix)
            .ok_or_else(|| Error::invalid("Affine: matrix is singular (det == 0)"))?;
        let src = &input.planes[0];

        for py in 0..dst_h {
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..dst_w {
                let (sx, sy) = apply_affine(inv, px as f32, py as f32);
                let base = px * bpp;
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

/// Apply a 2×3 affine matrix `(sx, ry, rx, sy, tx, ty)` to `(x, y)`.
#[inline]
pub(crate) fn apply_affine(m: [f32; 6], x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[1] * y + m[4], m[2] * x + m[3] * y + m[5])
}

/// Invert a 2×3 affine matrix. Returns `None` if the linear part is
/// singular (`|sx*sy - ry*rx| < eps`).
pub(crate) fn invert_affine(m: [f32; 6]) -> Option<[f32; 6]> {
    let (a, b, c, d, tx, ty) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    let det = a * d - b * c;
    if det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);
    Some([ia, ib, ic, id, itx, ity])
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

    #[test]
    fn identity_passes_pixels_through() {
        let input = rgb(4, 4, |x, y| ((x * 32) as u8, (y * 32) as u8, 64));
        let out = Affine::identity()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn translation_shifts_pixels() {
        let input = rgb(4, 4, |x, y| (((x * 32) as u8), (y * 32) as u8, 0));
        let f = Affine::new([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).with_background([255, 255, 255, 255]);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(&out.planes[0].data[0..3], &[255, 255, 255]);
        assert_eq!(&out.planes[0].data[3..6], &[0, 0, 0]);
        assert_eq!(&out.planes[0].data[6..9], &[32, 0, 0]);
    }

    #[test]
    fn scale_two_doubles_canvas_use() {
        let input = rgb(2, 2, |x, _| (((x * 100) as u8), 0, 0));
        let f = Affine::new([2.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .with_output_size(4, 4)
            .with_background([255, 255, 255, 255]);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 0);
        let r1 = out.planes[0].data[3];
        assert!((r1 as i16 - 50).abs() <= 2, "got {r1}");
        assert_eq!(out.planes[0].data[6], 100);
        assert_eq!(&out.planes[0].data[9..12], &[255, 255, 255]);
    }

    #[test]
    fn rotate_90_via_matrix() {
        let input = rgb(3, 3, |x, y| (((x + y * 3) * 20) as u8, 0, 0));
        let f = Affine::new([0.0, -1.0, 1.0, 0.0, 2.0, 0.0]);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 3,
                    height: 3,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 120);
        let bottom_right = (2 * 3 + 2) * 3;
        assert_eq!(out.planes[0].data[bottom_right], 40);
    }

    #[test]
    fn singular_matrix_is_invalid() {
        let input = rgb(2, 2, |_, _| (0, 0, 0));
        let f = Affine::new([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        let err = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("singular"), "{err}");
    }

    #[test]
    fn invert_then_apply_roundtrips_to_identity() {
        let m = [1.5, 0.3, -0.2, 0.8, 5.0, -3.0];
        let inv = invert_affine(m).unwrap();
        let (fx, fy) = apply_affine(m, 0.0, 0.0);
        let (bx, by) = apply_affine(inv, fx, fy);
        assert!(bx.abs() < 1e-4, "bx = {bx}");
        assert!(by.abs() < 1e-4, "by = {by}");
        let (fx, fy) = apply_affine(m, 7.0, 11.0);
        let (bx, by) = apply_affine(inv, fx, fy);
        assert!((bx - 7.0).abs() < 1e-3, "bx = {bx}");
        assert!((by - 11.0).abs() < 1e-3, "by = {by}");
    }
}
