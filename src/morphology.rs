//! Greyscale morphology: dilate / erode / open / close.
//!
//! Each iteration scans the input with a small structuring element and
//! replaces every output sample by either the **maximum** (dilate) or
//! **minimum** (erode) of the source samples it covers. Out-of-bounds
//! taps clamp to the nearest edge so the image bounds don't introduce
//! spurious dark/bright fringes.
//!
//! - **Open** = erode → dilate. Removes bright specks smaller than the
//!   structuring element while preserving the bulk of larger features.
//! - **Close** = dilate → erode. Fills in dark holes smaller than the
//!   structuring element.
//!
//! ```text
//! For each iteration, output (px, py), channel c:
//!     dilate: out[py, px, c] = max over (kx, ky) in element of in[py+ky, px+kx, c]
//!     erode:  out[py, px, c] = min over (kx, ky) in element of in[py+ky, px+kx, c]
//! ```
//!
//! Two structuring elements are supported:
//!
//! - **`Square3x3`** — every offset in `[-1, 1] × [-1, 1]` (8-connected).
//! - **`Cross3x3`** — only `(0, 0)`, `(±1, 0)`, `(0, ±1)` (4-connected).
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV (`Yuv420P` /
//! `Yuv422P` / `Yuv444P`). On RGBA the alpha channel passes through
//! unchanged (matches the principle that morphology is a per-channel
//! tonal operation, and alpha isn't tonal). On planar YUV every plane
//! is processed (chroma included) — a future revision could grow a
//! `Planes` selector if a luma-only mode is needed.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, plane_dims};
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which extremum the morphology pass takes over its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphologyOp {
    /// Pixel becomes the **maximum** of the window — bright regions
    /// expand. IM: `-morphology Dilate`.
    Dilate,
    /// Pixel becomes the **minimum** of the window — bright regions
    /// shrink. IM: `-morphology Erode`.
    Erode,
}

/// Structuring element shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StructuringElement {
    /// 3×3 square (8-connected). 9 taps.
    #[default]
    Square3x3,
    /// 3×3 cross / plus (4-connected). 5 taps.
    Cross3x3,
}

impl StructuringElement {
    /// Yields the `(dx, dy)` offsets covered by this structuring element.
    pub fn offsets(&self) -> &'static [(i32, i32)] {
        match self {
            StructuringElement::Square3x3 => &[
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ],
            StructuringElement::Cross3x3 => &[(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)],
        }
    }
}

/// Repeated dilate / erode pass. `iterations` controls how many times
/// the same op is applied before returning; `0` is a clone of the
/// input. For `Open` and `Close` build it as two `Morphology` filters
/// chained — see [`Morphology::open`] / [`Morphology::close`].
#[derive(Clone, Debug)]
pub struct Morphology {
    pub op: MorphologyOp,
    pub element: StructuringElement,
    pub iterations: u32,
}

impl Morphology {
    /// Construct a single-direction morphology pass.
    pub fn new(op: MorphologyOp, element: StructuringElement, iterations: u32) -> Self {
        Self {
            op,
            element,
            iterations,
        }
    }

    /// Convenience: `iterations` rounds of dilation with the given
    /// structuring element.
    pub fn dilate(element: StructuringElement, iterations: u32) -> Self {
        Self::new(MorphologyOp::Dilate, element, iterations)
    }

    /// Convenience: `iterations` rounds of erosion with the given
    /// structuring element.
    pub fn erode(element: StructuringElement, iterations: u32) -> Self {
        Self::new(MorphologyOp::Erode, element, iterations)
    }
}

impl ImageFilter for Morphology {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Morphology does not yet handle {:?}",
                params.format
            )));
        }
        if self.iterations == 0 {
            return Ok(input.clone());
        }

        let (cx, cy) = chroma_subsampling(params.format);
        let mut out = input.clone();
        let offsets = self.element.offsets();

        for (idx, plane) in out.planes.iter_mut().enumerate() {
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            let alpha_passthrough = matches!(params.format, PixelFormat::Rgba) && idx == 0;
            for _ in 0..self.iterations {
                *plane = morph_plane(plane, pw, ph, bpp, self.op, offsets, alpha_passthrough);
            }
        }

        Ok(out)
    }
}

/// `Open` (erode then dilate) and `Close` (dilate then erode) compose
/// two passes. Implemented as a free-standing helper so callers can
/// stitch them through a single `apply` rather than chaining two
/// filters manually.
impl Morphology {
    /// Build an opening: erode then dilate. Returns a closure-like
    /// helper expressed as a [`MorphologyChain`].
    pub fn open(element: StructuringElement, iterations: u32) -> MorphologyChain {
        MorphologyChain {
            first: Morphology::erode(element, iterations),
            second: Morphology::dilate(element, iterations),
        }
    }

    /// Build a closing: dilate then erode.
    pub fn close(element: StructuringElement, iterations: u32) -> MorphologyChain {
        MorphologyChain {
            first: Morphology::dilate(element, iterations),
            second: Morphology::erode(element, iterations),
        }
    }
}

/// Two-pass morphology composition (open / close).
#[derive(Clone, Debug)]
pub struct MorphologyChain {
    pub first: Morphology,
    pub second: Morphology,
}

impl ImageFilter for MorphologyChain {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let mid = self.first.apply(input, params)?;
        self.second.apply(&mid, params)
    }
}

/// Morphological edge operators built on top of dilate / erode.
///
/// All three pick out per-pixel changes induced by the structuring
/// element relative to the source — i.e. the boundary between
/// "inside" and "outside" of bright regions:
///
/// - `EdgeIn = src - erode(src)`. Highlights pixels that are present in
///   the source but disappear after erosion. These sit on the **inner
///   boundary** of bright regions.
/// - `EdgeOut = dilate(src) - src`. Highlights pixels that the
///   structuring element grows into but the source didn't hold —
///   the **outer boundary** of bright regions.
/// - `EdgeMagnitude = dilate(src) - erode(src)`. The full per-pixel
///   "morphological gradient" — brightest along edges, dark inside
///   uniform regions.
///
/// Subtraction is per-channel and clamped at 0 (saturating). Built from
/// the same primitives as [`Morphology`] so the structuring element
/// behaviour matches exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeOp {
    /// `src - erode(src)` (inner boundary).
    EdgeIn,
    /// `dilate(src) - src` (outer boundary).
    EdgeOut,
    /// `dilate(src) - erode(src)` (full morphological gradient).
    EdgeMagnitude,
}

/// Morphological edge / gradient filter — see [`EdgeOp`] for the
/// operator family.
///
/// Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV. On RGBA the
/// alpha channel passes through unchanged (matching `Morphology`'s
/// "alpha isn't tonal" rule). On planar YUV every plane is processed.
#[derive(Clone, Debug)]
pub struct MorphologyEdge {
    pub op: EdgeOp,
    pub element: StructuringElement,
}

impl MorphologyEdge {
    /// Build an edge filter with the given operator and structuring
    /// element.
    pub fn new(op: EdgeOp, element: StructuringElement) -> Self {
        Self { op, element }
    }

    /// Convenience: inner-boundary edge.
    pub fn edge_in(element: StructuringElement) -> Self {
        Self::new(EdgeOp::EdgeIn, element)
    }

    /// Convenience: outer-boundary edge.
    pub fn edge_out(element: StructuringElement) -> Self {
        Self::new(EdgeOp::EdgeOut, element)
    }

    /// Convenience: full morphological gradient.
    pub fn edge_magnitude(element: StructuringElement) -> Self {
        Self::new(EdgeOp::EdgeMagnitude, element)
    }
}

impl ImageFilter for MorphologyEdge {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        // Reuse the same code path as `Morphology` for both passes.
        let dilated_or_erode = match self.op {
            EdgeOp::EdgeIn => Morphology::erode(self.element, 1).apply(input, params)?,
            EdgeOp::EdgeOut | EdgeOp::EdgeMagnitude => {
                Morphology::dilate(self.element, 1).apply(input, params)?
            }
        };
        let other = match self.op {
            EdgeOp::EdgeIn | EdgeOp::EdgeOut => None,
            EdgeOp::EdgeMagnitude => Some(Morphology::erode(self.element, 1).apply(input, params)?),
        };

        // Per-channel saturating subtract, alpha-preserving on RGBA.
        let alpha_passthrough = matches!(params.format, PixelFormat::Rgba);
        let mut out = input.clone();
        for (idx, plane) in out.planes.iter_mut().enumerate() {
            let bpp = bytes_per_plane_pixel(params.format, idx);
            // Choose source operands per op:
            //   EdgeIn        : src - erode
            //   EdgeOut       : dilate - src
            //   EdgeMagnitude : dilate - erode
            let (a_data, b_data) = match self.op {
                EdgeOp::EdgeIn => (
                    input.planes[idx].data.as_slice(),
                    dilated_or_erode.planes[idx].data.as_slice(),
                ),
                EdgeOp::EdgeOut => (
                    dilated_or_erode.planes[idx].data.as_slice(),
                    input.planes[idx].data.as_slice(),
                ),
                EdgeOp::EdgeMagnitude => (
                    dilated_or_erode.planes[idx].data.as_slice(),
                    other.as_ref().unwrap().planes[idx].data.as_slice(),
                ),
            };
            let a_stride = match self.op {
                EdgeOp::EdgeIn => input.planes[idx].stride,
                EdgeOp::EdgeOut | EdgeOp::EdgeMagnitude => dilated_or_erode.planes[idx].stride,
            };
            let b_stride = match self.op {
                EdgeOp::EdgeIn => dilated_or_erode.planes[idx].stride,
                EdgeOp::EdgeOut => input.planes[idx].stride,
                EdgeOp::EdgeMagnitude => other.as_ref().unwrap().planes[idx].stride,
            };
            sub_plane(
                plane,
                a_data,
                a_stride,
                b_data,
                b_stride,
                bpp,
                alpha_passthrough,
            );
        }

        Ok(out)
    }
}

/// In-place per-channel saturating subtract `out = a - b`. Skips the
/// alpha channel on RGBA so the original alpha survives the edge pass.
fn sub_plane(
    out: &mut VideoPlane,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    bpp: usize,
    alpha_passthrough: bool,
) {
    // Output rows are tightly packed (no padding) — Morphology emits
    // them that way and the input plane (when used as `a` in EdgeIn)
    // may have different stride. We respect that.
    let out_stride = out.stride;
    let row_bytes = out_stride; // tightly packed for our generated planes
    let h = out.data.len() / out_stride;
    for y in 0..h {
        for x in 0..(row_bytes / bpp) {
            for ch in 0..bpp {
                let off = x * bpp + ch;
                if alpha_passthrough && bpp == 4 && ch == 3 {
                    // Preserve whichever alpha lives at `a` — by
                    // construction (above) `a` is always the operand
                    // whose alpha we want to keep (input or dilate-of-
                    // input; both carry the original alpha because
                    // dilate alpha-passes-through too).
                    let _ = b;
                    out.data[y * out_stride + off] = a[y * a_stride + off];
                    continue;
                }
                let av = a[y * a_stride + off];
                let bv = b[y * b_stride + off];
                out.data[y * out_stride + off] = av.saturating_sub(bv);
            }
        }
    }
}

fn morph_plane(
    plane: &VideoPlane,
    w: u32,
    h: u32,
    bpp: usize,
    op: MorphologyOp,
    offsets: &[(i32, i32)],
    alpha_passthrough: bool,
) -> VideoPlane {
    let w = w as usize;
    let h = h as usize;
    let stride = plane.stride;
    let mut out = vec![0u8; w * bpp * h];
    let last_x = (w as i32 - 1).max(0);
    let last_y = (h as i32 - 1).max(0);

    for y in 0..h {
        for x in 0..w {
            for ch in 0..bpp {
                if alpha_passthrough && bpp == 4 && ch == 3 {
                    out[y * w * bpp + x * bpp + ch] = plane.data[y * stride + x * bpp + ch];
                    continue;
                }
                let mut acc: u8 = match op {
                    MorphologyOp::Dilate => 0,
                    MorphologyOp::Erode => 255,
                };
                for (dx, dy) in offsets {
                    let sx = (x as i32 + dx).clamp(0, last_x) as usize;
                    let sy = (y as i32 + dy).clamp(0, last_y) as usize;
                    let v = plane.data[sy * stride + sx * bpp + ch];
                    acc = match op {
                        MorphologyOp::Dilate => acc.max(v),
                        MorphologyOp::Erode => acc.min(v),
                    };
                }
                out[y * w * bpp + x * bpp + ch] = acc;
            }
        }
    }

    VideoPlane {
        stride: w * bpp,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_frame(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    fn params_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flat_image_unchanged_by_dilate() {
        let input = gray_frame(8, 8, |_, _| 100);
        let out = Morphology::dilate(StructuringElement::Square3x3, 3)
            .apply(&input, params_gray(8, 8))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 100));
    }

    #[test]
    fn flat_image_unchanged_by_erode() {
        let input = gray_frame(8, 8, |_, _| 60);
        let out = Morphology::erode(StructuringElement::Square3x3, 3)
            .apply(&input, params_gray(8, 8))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 60));
    }

    #[test]
    fn dilate_grows_bright_spot() {
        // Single bright pixel at the centre of a dark field.
        let input = gray_frame(7, 7, |x, y| if x == 3 && y == 3 { 200 } else { 10 });
        let out = Morphology::dilate(StructuringElement::Square3x3, 1)
            .apply(&input, params_gray(7, 7))
            .unwrap();
        // After 1 iteration of square dilation the 3×3 block around the
        // centre should all be 200.
        for y in 2..=4 {
            for x in 2..=4 {
                assert_eq!(
                    out.planes[0].data[y * 7 + x],
                    200,
                    "(x={x}, y={y}) should have grown"
                );
            }
        }
        // Outside the 3×3 block the dark field still dominates.
        assert_eq!(out.planes[0].data[0], 10);
    }

    #[test]
    fn erode_shrinks_bright_spot() {
        // A 3×3 bright square at the centre.
        let input = gray_frame(7, 7, |x, y| {
            if (2..=4).contains(&x) && (2..=4).contains(&y) {
                200
            } else {
                10
            }
        });
        let out = Morphology::erode(StructuringElement::Square3x3, 1)
            .apply(&input, params_gray(7, 7))
            .unwrap();
        // After erosion only the centre survives — every pixel of the
        // 3×3 square had at least one dark neighbour except the
        // innermost (3, 3), whose 8-neighbourhood is fully bright.
        assert_eq!(out.planes[0].data[3 * 7 + 3], 200);
        // Edges of the previous square should have gone dark.
        assert_eq!(out.planes[0].data[2 * 7 + 2], 10);
    }

    #[test]
    fn cross_element_only_touches_4_neighbours() {
        // Centre is bright; corners 1 step away should NOT pick it up
        // through a cross structuring element.
        let input = gray_frame(5, 5, |x, y| if x == 2 && y == 2 { 200 } else { 0 });
        let out = Morphology::dilate(StructuringElement::Cross3x3, 1)
            .apply(&input, params_gray(5, 5))
            .unwrap();
        // 4-neighbours of (2, 2) should have been lifted to 200.
        assert_eq!(out.planes[0].data[2 * 5 + 1], 200);
        assert_eq!(out.planes[0].data[2 * 5 + 3], 200);
        assert_eq!(out.planes[0].data[5 + 2], 200);
        assert_eq!(out.planes[0].data[3 * 5 + 2], 200);
        // Diagonal neighbour (1, 1) must NOT have been lifted.
        assert_eq!(out.planes[0].data[5 + 1], 0);
    }

    #[test]
    fn open_removes_isolated_speck() {
        // A lone bright pixel surrounded by darkness — open should kill
        // it (erode wipes it; dilate has nothing left to grow back).
        let input = gray_frame(7, 7, |x, y| if x == 3 && y == 3 { 200 } else { 10 });
        let out = Morphology::open(StructuringElement::Square3x3, 1)
            .apply(&input, params_gray(7, 7))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 10));
    }

    #[test]
    fn close_fills_isolated_hole() {
        // A lone dark hole inside a bright field — close should patch it.
        let input = gray_frame(7, 7, |x, y| if x == 3 && y == 3 { 10 } else { 200 });
        let out = Morphology::close(StructuringElement::Square3x3, 1)
            .apply(&input, params_gray(7, 7))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 200));
    }

    #[test]
    fn iterations_zero_is_passthrough() {
        let input = gray_frame(4, 4, |x, y| (x * 30 + y * 20) as u8);
        let out = Morphology::dilate(StructuringElement::Square3x3, 0)
            .apply(&input, params_gray(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rgba_alpha_passes_through_dilate() {
        // Bright G channel single dot, varying alpha — alpha must
        // survive untouched.
        let mut data = Vec::with_capacity(16 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let g = if x == 1 && y == 1 { 200 } else { 50 };
                let a = (x * 10 + y * 5) as u8;
                data.extend_from_slice(&[10, g, 30, a]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Morphology::dilate(StructuringElement::Square3x3, 1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let off = ((y * 4 + x) * 4 + 3) as usize;
                let expected_alpha = (x * 10 + y * 5) as u8;
                assert_eq!(
                    out.planes[0].data[off], expected_alpha,
                    "alpha changed at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn edge_in_picks_inner_boundary() {
        // 3×3 bright square at the centre of a 5×5 dark field. With
        // EdgeIn we keep only the pixels that erode wipes out — the
        // outer ring of the bright square. The dead-centre pixel
        // (whose 8-neighbours are all bright) survives erosion and so
        // gets subtracted to zero.
        let input = gray_frame(5, 5, |x, y| {
            if (1..=3).contains(&x) && (1..=3).contains(&y) {
                200
            } else {
                0
            }
        });
        let out = MorphologyEdge::edge_in(StructuringElement::Square3x3)
            .apply(&input, params_gray(5, 5))
            .unwrap();
        // Centre survives erosion: src - erode = 200 - 200 = 0.
        assert_eq!(out.planes[0].data[2 * 5 + 2], 0);
        // Ring pixels are bright in src but were eroded to 0 → 200 - 0 = 200.
        assert_eq!(out.planes[0].data[5 + 1], 200);
        assert_eq!(out.planes[0].data[5 + 2], 200);
        assert_eq!(out.planes[0].data[5 + 3], 200);
        // Background pixels are 0 in src → 0 - 0 = 0.
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn edge_out_picks_outer_boundary() {
        // 1×1 bright pixel on a dark field. EdgeOut should highlight
        // exactly the 8-neighbour ring (those pixels go from 0 to 200
        // through dilation). The original bright pixel is preserved by
        // dilation and so subtracts to 0.
        let input = gray_frame(5, 5, |x, y| if x == 2 && y == 2 { 200 } else { 0 });
        let out = MorphologyEdge::edge_out(StructuringElement::Square3x3)
            .apply(&input, params_gray(5, 5))
            .unwrap();
        // Original centre: dilate = 200, src = 200 → 0.
        assert_eq!(out.planes[0].data[2 * 5 + 2], 0);
        // 8-neighbour pixels: dilate = 200, src = 0 → 200.
        assert_eq!(out.planes[0].data[5 + 1], 200);
        assert_eq!(out.planes[0].data[5 + 2], 200);
        assert_eq!(out.planes[0].data[5 + 3], 200);
        assert_eq!(out.planes[0].data[3 * 5 + 1], 200);
        // Outside the 3×3 neighbourhood: dilate = 0, src = 0 → 0.
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn edge_magnitude_lights_full_boundary() {
        // 1×1 bright pixel — gradient lights the centre + the 8-ring.
        let input = gray_frame(5, 5, |x, y| if x == 2 && y == 2 { 200 } else { 0 });
        let out = MorphologyEdge::edge_magnitude(StructuringElement::Square3x3)
            .apply(&input, params_gray(5, 5))
            .unwrap();
        // Centre: dilate = 200, erode = 0 → 200.
        assert_eq!(out.planes[0].data[2 * 5 + 2], 200);
        // 8-neighbour: dilate = 200, erode = 0 → 200.
        assert_eq!(out.planes[0].data[5 + 2], 200);
        // Outside 3×3 neighbourhood: dilate = 0, erode = 0 → 0.
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn flat_image_edge_is_zero() {
        // Flat field: dilate = src = erode → all three operators produce 0.
        let input = gray_frame(4, 4, |_, _| 77);
        let out = MorphologyEdge::edge_magnitude(StructuringElement::Cross3x3)
            .apply(&input, params_gray(4, 4))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: vec![0; 16],
            }],
        };
        let err = Morphology::dilate(StructuringElement::Square3x3, 1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P10Le,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Morphology"));
    }
}
