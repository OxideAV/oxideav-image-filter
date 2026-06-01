//! Scharr first-derivative edge operator.
//!
//! The Scharr operator (Hannes Scharr, "Optimal Operators in Digital
//! Image Processing", PhD thesis, University of Heidelberg, 2000) is
//! a pair of 3×3 convolution kernels designed to give the most
//! rotationally-symmetric finite-difference gradient on a 3×3 stencil.
//! Where the Sobel kernels use the row weights `1 2 1` and the
//! Prewitt kernels use the flat `1 1 1`, Scharr uses `3 10 3`:
//!
//! ```text
//!        [ -3    0   +3 ]              [ -3  -10  -3 ]
//! Gx  =  [ -10   0  +10 ]      Gy  =   [  0    0   0 ]
//!        [ -3    0   +3 ]              [ +3  +10  +3 ]
//! ```
//!
//! The central row / column is weighted ten parts to the outer rows'
//! three; this places more emphasis on the centred difference and
//! suppresses the off-axis bias that Sobel still carries, giving a
//! gradient estimate whose orientation error is markedly smaller for
//! mid-frequency edges.
//!
//! `Gx` is the difference between the right column and the left column
//! under the Scharr weighting; `Gy` is the difference between the
//! bottom row and the top row likewise. The edge magnitude is the
//! (optionally `L1`) combination
//!
//! ```text
//! G = sqrt(Gx² + Gy²)        (Magnitude::L2, default)
//! G = |Gx| + |Gy|            (Magnitude::L1, cheaper)
//! ```
//!
//! Because the row weights sum to `16` (vs Sobel's `4` and Prewitt's
//! `3`), the raw magnitude is divided by `4` before clamping to
//! `[0, 255]` so the response lands in the same band as the other
//! 3×3 operators in this crate. The 3×3 window is centred on the
//! output sample; samples that would fall outside the image (the
//! one-pixel border) are clamped to the nearest in-bounds sample so
//! the output keeps the input dimensions.
//!
//! Scharr sits alongside [`Edge`](crate::Edge) (Sobel, weights
//! `±1 ±2 ±1`) and [`Prewitt`](crate::prewitt::Prewitt) (flat
//! `±1 ±1 ±1`) as a 3×3 first-derivative detector; it trades a
//! slightly higher per-pixel cost (one multiply by ten, two by
//! three) for the cleanest rotational symmetry of the three. The
//! 2×2 [`Roberts`](crate::roberts::Roberts) cross is the cheapest
//! and noisiest member of the family; the second-derivative
//! [`Laplacian`](crate::laplacian::Laplacian) is the isotropic
//! complement.
//!
//! The multi-kernel [`EdgeDetect`](crate::edge_multi::EdgeDetect)
//! filter also offers a Scharr kernel, but only the cheap `L1`
//! combination; this dedicated filter adds the true Euclidean (`L2`)
//! magnitude as its default and a first-class [`Scharr`] type.
//!
//! The filter is stateless, single-frame, and `O(W·H)`.
//!
//! # Pixel formats
//!
//! Like [`Roberts`](crate::roberts::Roberts) /
//! [`Prewitt`](crate::prewitt::Prewitt) / [`Edge`](crate::Edge), any
//! supported input is collapsed to a luma plane first (Gray8 / YUV
//! use the Y plane directly; RGB / RGBA use the `(R + 2G + B) / 4`
//! quick luma). The output is always `Gray8`.
//!
//! # Clean-room reference
//!
//! The two 3×3 kernels and the gradient-magnitude combination are the
//! textbook Scharr operator as given in Scharr's 2000 thesis and
//! reproduced in Jähne, *Digital Image Processing*, 6th ed. (chapter
//! "Edges") and Sonka, Hlavac & Boyle, *Image Processing, Analysis,
//! and Machine Vision*. The
//! expected outputs in the tests are hand-derived from the formulae
//! above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// How to combine the two gradient responses into a single edge
/// magnitude.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScharrMagnitude {
    /// `sqrt(Gx² + Gy²)` — the true Euclidean gradient magnitude.
    #[default]
    L2,
    /// `|Gx| + |Gy|` — the cheaper Manhattan approximation.
    L1,
}

/// Scharr edge-detection filter.
///
/// See the [module-level docs](self) for the kernels and the
/// pixel-format coverage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scharr {
    /// Magnitude combination — [`ScharrMagnitude::L2`] (default) or
    /// [`ScharrMagnitude::L1`].
    pub magnitude: ScharrMagnitude,
}

impl Scharr {
    /// Build the default Scharr operator (L2 magnitude).
    pub fn new() -> Self {
        Self {
            magnitude: ScharrMagnitude::L2,
        }
    }

    /// Build a Scharr operator using the cheaper `L1` (`|Gx| + |Gy|`)
    /// magnitude combination.
    pub fn l1() -> Self {
        Self {
            magnitude: ScharrMagnitude::L1,
        }
    }

    /// Override the magnitude combination.
    pub fn with_magnitude(mut self, magnitude: ScharrMagnitude) -> Self {
        self.magnitude = magnitude;
        self
    }
}

impl ImageFilter for Scharr {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Scharr does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let out = scharr(&luma, w, h, self.magnitude);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

fn scharr(luma: &[u8], w: usize, h: usize, mag: ScharrMagnitude) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    // Clamped sampler so the one-pixel border stays in-bounds.
    let px = |x: i64, y: i64| -> i32 {
        let cx = x.clamp(0, w as i64 - 1) as usize;
        let cy = y.clamp(0, h as i64 - 1) as usize;
        luma[cy * w + cx] as i32
    };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            // 3×3 neighbourhood centred on (x, y):
            //   p00 p10 p20      (top    row, y-1)
            //   p01 p11 p21      (centre row, y  )
            //   p02 p12 p22      (bottom row, y+1)
            let p00 = px(x - 1, y - 1);
            let p10 = px(x, y - 1);
            let p20 = px(x + 1, y - 1);
            let p01 = px(x - 1, y);
            let p21 = px(x + 1, y);
            let p02 = px(x - 1, y + 1);
            let p12 = px(x, y + 1);
            let p22 = px(x + 1, y + 1);
            // Gx weights:
            //   [ -3   0  +3 ]
            //   [-10   0 +10 ]
            //   [ -3   0  +3 ]
            let gx = -3 * p00 + 3 * p20 - 10 * p01 + 10 * p21 - 3 * p02 + 3 * p22;
            // Gy weights:
            //   [ -3  -10  -3 ]
            //   [  0    0   0 ]
            //   [ +3  +10  +3 ]
            let gy = -3 * p00 - 10 * p10 - 3 * p20 + 3 * p02 + 10 * p12 + 3 * p22;
            // Sum of |weights| on each axis is 16, so a flat 0..255 step
            // produces |G*| up to ~16·255 ≈ 4080. Divide by 4 (same
            // normalisation `EdgeDetect`'s Scharr branch uses) so the
            // output band matches the other 3×3 detectors here.
            let g = match mag {
                ScharrMagnitude::L2 => (((gx * gx + gy * gy) as f64).sqrt() / 4.0).round() as i32,
                ScharrMagnitude::L1 => (gx.abs() + gy.abs()) / 4,
            };
            out[y as usize * w + x as usize] = g.clamp(0, 255) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

    fn gray(w: u32, h: u32, pat: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pat(x, y));
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flat_input_yields_zero() {
        let input = gray(8, 8, |_, _| 100);
        let out = Scharr::new().apply(&input, p_gray(8, 8)).unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn vertical_step_hand_derived() {
        // 3-wide image, left two columns 0, right column 16:
        //   0 0 16
        //   0 0 16
        //   0 0 16
        // Centre pixel (1,1): 3×3 window spans the whole image.
        //   Gx = -3·p00 + 3·p20 -10·p01 + 10·p21 -3·p02 + 3·p22
        //      = -3·0 + 3·16 -10·0 + 10·16 -3·0 + 3·16
        //      = 48 + 160 + 48 = 256
        //   Gy = -3·p00 -10·p10 -3·p20 + 3·p02 +10·p12 + 3·p22
        //      = -3·0 -10·0 -3·16 + 3·0 +10·0 + 3·16
        //      = -48 + 48 = 0
        //   L1 = (|256| + |0|) / 4 = 64
        let input = gray(3, 3, |x, _| if x == 2 { 16 } else { 0 });
        let out = Scharr::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 64);
    }

    #[test]
    fn horizontal_edge_uses_gy() {
        // Bottom row = 8, rest 0. Centre pixel (1,1):
        //   Gx = -3·0 + 3·0 -10·0 + 10·0 -3·8 + 3·8 = 0
        //   Gy = -3·0 -10·0 -3·0 + 3·8 +10·8 + 3·8
        //      = 24 + 80 + 24 = 128
        //   L1 = (0 + 128) / 4 = 32
        let input = gray(3, 3, |_, y| if y == 2 { 8 } else { 0 });
        let out = Scharr::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 32);
    }

    #[test]
    fn l1_vs_l2_differ_on_a_corner() {
        // Bottom-right 2×2 block = 8, rest 0. Centre pixel (1,1):
        //   p00 p10 p20 = 0 0 0
        //   p01 p11 p21 = 0 0 8
        //   p02 p12 p22 = 0 8 8
        //   Gx = -3·0 + 3·0 -10·0 + 10·8 -3·0 + 3·8
        //      = 80 + 24 = 104
        //   Gy = -3·0 -10·0 -3·0 + 3·0 +10·8 + 3·8
        //      = 80 + 24 = 104
        //   L1 = (104 + 104) / 4 = 208 / 4 = 52
        //   L2 = sqrt(104² + 104²) / 4 = sqrt(21632) / 4
        //      ≈ 147.08 / 4 ≈ 36.77 → 37
        let input = gray(3, 3, |x, y| if x >= 1 && y >= 1 { 8 } else { 0 });
        let l1 = Scharr::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(l1.planes[0].data[4], 52);
        let l2 = Scharr::new().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(l2.planes[0].data[4], 37);
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = Scharr::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 4);
    }

    #[test]
    fn magnitude_response_clamps_to_255() {
        // Right column maxed (255), left zero. Centre pixel (1,1):
        //   Gx = 3·255 + 10·255 + 3·255 = 16·255 = 4080
        //   Gy = 0
        //   L1 = 4080 / 4 = 1020 → clamp 255.
        let input = gray(3, 3, |x, _| if x == 2 { 255 } else { 0 });
        let out = Scharr::new().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 255);
    }

    #[test]
    fn pts_pass_through() {
        let input = gray(3, 3, |x, _| if x == 2 { 50 } else { 0 });
        let mut frame = input;
        frame.pts = Some(11);
        let out = Scharr::new().apply(&frame, p_gray(3, 3)).unwrap();
        assert_eq!(out.pts, Some(11));
    }

    #[test]
    fn one_by_one_is_zero() {
        // A single pixel: every clamped neighbour equals it → zero
        // gradient.
        let input = gray(1, 1, |_, _| 200);
        let out = Scharr::new().apply(&input, p_gray(1, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![0]);
    }

    #[test]
    fn with_magnitude_setter_swaps_norm() {
        // Same fixture as `horizontal_edge_uses_gy` to confirm the
        // builder setter swaps the magnitude rather than building a
        // fresh default.
        let input = gray(3, 3, |_, y| if y == 2 { 8 } else { 0 });
        let f = Scharr::new().with_magnitude(ScharrMagnitude::L1);
        let out = f.apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 32);
    }

    #[test]
    fn rejects_unsupported_format() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let err = Scharr::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Nv12,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Scharr"));
    }
}
