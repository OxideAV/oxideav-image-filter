//! Prewitt first-derivative edge operator.
//!
//! The Prewitt operator (Judith M. S. Prewitt, "Object Enhancement and
//! Extraction", in *Picture Processing and Psychopictorics*, 1970) is a
//! pair of 3×3 convolution kernels that approximate the horizontal and
//! vertical components of the image gradient. Unlike Sobel, the three
//! rows / columns are weighted equally (flat `±1`) — there is no extra
//! emphasis on the central neighbour, so it is a plain box-smoothed
//! finite difference:
//!
//! ```text
//!        [ -1  0  +1 ]              [ -1  -1  -1 ]
//! Gx  =  [ -1  0  +1 ]      Gy  =   [  0   0   0 ]
//!        [ -1  0  +1 ]              [ +1  +1  +1 ]
//! ```
//!
//! `Gx` is the difference between the right column and the left column
//! (summed over the three rows); `Gy` is the difference between the
//! bottom row and the top row (summed over the three columns). The edge
//! magnitude is the (optionally `L1`) combination
//!
//! ```text
//! G = sqrt(Gx² + Gy²)        (Magnitude::L2, default)
//! G = |Gx| + |Gy|            (Magnitude::L1, cheaper)
//! ```
//!
//! clamped to `[0, 255]`. The 3×3 window is centred on the output
//! sample; samples that would fall outside the image (the one-pixel
//! border) are clamped to the nearest in-bounds sample so the output
//! keeps the input dimensions.
//!
//! Prewitt sits between [`Roberts`](crate::roberts::Roberts) (2×2,
//! cheapest, noisiest) and [`Edge`](crate::Edge) (3×3 Sobel, central
//! row/column weighted `±2`): the wider 3×3 support makes it less
//! noise-sensitive than Roberts, while the flat weighting gives it
//! slightly less directional bias but marginally more noise than Sobel.
//! Contrast with [`Laplacian`](crate::laplacian::Laplacian), the
//! second-derivative isotropic detector.
//!
//! The multi-kernel [`EdgeDetect`](crate::edge_multi::EdgeDetect) filter
//! also offers a Prewitt kernel, but only the cheap `L1` combination;
//! this dedicated filter adds the true Euclidean (`L2`) magnitude as its
//! default and a first-class [`Prewitt`] type.
//!
//! The filter is stateless, single-frame, and `O(W·H)`.
//!
//! # Pixel formats
//!
//! Like [`Roberts`](crate::roberts::Roberts) /
//! [`Edge`](crate::Edge), any supported input is collapsed to a luma
//! plane first (Gray8 / YUV use the Y plane directly; RGB / RGBA use the
//! `(R + 2G + B) / 4` quick luma). The output is always `Gray8`.
//!
//! # Clean-room reference
//!
//! The two 3×3 kernels and the gradient-magnitude combination are the
//! textbook Prewitt operator as given in Prewitt 1970 and reproduced in
//! every image-processing text (e.g. Gonzalez & Woods, *Digital Image
//! Processing*; Sonka, Hlavac & Boyle, *Image Processing, Analysis, and
//! Machine Vision*). No external library source was consulted; the
//! expected outputs in the tests are hand-derived from the formulae
//! above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// How to combine the two gradient responses into a single edge
/// magnitude.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrewittMagnitude {
    /// `sqrt(Gx² + Gy²)` — the true Euclidean gradient magnitude.
    #[default]
    L2,
    /// `|Gx| + |Gy|` — the cheaper Manhattan approximation.
    L1,
}

/// Prewitt edge-detection filter.
///
/// See the [module-level docs](self) for the kernels and the
/// pixel-format coverage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prewitt {
    /// Magnitude combination — [`PrewittMagnitude::L2`] (default) or
    /// [`PrewittMagnitude::L1`].
    pub magnitude: PrewittMagnitude,
}

impl Prewitt {
    /// Build the default Prewitt operator (L2 magnitude).
    pub fn new() -> Self {
        Self {
            magnitude: PrewittMagnitude::L2,
        }
    }

    /// Build a Prewitt operator using the cheaper `L1` (`|Gx| + |Gy|`)
    /// magnitude combination.
    pub fn l1() -> Self {
        Self {
            magnitude: PrewittMagnitude::L1,
        }
    }

    /// Override the magnitude combination.
    pub fn with_magnitude(mut self, magnitude: PrewittMagnitude) -> Self {
        self.magnitude = magnitude;
        self
    }
}

impl ImageFilter for Prewitt {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Prewitt does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let out = prewitt(&luma, w, h, self.magnitude);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

fn prewitt(luma: &[u8], w: usize, h: usize, mag: PrewittMagnitude) -> Vec<u8> {
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
            // Gx = (right column) − (left column), summed over rows.
            let gx = (p20 + p21 + p22) - (p00 + p01 + p02);
            // Gy = (bottom row) − (top row), summed over columns.
            let gy = (p02 + p12 + p22) - (p00 + p10 + p20);
            let g = match mag {
                PrewittMagnitude::L2 => (((gx * gx + gy * gy) as f64).sqrt()).round() as i32,
                PrewittMagnitude::L1 => gx.abs() + gy.abs(),
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
        let out = Prewitt::new().apply(&input, p_gray(8, 8)).unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn vertical_step_hand_derived() {
        // 3-wide image, left half 0, right half 200:
        //   0 0 200
        //   0 0 200
        //   0 0 200
        // Centre pixel (1,1): 3×3 window over the whole image.
        //   left  col = (0,0,0) = 0
        //   right col = (200,200,200) = 600
        //   Gx = 600 − 0 = 600
        //   top row = (0,0,200) = 200, bottom row = (0,0,200) = 200
        //   Gy = 200 − 200 = 0
        //   L1 = 600 + 0 = 600 → clamp 255.
        let input = gray(3, 3, |x, _| if x == 2 { 200 } else { 0 });
        let out = Prewitt::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 255, "centre pixel saturates");
    }

    #[test]
    fn small_vertical_edge_l1_unsaturated() {
        // Right column = 30 (others 0). Centre pixel (1,1):
        //   right col = 30+30+30 = 90, left col = 0 → Gx = 90
        //   top/bottom rows symmetric → Gy = 0
        //   L1 = 90 + 0 = 90 (well under 255).
        let input = gray(3, 3, |x, _| if x == 2 { 30 } else { 0 });
        let out = Prewitt::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 90);
    }

    #[test]
    fn horizontal_edge_uses_gy() {
        // Bottom row = 20, rest 0. Centre pixel (1,1):
        //   Gx = 0 (left/right columns symmetric)
        //   bottom row = 20+20+20 = 60, top row = 0 → Gy = 60
        //   L1 = 0 + 60 = 60.
        let input = gray(3, 3, |_, y| if y == 2 { 20 } else { 0 });
        let out = Prewitt::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 60);
    }

    #[test]
    fn l1_vs_l2_differ_on_a_corner() {
        // Bottom-right 2×2 block = 10, rest 0.  Centre pixel (1,1)
        // has neighbours:
        //   p00 p10 p20 =  0  0  0
        //   p01 p11 p21 =  0  0 10
        //   p02 p12 p22 =  0 10 10
        //   Gx = (p20+p21+p22) − (p00+p01+p02) = (0+10+10) − (0+0+0) = 20
        //   Gy = (p02+p12+p22) − (p00+p10+p20) = (0+10+10) − (0+0+0) = 20
        //   L1 = 20 + 20 = 40
        //   L2 = sqrt(400 + 400) = sqrt(800) = 28.28 → 28
        let input = gray(3, 3, |x, y| if x >= 1 && y >= 1 { 10 } else { 0 });
        let l1 = Prewitt::l1().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(l1.planes[0].data[4], 40);
        let l2 = Prewitt::new().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(l2.planes[0].data[4], 28);
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = Prewitt::new()
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
        // Right column maxed, left zero: Gx = 255*3 = 765 → clamp 255.
        let input = gray(3, 3, |x, _| if x == 2 { 255 } else { 0 });
        let out = Prewitt::new().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data[4], 255);
    }

    #[test]
    fn pts_pass_through() {
        let input = gray(3, 3, |x, _| if x == 2 { 50 } else { 0 });
        let mut frame = input;
        frame.pts = Some(7);
        let out = Prewitt::new().apply(&frame, p_gray(3, 3)).unwrap();
        assert_eq!(out.pts, Some(7));
    }

    #[test]
    fn one_by_one_is_zero() {
        // A single pixel: every clamped neighbour equals it → zero
        // gradient.
        let input = gray(1, 1, |_, _| 123);
        let out = Prewitt::new().apply(&input, p_gray(1, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![0]);
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
        let err = Prewitt::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Nv12,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Prewitt"));
    }
}
