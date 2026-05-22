//! Roberts cross first-derivative edge operator.
//!
//! The Roberts cross (Lawrence Roberts, *Machine Perception of
//! Three-Dimensional Solids*, MIT PhD thesis, 1963) is the smallest
//! discrete gradient operator: a pair of 2×2 diagonal-difference
//! kernels that approximate the image gradient along the two diagonals.
//! For a pixel `Z(x, y)` the two responses are
//!
//! ```text
//! Gx =  Z(x, y)   - Z(x+1, y+1)        kernel  [ 1  0 ; 0 -1 ]
//! Gy =  Z(x+1, y) - Z(x, y+1)          kernel  [ 0  1 ; -1 0 ]
//! ```
//!
//! and the edge magnitude is the (optionally `L1`) combination
//!
//! ```text
//! G = sqrt(Gx² + Gy²)        (Magnitude::L2, default)
//! G = |Gx| + |Gy|            (Magnitude::L1, cheaper)
//! ```
//!
//! clamped to `[0, 255]`. Because the 2×2 window straddles the pixel
//! and its lower-right neighbours, the response is conventionally
//! assigned to the top-left sample `(x, y)`; the right and bottom
//! image borders (which would read past the edge) are clamped to the
//! nearest in-bounds sample so the output keeps the input dimensions.
//!
//! This is a first-derivative operator like [`Edge`](crate::Edge)
//! (Sobel) but with the tiniest possible support, which makes it fast
//! and noise-sensitive — it lights up fine, high-contrast diagonal
//! detail. Contrast with [`Laplacian`](crate::laplacian::Laplacian),
//! the second-derivative isotropic detector.
//!
//! The multi-kernel [`EdgeDetect`](crate::edge_multi::EdgeDetect)
//! filter also offers a Roberts kernel, but only the cheap `L1`
//! combination; this dedicated filter adds the true Euclidean (`L2`)
//! magnitude as its default and a first-class [`Roberts`] type.
//!
//! The filter is stateless, single-frame, and `O(W·H)`.
//!
//! # Pixel formats
//!
//! Like [`Edge`](crate::Edge) / [`Laplacian`](crate::laplacian::Laplacian),
//! any supported input is collapsed to a luma plane first (Gray8 / YUV
//! use the Y plane directly; RGB / RGBA use the `(R + 2G + B) / 4` quick
//! luma). The output is always `Gray8`.
//!
//! # Clean-room reference
//!
//! The two 2×2 kernels and the gradient-magnitude combination are the
//! textbook Roberts cross as given in Roberts 1963 and reproduced in
//! every image-processing text (e.g. Gonzalez & Woods, *Digital Image
//! Processing*). No external library source was consulted; the
//! expected outputs in the tests are hand-derived from the formulae
//! above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// How to combine the two diagonal gradient responses into a single
/// edge magnitude.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RobertsMagnitude {
    /// `sqrt(Gx² + Gy²)` — the true Euclidean gradient magnitude.
    #[default]
    L2,
    /// `|Gx| + |Gy|` — the cheaper Manhattan approximation.
    L1,
}

/// Roberts cross edge-detection filter.
///
/// See the [module-level docs](self) for the kernels and the
/// pixel-format coverage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Roberts {
    /// Magnitude combination — [`RobertsMagnitude::L2`] (default) or
    /// [`RobertsMagnitude::L1`].
    pub magnitude: RobertsMagnitude,
}

impl Roberts {
    /// Build the default Roberts cross (L2 magnitude).
    pub fn new() -> Self {
        Self {
            magnitude: RobertsMagnitude::L2,
        }
    }

    /// Build a Roberts cross using the cheaper `L1` (`|Gx| + |Gy|`)
    /// magnitude combination.
    pub fn l1() -> Self {
        Self {
            magnitude: RobertsMagnitude::L1,
        }
    }

    /// Override the magnitude combination.
    pub fn with_magnitude(mut self, magnitude: RobertsMagnitude) -> Self {
        self.magnitude = magnitude;
        self
    }
}

impl ImageFilter for Roberts {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Roberts does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let out = roberts_cross(&luma, w, h, self.magnitude);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

fn roberts_cross(luma: &[u8], w: usize, h: usize, mag: RobertsMagnitude) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    // Clamped sampler so the right / bottom borders stay in-bounds.
    let px = |x: usize, y: usize| -> i32 {
        let cx = x.min(w - 1);
        let cy = y.min(h - 1);
        luma[cy * w + cx] as i32
    };
    for y in 0..h {
        for x in 0..w {
            // 2×2 window anchored at (x, y):
            //   a b     a = (x,   y)     b = (x+1, y)
            //   c d     c = (x,   y+1)   d = (x+1, y+1)
            let a = px(x, y);
            let b = px(x + 1, y);
            let c = px(x, y + 1);
            let d = px(x + 1, y + 1);
            let gx = a - d; // [ 1 0 ; 0 -1 ]
            let gy = b - c; // [ 0 1 ; -1 0 ]
            let g = match mag {
                RobertsMagnitude::L2 => (((gx * gx + gy * gy) as f64).sqrt()).round() as i32,
                RobertsMagnitude::L1 => gx.abs() + gy.abs(),
            };
            out[y * w + x] = g.clamp(0, 255) as u8;
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
        let out = Roberts::new().apply(&input, p_gray(8, 8)).unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn diagonal_step_hand_derived_l2() {
        // 2×2 image:
        //   a b   =   0   0
        //   c d       0 200
        // Pixel (0,0): a=0, b=0, c=0, d=200
        //   Gx = a - d = -200, Gy = b - c = 0
        //   G = sqrt(200² + 0²) = 200.
        let input = gray(2, 2, |x, y| if x == 1 && y == 1 { 200 } else { 0 });
        let out = Roberts::new().apply(&input, p_gray(2, 2)).unwrap();
        assert_eq!(out.planes[0].data[0], 200, "top-left edge magnitude");
        // Pixel (1,0): clamped right border → a=b=0 (col 1), c=d=200.
        //   Gx = 0 - 200 = -200, Gy = 0 - 200 = -200
        //   G = sqrt(200²+200²) = 282.8 → clamp 255.
        assert_eq!(out.planes[0].data[1], 255);
    }

    #[test]
    fn anti_diagonal_uses_gy_l1() {
        // 2×2 image:
        //   a b   =   0  100
        //   c d       0    0
        // Pixel (0,0): a=0, b=100, c=0, d=0
        //   Gx = a - d = 0, Gy = b - c = 100
        //   L1 = |0| + |100| = 100.
        let input = gray(2, 2, |x, y| if x == 1 && y == 0 { 100 } else { 0 });
        let out = Roberts::l1().apply(&input, p_gray(2, 2)).unwrap();
        assert_eq!(out.planes[0].data[0], 100);
    }

    #[test]
    fn l1_vs_l2_differ_on_both_diagonals() {
        // a=0 b=30 c=40 d=0  →  Gx = 0-0 = 0?  No: Gx = a-d = 0.
        // Use a=50, d=0, b=0, c=30 so both gradients are nonzero.
        //   a b = 50  0
        //   c d = 30  0
        //   Gx = a - d = 50, Gy = b - c = -30
        //   L1 = 50 + 30 = 80
        //   L2 = sqrt(2500 + 900) = sqrt(3400) = 58.3 → 58
        let mut data = vec![50u8, 0, 30, 0];
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: std::mem::take(&mut data),
            }],
        };
        let l1 = Roberts::l1().apply(&frame, p_gray(2, 2)).unwrap();
        assert_eq!(l1.planes[0].data[0], 80);
        let l2 = Roberts::new().apply(&frame, p_gray(2, 2)).unwrap();
        assert_eq!(l2.planes[0].data[0], 58);
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = Roberts::new()
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
        // Maximal diagonal step both ways: a=255, d=0, b=255, c=0.
        //   Gx = 255, Gy = 255 → L2 = 360.6 → clamp 255.
        let data = vec![255u8, 255, 0, 0];
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 2, data }],
        };
        let out = Roberts::new().apply(&frame, p_gray(2, 2)).unwrap();
        assert_eq!(out.planes[0].data[0], 255);
    }

    #[test]
    fn pts_pass_through() {
        let input = VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![0, 0, 0, 100],
            }],
        };
        let out = Roberts::new().apply(&input, p_gray(2, 2)).unwrap();
        assert_eq!(out.pts, Some(7));
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
        let err = Roberts::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Nv12,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Roberts"));
    }
}
