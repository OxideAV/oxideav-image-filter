//! User-supplied convolution kernel (ImageMagick `-convolve "..."`).
//!
//! Applies an arbitrary square `N×N` kernel (N must be odd: 1, 3, 5, 7,
//! ...) to a video frame. For each output pixel, sums the elementwise
//! product of the kernel with the surrounding `N×N` window of source
//! samples; out-of-bounds samples clamp to the nearest edge. The
//! optional `bias` is added after summation; the optional `divisor`
//! divides the running sum (defaults to the kernel-element sum if
//! non-zero, else 1.0 — IM's "normalise" convention).
//!
//! ```text
//! For each output (px, py) and each colour channel c:
//!     acc = bias
//!     for ky in -r..=r:
//!         for kx in -r..=r:
//!             acc += in[clamp(py+ky), clamp(px+kx)] * kernel[ky+r, kx+r]
//!     out[c] = clamp(round(acc / divisor), 0, 255)
//! ```
//!
//! Works on `Gray8`, `Rgb24`, and `Rgba`. On `Rgba` the alpha channel
//! is left untouched (clean-room: matches IM's `-channel rgb -convolve`
//! default). Planar YUV is rejected — use the per-plane filters in this
//! crate (Blur / Sharpen) for YUV streams.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Square convolution kernel.
///
/// `size` must be odd. `kernel` must be exactly `size * size` long, in
/// row-major order (top-left first). `bias` is added to each output
/// sample after the weighted sum. `divisor` scales the result; `None`
/// means "auto" — sum of kernel weights, falling back to 1.0 if zero.
#[derive(Clone, Debug)]
pub struct Convolve {
    pub size: u32,
    pub kernel: Vec<f32>,
    pub bias: f32,
    pub divisor: Option<f32>,
}

impl Convolve {
    /// Build a convolution filter from a flat row-major kernel. Returns
    /// `Error::invalid` if `size` is even / zero, or if `kernel.len() !=
    /// size * size`, or if any element is not finite.
    pub fn new(size: u32, kernel: Vec<f32>) -> Result<Self, Error> {
        if size == 0 || size % 2 == 0 {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: Convolve size must be odd and >= 1, got {size}"
            )));
        }
        let expected = (size as usize) * (size as usize);
        if kernel.len() != expected {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: Convolve kernel length {} doesn't match size {} (need {})",
                kernel.len(),
                size,
                expected
            )));
        }
        if !kernel.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid(
                "oxideav-image-filter: Convolve kernel contains non-finite values",
            ));
        }
        Ok(Self {
            size,
            kernel,
            bias: 0.0,
            divisor: None,
        })
    }

    /// Set an additive bias applied after the weighted sum.
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = if bias.is_finite() { bias } else { 0.0 };
        self
    }

    /// Override the divisor. `None` means "auto" (sum of weights).
    pub fn with_divisor(mut self, divisor: Option<f32>) -> Self {
        self.divisor = divisor.filter(|v| v.is_finite() && *v != 0.0);
        self
    }

    fn effective_divisor(&self) -> f32 {
        if let Some(d) = self.divisor {
            return d;
        }
        let s: f32 = self.kernel.iter().sum();
        if s.abs() < 1e-9 {
            1.0
        } else {
            s
        }
    }
}

impl ImageFilter for Convolve {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, conv_channels) = match params.format {
            PixelFormat::Gray8 => (1usize, 1usize),
            PixelFormat::Rgb24 => (3usize, 3usize),
            // Convolve R/G/B; alpha pass-through.
            PixelFormat::Rgba => (4usize, 3usize),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Convolve requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let r = (self.size as i32 - 1) / 2;
        let n = self.size as usize;
        let inv_div = 1.0 / self.effective_divisor();

        let mut out = vec![0u8; row_bytes * h];

        for y in 0..h {
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                for ch in 0..conv_channels {
                    let mut acc = self.bias;
                    for ky in 0..n {
                        let sy = (y as i32 + ky as i32 - r).clamp(0, h as i32 - 1) as usize;
                        let row_off = sy * src.stride;
                        for kx in 0..n {
                            let sx = (x as i32 + kx as i32 - r).clamp(0, w as i32 - 1) as usize;
                            let weight = self.kernel[ky * n + kx];
                            acc += weight * src.data[row_off + sx * bpp + ch] as f32;
                        }
                    }
                    let v = (acc * inv_div).round().clamp(0.0, 255.0) as u8;
                    dst_row[x * bpp + ch] = v;
                }
                // Alpha pass-through (RGBA only).
                if conv_channels < bpp {
                    let src_off = y * src.stride + x * bpp + conv_channels;
                    for ch in conv_channels..bpp {
                        dst_row[x * bpp + ch] = src.data[src_off + ch - conv_channels];
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
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
    fn rejects_even_size() {
        assert!(Convolve::new(2, vec![0.25; 4]).is_err());
        assert!(Convolve::new(0, vec![]).is_err());
    }

    #[test]
    fn rejects_wrong_kernel_length() {
        assert!(Convolve::new(3, vec![0.0; 8]).is_err());
        assert!(Convolve::new(5, vec![0.0; 24]).is_err());
    }

    #[test]
    fn rejects_non_finite_kernel() {
        assert!(Convolve::new(3, vec![f32::NAN; 9]).is_err());
        assert!(Convolve::new(3, vec![f32::INFINITY; 9]).is_err());
    }

    #[test]
    fn identity_kernel_is_passthrough() {
        // 3×3 identity kernel: 1 at the centre, 0 elsewhere.
        let mut k = vec![0.0f32; 9];
        k[4] = 1.0;
        let f = Convolve::new(3, k).unwrap();
        let input = gray(8, 8, |x, y| (x as u8 + y as u8 * 8) * 3);
        let out = f.apply(&input, params_gray(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn box_blur_3x3_averages_correctly() {
        // 3×3 box blur (1/9 weights).
        let f = Convolve::new(3, vec![1.0; 9]).unwrap();
        // Flat input ⇒ average is the input value, output unchanged.
        let input = gray(8, 8, |_, _| 100);
        let out = f.apply(&input, params_gray(8, 8)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn box_blur_smooths_a_step() {
        // Step input: left half 0, right half 200. After 3×3 box blur,
        // the seam column should land near (200/3) and (2*200/3) — values
        // that are clearly intermediate, not 0 or 200.
        let f = Convolve::new(3, vec![1.0; 9]).unwrap();
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = f.apply(&input, params_gray(8, 8)).unwrap();
        // Column 3 (last "0" column): output sums 6 zeros and 3 200s
        // ⇒ 200*3/9 = ~66.7 ⇒ rounded to 67.
        assert_eq!(out.planes[0].data[3], 67);
        // Column 4 (first "200" column): 3 zeros + 6 200s = 200*6/9 = 133.
        assert_eq!(out.planes[0].data[4], 133);
    }

    #[test]
    fn bias_is_added_after_sum() {
        // 3×3 zero kernel with bias 100 ⇒ every output is 100.
        let f = Convolve::new(3, vec![0.0; 9]).unwrap().with_bias(100.0);
        let input = gray(4, 4, |_, _| 50);
        let out = f.apply(&input, params_gray(4, 4)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn explicit_divisor_overrides_auto() {
        // Box-blur weights with explicit divisor 1.0 ⇒ output is sum,
        // clamped to 255 (a single 100 input fills 9 cells, sum = 900).
        let f = Convolve::new(3, vec![1.0; 9])
            .unwrap()
            .with_divisor(Some(1.0));
        let input = gray(4, 4, |_, _| 100);
        let out = f.apply(&input, params_gray(4, 4)).unwrap();
        // Centre pixel: full 9-cell window inside ⇒ sum = 900 ⇒ clamp 255.
        assert_eq!(out.planes[0].data[5], 255);
    }

    #[test]
    fn rgba_alpha_pass_through() {
        let f = Convolve::new(3, vec![1.0; 9]).unwrap();
        // 4×4 RGBA, uniform colour & alpha.
        let data: Vec<u8> = (0..16).flat_map(|_| [40u8, 80, 120, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Box blur of flat input ⇒ unchanged colour. Alpha untouched.
        for px in 0..16 {
            assert_eq!(out.planes[0].data[px * 4], 40);
            assert_eq!(out.planes[0].data[px * 4 + 1], 80);
            assert_eq!(out.planes[0].data[px * 4 + 2], 120);
            assert_eq!(out.planes[0].data[px * 4 + 3], 222, "alpha at {px}");
        }
    }

    #[test]
    fn rejects_yuv420p() {
        let f = Convolve::new(3, vec![1.0; 9]).unwrap();
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let err = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Convolve"));
    }

    #[test]
    fn larger_5x5_kernel_works() {
        // 5×5 box blur (25 ones). Flat input ⇒ unchanged.
        let f = Convolve::new(5, vec![1.0; 25]).unwrap();
        let input = gray(8, 8, |_, _| 77);
        let out = f.apply(&input, params_gray(8, 8)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 77);
        }
    }
}
