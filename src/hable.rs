//! Hable "filmic" tone-mapping (Uncharted-2 operator).
//!
//! John Hable, "Filmic Tonemapping Operators" (Naughty Dog, GDC/SIGGRAPH
//! 2010 talk; also documented in Hable's own follow-up blog
//! "Filmic Worlds — Filmic Tonemapping with Piecewise Power Curves",
//! 2017). The Uncharted-2 paper-listed form is the rational function:
//!
//! ```text
//! f(x) = ((x · (A·x + C·B) + D·E) / (x · (A·x + B) + D·F)) - E/F
//! ```
//!
//! with the classic constants `A = 0.15`, `B = 0.50`, `C = 0.10`,
//! `D = 0.20`, `E = 0.02`, `F = 0.30`, and a "linear white-point" `W`
//! (Uncharted-2 used `W = 11.2`). The display value is
//! `Ld = f(exposure · L) / f(W)`. By construction `f(0) = -E/F + E·D/(F·D)
//! = 0` (the `-E/F` term cancels the constant) and the function is
//! monotonic on `[0, W]`.
//!
//! Algorithm (clean-room from the published formula):
//!
//! 1. De-gamma sRGB → linear.
//! 2. Scale by `exposure_bias` (default `2.0` per the GDC slides).
//! 3. Apply `f(·)` per channel.
//! 4. Divide by `f(W)` so the white point lands exactly at `1.0`.
//! 5. Re-encode through the sRGB OETF.
//!
//! The operator is applied per channel (the paper does both per-luma
//! and per-channel variants; per-channel is the GDC default). Operates
//! on `Gray8` / `Rgb24` / `Rgba`; YUV returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Hable / Uncharted-2 "filmic" tone-mapping operator.
#[derive(Clone, Copy, Debug)]
pub struct Hable {
    /// Multiplier applied before the curve. Hable's GDC slides used
    /// `2.0` ("exposure bias"). Larger values brighten.
    pub exposure_bias: f32,
    /// Linear white-point used to renormalise the curve so `Ld(W) = 1`.
    /// Uncharted-2 used `W = 11.2`.
    pub white: f32,
}

impl Default for Hable {
    fn default() -> Self {
        Self {
            exposure_bias: 2.0,
            white: 11.2,
        }
    }
}

impl Hable {
    pub fn new(exposure_bias: f32) -> Self {
        Self {
            exposure_bias: if exposure_bias.is_finite() && exposure_bias > 0.0 {
                exposure_bias
            } else {
                2.0
            },
            white: 11.2,
        }
    }

    pub fn with_white(mut self, white: f32) -> Self {
        if white.is_finite() && white > 0.0 {
            self.white = white;
        }
        self
    }
}

#[inline]
fn hable_curve(x: f32) -> f32 {
    // Hable / Uncharted-2 constants.
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl ImageFilter for Hable {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Hable requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Hable requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let stride_out = w * bpp;
        let src = &input.planes[0];
        let bias = self.exposure_bias;
        let f_white = hable_curve(self.white).max(1.0e-6);

        // Pre-bake a 256-entry LUT for the chromatic channels (the
        // curve is applied per channel, so a LUT covers it exactly).
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            let lin = srgb_to_linear(v as u8) * bias;
            let mapped = hable_curve(lin) / f_white;
            *slot = linear_to_srgb(mapped);
        }

        let mut out = vec![0u8; stride_out * h];
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + w * bpp];
            let d = &mut out[y * stride_out..(y + 1) * stride_out];
            match bpp {
                1 => {
                    for (di, si) in d.iter_mut().zip(s.iter()) {
                        *di = lut[*si as usize];
                    }
                }
                3 => {
                    for x in 0..w {
                        let b = x * 3;
                        d[b] = lut[s[b] as usize];
                        d[b + 1] = lut[s[b + 1] as usize];
                        d[b + 2] = lut[s[b + 2] as usize];
                    }
                }
                4 => {
                    for x in 0..w {
                        let b = x * 4;
                        d[b] = lut[s[b] as usize];
                        d[b + 1] = lut[s[b + 1] as usize];
                        d[b + 2] = lut[s[b + 2] as usize];
                        d[b + 3] = s[b + 3];
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: stride_out,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn black_stays_black() {
        let input = rgb(4, 4, |_, _| [0, 0, 0]);
        let out = Hable::default().apply(&input, p_rgb(4, 4)).unwrap();
        // f(0) = (0·(0+0) + D·E) / (0·(0+0) + D·F) - E/F = E/F - E/F = 0.
        for &v in &out.planes[0].data {
            assert_eq!(v, 0, "black should map to black");
        }
    }

    #[test]
    fn monotone_increasing() {
        let input = rgb(8, 1, |x, _| {
            [(x * 32) as u8, (x * 32) as u8, (x * 32) as u8]
        });
        let out = Hable::default().apply(&input, p_rgb(8, 1)).unwrap();
        // Check monotone non-decreasing.
        let mut prev = 0u8;
        for px in 0..8 {
            let v = out.planes[0].data[px * 3];
            assert!(v >= prev, "non-monotone at x={px}: {prev} → {v}");
            prev = v;
        }
    }

    #[test]
    fn higher_exposure_brightens_midtones() {
        let input = rgb(2, 2, |_, _| [80, 80, 80]);
        let dim = Hable::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let bright = Hable::new(4.0).apply(&input, p_rgb(2, 2)).unwrap();
        assert!(
            bright.planes[0].data[0] > dim.planes[0].data[0],
            "bigger exposure_bias must brighten midtones (dim={}, bright={})",
            dim.planes[0].data[0],
            bright.planes[0].data[0]
        );
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 50, 200, 99]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Hable::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 99);
        }
    }

    #[test]
    fn rejects_yuv() {
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
        let err = Hable::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Hable"));
    }
}
