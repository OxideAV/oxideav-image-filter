//! Colour-temperature shift: blue ⇄ red balance, parameterised by a
//! signed warmth scalar in `[-1, 1]`.
//!
//! Photographic colour temperature is measured in Kelvin (lower = warm /
//! tungsten / candlelight, higher = cool / shade / overcast). A full
//! white-balance computation involves the Planckian locus + a chromatic
//! adaptation transform; this filter ships a much simpler 1-D
//! warmth-axis approximation suitable for a creative-grade slider:
//!
//! - `warmth = 0.0` is identity.
//! - `warmth > 0.0` boosts the red channel and attenuates the blue one
//!   (the picture looks warmer / more tungsten-tinted).
//! - `warmth < 0.0` does the reverse (cooler / more daylight-tinted).
//!
//! The shift is implemented as per-channel multipliers built into two
//! 256-entry LUTs (R-channel and B-channel), so the per-pixel cost is
//! `O(1)` regardless of `warmth`. The maximum gain is ±50 % at
//! `|warmth| == 1.0`, matching a roughly ±1500 K creative shift on a
//! daylight-balanced source. Green channel and alpha pass through
//! untouched.
//!
//! Operates on `Rgb24`, `Rgba`. `Gray8` is rejected (greyscale has no
//! red / blue split). YUV inputs return `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Colour-temperature shift filter.
#[derive(Clone, Copy, Debug)]
pub struct Temperature {
    /// Warmth slider in `[-1, 1]`. Non-finite values coerce to `0.0`.
    pub warmth: f32,
}

impl Default for Temperature {
    fn default() -> Self {
        Self { warmth: 0.0 }
    }
}

impl Temperature {
    pub fn new(warmth: f32) -> Self {
        Self {
            warmth: if warmth.is_finite() {
                warmth.clamp(-1.0, 1.0)
            } else {
                0.0
            },
        }
    }

    fn build_luts(&self) -> ([u8; 256], [u8; 256]) {
        // Up to ±50% per-channel gain at the extremes.
        let r_gain = 1.0 + 0.5 * self.warmth;
        let b_gain = 1.0 - 0.5 * self.warmth;
        let mut r_lut = [0u8; 256];
        let mut b_lut = [0u8; 256];
        for v in 0..256 {
            let x = v as f32 / 255.0;
            r_lut[v] = ((x * r_gain).clamp(0.0, 1.0) * 255.0).round() as u8;
            b_lut[v] = ((x * b_gain).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        (r_lut, b_lut)
    }
}

impl ImageFilter for Temperature {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3, false),
            PixelFormat::Rgba => (4, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Temperature requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Temperature requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let stride = w * bpp;
        let (r_lut, b_lut) = self.build_luts();
        let src = &input.planes[0];
        let mut out = vec![0u8; stride * h];
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + stride];
            let d = &mut out[y * stride..(y + 1) * stride];
            for x in 0..w {
                let b = x * bpp;
                d[b] = r_lut[s[b] as usize];
                d[b + 1] = s[b + 1];
                d[b + 2] = b_lut[s[b + 2] as usize];
                if has_alpha {
                    d[b + 3] = s[b + 3];
                }
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data: out }],
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
    fn zero_warmth_is_identity() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = Temperature::new(0.0).apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn positive_warmth_boosts_r_attenuates_b() {
        let input = rgb(2, 2, |_, _| [100, 100, 100]);
        let out = Temperature::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let g = out.planes[0].data[1];
        let b = out.planes[0].data[2];
        assert!(r > g, "warmth must boost R: r={r}, g={g}");
        assert!(b < g, "warmth must reduce B: b={b}, g={g}");
        assert_eq!(g, 100, "G untouched");
    }

    #[test]
    fn negative_warmth_inverse_sign() {
        let input = rgb(2, 2, |_, _| [100, 100, 100]);
        let out = Temperature::new(-1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let b = out.planes[0].data[2];
        assert!(r < 100, "negative warmth reduces R: r={r}");
        assert!(b > 100, "negative warmth boosts B: b={b}");
    }

    #[test]
    fn warmth_clamped_into_minus_one_to_one() {
        let f = Temperature::new(5.0);
        assert!((f.warmth - 1.0).abs() < 1e-6);
        let f = Temperature::new(-9.0);
        assert!((f.warmth + 1.0).abs() < 1e-6);
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 100, 100, 234]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Temperature::new(0.3)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 234);
        }
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
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
        let err = Temperature::new(0.5)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Temperature"));
    }
}
