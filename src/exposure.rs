//! EV-stop exposure adjustment in linear-light space.
//!
//! Photographic "EV stops" double the captured light per +1 EV. The
//! perceptually-correct way to mimic that on an sRGB image is:
//!
//! 1. De-gamma the input from sRGB to linear-light (`pow(x, 2.4)` with
//!    the standard `0.04045` linear segment).
//! 2. Multiply by `2^EV` (`+1.0` EV = ×2, `-1.0` EV = ×0.5).
//! 3. Re-encode through the sRGB OETF.
//!
//! Doing the multiplication in linear-light space matches the way the
//! sensor would respond to a longer / shorter exposure; multiplying in
//! gamma-space (the trivial alternative) crushes highlights faster and
//! lifts shadows slower than physics says. Same end-to-end formula
//! ImageMagick reaches for via `-evaluate Multiply` applied between
//! `-colorspace RGB` and `-colorspace sRGB` round-trips, except we fold
//! the round-trip into a single 256-entry LUT so the cost is `O(W·H)`.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`. YUV inputs return `Unsupported`
//! because the linear-light round-trip needs RGB-space data — the
//! ImageMagick analogue (`-colorspace RGB -evaluate Multiply k -colorspace
//! sRGB`) likewise refuses YUV. Alpha is preserved on RGBA.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// EV-stop linear-light exposure adjustment.
#[derive(Clone, Copy, Debug)]
pub struct Exposure {
    /// Stops of exposure to add. Positive brightens, negative darkens.
    /// `0.0` is identity (within sRGB round-trip rounding). Non-finite
    /// values coerce to `0.0`.
    pub ev: f32,
}

impl Default for Exposure {
    fn default() -> Self {
        Self { ev: 0.0 }
    }
}

impl Exposure {
    pub fn new(ev: f32) -> Self {
        Self {
            ev: if ev.is_finite() { ev } else { 0.0 },
        }
    }

    fn build_lut(&self) -> [u8; 256] {
        let mult = (2.0f32).powf(self.ev);
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32 / 255.0;
            // sRGB EOTF.
            let lin = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
            let scaled = (lin * mult).clamp(0.0, 1.0);
            // sRGB OETF.
            let y = if scaled <= 0.003_130_8 {
                scaled * 12.92
            } else {
                1.055 * scaled.powf(1.0 / 2.4) - 0.055
            };
            *slot = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        lut
    }
}

impl ImageFilter for Exposure {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Exposure requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Exposure requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let stride = w * bpp;
        let lut = self.build_lut();
        let src = &input.planes[0];
        let mut out = vec![0u8; stride * h];
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + stride];
            let d = &mut out[y * stride..(y + 1) * stride];
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
    fn zero_ev_round_trips_within_rounding() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = Exposure::new(0.0).apply(&input, p_rgb(4, 4)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1, "got {a}, expected {b}");
        }
    }

    #[test]
    fn positive_ev_brightens() {
        let input = rgb(2, 2, |_, _| [60, 60, 60]);
        let bright = Exposure::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let dark = Exposure::new(-1.0).apply(&input, p_rgb(2, 2)).unwrap();
        assert!(
            bright.planes[0].data[0] > input.planes[0].data[0],
            "+1 EV must brighten"
        );
        assert!(
            dark.planes[0].data[0] < input.planes[0].data[0],
            "-1 EV must darken"
        );
    }

    #[test]
    fn extreme_negative_drives_to_black() {
        let input = rgb(2, 2, |_, _| [200, 200, 200]);
        let out = Exposure::new(-10.0).apply(&input, p_rgb(2, 2)).unwrap();
        for v in out.planes[0].data.iter() {
            assert!(*v < 20, "-10 EV must crush near-black");
        }
    }

    #[test]
    fn extreme_positive_saturates_to_white() {
        let input = rgb(2, 2, |_, _| [50, 50, 50]);
        let out = Exposure::new(10.0).apply(&input, p_rgb(2, 2)).unwrap();
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 255, "+10 EV must clamp to 255");
        }
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 100, 100, 77]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Exposure::new(0.5)
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
            assert_eq!(chunk[3], 77);
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
        let err = Exposure::new(1.0)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Exposure"));
    }
}
