//! Three-way colour balance: per-channel `lift` (shadows), `gamma`
//! (midtones), and `gain` (highlights) — the classic colourist
//! `ASC CDL`-style adjustment.
//!
//! ASC CDL formula (per channel `c ∈ {R, G, B}`):
//!
//! ```text
//! out = ((in × gain) + lift) ^ (1 / gamma)
//! ```
//!
//! All inputs are normalised to `[0, 1]` before evaluation; the output
//! is clamped to `[0, 1]` and re-quantised. `gain == 1.0`,
//! `lift == 0.0`, `gamma == 1.0` is the identity transform.
//!
//! ImageMagick analogue: there is no single one-liner for ASC CDL —
//! `-evaluate Multiply` is the gain row, `-evaluate Add` is the lift
//! row, `-gamma` is the gamma row, and applying all three per channel
//! requires a three-step `-channel` pipeline. This filter folds the
//! three rows together into one pass with a 256-entry per-channel LUT
//! so the cost is `O(W·H)` regardless of how many adjustments are
//! enabled.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`. YUV inputs return
//! `Unsupported` (the per-channel knobs only make sense on
//! direct-colour data).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Three-way ASC CDL-style colour balance.
#[derive(Clone, Copy, Debug)]
pub struct ColorBalance {
    /// Per-channel additive offset applied before gamma. Sensible
    /// range is roughly `[-0.5, 0.5]`.
    pub lift: [f32; 3],
    /// Per-channel gamma exponent (`out = pow(x, 1/gamma)`); positive,
    /// `1.0` is identity. Non-finite or non-positive values coerce to
    /// `1.0`.
    pub gamma: [f32; 3],
    /// Per-channel multiplicative gain applied to the input.
    pub gain: [f32; 3],
}

impl Default for ColorBalance {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        }
    }
}

impl ColorBalance {
    pub fn new(lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) -> Self {
        Self { lift, gamma, gain }
    }

    fn build_luts(&self) -> [[u8; 256]; 3] {
        let mut luts = [[0u8; 256]; 3];
        for (c, lut) in luts.iter_mut().enumerate() {
            let lift = self.lift[c];
            let gamma = if self.gamma[c].is_finite() && self.gamma[c] > 0.0 {
                self.gamma[c]
            } else {
                1.0
            };
            let gain = self.gain[c];
            let inv_g = 1.0 / gamma;
            for (v, slot) in lut.iter_mut().enumerate() {
                let x = v as f32 / 255.0;
                let mut y = x * gain + lift;
                if y < 0.0 {
                    y = 0.0;
                }
                let y = if y > 0.0 { y.powf(inv_g) } else { 0.0 };
                *slot = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        luts
    }
}

impl ImageFilter for ColorBalance {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ColorBalance requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: ColorBalance requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let stride = w * bpp;
        let luts = self.build_luts();
        let mut out = vec![0u8; stride * h];
        let src = &input.planes[0];
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + stride];
            let d = &mut out[y * stride..(y + 1) * stride];
            match bpp {
                1 => {
                    // Gray8: apply luma channel only (use lift[0]/gamma[0]/gain[0]).
                    for (di, si) in d.iter_mut().zip(s.iter()) {
                        *di = luts[0][*si as usize];
                    }
                }
                3 => {
                    for x in 0..w {
                        let b = x * 3;
                        d[b] = luts[0][s[b] as usize];
                        d[b + 1] = luts[1][s[b + 1] as usize];
                        d[b + 2] = luts[2][s[b + 2] as usize];
                    }
                }
                4 => {
                    for x in 0..w {
                        let b = x * 4;
                        d[b] = luts[0][s[b] as usize];
                        d[b + 1] = luts[1][s[b + 1] as usize];
                        d[b + 2] = luts[2][s[b + 2] as usize];
                        d[b + 3] = s[b + 3]; // alpha untouched
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
    fn identity_is_passthrough() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = ColorBalance::default().apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn red_gain_brightens_only_red() {
        let input = rgb(2, 2, |_, _| [100, 100, 100]);
        let f = ColorBalance::new([0.0; 3], [1.0; 3], [1.5, 1.0, 1.0]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let g = out.planes[0].data[1];
        let b = out.planes[0].data[2];
        assert!(r > g, "R should be boosted: r={r}, g={g}");
        assert_eq!(g, 100);
        assert_eq!(b, 100);
    }

    #[test]
    fn blue_lift_brightens_shadows() {
        let input = rgb(2, 2, |_, _| [0, 0, 0]);
        let f = ColorBalance::new([0.0, 0.0, 0.2], [1.0; 3], [1.0; 3]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        let b = out.planes[0].data[2];
        assert!(b > 30, "blue lift should brighten black pixel: got {b}");
    }

    #[test]
    fn alpha_is_passthrough_on_rgba() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[10u8, 20, 30, 222]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let f = ColorBalance::new([0.1; 3], [0.8; 3], [1.2; 3]);
        let out = f
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
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![0; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
            ],
        };
        let err = ColorBalance::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("ColorBalance"));
    }
}
