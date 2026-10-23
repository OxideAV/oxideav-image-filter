//! Levels adjustment: remap the input range `[black, white]` onto the
//! full output range `[0, 255]`, with an optional mid-tone gamma.
//!
//! Clean-room formula from the textbook definition:
//!
//! ```text
//! v' = clamp(255 * ((v - black) / (white - black))^(1/gamma))
//! ```
//!
//! with endpoints clamped at 0 and 255 to avoid divide-by-zero when
//! `white == black`. Mirrors the documented `-level LOW,MID,HIGH` CLI.

use crate::{PixelFilter, VideoStreamParams, tonal_lut::apply_tone_lut};
use oxideav_core::{Error, VideoFrame};

/// Linear/gamma levels adjustment.
///
/// - `black` / `white`: input black/white cut-offs (0..=255). Samples
///   at or below `black` map to 0; samples at or above `white` map to
///   255. `black` and `white` must be strictly ordered; the
///   constructor swaps them if `black >= white` to avoid division by
///   zero.
/// - `gamma`: mid-tone gamma (default 1.0). `> 1.0` brightens midtones
///   after the linear remap; `< 1.0` darkens them.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Input black point, normalised (`0.0` = code 0, `1.0` = full scale).
    black: f32,
    /// Input white point, normalised.
    white: f32,
    gamma: f32,
}

impl Default for Level {
    fn default() -> Self {
        Self::new(0, 255, 1.0)
    }
}

impl Level {
    /// Build a levels adjustment. If `black >= white` the two values
    /// are swapped and `white` is nudged up by 1 sample to keep the
    /// denominator non-zero; `gamma` is clamped to a small positive
    /// epsilon to avoid dividing by zero.
    pub fn new(black: u8, white: u8, gamma: f32) -> Self {
        let (b, w) = if black < white {
            (black, white)
        } else if white < 255 {
            (white, (white as u16 + 1) as u8)
        } else {
            (254, 255)
        };
        Self {
            black: b as f32 / 255.0,
            white: w as f32 / 255.0,
            gamma: gamma.max(1e-6),
        }
    }

    /// Build a levels adjustment from **normalised** endpoints: `0.0` is
    /// code 0 and `1.0` full scale on every depth, and the endpoints may
    /// lie outside `[0, 1]` (`-level -20%,120%` style stretches) or be
    /// reversed (which inverts the tone). Endpoints closer than `1e-6`
    /// are nudged apart. On a float picture the output keeps its
    /// out-of-range headroom; on an integer picture it saturates at the
    /// word's full scale.
    pub fn normalized(black: f32, white: f32, gamma: f32) -> Self {
        let black = if black.is_finite() { black } else { 0.0 };
        let mut white = if white.is_finite() { white } else { 1.0 };
        if (white - black).abs() < 1e-6 {
            white = black + 1e-6;
        }
        Self {
            black,
            white,
            gamma: if gamma.is_finite() {
                gamma.max(1e-6)
            } else {
                1.0
            },
        }
    }

    /// The normalised input black point.
    pub fn black_point(&self) -> f32 {
        self.black
    }

    /// The normalised input white point.
    pub fn white_point(&self) -> f32 {
        self.white
    }

    /// The mid-tone gamma.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// The levels curve on one normalised tone.
    #[inline]
    pub fn tone(&self, v: f32) -> f32 {
        level_tone(v, self.black, self.white, 1.0 / self.gamma)
    }
}

/// `((v − black) / (white − black))^(1/γ)`: linear remap unclamped
/// (headroom and negative excursions survive), the power applied to the
/// positive part only (a negative base has no real power) — so with
/// `γ = 1` the curve is a pure affine stretch.
#[inline]
pub(crate) fn level_tone(v: f32, black: f32, white: f32, inv_gamma: f32) -> f32 {
    let t = (v - black) / (white - black);
    if inv_gamma == 1.0 {
        t
    } else if t <= 0.0 {
        0.0
    } else {
        t.powf(inv_gamma)
    }
}

impl PixelFilter for Level {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::TONAL
    }

    /// Each output pixel is a function of the input pixel at the same
    /// position only, so `Pal8` is taken through the palette route.
    fn is_pointwise(&self) -> bool {
        true
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        if crate::deep::is_float(params.format) {
            let me = *self;
            return crate::deep::map_tone(input, params, |v| me.tone(v));
        }
        let lut = match self.u8_endpoints() {
            // Endpoints on the 8-bit grid: the exact integer table.
            Some((b, w)) => build_level_lut(b, w, self.gamma),
            None => {
                let mut lut = [0u8; 256];
                for (i, entry) in lut.iter_mut().enumerate() {
                    let mapped = self.tone(i as f32 / 255.0) * 255.0;
                    *entry = if mapped.is_nan() {
                        0
                    } else {
                        mapped.round().clamp(0.0, 255.0) as u8
                    };
                }
                lut
            }
        };
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

impl Level {
    /// The endpoints as 8-bit codes when both sit on the 8-bit grid in
    /// order (`black < white`).
    fn u8_endpoints(&self) -> Option<(u8, u8)> {
        let b = self.black * 255.0;
        let w = self.white * 255.0;
        let on_grid = |x: f32| (0.0..=255.0).contains(&x) && (x - x.round()).abs() < 1e-4;
        if on_grid(b) && on_grid(w) && b < w {
            Some((b.round() as u8, w.round() as u8))
        } else {
            None
        }
    }
}

pub(crate) fn build_level_lut(black: u8, white: u8, gamma: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let b = black as f32;
    let w = white as f32;
    let range = (w - b).max(1.0);
    let inv = 1.0f32 / gamma;
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = (i as f32 - b) / range;
        if v <= 0.0 {
            *entry = 0;
        } else if v >= 1.0 {
            *entry = 255;
        } else {
            let mapped = v.powf(inv) * 255.0;
            *entry = mapped.round().clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
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

    #[test]
    fn identity_levels() {
        let input = gray(4, 4, |x, y| ((x + y * 4) * 15) as u8);
        let out = Level::new(0, 255, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn level_clips_below_black() {
        let input = gray(1, 1, |_, _| 20);
        let out = Level::new(50, 200, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn level_clips_above_white() {
        let input = gray(1, 1, |_, _| 220);
        let out = Level::new(50, 200, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 255);
    }

    #[test]
    fn level_midpoint_remap() {
        // black=50, white=200 → midpoint 125 should map to ≈ 127 with gamma=1.
        let input = gray(1, 1, |_, _| 125);
        let out = Level::new(50, 200, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        let v = out.planes[0].data[0];
        assert!(v > 120 && v < 135, "v = {v}");
    }

    #[test]
    fn level_gamma_biases_midtones() {
        // With gamma 2.0, midpoint of the remapped range should rise.
        let input = gray(1, 1, |_, _| 125);
        let plain = Level::new(50, 200, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap()
            .planes[0]
            .data[0];
        let gamma = Level::new(50, 200, 2.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap()
            .planes[0]
            .data[0];
        assert!(gamma > plain, "gamma={gamma} plain={plain}");
    }

    #[test]
    fn level_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|_| [120u8, 120, 120, 77]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Level::new(30, 220, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 77);
        }
    }
}
