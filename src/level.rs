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
//! `white == black`. Mirrors ImageMagick's `-level LOW,MID,HIGH`.

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
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
    black: u8,
    white: u8,
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
            black: b,
            white: w,
            gamma: gamma.max(1e-6),
        }
    }
}

impl ImageFilter for Level {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Level does not yet handle {:?}",
                params.format
            )));
        }
        let lut = build_level_lut(self.black, self.white, self.gamma);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
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
    use oxideav_core::{PixelFormat, TimeBase, VideoPlane};

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
