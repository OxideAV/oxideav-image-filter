//! Sample clamping (`-clamp`).
//!
//! Maps every sample below `low` to `low` and every sample above
//! `high` to `high`; in-range samples pass through untouched. Useful
//! after operators that may push values outside `[0, 255]` and the
//! caller wants explicit floor/ceiling behaviour.
//!
//! Clean-room implementation — a one-line per-sample piecewise map.
//! Mirrors ImageMagick's `-clamp` (which only forces samples back into
//! `[0, 1]` in normalised float space; we extend it slightly so callers
//! can also specify a tighter window).
//!
//! # Pixel formats
//!
//! - `Gray8` / `Rgb24`: every channel is clamped.
//! - `Rgba`: alpha is preserved.
//! - `Yuv*P`: only the Y plane is clamped; chroma pass-through (chroma
//!   has its own legal range that doesn't match 0/255 endpoints).

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Clamp every tone sample into `[low, high]`.
#[derive(Clone, Copy, Debug)]
pub struct Clamp {
    low: u8,
    high: u8,
}

impl Default for Clamp {
    fn default() -> Self {
        Self::new(0, 255)
    }
}

impl Clamp {
    /// Build a clamp with inclusive endpoints. If `low > high` the two
    /// values are swapped — both ends are kept reachable so the output
    /// stays well-defined.
    pub fn new(low: u8, high: u8) -> Self {
        let (lo, hi) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };
        Self { low: lo, high: hi }
    }

    /// Lower bound (inclusive).
    pub fn low(&self) -> u8 {
        self.low
    }

    /// Upper bound (inclusive).
    pub fn high(&self) -> u8 {
        self.high
    }
}

impl ImageFilter for Clamp {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Clamp does not yet handle {:?}",
                params.format
            )));
        }
        let lo = self.low;
        let hi = self.high;
        let mut lut = [0u8; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as u8).clamp(lo, hi);
        }
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, VideoPlane};

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

    #[test]
    fn identity_window_passes_through() {
        let input = gray(4, 4, |x, y| (x * 16 + y * 16) as u8);
        let out = Clamp::new(0, 255)
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
    fn clips_below_low_above_high() {
        let input = gray(4, 1, |x, _| [10u8, 50, 200, 250][x as usize]);
        let out = Clamp::new(50, 200)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![50, 50, 200, 200]);
    }

    #[test]
    fn swapped_endpoints_are_normalised() {
        let c = Clamp::new(200, 50);
        assert_eq!(c.low(), 50);
        assert_eq!(c.high(), 200);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|i| [10u8, 50, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Clamp::new(50, 200)
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
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[0], 50);
            assert_eq!(px[1], 50);
            assert_eq!(px[2], 200);
            assert_eq!(px[3], i as u8);
        }
    }
}
