//! Solarisation — invert only the bright half of the tone range,
//! simulating the classic darkroom effect of exposing a print to
//! light mid-development.
//!
//! Clean-room formula from the standard definition:
//!
//! ```text
//! v' = if v >= threshold { 255 - v } else { v }
//! ```
//!
//! ImageMagick spells this `-solarize N%`; the CLI is responsible for
//! mapping percent syntax to the `u8` threshold taken here.

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Invert every sample above `threshold` ("solarise" effect).
///
/// - Gray / R / G / B: the rule is applied per channel.
/// - RGBA: alpha preserved.
/// - YUV planar: Y plane solarised; chroma left alone so the image
///   retains its colour identity.
#[derive(Clone, Copy, Debug)]
pub struct Solarize {
    threshold: u8,
}

impl Default for Solarize {
    fn default() -> Self {
        Self::new(128)
    }
}

impl Solarize {
    /// Build the filter with a `u8` threshold.
    pub fn new(threshold: u8) -> Self {
        Self { threshold }
    }
}

impl ImageFilter for Solarize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Solarize does not yet handle {:?}",
                params.format
            )));
        }
        let lut = build_solarize_lut(self.threshold);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

fn build_solarize_lut(threshold: u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = i as u8;
        *entry = if v >= threshold { 255 - v } else { v };
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
    fn solarize_below_threshold_is_identity() {
        let input = gray(4, 1, |x, _| x as u8 * 30);
        // All samples < 128 with threshold 128 → identity.
        let out = Solarize::new(128).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 1 }).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn solarize_above_threshold_inverts() {
        let input = gray(3, 1, |x, _| match x {
            0 => 50,
            1 => 128,
            _ => 200,
        });
        let out = Solarize::new(128).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 3, height: 1 }).unwrap();
        assert_eq!(out.planes[0].data[0], 50); // below threshold → unchanged
        assert_eq!(out.planes[0].data[1], 127); // 255 - 128
        assert_eq!(out.planes[0].data[2], 55); // 255 - 200
    }

    #[test]
    fn solarize_threshold_zero_is_full_negate() {
        let input = gray(4, 1, |x, _| x as u8 * 60);
        let out = Solarize::new(0).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 1 }).unwrap();
        for (i, v) in out.planes[0].data.iter().enumerate() {
            assert_eq!(*v, 255 - input.planes[0].data[i]);
        }
    }

    #[test]
    fn solarize_threshold_255_is_almost_identity() {
        let input = gray(4, 1, |x, _| x as u8 * 60);
        let out = Solarize::new(255).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 1 }).unwrap();
        for i in 0..3 {
            assert_eq!(out.planes[0].data[i], input.planes[0].data[i]);
        }
    }

    #[test]
    fn solarize_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|_| [200u8, 50, 200, 77]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Solarize::new(128).apply(&input, VideoStreamParams { format: PixelFormat::Rgba, width: 4, height: 4 }).unwrap();
        for i in 0..16 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[0], 55); // 255 - 200
            assert_eq!(px[1], 50); // below threshold
            assert_eq!(px[2], 55);
            assert_eq!(px[3], 77); // alpha preserved
        }
    }
}
