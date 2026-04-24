//! ImageMagick-style `-brightness-contrast` combining linear brightness
//! and contrast adjustments in one pass.
//!
//! Clean-room formula from the textbook definition of a linear tone
//! operator:
//!
//! ```text
//! v' = clamp(((v - 128) * (1 + C/100)) + 128 + B*1.28)
//! ```
//!
//! Both `B` and `C` live in `-100..=100`. Applied via a 256-entry LUT.

use crate::{is_supported, tonal_lut::apply_tone_lut, ImageFilter};
use oxideav_core::{Error, VideoFrame};

/// Combined brightness and contrast adjustment.
///
/// - `brightness`: additive offset on a `-100..=100` scale. Each step
///   shifts the output by 1.28 sample units (so `±100` covers the full
///   dynamic range before clamping).
/// - `contrast`: multiplicative gain around mid-grey (128) on a
///   `-100..=100` scale. `0` is no change; `100` doubles the gain;
///   `-100` flattens the image to mid-grey.
///
/// Applied to Y / R / G / B channels. Alpha and YUV chroma are left
/// untouched by design.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrightnessContrast {
    brightness: f32,
    contrast: f32,
}

impl BrightnessContrast {
    /// Build a filter with explicit brightness and contrast in the
    /// range `-100..=100`.
    pub fn new(brightness: f32, contrast: f32) -> Self {
        Self {
            brightness: brightness.clamp(-100.0, 100.0),
            contrast: contrast.clamp(-100.0, 100.0),
        }
    }
}

impl ImageFilter for BrightnessContrast {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: BrightnessContrast does not yet handle {:?}",
                input.format
            )));
        }
        let lut = build_lut(self.brightness, self.contrast);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut);
        Ok(out)
    }
}

fn build_lut(brightness: f32, contrast: f32) -> [u8; 256] {
    let gain = 1.0 + contrast / 100.0;
    let offset = 128.0 + brightness * 1.28;
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = (i as f32 - 128.0) * gain + offset;
        *entry = v.round().clamp(0.0, 255.0) as u8;
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
            format: PixelFormat::Gray8,
            width: w,
            height: h,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    #[test]
    fn zero_zero_is_identity() {
        let input = gray(4, 4, |x, y| ((x + y * 4) * 15) as u8);
        let out = BrightnessContrast::new(0.0, 0.0).apply(&input).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn pure_brightness_shifts_up() {
        // +50 brightness adds 64 (50 * 1.28) to every sample.
        let input = gray(1, 1, |_, _| 100);
        let out = BrightnessContrast::new(50.0, 0.0).apply(&input).unwrap();
        assert_eq!(out.planes[0].data[0], 164);
    }

    #[test]
    fn pure_brightness_clamps_at_255() {
        let input = gray(1, 1, |_, _| 240);
        let out = BrightnessContrast::new(100.0, 0.0).apply(&input).unwrap();
        assert_eq!(out.planes[0].data[0], 255);
    }

    #[test]
    fn pure_contrast_pivots_on_128() {
        // Contrast +100 doubles the gain around 128.
        // For v=64 → (64-128)*2 + 128 = 0. For v=192 → (192-128)*2 + 128 = 256 → clamp 255.
        let input = gray(2, 1, |x, _| if x == 0 { 64 } else { 192 });
        let out = BrightnessContrast::new(0.0, 100.0).apply(&input).unwrap();
        assert_eq!(out.planes[0].data[0], 0);
        assert_eq!(out.planes[0].data[1], 255);
    }

    #[test]
    fn contrast_negative_hundred_flattens_to_mid() {
        // Gain 0 → every sample collapses to 128.
        let input = gray(4, 4, |x, y| ((x + y * 4) * 15) as u8);
        let out = BrightnessContrast::new(0.0, -100.0).apply(&input).unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 128);
        }
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|_| [50u8, 50, 50, 77]).collect();
        let input = VideoFrame {
            format: PixelFormat::Rgba,
            width: 4,
            height: 4,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = BrightnessContrast::new(20.0, 20.0).apply(&input).unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 77);
        }
    }
}
