//! Power-law gamma correction.
//!
//! Standard definition: `v' = 255 * (v/255)^(1/gamma)`. A gamma of 1.0
//! is the identity; `gamma > 1.0` brightens midtones; `gamma < 1.0`
//! darkens them. The transform is evaluated once into a 256-entry LUT
//! and then applied per pixel, which matches ImageMagick's `-gamma G`
//! in parameter semantics while being implemented from the textbook
//! formula (no source borrowed).

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Apply a power-law gamma curve to the tone channels.
///
/// - Gray / RGB / RGBA: gamma is applied per colour channel (alpha
///   preserved for RGBA).
/// - YUV planar: gamma is applied to the Y plane only; chroma is left
///   alone so hue/saturation stay stable.
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    gamma: f32,
}

impl Default for Gamma {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Gamma {
    /// Build a gamma filter. Values `<= 0.0` are clamped to a tiny
    /// positive epsilon so the LUT never divides by zero.
    pub fn new(gamma: f32) -> Self {
        Self {
            gamma: gamma.max(1e-6),
        }
    }
}

impl ImageFilter for Gamma {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Gamma does not yet handle {:?}",
                params.format
            )));
        }
        let lut = build_gamma_lut(self.gamma);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

fn build_gamma_lut(gamma: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let inv = 1.0f32 / gamma;
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = (i as f32) / 255.0;
        let mapped = v.powf(inv) * 255.0;
        *entry = mapped.round().clamp(0.0, 255.0) as u8;
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
    fn gamma_one_is_identity() {
        let input = gray(4, 4, |x, y| ((x + y * 4) * 15) as u8);
        let out = Gamma::new(1.0).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 }).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn gamma_two_brightens_midtones() {
        // Midtone 128 should map higher with gamma > 1.
        let input = gray(1, 1, |_, _| 128);
        let out = Gamma::new(2.0).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 1, height: 1 }).unwrap();
        let mapped = out.planes[0].data[0];
        // Expected ≈ 255 * sqrt(128/255) ≈ 181.
        assert!(mapped > 170 && mapped < 190, "mapped = {mapped}");
    }

    #[test]
    fn gamma_half_darkens_midtones() {
        let input = gray(1, 1, |_, _| 128);
        let out = Gamma::new(0.5).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 1, height: 1 }).unwrap();
        let mapped = out.planes[0].data[0];
        // Expected ≈ 255 * (128/255)^2 ≈ 64.
        assert!(mapped > 55 && mapped < 75, "mapped = {mapped}");
    }

    #[test]
    fn gamma_preserves_endpoints() {
        let input = gray(2, 1, |x, _| if x == 0 { 0 } else { 255 });
        let out = Gamma::new(2.2).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 2, height: 1 }).unwrap();
        assert_eq!(out.planes[0].data[0], 0);
        assert_eq!(out.planes[0].data[1], 255);
    }

    #[test]
    fn gamma_rgba_leaves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|_| [100u8, 100, 100, 77]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Gamma::new(2.0).apply(&input, VideoStreamParams { format: PixelFormat::Rgba, width: 4, height: 4 }).unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 77, "alpha preserved");
        }
    }
}
