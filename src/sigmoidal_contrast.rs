//! Sigmoid-curve contrast adjustment.
//!
//! ImageMagick's `-sigmoidal-contrast CxM%` operator: a smooth S-curve
//! around a midpoint. With `contrast = 0` the curve degenerates to the
//! identity; positive values steepen contrast around the midpoint
//! while keeping the endpoints anchored at 0 / 255.
//!
//! ```text
//! alpha = contrast
//! beta  = midpoint / 255
//! sigmoid(x) = 1 / (1 + exp(-alpha * (x - beta)))
//! out_norm   = (sigmoid(in/255) - sigmoid(0)) / (sigmoid(1) - sigmoid(0))
//! out_byte   = clamp(out_norm * 255, 0, 255)
//! ```
//!
//! Implemented as a 256-entry LUT for cheap repeat-application; the
//! tone-channel routing (Y for YUV, RGB for packed RGB, alpha
//! preserved on RGBA) is the same as for [`Gamma`](crate::Gamma) and
//! friends — see [`apply_tone_lut`](crate::tonal_lut::apply_tone_lut).

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Sigmoid-curve contrast filter.
///
/// `contrast` is the slope at the midpoint; values in `1.0..=10.0` give
/// the visually-meaningful range. `0.0` is bit-exact identity.
/// `midpoint` is the inflection point on the input intensity scale
/// (`0..=255`); the default 128 places it at mid-grey.
#[derive(Clone, Copy, Debug)]
pub struct SigmoidalContrast {
    pub contrast: f32,
    pub midpoint: f32,
}

impl Default for SigmoidalContrast {
    fn default() -> Self {
        Self {
            contrast: 3.0,
            midpoint: 128.0,
        }
    }
}

impl SigmoidalContrast {
    /// Build a sigmoidal-contrast filter with explicit slope and
    /// midpoint. `contrast` is clamped to a finite value (NaN ⇒ 0).
    pub fn new(contrast: f32, midpoint: f32) -> Self {
        Self {
            contrast: if contrast.is_finite() { contrast } else { 0.0 },
            midpoint: if midpoint.is_finite() {
                midpoint.clamp(0.0, 255.0)
            } else {
                128.0
            },
        }
    }
}

impl ImageFilter for SigmoidalContrast {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: SigmoidalContrast does not yet handle {:?}",
                params.format
            )));
        }

        let lut = build_lut(self.contrast, self.midpoint);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

fn build_lut(contrast: f32, midpoint: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if contrast.abs() < 1e-6 {
        for (i, e) in lut.iter_mut().enumerate() {
            *e = i as u8;
        }
        return lut;
    }
    let alpha = contrast;
    let beta = (midpoint / 255.0).clamp(0.0, 1.0);
    let s0 = sigmoid(0.0, alpha, beta);
    let s1 = sigmoid(1.0, alpha, beta);
    let denom = s1 - s0;
    if denom.abs() < 1e-12 {
        for (i, e) in lut.iter_mut().enumerate() {
            *e = i as u8;
        }
        return lut;
    }
    for (i, e) in lut.iter_mut().enumerate() {
        let x = i as f32 / 255.0;
        let s = sigmoid(x, alpha, beta);
        let normalised = ((s - s0) / denom).clamp(0.0, 1.0);
        *e = (normalised * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

#[inline]
fn sigmoid(x: f32, alpha: f32, beta: f32) -> f32 {
    1.0 / (1.0 + (-alpha * (x - beta)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn contrast_zero_is_identity() {
        let input = gray(8, 8, |x, y| (x + y * 8) as u8 * 2);
        let out = SigmoidalContrast::new(0.0, 128.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn contrast_maps_endpoints_to_endpoints() {
        // Anchored sigmoid: lut[0] should be 0 and lut[255] should be 255.
        let lut = build_lut(5.0, 128.0);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn contrast_steepens_around_midpoint() {
        // For positive contrast values the output for a sample below the
        // midpoint should be darker than identity; above the midpoint
        // brighter.
        let lut = build_lut(5.0, 128.0);
        // 64 (below midpoint) should be pulled toward 0.
        assert!(lut[64] < 64, "lut[64] = {} should be < 64", lut[64]);
        // 192 (above midpoint) should be pushed toward 255.
        assert!(lut[192] > 192, "lut[192] = {} should be > 192", lut[192]);
        // Midpoint should stay near 128 by construction.
        assert!(
            (lut[128] as i16 - 128).abs() <= 2,
            "lut[128] = {}",
            lut[128]
        );
    }

    #[test]
    fn rgba_alpha_preserved() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = (i * 16) as u8;
                [v, v, v, 222]
            })
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = SigmoidalContrast::new(3.0, 128.0)
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
            assert_eq!(out.planes[0].data[i * 4 + 3], 222, "alpha at pixel {i}");
        }
    }

    #[test]
    fn yuv_only_luma_changes() {
        let y_data = vec![64u8; 16];
        let cb_data = vec![100u8; 4];
        let cr_data = vec![160u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: y_data,
                },
                VideoPlane {
                    stride: 2,
                    data: cb_data.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: cr_data.clone(),
                },
            ],
        };
        let out = SigmoidalContrast::new(5.0, 128.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Y plane should be darker than 64 (below midpoint, +contrast).
        assert!(out.planes[0].data[0] < 64);
        // Chroma planes untouched.
        assert_eq!(out.planes[1].data, cb_data);
        assert_eq!(out.planes[2].data, cr_data);
    }
}
