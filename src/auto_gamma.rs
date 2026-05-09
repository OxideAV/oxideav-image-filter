//! Auto-gamma: pick a gamma so the geometric mean of the input lands at
//! 0.5, then apply a power-law correction.
//!
//! Standard textbook formula:
//!
//! ```text
//! mean  = exp(sum(log(in / 255 + eps)) / N)
//! gamma = log(mean) / log(0.5)
//! out   = 255 * (in / 255)^(1/gamma)
//! ```
//!
//! With `mean < 0.5` (dark input) we get `gamma > 1.0` which raises
//! `in/255` to an exponent `< 1.0` → brighter midtones; with
//! `mean > 0.5` we get `gamma < 1.0` which darkens. The choice
//! satisfies `mean^(1/gamma) = 0.5` so the geometric mean of the
//! output lands exactly at mid-grey.
//!
//! Per-channel for RGB / RGBA so any colour cast in the input is
//! corrected as a side effect (each channel is normalised toward a
//! mid-grey 0.5 mean independently). YUV planar runs the correction on
//! the luma plane only — chroma is left intact so hue/saturation
//! stay stable. Mirrors ImageMagick's `-auto-gamma` parameter
//! semantics.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Auto-gamma correction.
///
/// Carries no parameters. The gamma value is computed per channel from
/// the input data each time `apply()` runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoGamma;

impl AutoGamma {
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for AutoGamma {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: AutoGamma does not yet handle {:?}",
                params.format
            )));
        }

        let w = params.width as usize;
        let h = params.height as usize;
        let mut out = input.clone();

        match params.format {
            PixelFormat::Gray8
            | PixelFormat::Yuv420P
            | PixelFormat::Yuv422P
            | PixelFormat::Yuv444P => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                let g = compute_gamma(&plane.data, stride, w, h, 1, 0);
                let lut = build_gamma_lut(g);
                for y in 0..h {
                    for v in &mut plane.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                let gr = compute_gamma(&plane.data, stride, w, h, 3, 0);
                let gg = compute_gamma(&plane.data, stride, w, h, 3, 1);
                let gb = compute_gamma(&plane.data, stride, w, h, 3, 2);
                let lr = build_gamma_lut(gr);
                let lg = build_gamma_lut(gg);
                let lb = build_gamma_lut(gb);
                for y in 0..h {
                    let row = &mut plane.data[y * stride..y * stride + 3 * w];
                    for x in 0..w {
                        row[3 * x] = lr[row[3 * x] as usize];
                        row[3 * x + 1] = lg[row[3 * x + 1] as usize];
                        row[3 * x + 2] = lb[row[3 * x + 2] as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                let gr = compute_gamma(&plane.data, stride, w, h, 4, 0);
                let gg = compute_gamma(&plane.data, stride, w, h, 4, 1);
                let gb = compute_gamma(&plane.data, stride, w, h, 4, 2);
                let lr = build_gamma_lut(gr);
                let lg = build_gamma_lut(gg);
                let lb = build_gamma_lut(gb);
                for y in 0..h {
                    let row = &mut plane.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        row[4 * x] = lr[row[4 * x] as usize];
                        row[4 * x + 1] = lg[row[4 * x + 1] as usize];
                        row[4 * x + 2] = lb[row[4 * x + 2] as usize];
                        // alpha (index 3) preserved
                    }
                }
            }
            _ => {}
        }

        Ok(out)
    }
}

/// Pick a gamma for one interleaved channel so its geometric mean
/// lands at 0.5.
fn compute_gamma(data: &[u8], stride: usize, w: usize, h: usize, bpp: usize, offset: usize) -> f32 {
    const EPS: f32 = 1.0 / 1024.0;
    let mut log_sum: f64 = 0.0;
    let mut count: u32 = 0;
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * bpp];
        for x in 0..w {
            let v = row[x * bpp + offset] as f32 / 255.0 + EPS;
            log_sum += (v as f64).ln();
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    let mean = (log_sum / count as f64).exp();
    // Avoid log(<=0) and degenerate gamma values.
    if !(0.0..=1.0).contains(&mean) || mean <= 1e-6 {
        return 1.0;
    }
    let log_mean = mean.ln();
    if log_mean.abs() < 1e-6 {
        return 1.0;
    }
    // gamma = log(mean) / log(0.5); applying out = in^(1/gamma) then
    // gives mean_out = mean^(1/gamma) = 0.5.
    let gamma = (log_mean / 0.5_f64.ln()) as f32;
    if !gamma.is_finite() || gamma <= 0.0 {
        1.0
    } else {
        gamma.clamp(0.01, 100.0)
    }
}

fn build_gamma_lut(gamma: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let inv = 1.0_f32 / gamma;
    for (i, e) in lut.iter_mut().enumerate() {
        let v = (i as f32) / 255.0;
        let mapped = v.powf(inv) * 255.0;
        *e = mapped.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

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
    fn dark_image_is_brightened_toward_mid_grey() {
        // All samples at 30 ⇒ mean ~0.117 ⇒ gamma > 1 ⇒ output > 30.
        let input = gray(8, 8, |_, _| 30);
        let out = AutoGamma::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        let v = out.planes[0].data[0];
        // Every output should be brighter than the input mean of 30 and
        // close to the target mid-grey of 128.
        assert!(v > 100, "auto-gamma did not brighten dark input: {v}");
        assert!(out.planes[0].data.iter().all(|&x| x == v));
    }

    #[test]
    fn bright_image_is_darkened_toward_mid_grey() {
        let input = gray(8, 8, |_, _| 220);
        let out = AutoGamma::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        let v = out.planes[0].data[0];
        assert!(v < 180, "auto-gamma did not darken bright input: {v}");
    }

    #[test]
    fn near_mid_grey_input_is_near_identity() {
        // Geometric mean already very close to 0.5 ⇒ gamma ≈ 1 ⇒ near
        // pass-through.
        let input = gray(4, 4, |_, _| 128);
        let out = AutoGamma::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        let v = out.planes[0].data[0];
        assert!((v as i16 - 128).abs() <= 4, "drift = {}", v as i16 - 128);
    }

    #[test]
    fn endpoints_preserved() {
        // 0 stays 0, 255 stays 255 regardless of the chosen gamma.
        let mut data = vec![0u8; 16];
        data[0] = 0;
        data[15] = 255;
        for v in data.iter_mut().take(15).skip(1) {
            *v = 100;
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 4, data }],
        };
        let out = AutoGamma::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 0);
        assert_eq!(out.planes[0].data[15], 255);
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let data: Vec<u8> = (0..16).flat_map(|_| [50u8, 100, 150, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = AutoGamma::new()
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
}
