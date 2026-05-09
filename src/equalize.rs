//! Histogram equalisation.
//!
//! Standard textbook two-pass algorithm:
//!
//! 1. Build a 256-bin histogram (one per relevant tone channel).
//! 2. Compute the cumulative distribution function (CDF) by prefix-sum.
//! 3. Map every input value `v` to `(cdf[v] - cdf_min) / (N - cdf_min) * 255`
//!    so the resulting CDF is as linear as possible.
//!
//! Per-channel for RGB / RGBA (each channel independently equalised);
//! luma-only for YUV planar (chroma is left intact). Alpha passes
//! through unchanged.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Histogram-equalise a frame.
///
/// Carries no parameters. RGB/RGBA equalises each colour channel
/// independently; YUV equalises only the luma plane to leave hue
/// untouched.
#[derive(Clone, Copy, Debug, Default)]
pub struct Equalize;

impl Equalize {
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for Equalize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Equalize does not yet handle {:?}",
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
                let lut = build_channel_lut(&out.planes[0].data, out.planes[0].stride, w, h, 1, 0);
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                for y in 0..h {
                    for v in &mut plane.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
                // chroma planes untouched.
            }
            PixelFormat::Rgb24 => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                let lut_r = build_channel_lut(&plane.data, stride, w, h, 3, 0);
                let lut_g = build_channel_lut(&plane.data, stride, w, h, 3, 1);
                let lut_b = build_channel_lut(&plane.data, stride, w, h, 3, 2);
                for y in 0..h {
                    let row = &mut plane.data[y * stride..y * stride + 3 * w];
                    for x in 0..w {
                        row[3 * x] = lut_r[row[3 * x] as usize];
                        row[3 * x + 1] = lut_g[row[3 * x + 1] as usize];
                        row[3 * x + 2] = lut_b[row[3 * x + 2] as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                let lut_r = build_channel_lut(&plane.data, stride, w, h, 4, 0);
                let lut_g = build_channel_lut(&plane.data, stride, w, h, 4, 1);
                let lut_b = build_channel_lut(&plane.data, stride, w, h, 4, 2);
                for y in 0..h {
                    let row = &mut plane.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        row[4 * x] = lut_r[row[4 * x] as usize];
                        row[4 * x + 1] = lut_g[row[4 * x + 1] as usize];
                        row[4 * x + 2] = lut_b[row[4 * x + 2] as usize];
                        // alpha (index 3) preserved
                    }
                }
            }
            _ => {} // unreachable thanks to the is_supported_format guard.
        }

        Ok(out)
    }
}

/// Build a histogram-equalisation LUT for one interleaved channel.
///
/// `bpp` is the bytes-per-pixel stride between successive samples of
/// the same channel; `offset` selects which channel within the pixel.
fn build_channel_lut(
    data: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    bpp: usize,
    offset: usize,
) -> [u8; 256] {
    let mut hist = [0u32; 256];
    let mut total: u32 = 0;
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * bpp];
        for x in 0..w {
            let v = row[x * bpp + offset] as usize;
            hist[v] += 1;
            total += 1;
        }
    }

    // Prefix-sum into a CDF, then map via the standard formula.
    let mut cdf = [0u32; 256];
    let mut acc: u32 = 0;
    for (i, h_) in hist.iter().enumerate() {
        acc += *h_;
        cdf[i] = acc;
    }

    let mut lut = [0u8; 256];
    if total == 0 {
        for (i, e) in lut.iter_mut().enumerate() {
            *e = i as u8;
        }
        return lut;
    }
    // First non-empty bin: cdf_min for the standard formula.
    let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
    let denom = (total - cdf_min).max(1) as f32;
    for (i, e) in lut.iter_mut().enumerate() {
        let v = if cdf[i] > cdf_min {
            ((cdf[i] - cdf_min) as f32 / denom * 255.0).round()
        } else {
            0.0
        };
        *e = v.clamp(0.0, 255.0) as u8;
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
    fn equalise_spreads_compressed_range_to_full_range() {
        // Input samples sit only in [60, 80]; after equalize they should
        // span as close to [0, 255] as the discrete CDF allows.
        let input = gray(8, 8, |x, y| 60 + ((x + y) as u8) % 21);
        let out = Equalize::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0, "equalize should pull min to 0, got {min}");
        assert_eq!(max, 255, "equalize should push max to 255, got {max}");
    }

    #[test]
    fn equalise_already_uniform_is_near_identity() {
        // 4x4 with samples 0, 16, 32, ..., 240 covering the full range
        // uniformly. The CDF is already linear so equalisation should
        // approximate the identity.
        let input = gray(4, 4, |x, y| ((x + y * 4) * 17) as u8);
        let out = Equalize::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for (i, (&a, &b)) in out.planes[0]
            .data
            .iter()
            .zip(input.planes[0].data.iter())
            .enumerate()
        {
            // The discrete CDF can shift uniformly-spaced samples by a
            // constant amount; anchor on relative ordering instead.
            assert!(
                a as i16 >= b as i16 - 30 && a as i16 <= b as i16 + 30,
                "byte {i}: out {a} drifted too far from in {b}"
            );
        }
        // Max should still hit 255 and min should still hit 0.
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn equalise_flat_image_is_safe() {
        let input = gray(4, 4, |_, _| 100);
        let out = Equalize::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // A flat image equalises to a constant (single bin contains
        // everything).
        let v = out.planes[0].data[0];
        assert!(out.planes[0].data.iter().all(|&x| x == v));
    }

    #[test]
    fn equalise_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = 60 + (i as u8);
                [v, v.saturating_add(20), v.saturating_add(40), 222]
            })
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Equalize::new()
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
