//! Warm-brown "sepia tone" colour transform.
//!
//! Clean-room implementation based on the linear colour-mixing matrix
//! listed on the Wikipedia "Sepia tone" article.
//!
//! ```text
//! r' = 0.393·r + 0.769·g + 0.189·b
//! g' = 0.349·r + 0.686·g + 0.168·b
//! b' = 0.272·r + 0.534·g + 0.131·b
//! ```
//!
//! A `threshold` parameter in `0.0..=1.0` blends between the original
//! (0) and the full sepia conversion (1).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Sepia-tone colour remap for RGB/RGBA frames.
///
/// `threshold` weights the mix between the original pixel (0.0) and
/// the full sepia-transformed pixel (1.0). Alpha passes through.
#[derive(Clone, Debug)]
pub struct Sepia {
    pub threshold: f32,
}

impl Default for Sepia {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

impl Sepia {
    /// Build with an explicit strength in `0.0..=1.0`.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }
}

impl ImageFilter for Sepia {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Sepia requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let t = self.threshold.clamp(0.0, 1.0);
        let mut out = vec![0u8; row_bytes * h];

        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let r = src_row[x * bpp] as f32;
                let g = src_row[x * bpp + 1] as f32;
                let b = src_row[x * bpp + 2] as f32;

                let sr = 0.393 * r + 0.769 * g + 0.189 * b;
                let sg = 0.349 * r + 0.686 * g + 0.168 * b;
                let sb = 0.272 * r + 0.534 * g + 0.131 * b;

                let nr = (r * (1.0 - t) + sr * t).clamp(0.0, 255.0);
                let ng = (g * (1.0 - t) + sg * t).clamp(0.0, 255.0);
                let nb = (b * (1.0 - t) + sb * t).clamp(0.0, 255.0);

                dst_row[x * bpp] = nr.round() as u8;
                dst_row[x * bpp + 1] = ng.round() as u8;
                dst_row[x * bpp + 2] = nb.round() as u8;
                if has_alpha {
                    dst_row[x * bpp + 3] = src_row[x * bpp + 3];
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_bytes,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = f(x, y);
                data.extend_from_slice(&[r, g, b]);
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

    #[test]
    fn pure_red_to_warm_brown() {
        let input = rgb(2, 2, |_, _| (255, 0, 0));
        let out = Sepia::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // 0.393*255 = 100.215 -> 100
        // 0.349*255 =  88.995 ->  89
        // 0.272*255 =  69.36  ->  69
        for chunk in out.planes[0].data.chunks(3) {
            assert!((chunk[0] as i16 - 100).abs() <= 1, "r = {}", chunk[0]);
            assert!((chunk[1] as i16 - 89).abs() <= 1, "g = {}", chunk[1]);
            assert!((chunk[2] as i16 - 69).abs() <= 1, "b = {}", chunk[2]);
        }
    }

    #[test]
    fn threshold_zero_is_identity() {
        let input = rgb(4, 4, |x, y| ((x * 40) as u8, (y * 40) as u8, 77));
        let out = Sepia::new(0.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn threshold_half_interpolates() {
        let input = rgb(1, 1, |_, _| (255, 0, 0));
        let out = Sepia::new(0.5)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        // Halfway between (255, 0, 0) and (100, 89, 69) ≈ (177, 44, 34).
        assert!((out.planes[0].data[0] as i16 - 178).abs() <= 2);
        assert!((out.planes[0].data[1] as i16 - 45).abs() <= 2);
        assert!((out.planes[0].data[2] as i16 - 35).abs() <= 2);
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let data = vec![100, 50, 25, 200, 0, 255, 0, 77];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Sepia::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[3], 200);
        assert_eq!(out.planes[0].data[7], 77);
    }

    #[test]
    fn rejects_yuv() {
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let err = Sepia::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Sepia"));
    }
}
