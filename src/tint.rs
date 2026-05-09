//! Luminance-weighted tint toward a target colour.
//!
//! Differs from [`Colorize`](crate::Colorize) (which uses a uniform
//! `amount` blend across every pixel) by scaling the blend strength per
//! pixel by Rec. 709 luma:
//!
//! ```text
//! luma   = 0.2126 * R + 0.7152 * G + 0.0722 * B
//! weight = amount * luma / 255
//! out_R  = R * (1 - weight) + color_R * weight
//! ```
//!
//! Bright pixels pull toward the target colour more than dark pixels —
//! the classic ImageMagick `-tint` behaviour.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Luminance-weighted tint.
///
/// `color` is `[R, G, B, A]`; the target's alpha channel is ignored
/// (input alpha passes through unchanged on RGBA). `amount` is
/// effectively a global gain on the per-pixel luma weight: 0.0 returns
/// the input unchanged, 1.0 lets fully-bright pixels reach the target
/// colour completely while leaving black pixels untouched.
#[derive(Clone, Copy, Debug)]
pub struct Tint {
    pub color: [u8; 4],
    pub amount: f32,
}

impl Default for Tint {
    fn default() -> Self {
        Self {
            color: [128, 128, 128, 255],
            amount: 1.0,
        }
    }
}

impl Tint {
    /// Build a tint with explicit target colour and amount. `amount`
    /// is clamped to a non-negative value (negative tints are not
    /// meaningful in this model).
    pub fn new(color: [u8; 4], amount: f32) -> Self {
        Self {
            color,
            amount: amount.max(0.0),
        }
    }
}

impl ImageFilter for Tint {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Tint requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let amount = self.amount.max(0.0);
        let tr = self.color[0] as f32;
        let tg = self.color[1] as f32;
        let tb = self.color[2] as f32;

        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32;
                let g = src_row[base + 1] as f32;
                let b = src_row[base + 2] as f32;
                // Rec. 709 luma weights.
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let weight = (amount * luma / 255.0).clamp(0.0, 1.0);
                let inv = 1.0 - weight;
                let or = r * inv + tr * weight;
                let og = g * inv + tg * weight;
                let ob = b * inv + tb * weight;
                dst_row[base] = or.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = og.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = ob.round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    dst_row[base + 3] = src_row[base + 3];
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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn amount_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Tint::new([255, 0, 0, 255], 0.0)
            .apply(&input, params_rgb(8, 8))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn black_pixels_are_untouched() {
        let input = rgb(4, 4, |_, _| (0, 0, 0));
        let out = Tint::new([255, 0, 0, 255], 1.0)
            .apply(&input, params_rgb(4, 4))
            .unwrap();
        // luma = 0 ⇒ weight = 0 ⇒ output unchanged.
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 0);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    #[test]
    fn white_pixels_take_full_target() {
        let input = rgb(4, 4, |_, _| (255, 255, 255));
        let out = Tint::new([100, 50, 25, 255], 1.0)
            .apply(&input, params_rgb(4, 4))
            .unwrap();
        // luma ≈ 255 ⇒ weight ≈ 1 ⇒ output ≈ target.
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 50);
            assert_eq!(chunk[2], 25);
        }
    }

    #[test]
    fn mid_grey_blends_halfway() {
        let input = rgb(2, 2, |_, _| (128, 128, 128));
        let out = Tint::new([0, 0, 0, 255], 1.0)
            .apply(&input, params_rgb(2, 2))
            .unwrap();
        // luma = 128 ⇒ weight ≈ 0.502 ⇒ out ≈ 128 * (1 - 0.502) = 63.74.
        for chunk in out.planes[0].data.chunks(3) {
            assert!((chunk[0] as i16 - 64).abs() <= 1, "chunk[0] = {}", chunk[0]);
            assert!((chunk[1] as i16 - 64).abs() <= 1);
            assert!((chunk[2] as i16 - 64).abs() <= 1);
        }
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let data: Vec<u8> = (0..16).flat_map(|_| [255u8, 255, 255, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Tint::new([10, 20, 30, 255], 1.0)
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
            // White input + amount=1 ⇒ near-target colour.
            assert_eq!(out.planes[0].data[i * 4], 10);
            assert_eq!(out.planes[0].data[i * 4 + 1], 20);
            assert_eq!(out.planes[0].data[i * 4 + 2], 30);
        }
    }

    #[test]
    fn rejects_yuv420p() {
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
        let err = Tint::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Tint"));
    }
}
