//! Tint each pixel toward a target colour by a configurable amount.
//!
//! Standard linear blend:
//!
//! ```text
//! out_rgb = in_rgb * (1 - amount) + target_rgb * amount
//! ```
//!
//! Alpha is preserved on RGBA. Mirrors ImageMagick's `-colorize <pct>`
//! using its source-fill colour, with `amount = pct / 100`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Linear-blend colorize.
///
/// `color` is `[R, G, B, A]`; the alpha channel of `color` is ignored
/// (the input's alpha passes through unchanged). `amount` is clamped to
/// `[0.0, 1.0]`; 0.0 returns the input unchanged, 1.0 returns the
/// target colour for every channel.
#[derive(Clone, Copy, Debug)]
pub struct Colorize {
    pub color: [u8; 4],
    pub amount: f32,
}

impl Default for Colorize {
    fn default() -> Self {
        Self {
            color: [255, 255, 255, 255],
            amount: 0.5,
        }
    }
}

impl Colorize {
    /// Build a colorize filter with explicit target colour and blend
    /// amount. `amount` is clamped to `[0.0, 1.0]`.
    pub fn new(color: [u8; 4], amount: f32) -> Self {
        Self {
            color,
            amount: amount.clamp(0.0, 1.0),
        }
    }
}

impl ImageFilter for Colorize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Colorize requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let a = self.amount.clamp(0.0, 1.0);
        let inv = 1.0 - a;
        let tr = self.color[0] as f32;
        let tg = self.color[1] as f32;
        let tb = self.color[2] as f32;

        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32 * inv + tr * a;
                let g = src_row[base + 1] as f32 * inv + tg * a;
                let b = src_row[base + 2] as f32 * inv + tb * a;
                dst_row[base] = r.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = g.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = b.round().clamp(0.0, 255.0) as u8;
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
        let out = Colorize::new([255, 0, 0, 255], 0.0)
            .apply(&input, params_rgb(8, 8))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn amount_one_replaces_with_target() {
        let input = rgb(4, 4, |_, _| (10, 20, 30));
        let out = Colorize::new([200, 100, 50, 255], 1.0)
            .apply(&input, params_rgb(4, 4))
            .unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 200);
            assert_eq!(chunk[1], 100);
            assert_eq!(chunk[2], 50);
        }
    }

    #[test]
    fn amount_half_blends() {
        let input = rgb(2, 2, |_, _| (0, 0, 0));
        let out = Colorize::new([200, 100, 50, 255], 0.5)
            .apply(&input, params_rgb(2, 2))
            .unwrap();
        // Halfway between (0,0,0) and (200,100,50) ⇒ (100,50,25).
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 50);
            assert_eq!(chunk[2], 25);
        }
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let data: Vec<u8> = (0..16).flat_map(|_| [50u8, 100, 150, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Colorize::new([0, 0, 0, 255], 1.0)
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
            // Colour fully replaced.
            assert_eq!(out.planes[0].data[i * 4], 0);
            assert_eq!(out.planes[0].data[i * 4 + 1], 0);
            assert_eq!(out.planes[0].data[i * 4 + 2], 0);
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
        let err = Colorize::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Colorize"));
    }
}
