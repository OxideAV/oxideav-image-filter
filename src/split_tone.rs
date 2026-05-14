//! Split-toning — tint shadows and highlights with two independent
//! colours, leaving midtones untouched.
//!
//! Driven by a soft tonal mask `w(v) = (1 - 2|v - 0.5|).max(0)` that
//! peaks at zero in the midtone and one at the extremes:
//!
//! ```text
//! shadow_mask(v)    = max(0, 1 - 2v)        // 1 at v=0, 0 at v>=0.5
//! highlight_mask(v) = max(0, 2v - 1)        // 0 at v<=0.5, 1 at v=1
//! ```
//!
//! For each pixel the shadow tint and highlight tint are blended in
//! proportional to those two masks, scaled by `balance` (which biases
//! the shadow vs highlight side) and `amount` (overall strength).
//!
//! Inspired by Lightroom's Split Toning panel. The maths is clean-room.
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba`. Alpha is preserved on RGBA. Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Split-toning filter.
#[derive(Clone, Copy, Debug)]
pub struct SplitTone {
    /// Tint colour applied to the shadow half (RGB).
    pub shadow_color: [u8; 3],
    /// Tint colour applied to the highlight half (RGB).
    pub highlight_color: [u8; 3],
    /// `[-1, 1]` — negative biases shadows, positive biases highlights;
    /// 0 keeps them equal.
    pub balance: f32,
    /// `[0, 1]` — global blend amount. 0 returns input, 1 saturates the
    /// per-pixel tint.
    pub amount: f32,
}

impl Default for SplitTone {
    fn default() -> Self {
        Self {
            shadow_color: [0, 30, 70],
            highlight_color: [255, 220, 80],
            balance: 0.0,
            amount: 0.5,
        }
    }
}

impl SplitTone {
    /// Build with explicit shadow + highlight tints, balance, and amount.
    pub fn new(shadow: [u8; 3], highlight: [u8; 3], balance: f32, amount: f32) -> Self {
        Self {
            shadow_color: shadow,
            highlight_color: highlight,
            balance: balance.clamp(-1.0, 1.0),
            amount: amount.clamp(0.0, 1.0),
        }
    }
}

impl ImageFilter for SplitTone {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: SplitTone requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let row = w * bpp;
        let p = &input.planes[0];
        let mut data = vec![0u8; row * h];

        let amount = self.amount.clamp(0.0, 1.0);
        let bal = self.balance.clamp(-1.0, 1.0);
        // Re-balance: positive `bal` boosts highlights; negative boosts
        // shadows. Each weight stays in `[0, 1]` after the bias.
        let shadow_scale = (1.0 - bal).clamp(0.0, 1.0);
        let highlight_scale = (1.0 + bal).clamp(0.0, 1.0);
        let sc = [
            self.shadow_color[0] as f32,
            self.shadow_color[1] as f32,
            self.shadow_color[2] as f32,
        ];
        let hc = [
            self.highlight_color[0] as f32,
            self.highlight_color[1] as f32,
            self.highlight_color[2] as f32,
        ];

        for y in 0..h {
            let src_row = &p.data[y * p.stride..y * p.stride + row];
            let dst_row = &mut data[y * row..(y + 1) * row];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32;
                let g = src_row[base + 1] as f32;
                let b = src_row[base + 2] as f32;
                // Rec. 601 luma → normalised tone position in [0, 1].
                let v = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                let sw = (1.0 - 2.0 * v).max(0.0) * shadow_scale * amount;
                let hw = (2.0 * v - 1.0).max(0.0) * highlight_scale * amount;
                // The "base" weight that keeps the original colour.
                let kw = (1.0 - sw - hw).max(0.0);
                let r2 = kw * r + sw * sc[0] + hw * hc[0];
                let g2 = kw * g + sw * sc[1] + hw * hc[1];
                let b2 = kw * b + sw * sc[2] + hw * hc[2];
                dst_row[base] = r2.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = g2.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = b2.round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    dst_row[base + 3] = src_row[base + 3];
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride: row, data }],
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn amount_zero_is_passthrough() {
        let input = rgb(4, 4, |x, y| (x as u8 * 30, y as u8 * 30, 77));
        let out = SplitTone::new([0, 0, 255], [255, 255, 0], 0.0, 0.0)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn midtone_is_untouched() {
        // RGB (128, 128, 128) ⇒ v = 0.502 ⇒ both masks ≈ 0.
        let input = rgb(1, 1, |_, _| (128, 128, 128));
        let out = SplitTone::new([255, 0, 0], [0, 0, 255], 0.0, 1.0)
            .apply(&input, p_rgb(1, 1))
            .unwrap();
        // The midtone is the soft mask's null point — output stays
        // within a couple of LSBs of the input.
        for c in 0..3 {
            let v = out.planes[0].data[c] as i16;
            assert!((v - 128).abs() <= 4, "channel {c}: expected ≈128, got {v}");
        }
    }

    #[test]
    fn black_takes_shadow_colour() {
        let input = rgb(1, 1, |_, _| (0, 0, 0));
        let out = SplitTone::new([200, 60, 40], [0, 0, 0], 0.0, 1.0)
            .apply(&input, p_rgb(1, 1))
            .unwrap();
        // v=0 ⇒ shadow_mask=1, highlight_mask=0 ⇒ saturates toward
        // shadow_colour.
        assert!((out.planes[0].data[0] as i16 - 200).abs() <= 4);
        assert!((out.planes[0].data[1] as i16 - 60).abs() <= 4);
        assert!((out.planes[0].data[2] as i16 - 40).abs() <= 4);
    }

    #[test]
    fn white_takes_highlight_colour() {
        let input = rgb(1, 1, |_, _| (255, 255, 255));
        let out = SplitTone::new([0, 0, 0], [10, 200, 50], 0.0, 1.0)
            .apply(&input, p_rgb(1, 1))
            .unwrap();
        // v=1 ⇒ highlight_mask=1, shadow_mask=0.
        assert!((out.planes[0].data[0] as i16 - 10).abs() <= 4);
        assert!((out.planes[0].data[1] as i16 - 200).abs() <= 4);
        assert!((out.planes[0].data[2] as i16 - 50).abs() <= 4);
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
        let err = SplitTone::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("SplitTone"));
    }
}
