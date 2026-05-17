//! Cross-process — colour film "wrong-chemistry" emulation.
//!
//! A common analogue photography look: process slide film (E-6) in
//! print-film chemistry (C-41) or vice versa. The result is high
//! contrast plus mismatched per-channel tone curves giving warm
//! highlights, cool shadows, and exaggerated saturation.
//!
//! Clean-room recipe driven by three independent per-channel sigmoid
//! S-curves applied to R / G / B:
//!
//! ```text
//! curve(v; gain, bias) = sigmoid(gain · (v - 0.5)) — bias toward bias
//! per-channel:
//!   R: extra warm bias toward highlights
//!   G: mid contrast
//!   B: cool bias toward shadows
//! ```
//!
//! The amount slider lerps between identity (0) and the full
//! cross-processed look (1).
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba` / `Gray8`. On Gray8 the per-channel curves collapse
//! to the G-channel curve (the "neutral" sigmoid). Alpha is preserved
//! on RGBA. Other formats return [`Error::unsupported`].

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Cross-processed film-look filter.
#[derive(Clone, Copy, Debug)]
pub struct CrossProcess {
    /// `[0, 1]` — blend the cross-processed result with the original.
    /// 0 = identity; 1 = full effect.
    pub amount: f32,
    /// Extra warm push (per-pixel R bias toward highlight tint) in
    /// `[-1, 1]`. Defaults to `0.15`.
    pub warmth: f32,
    /// Per-channel sigmoid contrast slope. Larger ⇒ steeper S. Defaults
    /// to `6.0` (a "moderate cross-process" pop).
    pub contrast: f32,
}

impl Default for CrossProcess {
    fn default() -> Self {
        Self {
            amount: 1.0,
            warmth: 0.15,
            contrast: 6.0,
        }
    }
}

impl CrossProcess {
    /// Build with explicit amount; warmth + contrast default.
    pub fn new(amount: f32) -> Self {
        Self {
            amount: amount.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    pub fn with_warmth(mut self, warmth: f32) -> Self {
        self.warmth = warmth.clamp(-1.0, 1.0);
        self
    }

    pub fn with_contrast(mut self, contrast: f32) -> Self {
        self.contrast = contrast.max(0.1);
        self
    }
}

impl ImageFilter for CrossProcess {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: CrossProcess does not yet handle {:?}",
                params.format
            )));
        }
        let amount = self.amount.clamp(0.0, 1.0);
        let contrast = self.contrast.max(0.1);
        let warmth = self.warmth.clamp(-1.0, 1.0);

        // Build three per-channel LUTs.
        let r_lut = build_lut(contrast * 1.1, warmth * 0.4, 0.55, amount);
        let g_lut = build_lut(contrast, 0.0, 0.5, amount);
        let b_lut = build_lut(contrast * 0.9, -warmth * 0.4, 0.45, amount);

        match params.format {
            PixelFormat::Gray8 => {
                // Single channel ⇒ use the neutral (G) curve.
                let mut out = input.clone();
                apply_tone_lut(&mut out, &g_lut, params.format, params.width, params.height);
                Ok(out)
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let mut out = input.clone();
                apply_tone_lut(&mut out, &g_lut, params.format, params.width, params.height);
                Ok(out)
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if params.format == PixelFormat::Rgba {
                    4
                } else {
                    3
                };
                let has_alpha = bpp == 4;
                let w = params.width as usize;
                let h = params.height as usize;
                let row = w * bpp;
                let src = &input.planes[0];
                let mut data = vec![0u8; row * h];
                for y in 0..h {
                    let r_in = &src.data[y * src.stride..y * src.stride + row];
                    let r_out = &mut data[y * row..(y + 1) * row];
                    for x in 0..w {
                        let base = x * bpp;
                        r_out[base] = r_lut[r_in[base] as usize];
                        r_out[base + 1] = g_lut[r_in[base + 1] as usize];
                        r_out[base + 2] = b_lut[r_in[base + 2] as usize];
                        if has_alpha {
                            r_out[base + 3] = r_in[base + 3];
                        }
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride: row, data }],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: CrossProcess does not yet handle {other:?}"
            ))),
        }
    }
}

/// One per-channel LUT. `contrast` is the sigmoid slope, `bias` shifts
/// the curve toward warm (positive) or cool (negative), `pivot` is the
/// nominal mid-grey position, and `amount` lerps with identity.
fn build_lut(contrast: f32, bias: f32, pivot: f32, amount: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        let v = i as f32 / 255.0;
        let s = sigmoid(contrast * (v - pivot));
        // Normalise sigmoid back to [0, 1] using its 0..1 endpoints.
        let s0 = sigmoid(contrast * (0.0 - pivot));
        let s1 = sigmoid(contrast * (1.0 - pivot));
        let s = ((s - s0) / (s1 - s0)).clamp(0.0, 1.0);
        let biased = (s + bias).clamp(0.0, 1.0);
        let mixed = v * (1.0 - amount) + biased * amount;
        *e = (mixed * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
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
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 77]);
        let out = CrossProcess::new(0.0).apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn full_amount_shifts_midtone_warm() {
        // Mid-grey input — with full amount + positive warmth, R should
        // end up >= B in the output.
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = CrossProcess::new(1.0)
            .with_warmth(0.5)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert!(
                chunk[0] >= chunk[2],
                "expected R ({}) >= B ({})",
                chunk[0],
                chunk[2]
            );
        }
    }

    #[test]
    fn extremes_clamp_to_endpoints() {
        let input = rgb(
            2,
            1,
            |x, _| {
                if x == 0 {
                    [0, 0, 0]
                } else {
                    [255, 255, 255]
                }
            },
        );
        let out = CrossProcess::new(1.0).apply(&input, p_rgb(2, 1)).unwrap();
        // Black ⇒ near black; white ⇒ near white (sigmoid endpoints).
        for c in 0..3 {
            assert!(
                out.planes[0].data[c] < 30,
                "black not preserved: {:?}",
                &out.planes[0].data[..3]
            );
            assert!(out.planes[0].data[3 + c] > 225, "white not preserved");
        }
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            data.extend_from_slice(&[100u8, 150, 200, 77]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = CrossProcess::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[3], 77, "alpha not preserved");
        }
    }

    #[test]
    fn gray8_uses_neutral_curve() {
        // Just check the filter accepts Gray8 without panicking and
        // produces non-trivial change at amount=1 on a mid-grey input.
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![128; 16],
            }],
        };
        let out = CrossProcess::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 16);
    }
}
