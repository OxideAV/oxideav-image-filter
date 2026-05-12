//! Saturation boost that preserves already-saturated colours.
//!
//! Where [`Modulate`](crate::modulate::Modulate) multiplies the
//! saturation channel by a constant — driving all colours toward (or
//! away from) full saturation uniformly — `Vibrance` scales the boost
//! per-pixel by `1 - current_saturation`. The net effect:
//!
//! - Already-vivid pixels (`s ≈ 1`) move very little.
//! - Drab / pastel pixels (`s ≈ 0`) take the full boost.
//!
//! This is the Lightroom-style "Vibrance" slider — visually distinct from
//! a flat saturation crank because it skips the look of over-saturated
//! skin tones / sky highlights.
//!
//! Algorithm (per pixel, on `Rgb24` / `Rgba`):
//!
//! 1. Convert `(R, G, B)` → `(H, S, L)` in HSL.
//! 2. `weight = 1 - s` so already-saturated pixels are spared.
//! 3. `s' = clamp(s + amount · weight, 0, 1)` (or `s' = clamp(s · (1 +
//!    amount · weight), 0, 1)` for negative-`amount` desaturation; the
//!    additive form is biased toward the boost case).
//! 4. Convert `(H, s', L)` back to RGB.
//!
//! Operates on `Rgb24` / `Rgba`. YUV inputs return `Unsupported`. Alpha
//! is preserved.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Saturation booster that spares already-saturated pixels.
#[derive(Clone, Copy, Debug)]
pub struct Vibrance {
    /// Strength of the boost in `[-1, 1]`. Positive values increase
    /// saturation of drab pixels; negative values desaturate (mirroring
    /// the slider semantics).
    pub amount: f32,
}

impl Default for Vibrance {
    fn default() -> Self {
        Self { amount: 0.0 }
    }
}

impl Vibrance {
    pub fn new(amount: f32) -> Self {
        Self {
            amount: if amount.is_finite() {
                amount.clamp(-1.0, 1.0)
            } else {
                0.0
            },
        }
    }
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l < 0.5 {
        d / (max + min)
    } else {
        d / (2.0 - max - min)
    };
    let h = if (max - r).abs() < 1e-6 {
        let t = (g - b) / d;
        if t < 0.0 {
            t + 6.0
        } else {
            t
        }
    } else if (max - g).abs() < 1e-6 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s, l)
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let mut tt = t;
    if tt < 0.0 {
        tt += 1.0;
    }
    if tt > 1.0 {
        tt -= 1.0;
    }
    if tt < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * tt;
    }
    if tt < 0.5 {
        return q;
    }
    if tt < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - tt) * 6.0;
    }
    p
}

fn hsl_to_rgb(h_deg: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s <= 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h = h_deg.rem_euclid(360.0) / 360.0;
    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    (r, g, b)
}

impl ImageFilter for Vibrance {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Vibrance requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Vibrance requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let stride = w * bpp;
        let src = &input.planes[0];
        let mut out = vec![0u8; stride * h];
        let amount = self.amount;
        for y in 0..h {
            let s_row = &src.data[y * src.stride..y * src.stride + stride];
            let d_row = &mut out[y * stride..(y + 1) * stride];
            for x in 0..w {
                let base = x * bpp;
                let r = s_row[base] as f32 / 255.0;
                let g = s_row[base + 1] as f32 / 255.0;
                let b = s_row[base + 2] as f32 / 255.0;
                let (hh, ss, ll) = rgb_to_hsl(r, g, b);
                // Per-pixel weighting: drab pixels (low s) get full boost.
                let weight = 1.0 - ss;
                let ss_new = (ss + amount * weight).clamp(0.0, 1.0);
                let (nr, ng, nb) = hsl_to_rgb(hh, ss_new, ll);
                d_row[base] = (nr * 255.0).round().clamp(0.0, 255.0) as u8;
                d_row[base + 1] = (ng * 255.0).round().clamp(0.0, 255.0) as u8;
                d_row[base + 2] = (nb * 255.0).round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    d_row[base + 3] = s_row[base + 3];
                }
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data: out }],
        })
    }
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
    fn zero_amount_round_trips() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = Vibrance::default().apply(&input, p_rgb(4, 4)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn already_saturated_red_is_spared() {
        // Pure red already at s=1 — should not move under positive amount.
        let input = rgb(2, 2, |_, _| [255, 0, 0]);
        let out = Vibrance::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let g = out.planes[0].data[1];
        let b = out.planes[0].data[2];
        assert!(r >= 254);
        assert!(g <= 1);
        assert!(b <= 1);
    }

    #[test]
    fn drab_pixel_gets_boost() {
        // (140, 100, 100) is slightly red — boost should make it more red.
        let input = rgb(2, 2, |_, _| [140, 100, 100]);
        let baseline = rgb(2, 2, |_, _| [140, 100, 100]);
        let out = Vibrance::new(0.5).apply(&input, p_rgb(2, 2)).unwrap();
        let r_in = baseline.planes[0].data[0];
        let g_in = baseline.planes[0].data[1];
        let r_out = out.planes[0].data[0];
        let g_out = out.planes[0].data[1];
        let delta_in = r_in as i16 - g_in as i16;
        let delta_out = r_out as i16 - g_out as i16;
        assert!(
            delta_out > delta_in,
            "drab colour must gain saturation: delta_in={delta_in} delta_out={delta_out}"
        );
    }

    #[test]
    fn negative_amount_desaturates() {
        let input = rgb(2, 2, |_, _| [200, 100, 100]);
        let out = Vibrance::new(-1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let g = out.planes[0].data[1];
        let b = out.planes[0].data[2];
        // Fully drained: R == G == B (within rounding).
        assert!((r as i16 - g as i16).abs() <= 1);
        assert!((g as i16 - b as i16).abs() <= 1);
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[200u8, 100, 100, 17]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Vibrance::new(0.5)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 17);
        }
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
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
        let err = Vibrance::new(0.3)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Vibrance"));
    }
}
