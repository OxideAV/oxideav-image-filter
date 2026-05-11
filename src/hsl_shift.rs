//! Independent H / S / L shifts in HSL space — a richer cousin of
//! [`HslRotate`](crate::hsl_rotate::HslRotate) which only touches the
//! hue channel.
//!
//! Per pixel:
//!
//! 1. Convert `(R, G, B)` → `(H_deg, S, L)` (textbook HSL).
//! 2. Shift `H' = (H + h_shift) mod 360`.
//! 3. Add `s_shift` and `l_shift` to `S` and `L` and clamp into `[0, 1]`.
//! 4. Convert `(H', S', L')` back to `(R, G, B)`.
//!
//! Achromatic greys (R == G == B) preserve their hue=undefined semantics
//! — only the L shift affects them. Alpha is preserved on `Rgba`.
//!
//! Operates on `Rgb24` / `Rgba`. YUV inputs return `Unsupported`.
//! ImageMagick has no direct one-liner for this — `-modulate` mixes
//! `B,S,H` but on different scales (percentages of brightness, etc.);
//! this filter uses normalised additive shifts on the same `[0, 1]`
//! axis as `S`/`L`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Independent H / S / L shifts.
#[derive(Clone, Copy, Debug)]
pub struct HslShift {
    /// Hue shift in degrees, applied modulo 360. Non-finite ⇒ 0.
    pub h_shift: f32,
    /// Saturation shift in `[-1, 1]` (added to S, then clamped).
    pub s_shift: f32,
    /// Luminance shift in `[-1, 1]` (added to L, then clamped).
    pub l_shift: f32,
}

impl Default for HslShift {
    fn default() -> Self {
        Self {
            h_shift: 0.0,
            s_shift: 0.0,
            l_shift: 0.0,
        }
    }
}

impl HslShift {
    pub fn new(h: f32, s: f32, l: f32) -> Self {
        let f = |v: f32| if v.is_finite() { v } else { 0.0 };
        Self {
            h_shift: f(h),
            s_shift: f(s),
            l_shift: f(l),
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

impl ImageFilter for HslShift {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: HslShift requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: HslShift requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let src = &input.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let srow = &src.data[y * src.stride..y * src.stride + row_bytes];
            let drow = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = srow[base] as f32 / 255.0;
                let g = srow[base + 1] as f32 / 255.0;
                let b = srow[base + 2] as f32 / 255.0;
                let (hh, ss, ll) = rgb_to_hsl(r, g, b);
                let (nr, ng, nb) = hsl_to_rgb(
                    hh + self.h_shift,
                    (ss + self.s_shift).clamp(0.0, 1.0),
                    (ll + self.l_shift).clamp(0.0, 1.0),
                );
                drow[base] = (nr * 255.0).round().clamp(0.0, 255.0) as u8;
                drow[base + 1] = (ng * 255.0).round().clamp(0.0, 255.0) as u8;
                drow[base + 2] = (nb * 255.0).round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    drow[base + 3] = srow[base + 3];
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
    fn zero_shift_is_passthrough_within_rounding() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = HslShift::default().apply(&input, p_rgb(4, 4)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn negative_saturation_drains_colour() {
        // A pure red pixel with full saturation drop should come out grey.
        let input = rgb(2, 2, |_, _| [255, 0, 0]);
        let out = HslShift::new(0.0, -1.0, 0.0)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            // After full desaturation, R == G == B.
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
        }
    }

    #[test]
    fn positive_l_shift_brightens_grey() {
        // Mid-grey + L shift = +0.4 should lighten substantially.
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = HslShift::new(0.0, 0.0, 0.4)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let v = out.planes[0].data[0];
        assert!(v > 220, "lightness shift should brighten grey, got {v}");
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[200u8, 50, 50, 33]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = HslShift::new(60.0, 0.2, -0.1)
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
            assert_eq!(chunk[3], 33);
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
        let err = HslShift::new(45.0, 0.1, 0.1)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("HslShift"));
    }
}
