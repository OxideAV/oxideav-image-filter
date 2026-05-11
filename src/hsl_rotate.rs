//! Rotate the hue channel by N degrees in HSL space.
//!
//! Clean-room implementation of the textbook RGB ⇄ HSL round-trip:
//!
//! 1. Per pixel, normalise `(R, G, B)` into `[0, 1]` and compute
//!    `max`, `min`, `L = (max + min) / 2`.
//! 2. If `max == min` the pixel is achromatic and is passed through
//!    unchanged (saturation = 0, hue undefined).
//! 3. Otherwise pick `S` from the standard piecewise formula
//!    (`d / (max + min)` when `L < 0.5`, else `d / (2 - max - min)`)
//!    and `H` from the per-major-channel cases:
//!    - `R` max: `H = ((G - B) / d) mod 6`
//!    - `G` max: `H = (B - R) / d + 2`
//!    - `B` max: `H = (R - G) / d + 4`
//!
//!    Multiply by 60 to land in `[0, 360)` degrees.
//! 4. Rotate `H' = (H + degrees) mod 360`.
//! 5. Convert `(H', S, L)` back to RGB via the standard
//!    `temp1 = L < 0.5 ? L * (1 + S) : L + S - L * S`, `temp2 = 2L - t1`
//!    machinery + the per-channel `tc` evaluator.
//!
//! Alpha is preserved on `Rgba`. Only `Rgb24` / `Rgba` are supported —
//! YUV inputs return `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// HSL hue-rotation filter.
#[derive(Clone, Copy, Debug)]
pub struct HslRotate {
    /// Hue rotation in degrees. Positive rotates from red → yellow →
    /// green → cyan → blue → magenta. Non-finite values coerce to `0`.
    /// The rotation is taken modulo 360 internally.
    pub degrees: f32,
}

impl Default for HslRotate {
    fn default() -> Self {
        Self { degrees: 0.0 }
    }
}

impl HslRotate {
    /// Build a hue-rotation filter with explicit angle.
    pub fn new(degrees: f32) -> Self {
        Self {
            degrees: if degrees.is_finite() { degrees } else { 0.0 },
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
        // Match the canonical "mod 6" convention without using the
        // negative-aware `rem_euclid` directly on a float that may sit
        // on a six-boundary (we want exactly 0 for the wrap, not 6).
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

impl ImageFilter for HslRotate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: HslRotate requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: HslRotate requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let src = &input.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        let rot = self.degrees;
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32 / 255.0;
                let g = src_row[base + 1] as f32 / 255.0;
                let b = src_row[base + 2] as f32 / 255.0;
                let (hh, ss, ll) = rgb_to_hsl(r, g, b);
                let (nr, ng, nb) = hsl_to_rgb(hh + rot, ss, ll);
                dst_row[base] = (nr * 255.0).round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = (ng * 255.0).round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = (nb * 255.0).round().clamp(0.0, 255.0) as u8;
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

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

    #[test]
    fn zero_rotation_is_passthrough_within_rounding() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = HslRotate::new(0.0).apply(&input, p_rgb(4, 4)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1, "delta too large");
        }
    }

    #[test]
    fn rotate_180_maps_pure_red_to_cyan_band() {
        // Pure red is (255, 0, 0). 180° hue rotation should approach cyan.
        let input = rgb(2, 2, |_, _| [255, 0, 0]);
        let out = HslRotate::new(180.0).apply(&input, p_rgb(2, 2)).unwrap();
        let r = out.planes[0].data[0];
        let g = out.planes[0].data[1];
        let b = out.planes[0].data[2];
        assert!(
            r < 10,
            "R should be near 0 after 180° rotate of pure red, got {r}"
        );
        assert!(g > 200, "G should be high (cyan), got {g}");
        assert!(b > 200, "B should be high (cyan), got {b}");
    }

    #[test]
    fn achromatic_grey_unchanged() {
        // Pure grey (R==G==B) has S = 0; hue rotation should keep RGB.
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = HslRotate::new(90.0).apply(&input, p_rgb(2, 2)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 128);
            assert_eq!(chunk[1], 128);
            assert_eq!(chunk[2], 128);
        }
    }

    #[test]
    fn shape_preserved() {
        let input = rgb(5, 3, |_, _| [200, 50, 100]);
        let out = HslRotate::new(60.0).apply(&input, p_rgb(5, 3)).unwrap();
        assert_eq!(out.planes[0].stride, 15);
        assert_eq!(out.planes[0].data.len(), 45);
    }

    #[test]
    fn alpha_passes_through_on_rgba() {
        let data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 222]).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = HslRotate::new(120.0)
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
            assert_eq!(chunk[3], 222);
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
        let err = HslRotate::new(45.0)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("HslRotate"));
    }
}
