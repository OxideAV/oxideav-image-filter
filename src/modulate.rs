//! ImageMagick-style `-modulate B,S,H` — scales brightness/saturation
//! and rotates hue through HSL colour space.
//!
//! Clean-room implementation of the RGB↔HSL conversion following the
//! standard formulas from the HSL Wikipedia article. `B=S=100, H=0`
//! is identity.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Adjusts brightness, saturation, and hue of an RGB or RGBA frame.
///
/// - `brightness` (percent, 100 = identity) scales the lightness
///   channel. `0` produces black, `200` doubles lightness (clamped).
/// - `saturation` (percent, 100 = identity) scales the HSL saturation.
///   `0` produces greyscale; `200` doubles the distance from grey.
/// - `hue_degrees` rotates hue by that many degrees (`0 = identity`).
///
/// Maps to `convert -modulate B,S,H` in ImageMagick.
///
/// Pixel formats: `Rgb24`, `Rgba`. YUV inputs return
/// [`Error::unsupported`] because a round-trip to RGB isn't done here.
#[derive(Clone, Debug)]
pub struct Modulate {
    pub brightness: f32,
    pub saturation: f32,
    pub hue_degrees: f32,
}

impl Default for Modulate {
    fn default() -> Self {
        Self {
            brightness: 100.0,
            saturation: 100.0,
            hue_degrees: 0.0,
        }
    }
}

impl Modulate {
    /// Build with explicit brightness/saturation percentages (100 = no
    /// change) and hue rotation in degrees.
    pub fn new(brightness: f32, saturation: f32, hue_degrees: f32) -> Self {
        Self {
            brightness,
            saturation,
            hue_degrees,
        }
    }
}

impl ImageFilter for Modulate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Modulate requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        let b_scale = self.brightness / 100.0;
        let s_scale = self.saturation / 100.0;
        let hue_shift = self.hue_degrees / 360.0;

        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let r = src_row[x * bpp] as f32 / 255.0;
                let g = src_row[x * bpp + 1] as f32 / 255.0;
                let b = src_row[x * bpp + 2] as f32 / 255.0;

                let (mut hh, mut ss, mut ll) = rgb_to_hsl(r, g, b);
                ll = (ll * b_scale).clamp(0.0, 1.0);
                ss = (ss * s_scale).clamp(0.0, 1.0);
                hh = (hh + hue_shift).rem_euclid(1.0);
                let (r2, g2, b2) = hsl_to_rgb(hh, ss, ll);

                dst_row[x * bpp] = (r2 * 255.0).round().clamp(0.0, 255.0) as u8;
                dst_row[x * bpp + 1] = (g2 * 255.0).round().clamp(0.0, 255.0) as u8;
                dst_row[x * bpp + 2] = (b2 * 255.0).round().clamp(0.0, 255.0) as u8;
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

/// Convert RGB in [0,1] → HSL in [0,1]. `h` is fractional (0..1 = 0..360°).
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let d = max - min;
    if d == 0.0 {
        return (0.0, 0.0, l);
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        ((g - b) / d) + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        ((b - r) / d) + 2.0
    } else {
        ((r - g) / d) + 4.0
    };
    (h / 6.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = t.rem_euclid(1.0);
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

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
    fn identity_preserves_colours() {
        let input = rgb(4, 4, |x, y| ((x * 40) as u8, (y * 40) as u8, 128));
        let out = Modulate::default().apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 4, height: 4 }).unwrap();
        // Identity = very small rounding drift allowed.
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            let diff = (*a as i16 - *b as i16).abs();
            assert!(diff <= 1, "identity drift {diff}");
        }
    }

    #[test]
    fn brightness_zero_gives_black() {
        let input = rgb(2, 2, |_, _| (200, 100, 50));
        let out = Modulate::new(0.0, 100.0, 0.0).apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 2, height: 2 }).unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 0);
        }
    }

    #[test]
    fn saturation_zero_gives_gray() {
        // Pure red → grey when S=0.
        let input = rgb(2, 2, |_, _| (255, 0, 0));
        let out = Modulate::new(100.0, 0.0, 0.0).apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 2, height: 2 }).unwrap();
        // L for pure red = 0.5 → grey = 127 or 128.
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
            assert!((chunk[0] as i16 - 128).abs() <= 2, "gray = {}", chunk[0]);
        }
    }

    #[test]
    fn hue_rotation_120_swaps_red_to_green() {
        let input = rgb(2, 2, |_, _| (255, 0, 0));
        let out = Modulate::new(100.0, 100.0, 120.0).apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 2, height: 2 }).unwrap();
        // After +120° rotation, red should rotate to green.
        for chunk in out.planes[0].data.chunks(3) {
            assert!(chunk[0] < 5, "r = {}", chunk[0]);
            assert!(chunk[1] > 250, "g = {}", chunk[1]);
            assert!(chunk[2] < 5, "b = {}", chunk[2]);
        }
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
        let err = Modulate::default().apply(&input, VideoStreamParams { format: PixelFormat::Yuv420P, width: 4, height: 4 }).unwrap_err();
        assert!(format!("{err}").contains("Modulate"));
    }

    #[test]
    fn alpha_is_preserved() {
        let w = 2;
        let h = 2;
        let data: Vec<u8> = vec![
            200, 100, 50, 77, 50, 80, 120, 200, 10, 20, 30, 40, 0, 0, 0, 99,
        ];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        };
        let out = Modulate::new(120.0, 110.0, 10.0)
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Rgba, width: 2, height: 2 },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[3], 77);
        assert_eq!(out.planes[0].data[7], 200);
        assert_eq!(out.planes[0].data[11], 40);
        assert_eq!(out.planes[0].data[15], 99);
    }
}
