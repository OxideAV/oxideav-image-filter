//! Night-vision blue-shift effect (`+blue-shift`).
//!
//! Mimics the human visual system's "moonlight" / scotopic-vision
//! response: at low ambient brightness the eye's rods take over from
//! the cones and colour information collapses toward a desaturated
//! blue-grey. ImageMagick's `-blue-shift` applies a similar effect
//! by replacing each pixel with `(min(R, G, B), min(R, G, B), max(R,
//! G, B))` scaled by `factor`.
//!
//! Clean-room implementation derived from the documented behaviour:
//!
//! ```text
//! lo = min(R, G, B)
//! hi = max(R, G, B)
//! out_R = lo / factor
//! out_G = lo / factor
//! out_B = hi / factor
//! ```
//!
//! `factor = 1.0` is identity-ish (low blue band). `factor < 1.0` (the
//! ImageMagick default of `1.5`) brightens the blue channel relative to
//! the red / green and produces the characteristic moonlit tint.
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba`: full effect; alpha is preserved on RGBA.
//! - Other formats return `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Blue-shift / moonlight effect.
#[derive(Clone, Copy, Debug)]
pub struct BlueShift {
    /// Scale divisor applied to all three output channels. ImageMagick
    /// uses `1.5` by default (so `255 / 1.5 = 170` is the saturation
    /// ceiling on the blue channel). `1.0` is a no-scale moonlight tint;
    /// larger values darken further.
    pub factor: f32,
}

impl Default for BlueShift {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl BlueShift {
    /// Build a blue-shift filter with the given scale `factor`. Values
    /// `<= 0` are clamped to a tiny positive epsilon so the per-pixel
    /// division stays defined.
    pub fn new(factor: f32) -> Self {
        let f = if factor.is_finite() && factor > 0.0 {
            factor
        } else {
            1.0
        };
        Self { factor: f }
    }
}

impl ImageFilter for BlueShift {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: BlueShift requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];
        let f = self.factor.max(1e-6);
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base];
                let g = src_row[base + 1];
                let b = src_row[base + 2];
                let lo = r.min(g).min(b) as f32;
                let hi = r.max(g).max(b) as f32;
                let or = (lo / f).round().clamp(0.0, 255.0) as u8;
                let og = (lo / f).round().clamp(0.0, 255.0) as u8;
                let ob = (hi / f).round().clamp(0.0, 255.0) as u8;
                dst_row[base] = or;
                dst_row[base + 1] = og;
                dst_row[base + 2] = ob;
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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn pure_grey_stays_grey_at_factor_1() {
        // R = G = B = v ⇒ lo = hi = v ⇒ out = (v, v, v) when factor = 1.
        let input = rgb(4, 4, |x, _| {
            let v = (x as u8) * 30;
            [v, v, v]
        });
        let out = BlueShift::new(1.0).apply(&input, params_rgb(4, 4)).unwrap();
        for (i, chunk) in out.planes[0].data.chunks(3).enumerate() {
            let v = (i as u32 % 4) as u8 * 30;
            assert_eq!(chunk, &[v, v, v]);
        }
    }

    #[test]
    fn highlights_the_blue_channel() {
        // (200, 100, 50) ⇒ lo = 50, hi = 200. With factor = 1: out = (50, 50, 200).
        let input = rgb(2, 1, |_, _| [200, 100, 50]);
        let out = BlueShift::new(1.0).apply(&input, params_rgb(2, 1)).unwrap();
        assert_eq!(&out.planes[0].data[0..3], &[50, 50, 200]);
        assert_eq!(&out.planes[0].data[3..6], &[50, 50, 200]);
    }

    #[test]
    fn factor_scales_brightness() {
        // (200, 100, 50) with factor = 2 ⇒ (25, 25, 100).
        let input = rgb(1, 1, |_, _| [200, 100, 50]);
        let out = BlueShift::new(2.0).apply(&input, params_rgb(1, 1)).unwrap();
        assert_eq!(&out.planes[0].data[0..3], &[25, 25, 100]);
    }

    #[test]
    fn rgba_alpha_preserved() {
        let data: Vec<u8> = (0..16).flat_map(|i| [200u8, 100, 50, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = BlueShift::new(1.0)
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
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8);
            assert_eq!(out.planes[0].data[i * 4 + 2], 200);
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
        let err = BlueShift::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("BlueShift"));
    }

    #[test]
    fn non_finite_factor_falls_back_to_1() {
        let bs = BlueShift::new(f32::NAN);
        assert_eq!(bs.factor, 1.0);
        let bs2 = BlueShift::new(-3.0);
        assert_eq!(bs2.factor, 1.0);
    }
}
