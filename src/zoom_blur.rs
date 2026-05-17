//! Zoom blur — radial outward "rocket exhaust" style blur from a centre
//! point.
//!
//! For each output pixel a small number of samples are taken along the
//! line connecting the pixel to a configurable centre. Sample positions
//! shrink toward the centre in geometric steps `(1 - t · strength)`, so
//! the rendered streaks always point outward from `centre`. Averaging
//! the sample values produces the classic "warp drive" radial smear
//! found in motion-streak compositing.
//!
//! Algorithm summary (clean-room from the textbook radial-blur
//! formulation):
//!
//! ```text
//! For each pixel (x, y):
//!   v  = (x - cx, y - cy)
//!   sum = 0
//!   for s in 0..samples:
//!       t = s / (samples - 1)            ∈ [0, 1]
//!       sample = (cx, cy) + v · (1 - t · strength)
//!       sum += bilinear(sample)
//!   out = sum / samples
//! ```
//!
//! Bilinear sampling, edge-clamped at the border. Alpha on RGBA is
//! blurred along with the colour channels (matching what a focus-sweep
//! pull would render through a translucent subject).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Zoom-blur filter.
#[derive(Clone, Copy, Debug)]
pub struct ZoomBlur {
    /// Centre (normalised image coordinates): `(0.5, 0.5)` is the
    /// frame centre.
    pub centre_x: f32,
    /// Centre y in `[0, 1]`.
    pub centre_y: f32,
    /// Strength in `[0, 1]`. `0` is identity, `1` pulls samples all the
    /// way back to the centre (heavy streak).
    pub strength: f32,
    /// Number of samples along the radial line (≥ 2). Higher values
    /// look smoother but cost linearly more.
    pub samples: u32,
}

impl Default for ZoomBlur {
    fn default() -> Self {
        Self {
            centre_x: 0.5,
            centre_y: 0.5,
            strength: 0.2,
            samples: 8,
        }
    }
}

impl ZoomBlur {
    /// Build with the given strength (`[0, 1]` clamp); centre defaults
    /// to the image centre and `samples` defaults to 8.
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    /// Override the centre (normalised `[0, 1]`).
    pub fn with_centre(mut self, x: f32, y: f32) -> Self {
        self.centre_x = x;
        self.centre_y = y;
        self
    }

    /// Override the sample count (clamped to ≥ 2).
    pub fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples.max(2);
        self
    }
}

impl ImageFilter for ZoomBlur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ZoomBlur requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];
        let row_stride = w * bpp;
        let mut data = vec![0u8; row_stride * h];
        let cx = self.centre_x * params.width as f32;
        let cy = self.centre_y * params.height as f32;
        let samples = self.samples.max(2) as usize;
        let strength = self.strength.clamp(0.0, 1.0);

        for y in 0..h {
            for x in 0..w {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let vx = px - cx;
                let vy = py - cy;
                let mut acc = [0f32; 4];
                for s in 0..samples {
                    let t = s as f32 / (samples - 1) as f32;
                    let k = 1.0 - t * strength;
                    let sx = cx + vx * k;
                    let sy = cy + vy * k;
                    let sample = bilinear_sample(src, w, h, bpp, sx, sy);
                    for c in 0..bpp {
                        acc[c] += sample[c];
                    }
                }
                let dst_base = y * row_stride + x * bpp;
                for c in 0..bpp {
                    data[dst_base + c] = (acc[c] / samples as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_stride,
                data,
            }],
        })
    }
}

/// Bilinear sample of `src` at fractional `(x, y)`. Centre of the
/// `(0, 0)` pixel is `(0.5, 0.5)`. Out-of-bounds samples clamp to the
/// nearest edge pixel.
pub(crate) fn bilinear_sample(
    src: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    x: f32,
    y: f32,
) -> [f32; 4] {
    let xf = x - 0.5;
    let yf = y - 0.5;
    let x0 = (xf.floor() as i32).clamp(0, w as i32 - 1) as usize;
    let y0 = (yf.floor() as i32).clamp(0, h as i32 - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = (xf - x0 as f32).clamp(0.0, 1.0);
    let dy = (yf - y0 as f32).clamp(0.0, 1.0);

    let mut out = [0f32; 4];
    for (c, slot) in out.iter_mut().enumerate().take(bpp) {
        let a = src.data[y0 * src.stride + x0 * bpp + c] as f32;
        let b = src.data[y0 * src.stride + x1 * bpp + c] as f32;
        let cc = src.data[y1 * src.stride + x0 * bpp + c] as f32;
        let d = src.data[y1 * src.stride + x1 * bpp + c] as f32;
        let top = a * (1.0 - dx) + b * dx;
        let bot = cc * (1.0 - dx) + d * dx;
        *slot = top * (1.0 - dy) + bot * dy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn zero_strength_is_identity() {
        let input = gray(16, 16, |x, y| ((x * 17 + y * 23) % 251) as u8);
        let out = ZoomBlur::new(0.0).apply(&input, p_gray(16, 16)).unwrap();
        // Bilinear sampling at half-pixel offsets is exact for integer
        // coordinates here because vx/vy * 1.0 = original delta and the
        // sample falls back on the exact pixel grid.
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn constant_image_is_fixed_point() {
        let input = gray(16, 16, |_, _| 200);
        let out = ZoomBlur::new(0.5).apply(&input, p_gray(16, 16)).unwrap();
        for &v in &out.planes[0].data {
            // Allow ±1 for bilinear rounding.
            assert!((v as i32 - 200).abs() <= 1, "got {v}");
        }
    }

    #[test]
    fn centre_pixel_unchanged() {
        // At (cx, cy) the displacement vector is (0,0) so every sample
        // is the centre pixel — output should equal that pixel.
        let input = gray(15, 15, |x, y| (x as i32 * 10 + y as i32) as u8);
        let out = ZoomBlur::new(0.7).apply(&input, p_gray(15, 15)).unwrap();
        let centre = out.planes[0].data[7 * 15 + 7];
        let expected = input.planes[0].data[7 * 15 + 7];
        assert!(
            (centre as i32 - expected as i32).abs() <= 2,
            "centre drift: got {centre} expected {expected}"
        );
    }

    #[test]
    fn samples_min_two() {
        assert_eq!(ZoomBlur::default().with_samples(0).samples, 2);
        assert_eq!(ZoomBlur::default().with_samples(1).samples, 2);
    }

    #[test]
    fn rgb_works() {
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8 {
            for x in 0..8 {
                data.extend_from_slice(&[(x * 32) as u8, (y * 32) as u8, 50]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 8,
            height: 8,
        };
        let out = ZoomBlur::new(0.4).apply(&input, p).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
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
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = ZoomBlur::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("ZoomBlur"));
    }
}
