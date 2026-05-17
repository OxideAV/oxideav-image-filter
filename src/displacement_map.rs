//! Displacement map — two-input filter that uses the `dst` (map) image to
//! perturb the per-pixel sampling coordinates of the `src` image.
//!
//! For each output pixel `(x, y)` the map's R / G channels are read at
//! `(x, y)` and re-centred to `[-0.5, 0.5]`. The recovered vector is
//! scaled by `scale_x` / `scale_y` (in pixels) and added to the source
//! sampling coordinate; the source is then sampled with bilinear
//! interpolation:
//!
//! ```text
//! dx = (map.R[x, y] / 255 - 0.5) · scale_x
//! dy = (map.G[x, y] / 255 - 0.5) · scale_y
//! out[x, y] = bilinear(src, x + dx, y + dy)
//! ```
//!
//! Classic compositing primitive used for water-ripple, heat-haze,
//! glass-distortion and "look through frosted glass" effects.
//!
//! Both inputs must share `(width, height)` and `PixelFormat`. The map
//! is treated as RGB (its R/G channels carry the vector); a Gray8 map
//! uses the single channel as both X and Y displacement.
//!
//! Distinct from:
//! - [`Distort`](crate::Distort) — fixed-form parametric radial / barrel
//!   distortion.
//! - [`Perspective`](crate::Perspective) — projective homography.
//! - [`Spread`](crate::Spread) — random per-pixel displacement (no map).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba` on both `src` and `dst`. Other formats
//! return [`Error::unsupported`].

use crate::zoom_blur::bilinear_sample;
use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Displacement-map filter.
#[derive(Clone, Copy, Debug)]
pub struct DisplacementMap {
    /// Maximum X displacement in pixels (full-amplitude swing from `0` to
    /// `255` in the R channel of the map).
    pub scale_x: f32,
    /// Maximum Y displacement in pixels.
    pub scale_y: f32,
}

impl Default for DisplacementMap {
    fn default() -> Self {
        Self {
            scale_x: 10.0,
            scale_y: 10.0,
        }
    }
}

impl DisplacementMap {
    /// Build with the given scale (same value for X and Y).
    pub fn new(scale: f32) -> Self {
        Self {
            scale_x: scale,
            scale_y: scale,
        }
    }

    /// Override the X / Y scales independently.
    pub fn with_scales(mut self, scale_x: f32, scale_y: f32) -> Self {
        self.scale_x = scale_x;
        self.scale_y = scale_y;
        self
    }
}

impl TwoInputImageFilter for DisplacementMap {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: DisplacementMap requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if src.planes.is_empty() || dst.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DisplacementMap needs non-empty src/dst planes",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(src.clone());
        }
        let sp = &src.planes[0];
        let dp = &dst.planes[0];
        let row_stride = w * bpp;
        let mut data = vec![0u8; row_stride * h];

        for y in 0..h {
            for x in 0..w {
                let map_off = y * dp.stride + x * bpp;
                let (mr, mg) = match bpp {
                    1 => {
                        let v = dp.data[map_off] as f32;
                        (v, v)
                    }
                    _ => (dp.data[map_off] as f32, dp.data[map_off + 1] as f32),
                };
                let dx_norm = mr / 255.0 - 0.5;
                let dy_norm = mg / 255.0 - 0.5;
                let sx = x as f32 + 0.5 + dx_norm * self.scale_x;
                let sy = y as f32 + 0.5 + dy_norm * self.scale_y;
                let sample = bilinear_sample(sp, w, h, bpp, sx, sy);
                let dst_base = y * row_stride + x * bpp;
                for c in 0..bpp {
                    data[dst_base + c] = sample[c].round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(VideoFrame {
            pts: src.pts.or(dst.pts),
            planes: vec![VideoPlane {
                stride: row_stride,
                data,
            }],
        })
    }
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
    fn neutral_grey_map_is_identity() {
        let src = gray(8, 8, |x, y| ((x * 17 + y * 11) % 251) as u8);
        // Neutral grey 128 ⇒ (128/255 - 0.5) ≈ 0.00196 ⇒ near-zero shift.
        let map = gray(8, 8, |_, _| 128);
        let out = DisplacementMap::new(10.0)
            .apply_two(&src, &map, p_gray(8, 8))
            .unwrap();
        for i in 0..64 {
            let d = (out.planes[0].data[i] as i32 - src.planes[0].data[i] as i32).abs();
            assert!(d <= 5, "drift {d} at {i}");
        }
    }

    #[test]
    fn zero_map_pulls_from_upper_left() {
        // R = 0 ⇒ dx_norm = -0.5 ⇒ shift = -scale/2 (sample from left).
        let src = gray(8, 8, |x, _| (x * 30) as u8);
        let map = gray(8, 8, |_, _| 0);
        let out = DisplacementMap::new(4.0)
            .apply_two(&src, &map, p_gray(8, 8))
            .unwrap();
        // Each output pixel should come from `x - 2` (clamped).
        for x in 2..8 {
            let expected = src.planes[0].data[x - 2];
            let got = out.planes[0].data[x];
            // Bilinear gives an exact integer when fractional part is 0.
            assert!(
                (got as i32 - expected as i32).abs() <= 1,
                "x={x} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn constant_src_is_fixed_point() {
        let src = gray(8, 8, |_, _| 80);
        let map = gray(8, 8, |x, _| (x * 30) as u8);
        let out = DisplacementMap::new(8.0)
            .apply_two(&src, &map, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 80);
        }
    }

    #[test]
    fn rejects_unsupported() {
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![0; 4],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 2,
            height: 2,
        };
        let err = DisplacementMap::default()
            .apply_two(&src, &src, p)
            .unwrap_err();
        assert!(format!("{err}").contains("DisplacementMap"));
    }
}
