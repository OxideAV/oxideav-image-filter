//! Local-contrast boost ("clarity" in Lr / ACR terminology).
//!
//! Where [`Sharpen`](crate::sharpen::Sharpen) / [`Unsharp`](crate::unsharp::Unsharp)
//! use a small radius (single-digit pixels) to pop fine detail, "clarity"
//! is the same unsharp-mask algorithm dialled to a *large* radius (tens
//! of pixels) with a moderate amount. The net effect is a tonal "pop" of
//! mid-frequencies without altering the small-scale edge structure.
//!
//! Internally we delegate to [`Unsharp`] (so YUV-luma-only / RGB-all-planes
//! semantics match the rest of the unsharp family); the `Clarity` filter
//! is essentially a preset with a sensible default radius / sigma /
//! amount / threshold:
//!
//! - `radius = 30`, `sigma = 15.0` — picks up mid-tone tonal regions.
//! - `amount = 0.5` — gentle boost (a full `1.0` is the unsharp-mask default).
//! - `threshold = 10` — drops floor noise on smooth gradients.
//!
//! All four parameters are exposed so callers can override them. ImageMagick
//! analogue: `-unsharp 30x15+0.5+10`.

use crate::{ImageFilter, Unsharp, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Mid-frequency local-contrast boost ("clarity").
#[derive(Clone, Debug)]
pub struct Clarity {
    /// Blur radius for the unsharp pass (large = mid-frequency emphasis).
    pub radius: u32,
    /// Blur sigma; defaults to `radius / 2.0` if non-positive.
    pub sigma: f32,
    /// Mask strength (0 = passthrough, 1.0 = full unsharp).
    pub amount: f32,
    /// Skip pixels with `|orig - blur| < threshold` to spare smooth regions.
    pub threshold: u8,
}

impl Default for Clarity {
    fn default() -> Self {
        Self {
            radius: 30,
            sigma: 15.0,
            amount: 0.5,
            threshold: 10,
        }
    }
}

impl Clarity {
    pub fn new(amount: f32) -> Self {
        Self {
            amount,
            ..Self::default()
        }
    }

    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }

    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }
}

impl ImageFilter for Clarity {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let sigma = if self.sigma > 0.0 {
            self.sigma
        } else {
            (self.radius as f32) / 2.0
        };
        let inner = Unsharp::new(self.radius, sigma, self.amount, self.threshold);
        inner.apply(input, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn rgb_grad(w: u32, h: u32) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) * 4).min(255) as u8;
                data.extend_from_slice(&[v, v, v]);
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
    fn zero_amount_is_identity_clone() {
        let input = rgb_grad(8, 8);
        let out = Clarity {
            amount: 0.0,
            ..Default::default()
        }
        .apply(&input, p_rgb(8, 8))
        .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn default_amount_produces_valid_frame() {
        let input = rgb_grad(16, 16);
        let out = Clarity::default().apply(&input, p_rgb(16, 16)).unwrap();
        assert_eq!(out.planes[0].data.len(), input.planes[0].data.len());
    }

    #[test]
    fn shape_is_preserved_on_rgb() {
        let input = rgb_grad(8, 8);
        let out = Clarity::new(0.7).apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].stride, input.planes[0].stride);
        assert_eq!(out.planes[0].data.len(), input.planes[0].data.len());
    }

    #[test]
    fn yuv_passes_through_chroma() {
        // Same policy as Unsharp: YUV touches only luma.
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 8,
                    data: vec![100; 64],
                },
                VideoPlane {
                    stride: 4,
                    data: vec![70; 16],
                },
                VideoPlane {
                    stride: 4,
                    data: vec![180; 16],
                },
            ],
        };
        let out = Clarity::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        // Chroma planes preserved bit-exact.
        assert_eq!(out.planes[1].data, frame.planes[1].data);
        assert_eq!(out.planes[2].data, frame.planes[2].data);
    }
}
