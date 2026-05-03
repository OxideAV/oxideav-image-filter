//! Explicit unsharp-mask with a per-pixel threshold gate.
//!
//! Formula: `diff = orig - blurred;`
//! If `|diff| < threshold` the original pixel is kept (suppresses
//! amplifying noise in smooth regions). Otherwise
//! `out = clamp(orig + amount*diff)`.
//!
//! This is the ImageMagick `-unsharp RxS+A+T` behaviour.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, plane_dims};
use crate::sharpen::unsharp_mask_plane;
use crate::{is_supported_format, Blur, ImageFilter, Planes, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Unsharp-mask with a contrast threshold.
///
/// See [`crate::Sharpen`] for a simpler variant that always applies
/// the mask.
#[derive(Clone, Debug)]
pub struct Unsharp {
    pub radius: u32,
    pub sigma: f32,
    pub amount: f32,
    pub threshold: u8,
}

impl Default for Unsharp {
    fn default() -> Self {
        Self {
            radius: 1,
            sigma: 0.5,
            amount: 1.0,
            threshold: 0,
        }
    }
}

impl Unsharp {
    /// Build with explicit radius, sigma, amount, and threshold.
    pub fn new(radius: u32, sigma: f32, amount: f32, threshold: u8) -> Self {
        Self {
            radius: radius.max(1),
            sigma: sigma.max(f32::EPSILON),
            amount,
            threshold,
        }
    }
}

impl ImageFilter for Unsharp {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Unsharp does not yet handle {:?}",
                params.format
            )));
        }

        // Match Sharpen's policy: YUV touches only luma.
        let planes_selector = match params.format {
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => Planes::Luma,
            _ => Planes::All,
        };

        let blurred = Blur::new(self.radius)
            .with_sigma(self.sigma)
            .with_planes(planes_selector)
            .apply(input, params)?;

        if self.amount.abs() < f32::EPSILON {
            return Ok(input.clone());
        }

        let (cx, cy) = chroma_subsampling(params.format);
        let mut out = input.clone();
        for (idx, plane) in out.planes.iter_mut().enumerate() {
            if !planes_selector.matches(idx, input.planes.len()) {
                continue;
            }
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            *plane = unsharp_mask_plane(
                &input.planes[idx],
                &blurred.planes[idx],
                pw as usize,
                ph as usize,
                bpp,
                self.amount,
                self.threshold,
            );
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
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

    #[test]
    fn high_threshold_suppresses_sharpening() {
        // Gentle gradient — contrast below threshold so unsharp is a no-op.
        let input = gray(8, 8, |x, _| (100 + x * 2) as u8);
        let out = Unsharp::new(2, 1.0, 2.0, 50)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn zero_threshold_matches_classic_unsharp() {
        let input = gray(8, 1, |x, _| if x < 4 { 60 } else { 200 });
        let out = Unsharp::new(2, 1.0, 1.5, 0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 1,
                },
            )
            .unwrap();
        // At the step, we should see enhancement.
        assert!(out.planes[0].data[4] >= 200);
        assert!(out.planes[0].data[3] <= 60);
    }

    #[test]
    fn threshold_gates_small_diffs_only() {
        // Sharp step: diff at edge is large, should sharpen; nearby flat
        // pixels should stay put.
        let input = gray(8, 1, |x, _| if x < 4 { 50 } else { 200 });
        let out = Unsharp::new(2, 1.0, 1.0, 20)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 1,
                },
            )
            .unwrap();
        // Far-left pixel has no meaningful diff → unchanged.
        assert_eq!(out.planes[0].data[0], 50);
        // Far-right likewise.
        assert_eq!(out.planes[0].data[7], 200);
    }

    #[test]
    fn amount_zero_is_identity() {
        let input = gray(8, 8, |x, y| ((x * 30 + y * 7) % 251) as u8);
        let out = Unsharp::new(2, 1.0, 0.0, 0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }
}
