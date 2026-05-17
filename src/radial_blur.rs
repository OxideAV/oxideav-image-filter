//! Radial (spin) blur — rotational smear around a centre point.
//!
//! For each output pixel a small number of samples are taken along the
//! *arc* swept by rotating that pixel around a configurable centre by
//! small angle increments in `[-angle/2, +angle/2]`. Averaging the
//! sampled colours produces the classic "spinning wheel" / propeller
//! motion-blur look used in compositing software.
//!
//! Algorithm summary (clean-room from the textbook angular-blur
//! formulation):
//!
//! ```text
//! For each pixel (x, y):
//!   r       = |(x, y) - centre|, base_angle = atan2(y-cy, x-cx)
//!   sum     = 0
//!   for s in 0..samples:
//!       t        = s / (samples - 1)             ∈ [0, 1]
//!       offset   = (t - 0.5) · angle             (radians)
//!       sample_a = base_angle + offset
//!       sx, sy   = centre + r · (cos a, sin a)
//!       sum     += bilinear(sx, sy)
//!   out = sum / samples
//! ```
//!
//! Bilinear sampling, edge-clamped at the border. Distinct from
//! [`ZoomBlur`](crate::ZoomBlur) (which moves samples radially toward
//! the centre instead of around it).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Other formats return
//! [`Error::unsupported`].

use crate::zoom_blur::bilinear_sample;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Radial / spin-blur filter.
#[derive(Clone, Copy, Debug)]
pub struct RadialBlur {
    /// Centre x in `[0, 1]` (normalised image coords).
    pub centre_x: f32,
    /// Centre y in `[0, 1]`.
    pub centre_y: f32,
    /// Total swept angle (degrees). The samples span
    /// `[-angle/2, +angle/2]` around each pixel's base angle.
    pub angle_degrees: f32,
    /// Number of samples along the arc (≥ 2).
    pub samples: u32,
}

impl Default for RadialBlur {
    fn default() -> Self {
        Self {
            centre_x: 0.5,
            centre_y: 0.5,
            angle_degrees: 10.0,
            samples: 8,
        }
    }
}

impl RadialBlur {
    /// Build with the given total swept angle (degrees). Centre
    /// defaults to the image centre, samples to 8.
    pub fn new(angle_degrees: f32) -> Self {
        Self {
            angle_degrees,
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

impl ImageFilter for RadialBlur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: RadialBlur requires Gray8/Rgb24/Rgba, got {other:?}"
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
        let total_rad = self.angle_degrees.to_radians();

        for y in 0..h {
            for x in 0..w {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let dx = px - cx;
                let dy = py - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let base_a = dy.atan2(dx);
                let mut acc = [0f32; 4];
                for s in 0..samples {
                    let t = s as f32 / (samples - 1) as f32;
                    let off = (t - 0.5) * total_rad;
                    let a = base_a + off;
                    let sx = cx + r * a.cos();
                    let sy = cy + r * a.sin();
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
    fn zero_angle_is_identity() {
        let input = gray(16, 16, |x, y| ((x * 13 + y * 7) % 200) as u8);
        let out = RadialBlur::new(0.0).apply(&input, p_gray(16, 16)).unwrap();
        // At angle=0 every sample is the same point — bilinear of an
        // exact integer-pixel position should equal the source within
        // rounding.
        for i in 0..256 {
            let d = (out.planes[0].data[i] as i32 - input.planes[0].data[i] as i32).abs();
            assert!(d <= 1, "drift {d} at {i}");
        }
    }

    #[test]
    fn constant_image_is_fixed_point() {
        let input = gray(16, 16, |_, _| 200);
        let out = RadialBlur::new(20.0).apply(&input, p_gray(16, 16)).unwrap();
        for &v in &out.planes[0].data {
            assert!((v as i32 - 200).abs() <= 1, "got {v}");
        }
    }

    #[test]
    fn centre_pixel_unchanged() {
        // Radius 0 ⇒ arc collapses to centre.
        let input = gray(15, 15, |x, y| (x as i32 * 10 + y as i32) as u8);
        let out = RadialBlur::new(45.0).apply(&input, p_gray(15, 15)).unwrap();
        let centre = out.planes[0].data[7 * 15 + 7];
        let expected = input.planes[0].data[7 * 15 + 7];
        assert!(
            (centre as i32 - expected as i32).abs() <= 2,
            "centre drift: got {centre} expected {expected}"
        );
    }

    #[test]
    fn samples_min_two() {
        assert_eq!(RadialBlur::default().with_samples(0).samples, 2);
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
        let err = RadialBlur::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("RadialBlur"));
    }
}
