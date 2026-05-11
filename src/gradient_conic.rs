//! Conic (angular) gradient generator: paint each output pixel based
//! on the angle from a centre point, blending between an inner and an
//! outer colour as the angle sweeps from `start_angle` (degrees) round
//! to `start_angle + 360°`.
//!
//! Output shape comes from the stream parameters; the input frame is
//! consumed only for its `pts`. ImageMagick analogue:
//! `gradient:` with the `gradient-direction:angular` setting (no direct
//! one-liner — this is a textbook angular sweep).
//!
//! Clean-room: for each pixel `(x, y)` we compute
//! `θ = atan2(y - cy, x - cx)` in degrees, normalise to
//! `t = ((θ - start_angle) mod 360) / 360 ∈ [0, 1)`, and lerp
//! `inner` ⇒ `outer` per channel.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`. YUV returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Conic-gradient filter.
#[derive(Clone, Copy, Debug)]
pub struct GradientConic {
    pub centre_x: f32,
    pub centre_y: f32,
    /// Angle in degrees that maps to `t = 0` (the seam between
    /// `inner` and `outer`). 0° is the +x axis, 90° is the +y axis
    /// (downward in image coordinates).
    pub start_angle: f32,
    pub inner: [u8; 4],
    pub outer: [u8; 4],
}

impl Default for GradientConic {
    fn default() -> Self {
        Self {
            centre_x: 0.5,
            centre_y: 0.5,
            start_angle: 0.0,
            inner: [255, 255, 255, 255],
            outer: [0, 0, 0, 255],
        }
    }
}

impl GradientConic {
    pub fn new(inner: [u8; 4], outer: [u8; 4]) -> Self {
        Self {
            inner,
            outer,
            ..Default::default()
        }
    }

    pub fn with_centre(mut self, x: f32, y: f32) -> Self {
        self.centre_x = x;
        self.centre_y = y;
        self
    }

    pub fn with_start_angle(mut self, deg: f32) -> Self {
        self.start_angle = deg;
        self
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let v = a as f32 + (b as f32 - a as f32) * t;
    v.round().clamp(0.0, 255.0) as u8
}

impl ImageFilter for GradientConic {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: GradientConic requires non-zero width/height",
            ));
        }
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: GradientConic requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let cx = self.centre_x * (w - 1) as f32;
        let cy = self.centre_y * (h - 1) as f32;
        let stride = w * bpp;
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let theta_rad = (y as f32 - cy).atan2(x as f32 - cx);
                let theta_deg = theta_rad.to_degrees();
                let t = ((theta_deg - self.start_angle).rem_euclid(360.0)) / 360.0;
                let base = y * stride + x * bpp;
                match bpp {
                    1 => data[base] = lerp_u8(self.inner[0], self.outer[0], t),
                    3 => {
                        data[base] = lerp_u8(self.inner[0], self.outer[0], t);
                        data[base + 1] = lerp_u8(self.inner[1], self.outer[1], t);
                        data[base + 2] = lerp_u8(self.inner[2], self.outer[2], t);
                    }
                    4 => {
                        data[base] = lerp_u8(self.inner[0], self.outer[0], t);
                        data[base + 1] = lerp_u8(self.inner[1], self.outer[1], t);
                        data[base + 2] = lerp_u8(self.inner[2], self.outer[2], t);
                        data[base + 3] = lerp_u8(self.inner[3], self.outer[3], t);
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_rgba(w: u32, h: u32) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data: vec![0; (w * h * 4) as usize],
            }],
        }
    }

    fn p_rgba(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn opposite_angles_differ() {
        // With start_angle=0, the +x direction (right of centre) is t=0
        // ⇒ inner colour, and the -x direction is t=0.5 ⇒ midpoint.
        let f = GradientConic::new([255, 0, 0, 255], [0, 0, 255, 255]);
        let out = f.apply(&empty_rgba(7, 7), p_rgba(7, 7)).unwrap();
        // Pixel just right of centre (4, 3): theta near 0, t ≈ 0 ⇒ inner-ish.
        let r_right = out.planes[0].data[(3 * 7 + 4) * 4];
        // Pixel just left of centre (2, 3): theta = π, t = 0.5 ⇒ mid.
        let r_left = out.planes[0].data[(3 * 7 + 2) * 4];
        assert!(
            r_right > r_left,
            "right (R={r_right}) should be redder than left (R={r_left})"
        );
    }

    #[test]
    fn start_angle_rotates_seam() {
        // Rotate so that t=0 lands at +y (90°) instead of +x.
        let f = GradientConic::new([255, 0, 0, 255], [0, 0, 0, 255]).with_start_angle(90.0);
        let out = f.apply(&empty_rgba(5, 5), p_rgba(5, 5)).unwrap();
        // Pixel below centre (2, 4): theta = +π/2 = 90°, after subtracting
        // start_angle=90 ⇒ 0 ⇒ t ≈ 0 ⇒ inner red.
        let base = (4 * 5 + 2) * 4;
        let v = out.planes[0].data[base];
        assert!(v > 200, "below-centre pixel should be near-red, got {v}");
    }

    #[test]
    fn rejects_yuv() {
        let f = GradientConic::default();
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![0; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
            ],
        };
        let err = f
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("GradientConic"));
    }

    #[test]
    fn shape_preserved_rgb24() {
        let f = GradientConic::new([1, 2, 3, 0], [4, 5, 6, 0]);
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 6,
                data: vec![0; 12],
            }],
        };
        let out = f
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].stride, 6);
        assert_eq!(out.planes[0].data.len(), 12);
    }
}
