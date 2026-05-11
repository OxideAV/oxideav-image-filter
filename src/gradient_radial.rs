//! Radial gradient generator: paint each output pixel based on its
//! distance from a centre point, blending between an inner colour and
//! an outer colour with a smooth `0..=1` cosine-like ramp.
//!
//! Output shape (`format`, `width`, `height`) is taken from the stream
//! parameters — like [`Canvas`](crate::canvas::Canvas) the input frame
//! is consumed only for its `pts`. ImageMagick analogue:
//! `radial-gradient:colour1-colour2`.
//!
//! Clean-room: for each pixel `(x, y)` compute
//! `r = hypot(x - cx, y - cy) / radius`, clamp to `[0, 1]`, and linearly
//! interpolate between the inner and outer colours per channel. We use
//! plain linear blend (not gamma-aware) — matches IM's default.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`. YUV formats return
//! `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Radial-gradient filter.
#[derive(Clone, Copy, Debug)]
pub struct GradientRadial {
    /// Normalised centre `x` in `[0, 1]` (0 = left edge, 1 = right edge).
    pub centre_x: f32,
    /// Normalised centre `y` in `[0, 1]` (0 = top edge, 1 = bottom edge).
    pub centre_y: f32,
    /// Outer radius in pixels at which `t == 1`. If non-positive, falls
    /// back to `hypot(width, height) / 2` (the half-diagonal).
    pub radius: f32,
    /// Colour at the centre (`t == 0`).
    pub inner: [u8; 4],
    /// Colour at / beyond the outer radius (`t == 1`).
    pub outer: [u8; 4],
}

impl Default for GradientRadial {
    fn default() -> Self {
        Self {
            centre_x: 0.5,
            centre_y: 0.5,
            radius: 0.0,
            inner: [255, 255, 255, 255],
            outer: [0, 0, 0, 255],
        }
    }
}

impl GradientRadial {
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

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let v = a as f32 + (b as f32 - a as f32) * t;
    v.round().clamp(0.0, 255.0) as u8
}

impl ImageFilter for GradientRadial {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: GradientRadial requires non-zero width/height",
            ));
        }
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: GradientRadial requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let cx = self.centre_x * (w - 1) as f32;
        let cy = self.centre_y * (h - 1) as f32;
        let r = if self.radius > 0.0 {
            self.radius
        } else {
            ((w as f32).hypot(h as f32)) * 0.5
        };
        let stride = w * bpp;
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let t = (dx.hypot(dy) / r).clamp(0.0, 1.0);
                let base = y * stride + x * bpp;
                match bpp {
                    1 => {
                        data[base] = lerp_u8(self.inner[0], self.outer[0], t);
                    }
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
    fn centre_pixel_is_inner_colour() {
        let f = GradientRadial::new([255, 255, 255, 255], [0, 0, 0, 255]);
        let out = f.apply(&empty_rgba(5, 5), p_rgba(5, 5)).unwrap();
        // (2, 2) is the exact centre on a 5×5 grid.
        let base = 2 * 20 + 2 * 4;
        assert_eq!(out.planes[0].data[base], 255);
        assert_eq!(out.planes[0].data[base + 1], 255);
        assert_eq!(out.planes[0].data[base + 2], 255);
    }

    #[test]
    fn corner_is_close_to_outer_colour() {
        // With default radius (half-diagonal) the corners sit very near
        // t = 1 ⇒ should be near the outer colour.
        let f = GradientRadial::new([255, 255, 255, 255], [0, 0, 0, 255]);
        let out = f.apply(&empty_rgba(7, 7), p_rgba(7, 7)).unwrap();
        let v = out.planes[0].data[0]; // (0, 0)
        assert!(v < 60, "corner sample should approach outer (0), got {v}");
    }

    #[test]
    fn explicit_radius_clamps_outside() {
        let f = GradientRadial::new([255, 0, 0, 255], [0, 255, 0, 255]).with_radius(1.0);
        let out = f.apply(&empty_rgba(6, 1), p_rgba(6, 1)).unwrap();
        // (5, 0) is well beyond r = 1 so t = 1 ⇒ outer colour exactly.
        let base = 5 * 4;
        assert_eq!(out.planes[0].data[base], 0);
        assert_eq!(out.planes[0].data[base + 1], 255);
        assert_eq!(out.planes[0].data[base + 2], 0);
    }

    #[test]
    fn rejects_yuv() {
        let f = GradientRadial::default();
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
        assert!(format!("{err}").contains("GradientRadial"));
    }

    #[test]
    fn shape_preserved_gray8() {
        let f = GradientRadial::new([200, 0, 0, 0], [10, 0, 0, 0]);
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![0; 32],
            }],
        };
        let out = f
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].stride, 8);
        assert_eq!(out.planes[0].data.len(), 32);
    }
}
