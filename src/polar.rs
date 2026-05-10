//! Cartesian ⇄ polar coordinate distortion (ImageMagick `-distort Polar`
//! / `-distort DePolar`).
//!
//! The "polar" direction takes a rectangular Cartesian image and bends
//! it into an annular fan: the output's horizontal axis becomes the
//! polar **angle** (0 → 2π, sweeping clockwise from the vertical "up"
//! direction) and the vertical axis becomes the **radius** (0 at the
//! top, `max_r` at the bottom). The "depolar" direction unrolls the
//! disc back into a rectangle.
//!
//! ```text
//! Polar:  for each output (px, py):
//!     angle = (px / w) * 2π
//!     r     = (py / h) * max_r
//!     sx    = cx + r * sin(angle)
//!     sy    = cy - r * cos(angle)
//!     out   = bilinear_sample(in, sx, sy)
//!
//! DePolar: for each output (px, py):
//!     dx = px - cx; dy = py - cy
//!     r  = sqrt(dx*dx + dy*dy)
//!     a  = atan2(dx, -dy)            // angle from +Y axis, clockwise
//!     if a < 0: a += 2π
//!     sx = (a / 2π) * w
//!     sy = (r / max_r) * h
//!     out = bilinear_sample(in, sx, sy)
//! ```
//!
//! Centre defaults to `((w-1)/2, (h-1)/2)`. `max_r` defaults to the
//! distance from the centre to the farthest corner — covers the whole
//! image so `Polar→DePolar` round-trips reasonably (border resampling
//! still loses 1-pixel accuracy).
//!
//! Operates on packed `Rgb24` / `Rgba`. Output dimensions match the
//! input — no canvas resize.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Direction of the polar distortion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolarDirection {
    /// Cartesian → polar (image bent into a fan).
    #[default]
    Polar,
    /// Polar → Cartesian (fan unrolled back to a rectangle).
    DePolar,
}

/// Polar / Cartesian distortion.
#[derive(Clone, Copy, Debug, Default)]
pub struct Polar {
    pub direction: PolarDirection,
}

impl Polar {
    /// Build a Cartesian → polar distortion.
    pub fn forward() -> Self {
        Self {
            direction: PolarDirection::Polar,
        }
    }

    /// Build a polar → Cartesian distortion.
    pub fn inverse() -> Self {
        Self {
            direction: PolarDirection::DePolar,
        }
    }

    /// Override the distortion direction explicitly.
    pub fn with_direction(mut self, direction: PolarDirection) -> Self {
        self.direction = direction;
        self
    }
}

impl ImageFilter for Polar {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Polar requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];
        if w == 0 || h == 0 {
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        let cx = (w as f32 - 1.0) * 0.5;
        let cy = (h as f32 - 1.0) * 0.5;
        // Distance to the farthest corner — the smallest radius that
        // covers the whole input image.
        let max_r = ((w.max(1) as f32 - 1.0 - cx).hypot(h.max(1) as f32 - 1.0 - cy)).max(1.0);

        match self.direction {
            PolarDirection::Polar => {
                let two_pi = std::f32::consts::TAU;
                for py in 0..h {
                    let r = (py as f32 / h as f32) * max_r;
                    let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
                    for px in 0..w {
                        let angle = (px as f32 / w as f32) * two_pi;
                        let (sin_a, cos_a) = angle.sin_cos();
                        let sx = cx + r * sin_a;
                        let sy = cy - r * cos_a;
                        let base = px * bpp;
                        bilinear_sample_into(
                            &src.data,
                            src.stride,
                            w,
                            h,
                            bpp,
                            sx,
                            sy,
                            &mut dst_row[base..base + bpp],
                        );
                    }
                }
            }
            PolarDirection::DePolar => {
                let two_pi = std::f32::consts::TAU;
                let inv_two_pi = 1.0 / two_pi;
                for py in 0..h {
                    let dy = py as f32 - cy;
                    let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
                    for px in 0..w {
                        let dx = px as f32 - cx;
                        let r = (dx * dx + dy * dy).sqrt();
                        // atan2(dx, -dy) measures the clockwise angle
                        // from the +Y ("up") axis — matches the Polar
                        // direction's `(sin_a, -cos_a)` mapping.
                        let mut a = dx.atan2(-dy);
                        if a < 0.0 {
                            a += two_pi;
                        }
                        let sx = a * inv_two_pi * w as f32;
                        let sy = (r / max_r) * h as f32;
                        let base = px * bpp;
                        bilinear_sample_into(
                            &src.data,
                            src.stride,
                            w,
                            h,
                            bpp,
                            sx,
                            sy,
                            &mut dst_row[base..base + bpp],
                        );
                    }
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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flat_image_is_invariant_under_polar() {
        let input = rgb(16, 16, |_, _| (100, 150, 200));
        let out = Polar::forward().apply(&input, params_rgb(16, 16)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn flat_image_is_invariant_under_depolar() {
        let input = rgb(16, 16, |_, _| (33, 66, 99));
        let out = Polar::inverse().apply(&input, params_rgb(16, 16)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 33);
            assert_eq!(chunk[1], 66);
            assert_eq!(chunk[2], 99);
        }
    }

    #[test]
    fn polar_top_row_samples_centre() {
        // Polar maps row 0 → r = 0 ⇒ every output pixel of row 0 should
        // equal the centre pixel of the input.
        let input = rgb(16, 16, |x, y| (x as u8 * 16, y as u8 * 16, 50));
        let out = Polar::forward().apply(&input, params_rgb(16, 16)).unwrap();
        // Centre of 16×16 is (7.5, 7.5) ⇒ bilinear on (7,7)/(8,8). For
        // x≠y inputs this isn't a single pixel but every column of out's
        // row 0 should be identical (they all sample the same point).
        let r0 = &out.planes[0].data[0..3];
        for x in 0..16 {
            let off = x * 3;
            assert_eq!(&out.planes[0].data[off..off + 3], r0);
        }
    }

    #[test]
    fn polar_output_dims_match_input() {
        let input = rgb(8, 16, |_, _| (10, 20, 30));
        let out = Polar::forward().apply(&input, params_rgb(8, 16)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 16 * 3);
        assert_eq!(out.planes[0].stride, 24);
    }

    #[test]
    fn rgba_alpha_passes_through_for_uniform_alpha() {
        let data: Vec<u8> = (0..256).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 64, data }],
        };
        let out = Polar::forward()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for i in 0..256 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 222, "alpha at pixel {i}");
        }
    }

    #[test]
    fn depolar_actually_changes_a_radial_image() {
        // Build a radial-ish input: a bright ring at radius ~4. After
        // depolar, the bright ring should map to a horizontal stripe
        // (constant radius ⇒ constant output row).
        let mut input = rgb(16, 16, |_, _| (10, 10, 10));
        let cx = 7.5f32;
        let cy = 7.5f32;
        for y in 0..16 {
            for x in 0..16 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                if (r - 4.0).abs() < 0.7 {
                    let off = (y * 16 + x) * 3;
                    input.planes[0].data[off] = 200;
                    input.planes[0].data[off + 1] = 200;
                    input.planes[0].data[off + 2] = 200;
                }
            }
        }
        let out = Polar::inverse().apply(&input, params_rgb(16, 16)).unwrap();
        // Pick a row where the depolar(r) maps to ~4 / max_r ⇒
        // py ≈ 4 / max_r * 16. For 16×16, max_r ≈ √(7.5²+7.5²) = 10.6,
        // so the bright row should sit roughly at py = 6. The bright
        // pixels there should be brighter than at py = 0 (which samples
        // r ≈ 0 ⇒ the centre dim background).
        let bright_row_avg: u32 = (0..16)
            .map(|x| out.planes[0].data[6 * 48 + x * 3] as u32)
            .sum();
        let dark_row_avg: u32 = (0..16).map(|x| out.planes[0].data[x * 3] as u32).sum();
        assert!(
            bright_row_avg > dark_row_avg,
            "depolar didn't lift bright ring into a row: {bright_row_avg} vs {dark_row_avg}"
        );
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
        let err = Polar::forward()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Polar"));
    }
}
