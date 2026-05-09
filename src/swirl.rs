//! Radius-decaying rotational distortion (ImageMagick `-swirl N`).
//!
//! Each pixel is rotated about the image centre by an angle that
//! decreases linearly with radius — pixels at the centre rotate by the
//! full configured angle, pixels at the inscribed-circle edge rotate by
//! zero. Outside the inscribed circle the source is sampled directly.
//!
//! ```text
//! For each output (px, py):
//!     dx = px - cx; dy = py - cy
//!     max_r = min(w/2, h/2)
//!     r = sqrt(dx*dx + dy*dy) / max_r
//!     if r >= 1.0: out = in
//!     else:
//!         angle = degrees * (1 - r) * π / 180
//!         sx = cx + dx*cos(angle) - dy*sin(angle)
//!         sy = cy + dx*sin(angle) + dy*cos(angle)
//!         out = bilinear_sample(in, sx, sy)
//! ```
//!
//! `degrees = 0.0` is bit-exact identity. Only operates on packed
//! Rgb24/Rgba; alpha is sampled (and so survives) as just another channel.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Radius-decaying rotational distortion.
///
/// `degrees` is the rotation applied at the centre; the rotation falls
/// off linearly to 0° at the inscribed-circle edge. Negative values
/// swirl counter-clockwise.
#[derive(Clone, Copy, Debug)]
pub struct Swirl {
    pub degrees: f32,
}

impl Default for Swirl {
    fn default() -> Self {
        Self { degrees: 90.0 }
    }
}

impl Swirl {
    /// Build a swirl with explicit centre rotation in degrees.
    pub fn new(degrees: f32) -> Self {
        Self {
            degrees: if degrees.is_finite() { degrees } else { 0.0 },
        }
    }
}

impl ImageFilter for Swirl {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Swirl requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        // Identity fast-path (bit-exact pass-through).
        if self.degrees.abs() < 1e-9 {
            for y in 0..h {
                let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
                let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
                dst_row.copy_from_slice(src_row);
            }
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
        let max_r = (w.min(h) as f32) * 0.5;
        let radians = self.degrees.to_radians();

        for py in 0..h {
            let dy = py as f32 - cy;
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..w {
                let dx = px as f32 - cx;
                let dist = (dx * dx + dy * dy).sqrt();
                let r = if max_r > 0.0 { dist / max_r } else { 0.0 };
                let base = px * bpp;
                if r >= 1.0 || max_r == 0.0 {
                    let src_off = py * src.stride + base;
                    dst_row[base..base + bpp].copy_from_slice(&src.data[src_off..src_off + bpp]);
                    continue;
                }
                let angle = radians * (1.0 - r);
                let (sin_a, cos_a) = angle.sin_cos();
                let sx = cx + dx * cos_a - dy * sin_a;
                let sy = cy + dx * sin_a + dy * cos_a;
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
    fn degrees_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Swirl::new(0.0).apply(&input, params_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_invariant_under_swirl() {
        let input = rgb(8, 8, |_, _| (100, 150, 200));
        let out = Swirl::new(120.0).apply(&input, params_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn outside_inscribed_circle_unchanged() {
        let input = rgb(8, 8, |x, y| {
            ((x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8)
        });
        let out = Swirl::new(180.0).apply(&input, params_rgb(8, 8)).unwrap();
        let stride = out.planes[0].stride;
        // Corners are clearly outside the inscribed circle (r > 1).
        for &x in &[0usize, 7] {
            for &y in &[0usize, 7] {
                let off = y * stride + x * 3;
                assert_eq!(out.planes[0].data[off], input.planes[0].data[off]);
                assert_eq!(out.planes[0].data[off + 1], input.planes[0].data[off + 1]);
                assert_eq!(out.planes[0].data[off + 2], input.planes[0].data[off + 2]);
            }
        }
    }

    #[test]
    fn swirl_actually_moves_pixels() {
        // Place a single bright pixel near (but inside) the inscribed
        // circle of an 8×8 image; check that swirling 180° moves it.
        let mut data = vec![0u8; 8 * 8 * 3];
        // Set pixel (5, 4) red.
        let idx = (4 * 8 + 5) * 3;
        data[idx] = 255;
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = Swirl::new(180.0).apply(&input, params_rgb(8, 8)).unwrap();
        // The red splash shouldn't sit at exactly (5, 4) any more.
        let still_there = out.planes[0].data[idx];
        assert!(
            still_there < 255,
            "single red pixel did not move under 180° swirl: {still_there}"
        );
    }

    #[test]
    fn rgba_alpha_passes_through_for_uniform_alpha() {
        let data: Vec<u8> = (0..64).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Swirl::new(90.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        for i in 0..64 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 222, "alpha at pixel {i}");
        }
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
        let err = Swirl::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Swirl"));
    }
}
