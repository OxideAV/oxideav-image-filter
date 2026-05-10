//! Sinusoidal pixel displacement (ImageMagick `-wave AxL`).
//!
//! Bends each row of the image up or down by a sine wave whose
//! amplitude (peak vertical shift in pixels) and wavelength (horizontal
//! period in pixels) are configurable. Works as an inverse map with
//! bilinear sampling of the source plane.
//!
//! ```text
//! For each output (px, py):
//!     sx = px
//!     sy = py - amplitude * sin(2π * px / wavelength)
//!     out = bilinear_sample(in, sx, sy)
//! ```
//!
//! - `amplitude = 0.0` is bit-exact identity.
//! - `wavelength <= 0.0` is treated as identity (avoids divide-by-zero).
//! - Out-of-bounds samples clamp to the nearest edge (same border policy
//!   as the rest of the geometric filters in this crate).
//!
//! Operates on packed `Rgb24` / `Rgba`; alpha sampled along with the
//! colour channels.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Sinusoidal vertical displacement.
///
/// `amplitude` is the peak vertical shift in pixels. `wavelength` is the
/// horizontal distance between adjacent wave crests in pixels.
/// Negative amplitudes simply flip the wave's phase.
#[derive(Clone, Copy, Debug)]
pub struct Wave {
    pub amplitude: f32,
    pub wavelength: f32,
}

impl Default for Wave {
    fn default() -> Self {
        Self {
            amplitude: 5.0,
            wavelength: 25.0,
        }
    }
}

impl Wave {
    /// Build a wave filter with explicit amplitude (px) and wavelength (px).
    pub fn new(amplitude: f32, wavelength: f32) -> Self {
        Self {
            amplitude: if amplitude.is_finite() {
                amplitude
            } else {
                0.0
            },
            wavelength: if wavelength.is_finite() && wavelength > 0.0 {
                wavelength
            } else {
                0.0
            },
        }
    }
}

impl ImageFilter for Wave {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Wave requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        // Identity fast-path: zero amplitude or zero wavelength means no
        // displacement, so copy the input bit-for-bit.
        if self.amplitude.abs() < 1e-9 || self.wavelength <= 0.0 {
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

        let amp = self.amplitude;
        let two_pi_over_lambda = std::f32::consts::TAU / self.wavelength;

        for py in 0..h {
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..w {
                let phase = (px as f32) * two_pi_over_lambda;
                let sy = (py as f32) - amp * phase.sin();
                let sx = px as f32;
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
    fn amplitude_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Wave::new(0.0, 16.0)
            .apply(&input, params_rgb(8, 8))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn wavelength_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Wave::new(3.0, 0.0).apply(&input, params_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_invariant_under_wave() {
        // A flat image stays flat under any displacement (clamp-to-edge
        // sampling preserves the constant colour).
        let input = rgb(8, 8, |_, _| (100, 150, 200));
        let out = Wave::new(3.0, 8.0).apply(&input, params_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn columns_with_zero_phase_are_unshifted() {
        // sin(0) = 0, so column 0 (px=0) and column wavelength (px=8 if
        // wavelength=8) sample at the original y-coordinates — the
        // displacement vanishes there. Pick a vertical-stripe input so
        // the test is sensitive to row reshuffling.
        let input = rgb(8, 8, |_, y| ((y * 30) as u8, 0, 0));
        let out = Wave::new(2.0, 8.0).apply(&input, params_rgb(8, 8)).unwrap();
        for y in 0..8 {
            let off_in = y * 24;
            let off_out = y * 24;
            // Column 0 (phase 0).
            assert_eq!(out.planes[0].data[off_out], input.planes[0].data[off_in]);
        }
    }

    #[test]
    fn rgba_alpha_passes_through_for_uniform_alpha() {
        let data: Vec<u8> = (0..64).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Wave::new(2.0, 8.0)
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
    fn wave_actually_displaces_pixels() {
        // Horizontal stripes input — wave should shuffle rows. Pick a
        // wavelength that puts a non-trivial phase at most columns.
        let input = rgb(16, 16, |_, y| ((y * 16) as u8, 0, 0));
        let out = Wave::new(3.0, 16.0)
            .apply(&input, params_rgb(16, 16))
            .unwrap();
        // At column 4 (phase = π/2 ⇒ sin = 1), output row 0 samples
        // row -3 of input ⇒ clamps to row 0 ⇒ R = 0. At column 4
        // row 5, samples row 5 - 3 = row 2 ⇒ R = 2*16 = 32.
        let off_col4_row5 = 5 * 16 * 3 + 4 * 3;
        assert_eq!(out.planes[0].data[off_col4_row5], 32);
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
        let err = Wave::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Wave"));
    }
}
