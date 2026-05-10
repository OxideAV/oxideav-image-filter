//! Random pixel-position perturbation (ImageMagick `-spread N`).
//!
//! For each output pixel, sample the source from a random offset in the
//! square `[-radius, radius]² ⊂ ℤ²`. The result is a "frosted glass"
//! displacement noise. This crate uses a deterministic, reproducible
//! PRNG seeded by `(seed, pixel_x, pixel_y)` so the same input + seed
//! always produces the same output.
//!
//! ```text
//! For each output (px, py):
//!     dx = rand_in[-r, r]; dy = rand_in[-r, r]
//!     sx = clamp(px + dx, 0, w-1); sy = clamp(py + dy, 0, h-1)
//!     out = in[sy, sx]
//! ```
//!
//! - `radius = 0` is bit-exact identity.
//! - Random offsets are integer (no bilinear sampling; matches IM's
//!   classical behaviour).
//! - Operates on packed `Rgb24` / `Rgba`. Alpha is sampled along with
//!   the colour channels — same offset is used for all components of a
//!   pixel, so RGBA stays self-consistent.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Random pixel-position perturbation with a fixed neighbourhood.
///
/// `radius` is the half-side of the square sampling window in pixels.
/// `seed` makes the noise reproducible — calling `apply` twice with the
/// same seed and input gives byte-identical output.
#[derive(Clone, Copy, Debug)]
pub struct Spread {
    pub radius: u32,
    pub seed: u64,
}

impl Default for Spread {
    fn default() -> Self {
        Self {
            radius: 3,
            seed: 0x9E37_79B9_7F4A_7C15, // golden-ratio constant
        }
    }
}

impl Spread {
    /// Build a spread filter with explicit radius. Default seed.
    pub fn new(radius: u32) -> Self {
        Self {
            radius,
            seed: Self::default().seed,
        }
    }

    /// Override the PRNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl ImageFilter for Spread {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Spread requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        // Identity fast-path.
        if self.radius == 0 {
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

        let r = self.radius as i32;
        let span = (2 * r + 1) as u32;
        let seed = self.seed;
        for py in 0..h {
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..w {
                // Mix coords + seed via a SplitMix-style avalanche;
                // gives well-distributed offsets per (x, y).
                let mut h64 = seed
                    .wrapping_add((px as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add((py as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                h64 ^= h64 >> 30;
                h64 = h64.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h64 ^= h64 >> 27;
                h64 = h64.wrapping_mul(0x94D0_49BB_1331_11EB);
                h64 ^= h64 >> 31;

                let dx = ((h64 as u32) % span) as i32 - r;
                let dy = (((h64 >> 32) as u32) % span) as i32 - r;
                let sx = (px as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let sy = (py as i32 + dy).clamp(0, h as i32 - 1) as usize;
                let src_off = sy * src.stride + sx * bpp;
                let dst_off = px * bpp;
                dst_row[dst_off..dst_off + bpp].copy_from_slice(&src.data[src_off..src_off + bpp]);
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
    fn radius_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Spread::new(0).apply(&input, params_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_invariant_under_spread() {
        let input = rgb(8, 8, |_, _| (100, 150, 200));
        let out = Spread::new(2).apply(&input, params_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let input = rgb(16, 16, |x, y| {
            ((x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8)
        });
        let a = Spread::new(3).apply(&input, params_rgb(16, 16)).unwrap();
        let b = Spread::new(3).apply(&input, params_rgb(16, 16)).unwrap();
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn different_seed_produces_different_output() {
        let input = rgb(16, 16, |x, y| {
            ((x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8)
        });
        let a = Spread::new(3)
            .with_seed(0xDEAD_BEEF)
            .apply(&input, params_rgb(16, 16))
            .unwrap();
        let b = Spread::new(3)
            .with_seed(0xCAFE_BABE)
            .apply(&input, params_rgb(16, 16))
            .unwrap();
        assert_ne!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn spread_actually_moves_pixels() {
        // Lone bright pixel near centre should be displaced (or rather,
        // most output pixels should not equal their input counterpart at
        // (4,4) because they sample from elsewhere).
        let mut data = vec![10u8; 8 * 8 * 3];
        let idx = (4 * 8 + 4) * 3;
        data[idx] = 255;
        data[idx + 1] = 255;
        data[idx + 2] = 255;
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = Spread::new(2).apply(&input, params_rgb(8, 8)).unwrap();
        // Output should differ from input somewhere (overwhelmingly
        // likely with a 2-radius spread on this fixture).
        assert_ne!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rgba_alpha_stays_consistent() {
        let data: Vec<u8> = (0..64).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Spread::new(2)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        // Uniform alpha source ⇒ every sampled alpha is also 222.
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
        let err = Spread::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Spread"));
    }
}
