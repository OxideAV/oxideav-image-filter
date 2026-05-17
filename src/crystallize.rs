//! Crystallize — Voronoi-cell averaging on a jittered point lattice.
//!
//! Builds a regular grid of "seed" sites spaced `cell_size` pixels
//! apart, jitters each seed by a deterministic PRNG (so a given `seed`
//! reproduces bit-exact output), then assigns every pixel to the
//! nearest seed. The cell's mean colour is painted into every member
//! pixel — the result is a stained-glass / crystal-shard appearance.
//!
//! Algorithm summary (clean-room from the textbook Voronoi
//! formulation):
//!
//! ```text
//! 1. cols = ceil(w / cell_size); rows = ceil(h / cell_size)
//! 2. for (cy, cx) in grid:
//!        seed[cy, cx] = (cx + 0.5 ± jitter, cy + 0.5 ± jitter) · cell_size
//! 3. for each pixel (x, y):
//!        find the seed in the 3×3 neighbourhood of grid cell
//!        (x/cell_size, y/cell_size) that minimises Euclidean²
//!        distance → owner_cell.
//! 4. mean colour of each owner_cell ⇒ paint every member.
//! ```
//!
//! The 3×3 neighbour scan is sufficient because the jitter is clamped
//! to `± cell_size / 2` so a seed cannot escape its own cell-radius
//! neighbourhood.
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Alpha is preserved on RGBA. Other
//! formats return [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Crystallize / Voronoi-cell averaging filter.
#[derive(Clone, Copy, Debug)]
pub struct Crystallize {
    /// Side length of the underlying grid cell in pixels (≥ 2). Larger
    /// values produce bigger crystal shards.
    pub cell_size: u32,
    /// `[0, 1]` — jitter the seeds by ±`jitter · cell_size / 2`. `0`
    /// is a regular rectangular tessellation; `1` lets each seed roam
    /// the full half-cell. Defaults to `0.5`.
    pub jitter: f32,
    /// PRNG seed for reproducible jitter. Same seed ⇒ identical output.
    pub seed: u64,
}

impl Default for Crystallize {
    fn default() -> Self {
        Self {
            cell_size: 8,
            jitter: 0.5,
            seed: 0xC0DE_F00D,
        }
    }
}

impl Crystallize {
    /// Build with a given cell size; jitter / seed default to `0.5`
    /// and a stable constant.
    pub fn new(cell_size: u32) -> Self {
        Self {
            cell_size: cell_size.max(2),
            ..Self::default()
        }
    }

    /// Override the jitter amount (`[0, 1]` clamp).
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Override the PRNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl ImageFilter for Crystallize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Crystallize requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let cell = self.cell_size.max(2);
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }

        let cols = (w as u32).div_ceil(cell);
        let rows = (h as u32).div_ceil(cell);
        let cell_f = cell as f32;
        let half = cell_f * 0.5;
        let jitter = self.jitter.clamp(0.0, 1.0) * half;

        // Generate jittered seed positions for every grid cell.
        let mut seeds: Vec<(f32, f32)> = Vec::with_capacity((rows * cols) as usize);
        for cy in 0..rows {
            for cx in 0..cols {
                let key = (cy as u64) * (cols as u64) + (cx as u64);
                let r1 = next_rand(self.seed, key, 0);
                let r2 = next_rand(self.seed, key, 1);
                let dx = ((r1 as f32 / u32::MAX as f32) * 2.0 - 1.0) * jitter;
                let dy = ((r2 as f32 / u32::MAX as f32) * 2.0 - 1.0) * jitter;
                let sx = (cx as f32) * cell_f + half + dx;
                let sy = (cy as f32) * cell_f + half + dy;
                seeds.push((sx, sy));
            }
        }

        // Two-pass accumulator: sum colours per cell, then count, then
        // mean. We need to know each pixel's owner cell before we can
        // start averaging so this is unavoidable.
        let mut owners = vec![0u32; w * h];
        let mut sums: Vec<[u64; 4]> = vec![[0u64; 4]; seeds.len()];
        let mut counts: Vec<u32> = vec![0u32; seeds.len()];

        let row_stride = w * bpp;
        let src = &input.planes[0];

        for y in 0..h {
            let yf = y as f32 + 0.5;
            let row_cy = ((y as u32) / cell).min(rows - 1);
            let src_row = &src.data[y * src.stride..y * src.stride + row_stride];
            for x in 0..w {
                let xf = x as f32 + 0.5;
                let row_cx = ((x as u32) / cell).min(cols - 1);
                // Scan the 3×3 cell neighbourhood for nearest seed.
                let mut best = u32::MAX as f32 * u32::MAX as f32; // big number
                let mut owner = 0u32;
                for dy in -1i32..=1 {
                    let ny = row_cy as i32 + dy;
                    if ny < 0 || ny as u32 >= rows {
                        continue;
                    }
                    for dx in -1i32..=1 {
                        let nx = row_cx as i32 + dx;
                        if nx < 0 || nx as u32 >= cols {
                            continue;
                        }
                        let idx = (ny as u32) * cols + (nx as u32);
                        let (sx, sy) = seeds[idx as usize];
                        let dxs = sx - xf;
                        let dys = sy - yf;
                        let d2 = dxs * dxs + dys * dys;
                        if d2 < best {
                            best = d2;
                            owner = idx;
                        }
                    }
                }
                owners[y * w + x] = owner;
                let cell_sum = &mut sums[owner as usize];
                let base = x * bpp;
                for c in 0..bpp {
                    cell_sum[c] += src_row[base + c] as u64;
                }
                counts[owner as usize] += 1;
            }
        }

        // Compute the per-cell mean. For empty cells (shouldn't happen
        // for this 3×3 sweep but defensive), fall back to zero.
        let mut means: Vec<[u8; 4]> = vec![[0u8; 4]; seeds.len()];
        for (i, m) in means.iter_mut().enumerate() {
            let c = counts[i].max(1) as u64;
            for ch in 0..bpp {
                m[ch] = ((sums[i][ch] + c / 2) / c).min(255) as u8;
            }
        }

        let mut out = vec![0u8; row_stride * h];
        for y in 0..h {
            let dst_row = &mut out[y * row_stride..(y + 1) * row_stride];
            let src_row = &src.data[y * src.stride..y * src.stride + row_stride];
            for x in 0..w {
                let owner = owners[y * w + x] as usize;
                let base = x * bpp;
                let m = &means[owner];
                match bpp {
                    1 => dst_row[base] = m[0],
                    3 => {
                        dst_row[base] = m[0];
                        dst_row[base + 1] = m[1];
                        dst_row[base + 2] = m[2];
                    }
                    4 => {
                        dst_row[base] = m[0];
                        dst_row[base + 1] = m[1];
                        dst_row[base + 2] = m[2];
                        // Alpha stays per-pixel rather than averaged.
                        dst_row[base + 3] = src_row[base + 3];
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_stride,
                data: out,
            }],
        })
    }
}

/// Deterministic, well-mixed 32-bit hash of `(seed, key, salt)`. Used
/// to derive a per-cell jitter without pulling a PRNG crate in.
fn next_rand(seed: u64, key: u64, salt: u64) -> u32 {
    // SplitMix-flavoured mixer; sufficient for visual jitter, not
    // cryptographic. Clean-room scrambler.
    let mut z = seed
        .wrapping_add(key.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z & 0xFFFF_FFFF) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
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
    fn flat_input_stays_flat() {
        let input = rgb_frame(16, 16, |_, _| [200, 100, 50]);
        let out = Crystallize::new(4).apply(&input, p_rgb(16, 16)).unwrap();
        for c in out.planes[0].data.chunks(3) {
            assert_eq!(c, [200, 100, 50]);
        }
    }

    #[test]
    fn cell_size_clamped_to_two() {
        let f = Crystallize::new(0);
        assert_eq!(f.cell_size, 2);
        let f = Crystallize::new(1);
        assert_eq!(f.cell_size, 2);
    }

    #[test]
    fn zero_jitter_is_regular_grid() {
        // With zero jitter and a perfectly aligned grid, every pixel in
        // a cell should pick the cell's own seed (i.e. the same cell-mean).
        // Build a chequerboard that varies *per cell* — each cell is
        // internally constant.
        let cell = 4u32;
        let input = rgb_frame(16, 8, |x, y| {
            let cx = x / cell;
            let cy = y / cell;
            if (cx + cy) & 1 == 0 {
                [200, 50, 50]
            } else {
                [50, 50, 200]
            }
        });
        let out = Crystallize::new(cell)
            .with_jitter(0.0)
            .apply(&input, p_rgb(16, 8))
            .unwrap();
        // Each output cell should match its input cell's flat colour.
        for y in 0..8u32 {
            for x in 0..16u32 {
                let cx = x / cell;
                let cy = y / cell;
                let expect = if (cx + cy) & 1 == 0 {
                    [200, 50, 50]
                } else {
                    [50, 50, 200]
                };
                let off = ((y * 16 + x) * 3) as usize;
                assert_eq!(
                    [
                        out.planes[0].data[off],
                        out.planes[0].data[off + 1],
                        out.planes[0].data[off + 2],
                    ],
                    expect,
                    "mismatch at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn deterministic_seed() {
        let input = rgb_frame(32, 32, |x, y| [(x * 8) as u8, (y * 8) as u8, 128]);
        let out1 = Crystallize::new(4)
            .with_seed(42)
            .apply(&input, p_rgb(32, 32))
            .unwrap();
        let out2 = Crystallize::new(4)
            .with_seed(42)
            .apply(&input, p_rgb(32, 32))
            .unwrap();
        assert_eq!(out1.planes[0].data, out2.planes[0].data);
    }

    #[test]
    fn different_seed_produces_different_output() {
        let input = rgb_frame(32, 32, |x, y| [(x * 8) as u8, (y * 8) as u8, 128]);
        let a = Crystallize::new(4)
            .with_seed(1)
            .apply(&input, p_rgb(32, 32))
            .unwrap();
        let b = Crystallize::new(4)
            .with_seed(2)
            .apply(&input, p_rgb(32, 32))
            .unwrap();
        assert_ne!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(16 * 16 * 4);
        for _ in 0..(16 * 16) {
            data.extend_from_slice(&[100, 150, 200, 77]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 64, data }],
        };
        let out = Crystallize::new(4)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[3], 77, "alpha not preserved");
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
        let err = Crystallize::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Crystallize"));
    }
}
