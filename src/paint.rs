//! Oil-paint stylise (`-paint radius`).
//!
//! Replaces every pixel by the colour that occurs most frequently among
//! the samples in a `(2*radius + 1)²` window around it — quantised to
//! a coarse intensity bucket so similar shades count together. The
//! result looks like a painting where smooth gradients have been
//! collapsed into a small number of flat tone "strokes".
//!
//! Clean-room implementation derived from the textbook description.
//! For each output pixel `(x, y)`:
//!
//! 1. Walk the `(2r + 1)²` window centred on `(x, y)`.
//! 2. Compute each window-pixel's intensity bucket
//!    `b = luma(px) * levels / 256` (default `levels = 20`).
//! 3. Tally a histogram of bucket counts and accumulate per-bucket
//!    channel sums.
//! 4. Pick the bucket with the highest count — break ties by picking
//!    the lower bucket index (deterministic).
//! 5. Output `(sum_R / count, sum_G / count, sum_B / count)`.
//!
//! 1×1 windows (`radius = 0`) are identity.
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba`: full effect; alpha is preserved on RGBA.
//! - `Gray8`: per-pixel intensity is the sample itself; output is the
//!   mean of the modal bucket.
//! - YUV / other formats return `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Oil-paint stylise filter.
#[derive(Clone, Copy, Debug)]
pub struct Paint {
    /// Window radius in pixels. The window is `(2 * radius + 1)²`.
    pub radius: u32,
    /// Number of intensity buckets used for the modal-bucket vote.
    /// Default `20`; smaller values create flatter / more cartoonish
    /// strokes, larger values approach an identity blur.
    pub levels: u32,
}

impl Default for Paint {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Paint {
    /// Build a paint filter with the given window radius and the
    /// default 20 intensity buckets.
    pub fn new(radius: u32) -> Self {
        Self { radius, levels: 20 }
    }

    /// Override the bucket count. Clamped to `[1, 256]`.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels.clamp(1, 256);
        self
    }
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8
}

impl ImageFilter for Paint {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, false),
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Paint requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        let r = self.radius as i32;
        let levels = self.levels.clamp(1, 256) as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        if r == 0 {
            // Identity — copy each row.
            for y in 0..h {
                let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
                out[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(src_row);
            }
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        // Per-pixel modal-bucket vote.
        let mut counts = vec![0u32; levels];
        let mut sum_r = vec![0u32; levels];
        let mut sum_g = vec![0u32; levels];
        let mut sum_b = vec![0u32; levels];
        for y in 0..h {
            for x in 0..w {
                counts.iter_mut().for_each(|c| *c = 0);
                sum_r.iter_mut().for_each(|s| *s = 0);
                sum_g.iter_mut().for_each(|s| *s = 0);
                sum_b.iter_mut().for_each(|s| *s = 0);
                for dy in -r..=r {
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let row = &src.data[yy * src.stride..yy * src.stride + row_bytes];
                    for dx in -r..=r {
                        let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let base = xx * bpp;
                        let (rv, gv, bv) = match bpp {
                            1 => (row[base], row[base], row[base]),
                            3 | 4 => (row[base], row[base + 1], row[base + 2]),
                            _ => unreachable!(),
                        };
                        let l = luma(rv, gv, bv) as u32;
                        let bucket = ((l * levels as u32) / 256).min(levels as u32 - 1) as usize;
                        counts[bucket] += 1;
                        sum_r[bucket] += rv as u32;
                        sum_g[bucket] += gv as u32;
                        sum_b[bucket] += bv as u32;
                    }
                }
                // Pick the lowest-index bucket with the highest count.
                let mut best = 0usize;
                let mut best_count = counts[0];
                for (i, &c) in counts.iter().enumerate().skip(1) {
                    if c > best_count {
                        best_count = c;
                        best = i;
                    }
                }
                let n = counts[best].max(1);
                let or = (sum_r[best] / n) as u8;
                let og = (sum_g[best] / n) as u8;
                let ob = (sum_b[best] / n) as u8;
                let dst_base = y * row_bytes + x * bpp;
                if bpp == 1 {
                    out[dst_base] = or; // mean of modal bucket
                } else {
                    out[dst_base] = or;
                    out[dst_base + 1] = og;
                    out[dst_base + 2] = ob;
                    if has_alpha {
                        // Alpha is taken straight from the source pixel
                        // (we don't mode-vote alpha — that would
                        // introduce artefacts at translucent edges).
                        let src_base = y * src.stride + x * bpp;
                        out[dst_base + 3] = src.data[src_base + 3];
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

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn radius_zero_is_identity() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 77]);
        let out = Paint::new(0).apply(&input, params_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn solid_image_passes_through() {
        let input = rgb(4, 4, |_, _| [100, 50, 200]);
        let out = Paint::new(1).apply(&input, params_rgb(4, 4)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[100, 50, 200]);
        }
    }

    #[test]
    fn picks_modal_colour_in_window() {
        // 3×3 input where 8 pixels are colour A and 1 is colour B at
        // the centre. With radius 1 every output pixel sees the same
        // window (border-clamped), so all output should be A.
        let input = rgb(3, 3, |x, y| {
            if x == 1 && y == 1 {
                [255, 0, 0]
            } else {
                [10, 10, 10]
            }
        });
        let out = Paint::new(1).apply(&input, params_rgb(3, 3)).unwrap();
        // Modal bucket (low intensity) wins ⇒ output is roughly the
        // mean of those 8 dark pixels.
        let centre = &out.planes[0].data[(3 + 1) * 3..(3 + 1) * 3 + 3];
        assert!(
            centre[0] < 30 && centre[1] < 30 && centre[2] < 30,
            "expected dark modal pixel, got {centre:?}"
        );
    }

    #[test]
    fn rgba_alpha_is_taken_from_source() {
        let data: Vec<u8> = (0..16).flat_map(|i| [10u8, 10, 10, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Paint::new(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8, "alpha at {i}");
        }
    }

    #[test]
    fn levels_clamped_to_1_256() {
        let p = Paint::new(1).with_levels(0);
        assert_eq!(p.levels, 1);
        let p = Paint::new(1).with_levels(1000);
        assert_eq!(p.levels, 256);
    }

    #[test]
    fn rejects_yuv() {
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
        let err = Paint::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Paint"));
    }
}
