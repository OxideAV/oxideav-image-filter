//! Comic — manga / comic-book stylise.
//!
//! Two-stage pipeline:
//!
//! 1. **Flat fill**: an edge-preserving smooth (bilateral-style: an
//!    average over a `(2*radius+1)²` window weighted by colour
//!    similarity) collapses fine texture while keeping shape
//!    boundaries crisp. Then per-channel quantisation snaps the colour
//!    to a small palette so the result reads as comic ink rather than
//!    photo.
//! 2. **Ink lines**: a Sobel edge magnitude on the luma image is
//!    thresholded into a binary mask; mask pixels are painted with the
//!    configurable ink colour.
//!
//! Distinct from [`Toon`](crate::Toon): `Toon` posterises straight, no
//! pre-smoothing. `Comic` pre-smooths to suppress photo noise and uses
//! a thicker ink threshold for a heavier outline.
//!
//! Clean-room composition of textbook operators; no external library
//! code was referenced.
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba` only. Alpha is preserved on RGBA. Other formats
//! return [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Comic / manga stylise filter.
#[derive(Clone, Copy, Debug)]
pub struct Comic {
    /// Smoothing radius (≥ 1). Larger values flatten more fine detail.
    pub smooth_radius: u32,
    /// Colour similarity sigma for the smoothing pass (0 = pure box
    /// average; large values become edge-blind).
    pub colour_sigma: f32,
    /// Per-channel quantisation levels for the flat fill (≥ 2).
    pub levels: u32,
    /// 0..255 Sobel-magnitude cut-off for the ink mask.
    pub edge_threshold: u8,
    /// Ink colour. Defaults to black.
    pub ink_color: [u8; 3],
}

impl Default for Comic {
    fn default() -> Self {
        Self {
            smooth_radius: 2,
            colour_sigma: 40.0,
            levels: 4,
            edge_threshold: 40,
            ink_color: [0, 0, 0],
        }
    }
}

impl Comic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_smooth_radius(mut self, r: u32) -> Self {
        self.smooth_radius = r.max(1);
        self
    }

    pub fn with_colour_sigma(mut self, s: f32) -> Self {
        self.colour_sigma = s.max(1.0);
        self
    }

    pub fn with_levels(mut self, n: u32) -> Self {
        self.levels = n.max(2);
        self
    }

    pub fn with_edge_threshold(mut self, t: u8) -> Self {
        self.edge_threshold = t;
        self
    }

    pub fn with_ink_color(mut self, c: [u8; 3]) -> Self {
        self.ink_color = c;
        self
    }
}

impl ImageFilter for Comic {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Comic requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let radius = self.smooth_radius.max(1) as i32;
        let sigma = self.colour_sigma.max(1.0);
        let levels = self.levels.max(2);
        let row_stride = w * bpp;
        let src = &input.planes[0];

        // 1) Bilateral-style smooth (3-channel; reuses the in-tree
        // formulation but kept local so we don't pull a full
        // BilateralBlur instance).
        let mut smoothed = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let base = x * bpp;
                let centre = [
                    src.data[y * src.stride + base] as f32,
                    src.data[y * src.stride + base + 1] as f32,
                    src.data[y * src.stride + base + 2] as f32,
                ];
                let mut acc = [0f32; 3];
                let mut wsum = 0f32;
                for dy in -radius..=radius {
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let row = &src.data[yy * src.stride..yy * src.stride + row_stride];
                    for dx in -radius..=radius {
                        let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let b = xx * bpp;
                        let p = [row[b] as f32, row[b + 1] as f32, row[b + 2] as f32];
                        let dc = (p[0] - centre[0]).powi(2)
                            + (p[1] - centre[1]).powi(2)
                            + (p[2] - centre[2]).powi(2);
                        let wt = (-dc / (2.0 * sigma * sigma)).exp();
                        acc[0] += p[0] * wt;
                        acc[1] += p[1] * wt;
                        acc[2] += p[2] * wt;
                        wsum += wt;
                    }
                }
                let off = (y * w + x) * 3;
                smoothed[off] = (acc[0] / wsum.max(1e-6)).round().clamp(0.0, 255.0) as u8;
                smoothed[off + 1] = (acc[1] / wsum.max(1e-6)).round().clamp(0.0, 255.0) as u8;
                smoothed[off + 2] = (acc[2] / wsum.max(1e-6)).round().clamp(0.0, 255.0) as u8;
            }
        }

        // 2) Per-channel quantisation of the smoothed buffer.
        let lvl_f = levels as f32;
        let step = 255.0 / (lvl_f - 1.0);
        let mut quantise_lut = [0u8; 256];
        for (i, e) in quantise_lut.iter_mut().enumerate() {
            let bucket = (i as f32 * lvl_f / 256.0).floor().min(lvl_f - 1.0);
            *e = (bucket * step).round().clamp(0.0, 255.0) as u8;
        }
        for v in smoothed.iter_mut() {
            *v = quantise_lut[*v as usize];
        }

        // 3) Sobel edge mask on the *original* luma.
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + row_stride];
            for x in 0..w {
                let b = x * bpp;
                luma[y * w + x] =
                    ((row[b] as u32 * 299 + row[b + 1] as u32 * 587 + row[b + 2] as u32 * 114)
                        / 1000)
                        .min(255) as u8;
            }
        }
        let mut edge_mask = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                let l = |yy: usize, xx: usize| luma[yy * w + xx] as i32;
                let gx =
                    -l(ym, xm) - 2 * l(y, xm) - l(yp, xm) + l(ym, xp) + 2 * l(y, xp) + l(yp, xp);
                let gy =
                    -l(ym, xm) - 2 * l(ym, x) - l(ym, xp) + l(yp, xm) + 2 * l(yp, x) + l(yp, xp);
                let mag = ((gx * gx + gy * gy) as f32).sqrt().min(255.0) as u8;
                edge_mask[y * w + x] = if mag >= self.edge_threshold { 255 } else { 0 };
            }
        }

        // 4) Composite: ink on top of flat fill; copy alpha.
        let mut data = vec![0u8; row_stride * h];
        for y in 0..h {
            let dst = &mut data[y * row_stride..(y + 1) * row_stride];
            let src_row = &src.data[y * src.stride..y * src.stride + row_stride];
            for x in 0..w {
                let base = x * bpp;
                let off3 = (y * w + x) * 3;
                let is_edge = edge_mask[y * w + x] == 255;
                if is_edge {
                    dst[base] = self.ink_color[0];
                    dst[base + 1] = self.ink_color[1];
                    dst[base + 2] = self.ink_color[2];
                } else {
                    dst[base] = smoothed[off3];
                    dst[base + 1] = smoothed[off3 + 1];
                    dst[base + 2] = smoothed[off3 + 2];
                }
                if has_alpha {
                    dst[base + 3] = src_row[base + 3];
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn output_shape_matches_input() {
        let input = rgb(8, 8, |x, y| [(x * 30) as u8, (y * 30) as u8, 77]);
        let out = Comic::new().apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn pure_flat_rgb_quantises_to_palette_entry() {
        // No edges; flat input. Output must be the quantised value of
        // the input (no ink, just flat fill).
        let input = rgb(4, 4, |_, _| [120, 120, 120]);
        let out = Comic::new()
            .with_levels(4)
            .with_edge_threshold(250) // very strict ⇒ no ink
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // With 4 levels, every channel must be in {0, 85, 170, 255}.
        let allowed = [0u8, 85, 170, 255];
        for chunk in out.planes[0].data.chunks(3) {
            for &c in chunk {
                assert!(
                    allowed.contains(&c),
                    "comic output {c} not in {allowed:?} for flat input"
                );
            }
        }
    }

    #[test]
    fn edges_paint_ink_colour() {
        // Build a hard vertical edge: left half black, right half white.
        let input = rgb(8, 8, |x, _| if x < 4 { [0, 0, 0] } else { [255, 255, 255] });
        let out = Comic::new()
            .with_edge_threshold(20)
            .with_ink_color([200, 30, 30])
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        // Centre column should contain ink-coloured pixels.
        let mut found_ink = false;
        for y in 0..8usize {
            for x in 3..=4usize {
                let off = (y * 8 + x) * 3;
                if out.planes[0].data[off] == 200
                    && out.planes[0].data[off + 1] == 30
                    && out.planes[0].data[off + 2] == 30
                {
                    found_ink = true;
                }
            }
        }
        assert!(found_ink, "no ink pixels found at vertical edge");
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            data.extend_from_slice(&[100u8, 150, 200, 77]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Comic::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
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
        let err = Comic::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Comic"));
    }

    #[test]
    fn radius_clamped_to_one_minimum() {
        let f = Comic::new().with_smooth_radius(0);
        assert_eq!(f.smooth_radius, 1);
    }

    #[test]
    fn levels_clamped_to_two_minimum() {
        let f = Comic::new().with_levels(1);
        assert_eq!(f.levels, 2);
    }
}
