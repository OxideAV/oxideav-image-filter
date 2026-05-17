//! Halftone — variable-size dot screening simulating offset printing.
//!
//! Divides the image into a grid of `cell_size`-pixel cells. Each cell's
//! mean luminance drives the *radius* of a black dot painted at the cell
//! centre on a configurable background — darker cells produce larger
//! dots so the total ink coverage matches the original tone.
//!
//! Clean-room derivation from the textbook AM-screening formula:
//!
//! ```text
//! For each cell:
//!   mean_y     = average Rec. 601 luma of the cell
//!   coverage   = (255 - mean_y) / 255              ∈ [0, 1]
//!   dot_radius = (cell_size / 2) · sqrt(coverage)  (constant ink area)
//! ```
//!
//! The `sqrt` keeps the *area* of the dot proportional to coverage —
//! this is the textbook trick that makes amplitude-modulated halftones
//! tonally accurate.
//!
//! # Pixel formats
//!
//! Input: `Gray8` / `Rgb24` / `Rgba`. Output preserves the input format
//! (the alpha plane on RGBA is left untouched). Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Halftone / dot-screen filter.
#[derive(Clone, Copy, Debug)]
pub struct Halftone {
    /// Side length of one halftone cell in pixels (≥ 2). Bigger cells
    /// make chunkier dots.
    pub cell_size: u32,
    /// Foreground (ink) colour painted into the dot. Defaults to black.
    pub ink_color: [u8; 3],
    /// Background (paper) colour painted around the dot. Defaults to
    /// white.
    pub paper_color: [u8; 3],
}

impl Default for Halftone {
    fn default() -> Self {
        Self {
            cell_size: 8,
            ink_color: [0, 0, 0],
            paper_color: [255, 255, 255],
        }
    }
}

impl Halftone {
    /// Build with a cell size; ink / paper default to black / white.
    pub fn new(cell_size: u32) -> Self {
        Self {
            cell_size: cell_size.max(2),
            ..Self::default()
        }
    }

    /// Override the ink colour.
    pub fn with_ink(mut self, ink: [u8; 3]) -> Self {
        self.ink_color = ink;
        self
    }

    /// Override the paper colour.
    pub fn with_paper(mut self, paper: [u8; 3]) -> Self {
        self.paper_color = paper;
        self
    }
}

impl ImageFilter for Halftone {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Halftone requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let cell = self.cell_size.max(2);
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }

        let cell_u = cell as usize;
        let cols = w.div_ceil(cell_u);
        let rows = h.div_ceil(cell_u);
        let src = &input.planes[0];
        let row_stride = w * bpp;

        // Per-cell mean luma.
        let mut means = vec![0u8; rows * cols];
        for cy in 0..rows {
            let y0 = cy * cell_u;
            let y1 = ((cy + 1) * cell_u).min(h);
            for cx in 0..cols {
                let x0 = cx * cell_u;
                let x1 = ((cx + 1) * cell_u).min(w);
                let mut sum: u64 = 0;
                let mut n: u64 = 0;
                for y in y0..y1 {
                    let row = &src.data[y * src.stride..y * src.stride + row_stride];
                    for x in x0..x1 {
                        let base = x * bpp;
                        let l = match bpp {
                            1 => row[base] as u32,
                            3 => rec601_luma(row[base], row[base + 1], row[base + 2]) as u32,
                            4 => rec601_luma(row[base], row[base + 1], row[base + 2]) as u32,
                            _ => 0,
                        };
                        sum += l as u64;
                        n += 1;
                    }
                }
                means[cy * cols + cx] = (sum / n.max(1)) as u8;
            }
        }

        // Paint output.
        let mut data = vec![0u8; row_stride * h];
        let half = cell as f32 * 0.5;
        for cy in 0..rows {
            let centre_y = (cy as f32) * cell as f32 + half;
            let y0 = cy * cell_u;
            let y1 = ((cy + 1) * cell_u).min(h);
            for cx in 0..cols {
                let centre_x = (cx as f32) * cell as f32 + half;
                let mean = means[cy * cols + cx] as f32;
                let coverage = ((255.0 - mean) / 255.0).clamp(0.0, 1.0);
                let r = half * coverage.sqrt();
                let r2 = r * r;
                for y in y0..y1 {
                    let dy = y as f32 + 0.5 - centre_y;
                    let dst_row = &mut data[y * row_stride..(y + 1) * row_stride];
                    let src_row = &src.data[y * src.stride..y * src.stride + row_stride];
                    for x in x0_x1(cx, cell_u, w).0..x0_x1(cx, cell_u, w).1 {
                        let dx = x as f32 + 0.5 - centre_x;
                        let inside = (dx * dx + dy * dy) <= r2;
                        let base = x * bpp;
                        let c = if inside {
                            self.ink_color
                        } else {
                            self.paper_color
                        };
                        match bpp {
                            1 => dst_row[base] = rec601_luma(c[0], c[1], c[2]),
                            3 => {
                                dst_row[base] = c[0];
                                dst_row[base + 1] = c[1];
                                dst_row[base + 2] = c[2];
                            }
                            4 => {
                                dst_row[base] = c[0];
                                dst_row[base + 1] = c[1];
                                dst_row[base + 2] = c[2];
                                dst_row[base + 3] = src_row[base + 3];
                            }
                            _ => unreachable!(),
                        }
                    }
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

fn x0_x1(cx: usize, cell_u: usize, w: usize) -> (usize, usize) {
    (cx * cell_u, ((cx + 1) * cell_u).min(w))
}

#[inline]
fn rec601_luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000).min(255) as u8
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
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
    fn pure_white_yields_full_paper() {
        let input = gray(8, 8, |_, _| 255);
        let out = Halftone::new(4).apply(&input, p_gray(8, 8)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn pure_black_yields_full_ink_fill() {
        // Coverage = 1 ⇒ radius = half ⇒ inscribed disk in the cell;
        // we expect a non-trivial number of ink pixels (centre region).
        let input = gray(8, 8, |_, _| 0);
        let out = Halftone::new(4).apply(&input, p_gray(8, 8)).unwrap();
        let ink_count = out.planes[0].data.iter().filter(|&&v| v == 0).count();
        // At minimum the centre 2×2 of each cell should be inked.
        assert!(ink_count >= 4 * 2 * 2, "got ink_count = {ink_count}");
    }

    #[test]
    fn rgb_output_preserves_format() {
        let input = rgb(16, 16, |x, _| [(x * 16) as u8, 100, 200]);
        let out = Halftone::new(4).apply(&input, p_rgb(16, 16)).unwrap();
        assert_eq!(out.planes[0].data.len(), 16 * 16 * 3);
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(16 * 16 * 4);
        for _ in 0..(16 * 16) {
            data.extend_from_slice(&[100u8, 100, 100, 99]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 64, data }],
        };
        let out = Halftone::new(4)
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
            assert_eq!(px[3], 99, "alpha not preserved");
        }
    }

    #[test]
    fn cell_size_clamped_to_two() {
        assert_eq!(Halftone::new(0).cell_size, 2);
        assert_eq!(Halftone::new(1).cell_size, 2);
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
        let err = Halftone::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Halftone"));
    }

    #[test]
    fn custom_ink_paper_colours_appear() {
        // Mid-grey, single cell ⇒ dot smaller than cell ⇒ both inks visible.
        let input = gray(8, 8, |_, _| 128);
        let out = Halftone::new(8)
            .with_ink([200, 0, 0]) // not used on Gray8 directly — luma form
            .with_paper([0, 200, 0])
            .apply(&input, p_gray(8, 8))
            .unwrap();
        // Output luma == rec601(red) and rec601(green) for ink/paper.
        let ink_l = rec601_luma(200, 0, 0);
        let paper_l = rec601_luma(0, 200, 0);
        let has_ink = out.planes[0].data.contains(&ink_l);
        let has_paper = out.planes[0].data.contains(&paper_l);
        assert!(has_ink, "ink colour missing from halftone output");
        assert!(has_paper, "paper colour missing from halftone output");
    }
}
