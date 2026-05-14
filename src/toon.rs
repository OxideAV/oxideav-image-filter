//! Toon — cel-shaded "cartoon" look: posterise the colours to a small
//! palette and overlay a black edge trace.
//!
//! Implementation steps (clean-room):
//!
//! 1. Per-channel posterise to `levels` evenly spaced buckets — gives
//!    the flat-colour "cel shading" body.
//! 2. Sobel edge magnitude on a Rec. 601 luma proxy.
//! 3. Threshold the magnitude at `edge_threshold`; pixels above are
//!    "ink" and overwritten with `edge_color`.
//!
//! Inspired by the classic NPR cartoon recipe. The maths is the same
//! quantise / Sobel kit used by `Posterize` + `Edge` elsewhere in this
//! crate; the only novelty is the threshold + overlay composition.
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba`. Alpha (RGBA) is preserved. Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Toon / cel-shading filter.
#[derive(Clone, Copy, Debug)]
pub struct Toon {
    /// Per-channel palette buckets (≥ 2). Smaller ⇒ flatter look.
    pub levels: u32,
    /// Sobel-magnitude cut-off above which a pixel is considered an
    /// edge / "ink" stroke. `[0, 255]`. Smaller ⇒ more lines.
    pub edge_threshold: u8,
    /// Colour painted over the detected edges (RGB).
    pub edge_color: [u8; 3],
}

impl Default for Toon {
    fn default() -> Self {
        Self::new(4, 60, [0, 0, 0])
    }
}

impl Toon {
    /// Build a toon filter with explicit palette buckets, edge
    /// threshold, and edge colour.
    pub fn new(levels: u32, edge_threshold: u8, edge_color: [u8; 3]) -> Self {
        Self {
            levels: levels.max(2),
            edge_threshold,
            edge_color,
        }
    }
}

impl ImageFilter for Toon {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Toon requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        if w < 3 || h < 3 {
            // Sobel needs a 3×3 neighbourhood; fall back to plain
            // posterise on tiny inputs.
            return Ok(posterise_only(input, params, self.levels, bpp, has_alpha));
        }

        let lut = build_lut(self.levels);
        let row = w * bpp;
        let sp = &input.planes[0];

        // Pre-compute a luma plane for Sobel.
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            let s = &sp.data[y * sp.stride..y * sp.stride + row];
            for x in 0..w {
                let base = x * bpp;
                let r = s[base] as u32;
                let g = s[base + 1] as u32;
                let b = s[base + 2] as u32;
                luma[y * w + x] = ((77 * r + 150 * g + 29 * b) >> 8).min(255) as u8;
            }
        }

        let mut data = vec![0u8; row * h];
        for y in 0..h {
            let s = &sp.data[y * sp.stride..y * sp.stride + row];
            let dst_row = &mut data[y * row..(y + 1) * row];
            for x in 0..w {
                let base = x * bpp;
                let is_edge = if x == 0 || y == 0 || x + 1 >= w || y + 1 >= h {
                    false
                } else {
                    // Sobel 3x3 on luma.
                    let p = |yy: usize, xx: usize| luma[yy * w + xx] as i32;
                    let gx = -p(y - 1, x - 1) - 2 * p(y, x - 1) - p(y + 1, x - 1)
                        + p(y - 1, x + 1)
                        + 2 * p(y, x + 1)
                        + p(y + 1, x + 1);
                    let gy = -p(y - 1, x - 1) - 2 * p(y - 1, x) - p(y - 1, x + 1)
                        + p(y + 1, x - 1)
                        + 2 * p(y + 1, x)
                        + p(y + 1, x + 1);
                    let mag = ((gx * gx + gy * gy) as f32).sqrt();
                    mag.round().clamp(0.0, 255.0) as u8 > self.edge_threshold
                };
                if is_edge {
                    dst_row[base] = self.edge_color[0];
                    dst_row[base + 1] = self.edge_color[1];
                    dst_row[base + 2] = self.edge_color[2];
                    if has_alpha {
                        dst_row[base + 3] = s[base + 3];
                    }
                } else {
                    dst_row[base] = lut[s[base] as usize];
                    dst_row[base + 1] = lut[s[base + 1] as usize];
                    dst_row[base + 2] = lut[s[base + 2] as usize];
                    if has_alpha {
                        dst_row[base + 3] = s[base + 3];
                    }
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride: row, data }],
        })
    }
}

fn posterise_only(
    input: &VideoFrame,
    params: VideoStreamParams,
    levels: u32,
    bpp: usize,
    has_alpha: bool,
) -> VideoFrame {
    let lut = build_lut(levels);
    let w = params.width as usize;
    let h = params.height as usize;
    let row = w * bpp;
    let sp = &input.planes[0];
    let mut data = vec![0u8; row * h];
    for y in 0..h {
        let s = &sp.data[y * sp.stride..y * sp.stride + row];
        let dst_row = &mut data[y * row..(y + 1) * row];
        for x in 0..w {
            let base = x * bpp;
            dst_row[base] = lut[s[base] as usize];
            dst_row[base + 1] = lut[s[base + 1] as usize];
            dst_row[base + 2] = lut[s[base + 2] as usize];
            if has_alpha {
                dst_row[base + 3] = s[base + 3];
            }
        }
    }
    VideoFrame {
        pts: input.pts,
        planes: vec![VideoPlane { stride: row, data }],
    }
}

fn build_lut(levels: u32) -> [u8; 256] {
    let n = levels.max(2) as f32;
    let mut lut = [0u8; 256];
    let step = 255.0 / (n - 1.0);
    for (i, slot) in lut.iter_mut().enumerate() {
        let bucket = ((i as f32) * n / 256.0).floor().min(n - 1.0);
        *slot = (bucket * step).round().clamp(0.0, 255.0) as u8;
    }
    lut
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flat_input_only_posterises() {
        // 8x8 constant ⇒ no edges anywhere ⇒ result is pure posterise.
        let input = rgb(8, 8, |_, _| (200, 100, 50));
        let out = Toon::new(4, 30, [0, 0, 0])
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        let lut = build_lut(4);
        let expect = [lut[200], lut[100], lut[50]];
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &expect);
        }
    }

    #[test]
    fn edges_paint_ink_colour() {
        // Half-light / half-dark step at x=4 in an 8x8 image.
        let input = rgb(8, 8, |x, _| if x < 4 { (0, 0, 0) } else { (255, 255, 255) });
        let out = Toon::new(2, 30, [255, 0, 0])
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        // Some pixel along the step (x=3 or 4) must have picked up the
        // red ink colour.
        let mut found_red = false;
        for y in 1..7 {
            for x in 3..5 {
                let base = (y * 8 + x) * 3;
                let p = &out.planes[0].data[base..base + 3];
                if p == [255, 0, 0] {
                    found_red = true;
                }
            }
        }
        assert!(found_red, "expected at least one edge pixel painted red");
    }

    #[test]
    fn rgba_alpha_preserved() {
        let data: Vec<u8> = (0..16).flat_map(|_| [200u8, 100, 50, 77]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Toon::default()
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
            assert_eq!(out.planes[0].data[i * 4 + 3], 77, "alpha at pixel {i}");
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
        let err = Toon::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Toon"));
    }
}
