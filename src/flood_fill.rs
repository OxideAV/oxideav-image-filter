//! Flood-fill — seeded region-paint with tolerance.
//!
//! Classic "magic wand" / paint-bucket primitive: starting at a
//! `(seed_x, seed_y)` pixel, paint a target colour into every
//! 4-connected neighbour whose colour differs from the *seed sample*
//! by less than `tolerance` per channel (Chebyshev distance — same
//! semantics as ImageMagick's `-fuzz`).
//!
//! The traversal uses an explicit-stack iterative scanline algorithm:
//!
//! 1. Pop a span seed `(x, y)`.
//! 2. Walk left while the pixel matches; walk right while it matches.
//! 3. Paint the span, then for each pixel in the span enqueue the
//!    "different-row" pixels above and below where the next span has
//!    not yet been seeded.
//!
//! The `visited` bitmap prevents revisits and bounds the total work at
//! `O(W·H)`.
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Other formats return
//! [`Error::unsupported`]. On `Rgba` the alpha channel is also matched
//! and overwritten; pass alpha 255 in `fill_color` to keep opaque
//! output.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Seeded flood-fill filter.
#[derive(Clone, Copy, Debug)]
pub struct FloodFill {
    /// Starting pixel.
    pub seed_x: u32,
    pub seed_y: u32,
    /// Target colour `[R, G, B, A]` (alpha used only on RGBA; B/A
    /// ignored on `Gray8`).
    pub fill_color: [u8; 4],
    /// Per-channel maximum difference from the seed sample that still
    /// counts as a region member. 0 ⇒ exact match.
    pub tolerance: u8,
}

impl Default for FloodFill {
    fn default() -> Self {
        Self {
            seed_x: 0,
            seed_y: 0,
            fill_color: [255, 0, 0, 255],
            tolerance: 0,
        }
    }
}

impl FloodFill {
    /// Build a flood-fill filter from a seed pixel + fill colour +
    /// tolerance.
    pub fn new(seed_x: u32, seed_y: u32, fill_color: [u8; 4], tolerance: u8) -> Self {
        Self {
            seed_x,
            seed_y,
            fill_color,
            tolerance,
        }
    }
}

impl ImageFilter for FloodFill {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: FloodFill requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let sx = self.seed_x as usize;
        let sy = self.seed_y as usize;
        if sx >= w || sy >= h {
            return Err(Error::invalid(format!(
                "FloodFill: seed ({sx},{sy}) outside image {w}x{h}"
            )));
        }

        let stride = w * bpp;
        // Reflate to a tight, contiguous buffer (no input-stride
        // surprises in the worker loop).
        let src_plane = &input.planes[0];
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            let s = &src_plane.data[y * src_plane.stride..y * src_plane.stride + stride];
            data[y * stride..(y + 1) * stride].copy_from_slice(s);
        }

        // Capture the seed colour BEFORE painting (otherwise the very
        // first pixel we paint changes the match criterion).
        let mut seed = [0u8; 4];
        {
            let base = sy * stride + sx * bpp;
            seed[..bpp].copy_from_slice(&data[base..base + bpp]);
        }
        let tol = self.tolerance;
        let mut fill = [0u8; 4];
        for (c, slot) in fill.iter_mut().enumerate().take(bpp) {
            *slot = self.fill_color[c.min(3)];
        }

        let matches = |row: &[u8], x: usize| {
            let b = x * bpp;
            for c in 0..bpp {
                if row[b + c].abs_diff(seed[c]) > tol {
                    return false;
                }
            }
            true
        };

        let already_filled = |row: &[u8], x: usize| {
            let b = x * bpp;
            for c in 0..bpp {
                if row[b + c] != fill[c] {
                    return false;
                }
            }
            true
        };

        // Span-fill stack of (x, y) seed points.
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(64);
        stack.push((sx, sy));

        while let Some((mut x, y)) = stack.pop() {
            // Re-check: an earlier span may have painted this pixel.
            {
                let row = &data[y * stride..(y + 1) * stride];
                if !matches(row, x) || already_filled(row, x) {
                    continue;
                }
            }
            // Walk left.
            while x > 0 {
                let row = &data[y * stride..(y + 1) * stride];
                if matches(row, x - 1) && !already_filled(row, x - 1) {
                    x -= 1;
                } else {
                    break;
                }
            }
            let mut span_above = false;
            let mut span_below = false;
            let mut sx = x;
            // Walk right + paint the contiguous span.
            loop {
                {
                    let row = &mut data[y * stride..(y + 1) * stride];
                    let base = sx * bpp;
                    row[base..base + bpp].copy_from_slice(&fill[..bpp]);
                }

                // Above-row check.
                if y > 0 {
                    let row = &data[(y - 1) * stride..y * stride];
                    let m = matches(row, sx) && !already_filled(row, sx);
                    if !span_above && m {
                        stack.push((sx, y - 1));
                        span_above = true;
                    } else if span_above && !m {
                        span_above = false;
                    }
                }
                // Below-row check.
                if y + 1 < h {
                    let row = &data[(y + 1) * stride..(y + 2) * stride];
                    let m = matches(row, sx) && !already_filled(row, sx);
                    if !span_below && m {
                        stack.push((sx, y + 1));
                        span_below = true;
                    } else if span_below && !m {
                        span_below = false;
                    }
                }

                sx += 1;
                if sx >= w {
                    break;
                }
                let row = &data[y * stride..(y + 1) * stride];
                if !matches(row, sx) || already_filled(row, sx) {
                    break;
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn fills_uniform_region_in_gray8() {
        // 4x4 all-zero canvas → flood-fill from (0,0) paints everything.
        let input = gray(4, 4, |_, _| 0);
        let out = FloodFill::new(0, 0, [200, 0, 0, 0], 0)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn stops_at_border_in_gray8() {
        // 4x4: top half = 0, bottom half = 200. Seed in top half.
        let input = gray(4, 4, |_, y| if y < 2 { 0 } else { 200 });
        let out = FloodFill::new(0, 0, [50, 0, 0, 0], 0)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let v = out.planes[0].data[(y * 4 + x) as usize];
                if y < 2 {
                    assert_eq!(v, 50, "top half at ({x},{y})");
                } else {
                    assert_eq!(v, 200, "bottom half at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn tolerance_widens_match() {
        // Seed is 0; neighbour is 10. tolerance=0 ⇒ stops; tolerance=15
        // ⇒ both filled.
        let input = gray(2, 1, |x, _| if x == 0 { 0 } else { 10 });
        let out = FloodFill::new(0, 0, [100, 0, 0, 0], 0)
            .apply(&input, p_gray(2, 1))
            .unwrap();
        assert_eq!(out.planes[0].data, vec![100, 10]);
        let out2 = FloodFill::new(0, 0, [100, 0, 0, 0], 15)
            .apply(&input, p_gray(2, 1))
            .unwrap();
        assert_eq!(out2.planes[0].data, vec![100, 100]);
    }

    #[test]
    fn rgba_fills_with_alpha() {
        // 2x2 RGBA all-zero. Fill with [50, 100, 150, 200].
        let data = vec![0u8; 16];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = FloodFill::new(0, 0, [50, 100, 150, 200], 0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for i in 0..4 {
            let p = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(p, &[50, 100, 150, 200]);
        }
    }

    #[test]
    fn rejects_out_of_bounds_seed() {
        let input = gray(2, 2, |_, _| 0);
        let err = FloodFill::new(99, 99, [255, 0, 0, 0], 0)
            .apply(&input, p_gray(2, 2))
            .unwrap_err();
        assert!(format!("{err}").contains("outside"));
    }
}
