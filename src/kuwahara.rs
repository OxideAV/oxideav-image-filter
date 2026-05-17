//! Kuwahara — edge-preserving smoothing via quadrant variance selection.
//!
//! Classical noise-reduction operator described in the 1976 Kuwahara /
//! Hachimura / Eiho / Kinoshita paper *Processing of RI-angiocardiographic
//! images*. For each output pixel the surrounding `(2*radius+1)²` window
//! is split into four overlapping quadrants of side `radius+1`; the
//! quadrant with the lowest luminance variance "wins" and its mean colour
//! is written to the output. Smooth regions still blur cleanly while
//! step edges survive because the low-variance quadrant always lies on
//! the homogeneous side of the edge.
//!
//! Algorithm summary (clean-room from the textbook formulation):
//!
//! ```text
//! For each pixel (x, y):
//!   1. Build four overlapping quadrants of size (radius+1)×(radius+1),
//!      anchored at (x-r..=x, y-r..=y) / (x..=x+r, y-r..=y) / ... etc.
//!   2. Compute mean and variance of *luma* over each quadrant.
//!   3. Pick quadrant Q* with the smallest variance.
//!   4. Write mean(Q*) per channel to out[x, y].
//! ```
//!
//! Border samples clamp to the nearest in-bounds coordinate. Alpha on
//! RGBA passes through (the alpha channel does not contribute to the
//! variance criterion).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Kuwahara edge-preserving smoothing filter.
#[derive(Clone, Copy, Debug)]
pub struct Kuwahara {
    /// Half-window radius (≥ 1). The full window is `(2*radius+1)²`;
    /// each quadrant is `(radius+1)²`.
    pub radius: u32,
}

impl Default for Kuwahara {
    fn default() -> Self {
        Self { radius: 3 }
    }
}

impl Kuwahara {
    /// Build with the given radius (clamped to ≥ 1).
    pub fn new(radius: u32) -> Self {
        Self {
            radius: radius.max(1),
        }
    }
}

impl ImageFilter for Kuwahara {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Kuwahara requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let r = self.radius as i32;
        let src = &input.planes[0];
        let row_stride = w * bpp;
        let mut data = vec![0u8; row_stride * h];

        for y in 0..h as i32 {
            for x in 0..w as i32 {
                // Four quadrants. Each is an axis-aligned (radius+1)²
                // box anchored at one corner of the central pixel:
                //   NW: [x-r..=x, y-r..=y]
                //   NE: [x..=x+r, y-r..=y]
                //   SW: [x-r..=x, y..=y+r]
                //   SE: [x..=x+r, y..=y+r]
                let quads = [
                    (x - r, x, y - r, y),
                    (x, x + r, y - r, y),
                    (x - r, x, y, y + r),
                    (x, x + r, y, y + r),
                ];
                let mut best_var = f32::INFINITY;
                let mut best_mean = [0f32; 4];
                for (x0, x1, y0, y1) in quads {
                    let (mean, var) = quad_stats(src, w, h, bpp, x0, x1, y0, y1);
                    if var < best_var {
                        best_var = var;
                        best_mean = mean;
                    }
                }
                let dst_base = (y as usize) * row_stride + (x as usize) * bpp;
                for (c, m) in best_mean.iter().take(bpp).enumerate() {
                    data[dst_base + c] = m.round().clamp(0.0, 255.0) as u8;
                }
                if bpp == 4 {
                    // Alpha pass-through (from the centre pixel).
                    let cy = y.clamp(0, h as i32 - 1) as usize;
                    let cx = x.clamp(0, w as i32 - 1) as usize;
                    data[dst_base + 3] = src.data[cy * src.stride + cx * 4 + 3];
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

/// Compute the mean colour and luminance variance over the rectangle
/// `[x0..=x1] × [y0..=y1]` of `src`. Coordinates outside the image
/// clamp to the nearest in-bounds sample.
#[allow(clippy::too_many_arguments)]
fn quad_stats(
    src: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
) -> ([f32; 4], f32) {
    let mut sum = [0f32; 4];
    let mut sum_l = 0f32;
    let mut sum_l2 = 0f32;
    let mut n = 0u32;
    for yy in y0..=y1 {
        let cy = yy.clamp(0, h as i32 - 1) as usize;
        let row_off = cy * src.stride;
        for xx in x0..=x1 {
            let cx = xx.clamp(0, w as i32 - 1) as usize;
            let base = row_off + cx * bpp;
            let l = match bpp {
                1 => {
                    let v = src.data[base] as f32;
                    sum[0] += v;
                    v
                }
                3 => {
                    let r = src.data[base] as f32;
                    let g = src.data[base + 1] as f32;
                    let b = src.data[base + 2] as f32;
                    sum[0] += r;
                    sum[1] += g;
                    sum[2] += b;
                    rec601_luma_f(r, g, b)
                }
                4 => {
                    let r = src.data[base] as f32;
                    let g = src.data[base + 1] as f32;
                    let b = src.data[base + 2] as f32;
                    sum[0] += r;
                    sum[1] += g;
                    sum[2] += b;
                    rec601_luma_f(r, g, b)
                }
                _ => 0.0,
            };
            sum_l += l;
            sum_l2 += l * l;
            n += 1;
        }
    }
    let nf = n as f32;
    let mean = [sum[0] / nf, sum[1] / nf, sum[2] / nf, sum[3] / nf];
    let mean_l = sum_l / nf;
    // var = E[L²] - E[L]²
    let var = (sum_l2 / nf - mean_l * mean_l).max(0.0);
    (mean, var)
}

#[inline]
fn rec601_luma_f(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
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
    fn constant_input_is_fixed_point() {
        let input = gray(16, 16, |_, _| 128);
        let out = Kuwahara::new(2).apply(&input, p_gray(16, 16)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn radius_clamped_to_one() {
        assert_eq!(Kuwahara::new(0).radius, 1);
        assert_eq!(Kuwahara::new(1).radius, 1);
    }

    #[test]
    fn step_edge_is_preserved() {
        // Vertical edge at x=8 between black (left) and white (right).
        let input = gray(16, 16, |x, _| if x < 8 { 0 } else { 255 });
        let out = Kuwahara::new(2).apply(&input, p_gray(16, 16)).unwrap();
        // Far from the seam, both sides should remain saturated.
        for y in 4..12 {
            assert_eq!(out.planes[0].data[y * 16], 0, "left dark side decayed");
            assert_eq!(
                out.planes[0].data[y * 16 + 15],
                255,
                "right bright side decayed"
            );
        }
    }

    #[test]
    fn noise_in_flat_region_is_suppressed() {
        // Salt-and-pepper noise in an otherwise-128 image; Kuwahara picks
        // the low-variance quadrant, so the mean should be close to 128.
        let input = gray(16, 16, |x, y| {
            if (x + y) % 7 == 0 {
                0
            } else if (x + y) % 11 == 0 {
                255
            } else {
                128
            }
        });
        let out = Kuwahara::new(3).apply(&input, p_gray(16, 16)).unwrap();
        // Check centre pixel — far enough from the border that all four
        // quadrants are interior.
        let v = out.planes[0].data[8 * 16 + 8] as i32;
        assert!((v - 128).abs() < 60, "centre pixel drifted: {v}");
    }

    #[test]
    fn rgb_output_keeps_format() {
        let input = rgb(8, 8, |x, _| [(x * 32) as u8, 100, 200]);
        let out = Kuwahara::new(2).apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8 {
            for x in 0..8 {
                data.extend_from_slice(&[(x * 16) as u8, (y * 16) as u8, 100, 77]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 8,
            height: 8,
        };
        let out = Kuwahara::new(2).apply(&input, p).unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[3], 77, "alpha drifted");
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
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = Kuwahara::new(1).apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("Kuwahara"));
    }
}
