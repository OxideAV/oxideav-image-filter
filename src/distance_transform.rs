//! Distance Transform — two-pass chamfer / Borgefors.
//!
//! Computes the distance from every "background" pixel to the nearest
//! "foreground" pixel of a binary mask. Output is a `Gray8` image where
//! each pixel's value is its (clamped) distance to the closest
//! foreground pixel — useful for stroke generation, contour
//! visualisation, dilation by distance, signed-distance fields, etc.
//!
//! Algorithm (clean-room from Gunilla Borgefors, "Distance
//! Transformations in Digital Images", Computer Vision, Graphics, and
//! Image Processing 34 (1986), §3.1 "The Sequential Algorithm",
//! eq. (5)+(6) — the 3-4 chamfer mask):
//!
//! 1. Threshold the input to a binary mask. Foreground pixels start at
//!    distance 0; background pixels at +∞.
//! 2. Forward pass (top-left → bottom-right): for each background
//!    pixel `p`, update
//!    `d(p) = min(d(p), d(upper-left)+4, d(up)+3, d(upper-right)+4,
//!     d(left)+3)`.
//! 3. Backward pass (bottom-right → top-left): for each pixel `p`,
//!    update
//!    `d(p) = min(d(p), d(right)+3, d(lower-left)+4, d(below)+3,
//!     d(lower-right)+4)`.
//! 4. Optionally rescale the integer chamfer-3-4 metric back to
//!    pixel-units (`distance ≈ chamfer / 3`).
//!
//! The 3-4 chamfer (vs the 1-1 city-block or 1-√2 chessboard masks) is
//! the cheapest two-pass approximation to Euclidean distance with
//! ≤ 4% error per Borgefors §4. Output is clamped to `[0, 255]` so
//! large empty regions saturate to white.
//!
//! Operates on `Gray8` input only; the mask is computed as
//! `value >= threshold ⇒ foreground`. Output is always `Gray8`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Distance-Transform filter (3-4 chamfer / Borgefors).
#[derive(Clone, Copy, Debug)]
pub struct DistanceTransform {
    /// Foreground threshold. Pixels `>= threshold` count as foreground
    /// (distance `0`); others are background.
    pub threshold: u8,
    /// If `true`, invert the mask so dark pixels become the foreground
    /// (useful when the source has a black-on-white drawing).
    pub invert: bool,
    /// Multiplier on the output distance. `1.0` (default) is "show the
    /// raw distance up to 255"; `0.5` darkens / compresses the gradient
    /// so the visible falloff fits a smaller dynamic range.
    pub scale: f32,
}

impl Default for DistanceTransform {
    fn default() -> Self {
        Self {
            threshold: 128,
            invert: false,
            scale: 1.0,
        }
    }
}

impl DistanceTransform {
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            invert: false,
            scale: 1.0,
        }
    }

    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
        }
        self
    }
}

impl ImageFilter for DistanceTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: DistanceTransform requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DistanceTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];

        // Borgefors §3.1: chamfer-3-4 sentinel large enough that
        // `INF + 4` never wraps. `i32` is comfortable; max distance for
        // an N×M grid is `4·(N+M)`.
        const INF: i32 = 1_000_000;
        let mut d = vec![INF; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = src.data[y * src.stride + x];
                let fg = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
                if fg {
                    d[y * w + x] = 0;
                }
            }
        }

        // Forward pass: scan rows top → bottom, columns left → right.
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut best = d[i];
                if y > 0 {
                    // up
                    best = best.min(d[(y - 1) * w + x].saturating_add(3));
                    if x > 0 {
                        best = best.min(d[(y - 1) * w + (x - 1)].saturating_add(4));
                    }
                    if x + 1 < w {
                        best = best.min(d[(y - 1) * w + (x + 1)].saturating_add(4));
                    }
                }
                if x > 0 {
                    best = best.min(d[y * w + (x - 1)].saturating_add(3));
                }
                d[i] = best;
            }
        }

        // Backward pass: scan rows bottom → top, columns right → left.
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut best = d[i];
                if y + 1 < h {
                    best = best.min(d[(y + 1) * w + x].saturating_add(3));
                    if x > 0 {
                        best = best.min(d[(y + 1) * w + (x - 1)].saturating_add(4));
                    }
                    if x + 1 < w {
                        best = best.min(d[(y + 1) * w + (x + 1)].saturating_add(4));
                    }
                }
                if x + 1 < w {
                    best = best.min(d[y * w + (x + 1)].saturating_add(3));
                }
                d[i] = best;
            }
        }

        // Render to Gray8. The chamfer metric is `3·euclidean` so
        // divide by 3 to get pixel-distance, then apply `scale`.
        let mut out = vec![0u8; w * h];
        for (i, &chamfer) in d.iter().enumerate() {
            let pix = (chamfer as f32 / 3.0) * self.scale;
            out[i] = pix.round().clamp(0.0, 255.0) as u8;
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
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
    fn foreground_pixel_has_zero_distance() {
        // Single bright pixel at centre; that pixel itself must be 0.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
    }

    #[test]
    fn distance_increases_with_radius() {
        // Single FG pixel — distance should increase monotonically as
        // we move away (chamfer 3-4 → euclidean approximation).
        let input = gray(11, 11, |x, y| if x == 5 && y == 5 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .apply(&input, p_gray(11, 11))
            .unwrap();
        // Manhattan-1 (4 cardinals) should be exactly 1 px after divide.
        let centre = (5usize, 5usize);
        let d1 = out.planes[0].data[(centre.1) * 11 + centre.0 + 1];
        let d2 = out.planes[0].data[(centre.1) * 11 + centre.0 + 2];
        let d3 = out.planes[0].data[(centre.1) * 11 + centre.0 + 3];
        assert!(d1 < d2, "d1={d1} d2={d2}");
        assert!(d2 < d3, "d2={d2} d3={d3}");
    }

    #[test]
    fn all_background_saturates() {
        // No foreground at all → every pixel stays at INF / 3 ⇒ clamps to 255.
        let input = gray(8, 8, |_, _| 0);
        let out = DistanceTransform::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255, "no foreground should saturate");
        }
    }

    #[test]
    fn invert_picks_dark_pixels_as_foreground() {
        // Dark dot on bright background.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 0 } else { 255 });
        let out = DistanceTransform::new(128)
            .with_invert(true)
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
    }

    #[test]
    fn rejects_rgb() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: vec![0; 48],
            }],
        };
        let err = DistanceTransform::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("DistanceTransform"));
    }
}
