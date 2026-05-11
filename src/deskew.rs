//! Auto-deskew via projection-variance scoring (ImageMagick `-deskew threshold`).
//!
//! Estimates a small in-plane rotation that aligns text / scanlines with
//! the horizontal axis, then applies the inverse rotation. Algorithm
//! (clean-room — no IM source consulted):
//!
//! 1. Reduce input to `Gray8` and binarise against `threshold` (any
//!    sample at or below the cutoff is "ink" = 1, anything brighter
//!    is "background" = 0).
//! 2. For each candidate angle in `-max_angle..=+max_angle` (default
//!    -10°..=+10° in 0.5° steps), rotate the (x, y) coordinate of every
//!    "ink" pixel by that angle, bucketing the rotated `y'` value into
//!    integer rows and accumulating a per-row count.
//! 3. The best angle is the one whose row-count histogram has the
//!    highest variance — well-aligned text has the spikiest histogram
//!    (lots of "row-with-text" bins next to "row-with-whitespace" bins);
//!    misaligned text smears its ink across more rows.
//! 4. Apply the negative of the best angle to the input using the same
//!    rotate-with-bilinear-fill path as [`Rotate`](crate::rotate::Rotate).
//!
//! Output dimensions match the rotated bounding box (same as `Rotate`).
//! The background fill colour is configurable.

use crate::rotate::Rotate;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Auto-deskew filter.
///
/// `threshold` is the binarisation cutoff (samples `<= threshold` are
/// "ink"). `max_angle_degrees` bounds the search range — the filter
/// scans `(-max, +max)` in `step_degrees` increments.
#[derive(Clone, Copy, Debug)]
pub struct Deskew {
    /// Binarisation threshold on the 0..=255 luma scale. Values at or
    /// below this count as "ink" pixels for the histogram-variance
    /// scoring.
    pub threshold: u8,
    /// Half-range (in degrees) of the angle search. Defaults to 10°
    /// (search ±10°). Negative values are coerced to 0.
    pub max_angle_degrees: f32,
    /// Step (in degrees) between candidate angles. Defaults to 0.5°.
    pub step_degrees: f32,
    /// Background fill for the rotated output (`[R, G, B, A]`).
    pub background: [u8; 4],
}

impl Default for Deskew {
    fn default() -> Self {
        Self {
            threshold: 64,
            max_angle_degrees: 10.0,
            step_degrees: 0.5,
            background: [255, 255, 255, 255],
        }
    }
}

impl Deskew {
    /// Build with explicit `threshold`. Other parameters default to
    /// 10°-half-range / 0.5°-step / white background.
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Override the half-range of the angle search (degrees).
    pub fn with_max_angle(mut self, max_angle_degrees: f32) -> Self {
        self.max_angle_degrees = max_angle_degrees.max(0.0);
        self
    }

    /// Override the step between candidate angles (degrees). Values
    /// `<= 0` collapse to a single-angle (0°) search.
    pub fn with_step(mut self, step_degrees: f32) -> Self {
        self.step_degrees = if step_degrees.is_finite() && step_degrees > 0.0 {
            step_degrees
        } else {
            0.5
        };
        self
    }

    /// Override the background fill for the rotated output.
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }

    /// Estimate the skew angle (in degrees) for `input` without
    /// applying the rotation. Returns `0.0` for empty / degenerate
    /// frames.
    pub fn estimate_angle(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<f32, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Deskew does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(0.0);
        }
        let luma = build_luma(input, params)?;
        Ok(estimate_skew(
            &luma,
            w,
            h,
            self.threshold,
            self.max_angle_degrees.max(0.0),
            self.step_degrees,
        ))
    }
}

impl ImageFilter for Deskew {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let angle = self.estimate_angle(input, params)?;
        // Rotate by the negative of the estimated skew to restore
        // horizontal alignment.
        Rotate::new(-angle)
            .with_background(self.background)
            .apply(input, params)
    }
}

fn estimate_skew(luma: &[u8], w: usize, h: usize, threshold: u8, max_angle: f32, step: f32) -> f32 {
    // Build the ink-pixel list once — every candidate angle re-uses it.
    let mut ink_xy = Vec::new();
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    for y in 0..h {
        for x in 0..w {
            if luma[y * w + x] <= threshold {
                ink_xy.push((x as f32 - cx, y as f32 - cy));
            }
        }
    }
    if ink_xy.is_empty() {
        return 0.0;
    }
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        0.5
    };
    let max_angle = max_angle.max(0.0);
    if max_angle == 0.0 {
        return 0.0;
    }
    let n_steps = ((2.0 * max_angle / step).round() as i32).max(1);
    let mut best_angle = 0.0f32;
    let mut best_var = -1.0f32;
    // The rotated row-bucket range depends on the corner radius — use the
    // diagonal as an upper bound so every bucket is in range.
    let diag = ((w * w + h * h) as f32).sqrt();
    let n_buckets = (diag.ceil() as usize + 2).max(1);
    let offset = (n_buckets / 2) as i32;
    for i in 0..=n_steps {
        let angle = -max_angle + (i as f32) * (2.0 * max_angle / n_steps as f32);
        let theta = angle.to_radians();
        let (sin_t, cos_t) = theta.sin_cos();
        let mut bins = vec![0u32; n_buckets];
        for &(dx, dy) in &ink_xy {
            let y_rot = -dx * sin_t + dy * cos_t;
            let bucket = (y_rot.round() as i32 + offset).clamp(0, n_buckets as i32 - 1) as usize;
            bins[bucket] = bins[bucket].saturating_add(1);
        }
        let var = histogram_variance(&bins);
        if var > best_var {
            best_var = var;
            best_angle = angle;
        }
    }
    best_angle
}

/// Variance of the per-row ink counts. High variance ↔ a few very
/// "inky" rows and lots of empty rows ↔ well-aligned text.
fn histogram_variance(bins: &[u32]) -> f32 {
    if bins.is_empty() {
        return 0.0;
    }
    let n = bins.len() as f64;
    let mean = bins.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = bins
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var as f32
}

fn build_luma(f: &VideoFrame, params: VideoStreamParams) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return Ok(out);
    }
    let src = &f.planes[0];
    match params.format {
        PixelFormat::Gray8 | PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        PixelFormat::Rgb24 => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 3 * w];
                for x in 0..w {
                    let r = row[3 * x] as u16;
                    let g = row[3 * x + 1] as u16;
                    let b = row[3 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        PixelFormat::Rgba => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                for x in 0..w {
                    let r = row[4 * x] as u16;
                    let g = row[4 * x + 1] as u16;
                    let b = row[4 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "Deskew: cannot derive luma from {other:?}"
            )));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

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
    fn straight_horizontal_lines_estimate_to_zero() {
        // Three horizontal black stripes on white background — already
        // aligned. The histogram-variance scorer should pick 0° (the
        // baseline) as the most variance-maximising angle.
        let input = gray(
            32,
            32,
            |_, y| if y == 8 || y == 16 || y == 24 { 0 } else { 255 },
        );
        let est = Deskew::default()
            .estimate_angle(&input, p_gray(32, 32))
            .unwrap();
        assert!(est.abs() < 1.0, "expected ~0°, got {est}");
    }

    #[test]
    fn estimate_picks_nonzero_for_tilted_lines() {
        // A black diagonal "/\" stripe — the variance-maximising angle
        // is nonzero, so the estimator should not return 0.
        let input = gray(48, 48, |x, y| {
            if y as i32 + (x as i32 / 4) == 24 {
                0
            } else {
                255
            }
        });
        let est = Deskew::default()
            .estimate_angle(&input, p_gray(48, 48))
            .unwrap();
        assert!(est.abs() > 0.5, "estimate should be nonzero, got {est}");
    }

    #[test]
    fn apply_passes_through_for_aligned_input() {
        // For an already-aligned image, the rotation is ~0° and the
        // output has the same shape as the input.
        let input = gray(16, 16, |_, y| if y == 7 { 0 } else { 200 });
        let out = Deskew::default().apply(&input, p_gray(16, 16)).unwrap();
        // The estimate is 0° → Rotate(0°) keeps the same canvas size.
        assert_eq!(out.planes[0].stride, 16);
        assert_eq!(out.planes[0].data.len(), 256);
    }

    #[test]
    fn estimate_zero_for_empty_ink() {
        // Pure-white image — no "ink" pixels survive the threshold, so
        // the estimator returns 0° (no preferred angle).
        let input = gray(16, 16, |_, _| 255);
        let est = Deskew::default()
            .estimate_angle(&input, p_gray(16, 16))
            .unwrap();
        assert_eq!(est, 0.0);
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 4,
            height: 4,
        };
        let err = Deskew::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("Deskew"));
    }

    #[test]
    fn zero_max_angle_returns_zero_estimate() {
        let input = gray(16, 16, |_, y| if y == 5 { 0 } else { 200 });
        let est = Deskew::default()
            .with_max_angle(0.0)
            .estimate_angle(&input, p_gray(16, 16))
            .unwrap();
        assert_eq!(est, 0.0);
    }
}
