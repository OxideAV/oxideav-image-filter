//! Hough-transform circle detection (complement to [`HoughLines`](crate::HoughLines)).
//!
//! Detects circles in a binary edge map and renders their outlines onto
//! a single-plane `Gray8` canvas (black background, white circle trace).
//! Algorithm — textbook 3-D parameter-space Hough, clean-room:
//!
//! 1. Reduce the input to `Gray8` and run a 3×3 Sobel to obtain the edge
//!    magnitude. Any pixel with magnitude `>= edge_threshold` becomes a
//!    voter.
//! 2. For each voter `(x, y)` and each candidate radius
//!    `r ∈ [min_radius, max_radius]`, accumulate a vote into every
//!    `(cx, cy)` cell whose distance to `(x, y)` equals `r` (i.e. trace
//!    the full Bresenham circle of radius `r` centred on the voter into
//!    a 3-D accumulator indexed by `(r, cx, cy)`).
//! 3. Pick the `top_k` largest accumulator cells whose vote count
//!    exceeds `vote_threshold`. Each survivor `(r, cx, cy)` is rendered
//!    by drawing a 1-pixel-thick circle on the output canvas.
//!
//! The 3-D accumulator is `(max_radius - min_radius + 1) × w × h` cells —
//! quadratic in image size, linear in radius range. For practical inputs
//! the call is `O(N_voters · (max_radius - min_radius + 1) · 2π · r̄)`.
//!
//! Pixel formats: same `Gray8` / `Rgb24` / `Rgba` / planar-YUV set as
//! [`HoughLines`]; output is always single-plane `Gray8`.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Hough-circle detector.
#[derive(Clone, Debug)]
pub struct HoughCircles {
    /// Sobel edge-magnitude threshold (`|Gx| + |Gy|`); pixels at or
    /// above this value vote.
    pub edge_threshold: u32,
    /// Minimum candidate radius (inclusive). Clamped to `>= 1`.
    pub min_radius: u32,
    /// Maximum candidate radius (inclusive). Must satisfy
    /// `max_radius >= min_radius`.
    pub max_radius: u32,
    /// Minimum accumulator vote count for a peak to survive.
    pub vote_threshold: u32,
    /// Maximum number of circle peaks to render. Strongest peaks are
    /// kept; ties broken by lower `(r, cy, cx)`.
    pub top_k: u32,
}

impl Default for HoughCircles {
    fn default() -> Self {
        Self {
            edge_threshold: 64,
            min_radius: 4,
            max_radius: 32,
            vote_threshold: 16,
            top_k: 8,
        }
    }
}

impl HoughCircles {
    /// Build a Hough-circle detector with explicit radius range and
    /// vote thresholds.
    pub fn new(min_radius: u32, max_radius: u32, vote_threshold: u32) -> Self {
        Self {
            min_radius: min_radius.max(1),
            max_radius: max_radius.max(min_radius.max(1)),
            vote_threshold,
            ..Default::default()
        }
    }

    /// Override the Sobel edge threshold.
    pub fn with_edge_threshold(mut self, t: u32) -> Self {
        self.edge_threshold = t;
        self
    }

    /// Override the maximum number of rendered circles.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }
}

impl ImageFilter for HoughCircles {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: HoughCircles does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut out_data = vec![0u8; w * h];
        if w == 0 || h == 0 {
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: w,
                    data: out_data,
                }],
            });
        }
        let min_r = self.min_radius.max(1) as i32;
        let max_r = self.max_radius.max(self.min_radius.max(1)) as i32;
        if max_r < min_r {
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: w,
                    data: out_data,
                }],
            });
        }
        let r_count = (max_r - min_r + 1) as usize;

        // Step 1: build luma + Sobel.
        let luma = build_luma(input, params)?;
        let mag = sobel_magnitude(&luma, w, h);

        // Step 2: precompute Bresenham circle offsets for each radius.
        let mut offsets_per_r: Vec<Vec<(i32, i32)>> = Vec::with_capacity(r_count);
        for r in min_r..=max_r {
            offsets_per_r.push(bresenham_circle(r));
        }

        // Step 3: 3-D accumulator `(r, cy, cx)` packed as 1-D.
        let mut acc = vec![0u32; r_count * w * h];
        for y in 0..h {
            for x in 0..w {
                if mag[y * w + x] < self.edge_threshold {
                    continue;
                }
                for (ri, offsets) in offsets_per_r.iter().enumerate() {
                    let base = ri * w * h;
                    for &(dx, dy) in offsets {
                        let cx = x as i32 + dx;
                        let cy = y as i32 + dy;
                        if cx >= 0 && cx < w as i32 && cy >= 0 && cy < h as i32 {
                            let idx = base + (cy as usize) * w + (cx as usize);
                            acc[idx] = acc[idx].saturating_add(1);
                        }
                    }
                }
            }
        }

        // Step 4: pick top_k peaks.
        let mut peaks: Vec<(u32, i32, usize, usize)> = Vec::new();
        for ri in 0..r_count {
            for cy in 0..h {
                for cx in 0..w {
                    let v = acc[ri * w * h + cy * w + cx];
                    if v >= self.vote_threshold {
                        peaks.push((v, min_r + ri as i32, cx, cy));
                    }
                }
            }
        }
        peaks.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(a.1.cmp(&b.1))
                .then(a.3.cmp(&b.3))
                .then(a.2.cmp(&b.2))
        });
        peaks.truncate(self.top_k as usize);

        // Step 5: render each peak.
        for (_, r, cx, cy) in peaks {
            let offsets = bresenham_circle(r);
            for (dx, dy) in offsets {
                let px = cx as i32 + dx;
                let py = cy as i32 + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    out_data[(py as usize) * w + (px as usize)] = 255;
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out_data,
            }],
        })
    }
}

fn build_luma(f: &VideoFrame, params: VideoStreamParams) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];
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
                "HoughCircles: cannot derive luma from {other:?}"
            )));
        }
    }
    Ok(out)
}

fn sobel_magnitude(luma: &[u8], w: usize, h: usize) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p00 = luma[(y - 1) * w + (x - 1)] as i32;
            let p01 = luma[(y - 1) * w + x] as i32;
            let p02 = luma[(y - 1) * w + (x + 1)] as i32;
            let p10 = luma[y * w + (x - 1)] as i32;
            let p12 = luma[y * w + (x + 1)] as i32;
            let p20 = luma[(y + 1) * w + (x - 1)] as i32;
            let p21 = luma[(y + 1) * w + x] as i32;
            let p22 = luma[(y + 1) * w + (x + 1)] as i32;
            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
            out[y * w + x] = gx.unsigned_abs() + gy.unsigned_abs();
        }
    }
    out
}

/// Bresenham midpoint circle generator. Returns the eight-symmetry
/// expanded set of `(dx, dy)` offsets from the centre that trace a
/// 1-pixel-thick circle of radius `r`.
fn bresenham_circle(r: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    if r <= 0 {
        return out;
    }
    let mut x = 0i32;
    let mut y = r;
    let mut d = 1 - r;
    while x <= y {
        for &(dx, dy) in &[
            (x, y),
            (-x, y),
            (x, -y),
            (-x, -y),
            (y, x),
            (-y, x),
            (y, -x),
            (-y, -x),
        ] {
            out.push((dx, dy));
        }
        if d < 0 {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y -= 1;
        }
        x += 1;
    }
    out.sort();
    out.dedup();
    out
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
    fn empty_input_no_panics() {
        let input = gray(0, 0, |_, _| 0);
        let out = HoughCircles::default().apply(&input, p_gray(0, 0)).unwrap();
        assert_eq!(out.planes[0].data.len(), 0);
    }

    #[test]
    fn flat_image_emits_no_circles() {
        let input = gray(24, 24, |_, _| 128);
        let out = HoughCircles::default()
            .apply(&input, p_gray(24, 24))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn detects_drawn_circle() {
        // Draw a single circle of radius 6 centred at (12, 12) on a
        // 24×24 black canvas. Then ask the detector to find it.
        let mut data = vec![0u8; 24 * 24];
        let offsets = bresenham_circle(6);
        for (dx, dy) in offsets {
            let px = 12 + dx;
            let py = 12 + dy;
            if (0..24).contains(&px) && (0..24).contains(&py) {
                data[(py as usize) * 24 + (px as usize)] = 255;
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = HoughCircles::new(4, 8, 4)
            .with_top_k(1)
            .apply(&input, p_gray(24, 24))
            .unwrap();
        // At least some output pixels should be non-zero (the detector
        // re-renders the circle).
        let nonzero: usize = out.planes[0].data.iter().filter(|&&v| v == 255).count();
        assert!(nonzero >= 8, "expected ≥8 detected pixels, got {nonzero}");
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
        let err = HoughCircles::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("HoughCircles"));
    }

    #[test]
    fn bresenham_circle_is_eight_symmetric() {
        let pts = bresenham_circle(5);
        // For a positive radius the trace must visit at least the 4
        // cardinal extremes (±r, 0) / (0, ±r).
        assert!(pts.contains(&(5, 0)));
        assert!(pts.contains(&(-5, 0)));
        assert!(pts.contains(&(0, 5)));
        assert!(pts.contains(&(0, -5)));
    }
}
