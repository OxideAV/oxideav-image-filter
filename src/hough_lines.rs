//! Hough-transform line detection (ImageMagick `-hough-lines WxH`).
//!
//! Detects straight lines in a binary edge map and renders them onto a
//! single-plane `Gray8` canvas (black background, white line trace).
//! Algorithm — textbook polar-Hough, clean-room:
//!
//! 1. Reduce input to `Gray8` and run a 3×3 Sobel to get the edge
//!    magnitude. Pixels with magnitude `>= threshold` are voters.
//! 2. Each voter `(x, y)` casts one vote per `theta` bucket
//!    `0..n_theta`, accumulating into the accumulator cell
//!    `(rho_bin, theta_bin)` where
//!    `rho = (x - cx)·cos(θ) + (y - cy)·sin(θ)`.
//! 3. Find the `top_k` largest accumulator cells whose value exceeds
//!    `vote_threshold`. Each survivor is rendered as a full-image line
//!    by walking `rho = x·cos(θ) + y·sin(θ)` across all output pixels.
//!
//! `theta` is quantised to `n_theta` evenly-spaced bins in `[0, π)`;
//! `rho` to `n_rho` bins of width `1.0` covering `±sqrt(w² + h²) / 2`.
//! The defaults (`n_theta = 180`, ½°-bins) match the ImageMagick model.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Hough-line detector.
#[derive(Clone, Debug)]
pub struct HoughLines {
    /// Sobel edge-magnitude threshold; pixels at or above this value
    /// vote. Lower values produce more voters (and more noise).
    pub edge_threshold: u32,
    /// Number of angular bins in `[0, π)`. Defaults to 180 (1° per bin).
    pub n_theta: u32,
    /// Minimum accumulator-cell vote count for a peak to survive.
    pub vote_threshold: u32,
    /// Maximum number of line peaks to draw. The strongest peaks are
    /// kept; ties broken by lower `(rho_bin, theta_bin)` index.
    pub top_k: u32,
}

impl Default for HoughLines {
    fn default() -> Self {
        Self {
            edge_threshold: 64,
            n_theta: 180,
            vote_threshold: 20,
            top_k: 16,
        }
    }
}

impl HoughLines {
    /// Build a Hough-line detector with explicit `edge_threshold` /
    /// `vote_threshold` and the default angular resolution / top-k.
    pub fn new(edge_threshold: u32, vote_threshold: u32) -> Self {
        Self {
            edge_threshold,
            vote_threshold,
            ..Default::default()
        }
    }

    /// Override the angular bin count. Clamped to at least 1.
    pub fn with_n_theta(mut self, n_theta: u32) -> Self {
        self.n_theta = n_theta.max(1);
        self
    }

    /// Override the maximum number of drawn lines.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }
}

impl ImageFilter for HoughLines {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: HoughLines does not yet handle {:?}",
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

        // Step 1: luma + Sobel magnitude (single i32 buffer; full 0..=2040 range).
        let luma = build_luma(input, params)?;
        let mag = sobel_magnitude(&luma, w, h);

        // Step 2: vote into the rho × theta accumulator.
        let n_theta = self.n_theta.max(1) as usize;
        let diag = ((w * w + h * h) as f32).sqrt();
        let r_max = (diag * 0.5).ceil() as i32 + 1;
        let n_rho = (2 * r_max + 1) as usize;
        let cx = (w as f32 - 1.0) * 0.5;
        let cy = (h as f32 - 1.0) * 0.5;
        // Precompute sin / cos.
        let mut sin_t = vec![0.0f32; n_theta];
        let mut cos_t = vec![0.0f32; n_theta];
        for t in 0..n_theta {
            let theta = std::f32::consts::PI * (t as f32) / (n_theta as f32);
            sin_t[t] = theta.sin();
            cos_t[t] = theta.cos();
        }
        let mut acc = vec![0u32; n_rho * n_theta];
        for y in 0..h {
            for x in 0..w {
                if mag[y * w + x] < self.edge_threshold {
                    continue;
                }
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                for t in 0..n_theta {
                    let rho = dx * cos_t[t] + dy * sin_t[t];
                    let bin = (rho.round() as i32 + r_max).clamp(0, n_rho as i32 - 1) as usize;
                    acc[bin * n_theta + t] = acc[bin * n_theta + t].saturating_add(1);
                }
            }
        }

        // Step 3: pick top_k peaks above `vote_threshold`.
        let mut peaks: Vec<(u32, usize, usize)> = Vec::new();
        for r in 0..n_rho {
            for t in 0..n_theta {
                let v = acc[r * n_theta + t];
                if v >= self.vote_threshold {
                    peaks.push((v, r, t));
                }
            }
        }
        peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        peaks.truncate(self.top_k as usize);

        // Step 4: draw each surviving peak as a full-image line.
        for &(_votes, r_bin, t_bin) in &peaks {
            let rho = (r_bin as i32 - r_max) as f32;
            let theta_sin = sin_t[t_bin];
            let theta_cos = cos_t[t_bin];
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r_here = dx * theta_cos + dy * theta_sin;
                    if (r_here - rho).abs() < 0.5 {
                        out_data[y * w + x] = 255;
                    }
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
                "HoughLines: cannot derive luma from {other:?}"
            )));
        }
    }
    Ok(out)
}

/// 3×3 Sobel magnitude as `|Gx| + |Gy|` (max ~ 2040). Borders set to 0.
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
        let p = p_gray(0, 0);
        let out = HoughLines::default().apply(&input, p).unwrap();
        assert_eq!(out.planes[0].data.len(), 0);
    }

    #[test]
    fn flat_image_emits_no_lines() {
        // No edges → no votes → no peaks → output is all zero.
        let input = gray(16, 16, |_, _| 128);
        let out = HoughLines::default().apply(&input, p_gray(16, 16)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn horizontal_band_detected() {
        // A solid horizontal stripe in the middle creates two strong
        // horizontal edges; the Hough peak at θ ≈ π/2 should accumulate
        // a lot of votes and produce a drawn line.
        let input = gray(32, 32, |_, y| if y == 16 { 255 } else { 0 });
        let out = HoughLines::new(64, 8)
            .with_top_k(4)
            .apply(&input, p_gray(32, 32))
            .unwrap();
        // The horizontal peak should paint a row of 255s — count them.
        let nonzero: usize = out.planes[0].data.iter().filter(|&&v| v == 255).count();
        assert!(nonzero >= 16, "expected ≥16 line pixels, got {nonzero}");
    }

    #[test]
    fn output_is_gray8_shape() {
        let input = gray(8, 8, |_, _| 0);
        let out = HoughLines::default().apply(&input, p_gray(8, 8)).unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 8);
        assert_eq!(out.planes[0].data.len(), 64);
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
        let err = HoughLines::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("HoughLines"));
    }

    #[test]
    fn top_k_zero_emits_no_lines() {
        let input = gray(16, 16, |_, y| if y == 8 { 255 } else { 0 });
        let out = HoughLines::default()
            .with_top_k(0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }
}
