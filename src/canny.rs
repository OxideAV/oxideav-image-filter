//! Canny edge detector (`-canny`).
//!
//! The Canny detector is the textbook four-step pipeline:
//!
//! 1. Smooth the input with a Gaussian to suppress noise.
//! 2. Compute the gradient magnitude and direction (Sobel).
//! 3. Non-maximum suppression along the local gradient direction.
//! 4. Hysteresis thresholding — pixels above `high` are seed edges;
//!    weaker pixels above `low` are kept iff they are 8-connected to a
//!    seed.
//!
//! Clean-room: all four steps are textbook closed-form algorithms,
//! implemented inline. Output is always `Gray8`: `255` for edge
//! pixels, `0` elsewhere. Mirrors ImageMagick's `-canny RxS+L%+H%`
//! (we accept the thresholds in 0..=255 sample space directly so the
//! caller doesn't have to track which percentage convention is in
//! play).
//!
//! # Pixel formats
//!
//! Same as [`Edge`](crate::Edge): any supported input is collapsed
//! to a luma plane up front.

use crate::{is_supported_format, laplacian::build_luma_plane, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Canny edge detector.
#[derive(Clone, Copy, Debug)]
pub struct Canny {
    /// Gaussian pre-blur radius (kernel half-width). `0` skips the
    /// pre-blur step.
    pub radius: u32,
    /// Gaussian σ for the pre-blur. Ignored when `radius == 0`.
    pub sigma: f32,
    /// Lower hysteresis threshold (0..=255).
    pub low: u8,
    /// Upper hysteresis threshold (0..=255). Must be `>= low`.
    pub high: u8,
}

impl Default for Canny {
    fn default() -> Self {
        Self::new(1, 1.0, 30, 90)
    }
}

impl Canny {
    /// Build a Canny detector with the given pre-blur and hysteresis
    /// thresholds. `low` and `high` are swapped if `high < low` so
    /// the caller doesn't have to remember the order.
    pub fn new(radius: u32, sigma: f32, low: u8, high: u8) -> Self {
        let (lo, hi) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };
        Self {
            radius,
            sigma: if sigma > 0.0 { sigma } else { 1.0 },
            low: lo,
            high: hi,
        }
    }
}

impl ImageFilter for Canny {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Canny does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut luma = build_luma_plane(input, params)?;
        if self.radius > 0 {
            luma = gaussian_blur(&luma, w, h, self.radius as usize, self.sigma);
        }
        let (mag, dir) = sobel_with_dir(&luma, w, h);
        let suppressed = non_max_suppress(&mag, &dir, w, h);
        let edges = hysteresis(&suppressed, w, h, self.low, self.high);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: edges,
            }],
        })
    }
}

fn gaussian_blur(src: &[u8], w: usize, h: usize, radius: usize, sigma: f32) -> Vec<u8> {
    let kernel = make_gaussian_kernel(radius, sigma);
    // Horizontal pass.
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, weight) in kernel.iter().enumerate() {
                let xx = (x as i32 + k as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + xx] as f32 * weight;
            }
            tmp[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    // Vertical pass.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, weight) in kernel.iter().enumerate() {
                let yy = (y as i32 + k as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                acc += tmp[yy * w + x] as f32 * weight;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn make_gaussian_kernel(radius: usize, sigma: f32) -> Vec<f32> {
    let n = radius * 2 + 1;
    let mut k = Vec::with_capacity(n);
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = i as f32 - radius as f32;
        let v = (-d * d / s2).exp();
        k.push(v);
        sum += v;
    }
    if sum > 0.0 {
        for v in k.iter_mut() {
            *v /= sum;
        }
    }
    k
}

/// Sobel returns gradient magnitude (0..=≈1442 → clamped to u32) and
/// direction quantised to 4 bins {0=horizontal, 1=±45, 2=vertical, 3=∓45}.
fn sobel_with_dir(luma: &[u8], w: usize, h: usize) -> (Vec<u32>, Vec<u8>) {
    let mut mag = vec![0u32; w * h];
    let mut dir = vec![0u8; w * h];
    if w < 3 || h < 3 {
        return (mag, dir);
    }
    let px = |x: i32, y: i32| -> i32 {
        let cx = x.clamp(0, w as i32 - 1) as usize;
        let cy = y.clamp(0, h as i32 - 1) as usize;
        luma[cy * w + cx] as i32
    };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = -px(x - 1, y - 1) - 2 * px(x - 1, y) - px(x - 1, y + 1)
                + px(x + 1, y - 1)
                + 2 * px(x + 1, y)
                + px(x + 1, y + 1);
            let gy = -px(x - 1, y - 1) - 2 * px(x, y - 1) - px(x + 1, y - 1)
                + px(x - 1, y + 1)
                + 2 * px(x, y + 1)
                + px(x + 1, y + 1);
            let m = ((gx as i64).pow(2) as f64 + (gy as i64).pow(2) as f64).sqrt();
            mag[y as usize * w + x as usize] = m.round() as u32;
            dir[y as usize * w + x as usize] = quantise_angle(gx, gy);
        }
    }
    (mag, dir)
}

/// Quantise (gx, gy) into one of four neighbour-aligned directions:
/// 0 = horizontal (E-W neighbours)
/// 1 = NE-SW diagonal
/// 2 = vertical (N-S neighbours)
/// 3 = NW-SE diagonal
fn quantise_angle(gx: i32, gy: i32) -> u8 {
    if gx == 0 && gy == 0 {
        return 0;
    }
    let angle = (gy as f32).atan2(gx as f32).to_degrees();
    // Normalise to [0, 180).
    let mut a = angle;
    while a < 0.0 {
        a += 180.0;
    }
    while a >= 180.0 {
        a -= 180.0;
    }
    if !(22.5..157.5).contains(&a) {
        0 // horizontal
    } else if a < 67.5 {
        1 // NE-SW
    } else if a < 112.5 {
        2 // vertical
    } else {
        3 // NW-SE
    }
}

fn non_max_suppress(mag: &[u32], dir: &[u8], w: usize, h: usize) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let m = mag[i];
            // Pull the two neighbours along the gradient direction.
            let (a, b) = match dir[i] {
                0 => (mag[i - 1], mag[i + 1]),         // horizontal: W, E
                1 => (mag[i + w - 1], mag[i - w + 1]), // NE-SW: SW, NE
                2 => (mag[i - w], mag[i + w]),         // vertical: N, S
                _ => (mag[i - w - 1], mag[i + w + 1]), // NW-SE: NW, SE
            };
            if m >= a && m >= b {
                out[i] = m;
            }
        }
    }
    out
}

fn hysteresis(mag: &[u32], w: usize, h: usize, low: u8, high: u8) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let lo = low as u32;
    let hi = high as u32;
    // Seed pass — mark strong pixels and stack them for propagation.
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..w * h {
        if mag[i] >= hi {
            out[i] = 255;
            stack.push(i);
        }
    }
    // Propagate to weak (>= low) 8-neighbours.
    while let Some(i) = stack.pop() {
        let x = i % w;
        let y = i / w;
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(w - 1);
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(h - 1);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let j = ny * w + nx;
                if out[j] == 0 && mag[j] >= lo {
                    out[j] = 255;
                    stack.push(j);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

    fn gray(w: u32, h: u32, pat: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pat(x, y));
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

    #[test]
    fn flat_input_has_no_edges() {
        let input = gray(16, 16, |_, _| 100);
        let out = Canny::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn vertical_step_produces_thin_edge() {
        // Half-bright / half-dark image; expect a single column of edge
        // pixels at the boundary after non-max suppression.
        let input = gray(16, 16, |x, _| if x < 8 { 30 } else { 220 });
        let out = Canny::new(0, 1.0, 30, 90)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        // Some interior row should have at least one edge pixel and
        // stay sparse (NMS thins).
        let row = 8usize;
        let edges_in_row: usize = (0..16)
            .filter(|x| out.planes[0].data[row * 16 + x] == 255)
            .count();
        assert!(edges_in_row >= 1, "no edges in row {row}");
        assert!(edges_in_row <= 4, "edge too thick: {edges_in_row}");
    }

    #[test]
    fn output_only_zero_or_255() {
        let input = gray(16, 16, |x, y| (((x ^ y) & 1) * 255) as u8);
        let out = Canny::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "non-binary output: {v}");
        }
    }

    #[test]
    fn high_threshold_suppresses_weak_edges() {
        // A faint gradient → with a very high threshold no edges should
        // survive.
        let input = gray(16, 16, |x, _| 100 + x as u8 / 2);
        let out = Canny::new(0, 1.0, 200, 250)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn rgba_input_emits_gray8_dimensions() {
        let data: Vec<u8> = (0..16 * 16)
            .flat_map(|i| [(i % 255) as u8, (i % 255) as u8, (i % 255) as u8, 200])
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16 * 4,
                data,
            }],
        };
        let out = Canny::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 16);
    }
}
