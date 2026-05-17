//! Otsu's automatic threshold — binarise a frame by maximising the
//! between-class variance of two intensity classes.
//!
//! Clean-room implementation of the original 1979 paper (Otsu, IEEE
//! Trans. SMC vol. 9): for every possible cut `t ∈ [0, 255]` compute
//!
//! ```text
//! σ²_b(t) = w0(t) · w1(t) · (μ0(t) - μ1(t))²
//! ```
//!
//! and pick the `t` that maximises the inter-class variance. Pixels
//! `< t` map to `0`, pixels `≥ t` map to `255`. Output is always
//! single-plane `Gray8` (RGB / RGBA are collapsed via Rec. 601 luma
//! first; YUV reads the Y plane).
//!
//! Distinct from [`AdaptiveThreshold`](crate::AdaptiveThreshold) which
//! uses a *local* window mean; Otsu is a single global cut whose value
//! is data-driven (so the caller need not pre-pick a level).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Otsu auto-thresholder.
#[derive(Clone, Copy, Debug, Default)]
pub struct OtsuThreshold {
    /// Invert the binarisation. Default `false`: `< t` → 0, `≥ t` → 255.
    /// `true` swaps the two.
    pub invert: bool,
}

impl OtsuThreshold {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Compute the Otsu threshold value for an arbitrary `u8` sample
    /// stream. Exposed so callers can run their own segmentation step
    /// off the same histogram.
    pub fn threshold_of(samples: impl IntoIterator<Item = u8>) -> u8 {
        let mut hist = [0u64; 256];
        let mut total = 0u64;
        for v in samples {
            hist[v as usize] += 1;
            total += 1;
        }
        otsu_from_histogram(&hist, total)
    }
}

fn otsu_from_histogram(hist: &[u64; 256], total: u64) -> u8 {
    if total == 0 {
        return 128;
    }
    // Pre-compute cumulative sum and weighted sum.
    let sum_total: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let mut w_back: f64 = 0.0;
    let mut sum_back: f64 = 0.0;
    // Tie-break midpoint: track first and last `t` that reach the
    // current best variance; pick the midpoint at the end (a textbook
    // Otsu refinement that places the cut in the centre of any flat
    // variance plateau between modes).
    let mut best_var: f64 = -1.0;
    let mut best_lo: u32 = 0;
    let mut best_hi: u32 = 0;
    let total_f = total as f64;
    for t in 0..256u32 {
        w_back += hist[t as usize] as f64;
        if w_back == 0.0 {
            continue;
        }
        let w_fore = total_f - w_back;
        if w_fore == 0.0 {
            break;
        }
        sum_back += (t as f64) * (hist[t as usize] as f64);
        let mu_back = sum_back / w_back;
        let mu_fore = (sum_total - sum_back) / w_fore;
        let var = w_back * w_fore * (mu_back - mu_fore).powi(2);
        if var > best_var {
            best_var = var;
            best_lo = t;
            best_hi = t;
        } else if (var - best_var).abs() < 1e-9 {
            best_hi = t;
        }
    }
    ((best_lo + best_hi) / 2) as u8
}

#[inline]
fn rec601(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000).min(255) as u8
}

impl ImageFilter for OtsuThreshold {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        // Build a Gray8 view of the source, then a histogram from that.
        let src = &input.planes[0];
        let mut grey = vec![0u8; w * h];
        match params.format {
            PixelFormat::Gray8 => {
                for y in 0..h {
                    grey[y * w..(y + 1) * w]
                        .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
                }
            }
            PixelFormat::Rgb24 => {
                for y in 0..h {
                    let row = &src.data[y * src.stride..y * src.stride + 3 * w];
                    let g = &mut grey[y * w..(y + 1) * w];
                    for x in 0..w {
                        g[x] = rec601(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
                    }
                }
            }
            PixelFormat::Rgba => {
                for y in 0..h {
                    let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                    let g = &mut grey[y * w..(y + 1) * w];
                    for x in 0..w {
                        g[x] = rec601(row[4 * x], row[4 * x + 1], row[4 * x + 2]);
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                for y in 0..h {
                    grey[y * w..(y + 1) * w]
                        .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: OtsuThreshold does not yet handle {other:?}"
                )));
            }
        }
        let mut hist = [0u64; 256];
        for &v in &grey {
            hist[v as usize] += 1;
        }
        let t = otsu_from_histogram(&hist, grey.len() as u64);
        let (low, high) = if self.invert {
            (255u8, 0u8)
        } else {
            (0u8, 255u8)
        };
        let mut data = vec![0u8; w * h];
        for (i, &v) in grey.iter().enumerate() {
            data[i] = if v < t { low } else { high };
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride: w, data }],
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
    fn binarises_bimodal_data() {
        // 32 dark pixels at 30, 32 bright pixels at 200 ⇒ Otsu should
        // pick a t somewhere in the middle and binarise to 0 / 255.
        let input = gray(8, 8, |x, _| if x < 4 { 30 } else { 200 });
        let out = OtsuThreshold::new().apply(&input, p_gray(8, 8)).unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 0 || v == 255));
        // Dark half goes to 0, bright half to 255.
        for y in 0..8 {
            for x in 0..4 {
                assert_eq!(out.planes[0].data[y * 8 + x], 0);
            }
            for x in 4..8 {
                assert_eq!(out.planes[0].data[y * 8 + x], 255);
            }
        }
    }

    #[test]
    fn invert_flips_output() {
        let input = gray(8, 8, |x, _| if x < 4 { 30 } else { 200 });
        let out = OtsuThreshold::new()
            .with_invert(true)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for y in 0..8 {
            for x in 0..4 {
                assert_eq!(out.planes[0].data[y * 8 + x], 255);
            }
            for x in 4..8 {
                assert_eq!(out.planes[0].data[y * 8 + x], 0);
            }
        }
    }

    #[test]
    fn threshold_of_returns_value_between_modes() {
        let samples: Vec<u8> = (0..32)
            .map(|_| 30u8)
            .chain((0..32).map(|_| 200u8))
            .collect();
        let t = OtsuThreshold::threshold_of(samples);
        assert!(
            t > 30 && t < 200,
            "Otsu picked t={t}; expected between 30 and 200"
        );
    }

    #[test]
    fn rgb_uses_luma_proxy() {
        // 50%/50% RGB image whose luma is bimodal.
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for _y in 0..4u32 {
            for x in 0..4u32 {
                let val = if x < 2 { 40 } else { 210 };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = OtsuThreshold::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Output is Gray8.
        assert_eq!(out.planes[0].data.len(), 16);
        for y in 0..4 {
            for x in 0..2 {
                assert_eq!(out.planes[0].data[y * 4 + x], 0);
            }
            for x in 2..4 {
                assert_eq!(out.planes[0].data[y * 4 + x], 255);
            }
        }
    }

    #[test]
    fn empty_histogram_defaults_to_128() {
        let t = otsu_from_histogram(&[0u64; 256], 0);
        assert_eq!(t, 128);
    }

    #[test]
    fn flat_histogram_picks_some_threshold() {
        // All same value ⇒ no inter-class variance possible.
        let input = gray(4, 4, |_, _| 128);
        let out = OtsuThreshold::new().apply(&input, p_gray(4, 4)).unwrap();
        // Output may be all 0 or all 255 depending on tie-break; just
        // make sure it's binary and self-consistent.
        let v0 = out.planes[0].data[0];
        assert!(v0 == 0 || v0 == 255);
        assert!(out.planes[0].data.iter().all(|&v| v == v0));
    }
}
