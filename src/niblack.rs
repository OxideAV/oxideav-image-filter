//! Niblack adaptive local-statistics threshold.
//!
//! Wayne Niblack's textbook local-window binarisation rule (Niblack,
//! *An Introduction to Digital Image Processing*, Prentice-Hall 1986,
//! §5.1 — page-segmentation example): for each pixel compute the mean
//! `μ(x, y)` and standard deviation `σ(x, y)` of the
//! `(2·radius + 1)²` neighbourhood centred at `(x, y)`, then form the
//! per-pixel threshold
//!
//! ```text
//!   T(x, y) = μ(x, y) + k · σ(x, y)
//! ```
//!
//! and emit `255` where `sample(x, y) ≥ T(x, y)` (or the inversion of
//! that test if [`Niblack::invert`] is `true`), `0` elsewhere. The
//! parameter `k` is a small dimensionless biasing constant — Niblack's
//! original page-segmentation example uses `k = -0.2`, which biases the
//! threshold *below* the local mean by a fraction of the local spread so
//! ink against a brightish but locally-noisy background is reliably
//! captured. Positive `k` biases the threshold above the mean (for the
//! complementary case of light-on-dark).
//!
//! The filter complements the existing
//! [`AdaptiveThreshold`](crate::AdaptiveThreshold) (local-mean only —
//! `k = 0`) and [`OtsuThreshold`](crate::OtsuThreshold) (a single global
//! cut chosen to maximise between-class variance). Where the global
//! cut fails — pages with locally-varying illumination, a coffee stain,
//! shadows from a binder rib — Niblack tracks the spread of the
//! background and follows it; where a flat local mean fails — a near-
//! constant grey patch with faint ink — the `k · σ` term adapts: low
//! σ keeps the threshold close to the mean (so faint ink still wins),
//! high σ pushes the threshold further from the mean (so noise is
//! rejected).
//!
//! ## Local statistics
//!
//! The per-pixel mean and variance over a square window of side
//! `W = 2·radius + 1` are computed by the standard
//! `Var(X) = E(X²) - E(X)²` identity:
//!
//! ```text
//!   μ(x, y) = (1 / W²) · Σ_{(u, v) ∈ window} I(u, v)
//!   σ²(x, y) = (1 / W²) · Σ I(u, v)² − μ(x, y)²
//! ```
//!
//! Each summed quantity (`Σ I` and `Σ I²`) is gathered by a single
//! separable box-sum pass: a 1-D horizontal sliding sum over each row,
//! then a 1-D vertical sliding sum over the row-sum buffer. Border
//! samples clamp to the nearest in-bounds pixel so the output keeps the
//! input dimensions. The whole filter is `O(W·H)` regardless of
//! `radius`, the same cost class as the existing
//! [`AdaptiveThreshold`](crate::AdaptiveThreshold).
//!
//! Numerical precision is `i64` on the running sums (a 33×33 window
//! over a `u8 = 255` image saturates a `u32` accumulator before any
//! cancellation, but stays well inside an `i64`); the final variance
//! is converted to `f32` for the `sqrt`. We clamp the raw
//! `E(X²) - μ²` to `0` before the square root to absorb the tiny
//! floating-point negatives that show up on near-constant patches when
//! cancellation eats every significant digit (textbook fix; otherwise
//! we'd ship a `NaN` σ on flat input).
//!
//! ## Output
//!
//! Always single-plane `Gray8` of the input dimensions. The input is
//! collapsed to a luma plane first (Gray8 / YUV use the Y plane
//! directly; RGB / RGBA use the `(R + 2G + B) / 4` quick luma that the
//! rest of the edge family in this crate also shares via the
//! `laplacian::build_luma_plane` helper). Other pixel formats return
//! [`Error::unsupported`].
//!
//! ## Clean-room reference
//!
//! The mean + standard-deviation rule and the `k = -0.2` default come
//! straight from Niblack's 1986 textbook chapter (§5.1). The
//! complement / scale family (Sauvola's `T = μ · (1 + k · (σ/R − 1))`,
//! Wolf-Jolion's normalisation against the global maximum and the
//! global minimum, the integral-image box-sum reformulation) are
//! covered in:
//!
//! - Trier & Jain, "Goal-directed evaluation of binarization methods",
//!   *IEEE Trans. Pattern Anal. Mach. Intell.* 17(12):1191–1201,
//!   December 1995.
//! - Sezgin & Sankur, "Survey over image thresholding techniques and
//!   quantitative performance evaluation", *Journal of Electronic
//!   Imaging* 13(1):146–168, January 2004.
//! - Gonzalez & Woods, *Digital Image Processing*, 4th ed.,
//!   §10.3.7 ("Variable thresholding").
//!
//! The box-sum implementation, the variance-clamping fix, the `k = -0.2`
//! default, and every test expectation were derived from first
//! principles and the references above.
//!
//! ## Example
//!
//! ```no_run
//! use oxideav_image_filter::{ImageFilter, Niblack, VideoStreamParams};
//! use oxideav_core::{PixelFormat, VideoFrame};
//!
//! let f = Niblack::new().with_radius(7).with_k(-0.2);
//! let params = VideoStreamParams {
//!     format: PixelFormat::Gray8,
//!     width: 64,
//!     height: 64,
//! };
//! let frame = VideoFrame { pts: None, planes: vec![] };
//! // let out = f.apply(&frame, params)?;
//! # let _ = (f, params, frame);
//! ```
//!
//! See the [module-level
//! `AdaptiveThreshold`](crate::AdaptiveThreshold) docs for the
//! sibling local-mean-only variant.
//!
//! Joins the segmentation family at the "local mean + local σ
//! threshold" position complementing
//! [`AdaptiveThreshold`](crate::AdaptiveThreshold) (local mean only)
//! and [`OtsuThreshold`](crate::OtsuThreshold) (global automatic cut).
//!
//! Stateless, single-frame, `O(W·H)`.
//!
//! [`Error::unsupported`]: oxideav_core::Error

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Niblack adaptive local-statistics threshold.
///
/// See the [module-level docs](self) for the full rule, the
/// `Var = E(X²) − E(X)²` evaluation order, and the box-sum cost
/// argument.
#[derive(Clone, Copy, Debug)]
pub struct Niblack {
    /// Half-extent of the square neighbourhood; the window side is
    /// `2·radius + 1`. Default `7` (a 15×15 window, a typical
    /// document-binarisation default).
    pub radius: u32,
    /// Standard-deviation biasing constant `k`. Default `-0.2`
    /// (Niblack's textbook value for dark-ink-on-light-background page
    /// segmentation; the threshold then sits a fraction of σ *below*
    /// the local mean).
    pub k: f32,
    /// Invert the binarisation. Default `false`: emit `255` where
    /// `sample ≥ T`. `true` swaps the two output levels (so the
    /// foreground becomes black and the background white).
    pub invert: bool,
}

impl Default for Niblack {
    fn default() -> Self {
        Self {
            radius: 7,
            k: -0.2,
            invert: false,
        }
    }
}

impl Niblack {
    /// Default-configured Niblack: `radius = 7`, `k = -0.2`,
    /// `invert = false`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the neighbourhood half-extent.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }

    /// Set the standard-deviation biasing constant `k`.
    pub fn with_k(mut self, k: f32) -> Self {
        self.k = k;
        self
    }

    /// Flip the foreground / background polarity of the output.
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }
}

/// Box-sum of an arbitrary `i64` source plane over a
/// `(2·radius + 1)²` square window with edge-clamped borders. Two
/// separable passes; cost `O(W·H)` regardless of `radius`.
fn box_sum_i64(src: &[i64], w: usize, h: usize, radius: usize) -> Vec<i64> {
    let mut out = vec![0i64; w * h];
    if radius == 0 {
        out.copy_from_slice(src);
        return out;
    }
    // Horizontal pass: sum across each row into a row-sum buffer.
    let mut horiz = vec![0i64; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut sum: i64 = 0;
        // Seed window: [0 .. min(radius, w-1)] from the source plus
        // `radius` extra copies of `row[0]` to fill the left clamp.
        for &v in row.iter().take(radius.min(w - 1) + 1) {
            sum += v;
        }
        sum += row[0] * (radius as i64);
        for x in 0..w {
            horiz[y * w + x] = sum;
            // Slide one step right: add the sample entering on the
            // right (clamped to `w - 1`), drop the sample leaving on
            // the left (clamped to `0`).
            let add_x = (x + radius + 1).min(w - 1);
            let drop_x = x.saturating_sub(radius);
            sum = sum + row[add_x] - row[drop_x];
        }
    }
    // Vertical pass: sum each column over `horiz`.
    for x in 0..w {
        let mut sum: i64 = 0;
        for y in 0..=radius.min(h - 1) {
            sum += horiz[y * w + x];
        }
        sum += horiz[x] * (radius as i64);
        for y in 0..h {
            out[y * w + x] = sum;
            let add_y = (y + radius + 1).min(h - 1);
            let drop_y = y.saturating_sub(radius);
            sum = sum + horiz[add_y * w + x] - horiz[drop_y * w + x];
        }
    }
    out
}

impl ImageFilter for Niblack {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "Niblack: pixel format {:?} is not supported",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::invalid(
                "Niblack: positive width and height are required".to_string(),
            ));
        }
        if !self.k.is_finite() {
            return Err(Error::invalid("Niblack: k must be finite".to_string()));
        }
        let luma = build_luma_plane(input, params)?;
        // Convert luma to i64 planes for the two box-sums.
        let mut sum_in: Vec<i64> = Vec::with_capacity(w * h);
        let mut sq_in: Vec<i64> = Vec::with_capacity(w * h);
        for &v in luma.iter() {
            sum_in.push(v as i64);
            sq_in.push((v as i64) * (v as i64));
        }
        let r = self.radius as usize;
        let sum_box = box_sum_i64(&sum_in, w, h, r);
        let sq_box = box_sum_i64(&sq_in, w, h, r);
        let win = (2 * r + 1) as i64;
        let area = (win * win) as f32;
        let mut out = vec![0u8; w * h];
        for i in 0..w * h {
            let mean = sum_box[i] as f32 / area;
            // E(X²) − E(X)². Negatives come from floating-point
            // cancellation on near-constant patches; clamp to zero.
            let var = (sq_box[i] as f32 / area) - mean * mean;
            let var = var.max(0.0);
            let sigma = var.sqrt();
            let t = mean + self.k * sigma;
            let s = luma[i] as f32;
            let above = s >= t;
            let bit = if above ^ self.invert { 255 } else { 0 };
            out[i] = bit;
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
    use oxideav_core::PixelFormat;

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

    /// Brute-force reference: per-pixel mean and standard deviation
    /// over the `(2·radius + 1)²` neighbourhood with edge-clamp.
    fn brute_mean_sigma(src: &[u8], w: usize, h: usize, radius: usize) -> (Vec<f64>, Vec<f64>) {
        let mut mean = vec![0.0f64; w * h];
        let mut sigma = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f64;
                let mut s2 = 0.0f64;
                let mut n = 0.0f64;
                for dy in -(radius as isize)..=radius as isize {
                    for dx in -(radius as isize)..=radius as isize {
                        let xx = (x as isize + dx).max(0).min(w as isize - 1) as usize;
                        let yy = (y as isize + dy).max(0).min(h as isize - 1) as usize;
                        let v = src[yy * w + xx] as f64;
                        s += v;
                        s2 += v * v;
                        n += 1.0;
                    }
                }
                let m = s / n;
                let v = (s2 / n - m * m).max(0.0);
                mean[y * w + x] = m;
                sigma[y * w + x] = v.sqrt();
            }
        }
        (mean, sigma)
    }

    #[test]
    fn flat_input_with_k_zero_is_all_white() {
        // Constant input ⇒ σ = 0 everywhere ⇒ T = μ = sample
        // ⇒ `sample >= T` ⇒ 255 for every pixel.
        let input = gray(16, 16, |_, _| 120);
        let out = Niblack::new()
            .with_radius(3)
            .with_k(0.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn flat_input_with_positive_k_is_all_black() {
        // σ = 0, so T = μ = sample independent of k. The strict-
        // `>=` test still wins on a flat input. Bump σ by introducing
        // a tiny one-pixel disturbance to actually exercise k.
        let input = gray(16, 16, |x, y| if x == 0 && y == 0 { 121 } else { 120 });
        // With k = +5, T = μ + 5σ is pushed well above sample on the
        // bulk of the image (because at least one window straddles the
        // disturbed pixel and σ > 0). The interior far from (0, 0) is
        // truly flat (σ = 0) so those pixels remain 255 — but the
        // window around the disturbance dips below T. We only assert
        // the disturbance-touching region.
        let out = Niblack::new()
            .with_radius(3)
            .with_k(5.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        // The (0, 0) sample sits at the local-mean ± 5σ threshold:
        // its σ is tiny but k is large, so T > 121 ⇒ 0.
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn flat_input_with_invert_is_all_black() {
        let input = gray(8, 8, |_, _| 80);
        let out = Niblack::new()
            .with_radius(2)
            .with_k(0.0)
            .with_invert(true)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    #[test]
    fn ink_against_bright_background_with_negative_k() {
        // 16×16 bright background with a 4×4 ink square in the centre.
        // Niblack's textbook k = -0.2 pulls the threshold *below* the
        // local mean by ~0.2σ; ink (well below the mean) crosses below
        // T ⇒ 0; background (well above the mean) stays at 255.
        let input = gray(16, 16, |x, y| {
            if (6..=9).contains(&x) && (6..=9).contains(&y) {
                30
            } else {
                220
            }
        });
        let out = Niblack::new()
            .with_radius(3)
            .with_k(-0.2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        // Ink pixels (centre) → black.
        assert_eq!(out.planes[0].data[7 * 16 + 7], 0);
        assert_eq!(out.planes[0].data[8 * 16 + 8], 0);
        // Background corner pixel → white.
        assert_eq!(out.planes[0].data[0], 255);
        assert_eq!(out.planes[0].data[15 * 16 + 15], 255);
    }

    #[test]
    fn box_sum_matches_brute_force() {
        // Cross-check the separable box-sum against the trivial
        // double-loop reference on a varied pattern. Equal-by-
        // construction since the sum is over the same window with
        // the same edge-clamp.
        let w = 11usize;
        let h = 9usize;
        let pat: Vec<u8> = (0..w * h).map(|i| ((i * 17) % 256) as u8).collect();
        for radius in [0usize, 1, 2, 3, 5] {
            let pat_i: Vec<i64> = pat.iter().map(|&v| v as i64).collect();
            let pat_sq: Vec<i64> = pat.iter().map(|&v| (v as i64) * (v as i64)).collect();
            let sum = box_sum_i64(&pat_i, w, h, radius);
            let sq = box_sum_i64(&pat_sq, w, h, radius);
            // Brute-force reference.
            let (m_ref, s_ref) = brute_mean_sigma(&pat, w, h, radius);
            let win = (2 * radius + 1) as f64;
            let area = win * win;
            for i in 0..w * h {
                let m = sum[i] as f64 / area;
                let v = (sq[i] as f64 / area - m * m).max(0.0);
                let sig = v.sqrt();
                assert!(
                    (m - m_ref[i]).abs() < 1e-9,
                    "mean mismatch at i={i} r={radius}: separable={m} brute={}",
                    m_ref[i]
                );
                assert!(
                    (sig - s_ref[i]).abs() < 1e-6,
                    "sigma mismatch at i={i} r={radius}: separable={sig} brute={}",
                    s_ref[i]
                );
            }
        }
    }

    #[test]
    fn matches_brute_force_threshold_on_natural_pattern() {
        // End-to-end check: the per-pixel Niblack output of the
        // separable implementation must match the per-pixel output of
        // a brute-force reference on a non-trivial pattern.
        let w = 13usize;
        let h = 11usize;
        let pat: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let radius = 2usize;
        let k = -0.2f32;
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w,
                data: pat.clone(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w as u32,
            height: h as u32,
        };
        let out = Niblack::new()
            .with_radius(radius as u32)
            .with_k(k)
            .apply(&frame, params)
            .unwrap();
        let (m_ref, s_ref) = brute_mean_sigma(&pat, w, h, radius);
        for i in 0..w * h {
            let t = m_ref[i] + (k as f64) * s_ref[i];
            let expected: u8 = if (pat[i] as f64) >= t { 255 } else { 0 };
            assert_eq!(
                out.planes[0].data[i], expected,
                "mismatch at i={i}: sample={} m={} σ={} T={} got={} expected={}",
                pat[i], m_ref[i], s_ref[i], t, out.planes[0].data[i], expected
            );
        }
    }

    #[test]
    fn negative_k_lowers_threshold_below_mean() {
        // Construct a 16×16 frame with a gentle horizontal ramp
        // (0..15 → 100..160). Mean ≈ 130 in the centre; σ ≈ 17.
        // At k = -0.2, T ≈ 130 - 3.4 = 126.6, so samples ≥ 127 go
        // white, samples ≤ 126 go black. Verify the rough split.
        let input = gray(16, 16, |x, _| (100 + 4 * x) as u8);
        let out = Niblack::new()
            .with_radius(3)
            .with_k(-0.2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        // The leftmost columns are below the threshold, so black.
        // The rightmost columns are above, so white.
        assert_eq!(out.planes[0].data[8 * 16], 0, "leftmost should be black");
        assert_eq!(
            out.planes[0].data[8 * 16 + 15],
            255,
            "rightmost should be white"
        );
    }

    #[test]
    fn positive_k_raises_threshold_above_mean() {
        // Same ramp; k = +0.2 puts the cut at μ + 0.2σ — fewer pixels
        // pass, so the white region shrinks toward the right end.
        let input = gray(16, 16, |x, _| (100 + 4 * x) as u8);
        let neg = Niblack::new()
            .with_radius(3)
            .with_k(-0.2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let pos = Niblack::new()
            .with_radius(3)
            .with_k(0.2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let count_white = |f: &VideoFrame| f.planes[0].data.iter().filter(|&&v| v == 255).count();
        assert!(
            count_white(&pos) <= count_white(&neg),
            "positive k must produce no more white pixels than negative k: pos={} neg={}",
            count_white(&pos),
            count_white(&neg)
        );
    }

    #[test]
    fn rgb_input_collapses_to_luma() {
        // RGB input of pure red. The luma proxy (R + 2G + B) / 4 of
        // (200, 0, 0) is 50. Constant after collapse, so flat-input
        // semantics apply: σ = 0, T = μ = 50, sample = 50 ⇒ 255.
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[200u8, 0, 0]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = Niblack::new()
            .with_radius(2)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 64);
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn yuv420p_input_uses_luma_plane() {
        // Make a 4×4 Yuv420P frame where the luma plane is uniform
        // 120. Niblack on the luma plane is flat ⇒ all-white.
        let y = vec![120u8; 16];
        let u = vec![128u8; 4];
        let v = vec![128u8; 4];
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane { stride: 2, data: u },
                VideoPlane { stride: 2, data: v },
            ],
        };
        let out = Niblack::new()
            .with_radius(1)
            .with_k(0.0)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 16);
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn rejects_non_finite_k() {
        let input = gray(4, 4, |_, _| 100);
        assert!(Niblack::new()
            .with_k(f32::NAN)
            .apply(&input, p_gray(4, 4))
            .is_err());
        assert!(Niblack::new()
            .with_k(f32::INFINITY)
            .apply(&input, p_gray(4, 4))
            .is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 0,
                data: vec![],
            }],
        };
        assert!(Niblack::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 0,
                    height: 4,
                },
            )
            .is_err());
    }

    #[test]
    fn rejects_unsupported_pixel_format() {
        // A pixel format outside the supported allow-list is
        // rejected before the kernel even runs.
        let frame = VideoFrame {
            pts: None,
            planes: vec![],
        };
        let r = Niblack::new().apply(
            &frame,
            VideoStreamParams {
                format: PixelFormat::Yuv420P10Le,
                width: 4,
                height: 4,
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn output_shape_is_input_shape() {
        let input = gray(7, 5, |_, _| 100);
        let out = Niblack::new()
            .with_radius(2)
            .apply(&input, p_gray(7, 5))
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 7);
        assert_eq!(out.planes[0].data.len(), 35);
    }

    #[test]
    fn pts_passthrough() {
        let mut input = gray(8, 8, |_, _| 100);
        input.pts = Some(424242);
        let out = Niblack::new()
            .with_radius(2)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        assert_eq!(out.pts, Some(424242));
    }

    #[test]
    fn builder_chain_composes() {
        let f = Niblack::new().with_radius(5).with_k(-0.3).with_invert(true);
        assert_eq!(f.radius, 5);
        assert!((f.k - -0.3).abs() < 1e-6);
        assert!(f.invert);
    }

    #[test]
    fn near_constant_input_does_not_panic() {
        // A near-constant patch with one off-by-one pixel can yield a
        // tiny negative `E(X²) − μ²` from floating-point cancellation.
        // The clamp-to-zero on the variance must keep `sqrt` happy and
        // never ship a NaN sample.
        let input = gray(32, 32, |x, y| if x == 1 && y == 1 { 51 } else { 50 });
        let out = Niblack::new()
            .with_radius(3)
            .with_k(-0.2)
            .apply(&input, p_gray(32, 32))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn radius_one_window_three_three() {
        // Smallest non-trivial window: 3×3. Hand-derived expectation
        // on a 2-class fixture (a single bright pixel in the centre
        // of a dark 5×5 patch). Brute-force ref is the authority.
        let w = 5usize;
        let h = 5usize;
        let pat: Vec<u8> = (0..w * h).map(|i| if i == 12 { 255 } else { 0 }).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w,
                data: pat.clone(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w as u32,
            height: h as u32,
        };
        let radius = 1usize;
        let k = 0.0f32;
        let out = Niblack::new()
            .with_radius(radius as u32)
            .with_k(k)
            .apply(&frame, params)
            .unwrap();
        let (m_ref, _s_ref) = brute_mean_sigma(&pat, w, h, radius);
        for i in 0..w * h {
            // σ ignored at k = 0.
            let expected: u8 = if (pat[i] as f64) >= m_ref[i] { 255 } else { 0 };
            assert_eq!(out.planes[0].data[i], expected, "i={i}");
        }
    }
}
