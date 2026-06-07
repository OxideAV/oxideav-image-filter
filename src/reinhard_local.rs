//! Local "dodging-and-burning" Reinhard tone-mapping operator.
//!
//! Reinhard, Stark, Shirley, Ferwerda — "Photographic Tone Reproduction
//! for Digital Images" (SIGGRAPH 2002), §3 "Automatic Dodging-and-Burning".
//! Implements the local variant of the global operator (see
//! [`crate::Reinhard`] / [`crate::ReinhardExtended`]) in which the
//! denominator `1 + L` of the global curve is replaced by a
//! **spatially-varying local average** chosen per-pixel from a
//! geometric Gaussian scale pyramid. The largest scale at which the
//! centre/surround neighbourhood is still "locally uniform" is selected;
//! a high-contrast edge truncates the search to the smallest scale,
//! preserving local detail the way a photographer's dodge-and-burn does.
//!
//! See `docs/image/filter/tone-mapping-operators.md` §3.
//!
//! Algorithm (clean-room from the reference doc):
//!
//! 1. De-gamma sRGB → linear-light per channel.
//! 2. Compute scene luminance `L` per pixel using the Rec.709
//!    `L = 0.2126·R + 0.7152·G + 0.0722·B` weights.
//! 3. Apply key-scaling so the log-average luminance maps to the
//!    middle-grey key `a` (§2.1):
//!    `L_s(x,y) = (a / L̄_w) · L(x,y)`, where
//!    `L̄_w = exp((1/N) · Σ ln(δ + L))`.
//! 4. For each scale `s_k = s_0 · 1.6^k` in a geometric pyramid build
//!    two Gaussian-blurred versions of `L_s`:
//!    - `V1(·,s)` — centre, scale `α_1·s` with `α_1 = 1/(2√2)`.
//!    - `V2(·,s)` — surround, scale `α_2·s` with `α_2 = 1.6·α_1`.
//!
//!    Convolution with the circularly-symmetric Gaussian profile
//!    `R_i(x,y,s) = (1/(π(α_i s)²)) · exp(−(x²+y²)/(α_i s)²)`; the
//!    constant factor `1/(π·…²)` normalises the kernel to unit weight.
//! 5. Form the normalised centre-surround difference
//!    `V(·,s) = (V1 − V2) / (2^φ · a / s² + V1)` with `φ = 8` (the §3.2
//!    sharpening default). The `2^φ · a / s²` term keeps `V` well-behaved
//!    in dark regions.
//! 6. Pick the **largest** scale index `m(x,y)` such that
//!    `|V(x,y, s_{m})| < ε` (`ε = 0.05` default). If every scale already
//!    exceeds `ε` the smallest scale is used (truncating to the finest
//!    available local mean).
//! 7. Apply the local tone curve (§3.4):
//!    `Ld(x,y) = L_s(x,y) / (1 + V1(x,y, s_{m(x,y)}))`.
//! 8. Re-modulate the original linear RGB by `Ld / L_w` (chroma
//!    preservation, paper §3.4 final paragraph).
//! 9. Re-encode through the sRGB OETF.
//!
//! Operates on `Rgb24` / `Rgba`. `Gray8` falls back to the scalar form
//! (no chroma preservation; the curve is applied to the single channel).
//! YUV returns `Unsupported` (same constraint as `Reinhard` /
//! `ReinhardExtended`).
//!
//! Cost is `O(W·H·K)` where `K` is the number of scales; the default
//! `K = 8` matches the §3.3 recommendation. Each scale runs a separable
//! Gaussian, so the per-scale cost is linear in the kernel radius.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Local Reinhard "dodging-and-burning" tone-mapping operator
/// (Reinhard et al. 2002 §3).
///
/// The scale-pyramid parameters default to the paper's recommendation:
/// eight scales with a geometric ratio of `1.6`, the smallest centre
/// radius derived from `α_1 = 1/(2√2)` and `s_0 = 1.0`. The
/// per-pixel scale selection is controlled by `phi` (sharpening, §3.2)
/// and `epsilon` (§3.3 uniformity threshold).
#[derive(Clone, Copy, Debug)]
pub struct ReinhardLocal {
    /// Reinhard middle-grey key `a` (§2.1). `0.18` is the photographic
    /// middle-grey default; lower values preserve shadows (low-key),
    /// higher compress highlights more aggressively (high-key).
    pub key: f32,
    /// Sharpening parameter `φ` in `2^φ · a / s²` (§3.2). Larger values
    /// stabilise the denominator at very low luminances and effectively
    /// soften the local-detail enhancement. Default `8`.
    pub phi: f32,
    /// Uniformity threshold `ε` for scale selection (§3.3). The largest
    /// scale at which `|V(·, s_m)| < ε` is chosen; smaller values keep
    /// finer detail. Default `0.05`.
    pub epsilon: f32,
    /// Number of scales in the geometric pyramid (§3.3). Default `8`.
    pub scales: u32,
    /// Initial scale `s_0`. The k-th scale is `s_0 · 1.6^k`. Default
    /// `1.0` pixel.
    pub initial_scale: f32,
    /// Log-average bias `δ` (§1.1) to avoid `ln(0)` on black pixels.
    /// Default `1e-6`.
    pub delta: f32,
}

impl Default for ReinhardLocal {
    fn default() -> Self {
        Self {
            key: 0.18,
            phi: 8.0,
            epsilon: 0.05,
            scales: 8,
            initial_scale: 1.0,
            delta: 1.0e-6,
        }
    }
}

impl ReinhardLocal {
    /// Construct with a given key value. Non-finite or non-positive
    /// values fall back to the §2.1 photographic default `0.18`.
    pub fn new(key: f32) -> Self {
        let key = if key.is_finite() && key > 0.0 {
            key
        } else {
            0.18
        };
        Self {
            key,
            ..Self::default()
        }
    }

    /// Override the §3.2 sharpening parameter `φ`. Non-finite values
    /// fall back to the paper default `8`.
    pub fn with_phi(mut self, phi: f32) -> Self {
        if phi.is_finite() {
            self.phi = phi;
        }
        self
    }

    /// Override the §3.3 uniformity threshold `ε`. Non-finite or
    /// non-positive values fall back to the paper default `0.05`.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        if epsilon.is_finite() && epsilon > 0.0 {
            self.epsilon = epsilon;
        }
        self
    }

    /// Override the number of pyramid scales. `0` falls back to `1`
    /// (a single-scale evaluation — equivalent to global-Reinhard at
    /// `s_0`).
    pub fn with_scales(mut self, scales: u32) -> Self {
        self.scales = scales.max(1);
        self
    }

    /// Override the initial scale `s_0`. Non-finite or non-positive
    /// values fall back to the paper default `1.0`.
    pub fn with_initial_scale(mut self, s0: f32) -> Self {
        if s0.is_finite() && s0 > 0.0 {
            self.initial_scale = s0;
        }
        self
    }
}

#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// 1D normalised Gaussian kernel of half-width `radius`.
fn gauss_kernel(radius: usize, sigma: f32) -> Vec<f32> {
    let n = radius * 2 + 1;
    let mut k = Vec::with_capacity(n);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = i as f32 - radius as f32;
        let v = (-d * d / two_sigma_sq).exp();
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

/// Separable Gaussian blur on a `w·h` float field, edge-clamped.
fn blur_field(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    // Truncate the kernel at 3 sigma (>= 99.7 % of mass) for a
    // reasonable cost / accuracy trade. Always at least radius 1 so a
    // sigma << 1 still produces a non-degenerate kernel.
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let k = gauss_kernel(radius, sigma);
    let mut tmp = vec![0f32; w * h];
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &kv) in k.iter().enumerate() {
                let xx = (x as i32 + i as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + xx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &kv) in k.iter().enumerate() {
                let yy = (y as i32 + i as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                acc += tmp[yy * w + x] * kv;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

impl ImageFilter for ReinhardLocal {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ReinhardLocal requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: ReinhardLocal requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let stride_out = w * bpp;
        let src = &input.planes[0];
        let n = w * h;

        // §1 decode + per-pixel luminance.
        let mut linr = vec![0f32; n];
        let mut ling = vec![0f32; n];
        let mut linb = vec![0f32; n];
        let mut lum = vec![0f32; n];
        let mut log_sum = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let b = y * src.stride + x * bpp;
                let (r_lin, g_lin, b_lin) = match bpp {
                    1 => {
                        let v = srgb_to_linear(src.data[b]);
                        (v, v, v)
                    }
                    _ => (
                        srgb_to_linear(src.data[b]),
                        srgb_to_linear(src.data[b + 1]),
                        srgb_to_linear(src.data[b + 2]),
                    ),
                };
                linr[i] = r_lin;
                ling[i] = g_lin;
                linb[i] = b_lin;
                let l = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
                lum[i] = l;
                log_sum += ((self.delta + l) as f64).ln();
            }
        }
        // §1.1 log-average luminance.
        let log_avg = (log_sum / n as f64).exp() as f32;
        let log_avg = if log_avg > 0.0 { log_avg } else { self.delta };

        // §2.1 key-scaled luminance.
        let scale_factor = self.key / log_avg;
        let mut l_s = vec![0f32; n];
        for i in 0..n {
            l_s[i] = lum[i] * scale_factor;
        }

        // §3.1 build the centre/surround Gaussian pyramid.
        // α_1 = 1 / (2 · √2) → the centre Gaussian profile parameter.
        // α_2 = 1.6 · α_1   → the surround Gaussian profile parameter.
        // For the reference paper's circularly-symmetric Gaussian
        // `R_i(x,y,s) = (1/(π(α_i s)²)) · exp(−(x²+y²)/(α_i s)²)`,
        // matching the (un-normalised) form `exp(−r² / σ²)`, the
        // standard-deviation `σ_separable` used by a separable
        // `exp(−x² / (2σ²))` kernel is `σ = (α_i · s) / √2`.
        const ALPHA1: f32 = 0.353_553_4; // 1 / (2 · √2)
        const ALPHA2_RATIO: f32 = 1.6; // α_2 / α_1
        const SCALE_RATIO: f32 = 1.6; // s_{k+1} / s_k
        let kk = self.scales as usize;
        let mut v1_stack: Vec<Vec<f32>> = Vec::with_capacity(kk);
        let mut v_stack: Vec<Vec<f32>> = Vec::with_capacity(kk);
        for k in 0..kk {
            let s = self.initial_scale * SCALE_RATIO.powi(k as i32);
            // Convert R_i's `(α_i · s)²` denominator to the separable
            // Gaussian sigma `σ = (α_i · s) / √2` (i.e. `2σ² = (α_i · s)²`).
            let sigma_c = (ALPHA1 * s) / std::f32::consts::SQRT_2;
            let sigma_sur = (ALPHA1 * ALPHA2_RATIO * s) / std::f32::consts::SQRT_2;
            let v1 = blur_field(&l_s, w, h, sigma_c);
            let v2 = blur_field(&l_s, w, h, sigma_sur);
            // §3.2 normalised centre-surround difference.
            // `2^φ · a / s²` is the stabilisation term that keeps V
            // bounded as luminance approaches 0.
            let denom_stab = (2.0f32.powf(self.phi) * self.key) / (s * s);
            let mut v = vec![0f32; n];
            for i in 0..n {
                let d = v1[i] - v2[i];
                let denom = denom_stab + v1[i];
                v[i] = if denom > 0.0 { d / denom } else { 0.0 };
            }
            v1_stack.push(v1);
            v_stack.push(v);
        }

        // §3.3 per-pixel scale selection: pick the largest k with
        // |V(x,y, s_k)| < ε; if every scale already exceeds ε we keep
        // the smallest (k=0) — the safest local average available.
        let eps = self.epsilon;

        // §3.4 + §3 final paragraph: local tone-reproduction curve and
        // chroma-preserving re-modulation.
        const EPS_DIV: f32 = 1.0e-6;
        let mut out = vec![0u8; stride_out * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut sel = 0usize;
                for (k, v_at_scale) in v_stack.iter().enumerate().take(kk) {
                    if v_at_scale[i].abs() < eps {
                        sel = k;
                    } else {
                        // Largest still-uniform scale found — once V
                        // crosses ε any larger scale would straddle an
                        // edge so we stop searching upward.
                        break;
                    }
                }
                let v1_sel = v1_stack[sel][i];
                let l_s_i = l_s[i];
                let l_d = if l_s_i > 0.0 {
                    l_s_i / (1.0 + v1_sel)
                } else {
                    0.0
                };
                // Chroma-preserving re-modulation against the original
                // linear-light L_w (paper §3.4 final paragraph).
                let l_w = lum[i];
                let g = if l_w > EPS_DIV { l_d / l_w } else { 0.0 };
                let r = linr[i] * g;
                let gv = ling[i] * g;
                let bv = linb[i] * g;
                let db = y * stride_out + x * bpp;
                match bpp {
                    1 => out[db] = linear_to_srgb(gv),
                    3 => {
                        out[db] = linear_to_srgb(r);
                        out[db + 1] = linear_to_srgb(gv);
                        out[db + 2] = linear_to_srgb(bv);
                    }
                    4 => {
                        out[db] = linear_to_srgb(r);
                        out[db + 1] = linear_to_srgb(gv);
                        out[db + 2] = linear_to_srgb(bv);
                        out[db + 3] = src.data[y * src.stride + x * 4 + 3];
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: stride_out,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn output_in_range_rgb() {
        // Smoke test: a smooth gradient produces a same-shape output
        // buffer with no NaN / clipping panics.
        let input = rgb(16, 16, |x, y| {
            [
                ((x * 16).min(255)) as u8,
                ((y * 16).min(255)) as u8,
                (((x + y) * 8).min(255)) as u8,
            ]
        });
        let out = ReinhardLocal::default()
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 16 * 16 * 3);
        // u8 is trivially in [0, 255]; the real sanity test is that
        // the apply call returned at all (no NaN / panic / clamp
        // overflow during the pyramid evaluation).
        assert!(!out.planes[0].data.is_empty());
    }

    #[test]
    fn black_input_stays_black() {
        let input = rgb(8, 8, |_, _| [0, 0, 0]);
        let out = ReinhardLocal::default().apply(&input, p_rgb(8, 8)).unwrap();
        assert!(out.planes[0].data.iter().all(|&b| b == 0));
    }

    #[test]
    fn alpha_preserved_on_rgba() {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[120u8, 80, 200, 42]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = ReinhardLocal::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 42);
        }
    }

    #[test]
    fn rejects_yuv() {
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
        let err = ReinhardLocal::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("ReinhardLocal"));
    }

    #[test]
    fn local_differs_from_global_on_edge() {
        // A pure step image (left half bright, right half dark) is the
        // textbook case that distinguishes the local from the global
        // operator. The global form divides every pixel by 1+L, which
        // is constant per pixel; the local form divides each pixel by
        // a *neighbourhood* average that on the dark side reflects the
        // dark neighbours and on the bright side reflects the bright
        // ones. So the local operator preserves the per-side detail
        // better — at minimum it must not produce a byte-identical
        // output to the global form when there is a clear edge.
        let input = rgb(
            16,
            16,
            |x, _| if x < 8 { [240, 240, 240] } else { [40, 40, 40] },
        );
        let local = ReinhardLocal::default()
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        let global = crate::Reinhard::new(0.18)
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        assert_ne!(
            local.planes[0].data, global.planes[0].data,
            "local Reinhard must diverge from the global form on a step edge"
        );
    }

    #[test]
    fn single_scale_reduces_to_simple_blur_division() {
        // With `scales = 1` and the smallest scale, the per-pixel scale
        // selector has nothing to pick from — every pixel uses k=0. The
        // operator simplifies to `L_s / (1 + V1(·, s_0))` everywhere,
        // which must still produce a valid finite output identical in
        // shape to the multi-scale form (and pass the no-NaN smoke
        // check).
        let input = rgb(8, 8, |x, y| {
            [((x * 32).min(255)) as u8, ((y * 32).min(255)) as u8, 100]
        });
        let f = ReinhardLocal::default().with_scales(1);
        let out = f.apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn epsilon_zero_picks_smallest_scale() {
        // ε = 0 means **no** scale can satisfy |V| < ε (except where V
        // happens to be exactly 0). For our default initial scale the
        // selector then falls back to k = 0 for almost every pixel.
        // This must not panic and must produce shape-correct output.
        let input = rgb(8, 8, |x, y| {
            [
                ((x * 30).min(255)) as u8,
                ((y * 30).min(255)) as u8,
                (((x ^ y) * 8).min(255)) as u8,
            ]
        });
        // Use a tiny positive ε rather than 0 (with_epsilon rejects 0
        // as non-positive per the doc).
        let f = ReinhardLocal::default().with_epsilon(1.0e-12);
        let out = f.apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn constructor_normalises_nonfinite_key() {
        let nan = ReinhardLocal::new(f32::NAN);
        let neg = ReinhardLocal::new(-0.5);
        let zero = ReinhardLocal::new(0.0);
        assert_eq!(nan.key, 0.18);
        assert_eq!(neg.key, 0.18);
        assert_eq!(zero.key, 0.18);
    }

    #[test]
    fn with_scales_zero_clamps_to_one() {
        let f = ReinhardLocal::default().with_scales(0);
        assert_eq!(f.scales, 1);
    }

    #[test]
    fn gray8_path_runs() {
        let mut data = vec![0u8; 64];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i * 4).min(255) as u8;
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = ReinhardLocal::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 64);
    }
}
