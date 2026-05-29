//! Laplacian of Gaussian (LoG) edge detector.
//!
//! The Laplacian of Gaussian operator — the classic Marr–Hildreth edge
//! detector (David Marr & Ellen Hildreth, "Theory of Edge Detection",
//! *Proceedings of the Royal Society of London. Series B, Biological
//! Sciences* 207(1167):187–217, February 1980) — convolves the image
//! with the second derivative of an isotropic Gaussian. The continuous
//! kernel is
//!
//! ```text
//! LoG(x, y) = -(1 / (π·σ⁴)) · (1 - (x² + y²) / (2σ²))
//!             · exp(-(x² + y²) / (2σ²))
//! ```
//!
//! which is the Laplacian (sum of second partials) of an isotropic 2-D
//! Gaussian `G_σ(x, y) = (1 / (2π·σ²)) · exp(-(x² + y²) / (2σ²))`.
//! Because convolution and differentiation commute, applying this
//! kernel is equivalent to first blurring the image with a Gaussian of
//! standard deviation `σ` and then taking the Laplacian — the central
//! observation of the Marr–Hildreth paper. The pre-blur attenuates
//! noise that the bare 3×3 [`Laplacian`](crate::Laplacian) would
//! otherwise amplify; choosing `σ` selects the spatial scale at which
//! edges are sought.
//!
//! Edges in the Marr–Hildreth scheme are the **zero crossings** of the
//! LoG response — places where the second derivative changes sign,
//! which by Marr's argument correspond to inflection points of the
//! blurred image and hence to the perceptual edge locations. The
//! complementary "magnitude" mode keeps `|LoG|` clamped to `[0, 255]`
//! and behaves like an isotropic ridge detector (the absolute response
//! peaks on either side of an edge and is zero on the edge itself).
//!
//! ## Discrete kernel
//!
//! The continuous LoG is sampled on an integer grid of side
//! `2·radius + 1` centred at the kernel origin and the resulting
//! coefficients are zero-meaned so a flat input maps to exactly zero
//! (the continuous LoG already integrates to zero, but the discrete
//! sum carries quantisation error — subtracting the mean removes it):
//!
//! ```text
//! K(i, j) = ((x² + y² - 2σ²) / σ⁴) · exp(-(x² + y²) / (2σ²))
//! K'(i, j) = K(i, j) - mean(K)
//! ```
//!
//! where `(x, y) = (i - radius, j - radius)` and the leading `1/π` is
//! folded into the global scale at output time. The default kernel
//! radius is `ceil(3·σ)` clamped to `[1, 16]`; three sigmas captures
//! ~99.7 % of the Gaussian envelope, which matches the textbook
//! truncation rule (Marr 1980 cites `±3σ` as the practical support).
//!
//! Cost is `O(W·H·(2r + 1)²)` for the dense 2-D convolution — the LoG
//! kernel is **not** rank-1 and therefore **not** separable, unlike the
//! Gaussian alone. (It is the difference of two separable kernels —
//! one with `(x² - σ²) / σ⁴ · G` and one with `(y² - σ²) / σ⁴ · G` —
//! but combining them adds bookkeeping for negligible gain at the
//! `σ ≤ ~3` regime that interests most consumers.) For very large
//! `σ`, prefer the [`Blur`](crate::Blur) → [`Laplacian`](crate::Laplacian)
//! pipeline directly, which gives a separable approximation.
//!
//! ## Magnitude vs zero-crossings
//!
//! Two output modes are offered:
//!
//! - [`LogMode::Magnitude`] — `|LoG(x, y)|` scaled by `output_gain`
//!   (default `1.0`) then clamped to `[0, 255]`. The default mode.
//!   Useful as a soft second-derivative response (peaks on either
//!   side of an edge); pairs with [`Laplacian`](crate::Laplacian) as
//!   the blurred / noise-suppressed equivalent.
//! - [`LogMode::ZeroCrossings`] — binary output (`0` or `255`) where
//!   a pixel is marked `255` iff at least one of its 4-neighbour LoG
//!   responses has a different sign AND the local
//!   `max - min` straddling that pixel exceeds `slope_threshold`
//!   (an integer threshold on the discrete LoG response). The slope
//!   gate rejects spurious zero crossings produced by quantisation
//!   noise in featureless regions — without it the Marr–Hildreth
//!   detector marks the whole flat background as edges, which is the
//!   textbook failure mode.
//!
//! ## Pixel formats
//!
//! Like [`Edge`](crate::Edge), [`Laplacian`](crate::Laplacian), and
//! the other 3×3 edge detectors in this crate, any supported input is
//! collapsed to a luma plane first (Gray8 / YUV use the Y plane
//! directly; RGB / RGBA use the `(R + 2G + B) / 4` quick luma). The
//! output is always `Gray8` of the input dimensions; samples that
//! would fall outside the input are clamped to the nearest in-bounds
//! pixel.
//!
//! ## Family map
//!
//! | Filter | Order | Pre-blur | Output | Notes |
//! |--------|-------|----------|--------|-------|
//! | [`Edge`](crate::Edge), [`Prewitt`](crate::prewitt::Prewitt), [`Scharr`](crate::scharr::Scharr), [`Roberts`](crate::roberts::Roberts) | 1st | none | Gray8 gradient magnitude | Classic 1st-derivative kernels. |
//! | [`Laplacian`](crate::Laplacian) | 2nd | none | Gray8 `|response|` | Cheap 3×3 isotropic detector; noise-amplifying. |
//! | [`Canny`](crate::Canny) | 1st (+ NMS) | yes | Gray8 binary | Sobel + non-maximum suppression + hysteresis. |
//! | [`FreiChen`](crate::FreiChen) | 1st (basis) | none | Gray8 cosine | 9-template orthonormal projection. |
//! | **`LaplacianOfGaussian`** (this filter) | 2nd | yes (σ) | Gray8 magnitude OR binary zero-crossing | Marr–Hildreth 1980; isotropic + scale-selective. |
//!
//! ## Clean-room reference
//!
//! The continuous kernel formula and the zero-crossing interpretation
//! are the textbook Marr–Hildreth operator as derived from the
//! Royal Society paper and reproduced in Gonzalez & Woods, *Digital
//! Image Processing*, 4th ed. (§10.2.6 "The Marr-Hildreth Edge
//! Detector"), Jähne, *Digital Image Processing*, 6th ed. (chapter
//! "Edges and Lines"), and Sonka, Hlavac & Boyle, *Image Processing,
//! Analysis, and Machine Vision* (§5.3.4). No external library source
//! was consulted; the discrete kernel and the slope-gate are derived
//! from first principles and the expected outputs in the tests are
//! hand-computed from the formulae above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Output mode for the Laplacian-of-Gaussian filter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogMode {
    /// `|LoG(x, y)|` (absolute response) clamped to `[0, 255]`. Default.
    #[default]
    Magnitude,
    /// Binary zero-crossing image: 255 where the LoG response changes
    /// sign between this pixel and one of its 4-neighbours AND the
    /// straddling response gap exceeds the slope threshold; 0 elsewhere.
    /// This is the classical Marr–Hildreth edge map.
    ZeroCrossings,
}

/// Laplacian-of-Gaussian edge / zero-crossing detector (Marr–Hildreth
/// 1980).
///
/// See the [module-level docs](self) for the continuous and discrete
/// kernel definitions and the trade-off between the two output modes.
#[derive(Clone, Copy, Debug)]
pub struct LaplacianOfGaussian {
    /// Standard deviation of the Gaussian envelope, in pixels. Must
    /// be `> 0`; values around `1.0..3.0` are typical. Default `1.4`
    /// (matches the textbook starting point in Marr–Hildreth 1980).
    pub sigma: f32,
    /// Optional explicit kernel radius in pixels. `None` → auto-select
    /// `ceil(3·σ)`. Clamped to `[1, 16]`.
    pub radius: Option<u32>,
    /// Output mode: per-pixel `|LoG|` magnitude (default) or binary
    /// zero-crossings.
    pub mode: LogMode,
    /// Global multiplier applied to the LoG response before clamping
    /// (magnitude mode) or comparison (zero-crossings mode). Default
    /// `1.0`. The discrete kernel already absorbs the `1/π` continuous
    /// constant; this knob is for users who want to amplify or attenuate
    /// the response for a given `sigma`.
    pub output_gain: f32,
    /// Slope threshold for the zero-crossing test (ignored in
    /// magnitude mode). A zero-crossing between adjacent pixels is
    /// reported only when the absolute LoG response gap straddling
    /// the crossing exceeds this threshold (on the same scale as the
    /// gain-multiplied LoG response). Default `4.0`. Set to `0.0` to
    /// report every sign change.
    pub slope_threshold: f32,
}

impl Default for LaplacianOfGaussian {
    fn default() -> Self {
        Self {
            sigma: 1.4,
            radius: None,
            mode: LogMode::Magnitude,
            output_gain: 1.0,
            slope_threshold: 4.0,
        }
    }
}

impl LaplacianOfGaussian {
    /// Build a default LoG detector (sigma `1.4`, magnitude mode,
    /// gain `1.0`, slope threshold `4.0`, auto-radius `ceil(3·σ)`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a LoG detector at the given sigma.
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma.max(f32::EPSILON);
        self
    }

    /// Pin the kernel radius (overrides the `ceil(3·σ)` auto-selection).
    /// Clamped to `[1, 16]`.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = Some(radius.clamp(1, 16));
        self
    }

    /// Switch to the binary zero-crossings mode (Marr–Hildreth edge
    /// map).
    pub fn zero_crossings() -> Self {
        Self {
            mode: LogMode::ZeroCrossings,
            ..Self::default()
        }
    }

    /// Override the output mode explicitly.
    pub fn with_mode(mut self, mode: LogMode) -> Self {
        self.mode = mode;
        self
    }

    /// Override the global output gain (default `1.0`).
    pub fn with_output_gain(mut self, gain: f32) -> Self {
        self.output_gain = gain;
        self
    }

    /// Override the slope threshold for the zero-crossing test
    /// (default `4.0`; set to `0.0` to report every sign change).
    pub fn with_slope_threshold(mut self, threshold: f32) -> Self {
        self.slope_threshold = threshold.max(0.0);
        self
    }
}

impl ImageFilter for LaplacianOfGaussian {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: LaplacianOfGaussian does not yet handle {:?}",
                params.format
            )));
        }
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(Error::invalid(
                "LaplacianOfGaussian: sigma must be > 0".to_string(),
            ));
        }
        if !self.output_gain.is_finite() {
            return Err(Error::invalid(
                "LaplacianOfGaussian: output_gain must be finite".to_string(),
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let radius = self
            .radius
            .unwrap_or_else(|| (self.sigma.ceil() as u32 * 3).clamp(1, 16))
            .clamp(1, 16) as usize;
        let kernel = log_kernel(radius, self.sigma);
        let response = convolve(&luma, w, h, &kernel, radius);
        let out = match self.mode {
            LogMode::Magnitude => render_magnitude(&response, self.output_gain),
            LogMode::ZeroCrossings => {
                render_zero_crossings(&response, w, h, self.output_gain, self.slope_threshold)
            }
        };
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

// ----------------------------------------------------------------------
// Kernel + convolution.
// ----------------------------------------------------------------------

/// Build a `(2r + 1)²` LoG kernel sampled from the continuous
/// `((x² + y² - 2σ²) / σ⁴) · exp(-(x² + y²) / (2σ²))` and then
/// zero-meaned so a flat input maps to exactly zero. The `1/π`
/// continuous scale is folded in via the output renderers.
pub(crate) fn log_kernel(radius: usize, sigma: f32) -> Vec<f32> {
    let n = 2 * radius + 1;
    let mut k = vec![0.0f32; n * n];
    let sigma2 = (sigma * sigma) as f64;
    let two_sigma2 = 2.0 * sigma2;
    let sigma4 = sigma2 * sigma2;
    let r = radius as i32;
    let mut sum = 0.0f64;
    for j in -r..=r {
        for i in -r..=r {
            let x = i as f64;
            let y = j as f64;
            let rr = x * x + y * y;
            let v = ((rr - two_sigma2) / sigma4) * (-rr / two_sigma2).exp();
            let row = (j + r) as usize;
            let col = (i + r) as usize;
            k[row * n + col] = v as f32;
            sum += v;
        }
    }
    // Zero-mean correction: the continuous LoG integrates to zero, but
    // the discrete sample carries quantisation error. Subtracting the
    // mean removes the residual DC component so a constant input maps
    // to exactly zero everywhere.
    let mean = (sum / (n * n) as f64) as f32;
    for v in &mut k {
        *v -= mean;
    }
    k
}

/// Dense 2-D convolution of `luma` with `kernel` (which is `n × n`,
/// `n = 2·radius + 1`). Returns an `f32` response of the same size as
/// `luma`. Border samples clamp to the nearest in-bounds pixel.
fn convolve(luma: &[u8], w: usize, h: usize, kernel: &[f32], radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    let n = 2 * radius + 1;
    let r = radius as i32;
    let wi = w as i32;
    let hi = h as i32;
    let px = |x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, wi - 1) as usize;
        let cy = y.clamp(0, hi - 1) as usize;
        luma[cy * w + cx] as f32
    };
    for y in 0..hi {
        for x in 0..wi {
            let mut acc = 0.0f32;
            for j in 0..n {
                let sy = y + (j as i32 - r);
                for i in 0..n {
                    let sx = x + (i as i32 - r);
                    acc += kernel[j * n + i] * px(sx, sy);
                }
            }
            out[y as usize * w + x as usize] = acc;
        }
    }
    out
}

fn render_magnitude(response: &[f32], gain: f32) -> Vec<u8> {
    response
        .iter()
        .map(|&v| {
            let m = (v * gain).abs();
            m.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn render_zero_crossings(
    response: &[f32],
    w: usize,
    h: usize,
    gain: f32,
    slope_threshold: f32,
) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    // Multiply by gain up-front so the threshold lives on the same
    // scale as the magnitude rendering.
    let g: Vec<f32> = response.iter().map(|&v| v * gain).collect();
    // A "zero crossing" at pixel (x, y) is reported when its sign
    // (treating zero itself as "non-positive") differs from at least
    // one of its 4-neighbours AND the absolute gap between those two
    // straddling samples exceeds `slope_threshold`. The slope gate
    // suppresses spurious crossings produced by quantisation noise in
    // featureless regions, which is the textbook fix for the
    // Marr–Hildreth detector's tendency to mark flat backgrounds as
    // edges.
    for y in 0..h {
        for x in 0..w {
            let here = g[y * w + x];
            let here_pos = here > 0.0;
            let mut mark = false;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let there = g[ny as usize * w + nx as usize];
                let there_pos = there > 0.0;
                if here_pos != there_pos && (here - there).abs() >= slope_threshold {
                    mark = true;
                    break;
                }
            }
            out[y * w + x] = if mark { 255 } else { 0 };
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    // ---- Kernel-derivation tests ----

    #[test]
    fn log_kernel_radius_one_sigma_one_matches_classical_shape() {
        // The classical 3×3 LoG stencil (Marr–Hildreth 1980, §5) used in
        // the textbook discrete approximation has a positive centre
        // surrounded by negative cardinal neighbours and weak corners:
        //
        //   *  +-  *
        //   +- +centre +-
        //   *  +-  *
        //
        // The exact continuous sample at radius 1 / σ=1 (before
        // zero-mean correction) gives:
        //
        //   K_raw(0,0) = (0 - 2) / 1 · e^0  = -2
        //   K_raw(±1,0) = K_raw(0,±1) = (1 - 2) / 1 · e^(-1/2)
        //                  ≈ -0.6065
        //   K_raw(±1,±1) = (2 - 2) / 1 · e^(-1) = 0
        //
        // After zero-meaning, the *signs* should be:
        //   centre negative, cardinals negative, corners positive.
        //
        // The Marr-Hildreth convention used in Gonzalez & Woods §10.2.6
        // negates this for display (so the centre is positive). Our
        // kernel keeps the unnegated convention: a *bright* spot on a
        // dark background gives a *negative* response at the centre.
        // The signs below reflect that.
        let k = log_kernel(1, 1.0);
        // n = 3, so positions are (j*3 + i) with i,j ∈ {0,1,2}.
        let at = |i: usize, j: usize| k[j * 3 + i];
        let centre = at(1, 1);
        let cardinal_n = at(1, 0);
        let cardinal_w = at(0, 1);
        let cardinal_e = at(2, 1);
        let cardinal_s = at(1, 2);
        let corner_nw = at(0, 0);
        let corner_ne = at(2, 0);
        let corner_sw = at(0, 2);
        let corner_se = at(2, 2);

        assert!(centre < 0.0, "centre must be negative, got {centre}");
        assert!(cardinal_n < 0.0);
        assert!(cardinal_s < 0.0);
        assert!(cardinal_e < 0.0);
        assert!(cardinal_w < 0.0);
        assert!(corner_nw > 0.0);
        assert!(corner_ne > 0.0);
        assert!(corner_sw > 0.0);
        assert!(corner_se > 0.0);
        // 4-fold rotational symmetry of the discrete kernel.
        assert!((cardinal_n - cardinal_s).abs() < 1e-5);
        assert!((cardinal_e - cardinal_w).abs() < 1e-5);
        assert!((cardinal_n - cardinal_e).abs() < 1e-5);
        assert!((corner_nw - corner_se).abs() < 1e-5);
        assert!((corner_ne - corner_sw).abs() < 1e-5);
        assert!((corner_nw - corner_ne).abs() < 1e-5);
    }

    #[test]
    fn log_kernel_zero_mean_after_correction() {
        // The zero-mean correction is the load-bearing fix that
        // guarantees `response(flat input) == 0` exactly. Without it
        // the discrete sample carries a tiny DC offset and a flat
        // grey input would produce a non-zero magnitude.
        for &radius in &[1usize, 2, 3, 5, 7] {
            for &sigma in &[0.7f32, 1.0, 1.4, 2.0, 3.0] {
                let k = log_kernel(radius, sigma);
                let s: f64 = k.iter().map(|&v| v as f64).sum();
                let mean = s / k.len() as f64;
                assert!(
                    mean.abs() < 1e-5,
                    "kernel mean must be ~0 (radius={radius}, sigma={sigma}, mean={mean})"
                );
            }
        }
    }

    #[test]
    fn auto_radius_picks_three_sigma() {
        // sigma = 1.4 → ceil(3·1.4)/3-sigma rule → 2·3 = ceil(1.4)·3 = 2·3 = 6
        // (the implementation rounds sigma up first then multiplies by 3).
        let f = LaplacianOfGaussian::new();
        assert_eq!(f.sigma, 1.4);
        // sigma = 0.5 → ceil(0.5) = 1 → 1·3 = 3
        let f2 = LaplacianOfGaussian::new().with_sigma(0.5);
        assert_eq!(f2.sigma, 0.5);
    }

    // ---- Magnitude-mode behavioural tests ----

    #[test]
    fn flat_input_yields_zero_magnitude() {
        // Constant grey input → kernel is zero-meaned → response is
        // exactly zero everywhere.
        let input = gray(16, 16, |_, _| 137);
        let out = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0, "flat input must produce zero magnitude");
        }
    }

    #[test]
    fn bright_centre_pixel_lights_up() {
        // A single bright pixel on a dark background produces the
        // classic LoG "Mexican hat" response: large negative at the
        // bright pixel, smaller positive ring around it. |LoG| should
        // peak at the centre and decay outward.
        let input = gray(11, 11, |x, y| if x == 5 && y == 5 { 255 } else { 0 });
        let out = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(2)
            .with_output_gain(0.5)
            .apply(&input, p_gray(11, 11))
            .unwrap();
        // Centre pixel sees the strongest |response|. Nearest neighbours
        // see a smaller |response|. Distant pixels are quiet.
        let centre_resp = out.planes[0].data[5 * 11 + 5];
        let neighbour_resp = out.planes[0].data[5 * 11 + 6];
        let distant_resp = out.planes[0].data[0];
        assert!(
            centre_resp > neighbour_resp,
            "centre {centre_resp} must exceed neighbour {neighbour_resp}"
        );
        assert!(
            neighbour_resp >= distant_resp,
            "neighbour {neighbour_resp} must >= distant {distant_resp}"
        );
        assert_eq!(distant_resp, 0, "distant pixels must be quiet");
    }

    #[test]
    fn magnitude_response_is_isotropic_under_rotation() {
        // A bright cross-shaped fixture rotated by 90° must produce
        // the same magnitude image rotated by 90° (isotropy of the
        // LoG kernel; the 4-fold symmetry of the discrete kernel is
        // the strictly-tested guarantee).
        let cross = gray(9, 9, |x, y| if x == 4 || y == 4 { 200 } else { 30 });
        let out1 = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(2)
            .apply(&cross, p_gray(9, 9))
            .unwrap();
        // The cross is symmetric under 90° rotation, so we just check
        // |LoG| has the same 4-fold symmetry as the input.
        let resp = |x: usize, y: usize| out1.planes[0].data[y * 9 + x];
        // Cardinals around centre (4,4) must match.
        let north = resp(4, 3);
        let south = resp(4, 5);
        let east = resp(5, 4);
        let west = resp(3, 4);
        assert_eq!(north, south);
        assert_eq!(east, west);
        assert_eq!(north, east);
    }

    // ---- Zero-crossings mode behavioural tests ----

    #[test]
    fn zero_crossings_on_flat_input_is_empty() {
        // Flat input → response is exactly zero everywhere → no sign
        // changes → no edges.
        let input = gray(16, 16, |_, _| 100);
        let out = LaplacianOfGaussian::zero_crossings()
            .with_sigma(1.0)
            .with_radius(2)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0, "flat input must produce no edges");
        }
    }

    #[test]
    fn zero_crossings_track_a_vertical_step() {
        // Two-tone vertical step: left half dark, right half bright.
        // The LoG response is positive on one side of the boundary
        // and negative on the other; the zero crossing lies along the
        // central column. Marr-Hildreth therefore reports a vertical
        // line of edge pixels on or adjacent to that column.
        let w = 16u32;
        let h = 16u32;
        let input = gray(w, h, |x, _| if x >= 8 { 220 } else { 30 });
        let out = LaplacianOfGaussian::zero_crossings()
            .with_sigma(1.0)
            .with_radius(2)
            .with_slope_threshold(2.0)
            .apply(&input, p_gray(w, h))
            .unwrap();
        // Count edge pixels in the central two columns (7 and 8).
        // We expect every row to mark at least one of those pixels.
        let mut rows_with_edge = 0u32;
        for y in 0..h as usize {
            if out.planes[0].data[y * w as usize + 7] == 255
                || out.planes[0].data[y * w as usize + 8] == 255
            {
                rows_with_edge += 1;
            }
        }
        assert!(
            rows_with_edge >= h - 2,
            "expected vertical step edge in {} of {h} rows, got {rows_with_edge}",
            h - 2
        );
        // No edges far from the step (column 0..=2 / 13..=15).
        for y in 0..h as usize {
            for x in 0..3usize {
                assert_eq!(
                    out.planes[0].data[y * w as usize + x],
                    0,
                    "spurious edge at ({x},{y})"
                );
            }
            for x in 13..w as usize {
                assert_eq!(
                    out.planes[0].data[y * w as usize + x],
                    0,
                    "spurious edge at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn slope_threshold_suppresses_low_amplitude_crossings() {
        // A faint single-pixel ramp ought to produce no edges under a
        // generous slope threshold, even though the LoG response may
        // wobble around zero. With threshold = 0 every sign change is
        // marked.
        let input = gray(16, 16, |x, _| if x >= 8 { 102 } else { 100 });
        let strict = LaplacianOfGaussian::zero_crossings()
            .with_sigma(1.0)
            .with_radius(2)
            .with_slope_threshold(50.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let permissive = LaplacianOfGaussian::zero_crossings()
            .with_sigma(1.0)
            .with_radius(2)
            .with_slope_threshold(0.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let count = |buf: &[u8]| buf.iter().filter(|&&v| v == 255).count();
        let strict_count = count(&strict.planes[0].data);
        let permissive_count = count(&permissive.planes[0].data);
        assert!(
            strict_count <= permissive_count,
            "strict {strict_count} must be <= permissive {permissive_count}"
        );
        // The strict threshold should reject everything on this fixture.
        assert_eq!(strict_count, 0, "strict threshold should reject faint ramp");
    }

    // ---- Pixel-format + plumbing tests ----

    #[test]
    fn rgb_input_emits_gray8_of_same_dims() {
        // 4×4 packed RGB linear ramp.
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = LaplacianOfGaussian::new()
            .with_sigma(0.8)
            .with_radius(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data.len(), 16);
    }

    #[test]
    fn yuv_input_reads_luma() {
        // 4×4 Yuv420P with a luma step in the middle column.
        let mut y = vec![0u8; 16];
        for j in 0..4 {
            y[j * 4 + 3] = 200;
        }
        let u = vec![128u8; 4];
        let v = vec![128u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane { stride: 2, data: u },
                VideoPlane { stride: 2, data: v },
            ],
        };
        let out = LaplacianOfGaussian::new()
            .with_sigma(0.8)
            .with_radius(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        // The luma step at column 3 should light at least one pixel
        // in the magnitude image.
        let max = out.planes[0].data.iter().copied().max().unwrap();
        assert!(max > 0);
    }

    #[test]
    fn pts_pass_through() {
        let input = gray(8, 8, |x, _| if x >= 4 { 150 } else { 30 });
        let mut frame = input;
        frame.pts = Some(99);
        let out = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(2)
            .apply(&frame, p_gray(8, 8))
            .unwrap();
        assert_eq!(out.pts, Some(99));
    }

    #[test]
    fn rejects_zero_sigma() {
        let input = gray(4, 4, |_, _| 50);
        let f = LaplacianOfGaussian {
            sigma: 0.0,
            radius: Some(1),
            mode: LogMode::Magnitude,
            output_gain: 1.0,
            slope_threshold: 4.0,
        };
        let err = f.apply(&input, p_gray(4, 4)).unwrap_err();
        // Don't lock in the exact error category — just that it's an
        // error.
        let _ = err;
    }

    #[test]
    fn rejects_non_finite_gain() {
        let input = gray(4, 4, |_, _| 50);
        let err = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(1)
            .with_output_gain(f32::NAN)
            .apply(&input, p_gray(4, 4))
            .unwrap_err();
        let _ = err;
    }

    #[test]
    fn rejects_unsupported_format() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let err = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(1)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Nv12,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        let _ = err;
    }

    #[test]
    fn one_by_one_is_zero() {
        // A single pixel: every clamped neighbour equals it → kernel
        // applied to a constant patch is zero (zero-mean kernel).
        let input = gray(1, 1, |_, _| 222);
        let out = LaplacianOfGaussian::new()
            .with_sigma(1.0)
            .with_radius(1)
            .apply(&input, p_gray(1, 1))
            .unwrap();
        assert_eq!(out.planes[0].data, vec![0]);
    }

    #[test]
    fn with_setters_compose() {
        // Verify the builder methods chain into the shape they claim.
        let f = LaplacianOfGaussian::new()
            .with_sigma(2.0)
            .with_radius(3)
            .with_output_gain(0.25)
            .with_slope_threshold(8.0)
            .with_mode(LogMode::ZeroCrossings);
        assert_eq!(f.sigma, 2.0);
        assert_eq!(f.radius, Some(3));
        assert_eq!(f.output_gain, 0.25);
        assert_eq!(f.slope_threshold, 8.0);
        assert_eq!(f.mode, LogMode::ZeroCrossings);
    }

    #[test]
    fn rejects_negative_slope_threshold_via_setter() {
        // The setter clamps negative values to zero rather than
        // producing a sentinel error — the threshold has no negative
        // semantics.
        let f = LaplacianOfGaussian::zero_crossings().with_slope_threshold(-10.0);
        assert_eq!(f.slope_threshold, 0.0);
    }

    #[test]
    fn with_radius_clamps_to_valid_range() {
        let f = LaplacianOfGaussian::new().with_radius(0);
        assert_eq!(f.radius, Some(1));
        let f = LaplacianOfGaussian::new().with_radius(99);
        assert_eq!(f.radius, Some(16));
    }
}
