//! Gabor filter — oriented narrow-band Gaussian-modulated cosine.
//!
//! The Gabor function is the product of an axis-rotated 2-D Gaussian
//! envelope and a complex (or real cosine / sine) sinusoidal carrier.
//! It is the canonical jointly-localised filter in space and spatial
//! frequency: among all square-integrable functions it achieves the
//! minimum product of position-uncertainty and frequency-uncertainty
//! permitted by the 2-D Fourier uncertainty inequality (Gabor 1946,
//! *Theory of Communication*, J. Inst. Elec. Eng. 93(III):429–457;
//! Daugman 1985, "Uncertainty relation for resolution in space, spatial
//! frequency, and orientation optimized by two-dimensional visual cortical
//! filters", *J. Opt. Soc. Am. A* 2(7):1160–1169). This optimality makes
//! the Gabor a textbook bandpass element for texture analysis, oriented-
//! edge response, and biologically-motivated simple-cell modelling of
//! the primary visual cortex (V1).
//!
//! ## Continuous kernel
//!
//! Following Daugman's 1985 parametrisation we expose five knobs and
//! sample the real-valued, even-symmetric (cosine-carrier) Gabor:
//!
//! ```text
//!   x' =  (x - cx) · cos θ + (y - cy) · sin θ
//!   y' = -(x - cx) · sin θ + (y - cy) · cos θ
//!
//!   g(x, y) = exp( -(x'² + γ² · y'²) / (2σ²) )
//!              · cos( 2π · x'/λ + ψ )
//! ```
//!
//! where
//!
//! - `λ` is the carrier wavelength in pixels (period of the cosine
//!   along the rotated `x'` axis). `λ ≥ 2` is required by the spatial
//!   Nyquist limit; the filter rejects smaller values.
//! - `θ` is the orientation (degrees CCW from the positive image-x axis).
//!   `θ = 0` ⇒ vertical stripes (the carrier oscillates along `x`).
//! - `ψ` is the carrier phase offset (degrees). `ψ = 0` ⇒ even-symmetric
//!   cosine (preserves bar-like features); `ψ = 90` ⇒ odd-symmetric
//!   sine (preserves step-edge features).
//! - `σ` is the Gaussian standard deviation in pixels (envelope spatial
//!   support along `x'`). Daugman's bandwidth relation
//!   `σ ≈ λ · (1/π) · √(ln 2 / 2) · (2^b + 1) / (2^b − 1)` ties σ to a
//!   half-amplitude bandwidth `b` in octaves; we leave σ exposed so the
//!   caller can encode either bandwidth or absolute support directly.
//! - `γ` is the spatial-aspect ratio (`σ_y / σ_x`). `γ = 1` ⇒ isotropic
//!   envelope; `γ < 1` ⇒ elongated parallel to the carrier (typical
//!   V1-tuned values are `0.3..0.7`).
//!
//! ## DC removal
//!
//! Sampled on a finite grid the discrete kernel has a residual DC
//! component (the cosine half-cycle that falls inside the envelope does
//! not perfectly cancel against its negative half-cycle). A flat input
//! would then produce a constant non-zero response that swamps the true
//! oriented response. We subtract the discrete arithmetic mean of the
//! coefficients so the kernel sums to exactly zero:
//!
//! ```text
//!   K'(i, j) = K(i, j) − mean(K)
//! ```
//!
//! This is the standard textbook fix (Petkov & Kruizinga 1997,
//! "Computational models of visual neurons specialised in the detection
//! of periodic and aperiodic oriented visual stimuli: bar and grating
//! cells", *Biological Cybernetics* 76(2):83–96) and matches the same
//! treatment applied to the discrete
//! [`LaplacianOfGaussian`](crate::LaplacianOfGaussian) kernel.
//!
//! ## Output modes
//!
//! Two output modes are exposed via [`GaborMode`]:
//!
//! - [`GaborMode::Signed`] — the raw (sign-preserving) filter response
//!   linearly remapped from `[−R, +R]` to `[0, 255]` with neutral grey
//!   128 marking zero response. `R` is the maximum absolute discrete
//!   response of the kernel against a unit-amplitude grating, computed
//!   once at kernel-build time as `Σ |K'(i, j)| / 2`. Useful when the
//!   downstream pipeline distinguishes excitation from inhibition (the
//!   "simple cell" linear response).
//! - [`GaborMode::Magnitude`] — `|response|` linearly remapped from
//!   `[0, R]` to `[0, 255]`. The classical edge / texture energy used
//!   when the pipeline cares about *presence* of the oriented feature
//!   without polarity. (Equivalent to a half-rectified simple cell or
//!   one half of a quadrature-pair complex-cell magnitude.)
//!
//! Both modes are scaled by an `output_gain` knob (default `1.0`) for
//! callers that prefer to clip at a tighter range.
//!
//! ## Discrete support
//!
//! The kernel is sampled on a square `(2·radius + 1)²` grid. The
//! default radius is `ceil(3·σ·max(1, 1/γ))` clamped to `[1, 32]` —
//! the `3σ` rule (Marr's textbook truncation) along the elongated
//! axis, expanded by `1/γ` to cover the perpendicular extent. The
//! convolution is dense 2-D (the Gabor is **not** rank-1 separable
//! for general `γ ≠ 1` and arbitrary `θ`); cost is
//! `O(W·H·(2·radius + 1)²)`.
//!
//! ## Pixel formats
//!
//! Like [`Edge`](crate::Edge) / [`Laplacian`](crate::Laplacian) /
//! [`LaplacianOfGaussian`](crate::LaplacianOfGaussian), any supported
//! input is collapsed to a luma plane first (Gray8 / YUV use the Y
//! plane directly; RGB / RGBA use the `(R + 2G + B) / 4` quick luma).
//! Border samples clamp to the nearest in-bounds pixel. Output is
//! always single-plane `Gray8` of the input dimensions.
//!
//! ## Clean-room reference
//!
//! The continuous-kernel formula, the bandwidth relation, and the
//! orientation / phase parametrisation are the textbook Gabor / Daugman
//! 1985 operator as derived from the original publications cited above
//! and reproduced in Gonzalez & Woods, *Digital Image Processing*,
//! 4th ed. (§4.4.2 "Filtering in the Spatial Domain — Gabor"), Jähne,
//! *Digital Image Processing*, 6th ed. (chapter "Quadrature filters
//! and the analytic signal"), and Sonka, Hlavac & Boyle, *Image
//! Processing, Analysis, and Machine Vision* (§5.3.7). No external
//! library source was consulted; the discrete kernel, the DC-removal
//! step, the signed-vs-magnitude renderer, and every test expectation
//! were derived from first principles and the formulae above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Output mode for the [`Gabor`] filter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaborMode {
    /// Signed response: raw filter output linearly remapped from
    /// `[−R, +R]` to `[0, 255]`. Neutral grey (`128`) marks zero
    /// response; values above are excitatory, below inhibitory.
    /// Preserves the polarity (orientation + phase) of the carrier.
    #[default]
    Signed,
    /// Absolute response: `|filter output|` linearly remapped from
    /// `[0, R]` to `[0, 255]`. Discards polarity; classical edge /
    /// texture energy.
    Magnitude,
}

/// Gabor filter — oriented Gaussian-modulated cosine kernel.
///
/// See the [module-level docs](self) for the continuous kernel,
/// the parametrisation, the DC-removal step, and the two output
/// modes.
#[derive(Clone, Copy, Debug)]
pub struct Gabor {
    /// Carrier wavelength in pixels (period of the cosine along the
    /// rotated `x'` axis). Must be `≥ 2.0` by the spatial Nyquist
    /// limit. Default `6.0` (a mid-frequency band suitable for
    /// general-purpose oriented-edge response on natural images).
    pub wavelength: f32,
    /// Orientation in degrees CCW from the positive image-x axis.
    /// `0.0` ⇒ vertical stripes (the carrier oscillates along the
    /// image-x axis). Default `0.0`.
    pub orientation_degrees: f32,
    /// Carrier phase offset in degrees. `0.0` ⇒ even-symmetric cosine
    /// (bar-like response); `90.0` ⇒ odd-symmetric sine (step-edge
    /// response). Default `0.0`.
    pub phase_degrees: f32,
    /// Gaussian standard deviation in pixels (envelope spatial support
    /// along the rotated `x'` axis). Default `3.0`. Must be `> 0`.
    pub sigma: f32,
    /// Spatial-aspect ratio (`σ_y / σ_x`). `1.0` ⇒ isotropic envelope;
    /// `< 1.0` ⇒ elongated parallel to the carrier. Default `0.5`
    /// (a V1-tuned ellipticity). Must be `> 0`.
    pub gamma: f32,
    /// Optional explicit kernel half-radius in pixels. `None` ⇒
    /// auto-select `ceil(3·σ·max(1, 1/γ))`. Clamped to `[1, 32]`.
    pub radius: Option<u32>,
    /// Output mode (signed-with-bias vs. magnitude). Default
    /// [`GaborMode::Signed`].
    pub mode: GaborMode,
    /// Global multiplier applied to the response before remapping into
    /// `[0, 255]`. Default `1.0`; values `> 1` saturate weaker
    /// responses to full white.
    pub output_gain: f32,
}

impl Default for Gabor {
    fn default() -> Self {
        Self {
            wavelength: 6.0,
            orientation_degrees: 0.0,
            phase_degrees: 0.0,
            sigma: 3.0,
            gamma: 0.5,
            radius: None,
            mode: GaborMode::Signed,
            output_gain: 1.0,
        }
    }
}

impl Gabor {
    /// Build a Gabor filter with the default parameters
    /// (`wavelength = 6`, `orientation = 0°`, `phase = 0°`,
    /// `sigma = 3`, `gamma = 0.5`, auto-radius, signed mode,
    /// gain `1.0`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the carrier wavelength (must be `≥ 2.0`; clamped).
    pub fn with_wavelength(mut self, wavelength: f32) -> Self {
        self.wavelength = wavelength.max(2.0);
        self
    }

    /// Override the orientation in degrees CCW.
    pub fn with_orientation_degrees(mut self, orientation: f32) -> Self {
        self.orientation_degrees = orientation;
        self
    }

    /// Override the carrier phase offset in degrees.
    pub fn with_phase_degrees(mut self, phase: f32) -> Self {
        self.phase_degrees = phase;
        self
    }

    /// Override the Gaussian envelope standard deviation in pixels.
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma.max(f32::EPSILON);
        self
    }

    /// Override the spatial-aspect ratio `γ = σ_y / σ_x`.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma.max(f32::EPSILON);
        self
    }

    /// Pin the kernel half-radius (overrides the
    /// `ceil(3·σ·max(1, 1/γ))` auto-selection). Clamped to `[1, 32]`.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = Some(radius.clamp(1, 32));
        self
    }

    /// Override the output mode (signed-with-bias vs. magnitude).
    pub fn with_mode(mut self, mode: GaborMode) -> Self {
        self.mode = mode;
        self
    }

    /// Switch to magnitude output mode.
    pub fn magnitude() -> Self {
        Self {
            mode: GaborMode::Magnitude,
            ..Self::default()
        }
    }

    /// Override the global output gain (default `1.0`).
    pub fn with_output_gain(mut self, gain: f32) -> Self {
        self.output_gain = gain;
        self
    }
}

impl ImageFilter for Gabor {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Gabor does not yet handle {:?}",
                params.format
            )));
        }
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(Error::invalid("Gabor: sigma must be > 0".to_string()));
        }
        if !(self.gamma.is_finite() && self.gamma > 0.0) {
            return Err(Error::invalid("Gabor: gamma must be > 0".to_string()));
        }
        if !(self.wavelength.is_finite() && self.wavelength >= 2.0) {
            return Err(Error::invalid(
                "Gabor: wavelength must be >= 2 (spatial Nyquist)".to_string(),
            ));
        }
        if !self.output_gain.is_finite() {
            return Err(Error::invalid(
                "Gabor: output_gain must be finite".to_string(),
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        // Auto-radius spans 3σ along the elongated axis. The carrier's
        // envelope is wider along the perpendicular axis when γ < 1, so
        // we expand by max(1, 1/γ).
        let radius = self
            .radius
            .unwrap_or_else(|| {
                let aspect_factor = if self.gamma < 1.0 {
                    1.0 / self.gamma
                } else {
                    1.0
                };
                (self.sigma * 3.0 * aspect_factor).ceil().max(1.0) as u32
            })
            .clamp(1, 32) as usize;
        let (kernel, scale) = build_kernel(
            radius,
            self.wavelength,
            self.orientation_degrees,
            self.phase_degrees,
            self.sigma,
            self.gamma,
        );
        let response = convolve(&luma, w, h, &kernel, radius);
        let out = match self.mode {
            GaborMode::Signed => render_signed(&response, scale, self.output_gain),
            GaborMode::Magnitude => render_magnitude(&response, scale, self.output_gain),
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
// Kernel construction.
// ----------------------------------------------------------------------

/// Build a `(2r + 1)²` Gabor kernel sampled from the continuous
/// `exp(-(x'² + γ²·y'²) / (2σ²)) · cos(2π·x'/λ + ψ)` and then
/// zero-meaned so flat input maps to exactly zero. Also returns the
/// renderer normalisation `scale = Σ |K'(i, j)| / 2` which equals the
/// maximum absolute discrete response against a unit-amplitude
/// 0..255 grating (so dividing the response by `scale · 255` lands the
/// signal back in `[-1, +1]`).
pub(crate) fn build_kernel(
    radius: usize,
    wavelength: f32,
    orientation_degrees: f32,
    phase_degrees: f32,
    sigma: f32,
    gamma: f32,
) -> (Vec<f32>, f32) {
    use core::f64::consts::PI;
    let n = 2 * radius + 1;
    let mut k = vec![0.0f32; n * n];
    let theta = (orientation_degrees as f64).to_radians();
    let psi = (phase_degrees as f64).to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sigma_d = sigma as f64;
    let two_sigma2 = 2.0 * sigma_d * sigma_d;
    let gamma2 = (gamma as f64) * (gamma as f64);
    let lambda = wavelength as f64;
    let r_i = radius as i32;
    for j in 0..n {
        let dy = (j as i32 - r_i) as f64;
        for i in 0..n {
            let dx = (i as i32 - r_i) as f64;
            let xp = dx * cos_t + dy * sin_t;
            let yp = -dx * sin_t + dy * cos_t;
            let envelope = (-(xp * xp + gamma2 * yp * yp) / two_sigma2).exp();
            let carrier = (2.0 * PI * xp / lambda + psi).cos();
            k[j * n + i] = (envelope * carrier) as f32;
        }
    }
    // Zero-mean correction: subtract the discrete arithmetic mean so a
    // flat input produces exactly zero response.
    let mean: f64 = k.iter().map(|v| *v as f64).sum::<f64>() / (k.len() as f64);
    let mean_f32 = mean as f32;
    for v in k.iter_mut() {
        *v -= mean_f32;
    }
    // Renderer scale: half the L1 norm. A constant-grating fixture in
    // [0, 255] reaches at most this absolute response, so dividing the
    // raw response by `scale` lands magnitudes in [0, 255].
    let abs_sum: f64 = k.iter().map(|v| (*v as f64).abs()).sum();
    let scale = (abs_sum / 2.0).max(f64::EPSILON) as f32;
    (k, scale)
}

fn convolve(luma: &[u8], w: usize, h: usize, kernel: &[f32], radius: usize) -> Vec<f32> {
    let n = 2 * radius + 1;
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            // Accumulate in f64 so the zero-mean kernel actually
            // produces ~zero on flat input regardless of order.
            let mut acc = 0.0f64;
            for kj in 0..n {
                let sy = (y as i32 + kj as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                let row = sy * w;
                for ki in 0..n {
                    let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                    acc += kernel[kj * n + ki] as f64 * luma[row + sx] as f64;
                }
            }
            out[y * w + x] = acc as f32;
        }
    }
    out
}

fn render_signed(response: &[f32], scale: f32, gain: f32) -> Vec<u8> {
    // Linearly remap [-scale, +scale] (full discrete response range) to
    // [0, 255] with 128 marking zero. Apply gain before the remap so
    // callers can saturate weak responses. The asymmetric 127/128 split
    // around 128 (instead of 127.5) ensures small positive responses
    // round up away from the neutral grey value and small negative
    // responses round down — preventing flat input from oscillating
    // between 127 and 128 with float quantisation noise.
    let inv = 1.0 / scale;
    response
        .iter()
        .map(|v| {
            let scaled = (*v * inv * gain).clamp(-1.0, 1.0);
            // 128.0 is the neutral midpoint; positive responses occupy
            // [128, 255] (127 cells), negative responses occupy
            // [0, 128) (128 cells). Linear scaling by 127 keeps both
            // halves bit-exact in the worst case.
            let mapped = 128.0 + scaled * 127.0;
            mapped.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn render_magnitude(response: &[f32], scale: f32, gain: f32) -> Vec<u8> {
    // Linearly remap |response| from [0, scale] to [0, 255].
    let inv = 255.0 / scale;
    response
        .iter()
        .map(|v| {
            let m = (v.abs() * gain * inv).clamp(0.0, 255.0);
            m.round() as u8
        })
        .collect()
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

    #[test]
    fn kernel_is_zero_mean_after_correction() {
        // Sweep a few (sigma, gamma, wavelength, theta, psi) tuples;
        // every kernel must sum to ~0 after the DC-removal step.
        for sigma in [0.5_f32, 1.0, 2.0, 3.0, 5.0] {
            for gamma in [0.5_f32, 1.0, 2.0] {
                for lambda in [3.0_f32, 6.0, 10.0] {
                    for theta in [0.0_f32, 30.0, 45.0, 90.0, 135.0] {
                        for psi in [0.0_f32, 45.0, 90.0] {
                            let radius = (sigma * 3.0).ceil().max(1.0) as usize;
                            let (k, _) = build_kernel(radius, lambda, theta, psi, sigma, gamma);
                            let sum: f64 = k.iter().map(|v| *v as f64).sum();
                            assert!(
                                sum.abs() < 1e-4,
                                "kernel DC sum {sum} at sigma={sigma} gamma={gamma} lambda={lambda} theta={theta} psi={psi}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn flat_input_signed_is_neutral_grey() {
        // Flat 128 input: zero-mean kernel ⇒ exactly zero response ⇒
        // signed remap puts every pixel at 128.
        let input = gray(20, 20, |_, _| 128);
        let out = Gabor::new().apply(&input, p_gray(20, 20)).unwrap();
        // Interior pixels (away from clamping artefacts in the response
        // tail) must round to 128.
        for y in 8..12 {
            for x in 8..12 {
                let v = out.planes[0].data[y * 20 + x];
                assert_eq!(v, 128, "flat field deviates at ({x}, {y}): {v}");
            }
        }
    }

    #[test]
    fn flat_input_magnitude_is_zero() {
        let input = gray(20, 20, |_, _| 200);
        let out = Gabor::magnitude().apply(&input, p_gray(20, 20)).unwrap();
        for y in 8..12 {
            for x in 8..12 {
                assert_eq!(out.planes[0].data[y * 20 + x], 0);
            }
        }
    }

    #[test]
    fn horizontal_grating_excites_zero_orientation() {
        // A vertical-stripe carrier matching the filter's
        // orientation=0/wavelength=8 should produce a strong response
        // (well off neutral grey) in signed mode.
        let lambda = 8.0_f32;
        let input = gray(40, 40, |x, _| {
            // Cosine carrier mapped to [0, 255].
            let phase = 2.0 * core::f32::consts::PI * (x as f32) / lambda;
            ((phase.cos() * 0.5 + 0.5) * 255.0).round() as u8
        });
        let g = Gabor::new()
            .with_wavelength(lambda)
            .with_orientation_degrees(0.0)
            .with_sigma(3.0)
            .with_gamma(0.5);
        let out = g.apply(&input, p_gray(40, 40)).unwrap();
        // Interior centre column: with the aligned carrier we expect
        // saturation toward 0 or 255 (way off neutral grey). The exact
        // value depends on phase but the deviation from 128 should be
        // large.
        let mut max_dev = 0i32;
        for y in 16..24 {
            for x in 16..24 {
                let dev = (out.planes[0].data[y * 40 + x] as i32 - 128).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }
        assert!(
            max_dev > 50,
            "aligned grating should excite signed response; got max deviation {max_dev}"
        );
    }

    #[test]
    fn perpendicular_grating_does_not_excite() {
        // A vertical-stripe grating runs along the y-axis, so its
        // spatial-frequency vector points along x. A Gabor with
        // orientation θ=0 has its carrier oscillating along x and so
        // *matches* the grating's frequency vector — high response. The
        // same Gabor rotated to θ=90 has its carrier oscillating along
        // y, which is constant along the grating's stripes — so the
        // mean response over the constant-along-y signal must be much
        // weaker. Tests the orientation selectivity that gives Gabor
        // its name as an oriented filter. Use mean (not max) to avoid
        // boundary-clamping artefacts dominating the comparison.
        let lambda = 8.0_f32;
        let input = gray(48, 48, |x, _| {
            let phase = 2.0 * core::f32::consts::PI * (x as f32) / lambda;
            ((phase.cos() * 0.5 + 0.5) * 255.0).round() as u8
        });
        // Use γ=0.3 (strongly elongated) and σ=4 so the filter integrates
        // multiple stripe periods along the carrier direction. A θ=0
        // filter senses the carrier; a θ=90 filter integrates along the
        // constant-along-y stripes and so produces a much weaker response.
        let aligned = Gabor::new()
            .with_wavelength(lambda)
            .with_orientation_degrees(0.0)
            .with_sigma(4.0)
            .with_gamma(0.3)
            .with_mode(GaborMode::Magnitude);
        let perpendicular = Gabor::new()
            .with_wavelength(lambda)
            .with_orientation_degrees(90.0)
            .with_sigma(4.0)
            .with_gamma(0.3)
            .with_mode(GaborMode::Magnitude);
        let out_a = aligned.apply(&input, p_gray(48, 48)).unwrap();
        let out_p = perpendicular.apply(&input, p_gray(48, 48)).unwrap();
        // Compare max in the deep interior — the aligned response peaks
        // at the carrier crests and so its max is large, while the
        // perpendicular response is bounded by the stripe-average
        // residual.
        let mut max_a = 0u8;
        let mut max_p = 0u8;
        for y in 18..30 {
            for x in 18..30 {
                if out_a.planes[0].data[y * 48 + x] > max_a {
                    max_a = out_a.planes[0].data[y * 48 + x];
                }
                if out_p.planes[0].data[y * 48 + x] > max_p {
                    max_p = out_p.planes[0].data[y * 48 + x];
                }
            }
        }
        assert!(
            max_a as i32 > max_p as i32 + 30 && max_a > 100,
            "orientation selectivity broken: max aligned={max_a} perpendicular={max_p}"
        );
    }

    #[test]
    fn rotating_filter_180_preserves_magnitude() {
        // The Gabor kernel with phase=0 is even-symmetric, so rotating
        // by 180° (which sends x' to -x' for both axes simultaneously)
        // produces an identical real kernel up to sign of the carrier;
        // the cosine carrier with ψ=0 is even, so the kernels are
        // literally identical. Magnitude output therefore must match.
        let g0 = Gabor::new().with_orientation_degrees(0.0);
        let g180 = Gabor::new().with_orientation_degrees(180.0);
        let input = gray(24, 24, |x, y| ((x + y) * 7 % 256) as u8);
        let p = p_gray(24, 24);
        let out0 = g0.apply(&input, p).unwrap();
        let out180 = g180.apply(&input, p).unwrap();
        // Interior pixels (away from the kernel-tail clamping) must
        // agree exactly.
        for y in 8..16 {
            for x in 8..16 {
                let a = out0.planes[0].data[y * 24 + x];
                let b = out180.planes[0].data[y * 24 + x];
                assert!(
                    (a as i32 - b as i32).abs() <= 1,
                    "180° rotation breaks even symmetry at ({x}, {y}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn phase_90_is_odd_symmetric_so_responds_to_step() {
        // A ψ=90 (sine carrier) Gabor is odd-symmetric along x' and so
        // peaks on step edges instead of bars. Construct a vertical
        // step edge: left half dark, right half bright. Expect a strong
        // response in magnitude mode close to the seam, weaker far from
        // it.
        let input = gray(40, 40, |x, _| if x < 20 { 30 } else { 220 });
        let g = Gabor::new()
            .with_orientation_degrees(0.0)
            .with_phase_degrees(90.0)
            .with_wavelength(8.0)
            .with_sigma(3.0)
            .with_gamma(0.5)
            .with_mode(GaborMode::Magnitude);
        let out = g.apply(&input, p_gray(40, 40)).unwrap();
        // Seam column 19/20 must produce a meaningfully larger response
        // than column 5 (far interior of the dark side, well outside the
        // kernel support).
        let mut seam_max = 0u8;
        for y in 8..32 {
            for x in 17..23 {
                if out.planes[0].data[y * 40 + x] > seam_max {
                    seam_max = out.planes[0].data[y * 40 + x];
                }
            }
        }
        let mut far = 0u8;
        for y in 8..32 {
            if out.planes[0].data[y * 40 + 5] > far {
                far = out.planes[0].data[y * 40 + 5];
            }
        }
        assert!(
            seam_max as i32 > 4 * far as i32 + 30,
            "odd-symmetric Gabor should respond to step edge: seam={seam_max} far={far}"
        );
    }

    #[test]
    fn rgb_input_collapses_to_luma() {
        // RGB input collapses to (R + 2G + B) / 4 luma before
        // convolution. A flat RGB field must still produce the neutral
        // 128 in signed mode.
        let mut data = Vec::with_capacity(20 * 20 * 3);
        for _ in 0..(20 * 20) {
            data.extend_from_slice(&[100, 150, 200]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 60, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 20,
            height: 20,
        };
        let out = Gabor::new().apply(&input, p).unwrap();
        // Output must be single-plane Gray8 (one byte per pixel).
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 20 * 20);
        // Interior pixels: flat luma → zero response → neutral grey.
        for y in 8..12 {
            for x in 8..12 {
                assert_eq!(out.planes[0].data[y * 20 + x], 128);
            }
        }
    }

    #[test]
    fn yuv420p_uses_luma_plane_directly() {
        // Yuv420P: Gabor reads only the Y plane; chroma is ignored.
        let w = 16usize;
        let h = 16usize;
        let y = vec![128u8; w * h];
        let u = vec![64u8; (w / 2) * (h / 2)];
        let v = vec![192u8; (w / 2) * (h / 2)];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: w / 2,
                    data: u,
                },
                VideoPlane {
                    stride: w / 2,
                    data: v,
                },
            ],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: w as u32,
            height: h as u32,
        };
        let out = Gabor::new().apply(&input, p).unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), w * h);
        for y in 6..10 {
            for x in 6..10 {
                assert_eq!(out.planes[0].data[y * w + x], 128);
            }
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        let input = gray(8, 8, |_, _| 128);
        let p = p_gray(8, 8);
        // The builders clamp into the valid range; reach past them to
        // exercise the runtime guards directly.
        // sigma must be > 0
        let mut g = Gabor::new();
        g.sigma = 0.0;
        assert!(g.apply(&input, p).is_err());
        let mut g = Gabor::new();
        g.sigma = f32::NAN;
        assert!(g.apply(&input, p).is_err());
        // gamma must be > 0
        let mut g = Gabor::new();
        g.gamma = 0.0;
        assert!(g.apply(&input, p).is_err());
        // wavelength < 2 violates spatial Nyquist
        let mut g = Gabor::new();
        g.wavelength = 1.0;
        assert!(g.apply(&input, p).is_err());
        // non-finite gain
        let mut g = Gabor::new();
        g.output_gain = f32::NAN;
        assert!(g.apply(&input, p).is_err());
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![128; 64],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P10Le,
            width: 8,
            height: 8,
        };
        let err = Gabor::new().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("Gabor"));
    }

    #[test]
    fn radius_override_clamps_to_range() {
        // Out-of-range radius requests get clamped to [1, 32].
        let g = Gabor::new().with_radius(0);
        assert_eq!(g.radius, Some(1));
        let g = Gabor::new().with_radius(1000);
        assert_eq!(g.radius, Some(32));
    }

    #[test]
    fn pts_pass_through() {
        let mut input = gray(8, 8, |_, _| 100);
        input.pts = Some(42);
        let out = Gabor::new().apply(&input, p_gray(8, 8)).unwrap();
        assert_eq!(out.pts, Some(42));
    }

    #[test]
    fn builder_methods_compose() {
        let g = Gabor::new()
            .with_wavelength(7.5)
            .with_orientation_degrees(45.0)
            .with_phase_degrees(30.0)
            .with_sigma(2.5)
            .with_gamma(0.8)
            .with_radius(7)
            .with_mode(GaborMode::Magnitude)
            .with_output_gain(2.0);
        assert_eq!(g.wavelength, 7.5);
        assert_eq!(g.orientation_degrees, 45.0);
        assert_eq!(g.phase_degrees, 30.0);
        assert_eq!(g.sigma, 2.5);
        assert_eq!(g.gamma, 0.8);
        assert_eq!(g.radius, Some(7));
        assert_eq!(g.mode, GaborMode::Magnitude);
        assert_eq!(g.output_gain, 2.0);
    }

    #[test]
    fn small_input_smaller_than_kernel_still_works() {
        // 3×3 input with default kernel radius (≥ 6 for default sigma).
        // Border clamping must keep the convolution defined.
        let input = gray(3, 3, |_, _| 128);
        let out = Gabor::new().apply(&input, p_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].data.len(), 9);
        // Flat input ⇒ neutral grey everywhere.
        for v in &out.planes[0].data {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn magnitude_is_nonnegative_and_signed_centred() {
        // Sanity: magnitude output never lands below 0 (trivially true
        // for u8) and the signed mode centres unbiased noise around 128
        // because the kernel sums to zero.
        let g_mag = Gabor::magnitude();
        let g_sgn = Gabor::new();
        // Pseudo-random fill from a deterministic LCG. Use the closure
        // arguments to pick the seed step so the helper can stay `Fn`.
        let input = gray(32, 32, |x, y| {
            let mut seed: u32 = 0x12345u32
                .wrapping_add(x.wrapping_mul(2654435761))
                .wrapping_add(y.wrapping_mul(40503));
            for _ in 0..3 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            }
            ((seed >> 16) & 0xff) as u8
        });
        let p = p_gray(32, 32);
        let _ = g_mag.apply(&input, p).unwrap();
        let out_sgn = g_sgn.apply(&input, p).unwrap();
        // Average signed output over the deep interior should be close
        // to 128 (the kernel is zero-sum and the input is roughly
        // uniform).
        let mut acc = 0i64;
        let mut n = 0i64;
        for y in 10..22 {
            for x in 10..22 {
                acc += out_sgn.planes[0].data[y * 32 + x] as i64;
                n += 1;
            }
        }
        let mean = acc as f64 / n as f64;
        // Random noise won't be perfectly uniform; the kernel response is
        // a weighted sum of an unbiased noise mean, so the output mean
        // sits near 128 with some sample-dependent drift. A generous
        // ±40 window covers reasonable LCG fluctuations on a 12×12 patch.
        assert!(
            (mean - 128.0).abs() < 40.0,
            "signed Gabor of noise should centre near 128; got mean={mean}"
        );
    }
}
