//! Edge-aware smoothing and sharpening of issue #12: `-adaptive-blur`,
//! `-adaptive-sharpen`, `-selective-blur`, `-enhance`,
//! `-wavelet-denoise` and `-mean-shift`.
//!
//! All run in binary32 on 8-bit, 16-bit and float pictures, row bands
//! under the thread budget, output identical at every budget.
//!
//! - **`AdaptiveBlur`** — a Gaussian whose width shrinks where the
//!   picture has an edge. The edge strength `e ∈ [0, 1]` is the
//!   gradient magnitude of the slightly smoothed intensity divided by
//!   its largest value; a pixel is blurred with `σ · (1 − e)`. The
//!   variable-width blur is read from a stack of five blurs at `σ · k /
//!   4`, interpolated linearly between its two nearest members.
//! - **`AdaptiveSharpen`** — the reverse: `v + e · (v − G_σ ∗ v)`, the
//!   unsharp detail added in proportion to the edge strength, so flat
//!   areas (and their noise) are left alone.
//! - **`SelectiveBlur`** — a Gaussian mean over only those neighbours
//!   whose intensity is within `threshold` of the centre pixel's: noise
//!   inside a region is averaged away, a step larger than the threshold
//!   is not crossed.
//! - **`Enhance`** — the same idea with a fixed 5 × 5 binomial window
//!   and the test on the colour distance, for cleaning a noisy picture.
//! - **`WaveletDenoise`** — each channel is split into detail layers by
//!   the à-trous scheme (`c₀` the channel, `cⱼ₊₁ = cⱼ ∗ hⱼ` with `h` the
//!   five-tap binomial spread by `2ʲ`, detail `wⱼ = cⱼ − cⱼ₊₁`); details
//!   smaller than the noise level are shrunk, and the layers are summed
//!   again. White noise of deviation `t` leaves a deviation `t · gⱼ` in
//!   layer `j`, where `gⱼ` is the norm of that layer's kernel (computed
//!   from the binomial itself), so layer `j` is thresholded at `t · gⱼ`.
//!   `softness` blends hard thresholding (0: keep or kill) with soft
//!   (1: shrink towards zero by the threshold).
//! - **`MeanShift`** — every pixel walks to the mode of its
//!   neighbourhood: the mean position and colour of the window pixels
//!   whose colour is within `distance` of the current colour replace
//!   the current ones until they stop moving; the pixel takes the
//!   colour it arrives at. Regions flatten, edges stay.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::tone_auto::gaussian;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

fn finite(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

fn intensity_plane(img: &FloatImage) -> Vec<f32> {
    let (ch, cc) = (img.channels(), img.color_channels());
    img.samples()
        .chunks_exact(ch)
        .map(|px| {
            if cc >= 3 {
                0.212_656 * px[0] + 0.715_158 * px[1] + 0.072_186 * px[2]
            } else {
                px[0]
            }
        })
        .collect()
}

/// Edge strength in `[0, 1]`: central-difference gradient magnitude of
/// the intensity smoothed at σ = 1, divided by its maximum.
fn edge_strength(img: &FloatImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let smooth =
        crate::tone_auto::blur_plane(&intensity_plane(img), img.width(), img.height(), 1.0);
    let at = |x: usize, y: usize| smooth[y * w + x];
    let mut e = vec![0.0f32; w * h];
    crate::exec::par_rows(&mut e, w.max(1), |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let gx = at((x + 1).min(w - 1), y) - at(x.saturating_sub(1), y);
            let gy = at(x, (y + 1).min(h - 1)) - at(x, y.saturating_sub(1));
            *out = gx.hypot(gy);
        }
    });
    let max = e
        .iter()
        .fold(0.0f32, |m, &v| if v.is_finite() { m.max(v) } else { m });
    if max > 0.0 {
        e.iter_mut().for_each(|v| *v = finite(*v / max, 0.0));
    }
    e
}

/// Blur less where the picture has an edge.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveBlur {
    /// The Gaussian's standard deviation away from edges.
    pub sigma: f32,
}

impl AdaptiveBlur {
    /// Adaptive blur of at most `sigma`.
    pub fn new(sigma: f32) -> Self {
        Self { sigma }
    }
}

impl PixelFilter for AdaptiveBlur {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            const STEPS: usize = 4;
            let sigma = finite(self.sigma, 0.0).max(0.0);
            let edge = edge_strength(&img);
            let stack: Vec<FloatImage> = (0..=STEPS)
                .map(|k| {
                    let g = gaussian(sigma * k as f32 / STEPS as f32);
                    deep::convolve_separable_covered(&img, &g, &g)
                })
                .collect();
            let mut out = img.clone();
            let ch = img.channels();
            let row = img.width() as usize * ch;
            let w = img.width() as usize;
            crate::exec::par_rows(out.samples_mut(), row.max(1), |y, line| {
                for (x, px) in line.chunks_exact_mut(ch).enumerate() {
                    let s = (1.0 - edge[y * w + x]).clamp(0.0, 1.0) * STEPS as f32;
                    let k = (s.floor() as usize).min(STEPS - 1);
                    let f = s - k as f32;
                    let i = (y * w + x) * ch;
                    let (a, b) = (
                        &stack[k].samples()[i..i + ch],
                        &stack[k + 1].samples()[i..i + ch],
                    );
                    for ((o, p), q) in px.iter_mut().zip(a).zip(b) {
                        *o = p + (q - p) * f;
                    }
                }
            });
            Ok(out)
        })
    }
}

/// Sharpen more where the picture has an edge.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveSharpen {
    /// Standard deviation of the Gaussian whose detail is added back.
    pub sigma: f32,
    /// Detail gain at the strongest edge.
    pub amount: f32,
}

impl AdaptiveSharpen {
    /// Adaptive sharpening at `sigma` with unit gain.
    pub fn new(sigma: f32) -> Self {
        Self { sigma, amount: 1.0 }
    }

    /// Set the gain at the strongest edge.
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.amount = amount;
        self
    }
}

impl PixelFilter for AdaptiveSharpen {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let g = gaussian(finite(self.sigma, 0.0).max(0.0));
            let amount = finite(self.amount, 0.0);
            let edge = edge_strength(&img);
            let soft = deep::convolve_separable_covered(&img, &g, &g);
            let mut out = img.clone();
            let (ch, cc) = (img.channels(), img.color_channels());
            for ((px, s), e) in out
                .samples_mut()
                .chunks_exact_mut(ch)
                .zip(soft.samples().chunks_exact(ch))
                .zip(&edge)
            {
                for (v, b) in px[..cc].iter_mut().zip(s) {
                    *v += amount * e * (*v - b);
                }
            }
            Ok(out)
        })
    }
}

/// The window mean shared by [`SelectiveBlur`] and [`Enhance`]: a
/// kernel-weighted mean over the neighbours `close(centre, neighbour)`
/// admits (the centre always is).
fn gated_mean(
    img: &FloatImage,
    kernel: &[f32],
    close: impl Fn(&[f32], &[f32]) -> bool + Sync,
) -> FloatImage {
    let (w, h, ch) = (img.width() as usize, img.height() as usize, img.channels());
    let r = (kernel.len() / 2) as i64;
    let src = img.samples();
    let mut out = img.clone();
    crate::exec::par_rows(out.samples_mut(), (w * ch).max(1), |y, line| {
        let mut acc = vec![0.0f32; ch];
        for (x, px) in line.chunks_exact_mut(ch).enumerate() {
            let centre = &src[(y * w + x) * ch..][..ch];
            acc.fill(0.0);
            let mut weight = 0.0f32;
            for dy in -r..=r {
                let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                let ky = kernel[(dy + r) as usize];
                for dx in -r..=r {
                    let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    let other = &src[(sy * w + sx) * ch..][..ch];
                    if (dx == 0 && dy == 0) || close(centre, other) {
                        let k = ky * kernel[(dx + r) as usize];
                        weight += k;
                        for (a, v) in acc.iter_mut().zip(other) {
                            *a += k * v;
                        }
                    }
                }
            }
            if weight > 0.0 {
                for (o, a) in px.iter_mut().zip(&acc) {
                    *o = a / weight;
                }
            }
        }
    });
    out
}

/// Gaussian blur that does not cross a contrast step.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectiveBlur {
    /// The Gaussian's standard deviation.
    pub sigma: f32,
    /// Largest intensity difference (normalised) a neighbour may have.
    pub threshold: f32,
}

impl SelectiveBlur {
    /// Selective blur at `sigma` within `threshold`.
    pub fn new(sigma: f32, threshold: f32) -> Self {
        Self { sigma, threshold }
    }
}

impl PixelFilter for SelectiveBlur {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let kernel = gaussian(finite(self.sigma, 0.0).clamp(0.0, 64.0));
            let t = finite(self.threshold, 0.0);
            // The gate reads one intensity plane; the window walks whole
            // rows of it (edge columns resolved once).
            let lum = intensity_plane(&img);
            let (w, h, ch) = (img.width() as usize, img.height() as usize, img.channels());
            let r = kernel.len() / 2;
            let cols: Vec<usize> = (0..w + 2 * r)
                .map(|p| p.saturating_sub(r).min(w.saturating_sub(1)))
                .collect();
            let src = img.samples();
            let mut out = img.clone();
            crate::exec::par_rows(out.samples_mut(), (w * ch).max(1), |y, line| {
                for (x, px) in line.chunks_exact_mut(ch).enumerate() {
                    let centre = lum[y * w + x];
                    let mut acc = [0.0f32; 4];
                    let mut weight = 0.0f32;
                    for (dy, &ky) in kernel.iter().enumerate() {
                        let sy = (y + dy).saturating_sub(r).min(h - 1);
                        let (lrow, srow) = (&lum[sy * w..][..w], &src[sy * w * ch..][..w * ch]);
                        for (&sx, &kx) in cols[x..].iter().zip(&kernel) {
                            if (lrow[sx] - centre).abs() <= t {
                                let k = ky * kx;
                                weight += k;
                                for (a, v) in acc.iter_mut().zip(&srow[sx * ch..][..ch]) {
                                    *a += k * v;
                                }
                            }
                        }
                    }
                    if weight > 0.0 {
                        for (o, a) in px.iter_mut().zip(&acc) {
                            *o = a / weight;
                        }
                    }
                }
            });
            Ok(out)
        })
    }
}

/// Clean a noisy picture: a 5 × 5 binomial mean over the neighbours
/// whose colour is close to the pixel's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enhance {
    /// Largest colour distance (Euclidean, normalised) a neighbour may
    /// have.
    pub distance: f32,
}

impl Enhance {
    /// The operator at its default distance (0.1).
    pub fn new() -> Self {
        Self { distance: 0.1 }
    }

    /// Set the colour distance.
    pub fn with_distance(mut self, distance: f32) -> Self {
        self.distance = distance;
        self
    }
}

impl Default for Enhance {
    fn default() -> Self {
        Self::new()
    }
}

impl PixelFilter for Enhance {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let binomial = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
            let limit = finite(self.distance, 0.0).powi(2);
            Ok(gated_mean(&img, &binomial, |a, b| {
                a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() <= limit
            }))
        })
    }
}

/// Remove noise by shrinking small wavelet details.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveletDenoise {
    /// The noise level (standard deviation, normalised units).
    pub threshold: f32,
    /// 0: details under the threshold are removed, the rest kept; 1:
    /// every detail is shrunk by the threshold.
    pub softness: f32,
    /// Detail layers (1 … 8).
    pub levels: u32,
}

impl WaveletDenoise {
    /// Denoise at `threshold` with `softness`, five layers.
    pub fn new(threshold: f32, softness: f32) -> Self {
        Self {
            threshold,
            softness,
            levels: 5,
        }
    }

    /// Set the number of detail layers.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels.clamp(1, 8);
        self
    }
}

/// The five-tap binomial with `2^level − 1` zeros between taps.
fn holed_binomial(level: u32) -> Vec<f32> {
    let step = 1usize << level;
    let mut k = vec![0.0f32; 4 * step + 1];
    for (i, w) in [1.0, 4.0, 6.0, 4.0, 1.0].into_iter().enumerate() {
        k[i * step] = w / 16.0;
    }
    k
}

/// Full convolution of two 1-D kernels.
fn convolve_kernels(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// The deviation white noise of unit deviation leaves in each detail
/// layer: the norm of `aⱼ ⊗ aⱼ − aⱼ₊₁ ⊗ aⱼ₊₁`, `aⱼ` the 1-D smoothing
/// kernel after `j` passes.
fn layer_gains(levels: u32) -> Vec<f32> {
    let mut smooth = vec![1.0f64];
    let mut gains = Vec::with_capacity(levels as usize);
    for level in 0..levels {
        let hole: Vec<f64> = holed_binomial(level).iter().map(|&v| v as f64).collect();
        let next = convolve_kernels(&smooth, &hole);
        // Centre `smooth` inside `next` to take their inner product.
        let pad = (next.len() - smooth.len()) / 2;
        let dot: f64 = smooth.iter().zip(&next[pad..]).map(|(a, b)| a * b).sum();
        let (na, nb): (f64, f64) = (
            smooth.iter().map(|v| v * v).sum(),
            next.iter().map(|v| v * v).sum(),
        );
        gains.push((na * na - 2.0 * dot * dot + nb * nb).max(0.0).sqrt() as f32);
        smooth = next;
    }
    gains
}

impl PixelFilter for WaveletDenoise {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let levels = self.levels.clamp(1, 8);
            let t = finite(self.threshold, 0.0).max(0.0);
            let soft = finite(self.softness, 0.0).clamp(0.0, 1.0);
            let gains = layer_gains(levels);
            let cc = img.color_channels();
            let ch = img.channels();
            let mut current = img.clone();
            let mut kept = vec![0.0f32; img.samples().len()];
            for (level, gain) in gains.iter().enumerate() {
                let k = holed_binomial(level as u32);
                let next = deep::convolve_separable(&current, &k, &k, false);
                let limit = t * gain;
                for ((acc, c), n) in kept
                    .chunks_exact_mut(ch)
                    .zip(current.samples().chunks_exact(ch))
                    .zip(next.samples().chunks_exact(ch))
                {
                    for ((a, c), n) in acc[..cc].iter_mut().zip(c).zip(n) {
                        let d = c - n;
                        let hard = if d.abs() > limit { d } else { 0.0 };
                        let shrunk = d.signum() * (d.abs() - limit).max(0.0);
                        *a += finite(hard + (shrunk - hard) * soft, 0.0);
                    }
                }
                current = next;
            }
            let mut out = img.clone();
            for ((o, c), k) in out
                .samples_mut()
                .chunks_exact_mut(ch)
                .zip(current.samples().chunks_exact(ch))
                .zip(kept.chunks_exact(ch))
            {
                for ((v, c), k) in o[..cc].iter_mut().zip(c).zip(k) {
                    *v = c + k;
                }
            }
            Ok(out)
        })
    }
}

/// Flatten regions by moving every pixel to the mode of its
/// neighbourhood.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanShift {
    /// Window width in pixels.
    pub width: u32,
    /// Window height in pixels.
    pub height: u32,
    /// Largest colour distance (Euclidean, normalised) inside a mode.
    pub distance: f32,
    /// Most steps a pixel takes.
    pub max_iterations: u32,
}

impl MeanShift {
    /// Mean shift over a `width × height` window within `distance`.
    pub fn new(width: u32, height: u32, distance: f32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            distance,
            max_iterations: 100,
        }
    }

    /// Set the iteration limit.
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n.max(1);
        self
    }
}

impl PixelFilter for MeanShift {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let (w, h, ch) = (img.width() as i64, img.height() as i64, img.channels());
            let cc = img.color_channels();
            let (rx, ry) = (
                (self.width.min(255) / 2) as i64,
                (self.height.min(255) / 2) as i64,
            );
            let limit = finite(self.distance, 0.0).powi(2);
            let src = img.samples();
            let mut out = img.clone();
            crate::exec::par_rows(out.samples_mut(), (w as usize * ch).max(1), |y, line| {
                let mut colour = vec![0.0f32; cc];
                let mut sum = vec![0.0f32; cc];
                for (x, px) in line.chunks_exact_mut(ch).enumerate() {
                    colour.copy_from_slice(&px[..cc]);
                    let (mut cx, mut cy) = (x as f32, y as f32);
                    for _ in 0..self.max_iterations.max(1) {
                        sum.fill(0.0);
                        let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0.0f32);
                        let (ix, iy) = (cx.round() as i64, cy.round() as i64);
                        for v in (iy - ry).max(0)..=(iy + ry).min(h - 1) {
                            for u in (ix - rx).max(0)..=(ix + rx).min(w - 1) {
                                let other = &src[(v * w + u) as usize * ch..][..cc];
                                let d: f32 = other
                                    .iter()
                                    .zip(&colour)
                                    .map(|(p, q)| (p - q) * (p - q))
                                    .sum();
                                if d <= limit {
                                    sum.iter_mut().zip(other).for_each(|(s, p)| *s += p);
                                    sx += u as f32;
                                    sy += v as f32;
                                    n += 1.0;
                                }
                            }
                        }
                        if n == 0.0 {
                            break;
                        }
                        let (nx, ny) = (sx / n, sy / n);
                        let mut moved = (nx - cx).powi(2) + (ny - cy).powi(2);
                        for (c, s) in colour.iter_mut().zip(&sum) {
                            let m = s / n;
                            moved += (m - *c).powi(2);
                            *c = m;
                        }
                        (cx, cy) = (nx, ny);
                        if moved < 1e-8 {
                            break;
                        }
                    }
                    px[..cc].copy_from_slice(&colour);
                }
            });
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> FloatImage {
        let data = (0..w * h).map(|i| f(i % w, i / w)).collect();
        FloatImage::from_samples(w, h, 1, data).unwrap()
    }

    fn run(f: &dyn ImageFilter, img: &FloatImage) -> FloatImage {
        let out = f
            .apply(&img.clone().into_frame(None), img.params())
            .unwrap();
        FloatImage::from_frame(&out, img.params()).unwrap()
    }

    /// A step edge with deterministic ±0.03 noise.
    fn noisy_step() -> FloatImage {
        let mut rng = crate::noise::Rng::new(3);
        let noise: Vec<f32> = (0..48 * 48)
            .map(|_| (rng.unit() as f32 - 0.5) * 0.06)
            .collect();
        gray(48, 48, |x, y| {
            (if x < 24 { 0.25 } else { 0.75 }) + noise[(y * 48 + x) as usize]
        })
    }

    fn deviation(img: &FloatImage, x0: u32, x1: u32, mean: f32) -> f32 {
        let mut s = 0.0;
        let mut n = 0.0;
        for y in 4..44 {
            for x in x0..x1 {
                s += (img.pixel(x, y)[0] - mean).powi(2);
                n += 1.0;
            }
        }
        (s / n).sqrt()
    }

    #[test]
    fn the_edge_aware_smoothers_remove_noise_and_keep_the_step() {
        let img = noisy_step();
        let before = deviation(&img, 4, 20, 0.25);
        let filters: Vec<(&str, Box<dyn ImageFilter>)> = vec![
            ("selective", Box::new(SelectiveBlur::new(2.0, 0.2))),
            ("enhance", Box::new(Enhance::new().with_distance(0.2))),
            ("wavelet", Box::new(WaveletDenoise::new(0.03, 0.5))),
            ("mean-shift", Box::new(MeanShift::new(7, 7, 0.2))),
            ("adaptive", Box::new(AdaptiveBlur::new(2.0))),
        ];
        for (name, f) in &filters {
            let out = run(f.as_ref(), &img);
            let after = deviation(&out, 4, 20, 0.25);
            assert!(after < 0.6 * before, "{name}: noise {before} -> {after}");
            // The step is still a step: the two columns beside it differ
            // by nearly the full height.
            let step = out.pixel(25, 24)[0] - out.pixel(22, 24)[0];
            assert!(step > 0.4, "{name}: step {step}");
        }
    }

    #[test]
    fn flat_pictures_are_fixed_points() {
        let img = gray(16, 12, |_, _| 0.3);
        let filters: Vec<Box<dyn ImageFilter>> = vec![
            Box::new(SelectiveBlur::new(1.5, 0.1)),
            Box::new(Enhance::new()),
            Box::new(WaveletDenoise::new(0.05, 1.0)),
            Box::new(MeanShift::new(5, 5, 0.1)),
            Box::new(AdaptiveBlur::new(3.0)),
            Box::new(AdaptiveSharpen::new(2.0)),
        ];
        for f in &filters {
            let out = run(f.as_ref(), &img);
            assert!(out.samples().iter().all(|v| (v - 0.3).abs() < 1e-5));
        }
    }

    #[test]
    fn adaptive_sharpen_steepens_edges_only() {
        let img = gray(48, 8, |x, _| {
            0.3 + 0.4 / (1.0 + (-(x as f32 - 24.0) / 2.0).exp())
        });
        let out = run(&AdaptiveSharpen::new(2.0).with_amount(2.0), &img);
        let slope = |i: &FloatImage| i.pixel(25, 4)[0] - i.pixel(22, 4)[0];
        assert!(slope(&out) > slope(&img) * 1.1);
        assert!((out.pixel(2, 4)[0] - img.pixel(2, 4)[0]).abs() < 1e-3);
    }

    #[test]
    fn wavelet_layers_sum_back_and_gains_follow_the_kernels() {
        // Threshold 0: the decomposition is exact.
        let img = noisy_step();
        let out = run(&WaveletDenoise::new(0.0, 0.0), &img);
        for (a, b) in out.samples().iter().zip(img.samples()) {
            assert!((a - b).abs() < 1e-5);
        }
        // Layer 0: δ⊗δ − h⊗h with h the binomial; ‖h‖² = 70/256.
        let g = layer_gains(3);
        let n: f64 = 70.0 / 256.0;
        let c: f64 = 6.0 / 16.0;
        let want = (1.0 - 2.0 * c * c + n * n).sqrt();
        assert!((g[0] as f64 - want).abs() < 1e-6, "{} vs {want}", g[0]);
        assert!(g[0] > g[1] && g[1] > g[2]);
    }

    #[test]
    fn mean_shift_merges_a_region_to_one_colour() {
        let img = gray(24, 24, |x, y| {
            if x < 12 {
                0.2 + 0.01 * ((x + y) % 3) as f32
            } else {
                0.8
            }
        });
        let out = run(&MeanShift::new(9, 9, 0.1), &img);
        let v = out.pixel(5, 12)[0];
        assert!((v - 0.21).abs() < 0.006, "{v}");
        assert!((out.pixel(20, 12)[0] - 0.8).abs() < 1e-5);
    }
}
