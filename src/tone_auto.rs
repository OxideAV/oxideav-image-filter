//! Statistics-driven tone operators of issue #12: `-auto-threshold`,
//! `-white-balance`, `-local-contrast` and `-clahe`.
//!
//! All four run in binary32 on 8-bit, 16-bit and float pictures.
//!
//! # `AutoThreshold`
//!
//! The picture's intensity (Rec. 709 weights on the stored values) is
//! binned into 256 levels and a threshold level `t` is chosen from the
//! histogram `p`; every pixel brighter than level `t` becomes white,
//! the rest black (alpha is kept).
//!
//! - **Otsu** — the `t` that maximises the between-class variance
//!   `ω₀ ω₁ (μ₀ − μ₁)²` of the two classes `≤ t` and `> t`.
//! - **Kapur** — the `t` that maximises the sum of the two classes'
//!   entropies, `−Σ (pᵢ/ω) ln (pᵢ/ω)` over each class.
//! - **Triangle** — a line is drawn from the histogram's peak to the
//!   far end of its longer tail; `t` is the level under the line whose
//!   bin is farthest from it.
//!
//! # `WhiteBalance`
//!
//! The grey-world assumption in an opponent space: the picture is
//! taken to CIE L\*a\*b\* (sRGB primaries, D65), the means of `a*` and
//! `b*` — the picture's overall cast — are subtracted from every pixel,
//! and the result is taken back. Lightness is untouched.
//!
//! # `LocalContrast`
//!
//! Lightness minus its wide Gaussian blur is the local detail; the
//! operator adds `strength` times that detail back to the lightness
//! and scales the colour by the lightness ratio, so hue and saturation
//! are kept.
//!
//! # `Clahe`
//!
//! Contrast-limited adaptive histogram equalisation of the lightness.
//! The picture is cut into `tiles_x × tiles_y` tiles; each tile's
//! histogram is clipped at `clip_limit` times its mean bin height, the
//! clipped excess is spread evenly over all bins, and the cumulative
//! histogram becomes that tile's tone map. A pixel takes the bilinear
//! blend of the four maps of the tiles whose centres surround it, so
//! the maps change smoothly across tile borders. A clip limit of 1
//! leaves the picture alone (a flat histogram is the identity map); a
//! large one is plain adaptive equalisation.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// Rec. 709 intensity of a colour slice (the sample of a grey picture).
fn intensity(px: &[f32]) -> f32 {
    if px.len() >= 3 {
        0.212_656 * px[0] + 0.715_158 * px[1] + 0.072_186 * px[2]
    } else {
        px[0]
    }
}

/// The intensity plane of a picture.
fn intensity_plane(img: &FloatImage) -> Vec<f32> {
    let (ch, cc) = (img.channels(), img.color_channels());
    img.samples()
        .chunks_exact(ch)
        .map(|px| intensity(&px[..cc]))
        .collect()
}

/// Replace the intensity of every pixel by `target`, scaling the colour
/// (a black pixel takes the target as grey).
fn set_intensity(img: &mut FloatImage, target: &[f32]) {
    let (ch, cc) = (img.channels(), img.color_channels());
    for (px, &t) in img.samples_mut().chunks_exact_mut(ch).zip(target) {
        let old = intensity(&px[..cc]);
        if old > 1e-6 {
            let k = t / old;
            px[..cc].iter_mut().for_each(|v| *v *= k);
        } else {
            px[..cc].iter_mut().for_each(|v| *v = t);
        }
    }
}

/// How [`AutoThreshold`] picks its level.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AutoThresholdMethod {
    /// Largest between-class variance.
    #[default]
    Otsu,
    /// Largest sum of class entropies.
    Kapur,
    /// Farthest histogram point from the peak-to-tail line.
    Triangle,
}

impl AutoThresholdMethod {
    /// A method by name (`otsu`, `kapur`, `triangle`).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "otsu" => Self::Otsu,
            "kapur" => Self::Kapur,
            "triangle" => Self::Triangle,
            _ => return None,
        })
    }

    /// The threshold level (`0 … bins − 1`) for a histogram: levels
    /// above it are foreground.
    pub fn level(self, histogram: &[u64]) -> usize {
        let n = histogram.len();
        let total: f64 = histogram.iter().map(|&c| c as f64).sum();
        if n < 2 || total == 0.0 {
            return 0;
        }
        match self {
            Self::Otsu => {
                let sum_all: f64 = histogram
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| i as f64 * c as f64)
                    .sum();
                let (mut w0, mut sum0) = (0.0, 0.0);
                let mut best = (-1.0, 0);
                for (t, &c) in histogram.iter().enumerate().take(n - 1) {
                    w0 += c as f64;
                    sum0 += t as f64 * c as f64;
                    let w1 = total - w0;
                    if w0 == 0.0 || w1 == 0.0 {
                        continue;
                    }
                    let d = sum0 / w0 - (sum_all - sum0) / w1;
                    let between = w0 * w1 * d * d;
                    if between > best.0 {
                        best = (between, t);
                    }
                }
                best.1
            }
            Self::Kapur => {
                let p: Vec<f64> = histogram.iter().map(|&c| c as f64 / total).collect();
                // Prefix sums of p and of p ln p.
                let mut acc = (0.0, 0.0);
                let prefix: Vec<(f64, f64)> = p
                    .iter()
                    .map(|&v| {
                        acc.0 += v;
                        if v > 0.0 {
                            acc.1 += v * v.ln();
                        }
                        acc
                    })
                    .collect();
                let (all_w, all_e) = prefix[n - 1];
                let mut best = (f64::NEG_INFINITY, 0);
                for (t, &(w0, e0)) in prefix.iter().enumerate().take(n - 1) {
                    let (w1, e1) = (all_w - w0, all_e - e0);
                    if w0 <= 0.0 || w1 <= 1e-12 {
                        continue;
                    }
                    // −Σ (p/ω) ln (p/ω) = ln ω − (Σ p ln p) / ω.
                    let h = w0.ln() - e0 / w0 + w1.ln() - e1 / w1;
                    if h > best.0 {
                        best = (h, t);
                    }
                }
                best.1
            }
            Self::Triangle => {
                let peak = histogram
                    .iter()
                    .enumerate()
                    .fold((0, 0u64), |b, (i, &c)| if c > b.1 { (i, c) } else { b })
                    .0;
                let lo = histogram.iter().position(|&c| c > 0).unwrap_or(0);
                let hi = histogram.iter().rposition(|&c| c > 0).unwrap_or(n - 1);
                // The longer tail.
                let end = if peak - lo > hi - peak { lo } else { hi };
                if end == peak {
                    return peak;
                }
                let (x0, y0) = (peak as f64, histogram[peak] as f64);
                let (x1, y1) = (end as f64, histogram[end] as f64);
                let (a, b) = (end.min(peak), end.max(peak));
                let mut best = (-1.0, peak);
                for (i, &c) in histogram.iter().enumerate().take(b + 1).skip(a) {
                    // Twice the area of the triangle (peak, end, point):
                    // proportional to the distance from the line.
                    let d = ((x1 - x0) * (y0 - c as f64) - (x0 - i as f64) * (y1 - y0)).abs();
                    if d > best.0 {
                        best = (d, i);
                    }
                }
                best.1
            }
        }
    }
}

/// Binarise a picture at a threshold chosen from its histogram.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AutoThreshold {
    /// How the level is chosen.
    pub method: AutoThresholdMethod,
}

impl AutoThreshold {
    /// Threshold by `method`.
    pub fn new(method: AutoThresholdMethod) -> Self {
        Self { method }
    }

    /// The threshold chosen for `img`, as a normalised intensity
    /// (`level / 255`): brighter pixels become white.
    pub fn threshold_of(&self, img: &FloatImage) -> f32 {
        let mut histogram = [0u64; 256];
        for v in intensity_plane(img) {
            if !v.is_nan() {
                histogram[(v.clamp(0.0, 1.0) * 255.0 + 0.5) as usize] += 1;
            }
        }
        self.method.level(&histogram) as f32 / 255.0
    }
}

impl PixelFilter for AutoThreshold {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let t = self.threshold_of(&img);
            let (ch, cc) = (img.channels(), img.color_channels());
            let row = img.width() as usize * ch;
            crate::exec::par_rows(img.samples_mut(), row, |_, line| {
                for px in line.chunks_exact_mut(ch) {
                    // Compare on the histogram's own grid.
                    let level =
                        (intensity(&px[..cc]).clamp(0.0, 1.0) * 255.0 + 0.5).floor() / 255.0;
                    let v = if level > t { 1.0 } else { 0.0 };
                    px[..cc].iter_mut().for_each(|s| *s = v);
                }
            });
            Ok(img)
        })
    }
}

/// CIE L\*a\*b\* (D65) of stored sRGB values and back.
pub(crate) mod lab {
    const WHITE: [f32; 3] = [0.950_47, 1.0, 1.088_83];
    const DELTA: f32 = 6.0 / 29.0;

    fn f(t: f32) -> f32 {
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    }

    fn f_inv(t: f32) -> f32 {
        if t > DELTA {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
        }
    }

    /// Stored sRGB to `[L, a, b]` (`L` in 0 … 100).
    pub(crate) fn from_srgb(rgb: [f32; 3]) -> [f32; 3] {
        let [r, g, b] = rgb.map(crate::deep::srgb_decode);
        let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
        let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
        let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;
        let (fx, fy, fz) = (f(x / WHITE[0]), f(y / WHITE[1]), f(z / WHITE[2]));
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
    }

    /// `[L, a, b]` to stored sRGB (not clamped).
    pub(crate) fn to_srgb(lab: [f32; 3]) -> [f32; 3] {
        let fy = (lab[0] + 16.0) / 116.0;
        let (fx, fz) = (fy + lab[1] / 500.0, fy - lab[2] / 200.0);
        let (x, y, z) = (
            WHITE[0] * f_inv(fx),
            WHITE[1] * f_inv(fy),
            WHITE[2] * f_inv(fz),
        );
        [
            3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
            -0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z,
            0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
        ]
        .map(crate::deep::srgb_encode)
    }
}

/// Remove a colour cast under the grey-world assumption (see the
/// [module docs](crate::tone_auto)). A grey picture is returned as is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct WhiteBalance {}

impl WhiteBalance {
    /// The operator.
    pub fn new() -> Self {
        Self {}
    }
}

impl PixelFilter for WhiteBalance {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let ch = img.channels();
            if img.color_channels() < 3 {
                return Ok(img);
            }
            let mut labs: Vec<[f32; 3]> = img
                .samples()
                .chunks_exact(ch)
                .map(|px| {
                    lab::from_srgb(
                        [px[0], px[1], px[2]].map(|v| if v.is_finite() { v } else { 0.0 }),
                    )
                })
                .collect();
            let n = labs.len().max(1) as f64;
            let (ma, mb) = labs.iter().fold((0.0f64, 0.0f64), |m, l| {
                (m.0 + l[1] as f64, m.1 + l[2] as f64)
            });
            let (ma, mb) = ((ma / n) as f32, (mb / n) as f32);
            for l in &mut labs {
                l[1] -= ma;
                l[2] -= mb;
            }
            for (px, l) in img.samples_mut().chunks_exact_mut(ch).zip(&labs) {
                px[..3].copy_from_slice(&lab::to_srgb(*l));
            }
            Ok(img)
        })
    }
}

/// Add back `strength` times the lightness detail a Gaussian of
/// `radius` removes (see the [module docs](crate::tone_auto)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalContrast {
    /// Gaussian radius in pixels (σ = radius / 2).
    pub radius: f32,
    /// Fraction of the detail added (`0.125` = 12.5 %).
    pub strength: f32,
}

impl LocalContrast {
    /// Local contrast at `radius` pixels, `strength` as a fraction.
    pub fn new(radius: f32, strength: f32) -> Self {
        Self { radius, strength }
    }
}

impl Default for LocalContrast {
    fn default() -> Self {
        Self::new(10.0, 0.125)
    }
}

/// A normalised Gaussian of standard deviation `sigma`, cut at 3σ.
pub(crate) fn gaussian(sigma: f32) -> Vec<f32> {
    let sigma = if sigma.is_finite() {
        sigma.clamp(0.0, 1024.0)
    } else {
        0.0
    };
    if sigma < 1e-3 {
        return vec![1.0];
    }
    let r = (3.0 * sigma).ceil() as i64;
    let mut k: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Blur a one-channel plane (replicated edges).
pub(crate) fn blur_plane(plane: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    let k = gaussian(sigma);
    let img = FloatImage::from_samples(w, h, 1, plane.to_vec()).expect("plane matches extent");
    deep::convolve_separable(&img, &k, &k, false).into_samples()
}

impl PixelFilter for LocalContrast {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let light = intensity_plane(&img);
            let wide = blur_plane(
                &light,
                img.width(),
                img.height(),
                self.radius.max(0.0) / 2.0,
            );
            let strength = if self.strength.is_finite() {
                self.strength
            } else {
                0.0
            };
            let target: Vec<f32> = light
                .iter()
                .zip(&wide)
                .map(|(l, b)| (l + strength * (l - b)).max(0.0))
                .collect();
            set_intensity(&mut img, &target);
            Ok(img)
        })
    }
}

/// Contrast-limited adaptive histogram equalisation (see the
/// [module docs](crate::tone_auto)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clahe {
    /// Tiles across.
    pub tiles_x: u32,
    /// Tiles down.
    pub tiles_y: u32,
    /// Histogram bins per tile (2 … 65 536).
    pub bins: u32,
    /// Clip limit as a multiple of the mean bin height (`≥ 1`).
    pub clip_limit: f32,
}

impl Clahe {
    /// CLAHE over `tiles_x × tiles_y` tiles.
    pub fn new(tiles_x: u32, tiles_y: u32, bins: u32, clip_limit: f32) -> Self {
        Self {
            tiles_x: tiles_x.max(1),
            tiles_y: tiles_y.max(1),
            bins: bins.clamp(2, 65_536),
            clip_limit,
        }
    }

    /// The tone map of one tile: `bins` output levels in `[0, 1]`.
    fn tile_map(&self, values: impl Iterator<Item = f32>) -> Vec<f32> {
        let bins = self.bins.clamp(2, 65_536) as usize;
        let mut hist = vec![0.0f64; bins];
        let mut n = 0.0f64;
        for v in values {
            hist[bin_of(v, bins)] += 1.0;
            n += 1.0;
        }
        if n == 0.0 {
            return (0..bins).map(|i| i as f32 / (bins - 1) as f32).collect();
        }
        let limit = if self.clip_limit.is_finite() {
            self.clip_limit.max(1.0) as f64
        } else {
            f64::INFINITY
        } * n
            / bins as f64;
        // Clip, and spread the excess evenly; repeat while the spread
        // pushes bins back over the limit.
        for _ in 0..32 {
            let excess: f64 = hist.iter().map(|&c| (c - limit).max(0.0)).sum();
            if excess < 1e-9 {
                break;
            }
            let share = excess / bins as f64;
            hist.iter_mut().for_each(|c| *c = c.min(limit) + share);
        }
        // The cumulative histogram at bin centres.
        let mut acc = 0.0;
        hist.iter()
            .map(|&c| {
                let mid = acc + c / 2.0;
                acc += c;
                (mid / n) as f32
            })
            .collect()
    }
}

fn bin_of(v: f32, bins: usize) -> usize {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    ((v * bins as f32) as usize).min(bins - 1)
}

impl Default for Clahe {
    fn default() -> Self {
        Self::new(8, 8, 128, 3.0)
    }
}

impl PixelFilter for Clahe {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let (w, h) = (img.width() as usize, img.height() as usize);
            if w == 0 || h == 0 {
                return Ok(img);
            }
            let light = intensity_plane(&img);
            let (tx, ty) = (
                (self.tiles_x as usize).clamp(1, w),
                (self.tiles_y as usize).clamp(1, h),
            );
            let bins = self.bins.clamp(2, 65_536) as usize;
            // Tile `i` spans [i·w/tx, (i+1)·w/tx).
            let edge = |i: usize, n: usize, len: usize| i * len / n;
            let mut maps = Vec::with_capacity(tx * ty);
            for j in 0..ty {
                for i in 0..tx {
                    let (x0, x1) = (edge(i, tx, w), edge(i + 1, tx, w));
                    let (y0, y1) = (edge(j, ty, h), edge(j + 1, ty, h));
                    maps.push(self.tile_map(
                        (y0..y1).flat_map(|y| light[y * w + x0..y * w + x1].iter().copied()),
                    ));
                }
            }
            // Position of a pixel centre in tile-centre units.
            let place = |p: usize, len: usize, n: usize| -> (usize, usize, f32) {
                let t = ((p as f32 + 0.5) * n as f32 / len as f32 - 0.5).clamp(0.0, (n - 1) as f32);
                let i = (t.floor() as usize).min(n - 1);
                (i, (i + 1).min(n - 1), t - i as f32)
            };
            let mut target = vec![0.0f32; w * h];
            crate::exec::par_rows(&mut target, w, |y, row| {
                let (j0, j1, fy) = place(y, h, ty);
                for (x, out) in row.iter_mut().enumerate() {
                    let (i0, i1, fx) = place(x, w, tx);
                    let b = bin_of(light[y * w + x], bins);
                    let at = |i: usize, j: usize| maps[j * tx + i][b];
                    let top = at(i0, j0) + (at(i1, j0) - at(i0, j0)) * fx;
                    let bottom = at(i0, j1) + (at(i1, j1) - at(i0, j1)) * fx;
                    *out = top + (bottom - top) * fy;
                }
            });
            set_intensity(&mut img, &target);
            Ok(img)
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

    #[test]
    fn thresholds_split_a_two_mode_histogram_between_the_modes() {
        // Two narrow modes at levels 60 and 190.
        let mut h = [0u64; 256];
        for (centre, weight) in [(60usize, 300u64), (190, 500)] {
            for d in 0..5usize {
                h[centre - 2 + d] += weight * [1, 4, 6, 4, 1][d];
            }
        }
        let t = AutoThresholdMethod::Otsu.level(&h);
        assert!((62..188).contains(&t), "Otsu: {t}");
        // The entropy criterion may take the first bin of the upper mode
        // into the lower class; it never cuts through either mode's bulk.
        let t = AutoThresholdMethod::Kapur.level(&h);
        assert!((62..190).contains(&t), "Kapur: {t}");
        // Otsu on two equal spikes: anywhere between; the first maximum.
        let mut h = [0u64; 256];
        h[10] = 5;
        h[200] = 5;
        assert_eq!(AutoThresholdMethod::Otsu.level(&h), 10);
        // Triangle: a peak with a long linear tail to the right; the
        // farthest point under the peak-to-end line is the tail's knee.
        let mut h = [0u64; 256];
        h[20] = 1000;
        for (i, slot) in h.iter_mut().enumerate().skip(21).take(10) {
            *slot = 1000 - 95 * (i as u64 - 20);
        }
        for slot in h.iter_mut().skip(31).take(200) {
            *slot = 50;
        }
        let t = AutoThresholdMethod::Triangle.level(&h);
        assert!((29..=33).contains(&t), "triangle knee at {t}");
        assert_eq!(AutoThresholdMethod::Otsu.level(&[0; 256]), 0);
    }

    #[test]
    fn auto_threshold_binarises() {
        let img = gray(32, 32, |x, _| if x < 12 { 0.2 } else { 0.7 });
        let out = run(&AutoThreshold::default(), &img);
        for (o, i) in out.samples().iter().zip(img.samples()) {
            assert_eq!(*o, if *i > 0.5 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn lab_round_trips_and_white_balance_neutralises_a_cast() {
        for rgb in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.5, 0.9],
            [0.9, 0.1, 0.3],
        ] {
            let back = lab::to_srgb(lab::from_srgb(rgb));
            for (b, r) in back.iter().zip(&rgb) {
                assert!((b - r).abs() < 2e-4, "{rgb:?} -> {back:?}");
            }
        }
        let grey = lab::from_srgb([0.5, 0.5, 0.5]);
        assert!(grey[1].abs() < 0.01 && grey[2].abs() < 0.01);
        // A grey ramp with a warm cast: balanced, every pixel is grey.
        let mut data = Vec::new();
        for i in 0..64 {
            let v = 0.2 + 0.6 * i as f32 / 63.0;
            let cast = lab::from_srgb([v, v, v]);
            data.extend(lab::to_srgb([cast[0], cast[1] + 12.0, cast[2] + 20.0]));
        }
        let img = FloatImage::from_samples(8, 8, 3, data).unwrap();
        let out = run(&WhiteBalance::new(), &img);
        for px in out.samples().as_chunks::<3>().0 {
            assert!(
                (px[0] - px[1]).abs() < 2e-3 && (px[1] - px[2]).abs() < 2e-3,
                "{px:?}"
            );
        }
    }

    #[test]
    fn local_contrast_keeps_flat_areas_and_steepens_edges() {
        let flat = gray(32, 32, |_, _| 0.4);
        let out = run(&LocalContrast::new(8.0, 0.5), &flat);
        assert!(out.samples().iter().all(|v| (v - 0.4).abs() < 1e-5));
        let edge = gray(64, 8, |x, _| if x < 32 { 0.3 } else { 0.6 });
        let out = run(&LocalContrast::new(8.0, 0.5), &edge);
        assert!(out.pixel(31, 4)[0] < 0.3 - 0.02);
        assert!(out.pixel(32, 4)[0] > 0.6 + 0.02);
        assert!((out.pixel(2, 4)[0] - 0.3).abs() < 1e-3);
    }

    #[test]
    fn clahe_spreads_a_narrow_range_and_respects_its_limit() {
        // A low-contrast ramp: unclipped equalisation spreads it over the
        // whole range; clip limit 1 leaves it where it was.
        let img = gray(64, 64, |x, y| 0.4 + 0.1 * ((x + y) as f32 / 126.0));
        let wide = run(&Clahe::new(1, 1, 256, f32::INFINITY), &img);
        let (lo, hi) = wide
            .samples()
            .iter()
            .fold((1.0f32, 0.0f32), |m, &v| (m.0.min(v), m.1.max(v)));
        assert!(lo < 0.05 && hi > 0.95, "{lo} … {hi}");
        let same = run(&Clahe::new(4, 4, 256, 1.0), &img);
        for (o, i) in same.samples().iter().zip(img.samples()) {
            assert!((o - i).abs() < 0.01, "{o} vs {i}");
        }
        // Monotone within a tile: order of two pixels is kept.
        let limited = run(&Clahe::new(2, 2, 64, 2.5), &img);
        assert!(limited.pixel(60, 60)[0] >= limited.pixel(40, 40)[0]);
        // Tiles more numerous than pixels, and an empty-range picture.
        let tiny = gray(3, 2, |x, _| x as f32 / 2.0);
        run(&Clahe::new(64, 64, 16, 2.0), &tiny);
    }
}
