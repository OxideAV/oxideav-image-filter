//! Image statistics, comparison metrics, moments, perceptual hash,
//! histogram and bounds — the measurement API behind
//! `getImageChannelStatistics`, `compareImages`, `getImageChannelMoments`,
//! `getImagePerceptualHash`, `getImageHistogram` and
//! `getImageBoundingBox`-style queries (issue #13).
//!
//! Every function takes a picture in any layout the crate's float hop
//! reaches (8-bit packed natively, the float trio natively, everything
//! else through `oxideav-pixfmt`) and measures it in normalised units:
//! `0.0` is code 0 and `1.0` the full scale of the input's depth, so a
//! mean of `0.5714` on an 8-bit picture is code `145.7`.
//!
//! The definitions were fixed black-box against `magick` 7.1.2
//! (`identify -verbose`, `compare -metric`, `identify -moments`):
//!
//! - standard deviation, skewness and kurtosis use the sample standard
//!   deviation (`n − 1`); kurtosis is the excess kurtosis;
//! - entropy is `−Σ p log₂ p` over the code histogram, normalised by
//!   `log₂` of the number of **occupied** bins;
//! - `compare -metric NCC` reports `(1 − NCC) / 2` with the correlation
//!   taken jointly over every channel; `SSIM` and `DSSIM` both report
//!   `(1 − SSIM) / 2` with the 11 × 11 Gaussian window (σ 1.5) and
//!   `K₁ = 0.01`, `K₂ = 0.03`; `PSNR` is `10 log₁₀(1 / MSE)`;
//! - the Hu invariants are the classic seven plus Flusser's eighth on
//!   the normalised central moments of the channel intensity; the
//!   ellipse semi-axes are `2 √λ` of the covariance eigenvalues, the
//!   angle in `[0°, 180°)`, the intensity `M₀₀ / (π a b)`.
//!
//! [`CompareMetric::Fuzz`] and [`perceptual_hash`] follow the documented
//! recipes (fuzz-thresholded RMSE; `−log₁₀` of seven Hu invariants per
//! channel in the xyY and HSB colour spaces) but do not reproduce
//! `magick`'s figures exactly — its own colour-space conversions are not
//! staged — and are documented as such.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{ConvertContext, ConvertOptions, FrameInfo, convert_with, supports};

use crate::deep::{self, FloatImage};
use crate::{VideoStreamParams, layout, side_channels};

/// The picture as a [`FloatImage`]: 8-bit packed widened, float packed
/// read as-is, every other layout through `oxideav-pixfmt` (grey without
/// alpha to `GrayF32Le`, everything else to `RgbaF32Le`).
pub fn float_image_of(frame: &VideoFrame, params: VideoStreamParams) -> Result<FloatImage, Error> {
    layout::validate_frame(frame, params, "measure")?;
    let pixels = side_channels::image_only(frame);
    match params.format {
        PixelFormat::Gray8 | PixelFormat::Rgb24 | PixelFormat::Rgba => {
            deep::widen_u8(&pixels, params)
        }
        f if deep::is_float(f) => FloatImage::from_frame(&pixels, params),
        f => {
            let via = if matches!(
                f,
                PixelFormat::Gray10Le
                    | PixelFormat::Gray12Le
                    | PixelFormat::Gray16Le
                    | PixelFormat::MonoBlack
                    | PixelFormat::MonoWhite
            ) {
                PixelFormat::GrayF32Le
            } else {
                PixelFormat::RgbaF32Le
            };
            if !supports(f, via) {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: measure cannot read {f:?}"
                )));
            }
            let ctx = ConvertContext::new().with_signal(frame.color_signal().unwrap_or_default());
            let info = FrameInfo::new(f, params.width, params.height);
            let hopped = convert_with(frame, info, via, &ConvertOptions::default(), &ctx)?;
            let hopped = side_channels::image_only(&hopped);
            FloatImage::from_frame(
                &hopped,
                VideoStreamParams {
                    format: via,
                    ..params
                },
            )
        }
    }
}

/// Histogram bins a picture's codes are counted in: 256 for 8-bit
/// storage, 65 536 otherwise.
fn bins_for(fmt: PixelFormat) -> usize {
    match layout::sample_kind(fmt) {
        Some(layout::SampleKind::U8) | None => 256,
        _ => 65536,
    }
}

/// Statistics of one channel (or of all colour channels pooled), in
/// normalised units.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelStatistics {
    /// Smallest sample.
    pub min: f64,
    /// Largest sample.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// The middle sample of the sorted population (`n / 2`).
    pub median: f64,
    /// Sample standard deviation (`n − 1`).
    pub std_dev: f64,
    /// Excess kurtosis `m₄ / s⁴ − 3`.
    pub kurtosis: f64,
    /// Skewness `m₃ / s³`.
    pub skewness: f64,
    /// Normalised Shannon entropy of the code histogram.
    pub entropy: f64,
}

impl ChannelStatistics {
    fn of(values: &mut [f64], bins: usize) -> Self {
        let n = values.len() as f64;
        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                kurtosis: 0.0,
                skewness: 0.0,
                entropy: 0.0,
            };
        }
        let mean = values.iter().sum::<f64>() / n;
        let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
        let mut hist = vec![0u64; bins];
        let full = (bins - 1) as f64;
        for &v in values.iter() {
            let d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
            let bin = if v.is_nan() {
                0
            } else {
                ((v.clamp(0.0, 1.0) * full).round() as usize).min(bins - 1)
            };
            hist[bin] += 1;
        }
        let var_s = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        let std_dev = var_s.sqrt();
        let (skewness, kurtosis) = if std_dev > 0.0 {
            (
                m3 / n / (std_dev * std_dev * std_dev),
                m4 / n / (var_s * var_s) - 3.0,
            )
        } else {
            (0.0, 0.0)
        };
        let occupied = hist.iter().filter(|&&h| h > 0).count();
        let entropy = if occupied > 1 {
            -hist
                .iter()
                .filter(|&&h| h > 0)
                .map(|&h| {
                    let p = h as f64 / n;
                    p * p.log2()
                })
                .sum::<f64>()
                / (occupied as f64).log2()
        } else {
            0.0
        };
        values.sort_by(f64::total_cmp);
        Self {
            min: values[0],
            max: values[values.len() - 1],
            mean,
            median: values[values.len() / 2],
            std_dev,
            kurtosis,
            skewness,
            entropy,
        }
    }
}

/// Per-channel and pooled statistics of a picture.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ImageStatistics {
    /// Pixels measured.
    pub pixels: u64,
    /// One entry per channel of the picture (alpha last when present).
    pub channels: Vec<ChannelStatistics>,
    /// `identify -verbose`'s "Overall": the colour channels' figures
    /// averaged (minimum and maximum as the extremes), alpha excluded.
    pub overall: ChannelStatistics,
}

/// `getImageChannelStatistics`: statistics per channel and pooled.
pub fn statistics(frame: &VideoFrame, params: VideoStreamParams) -> Result<ImageStatistics, Error> {
    let img = float_image_of(frame, params)?;
    let bins = bins_for(params.format);
    let ch = img.channels();
    let cc = img.color_channels();
    let mut channels = Vec::with_capacity(ch);
    for c in 0..ch {
        let mut v: Vec<f64> = deep::channel_iter(&img, c).map(|x| x as f64).collect();
        channels.push(ChannelStatistics::of(&mut v, bins));
    }
    // `identify`'s "Overall": the colour channels' figures averaged
    // (min / max as the extremes).
    let colour = &channels[..cc];
    let avg = |f: fn(&ChannelStatistics) -> f64| colour.iter().map(f).sum::<f64>() / cc as f64;
    let overall = ChannelStatistics {
        min: colour.iter().map(|c| c.min).fold(f64::INFINITY, f64::min),
        max: colour
            .iter()
            .map(|c| c.max)
            .fold(f64::NEG_INFINITY, f64::max),
        mean: avg(|c| c.mean),
        median: avg(|c| c.median),
        std_dev: avg(|c| c.std_dev),
        kurtosis: avg(|c| c.kurtosis),
        skewness: avg(|c| c.skewness),
        entropy: avg(|c| c.entropy),
    };
    Ok(ImageStatistics {
        pixels: img.width() as u64 * img.height() as u64,
        channels,
        overall,
    })
}

/// `getImageTotalInkDensity`: the largest per-pixel ink sum. On a CMYK
/// picture that is `C + M + Y + K` (normalised, so up to `4.0`); on an
/// RGB / grey picture each channel's ink is `1 − v`.
pub fn total_ink_density(frame: &VideoFrame, params: VideoStreamParams) -> Result<f64, Error> {
    if matches!(params.format, PixelFormat::Cmyk | PixelFormat::CmykInverted) {
        layout::validate_frame(frame, params, "total_ink_density")?;
        let p = &frame.planes[0];
        let w = params.width as usize;
        let inverted = params.format == PixelFormat::CmykInverted;
        let mut best = 0.0f64;
        for y in 0..params.height as usize {
            for px in p.data[y * p.stride..y * p.stride + 4 * w]
                .as_chunks::<4>()
                .0
            {
                let sum: f64 = px
                    .iter()
                    .map(|&v| {
                        let v = v as f64 / 255.0;
                        if inverted { 1.0 - v } else { v }
                    })
                    .sum();
                best = best.max(sum);
            }
        }
        return Ok(best);
    }
    let img = float_image_of(frame, params)?;
    let cc = img.color_channels();
    Ok(img
        .samples()
        .chunks_exact(img.channels())
        .map(|px| px[..cc].iter().map(|&v| 1.0 - v as f64).sum::<f64>())
        .fold(0.0, f64::max))
}

/// A comparison metric (`compare -metric`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareMetric {
    /// Count of pixels whose channels differ by more than the fuzz.
    AbsoluteError,
    /// Mean absolute difference.
    MeanAbsoluteError,
    /// Mean error per pixel: the mean absolute difference, with the
    /// normalised peak difference in [`Comparison::peak`].
    MeanErrorPerPixel,
    /// Mean squared difference.
    MeanSquaredError,
    /// `(1 − NCC) / 2`, the correlation taken over every channel jointly.
    NormalizedCrossCorrelation,
    /// Largest absolute difference.
    PeakAbsoluteError,
    /// `10 log₁₀(1 / MSE)` in dB (infinite for identical pictures).
    PeakSignalToNoiseRatio,
    /// Square root of the mean squared difference.
    RootMeanSquaredError,
    /// `(1 − SSIM) / 2`, the 11 × 11 Gaussian window of σ 1.5.
    StructuralSimilarity,
    /// Same figure as [`Self::StructuralSimilarity`] (what `magick`
    /// prints for both).
    StructuralDissimilarity,
    /// Root mean squared difference after the fuzz is subtracted from
    /// every channel difference (approximate: `magick`'s own figure is
    /// not reproduced).
    Fuzz,
    /// Sum of squared differences of the two perceptual hashes (see
    /// [`perceptual_hash`]).
    PerceptualHash,
}

impl CompareMetric {
    /// Every metric.
    pub const ALL: &'static [CompareMetric] = &[
        Self::AbsoluteError,
        Self::MeanAbsoluteError,
        Self::MeanErrorPerPixel,
        Self::MeanSquaredError,
        Self::NormalizedCrossCorrelation,
        Self::PeakAbsoluteError,
        Self::PeakSignalToNoiseRatio,
        Self::RootMeanSquaredError,
        Self::StructuralSimilarity,
        Self::StructuralDissimilarity,
        Self::Fuzz,
        Self::PerceptualHash,
    ];

    /// The `compare -metric` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::AbsoluteError => "AE",
            Self::MeanAbsoluteError => "MAE",
            Self::MeanErrorPerPixel => "MEPP",
            Self::MeanSquaredError => "MSE",
            Self::NormalizedCrossCorrelation => "NCC",
            Self::PeakAbsoluteError => "PAE",
            Self::PeakSignalToNoiseRatio => "PSNR",
            Self::RootMeanSquaredError => "RMSE",
            Self::StructuralSimilarity => "SSIM",
            Self::StructuralDissimilarity => "DSSIM",
            Self::Fuzz => "Fuzz",
            Self::PerceptualHash => "PHASH",
        }
    }

    /// Parse a `compare -metric` name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|m| m.name().eq_ignore_ascii_case(name))
    }
}

/// The result of [`compare`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    /// The metric measured.
    pub metric: CompareMetric,
    /// The figure over every colour channel (what `compare` prints in
    /// parentheses).
    pub value: f64,
    /// The figure per colour channel (empty for the joint metrics NCC,
    /// SSIM, DSSIM, PHASH).
    pub channels: Vec<f64>,
    /// The normalised peak absolute difference (the third number of
    /// `MEPP`; filled for every metric).
    pub peak: f64,
}

fn same_shape(a: &FloatImage, b: &FloatImage) -> Result<(), Error> {
    if a.width() != b.width() || a.height() != b.height() || a.channels() != b.channels() {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: compare needs two pictures of one shape, got {}x{}x{} and {}x{}x{}",
            a.width(),
            a.height(),
            a.channels(),
            b.width(),
            b.height(),
            b.channels()
        )));
    }
    Ok(())
}

/// `compareImages` / `getImageDistortion`: measure `metric` between two
/// pictures of one layout and extent. `fuzz` is the normalised colour
/// distance below which a difference does not count (AE, Fuzz).
pub fn compare(
    a: &VideoFrame,
    b: &VideoFrame,
    params: VideoStreamParams,
    metric: CompareMetric,
    fuzz: f64,
) -> Result<Comparison, Error> {
    let ia = float_image_of(a, params)?;
    let ib = float_image_of(b, params)?;
    compare_float(&ia, &ib, metric, fuzz)
}

/// [`compare`] on float pictures.
pub fn compare_float(
    a: &FloatImage,
    b: &FloatImage,
    metric: CompareMetric,
    fuzz: f64,
) -> Result<Comparison, Error> {
    same_shape(a, b)?;
    let ch = a.channels();
    let cc = a.color_channels();
    let n = (a.width() as usize * a.height() as usize) as f64;
    let fuzz = if fuzz.is_finite() { fuzz.max(0.0) } else { 0.0 };
    let mut peak = 0.0f64;
    let mut per_abs = vec![0.0f64; cc];
    let mut per_sq = vec![0.0f64; cc];
    let mut per_peak = vec![0.0f64; cc];
    let mut ae = 0u64;
    let mut fuzz_sq = 0.0f64;
    for (pa, pb) in a
        .samples()
        .chunks_exact(ch)
        .zip(b.samples().chunks_exact(ch))
    {
        let mut differs = false;
        for c in 0..cc {
            let d = (pa[c] as f64 - pb[c] as f64).abs();
            per_abs[c] += d;
            per_sq[c] += d * d;
            per_peak[c] = per_peak[c].max(d);
            peak = peak.max(d);
            if d > fuzz {
                differs = true;
            }
            let f = (d - fuzz).max(0.0);
            fuzz_sq += f * f;
        }
        if differs {
            ae += 1;
        }
    }
    let mae: Vec<f64> = per_abs.iter().map(|s| s / n).collect();
    let mse: Vec<f64> = per_sq.iter().map(|s| s / n).collect();
    let all_mae = mae.iter().sum::<f64>() / cc as f64;
    let all_mse = mse.iter().sum::<f64>() / cc as f64;
    let (value, channels) = match metric {
        CompareMetric::AbsoluteError => (ae as f64, Vec::new()),
        CompareMetric::MeanAbsoluteError | CompareMetric::MeanErrorPerPixel => (all_mae, mae),
        CompareMetric::MeanSquaredError => (all_mse, mse),
        CompareMetric::RootMeanSquaredError => {
            (all_mse.sqrt(), mse.iter().map(|v| v.sqrt()).collect())
        }
        CompareMetric::PeakAbsoluteError => (peak, per_peak),
        CompareMetric::PeakSignalToNoiseRatio => {
            let psnr = |m: f64| {
                if m > 0.0 {
                    10.0 * (1.0 / m).log10()
                } else {
                    f64::INFINITY
                }
            };
            (psnr(all_mse), mse.iter().map(|&m| psnr(m)).collect())
        }
        CompareMetric::Fuzz => ((fuzz_sq / (n * cc as f64)).sqrt(), Vec::new()),
        CompareMetric::NormalizedCrossCorrelation => {
            let mean = |img: &FloatImage, c: usize| {
                deep::channel_iter(img, c).map(|v| v as f64).sum::<f64>() / n
            };
            let (mut num, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
            for c in 0..cc {
                let (ma, mb) = (mean(a, c), mean(b, c));
                for (x, y) in deep::channel_iter(a, c).zip(deep::channel_iter(b, c)) {
                    let (x, y) = (x as f64 - ma, y as f64 - mb);
                    num += x * y;
                    da += x * x;
                    db += y * y;
                }
            }
            let ncc = if da > 0.0 && db > 0.0 {
                num / (da * db).sqrt()
            } else if da == 0.0 && db == 0.0 {
                1.0
            } else {
                0.0
            };
            ((1.0 - ncc) / 2.0, Vec::new())
        }
        CompareMetric::StructuralSimilarity | CompareMetric::StructuralDissimilarity => {
            ((1.0 - ssim(a, b)) / 2.0, Vec::new())
        }
        CompareMetric::PerceptualHash => {
            let (ha, hb) = (perceptual_hash_float(a), perceptual_hash_float(b));
            (perceptual_hash_distance(&ha, &hb), Vec::new())
        }
    };
    Ok(Comparison {
        metric,
        value,
        channels,
        peak,
    })
}

/// Mean SSIM over the colour channels: 11 × 11 Gaussian window (σ 1.5),
/// `K₁ = 0.01`, `K₂ = 0.03`, replicate edges.
fn ssim(a: &FloatImage, b: &FloatImage) -> f64 {
    const R: usize = 5;
    let kernel: Vec<f32> = {
        let k: Vec<f32> = (0..=2 * R)
            .map(|i| {
                let d = i as f32 - R as f32;
                (-d * d / (2.0 * 1.5 * 1.5)).exp()
            })
            .collect();
        let s: f32 = k.iter().sum();
        k.iter().map(|v| v / s).collect()
    };
    let cc = a.color_channels();
    let (w, h) = (a.width() as usize, a.height() as usize);
    let blur = |img: &FloatImage| deep::convolve_separable(img, &kernel, &kernel, true);
    let mul = |x: &FloatImage, y: &FloatImage| {
        let data = x
            .samples()
            .iter()
            .zip(y.samples())
            .map(|(p, q)| p * q)
            .collect();
        FloatImage::from_samples(x.width(), x.height(), x.channels(), data).expect("same shape")
    };
    let (mx, my) = (blur(a), blur(b));
    let (xx, yy, xy) = (blur(&mul(a, a)), blur(&mul(b, b)), blur(&mul(a, b)));
    let (c1, c2) = (0.01f64 * 0.01, 0.03f64 * 0.03);
    let mut total = 0.0f64;
    let ch = a.channels();
    for i in 0..w * h {
        for c in 0..cc {
            let k = i * ch + c;
            let (ux, uy) = (mx.samples()[k] as f64, my.samples()[k] as f64);
            let vx = xx.samples()[k] as f64 - ux * ux;
            let vy = yy.samples()[k] as f64 - uy * uy;
            let cov = xy.samples()[k] as f64 - ux * uy;
            total += ((2.0 * ux * uy + c1) * (2.0 * cov + c2))
                / ((ux * ux + uy * uy + c1) * (vx + vy + c2));
        }
    }
    total / (w * h * cc) as f64
}

/// `compareImages`'s difference picture: `highlight` where a pixel
/// differs beyond `fuzz` on any colour channel, `lowlight` elsewhere.
/// Output is `Rgba`.
pub fn difference_image(
    a: &VideoFrame,
    b: &VideoFrame,
    params: VideoStreamParams,
    fuzz: f64,
    highlight: [u8; 4],
    lowlight: [u8; 4],
) -> Result<VideoFrame, Error> {
    let ia = float_image_of(a, params)?;
    let ib = float_image_of(b, params)?;
    same_shape(&ia, &ib)?;
    let ch = ia.channels();
    let cc = ia.color_channels();
    let w = ia.width() as usize;
    let mut data = Vec::with_capacity(w * ia.height() as usize * 4);
    for (pa, pb) in ia
        .samples()
        .chunks_exact(ch)
        .zip(ib.samples().chunks_exact(ch))
    {
        let differs = (0..cc).any(|c| (pa[c] as f64 - pb[c] as f64).abs() > fuzz);
        data.extend_from_slice(if differs { &highlight } else { &lowlight });
    }
    Ok(VideoFrame {
        pts: a.pts,
        planes: vec![VideoPlane {
            stride: w * 4,
            data,
        }],
    })
}

/// The best match of [`compare`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityMatch {
    /// Offset of the needle's top-left corner in the haystack.
    pub x: u32,
    /// Offset of the needle's top-left corner in the haystack.
    pub y: u32,
    /// The metric at that offset (a distortion: smaller is better).
    pub score: f64,
}

/// `similarityImage`: slide `needle` over `haystack` (both in the
/// haystack's layout) and return the offset with the smallest distortion
/// under `metric`.
pub fn similarity_search(
    haystack: &VideoFrame,
    haystack_params: VideoStreamParams,
    needle: &VideoFrame,
    needle_params: VideoStreamParams,
    metric: CompareMetric,
) -> Result<SimilarityMatch, Error> {
    let hay = float_image_of(haystack, haystack_params)?;
    let nee = float_image_of(needle, needle_params)?;
    if nee.width() > hay.width() || nee.height() > hay.height() {
        return Err(Error::invalid(
            "oxideav-image-filter: similarity_search needle is larger than the haystack",
        ));
    }
    let nee = nee.to_channels(hay.channels());
    let (nw, nh, ch) = (nee.width(), nee.height(), hay.channels());
    let mut best = SimilarityMatch {
        x: 0,
        y: 0,
        score: f64::INFINITY,
    };
    let mut window = FloatImage::new(nw, nh, ch);
    for oy in 0..=hay.height() - nh {
        for ox in 0..=hay.width() - nw {
            for y in 0..nh {
                for x in 0..nw {
                    let src = hay.pixel(ox + x, oy + y).to_vec();
                    window.pixel_mut(x, y).copy_from_slice(&src);
                }
            }
            let score = compare_float(&window, &nee, metric, 0.0)?.value;
            // PSNR is a similarity, every other metric a distortion.
            let score = if metric == CompareMetric::PeakSignalToNoiseRatio {
                -score
            } else {
                score
            };
            if score < best.score {
                best = SimilarityMatch {
                    x: ox,
                    y: oy,
                    score,
                };
            }
        }
    }
    if metric == CompareMetric::PeakSignalToNoiseRatio {
        best.score = -best.score;
    }
    Ok(best)
}

/// Moments of one channel (or of the pooled intensity).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelMoments {
    /// Intensity-weighted centroid `(x, y)` in pixels.
    pub centroid: (f64, f64),
    /// Semi-major axis of the inertia ellipse (`2 √λ₁`).
    pub semi_major: f64,
    /// Semi-minor axis (`2 √λ₂`).
    pub semi_minor: f64,
    /// Orientation of the major axis in degrees, `[0, 180)`.
    pub angle_degrees: f64,
    /// `√(1 − λ₂ / λ₁)`.
    pub eccentricity: f64,
    /// Mean intensity over the ellipse, `M₀₀ / (π a b)`, normalised.
    pub intensity: f64,
    /// Hu's seven invariants and Flusser's eighth, on the normalised
    /// central moments.
    pub hu: [f64; 8],
}

/// Raw and central moments of one intensity field: `m00`, the
/// first-order sums and the central moments `mu[p][q]` (`p + q ≤ 3`)
/// about the field's own centroid.
#[derive(Clone, Copy, Debug, Default)]
struct RawMoments {
    m00: f64,
    m10: f64,
    m01: f64,
    mu: [[f64; 4]; 4],
}

impl RawMoments {
    fn of(width: usize, height: usize, intensity: impl Fn(usize, usize) -> f64) -> Self {
        let mut r = Self::default();
        for y in 0..height {
            for x in 0..width {
                let v = intensity(x, y);
                r.m00 += v;
                r.m10 += v * x as f64;
                r.m01 += v * y as f64;
            }
        }
        if r.m00.abs() < 1e-12 {
            return r;
        }
        let (cx, cy) = (r.m10 / r.m00, r.m01 / r.m00);
        for y in 0..height {
            let dy = y as f64 - cy;
            for x in 0..width {
                let dx = x as f64 - cx;
                let v = intensity(x, y);
                let px = [1.0, dx, dx * dx, dx * dx * dx];
                let py = [1.0, dy, dy * dy, dy * dy * dy];
                for (p, &xp) in px.iter().enumerate() {
                    for (q, &yq) in py.iter().enumerate() {
                        if p + q <= 3 {
                            r.mu[p][q] += v * xp * yq;
                        }
                    }
                }
            }
        }
        r
    }

    /// Channel sums: the overall picture pools the raw sums and the
    /// per-channel central moments (each about its own centroid), which
    /// is how `identify -moments` reports "Image moments".
    fn add(&mut self, o: &Self) {
        self.m00 += o.m00;
        self.m10 += o.m10;
        self.m01 += o.m01;
        for p in 0..4 {
            for q in 0..4 {
                self.mu[p][q] += o.mu[p][q];
            }
        }
    }

    fn scale(&mut self, k: f64) {
        self.m00 *= k;
        self.m10 *= k;
        self.m01 *= k;
        for row in &mut self.mu {
            for v in row {
                *v *= k;
            }
        }
    }

    fn finish(&self) -> ChannelMoments {
        let m00 = self.m00;
        if m00.abs() < 1e-12 {
            return ChannelMoments {
                centroid: (0.0, 0.0),
                semi_major: 0.0,
                semi_minor: 0.0,
                angle_degrees: 0.0,
                eccentricity: 0.0,
                intensity: 0.0,
                hu: [0.0; 8],
            };
        }
        let mu = &self.mu;
        let (cx, cy) = (self.m10 / m00, self.m01 / m00);
        let eta = |p: usize, q: usize| mu[p][q] / m00.powf(1.0 + (p + q) as f64 / 2.0);
        let (n20, n02, n11) = (eta(2, 0), eta(0, 2), eta(1, 1));
        let (n30, n03, n21, n12) = (eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2));
        let hu = [
            n20 + n02,
            (n20 - n02).powi(2) + 4.0 * n11 * n11,
            (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2),
            (n30 + n12).powi(2) + (n21 + n03).powi(2),
            (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
                + (3.0 * n21 - n03)
                    * (n21 + n03)
                    * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2)),
            (n20 - n02) * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
                + 4.0 * n11 * (n30 + n12) * (n21 + n03),
            (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
                - (n30 - 3.0 * n12)
                    * (n21 + n03)
                    * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2)),
            n11 * ((n30 + n12).powi(2) - (n03 + n21).powi(2))
                - (n20 - n02) * (n30 + n12) * (n03 + n21),
        ];
        let (a20, a02, a11) = (mu[2][0] / m00, mu[0][2] / m00, mu[1][1] / m00);
        let tr = a20 + a02;
        let disc = (4.0 * a11 * a11 + (a20 - a02) * (a20 - a02)).sqrt();
        let l1 = ((tr + disc) / 2.0).max(0.0);
        let l2 = ((tr - disc) / 2.0).max(0.0);
        let semi_major = 2.0 * l1.sqrt();
        let semi_minor = 2.0 * l2.sqrt();
        let mut angle = (0.5 * (2.0 * a11).atan2(a20 - a02)).to_degrees();
        if angle < 0.0 {
            angle += 180.0;
        }
        let eccentricity = if l1 > 0.0 {
            (1.0 - l2 / l1).max(0.0).sqrt()
        } else {
            0.0
        };
        let intensity = if semi_major > 0.0 && semi_minor > 0.0 {
            m00 / (std::f64::consts::PI * semi_major * semi_minor)
        } else {
            0.0
        };
        ChannelMoments {
            centroid: (cx, cy),
            semi_major,
            semi_minor,
            angle_degrees: angle,
            eccentricity,
            intensity,
            hu,
        }
    }
}

fn moments_of(
    width: usize,
    height: usize,
    intensity: impl Fn(usize, usize) -> f64,
) -> ChannelMoments {
    RawMoments::of(width, height, intensity).finish()
}

/// `getImageChannelMoments`: moments per channel plus the pooled
/// intensity (the mean of the colour channels).
pub fn moments(
    frame: &VideoFrame,
    params: VideoStreamParams,
) -> Result<(Vec<ChannelMoments>, ChannelMoments), Error> {
    let img = float_image_of(frame, params)?;
    Ok(moments_float(&img))
}

/// [`moments`] on a float picture.
pub fn moments_float(img: &FloatImage) -> (Vec<ChannelMoments>, ChannelMoments) {
    let (w, h, ch, cc) = (
        img.width() as usize,
        img.height() as usize,
        img.channels(),
        img.color_channels(),
    );
    let raw: Vec<RawMoments> = (0..ch)
        .map(|c| RawMoments::of(w, h, |x, y| img.samples()[(y * w + x) * ch + c] as f64))
        .collect();
    let per: Vec<ChannelMoments> = raw.iter().map(RawMoments::finish).collect();
    // "Image moments": the per-channel raw sums and central moments
    // (each about its own centroid) averaged over the colour channels.
    let mut pooled = RawMoments::default();
    for r in &raw[..cc] {
        pooled.add(r);
    }
    pooled.scale(1.0 / cc as f64);
    (per, pooled.finish())
}

/// A perceptual hash: for each of the three channels, `−log₁₀|Iᵢ|` of
/// the first seven Hu invariants in the xyY colour space (first seven
/// entries) and in HSB (next seven).
pub type PerceptualHash = [[f64; 14]; 3];

/// `getImagePerceptualHash`. Grey pictures are read as `R = G = B`.
/// The recipe is ImageMagick's (seven Hu invariants per channel in xyY
/// and HSB, log-scaled); the colour-space conversions are the standard
/// sRGB → XYZ (D65) and HSB formulas, so the figures track `magick`'s
/// closely but not to the last digit.
pub fn perceptual_hash(
    frame: &VideoFrame,
    params: VideoStreamParams,
) -> Result<PerceptualHash, Error> {
    Ok(perceptual_hash_float(&float_image_of(frame, params)?))
}

/// [`perceptual_hash`] on a float picture.
pub fn perceptual_hash_float(img: &FloatImage) -> PerceptualHash {
    let rgba = img.to_rgba();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let xyy: Vec<[f64; 3]> = rgba
        .samples()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| to_xyy(p))
        .collect();
    let hsb: Vec<[f64; 3]> = rgba
        .samples()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| to_hsb(p))
        .collect();
    let mut hash = [[0.0f64; 14]; 3];
    for (c, row) in hash.iter_mut().enumerate() {
        let mx = moments_of(w, h, |x, y| xyy[y * w + x][c]);
        let mh = moments_of(w, h, |x, y| hsb[y * w + x][c]);
        for i in 0..7 {
            row[i] = log_hash(mx.hu[i]);
            row[7 + i] = log_hash(mh.hu[i]);
        }
    }
    hash
}

fn log_hash(v: f64) -> f64 {
    if v.abs() < 1e-300 {
        0.0
    } else {
        -v.abs().log10()
    }
}

/// Sum of squared differences of two perceptual hashes.
pub fn perceptual_hash_distance(a: &PerceptualHash, b: &PerceptualHash) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y) * (x - y)))
        .sum()
}

fn to_xyy(p: &[f32]) -> [f64; 3] {
    let lin = |v: f32| deep::srgb_decode(v) as f64;
    let (r, g, b) = (lin(p[0]), lin(p[1]), lin(p[2]));
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b;
    let s = x + y + z;
    if s > 0.0 {
        [x / s, y / s, y]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn to_hsb(p: &[f32]) -> [f64; 3] {
    let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = mx - mn;
    let s = if mx > 0.0 { d / mx } else { 0.0 };
    let h = if d <= 0.0 {
        0.0
    } else if mx == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    [h / 6.0, s, mx]
}

/// One histogram entry: a colour as its per-channel codes at the
/// picture's depth (8-bit codes for 8-bit storage, 16-bit otherwise,
/// alpha last when present) and its pixel count.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorCount {
    /// The colour's channel codes.
    pub color: Vec<u16>,
    /// Pixels of that colour.
    pub count: u64,
}

/// `getImageHistogram`: every distinct colour with its count, most
/// frequent first.
pub fn histogram(frame: &VideoFrame, params: VideoStreamParams) -> Result<Vec<ColorCount>, Error> {
    let img = float_image_of(frame, params)?;
    let full = (bins_for(params.format) - 1) as f32;
    let ch = img.channels();
    let mut counts: std::collections::HashMap<Vec<u16>, u64> = std::collections::HashMap::new();
    for px in img.samples().chunks_exact(ch) {
        let key: Vec<u16> = px
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    0
                } else {
                    (v.clamp(0.0, 1.0) * full).round() as u16
                }
            })
            .collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut out: Vec<ColorCount> = counts
        .into_iter()
        .map(|(color, count)| ColorCount { color, count })
        .collect();
    out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.color.cmp(&b.color)));
    Ok(out)
}

/// `getImageColors`: the number of distinct colours.
pub fn unique_colors(frame: &VideoFrame, params: VideoStreamParams) -> Result<usize, Error> {
    Ok(histogram(frame, params)?.len())
}

/// `getImageBoundingBox` (`-trim` without cropping): the box of pixels
/// that differ from `background` (normalised channels, alpha last; the
/// top-left pixel when `None`) by more than `fuzz` on any channel, as
/// `(x, y, width, height)`; `None` when every pixel is background.
pub fn trim_bounds(
    frame: &VideoFrame,
    params: VideoStreamParams,
    fuzz: f32,
    background: Option<&[f32]>,
) -> Result<Option<(u32, u32, u32, u32)>, Error> {
    let img = float_image_of(frame, params)?;
    let bg: Vec<f32> = match background {
        Some(b) => {
            let mut v = b.to_vec();
            v.resize(img.channels(), 1.0);
            v
        }
        None => img.pixel(0, 0).to_vec(),
    };
    Ok(crate::trim::bbox_float(&img, &bg, fuzz))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> (VideoFrame, VideoStreamParams) {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        (
            VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: w as usize * 3,
                    data,
                }],
            },
            VideoStreamParams {
                format: PixelFormat::Rgb24,
                width: w,
                height: h,
            },
        )
    }

    #[test]
    fn statistics_of_a_ramp() {
        let (f, p) = rgb(256, 1, |x, _| [x as u8, 128, 0]);
        let s = statistics(&f, p).unwrap();
        assert_eq!(s.pixels, 256);
        let r = s.channels[0];
        assert!((r.mean - 127.5 / 255.0).abs() < 1e-6);
        assert_eq!(r.min, 0.0);
        assert_eq!(r.max, 1.0);
        assert!((r.entropy - 1.0).abs() < 1e-9, "flat histogram → entropy 1");
        assert!(r.skewness.abs() < 1e-6);
        // Uniform distribution: excess kurtosis −1.2 (sample-corrected).
        assert!((r.kurtosis + 1.2).abs() < 0.02, "{}", r.kurtosis);
        let g = s.channels[1];
        assert_eq!(g.std_dev, 0.0);
        assert_eq!(g.entropy, 0.0);
        assert_eq!(s.channels[2].max, 0.0);
    }

    #[test]
    fn identical_pictures_compare_as_perfect() {
        let (f, p) = rgb(9, 7, |x, y| [(x * 20) as u8, (y * 30) as u8, 77]);
        for &m in CompareMetric::ALL {
            let c = compare(&f, &f, p, m, 0.0).unwrap();
            match m {
                CompareMetric::PeakSignalToNoiseRatio => assert!(c.value.is_infinite()),
                _ => assert!(c.value.abs() < 1e-9, "{m:?} = {}", c.value),
            }
            assert_eq!(c.peak, 0.0);
        }
    }

    #[test]
    fn error_metrics_on_a_known_difference() {
        let (a, p) = rgb(4, 1, |_, _| [100, 100, 100]);
        let (b, _) = rgb(4, 1, |x, _| [100 + (x as u8) * 10, 100, 100]);
        // Red differs by 0, 10, 20, 30 codes.
        let mae = compare(&a, &b, p, CompareMetric::MeanAbsoluteError, 0.0).unwrap();
        assert!((mae.channels[0] - 15.0 / 255.0).abs() < 1e-6);
        assert!((mae.value - 5.0 / 255.0).abs() < 1e-6);
        assert!((mae.peak - 30.0 / 255.0).abs() < 1e-6);
        let ae = compare(&a, &b, p, CompareMetric::AbsoluteError, 0.0).unwrap();
        assert_eq!(ae.value, 3.0);
        let ae = compare(&a, &b, p, CompareMetric::AbsoluteError, 15.0 / 255.0).unwrap();
        assert_eq!(ae.value, 2.0);
        let mse = compare(&a, &b, p, CompareMetric::MeanSquaredError, 0.0).unwrap();
        let expect = (100.0 + 400.0 + 900.0) / 4.0 / (255.0 * 255.0) / 3.0;
        assert!((mse.value - expect).abs() < 1e-9);
        let rmse = compare(&a, &b, p, CompareMetric::RootMeanSquaredError, 0.0).unwrap();
        assert!((rmse.value - expect.sqrt()).abs() < 1e-7);
        let psnr = compare(&a, &b, p, CompareMetric::PeakSignalToNoiseRatio, 0.0).unwrap();
        assert!((psnr.value - 10.0 * (1.0 / expect).log10()).abs() < 1e-5);
        let diff = difference_image(&a, &b, p, 0.0, [255, 0, 0, 255], [0, 0, 0, 0]).unwrap();
        assert_eq!(diff.planes[0].data[..4], [0, 0, 0, 0]);
        assert_eq!(diff.planes[0].data[4..8], [255, 0, 0, 255]);
    }

    #[test]
    fn moments_of_a_centred_disc() {
        let (f, p) = rgb(31, 31, |x, y| {
            let (dx, dy) = (x as f32 - 15.0, y as f32 - 15.0);
            if dx * dx + dy * dy <= 100.0 {
                [255, 255, 255]
            } else {
                [0, 0, 0]
            }
        });
        let (per, all) = moments(&f, p).unwrap();
        assert_eq!(per.len(), 3);
        assert!((all.centroid.0 - 15.0).abs() < 1e-9 && (all.centroid.1 - 15.0).abs() < 1e-9);
        assert!(
            (all.semi_major - all.semi_minor).abs() < 1e-6,
            "a disc is round"
        );
        assert!(all.eccentricity < 1e-3);
        // A filled disc of radius r has I1 = 1 / (2π).
        assert!(
            (all.hu[0] - 1.0 / (2.0 * std::f64::consts::PI)).abs() < 0.01,
            "{}",
            all.hu[0]
        );
        let hash = perceptual_hash(&f, p).unwrap();
        assert_eq!(perceptual_hash_distance(&hash, &hash), 0.0);
    }

    #[test]
    fn histogram_counts_and_bounds() {
        let (f, p) = rgb(4, 2, |x, _| if x < 3 { [1, 2, 3] } else { [9, 9, 9] });
        let h = histogram(&f, p).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].color, vec![1, 2, 3]);
        assert_eq!(h[0].count, 6);
        assert_eq!(unique_colors(&f, p).unwrap(), 2);
        assert_eq!(trim_bounds(&f, p, 0.0, None).unwrap(), Some((3, 0, 1, 2)));
        assert_eq!(trim_bounds(&f, p, 0.1, None).unwrap(), None);
        let (needle, np) = rgb(1, 1, |_, _| [9, 9, 9]);
        let m = similarity_search(&f, p, &needle, np, CompareMetric::RootMeanSquaredError).unwrap();
        assert_eq!((m.x, m.y), (3, 0));
        assert_eq!(m.score, 0.0);
    }
}
