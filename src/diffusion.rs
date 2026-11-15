//! One dither engine for every method (issue #15): threshold maps,
//! error diffusion through any stencil in raster or serpentine order,
//! and error diffusion along a Hilbert curve — each quantising either
//! every channel to a number of levels ([`ChannelDither`]) or whole
//! pixels to a palette ([`crate::palette::PaletteQuantize`]).
//!
//! A *quantiser* maps a pixel to the nearest value the output can
//! hold. Dithering replaces the pixel `v` by a *target* `t` before the
//! quantiser sees it:
//!
//! - **Ordered** — `t = v + d · s`, with `d ∈ (−½, ½)` the deviation of
//!   the threshold cell under the pixel (`(cell + ½) / divisor − ½`) and
//!   `s` the spacing of the output values (`1 / (L − 1)` for `L` levels;
//!   the mean distance between neighbouring palette colours for a
//!   palette). No feedback: rows are independent and run under the
//!   thread budget.
//! - **Stencil diffusion** — `t = v + e`, with `e` the error earlier
//!   pixels pushed onto this one; the residue `t − q(t)` goes to the
//!   stencil's not-yet-visited neighbours in proportion to its weights
//!   (`docs/image/filter/dithering-kernels.md` §1.1). The error is a
//!   vector over the channels, so a palette quantiser diffuses the full
//!   colour error. On a serpentine scan odd rows run right to left with
//!   the stencil mirrored (§1.3).
//! - **Hilbert diffusion** — the pixels are visited along a Hilbert
//!   curve, which never leaves a neighbourhood before it has filled it,
//!   and `t = v + Σ wₖ eₖ` over the errors of the last `n` pixels
//!   visited, `wₖ ∝ r^(−k / (n − 1))` (`k = 0` the most recent, weights
//!   summing to one): an error fades by the factor `r` over the memory
//!   and is spread over the next `n` pixels of the path — which lie all
//!   round it, not only below and to the right. A picture that is not a
//!   power-of-two square is covered by a row of such squares along its
//!   longer side, each entered where the previous one ended.
//!
//! The error carried into a pixel is limited to one full range per
//! channel, so a colour the palette cannot reach (a saturated red on a
//! grey palette) saturates instead of accumulating without bound.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::dither::{DiffusionKernel, ScanOrder};
use crate::ordered_dither::ThresholdMap;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// An error-diffusion stencil: where the quantisation error of a pixel
/// goes, and how much of it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionStencil {
    /// `(dx, dy, weight)`: the fraction `weight` of the error goes to
    /// the pixel `dx` to the right and `dy` below.
    taps: Vec<(i32, i32, f32)>,
    reach_x: usize,
    reach_y: usize,
}

impl DiffusionStencil {
    /// Largest horizontal / vertical reach of a stencil.
    pub const MAX_REACH: i32 = 16;

    /// A stencil from `(dx, dy, weight)` taps, each weight divided by
    /// `divisor`. Every tap must point at a pixel a raster scan has not
    /// visited yet (`dy > 0`, or `dy = 0` and `dx > 0`) within
    /// [`Self::MAX_REACH`]; weights must be finite. The weights need
    /// not sum to the divisor (a stencil may drop part of the error).
    pub fn new(taps: &[(i32, i32, f32)], divisor: f32) -> Result<Self, Error> {
        if taps.is_empty() || !divisor.is_finite() || divisor == 0.0 {
            return Err(Error::invalid(
                "oxideav-image-filter: DiffusionStencil needs taps and a non-zero divisor",
            ));
        }
        let mut out = Vec::with_capacity(taps.len());
        let (mut rx, mut ry) = (0, 0);
        for &(dx, dy, w) in taps {
            let ahead = dy > 0 || (dy == 0 && dx > 0);
            if !ahead || dx.abs() > Self::MAX_REACH || dy > Self::MAX_REACH || !w.is_finite() {
                return Err(Error::invalid(format!(
                    "oxideav-image-filter: DiffusionStencil tap ({dx}, {dy}, {w}) is not a \
                     finite weight on a pixel ahead of the scan within {} pixels",
                    Self::MAX_REACH
                )));
            }
            rx = rx.max(dx.unsigned_abs() as usize);
            ry = ry.max(dy as usize);
            out.push((dx, dy, w * (1.0 / divisor)));
        }
        Ok(Self {
            taps: out,
            reach_x: rx,
            reach_y: ry,
        })
    }

    /// A stencil drawn as rows: `rows[0]` is the current scanline with
    /// the current pixel at column `origin` (entries at and left of it
    /// are ignored), `rows[1]` the line below, and so on; zero entries
    /// are skipped. For instance the stencil of
    /// `docs/image/filter/dithering-kernels.md` §3 is
    /// `from_rows(&[&[0., 0., 7.], &[3., 5., 1.]], 1, 16.)`.
    pub fn from_rows(rows: &[&[f32]], origin: usize, divisor: f32) -> Result<Self, Error> {
        let mut taps = Vec::new();
        for (dy, row) in rows.iter().enumerate() {
            for (x, &w) in row.iter().enumerate() {
                let dx = x as i32 - origin as i32;
                if w != 0.0 && (dy > 0 || dx > 0) {
                    taps.push((dx, dy as i32, w));
                }
            }
        }
        Self::new(&taps, divisor)
    }

    /// The stencil of a tabulated kernel
    /// (`docs/image/filter/dithering-kernels.md` §3 – §7).
    pub fn of(kernel: DiffusionKernel) -> Self {
        let (taps, divisor) = crate::dither::kernel_taps(kernel);
        let taps: Vec<(i32, i32, f32)> =
            taps.iter().map(|&(dx, dy, w)| (dx, dy, w as f32)).collect();
        Self::new(&taps, divisor as f32).expect("tabulated stencils are valid")
    }

    /// The taps as `(dx, dy, fraction of the error)`.
    pub fn taps(&self) -> &[(i32, i32, f32)] {
        &self.taps
    }
}

/// How the quantisation error (or a threshold) is spread.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum DitherMethod {
    /// No dither: every pixel goes to its nearest output value.
    None,
    /// Ordered dither against a threshold map (Bayer, clustered dot,
    /// blue noise, or a supplied one).
    Ordered(ThresholdMap),
    /// Error diffusion through a stencil.
    Diffusion {
        /// Where the error goes.
        stencil: DiffusionStencil,
        /// Row order.
        scan: ScanOrder,
    },
    /// Error diffusion along a Hilbert curve.
    Hilbert {
        /// How many past errors are remembered (2 … 256).
        history: u32,
        /// How much less the oldest remembered error weighs than the
        /// newest (`≥ 1`).
        ratio: f32,
    },
}

impl DitherMethod {
    /// Diffusion through a tabulated kernel on a serpentine scan.
    pub fn diffusion(kernel: DiffusionKernel) -> Self {
        Self::Diffusion {
            stencil: DiffusionStencil::of(kernel),
            scan: ScanOrder::Serpentine,
        }
    }

    /// The four-neighbour stencil of
    /// `docs/image/filter/dithering-kernels.md` §3 on a serpentine scan.
    pub fn floyd_steinberg() -> Self {
        Self::diffusion(DiffusionKernel::FloydSteinberg)
    }

    /// Hilbert-curve diffusion with a memory of 16 errors fading by a
    /// factor of 16.
    pub fn hilbert() -> Self {
        Self::Hilbert {
            history: 16,
            ratio: 16.0,
        }
    }

    /// Ordered dither against the 64 × 64 blue-noise mask.
    pub fn blue_noise() -> Self {
        Self::Ordered(ThresholdMap::blue_noise(64))
    }

    /// A method by name: `none`, `floyd-steinberg` / `fs`, `jjn` /
    /// `jarvis`, `stucki`, `sierra3`, `sierra2`, `sierra-lite`,
    /// `atkinson` (serpentine), `hilbert` / `riemersma`, or any
    /// [`ThresholdMap::named`] map (`blue-noise`, `o8x8`, `dot6@45`, …).
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase().replace('_', "-");
        Some(match n.as_str() {
            "none" | "no" | "off" => Self::None,
            "floyd-steinberg" | "floydsteinberg" | "fs" => Self::floyd_steinberg(),
            "jjn" | "jarvis" => Self::diffusion(DiffusionKernel::Jjn),
            "stucki" => Self::diffusion(DiffusionKernel::Stucki),
            "sierra3" | "sierra-3" | "sierra" => Self::diffusion(DiffusionKernel::Sierra3),
            "sierra2" | "sierra-2" => Self::diffusion(DiffusionKernel::Sierra2),
            "sierra-lite" | "sierralite" => Self::diffusion(DiffusionKernel::SierraLite),
            "atkinson" => Self::diffusion(DiffusionKernel::Atkinson),
            "hilbert" | "riemersma" => Self::hilbert(),
            other => Self::Ordered(ThresholdMap::named(other)?),
        })
    }
}

/// Visit every pixel of a `w × h` picture along Hilbert curves: squares
/// of side `2^k ≥ min(w, h)` laid along the longer side, each entered
/// next to where the previous one ended.
pub(crate) fn hilbert_walk(w: usize, h: usize, mut visit: impl FnMut(usize, usize)) {
    if w == 0 || h == 0 {
        return;
    }
    let side = w.min(h).next_power_of_two() as i64;
    let wide = w >= h;
    let tiles = (w.max(h)).div_ceil(side as usize) as i64;
    // The curve of the square spanned by `a` and `b` from corner `p`
    // starts at `p` and ends one square further along `a`; its four
    // quarters are the same curve with the axes exchanged, kept, kept,
    // and exchanged with both reversed.
    fn curve(p: (i64, i64), a: (i64, i64), b: (i64, i64), visit: &mut impl FnMut(i64, i64)) {
        if a.0.abs() + a.1.abs() == 1 {
            visit(p.0, p.1);
            return;
        }
        let (ha, hb) = ((a.0 / 2, a.1 / 2), (b.0 / 2, b.1 / 2));
        let (ua, ub) = ((a.0.signum(), a.1.signum()), (b.0.signum(), b.1.signum()));
        curve(p, hb, ha, visit);
        curve((p.0 + hb.0, p.1 + hb.1), ha, hb, visit);
        curve((p.0 + hb.0 + ha.0, p.1 + hb.1 + ha.1), ha, hb, visit);
        curve(
            (
                p.0 + a.0 - ua.0 + hb.0 - ub.0,
                p.1 + a.1 - ua.1 + hb.1 - ub.1,
            ),
            (-hb.0, -hb.1),
            (-ha.0, -ha.1),
            visit,
        );
    }
    let mut inside = |x: i64, y: i64| {
        if (x as usize) < w && (y as usize) < h {
            visit(x as usize, y as usize);
        }
    };
    for t in 0..tiles {
        if wide {
            curve((t * side, 0), (side, 0), (0, side), &mut inside);
        } else {
            curve((0, t * side), (0, side), (side, 0), &mut inside);
        }
    }
}

/// A tight `w × h` picture of `ch` interleaved samples whose first
/// `active` channels are dithered; `spread[c]` is the spacing of the
/// output values on channel `c` (ordered dither).
pub(crate) struct Job<'a> {
    pub(crate) data: &'a mut [f32],
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) ch: usize,
    pub(crate) active: usize,
    pub(crate) spread: &'a [f32],
}

/// Dither `job` in place. `quant` replaces a target pixel (its `active`
/// channels) by the output value nearest to it.
pub(crate) fn run(method: &DitherMethod, job: Job, quant: &(dyn Fn(&mut [f32]) + Sync)) {
    let Job {
        data,
        w,
        h,
        ch,
        active,
        spread,
    } = job;
    if w == 0 || h == 0 || active == 0 {
        return;
    }
    let row_len = w * ch;
    // The error taken into a pixel, bounded (see the module docs); a
    // non-finite residue (a NaN sample) diffuses nothing.
    let bounded = |e: f32| {
        if e.is_finite() {
            e.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    };
    match method {
        DitherMethod::None => {
            crate::exec::par_rows(data, row_len, |_, row| {
                for px in row.chunks_exact_mut(ch) {
                    quant(&mut px[..active]);
                }
            });
        }
        DitherMethod::Ordered(map) => {
            let inv = 1.0 / map.divisor as f32;
            let (mw, mh) = (map.width as usize, map.height as usize);
            crate::exec::par_rows(data, row_len, |y, row| {
                let cells = &map.cells[(y % mh) * mw..(y % mh + 1) * mw];
                for (x, px) in row.chunks_exact_mut(ch).enumerate() {
                    let d = (cells[x % mw] as f32 + 0.5) * inv - 0.5;
                    for (v, s) in px[..active].iter_mut().zip(spread) {
                        *v += d * s;
                    }
                    quant(&mut px[..active]);
                }
            });
        }
        DitherMethod::Diffusion { stencil, scan } => {
            let pad = stencil.reach_x;
            let rows = stencil.reach_y + 1;
            let padded = (w + 2 * pad) * active;
            // `rows` error lines, rotated as the scan descends.
            let mut err = vec![0.0f32; rows * padded];
            let mut target = vec![0.0f32; active];
            for y in 0..h {
                let reverse = matches!(scan, ScanOrder::Serpentine) && y % 2 == 1;
                let base = (y % rows) * padded;
                for col in 0..w {
                    let x = if reverse { w - 1 - col } else { col };
                    let px = &mut data[y * row_len + x * ch..][..active];
                    let own = base + (pad + x) * active;
                    for c in 0..active {
                        target[c] = px[c] + bounded(err[own + c]);
                    }
                    px.copy_from_slice(&target);
                    quant(px);
                    for (t, q) in target.iter_mut().zip(px.iter()) {
                        *t -= *q;
                        if !t.is_finite() {
                            *t = 0.0;
                        }
                    }
                    for &(dx, dy, wgt) in &stencil.taps {
                        let sdx = if reverse { -dx } else { dx };
                        let line = ((y + dy as usize) % rows) * padded;
                        let at = line + (pad + x).wrapping_add_signed(sdx as isize) * active;
                        for (e, t) in err[at..at + active].iter_mut().zip(&target) {
                            *e += t * wgt;
                        }
                    }
                }
                err[base..base + padded].fill(0.0);
            }
        }
        DitherMethod::Hilbert { history, ratio } => {
            let n = (*history).clamp(2, 256) as usize;
            let ratio = if ratio.is_finite() {
                ratio.max(1.0)
            } else {
                16.0
            };
            // w[k] for the error k steps back, summing to one.
            let mut weights: Vec<f32> = (0..n)
                .map(|k| ratio.powf(-(k as f32) / (n - 1) as f32))
                .collect();
            let total: f32 = weights.iter().sum();
            weights.iter_mut().for_each(|v| *v /= total);
            // Ring of the last `n` errors; `head` is the newest.
            let mut ring = vec![0.0f32; n * active];
            let mut head = 0usize;
            let mut target = vec![0.0f32; active];
            hilbert_walk(w, h, |x, y| {
                let px = &mut data[y * row_len + x * ch..][..active];
                target.fill(0.0);
                for (k, wgt) in weights.iter().enumerate() {
                    let slot = ((head + n - k) % n) * active;
                    for (t, e) in target.iter_mut().zip(&ring[slot..slot + active]) {
                        *t += e * wgt;
                    }
                }
                for (t, v) in target.iter_mut().zip(px.iter()) {
                    *t = v + bounded(*t);
                }
                px.copy_from_slice(&target);
                quant(px);
                head = (head + 1) % n;
                let slot = head * active;
                for ((e, t), q) in ring[slot..slot + active]
                    .iter_mut()
                    .zip(&target)
                    .zip(px.iter())
                {
                    *e = bounded(t - q);
                }
            });
        }
    }
}

/// Dither every colour channel (and alpha on request) to a number of
/// evenly spaced levels with any [`DitherMethod`].
///
/// The quantiser is `q(v) = round(clamp(v, 0, 1) · (L − 1)) / (L − 1)`
/// per channel. With [`DitherMethod::Ordered`] the threshold deviation
/// is scaled to the level spacing, so black stays black and white stays
/// white at any level count. ([`crate::OrderedDither`] is the same
/// family under the `-ordered-dither` boundary rule;
/// [`crate::Dither`] holds the tabulated kernels on the classic 8-bit
/// YUV layouts as well.)
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDither {
    /// How the error is spread.
    pub method: DitherMethod,
    /// Levels per colour channel `[R, G, B]` (grey uses the first), `≥ 2`.
    pub levels: [u32; 3],
    /// Levels for alpha (`None`: alpha untouched).
    pub alpha_levels: Option<u32>,
}

impl ChannelDither {
    /// Two levels per colour channel through `method`.
    pub fn new(method: DitherMethod) -> Self {
        Self {
            method,
            levels: [2; 3],
            alpha_levels: None,
        }
    }

    /// One level count for every colour channel.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = [levels.max(2); 3];
        self
    }

    /// Levels per colour channel.
    pub fn with_channel_levels(mut self, r: u32, g: u32, b: u32) -> Self {
        self.levels = [r.max(2), g.max(2), b.max(2)];
        self
    }

    /// Dither alpha too, to `levels` levels.
    pub fn with_alpha_levels(mut self, levels: u32) -> Self {
        self.alpha_levels = Some(levels.max(2));
        self
    }

    /// Dither a float picture in place.
    pub fn apply_float(&self, img: &mut deep::FloatImage) {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (ch, cc) = (img.channels(), img.color_channels());
        let active = if img.has_alpha() && self.alpha_levels.is_some() {
            ch
        } else {
            cc
        };
        let span: Vec<f32> = (0..active)
            .map(|c| {
                let l = if c < cc {
                    self.levels[c.min(2)]
                } else {
                    self.alpha_levels.unwrap_or(2)
                };
                (l.max(2) - 1) as f32
            })
            .collect();
        let spread: Vec<f32> = span.iter().map(|s| 1.0 / s).collect();
        let quant = |px: &mut [f32]| {
            for (v, s) in px.iter_mut().zip(&span) {
                *v = if v.is_nan() {
                    0.0
                } else {
                    (v.clamp(0.0, 1.0) * s + 0.5).floor() / s
                };
            }
        };
        let job = Job {
            data: img.samples_mut(),
            w,
            h,
            ch,
            active,
            spread: &spread,
        };
        run(&self.method, job, &quant);
    }
}

impl PixelFilter for ChannelDither {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            self.apply_float(&mut img);
            Ok(img)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deep::FloatImage;

    fn flat(w: u32, h: u32, v: f32) -> FloatImage {
        FloatImage::from_samples(w, h, 1, vec![v; (w * h) as usize]).unwrap()
    }

    fn mean(img: &FloatImage) -> f64 {
        img.samples().iter().map(|&v| v as f64).sum::<f64>() / img.samples().len() as f64
    }

    #[test]
    fn hilbert_walk_visits_every_pixel_once_in_unit_steps_per_square() {
        for (w, h) in [(1, 1), (8, 8), (5, 3), (3, 7), (16, 4), (33, 9), (2, 64)] {
            let mut seen = vec![0u8; w * h];
            let mut path = Vec::new();
            hilbert_walk(w, h, |x, y| {
                seen[y * w + x] += 1;
                path.push((x as i64, y as i64));
            });
            assert!(seen.iter().all(|&s| s == 1), "{w}x{h}");
            if w == h && w.is_power_of_two() {
                for s in path.windows(2) {
                    let d = (s[0].0 - s[1].0).abs() + (s[0].1 - s[1].1).abs();
                    assert_eq!(d, 1, "{w}x{h}: {:?} -> {:?}", s[0], s[1]);
                }
            }
        }
        // Adjacent squares join edge to edge.
        let mut path = Vec::new();
        hilbert_walk(16, 4, |x, y| path.push((x as i64, y as i64)));
        for s in path.windows(2) {
            assert_eq!((s[0].0 - s[1].0).abs() + (s[0].1 - s[1].1).abs(), 1);
        }
    }

    #[test]
    fn every_method_keeps_the_mean_of_a_flat_field() {
        let methods = [
            DitherMethod::floyd_steinberg(),
            DitherMethod::diffusion(DiffusionKernel::Jjn),
            DitherMethod::diffusion(DiffusionKernel::Stucki),
            DitherMethod::diffusion(DiffusionKernel::Sierra3),
            DitherMethod::diffusion(DiffusionKernel::Sierra2),
            DitherMethod::diffusion(DiffusionKernel::SierraLite),
            DitherMethod::hilbert(),
            DitherMethod::blue_noise(),
            DitherMethod::Ordered(ThresholdMap::bayer(8)),
            DitherMethod::Ordered(ThresholdMap::clustered_dot(8.0, 45.0, Default::default())),
        ];
        for (which, m) in methods.iter().enumerate() {
            for &v in &[0.1f32, 0.37, 0.5, 0.8] {
                for levels in [2u32, 3, 5] {
                    let mut img = flat(64, 64, v);
                    ChannelDither::new(m.clone())
                        .with_levels(levels)
                        .apply_float(&mut img);
                    let span = (levels - 1) as f32;
                    assert!(
                        img.samples()
                            .iter()
                            .all(|&s| (s * span - (s * span).round()).abs() < 1e-5),
                        "method {which}: a sample is not a level"
                    );
                    let got = mean(&img);
                    assert!(
                        (got - v as f64).abs() < 0.02,
                        "method {which} at {v} / {levels} levels: mean {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn extremes_stay_put_and_none_rounds() {
        for m in [
            DitherMethod::floyd_steinberg(),
            DitherMethod::hilbert(),
            DitherMethod::blue_noise(),
        ] {
            for v in [0.0f32, 1.0] {
                let mut img = flat(16, 16, v);
                ChannelDither::new(m.clone()).apply_float(&mut img);
                assert!(img.samples().iter().all(|&s| s == v), "a dither moved {v}");
            }
        }
        let mut img = flat(4, 4, 0.6);
        ChannelDither::new(DitherMethod::None)
            .with_levels(3)
            .apply_float(&mut img);
        assert!(img.samples().iter().all(|&s| s == 0.5));
    }

    #[test]
    fn stencil_rows_and_taps_agree_and_bad_taps_are_refused() {
        let rows = DiffusionStencil::from_rows(&[&[0., 0., 7.], &[3., 5., 1.]], 1, 16.).unwrap();
        assert_eq!(rows, DiffusionStencil::of(DiffusionKernel::FloydSteinberg));
        assert!(DiffusionStencil::new(&[(0, 0, 1.0)], 1.0).is_err());
        assert!(DiffusionStencil::new(&[(-1, 0, 1.0)], 1.0).is_err());
        assert!(DiffusionStencil::new(&[(1, 0, 1.0)], 0.0).is_err());
        assert!(DiffusionStencil::new(&[(40, 1, 1.0)], 1.0).is_err());
        assert!(DiffusionStencil::new(&[], 1.0).is_err());
        // A custom two-row stencil (÷32) runs like any other.
        let s =
            DiffusionStencil::from_rows(&[&[0., 0., 0., 8., 4.], &[2., 4., 8., 4., 2.]], 2, 32.)
                .unwrap();
        let mut img = flat(48, 48, 0.3);
        ChannelDither::new(DitherMethod::Diffusion {
            stencil: s,
            scan: ScanOrder::Serpentine,
        })
        .apply_float(&mut img);
        assert!((mean(&img) - 0.3).abs() < 0.02);
    }

    #[test]
    fn stencil_diffusion_matches_the_tabulated_dither() {
        // The engine and `Dither` are two implementations of §1.1.
        use crate::{Dither, ImageFilter};
        use oxideav_core::VideoPlane;
        let (w, h) = (23u32, 9u32);
        let data: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37) % 101) as f32 / 100.0)
            .collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize * 4,
                data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::GrayF32Le,
            width: w,
            height: h,
        };
        for kernel in [
            DiffusionKernel::FloydSteinberg,
            DiffusionKernel::Jjn,
            DiffusionKernel::Atkinson,
        ] {
            for scan in [ScanOrder::Raster, ScanOrder::Serpentine] {
                let want = Dither::error_diffusion(kernel)
                    .with_levels(3)
                    .with_scan(scan)
                    .apply(&frame, params)
                    .unwrap();
                let got = ChannelDither::new(DitherMethod::Diffusion {
                    stencil: DiffusionStencil::of(kernel),
                    scan,
                })
                .with_levels(3)
                .apply(&frame, params)
                .unwrap();
                assert_eq!(
                    want.planes[0].data, got.planes[0].data,
                    "{kernel:?} {scan:?}"
                );
            }
        }
    }
}
