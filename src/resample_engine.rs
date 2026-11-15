//! The resampling engine behind the distortion framework and the
//! cylindrical resize path: one reverse map (output position → source
//! position) with its local Jacobian, read through
//!
//! - a **virtual-pixel rule** for lookups outside the picture,
//! - a **point interpolation** of one continuous position, or
//! - an **elliptical weighted average**: the unit circle around an output
//!   pixel maps, to first order, onto the ellipse `{J q : |q| ≤ 1}` in the
//!   source; every source pixel `Δ` inside that ellipse scaled by the
//!   kernel radius is weighted by a radially symmetric kernel evaluated
//!   at `|J⁻¹ Δ|`. The ellipse axes are clamped to at least one source
//!   pixel (so an enlargement reconstructs instead of point-sampling) and
//!   very large footprints are gathered from a box-prefiltered pyramid
//!   level, which bounds the taps per output pixel.
//!
//! Colour is averaged weighted by coverage (premultiplied) wherever the
//! picture carries alpha, so a transparent pixel's colour never bleeds.
//! Output rows are independent and run under the caller's thread budget.

use std::sync::OnceLock;

use crate::distort_image::{PixelInterpolate, ReverseMap, VirtualPixel};
use crate::resize::RadialKernel;

/// Samples per pixel the engine carries (grey, grey + alpha, RGB, RGBA,
/// CMYK, a chroma pair).
pub(crate) const MAX_CH: usize = 4;

/// One pixel in the picture's own channel layout (unused tail is zero).
pub(crate) type Px = [f32; MAX_CH];

/// Entries of the radial weight table (indexed by squared distance).
const LUT_SIZE: usize = 4096;

/// Longest reach of the footprint (kernel radius × long semi-axis), in
/// pixels of the pyramid level read, before a coarser level is chosen.
const MAX_REACH: f64 = 64.0;

/// Largest kernel radius taken, in output pixels (a wider request is
/// narrowed to it).
const MAX_RADIUS: f64 = 32.0;

/// A radial kernel tabulated over the squared distance `q ∈ [0, R²)`.
pub(crate) struct EwaKernel {
    lut: Vec<f32>,
    radius: f64,
    /// `LUT_SIZE / R²`.
    index_scale: f64,
}

impl EwaKernel {
    pub(crate) fn new(k: &RadialKernel) -> Self {
        let radius = (k.radius() as f64).clamp(0.5, MAX_RADIUS);
        let r2 = radius * radius;
        let lut = (0..=LUT_SIZE + 1)
            .map(|i| {
                let q = (i as f64 / LUT_SIZE as f64 * r2).min(r2);
                // A kernel wider than the cap is compressed onto it.
                k.weight((q.sqrt() * (k.radius() as f64).max(0.5) / radius) as f32)
            })
            .collect();
        Self {
            lut,
            radius,
            index_scale: LUT_SIZE as f64 / r2,
        }
    }

    /// Weight at squared distance `q < R²` (linear interpolation of the
    /// table).
    #[inline]
    fn weight(&self, q: f64) -> f32 {
        let t = q * self.index_scale;
        let i = t as usize;
        let f = (t - i as f64) as f32;
        let a = self.lut[i];
        a + (self.lut[i + 1] - a) * f
    }
}

/// How each output pixel gathers its value.
pub(crate) enum Gather {
    Point(PixelInterpolate),
    Ewa(EwaKernel),
}

/// The constant colours the virtual-pixel rules use, in the picture's
/// channel layout.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeColours {
    pub background: Px,
    pub black: Px,
    pub white: Px,
    pub gray: Px,
    pub transparent: Px,
}

struct Level {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

/// A source picture with its edge rule.
pub(crate) struct Source<'a> {
    data: &'a [f32],
    w: usize,
    h: usize,
    ch: usize,
    /// Index of the alpha sample within a pixel, when there is one.
    alpha: Option<usize>,
    /// `1 / full scale` of the alpha sample.
    inv_full: f32,
    vp: VirtualPixel,
    colours: EdgeColours,
    /// Box-prefiltered halvings of the picture (level 1 upward), built
    /// on first use.
    pyramid: OnceLock<Vec<Level>>,
}

/// The 4 × 4 ordered-dither threshold map of
/// `docs/image/filter/dithering-kernels.md` (Bayer recurrence), used to
/// spread the `Dither` virtual pixel over the picture's rim.
const BAYER4: [[i64; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

#[inline]
fn tile(v: i64, n: i64) -> i64 {
    v.rem_euclid(n)
}

#[inline]
fn mirror(v: i64, n: i64) -> i64 {
    let m = v.rem_euclid(2 * n);
    if m < n { m } else { 2 * n - 1 - m }
}

impl<'a> Source<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        data: &'a [f32],
        w: usize,
        h: usize,
        ch: usize,
        alpha: Option<usize>,
        full: f32,
        vp: VirtualPixel,
        colours: EdgeColours,
    ) -> Self {
        debug_assert!((1..=MAX_CH).contains(&ch) && data.len() == w * h * ch);
        Self {
            data,
            w,
            h,
            ch,
            alpha,
            inv_full: 1.0 / full,
            vp,
            colours,
            pyramid: OnceLock::new(),
        }
    }

    #[inline]
    fn load(data: &[f32], ch: usize, i: usize) -> Px {
        let mut p = [0.0; MAX_CH];
        p[..ch].copy_from_slice(&data[i..i + ch]);
        p
    }

    /// The source index a virtual lookup resolves to, or the constant it
    /// reads.
    fn resolve(&self, x: i64, y: i64, w: i64, h: i64) -> Result<(i64, i64), Px> {
        let cx = x.clamp(0, w - 1);
        let cy = y.clamp(0, h - 1);
        let c = &self.colours;
        Ok(match self.vp {
            VirtualPixel::Edge => (cx, cy),
            VirtualPixel::Background => return Err(c.background),
            VirtualPixel::Black => return Err(c.black),
            VirtualPixel::White => return Err(c.white),
            VirtualPixel::Gray => return Err(c.gray),
            VirtualPixel::Transparent => return Err(c.transparent),
            VirtualPixel::Tile => (tile(x, w), tile(y, h)),
            VirtualPixel::Mirror => (mirror(x, w), mirror(y, h)),
            VirtualPixel::HorizontalTile => {
                if y < 0 || y >= h {
                    return Err(c.background);
                }
                (tile(x, w), y)
            }
            VirtualPixel::VerticalTile => {
                if x < 0 || x >= w {
                    return Err(c.background);
                }
                (x, tile(y, h))
            }
            VirtualPixel::HorizontalTileEdge => (tile(x, w), cy),
            VirtualPixel::VerticalTileEdge => (cx, tile(y, h)),
            VirtualPixel::CheckerTile => {
                if (x.div_euclid(w) + y.div_euclid(h)).rem_euclid(2) != 0 {
                    return Err(c.background);
                }
                (tile(x, w), tile(y, h))
            }
            VirtualPixel::Dither => {
                // An ordered offset of up to ±8 pixels, then the nearest
                // pixel of the picture: the rim is scattered outward
                // instead of smeared.
                let ox = BAYER4[(y & 3) as usize][(x & 3) as usize] - 8;
                let oy = BAYER4[(x & 3) as usize][((y + 2) & 3) as usize] - 8;
                ((x + ox).clamp(0, w - 1), (y + oy).clamp(0, h - 1))
            }
            VirtualPixel::Random => {
                let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 29;
                ((z % w as u64) as i64, ((z >> 32) % h as u64) as i64)
            }
        })
    }

    /// The pixel at integer index `(x, y)` of a level, virtual outside.
    #[inline]
    fn at_level(&self, data: &[f32], w: usize, h: usize, x: i64, y: i64) -> Px {
        let ch = self.ch;
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            return Self::load(data, ch, (y as usize * w + x as usize) * ch);
        }
        match self.resolve(x, y, w as i64, h as i64) {
            Ok((sx, sy)) => Self::load(data, ch, (sy as usize * w + sx as usize) * ch),
            Err(constant) => constant,
        }
    }

    /// The pixel at integer index `(x, y)`, virtual outside.
    #[inline]
    pub(crate) fn at(&self, x: i64, y: i64) -> Px {
        self.at_level(self.data, self.w, self.h, x, y)
    }

    /// Normalise a weighted gather: `colour` holds `Σ w·c·a` (or `Σ w·c`
    /// without alpha), `cover` `Σ w·a`, `wsum` `Σ w`.
    #[inline]
    fn finish(&self, mut acc: Px, cover: f32, wsum: f32) -> Option<Px> {
        if wsum.abs() < 1e-9 {
            return None;
        }
        match self.alpha {
            None => {
                let inv = 1.0 / wsum;
                for v in &mut acc[..self.ch] {
                    *v *= inv;
                }
            }
            Some(ai) => {
                let inv = if cover.abs() > 1e-9 { 1.0 / cover } else { 0.0 };
                for v in &mut acc[..self.ch] {
                    *v *= inv;
                }
                acc[ai] = cover / (wsum * self.inv_full);
            }
        }
        Some(acc)
    }

    /// Add `w · p` to a gather.
    #[inline]
    fn add(&self, acc: &mut Px, cover: &mut f32, p: &[f32], w: f32) {
        match self.alpha {
            None => {
                for (a, v) in acc[..self.ch].iter_mut().zip(p) {
                    *a += w * v;
                }
            }
            Some(ai) => {
                let wa = w * p[ai] * self.inv_full;
                for (a, v) in acc[..self.ch].iter_mut().zip(p) {
                    *a += wa * v;
                }
                *cover += wa;
            }
        }
    }

    /// Weighted mean of `taps` (coverage-weighted colour).
    fn blend<const N: usize>(&self, taps: &[(Px, f32); N]) -> Px {
        let mut acc = [0.0; MAX_CH];
        let (mut cover, mut wsum) = (0.0, 0.0);
        for (p, w) in taps {
            self.add(&mut acc, &mut cover, p, *w);
            wsum += w;
        }
        self.finish(acc, cover, wsum).unwrap_or([0.0; MAX_CH])
    }

    fn mean(&self, x0: i64, y0: i64, n: i64) -> Px {
        let mut acc = [0.0; MAX_CH];
        let (mut cover, mut wsum) = (0.0, 0.0);
        for y in y0..y0 + n {
            for x in x0..x0 + n {
                self.add(&mut acc, &mut cover, &self.at(x, y), 1.0);
                wsum += 1.0;
            }
        }
        self.finish(acc, cover, wsum).unwrap_or([0.0; MAX_CH])
    }

    /// Point read at continuous index position `(x, y)` (pixel `i`'s
    /// centre is at index `i`).
    pub(crate) fn interpolate(&self, x: f64, y: f64, interp: PixelInterpolate) -> Px {
        if !(x.is_finite() && y.is_finite()) {
            return self.colours.background;
        }
        // Positions far outside every representable index read as the
        // nearest representable one (the edge rule decides the value).
        let lim = 1e15;
        let (x, y) = (x.clamp(-lim, lim), y.clamp(-lim, lim));
        let (fx, fy) = (x.floor(), y.floor());
        let (ix, iy) = (fx as i64, fy as i64);
        let (tx, ty) = ((x - fx) as f32, (y - fy) as f32);
        match interp {
            PixelInterpolate::Integer => self.at(ix, iy),
            PixelInterpolate::Nearest => {
                self.at((x + 0.5).floor() as i64, (y + 0.5).floor() as i64)
            }
            PixelInterpolate::Background => self.colours.background,
            PixelInterpolate::Average => self.mean(ix, iy, 2),
            PixelInterpolate::Average9 => self.mean(
                (x + 0.5).floor() as i64 - 1,
                (y + 0.5).floor() as i64 - 1,
                3,
            ),
            PixelInterpolate::Average16 => self.mean(ix - 1, iy - 1, 4),
            PixelInterpolate::Bilinear => {
                if tx == 0.0 && ty == 0.0 {
                    return self.at(ix, iy);
                }
                self.blend(&[
                    (self.at(ix, iy), (1.0 - tx) * (1.0 - ty)),
                    (self.at(ix + 1, iy), tx * (1.0 - ty)),
                    (self.at(ix, iy + 1), (1.0 - tx) * ty),
                    (self.at(ix + 1, iy + 1), tx * ty),
                ])
            }
            PixelInterpolate::Mesh => {
                if tx == 0.0 && ty == 0.0 {
                    return self.at(ix, iy);
                }
                // The cell is split along the diagonal whose two ends
                // differ least and the position is read from the plane
                // through its triangle's three corners.
                let p00 = self.at(ix, iy);
                let p10 = self.at(ix + 1, iy);
                let p01 = self.at(ix, iy + 1);
                let p11 = self.at(ix + 1, iy + 1);
                let diff = |a: &Px, b: &Px| -> f32 {
                    a[..self.ch]
                        .iter()
                        .zip(&b[..self.ch])
                        .map(|(u, v)| (u - v).abs())
                        .sum()
                };
                if diff(&p00, &p11) <= diff(&p10, &p01) {
                    // Diagonal (0,0)–(1,1).
                    if tx >= ty {
                        self.blend(&[(p00, 1.0 - tx), (p10, tx - ty), (p11, ty)])
                    } else {
                        self.blend(&[(p00, 1.0 - ty), (p01, ty - tx), (p11, tx)])
                    }
                } else if tx + ty <= 1.0 {
                    // Diagonal (1,0)–(0,1).
                    self.blend(&[(p00, 1.0 - tx - ty), (p10, tx), (p01, ty)])
                } else {
                    self.blend(&[(p11, tx + ty - 1.0), (p10, 1.0 - ty), (p01, 1.0 - tx)])
                }
            }
            PixelInterpolate::Catrom | PixelInterpolate::Spline => {
                let (b, c) = if interp == PixelInterpolate::Catrom {
                    (0.0, 0.5)
                } else {
                    (1.0, 0.0)
                };
                if b == 0.0 && tx == 0.0 && ty == 0.0 {
                    // An interpolating cubic at a sample is the sample.
                    return self.at(ix, iy);
                }
                let wx = cubic_weights(tx, b, c);
                let wy = cubic_weights(ty, b, c);
                let mut acc = [0.0; MAX_CH];
                let (mut cover, mut wsum) = (0.0, 0.0);
                for (m, &wym) in wy.iter().enumerate() {
                    for (n, &wxn) in wx.iter().enumerate() {
                        let w = wxn * wym;
                        let p = self.at(ix - 1 + n as i64, iy - 1 + m as i64);
                        self.add(&mut acc, &mut cover, &p, w);
                        wsum += w;
                    }
                }
                self.finish(acc, cover, wsum).unwrap_or([0.0; MAX_CH])
            }
        }
    }

    /// The pyramid: level `l ≥ 1` halves level `l − 1` (level 0 is the
    /// picture), each pixel the coverage-weighted mean of its 2 × 2
    /// block, down to a single pixel.
    fn pyramid(&self) -> &[Level] {
        self.pyramid.get_or_init(|| {
            let mut levels: Vec<Level> = Vec::new();
            let ch = self.ch;
            loop {
                let (pw, ph, prev): (usize, usize, &[f32]) = match levels.last() {
                    None => (self.w, self.h, self.data),
                    Some(l) => (l.w, l.h, &l.data),
                };
                if pw <= 1 && ph <= 1 {
                    break;
                }
                let (w, h) = (pw.div_ceil(2), ph.div_ceil(2));
                let mut data = vec![0.0f32; w * h * ch];
                for (y, row) in data.chunks_exact_mut(w * ch).enumerate() {
                    for (x, out) in row.chunks_exact_mut(ch).enumerate() {
                        let mut acc = [0.0; MAX_CH];
                        let mut plain = [0.0; MAX_CH];
                        let (mut cover, mut n) = (0.0f32, 0.0f32);
                        for sy in (2 * y)..(2 * y + 2).min(ph) {
                            for sx in (2 * x)..(2 * x + 2).min(pw) {
                                let p = &prev[(sy * pw + sx) * ch..(sy * pw + sx) * ch + ch];
                                self.add(&mut acc, &mut cover, p, 1.0);
                                for (a, v) in plain.iter_mut().zip(p) {
                                    *a += v;
                                }
                                n += 1.0;
                            }
                        }
                        // Where nothing is covered the plain mean keeps
                        // the colour.
                        let px = match self.alpha {
                            Some(_) if cover.abs() <= 1e-9 => plain.map(|v| v / n),
                            _ => self.finish(acc, cover, n).unwrap_or([0.0; MAX_CH]),
                        };
                        out.copy_from_slice(&px[..ch]);
                    }
                }
                levels.push(Level { w, h, data });
            }
            levels
        })
    }

    /// Elliptical weighted average at continuous position `(u, v)`
    /// (pixel `i` covers `[i, i + 1)`) under the reverse map's Jacobian
    /// `jac = [[∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y]]`.
    pub(crate) fn ewa(&self, u: f64, v: f64, jac: [[f64; 2]; 2], k: &EwaKernel) -> Px {
        let [[a, b], [c, d]] = jac;
        // G = J Jᵀ: its eigenvalues are the squared semi-axes of the
        // footprint ellipse, its eigenvectors their directions.
        let g00 = a * a + b * b;
        let g01 = a * c + b * d;
        let g11 = c * c + d * d;
        if !(g00.is_finite()
            && g01.is_finite()
            && g11.is_finite()
            && u.is_finite()
            && v.is_finite())
        {
            return self.interpolate(u - 0.5, v - 0.5, PixelInterpolate::Bilinear);
        }
        let (u, v) = (u.clamp(-1e15, 1e15), v.clamp(-1e15, 1e15));
        let mean = 0.5 * (g00 + g11);
        let half = 0.5 * (g00 - g11);
        let spread = (half * half + g01 * g01).sqrt();
        let theta = if spread > 0.0 {
            0.5 * g01.atan2(half)
        } else {
            0.0
        };
        // Clamp: never thinner than one source pixel, never longer than
        // the picture.
        let cap = (self.w.max(self.h) as f64).max(4.0);
        let cap2 = cap * cap;
        let mut major2 = (mean + spread).clamp(1.0, cap2);
        let mut minor2 = (mean - spread).clamp(1.0, cap2);

        // Pyramid level: exact gathering while the thin axis is under
        // four pixels; beyond that read a level where it spans two to
        // four, and a coarser one still while the long axis is too long.
        let mut level = 0usize;
        let minor = minor2.sqrt();
        if minor >= 4.0 {
            level = (minor.log2().floor() as usize).saturating_sub(1);
        }
        let major = major2.sqrt();
        while k.radius * major / (1u64 << level.min(40)) as f64 > MAX_REACH && level < 40 {
            level += 1;
        }
        let (data, w, h): (&[f32], usize, usize) = if level == 0 {
            (self.data, self.w, self.h)
        } else {
            let pyr = self.pyramid();
            level = level.min(pyr.len());
            match level {
                0 => (self.data, self.w, self.h),
                l => (&pyr[l - 1].data, pyr[l - 1].w, pyr[l - 1].h),
            }
        };
        let f = (1u64 << level) as f64;
        major2 = (major2 / (f * f)).max(1.0);
        minor2 = (minor2 / (f * f)).max(1.0);

        // Quadratic form Q(Δ) = Δᵀ G'⁻¹ Δ = A dx² + 2B dx dy + C dy².
        let (sn, cs) = theta.sin_cos();
        let (i1, i2) = (1.0 / major2, 1.0 / minor2);
        let qa = cs * cs * i1 + sn * sn * i2;
        let qb = cs * sn * (i1 - i2);
        let qc = sn * sn * i1 + cs * cs * i2;
        let r2 = k.radius * k.radius;
        let det = qa * qc - qb * qb;
        let (cx, cy) = (u / f - 0.5, v / f - 0.5);
        let y_reach = k.radius * (qa / det).sqrt();
        let (y0, y1) = ((cy - y_reach).ceil() as i64, (cy + y_reach).floor() as i64);

        let ch = self.ch;
        let mut acc = [0.0; MAX_CH];
        let (mut cover, mut wsum) = (0.0f32, 0.0f32);
        for y in y0..=y1 {
            let dy = y as f64 - cy;
            // Roots of A dx² + 2B dy dx + (C dy² − R²) = 0.
            let disc = qb * qb * dy * dy - qa * (qc * dy * dy - r2);
            if disc <= 0.0 {
                continue;
            }
            let root = disc.sqrt();
            let mid = -qb * dy / qa;
            let x0 = (cx + mid - root / qa).ceil() as i64;
            let x1 = (cx + mid + root / qa).floor() as i64;
            let row_inside = y >= 0 && (y as usize) < h && x0 >= 0 && x1 < w as i64;
            let base = if row_inside { y as usize * w * ch } else { 0 };
            for x in x0..=x1 {
                let dx = x as f64 - cx;
                let q = qa * dx * dx + 2.0 * qb * dx * dy + qc * dy * dy;
                if q >= r2 {
                    continue;
                }
                let wgt = k.weight(q);
                if wgt == 0.0 {
                    continue;
                }
                if row_inside {
                    let i = base + x as usize * ch;
                    self.add(&mut acc, &mut cover, &data[i..i + ch], wgt);
                } else {
                    let p = self.at_level(data, w, h, x, y);
                    self.add(&mut acc, &mut cover, &p, wgt);
                }
                wsum += wgt;
            }
        }
        match self.finish(acc, cover, wsum) {
            Some(p) => p,
            None => self.interpolate(u - 0.5, v - 0.5, PixelInterpolate::Bilinear),
        }
    }
}

/// The four taps of the cubic `BC(b, c)` at fraction `t`.
fn cubic_weights(t: f32, b: f32, c: f32) -> [f32; 4] {
    let k = |x: f32| {
        let ax = x.abs();
        let v = if ax < 1.0 {
            (12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
                + (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
                + (6.0 - 2.0 * b)
        } else if ax < 2.0 {
            (-b - 6.0 * c) * ax * ax * ax
                + (6.0 * b + 30.0 * c) * ax * ax
                + (-12.0 * b - 48.0 * c) * ax
                + (8.0 * b + 24.0 * c)
        } else {
            0.0
        };
        v / 6.0
    };
    [k(t + 1.0), k(t), k(t - 1.0), k(t - 2.0)]
}

/// The output window of a render: `width × height` pixels whose
/// top-left corner sits at `(x, y)` of the map's output plane, each
/// output pixel `1 / scale` wide there.
#[derive(Clone, Copy, Debug)]
pub(crate) struct View {
    pub width: usize,
    pub height: usize,
    pub x: f64,
    pub y: f64,
    pub scale: f64,
}

/// Render `view` of `map` over `src`: `width × height × ch` samples.
/// `undefined` is written where the map has no source position.
pub(crate) fn render<M: ReverseMap + ?Sized>(
    src: &Source<'_>,
    map: &M,
    view: View,
    gather: &Gather,
    undefined: Px,
) -> Vec<f32> {
    let ch = src.ch;
    let mut out = vec![0.0f32; view.width * view.height * ch];
    let step = 1.0 / view.scale;
    crate::exec::par_rows(&mut out, view.width * ch, |j, row| {
        let y = (j as f64 + 0.5) * step + view.y;
        for (i, px) in row.chunks_exact_mut(ch).enumerate() {
            let x = (i as f64 + 0.5) * step + view.x;
            let value = match map.source(x, y) {
                None => undefined,
                Some((u, v)) => match gather {
                    Gather::Point(interp) => src.interpolate(u - 0.5, v - 0.5, *interp),
                    Gather::Ewa(k) => {
                        let [[a, b], [c, d]] = map.jacobian(x, y);
                        src.ewa(u, v, [[a * step, b * step], [c * step, d * step]], k)
                    }
                },
            };
            px.copy_from_slice(&value[..ch]);
        }
    });
    out
}

/// Plain axis scaling `(x, y) → (x·sx, y·sy)`: the reverse map of a
/// resize.
pub(crate) struct AxisScale(pub f64, pub f64);

impl ReverseMap for AxisScale {
    fn source(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x * self.0, y * self.1))
    }

    fn jacobian(&self, _x: f64, _y: f64) -> [[f64; 2]; 2] {
        [[self.0, 0.0], [0.0, self.1]]
    }
}
