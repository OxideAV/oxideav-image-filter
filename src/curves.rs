//! Curves — per-channel monotone cubic curves adjustment.
//!
//! "Curves" is the per-channel tonal-mapping primitive in every classical
//! photo-editor. The user specifies a small set of control points
//! `(x, y)` on `[0, 255]² → [0, 255]²` and the editor fits a smooth
//! monotone curve through them. Four classic interpolants apply:
//!
//! 1. **Linear** — straight segments between control points. Cheapest;
//!    visible kinks at every knot.
//! 2. **Catmull-Rom** — cubic interpolation through the points with
//!    tangents inferred from neighbours (Catmull, Rom — "A class of
//!    local interpolating splines", Computer-Aided Geometric Design,
//!    1974). Smooth but **not** monotone-safe (can overshoot below `0`
//!    or above `1`).
//! 3. **Monotone-cubic (Fritsch-Carlson)** — Fritsch, Carlson —
//!    "Monotone Piecewise Cubic Interpolation", SIAM J. Numer. Anal.
//!    17 (1980), §3 algorithm. Cubic Hermite with tangents adjusted so
//!    the interpolant is monotone if the input data is monotone. This
//!    is what photo-editors actually want — sliders moving up should
//!    never invert a tone range.
//! 4. **Natural cubic spline** — the classical `C²` interpolant minimising
//!    curvature with the "natural" boundary condition `P''(x_0) =
//!    P''(x_{n−1}) = 0` (de Boor — *A Practical Guide to Splines*,
//!    Springer 1978; tridiagonal step credited to L. H. Thomas, Watson
//!    Sci. Comp. Lab. 1949). Unlike the three local methods above the
//!    second-derivative samples `M_i = P''(x_i)` are obtained from a
//!    single global `O(n)` tridiagonal solve, then each segment is
//!    evaluated as the cubic of equation `P(x) = y_i + (Δ_i −
//!    h_i·(2M_i + M_{i+1})/6)·t + (M_i / 2)·t² + ((M_{i+1} − M_i) /
//!    (6 h_i))·t³` with `t = x − x_i`. C² smoothness (the second
//!    derivative is continuous, not just the first) gives this
//!    interpolant the gentlest visible inflections of the four, at the
//!    cost of being **not** monotone-safe — it can overshoot just like
//!    Catmull-Rom and should not be the default for tone-curve UIs
//!    where slider monotonicity must hold.
//!
//! We implement all four under [`CurveInterpolation`] and apply the
//! curve as a per-channel 256-entry LUT.
//!
//! Channels supported: a single `master` curve always runs on every
//! tone channel; optional `red`/`green`/`blue` curves replace the
//! master on the corresponding RGB / RGBA channels. Alpha is
//! pass-through.
//!
//! Operates on `Gray8` / `Rgb24` / `Rgba` and planar YUV (master curve
//! on luma plane only — chroma untouched, matching the documented CLI's
//! `-channel RGB -function …` convention for tonal ops).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Interpolation mode for [`Curves`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurveInterpolation {
    /// Straight segments between knots — no smoothing.
    Linear,
    /// Catmull-Rom cubic. Smooth but can overshoot.
    CatmullRom,
    /// Fritsch-Carlson monotone-cubic Hermite. Default — smooth and
    /// monotonicity-preserving.
    #[default]
    MonotoneCubic,
    /// Natural cubic spline (`C²`) via the Thomas tridiagonal solver
    /// (de Boor 1978; tridiagonal step credited to L. H. Thomas, Watson
    /// Sci. Comp. Lab. 1949). Smoothest of the four (continuous second
    /// derivative) but can overshoot like Catmull-Rom; suited to
    /// gentle-curve work where C² visual smoothness matters more than
    /// strict monotone safety.
    NaturalCubic,
}

/// A single tonal curve: an ordered list of `(input, output)` control
/// points on `[0, 255]`.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    /// Control points sorted by `x` ascending. Endpoints `(0, …)` and
    /// `(255, …)` are auto-injected if missing.
    pub points: Vec<(u8, u8)>,
}

impl Curve {
    /// New curve through these control points (auto-sorts, dedups by
    /// `x`).
    pub fn new(points: impl IntoIterator<Item = (u8, u8)>) -> Self {
        let mut points: Vec<_> = points.into_iter().collect();
        points.sort_by_key(|p| p.0);
        points.dedup_by_key(|p| p.0);
        Self { points }
    }

    /// Identity curve `y = x` (single segment).
    pub fn identity() -> Self {
        Self {
            points: vec![(0, 0), (255, 255)],
        }
    }

    fn ensure_endpoints(&self) -> Vec<(f32, f32)> {
        let mut out: Vec<(f32, f32)> = self
            .points
            .iter()
            .map(|p| (p.0 as f32, p.1 as f32))
            .collect();
        if out.is_empty() || out[0].0 > 0.0 {
            // Inject the natural left endpoint at y = first y (held).
            let y0 = out.first().map(|p| p.1).unwrap_or(0.0);
            out.insert(0, (0.0, y0));
        }
        if out.last().is_none() || out.last().unwrap().0 < 255.0 {
            let yn = out.last().map(|p| p.1).unwrap_or(255.0);
            out.push((255.0, yn));
        }
        out
    }

    fn build_lut(&self, mode: CurveInterpolation) -> [u8; 256] {
        let pts = self.ensure_endpoints();
        let n = pts.len();
        if n < 2 {
            // Trivial: identity.
            let mut lut = [0u8; 256];
            for (v, slot) in lut.iter_mut().enumerate() {
                *slot = v as u8;
            }
            return lut;
        }
        // Natural cubic spline takes a different path: a tridiagonal
        // solve for the per-knot second derivatives `M_i` instead of
        // per-knot tangents `m_i`.
        if matches!(mode, CurveInterpolation::NaturalCubic) {
            return Self::build_natural_cubic_lut(&pts, n);
        }
        // Per-segment tangents for the cubic-Hermite paths.
        let mut tangents = vec![0f32; n];
        match mode {
            CurveInterpolation::Linear => {}
            CurveInterpolation::NaturalCubic => unreachable!(),
            CurveInterpolation::CatmullRom => {
                // Standard Catmull-Rom tangents (centred differences;
                // one-sided at the boundaries).
                for i in 0..n {
                    let (xl, yl) = if i == 0 { pts[0] } else { pts[i - 1] };
                    let (xr, yr) = if i + 1 == n { pts[n - 1] } else { pts[i + 1] };
                    let dx = xr - xl;
                    tangents[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        (yr - yl) / dx
                    };
                }
            }
            CurveInterpolation::MonotoneCubic => {
                // Fritsch-Carlson §3. Step 1: secant slopes.
                let mut delta = vec![0f32; n - 1];
                for i in 0..n - 1 {
                    let dx = pts[i + 1].0 - pts[i].0;
                    delta[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        (pts[i + 1].1 - pts[i].1) / dx
                    };
                }
                // Step 2: initial tangents = average of neighbouring
                // secants (Hermite default), one-sided at boundary.
                tangents[0] = delta[0];
                tangents[n - 1] = delta[n - 2];
                for i in 1..n - 1 {
                    tangents[i] = 0.5 * (delta[i - 1] + delta[i]);
                }
                // Step 3: enforce monotonicity (Fritsch-Carlson eq. 4).
                for i in 0..n - 1 {
                    if delta[i].abs() < 1.0e-9 {
                        tangents[i] = 0.0;
                        tangents[i + 1] = 0.0;
                        continue;
                    }
                    let a = tangents[i] / delta[i];
                    let b = tangents[i + 1] / delta[i];
                    if a < 0.0 {
                        tangents[i] = 0.0;
                    }
                    if b < 0.0 {
                        tangents[i + 1] = 0.0;
                    }
                    let s = a * a + b * b;
                    if s > 9.0 {
                        let t = 3.0 / s.sqrt();
                        tangents[i] = t * a * delta[i];
                        tangents[i + 1] = t * b * delta[i];
                    }
                }
            }
        }

        let mut lut = [0u8; 256];
        let mut seg = 0;
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32;
            while seg + 1 < n && pts[seg + 1].0 < x {
                seg += 1;
            }
            let i = seg;
            let j = (i + 1).min(n - 1);
            let (x0, y0) = pts[i];
            let (x1, y1) = pts[j];
            let dx = x1 - x0;
            let t = if dx.abs() < 1.0e-6 {
                0.0
            } else {
                (x - x0) / dx
            };
            let y = match mode {
                CurveInterpolation::Linear => y0 + (y1 - y0) * t,
                _ => {
                    // Cubic Hermite (Wikipedia §"Cubic Hermite spline"
                    // / Fritsch-Carlson §2.1).
                    let h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
                    let h10 = t * (1.0 - t) * (1.0 - t);
                    let h01 = t * t * (3.0 - 2.0 * t);
                    let h11 = t * t * (t - 1.0);
                    h00 * y0 + h10 * dx * tangents[i] + h01 * y1 + h11 * dx * tangents[j]
                }
            };
            *slot = y.clamp(0.0, 255.0).round() as u8;
        }
        lut
    }

    /// Build the LUT for the natural cubic spline (§4 of the curve-
    /// interpolation reference doc). The path is separate from
    /// [`Self::build_lut`] because the natural spline is a `C²`
    /// interpolant parameterised by per-knot second-derivatives
    /// `M_i = P''(x_i)`, not by per-knot tangents `m_i = P'(x_i)` like
    /// the cubic-Hermite paths.
    ///
    /// Steps (cited from `docs/image/filter/curve-interpolation.md`
    /// §4.1–4.4, originally de Boor 1978 / Thomas 1949):
    ///
    /// 1. Build the tridiagonal system
    ///    `h_{i−1}·M_{i−1} + 2(h_{i−1}+h_i)·M_i + h_i·M_{i+1}
    ///       = 6·(Δ_i − Δ_{i−1})` for `i = 1 … n−2`, with the
    ///    natural boundary `M_0 = M_{n−1} = 0`.
    /// 2. Solve the system in `O(n)` with the Thomas forward-
    ///    elimination + back-substitution sweep.
    /// 3. Evaluate each segment as
    ///    `P(x) = y_i + (Δ_i − h_i·(2M_i + M_{i+1})/6)·t
    ///          + (M_i / 2)·t² + ((M_{i+1} − M_i)/(6 h_i))·t³`
    ///    with `t = x − x_i`.
    fn build_natural_cubic_lut(pts: &[(f32, f32)], n: usize) -> [u8; 256] {
        // n ≥ 2 is guaranteed by `build_lut`'s early-return. With
        // exactly 2 points there are no interior knots so M ≡ 0 and the
        // spline is a straight line — fall through to the same
        // evaluator with empty M vector.
        let mut h = vec![0f32; n.saturating_sub(1)];
        let mut delta = vec![0f32; n.saturating_sub(1)];
        for i in 0..n - 1 {
            let dx = pts[i + 1].0 - pts[i].0;
            h[i] = dx;
            delta[i] = if dx.abs() < 1.0e-6 {
                0.0
            } else {
                (pts[i + 1].1 - pts[i].1) / dx
            };
        }

        // Per-knot second derivatives. Boundary M_0 = M_{n−1} = 0 by
        // construction; only the interior `i = 1 … n−2` slots are
        // touched by the Thomas sweep.
        let mut m = vec![0f32; n];

        if n >= 3 {
            // Thomas algorithm on the interior system of size n − 2.
            // Coefficients per §4.3:
            //   a_i = h_{i−1},  b_i = 2(h_{i−1} + h_i),  c_i = h_i,
            //   d_i = 6 (Δ_i − Δ_{i−1}).
            // c'_i and d'_i are the elimination's modified
            // super-diagonal and RHS; `inv` caches `1 / (b_i −
            // a_i·c'_{i−1})` to avoid two divisions per step.
            let interior = n - 2;
            let mut c_prime = vec![0f32; interior];
            let mut d_prime = vec![0f32; interior];

            // Forward elimination.
            for k in 0..interior {
                // i is the global knot index (1-based interior).
                let i = k + 1;
                let a = h[i - 1];
                let b = 2.0 * (h[i - 1] + h[i]);
                let c = h[i];
                let d = 6.0 * (delta[i] - delta[i - 1]);
                if k == 0 {
                    let inv = if b.abs() < 1.0e-12 { 0.0 } else { 1.0 / b };
                    c_prime[k] = c * inv;
                    d_prime[k] = d * inv;
                } else {
                    let denom = b - a * c_prime[k - 1];
                    let inv = if denom.abs() < 1.0e-12 {
                        0.0
                    } else {
                        1.0 / denom
                    };
                    c_prime[k] = c * inv;
                    d_prime[k] = (d - a * d_prime[k - 1]) * inv;
                }
            }

            // Back substitution: M_i = d'_i − c'_i · M_{i+1}. The
            // "next" knot starts as the closed boundary M_{n−1} = 0.
            let mut next = 0.0f32;
            for k in (0..interior).rev() {
                let i = k + 1;
                m[i] = d_prime[k] - c_prime[k] * next;
                next = m[i];
            }
        }

        // §4.4 evaluation: walk x = 0..=255 and pick segment `i` such
        // that `pts[i].0 ≤ x < pts[i+1].0`.
        let mut lut = [0u8; 256];
        let mut seg = 0;
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32;
            while seg + 1 < n && pts[seg + 1].0 < x {
                seg += 1;
            }
            let i = seg;
            let j = (i + 1).min(n - 1);
            let (x0, y0) = pts[i];
            let hi = h[i.min(h.len().saturating_sub(1))];
            let dt = x - x0;
            // Defensive: zero-width segment (degenerate knot collision)
            // collapses to the segment's left endpoint value to mirror
            // the Hermite path's handling.
            let y = if hi.abs() < 1.0e-6 {
                y0
            } else {
                let mi = m[i];
                let mj = m[j];
                let di = delta[i.min(delta.len().saturating_sub(1))];
                // P(x) = y_i + (Δ_i − h_i·(2 M_i + M_{i+1})/6)·t
                //          + (M_i / 2)·t² + ((M_{i+1} − M_i)/(6 h_i))·t³
                let c1 = di - hi * (2.0 * mi + mj) / 6.0;
                let c2 = mi / 2.0;
                let c3 = (mj - mi) / (6.0 * hi);
                y0 + c1 * dt + c2 * dt * dt + c3 * dt * dt * dt
            };
            *slot = y.clamp(0.0, 255.0).round() as u8;
        }
        lut
    }
}

/// Per-channel curves filter.
#[derive(Clone, Debug, Default)]
pub struct Curves {
    /// Master curve applied to every tone channel by default.
    pub master: Curve,
    /// Per-channel overrides. `Some(c)` replaces the master on that
    /// channel; `None` falls back to the master.
    pub red: Option<Curve>,
    pub green: Option<Curve>,
    pub blue: Option<Curve>,
    /// Interpolation kind for all curves in this filter.
    pub interpolation: CurveInterpolation,
}

impl Curves {
    pub fn new(master: Curve) -> Self {
        Self {
            master,
            red: None,
            green: None,
            blue: None,
            interpolation: CurveInterpolation::default(),
        }
    }

    pub fn with_interpolation(mut self, i: CurveInterpolation) -> Self {
        self.interpolation = i;
        self
    }

    pub fn with_red(mut self, c: Curve) -> Self {
        self.red = Some(c);
        self
    }

    pub fn with_green(mut self, c: Curve) -> Self {
        self.green = Some(c);
        self
    }

    pub fn with_blue(mut self, c: Curve) -> Self {
        self.blue = Some(c);
        self
    }

    fn luts(&self) -> ([u8; 256], [u8; 256], [u8; 256]) {
        let master = self.master.build_lut(self.interpolation);
        let r = self
            .red
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        let g = self
            .green
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        let b = self
            .blue
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        (r, g, b)
    }
}

impl ImageFilter for Curves {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (lr, lg, lb) = self.luts();
        match params.format {
            PixelFormat::Gray8 => {
                if input.planes.is_empty() {
                    return Err(Error::invalid("Curves: missing plane"));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let mut out = vec![0u8; w * h];
                let s = &input.planes[0];
                let master = self.master.build_lut(self.interpolation);
                for y in 0..h {
                    for x in 0..w {
                        out[y * w + x] = master[s.data[y * s.stride + x] as usize];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: w,
                        data: out,
                    }],
                })
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if matches!(params.format, PixelFormat::Rgb24) {
                    3
                } else {
                    4
                };
                if input.planes.is_empty() {
                    return Err(Error::invalid("Curves: missing plane"));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let stride = w * bpp;
                let mut out = vec![0u8; stride * h];
                let s = &input.planes[0];
                for y in 0..h {
                    let src = &s.data[y * s.stride..y * s.stride + stride];
                    let dst = &mut out[y * stride..(y + 1) * stride];
                    for x in 0..w {
                        let b = x * bpp;
                        dst[b] = lr[src[b] as usize];
                        dst[b + 1] = lg[src[b + 1] as usize];
                        dst[b + 2] = lb[src[b + 2] as usize];
                        if bpp == 4 {
                            dst[b + 3] = src[b + 3];
                        }
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data: out }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                // Master curve on luma only — chroma planes pass-through.
                let master = self.master.build_lut(self.interpolation);
                let w = params.width as usize;
                let h = params.height as usize;
                if input.planes.len() < 3 {
                    return Err(Error::invalid("Curves YUV: expected 3 planes"));
                }
                let s = &input.planes[0];
                let mut y_out = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        y_out[y * w + x] = master[s.data[y * s.stride + x] as usize];
                    }
                }
                let planes = vec![
                    VideoPlane {
                        stride: w,
                        data: y_out,
                    },
                    input.planes[1].clone(),
                    input.planes[2].clone(),
                ];
                Ok(VideoFrame {
                    pts: input.pts,
                    planes,
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Curves does not yet handle {other:?}"
            ))),
        }
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
    fn identity_is_identity_linear() {
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Curves::new(Curve::identity())
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn identity_is_identity_cubic() {
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Curves::new(Curve::identity())
            .with_interpolation(CurveInterpolation::MonotoneCubic)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Cubic on a 2-point straight line should still be identity
        // (the Hermite tangents collapse to the secant slope).
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1, "got {a} expected {b}");
        }
    }

    #[test]
    fn s_curve_brightens_midtones() {
        // Classic S-curve: (0,0), (64,32), (128,160), (192,224), (255,255).
        let curve = Curve::new([(0, 0), (64, 32), (128, 160), (192, 224), (255, 255)]);
        let input = rgb(4, 4, |_, _| [128, 128, 128]);
        let out = Curves::new(curve)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Midtone 128 should map to 160.
        assert_eq!(out.planes[0].data[0], 160);
    }

    #[test]
    fn monotone_cubic_does_not_overshoot() {
        // Construct a curve with steep rises but request monotone cubic.
        // The Fritsch-Carlson tangents must prevent overshoot below
        // segment endpoints.
        let curve = Curve::new([(0, 0), (10, 0), (245, 255), (255, 255)]);
        let curves = Curves::new(curve).with_interpolation(CurveInterpolation::MonotoneCubic);
        // Apply on a synthetic ramp and verify monotone-non-decreasing
        // output.
        let input = rgb(8, 1, |x, _| {
            let v = (x * 32).min(255) as u8;
            [v, v, v]
        });
        let out = curves.apply(&input, p_rgb(8, 1)).unwrap();
        let mut prev = 0u8;
        for x in 0..8 {
            let v = out.planes[0].data[x * 3];
            assert!(v >= prev, "non-monotone at x={x}: {prev} → {v}");
            prev = v;
        }
    }

    #[test]
    fn per_channel_curve_overrides_master() {
        let master = Curve::new([(0, 0), (255, 255)]); // identity
        let red = Curve::new([(0, 0), (128, 64), (255, 128)]); // halve red
        let input = rgb(2, 2, |_, _| [200, 200, 200]);
        let out = Curves::new(master)
            .with_red(red)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let pix = &out.planes[0].data[0..3];
        assert!(pix[0] < 128, "red should be halved, got {}", pix[0]);
        assert_eq!(pix[1], 200);
        assert_eq!(pix[2], 200);
    }

    #[test]
    fn alpha_pass_through_rgba() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 100, 100, 88]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let curve = Curve::new([(0, 0), (255, 200)]);
        let out = Curves::new(curve)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 88);
        }
    }

    #[test]
    fn identity_is_identity_natural() {
        // 2-point identity (only the endpoints (0,0) and (255,255)):
        // with no interior knots the natural spline reduces to a
        // straight line so we should recover the input exactly.
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Curves::new(Curve::identity())
            .with_interpolation(CurveInterpolation::NaturalCubic)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn natural_passes_through_control_points() {
        // §4 of the curve-interpolation reference: a natural cubic
        // spline is an *interpolant*, so it must pass through every
        // control point. We verify P(x_i) = y_i (±1 LSB after the
        // f32→u8 round) at four interior knots.
        let pts = [(0u8, 0u8), (64, 32), (128, 200), (192, 96), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::NaturalCubic);
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "natural at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn natural_is_c2_smooth() {
        // Continuity of the second derivative at interior knots is the
        // defining property of the natural spline (§4 boundary
        // condition is `M_0 = M_{n−1} = 0`; interior `M_i` come out of
        // the tridiagonal solve and live in *both* adjacent segments
        // by construction). Approximate `P''` at integer x as the
        // 3-tap central second-difference `lut[x+1] − 2·lut[x] +
        // lut[x−1]` (which is a finite-difference proxy for `P''` to
        // O(h²) on a smooth curve). The discontinuity in `P''` that
        // a non-`C²` interpolant (Linear / CatmullRom / MonotoneCubic)
        // exhibits at a knot manifests as a sample-to-sample jump on
        // that proxy that is large vs. the smooth-segment baseline.
        // Here we just check that the proxy does not spike across the
        // (interior) knot at x=128.
        let curve = Curve::new([(0, 0), (64, 32), (128, 200), (192, 96), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::NaturalCubic);
        // Second-difference at x just left of the knot vs. just right.
        let left = lut[127] as i32 - 2 * lut[126] as i32 + lut[125] as i32;
        let right = lut[130] as i32 - 2 * lut[129] as i32 + lut[128] as i32;
        // On a `C²` curve these are the same continuous quantity sampled
        // 5 pixels apart — they can drift a small amount but should not
        // disagree by the swing-of-sign jump that a `C¹`-only spline
        // exhibits. Empirical bound `|left − right| ≤ 8` on this
        // fixture; a non-`C²` interpolant easily exceeds 20 here.
        assert!(
            (left - right).abs() <= 8,
            "natural-cubic `C²` continuity proxy: left={left} right={right} delta={}",
            (left - right).abs(),
        );
    }

    #[test]
    fn natural_endpoint_second_derivative_is_zero() {
        // §4.2 imposes M_0 = M_{n−1} = 0 (the "natural" boundary).
        // The second-difference proxy at the very first interior sample
        // should therefore be small in magnitude (the curve approaches
        // the endpoint with zero second derivative; finite-difference
        // noise stays single-digit).
        let curve = Curve::new([(0, 0), (64, 100), (192, 200), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::NaturalCubic);
        let near_start = lut[2] as i32 - 2 * lut[1] as i32 + lut[0] as i32;
        let near_end = lut[255] as i32 - 2 * lut[254] as i32 + lut[253] as i32;
        // u8-quantised proxy — accept a few-LSB rounding band around 0.
        assert!(
            near_start.abs() <= 3,
            "natural left endpoint second-difference proxy = {near_start}, expected ≈ 0",
        );
        assert!(
            near_end.abs() <= 3,
            "natural right endpoint second-difference proxy = {near_end}, expected ≈ 0",
        );
    }

    #[test]
    fn natural_two_points_is_straight_line() {
        // No interior knots ⇒ the natural-cubic tridiagonal system has
        // zero rows ⇒ M ≡ 0 ⇒ the evaluator collapses to the linear
        // Hermite identity `y_0 + Δ·t`. Verify against the linear path
        // sample-by-sample.
        let curve = Curve::new([(0, 0), (255, 200)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let nat = curve.build_lut(CurveInterpolation::NaturalCubic);
        for (i, (a, b)) in lin.iter().zip(nat.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "natural ≠ linear on 2-point ramp at x={i}: linear={a} natural={b}",
            );
        }
    }

    #[test]
    fn natural_clamps_to_byte_range() {
        // A pathological control set (sharp valley then sharp peak)
        // would overshoot below 0 or above 255 under any non-monotone
        // cubic; the LUT must clamp to `[0, 255]`. (Behavioural check —
        // the spline is *allowed* to overshoot since this is the trade-
        // off vs. MonotoneCubic; we only verify the byte-quantised
        // output stays in range.)
        let curve = Curve::new([(0, 0), (50, 5), (60, 250), (200, 5), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::NaturalCubic);
        for (i, &v) in lut.iter().enumerate() {
            // u8 cannot represent out-of-range, so the bound is implicit;
            // the test is that no panic / nan-cast happened. Re-read v
            // through an arithmetic op to silence any unused-binding
            // worry.
            let _ = v as u16 + i as u16;
        }
    }

    #[test]
    fn yuv_luma_only() {
        let y_data = vec![100u8; 16];
        let u_data = vec![64u8; 4];
        let v_data = vec![192u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: y_data,
                },
                VideoPlane {
                    stride: 2,
                    data: u_data.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: v_data.clone(),
                },
            ],
        };
        let out = Curves::new(Curve::new([(0, 0), (255, 128)]))
            .with_interpolation(CurveInterpolation::Linear)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Luma 100 → curve(100) = 100 * 128 / 255 ≈ 50.
        assert!((out.planes[0].data[0] as i32 - 50).abs() <= 1);
        assert_eq!(out.planes[1].data, u_data);
        assert_eq!(out.planes[2].data, v_data);
    }
}
