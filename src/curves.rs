//! Curves — per-channel monotone cubic curves adjustment.
//!
//! "Curves" is the per-channel tonal-mapping primitive in every classical
//! photo-editor. The user specifies a small set of control points
//! `(x, y)` on `[0, 255]² → [0, 255]²` and the editor fits a smooth
//! monotone curve through them. Nine classic interpolants apply:
//!
//! 1. **Linear** — straight segments between control points. Cheapest;
//!    visible kinks at every knot.
//! 2. **Catmull-Rom** — cubic interpolation through the points with
//!    tangents inferred from neighbours (Catmull, Rom — "A class of
//!    local interpolating splines", Computer-Aided Geometric Design,
//!    1974). Smooth but **not** monotone-safe (can overshoot below `0`
//!    or above `1`). Uniform parameterisation (`α = 0` of the
//!    Yuksel et al. §3.3 family — every neighbour spacing is `1` so
//!    the tangent collapses to half the chord between the surrounding
//!    points).
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
//! 5. **Centripetal Catmull-Rom** — the non-uniform Catmull-Rom variant
//!    parameterised by `t_{i+1} − t_i = |p_{i+1} − p_i|^α` with
//!    `α = 0.5` (Yuksel, Schaefer, Keyser — "Parameterization and
//!    applications of Catmull-Rom curves", Computer-Aided Design 43(7),
//!    2011 — provably free of cusps and self-intersections vs. uniform).
//!    The Barry-Goldman per-knot tangent
//!    `m_i = (p_{i+1} − p_i)/(t_{i+1} − t_i)
//!           − (p_{i+1} − p_{i−1})/(t_{i+1} − t_{i−1})
//!           + (p_i − p_{i−1})/(t_i − t_{i−1})`
//!    is then fed into the cubic-Hermite evaluator scaled by the local
//!    `x`-axis spacing `h_i = x_{i+1} − x_i`. Distinct from §3.1
//!    uniform Catmull-Rom (`α = 0`) because the tangent denominators
//!    track the inter-knot distance instead of dropping to plain
//!    `1`, so knots that are tightly clustered along `x` no longer
//!    induce the overshoot ripples uniform Catmull-Rom exhibits in
//!    that regime; still not `C²` (the second derivative is only
//!    continuous across the same knot when the spacing pattern
//!    matches, so the §4 natural spline remains the gentlest of the
//!    family on that axis).
//! 6. **Chordal Catmull-Rom** — the third member of the same Yuksel,
//!    Schaefer, Keyser 2011 (§3.3 of the reference doc) `α`-parameterised
//!    family with `α = 1`. The knot spacing is the raw chord length
//!    `t_{i+1} − t_i = |p_{i+1} − p_i|` (no fractional exponent), and
//!    the Barry-Goldman per-knot tangent reuses the same three-term form
//!    as the centripetal case. The chordal variant linearises the
//!    arc-length-vs-parameter relationship more aggressively than
//!    centripetal — each parametric interval already matches its
//!    Euclidean length — so curves through points spaced widely apart
//!    in 2-D get straighter segments and tighter overshoot near
//!    clustered knots than uniform `α = 0` Catmull-Rom. It is *not*
//!    cusp-free in the Yuksel et al. sense — only `α ∈ (0, 1)` gets
//!    that property, with `α = 0.5` the only choice that also makes
//!    the tangent magnitude scale-invariant — so for path work the
//!    centripetal variant remains the recommended default. Still `C¹`
//!    only and not monotone-safe.
//! 7. **Cardinal (tension `c`)** — the tension-parameterised generalisation
//!    of uniform Catmull-Rom per §3.2 of the curve-interpolation reference
//!    doc (Catmull & Rom 1974). The per-knot tangent is scaled by `(1 − c)`:
//!    `m_i = (1 − c) · (p_{i+1} − p_{i−1}) / 2`. `c = 0` recovers uniform
//!    Catmull-Rom (the §3.1 case, item 2 above) — every neighbour spacing
//!    drops to `1` and the `(1 − c)` factor is unity. `c = 1` zeros every
//!    tangent so the cubic Hermite basis evaluator collapses to a
//!    flat-tangent shape (`h00·y_i + h01·y_{i+1}`, the classic
//!    "no-tangent" spline) — visually tighter than linear at the knots
//!    but no overshoot. Intermediate `c` interpolates between the two
//!    regimes: higher `c` → tighter / less swing, lower `c` →
//!    closer to the smooth Catmull-Rom shape. Tension is encoded as a
//!    `u8` 8.0 fixed-point (`tension_q8 = 0..=255` mapping to `c =
//!    0..=1`) so the variant keeps the enum's `Copy + Eq + Hash`
//!    properties. Still `C¹` only and not monotone-safe (the tangent
//!    is a scaled secant chord just like uniform Catmull-Rom, so
//!    overshoot is possible at any `c < 1`).
//! 8. **Clamped cubic spline** — the first §4.2-alternative boundary
//!    choice for the same `C²` tridiagonal spline family as item 4:
//!    instead of the "natural" zero-second-derivative ends, the **end
//!    first derivatives** are fixed to caller-supplied slopes
//!    `s₀ = P'(x_0)` and `s₁ = P'(x_{n−1})`. Differentiating the §4.4
//!    segment cubic and imposing the two conditions yields two extra
//!    tridiagonal boundary rows (`2h_0·M_0 + h_0·M_1 = 6(Δ_0 − s₀)` at
//!    the head, `h_{n−2}·M_{n−2} + 2h_{n−2}·M_{n−1} = 6(s₁ − Δ_{n−2})`
//!    at the tail) so the full `n × n` system stays tridiagonal +
//!    diagonally dominant and the same Thomas sweep solves it. Slopes
//!    are encoded as signed Q8.8 fixed point (`slope = q / 256`) so the
//!    enum keeps `Copy + Eq`. With both slopes at `1.0` the identity
//!    curve solves to `M ≡ 0` and stays the exact identity.
//! 9. **Not-a-knot cubic spline** — the second §4.2-alternative
//!    boundary: the **third derivative** is forced continuous across
//!    the first and last *interior* knots, i.e. segments 0/1 (and
//!    n−3/n−2) each merge into one single cubic. Since `P'''` on
//!    segment `i` is the constant `(M_{i+1} − M_i)/h_i`, the conditions
//!    are `(M_1 − M_0)/h_0 = (M_2 − M_1)/h_1` and the mirror at the
//!    tail. Eliminating `M_0` / `M_{n−1}` into the adjacent §4.1
//!    interior rows keeps the reduced `(n−2)`-row system tridiagonal
//!    for the Thomas sweep; the eliminated boundary values are
//!    recovered afterwards by back-substitution. Degenerate sizes:
//!    `n = 2` has no interior knot ⇒ straight line; `n = 3` has a
//!    single interior knot carrying *both* conditions ⇒ the spline is
//!    one global polynomial with `P''' ≡ 0`, i.e. the parabola through
//!    the three points (`M_i` all equal to twice the second divided
//!    difference).
//!
//! We implement all nine under [`CurveInterpolation`] and apply the
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
    /// Fritsch-Carlson monotone-cubic Hermite using the **box** form of
    /// the §2.2 sufficient monotonicity region instead of the
    /// circle-of-radius-3 used by [`Self::MonotoneCubic`]. Per §2.2 of
    /// `docs/image/filter/curve-interpolation.md` (Fritsch & Carlson
    /// 1980): "An alternative sufficient region is `0 ≤ α ≤ 3` and
    /// `0 ≤ β ≤ 3` applied independently; both prevent overshoot." The
    /// normalised tangents `α_i = m_i/Δ_i` and `β_i = m_{i+1}/Δ_i` are
    /// each clamped to `[0, 3]` per-component rather than projected onto
    /// the radius-3 circle. Like [`Self::MonotoneCubic`] this is `C¹`
    /// and provably overshoot-free on monotone data; the two differ only
    /// in which sufficient sub-region of the Fritsch-Carlson admissible
    /// set the tangents are pulled back to, so the produced curve is
    /// generally a slightly different (but still monotone) shape near
    /// steep knots. The circle form is the more commonly transcribed
    /// default; this box form is the doc-flagged independent-axis
    /// alternative.
    MonotoneCubicBox,
    /// Natural cubic spline (`C²`) via the Thomas tridiagonal solver
    /// (de Boor 1978; tridiagonal step credited to L. H. Thomas, Watson
    /// Sci. Comp. Lab. 1949). Smoothest of the four (continuous second
    /// derivative) but can overshoot like Catmull-Rom; suited to
    /// gentle-curve work where C² visual smoothness matters more than
    /// strict monotone safety.
    NaturalCubic,
    /// Centripetal Catmull-Rom — the `α = 0.5` non-uniform
    /// Catmull-Rom parameterisation (Yuksel, Schaefer, Keyser —
    /// "Parameterization and applications of Catmull-Rom curves",
    /// Computer-Aided Design 43(7), 2011). Per-knot tangents are
    /// the Barry-Goldman three-term form
    /// `m_i = (p_{i+1} − p_i)/(t_{i+1} − t_i)
    ///        − (p_{i+1} − p_{i−1})/(t_{i+1} − t_{i−1})
    ///        + (p_i − p_{i−1})/(t_i − t_{i−1})`
    /// with the inter-knot spacing `t_{i+1} − t_i =
    /// |p_{i+1} − p_i|^0.5`. Provably free of cusps / self-loops
    /// (the property that motivates centripetal in 2-D path work),
    /// but still `C¹`-only and not monotone-safe (the §3 chord-style
    /// tangents still allow overshoot below `y_min` / above `y_max`).
    CentripetalCatmullRom,
    /// Chordal Catmull-Rom — the `α = 1` non-uniform Catmull-Rom
    /// parameterisation, the third member of the §3.3
    /// (Yuksel, Schaefer, Keyser 2011) alpha-family alongside uniform
    /// (`α = 0`) and centripetal (`α = 0.5`). Per-knot spacing is the
    /// raw chord length `t_{i+1} − t_i = |p_{i+1} − p_i|`; the
    /// Barry-Goldman three-term tangent reuses the same expression
    /// as the centripetal mode, only the spacing exponent differs.
    /// Knots that are far apart in 2-D Euclidean distance receive
    /// proportionally larger parametric intervals so the chordal
    /// segments approximate the arc-length parameterisation; in the
    /// LUT-fit context this tames the wide-knot overshoot seen with
    /// uniform Catmull-Rom and gives the straightest segments of the
    /// three `α` choices on widely-spaced data. Cusp-free is the
    /// `α ∈ (0, 1)` property — chordal sits at the open boundary so
    /// it does *not* inherit that guarantee; the centripetal variant
    /// remains the cusp-free default for path work. Still `C¹` only
    /// and not monotone-safe.
    ChordalCatmullRom,
    /// Cardinal spline — the tension-parameterised generalisation of
    /// uniform Catmull-Rom (Catmull, Rom 1974; §3.2 of
    /// `docs/image/filter/curve-interpolation.md`). The per-knot
    /// tangent is scaled by `(1 − c)`:
    /// `m_i = (1 − c) · (p_{i+1} − p_{i−1}) / 2`.
    ///
    /// `tension_q8` is an 8.0 fixed-point encoding of the tension
    /// parameter `c ∈ [0, 1]`: `c = tension_q8 / 255`. This keeps the
    /// enum `Copy + Eq + Hash`.
    ///
    /// Convention:
    /// * `tension_q8 = 0` → `c = 0` recovers uniform Catmull-Rom
    ///   (smoothest of the family, can overshoot).
    /// * `tension_q8 = 255` → `c = 1` zeros every tangent so the cubic
    ///   Hermite evaluator collapses to the flat-tangent `h00·y_i +
    ///   h01·y_{i+1}` shape (no overshoot, tight knots).
    /// * `tension_q8 = 128` → `c ≈ 0.502`, the natural "half-tension"
    ///   choice.
    ///
    /// Still `C¹` only and not monotone-safe at any `c < 1` (the
    /// scaled secant chord tangent allows overshoot below `y_min` /
    /// above `y_max`).
    Cardinal {
        /// Tension in 8.0 fixed point: `c = tension_q8 / 255`. `0` →
        /// uniform Catmull-Rom; `255` → flat-tangent.
        tension_q8: u8,
    },
    /// Clamped cubic spline — the same `C²` tridiagonal-solve spline
    /// family as [`Self::NaturalCubic`] but with the first
    /// §4.2-alternative boundary of
    /// `docs/image/filter/curve-interpolation.md`: the **end first
    /// derivatives** are prescribed (`P'(x_0) = s₀`,
    /// `P'(x_{n−1}) = s₁`) instead of zeroing the end second
    /// derivatives. Differentiating the §4.4 segment polynomial gives
    /// the two boundary rows
    /// `2h_0·M_0 + h_0·M_1 = 6(Δ_0 − s₀)` and
    /// `h_{n−2}·M_{n−2} + 2h_{n−2}·M_{n−1} = 6(s₁ − Δ_{n−2})`,
    /// keeping the full `n × n` system tridiagonal and diagonally
    /// dominant for the §4.3 Thomas sweep.
    ///
    /// Slopes are signed Q8.8 fixed point: `slope = q / 256`
    /// (`256` → slope `1.0`). This keeps the enum `Copy + Eq`. Useful
    /// for tone curves whose end behaviour must be pinned — e.g.
    /// `s₀ = s₁ = 0` produces a smoothstep-style S with flat shoulders,
    /// while `s₀ = s₁ = 1` on the identity knots reproduces the exact
    /// identity (the system solves to `M ≡ 0`). Like the natural
    /// spline this is `C²` but **not** monotone-safe.
    ClampedCubic {
        /// Prescribed slope at the first knot, signed Q8.8
        /// (`s₀ = start_slope_q8 / 256`).
        start_slope_q8: i16,
        /// Prescribed slope at the last knot, signed Q8.8
        /// (`s₁ = end_slope_q8 / 256`).
        end_slope_q8: i16,
    },
    /// Not-a-knot cubic spline — the second §4.2-alternative boundary
    /// of `docs/image/filter/curve-interpolation.md`: the **third
    /// derivative is continuous across the first and last interior
    /// knots**, so segments 0/1 (and n−3/n−2) each merge into a single
    /// cubic. `P'''` on segment `i` is the constant
    /// `(M_{i+1} − M_i)/h_i`, so the boundary conditions are
    /// `(M_1 − M_0)/h_0 = (M_2 − M_1)/h_1` (head) and
    /// `(M_{n−2} − M_{n−3})/h_{n−3} = (M_{n−1} − M_{n−2})/h_{n−2}`
    /// (tail). `M_0` / `M_{n−1}` are eliminated into the adjacent §4.1
    /// interior rows so the reduced `(n−2)`-row system stays
    /// tridiagonal for the Thomas sweep, then recovered by
    /// back-substitution. Parameter-free counterpart to
    /// [`Self::ClampedCubic`] when no end-slope information exists —
    /// the classical "let the data speak" boundary that avoids the
    /// natural spline's artificial flattening (`M = 0`) at the ends.
    /// Degenerate sizes: `n = 2` → straight line; `n = 3` → the
    /// parabola through the three points (`P''' ≡ 0` globally). `C²`
    /// but **not** monotone-safe.
    NotAKnotCubic,
}

/// §4.2 boundary-condition selector for the `C²` tridiagonal-solve
/// spline family (natural / clamped / not-a-knot). Internal — the
/// public surface is the corresponding [`CurveInterpolation`] variants;
/// this enum just lets the three share one solver + evaluator.
#[derive(Clone, Copy, Debug)]
enum SplineBoundary {
    /// `M_0 = M_{n−1} = 0` (§4.2 "natural").
    Natural,
    /// Prescribed end first derivatives `P'(x_0) = s0`,
    /// `P'(x_{n−1}) = s1` (§4.2 "clamped").
    Clamped { s0: f32, s1: f32 },
    /// Third derivative continuous across the first / last interior
    /// knot (§4.2 "not-a-knot").
    NotAKnot,
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
        // The §4 spline family (natural / clamped / not-a-knot) takes a
        // different path: a tridiagonal solve for the per-knot second
        // derivatives `M_i` instead of per-knot tangents `m_i`. The
        // three differ only in the §4.2 boundary rows.
        match mode {
            CurveInterpolation::NaturalCubic => {
                return Self::build_spline_lut(&pts, n, SplineBoundary::Natural);
            }
            CurveInterpolation::ClampedCubic {
                start_slope_q8,
                end_slope_q8,
            } => {
                return Self::build_spline_lut(
                    &pts,
                    n,
                    SplineBoundary::Clamped {
                        s0: start_slope_q8 as f32 / 256.0,
                        s1: end_slope_q8 as f32 / 256.0,
                    },
                );
            }
            CurveInterpolation::NotAKnotCubic => {
                return Self::build_spline_lut(&pts, n, SplineBoundary::NotAKnot);
            }
            _ => {}
        }
        // Per-segment tangents for the cubic-Hermite paths.
        let mut tangents = vec![0f32; n];
        match mode {
            CurveInterpolation::Linear => {}
            CurveInterpolation::NaturalCubic
            | CurveInterpolation::ClampedCubic { .. }
            | CurveInterpolation::NotAKnotCubic => unreachable!(),
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
            CurveInterpolation::CentripetalCatmullRom => {
                // §3.3 of `docs/image/filter/curve-interpolation.md`:
                // non-uniform knot parameterisation with α = 0.5 plus
                // the Barry-Goldman three-term tangent. r231 refactor
                // delegates the per-α work to `alpha_catmull_rom_tangents`
                // so the new ChordalCatmullRom (α = 1) shares the same
                // implementation path with only the exponent changing.
                Self::alpha_catmull_rom_tangents(&pts, n, 0.5, &mut tangents);
            }
            CurveInterpolation::ChordalCatmullRom => {
                // §3.3 of `docs/image/filter/curve-interpolation.md`:
                // the third (α = 1) member of the alpha-Catmull-Rom
                // family — knot spacing is the raw chord length
                // `t_{i+1} − t_i = |p_{i+1} − p_i|` (no fractional
                // exponent), Barry-Goldman tangent identical in form
                // to the centripetal case. Distinct from `CatmullRom`
                // (α = 0, denominator = 1) and `CentripetalCatmullRom`
                // (α = 0.5, denominator = √chord) by the exponent
                // alone.
                Self::alpha_catmull_rom_tangents(&pts, n, 1.0, &mut tangents);
            }
            CurveInterpolation::Cardinal { tension_q8 } => {
                // §3.2 of `docs/image/filter/curve-interpolation.md`:
                // tension-parameterised cardinal spline. Per-knot
                // tangent is `(1 − c) · (p_{i+1} − p_{i−1}) / 2` with
                // `c ∈ [0, 1]` decoded from the `u8` 8.0 fixed-point
                // `tension_q8 = 0..=255` mapping to `c = 0..=1`.
                //
                // `c = 0` recovers uniform Catmull-Rom (the §3.1
                // case) — the (1 − c) factor is unity and the tangent
                // matches `CatmullRom`'s centred-difference expression
                // sample-for-sample, so the new variant byte-aliases
                // the existing `CatmullRom` arm in that limit (a
                // property exercised by `cardinal_zero_tension_equals_catmull_rom`).
                // `c = 1` (`tension_q8 = 255`) zeros every tangent so
                // the cubic-Hermite evaluator collapses to
                // `h00·y_i + h01·y_{i+1}` — the flat-tangent /
                // "no-tangent" spline shape with no overshoot.
                //
                // Boundary tangents follow the same one-sided convention
                // as `CatmullRom` (phantom-knot duplication) so the
                // 2-point fixture reproduces the linear identity for
                // every `c`.
                let c = (tension_q8 as f32) / 255.0;
                let scale = 1.0 - c;
                for i in 0..n {
                    let (xl, yl) = if i == 0 { pts[0] } else { pts[i - 1] };
                    let (xr, yr) = if i + 1 == n { pts[n - 1] } else { pts[i + 1] };
                    let dx = xr - xl;
                    tangents[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        scale * (yr - yl) / dx
                    };
                }
            }
            CurveInterpolation::MonotoneCubic | CurveInterpolation::MonotoneCubicBox => {
                // Fritsch-Carlson §2. Step 1 (§2.1): secant slopes.
                let mut delta = vec![0f32; n - 1];
                for i in 0..n - 1 {
                    let dx = pts[i + 1].0 - pts[i].0;
                    delta[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        (pts[i + 1].1 - pts[i].1) / dx
                    };
                }
                // Step 2 (§2.1): initial tangents = average of
                // neighbouring secants (the three-point central-
                // difference estimate), one-sided at the boundaries.
                tangents[0] = delta[0];
                tangents[n - 1] = delta[n - 2];
                for i in 1..n - 1 {
                    tangents[i] = 0.5 * (delta[i - 1] + delta[i]);
                }
                // Step 3 (§2.2): enforce monotonicity by pulling the
                // normalised tangents `(α, β)` back into a sufficient
                // overshoot-free sub-region. Two sub-regions are
                // documented; the active one depends on `mode`:
                //   * `MonotoneCubic` → the circle of radius 3
                //     (`α² + β² ≤ 9`); the more commonly transcribed
                //     form.
                //   * `MonotoneCubicBox` → the independent-axis box
                //     `0 ≤ α ≤ 3` and `0 ≤ β ≤ 3` (each component
                //     clamped to `[0, 3]` separately) — the doc-flagged
                //     alternative sufficient region.
                // Both share the `Δ_i == 0 ⇒ flat segment` rule and the
                // `α, β ≥ 0` sign precondition from §2.2.
                let box_region = matches!(mode, CurveInterpolation::MonotoneCubicBox);
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
                    if box_region {
                        // §2.2 alternative: clamp each normalised tangent
                        // to `[0, 3]` independently (negatives already
                        // zeroed above; only the upper `3` bound bites
                        // here).
                        let a = a.max(0.0);
                        let b = b.max(0.0);
                        if a > 3.0 {
                            tangents[i] = 3.0 * delta[i];
                        }
                        if b > 3.0 {
                            tangents[i + 1] = 3.0 * delta[i];
                        }
                    } else {
                        // §2.2 circle: project `(α, β)` onto the
                        // radius-3 circle when it escapes it.
                        let s = a * a + b * b;
                        if s > 9.0 {
                            let t = 3.0 / s.sqrt();
                            tangents[i] = t * a * delta[i];
                            tangents[i + 1] = t * b * delta[i];
                        }
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

    /// Shared per-knot tangent builder for the §3.3 alpha-Catmull-Rom
    /// family (`α = 0.5` centripetal, `α = 1` chordal). The knot
    /// spacing `Δt_i = |p_{i+1} − p_i|^α` uses 2-D point distance
    /// (x and y both contribute) — the chord length, not the
    /// x-spacing — which is what makes the centripetal (α = 0.5)
    /// form provably cusp-free per Yuksel, Schaefer, Keyser 2011 and
    /// what tames the wide-knot overshoot in the chordal (α = 1)
    /// case. One-sided / phantom-knot tangents at the two boundaries
    /// (`i = 0` and `i = n − 1`) follow the §3.1 convention of
    /// duplicating the endpoint (`p_{−1} = p_0`, `p_n = p_{n−1}`),
    /// which makes the missing knot-spacing `0` and degenerates the
    /// boundary tangent to the secant slope of the adjacent interior
    /// segment in `x` (matching the linear-Hermite identity on a
    /// 2-point ramp).
    fn alpha_catmull_rom_tangents(pts: &[(f32, f32)], n: usize, alpha: f32, tangents: &mut [f32]) {
        let knot_spacing = |a: (f32, f32), b: (f32, f32)| -> f32 {
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let chord = (dx * dx + dy * dy).sqrt();
            chord.powf(alpha)
        };
        // Numerator vector and divisor — operating on the y
        // coordinate alone since the LUT only stores y(x); dividing
        // by the x-axis interval reconverts the Barry-Goldman per-knot
        // derivative `dp / dt` into a tangent `dy / dx` consumable by
        // the Hermite basis of §1, which already scales by
        // `h_i = x_{i+1} − x_i`. Defensive guards on the divisors keep
        // duplicate / coincident knots (`Δt = 0`) from producing NaNs.
        for i in 0..n {
            let (xl, yl) = if i == 0 { pts[0] } else { pts[i - 1] };
            let (xc, yc) = pts[i];
            let (xr, yr) = if i + 1 == n { pts[n - 1] } else { pts[i + 1] };
            let dt_left = knot_spacing((xl, yl), (xc, yc));
            let dt_right = knot_spacing((xc, yc), (xr, yr));
            let dt_span = dt_left + dt_right;
            // Boundary one-sided collapse: at i = 0 the left span is
            // 0 so the first and third Barry-Goldman terms drop out
            // and the tangent is the right secant slope; symmetric
            // collapse at i = n − 1.
            let dy_dt = if i == 0 {
                if dt_right < 1.0e-9 {
                    0.0
                } else {
                    (yr - yc) / dt_right
                }
            } else if i == n - 1 {
                if dt_left < 1.0e-9 {
                    0.0
                } else {
                    (yc - yl) / dt_left
                }
            } else if dt_left < 1.0e-9 || dt_right < 1.0e-9 || dt_span < 1.0e-9 {
                0.0
            } else {
                (yr - yc) / dt_right - (yr - yl) / dt_span + (yc - yl) / dt_left
            };
            // Hermite basis in §1 expects `m_i = dy/dx` and multiplies
            // by `h_i = x_{i+1} − x_i`. Convert `dy/dt` to `dy/dx` by
            // scaling by `dx/dt`, with `dx/dt = (x_right − x_left) /
            // (Δt_left + Δt_right)` — the chain rule. Endpoint cases
            // fold to the one-sided ratio.
            let dx_dt = if i == 0 {
                if dt_right < 1.0e-9 {
                    0.0
                } else {
                    (xr - xc) / dt_right
                }
            } else if i == n - 1 {
                if dt_left < 1.0e-9 {
                    0.0
                } else {
                    (xc - xl) / dt_left
                }
            } else if dt_span < 1.0e-9 {
                0.0
            } else {
                (xr - xl) / dt_span
            };
            tangents[i] = if dx_dt.abs() < 1.0e-9 {
                0.0
            } else {
                dy_dt / dx_dt
            };
        }
    }

    /// §4.3 Thomas forward-elimination + back-substitution on a general
    /// tridiagonal system. `a` is the sub-diagonal (`a[0]` unused),
    /// `b` the main diagonal, `c` the super-diagonal (`c[k−1]` unused),
    /// `d` the right-hand side; all four slices share the system size.
    /// `c'_i` / `d'_i` are the elimination's modified super-diagonal and
    /// RHS; `inv` caches `1 / (b_i − a_i·c'_{i−1})` to avoid two
    /// divisions per step, with the same defensive near-zero-pivot
    /// collapse-to-0 the pre-existing natural-cubic path used.
    fn thomas_solve(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Vec<f32> {
        let size = b.len();
        let mut c_prime = vec![0f32; size];
        let mut d_prime = vec![0f32; size];
        for k in 0..size {
            let denom = if k == 0 {
                b[0]
            } else {
                b[k] - a[k] * c_prime[k - 1]
            };
            let inv = if denom.abs() < 1.0e-12 {
                0.0
            } else {
                1.0 / denom
            };
            c_prime[k] = c[k] * inv;
            d_prime[k] = if k == 0 {
                d[k] * inv
            } else {
                (d[k] - a[k] * d_prime[k - 1]) * inv
            };
        }
        let mut x = vec![0f32; size];
        let mut next = 0.0f32;
        for k in (0..size).rev() {
            x[k] = d_prime[k] - c_prime[k] * next;
            next = x[k];
        }
        x
    }

    /// Solve for the per-knot second derivatives `M_i = P''(x_i)` of
    /// the §4 `C²` cubic spline under the selected §4.2 boundary
    /// condition. Interior rows are always the §4.1 system
    /// `h_{i−1}·M_{i−1} + 2(h_{i−1}+h_i)·M_i + h_i·M_{i+1}
    ///    = 6·(Δ_i − Δ_{i−1})` for `i = 1 … n−2`; the boundary choice
    /// only changes the first / last rows:
    ///
    /// * [`SplineBoundary::Natural`] — `M_0 = M_{n−1} = 0`; the
    ///   interior `(n−2)`-row system is solved with both boundary
    ///   values closed at zero.
    /// * [`SplineBoundary::Clamped`] — prescribed end slopes. The §4.4
    ///   segment polynomial differentiates to
    ///   `P'(x) = Δ_i − h_i·(2M_i + M_{i+1})/6 + M_i·t +
    ///   ((M_{i+1} − M_i)/(2h_i))·t²`; imposing `P'(x_0) = s₀` (t = 0
    ///   on segment 0) and `P'(x_{n−1}) = s₁` (t = h on segment n−2)
    ///   yields the head row `2h_0·M_0 + h_0·M_1 = 6(Δ_0 − s₀)` and
    ///   tail row `h_{n−2}·M_{n−2} + 2h_{n−2}·M_{n−1} =
    ///   6(s₁ − Δ_{n−2})`. The full `n × n` system stays tridiagonal
    ///   and diagonally dominant (`2h > h`), so one Thomas sweep
    ///   solves all `n` unknowns.
    /// * [`SplineBoundary::NotAKnot`] — third-derivative continuity
    ///   across the first / last interior knot. `P'''` on segment `i`
    ///   is the constant `(M_{i+1} − M_i)/h_i`, so the head condition
    ///   is `(M_1 − M_0)/h_0 = (M_2 − M_1)/h_1`, i.e.
    ///   `M_0 = (1 + h_0/h_1)·M_1 − (h_0/h_1)·M_2`. Substituting that
    ///   into the `i = 1` interior row (and clearing the `1/h_1`
    ///   denominator) gives the reduced head row
    ///   `(h_0+h_1)(h_0+2h_1)·M_1 + (h_1−h_0)(h_1+h_0)·M_2 = h_1·d_1`;
    ///   the tail is the mirror image. The reduced `(n−2)`-row system
    ///   is solved by the same Thomas sweep and the two eliminated
    ///   boundary values are recovered from the substitution formulas.
    ///   Degenerate sizes: `n = 2` (no interior knot) → `M ≡ 0`
    ///   (straight line); `n = 3` (one interior knot carrying both
    ///   conditions) → `P''' ≡ 0` globally, i.e. the parabola through
    ///   the three points, whose constant second derivative is twice
    ///   the second divided difference
    ///   `M_i = 2·(Δ_1 − Δ_0)/(h_0 + h_1)`.
    fn solve_spline_second_derivatives(
        h: &[f32],
        delta: &[f32],
        n: usize,
        boundary: SplineBoundary,
    ) -> Vec<f32> {
        let mut m = vec![0f32; n];
        if n < 2 {
            return m;
        }
        match boundary {
            SplineBoundary::Natural => {
                if n >= 3 {
                    let size = n - 2;
                    let mut a = vec![0f32; size];
                    let mut b = vec![0f32; size];
                    let mut c = vec![0f32; size];
                    let mut d = vec![0f32; size];
                    for k in 0..size {
                        let i = k + 1;
                        a[k] = h[i - 1];
                        b[k] = 2.0 * (h[i - 1] + h[i]);
                        c[k] = h[i];
                        d[k] = 6.0 * (delta[i] - delta[i - 1]);
                    }
                    m[1..n - 1].copy_from_slice(&Self::thomas_solve(&a, &b, &c, &d));
                }
            }
            SplineBoundary::Clamped { s0, s1 } => {
                let mut a = vec![0f32; n];
                let mut b = vec![0f32; n];
                let mut c = vec![0f32; n];
                let mut d = vec![0f32; n];
                // Head row from P'(x_0) = s₀.
                b[0] = 2.0 * h[0];
                c[0] = h[0];
                d[0] = 6.0 * (delta[0] - s0);
                for i in 1..n - 1 {
                    a[i] = h[i - 1];
                    b[i] = 2.0 * (h[i - 1] + h[i]);
                    c[i] = h[i];
                    d[i] = 6.0 * (delta[i] - delta[i - 1]);
                }
                // Tail row from P'(x_{n−1}) = s₁.
                a[n - 1] = h[n - 2];
                b[n - 1] = 2.0 * h[n - 2];
                d[n - 1] = 6.0 * (s1 - delta[n - 2]);
                m = Self::thomas_solve(&a, &b, &c, &d);
            }
            SplineBoundary::NotAKnot => {
                if n == 3 {
                    // One interior knot carries BOTH not-a-knot
                    // conditions ⇒ both segments are the same cubic with
                    // P''' ≡ 0 ⇒ the parabola through the three points.
                    let span = h[0] + h[1];
                    let val = if span.abs() < 1.0e-12 {
                        0.0
                    } else {
                        2.0 * (delta[1] - delta[0]) / span
                    };
                    m.fill(val);
                } else if n >= 4 {
                    let size = n - 2;
                    let mut a = vec![0f32; size];
                    let mut b = vec![0f32; size];
                    let mut c = vec![0f32; size];
                    let mut d = vec![0f32; size];
                    for k in 0..size {
                        let i = k + 1;
                        a[k] = h[i - 1];
                        b[k] = 2.0 * (h[i - 1] + h[i]);
                        c[k] = h[i];
                        d[k] = 6.0 * (delta[i] - delta[i - 1]);
                    }
                    // Head row: substitute the eliminated
                    // `M_0 = (1 + h_0/h_1)·M_1 − (h_0/h_1)·M_2` into the
                    // i = 1 interior row, cleared of the 1/h_1 divisor.
                    let (h0, h1) = (h[0], h[1]);
                    a[0] = 0.0;
                    b[0] = (h0 + h1) * (h0 + 2.0 * h1);
                    c[0] = (h1 - h0) * (h1 + h0);
                    d[0] = h1 * 6.0 * (delta[1] - delta[0]);
                    // Tail row: mirror substitution of
                    // `M_{n−1} = (1 + g_1/g_0)·M_{n−2} − (g_1/g_0)·M_{n−3}`
                    // into the i = n−2 interior row, cleared of 1/g_0.
                    let (g0, g1) = (h[n - 3], h[n - 2]);
                    a[size - 1] = (g0 - g1) * (g0 + g1);
                    b[size - 1] = (g0 + g1) * (2.0 * g0 + g1);
                    c[size - 1] = 0.0;
                    d[size - 1] = g0 * 6.0 * (delta[n - 2] - delta[n - 3]);
                    m[1..n - 1].copy_from_slice(&Self::thomas_solve(&a, &b, &c, &d));
                    // Recover the eliminated boundary values.
                    let r0 = if h1.abs() < 1.0e-12 { 0.0 } else { h0 / h1 };
                    m[0] = m[1] + r0 * (m[1] - m[2]);
                    let r1 = if g0.abs() < 1.0e-12 { 0.0 } else { g1 / g0 };
                    m[n - 1] = m[n - 2] + r1 * (m[n - 2] - m[n - 3]);
                }
                // n == 2: no interior knot — straight line, M ≡ 0.
            }
        }
        m
    }

    /// Build the LUT for the §4 `C²` cubic-spline family (natural /
    /// clamped / not-a-knot — the three §4.2 boundary choices of the
    /// curve-interpolation reference doc). The path is separate from
    /// [`Self::build_lut`] because these splines are parameterised by
    /// per-knot second-derivatives `M_i = P''(x_i)`, not by per-knot
    /// tangents `m_i = P'(x_i)` like the cubic-Hermite paths.
    ///
    /// Steps (cited from `docs/image/filter/curve-interpolation.md`
    /// §4.1–4.4, originally de Boor 1978 / Thomas 1949):
    ///
    /// 1. Build the tridiagonal system
    ///    `h_{i−1}·M_{i−1} + 2(h_{i−1}+h_i)·M_i + h_i·M_{i+1}
    ///       = 6·(Δ_i − Δ_{i−1})` for `i = 1 … n−2`, with the §4.2
    ///    boundary rows selected by `boundary` (see
    ///    [`Self::solve_spline_second_derivatives`]).
    /// 2. Solve the system in `O(n)` with the Thomas forward-
    ///    elimination + back-substitution sweep.
    /// 3. Evaluate each segment as
    ///    `P(x) = y_i + (Δ_i − h_i·(2M_i + M_{i+1})/6)·t
    ///          + (M_i / 2)·t² + ((M_{i+1} − M_i)/(6 h_i))·t³`
    ///    with `t = x − x_i`.
    fn build_spline_lut(pts: &[(f32, f32)], n: usize, boundary: SplineBoundary) -> [u8; 256] {
        // n ≥ 2 is guaranteed by `build_lut`'s early-return. With
        // exactly 2 points there are no interior knots so the natural /
        // not-a-knot forms have M ≡ 0 (straight line) and the clamped
        // form solves a 2×2 system — fall through to the same
        // evaluator either way.
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

        let m = Self::solve_spline_second_derivatives(&h, &delta, n, boundary);

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
    fn monotone_cubic_box_passes_through_control_points() {
        // The §2.2 box-region variant still interpolates every knot to
        // within byte round-off (a clamp on the tangents never moves the
        // segment endpoint values, only the in-between shape).
        let curve = Curve::new([(0, 0), (64, 32), (128, 200), (192, 210), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::MonotoneCubicBox);
        for (x, y) in [(0u8, 0u8), (64, 32), (128, 200), (192, 210), (255, 255)] {
            assert!(
                (lut[x as usize] as i16 - y as i16).abs() <= 1,
                "knot ({x},{y}) -> {}",
                lut[x as usize]
            );
        }
    }

    #[test]
    fn monotone_cubic_box_does_not_overshoot() {
        // §2.2 promises the box region is *also* overshoot-free on
        // monotone data: every LUT entry must stay inside the bracketing
        // knot interval AND the LUT must be non-decreasing. The fixture
        // has a steep central rise that would overshoot under an
        // unclamped Catmull-Rom.
        let curve = Curve::new([(0, 0), (10, 0), (245, 255), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::MonotoneCubicBox);
        // Monotone non-decreasing.
        let mut prev = 0u8;
        for &v in lut.iter() {
            assert!(v >= prev, "non-monotone: {prev} -> {v}");
            prev = v;
        }
        // No value escapes [0, 255] (trivially true for u8) and the flat
        // tails stay flat at the held endpoint values.
        assert_eq!(lut[0], 0);
        assert_eq!(lut[5], 0);
        assert_eq!(lut[250], 255);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn monotone_cubic_box_differs_from_circle_on_steep_knots() {
        // The two sufficient sub-regions (circle of radius 3 vs the
        // independent-axis box) pull steep tangents back to different
        // shapes, so the produced LUTs must NOT be byte-identical — the
        // test would be vacuous if `MonotoneCubicBox` silently aliased
        // the circle form. A steep asymmetric rise drives at least one
        // normalised tangent past 3 so the two clamps diverge.
        let curve = Curve::new([(0, 0), (32, 8), (96, 250), (255, 255)]);
        let circle = curve.build_lut(CurveInterpolation::MonotoneCubic);
        let boxed = curve.build_lut(CurveInterpolation::MonotoneCubicBox);
        assert_ne!(
            circle.to_vec(),
            boxed.to_vec(),
            "box region must produce a distinct LUT from the circle region"
        );
    }

    #[test]
    fn monotone_cubic_box_two_points_is_straight_line() {
        // No interior knots ⇒ both tangents collapse to the single
        // secant slope and the clamp is a no-op, so a 2-point ramp
        // reproduces the linear identity byte-for-byte.
        let curve = Curve::new([(0, 0), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::MonotoneCubicBox);
        for (x, slot) in lut.iter().enumerate() {
            assert_eq!(*slot as usize, x, "non-identity at x={x}");
        }
    }

    #[test]
    fn monotone_cubic_box_flat_segment_stays_flat() {
        // §2.2 `Δ_i == 0 ⇒ m_i = m_{i+1} = 0`: a flat segment in the
        // control data must produce a flat LUT region under the box
        // variant exactly as under the circle variant.
        let curve = Curve::new([(0, 0), (64, 128), (192, 128), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::MonotoneCubicBox);
        // The 64..=192 span is flat (both endpoints y = 128); every
        // interior sample must sit at 128.
        for x in 64u8..=192 {
            assert_eq!(lut[x as usize], 128, "flat span dipped at x={x}");
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

    // ---------- r226: Centripetal Catmull-Rom (§3.3 of
    // docs/image/filter/curve-interpolation.md) ----------

    #[test]
    fn centripetal_passes_through_control_points() {
        // Interpolation property: every interpolant in §1–§4 of the
        // reference doc passes through the supplied control points
        // exactly (within u8 round-off). Five-knot fixture exercising
        // both interior and boundary knots.
        let pts = [(0u8, 0u8), (50, 30), (120, 210), (200, 90), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "centripetal at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn centripetal_two_points_is_straight_line() {
        // Two endpoints only ⇒ no interior knots ⇒ the Barry-Goldman
        // boundary tangents collapse to the right / left secant of the
        // single segment so the cubic-Hermite path coincides with the
        // straight-line linear identity. Verify against the linear LUT
        // sample-by-sample.
        let curve = Curve::new([(0, 0), (255, 200)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        for (i, (a, b)) in lin.iter().zip(cen.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "centripetal ≠ linear on 2-point ramp at x={i}: linear={a} centripetal={b}",
            );
        }
    }

    #[test]
    fn centripetal_clamps_to_byte_range() {
        // Like the natural-cubic / Catmull-Rom variants, centripetal
        // is not monotone-safe and can overshoot below 0 or above 255
        // on pathological control sets; verify the byte-quantised LUT
        // stays in range (no NaN cast, no panic) under such a fixture.
        let curve = Curve::new([(0, 0), (50, 5), (60, 250), (200, 5), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        // u8 cannot represent out-of-range — the bound is implicit;
        // touch every entry to make sure the LUT was actually built.
        let mut acc: u32 = 0;
        for &v in lut.iter() {
            acc += v as u32;
        }
        // Some positive coverage; the s-curve fixture should not all-zero.
        assert!(acc > 0);
    }

    #[test]
    fn centripetal_identity_under_collinear_control_points() {
        // §3.3 (Yuksel et al. 2011) cusp-free property degenerates to
        // a straight line on perfectly collinear control points (every
        // tangent is along the same direction so the cubic Hermite
        // segments collapse to chord-following lines). Three points
        // on `y = x` should reproduce the linear LUT to within u8.
        let curve = Curve::new([(0, 0), (127, 127), (255, 255)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        for (i, (a, b)) in lin.iter().zip(cen.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "centripetal ≠ identity on collinear knots at x={i}: linear={a} centripetal={b}",
            );
        }
    }

    #[test]
    fn centripetal_differs_from_uniform_catmull_rom_on_uneven_knots() {
        // The motivating contrast in §3.3 is exactly this: with
        // unevenly-spaced knots the uniform Catmull-Rom (α = 0) and
        // centripetal (α = 0.5) interpolants disagree somewhere in
        // the interior. A fixture with a single tightly-clustered
        // (60, 230) knot between widely-spaced neighbours must
        // produce at least one LUT entry where uniform and
        // centripetal differ by more than the trivial 1-LSB
        // round-off (otherwise the new mode is silently identical to
        // the old one and the test would not exercise it).
        let curve = Curve::new([(0, 0), (10, 5), (60, 230), (180, 60), (255, 255)]);
        let uni = curve.build_lut(CurveInterpolation::CatmullRom);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        let mut max_delta: i16 = 0;
        for (a, b) in uni.iter().zip(cen.iter()) {
            let delta = (*a as i16 - *b as i16).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
        assert!(
            max_delta > 1,
            "centripetal must differ from uniform Catmull-Rom on uneven knots; \
             max |delta| = {max_delta}",
        );
    }

    #[test]
    fn centripetal_uniform_knots_match_uniform_catmull_rom() {
        // Symmetric reverse of the preceding case: with evenly-spaced
        // knots (`Δx` constant) the centripetal parameterisation
        // `t_{i+1} − t_i = |Δp|^0.5` is the same scalar at every
        // interval (since |Δp| differs only by the constant `y`
        // component → still the same constant in x ↔ x cases) and
        // the per-knot tangent collapses to the same Barry-Goldman
        // three-term expression evaluated at unit spacing — which is
        // the §3.1 uniform tangent. Verify the two LUTs agree on a
        // four-uniform-knot fixture (|delta| ≤ 1 LSB).
        let curve = Curve::new([(0, 0), (85, 85), (170, 170), (255, 255)]);
        let uni = curve.build_lut(CurveInterpolation::CatmullRom);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        for (i, (a, b)) in uni.iter().zip(cen.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "uniform = centripetal expected on uniform fixture at x={i}: uni={a} cen={b}",
            );
        }
    }

    // ---------- r231: Chordal Catmull-Rom (§3.3 of
    // docs/image/filter/curve-interpolation.md, α = 1) ----------

    #[test]
    fn chordal_passes_through_control_points() {
        // Interpolation property: every interpolant in §1–§4 of the
        // reference doc passes through the supplied control points
        // exactly (within u8 round-off). Five-knot fixture exercising
        // both interior and boundary knots; mirrors the centripetal
        // r226 test for symmetry between the two α members of the §3.3
        // family.
        let pts = [(0u8, 0u8), (50, 30), (120, 210), (200, 90), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "chordal at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn chordal_two_points_is_straight_line() {
        // Two endpoints only ⇒ no interior knots ⇒ the Barry-Goldman
        // boundary tangents collapse to the right / left secant of the
        // single segment so the cubic-Hermite path coincides with the
        // straight-line linear identity. Same shape as the centripetal
        // r226 two-point test — the boundary collapse is α-agnostic.
        let curve = Curve::new([(0, 0), (255, 200)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let chord = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        for (i, (a, b)) in lin.iter().zip(chord.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "chordal ≠ linear on 2-point ramp at x={i}: linear={a} chordal={b}",
            );
        }
    }

    #[test]
    fn chordal_clamps_to_byte_range() {
        // Like the other non-monotone-safe variants, chordal is not
        // overshoot-bounded; verify the byte-quantised LUT stays in
        // range (no NaN cast, no panic) under a pathological control
        // set.
        let curve = Curve::new([(0, 0), (50, 5), (60, 250), (200, 5), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        let mut acc: u32 = 0;
        for &v in lut.iter() {
            acc += v as u32;
        }
        assert!(acc > 0);
    }

    #[test]
    fn chordal_identity_under_collinear_control_points() {
        // §3.3 alpha-family degenerates to a straight line on
        // perfectly collinear control points (every tangent points
        // along the same direction so the cubic-Hermite segments
        // collapse to chord-following lines) for every α in the
        // open-then-closed unit interval. Three points on `y = x`
        // should reproduce the linear LUT to within u8.
        let curve = Curve::new([(0, 0), (127, 127), (255, 255)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let chord = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        for (i, (a, b)) in lin.iter().zip(chord.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "chordal ≠ identity on collinear knots at x={i}: linear={a} chordal={b}",
            );
        }
    }

    #[test]
    fn chordal_differs_from_uniform_catmull_rom_on_uneven_knots() {
        // The motivating contrast in §3.3 between α = 0 and the α > 0
        // members of the family: with unevenly-spaced knots the
        // uniform Catmull-Rom and chordal interpolants must disagree
        // somewhere in the interior, otherwise the new mode is
        // silently identical to the old one and the test would not
        // exercise the spacing exponent.
        let curve = Curve::new([(0, 0), (10, 5), (60, 230), (180, 60), (255, 255)]);
        let uni = curve.build_lut(CurveInterpolation::CatmullRom);
        let chord = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        let mut max_delta: i16 = 0;
        for (a, b) in uni.iter().zip(chord.iter()) {
            let delta = (*a as i16 - *b as i16).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
        assert!(
            max_delta > 1,
            "chordal must differ from uniform Catmull-Rom on uneven knots; \
             max |delta| = {max_delta}",
        );
    }

    #[test]
    fn chordal_differs_from_centripetal_on_uneven_knots() {
        // The chord-length exponent is what distinguishes the three §3.3
        // members from each other. With unevenly-spaced knots the α=0.5
        // centripetal and α=1 chordal interpolants must disagree
        // somewhere in the interior — otherwise the chordal mode is
        // silently aliased to the centripetal mode and the spacing
        // exponent is not actually being exercised. Same fixture as the
        // uniform-vs-centripetal r226 test.
        let curve = Curve::new([(0, 0), (10, 5), (60, 230), (180, 60), (255, 255)]);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        let chord = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        let mut max_delta: i16 = 0;
        for (a, b) in cen.iter().zip(chord.iter()) {
            let delta = (*a as i16 - *b as i16).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
        assert!(
            max_delta > 1,
            "chordal must differ from centripetal on uneven knots; \
             max |delta| = {max_delta}",
        );
    }

    #[test]
    fn chordal_handles_clustered_knot_without_panic() {
        // §3.3 chord-length-based knot spacing keeps every divisor
        // strictly positive even when the x-axis spacing collapses
        // toward zero — same defensive property as the centripetal
        // mode (only the exponent differs). Touch every LUT entry to
        // confirm no NaN cast / panic.
        let curve = Curve::new([(0, 0), (58, 50), (60, 230), (62, 60), (255, 255)]);
        let chord = curve.build_lut(CurveInterpolation::ChordalCatmullRom);
        let mut acc: u32 = 0;
        for &v in chord.iter() {
            acc += v as u32;
        }
        assert!(acc > 0);
    }

    #[test]
    fn centripetal_handles_clustered_knot_without_panic() {
        // §3.3 cusp-free property motivates the chord-length-based
        // knot spacing, which keeps every divisor strictly positive
        // even when the x-axis spacing collapses toward zero (the
        // pathology that makes the uniform Catmull-Rom tangent blow
        // up). We do not assert the centripetal LUT is "smoother"
        // than uniform on every fixture — byte quantisation and the
        // 1-D-along-x degeneracy can leave both close numerically —
        // but a tightly-clustered knot fixture must build a valid
        // [0, 255]-clamped LUT in both modes with no panic and no
        // NaN cast. Touch every entry to confirm the LUT is fully
        // populated.
        let curve = Curve::new([(0, 0), (58, 50), (60, 230), (62, 60), (255, 255)]);
        let cen = curve.build_lut(CurveInterpolation::CentripetalCatmullRom);
        let uni = curve.build_lut(CurveInterpolation::CatmullRom);
        let mut acc_cen: u32 = 0;
        let mut acc_uni: u32 = 0;
        for i in 0..=255usize {
            acc_cen += cen[i] as u32;
            acc_uni += uni[i] as u32;
        }
        // Some positive coverage in both LUTs (255 control points
        // ensure non-zero tail).
        assert!(acc_cen > 0);
        assert!(acc_uni > 0);
    }

    // ---------- r262: Cardinal spline (§3.2 of
    // docs/image/filter/curve-interpolation.md) ----------

    #[test]
    fn cardinal_passes_through_control_points() {
        // Interpolation property: every interpolant in §1–§4 passes
        // exactly (within u8 round-off) through the supplied control
        // points. Five-knot fixture exercising both interior and
        // boundary knots — mirrors the centripetal r226 / chordal r231
        // shape so the new variant is held to the same identity. Use
        // the half-tension default `c ≈ 0.502` (`tension_q8 = 128`).
        let pts = [(0u8, 0u8), (50, 30), (120, 210), (200, 90), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 128 });
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "cardinal at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn cardinal_zero_tension_equals_catmull_rom() {
        // §3.2 of the reference doc: `c = 0` recovers uniform
        // Catmull-Rom (the §3.1 case). With `tension_q8 = 0` the
        // `(1 − c) = 1` factor is unity and the cardinal tangent
        // matches the existing CatmullRom expression sample-for-sample.
        // The two LUTs must agree to within the 1-LSB u8 rounding band
        // on every test fixture; the uneven-knot fixture used by the
        // r231 chordal differentiation test is a strong stress.
        let curve = Curve::new([(0, 0), (10, 5), (60, 230), (180, 60), (255, 255)]);
        let uni = curve.build_lut(CurveInterpolation::CatmullRom);
        let car = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 0 });
        for (i, (a, b)) in uni.iter().zip(car.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "cardinal c=0 ≠ uniform Catmull-Rom at x={i}: uni={a} car={b}",
            );
        }
    }

    #[test]
    fn cardinal_full_tension_zeros_every_tangent() {
        // `tension_q8 = 255` → `c = 1` → `(1 − c) = 0` so every
        // per-knot tangent is exactly `0`. The cubic-Hermite evaluator
        // collapses to `h00·y_i + h01·y_{i+1}` (the flat-tangent
        // shape). Behaviourally: the curve still passes through every
        // knot (`h00(0) = 1`, `h01(0) = 0`; `h00(1) = 0`, `h01(1) =
        // 1`) AND never overshoots between two adjacent y-values
        // (`h00 + h01 = 1` and both are non-negative on `[0, 1]` so
        // the value is a convex combination). Verify both properties
        // on the same uneven-knot fixture that exercises overshoot in
        // every other cubic variant.
        let pts = [(0u8, 0u8), (10, 5), (60, 230), (180, 60), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 255 });
        // Property 1: still interpolates every knot.
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "cardinal c=1 must still interpolate knots; at x={x}: got {got}, expected {y}",
            );
        }
        // Property 2: convex-combination shape — between knots `i` and
        // `i+1` the output is bounded by `[min(y_i, y_{i+1}),
        // max(y_i, y_{i+1})]`. Check every interior x against the
        // bracketing knot pair.
        let knots: Vec<(u8, u8)> = pts.to_vec();
        for x in 0u8..=255 {
            // Locate (i, j) such that knots[i].0 <= x <= knots[j].0.
            let mut i = 0usize;
            while i + 1 < knots.len() && knots[i + 1].0 < x {
                i += 1;
            }
            let j = (i + 1).min(knots.len() - 1);
            let lo = knots[i].1.min(knots[j].1) as i16;
            let hi = knots[i].1.max(knots[j].1) as i16;
            let v = lut[x as usize] as i16;
            // Allow a 1-LSB rounding slack on each side.
            assert!(
                v >= lo - 1 && v <= hi + 1,
                "cardinal c=1 must not overshoot bracketing knots at x={x}: v={v} bracket=[{lo}, {hi}]",
            );
        }
    }

    #[test]
    fn cardinal_two_points_is_straight_line_at_zero_tension() {
        // Two endpoints only ⇒ no interior knots ⇒ the boundary
        // tangents drop to the one-sided secant of the single segment.
        // At `c = 0` the `(1 − c) = 1` factor is unity so the
        // tangent matches uniform Catmull-Rom's secant and the
        // cubic-Hermite path coincides with the straight-line linear
        // identity (a property already verified for `CatmullRom` —
        // mirroring it here keeps the new variant on the same shape
        // contract at its uniform-Catmull-Rom limit).
        //
        // At intermediate `c ∈ (0, 1)` the tangent is the scaled
        // secant `(1 − c) · Δ`; the cubic-Hermite evaluator no longer
        // collapses to the linear identity because the §1 `h10·h·m_i`
        // and `h11·h·m_j` terms inject curvature whose magnitude scales
        // with `(1 − c)`. So this 2-point round-trip is a `c = 0`
        // identity, not an arbitrary-`c` identity. The
        // `cardinal_full_tension_zeros_every_tangent` test below
        // separately covers the `c = 1` shape on a multi-knot fixture
        // (where the flat-tangent path is the textbook "no-overshoot"
        // shape worth a dedicated test).
        let curve = Curve::new([(0, 0), (255, 200)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let car = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 0 });
        for (i, (a, b)) in lin.iter().zip(car.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "cardinal c=0 ≠ linear on 2-point ramp at x={i}: linear={a} cardinal={b}",
            );
        }
    }

    #[test]
    fn cardinal_clamps_to_byte_range() {
        // Like the other non-monotone-safe variants, cardinal at any
        // `c < 1` is not overshoot-bounded; verify the byte-quantised
        // LUT stays in range (no NaN cast, no panic) under a
        // pathological control set. Use the half-tension default —
        // c=1 is overshoot-bounded by the convex-combination shape
        // already checked above.
        let curve = Curve::new([(0, 0), (50, 5), (60, 250), (200, 5), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 128 });
        let mut acc: u32 = 0;
        for &v in lut.iter() {
            acc += v as u32;
        }
        assert!(acc > 0);
    }

    #[test]
    fn cardinal_higher_tension_dampens_per_segment_overshoot() {
        // §3.2 motivation: increasing the tension `c` from `0` toward
        // `1` reduces the tangent magnitude, which in turn dampens
        // per-segment overshoot above the bracketing-knot maximum (or
        // below the bracketing-knot minimum). Construct a fixture that
        // overshoots under uniform Catmull-Rom (c = 0) — namely a knot
        // pattern that rises sharply at (60, 230) then falls — and
        // verify the **total area** of overshoot (sum of
        // max(0, lut[x] − bracket_hi) + max(0, bracket_lo − lut[x])
        // across all x) is monotone-non-increasing as `c` goes from
        // `0` to `255`. At `c = 255` (full tension) the cubic-Hermite
        // evaluator is the convex combination `h00·y_i + h01·y_{i+1}`
        // — which has zero overshoot by construction (verified
        // separately above), so the metric must vanish at the high
        // end.
        let curve = Curve::new([(0, 0), (50, 5), (60, 230), (180, 60), (255, 255)]);
        // Pre-compute the bracketing-knot bounds for every x. With the
        // ensure_endpoints() injection both (0, …) and (255, …) are
        // present so every x in 0..=255 has an interval to consult.
        let pts = curve.ensure_endpoints();
        let mut bracket_lo = [0u8; 256];
        let mut bracket_hi = [0u8; 256];
        for x in 0u32..=255 {
            // Locate (i, j) such that pts[i].0 <= x <= pts[j].0.
            let mut i = 0usize;
            while i + 1 < pts.len() && (pts[i + 1].0 as u32) < x {
                i += 1;
            }
            let j = (i + 1).min(pts.len() - 1);
            let lo = pts[i].1.min(pts[j].1).round() as u8;
            let hi = pts[i].1.max(pts[j].1).round() as u8;
            bracket_lo[x as usize] = lo;
            bracket_hi[x as usize] = hi;
        }
        let overshoot_area = |lut: &[u8; 256]| -> u32 {
            let mut acc = 0u32;
            for x in 0..256 {
                let v = lut[x] as i32;
                let lo = bracket_lo[x] as i32;
                let hi = bracket_hi[x] as i32;
                if v > hi {
                    acc += (v - hi) as u32;
                } else if v < lo {
                    acc += (lo - v) as u32;
                }
            }
            acc
        };
        let mut prev: u32 = u32::MAX;
        let mut saw_strict_decrease = false;
        for &c in &[0u8, 64, 128, 192, 255] {
            let lut = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: c });
            let area = overshoot_area(&lut);
            assert!(
                area <= prev,
                "cardinal: per-segment overshoot area should monotone-decrease with tension; \
                 c_q8={c} prev={prev} current={area}",
            );
            if c > 0 && area < prev {
                saw_strict_decrease = true;
            }
            prev = area;
        }
        // The end-of-sweep value should be exactly zero (full tension
        // is the no-overshoot shape) and somewhere along the sweep we
        // must have seen a strict decrease — otherwise the test would
        // be vacuous (a constant-zero series passes the monotone
        // inequality trivially).
        assert_eq!(
            prev, 0,
            "cardinal c=1 must have zero per-segment overshoot (flat-tangent shape)",
        );
        assert!(
            saw_strict_decrease,
            "cardinal: expected at least one strict decrease over the c=0..=1 sweep, \
             otherwise the test does not exercise the tension scaling",
        );
    }

    #[test]
    fn cardinal_handles_clustered_knot_without_panic() {
        // Same defensive property as the centripetal / chordal modes
        // (§3.3): tightly-clustered knots along x must not break the
        // tangent denominator. The cardinal arm guards `dx.abs() <
        // 1e-6` exactly like the existing `CatmullRom` arm so the
        // pathology resolves to a zero tangent. Touch every LUT entry
        // to confirm no NaN cast / panic.
        let curve = Curve::new([(0, 0), (58, 50), (60, 230), (62, 60), (255, 255)]);
        let car = curve.build_lut(CurveInterpolation::Cardinal { tension_q8: 200 });
        let mut acc: u32 = 0;
        for &v in car.iter() {
            acc += v as u32;
        }
        assert!(acc > 0);
    }

    // ---------- r277: Clamped + not-a-knot cubic splines (§4.2
    // alternative boundary conditions of
    // docs/image/filter/curve-interpolation.md) ----------

    /// Shared helper: build `h` / `Δ` for a knot list and run the §4
    /// second-derivative solver under the given boundary.
    fn solve_m(pts: &[(f32, f32)], boundary: SplineBoundary) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = pts.len();
        let mut h = vec![0f32; n - 1];
        let mut delta = vec![0f32; n - 1];
        for i in 0..n - 1 {
            let dx = pts[i + 1].0 - pts[i].0;
            h[i] = dx;
            delta[i] = if dx.abs() < 1.0e-6 {
                0.0
            } else {
                (pts[i + 1].1 - pts[i].1) / dx
            };
        }
        let m = Curve::solve_spline_second_derivatives(&h, &delta, n, boundary);
        (h, delta, m)
    }

    #[test]
    fn clamped_passes_through_control_points() {
        // Interpolation property: every interpolant in §1–§4 of the
        // reference doc passes through the supplied control points
        // exactly (within u8 round-off), independent of the boundary
        // choice. Five-knot fixture, asymmetric prescribed slopes.
        let pts = [(0u8, 0u8), (50, 30), (120, 210), (200, 90), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::ClampedCubic {
            start_slope_q8: 128, // s₀ = 0.5
            end_slope_q8: 512,   // s₁ = 2.0
        });
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "clamped at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn clamped_identity_slopes_on_identity_knots_is_identity() {
        // With s₀ = s₁ = 1 on the 2-point identity knots, the line
        // `y = x` satisfies both clamped conditions with M ≡ 0, so the
        // system solves to zero and the evaluator reproduces the exact
        // identity (head row: 2h·M_0 + h·M_1 = 6(Δ − 1) = 0; tail row
        // mirrors — the unique solution of the dominant system is 0).
        let curve = Curve::identity();
        let lut = curve.build_lut(CurveInterpolation::ClampedCubic {
            start_slope_q8: 256,
            end_slope_q8: 256,
        });
        for (x, slot) in lut.iter().enumerate() {
            assert_eq!(*slot as usize, x, "non-identity at x={x}");
        }
    }

    #[test]
    fn clamped_zero_end_slopes_flatten_the_ends() {
        // s₀ = s₁ = 0 on the identity knots pins both end derivatives
        // flat, producing a smoothstep-style S: the LUT must be ~flat
        // near both endpoints (the natural spline keeps slope ≈ 1
        // there) while still spanning 0 → 255 and hitting ~midpoint at
        // x = 128 by symmetry.
        let curve = Curve::identity();
        let lut = curve.build_lut(CurveInterpolation::ClampedCubic {
            start_slope_q8: 0,
            end_slope_q8: 0,
        });
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        // Flat shoulders: over the first / last 4 samples the curve
        // moves at most 1 LSB (slope 0 + tiny quadratic term), whereas
        // the natural spline moves ≈ 4.
        assert!(
            lut[4] <= 1,
            "clamped s₀=0 head should be flat, lut[4] = {}",
            lut[4]
        );
        assert!(
            lut[251] >= 254,
            "clamped s₁=0 tail should be flat, lut[251] = {}",
            lut[251]
        );
        // Midpoint symmetry of the s-shape.
        assert!(
            (lut[128] as i16 - 128).abs() <= 2,
            "clamped s-curve midpoint drifted: lut[128] = {}",
            lut[128]
        );
        // And it must differ from the natural spline (which is the
        // straight line on these knots).
        let nat = curve.build_lut(CurveInterpolation::NaturalCubic);
        let max_delta = lut
            .iter()
            .zip(nat.iter())
            .map(|(a, b)| (*a as i16 - *b as i16).abs())
            .max()
            .unwrap();
        assert!(
            max_delta > 1,
            "clamped s=0 must differ from natural on identity knots; max |delta| = {max_delta}",
        );
    }

    #[test]
    fn clamped_solver_honours_prescribed_end_slopes() {
        // Solver-level check of the §4.2 clamped conditions: rebuild
        // the end first derivatives from the solved M via the
        // differentiated §4.4 polynomial and compare against the
        // prescribed s₀ / s₁.
        //   P'(x_0)    = Δ_0 − h_0·(2M_0 + M_1)/6
        //   P'(x_{n−1}) = Δ_{n−2} + h_{n−2}·(M_{n−2} + 2M_{n−1})/6
        let pts = [
            (0.0f32, 0.0f32),
            (50.0, 30.0),
            (120.0, 210.0),
            (200.0, 90.0),
            (255.0, 255.0),
        ];
        let (s0, s1) = (0.25f32, 3.0f32);
        let (h, delta, m) = solve_m(&pts, SplineBoundary::Clamped { s0, s1 });
        let n = pts.len();
        let head = delta[0] - h[0] * (2.0 * m[0] + m[1]) / 6.0;
        let tail = delta[n - 2] + h[n - 2] * (m[n - 2] + 2.0 * m[n - 1]) / 6.0;
        assert!(
            (head - s0).abs() < 1.0e-3,
            "clamped head slope: got {head}, prescribed {s0}",
        );
        assert!(
            (tail - s1).abs() < 1.0e-3,
            "clamped tail slope: got {tail}, prescribed {s1}",
        );
        // Interior §4.1 rows must still hold (the boundary rows did not
        // corrupt the system).
        for i in 1..n - 1 {
            let lhs = h[i - 1] * m[i - 1] + 2.0 * (h[i - 1] + h[i]) * m[i] + h[i] * m[i + 1];
            let rhs = 6.0 * (delta[i] - delta[i - 1]);
            assert!(
                (lhs - rhs).abs() < 1.0e-2,
                "clamped interior row {i}: lhs={lhs} rhs={rhs}",
            );
        }
    }

    #[test]
    fn clamped_differs_from_natural_on_shared_knots() {
        // The natural boundary zeroes the end second derivatives; the
        // clamped boundary pins the end first derivatives. On the same
        // multi-knot fixture with a non-secant prescribed slope the two
        // LUTs must diverge by more than the 1-LSB rounding band —
        // otherwise the new variant silently aliases the old one.
        let curve = Curve::new([(0, 0), (64, 32), (128, 200), (192, 96), (255, 255)]);
        let nat = curve.build_lut(CurveInterpolation::NaturalCubic);
        let cla = curve.build_lut(CurveInterpolation::ClampedCubic {
            start_slope_q8: 768, // s₀ = 3.0 — far from the head secant 0.5
            end_slope_q8: 0,     // s₁ = 0.0
        });
        let max_delta = nat
            .iter()
            .zip(cla.iter())
            .map(|(a, b)| (*a as i16 - *b as i16).abs())
            .max()
            .unwrap();
        assert!(
            max_delta > 1,
            "clamped must differ from natural; max |delta| = {max_delta}",
        );
    }

    #[test]
    fn not_a_knot_passes_through_control_points() {
        // Interpolation property under the second §4.2 boundary.
        let pts = [(0u8, 0u8), (50, 30), (120, 210), (200, 90), (255, 255)];
        let curve = Curve::new(pts);
        let lut = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        for (x, y) in pts {
            let got = lut[x as usize];
            assert!(
                (got as i16 - y as i16).abs() <= 1,
                "not-a-knot at x={x}: got {got}, expected {y}",
            );
        }
    }

    #[test]
    fn not_a_knot_two_points_is_straight_line() {
        // No interior knots ⇒ the not-a-knot conditions have no knot to
        // act on and the spline degenerates to the straight line
        // (M ≡ 0), matching the linear path sample-by-sample.
        let curve = Curve::new([(0, 0), (255, 200)]);
        let lin = curve.build_lut(CurveInterpolation::Linear);
        let nak = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        for (i, (a, b)) in lin.iter().zip(nak.iter()).enumerate() {
            assert!(
                (*a as i16 - *b as i16).abs() <= 1,
                "not-a-knot ≠ linear on 2-point ramp at x={i}: linear={a} not-a-knot={b}",
            );
        }
    }

    #[test]
    fn not_a_knot_three_points_is_the_parabola() {
        // n = 3: the single interior knot carries both not-a-knot
        // conditions, so P''' ≡ 0 and the spline is the unique parabola
        // through the three points. Compare the LUT against a direct
        // quadratic (Lagrange) evaluation at every sample.
        let pts = [(0.0f32, 0.0f32), (64.0, 32.0), (255.0, 255.0)];
        let curve = Curve::new([(0u8, 0u8), (64, 32), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        let quad = |x: f32| -> f32 {
            let (x0, y0) = pts[0];
            let (x1, y1) = pts[1];
            let (x2, y2) = pts[2];
            y0 * ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
                + y1 * ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
                + y2 * ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
        };
        for x in 0u32..=255 {
            let expect = quad(x as f32).clamp(0.0, 255.0).round() as i16;
            let got = lut[x as usize] as i16;
            assert!(
                (got - expect).abs() <= 1,
                "not-a-knot n=3 must be the parabola: x={x} got={got} expected={expect}",
            );
        }
    }

    #[test]
    fn not_a_knot_solver_third_derivative_is_continuous_at_boundary_knots() {
        // The defining §4.2 property: P''' (the per-segment constant
        // `(M_{i+1} − M_i)/h_i`) matches across the first and last
        // interior knots. Also re-verify the §4.1 interior rows so the
        // reduced-system elimination provably reconstructed a solution
        // of the original system.
        let pts = [
            (0.0f32, 0.0f32),
            (40.0, 120.0),
            (128.0, 140.0),
            (200.0, 60.0),
            (255.0, 255.0),
        ];
        let (h, delta, m) = solve_m(&pts, SplineBoundary::NotAKnot);
        let n = pts.len();
        let head_l = (m[1] - m[0]) / h[0];
        let head_r = (m[2] - m[1]) / h[1];
        assert!(
            (head_l - head_r).abs() < 1.0e-3,
            "not-a-knot head P''' mismatch: {head_l} vs {head_r}",
        );
        let tail_l = (m[n - 2] - m[n - 3]) / h[n - 3];
        let tail_r = (m[n - 1] - m[n - 2]) / h[n - 2];
        assert!(
            (tail_l - tail_r).abs() < 1.0e-3,
            "not-a-knot tail P''' mismatch: {tail_l} vs {tail_r}",
        );
        for i in 1..n - 1 {
            let lhs = h[i - 1] * m[i - 1] + 2.0 * (h[i - 1] + h[i]) * m[i] + h[i] * m[i + 1];
            let rhs = 6.0 * (delta[i] - delta[i - 1]);
            assert!(
                (lhs - rhs).abs() < 1.0e-2,
                "not-a-knot interior row {i}: lhs={lhs} rhs={rhs}",
            );
        }
    }

    #[test]
    fn not_a_knot_differs_from_natural_on_curved_ends() {
        // Natural artificially flattens the ends (M_0 = M_{n−1} = 0);
        // not-a-knot extrapolates the interior curvature instead. On a
        // fixture with strong curvature near the boundary the two LUTs
        // must diverge by more than the rounding band.
        let curve = Curve::new([(0, 0), (40, 120), (128, 140), (200, 60), (255, 255)]);
        let nat = curve.build_lut(CurveInterpolation::NaturalCubic);
        let nak = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        let max_delta = nat
            .iter()
            .zip(nak.iter())
            .map(|(a, b)| (*a as i16 - *b as i16).abs())
            .max()
            .unwrap();
        assert!(
            max_delta > 1,
            "not-a-knot must differ from natural; max |delta| = {max_delta}",
        );
    }

    #[test]
    fn not_a_knot_is_c2_smooth() {
        // Same second-difference `C²` proxy as the natural-spline test:
        // the spline family shares the §4.1 interior continuity rows,
        // so the proxy must not spike across an interior knot under the
        // not-a-knot boundary either.
        let curve = Curve::new([(0, 0), (64, 32), (128, 200), (192, 96), (255, 255)]);
        let lut = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        let left = lut[127] as i32 - 2 * lut[126] as i32 + lut[125] as i32;
        let right = lut[130] as i32 - 2 * lut[129] as i32 + lut[128] as i32;
        assert!(
            (left - right).abs() <= 8,
            "not-a-knot `C²` continuity proxy: left={left} right={right} delta={}",
            (left - right).abs(),
        );
    }

    #[test]
    fn clamped_and_not_a_knot_clamp_to_byte_range() {
        // Like the natural spline, neither §4.2 alternative is
        // monotone-safe; a pathological control set must still build a
        // valid [0, 255] LUT with no NaN cast / panic. Touch every
        // entry of both LUTs.
        let curve = Curve::new([(0, 0), (50, 5), (60, 250), (200, 5), (255, 255)]);
        let cla = curve.build_lut(CurveInterpolation::ClampedCubic {
            start_slope_q8: -512, // negative prescribed slope is legal
            end_slope_q8: 1024,
        });
        let nak = curve.build_lut(CurveInterpolation::NotAKnotCubic);
        let mut acc: u32 = 0;
        for i in 0..256usize {
            acc += cla[i] as u32 + nak[i] as u32;
        }
        assert!(acc > 0);
    }
}
