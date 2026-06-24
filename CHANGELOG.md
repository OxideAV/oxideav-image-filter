# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- r369: shared `edt_squared_2d` driver — the exact 2-D squared-Euclidean
  distance transform of `docs/image/filter/distance-transform.md` §2.4
  factored into one `pub(crate)` helper. `EuclideanDistanceTransform`'s
  `apply` now calls it instead of re-inlining the two separable passes;
  output is bit-identical (the brute-force oracle tests are unchanged).
  Unblocks the exact-Euclidean morphology filters below sharing one
  transcribed driver.

- r358: `ProximityFill` — exact nearest-seed value propagation (Voronoi
  region fill). Paints every pixel with the source intensity of its
  nearest feature site, built on the same exact `feature_transform_2d`
  as `VoronoiTransform`. The standard nearest-neighbour region-grow /
  sparse-inpainting primitive: extend a handful of known samples to
  fill the whole frame, exact over an arbitrary seed set (not the
  local-window approximation of `Crystallize`). No seeds → verbatim
  input copy. Knobs `threshold` / `invert`; Gray8-only in/out. Factory
  aliases: `proximity-fill`, `voronoi-fill`, `nearest-fill`. Clean-room
  from `docs/image/filter/distance-transform.md` §1 + §2.

- r358: `VoronoiTransform` — exact nearest-feature (Voronoi) transform.
  Labels every pixel with the coordinate of its closest foreground site,
  the `argmin` counterpart to the distance the
  `EuclideanDistanceTransform` already computes. The §2.3
  lower-envelope march of `docs/image/filter/distance-transform.md`
  already tracks `v[k]` (the sample whose parabola is lowest), so a new
  `dt_1d_arg` argmin variant of the 1-D driver carries that index
  through both separable passes, yielding the exact nearest feature in
  `O(d · N)` total time. The output renders a deterministic per-cell
  hash of the nearest seed so adjacent Voronoi cells get distinct, flat
  grey levels (cracked-glass mosaic). Knobs `threshold` / `invert`;
  Gray8-only in/out. Factory aliases: `voronoi`, `voronoi-transform`,
  `nearest-feature`. Clean-room from
  `docs/image/filter/distance-transform.md` §1 (generalised DT) + §2
  (Felzenszwalb–Huttenlocher exact-Euclidean transform).

- r352: drive the `Drago` adaptive-logarithmic tone mapper to full
  `docs/image/filter/tone-mapping-operators.md` §4 fidelity.
  - **§4.2 `Ld_max` parameterisation** — new `Drago::ld_max` field +
    `with_ld_max` builder exposing the doc's target maximum display
    luminance in cd/m². The §4.2 leading factor becomes
    `Ld_max · 0.01 / log10(Lwmax + 1)`; the params-table default `100`
    keeps `Ld_max · 0.01 == 1.0`, so existing default output is
    byte-identical while a caller can now dim/brighten by the cd/m²
    target independently of the auxiliary `display_scale` multiplier.
  - **§4.1 exposure-independent pre-scaling** — new `Drago::key` field +
    `with_key` (explicit log-average key) and `with_auto_key` (use the
    image's own §1.1 log-average luminance) builders. Dividing scene
    luminance by the log-average key before the curve makes the mapping
    invariant to a global exposure change (verified by a two-exposure
    mean-absolute-difference test). Chroma re-modulation divides by the
    **original** (un-prescaled) luminance per §1 step 3, so the key only
    reshapes the tone curve.
  - Registry factory `drago` / `tonemap-drago` now accepts
    `{"ld_max": <cd/m²>}`, `{"key": <v>}`, `{"auto_key": true}`, and the
    string alias `{"key": "auto"}`.
  - Drago test coverage `6 → 12`; one new registry factory test.

- r333: add the Bayer `16×16` ordered-dither matrix (`BayerMatrix::M16`) —
  one more turn of the `M_2n = 4·M_n + M_2[hi, lo]` recurrence past the
  existing `8×8` map (`docs/image/filter/dithering-kernels.md` §2). Its
  256-entry threshold map (a full permutation of `0..=255`) is the finest
  dispersed-dot ordered dither, showing the least cross-hatch texture on
  smooth gradients. Wired into the `dither-bayer` / `ordered-dither`
  factory via `{"matrix": 16}` and the `kernel` string aliases
  `bayer16` / `bayer-16` / `ordered16`.

- r329: add `WeightedDistanceTransform` — the **continuous-seed** form of
  the generalised distance transform `D_f(p) = min_q(‖p − q‖² + f(q))`
  named in `docs/image/filter/distance-transform.md` §1 ("the generalized
  form (arbitrary `f`) is what makes the algorithm reusable beyond binary
  masks"). Unlike `EuclideanDistanceTransform`, which seeds the
  Felzenszwalb–Huttenlocher driver with a binary mask (`f(q) ∈ {0, +∞}`),
  this filter seeds every pixel with a finite cost
  `f(q) = (weight · c(q))²` derived from its own intensity, so the output
  field is a smooth blend of squared-Euclidean travel distance and per-
  pixel "faintness" cost (cost-weighted feature maps, intensity-aware
  feathering, grey-weighted morphology). Reuses the exact `dt_1d`
  lower-envelope driver of `EuclideanDistanceTransform` (§2) so the whole
  filter stays `O(d · N)` for an exact generalised result; `weight = 0`
  yields an all-black field (every pixel a free seed) and a large `weight`
  recovers the binary transform. Knobs `weight` / `invert` / `scale`;
  Gray8 input only, Gray8 output of the same dimensions. Factory aliases
  `weighted-distance-transform`, `weighted-distance`, `wdt`. Verified
  pixel-exact against a brute-force §1 minimisation oracle.

- r324: add `SrgbTransform` — the documented display transfer function
  (`docs/image/filter/tone-mapping-operators.md` §5.2) exposed as a
  standalone 256-entry LUT primitive that converts between display-encoded
  and linear-light pixel values (so a pipeline can wrap linear-light
  operations — additive light, blur, alpha compositing — in a correct
  round-trip). `SrgbCurve::Srgb` (default) is the IEC 61966-2-1 piecewise
  curve: encode `V = 12.92·Lin` (`Lin ≤ 0.0031308`) /
  `1.055·Lin^(1/2.4) − 0.055`, decode the analytic inverse
  (`Lin = V/12.92` for `V ≤ 0.04045`, else `((V+0.055)/1.055)^2.4`).
  `SrgbCurve::Gamma { .. }` (built via `SrgbCurve::gamma(γ)`) is the §5.2
  pure power-law approximation `V = Lin^(1/γ)` / `Lin = V^γ`.
  `SrgbDirection::Encode` runs the OETF (linear → display);
  `SrgbDirection::Decode` (default) runs the EOTF (display → linear);
  encode∘decode of the same curve is identity within 8-bit rounding (deep
  shadows excepted — the EOTF compresses the bottom of the encoded range
  onto a handful of linear codes, inherent 8-bit shadow precision).
  Endpoints `0`/`255` are fixed points. Gray8 / Rgb24 / Rgba (alpha is
  coverage, not light — passed through unchanged); YUV → `Unsupported`.
  Cost is one LUT build + `O(W·H)`. Registry factories: `srgb-decode` /
  `linearize` / `srgb-to-linear` (decode), `srgb-encode` / `delinearize`
  / `linear-to-srgb` (encode), and `srgb-transform` (decode default;
  `direction` + `curve` + `gamma` JSON keys override). Distinct from the
  existing `Gamma` filter, which applies a display-gamma power curve in
  encoded space rather than the colorimetric sRGB↔linear transfer pair.
- r316: add the §1 step 3 **saturation exponent** `s` to the `Reinhard`
  tone-mapping operator per `docs/image/filter/tone-mapping-operators.md`
  §1 (Conventions, step 3). The doc's general re-colouring form is
  `C_out = (C_in / L_w)^s · L_d`; the prior operator hard-coded the
  `s = 1` special case `C_out = C_in · (L_d / L_w)` (plain
  chroma-preserving re-modulation). The new `saturation: f32` field
  (default `1.0`) + `with_saturation(s)` builder exposes the exponent:
  `0 ≤ s < 1` pulls colours toward neutral grey (chroma compresses along
  with luminance — useful when the tone curve would otherwise leave
  highlights looking unnaturally vivid), `s > 1` boosts chroma. The
  `s = 1` fast path keeps byte-identical output to the previous default.
  Non-finite / negative inputs to `with_saturation` are ignored; `s = 0`
  is accepted (fully desaturated). Registry factory `reinhard` /
  `tonemap-reinhard` accept the JSON key `saturation` (alias `s`).
- r303: add `Feather` — soft edge feathering driven by the exact
  Euclidean distance transform per
  `docs/image/filter/distance-transform.md` §1 (the generalised DT
  `D_f(p) = min_q(‖p − q‖² + f(q))` whose intro names "feathering" as a
  backed use-case) and §2 (Felzenszwalb–Huttenlocher 2012). The inner
  distance `d_in(p)` is the EDT of the inverted mask, and the per-pixel
  coverage ramp is `cov(p) = clamp(d_in(p) / radius, 0, 1)` with
  background fixed at `0`: a boundary pixel maps to `0`, a full `radius`
  inside reaches `1`, and the interior beyond the band saturates at `1`.
  Reuses the same `dt_1d` lower-envelope-of-parabolas driver as
  `EuclideanDistanceTransform` and `SignedDistanceField`, so the whole
  filter is `O(d · N)` for an exact distance field. `Rgba` (alpha is the
  coverage mask, RGB passes through unchanged, output alpha becomes the
  feathered coverage) or `Gray8` (luma is the mask, output is the
  single-plane coverage ramp); other formats return `Unsupported`.
  Knobs: `radius` (feather width in pixels, `0` = hard binary mask),
  `threshold` (foreground cut on the mask channel), `invert` (flip the
  foreground test). Verified pixel-exact against a brute-force inner
  Euclidean distance oracle. Registry factory aliases: `feather`,
  `feather-edge`, `soft-edge`.

- r297: add serpentine (boustrophedon) scan order to `Dither`'s
  error-diffusion path via a new `ScanOrder` enum and `Dither::with_scan`
  builder, per `docs/image/filter/dithering-kernels.md` §1.3. On
  `ScanOrder::Serpentine` even scanlines run left→right and odd scanlines
  run right→left with the stencil mirrored horizontally (every `dx`
  negated), so error still flows into not-yet-quantised neighbours; this
  breaks up the directional "worm"/hysteresis artifacts a pure raster
  scan accumulates, the artifact-suppression order Ulichney recommends.
  The kernel coefficients and divisors are unchanged, the quantiser is
  unchanged (output is still bit-exact 1-bit at `levels = 2`), and the
  conserving kernels still conserve the mean. `ScanOrder::Raster` is the
  default, so pre-existing behaviour is preserved byte-for-byte. Scan
  order has no effect on the feedback-free Bayer ordered-dither mode.
  The registry error-diffusion factories accept `{"scan": "serpentine"}`
  (or the `{"serpentine": true}` boolean shorthand).

- r288: add `SignedDistanceField` — an exact signed distance field
  built as the difference of two exact-Euclidean distance transforms
  per `docs/image/filter/distance-transform.md` §1 (the generalised DT
  `D_f(p) = min_q(‖p − q‖² + f(q))` whose intro names "SDF generation"
  as a use-case) and §2 (Felzenszwalb–Huttenlocher 2012). The field is
  `sdf(p) = d_out(p) − d_in(p)`, where `d_out` is the EDT of the mask
  and `d_in` the EDT of the inverted mask, so foreground pixels carry
  negative distance, background pixels positive, and the zero level-set
  traces the shape boundary. Renders as
  `clamp(midpoint + scale · sdf, 0, 255)` so an on-boundary pixel lands
  at the neutral `midpoint` (default `128`). Reuses the `dt_1d`
  lower-envelope driver of `EuclideanDistanceTransform` (now
  `pub(crate)`), keeping the whole filter `O(d · N)` for an exact
  result. Knobs: `threshold` / `invert` / `scale` / `midpoint`.
  Single-plane `Gray8` in / out. Factory aliases:
  `signed-distance-field`, `signed-distance`, `sdf`. Validated against
  a brute-force quadratic signed-field oracle on random / solid-block
  masks, plus sign-convention, midpoint-bias, scale, invert, and
  degenerate all-FG / all-BG saturation tests.
- r277: extend `CurveInterpolation` with the two §4.2-alternative
  boundary conditions of the `C²` tridiagonal cubic-spline family per
  `docs/image/filter/curve-interpolation.md` §4.2 (the natural spline's
  doc-flagged alternatives; de Boor 1978, Thomas 1949):
  `ClampedCubic { start_slope_q8, end_slope_q8 }` prescribes the **end
  first derivatives** — differentiating the §4.4 segment polynomial and
  imposing `P'(x_0) = s₀` / `P'(x_{n−1}) = s₁` yields the boundary rows
  `2h_0·M_0 + h_0·M_1 = 6(Δ_0 − s₀)` and `h_{n−2}·M_{n−2} +
  2h_{n−2}·M_{n−1} = 6(s₁ − Δ_{n−2})`, keeping the full `n × n` system
  tridiagonal + diagonally dominant for the §4.3 Thomas sweep (slopes
  are signed Q8.8 fixed point, `slope = q / 256`, so the enum stays
  `Copy + Eq`; identity knots with both slopes at `1.0` solve to
  `M ≡ 0`, the exact identity). `NotAKnotCubic` forces the **third
  derivative continuous** across the first / last interior knot —
  `P'''` on segment `i` is the constant `(M_{i+1} − M_i)/h_i`, so
  `M_0` / `M_{n−1}` are eliminated into the adjacent §4.1 interior rows
  (head row `(h_0+h_1)(h_0+2h_1)·M_1 + (h_1−h_0)(h_1+h_0)·M_2 =
  h_1·d_1`, tail mirrored), the reduced `(n−2)`-row system is solved by
  the same Thomas sweep and the eliminated boundary values are
  recovered by back-substitution; degenerate `n = 3` (one interior knot
  carrying both conditions) resolves to `P''' ≡ 0`, the parabola
  through the three points, and `n = 2` to the straight line. The
  natural-cubic solver + §4.4 evaluator were refactored into a shared
  `SplineBoundary`-parameterised path (`thomas_solve` +
  `solve_spline_second_derivatives` + `build_spline_lut`); the
  `NaturalCubic` output is unchanged. JSON factory parser accepts
  `"clamped"` / `"clamped-cubic"` / `"clamped_cubic"` /
  `"clampedcubic"` (slopes via `start_slope`/`slope0`/`s0` and
  `end_slope`/`slope1`/`s1`, default `1.0`, out-of-range clamped to
  `[-127, 127]`) and `"not-a-knot"` / `"not_a_knot"` / `"notaknot"` /
  `"not-a-knot-cubic"` / `"not_a_knot_cubic"`. Ten new unit tests
  (`clamped_passes_through_control_points`,
  `clamped_identity_slopes_on_identity_knots_is_identity`,
  `clamped_zero_end_slopes_flatten_the_ends`,
  `clamped_solver_honours_prescribed_end_slopes` — the solved `M`
  reconstruct the prescribed end slopes AND satisfy every §4.1 interior
  row, `clamped_differs_from_natural_on_shared_knots`,
  `not_a_knot_passes_through_control_points`,
  `not_a_knot_two_points_is_straight_line`,
  `not_a_knot_three_points_is_the_parabola` — LUT matches the direct
  Lagrange quadratic at all 256 samples,
  `not_a_knot_solver_third_derivative_is_continuous_at_boundary_knots`,
  `not_a_knot_differs_from_natural_on_curved_ends`,
  `not_a_knot_is_c2_smooth`, `clamped_and_not_a_knot_clamp_to_byte_range`)
  plus two JSON-factory alias tests in `src/registry.rs`.

- r270: extend `CurveInterpolation` with `MonotoneCubicBox` — the
  Fritsch-Carlson monotone-cubic interpolant using the **box** form of
  the §2.2 sufficient monotonicity region per
  `docs/image/filter/curve-interpolation.md` §2.2 (F. N. Fritsch,
  R. E. Carlson, *"Monotone Piecewise Cubic Interpolation,"* SIAM J.
  Numer. Anal. 17(2):238–246, 1980). The reference doc flags two
  documented sufficient overshoot-free sub-regions: the circle of
  radius 3 (`α² + β² ≤ 9`, used by the existing `MonotoneCubic`
  default) and the independent-axis box (`0 ≤ α ≤ 3` and `0 ≤ β ≤ 3`
  "applied independently"). The new variant shares every step of the
  existing Fritsch-Carlson tangent computation — §2.1 secant slopes,
  §2.1 three-point central-difference initial tangents, the §2.2
  `Δ_i == 0 ⇒ flat segment` rule and `α, β ≥ 0` sign precondition —
  and differs **only** in the final clamp: instead of projecting the
  normalised tangent pair `(α, β)` onto the radius-3 circle, each
  component is clamped to `[0, 3]` separately. Both forms are provably
  overshoot-free on monotone data; they produce slightly different
  (but still monotone) curve shapes near steep knots because they pull
  escaping tangents back to different boundaries of the Fritsch-Carlson
  admissible set. The plain `MonotoneCubic` default is unchanged, so
  existing callers and JSON jobs are unaffected. JSON factory parser
  accepts six new spellings for the mode parameter (`"monotone-box"`,
  `"monotone_box"`, `"monotonebox"`, `"monotone-cubic-box"`,
  `"monotone_cubic_box"`, `"monotonecubicbox"`); unknown spellings
  continue to fall back to `MonotoneCubic`. Five new unit tests in
  `src/curves.rs` (`monotone_cubic_box_passes_through_control_points` —
  every knot is hit to within byte round-off on a five-knot fixture,
  `monotone_cubic_box_does_not_overshoot` — the LUT is non-decreasing
  and the flat endpoint tails stay flat on a steep-rise fixture,
  `monotone_cubic_box_differs_from_circle_on_steep_knots` — the box
  region produces a distinct LUT from the circle region on a steep
  asymmetric rise so the variant is not a silent alias,
  `monotone_cubic_box_two_points_is_straight_line` — a 2-point ramp
  reproduces the linear identity byte-for-byte, and
  `monotone_cubic_box_flat_segment_stays_flat` — a flat control span
  produces a flat LUT region per the §2.2 `Δ_i == 0` rule). One new
  registry test `curves_factory_accepts_monotone_box_mode_aliases`
  walks all six mode-name spellings. Curve-interpolation count goes
  7 → 8. Backed by §2.2 of
  `docs/image/filter/curve-interpolation.md`, which itself cites the
  original Fritsch-Carlson publication by author/year + DOI link only,
  per the *Feist* uncopyrightable-facts posture.
- r262: extend `CurveInterpolation` with `Cardinal { tension_q8: u8 }`
  — the tension-parameterised generalisation of uniform Catmull-Rom
  per §3.2 of `docs/image/filter/curve-interpolation.md` (E. Catmull,
  R. Rom, *"A class of local interpolating splines,"* Computer Aided
  Geometric Design, Academic Press, 1974). The per-knot tangent
  scales the §3.1 uniform-Catmull-Rom expression by `(1 − c)`:
  `m_i = (1 − c) · (p_{i+1} − p_{i−1}) / 2`. Tension `c ∈ [0, 1]` is
  encoded as `u8` 8.0 fixed-point (`tension_q8 = 0..=255` mapping to
  `c = 0..=1`) so the enum keeps `Copy + Eq + Hash`. Endpoints fall
  through `Catmull-Rom`'s standard centred-difference + one-sided
  collapse with the same `(1 − c)` scaling, so the §1 cubic-Hermite
  evaluator reduces to: the existing `CatmullRom` arm sample-for-sample
  at `c = 0` (the `(1 − c) = 1` factor leaves the secant slope
  unchanged); the flat-tangent convex-combination shape
  `h00·y_i + h01·y_{i+1}` at `c = 1` (every tangent is exactly `0`).
  Intermediate `c` blends between the two regimes, with higher `c`
  monotonically dampening per-segment overshoot above the bracketing
  knot maximum / below the bracketing knot minimum. Distinct from the
  six existing siblings: only `Cardinal` carries a runtime knob (the
  other six are fixed-parameter modes), and only `Cardinal` can be
  driven to a no-overshoot shape without forfeiting the cubic-Hermite
  visual smoothness of the underlying basis. Still `C¹` only and not
  monotone-safe at any `c < 1` (the scaled secant chord tangent allows
  overshoot just like uniform Catmull-Rom — for tone-curve UIs the
  `MonotoneCubic` default remains the safe choice). JSON factory
  parser accepts four new spellings for the mode parameter
  (`"cardinal"`, `"cardinal-spline"`, `"cardinal_spline"`,
  `"cardinalspline"`) plus a `tension` (or `c`) field carrying the
  tension parameter as a JSON number; out-of-range values clamp to
  `[0, 1]` and a missing tension defaults to `0.5`. Eight new unit
  tests in `src/curves.rs`
  (`cardinal_passes_through_control_points` — every knot is hit to
  within u8 round-off on the five-knot reference fixture used by the
  centripetal / chordal siblings,
  `cardinal_zero_tension_equals_catmull_rom` — `tension_q8 = 0` is
  byte-equivalent to `CatmullRom` on the uneven-knot stress fixture,
  `cardinal_full_tension_zeros_every_tangent` — `tension_q8 = 255`
  still interpolates every knot AND is per-segment overshoot-free
  (every LUT entry lies inside the bracketing-knot interval),
  `cardinal_two_points_is_straight_line_at_zero_tension` — 2-point
  ramp at `c = 0` reproduces the linear identity (the
  arbitrary-`c` round-trip does NOT — documented in the test),
  `cardinal_clamps_to_byte_range` — pathological control set at the
  half-tension default still produces a valid `[0, 255]`-clamped LUT
  with no NaN cast / panic,
  `cardinal_higher_tension_dampens_per_segment_overshoot` — sweep
  `c ∈ {0, 0.25, 0.5, 0.75, 1}` and verify the **total per-segment
  overshoot area** (sum of `max(0, lut[x] − bracket_hi[x]) +
  max(0, bracket_lo[x] − lut[x])`) is monotone-non-increasing AND
  vanishes at the high end AND strictly decreases at least once over
  the sweep — the test would be vacuous if cardinal silently aliased
  to a fixed-tangent mode,
  `cardinal_handles_clustered_knot_without_panic` — defensive guard
  against zero `dx` on tightly-clustered knots, mirroring the
  centripetal / chordal stress fixture),
  and one registry test `curves_factory_accepts_cardinal_mode_aliases`
  covering all four mode-name spellings + the `tension` and `c` JSON
  field spellings + the default-when-omitted contract + the
  out-of-range clamp contract.
- r248: add `ReinhardLocal` — the local "dodging-and-burning" variant
  of the Reinhard 2002 tone-mapping operator per
  `docs/image/filter/tone-mapping-operators.md` §3 (E. Reinhard,
  M. Stark, P. Shirley, J. Ferwerda, *"Photographic Tone Reproduction
  for Digital Images,"* ACM TOG / SIGGRAPH 2002, 21(3):267–276). The
  global form's `1 + L` denominator is replaced with a
  spatially-varying local average chosen per-pixel from a geometric
  Gaussian centre / surround scale pyramid: at each scale `s_k = s_0
  · 1.6^k` two Gaussian-blurred copies of the key-scaled luminance
  are built — centre `V1` with `α_1 = 1/(2√2)`, surround `V2` with
  `α_2 = 1.6·α_1` — and the §3.2 normalised difference
  `V = (V1 − V2) / (2^φ · a / s² + V1)` is formed (`φ = 8`). §3.3
  selects the largest scale for which `|V| < ε` (default `0.05`); a
  high-contrast edge truncates the search to the finest scale so
  local detail is preserved. §3.4 evaluates `Ld = L_s / (1 + V1(·,
  s_m))` and chroma-preservingly re-modulates the original linear RGB
  by `Ld / L_w` before re-encoding through the sRGB OETF.
  Parameters: `key` (§2.1 `a`, default `0.18`), `phi` (§3.2,
  default `8`), `epsilon` (§3.3, default `0.05`), `scales` (pyramid
  depth, default `8`), `initial_scale` (`s_0`, default `1.0`). Builder
  helpers normalise non-finite / non-positive values back to the paper
  defaults. Operates on `Gray8` / `Rgb24` / `Rgba`; YUV returns
  `Unsupported` (same constraint as `Reinhard` / `ReinhardExtended`).
  Registry exposes three factory IDs `"reinhard-local"`,
  `"tonemap-reinhard-local"`, and `"dodge-and-burn"` accepting `key`
  (or `a`), `phi` (or `sharpening`), `epsilon` (or `eps` /
  `threshold`), `scales` (or `levels`), and `initial_scale` (or `s0`).
  Eight new unit tests in `src/reinhard_local.rs`
  (`output_in_range_rgb` — shape preservation on an RGB gradient,
  `black_input_stays_black` — zero-luminance frame stays zero,
  `alpha_preserved_on_rgba` — RGBA alpha pass-through,
  `rejects_yuv` — YUV → `Unsupported`,
  `local_differs_from_global_on_edge` — local form must diverge from
  the global form on a step edge (the textbook differentiating case),
  `single_scale_reduces_to_simple_blur_division` — `scales = 1` still
  produces a valid output,
  `epsilon_zero_picks_smallest_scale` — vanishing ε forces the finest
  scale and must not panic,
  `constructor_normalises_nonfinite_key` — `NaN` / negative / zero `key`
  fall back to the §2.1 `0.18` default,
  `with_scales_zero_clamps_to_one` — pyramid depth `0` clamps to `1`,
  `gray8_path_runs` — single-plane Gray8 falls back to the scalar form
  cleanly).
- r237: add `ReinhardExtended` — the unkeyed white-clamping form of
  the Reinhard 2002 tone-mapping operator per
  `docs/image/filter/tone-mapping-operators.md` §5.1
  (E. Reinhard, M. Stark, P. Shirley, J. Ferwerda, *"Photographic
  Tone Reproduction for Digital Images,"* ACM TOG / SIGGRAPH 2002,
  21(3):267–276 — §3.1 "Burn-out" curve, applied without the §3
  log-average key-scaling pre-step). Implements
  `L_d = L · (1 + L / L_white²) / (1 + L)` chroma-preservingly on
  the de-gamma'd linear-light luminance and re-encodes through the
  sRGB OETF. Single `l_white` parameter; `l_white <= 0` (the
  default — also the result of constructing with `f32::NAN` or any
  non-positive value) auto-picks the per-frame `L_max` so the
  brightest input pixel maps exactly to `1`, matching the §5.1
  recommendation. Distinct from the existing keyed `Reinhard`: no
  log-average accumulation across the frame (single pass vs.
  two-pass), no `key` knob — useful when the caller already has
  exposure-correct linear luminance and just wants a soft highlight
  clamp. Operates on `Rgb24`/`Rgba`/`Gray8`; YUV returns
  `Unsupported` (same constraint as `Reinhard`). Registry exposes
  two factory IDs `"reinhard-extended"` + `"tonemap-reinhard-extended"`
  accepting `l_white`/`white`/`white_point` spellings of the
  parameter. Eight new unit tests in `src/reinhard_extended.rs`
  (`output_in_range` — RGB shape preservation,
  `auto_white_brightest_pixel_reaches_white` — auto-`L_max` maps
  the single brightest pixel exactly to encoded sRGB 255,
  `black_input_stays_black` — zero-luminance frame stays zero,
  `small_explicit_white_saturates_highlights_faster` — a small
  explicit `l_white` lifts the dark stripe more than the auto
  mode while not dimming highlights,
  `explicit_white_equals_lmax_matches_auto_mode` — passing the
  re-derived `L_max` explicitly is byte-equivalent to the auto
  default, `non_positive_white_falls_back_to_auto` — `NaN` /
  negative / zero all normalise to the auto sentinel including
  through `with_white`, `alpha_preserved_on_rgba` — RGBA alpha
  pass-through, `rejects_yuv` — YUV returns `Unsupported`).
- r231: extend `CurveInterpolation` with `ChordalCatmullRom` — the
  `α = 1` non-uniform Catmull-Rom variant per §3.3 of
  `docs/image/filter/curve-interpolation.md` (Yuksel, Schaefer, Keyser,
  *"Parameterization and applications of Catmull-Rom curves,"*
  Computer-Aided Design 43(7):747–755, 2011 — the third and final member
  of the alpha-parameterised family after the existing uniform
  (`CatmullRom`, `α = 0`) and centripetal (`CentripetalCatmullRom`,
  `α = 0.5`) modes). Per-knot spacing is the raw chord length
  `t_{i+1} − t_i = |p_{i+1} − p_i|` (the centripetal case's
  fractional `^0.5` exponent dropped to plain `1`); the
  Barry-Goldman three-term tangent shape is unchanged. To avoid
  duplicating the centripetal implementation the r231 patch refactors
  the centripetal arm into a shared `Curve::alpha_catmull_rom_tangents`
  helper parameterised by the `α` exponent — the new chordal arm
  calls into the same helper with `α = 1`, while the centripetal arm
  still calls it with `α = 0.5`. Boundary tangents collapse to the
  one-sided secant of the adjacent segment (one-sided phantom-knot
  duplication as in §3.1 of the reference doc), so a 2-point fixture
  reproduces the linear identity byte-for-byte. Distinct from both
  existing siblings: the chord-length-based spacing is non-trivial
  vs uniform (`Δt` = `1`) on uneven knots, and the raw-length
  spacing is non-trivial vs centripetal (`Δt = √chord`) on the same
  fixture; unlike centripetal the chordal mode does *not* inherit
  the Yuksel et al. cusp-free property (that property holds for
  `α ∈ (0, 1)`, with `α = 0.5` being the only scale-invariant
  choice), so the centripetal variant remains the recommended
  default for path work. Still `C¹` only and not monotone-safe; the
  textbook tone-curve default stays `MonotoneCubic`. JSON factory
  parser accepts five new spellings for the mode parameter:
  `"chordal"`, `"chordal-catmull-rom"`, `"chordal_catmull_rom"`,
  `"chordalcatmullrom"`, `"catmull-rom-chordal"`; unknown spellings
  continue to fall back to the default `MonotoneCubic` so existing
  JSON jobs are unaffected. Seven new unit tests on top of the
  existing curves suite (`chordal_passes_through_control_points` —
  interpolation property at five knots, `chordal_two_points_is_straight_line`
  — 2-point collapse to the linear identity,
  `chordal_clamps_to_byte_range` — pathological overshoot fixture
  stays inside `[0, 255]` byte quantisation,
  `chordal_identity_under_collinear_control_points` — collinear
  knots reproduce the linear identity inside the §3.3 family,
  `chordal_differs_from_uniform_catmull_rom_on_uneven_knots` — the
  new mode is not silently identical to the uniform `α = 0` default
  on uneven knots, `chordal_differs_from_centripetal_on_uneven_knots`
  — the new mode is not silently identical to the centripetal
  `α = 0.5` mode on the same fixture, and
  `chordal_handles_clustered_knot_without_panic` — chord-based
  divisor never collapses to zero even at tightly-clustered x knots
  for `α = 1` either). One new registry test
  (`curves_factory_accepts_chordal_mode_aliases` walks all five
  spellings). Curve-interpolation count goes 5 → 6. Backed by §3.3
  of `docs/image/filter/curve-interpolation.md`, which itself cites
  the original publication (Yuksel et al. 2011 with DOI link only),
  per the *Feist* uncopyrightable-facts posture.

- r226: extend `CurveInterpolation` with `CentripetalCatmullRom` — the
  `α = 0.5` non-uniform Catmull-Rom variant per §3.3 of
  `docs/image/filter/curve-interpolation.md` (Yuksel, Schaefer, Keyser,
  *"Parameterization and applications of Catmull-Rom curves,"*
  Computer-Aided Design 43(7):747–755, 2011 — the centripetal analysis
  on top of the classical 1974 Catmull-Rom construction). Per-knot
  spacing is the chord-length raised to `0.5`,
  `t_{i+1} − t_i = |p_{i+1} − p_i|^0.5`; the per-knot derivative comes
  from the Barry-Goldman three-term form
  `dy/dt = (y_{i+1} − y_i)/(t_{i+1} − t_i)
         − (y_{i+1} − y_{i−1})/(t_{i+1} − t_{i−1})
         + (y_i − y_{i−1})/(t_i − t_{i−1})`, then chain-rule-converted
  to the `dy/dx` tangent the cubic Hermite basis expects via the
  matching `dx/dt = (x_{i+1} − x_{i−1})/(t_{i+1} − t_{i−1})` ratio.
  Boundary tangents collapse to the one-sided secant of the adjacent
  segment (one-sided phantom-knot duplication as in §3.1 of the
  reference doc), so a 2-point fixture reproduces the linear identity
  byte-for-byte. Distinct from the existing uniform `CatmullRom` mode
  (`α = 0` in the same §3.3 family) because the tangent denominators
  track inter-knot distance rather than dropping to plain `1`, so the
  characteristic overshoot ripples of uniform Catmull-Rom around
  tightly-clustered control points are bounded — the cusp / self-loop
  property motivating Yuksel et al. — while keeping the same `C¹`
  continuity. Still not monotone-safe (the cusp-free property does not
  forbid `y`-axis overshoot below `y_min` or above `y_max`); the
  textbook tone-curve default remains `MonotoneCubic`. JSON factory
  parser accepts five new spellings for the mode parameter:
  `"centripetal"`, `"centripetal-catmull-rom"`,
  `"centripetal_catmull_rom"`, `"centripetalcatmullrom"`,
  `"catmull-rom-centripetal"`; unknown spellings continue to fall back
  to the default `MonotoneCubic` so existing JSON jobs are unaffected.
  Seven new unit tests on top of the existing curves suite
  (`centripetal_passes_through_control_points` — interpolation property,
  `centripetal_two_points_is_straight_line` — 2-point collapse to the
  linear identity, `centripetal_clamps_to_byte_range` — pathological
  overshoot fixture stays inside `[0, 255]` byte quantisation,
  `centripetal_identity_under_collinear_control_points` — collinear
  knots reproduce the linear identity inside the cusp-free family,
  `centripetal_differs_from_uniform_catmull_rom_on_uneven_knots` — the
  new mode is not silently identical to the uniform default,
  `centripetal_uniform_knots_match_uniform_catmull_rom` — symmetric
  reverse: evenly-spaced knots have `Δt` constant so the centripetal
  tangent collapses to the §3.1 uniform form, and
  `centripetal_handles_clustered_knot_without_panic` — chord-based
  divisor never collapses to zero even at tightly-clustered x knots).
  One new registry test (`curves_factory_accepts_centripetal_mode_aliases`
  walks all five spellings). Curve-interpolation count goes 4 → 5.
  Backed by §3.3 of `docs/image/filter/curve-interpolation.md`, which
  itself cites the original publication (Yuksel et al. 2011 with DOI
  link only), per the *Feist* uncopyrightable-facts posture.

- r220: extend `DistanceTransform` with a runtime-selectable chamfer
  kernel via the new `ChamferKind` enum + `with_kind` builder. Clean-
  room transcription of `docs/image/filter/distance-transform.md` §3.2,
  generalising the original hard-coded 3-4 mask (Borgefors,
  *Distance transformations in digital images*, CVGIP 34, 1986) to
  every kernel in the reference table:
  - `Chamfer34` (existing default, divisor `a = 3`).
  - `Chamfer5711` — 5-7-11 mask: orthogonal = 5, diagonal = 7,
    knight-move (±1, ±2) / (±2, ±1) = 11 (`a = 5`). The doc's "better
    than 3-4" entry; per-step diagonal ratio `7 / 5 = 1.4` vs the true
    `√2 ≈ 1.414` (3-4 reports `4 / 3 ≈ 1.333`).
  - `CityBlock` — exact L1 / Manhattan metric (orthogonal-only mask
    weighted `1`; `a = 1`); trace ultimately to A. Rosenfeld &
    J. L. Pfaltz, *Sequential operations in digital picture processing*,
    J. ACM 13(4), 1966.
  - `Chessboard` — exact L∞ / Chebyshev metric (orthogonal +
    diagonal weight `1`; `a = 1`).
  The propagation loop is refactored around a per-kernel `Offset =
  (dx, dy, weight)` table (separate forward / backward halves keep
  the two-pass raster invariant from doc §3.1), and the final divide
  uses `ChamferKind::divisor()` so the rescale path stays branch-
  free. The JSON factory (`distance-transform` / `distance`) gains a
  `kind` (or `kernel`) parameter accepting `chamfer-3-4`,
  `chamfer-5-7-11`, `city-block`, `chessboard`, plus the metric-name
  spellings `manhattan` / `l1` / `chebyshev` / `l-infinity` / `linf`;
  unknown values surface as an `Error::invalid` rather than silently
  falling back. Backwards-compatible: the historical builder API
  (`DistanceTransform::new(threshold)` + `with_invert` / `with_scale`)
  and the factory default both still pick `Chamfer34`, so existing
  callers and JSON jobs are unaffected. Seven new unit tests on top
  of the existing distance-transform suite (`city_block_is_l1_metric`
  checks L1 = `|dx| + |dy|`, `chessboard_is_linf_metric` checks
  L∞ = `max(|dx|, |dy|)`, `chamfer_5_7_11_diagonal_is_closer_to_euclid`
  bounds the 5-7-11 estimate above the 3-4 estimate and below `√2`,
  `chamfer_5_7_11_knight_move_uses_weight_11` verifies the
  `11 / 5 → 2 px` rescale on a (±1, ±2) neighbour, `divisor_table`
  pins the published `a` weights, `foreground_pixel_zero_under_every_kernel`
  enforces the seed-pixel invariant across all four kernels, and
  `city_block_distance_is_separable_sum` checks a non-axial L1
  sample). Two new registry tests
  (`distance_transform_factory_accepts_kind_aliases` walks all 17
  spellings under both the `kind` and `kernel` keys;
  `distance_transform_factory_rejects_unknown_kind` asserts the
  unknown-kernel error path). Kernel count goes 1 → 4. Backed by §3
  of `docs/image/filter/distance-transform.md`.

- r215: extend `CurveInterpolation` with `NaturalCubic` — the classical
  `C²` natural cubic spline (de Boor, *A Practical Guide to Splines*,
  Springer 1978; tridiagonal step credited to L. H. Thomas, Watson Sci.
  Comp. Lab. 1949). Clean-room transcription of
  `docs/image/filter/curve-interpolation.md` §4 (§4.1 system definition,
  §4.2 natural boundary `M_0 = M_{n−1} = 0`, §4.3 Thomas forward-
  elimination + back-substitution, §4.4 evaluator). The system
  `h_{i−1}·M_{i−1} + 2(h_{i−1}+h_i)·M_i + h_i·M_{i+1} = 6·(Δ_i −
  Δ_{i−1})` is solved in a single `O(n)` sweep for the per-knot second
  derivatives `M_i = P''(x_i)`; each segment is then evaluated as the
  cubic `P(x) = y_i + (Δ_i − h_i·(2M_i + M_{i+1})/6)·t + (M_i / 2)·t² +
  ((M_{i+1} − M_i)/(6 h_i))·t³` with `t = x − x_i`. Continuous second
  derivative across interior knots makes this the gentlest of the four
  interpolants (compared to the existing piecewise-linear, `C¹`-only
  Catmull-Rom, and `C¹` monotone-clamped Fritsch-Carlson modes); the
  trade-off is overshoot risk, so `MonotoneCubic` remains the default
  for tone-curve sliders where monotonicity must hold. Edge cases:
  with only 2 control points (no interior knots) the tridiagonal
  system has zero rows so `M ≡ 0` and the evaluator collapses to the
  straight-line identity, matching the linear-Hermite path
  sample-by-sample. JSON factory parser accepts three new spellings
  for the mode parameter: `"natural-cubic"`, `"natural_cubic"`,
  `"natural"`. Six new unit tests on top of the existing curves suite
  (identity-on-2-point preservation, control-point pass-through at
  five knots, second-difference proxy stays stable across an interior
  knot — the `C²` continuity signature, second-difference proxy near
  both endpoints is `≈ 0` — the natural boundary signature, 2-point
  collapse-to-linear, byte-range clamp under a pathological overshoot
  fixture). Curve-interpolation count goes 3 → 4. Backed by §4 of
  `docs/image/filter/curve-interpolation.md`, which itself cites the
  original publications by author/year + DOI link only, per the
  *Feist* uncopyrightable-facts posture.

- r209: land `EuclideanDistanceTransform` — exact-Euclidean distance
  transform (Felzenszwalb–Huttenlocher, *"Distance Transforms of
  Sampled Functions"*, Theory of Computing 8(19), 2012). Computes,
  for every pixel of a thresholded binary mask, the **exact**
  Euclidean distance to the nearest foreground pixel — the exact
  counterpart to the existing `DistanceTransform`, which ships the
  cheaper Borgefors 3-4 chamfer approximation. The squared distance
  is the lower envelope of upward parabolas `(p − q)² + f(q)`, one
  per sample; since all parabolas share the same curvature, any two
  intersect at a single abscissa
  `s = ((f[q] + q²) − (f[v] + v²)) / (2q − 2v)` and the envelope is
  built by a single left-to-right march that pushes and pops
  candidates at most once each (`O(n)` per row). A second
  left-to-right pass fills `D(q)` by walking the envelope. The 2-D
  transform runs the 1-D driver once per column then once per row
  (using the column-pass output as `f` for the row pass), so the
  whole transform is `O(d · N)` for an exact result. Single-plane
  `Gray8` input and output of the same dimensions; foreground pixels
  (the `value >= threshold` test, or its inversion when
  `EuclideanDistanceTransform::invert` is set) emit distance `0`,
  every other pixel emits the rounded Euclidean distance to the
  nearest foreground site scaled by `scale` and clamped to
  `[0, 255]`. Knobs: `threshold` (mask cut-off), `invert` (flip the
  FG/BG test), `scale` (compress / expand the rendered dynamic
  range; non-finite or non-positive inputs silently keep the
  previous value). Useful for stroke generation, contour rings,
  signed-distance-field glyph atlases, feathering masks, and
  morphology effects that need a true Euclidean metric rather than
  the chamfer approximation. 18 unit tests (foreground-pixel
  zero-distance, single-source-matches-actual-Euclidean,
  monotone-increasing-along-radius, no-foreground saturation,
  fully-foreground all-zero, invert-swaps-FG-BG, scale
  compress / expand, scale silent-reject of bad input, far-distance
  clamp at 255, 0×0 bypass, RGB / YUV rejection; the 1-D driver
  `dt_1d` is exercised directly against a brute-force squared-
  Euclidean oracle on patterns with two features, all `+∞`, and a
  `k == 0` anchor-replacement edge case; the 2-D path is checked
  against the same brute-force oracle on three deterministic
  patterns — a random-ish point cloud, a solid border ring that
  stresses many envelope candidates, and a diagonal-line FG that
  stresses separable-pass coupling — all agree to `|Δ| < 1e-6`)
  plus 3 factory smoke tests (`euclidean-distance-transform` /
  `euclidean-distance` / `edt` aliases all resolve; output-port
  spec is `Gray8`; non-positive and non-finite `scale` are
  factory-rejected). Filter count goes 130 → 131; factory count
  178 → 181. Clean-room transcription from
  `docs/image/filter/distance-transform.md` §2 (which itself cites
  the original Felzenszwalb–Huttenlocher publication by DOI link
  only, per the *Feist* uncopyrightable-facts posture).

- r209 hygiene: scrub the 142 pre-existing parameter-name brand
  citations across this crate — 36 in `src/lib.rs`, 1 in
  `src/frame.rs`, 2 in `src/morphology.rs`, and 103 in `README.md` —
  rewriting the two-letter abbreviation lead-ins (and their
  "analogue" variant) in doc-comments to the neutral
  "documented CLI" / "documented-CLI analogue" phrasing already
  used in many adjacent comments. No public API change.

- r205: land `Niblack` — Wayne Niblack's adaptive local-statistics
  threshold (Niblack, *An Introduction to Digital Image Processing*,
  Prentice-Hall 1986, §5.1 — page-segmentation example). For each
  pixel the filter computes the mean `μ(x, y)` and standard deviation
  `σ(x, y)` of the `(2·radius + 1)²` neighbourhood centred at
  `(x, y)`, then emits `255` where `sample(x, y) ≥ T(x, y)` with
  `T(x, y) = μ(x, y) + k · σ(x, y)`, `0` elsewhere (or the inverted
  test if [`Niblack::invert`] is set). The default `k = -0.2` is
  Niblack's textbook page-segmentation bias — the threshold sits
  below the local mean by a fraction of the local spread, so dark
  ink against a brightish but locally-noisy background is reliably
  captured; positive `k` biases the threshold above the mean for the
  complementary light-on-dark case. Local mean and standard
  deviation are computed via the `Var(X) = E(X²) − E(X)²` identity
  with two separable box-sum passes (one over `I`, one over `I²`);
  the variance is clamped to `0` before the `sqrt` to absorb the
  tiny floating-point negatives that show up on near-constant
  patches when cancellation eats every significant digit (the
  textbook fix — otherwise we'd ship a `NaN` σ on flat input). The
  whole pipeline runs in `O(W · H)` regardless of `radius`, the
  same cost class as the existing `AdaptiveThreshold`. Border
  samples clamp to the nearest in-bounds pixel so the output keeps
  the input dimensions. Any supported input is collapsed to a luma
  plane first (Gray8 / YUV use the Y plane directly; RGB / RGBA use
  the `(R + 2G + B) / 4` quick luma) via the existing
  `laplacian::build_luma_plane` helper; output is always single-
  plane `Gray8`. The filter joins the segmentation family at the
  "local mean + local σ threshold" position complementing
  [`AdaptiveThreshold`] (local mean only — Niblack with `k = 0`)
  and [`OtsuThreshold`] (a single global cut chosen to maximise
  between-class variance). Tests: 15 (flat input with `k = 0` is
  all-white; flat input with `invert` is all-black; flat input with
  positive `k` near a one-pixel disturbance dips below the
  threshold; ink against a bright background with `k = -0.2`
  correctly binarises ink to black and background to white; the
  separable box-sum implementation matches a brute-force double-
  loop reference on a varied 11×9 pattern across `radius ∈ {0, 1,
  2, 3, 5}` to `|Δ mean| < 1e-9` and `|Δ σ| < 1e-6`; end-to-end
  per-pixel binarisation matches a brute-force reference on a non-
  trivial 13×11 pattern at `radius = 2, k = -0.2`; negative `k`
  splits a ramp toward black on the left and white on the right;
  positive `k` produces no more white pixels than negative `k` on
  the same ramp; RGB and Yuv420P inputs collapse to luma correctly;
  non-finite `k` and zero dimensions are rejected; unsupported
  pixel formats are rejected; output shape matches input; PTS
  pass-through; builder chain composes; a near-constant patch
  with one off-by-one pixel does not ship NaNs through the
  variance-clamping fix; the smallest 3×3 window matches the
  brute-force reference on a bright-centre fixture). Two new
  factory names wired into `register()`: `niblack`,
  `niblack-threshold`. The factory accepts `radius` (integer),
  `k` (finite float), and `invert` (boolean). Three new factory
  smoke tests cover the Gray8 output port, every alias name plus
  the invert flag, and the happy-path k acceptance. Filter count
  climbs from 129 → 130 named types; factory count climbs from
  176 → 178 names.

- r198: land `Gabor` + `GaborMode` — the oriented Gaussian-modulated
  cosine filter (Dennis Gabor, "Theory of Communication", *Journal of
  the Institution of Electrical Engineers* 93(III):429–457, 1946; John
  G. Daugman, "Uncertainty relation for resolution in space, spatial
  frequency, and orientation optimized by two-dimensional visual
  cortical filters", *Journal of the Optical Society of America A*
  2(7):1160–1169, July 1985). The filter samples the continuous
  `exp(-(x'² + γ²·y'²) / (2σ²)) · cos(2π·x'/λ + ψ)` kernel on a
  `(2·radius+1)²` integer grid centred at the kernel origin (where
  `x' = (x − cx)·cos θ + (y − cy)·sin θ`,
  `y' = -(x − cx)·sin θ + (y − cy)·cos θ`); the default radius
  is `ceil(3·σ·max(1, 1/γ))` clamped to `[1, 32]`, so the kernel
  support tracks the elongated envelope when `γ < 1`. The discrete
  coefficients are zero-meaned (DC component subtracted) so a flat
  constant input produces exactly zero response — the same fix
  already applied to the discrete LoG kernel, and the textbook
  recommendation (Petkov & Kruizinga 1997 *Biological Cybernetics*).
  The convolution is dense 2-D (the Gabor is not rank-1 separable
  for general θ and γ ≠ 1) at `O(W·H·(2r+1)²)` cost; border samples
  edge-clamp to the nearest in-bounds pixel so the output keeps the
  input dimensions. Two output modes are offered:
  [`GaborMode::Signed`] (default) linearly remaps the raw response
  from `[-R, +R]` to `[0, 255]` with neutral grey 128 marking zero
  response (preserves the orientation + phase polarity of the
  carrier — the linear "simple cell" response), and
  [`GaborMode::Magnitude`] linearly remaps `|response|` from
  `[0, R]` to `[0, 255]` (classical oriented-energy response — the
  half-rectified simple-cell or one half of a quadrature-pair
  complex-cell magnitude). The renderer's `R` is the maximum
  absolute discrete response against a `[0, 255]` grating
  (half the L1 norm of the zero-mean kernel), computed once at
  kernel-build time. Configurable knobs: `wavelength` (carrier
  period in pixels along `x'`; rejected if `< 2` per the spatial
  Nyquist limit), `orientation_degrees` (CCW from positive
  image-x), `phase_degrees` (0 = even-symmetric cosine for
  bar-like response, 90 = odd-symmetric sine for step-edge
  response), `sigma` (envelope standard deviation in pixels;
  must be `> 0`), `gamma` (envelope spatial-aspect ratio
  `σ_y / σ_x`; `1.0` = isotropic, `< 1.0` = elongated along the
  carrier; must be `> 0`), `radius` (override the
  `ceil(3·σ·max(1, 1/γ))` auto-selection; clamped to `[1, 32]`),
  `output_gain` (global multiplier applied before clamp). Any
  supported input is collapsed to a luma plane first (Gray8 / YUV
  use the Y plane directly; RGB / RGBA use the `(R + 2G + B) / 4`
  quick luma, sharing the existing `laplacian::build_luma_plane`
  helper); the output is always single-plane `Gray8` of the input
  dimensions. The filter joins the edge / texture-energy family at
  the "oriented bandpass" position complementing the isotropic
  `LaplacianOfGaussian` (Marr–Hildreth) and the 1st-derivative
  Sobel / Prewitt / Scharr / Roberts. Tests: 20 (zero-mean
  correction proven across a sweep of `radius × sigma × gamma ×
  wavelength × orientation × phase` to `|sum| < 1e-4`; flat
  Gray8 input → exactly neutral grey 128 on signed output; flat
  input → exactly 0 on magnitude output; a vertical-stripe carrier
  matching the filter's orientation and wavelength excites the
  signed response well off neutral grey; the orientation
  selectivity test compares aligned vs perpendicular filter
  responses on the same grating — aligned mean ≫ perpendicular
  mean; the 180° rotation invariance of the even-symmetric (ψ=0)
  Gabor kernel is preserved bit-exact in the interior; the
  ψ=90° (odd-symmetric sine carrier) Gabor responds strongly at a
  vertical step edge and weakly far from it; RGB / Yuv420P luma
  collapse; PTS pass-through; sigma / gamma / wavelength /
  output_gain rejection of out-of-range values; unsupported-format
  rejection; 3×3 input smaller than the kernel still produces
  zero-meaned flat output via border clamping; signed-noise mean
  centres near 128; radius-clamping at the `[1, 32]` builder
  endpoints; builder-method composition). Three new factory names
  wired into `register()`: `gabor`, `gabor-filter`, `gabor-wavelet`.
  The factory accepts `wavelength` (≥ 2; rejected otherwise),
  `orientation_degrees` (or `orientation` / `theta`),
  `phase_degrees` (or `phase` / `psi`), `sigma` (> 0), `gamma`
  (or `aspect`; > 0), `radius` (integer, clamped),
  `mode` (`"signed"` / `"magnitude"`), `magnitude` (boolean
  shorthand for flipping the mode), and `output_gain` (finite
  float). Four new factory smoke tests cover the Gray8 output
  shape, every alias name, the magnitude / parameter-alias path,
  and the wavelength / sigma / gamma rejection branches. Filter
  count climbs from 128 → 129 named types (one new filter struct
  `Gabor` plus the public companion `GaborMode`); factory count
  climbs from 173 → 176 names.

## [0.1.2](https://github.com/OxideAV/oxideav-image-filter/compare/v0.1.1...v0.1.2) - 2026-05-29

### Added

- add Roberts cross edge operator (r24)

### Other

- r186 — Dither (Bayer ordered + 7 error-diffusion kernels) from clean-room kernel transcription
- r181 — LaplacianOfGaussian (Marr–Hildreth) edge / zero-crossing detector
- r174 — FreiChen 3×3 orthonormal-basis edge / line detector
- r131 — BilateralBlur Tomasi-Manduchi 1998 citation + quantitative tests
- r105 — Scharr 3×3 first-derivative edge operator
- r101 — Prewitt 3×3 first-derivative edge operator
- r23 — MaxRgb / MinRgb (HSV-V) per-pixel channel collapse
- r22 — Reinhard / Hable / Drago tone-mapping + Curves + DistanceTransform + Cyanotype
- r21 — Kuwahara / AnisotropicBlur / ZoomBlur / RadialBlur / EmbossDirectional + DisplacementMap (two-input)
- r20 — Crystallize / Halftone / GradientMap / SelectiveColor / CrossProcess / OtsuThreshold + Comic
- r19 — Heatmap / SplitTone / FloodFill / PosterizeChannels / Toon + Watermark (two-input)
- r18 — HoughCircles / AutoTrim / DropShadow / InnerShadow / Bloom / EdgeDetect (multi) + Difference (two-input)
- r17 — Exposure / Temperature / Vibrance / BwMix / Clarity / ShadowHighlight / BorderedFrame
- r16 — BilateralBlur / Canvas / ColorBalance / GradientRadial / GradientConic / GravityTranslate / HslShift
- r15 — HslRotate / VignetteSoft / ChromaticAberration / Pixelate / ChannelMixer / AdaptiveThreshold
- r14 — Sketch / Deskew / HoughLines / BarrelInverse + Stegano / Stereo (two-input)
- r13 — BlueShift / Frame / Shade / Paint / Quantize + Clut / HaldClut (two-input)
- r12 — Clamp / AutoLevel / Contrast/LinearStretch / ColorMatrix / Function / Laplacian / Canny
- r11 — Evaluate / Cycle / Statistic / Affine / Srt + 4 composite ops
- r10 — Roll / Shave / Extent / Trim / ChannelExtract / MorphologyEdge
- r9 — TiltShift rotated focus band (angle_degrees)
- r9 — Distort tangential coefficients (Brown-Conrady p1, p2)
- r9 — Perspective output_size canvas resize
- r9 — Charcoal Gaussian pre-blur (radius)
- add Composite (Porter–Duff + arithmetic) two-input family
- add TiltShift + surface Polar cx / cy / max_radius keys
- add Distort radial barrel / pincushion lens warp
- add Perspective 4-corner homography warp
- add Morphology dilate / erode / open / close
- add Wave / Spread / Charcoal / Convolve / Polar + factories
- list Tint / SigmoidalContrast / Implode / Swirl / Despeckle
- add Tint / SigmoidalContrast / Implode / Swirl / Despeckle + factories
- list Vignette / Colorize / Equalize / AutoGamma
- add Vignette / Colorize / Equalize / AutoGamma + factories
- wire 16 more filter factories into register()
- wire sharpen / gamma / brightness-contrast factories

### Added

- r186: land `Dither` + `DitherMode` + `BayerMatrix` +
  `DiffusionKernel` — the classic bit-depth-reduction dither family
  backed by the freshly-staged
  `docs/image/filter/dithering-kernels.md` clean-room reference. The
  module ships both ordered-dither (tiled Bayer threshold maps, sizes
  `2×2` / `4×4` / `8×8`, built by the recursive
  `M_2n = 4·M_n + M_2[hi, lo]` rule from Bayer 1973 IEEE ICC) and the
  seven canonical error-diffusion kernels: Floyd–Steinberg (÷16,
  Floyd & Steinberg 1976 SID), Jarvis–Judice–Ninke (÷48, Jarvis et al.
  1976 CGIP), Stucki (÷42, Stucki 1981 IBM RZ1060), Sierra-3 (÷32),
  Sierra-2 (÷16), Sierra-Lite / Filter-Lite / Sierra-2-4A (÷4) — the
  three Sierra kernels are community-attributed c. 1989–1990 — and
  Atkinson (÷8 with coefficient sum 6, so 2/8 of each pixel's
  quantisation error is intentionally discarded; Apple, Macintosh-era
  late 1980s). The error-diffusion path accumulates per-pixel
  residues in an `i32` buffer indexed by `(y·width + x)` so the
  arithmetic stays exact across the whole scan; the per-neighbour
  residue is divided by the kernel divisor exactly once at the
  destination read. Output bit-depth is configurable via `with_levels`
  — the default `levels = 2` gives a 1-bit black-and-white halftone
  (the Macintosh / newsprint use case); `levels = 4` gives a 2-bit
  dithered quantisation; etc. Per-channel for `Rgb24` / `Rgba` (alpha
  on RGBA is pass-through unchanged), luma-only on YUV (chroma
  planes are left untouched so hue/saturation stay put, matching every
  other "tone" filter in this crate). 12 new factory aliases —
  `dither` (configurable via JSON `kernel: "..."` / `levels: N` /
  `matrix: 2|4|8`), `dither-floyd-steinberg` & `dither-fs`,
  `dither-jjn` & `dither-jarvis`, `dither-stucki`, `dither-sierra3`,
  `dither-sierra2`, `dither-sierra-lite`, `dither-atkinson`,
  `dither-bayer` & `ordered-dither`. 24 module-level unit tests pin
  the Bayer recurrence (including the
  `M_4[i, j] = 4·M_2[i mod 2, j mod 2] + M_2[i/2, j/2]` self-similar
  identity), the index-set permutation property of the 8×8 map, the
  coefficient-sum / divisor invariant for every kernel (FS / JJN /
  Stucki / Sierra-3 / Sierra-2 / Sierra-Lite all error-conserving;
  Atkinson divisor 8 with Σ = 6), the binary `{0, 255}` output at
  `levels = 2`, the quartile-only `{0, 85, 170, 255}` output at
  `levels = 4`, mean conservation for Floyd–Steinberg on a flat-grey
  field, the demonstrably non-conserving behaviour for Atkinson,
  endpoint preservation (pure black / pure white round-trip on
  ordered dither), statelessness across repeated `apply` calls for
  both modes, RGB / RGBA / YUV plane handling (chroma untouched on
  YUV; alpha untouched on RGBA), and rejection of unsupported pixel
  formats. Five additional registry smoke tests cover every alias
  via the JSON parameter path. Factory count climbs from 161 → 173
  filter names.

- r181: land `LaplacianOfGaussian` + `LogMode` — the Marr–Hildreth
  Laplacian-of-Gaussian edge / zero-crossing detector (David Marr &
  Ellen Hildreth, "Theory of Edge Detection", *Proceedings of the Royal
  Society of London. Series B, Biological Sciences*
  207(1167):187–217, February 1980). The filter samples the continuous
  LoG kernel `K(x, y) = ((x²+y²−2σ²)/σ⁴) · exp(−(x²+y²)/(2σ²))` on a
  `(2·radius+1)²` integer grid centred at the kernel origin (default
  `radius = ceil(3·σ)` clamped to `[1, 16]`; three sigmas captures
  ~99.7 % of the Gaussian envelope and matches the textbook truncation
  rule), then zero-means the coefficients (subtracts the discrete
  arithmetic mean) so that a flat constant input produces exactly zero
  response everywhere — the load-bearing fix for the discrete
  quantisation error in the kernel's DC component. The convolution is
  dense 2-D (the LoG is **not** rank-1 separable, unlike the bare
  Gaussian) at `O(W·H·(2r+1)²)`; samples on the one-pixel-or-greater
  border are edge-clamped to the nearest in-bounds pixel so the output
  keeps the input dimensions. Two output modes are offered: the
  default [`LogMode::Magnitude`] returns `|LoG(x, y) · output_gain|`
  clamped to `[0, 255]` (a soft second-derivative response that peaks
  on either side of an edge and is zero on the edge itself — the
  scale-selective, noise-suppressed complement to the bare 3×3
  `Laplacian`), and [`LogMode::ZeroCrossings`] returns the binary
  Marr–Hildreth edge map (`255` where the gain-multiplied LoG response
  changes sign between a pixel and one of its 4-neighbours AND the
  absolute response gap straddling the crossing exceeds the configurable
  `slope_threshold`; `0` elsewhere — the slope gate is the textbook
  fix for the detector's tendency to mark flat backgrounds as edges
  through quantisation-noise sign changes). Configurable knobs:
  `sigma` (Gaussian standard deviation in pixels; default `1.4` per
  Marr–Hildreth's worked example), `radius` (override the auto-radius;
  clamped to `[1, 16]`), `output_gain` (global multiplier applied
  before clamping / thresholding; default `1.0`), `slope_threshold`
  (zero-crossings mode only; default `4.0`, set to `0.0` to report
  every sign change). Any supported input is collapsed to a luma
  plane first (Gray8 / YUV use the Y plane directly; RGB / RGBA use
  the `(R + 2G + B) / 4` quick luma, sharing the existing
  `Laplacian::build_luma_plane` helper); the output is always `Gray8`
  of the input dimensions. The filter complements the
  noise-amplifying [`Laplacian`] (bare 3×3, no pre-blur), the
  magnitude-based 1st-derivative kernels (`Edge` Sobel / `Prewitt` /
  `Scharr` / `Roberts`), the multi-stage `Canny` (1st-derivative with
  NMS + hysteresis), and the orthonormal-basis `FreiChen` (1st
  cosine projection) by adding an *isotropic, second-derivative,
  scale-selective* member to the edge-detector family. Tests: 19
  (hand-derived classical sign pattern for the radius-1 / σ=1
  kernel — centre & cardinals negative, corners positive, 4-fold
  rotational symmetry of cardinals & corners; zero-mean correction
  proven across a sweep of `radius ∈ {1,2,3,5,7}` × `sigma ∈
  {0.7,1.0,1.4,2.0,3.0}` to `|mean| < 1e-5`; flat input → zero
  magnitude; single bright pixel produces decaying |response|
  envelope; 4-fold isotropy of the magnitude response under a
  cross-shaped fixture; flat input → empty zero-crossings map;
  vertical step edge produces zero-crossings on or adjacent to the
  central column with quiet far columns; slope-threshold gate
  suppresses faint-ramp crossings entirely while a zero threshold
  reports every sign change; RGB / Yuv420P luma collapse; PTS
  pass-through; sigma-zero rejection; non-finite gain rejection;
  unsupported-format rejection; 1×1 input → zero; builder-method
  composition; negative-slope-threshold clamp; with-radius clamp to
  `[1, 16]`). Three new factory names wired into `register()`:
  `laplacian-of-gaussian`, `log` (short alias), `marr-hildreth`
  (paper-author alias). The factory accepts `sigma` (positive
  float), `radius` (integer, clamped), `mode` (`"magnitude"` /
  `"zero-crossings"`), `zero_crossings` (boolean shorthand for
  flipping the mode), `output_gain` (finite float), and
  `slope_threshold` (non-negative float). Three new factory smoke
  tests cover the magnitude output shape, the zero-crossings mode
  acceptance via both the long-form spelling and the boolean
  shorthand, and sigma rejection.

- r174: land `FreiChen` + `FreiChenMode` — the Frei–Chen 3×3
  orthonormal-basis edge / line detector (Werner Frei & Chung-Ching
  Chen, "Fast Boundary Detection: A Generalization and a New
  Algorithm", *IEEE Transactions on Computers* C-26(10):988–998,
  October 1977). The filter treats each 3×3 luma neighbourhood as a
  9-dimensional vector and projects it onto an orthonormal basis of
  9 templates partitioned into three sub-spaces: an **edge sub-space**
  (`S1..S4`: two `1/(2√2)`-scaled cardinal gradients plus two
  diagonal gradients), a **line sub-space** (`S5..S8`: two cardinal
  ripples plus two discrete-Laplacian variants), and a **mean sub-
  space** (`S9 = 1/3 · 1`). The output at each pixel is the cosine of
  the angle between the neighbourhood vector and the requested sub-
  space — `sqrt(E/T)` for `FreiChenMode::Edge` (default, sensitive
  to step edges) or `sqrt(L/T)` for `FreiChenMode::Line` (sensitive
  to ripples / Laplacian-like ridges) where `T = E + L + p9²`. Because
  the metric is a normalised ratio rather than a raw magnitude, a
  perfect edge / line gives `255` regardless of contrast — a
  qualitatively distinct response from the magnitude-based 3×3
  detectors (`Edge` Sobel / `Prewitt` / `Scharr`). Any supported
  input is luma-collapsed first (Gray8 / YUV use the Y plane
  directly; RGB / RGBA use the `(R + 2G + B) / 4` quick luma); the
  one-pixel border is clamped to the nearest in-bounds sample; the
  output is always `Gray8` of the input dimensions. Cost is `O(W·H)`
  (each 3×3 neighbourhood projects through nine fixed weight vectors).
  Tests: 14 (orthogonality + unit-norm of all 9 templates, hand-
  derived horizontal step response at the centre pixel
  (edge ≈128, line ≈74), pure-edge cosine = 1 for a mean-zero step
  patch, pure-edge cosine for an S2-aligned column step, Laplacian-
  pattern cosine = `sqrt(1/2) ≈ 180` on the line metric and zero on
  the edge metric, `cos²(edge) + cos²(line) ≤ 1` Pythagorean spot-
  check across a 16×16 ramp, RGB / YUV / Gray8 / format-rejection /
  PTS-pass-through). Factory aliases: `frei-chen`, `freichen`.

- r131: add a `BilateralBlur` derivation block citing Tomasi &
  Manduchi (ICCV 1998), document the σ_r → 0 / σ_r → ∞ limits, and
  ship two quantitative behavioural tests: (a) a seeded ±20 LCG
  noise patch at `radius=3, σ_s=2.0, σ_r=25` whose channel
  variance drops ~13.2× (137.95 → 10.43), and (b) a 150-step
  vertical edge at `radius=3, σ_s=1.5, σ_r=8` whose left/right
  pixels stay at 50 / 200 (gap = 150) where a same-radius box mean
  collapses the gap to ~21. No public API change — implementation
  was already correct since r16; round 131 just nails down the
  spec citation and the noise-reduction proof.

- r105: land `Scharr` + `ScharrMagnitude` — the Scharr 3×3 first-
  derivative edge operator (Hannes Scharr, "Optimal Operators in
  Digital Image Processing", PhD thesis, University of Heidelberg,
  2000). The two `±3 ±10 ±3`-weighted 3×3 kernels give `Gx` as the
  Scharr-weighted right-minus-left column and `Gy` as the Scharr-
  weighted bottom-minus-top row over the 3×3 neighbourhood centred
  on `(x, y)`; the row / column weights sum to `16`, so the raw
  magnitude is divided by `4` so the output lands in the same
  `[0, 255]` band as `Edge` (Sobel) and `Prewitt`. The edge
  magnitude is the Euclidean `sqrt(Gx²+Gy²)` (`ScharrMagnitude::L2`,
  default) or the Manhattan `|Gx|+|Gy|` (`L1`), clamped to
  `[0, 255]`. The one-pixel border is edge-clamped so the output
  keeps the input dimensions. Any supported input is luma-collapsed
  first; the output is always `Gray8`. The `±3 ±10 ±3` weighting
  gives the lowest orientation error of the three 3×3 first-
  derivative kernels in this crate (Sobel `±1 ±2 ±1`, Prewitt flat
  `±1`). Unlike `EdgeDetect`'s `L1`-only Scharr kernel, this
  dedicated filter defaults to the true Euclidean magnitude. Factory
  name `scharr` wired into `register()`; accepts `magnitude:
  "l1"|"l2"` (or the `l1: true` boolean shorthand). Stateless,
  single-frame, `O(W·H)`.

- r101: land `Prewitt` + `PrewittMagnitude` — the Prewitt 3×3
  first-derivative edge operator (Judith M. S. Prewitt, "Object
  Enhancement and Extraction", 1970). The two flat `±1`-weighted 3×3
  kernels give `Gx` = (right column − left column) and `Gy` =
  (bottom row − top row), each summed over the orthogonal axis of the
  3×3 neighbourhood centred on `(x, y)`; the edge magnitude is the
  Euclidean `sqrt(Gx²+Gy²)` (`PrewittMagnitude::L2`, default) or the
  Manhattan `|Gx|+|Gy|` (`L1`), clamped to `[0, 255]`. The one-pixel
  border is edge-clamped so the output keeps the input dimensions.
  Any supported input is luma-collapsed first; the output is always
  `Gray8`. Wider, less noise-sensitive support than `Roberts`;
  flatter, less directionally-biased weighting than `Edge` (Sobel).
  Unlike `EdgeDetect`'s `L1`-only Prewitt kernel, this dedicated
  filter defaults to the true Euclidean magnitude. Factory name
  `prewitt` wired into `register()`; accepts `magnitude: "l1"|"l2"`
  (or the `l1: true` boolean shorthand). Stateless, single-frame,
  `O(W·H)`.

- r24: land `Roberts` + `RobertsMagnitude` — the Roberts cross 2×2
  diagonal-difference edge operator (Lawrence Roberts, MIT PhD thesis,
  1963). The two responses are `Gx = a − d` and `Gy = b − c` over the
  2×2 window anchored at `(x, y)`; the edge magnitude is the Euclidean
  `sqrt(Gx²+Gy²)` (`RobertsMagnitude::L2`, default) or the Manhattan
  `|Gx|+|Gy|` (`L1`), clamped to `[0, 255]`. Any supported input is
  luma-collapsed first; the output is always `Gray8`. The smallest
  classic first-derivative detector — a sibling of `Edge` (Sobel) and
  `Laplacian` (second-derivative). Unlike `EdgeDetect`'s `L1`-only
  Roberts kernel, this dedicated filter defaults to the true Euclidean
  magnitude. Two factory names wired into `register()`: `roberts`,
  `roberts-cross`; the factory accepts `magnitude: "l1"|"l2"` (or the
  `l1: true` boolean shorthand). Stateless, single-frame, `O(W·H)`.

- r23: land `MaxRgb` + `MaxRgbMode` — per-pixel `max(R, G, B)` (HSV-V)
  or `min(R, G, B)` channel-collapse to greyscale. Stateless,
  branch-free, `O(W·H)`. Default output is single-plane `Gray8`; the
  `keep_format` flag preserves the input RGB / RGBA shape with all
  three colour channels equal to the picked value (RGBA alpha
  pass-through). Gray8 input is identity; YUV returns `Unsupported`
  because the operator is defined per-RGB-channel. Three new factory
  names wired into `register()`: `max-rgb`, `hsv-value` (alias —
  matches the HSV value-channel definition from Smith 1978), `min-rgb`.
  Filter count: 122 → 123 named types; factory count: 157 → 160 names.

- r22: land six new filters covering three classic global HDR tone-
  mapping operators (Reinhard 2002, Hable Uncharted-2 GDC 2010, Drago
  Eurographics 2003), a per-channel `Curves` adjustment primitive with
  three interpolation modes (Linear / Catmull-Rom 1974 / Fritsch-Carlson
  1980 monotone-cubic), a binary-mask `DistanceTransform` (Borgefors
  1986 3-4 chamfer), and `Cyanotype` (vintage blueprint colour remap).
  New filters: `Reinhard` (chroma-preserving log-average luminance
  scaling with the extended `L · (1 + L/white²) / (1 + L)` curve;
  Gray8 / RGB / RGBA), `Hable` (Uncharted-2 filmic rational-function
  curve `((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F` normalised so a
  configurable linear-white point lands at `1.0`; per-channel LUT;
  Gray8 / RGB / RGBA), `Drago` (adaptive logarithmic compression
  `log(Lw+1) / log(2 + 8·(Lw/Lwmax)^(log(bias)/log(0.5)))` with
  per-frame Lwmax adaptation; chroma-preserving; Gray8 / RGB / RGBA),
  `Curves` + `Curve` + `CurveInterpolation` (per-channel tonal curves
  with master + optional R / G / B overrides; Linear / CatmullRom /
  MonotoneCubic interpolants; cost `O(W·H)` via 256-LUT; Gray8 / RGB /
  RGBA + planar YUV with master curve on luma only),
  `DistanceTransform` (3-4 chamfer two-pass distance from binary mask
  with configurable `threshold` / `invert` / `scale`; emits `Gray8`;
  Gray8 input only), and `Cyanotype` (luminance-driven blend between
  shadow + highlight endpoints, defaults to Prussian blue → paper
  white; configurable endpoints, `strength`, alpha pass-through;
  Gray8 input upgraded to RGB; YUV `Unsupported`). Twelve new factory
  names wired into `register()`: `reinhard`, `tonemap-reinhard`
  (alias), `hable`, `tonemap-hable`, `uncharted2` (aliases), `drago`,
  `tonemap-drago` (alias), `curves`, `distance-transform`, `distance`
  (alias), `cyanotype`, `blueprint` (alias). Filter count: 116 → 122
  named types; factory count: 145 → 157 names.

- r21: land six new filters covering quadrant-variance edge-preserving
  smoothing, iterative anisotropic diffusion, two flavours of motion
  blur (radial-outward and rotational-around-centre), a configurable-
  azimuth relief operator, and a two-input warp driven by a colour-
  encoded vector field. New filters: `Kuwahara` (1976 quadrant-variance
  edge-preserving smoothing — splits each `(2*radius+1)²` window into
  four overlapping quadrants and writes the mean of the lowest-luma-
  variance quadrant, so flat regions still blur cleanly while step edges
  survive; Gray8 / RGB / RGBA, alpha pass-through), `AnisotropicBlur`
  (Perona-Malik 1990 anisotropic diffusion with the Lorentzian edge-
  stopping function `g(s) = 1 / (1 + (s/κ)²)`; iterative four-neighbour
  explicit-Euler — smooth areas diffuse, edges freeze; `iterations` /
  `kappa` / `lambda` knobs with `λ ≤ 0.25` enforced for stability;
  Gray8 / RGB / RGBA, alpha pass-through), `ZoomBlur` (radial outward
  "warp drive" blur sampling along the line from each pixel to a
  configurable centre with geometric step `1 - t · strength`; bilinear
  sampling, edge-clamped; Gray8 / RGB / RGBA), `RadialBlur` (rotational
  / "spin" blur around a configurable centre — samples are bilinearly
  fetched along the arc swept by ±`angle/2` around each pixel's base
  polar angle; complement to `ZoomBlur`; Gray8 / RGB / RGBA),
  `EmbossDirectional` (3×3 relief with a configurable light azimuth
  in degrees CCW from East and a `depth` multiplier; kernel weights are
  `(dx, -dy) · (cos a, sin a)`, so a 180° azimuth flip inverts the
  embossed polarity; distinct from the existing fixed-kernel `Emboss`;
  Gray8 / RGB / RGBA, alpha pass-through, + planar YUV on the luma
  plane only), and `DisplacementMap` (two-input warp — the `dst` map's
  R / G channels carry per-pixel `(dx, dy)` vectors re-centred to
  `[-0.5, 0.5]` and scaled by `scale_x` / `scale_y`; the source is
  bilinear-sampled at the perturbed coordinate; classic water-ripple /
  heat-haze / frosted-glass primitive; both inputs must share
  `(width, height, format)`; Gray8 / RGB / RGBA). Eight new factory
  names wired into `register()`: `kuwahara`, `anisotropic-blur`,
  `anisotropic` (alias), `zoom-blur`, `radial-blur`, `spin-blur`
  (alias), `emboss-directional`, `displacement-map`. Filter count:
  110 → 116 named types; factory count: 137 → 145 names.

- r20: land six new filters covering Voronoi-cell averaging, AM-screen
  halftone, arbitrary gradient remap, per-hue-band HSL adjustment,
  film cross-processing emulation, automatic global threshold, plus a
  comic-book stylise pipeline. New filters: `Crystallize`
  (Voronoi-cell averaging on a jittered point lattice — `cell_size` /
  `jitter` / deterministic `seed`; 3×3 cell-neighbourhood scan keeps
  the work `O(W·H)`; Gray8 / RGB / RGBA), `Halftone` (amplitude-
  modulated dot screening — each cell's luma drives a centred-dot
  radius `(cell/2) · sqrt(coverage)` so ink area tracks coverage;
  `ink_color` / `paper_color` configurable; Gray8 / RGB / RGBA),
  `GradientMap` + `GradientStop` (recolour by per-pixel luminance ⇒
  arbitrary `(position, RGB)` gradient sample; convenience
  constructors `duotone(...)` and `tritone(...)`; Gray8 input is
  upgraded to RGB on output; RGBA alpha preserved; distinct from
  `Heatmap` which only ships fixed built-in ramps), `SelectiveColor`
  + `HueBand` + `BandAdjust` (per-hue-band HSL shifts for Reds /
  Yellows / Greens / Cyans / Blues / Magentas at 60° spacing;
  triangular weighting across band boundaries; achromatic pixels
  pass through; RGB / RGBA only), `CrossProcess` (analogue film
  cross-processing emulation via three independent per-channel
  sigmoid S-curves — warm-biased R, neutral G, cool-biased B; folded
  into per-channel LUTs so cost is `O(W·H)`; `amount` / `warmth` /
  `contrast` knobs; Gray8 / RGB / RGBA / YUV-luma), `OtsuThreshold`
  (1979 Otsu global automatic threshold maximising inter-class
  variance; tie-breaks plateaus at their midpoint so bimodal cuts
  land between the modes; optional `invert`; emits binary `Gray8`;
  Gray8 / RGB / RGBA / planar YUV input), and `Comic` (manga /
  comic-book stylise: edge-preserving bilateral-style smooth +
  per-channel quantise for the flat fill, Sobel-on-luma + threshold
  for the ink overlay; distinct from `Toon` which skips the
  pre-smooth; `smooth_radius` / `colour_sigma` / `levels` /
  `edge_threshold` / `ink_color`; RGB / RGBA only). Twelve new
  factory names wired into `register()`: `crystallize`, `halftone`,
  `gradient-map`, `duotone` (alias), `selective-color`,
  `selective-colour` (alias), `cross-process`, `crossprocess`
  (alias), `otsu-threshold`, `otsu` (alias), `comic`, `manga`
  (alias). Filter count: 103 → 110 named types (seven new filter
  structs: `Crystallize`, `Halftone`, `GradientMap`, `SelectiveColor`,
  `CrossProcess`, `OtsuThreshold`, `Comic`; plus public companion
  types `GradientStop`, `HueBand`, `BandAdjust`); factory count:
  125 → 137 names.

- r19: land six new filters covering false-colour visualisation,
  shadow / highlight tint, per-channel posterise, cel-shading, seeded
  region fill, and two-input watermark overlay. New filters:
  `Heatmap` + `HeatmapRamp` (apply a built-in colour ramp — `Jet` /
  `Viridis` / `Plasma` / `Hot` / `Cool` / `Grayscale` — to a
  luminance-reduced source; input may be Gray8 / RGB / RGBA / planar
  YUV; output is always RGB24 or RGBA when the source was RGBA, with
  alpha preserved), `SplitTone` (independent shadow + highlight tints
  via a triangular tonal mask `w(v) = (1 - 2|v - 0.5|).max(0)` peaking
  at the extremes; `balance` biases shadow vs highlight; RGB / RGBA),
  `FloodFill` (seeded scanline flood-fill with per-channel Chebyshev
  tolerance — matching the documented `-fuzz` CLI; iterative span-fill keeps
  the work `O(W·H)` and rejects out-of-bounds seeds; Gray8 / RGB /
  RGBA), `PosterizeChannels` (per-channel posterise with independent
  `r_levels` / `g_levels` / `b_levels` and optional `alpha_levels`;
  distinct from the existing `Posterize` which collapses every channel
  to the same level count; Gray8 / RGB / RGBA), `Toon` (cel-shaded
  "cartoon" look — per-channel posterise + Sobel edge overlay with
  configurable ink colour; RGB / RGBA), and `Watermark` (two-input
  over-place of a secondary image onto the source at
  `(offset_x, offset_y)` with a `[0, 1]` `opacity` multiplier — the
  overlay shape is independent of the source and clipped on negative
  or oversize placements; straight-alpha on RGBA; the two ports must
  share `PixelFormat`). Eight new factory names wired into
  `register()`: `heatmap`, `false-color` (alias), `split-tone`,
  `flood-fill`, `posterize-channels`, `toon`, `cartoon` (alias),
  `watermark`. Filter count: 97 → 103 named types; factory count:
  117 → 125 names.

- r18: land seven new filters covering Hough-space circle detection,
  arithmetic differencing, content-aware cropping, layered shadow
  effects, additive glow, and runtime-selectable edge kernels. New
  filters: `HoughCircles` (3-D Hough accumulator indexed by
  `(radius, cx, cy)` over Sobel-magnitude voters; emits the rendered
  circle trace on `Gray8`; the radius-axis complement to the existing
  `HoughLines`), `Difference` (two-input pixel-wise `|src - dst|` —
  cheap change-detection / motion mask, no Porter–Duff coverage
  algebra; Gray8 / RGB / RGBA / planar YUV), `AutoTrim` (like `Trim`
  but picks the dominant colour via a 4-bit-per-channel histogram
  vote first, so a noisy `(0, 0)` corner sample no longer poisons the
  inferred background), `DropShadow` (soft offset shadow underlay
  composited *behind* opaque RGBA subject pixels — straight-alpha
  `over`; box-blur approximates Gaussian for cheap halos), `InnerShadow`
  (soft offset shadow rendered *inside* the subject coverage via an
  inverted-alpha mask + offset + blur), `Bloom` (Gaussian-blurred
  highlights additively composited on top of the source — emissive /
  HDR-look glow; Gray8 / RGB / RGBA), and `EdgeDetect` + `EdgeKernel`
  (runtime-selectable gradient kernel — `Sobel` / `Prewitt` / `Scharr`
  / `Roberts`; complements the fixed-Sobel `Edge`). Nine new factory
  names wired into `register()`: `hough-circles`, `auto-trim`,
  `drop-shadow`, `inner-shadow`, `bloom`, `glow` (alias), `edge-detect`,
  `edge-multi` (alias), `difference`. Filter count: 90 → 97 named
  types; factory count: 108 → 117 names.

- r17: land seven new filters covering linear-light exposure
  control, colourist tonal sliders, B&W conversion presets, and a
  flat solid-colour border. New filters: `Exposure` (EV-stop
  adjustment in linear-light space — sRGB → linear → multiply by
  `2^EV` → sRGB, folded into a single 256-entry LUT; `+1.0 EV`
  doubles captured light, `-1.0 EV` halves it; RGB / RGBA / Gray8),
  `Temperature` (warmth slider in `[-1, 1]` driving per-channel R / B
  multipliers up to ±50 % — G and alpha pass through; per-channel
  LUT so the cost is `O(W·H)` regardless of `warmth`; RGB / RGBA),
  `Vibrance` (photo-editor-style saturation boost that spares already-saturated
  pixels via `1 - s` per-pixel weighting — visually distinct from
  `Modulate`'s flat S-multiplier; RGB / RGBA only), `BwMix`
  (black-and-white conversion with per-channel weights; convenience
  constructors `red_filter()` / `green_filter()` / `blue_filter()`
  for the classic photo-filter recipes; default Rec. 601 matches
  `Grayscale`; opt-in `keep_format` flag emits grey-equalled RGB /
  RGBA instead of single-plane `Gray8`), `Clarity` (mid-frequency
  local-contrast boost via large-radius / moderate-amount unsharp
  mask — defaults `radius = 30`, `sigma = 15.0`, `amount = 0.5`,
  `threshold = 10`; delegates to `Unsharp` so YUV-luma-only / RGB-
  all-planes semantics match), `ShadowHighlight` (independent
  shadow lift + highlight recovery gated by a soft tonal mask that
  peaks at the extremes and is zero at midtone — midgrey passes
  through unchanged; RGB / RGBA / Gray8 / YUV-luma-only), and
  `BorderedFrame` (flat solid-coloured border with independent
  per-side widths; distinct from `Frame`'s 3-D bevel; Gray8 / RGB /
  RGBA — IM analogue `-bordercolor C -border WxH`). Nine new factory
  names wired into `register()`: `exposure`, `temperature`,
  `vibrance`, `bw-mix`, `black-and-white` (alias for `bw-mix`),
  `clarity`, `shadow-highlight`, `bordered-frame`, `border` (alias).
  Filter count: 83 → 90 named types; factory count: 99 → 108 names.

- r16: land seven new filters expanding generators, edge-preserving
  smoothing, and colourist tonal control. New filters: `BilateralBlur`
  (joint spatial + range Gaussian — edges are weighted by both pixel
  distance and intensity similarity, so step edges survive a heavy
  smoothing pass; alpha pass-through on RGBA; IM analogue
  `-define blur:bilateral=1` on the Gaussian path), `Canvas`
  (constant-colour generator — every output sample is the configured
  `[R, G, B, A]`; YUV chroma planes are painted neutral 128;
  `Gray8 / Rgb24 / Rgba` + planar YUV; the input frame is consumed
  only for its `pts` so `Canvas` works as a pipeline-head generator),
  `ColorBalance` (three-way ASC CDL-style per-channel `lift` /
  `gamma` / `gain` folded into a single 256-entry per-channel LUT —
  `O(W·H)` regardless of how many adjustments are enabled; alpha
  pass-through on RGBA), `GradientRadial` (radial gradient generator
  centred at `(cx, cy)` with explicit or default-half-diagonal
  `radius`; per-channel linear interpolation between an inner and
  outer colour; RGB / RGBA / Gray8), `GradientConic` (angular sweep
  generator parameterised by `(cx, cy)` + `start_angle`; `t = 0` lies
  along the angle ray and increases counter-clockwise round to 1;
  RGB / RGBA / Gray8), `GravityTranslate` (9-point compass anchor
  placement on a fixed canvas — `NorthWest` / `North` / ... /
  `SouthEast` plus `Centre`; backed by `Extent` so YUV chroma
  alignment matches; IM `-gravity` operator), and `HslShift`
  (independent additive shifts for H (degrees), S, and L (each in
  `[-1, 1]`) via the same RGB ⇄ HSL round-trip as `HslRotate`; alpha
  pass-through; RGB / RGBA only). Ten new factory names wired into
  `register()`: `bilateral-blur`, `canvas`, `gradient-radial`,
  `radial-gradient` (alias), `gradient-conic`, `conic-gradient`
  (alias), `gravity-translate`, `gravity` (alias), `color-balance`,
  `hsl-shift`. Filter count: 76 → 83 named types; factory count:
  89 → 99 names.

- r15: land six new documented-CLI-compatible filters covering colour
  transforms, photographic stylise, and adaptive thresholding. New
  filters: `HslRotate` (rotate the hue channel by N degrees via an
  RGB ⇒ HSL ⇒ rotate ⇒ RGB round-trip; achromatic greys are passed
  through; RGB / RGBA only — IM has no direct one-liner, this is a
  textbook recipe), `VignetteSoft` (raised-cosine soft vignette with
  separate inner / outer normalised radii — smoother seam than the
  Gaussian `Vignette` because both endpoints have zero derivative;
  RGB / RGBA only), `ChromaticAberration` (per-channel pixel offset
  on R and B with G untouched; convenience constructors
  `horizontal(n)` / `vertical(n)` plus explicit `(r_dx, r_dy, b_dx,
  b_dy)` overrides; RGB / RGBA only), `Pixelate` (block-average
  spatial mosaic; each `N×N` tile collapses to its mean colour;
  Gray8 / RGB / RGBA + planar YUV with luma-only pixelation),
  `ChannelMixer` (4×4 linear combination of source channels into
  destination channels plus a 4-vector offset; super-set of
  `ColorMatrix` because it also touches the alpha row; convenience
  constructors `identity()` / `sepia_classic()` / `from_color_matrix(m)`;
  RGB / RGBA only), and `AdaptiveThreshold` (local-mean-based
  threshold with a configurable `(2*radius+1)²` window and signed
  offset; separable box-sum implementation gives `O(W*H)` cost
  regardless of radius; always emits Gray8 — IM `-threshold local`).
  Seven new factory names wired into `register()`: `hsl-rotate`,
  `vignette-soft`, `chromatic-aberration`, `pixelate`, `mosaic`
  (alias for `pixelate`), `channel-mixer`, `adaptive-threshold`.
  Filter count: 70 → 76 named types; factory count: 82 → 89 names.

- r14: land six new documented-CLI-compatible filters covering edge-aware
  stylise, geometric correction, and stereo / steganography composition.
  New filters: `Sketch` (pencil-sketch stylise: directional motion blur
  + Sobel edge magnitude + invert; IM `-sketch radius,sigma,angle`),
  `Deskew` (auto-deskew via histogram-variance scoring of rotated ink
  projections then corrective rotation; IM `-deskew threshold`),
  `HoughLines` (polar Hough-transform line detection on Sobel edge
  magnitude with top-K peak picking and full-image line rendering on
  `Gray8`; IM `-hough-lines WxH`), `BarrelInverse` (inverse-polynomial
  radial distortion `r → r / (a·r³+b·r²+c·r+d)` — the pair to the
  existing forward-radial `Distort`; IM `-distort BarrelInverse
  a,b,c,d`), `Stegano` (two-input LSB-plane steganographic embed —
  stamps `src`'s configurable bit position into `dst`'s LSB per
  channel, alpha pass-through; IM `-stegano offset`), and `Stereo`
  (two-input red/cyan anaglyph: R from `src`, G/B from `dst`, alpha
  from `src`; IM `-stereo`). Six new factory names wired into
  `register()`: `sketch`, `deskew`, `hough-lines`, `barrel-inverse`,
  `stegano`, `stereo`. Filter count: 64 → 70 named types; factory
  count: 76 → 82 names.

- r13: land seven new documented-CLI-style filters: five single-input
  effects + two two-input colour-grading CLUTs. New filters:
  `BlueShift` (moonlight / scotopic-vision tint, per-pixel
  `(min/factor, min/factor, max/factor)`, IM `+blue-shift factor`),
  `Frame` (decorative bordered frame with a 3-D bevel — top / left
  highlight, bottom / right shadow — RGB / RGBA only, IM
  `-frame WxH+inner+outer-mat`), `Shade` (directional Lambertian
  relief shading from an `(azimuth, elevation)` light vector;
  optional colour pass-through `+shade` mode, IM `-shade az,el`),
  `Paint` (oil-paint stylise: per-pixel modal-bucket vote in a
  `(2*radius+1)²` window then mean-of-mode RGB, IM `-paint radius`),
  `Quantize` (uniform-grid colour quantizer: round each channel to
  one of `cbrt(N)` evenly-spaced palette entries, IM `-colors N`),
  `Clut` (two-input 1-D Colour Look-Up Table — `src` is the image,
  `dst` is the CLUT read row-major; per-channel index lookup; alpha
  pass-through, IM `-clut`), and `HaldClut` (two-input Hald CLUT
  image-as-LUT — `dst` is a `(L²)×(L²)` Hald cube; trilinear
  sampling per pixel; RGB / RGBA only, IM `-hald-clut`). Seven new
  factory names wired into `register()`: `blue-shift`, `frame`,
  `shade`, `paint`, `quantize`, `clut`, `hald-clut`. Filter count:
  57 → 64 named types; factory count: 69 → 76 names.

- r12: land seven new documented-CLI-style filters covering tonal range
  control, colour matrices, parametric pixel-function maps, and
  spectral edge detectors. New filters: `Clamp` (clamp every tone
  sample into `[low, high]` with alpha / chroma preservation, IM
  `-clamp`), `AutoLevel` (per-channel auto-stretch — independently
  fill `[0, 255]` for each of R / G / B, IM `-auto-level`),
  `ContrastStretch` (burn `black%` darkest + `white%` brightest
  pixels per channel before linear remap, IM
  `-contrast-stretch black%xwhite%`), `LinearStretch` (same algorithm
  with absolute pixel-count cut-offs, IM `-linear-stretch`),
  `ColorMatrix` (3×3 colour matrix with optional 3-vector offset,
  RGB / RGBA only, IM `-color-matrix` / `-recolor`), `Function` +
  `FunctionOp` (per-pixel mathematical map evaluated in normalised
  `[0, 1]` space — `Polynomial` / `Sinusoid` / `ArcSin` / `ArcTan`,
  IM `-function <kind> args`), `Laplacian` (3×3 second-derivative
  Laplacian → `Gray8`, IM `-laplacian`), and `Canny` (textbook
  4-step pipeline: Gaussian → Sobel + direction → non-max
  suppression → hysteresis → binary `Gray8`, IM
  `-canny RxS+L%+H%`). Nine new factory names wired into
  `register()`: `clamp`, `auto-level`, `contrast-stretch`,
  `linear-stretch`, `color-matrix`, `recolor` (alias for
  `color-matrix`), `function`, `laplacian`, `canny`. Filter count:
  50 → 57 named types; factory count: 60 → 69 names.

- r11: land five new documented-CLI-style filters plus four overlay-family
  composite operators. New filters: `Evaluate` (per-pixel arithmetic
  LUT — `Add` / `Subtract` / `Multiply` / `Divide` / `Pow` / `Max` /
  `Min` / `Set` / `And` / `Or` / `Xor` / `Threshold`, IM
  `-evaluate <op> N`), `Cycle` (modular per-channel value rotation
  `out = (src + amount) mod 256`, IM analogue `-cycle N`),
  `Statistic` (rolling-window `Median` / `Min` / `Max` / `Mean` over
  a `WxH` neighbourhood, IM `-statistic <op> WxH`), `Affine`
  (six-coefficient 2-D affine warp with bilinear resampling, IM
  `-distort Affine "sx,ry,rx,sy,tx,ty"`), and `Srt` (Scale / Rotate
  / Translate composite warp collapsing to a single 2×3 affine
  matrix, IM `-distort SRT "ox,oy sx[,sy] angle tx,ty"`). The
  `Composite` family also gains `HardLight`, `SoftLight`,
  `ColorDodge`, `ColorBurn` (PDF 1.7 §11.3.5 conventions, with
  Pegtop's smooth soft-light formula). Five new single-input
  factories wired into `register()`: `evaluate`, `cycle`,
  `statistic`, `affine`, `srt`. Four new two-input factories:
  `composite-hardlight`, `composite-softlight`,
  `composite-colordodge`, `composite-colorburn`. Filter count:
  45 → 50 named types; factory count: 51 → 60 names.

- r10: land six documented-CLI-style geometric / channel / morphology
  filters: `Roll` (circular pixel shift, IM `-roll +X+Y`), `Shave`
  (uniform border trim, IM `-shave XxY`), `Extent` (set canvas to
  `WxH+X+Y` with configurable background, IM `-extent`), `Trim`
  (auto-crop to non-background bbox with `fuzz` tolerance, IM
  `-fuzz N% -trim`), `ChannelExtract` (pull one channel out as a
  single-plane `Gray8` frame, IM `-channel <ch> -separate`), and
  `MorphologyEdge` (morphological edge / gradient operators —
  `EdgeIn`, `EdgeOut`, `EdgeMagnitude` — built on the existing
  dilate / erode primitives, IM `-morphology EdgeIn|EdgeOut|Edge`).
  Eight new factory names wired into `register()`: `roll`, `shave`,
  `extent`, `trim`, `channel-extract`, `morphology-edgein`,
  `morphology-edgeout`, `morphology-edge-magnitude`. Filter count:
  39 → 45 named types; factory count: 43 → 51 names (+8 new
  factories on top of the existing twelve `composite-<op>` ones).

- r9: rotate the `TiltShift` focus band around the image centre via a
  new `angle_degrees: f32` parameter (default `0.0` = the legacy
  horizontal band; `90.0` = vertical band; any finite angle works).
  The pre-r9 row-only fast path is kept for `angle_degrees == 0` so
  existing horizontal-band outputs stay bit-exact identical; non-zero
  angles fall back to a per-pixel `focus_weight_xy` evaluator. New
  public surface: `TiltShift::angle_degrees` field,
  `TiltShift::with_angle_degrees(angle)` builder, JSON keys
  `angle_degrees` (long) + `angle` (alias) on the `tilt-shift`
  factory.

- r9: extend `Distort` with the standard Brown-Conrady tangential
  ("decentering") coefficients `p1` and `p2`. Both default to `0.0`
  — the pre-r9 pure-radial output is bit-exact preserved when
  tangential coefficients are not supplied. Non-zero values warp the
  image off-axis (the asymmetry produced when lens elements are not
  perfectly aligned with the sensor). New public surface:
  `Distort::p1`, `Distort::p2` fields, `Distort::with_tangential(p1, p2)`
  builder, JSON keys `p1` + `p2` on the `distort` factory. Source-
  rectangle bounds checks pick up a tiny epsilon so floating-point
  round-off on identity back-maps doesn't kick on-boundary pixels
  into the background fill.

- r9: add an optional output canvas size to `Perspective` (new
  `output_size: Option<(u32, u32)>` field + `Perspective::with_output_size`
  builder; new JSON keys `output_width` + `output_height`, both
  required together). When set, the filter emits a `w × h` frame
  regardless of the input shape — pixels outside the destination
  quad's footprint receive the configured background colour. The
  `make_perspective` factory rebuilds the output port spec to the new
  shape so downstream pipeline nodes see the updated dimensions.
  Source-rectangle bounds checks pick up a tiny epsilon to keep
  floating-point round-off from kicking on-boundary back-maps into
  the background fill. Three new tests
  (`output_size_grows_canvas_and_preserves_quad_mapping`,
  `output_size_shrink_canvas_clamps_to_quad_footprint`,
  `perspective_factory_accepts_output_size_keys`,
  `perspective_factory_rejects_partial_output_size`).

- r9: extend `Charcoal` with an optional Gaussian pre-blur (new
  `radius: u32` field + `Charcoal::with_radius` builder; new JSON key
  `radius`). Default `radius = 0` is bit-exact identical to the r6
  charcoal output (no pre-blur). Larger radii thicken the strokes by
  smoothing fine texture before the Sobel pass. The pre-blur uses a
  separable Gaussian with sigma = `radius / 2` (matching `Blur`'s
  default heuristic), edge-clamped at the borders.

- r8: introduce a two-input filter contract (`TwoInputImageFilter`
  trait + `TwoInputImageFilterAdapter` `StreamFilter` shim) and land
  the Porter–Duff and arithmetic `Composite` family. Twelve operator
  variants — `Over`, `In`, `Out`, `Atop`, `Xor`, `Plus`, `Multiply`,
  `Screen`, `Overlay`, `Darken`, `Lighten`, `Difference` — each
  registered as a `composite-<op>` factory. The adapter buffers the
  most-recent frame from each input port (`0 = src`, `1 = dst`) and
  emits whenever both are in hand; mismatched input shapes are
  rejected at factory time. Operates on `Gray8` / `Rgb24` / `Rgba`
  with straight (non-premultiplied) alpha; YUV / planar / higher-bit
  formats return `Unsupported`. Each operator has a 2×2 fixture-pair
  unit test through the registered factory plus per-pixel arithmetic
  unit tests in the `composite` module. Filter count: 38 → 39 named
  types (12 new factories).
- r7: implement `Morphology` (greyscale dilate / erode with a 3×3 square
  or cross structuring element + N iterations) and `MorphologyChain`
  (open / close composition). Wires four factory names —
  `morphology-dilate`, `morphology-erode`, `morphology-open`,
  `morphology-close` — into `register()`. Supports Gray8 / Rgb24 / Rgba
  (alpha pass-through) and planar YUV; out-of-bounds taps clamp to the
  nearest edge.
- r7: implement `TiltShift` — selective Gaussian blur masked by a
  horizontal in-focus band (miniature-photography depth-of-field).
  Builds the blurred reference frame via the existing `Blur` filter
  then per-row blends `weight · sharp + (1 - weight) · blur`, where
  weight is `1.0` inside the focus band and ramps linearly to `0.0`
  across `falloff_height`. Supports Gray8 / Rgb24 / Rgba (alpha
  pass-through) / planar YUV. JSON schema: `focus_centre`,
  `focus_height`, `falloff_height` (or short aliases `centre` /
  `height` / `falloff`), plus `blur_radius` / `blur_sigma` (or short
  aliases `radius` / `sigma`).
- r7: surface `cx` / `cy` / `max_radius` (alias `max_r`) overrides
  through the JSON schema for `polar` and `depolar` factories. Also
  exposes new `Polar::with_centre` / `Polar::with_max_radius` builder
  methods (the previous defaults — image centre + farthest-corner
  radius — remain unchanged when no override is supplied).
- r7: implement `Distort` — radial-polynomial barrel / pincushion lens
  distortion. Inverse-maps each output pixel through the polynomial
  `r' = r · (1 + k1·r² + k2·r⁴)` (with `r` normalised against
  `min(w, h) / 2` by default; overridable via `with_r_norm`) then
  bilinear-samples. Convenience constructors `Distort::barrel(s)` /
  `Distort::pincushion(s)` set `k1` directly. Operates on Rgb24 / Rgba.
  Wires the `distort` factory into `register()` accepting either
  `mode` + `strength` sugar or explicit `k1` / `k2` coefficients, plus
  optional `cx`/`cy`, `r_norm`, and `background` keys.
- r7: implement `Perspective` — 4-corner perspective warp. Solves the
  3×3 homography between source and destination quads via 8×8
  Gauss-Jordan elimination, then inverse-maps every output pixel
  through `H⁻¹` with bilinear sampling. Out-of-bounds back-maps fall
  back to a configurable `with_background([R, G, B, A])` colour
  (default opaque black). Operates on Rgb24 / Rgba; rejects YUV /
  planar / higher-bit formats. Wires the `perspective` factory into
  `register()` accepting either flat 8-element or nested 4×2 corner
  arrays.
- Wire `sharpen`, `gamma`, `brightness-contrast`, `brightness`, and `contrast`
  filter factories into the runtime registry so pipeline jobs and
  oxideav-cli-convert IM-flag mapping can resolve them by name.
- Wire 16 additional filter factories into `register()`: `unsharp`,
  `threshold`, `level`, `normalize`, `posterize`, `solarize`, `flip`, `flop`,
  `rotate`, `crop`, `negate`, `sepia`, `modulate`, `grayscale`,
  `motion-blur`, and `emboss`. Output port shape is recomputed when the
  filter changes dimensions/format (`crop`, `rotate`, `grayscale` with
  `output_gray8`). One smoke test per newly-wired filter exercises the
  factory + adapter `push` path on a 4×4 fixture.
- `Crop` is now re-exported from the crate root.
- Implement four new documented-CLI-style filters and wire factories into
  `register()`: `Vignette` (Gaussian radial darkening; Rgb24/Rgba),
  `Colorize` (linear blend toward a target colour by `amount`;
  Rgb24/Rgba), `Equalize` (per-channel CDF histogram equalisation;
  Gray8/Rgb24/Rgba/YUV planar), and `AutoGamma` (geometric-mean-driven
  per-channel gamma correction; Gray8/Rgb24/Rgba/YUV planar). Alpha is
  preserved on Rgba; YUV runs only on the luma plane for Equalize /
  AutoGamma.
- Implement five more documented-CLI-style filters and wire factories into
  `register()`: `Tint` (luminance-weighted blend toward a target colour;
  Rgb24/Rgba), `SigmoidalContrast` (sigmoid-curve contrast adjustment
  via 256-entry LUT; Gray8/Rgb24/Rgba/YUV-luma), `Implode` (radial
  pinch / explode bilinear-resampled; Rgb24/Rgba), `Swirl`
  (radius-decaying rotational distortion; Rgb24/Rgba), and `Despeckle`
  (median-window edge-preserving noise reduction; Rgb24/Rgba with
  alpha pass-through). Each filter has identity-case bit-exact
  pass-through tests.
- r6: implement five new filters and wire factories into `register()`
  (six factory names; `Polar` ships under both `polar` and `depolar`):
  `Wave` (sinusoidal vertical pixel displacement; Rgb24/Rgba),
  `Spread` (random pixel perturbation with a deterministic PRNG;
  Rgb24/Rgba), `Charcoal` (Sobel-on-luma + invert ⇒ Gray8 sketch;
  Gray8/Rgb24/Rgba/YUV planar), `Convolve` (user-supplied square `N×N`
  kernel with optional bias / divisor; Gray8/Rgb24/Rgba, alpha
  pass-through on RGBA), and `Polar` / `DePolar` (Cartesian ⇄ polar
  coordinate distortion with bilinear sampling; Rgb24/Rgba).

## [0.1.1](https://github.com/OxideAV/oxideav-image-filter/compare/v0.1.0...v0.1.1) - 2026-05-06

### Other

- reframe FFI claim — HW-engine crates use OS FFI by necessity
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- release v0.1.0

## [0.1.0](https://github.com/OxideAV/oxideav-image-filter/compare/v0.0.4...v0.1.0) - 2026-05-03

### Other

- promote to 0.1

## [0.0.4](https://github.com/OxideAV/oxideav-image-filter/compare/v0.0.3...v0.0.4) - 2026-05-03

### Other

- drop unused TimeBase imports + ignore unused let-binds in tests
- cargo fmt: pending rustfmt cleanup
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- stay on 0.1.x during heavy dev (semver_check=false)
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

## [0.0.3](https://github.com/OxideAV/oxideav-image-filter/compare/v0.0.2...v0.0.3) - 2026-04-25

### Other

- don't commit Cargo.lock for library crates
- drop Cargo.lock — library crate, lock causes CI drift
- add `register(&mut RuntimeContext)` + adopt image-filter factories

## [0.0.2](https://github.com/OxideAV/oxideav-image-filter/compare/v0.0.1...v0.0.2) - 2026-04-24

### Other

- rustfmt pass + README now lists all 19 filters
- add MotionBlur
- add Solarize
- fix emboss clippy warnings
- add Posterize
- add Emboss
- add Crop
- add Normalize
- add Unsharp
- add Sharpen
- add Level
- add Grayscale
- add BrightnessContrast
- add Sepia
- add Gamma
- add Modulate
- add Flop
- add Threshold
- add Flip
- add Negate
