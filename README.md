# oxideav-image-filter

Pure-Rust single-frame image filters for the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates).

This crate specialises in **stateless per-frame effects** — there is no
frame-to-frame context. That's what separates it from sibling crates:

- `oxideav-audio-filter` carries streaming state (echo, resample).
- `oxideav-pixfmt` does colour-space conversion / palette / dither.
- `oxideav-image-filter` (this crate) takes a `VideoFrame` and returns a
  new `VideoFrame` — same filter, same input → same output.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-image-filter = "0.0"
```

## Filters

### Blur

Separable Gaussian blur with configurable radius + sigma and an optional
plane selector:

```rust
use oxideav_image_filter::{Blur, ImageFilter, Planes};

// Blur all planes with radius 3, sigma 1.5 (default).
let f = Blur::new(3).with_sigma(1.5);
let out = f.apply(&input_frame)?;

// Blur only the luma plane — leave chroma untouched.
let f = Blur::new(5).with_sigma(2.0).with_planes(Planes::Luma);
```

### Edge

3×3 Sobel edge-magnitude; output is a `Gray8` frame of the same size:

```rust
use oxideav_image_filter::{Edge, ImageFilter};

let edges = Edge::new().apply(&input_frame)?;
```

For `Rgb24` / `Rgba` input the filter first computes a luma proxy
`Y = (R + 2G + B) / 4` and applies Sobel to it. YUV inputs use plane 0
directly.

### Resize

Rescale to arbitrary dimensions, nearest-neighbour or bilinear:

```rust
use oxideav_image_filter::{Interpolation, ImageFilter, Resize};

let half = Resize::new(input.width / 2, input.height / 2)
    .with_interpolation(Interpolation::Bilinear)
    .apply(&input)?;
```

For planar YUV the chroma planes are resized to the matching subsampled
output dimensions (e.g. 4:2:0 halves both chroma axes).

### Geometric

- **`Flip`** — mirror vertically (top ↔ bottom).
- **`Flop`** — mirror horizontally (left ↔ right).
- **`Rotate`** — arbitrary-degree rotation with bilinear resampling and
  canvas grow; configurable background colour. 90°-multiples use an
  exact axis-aligned fast path. documented CLI: `-rotate N`.
- **`Crop`** — `(x, y, width, height)` subregion extraction; chroma
  rects are ceil/floor-aligned for YUV subsampling. documented CLI: `-crop WxH+X+Y`.
- **`Roll`** — circular pixel shift `(dx, dy)`; rows / columns wrap
  around the borders. Chroma offsets on planar YUV are scaled by the
  subsampling factor so the visible image translates as a rigid block.
  documented CLI: `-roll +X+Y`.
- **`Shave`** — strip a uniform `(x_border, y_border)` margin off
  every edge (centred crop). Backed by `Crop` so YUV chroma alignment
  matches. documented CLI: `-shave XxY`.
- **`Extent`** — set the output canvas to a fixed `(width, height)`
  with a placement offset, padding the gaps with a configurable
  background colour. Negative offsets translate the input toward the
  upper-left so its right / bottom edge can land inside the window.
  documented CLI: `-extent WxH+X+Y`.
- **`Trim`** — auto-crop to the bounding box of pixels that differ
  from a reference background colour by more than `fuzz` per channel
  (Chebyshev / L-infinity tolerance). Background defaults to the
  source's `(0, 0)` pixel. documented CLI: `-fuzz N% -trim`.
- **`AutoTrim`** — like `Trim`, but picks the dominant background
  colour via a 4-bit-per-channel histogram vote across the whole
  frame instead of trusting the `(0, 0)` corner sample. Noisy
  corners no longer poison the inferred background.
- **`GravityTranslate`** + **`Gravity`** — 9-point compass anchor
  placement on a fixed `(width, height)` canvas (`NorthWest` /
  `North` / `NorthEast` / `West` / `Centre` / `East` / `SouthWest` /
  `South` / `SouthEast`). Backed by `Extent` so YUV chroma
  subsampling alignment matches. Factory aliases: `gravity-translate`,
  `gravity`. documented CLI: `-gravity <anchor> -extent WxH`.

### Tonal (LUT-based, typically fast)

- **`Negate`** — invert pixel values. On YUV inverts only Y so
  chroma (hue/saturation) is preserved. documented CLI: `-negate`.
- **`Threshold`** — binarise at a per-channel threshold; chroma →
  neutral 128 for YUV. documented CLI: `-threshold N`.
- **`Gamma`** — power-law gamma correction. documented CLI: `-gamma G`.
- **`BrightnessContrast`** — linear brightness + contrast in the IM
  range `−100..100`. documented CLI: `-brightness-contrast BxC`.
- **`Level`** — remap `[black, white]` to `[0, 255]` with optional
  mid-tone gamma. documented CLI: `-level LOW,MID,HIGH`.
- **`Normalize`** — two-pass auto-level stretch to use the full
  range. documented CLI: `-normalize`.
- **`Posterize`** — reduce each channel to `N` intensity levels. documented CLI: `-posterize N`.
- **`Dither`** + **`DitherMode`** + **`BayerMatrix`** +
  **`DiffusionKernel`** — bit-depth-reduction dither with two
  algorithm families:
  - **Bayer ordered dither** with tiled `2×2` / `4×4` / `8×8`
    threshold maps built by the standard
    `M_2n = 4·M_n + M_2[hi, lo]` recurrence (Bayer 1973 IEEE ICC).
    No state, perfectly parallel, produces the classic crosshatch
    texture.
  - **Error diffusion** along a left-to-right raster scan: the seven
    canonical kernels Floyd–Steinberg (÷16, 1976 SID),
    Jarvis–Judice–Ninke (÷48, 1976 CGIP), Stucki (÷42, 1981 IBM
    RZ1060), Sierra-3 (÷32), Sierra-2 (÷16), Sierra-Lite (÷4) — the
    three Sierra kernels are community-attributed c. 1989–1990 —
    and Atkinson (÷8 with coefficient sum 6, so 2/8 of each pixel's
    quantisation error is intentionally discarded; Apple, late
    1980s). Per-channel for R / G / B (alpha pass-through on
    RGBA), luma-only on YUV.

  Output bit-depth is configurable via `with_levels`. Default
  `levels = 2` gives a 1-bit black-and-white halftone (Macintosh /
  newsprint / 1-bit-display use case); higher `levels` give a
  multi-tone dithered posterisation (the colour count drops but
  the visible banding does not appear). Factory aliases: `dither`
  (configurable via `{"kernel": "atkinson", "levels": 4}`),
  `dither-floyd-steinberg` / `dither-fs`, `dither-jjn` /
  `dither-jarvis`, `dither-stucki`, `dither-sierra3`,
  `dither-sierra2`, `dither-sierra-lite`, `dither-atkinson`,
  `dither-bayer` / `ordered-dither` (matrix size via
  `{"matrix": 2|4|8}`).

- **`Solarize`** — invert pixels above a threshold. documented CLI: `-solarize N%`.
- **`AdaptiveThreshold`** — local-mean-based binarisation with a
  configurable `(2*radius+1)²` window and signed offset; box-sum
  implementation is `O(W*H)` regardless of radius. Always emits
  `Gray8` (RGB collapsed via Rec. 601 luma). documented CLI: `-threshold local`.
- **`OtsuThreshold`** — global automatic threshold maximising
  inter-class variance (1979 Otsu paper). Optional `invert` flag.
  Emits binary `Gray8` (RGB / RGBA collapsed via Rec. 601 luma; YUV
  reads the Y plane). Complement to `AdaptiveThreshold` (local window)
  and `Threshold` (manual cut). Factory aliases: `otsu-threshold`,
  `otsu`.
- **`Niblack`** — Niblack 1986 adaptive local-statistics threshold
  (Niblack, *An Introduction to Digital Image Processing*, §5.1).
  For each pixel computes the local mean `μ(x, y)` and standard
  deviation `σ(x, y)` over the `(2·radius + 1)²` neighbourhood, then
  binarises against `T(x, y) = μ + k · σ`. Default `k = -0.2` is the
  textbook page-segmentation bias (threshold sits below the local
  mean by a fraction of the local spread so faint ink against a
  brightish, locally-noisy background is reliably captured); positive
  `k` biases the threshold above the mean for the complementary
  light-on-dark case. Separable box-sum implementation runs in
  `O(W · H)` regardless of `radius`. Optional `invert` flag swaps the
  two output levels. Emits binary `Gray8` (luma-collapses Gray8 / RGB
  / RGBA / planar YUV via the existing edge-family helper). Joins
  the segmentation family at the "local mean + local σ threshold"
  position complementing `AdaptiveThreshold` (local mean only) and
  `OtsuThreshold` (global automatic cut). Factory aliases: `niblack`,
  `niblack-threshold`.
- **`Equalize`** — per-channel histogram equalisation via CDF
  mapping. Luma-only on YUV. documented CLI: `-equalize`.
- **`AutoGamma`** — pick a per-channel gamma so the geometric mean
  lands at mid-grey 0.5 (`gamma = log(mean) / log(0.5)`). documented CLI: `-auto-gamma`.
- **`SigmoidalContrast`** — sigmoid-curve contrast around a midpoint.
  documented CLI: `-sigmoidal-contrast CxM%`.
- **`Exposure`** — EV-stop adjustment in linear-light space (sRGB →
  linear → ×2^EV → sRGB). `+1.0` EV doubles the captured light,
  `-1.0` EV halves it. RGB / RGBA / Gray8. documented-CLI analogue: `-colorspace
  RGB -evaluate Multiply k -colorspace sRGB` folded into one LUT pass.
- **`ShadowHighlight`** — independent shadow lift + highlight recovery
  gated by a soft tonal mask (peaks at the extremes, zero at midtone).
  Mid-grey passes through unchanged. RGB / RGBA / Gray8 / YUV (luma
  only). Photo-editor-style "Shadows" / "Highlights" sliders.
- **`Evaluate`** + **`EvaluateOp`** — per-pixel arithmetic LUT with a
  single scalar operand. Operators: `Add`, `Subtract`, `Multiply`,
  `Divide`, `Pow`, `Max`, `Min`, `Set`, `And`, `Or`, `Xor`,
  `Threshold`. Alpha (RGBA) and chroma (YUV) pass through. documented CLI: `-evaluate <op> N`.
- **`Cycle`** — modular per-channel value rotation
  (`out = (src + amount) mod 256`); alpha and chroma preserved. IM
  analogue: `-cycle N`.
- **`Statistic`** + **`StatisticOp`** — rolling-window per-pixel
  `Median` / `Min` / `Max` / `Mean` over a `WxH` neighbourhood
  (border-clamped). 1×1 window is identity. documented CLI: `-statistic <op> WxH`.
- **`Clamp`** — clamp every tone sample into `[low, high]`. Alpha
  preserved on RGBA; YUV touches only luma. documented CLI: `-clamp` (extended
  with explicit endpoints).
- **`AutoLevel`** — per-channel auto-stretch: independently fill
  `[0, 255]` for each of R / G / B (RGB / RGBA). documented CLI: `-auto-level`.
- **`ContrastStretch`** — burn the darkest `black%` and brightest
  `white%` of pixels then linearly stretch the rest, **per channel**
  for RGB. documented CLI: `-contrast-stretch black%xwhite%`.
- **`LinearStretch`** — like `ContrastStretch` but cut-offs are
  absolute pixel counts, not fractions. documented CLI: `-linear-stretch black-pixels{xwhite-pixels}`.
- **`Function`** + **`FunctionOp`** — per-pixel mathematical-function
  map evaluated in normalised `[0, 1]` space. Operators:
  `Polynomial` (Horner-rule on descending coefficients), `Sinusoid`
  (`bias + amp · sin(2π · (freq · x + phase / 360))`), `ArcSin`,
  `ArcTan`. documented CLI: `-function <kind> args`.
- **`Curves`** + **`Curve`** + **`CurveInterpolation`** — per-channel
  tonal curves through user-supplied `(x, y)` control points.
  Six interpolants: `Linear` (segment-wise), `CatmullRom` (1974
  Catmull-Rom cubic, uniform `α = 0` parameterisation),
  `MonotoneCubic` (1980 Fritsch-Carlson monotonicity-preserving
  Hermite — default; never overshoots), `NaturalCubic` (de Boor 1978
  `C²` interpolant — single global tridiagonal solve for per-knot
  second derivatives via the Thomas 1949 algorithm, with the
  "natural" boundary `P''(x_0) = P''(x_{n−1}) = 0`),
  `CentripetalCatmullRom` (Yuksel, Schaefer & Keyser 2011, `α = 0.5`
  non-uniform Catmull-Rom — Barry-Goldman three-term tangent
  `m_i = (p_{i+1} − p_i)/(t_{i+1} − t_i) − (p_{i+1} − p_{i−1})/
  (t_{i+1} − t_{i−1}) + (p_i − p_{i−1})/(t_i − t_{i−1})` with
  inter-knot spacing `t_{i+1} − t_i = |p_{i+1} − p_i|^0.5`,
  provably cusp / self-loop free), and `ChordalCatmullRom`
  (Yuksel, Schaefer & Keyser 2011, `α = 1` — the third and final
  member of the same §3.3 alpha-family, same Barry-Goldman tangent
  shape with the chord-length spacing `t_{i+1} − t_i = |p_{i+1} −
  p_i|`; tames the wide-knot overshoot of uniform Catmull-Rom but
  is *not* cusp-free in the Yuksel et al. sense — only `α ∈ (0, 1)`
  is, so the centripetal variant remains the recommended default
  for path work). The natural spline is the smoothest of the six
  (continuous second derivative across knots), at the cost of
  overshoot risk; the centripetal variant gives the cusp-free
  property in the same `C¹` Hermite family as the uniform
  Catmull-Rom — pick `MonotoneCubic` when slider monotonicity must
  hold. JSON factory aliases for the `ChordalCatmullRom` mode:
  `"chordal"`, `"chordal-catmull-rom"`, `"chordal_catmull_rom"`,
  `"chordalcatmullrom"`, `"catmull-rom-chordal"` (the existing
  `"centripetal"` / `"centripetal-catmull-rom"` /
  `"centripetal_catmull_rom"` / `"centripetalcatmullrom"` /
  `"catmull-rom-centripetal"` aliases for `CentripetalCatmullRom`,
  and the `"natural-cubic"` / `"natural_cubic"` / `"natural"`
  aliases for `NaturalCubic`, are unchanged). Master curve runs on
  every tone channel; optional `red` / `green` / `blue` overrides on
  RGB / RGBA. Gray8 / RGB / RGBA + planar YUV (master curve on luma
  only). Cost is `O(W·H)` (per-channel 256-LUT).

### Tone mapping (HDR-style)

Global and local tone-mapping operators for high-contrast scenes; all
do the sRGB ↔ linear-light round-trip internally so the curve runs on
physically-meaningful luminance.

- **`Reinhard`** — Reinhard, Stark, Shirley, Ferwerda 2002 SIGGRAPH
  paper "Photographic Tone Reproduction for Digital Images". Log-
  average luminance scaling (`key` / "a" parameter, default `0.18`
  middle-grey) + extended `L · (1 + L/white²) / (1 + L)` compression
  curve. Chroma is preserved by re-modulating linear RGB through
  `L_d / L_w`. Gray8 / RGB / RGBA. Factory aliases: `reinhard`,
  `tonemap-reinhard`.
- **`Hable`** — Uncharted-2 / John Hable filmic operator from the
  2010 GDC slides. Rational-function curve
  `((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F` with the classic constants
  `A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30`; output normalised
  so a configurable linear-white point lands exactly at `1.0`.
  Per-channel LUT (cost `O(W·H)`). Gray8 / RGB / RGBA. Factory
  aliases: `hable`, `tonemap-hable`, `uncharted2`.
- **`Drago`** — Drago, Myszkowski, Annen, Chiba 2003 Eurographics
  paper "Adaptive Logarithmic Mapping For Displaying High Contrast
  Scenes". Per-pixel logarithmic compression
  `log(Lw+1) / log(2 + 8·(Lw/Lwmax)^(log(bias)/log(0.5)))` with bias
  parameter `b ∈ [0.5, 1.0]` (default `0.85`); maps the scene
  maximum to the display maximum, adapting per frame. Chroma-
  preserving sRGB round-trip. Gray8 / RGB / RGBA. Factory aliases:
  `drago`, `tonemap-drago`.
- **`ReinhardExtended`** — the unkeyed white-clamping form of the
  Reinhard 2002 operator per
  `docs/image/filter/tone-mapping-operators.md` §5.1: the same
  `L · (1 + L/L_white²) / (1 + L)` curve as `Reinhard`'s extended
  branch, but applied **without** the log-average key-scaling
  pre-step. Single `l_white` parameter (linear luminance mapped to
  pure white); when `l_white <= 0` the operator auto-picks
  `L_white = L_max(frame)` so the brightest input pixel maps
  exactly to `1`, matching the §5.1 default. Distinct from `Reinhard`
  in that the curve is applied in a single pass without an
  `L_avg`-accumulation pre-pass — useful when the caller already has
  exposure-correct linear luminance. Chroma-preserving sRGB
  round-trip; Gray8 / RGB / RGBA (YUV → `Unsupported`). Factory
  aliases: `reinhard-extended`, `tonemap-reinhard-extended`.
- **`ReinhardLocal`** — the local "dodging-and-burning" variant of
  the Reinhard 2002 operator per
  `docs/image/filter/tone-mapping-operators.md` §3. The global form's
  `1 + L` denominator is replaced with a spatially-varying local
  average chosen per-pixel from a geometric Gaussian centre / surround
  scale pyramid: at each scale `s_k = s_0 · 1.6^k` two Gaussian-
  blurred versions of the key-scaled luminance are built (centre `V1`
  with `α_1 = 1/(2√2)`, surround `V2` with `α_2 = 1.6 · α_1`), the
  normalised difference `V(·, s) = (V1 − V2) / (2^φ · a / s² + V1)`
  (§3.2; sharpening `φ = 8`) is formed, and for each pixel the
  **largest** scale for which `|V| < ε` (§3.3; uniformity threshold
  `ε ≈ 0.05`) is selected. The local tone curve is then `Ld = L_s /
  (1 + V1(·, s_m))` (§3.4). A high-contrast edge inside the
  neighbourhood drives `|V|` above ε so the scale search truncates to
  the finest scale, preserving local detail the way a photographer's
  dodge-and-burn does. Knobs: `key` (§2.1 `a`, default `0.18`),
  `phi` (sharpening, default `8`), `epsilon` (threshold, default
  `0.05`), `scales` (pyramid depth, default `8`),
  `initial_scale` (`s_0`, default `1.0`). Chroma-preserving sRGB
  round-trip; Gray8 / RGB / RGBA (YUV → `Unsupported`). Cost is
  `O(W·H·K)` with `K` scales. Factory aliases: `reinhard-local`,
  `tonemap-reinhard-local`, `dodge-and-burn`.

### Distance / signed-distance

- **`DistanceTransform`** + **`ChamferKind`** — Borgefors 1986 two-pass
  chamfer distance transform with a runtime-selectable kernel. Two
  sequential passes (forward top-left → bottom-right, backward
  bottom-right → top-left) propagate one of four integer chamfer
  metrics on a binary mask thresholded from the input intensity:
  - `Chamfer34` (default) — 3-4 mask (orthogonal=3, diagonal=4,
    divisor=3); ~8% Euclidean error.
  - `Chamfer5711` — 5-7-11 mask with knight-move (±1,±2)/(±2,±1)
    neighbours weighted 11 (orthogonal=5, diagonal=7, divisor=5);
    ~2% Euclidean error, the closer integer approximation in the
    reference table.
  - `CityBlock` — orthogonal-only mask weighted 1 (L1 / Manhattan
    metric; Rosenfeld & Pfaltz 1966).
  - `Chessboard` — orthogonal + diagonal both weighted 1 (L∞ /
    Chebyshev metric).

  Output is `Gray8` whose value is the (scaled, clamped) distance from
  each pixel to the nearest foreground pixel; useful for stroke
  thickening, signed-distance fields, contour render. Gray8 input
  only. Factory aliases: `distance-transform`, `distance` (JSON
  `kind` / `kernel` keys accept `chamfer-3-4`, `chamfer-5-7-11`,
  `city-block`, `chessboard` — and the metric-name spellings
  `manhattan` / `l1` / `chebyshev` / `l-infinity`).
- **`EuclideanDistanceTransform`** — exact-Euclidean distance
  transform (Felzenszwalb–Huttenlocher 2012). Squared distance is
  the lower envelope of upward parabolas `(p − q)² + f(q)`, one per
  sample; the envelope is built by a single left-to-right march
  (parabolas push/pop at most once each) with pairwise intersection
  `s = ((f[q] + q²) − (f[v] + v²)) / (2q − 2v)`, and a second pass
  reads distances by walking the envelope. The 2-D transform runs
  the 1-D driver once per column then once per row, so the whole
  filter is `O(d · N)` for an **exact** result — the precise
  counterpart to `DistanceTransform`, which trades accuracy for
  speed in the integer-chamfer approximation. Knobs:
  `threshold` / `invert` / `scale`. Gray8 input only; Gray8 output
  of the same dimensions. Factory aliases:
  `euclidean-distance-transform`, `euclidean-distance`, `edt`.

### Colour

- **`Modulate`** — brightness / saturation / hue via HSL round-trip.
  RGB / RGBA only (YUV returns `Unsupported`). documented CLI: `-modulate B,S,H`.
- **`Sepia`** — sepia-tone matrix with optional `threshold` mix back
  to grayscale. documented CLI: `-sepia-tone N%`.
- **`Cyanotype`** — vintage blueprint colour remap. Reduces input
  to Rec.709 luminance and interpolates between a configurable
  shadow ("Prussian blue" `(15, 42, 111)`) and highlight (paper
  white `(217, 235, 248)`) endpoint. `strength` dials the blend
  with the original RGB; custom endpoints support cousin processes
  (sepia is similar but ships separately). Gray8 input upgrades to
  RGB; RGBA alpha pass-through. Factory aliases: `cyanotype`,
  `blueprint`.
- **`Grayscale`** — Rec. 601 desaturate; optional `Gray8` collapse.
  documented CLI: `-colorspace Gray`.
- **`Colorize`** — linear blend toward a target `[R, G, B, A]` colour
  by a `0.0..=1.0` amount. documented CLI: `-colorize N%`.
- **`Tint`** — luminance-weighted blend toward a target colour;
  bright pixels reach the target, dark pixels stay put. documented CLI: `-tint N`.
- **`Vignette`** — Gaussian radial darkening centred at `(x*w, y*h)`
  with `radius` + `sigma`. documented CLI: `-vignette RxS{+x{+y}}`.
- **`ColorMatrix`** — 3×3 colour matrix with optional 3-vector
  offset; equivalent of an affine 3×4 matrix on R / G / B. RGB / RGBA
  only (YUV → `Unsupported`). documented CLI: `-color-matrix matrix`,
  `-recolor matrix`.
- **`BlueShift`** — moonlight / scotopic-vision tint: per-pixel
  `(min/factor, min/factor, max/factor)`. documented CLI: `+blue-shift factor`.
- **`Quantize`** — uniform-grid colour quantizer: round each channel
  to one of `cbrt(N)` evenly-spaced palette entries. documented CLI: `-colors N`
  (uniform-cube variant).
- **`Frame`** — decorative bordered frame with a 3-D bevel
  (highlight on top / left, shadow on bottom / right). RGB / RGBA
  only. documented CLI: `-frame WxH+inner+outer-mat`.
- **`HslRotate`** — rotate the hue channel by N degrees in HSL space
  (`RGB ⇒ HSL ⇒ rotate ⇒ RGB` round-trip). Achromatic greys
  (R = G = B) are passed through unchanged. RGB / RGBA only.
- **`ChannelMixer`** — 4×4 linear combination of source channels into
  destination channels (plus a 4-vector offset). Super-set of
  `ColorMatrix` because it also touches the alpha row. Convenience
  constructors: `identity()`, `sepia_classic()`,
  `from_color_matrix(m)`. RGB / RGBA only.
- **`ChromaticAberration`** — per-channel pixel offset on R and B
  with G undisturbed; simulates lateral lens chromatic aberration.
  Convenience constructors `horizontal(n)` / `vertical(n)` plus
  explicit `(r_dx, r_dy, b_dx, b_dy)` overrides. RGB / RGBA only.
- **`VignetteSoft`** — raised-cosine soft vignette with separate
  inner / outer normalised radii. Smoother seam than the Gaussian
  [`Vignette`] because both endpoints have zero derivative. RGB /
  RGBA only.
- **`HslShift`** — independent additive shifts for H (degrees), S,
  and L (each in `[-1, 1]`) via the same RGB ⇄ HSL round-trip as
  [`HslRotate`]. Achromatic greys keep their hue=undefined semantics
  — only the L shift moves them. Alpha pass-through; RGB / RGBA only.
- **`ColorBalance`** — three-way ASC CDL-style per-channel `lift` /
  `gamma` / `gain` (`out = ((in × gain) + lift)^(1/gamma)`) folded
  into a single 256-entry per-channel LUT so the cost is `O(W·H)`
  regardless of how many adjustments are enabled. Alpha
  pass-through; RGB / RGBA / Gray8.
- **`Canvas`** — constant-colour generator: every output sample is
  the configured `[R, G, B, A]`. YUV chroma planes are painted
  neutral 128 (matching IM's behaviour when an RGB colour lands on
  a YUV pipeline). Useful as a pipeline-head generator (a flat
  backdrop for `composite-over`) or as a wipe step.
- **`GradientRadial`** — radial gradient generator: per-pixel
  distance from `(centre_x · w, centre_y · h)` is normalised against
  `radius` (defaults to half-diagonal), then linearly interpolated
  between an `inner` and `outer` colour per channel. Gray8 / RGB /
  RGBA. Factory alias: `radial-gradient`.
- **`GradientConic`** — angular sweep generator parameterised by
  `(centre_x, centre_y)` + `start_angle` (degrees). `t = 0` lies
  along the angle ray and increases counter-clockwise round to 1.
  Gray8 / RGB / RGBA. Factory alias: `conic-gradient`.
- **`Temperature`** — warmth slider in `[-1, 1]` biasing the R / B
  channel multipliers (up to ±50 % at the extremes); G and alpha pass
  through. Per-channel 256-entry LUT, `O(W·H)` regardless of `warmth`.
  RGB / RGBA.
- **`Vibrance`** — photo-editor-style saturation boost that spares already-
  saturated pixels via `1 - s` per-pixel weighting (vs `Modulate`
  which scales `S` uniformly). RGB / RGBA only.
- **`BwMix`** — black-and-white conversion with per-channel weights.
  Convenience constructors `red_filter()` / `green_filter()` /
  `blue_filter()`. Output defaults to `Gray8`; opt into `keep_format`
  to emit grey-equalled RGB / RGBA. RGB / RGBA only.
- **`MaxRgb`** + **`MaxRgbMode`** — per-pixel `max(R, G, B)` collapse
  to greyscale (this is the **V** channel of HSV per Smith 1978
  *Color Gamut Transform Pairs*). `MaxRgbMode::Min` flips it to the
  dual `min(R, G, B)` (useful as a chroma proxy in
  `S_HSV = (V - m) / V`). Output defaults to `Gray8`; opt into
  `keep_format` to emit grey-equalled RGB / RGBA with alpha
  pass-through. Gray8 input is identity. YUV → `Unsupported`. Factory
  aliases: `max-rgb`, `hsv-value`, `min-rgb`.
- **`BorderedFrame`** — flat solid-coloured border with independent
  per-side widths. Distinct from `Frame`, which paints a 3-D bevel.
  Gray8 / RGB / RGBA. documented-CLI analogue: `-bordercolor C -border WxH`.
- **`DropShadow`** — soft offset shadow composited *behind* opaque
  RGBA subject pixels. Configurable `(offset_x, offset_y)`, blur
  `radius`, `opacity`, and `colour`. Box-blur approximates Gaussian
  for cheap soft halos. RGBA only.
- **`InnerShadow`** — soft offset shadow rendered *inside* the subject
  coverage instead of behind it. Builds the shadow mask from the
  inverted alpha channel + offset + box-blur, then darkens interior
  pixels toward `colour` weighted by `opacity`. RGBA only.
- **`Bloom`** — Gaussian-blurred highlights additively composited on
  top of the source for an emissive / HDR-look glow. Configurable
  `threshold` (luma cut-off) + `radius` + `intensity`. Gray8 / RGB /
  RGBA. Factory alias: `glow`.
- **`Heatmap`** + **`HeatmapRamp`** — false-colour ramp applied to a
  luminance-reduced source. Built-in ramps `Jet` / `Viridis` /
  `Plasma` / `Hot` / `Cool` / `Grayscale`. Input may be Gray8 / RGB /
  RGBA / planar YUV; output is RGB24 (or RGBA when input was RGBA,
  with alpha preserved). Factory alias: `false-color`.
- **`SplitTone`** — independent shadow + highlight tints driven by a
  triangular tonal mask `w(v) = (1 - 2|v - 0.5|).max(0)`. `balance`
  in `[-1, 1]` biases shadow vs highlight emphasis; `amount` is the
  global strength. RGB / RGBA only.
- **`PosterizeChannels`** — per-channel posterise with independent
  `r_levels` / `g_levels` / `b_levels` (and optional `alpha_levels`).
  Distinct from the existing `Posterize` which collapses every
  channel to the same level count. Gray8 / RGB / RGBA.
- **`Toon`** — cel-shaded cartoon look: per-channel posterise + Sobel
  edge overlay with configurable ink colour. Configurable `levels` /
  `edge_threshold` / `edge_color`. RGB / RGBA only. Factory alias:
  `cartoon`.
- **`CrossProcess`** — analogue film cross-processing emulation via
  three independent per-channel sigmoid S-curves (warm-biased R,
  neutral G, cool-biased B). `amount` / `warmth` / `contrast` knobs.
  Gray8 / RGB / RGBA / YUV-luma. Factory aliases: `cross-process`,
  `crossprocess`.
- **`GradientMap`** + **`GradientStop`** — recolour by per-pixel
  luminance mapped to a position in an arbitrary `(position, RGB)`
  gradient. Convenience constructors `duotone(shadow, highlight)`
  and `tritone(shadow, midtone, highlight)`. Gray8 input is upgraded
  to RGB; RGBA alpha is preserved. Distinct from `Heatmap` (fixed
  built-in ramps only). Factory aliases: `gradient-map`, `duotone`.
- **`SelectiveColor`** + **`HueBand`** + **`BandAdjust`** — per-hue-
  band HSL shifts (`Reds` / `Yellows` / `Greens` / `Cyans` / `Blues`
  / `Magentas` at 0°/60°/120°/180°/240°/300°). Each band contributes
  triangular-weighted H / S / L deltas. Achromatic pixels pass
  through. RGB / RGBA only. Factory aliases: `selective-color`,
  `selective-colour`.
- **`FloodFill`** — seeded scanline flood-fill with per-channel
  Chebyshev tolerance (matching the documented `-fuzz` CLI). Iterative
  span-fill keeps the work `O(W·H)`; out-of-bounds seeds error.
  Gray8 / RGB / RGBA.

### Sharpening + artistic

- **`BilateralBlur`** — edge-preserving Gaussian blur (Tomasi &
  Manduchi, *Bilateral Filtering for Gray and Color Images*, ICCV
  1998): each tap is weighted by the product of two analytic
  Gaussians, one over spatial distance (`sigma_spatial`) and one
  over intensity distance (`sigma_range`), so step edges survive a
  heavy smoothing pass. Reference O(N·k²) implementation (no
  bilateral-grid / O(1) recursion shortcut). Measured behaviour
  (see `bilateral_blur::tests`): a seeded ±20 noise patch sees a
  ~13× variance reduction at `radius=3, σ_s=2.0, σ_r=25`; a 150-
  amplitude vertical step edge at `radius=3, σ_s=1.5, σ_r=8` is
  preserved with gap=150 (vs a same-radius box mean's gap=21).
  Border samples clamp to the nearest in-bounds coordinate; alpha
  pass-through on RGBA. Gray8 / RGB / RGBA. documented-CLI analogue: `-define blur:bilateral=1` on the Gaussian blur path.
- **`Kuwahara`** — quadrant-variance edge-preserving smoothing
  (1976 Kuwahara/Hachimura paper). For each pixel splits the
  `(2*radius+1)²` window into four overlapping quadrants; the
  lowest-luminance-variance quadrant's mean colour wins. Flat
  regions blur cleanly while step edges survive because the
  homogeneous side always wins the variance race. Gray8 / RGB /
  RGBA; alpha pass-through.
- **`AnisotropicBlur`** — Perona-Malik 1990 anisotropic diffusion
  with the Lorentzian edge-stopping function
  `g(s) = 1 / (1 + (s/κ)²)`. Iterative four-neighbour explicit-Euler
  update; smooth areas diffuse freely, edges freeze. `iterations`
  / `kappa` / `lambda` (`λ ≤ 0.25` for stability). Gray8 / RGB /
  RGBA; alpha pass-through. Factory aliases: `anisotropic-blur`,
  `anisotropic`.
- **`ZoomBlur`** — radial outward "warp drive" blur. Samples along
  the line from each pixel toward the configurable centre (sample
  positions geometrically shrink toward the centre as
  `1 - t · strength`); averaging the sampled colours produces the
  classic motion-streak look. Bilinear sampling, edge-clamped.
  Gray8 / RGB / RGBA. Factory: `zoom-blur`.
- **`RadialBlur`** — rotational ("spin") blur around a configurable
  centre. For each pixel sweeps the sample arc by ±`angle/2`
  around the pixel's base angle and averages the bilinear samples.
  Distinct from `ZoomBlur` (which moves samples radially toward
  the centre instead of around it). Gray8 / RGB / RGBA. Factory
  aliases: `radial-blur`, `spin-blur`.
- **`Sharpen`** — unsharp-mask (Gaussian-blurred subtract +
  re-addition). documented CLI: `-sharpen RxS`.
- **`Unsharp`** — explicit unsharp-mask with `threshold` gate: only
  high-contrast regions are sharpened. documented CLI: `-unsharp RxS+A+T`.
- **`Clarity`** — large-radius unsharp-mask preset for mid-frequency
  tonal pop (classical photo-editor "Clarity" slider). Defaults: `radius = 30`,
  `sigma = 15.0`, `amount = 0.5`, `threshold = 10`. documented-CLI analogue: `-unsharp 30x15+0.5+10`.
- **`Emboss`** — 3×3 relief convolution with `+128` bias. documented CLI: `-emboss R`.
- **`EmbossDirectional`** — 3×3 relief with a configurable light
  azimuth (degrees CCW from East) and `depth` multiplier. Kernel
  weights track `(dx, -dy) · (cos a, sin a)`, so flipping the
  azimuth by 180° inverts polarity. Gray8 / RGB / RGBA (alpha
  pass-through) + planar YUV (luma plane only, chroma untouched).
  Factory: `emboss-directional`.
- **`MotionBlur`** — 1-D Gaussian blur along `angle_degrees`. documented CLI: `-motion-blur RxS+A`.
- **`Implode`** — radial pinch / explode with bilinear resampling.
  documented CLI: `-implode N`.
- **`Swirl`** — radius-decaying rotational distortion. documented CLI: `-swirl N`.
- **`Despeckle`** — median-window edge-preserving noise reduction
  (alpha pass-through). documented CLI: `-despeckle`.
- **`Wave`** — sinusoidal vertical pixel displacement (amplitude /
  wavelength in pixels). documented CLI: `-wave AxL`.
- **`Spread`** — random pixel-position perturbation inside a
  `[-radius, radius]²` neighbourhood, with a deterministic seed for
  reproducibility. documented CLI: `-spread N`.
- **`Charcoal`** — non-photorealistic stylise: optional Gaussian
  pre-blur (`radius`, default `0`) ⇒ Sobel-on-luma ⇒ invert ⇒ `Gray8`
  sketch. Larger pre-blur radii thicken the strokes by suppressing
  fine texture before the edge pass. documented CLI: `-charcoal R`.
- **`Paint`** — oil-paint stylise: per-pixel modal-bucket vote in a
  `(2*radius+1)²` window then mean-of-mode RGB. documented CLI: `-paint radius`.
- **`Pixelate`** — block-average spatial mosaic; each `N×N` tile
  collapses to its mean colour. Distinct from `Quantize` which
  coarsens the colour axis only. Gray8 / RGB / RGBA + planar YUV
  (luma plane only). Factory aliases: `pixelate`, `mosaic`.
- **`Crystallize`** — Voronoi-cell averaging on a jittered grid.
  Builds `cell_size`-spaced seed sites, jitters them by ±`jitter ·
  cell/2` with a deterministic PRNG (`seed`), assigns every pixel to
  the nearest seed in the 3×3 cell neighbourhood, and paints the
  cell's mean colour. Stained-glass / crystal-shard appearance.
  Gray8 / RGB / RGBA. Factory: `crystallize`.
- **`Halftone`** — variable-size dot screening simulating offset-print
  amplitude-modulated screens. Each `cell_size` cell's mean luma drives
  the radius of a centred dot painted in `ink_color` over a
  `paper_color` background; `radius = (cell/2) · sqrt(coverage)` keeps
  ink area proportional to coverage. Gray8 / RGB / RGBA. Factory:
  `halftone`.
- **`Comic`** — manga / comic-book stylise. Two stages: edge-preserving
  bilateral-style smooth + per-channel quantisation for flat fills,
  then Sobel-on-luma + threshold for the ink overlay. Distinct from
  `Toon` (no pre-smooth; thinner outline). Configurable
  `smooth_radius` / `colour_sigma` / `levels` / `edge_threshold` /
  `ink_color`. RGB / RGBA only. Factory aliases: `comic`, `manga`.
- **`Shade`** — directional Lambertian relief shading from an
  `(azimuth, elevation)` light vector. Optional colour pass-through
  mode (`+shade`). documented CLI: `-shade az,el`.
- **`Convolve`** — user-supplied square `N×N` kernel (odd `N`); optional
  bias / divisor; alpha pass-through on RGBA. documented CLI: `-convolve "..."`.
- **`Laplacian`** — 3×3 Laplacian (second-derivative) edge filter.
  Output is `|response|` clamped to `[0, 255]` as a `Gray8` frame.
  documented CLI: `-laplacian`.
- **`LaplacianOfGaussian`** + **`LogMode`** — Marr–Hildreth 1980
  Laplacian-of-Gaussian edge / zero-crossing detector. Samples the
  continuous `((x²+y²−2σ²)/σ⁴) · exp(−(x²+y²)/(2σ²))` kernel on a
  `(2·radius+1)²` grid (auto-radius `ceil(3·σ)`, clamped to `[1, 16]`),
  zero-means the coefficients so flat input produces zero response,
  then runs a dense 2-D convolution. Two output modes:
  `LogMode::Magnitude` (default) returns `|LoG · output_gain|` clamped
  to `[0, 255]` — the scale-selective second-derivative complement to
  the noise-amplifying bare 3×3 `Laplacian`. `LogMode::ZeroCrossings`
  returns the binary Marr–Hildreth edge map (255 where the gain-scaled
  LoG response changes sign between adjacent pixels AND the straddling
  gap exceeds `slope_threshold`; default `4.0`, set to `0.0` to report
  every crossing). Configurable `sigma` (default `1.4`), `radius`,
  `output_gain` (default `1.0`), `slope_threshold`. Luma-collapses any
  supported input; output is always `Gray8`. Factory aliases:
  `laplacian-of-gaussian`, `log`, `marr-hildreth`.
- **`Canny`** — full Canny edge pipeline: Gaussian pre-blur →
  Sobel gradient + direction → non-maximum suppression →
  hysteresis thresholding. Output is binary `Gray8` (0 / 255).
  documented CLI: `-canny RxS+L%+H%`.
- **`EdgeDetect`** + **`EdgeKernel`** — runtime-selectable gradient
  kernel (`Sobel` / `Prewitt` / `Scharr` / `Roberts`). Complements
  the fixed-Sobel `Edge`; same `Gray8` output shape and pixel-format
  coverage. Factory aliases: `edge-detect`, `edge-multi`.
- **`Roberts`** + **`RobertsMagnitude`** — Roberts cross 2×2
  diagonal-difference edge operator (Roberts 1963): `Gx = a − d`,
  `Gy = b − c`, magnitude `sqrt(Gx²+Gy²)` (`L2`, default) or
  `|Gx|+|Gy|` (`L1`), clamped to `[0, 255]`. The smallest first-
  derivative detector; output `Gray8`. Unlike `EdgeDetect`'s
  `L1`-only Roberts kernel, the default here is the true Euclidean
  magnitude. Factory aliases: `roberts`, `roberts-cross`.
- **`Prewitt`** + **`PrewittMagnitude`** — Prewitt 3×3 first-
  derivative edge operator (Prewitt 1970): flat `±1`-weighted
  horizontal / vertical kernels, `Gx` = right-column − left-column,
  `Gy` = bottom-row − top-row, magnitude `sqrt(Gx²+Gy²)` (`L2`,
  default) or `|Gx|+|Gy|` (`L1`), clamped to `[0, 255]`. Wider, less
  noise-sensitive support than `Roberts`; flatter weighting than
  `Edge` (Sobel). Output `Gray8`. Unlike `EdgeDetect`'s `L1`-only
  Prewitt kernel, the default here is the true Euclidean magnitude.
  Factory alias: `prewitt`.
- **`Scharr`** + **`ScharrMagnitude`** — Scharr 3×3 first-derivative
  edge operator (Scharr 2000): row / column weights `±3 ±10 ±3`,
  giving the lowest orientation error of the three 3×3 first-
  derivative kernels here. `Gx` is the Scharr-weighted right-minus-
  left column, `Gy` the Scharr-weighted bottom-minus-top row; raw
  magnitude is divided by `4` so the response lands in the same band
  as `Edge` / `Prewitt`. Magnitude `sqrt(Gx²+Gy²)` (`L2`, default) or
  `|Gx|+|Gy|` (`L1`), clamped to `[0, 255]`. Output `Gray8`. Unlike
  `EdgeDetect`'s `L1`-only Scharr kernel, the default here is the
  true Euclidean magnitude. Factory alias: `scharr`.
- **`Gabor`** + **`GaborMode`** — Gabor 1946 / Daugman 1985 oriented
  Gaussian-modulated cosine filter. Samples the continuous
  `exp(-(x'² + γ²·y'²) / (2σ²)) · cos(2π·x'/λ + ψ)` kernel on a
  `(2·radius+1)²` grid (auto-radius `ceil(3·σ·max(1, 1/γ))`,
  clamped to `[1, 32]`), zero-means the coefficients so flat input
  produces zero response, then runs a dense 2-D convolution. Two
  output modes: `GaborMode::Signed` (default) maps `[-R, +R]` to
  `[0, 255]` with neutral grey 128 = zero (preserves orientation
  + phase polarity — the linear "simple cell" response), or
  `GaborMode::Magnitude` maps `[0, R]` to `[0, 255]` (classical
  oriented-energy response). Configurable `wavelength` (carrier
  period in pixels, must be `≥ 2`), `orientation_degrees`,
  `phase_degrees` (0 = even cosine for bar response, 90 = odd sine
  for step-edge response), `sigma`, `gamma` (envelope aspect ratio
  `σ_y / σ_x`; `1.0` isotropic, `< 1.0` elongated along the
  carrier), `radius`, `output_gain`. Luma-collapses any supported
  input; output always `Gray8`. Joins the edge / texture-energy
  family at the "oriented bandpass" position complementing the
  isotropic `LaplacianOfGaussian` (Marr–Hildreth) and the
  1st-derivative `Edge` (Sobel) / `Prewitt` / `Scharr` /
  `Roberts`. Factory aliases: `gabor`, `gabor-filter`,
  `gabor-wavelet`.
- **`FreiChen`** + **`FreiChenMode`** — Frei–Chen 3×3 orthonormal-
  basis edge / line detector (Frei & Chen 1977). Projects each 3×3
  luma neighbourhood onto 9 mutually-orthogonal templates partitioned
  into an edge sub-space (4 templates: two cardinal gradients + two
  diagonal gradients, all scaled by `1/(2√2)`), a line sub-space
  (4 templates: two cardinal ripples + two discrete-Laplacian
  variants), and a 1-template mean sub-space. The output is the
  cosine of the angle between the neighbourhood vector and the
  selected sub-space — `sqrt(E/T)` for `FreiChenMode::Edge` (default,
  sensitive to step edges) or `sqrt(L/T)` for `FreiChenMode::Line`
  (sensitive to ripples / ridges). Because the metric is a normalised
  ratio, a perfect edge / line gives `255` regardless of contrast —
  qualitatively distinct from the magnitude-based detectors above.
  Always emits `Gray8`. Factory aliases: `frei-chen`, `freichen`.
- **`HoughCircles`** — circle detection via a 3-D Hough accumulator
  indexed by `(radius, cx, cy)`. Sobel-magnitude voters cast votes
  along Bresenham circles of each candidate radius into the
  accumulator; top-K peaks above `vote_threshold` are rendered onto
  a `Gray8` canvas. Complement (radius axis) to `HoughLines`.
- **`Polar`** / **`PolarDirection::DePolar`** — Cartesian ⇄ polar
  coordinate distortion (`-distort Polar` / `-distort DePolar`). Bends
  an image into a fan or unrolls a fan back to a rectangle. r7
  surfaces optional `cx` / `cy` / `max_radius` overrides through both
  the builder API and the JSON schema.
- **`Morphology`** + **`MorphologyChain`** — N-iteration greyscale
  dilate / erode with a 3×3 square or cross structuring element; plus
  `open` (erode → dilate) and `close` (dilate → erode) compositions.
  documented CLI: `-morphology Dilate|Erode|Open|Close`.
- **`MorphologyEdge`** — morphological edge / gradient operators
  built on the dilate / erode primitives:
  - `EdgeIn`  = `src - erode(src)`     — inner boundary.
  - `EdgeOut` = `dilate(src) - src`    — outer boundary.
  - `EdgeMagnitude` = `dilate - erode` — full per-pixel gradient.
  documented CLI: `-morphology EdgeIn|EdgeOut|Edge`.
- **`Perspective`** — 4-corner perspective warp; solves the 3×3
  homography from src/dst quads with 8×8 Gauss-Jordan elimination,
  then inverse-maps each output pixel via `H⁻¹` with bilinear
  sampling. Optional `output_size: (w, h)` (JSON keys `output_width`
  + `output_height`) emits a custom canvas size — useful when the dst
  quad escapes the source rectangle. documented CLI: `-distort Perspective "..."`.
- **`Affine`** — 2-D affine warp with bilinear resampling. Six
  coefficients in `(sx, ry, rx, sy, tx, ty)` order; supplied either
  as `matrix: [...]` or per-named keys on the JSON factory.
  Optional `output_size` for translation off-canvas. documented CLI: `-distort
  Affine "sx,ry,rx,sy,tx,ty"`.
- **`Srt`** — Scale / Rotate / Translate composite warp. Collapses
  `T(tx,ty) · R(θ) · S(sx,sy) · T(-ox,-oy)` into a single 2×3 affine
  matrix and reuses the [`Affine`] machinery. Origin defaults to the
  image centre on the JSON factory; uniform `scale` overrides
  `sx`/`sy`. documented CLI: `-distort SRT "ox,oy sx[,sy] angle tx,ty"`.
- **`Distort`** — full Brown-Conrady lens distortion model: radial
  (`k1` quadratic + `k2` quartic) plus optional tangential
  (`p1` + `p2`, default `0`) decentering coefficients. Pure-radial
  mode (`p1 = p2 = 0`) is bit-identical to the legacy r7 output. documented CLI: `-distort barrel "k1 k2 ..."`.
- **`TiltShift`** — selective Gaussian blur masked by an in-focus
  band (miniature-photography depth-of-field). Configurable
  `focus_centre`, `focus_height`, `falloff_height`, plus the
  underlying blur `radius` / `sigma`. Optional `angle_degrees`
  (default `0` = horizontal band; `90` = vertical band) rotates the
  focus band around the image centre.

### Channel ops

- **`ChannelExtract`** + **`Channel`** — pull one channel out of a
  multi-channel frame as a single-plane `Gray8` frame. Accepts
  `R` / `G` / `B` / `A` on packed RGB / RGBA and `Y` / `U` / `V` on
  planar YUV (chroma channels return at the subsampled grid size).
  documented-CLI rough analogue: `-channel <ch> -separate`.

### Two-input compositing

- **`Clut`** — 1-D Colour Look-Up Table. `src` is the image; `dst`
  is the CLUT (read row-major). Per-channel index lookup; alpha
  pass-through. documented CLI: `-clut`.
- **`HaldClut`** — Hald CLUT image-as-LUT colour grading. `dst` is a
  `(L²)×(L²)` Hald cube; trilinear sampling per pixel. RGB / RGBA
  only. documented CLI: `-hald-clut`.
- **`Difference`** — two-input pixel-wise absolute difference
  (`out = |src - dst|`). Cheap change-detection / motion mask, no
  Porter–Duff coverage algebra. Gray8 / RGB / RGBA / planar YUV.
- **`DisplacementMap`** — two-input warp: the `dst` (map) image's
  R / G channels (or the single channel on `Gray8`) carry per-pixel
  `(dx, dy)` vectors re-centred to `[-0.5, 0.5]` and scaled by
  `scale_x` / `scale_y`; the source is then bilinear-sampled at the
  perturbed coordinate. Classic compositing primitive for water
  ripple, heat haze, frosted-glass distortion. Both inputs must
  share `(width, height, format)`. Gray8 / RGB / RGBA.
- **`Watermark`** — two-input over-place of a secondary image
  (`dst`) onto the source (`src`) at `(offset_x, offset_y)` with a
  `[0, 1]` `opacity` multiplier. The overlay shape is independent of
  the source — clipped on negative or oversize placements. Straight-
  alpha on RGBA; opacity scales the per-pixel alpha. Gray8 / RGB /
  RGBA (the two ports must share `PixelFormat`).
- **`Composite`** + **`CompositeOp`** — Porter–Duff and arithmetic
  blends of a foreground (`src`) over a background (`dst`) frame.
  Sixteen operators registered as `composite-<op>` factories:
  - Porter–Duff coverage: `over`, `in`, `out`, `atop`, `xor`.
  - Arithmetic / per-channel: `plus` (clamped sum), `multiply`,
    `screen`, `overlay` (multiply-or-screen on `dst < 128`),
    `darken` (per-channel min), `lighten` (per-channel max),
    `difference` (`|src - dst|`).
  - Overlay family (r11): `hardlight` (overlay driven by `src` instead
    of `dst`), `softlight` (Pegtop continuous formula),
    `colordodge` (`dst / (1 - src/255)`), `colorburn`
    (`255 - (255 - dst) / (src/255)`).
  - Operates on `Gray8` / `Rgb24` / `Rgba` with **straight (non-
    premultiplied)** alpha throughout. The new `TwoInputImageFilter`
    trait formalises the two-port shape; the
    `TwoInputImageFilterAdapter` shim buffers per-port frames and
    emits whenever both ports have a frame in hand.
  - documented-CLI analogue: `-compose <op> -composite`.

All single-input filters listed above share the `ImageFilter` trait — chain them
manually with repeated `.apply()` calls, or feed them through
`oxideav-pipeline` as `video.<name>` filter nodes in a JSON job.

## Pixel formats

Supported in v0:

- `Gray8`, `Rgb24`, `Rgba`
- `Yuv420P`, `Yuv422P`, `Yuv444P`

Other formats return `Error::unsupported`. Planar higher-bit formats
(`Yuv420P10Le` etc.) will land in a later release as filters grow
per-format specialisations.

## License

MIT. See [LICENSE](LICENSE).
