# oxideav-image-filter

[![CI](https://github.com/OxideAV/oxideav-image-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-image-filter/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-image-filter.svg)](https://crates.io/crates/oxideav-image-filter) [![docs.rs](https://docs.rs/oxideav-image-filter/badge.svg)](https://docs.rs/oxideav-image-filter) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
oxideav-image-filter = "0.1"
```

## Filters

### Blur

Separable Gaussian blur with configurable radius + sigma and an optional
plane selector. The two passes keep their intermediate in binary32 (the
result is the 2-D Gaussian rounded once), a picture with alpha blurs its
colour premultiplied (a transparent pixel's colour has no weight), and
the kernel is cut at `8σ`:

```rust
# use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
# let input_frame = VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: 64 * 3, data: vec![0; 64 * 64 * 3] }] };
use oxideav_image_filter::{Blur, ImageFilter, Planes, VideoStreamParams};

// Stream shape (format / width / height) comes from the stream's
// CodecParameters, not from the frame.
let params = VideoStreamParams { format: PixelFormat::Rgb24, width: 64, height: 64 };

// Blur all planes with radius 3, sigma 1.5 (default).
let f = Blur::new(3).with_sigma(1.5);
let out = f.apply(&input_frame, params)?;

// Blur only the luma plane — leave chroma untouched.
let f = Blur::new(5).with_sigma(2.0).with_planes(Planes::Luma);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Edge

3×3 Sobel edge-magnitude; output is a `Gray8` frame of the same size:

```rust
# use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
# let input_frame = VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: 64 * 3, data: vec![0; 64 * 64 * 3] }] };
use oxideav_image_filter::{Edge, ImageFilter, VideoStreamParams};

let params = VideoStreamParams { format: PixelFormat::Rgb24, width: 64, height: 64 };
let edges = Edge::new().apply(&input_frame, params)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For `Rgb24` / `Rgba` input the filter first computes a luma proxy
`Y = (R + 2G + B) / 4` and applies Sobel to it. YUV inputs use plane 0
directly.

### Resize

Rescale to arbitrary dimensions with seven reconstruction kernels:

```rust
# use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
# let input = VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: 64 * 3, data: vec![0; 64 * 64 * 3] }] };
use oxideav_image_filter::{Interpolation, ImageFilter, Resize, VideoStreamParams};

let params = VideoStreamParams { format: PixelFormat::Rgb24, width: 64, height: 64 };
let half = Resize::new(params.width / 2, params.height / 2)
    .with_interpolation(Interpolation::Bilinear)
    .apply(&input, params)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

- **`Nearest`** — point sample; fast, blocky, ideal for pixel art.
- **`Bilinear`** (default) — 2×2 linear blend; smooth, the natural-image
  default.
- **`Bicubic`** — separable 4-tap cubic convolution using the uniform
  **Catmull-Rom** cubic of `docs/image/filter/curve-interpolation.md`
  §3.1 (`m_i = (p_{i+1} − p_{i−1}) / 2`, partition-of-unity weights so
  flat input reproduces exactly). Sharper than bilinear on upscale and
  can ring slightly at hard edges (the cubic's negative side-lobes) —
  the standard cubic-reconstruction quality/overshoot trade. Border taps
  clamp to the nearest in-bounds sample.
- **`Area`** — coverage-weighted area average ("box"): every output
  pixel is the mean of the source pixels its footprint
  `[k·scale, (k+1)·scale)` spans, with fractional weights at the two
  ragged ends so non-integer ratios are exact. This is the correct
  **downscale** kernel — point-sampling kernels alias below ~0.5 scale
  because they read only a few taps, whereas the area average integrates
  the whole shrinking footprint and is alias-free by construction.
  Separable two-pass (horizontal then vertical). On upscale (footprint
  < 1 px) it degenerates to nearest-neighbour.
- **`Lanczos { a }`** — windowed-sinc reconstruction
  `L(x) = sinc(x)·sinc(x/a)` on `|x| < a` (`a` lobes, clamped `1..=8`;
  `a = 3` the sharper default). `L(0) = 1`, `L(k) = 0` at non-zero
  integers, so it interpolates the samples (a 1:1 pass reproduces the
  input exactly). Runs through a general separable weighted-tap driver
  that **scales the kernel by the shrink ratio on downscale**, so the
  reconstruction filter doubles as the anti-alias low-pass — alias-free
  below ~0.5 scale the way `Area` is, while keeping the sharper
  windowed-sinc response on upscale. Per-output weights renormalise to
  sum to 1 so flat input reproduces exactly even against a clamped edge.
  Derived from first principles (windowed sinc); it rings slightly at
  hard edges (the sinc's negative side-lobes).
- **`Mitchell`** — the balanced Mitchell–Netravali cubic `BC(1/3, 1/3)`
  (the uniform Catmull-Rom of `Bicubic` is `BC(0, 1/2)`); a good
  blur/ring compromise for photographic resampling. Same anti-aliasing
  separable driver as `Lanczos`.
- **`BSpline`** — the everywhere-non-negative cubic B-spline `BC(1, 0)`
  of the same Mitchell–Netravali family. Never rings or overshoots (a
  monotone step stays monotone) at the cost of being the softest cubic —
  `k(0) = 2/3`, so it *approximates* rather than interpolates the
  samples. Same anti-aliasing separable driver.

- **`Filter(ResampleFilter)`** — ImageMagick's named filter set, the
  `-filter` list: `Point`, `Box`, `Triangle`, `Hermite`, `Quadratic`,
  `Cubic` / `Spline`, `Catrom`, `Mitchell`, `CubicSpline`, `Lagrange`,
  the windowed sincs `Hann`, `Hamming`, `Blackman`, `Welch`, `Parzen`,
  `Bohman`, `Bartlett`, `Kaiser` (β 6.5), `Cosine`, `Sinc` / `SincFast`,
  `Lanczos` / `LanczosSharp` / `Lanczos2` / `Lanczos2Sharp` /
  `LanczosRadius`, `Gaussian` (σ 0.5), `Robidoux`, `RobidouxSharp` and
  `Jinc`. Each is a weighting function with a support radius, optionally
  multiplied by a window over that support; `Resize::with_filter_options`
  carries the expert defines (`blur`, `support` / `lobes`, `window`,
  `sigma`, `b` / `c`, the Kaiser β). The shrink rule widens the kernel by
  the scale, taps outside the picture are dropped and the rest
  renormalised — the rule `magick` follows. Every filter was fixed
  against the `magick` binary (7.1.2 Q16-HDRI): a one-pixel impulse
  enlarged eight times gives the kernel sampled every eighth of a pixel,
  and `tests/fixtures/magick_impulse.txt` holds those responses (plus the
  edge and shrink cases and the expert defines) for the fixture test to
  reproduce within `2e-5`; with `magick` on the path the live test
  resizes `rose:` through all 31 filters at 8 and 16 bits and holds
  PSNR ≥ 45 dB (measured: identical to the last code on most).

A packed layout with alpha (`Rgba`, `Bgra`, `Argb`, `Abgr`, `Ya8`,
`Ya16Le`, `Rgba64Le`, `RgbaF32Le`) averages its colour weighted by
coverage, so a transparent pixel's colour does not fringe its
neighbours; an output pixel no coverage reaches keeps the plain average,
and an opaque picture is untouched by the rule. Both passes of the
separable resampler and the bilinear / bicubic path run under the thread
budget. `tests/depth_analytic.rs` reproduces the `BC(B, C)` cubic as the
impulse response of `Mitchell`, `Catrom`, `Cubic`, `Robidoux`, a custom
`(B, C)` and `BSpline` at 8 bits, 16 bits and float.

For planar YUV the chroma planes are resized to the matching subsampled
output dimensions (e.g. 4:2:0 halves both chroma axes). Pipeline JSON
selects the kernel via `"interpolation"`: `nearest` / `bilinear` /
`bicubic` (aliases `cubic`, `catmull-rom`) / `area` (aliases `box`,
`average`) / `lanczos` (3-lobe; also `lanczos2` / `lanczos3`) /
`mitchell` (alias `mitchell-netravali`) / `b-spline` (aliases `bspline`,
`cubic-b-spline`) / any ImageMagick filter name (`robidoux`, `hann`,
`lanczos-sharp`, …; or `"filter"` outright), with the expert defines as
`blur`, `support`, `lobes`, `window`, `sigma`, `b`, `c`, `kaiser_beta`.

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
  **`DiffusionKernel`** + **`ScanOrder`** — bit-depth-reduction dither
  with two algorithm families:
  - **Bayer ordered dither** with tiled `2×2` / `4×4` / `8×8` / `16×16`
    threshold maps built by the standard
    `M_2n = 4·M_n + M_2[hi, lo]` recurrence (Bayer 1973 IEEE ICC).
    No state, perfectly parallel, produces the classic crosshatch
    texture; the `16×16` map (256 distinct thresholds) is the finest
    dispersed-dot variant and shows the least texture on smooth
    gradients.
  - Both families run one binary32 kernel at every depth: 8-bit
    pictures are widened to it, 16-bit and float pictures are dithered
    at their own precision (a 16-bit tone between two 8-bit codes keeps
    its mean, the output levels are 16-bit codes), and the diffused
    error is never truncated to whole codes.
  - **Error diffusion** along a left-to-right raster scan: the seven
    canonical kernels Floyd–Steinberg (÷16, 1976 SID),
    Jarvis–Judice–Ninke (÷48, 1976 CGIP), Stucki (÷42, 1981 IBM
    RZ1060), Sierra-3 (÷32), Sierra-2 (÷16), Sierra-Lite (÷4) — the
    three Sierra kernels are community-attributed c. 1989–1990 —
    and Atkinson (÷8 with coefficient sum 6, so 2/8 of each pixel's
    quantisation error is intentionally discarded; Apple, late
    1980s). Per-channel for R / G / B (alpha pass-through on
    RGBA), luma-only on YUV. The scan order is selectable via
    `with_scan(ScanOrder::Serpentine)`: a boustrophedon traversal
    that alternates row direction and mirrors the stencil
    horizontally on the reversed rows, suppressing the directional
    "worm"/hysteresis artifacts of a pure raster scan at no extra
    cost (dithering-kernels.md §1.3). `ScanOrder::Raster` (the
    default) keeps the classic left→right traversal; scan order has
    no effect on the feedback-free Bayer mode.

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
  `{"matrix": 2|4|8|16}`). The error-diffusion factories accept
  `{"scan": "serpentine"}` (or the `{"serpentine": true}` shorthand)
  to switch to the boustrophedon scan.

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
  (border-clamped). 1×1 window is identity. The cost per pixel does not
  grow with the window area: sliding-histogram median (8- and 16-bit;
  a sorted window or, past 1024 samples, a binary indexed tree over the
  picture's distinct values for arbitrary floats), running extremes,
  prefix-sum mean. A window larger than the picture — any size up to
  `2³² − 1` a side — is exact (edge samples weighted as replication
  weights them) and costs no more than the picture.
  documented CLI: `-statistic <op> WxH`.
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
  Ten interpolants: `Linear` (segment-wise), `CatmullRom` (1974
  Catmull-Rom cubic, uniform `α = 0` parameterisation),
  `MonotoneCubic` (1980 Fritsch-Carlson monotonicity-preserving
  Hermite — default; never overshoots; pulls the §2.2 normalised
  tangents back to the circle of radius 3), `MonotoneCubicBox`
  (same Fritsch-Carlson interpolant but using the §2.2 *box* sufficient
  region — `0 ≤ α ≤ 3` and `0 ≤ β ≤ 3` clamped independently — the
  doc-flagged independent-axis alternative to the radius-3 circle;
  also overshoot-free on monotone data, just a slightly different
  monotone shape near steep knots; JSON aliases `monotone-box` /
  `monotone-cubic-box`), `NaturalCubic` (de Boor 1978
  `C²` interpolant — single global tridiagonal solve for per-knot
  second derivatives via the Thomas 1949 algorithm, with the
  "natural" boundary `P''(x_0) = P''(x_{n−1}) = 0`),
  `CentripetalCatmullRom` (Yuksel, Schaefer & Keyser 2011, `α = 0.5`
  non-uniform Catmull-Rom — Barry-Goldman three-term tangent
  `m_i = (p_{i+1} − p_i)/(t_{i+1} − t_i) − (p_{i+1} − p_{i−1})/
  (t_{i+1} − t_{i−1}) + (p_i − p_{i−1})/(t_i − t_{i−1})` with
  inter-knot spacing `t_{i+1} − t_i = |p_{i+1} − p_i|^0.5`,
  provably cusp / self-loop free), `ChordalCatmullRom`
  (Yuksel, Schaefer & Keyser 2011, `α = 1` — the third and final
  member of the same §3.3 alpha-family, same Barry-Goldman tangent
  shape with the chord-length spacing `t_{i+1} − t_i = |p_{i+1} −
  p_i|`; tames the wide-knot overshoot of uniform Catmull-Rom but
  is *not* cusp-free in the Yuksel et al. sense — only `α ∈ (0, 1)`
  is, so the centripetal variant remains the recommended default
  for path work), `Cardinal { tension_q8: u8 }` (Catmull & Rom
  1974, §3.2 of the reference doc — the tension-parameterised
  generalisation of uniform Catmull-Rom whose per-knot tangent is
  the §3.1 secant slope scaled by `(1 − c)`. Tension `c ∈ [0, 1]`
  is encoded as 8.0 fixed-point `tension_q8 = 0..=255` so the enum
  keeps `Copy + Eq + Hash`. `c = 0` is byte-equivalent to
  `CatmullRom`; `c = 1` zeros every tangent so the cubic-Hermite
  evaluator collapses to the convex-combination shape
  `h00·y_i + h01·y_{i+1}` with provable per-segment
  no-overshoot — the only `CurveInterpolation` variant that can be
  driven to an overshoot-free shape while keeping the cubic-Hermite
  smoothness of the underlying basis),
  `ClampedCubic { start_slope_q8, end_slope_q8 }` (the first §4.2
  alternative boundary for the `C²` tridiagonal spline family —
  the **end first derivatives** are prescribed instead of zeroing
  the end second derivatives; differentiating the §4.4 segment
  cubic yields the head/tail rows `2h_0·M_0 + h_0·M_1 =
  6(Δ_0 − s₀)` and `h_{n−2}·M_{n−2} + 2h_{n−2}·M_{n−1} =
  6(s₁ − Δ_{n−2})` so the full `n × n` system stays tridiagonal +
  diagonally dominant for the same Thomas sweep; slopes are signed
  Q8.8 fixed point — `slope = q / 256` — keeping the enum
  `Copy + Eq`; `s₀ = s₁ = 1` on identity knots solves to `M ≡ 0`
  i.e. the exact identity, `s₀ = s₁ = 0` gives a smoothstep-style
  S with flat shoulders), and `NotAKnotCubic` (the second §4.2
  alternative boundary — the **third derivative** is forced
  continuous across the first/last interior knot so segments 0/1
  and n−3/n−2 each merge into one cubic; `P'''` on segment `i` is
  the constant `(M_{i+1} − M_i)/h_i`, and eliminating
  `M_0`/`M_{n−1}` into the adjacent §4.1 interior rows keeps the
  reduced system tridiagonal; degenerate `n = 3` resolves to the
  parabola through the three points, `n = 2` to the straight
  line — the parameter-free "let the data speak" boundary that
  avoids the natural spline's artificial end-flattening). The
  `C²` spline trio (natural / clamped / not-a-knot) is the
  smoothest of the family (continuous second derivative across
  knots), at the cost of overshoot risk; the centripetal variant
  gives the cusp-free property in the same `C¹` Hermite family as
  the uniform Catmull-Rom; the cardinal variant is the only one
  with a runtime knob — pick `MonotoneCubic` when slider
  monotonicity must hold across any input, or `Cardinal { c = 1 }`
  when the no-overshoot guarantee is the only constraint. JSON
  factory aliases for the `Cardinal` mode: `"cardinal"`,
  `"cardinal-spline"`, `"cardinal_spline"`, `"cardinalspline"`,
  with the tension carried by a `"tension"` (or `"c"`) JSON field
  — out-of-range values clamp to `[0, 1]` and a missing tension
  defaults to `0.5`. `ClampedCubic` aliases: `"clamped"`,
  `"clamped-cubic"`, `"clamped_cubic"`, `"clampedcubic"`, with
  slopes carried by `"start_slope"`/`"slope0"`/`"s0"` and
  `"end_slope"`/`"slope1"`/`"s1"` (default `1.0`, clamped to
  `[-127, 127]`). `NotAKnotCubic` aliases: `"not-a-knot"`,
  `"not_a_knot"`, `"notaknot"`, `"not-a-knot-cubic"`,
  `"not_a_knot_cubic"`. Pre-existing aliases for `ChordalCatmullRom`
  (`"chordal"` / `"chordal-catmull-rom"` / `"chordal_catmull_rom"` /
  `"chordalcatmullrom"` / `"catmull-rom-chordal"`),
  `CentripetalCatmullRom` (`"centripetal"` /
  `"centripetal-catmull-rom"` / `"centripetal_catmull_rom"` /
  `"centripetalcatmullrom"` / `"catmull-rom-centripetal"`), and
  `NaturalCubic` (`"natural-cubic"` / `"natural_cubic"` /
  `"natural"`) are unchanged. Master curve runs on every tone
  channel; optional `red` / `green` / `blue` overrides on RGB /
  RGBA. Gray8 / RGB / RGBA + planar YUV (master curve on luma
  only). Cost is `O(W·H)` (per-channel 256-LUT).

### Tone mapping (HDR-style)

Global and local tone-mapping operators for high-contrast scenes; all
do the sRGB ↔ linear-light round-trip internally so the curve runs on
physically-meaningful luminance.

- **`Reinhard`** — Reinhard, Stark, Shirley, Ferwerda 2002 SIGGRAPH
  paper "Photographic Tone Reproduction for Digital Images". Log-
  average luminance scaling (`key` / "a" parameter, default `0.18`
  middle-grey) + extended `L · (1 + L/white²) / (1 + L)` compression
  curve. Re-colours via the §1 step 3 saturation-controlled form
  `C_out = (C_in / L_w)^s · L_d`: the default `s = 1` is the plain
  chroma-preserving re-modulation `C_in · (L_d / L_w)`, `0 ≤ s < 1`
  desaturates toward neutral grey, `s > 1` boosts chroma
  (`with_saturation`, JSON key `saturation` / `s`). Gray8 / RGB / RGBA.
  Factory aliases: `reinhard`, `tonemap-reinhard`.
- **`Hable`** — Uncharted-2 / John Hable filmic operator from the
  2010 GDC slides. Rational-function curve
  `((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F` with the classic constants
  `A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30`; output normalised
  so a configurable linear-white point lands exactly at `1.0`.
  Per-channel LUT (cost `O(W·H)`). Gray8 / RGB / RGBA. Factory
  aliases: `hable`, `tonemap-hable`, `uncharted2`.
- **`Drago`** — Drago, Myszkowski, Annen, Chiba 2003 Eurographics
  "Adaptive Logarithmic Mapping" per
  `docs/image/filter/tone-mapping-operators.md` §4. Per-pixel log
  compression `log(Lw+1) / log(2 + 8·(Lw/Lwmax)^(log(bias)/log(0.5)))`
  with bias `b ∈ [0.5, 1.0]` (default `0.85`). The §4.2 leading factor
  `Ld_max · 0.01 / log10(Lwmax+1)` is exposed as `ld_max` (cd/m²,
  default `100` → `[0,1]` normalisation). Optional §4.1 exposure-
  independent pre-scaling: `with_key` (explicit §1.1 log-average key) /
  `with_auto_key` (image's own log-average) divides luminance before
  the curve so the mapping is invariant to scene exposure. Chroma-
  preserving sRGB round-trip. Gray8 / RGB / RGBA. Factory aliases:
  `drago`, `tonemap-drago`; params `{"bias", "ld_max", "key",
  "auto_key", "display_scale"}` (`"key": "auto"` = auto-key).
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
- **`SrgbTransform`** + **`SrgbCurve`** + **`SrgbDirection`** — the
  documented display transfer function
  (`docs/image/filter/tone-mapping-operators.md` §5.2) exposed as a
  standalone 256-entry LUT that converts between display-encoded and
  linear-light pixel values, so a pipeline can wrap a linear-light
  operation (additive light, blur, alpha compositing) in a correct
  round-trip. `SrgbCurve::Srgb` (default) is the IEC 61966-2-1 piecewise
  curve — encode `V = 12.92·Lin` (`Lin ≤ 0.0031308`) /
  `1.055·Lin^(1/2.4) − 0.055`, decode the analytic inverse;
  `SrgbCurve::Gamma` (via `SrgbCurve::gamma(γ)`) is the §5.2 pure
  power-law `V = Lin^(1/γ)` / `Lin = V^γ`. `SrgbDirection::Encode` runs
  the OETF (linear → display); `SrgbDirection::Decode` (default) runs the
  EOTF (display → linear). Encode∘decode is identity within 8-bit
  rounding (deep shadows excepted — inherent 8-bit shadow precision).
  Gray8 / RGB / RGBA (alpha = coverage, passed through); YUV →
  `Unsupported`. Distinct from `Gamma` (display-gamma power curve in
  encoded space) — this is the colorimetric sRGB↔linear transfer pair.
  Factory aliases: `srgb-decode` / `linearize` / `srgb-to-linear`,
  `srgb-encode` / `delinearize` / `linear-to-srgb`, and `srgb-transform`
  (`direction` / `curve` / `gamma` JSON keys).

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
- **`WeightedDistanceTransform`** — weighted (generalised) distance
  transform: the **continuous-seed** form of the generalised DT
  `D_f(p) = min_q(‖p − q‖² + f(q))` named in
  `docs/image/filter/distance-transform.md` §1 ("the generalized form
  (arbitrary `f`) is what makes the algorithm reusable beyond binary
  masks"). Where `EuclideanDistanceTransform` seeds the F–H driver with
  a binary mask (`f(q) ∈ {0, +∞}`), this filter seeds **every** pixel
  with a finite cost `f(q) = (weight · c(q))²` derived from its own
  intensity (`c(q) ∈ [0, 1]` the normalised faintness — `0` for a free
  foreground site, `1` for distant background). The output is then the
  cheapest combined cost of *travelling* squared-Euclidean distance to a
  site plus *paying* that site's intrinsic cost, so the field is a smooth
  blend of "how far" and "how faint" (cost-weighted feature maps,
  intensity-aware feathering, grey-weighted morphology). Reuses the exact
  `dt_1d` lower-envelope driver of `EuclideanDistanceTransform`, so the
  whole filter stays `O(d · N)` for an exact generalised result;
  `weight = 0` makes every pixel a free seed (all-black field) and a very
  large `weight` recovers the binary transform. Knobs:
  `weight` / `invert` / `scale`. Gray8 input only; Gray8 output of the
  same dimensions. Factory aliases: `weighted-distance-transform`,
  `weighted-distance`, `wdt`.
- **`SignedDistanceField`** — exact signed distance field, the
  difference of two exact-Euclidean transforms per
  `docs/image/filter/distance-transform.md` §1 (the generalised DT
  `D_f(p) = min_q(‖p − q‖² + f(q))` whose intro names "SDF generation"
  as a use-case) and §2 (Felzenszwalb–Huttenlocher 2012). `sdf(p) =
  d_out(p) − d_in(p)` where `d_out` is the distance to the nearest
  foreground pixel (the EDT of the mask) and `d_in` the distance to the
  nearest background pixel (the EDT of the inverted mask) — so
  foreground pixels carry negative distance and background pixels
  positive, with the zero level-set tracing the shape boundary. Renders
  as `clamp(midpoint + scale · sdf, 0, 255)` so an on-boundary pixel
  lands at the neutral `midpoint` (default `128`), the interior darkens
  below it, and the exterior brightens above it. Reuses the `dt_1d`
  lower-envelope driver of `EuclideanDistanceTransform`, so the whole
  filter is `O(d · N)` for an exact result. Knobs:
  `threshold` / `invert` / `scale` / `midpoint`. Gray8 input only;
  Gray8 output of the same dimensions. The signed counterpart to the
  unsigned `EuclideanDistanceTransform` — useful for
  resolution-independent glyph / shape rendering, feathering, outline /
  glow generation, and iso-level morphological grow / shrink. Factory
  aliases: `signed-distance-field`, `signed-distance`, `sdf`.
- **`Feather`** — soft edge feathering driven by the exact Euclidean
  distance transform per `docs/image/filter/distance-transform.md` §1
  (the generalised DT `D_f(p) = min_q(‖p − q‖² + f(q))` whose intro
  names "feathering" as a backed use-case) and §2
  (Felzenszwalb–Huttenlocher 2012). The inner distance `d_in(p)` is the
  EDT of the inverted mask, and the coverage ramp is `cov(p) =
  clamp(d_in(p) / radius, 0, 1)` (background → 0): a boundary pixel maps
  to `0`, a full `radius` inside reaches `1`, and the interior beyond the
  band saturates at `1`. Reuses the same `dt_1d` lower-envelope driver
  as `EuclideanDistanceTransform` / `SignedDistanceField`, so the whole
  filter is `O(d · N)` for an exact distance field. `Rgba` (alpha =
  coverage mask, RGB pass-through, output alpha = feathered coverage) or
  `Gray8` (luma = mask, output = single-plane coverage ramp). Knobs:
  `radius` / `threshold` / `invert`. The soft-matte complement to the
  hard binary `Threshold` and the signed `SignedDistanceField`. Factory
  aliases: `feather`, `feather-edge`, `soft-edge`.
- **`VoronoiTransform`** — exact nearest-feature (Voronoi) transform: the
  **argmin** counterpart of the exact-Euclidean distance per
  `docs/image/filter/distance-transform.md` §2.3. The
  lower-envelope march of `EuclideanDistanceTransform` already tracks
  `v[k]` (the sample whose parabola is lowest at each output position),
  so carrying that index through both separable passes labels every
  pixel with the coordinate of its closest foreground site — the
  discrete Voronoi / nearest-neighbour partition of the grid — at no
  extra cost (`O(d · N)` total, exact). Output renders a deterministic
  per-cell hash of the nearest seed so adjacent Voronoi cells get
  distinct, flat grey levels (cracked-glass mosaic). Distinct from
  `Crystallize` (which approximates with a jittered grid + 3×3 local
  search); this is the exact global nearest over an arbitrary seed set.
  Gray8 input only; Gray8 output of the same dimensions. Knobs:
  `threshold` / `invert`. Factory aliases: `voronoi`,
  `voronoi-transform`, `nearest-feature`.
- **`ProximityFill`** — exact nearest-seed **value** propagation
  (Voronoi region fill). Where `VoronoiTransform` renders a synthetic
  per-cell label, this paints every pixel with the source intensity of
  its nearest feature site — the standard nearest-neighbour region-grow
  / sparse-inpainting primitive (extend a handful of known samples to
  fill the whole frame). Built on the same exact feature transform, so
  `O(d · N)` and exact (the global nearest over an arbitrary seed set,
  not the local-window approximation of `Crystallize`). With no seed
  pixels the output copies the input verbatim. Gray8 input only; Gray8
  output of the same dimensions. Knobs: `threshold` / `invert`. Factory
  aliases: `proximity-fill`, `voronoi-fill`, `nearest-fill`.
- **`DistanceMorphology`** + **`MorphOp`** — exact-Euclidean binary
  morphology (dilate / erode / open / close) derived directly from the
  `docs/image/filter/distance-transform.md` §1 nearest-feature distance.
  A disc dilation of radius `r` keeps every pixel within Euclidean
  distance `r` of a foreground site (`D_FG(p) ≤ r²`); erosion is the
  dual `D_BG(p) > r²` (dilation of the complement); **opening**
  `δ_r(ε_r(·))` clears foreground specks; **closing** `ε_r(δ_r(·))`
  fills background holes. Each primitive is one exact §2.4
  squared-Euclidean transform plus an `r²` threshold, so the structuring
  element is a true Euclidean circle — not the octagon a 3×3 iteration
  produces — and the test is `sqrt`-free and exact for integer `r`.
  Distinct from the structuring-element `Morphology` family below.
  Gray8 in/out. Knobs: `radius`, `threshold`, `invert`, `fg_value`.
  Factory aliases: `distance-dilate` / `distance-erode` /
  `distance-open` / `distance-close` (+ `euclidean-` forms), with an
  `op` JSON override.
- **`DistanceOutline`** — exact-Euclidean boundary band (outline /
  stroke). Paints the band `{ p : −inner ≤ s(p) ≤ outer }` straddling
  the shape contour, where `s` is the signed Euclidean distance
  (`+D_FG` outside, `−D_BG` inside): `inner = 0` gives a purely-outer
  stroke, `outer = 0` a purely-inner one, equal radii a centred stroke.
  Exactly the set difference `dilate(outer) − erode(inner)`; two exact
  §2.4 transforms, `sqrt`-free. Gray8 in/out. Knobs: `inner`, `outer`
  (or `width` for a centred stroke), `threshold`, `invert`, `fg_value`.
  Factory aliases: `distance-outline`, `euclidean-outline`,
  `distance-stroke`.

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
  (alpha pass-through), on the same sliding median as `Statistic`.
  documented CLI: `-despeckle`.
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
  `(2*radius+1)²` window then mean-of-mode RGB. Every pixel falls in
  intensity bucket `⌊Y · levels⌋` (`Y = (77 R + 150 G + 29 B) / 256`);
  the window slides along the row, so the cost per pixel is
  `O(radius + levels)`. documented CLI: `-paint radius`.
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
  an image into a fan or unrolls a fan back to a rectangle. Optional
  `cx` / `cy` / `max_radius` overrides are available through both the
  builder API and the JSON schema.
- **`Morphology`** + **`MorphologyChain`** — N-iteration greyscale
  dilate / erode; plus `open` (erode → dilate) and `close` (dilate →
  erode) compositions. Structuring elements: the 3×3 `Square3x3` /
  `Cross3x3`, and `Square` / `Diamond` / `Disk` at the filter's
  `radius` (`with_radius`; `max(|dx|, |dy|) ≤ r`, `|dx| + |dy| ≤ r`,
  `dx² + dy² ≤ r²`). `k` passes of a square or diamond run as one pass
  of radius `k·r`; a square is a separable running extreme whose cost
  does not depend on its size. JSON: `{"element": "square" | "cross" |
  "diamond" | "disk", "radius": r, "iterations": k}`.
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
- **`Composite`** + **`CompositeOp`** — a foreground (`src`) composed
  onto a background (`dst`) with any of 70 operators
  (`CompositeOp::ALL`), each registered as a `composite-<op>` factory:
  - Porter–Duff coverage: `clear`, `src`, `dst`, `over`, `dst-over`,
    `in`, `dst-in`, `out`, `dst-out`, `atop`, `dst-atop`, `xor`.
  - Separable blends, `B(Cb, Cs)` weighted by coverage as in the
    Porter–Duff source-over sum: `multiply`, `screen`, `overlay`,
    `darken`, `lighten`, `color-dodge`, `color-burn`, `hard-light`,
    `soft-light`, `difference`, `exclusion`, plus `plus`, `minus`,
    `divide`, the modulus pair, `linear-burn` / `linear-dodge` /
    `linear-light`, `vivid-light`, `pin-light`, `hard-mix`,
    `pegtop-light`, `soft-burn` / `soft-dodge`, `stamp`, `freeze`,
    `reflect`, `negate`, `interpolate`.
  - Non-separable (hue / saturation / luminosity) blends: `hue`,
    `saturate`, `luminize`, `colorize`, and the intensity-selected
    `darken-intensity` / `lighten-intensity`.
  - Parametrised: `mathematics`, `blend`, `dissolve`, `modulate`,
    `threshold`, `change-mask`, `rmse`; channel copies (`copy-red` …
    `copy-black`); and the map operators `displace`, `distort`, `blur`,
    `bumpmap`, `intensity`.
  - **Placement** ([`Composite::compose`], pictures of different
    extents): `with_offset(x, y)` (negative allowed, clipped),
    `with_gravity` (nine anchors; the offset is added to it),
    `with_tile(true)` (the overlay repeated across the whole destination
    from the placement point), `with_outside_overlay(false)` (leave the
    destination untouched outside the overlay), `with_channels`.
  - Every depth: the operators are evaluated in binary32 on straight
    alpha; 8-bit, 16-bit (`Gray16Le` / `Rgb48Le` / `Rgba64Le` come back
    as words) and float destinations keep their layout.
    `tests/depth_two_input.rs` holds `over` and eight blend functions
    to their closed forms to the 16-bit code, and every placement mode
    exactly; `tests/magick_composite.rs` compares all operators with
    `magick` on partial-alpha pictures.
  - documented-CLI analogue: `-compose <op> -composite`.

All single-input filters listed above share the `ImageFilter` trait — chain them
manually with repeated `.apply()` calls, or feed them through
`oxideav-pipeline` as `video.<name>` filter nodes in a JSON job.

## Distortion and generators

`DistortImage` is the `-distort` framework: Affine, AffineProjection,
SRT, Perspective, PerspectiveProjection, BilinearForward / Reverse,
Polynomial, Arc, Polar, DePolar, Barrel, BarrelInverse and Shepards, with
`+distort` bestfit and viewports, 14 virtual-pixel methods, 9 point
interpolations and elliptical weighted averaging through any resize
filter (Robidoux by default). The conventions were recovered from
`magick` with pictures that encode their own coordinates;
`tests/magick_distort.rs` holds every method at ≥ 45 dB with point
sampling (most ≥ 100 dB) and EWA at ≥ 35 dB (55–73 dB measured).

Also from the `-fx` family of operations: `Fx` (an expression evaluator,
46 forms matching `magick`), `SparseColor` (six methods, bit-exact),
`connected_components` / `ConnectedComponents`, `Kmeans`, `AddNoise`
(seven distributions matched in mean and deviation), `fourier`
(magnitude / phase DFT pair), `OrderedDither` (threshold maps with
per-channel levels, bit-exact) and `Transpose` / `Transverse`.

More of the Imagick surface, each rule measured on `magick` 7.1.2 and
held by `tests/magick_ops.rs`: the threshold family (`WhiteThreshold`,
`BlackThreshold`, `RangeThreshold`, `ColorThreshold`,
`RandomThreshold`), `Splice` / `Chop` with gravity, `Raise`, `Shear`,
`Sample` / `Scale`, `Cdl`, and the image-list operations
`evaluate_sequence` (15 operators), `separate` and `combine_channels`.
`OrderedDither` carries every map `magick -list threshold` names, their
cells recovered black-box and bit-exact at any level count.

## Measurement

`measure` is the read-only side of the Imagick surface: `statistics`
(per-channel and overall min / max / mean / median / standard deviation /
kurtosis / skewness / entropy, with `identify -verbose`'s conventions —
sample deviation, excess kurtosis, entropy over the occupied bins),
`compare` with every `compare -metric` (AE, MAE, MEPP, MSE, NCC, PAE,
PSNR, RMSE, SSIM, DSSIM, Fuzz, PHASH — the joint `(1 − NCC) / 2` and
`(1 − SSIM) / 2` figures `magick` prints), `difference_image`,
`similarity_search`, `moments` (centroid, inertia ellipse, Hu I1–I8, per
channel and pooled as `identify -moments` pools them),
`perceptual_hash`, `histogram` / `unique_colors` and `trim_bounds`. Every
layout the float hop reaches is measured in normalised units; the live
test compares all of it against `magick` on `rose:`. `Fuzz` and the
perceptual hash follow the documented recipes but not `magick`'s figures
to the digit (its colour-space conversions are not staged).

## Depth support

Every filter runs at the precision of the picture it is given. An 8-bit
picture takes the filter's 8-bit path; a 16-bit picture (`Gray16Le`,
`Rgb48Le`, `Rgba64Le`, `Ya16Le`, the 10/12/14/16-bit planar YUV(A) and
GBR(A) families) and a float picture run the filter's binary32 kernel —
16-bit words are normalised (`code / 65535`), filtered with no
intermediate quantisation and rounded back once, so nothing passes
through 8 bits. The output layout is the input layout (a filter that
changes layout — an edge detector, a channel extractor — emits its grey
counterpart at the same precision). Alpha is carried at depth and, where
a kernel averages colour across pixels of different coverage (`Blur`,
`Sharpen`, `Unsharp`, `Resize` on the packed layouts, the distortions'
EWA), the colour is weighted by it.

155 of 170 catalogued filters take 16-bit and float pictures; the rest refuse a deep picture with a typed
`Unsupported` (never a silent narrowing). `tests/depth_support_table.rs`
regenerates this table from the filters' own declarations and fails
when a row here is out of date; `tests/depth_analytic.rs` and
`tests/depth_two_input.rs` hold the 8-bit, 16-bit and float results of
blur, statistic, despeckle, morphology, paint, both dither families and
the two-input filters to their closed-form definitions, with and
without alpha; `tests/deep_parity.rs` holds every other float path to
its 8-bit result on widened pictures.

The mask operators (the distance transforms, `SignedDistanceField`,
`VoronoiTransform`, `ProximityFill`, `DistanceMorphology`,
`DistanceOutline`) read a picture only through an integer threshold on
the 8-bit scale and write a result that is 8-bit codes by definition (a
distance in pixels, a label, a binary mask). A deep picture is
thresholded at its own precision — code 32 768 of 65 535 is 127.5 on the
8-bit scale and stays below a threshold of 128 — and the result codes
come back as `code / 255` in the picture's layout.
`WeightedDistanceTransform` seeds its continuous cost from the deep
sample directly.

| filter | 8-bit | 16-bit | float | why not |
|---|---|---|---|---|
| `AdaptiveThreshold` | yes | yes | yes |  |
| `Affine` | yes | yes | yes |  |
| `AnisotropicBlur` | yes | yes | yes |  |
| `AutoGamma` | yes | yes | yes |  |
| `AutoLevel` | yes | yes | yes |  |
| `AutoTrim` | yes | yes | yes |  |
| `BarrelInverse` | yes | yes | yes |  |
| `BilateralBlur` | yes | yes | yes |  |
| `BlackThreshold` | yes | yes | yes |  |
| `Bloom` | yes | yes | yes |  |
| `BlueShift` | yes | yes | yes |  |
| `Blur` | yes | yes | yes |  |
| `BorderedFrame` | yes | yes | yes |  |
| `BrightnessContrast` | yes | yes | yes |  |
| `BwMix` | yes | yes | yes |  |
| `Canny` | yes | yes | yes |  |
| `Canvas` | yes | yes | yes |  |
| `ColorThreshold` | yes | yes | yes |  |
| `ChannelExtract` | yes | yes | yes |  |
| `ChannelExtract(U)` | yes | no | no | n/a: reads a chroma plane of 8-bit planar YUV; the deep layouts reach filters as packed RGB / grey |
| `ChannelMixer` | yes | yes | yes |  |
| `Charcoal` | yes | yes | yes |  |
| `ChromaticAberration` | yes | yes | yes |  |
| `Clamp` | yes | yes | yes |  |
| `Cdl` | yes | yes | yes |  |
| `Chop` | yes | yes | yes |  |
| `Clarity` | yes | yes | yes |  |
| `ColorBalance` | yes | yes | yes |  |
| `ColorMatrix` | yes | yes | yes |  |
| `Colorize` | yes | yes | yes |  |
| `Comic` | yes | no | no | 8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal) |
| `Composite(self)` | yes | yes | yes |  |
| `ConnectedComponents` | yes | yes | yes |  |
| `ContrastStretch` | yes | yes | yes |  |
| `Convolve` | yes | yes | yes |  |
| `Crop` | yes | yes | yes |  |
| `CrossProcess` | yes | yes | yes |  |
| `Crystallize` | yes | no | no | 8-bit only: float path not written yet (typed refusal, never narrowed) |
| `Curves` | yes | yes | yes |  |
| `Cyanotype` | yes | yes | yes |  |
| `Cycle` | yes | yes | yes |  |
| `Deskew` | yes | no | no | 8-bit only: analysis on an 8-bit luma / edge map; float path not written yet (typed refusal) |
| `Despeckle` | yes | yes | yes |  |
| `DistanceMorphology` | yes | yes | yes |  |
| `DistanceOutline` | yes | yes | yes |  |
| `DistanceTransform` | yes | yes | yes |  |
| `Distort` | yes | yes | yes |  |
| `DistortImage` | yes | yes | yes |  |
| `Dither` | yes | yes | yes |  |
| `Drago` | yes | yes | yes |  |
| `DropShadow` | yes | yes | yes |  |
| `Edge` | yes | yes | yes |  |
| `EdgeDetect` | yes | yes | yes |  |
| `Emboss` | yes | yes | yes |  |
| `EmbossDirectional` | yes | yes | yes |  |
| `Equalize` | yes | yes | yes |  |
| `EuclideanDistanceTransform` | yes | yes | yes |  |
| `Evaluate` | yes | yes | yes |  |
| `Exposure` | yes | yes | yes |  |
| `Extent` | yes | yes | yes |  |
| `Feather` | yes | no | no | 8-bit only: float path not written yet (typed refusal, never narrowed) |
| `VoronoiTransform` | yes | yes | yes |  |
| `ProximityFill` | yes | yes | yes |  |
| `Flip` | yes | yes | yes |  |
| `FloodFill` | yes | no | no | 8-bit only: float path not written yet (typed refusal, never narrowed) |
| `Flop` | yes | yes | yes |  |
| `Frame` | yes | yes | yes |  |
| `FreiChen` | yes | no | no | 8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal) |
| `Function` | yes | yes | yes |  |
| `Fx` | yes | yes | yes |  |
| `Gabor` | yes | no | no | 8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal) |
| `Gamma` | yes | yes | yes |  |
| `GradientConic` | yes | yes | yes |  |
| `GradientMap` | yes | yes | yes |  |
| `GradientRadial` | yes | yes | yes |  |
| `GravityTranslate` | yes | yes | yes |  |
| `Grayscale` | yes | yes | yes |  |
| `Grayscale(gray8)` | yes | yes | yes |  |
| `Hable` | yes | yes | yes |  |
| `Halftone` | yes | no | no | 8-bit only: float path not written yet (typed refusal, never narrowed) |
| `Heatmap` | yes | yes | yes |  |
| `HoughCircles` | yes | no | no | 8-bit only: analysis on an 8-bit luma / edge map; float path not written yet (typed refusal) |
| `HoughLines` | yes | no | no | 8-bit only: analysis on an 8-bit luma / edge map; float path not written yet (typed refusal) |
| `HslRotate` | yes | yes | yes |  |
| `HslShift` | yes | yes | yes |  |
| `Implode` | yes | yes | yes |  |
| `InnerShadow` | yes | no | no | 8-bit only: float path not written yet (typed refusal, never narrowed) |
| `Kmeans` | yes | yes | yes |  |
| `Kuwahara` | yes | yes | yes |  |
| `Laplacian` | yes | yes | yes |  |
| `LaplacianOfGaussian` | yes | yes | yes |  |
| `Level` | yes | yes | yes |  |
| `LinearStretch` | yes | yes | yes |  |
| `MaxRgb` | yes | yes | yes |  |
| `Modulate` | yes | yes | yes |  |
| `Morphology` | yes | yes | yes |  |
| `MorphologyChain` | yes | yes | yes |  |
| `MorphologyEdge` | yes | yes | yes |  |
| `MotionBlur` | yes | yes | yes |  |
| `Negate` | yes | yes | yes |  |
| `Niblack` | yes | yes | yes |  |
| `AddNoise` | yes | yes | yes |  |
| `Normalize` | yes | yes | yes |  |
| `OrderedDither` | yes | yes | yes |  |
| `OtsuThreshold` | yes | yes | yes |  |
| `Paint` | yes | yes | yes |  |
| `Perspective` | yes | yes | yes |  |
| `Pixelate` | yes | yes | yes |  |
| `Polar` | yes | yes | yes |  |
| `Posterize` | yes | yes | yes |  |
| `PosterizeChannels` | yes | yes | yes |  |
| `Prewitt` | yes | yes | yes |  |
| `Quantize` | yes | yes | yes |  |
| `RadialBlur` | yes | yes | yes |  |
| `Reinhard` | yes | yes | yes |  |
| `ReinhardExtended` | yes | yes | yes |  |
| `ReinhardLocal` | yes | yes | yes |  |
| `Resize` | yes | yes | yes |  |
| `Roberts` | yes | yes | yes |  |
| `Roll` | yes | yes | yes |  |
| `Rotate` | yes | yes | yes |  |
| `Scharr` | yes | yes | yes |  |
| `SelectiveColor` | yes | yes | yes |  |
| `Sample` | yes | yes | yes |  |
| `Scale` | yes | yes | yes |  |
| `Sepia` | yes | yes | yes |  |
| `Shade` | yes | yes | yes |  |
| `Shade(color)` | yes | yes | yes |  |
| `ShadowHighlight` | yes | yes | yes |  |
| `Sharpen` | yes | yes | yes |  |
| `Shave` | yes | yes | yes |  |
| `Shear` | yes | yes | yes |  |
| `SigmoidalContrast` | yes | yes | yes |  |
| `SignedDistanceField` | yes | yes | yes |  |
| `Sketch` | yes | no | no | 8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal) |
| `Solarize` | yes | yes | yes |  |
| `SplitTone` | yes | yes | yes |  |
| `SparseColor` | yes | yes | yes |  |
| `Splice` | yes | yes | yes |  |
| `Spread` | yes | yes | yes |  |
| `SrgbTransform` | yes | yes | yes |  |
| `Srt` | yes | yes | yes |  |
| `Statistic` | yes | yes | yes |  |
| `Swirl` | yes | yes | yes |  |
| `Temperature` | yes | yes | yes |  |
| `Threshold` | yes | yes | yes |  |
| `RandomThreshold` | yes | yes | yes |  |
| `RangeThreshold` | yes | yes | yes |  |
| `TiltShift` | yes | yes | yes |  |
| `Tint` | yes | yes | yes |  |
| `Toon` | yes | no | no | 8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal) |
| `Transpose` | yes | yes | yes |  |
| `Transverse` | yes | yes | yes |  |
| `Trim` | yes | yes | yes |  |
| `Unsharp` | yes | yes | yes |  |
| `Vibrance` | yes | yes | yes |  |
| `Vignette` | yes | yes | yes |  |
| `VignetteSoft` | yes | yes | yes |  |
| `Wave` | yes | yes | yes |  |
| `WhiteThreshold` | yes | yes | yes |  |
| `WeightedDistanceTransform` | yes | yes | yes |  |
| `ZoomBlur` | yes | yes | yes |  |
| `Clut (two-input)` | yes | yes | yes |  |
| `Composite (two-input)` | yes | yes | yes |  |
| `Difference (two-input)` | yes | yes | yes |  |
| `DisplacementMap (two-input)` | yes | yes | yes |  |
| `HaldClut (two-input)` | yes | yes | yes |  |
| `Stegano (two-input)` | yes | no | no | n/a: hides a picture in the low bits of 8-bit codes; defined on that storage |
| `Stereo (two-input)` | yes | yes | yes |  |
| `Watermark (two-input)` | yes | yes | yes |  |

## Thread budget

Filters are serial unless the caller grants a budget:

```rust
# use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
# let frame = VideoFrame { pts: None, planes: vec![VideoPlane { stride: 64, data: vec![9; 64 * 16] }] };
use oxideav_image_filter::{Blur, ImageFilter, VideoStreamParams, with_execution_context};

let params = VideoStreamParams { format: PixelFormat::Rgba, width: 16, height: 16 };
let ctx = ExecutionContext::with_threads(8);
let out = with_execution_context(&ctx, || Blur::new(3).apply(&frame, params))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Inside the closure every filter call on this thread may fan out over at
most `ctx.threads` scoped threads. The split is by whole output rows,
each computed exactly as the serial loop computes it, so the output is
identical at every budget (held by `tests/depth_analytic.rs`). Kernels
that split: the separable convolution (`Blur`, `Sharpen`, `Unsharp`,
`TiltShift`, `Charcoal`, `Canny`), the window filters (`Statistic`,
`Despeckle`, `Morphology`, `Paint`, `Bloom`), `Resize`, `Rotate`,
`Composite`, `ColorMatrix`, `AnisotropicBlur`, the ordered mode of
`Dither`, the float path of the pointwise tonal filters, and the 16-bit
/ float packing every deep path goes through. Error diffusion is serial
by definition (each pixel depends on the ones before it).

## Performance

Wall-clock milliseconds per picture (`benches/throughput.rs`, `Rgba` and
`Rgba64Le`, 1152 × 868 and 4000 × 3000, medians, one Apple-silicon
laptop shared with other builds — read the ratios, not the digits).
"Before" is release 0.1.5, serial; "after" is serial unless a thread
count is given.

| filter | depth | 1 MP before | 1 MP after | 1 MP, 8 threads | 12 MP before | 12 MP after | 12 MP, 8 threads |
|---|---|---:|---:|---:|---:|---:|---:|
| blur r3 s1.5 | 8-bit | 32.2 | 7.1 | 2.2 | 383 | 89.4 | 16.6 |
| blur r12 s6 | 8-bit | 111 | 16.4 | 4.5 | 1268 | 203 | 37.2 |
| resize half lanczos3 | 8-bit | 16.2 | 10.7 | 3.3 | 172 | 125 | 20.3 |
| resize 1.5x bicubic | 8-bit | 53.0 | 53.6 | 10.9 | 538 | 577 | 90.4 |
| sharpen r2 s1 | 8-bit | 27.6 | 7.6 | 3.4 | 264 | 82.6 | 31.9 |
| unsharp r3 s1.5 | 8-bit | 35.0 | 8.1 | 3.4 | 343 | 93.2 | 33.8 |
| morphology dilate sq x2 | 8-bit | 53.3 | 1.7 | 0.7 | 512 | 21.6 | 4.6 |
| morphology erode disk r5 | 8-bit | — | 6.4 | 1.9 | — | 79.1 | 14.4 |
| statistic median 5x5 | 8-bit | 256 | 69.7 | 10.4 | 2760 | 816 | 119 |
| statistic max 9x9 | 8-bit | 363 | 2.6 | 0.8 | 3951 | 29.6 | 5.4 |
| statistic mean 9x9 | 8-bit | 281 | 9.3 | 3.4 | 3230 | 110 | 30.1 |
| rotate 17deg | 8-bit | 14.7 | 15.0 | 5.2 | 189 | 182 | 67.9 |
| colour-matrix | 8-bit | 2.7 | 2.8 | 0.9 | 34.2 | 36.8 | 6.0 |
| threshold | 8-bit | 0.6 | 0.6 | 0.6 | 6.5 | 7.1 | 7.0 |
| dither floyd-steinberg | 8-bit | 34.2 | 29.1 | 30.5 | 408 | 495 | 354 |
| ordered dither o8x8 | 8-bit | 6.5 | 6.5 | 7.0 | 77.0 | 85.8 | 80.7 |
| composite over | 8-bit | 5.2 | 6.6 | 3.9 | 65.2 | 83.8 | 36.2 |
| blur r3 s1.5 | 16-bit | 42.8 | 8.0 | 3.2 | 492 | 113 | 25.3 |
| blur r12 s6 | 16-bit | 122 | 17.3 | 5.6 | 1317 | 239 | 44.5 |
| resize half lanczos3 | 16-bit | 16.6 | 10.4 | 3.4 | 176 | 125 | 20.9 |
| resize 1.5x bicubic | 16-bit | 57.8 | 50.6 | 11.4 | 628 | 604 | 102 |
| sharpen r2 s1 | 16-bit | 40.7 | 9.0 | 4.8 | 419 | 112 | 46.2 |
| unsharp r3 s1.5 | 16-bit | 46.7 | 10.1 | 5.0 | 642 | 127 | 48.3 |
| morphology dilate sq x2 | 16-bit | 60.3 | 6.9 | 2.9 | 790 | 83.8 | 25.9 |
| morphology erode disk r5 | 16-bit | — | 24.9 | 7.7 | — | 330 | 73.8 |
| statistic median 5x5 | 16-bit | 537 | 148 | 26.0 | 6086 | 1691 | 304 |
| statistic max 9x9 | 16-bit | 278 | 8.9 | 3.3 | 3252 | 100 | 29.0 |
| statistic mean 9x9 | 16-bit | 364 | 11.8 | 3.7 | 4558 | 138 | 38.7 |
| rotate 17deg | 16-bit | 37.6 | 22.4 | 22.3 | 415 | 265 | 258 |
| colour-matrix | 16-bit | 12.4 | 3.2 | 1.3 | 148 | 36.5 | 17.9 |
| threshold | 16-bit | 12.7 | 3.2 | 1.3 | 158 | 36.9 | 17.7 |
| dither floyd-steinberg | 16-bit | refused | 29.7 | 30.0 | refused | 364 | 353 |
| ordered dither o8x8 | 16-bit | 18.4 | 7.8 | 6.4 | 196 | 83.5 | 79.4 |
| composite over | 16-bit | refused | 8.1 | 3.3 | refused | 88.9 | 36.5 |

What changed: separable kernels keep a binary32 intermediate and run
whole rows at a time (auto-vectorised); the median is a sliding
histogram instead of a per-pixel sort; minimum / maximum are whole-row
running extremes (block scans past 32 taps); the mean is a prefix-sum
difference; morphology collapses iterations into one larger element; the
oil-paint window slides; the 16-bit float hop is done in-crate; nothing
allocates per pixel. `Composite` and `Dither` were refused at 16 bits
before.

## Pixel layouts and the frame contract

Every filter runs behind one wrapper (`PixelFilter` → blanket
`ImageFilter`, `TwoInputPixelFilter` → blanket `TwoInputImageFilter`)
that enforces the `oxideav-core` 0.1.40 frame contract before a filter
body sees a sample:

1. **Layout gate.** Each filter declares the layouts it takes natively
   (`accepted_formats()`); `ImageFilter::accepted_pixel_formats()` lists
   every layout it takes at all (natively or through a route below) and
   `output_pixel_format(input)` the layout it emits. A layout outside
   that set is a typed `Error::Unsupported` naming the layout — never a
   panic, never a silent narrowing or widening.
2. **Frame validation.** The frame must hold exactly the format's image
   planes, each `stride × rows` bytes with `stride ≥` its visible row,
   for a non-empty extent; otherwise a typed `Error::InvalidData`.
3. **Side channels.** A frame may end with core's side-channel records
   (metadata blobs, palette, palette alpha, per-plane significant bits,
   colour signal, layer identity — entries whose `stride` no image plane
   can have). They are stripped before the filter body runs and
   re-attached to the output under this ruling:

   | record | carried when |
   |---|---|
   | metadata blobs | always, byte-for-byte |
   | layer identity | always |
   | colour signal | the output stays in the input's carriage (YUV → YUV, RGB/grey → RGB/grey) |
   | significant bits | the output layout equals the input layout |
   | palette + palette alpha | the output is still `Pal8` (indices moved, not recoloured); the palette route rewrites them |

   Two-input filters carry port 0's (`src`) records.

### Routes

| route | when | output label |
|---|---|---|
| native | the layout is in the filter's declared set | `output_format(input)` |
| palette | `Pal8`, a pointwise filter (36 declare `is_pointwise()`): the palette is filtered as a one-row `Rgba` picture, alpha from the palette-alpha record, the index plane untouched | `Pal8` with the recoloured table |
| packed-RGB hop | any other **8-bit** layout `oxideav-pixfmt` converts (planar / semi-planar YUV(A) at either range with the frame's colour signal honoured, `Bgr24` / `Bgra` / `Argb` / `Abgr`, `Ya8`, `Yuyv422` / `Uyvy422`, `Cmyk`, bit-packed mono, `Pal8` for a non-pointwise filter) into the first of `Rgba` / `Rgb24` / `Gray8` the filter takes, and back when the filter keeps that layout | the input layout; the hop layout (or the filter's own, e.g. `Gray8`) when it does not convert back or the source was `Pal8` |
| float hop | any 10/12/14/16-bit or float layout through a filter that takes the packed float layouts: converted to `RgbaF32Le` / `RgbF32Le` / `GrayF32Le` (the packed 16-bit words in-crate, `code / 65535`; everything else by `oxideav-pixfmt`), filtered in binary32, converted back | the input layout (the float layout when the filter changes layout) |
| refusal | 10/12/14/16-bit words and float through a filter without a float path (they are never narrowed); the macropixel / mono layouts through a filter with no packed target | typed `Unsupported` |

Packed 4:2:2 and 4:1:1 take the hop at the widths pixfmt accepts (even /
multiple of 4) on both legs; otherwise a typed `InvalidData`.

### Who takes what

| filters | layouts |
|---|---|
| `Flip`, `Flop`, `Crop` (and `Shave`), `Roll`, `Rotate` by a quarter turn, `Resize` with `Nearest` | all 66 byte-addressable layouts natively (16-bit words move as words, palette indices with their table, `Nv12` pairs as pairs, `Yuva*` alpha at picture resolution, 4:2:2 / 4:4:0 / 4:1:1 chroma on their own grids) + the 4 others through the hop |
| `Resize` with an interpolating kernel | the 65 byte-addressable layouts but `Pal8` natively, depth-aware (255 / `2^bits − 1` / float unclamped; chroma on its own grid); `Pal8` through the hop (emits `Rgba`) |
| the per-plane classics (`Blur`, `Sharpen`, `Unsharp`, `Emboss`, `MotionBlur`, `Negate`, `Gamma`, `Level`, `Equalize`, `Threshold`, the edge detectors, `Statistic`, `Pixelate`, `Canvas`, `Extent`, `Trim`, …) | `Gray8`, `Rgb24`, `Rgba`, `Yuv420P` / `Yuv422P` / `Yuv444P` and the `YuvJ*` trio natively; every other 8-bit layout through the hop |
| the packed-RGB colour filters (`Sepia`, `Tint`, `Modulate`, the HSL / balance / curve / tone-mapping families, `Kuwahara`, `Swirl`, …) | `Rgb24` / `Rgba` (some `Gray8`) natively; every other 8-bit layout through the hop; `Pal8` through the palette route when pointwise |
| the mask operators (`DistanceTransform`, `EuclideanDistanceTransform`, `SignedDistanceField`, `VoronoiTransform`, `ProximityFill`, `WeightedDistanceTransform`) | `Gray8` natively; every other 8-bit layout through the hop (luma) |
| `ChannelExtract` | per channel: `Red` grey + packed RGB(A); `Green` / `Blue` packed RGB(A); `Alpha` `Rgba`; `Y` grey + planar YUV(J); `U` / `V` planar YUV(J) (no hop for `U` / `V`) |
| two-input (`Composite`, `Clut`, `Watermark`, `DisplacementMap`: `Gray8` / `Rgb24` / `Rgba` + float; `HaldClut`, `Stereo`: `Rgb24` / `Rgba` + float; `Difference`: the classic nine + float; `Stegano`: 8-bit packed) | natively, and through the same packed-RGB and float hops as single-input filters (both pictures converted, the result converted back) |

The deep path: every filter whose native set includes the packed float
layouts (`GrayF32Le` / `RgbF32Le` / `RgbaF32Le` — all but a handful of
mask / distance / analysis operators) takes every 16-bit and float
layout through the **float hop**: converted by `oxideav-pixfmt` (pure
normalisation for RGB / grey / GBR, the matrix at depth for YUV),
filtered in binary32 with no intermediate quantisation, converted back
saturating. `tests/deep_parity.rs` holds every such filter to its 8-bit
result within one code on widened pictures, and a 16-bit level → gamma
chain to thousands of distinct codes.

In numbers: 162 catalogued filter instances × 70 layouts × 2 sizes =
22 680 cells, 20 872 accepted, the rest typed refusals; the 8 two-input
filters accept 519 of their 560 layout cells. The registry labels every output port from the
filter's own declaration; filters whose extent depends on the picture
(`trim`, `auto-trim`, `deskew`) mirror the input extent on the port.

### Proof

`tests/layout_matrix.rs` runs every filter instance over all 70
layouts at an even and an odd size on a picture carrying all six
records (22 680 cells: 20 872 accepted, the rest typed refusals, zero
panics) and checks each pointwise flag pixel by pixel and palette entry
by palette entry; `tests/registry_truth.rs` builds all 300 factory
names on seven layouts and checks the emitted planes against the
declared port; `tests/geometry_layouts.rs` and `tests/resize_depth.rs`
are pixel-exact on the widened paths; `fuzz/` drives the same two
surfaces with arbitrary layouts, plane shapes, records and factory
parameters (daily `Fuzz` workflow).

## Mid-stream shape changes

Every filter built through the registry answers core's in-band
`Frame::StreamParams` event (oxideav-core 0.1.43) on any input port.
One wrapper serves all of them: since each factory derives its whole
plan from its input port specs, the wrapper re-runs the filter's own
factory with the same parameters and the updated input port, so the
re-planned filter is exactly the one a pipeline built for the new shape
from the start would get.

- **Re-plan.** `input_ports()` / `output_ports()` move to the new
  shape. On every output port whose shape (width, height, layout)
  changes — or whose shape follows its input's, the pass-through case —
  the filter emits its own `Frame::StreamParams` (the upstream event's
  parameters with the port's picture shape) before the first frame of
  the new shape. A fixed-target scale whose output does not move
  announces nothing.
- **Refuse.** When the factory cannot plan for the new input — a crop
  window that no longer fits, a layout the filter does not take, a
  two-input filter whose inputs must agree on `(format, width, height)`
  and now would not — the event fails with `Error::Unsupported` naming
  the filter and both shapes; nothing is emitted and the previous plan
  stands.
- **State.** Two-input filters hold the frame waiting on each port. A
  re-plan hands the frames held on the unchanged ports to the new plan
  (`watermark`'s overlay survives a base-size change) and drops the
  changed port's old-shape frame; `reset()` drops them all.

`tests/stream_params.rs` covers scale, crop, colour collapse, a
convolution, transpose, compose and the overlay buffer (frames after
the event equal those of a filter planned for the new shape);
`tests/registry_truth.rs` drives every registered name through a size
change and a layout change (447 re-planned cells, 77 typed refusals —
the composite family and the shape-matched two-input filters — and
every announced shape equals the re-declared port and the next frame's
geometry).

## License

MIT. See [LICENSE](LICENSE).
