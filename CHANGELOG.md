# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  tolerance — matching ImageMagick's `-fuzz`; iterative span-fill keeps
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
  `Vibrance` (Lr-style saturation boost that spares already-saturated
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

- r15: land six new ImageMagick-compatible filters covering colour
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

- r14: land six new ImageMagick-compatible filters covering edge-aware
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

- r13: land seven new ImageMagick-style filters: five single-input
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

- r12: land seven new ImageMagick-style filters covering tonal range
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

- r11: land five new ImageMagick-style filters plus four overlay-family
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

- r10: land six ImageMagick-style geometric / channel / morphology
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
- Implement four new ImageMagick-style filters and wire factories into
  `register()`: `Vignette` (Gaussian radial darkening; Rgb24/Rgba),
  `Colorize` (linear blend toward a target colour by `amount`;
  Rgb24/Rgba), `Equalize` (per-channel CDF histogram equalisation;
  Gray8/Rgb24/Rgba/YUV planar), and `AutoGamma` (geometric-mean-driven
  per-channel gamma correction; Gray8/Rgb24/Rgba/YUV planar). Alpha is
  preserved on Rgba; YUV runs only on the luma plane for Equalize /
  AutoGamma.
- Implement five more ImageMagick-style filters and wire factories into
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
