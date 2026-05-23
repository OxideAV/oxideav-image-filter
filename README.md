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
  exact axis-aligned fast path. IM: `-rotate N`.
- **`Crop`** — `(x, y, width, height)` subregion extraction; chroma
  rects are ceil/floor-aligned for YUV subsampling. IM: `-crop WxH+X+Y`.
- **`Roll`** — circular pixel shift `(dx, dy)`; rows / columns wrap
  around the borders. Chroma offsets on planar YUV are scaled by the
  subsampling factor so the visible image translates as a rigid block.
  IM: `-roll +X+Y`.
- **`Shave`** — strip a uniform `(x_border, y_border)` margin off
  every edge (centred crop). Backed by `Crop` so YUV chroma alignment
  matches. IM: `-shave XxY`.
- **`Extent`** — set the output canvas to a fixed `(width, height)`
  with a placement offset, padding the gaps with a configurable
  background colour. Negative offsets translate the input toward the
  upper-left so its right / bottom edge can land inside the window.
  IM: `-extent WxH+X+Y`.
- **`Trim`** — auto-crop to the bounding box of pixels that differ
  from a reference background colour by more than `fuzz` per channel
  (Chebyshev / L-infinity tolerance). Background defaults to the
  source's `(0, 0)` pixel. IM: `-fuzz N% -trim`.
- **`AutoTrim`** — like `Trim`, but picks the dominant background
  colour via a 4-bit-per-channel histogram vote across the whole
  frame instead of trusting the `(0, 0)` corner sample. Noisy
  corners no longer poison the inferred background.
- **`GravityTranslate`** + **`Gravity`** — 9-point compass anchor
  placement on a fixed `(width, height)` canvas (`NorthWest` /
  `North` / `NorthEast` / `West` / `Centre` / `East` / `SouthWest` /
  `South` / `SouthEast`). Backed by `Extent` so YUV chroma
  subsampling alignment matches. Factory aliases: `gravity-translate`,
  `gravity`. IM: `-gravity <anchor> -extent WxH`.

### Tonal (LUT-based, typically fast)

- **`Negate`** — invert pixel values. On YUV inverts only Y so
  chroma (hue/saturation) is preserved. IM: `-negate`.
- **`Threshold`** — binarise at a per-channel threshold; chroma →
  neutral 128 for YUV. IM: `-threshold N`.
- **`Gamma`** — power-law gamma correction. IM: `-gamma G`.
- **`BrightnessContrast`** — linear brightness + contrast in the IM
  range `−100..100`. IM: `-brightness-contrast BxC`.
- **`Level`** — remap `[black, white]` to `[0, 255]` with optional
  mid-tone gamma. IM: `-level LOW,MID,HIGH`.
- **`Normalize`** — two-pass auto-level stretch to use the full
  range. IM: `-normalize`.
- **`Posterize`** — reduce each channel to `N` intensity levels. IM:
  `-posterize N`.
- **`Solarize`** — invert pixels above a threshold. IM: `-solarize N%`.
- **`AdaptiveThreshold`** — local-mean-based binarisation with a
  configurable `(2*radius+1)²` window and signed offset; box-sum
  implementation is `O(W*H)` regardless of radius. Always emits
  `Gray8` (RGB collapsed via Rec. 601 luma). IM: `-threshold local`.
- **`OtsuThreshold`** — global automatic threshold maximising
  inter-class variance (1979 Otsu paper). Optional `invert` flag.
  Emits binary `Gray8` (RGB / RGBA collapsed via Rec. 601 luma; YUV
  reads the Y plane). Complement to `AdaptiveThreshold` (local window)
  and `Threshold` (manual cut). Factory aliases: `otsu-threshold`,
  `otsu`.
- **`Equalize`** — per-channel histogram equalisation via CDF
  mapping. Luma-only on YUV. IM: `-equalize`.
- **`AutoGamma`** — pick a per-channel gamma so the geometric mean
  lands at mid-grey 0.5 (`gamma = log(mean) / log(0.5)`). IM:
  `-auto-gamma`.
- **`SigmoidalContrast`** — sigmoid-curve contrast around a midpoint.
  IM: `-sigmoidal-contrast CxM%`.
- **`Exposure`** — EV-stop adjustment in linear-light space (sRGB →
  linear → ×2^EV → sRGB). `+1.0` EV doubles the captured light,
  `-1.0` EV halves it. RGB / RGBA / Gray8. IM analogue: `-colorspace
  RGB -evaluate Multiply k -colorspace sRGB` folded into one LUT pass.
- **`ShadowHighlight`** — independent shadow lift + highlight recovery
  gated by a soft tonal mask (peaks at the extremes, zero at midtone).
  Mid-grey passes through unchanged. RGB / RGBA / Gray8 / YUV (luma
  only). Lr-style "Shadows" / "Highlights" sliders.
- **`Evaluate`** + **`EvaluateOp`** — per-pixel arithmetic LUT with a
  single scalar operand. Operators: `Add`, `Subtract`, `Multiply`,
  `Divide`, `Pow`, `Max`, `Min`, `Set`, `And`, `Or`, `Xor`,
  `Threshold`. Alpha (RGBA) and chroma (YUV) pass through. IM:
  `-evaluate <op> N`.
- **`Cycle`** — modular per-channel value rotation
  (`out = (src + amount) mod 256`); alpha and chroma preserved. IM
  analogue: `-cycle N`.
- **`Statistic`** + **`StatisticOp`** — rolling-window per-pixel
  `Median` / `Min` / `Max` / `Mean` over a `WxH` neighbourhood
  (border-clamped). 1×1 window is identity. IM: `-statistic <op> WxH`.
- **`Clamp`** — clamp every tone sample into `[low, high]`. Alpha
  preserved on RGBA; YUV touches only luma. IM: `-clamp` (extended
  with explicit endpoints).
- **`AutoLevel`** — per-channel auto-stretch: independently fill
  `[0, 255]` for each of R / G / B (RGB / RGBA). IM: `-auto-level`.
- **`ContrastStretch`** — burn the darkest `black%` and brightest
  `white%` of pixels then linearly stretch the rest, **per channel**
  for RGB. IM: `-contrast-stretch black%xwhite%`.
- **`LinearStretch`** — like `ContrastStretch` but cut-offs are
  absolute pixel counts, not fractions. IM:
  `-linear-stretch black-pixels{xwhite-pixels}`.
- **`Function`** + **`FunctionOp`** — per-pixel mathematical-function
  map evaluated in normalised `[0, 1]` space. Operators:
  `Polynomial` (Horner-rule on descending coefficients), `Sinusoid`
  (`bias + amp · sin(2π · (freq · x + phase / 360))`), `ArcSin`,
  `ArcTan`. IM: `-function <kind> args`.
- **`Curves`** + **`Curve`** + **`CurveInterpolation`** — per-channel
  tonal curves through user-supplied `(x, y)` control points.
  Three interpolants: `Linear` (segment-wise), `CatmullRom` (1974
  Catmull-Rom cubic), and `MonotoneCubic` (1980 Fritsch-Carlson
  monotonicity-preserving Hermite — default; never overshoots).
  Master curve runs on every tone channel; optional `red` / `green`
  / `blue` overrides on RGB / RGBA. Gray8 / RGB / RGBA + planar YUV
  (master curve on luma only). Cost is `O(W·H)` (per-channel 256-LUT).

### Tone mapping (HDR-style)

Global tone-mapping operators for high-contrast scenes; all do the
sRGB ↔ linear-light round-trip internally so the curve runs on
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

### Distance / signed-distance

- **`DistanceTransform`** — Borgefors 1986 3-4 chamfer distance
  transform. Two sequential passes (forward top-left → bottom-right,
  backward bottom-right → top-left) compute the chamfer-3-4 metric
  on a binary mask thresholded from the input intensity. Output is
  `Gray8` whose value is the (scaled, clamped) distance from each
  pixel to the nearest foreground pixel; useful for stroke
  thickening, signed-distance fields, contour render. Gray8 input
  only. Factory aliases: `distance-transform`, `distance`.

### Colour

- **`Modulate`** — brightness / saturation / hue via HSL round-trip.
  RGB / RGBA only (YUV returns `Unsupported`). IM: `-modulate B,S,H`.
- **`Sepia`** — sepia-tone matrix with optional `threshold` mix back
  to grayscale. IM: `-sepia-tone N%`.
- **`Cyanotype`** — vintage blueprint colour remap. Reduces input
  to Rec.709 luminance and interpolates between a configurable
  shadow ("Prussian blue" `(15, 42, 111)`) and highlight (paper
  white `(217, 235, 248)`) endpoint. `strength` dials the blend
  with the original RGB; custom endpoints support cousin processes
  (sepia is similar but ships separately). Gray8 input upgrades to
  RGB; RGBA alpha pass-through. Factory aliases: `cyanotype`,
  `blueprint`.
- **`Grayscale`** — Rec. 601 desaturate; optional `Gray8` collapse.
  IM: `-colorspace Gray`.
- **`Colorize`** — linear blend toward a target `[R, G, B, A]` colour
  by a `0.0..=1.0` amount. IM: `-colorize N%`.
- **`Tint`** — luminance-weighted blend toward a target colour;
  bright pixels reach the target, dark pixels stay put. IM: `-tint N`.
- **`Vignette`** — Gaussian radial darkening centred at `(x*w, y*h)`
  with `radius` + `sigma`. IM: `-vignette RxS{+x{+y}}`.
- **`ColorMatrix`** — 3×3 colour matrix with optional 3-vector
  offset; equivalent of an affine 3×4 matrix on R / G / B. RGB / RGBA
  only (YUV → `Unsupported`). IM: `-color-matrix matrix`,
  `-recolor matrix`.
- **`BlueShift`** — moonlight / scotopic-vision tint: per-pixel
  `(min/factor, min/factor, max/factor)`. IM: `+blue-shift factor`.
- **`Quantize`** — uniform-grid colour quantizer: round each channel
  to one of `cbrt(N)` evenly-spaced palette entries. IM: `-colors N`
  (uniform-cube variant).
- **`Frame`** — decorative bordered frame with a 3-D bevel
  (highlight on top / left, shadow on bottom / right). RGB / RGBA
  only. IM: `-frame WxH+inner+outer-mat`.
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
- **`Vibrance`** — Lr-style saturation boost that spares already-
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
  Gray8 / RGB / RGBA. IM analogue: `-bordercolor C -border WxH`.
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
  Chebyshev tolerance (matching ImageMagick's `-fuzz`). Iterative
  span-fill keeps the work `O(W·H)`; out-of-bounds seeds error.
  Gray8 / RGB / RGBA.

### Sharpening + artistic

- **`BilateralBlur`** — edge-preserving Gaussian blur: each tap is
  weighted by both spatial proximity (`sigma_spatial`) and
  intensity proximity (`sigma_range`), so step edges survive a
  heavy smoothing pass. Border samples clamp to the nearest
  in-bounds coordinate; alpha pass-through on RGBA. Gray8 / RGB /
  RGBA. IM analogue: `-define blur:bilateral=1` on the Gaussian
  blur path.
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
  re-addition). IM: `-sharpen RxS`.
- **`Unsharp`** — explicit unsharp-mask with `threshold` gate: only
  high-contrast regions are sharpened. IM: `-unsharp RxS+A+T`.
- **`Clarity`** — large-radius unsharp-mask preset for mid-frequency
  tonal pop ("Lightroom Clarity" slider). Defaults: `radius = 30`,
  `sigma = 15.0`, `amount = 0.5`, `threshold = 10`. IM analogue:
  `-unsharp 30x15+0.5+10`.
- **`Emboss`** — 3×3 relief convolution with `+128` bias. IM: `-emboss R`.
- **`EmbossDirectional`** — 3×3 relief with a configurable light
  azimuth (degrees CCW from East) and `depth` multiplier. Kernel
  weights track `(dx, -dy) · (cos a, sin a)`, so flipping the
  azimuth by 180° inverts polarity. Gray8 / RGB / RGBA (alpha
  pass-through) + planar YUV (luma plane only, chroma untouched).
  Factory: `emboss-directional`.
- **`MotionBlur`** — 1-D Gaussian blur along `angle_degrees`. IM:
  `-motion-blur RxS+A`.
- **`Implode`** — radial pinch / explode with bilinear resampling.
  IM: `-implode N`.
- **`Swirl`** — radius-decaying rotational distortion. IM: `-swirl N`.
- **`Despeckle`** — median-window edge-preserving noise reduction
  (alpha pass-through). IM: `-despeckle`.
- **`Wave`** — sinusoidal vertical pixel displacement (amplitude /
  wavelength in pixels). IM: `-wave AxL`.
- **`Spread`** — random pixel-position perturbation inside a
  `[-radius, radius]²` neighbourhood, with a deterministic seed for
  reproducibility. IM: `-spread N`.
- **`Charcoal`** — non-photorealistic stylise: optional Gaussian
  pre-blur (`radius`, default `0`) ⇒ Sobel-on-luma ⇒ invert ⇒ `Gray8`
  sketch. Larger pre-blur radii thicken the strokes by suppressing
  fine texture before the edge pass. IM: `-charcoal R`.
- **`Paint`** — oil-paint stylise: per-pixel modal-bucket vote in a
  `(2*radius+1)²` window then mean-of-mode RGB. IM: `-paint radius`.
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
  mode (`+shade`). IM: `-shade az,el`.
- **`Convolve`** — user-supplied square `N×N` kernel (odd `N`); optional
  bias / divisor; alpha pass-through on RGBA. IM: `-convolve "..."`.
- **`Laplacian`** — 3×3 Laplacian (second-derivative) edge filter.
  Output is `|response|` clamped to `[0, 255]` as a `Gray8` frame.
  IM: `-laplacian`.
- **`Canny`** — full Canny edge pipeline: Gaussian pre-blur →
  Sobel gradient + direction → non-maximum suppression →
  hysteresis thresholding. Output is binary `Gray8` (0 / 255).
  IM: `-canny RxS+L%+H%`.
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
  IM: `-morphology Dilate|Erode|Open|Close`.
- **`MorphologyEdge`** — morphological edge / gradient operators
  built on the dilate / erode primitives:
  - `EdgeIn`  = `src - erode(src)`     — inner boundary.
  - `EdgeOut` = `dilate(src) - src`    — outer boundary.
  - `EdgeMagnitude` = `dilate - erode` — full per-pixel gradient.
  IM: `-morphology EdgeIn|EdgeOut|Edge`.
- **`Perspective`** — 4-corner perspective warp; solves the 3×3
  homography from src/dst quads with 8×8 Gauss-Jordan elimination,
  then inverse-maps each output pixel via `H⁻¹` with bilinear
  sampling. Optional `output_size: (w, h)` (JSON keys `output_width`
  + `output_height`) emits a custom canvas size — useful when the dst
  quad escapes the source rectangle. IM: `-distort Perspective "..."`.
- **`Affine`** — 2-D affine warp with bilinear resampling. Six
  coefficients in `(sx, ry, rx, sy, tx, ty)` order; supplied either
  as `matrix: [...]` or per-named keys on the JSON factory.
  Optional `output_size` for translation off-canvas. IM: `-distort
  Affine "sx,ry,rx,sy,tx,ty"`.
- **`Srt`** — Scale / Rotate / Translate composite warp. Collapses
  `T(tx,ty) · R(θ) · S(sx,sy) · T(-ox,-oy)` into a single 2×3 affine
  matrix and reuses the [`Affine`] machinery. Origin defaults to the
  image centre on the JSON factory; uniform `scale` overrides
  `sx`/`sy`. IM: `-distort SRT "ox,oy sx[,sy] angle tx,ty"`.
- **`Distort`** — full Brown-Conrady lens distortion model: radial
  (`k1` quadratic + `k2` quartic) plus optional tangential
  (`p1` + `p2`, default `0`) decentering coefficients. Pure-radial
  mode (`p1 = p2 = 0`) is bit-identical to the legacy r7 output. IM:
  `-distort barrel "k1 k2 ..."`.
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
  IM rough analogue: `-channel <ch> -separate`.

### Two-input compositing

- **`Clut`** — 1-D Colour Look-Up Table. `src` is the image; `dst`
  is the CLUT (read row-major). Per-channel index lookup; alpha
  pass-through. IM: `-clut`.
- **`HaldClut`** — Hald CLUT image-as-LUT colour grading. `dst` is a
  `(L²)×(L²)` Hald cube; trilinear sampling per pixel. RGB / RGBA
  only. IM: `-hald-clut`.
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
  - IM analogue: `-compose <op> -composite`.

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
