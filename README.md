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
- **`Equalize`** — per-channel histogram equalisation via CDF
  mapping. Luma-only on YUV. IM: `-equalize`.
- **`AutoGamma`** — pick a per-channel gamma so the geometric mean
  lands at mid-grey 0.5 (`gamma = log(mean) / log(0.5)`). IM:
  `-auto-gamma`.
- **`SigmoidalContrast`** — sigmoid-curve contrast around a midpoint.
  IM: `-sigmoidal-contrast CxM%`.
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

### Colour

- **`Modulate`** — brightness / saturation / hue via HSL round-trip.
  RGB / RGBA only (YUV returns `Unsupported`). IM: `-modulate B,S,H`.
- **`Sepia`** — sepia-tone matrix with optional `threshold` mix back
  to grayscale. IM: `-sepia-tone N%`.
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

### Sharpening + artistic

- **`Sharpen`** — unsharp-mask (Gaussian-blurred subtract +
  re-addition). IM: `-sharpen RxS`.
- **`Unsharp`** — explicit unsharp-mask with `threshold` gate: only
  high-contrast regions are sharpened. IM: `-unsharp RxS+A+T`.
- **`Emboss`** — 3×3 relief convolution with `+128` bias. IM: `-emboss R`.
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
