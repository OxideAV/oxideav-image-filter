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

### Colour

- **`Modulate`** — brightness / saturation / hue via HSL round-trip.
  RGB / RGBA only (YUV returns `Unsupported`). IM: `-modulate B,S,H`.
- **`Sepia`** — sepia-tone matrix with optional `threshold` mix back
  to grayscale. IM: `-sepia-tone N%`.
- **`Grayscale`** — Rec. 601 desaturate; optional `Gray8` collapse.
  IM: `-colorspace Gray`.

### Sharpening + artistic

- **`Sharpen`** — unsharp-mask (Gaussian-blurred subtract +
  re-addition). IM: `-sharpen RxS`.
- **`Unsharp`** — explicit unsharp-mask with `threshold` gate: only
  high-contrast regions are sharpened. IM: `-unsharp RxS+A+T`.
- **`Emboss`** — 3×3 relief convolution with `+128` bias. IM: `-emboss R`.
- **`MotionBlur`** — 1-D Gaussian blur along `angle_degrees`. IM:
  `-motion-blur RxS+A`.

All filters listed here share the `ImageFilter` trait — chain them
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
