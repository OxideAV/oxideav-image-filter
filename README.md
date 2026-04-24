# oxideav-image-filter

Pure-Rust single-frame image filters for
[oxideav](https://github.com/OxideAV/oxideav-workspace) — a 100% pure Rust
media framework. No C libraries, no FFI wrappers, no `*-sys` crates.

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

## Pixel formats

Supported in v0:

- `Gray8`, `Rgb24`, `Rgba`
- `Yuv420P`, `Yuv422P`, `Yuv444P`

Other formats return `Error::unsupported`. Planar higher-bit formats
(`Yuv420P10Le` etc.) will land in a later release as filters grow
per-format specialisations.

## License

MIT. See [LICENSE](LICENSE).
