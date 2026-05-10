# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
