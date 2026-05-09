# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
