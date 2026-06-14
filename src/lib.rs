//! Pure-Rust single-frame image filters for the oxideav framework.
//!
//! Each filter implements [`ImageFilter`] and transforms a [`VideoFrame`]
//! into a new [`VideoFrame`]. Filters are **stateless** — there is no
//! frame-to-frame history. That's what separates this crate from
//! `oxideav-audio-filter` (which has streaming echo/resample state) and
//! from `oxideav-pixfmt` (which handles pixel-format conversion and
//! palette quantisation).
//!
//! # Available filters
//!
//! - [`Affine`](affine::Affine) — 2-D affine transform with bilinear
//!   resampling. Six-coefficient `(sx, ry, rx, sy, tx, ty)` matrix
//!   matching the documented `-distort Affine` argument convention.
//! - [`AutoGamma`](auto_gamma::AutoGamma) — auto-gamma: pick a per-channel
//!   gamma so the geometric mean lands at 0.5.
//! - [`BlueShift`](blue_shift::BlueShift) — night-vision / moonlight
//!   tint: per-pixel `(min/factor, min/factor, max/factor)`. documented CLI: //!   `+blue-shift factor`.
//! - [`AutoLevel`](auto_level::AutoLevel) — per-channel auto-stretch:
//!   independently fill `[0, 255]` for each of `R`/`G`/`B`. documented CLI: //!   `-auto-level`.
//! - [`Canny`](canny::Canny) — Canny edge detector (Gaussian → Sobel
//!   → non-max suppression → hysteresis). Output is binary `Gray8`.
//!   documented CLI: `-canny RxS+L%+H%`.
//! - [`Clamp`](clamp::Clamp) — clamp every tone sample into
//!   `[low, high]`. documented CLI: `-clamp` (extended with explicit endpoints).
//! - [`Clut`](clut::Clut) — 1-D Colour Look-Up Table (two-input).
//!   `src` is the image; `dst` is the CLUT (read row-major). Per-channel
//!   index lookup; alpha pass-through. documented CLI: `-clut`.
//! - [`ColorMatrix`](color_matrix::ColorMatrix) — 3×3 colour matrix
//!   with optional offset. RGB / RGBA only. documented CLI: `-color-matrix`,
//!   `-recolor`.
//! - [`ContrastStretch`](contrast_stretch::ContrastStretch) — burn the
//!   darkest `black%` and brightest `white%` of pixels then linearly
//!   stretch the rest (per-channel for RGB). documented CLI: //!   `-contrast-stretch black%xwhite%`.
//! - [`Blur`](blur::Blur) — separable Gaussian blur with configurable
//!   radius + sigma, optional plane selector (luma only / chroma only /
//!   specific plane / all).
//! - [`BrightnessContrast`](brightness_contrast::BrightnessContrast) —
//!   linear brightness + contrast adjustment (LUT-based).
//! - [`Charcoal`](charcoal::Charcoal) — non-photorealistic stylise
//!   (Sobel-on-luma + invert ⇒ Gray8 sketch). documented CLI: `-charcoal R`.
//! - [`Colorize`](colorize::Colorize) — linear blend toward a target
//!   `[R, G, B, A]` colour by a `0.0..=1.0` amount.
//! - [`Composite`](composite::Composite) — two-input Porter–Duff and
//!   arithmetic blend (`over`, `in`, `out`, `atop`, `xor`, `plus`,
//!   `multiply`, `screen`, `overlay`, `darken`, `lighten`,
//!   `difference`, `hardlight`, `softlight`, `colordodge`,
//!   `colorburn`). Implements [`TwoInputImageFilter`].
//! - [`Convolve`](convolve::Convolve) — user-supplied square `N×N`
//!   convolution kernel (odd `N`); optional bias / divisor; alpha
//!   pass-through on RGBA. documented CLI: `-convolve "..."`.
//! - [`ChannelExtract`](channel_extract::ChannelExtract) — pull one
//!   channel (R/G/B/A or Y/U/V) out as a single-plane `Gray8` frame.
//!   documented CLI: `-channel <ch> -separate`.
//! - [`Crop`](crop::Crop) — extract a rectangular subregion
//!   `(x, y, width, height)` (documented `-crop WxH+X+Y` CLI).
//! - [`Cycle`](cycle::Cycle) — modular per-channel value rotation
//!   (`out = (src + amount) mod 256` per RGB / luma byte; alpha and
//!   chroma preserved). documented-CLI analogue: `-cycle N` (paletted-style sample
//!   shift on direct-colour data).
//! - [`Despeckle`](despeckle::Despeckle) — median-window
//!   edge-preserving noise reduction; alpha pass-through.
//! - [`Dither`](dither::Dither) — bit-depth-reduction dither with the
//!   classic Bayer ordered (`2×2` / `4×4` / `8×8`) threshold maps and
//!   the Floyd–Steinberg / JJN / Stucki / Sierra-3 / Sierra-2 /
//!   Sierra-Lite / Atkinson error-diffusion kernel family. Default
//!   `levels = 2` (1-bit B&W halftone); higher `levels` give multi-tone
//!   dithered posterisation. Per-channel for RGB / RGBA (alpha
//!   pass-through), luma-only on YUV.
//! - [`Distort`](distort::Distort) — radial-polynomial barrel /
//!   pincushion lens distortion (`k1` quadratic + `k2` quartic
//!   coefficients). documented CLI: `-distort barrel "k1 k2 ..."`.
//! - [`Edge`](edge::Edge) — Sobel edge magnitude; accepts colour input
//!   and returns a single-plane luma-ish intensity image.
//! - [`Emboss`](emboss::Emboss) — 3×3 relief convolution; luma-only
//!   on YUV, every channel on RGB.
//! - [`Equalize`](equalize::Equalize) — per-channel histogram
//!   equalisation via CDF mapping.
//! - [`Evaluate`](evaluate::Evaluate) — per-pixel arithmetic LUT
//!   (Add / Sub / Mul / Div / Pow / Max / Min / Set / And / Or / Xor /
//!   Threshold) with a single scalar operand. documented CLI: `-evaluate <op> N`.
//! - [`Extent`](extent::Extent) — set the output canvas to a fixed
//!   `(width, height)` with a placement offset, padding the gaps with a
//!   configurable background colour. documented CLI: `-extent WxH+X+Y`.
//! - [`Function`](function::Function) — per-pixel mathematical
//!   function map (`Polynomial`, `Sinusoid`, `ArcSin`, `ArcTan`)
//!   evaluated in normalised `[0, 1]` space. documented CLI: `-function`.
//! - [`Flip`](flip::Flip) — mirror vertically (top row ↔ bottom row).
//! - [`Flop`](flop::Flop) — mirror horizontally (left col ↔ right col).
//! - [`Frame`](frame::Frame) — decorative bordered frame with a 3-D
//!   bevel (highlight on top / left, shadow on bottom / right). RGB /
//!   RGBA only. documented CLI: `-frame WxH+inner+outer-mat`.
//! - [`Gamma`](gamma::Gamma) — power-law gamma curve applied per tone
//!   channel (LUT-based; YUV only touches luma).
//! - [`Grayscale`](grayscale::Grayscale) — desaturate RGB/RGBA with
//!   Rec. 601 luma weights; optional Gray8 collapse.
//! - [`HaldClut`](hald_clut::HaldClut) — Hald CLUT image-as-LUT
//!   colour grading (two-input). `dst` is a `(L²)×(L²)` Hald cube;
//!   trilinear sampling per pixel. RGB / RGBA only. documented CLI: `-hald-clut`.
//! - [`Implode`](implode::Implode) — radial pinch / explode (documented CLI
//!   `-implode N`); bilinear-resampled inverse mapping inside the
//!   inscribed circle.
//! - [`Laplacian`](laplacian::Laplacian) — 3×3 Laplacian
//!   second-derivative filter; output is `|response|` clamped to
//!   `[0, 255]`. documented CLI: `-laplacian`.
//! - [`LaplacianOfGaussian`](laplacian_of_gaussian::LaplacianOfGaussian)
//!   — Marr–Hildreth 1980 Laplacian-of-Gaussian detector: discrete
//!   sampling of `((x²+y²−2σ²)/σ⁴) · exp(−(x²+y²)/(2σ²))` on a
//!   `(2·radius+1)²` grid; zero-mean-corrected so flat input maps to
//!   zero. Two modes: `Magnitude` (|LoG|) or `ZeroCrossings` (binary
//!   Marr–Hildreth edge map with slope-threshold gate). Configurable
//!   `sigma` / `radius` / `output_gain` / `slope_threshold`.
//! - [`LinearStretch`](linear_stretch::LinearStretch) — like
//!   [`ContrastStretch`] but cut-offs are absolute pixel counts. documented CLI: //!   `-linear-stretch black-pixels{xwhite-pixels}`.
//! - [`Level`](level::Level) — remap `[black, white]` to `[0, 255]`
//!   with optional mid-tone gamma (documented `-level` CLI).
//! - [`Modulate`](modulate::Modulate) — adjust brightness, saturation,
//!   and hue via HSL round-trip (documented `-modulate` CLI).
//! - [`Morphology`](morphology::Morphology) — N-iteration greyscale
//!   dilate / erode with a 3×3 square or cross structuring element;
//!   plus [`MorphologyChain`](morphology::MorphologyChain) for the
//!   open / close compositions. documented CLI: `-morphology Dilate|Erode|Open|Close`.
//! - [`MorphologyEdge`](morphology::MorphologyEdge) — morphological
//!   edge / gradient operators (`EdgeIn`, `EdgeOut`, `EdgeMagnitude`)
//!   built from the same dilate / erode primitives. documented CLI: //!   `-morphology EdgeIn|EdgeOut|Edge`.
//! - [`MotionBlur`](motion_blur::MotionBlur) — directional 1-D Gaussian
//!   blur along `angle_degrees` (documented `-motion-blur RxS+A` CLI).
//! - [`Negate`](negate::Negate) — photo-negative of RGB/Gray channels;
//!   on YUV inverts only Y so chroma (hue/saturation) is preserved.
//! - [`Normalize`](normalize::Normalize) — auto-levels: stretch the
//!   observed luma range to fill `[0, 255]` (documented `-normalize` CLI).
//! - [`Paint`](paint::Paint) — oil-paint stylise: per-pixel modal-bucket
//!   vote in a `(2*radius+1)²` window then mean-of-mode RGB. documented CLI: //!   `-paint radius`.
//! - [`Perspective`](perspective::Perspective) — 4-corner perspective
//!   warp (homography solved from src/dst quads, inverse-mapped with
//!   bilinear sampling). documented CLI: `-distort Perspective "..."`.
//! - [`Polar`](polar::Polar) — Cartesian ⇄ polar coordinate distortion
//!   (`-distort Polar` / `-distort DePolar`). Bends an image into a fan
//!   or unrolls a fan back into a rectangle; bilinear-sampled.
//! - [`Posterize`](posterize::Posterize) — reduce each channel to `N`
//!   intensity levels (documented `-posterize` CLI).
//! - [`Quantize`](quantize::Quantize) — uniform-grid colour quantizer:
//!   round each channel to one of `cbrt(N)` evenly-spaced palette
//!   entries. documented CLI: `-colors N` (uniform-cube variant).
//! - [`Resize`](resize::Resize) — rescale to arbitrary dimensions with
//!   [`Interpolation`](resize::Interpolation) = Nearest / Bilinear.
//! - [`Roll`](roll::Roll) — circular pixel shift `(dx, dy)`; rows /
//!   columns wrap around the borders. documented CLI: `-roll +X+Y`.
//! - [`Rotate`](rotate::Rotate) — arbitrary-angle rotation with bilinear
//!   resampling; grows the canvas and fills gaps with a configurable
//!   background colour.
//! - [`Sepia`](sepia::Sepia) — warm-brown colour remap (documented CLI
//!   `-sepia-tone`); threshold controls the mix with the original.
//! - [`Shade`](shade::Shade) — directional Lambertian relief shading
//!   from an `(azimuth, elevation)` light vector. Optional colour
//!   pass-through mode (`+shade`). documented CLI: `-shade az,el`.
//! - [`Solarize`](solarize::Solarize) — invert samples above a
//!   threshold (documented `-solarize N%` CLI).
//! - [`Spread`](spread::Spread) — random pixel-position perturbation
//!   inside a `[-radius, radius]²` neighbourhood with a deterministic
//!   PRNG (documented `-spread N` CLI).
//! - [`Srt`](srt::Srt) — Scale / Rotate / Translate composite warp
//!   collapsing to a single 2×3 affine matrix; mirrors the documented CLI's
//!   `-distort SRT "ox,oy sx[,sy] angle tx,ty"` shorthand.
//! - [`Statistic`](statistic::Statistic) — rolling-window per-pixel
//!   statistic (`Median` / `Min` / `Max` / `Mean`) over a configurable
//!   `WxH` neighbourhood. documented CLI: `-statistic <op> WxH`.
//! - [`Shave`](shave::Shave) — strip a uniform `(x_border, y_border)`
//!   margin off every edge (centred crop). documented CLI: `-shave XxY`.
//! - [`Sharpen`](sharpen::Sharpen) — unsharp-mask sharpening with
//!   `radius`/`sigma`/`amount`; YUV touches only luma.
//! - [`SigmoidalContrast`](sigmoidal_contrast::SigmoidalContrast) —
//!   sigmoid-curve contrast adjustment (documented CLI
//!   `-sigmoidal-contrast CxM%`).
//! - [`Swirl`](swirl::Swirl) — radius-decaying rotational distortion
//!   (documented `-swirl N` CLI).
//! - [`Threshold`](threshold::Threshold) — binarise each sample to
//!   black/white against a cut-off (YUV sets chroma to neutral 128).
//! - [`TiltShift`](tilt_shift::TiltShift) — selective Gaussian blur
//!   masked by a horizontal in-focus band (miniature-photography
//!   depth-of-field). documented-CLI rough analogue: `-blur` masked by a vertical
//!   gradient.
//! - [`Tint`](tint::Tint) — luminance-weighted tint toward a target
//!   colour (documented `-tint` CLI); bright pixels reach the target,
//!   dark pixels stay put.
//! - [`Trim`](trim::Trim) — crop to the bounding box of pixels that
//!   differ from a reference background colour by more than `fuzz` per
//!   channel. documented CLI: `-fuzz N% -trim`.
//! - [`Unsharp`](unsharp::Unsharp) — threshold-gated unsharp-mask
//!   (documented `-unsharp RxS+A+T` CLI).
//! - [`Vignette`](vignette::Vignette) — Gaussian radial darkening
//!   centred at `(x, y)` with `radius` + `sigma` (documented CLI
//!   `-vignette RxS{+x{+y}}`).
//! - [`Wave`](wave::Wave) — sinusoidal vertical displacement with
//!   configurable amplitude (px) and wavelength (px). documented CLI: `-wave AxL`.
//!
//! r288 additions:
//!
//! - [`SignedDistanceField`](signed_distance_field::SignedDistanceField)
//!   — exact signed distance field built from two exact-Euclidean
//!   distance transforms (Felzenszwalb–Huttenlocher 2012). Per
//!   `docs/image/filter/distance-transform.md` §1 (the generalised DT
//!   `D_f(p) = min_q(‖p−q‖² + f(q))` whose intro names "SDF generation"
//!   as a use-case) and §2 (the separable lower-envelope transform): the
//!   signed field is `sdf(p) = d_out(p) − d_in(p)` where `d_out` is the
//!   EDT of the mask and `d_in` the EDT of the inverted mask, so
//!   foreground pixels carry negative distance and background pixels
//!   positive. Renders as `clamp(midpoint + scale·sdf, 0, 255)` so an
//!   on-boundary pixel lands at the neutral `midpoint` (default `128`);
//!   `threshold` / `invert` / `scale` / `midpoint` knobs. Reuses the
//!   `dt_1d` driver of
//!   [`EuclideanDistanceTransform`](euclidean_distance_transform::EuclideanDistanceTransform)
//!   so the whole filter is `O(d·N)` for an exact result. Single-plane
//!   `Gray8` in / out; useful for resolution-independent glyph / shape
//!   rendering, feathering, outline / glow, and iso-level morphology.
//!
//! r14 additions:
//!
//! - [`BarrelInverse`](barrel_inverse::BarrelInverse) — polynomial
//!   inverse radial distortion `r → r / (a·r³+b·r²+c·r+d)`. documented CLI: //!   `-distort BarrelInverse a,b,c,d`.
//! - [`Deskew`](deskew::Deskew) — auto-deskew via projection-variance
//!   scoring + corrective rotation. documented CLI: `-deskew threshold`.
//! - [`HoughLines`](hough_lines::HoughLines) — polar Hough-transform
//!   line detection; emits a `Gray8` line-trace canvas. documented CLI: //!   `-hough-lines WxH`.
//! - [`Sketch`](sketch::Sketch) — pencil-sketch stylise (directional
//!   blur + Sobel + invert). documented CLI: `-sketch radius,sigma,angle`.
//! - [`Stegano`](stegano::Stegano) — two-input LSB steganographic
//!   embed: stamps `src`'s MSBs into `dst`'s LSBs. documented CLI: `-stegano offset`.
//! - [`Stereo`](stereo::Stereo) — two-input red/cyan anaglyph stereo
//!   composition. documented CLI: `-stereo`.
//!
//! r15 additions:
//!
//! - [`AdaptiveThreshold`](adaptive_threshold::AdaptiveThreshold) —
//!   local-mean-based threshold (window radius + offset); emits
//!   `Gray8`. documented CLI: `-threshold local`.
//! - [`ChannelMixer`](channel_mixer::ChannelMixer) — 4×4 linear
//!   combination of source channels into destination channels plus a
//!   4-vector offset; super-set of `ColorMatrix`.
//! - [`ChromaticAberration`](chromatic_aberration::ChromaticAberration)
//!   — per-channel pixel offset on R and B (G untouched), simulating
//!   lateral lens aberration.
//! - [`HslRotate`](hsl_rotate::HslRotate) — rotate hue by N degrees in
//!   HSL space (RGB ⇒ HSL ⇒ rotate ⇒ RGB round-trip).
//! - [`Pixelate`](pixelate::Pixelate) — block-average mosaic; each
//!   `N×N` tile collapses to its mean colour.
//! - [`VignetteSoft`](vignette_soft::VignetteSoft) — raised-cosine
//!   soft vignette with separate inner / outer normalised radii;
//!   smoother seam than the existing Gaussian `Vignette`.
//!
//! r18 additions:
//!
//! - [`AutoTrim`](auto_trim::AutoTrim) — like [`Trim`] but picks the
//!   dominant colour via a 4-bit-per-channel histogram vote first, so
//!   noisy corners no longer poison the inferred background.
//! - [`Bloom`](bloom::Bloom) — Gaussian (box-approx) blur of bright
//!   regions, additively composited onto the source for a soft glow.
//!   Gray8 / RGB / RGBA.
//! - [`Difference`](difference::Difference) — two-input pixel-wise
//!   `|src - dst|`. Useful for change-detection / motion masks.
//! - [`DropShadow`](drop_shadow::DropShadow) — soft offset shadow
//!   composited *behind* opaque RGBA subject pixels.
//! - [`EdgeDetect`](edge_multi::EdgeDetect) +
//!   [`EdgeKernel`](edge_multi::EdgeKernel) — runtime-selectable
//!   gradient kernel (`Sobel` / `Prewitt` / `Scharr` / `Roberts`).
//! - [`HoughCircles`](hough_circles::HoughCircles) — circle detection
//!   via a `(r, cx, cy)` Hough accumulator; outputs the rendered
//!   circle trace on `Gray8`. Complement to existing [`HoughLines`].
//! - [`InnerShadow`](inner_shadow::InnerShadow) — soft offset shadow
//!   rendered *inside* the subject coverage instead of behind it.
//!
//! r17 additions:
//!
//! - [`BorderedFrame`](bordered_frame::BorderedFrame) — flat solid-colour
//!   border with independent per-side widths. Distinct from
//!   [`Frame`](frame::Frame), which paints a 3-D bevel.
//! - [`BwMix`](bw_mix::BwMix) — black-and-white conversion with
//!   per-channel weights (red / green / blue filter presets); optional
//!   `keep_format` to emit greyscale into the source's RGB / RGBA shape.
//! - [`Clarity`](clarity::Clarity) — mid-frequency local-contrast boost
//!   via a large-radius / moderate-amount unsharp mask.
//! - [`Exposure`](exposure::Exposure) — EV-stop adjustment in
//!   linear-light space (sRGB → linear → ×2^EV → sRGB).
//! - [`ShadowHighlight`](shadow_highlight::ShadowHighlight) — independent
//!   shadow lift + highlight recovery gated by a soft tonal mask.
//! - [`Temperature`](temperature::Temperature) — warmth slider that
//!   biases R / B channels (per-channel LUT, alpha pass-through).
//! - [`Vibrance`](vibrance::Vibrance) — photo-editor-style saturation boost that
//!   spares already-saturated pixels via `1 - s` weighting.
//!
//! r19 additions:
//!
//! - [`Heatmap`](heatmap::Heatmap) + [`HeatmapRamp`](heatmap::HeatmapRamp) —
//!   false-colour ramp applied to a luminance-reduced source. Built-in
//!   ramps: `Jet`, `Viridis`, `Plasma`, `Hot`, `Cool`, `Grayscale`.
//! - [`SplitTone`](split_tone::SplitTone) — independent shadow + highlight
//!   tints driven by a triangular tonal mask peaking at the extremes.
//! - [`FloodFill`](flood_fill::FloodFill) — seeded scanline flood-fill
//!   with per-channel Chebyshev tolerance; `Gray8` / `Rgb24` / `Rgba`.
//! - [`Watermark`](watermark::Watermark) — two-input over-place of a
//!   secondary image onto the source at `(offset_x, offset_y)` with a
//!   `opacity` multiplier; clips on negative or oversize overlays.
//! - [`PosterizeChannels`](posterize_channels::PosterizeChannels) —
//!   per-channel posterise (`r_levels`, `g_levels`, `b_levels`, optional
//!   `alpha_levels`).
//! - [`Toon`](toon::Toon) — cel-shaded cartoon look: per-channel
//!   posterise + Sobel edge overlay with configurable ink colour.
//!
//! r20 additions:
//!
//! - [`Crystallize`](crystallize::Crystallize) — Voronoi-cell averaging
//!   on a jittered grid; `cell_size` / `jitter` / deterministic `seed`.
//! - [`Halftone`](halftone::Halftone) — variable-size dot screening
//!   simulating offset-print AM screens; ink + paper colour configurable.
//! - [`GradientMap`](gradient_map::GradientMap) +
//!   [`GradientStop`](gradient_map::GradientStop) — recolour by per-pixel
//!   luminance ⇒ position in an arbitrary `(position, RGB)` gradient.
//!   Convenience constructors `duotone(...)` / `tritone(...)`.
//! - [`SelectiveColor`](selective_color::SelectiveColor) +
//!   [`HueBand`](selective_color::HueBand) +
//!   [`BandAdjust`](selective_color::BandAdjust) — per-hue-band HSL
//!   shifts (Reds / Yellows / Greens / Cyans / Blues / Magentas).
//! - [`CrossProcess`](cross_process::CrossProcess) — analogue film
//!   cross-processing emulation via three per-channel sigmoid S-curves.
//! - [`OtsuThreshold`](otsu_threshold::OtsuThreshold) — global
//!   automatic threshold maximising inter-class variance; emits binary
//!   `Gray8`. Complement to `AdaptiveThreshold` (local-mean) /
//!   `Threshold` (manual cut).
//!
//! r209 additions:
//!
//! - [`EuclideanDistanceTransform`](euclidean_distance_transform::EuclideanDistanceTransform)
//!   — exact-Euclidean distance transform (Felzenszwalb–Huttenlocher
//!   2012). Squared-distance is the lower envelope of upward parabolas
//!   `(p−q)² + f(q)`; pairwise intersection `s = ((f[q]+q²)−(f[v]+v²))/
//!   (2q−2v)`; separable one-pass-per-dimension gives `O(d·N)` total
//!   for an exact result, in contrast to the cheaper but approximate
//!   integer-chamfer [`DistanceTransform`](distance_transform::DistanceTransform)
//!   already in the crate. Single-plane `Gray8` input and output;
//!   `threshold`, `invert`, `scale` knobs to control mask polarity and
//!   rendered dynamic range. Useful for stroke generation, contour
//!   rings, SDF glyph atlases, feathering, and morphology effects that
//!   need a true Euclidean metric.
//!
//! r205 additions:
//!
//! - [`Niblack`](niblack::Niblack) — Niblack 1986 adaptive local-statistics
//!   threshold. For each pixel computes the mean `μ(x, y)` and standard
//!   deviation `σ(x, y)` of the `(2·radius + 1)²` neighbourhood, then
//!   binarises against `T(x, y) = μ + k · σ`. Default `k = -0.2` is
//!   the textbook page-segmentation bias (threshold sits below the
//!   local mean by a fraction of the local spread so faint ink against
//!   a brightish, locally-noisy background is reliably captured).
//!   Separable box-sum implementation runs in `O(W · H)` regardless of
//!   `radius`. Output `Gray8`. Joins the segmentation family at the
//!   "local mean + local σ threshold" position complementing
//!   [`AdaptiveThreshold`](adaptive_threshold::AdaptiveThreshold)
//!   (local mean only) and
//!   [`OtsuThreshold`](otsu_threshold::OtsuThreshold) (global
//!   automatic cut).
//!
//! r198 additions:
//!
//! - [`Gabor`](gabor::Gabor) + [`GaborMode`](gabor::GaborMode) — oriented
//!   Gaussian-modulated cosine filter (Gabor 1946; Daugman 1985). Discrete
//!   sampling of `exp(-(x'² + γ²·y'²) / (2σ²)) · cos(2π·x'/λ + ψ)` on a
//!   `(2·radius+1)²` grid with DC-removal so flat input maps to zero. Two
//!   output modes: [`GaborMode::Signed`] (linear remap of `[-R, +R]` to
//!   `[0, 255]` with neutral grey 128; preserves orientation + phase
//!   polarity) and [`GaborMode::Magnitude`] (linear remap of `[0, R]` to
//!   `[0, 255]`; classical oriented-energy response). Knobs: `wavelength`
//!   (carrier period in pixels, `≥ 2`), `orientation_degrees`,
//!   `phase_degrees`, `sigma`, `gamma` (envelope aspect ratio),
//!   `radius`, `output_gain`. Luma-collapses any supported input;
//!   output `Gray8`. Joins the edge / texture-energy family at the
//!   "oriented bandpass" position complementing the isotropic
//!   [`LaplacianOfGaussian`](laplacian_of_gaussian::LaplacianOfGaussian)
//!   and the 1st-derivative [`Edge`] / [`Prewitt`](prewitt::Prewitt) /
//!   [`Scharr`](scharr::Scharr) / [`Roberts`](roberts::Roberts).
//!
//! r181 additions:
//!
//! - [`LaplacianOfGaussian`](laplacian_of_gaussian::LaplacianOfGaussian)
//!   and [`LogMode`](laplacian_of_gaussian::LogMode) — Marr–Hildreth
//!   Laplacian-of-Gaussian edge / zero-crossing detector (Marr and
//!   Hildreth 1980). Samples the continuous `((x²+y²−2σ²)/σ⁴) ·
//!   exp(−(x²+y²)/(2σ²))` kernel on a `(2r+1)²` grid (auto-radius
//!   `ceil(3·σ)` by default), zero-means the coefficients so flat
//!   input maps to exactly zero, and runs a dense 2-D convolution.
//!   Two output modes: [`LogMode::Magnitude`] (default — `|LoG|`
//!   clamped to `[0,255]`, second-derivative response complementing
//!   the noise-amplifying [`Laplacian`](laplacian::Laplacian)) or
//!   [`LogMode::ZeroCrossings`] (binary Marr–Hildreth edge map with a
//!   slope-threshold gate to suppress quantisation-noise sign
//!   changes). Configurable `sigma`, `radius`, `output_gain`,
//!   `slope_threshold`. Luma-collapses any supported input; output
//!   `Gray8`. The pre-blur makes this the scale-selective complement
//!   to the bare 3×3 Laplacian and the 1st-derivative
//!   [`Edge`] / [`Prewitt`](prewitt::Prewitt) /
//!   [`Scharr`](scharr::Scharr) / [`Roberts`](roberts::Roberts).
//!
//! r174 additions:
//!
//! - [`FreiChen`](frei_chen::FreiChen) + [`FreiChenMode`](frei_chen::FreiChenMode)
//!   — Frei–Chen 3×3 orthonormal-basis edge / line detector
//!   (Frei & Chen 1977). Projects every 3×3 luma neighbourhood onto
//!   a basis of 9 mutually-orthogonal templates partitioned into edge
//!   (`S1..S4`), line (`S5..S8`), and mean (`S9`) sub-spaces, then
//!   reports the cosine of the angle to the requested sub-space —
//!   [`FreiChenMode::Edge`] (default; sensitive to step edges) or
//!   [`FreiChenMode::Line`] (sensitive to ripples / Laplacian-like
//!   ridges). Luma-collapses any supported input; output `Gray8`.
//!   Complements the magnitude-based 3×3 detectors
//!   ([`Edge`] / [`Prewitt`](prewitt::Prewitt) / [`Scharr`](scharr::Scharr))
//!   with a sub-space cosine, so a perfect edge / line gives `255`
//!   regardless of contrast.
//!
//! r105 additions:
//!
//! - [`Scharr`](scharr::Scharr) + [`ScharrMagnitude`](scharr::ScharrMagnitude)
//!   — Scharr 3×3 first-derivative edge operator (Scharr 2000). Row /
//!   column weights `±3 ±10 ±3`, magnitude divided by `4` so the
//!   output lands in the same band as the other 3×3 detectors;
//!   magnitude `sqrt(Gx²+Gy²)` (L2, default) or `|Gx|+|Gy|` (L1),
//!   clamped to `[0,255]`. Luma-collapses any supported input;
//!   output `Gray8`. Lowest orientation error of the three 3×3
//!   first-derivative kernels here ([`Prewitt`](prewitt::Prewitt) is
//!   flat ±1; [`Edge`] is Sobel ±1 ±2 ±1).
//!
//! r101 additions:
//!
//! - [`Prewitt`](prewitt::Prewitt) + [`PrewittMagnitude`](prewitt::PrewittMagnitude)
//!   — Prewitt 3×3 first-derivative edge operator (Prewitt 1970). Flat
//!   `±1`-weighted horizontal / vertical kernels; `Gx` is right-column
//!   minus left-column, `Gy` is bottom-row minus top-row. Magnitude is
//!   `sqrt(Gx²+Gy²)` (L2, default) or `|Gx|+|Gy|` (L1), clamped to
//!   `[0,255]`. Luma-collapses any supported input; output `Gray8`.
//!   Wider, less noise-sensitive support than [`Roberts`](roberts::Roberts);
//!   flatter weighting than [`Edge`] (Sobel).
//!
//! r24 additions:
//!
//! - [`Roberts`](roberts::Roberts) + [`RobertsMagnitude`](roberts::RobertsMagnitude)
//!   — Roberts cross 2×2 diagonal-difference edge operator (Roberts
//!   1963). `Gx = a − d`, `Gy = b − c` over the 2×2 window; magnitude
//!   is `sqrt(Gx²+Gy²)` (L2, default) or `|Gx|+|Gy|` (L1), clamped to
//!   `[0,255]`. Luma-collapses any supported input; output `Gray8`.
//!   The tiniest first-derivative detector — complement to
//!   [`Edge`] (Sobel) and [`Laplacian`](laplacian::Laplacian).
//!
//! r23 additions:
//!
//! - [`MaxRgb`](max_rgb::MaxRgb) + [`MaxRgbMode`](max_rgb::MaxRgbMode) —
//!   per-pixel `max(R, G, B)` or `min(R, G, B)` collapse to greyscale
//!   (HSV-V extractor / chroma proxy). Stateless branch-free `O(W·H)`.
//!   Default emits `Gray8`; opt into `keep_format` to preserve the
//!   input RGB / RGBA shape. RGB / RGBA / Gray8 (identity); YUV →
//!   `Unsupported`.
//!
//! r22 additions:
//!
//! - [`Reinhard`](reinhard::Reinhard) — 2002 Reinhard et al. global
//!   tone-mapping operator. Log-average luminance scaling + extended
//!   `L · (1 + L/white²) / (1 + L)` curve; chroma-preserving sRGB
//!   round-trip. Gray8 / RGB / RGBA.
//! - [`Hable`](hable::Hable) — Uncharted-2 "filmic" rational-function
//!   tone mapper (Hable, GDC 2010). Per-channel LUT; `exposure_bias`
//!   and `white` knobs.
//! - [`Drago`](drago::Drago) — 2003 Drago et al. adaptive logarithmic
//!   compression with `bias` parameter; chroma-preserving.
//! - [`ReinhardExtended`](reinhard_extended::ReinhardExtended) — r237:
//!   the unkeyed white-clamping form of the Reinhard 2002 operator
//!   (`Ld = L · (1 + L / L_white²) / (1 + L)` per
//!   `docs/image/filter/tone-mapping-operators.md` §5.1). Single
//!   parameter `l_white` (linear luminance mapped to pure white);
//!   `l_white <= 0` auto-picks the per-frame `L_max` so the brightest
//!   input pixel maps exactly to 1. Distinct from the keyed `Reinhard`
//!   in that it skips the log-average scaling pass — useful when the
//!   caller already has exposure-correct linear luminance.
//! - [`ReinhardLocal`](reinhard_local::ReinhardLocal) — r248: the
//!   local "dodging-and-burning" variant of the Reinhard 2002
//!   operator (`docs/image/filter/tone-mapping-operators.md` §3).
//!   Replaces the global form's `1 + L` denominator with a
//!   spatially-varying local average chosen per-pixel from a
//!   geometric Gaussian centre / surround pyramid. The largest scale
//!   at which the centre / surround difference `V(·, s)` is still
//!   below the §3.3 uniformity threshold `ε` (default `0.05`) is
//!   selected, so high-contrast edges truncate the search to the
//!   smallest scale and preserve local detail. Knobs: `key` (§2.1
//!   middle-grey `a`, default `0.18`), `phi` (§3.2 sharpening, default
//!   `8`), `epsilon` (§3.3 threshold, default `0.05`), `scales`
//!   (pyramid depth, default `8`), `initial_scale` (`s_0`, default
//!   `1.0`). Chroma-preserving sRGB round-trip on `Gray8`/`Rgb24`/
//!   `Rgba`; YUV returns `Unsupported`.
//! - [`Curves`](curves::Curves) + [`Curve`](curves::Curve) +
//!   [`CurveInterpolation`](curves::CurveInterpolation) — per-channel
//!   tonal curves with Linear / Catmull-Rom / Fritsch-Carlson
//!   monotone-cubic / natural cubic spline interpolation through user
//!   control points. r215 adds the `NaturalCubic` mode (de Boor 1978
//!   `C²` interpolant with the Thomas tridiagonal solver per
//!   `docs/image/filter/curve-interpolation.md` §4); r226 adds the
//!   `CentripetalCatmullRom` mode (Yuksel et al. 2011, α = 0.5 of the
//!   §3.3 alpha-Catmull-Rom family); r231 adds the
//!   `ChordalCatmullRom` mode (α = 1 of the same family — the third
//!   and final §3.3 member after uniform `CatmullRom` and centripetal).
//! - [`DistanceTransform`](distance_transform::DistanceTransform) +
//!   [`ChamferKind`](distance_transform::ChamferKind) — two-pass
//!   integer chamfer (Borgefors 1986) binary-mask distance transform
//!   with a runtime-selectable kernel: 3-4 (default), 5-7-11
//!   (knight-move-augmented, closer Euclidean), city-block
//!   (L1 / Manhattan; Rosenfeld & Pfaltz 1966), and chessboard
//!   (L∞ / Chebyshev). r220 generalises the original hard-coded 3-4
//!   path per `docs/image/filter/distance-transform.md` §3.2.
//!   Emits `Gray8`.
//! - [`Cyanotype`](cyanotype::Cyanotype) — vintage blueprint colour
//!   remap between configurable shadow / highlight endpoints
//!   (defaults to Prussian blue → paper white).
//!
//! r21 additions:
//!
//! - [`Kuwahara`](kuwahara::Kuwahara) — quadrant-variance edge-preserving
//!   smoothing (1976 Kuwahara paper). Picks the lowest-variance quadrant
//!   of the `(2*radius+1)²` window and writes its mean colour, so flat
//!   regions blur cleanly while step edges survive.
//! - [`AnisotropicBlur`](anisotropic::AnisotropicBlur) — Perona-Malik
//!   1990 anisotropic diffusion (Lorentzian edge-stopping). Iterative
//!   four-neighbour explicit Euler; smooth areas diffuse, edges freeze.
//! - [`ZoomBlur`](zoom_blur::ZoomBlur) — radial outward "warp drive"
//!   blur from a configurable centre; bilinear sampling.
//! - [`RadialBlur`](radial_blur::RadialBlur) — rotational ("spin")
//!   blur around a configurable centre; bilinear arc sampling.
//! - [`EmbossDirectional`](emboss_directional::EmbossDirectional) — 3×3
//!   relief with a configurable light azimuth (vs the fixed-kernel
//!   [`Emboss`]); RGB / RGBA / Gray8 + YUV (luma only).
//! - [`DisplacementMap`](displacement_map::DisplacementMap) — two-input
//!   warp; `dst` image's R/G channels carry per-pixel `(dx, dy)`
//!   vectors that bilinear-resample `src`.
//!
//! r16 additions:
//!
//! - [`BilateralBlur`](bilateral_blur::BilateralBlur) — edge-preserving
//!   joint spatial + range Gaussian blur.
//! - [`Canvas`](canvas::Canvas) — fixed-colour canvas generator
//!   (output-only fill).
//! - [`ColorBalance`](color_balance::ColorBalance) — three-way ASC
//!   CDL-style lift / gamma / gain per RGB channel.
//! - [`GradientConic`](gradient_conic::GradientConic) — angular
//!   gradient generator (centre + start angle + colour endpoints).
//! - [`GradientRadial`](gradient_radial::GradientRadial) — radial
//!   gradient generator (centre + radius + colour endpoints).
//! - [`GravityTranslate`](gravity_translate::GravityTranslate) —
//!   gravity-anchored canvas placement (9-point compass anchors); IM
//!   `-gravity` operator over `Extent`.
//! - [`HslShift`](hsl_shift::HslShift) — independent H / S / L
//!   shifts in HSL space.
//!
//! # Pixel formats
//!
//! Filters operate natively on the 8-bit single-plane and planar YUV
//! formats: `Gray8`, `Rgb24`, `Rgba`, `Yuv420P`, `Yuv422P`, `Yuv444P`.
//! Other formats return [`Error::unsupported`](oxideav_core::Error).

use oxideav_core::{Error, PixelFormat, VideoFrame};

pub mod adaptive_threshold;
pub mod affine;
pub mod anisotropic;
pub mod auto_gamma;
pub mod auto_level;
pub mod auto_trim;
pub mod barrel_inverse;
pub mod bilateral_blur;
pub mod bloom;
pub mod blue_shift;
pub mod blur;
pub mod bordered_frame;
pub mod brightness_contrast;
pub mod bw_mix;
pub mod canny;
pub mod canvas;
pub mod channel_extract;
pub mod channel_mixer;
pub mod charcoal;
pub mod chromatic_aberration;
pub mod clamp;
pub mod clarity;
pub mod clut;
pub mod color_balance;
pub mod color_matrix;
pub mod colorize;
pub mod comic;
pub mod composite;
pub mod contrast_stretch;
pub mod convolve;
pub mod crop;
pub mod cross_process;
pub mod crystallize;
pub mod curves;
pub mod cyanotype;
pub mod cycle;
pub mod deskew;
pub mod despeckle;
pub mod difference;
pub mod displacement_map;
pub mod distance_transform;
pub mod distort;
pub mod dither;
pub mod drago;
pub mod drop_shadow;
pub mod edge;
pub mod edge_multi;
pub mod emboss;
pub mod emboss_directional;
pub mod equalize;
pub mod euclidean_distance_transform;
pub mod evaluate;
pub mod exposure;
pub mod extent;
pub mod flip;
pub mod flood_fill;
pub mod flop;
pub mod frame;
pub mod frei_chen;
pub mod function;
pub mod gabor;
pub mod gamma;
pub mod gradient_conic;
pub mod gradient_map;
pub mod gradient_radial;
pub mod gravity_translate;
pub mod grayscale;
pub mod hable;
pub mod hald_clut;
pub mod halftone;
pub mod heatmap;
pub mod hough_circles;
pub mod hough_lines;
pub mod hsl_rotate;
pub mod hsl_shift;
pub mod implode;
pub mod inner_shadow;
pub mod kuwahara;
pub mod laplacian;
pub mod laplacian_of_gaussian;
pub mod level;
pub mod linear_stretch;
pub mod max_rgb;
pub mod modulate;
pub mod morphology;
pub mod motion_blur;
pub mod negate;
pub mod niblack;
pub mod normalize;
pub mod otsu_threshold;
pub mod paint;
pub mod perspective;
pub mod pixelate;
pub mod polar;
pub mod posterize;
pub mod posterize_channels;
pub mod prewitt;
pub mod quantize;
pub mod radial_blur;
pub mod registry;
pub mod reinhard;
pub mod reinhard_extended;
pub mod reinhard_local;
pub mod resize;
pub mod roberts;
pub mod roll;
pub mod rotate;
pub mod scharr;
pub mod selective_color;
pub mod sepia;
pub mod shade;
pub mod shadow_highlight;
pub mod sharpen;
pub mod shave;
pub mod sigmoidal_contrast;
pub mod signed_distance_field;
pub mod sketch;
pub mod solarize;
pub mod split_tone;
pub mod spread;
pub mod srt;
pub mod statistic;
pub mod stegano;
pub mod stereo;
pub mod swirl;
pub mod temperature;
pub mod threshold;
pub mod tilt_shift;
pub mod tint;
pub(crate) mod tonal_lut;
pub mod toon;
pub mod trim;
pub mod unsharp;
pub mod vibrance;
pub mod vignette;
pub mod vignette_soft;
pub mod watermark;
pub mod wave;
pub mod zoom_blur;

pub use adaptive_threshold::AdaptiveThreshold;
pub use affine::Affine;
pub use anisotropic::AnisotropicBlur;
pub use auto_gamma::AutoGamma;
pub use auto_level::AutoLevel;
pub use auto_trim::AutoTrim;
pub use barrel_inverse::BarrelInverse;
pub use bilateral_blur::BilateralBlur;
pub use bloom::Bloom;
pub use blue_shift::BlueShift;
pub use blur::Blur;
pub use bordered_frame::BorderedFrame;
pub use brightness_contrast::BrightnessContrast;
pub use bw_mix::BwMix;
pub use canny::Canny;
pub use canvas::Canvas;
pub use channel_extract::{Channel, ChannelExtract};
pub use channel_mixer::ChannelMixer;
pub use charcoal::Charcoal;
pub use chromatic_aberration::ChromaticAberration;
pub use clamp::Clamp;
pub use clarity::Clarity;
pub use clut::Clut;
pub use color_balance::ColorBalance;
pub use color_matrix::ColorMatrix;
pub use colorize::Colorize;
pub use comic::Comic;
pub use composite::{Composite, CompositeOp};
pub use contrast_stretch::ContrastStretch;
pub use convolve::Convolve;
pub use crop::Crop;
pub use cross_process::CrossProcess;
pub use crystallize::Crystallize;
pub use curves::{Curve, CurveInterpolation, Curves};
pub use cyanotype::Cyanotype;
pub use cycle::Cycle;
pub use deskew::Deskew;
pub use despeckle::Despeckle;
pub use difference::Difference;
pub use displacement_map::DisplacementMap;
pub use distance_transform::{ChamferKind, DistanceTransform};
pub use distort::Distort;
pub use dither::{BayerMatrix, DiffusionKernel, Dither, DitherMode, ScanOrder};
pub use drago::Drago;
pub use drop_shadow::DropShadow;
pub use edge::Edge;
pub use edge_multi::{EdgeDetect, EdgeKernel};
pub use emboss::Emboss;
pub use emboss_directional::EmbossDirectional;
pub use equalize::Equalize;
pub use euclidean_distance_transform::EuclideanDistanceTransform;
pub use evaluate::{Evaluate, EvaluateOp};
pub use exposure::Exposure;
pub use extent::Extent;
pub use flip::Flip;
pub use flood_fill::FloodFill;
pub use flop::Flop;
pub use frame::Frame;
pub use frei_chen::{FreiChen, FreiChenMode};
pub use function::{Function, FunctionOp};
pub use gabor::{Gabor, GaborMode};
pub use gamma::Gamma;
pub use gradient_conic::GradientConic;
pub use gradient_map::{GradientMap, GradientStop};
pub use gradient_radial::GradientRadial;
pub use gravity_translate::{Gravity, GravityTranslate};
pub use grayscale::Grayscale;
pub use hable::Hable;
pub use hald_clut::HaldClut;
pub use halftone::Halftone;
pub use heatmap::{Heatmap, HeatmapRamp};
pub use hough_circles::HoughCircles;
pub use hough_lines::HoughLines;
pub use hsl_rotate::HslRotate;
pub use hsl_shift::HslShift;
pub use implode::Implode;
pub use inner_shadow::InnerShadow;
pub use kuwahara::Kuwahara;
pub use laplacian::Laplacian;
pub use laplacian_of_gaussian::{LaplacianOfGaussian, LogMode};
pub use level::Level;
pub use linear_stretch::LinearStretch;
pub use max_rgb::{MaxRgb, MaxRgbMode};
pub use modulate::Modulate;
pub use morphology::{
    EdgeOp, Morphology, MorphologyChain, MorphologyEdge, MorphologyOp, StructuringElement,
};
pub use motion_blur::MotionBlur;
pub use negate::Negate;
pub use niblack::Niblack;
pub use normalize::Normalize;
pub use otsu_threshold::OtsuThreshold;
pub use paint::Paint;
pub use perspective::Perspective;
pub use pixelate::Pixelate;
pub use polar::{Polar, PolarDirection};
pub use posterize::Posterize;
pub use posterize_channels::PosterizeChannels;
pub use prewitt::{Prewitt, PrewittMagnitude};
pub use quantize::Quantize;
pub use radial_blur::RadialBlur;
pub use registry::{__oxideav_entry, register};
pub use reinhard::Reinhard;
pub use reinhard_extended::ReinhardExtended;
pub use reinhard_local::ReinhardLocal;
pub use resize::{Interpolation, Resize};
pub use roberts::{Roberts, RobertsMagnitude};
pub use roll::Roll;
pub use rotate::Rotate;
pub use scharr::{Scharr, ScharrMagnitude};
pub use selective_color::{BandAdjust, HueBand, SelectiveColor};
pub use sepia::Sepia;
pub use shade::Shade;
pub use shadow_highlight::ShadowHighlight;
pub use sharpen::Sharpen;
pub use shave::Shave;
pub use sigmoidal_contrast::SigmoidalContrast;
pub use signed_distance_field::SignedDistanceField;
pub use sketch::Sketch;
pub use solarize::Solarize;
pub use split_tone::SplitTone;
pub use spread::Spread;
pub use srt::Srt;
pub use statistic::{Statistic, StatisticOp};
pub use stegano::Stegano;
pub use stereo::Stereo;
pub use swirl::Swirl;
pub use temperature::Temperature;
pub use threshold::Threshold;
pub use tilt_shift::TiltShift;
pub use tint::Tint;
pub use toon::Toon;
pub use trim::Trim;
pub use unsharp::Unsharp;
pub use vibrance::Vibrance;
pub use vignette::Vignette;
pub use vignette_soft::VignetteSoft;
pub use watermark::Watermark;
pub use wave::Wave;
pub use zoom_blur::ZoomBlur;

/// Stream-level video parameters threaded into [`ImageFilter::apply`].
///
/// Used to live on every `VideoFrame` (`format` / `width` / `height`);
/// the slim moved them to the stream's
/// [`CodecParameters`](oxideav_core::CodecParameters). The
/// [`ImageFilterAdapter`](crate::registry) shim reads them once from
/// the input port spec at construction and re-supplies them per call
/// so concrete filters don't have to negotiate per-frame.
#[derive(Clone, Copy, Debug)]
pub struct VideoStreamParams {
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
}

/// A filter that transforms a single video frame without any external
/// state. Calling [`apply`](Self::apply) twice with the same input
/// produces the same output.
///
/// Stream-level shape (`format` / `width` / `height`) used to live on
/// each `VideoFrame`; with the slim it lives on the stream's
/// `CodecParameters` and is threaded in via [`VideoStreamParams`].
/// Filters that change the output shape (Edge → Gray8, Resize → new
/// width/height, etc.) document the new shape in their per-filter
/// docs; the adapter rebuilds the output port spec accordingly.
pub trait ImageFilter: Send {
    /// Apply the filter to `input`, producing a new frame. The filter
    /// must not retain any reference to `input`.
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error>;
}

/// A filter that combines two video frames — `src` (foreground) and
/// `dst` (background) — into a single output frame. Like
/// [`ImageFilter`] this contract is stateless: the same `(src, dst)`
/// pair always yields the same output.
///
/// Used by the [`Composite`] family of Porter–Duff and arithmetic
/// blend operators. The two-input adapter shim in [`crate::registry`]
/// plumbs the
/// pair of input ports through the [`StreamFilter`](oxideav_core::StreamFilter)
/// trait by buffering the most recent frame from each port and
/// emitting whenever both ports have a frame in hand.
///
/// `src` arrives on input port `0`, `dst` on input port `1`. Both
/// frames must share the same `format` / `width` / `height` —
/// implementations can assume that and panic / err out on mismatched
/// shapes.
pub trait TwoInputImageFilter: Send {
    /// Combine `src` and `dst` into a new frame. Neither input is
    /// retained beyond this call.
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error>;
}

/// Selects which planes of a video frame a filter operates on.
///
/// For planar YUV (e.g. `Yuv420P`) the plane indices are 0 = Y, 1 = Cb,
/// 2 = Cr. For `Rgba` the planes are a single 4-channel packed plane
/// (index 0) — use `All` to touch every channel, or let the per-filter
/// documentation describe how it handles packed layouts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Planes {
    /// Apply to every plane of the input frame.
    #[default]
    All,
    /// Apply only to the luma / Y plane (index 0). For packed RGB
    /// formats this is the whole image.
    Luma,
    /// Apply only to the chroma planes (indices 1 and 2 on planar
    /// YUV). For packed or single-plane formats this is a no-op.
    Chroma,
    /// Apply to a specific plane by index. Returns
    /// [`Error::invalid`] if the index is out of range for the input
    /// frame.
    Index(usize),
}

impl Planes {
    /// True if this selector targets `plane_index` for the given frame
    /// layout.
    pub fn matches(&self, plane_index: usize, plane_count: usize) -> bool {
        match *self {
            Planes::All => plane_index < plane_count,
            Planes::Luma => plane_index == 0,
            Planes::Chroma => plane_count >= 3 && (plane_index == 1 || plane_index == 2),
            Planes::Index(i) => i == plane_index,
        }
    }
}

/// Returns `true` when the filters in this crate can operate on `format`.
///
/// Used internally by [`Blur`] / [`Edge`] / [`Resize`] to reject
/// unsupported pixel formats with a descriptive error. Reads the
/// stream's pixel format (now off [`VideoStreamParams`]) since the
/// slim removed it from `VideoFrame`.
pub(crate) fn is_supported_format(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Gray8
            | PixelFormat::Rgb24
            | PixelFormat::Rgba
            | PixelFormat::Yuv420P
            | PixelFormat::Yuv422P
            | PixelFormat::Yuv444P
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_matches_semantics() {
        assert!(Planes::All.matches(0, 3));
        assert!(Planes::All.matches(2, 3));
        assert!(!Planes::All.matches(3, 3));

        assert!(Planes::Luma.matches(0, 3));
        assert!(!Planes::Luma.matches(1, 3));

        assert!(!Planes::Chroma.matches(0, 3));
        assert!(Planes::Chroma.matches(1, 3));
        assert!(Planes::Chroma.matches(2, 3));
        // Chroma is a no-op on packed / single-plane frames.
        assert!(!Planes::Chroma.matches(0, 1));

        assert!(Planes::Index(2).matches(2, 3));
        assert!(!Planes::Index(2).matches(1, 3));
    }
}
