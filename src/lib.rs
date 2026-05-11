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
//!   matching ImageMagick's `-distort Affine` argument convention.
//! - [`AutoGamma`](auto_gamma::AutoGamma) — auto-gamma: pick a per-channel
//!   gamma so the geometric mean lands at 0.5.
//! - [`BlueShift`](blue_shift::BlueShift) — night-vision / moonlight
//!   tint: per-pixel `(min/factor, min/factor, max/factor)`. IM:
//!   `+blue-shift factor`.
//! - [`AutoLevel`](auto_level::AutoLevel) — per-channel auto-stretch:
//!   independently fill `[0, 255]` for each of `R`/`G`/`B`. IM:
//!   `-auto-level`.
//! - [`Canny`](canny::Canny) — Canny edge detector (Gaussian → Sobel
//!   → non-max suppression → hysteresis). Output is binary `Gray8`.
//!   IM: `-canny RxS+L%+H%`.
//! - [`Clamp`](clamp::Clamp) — clamp every tone sample into
//!   `[low, high]`. IM: `-clamp` (extended with explicit endpoints).
//! - [`Clut`](clut::Clut) — 1-D Colour Look-Up Table (two-input).
//!   `src` is the image; `dst` is the CLUT (read row-major). Per-channel
//!   index lookup; alpha pass-through. IM: `-clut`.
//! - [`ColorMatrix`](color_matrix::ColorMatrix) — 3×3 colour matrix
//!   with optional offset. RGB / RGBA only. IM: `-color-matrix`,
//!   `-recolor`.
//! - [`ContrastStretch`](contrast_stretch::ContrastStretch) — burn the
//!   darkest `black%` and brightest `white%` of pixels then linearly
//!   stretch the rest (per-channel for RGB). IM:
//!   `-contrast-stretch black%xwhite%`.
//! - [`Blur`](blur::Blur) — separable Gaussian blur with configurable
//!   radius + sigma, optional plane selector (luma only / chroma only /
//!   specific plane / all).
//! - [`BrightnessContrast`](brightness_contrast::BrightnessContrast) —
//!   linear brightness + contrast adjustment (LUT-based).
//! - [`Charcoal`](charcoal::Charcoal) — non-photorealistic stylise
//!   (Sobel-on-luma + invert ⇒ Gray8 sketch). IM: `-charcoal R`.
//! - [`Colorize`](colorize::Colorize) — linear blend toward a target
//!   `[R, G, B, A]` colour by a `0.0..=1.0` amount.
//! - [`Composite`](composite::Composite) — two-input Porter–Duff and
//!   arithmetic blend (`over`, `in`, `out`, `atop`, `xor`, `plus`,
//!   `multiply`, `screen`, `overlay`, `darken`, `lighten`,
//!   `difference`, `hardlight`, `softlight`, `colordodge`,
//!   `colorburn`). Implements [`TwoInputImageFilter`].
//! - [`Convolve`](convolve::Convolve) — user-supplied square `N×N`
//!   convolution kernel (odd `N`); optional bias / divisor; alpha
//!   pass-through on RGBA. IM: `-convolve "..."`.
//! - [`ChannelExtract`](channel_extract::ChannelExtract) — pull one
//!   channel (R/G/B/A or Y/U/V) out as a single-plane `Gray8` frame.
//!   IM: `-channel <ch> -separate`.
//! - [`Crop`](crop::Crop) — extract a rectangular subregion
//!   `(x, y, width, height)` (ImageMagick `-crop WxH+X+Y`).
//! - [`Cycle`](cycle::Cycle) — modular per-channel value rotation
//!   (`out = (src + amount) mod 256` per RGB / luma byte; alpha and
//!   chroma preserved). IM analogue: `-cycle N` (paletted-style sample
//!   shift on direct-colour data).
//! - [`Despeckle`](despeckle::Despeckle) — median-window
//!   edge-preserving noise reduction; alpha pass-through.
//! - [`Distort`](distort::Distort) — radial-polynomial barrel /
//!   pincushion lens distortion (`k1` quadratic + `k2` quartic
//!   coefficients). IM: `-distort barrel "k1 k2 ..."`.
//! - [`Edge`](edge::Edge) — Sobel edge magnitude; accepts colour input
//!   and returns a single-plane luma-ish intensity image.
//! - [`Emboss`](emboss::Emboss) — 3×3 relief convolution; luma-only
//!   on YUV, every channel on RGB.
//! - [`Equalize`](equalize::Equalize) — per-channel histogram
//!   equalisation via CDF mapping.
//! - [`Evaluate`](evaluate::Evaluate) — per-pixel arithmetic LUT
//!   (Add / Sub / Mul / Div / Pow / Max / Min / Set / And / Or / Xor /
//!   Threshold) with a single scalar operand. IM: `-evaluate <op> N`.
//! - [`Extent`](extent::Extent) — set the output canvas to a fixed
//!   `(width, height)` with a placement offset, padding the gaps with a
//!   configurable background colour. IM: `-extent WxH+X+Y`.
//! - [`Function`](function::Function) — per-pixel mathematical
//!   function map (`Polynomial`, `Sinusoid`, `ArcSin`, `ArcTan`)
//!   evaluated in normalised `[0, 1]` space. IM: `-function`.
//! - [`Flip`](flip::Flip) — mirror vertically (top row ↔ bottom row).
//! - [`Flop`](flop::Flop) — mirror horizontally (left col ↔ right col).
//! - [`Frame`](frame::Frame) — decorative bordered frame with a 3-D
//!   bevel (highlight on top / left, shadow on bottom / right). RGB /
//!   RGBA only. IM: `-frame WxH+inner+outer-mat`.
//! - [`Gamma`](gamma::Gamma) — power-law gamma curve applied per tone
//!   channel (LUT-based; YUV only touches luma).
//! - [`Grayscale`](grayscale::Grayscale) — desaturate RGB/RGBA with
//!   Rec. 601 luma weights; optional Gray8 collapse.
//! - [`HaldClut`](hald_clut::HaldClut) — Hald CLUT image-as-LUT
//!   colour grading (two-input). `dst` is a `(L²)×(L²)` Hald cube;
//!   trilinear sampling per pixel. RGB / RGBA only. IM: `-hald-clut`.
//! - [`Implode`](implode::Implode) — radial pinch / explode (ImageMagick
//!   `-implode N`); bilinear-resampled inverse mapping inside the
//!   inscribed circle.
//! - [`Laplacian`](laplacian::Laplacian) — 3×3 Laplacian
//!   second-derivative filter; output is `|response|` clamped to
//!   `[0, 255]`. IM: `-laplacian`.
//! - [`LinearStretch`](linear_stretch::LinearStretch) — like
//!   [`ContrastStretch`] but cut-offs are absolute pixel counts. IM:
//!   `-linear-stretch black-pixels{xwhite-pixels}`.
//! - [`Level`](level::Level) — remap `[black, white]` to `[0, 255]`
//!   with optional mid-tone gamma (ImageMagick `-level`).
//! - [`Modulate`](modulate::Modulate) — adjust brightness, saturation,
//!   and hue via HSL round-trip (ImageMagick `-modulate`).
//! - [`Morphology`](morphology::Morphology) — N-iteration greyscale
//!   dilate / erode with a 3×3 square or cross structuring element;
//!   plus [`MorphologyChain`](morphology::MorphologyChain) for the
//!   open / close compositions. IM: `-morphology Dilate|Erode|Open|Close`.
//! - [`MorphologyEdge`](morphology::MorphologyEdge) — morphological
//!   edge / gradient operators (`EdgeIn`, `EdgeOut`, `EdgeMagnitude`)
//!   built from the same dilate / erode primitives. IM:
//!   `-morphology EdgeIn|EdgeOut|Edge`.
//! - [`MotionBlur`](motion_blur::MotionBlur) — directional 1-D Gaussian
//!   blur along `angle_degrees` (ImageMagick `-motion-blur RxS+A`).
//! - [`Negate`](negate::Negate) — photo-negative of RGB/Gray channels;
//!   on YUV inverts only Y so chroma (hue/saturation) is preserved.
//! - [`Normalize`](normalize::Normalize) — auto-levels: stretch the
//!   observed luma range to fill `[0, 255]` (ImageMagick `-normalize`).
//! - [`Paint`](paint::Paint) — oil-paint stylise: per-pixel modal-bucket
//!   vote in a `(2*radius+1)²` window then mean-of-mode RGB. IM:
//!   `-paint radius`.
//! - [`Perspective`](perspective::Perspective) — 4-corner perspective
//!   warp (homography solved from src/dst quads, inverse-mapped with
//!   bilinear sampling). IM: `-distort Perspective "..."`.
//! - [`Polar`](polar::Polar) — Cartesian ⇄ polar coordinate distortion
//!   (`-distort Polar` / `-distort DePolar`). Bends an image into a fan
//!   or unrolls a fan back into a rectangle; bilinear-sampled.
//! - [`Posterize`](posterize::Posterize) — reduce each channel to `N`
//!   intensity levels (ImageMagick `-posterize`).
//! - [`Quantize`](quantize::Quantize) — uniform-grid colour quantizer:
//!   round each channel to one of `cbrt(N)` evenly-spaced palette
//!   entries. IM: `-colors N` (uniform-cube variant).
//! - [`Resize`](resize::Resize) — rescale to arbitrary dimensions with
//!   [`Interpolation`](resize::Interpolation) = Nearest / Bilinear.
//! - [`Roll`](roll::Roll) — circular pixel shift `(dx, dy)`; rows /
//!   columns wrap around the borders. IM: `-roll +X+Y`.
//! - [`Rotate`](rotate::Rotate) — arbitrary-angle rotation with bilinear
//!   resampling; grows the canvas and fills gaps with a configurable
//!   background colour.
//! - [`Sepia`](sepia::Sepia) — warm-brown colour remap (ImageMagick
//!   `-sepia-tone`); threshold controls the mix with the original.
//! - [`Shade`](shade::Shade) — directional Lambertian relief shading
//!   from an `(azimuth, elevation)` light vector. Optional colour
//!   pass-through mode (`+shade`). IM: `-shade az,el`.
//! - [`Solarize`](solarize::Solarize) — invert samples above a
//!   threshold (ImageMagick `-solarize N%`).
//! - [`Spread`](spread::Spread) — random pixel-position perturbation
//!   inside a `[-radius, radius]²` neighbourhood with a deterministic
//!   PRNG (ImageMagick `-spread N`).
//! - [`Srt`](srt::Srt) — Scale / Rotate / Translate composite warp
//!   collapsing to a single 2×3 affine matrix; mirrors ImageMagick's
//!   `-distort SRT "ox,oy sx[,sy] angle tx,ty"` shorthand.
//! - [`Statistic`](statistic::Statistic) — rolling-window per-pixel
//!   statistic (`Median` / `Min` / `Max` / `Mean`) over a configurable
//!   `WxH` neighbourhood. IM: `-statistic <op> WxH`.
//! - [`Shave`](shave::Shave) — strip a uniform `(x_border, y_border)`
//!   margin off every edge (centred crop). IM: `-shave XxY`.
//! - [`Sharpen`](sharpen::Sharpen) — unsharp-mask sharpening with
//!   `radius`/`sigma`/`amount`; YUV touches only luma.
//! - [`SigmoidalContrast`](sigmoidal_contrast::SigmoidalContrast) —
//!   sigmoid-curve contrast adjustment (ImageMagick
//!   `-sigmoidal-contrast CxM%`).
//! - [`Swirl`](swirl::Swirl) — radius-decaying rotational distortion
//!   (ImageMagick `-swirl N`).
//! - [`Threshold`](threshold::Threshold) — binarise each sample to
//!   black/white against a cut-off (YUV sets chroma to neutral 128).
//! - [`TiltShift`](tilt_shift::TiltShift) — selective Gaussian blur
//!   masked by a horizontal in-focus band (miniature-photography
//!   depth-of-field). IM rough analogue: `-blur` masked by a vertical
//!   gradient.
//! - [`Tint`](tint::Tint) — luminance-weighted tint toward a target
//!   colour (ImageMagick `-tint`); bright pixels reach the target,
//!   dark pixels stay put.
//! - [`Trim`](trim::Trim) — crop to the bounding box of pixels that
//!   differ from a reference background colour by more than `fuzz` per
//!   channel. IM: `-fuzz N% -trim`.
//! - [`Unsharp`](unsharp::Unsharp) — threshold-gated unsharp-mask
//!   (ImageMagick `-unsharp RxS+A+T`).
//! - [`Vignette`](vignette::Vignette) — Gaussian radial darkening
//!   centred at `(x, y)` with `radius` + `sigma` (ImageMagick
//!   `-vignette RxS{+x{+y}}`).
//! - [`Wave`](wave::Wave) — sinusoidal vertical displacement with
//!   configurable amplitude (px) and wavelength (px). IM: `-wave AxL`.
//!
//! r14 additions:
//!
//! - [`BarrelInverse`](barrel_inverse::BarrelInverse) — polynomial
//!   inverse radial distortion `r → r / (a·r³+b·r²+c·r+d)`. IM:
//!   `-distort BarrelInverse a,b,c,d`.
//! - [`Deskew`](deskew::Deskew) — auto-deskew via projection-variance
//!   scoring + corrective rotation. IM: `-deskew threshold`.
//! - [`HoughLines`](hough_lines::HoughLines) — polar Hough-transform
//!   line detection; emits a `Gray8` line-trace canvas. IM:
//!   `-hough-lines WxH`.
//! - [`Sketch`](sketch::Sketch) — pencil-sketch stylise (directional
//!   blur + Sobel + invert). IM: `-sketch radius,sigma,angle`.
//! - [`Stegano`](stegano::Stegano) — two-input LSB steganographic
//!   embed: stamps `src`'s MSBs into `dst`'s LSBs. IM: `-stegano offset`.
//! - [`Stereo`](stereo::Stereo) — two-input red/cyan anaglyph stereo
//!   composition. IM: `-stereo`.
//!
//! # Pixel formats
//!
//! Filters operate natively on the 8-bit single-plane and planar YUV
//! formats: `Gray8`, `Rgb24`, `Rgba`, `Yuv420P`, `Yuv422P`, `Yuv444P`.
//! Other formats return [`Error::unsupported`](oxideav_core::Error).

use oxideav_core::{Error, PixelFormat, VideoFrame};

pub mod affine;
pub mod auto_gamma;
pub mod auto_level;
pub mod barrel_inverse;
pub mod blue_shift;
pub mod blur;
pub mod brightness_contrast;
pub mod canny;
pub mod channel_extract;
pub mod charcoal;
pub mod clamp;
pub mod clut;
pub mod color_matrix;
pub mod colorize;
pub mod composite;
pub mod contrast_stretch;
pub mod convolve;
pub mod crop;
pub mod cycle;
pub mod deskew;
pub mod despeckle;
pub mod distort;
pub mod edge;
pub mod emboss;
pub mod equalize;
pub mod evaluate;
pub mod extent;
pub mod flip;
pub mod flop;
pub mod frame;
pub mod function;
pub mod gamma;
pub mod grayscale;
pub mod hald_clut;
pub mod hough_lines;
pub mod implode;
pub mod laplacian;
pub mod level;
pub mod linear_stretch;
pub mod modulate;
pub mod morphology;
pub mod motion_blur;
pub mod negate;
pub mod normalize;
pub mod paint;
pub mod perspective;
pub mod polar;
pub mod posterize;
pub mod quantize;
pub mod registry;
pub mod resize;
pub mod roll;
pub mod rotate;
pub mod sepia;
pub mod shade;
pub mod sharpen;
pub mod shave;
pub mod sigmoidal_contrast;
pub mod sketch;
pub mod solarize;
pub mod spread;
pub mod srt;
pub mod statistic;
pub mod stegano;
pub mod stereo;
pub mod swirl;
pub mod threshold;
pub mod tilt_shift;
pub mod tint;
pub(crate) mod tonal_lut;
pub mod trim;
pub mod unsharp;
pub mod vignette;
pub mod wave;

pub use affine::Affine;
pub use auto_gamma::AutoGamma;
pub use auto_level::AutoLevel;
pub use barrel_inverse::BarrelInverse;
pub use blue_shift::BlueShift;
pub use blur::Blur;
pub use brightness_contrast::BrightnessContrast;
pub use canny::Canny;
pub use channel_extract::{Channel, ChannelExtract};
pub use charcoal::Charcoal;
pub use clamp::Clamp;
pub use clut::Clut;
pub use color_matrix::ColorMatrix;
pub use colorize::Colorize;
pub use composite::{Composite, CompositeOp};
pub use contrast_stretch::ContrastStretch;
pub use convolve::Convolve;
pub use crop::Crop;
pub use cycle::Cycle;
pub use deskew::Deskew;
pub use despeckle::Despeckle;
pub use distort::Distort;
pub use edge::Edge;
pub use emboss::Emboss;
pub use equalize::Equalize;
pub use evaluate::{Evaluate, EvaluateOp};
pub use extent::Extent;
pub use flip::Flip;
pub use flop::Flop;
pub use frame::Frame;
pub use function::{Function, FunctionOp};
pub use gamma::Gamma;
pub use grayscale::Grayscale;
pub use hald_clut::HaldClut;
pub use hough_lines::HoughLines;
pub use implode::Implode;
pub use laplacian::Laplacian;
pub use level::Level;
pub use linear_stretch::LinearStretch;
pub use modulate::Modulate;
pub use morphology::{
    EdgeOp, Morphology, MorphologyChain, MorphologyEdge, MorphologyOp, StructuringElement,
};
pub use motion_blur::MotionBlur;
pub use negate::Negate;
pub use normalize::Normalize;
pub use paint::Paint;
pub use perspective::Perspective;
pub use polar::{Polar, PolarDirection};
pub use posterize::Posterize;
pub use quantize::Quantize;
pub use registry::{__oxideav_entry, register};
pub use resize::{Interpolation, Resize};
pub use roll::Roll;
pub use rotate::Rotate;
pub use sepia::Sepia;
pub use shade::Shade;
pub use sharpen::Sharpen;
pub use shave::Shave;
pub use sigmoidal_contrast::SigmoidalContrast;
pub use sketch::Sketch;
pub use solarize::Solarize;
pub use spread::Spread;
pub use srt::Srt;
pub use statistic::{Statistic, StatisticOp};
pub use stegano::Stegano;
pub use stereo::Stereo;
pub use swirl::Swirl;
pub use threshold::Threshold;
pub use tilt_shift::TiltShift;
pub use tint::Tint;
pub use trim::Trim;
pub use unsharp::Unsharp;
pub use vignette::Vignette;
pub use wave::Wave;

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
