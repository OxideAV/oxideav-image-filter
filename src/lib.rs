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
//! - [`AutoGamma`](auto_gamma::AutoGamma) — auto-gamma: pick a per-channel
//!   gamma so the geometric mean lands at 0.5.
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
//!   `difference`). Implements [`TwoInputImageFilter`].
//! - [`Convolve`](convolve::Convolve) — user-supplied square `N×N`
//!   convolution kernel (odd `N`); optional bias / divisor; alpha
//!   pass-through on RGBA. IM: `-convolve "..."`.
//! - [`ChannelExtract`](channel_extract::ChannelExtract) — pull one
//!   channel (R/G/B/A or Y/U/V) out as a single-plane `Gray8` frame.
//!   IM: `-channel <ch> -separate`.
//! - [`Crop`](crop::Crop) — extract a rectangular subregion
//!   `(x, y, width, height)` (ImageMagick `-crop WxH+X+Y`).
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
//! - [`Extent`](extent::Extent) — set the output canvas to a fixed
//!   `(width, height)` with a placement offset, padding the gaps with a
//!   configurable background colour. IM: `-extent WxH+X+Y`.
//! - [`Flip`](flip::Flip) — mirror vertically (top row ↔ bottom row).
//! - [`Flop`](flop::Flop) — mirror horizontally (left col ↔ right col).
//! - [`Gamma`](gamma::Gamma) — power-law gamma curve applied per tone
//!   channel (LUT-based; YUV only touches luma).
//! - [`Grayscale`](grayscale::Grayscale) — desaturate RGB/RGBA with
//!   Rec. 601 luma weights; optional Gray8 collapse.
//! - [`Implode`](implode::Implode) — radial pinch / explode (ImageMagick
//!   `-implode N`); bilinear-resampled inverse mapping inside the
//!   inscribed circle.
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
//! - [`Perspective`](perspective::Perspective) — 4-corner perspective
//!   warp (homography solved from src/dst quads, inverse-mapped with
//!   bilinear sampling). IM: `-distort Perspective "..."`.
//! - [`Polar`](polar::Polar) — Cartesian ⇄ polar coordinate distortion
//!   (`-distort Polar` / `-distort DePolar`). Bends an image into a fan
//!   or unrolls a fan back into a rectangle; bilinear-sampled.
//! - [`Posterize`](posterize::Posterize) — reduce each channel to `N`
//!   intensity levels (ImageMagick `-posterize`).
//! - [`Resize`](resize::Resize) — rescale to arbitrary dimensions with
//!   [`Interpolation`](resize::Interpolation) = Nearest / Bilinear.
//! - [`Roll`](roll::Roll) — circular pixel shift `(dx, dy)`; rows /
//!   columns wrap around the borders. IM: `-roll +X+Y`.
//! - [`Rotate`](rotate::Rotate) — arbitrary-angle rotation with bilinear
//!   resampling; grows the canvas and fills gaps with a configurable
//!   background colour.
//! - [`Sepia`](sepia::Sepia) — warm-brown colour remap (ImageMagick
//!   `-sepia-tone`); threshold controls the mix with the original.
//! - [`Solarize`](solarize::Solarize) — invert samples above a
//!   threshold (ImageMagick `-solarize N%`).
//! - [`Spread`](spread::Spread) — random pixel-position perturbation
//!   inside a `[-radius, radius]²` neighbourhood with a deterministic
//!   PRNG (ImageMagick `-spread N`).
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
//! # Pixel formats
//!
//! Filters operate natively on the 8-bit single-plane and planar YUV
//! formats: `Gray8`, `Rgb24`, `Rgba`, `Yuv420P`, `Yuv422P`, `Yuv444P`.
//! Other formats return [`Error::unsupported`](oxideav_core::Error).

use oxideav_core::{Error, PixelFormat, VideoFrame};

pub mod auto_gamma;
pub mod blur;
pub mod brightness_contrast;
pub mod channel_extract;
pub mod charcoal;
pub mod colorize;
pub mod composite;
pub mod convolve;
pub mod crop;
pub mod despeckle;
pub mod distort;
pub mod edge;
pub mod emboss;
pub mod equalize;
pub mod extent;
pub mod flip;
pub mod flop;
pub mod gamma;
pub mod grayscale;
pub mod implode;
pub mod level;
pub mod modulate;
pub mod morphology;
pub mod motion_blur;
pub mod negate;
pub mod normalize;
pub mod perspective;
pub mod polar;
pub mod posterize;
pub mod registry;
pub mod resize;
pub mod roll;
pub mod rotate;
pub mod sepia;
pub mod sharpen;
pub mod shave;
pub mod sigmoidal_contrast;
pub mod solarize;
pub mod spread;
pub mod swirl;
pub mod threshold;
pub mod tilt_shift;
pub mod tint;
pub(crate) mod tonal_lut;
pub mod trim;
pub mod unsharp;
pub mod vignette;
pub mod wave;

pub use auto_gamma::AutoGamma;
pub use blur::Blur;
pub use brightness_contrast::BrightnessContrast;
pub use channel_extract::{Channel, ChannelExtract};
pub use charcoal::Charcoal;
pub use colorize::Colorize;
pub use composite::{Composite, CompositeOp};
pub use convolve::Convolve;
pub use crop::Crop;
pub use despeckle::Despeckle;
pub use distort::Distort;
pub use edge::Edge;
pub use emboss::Emboss;
pub use equalize::Equalize;
pub use extent::Extent;
pub use flip::Flip;
pub use flop::Flop;
pub use gamma::Gamma;
pub use grayscale::Grayscale;
pub use implode::Implode;
pub use level::Level;
pub use modulate::Modulate;
pub use morphology::{
    EdgeOp, Morphology, MorphologyChain, MorphologyEdge, MorphologyOp, StructuringElement,
};
pub use motion_blur::MotionBlur;
pub use negate::Negate;
pub use normalize::Normalize;
pub use perspective::Perspective;
pub use polar::{Polar, PolarDirection};
pub use posterize::Posterize;
pub use registry::{__oxideav_entry, register};
pub use resize::{Interpolation, Resize};
pub use roll::Roll;
pub use rotate::Rotate;
pub use sepia::Sepia;
pub use sharpen::Sharpen;
pub use shave::Shave;
pub use sigmoidal_contrast::SigmoidalContrast;
pub use solarize::Solarize;
pub use spread::Spread;
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
