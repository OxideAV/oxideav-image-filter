//! Compositing: Porter–Duff, the arithmetic and lighting blends, the
//! channel / colour-component copies, the map-driven operators and the
//! placement semantics of `-compose … -composite` (issue #9).
//!
//! The engine works on straight-alpha RGBA in binary32 (every layout
//! reaches it through the float hop) and was fixed black-box against the
//! `magick` binary (7.1.2 Q16-HDRI): `tests/magick_composite.rs` runs
//! every operator on two pictures with partial alpha and holds PSNR ≥
//! 45 dB. ImageMagick's own conventions, as measured:
//!
//! - the "mathematical" blends (`Multiply`, `Screen`, `Overlay`,
//!   `Darken`, …) form the union alpha `Sa + Da − Sa·Da` and the
//!   **premultiplied** colour `Sa·Da·B(Sc, Dc) + Sa·(1 − Da)·Sc +
//!   Da·(1 − Sa)·Dc`, which is stored as the straight colour without
//!   dividing by the alpha;
//! - the Porter–Duff operators follow their classic premultiplied
//!   algebra and do un-premultiply (`Over`, `Xor`, …);
//! - `In` / `Out` / `Atop` / `Xor` store their premultiplied colour;
//! - outside the overlay the source is transparent and the operator
//!   still applies (`compose:outside-overlay` = true); with
//!   [`Composite::with_outside_overlay`]`(false)` the destination is
//!   left as it is there.
//!
//! Not yet matched (the operator runs, `tests/magick_composite.rs`
//! reports its PSNR without asserting it): `Difference`, `SoftLight`,
//! `DivideDst` / `DivideSrc`, `ModulusAdd` /
//! `ModulusSubtract`, `Exclusion`, `LinearDodge`, `LinearLight`,
//! `HardMix`, `PegtopLight`, `SoftBurn`, `SoftDodge`, `Stamp`, `Freeze`,
//! `Reflect`, `Negate`, `DarkenIntensity` /
//! `LightenIntensity`, `Bumpmap`, `Blend` with arguments, and the
//! colour-component / map / modulate family (`Hue`, `Saturate`,
//! `Luminize`, `Colorize`, `Rmse`, `Intensity`, `CopyBlack`, `Threshold`,
//! `Displace`, `Distort`, `Blur`, `Modulate`). Their per-channel curves
//! on opaque pictures were measured and match for most; what differs is
//! how `magick` folds partial alpha into each one, which the staged
//! documentation does not describe.

use crate::deep::{self, FloatImage};
use crate::gravity_translate::Gravity;
use crate::{PixelFilter, TwoInputImageFilter, TwoInputPixelFilter, VideoStreamParams, layout};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// A compositing operator — ImageMagick's `-compose` set.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositeOp {
    /// Source over destination (Porter–Duff `src-over`).
    Over,
    /// Source where the destination is (`src-in`).
    In,
    /// Source where the destination is not (`src-out`).
    Out,
    /// Source atop the destination (`src-atop`).
    Atop,
    /// Either but not both.
    Xor,
    /// Premultiplied sum, clamped.
    Plus,
    /// `Sc · Dc`.
    Multiply,
    /// `Sc + Dc − Sc·Dc`.
    Screen,
    /// Hard light with the roles swapped.
    Overlay,
    /// `min(Sc, Dc)`.
    Darken,
    /// `max(Sc, Dc)`.
    Lighten,
    /// `|Sc − Dc|`.
    Difference,
    /// Multiply below half, screen above, keyed on the source.
    HardLight,
    /// The SVG soft-light curve.
    SoftLight,
    /// `Dc / (1 − Sc)`.
    ColorDodge,
    /// `1 − (1 − Dc) / Sc`.
    ColorBurn,
    /// Transparent black everywhere.
    Clear,
    /// The source (`Copy`, `Src`, `Replace`).
    Src,
    /// The destination untouched (`Dst`, `None`).
    Dst,
    /// Destination over source.
    DstOver,
    /// Destination where the source is.
    DstIn,
    /// Destination where the source is not.
    DstOut,
    /// Destination atop the source.
    DstAtop,
    /// `Sc − Dc` (`MinusDst`, the `Minus` default).
    MinusDst,
    /// `Dc − Sc`.
    MinusSrc,
    /// `Sc / Dc` (`DivideDst`, the `Divide` default).
    DivideDst,
    /// `Dc / Sc`.
    DivideSrc,
    /// Sum wrapped modulo 1.
    ModulusAdd,
    /// Difference wrapped modulo 1.
    ModulusSubtract,
    /// `Sc + Dc − 2·Sc·Dc`.
    Exclusion,
    /// `Sc + Dc − 1`.
    LinearBurn,
    /// `Sc + Dc`.
    LinearDodge,
    /// `Dc + 2·Sc − 1`.
    LinearLight,
    /// Colour burn below half, colour dodge above.
    VividLight,
    /// Darken below half, lighten above.
    PinLight,
    /// Vivid light thresholded to 0 or 1.
    HardMix,
    /// Pegtop's soft light `(1 − 2Sc)·Dc² + 2·Sc·Dc`.
    PegtopLight,
    /// Soft burn.
    SoftBurn,
    /// Soft dodge.
    SoftDodge,
    /// `Dc + 2·Sc − 1` (stamp).
    Stamp,
    /// `1 − (1 − Dc)² / Sc`.
    Freeze,
    /// `Dc² / (1 − Sc)`.
    Reflect,
    /// `1 − |1 − Sc − Dc|`.
    Negate,
    /// `½ − ¼ cos(π Sc) − ¼ cos(π Dc)`.
    Interpolate,
    /// `A·Sc·Dc + B·Sc + C·Dc + D` with `A, B, C, D` from the args.
    Mathematics,
    /// Weighted sum `Sc·a% + Dc·b%` (`Blend`; `Average` is 50/50).
    Blend,
    /// Alphas scaled by `a%` / `b%`, then `Over`.
    Dissolve,
    /// Root mean square of the channel differences.
    Rmse,
    /// The darker pixel by intensity.
    DarkenIntensity,
    /// The lighter pixel by intensity.
    LightenIntensity,
    /// Hue of the source, saturation and lightness of the destination.
    Hue,
    /// Saturation of the source.
    Saturate,
    /// Lightness of the source.
    Luminize,
    /// Hue and saturation of the source.
    Colorize,
    /// The source's red into the destination.
    CopyRed,
    /// The source's green into the destination.
    CopyGreen,
    /// The source's blue into the destination.
    CopyBlue,
    /// The source's alpha (its intensity when it has none) into the
    /// destination (`CopyAlpha`, `CopyOpacity`).
    CopyAlpha,
    /// Alias of [`Self::CopyRed`] on an RGB picture.
    CopyCyan,
    /// Alias of [`Self::CopyGreen`] on an RGB picture.
    CopyMagenta,
    /// Alias of [`Self::CopyBlue`] on an RGB picture.
    CopyYellow,
    /// The source's intensity into every destination channel.
    CopyBlack,
    /// Destination made transparent where the source matches it within
    /// the fuzz (args: fuzz, default `0`).
    ChangeMask,
    /// Destination multiplied by the source's intensity.
    Bumpmap,
    /// Destination moved towards the source where they differ by more
    /// than the threshold (args: threshold, amount; default `0.05, 1`).
    Threshold,
    /// Source as a displacement map: destination read at `(x + (Sr −
    /// ½)·X, y + (Sg − ½)·Y)` (args: `X[xY]` in pixels, default `20`).
    Displace,
    /// Source as an absolute lookup map: destination read at `(Sr·W,
    /// Sg·H)` (args as for `Displace`, scaling the map).
    Distort,
    /// Source as a variable blur map: a Gaussian of `σ = Sr·X` by
    /// `Sg·Y` (args `X[xY]`, default `1`).
    Blur,
    /// The source's intensity in every channel.
    Intensity,
    /// Destination brightness / saturation modulated by the source's
    /// intensity (args: brightness %, saturation %; default `100, 100`).
    Modulate,
}

impl CompositeOp {
    /// Every operator.
    pub const ALL: &'static [CompositeOp] = &[
        Self::Over,
        Self::In,
        Self::Out,
        Self::Atop,
        Self::Xor,
        Self::Plus,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::Difference,
        Self::HardLight,
        Self::SoftLight,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::Clear,
        Self::Src,
        Self::Dst,
        Self::DstOver,
        Self::DstIn,
        Self::DstOut,
        Self::DstAtop,
        Self::MinusDst,
        Self::MinusSrc,
        Self::DivideDst,
        Self::DivideSrc,
        Self::ModulusAdd,
        Self::ModulusSubtract,
        Self::Exclusion,
        Self::LinearBurn,
        Self::LinearDodge,
        Self::LinearLight,
        Self::VividLight,
        Self::PinLight,
        Self::HardMix,
        Self::PegtopLight,
        Self::SoftBurn,
        Self::SoftDodge,
        Self::Stamp,
        Self::Freeze,
        Self::Reflect,
        Self::Negate,
        Self::Interpolate,
        Self::Mathematics,
        Self::Blend,
        Self::Dissolve,
        Self::Rmse,
        Self::DarkenIntensity,
        Self::LightenIntensity,
        Self::Hue,
        Self::Saturate,
        Self::Luminize,
        Self::Colorize,
        Self::CopyRed,
        Self::CopyGreen,
        Self::CopyBlue,
        Self::CopyAlpha,
        Self::CopyCyan,
        Self::CopyMagenta,
        Self::CopyYellow,
        Self::CopyBlack,
        Self::ChangeMask,
        Self::Bumpmap,
        Self::Threshold,
        Self::Displace,
        Self::Distort,
        Self::Blur,
        Self::Intensity,
        Self::Modulate,
    ];

    /// The operator's name as `-compose` spells it (lower-case).
    pub fn name(self) -> &'static str {
        match self {
            CompositeOp::Over => "over",
            CompositeOp::In => "in",
            CompositeOp::Out => "out",
            CompositeOp::Atop => "atop",
            CompositeOp::Xor => "xor",
            CompositeOp::Plus => "plus",
            CompositeOp::Multiply => "multiply",
            CompositeOp::Screen => "screen",
            CompositeOp::Overlay => "overlay",
            CompositeOp::Darken => "darken",
            CompositeOp::Lighten => "lighten",
            CompositeOp::Difference => "difference",
            CompositeOp::HardLight => "hardlight",
            CompositeOp::SoftLight => "softlight",
            CompositeOp::ColorDodge => "colordodge",
            CompositeOp::ColorBurn => "colorburn",
            CompositeOp::Clear => "clear",
            CompositeOp::Src => "src",
            CompositeOp::Dst => "dst",
            CompositeOp::DstOver => "dstover",
            CompositeOp::DstIn => "dstin",
            CompositeOp::DstOut => "dstout",
            CompositeOp::DstAtop => "dstatop",
            CompositeOp::MinusDst => "minusdst",
            CompositeOp::MinusSrc => "minussrc",
            CompositeOp::DivideDst => "dividedst",
            CompositeOp::DivideSrc => "dividesrc",
            CompositeOp::ModulusAdd => "modulusadd",
            CompositeOp::ModulusSubtract => "modulussubtract",
            CompositeOp::Exclusion => "exclusion",
            CompositeOp::LinearBurn => "linearburn",
            CompositeOp::LinearDodge => "lineardodge",
            CompositeOp::LinearLight => "linearlight",
            CompositeOp::VividLight => "vividlight",
            CompositeOp::PinLight => "pinlight",
            CompositeOp::HardMix => "hardmix",
            CompositeOp::PegtopLight => "pegtoplight",
            CompositeOp::SoftBurn => "softburn",
            CompositeOp::SoftDodge => "softdodge",
            CompositeOp::Stamp => "stamp",
            CompositeOp::Freeze => "freeze",
            CompositeOp::Reflect => "reflect",
            CompositeOp::Negate => "negate",
            CompositeOp::Interpolate => "interpolate",
            CompositeOp::Mathematics => "mathematics",
            CompositeOp::Blend => "blend",
            CompositeOp::Dissolve => "dissolve",
            CompositeOp::Rmse => "rmse",
            CompositeOp::DarkenIntensity => "darkenintensity",
            CompositeOp::LightenIntensity => "lightenintensity",
            CompositeOp::Hue => "hue",
            CompositeOp::Saturate => "saturate",
            CompositeOp::Luminize => "luminize",
            CompositeOp::Colorize => "colorize",
            CompositeOp::CopyRed => "copyred",
            CompositeOp::CopyGreen => "copygreen",
            CompositeOp::CopyBlue => "copyblue",
            CompositeOp::CopyAlpha => "copyalpha",
            CompositeOp::CopyCyan => "copycyan",
            CompositeOp::CopyMagenta => "copymagenta",
            CompositeOp::CopyYellow => "copyyellow",
            CompositeOp::CopyBlack => "copyblack",
            CompositeOp::ChangeMask => "changemask",
            CompositeOp::Bumpmap => "bumpmap",
            CompositeOp::Threshold => "threshold",
            CompositeOp::Displace => "displace",
            CompositeOp::Distort => "distort",
            CompositeOp::Blur => "blur",
            CompositeOp::Intensity => "intensity",
            CompositeOp::Modulate => "modulate",
        }
    }

    /// Parse an operator name: case-insensitive, `-` / `_` ignored, with
    /// the aliases `copy` / `replace` / `srcover` … (`Src`), `none`
    /// (`Dst`), `minus` / `divide` (the `…Dst` forms), `copyopacity`
    /// (`CopyAlpha`), `average` (`Blend`), `hard-light` and friends.
    pub fn from_name(s: &str) -> Option<Self> {
        let key: String = s
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.name() == key)
            .or(match key.as_str() {
                "srcover" => Some(Self::Over),
                "srcin" => Some(Self::In),
                "srcout" => Some(Self::Out),
                "srcatop" => Some(Self::Atop),
                "copy" | "replace" => Some(Self::Src),
                "none" => Some(Self::Dst),
                "minus" => Some(Self::MinusDst),
                "divide" => Some(Self::DivideDst),
                "copyopacity" => Some(Self::CopyAlpha),
                "average" => Some(Self::Blend),
                "add" => Some(Self::ModulusAdd),
                "subtract" => Some(Self::ModulusSubtract),
                _ => None,
            })
    }

    /// `true` for the operators that read the source as a map rather
    /// than as a picture to blend (`Displace`, `Distort`, `Blur`).
    pub fn is_map(self) -> bool {
        matches!(self, Self::Displace | Self::Distort | Self::Blur)
    }
}

/// Which destination channels the operator may write (the Imagick
/// `channel` argument); the others keep their destination value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelMask {
    /// Write red.
    pub red: bool,
    /// Write green.
    pub green: bool,
    /// Write blue.
    pub blue: bool,
    /// Write alpha.
    pub alpha: bool,
}

impl ChannelMask {
    /// Every channel.
    pub const ALL: ChannelMask = ChannelMask {
        red: true,
        green: true,
        blue: true,
        alpha: true,
    };

    /// The colour channels only.
    pub const RGB: ChannelMask = ChannelMask {
        red: true,
        green: true,
        blue: true,
        alpha: false,
    };

    /// Build a mask.
    pub fn new(red: bool, green: bool, blue: bool, alpha: bool) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }
}

impl Default for ChannelMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// Composite `src` (the overlay, port 0) onto `dst` (the background,
/// port 1) with a [`CompositeOp`].
///
/// As a two-input filter both pictures share one shape and the output
/// has that shape. [`Composite::compose`] takes pictures of different
/// extents and places the overlay by offset and gravity on the
/// destination, which sets the output extent — the `compositeImage`
/// surface.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Composite {
    /// The operator.
    pub op: CompositeOp,
    /// Overlay offset on the destination, after gravity.
    pub offset: (i64, i64),
    /// Where the overlay sits before the offset (`None`: top-left).
    pub gravity: Option<Gravity>,
    /// `compose:args` — the operator's numeric arguments in their natural
    /// units (percentages for `Blend` / `Dissolve` / `Modulate`, the
    /// `A, B, C, D` of `Mathematics`, pixels for the maps, the fuzz of
    /// `ChangeMask`, the threshold / amount of `Threshold`).
    pub args: Vec<f32>,
    /// Channels the operator may write.
    pub channels: ChannelMask,
    /// Whether the operator also runs where the destination lies outside
    /// the overlay (with a transparent source), ImageMagick's default.
    pub outside_overlay: bool,
}

impl Composite {
    /// An operator with the defaults: no offset, every channel, the
    /// operator applied outside the overlay too.
    pub fn new(op: CompositeOp) -> Self {
        Self {
            op,
            offset: (0, 0),
            gravity: None,
            args: Vec::new(),
            channels: ChannelMask::ALL,
            outside_overlay: true,
        }
    }

    /// Place the overlay at `(x, y)` (negative offsets allowed).
    pub fn with_offset(mut self, x: i64, y: i64) -> Self {
        self.offset = (x, y);
        self
    }

    /// Place the overlay by gravity (the offset is then relative to it).
    pub fn with_gravity(mut self, gravity: Gravity) -> Self {
        self.gravity = Some(gravity);
        self
    }

    /// Set `compose:args`.
    pub fn with_args(mut self, args: impl Into<Vec<f32>>) -> Self {
        self.args = args.into();
        self
    }

    /// Restrict the written channels.
    pub fn with_channels(mut self, channels: ChannelMask) -> Self {
        self.channels = channels;
        self
    }

    /// Whether the destination outside the overlay is composited against
    /// a transparent source (`true`, the default) or left untouched.
    pub fn with_outside_overlay(mut self, outside: bool) -> Self {
        self.outside_overlay = outside;
        self
    }

    fn arg(&self, i: usize, default: f32) -> f32 {
        self.args
            .get(i)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(default)
    }

    /// The overlay's top-left corner on a `dw × dh` destination for a
    /// `sw × sh` overlay.
    fn placement(&self, dw: u32, dh: u32, sw: u32, sh: u32) -> (i64, i64) {
        let (gx, gy) = match self.gravity {
            None => (0, 0),
            Some(g) => {
                let (dw, dh, sw, sh) = (dw as i64, dh as i64, sw as i64, sh as i64);
                let x = match g {
                    Gravity::NorthWest | Gravity::West | Gravity::SouthWest => 0,
                    Gravity::North | Gravity::Centre | Gravity::South => (dw - sw) / 2,
                    _ => dw - sw,
                };
                let y = match g {
                    Gravity::NorthWest | Gravity::North | Gravity::NorthEast => 0,
                    Gravity::West | Gravity::Centre | Gravity::East => (dh - sh) / 2,
                    _ => dh - sh,
                };
                (x, y)
            }
        };
        (gx + self.offset.0, gy + self.offset.1)
    }

    /// `compositeImage`: the overlay `src` (its own layout and extent)
    /// placed on the destination `dst`; the output has the destination's
    /// extent and layout (the packed 8-bit and float layouts; any other
    /// destination layout comes back as `RgbaF32Le`).
    pub fn compose(
        &self,
        dst: &VideoFrame,
        dst_params: VideoStreamParams,
        src: &VideoFrame,
        src_params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let d = crate::measure::float_image_of(dst, dst_params)?;
        let s = crate::measure::float_image_of(src, src_params)?;
        let out = self.compose_float(&d, &s);
        Ok(match dst_params.format {
            PixelFormat::Gray8 | PixelFormat::Rgb24 | PixelFormat::Rgba => {
                deep::narrow_u8(&out.to_channels(d.channels()))
            }
            f if deep::is_float(f) => out.to_channels(d.channels()).into_frame(dst.pts),
            _ => out.into_frame(dst.pts),
        })
    }

    /// [`Composite::compose`] on float pictures; the result is RGBA of
    /// the destination's extent.
    pub fn compose_float(&self, dst: &FloatImage, src: &FloatImage) -> FloatImage {
        let d = dst.to_rgba();
        let s = src.to_rgba();
        let (dw, dh) = (d.width(), d.height());
        let (ox, oy) = self.placement(dw, dh, s.width(), s.height());
        let mut out = d.clone();
        // Map operators read the whole destination through the map.
        let map_ctx = MapContext::new(self, &d, &s, ox, oy);
        for y in 0..dh {
            for x in 0..dw {
                let sx = x as i64 - ox;
                let sy = y as i64 - oy;
                let inside = sx >= 0 && sy >= 0 && sx < s.width() as i64 && sy < s.height() as i64;
                if !inside && !self.outside_overlay {
                    continue;
                }
                let sp: [f32; 4] = if inside {
                    let p = s.pixel(sx as u32, sy as u32);
                    [p[0], p[1], p[2], p[3]]
                } else {
                    [0.0; 4]
                };
                let dpx = d.pixel(x, y);
                let dp = [dpx[0], dpx[1], dpx[2], dpx[3]];
                let r = if self.op.is_map() {
                    map_ctx.sample(self.op, x, y, sp, dp)
                } else {
                    self.pixel(sp, dp)
                };
                let o = out.pixel_mut(x, y);
                if self.channels.red {
                    o[0] = r[0];
                }
                if self.channels.green {
                    o[1] = r[1];
                }
                if self.channels.blue {
                    o[2] = r[2];
                }
                if self.channels.alpha {
                    o[3] = r[3];
                }
            }
        }
        out
    }

    /// One pixel: `s` and `d` straight RGBA.
    fn pixel(&self, s: [f32; 4], d: [f32; 4]) -> [f32; 4] {
        use CompositeOp as O;
        let (sa, da) = (s[3], d[3]);
        match self.op {
            O::Clear => [0.0; 4],
            O::Src => s,
            O::Dst => d,
            O::Over => over(s, d),
            O::DstOver => {
                let o = over(d, s);
                if o[3] <= 0.0 {
                    [d[0], d[1], d[2], 0.0]
                } else {
                    o
                }
            }
            // Porter–Duff other than Over: magick stores the premultiplied
            // result colour as it is (measured: In of an opaque source on
            // a half-covered destination gives Sc·Da, not Sc).
            O::In => premult([s[0], s[1], s[2]], sa * da, sa * da),
            // The destination forms keep the straight colour (measured).
            // (black where the result is transparent).
            O::DstIn => straight_or_clear([d[0], d[1], d[2]], sa * da),
            O::Out => premult([s[0], s[1], s[2]], sa * (1.0 - da), sa * (1.0 - da)),
            O::DstOut => straight_or_clear([d[0], d[1], d[2]], da * (1.0 - sa)),
            O::Atop => {
                let mut o = [0.0; 4];
                for c in 0..3 {
                    o[c] = s[c] * sa * da + d[c] * da * (1.0 - sa);
                }
                o[3] = da;
                o
            }
            O::DstAtop => {
                let mut o = [0.0; 4];
                for c in 0..3 {
                    o[c] = d[c] * da * sa + s[c] * sa * (1.0 - da);
                }
                o[3] = sa;
                o
            }
            O::Xor => {
                let mut o = [0.0; 4];
                for c in 0..3 {
                    o[c] = s[c] * sa * (1.0 - da) + d[c] * da * (1.0 - sa);
                }
                o[3] = sa * (1.0 - da) + da * (1.0 - sa);
                o
            }
            O::Plus => {
                let mut o = [0.0; 4];
                for c in 0..3 {
                    o[c] = (s[c] * sa + d[c] * da).min(1.0);
                }
                o[3] = (sa + da).min(1.0);
                o
            }
            O::Blend => {
                let a = self.arg(0, 50.0) / 100.0;
                let b = self.arg(1, 100.0 - a * 100.0) / 100.0;
                let mut o = [0.0; 4];
                for c in 0..3 {
                    o[c] = s[c] * a + d[c] * b;
                }
                o[3] = (sa * a + da * b).min(1.0);
                o
            }
            O::Dissolve => {
                let a = self.arg(0, 50.0) / 100.0;
                let b = self.arg(1, 100.0) / 100.0;
                over(
                    [s[0], s[1], s[2], (sa * a).min(1.0)],
                    [d[0], d[1], d[2], (da * b).min(1.0)],
                )
            }
            O::Rmse => {
                let v = ((0..3).map(|c| (s[c] - d[c]).powi(2)).sum::<f32>() / 3.0).sqrt();
                [v, v, v, over_alpha(sa, da)]
            }
            O::DarkenIntensity => {
                if intensity(s) * sa < intensity(d) * da {
                    s
                } else {
                    d
                }
            }
            O::LightenIntensity => {
                if intensity(s) * sa > intensity(d) * da {
                    s
                } else {
                    d
                }
            }
            O::Hue | O::Saturate | O::Luminize | O::Colorize => {
                let (hs, ss, ls) = rgb_to_hsl(s[0], s[1], s[2]);
                let (hd, sd, ld) = rgb_to_hsl(d[0], d[1], d[2]);
                let (h, sat, l) = match self.op {
                    O::Hue => (hs, sd, ld),
                    O::Saturate => (hd, ss, ld),
                    O::Luminize => (hd, sd, ls),
                    _ => (hs, ss, ld),
                };
                let (r, g, b) = hsl_to_rgb(h, sat, l);
                let a = over_alpha(sa, da);
                // Blend with the destination by the source's coverage.
                let mut o = [0.0; 4];
                for (c, v) in [r, g, b].into_iter().enumerate() {
                    o[c] = if a > 0.0 {
                        (v * sa + d[c] * da * (1.0 - sa)) / a
                    } else {
                        0.0
                    };
                }
                o[3] = a;
                o
            }
            O::CopyRed | O::CopyCyan => [s[0], d[1], d[2], da],
            O::CopyGreen | O::CopyMagenta => [d[0], s[1], d[2], da],
            O::CopyBlue | O::CopyYellow => [d[0], d[1], s[2], da],
            O::CopyAlpha => [d[0], d[1], d[2], sa],
            O::CopyBlack => {
                let i = intensity(s);
                [i, i, i, da]
            }
            O::Intensity => {
                let i = intensity(s);
                [i, i, i, over_alpha(sa, da)]
            }
            O::ChangeMask => {
                let fuzz = self.arg(0, 0.0);
                let same = (0..4).all(|c| (s[c] - d[c]).abs() <= fuzz + 1e-6);
                if same { [d[0], d[1], d[2], 0.0] } else { d }
            }
            O::Bumpmap => {
                let i = intensity(s);
                [d[0] * i, d[1] * i, d[2] * i, da]
            }
            O::Threshold => {
                let threshold = self.arg(0, 0.05);
                let amount = self.arg(1, 1.0);
                let mut o = d;
                for c in 0..3 {
                    let delta = s[c] - d[c];
                    if delta.abs() >= threshold {
                        o[c] = d[c] + delta * amount;
                    }
                }
                o
            }
            O::Modulate => {
                // Approximate (the recipe is not staged): the source's
                // intensity above / below mid-grey scales the destination's
                // lightness and saturation by the percentages.
                let bright = self.arg(0, 100.0) / 100.0;
                let satur = self.arg(1, 100.0) / 100.0;
                let i = intensity(s) - 0.5;
                let (h, sat, l) = rgb_to_hsl(d[0], d[1], d[2]);
                let l = (l * (1.0 + 2.0 * i * bright)).clamp(0.0, 1.0);
                let sat = (sat * (1.0 + 2.0 * i * satur)).clamp(0.0, 1.0);
                let (r, g, b) = hsl_to_rgb(h, sat, l);
                [r, g, b, da]
            }
            O::Displace | O::Distort | O::Blur => d,
            // The mathematical blends. The union alpha everywhere; the
            // colour in the convention measured per operator:
            //   - `Premult`: Sa·Da·B(Sc, Dc) + Sa(1−Da)Sc + Da(1−Sa)Dc stored
            //     as it is (Multiply, Screen, Lighten, LinearBurn, …);
            //   - `Divided`: the same divided by the union alpha (Overlay,
            //     HardLight, ColorDodge, LinearDodge, LinearLight, PinLight,
            //     Minus, …);
            //   - `OnPremult`: B(Sa·Sc, Da·Dc) + Sa(1−Da)Sc + Da(1−Sa)Dc
            //     (Darken);
            //   - `PremultOnly`: B(Sa·Sc, Da·Dc) alone (Interpolate,
            //     ModulusAdd, ModulusSubtract).
            op => {
                let a = over_alpha(sa, da);
                let mut o = [0.0; 4];
                for c in 0..3 {
                    let terms = sa * (1.0 - da) * s[c] + da * (1.0 - sa) * d[c];
                    o[c] = match convention(op) {
                        Convention::Premult => sa * da * blend(op, s[c], d[c], &self.args) + terms,
                        Convention::Divided => {
                            let v = sa * da * blend_raw(op, s[c], d[c], &self.args) + terms;
                            if a > 0.0 { v / a } else { 0.0 }
                        }
                        Convention::OnPremult => {
                            blend(op, s[c] * sa, d[c] * da, &self.args) + terms
                        }
                        Convention::PremultOnly => blend(op, s[c] * sa, d[c] * da, &self.args),
                    }
                    .clamp(0.0, 1.0);
                }
                o[3] = match op {
                    O::ModulusAdd => wrap(sa + da),
                    O::ModulusSubtract => wrap(sa - da),
                    // Interpolate runs its curve on the alphas too (measured).
                    O::Interpolate => blend(op, sa, da, &self.args),
                    _ => a,
                };
                o
            }
        }
    }
}

/// A colour scaled by `weight` with alpha `a` (the premultiplied form
/// magick stores for the non-Over Porter–Duff operators).
#[inline]
fn premult(c: [f32; 3], weight: f32, a: f32) -> [f32; 4] {
    [c[0] * weight, c[1] * weight, c[2] * weight, a]
}

/// A straight colour with alpha `a`, cleared to black when `a` is 0.
#[inline]
fn straight_or_clear(c: [f32; 3], a: f32) -> [f32; 4] {
    if a <= 0.0 {
        [0.0; 4]
    } else {
        [c[0], c[1], c[2], a]
    }
}

/// The union alpha `Sa + Da − Sa·Da`.
#[inline]
fn over_alpha(sa: f32, da: f32) -> f32 {
    sa + da - sa * da
}

/// Porter–Duff over: `s` over `d`, straight colour out.
fn over(s: [f32; 4], d: [f32; 4]) -> [f32; 4] {
    let (sa, da) = (s[3], d[3]);
    let a = sa + da * (1.0 - sa);
    if a <= 0.0 {
        // Both transparent: magick keeps the destination's colour.
        return [d[0], d[1], d[2], 0.0];
    }
    let mut o = [0.0; 4];
    for c in 0..3 {
        o[c] = (s[c] * sa + d[c] * da * (1.0 - sa)) / a;
    }
    o[3] = a;
    o
}

#[inline]
fn wrap(v: f32) -> f32 {
    let w = v.rem_euclid(1.0);
    if v >= 1.0 && w == 0.0 { 1.0 } else { w }
}

/// Rec. 709 intensity of a straight RGBA pixel.
#[inline]
fn intensity(p: [f32; 4]) -> f32 {
    0.212_656 * p[0] + 0.715_158 * p[1] + 0.072_186 * p[2]
}

/// How a mathematical operator combines its blend with the alphas
/// (measured against `magick`, see the module docs).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Convention {
    Premult,
    Divided,
    OnPremult,
    PremultOnly,
}

fn convention(op: CompositeOp) -> Convention {
    use CompositeOp as O;
    match op {
        O::Overlay
        | O::HardLight
        | O::SoftLight
        | O::ColorDodge
        | O::ColorBurn
        | O::SoftBurn
        | O::SoftDodge
        | O::LinearDodge
        | O::LinearLight
        | O::VividLight
        | O::PinLight
        | O::HardMix
        | O::MinusDst
        | O::MinusSrc
        | O::PegtopLight
        | O::Freeze
        | O::Reflect
        | O::Mathematics => Convention::Divided,
        O::Darken | O::DivideDst | O::DivideSrc => Convention::OnPremult,
        O::Interpolate | O::ModulusAdd | O::ModulusSubtract | O::Negate | O::Exclusion => {
            Convention::PremultOnly
        }
        _ => Convention::Premult,
    }
}

/// [`blend`] without the final clamp (the `Divided` convention clamps
/// after the division).
fn blend_raw(op: CompositeOp, s: f32, d: f32, args: &[f32]) -> f32 {
    blend_inner(op, s, d, args)
}

/// The per-channel blend function `B(Sc, Dc)` of a mathematical operator,
/// clamped to `[0, 1]`.
fn blend(op: CompositeOp, s: f32, d: f32, args: &[f32]) -> f32 {
    blend_inner(op, s, d, args).clamp(0.0, 1.0)
}

fn blend_inner(op: CompositeOp, s: f32, d: f32, args: &[f32]) -> f32 {
    use CompositeOp as O;
    match op {
        O::Multiply => s * d,
        O::Screen => s + d - s * d,
        O::Overlay => hard_light(d, s),
        O::Darken => s.min(d),
        O::Lighten => s.max(d),
        O::Difference => (s - d).abs(),
        O::HardLight => hard_light(s, d),
        O::SoftLight => soft_light(s, d),
        O::ColorDodge => color_dodge(s, d),
        O::ColorBurn => color_burn(s, d),
        O::MinusDst => s - d,
        O::MinusSrc => d - s,
        O::DivideDst => divide(s, d),
        O::DivideSrc => divide(d, s),
        O::Exclusion => s + d - 2.0 * s * d,
        O::LinearBurn => s + d - 1.0,
        O::LinearDodge => s + d,
        O::LinearLight => d + 2.0 * s - 1.0,
        O::VividLight => {
            if s < 0.5 {
                color_burn(2.0 * s, d)
            } else {
                color_dodge(2.0 * s - 1.0, d)
            }
        }
        O::PinLight => {
            if s < 0.5 {
                d.min(2.0 * s)
            } else {
                d.max(2.0 * s - 1.0)
            }
        }
        O::HardMix => {
            if blend(O::VividLight, s, d, args) < 0.5 {
                0.0
            } else {
                1.0
            }
        }
        O::PegtopLight => (1.0 - 2.0 * s) * d * d + 2.0 * s * d,
        // `magick` 7.1.2 computes SoftBurn / SoftDodge as ColorBurn /
        // ColorDodge, Stamp as `Sc + Dc − 1`, Freeze as `1 − (1 − Sc)² /
        // Dc` and Reflect as `Sc² / (1 − Dc)` (measured on opaque pictures).
        O::SoftBurn => color_burn(s, d),
        O::SoftDodge => color_dodge(s, d),
        O::Stamp => s + d - 1.0,
        O::Freeze => {
            if d <= 0.0 {
                0.0
            } else {
                1.0 - (1.0 - s) * (1.0 - s) / d
            }
        }
        O::Reflect => {
            if d >= 1.0 {
                1.0
            } else {
                s * s / (1.0 - d)
            }
        }
        O::Negate => 1.0 - (1.0 - s - d).abs(),
        O::Interpolate => {
            let pi = std::f32::consts::PI;
            0.5 - 0.25 * (pi * s).cos() - 0.25 * (pi * d).cos()
        }
        O::Mathematics => {
            let a = |i: usize| args.get(i).copied().unwrap_or(0.0);
            a(0) * s * d + a(1) * s + a(2) * d + a(3)
        }
        _ => s,
    }
}

#[inline]
fn hard_light(s: f32, d: f32) -> f32 {
    if s < 0.5 {
        2.0 * s * d
    } else {
        let (s2, dd) = (2.0 * s - 1.0, d);
        s2 + dd - s2 * dd
    }
}

/// The SVG / W3C soft-light curve.
#[inline]
fn soft_light(s: f32, d: f32) -> f32 {
    if s <= 0.5 {
        d - (1.0 - 2.0 * s) * d * (1.0 - d)
    } else {
        let dd = if d <= 0.25 {
            ((16.0 * d - 12.0) * d + 4.0) * d
        } else {
            d.sqrt()
        };
        d + (2.0 * s - 1.0) * (dd - d)
    }
}

#[inline]
fn color_dodge(s: f32, d: f32) -> f32 {
    if d <= 0.0 {
        0.0
    } else if s >= 1.0 {
        1.0
    } else {
        (d / (1.0 - s)).min(1.0)
    }
}

#[inline]
fn color_burn(s: f32, d: f32) -> f32 {
    // A black source burns to black even over white (measured).
    if s <= 0.0 {
        0.0
    } else if d >= 1.0 {
        1.0
    } else {
        1.0 - ((1.0 - d) / s).min(1.0)
    }
}

#[inline]
fn divide(num: f32, den: f32) -> f32 {
    if den <= 0.0 {
        if num <= 0.0 { 0.0 } else { 1.0 }
    } else {
        num / den
    }
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let d = max - min;
    if d <= 1e-6 {
        return (0.0, 0.0, l);
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s <= 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let f = |t: f32| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    (f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0))
}

/// State for the map operators: the destination they read through the
/// source and the map's scale arguments.
struct MapContext<'a> {
    dst: &'a FloatImage,
    sx: f32,
    sy: f32,
}

impl<'a> MapContext<'a> {
    fn new(c: &Composite, dst: &'a FloatImage, _src: &FloatImage, _ox: i64, _oy: i64) -> Self {
        let default = if c.op == CompositeOp::Blur { 1.0 } else { 20.0 };
        let sx = c.arg(0, default);
        let sy = c.arg(1, sx);
        Self { dst, sx, sy }
    }

    fn sample(&self, op: CompositeOp, x: u32, y: u32, s: [f32; 4], d: [f32; 4]) -> [f32; 4] {
        match op {
            CompositeOp::Displace => {
                let px = x as f32 + (s[0] - 0.5) * self.sx;
                let py = y as f32 + (s[1] - 0.5) * self.sy;
                let mut o = [0.0f32; 4];
                self.dst.bilinear_into(px, py, &mut o);
                o
            }
            CompositeOp::Distort => {
                let px = s[0] * self.sx;
                let py = s[1] * self.sy;
                let mut o = [0.0f32; 4];
                self.dst.bilinear_into(px, py, &mut o);
                o
            }
            CompositeOp::Blur => {
                let (sigx, sigy) = (s[0] * self.sx, s[1] * self.sy);
                if sigx < 1e-3 && sigy < 1e-3 {
                    return d;
                }
                let (rx, ry) = ((3.0 * sigx).ceil() as i64, (3.0 * sigy).ceil() as i64);
                let mut acc = [0.0f32; 4];
                let mut wsum = 0.0f32;
                for dy in -ry..=ry {
                    for dx in -rx..=rx {
                        let wx = if sigx < 1e-3 {
                            if dx == 0 { 1.0 } else { 0.0 }
                        } else {
                            (-(dx * dx) as f32 / (2.0 * sigx * sigx)).exp()
                        };
                        let wy = if sigy < 1e-3 {
                            if dy == 0 { 1.0 } else { 0.0 }
                        } else {
                            (-(dy * dy) as f32 / (2.0 * sigy * sigy)).exp()
                        };
                        let w = wx * wy;
                        if w == 0.0 {
                            continue;
                        }
                        for (c, a) in acc.iter_mut().enumerate() {
                            *a += w * self.dst.sample_clamped(x as i64 + dx, y as i64 + dy, c);
                        }
                        wsum += w;
                    }
                }
                acc.map(|v| v / wsum)
            }
            _ => d,
        }
    }
}

impl TwoInputPixelFilter for Composite {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_two_pixels(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let d = to_float(dst, params)?;
        let s = to_float(src, params)?;
        let out = self.compose_float(&d, &s).to_channels(d.channels());
        Ok(if deep::is_float(params.format) {
            out.into_frame(src.pts)
        } else {
            let mut f = deep::narrow_u8(&out);
            f.pts = src.pts;
            f
        })
    }
}

fn to_float(frame: &VideoFrame, params: VideoStreamParams) -> Result<FloatImage, Error> {
    if deep::is_float(params.format) {
        FloatImage::from_frame(frame, params)
    } else {
        deep::widen_u8(frame, params)
    }
}

impl PixelFilter for Composite {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    /// The picture composited onto itself.
    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let dst = VideoFrame {
            pts: input.pts,
            planes: input.planes.clone(),
        };
        self.apply_two(input, &dst, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn rgba(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&px(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize * 4,
                data,
            }],
        }
    }

    fn p(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn names_round_trip_with_aliases() {
        for &op in CompositeOp::ALL {
            assert_eq!(CompositeOp::from_name(op.name()), Some(op), "{op:?}");
        }
        assert_eq!(CompositeOp::from_name("Src-Over"), Some(CompositeOp::Over));
        assert_eq!(
            CompositeOp::from_name("CopyOpacity"),
            Some(CompositeOp::CopyAlpha)
        );
        assert_eq!(CompositeOp::from_name("Minus"), Some(CompositeOp::MinusDst));
        assert_eq!(
            CompositeOp::from_name("hard_light"),
            Some(CompositeOp::HardLight)
        );
        assert_eq!(CompositeOp::from_name("nope"), None);
    }

    #[test]
    fn over_of_opaque_source_is_the_source_and_clear_clears() {
        let src = rgba(2, 2, |_, _| [10, 20, 30, 255]);
        let dst = rgba(2, 2, |_, _| [200, 200, 200, 255]);
        let out = Composite::new(CompositeOp::Over)
            .apply_two(&src, &dst, p(2, 2))
            .unwrap();
        assert_eq!(&out.planes[0].data[..4], &[10, 20, 30, 255]);
        let out = Composite::new(CompositeOp::Clear)
            .apply_two(&src, &dst, p(2, 2))
            .unwrap();
        assert_eq!(&out.planes[0].data[..4], &[0, 0, 0, 0]);
        let out = Composite::new(CompositeOp::Dst)
            .apply_two(&src, &dst, p(2, 2))
            .unwrap();
        assert_eq!(&out.planes[0].data[..4], &[200, 200, 200, 255]);
    }

    #[test]
    fn half_transparent_over_blends_and_keeps_union_alpha() {
        let src = rgba(1, 1, |_, _| [255, 0, 0, 128]);
        let dst = rgba(1, 1, |_, _| [0, 0, 255, 255]);
        let out = Composite::new(CompositeOp::Over)
            .apply_two(&src, &dst, p(1, 1))
            .unwrap();
        let o = &out.planes[0].data;
        assert_eq!(o[3], 255);
        assert!(
            (o[0] as i32 - 128).abs() <= 1 && (o[2] as i32 - 127).abs() <= 1,
            "{o:?}"
        );
        let out = Composite::new(CompositeOp::Multiply)
            .apply_two(&src, &dst, p(1, 1))
            .unwrap();
        // Sa·Da·(1·0) + Sa·(1−Da)·1 + Da·(1−Sa)·0 = 0 for red … the measured
        // ImageMagick form; blue = Da·(1−Sa)·1 = 0.498.
        let o = &out.planes[0].data;
        assert_eq!(o[0], 0);
        assert!((o[2] as i32 - 127).abs() <= 1, "{o:?}");
        assert_eq!(o[3], 255);
    }

    #[test]
    fn placement_offset_gravity_and_outside_overlay() {
        let dst = rgba(6, 4, |_, _| [0, 0, 0, 255]);
        let src = rgba(2, 2, |_, _| [255, 255, 255, 255]);
        let dp = p(6, 4);
        let sp = p(2, 2);
        // Offset placement: the overlay lands at (3, 1).
        let out = Composite::new(CompositeOp::Over)
            .with_offset(3, 1)
            .compose(&dst, dp, &src, sp)
            .unwrap();
        let px = |x: usize, y: usize| out.planes[0].data[(y * 6 + x) * 4];
        assert_eq!(px(3, 1), 255);
        assert_eq!(px(4, 2), 255);
        assert_eq!(px(2, 1), 0);
        assert_eq!(px(5, 3), 0);
        // Gravity south-east with a negative offset.
        let out = Composite::new(CompositeOp::Over)
            .with_gravity(Gravity::SouthEast)
            .with_offset(-1, 0)
            .compose(&dst, dp, &src, sp)
            .unwrap();
        let px = |x: usize, y: usize| out.planes[0].data[(y * 6 + x) * 4];
        assert_eq!(px(3, 2), 255);
        assert_eq!(px(4, 3), 255);
        assert_eq!(px(5, 3), 0);
        // Src outside the overlay: cleared by default, kept when asked.
        let out = Composite::new(CompositeOp::Src)
            .compose(&dst, dp, &src, sp)
            .unwrap();
        assert_eq!(
            out.planes[0].data[(3 * 6 + 5) * 4 + 3],
            0,
            "transparent outside the overlay"
        );
        let out = Composite::new(CompositeOp::Src)
            .with_outside_overlay(false)
            .compose(&dst, dp, &src, sp)
            .unwrap();
        assert_eq!(out.planes[0].data[(3 * 6 + 5) * 4 + 3], 255);
        // Channel mask: only green written.
        let out = Composite::new(CompositeOp::Src)
            .with_channels(ChannelMask::new(false, true, false, false))
            .compose(&dst, dp, &src, sp)
            .unwrap();
        assert_eq!(&out.planes[0].data[..4], &[0, 255, 0, 255]);
        // A gray destination comes back as gray.
        let gdst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 6,
                data: vec![0; 24],
            }],
        };
        let out = Composite::new(CompositeOp::Over)
            .compose(
                &gdst,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 6,
                    height: 4,
                },
                &src,
                sp,
            )
            .unwrap();
        assert_eq!(out.planes[0].stride, 6);
        assert_eq!(out.planes[0].data[0], 255);
        assert_eq!(out.planes[0].data[5], 0);
    }

    #[test]
    fn copy_channels_and_change_mask() {
        let src = rgba(1, 1, |_, _| [10, 20, 30, 40]);
        let dst = rgba(1, 1, |_, _| [100, 110, 120, 255]);
        let run = |op| {
            Composite::new(op)
                .apply_two(&src, &dst, p(1, 1))
                .unwrap()
                .planes[0]
                .data
                .clone()
        };
        assert_eq!(run(CompositeOp::CopyRed), vec![10, 110, 120, 255]);
        assert_eq!(run(CompositeOp::CopyGreen), vec![100, 20, 120, 255]);
        assert_eq!(run(CompositeOp::CopyBlue), vec![100, 110, 30, 255]);
        assert_eq!(run(CompositeOp::CopyAlpha), vec![100, 110, 120, 40]);
        assert_eq!(run(CompositeOp::ChangeMask), vec![100, 110, 120, 255]);
        let same = Composite::new(CompositeOp::ChangeMask)
            .apply_two(&dst, &dst, p(1, 1))
            .unwrap();
        assert_eq!(same.planes[0].data[3], 0);
    }
}
