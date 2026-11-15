//! The general distortion framework behind `-distort` / `distortImage`
//! (issue #11): one reverse mapping per method ([`ReverseMap`]: output
//! position → source position, with its local Jacobian), a shared edge
//! rule ([`VirtualPixel`]), a shared point sampler ([`PixelInterpolate`])
//! and elliptical weighted averaging with the resize kernels used as
//! radial (cylindrical) filters. A caller's own [`ReverseMap`] runs
//! through the same engine ([`DistortImage::distort_with`]).
//!
//! Coordinates follow ImageMagick's: pixel `(i, j)` covers `[i, i + 1) ×
//! [j, j + 1)`, so its centre is `(i + ½, j + ½)`. Every method maps an
//! output centre back to a continuous source position; the point
//! sampler reads it at `(u − ½, v − ½)` in index space. The conventions
//! (argument order, angle direction, default centres and radii, output
//! geometry) were fixed black-box against `magick` 7.1.2 Q16-HDRI —
//! `tests/magick_distort.rs` states the tolerance per method.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::{self, FloatImage};
use crate::resample_engine::{EdgeColours, EwaKernel, Gather, Px, Source, View, render};
use crate::resize::RadialKernel;
use crate::{FilterOptions, PixelFilter, ResampleFilter, VideoStreamParams, layout};

/// What is read outside the picture (`-virtual-pixel`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VirtualPixel {
    /// Replicate the nearest edge pixel (`Edge`, `Undefined`).
    #[default]
    Edge,
    /// The background colour.
    Background,
    /// Opaque black.
    Black,
    /// Opaque white.
    White,
    /// Opaque 50 % grey.
    Gray,
    /// Transparent black.
    Transparent,
    /// Tile the picture.
    Tile,
    /// Tile with every other copy mirrored.
    Mirror,
    /// Tile horizontally; background above and below.
    HorizontalTile,
    /// Tile vertically; background left and right.
    VerticalTile,
    /// Tile horizontally; replicate the edge above and below.
    HorizontalTileEdge,
    /// Tile vertically; replicate the edge left and right.
    VerticalTileEdge,
    /// Tile in a checkerboard of picture and background.
    CheckerTile,
    /// A pseudo-random pixel of the picture (seeded, deterministic).
    Random,
    /// The picture's rim scattered outward: the lookup is displaced by
    /// an ordered (4 × 4 threshold map) offset of up to eight pixels and
    /// then clamped into the picture.
    Dither,
}

impl VirtualPixel {
    /// Parse a `-virtual-pixel` name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "edge" | "undefined" => Self::Edge,
            "background" => Self::Background,
            "black" => Self::Black,
            "white" => Self::White,
            "gray" | "grey" => Self::Gray,
            "transparent" | "none" => Self::Transparent,
            "tile" => Self::Tile,
            "mirror" => Self::Mirror,
            "horizontaltile" => Self::HorizontalTile,
            "verticaltile" => Self::VerticalTile,
            "horizontaltileedge" => Self::HorizontalTileEdge,
            "verticaltileedge" => Self::VerticalTileEdge,
            "checkertile" => Self::CheckerTile,
            "random" => Self::Random,
            "dither" => Self::Dither,
            _ => return None,
        })
    }
}

/// How a single continuous position is read (`-interpolate`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PixelInterpolate {
    /// The pixel containing the position (floor).
    Integer,
    /// The pixel whose centre is nearest.
    Nearest,
    /// Mean of the 2 × 2 neighbourhood.
    Average,
    /// Mean of the 3 × 3 neighbourhood.
    Average9,
    /// Mean of the 4 × 4 neighbourhood.
    Average16,
    /// Bilinear blend of the 2 × 2 neighbourhood.
    #[default]
    Bilinear,
    /// Catmull–Rom over 4 × 4.
    Catrom,
    /// Cubic B-spline over 4 × 4.
    Spline,
    /// The background colour.
    Background,
    /// The 2 × 2 cell split into two triangles along the diagonal whose
    /// ends differ least, read from the plane through the triangle's
    /// corners (no blending across the stronger edge).
    Mesh,
}

impl PixelInterpolate {
    /// Parse an `-interpolate` name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "integer" => Self::Integer,
            "nearest" | "nearestneighbor" => Self::Nearest,
            "average" | "average4" => Self::Average,
            "average9" => Self::Average9,
            "average16" => Self::Average16,
            "bilinear" => Self::Bilinear,
            "catrom" | "bicubic" => Self::Catrom,
            "spline" => Self::Spline,
            "background" => Self::Background,
            "mesh" => Self::Mesh,
            _ => return None,
        })
    }
}

/// A reverse mapping: for an output position, the source position that
/// lands there. Positions are continuous with pixel `(i, j)` covering
/// `[i, i + 1) × [j, j + 1)` (centre at `(i + ½, j + ½)`).
///
/// The engine calls [`source`](Self::source) once per output pixel and,
/// for area resampling, [`jacobian`](Self::jacobian) to size the
/// footprint. The default Jacobian is a central difference of `source`;
/// a mapping that knows its derivatives should give them.
pub trait ReverseMap: Sync {
    /// The source position read at output position `(x, y)`; `None`
    /// where the mapping is undefined (painted with the matte colour).
    fn source(&self, x: f64, y: f64) -> Option<(f64, f64)>;

    /// `[[∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y]]` of `(u, v) = source(x, y)`.
    fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let h = 0.25;
        let f = |x, y| self.source(x, y);
        let diff = |p: Option<(f64, f64)>, m: Option<(f64, f64)>| match (p, m) {
            (Some(p), Some(m)) => ((p.0 - m.0) / (2.0 * h), (p.1 - m.1) / (2.0 * h)),
            _ => (f64::NAN, f64::NAN),
        };
        let (ux, vx) = diff(f(x + h, y), f(x - h, y));
        let (uy, vy) = diff(f(x, y + h), f(x, y - h));
        [[ux, uy], [vx, vy]]
    }
}

impl<F> ReverseMap for F
where
    F: Fn(f64, f64) -> Option<(f64, f64)> + Sync,
{
    fn source(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self(x, y)
    }
}

/// A distortion method (`-distort` name) — its arguments are passed in
/// ImageMagick's documented order (see each variant).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DistortMethod {
    /// Control points `u,v x,y …` (source then destination); one pair
    /// translates, two fit a similarity, three or more a least-squares
    /// affine.
    Affine,
    /// `sx, ry, rx, sy, tx, ty`: the forward matrix
    /// `x = sx·u + ry·v + tx`, `y = rx·u + sy·v + ty`.
    AffineProjection,
    /// `[X,Y] [Scale[,ScaleY]] Angle [NewX,NewY]` (1 to 7 values).
    ScaleRotateTranslate,
    /// Four or more control point pairs; least-squares homography.
    Perspective,
    /// The eight forward homography coefficients `a … h`:
    /// `x = (a u + b v + c) / (g u + h v + 1)`, `y = (d u + e v + f) /
    /// (g u + h v + 1)`.
    PerspectiveProjection,
    /// Control points; the source is a bilinear function of the
    /// destination.
    BilinearReverse,
    /// Control points; the destination is a bilinear function of the
    /// source (inverted per pixel).
    BilinearForward,
    /// `order` then control points; least-squares polynomial of the
    /// destination giving the source (`order` 1 … 5, `1.5` bilinear).
    Polynomial,
    /// `arc_angle [rotate [top_radius [bottom_radius]]]`.
    Arc,
    /// `[Rmax [Rmin [CenterX,CenterY [start,end]]]]`: polar to cartesian.
    Polar,
    /// `[Rmax [Rmin [CenterX,CenterY [start,end]]]]`: cartesian to polar.
    DePolar,
    /// `A B C [D [X,Y]]`: `Rsrc = Rdst · (A r³ + B r² + C r + D)`.
    Barrel,
    /// `A B C [D [X,Y]]`: `Rsrc = Rdst / (A r³ + B r² + C r + D)`.
    BarrelInverse,
    /// Control points; inverse-distance-weighted displacement
    /// (weights `1 / dᵖ`, `p` from
    /// [`DistortImage::with_shepards_power`], default 2).
    Shepards,
    /// `fov_angle [CenterX,CenterY]`: the picture is a strip of a
    /// cylinder spanning `fov_angle` degrees (radius `f = w / fov`),
    /// re-projected onto the tangent plane: `u = cx + f·atan(dx / f)`,
    /// `v = cy + dy·f / √(dx² + f²)`.
    Cylinder2Plane,
    /// `fov_angle [CenterX,CenterY]`: the picture is a plane seen under
    /// `fov_angle` degrees (focal length `f = (w/2) / tan(fov/2)`),
    /// wrapped onto the cylinder of radius `f`: `u = cx + f·tan(dx / f)`,
    /// `v = cy + dy / cos(dx / f)`.
    Plane2Cylinder,
    /// `width, height`: plain resampling to that extent through the
    /// distortion engine (the output window is `width × height`).
    Resize,
}

impl DistortMethod {
    /// Parse a `-distort` method name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "affine" => Self::Affine,
            "affineprojection" => Self::AffineProjection,
            "scalerotatetranslate" | "srt" => Self::ScaleRotateTranslate,
            "perspective" => Self::Perspective,
            "perspectiveprojection" => Self::PerspectiveProjection,
            // `Bilinear` is magick's name for the forward form (measured).
            "bilinearreverse" => Self::BilinearReverse,
            "bilinear" | "bilinearforward" => Self::BilinearForward,
            "polynomial" => Self::Polynomial,
            "arc" => Self::Arc,
            "polar" => Self::Polar,
            "depolar" => Self::DePolar,
            "barrel" => Self::Barrel,
            "barrelinverse" => Self::BarrelInverse,
            "shepards" => Self::Shepards,
            "cylinder2plane" => Self::Cylinder2Plane,
            "plane2cylinder" => Self::Plane2Cylinder,
            "resize" => Self::Resize,
            _ => return None,
        })
    }
}

/// How each output pixel gathers its value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resampling {
    /// A single point read through [`PixelInterpolate`] (`-filter point`).
    Point(PixelInterpolate),
    /// Elliptical weighted average of the pixel's footprint with a
    /// cylindrical use of a resize filter (ImageMagick's default for
    /// distortions is Robidoux).
    Ewa(ResampleFilter),
}

/// `distortImage`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DistortImage {
    /// The method.
    pub method: DistortMethod,
    /// Its arguments.
    pub args: Vec<f64>,
    /// `+distort`: size the output to the distorted picture's bounding
    /// box (the offset is reported by [`DistortImage::distort`]).
    pub bestfit: bool,
    /// An explicit output window `(width, height, x, y)`
    /// (`distort:viewport`).
    pub viewport: Option<(u32, u32, i64, i64)>,
    /// Edge handling.
    pub virtual_pixel: VirtualPixel,
    /// Per-pixel gathering.
    pub resampling: Resampling,
    /// Background colour (normalised RGBA) for the virtual-pixel methods
    /// that use it.
    pub background: [f32; 4],
    /// Colour of the region a mapping leaves undefined — the sky beyond a
    /// perspective horizon (`-mattecolor`, default `#BDBDBD`, measured).
    pub matte: [f32; 4],
    /// Overrides of the area-resampling filter (blur, support, window,
    /// `B` / `C`, …), applied to its radial form.
    pub filter_options: FilterOptions,
    /// Output scaling (`distort:scale`): the output window is rendered
    /// `scale` times larger, each output pixel covering `1 / scale` of a
    /// unit of the distorted plane.
    pub scale: f64,
    /// Exponent `p` of the Shepard weights `1 / dᵖ` (`shepards:power`).
    pub shepards_power: f64,
}

/// The output of [`DistortImage::distort`]: the picture and the page
/// offset of its top-left corner.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Distorted {
    /// The distorted picture (RGBA, or the input's channel count).
    pub image: FloatImage,
    /// Page offset of the output's top-left corner.
    pub offset: (i64, i64),
}

/// An RGBA colour in a `channels`-sample layout (grey takes the Rec. 601
/// luma).
fn colour_in_layout(rgba: [f32; 4], channels: usize) -> Px {
    match channels {
        1 => [FloatImage::luma(&rgba[..3]), 0.0, 0.0, 0.0],
        2 => [FloatImage::luma(&rgba[..3]), rgba[3], 0.0, 0.0],
        3 => [rgba[0], rgba[1], rgba[2], 0.0],
        _ => rgba,
    }
}

impl DistortImage {
    /// A method with its arguments; Edge virtual pixels, EWA Robidoux,
    /// white background (ImageMagick's defaults).
    pub fn new(method: DistortMethod, args: impl Into<Vec<f64>>) -> Self {
        Self {
            method,
            args: args.into(),
            bestfit: false,
            viewport: None,
            virtual_pixel: VirtualPixel::Edge,
            resampling: Resampling::Ewa(ResampleFilter::Robidoux),
            background: [1.0, 1.0, 1.0, 1.0],
            matte: [189.0 / 255.0, 189.0 / 255.0, 189.0 / 255.0, 1.0],
            filter_options: FilterOptions::new(),
            scale: 1.0,
            shepards_power: 2.0,
        }
    }

    /// The sampling settings alone (identity method), for a caller's own
    /// mapping through [`DistortImage::distort_with`].
    pub fn engine() -> Self {
        Self::new(
            DistortMethod::AffineProjection,
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
    }

    /// `-mattecolor` (normalised RGBA): the colour beyond a horizon.
    pub fn with_matte(mut self, rgba: [f32; 4]) -> Self {
        self.matte = rgba;
        self
    }

    /// `+distort`.
    pub fn with_bestfit(mut self, bestfit: bool) -> Self {
        self.bestfit = bestfit;
        self
    }

    /// `distort:viewport=WxH+X+Y`.
    pub fn with_viewport(mut self, w: u32, h: u32, x: i64, y: i64) -> Self {
        self.viewport = Some((w.max(1), h.max(1), x, y));
        self
    }

    /// `-virtual-pixel`.
    pub fn with_virtual_pixel(mut self, vp: VirtualPixel) -> Self {
        self.virtual_pixel = vp;
        self
    }

    /// Point sampling through `interp` (`-filter point -interpolate …`).
    pub fn with_interpolate(mut self, interp: PixelInterpolate) -> Self {
        self.resampling = Resampling::Point(interp);
        self
    }

    /// EWA with `filter`.
    pub fn with_ewa(mut self, filter: ResampleFilter) -> Self {
        self.resampling = Resampling::Ewa(filter);
        self
    }

    /// Overrides of the EWA filter (`-define filter:…`).
    pub fn with_filter_options(mut self, options: FilterOptions) -> Self {
        self.filter_options = options;
        self
    }

    /// `-background` (normalised RGBA).
    pub fn with_background(mut self, rgba: [f32; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// `distort:scale` (non-positive or non-finite → 1).
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        self
    }

    /// `shepards:power` (non-positive or non-finite → 2).
    pub fn with_shepards_power(mut self, power: f64) -> Self {
        self.shepards_power = if power.is_finite() && power > 0.0 {
            power
        } else {
            2.0
        };
        self
    }

    /// The output window `(width, height, x, y)` before output scaling
    /// for a `width × height` input.
    fn window(&self, map: &Mapping, w: u32, h: u32) -> (u32, u32, i64, i64) {
        if let Some(v) = self.viewport {
            return v;
        }
        if let Some((rw, rh)) = map.forced_extent() {
            return (rw, rh, 0, 0);
        }
        if self.bestfit || map.forces_bestfit() {
            return map.bestfit(w as f64, h as f64).unwrap_or((w, h, 0, 0));
        }
        (w, h, 0, 0)
    }

    /// The extent [`DistortImage::distort`] gives a `width × height`
    /// input (output scaling applied).
    pub fn output_extent(&self, width: u32, height: u32) -> Result<(u32, u32), Error> {
        let map = Mapping::build(self, width as f64, height as f64)?;
        let (w, h, _, _) = self.window(&map, width, height);
        Ok(self.scaled(w, h))
    }

    fn scaled(&self, w: u32, h: u32) -> (u32, u32) {
        let s = |v: u32| ((v as f64 * self.scale).round().clamp(1.0, 1e9)) as u32;
        (s(w), s(h))
    }

    /// Distort a float picture.
    pub fn distort(&self, img: &FloatImage) -> Result<Distorted, Error> {
        let map = Mapping::build(self, img.width() as f64, img.height() as f64)?;
        let window = self.window(&map, img.width(), img.height());
        // A whole-pixel shift moves samples without resampling them.
        let exact = self.scale == 1.0 && map.is_whole_pixel_shift();
        self.run(img, &map, window, exact)
    }

    /// Run a caller's reverse mapping through the engine with this
    /// value's sampling settings (virtual pixels, interpolation or EWA
    /// filter, colours, viewport, output scaling). The method and its
    /// arguments are not used; without a viewport the output window is
    /// the input extent.
    pub fn distort_with(
        &self,
        img: &FloatImage,
        map: &(impl ReverseMap + ?Sized),
    ) -> Result<Distorted, Error> {
        let window = self.viewport.unwrap_or((img.width(), img.height(), 0, 0));
        self.run(img, map, window, false)
    }

    fn run<M: ReverseMap + ?Sized>(
        &self,
        img: &FloatImage,
        map: &M,
        (ow, oh, ox, oy): (u32, u32, i64, i64),
        exact: bool,
    ) -> Result<Distorted, Error> {
        if img.width() == 0 || img.height() == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: distort needs a non-empty picture",
            ));
        }
        let (vw, vh) = self.scaled(ow, oh);
        if (vw as u64) * (vh as u64) > (1 << 28) {
            return Err(Error::invalid(
                "oxideav-image-filter: distort output window is too large",
            ));
        }
        // The edge rule may introduce transparency the input cannot hold.
        let widened;
        let work = if !img.has_alpha() && self.produces_alpha() {
            widened = img.to_rgba();
            &widened
        } else {
            img
        };
        let ch = work.channels();
        let alpha = work.has_alpha().then(|| ch - 1);
        let colours = EdgeColours {
            background: colour_in_layout(self.background, ch),
            black: colour_in_layout([0.0, 0.0, 0.0, 1.0], ch),
            white: colour_in_layout([1.0, 1.0, 1.0, 1.0], ch),
            gray: colour_in_layout([0.5, 0.5, 0.5, 1.0], ch),
            transparent: [0.0; 4],
        };
        let src = Source::new(
            work.samples(),
            work.width() as usize,
            work.height() as usize,
            ch,
            alpha,
            1.0,
            self.virtual_pixel,
            colours,
        );
        let gather = match self.resampling {
            _ if exact => Gather::Point(PixelInterpolate::Nearest),
            Resampling::Point(interp) => Gather::Point(interp),
            Resampling::Ewa(f) => Gather::Ewa(EwaKernel::new(&RadialKernel::cylindrical(
                f,
                &self.filter_options,
            ))),
        };
        let view = View {
            width: vw as usize,
            height: vh as usize,
            x: ox as f64,
            y: oy as f64,
            scale: self.scale,
        };
        let data = render(&src, map, view, &gather, colour_in_layout(self.matte, ch));
        Ok(Distorted {
            image: FloatImage::from_samples(vw, vh, ch, data)?,
            offset: (ox, oy),
        })
    }

    fn produces_alpha(&self) -> bool {
        matches!(self.virtual_pixel, VirtualPixel::Transparent) || self.background[3] < 1.0
    }
}

impl PixelFilter for DistortImage {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    /// The viewport output (no page offset carriage); `+distort` /
    /// viewport / output scaling change the extent.
    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let img = if deep::is_float(params.format) {
            FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        // Keep the input's channel count so the declared layout holds.
        let out = self.distort(&img)?.image.to_channels(img.channels());
        Ok(if deep::is_float(params.format) {
            out.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&out);
            f.pts = input.pts;
            f
        })
    }
}

/// A resolved reverse mapping (destination → source), pixel-centre
/// coordinates.
#[derive(Clone, Debug)]
enum Mapping {
    /// `u = m0 x + m1 y + m2`, `v = m3 x + m4 y + m5`.
    Affine([f64; 6]),
    /// Reverse homography (row-major 3×3).
    Perspective([f64; 9]),
    /// Polynomial in the destination, evaluated on coordinates
    /// normalised as `(x − origin) / span` (keeps the least-squares
    /// system well conditioned at high order).
    Poly {
        /// Exponents `(i, j)` of each monomial `xⁱ yʲ`.
        terms: Vec<(i32, i32)>,
        cu: Vec<f64>,
        cv: Vec<f64>,
        origin: (f64, f64),
        span: f64,
    },
    /// Forward bilinear (inverted per pixel).
    BilinearForward { cx: [f64; 4], cy: [f64; 4] },
    Arc {
        w: f64,
        h: f64,
        angle0: f64,
        arc: f64,
        r_top: f64,
        r_bot: f64,
    },
    Polar {
        w: f64,
        h: f64,
        r_max: f64,
        r_min: f64,
        cx: f64,
        cy: f64,
        a0: f64,
        a1: f64,
        de: bool,
    },
    Barrel {
        coef: [f64; 4],
        cx: f64,
        cy: f64,
        norm: f64,
        inverse: bool,
    },
    Shepards {
        pts: Vec<(f64, f64, f64, f64)>,
        power: f64,
    },
    /// Cylinder of radius `f` ↔ its tangent plane, about `(cx, cy)`.
    Cylinder {
        f: f64,
        cx: f64,
        cy: f64,
        /// The output is the plane (the source the cylinder strip).
        to_plane: bool,
    },
    /// Axis scaling to a fixed output extent.
    Resize {
        sx: f64,
        sy: f64,
        width: u32,
        height: u32,
    },
}

/// Solve `A x = b` (Gaussian elimination with partial pivoting).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (dst, src) in a[row][col..n].iter_mut().zip(&pivot_row[col..n]) {
                *dst -= f * src;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// Least squares `rows · x ≈ ys` through the normal equations.
fn lstsq(rows: &[Vec<f64>], ys: &[f64]) -> Option<Vec<f64>> {
    let n = rows.first()?.len();
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for (r, y) in rows.iter().zip(ys) {
        for (i, ri) in r.iter().enumerate() {
            b[i] += ri * y;
            for (dst, rk) in a[i].iter_mut().zip(r) {
                *dst += ri * rk;
            }
        }
    }
    solve(a, b)
}

/// Highest polynomial order taken (153 coefficients per axis).
const MAX_POLY_ORDER: f64 = 16.0;

/// The monomial exponents of a polynomial of `order` (1.5 = bilinear:
/// `1, x, y, xy`), by total degree.
fn poly_terms(order: f64) -> Vec<(i32, i32)> {
    if (order - 1.5).abs() < 1e-9 {
        return vec![(0, 0), (1, 0), (0, 1), (1, 1)];
    }
    let n = order.round() as i32;
    let mut out = Vec::new();
    for total in 0..=n {
        for j in 0..=total {
            out.push((total - j, j));
        }
    }
    out
}

fn points(args: &[f64], skip: usize) -> Result<Vec<(f64, f64, f64, f64)>, Error> {
    let a = &args[skip.min(args.len())..];
    if a.len() < 4 || !a.len().is_multiple_of(4) {
        return Err(Error::invalid(
            "oxideav-image-filter: distort control points come in groups of four (u,v x,y)",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid(
            "oxideav-image-filter: distort control points must be finite",
        ));
    }
    Ok(a.as_chunks::<4>()
        .0
        .iter()
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect())
}

fn invert_affine(m: [f64; 6]) -> Option<[f64; 6]> {
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    Some([a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])])
}

fn invert3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-15 || !det.is_finite() {
        return None;
    }
    let i = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * i,
        (m[2] * m[7] - m[1] * m[8]) * i,
        (m[1] * m[5] - m[2] * m[4]) * i,
        (m[5] * m[6] - m[3] * m[8]) * i,
        (m[0] * m[8] - m[2] * m[6]) * i,
        (m[2] * m[3] - m[0] * m[5]) * i,
        (m[3] * m[7] - m[4] * m[6]) * i,
        (m[1] * m[6] - m[0] * m[7]) * i,
        (m[0] * m[4] - m[1] * m[3]) * i,
    ])
}

/// Scale a homography (an overall sign is free) so its denominator is
/// positive at the destination point `(x, y)` known to be visible; the
/// other side of the horizon is then where it turns negative.
fn orient(m: [f64; 9], x: f64, y: f64) -> [f64; 9] {
    if m[6] * x + m[7] * y + m[8] < 0.0 {
        m.map(|v| -v)
    } else {
        m
    }
}

fn bad(method: &str, what: &str) -> Error {
    Error::invalid(format!("oxideav-image-filter: distort {method}: {what}"))
}

impl Mapping {
    fn build(spec: &DistortImage, w: f64, h: f64) -> Result<Self, Error> {
        use DistortMethod as M;
        let (method, args) = (spec.method, spec.args.as_slice());
        if args.iter().any(|v| !v.is_finite()) {
            return Err(bad("arguments", "must be finite"));
        }
        Ok(match method {
            M::AffineProjection => {
                if args.len() != 6 {
                    return Err(bad("AffineProjection", "needs 6 coefficients"));
                }
                let fwd = [args[0], args[2], args[4], args[1], args[3], args[5]];
                Mapping::Affine(
                    invert_affine(fwd).ok_or_else(|| bad("AffineProjection", "singular"))?,
                )
            }
            M::Affine => {
                let pts = points(args, 0)?;
                let m = match pts.len() {
                    1 => {
                        let (u, v, x, y) = pts[0];
                        [1.0, 0.0, u - x, 0.0, 1.0, v - y]
                    }
                    2 => {
                        // Similarity: u = a x − b y + c, v = b x + a y + d.
                        let rows: Vec<Vec<f64>> = pts
                            .iter()
                            .flat_map(|&(_, _, x, y)| [vec![x, -y, 1.0, 0.0], vec![y, x, 0.0, 1.0]])
                            .collect();
                        let ys: Vec<f64> = pts.iter().flat_map(|&(u, v, _, _)| [u, v]).collect();
                        let s =
                            lstsq(&rows, &ys).ok_or_else(|| bad("Affine", "degenerate points"))?;
                        [s[0], -s[1], s[2], s[1], s[0], s[3]]
                    }
                    _ => {
                        let rows: Vec<Vec<f64>> =
                            pts.iter().map(|&(_, _, x, y)| vec![x, y, 1.0]).collect();
                        let us: Vec<f64> = pts.iter().map(|p| p.0).collect();
                        let vs: Vec<f64> = pts.iter().map(|p| p.1).collect();
                        let cu =
                            lstsq(&rows, &us).ok_or_else(|| bad("Affine", "degenerate points"))?;
                        let cv =
                            lstsq(&rows, &vs).ok_or_else(|| bad("Affine", "degenerate points"))?;
                        [cu[0], cu[1], cu[2], cv[0], cv[1], cv[2]]
                    }
                };
                Mapping::Affine(m)
            }
            M::ScaleRotateTranslate => {
                // 1: angle · 2: scale angle · 3: sx sy angle ·
                // 4: x y scale angle · 5: x y sx sy angle ·
                // 6: x y scale angle nx ny · 7: x y sx sy angle nx ny.
                let n = args.len();
                if !(1..=7).contains(&n) {
                    return Err(bad("SRT", "takes 1 to 7 values"));
                }
                let (cx, cy) = if n >= 4 {
                    (args[0], args[1])
                } else {
                    (w / 2.0, h / 2.0)
                };
                let rest = if n >= 4 { &args[2..] } else { args };
                let (sx, sy, angle) = match n {
                    1 => (1.0, 1.0, rest[0]),
                    2 | 4 | 6 => (rest[0], rest[0], rest[1]),
                    _ => (rest[0], rest[1], rest[2]),
                };
                let (nx, ny) = if n >= 6 {
                    (args[n - 2], args[n - 1])
                } else {
                    (cx, cy)
                };
                let (s, c) = angle.to_radians().sin_cos();
                // Forward: x = sx c (u − cx) − sy s (v − cy) + nx, …
                let (f0, f1, f3, f4) = (sx * c, -sy * s, sx * s, sy * c);
                let fwd = [
                    f0,
                    f1,
                    nx - f0 * cx - f1 * cy,
                    f3,
                    f4,
                    ny - f3 * cx - f4 * cy,
                ];
                Mapping::Affine(invert_affine(fwd).ok_or_else(|| bad("SRT", "zero scale"))?)
            }
            M::PerspectiveProjection => {
                if args.len() != 8 {
                    return Err(bad("PerspectiveProjection", "needs 8 coefficients"));
                }
                let fwd = [
                    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], 1.0,
                ];
                let inv = invert3(fwd).ok_or_else(|| bad("PerspectiveProjection", "singular"))?;
                // The picture's centre is on the visible side.
                let (u, v) = (w / 2.0, h / 2.0);
                let wq = fwd[6] * u + fwd[7] * v + 1.0;
                let (rx, ry) = (
                    (fwd[0] * u + fwd[1] * v + fwd[2]) / wq,
                    (fwd[3] * u + fwd[4] * v + fwd[5]) / wq,
                );
                Mapping::Perspective(orient(inv, rx, ry))
            }
            M::Perspective => {
                let pts = points(args, 0)?;
                if pts.len() < 4 {
                    return Err(bad("Perspective", "needs four control point pairs"));
                }
                // Destination → source directly: u (g x + h y + 1) =
                // a x + b y + c is linear in the eight unknowns.
                let mut rows = Vec::new();
                let mut ys = Vec::new();
                for &(u, v, x, y) in &pts {
                    rows.push(vec![x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
                    ys.push(u);
                    rows.push(vec![0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
                    ys.push(v);
                }
                let c = lstsq(&rows, &ys).ok_or_else(|| bad("Perspective", "degenerate points"))?;
                let n = pts.len() as f64;
                let (rx, ry) = pts
                    .iter()
                    .fold((0.0, 0.0), |a, p| (a.0 + p.2 / n, a.1 + p.3 / n));
                Mapping::Perspective(orient(
                    [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 1.0],
                    rx,
                    ry,
                ))
            }
            M::BilinearReverse | M::Polynomial => {
                let (order, skip) = if method == M::Polynomial {
                    (
                        *args
                            .first()
                            .ok_or_else(|| bad("Polynomial", "needs an order"))?,
                        1,
                    )
                } else {
                    (1.5, 0)
                };
                if !(1.0..=MAX_POLY_ORDER).contains(&order) {
                    return Err(bad("Polynomial", "order must be 1 … 16"));
                }
                let pts = points(args, skip)?;
                let terms = poly_terms(order);
                if pts.len() < terms.len() {
                    return Err(bad("Polynomial", "too few control points for the order"));
                }
                // Normalise the destination coordinates over the control
                // points' bounding box.
                let (mut x0, mut y0, mut x1, mut y1) = (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                );
                for &(_, _, x, y) in &pts {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
                let origin = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
                let span = ((x1 - x0).max(y1 - y0) / 2.0).max(1e-9);
                let rows: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|&(_, _, x, y)| {
                        let (xn, yn) = ((x - origin.0) / span, (y - origin.1) / span);
                        terms
                            .iter()
                            .map(|&(i, j)| xn.powi(i) * yn.powi(j))
                            .collect()
                    })
                    .collect();
                let us: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let vs: Vec<f64> = pts.iter().map(|p| p.1).collect();
                Mapping::Poly {
                    cu: lstsq(&rows, &us).ok_or_else(|| bad("Polynomial", "degenerate points"))?,
                    cv: lstsq(&rows, &vs).ok_or_else(|| bad("Polynomial", "degenerate points"))?,
                    terms,
                    origin,
                    span,
                }
            }
            M::BilinearForward => {
                let pts = points(args, 0)?;
                if pts.len() < 4 {
                    return Err(bad("BilinearForward", "needs four control point pairs"));
                }
                let rows: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|&(u, v, _, _)| vec![1.0, u, v, u * v])
                    .collect();
                let xs: Vec<f64> = pts.iter().map(|p| p.2).collect();
                let ys: Vec<f64> = pts.iter().map(|p| p.3).collect();
                let cx =
                    lstsq(&rows, &xs).ok_or_else(|| bad("BilinearForward", "degenerate points"))?;
                let cy =
                    lstsq(&rows, &ys).ok_or_else(|| bad("BilinearForward", "degenerate points"))?;
                Mapping::BilinearForward {
                    cx: [cx[0], cx[1], cx[2], cx[3]],
                    cy: [cy[0], cy[1], cy[2], cy[3]],
                }
            }
            M::Arc => {
                // Measured: the arc starts at −90° − arc/2 + rotate (y
                // down); by default the middle row has the picture's width
                // as arc length and the radii span h − 1; a lone top
                // radius keeps the default bottom / top ratio.
                let arc = args.first().copied().unwrap_or(90.0).clamp(1e-6, 360.0);
                let rot = args.get(1).copied().unwrap_or(0.0);
                let mid = w / arc.to_radians();
                let (dt, db) = (mid + (h - 1.0) / 2.0, mid - (h - 1.0) / 2.0);
                let r_top = args.get(2).copied().unwrap_or(dt);
                let r_bot = args.get(3).copied().unwrap_or(if args.len() > 2 {
                    r_top * db / dt
                } else {
                    db
                });
                if (r_top - r_bot).abs() < 1e-9 {
                    return Err(bad("Arc", "top and bottom radius coincide"));
                }
                Mapping::Arc {
                    w,
                    h,
                    angle0: -90.0 - arc / 2.0 + rot,
                    arc,
                    r_top,
                    r_bot,
                }
            }
            M::Polar | M::DePolar => {
                let r_max = match args.first() {
                    Some(&r) if r > 0.0 => r,
                    _ => (w / 2.0).min(h / 2.0),
                };
                let r_min = args.get(1).copied().unwrap_or(0.0);
                let cx = args.get(2).copied().unwrap_or(w / 2.0);
                let cy = args.get(3).copied().unwrap_or(h / 2.0);
                let a0 = args.get(4).copied().unwrap_or(-180.0);
                let a1 = args.get(5).copied().unwrap_or(a0 + 360.0);
                if (r_max - r_min).abs() < 1e-9 || (a1 - a0).abs() < 1e-9 {
                    return Err(bad("Polar", "empty radius or angle range"));
                }
                Mapping::Polar {
                    w,
                    h,
                    r_max,
                    r_min,
                    cx,
                    cy,
                    a0,
                    a1,
                    de: method == M::DePolar,
                }
            }
            M::Barrel | M::BarrelInverse => {
                if args.len() < 3 {
                    return Err(bad("Barrel", "needs A B C"));
                }
                let (a, b, c) = (args[0], args[1], args[2]);
                let d = args.get(3).copied().unwrap_or(1.0 - (a + b + c));
                let cx = args.get(4).copied().unwrap_or(w / 2.0);
                let cy = args.get(5).copied().unwrap_or(h / 2.0);
                Mapping::Barrel {
                    coef: [a, b, c, d],
                    cx,
                    cy,
                    norm: w.min(h) / 2.0,
                    inverse: method == M::BarrelInverse,
                }
            }
            M::Shepards => Mapping::Shepards {
                pts: points(args, 0)?,
                power: spec.shepards_power,
            },
            M::Cylinder2Plane | M::Plane2Cylinder => {
                let fov = args
                    .first()
                    .copied()
                    .ok_or_else(|| bad("Cylinder", "needs the field-of-view angle"))?
                    .to_radians();
                let to_plane = method == M::Cylinder2Plane;
                // A plane cannot be seen under 180° or more; a cylinder
                // strip can span any positive angle.
                let limit = if to_plane {
                    f64::INFINITY
                } else {
                    std::f64::consts::PI
                };
                if !(fov > 1e-9 && fov < limit) {
                    return Err(bad("Cylinder", "field of view out of range"));
                }
                let f = if to_plane {
                    w / fov
                } else {
                    (w / 2.0) / (fov / 2.0).tan()
                };
                Mapping::Cylinder {
                    f,
                    cx: args.get(1).copied().unwrap_or(w / 2.0),
                    cy: args.get(2).copied().unwrap_or(h / 2.0),
                    to_plane,
                }
            }
            M::Resize => {
                if args.len() != 2 || args[0] < 1.0 || args[1] < 1.0 {
                    return Err(bad("Resize", "needs a width and a height of at least 1"));
                }
                let (rw, rh) = (args[0].round().min(1e9), args[1].round().min(1e9));
                Mapping::Resize {
                    sx: w / rw,
                    sy: h / rh,
                    width: rw as u32,
                    height: rh as u32,
                }
            }
        })
    }

    fn forces_bestfit(&self) -> bool {
        matches!(self, Mapping::Arc { .. })
    }

    /// The output extent a method fixes by itself.
    fn forced_extent(&self) -> Option<(u32, u32)> {
        match self {
            Mapping::Resize { width, height, .. } => Some((*width, *height)),
            _ => None,
        }
    }

    /// `true` when the map moves the picture by whole pixels only (the
    /// identity included): every output pixel is then exactly one source
    /// pixel.
    fn is_whole_pixel_shift(&self) -> bool {
        match self {
            Mapping::Affine(m) => {
                m[0] == 1.0
                    && m[1] == 0.0
                    && m[3] == 0.0
                    && m[4] == 1.0
                    && m[2].fract() == 0.0
                    && m[5].fract() == 0.0
            }
            Mapping::Resize { sx, sy, .. } => *sx == 1.0 && *sy == 1.0,
            _ => false,
        }
    }

    /// The `+distort` output window `(w, h, x, y)` for the methods with a
    /// forward map: the distorted outline's bounding box widened to whole
    /// pixels as `floor(min − ½) … ceil(max + ½)` (measured).
    fn bestfit(&self, w: f64, h: f64) -> Option<(u32, u32, i64, i64)> {
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
        let mut pts: Vec<(f64, f64)> = Vec::new();
        match self {
            Mapping::Affine(m) => {
                let f = invert_affine(*m)?;
                for (u, v) in corners {
                    pts.push((f[0] * u + f[1] * v + f[2], f[3] * u + f[4] * v + f[5]));
                }
            }
            Mapping::Perspective(m) => {
                let f = invert3(*m)?;
                for (u, v) in corners {
                    let wq = f[6] * u + f[7] * v + f[8];
                    if wq.abs() < 1e-12 {
                        return None;
                    }
                    pts.push((
                        (f[0] * u + f[1] * v + f[2]) / wq,
                        (f[3] * u + f[4] * v + f[5]) / wq,
                    ));
                }
            }
            Mapping::BilinearForward { cx, cy } => {
                for (u, v) in corners {
                    pts.push((
                        cx[0] + cx[1] * u + cx[2] * v + cx[3] * u * v,
                        cy[0] + cy[1] * u + cy[2] * v + cy[3] * u * v,
                    ));
                }
            }
            Mapping::Arc {
                angle0,
                arc,
                r_top,
                r_bot,
                ..
            } => {
                let mut angles = vec![*angle0, angle0 + arc];
                let mut k = (angle0 / 90.0).ceil() * 90.0;
                while k < angle0 + arc {
                    angles.push(k);
                    k += 90.0;
                }
                for a in angles {
                    let (s, c) = a.to_radians().sin_cos();
                    for r in [*r_top, *r_bot] {
                        pts.push((r * c, r * s));
                    }
                }
            }
            Mapping::Cylinder {
                f,
                cx,
                cy,
                to_plane,
            } => {
                // The outline's extremes are at the corners and, for the
                // rows, on the axis column.
                for (u, v) in corners.into_iter().chain([(*cx, 0.0), (*cx, h)]) {
                    let (du, dv) = (u - cx, v - cy);
                    if *to_plane {
                        let t = du / f;
                        if t.abs() >= std::f64::consts::FRAC_PI_2 - 1e-3 {
                            return None;
                        }
                        pts.push((cx + f * t.tan(), cy + dv / t.cos()));
                    } else {
                        let t = (du / f).atan();
                        pts.push((cx + f * t, cy + dv * t.cos()));
                    }
                }
            }
            _ => return None,
        }
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y) in pts {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (fx0, fy0) = ((x0 - 0.5).floor(), (y0 - 0.5).floor());
        let (fx1, fy1) = ((x1 + 0.5).ceil(), (y1 + 0.5).ceil());
        if !(fx0.is_finite() && fy0.is_finite() && fx1.is_finite() && fy1.is_finite())
            || fx1 - fx0 > 1e6
            || fy1 - fy0 > 1e6
        {
            return None;
        }
        Some((
            ((fx1 - fx0) as u32).max(1),
            ((fy1 - fy0) as u32).max(1),
            fx0 as i64,
            fy0 as i64,
        ))
    }
}

impl ReverseMap for Mapping {
    /// Destination centre → source position; `None` where undefined.
    fn source(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        match self {
            Mapping::Affine(m) => Some((m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])),
            Mapping::Perspective(m) => {
                let wq = m[6] * x + m[7] * y + m[8];
                if wq <= 1e-12 {
                    return None;
                }
                Some((
                    (m[0] * x + m[1] * y + m[2]) / wq,
                    (m[3] * x + m[4] * y + m[5]) / wq,
                ))
            }
            Mapping::Poly {
                terms,
                cu,
                cv,
                origin,
                span,
            } => {
                let (xn, yn) = ((x - origin.0) / span, (y - origin.1) / span);
                let (mut u, mut v) = (0.0, 0.0);
                for ((&(i, j), a), b) in terms.iter().zip(cu).zip(cv) {
                    let m = xn.powi(i) * yn.powi(j);
                    u += a * m;
                    v += b * m;
                }
                Some((u, v))
            }
            Mapping::BilinearForward { cx, cy } => {
                // Newton on forward(u, v) = (x, y).
                let (mut u, mut v) = (x, y);
                for _ in 0..30 {
                    let fx = cx[0] + cx[1] * u + cx[2] * v + cx[3] * u * v - x;
                    let fy = cy[0] + cy[1] * u + cy[2] * v + cy[3] * u * v - y;
                    let (j00, j01) = (cx[1] + cx[3] * v, cx[2] + cx[3] * u);
                    let (j10, j11) = (cy[1] + cy[3] * v, cy[2] + cy[3] * u);
                    let det = j00 * j11 - j01 * j10;
                    if det.abs() < 1e-12 {
                        return None;
                    }
                    let du = (fx * j11 - fy * j01) / det;
                    let dv = (fy * j00 - fx * j10) / det;
                    u -= du;
                    v -= dv;
                    if du.abs() + dv.abs() < 1e-9 {
                        break;
                    }
                }
                (u.is_finite() && v.is_finite()).then_some((u, v))
            }
            Mapping::Arc {
                w,
                h,
                angle0,
                arc,
                r_top,
                r_bot,
            } => {
                let r = (x * x + y * y).sqrt();
                let a = (y.atan2(x).to_degrees() - angle0).rem_euclid(360.0);
                Some((a / arc * w, (r_top - r) / (r_top - r_bot) * h))
            }
            Mapping::Polar {
                w,
                h,
                r_max,
                r_min,
                cx,
                cy,
                a0,
                a1,
                de,
            } => {
                // Measured: angle a sits at (cx + r sin a, cy + r cos a),
                // so the default start −180° is straight up.
                if *de {
                    // Output is the polar picture: x → angle, y → radius.
                    let ang = (a0 + (x / w) * (a1 - a0)).to_radians();
                    let r = r_min + (y / h) * (r_max - r_min);
                    Some((cx + r * ang.sin(), cy + r * ang.cos()))
                } else {
                    let (dx, dy) = (x - cx, y - cy);
                    let r = (dx * dx + dy * dy).sqrt();
                    let ang = a0 + (dx.atan2(dy).to_degrees() - a0).rem_euclid(360.0);
                    Some((
                        (ang - a0) / (a1 - a0) * w,
                        (r - r_min) / (r_max - r_min) * h,
                    ))
                }
            }
            Mapping::Barrel {
                coef,
                cx,
                cy,
                norm,
                inverse,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt() / norm;
                let f = coef[0] * r * r * r + coef[1] * r * r + coef[2] * r + coef[3];
                let s = if *inverse {
                    if f.abs() < 1e-12 {
                        return None;
                    }
                    1.0 / f
                } else {
                    f
                };
                Some((cx + dx * s, cy + dy * s))
            }
            Mapping::Shepards { pts, power } => {
                let (mut su, mut sv, mut sw) = (0.0, 0.0, 0.0);
                for &(u, v, px, py) in pts {
                    let d2 = (x - px) * (x - px) + (y - py) * (y - py);
                    if d2 < 1e-12 {
                        return Some((u, v));
                    }
                    let wgt = if *power == 2.0 {
                        1.0 / d2
                    } else {
                        d2.powf(-power / 2.0)
                    };
                    su += wgt * (u - px);
                    sv += wgt * (v - py);
                    sw += wgt;
                }
                Some((x + su / sw, y + sv / sw))
            }
            Mapping::Cylinder {
                f,
                cx,
                cy,
                to_plane,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                if *to_plane {
                    Some((
                        cx + f * (dx / f).atan(),
                        cy + dy * f / (dx * dx + f * f).sqrt(),
                    ))
                } else {
                    let t = dx / f;
                    if t.abs() >= std::f64::consts::FRAC_PI_2 {
                        return None;
                    }
                    Some((cx + f * t.tan(), cy + dy / t.cos()))
                }
            }
            Mapping::Resize { sx, sy, .. } => Some((x * sx, y * sy)),
        }
    }

    /// Analytic where the derivative has a closed form; a central
    /// difference otherwise.
    fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        match self {
            Mapping::Affine(m) => [[m[0], m[1]], [m[3], m[4]]],
            Mapping::Resize { sx, sy, .. } => [[*sx, 0.0], [0.0, *sy]],
            Mapping::Perspective(m) => {
                let wq = m[6] * x + m[7] * y + m[8];
                let nu = m[0] * x + m[1] * y + m[2];
                let nv = m[3] * x + m[4] * y + m[5];
                let i2 = 1.0 / (wq * wq);
                [
                    [(m[0] * wq - nu * m[6]) * i2, (m[1] * wq - nu * m[7]) * i2],
                    [(m[3] * wq - nv * m[6]) * i2, (m[4] * wq - nv * m[7]) * i2],
                ]
            }
            Mapping::Poly {
                terms,
                cu,
                cv,
                origin,
                span,
            } => {
                let (xn, yn) = ((x - origin.0) / span, (y - origin.1) / span);
                let mut j = [[0.0; 2]; 2];
                for ((&(i, k), a), b) in terms.iter().zip(cu).zip(cv) {
                    let dx = if i > 0 {
                        i as f64 * xn.powi(i - 1) * yn.powi(k)
                    } else {
                        0.0
                    };
                    let dy = if k > 0 {
                        k as f64 * xn.powi(i) * yn.powi(k - 1)
                    } else {
                        0.0
                    };
                    j[0][0] += a * dx;
                    j[0][1] += a * dy;
                    j[1][0] += b * dx;
                    j[1][1] += b * dy;
                }
                j.map(|r| r.map(|v| v / span))
            }
            Mapping::BilinearForward { cx, cy } => {
                // The inverse of the forward Jacobian at the solution.
                let Some((u, v)) = self.source(x, y) else {
                    return [[f64::NAN; 2]; 2];
                };
                let (j00, j01) = (cx[1] + cx[3] * v, cx[2] + cx[3] * u);
                let (j10, j11) = (cy[1] + cy[3] * v, cy[2] + cy[3] * u);
                let det = j00 * j11 - j01 * j10;
                [[j11 / det, -j01 / det], [-j10 / det, j00 / det]]
            }
            Mapping::Arc {
                w,
                h,
                arc,
                r_top,
                r_bot,
                ..
            } => {
                let r2 = x * x + y * y;
                if r2 < 1e-18 {
                    return [[0.0; 2]; 2];
                }
                let r = r2.sqrt();
                // u = (atan2(y, x) − angle0) · w / arc (degrees).
                let ku = w / arc.to_radians();
                let kv = -h / (r_top - r_bot);
                [[-y / r2 * ku, x / r2 * ku], [x / r * kv, y / r * kv]]
            }
            Mapping::Polar {
                w,
                h,
                r_max,
                r_min,
                cx,
                cy,
                a0,
                a1,
                de,
            } => {
                if *de {
                    let ka = (a1 - a0).to_radians() / w;
                    let kr = (r_max - r_min) / h;
                    let (s, c) = (a0 + (x / w) * (a1 - a0)).to_radians().sin_cos();
                    let r = r_min + (y / h) * (r_max - r_min);
                    [[r * c * ka, s * kr], [-r * s * ka, c * kr]]
                } else {
                    let (dx, dy) = (x - cx, y - cy);
                    let r2 = dx * dx + dy * dy;
                    if r2 < 1e-18 {
                        return [[0.0; 2]; 2];
                    }
                    let r = r2.sqrt();
                    // u = (atan2(dx, dy) − a0) · w / (a1 − a0).
                    let ku = w / (a1 - a0).to_radians();
                    let kv = h / (r_max - r_min);
                    [[dy / r2 * ku, -dx / r2 * ku], [dx / r * kv, dy / r * kv]]
                }
            }
            Mapping::Barrel {
                coef,
                cx,
                cy,
                norm,
                inverse,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt() / norm;
                let f = coef[0] * r * r * r + coef[1] * r * r + coef[2] * r + coef[3];
                let df = 3.0 * coef[0] * r * r + 2.0 * coef[1] * r + coef[2];
                let (s, ds) = if *inverse {
                    (1.0 / f, -df / (f * f))
                } else {
                    (f, df)
                };
                if r < 1e-12 {
                    return [[s, 0.0], [0.0, s]];
                }
                // ∂r/∂x = dx / (norm² r).
                let g = ds / (norm * norm * r);
                [
                    [s + dx * dx * g, dx * dy * g],
                    [dx * dy * g, s + dy * dy * g],
                ]
            }
            Mapping::Cylinder {
                f,
                cx,
                cy,
                to_plane,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                if *to_plane {
                    let q = dx * dx + f * f;
                    [
                        [f * f / q, 0.0],
                        [-dy * f * dx / (q * q.sqrt()), f / q.sqrt()],
                    ]
                } else {
                    let (s, c) = (dx / f).sin_cos();
                    [[1.0 / (c * c), 0.0], [dy * s / (c * c * f), 1.0 / c]]
                }
            }
            Mapping::Shepards { .. } => {
                let h = 0.25;
                let f = |x, y| self.source(x, y).unwrap_or((x, y));
                let (ux1, vx1) = f(x + h, y);
                let (ux0, vx0) = f(x - h, y);
                let (uy1, vy1) = f(x, y + h);
                let (uy0, vy0) = f(x, y - h);
                [
                    [(ux1 - ux0) / (2.0 * h), (uy1 - uy0) / (2.0 * h)],
                    [(vx1 - vx0) / (2.0 * h), (vy1 - vy0) / (2.0 * h)],
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> FloatImage {
        let mut img = FloatImage::new(w, h, 4);
        for y in 0..h {
            for x in 0..w {
                img.pixel_mut(x, y).copy_from_slice(&[
                    x as f32 / w as f32,
                    y as f32 / h as f32,
                    0.5,
                    1.0,
                ]);
            }
        }
        img
    }

    #[test]
    fn identity_affine_with_point_sampling_is_exact() {
        let img = ramp(9, 7);
        {
            let args = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let d = DistortImage::new(DistortMethod::AffineProjection, args)
                .with_interpolate(PixelInterpolate::Bilinear)
                .distort(&img)
                .unwrap();
            assert_eq!(d.image, img);
        }
        let d = DistortImage::new(DistortMethod::ScaleRotateTranslate, vec![0.0])
            .with_interpolate(PixelInterpolate::Nearest)
            .distort(&img)
            .unwrap();
        assert_eq!(d.image, img);
    }

    fn sampler(img: &FloatImage, vp: VirtualPixel) -> Source<'_> {
        Source::new(
            img.samples(),
            img.width() as usize,
            img.height() as usize,
            4,
            Some(3),
            1.0,
            vp,
            EdgeColours::default(),
        )
    }

    #[test]
    fn virtual_pixels_tile_and_mirror() {
        let img = ramp(4, 2);
        let s = sampler(&img, VirtualPixel::Tile);
        assert_eq!(s.at(4, 0), s.at(0, 0));
        assert_eq!(s.at(-1, -1), s.at(3, 1));
        let s = sampler(&img, VirtualPixel::Mirror);
        assert_eq!(s.at(4, 0), s.at(3, 0));
        assert_eq!(s.at(-1, 0), s.at(0, 0));
        let s = sampler(&img, VirtualPixel::Transparent);
        assert_eq!(s.at(-1, 0), [0.0; 4]);
        // Dither reads a pixel of the picture within eight of the rim.
        let img = ramp(40, 30);
        let s = sampler(&img, VirtualPixel::Dither);
        for (x, y) in [(-1, 5), (45, 3), (7, -3), (12, 33), (-5, -5)] {
            let p = s.at(x, y);
            let (sx, sy) = ((p[0] * 40.0).round() as i64, (p[1] * 30.0).round() as i64);
            assert!((sx - x.clamp(0, 39)).abs() <= 8 && (sy - y.clamp(0, 29)).abs() <= 8);
        }
    }

    #[test]
    fn perspective_from_points_reproduces_the_points() {
        let pts = vec![
            0.0, 0.0, 2.0, 1.0, 10.0, 0.0, 9.0, 0.0, 10.0, 10.0, 10.0, 9.0, 0.0, 10.0, 1.0, 10.0,
        ];
        let m = Mapping::build(
            &DistortImage::new(DistortMethod::Perspective, pts.clone()),
            10.0,
            10.0,
        )
        .unwrap();
        for c in pts.as_chunks::<4>().0 {
            let (u, v) = m.source(c[2], c[3]).unwrap();
            assert!((u - c[0]).abs() < 1e-6 && (v - c[1]).abs() < 1e-6);
        }
    }

    /// The mapping's `source` alone, so the trait's central difference
    /// stands in for the closed-form Jacobian.
    struct Numeric<'a>(&'a Mapping);

    impl ReverseMap for Numeric<'_> {
        fn source(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            self.0.source(x, y)
        }
    }

    #[test]
    fn closed_form_jacobians_match_central_differences() {
        use DistortMethod as M;
        let (w, h) = (80.0, 60.0);
        let quad = vec![
            0.0, 0.0, 8.0, 5.0, w, 0.0, 70.0, 2.0, w, h, 78.0, 55.0, 0.0, h, 3.0, 50.0,
        ];
        let mut six = quad.clone();
        six.extend([40.0, 30.0, 43.0, 28.0, 40.0, 0.0, 38.0, 4.0]);
        let cases: Vec<(M, Vec<f64>)> = vec![
            (M::AffineProjection, vec![0.9, 0.2, -0.1, 1.1, 5.0, -3.0]),
            (M::ScaleRotateTranslate, vec![0.6, 33.0]),
            (M::Perspective, quad.clone()),
            (M::BilinearForward, quad.clone()),
            (M::BilinearReverse, quad.clone()),
            (M::Polynomial, [vec![2.0], six].concat()),
            (M::Arc, vec![100.0, 10.0]),
            (M::Polar, vec![0.0]),
            (M::Polar, vec![35.0, 5.0, 30.0, 20.0, 10.0, 200.0]),
            (M::DePolar, vec![0.0]),
            (M::Barrel, vec![0.1, -0.05, 0.02]),
            (M::BarrelInverse, vec![0.05, 0.0, -0.03]),
            (
                M::Shepards,
                vec![10.0, 10.0, 14.0, 8.0, 50.0, 40.0, 46.0, 43.0],
            ),
            (M::Cylinder2Plane, vec![80.0]),
            (M::Plane2Cylinder, vec![80.0]),
            (M::Resize, vec![33.0, 71.0]),
        ];
        for (method, args) in cases {
            let m = Mapping::build(&DistortImage::new(method, args), w, h).unwrap();
            // Away from the polar seam and the centre singularity.
            for (x, y) in [(52.3, 41.7), (21.5, 44.5), (60.5, 12.5), (47.0, 50.2)] {
                let (x, y) = match method {
                    M::Arc => (x - 40.0, y - 110.0),
                    _ => (x, y),
                };
                let exact = m.jacobian(x, y);
                let approx = Numeric(&m).jacobian(x, y);
                for (r, (e, a)) in exact.iter().zip(&approx).enumerate() {
                    for (c, (e, a)) in e.iter().zip(a).enumerate() {
                        assert!(
                            (e - a).abs() < 2e-2 * (1.0 + a.abs()),
                            "{method:?} at ({x}, {y}) [{r}][{c}]: {e} vs {a}"
                        );
                    }
                }
            }
        }
    }
}
