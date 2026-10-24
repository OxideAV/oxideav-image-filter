//! The general distortion framework behind `-distort` / `distortImage`
//! (issue #11): one inverse mapping per method, a shared edge sampler
//! ([`VirtualPixel`]), a shared point sampler ([`PixelInterpolate`]) and
//! elliptical weighted averaging with the resize kernels.
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
            _ => return None,
        })
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
    /// Control points; inverse-distance-weighted displacement.
    Shepards,
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
        }
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

    /// `-background` (normalised RGBA).
    pub fn with_background(mut self, rgba: [f32; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Distort a float picture.
    pub fn distort(&self, img: &FloatImage) -> Result<Distorted, Error> {
        let map = Mapping::build(
            self.method,
            &self.args,
            img.width() as f64,
            img.height() as f64,
        )?;
        let (ow, oh, ox, oy) = match self.viewport {
            Some(v) => v,
            None => {
                if self.bestfit || map.forces_bestfit() {
                    map.bestfit(img.width() as f64, img.height() as f64)
                        .unwrap_or((img.width(), img.height(), 0, 0))
                } else {
                    (img.width(), img.height(), 0, 0)
                }
            }
        };
        let rgba = img.to_rgba();
        let sampler = Sampler {
            img: &rgba,
            vp: self.virtual_pixel,
            bg: self.background,
        };
        let mut out = FloatImage::new(ow, oh, 4);
        let ewa = match self.resampling {
            Resampling::Ewa(f) => Some(RadialKernel::new(f, &FilterOptions::new())),
            Resampling::Point(_) => None,
        };
        for j in 0..oh {
            for i in 0..ow {
                let x = i as f64 + ox as f64 + 0.5;
                let y = j as f64 + oy as f64 + 0.5;
                let px = match map.inverse(x, y) {
                    None => self.vp_colour_for_invalid(),
                    Some((u, v)) => match (&ewa, self.resampling) {
                        (Some(k), _) => {
                            let jac = map.jacobian(x, y);
                            sampler.ewa(u, v, jac, k)
                        }
                        (None, Resampling::Point(interp)) => {
                            sampler.interpolate(u - 0.5, v - 0.5, interp)
                        }
                        _ => unreachable!(),
                    },
                };
                out.pixel_mut(i, j).copy_from_slice(&px);
            }
        }
        let image = if img.has_alpha() || self.produces_alpha() {
            out
        } else {
            out.to_channels(img.channels())
        };
        Ok(Distorted {
            image,
            offset: (ox, oy),
        })
    }

    fn produces_alpha(&self) -> bool {
        matches!(self.virtual_pixel, VirtualPixel::Transparent) || self.background[3] < 1.0
    }

    fn vp_colour_for_invalid(&self) -> [f32; 4] {
        self.matte
    }
}

impl PixelFilter for DistortImage {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    /// The viewport output (no page offset carriage); `+distort` /
    /// viewport change the extent.
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

/// Reads the source with virtual pixels.
struct Sampler<'a> {
    img: &'a FloatImage,
    vp: VirtualPixel,
    bg: [f32; 4],
}

impl Sampler<'_> {
    /// The pixel at integer index `(x, y)`, virtual outside.
    fn at(&self, x: i64, y: i64) -> [f32; 4] {
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        let inside = |x: i64, y: i64| x >= 0 && y >= 0 && x < w && y < h;
        let get = |x: i64, y: i64| {
            let p = self.img.pixel(x as u32, y as u32);
            [p[0], p[1], p[2], p[3]]
        };
        if inside(x, y) {
            return get(x, y);
        }
        let clampx = x.clamp(0, w - 1);
        let clampy = y.clamp(0, h - 1);
        let tile = |v: i64, n: i64| v.rem_euclid(n);
        let mirror = |v: i64, n: i64| {
            let m = v.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        };
        match self.vp {
            VirtualPixel::Edge => get(clampx, clampy),
            VirtualPixel::Background => self.bg,
            VirtualPixel::Black => [0.0, 0.0, 0.0, 1.0],
            VirtualPixel::White => [1.0, 1.0, 1.0, 1.0],
            VirtualPixel::Gray => [0.5, 0.5, 0.5, 1.0],
            VirtualPixel::Transparent => [0.0, 0.0, 0.0, 0.0],
            VirtualPixel::Tile => get(tile(x, w), tile(y, h)),
            VirtualPixel::Mirror => get(mirror(x, w), mirror(y, h)),
            VirtualPixel::HorizontalTile => {
                if y < 0 || y >= h {
                    self.bg
                } else {
                    get(tile(x, w), y)
                }
            }
            VirtualPixel::VerticalTile => {
                if x < 0 || x >= w {
                    self.bg
                } else {
                    get(x, tile(y, h))
                }
            }
            VirtualPixel::HorizontalTileEdge => get(tile(x, w), clampy),
            VirtualPixel::VerticalTileEdge => get(clampx, tile(y, h)),
            VirtualPixel::CheckerTile => {
                let (tx, ty) = (x.div_euclid(w), y.div_euclid(h));
                if (tx + ty).rem_euclid(2) == 0 {
                    get(tile(x, w), tile(y, h))
                } else {
                    self.bg
                }
            }
            VirtualPixel::Random => {
                let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 29;
                get((z % w as u64) as i64, ((z >> 32) % h as u64) as i64)
            }
        }
    }

    /// Point read at continuous index position `(x, y)`.
    fn interpolate(&self, x: f64, y: f64, interp: PixelInterpolate) -> [f32; 4] {
        let (fx, fy) = (x.floor(), y.floor());
        let (ix, iy) = (fx as i64, fy as i64);
        let (tx, ty) = ((x - fx) as f32, (y - fy) as f32);
        match interp {
            PixelInterpolate::Integer => self.at(ix, iy),
            PixelInterpolate::Nearest => {
                self.at((x + 0.5).floor() as i64, (y + 0.5).floor() as i64)
            }
            PixelInterpolate::Background => self.bg,
            PixelInterpolate::Average => self.mean(ix, iy, 2),
            PixelInterpolate::Average9 => self.mean(
                (x + 0.5).floor() as i64 - 1,
                (y + 0.5).floor() as i64 - 1,
                3,
            ),
            PixelInterpolate::Average16 => self.mean(ix - 1, iy - 1, 4),
            PixelInterpolate::Bilinear => {
                let (a, b, c, d) = (
                    self.at(ix, iy),
                    self.at(ix + 1, iy),
                    self.at(ix, iy + 1),
                    self.at(ix + 1, iy + 1),
                );
                weighted(&[
                    (a, (1.0 - tx) * (1.0 - ty)),
                    (b, tx * (1.0 - ty)),
                    (c, (1.0 - tx) * ty),
                    (d, tx * ty),
                ])
            }
            PixelInterpolate::Catrom | PixelInterpolate::Spline => {
                let (b, c) = if interp == PixelInterpolate::Catrom {
                    (0.0, 0.5)
                } else {
                    (1.0, 0.0)
                };
                let wx = cubic_weights(tx, b, c);
                let wy = cubic_weights(ty, b, c);
                let mut taps = Vec::with_capacity(16);
                for (m, &wym) in wy.iter().enumerate() {
                    for (n, &wxn) in wx.iter().enumerate() {
                        taps.push((self.at(ix - 1 + n as i64, iy - 1 + m as i64), wxn * wym));
                    }
                }
                weighted(&taps)
            }
        }
    }

    fn mean(&self, x0: i64, y0: i64, n: i64) -> [f32; 4] {
        let mut taps = Vec::with_capacity((n * n) as usize);
        for y in y0..y0 + n {
            for x in x0..x0 + n {
                taps.push((self.at(x, y), 1.0));
            }
        }
        weighted(&taps)
    }

    /// Elliptical weighted average at `(u, v)` (continuous, pixel-centre
    /// coordinates) with the inverse map's Jacobian.
    fn ewa(&self, u: f64, v: f64, j: [[f64; 2]; 2], k: &RadialKernel) -> [f32; 4] {
        // Footprint of a unit output pixel: the columns of the Jacobian,
        // never smaller than one source pixel (reconstruction on enlarge).
        let (mut a, mut b, mut c, mut d) = (j[0][0], j[0][1], j[1][0], j[1][1]);
        // Inverse of J·Jᵀ is the ellipse quadratic form; clamp singular
        // values to ≥ 1 via the SVD of the 2×2.
        let (s1, s2, rot) = svd2(a, b, c, d);
        // Never smaller than one source pixel (reconstruction on enlarge);
        // capped so a footprint near a perspective horizon stays bounded.
        if !(s1.is_finite() && s2.is_finite()) {
            return self.interpolate(u - 0.5, v - 0.5, PixelInterpolate::Bilinear);
        }
        let cap = (self.img.width().max(self.img.height()) as f64).max(4.0);
        let (s1, s2) = (s1.clamp(1.0, cap), s2.clamp(1.0, cap));
        let (cr, sr) = (rot.cos(), rot.sin());
        // Reconstructed J' = R · diag(s1, s2) (orientation only matters).
        a = cr * s1;
        b = -sr * s2;
        c = sr * s1;
        d = cr * s2;
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return self.interpolate(u - 0.5, v - 0.5, PixelInterpolate::Bilinear);
        }
        // Map a source offset Δ to the output unit disc: J'^{-1} Δ.
        let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
        let r = k.radius() as f64;
        let ext = r * s1.max(s2);
        let (cx, cy) = (u - 0.5, v - 0.5);
        let (x0, x1) = ((cx - ext).ceil() as i64, (cx + ext).floor() as i64);
        let (y0, y1) = ((cy - ext).ceil() as i64, (cy + ext).floor() as i64);
        let mut acc = [0.0f64; 4];
        let mut wsum = 0.0f64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                let (qx, qy) = (ia * dx + ib * dy, ic * dx + id * dy);
                let dist = (qx * qx + qy * qy).sqrt();
                if dist >= r {
                    continue;
                }
                let w = k.weight(dist as f32) as f64;
                if w == 0.0 {
                    continue;
                }
                let p = self.at(x, y);
                // Alpha-weighted colour, as a premultiplied average.
                let pa = p[3] as f64;
                for (ch, a) in acc.iter_mut().enumerate().take(3) {
                    *a += w * p[ch] as f64 * pa;
                }
                acc[3] += w * pa;
                wsum += w;
            }
        }
        if wsum.abs() < 1e-12 {
            return self.interpolate(cx, cy, PixelInterpolate::Bilinear);
        }
        let alpha = acc[3] / wsum;
        let mut out = [0.0f32; 4];
        for ch in 0..3 {
            out[ch] = if acc[3].abs() > 1e-12 {
                (acc[ch] / acc[3]) as f32
            } else {
                0.0
            };
        }
        out[3] = alpha as f32;
        out
    }
}

/// SVD of the 2×2 `[[a, b], [c, d]]`: singular values and the rotation
/// of the left singular vectors.
fn svd2(a: f64, b: f64, c: f64, d: f64) -> (f64, f64, f64) {
    let e = (a + d) / 2.0;
    let f = (a - d) / 2.0;
    let g = (c + b) / 2.0;
    let h = (c - b) / 2.0;
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    let s1 = q + r;
    let s2 = (q - r).abs();
    let a1 = g.atan2(f);
    let a2 = h.atan2(e);
    (s1, s2, (a2 + a1) / 2.0)
}

fn weighted(taps: &[([f32; 4], f32)]) -> [f32; 4] {
    // Alpha-weighted colour (no dark fringes from transparent taps).
    let mut acc = [0.0f32; 4];
    let mut wsum = 0.0f32;
    for (p, w) in taps {
        for c in 0..3 {
            acc[c] += w * p[c] * p[3];
        }
        acc[3] += w * p[3];
        wsum += w;
    }
    let mut out = [0.0f32; 4];
    for c in 0..3 {
        out[c] = if acc[3].abs() > 1e-9 {
            acc[c] / acc[3]
        } else {
            0.0
        };
    }
    out[3] = if wsum.abs() > 1e-9 {
        acc[3] / wsum
    } else {
        0.0
    };
    out
}

fn cubic_weights(t: f32, b: f32, c: f32) -> [f32; 4] {
    let k = |x: f32| {
        let ax = x.abs();
        let v = if ax < 1.0 {
            (12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
                + (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
                + (6.0 - 2.0 * b)
        } else if ax < 2.0 {
            (-b - 6.0 * c) * ax * ax * ax
                + (6.0 * b + 30.0 * c) * ax * ax
                + (-12.0 * b - 48.0 * c) * ax
                + (8.0 * b + 24.0 * c)
        } else {
            0.0
        };
        v / 6.0
    };
    [k(t + 1.0), k(t), k(t - 1.0), k(t - 2.0)]
}

/// A resolved inverse mapping (destination → source), pixel-centre
/// coordinates.
#[derive(Clone, Debug)]
enum Mapping {
    /// `u = m0 x + m1 y + m2`, `v = m3 x + m4 y + m5`.
    Affine([f64; 6]),
    /// Inverse homography (row-major 3×3).
    Perspective([f64; 9]),
    /// Polynomial in the destination: coefficients over the monomial
    /// list of `order`.
    Poly {
        order: f64,
        cu: Vec<f64>,
        cv: Vec<f64>,
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
        out_cx: f64,
        out_cy: f64,
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
}

/// Solve the normal equations `A x = b` (Gaussian elimination with
/// partial pivoting).
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

/// Least squares `rows · x ≈ ys`.
fn lstsq(rows: &[Vec<f64>], ys: &[f64]) -> Option<Vec<f64>> {
    let n = rows.first()?.len();
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for (r, y) in rows.iter().zip(ys) {
        for i in 0..n {
            b[i] += r[i] * y;
            for k in 0..n {
                a[i][k] += r[i] * r[k];
            }
        }
    }
    solve(a, b)
}

/// The monomials of a polynomial of `order` (1.5 = bilinear) at `(x, y)`.
fn monomials(order: f64, x: f64, y: f64) -> Vec<f64> {
    if (order - 1.5).abs() < 1e-9 {
        return vec![1.0, x, y, x * y];
    }
    let n = order.round() as i32;
    let mut out = Vec::new();
    for total in 0..=n {
        for j in 0..=total {
            let i = total - j;
            out.push(x.powi(i) * y.powi(j));
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
    Ok(a.as_chunks::<4>()
        .0
        .iter()
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect())
}

fn invert_affine(m: [f64; 6]) -> Option<[f64; 6]> {
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() < 1e-12 {
        return None;
    }
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    Some([a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])])
}

fn invert3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-15 {
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
    fn build(method: DistortMethod, args: &[f64], w: f64, h: f64) -> Result<Self, Error> {
        use DistortMethod as M;
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
                let (mut cx, mut cy) = (w / 2.0, h / 2.0);
                let (mut sx, mut sy) = (1.0, 1.0);
                let angle;
                let (mut nx, mut ny);
                match args.len() {
                    1 => angle = args[0],
                    2 => {
                        sx = args[0];
                        sy = args[0];
                        angle = args[1];
                    }
                    3 => {
                        sx = args[0];
                        sy = args[1];
                        angle = args[2];
                    }
                    4 => {
                        cx = args[0];
                        cy = args[1];
                        sx = args[2];
                        sy = args[2];
                        angle = args[3];
                    }
                    5 => {
                        cx = args[0];
                        cy = args[1];
                        sx = args[2];
                        sy = args[3];
                        angle = args[4];
                    }
                    6 => {
                        cx = args[0];
                        cy = args[1];
                        sx = args[2];
                        sy = args[2];
                        angle = args[3];
                    }
                    7 => {
                        cx = args[0];
                        cy = args[1];
                        sx = args[2];
                        sy = args[3];
                        angle = args[4];
                    }
                    _ => return Err(bad("SRT", "takes 1 to 7 values")),
                }
                nx = cx;
                ny = cy;
                if args.len() == 6 {
                    nx = args[4];
                    ny = args[5];
                } else if args.len() == 7 {
                    nx = args[5];
                    ny = args[6];
                }
                let (s, c) = angle.to_radians().sin_cos();
                // Forward: x = sx c (u−cx) − sy s (v−cy) + nx …
                let fwd = [sx * c, -sy * s, 0.0, sx * s, sy * c, 0.0];
                let fwd = [
                    fwd[0],
                    fwd[1],
                    nx - fwd[0] * cx - fwd[1] * cy,
                    fwd[3],
                    fwd[4],
                    ny - fwd[3] * cx - fwd[4] * cy,
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
                // Destination → source directly.
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
                if !(1.0..=5.0).contains(&order) {
                    return Err(bad("Polynomial", "order must be 1 … 5"));
                }
                let pts = points(args, skip)?;
                let rows: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|&(_, _, x, y)| monomials(order, x, y))
                    .collect();
                if rows.len() < rows[0].len() {
                    return Err(bad("Polynomial", "too few control points for the order"));
                }
                let us: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let vs: Vec<f64> = pts.iter().map(|p| p.1).collect();
                Mapping::Poly {
                    order,
                    cu: lstsq(&rows, &us).ok_or_else(|| bad("Polynomial", "degenerate points"))?,
                    cv: lstsq(&rows, &vs).ok_or_else(|| bad("Polynomial", "degenerate points"))?,
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
                Mapping::Arc {
                    w,
                    h,
                    angle0: -90.0 - arc / 2.0 + rot,
                    arc,
                    r_top,
                    r_bot,
                    out_cx: 0.0,
                    out_cy: 0.0,
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
                power: 2.0,
            },
        })
    }

    fn forces_bestfit(&self) -> bool {
        matches!(self, Mapping::Arc { .. })
    }

    /// Destination centre → source position; `None` where undefined.
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
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
            Mapping::Poly { order, cu, cv } => {
                let mono = monomials(*order, x, y);
                Some((
                    mono.iter().zip(cu).map(|(a, b)| a * b).sum(),
                    mono.iter().zip(cv).map(|(a, b)| a * b).sum(),
                ))
            }
            Mapping::BilinearForward { cx, cy } => {
                // Newton on x(u,v) = target.
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
                Some((u, v))
            }
            Mapping::Arc {
                w,
                h,
                angle0,
                arc,
                r_top,
                r_bot,
                out_cx,
                out_cy,
            } => {
                let (dx, dy) = (x - out_cx, y - out_cy);
                let r = (dx * dx + dy * dy).sqrt();
                let mut a = dy.atan2(dx).to_degrees() - angle0;
                a = a.rem_euclid(360.0);
                let u = a / arc * w;
                let v = (r_top - r) / (r_top - r_bot) * h;
                Some((u, v))
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
                    let mut ang = dx.atan2(dy).to_degrees();
                    while ang < *a0 {
                        ang += 360.0;
                    }
                    while ang >= *a0 + 360.0 {
                        ang -= 360.0;
                    }
                    let u = (ang - a0) / (a1 - a0) * w;
                    let v = (r - r_min) / (r_max - r_min) * h;
                    Some((u, v))
                }
            }
            Mapping::Barrel {
                coef,
                cx,
                cy,
                norm,
                inverse,
            } => {
                let (dx, dy) = ((x - cx) / norm, (y - cy) / norm);
                let r = (dx * dx + dy * dy).sqrt();
                let f = coef[0] * r * r * r + coef[1] * r * r + coef[2] * r + coef[3];
                let s = if *inverse {
                    if f.abs() < 1e-12 {
                        return None;
                    }
                    1.0 / f
                } else {
                    f
                };
                Some((cx + dx * s * norm, cy + dy * s * norm))
            }
            Mapping::Shepards { pts, power } => {
                let (mut su, mut sv, mut sw) = (0.0, 0.0, 0.0);
                for &(u, v, px, py) in pts {
                    let d2 = (x - px) * (x - px) + (y - py) * (y - py);
                    if d2 < 1e-12 {
                        return Some((u, v));
                    }
                    let wgt = 1.0 / d2.powf(power / 2.0);
                    su += wgt * (u - px);
                    sv += wgt * (v - py);
                    sw += wgt;
                }
                Some((x + su / sw, y + sv / sw))
            }
        }
    }

    /// Numeric Jacobian of the inverse map at `(x, y)`: `[[du/dx,
    /// du/dy], [dv/dx, dv/dy]]`.
    fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let h = 0.25;
        let f = |x, y| self.inverse(x, y).unwrap_or((x, y));
        let (ux1, vx1) = f(x + h, y);
        let (ux0, vx0) = f(x - h, y);
        let (uy1, vy1) = f(x, y + h);
        let (uy0, vy0) = f(x, y - h);
        let mut du_dx = (ux1 - ux0) / (2.0 * h);
        // Angular methods wrap: undo a 360° jump in u.
        if du_dx.abs() > 1e4 {
            du_dx = 0.0;
        }
        [
            [du_dx, (uy1 - uy0) / (2.0 * h)],
            [(vx1 - vx0) / (2.0 * h), (vy1 - vy0) / (2.0 * h)],
        ]
    }

    /// The `+distort` output window `(w, h, x, y)` for the methods with a
    /// forward map: the distorted corners' bounding box widened to whole
    /// pixels as `floor(min − ½) … ceil(max + ½)` (measured).
    fn bestfit(&self, w: f64, h: f64) -> Option<(u32, u32, i64, i64)> {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        match self {
            Mapping::Affine(m) => {
                let f = invert_affine(*m)?;
                for (u, v) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
                    pts.push((f[0] * u + f[1] * v + f[2], f[3] * u + f[4] * v + f[5]));
                }
            }
            Mapping::Perspective(m) => {
                let f = invert3(*m)?;
                for (u, v) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
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
        Some((
            ((fx1 - fx0) as u32).max(1),
            ((fy1 - fy0) as u32).max(1),
            fx0 as i64,
            fy0 as i64,
        ))
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

    #[test]
    fn virtual_pixels_tile_and_mirror() {
        let img = ramp(4, 2);
        let s = Sampler {
            img: &img,
            vp: VirtualPixel::Tile,
            bg: [0.0; 4],
        };
        assert_eq!(s.at(4, 0), s.at(0, 0));
        assert_eq!(s.at(-1, -1), s.at(3, 1));
        let s = Sampler {
            img: &img,
            vp: VirtualPixel::Mirror,
            bg: [0.0; 4],
        };
        assert_eq!(s.at(4, 0), s.at(3, 0));
        assert_eq!(s.at(-1, 0), s.at(0, 0));
        let s = Sampler {
            img: &img,
            vp: VirtualPixel::Transparent,
            bg: [0.0; 4],
        };
        assert_eq!(s.at(-1, 0), [0.0; 4]);
    }

    #[test]
    fn perspective_from_points_reproduces_the_points() {
        let pts = vec![
            0.0, 0.0, 2.0, 1.0, 10.0, 0.0, 9.0, 0.0, 10.0, 10.0, 10.0, 9.0, 0.0, 10.0, 1.0, 10.0,
        ];
        let m = Mapping::build(DistortMethod::Perspective, &pts, 10.0, 10.0).unwrap();
        for c in pts.as_chunks::<4>().0 {
            let (u, v) = m.inverse(c[2], c[3]).unwrap();
            assert!((u - c[0]).abs() < 1e-6 && (v - c[1]).abs() < 1e-6);
        }
    }
}
