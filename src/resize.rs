//! Rescale a frame to arbitrary dimensions.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// How samples are reconstructed during rescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest-neighbour — fast, blocky. Good for pixel art.
    Nearest,
    /// Bilinear — smooth, default for natural images.
    #[default]
    Bilinear,
    /// Bicubic — separable 4-tap cubic convolution using the uniform
    /// Catmull-Rom cubic of `docs/image/filter/curve-interpolation.md`
    /// §3.1 (`m_i = (p_{i+1} − p_{i−1}) / 2`). Sharper than bilinear on
    /// upscale (it preserves more high-frequency detail because the
    /// interpolating cubic has a steeper transition band) and can ring
    /// slightly at hard edges (the cubic's negative side-lobes), which
    /// is the standard quality/overshoot trade for cubic reconstruction.
    /// Border taps clamp to the nearest in-bounds sample.
    Bicubic,
    /// Area-average ("box") resampling — every output pixel is the mean
    /// of the source pixels its footprint covers. This is the correct
    /// **downscale** kernel: a point-sampling kernel (nearest / bilinear /
    /// bicubic) reads only a few source taps per output pixel and so
    /// aliases when the scale factor drops below ~0.5, whereas the area
    /// average integrates the *entire* shrinking footprint and is
    /// alias-free by construction. The footprint is the half-open source
    /// interval `[x·sx, (x+1)·sx)` weighted by fractional pixel coverage
    /// at the two ragged ends, so non-integer ratios are handled exactly.
    /// On upscale (footprint < 1 px) it degenerates to nearest-neighbour.
    Area,
    /// Lanczos windowed-sinc resampling with lobe count `a` (`a = 2` and
    /// `a = 3` are the usual choices; `a = 3` is the sharper default).
    ///
    /// The ideal band-limited reconstruction kernel is the normalised
    /// cardinal sine `sinc(x) = sin(πx)/(πx)`, whose infinite support is
    /// impractical. Lanczos truncates it with a second, wider sinc window
    /// so the kernel has compact support `[−a, a]`:
    ///
    /// ```text
    ///   L(x) = sinc(x) · sinc(x / a)     for |x| < a
    ///        = 0                          otherwise
    /// ```
    ///
    /// `L(0) = 1` and `L(k) = 0` at every non-zero integer `k`, so the
    /// kernel *interpolates* the samples (an unscaled unit-scale pass
    /// reproduces the input exactly at integer positions). Unlike the
    /// point-sampling kernels above, this variant runs through the general
    /// separable driver that **scales the kernel by the resampling ratio on
    /// downscale** (`weight(x) = L(x / s) / s` for shrink factor `s > 1`),
    /// so the reconstruction filter doubles as the anti-alias low-pass and
    /// is alias-free below 0.5× the way [`Interpolation::Area`] is — while
    /// keeping the sharper windowed-sinc frequency response on upscale.
    /// Border taps clamp to the nearest in-bounds sample; the per-output
    /// weights are renormalised to sum to 1 so flat input reproduces
    /// exactly even against a clamped edge.
    Lanczos {
        /// Number of sinc lobes on each side of the centre (support
        /// radius). Clamped to `1..=8`; `0` is treated as `1`.
        a: u8,
    },
    /// Separable cubic convolution from the two-parameter Mitchell–Netravali
    /// family `BC(B, C)`. `Mitchell = BC(1/3, 1/3)` is the balanced default
    /// (a good blur/ring compromise for photographic resampling); the
    /// uniform Catmull–Rom of [`Interpolation::Bicubic`] is `BC(0, 1/2)`.
    ///
    /// The piecewise cubic (support `[−2, 2]`) is
    ///
    /// ```text
    ///   6·k(x) = (12−9B−6C)|x|³ + (−18+12B+6C)|x|²           + (6−2B)       , |x|<1
    ///          = (−B−6C)|x|³    + (6B+30C)|x|²   + (−12B−48C)|x| + (8B+24C)  , 1≤|x|<2
    ///          = 0                                                          , |x|≥2
    /// ```
    ///
    /// with `B = C = 1/3`. Runs through the same anti-aliasing separable
    /// driver as [`Interpolation::Lanczos`] (kernel scaled by the shrink
    /// ratio on downscale), so it low-passes correctly when shrinking.
    Mitchell,
    /// Separable cubic B-spline reconstruction — the `BC(1, 0)` member of
    /// the Mitchell–Netravali family (see [`Interpolation::Mitchell`] for
    /// the shared piecewise form). This kernel is everywhere non-negative,
    /// so it never rings or overshoots, at the cost of being the softest of
    /// the cubics: `k(0) = 2/3`, `k(±1) = 1/6`, so it does **not**
    /// interpolate the samples (it approximates them, blurring slightly).
    /// The right choice when smoothness matters more than edge sharpness.
    /// Runs through the anti-aliasing separable driver.
    BSpline,
}

/// Rescale to `target_width × target_height`.
#[derive(Clone, Debug)]
pub struct Resize {
    target_width: u32,
    target_height: u32,
    interp: Interpolation,
}

impl Resize {
    /// Create a resize filter targeting the given output dimensions.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width: target_width.max(1),
            target_height: target_height.max(1),
            interp: Interpolation::default(),
        }
    }

    /// Choose the interpolation kernel.
    pub fn with_interpolation(mut self, interp: Interpolation) -> Self {
        self.interp = interp;
        self
    }
}

impl ImageFilter for Resize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Resize does not yet handle {:?}",
                params.format
            )));
        }

        let (cx, cy) = crate::blur::chroma_subsampling(params.format);
        let new_w = self.target_width;
        let new_h = self.target_height;

        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (src_w, src_h) =
                crate::blur::plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let (dst_w, dst_h) = crate::blur::plane_dims(new_w, new_h, params.format, idx, cx, cy);
            let bpp = crate::blur::bytes_per_plane_pixel(params.format, idx);
            new_planes.push(resize_plane(
                plane,
                src_w,
                src_h,
                dst_w,
                dst_h,
                bpp,
                self.interp,
            ));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

fn resize_plane(
    src: &VideoPlane,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    bpp: usize,
    interp: Interpolation,
) -> VideoPlane {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut out = vec![0u8; dw * dh * bpp];

    // Scale factors (source units per destination unit). Use a half-pixel
    // offset so the output samples centre on input pixels — prevents the
    // top-left bias that integer-math resize often shows.
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    // Area-average resampling integrates a whole source footprint per
    // output pixel rather than point-sampling a fixed tap stencil, so it
    // gets its own separable two-pass driver below.
    if interp == Interpolation::Area {
        return resize_plane_area(src, sw, sh, dw, dh, bpp);
    }

    // The windowed-sinc / parametric-cubic family runs through a general
    // separable weighted-tap driver that scales its kernel by the shrink
    // ratio on downscale (so the reconstruction filter doubles as the
    // anti-alias low-pass) and renormalises the per-output weights.
    if let Some(kernel) = ResampleKernel::from_interpolation(interp) {
        return resize_plane_kernel(src, sw, sh, dw, dh, bpp, kernel);
    }

    for y in 0..dh {
        let sy = (y as f32 + 0.5) * scale_y - 0.5;
        for x in 0..dw {
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            match interp {
                Interpolation::Nearest => {
                    let nx = sx.round().clamp(0.0, (sw - 1) as f32) as usize;
                    let ny = sy.round().clamp(0.0, (sh - 1) as f32) as usize;
                    let src_row_off = ny * src.stride + nx * bpp;
                    let dst_off = y * dw * bpp + x * bpp;
                    out[dst_off..dst_off + bpp]
                        .copy_from_slice(&src.data[src_row_off..src_row_off + bpp]);
                }
                Interpolation::Bilinear => {
                    let x0 = sx.floor().clamp(0.0, (sw - 1) as f32) as usize;
                    let y0 = sy.floor().clamp(0.0, (sh - 1) as f32) as usize;
                    let x1 = (x0 + 1).min(sw - 1);
                    let y1 = (y0 + 1).min(sh - 1);
                    let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                    let fy = (sy - y0 as f32).clamp(0.0, 1.0);

                    for ch in 0..bpp {
                        let p00 = src.data[y0 * src.stride + x0 * bpp + ch] as f32;
                        let p10 = src.data[y0 * src.stride + x1 * bpp + ch] as f32;
                        let p01 = src.data[y1 * src.stride + x0 * bpp + ch] as f32;
                        let p11 = src.data[y1 * src.stride + x1 * bpp + ch] as f32;
                        let v = p00 * (1.0 - fx) * (1.0 - fy)
                            + p10 * fx * (1.0 - fy)
                            + p01 * (1.0 - fx) * fy
                            + p11 * fx * fy;
                        out[y * dw * bpp + x * bpp + ch] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
                Interpolation::Bicubic => {
                    // Separable 4-tap cubic convolution. The horizontal
                    // weights `wx` and vertical weights `wy` are the
                    // uniform Catmull-Rom cubic of
                    // curve-interpolation.md §3.1, evaluated at the
                    // fractional source position. We sum over the 4×4
                    // tap neighbourhood `(ix0-1 .. ix0+2, iy0-1 .. iy0+2)`.
                    let ix0 = sx.floor();
                    let iy0 = sy.floor();
                    let fx = sx - ix0;
                    let fy = sy - iy0;
                    let wx = catmull_rom_weights(fx);
                    let wy = catmull_rom_weights(fy);
                    let bx = ix0 as isize;
                    let by = iy0 as isize;
                    for ch in 0..bpp {
                        let mut acc = 0.0f32;
                        for (j, &wyj) in wy.iter().enumerate() {
                            let syy = clamp_coord(by - 1 + j as isize, sh);
                            let row = syy * src.stride;
                            let mut rowacc = 0.0f32;
                            for (i, &wxi) in wx.iter().enumerate() {
                                let sxx = clamp_coord(bx - 1 + i as isize, sw);
                                rowacc += wxi * src.data[row + sxx * bpp + ch] as f32;
                            }
                            acc += wyj * rowacc;
                        }
                        out[y * dw * bpp + x * bpp + ch] = acc.round().clamp(0.0, 255.0) as u8;
                    }
                }
                Interpolation::Area
                | Interpolation::Lanczos { .. }
                | Interpolation::Mitchell
                | Interpolation::BSpline => {
                    unreachable!("handled by a dedicated separable driver above")
                }
            }
        }
    }

    VideoPlane {
        stride: dw * bpp,
        data: out,
    }
}

/// Clamp an integer source coordinate into `[0, extent − 1]` (replicate
/// the border sample for out-of-bounds taps).
#[inline]
fn clamp_coord(c: isize, extent: usize) -> usize {
    c.clamp(0, extent as isize - 1) as usize
}

/// The four uniform Catmull-Rom cubic weights for a fractional offset
/// `t ∈ [0, 1)` between the second and third of four equally-spaced
/// taps `p_{-1}, p_0, p_1, p_2`.
///
/// Derived directly from the §3.1 segment matrix of
/// `docs/image/filter/curve-interpolation.md`: `P(t) = 0.5 · [1 t t² t³] ·
/// M · [p_{-1} p_0 p_1 p_2]ᵀ` with
///
/// ```text
///       |  0   2   0   0 |
///   M = | -1   0   1   0 |
///       |  2  -5   4  -1 |
///       | -1   3  -3   1 |
/// ```
///
/// Collapsing the row vector against each column gives the per-tap
/// weights below; they sum to 1 for every `t` (partition of unity), so a
/// flat input reproduces exactly.
#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

/// Separable area-average ("box") resampling. Each axis is integrated
/// independently: a horizontal pass collapses the source columns into
/// `dw` averaged columns, then a vertical pass collapses the rows into
/// `dh` averaged rows. Each output sample is the coverage-weighted mean
/// of the source samples its footprint `[k·scale, (k+1)·scale)` spans,
/// with fractional weights at the two ragged ends so non-integer ratios
/// are exact. On upscale (footprint < 1 px) the single covered sample
/// carries full weight, degenerating to nearest-neighbour.
fn resize_plane_area(
    src: &VideoPlane,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    bpp: usize,
) -> VideoPlane {
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    // Pass 1: horizontal — produce a (dw × sh) intermediate in f32.
    let mut mid = vec![0.0f32; dw * sh * bpp];
    for x in 0..dw {
        let (lo, hi) = footprint(x, scale_x, sw);
        for sy in 0..sh {
            let row = sy * src.stride;
            for ch in 0..bpp {
                let v = integrate_axis(lo, hi, |sx| src.data[row + sx * bpp + ch] as f32);
                mid[(sy * dw + x) * bpp + ch] = v;
            }
        }
    }

    // Pass 2: vertical — collapse the (dw × sh) intermediate to (dw × dh).
    let mut out = vec![0u8; dw * dh * bpp];
    for y in 0..dh {
        let (lo, hi) = footprint(y, scale_y, sh);
        for x in 0..dw {
            for ch in 0..bpp {
                let v = integrate_axis(lo, hi, |sy| mid[(sy * dw + x) * bpp + ch]);
                out[(y * dw + x) * bpp + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    VideoPlane {
        stride: dw * bpp,
        data: out,
    }
}

/// A separable reconstruction kernel `φ(x)` with compact support radius
/// `support` (in source-pixel units at unit scale). The general driver
/// evaluates `φ` per tap; downscale widening and weight normalisation are
/// handled by the driver, not the kernel.
#[derive(Clone, Copy)]
struct ResampleKernel {
    /// Half-width of the support in unit-scale source pixels (`φ(x) = 0`
    /// for `|x| ≥ support`).
    support: f32,
    /// The kernel function evaluated at signed distance `x` (source-pixel
    /// units, unit scale).
    phi: fn(f32) -> f32,
}

impl ResampleKernel {
    /// Map the public [`Interpolation`] selector onto a driver kernel, or
    /// `None` for the variants handled by their own specialised path
    /// (Nearest / Bilinear / Bicubic / Area).
    fn from_interpolation(interp: Interpolation) -> Option<Self> {
        match interp {
            Interpolation::Lanczos { a } => {
                let a = a.clamp(1, 8);
                // A `fn` pointer cannot close over `a`, so route each
                // supported lobe count to its own fixed-`a` evaluator.
                Some(Self {
                    support: a as f32,
                    phi: match a {
                        1 => |x| lanczos(x, 1.0),
                        2 => |x| lanczos(x, 2.0),
                        3 => |x| lanczos(x, 3.0),
                        4 => |x| lanczos(x, 4.0),
                        5 => |x| lanczos(x, 5.0),
                        6 => |x| lanczos(x, 6.0),
                        7 => |x| lanczos(x, 7.0),
                        _ => |x| lanczos(x, 8.0),
                    },
                })
            }
            // Mitchell–Netravali BC(1/3, 1/3).
            Interpolation::Mitchell => Some(Self {
                support: 2.0,
                phi: |x| mitchell_netravali(x, 1.0 / 3.0, 1.0 / 3.0),
            }),
            // Cubic B-spline BC(1, 0).
            Interpolation::BSpline => Some(Self {
                support: 2.0,
                phi: |x| mitchell_netravali(x, 1.0, 0.0),
            }),
            _ => None,
        }
    }
}

/// Normalised cardinal sine `sinc(x) = sin(πx)/(πx)`, with the removable
/// singularity `sinc(0) = 1` filled analytically.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

/// Lanczos windowed sinc with `a` lobes: `L(x) = sinc(x)·sinc(x/a)` on
/// `|x| < a`, zero outside. `L(0) = 1`, `L(k) = 0` at non-zero integers.
#[inline]
fn lanczos(x: f32, a: f32) -> f32 {
    if x.abs() >= a {
        0.0
    } else {
        sinc(x) * sinc(x / a)
    }
}

/// The two-parameter Mitchell–Netravali cubic `BC(B, C)` (support `[−2, 2]`).
///
/// Derived from the piecewise cubic that is `C¹` at `|x| = 1` and `|x| = 2`
/// and reproduces constants (partition of unity):
///
/// ```text
///   6·k(x) = (12−9B−6C)|x|³ + (−18+12B+6C)|x|²          + (6−2B)      , |x|<1
///          = (−B−6C)|x|³    + (6B+30C)|x|²  + (−12B−48C)|x| + (8B+24C) , 1≤|x|<2
///          = 0                                                        , |x|≥2
/// ```
///
/// `BC(0, 1/2)` is uniform Catmull–Rom, `BC(1/3, 1/3)` the balanced
/// Mitchell kernel, `BC(1, 0)` the everywhere-non-negative cubic B-spline.
#[inline]
fn mitchell_netravali(x: f32, b: f32, c: f32) -> f32 {
    let ax = x.abs();
    let ax2 = ax * ax;
    let ax3 = ax2 * ax;
    let v = if ax < 1.0 {
        (12.0 - 9.0 * b - 6.0 * c) * ax3 + (-18.0 + 12.0 * b + 6.0 * c) * ax2 + (6.0 - 2.0 * b)
    } else if ax < 2.0 {
        (-b - 6.0 * c) * ax3
            + (6.0 * b + 30.0 * c) * ax2
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c)
    } else {
        0.0
    };
    v / 6.0
}

/// One axis of the general separable weighted-tap resampler: for each
/// output index `k` build the list of `(source_index, weight)` taps whose
/// kernel support overlaps `k`.
///
/// The kernel is centred on the continuous source position
/// `c = (k + 0.5)·scale − 0.5` (the same half-pixel-centred mapping the
/// point-sampling path uses). On **downscale** (`scale > 1`) the kernel is
/// stretched by `scale` — `weight ∝ φ((i − c)/scale)` over a support of
/// `support·scale` source pixels — so the reconstruction filter's cutoff
/// drops to the output Nyquist and the pass low-passes the input before
/// decimating (alias-free). On upscale (`scale ≤ 1`) the kernel keeps its
/// native unit width. Out-of-range taps clamp to the border sample and the
/// weights are renormalised to sum to 1, so a flat field always reproduces.
fn axis_contributions(dst_len: usize, src_len: usize, scale: f32, k: &ResampleKernel) -> Contribs {
    // Filter scale: widen on shrink, native on enlarge.
    let filt = scale.max(1.0);
    let radius = k.support * filt;
    let mut per_out: Vec<Vec<(usize, f32)>> = Vec::with_capacity(dst_len);
    for out in 0..dst_len {
        let centre = (out as f32 + 0.5) * scale - 0.5;
        let lo = (centre - radius).ceil() as isize;
        let hi = (centre + radius).floor() as isize;
        let mut taps: Vec<(usize, f32)> = Vec::new();
        let mut wsum = 0.0f32;
        let mut i = lo;
        while i <= hi {
            let w = (k.phi)((i as f32 - centre) / filt);
            if w != 0.0 {
                let si = i.clamp(0, src_len as isize - 1) as usize;
                // Accumulate onto the clamped index so border taps fold
                // their weight into the edge sample (partition-preserving).
                if let Some(last) = taps.last_mut().filter(|t| t.0 == si) {
                    last.1 += w;
                } else {
                    taps.push((si, w));
                }
                wsum += w;
            }
            i += 1;
        }
        if taps.is_empty() {
            // Degenerate (all weights zero — e.g. a 1-px source): sample the
            // nearest source pixel with unit weight.
            let si = centre.round().clamp(0.0, src_len as f32 - 1.0) as usize;
            taps.push((si, 1.0));
        } else if wsum != 0.0 {
            let inv = 1.0 / wsum;
            for t in &mut taps {
                t.1 *= inv;
            }
        }
        per_out.push(taps);
    }
    Contribs { per_out }
}

/// Precomputed per-output-index tap lists for one axis.
struct Contribs {
    per_out: Vec<Vec<(usize, f32)>>,
}

/// General separable weighted-kernel resampler shared by the windowed-sinc
/// (Lanczos) and parametric-cubic (Mitchell / B-spline) variants. A
/// horizontal pass collapses the source columns onto `dw` weighted columns
/// (f32 intermediate), then a vertical pass collapses the rows onto `dh`.
fn resize_plane_kernel(
    src: &VideoPlane,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    bpp: usize,
    kernel: ResampleKernel,
) -> VideoPlane {
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    let cx = axis_contributions(dw, sw, scale_x, &kernel);
    let cy = axis_contributions(dh, sh, scale_y, &kernel);

    // Pass 1: horizontal — (sw × sh) → (dw × sh) in f32.
    let mut mid = vec![0.0f32; dw * sh * bpp];
    for sy in 0..sh {
        let row = sy * src.stride;
        for (x, taps) in cx.per_out.iter().enumerate() {
            for ch in 0..bpp {
                let mut acc = 0.0f32;
                for &(sx, w) in taps {
                    acc += w * src.data[row + sx * bpp + ch] as f32;
                }
                mid[(sy * dw + x) * bpp + ch] = acc;
            }
        }
    }

    // Pass 2: vertical — (dw × sh) → (dw × dh) in u8.
    let mut out = vec![0u8; dw * dh * bpp];
    for (y, taps) in cy.per_out.iter().enumerate() {
        for x in 0..dw {
            for ch in 0..bpp {
                let mut acc = 0.0f32;
                for &(sy, w) in taps {
                    acc += w * mid[(sy * dw + x) * bpp + ch];
                }
                out[(y * dw + x) * bpp + ch] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    VideoPlane {
        stride: dw * bpp,
        data: out,
    }
}

/// The continuous source footprint `[lo, hi)` of output index `k` at the
/// given scale, clamped into `[0, extent]`. On upscale the footprint can
/// be narrower than one pixel; we widen it to a minimum of one source
/// pixel so the integral always has support (nearest-neighbour limit).
#[inline]
fn footprint(k: usize, scale: f32, extent: usize) -> (f32, f32) {
    let mut lo = k as f32 * scale;
    let mut hi = (k as f32 + 1.0) * scale;
    if hi - lo < 1.0 {
        // Upscale: centre a unit-wide window on the footprint midpoint.
        let centre = 0.5 * (lo + hi);
        lo = centre - 0.5;
        hi = centre + 0.5;
    }
    let ext = extent as f32;
    (lo.clamp(0.0, ext), hi.clamp(0.0, ext))
}

/// Coverage-weighted mean of `sample(i)` over the continuous interval
/// `[lo, hi)`: each whole source pixel `i` contributes the length of its
/// overlap with the interval, divided by the total interval length.
#[inline]
fn integrate_axis(lo: f32, hi: f32, sample: impl Fn(usize) -> f32) -> f32 {
    let total = hi - lo;
    if total <= 0.0 {
        return sample(lo.floor() as usize);
    }
    let first = lo.floor() as usize;
    let last = ((hi - 1e-6).floor() as usize).max(first);
    let mut acc = 0.0f32;
    for i in first..=last {
        let pl = (i as f32).max(lo);
        let pr = ((i + 1) as f32).min(hi);
        let w = (pr - pl).max(0.0);
        acc += w * sample(i);
    }
    acc / total
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    #[test]
    fn upscale_nearest_duplicates() {
        let input = gray(2, 2, |x, y| (y * 2 + x + 1) as u8 * 50); // 50, 100, 150, 200
        let out = Resize::new(4, 4)
            .with_interpolation(Interpolation::Nearest)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // Every 2×2 block should carry one source value.
        let v_tl = out.planes[0].data[0];
        let v_tr = out.planes[0].data[3];
        let v_bl = out.planes[0].data[4 * 3];
        let v_br = out.planes[0].data[4 * 3 + 3];
        assert!([50u8, 100, 150, 200].contains(&v_tl));
        assert!([50u8, 100, 150, 200].contains(&v_tr));
        assert!([50u8, 100, 150, 200].contains(&v_bl));
        assert!([50u8, 100, 150, 200].contains(&v_br));
    }

    #[test]
    fn downscale_bilinear_averages() {
        // 4×4 flat at 100 → 2×2 must also be flat at 100.
        let input = gray(4, 4, |_, _| 100);
        let out = Resize::new(2, 2)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 100);
        }
    }

    #[test]
    fn yuv420_resize_preserves_plane_layout() {
        let _w = 16u32;
        let _h = 16u32;
        let y: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let u = vec![77u8; 8 * 8];
        let v = vec![128u8; 8 * 8];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y,
                },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };
        let out = Resize::new(8, 8)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 3);
        // Y plane is 8×8 after resize, stride tight at 8.
        assert_eq!(out.planes[0].stride, 8);
        assert_eq!(out.planes[0].data.len(), 8 * 8);
        // Chroma planes are 4×4 after 4:2:0 subsampling of 8×8.
        assert_eq!(out.planes[1].stride, 4);
        assert_eq!(out.planes[1].data.len(), 4 * 4);
        assert_eq!(out.planes[2].data.len(), 4 * 4);
        // Flat chroma stays flat.
        for b in &out.planes[1].data {
            assert_eq!(*b, 77);
        }
        for b in &out.planes[2].data {
            assert_eq!(*b, 128);
        }
    }

    #[test]
    fn catmull_rom_weights_partition_of_unity() {
        // The four Catmull-Rom taps must sum to 1 for every fraction, and
        // at t = 0 the centre tap (p_0) must carry the whole weight (the
        // interpolating-spline property: it passes exactly through knots).
        for n in 0..=10 {
            let t = n as f32 / 10.0;
            let w = catmull_rom_weights(t);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum {sum} at t={t}");
        }
        let w0 = catmull_rom_weights(0.0);
        assert!((w0[0]).abs() < 1e-6);
        assert!((w0[1] - 1.0).abs() < 1e-6);
        assert!((w0[2]).abs() < 1e-6);
        assert!((w0[3]).abs() < 1e-6);
    }

    #[test]
    fn bicubic_flat_input_is_flat() {
        // A constant field must reproduce exactly under bicubic (partition
        // of unity + border clamp guarantee no overshoot on flat data).
        let input = gray(8, 8, |_, _| 137);
        let out = Resize::new(13, 5)
            .with_interpolation(Interpolation::Bicubic)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 137);
        }
    }

    #[test]
    fn bicubic_linear_ramp_stays_monotone() {
        // A horizontal ramp upscaled 2× must remain monotone non-decreasing
        // along each row — the cubic reconstructs the underlying line
        // without inverting it.
        let input = gray(4, 1, |x, _| (x * 60) as u8); // 0,60,120,180
        let out = Resize::new(8, 1)
            .with_interpolation(Interpolation::Bicubic)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        let row = &out.planes[0].data;
        for w in row.windows(2) {
            assert!(w[1] >= w[0], "non-monotone bicubic ramp: {row:?}");
        }
    }

    #[test]
    fn area_downscale_2x_is_block_mean() {
        // 4×4 with a known 2×2-block pattern → 2×2 area downscale must be
        // the exact mean of each 2×2 source block.
        // Blocks: TL=10, TR=20, BL=30, BR=40 (each filling a 2×2 quadrant).
        let input = gray(4, 4, |x, y| {
            let q = (y / 2) * 2 + (x / 2);
            ((q + 1) * 10) as u8
        });
        let out = Resize::new(2, 2)
            .with_interpolation(Interpolation::Area)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn area_non_integer_ratio_weights_fractional_coverage() {
        // 3 → 2 downscale: each output pixel spans 1.5 source pixels.
        // Output[0] footprint = [0, 1.5): pixel0 weight 1.0, pixel1 weight
        // 0.5 → (1.0·0 + 0.5·100)/1.5 = 33.33 → 33.
        // Output[1] footprint = [1.5, 3): pixel1 weight 0.5, pixel2 weight
        // 1.0 → (0.5·100 + 1.0·200)/1.5 = 166.67 → 167.
        let input = gray(3, 1, |x, _| (x * 100) as u8); // 0, 100, 200
        let out = Resize::new(2, 1)
            .with_interpolation(Interpolation::Area)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 3,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![33, 167]);
    }

    #[test]
    fn area_flat_input_is_flat() {
        let input = gray(7, 9, |_, _| 88);
        let out = Resize::new(3, 4)
            .with_interpolation(Interpolation::Area)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 7,
                    height: 9,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 88);
        }
    }

    #[test]
    fn area_rgb_channels_independent() {
        // Each channel must be averaged independently — no bleed.
        let mut data = Vec::new();
        for _ in 0..(4 * 4) {
            data.extend_from_slice(&[10, 20, 30]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4 * 3,
                data,
            }],
        };
        let out = Resize::new(2, 2)
            .with_interpolation(Interpolation::Area)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for px in out.planes[0].data.chunks_exact(3) {
            assert_eq!(px, [10, 20, 30]);
        }
    }

    #[test]
    fn area_yuv420_subsamples_each_plane() {
        // Area resize must drive every plane through the same subsampled
        // dst dimensions as the other kernels. A flat 16×16 4:2:0 frame
        // (Y=200, U=77, V=128) → 8×8 must stay flat with 4×4 chroma.
        let y = vec![200u8; 16 * 16];
        let u = vec![77u8; 8 * 8];
        let v = vec![128u8; 8 * 8];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y,
                },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };
        let out = Resize::new(8, 8)
            .with_interpolation(Interpolation::Area)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8);
        assert_eq!(out.planes[1].data.len(), 4 * 4);
        assert_eq!(out.planes[2].data.len(), 4 * 4);
        assert!(out.planes[0].data.iter().all(|&b| b == 200));
        assert!(out.planes[1].data.iter().all(|&b| b == 77));
        assert!(out.planes[2].data.iter().all(|&b| b == 128));
    }

    /// Sum-of-absolute-differences of a plane against its own mean — a
    /// cheap proxy for residual high-frequency energy / aliasing.
    fn dispersion(data: &[u8]) -> u32 {
        let mean = (data.iter().map(|&b| b as u32).sum::<u32>() / data.len() as u32) as i32;
        data.iter().map(|&b| (b as i32 - mean).unsigned_abs()).sum()
    }

    #[test]
    fn area_downscale_is_less_aliased_than_nearest() {
        // 1-px vertical stripes are pure Nyquist-rate horizontal detail.
        // Downscaling 8→3 with nearest point-samples individual stripes, so
        // the output keeps full 0/255 contrast (aliasing — the stripe phase
        // shows through). The area average integrates each ~2.67-px-wide
        // footprint, which always spans both a black and a white stripe, so
        // it collapses to near-uniform mid-grey. Area's residual dispersion
        // must therefore be strictly lower.
        let stripes = gray(8, 1, |x, _| if x % 2 == 0 { 0 } else { 255 });
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 8,
            height: 1,
        };
        let near = Resize::new(3, 1)
            .with_interpolation(Interpolation::Nearest)
            .apply(&stripes, params)
            .unwrap();
        let area = Resize::new(3, 1)
            .with_interpolation(Interpolation::Area)
            .apply(&stripes, params)
            .unwrap();
        assert!(
            dispersion(&near.planes[0].data) > dispersion(&area.planes[0].data),
            "nearest ({}) should alias more than area ({})",
            dispersion(&near.planes[0].data),
            dispersion(&area.planes[0].data)
        );
    }

    // ---- Windowed-sinc (Lanczos) + parametric-cubic (Mitchell / B-spline)

    #[test]
    fn sinc_and_lanczos_analytic_properties() {
        // sinc(0) = 1, sinc(non-zero integer) = 0 — the interpolation
        // (cardinal) property the whole reconstruction rests on.
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
        for k in 1..=5 {
            assert!(sinc(k as f32).abs() < 1e-5, "sinc({k}) != 0");
            assert!(sinc(-(k as f32)).abs() < 1e-5);
        }
        // Lanczos inherits it and is exactly zero at/after the last lobe.
        for a in [1.0f32, 2.0, 3.0] {
            assert!((lanczos(0.0, a) - 1.0).abs() < 1e-6);
            assert_eq!(lanczos(a, a), 0.0);
            assert_eq!(lanczos(a + 0.5, a), 0.0);
            // Even symmetry.
            assert!((lanczos(0.37, a) - lanczos(-0.37, a)).abs() < 1e-6);
        }
    }

    #[test]
    fn mitchell_family_values_and_partition_of_unity() {
        // Cubic B-spline BC(1,0): k(0)=2/3, k(±1)=1/6, everywhere ≥ 0.
        assert!((mitchell_netravali(0.0, 1.0, 0.0) - 2.0 / 3.0).abs() < 1e-6);
        assert!((mitchell_netravali(1.0, 1.0, 0.0) - 1.0 / 6.0).abs() < 1e-6);
        for n in -40..=40 {
            let x = n as f32 / 10.0;
            assert!(
                mitchell_netravali(x, 1.0, 0.0) >= -1e-6,
                "B-spline negative at {x}"
            );
        }
        // Mitchell BC(1/3,1/3): k(0) = (6-2B)/6 = 8/9.
        let b = 1.0 / 3.0;
        assert!((mitchell_netravali(0.0, b, b) - 8.0 / 9.0).abs() < 1e-6);
        // Any interpolating/approximating kernel with support 2 reproduces a
        // constant only if its integer samples partition unity: for the
        // continuous kernel the three integer taps k(f-1)+k(f)+k(f+1)+k(f+2)
        // (over a unit cell) sum to 1 for every fractional offset f.
        for (bb, cc) in [(0.0, 0.5), (1.0 / 3.0, 1.0 / 3.0), (1.0, 0.0)] {
            for n in 0..=10 {
                let f = n as f32 / 10.0;
                let sum = mitchell_netravali(f + 1.0, bb, cc)
                    + mitchell_netravali(f, bb, cc)
                    + mitchell_netravali(f - 1.0, bb, cc)
                    + mitchell_netravali(f - 2.0, bb, cc);
                assert!((sum - 1.0).abs() < 1e-5, "BC({bb},{cc}) sum {sum} at f={f}");
            }
        }
    }

    #[test]
    fn lanczos_interpolates_at_integer_upscale() {
        // At an exact 1:1 "resize" Lanczos must reproduce the input (the
        // kernel is zero at every non-zero integer offset, unit at zero).
        let input = gray(6, 1, |x, _| (x * 40) as u8);
        let out = Resize::new(6, 1)
            .with_interpolation(Interpolation::Lanczos { a: 3 })
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 6,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn kernel_variants_flat_input_is_flat() {
        for interp in [
            Interpolation::Lanczos { a: 2 },
            Interpolation::Lanczos { a: 3 },
            Interpolation::Mitchell,
            Interpolation::BSpline,
        ] {
            let input = gray(9, 7, |_, _| 173);
            let out = Resize::new(4, 12)
                .with_interpolation(interp)
                .apply(
                    &input,
                    VideoStreamParams {
                        format: PixelFormat::Gray8,
                        width: 9,
                        height: 7,
                    },
                )
                .unwrap();
            for b in &out.planes[0].data {
                assert_eq!(*b, 173, "{interp:?} not flat");
            }
        }
    }

    #[test]
    fn lanczos_downscale_is_less_aliased_than_nearest() {
        // Same Nyquist-stripe aliasing probe as the Area test: the scaled
        // (widened) sinc integrates several stripes on downscale, so its
        // residual dispersion must be strictly below nearest's.
        let stripes = gray(8, 1, |x, _| if x % 2 == 0 { 0 } else { 255 });
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 8,
            height: 1,
        };
        let near = Resize::new(3, 1)
            .with_interpolation(Interpolation::Nearest)
            .apply(&stripes, params)
            .unwrap();
        let lanc = Resize::new(3, 1)
            .with_interpolation(Interpolation::Lanczos { a: 3 })
            .apply(&stripes, params)
            .unwrap();
        assert!(
            dispersion(&near.planes[0].data) > dispersion(&lanc.planes[0].data),
            "nearest ({}) should alias more than lanczos ({})",
            dispersion(&near.planes[0].data),
            dispersion(&lanc.planes[0].data)
        );
    }

    #[test]
    fn lanczos_upscale_sharper_than_bspline() {
        // On a centre spike, the windowed-sinc keeps a higher peak than the
        // everywhere-positive B-spline (which is the softest cubic).
        let mut data = vec![0u8; 5 * 5];
        data[2 * 5 + 2] = 255;
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 5, data }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 5,
            height: 5,
        };
        let peak = |interp| {
            let o = Resize::new(15, 15)
                .with_interpolation(interp)
                .apply(&input, params)
                .unwrap();
            *o.planes[0].data.iter().max().unwrap()
        };
        assert!(
            peak(Interpolation::Lanczos { a: 3 }) > peak(Interpolation::BSpline),
            "lanczos should out-peak the softer b-spline"
        );
    }

    #[test]
    fn bspline_step_stays_monotone_no_ringing() {
        // Because the B-spline kernel is everywhere non-negative, every
        // output sample is a *convex combination* of source samples — so a
        // monotone (step) input upscales to a monotone-non-decreasing output
        // with no ringing dips/overshoots. The interpolating Lanczos, by
        // contrast, has negative side-lobes and *does* ring at a hard step.
        let input = gray(6, 1, |x, _| if x < 3 { 40 } else { 200 });
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 6,
            height: 1,
        };
        let bs = Resize::new(24, 1)
            .with_interpolation(Interpolation::BSpline)
            .apply(&input, params)
            .unwrap();
        let row = &bs.planes[0].data;
        for w in row.windows(2) {
            assert!(w[1] >= w[0], "b-spline step not monotone: {row:?}");
        }
        // Every B-spline sample stays inside the [40, 200] source envelope.
        assert!(row.iter().all(|&b| (40..=200).contains(&b)));
        // Lanczos should ring: at least one sample escapes the envelope.
        let lz = Resize::new(24, 1)
            .with_interpolation(Interpolation::Lanczos { a: 3 })
            .apply(&input, params)
            .unwrap();
        assert!(
            lz.planes[0].data.iter().any(|&b| !(40..=200).contains(&b)),
            "lanczos should overshoot the step envelope"
        );
    }

    #[test]
    fn kernel_variants_yuv420_subsample_each_plane() {
        for interp in [Interpolation::Lanczos { a: 3 }, Interpolation::Mitchell] {
            let y = vec![200u8; 16 * 16];
            let u = vec![77u8; 8 * 8];
            let v = vec![128u8; 8 * 8];
            let input = VideoFrame {
                pts: None,
                planes: vec![
                    VideoPlane {
                        stride: 16,
                        data: y,
                    },
                    VideoPlane { stride: 8, data: u },
                    VideoPlane { stride: 8, data: v },
                ],
            };
            let out = Resize::new(8, 8)
                .with_interpolation(interp)
                .apply(
                    &input,
                    VideoStreamParams {
                        format: PixelFormat::Yuv420P,
                        width: 16,
                        height: 16,
                    },
                )
                .unwrap();
            assert_eq!(out.planes[0].data.len(), 8 * 8, "{interp:?}");
            assert_eq!(out.planes[1].data.len(), 4 * 4);
            assert_eq!(out.planes[2].data.len(), 4 * 4);
            assert!(out.planes[0].data.iter().all(|&b| b == 200));
            assert!(out.planes[1].data.iter().all(|&b| b == 77));
            assert!(out.planes[2].data.iter().all(|&b| b == 128));
        }
    }

    #[test]
    fn kernel_variants_rgb_channels_independent() {
        for interp in [Interpolation::Lanczos { a: 2 }, Interpolation::BSpline] {
            let mut data = Vec::new();
            for _ in 0..(4 * 4) {
                data.extend_from_slice(&[10, 20, 30]);
            }
            let input = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: 4 * 3,
                    data,
                }],
            };
            let out = Resize::new(7, 3)
                .with_interpolation(interp)
                .apply(
                    &input,
                    VideoStreamParams {
                        format: PixelFormat::Rgb24,
                        width: 4,
                        height: 4,
                    },
                )
                .unwrap();
            for px in out.planes[0].data.chunks_exact(3) {
                assert_eq!(px, [10, 20, 30], "{interp:?} channel bleed");
            }
        }
    }

    #[test]
    fn lanczos_a_is_clamped_not_panicking() {
        // a = 0 folds to 1; absurd a folds to 8. Neither panics.
        let input = gray(4, 4, |x, y| (x + y) as u8 * 20);
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 4,
            height: 4,
        };
        for a in [0u8, 1, 8, 200] {
            let out = Resize::new(6, 6)
                .with_interpolation(Interpolation::Lanczos { a })
                .apply(&input, params)
                .unwrap();
            assert_eq!(out.planes[0].data.len(), 36);
        }
    }

    #[test]
    fn bicubic_upscale_is_sharper_than_bilinear() {
        // On a single bright pixel surrounded by black, an upscaled
        // bicubic reconstruction keeps a higher peak than bilinear because
        // the cubic has a narrower main lobe (it concentrates more energy
        // at the centre). Compare the brightest output sample.
        let mut data = vec![0u8; 5 * 5];
        data[2 * 5 + 2] = 255; // centre spike
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 5, data }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 5,
            height: 5,
        };
        let bil = Resize::new(15, 15)
            .with_interpolation(Interpolation::Bilinear)
            .apply(&input, params)
            .unwrap();
        let bic = Resize::new(15, 15)
            .with_interpolation(Interpolation::Bicubic)
            .apply(&input, params)
            .unwrap();
        let peak = |d: &[u8]| *d.iter().max().unwrap();
        assert!(
            peak(&bic.planes[0].data) >= peak(&bil.planes[0].data),
            "bicubic peak {} should be >= bilinear peak {}",
            peak(&bic.planes[0].data),
            peak(&bil.planes[0].data)
        );
    }
}
