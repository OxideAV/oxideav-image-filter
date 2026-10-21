//! The wrapper every filter runs behind: layout gate, frame validation,
//! side-channel stripping and re-attachment, and the pointwise palette
//! route for `Pal8` pictures.
//!
//! [`crate::ImageFilter::apply`] on a [`crate::PixelFilter`] is exactly
//! [`apply_single`]; [`crate::TwoInputImageFilter::apply_two`] on a
//! [`crate::TwoInputPixelFilter`] is exactly [`apply_pair`].
//!
//! # The packed-RGB hop
//!
//! A filter whose native paths are packed 8-bit (`Rgba` / `Rgb24` /
//! `Gray8`) still takes every other **8-bit** layout `oxideav-pixfmt`
//! can convert — the planar and semi-planar YUV(A) families (limited or
//! full range, the frame's colour signal honoured), `Bgr24` / `Bgra` /
//! `Argb` / `Abgr`, `Ya8`, `Yuyv422` / `Uyvy422`, `Cmyk`, bit-packed
//! mono, and `Pal8` for a non-pointwise filter: the picture is
//! converted to the first of `Rgba`, `Rgb24`, `Gray8` the filter takes,
//! filtered, and — when the filter keeps that layout and the way back
//! exists — converted back to the source layout, so the output label
//! is the input label. A filter that changes layout (an edge detector
//! emitting `Gray8`) or a `Pal8` source (quantising back would be a
//! lossy decision, not a filter) leaves the output in the hop layout,
//! and the registry label says so. Deep layouts (10/12/14/16-bit words,
//! float) never take the hop: that would narrow samples silently, and
//! the filter refuses them with a typed error instead.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert_with, supports, ConvertContext, ConvertOptions, FrameInfo};

use crate::layout::SampleKind;
use crate::{layout, side_channels, PixelFilter, TwoInputPixelFilter, VideoStreamParams};

/// How a filter reaches a picture in `input` layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// The filter takes the layout as-is.
    Native,
    /// `Pal8` through a pointwise filter: the palette is filtered as a
    /// one-row `Rgba` picture and the index plane is untouched.
    Palette,
    /// Convert to `via`, filter, and convert back to the source layout
    /// when `back` (see the module docs).
    Hop { via: PixelFormat, back: bool },
}

/// The hop layouts in preference order: alpha-carrying first so a
/// `Bgra` / `Yuva420P` / `Ya8` source keeps its alpha, grey last.
const HOP_TARGETS: [PixelFormat; 3] = [PixelFormat::Rgba, PixelFormat::Rgb24, PixelFormat::Gray8];

/// The hop a filter takes for an 8-bit layout outside its native set,
/// if any.
fn hop_for<F: PixelFilter + ?Sized>(f: &F, fmt: PixelFormat) -> Option<Route> {
    // Only 8-bit storage hops: a 16-bit word or a float would be
    // narrowed on the way in.
    match layout::sample_kind(fmt) {
        Some(SampleKind::U8) | None => {}
        Some(SampleKind::U16 { .. }) | Some(SampleKind::F32) => return None,
    }
    let accepted = f.accepted_formats();
    let via = HOP_TARGETS
        .into_iter()
        .find(|t| accepted.contains(t) && supports(fmt, *t))?;
    let back = fmt != PixelFormat::Pal8 && f.output_format(via) == via && supports(via, fmt);
    Some(Route::Hop { via, back })
}

/// Short type name of a filter for error messages
/// (`oxideav_image_filter::sepia::Sepia` → `Sepia`).
pub(crate) fn short_name<F: ?Sized>() -> &'static str {
    let full = std::any::type_name::<F>();
    full.rsplit("::").next().unwrap_or(full)
}

fn refusal(name: &str, fmt: PixelFormat, accepted: &[PixelFormat]) -> Error {
    Error::unsupported(format!(
        "oxideav-image-filter: {name} does not take {fmt:?} (accepted: {accepted:?})"
    ))
}

/// The route a single-input filter takes for `fmt`, or the typed
/// refusal.
pub(crate) fn route_single<F: PixelFilter + ?Sized>(
    f: &F,
    fmt: PixelFormat,
) -> Result<Route, Error> {
    let accepted = f.accepted_formats();
    if accepted.contains(&fmt) {
        return Ok(Route::Native);
    }
    if fmt == PixelFormat::Pal8
        && f.is_pointwise()
        && accepted.contains(&PixelFormat::Rgba)
        && f.output_format(PixelFormat::Rgba) == PixelFormat::Rgba
    {
        return Ok(Route::Palette);
    }
    if let Some(hop) = hop_for(f, fmt) {
        return Ok(hop);
    }
    Err(refusal(short_name::<F>(), fmt, accepted))
}

/// Every layout a single-input filter takes, natively or through a
/// route, in [`layout::ALL`] order.
pub(crate) fn accepted_single<F: PixelFilter + ?Sized>(f: &F) -> Vec<PixelFormat> {
    layout::ALL
        .iter()
        .copied()
        .filter(|&fmt| route_single(f, fmt).is_ok())
        .collect()
}

/// The layout a single-input filter emits for `input`, when it takes
/// `input` at all.
pub(crate) fn output_single<F: PixelFilter + ?Sized>(
    f: &F,
    input: PixelFormat,
) -> Option<PixelFormat> {
    match route_single(f, input).ok()? {
        Route::Native => Some(f.output_format(input)),
        Route::Palette => Some(PixelFormat::Pal8),
        Route::Hop { via, back } => Some(if back { input } else { f.output_format(via) }),
    }
}

/// [`crate::ImageFilter::apply`] for a [`PixelFilter`].
pub(crate) fn apply_single<F: PixelFilter + ?Sized>(
    f: &F,
    input: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    let route = route_single(f, params.format)?;
    layout::validate_frame(input, params, short_name::<F>())?;
    let pixels = side_channels::image_only(input);
    match route {
        Route::Native => {
            let mut out = f.apply_pixels(&pixels, params)?;
            side_channels::carry(
                input,
                &mut out,
                params.format,
                f.output_format(params.format),
            );
            Ok(out)
        }
        Route::Palette => apply_palette(f, input, &pixels, params),
        Route::Hop { via, back } => apply_hop(f, input, params, via, back),
    }
}

/// Extent of a frame from its first plane (filters emit tight strides).
fn extent_of(frame: &VideoFrame, fmt: PixelFormat) -> Option<(u32, u32)> {
    let p0 = frame.planes.first()?;
    let pb = fmt.plane_row_bytes(0, 1)?;
    if pb == 0 || p0.stride == 0 || p0.stride % pb != 0 || p0.data.len() % p0.stride != 0 {
        return None;
    }
    Some(((p0.stride / pb) as u32, (p0.data.len() / p0.stride) as u32))
}

/// The packed-RGB hop (module docs). The conversion sees the original
/// frame so its colour-signal and significant-bits records steer the
/// matrix / range; the result is stripped of whatever records the
/// converter emits and the input's records are carried under the usual
/// ruling.
fn apply_hop<F: PixelFilter + ?Sized>(
    f: &F,
    input: &VideoFrame,
    params: VideoStreamParams,
    via: PixelFormat,
    back: bool,
) -> Result<VideoFrame, Error> {
    let opts = ConvertOptions::default();
    let ctx = ConvertContext::new().with_signal(input.color_signal().unwrap_or_default());
    let info = FrameInfo::new(params.format, params.width, params.height);
    let hopped = convert_with(input, info, via, &opts, &ctx)?;
    let hopped = side_channels::image_only(&hopped).into_owned();
    let via_params = VideoStreamParams {
        format: via,
        ..params
    };
    let filtered = f.apply_pixels(&hopped, via_params)?;
    let mut out = if back {
        let (w, h) = extent_of(&filtered, via).ok_or_else(|| {
            Error::invalid(format!(
                "oxideav-image-filter: {}: hop output has no readable extent",
                short_name::<F>()
            ))
        })?;
        let returned = convert_with(
            &filtered,
            FrameInfo::new(via, w, h),
            params.format,
            &opts,
            &ctx,
        )?;
        side_channels::image_only(&returned).into_owned()
    } else {
        filtered
    };
    let out_fmt = if back {
        params.format
    } else {
        f.output_format(via)
    };
    side_channels::carry(input, &mut out, params.format, out_fmt);
    Ok(out)
}

/// The pointwise palette route: filter the palette entries as a single
/// row of `Rgba` pixels (alpha from the palette-alpha record, opaque
/// where it is silent), keep the index plane, and write the recoloured
/// table back as the output's palette records.
fn apply_palette<F: PixelFilter + ?Sized>(
    f: &F,
    input: &VideoFrame,
    pixels: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    let name = short_name::<F>();
    let palette = input.palette().ok_or_else(|| {
        Error::invalid(format!(
            "oxideav-image-filter: {name}: Pal8 picture carries no palette record"
        ))
    })?;
    let entries = palette.len() / 3;
    if entries == 0 || entries > 256 {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: {name}: palette holds {entries} entries (need 1..=256)"
        )));
    }
    let alpha = input.palette_alpha();
    let mut row = Vec::with_capacity(entries * 4);
    for i in 0..entries {
        row.extend_from_slice(&palette[3 * i..3 * i + 3]);
        row.push(alpha.and_then(|a| a.get(i).copied()).unwrap_or(255));
    }
    let table = VideoFrame {
        pts: input.pts,
        planes: vec![VideoPlane {
            stride: entries * 4,
            data: row,
        }],
    };
    let table_params = VideoStreamParams {
        format: PixelFormat::Rgba,
        width: entries as u32,
        height: 1,
    };
    let filtered = f.apply_pixels(&table, table_params)?;
    let plane = filtered.planes.first().ok_or_else(|| {
        Error::invalid(format!(
            "oxideav-image-filter: {name}: palette pass returned no plane"
        ))
    })?;
    if plane.data.len() < entries * 4 {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: {name}: palette pass returned {} bytes for {entries} entries",
            plane.data.len()
        )));
    }
    let mut rgb = Vec::with_capacity(entries * 3);
    let mut a = Vec::with_capacity(entries);
    for i in 0..entries {
        rgb.extend_from_slice(&plane.data[4 * i..4 * i + 3]);
        a.push(plane.data[4 * i + 3]);
    }
    let mut out = VideoFrame {
        pts: input.pts,
        planes: pixels.planes.clone(),
    };
    side_channels::carry(input, &mut out, params.format, PixelFormat::Pal8);
    out.set_palette(rgb);
    if alpha.is_some() || a.iter().any(|&v| v != 255) {
        out.set_palette_alpha(a);
    } else {
        out.take_palette_alpha();
    }
    Ok(out)
}

/// The route a two-input filter takes for `fmt`, or the typed refusal.
/// Two-input filters have no palette route: both pictures would need
/// their tables merged, which is a quantisation decision, not a filter.
pub(crate) fn route_pair<F: TwoInputPixelFilter + ?Sized>(
    f: &F,
    fmt: PixelFormat,
) -> Result<Route, Error> {
    let accepted = f.accepted_formats();
    if accepted.contains(&fmt) {
        Ok(Route::Native)
    } else {
        Err(refusal(short_name::<F>(), fmt, accepted))
    }
}

/// Every layout a two-input filter takes, in [`layout::ALL`] order.
pub(crate) fn accepted_pair<F: TwoInputPixelFilter + ?Sized>(f: &F) -> Vec<PixelFormat> {
    layout::ALL
        .iter()
        .copied()
        .filter(|&fmt| route_pair(f, fmt).is_ok())
        .collect()
}

/// The layout a two-input filter emits for `input`, when it takes it.
pub(crate) fn output_pair<F: TwoInputPixelFilter + ?Sized>(
    f: &F,
    input: PixelFormat,
) -> Option<PixelFormat> {
    route_pair(f, input).ok().map(|_| f.output_format(input))
}

/// [`crate::TwoInputImageFilter::apply_two`] for a
/// [`TwoInputPixelFilter`]. Side-channel records are carried from
/// `src` (input port 0).
pub(crate) fn apply_pair<F: TwoInputPixelFilter + ?Sized>(
    f: &F,
    src: &VideoFrame,
    dst: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    route_pair(f, params.format)?;
    let name = short_name::<F>();
    layout::validate_frame(src, params, name)?;
    layout::validate_frame(dst, f.dst_params(params), name)?;
    let src_pixels = side_channels::image_only(src);
    let dst_pixels = side_channels::image_only(dst);
    let mut out = f.apply_two_pixels(&src_pixels, &dst_pixels, params)?;
    side_channels::carry(src, &mut out, params.format, f.output_format(params.format));
    Ok(out)
}
