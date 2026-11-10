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
//! and the registry label says so.
//!
//! # The float hop
//!
//! Deep layouts (10/12/14/16-bit words, float) never take the packed-RGB
//! hop — that would narrow samples silently. A filter whose native paths
//! include the packed float layouts (`RgbaF32Le` / `RgbF32Le` /
//! `GrayF32Le`, see [`crate::deep`]) takes every deep layout the same
//! way instead: converted to the first of those three it takes (a pure
//! normalisation for the integer RGB / grey / GBR families, the matrix
//! at depth for YUV), filtered in binary32, and converted back
//! (saturating to the word's full scale) when the filter keeps the hop
//! layout. A filter without a float path refuses deep layouts with a
//! typed error.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{ConvertContext, ConvertOptions, FrameInfo, convert_with, supports};

use crate::layout::SampleKind;
use crate::{PixelFilter, TwoInputPixelFilter, VideoStreamParams, layout, side_channels};

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

/// The float-hop layouts for the deep (16-bit word and float) sources,
/// same preference order.
const DEEP_HOP_TARGETS: [PixelFormat; 3] = [
    PixelFormat::RgbaF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::GrayF32Le,
];

/// The hop a filter takes for a layout outside its native set, if any:
/// 8-bit storage hops to packed 8-bit, deep storage to packed float — a
/// hop never narrows a sample.
fn hop_for<F: PixelFilter + ?Sized>(f: &F, fmt: PixelFormat) -> Option<Route> {
    hop_among(f.accepted_formats(), |via| f.output_format(via), fmt)
}

/// [`hop_for`] on a filter described by its native set and its output
/// rule (shared by the single- and two-input routes).
fn hop_among(
    accepted: &[PixelFormat],
    output_format: impl Fn(PixelFormat) -> PixelFormat,
    fmt: PixelFormat,
) -> Option<Route> {
    // Only the layouts this crate knows hop; a layout outside
    // `layout::ALL` is refused before its storage is ever described.
    if !layout::ALL.contains(&fmt) {
        return None;
    }
    let mut targets = match layout::sample_kind(fmt) {
        Some(SampleKind::U8) | None => HOP_TARGETS,
        Some(SampleKind::U16 { .. }) | Some(SampleKind::F32) => DEEP_HOP_TARGETS,
    };
    // A grey source without alpha hops to grey when the filter takes
    // it: a colour filter run on a replicated grey would otherwise tint
    // its channels differently and the way back would take the luma.
    if is_plain_gray(fmt) {
        targets.reverse();
    }
    let via = targets
        .into_iter()
        .find(|t| accepted.contains(t) && supports(fmt, *t))?;
    let back = fmt != PixelFormat::Pal8 && output_format(via) == via && supports(via, fmt);
    Some(Route::Hop { via, back })
}

/// Grey layouts without alpha (the sources whose natural hop is grey).
fn is_plain_gray(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8
            | PixelFormat::Gray10Le
            | PixelFormat::Gray12Le
            | PixelFormat::Gray16Le
            | PixelFormat::GrayF32Le
            | PixelFormat::MonoBlack
            | PixelFormat::MonoWhite
    )
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

/// The channel count when the hop `fmt → via` is a packed 16-bit word
/// layout to its packed float twin (`Rgba64Le` ↔ `RgbaF32Le`, `Rgb48Le`
/// ↔ `RgbF32Le`, `Gray16Le` ↔ `GrayF32Le`) on a frame that claims all
/// 16 bits: that hop is the pure normalisation `code / 65535` and is
/// done here, row-parallel, with the results `oxideav-pixfmt` gives
/// (held equal by the tests below). Everything else converts through
/// `oxideav-pixfmt`.
fn word_hop(frame: &VideoFrame, fmt: PixelFormat, via: PixelFormat) -> Option<usize> {
    if frame.significant_bits().is_some() {
        return None;
    }
    match (fmt, via) {
        (PixelFormat::Rgba64Le, PixelFormat::RgbaF32Le) => Some(4),
        (PixelFormat::Rgb48Le, PixelFormat::RgbF32Le) => Some(3),
        (PixelFormat::Gray16Le, PixelFormat::GrayF32Le) => Some(1),
        _ => None,
    }
}

/// `code / 65535` on every little-endian word of plane 0, into a tight
/// packed float frame.
fn widen_words(frame: &VideoFrame, w: usize, h: usize, ch: usize) -> VideoFrame {
    let plane = &frame.planes[0];
    let n = w * ch;
    let mut data = vec![0u8; n * 4 * h];
    crate::exec::par_rows(&mut data, n * 4, |y, row| {
        let src = &plane.data[y * plane.stride..y * plane.stride + n * 2];
        for (d, s) in row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<2>().0)
        {
            *d = (u16::from_le_bytes(*s) as f32 / 65535.0).to_le_bytes();
        }
    });
    VideoFrame {
        pts: frame.pts,
        planes: vec![VideoPlane {
            stride: n * 4,
            data,
        }],
    }
}

/// The 16-bit word of a normalised sample: round to nearest, saturating
/// at both ends, NaN to zero.
#[inline]
fn word_of(v: f32) -> u16 {
    // A NaN stays NaN through `clamp` and casts to 0.
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// [`word_of`] on every sample of a packed float plane, into a tight
/// packed 16-bit frame.
fn narrow_words(frame: &VideoFrame, w: usize, h: usize, ch: usize) -> VideoFrame {
    let plane = &frame.planes[0];
    let n = w * ch;
    let mut data = vec![0u8; n * 2 * h];
    crate::exec::par_rows(&mut data, n * 2, |y, row| {
        let src = &plane.data[y * plane.stride..y * plane.stride + n * 4];
        for (d, s) in row
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<4>().0)
        {
            *d = word_of(f32::from_le_bytes(*s)).to_le_bytes();
        }
    });
    VideoFrame {
        pts: frame.pts,
        planes: vec![VideoPlane {
            stride: n * 2,
            data,
        }],
    }
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
    let words = word_hop(input, params.format, via);
    let hopped = match words {
        Some(ch) => widen_words(input, params.width as usize, params.height as usize, ch),
        None => {
            let hopped = convert_with(input, info, via, &opts, &ctx)?;
            side_channels::image_only(&hopped).into_owned()
        }
    };
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
        match words {
            Some(ch) => narrow_words(&filtered, w as usize, h as usize, ch),
            None => {
                let returned = convert_with(
                    &filtered,
                    FrameInfo::new(via, w, h),
                    params.format,
                    &opts,
                    &ctx,
                )?;
                side_channels::image_only(&returned).into_owned()
            }
        }
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
/// Two-input filters have no palette route (both pictures would need
/// their tables merged, which is a quantisation decision, not a
/// filter); they take the same two hops as single-input filters, both
/// pictures converted to the hop layout: 8-bit storage to packed 8-bit,
/// deep storage to packed float.
pub(crate) fn route_pair<F: TwoInputPixelFilter + ?Sized>(
    f: &F,
    fmt: PixelFormat,
) -> Result<Route, Error> {
    let accepted = f.accepted_formats();
    if accepted.contains(&fmt) {
        return Ok(Route::Native);
    }
    if let Some(hop) = hop_among(accepted, |via| f.output_format(via), fmt) {
        return Ok(hop);
    }
    Err(refusal(short_name::<F>(), fmt, accepted))
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
    match route_pair(f, input).ok()? {
        Route::Native | Route::Palette => Some(f.output_format(input)),
        Route::Hop { via, back } => Some(if back { input } else { f.output_format(via) }),
    }
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
    let route = route_pair(f, params.format)?;
    let name = short_name::<F>();
    let dst_params = f.dst_params(params);
    layout::validate_frame(src, params, name)?;
    layout::validate_frame(dst, dst_params, name)?;
    let (via, back) = match route {
        Route::Hop { via, back } => (via, back),
        Route::Native | Route::Palette => {
            let src_pixels = side_channels::image_only(src);
            let dst_pixels = side_channels::image_only(dst);
            let mut out = f.apply_two_pixels(&src_pixels, &dst_pixels, params)?;
            side_channels::carry(src, &mut out, params.format, f.output_format(params.format));
            return Ok(out);
        }
    };
    // The hop: each picture is converted under its own colour signal.
    let opts = ConvertOptions::default();
    let hop_in = |frame: &VideoFrame, p: VideoStreamParams| -> Result<VideoFrame, Error> {
        if let Some(ch) = word_hop(frame, p.format, via) {
            return Ok(widen_words(frame, p.width as usize, p.height as usize, ch));
        }
        let ctx = ConvertContext::new().with_signal(frame.color_signal().unwrap_or_default());
        let info = FrameInfo::new(p.format, p.width, p.height);
        let hopped = convert_with(frame, info, via, &opts, &ctx)?;
        Ok(side_channels::image_only(&hopped).into_owned())
    };
    let src_hopped = hop_in(src, params)?;
    let dst_hopped = hop_in(dst, dst_params)?;
    let via_params = VideoStreamParams {
        format: via,
        ..params
    };
    let filtered = f.apply_two_pixels(&src_hopped, &dst_hopped, via_params)?;
    let mut out = if back {
        let (w, h) = extent_of(&filtered, via).ok_or_else(|| {
            Error::invalid(format!(
                "oxideav-image-filter: {name}: hop output has no readable extent"
            ))
        })?;
        if let Some(ch) = word_hop(src, params.format, via) {
            narrow_words(&filtered, w as usize, h as usize, ch)
        } else {
            let ctx = ConvertContext::new().with_signal(src.color_signal().unwrap_or_default());
            let returned = convert_with(
                &filtered,
                FrameInfo::new(via, w, h),
                params.format,
                &opts,
                &ctx,
            )?;
            side_channels::image_only(&returned).into_owned()
        }
    } else {
        filtered
    };
    let out_fmt = if back {
        params.format
    } else {
        f.output_format(via)
    };
    side_channels::carry(src, &mut out, params.format, out_fmt);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-crate word hop and `oxideav-pixfmt` agree on every 16-bit
    /// code and on floats across, between and beyond the code grid.
    #[test]
    fn word_hop_equals_pixfmt_on_every_code_and_on_odd_floats() {
        let opts = ConvertOptions::default();
        let ctx = ConvertContext::new();
        for (deep, float, ch) in [
            (PixelFormat::Gray16Le, PixelFormat::GrayF32Le, 1usize),
            (PixelFormat::Rgb48Le, PixelFormat::RgbF32Le, 3),
            (PixelFormat::Rgba64Le, PixelFormat::RgbaF32Le, 4),
        ] {
            // 65536 codes, padded to a whole number of pixels.
            let w = 65536usize.div_ceil(ch);
            let words: Vec<u8> = (0..w * ch)
                .flat_map(|i| ((i % 65536) as u16).to_le_bytes())
                .collect();
            let frame = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: w * ch * 2,
                    data: words,
                }],
            };
            assert_eq!(word_hop(&frame, deep, float), Some(ch));
            let ours = widen_words(&frame, w, 1, ch);
            let theirs = convert_with(
                &frame,
                FrameInfo::new(deep, w as u32, 1),
                float,
                &opts,
                &ctx,
            )
            .unwrap();
            assert_eq!(ours.planes[0].data, theirs.planes[0].data, "{deep:?} widen");

            let mut floats: Vec<f32> = Vec::new();
            for i in 0..65536u32 {
                let c = i as f32;
                for off in [-0.5f32, -0.4999, -0.25, 0.0, 0.25, 0.4999, 0.5] {
                    floats.push((c + off) / 65535.0);
                }
            }
            floats.extend([
                -1.0,
                -0.0,
                1.0,
                1.000_001,
                2.5,
                1e30,
                -1e30,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                f32::MIN_POSITIVE,
                f32::EPSILON,
            ]);
            while !floats.len().is_multiple_of(ch) {
                floats.push(0.5);
            }
            let w = floats.len() / ch;
            let frame = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: w * ch * 4,
                    data: floats.iter().flat_map(|v| v.to_le_bytes()).collect(),
                }],
            };
            let ours = narrow_words(&frame, w, 1, ch);
            let theirs = convert_with(
                &frame,
                FrameInfo::new(float, w as u32, 1),
                deep,
                &opts,
                &ctx,
            )
            .unwrap();
            for (i, (a, b)) in ours.planes[0]
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .zip(theirs.planes[0].data.as_chunks::<2>().0)
                .enumerate()
            {
                assert_eq!(
                    u16::from_le_bytes(*a),
                    u16::from_le_bytes(*b),
                    "{deep:?} narrow of {}",
                    floats[i]
                );
            }
        }
    }

    #[test]
    fn a_depth_claim_keeps_the_hop_in_pixfmt() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![0, 1],
            }],
        }
        .with_significant_bits(vec![12]);
        assert_eq!(
            word_hop(&frame, PixelFormat::Gray16Le, PixelFormat::GrayF32Le),
            None
        );
    }
}
