//! The wrapper every filter runs behind: layout gate, frame validation,
//! side-channel stripping and re-attachment, and the pointwise palette
//! route for `Pal8` pictures.
//!
//! [`crate::ImageFilter::apply`] on a [`crate::PixelFilter`] is exactly
//! [`apply_single`]; [`crate::TwoInputImageFilter::apply_two`] on a
//! [`crate::TwoInputPixelFilter`] is exactly [`apply_pair`].

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

use crate::{layout, side_channels, PixelFilter, TwoInputPixelFilter, VideoStreamParams};

/// How a filter reaches a picture in `input` layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// The filter takes the layout as-is.
    Native,
    /// `Pal8` through a pointwise filter: the palette is filtered as a
    /// one-row `Rgba` picture and the index plane is untouched.
    Palette,
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
    }
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
