//! Filter × layout matrix: every filter type in the crate, over every
//! pixel layout `oxideav-core` defines, on a picture carrying every
//! side-channel record core 0.1.40 knows (metadata blobs, palette,
//! palette alpha, significant bits, colour signal, layer identity).
//!
//! For each cell the contract is:
//!
//! - the call returns — a panic is a failure whatever the layout;
//! - a layout the filter declares produces `Ok`, and the output's
//!   image planes match the declared output layout's geometry;
//! - a layout it does not declare is a typed `Error::Unsupported`
//!   naming the layout;
//! - the input's blobs and layer identity ride on every output
//!   byte-for-byte, the palette records only onto a `Pal8` output, the
//!   colour signal only within the same carriage, the significant-bits
//!   record only onto an unchanged layout;
//! - a picture without records yields an output without records.

use std::panic::{AssertUnwindSafe, catch_unwind};

use oxideav_core::{
    BlobKind, ColorSignal, Error, LayerIdentity, MetadataBlob, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_image_filter::catalog::{all_pair, all_single};
use oxideav_image_filter::layout::{self, SampleKind, is_yuv, sample_kind};
use oxideav_image_filter::{ImageFilter, TwoInputImageFilter, VideoStreamParams};

/// A `w × h` picture in `fmt` with a deterministic sample pattern on
/// every plane (tight strides), no side-channel records.
fn picture(fmt: PixelFormat, w: u32, h: u32, seed: u8) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            let data = (0..stride * rows as usize)
                .map(|i| ((i as u32 * 37 + p as u32 * 11 + seed as u32) % 251) as u8)
                .collect();
            VideoPlane { stride, data }
        })
        .collect();
    VideoFrame {
        pts: Some(9),
        planes,
    }
}

/// Significant-bits record that is valid for `fmt` (at or below the
/// nominal depth of every plane).
fn sig_bits(fmt: PixelFormat) -> Vec<u8> {
    let bits = match (fmt, sample_kind(fmt)) {
        (PixelFormat::MonoBlack | PixelFormat::MonoWhite, _) => 1,
        (_, Some(SampleKind::U16 { bits })) => bits.saturating_sub(1).max(1),
        _ => 7,
    };
    vec![bits; fmt.plane_count()]
}

fn blobs() -> Vec<MetadataBlob> {
    vec![
        MetadataBlob::new(BlobKind::ICC, vec![1, 2, 3, 4]).with_mime("application/vnd.iccprofile"),
        MetadataBlob::new(BlobKind::XMP, b"<x:xmpmeta/>".to_vec()),
    ]
}

/// A 4-entry palette (black, red, green, white) with a transparent
/// first entry.
fn palette() -> (Vec<u8>, Vec<u8>) {
    (
        vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255],
        vec![0, 255, 255, 255],
    )
}

/// `picture` plus every side-channel record.
fn tagged(fmt: PixelFormat, w: u32, h: u32, seed: u8) -> VideoFrame {
    let mut f = picture(fmt, w, h, seed);
    if fmt == PixelFormat::Pal8 {
        // Indices must stay inside the 4-entry table.
        for v in &mut f.planes[0].data {
            *v %= 4;
        }
    }
    let (rgb, alpha) = palette();
    f.with_blobs(blobs())
        .with_palette(rgb)
        .with_palette_alpha(alpha)
        .with_significant_bits(sig_bits(fmt))
        .with_color_signal(ColorSignal::default())
        .with_layer(LayerIdentity::default())
}

fn params(fmt: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
    VideoStreamParams {
        format: fmt,
        width: w,
        height: h,
    }
}

/// The trailing side-channel run of a frame, as raw `(stride, data)`.
fn records(f: &VideoFrame) -> Vec<(usize, Vec<u8>)> {
    f.planes[f.image_plane_count()..]
        .iter()
        .map(|p| (p.stride, p.data.clone()))
        .collect()
}

/// Filters whose output extent differs from the input extent for the
/// catalog parameters (their planes are checked for mutual consistency
/// only).
const CHANGES_EXTENT: &[&str] = &[
    "AutoTrim",
    "BorderedFrame",
    "ChannelExtract(U)",
    "Crop",
    "Deskew",
    "DropShadow",
    "Extent",
    "Frame",
    "GravityTranslate",
    "Resize",
    "Rotate",
    "Shave",
    "Trim",
];

/// Check that `out` holds exactly the image planes of `fmt`, sized
/// consistently (exactly `w × h` unless `free_extent`).
fn check_geometry(
    name: &str,
    in_fmt: PixelFormat,
    out: &VideoFrame,
    fmt: PixelFormat,
    w: u32,
    h: u32,
    free_extent: bool,
) {
    let n = out.image_plane_count();
    assert_eq!(
        n,
        fmt.plane_count(),
        "{name} on {in_fmt:?}: declared output {fmt:?} has {} planes, emitted {n}",
        fmt.plane_count()
    );
    let (w, h) = if free_extent {
        let p0 = &out.planes[0];
        assert!(
            p0.stride > 0 && p0.data.len().is_multiple_of(p0.stride),
            "{name} on {in_fmt:?}: plane 0 ragged"
        );
        let rows = (p0.data.len() / p0.stride) as u32;
        let w = match fmt {
            // Two pixels per 4-byte macropixel.
            PixelFormat::Yuyv422 | PixelFormat::Uyvy422 => {
                assert_eq!(
                    p0.stride % 4,
                    0,
                    "{name} on {in_fmt:?}: macropixel stride {}",
                    p0.stride
                );
                (p0.stride / 2) as u32
            }
            // Eight pixels per byte: the extent is not recoverable from a
            // row; row consistency was checked above.
            PixelFormat::MonoBlack | PixelFormat::MonoWhite => return,
            _ => {
                let pb = fmt.plane_row_bytes(0, 1).unwrap();
                assert!(
                    p0.stride.is_multiple_of(pb),
                    "{name} on {in_fmt:?}: plane 0 stride {}",
                    p0.stride
                );
                (p0.stride / pb) as u32
            }
        };
        (w, rows)
    } else {
        (w, h)
    };
    for (i, p) in out.image_planes().iter().enumerate() {
        let (_, rows) = fmt.plane_dimensions(i, w, h).unwrap();
        let row_bytes = fmt.plane_row_bytes(i, w).unwrap();
        assert_eq!(
            p.stride, row_bytes,
            "{name} on {in_fmt:?} → {fmt:?}: plane {i} stride {} vs row {row_bytes} at {w}x{h}",
            p.stride
        );
        assert_eq!(
            p.data.len(),
            p.stride * rows as usize,
            "{name} on {in_fmt:?} → {fmt:?}: plane {i} holds {} bytes, {} expected at {w}x{h}",
            p.data.len(),
            p.stride * rows as usize
        );
    }
}

fn check_records(
    name: &str,
    input: &VideoFrame,
    in_fmt: PixelFormat,
    out: &VideoFrame,
    out_fmt: PixelFormat,
    palette_rewritten: bool,
) {
    assert_eq!(out.blobs(), input.blobs(), "{name} on {in_fmt:?}: blobs");
    // The blobs record itself is byte-identical (not merely equal after
    // a decode / re-encode).
    let in_blob_rec = records(input).into_iter().find(|r| r.0 == usize::MAX - 4);
    let out_blob_rec = records(out).into_iter().find(|r| r.0 == usize::MAX - 4);
    assert_eq!(
        in_blob_rec, out_blob_rec,
        "{name} on {in_fmt:?}: blobs record bytes"
    );
    assert_eq!(
        out.layer(),
        input.layer(),
        "{name} on {in_fmt:?}: layer identity"
    );
    if out_fmt == PixelFormat::Pal8 {
        assert!(
            out.palette().is_some(),
            "{name} on {in_fmt:?}: Pal8 output lost its palette"
        );
        if !palette_rewritten {
            assert_eq!(out.palette(), input.palette(), "{name}: palette");
            assert_eq!(
                out.palette_alpha(),
                input.palette_alpha(),
                "{name}: palette alpha"
            );
        }
    } else {
        assert!(
            out.palette().is_none(),
            "{name} on {in_fmt:?}: palette on a {out_fmt:?} output"
        );
        assert!(
            out.palette_alpha().is_none(),
            "{name} on {in_fmt:?}: palette alpha on a {out_fmt:?} output"
        );
    }
    if is_yuv(in_fmt) == is_yuv(out_fmt) {
        assert_eq!(
            out.color_signal(),
            input.color_signal(),
            "{name} on {in_fmt:?}: colour signal"
        );
    } else {
        assert!(
            out.color_signal().is_none(),
            "{name} on {in_fmt:?}: colour signal across a matrix hop"
        );
    }
    if in_fmt == out_fmt {
        assert_eq!(
            out.significant_bits(),
            input.significant_bits(),
            "{name} on {in_fmt:?}: significant bits"
        );
    } else {
        assert!(
            out.significant_bits().is_none(),
            "{name} on {in_fmt:?}: significant bits on a {out_fmt:?} output"
        );
    }
}

/// One matrix cell for a single-input filter. Returns `true` when the
/// layout was accepted.
fn cell_single(name: &str, f: &dyn ImageFilter, fmt: PixelFormat, w: u32, h: u32) -> bool {
    let input = tagged(fmt, w, h, 3);
    let accepted = f.accepted_pixel_formats();
    let declared = accepted.contains(&fmt);
    let result = catch_unwind(AssertUnwindSafe(|| f.apply(&input, params(fmt, w, h))));
    let result = match result {
        Ok(r) => r,
        Err(_) => panic!("{name} on {fmt:?} {w}x{h}: PANICKED"),
    };
    match (declared, result) {
        (true, Ok(out)) => {
            let out_fmt = f
                .output_pixel_format(fmt)
                .unwrap_or_else(|| panic!("{name}: accepts {fmt:?} but declares no output layout"));
            check_geometry(
                name,
                fmt,
                &out,
                out_fmt,
                w,
                h,
                CHANGES_EXTENT.contains(&name),
            );
            let palette_route = fmt == PixelFormat::Pal8 && !layout_native(f, fmt);
            check_records(name, &input, fmt, &out, out_fmt, palette_route);
            // A clean picture yields a clean output.
            let clean = picture(fmt, w, h, 3);
            let clean = if fmt == PixelFormat::Pal8 {
                let mut c = clean;
                for v in &mut c.planes[0].data {
                    *v %= 4;
                }
                let (rgb, _) = palette();
                c.with_palette(rgb)
            } else {
                clean
            };
            let Ok(out) = f.apply(&clean, params(fmt, w, h)) else {
                // Same picture, same typed geometry refusal as above.
                return true;
            };
            if fmt == PixelFormat::Pal8 {
                assert!(
                    out.blobs().is_empty() && out.layer().is_none(),
                    "{name} on {fmt:?}: records appeared"
                );
            } else {
                assert_eq!(
                    out.image_plane_count(),
                    out.planes.len(),
                    "{name} on {fmt:?}: records appeared"
                );
            }
            true
        }
        // The packed-RGB hop converts through `oxideav-pixfmt`, which
        // takes packed 4:2:2 at even widths and 4:1:1 at widths that are
        // multiples of 4 only — on the way in and on the way back, so a
        // filter whose output extent breaks the rule (a 3-wide crop)
        // refuses with a typed InvalidData. That is the contract (never
        // a panic).
        (true, Err(Error::InvalidData(msg)))
            if matches!(
                fmt,
                PixelFormat::Yuyv422 | PixelFormat::Uyvy422 | PixelFormat::Yuv411P
            ) =>
        {
            assert!(msg.contains("pixfmt"), "{name} on {fmt:?}: {msg}");
            true
        }
        (true, Err(e)) => panic!("{name} declares {fmt:?} but failed on {w}x{h}: {e}"),
        (false, Ok(_)) => panic!("{name} does not declare {fmt:?} but accepted it"),
        (false, Err(Error::Unsupported(msg))) => {
            assert!(
                msg.contains(&format!("{fmt:?}")),
                "{name}: refusal does not name {fmt:?}: {msg}"
            );
            false
        }
        (false, Err(e)) => panic!("{name} on {fmt:?}: refusal is not typed Unsupported: {e}"),
    }
}

/// `true` when `fmt` is handled without the palette route — i.e. the
/// filter declares a `Pal8`-preserving native path. The public surface
/// exposes only the union, so infer from the output label: the palette
/// route keeps `Pal8` while rewriting the table, native paths keep the
/// input's table.
fn layout_native(f: &dyn ImageFilter, fmt: PixelFormat) -> bool {
    if fmt != PixelFormat::Pal8 {
        return true;
    }
    // Probe: a native Pal8 path must keep the palette bytes; the palette
    // route recolours. A 1×1 picture distinguishes them cheaply for
    // every pointwise filter in the catalog (identity-shaped parameters
    // are avoided in the catalog precisely for this reason).
    let input = tagged(fmt, 2, 2, 5);
    match f.apply(&input, params(fmt, 2, 2)) {
        Ok(out) => out.palette() == input.palette() && out.palette_alpha() == input.palette_alpha(),
        Err(_) => true,
    }
}

#[test]
fn every_single_input_filter_over_every_layout() {
    let filters = all_single();
    assert!(
        filters.len() >= 135,
        "catalog lists {} single-input filters",
        filters.len()
    );
    let mut accepted_cells = 0usize;
    let mut refused_cells = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for entry in &filters {
        for &fmt in layout::ALL {
            for &(w, h) in &[(8u32, 6u32), (7, 5)] {
                match catch_unwind(AssertUnwindSafe(|| {
                    cell_single(entry.name, entry.filter.as_ref(), fmt, w, h)
                })) {
                    Ok(true) => accepted_cells += 1,
                    Ok(false) => refused_cells += 1,
                    Err(e) => {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        failures.push(format!("{} on {fmt:?} {w}x{h}: {msg}", entry.name));
                    }
                }
            }
        }
    }
    eprintln!(
        "single-input matrix: {} filters × {} layouts × 2 sizes = {} cells: {accepted_cells} ok, {refused_cells} typed refusals",
        filters.len(),
        layout::ALL.len(),
        accepted_cells + refused_cells
    );
    assert!(
        failures.is_empty(),
        "{} failing cells:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(
        accepted_cells + refused_cells,
        filters.len() * layout::ALL.len() * 2
    );
}

#[test]
fn every_two_input_filter_over_every_layout() {
    let filters = all_pair();
    assert_eq!(filters.len(), 8);
    let mut accepted_cells = 0usize;
    for entry in &filters {
        let f: &dyn TwoInputImageFilter = entry.filter.as_ref();
        for &fmt in layout::ALL {
            let (w, h) = (16u32, 16u32);
            let src = tagged(fmt, w, h, 3);
            // Port 1 carries its own (different) records to prove the
            // output carries port 0's.
            let mut dst = tagged(fmt, w, h, 4);
            dst.set_blobs(vec![MetadataBlob::new(BlobKind::EXIF, vec![9])]);
            // Watermark's overlay is 2×2.
            let dst = if entry.name == "Watermark" {
                tagged(fmt, 2, 2, 4)
            } else {
                dst
            };
            let declared = f.accepted_pixel_formats().contains(&fmt);
            let result = catch_unwind(AssertUnwindSafe(|| {
                f.apply_two(&src, &dst, params(fmt, w, h))
            }));
            let result = match result {
                Ok(r) => r,
                Err(_) => panic!("{} on {fmt:?}: PANICKED", entry.name),
            };
            match (declared, result) {
                (true, Ok(out)) => {
                    let out_fmt = f.output_pixel_format(fmt).unwrap();
                    check_geometry(entry.name, fmt, &out, out_fmt, w, h, false);
                    check_records(entry.name, &src, fmt, &out, out_fmt, false);
                    accepted_cells += 1;
                }
                (true, Err(e)) => panic!("{} declares {fmt:?} but failed: {e}", entry.name),
                (false, Ok(_)) => panic!("{} does not declare {fmt:?} but accepted it", entry.name),
                (false, Err(Error::Unsupported(msg))) => {
                    assert!(msg.contains(&format!("{fmt:?}")), "{}: {msg}", entry.name);
                }
                (false, Err(e)) => panic!("{} on {fmt:?}: refusal is not typed: {e}", entry.name),
            }
        }
    }
    eprintln!("two-input matrix: {accepted_cells} accepted cells");
    assert!(accepted_cells >= 8 * 2);
}

#[test]
fn malformed_frames_are_typed_refusals_everywhere() {
    for entry in all_single() {
        let f = entry.filter.as_ref();
        for &fmt in &[PixelFormat::Rgba, PixelFormat::Gray8, PixelFormat::Yuv420P] {
            if !f.accepted_pixel_formats().contains(&fmt) {
                continue;
            }
            // Short plane.
            let mut short = tagged(fmt, 8, 6, 1);
            short.planes[0].data.truncate(5);
            let r = catch_unwind(AssertUnwindSafe(|| f.apply(&short, params(fmt, 8, 6))));
            assert!(
                matches!(r, Ok(Err(Error::InvalidData(_)))),
                "{} on short {fmt:?}",
                entry.name
            );
            // Missing plane.
            let mut missing = tagged(fmt, 8, 6, 1);
            missing.planes.remove(0);
            let r = catch_unwind(AssertUnwindSafe(|| f.apply(&missing, params(fmt, 8, 6))));
            assert!(
                matches!(r, Ok(Err(Error::InvalidData(_)))),
                "{} on missing {fmt:?}",
                entry.name
            );
            // Empty extent.
            let r = catch_unwind(AssertUnwindSafe(|| {
                f.apply(&tagged(fmt, 8, 6, 1), params(fmt, 0, 6))
            }));
            assert!(
                matches!(r, Ok(Err(Error::InvalidData(_)))),
                "{} on empty {fmt:?}",
                entry.name
            );
        }
    }
}

/// A filter that declares itself pointwise must produce, for every
/// pixel, exactly what it produces for that pixel alone — and its
/// `Pal8` route must recolour each palette entry the same way.
#[test]
fn pointwise_filters_are_pointwise_and_recolour_palettes_entrywise() {
    let mut checked = 0usize;
    for entry in all_single() {
        let f = entry.filter.as_ref();
        if !f.is_pointwise() {
            continue;
        }
        assert!(
            f.accepted_pixel_formats().contains(&PixelFormat::Pal8),
            "{}",
            entry.name
        );
        checked += 1;
        let (w, h) = (3u32, 2u32);
        let input = picture(PixelFormat::Rgba, w, h, 11);
        let whole = f.apply(&input, params(PixelFormat::Rgba, w, h)).unwrap();
        for i in 0..(w * h) as usize {
            let px = &input.planes[0].data[4 * i..4 * i + 4];
            let one = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: 4,
                    data: px.to_vec(),
                }],
            };
            let alone = f.apply(&one, params(PixelFormat::Rgba, 1, 1)).unwrap();
            assert_eq!(
                &whole.planes[0].data[4 * i..4 * i + 4],
                &alone.planes[0].data[..4],
                "{}: pixel {i} depends on its neighbours",
                entry.name
            );
        }
        // Palette route: entry k of the output table == the filter on
        // entry k as a lone Rgba pixel.
        let pal = tagged(PixelFormat::Pal8, w, h, 11);
        let out = f.apply(&pal, params(PixelFormat::Pal8, w, h)).unwrap();
        assert_eq!(
            out.planes[0].data, pal.planes[0].data,
            "{}: indices changed",
            entry.name
        );
        let (rgb, alpha) = palette();
        let out_rgb = out.palette().unwrap();
        for k in 0..4 {
            let one = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: 4,
                    data: vec![rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2], alpha[k]],
                }],
            };
            let alone = f.apply(&one, params(PixelFormat::Rgba, 1, 1)).unwrap();
            assert_eq!(
                &out_rgb[3 * k..3 * k + 3],
                &alone.planes[0].data[..3],
                "{} entry {k}",
                entry.name
            );
            let a = out.palette_alpha().map(|a| a[k]).unwrap_or(255);
            assert_eq!(a, alone.planes[0].data[3], "{} entry {k} alpha", entry.name);
        }
    }
    eprintln!("pointwise filters checked: {checked}");
    assert!(checked >= 30);
}
