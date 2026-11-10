//! The README's depth-support table is generated from the filters
//! themselves: every catalogued filter is asked which layouts it takes,
//! and each resulting row must appear verbatim in `README.md`. A filter
//! that does not take 16-bit or float pictures must carry a reason.
//!
//! `IMF_PRINT_TABLE=1 cargo test --test depth_support_table -- --nocapture`
//! prints the table for pasting.

use oxideav_core::PixelFormat;
use oxideav_image_filter::catalog::{all_pair, all_single};

const EIGHT: [PixelFormat; 4] = [
    PixelFormat::Rgba,
    PixelFormat::Rgb24,
    PixelFormat::Gray8,
    PixelFormat::Yuv420P,
];
const SIXTEEN: [PixelFormat; 3] = [
    PixelFormat::Rgba64Le,
    PixelFormat::Rgb48Le,
    PixelFormat::Gray16Le,
];
const FLOAT: [PixelFormat; 3] = [
    PixelFormat::RgbaF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::GrayF32Le,
];

/// Why a filter has no 16-bit / float path (keyed by catalogue name).
fn reason(name: &str) -> &'static str {
    match name {
        "ChannelExtract(U)" => {
            "n/a: reads a chroma plane of 8-bit planar YUV; the deep layouts reach filters as packed RGB / grey"
        }
        "Stegano" => "n/a: hides a picture in the low bits of 8-bit codes; defined on that storage",
        "HoughLines" | "HoughCircles" | "Deskew" => {
            "8-bit only: analysis on an 8-bit luma / edge map; float path not written yet (typed refusal)"
        }
        "FreiChen" | "Gabor" | "Sketch" | "Toon" | "Comic" => {
            "8-bit only: built on an 8-bit luma edge map; float path not written yet (typed refusal)"
        }
        "Crystallize" | "FloodFill" | "Halftone" | "InnerShadow" | "Feather" => {
            "8-bit only: float path not written yet (typed refusal, never narrowed)"
        }
        _ => "",
    }
}

fn mark(accepted: &[PixelFormat], probe: &[PixelFormat]) -> bool {
    probe.iter().any(|f| accepted.contains(f))
}

#[test]
fn readme_depth_table_is_what_the_filters_declare() {
    let mut rows: Vec<(String, bool, bool, bool)> = Vec::new();
    for e in all_single() {
        let a = e.filter.accepted_pixel_formats();
        rows.push((
            e.name.to_string(),
            mark(&a, &EIGHT),
            mark(&a, &SIXTEEN),
            mark(&a, &FLOAT),
        ));
    }
    for e in all_pair() {
        let a = e.filter.accepted_pixel_formats();
        rows.push((
            format!("{} (two-input)", e.name),
            mark(&a, &EIGHT),
            mark(&a, &SIXTEEN),
            mark(&a, &FLOAT),
        ));
    }
    let readme = include_str!("../README.md");
    let print = std::env::var_os("IMF_PRINT_TABLE").is_some();
    let yes = |b: bool| if b { "yes" } else { "no" };
    let (mut deep, mut total) = (0usize, 0usize);
    let mut missing = Vec::new();
    for (name, b8, b16, bf) in &rows {
        total += 1;
        let base = name.trim_end_matches(" (two-input)");
        let why = if *b16 && *bf { "" } else { reason(base) };
        assert!(*b8, "{name} takes no 8-bit layout");
        assert_eq!(b16, bf, "{name}: 16-bit pictures run on the float path");
        if *b16 {
            deep += 1;
        } else {
            assert!(
                !why.is_empty(),
                "{name} has no deep path and no stated reason"
            );
        }
        let row = format!(
            "| `{name}` | {} | {} | {} | {why} |",
            yes(*b8),
            yes(*b16),
            yes(*bf)
        );
        if print {
            println!("{row}");
        }
        if !readme.contains(&row) {
            missing.push(row);
        }
    }
    let summary = format!("{deep} of {total} catalogued filters take 16-bit and float pictures");
    if print {
        println!("{summary}");
    }
    assert!(
        missing.is_empty(),
        "README.md depth table is missing {} rows, e.g.\n{}",
        missing.len(),
        missing[..missing.len().min(5)].join("\n")
    );
    assert!(
        readme.contains(&summary),
        "README.md should state: {summary}"
    );
}
