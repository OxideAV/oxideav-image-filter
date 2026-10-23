//! Resize kernels against `magick` (ImageMagick 7.1.2 Q16-HDRI).
//!
//! Two layers of evidence:
//!
//! 1. `tests/fixtures/magick_impulse.txt` holds magick's own response to
//!    a one-pixel impulse — a 65×1 picture enlarged to 520×1 (the kernel
//!    sampled every eighth of a pixel), the same impulse at pixel 0 (the
//!    edge rule), and a 65→23 shrink (the widening rule; the ratio puts no source centre exactly on a box edge, where magick's own rounding decides) — for every
//!    filter and for the expert defines (`blur`, `lobes`, `support`,
//!    `window`, `sigma`, `b` / `c`, the Kaiser `alpha`). The crate's
//!    resampler on the same picture must reproduce every sample to
//!    within `2e-5` (binary32 arithmetic against magick's doubles).
//! 2. When `magick` is on the path, the built-in `rose:` picture is
//!    resized live (shrunk and enlarged, 8-bit and 16-bit) with every
//!    filter and compared at PSNR ≥ 45 dB — the acceptance line of
//!    issue #10. Without the binary that layer is reported as skipped.

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    FilterOptions, ImageFilter, Interpolation, ResampleFilter, Resize, VideoStreamParams,
};

fn params(fmt: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
    VideoStreamParams {
        format: fmt,
        width: w,
        height: h,
    }
}

/// A `n × 1` GrayF32Le impulse picture (white at `pos`).
fn impulse(n: u32, pos: u32) -> VideoFrame {
    let mut data = Vec::with_capacity(n as usize * 4);
    for x in 0..n {
        data.extend_from_slice(&(if x == pos { 1.0f32 } else { 0.0 }).to_le_bytes());
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: n as usize * 4,
            data,
        }],
    }
}

fn floats(f: &VideoFrame) -> Vec<f32> {
    f.planes[0]
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Parse `filter:key=value|filter:key2=value2` into options.
fn options_from_defines(defs: &str) -> FilterOptions {
    let mut opts = FilterOptions::new();
    if defs == "-" {
        return opts;
    }
    for d in defs.split('|') {
        let d = d.strip_prefix("filter:").unwrap_or(d);
        let (k, v) = d.split_once('=').expect("define");
        opts = match k {
            "blur" => opts.with_blur(v.parse().unwrap()),
            "lobes" => opts.with_lobes(v.parse().unwrap()),
            "support" => opts.with_support(v.parse().unwrap()),
            "window" => opts.with_window(ResampleFilter::from_name(v).unwrap()),
            "sigma" => opts.with_sigma(v.parse().unwrap()),
            "b" => opts.with_b(v.parse().unwrap()),
            "c" => opts.with_c(v.parse().unwrap()),
            // magick's `filter:alpha` is the Kaiser β.
            "alpha" | "beta" => opts.with_kaiser_beta(v.parse().unwrap()),
            other => panic!("unknown define {other}"),
        };
    }
    opts
}

#[test]
fn every_filter_reproduces_magicks_impulse_response() {
    let fixture = include_str!("fixtures/magick_impulse.txt");
    let mut cases = 0usize;
    let mut worst = (0f32, String::new());
    let mut seen = std::collections::HashSet::new();
    for line in fixture
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut it = line.split(' ');
        let name = it.next().unwrap();
        let defs = it.next().unwrap();
        let n: u32 = it.next().unwrap().parse().unwrap();
        let m: u32 = it.next().unwrap().parse().unwrap();
        let pos: u32 = it.next().unwrap().parse().unwrap();
        let i0: usize = it.next().unwrap().parse().unwrap();
        let expected: Vec<f32> = it.map(|v| v.parse().unwrap()).collect();
        let filter = ResampleFilter::from_name(name).unwrap_or_else(|| panic!("{name}"));
        seen.insert(filter);
        let out = Resize::new(m, 1)
            .with_filter(filter)
            .with_filter_options(options_from_defines(defs))
            .apply(&impulse(n, pos), params(PixelFormat::GrayF32Le, n, 1))
            .unwrap();
        let got = floats(&out);
        assert_eq!(got.len(), m as usize);
        // Outside the recorded window the response is zero (an edge
        // tap of the Gaussian excepted, see below).
        for (i, &v) in got.iter().enumerate() {
            if i < i0 || i >= i0 + expected.len() {
                assert!(
                    v.abs() < 5e-4,
                    "{name} {defs} {n}->{m}@{pos}: sample {i} = {v} outside magick's support"
                );
            }
        }
        for (k, &e) in expected.iter().enumerate() {
            let g = got[i0 + k];
            let d = (g - e).abs();
            // A source centre exactly on the support edge is in or out by
            // magick's own double rounding; only the Gaussian has a
            // non-zero weight there (`exp(−8)` at `4σ`), worth < 5e-4 of
            // an output sample. Forgive that case.
            let edge_tap = e.abs() < 5e-4 && g.abs() < 5e-4;
            if d > worst.0 && !edge_tap {
                worst = (d, format!("{name} {defs} {n}->{m}@{pos} sample {}", i0 + k));
            }
            assert!(
                d <= 2e-5 || edge_tap,
                "{name} {defs} {n}->{m}@{pos}: sample {} is {g}, magick says {e}",
                i0 + k
            );
        }
        cases += 1;
    }
    eprintln!(
        "impulse cases: {cases}, worst |Δ| = {:.2e} at {}",
        worst.0, worst.1
    );
    assert!(cases >= 130);
    assert_eq!(
        seen.len(),
        ResampleFilter::ALL.len(),
        "every filter is covered"
    );
}

#[test]
fn filter_names_round_trip_and_aliases_parse() {
    for &f in ResampleFilter::ALL {
        assert_eq!(ResampleFilter::from_name(f.name()), Some(f));
        assert_eq!(ResampleFilter::from_name(&f.name().to_lowercase()), Some(f));
    }
    assert_eq!(
        ResampleFilter::from_name("lanczos-sharp"),
        Some(ResampleFilter::LanczosSharp)
    );
    assert_eq!(
        ResampleFilter::from_name("Hanning"),
        Some(ResampleFilter::Hann)
    );
    assert_eq!(ResampleFilter::from_name("nope"), None);
    assert_eq!(
        Resize::new(3, 3)
            .with_filter(ResampleFilter::Robidoux)
            .interpolation(),
        Interpolation::Filter(ResampleFilter::Robidoux)
    );
}

/// `magick` on the path, or `None` (the live layer is then skipped).
fn magick() -> Option<String> {
    let ok = Command::new("magick").arg("-version").output().ok()?;
    ok.status.success().then_some("magick".to_string())
}

/// `rose:` as packed RGB at `depth` (8 → Rgb24, 16 → Rgb48Le).
fn rose(bin: &str, depth: u32) -> Vec<u8> {
    Command::new(bin)
        .args([
            "rose:",
            "-depth",
            &depth.to_string(),
            "-endian",
            "LSB",
            "rgb:-",
        ])
        .output()
        .unwrap()
        .stdout
}

fn rose_resized(bin: &str, depth: u32, filter: ResampleFilter, w: u32, h: u32) -> Vec<u8> {
    Command::new(bin)
        .args([
            "rose:",
            "-filter",
            filter.name(),
            "-resize",
            &format!("{w}x{h}!"),
            "-depth",
            &depth.to_string(),
            "-endian",
            "LSB",
            "rgb:-",
        ])
        .output()
        .unwrap()
        .stdout
}

/// PSNR in dB of two packed samples streams of `bits` depth.
fn psnr(a: &[u8], b: &[u8], bits: u32) -> f64 {
    assert_eq!(a.len(), b.len());
    let full = ((1u32 << bits) - 1) as f64;
    let (va, vb): (Vec<f64>, Vec<f64>) = if bits == 8 {
        (
            a.iter().map(|&v| v as f64).collect(),
            b.iter().map(|&v| v as f64).collect(),
        )
    } else {
        (
            a.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c) as f64)
                .collect(),
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c) as f64)
                .collect(),
        )
    };
    let mse = va
        .iter()
        .zip(&vb)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / va.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (full * full / mse).log10()
    }
}

#[test]
fn rose_resized_live_matches_magick_at_45db() {
    let Some(bin) = magick() else {
        eprintln!("magick not on the path: live rose: comparison skipped");
        return;
    };
    let (rw, rh) = (70u32, 46u32);
    let mut report = Vec::new();
    for &(depth, fmt, bits) in &[
        (8u32, PixelFormat::Rgb24, 8u32),
        (16, PixelFormat::Rgb48Le, 16),
    ] {
        let src = rose(&bin, depth);
        assert_eq!(src.len(), (rw * rh * 3) as usize * (depth / 8) as usize);
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: rw as usize * 3 * (depth / 8) as usize,
                data: src,
            }],
        };
        for &filter in ResampleFilter::ALL {
            for &(w, h) in &[(37u32, 23u32), (131, 91), (97, 29)] {
                let want = rose_resized(&bin, depth, filter, w, h);
                let got = Resize::new(w, h)
                    .with_filter(filter)
                    .apply(&input, params(fmt, rw, rh))
                    .unwrap();
                let db = psnr(&got.planes[0].data, &want, bits);
                report.push(format!("{} {depth}-bit {w}x{h}: {db:.1} dB", filter.name()));
                assert!(
                    db >= 45.0,
                    "{} at {depth} bits to {w}x{h}: PSNR {db:.2} dB < 45",
                    filter.name()
                );
            }
        }
    }
    eprintln!("{}", report.join("\n"));
}
