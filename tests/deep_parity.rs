//! Deep-path parity: every catalogued filter with a float path gives,
//! on a picture widened to 16-bit words and to binary32, the result it
//! gives on the 8-bit original to within one 8-bit code once compared
//! at 8 bits — and a 16-bit picture through a chain of tonal filters
//! does not band.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::catalog::all_single;
use oxideav_image_filter::{Gamma, ImageFilter, Level, VideoStreamParams};

fn params(fmt: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
    VideoStreamParams {
        format: fmt,
        width: w,
        height: h,
    }
}

/// A deterministic 8-bit picture with `ch` interleaved channels.
fn picture8(w: u32, h: u32, ch: usize, seed: u32) -> Vec<u8> {
    (0..w as usize * h as usize * ch)
        .map(|i| {
            let c = i % ch;
            let p = i / ch;
            let x = (p % w as usize) as u32;
            let y = (p / w as usize) as u32;
            // Alpha partial but never zero, tones a slow ramp plus noise.
            if ch == 4 && c == 3 {
                (90 + (x * 37 + y * 11 + seed) % 150) as u8
            } else {
                ((x * 29 + y * 17 + c as u32 * 61 + seed * 7 + (i as u32 * 13) % 23) % 256) as u8
            }
        })
        .collect()
}

fn frame8(data: &[u8], w: u32, ch: usize) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * ch,
            data: data.to_vec(),
        }],
    }
}

fn frame16(data: &[u8], w: u32, ch: usize) -> VideoFrame {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &b in data {
        out.extend_from_slice(&(b as u16 * 257).to_le_bytes());
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * ch * 2,
            data: out,
        }],
    }
}

fn frame_f32(data: &[u8], w: u32, ch: usize) -> VideoFrame {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &b in data {
        out.extend_from_slice(&(b as f32 / 255.0).to_le_bytes());
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * ch * 4,
            data: out,
        }],
    }
}

/// Narrow a 16-bit plane to 8-bit codes (`round(v / 257)`).
fn narrow16(f: &VideoFrame) -> Vec<u8> {
    f.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| ((u16::from_le_bytes(*c) as u32 + 128) / 257) as u8)
        .collect()
}

/// Narrow a float plane to 8-bit codes (`round(clamp(v) · 255)`).
fn narrow_f32(f: &VideoFrame) -> Vec<u8> {
    f.planes[0]
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = f32::from_le_bytes(*c);
            if v.is_nan() {
                0
            } else {
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            }
        })
        .collect()
}

/// Filters whose 8-bit path quantises between passes (two rounded
/// separable passes, iterated kernels, integer histograms) and so may
/// sit two codes away from the unrounded float result.
fn tolerance(name: &str) -> i32 {
    match name {
        "" => 2,
        _ => 1,
    }
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> (i32, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0i32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x as i32 - y as i32).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

#[test]
fn widened_pictures_agree_with_their_8bit_original() {
    let (w, h) = (11u32, 7u32);
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for entry in all_single() {
        let f = entry.filter.as_ref();
        for (f8, f16, ff, ch) in [
            (
                PixelFormat::Rgba,
                PixelFormat::Rgba64Le,
                PixelFormat::RgbaF32Le,
                4usize,
            ),
            (
                PixelFormat::Rgb24,
                PixelFormat::Rgb48Le,
                PixelFormat::RgbF32Le,
                3,
            ),
            (
                PixelFormat::Gray8,
                PixelFormat::Gray16Le,
                PixelFormat::GrayF32Le,
                1,
            ),
        ] {
            let accepted = f.accepted_pixel_formats();
            if !accepted.contains(&ff) || !accepted.contains(&f8) {
                continue;
            }
            // Only the layout-preserving cells compare sample by sample.
            if f.output_pixel_format(f8) != Some(f8)
                || f.output_pixel_format(f16) != Some(f16)
                || f.output_pixel_format(ff) != Some(ff)
            {
                continue;
            }
            let data = picture8(w, h, ch, 5);
            let out8 = f.apply(&frame8(&data, w, ch), params(f8, w, h)).unwrap();
            let out16 = f.apply(&frame16(&data, w, ch), params(f16, w, h)).unwrap();
            let outf = f.apply(&frame_f32(&data, w, ch), params(ff, w, h)).unwrap();
            let a = &out8.planes[0].data;
            if a.len() != data.len() {
                // Extent-changing filters: geometry is the matrix test's job.
                continue;
            }
            checked += 1;
            let tol = tolerance(entry.name);
            let (d16, i16) = max_abs_diff(a, &narrow16(&out16));
            if d16 > tol {
                failures.push(format!(
                    "{} on {f16:?}: sample {i16} differs by {d16} codes (tolerance {tol})",
                    entry.name
                ));
            }
            let (df, i_f) = max_abs_diff(a, &narrow_f32(&outf));
            if df > tol {
                failures.push(format!(
                    "{} on {ff:?}: sample {i_f} differs by {df} codes (tolerance {tol})",
                    entry.name
                ));
            }
        }
    }
    eprintln!("deep parity cells checked: {checked}");
    assert!(
        failures.is_empty(),
        "{} parity failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked >= 20, "only {checked} deep cells");
}

#[test]
fn sixteen_bit_tonal_chain_does_not_band() {
    // A 16-bit ramp of 4 096 distinct codes through level → gamma keeps
    // thousands of distinct codes; the same chain through 8 bits would
    // collapse to at most 256.
    let w = 4096u32;
    let mut data = Vec::with_capacity(w as usize * 2);
    for x in 0..w {
        data.extend_from_slice(&((x * 16) as u16).to_le_bytes());
    }
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 2,
            data,
        }],
    };
    let p = params(PixelFormat::Gray16Le, w, 1);
    let levelled = Level::new(16, 235, 1.0).apply(&input, p).unwrap();
    let out = Gamma::new(2.2).apply(&levelled, p).unwrap();
    let codes: std::collections::BTreeSet<u16> = out.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    assert!(
        codes.len() > 2000,
        "16-bit chain banded to {} codes",
        codes.len()
    );
    // The float chain keeps every intermediate (no quantisation at all).
    let pf = params(PixelFormat::GrayF32Le, w, 1);
    let mut fdata = Vec::new();
    for x in 0..w {
        fdata.extend_from_slice(&(x as f32 / (w - 1) as f32).to_le_bytes());
    }
    let finput = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 4,
            data: fdata,
        }],
    };
    let fout = Gamma::new(2.2)
        .apply(&Level::new(16, 235, 1.0).apply(&finput, pf).unwrap(), pf)
        .unwrap();
    let distinct: std::collections::BTreeSet<u32> = fout.planes[0]
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert!(
        distinct.len() > 3500,
        "float chain collapsed to {}",
        distinct.len()
    );
}

#[test]
fn float_level_keeps_headroom_and_negative_excursions() {
    // `-level` with endpoints outside [0, 1] on a float picture: the
    // linear remap is not clamped.
    let p = params(PixelFormat::GrayF32Le, 3, 1);
    let mut data = Vec::new();
    for v in [0.0f32, 0.5, 1.0] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: 12, data }],
    };
    let out = Level::normalized(0.25, 0.75, 1.0).apply(&input, p).unwrap();
    let vals: Vec<f32> = out.planes[0]
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert!((vals[0] + 0.5).abs() < 1e-6, "{vals:?}");
    assert!((vals[1] - 0.5).abs() < 1e-6, "{vals:?}");
    assert!((vals[2] - 1.5).abs() < 1e-6, "{vals:?}");
    // The same picture as 16-bit words saturates at the full scale.
    let p16 = params(PixelFormat::Gray16Le, 3, 1);
    let mut d16 = Vec::new();
    for v in [0u16, 32768, 65535] {
        d16.extend_from_slice(&v.to_le_bytes());
    }
    let in16 = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 6,
            data: d16,
        }],
    };
    let out16 = Level::normalized(0.25, 0.75, 1.0)
        .apply(&in16, p16)
        .unwrap();
    let codes: Vec<u16> = out16.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    assert_eq!(codes[0], 0);
    assert!((codes[1] as i32 - 32768).abs() <= 1, "{codes:?}");
    assert_eq!(codes[2], 65535);
}
