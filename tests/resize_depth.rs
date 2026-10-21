//! Resize on layouts beyond 8-bit packed: 16-bit words stay 16-bit and
//! clamp at their own full scale, subsampled chroma planes resample on
//! their own grid, odd sizes are exact, float samples are not clamped,
//! semi-planar pairs resample as pairs, and `Pal8` is nearest-only.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{ImageFilter, Interpolation, Resize, VideoStreamParams};

const KERNELS: &[Interpolation] = &[
    Interpolation::Nearest,
    Interpolation::Bilinear,
    Interpolation::Bicubic,
    Interpolation::Area,
    Interpolation::Lanczos { a: 3 },
    Interpolation::Mitchell,
    Interpolation::BSpline,
];

fn params(fmt: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
    VideoStreamParams {
        format: fmt,
        width: w,
        height: h,
    }
}

/// A picture whose plane `p` holds `sample(p, x, y)` at every position,
/// written through `write` (bytes per sample).
fn picture(
    fmt: PixelFormat,
    w: u32,
    h: u32,
    sample: impl Fn(usize, usize, usize, usize) -> u32,
    sample_bytes: usize,
) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (pw, ph) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            let ch = stride / (pw as usize * sample_bytes);
            let mut data = Vec::with_capacity(stride * ph as usize);
            for y in 0..ph as usize {
                for x in 0..pw as usize {
                    for c in 0..ch {
                        let v = sample(p, x, y, c);
                        match sample_bytes {
                            1 => data.push(v as u8),
                            2 => data.extend_from_slice(&(v as u16).to_le_bytes()),
                            _ => data.extend_from_slice(&f32::from_bits(v).to_le_bytes()),
                        }
                    }
                }
            }
            VideoPlane { stride, data }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

fn u16s(plane: &VideoPlane) -> Vec<u16> {
    plane
        .data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn sixteen_bit_flat_fields_survive_every_kernel_at_full_scale() {
    let input = picture(PixelFormat::Gray16Le, 6, 5, |_, _, _, _| 60000, 2);
    for &k in KERNELS {
        for &(w, h) in &[(9u32, 7u32), (3, 2), (6, 5), (13, 1)] {
            let out = Resize::new(w, h)
                .with_interpolation(k)
                .apply(&input, params(PixelFormat::Gray16Le, 6, 5))
                .unwrap();
            assert_eq!(out.planes[0].stride, w as usize * 2, "{k:?}");
            assert_eq!(
                out.planes[0].data.len(),
                w as usize * h as usize * 2,
                "{k:?}"
            );
            assert!(
                u16s(&out.planes[0]).iter().all(|&v| v == 60000),
                "{k:?} {w}x{h}"
            );
        }
    }
}

#[test]
fn ten_bit_ringing_clamps_at_1023_not_255() {
    // A hard 0 / 1023 step: the windowed sinc overshoots on both sides;
    // the clamp must be the 10-bit full scale, never the 8-bit one.
    let input = picture(
        PixelFormat::Yuv444P10Le,
        16,
        4,
        |p, x, _, _| if p == 0 && x >= 8 { 1023 } else { 512 },
        2,
    );
    let out = Resize::new(40, 4)
        .with_interpolation(Interpolation::Lanczos { a: 3 })
        .apply(&input, params(PixelFormat::Yuv444P10Le, 16, 4))
        .unwrap();
    let y = u16s(&out.planes[0]);
    assert!(y.iter().all(|&v| v <= 1023));
    assert!(y.contains(&1023), "the high plateau reaches full scale");
    assert!(y.iter().any(|&v| v > 600), "values well above 255 are kept");
    // The flat chroma planes are untouched by the luma step.
    assert!(u16s(&out.planes[1]).iter().all(|&v| v == 512));
    assert!(u16s(&out.planes[2]).iter().all(|&v| v == 512));
}

#[test]
fn chroma_planes_resample_on_their_own_grid() {
    // 4:2:2, 9×5: chroma is 5×5. Target 5×3: chroma must be 3×3, taken
    // from the 5-wide chroma gradient — not from a 9-wide luma grid.
    let input = picture(
        PixelFormat::YuvJ422P,
        9,
        5,
        |p, x, _, _| if p == 1 { (x * 50) as u32 } else { 7 },
        1,
    );
    let out = Resize::new(5, 3)
        .with_interpolation(Interpolation::Nearest)
        .apply(&input, params(PixelFormat::YuvJ422P, 9, 5))
        .unwrap();
    assert_eq!(out.planes[0].stride, 5);
    assert_eq!(out.planes[0].data.len(), 15);
    assert_eq!(out.planes[1].stride, 3);
    assert_eq!(out.planes[1].data.len(), 9);
    // Nearest from a 5-wide ramp 0,50,100,150,200 onto 3 columns:
    // centres 0.33, 2.0, 3.67 → 0, 100, 200.
    assert_eq!(&out.planes[1].data[0..3], &[0, 100, 200]);
    assert!(out.planes[2].data.iter().all(|&v| v == 7));

    // 4:2:0 odd both ways, 7×5 → 3×2: chroma 4×3 → 2×1.
    let input = picture(
        PixelFormat::Yuv420P,
        7,
        5,
        |p, _, _, _| p as u32 * 40 + 9,
        1,
    );
    for &k in KERNELS {
        let out = Resize::new(3, 2)
            .with_interpolation(k)
            .apply(&input, params(PixelFormat::Yuv420P, 7, 5))
            .unwrap();
        assert_eq!(
            (out.planes[0].stride, out.planes[0].data.len()),
            (3, 6),
            "{k:?}"
        );
        assert_eq!(
            (out.planes[1].stride, out.planes[1].data.len()),
            (2, 2),
            "{k:?}"
        );
        assert_eq!(
            (out.planes[2].stride, out.planes[2].data.len()),
            (2, 2),
            "{k:?}"
        );
        assert!(out.planes[1].data.iter().all(|&v| v == 49), "{k:?}");
        assert!(out.planes[2].data.iter().all(|&v| v == 89), "{k:?}");
    }
    // 4:4:0 keeps full width, halves height: 8×6 → 4×3 means chroma
    // 8×3 → 4×2.
    let input = picture(PixelFormat::Yuv440P, 8, 6, |_, _, _, _| 3, 1);
    let out = Resize::new(4, 3)
        .with_interpolation(Interpolation::Area)
        .apply(&input, params(PixelFormat::Yuv440P, 8, 6))
        .unwrap();
    assert_eq!((out.planes[1].stride, out.planes[1].data.len()), (4, 8));
}

#[test]
fn float_samples_are_not_clamped() {
    let input = picture(
        PixelFormat::RgbF32Le,
        4,
        4,
        |_, _, _, c| match c {
            0 => 3.5f32.to_bits(),
            1 => (-0.25f32).to_bits(),
            _ => 0.5f32.to_bits(),
        },
        4,
    );
    for &k in KERNELS {
        let out = Resize::new(6, 7)
            .with_interpolation(k)
            .apply(&input, params(PixelFormat::RgbF32Le, 4, 4))
            .unwrap();
        assert_eq!(out.planes[0].stride, 6 * 12, "{k:?}");
        let v: Vec<f32> = out.planes[0]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for px in v.chunks_exact(3) {
            assert!((px[0] - 3.5).abs() < 1e-5, "{k:?}: {px:?}");
            assert!((px[1] + 0.25).abs() < 1e-5, "{k:?}: {px:?}");
            assert!((px[2] - 0.5).abs() < 1e-5, "{k:?}: {px:?}");
        }
    }
}

#[test]
fn semi_planar_pairs_and_deep_packed_pixels_resample_per_component() {
    // Nv12: the chroma plane is (Cb, Cr) pairs; a flat (10, 200) field
    // stays (10, 200) — the two bytes are never averaged together.
    let input = picture(
        PixelFormat::Nv12,
        8,
        6,
        |p, _, _, c| {
            if p == 0 {
                77
            } else if c == 0 {
                10
            } else {
                200
            }
        },
        1,
    );
    for &k in KERNELS {
        let out = Resize::new(5, 3)
            .with_interpolation(k)
            .apply(&input, params(PixelFormat::Nv12, 8, 6))
            .unwrap();
        assert_eq!(out.planes[1].stride, 6, "{k:?}");
        assert_eq!(out.planes[1].data.len(), 12, "{k:?}");
        for pair in out.planes[1].data.chunks_exact(2) {
            assert_eq!(pair, &[10, 200], "{k:?}");
        }
    }
    // Rgb48Le: three 16-bit components per pixel.
    let input = picture(
        PixelFormat::Rgb48Le,
        5,
        4,
        |_, _, _, c| [1u32, 65535, 32768][c],
        2,
    );
    for &k in KERNELS {
        let out = Resize::new(7, 2)
            .with_interpolation(k)
            .apply(&input, params(PixelFormat::Rgb48Le, 5, 4))
            .unwrap();
        assert_eq!(out.planes[0].stride, 7 * 6, "{k:?}");
        for px in u16s(&out.planes[0]).chunks_exact(3) {
            assert_eq!(px, &[1, 65535, 32768], "{k:?}");
        }
    }
}

#[test]
fn pal8_is_nearest_only_and_keeps_its_table() {
    let input = picture(PixelFormat::Pal8, 4, 2, |_, x, _, _| (x % 3) as u32, 1)
        .with_palette(vec![0, 0, 0, 255, 0, 0, 0, 0, 255])
        .with_palette_alpha(vec![255, 255, 0]);
    let out = Resize::new(8, 2)
        .with_interpolation(Interpolation::Nearest)
        .apply(&input, params(PixelFormat::Pal8, 4, 2))
        .unwrap();
    assert_eq!(out.planes[0].stride, 8);
    assert!(out.planes[0].data.iter().all(|&i| i < 3));
    assert_eq!(out.palette(), input.palette());
    assert_eq!(out.palette_alpha(), input.palette_alpha());

    let err = Resize::new(8, 2)
        .with_interpolation(Interpolation::Bilinear)
        .apply(&input, params(PixelFormat::Pal8, 4, 2))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(err.to_string().contains("Pal8"));
    assert!(!Resize::new(1, 1)
        .accepted_pixel_formats()
        .contains(&PixelFormat::Pal8));
    assert!(Resize::new(1, 1)
        .with_interpolation(Interpolation::Nearest)
        .accepted_pixel_formats()
        .contains(&PixelFormat::Pal8));
}

#[test]
fn eight_bit_results_are_unchanged_by_the_float_pipeline() {
    // A 1:1 pass with an interpolating kernel reproduces the input
    // exactly (the kernels interpolate), at every depth.
    for (fmt, bytes) in [
        (PixelFormat::Rgba, 1usize),
        (PixelFormat::Gray12Le, 2),
        (PixelFormat::Gbrap16Le, 2),
    ] {
        let input = picture(
            fmt,
            7,
            5,
            |p, x, y, c| ((p * 97 + x * 31 + y * 17 + c * 5) % 200) as u32,
            bytes,
        );
        for &k in &[
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Lanczos { a: 3 },
        ] {
            let out = Resize::new(7, 5)
                .with_interpolation(k)
                .apply(&input, params(fmt, 7, 5))
                .unwrap();
            for (a, b) in out.planes.iter().zip(&input.planes) {
                assert_eq!(a.data, b.data, "{fmt:?} {k:?}");
            }
        }
    }
}
