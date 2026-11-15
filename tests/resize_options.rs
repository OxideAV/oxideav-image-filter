//! The resize options beyond the kernel choice: cylindrical application
//! (one elliptical gather per output pixel instead of two separable
//! passes) and linear-light resampling, on every sample depth.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    FilterOptions, ImageFilter, Interpolation, ResampleFilter, Resize, VideoStreamParams,
    with_execution_context,
};

fn params(format: PixelFormat, width: u32, height: u32) -> VideoStreamParams {
    VideoStreamParams {
        format,
        width,
        height,
    }
}

/// A frame of `fmt` whose every plane is filled with the byte pattern
/// `fill(plane)` repeated.
fn filled(fmt: PixelFormat, w: u32, h: u32, fill: impl Fn(usize) -> Vec<u8>) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            VideoPlane {
                stride,
                data: fill(p)
                    .into_iter()
                    .cycle()
                    .take(stride * rows as usize)
                    .collect(),
            }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

/// A packed frame from a per-pixel function giving normalised samples.
fn packed(
    fmt: PixelFormat,
    w: u32,
    h: u32,
    ch: usize,
    bytes: usize,
    f: impl Fn(u32, u32) -> [f32; 4],
) -> VideoFrame {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for v in &f(x, y)[..ch] {
                match bytes {
                    1 => data.push((v * 255.0).round() as u8),
                    2 => data.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes()),
                    _ => data.extend_from_slice(&v.to_le_bytes()),
                }
            }
        }
    }
    let _ = fmt;
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * ch * bytes,
            data,
        }],
    }
}

fn sample(plane: &VideoPlane, i: usize, bytes: usize) -> f32 {
    match bytes {
        1 => plane.data[i] as f32 / 255.0,
        2 => u16::from_le_bytes([plane.data[2 * i], plane.data[2 * i + 1]]) as f32 / 65535.0,
        _ => f32::from_le_bytes(plane.data[4 * i..4 * i + 4].try_into().unwrap()),
    }
}

const KERNELS: [Interpolation; 9] = [
    Interpolation::Bilinear,
    Interpolation::Bicubic,
    Interpolation::Area,
    Interpolation::Lanczos { a: 3 },
    Interpolation::Mitchell,
    Interpolation::Filter(ResampleFilter::Robidoux),
    Interpolation::Filter(ResampleFilter::Lanczos2Sharp),
    Interpolation::Filter(ResampleFilter::Gaussian),
    Interpolation::Filter(ResampleFilter::Jinc),
];

#[test]
fn constant_pictures_stay_constant_on_every_layout_and_option() {
    let layouts: [(PixelFormat, Vec<u8>); 12] = [
        (PixelFormat::Gray8, vec![77]),
        (PixelFormat::Rgb24, vec![10, 120, 250]),
        (PixelFormat::Rgba, vec![10, 120, 250, 99]),
        (PixelFormat::Argb, vec![99, 10, 120, 250]),
        (PixelFormat::Ya8, vec![60, 200]),
        (PixelFormat::Cmyk, vec![1, 2, 3, 4]),
        (PixelFormat::Yuv420P, vec![90]),
        (PixelFormat::Nv12, vec![90]),
        (PixelFormat::Gray16Le, vec![0x34, 0x12]),
        (PixelFormat::Rgba64Le, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (PixelFormat::Yuv444P10Le, vec![0x21, 0x02]),
        (PixelFormat::Gbrap16Le, vec![0x99, 0x55]),
    ];
    for (fmt, bytes) in layouts {
        let input = filled(fmt, 24, 18, |_| bytes.clone());
        for interp in KERNELS {
            for (cyl, lin) in [(true, false), (false, true), (true, true)] {
                for (w, h) in [(9u32, 7u32), (40, 31), (24, 18)] {
                    let out = Resize::new(w, h)
                        .with_interpolation(interp)
                        .with_cylindrical(cyl)
                        .with_linear_light(lin)
                        .apply(&input, params(fmt, 24, 18))
                        .unwrap_or_else(|e| panic!("{fmt:?} {interp:?}: {e}"));
                    assert_eq!(out.planes.len(), input.planes.len());
                    for (p, plane) in out.planes.iter().enumerate() {
                        let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
                        assert_eq!(plane.stride, fmt.plane_row_bytes(p, w).unwrap());
                        assert_eq!(plane.data.len(), plane.stride * rows as usize);
                        // A conversion on the way (linear light on YUV /
                        // CMYK) may move a code by one.
                        let exact = !lin
                            || !matches!(
                                fmt,
                                PixelFormat::Cmyk
                                    | PixelFormat::Yuv420P
                                    | PixelFormat::Nv12
                                    | PixelFormat::Yuv444P10Le
                            );
                        let word = if bytes.len() % 2 == 0
                            && fmt != PixelFormat::Ya8
                            && fmt != PixelFormat::Cmyk
                            && fmt != PixelFormat::Rgba
                            && fmt != PixelFormat::Argb
                        {
                            2
                        } else {
                            1
                        };
                        for (i, got) in plane.data.chunks_exact(word).enumerate() {
                            let want = &input.planes[p].data[(i * word) % bytes.len().max(word)..];
                            let (g, w_) = if word == 2 {
                                (
                                    u16::from_le_bytes([got[0], got[1]]) as i32,
                                    u16::from_le_bytes([want[0], want[1]]) as i32,
                                )
                            } else {
                                (got[0] as i32, want[0] as i32)
                            };
                            let slack = if exact { 0 } else { 2 };
                            assert!(
                                (g - w_).abs() <= slack,
                                "{fmt:?} {interp:?} cyl {cyl} lin {lin} {w}x{h} plane {p} sample {i}: {g} vs {w_}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cylindrical_application_is_isotropic_where_separable_is_not() {
    // One bright pixel enlarged eight times samples the kernel on a fine
    // grid around it. Two output pixels at the same distance from the
    // impulse — offsets (1.5, 10.5) / 8 and (7.5, 7.5) / 8, both 1.33
    // pixels away — get the same weight from a radial kernel and
    // different ones from a separable product (except for the Gaussian,
    // whose product is itself radial).
    let n = 9u32;
    let input = packed(PixelFormat::GrayF32Le, n, n, 1, 4, |x, y| {
        [if (x, y) == (4, 4) { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]
    });
    let p = params(PixelFormat::GrayF32Le, n, n);
    let centre = 8 * 4 + 4; // the output pixel just past the impulse centre
    let at = |out: &VideoFrame, dx: usize, dy: usize| {
        sample(&out.planes[0], (centre + dy) * 72 + centre + dx, 4)
    };
    for filter in [
        ResampleFilter::Mitchell,
        ResampleFilter::Robidoux,
        ResampleFilter::Gaussian,
        ResampleFilter::Lanczos,
    ] {
        let cyl = Resize::new(72, 72)
            .with_filter(filter)
            .with_cylindrical(true)
            .apply(&input, p)
            .unwrap();
        let sep = Resize::new(72, 72)
            .with_filter(filter)
            .apply(&input, p)
            .unwrap();
        let (a, b) = (at(&cyl, 1, 10), at(&cyl, 7, 7));
        assert!((a - b).abs() < 0.004, "{filter:?} cylindrical: {a} vs {b}");
        let (a, b) = (at(&sep, 1, 10), at(&sep, 7, 7));
        if filter != ResampleFilter::Gaussian {
            assert!((a - b).abs() > 0.015, "{filter:?} separable: {a} vs {b}");
        }
        // Both keep the picture's sum (a unit impulse spreads to 64).
        for out in [&cyl, &sep] {
            let sum: f32 = (0..72 * 72).map(|i| sample(&out.planes[0], i, 4)).sum();
            assert!((sum - 64.0).abs() < 2.5, "{filter:?}: sum {sum}");
        }
    }
}

#[test]
fn cylindrical_shrink_does_not_alias_and_tracks_the_separable_result() {
    // A one-pixel checkerboard shrinks to flat grey.
    let board = packed(PixelFormat::Gray8, 120, 120, 1, 1, |x, y| {
        [((x + y) % 2) as f32, 0.0, 0.0, 0.0]
    });
    let p = params(PixelFormat::Gray8, 120, 120);
    for interp in KERNELS {
        let out = Resize::new(37, 23)
            .with_interpolation(interp)
            .with_cylindrical(true)
            .apply(&board, p)
            .unwrap();
        for (i, &v) in out.planes[0].data.iter().enumerate() {
            // (A radial cubic with negative lobes passes a little of the
            // diagonal pattern.)
            assert!((v as i32 - 128).abs() <= 10, "{interp:?} sample {i}: {v}");
        }
    }
    // On a smooth picture the two applications of one kernel agree
    // closely away from the border.
    let smooth = packed(PixelFormat::Rgb48Le, 96, 72, 3, 2, |x, y| {
        let (fx, fy) = (x as f32 / 96.0, y as f32 / 72.0);
        [
            0.2 + 0.6 * fx,
            0.5 + 0.3 * (5.0 * fx).sin() * (4.0 * fy).cos(),
            0.2 + 0.6 * fy,
            0.0,
        ]
    });
    let p = params(PixelFormat::Rgb48Le, 96, 72);
    for interp in [
        Interpolation::Mitchell,
        Interpolation::Lanczos { a: 3 },
        Interpolation::Filter(ResampleFilter::Robidoux),
    ] {
        for (w, h) in [(41usize, 30usize), (150, 110)] {
            let r = Resize::new(w as u32, h as u32).with_interpolation(interp);
            let sep = r.apply(&smooth, p).unwrap();
            let cyl = r.with_cylindrical(true).apply(&smooth, p).unwrap();
            for y in 4..h - 4 {
                for x in 4..w - 4 {
                    for c in 0..3 {
                        let i = (y * w + x) * 3 + c;
                        let (a, b) = (sample(&sep.planes[0], i, 2), sample(&cyl.planes[0], i, 2));
                        assert!((a - b).abs() < 0.01, "{interp:?} {w}x{h}: {a} vs {b}");
                    }
                }
            }
        }
    }
}

#[test]
fn linear_light_averages_light_not_codes() {
    // Black and white in equal parts: the mean of the codes is 0.5; the
    // mean of the light is 0.5 linear, which encodes to about 0.735.
    let want_linear = 1.055 * 0.5f32.powf(1.0 / 2.4) - 0.055;
    for (fmt, ch, bytes) in [
        (PixelFormat::Gray8, 1usize, 1usize),
        (PixelFormat::Rgba, 4, 1),
        (PixelFormat::Rgb48Le, 3, 2),
        (PixelFormat::Rgba64Le, 4, 2),
        (PixelFormat::RgbaF32Le, 4, 4),
    ] {
        let board = packed(fmt, 64, 64, ch, bytes, |x, y| {
            let v = ((x + y) % 2) as f32;
            // Alpha alternates too: it must average as plain coverage.
            if ch == 4 { [v, v, v, 1.0] } else { [v; 4] }
        });
        let p = params(fmt, 64, 64);
        for cyl in [false, true] {
            for (lin, want) in [(false, 0.5f32), (true, want_linear)] {
                let out = Resize::new(16, 16)
                    .with_filter(ResampleFilter::Triangle)
                    .with_filter_options(FilterOptions::new().with_blur(1.5))
                    .with_cylindrical(cyl)
                    .with_linear_light(lin)
                    .apply(&board, p)
                    .unwrap();
                for y in 2..14usize {
                    for x in 2..14usize {
                        let got = sample(&out.planes[0], (y * 16 + x) * ch, bytes);
                        assert!(
                            (got - want).abs() < 0.02,
                            "{fmt:?} cyl {cyl} lin {lin}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
    // Alpha is coverage, never a tone: a half-covered pattern keeps 0.5.
    let fade = packed(PixelFormat::Rgba, 64, 64, 4, 1, |x, y| {
        [0.8, 0.8, 0.8, ((x + y) % 2) as f32]
    });
    let out = Resize::new(16, 16)
        .with_interpolation(Interpolation::Area)
        .with_linear_light(true)
        .apply(&fade, params(PixelFormat::Rgba, 64, 64))
        .unwrap();
    for px in out.planes[0].data.as_chunks::<4>().0 {
        assert!((px[3] as i32 - 128).abs() <= 1, "alpha {}", px[3]);
        assert!((px[0] as i32 - 204).abs() <= 1, "colour {}", px[0]);
    }
    // Scene-referred float samples above 1 keep their range.
    let bright = packed(PixelFormat::GrayF32Le, 8, 8, 1, 4, |_, _| {
        [4.0, 0.0, 0.0, 0.0]
    });
    let out = Resize::new(3, 3)
        .with_interpolation(Interpolation::Mitchell)
        .with_linear_light(true)
        .apply(&bright, params(PixelFormat::GrayF32Le, 8, 8))
        .unwrap();
    for i in 0..9 {
        assert!((sample(&out.planes[0], i, 4) - 4.0).abs() < 1e-3);
    }
}

#[test]
fn coverage_weighting_holds_on_the_cylindrical_path() {
    // Opaque blue beside fully transparent red: red never shows.
    for (fmt, bytes, order) in [
        (PixelFormat::Rgba, 1usize, [0usize, 1, 2, 3]),
        (PixelFormat::Rgba64Le, 2, [0, 1, 2, 3]),
        (PixelFormat::Argb, 1, [1, 2, 3, 0]),
    ] {
        let input = packed(fmt, 32, 32, 4, bytes, |x, _| {
            let rgba: [f32; 4] = if x < 16 {
                [0.0, 0.0, 1.0, 1.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            let mut px = [0.0; 4];
            for (c, &slot) in order.iter().enumerate() {
                px[slot] = rgba[c];
            }
            px
        });
        for (w, h) in [(11u32, 13u32), (70, 50)] {
            for lin in [false, true] {
                let out = Resize::new(w, h)
                    .with_filter(ResampleFilter::Robidoux)
                    .with_cylindrical(true)
                    .with_linear_light(lin)
                    .apply(&input, params(fmt, 32, 32))
                    .unwrap();
                for i in 0..(w * h) as usize {
                    let px: Vec<f32> = (0..4)
                        .map(|c| sample(&out.planes[0], i * 4 + order[c], bytes))
                        .collect();
                    if px[3] > 0.02 {
                        assert!(
                            px[0] < 0.02 && px[2] > 0.98,
                            "{fmt:?} {w}x{h} lin {lin}: {px:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn options_give_the_same_output_at_every_thread_budget() {
    let input = packed(PixelFormat::Rgba64Le, 61, 47, 4, 2, |x, y| {
        let h = (x * 7919 + y * 104_729) % 251;
        [
            h as f32 / 251.0,
            (h * 3 % 251) as f32 / 251.0,
            (h * 5 % 251) as f32 / 251.0,
            0.3 + 0.7 * (h * 11 % 251) as f32 / 251.0,
        ]
    });
    let p = params(PixelFormat::Rgba64Le, 61, 47);
    for (w, h) in [(20u32, 19u32), (97, 80)] {
        for (cyl, lin) in [(true, false), (false, true), (true, true)] {
            let r = Resize::new(w, h)
                .with_filter(ResampleFilter::Lanczos)
                .with_cylindrical(cyl)
                .with_linear_light(lin);
            let serial = r.apply(&input, p).unwrap();
            for threads in [2, 7] {
                let ctx = ExecutionContext::with_threads(threads);
                let par = with_execution_context(&ctx, || r.apply(&input, p)).unwrap();
                assert_eq!(serial.planes[0].data, par.planes[0].data);
            }
        }
    }
}
