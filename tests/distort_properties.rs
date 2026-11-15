//! Properties of the distortion engine that hold whatever the method:
//! identity maps are exact, a map followed by its inverse returns the
//! picture, constant pictures stay constant, the engine agrees with the
//! dedicated resize / rotate filters, area resampling does not alias
//! under minification, colour is averaged weighted by coverage, and the
//! output is the same at every thread budget and on every sample depth.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::deep::FloatImage;
use oxideav_image_filter::{
    DistortImage, DistortMethod, FilterOptions, ImageFilter, Interpolation, PixelInterpolate,
    ResampleFilter, Resize, ReverseMap, Rotate, VideoStreamParams, VirtualPixel,
    with_execution_context,
};

use DistortMethod as M;

/// A smooth RGBA picture with gentle curvature and varying alpha.
fn smooth(w: u32, h: u32) -> FloatImage {
    let mut img = FloatImage::new(w, h, 4);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            img.pixel_mut(x, y).copy_from_slice(&[
                0.2 + 0.6 * fx,
                0.2 + 0.6 * fy,
                0.5 + 0.3 * (6.0 * fx).sin() * (5.0 * fy).cos(),
                1.0,
            ]);
        }
    }
    img
}

/// Deterministic noise (xorshift), every channel, alpha included.
fn noise(w: u32, h: u32, channels: usize) -> FloatImage {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let data = (0..w as usize * h as usize * channels)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect();
    FloatImage::from_samples(w, h, channels, data).unwrap()
}

fn constant(w: u32, h: u32, px: &[f32]) -> FloatImage {
    let mut img = FloatImage::new(w, h, px.len());
    for y in 0..h {
        for x in 0..w {
            img.pixel_mut(x, y).copy_from_slice(px);
        }
    }
    img
}

fn max_diff_inside(a: &FloatImage, b: &FloatImage, margin: u32) -> f32 {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    let mut worst = 0.0f32;
    for y in margin..a.height() - margin {
        for x in margin..a.width() - margin {
            for (p, q) in a.pixel(x, y).iter().zip(b.pixel(x, y)) {
                worst = worst.max((p - q).abs());
            }
        }
    }
    worst
}

const RESAMPLINGS: [Option<PixelInterpolate>; 8] = [
    None, // EWA
    Some(PixelInterpolate::Nearest),
    Some(PixelInterpolate::Integer),
    Some(PixelInterpolate::Bilinear),
    Some(PixelInterpolate::Catrom),
    Some(PixelInterpolate::Mesh),
    Some(PixelInterpolate::Spline),
    Some(PixelInterpolate::Average),
];

fn with_resampling(d: DistortImage, r: Option<PixelInterpolate>) -> DistortImage {
    match r {
        None => d,
        Some(i) => d.with_interpolate(i),
    }
}

/// Every method with arguments that leave the picture where it is.
fn identities(w: f64, h: f64) -> Vec<(M, Vec<f64>)> {
    let corners = vec![
        0.0, 0.0, 0.0, 0.0, w, 0.0, w, 0.0, w, h, w, h, 0.0, h, 0.0, h,
    ];
    vec![
        (M::AffineProjection, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        (M::ScaleRotateTranslate, vec![0.0]),
        (M::Affine, vec![3.0, 4.0, 3.0, 4.0]),
        (M::Resize, vec![w, h]),
        (M::ScaleRotateTranslate, vec![1.0, 360.0]),
        (
            M::PerspectiveProjection,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
        (M::Barrel, vec![0.0, 0.0, 0.0, 1.0]),
        (M::BarrelInverse, vec![0.0, 0.0, 0.0, 1.0]),
        (M::Affine, corners.clone()),
        (M::Perspective, corners.clone()),
        (M::BilinearForward, corners.clone()),
        (M::BilinearReverse, corners.clone()),
        (M::Shepards, corners.clone()),
        (M::Polynomial, [vec![1.0], corners].concat()),
    ]
}

#[test]
fn whole_pixel_shifts_are_exact_under_every_resampling() {
    let img = noise(23, 17, 4);
    for (method, args) in identities(23.0, 17.0).into_iter().take(4) {
        for r in RESAMPLINGS {
            // Spline and the averages are smoothing reads by definition.
            if matches!(
                r,
                Some(PixelInterpolate::Spline) | Some(PixelInterpolate::Average)
            ) {
                continue;
            }
            let d = with_resampling(DistortImage::new(method, args.clone()), r)
                .distort(&img)
                .unwrap();
            assert_eq!(d.image, img, "{method:?} {args:?} {r:?}");
        }
    }
    // A translation by whole pixels moves samples untouched, EWA included.
    let d = DistortImage::new(M::Affine, vec![0.0, 0.0, 3.0, -2.0])
        .with_virtual_pixel(VirtualPixel::Tile)
        .distort(&img)
        .unwrap();
    for y in 0..17u32 {
        for x in 0..23u32 {
            let (sx, sy) = ((x + 23 - 3) % 23, (y + 2) % 17);
            assert_eq!(d.image.pixel(x, y), img.pixel(sx, sy));
        }
    }
}

#[test]
fn identity_arguments_of_every_method_return_the_picture() {
    let img = smooth(40, 30);
    for (method, args) in identities(40.0, 30.0) {
        for interp in [
            PixelInterpolate::Nearest,
            PixelInterpolate::Bilinear,
            PixelInterpolate::Catrom,
            PixelInterpolate::Mesh,
        ] {
            let d = DistortImage::new(method, args.clone())
                .with_interpolate(interp)
                .distort(&img)
                .unwrap();
            let worst = max_diff_inside(&d.image, &img, 0);
            assert!(worst < 2e-4, "{method:?} {interp:?}: {worst}");
        }
    }
}

#[test]
fn a_map_followed_by_its_inverse_returns_the_picture() {
    let img = smooth(96, 72);
    let quad = [
        0.0, 0.0, 6.0, 4.0, 96.0, 0.0, 90.0, 2.0, 96.0, 72.0, 94.0, 70.0, 0.0, 72.0, 3.0, 66.0,
    ];
    let swapped: Vec<f64> = quad
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|c| [c[2], c[3], c[0], c[1]])
        .collect();
    // Plane2Cylinder under `fov` is undone by Cylinder2Plane under the
    // angle the same strip then spans: w / f with f = (w/2) / tan(fov/2).
    let fov = 60.0f64;
    let strip = (2.0 * (fov.to_radians() / 2.0).tan()).to_degrees();
    let pairs: Vec<(&str, DistortImage, DistortImage, f32)> = vec![
        (
            "srt",
            DistortImage::new(M::ScaleRotateTranslate, vec![0.8, 23.0]),
            DistortImage::new(M::ScaleRotateTranslate, vec![1.25, -23.0]),
            0.01,
        ),
        (
            "affine projection",
            DistortImage::new(M::AffineProjection, vec![1.1, 0.1, -0.2, 0.9, 4.0, -3.0]),
            {
                // Inverse of [[1.1, -0.2], [0.1, 0.9]] and its offset.
                let det = 1.1 * 0.9 + 0.2 * 0.1;
                let (a, b, c, d) = (0.9 / det, 0.2 / det, -0.1 / det, 1.1 / det);
                DistortImage::new(
                    M::AffineProjection,
                    vec![a, c, b, d, -(a * 4.0 + b * -3.0), -(c * 4.0 + d * -3.0)],
                )
            },
            0.01,
        ),
        (
            "perspective",
            DistortImage::new(M::Perspective, quad.to_vec()),
            DistortImage::new(M::Perspective, swapped.clone()),
            0.01,
        ),
        (
            "bilinear",
            // The destination as a bilinear function of the source is
            // undone by reading the source through that same function.
            DistortImage::new(M::BilinearForward, quad.to_vec()),
            DistortImage::new(M::BilinearReverse, swapped.clone()),
            0.01,
        ),
        (
            "cylinder",
            DistortImage::new(M::Plane2Cylinder, vec![fov]),
            DistortImage::new(M::Cylinder2Plane, vec![strip]),
            0.01,
        ),
    ];
    for (name, there, back, tol) in pairs {
        for r in [
            None,
            Some(PixelInterpolate::Catrom),
            Some(PixelInterpolate::Bilinear),
        ] {
            let mid = with_resampling(there.clone(), r).distort(&img).unwrap();
            let out = with_resampling(back.clone(), r)
                .distort(&mid.image)
                .unwrap();
            let worst = max_diff_inside(&out.image, &img, 24);
            // The default area filter is not interpolating: two passes
            // soften the picture's curvature a little.
            let tol = if r.is_none() { 6.0 * tol } else { tol };
            assert!(worst < tol, "{name} {r:?}: {worst}");
        }
    }
    // Polar and DePolar: the radius / angle grid is not area-preserving,
    // so the round trip is held on a band away from the centre and rim.
    let polar = DistortImage::new(M::DePolar, vec![0.0])
        .with_interpolate(PixelInterpolate::Catrom)
        .distort(&img)
        .unwrap();
    let back = DistortImage::new(M::Polar, vec![0.0])
        .with_interpolate(PixelInterpolate::Catrom)
        .distort(&polar.image)
        .unwrap();
    let mut worst = 0.0f32;
    for y in 0..72u32 {
        for x in 0..96u32 {
            let r = ((x as f32 + 0.5 - 48.0).powi(2) + (y as f32 + 0.5 - 36.0).powi(2)).sqrt();
            if (12.0..30.0).contains(&r) {
                for (p, q) in back.image.pixel(x, y).iter().zip(img.pixel(x, y)) {
                    worst = worst.max((p - q).abs());
                }
            }
        }
    }
    assert!(worst < 0.03, "polar round trip: {worst}");
}

/// Every method with arguments that really move the picture.
fn warps(w: f64, h: f64) -> Vec<(M, Vec<f64>)> {
    let quad = vec![
        0.0,
        0.0,
        w * 0.1,
        h * 0.05,
        w,
        0.0,
        w * 0.9,
        0.0,
        w,
        h,
        w,
        h,
        0.0,
        h,
        0.0,
        h * 0.8,
    ];
    let mut six = quad.clone();
    six.extend([w / 2.0, h / 2.0, w / 2.0 + 2.0, h / 2.0 - 1.0]);
    six.extend([w / 2.0, 0.0, w / 2.0 - 1.0, 2.0]);
    vec![
        (M::Affine, vec![0.0, 0.0, 3.5, 2.5]),
        (M::Affine, quad[..12].to_vec()),
        (M::AffineProjection, vec![0.9, 0.2, -0.1, 1.1, 5.0, -3.0]),
        (M::ScaleRotateTranslate, vec![0.4, 20.0]),
        (M::ScaleRotateTranslate, vec![2.3, -70.0]),
        (M::Perspective, quad.clone()),
        (
            M::PerspectiveProjection,
            vec![1.1, 0.1, -3.0, 0.05, 0.9, 2.0, 0.002, 0.001],
        ),
        (M::BilinearForward, quad.clone()),
        (M::BilinearReverse, quad.clone()),
        (M::Polynomial, [vec![2.0], six].concat()),
        (M::Arc, vec![120.0]),
        (M::Polar, vec![0.0]),
        (M::DePolar, vec![0.0]),
        (M::Barrel, vec![0.1, -0.05, 0.02]),
        (M::BarrelInverse, vec![0.05, 0.0, -0.03]),
        (
            M::Shepards,
            vec![10.0, 10.0, 14.0, 8.0, 30.0, 20.0, 26.0, 23.0],
        ),
        (M::Cylinder2Plane, vec![90.0]),
        (M::Plane2Cylinder, vec![90.0]),
        (M::Resize, vec![(w * 0.37).round(), (h * 1.6).round()]),
    ]
}

#[test]
fn constant_pictures_stay_constant() {
    let colours: [&[f32]; 3] = [&[0.25, 0.5, 0.75, 0.6], &[0.1, 0.9, 0.4], &[0.33]];
    for px in colours {
        let img = constant(48, 36, px);
        for (method, args) in warps(48.0, 36.0) {
            for r in RESAMPLINGS {
                let d = with_resampling(DistortImage::new(method, args.clone()), r)
                    .with_bestfit(matches!(method, M::ScaleRotateTranslate))
                    .distort(&img)
                    .unwrap();
                assert_eq!(d.image.channels(), px.len());
                for (i, got) in d.image.samples().chunks_exact(px.len()).enumerate() {
                    for (g, want) in got.iter().zip(px) {
                        assert!(
                            (g - want).abs() < 1e-5,
                            "{method:?} {args:?} {r:?} pixel {i}: {got:?}"
                        );
                    }
                }
            }
        }
    }
}

fn frame_of(img: &FloatImage, fmt: PixelFormat) -> (VideoFrame, VideoStreamParams) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (ch, bytes): (usize, usize) = match fmt {
        PixelFormat::Gray8 => (1, 1),
        PixelFormat::Rgb24 => (3, 1),
        PixelFormat::Rgba => (4, 1),
        PixelFormat::Gray16Le => (1, 2),
        PixelFormat::Rgb48Le => (3, 2),
        PixelFormat::Rgba64Le => (4, 2),
        PixelFormat::GrayF32Le => (1, 4),
        PixelFormat::RgbF32Le => (3, 4),
        PixelFormat::RgbaF32Le => (4, 4),
        other => panic!("no packer for {other:?}"),
    };
    let mut data = Vec::with_capacity(w * h * ch * bytes);
    for px in img.samples().as_chunks::<4>().0 {
        if ch == 1 {
            push(&mut data, FloatImage::luma(&px[..3]), bytes);
        } else {
            for &v in &px[..3] {
                push(&mut data, v, bytes);
            }
        }
        if ch == 4 {
            push(&mut data, px[3], bytes);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w * ch * bytes,
                data,
            }],
        },
        VideoStreamParams {
            format: fmt,
            width: w as u32,
            height: h as u32,
        },
    )
}

fn push(data: &mut Vec<u8>, v: f32, bytes: usize) {
    match bytes {
        1 => data.push((v * 255.0).round() as u8),
        2 => data.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes()),
        _ => data.extend_from_slice(&v.to_le_bytes()),
    }
}

fn sample(plane: &VideoPlane, i: usize, bytes: usize) -> f32 {
    match bytes {
        1 => plane.data[i] as f32 / 255.0,
        2 => u16::from_le_bytes([plane.data[2 * i], plane.data[2 * i + 1]]) as f32 / 65535.0,
        _ => f32::from_le_bytes(plane.data[4 * i..4 * i + 4].try_into().unwrap()),
    }
}

const DEPTH_LAYOUTS: [(PixelFormat, usize, usize); 9] = [
    (PixelFormat::Gray8, 1, 1),
    (PixelFormat::Rgb24, 3, 1),
    (PixelFormat::Rgba, 4, 1),
    (PixelFormat::Gray16Le, 1, 2),
    (PixelFormat::Rgb48Le, 3, 2),
    (PixelFormat::Rgba64Le, 4, 2),
    (PixelFormat::GrayF32Le, 1, 4),
    (PixelFormat::RgbF32Le, 3, 4),
    (PixelFormat::RgbaF32Le, 4, 4),
];

#[test]
fn every_depth_and_packed_layout_runs_every_method() {
    let img = smooth(40, 30);
    for (fmt, ch, bytes) in DEPTH_LAYOUTS {
        let (frame, params) = frame_of(&img, fmt);
        // The identity is exact on the stored codes.
        for r in [
            None,
            Some(PixelInterpolate::Bilinear),
            Some(PixelInterpolate::Catrom),
        ] {
            let out = with_resampling(DistortImage::new(M::ScaleRotateTranslate, vec![0.0]), r)
                .apply(&frame, params)
                .unwrap();
            assert_eq!(out.planes[0].data, frame.planes[0].data, "{fmt:?} {r:?}");
        }
        // Every method keeps the layout and extent, and a rotation agrees
        // across depths to within the coarser depth's quantisation.
        for (method, args) in warps(40.0, 30.0) {
            let filter = DistortImage::new(method, args.clone());
            let out = filter
                .apply(&frame, params)
                .unwrap_or_else(|e| panic!("{fmt:?} {method:?}: {e}"));
            let (ow, oh) = filter.output_extent(40, 30).unwrap();
            assert_eq!(
                out.planes[0].data.len(),
                ow as usize * oh as usize * ch * bytes,
                "{fmt:?} {method:?}"
            );
        }
        let rotated = DistortImage::new(M::ScaleRotateTranslate, vec![0.8, 17.0])
            .apply(&frame, params)
            .unwrap();
        let reference = DistortImage::new(M::ScaleRotateTranslate, vec![0.8, 17.0])
            .distort(&img)
            .unwrap()
            .image;
        let tol = if bytes == 1 { 0.012 } else { 0.004 };
        for y in 0..30usize {
            for x in 0..40usize {
                let want = reference.pixel(x as u32, y as u32);
                let want_c: Vec<f32> = match ch {
                    1 => vec![0.299 * want[0] + 0.587 * want[1] + 0.114 * want[2]],
                    3 => want[..3].to_vec(),
                    _ => want.to_vec(),
                };
                for (c, w) in want_c.iter().enumerate() {
                    let got = sample(&rotated.planes[0], (y * 40 + x) * ch + c, bytes);
                    assert!(
                        (got - w).abs() < tol,
                        "{fmt:?} ({x},{y}) channel {c}: {got} vs {w}"
                    );
                }
            }
        }
    }
}

#[test]
fn planar_and_reordered_layouts_reach_the_engine() {
    // The layouts outside the packed set are converted on the way in and
    // back on the way out; a constant picture must come back constant and
    // the label must hold.
    for fmt in [
        PixelFormat::Yuv420P,
        PixelFormat::Yuv444P,
        PixelFormat::Nv12,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Ya8,
        PixelFormat::Gbrp8,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Gbrap16Le,
        PixelFormat::Ya16Le,
    ] {
        let params = VideoStreamParams {
            format: fmt,
            width: 32,
            height: 24,
        };
        let planes: Vec<VideoPlane> = (0..fmt.plane_count())
            .map(|p| {
                let (w, h) = fmt.plane_dimensions(p, 32, 24).unwrap();
                let stride = fmt.plane_row_bytes(p, 32).unwrap();
                let _ = w;
                let bytes = match fmt {
                    PixelFormat::Yuv420P10Le => vec![0x00u8, 0x02],
                    PixelFormat::Gbrap16Le | PixelFormat::Ya16Le => vec![0x00, 0x80],
                    _ => vec![0x80],
                };
                VideoPlane {
                    stride,
                    data: bytes
                        .iter()
                        .copied()
                        .cycle()
                        .take(stride * h as usize)
                        .collect(),
                }
            })
            .collect();
        let frame = VideoFrame { pts: None, planes };
        let out = DistortImage::new(M::ScaleRotateTranslate, vec![1.3, 25.0])
            .apply(&frame, params)
            .unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
        assert_eq!(out.planes.len(), frame.planes.len(), "{fmt:?}");
        for (got, want) in out.planes.iter().zip(&frame.planes) {
            assert_eq!(got.data.len(), want.data.len(), "{fmt:?}");
            let bytes = if want.data[1] == want.data[0] { 1 } else { 2 };
            for (g, w) in got
                .data
                .chunks_exact(bytes)
                .zip(want.data.chunks_exact(bytes))
            {
                let (g, w) = if bytes == 1 {
                    (g[0] as i32, w[0] as i32)
                } else {
                    (
                        u16::from_le_bytes([g[0], g[1]]) as i32,
                        u16::from_le_bytes([w[0], w[1]]) as i32,
                    )
                };
                assert!((g - w).abs() <= 2, "{fmt:?}: {g} vs {w}");
            }
        }
    }
}

#[test]
fn the_engine_agrees_with_the_dedicated_resize() {
    let img = smooth(40, 30);
    let (frame, params) = frame_of(&img, PixelFormat::RgbaF32Le);
    // Point reads: the same half-pixel-centred positions, the same
    // bilinear blend, the same edge replication.
    for (w, h) in [(80u32, 60u32), (53, 41), (40, 30)] {
        let dedicated = Resize::new(w, h)
            .with_interpolation(Interpolation::Bilinear)
            .apply(&frame, params)
            .unwrap();
        let engine = DistortImage::new(M::Resize, vec![w as f64, h as f64])
            .with_interpolate(PixelInterpolate::Bilinear)
            .distort(&img)
            .unwrap()
            .image;
        assert_eq!((engine.width(), engine.height()), (w, h));
        for (i, want) in engine.samples().iter().enumerate() {
            let got = sample(&dedicated.planes[0], i, 4);
            assert!((got - want).abs() < 1e-4, "{w}x{h} sample {i}");
        }
    }
    // Area reads on a shrink: the separable and the cylindrical use of
    // the same kernel are different filters, but both are low-pass at
    // the output's resolution; on a smooth picture they agree closely.
    for filter in [
        ResampleFilter::Triangle,
        ResampleFilter::Mitchell,
        ResampleFilter::Gaussian,
    ] {
        let dedicated = Resize::new(20, 15)
            .with_filter(filter)
            .apply(&frame, params)
            .unwrap();
        let engine = DistortImage::new(M::Resize, vec![20.0, 15.0])
            .with_ewa(filter)
            .distort(&img)
            .unwrap()
            .image;
        for y in 2..13usize {
            for x in 2..18usize {
                for c in 0..4 {
                    let i = (y * 20 + x) * 4 + c;
                    let got = sample(&dedicated.planes[0], i, 4);
                    assert!(
                        (got - engine.samples()[i]).abs() < 0.02,
                        "{filter:?} ({x},{y},{c})"
                    );
                }
            }
        }
    }
}

#[test]
fn the_engine_agrees_with_the_dedicated_rotate() {
    let img = smooth(64, 48);
    let (frame, params) = frame_of(&img, PixelFormat::RgbF32Le);
    for degrees in [20.0f32, -35.0, 90.0] {
        let dedicated = Rotate::new(degrees).apply(&frame, params).unwrap();
        let stride = dedicated.planes[0].stride;
        let (dw, dh) = (stride / 12, dedicated.planes[0].data.len() / stride);
        // The same canvas, centred on the picture's centre.
        let engine = DistortImage::new(M::ScaleRotateTranslate, vec![degrees as f64])
            .with_interpolate(PixelInterpolate::Bilinear)
            .with_virtual_pixel(VirtualPixel::Black)
            .with_viewport(
                dw as u32,
                dh as u32,
                (64 - dw as i64).div_euclid(2),
                (48 - dh as i64).div_euclid(2),
            )
            .distort(&img)
            .unwrap()
            .image;
        // Compare where both read the picture's interior (the canvas
        // parity can differ by half a pixel; the picture's slope bounds
        // what that costs).
        let (s, c) = (degrees as f64).to_radians().sin_cos();
        let mut worst = 0.0f32;
        let mut compared = 0;
        for y in 0..dh {
            for x in 0..dw {
                let (dx, dy) = (
                    x as f64 + 0.5 - dw as f64 / 2.0,
                    y as f64 + 0.5 - dh as f64 / 2.0,
                );
                let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
                if u.abs() < 26.0 && v.abs() < 18.0 {
                    compared += 1;
                    for ch in 0..3 {
                        let got = sample(&dedicated.planes[0], (y * dw + x) * 3 + ch, 4);
                        let want = engine.pixel(x as u32, y as u32)[ch];
                        worst = worst.max((got - want).abs());
                    }
                }
            }
        }
        assert!(compared > 1000 && worst < 0.03, "{degrees}: {worst}");
    }
}

fn spread(img: &FloatImage, margin: u32) -> (f32, f32) {
    let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for y in margin..img.height() - margin {
        for x in margin..img.width() - margin {
            let v = img.pixel(x, y)[0] as f64;
            sum += v;
            sq += v * v;
            n += 1.0;
        }
    }
    let mean = sum / n;
    (mean as f32, (sq / n - mean * mean).max(0.0).sqrt() as f32)
}

#[test]
fn area_resampling_does_not_alias_under_minification() {
    // A one-pixel checkerboard carries all its energy at the Nyquist
    // limit: shrunk, it must become flat mid-grey. Point reads keep
    // picking single squares instead.
    let mut img = FloatImage::new(240, 240, 1);
    for y in 0..240u32 {
        for x in 0..240u32 {
            img.pixel_mut(x, y)[0] = ((x + y) % 2) as f32;
        }
    }
    for (scale, angle) in [(0.31, 0.0), (0.22, 13.0), (0.09, 31.0), (0.021, 5.0)] {
        let srt = DistortImage::new(M::ScaleRotateTranslate, vec![scale, angle]);
        let n = (240.0 * scale * 0.6) as u32;
        let view = srt
            .clone()
            .with_viewport(n, n, 120 - n as i64 / 2, 120 - n as i64 / 2);
        let (mean, dev) = spread(&view.distort(&img).unwrap().image, 0);
        assert!(
            (mean - 0.5).abs() < 0.02 && dev < 0.02,
            "ewa {scale}: mean {mean} dev {dev}"
        );
        if scale > 0.05 {
            let point = view.with_interpolate(PixelInterpolate::Bilinear);
            let (_, dev) = spread(&point.distort(&img).unwrap().image, 0);
            assert!(dev > 0.05, "point {scale}: dev {dev}");
        }
    }
    // Under perspective the far rows are minified many times over and
    // must be as flat as the near ones are sharp.
    let quad = vec![
        0.0, 0.0, 100.0, 60.0, 240.0, 0.0, 140.0, 60.0, 240.0, 240.0, 240.0, 240.0, 0.0, 240.0,
        0.0, 240.0,
    ];
    let out = DistortImage::new(M::Perspective, quad)
        .distort(&img)
        .unwrap()
        .image;
    let mut far = Vec::new();
    for y in 62..70u32 {
        for x in 110..130u32 {
            far.push(out.pixel(x, y)[0]);
        }
    }
    let mean = far.iter().sum::<f32>() / far.len() as f32;
    let dev = (far.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / far.len() as f32).sqrt();
    assert!(
        (mean - 0.5).abs() < 0.03 && dev < 0.03,
        "far rows: mean {mean} dev {dev}"
    );
}

#[test]
fn colour_is_averaged_weighted_by_coverage() {
    // Opaque blue on the left, fully transparent red on the right: no
    // read may pick up red.
    let mut img = FloatImage::new(32, 32, 4);
    for y in 0..32u32 {
        for x in 0..32u32 {
            let px: [f32; 4] = if x < 16 {
                [0.0, 0.0, 1.0, 1.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            img.pixel_mut(x, y).copy_from_slice(&px);
        }
    }
    for r in RESAMPLINGS {
        for args in [vec![0.5, 33.0], vec![1.7, -12.0], vec![0.05, 3.0]] {
            let out = with_resampling(DistortImage::new(M::ScaleRotateTranslate, args.clone()), r)
                .distort(&img)
                .unwrap()
                .image;
            for px in out.samples().as_chunks::<4>().0 {
                if px[3] > 1e-3 {
                    assert!(
                        px[0].abs() < 1e-3 && (px[2] - 1.0).abs() < 1e-3,
                        "{r:?} {args:?}: {px:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn output_is_identical_at_every_thread_budget() {
    let img = noise(97, 61, 4);
    for (method, args) in warps(97.0, 61.0) {
        for r in [None, Some(PixelInterpolate::Catrom)] {
            let d = with_resampling(DistortImage::new(method, args.clone()), r);
            let serial = d.distort(&img).unwrap().image;
            for threads in [2, 5, 16] {
                let ctx = ExecutionContext::with_threads(threads);
                let par = with_execution_context(&ctx, || d.distort(&img))
                    .unwrap()
                    .image;
                assert_eq!(serial, par, "{method:?} {r:?} at {threads} threads");
            }
        }
    }
}

struct Ripple;

impl ReverseMap for Ripple {
    fn source(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if x > 60.0 {
            return None;
        }
        Some((x + 3.0 * (y / 9.0).sin(), y))
    }
}

#[test]
fn a_callers_mapping_runs_through_the_engine() {
    let img = smooth(64, 48);
    // A closure is a mapping; the identity is exact with point reads.
    let same = DistortImage::engine()
        .with_interpolate(PixelInterpolate::Catrom)
        .distort_with(&img, &|x: f64, y: f64| Some((x, y)))
        .unwrap();
    assert_eq!(same.image, img);
    // A mapping without derivatives of its own gets them by central
    // difference; where it is undefined the matte colour is written.
    let matte = [0.1, 0.2, 0.3, 1.0];
    for r in [None, Some(PixelInterpolate::Bilinear)] {
        let out = with_resampling(DistortImage::engine(), r)
            .with_matte(matte)
            .distort_with(&img, &Ripple)
            .unwrap()
            .image;
        assert_eq!(out.pixel(62, 10), &matte);
        // Row 0 has no shift: column reads are the picture's own.
        let worst = (0..60u32)
            .map(|x| (out.pixel(x, 0)[0] - img.pixel(x, 0)[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.01, "{r:?}: {worst}");
        // Row 14 is shifted by 3 sin(14.5 / 9) ≈ +3 pixels.
        let shift = 3.0 * (14.5f32 / 9.0).sin();
        let want = 0.2 + 0.6 * (20.0 + shift) / 64.0;
        assert!((out.pixel(20, 14)[0] - want).abs() < 0.01, "{r:?}");
    }
    // A viewport and output scaling apply to a caller's mapping too.
    let zoom = DistortImage::engine()
        .with_viewport(16, 12, 8, 6)
        .with_scale(2.0)
        .with_interpolate(PixelInterpolate::Nearest)
        .distort_with(&img, &|x: f64, y: f64| Some((x, y)))
        .unwrap();
    assert_eq!((zoom.image.width(), zoom.image.height()), (32, 24));
    assert_eq!(zoom.offset, (8, 6));
    assert_eq!(zoom.image.pixel(5, 3), img.pixel(8 + 2, 6 + 1));
}

#[test]
fn viewports_bestfit_and_output_scaling_size_the_output() {
    let img = smooth(40, 30);
    // An explicit window.
    let d = DistortImage::new(M::ScaleRotateTranslate, vec![0.0])
        .with_viewport(10, 8, 5, 4)
        .with_interpolate(PixelInterpolate::Nearest)
        .distort(&img)
        .unwrap();
    assert_eq!(
        (d.image.width(), d.image.height(), d.offset),
        (10, 8, (5, 4))
    );
    assert_eq!(d.image.pixel(0, 0), img.pixel(5, 4));
    // Best fit of a quarter turn swaps the extent (plus the half-pixel
    // rim the rule adds on each side).
    let d = DistortImage::new(M::ScaleRotateTranslate, vec![90.0])
        .with_bestfit(true)
        .distort(&img)
        .unwrap();
    assert!((30..=32).contains(&d.image.width()) && (40..=42).contains(&d.image.height()));
    // Best fit of the methods with a forward map contains the whole
    // picture: no corner of the source lands outside the window.
    for (method, args) in warps(40.0, 30.0) {
        let d = DistortImage::new(method, args.clone()).with_bestfit(true);
        let out = d.distort(&img).unwrap();
        assert_eq!(
            d.output_extent(40, 30).unwrap(),
            (out.image.width(), out.image.height()),
            "{method:?}"
        );
    }
    // Output scaling renders the same window on a finer grid.
    let base = DistortImage::new(M::ScaleRotateTranslate, vec![1.0, 15.0])
        .with_interpolate(PixelInterpolate::Bilinear);
    let fine = base.clone().with_scale(3.0).distort(&img).unwrap().image;
    assert_eq!((fine.width(), fine.height()), (120, 90));
    let coarse = base.distort(&img).unwrap().image;
    // The centre sample of each 3 × 3 block sits on the coarse pixel's
    // centre.
    for y in 0..30u32 {
        for x in 0..40u32 {
            for (p, q) in fine
                .pixel(3 * x + 1, 3 * y + 1)
                .iter()
                .zip(coarse.pixel(x, y))
            {
                assert!((p - q).abs() < 1e-5);
            }
        }
    }
    // Resize fixes its own extent.
    let d = DistortImage::new(M::Resize, vec![17.0, 9.0])
        .distort(&img)
        .unwrap();
    assert_eq!((d.image.width(), d.image.height()), (17, 9));
}

#[test]
fn polynomial_orders_recover_a_polynomial_warp() {
    // Control points sampled from a known cubic warp: an order-3 fit (and
    // any higher order with enough points) reproduces it, so the warped
    // picture equals the one rendered from the warp itself.
    let warp = |x: f64, y: f64| {
        let (a, b) = (x / 64.0 - 0.5, y / 48.0 - 0.5);
        (
            x + 3.0 * a * a * b - 2.0 * b * b + 1.5 * a * a * a,
            y + 2.0 * a * b - 2.5 * b * b * b + a,
        )
    };
    let img = smooth(64, 48);
    let direct = DistortImage::engine()
        .with_interpolate(PixelInterpolate::Bilinear)
        .distort_with(&img, &|x: f64, y: f64| Some(warp(x, y)))
        .unwrap()
        .image;
    for order in [3.0, 4.0, 6.0] {
        let mut args = vec![order];
        for j in 0..=8 {
            for i in 0..=8 {
                let (x, y) = (i as f64 * 8.0, j as f64 * 6.0);
                let (u, v) = warp(x, y);
                args.extend([u, v, x, y]);
            }
        }
        let fitted = DistortImage::new(M::Polynomial, args)
            .with_interpolate(PixelInterpolate::Bilinear)
            .distort(&img)
            .unwrap()
            .image;
        let worst = max_diff_inside(&fitted, &direct, 0);
        assert!(worst < 2e-3, "order {order}: {worst}");
    }
    // Too few points for the order is a typed refusal.
    assert!(
        DistortImage::new(
            M::Polynomial,
            vec![3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        .distort(&img)
        .is_err()
    );
}

#[test]
fn shepards_power_sharpens_the_pull_of_each_point() {
    let img = smooth(64, 48);
    let args = vec![20.0, 20.0, 26.0, 20.0, 44.0, 30.0, 40.0, 30.0];
    // Every power moves each control point exactly.
    for power in [1.0, 2.0, 4.0] {
        let out = DistortImage::new(M::Shepards, args.clone())
            .with_shepards_power(power)
            .with_interpolate(PixelInterpolate::Bilinear)
            .with_viewport(1, 1, 26, 20)
            .with_scale(1.0)
            .distort(&img)
            .unwrap()
            .image;
        // Output pixel (26, 20) has its centre half a pixel past the
        // control point; it reads within a pixel of source (20, 20).
        let want = img.pixel(20, 20)[0];
        assert!((out.pixel(0, 0)[0] - want).abs() < 0.02, "power {power}");
    }
    let soft = DistortImage::new(M::Shepards, args.clone())
        .with_shepards_power(1.0)
        .distort(&img)
        .unwrap()
        .image;
    let hard = DistortImage::new(M::Shepards, args)
        .with_shepards_power(6.0)
        .distort(&img)
        .unwrap()
        .image;
    assert_ne!(soft, hard);
}

#[test]
fn ewa_filter_overrides_change_the_footprint() {
    let mut img = FloatImage::new(33, 33, 1);
    img.pixel_mut(16, 16)[0] = 1.0;
    let at = |d: DistortImage| {
        let out = d.distort(&img).unwrap().image;
        (out.pixel(16, 16)[0], out.pixel(18, 16)[0])
    };
    // A fractional shift so the kernel is really evaluated.
    let shift = vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.0];
    let plain =
        at(DistortImage::new(M::AffineProjection, shift.clone())
            .with_ewa(ResampleFilter::Gaussian));
    let wide = at(DistortImage::new(M::AffineProjection, shift.clone())
        .with_ewa(ResampleFilter::Gaussian)
        .with_filter_options(FilterOptions::new().with_blur(2.0)));
    assert!(wide.0 < plain.0 && wide.1 > plain.1, "{plain:?} {wide:?}");
    // A windowed sinc in radial form has its first zero at the jinc's
    // (≈ 1.22), not at 1: a neighbour one pixel away keeps weight.
    let mut line = FloatImage::new(33, 33, 1);
    line.pixel_mut(16, 16)[0] = 1.0;
    let lanczos = DistortImage::new(M::AffineProjection, vec![1.0, 0.0, 0.0, 1.0, 1e-9, 0.0])
        .with_ewa(ResampleFilter::Lanczos)
        .distort(&line)
        .unwrap()
        .image;
    assert!(lanczos.pixel(17, 16)[0] > 0.02);
    // LanczosRadius keeps a whole-pixel radius of three.
    let radius = DistortImage::new(M::AffineProjection, vec![1.0, 0.0, 0.0, 1.0, 1e-9, 0.0])
        .with_ewa(ResampleFilter::LanczosRadius)
        .distort(&line)
        .unwrap()
        .image;
    assert!(radius.pixel(19, 16)[0].abs() < 1e-6);
    assert!(lanczos.pixel(19, 16)[0].abs() > 1e-4);
}
