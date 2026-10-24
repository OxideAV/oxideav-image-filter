//! `-distort` against `magick` 7.1.2 Q16-HDRI, live when the reference
//! build is on the path. Point sampling (`-filter point -interpolate X`)
//! is held to 45 dB (35 dB where a perspective horizon crosses the
//! picture); EWA (the default resampler) to 35 dB — measured 55-73 dB on
//! scaled / rotated pictures, 36 dB across a horizon.

use std::process::Command;

use oxideav_image_filter::deep::FloatImage;
use oxideav_image_filter::{
    DistortImage, DistortMethod, PixelInterpolate, ResampleFilter, VirtualPixel,
};

fn reference() -> bool {
    Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| {
            let v = String::from_utf8_lossy(&o.stdout).to_string();
            o.status.success()
                && v.contains("ImageMagick 7.1.2")
                && v.contains("Q16-HDRI")
                && !cfg!(windows)
        })
        .unwrap_or(false)
}

fn magick_rgba(args: &[&str]) -> (Vec<f32>, u32, u32) {
    let mut a: Vec<&str> = args.to_vec();
    a.extend([
        "-depth", "16", "-endian", "LSB", "-format", "%w %h\n", "-write", "info:-", "rgba:-",
    ]);
    let out = Command::new("magick").args(&a).output().unwrap();
    assert!(
        out.status.success(),
        "magick {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let nl = out.stdout.iter().position(|&b| b == b'\n').unwrap();
    let hdr = String::from_utf8_lossy(&out.stdout[..nl]).to_string();
    let mut it = hdr.split_whitespace().map(|v| v.parse::<u32>().unwrap());
    let (w, h) = (it.next().unwrap(), it.next().unwrap());
    let data: Vec<f32> = out.stdout[nl + 1..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c) as f32 / 65535.0)
        .collect();
    (data, w, h)
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// `(magick method, args string, our method, args, bestfit, threshold dB)`.
type Case = (
    &'static str,
    &'static str,
    DistortMethod,
    Vec<f64>,
    bool,
    f64,
);

fn cases() -> Vec<Case> {
    use DistortMethod as M;
    vec![
        (
            "SRT",
            "30",
            M::ScaleRotateTranslate,
            vec![30.0],
            false,
            45.0,
        ),
        (
            "SRT",
            "1.3 -20",
            M::ScaleRotateTranslate,
            vec![1.3, -20.0],
            false,
            45.0,
        ),
        (
            "SRT",
            "20,10 0.8,1.2 15 40,30",
            M::ScaleRotateTranslate,
            vec![20.0, 10.0, 0.8, 1.2, 15.0, 40.0, 30.0],
            false,
            45.0,
        ),
        (
            "AffineProjection",
            "0.9,0.2,-0.1,1.1,5,-3",
            M::AffineProjection,
            vec![0.9, 0.2, -0.1, 1.1, 5.0, -3.0],
            false,
            45.0,
        ),
        (
            "Affine",
            "0,0 5,3  70,0 60,8  0,46 4,40",
            M::Affine,
            vec![
                0.0, 0.0, 5.0, 3.0, 70.0, 0.0, 60.0, 8.0, 0.0, 46.0, 4.0, 40.0,
            ],
            false,
            45.0,
        ),
        (
            "Affine",
            "10,10 12,14  60,30 55,35",
            M::Affine,
            vec![10.0, 10.0, 12.0, 14.0, 60.0, 30.0, 55.0, 35.0],
            false,
            45.0,
        ),
        (
            "Perspective",
            "0,0 6,4  70,0 62,0  70,46 70,46  0,46 0,40",
            M::Perspective,
            vec![
                0.0, 0.0, 6.0, 4.0, 70.0, 0.0, 62.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                40.0,
            ],
            false,
            45.0,
        ),
        // A horizon crosses the picture: magick blends the pixels next to
        // it towards the matte colour, which is not reproduced (35 dB).
        (
            "Perspective",
            "0,0 20,10  70,0 50,0  70,46 70,46  0,46 0,30",
            M::Perspective,
            vec![
                0.0, 0.0, 20.0, 10.0, 70.0, 0.0, 50.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                30.0,
            ],
            false,
            35.0,
        ),
        (
            "PerspectiveProjection",
            "1.1,0.1,-3,0.05,0.9,2,0.002,0.001",
            M::PerspectiveProjection,
            vec![1.1, 0.1, -3.0, 0.05, 0.9, 2.0, 0.002, 0.001],
            false,
            45.0,
        ),
        (
            "BilinearReverse",
            "0,0 6,4  70,0 62,0  70,46 70,46  0,46 0,40",
            M::BilinearReverse,
            vec![
                0.0, 0.0, 6.0, 4.0, 70.0, 0.0, 62.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                40.0,
            ],
            false,
            45.0,
        ),
        (
            "BilinearForward",
            "0,0 6,4  70,0 62,0  70,46 70,46  0,46 0,40",
            M::BilinearForward,
            vec![
                0.0, 0.0, 6.0, 4.0, 70.0, 0.0, 62.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                40.0,
            ],
            false,
            45.0,
        ),
        (
            "Polynomial",
            "2  0,0 3,2  70,0 66,1  70,46 69,44  0,46 2,45  35,23 37,21  35,0 34,3",
            M::Polynomial,
            vec![
                2.0, 0.0, 0.0, 3.0, 2.0, 70.0, 0.0, 66.0, 1.0, 70.0, 46.0, 69.0, 44.0, 0.0, 46.0,
                2.0, 45.0, 35.0, 23.0, 37.0, 21.0, 35.0, 0.0, 34.0, 3.0,
            ],
            false,
            45.0,
        ),
        (
            "Barrel",
            "0.1 -0.05 0.02",
            M::Barrel,
            vec![0.1, -0.05, 0.02],
            false,
            45.0,
        ),
        (
            "BarrelInverse",
            "0.05 0.0 -0.03",
            M::BarrelInverse,
            vec![0.05, 0.0, -0.03],
            false,
            45.0,
        ),
        (
            "Shepards",
            "20,20 25,18  50,30 45,33",
            M::Shepards,
            vec![20.0, 20.0, 25.0, 18.0, 50.0, 30.0, 45.0, 33.0],
            false,
            45.0,
        ),
        ("Polar", "0", M::Polar, vec![0.0], false, 45.0),
        ("DePolar", "0", M::DePolar, vec![0.0], false, 45.0),
        ("Arc", "60", M::Arc, vec![60.0], false, 45.0),
        (
            "Arc",
            "120 30 50 10",
            M::Arc,
            vec![120.0, 30.0, 50.0, 10.0],
            false,
            45.0,
        ),
        ("Arc", "90 0 60", M::Arc, vec![90.0, 0.0, 60.0], false, 45.0),
        ("Polar", "30 5", M::Polar, vec![30.0, 5.0], false, 45.0),
        (
            "DePolar",
            "40 0 30,20",
            M::DePolar,
            vec![40.0, 0.0, 30.0, 20.0],
            false,
            45.0,
        ),
        ("SRT", "25", M::ScaleRotateTranslate, vec![25.0], true, 45.0),
        (
            "Perspective",
            "0,0 6,4  70,0 62,0  70,46 70,46  0,46 0,40",
            M::Perspective,
            vec![
                0.0, 0.0, 6.0, 4.0, 70.0, 0.0, 62.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                40.0,
            ],
            true,
            45.0,
        ),
    ]
}

fn rose() -> FloatImage {
    let (data, w, h) = magick_rgba(&["rose:", "-alpha", "on"]);
    FloatImage::from_samples(w, h, 4, data).unwrap()
}

#[test]
fn distort_methods_with_point_sampling_match_magick() {
    if !reference() {
        eprintln!("reference magick not on the path: live distort comparison skipped");
        return;
    }
    let src = rose();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for (name, args, method, ours, bestfit, db_min) in cases() {
        let op = if bestfit { "+distort" } else { "-distort" };
        let (want, w, h) = magick_rgba(&[
            "rose:",
            "-alpha",
            "on",
            "-virtual-pixel",
            "black",
            "-filter",
            "point",
            "-interpolate",
            "bilinear",
            op,
            name,
            args,
        ]);
        let got = DistortImage::new(method, ours)
            .with_bestfit(bestfit)
            .with_virtual_pixel(VirtualPixel::Black)
            .with_interpolate(PixelInterpolate::Bilinear)
            .distort(&src);
        let line = match got {
            Err(e) => format!("{op} {name} {args}: error {e}"),
            Ok(d) if (d.image.width(), d.image.height()) != (w, h) => {
                format!(
                    "{op} {name} {args}: size {}x{} vs magick {w}x{h}",
                    d.image.width(),
                    d.image.height()
                )
            }
            Ok(d) => {
                let db = psnr(d.image.samples(), &want);
                if db < db_min {
                    failures.push(format!("{op} {name} {args}: {db:.1} dB"));
                }
                format!("{op} {name} {args}: {db:.1} dB")
            }
        };
        if line.contains("error") || line.contains("size") {
            failures.push(line.clone());
        }
        report.push(line);
    }
    eprintln!("{}", report.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn interpolation_and_virtual_pixel_methods_match_magick() {
    if !reference() {
        eprintln!("reference magick not on the path: live distort comparison skipped");
        return;
    }
    let src = rose();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for interp in [
        "integer",
        "nearest",
        "average",
        "average9",
        "average16",
        "bilinear",
        "catrom",
        "spline",
    ] {
        let (want, _, _) = magick_rgba(&[
            "rose:",
            "-alpha",
            "on",
            "-virtual-pixel",
            "edge",
            "-filter",
            "point",
            "-interpolate",
            interp,
            "-distort",
            "SRT",
            "1.37 17",
        ]);
        let got = DistortImage::new(DistortMethod::ScaleRotateTranslate, vec![1.37, 17.0])
            .with_virtual_pixel(VirtualPixel::Edge)
            .with_interpolate(PixelInterpolate::from_name(interp).unwrap())
            .distort(&src)
            .unwrap();
        let db = psnr(got.image.samples(), &want);
        report.push(format!("interpolate {interp}: {db:.1} dB"));
        if db < 45.0 {
            failures.push(format!("interpolate {interp}: {db:.1} dB"));
        }
    }
    for vp in [
        "edge",
        "black",
        "white",
        "gray",
        "transparent",
        "tile",
        "mirror",
        "horizontaltile",
        "verticaltile",
        "horizontaltileedge",
        "verticaltileedge",
        "checkertile",
        "background",
    ] {
        let (want, _, _) = magick_rgba(&[
            "rose:",
            "-alpha",
            "on",
            "-background",
            "#3366cc",
            "-virtual-pixel",
            vp,
            "-filter",
            "point",
            "-interpolate",
            "nearest",
            "-distort",
            "SRT",
            "0.45 33",
        ]);
        let got = DistortImage::new(DistortMethod::ScaleRotateTranslate, vec![0.45, 33.0])
            .with_virtual_pixel(VirtualPixel::from_name(vp).unwrap())
            .with_background([0.2, 0.4, 0.8, 1.0])
            .with_interpolate(PixelInterpolate::Nearest)
            .distort(&src)
            .unwrap();
        let db = psnr(got.image.samples(), &want);
        report.push(format!("virtual-pixel {vp}: {db:.1} dB"));
        if db < 45.0 {
            failures.push(format!("virtual-pixel {vp}: {db:.1} dB"));
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn ewa_resampling_tracks_magick() {
    if !reference() {
        eprintln!("reference magick not on the path: live distort comparison skipped");
        return;
    }
    let src = rose();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for (name, args, method, ours, db_min) in [
        (
            "SRT",
            "0.4 20",
            DistortMethod::ScaleRotateTranslate,
            vec![0.4, 20.0],
            35.0,
        ),
        (
            "SRT",
            "1.8 -10",
            DistortMethod::ScaleRotateTranslate,
            vec![1.8, -10.0],
            35.0,
        ),
        (
            "Perspective",
            "0,0 20,10  70,0 50,0  70,46 70,46  0,46 0,30",
            DistortMethod::Perspective,
            vec![
                0.0, 0.0, 20.0, 10.0, 70.0, 0.0, 50.0, 0.0, 70.0, 46.0, 70.0, 46.0, 0.0, 46.0, 0.0,
                30.0,
            ],
            35.0,
        ),
    ] {
        for filter in ["Robidoux", "Jinc"] {
            let (want, _, _) = magick_rgba(&[
                "rose:",
                "-alpha",
                "on",
                "-virtual-pixel",
                "black",
                "-filter",
                filter,
                "-distort",
                name,
                args,
            ]);
            let got = DistortImage::new(method, ours.clone())
                .with_virtual_pixel(VirtualPixel::Black)
                .with_ewa(ResampleFilter::from_name(filter).unwrap())
                .distort(&src)
                .unwrap();
            let db = psnr(got.image.samples(), &want);
            report.push(format!("ewa {filter} {name} {args}: {db:.1} dB"));
            if db < db_min {
                failures.push(format!(
                    "ewa {filter} {name} {args}: {db:.1} dB (< {db_min})"
                ));
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
