//! Issue #12 operations against `magick`, live when the binary is on the
//! path: `-transpose` and `-transverse` on `rose:`, bit-exact.

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{ImageFilter, Transpose, VideoStreamParams};

/// `true` for the reference build (ImageMagick 7.1 Q16-HDRI; other builds clamp or round differently).
fn reference() -> bool {
    magick(&["-version"]).is_some_and(|v| {
        let v = String::from_utf8_lossy(&v).to_string();
        v.contains("ImageMagick 7.1") && v.contains("Q16-HDRI")
    })
}

fn magick(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("magick").args(args).output().ok()?;
    out.status.success().then_some(out.stdout)
}

#[test]
fn transpose_and_transverse_match_magick() {
    let src = if reference() {
        magick(&["rose:", "-depth", "8", "rgb:-"])
    } else {
        None
    };
    let Some(src) = src else {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    };
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 70 * 3,
            data: src,
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 70,
        height: 46,
    };
    for (op, f) in [
        ("-transpose", Transpose::new()),
        ("-transverse", Transpose::transverse()),
    ] {
        let want = magick(&["rose:", op, "-depth", "8", "rgb:-"]).unwrap();
        let got = f.apply(&input, p).unwrap();
        assert_eq!(got.planes[0].stride, 46 * 3);
        assert_eq!(got.planes[0].data, want, "{op}");
    }
}

#[test]
fn sparse_color_matches_magick() {
    use oxideav_image_filter::{SparseColor, SparseMethod, SparsePoint};
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let (w, h) = (31u32, 23u32);
    let black = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 3) as usize,
            data: vec![0; (w * h * 3) as usize],
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: w,
        height: h,
    };
    let sets: [(&str, Vec<SparsePoint>); 3] = [
        (
            "3,2 red  27,5 lime  12,20 blue  29,21 white  15,10 #808000",
            vec![
                SparsePoint::new(3.0, 2.0, [1.0, 0.0, 0.0]),
                SparsePoint::new(27.0, 5.0, [0.0, 1.0, 0.0]),
                SparsePoint::new(12.0, 20.0, [0.0, 0.0, 1.0]),
                SparsePoint::new(29.0, 21.0, [1.0, 1.0, 1.0]),
                SparsePoint::new(15.0, 10.0, [128.0 / 255.0, 128.0 / 255.0, 0.0]),
            ],
        ),
        (
            "2,2 red  28,20 blue",
            vec![
                SparsePoint::new(2.0, 2.0, [1.0, 0.0, 0.0]),
                SparsePoint::new(28.0, 20.0, [0.0, 0.0, 1.0]),
            ],
        ),
        (
            "10,10 lime",
            vec![SparsePoint::new(10.0, 10.0, [0.0, 1.0, 0.0])],
        ),
    ];
    for (spec, pts) in &sets {
        for m in [
            "Barycentric",
            "Bilinear",
            "Shepards",
            "Inverse",
            "Voronoi",
            "Manhattan",
        ] {
            if m == "Bilinear" && pts.len() < 4 {
                continue;
            }
            let want = magick(&[
                "-size",
                &format!("{w}x{h}"),
                "xc:black",
                "-sparse-color",
                m,
                spec,
                "-depth",
                "8",
                "rgb:-",
            ])
            .unwrap();
            let got = SparseColor::new(SparseMethod::from_name(m).unwrap(), pts.clone())
                .unwrap()
                .apply(&black, p)
                .unwrap();
            let worst = got.planes[0]
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap();
            eprintln!("sparse-color {m} [{spec}]: worst {worst}");
            assert!(worst <= 1, "{m} [{spec}]: off by {worst} codes");
        }
    }
}

#[test]
fn connected_components_match_magick() {
    use oxideav_image_filter::connected_components;
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    // A posterised rose: a few colours, many regions.
    let src = magick(&["rose:", "-colors", "5", "+dither", "-depth", "8", "rgb:-"]).unwrap();
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 70 * 3,
            data: src,
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 70,
        height: 46,
    };
    for (conn, thr) in [(4u8, None), (8, None)] {
        let mut args = vec![
            "rose:".to_string(),
            "-colors".into(),
            "5".into(),
            "+dither".into(),
            "-define".into(),
            "connected-components:verbose=true".into(),
        ];
        if let Some(t) = thr {
            args.extend([
                "-define".into(),
                format!("connected-components:area-threshold={t}"),
            ]);
        }
        args.extend([
            "-connected-components".into(),
            conn.to_string(),
            "-depth".into(),
            "16".into(),
            "-endian".into(),
            "LSB".into(),
            "gray:-".into(),
        ]);
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let out = Command::new("magick").args(&a).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        // The verbose listing precedes the raw label plane on stdout.
        let listing_end = text.find("\n\n").map(|i| i + 2);
        let lines: Vec<&str> = text
            .lines()
            .skip(1)
            .take_while(|l| l.starts_with("  "))
            .collect();
        let map = connected_components(&input, p, conn, thr).unwrap();
        assert_eq!(
            map.objects.len(),
            lines.len(),
            "{conn} {thr:?}: object count"
        );
        // magick lists by area; among equal areas its order is not
        // stable, so objects are matched by id.
        for line in &lines {
            let mut it = line.split_whitespace();
            let id: u32 = it.next().unwrap().trim_end_matches(':').parse().unwrap();
            let bbox = it.next().unwrap();
            let centroid = it.next().unwrap();
            let area: u64 = it.next().unwrap().parse().unwrap();
            let o = map
                .objects
                .iter()
                .find(|o| o.id == id)
                .unwrap_or_else(|| panic!("{conn} {thr:?}: no object {id}"));
            assert_eq!(o.area, area, "{conn} {thr:?}: area of {id}");
            assert_eq!(
                format!("{}x{}+{}+{}", o.bbox.2, o.bbox.3, o.bbox.0, o.bbox.1),
                bbox,
                "bbox of {id}"
            );
            assert_eq!(
                format!("{:.1},{:.1}", o.centroid.0, o.centroid.1),
                centroid,
                "centroid of {id}"
            );
        }
        let _ = listing_end;
        let areas: Vec<u64> = map.objects.iter().map(|o| o.area).collect();
        assert!(areas.windows(2).all(|w| w[0] >= w[1]), "sorted by area");
        eprintln!(
            "connected-components {conn} {thr:?}: {} objects match",
            lines.len()
        );
    }
}

#[test]
fn noise_statistics_match_magick() {
    use oxideav_image_filter::{AddNoise, NoiseType};
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let (w, h) = (400u32, 400u32);
    for t in [
        "Uniform",
        "Gaussian",
        "Multiplicative",
        "Impulse",
        "Laplacian",
        "Poisson",
        "Random",
    ] {
        for (g, att) in [(25u32, 1.0f64), (50, 1.0), (75, 2.0), (50, 0.5)] {
            let out = Command::new("magick")
                .args([
                    "-size",
                    &format!("{w}x{h}"),
                    &format!("xc:gray({g}%)"),
                    "-seed",
                    "7",
                    "-attenuate",
                    &att.to_string(),
                    "+noise",
                    t,
                    "-format",
                    "%[fx:mean] %[fx:standard_deviation]",
                    "info:",
                ])
                .output()
                .unwrap();
            let s = String::from_utf8_lossy(&out.stdout).to_string();
            let mut it = s.split_whitespace().map(|v| v.parse::<f64>().unwrap());
            let (want_mean, want_std) = (it.next().unwrap(), it.next().unwrap());
            let v = g as f32 / 100.0;
            let mut data = Vec::with_capacity((w * h * 4) as usize);
            for _ in 0..w * h {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let f = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: (w * 4) as usize,
                    data,
                }],
            };
            let got = AddNoise::new(NoiseType::from_name(t).unwrap())
                .with_attenuate(att)
                .with_seed(7)
                .apply(
                    &f,
                    VideoStreamParams {
                        format: PixelFormat::GrayF32Le,
                        width: w,
                        height: h,
                    },
                )
                .unwrap();
            let xs: Vec<f64> = got.planes[0]
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c) as f64)
                .collect();
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            eprintln!(
                "noise {t} {g}% att {att}: mean {mean:.4} / {want_mean:.4}, std {std:.5} / {want_std:.5}"
            );
            assert!(
                (mean - want_mean).abs() < 0.004,
                "{t} {g}% {att}: mean {mean} vs {want_mean}"
            );
            assert!(
                (std - want_std).abs() <= 0.03 * want_std + 1e-4,
                "{t} {g}% {att}: std {std} vs {want_std}"
            );
        }
    }
}
