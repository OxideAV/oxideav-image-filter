//! Issue #12 operations against `magick`, live when the binary is on the
//! path: `-transpose` and `-transverse` on `rose:`, bit-exact.

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{ImageFilter, Transpose, VideoStreamParams};

/// `true` for the reference build (ImageMagick 7.1.2 Q16-HDRI, as on
/// macOS / Linux). The Windows runner's build ranks `-evaluate-sequence
/// Median` differently and clamps between steps, so it is skipped there.
fn reference() -> bool {
    !cfg!(windows)
        && magick(&["-version"]).is_some_and(|v| {
            let v = String::from_utf8_lossy(&v).to_string();
            v.contains("ImageMagick 7.1.2") && v.contains("Q16-HDRI")
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

#[test]
fn fx_expressions_match_magick() {
    use oxideav_image_filter::Fx;
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let float_rgb = |args: &[&str]| {
        let mut a: Vec<&str> = args.to_vec();
        a.extend([
            "-depth",
            "32",
            "-define",
            "quantum:format=floating-point",
            "-endian",
            "LSB",
            "rgb:-",
        ]);
        magick(&a).unwrap()
    };
    let src = float_rgb(&["rose:"]);
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 70 * 12,
            data: src,
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::RgbF32Le,
        width: 70,
        height: 46,
    };
    let exprs = [
        "u*0.5",
        "1-u",
        "u^2",
        "-u^2+1",
        "2^-1*u",
        "sin(u*pi)",
        "i/w",
        "j/h",
        "(i+j)%7/7",
        "r*0.3+g*0.6+b*0.1",
        "u>0.5?1:0",
        "max(r,b)",
        "min(i,j)/w",
        "p[-1,0]",
        "p[2,-3]",
        "p{10,10}",
        "p{i/2,j/2}",
        "p[1,0].g",
        "u.r",
        "abs(u-0.5)*2",
        "hypot(i-w/2,j-h/2)/w",
        "floor(u*4)/4",
        "atan2(j-h/2,i-w/2)/pi/2+0.5",
        "exp(-u)",
        "log(u+1)",
        "ln(u+1)",
        "logtwo(u+1)",
        "sqrt(u)",
        "pow(u,0.5)",
        "!(u<0.5)",
        "u<0.3||u>0.7",
        "u>0.21&&u<0.61",
        "intensity",
        "u==p[1,0]",
        "u!=p[1,0]",
        "round(u*3)/3",
        "int(u*5)/5",
        "mod(i,5)/5",
        "squish(u*8-4)",
        "1e-1*i/w*10",
        "cosh(u)-1",
        "tanh(u*2)",
        "sign(u-0.5)",
        "QuantumRange*u",
        "u*(1-u)*4",
    ];
    let mut bad = Vec::new();
    for e in exprs {
        let want = float_rgb(&["rose:", "-fx", e]);
        let got = Fx::new(e).unwrap().apply(&input, p).unwrap();
        let worst = got.planes[0]
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(want.as_chunks::<4>().0)
            .map(|(a, b)| {
                let (a, b) = (f32::from_le_bytes(*a), f32::from_le_bytes(*b));
                (a - b).abs() / b.abs().max(1.0)
            })
            .fold(0f32, f32::max);
        eprintln!("fx {e:28}: worst relative |Δ| {worst:.2e}");
        if worst.is_nan() || worst > 1e-4 {
            bad.push(format!("{e}: {worst}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn kmeans_from_seeds_matches_magick() {
    use oxideav_image_filter::Kmeans;
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let src = magick(&["rose:", "-depth", "8", "rgb:-"]).unwrap();
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
    for (spec, seeds) in [
        (
            "red;green;blue",
            vec![[1.0, 0.0, 0.0], [0.0, 128.0 / 255.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (
            "black;white;gray;yellow",
            vec![[0.0; 3], [1.0; 3], [190.0 / 255.0; 3], [1.0, 1.0, 0.0]],
        ),
    ] {
        let want = magick(&[
            "rose:",
            "-define",
            &format!("kmeans:seed-colors={spec}"),
            "-kmeans",
            &seeds.len().to_string(),
            "-depth",
            "8",
            "rgb:-",
        ])
        .unwrap();
        let got = Kmeans::new(seeds.len())
            .with_seeds(seeds.clone())
            .apply(&input, p)
            .unwrap();
        let g = &got.planes[0].data;
        let worst = g
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let off = g.iter().zip(&want).filter(|(a, b)| a != b).count();
        // Pixels assigned to a different cluster than magick's (more than
        // a few codes away).
        let moved = g
            .as_chunks::<3>()
            .0
            .iter()
            .zip(want.as_chunks::<3>().0)
            .filter(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| (*x as i32 - *y as i32).abs() > 8)
            })
            .count();
        eprintln!(
            "kmeans [{spec}]: worst {worst}, {off} samples differ, {moved} pixels in another cluster"
        );
        // Exact from three seeds; from four, magick's assignment drifts
        // from plain Lloyd's on a few boundary pixels (≤ 1 %, centroids
        // within ~1 code) — not pinned.
        assert!(
            moved * 100 <= 70 * 46,
            "kmeans [{spec}]: {moved} pixels in another cluster"
        );
        if seeds.len() == 3 {
            assert_eq!(worst, 0, "kmeans [{spec}]");
        }
    }
}

/// `rose:` with a vertical alpha ramp, as 8-bit RGBA, plus the path of a
/// raw copy `magick` can read back (`-size 70x46 -depth 8 rgba:<path>`).
fn rose_rgba(tag: &str) -> Option<(Vec<u8>, std::path::PathBuf)> {
    if !reference() {
        return None;
    }
    let data = magick(&[
        "rose:",
        "(",
        "-size",
        "70x46",
        "gradient:white-gray20",
        ")",
        "-alpha",
        "off",
        "-compose",
        "CopyOpacity",
        "-composite",
        "-depth",
        "8",
        "rgba:-",
    ])?;
    let path = std::env::temp_dir().join(format!(
        "oxideav-image-filter-{tag}-{}.rgba",
        std::process::id()
    ));
    std::fs::write(&path, &data).ok()?;
    Some((data, path))
}

/// Run `ops` on the raw RGBA picture at `path` and read 8-bit RGBA back.
fn magick_rgba(path: &std::path::Path, ops: &[&str]) -> Vec<u8> {
    try_magick_rgba(path, ops).expect("magick ran")
}

/// [`magick_rgba`], `None` when `magick` reports an error.
fn try_magick_rgba(path: &std::path::Path, ops: &[&str]) -> Option<Vec<u8>> {
    let src = format!("rgba:{}", path.display());
    let mut args = vec!["-size", "70x46", "-depth", "8", src.as_str()];
    args.extend_from_slice(ops);
    args.extend_from_slice(&["-depth", "8", "rgba:-"]);
    magick(&args)
}

fn worst(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0)
}

const ROSE_RGBA: VideoStreamParams = VideoStreamParams {
    format: PixelFormat::Rgba,
    width: 70,
    height: 46,
};

fn rgba_frame(data: &[u8]) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 70 * 4,
            data: data.to_vec(),
        }],
    }
}

#[test]
fn threshold_family_matches_magick() {
    use oxideav_image_filter::{
        BlackThreshold, ChannelMask, ColorThreshold, RangeThreshold, WhiteThreshold,
    };
    let Some((src, path)) = rose_rgba("thresholds") else {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    };
    let input = rgba_frame(&src);
    let rg = ChannelMask::new(true, true, false, false);
    let cases: Vec<(Vec<&str>, Box<dyn ImageFilter>)> = vec![
        (
            vec!["-white-threshold", "45%"],
            Box::new(WhiteThreshold::new(0.45)),
        ),
        (
            vec!["-white-threshold", "30%,50%,20%,60%"],
            Box::new(WhiteThreshold::per_channel(0.3, 0.5, 0.2, 0.6)),
        ),
        (
            vec!["-channel", "RG", "-white-threshold", "40%"],
            Box::new(WhiteThreshold::new(0.4).with_channels(rg)),
        ),
        (
            vec!["-black-threshold", "35%"],
            Box::new(BlackThreshold::new(0.35)),
        ),
        (
            vec!["-black-threshold", "35%,20%,25%,50%"],
            Box::new(BlackThreshold::per_channel(0.35, 0.2, 0.25, 0.5)),
        ),
        (
            vec!["-channel", "RGBA", "-black-threshold", "40%"],
            Box::new(BlackThreshold::new(0.4).with_channels(ChannelMask::ALL)),
        ),
        (
            vec!["-range-threshold", "10%,30%,50%,70%"],
            Box::new(RangeThreshold::new(0.1, 0.3, 0.5, 0.7)),
        ),
        (
            vec!["-channel", "RGB", "-range-threshold", "10%,30%,50%,70%"],
            Box::new(RangeThreshold::new(0.1, 0.3, 0.5, 0.7).with_channels(ChannelMask::RGB)),
        ),
        (
            vec!["-range-threshold", "20%,20%,60%,60%"],
            Box::new(RangeThreshold::new(0.2, 0.2, 0.6, 0.6)),
        ),
        (
            vec![
                "-color-threshold",
                "rgba(40%,0%,0%,0.5)-rgba(100%,60%,60%,1)",
            ],
            Box::new(ColorThreshold::new(
                [0.4, 0.0, 0.0, 0.5],
                [1.0, 0.6, 0.6, 1.0],
            )),
        ),
    ];
    for (ops, f) in &cases {
        let want = magick_rgba(&path, ops);
        let got = f.apply(&input, ROSE_RGBA).unwrap();
        let w = worst(&got.planes[0].data, &want);
        eprintln!("{ops:?}: worst {w}");
        assert!(w <= 1, "{ops:?}: off by {w} codes");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn splice_and_chop_match_magick() {
    use oxideav_image_filter::{Chop, Gravity, Splice};
    let Some((src, path)) = rose_rgba("splice") else {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    };
    let input = rgba_frame(&src);
    let gravities = [
        ("NorthWest", Gravity::NorthWest),
        ("North", Gravity::North),
        ("NorthEast", Gravity::NorthEast),
        ("West", Gravity::West),
        ("Center", Gravity::Centre),
        ("East", Gravity::East),
        ("SouthWest", Gravity::SouthWest),
        ("South", Gravity::South),
        ("SouthEast", Gravity::SouthEast),
    ];
    for (gname, g) in gravities {
        for (w, h, x, y) in [
            (7, 5, 0, 0),
            (7, 5, 3, 2),
            (4, 3, -2, -1),
            (1, 6, 9, 4),
            (12, 2, -5, 44),
        ] {
            let geo = format!("{w}x{h}{x:+}{y:+}");
            let want = try_magick_rgba(
                &path,
                &[
                    "-background",
                    "#3366cc80",
                    "-gravity",
                    gname,
                    "-splice",
                    &geo,
                ],
            );
            let got = Splice::new(w, h, x as i64, y as i64)
                .with_gravity(g)
                .with_background([0.2, 0.4, 0.8, 128.0 / 255.0])
                .apply(&input, ROSE_RGBA);
            match want {
                Some(want) => {
                    let got = got.unwrap();
                    assert_eq!(got.planes[0].stride, (70 + w as usize) * 4);
                    assert_eq!(got.planes[0].data, want, "splice {gname} {geo}");
                }
                None => assert!(got.is_err(), "splice {gname} {geo}: magick refuses it"),
            }
            let want = magick_rgba(&path, &["-gravity", gname, "-chop", &geo]);
            let got = Chop::new(w, h, x as i64, y as i64)
                .with_gravity(g)
                .apply(&input, ROSE_RGBA)
                .unwrap();
            assert_eq!(got.planes[0].data, want, "chop {gname} {geo}");
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn raise_matches_magick() {
    use oxideav_image_filter::Raise;
    let Some((src, path)) = rose_rgba("raise") else {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    };
    let input = rgba_frame(&src);
    for (w, h) in [(6u32, 4u32), (3, 9), (1, 1), (20, 2)] {
        for raised in [true, false] {
            let flag = if raised { "-raise" } else { "+raise" };
            let want = magick_rgba(&path, &[flag, &format!("{w}x{h}")]);
            let got = Raise::new(w, h)
                .with_raised(raised)
                .apply(&input, ROSE_RGBA)
                .unwrap();
            let worst = worst(&got.planes[0].data, &want);
            assert!(worst <= 1, "{flag} {w}x{h}: off by {worst}");
        }
    }
    assert!(Raise::new(35, 1).apply(&input, ROSE_RGBA).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn evaluate_sequence_separate_combine_match_magick() {
    use oxideav_image_filter::deep::{FloatImage, narrow_u8, widen_u8};
    use oxideav_image_filter::{SequenceOp, combine_channels, evaluate_sequence, separate};
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 70,
        height: 46,
    };
    let variants: [&[&str]; 3] = [
        &["rose:"],
        &["rose:", "-flop"],
        &["rose:", "-modulate", "80,40,150"],
    ];
    let dir = std::env::temp_dir();
    let mut paths = Vec::new();
    let mut imgs: Vec<FloatImage> = Vec::new();
    for (i, v) in variants.iter().enumerate() {
        let mut a = v.to_vec();
        a.extend(["-depth", "8", "rgb:-"]);
        let data = magick(&a).unwrap();
        let path = dir.join(format!("oxideav-if-seq-{i}-{}.rgb", std::process::id()));
        std::fs::write(&path, &data).unwrap();
        paths.push(format!("rgb:{}", path.display()));
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 210, data }],
        };
        imgs.push(widen_u8(&f, p).unwrap());
    }
    for name in [
        "Mean",
        "Median",
        "Min",
        "Max",
        "Add",
        "Abs",
        "Subtract",
        "Multiply",
        "Divide",
        "RMS",
        "AddModulus",
        "And",
        "Or",
        "Xor",
        "Set",
    ] {
        let mut a = vec!["-size", "70x46", "-depth", "8"];
        a.extend(paths.iter().map(String::as_str));
        a.extend(["-evaluate-sequence", name, "-depth", "8", "rgb:-"]);
        let want = magick(&a).unwrap();
        let op = SequenceOp::from_name(name).unwrap();
        let got = narrow_u8(&evaluate_sequence(&imgs, op).unwrap());
        let w = worst(&got.planes[0].data, &want);
        eprintln!("evaluate-sequence {name}: worst {w}");
        assert!(w <= 1, "{name}: off by {w}");
    }
    // -separate then -combine is the identity; each part matches magick.
    let parts = separate(&imgs[2]);
    for (i, part) in parts.iter().enumerate() {
        let want = magick(&[
            "-size",
            "70x46",
            "-depth",
            "8",
            &paths[2],
            "-channel",
            ["R", "G", "B"][i],
            "-separate",
            "-depth",
            "8",
            "gray:-",
        ])
        .unwrap();
        assert_eq!(narrow_u8(part).planes[0].data, want, "channel {i}");
    }
    assert_eq!(combine_channels(&parts).unwrap(), imgs[2]);
    for p in &paths {
        let _ = std::fs::remove_file(p.trim_start_matches("rgb:"));
    }
}

#[test]
fn sample_and_scale_match_magick() {
    use oxideav_image_filter::{Sample, Scale};
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let src = magick(&["rose:", "-depth", "8", "rgb:-"]).unwrap();
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 210,
            data: src,
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 70,
        height: 46,
    };
    for (w, h) in [
        (31u32, 19u32),
        (35, 23),
        (140, 92),
        (101, 67),
        (7, 5),
        (70, 13),
    ] {
        let geo = format!("{w}x{h}!");
        for (op, f) in [
            (
                "-sample",
                Box::new(Sample::new(w, h)) as Box<dyn ImageFilter>,
            ),
            ("-scale", Box::new(Scale::new(w, h))),
        ] {
            let want = magick(&["rose:", op, &geo, "-depth", "8", "rgb:-"]).unwrap();
            let got = f.apply(&input, p).unwrap();
            let worst = worst(&got.planes[0].data, &want);
            eprintln!("{op} {geo}: worst {worst}");
            assert!(worst <= 1, "{op} {geo}: off by {worst}");
        }
    }
    // With alpha.
    let (src, path) = rose_rgba("scale").unwrap();
    let input = rgba_frame(&src);
    for (w, h) in [(31u32, 19u32), (101, 67)] {
        let geo = format!("{w}x{h}!");
        let want = magick_rgba(&path, &["-scale", &geo]);
        let got = Scale::new(w, h).apply(&input, ROSE_RGBA).unwrap();
        let worst = worst(&got.planes[0].data, &want);
        eprintln!("-scale {geo} RGBA: worst {worst}");
        assert!(worst <= 1, "-scale {geo} RGBA: off by {worst}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cdl_matches_magick() {
    use oxideav_image_filter::Cdl;
    let Some((src, path)) = rose_rgba("cdl") else {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    };
    let input = rgba_frame(&src);
    for (slope, offset, power, sat) in [
        ("0.9 1.2 0.5", "0.4 -0.5 0.6", "1.0 0.8 1.5", "0.85"),
        ("1.1 0.95 1.3", "-0.05 0.02 0.0", "0.7 1.0 2.0", "1.4"),
    ] {
        let xml = format!(
            "<ColorCorrectionCollection><ColorCorrection id=\"a\"><SOPNode>\
             <Slope>{slope}</Slope><Offset>{offset}</Offset><Power>{power}</Power>\
             </SOPNode><SATNode><Saturation>{sat}</Saturation></SATNode>\
             </ColorCorrection></ColorCorrectionCollection>"
        );
        let cc = std::env::temp_dir().join(format!("oxideav-if-cdl-{}.cc", std::process::id()));
        std::fs::write(&cc, &xml).unwrap();
        let want = magick_rgba(&path, &["-cdl", cc.to_str().unwrap()]);
        let got = Cdl::from_xml(&xml)
            .unwrap()
            .apply(&input, ROSE_RGBA)
            .unwrap();
        let w = worst(&got.planes[0].data, &want);
        eprintln!("cdl {slope} / {offset} / {power} / {sat}: worst {w}");
        assert!(w <= 1, "cdl off by {w}");
        let _ = std::fs::remove_file(&cc);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn shear_matches_magick() {
    use oxideav_image_filter::Shear;
    if !reference() {
        eprintln!("magick not on the path: live comparison skipped");
        return;
    }
    let src = magick(&["rose:", "-depth", "8", "rgb:-"]).unwrap();
    let input = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 210,
            data: src,
        }],
    };
    let p = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 70,
        height: 46,
    };
    for (x, y) in [
        (10.0, 0.0),
        (30.0, 0.0),
        (-20.0, 0.0),
        (0.0, 10.0),
        (0.0, -25.0),
        (30.0, 20.0),
        (20.0, 30.0),
        (-15.0, 40.0),
        (10.0, 10.0),
        (40.0, 5.0),
        (-30.0, -20.0),
        (15.0, -40.0),
        (-25.0, 15.0),
        (5.0, 30.0),
        (33.0, 33.0),
    ] {
        let geo = format!("{x}x{y}");
        let dims = String::from_utf8(
            magick(&["rose:", "-shear", &geo, "-format", "%w %h", "info:"]).unwrap(),
        )
        .unwrap();
        let want = magick(&[
            "rose:",
            "-background",
            "#204080",
            "-shear",
            &geo,
            "-depth",
            "8",
            "rgb:-",
        ])
        .unwrap();
        let got = Shear::new(x, y)
            .with_background([32.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 1.0])
            .apply(&input, p)
            .unwrap();
        let gw = got.planes[0].stride / 3;
        let gh = got.planes[0].data.len() / got.planes[0].stride;
        assert_eq!(format!("{gw} {gh}"), dims.trim(), "-shear {geo} extent");
        let w = worst(&got.planes[0].data, &want);
        let mse = got.planes[0]
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / want.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10();
        eprintln!("-shear {geo}: worst {w}, {psnr:.1} dB");
        if x == 0.0 || y == 0.0 {
            assert!(w <= 1, "-shear {geo}: off by {w}");
        } else {
            // Both angles: the column pass matches to within a fraction
            // of a code on average, not bit-exactly (see the module docs).
            assert!(psnr >= 50.0, "-shear {geo}: {psnr:.1} dB");
        }
    }
}
