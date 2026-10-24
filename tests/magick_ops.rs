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
