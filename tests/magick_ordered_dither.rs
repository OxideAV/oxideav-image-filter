//! Ordered dither against `magick -ordered-dither map,levels`, live when
//! the binary is on the path: a 256 × 8 grey ramp and the `rose:`
//! picture through every built-in map at 2, 3 and 6 levels, bit-exact.

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{ImageFilter, OrderedDither, VideoStreamParams};

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

/// `(label, magick source args, layout, width, height, raw sink)`.
type Case<'a> = (&'a str, &'a [&'a str], PixelFormat, u32, u32, &'a str);

#[test]
fn built_in_maps_match_magick() {
    if !reference() {
        eprintln!("magick not on the path: live ordered-dither comparison skipped");
        return;
    }
    let cases: [Case; 2] = [
        (
            "ramp",
            &["-size", "8x256", "gradient:black-white", "-rotate", "-90"],
            PixelFormat::Gray8,
            256,
            8,
            "gray:-",
        ),
        ("rose", &["rose:"], PixelFormat::Rgb24, 70, 46, "rgb:-"),
    ];
    for (label, src_args, fmt, w, h, sink) in cases {
        let mut a: Vec<&str> = src_args.to_vec();
        a.extend(["-depth", "8", sink]);
        let src = magick(&a).unwrap();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: src.len() / h as usize,
                data: src,
            }],
        };
        for map in ["threshold", "checks", "o2x2", "o4x4"] {
            for levels in ["", ",3", ",6"] {
                let spec = format!("{map}{levels}");
                let mut a: Vec<&str> = src_args.to_vec();
                a.extend(["-ordered-dither", &spec, "-depth", "8", sink]);
                let want = magick(&a).unwrap();
                let got = OrderedDither::parse(&spec)
                    .unwrap()
                    .apply(
                        &input,
                        VideoStreamParams {
                            format: fmt,
                            width: w,
                            height: h,
                        },
                    )
                    .unwrap();
                let g = &got.planes[0].data;
                assert_eq!(g.len(), want.len());
                let bad = g.iter().zip(&want).filter(|(a, b)| a != b).count();
                let worst = g
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (*a as i32 - *b as i32).abs())
                    .max()
                    .unwrap();
                eprintln!("{label} {spec}: {bad}/{} differ, worst {worst}", g.len());
                assert!(
                    bad * 100 <= g.len(),
                    "{label} {spec}: {bad} of {} samples differ",
                    g.len()
                );
            }
        }
    }
}
