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
