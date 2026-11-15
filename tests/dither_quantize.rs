//! Issue #15 through the public API: the dither methods and the colour
//! quantiser on 8-bit, 16-bit and float pictures and at every thread
//! budget, plus — live when `magick` is on the path — the Radon skew
//! estimate and the quality of the chosen palette against the
//! validator's.

use std::process::Command;

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    ChannelDither, Deskew, DiffusionKernel, DitherMethod, ImageFilter, Palette, PaletteQuantize,
    ThresholdMap, VideoStreamParams, with_execution_context,
};

fn magick(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("magick").args(args).output().ok()?;
    out.status.success().then_some(out.stdout)
}

/// `true` for the build the comparisons were made on (ImageMagick 7.1.2
/// Q16-HDRI, macOS / Linux); other builds are not held to them.
fn reference() -> bool {
    !cfg!(windows)
        && magick(&["-version"]).is_some_and(|v| {
            let v = String::from_utf8_lossy(&v).to_string();
            v.contains("ImageMagick 7.1.2") && v.contains("Q16-HDRI")
        })
}

fn params(format: PixelFormat, width: u32, height: u32) -> VideoStreamParams {
    VideoStreamParams {
        format,
        width,
        height,
    }
}

/// A smooth colour picture as normalised RGB.
fn scene(w: u32, h: u32) -> Vec<f32> {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (u, t) = (x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32);
            v.push(u);
            v.push((0.5 + 0.5 * (6.0 * t).sin() * (4.0 * u).cos()).clamp(0.0, 1.0));
            v.push(t * t);
        }
    }
    v
}

fn frame8(rgb: &[f32], w: u32) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 3,
            data: rgb.iter().map(|v| (v * 255.0 + 0.5) as u8).collect(),
        }],
    }
}

fn frame16(rgb8: &[u8], w: u32) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 6,
            data: rgb8
                .iter()
                .flat_map(|&b| (b as u16 * 257).to_le_bytes())
                .collect(),
        }],
    }
}

fn frame_f32(rgb8: &[u8], w: u32) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 12,
            data: rgb8
                .iter()
                .flat_map(|&b| (b as f32 / 255.0).to_le_bytes())
                .collect(),
        }],
    }
}

fn methods() -> Vec<DitherMethod> {
    vec![
        DitherMethod::None,
        DitherMethod::floyd_steinberg(),
        DitherMethod::diffusion(DiffusionKernel::Atkinson),
        DitherMethod::hilbert(),
        DitherMethod::blue_noise(),
        DitherMethod::Ordered(ThresholdMap::clustered_dot(6.0, 45.0, Default::default())),
    ]
}

#[test]
fn every_depth_gives_the_same_picture() {
    let (w, h) = (48u32, 32u32);
    let base = frame8(&scene(w, h), w);
    let codes = &base.planes[0].data;
    let palette = Palette::from_rgb8(&[
        [0, 0, 0],
        [255, 255, 255],
        [200, 30, 30],
        [30, 160, 60],
        [40, 60, 200],
        [230, 220, 40],
    ])
    .unwrap();
    for method in methods() {
        let filters: Vec<Box<dyn ImageFilter>> = vec![
            Box::new(ChannelDither::new(method.clone()).with_channel_levels(4, 6, 3)),
            Box::new(PaletteQuantize::new(palette.clone()).with_dither(method.clone())),
        ];
        for f in filters {
            let out8 = f.apply(&base, params(PixelFormat::Rgb24, w, h)).unwrap();
            let out16 = f
                .apply(&frame16(codes, w), params(PixelFormat::Rgb48Le, w, h))
                .unwrap();
            let outf = f
                .apply(&frame_f32(codes, w), params(PixelFormat::RgbF32Le, w, h))
                .unwrap();
            let from16: Vec<u8> = out16.planes[0]
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| ((u16::from_le_bytes(*b) as f32 / 257.0) + 0.5) as u8)
                .collect();
            let fromf: Vec<u8> = outf.planes[0]
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| (f32::from_le_bytes(*b) * 255.0 + 0.5) as u8)
                .collect();
            let differ = |other: &[u8]| {
                out8.planes[0]
                    .data
                    .iter()
                    .zip(other)
                    .filter(|(a, b)| a != b)
                    .count()
            };
            // The same normalised picture: a diffusion may flip a pixel
            // where two depths round a tie differently, no more.
            let limit = (w * h * 3) as usize / 50;
            assert!(
                differ(&from16) <= limit,
                "{method:?}: 16-bit differs in {}",
                differ(&from16)
            );
            assert!(
                differ(&fromf) <= limit,
                "{method:?}: float differs in {}",
                differ(&fromf)
            );
        }
    }
}

#[test]
fn the_thread_budget_does_not_change_the_output() {
    let (w, h) = (64u32, 40u32);
    let base = frame8(&scene(w, h), w);
    let p = params(PixelFormat::Rgb24, w, h);
    for method in methods() {
        let filters: Vec<Box<dyn ImageFilter>> = vec![
            Box::new(ChannelDither::new(method.clone()).with_levels(3)),
            Box::new(PaletteQuantize::adaptive(12).with_dither(method.clone())),
        ];
        for f in filters {
            let serial = f.apply(&base, p).unwrap();
            for threads in [2usize, 5, 16] {
                let ctx = ExecutionContext::with_threads(threads);
                let par = with_execution_context(&ctx, || f.apply(&base, p)).unwrap();
                assert_eq!(
                    serial.planes[0].data, par.planes[0].data,
                    "{method:?} at {threads}"
                );
            }
        }
    }
}

#[test]
fn dithering_recovers_tones_the_palette_lacks() {
    // A grey ramp on black and white only: after a small blur the
    // dithered picture is the ramp again; the undithered one is a step.
    let (w, h) = (128u32, 64u32);
    let ramp: Vec<f32> = (0..w * h)
        .flat_map(|i| [(i % w) as f32 / (w - 1) as f32; 3])
        .collect();
    let base = frame8(&ramp, w);
    let p = params(PixelFormat::Rgb24, w, h);
    let bw = Palette::from_rgb8(&[[0, 0, 0], [255, 255, 255]]).unwrap();
    let error_after_blur = |method: DitherMethod| -> f64 {
        let out = PaletteQuantize::new(bw.clone())
            .with_dither(method)
            .apply(&base, p)
            .unwrap();
        let d = &out.planes[0].data;
        let mut total = 0.0;
        let mut n = 0.0;
        for y in (8..h - 8).step_by(4) {
            for x in (8..w - 8).step_by(4) {
                let mut s = 0.0;
                for yy in y - 8..y + 8 {
                    for xx in x - 8..x + 8 {
                        s += d[((yy * w + xx) * 3) as usize] as f64 / 255.0;
                    }
                }
                let want = (x as f64 - 0.5) / (w - 1) as f64;
                total += (s / 256.0 - want).powi(2);
                n += 1.0;
            }
        }
        (total / n).sqrt()
    };
    let plain = error_after_blur(DitherMethod::None);
    assert!(plain > 0.2, "undithered error {plain}");
    for (index, method) in methods().into_iter().enumerate().skip(1) {
        // (The stencil of dithering-kernels.md §7 drops a quarter of the
        // error by design and does not hold a ramp's mean.)
        if method == DitherMethod::diffusion(DiffusionKernel::Atkinson) {
            continue;
        }
        let e = error_after_blur(method);
        assert!(e < 0.03, "method {index}: error after blur {e}");
    }
}

#[test]
fn radon_skew_matches_magick() {
    if !reference() {
        eprintln!("magick not on the path: live deskew comparison skipped");
        return;
    }
    let dir = std::env::temp_dir();
    for angle in ["3", "-2.5", "7", "0.4", "-5"] {
        let path = dir.join(format!(
            "oxideav-if-skew-{}-{angle}.png",
            std::process::id()
        ));
        let path = path.to_str().unwrap();
        magick(&[
            "-size",
            "400x300",
            "xc:white",
            "-fill",
            "black",
            "-draw",
            "rectangle 30,60 370,64 rectangle 30,100 370,104 rectangle 30,140 330,144 \
             rectangle 30,180 370,184 rectangle 30,220 250,224",
            "-background",
            "white",
            "-rotate",
            angle,
            "+repage",
            path,
        ])
        .unwrap();
        let info = magick(&[
            path,
            "-deskew",
            "40%",
            "-format",
            "%[deskew:angle]",
            "info:",
        ])
        .unwrap();
        let theirs: f64 = String::from_utf8_lossy(&info).trim().parse().unwrap();
        let size = magick(&[path, "-format", "%w %h", "info:"]).unwrap();
        let size = String::from_utf8_lossy(&size).to_string();
        let mut it = size.split_whitespace().map(|v| v.parse::<u32>().unwrap());
        let (w, h) = (it.next().unwrap(), it.next().unwrap());
        let gray = magick(&[path, "-colorspace", "Gray", "-depth", "8", "gray:-"]).unwrap();
        let _ = std::fs::remove_file(path);
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data: gray,
            }],
        };
        let ours = Deskew::new(102)
            .with_max_angle(45.0)
            .estimate_angle(&frame, params(PixelFormat::Gray8, w, h))
            .unwrap() as f64;
        // The validator reports the correcting rotation (the negative
        // of the skew); both search slopes s / W with W = 512 here.
        let step = (1.0f64 / 512.0).atan().to_degrees();
        eprintln!("rotated {angle}: skew {ours}, validator rotation {theirs}");
        assert!(
            (ours + theirs).abs() <= 1.5 * step,
            "rotated {angle}: ours {ours}, validator {theirs}"
        );
    }
}

#[test]
fn chosen_palette_fits_no_worse_than_magick() {
    if !reference() {
        eprintln!("magick not on the path: live quantiser comparison skipped");
        return;
    }
    let src = magick(&["rose:", "-depth", "8", "rgb:-"]).unwrap();
    let (w, h) = (70u32, 46u32);
    let frame = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 3,
            data: src.clone(),
        }],
    };
    let mse = |a: &[u8], b: &[u8]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64
    };
    for n in ["4", "16", "64", "256"] {
        let theirs = magick(&["rose:", "+dither", "-colors", n, "-depth", "8", "rgb:-"]).unwrap();
        let ours = PaletteQuantize::adaptive(n.parse().unwrap())
            .with_dither(DitherMethod::None)
            .apply(&frame, params(PixelFormat::Rgb24, w, h))
            .unwrap();
        let (e_ours, e_theirs) = (mse(&ours.planes[0].data, &src), mse(&theirs, &src));
        let mut colours: Vec<[u8; 3]> = ours.planes[0].data.as_chunks::<3>().0.to_vec();
        colours.sort();
        colours.dedup();
        eprintln!(
            "{n} colours: mse ours {e_ours:.2} ({} used), validator {e_theirs:.2}",
            colours.len()
        );
        assert!(colours.len() <= n.parse::<usize>().unwrap());
        assert!(
            e_ours <= e_theirs * 1.02,
            "{n} colours: {e_ours} vs {e_theirs}"
        );
    }
}
