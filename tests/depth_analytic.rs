//! Analytically known results at every depth: each filter here is run
//! on 8-bit, 16-bit and binary32 pictures, with and without alpha, and
//! compared against the closed-form answer of its definition (not
//! against a stored picture), at the precision of the depth — so a
//! 16-bit result that had been rounded through 8 bits fails.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    Blur, Despeckle, ImageFilter, Statistic, StatisticOp, VideoStreamParams, with_execution_context,
};

/// Storage depth of a test picture.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Depth {
    U8,
    U16,
    F32,
}

const DEPTHS: [Depth; 3] = [Depth::U8, Depth::U16, Depth::F32];

impl Depth {
    fn format(self, ch: usize) -> PixelFormat {
        match (self, ch) {
            (Depth::U8, 1) => PixelFormat::Gray8,
            (Depth::U8, 3) => PixelFormat::Rgb24,
            (Depth::U8, 4) => PixelFormat::Rgba,
            (Depth::U16, 1) => PixelFormat::Gray16Le,
            (Depth::U16, 3) => PixelFormat::Rgb48Le,
            (Depth::U16, 4) => PixelFormat::Rgba64Le,
            (Depth::F32, 1) => PixelFormat::GrayF32Le,
            (Depth::F32, 3) => PixelFormat::RgbF32Le,
            (Depth::F32, 4) => PixelFormat::RgbaF32Le,
            _ => unreachable!(),
        }
    }

    /// Full scale of the integer depths (1 for float).
    fn scale(self) -> f64 {
        match self {
            Depth::U8 => 255.0,
            Depth::U16 => 65535.0,
            Depth::F32 => 1.0,
        }
    }

    /// One quantisation step in normalised units (float: rounding slack).
    fn step(self) -> f64 {
        match self {
            Depth::U8 => 1.0 / 255.0,
            Depth::U16 => 1.0 / 65535.0,
            Depth::F32 => 1e-6,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Depth::U8 => 1,
            Depth::U16 => 2,
            Depth::F32 => 4,
        }
    }
}

/// A picture as integer codes at `depth` (float: the value itself).
/// `codes` are in units of the depth's scale so integer pictures hold
/// exact codes.
fn frame(depth: Depth, w: usize, ch: usize, codes: &[f64]) -> (VideoFrame, VideoStreamParams) {
    let mut data = Vec::with_capacity(codes.len() * depth.bytes());
    for &v in codes {
        match depth {
            Depth::U8 => data.push(v as u8),
            Depth::U16 => data.extend_from_slice(&(v as u16).to_le_bytes()),
            Depth::F32 => data.extend_from_slice(&(v as f32).to_le_bytes()),
        }
    }
    let h = codes.len() / (w * ch);
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w * ch * depth.bytes(),
                data,
            }],
        },
        VideoStreamParams::new(depth.format(ch), w as u32, h as u32),
    )
}

/// The samples of plane 0 in the units [`frame`] takes.
fn codes(depth: Depth, f: &VideoFrame) -> Vec<f64> {
    let d = &f.planes[0].data;
    match depth {
        Depth::U8 => d.iter().map(|&b| b as f64).collect(),
        Depth::U16 => d
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c) as f64)
            .collect(),
        Depth::F32 => d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect(),
    }
}

/// Deterministic noise in `[0, 1)`.
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

/// A noise picture as exact codes of `depth` (float: arbitrary values
/// in `[-0.5, 1.5)` so headroom is exercised).
fn noise_codes(depth: Depth, n: usize, seed: u64) -> Vec<f64> {
    noise(n, seed)
        .into_iter()
        .map(|v| match depth {
            Depth::F32 => ((v * 2.0 - 0.5) as f32) as f64,
            _ => (v * (depth.scale() + 1.0)).floor().min(depth.scale()),
        })
        .collect()
}

fn clamp_index(p: usize, off: isize, len: usize) -> usize {
    (p as isize + off).clamp(0, len as isize - 1) as usize
}

/// The replicate-edge window of channel `c` around `(x, y)`.
#[allow(clippy::too_many_arguments)]
fn window(
    src: &[f64],
    w: usize,
    h: usize,
    ch: usize,
    c: usize,
    x: usize,
    y: usize,
    (left, right, up, down): (isize, isize, isize, isize),
) -> Vec<f64> {
    let mut out = Vec::new();
    for dy in -up..=down {
        for dx in -left..=right {
            out.push(src[(clamp_index(y, dy, h) * w + clamp_index(x, dx, w)) * ch + c]);
        }
    }
    out
}

#[test]
fn statistic_is_the_window_order_statistic_at_every_depth() {
    let (w, h) = (13usize, 9usize);
    for depth in DEPTHS {
        for ch in [1usize, 3, 4] {
            let src = noise_codes(depth, w * h * ch, 3 + ch as u64);
            let (f, p) = frame(depth, w, ch, &src);
            let nc = if ch == 4 { 3 } else { ch };
            for (ww, wh) in [(3u32, 3u32), (5, 5), (4, 2), (1, 7), (9, 1), (31, 31)] {
                let ext = (
                    ((ww - 1) / 2) as isize,
                    (ww / 2) as isize,
                    ((wh - 1) / 2) as isize,
                    (wh / 2) as isize,
                );
                for op in [
                    StatisticOp::Median,
                    StatisticOp::Min,
                    StatisticOp::Max,
                    StatisticOp::Mean,
                ] {
                    let out = Statistic::new(op, ww, wh).apply(&f, p).unwrap();
                    let got = codes(depth, &out);
                    for y in 0..h {
                        for x in 0..w {
                            for c in 0..ch {
                                let i = (y * w + x) * ch + c;
                                if c >= nc {
                                    assert_eq!(got[i], src[i], "alpha is carried");
                                    continue;
                                }
                                let mut win = window(&src, w, h, ch, c, x, y, ext);
                                win.sort_by(f64::total_cmp);
                                let n = win.len();
                                let (want, tol) = match op {
                                    StatisticOp::Median => (win[n / 2], 0.0),
                                    StatisticOp::Min => (win[0], 0.0),
                                    StatisticOp::Max => (win[n - 1], 0.0),
                                    // Integer means round to nearest;
                                    // the 16-bit one is computed in
                                    // binary32 on the normalised scale.
                                    _ => (
                                        win.iter().sum::<f64>() / n as f64,
                                        match depth {
                                            Depth::U8 => 0.5,
                                            Depth::U16 => 0.51,
                                            Depth::F32 => 1e-6,
                                        },
                                    ),
                                };
                                assert!(
                                    (got[i] - want).abs() <= tol,
                                    "{depth:?} ch{ch} {op:?} {ww}x{wh} at {x},{y},{c}: {} vs {want}",
                                    got[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn despeckle_is_the_square_window_median_at_every_depth() {
    let (w, h) = (11usize, 8usize);
    for depth in DEPTHS {
        for ch in [3usize, 4] {
            let src = noise_codes(depth, w * h * ch, 17);
            let (f, p) = frame(depth, w, ch, &src);
            for r in [0u32, 1, 2] {
                let out = Despeckle::new(r).apply(&f, p).unwrap();
                let got = codes(depth, &out);
                let e = r as isize;
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..ch {
                            let i = (y * w + x) * ch + c;
                            let want = if c == 3 {
                                src[i]
                            } else {
                                let mut win = window(&src, w, h, ch, c, x, y, (e, e, e, e));
                                win.sort_by(f64::total_cmp);
                                win[win.len() / 2]
                            };
                            assert_eq!(got[i], want, "{depth:?} ch{ch} r{r} at {x},{y},{c}");
                        }
                    }
                }
            }
        }
    }
}

/// The normalised Gaussian taps `Blur::new(radius).with_sigma(sigma)`
/// is defined by: `exp(−i² / 2σ²)` over `−r..=r`, summing to one.
fn gaussian(radius: usize, sigma: f64) -> Vec<f64> {
    let r = radius as isize;
    let raw: Vec<f64> = (-r..=r)
        .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|v| v / sum).collect()
}

#[test]
fn blur_is_the_2d_gaussian_sum_at_the_precision_of_the_depth() {
    let (w, h) = (12usize, 10usize);
    let (radius, sigma) = (3usize, 1.5f64);
    let k = gaussian(radius, sigma);
    for depth in DEPTHS {
        for ch in [1usize, 3] {
            let src = noise_codes(depth, w * h * ch, 29);
            let (f, p) = frame(depth, w, ch, &src);
            let out = Blur::new(radius as u32)
                .with_sigma(sigma as f32)
                .apply(&f, p)
                .unwrap();
            let got = codes(depth, &out);
            let r = radius as isize;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..ch {
                        let mut want = 0.0;
                        for (j, kj) in k.iter().enumerate() {
                            for (i, ki) in k.iter().enumerate() {
                                let sx = clamp_index(x, i as isize - r, w);
                                let sy = clamp_index(y, j as isize - r, h);
                                want += kj * ki * src[(sy * w + sx) * ch + c];
                            }
                        }
                        let i = (y * w + x) * ch + c;
                        // Rounded once: within just over half a code of
                        // the exact sum (a 16-bit result that had gone
                        // through 8 bits would be off by up to 128).
                        let tol = match depth {
                            Depth::U8 => 0.501,
                            Depth::U16 => 0.52,
                            Depth::F32 => 2e-6,
                        };
                        assert!(
                            (got[i] - want).abs() <= tol,
                            "{depth:?} ch{ch} at {x},{y},{c}: {} vs {want}",
                            got[i]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn blur_weights_colour_by_coverage_on_alpha_pictures() {
    // Left half opaque colour A, right half fully transparent colour B.
    // The blurred alpha is the Gaussian step; wherever it is non-zero
    // the colour is exactly A — B never bleeds in.
    let (w, h) = (16usize, 3usize);
    let k = gaussian(2, 1.0);
    for depth in DEPTHS {
        let s = depth.scale();
        let a = [0.8 * s, 0.2 * s, 0.4 * s].map(|v| {
            if depth == Depth::F32 {
                v as f32 as f64
            } else {
                v.round()
            }
        });
        let b = [0.1 * s, 0.9 * s, 0.7 * s].map(|v| {
            if depth == Depth::F32 {
                v as f32 as f64
            } else {
                v.round()
            }
        });
        let mut src = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                if x < w / 2 {
                    src.extend_from_slice(&[a[0], a[1], a[2], s]);
                } else {
                    src.extend_from_slice(&[b[0], b[1], b[2], 0.0]);
                }
            }
        }
        let (f, p) = frame(depth, w, 4, &src);
        let out = Blur::new(2).with_sigma(1.0).apply(&f, p).unwrap();
        let got = codes(depth, &out);
        for x in 0..w {
            let px = &got[(w + x) * 4..(w + x) * 4 + 4];
            let cover: f64 = k
                .iter()
                .enumerate()
                .filter(|(i, _)| clamp_index(x, *i as isize - 2, w) < w / 2)
                .map(|(_, v)| v)
                .sum();
            let tol = if depth == Depth::F32 { 2e-6 } else { 0.52 };
            assert!(
                (px[3] - cover * s).abs() <= tol,
                "{depth:?} alpha at {x}: {} vs {}",
                px[3],
                cover * s
            );
            if px[3] > 0.0 {
                // The division by a small coverage amplifies the last
                // rounding of the premultiplied sum.
                let ctol = if depth == Depth::F32 {
                    1e-4
                } else {
                    0.51 + 2.0 * depth.step()
                };
                for c in 0..3 {
                    assert!(
                        (px[c] - a[c]).abs() <= ctol.max(a[c] * 1e-4),
                        "{depth:?} colour at {x},{c}: {} vs {}",
                        px[c],
                        a[c]
                    );
                }
            } else {
                assert_eq!(&px[..3], &b[..], "uncovered pixels keep their colour");
            }
        }
    }
}

#[test]
fn degenerate_pictures_and_windows_do_not_panic() {
    for depth in DEPTHS {
        for (w, h) in [(1usize, 1usize), (1, 5), (5, 1), (2, 2)] {
            for ch in [1usize, 4] {
                let src = noise_codes(depth, w * h * ch, 5);
                let (f, p) = frame(depth, w, ch, &src);
                assert!(Blur::new(40).with_sigma(0.0).apply(&f, p).is_ok());
                assert!(Blur::new(1).with_sigma(f32::NAN).apply(&f, p).is_ok());
                for op in [
                    StatisticOp::Median,
                    StatisticOp::Min,
                    StatisticOp::Max,
                    StatisticOp::Mean,
                ] {
                    assert!(Statistic::new(op, 64, 64).apply(&f, p).is_ok());
                    assert!(Statistic::new(op, 0, 3).apply(&f, p).is_err());
                }
            }
        }
    }
}

#[test]
fn outputs_are_identical_at_every_thread_budget() {
    let (w, h) = (37usize, 29usize);
    for depth in DEPTHS {
        let src = noise_codes(depth, w * h * 4, 41);
        let (f, p) = frame(depth, w, 4, &src);
        let filters: Vec<Box<dyn ImageFilter>> = vec![
            Box::new(Blur::new(4).with_sigma(2.0)),
            Box::new(Statistic::new(StatisticOp::Median, 5, 3)),
            Box::new(Statistic::new(StatisticOp::Max, 4, 7)),
            Box::new(Statistic::new(StatisticOp::Mean, 6, 6)),
            Box::new(Despeckle::new(2)),
        ];
        for (i, filter) in filters.iter().enumerate() {
            let serial = filter.apply(&f, p).unwrap();
            for t in [2usize, 5, 16] {
                let par = with_execution_context(&ExecutionContext::with_threads(t), || {
                    filter.apply(&f, p)
                })
                .unwrap();
                assert_eq!(
                    serial.planes[0].data, par.planes[0].data,
                    "{depth:?} filter {i} threads {t}"
                );
            }
        }
    }
}
