//! Analytically known results at every depth: each filter here is run
//! on 8-bit, 16-bit and binary32 pictures, with and without alpha, and
//! compared against the closed-form answer of its definition (not
//! against a stored picture), at the precision of the depth — so a
//! 16-bit result that had been rounded through 8 bits fails.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    BayerMatrix, Blur, Despeckle, DiffusionKernel, Dither, EdgeOp, ImageFilter, Morphology,
    MorphologyEdge, MorphologyOp, ScanOrder, Statistic, StatisticOp, StructuringElement,
    VideoStreamParams, with_execution_context,
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
            Box::new(Morphology::dilate(StructuringElement::Disk(3), 2)),
            Box::new(Morphology::erode(StructuringElement::Square(2), 2)),
            Box::new(Dither::ordered(BayerMatrix::M8).with_levels(4)),
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

/// The error-diffusion stencils of `docs/image/filter/dithering-kernels.md`
/// §3–§7, as drawn: rows of weights with the column of `*`, and the
/// divisor.
fn stencil(kernel: DiffusionKernel) -> (Vec<Vec<i32>>, usize, f32) {
    match kernel {
        DiffusionKernel::FloydSteinberg => (vec![vec![0, 0, 7], vec![3, 5, 1]], 1, 16.0),
        DiffusionKernel::Jjn => (
            vec![
                vec![0, 0, 0, 7, 5],
                vec![3, 5, 7, 5, 3],
                vec![1, 3, 5, 3, 1],
            ],
            2,
            48.0,
        ),
        DiffusionKernel::Stucki => (
            vec![
                vec![0, 0, 0, 8, 4],
                vec![2, 4, 8, 4, 2],
                vec![1, 2, 4, 2, 1],
            ],
            2,
            42.0,
        ),
        DiffusionKernel::Sierra3 => (
            vec![
                vec![0, 0, 0, 5, 3],
                vec![2, 4, 5, 4, 2],
                vec![0, 2, 3, 2, 0],
            ],
            2,
            32.0,
        ),
        DiffusionKernel::Sierra2 => (vec![vec![0, 0, 0, 4, 3], vec![1, 2, 3, 2, 1]], 2, 16.0),
        DiffusionKernel::SierraLite => (vec![vec![0, 0, 2], vec![1, 1, 0]], 1, 4.0),
        DiffusionKernel::Atkinson => (
            vec![vec![0, 0, 1, 1], vec![1, 1, 1, 0], vec![0, 1, 0, 0]],
            1,
            8.0,
        ),
        _ => unreachable!(),
    }
}

const KERNELS: [DiffusionKernel; 7] = [
    DiffusionKernel::FloydSteinberg,
    DiffusionKernel::Jjn,
    DiffusionKernel::Stucki,
    DiffusionKernel::Sierra3,
    DiffusionKernel::Sierra2,
    DiffusionKernel::SierraLite,
    DiffusionKernel::Atkinson,
];

/// Error diffusion by its definition, on one channel of normalised
/// samples with a whole-picture error image: quantise `v + e` to the
/// nearest of `levels` values, push the residue through the stencil
/// (mirrored on the right→left rows of a serpentine scan).
fn diffuse_reference(
    src: &[f32],
    w: usize,
    h: usize,
    kernel: DiffusionKernel,
    levels: u32,
    serpentine: bool,
) -> Vec<f32> {
    let (rows, star, divisor) = stencil(kernel);
    let span = (levels - 1) as f32;
    let mut err = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let reverse = serpentine && y % 2 == 1;
        for col in 0..w {
            let x = if reverse { w - 1 - col } else { col };
            let v = src[y * w + x] + err[y * w + x];
            let q = (v.clamp(0.0, 1.0) * span + 0.5).floor() / span;
            out[y * w + x] = q;
            let e = v - q;
            for (dy, row) in rows.iter().enumerate() {
                for (i, &weight) in row.iter().enumerate() {
                    if weight == 0 {
                        continue;
                    }
                    let dx = i as isize - star as isize;
                    let nx = x as isize + if reverse { -dx } else { dx };
                    let ny = y + dy;
                    if nx < 0 || nx >= w as isize || ny >= h {
                        continue;
                    }
                    err[ny * w + nx as usize] += e * (weight as f32 * (1.0 / divisor));
                }
            }
        }
    }
    out
}

#[test]
fn error_diffusion_follows_the_published_stencils_at_every_depth() {
    let (w, h) = (23usize, 14usize);
    for depth in DEPTHS {
        let s = depth.scale();
        for ch in [1usize, 4] {
            let nc = if ch == 4 { 3 } else { 1 };
            // In-range tones (float included) so the reference's clamp
            // is the only clamp.
            let src: Vec<f64> = noise(w * h * ch, 77)
                .into_iter()
                .map(|v| match depth {
                    Depth::F32 => (v as f32) as f64,
                    _ => (v * (s + 1.0)).floor().min(s),
                })
                .collect();
            let (f, p) = frame(depth, w, ch, &src);
            for kernel in KERNELS {
                for scan in [ScanOrder::Raster, ScanOrder::Serpentine] {
                    for levels in [2u32, 3, 6, 17] {
                        let out = Dither::error_diffusion(kernel)
                            .with_levels(levels)
                            .with_scan(scan)
                            .apply(&f, p)
                            .unwrap();
                        let got = codes(depth, &out);
                        for c in 0..ch {
                            let plane: Vec<f32> = (0..w * h)
                                .map(|i| (src[i * ch + c] as f32) / s as f32)
                                .collect();
                            if c >= nc {
                                for i in 0..w * h {
                                    assert_eq!(got[i * ch + c], src[i * ch + c], "alpha carried");
                                }
                                continue;
                            }
                            let want = diffuse_reference(
                                &plane,
                                w,
                                h,
                                kernel,
                                levels,
                                scan == ScanOrder::Serpentine,
                            );
                            for i in 0..w * h {
                                // The level at the picture's depth.
                                let level = match depth {
                                    Depth::F32 => want[i] as f64,
                                    _ => (want[i] as f64 * s + 0.5).floor(),
                                };
                                assert!(
                                    (got[i * ch + c] - level).abs() <= 1e-6,
                                    "{depth:?} {kernel:?} {scan:?} L{levels} ch{c} sample {i}: {} vs {level}",
                                    got[i * ch + c]
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
fn conserving_kernels_keep_the_mean_of_a_16_bit_tone_no_8_bit_code_holds() {
    // A flat field at code 20 000 of 65 535 (between the 8-bit codes 77
    // and 78). Every conserving kernel reproduces that tone as the
    // density of its two output levels, to within the error that left
    // through the picture's edge.
    let (w, h) = (96usize, 96usize);
    let tone = 20_000.0f64;
    let src = vec![tone; w * h];
    let (f, p) = frame(Depth::U16, w, 1, &src);
    for kernel in KERNELS {
        for scan in [ScanOrder::Raster, ScanOrder::Serpentine] {
            let out = Dither::error_diffusion(kernel)
                .with_scan(scan)
                .apply(&f, p)
                .unwrap();
            let got = codes(Depth::U16, &out);
            assert!(got.iter().all(|&v| v == 0.0 || v == 65535.0));
            let mean = got.iter().sum::<f64>() / got.len() as f64;
            if kernel == DiffusionKernel::Atkinson {
                // A quarter of every residue is dropped: on a tone below
                // mid-grey the picture comes out darker than the tone.
                assert!(mean < tone - 300.0, "{scan:?}: {mean}");
            } else {
                assert!(
                    (mean - tone).abs() < 0.012 * 65535.0,
                    "{kernel:?} {scan:?}: mean {mean} vs {tone}"
                );
            }
        }
    }
}

#[test]
fn ordered_dither_compares_against_the_bayer_thresholds_at_every_depth() {
    // Two levels: the output is white exactly where the tone plus the
    // cell's centred threshold reaches one half, `v + (m + ½)/n² − ½ ≥ ½`
    // — on a flat tone `v` the white density is therefore the number of
    // cells with `(m + ½)/n² ≥ 1 − v`.
    for depth in DEPTHS {
        let s = depth.scale();
        for (matrix, n) in [
            (BayerMatrix::M2, 2usize),
            (BayerMatrix::M4, 4),
            (BayerMatrix::M8, 8),
            (BayerMatrix::M16, 16),
        ] {
            let cells = (n * n) as f64;
            for tone in [0.0f64, 0.1, 0.337, 0.5, 0.81, 1.0] {
                let code = match depth {
                    Depth::F32 => (tone as f32) as f64,
                    _ => (tone * s).round(),
                };
                let v = code / s;
                let src = vec![code; n * n * 4];
                let (f, p) = frame(depth, 2 * n, 1, &src);
                let out = Dither::ordered(matrix).apply(&f, p).unwrap();
                let got = codes(depth, &out);
                assert!(got.iter().all(|&g| g == 0.0 || g == s), "{depth:?}");
                let white = got.iter().filter(|&&g| g == s).count() as f64 / got.len() as f64;
                let want = (0..n * n)
                    .filter(|&m| (m as f64 + 0.5) / cells >= 1.0 - v - 1e-9)
                    .count() as f64
                    / cells;
                // A cell exactly on the decision edge may fall either
                // way in binary32.
                assert!(
                    (white - want).abs() <= 1.0 / cells + 1e-9,
                    "{depth:?} {matrix:?} tone {tone}: {white} vs {want}"
                );
            }
        }
        // Multi-level: every output is one of the L levels at depth and
        // within one level spacing of the input.
        let (w, h) = (16usize, 16usize);
        let src = noise_codes(depth, w * h * 3, 13);
        let (f, p) = frame(depth, w, 3, &src);
        for levels in [3u32, 5, 16] {
            let out = Dither::ordered(BayerMatrix::M8)
                .with_levels(levels)
                .apply(&f, p)
                .unwrap();
            let span = (levels - 1) as f64;
            for (g, v) in codes(depth, &out).iter().zip(&src) {
                // A level `k / (L − 1)`, stored at the picture's depth
                // (rounded to the nearest code on the integer depths).
                let k = (g / s * span).round();
                let level = match depth {
                    Depth::F32 => ((k as f32) / (span as f32)) as f64,
                    _ => (k / span * s + 0.5).floor(),
                };
                assert_eq!(*g, level, "{depth:?} L{levels}");
                let v = (v / s).clamp(0.0, 1.0);
                assert!((g / s - v).abs() <= 1.0 / span + depth.step());
            }
        }
    }
}

#[test]
fn dither_survives_degenerate_pictures_and_non_finite_samples() {
    for depth in DEPTHS {
        for (w, h) in [(1usize, 1usize), (1, 7), (7, 1), (2, 2)] {
            let src = noise_codes(depth, w * h * 4, 9);
            let (f, p) = frame(depth, w, 4, &src);
            for kernel in KERNELS {
                for scan in [ScanOrder::Raster, ScanOrder::Serpentine] {
                    for levels in [0u32, 1, 2, 255, u32::MAX] {
                        let d = Dither::error_diffusion(kernel)
                            .with_levels(levels)
                            .with_scan(scan);
                        assert!(d.apply(&f, p).is_ok());
                    }
                }
            }
            assert!(Dither::ordered(BayerMatrix::M16).apply(&f, p).is_ok());
        }
    }
    let odd = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e30, -1e30, 0.4];
    let (f, p) = frame(Depth::F32, 3, 1, &odd);
    let out = Dither::new().apply(&f, p).unwrap();
    let got = codes(Depth::F32, &out);
    assert_eq!(&got[..3], &[0.0, 1.0, 0.0]);
    assert!(got.iter().all(|v| *v == 0.0 || *v == 1.0));
}

/// The offsets of a structuring element from its definition.
fn element_offsets(element: StructuringElement) -> Vec<(isize, isize)> {
    let (r, inside): (isize, Box<dyn Fn(isize, isize) -> bool>) = match element {
        StructuringElement::Square3x3 => (1, Box::new(|_, _| true)),
        StructuringElement::Cross3x3 => (1, Box::new(|dx, dy| dx.abs() + dy.abs() <= 1)),
        StructuringElement::Square(r) => (r as isize, Box::new(|_, _| true)),
        StructuringElement::Diamond(r) => {
            let r = r as isize;
            (r, Box::new(move |dx, dy| dx.abs() + dy.abs() <= r))
        }
        StructuringElement::Disk(r) => {
            let r = r as isize;
            (r, Box::new(move |dx, dy| dx * dx + dy * dy <= r * r))
        }
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if inside(dx, dy) {
                out.push((dx, dy));
            }
        }
    }
    out
}

/// One pass of flat dilation / erosion by definition: the extreme over
/// the element's pixels that fall inside the picture.
fn morph_reference(
    src: &[f64],
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    offsets: &[(isize, isize)],
    dilate: bool,
) -> Vec<f64> {
    let mut out = src.to_vec();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = if dilate {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                for &(dx, dy) in offsets {
                    let (sx, sy) = (x as isize + dx, y as isize + dy);
                    if sx < 0 || sy < 0 || sx >= w as isize || sy >= h as isize {
                        continue;
                    }
                    let v = src[(sy as usize * w + sx as usize) * ch + c];
                    acc = if dilate { acc.max(v) } else { acc.min(v) };
                }
                out[(y * w + x) * ch + c] = acc;
            }
        }
    }
    out
}

const ELEMENTS: [StructuringElement; 8] = [
    StructuringElement::Square3x3,
    StructuringElement::Cross3x3,
    StructuringElement::Square(2),
    StructuringElement::Diamond(3),
    StructuringElement::Disk(1),
    StructuringElement::Disk(2),
    StructuringElement::Disk(4),
    StructuringElement::Square(0),
];

#[test]
fn morphology_is_the_extreme_over_the_element_at_every_depth() {
    let (w, h) = (14usize, 11usize);
    for depth in DEPTHS {
        for ch in [1usize, 3, 4] {
            let nc = if ch == 4 { 3 } else { ch };
            let src = noise_codes(depth, w * h * ch, 31 + ch as u64);
            let (f, p) = frame(depth, w, ch, &src);
            for element in ELEMENTS {
                let offsets = element_offsets(element);
                assert_eq!(
                    element
                        .footprint()
                        .iter()
                        .map(|&(dx, dy)| (dx as isize, dy as isize))
                        .collect::<Vec<_>>(),
                    {
                        let mut o = offsets.clone();
                        o.sort_by_key(|&(dx, dy)| (dy, dx));
                        o
                    },
                    "{element:?} footprint"
                );
                for (op, dilate) in [(MorphologyOp::Dilate, true), (MorphologyOp::Erode, false)] {
                    for iterations in [1u32, 2, 3] {
                        let out = Morphology::new(op, element, iterations)
                            .apply(&f, p)
                            .unwrap();
                        let mut want = src.clone();
                        for _ in 0..iterations {
                            want = morph_reference(&want, w, h, ch, nc, &offsets, dilate);
                        }
                        assert_eq!(
                            codes(depth, &out),
                            want,
                            "{depth:?} ch{ch} {element:?} {op:?} x{iterations}"
                        );
                    }
                }
                // Edge-in / edge-out / gradient are the differences of
                // the picture and its one-pass erosion / dilation.
                let dil = morph_reference(&src, w, h, ch, nc, &offsets, true);
                let ero = morph_reference(&src, w, h, ch, nc, &offsets, false);
                for (op, a, b) in [
                    (EdgeOp::EdgeIn, &src, &ero),
                    (EdgeOp::EdgeOut, &dil, &src),
                    (EdgeOp::EdgeMagnitude, &dil, &ero),
                ] {
                    let out = MorphologyEdge::new(op, element).apply(&f, p).unwrap();
                    let got = codes(depth, &out);
                    for i in 0..src.len() {
                        let want = if i % ch >= nc { src[i] } else { a[i] - b[i] };
                        // Float pictures subtract in binary32 on the
                        // normalised scale.
                        let tol = match depth {
                            Depth::F32 => 1e-6,
                            Depth::U16 => 0.51,
                            Depth::U8 => 0.0,
                        };
                        assert!(
                            (got[i] - want).abs() <= tol,
                            "{depth:?} ch{ch} {element:?} {op:?} sample {i}: {} vs {want}",
                            got[i]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn morphology_survives_huge_elements_and_tiny_pictures() {
    for depth in DEPTHS {
        for (w, h) in [(1usize, 1usize), (1, 6), (6, 1), (3, 3)] {
            let src = noise_codes(depth, w * h * 4, 2);
            let (f, p) = frame(depth, w, 4, &src);
            for element in [
                StructuringElement::Square(u32::MAX),
                StructuringElement::Diamond(u32::MAX),
                StructuringElement::Disk(u32::MAX),
                StructuringElement::Disk(0),
                StructuringElement::Cross3x3,
            ] {
                for iterations in [0u32, 1, u32::MAX] {
                    let out = Morphology::dilate(element, iterations)
                        .apply(&f, p)
                        .unwrap();
                    let got = codes(depth, &out);
                    if iterations > 0 && element.radius() > 1 {
                        // The element covers the picture: every pixel is
                        // the global maximum, alpha untouched.
                        for c in 0..3 {
                            let max = (0..w * h)
                                .map(|i| src[i * 4 + c])
                                .fold(f64::NEG_INFINITY, f64::max);
                            assert!((0..w * h).all(|i| got[i * 4 + c] == max), "{element:?}");
                        }
                    }
                    assert!((0..w * h).all(|i| got[i * 4 + 3] == src[i * 4 + 3]));
                }
            }
        }
    }
}
