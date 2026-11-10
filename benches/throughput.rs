//! Wall-clock throughput of the most used filters at 1 MP and 12 MP on
//! 8-bit (`Rgba`) and 16-bit (`Rgba64Le`) pictures.
//!
//! Plain `std::time` harness (no external bench crate):
//!
//! ```text
//! cargo bench --bench throughput                 # 1 MP, serial
//! IMF_BENCH_MP=12 cargo bench --bench throughput # 12 MP
//! IMF_BENCH_THREADS=8 cargo bench --bench throughput
//! IMF_BENCH_ONLY=blur,statistic cargo bench --bench throughput
//! ```
//!
//! Each line is the median of `IMF_BENCH_RUNS` (default 3) runs.

use std::time::Instant;

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    Blur, ColorMatrix, Composite, CompositeOp, DiffusionKernel, Dither, ImageFilter, Interpolation,
    Morphology, OrderedDither, Resize, Rotate, Sharpen, Statistic, StatisticOp, StructuringElement,
    Threshold, ThresholdMap, TwoInputImageFilter, Unsharp, VideoStreamParams,
};

/// Deterministic picture: smooth gradients plus xorshift noise so rank
/// filters see realistic windows.
fn picture(w: u32, h: u32, deep: bool) -> (VideoFrame, VideoStreamParams) {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = (w as usize) * (h as usize);
    let bytes = if deep { 8 } else { 4 };
    let mut data = Vec::with_capacity(n * bytes);
    for y in 0..h {
        for x in 0..w {
            let noise = (next() >> 40) as u32 & 0x0fff;
            let r = (x * 65535 / w.max(1)).saturating_add(noise).min(65535);
            let g = (y * 65535 / h.max(1)).saturating_add(noise).min(65535);
            let b = ((x + y) * 65535 / (w + h)).saturating_add(noise).min(65535);
            let a = 65535 - (noise << 2).min(30000);
            for v in [r, g, b, a] {
                if deep {
                    data.extend_from_slice(&(v as u16).to_le_bytes());
                } else {
                    data.push((v >> 8) as u8);
                }
            }
        }
    }
    let format = if deep {
        PixelFormat::Rgba64Le
    } else {
        PixelFormat::Rgba
    };
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize * bytes,
                data,
            }],
        },
        VideoStreamParams {
            format,
            width: w,
            height: h,
        },
    )
}

fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn main() {
    let mp: u32 = std::env::var("IMF_BENCH_MP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let runs: usize = std::env::var("IMF_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let threads: usize = std::env::var("IMF_BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let only: Option<Vec<String>> = std::env::var("IMF_BENCH_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_owned).collect());
    // 4:3 pictures: 1 MP = 1152 × 868, 12 MP = 4000 × 3000.
    let (w, h) = match mp {
        12 => (4000, 3000),
        _ => (1152, 868),
    };
    let ctx = ExecutionContext::with_threads(threads);

    let singles: Vec<(&str, Box<dyn ImageFilter>)> = vec![
        ("blur r3 s1.5", Box::new(Blur::new(3).with_sigma(1.5))),
        ("blur r12 s6", Box::new(Blur::new(12).with_sigma(6.0))),
        (
            "resize half lanczos3",
            Box::new(Resize::new(w / 2, h / 2).with_interpolation(Interpolation::Lanczos { a: 3 })),
        ),
        (
            "resize 1.5x bicubic",
            Box::new(Resize::new(w * 3 / 2, h * 3 / 2).with_interpolation(Interpolation::Bicubic)),
        ),
        ("sharpen r2 s1", Box::new(Sharpen::new(2, 1.0))),
        ("unsharp r3 s1.5", Box::new(Unsharp::new(3, 1.5, 1.0, 0))),
        (
            "morphology dilate sq x2",
            Box::new(Morphology::dilate(StructuringElement::Square3x3, 2)),
        ),
        (
            "morphology erode disk r5",
            Box::new(Morphology::erode(StructuringElement::Disk(5), 1)),
        ),
        (
            "statistic median 5x5",
            Box::new(Statistic::new(StatisticOp::Median, 5, 5)),
        ),
        (
            "statistic max 9x9",
            Box::new(Statistic::new(StatisticOp::Max, 9, 9)),
        ),
        (
            "statistic mean 9x9",
            Box::new(Statistic::new(StatisticOp::Mean, 9, 9)),
        ),
        ("rotate 17deg", Box::new(Rotate::new(17.0))),
        (
            "colour-matrix",
            Box::new(ColorMatrix::new([
                0.9, 0.1, 0.0, 0.05, 0.9, 0.05, 0.0, 0.1, 0.9,
            ])),
        ),
        ("threshold", Box::new(Threshold::new(128))),
        (
            "dither floyd-steinberg",
            Box::new(Dither::error_diffusion(DiffusionKernel::FloydSteinberg)),
        ),
        (
            "ordered dither o8x8",
            Box::new(OrderedDither::new(
                ThresholdMap::named("o8x8").expect("built-in map"),
            )),
        ),
    ];

    println!("# {w}x{h} ({mp} MP), threads = {threads}, median of {runs}");
    for deep in [false, true] {
        let (frame, params) = picture(w, h, deep);
        let depth = if deep { "16-bit" } else { "8-bit" };
        for (name, filter) in &singles {
            if only
                .as_ref()
                .is_some_and(|only| !only.iter().any(|o| name.contains(o.as_str())))
            {
                continue;
            }
            let mut status = "ok";
            let ms = median_ms(runs, || {
                let r = oxideav_image_filter::with_execution_context(&ctx, || {
                    filter.apply(&frame, params)
                });
                if r.is_err() {
                    status = "refused";
                }
            });
            println!("{depth:7} {name:26} {ms:10.1} ms  {status}");
        }
        let skip_composite = only
            .as_ref()
            .is_some_and(|o| !o.iter().any(|o| "composite over".contains(o.as_str())));
        if !skip_composite {
            let over = Composite::new(CompositeOp::Over);
            let mut status = "ok";
            let ms = median_ms(runs, || {
                let r = oxideav_image_filter::with_execution_context(&ctx, || {
                    over.apply_two(&frame, &frame, params)
                });
                if r.is_err() {
                    status = "refused";
                }
            });
            println!("{depth:7} {:26} {ms:10.1} ms  {status}", "composite over");
        }
    }
}
