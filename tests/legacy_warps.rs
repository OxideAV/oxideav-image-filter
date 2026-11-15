//! `Affine`, `Srt`, `Perspective` and `Polar` are rendered by the
//! distortion engine. Their output is what their definition says —
//! every output pixel is the bilinear blend of the four source pixels
//! round the mapped position (edges replicated), or the background
//! where the map leaves the source rectangle — to within one 8-bit code
//! (`2e-4` on float pictures), and is the same at every thread budget.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    Affine, ImageFilter, Perspective, Polar, Srt, VideoStreamParams, with_execution_context,
};

const W: usize = 37;
const H: usize = 23;

fn scene() -> Vec<f32> {
    let mut v = Vec::with_capacity(W * H * 4);
    for i in 0..W * H {
        let (x, y) = ((i % W) as f32, (i / W) as f32);
        v.push(((x * 7.0 + y * 3.0) % 32.0 / 31.0 * 255.0).round() / 255.0);
        v.push(((x * y) % 17.0 / 16.0 * 255.0).round() / 255.0);
        v.push((y / (H - 1) as f32 * 255.0).round() / 255.0);
        v.push(((x + 2.0 * y) % 5.0 / 4.0 * 255.0).round() / 255.0);
    }
    v
}

/// The definition: bilinear at `(sx, sy)` with replicated edges.
fn bilinear(src: &[f32], sx: f32, sy: f32) -> [f32; 4] {
    let (x0, y0) = (sx.floor(), sy.floor());
    let (tx, ty) = (sx - x0, sy - y0);
    let at = |x: f32, y: f32, c: usize| {
        let (x, y) = (
            (x.max(0.0) as usize).min(W - 1),
            (y.max(0.0) as usize).min(H - 1),
        );
        src[(y * W + x) * 4 + c]
    };
    std::array::from_fn(|c| {
        at(x0, y0, c) * (1.0 - tx) * (1.0 - ty)
            + at(x0 + 1.0, y0, c) * tx * (1.0 - ty)
            + at(x0, y0 + 1.0, c) * (1.0 - tx) * ty
            + at(x0 + 1.0, y0 + 1.0, c) * tx * ty
    })
}

/// The expected picture for a reverse map (`None`: background).
fn expected(src: &[f32], bg: [f32; 4], map: impl Fn(f32, f32) -> Option<(f32, f32)>) -> Vec<f32> {
    let mut out = Vec::with_capacity(W * H * 4);
    for i in 0..W * H {
        out.extend(match map((i % W) as f32, (i / W) as f32) {
            Some((sx, sy)) => bilinear(src, sx, sy),
            None => bg,
        });
    }
    out
}

fn check(name: &str, filter: &dyn ImageFilter, want: &[f32]) {
    let src = scene();
    let frames = [
        (
            PixelFormat::Rgba,
            src.iter()
                .map(|v| (v * 255.0 + 0.5) as u8)
                .collect::<Vec<u8>>(),
            4usize,
            1.0f32 / 255.0 + 1e-4,
        ),
        (
            PixelFormat::RgbaF32Le,
            src.iter().flat_map(|v| v.to_le_bytes()).collect(),
            16,
            2e-4,
        ),
    ];
    for (format, data, bytes, tolerance) in frames {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: W * bytes,
                data,
            }],
        };
        let params = VideoStreamParams {
            format,
            width: W as u32,
            height: H as u32,
        };
        let out = filter.apply(&frame, params).unwrap();
        let got: Vec<f32> = match format {
            PixelFormat::Rgba => out.planes[0]
                .data
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect(),
            _ => out.planes[0]
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        };
        assert_eq!(got.len(), want.len(), "{name} {format:?}");
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= tolerance, "{name} {format:?}: off by {worst}");
        for threads in [2usize, 7] {
            let ctx = ExecutionContext::with_threads(threads);
            let par = with_execution_context(&ctx, || filter.apply(&frame, params)).unwrap();
            assert!(
                out.planes[0].data == par.planes[0].data,
                "{name} {format:?} at {threads} threads"
            );
        }
    }
}

fn inside(sx: f32, sy: f32) -> Option<(f32, f32)> {
    const EPS: f32 = 1e-4;
    (sx >= -EPS && sx <= W as f32 - 1.0 + EPS && sy >= -EPS && sy <= H as f32 - 1.0 + EPS)
        .then_some((sx, sy))
}

#[test]
fn affine_and_srt_are_bilinear_reads_through_the_inverse_matrix() {
    let src = scene();
    let bg = [0.2, 0.4, 0.6, 1.0];
    let bg8 = [51, 102, 153, 255];
    // Forward: x' = 1.25 x + 0.5 y + 3, y' = −0.25 x + 0.75 y + 2.
    let (a, b, c, d, tx, ty) = (1.25f32, 0.5, -0.25, 0.75, 3.0, 2.0);
    let det = a * d - b * c;
    let want = expected(&src, bg, |x, y| {
        let (u, v) = (x - tx, y - ty);
        inside((d * u - b * v) / det, (-c * u + a * v) / det)
    });
    check(
        "affine",
        &Affine::new([a, b, c, d, tx, ty]).with_background(bg8),
        &want,
    );
    // Srt about (10, 8): scale 1.5, a quarter turn clockwise, moved to (20, 9).
    // (Srt's default background is opaque black.)
    let want = expected(&src, [0.0, 0.0, 0.0, 1.0], |x, y| {
        // Forward: p' = t + R(S(p − o)), R a clockwise quarter turn
        // (x, y) → (−y, x) in picture coordinates.
        let (dx, dy) = (x - 20.0, y - 9.0);
        inside(10.0 + dy / 1.5, 8.0 - dx / 1.5)
    });
    check(
        "srt",
        &Srt::new((10.0, 8.0), (1.5, 1.5), 90.0, (20.0, 9.0)),
        &want,
    );
}

#[test]
fn perspective_is_a_bilinear_read_through_the_inverse_homography() {
    let src = scene();
    let bg = [1.0, 0.0, 0.0, 1.0];
    // A homography that is a scale and shift: corners (x, y) → (0.5 x + 4, 0.5 y + 3).
    let (w1, h1) = (W as f32 - 1.0, H as f32 - 1.0);
    let corners = [(0.0, 0.0), (w1, 0.0), (w1, h1), (0.0, h1)];
    let moved = corners.map(|(x, y)| (0.5 * x + 4.0, 0.5 * y + 3.0));
    let want = expected(&src, bg, |x, y| inside((x - 4.0) * 2.0, (y - 3.0) * 2.0));
    check(
        "perspective",
        &Perspective::new(corners, moved).with_background([255, 0, 0, 255]),
        &want,
    );
}

#[test]
fn polar_and_depolar_are_bilinear_reads_through_their_formulas() {
    let src = scene();
    let (wf, hf) = (W as f32, H as f32);
    let (cx, cy) = ((wf - 1.0) / 2.0, (hf - 1.0) / 2.0);
    let max_r = (wf - 1.0 - cx).hypot(hf - 1.0 - cy).max(1.0);
    let tau = std::f32::consts::TAU;
    let want = expected(&src, [0.0; 4], |x, y| {
        let r = y / hf * max_r;
        let (s, c) = (x / wf * tau).sin_cos();
        Some((cx + r * s, cy - r * c))
    });
    check("polar", &Polar::forward(), &want);
    let want = expected(&src, [0.0; 4], |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let mut a = dx.atan2(-dy);
        if a < 0.0 {
            a += tau;
        }
        Some((a / tau * wf, dx.hypot(dy) / max_r * hf))
    });
    check("depolar", &Polar::inverse(), &want);
}
