//! Algebraic properties of the compose operators (issue #9): the
//! Porter–Duff identities and decompositions, commutativity and duality
//! of the blend functions, neutral colours, the channel mask, and
//! agreement of the 8-bit, 16-bit and float paths at every thread
//! budget — on pseudo-random pictures with every kind of alpha.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::deep::FloatImage;
use oxideav_image_filter::{
    ChannelMask, Composite, CompositeOp, VideoStreamParams, with_execution_context,
};

const W: u32 = 23;
const H: u32 = 9;

/// A deterministic RGBA picture: colours on the 8-bit grid, alpha
/// covering 0, 1 and everything between.
fn picture(seed: u64) -> FloatImage {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 24) as u8
    };
    let mut data = Vec::new();
    for i in 0..W * H {
        for _ in 0..3 {
            data.push(next() as f32 / 255.0);
        }
        let a = match i % 5 {
            0 => 0,
            1 => 255,
            _ => next(),
        };
        data.push(a as f32 / 255.0);
    }
    FloatImage::from_samples(W, H, 4, data).unwrap()
}

fn opaque(mut img: FloatImage) -> FloatImage {
    img.samples_mut()
        .iter_mut()
        .skip(3)
        .step_by(4)
        .for_each(|a| *a = 1.0);
    img
}

fn flat(rgba: [f32; 4]) -> FloatImage {
    FloatImage::from_samples(W, H, 4, rgba.repeat((W * H) as usize)).unwrap()
}

/// `op` with `src` composed onto `dst`.
fn compose(op: CompositeOp, dst: &FloatImage, src: &FloatImage) -> FloatImage {
    Composite::new(op).compose_float(dst, src)
}

/// How an operator's result holds its colour: `Over`, `DstOver`, the
/// destination forms `DstIn` / `DstOut` and the plain copies hold
/// straight colour; `In`, `Out`, `Atop`, `DstAtop`, `Xor`, `Plus` and
/// the premultiplied blends hold the coverage-weighted colour as it is
/// (the conventions measured on the validator, see the `composite`
/// module documentation).
#[derive(Clone, Copy)]
enum Held {
    Straight,
    Weighted,
}

/// A picture as coverage-weighted colour plus coverage — the form in
/// which the algebra is stated, and the only one in which colour under
/// zero coverage does not matter.
fn weighted(img: &FloatImage, held: Held) -> Vec<f32> {
    img.samples()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| match held {
            Held::Straight => [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]],
            Held::Weighted => *p,
        })
        .collect()
}

fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-5, "{what}: differs by {worst}");
}

fn assert_same(what: &str, a: &FloatImage, b: &FloatImage) {
    assert_close(
        what,
        &weighted(a, Held::Straight),
        &weighted(b, Held::Straight),
    );
}

/// The colour channels as they are held.
fn colours(img: &FloatImage) -> Vec<f32> {
    img.samples()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

fn sum(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(p, q)| p + q).collect()
}

fn alphas(img: &FloatImage) -> Vec<f32> {
    img.samples().iter().skip(3).step_by(4).copied().collect()
}

#[test]
fn porter_duff_coverage_and_identities() {
    use CompositeOp as Op;
    let (s, d) = (picture(1), picture(2));
    // Clear, Src, Dst.
    assert!(
        alphas(&compose(Op::Clear, &d, &s))
            .iter()
            .all(|&a| a == 0.0)
    );
    assert_same("src", &compose(Op::Src, &d, &s), &s);
    assert_same("dst", &compose(Op::Dst, &d, &s), &d);
    // Result coverage of each operator.
    type Rule = fn(f32, f32) -> f32;
    let rules: [(Op, Rule); 9] = [
        (Op::Over, |a, b| a + b - a * b),
        (Op::DstOver, |a, b| a + b - a * b),
        (Op::In, |a, b| a * b),
        (Op::DstIn, |a, b| a * b),
        (Op::Out, |a, b| a * (1.0 - b)),
        (Op::DstOut, |a, b| b * (1.0 - a)),
        (Op::Atop, |_, b| b),
        (Op::DstAtop, |a, _| a),
        (Op::Xor, |a, b| a + b - 2.0 * a * b),
    ];
    for (op, rule) in rules {
        let out = compose(op, &d, &s);
        for ((o, a), b) in alphas(&out).iter().zip(alphas(&s)).zip(alphas(&d)) {
            assert!(
                (o - rule(a, b)).abs() < 2e-5,
                "{op:?}: alpha {o} for src {a} dst {b}"
            );
        }
    }
    use Held::{Straight, Weighted};
    let w = |op: Op, dst: &FloatImage, src: &FloatImage, held: Held| {
        weighted(&compose(op, dst, src), held)
    };
    // The destination forms are the source forms with the roles exchanged.
    for (fwd, fwd_held, rev, rev_held) in [
        (Op::Over, Straight, Op::DstOver, Straight),
        (Op::In, Weighted, Op::DstIn, Straight),
        (Op::Out, Weighted, Op::DstOut, Straight),
        (Op::Atop, Weighted, Op::DstAtop, Weighted),
    ] {
        assert_close(
            &format!("{fwd:?} / {rev:?}"),
            &w(fwd, &d, &s, fwd_held),
            &w(rev, &s, &d, rev_held),
        );
    }
    // Xor and Plus do not care which is which.
    assert_close(
        "xor commutes",
        &w(Op::Xor, &d, &s, Weighted),
        &w(Op::Xor, &s, &d, Weighted),
    );
    assert_close(
        "plus commutes",
        &w(Op::Plus, &d, &s, Weighted),
        &w(Op::Plus, &s, &d, Weighted),
    );
    // Decompositions into disjoint parts.
    let s_in_d = w(Op::In, &d, &s, Weighted);
    let s_out_d = w(Op::Out, &d, &s, Weighted);
    let d_out_s = w(Op::DstOut, &d, &s, Straight);
    let whole_s = weighted(&s, Straight);
    assert_close(
        "over = S + (D out S)",
        &w(Op::Over, &d, &s, Straight),
        &sum(&whole_s, &d_out_s),
    );
    assert_close(
        "atop = (S in D) + (D out S)",
        &w(Op::Atop, &d, &s, Weighted),
        &sum(&s_in_d, &d_out_s),
    );
    assert_close(
        "xor = (S out D) + (D out S)",
        &w(Op::Xor, &d, &s, Weighted),
        &sum(&s_out_d, &d_out_s),
    );
    assert_close(
        "S = (S in D) + (S out D)",
        &whole_s,
        &sum(&s_in_d, &s_out_d),
    );
    // Over: an opaque source hides the destination, a clear one shows it;
    // over is associative.
    let solid = opaque(picture(3));
    assert_same("opaque over", &compose(Op::Over, &d, &solid), &solid);
    assert_same(
        "clear over",
        &compose(Op::Over, &d, &flat([0.3, 0.6, 0.9, 0.0])),
        &d,
    );
    let t = picture(4);
    assert_same(
        "over is associative",
        &compose(Op::Over, &compose(Op::Over, &d, &s), &t),
        &compose(Op::Over, &d, &compose(Op::Over, &s, &t)),
    );
}

#[test]
fn blend_functions_commute_or_are_dual_and_have_neutral_colours() {
    use CompositeOp as Op;
    let (s, d) = (opaque(picture(5)), opaque(picture(6)));
    for op in [
        Op::Multiply,
        Op::Screen,
        Op::Darken,
        Op::Lighten,
        Op::Difference,
        Op::Exclusion,
        Op::Plus,
        Op::LinearDodge,
        Op::LinearBurn,
    ] {
        assert_same(
            &format!("{op:?} commutes"),
            &compose(op, &d, &s),
            &compose(op, &s, &d),
        );
    }
    // Pairs that are each other with the operands exchanged.
    for (a, b) in [
        (Op::Overlay, Op::HardLight),
        (Op::MinusSrc, Op::MinusDst),
        (Op::DivideSrc, Op::DivideDst),
    ] {
        assert_same(
            &format!("{a:?} / {b:?}"),
            &compose(a, &d, &s),
            &compose(b, &s, &d),
        );
    }
    // With partial coverage the symmetric blends still commute (they
    // hold their colour the same way whichever is the source; Exclusion
    // does not: the validator weights only its source by coverage).
    let (ps, pd) = (picture(7), picture(8));
    for op in [
        Op::Multiply,
        Op::Screen,
        Op::Darken,
        Op::Lighten,
        Op::Difference,
    ] {
        assert_close(
            &format!("{op:?} commutes with alpha"),
            &weighted(&compose(op, &pd, &ps), Held::Weighted),
            &weighted(&compose(op, &ps, &pd), Held::Weighted),
        );
    }
    // Neutral source colours leave an opaque destination alone.
    let (black, white, grey) = (
        flat([0.0, 0.0, 0.0, 1.0]),
        flat([1.0, 1.0, 1.0, 1.0]),
        flat([0.5, 0.5, 0.5, 1.0]),
    );
    for (op, neutral) in [
        (Op::Multiply, &white),
        (Op::Darken, &white),
        (Op::ColorBurn, &white),
        (Op::LinearBurn, &white),
        (Op::Screen, &black),
        (Op::Lighten, &black),
        (Op::Exclusion, &black),
        (Op::Plus, &black),
        (Op::ColorDodge, &black),
        (Op::LinearDodge, &black),
        (Op::HardLight, &grey),
        (Op::LinearLight, &grey),
        (Op::SoftLight, &grey),
    ] {
        assert_same(
            &format!("{op:?} with its neutral colour"),
            &compose(op, &d, neutral),
            &d,
        );
    }
    // Absorbing colours.
    assert_same(
        "multiply by black",
        &compose(Op::Multiply, &d, &black),
        &black,
    );
    assert_same(
        "screen with white",
        &compose(Op::Screen, &d, &white),
        &white,
    );
    // Difference (whose result coverage is the difference of the
    // coverages, so nothing on opaque pictures — the colours are compared
    // as held): symmetric, zero with itself, the identity with black, the
    // negative with white.
    let diff = |dst: &FloatImage, src: &FloatImage| colours(&compose(Op::Difference, dst, src));
    assert_close("difference commutes", &diff(&d, &s), &diff(&s, &d));
    assert_close("difference with itself", &diff(&d, &d), &colours(&black));
    assert_close("difference with black", &diff(&d, &black), &colours(&d));
    let negative: Vec<f32> = colours(&d).iter().map(|v| 1.0 - v).collect();
    assert_close("difference with white", &diff(&d, &white), &negative);
    let partial = compose(Op::Difference, &picture(7), &picture(8));
    for ((o, a), b) in alphas(&partial)
        .iter()
        .zip(alphas(&picture(8)))
        .zip(alphas(&picture(7)))
    {
        assert!((o - (a - b).abs()).abs() < 2e-5);
    }
}

#[test]
fn the_channel_mask_keeps_the_other_channels() {
    let (s, d) = (picture(9), picture(10));
    for (mask, kept) in [
        (
            ChannelMask::new(true, false, false, false),
            [false, true, true, true],
        ),
        (
            ChannelMask::new(false, true, true, false),
            [true, false, false, true],
        ),
        (
            ChannelMask::new(false, false, false, true),
            [true, true, true, false],
        ),
    ] {
        for &op in CompositeOp::ALL {
            if op.is_map() {
                continue;
            }
            let out = Composite::new(op).with_channels(mask).compose_float(&d, &s);
            for (o, i) in out
                .samples()
                .as_chunks::<4>()
                .0
                .iter()
                .zip(d.samples().as_chunks::<4>().0)
            {
                for c in (0..4).filter(|&c| kept[c]) {
                    assert_eq!(o[c], i[c], "{op:?}: channel {c} changed under the mask");
                }
            }
        }
    }
}

fn frame_of(img: &FloatImage, format: PixelFormat) -> VideoFrame {
    let codes = img.samples().iter().map(|v| (v * 255.0 + 0.5) as u8);
    let (data, bytes): (Vec<u8>, usize) = match format {
        PixelFormat::Rgba => (codes.collect(), 4),
        PixelFormat::Rgba64Le => (
            codes.flat_map(|c| (c as u16 * 257).to_le_bytes()).collect(),
            8,
        ),
        _ => (
            codes
                .flat_map(|c| (c as f32 / 255.0).to_le_bytes())
                .collect(),
            16,
        ),
    };
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: W as usize * bytes,
            data,
        }],
    }
}

fn codes_of(frame: &VideoFrame, format: PixelFormat) -> Vec<f32> {
    let d = &frame.planes[0].data;
    match format {
        PixelFormat::Rgba => d.iter().map(|&b| b as f32).collect(),
        PixelFormat::Rgba64Le => d
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b) as f32 / 257.0)
            .collect(),
        _ => d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) * 255.0)
            .collect(),
    }
}

#[test]
fn every_operator_agrees_across_depths_and_thread_budgets() {
    let (s, d) = (picture(11), picture(12));
    let formats = [
        PixelFormat::Rgba,
        PixelFormat::Rgba64Le,
        PixelFormat::RgbaF32Le,
    ];
    for &op in CompositeOp::ALL {
        let filter = Composite::new(op).with_args(vec![0.4, 0.6, 0.1, 0.05]);
        let mut results = Vec::new();
        for format in formats {
            let p = VideoStreamParams {
                format,
                width: W,
                height: H,
            };
            let (df, sf) = (frame_of(&d, format), frame_of(&s, format));
            let serial = filter.compose(&df, p, &sf, p).unwrap();
            for threads in [3usize, 8] {
                let ctx = ExecutionContext::with_threads(threads);
                let par = with_execution_context(&ctx, || filter.compose(&df, p, &sf, p)).unwrap();
                assert!(
                    serial.planes[0].data == par.planes[0].data,
                    "{op:?} {format:?} at {threads} threads"
                );
            }
            let codes = codes_of(&serial, format);
            assert!(
                codes.iter().all(|v| v.is_finite()),
                "{op:?} {format:?}: non-finite output"
            );
            results.push(codes);
        }
        // The stored-integer depths saturate; compare where the float
        // result is in range. Colour under zero coverage is undefined.
        let float = &results[2];
        for (name, other, tolerance) in [("8-bit", &results[0], 1.0), ("16-bit", &results[1], 0.02)]
        {
            for (px, (f, o)) in float
                .as_chunks::<4>()
                .0
                .iter()
                .zip(other.as_chunks::<4>().0)
                .enumerate()
            {
                if f[3] < 1.0 {
                    continue;
                }
                for c in 0..4 {
                    let want = f[c].clamp(0.0, 255.0);
                    assert!(
                        (want - o[c]).abs() <= tolerance,
                        "{op:?}: {name} pixel {px} channel {c}: {} vs float {want}",
                        o[c]
                    );
                }
            }
        }
    }
}
