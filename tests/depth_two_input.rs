//! Two-input filters at 16 bits and binary32 against the closed form of
//! each operation: a 16-bit result must be right to the 16-bit code,
//! which a pass through 8 bits could not give.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{
    Clut, Composite, CompositeOp, Difference, DisplacementMap, HaldClut, TwoInputImageFilter,
    VideoStreamParams, Watermark,
};

fn frame16(w: usize, ch: usize, codes: &[u16]) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * ch * 2,
            data: codes.iter().flat_map(|c| c.to_le_bytes()).collect(),
        }],
    }
}

fn frame_f32(w: usize, ch: usize, v: &[f32]) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * ch * 4,
            data: v.iter().flat_map(|c| c.to_le_bytes()).collect(),
        }],
    }
}

fn codes16(f: &VideoFrame) -> Vec<u16> {
    f.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

fn floats(f: &VideoFrame) -> Vec<f32> {
    f.planes[0]
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn noise16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 48) as u16
        })
        .collect()
}

fn fmt16(ch: usize) -> PixelFormat {
    match ch {
        1 => PixelFormat::Gray16Le,
        3 => PixelFormat::Rgb48Le,
        _ => PixelFormat::Rgba64Le,
    }
}

#[test]
fn over_is_the_porter_duff_sum_to_the_16_bit_code() {
    // Straight-alpha source over straight-alpha backdrop:
    //   αo = αs + αb (1 − αs),  Co = (Cs αs + Cb αb (1 − αs)) / αo.
    let (w, h) = (16usize, 8usize);
    let src = noise16(w * h * 4, 3);
    let dst = noise16(w * h * 4, 11);
    let p = VideoStreamParams::new(PixelFormat::Rgba64Le, w as u32, h as u32);
    let out = Composite::new(CompositeOp::Over)
        .apply_two(&frame16(w, 4, &src), &frame16(w, 4, &dst), p)
        .unwrap();
    let got = codes16(&out);
    for i in 0..w * h {
        let s: Vec<f64> = src[i * 4..i * 4 + 4]
            .iter()
            .map(|v| *v as f64 / 65535.0)
            .collect();
        let b: Vec<f64> = dst[i * 4..i * 4 + 4]
            .iter()
            .map(|v| *v as f64 / 65535.0)
            .collect();
        let ao = s[3] + b[3] * (1.0 - s[3]);
        assert!(
            (got[i * 4 + 3] as f64 - ao * 65535.0).abs() <= 1.0,
            "alpha at {i}"
        );
        if ao < 1e-3 {
            continue;
        }
        for c in 0..3 {
            let co = (s[c] * s[3] + b[c] * b[3] * (1.0 - s[3])) / ao;
            // Binary32 arithmetic and the division by a small coverage
            // leave a few 16-bit codes of slack — far inside one 8-bit
            // step (257 codes).
            let tol = 1.5 + 4.0 / ao;
            assert!(
                (got[i * 4 + c] as f64 - co * 65535.0).abs() <= tol,
                "pixel {i} channel {c}: {} vs {}",
                got[i * 4 + c],
                co * 65535.0
            );
        }
    }
}

#[test]
fn opaque_operators_follow_their_blend_formula_at_16_bits() {
    // Opaque pictures: the result colour is the blend function itself.
    let (w, h) = (32usize, 4usize);
    let src = noise16(w * h * 3, 5);
    let dst = noise16(w * h * 3, 9);
    let p = VideoStreamParams::new(PixelFormat::Rgb48Le, w as u32, h as u32);
    type Blend = fn(f64, f64) -> f64;
    let cases: [(CompositeOp, Blend); 8] = [
        (CompositeOp::Multiply, |s, b| s * b),
        (CompositeOp::Screen, |s, b| s + b - s * b),
        (CompositeOp::Darken, |s, b| s.min(b)),
        (CompositeOp::Lighten, |s, b| s.max(b)),
        (CompositeOp::Difference, |s, b| (s - b).abs()),
        (CompositeOp::Exclusion, |s, b| s + b - 2.0 * s * b),
        (CompositeOp::Plus, |s, b| (s + b).min(1.0)),
        (CompositeOp::HardLight, |s, b| {
            if s <= 0.5 {
                2.0 * s * b
            } else {
                1.0 - 2.0 * (1.0 - s) * (1.0 - b)
            }
        }),
    ];
    for (op, blend) in cases {
        let out = Composite::new(op)
            .apply_two(&frame16(w, 3, &src), &frame16(w, 3, &dst), p)
            .unwrap();
        for (i, g) in codes16(&out).iter().enumerate() {
            let want = blend(src[i] as f64 / 65535.0, dst[i] as f64 / 65535.0) * 65535.0;
            assert!(
                (*g as f64 - want).abs() <= 1.5,
                "{op:?} sample {i}: {g} vs {want}"
            );
        }
    }
}

#[test]
fn difference_watermark_and_clut_are_exact_at_depth() {
    let (w, h) = (9usize, 5usize);
    for ch in [1usize, 3, 4] {
        let a = noise16(w * h * ch, 21);
        let b = noise16(w * h * ch, 22);
        let p = VideoStreamParams::new(fmt16(ch), w as u32, h as u32);
        // |a − b| on every channel.
        let out = Difference::new()
            .apply_two(&frame16(w, ch, &a), &frame16(w, ch, &b), p)
            .unwrap();
        for (i, g) in codes16(&out).iter().enumerate() {
            assert_eq!(*g, a[i].abs_diff(b[i]), "difference ch{ch} sample {i}");
        }

        // A 3×2 overlay placed at (4, 2) with opacity ½:
        //   out = base (1 − a) + overlay a,  a = overlay alpha · ½.
        let (ow, oh) = (3usize, 2usize);
        let over = noise16(ow * oh * ch, 23);
        let out = Watermark::new(4, 2, ow as u32, oh as u32)
            .with_opacity(0.5)
            .apply_two(&frame16(w, ch, &a), &frame16(ow, ch, &over), p)
            .unwrap();
        let got = codes16(&out);
        let cc = if ch == 4 { 3 } else { ch };
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let i = (y * w + x) * ch + c;
                    let inside = (4..4 + ow).contains(&x) && (2..2 + oh).contains(&y);
                    if !inside || c >= cc {
                        assert_eq!(got[i], a[i], "watermark keeps ch{ch} {x},{y},{c}");
                        continue;
                    }
                    let o = ((y - 2) * ow + (x - 4)) * ch;
                    let alpha = if ch == 4 {
                        over[o + 3] as f64 / 65535.0 * 0.5
                    } else {
                        0.5
                    };
                    let want = a[i] as f64 * (1.0 - alpha) + over[o + c] as f64 * alpha;
                    assert!(
                        (got[i] as f64 - want).abs() <= 0.51,
                        "watermark ch{ch} {x},{y},{c}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }

        // CLUT: sample v picks entry ⌊v (n − 1)⌋ of the table and takes
        // that entry's own channel (grey: its value), kept at 16 bits.
        let table = noise16(w * h * ch, 24);
        let out = Clut::new()
            .apply_two(&frame16(w, ch, &a), &frame16(w, ch, &table), p)
            .unwrap();
        let got = codes16(&out);
        let n = w * h;
        for i in 0..w * h * ch {
            let c = i % ch;
            if c >= cc {
                assert_eq!(got[i], a[i]);
                continue;
            }
            let t = a[i] as f64 / 65535.0 * (n - 1) as f64;
            // Skip samples within rounding of an entry boundary.
            if (t - t.round()).abs() < 1e-3 {
                continue;
            }
            assert_eq!(
                got[i],
                table[t.floor() as usize * ch + c],
                "clut ch{ch} sample {i}"
            );
        }
    }
}

#[test]
fn identity_hald_cube_returns_the_picture_and_a_cube_is_trilinear() {
    // This crate's level-2 cube is a 4×4 picture holding the 2×2×2
    // lattice at `x = r + b·L`, `y = g` (the rows below are unused):
    // the lattice colours are the corners of the RGB cube, so trilinear
    // interpolation of the identity cube is the identity and of a
    // negated cube the negative.
    let l = 2u32;
    let side = (l * l) as usize;
    let p = VideoStreamParams::new(PixelFormat::RgbF32Le, side as u32, side as u32);
    let mut identity = vec![0.0f32; side * side * 3];
    for b in 0..l {
        for g in 0..l {
            for r in 0..l {
                let i = (g as usize * side + (r + b * l) as usize) * 3;
                identity[i..i + 3].copy_from_slice(&[r as f32, g as f32, b as f32]);
            }
        }
    }
    let picture: Vec<f32> = noise16(side * side * 3, 31)
        .iter()
        .map(|c| *c as f32 / 65535.0)
        .collect();
    let out = HaldClut::new()
        .apply_two(
            &frame_f32(side, 3, &picture),
            &frame_f32(side, 3, &identity),
            p,
        )
        .unwrap();
    for (g, v) in floats(&out).iter().zip(&picture) {
        assert!((g - v).abs() < 1e-6, "{g} vs {v}");
    }
    // Negate the lattice entries (the unused rows stay as they are).
    let mut negated = identity.clone();
    for v in &mut negated[..side * l as usize * 3] {
        *v = 1.0 - *v;
    }
    let out = HaldClut::new()
        .apply_two(
            &frame_f32(side, 3, &picture),
            &frame_f32(side, 3, &negated),
            p,
        )
        .unwrap();
    for (g, v) in floats(&out).iter().zip(&picture) {
        assert!((g - (1.0 - v)).abs() < 1e-6);
    }
    // The same through 16-bit words: exact to the code.
    let p16 = VideoStreamParams::new(PixelFormat::Rgb48Le, side as u32, side as u32);
    let pic16 = noise16(side * side * 3, 32);
    let id16: Vec<u16> = identity.iter().map(|v| (*v * 65535.0) as u16).collect();
    let out = HaldClut::new()
        .apply_two(&frame16(side, 3, &pic16), &frame16(side, 3, &id16), p16)
        .unwrap();
    for (g, v) in codes16(&out).iter().zip(&pic16) {
        assert!(g.abs_diff(*v) <= 1, "{g} vs {v}");
    }
}

#[test]
fn a_neutral_displacement_map_is_the_identity_and_a_shift_is_a_shift() {
    let (w, h) = (8usize, 6usize);
    let pic: Vec<f32> = noise16(w * h, 41)
        .iter()
        .map(|c| *c as f32 / 65535.0)
        .collect();
    let p = VideoStreamParams::new(PixelFormat::GrayF32Le, w as u32, h as u32);
    // Map value ½ displaces by nothing.
    let neutral = vec![0.5f32; w * h];
    let out = DisplacementMap::new(10.0)
        .apply_two(&frame_f32(w, 1, &pic), &frame_f32(w, 1, &neutral), p)
        .unwrap();
    assert_eq!(floats(&out), pic);
    // Map value 1 with scale 2 displaces by (1 − ½)·2 = one whole pixel
    // on both axes: out(x, y) = in(x + 1, y + 1), edges replicated.
    let full = vec![1.0f32; w * h];
    let out = DisplacementMap::new(2.0)
        .apply_two(&frame_f32(w, 1, &pic), &frame_f32(w, 1, &full), p)
        .unwrap();
    let got = floats(&out);
    for y in 0..h {
        for x in 0..w {
            let want = pic[(y + 1).min(h - 1) * w + (x + 1).min(w - 1)];
            assert_eq!(got[y * w + x], want, "{x},{y}");
        }
    }
}
