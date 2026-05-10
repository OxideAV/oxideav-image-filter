//! Per-pixel arithmetic evaluation (`-evaluate Op value`).
//!
//! Applies a single arithmetic operator to every channel of every
//! pixel with a constant scalar operand. Mirrors ImageMagick's
//! `-evaluate <op> <value>` flag — common useful operators are
//! `Add`, `Subtract`, `Multiply`, `Divide`, `Pow`, `Max`, `Min`,
//! `Set` (replace), `And`, `Or`, `Xor`, and `Threshold`.
//!
//! Clean-room implementation from the standard arithmetic definitions;
//! every operator is independent of every other so this is essentially
//! a switch on the [`EvaluateOp`] discriminator wrapped around a
//! per-channel scalar map.
//!
//! # Pixel formats
//!
//! - `Gray8` / `Rgb24`: every channel is updated.
//! - `Rgba`: alpha is preserved (matching IM `-channel rgb -evaluate`).
//! - `Yuv*P`: only the Y plane is touched; chroma is pass-through.
//!   Arithmetic on chroma planes would wreck the colour interpretation
//!   (Cb / Cr are signed-around-128, not linear intensities) — callers
//!   wanting per-channel chroma maths should convert to RGB first.
//!
//! # Saturation
//!
//! All operators clamp the per-channel result to `[0, 255]`. `Divide`
//! by zero saturates to 255. `Pow` with a negative base saturates to 0.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Which per-pixel arithmetic operator to apply.
///
/// Names match ImageMagick's `-evaluate` operator argument so the
/// pipeline can map a JSON `op` string straight onto a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluateOp {
    /// `out = clamp(src + value, 0, 255)`.
    Add,
    /// `out = clamp(src - value, 0, 255)`.
    Subtract,
    /// `out = clamp(src * value, 0, 255)`. `value` is a real scalar
    /// (not normalised to `1.0 = unity`), so a value of `2.0` doubles.
    Multiply,
    /// `out = clamp(src / value, 0, 255)`. `value == 0` saturates
    /// every sample to `255`.
    Divide,
    /// `out = clamp((src/255)^value * 255, 0, 255)`. Power applied
    /// in normalised `[0, 1]` space.
    Pow,
    /// `out = max(src, value)`.
    Max,
    /// `out = min(src, value)`.
    Min,
    /// `out = value` for every pixel (`value` clamped to `[0, 255]`).
    Set,
    /// `out = src & (value as u8)` (bitwise AND on the integer sample).
    And,
    /// `out = src | (value as u8)`.
    Or,
    /// `out = src ^ (value as u8)`.
    Xor,
    /// `out = if src >= value { 255 } else { 0 }`. Equivalent to the
    /// existing [`Threshold`](crate::Threshold) filter on RGB/Gray
    /// inputs (kept as an `EvaluateOp` so the same flag surface
    /// covers IM's `-evaluate Threshold N`).
    Threshold,
}

impl EvaluateOp {
    /// Public name used by the JSON registry (`evaluate` factory).
    pub fn name(self) -> &'static str {
        match self {
            EvaluateOp::Add => "add",
            EvaluateOp::Subtract => "subtract",
            EvaluateOp::Multiply => "multiply",
            EvaluateOp::Divide => "divide",
            EvaluateOp::Pow => "pow",
            EvaluateOp::Max => "max",
            EvaluateOp::Min => "min",
            EvaluateOp::Set => "set",
            EvaluateOp::And => "and",
            EvaluateOp::Or => "or",
            EvaluateOp::Xor => "xor",
            EvaluateOp::Threshold => "threshold",
        }
    }

    /// Parse the short operator name (`"add"`, `"pow"`, …). Used by the
    /// registry factory.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "add" => EvaluateOp::Add,
            "subtract" | "sub" => EvaluateOp::Subtract,
            "multiply" | "mul" => EvaluateOp::Multiply,
            "divide" | "div" => EvaluateOp::Divide,
            "pow" | "power" => EvaluateOp::Pow,
            "max" => EvaluateOp::Max,
            "min" => EvaluateOp::Min,
            "set" => EvaluateOp::Set,
            "and" => EvaluateOp::And,
            "or" => EvaluateOp::Or,
            "xor" => EvaluateOp::Xor,
            "threshold" => EvaluateOp::Threshold,
            _ => return None,
        })
    }
}

/// Per-pixel arithmetic filter — apply [`EvaluateOp`] with the given
/// scalar `value` to every channel of every pixel.
#[derive(Clone, Copy, Debug)]
pub struct Evaluate {
    pub op: EvaluateOp,
    pub value: f32,
}

impl Evaluate {
    /// Build an evaluator with the given operator + scalar.
    pub fn new(op: EvaluateOp, value: f32) -> Self {
        Self { op, value }
    }
}

impl ImageFilter for Evaluate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Evaluate does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut out = input.clone();

        let lut = build_lut(self.op, self.value);

        match params.format {
            PixelFormat::Gray8 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + 3 * w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        for ch in 0..3 {
                            let v = &mut row[4 * x + ch];
                            *v = lut[*v as usize];
                        }
                        // alpha (index 3) preserved
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let y_plane = &mut out.planes[0];
                let stride = y_plane.stride;
                for y in 0..h {
                    for v in &mut y_plane.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
                // chroma planes pass-through
            }
            _ => unreachable!("is_supported gate already rejected this format"),
        }
        Ok(out)
    }
}

/// Pre-compute the per-input-byte mapping table once, then apply
/// it to every sample. Saves per-pixel branches and floating-point
/// work — every operator collapses into a 256-byte LUT.
fn build_lut(op: EvaluateOp, value: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = apply_scalar(op, i as u8, value);
    }
    lut
}

fn apply_scalar(op: EvaluateOp, src: u8, value: f32) -> u8 {
    let s = src as f32;
    let v = value;
    match op {
        EvaluateOp::Add => clamp_round(s + v),
        EvaluateOp::Subtract => clamp_round(s - v),
        EvaluateOp::Multiply => clamp_round(s * v),
        EvaluateOp::Divide => {
            if v == 0.0 {
                255
            } else {
                clamp_round(s / v)
            }
        }
        EvaluateOp::Pow => {
            let n = (s / 255.0).max(0.0);
            let r = n.powf(v);
            if r.is_finite() {
                clamp_round(r * 255.0)
            } else {
                0
            }
        }
        EvaluateOp::Max => clamp_round(s.max(v)),
        EvaluateOp::Min => clamp_round(s.min(v)),
        EvaluateOp::Set => clamp_round(v),
        EvaluateOp::And => {
            let mask = clamp_round(v);
            src & mask
        }
        EvaluateOp::Or => {
            let mask = clamp_round(v);
            src | mask
        }
        EvaluateOp::Xor => {
            let mask = clamp_round(v);
            src ^ mask
        }
        EvaluateOp::Threshold => {
            let t = clamp_round(v);
            if src >= t {
                255
            } else {
                0
            }
        }
    }
}

#[inline]
fn clamp_round(v: f32) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn gray(w: u32, h: u32, pat: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pat(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    fn run(op: EvaluateOp, value: f32, src: u8) -> u8 {
        let f = Evaluate::new(op, value);
        let input = gray(1, 1, |_, _| src);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        out.planes[0].data[0]
    }

    #[test]
    fn add_clamps_at_255() {
        assert_eq!(run(EvaluateOp::Add, 100.0, 200), 255);
        assert_eq!(run(EvaluateOp::Add, -10.0, 5), 0);
        assert_eq!(run(EvaluateOp::Add, 0.0, 128), 128);
    }

    #[test]
    fn subtract_clamps_at_zero() {
        assert_eq!(run(EvaluateOp::Subtract, 50.0, 30), 0);
        assert_eq!(run(EvaluateOp::Subtract, 50.0, 200), 150);
    }

    #[test]
    fn multiply_doubles_with_factor_two() {
        assert_eq!(run(EvaluateOp::Multiply, 2.0, 100), 200);
        assert_eq!(run(EvaluateOp::Multiply, 2.0, 200), 255);
        assert_eq!(run(EvaluateOp::Multiply, 0.5, 200), 100);
    }

    #[test]
    fn divide_by_zero_saturates_to_255() {
        assert_eq!(run(EvaluateOp::Divide, 0.0, 100), 255);
        assert_eq!(run(EvaluateOp::Divide, 2.0, 100), 50);
    }

    #[test]
    fn pow_in_normalised_space() {
        assert_eq!(run(EvaluateOp::Pow, 1.0, 128), 128);
        assert_eq!(run(EvaluateOp::Pow, 5.0, 255), 255);
        assert_eq!(run(EvaluateOp::Pow, 0.5, 0), 0);
    }

    #[test]
    fn max_min_set_basics() {
        assert_eq!(run(EvaluateOp::Max, 100.0, 50), 100);
        assert_eq!(run(EvaluateOp::Max, 100.0, 200), 200);
        assert_eq!(run(EvaluateOp::Min, 100.0, 50), 50);
        assert_eq!(run(EvaluateOp::Min, 100.0, 200), 100);
        assert_eq!(run(EvaluateOp::Set, 200.0, 50), 200);
        assert_eq!(run(EvaluateOp::Set, 200.0, 200), 200);
    }

    #[test]
    fn bitwise_ops() {
        assert_eq!(run(EvaluateOp::And, 0xF0 as f32, 0xAB), 0xA0);
        assert_eq!(run(EvaluateOp::Or, 0x0F as f32, 0xA0), 0xAF);
        assert_eq!(run(EvaluateOp::Xor, 0xFF as f32, 0xAB), 0x54);
    }

    #[test]
    fn threshold_op() {
        assert_eq!(run(EvaluateOp::Threshold, 128.0, 100), 0);
        assert_eq!(run(EvaluateOp::Threshold, 128.0, 200), 255);
        assert_eq!(run(EvaluateOp::Threshold, 128.0, 128), 255);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|i| [50u8, 150, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Evaluate::new(EvaluateOp::Set, 100.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[0], 100);
            assert_eq!(px[1], 100);
            assert_eq!(px[2], 100);
            assert_eq!(px[3], i as u8, "alpha preserved");
        }
    }

    #[test]
    fn yuv_chroma_passthrough() {
        let y: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let cb = vec![200u8; 4];
        let cr = vec![50u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane {
                    stride: 2,
                    data: cb.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: cr.clone(),
                },
            ],
        };
        let out = Evaluate::new(EvaluateOp::Add, 50.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for (i, v) in out.planes[0].data.iter().enumerate() {
            let expected = ((i * 16) as u16 + 50).min(255) as u8;
            assert_eq!(*v, expected);
        }
        assert_eq!(out.planes[1].data, cb);
        assert_eq!(out.planes[2].data, cr);
    }

    #[test]
    fn name_round_trip() {
        for op in [
            EvaluateOp::Add,
            EvaluateOp::Subtract,
            EvaluateOp::Multiply,
            EvaluateOp::Divide,
            EvaluateOp::Pow,
            EvaluateOp::Max,
            EvaluateOp::Min,
            EvaluateOp::Set,
            EvaluateOp::And,
            EvaluateOp::Or,
            EvaluateOp::Xor,
            EvaluateOp::Threshold,
        ] {
            assert_eq!(EvaluateOp::from_name(op.name()), Some(op));
        }
        assert_eq!(EvaluateOp::from_name("nope"), None);
        assert_eq!(EvaluateOp::from_name("sub"), Some(EvaluateOp::Subtract));
        assert_eq!(EvaluateOp::from_name("mul"), Some(EvaluateOp::Multiply));
        assert_eq!(EvaluateOp::from_name("div"), Some(EvaluateOp::Divide));
        assert_eq!(EvaluateOp::from_name("power"), Some(EvaluateOp::Pow));
    }
}
