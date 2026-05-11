//! Per-pixel mathematical-function map (`-function <kind> args`).
//!
//! Walks every tone sample through a parameterised function in the
//! normalised `[0, 1]` space, then clamps the result back to
//! `[0, 255]`. Mirrors ImageMagick's `-function` family: callers pick
//! the function "kind" via [`FunctionOp`] and pass a slice of `f32`
//! arguments whose meaning depends on the kind.
//!
//! Clean-room implementation: every kind is a textbook closed-form
//! expression evaluated once per input byte and cached in a 256-entry
//! LUT.
//!
//! # Supported kinds
//!
//! - [`FunctionOp::Polynomial`] — `Σ aᵢ · xⁱ` evaluated in normalised
//!   `[0, 1]`. Args are coefficients in **descending** order:
//!   `[a_n, ..., a_1, a_0]` (matches IM's argument order).
//! - [`FunctionOp::Sinusoid`] — `bias + amp · sin(2π · (freq · x + phase / 360))`.
//!   Args: `[freq, phase_deg, amp, bias]`. `freq` defaults to 1, `phase`
//!   to 0, `amp` to 0.5, `bias` to 0.5 (matches IM defaults).
//! - [`FunctionOp::ArcSin`] — `bias + amp · asin(2 · (x - centre) / width) / π`.
//!   Args: `[width, centre, amp, bias]` with sensible defaults.
//! - [`FunctionOp::ArcTan`] — `bias + amp · atan(slope · (x - centre)) / π`.
//!   Args: `[slope, centre, amp, bias]`.
//!
//! # Pixel formats
//!
//! Same convention as [`Evaluate`](crate::Evaluate): `Gray8`/`Rgb24`
//! transform every channel; `Rgba` preserves alpha; `Yuv*P` only the
//! Y plane.

use crate::{is_supported_format, tonal_lut::apply_tone_lut, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Which mathematical kind the [`Function`] filter applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionOp {
    /// `Σ aᵢ · xⁱ`, args descending.
    Polynomial,
    /// `bias + amp · sin(2π · (freq · x + phase / 360))`.
    Sinusoid,
    /// `bias + amp · asin(2 · (x - centre) / width) / π`.
    ArcSin,
    /// `bias + amp · atan(slope · (x - centre)) / π`.
    ArcTan,
}

impl FunctionOp {
    /// Public name used by the JSON registry.
    pub fn name(self) -> &'static str {
        match self {
            FunctionOp::Polynomial => "polynomial",
            FunctionOp::Sinusoid => "sinusoid",
            FunctionOp::ArcSin => "arcsin",
            FunctionOp::ArcTan => "arctan",
        }
    }

    /// Parse the short name (case-insensitive matching alternatives
    /// accepted).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "polynomial" | "poly" => FunctionOp::Polynomial,
            "sinusoid" | "sin" => FunctionOp::Sinusoid,
            "arcsin" | "asin" => FunctionOp::ArcSin,
            "arctan" | "atan" => FunctionOp::ArcTan,
            _ => return None,
        })
    }
}

/// Per-pixel mathematical function map.
#[derive(Clone, Debug)]
pub struct Function {
    pub op: FunctionOp,
    pub args: Vec<f32>,
}

impl Function {
    /// Build a function transform with the given op + arg vector.
    pub fn new(op: FunctionOp, args: Vec<f32>) -> Self {
        Self { op, args }
    }
}

impl ImageFilter for Function {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Function does not yet handle {:?}",
                params.format
            )));
        }
        let lut = build_lut(self.op, &self.args);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut, params.format, params.width, params.height);
        Ok(out)
    }
}

fn build_lut(op: FunctionOp, args: &[f32]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let x = i as f32 / 255.0;
        let y = eval(op, args, x);
        *slot = clamp_round(y * 255.0);
    }
    lut
}

fn eval(op: FunctionOp, args: &[f32], x: f32) -> f32 {
    match op {
        FunctionOp::Polynomial => {
            // Horner's rule on coefficients given in descending order:
            // args = [a_n, ..., a_1, a_0]
            // y = ((a_n * x + a_{n-1}) * x + ...) * x + a_0
            if args.is_empty() {
                return x;
            }
            let mut acc = args[0];
            for &c in &args[1..] {
                acc = acc * x + c;
            }
            acc
        }
        FunctionOp::Sinusoid => {
            let freq = args.first().copied().unwrap_or(1.0);
            let phase_deg = args.get(1).copied().unwrap_or(0.0);
            let amp = args.get(2).copied().unwrap_or(0.5);
            let bias = args.get(3).copied().unwrap_or(0.5);
            let phase_rad = phase_deg.to_radians();
            let theta = 2.0 * std::f32::consts::PI * freq * x + phase_rad;
            bias + amp * theta.sin()
        }
        FunctionOp::ArcSin => {
            // y = bias + amp * asin(2*(x - centre)/width) / π
            let width = args.first().copied().unwrap_or(1.0);
            let centre = args.get(1).copied().unwrap_or(0.5);
            let amp = args.get(2).copied().unwrap_or(1.0);
            let bias = args.get(3).copied().unwrap_or(0.5);
            if width == 0.0 {
                return bias;
            }
            let arg = (2.0 * (x - centre) / width).clamp(-1.0, 1.0);
            bias + amp * arg.asin() / std::f32::consts::PI
        }
        FunctionOp::ArcTan => {
            // y = bias + amp * atan(slope * (x - centre)) / π
            let slope = args.first().copied().unwrap_or(1.0);
            let centre = args.get(1).copied().unwrap_or(0.5);
            let amp = args.get(2).copied().unwrap_or(1.0);
            let bias = args.get(3).copied().unwrap_or(0.5);
            bias + amp * (slope * (x - centre)).atan() / std::f32::consts::PI
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
    use oxideav_core::{PixelFormat, VideoPlane};

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

    fn run(op: FunctionOp, args: Vec<f32>, src: u8) -> u8 {
        let f = Function::new(op, args);
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
    fn polynomial_identity_via_x() {
        // y = x → coefficients [1, 0] (a_1 = 1, a_0 = 0).
        for v in [0u8, 64, 128, 192, 255] {
            assert_eq!(run(FunctionOp::Polynomial, vec![1.0, 0.0], v), v);
        }
    }

    #[test]
    fn polynomial_invert_via_minus_one_x_plus_one() {
        // y = -x + 1 → [a_1, a_0] = [-1, 1] → 0 → 255, 255 → 0.
        assert_eq!(run(FunctionOp::Polynomial, vec![-1.0, 1.0], 0), 255);
        assert_eq!(run(FunctionOp::Polynomial, vec![-1.0, 1.0], 255), 0);
    }

    #[test]
    fn polynomial_quadratic() {
        // y = x² → [1, 0, 0] → 0.5 → 0.25 ≈ 64.
        let v = run(FunctionOp::Polynomial, vec![1.0, 0.0, 0.0], 128);
        assert!((v as i32 - 64).abs() <= 2, "v = {v}");
    }

    #[test]
    fn sinusoid_defaults_match_im() {
        // freq=1, phase=0, amp=0.5, bias=0.5
        // x = 0   → 0.5 + 0.5*sin(0)   = 0.5 → 128
        // x = 0.25 → 0.5 + 0.5*sin(π/2) = 1.0 → 255
        // x = 0.5  → 0.5 + 0.5*sin(π)   = 0.5 → 128
        // x = 0.75 → 0.5 + 0.5*sin(3π/2) = 0.0 → 0
        assert!((run(FunctionOp::Sinusoid, vec![], 0) as i32 - 128).abs() <= 1);
        assert!((run(FunctionOp::Sinusoid, vec![], 64) as i32 - 255).abs() <= 2);
        // x = 128/255 ≈ 0.5019 → sin slightly off zero; allow ~3 bytes
        // of slack to stay tolerant of f32 rounding inside the LUT build.
        assert!((run(FunctionOp::Sinusoid, vec![], 128) as i32 - 128).abs() <= 3);
        assert!((run(FunctionOp::Sinusoid, vec![], 192) as i32).abs() <= 2);
    }

    #[test]
    fn arctan_passes_through_centre() {
        // At x = 0.5 (centre), atan(0) = 0 → bias.
        assert!((run(FunctionOp::ArcTan, vec![1.0, 0.5, 1.0, 0.5], 128) as i32 - 128).abs() <= 1);
    }

    #[test]
    fn arcsin_clamps_outside_window() {
        // width 0.5, centre 0.5 → x outside [0.25, 0.75] hits ±π/2.
        // bias 0.5, amp 1.0 → 0.5 ± 0.5 → 0 or 255.
        assert_eq!(run(FunctionOp::ArcSin, vec![0.5, 0.5, 1.0, 0.5], 0), 0);
        assert_eq!(run(FunctionOp::ArcSin, vec![0.5, 0.5, 1.0, 0.5], 255), 255);
    }

    #[test]
    fn name_round_trip() {
        for op in [
            FunctionOp::Polynomial,
            FunctionOp::Sinusoid,
            FunctionOp::ArcSin,
            FunctionOp::ArcTan,
        ] {
            assert_eq!(FunctionOp::from_name(op.name()), Some(op));
        }
        assert_eq!(FunctionOp::from_name("poly"), Some(FunctionOp::Polynomial));
        assert_eq!(FunctionOp::from_name("sin"), Some(FunctionOp::Sinusoid));
        assert_eq!(FunctionOp::from_name("nope"), None);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|i| [50u8, 100, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Function::new(FunctionOp::Polynomial, vec![-1.0, 1.0])
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
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8);
        }
    }
}
