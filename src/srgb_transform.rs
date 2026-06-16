//! sRGB transfer-function transform (encode / decode between display-encoded
//! and linear-light pixel values).
//!
//! Pixel samples stored in an 8-bit image are almost always **encoded** with a
//! non-linear transfer function so the quantisation steps track human
//! brightness perception. Many physically-meaningful operations — additive
//! light, blending, blurring, alpha compositing — are only correct when the
//! samples are in **linear light**. This filter exposes the documented display
//! transfer function (`docs/image/filter/tone-mapping-operators.md` §5.2) as a
//! standalone LUT primitive so a pipeline can round-trip in and out of linear
//! light around such an operation.
//!
//! Two transfer functions are offered:
//!
//! - [`SrgbCurve::Srgb`] (default) — the IEC 61966-2-1 piecewise sRGB curve.
//!   Encode (OETF, linear → encoded) per §5.2:
//!
//!   ```text
//!   V = 12.92 · Lin                      if Lin ≤ 0.0031308
//!   V = 1.055 · Lin^(1/2.4) − 0.055      otherwise
//!   ```
//!
//!   Decode (EOTF, encoded → linear) is the analytic inverse:
//!
//!   ```text
//!   Lin = V / 12.92                      if V ≤ 0.04045
//!   Lin = ((V + 0.055) / 1.055)^2.4      otherwise
//!   ```
//!
//! - [`SrgbCurve::Gamma`] — the pure power-law approximation of §5.2:
//!   encode `V = Lin^(1/γ)`, decode `Lin = V^γ` (`γ ≈ 2.2` for a generic
//!   display). Cheaper, no piecewise linear toe; use it when colour accuracy
//!   does not matter.
//!
//! [`SrgbDirection`] selects which way the transform runs:
//! [`SrgbDirection::Encode`] takes linear-light samples to display-encoded
//! ones (the OETF), [`SrgbDirection::Decode`] (default) takes display-encoded
//! samples to linear light (the EOTF). Encoding then decoding the same curve is
//! an identity within 8-bit rounding.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`. YUV inputs return `Unsupported`
//! because the transfer function is defined on RGB-space tone channels. Alpha
//! is a linear coverage value, not a light value, so it is **passed through**
//! unchanged on `Rgba`. Cost is one 256-entry LUT build plus `O(W·H)`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which transfer curve the transform uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrgbCurve {
    /// IEC 61966-2-1 piecewise sRGB curve (linear toe + `^2.4` shoulder).
    Srgb,
    /// Pure power-law `Lin^(1/γ)` / `V^γ` with the configured exponent
    /// (stored as `γ × 1000` so the enum stays `Copy + Eq`).
    Gamma { gamma_milli: u32 },
}

impl SrgbCurve {
    /// Pure power-law curve with display gamma `γ`. Non-finite or
    /// non-positive values fall back to `2.2`.
    pub fn gamma(g: f32) -> Self {
        let g = if g.is_finite() && g > 0.0 { g } else { 2.2 };
        Self::Gamma {
            gamma_milli: (g * 1000.0).round().clamp(1.0, 1_000_000.0) as u32,
        }
    }

    fn gamma_value(self) -> f32 {
        match self {
            Self::Gamma { gamma_milli } => gamma_milli as f32 / 1000.0,
            Self::Srgb => 2.4,
        }
    }
}

/// Direction of the transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SrgbDirection {
    /// Linear light → display-encoded (the OETF).
    Encode,
    /// Display-encoded → linear light (the EOTF). The default.
    #[default]
    Decode,
}

/// sRGB / power-law transfer-function transform.
#[derive(Clone, Copy, Debug)]
pub struct SrgbTransform {
    /// Which transfer curve to apply.
    pub curve: SrgbCurve,
    /// Encode (to display) or decode (to linear).
    pub direction: SrgbDirection,
}

impl Default for SrgbTransform {
    fn default() -> Self {
        Self {
            curve: SrgbCurve::Srgb,
            direction: SrgbDirection::Decode,
        }
    }
}

/// Encode a normalised linear value through the piecewise sRGB OETF.
fn srgb_encode(lin: f32) -> f32 {
    let lin = lin.clamp(0.0, 1.0);
    if lin <= 0.003_130_8 {
        12.92 * lin
    } else {
        1.055 * lin.powf(1.0 / 2.4) - 0.055
    }
}

/// Decode a normalised display-encoded value through the piecewise sRGB EOTF.
fn srgb_decode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

impl SrgbTransform {
    /// Decode display-encoded sRGB samples to linear light (the default).
    pub fn decode() -> Self {
        Self {
            curve: SrgbCurve::Srgb,
            direction: SrgbDirection::Decode,
        }
    }

    /// Encode linear-light samples to display-encoded sRGB.
    pub fn encode() -> Self {
        Self {
            curve: SrgbCurve::Srgb,
            direction: SrgbDirection::Encode,
        }
    }

    /// Set the transfer curve.
    pub fn with_curve(mut self, curve: SrgbCurve) -> Self {
        self.curve = curve;
        self
    }

    /// Set the transform direction.
    pub fn with_direction(mut self, direction: SrgbDirection) -> Self {
        self.direction = direction;
        self
    }

    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32 / 255.0;
            let y = match (self.curve, self.direction) {
                (SrgbCurve::Srgb, SrgbDirection::Encode) => srgb_encode(x),
                (SrgbCurve::Srgb, SrgbDirection::Decode) => srgb_decode(x),
                (SrgbCurve::Gamma { .. }, SrgbDirection::Encode) => {
                    // OETF: V = Lin^(1/γ).
                    x.clamp(0.0, 1.0).powf(1.0 / self.curve.gamma_value())
                }
                (SrgbCurve::Gamma { .. }, SrgbDirection::Decode) => {
                    // EOTF: Lin = V^γ.
                    x.clamp(0.0, 1.0).powf(self.curve.gamma_value())
                }
            };
            *slot = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        lut
    }
}

impl ImageFilter for SrgbTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: SrgbTransform requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: SrgbTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let stride = w * bpp;
        let lut = self.build_lut();
        let src = &input.planes[0];
        let mut out = vec![0u8; stride * h];
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + stride];
            let d = &mut out[y * stride..(y + 1) * stride];
            match bpp {
                1 => {
                    for (di, si) in d.iter_mut().zip(s.iter()) {
                        *di = lut[*si as usize];
                    }
                }
                3 => {
                    for x in 0..w {
                        let b = x * 3;
                        d[b] = lut[s[b] as usize];
                        d[b + 1] = lut[s[b + 1] as usize];
                        d[b + 2] = lut[s[b + 2] as usize];
                    }
                }
                4 => {
                    for x in 0..w {
                        let b = x * 4;
                        d[b] = lut[s[b] as usize];
                        d[b + 1] = lut[s[b + 1] as usize];
                        d[b + 2] = lut[s[b + 2] as usize];
                        // Alpha is coverage, not light — pass through.
                        d[b + 3] = s[b + 3];
                    }
                }
                _ => unreachable!(),
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data: out }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn endpoints_are_fixed_points() {
        // 0 → 0 and 255 → 255 for both directions of both curves.
        for f in [
            SrgbTransform::decode(),
            SrgbTransform::encode(),
            SrgbTransform::default().with_curve(SrgbCurve::gamma(2.2)),
            SrgbTransform::default()
                .with_curve(SrgbCurve::gamma(2.2))
                .with_direction(SrgbDirection::Encode),
        ] {
            let input = rgb(
                2,
                1,
                |x, _| {
                    if x == 0 {
                        [0, 0, 0]
                    } else {
                        [255, 255, 255]
                    }
                },
            );
            let out = f.apply(&input, p_rgb(2, 1)).unwrap();
            assert_eq!(&out.planes[0].data[0..3], &[0, 0, 0]);
            assert_eq!(&out.planes[0].data[3..6], &[255, 255, 255]);
        }
    }

    #[test]
    fn decode_then_encode_round_trips_within_rounding() {
        // Apply EOTF (to linear) then OETF (back to display). For midtones
        // and highlights the round-trip is near-lossless; deep shadows are
        // crushed because the EOTF maps the bottom of the encoded range
        // onto only a handful of linear codes (8-bit shadow precision is
        // inherent to the transfer pair, not a bug), so we test from a
        // floor of 64 where the curve is well-conditioned.
        let input = rgb(16, 16, |x, y| {
            [
                64 + (x * 12) as u8,
                64 + (y * 12) as u8,
                64 + ((x + y) * 6) as u8,
            ]
        });
        let lin = SrgbTransform::decode()
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        let back = SrgbTransform::encode()
            .apply(
                &VideoFrame {
                    pts: None,
                    planes: lin.planes,
                },
                p_rgb(16, 16),
            )
            .unwrap();
        for (a, b) in back.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!(
                (*a as i16 - *b as i16).abs() <= 2,
                "round-trip got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn srgb_decode_darkens_midtones() {
        // Display-encoded sRGB mid-grey 188 decodes to linear ~0.5 → 128.
        // Decoding always pushes a midtone *down* (encoded is brighter than
        // its linear value above the toe).
        let input = rgb(1, 1, |_, _| [188, 188, 188]);
        let out = SrgbTransform::decode().apply(&input, p_rgb(1, 1)).unwrap();
        let v = out.planes[0].data[0];
        assert!(v < 188, "decode must darken an encoded midtone, got {v}");
        // sRGB 188/255 ≈ 0.737 → linear ≈ 0.503 → ~128.
        assert!((v as i16 - 128).abs() <= 3, "expected ~128, got {v}");
    }

    #[test]
    fn srgb_encode_brightens_midtones() {
        // Linear 128 (≈0.502) encodes up to sRGB ≈188.
        let input = rgb(1, 1, |_, _| [128, 128, 128]);
        let out = SrgbTransform::encode().apply(&input, p_rgb(1, 1)).unwrap();
        let v = out.planes[0].data[0];
        assert!(v > 128, "encode must brighten a linear midtone, got {v}");
        assert!((v as i16 - 188).abs() <= 3, "expected ~188, got {v}");
    }

    #[test]
    fn gamma_decode_matches_power_law() {
        // γ=2.2 decode of v → v^2.2. At 128 (0.502) → 0.224 → 57.
        let input = rgb(1, 1, |_, _| [128, 128, 128]);
        let out = SrgbTransform::default()
            .with_curve(SrgbCurve::gamma(2.2))
            .with_direction(SrgbDirection::Decode)
            .apply(&input, p_rgb(1, 1))
            .unwrap();
        let expect = ((128.0f32 / 255.0).powf(2.2) * 255.0).round() as u8;
        assert_eq!(out.planes[0].data[0], expect);
    }

    #[test]
    fn gamma_round_trips() {
        // As with the sRGB curve, the power-law decode crushes deep shadows
        // into a few linear codes, so round-trip fidelity is only tight for
        // midtones / highlights (floor of 96 here).
        let input = rgb(8, 8, |x, y| [96 + (x * 18) as u8, 96 + (y * 18) as u8, 150]);
        let dec = SrgbTransform::default()
            .with_curve(SrgbCurve::gamma(2.4))
            .with_direction(SrgbDirection::Decode)
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        let enc = SrgbTransform::default()
            .with_curve(SrgbCurve::gamma(2.4))
            .with_direction(SrgbDirection::Encode)
            .apply(
                &VideoFrame {
                    pts: None,
                    planes: dec.planes,
                },
                p_rgb(8, 8),
            )
            .unwrap();
        for (a, b) in enc.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 3, "got {a}, expected {b}");
        }
    }

    #[test]
    fn invalid_gamma_falls_back() {
        // Non-finite / non-positive γ must not panic and must default.
        let c = SrgbCurve::gamma(f32::NAN);
        assert_eq!(c, SrgbCurve::gamma(2.2));
        let c2 = SrgbCurve::gamma(-1.0);
        assert_eq!(c2, SrgbCurve::gamma(2.2));
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[200u8, 100, 50, 77]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = SrgbTransform::decode()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 77, "alpha must pass through");
        }
    }

    #[test]
    fn gray8_supported() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0, 64, 128, 255],
            }],
        };
        let out = SrgbTransform::decode()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 0);
        assert_eq!(out.planes[0].data[3], 255);
        // Decoded midtones drop.
        assert!(out.planes[0].data[2] < 128);
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let err = SrgbTransform::decode()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("SrgbTransform"));
    }

    #[test]
    fn empty_planes_error() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![],
        };
        let err = SrgbTransform::decode()
            .apply(&frame, p_rgb(2, 2))
            .unwrap_err();
        assert!(format!("{err}").contains("non-empty"));
    }
}
