//! Reinhard-extended (white-clamping, unkeyed) tone-mapping operator.
//!
//! Reinhard, Stark, Shirley, Ferwerda — "Photographic Tone Reproduction
//! for Digital Images" (SIGGRAPH 2002), §3.1 ("Burn-out"). The
//! "extended" variant uses the same white-point curve
//! `L_d = L · (1 + L / L_white²) / (1 + L)` but **without** the
//! geometric-mean (log-average) key-scaling pre-step of §3 — i.e. it
//! treats its scalar `L` input as the already-exposure-correct linear
//! luminance and simply applies a soft clamp at `L_white`. Algebraically
//! identical to the white-point form of the keyed operator; the
//! distinction is purely whether the key step is applied beforehand.
//!
//! See `docs/image/filter/tone-mapping-operators.md` §5.1.
//!
//! Why a separate type:
//!
//! - **Single pass.** No `log_avg` accumulation across the whole frame
//!   before the per-pixel curve; the curve is applied directly. Useful
//!   when the caller already has exposure-correct linear luminance
//!   (e.g. fixed-camera capture, pre-graded HDR plate).
//! - **One parameter.** Only `l_white` (the luminance that maps to
//!   pure white). When `l_white <= 0` the operator picks
//!   `L_white = L_max(frame)` per §5.1 — "typically set to the maximum
//!   scene luminance (so the brightest pixel maps exactly to 1)".
//! - **No `key`.** Removing the key knob makes the operator
//!   parameter-free in the auto-`L_max` mode, suitable for callers that
//!   want a deterministic clamp at the frame's existing range.
//!
//! Algorithm (clean-room from the paper, §3.1):
//!
//! 1. De-gamma sRGB → linear-light per channel.
//! 2. Compute scene luminance per pixel using the Rec.709
//!    `L = 0.2126·R + 0.7152·G + 0.0722·B` weights (matching the
//!    output sRGB primaries our crate consumes).
//! 3. Resolve `L_white`: if `self.l_white > 0` use it directly;
//!    otherwise set `L_white = max(L_w over the frame)` so the
//!    brightest pixel maps exactly to `1` (the §5.1 default).
//! 4. Apply `L_d = L_w · (1 + L_w / L_white²) / (1 + L_w)`.
//! 5. Re-modulate the original linear RGB by `L_d / L_w` (chroma
//!    preservation, paper §3.4).
//! 6. Re-encode through the sRGB OETF.
//!
//! Operates on `Rgb24` / `Rgba`. `Gray8` falls back to the scalar form
//! (no chroma preservation; the curve is applied to the single
//! channel). YUV returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Reinhard-extended (white-clamping, unkeyed) tone-mapping operator.
///
/// `l_white` is the linear luminance that maps to pure white. When
/// `l_white <= 0` (the default), `L_white` is auto-selected as the
/// per-frame maximum scene luminance — matching the §5.1 note that
/// `L_white` is "typically set to the maximum scene luminance (so the
/// brightest pixel maps exactly to 1)".
#[derive(Clone, Copy, Debug)]
pub struct ReinhardExtended {
    /// Linear luminance mapped to pure white. `<= 0` triggers
    /// auto-selection from the per-frame `L_max` (the §5.1 default).
    pub l_white: f32,
}

impl Default for ReinhardExtended {
    fn default() -> Self {
        // 0.0 → auto-pick L_white = L_max(frame).
        Self { l_white: 0.0 }
    }
}

impl ReinhardExtended {
    /// Construct with an explicit white-point luminance. Non-finite or
    /// non-positive values fall back to auto-`L_max` mode.
    pub fn new(l_white: f32) -> Self {
        let l_white = if l_white.is_finite() && l_white > 0.0 {
            l_white
        } else {
            0.0
        };
        Self { l_white }
    }

    /// Set the white-point explicitly (chainable). Non-finite or
    /// non-positive values switch to auto-`L_max` mode.
    pub fn with_white(mut self, l_white: f32) -> Self {
        self.l_white = if l_white.is_finite() && l_white > 0.0 {
            l_white
        } else {
            0.0
        };
        self
    }
}

#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl ImageFilter for ReinhardExtended {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ReinhardExtended requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: ReinhardExtended requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let stride_out = w * bpp;
        let src = &input.planes[0];
        const EPS: f32 = 1.0e-6;

        // Decode linear RGB + per-pixel luminance (Rec.709 weights).
        let n = w * h;
        let mut linr = vec![0f32; n];
        let mut ling = vec![0f32; n];
        let mut linb = vec![0f32; n];
        let mut lum = vec![0f32; n];
        let mut l_max = 0f32;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let b = y * src.stride + x * bpp;
                let (r_lin, g_lin, b_lin) = match bpp {
                    1 => {
                        let v = srgb_to_linear(src.data[b]);
                        (v, v, v)
                    }
                    _ => (
                        srgb_to_linear(src.data[b]),
                        srgb_to_linear(src.data[b + 1]),
                        srgb_to_linear(src.data[b + 2]),
                    ),
                };
                linr[i] = r_lin;
                ling[i] = g_lin;
                linb[i] = b_lin;
                // Rec.709 / sRGB linear-light luminance weights.
                let l = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
                lum[i] = l;
                if l > l_max {
                    l_max = l;
                }
            }
        }

        // Resolve L_white. If unset (<=0), pick per-frame L_max so the
        // brightest pixel maps exactly to 1 (doc §5.1 default).
        let l_white = if self.l_white > 0.0 {
            self.l_white
        } else if l_max > 0.0 {
            l_max
        } else {
            // All-black frame: any positive sentinel suffices; the
            // curve evaluates to 0/1 = 0 everywhere either way.
            1.0
        };
        let white2 = l_white * l_white;

        let mut out = vec![0u8; stride_out * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let l_w = lum[i];
                // Extended white-point curve, §5.1.
                let l_d = if l_w > 0.0 {
                    (l_w * (1.0 + l_w / white2)) / (1.0 + l_w)
                } else {
                    0.0
                };
                // Chroma-preserving re-modulation (paper §3.4).
                let g = if l_w > EPS { l_d / l_w } else { 0.0 };
                let r = linr[i] * g;
                let gv = ling[i] * g;
                let bv = linb[i] * g;
                let db = y * stride_out + x * bpp;
                match bpp {
                    1 => out[db] = linear_to_srgb(gv),
                    3 => {
                        out[db] = linear_to_srgb(r);
                        out[db + 1] = linear_to_srgb(gv);
                        out[db + 2] = linear_to_srgb(bv);
                    }
                    4 => {
                        out[db] = linear_to_srgb(r);
                        out[db + 1] = linear_to_srgb(gv);
                        out[db + 2] = linear_to_srgb(bv);
                        out[db + 3] = src.data[y * src.stride + x * 4 + 3];
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: stride_out,
                data: out,
            }],
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
    fn output_in_range() {
        let input = rgb(8, 8, |x, y| {
            [
                ((x * 32).min(255)) as u8,
                ((y * 32).min(255)) as u8,
                (((x + y) * 16).min(255)) as u8,
            ]
        });
        let out = ReinhardExtended::default()
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn auto_white_brightest_pixel_reaches_white() {
        // Single saturated-white pixel + surrounding mid-grey. In
        // auto-L_max mode the brightest input pixel must map to 1.0
        // (i.e. encoded sRGB = 255) per §5.1.
        let input = rgb(4, 4, |x, y| {
            if x == 0 && y == 0 {
                [255, 255, 255]
            } else {
                [128, 128, 128]
            }
        });
        let out = ReinhardExtended::default()
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Pixel (0,0) — the brightest — should encode at the sRGB
        // ceiling (255) since L_white = L_max maps it exactly to 1.
        assert_eq!(out.planes[0].data[0], 255);
        assert_eq!(out.planes[0].data[1], 255);
        assert_eq!(out.planes[0].data[2], 255);
    }

    #[test]
    fn black_input_stays_black() {
        // All-zero frame: every pixel maps to L_d = 0 → encoded 0.
        let input = rgb(4, 4, |_, _| [0, 0, 0]);
        let out = ReinhardExtended::default()
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&b| b == 0));
    }

    #[test]
    fn small_explicit_white_saturates_highlights_faster() {
        // Bright stripe (left half) + dark stripe (right half). With
        // an explicit small L_white the bright pixels saturate faster
        // than the auto-L_max default (which exactly maps the brightest
        // input to white but eases everything below).
        let input = rgb(
            4,
            4,
            |x, _| {
                if x < 2 {
                    [250, 250, 250]
                } else {
                    [60, 60, 60]
                }
            },
        );
        let auto = ReinhardExtended::default()
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        let small = ReinhardExtended::new(0.1)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        let auto_bright = auto.planes[0].data[0];
        let small_bright = small.planes[0].data[0];
        // Small white-point clips at saturation; auto-L_max maps the
        // brightest pixel exactly to 1 (also saturated). So a small
        // L_white should be **at least as bright** as the auto mode
        // on the bright stripe — and strictly brighter on the *dark*
        // stripe (since the dark pixel's L is closer to L_white now).
        assert!(
            small_bright >= auto_bright,
            "small L_white must not be dimmer than auto on the bright stripe \
             (auto={auto_bright}, small={small_bright})"
        );
        // Dark stripe occupies cols 2..4; sample row 1, col 2 (byte
        // offset = row * stride + col * bpp = 1*12 + 2*3 = 18).
        let auto_dark = auto.planes[0].data[18];
        let small_dark = small.planes[0].data[18];
        assert!(
            small_dark > auto_dark,
            "dark stripe should lift more with a smaller L_white \
             (auto={auto_dark}, small={small_dark})"
        );
    }

    #[test]
    fn explicit_white_equals_lmax_matches_auto_mode() {
        // §5.1: setting L_white explicitly to the per-frame L_max
        // should be byte-equivalent to the auto-L_max default.
        let input = rgb(8, 8, |x, y| {
            [
                ((x * 20).min(255)) as u8,
                ((y * 20).min(255)) as u8,
                (((x + y) * 10).min(255)) as u8,
            ]
        });
        let auto = ReinhardExtended::default()
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        // Re-derive L_max the same way the operator does.
        let src = &input.planes[0];
        let mut l_max = 0f32;
        for y in 0..8 {
            for x in 0..8 {
                let b = y * src.stride + x * 3;
                let r_lin = srgb_to_linear(src.data[b]);
                let g_lin = srgb_to_linear(src.data[b + 1]);
                let b_lin = srgb_to_linear(src.data[b + 2]);
                let l = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
                if l > l_max {
                    l_max = l;
                }
            }
        }
        let explicit = ReinhardExtended::new(l_max)
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        assert_eq!(auto.planes[0].data, explicit.planes[0].data);
    }

    #[test]
    fn non_positive_white_falls_back_to_auto() {
        // Constructors should normalise NaN / negative / zero to the
        // auto mode rather than producing NaN-tainted output.
        let nan = ReinhardExtended::new(f32::NAN);
        let neg = ReinhardExtended::new(-1.0);
        let zero = ReinhardExtended::new(0.0);
        assert_eq!(nan.l_white, 0.0);
        assert_eq!(neg.l_white, 0.0);
        assert_eq!(zero.l_white, 0.0);

        // with_white(NaN) on an existing instance also reverts to auto.
        let reverted = ReinhardExtended::new(2.0).with_white(f32::NAN);
        assert_eq!(reverted.l_white, 0.0);
    }

    #[test]
    fn alpha_preserved_on_rgba() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[120u8, 80, 200, 33]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = ReinhardExtended::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 33);
        }
    }

    #[test]
    fn rejects_yuv() {
        let input = VideoFrame {
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
        let err = ReinhardExtended::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("ReinhardExtended"));
    }
}
