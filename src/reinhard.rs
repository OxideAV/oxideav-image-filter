//! Reinhard tone-mapping (global operator).
//!
//! Reinhard, Stark, Shirley, Ferwerda — "Photographic Tone Reproduction
//! for Digital Images" (SIGGRAPH 2002), §3 ("Initial Luminance Mapping")
//! and §3.1 ("Burn-out"). The simplest global form is the Reinhard
//! "tone-mapping operator", referred to as `L_d = L / (1 + L)` in §3
//! ("simple operator") with the extended white-point form
//! `L_d = L · (1 + L / L_white²) / (1 + L)` in §3.1.
//!
//! Algorithm (clean-room from the paper):
//!
//! 1. De-gamma the sRGB input to linear-light.
//! 2. Compute scene luminance `L_w` per pixel
//!    (`L = 0.27·R + 0.67·G + 0.06·B`, the paper's eq. (3) weights).
//! 3. Scale: `L_s = key · L_w / L_avg`. `L_avg` is the geometric mean
//!    of the scene luminance over the whole image (paper eq. (1)) —
//!    here we accumulate `log(L_w + ε)` over every pixel.
//! 4. Compress: `L_d = L_s · (1 + L_s / white²) / (1 + L_s)`. With
//!    `white = +∞` (set very large) the term collapses to the simple
//!    `L_d = L_s / (1 + L_s)` form.
//! 5. Re-colour the original linear RGB per the doc's §1 step 3
//!    saturation-controlled form `C_out = (C_in / L_w)^s · L_d`
//!    (`tone-mapping-operators.md` §1, step 3). The default `s = 1`
//!    collapses to `C_out = C_in · (L_d / L_w)`, the chroma-preserving
//!    re-modulation; `s < 1` desaturates toward grey, `s > 1`
//!    over-saturates. Re-encode the result through the sRGB OETF.
//!
//! Operates on `Rgb24` / `Rgba`. Gray8 falls back to the scalar form
//! (no chroma preservation; just apply the curve to the channel). YUV
//! returns `Unsupported` since the round-trip needs RGB linear-light
//! data — same constraint as [`Exposure`](crate::Exposure).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Global Reinhard tone-mapping operator.
#[derive(Clone, Copy, Debug)]
pub struct Reinhard {
    /// Reinhard "key" value `a` (paper eq. (2)). Controls the overall
    /// exposure: low-key (`0.045`) preserves shadows, mid-key (`0.18`,
    /// the default — "middle grey"), high-key (`0.72`) compresses
    /// highlights heavily.
    pub key: f32,
    /// "Burn-out" white point (paper eq. (4)). Set to a large value
    /// (default `1e6`) to disable the extended form and use the simple
    /// `L / (1 + L)` curve. Small values (`1.0..3.0`) preserve highlight
    /// detail by softly clipping above white.
    pub white: f32,
    /// Saturation exponent `s` of the §1 step 3 re-colouring form
    /// `C_out = (C_in / L_w)^s · L_d` (`tone-mapping-operators.md` §1).
    /// `s = 1` (the default) is the plain chroma-preserving
    /// re-modulation `C_in · (L_d / L_w)`; `0 ≤ s < 1` pulls colours
    /// toward neutral grey (the tone curve compresses chroma along with
    /// luminance), `s > 1` boosts it. Clamped non-negative.
    pub saturation: f32,
}

impl Default for Reinhard {
    fn default() -> Self {
        Self {
            key: 0.18,
            white: 1.0e6,
            saturation: 1.0,
        }
    }
}

impl Reinhard {
    pub fn new(key: f32) -> Self {
        Self {
            key: if key.is_finite() && key > 0.0 {
                key
            } else {
                0.18
            },
            white: 1.0e6,
            saturation: 1.0,
        }
    }

    pub fn with_white(mut self, white: f32) -> Self {
        if white.is_finite() && white > 0.0 {
            self.white = white;
        }
        self
    }

    /// Set the §1 step 3 saturation exponent `s`. Non-finite or negative
    /// values are ignored (the field keeps its current value).
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        if saturation.is_finite() && saturation >= 0.0 {
            self.saturation = saturation;
        }
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

impl ImageFilter for Reinhard {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Reinhard requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Reinhard requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let stride_out = w * bpp;
        let src = &input.planes[0];
        const EPS: f32 = 1.0e-4;

        // Pre-decode linear RGB and luminance.
        let n = w * h;
        let mut linr = vec![0f32; n];
        let mut ling = vec![0f32; n];
        let mut linb = vec![0f32; n];
        let mut lum = vec![0f32; n];
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
                // Paper eq. (3) luminance weights.
                lum[i] = 0.27 * r_lin + 0.67 * g_lin + 0.06 * b_lin;
            }
        }

        // Log-average luminance L_avg (paper eq. (1)).
        let mut log_sum = 0f64;
        for &l in &lum {
            log_sum += (l as f64 + EPS as f64).ln();
        }
        let log_avg = (log_sum / n as f64).exp() as f32;
        let scale = self.key / log_avg.max(EPS);
        let white2 = self.white * self.white;
        // §1 step 3 saturation exponent; `s == 1` keeps the exact
        // chroma-preserving re-modulation `C_in · (L_d / L_w)`.
        let sat = self.saturation;
        let unit_sat = sat == 1.0;

        let mut out = vec![0u8; stride_out * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let l_w = lum[i].max(EPS);
                let l_s = l_w * scale;
                // Extended Reinhard, paper eq. (4).
                let l_d = (l_s * (1.0 + l_s / white2)) / (1.0 + l_s);
                // §1 step 3: C_out = (C_in / L_w)^s · L_d. The s == 1
                // fast path is byte-identical to the legacy g = L_d/L_w
                // re-modulation.
                let recolour = |c: f32| -> f32 {
                    if unit_sat {
                        c * (l_d / l_w)
                    } else {
                        (c / l_w).max(0.0).powf(sat) * l_d
                    }
                };
                let r = recolour(linr[i]);
                let gv = recolour(ling[i]);
                let bv = recolour(linb[i]);
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
                        // Alpha pass-through.
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
        let out = Reinhard::default().apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn high_key_brightens_dark_scene() {
        // Dark scene: most values low.
        let input = rgb(4, 4, |_, _| [20, 20, 20]);
        let lo = Reinhard::new(0.05).apply(&input, p_rgb(4, 4)).unwrap();
        let hi = Reinhard::new(0.72).apply(&input, p_rgb(4, 4)).unwrap();
        assert!(
            hi.planes[0].data[0] > lo.planes[0].data[0],
            "high key must lift midtones above low key (lo={}, hi={})",
            lo.planes[0].data[0],
            hi.planes[0].data[0]
        );
    }

    #[test]
    fn small_white_clips_highlights_softer() {
        // Synthesise a frame with a bright stripe so the extended form
        // pulls back on highlights while leaving shadows alone.
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
        let wide = Reinhard::default().apply(&input, p_rgb(4, 4)).unwrap();
        let narrow = Reinhard::default()
            .with_white(1.5)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Bright cell — narrow-white should be brighter (closer to white)
        // because the curve allows L_s → L_white to saturate.
        let bright_wide = wide.planes[0].data[0];
        let bright_narrow = narrow.planes[0].data[0];
        assert!(
            bright_narrow >= bright_wide,
            "narrow white should saturate highlights faster (wide={bright_wide} narrow={bright_narrow})"
        );
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
        let out = Reinhard::default()
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
    fn default_saturation_is_unit_and_unchanged() {
        // s == 1 (the default) must equal an explicit with_saturation(1.0)
        // and the field defaults to 1.0.
        let input = rgb(8, 8, |x, y| {
            [
                ((x * 32).min(255)) as u8,
                ((y * 32).min(255)) as u8,
                (((x + y) * 16).min(255)) as u8,
            ]
        });
        assert_eq!(Reinhard::default().saturation, 1.0);
        let a = Reinhard::default().apply(&input, p_rgb(8, 8)).unwrap();
        let b = Reinhard::default()
            .with_saturation(1.0)
            .apply(&input, p_rgb(8, 8))
            .unwrap();
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn low_saturation_desaturates_toward_grey() {
        // A strongly-coloured pixel: red-dominant. s < 1 should pull the
        // R/G/B channels closer together (toward neutral grey) than the
        // s == 1 chroma-preserving baseline.
        let input = rgb(4, 4, |_, _| [200, 40, 40]);
        let full = Reinhard::default().apply(&input, p_rgb(4, 4)).unwrap();
        let desat = Reinhard::default()
            .with_saturation(0.3)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        let spread = |d: &[u8]| -> i32 {
            let (r, g, b) = (d[0] as i32, d[1] as i32, d[2] as i32);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            max - min
        };
        let full_spread = spread(&full.planes[0].data);
        let desat_spread = spread(&desat.planes[0].data);
        assert!(
            desat_spread < full_spread,
            "low saturation must reduce channel spread (full={full_spread} desat={desat_spread})"
        );
    }

    #[test]
    fn high_saturation_boosts_chroma() {
        // s > 1 should widen the channel spread relative to the baseline.
        let input = rgb(4, 4, |_, _| [180, 90, 90]);
        let full = Reinhard::default().apply(&input, p_rgb(4, 4)).unwrap();
        let oversat = Reinhard::default()
            .with_saturation(1.8)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        let spread = |d: &[u8]| -> i32 {
            let (r, g, b) = (d[0] as i32, d[1] as i32, d[2] as i32);
            r.max(g).max(b) - r.min(g).min(b)
        };
        assert!(
            spread(&oversat.planes[0].data) > spread(&full.planes[0].data),
            "high saturation must widen channel spread"
        );
    }

    #[test]
    fn with_saturation_rejects_invalid() {
        let base = Reinhard::default().with_saturation(0.5);
        assert_eq!(base.saturation, 0.5);
        // Negative + non-finite are ignored (field unchanged).
        assert_eq!(base.with_saturation(-1.0).saturation, 0.5);
        assert_eq!(base.with_saturation(f32::NAN).saturation, 0.5);
        assert_eq!(base.with_saturation(f32::INFINITY).saturation, 0.5);
        // Zero is valid (fully desaturated).
        assert_eq!(base.with_saturation(0.0).saturation, 0.0);
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
        let err = Reinhard::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Reinhard"));
    }
}
