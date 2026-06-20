//! Drago adaptive logarithmic tone-mapping operator.
//!
//! Drago, Myszkowski, Annen, Chiba — "Adaptive Logarithmic Mapping For
//! Displaying High Contrast Scenes" (Eurographics 2003). Compresses a
//! high-dynamic-range scene through a `log_b` curve whose base `b`
//! varies smoothly between `log_2` (preserves contrast in shadows) and
//! `log_{10}` (compresses highlights) according to the pixel's
//! normalised luminance.
//!
//! Algorithm summary (clean-room from
//! `docs/image/filter/tone-mapping-operators.md` §4.2):
//!
//! ```text
//! Lw    = scene luminance (linear)
//! Lwmax = max(Lw)
//! Ld    = (Ld_max · 0.01 / log10(Lwmax + 1)) ·
//!         log(Lw + 1) /
//!         log(2 + 8·((Lw / Lwmax)^(log(bias)/log(0.5))))
//! ```
//!
//! The leading `Ld_max · 0.01 / log10(Lwmax + 1)` factor (§4.2)
//! normalises the output into `[0, 1]`: `Ld_max` is the **target
//! maximum display luminance in cd/m²** (the §4 params table default is
//! `100`, so the `Ld_max · 0.01` term is `1.0` at the default and the
//! brightest scene pixel maps to display white). `bias` is Drago's bias
//! parameter `b` (default `0.85`, §4 params table) — the "useful range"
//! the doc gives is `0.7–0.9`.
//!
//! Algorithm (clean-room from the doc §4):
//!
//! 1. De-gamma sRGB → linear.
//! 2. Compute per-pixel scene luminance `Lw` with Rec. 709 weights
//!    `0.2126R + 0.7152G + 0.0722B` (§1 step 1). The §4.1 "world
//!    adaptation luminance" pre-scaling is exposed separately (below).
//! 3. Find `Lwmax` over the whole image.
//! 4. Apply the Drago curve to get `Ld`, normalised by the §4.2
//!    `Ld_max · 0.01 / log10(Lwmax + 1)` factor.
//! 5. Re-modulate the original linear RGB by `Ld / Lw` (chroma
//!    preservation, matching §1 step 3 colour handling).
//! 6. sRGB OETF (§5.2).
//!
//! **Exposure-independence (§4.1).** The doc notes `Lw` and `Lwmax`
//! "are typically scaled by the world adaptation luminance / log-average
//! so the curve is exposure-independent." [`Drago::with_key`] enables
//! that pre-step: luminance is divided by the log-average key (§1.1)
//! before the curve, so two exposures of the same scene map identically.
//! Without it (the default) the operator runs on raw linear luminance.
//!
//! Operates on `Rgb24` / `Rgba` and `Gray8` (chroma-trivial). YUV
//! returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Drago adaptive logarithmic tone mapper.
#[derive(Clone, Copy, Debug)]
pub struct Drago {
    /// Bias parameter `b` from the doc §4 params table. `0.85` is the
    /// recommended default; the useful range is `0.7–0.9`.
    pub bias: f32,
    /// Target maximum display luminance in cd/m² (doc §4.2, `Ld_max`).
    /// The §4.2 leading factor is `Ld_max · 0.01 / log10(Lwmax + 1)`;
    /// the §4 params-table default `100` makes `Ld_max · 0.01 == 1.0`,
    /// so the brightest scene pixel maps to display white. Lower values
    /// dim the whole output; higher values brighten it.
    pub ld_max: f32,
    /// Optional log-average key for the §4.1 exposure-independent
    /// pre-scaling. `> 0` divides scene luminance by this value before
    /// the curve (the §1.1 log-average key when set via
    /// [`Drago::with_key`]); `<= 0` (the default) skips the pre-step and
    /// runs the curve on raw linear luminance.
    pub key: f32,
    /// Auxiliary display-luminance multiplier applied after the §4.2
    /// normalisation. `1.0` is the identity; values `< 1` darken,
    /// values `> 1` brighten. Kept separate from [`Self::ld_max`] so a
    /// caller can dim output without changing the cd/m² target.
    pub display_scale: f32,
}

impl Default for Drago {
    fn default() -> Self {
        Self {
            bias: 0.85,
            ld_max: 100.0,
            key: 0.0,
            display_scale: 1.0,
        }
    }
}

impl Drago {
    pub fn new(bias: f32) -> Self {
        Self {
            bias: if bias.is_finite() && (0.5..=1.0).contains(&bias) {
                bias
            } else {
                0.85
            },
            ..Self::default()
        }
    }

    /// Set the §4.2 target maximum display luminance `Ld_max` (cd/m²).
    /// Non-finite or non-positive values are ignored (the default `100`
    /// is retained). `100` reproduces the doc's `[0, 1]` normalisation.
    pub fn with_ld_max(mut self, ld_max: f32) -> Self {
        if ld_max.is_finite() && ld_max > 0.0 {
            self.ld_max = ld_max;
        }
        self
    }

    /// Enable the §4.1 exposure-independent pre-scaling with an explicit
    /// key. Scene luminance is divided by `key` before the curve; pass
    /// the §1.1 log-average luminance to make the mapping invariant to
    /// scene exposure. Non-finite or non-positive values disable the
    /// pre-step.
    pub fn with_key(mut self, key: f32) -> Self {
        self.key = if key.is_finite() && key > 0.0 {
            key
        } else {
            0.0
        };
        self
    }

    /// Enable the §4.1 pre-scaling using the image's **own** log-average
    /// luminance (§1.1) as the key, computed at apply time. This is the
    /// convenient "auto" form of [`Self::with_key`]: the caller need not
    /// pre-compute the key. A sentinel `key = -1.0` requests auto mode.
    pub fn with_auto_key(mut self) -> Self {
        self.key = -1.0;
        self
    }

    pub fn with_display_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.display_scale = scale;
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

impl ImageFilter for Drago {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Drago requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Drago requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let stride_out = w * bpp;
        let src = &input.planes[0];
        let n = w * h;
        let mut linr = vec![0f32; n];
        let mut ling = vec![0f32; n];
        let mut linb = vec![0f32; n];
        let mut lum = vec![0f32; n];

        // §1.1 log-average accumulator (only consumed when auto-key is
        // requested, but cheap enough to always gather).
        let mut log_sum = 0f64;
        const LOG_AVG_DELTA: f32 = 1.0e-6;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let b = y * src.stride + x * bpp;
                let (r, g, bb) = match bpp {
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
                linr[i] = r;
                ling[i] = g;
                linb[i] = bb;
                // Rec. 709 luminance — see module docs.
                let l = 0.2126 * r + 0.7152 * g + 0.0722 * bb;
                lum[i] = l;
                log_sum += ((LOG_AVG_DELTA + l.max(0.0)) as f64).ln();
            }
        }

        // §4.1 exposure-independent pre-scaling. `key == 0` skips it;
        // `key > 0` divides by the supplied key; `key < 0` (auto) uses
        // the image's own §1.1 log-average luminance as the key so the
        // mapping is invariant to scene exposure.
        let key = if self.key > 0.0 {
            self.key
        } else if self.key < 0.0 {
            ((log_sum / (n as f64)).exp() as f32).max(LOG_AVG_DELTA)
        } else {
            1.0
        };
        // Keep the original (un-prescaled) luminance for chroma
        // re-modulation; the curve consumes the pre-scaled copy.
        let lum_orig = lum.clone();
        if key != 1.0 {
            for l in lum.iter_mut() {
                *l /= key;
            }
        }

        let mut lwmax = 0f32;
        for &l in &lum {
            if l > lwmax {
                lwmax = l;
            }
        }

        let lwmax = lwmax.max(1.0e-6);
        // Doc §4.2:
        //   Ld = (Ld_max · 0.01 / log10(Lwmax + 1)) · log10(Lw + 1)
        //        / log10(2 + 8·(Lw/Lwmax)^(log(bias)/log(0.5)))
        // The denominator's base is `2 + 8·ratio^p` ∈ [2, 10]; log10
        // of that varies smoothly from log10(2) ≈ 0.301 to log10(10) =
        // 1.0, giving a soft curve over the dynamic range. The leading
        // factor's `Ld_max · 0.01` term is `1.0` at the default
        // `Ld_max = 100`, mapping the brightest pixel to display white.
        let ld_max_norm = self.ld_max * 0.01;
        let log_lwmax_plus_1 = (lwmax + 1.0).log10().max(1.0e-6);
        let log_bias_exp = self.bias.ln() / 0.5f32.ln();

        let mut out = vec![0u8; stride_out * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let lw = lum[i].max(0.0);
                let ratio = (lw / lwmax).clamp(0.0, 1.0);
                let denom_arg = 2.0 + 8.0 * ratio.powf(log_bias_exp);
                let denom = denom_arg.log10().max(1.0e-6);
                let num = (lw + 1.0).log10();
                let ld = self.display_scale * ld_max_norm * (num / log_lwmax_plus_1) / denom;
                // Chroma preservation (§1 step 3): `C_out = C_in · Ld /
                // Lw` against the **original** scene luminance so the
                // re-coloured pixel keeps its chromaticity. The §4.1 key
                // only reshapes the tone curve (via `lw`), it does not
                // multiply the output.
                let lw_orig = lum_orig[i].max(0.0);
                let scale = if lw_orig > 1.0e-6 {
                    (ld / lw_orig).max(0.0)
                } else {
                    0.0
                };
                let db = y * stride_out + x * bpp;
                match bpp {
                    1 => {
                        out[db] = linear_to_srgb(ling[i] * scale);
                    }
                    3 => {
                        out[db] = linear_to_srgb(linr[i] * scale);
                        out[db + 1] = linear_to_srgb(ling[i] * scale);
                        out[db + 2] = linear_to_srgb(linb[i] * scale);
                    }
                    4 => {
                        out[db] = linear_to_srgb(linr[i] * scale);
                        out[db + 1] = linear_to_srgb(ling[i] * scale);
                        out[db + 2] = linear_to_srgb(linb[i] * scale);
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
        let input = rgb(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Drago::default().apply(&input, p_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn uniform_scene_maps_to_white() {
        // Drago is adaptive — every Lw == Lwmax so the operator scales
        // the brightest pixel of the scene to the display max. This
        // mirrors the paper's §3.4 normalisation.
        let input = rgb(4, 4, |_, _| [10, 10, 10]);
        let out = Drago::default().apply(&input, p_rgb(4, 4)).unwrap();
        for &v in &out.planes[0].data {
            assert!(
                v >= 240,
                "uniform scene should normalise to display max, got {v}"
            );
        }
    }

    #[test]
    fn display_scale_darkens_output() {
        let input = rgb(4, 4, |_, _| [100, 100, 100]);
        let full = Drago::default().apply(&input, p_rgb(4, 4)).unwrap();
        let half = Drago::default()
            .with_display_scale(0.5)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert!(
            half.planes[0].data[0] < full.planes[0].data[0],
            "display_scale < 1 must darken"
        );
    }

    #[test]
    fn bright_pixel_mapped_below_max() {
        // Scene with one very bright pixel; the operator should clip
        // brightness so the max value is at or below 255.
        let input = rgb(4, 4, |x, y| {
            if x == 0 && y == 0 {
                [255, 255, 255]
            } else {
                [40, 40, 40]
            }
        });
        let out = Drago::default().apply(&input, p_rgb(4, 4)).unwrap();
        // Output length matches expected RGB byte count.
        assert_eq!(out.planes[0].data.len(), 4 * 4 * 3);
        // Bright pixel still at least as bright as a dark pixel.
        let bright = out.planes[0].data[0];
        let dark = out.planes[0].data[(3 * 4 + 3) * 3];
        assert!(
            bright >= dark,
            "bright pixel should be at least as bright (b={bright} d={dark})"
        );
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[200u8, 100, 50, 250]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Drago::default()
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
            assert_eq!(chunk[3], 250);
        }
    }

    #[test]
    fn ld_max_default_matches_unit_normalisation() {
        // Ld_max = 100 → Ld_max·0.01 = 1.0, the doc §4.2 unit
        // normalisation. Explicitly constructing with 100 must equal the
        // default (which also normalises to [0,1]).
        let input = rgb(6, 6, |x, y| [(x * 40) as u8, (y * 40) as u8, 90]);
        let def = Drago::default().apply(&input, p_rgb(6, 6)).unwrap();
        let explicit = Drago::default()
            .with_ld_max(100.0)
            .apply(&input, p_rgb(6, 6))
            .unwrap();
        assert_eq!(def.planes[0].data, explicit.planes[0].data);
    }

    #[test]
    fn ld_max_below_100_dims_output() {
        // Ld_max·0.01 < 1 scales the whole curve down → darker output.
        let input = rgb(4, 4, |_, _| [120, 120, 120]);
        let full = Drago::default().apply(&input, p_rgb(4, 4)).unwrap();
        let dim = Drago::default()
            .with_ld_max(50.0)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert!(
            dim.planes[0].data[0] < full.planes[0].data[0],
            "Ld_max = 50 (< 100) must dim the output (dim={} full={})",
            dim.planes[0].data[0],
            full.planes[0].data[0]
        );
    }

    #[test]
    fn ld_max_invalid_falls_back_to_default() {
        let d = Drago::default().with_ld_max(-5.0).with_ld_max(f32::NAN);
        assert_eq!(d.ld_max, 100.0);
    }

    #[test]
    fn explicit_key_reshapes_curve() {
        // Supplying a key < 1 pre-scales luminance up (Lw/key), pushing
        // mid-tones further up the compressive part of the curve. The
        // output must differ from the no-key default.
        let input = rgb(6, 6, |x, _| [(x * 30 + 20) as u8; 3]);
        let plain = Drago::default().apply(&input, p_rgb(6, 6)).unwrap();
        let keyed = Drago::default()
            .with_key(0.1)
            .apply(&input, p_rgb(6, 6))
            .unwrap();
        assert_ne!(
            plain.planes[0].data, keyed.planes[0].data,
            "an explicit §4.1 key must reshape the mapping"
        );
    }

    #[test]
    fn auto_key_is_exposure_independent() {
        // §4.1 promise: with the auto log-average key the mapping is
        // invariant to a global linear exposure change of the scene.
        // Build a scene, then a "darker exposure" of the same scene by
        // halving every linear-light value (re-encoded through sRGB).
        let base =
            |x: u32, y: u32| -> [f32; 3] { [0.02 + 0.10 * x as f32, 0.05 + 0.08 * y as f32, 0.30] };
        let to_srgb = |lin: f32| -> u8 {
            let lin = lin.clamp(0.0, 1.0);
            let v = if lin <= 0.003_130_8 {
                lin * 12.92
            } else {
                1.055 * lin.powf(1.0 / 2.4) - 0.055
            };
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        };
        let bright = rgb(5, 5, |x, y| {
            let l = base(x, y);
            [to_srgb(l[0]), to_srgb(l[1]), to_srgb(l[2])]
        });
        let dark = rgb(5, 5, |x, y| {
            let l = base(x, y);
            [
                to_srgb(l[0] * 0.5),
                to_srgb(l[1] * 0.5),
                to_srgb(l[2] * 0.5),
            ]
        });
        let out_b = Drago::default()
            .with_auto_key()
            .apply(&bright, p_rgb(5, 5))
            .unwrap();
        let out_d = Drago::default()
            .with_auto_key()
            .apply(&dark, p_rgb(5, 5))
            .unwrap();
        // The two tone-mapped outputs should be close (within sRGB
        // round-trip + log-average quantisation noise), demonstrating
        // exposure-independence. Compare mean absolute difference.
        let total: u32 = out_b.planes[0]
            .data
            .iter()
            .zip(&out_d.planes[0].data)
            .map(|(&a, &b)| a.abs_diff(b) as u32)
            .sum();
        let mad = total as f32 / out_b.planes[0].data.len() as f32;
        assert!(
            mad < 6.0,
            "auto-key Drago should be ~exposure-independent, MAD={mad}"
        );
    }

    #[test]
    fn key_invalid_disables_prestep() {
        let d = Drago::default().with_key(0.5).with_key(f32::INFINITY);
        assert_eq!(d.key, 0.0, "non-finite key must disable the §4.1 pre-step");
        let d2 = Drago::default().with_key(-2.0);
        assert_eq!(d2.key, 0.0, "non-positive key must disable the pre-step");
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
        let err = Drago::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Drago"));
    }
}
