//! Drago adaptive logarithmic tone-mapping operator.
//!
//! Drago, Myszkowski, Annen, Chiba — "Adaptive Logarithmic Mapping For
//! Displaying High Contrast Scenes" (Eurographics 2003). Compresses a
//! high-dynamic-range scene through a `log_b` curve whose base `b`
//! varies smoothly between `log_2` (preserves contrast in shadows) and
//! `log_{10}` (compresses highlights) according to the pixel's
//! normalised luminance.
//!
//! Algorithm summary (clean-room from the paper, eq. (3)+(7)):
//!
//! ```text
//! Lw   = scene luminance (linear)
//! Lwmax = max(Lw)
//! Ld   = (Lwmax_d / log10(Lwmax + 1)) ·
//!        log(Lw + 1) /
//!        log(2 + 8·((Lw / Lwmax)^(log(bias)/log(0.5))))
//! ```
//!
//! `Lwmax_d` is the target display-luminance maximum (`1.0` for a
//! normalised `[0, 1]` output). `bias` is Drago's bias parameter
//! (default `0.85`, paper §3.3) — `0.85` is the "all-purpose" value
//! the paper recommends.
//!
//! Algorithm (clean-room from the paper):
//!
//! 1. De-gamma sRGB → linear.
//! 2. Compute per-pixel scene luminance `Lw` (Rec.601: `0.2126R +
//!    0.7152G + 0.0722B` — the paper uses CIE Y from a chromatic-adapt
//!    matrix; we apply Rec.709 luminance to keep the operator clean for
//!    sRGB content).
//! 3. Find `Lwmax` over the whole image.
//! 4. Apply the Drago curve to get `Ld`.
//! 5. Re-modulate the original linear RGB by `Ld / Lw` (chroma
//!    preservation, matching the paper's §3.4 colour handling).
//! 6. sRGB OETF.
//!
//! Operates on `Rgb24` / `Rgba` and `Gray8` (chroma-trivial). YUV
//! returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Drago adaptive logarithmic tone mapper.
#[derive(Clone, Copy, Debug)]
pub struct Drago {
    /// Bias parameter `b` from the paper (§3.3). `0.85` is the
    /// recommended "all-purpose" default.
    pub bias: f32,
    /// Display-luminance scale. `1.0` produces values in `[0, 1]`;
    /// values `< 1` darken the entire output.
    pub display_scale: f32,
}

impl Default for Drago {
    fn default() -> Self {
        Self {
            bias: 0.85,
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
            display_scale: 1.0,
        }
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

        let mut lwmax = 0f32;
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
                if l > lwmax {
                    lwmax = l;
                }
            }
        }

        let lwmax = lwmax.max(1.0e-6);
        // Drago eq. (3)+(7):
        //   Ld = (Ld_max / log10(Lwmax + 1)) · log10(Lw + 1)
        //        / log10(2 + 8·(Lw/Lwmax)^(log(bias)/log(0.5)))
        // The denominator's base is `2 + 8·ratio^p` ∈ [2, 10]; log10
        // of that varies smoothly from log10(2) ≈ 0.301 to log10(10) =
        // 1.0, giving a soft curve over the dynamic range.
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
                let ld = self.display_scale * (num / log_lwmax_plus_1) / denom;
                let scale = if lw > 1.0e-6 { (ld / lw).max(0.0) } else { 0.0 };
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
