//! Independent shadow lift and highlight recovery — the Lightroom
//! "Shadows" + "Highlights" sliders in one filter.
//!
//! Two LUT-style adjustments folded into a single pass:
//!
//! - `shadow_amount > 0` brightens dark pixels (positive `shadows` slider).
//! - `shadow_amount < 0` deepens shadows.
//! - `highlight_amount > 0` recovers blown highlights (Lr's negative-direction
//!   slider — we use positive = recover for ergonomics).
//! - `highlight_amount < 0` boosts highlights.
//!
//! Each adjustment is gated by a soft tonal mask:
//!
//! - Shadow mask: weight = `1 - 2·x` for `x < 0.5`, else `0`. (Peaks at
//!   black, fades to zero at midtone.)
//! - Highlight mask: weight = `2·x - 1` for `x > 0.5`, else `0`. (Peaks at
//!   white, fades to zero at midtone.)
//!
//! The two corrections add (so they can both be applied per pixel) and
//! the result is clamped to `[0, 255]`. Because the masks are derived
//! from the luma value, midtones pass through untouched — only the
//! tonal extremes move. Implementation is a per-channel 256-entry LUT
//! so the cost is `O(W·H)`.
//!
//! Operates on `Gray8` / `Rgb24` / `Rgba`. YUV inputs touch luma only.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Independent shadow + highlight tonal-range recovery.
#[derive(Clone, Copy, Debug)]
pub struct ShadowHighlight {
    /// Lift / dip applied to dark pixels, `[-1, 1]`.
    pub shadow_amount: f32,
    /// Lift / dip applied to bright pixels, `[-1, 1]`. Positive recovers
    /// blown highlights, negative boosts them.
    pub highlight_amount: f32,
}

impl Default for ShadowHighlight {
    fn default() -> Self {
        Self {
            shadow_amount: 0.0,
            highlight_amount: 0.0,
        }
    }
}

impl ShadowHighlight {
    pub fn new(shadow: f32, highlight: f32) -> Self {
        let f = |v: f32| {
            if v.is_finite() {
                v.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };
        Self {
            shadow_amount: f(shadow),
            highlight_amount: f(highlight),
        }
    }

    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32 / 255.0;
            let shadow_w = if x < 0.5 { 1.0 - 2.0 * x } else { 0.0 };
            let highlight_w = if x > 0.5 { 2.0 * x - 1.0 } else { 0.0 };
            // Shadow lift: shadow_amount > 0 brightens darks toward 0.5.
            let shadow_delta = self.shadow_amount * shadow_w * 0.5;
            // Highlight recovery: highlight_amount > 0 darkens highs toward 0.5.
            let highlight_delta = -self.highlight_amount * highlight_w * 0.5;
            let y = (x + shadow_delta + highlight_delta).clamp(0.0, 1.0);
            *slot = (y * 255.0).round() as u8;
        }
        lut
    }
}

impl ImageFilter for ShadowHighlight {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let lut = self.build_lut();
        match params.format {
            PixelFormat::Gray8 => {
                if input.planes.is_empty() {
                    return Err(Error::invalid(
                        "oxideav-image-filter: ShadowHighlight requires a non-empty input plane",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let src = &input.planes[0];
                let stride = w;
                let mut out = vec![0u8; stride * h];
                for y in 0..h {
                    let s = &src.data[y * src.stride..y * src.stride + stride];
                    let d = &mut out[y * stride..(y + 1) * stride];
                    for (di, si) in d.iter_mut().zip(s.iter()) {
                        *di = lut[*si as usize];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data: out }],
                })
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if matches!(params.format, PixelFormat::Rgba) {
                    4
                } else {
                    3
                };
                if input.planes.is_empty() {
                    return Err(Error::invalid(
                        "oxideav-image-filter: ShadowHighlight requires a non-empty input plane",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let stride = w * bpp;
                let src = &input.planes[0];
                let mut out = vec![0u8; stride * h];
                for y in 0..h {
                    let s = &src.data[y * src.stride..y * src.stride + stride];
                    let d = &mut out[y * stride..(y + 1) * stride];
                    for x in 0..w {
                        let base = x * bpp;
                        d[base] = lut[s[base] as usize];
                        d[base + 1] = lut[s[base + 1] as usize];
                        d[base + 2] = lut[s[base + 2] as usize];
                        if bpp == 4 {
                            d[base + 3] = s[base + 3];
                        }
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data: out }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                // Apply only to luma; chroma pass-through.
                if input.planes.len() < 3 {
                    return Err(Error::invalid(
                        "oxideav-image-filter: ShadowHighlight YUV input requires 3 planes",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let src_y = &input.planes[0];
                let stride = w;
                let mut data_y = vec![0u8; stride * h];
                for yy in 0..h {
                    let s = &src_y.data[yy * src_y.stride..yy * src_y.stride + stride];
                    let d = &mut data_y[yy * stride..(yy + 1) * stride];
                    for (di, si) in d.iter_mut().zip(s.iter()) {
                        *di = lut[*si as usize];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![
                        VideoPlane {
                            stride,
                            data: data_y,
                        },
                        input.planes[1].clone(),
                        input.planes[2].clone(),
                    ],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: ShadowHighlight does not yet handle {other:?}"
            ))),
        }
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
    fn zero_amounts_is_identity() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = ShadowHighlight::default()
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn shadow_lift_brightens_dark_pixels() {
        let input = rgb(2, 2, |_, _| [10, 10, 10]);
        let out = ShadowHighlight::new(1.0, 0.0)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let v = out.planes[0].data[0];
        assert!(v > 40, "shadow lift should brighten near-black: got {v}");
    }

    #[test]
    fn highlight_recovery_darkens_bright_pixels() {
        let input = rgb(2, 2, |_, _| [240, 240, 240]);
        let out = ShadowHighlight::new(0.0, 1.0)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let v = out.planes[0].data[0];
        assert!(v < 230, "highlight recovery should darken whites: got {v}");
    }

    #[test]
    fn midtones_unaffected_at_extremes() {
        // The masks are zero at 0.5, so a midtone should pass through.
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = ShadowHighlight::new(1.0, 1.0)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        for v in out.planes[0].data.iter() {
            assert!((*v as i16 - 128).abs() <= 2, "midtone should not move: {v}");
        }
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[10u8, 10, 10, 99]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = ShadowHighlight::new(0.8, 0.0)
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
            assert_eq!(chunk[3], 99);
        }
    }

    #[test]
    fn yuv_chroma_passes_through() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![20; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![123; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![210; 4],
                },
            ],
        };
        let out = ShadowHighlight::new(1.0, 0.0)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Chroma planes preserved verbatim.
        assert_eq!(out.planes[1].data, vec![123; 4]);
        assert_eq!(out.planes[2].data, vec![210; 4]);
        // Luma plane brightened.
        assert!(out.planes[0].data[0] > 20);
    }
}
