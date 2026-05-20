//! Cyanotype — vintage blue-print colour remap.
//!
//! The cyanotype process (Sir John Herschel, 1842 — the original
//! "blueprint" photographic process) renders luminance as a graded mix
//! between two characteristic colours: dark "Prussian blue"
//! (≈ `#0F2A6F` / sRGB `(15, 42, 111)`) at black, and pale paper-white
//! (`(217, 235, 248)`) at white. The reproduction curve is documented
//! in the Library of Congress preservation note "Cyanotype
//! Photographs" (1999) and in Mike Ware, "Cyanotype: The History,
//! Science and Art of Photographic Printing in Prussian Blue" (1999),
//! §4.4 "Sensitometry of cyanotype".
//!
//! Our operator is the obvious clean-room realisation: reduce the
//! input to luminance, then linearly interpolate between the two
//! endpoint colours. Strength dials the blend with the original RGB
//! (`0.0` = original, `1.0` = full cyanotype look).
//!
//! Algorithm summary:
//!
//! 1. Compute scene luminance `L = (R·0.2126 + G·0.7152 + B·0.0722) /
//!    255` (Rec. 709 weights — same as the rest of this crate's tonal
//!    ops).
//! 2. Endpoint colours `C_lo = (15, 42, 111)`, `C_hi = (217, 235,
//!    248)` (configurable).
//! 3. `C_cyan = C_lo + L · (C_hi - C_lo)`.
//! 4. `out = (1 - strength) · src + strength · C_cyan`.
//!
//! Operates on `Gray8` (upgraded to `Rgb24` on output), `Rgb24`, and
//! `Rgba` (alpha pass-through). YUV returns `Unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Cyanotype vintage blue-print operator.
#[derive(Clone, Copy, Debug)]
pub struct Cyanotype {
    /// Blend strength `[0, 1]`. `0` is the original image; `1` is the
    /// full cyanotype look.
    pub strength: f32,
    /// Shadow / "Prussian blue" endpoint colour (sRGB 8-bit).
    pub shadow: [u8; 3],
    /// Highlight / paper colour (sRGB 8-bit).
    pub highlight: [u8; 3],
}

impl Default for Cyanotype {
    fn default() -> Self {
        Self {
            strength: 1.0,
            shadow: [15, 42, 111],
            highlight: [217, 235, 248],
        }
    }
}

impl Cyanotype {
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    pub fn with_endpoints(mut self, shadow: [u8; 3], highlight: [u8; 3]) -> Self {
        self.shadow = shadow;
        self.highlight = highlight;
        self
    }
}

#[inline]
fn luma(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

impl ImageFilter for Cyanotype {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Cyanotype requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];
        let s = self.strength.clamp(0.0, 1.0);
        let lo = [
            self.shadow[0] as f32,
            self.shadow[1] as f32,
            self.shadow[2] as f32,
        ];
        let hi = [
            self.highlight[0] as f32,
            self.highlight[1] as f32,
            self.highlight[2] as f32,
        ];
        let dv = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];

        match params.format {
            PixelFormat::Gray8 => {
                // Output is Rgb24.
                let mut out = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let v = src.data[y * src.stride + x];
                        let l = v as f32 / 255.0;
                        let cy = [lo[0] + l * dv[0], lo[1] + l * dv[1], lo[2] + l * dv[2]];
                        // Treat gray src as RGB (v, v, v) for the blend.
                        let r = (1.0 - s) * v as f32 + s * cy[0];
                        let g = (1.0 - s) * v as f32 + s * cy[1];
                        let bb = (1.0 - s) * v as f32 + s * cy[2];
                        let i = (y * w + x) * 3;
                        out[i] = r.round().clamp(0.0, 255.0) as u8;
                        out[i + 1] = g.round().clamp(0.0, 255.0) as u8;
                        out[i + 2] = bb.round().clamp(0.0, 255.0) as u8;
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: w * 3,
                        data: out,
                    }],
                })
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if matches!(params.format, PixelFormat::Rgb24) {
                    3
                } else {
                    4
                };
                let stride = w * bpp;
                let mut out = vec![0u8; stride * h];
                for y in 0..h {
                    let s_row = &src.data[y * src.stride..y * src.stride + stride];
                    let d_row = &mut out[y * stride..(y + 1) * stride];
                    for x in 0..w {
                        let b = x * bpp;
                        let r = s_row[b];
                        let g = s_row[b + 1];
                        let bb = s_row[b + 2];
                        let l = luma(r, g, bb);
                        let cy = [lo[0] + l * dv[0], lo[1] + l * dv[1], lo[2] + l * dv[2]];
                        let rr = (1.0 - s) * r as f32 + s * cy[0];
                        let gg = (1.0 - s) * g as f32 + s * cy[1];
                        let bv = (1.0 - s) * bb as f32 + s * cy[2];
                        d_row[b] = rr.round().clamp(0.0, 255.0) as u8;
                        d_row[b + 1] = gg.round().clamp(0.0, 255.0) as u8;
                        d_row[b + 2] = bv.round().clamp(0.0, 255.0) as u8;
                        if bpp == 4 {
                            d_row[b + 3] = s_row[b + 3];
                        }
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data: out }],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Cyanotype requires Gray8/Rgb24/Rgba, got {other:?}"
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
    fn zero_strength_is_identity() {
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Cyanotype::new(0.0).apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn black_maps_to_shadow_endpoint() {
        let input = rgb(2, 2, |_, _| [0, 0, 0]);
        let out = Cyanotype::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let pix = &out.planes[0].data[..3];
        assert_eq!(pix, &[15, 42, 111]);
    }

    #[test]
    fn white_maps_to_highlight_endpoint() {
        let input = rgb(2, 2, |_, _| [255, 255, 255]);
        let out = Cyanotype::new(1.0).apply(&input, p_rgb(2, 2)).unwrap();
        let pix = &out.planes[0].data[..3];
        assert_eq!(pix, &[217, 235, 248]);
    }

    #[test]
    fn custom_endpoints_respected() {
        let input = rgb(2, 2, |_, _| [0, 0, 0]);
        let out = Cyanotype::new(1.0)
            .with_endpoints([200, 0, 50], [255, 255, 255])
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let pix = &out.planes[0].data[..3];
        assert_eq!(pix, &[200, 0, 50]);
    }

    #[test]
    fn alpha_passes_through() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 100, 100, 77]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Cyanotype::new(1.0)
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
            assert_eq!(chunk[3], 77);
        }
    }

    #[test]
    fn gray_upgrades_to_rgb() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![128u8; 16],
            }],
        };
        let out = Cyanotype::new(1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // 4*4 RGB output.
        assert_eq!(out.planes[0].data.len(), 4 * 4 * 3);
        assert_eq!(out.planes[0].stride, 12);
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
        let err = Cyanotype::new(1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Cyanotype"));
    }
}
