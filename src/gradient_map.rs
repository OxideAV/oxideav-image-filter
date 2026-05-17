//! Gradient map — recolour an image by mapping per-pixel luminance to a
//! position in a user-supplied colour gradient.
//!
//! Clean-room formulation:
//!
//! ```text
//! For each pixel:
//!     y = Rec. 601 luma in [0, 1]
//!     out_rgb = piecewise-linear interpolation of the gradient stops at y
//! ```
//!
//! The gradient is supplied as a list of `(position, [R, G, B])` stops,
//! `position ∈ [0, 1]`. Stops are sorted on construction; out-of-range
//! positions are clamped. At least two stops are required.
//!
//! Distinct from [`Heatmap`](crate::Heatmap), which only ships a fixed
//! catalogue of built-in ramps; `GradientMap` lets the caller specify an
//! arbitrary palette gradient (good for duotone / tritone / poster-style
//! recolouring).
//!
//! # Pixel formats
//!
//! Input: `Gray8` / `Rgb24` / `Rgba`. Output preserves the input format
//! (alpha pass-through on RGBA). YUV / planar / higher-bit formats
//! return [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// A single gradient stop: position in `[0, 1]` + RGB colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    /// Position along the gradient, clamped to `[0, 1]`.
    pub position: f32,
    /// RGB colour at this stop.
    pub color: [u8; 3],
}

/// Recolour by luminance ⇒ gradient sample.
#[derive(Clone, Debug)]
pub struct GradientMap {
    /// Sorted-by-position stops; always ≥ 2 entries.
    stops: Vec<GradientStop>,
}

impl GradientMap {
    /// Build from raw `(position, [R, G, B])` tuples. Returns an error
    /// if fewer than two stops are supplied or any position is not
    /// finite. Positions are clamped to `[0, 1]` and stops are sorted
    /// by ascending position. Duplicate positions are allowed — they
    /// produce a hard step at that position.
    pub fn new(stops: Vec<GradientStop>) -> Result<Self, Error> {
        if stops.len() < 2 {
            return Err(Error::invalid(
                "oxideav-image-filter: GradientMap requires at least two stops",
            ));
        }
        for s in &stops {
            if !s.position.is_finite() {
                return Err(Error::invalid(
                    "oxideav-image-filter: GradientMap stop position must be finite",
                ));
            }
        }
        let mut s: Vec<GradientStop> = stops
            .into_iter()
            .map(|s| GradientStop {
                position: s.position.clamp(0.0, 1.0),
                color: s.color,
            })
            .collect();
        s.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        Ok(Self { stops: s })
    }

    /// Convenience constructor: build a duotone gradient (shadow ⇒
    /// highlight) with stops at positions 0 and 1.
    pub fn duotone(shadow: [u8; 3], highlight: [u8; 3]) -> Self {
        Self {
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: shadow,
                },
                GradientStop {
                    position: 1.0,
                    color: highlight,
                },
            ],
        }
    }

    /// Convenience constructor: build a tritone gradient (shadow ⇒
    /// midtone ⇒ highlight) with stops at 0 / 0.5 / 1.
    pub fn tritone(shadow: [u8; 3], midtone: [u8; 3], highlight: [u8; 3]) -> Self {
        Self {
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: shadow,
                },
                GradientStop {
                    position: 0.5,
                    color: midtone,
                },
                GradientStop {
                    position: 1.0,
                    color: highlight,
                },
            ],
        }
    }

    /// Sample the gradient at normalised `t ∈ [0, 1]`. Used both
    /// internally and as a low-cost LUT preview.
    pub fn sample(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let stops = &self.stops;
        // Find the segment.
        if t <= stops[0].position {
            return stops[0].color;
        }
        if t >= stops[stops.len() - 1].position {
            return stops[stops.len() - 1].color;
        }
        // Binary-friendly linear walk; gradients are small (~few stops).
        for i in 0..stops.len() - 1 {
            let a = stops[i];
            let b = stops[i + 1];
            if t >= a.position && t <= b.position {
                let span = b.position - a.position;
                if span <= 0.0 {
                    return b.color;
                }
                let f = (t - a.position) / span;
                return [
                    lerp_u8(a.color[0], b.color[0], f),
                    lerp_u8(a.color[1], b.color[1], f),
                    lerp_u8(a.color[2], b.color[2], f),
                ];
            }
        }
        stops[stops.len() - 1].color
    }

    fn build_lut(&self) -> [[u8; 3]; 256] {
        let mut lut = [[0u8; 3]; 256];
        for (i, e) in lut.iter_mut().enumerate() {
            *e = self.sample(i as f32 / 255.0);
        }
        lut
    }
}

#[inline]
fn lerp_u8(a: u8, b: u8, f: f32) -> u8 {
    let v = a as f32 + (b as f32 - a as f32) * f;
    v.round().clamp(0.0, 255.0) as u8
}

#[inline]
fn rec601(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000).min(255) as u8
}

impl ImageFilter for GradientMap {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: GradientMap requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let lut = self.build_lut();
        let w = params.width as usize;
        let h = params.height as usize;
        let row_stride = w * bpp;
        let src = &input.planes[0];

        let out_bpp = match params.format {
            // Gray8 input is upgraded to Rgb24 output — the gradient
            // wants real colour. Pixel format reported by the stream
            // therefore "lies" on Gray8 input; filter docs spell this
            // out. The factory rebuilds the output port.
            PixelFormat::Gray8 => 3usize,
            _ => bpp,
        };
        let out_stride = w * out_bpp;
        let mut data = vec![0u8; out_stride * h];

        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + row_stride];
            let dst = &mut data[y * out_stride..(y + 1) * out_stride];
            for x in 0..w {
                let base = x * bpp;
                let l = match bpp {
                    1 => row[base],
                    3 => rec601(row[base], row[base + 1], row[base + 2]),
                    4 => rec601(row[base], row[base + 1], row[base + 2]),
                    _ => 0,
                };
                let c = lut[l as usize];
                let dbase = x * out_bpp;
                match out_bpp {
                    3 => {
                        dst[dbase] = c[0];
                        dst[dbase + 1] = c[1];
                        dst[dbase + 2] = c[2];
                    }
                    4 => {
                        dst[dbase] = c[0];
                        dst[dbase + 1] = c[1];
                        dst[dbase + 2] = c[2];
                        dst[dbase + 3] = row[base + 3];
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: out_stride,
                data,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
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
    fn requires_two_stops() {
        let bad = GradientMap::new(vec![GradientStop {
            position: 0.0,
            color: [0, 0, 0],
        }]);
        assert!(bad.is_err());
    }

    #[test]
    fn duotone_black_to_red_maps_black_to_black() {
        let g = GradientMap::duotone([0, 0, 0], [255, 0, 0]);
        let input = gray(4, 1, |_, _| 0);
        let out = g.apply(&input, p_gray(4, 1)).unwrap();
        // Output is RGB24 (gradient is RGB).
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, [0u8, 0, 0]);
        }
    }

    #[test]
    fn duotone_black_to_red_maps_white_to_red() {
        let g = GradientMap::duotone([0, 0, 0], [255, 0, 0]);
        let input = gray(4, 1, |_, _| 255);
        let out = g.apply(&input, p_gray(4, 1)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, [255u8, 0, 0]);
        }
    }

    #[test]
    fn midtone_lerps_between_stops() {
        let g = GradientMap::duotone([0, 0, 0], [100, 200, 0]);
        // 128/255 ≈ 0.5 ⇒ approx half of [100, 200, 0].
        let c = g.sample(128.0 / 255.0);
        assert!((c[0] as i16 - 50).abs() <= 1);
        assert!((c[1] as i16 - 100).abs() <= 1);
        assert_eq!(c[2], 0);
    }

    #[test]
    fn tritone_three_stops() {
        let g = GradientMap::tritone([255, 0, 0], [0, 255, 0], [0, 0, 255]);
        assert_eq!(g.sample(0.0), [255, 0, 0]);
        assert_eq!(g.sample(0.5), [0, 255, 0]);
        assert_eq!(g.sample(1.0), [0, 0, 255]);
    }

    #[test]
    fn rgb_input_uses_luma_proxy() {
        // Mid-grey RGB (128, 128, 128) ⇒ luma ≈ 128 ⇒ midtone of
        // the gradient.
        let g = GradientMap::duotone([0, 0, 0], [200, 200, 200]);
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = g.apply(&input, p_rgb(2, 2)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            // ~ 200 * 128 / 255 = 100; allow rounding slack.
            assert!((chunk[0] as i16 - 100).abs() <= 2);
            assert!((chunk[1] as i16 - 100).abs() <= 2);
            assert!((chunk[2] as i16 - 100).abs() <= 2);
        }
    }

    #[test]
    fn rgba_alpha_preserved() {
        let g = GradientMap::duotone([0, 0, 0], [255, 0, 0]);
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            data.extend_from_slice(&[255u8, 255, 255, 77]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = g
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[0], 255);
            assert_eq!(px[1], 0);
            assert_eq!(px[2], 0);
            assert_eq!(px[3], 77);
        }
    }

    #[test]
    fn rejects_yuv420p() {
        let g = GradientMap::duotone([0, 0, 0], [255, 0, 0]);
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
        let err = g
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("GradientMap"));
    }
}
