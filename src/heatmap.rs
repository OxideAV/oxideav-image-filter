//! Heatmap — apply a built-in colour ramp to a grayscale image.
//!
//! Useful for scientific visualisations / depth maps / thermal-camera
//! style false-colour. The source is reduced to a single luminance value
//! per pixel (Rec. 601 weights for RGB; Y plane for YUV; the data byte
//! for Gray8), used as a `[0, 255]` index into a 256-entry per-channel
//! lookup table, and re-emitted as RGB / RGBA.
//!
//! Ramps shipped:
//!
//! - **Jet** — the classic blue → cyan → green → yellow → red
//!   sequence used by old MATLAB plots.
//! - **Viridis** — perceptually uniform purple → blue → green → yellow.
//! - **Plasma** — perceptually uniform deep-purple → magenta → yellow.
//! - **Hot** — black → red → yellow → white (matplotlib `hot`).
//! - **Cool** — cyan → magenta linear two-stop.
//! - **Grayscale** — degenerate identity ramp; useful for tests.
//!
//! The ramps are evaluated from clean-room piecewise-linear key points;
//! no external palette files are bundled.
//!
//! # Pixel formats
//!
//! Source: `Gray8`, `Rgb24`, `Rgba`, planar `Yuv420P` / `Yuv422P` /
//! `Yuv444P`. Output is always `Rgb24` (or `Rgba` when the input was
//! `Rgba` — the source alpha is preserved). Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Built-in colour ramps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeatmapRamp {
    /// Classic blue → cyan → green → yellow → red (MATLAB jet).
    #[default]
    Jet,
    /// Perceptually-uniform purple → blue → green → yellow (viridis).
    Viridis,
    /// Perceptually-uniform deep-purple → magenta → yellow (plasma).
    Plasma,
    /// Black → red → yellow → white (matplotlib hot).
    Hot,
    /// Cyan → magenta linear two-stop.
    Cool,
    /// Identity ramp (`v ↦ (v, v, v)`); useful for tests.
    Grayscale,
}

impl HeatmapRamp {
    /// Parse a ramp name (case-insensitive). Returns `None` on unknown
    /// names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "jet" => Some(Self::Jet),
            "viridis" => Some(Self::Viridis),
            "plasma" => Some(Self::Plasma),
            "hot" => Some(Self::Hot),
            "cool" => Some(Self::Cool),
            "grayscale" | "greyscale" | "gray" | "grey" => Some(Self::Grayscale),
            _ => None,
        }
    }
}

/// False-colour heatmap filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heatmap {
    ramp: HeatmapRamp,
}

impl Heatmap {
    /// Build with the named ramp.
    pub fn new(ramp: HeatmapRamp) -> Self {
        Self { ramp }
    }
}

impl ImageFilter for Heatmap {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let lut = build_lut(self.ramp);
        let w = params.width as usize;
        let h = params.height as usize;
        match params.format {
            PixelFormat::Gray8 => {
                let p = &input.planes[0];
                let mut data = vec![0u8; 3 * w * h];
                for y in 0..h {
                    let row = &p.data[y * p.stride..y * p.stride + w];
                    let out_row = &mut data[y * 3 * w..(y + 1) * 3 * w];
                    for x in 0..w {
                        let v = row[x] as usize;
                        out_row[3 * x] = lut[v][0];
                        out_row[3 * x + 1] = lut[v][1];
                        out_row[3 * x + 2] = lut[v][2];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: 3 * w,
                        data,
                    }],
                })
            }
            PixelFormat::Rgb24 => {
                let p = &input.planes[0];
                let mut data = vec![0u8; 3 * w * h];
                for y in 0..h {
                    let row = &p.data[y * p.stride..y * p.stride + 3 * w];
                    let out_row = &mut data[y * 3 * w..(y + 1) * 3 * w];
                    for x in 0..w {
                        let v = luma(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
                        out_row[3 * x] = lut[v as usize][0];
                        out_row[3 * x + 1] = lut[v as usize][1];
                        out_row[3 * x + 2] = lut[v as usize][2];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: 3 * w,
                        data,
                    }],
                })
            }
            PixelFormat::Rgba => {
                let p = &input.planes[0];
                let mut data = vec![0u8; 4 * w * h];
                for y in 0..h {
                    let row = &p.data[y * p.stride..y * p.stride + 4 * w];
                    let out_row = &mut data[y * 4 * w..(y + 1) * 4 * w];
                    for x in 0..w {
                        let v = luma(row[4 * x], row[4 * x + 1], row[4 * x + 2]);
                        out_row[4 * x] = lut[v as usize][0];
                        out_row[4 * x + 1] = lut[v as usize][1];
                        out_row[4 * x + 2] = lut[v as usize][2];
                        out_row[4 * x + 3] = row[4 * x + 3];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: 4 * w,
                        data,
                    }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let p = &input.planes[0];
                let mut data = vec![0u8; 3 * w * h];
                for y in 0..h {
                    let row = &p.data[y * p.stride..y * p.stride + w];
                    let out_row = &mut data[y * 3 * w..(y + 1) * 3 * w];
                    for x in 0..w {
                        let v = row[x] as usize;
                        out_row[3 * x] = lut[v][0];
                        out_row[3 * x + 1] = lut[v][1];
                        out_row[3 * x + 2] = lut[v][2];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: 3 * w,
                        data,
                    }],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Heatmap does not yet handle {other:?}"
            ))),
        }
    }
}

#[inline]
fn luma(r: u8, g: u8, b: u8) -> u8 {
    // Rec. 601 luma weights ×256 for integer math: 0.299, 0.587, 0.114
    let y = 77 * r as u32 + 150 * g as u32 + 29 * b as u32;
    (y >> 8).min(255) as u8
}

/// Piecewise-linear interpolation: build a 256-entry `[R, G, B]` table
/// by walking the provided keyframes (each at a normalised position
/// `t ∈ [0, 1]`).
fn build_from_keyframes(stops: &[(f32, [u8; 3])]) -> Vec<[u8; 3]> {
    let mut lut = vec![[0u8; 3]; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let t = (i as f32) / 255.0;
        // Find the bracketing key indices.
        let mut a = 0usize;
        while a + 1 < stops.len() && stops[a + 1].0 < t {
            a += 1;
        }
        let b = (a + 1).min(stops.len() - 1);
        let span = (stops[b].0 - stops[a].0).max(1e-6);
        let local = ((t - stops[a].0) / span).clamp(0.0, 1.0);
        for (c, ch) in slot.iter_mut().enumerate().take(3) {
            let p = stops[a].1[c] as f32;
            let q = stops[b].1[c] as f32;
            *ch = (p + (q - p) * local).round().clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

fn build_lut(ramp: HeatmapRamp) -> Vec<[u8; 3]> {
    match ramp {
        HeatmapRamp::Jet => build_from_keyframes(&[
            (0.0, [0, 0, 128]),
            (0.125, [0, 0, 255]),
            (0.375, [0, 255, 255]),
            (0.625, [255, 255, 0]),
            (0.875, [255, 0, 0]),
            (1.0, [128, 0, 0]),
        ]),
        HeatmapRamp::Viridis => build_from_keyframes(&[
            (0.0, [68, 1, 84]),
            (0.25, [59, 82, 139]),
            (0.5, [33, 145, 140]),
            (0.75, [94, 201, 98]),
            (1.0, [253, 231, 37]),
        ]),
        HeatmapRamp::Plasma => build_from_keyframes(&[
            (0.0, [13, 8, 135]),
            (0.25, [126, 3, 168]),
            (0.5, [204, 71, 120]),
            (0.75, [248, 149, 64]),
            (1.0, [240, 249, 33]),
        ]),
        HeatmapRamp::Hot => build_from_keyframes(&[
            (0.0, [0, 0, 0]),
            (0.4, [255, 0, 0]),
            (0.75, [255, 255, 0]),
            (1.0, [255, 255, 255]),
        ]),
        HeatmapRamp::Cool => build_from_keyframes(&[(0.0, [0, 255, 255]), (1.0, [255, 0, 255])]),
        HeatmapRamp::Grayscale => (0..256).map(|i| [i as u8, i as u8, i as u8]).collect(),
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn grayscale_ramp_is_identity_on_gray_input() {
        let input = gray(4, 1, |x, _| (x * 64).min(255) as u8);
        let out = Heatmap::new(HeatmapRamp::Grayscale)
            .apply(&input, p_gray(4, 1))
            .unwrap();
        // 3 bytes per output pixel; R=G=B=input.
        for x in 0..4 {
            let r = out.planes[0].data[3 * x];
            let g = out.planes[0].data[3 * x + 1];
            let b = out.planes[0].data[3 * x + 2];
            assert_eq!(r, g);
            assert_eq!(g, b);
            assert_eq!(r, input.planes[0].data[x]);
        }
    }

    #[test]
    fn jet_endpoints_match_keyframes() {
        let input = gray(2, 1, |x, _| if x == 0 { 0 } else { 255 });
        let out = Heatmap::new(HeatmapRamp::Jet)
            .apply(&input, p_gray(2, 1))
            .unwrap();
        // First key (t=0) ⇒ [0, 0, 128]; last (t=1) ⇒ [128, 0, 0].
        assert_eq!(&out.planes[0].data[0..3], &[0, 0, 128]);
        assert_eq!(&out.planes[0].data[3..6], &[128, 0, 0]);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..4).flat_map(|i| [i * 60, i * 60, i * 60, 77]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Heatmap::new(HeatmapRamp::Hot)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        for i in 0..4 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 77, "alpha at pixel {i}");
        }
        // First pixel maps from black ⇒ first hot stop is black.
        assert_eq!(&out.planes[0].data[0..3], &[0, 0, 0]);
    }

    #[test]
    fn from_name_table() {
        assert_eq!(HeatmapRamp::from_name("Jet"), Some(HeatmapRamp::Jet));
        assert_eq!(
            HeatmapRamp::from_name("VIRIDIS"),
            Some(HeatmapRamp::Viridis)
        );
        assert_eq!(
            HeatmapRamp::from_name("greyscale"),
            Some(HeatmapRamp::Grayscale)
        );
        assert_eq!(HeatmapRamp::from_name("blah"), None);
    }
}
