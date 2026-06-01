//! Selective colour — adjust hue, saturation, and lightness per
//! primary / secondary hue band (Reds / Yellows / Greens / Cyans /
//! Blues / Magentas).
//!
//! Each pixel is routed through HSL space, classified into the hue
//! band whose centre it sits closest to, weighted by a triangular
//! response that peaks at the band centre and rolls off to zero at the
//! neighbouring band centre. The chosen band's H / S / L deltas are
//! applied scaled by that weight, then the pixel is rendered back to
//! RGB.
//!
//! Clean-room hue bands follow the standard 6-band HSL convention used
//! by classical photo-editor selective-colour panels:
//!
//! ```text
//! Reds      0°
//! Yellows  60°
//! Greens  120°
//! Cyans   180°
//! Blues   240°
//! Magentas 300°
//! ```
//!
//! Achromatic pixels (`R = G = B`) bypass the per-band adjustment
//! because they have no meaningful hue.
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba` only. Alpha pass-through. Other formats return
//! [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// The six named hue bands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HueBand {
    Reds,
    Yellows,
    Greens,
    Cyans,
    Blues,
    Magentas,
}

impl HueBand {
    /// Hue centre in degrees.
    pub fn centre_degrees(&self) -> f32 {
        match self {
            HueBand::Reds => 0.0,
            HueBand::Yellows => 60.0,
            HueBand::Greens => 120.0,
            HueBand::Cyans => 180.0,
            HueBand::Blues => 240.0,
            HueBand::Magentas => 300.0,
        }
    }

    /// Parse from a case-insensitive name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "reds" | "red" => Some(Self::Reds),
            "yellows" | "yellow" => Some(Self::Yellows),
            "greens" | "green" => Some(Self::Greens),
            "cyans" | "cyan" => Some(Self::Cyans),
            "blues" | "blue" => Some(Self::Blues),
            "magentas" | "magenta" => Some(Self::Magentas),
            _ => None,
        }
    }
}

/// Per-band HSL deltas.
#[derive(Clone, Copy, Debug, Default)]
pub struct BandAdjust {
    /// Hue shift in degrees, applied to pixels in this band.
    pub hue_shift_degrees: f32,
    /// Saturation shift in `[-1, 1]`; `0` is identity.
    pub saturation_shift: f32,
    /// Lightness shift in `[-1, 1]`; `0` is identity.
    pub lightness_shift: f32,
}

/// Selective colour filter.
#[derive(Clone, Debug, Default)]
pub struct SelectiveColor {
    pub reds: BandAdjust,
    pub yellows: BandAdjust,
    pub greens: BandAdjust,
    pub cyans: BandAdjust,
    pub blues: BandAdjust,
    pub magentas: BandAdjust,
}

impl SelectiveColor {
    /// Build with all bands zeroed (identity filter).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-band adjustment, replacing the previous values.
    pub fn with_band(mut self, band: HueBand, adj: BandAdjust) -> Self {
        match band {
            HueBand::Reds => self.reds = adj,
            HueBand::Yellows => self.yellows = adj,
            HueBand::Greens => self.greens = adj,
            HueBand::Cyans => self.cyans = adj,
            HueBand::Blues => self.blues = adj,
            HueBand::Magentas => self.magentas = adj,
        }
        self
    }

    fn band_for(&self, band: HueBand) -> BandAdjust {
        match band {
            HueBand::Reds => self.reds,
            HueBand::Yellows => self.yellows,
            HueBand::Greens => self.greens,
            HueBand::Cyans => self.cyans,
            HueBand::Blues => self.blues,
            HueBand::Magentas => self.magentas,
        }
    }
}

impl ImageFilter for SelectiveColor {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: SelectiveColor requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        let row_stride = w * bpp;
        let src = &input.planes[0];
        let mut data = vec![0u8; row_stride * h];

        let bands = [
            HueBand::Reds,
            HueBand::Yellows,
            HueBand::Greens,
            HueBand::Cyans,
            HueBand::Blues,
            HueBand::Magentas,
        ];

        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + row_stride];
            let dst = &mut data[y * row_stride..(y + 1) * row_stride];
            for x in 0..w {
                let base = x * bpp;
                let r = row[base];
                let g = row[base + 1];
                let b = row[base + 2];
                let (mut hh, mut s, mut l) = rgb_to_hsl(r, g, b);
                let is_grey = r == g && g == b;
                if !is_grey {
                    // Determine the two flanking bands (60° spacing).
                    let h_norm = ((hh % 360.0) + 360.0) % 360.0;
                    let bin = (h_norm / 60.0).floor() as usize % 6;
                    let next_bin = (bin + 1) % 6;
                    let t = (h_norm - bin as f32 * 60.0) / 60.0; // [0, 1]
                                                                 // Triangular weights — left peaks at t=0, right at t=1.
                    let w_left = 1.0 - t;
                    let w_right = t;
                    let left = self.band_for(bands[bin]);
                    let right = self.band_for(bands[next_bin]);
                    let dh = left.hue_shift_degrees * w_left + right.hue_shift_degrees * w_right;
                    let ds = left.saturation_shift * w_left + right.saturation_shift * w_right;
                    let dl = left.lightness_shift * w_left + right.lightness_shift * w_right;
                    hh = (hh + dh) % 360.0;
                    if hh < 0.0 {
                        hh += 360.0;
                    }
                    s = (s + ds).clamp(0.0, 1.0);
                    l = (l + dl).clamp(0.0, 1.0);
                }
                let (r2, g2, b2) = hsl_to_rgb(hh, s, l);
                dst[base] = r2;
                dst[base + 1] = g2;
                dst[base + 2] = b2;
                if has_alpha {
                    dst[base + 3] = row[base + 3];
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_stride,
                data,
            }],
        })
    }
}

// Clean-room HSL ⇄ RGB; identical formulae as `hsl_rotate` /
// `hsl_shift` already in-tree, re-implemented here to keep this
// module self-contained.

fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let l = (max + min) * 0.5;
    let d = max - min;
    let (h, s) = if d == 0.0 {
        (0.0, 0.0)
    } else {
        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };
        let h = if max == rf {
            ((gf - bf) / d) + if gf < bf { 6.0 } else { 0.0 }
        } else if max == gf {
            ((bf - rf) / d) + 2.0
        } else {
            ((rf - gf) / d) + 4.0
        };
        (h * 60.0, s)
    };
    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    if s == 0.0 {
        let v = (l * 255.0).round().clamp(0.0, 255.0) as u8;
        return (v, v, v);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = ((h % 360.0 + 360.0) % 360.0) / 360.0;
    let r = hue_to_rgb(p, q, hk + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, hk);
    let b = hue_to_rgb(p, q, hk - 1.0 / 3.0);
    (
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = if t < 0.0 {
        t + 1.0
    } else if t > 1.0 {
        t - 1.0
    } else {
        t
    };
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
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
    fn identity_passes_through() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 80]);
        let out = SelectiveColor::new().apply(&input, p_rgb(4, 4)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn red_band_desaturates_pure_reds() {
        let f = SelectiveColor::new().with_band(
            HueBand::Reds,
            BandAdjust {
                hue_shift_degrees: 0.0,
                saturation_shift: -1.0,
                lightness_shift: 0.0,
            },
        );
        let input = rgb(2, 2, |_, _| [200, 0, 0]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        // Saturation crushed to zero ⇒ output is greyish (R == G == B).
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
        }
    }

    #[test]
    fn green_band_does_not_touch_blue_pixels() {
        let f = SelectiveColor::new().with_band(
            HueBand::Greens,
            BandAdjust {
                hue_shift_degrees: 90.0,
                saturation_shift: -1.0,
                lightness_shift: 0.0,
            },
        );
        // Pure blue is at 240° — well clear of greens (120°) and the
        // triangular response between Blues / Cyans means it gets zero
        // weight from the greens band.
        let input = rgb(1, 1, |_, _| [0, 0, 200]);
        let out = f.apply(&input, p_rgb(1, 1)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn achromatic_pixels_are_passthrough() {
        let f = SelectiveColor::new().with_band(
            HueBand::Reds,
            BandAdjust {
                hue_shift_degrees: 180.0,
                saturation_shift: 1.0,
                lightness_shift: 0.5,
            },
        );
        let input = rgb(3, 1, |x, _| {
            let v = [50u8, 128, 200][x as usize];
            [v, v, v]
        });
        let out = f.apply(&input, p_rgb(3, 1)).unwrap();
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn band_names_parse_case_insensitive() {
        assert_eq!(HueBand::from_name("Reds"), Some(HueBand::Reds));
        assert_eq!(HueBand::from_name("red"), Some(HueBand::Reds));
        assert_eq!(HueBand::from_name("CYAN"), Some(HueBand::Cyans));
        assert_eq!(HueBand::from_name("blues"), Some(HueBand::Blues));
        assert_eq!(HueBand::from_name("orchid"), None);
    }

    #[test]
    fn rejects_yuv420p() {
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
        let err = SelectiveColor::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("SelectiveColor"));
    }

    #[test]
    fn rgba_alpha_pass_through() {
        let f = SelectiveColor::new().with_band(
            HueBand::Reds,
            BandAdjust {
                hue_shift_degrees: 0.0,
                saturation_shift: -0.5,
                lightness_shift: 0.0,
            },
        );
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            data.extend_from_slice(&[200u8, 50, 30, 99]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = f
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
            assert_eq!(px[3], 99, "alpha not preserved");
        }
    }
}
