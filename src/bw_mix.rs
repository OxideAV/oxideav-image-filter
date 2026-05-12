//! Black-and-white conversion with per-channel weights — the
//! "channel mixer" approach photographers use to mimic colour-filter
//! greyscale film (orange filter for dramatic skies, green for warmer
//! foliage, etc.).
//!
//! Where [`Grayscale`](crate::grayscale::Grayscale) uses fixed Rec. 601
//! luma weights `(0.299, 0.587, 0.114)`, `BwMix` exposes the three
//! weights as parameters. The classic photo-filter recipes are:
//!
//! - **Red filter** (`(1.0, 0.0, 0.0)`) — darkens skies, lightens
//!   red foliage.
//! - **Orange filter** (`(0.5, 0.5, 0.0)`) — moderate sky drama.
//! - **Yellow filter** (`(0.3, 0.6, 0.1)`) — close to Rec. 601, slightly
//!   warmer.
//! - **Green filter** (`(0.0, 1.0, 0.0)`) — flattering on caucasian skin.
//! - **Blue filter** (`(0.0, 0.0, 1.0)`) — lightens skies, darkens foliage.
//!
//! Weights are not constrained to sum to 1.0 — a sum > 1 brightens, < 1
//! darkens. The output is always greyscale: by default we emit `Gray8`
//! (single-plane), but the `keep_format` flag preserves the input shape
//! (RGB / RGBA with all three channels equal). YUV inputs return
//! `Unsupported` because the per-channel weights only make sense on
//! RGB-space data — IM mirrors this on `-colorspace RGB`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// B&W mixer with per-channel weights.
#[derive(Clone, Copy, Debug)]
pub struct BwMix {
    /// `(r_weight, g_weight, b_weight)`. Non-finite entries coerce to
    /// `0.0`.
    pub weights: [f32; 3],
    /// If `false` (default) the output is a single-plane `Gray8`; if
    /// `true` the input shape (Rgb24 / Rgba) is preserved with all
    /// three channels set to the mixed value (alpha pass-through).
    pub keep_format: bool,
}

impl Default for BwMix {
    fn default() -> Self {
        // Rec. 601 baseline matches `Grayscale`.
        Self {
            weights: [0.299, 0.587, 0.114],
            keep_format: false,
        }
    }
}

impl BwMix {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        let f = |v: f32| if v.is_finite() { v } else { 0.0 };
        Self {
            weights: [f(r), f(g), f(b)],
            keep_format: false,
        }
    }

    pub fn with_keep_format(mut self, keep: bool) -> Self {
        self.keep_format = keep;
        self
    }

    /// "Red filter": dramatic skies.
    pub fn red_filter() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// "Green filter": flatters skin.
    pub fn green_filter() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// "Blue filter": light skies, dark foliage.
    pub fn blue_filter() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }
}

impl ImageFilter for BwMix {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: BwMix requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: BwMix requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let [wr, wg, wb] = self.weights;
        let mix = |r: u8, g: u8, b: u8| -> u8 {
            let v = r as f32 * wr + g as f32 * wg + b as f32 * wb;
            v.round().clamp(0.0, 255.0) as u8
        };
        if self.keep_format {
            let stride = w * bpp;
            let mut out = vec![0u8; stride * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + stride];
                let d = &mut out[y * stride..(y + 1) * stride];
                for x in 0..w {
                    let base = x * bpp;
                    let v = mix(s[base], s[base + 1], s[base + 2]);
                    d[base] = v;
                    d[base + 1] = v;
                    d[base + 2] = v;
                    if has_alpha {
                        d[base + 3] = s[base + 3];
                    }
                }
            }
            Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane { stride, data: out }],
            })
        } else {
            let stride = w;
            let mut out = vec![0u8; stride * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + (w * bpp)];
                let d = &mut out[y * stride..(y + 1) * stride];
                for (x, slot) in d.iter_mut().enumerate().take(w) {
                    let base = x * bpp;
                    *slot = mix(s[base], s[base + 1], s[base + 2]);
                }
            }
            Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane { stride, data: out }],
            })
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
    fn red_filter_picks_only_red_channel() {
        let input = rgb(2, 2, |_, _| [123, 200, 50]);
        let out = BwMix::red_filter().apply(&input, p_rgb(2, 2)).unwrap();
        // Single-plane Gray8 output.
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 2);
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 123, "red filter should pick R channel only");
        }
    }

    #[test]
    fn green_filter_picks_green_channel() {
        let input = rgb(2, 2, |_, _| [123, 200, 50]);
        let out = BwMix::green_filter().apply(&input, p_rgb(2, 2)).unwrap();
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 200);
        }
    }

    #[test]
    fn keep_format_emits_rgb_with_equal_channels() {
        let input = rgb(2, 2, |_, _| [123, 200, 50]);
        let out = BwMix::red_filter()
            .with_keep_format(true)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        assert_eq!(out.planes[0].stride, 6);
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
            assert_eq!(chunk[0], 123);
        }
    }

    #[test]
    fn default_matches_rec601_within_rounding() {
        let input = rgb(2, 2, |_, _| [200, 100, 50]);
        let out = BwMix::default().apply(&input, p_rgb(2, 2)).unwrap();
        // Rec. 601: 0.299*200 + 0.587*100 + 0.114*50 = ~124.2
        let v = out.planes[0].data[0];
        assert!((v as i16 - 124).abs() <= 1, "got {v}");
    }

    #[test]
    fn alpha_passes_through_in_keep_format() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[123u8, 200, 50, 77]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = BwMix::red_filter()
            .with_keep_format(true)
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
            assert_eq!(chunk[3], 77);
        }
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
        let err = BwMix::red_filter()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("BwMix"));
    }
}
