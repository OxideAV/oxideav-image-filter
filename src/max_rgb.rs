//! Per-pixel **MaxRGB / MinRGB** channel collapse — output the brightest
//! (or darkest) of the three RGB components as a single greyscale value.
//!
//! Where [`Grayscale`](crate::grayscale::Grayscale) computes the Rec. 601
//! luma `0.299·R + 0.587·G + 0.114·B` and where
//! [`BwMix`](crate::bw_mix::BwMix) generalises the same weighted-sum to
//! arbitrary `(wr, wg, wb)`, `MaxRgb` ignores the weighting entirely and
//! returns the pointwise *maximum* (or minimum) of the three channels:
//!
//! ```text
//! M(R, G, B) = max(R, G, B)        (Mode::Max)
//! m(R, G, B) = min(R, G, B)        (Mode::Min)
//! ```
//!
//! The "Max" mode is exactly the **V (value)** channel of the HSV
//! colour space (Smith, *Color Gamut Transform Pairs*, SIGGRAPH 1978),
//! making this filter a one-line HSV-V extractor without paying for the
//! full HSV round-trip. The "Min" mode is the symmetric darkest-channel
//! collapse used as a chroma proxy in some saturation estimators
//! (`S_HSV = (V - m) / V`).
//!
//! The filter is stateless, per-pixel, branch-free, and `O(W·H)`.
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba` — collapse R/G/B to a single byte per pixel.
//!   With `keep_format = false` (default) the output is `Gray8`; with
//!   `keep_format = true` the output keeps the original RGB / RGBA
//!   shape with all three colour channels equal to the picked
//!   max-or-min value. RGBA alpha is preserved unconditionally.
//! - `Gray8` — identity (every "channel" already coincides).
//! - `Yuv*P` — returns `Error::unsupported`: the operator is defined
//!   on RGB-space data; the Y plane alone does not carry the
//!   per-channel information the operator needs.
//!
//! # Clean-room reference
//!
//! The HSV value definition `V = max(R, G, B)` appears in Smith 1978
//! and is universal across colour-graphics texts; no external library
//! source was consulted. Stateless single-frame definition, derivable
//! from the one-line formula above.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Collapse direction for [`MaxRgb`]: take the pointwise channel
/// maximum (HSV-V) or minimum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaxRgbMode {
    /// `out = max(R, G, B)` — the V channel of HSV.
    #[default]
    Max,
    /// `out = min(R, G, B)` — the dual; useful as a chroma proxy.
    Min,
}

/// Per-pixel max-of-RGB (or min-of-RGB) collapse to greyscale.
///
/// See the [module-level docs](self) for the mathematical definition
/// and the pixel-format coverage.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxRgb {
    /// Which collapse direction — [`MaxRgbMode::Max`] (default) or
    /// [`MaxRgbMode::Min`].
    pub mode: MaxRgbMode,
    /// If `false` (default) the output is single-plane `Gray8`. If
    /// `true` the output preserves the input's RGB / RGBA shape with
    /// all three colour channels equalled to the picked value (alpha
    /// pass-through for RGBA).
    pub keep_format: bool,
}

impl MaxRgb {
    /// Build a `Max` (HSV-V) collapse with `keep_format = false`.
    pub fn new() -> Self {
        Self {
            mode: MaxRgbMode::Max,
            keep_format: false,
        }
    }

    /// Build a `Min` collapse with `keep_format = false`.
    pub fn min() -> Self {
        Self {
            mode: MaxRgbMode::Min,
            keep_format: false,
        }
    }

    /// Toggle output-shape preservation. With `keep = true` the output
    /// matches the input pixel format (RGB / RGBA) instead of
    /// collapsing to `Gray8`.
    pub fn with_keep_format(mut self, keep: bool) -> Self {
        self.keep_format = keep;
        self
    }
}

impl ImageFilter for MaxRgb {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            PixelFormat::Gray8 => {
                // Identity: every "channel" already coincides.
                return Ok(input.clone());
            }
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: MaxRgb requires Rgb24/Rgba/Gray8, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: MaxRgb requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        if src.stride < w * bpp {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: MaxRgb input stride {} < width*bpp {}",
                src.stride,
                w * bpp
            )));
        }
        let pick = |r: u8, g: u8, b: u8| -> u8 {
            match self.mode {
                MaxRgbMode::Max => r.max(g).max(b),
                MaxRgbMode::Min => r.min(g).min(b),
            }
        };

        if self.keep_format {
            let stride = w * bpp;
            let mut out = vec![0u8; stride * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + stride];
                let d = &mut out[y * stride..(y + 1) * stride];
                for x in 0..w {
                    let base = x * bpp;
                    let v = pick(s[base], s[base + 1], s[base + 2]);
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
                let s_row = &src.data[y * src.stride..y * src.stride + w * bpp];
                let d_row = &mut out[y * stride..(y + 1) * stride];
                for (x, slot) in d_row.iter_mut().enumerate().take(w) {
                    let base = x * bpp;
                    *slot = pick(s_row[base], s_row[base + 1], s_row[base + 2]);
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

    fn p_rgba(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn max_collapses_to_brightest_channel() {
        // Hand-derived expected output:
        //   pixel (10, 200, 50)  → max = 200, min = 10.
        let input = rgb(3, 2, |_, _| [10, 200, 50]);
        let out = MaxRgb::new().apply(&input, p_rgb(3, 2)).unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 3);
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 200);
        }
    }

    #[test]
    fn min_collapses_to_darkest_channel() {
        let input = rgb(3, 2, |_, _| [10, 200, 50]);
        let out = MaxRgb::min().apply(&input, p_rgb(3, 2)).unwrap();
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 10);
        }
    }

    #[test]
    fn max_equals_hsv_value_for_pure_primaries() {
        // HSV: V of pure red (255,0,0) = 255; pure green (0,255,0) = 255;
        // pure blue (0,0,255) = 255; mid-grey (128,128,128) = 128;
        // black (0,0,0) = 0; white (255,255,255) = 255.
        let cases: &[([u8; 3], u8, u8)] = &[
            ([255, 0, 0], 255, 0),
            ([0, 255, 0], 255, 0),
            ([0, 0, 255], 255, 0),
            ([128, 128, 128], 128, 128),
            ([0, 0, 0], 0, 0),
            ([255, 255, 255], 255, 255),
            ([12, 34, 56], 56, 12),
            ([200, 50, 100], 200, 50),
        ];
        for (rgb_in, expect_max, expect_min) in cases {
            let f = rgb(1, 1, |_, _| *rgb_in);
            let out = MaxRgb::new().apply(&f, p_rgb(1, 1)).unwrap();
            assert_eq!(out.planes[0].data[0], *expect_max, "max({rgb_in:?})");
            let out = MaxRgb::min().apply(&f, p_rgb(1, 1)).unwrap();
            assert_eq!(out.planes[0].data[0], *expect_min, "min({rgb_in:?})");
        }
    }

    #[test]
    fn keep_format_emits_rgb_with_equal_channels() {
        let input = rgb(2, 2, |_, _| [10, 200, 50]);
        let out = MaxRgb::new()
            .with_keep_format(true)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        assert_eq!(out.planes[0].stride, 6);
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 200);
            assert_eq!(chunk[1], 200);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn alpha_pass_through_on_rgba_keep_format() {
        // 2×2 RGBA: (10, 200, 50, alpha) where alpha cycles 0..4.
        let mut data = Vec::with_capacity(16);
        for i in 0..4u8 {
            data.extend_from_slice(&[10, 200, 50, i * 60]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = MaxRgb::new()
            .with_keep_format(true)
            .apply(&frame, p_rgba(2, 2))
            .unwrap();
        for (i, chunk) in out.planes[0].data.chunks(4).enumerate() {
            assert_eq!(chunk[0], 200);
            assert_eq!(chunk[1], 200);
            assert_eq!(chunk[2], 200);
            assert_eq!(chunk[3], (i as u8) * 60, "alpha must pass through");
        }
    }

    #[test]
    fn rgba_default_collapses_to_gray8_dropping_alpha() {
        // The default `keep_format = false` collapses to Gray8 so the
        // alpha channel is intentionally dropped (the output shape no
        // longer has an alpha slot).
        let mut data = Vec::with_capacity(16);
        for _ in 0..4 {
            data.extend_from_slice(&[10u8, 200, 50, 99]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = MaxRgb::new().apply(&frame, p_rgba(2, 2)).unwrap();
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(out.planes[0].data.len(), 4);
        for v in out.planes[0].data.iter() {
            assert_eq!(*v, 200);
        }
    }

    #[test]
    fn gray8_is_identity() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![
                    10, 50, 100, 200, 30, 80, 150, 220, 5, 5, 5, 5, 0, 64, 128, 255,
                ],
            }],
        };
        let out = MaxRgb::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
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
        let err = MaxRgb::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("MaxRgb"));
    }

    #[test]
    fn ignores_stride_padding() {
        // Wide stride: rows have 8 bytes per pixel of trailing padding.
        let w = 2u32;
        let h = 2u32;
        let stride = 12usize; // 2 px * 3 bpp = 6 valid bytes, 6 byte padding
        let mut data = Vec::with_capacity(stride * h as usize);
        for _ in 0..h {
            data.extend_from_slice(&[10, 200, 50]); // px 0
            data.extend_from_slice(&[3, 4, 250]); // px 1
            data.extend_from_slice(&[99, 99, 99, 99, 99, 99]); // padding
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        };
        let out = MaxRgb::new().apply(&frame, p_rgb(w, h)).unwrap();
        assert_eq!(out.planes[0].stride, 2);
        // Row 0: px 0 → max=200; px 1 → max=250.
        assert_eq!(out.planes[0].data[0], 200);
        assert_eq!(out.planes[0].data[1], 250);
        assert_eq!(out.planes[0].data[2], 200);
        assert_eq!(out.planes[0].data[3], 250);
    }

    #[test]
    fn pts_pass_through() {
        let input = VideoFrame {
            pts: Some(42),
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![10, 200, 50],
            }],
        };
        let out = MaxRgb::new().apply(&input, p_rgb(1, 1)).unwrap();
        assert_eq!(out.pts, Some(42));
    }
}
