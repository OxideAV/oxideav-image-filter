//! Uniform colour quantization to a small palette (`-quantize n`).
//!
//! Reduces the number of distinct colours in an image to at most `N`
//! by uniformly subdividing each RGB axis into `cbrt(N)` buckets and
//! snapping every pixel to the centroid of its bucket. Mirrors
//! ImageMagick's `-colors N` (the implicit `-quantize` step in IM 7
//! pipelines), simplified to a uniform-grid quantizer instead of a
//! popularity / median-cut palette.
//!
//! Clean-room implementation derived from the textbook description of
//! a uniform 3-D colour-cube quantizer:
//!
//! 1. Pick `k = ceil(cbrt(target_colors))` levels per axis. (`k = 4`
//!    ⇒ 64 colours, `k = 6` ⇒ 216 colours, etc.)
//! 2. For each pixel, bucket index per axis is `b = (channel * k +
//!    k/2) / 256` (clamped to `[0, k-1]`).
//! 3. Output channel value is `round(bucket * 255 / (k - 1))` so the
//!    palette spans the full `[0, 255]` range.
//!
//! The two-pass dithering / popularity-driven palette of a true
//! median-cut quantizer is intentionally out of scope here — this
//! filter's job is the cheap "snap to a small fixed palette" step
//! that posterise / colour-bucket pipelines need. Use
//! [`Posterize`](crate::Posterize) for an even simpler per-channel
//! version (no inter-channel snapping).
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba`: full effect; alpha is preserved on RGBA.
//! - `Gray8`: degenerates to a 1-D quantizer (each channel snapped
//!   independently).
//! - YUV / other formats return `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Uniform-grid colour quantizer.
#[derive(Clone, Copy, Debug)]
pub struct Quantize {
    /// Maximum number of palette entries. Internally rounded UP to the
    /// next perfect cube `k³` (RGB) or to the value itself (Gray8).
    pub colors: u32,
}

impl Default for Quantize {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Quantize {
    /// Build a quantizer with the given target colour count. Values
    /// `< 2` are clamped to `2` so the output keeps at least black /
    /// white.
    pub fn new(colors: u32) -> Self {
        Self {
            colors: colors.max(2),
        }
    }

    /// Levels-per-axis that this quantizer would use for the given
    /// `colors`. Exposed for unit-testing the rounding-up logic.
    pub fn levels_per_axis(&self) -> u32 {
        let c = self.colors.max(2) as f64;
        // ceil(cbrt(c)).
        let k = c.cbrt().ceil() as u32;
        k.max(2)
    }

    /// Snap a single sample into one of `k` evenly-spaced palette
    /// values. Pulled out so tests can pin the bucketing logic.
    pub fn snap(sample: u8, k: u32) -> u8 {
        let k = k.max(2);
        // Bucket index in [0, k-1].
        let bucket = (sample as u32 * k / 256).min(k - 1);
        // Map bucket back to an evenly-spaced output in [0, 255].
        ((bucket * 255 + (k - 1) / 2) / (k - 1)) as u8
    }
}

impl ImageFilter for Quantize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, false),
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Quantize requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        let k = if matches!(params.format, PixelFormat::Gray8) {
            self.colors.clamp(2, 256)
        } else {
            self.levels_per_axis()
        };
        // Build a per-sample LUT once.
        let mut lut = [0u8; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = Self::snap(i as u8, k);
        }
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                match bpp {
                    1 => {
                        dst_row[base] = lut[src_row[base] as usize];
                    }
                    3 => {
                        dst_row[base] = lut[src_row[base] as usize];
                        dst_row[base + 1] = lut[src_row[base + 1] as usize];
                        dst_row[base + 2] = lut[src_row[base + 2] as usize];
                    }
                    4 => {
                        dst_row[base] = lut[src_row[base] as usize];
                        dst_row[base + 1] = lut[src_row[base + 1] as usize];
                        dst_row[base + 2] = lut[src_row[base + 2] as usize];
                        if has_alpha {
                            dst_row[base + 3] = src_row[base + 3];
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_bytes,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn levels_per_axis_rounds_up() {
        // 8 colours ⇒ k = cbrt(8) = 2.
        assert_eq!(Quantize::new(8).levels_per_axis(), 2);
        // 27 colours ⇒ k = 3.
        assert_eq!(Quantize::new(27).levels_per_axis(), 3);
        // 64 colours ⇒ k = 4.
        assert_eq!(Quantize::new(64).levels_per_axis(), 4);
        // 65 colours rounds up to k = 5 (5³ = 125).
        assert_eq!(Quantize::new(65).levels_per_axis(), 5);
    }

    #[test]
    fn snap_with_k_2_outputs_extremes() {
        // k = 2 ⇒ output palette {0, 255}.
        assert_eq!(Quantize::snap(0, 2), 0);
        assert_eq!(Quantize::snap(127, 2), 0);
        assert_eq!(Quantize::snap(128, 2), 255);
        assert_eq!(Quantize::snap(255, 2), 255);
    }

    #[test]
    fn output_palette_is_at_most_k_per_axis() {
        let input = rgb(16, 16, |x, y| {
            [(x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8]
        });
        let out = Quantize::new(64).apply(&input, params_rgb(16, 16)).unwrap();
        let mut r = HashSet::new();
        let mut g = HashSet::new();
        let mut b = HashSet::new();
        for chunk in out.planes[0].data.chunks(3) {
            r.insert(chunk[0]);
            g.insert(chunk[1]);
            b.insert(chunk[2]);
        }
        // 64 colours ⇒ k = 4 ⇒ at most 4 distinct values per axis.
        assert!(r.len() <= 4, "R palette too large: {r:?}");
        assert!(g.len() <= 4, "G palette too large: {g:?}");
        assert!(b.len() <= 4, "B palette too large: {b:?}");
    }

    #[test]
    fn alpha_preserved_on_rgba() {
        let data: Vec<u8> = (0..16).flat_map(|i| [10u8, 50, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Quantize::new(8)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8);
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
        let err = Quantize::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Quantize"));
    }
}
