//! Per-channel auto-levels (`-auto-level`).
//!
//! For RGB / RGBA inputs each channel is stretched independently to
//! fill `[0, 255]`. This differs from [`Normalize`](crate::Normalize),
//! which scans a luma proxy and applies a single `[lo, hi]` window to
//! every channel — same window everywhere preserves hue, per-channel
//! stretch maximises contrast at the cost of a small global colour
//! cast. Both behaviours are useful; ImageMagick exposes both as
//! `-normalize` and `-auto-level` respectively.
//!
//! Clean-room implementation: per-channel histogram min/max → linear
//! remap.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Auto-level: stretch each channel independently to fill `[0, 255]`.
///
/// On `Gray8` and `Yuv*P` only the single tone plane is stretched, so
/// the result matches a per-luma `Normalize` with no clip percentage.
/// On `Rgb24` / `Rgba` each of `R`, `G`, `B` is stretched to its own
/// observed `[min, max]` independently. Alpha (RGBA) is preserved.
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoLevel;

impl AutoLevel {
    /// Build the auto-level filter (no parameters — it is a pure
    /// per-channel stretch).
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for AutoLevel {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: AutoLevel does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut out = input.clone();
        match params.format {
            PixelFormat::Gray8
            | PixelFormat::Yuv420P
            | PixelFormat::Yuv422P
            | PixelFormat::Yuv444P => {
                let lut = single_channel_lut(&out.planes[0].data, out.planes[0].stride, w, h);
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
                // chroma untouched on YUV — keep the colour balance intact.
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let lut_r = packed_channel_lut(&p.data, stride, w, h, 3, 0);
                let lut_g = packed_channel_lut(&p.data, stride, w, h, 3, 1);
                let lut_b = packed_channel_lut(&p.data, stride, w, h, 3, 2);
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 3 * w];
                    for x in 0..w {
                        row[3 * x] = lut_r[row[3 * x] as usize];
                        row[3 * x + 1] = lut_g[row[3 * x + 1] as usize];
                        row[3 * x + 2] = lut_b[row[3 * x + 2] as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let lut_r = packed_channel_lut(&p.data, stride, w, h, 4, 0);
                let lut_g = packed_channel_lut(&p.data, stride, w, h, 4, 1);
                let lut_b = packed_channel_lut(&p.data, stride, w, h, 4, 2);
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        row[4 * x] = lut_r[row[4 * x] as usize];
                        row[4 * x + 1] = lut_g[row[4 * x + 1] as usize];
                        row[4 * x + 2] = lut_b[row[4 * x + 2] as usize];
                        // alpha preserved
                    }
                }
            }
            _ => unreachable!("is_supported_format already gated"),
        }
        Ok(out)
    }
}

fn single_channel_lut(data: &[u8], stride: usize, w: usize, h: usize) -> [u8; 256] {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for y in 0..h {
        for v in &data[y * stride..y * stride + w] {
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
    }
    build_stretch_lut(lo, hi)
}

fn packed_channel_lut(
    data: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    bpp: usize,
    ch: usize,
) -> [u8; 256] {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for y in 0..h {
        let row = &data[y * stride..y * stride + bpp * w];
        for x in 0..w {
            let v = row[bpp * x + ch];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    build_stretch_lut(lo, hi)
}

fn build_stretch_lut(lo: u8, hi: u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if hi <= lo {
        // Flat channel — leave unchanged.
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = i as u8;
        }
        return lut;
    }
    let lo_f = lo as f32;
    let range = (hi - lo) as f32;
    for (i, slot) in lut.iter_mut().enumerate() {
        let v = (i as f32 - lo_f) / range;
        if v <= 0.0 {
            *slot = 0;
        } else if v >= 1.0 {
            *slot = 255;
        } else {
            *slot = (v * 255.0).round() as u8;
        }
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn gray(w: u32, h: u32, pat: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pat(x, y));
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

    #[test]
    fn gray_fills_full_range() {
        // Samples in [64, 192] should stretch to [0, 255].
        let input = gray(8, 1, |x, _| 64 + (x * 16) as u8);
        let out = AutoLevel::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 1,
                },
            )
            .unwrap();
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn rgba_stretches_each_channel_independently() {
        // R in [10, 60], G in [100, 150], B in [200, 250]. After auto-level
        // each channel's min should be 0 and max 255.
        let mut data = Vec::with_capacity(16 * 4);
        for i in 0..16u8 {
            data.extend_from_slice(&[
                10 + i * 3,  // R: 10..58
                100 + i * 3, // G: 100..148
                200 + i * 3, // B: 200..245
                64,
            ]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 64, data }],
        };
        let out = AutoLevel::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 16,
                    height: 1,
                },
            )
            .unwrap();
        let r0 = out.planes[0].data[0];
        let r15 = out.planes[0].data[15 * 4];
        let g0 = out.planes[0].data[1];
        let g15 = out.planes[0].data[15 * 4 + 1];
        let b0 = out.planes[0].data[2];
        let b15 = out.planes[0].data[15 * 4 + 2];
        assert_eq!(r0, 0);
        assert_eq!(r15, 255);
        assert_eq!(g0, 0);
        assert_eq!(g15, 255);
        assert_eq!(b0, 0);
        assert_eq!(b15, 255);
        // alpha preserved
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 64);
        }
    }

    #[test]
    fn flat_channel_passes_through() {
        let input = gray(4, 4, |_, _| 120);
        let out = AutoLevel::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 120);
        }
    }
}
