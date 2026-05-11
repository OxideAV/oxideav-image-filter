//! Linear stretch (`-linear-stretch black-pixels{xwhite-pixels}`).
//!
//! Like [`ContrastStretch`](crate::ContrastStretch) but the cut-off
//! amounts are given as **absolute pixel counts** rather than
//! percentages. ImageMagick's `-linear-stretch 100x50` says "burn 100
//! pixels to black, blow out 50 pixels to white" regardless of image
//! size.
//!
//! Clean-room: identical algorithm to ContrastStretch but the
//! count→budget step is direct. Differs from `Normalize` in two ways:
//!
//! - The cut-off is a count, not a fraction.
//! - The stretch is per-channel (matching IM behaviour).

use crate::{
    contrast_stretch::build_lut_from_hist, is_supported_format, ImageFilter, VideoStreamParams,
};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Linear stretch with per-end pixel-count cut-offs.
#[derive(Clone, Copy, Debug)]
pub struct LinearStretch {
    black_pixels: u32,
    white_pixels: u32,
}

impl Default for LinearStretch {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl LinearStretch {
    /// Build a stretch that drops the darkest `black_pixels` and the
    /// brightest `white_pixels` samples to the extremes before linearly
    /// remapping the rest.
    pub fn new(black_pixels: u32, white_pixels: u32) -> Self {
        Self {
            black_pixels,
            white_pixels,
        }
    }
}

impl ImageFilter for LinearStretch {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: LinearStretch does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let total = (w * h) as u32;
        if total == 0 {
            return Ok(input.clone());
        }
        // Convert counts → fractions, reuse ContrastStretch's LUT
        // builder.
        let black_pct = (self.black_pixels as f32 / total as f32).clamp(0.0, 0.5);
        let white_pct = (self.white_pixels as f32 / total as f32).clamp(0.0, 0.5);
        let mut out = input.clone();
        match params.format {
            PixelFormat::Gray8
            | PixelFormat::Yuv420P
            | PixelFormat::Yuv422P
            | PixelFormat::Yuv444P => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let mut hist = [0u32; 256];
                for y in 0..h {
                    for v in &p.data[y * stride..y * stride + w] {
                        hist[*v as usize] += 1;
                    }
                }
                let lut = build_lut_from_hist(&hist, total, black_pct, white_pct);
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgb24 => {
                stretch_packed(
                    &mut out.planes[0].data,
                    w,
                    h,
                    3,
                    total,
                    black_pct,
                    white_pct,
                );
            }
            PixelFormat::Rgba => {
                stretch_packed(
                    &mut out.planes[0].data,
                    w,
                    h,
                    4,
                    total,
                    black_pct,
                    white_pct,
                );
            }
            _ => unreachable!(),
        }
        Ok(out)
    }
}

fn stretch_packed(
    data: &mut [u8],
    w: usize,
    h: usize,
    bpp: usize,
    total: u32,
    black_pct: f32,
    white_pct: f32,
) {
    let stride = w * bpp;
    let channels = if bpp == 4 { 3 } else { bpp };
    let mut luts = [[0u8; 256]; 3];
    for ch in 0..channels {
        let mut hist = [0u32; 256];
        for y in 0..h {
            let row = &data[y * stride..y * stride + bpp * w];
            for x in 0..w {
                hist[row[bpp * x + ch] as usize] += 1;
            }
        }
        luts[ch] = build_lut_from_hist(&hist, total, black_pct, white_pct);
    }
    for y in 0..h {
        let row = &mut data[y * stride..y * stride + bpp * w];
        for x in 0..w {
            for ch in 0..channels {
                row[bpp * x + ch] = luts[ch][row[bpp * x + ch] as usize];
            }
        }
    }
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
    fn zero_counts_match_full_min_max_stretch() {
        let input = gray(8, 1, |x, _| 64 + (x * 16) as u8);
        let out = LinearStretch::new(0, 0)
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
    fn nonzero_counts_burn_extremes() {
        // 10 sample image → cut 1 pixel from each end → still hits 0/255
        // because the lowest / highest samples are themselves the extremes.
        let input = gray(10, 1, |x, _| (x * 25) as u8);
        let out = LinearStretch::new(1, 1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 10,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(*out.planes[0].data.iter().min().unwrap(), 0);
        assert_eq!(*out.planes[0].data.iter().max().unwrap(), 255);
    }

    #[test]
    fn rgba_alpha_passthrough() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| [50u8 + i * 4, 70 + i * 3, 90 + i * 2, 33])
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = LinearStretch::new(0, 0)
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
            assert_eq!(out.planes[0].data[i * 4 + 3], 33);
        }
    }
}
