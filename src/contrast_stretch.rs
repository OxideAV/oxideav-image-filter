//! Contrast stretch (`-contrast-stretch black%xwhite%`).
//!
//! Like [`Normalize`](crate::Normalize) but the clip percentages are
//! given as **pixel-count fractions** (the ImageMagick convention),
//! not as fractions of the histogram. ImageMagick's
//! `-contrast-stretch 2x1%` says "burn 2% of the darkest pixels to
//! black, blow out 1% of the brightest pixels to white".
//!
//! Clean-room: standard cumulative-histogram cut-off algorithm.
//! Differs from [`Normalize`] in:
//!
//! - `Normalize` defaults to `0.0/0.0` clip; `ContrastStretch` accepts
//!   any percentages and treats `0.0` as "use the actual min/max".
//! - For RGB / RGBA the cut-off endpoints are computed **per channel**
//!   (matching IM behaviour); `Normalize` uses a single luma proxy.
//! - For YUV the stretch is applied to luma only (chroma untouched).

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Contrast stretch with per-end pixel-count percentages.
#[derive(Clone, Copy, Debug)]
pub struct ContrastStretch {
    /// Fraction of darkest pixels to burn to black (0.0..=0.5).
    black_pct: f32,
    /// Fraction of brightest pixels to blow out to white (0.0..=0.5).
    white_pct: f32,
}

impl Default for ContrastStretch {
    fn default() -> Self {
        Self::new(0.02, 0.01)
    }
}

impl ContrastStretch {
    /// Build a stretch with per-end fractional cut-offs. Each is
    /// clamped to `[0.0, 0.5]`.
    pub fn new(black_pct: f32, white_pct: f32) -> Self {
        Self {
            black_pct: black_pct.clamp(0.0, 0.5),
            white_pct: white_pct.clamp(0.0, 0.5),
        }
    }
}

impl ImageFilter for ContrastStretch {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: ContrastStretch does not yet handle {:?}",
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
                let p = &mut out.planes[0];
                let stride = p.stride;
                let lut = stretch_lut_single(&p.data, stride, w, h, self.black_pct, self.white_pct);
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let bpp = 3usize;
                let lut_r = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    0,
                    self.black_pct,
                    self.white_pct,
                );
                let lut_g = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    1,
                    self.black_pct,
                    self.white_pct,
                );
                let lut_b = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    2,
                    self.black_pct,
                    self.white_pct,
                );
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + bpp * w];
                    for x in 0..w {
                        row[bpp * x] = lut_r[row[bpp * x] as usize];
                        row[bpp * x + 1] = lut_g[row[bpp * x + 1] as usize];
                        row[bpp * x + 2] = lut_b[row[bpp * x + 2] as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let bpp = 4usize;
                let lut_r = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    0,
                    self.black_pct,
                    self.white_pct,
                );
                let lut_g = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    1,
                    self.black_pct,
                    self.white_pct,
                );
                let lut_b = stretch_lut_packed(
                    &p.data,
                    stride,
                    w,
                    h,
                    bpp,
                    2,
                    self.black_pct,
                    self.white_pct,
                );
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + bpp * w];
                    for x in 0..w {
                        row[bpp * x] = lut_r[row[bpp * x] as usize];
                        row[bpp * x + 1] = lut_g[row[bpp * x + 1] as usize];
                        row[bpp * x + 2] = lut_b[row[bpp * x + 2] as usize];
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok(out)
    }
}

fn stretch_lut_single(
    data: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    black: f32,
    white: f32,
) -> [u8; 256] {
    let mut hist = [0u32; 256];
    for y in 0..h {
        for v in &data[y * stride..y * stride + w] {
            hist[*v as usize] += 1;
        }
    }
    build_lut_from_hist(&hist, (w * h) as u32, black, white)
}

#[allow(clippy::too_many_arguments)]
fn stretch_lut_packed(
    data: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    bpp: usize,
    ch: usize,
    black: f32,
    white: f32,
) -> [u8; 256] {
    let mut hist = [0u32; 256];
    for y in 0..h {
        let row = &data[y * stride..y * stride + bpp * w];
        for x in 0..w {
            hist[row[bpp * x + ch] as usize] += 1;
        }
    }
    build_lut_from_hist(&hist, (w * h) as u32, black, white)
}

pub(crate) fn build_lut_from_hist(
    hist: &[u32; 256],
    total: u32,
    black: f32,
    white: f32,
) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if total == 0 {
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = i as u8;
        }
        return lut;
    }
    let lo_budget = (total as f32 * black).round() as u32;
    let hi_budget = (total as f32 * white).round() as u32;

    // Walk from low end.
    let mut acc = 0u32;
    let mut lo: u8 = 0;
    for (i, h) in hist.iter().enumerate() {
        acc += *h;
        if acc > lo_budget {
            lo = i as u8;
            break;
        }
    }
    // Walk from high end.
    acc = 0;
    let mut hi: u8 = 255;
    for (i, h) in hist.iter().enumerate().rev() {
        acc += *h;
        if acc > hi_budget {
            hi = i as u8;
            break;
        }
    }
    if hi <= lo {
        if lo < 255 {
            hi = lo.saturating_add(1);
        } else {
            lo = 254;
            hi = 255;
        }
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
    fn zero_zero_stretches_full_range() {
        // With 0/0 cut-offs ContrastStretch is equivalent to a per-channel
        // min/max stretch.
        let input = gray(8, 1, |x, _| 64 + (x * 16) as u8);
        let out = ContrastStretch::new(0.0, 0.0)
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
    fn percentage_burns_outliers() {
        // 10% black-clip of a uniform spread should still produce min 0,
        // max 255 across the image (the clipped extremes are the lowest /
        // highest samples).
        let input = gray(10, 1, |x, _| (x * 25) as u8);
        let out = ContrastStretch::new(0.1, 0.1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 10,
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
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = 64 + (i as u8) * 4;
                [v, v, v, 200]
            })
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = ContrastStretch::default()
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
            assert_eq!(out.planes[0].data[i * 4 + 3], 200);
        }
    }
}
