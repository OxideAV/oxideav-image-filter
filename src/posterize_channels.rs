//! Per-channel posterisation — independent quantisation counts for
//! each of `R`, `G`, `B` (and optionally `A`).
//!
//! The existing [`Posterize`](crate::Posterize) collapses every channel
//! to the same level count. This variant takes one count per channel
//! so a caller can, say, leave R / G near-full but heavily quantise B
//! for a stylised look.
//!
//! Clean-room formula:
//!
//! ```text
//! bucket = floor(v * N / 256)        // in [0, N-1]
//! v'     = bucket * 255 / (N - 1)    // remap to full range
//! ```
//!
//! …applied separately for each channel with its own `N`.
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Alpha is quantised only if
//! `alpha_levels` is set; otherwise it passes through. YUV is not
//! supported because per-RGB-channel quantisation has no luma/chroma
//! correspondence.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Per-channel posterise.
#[derive(Clone, Copy, Debug)]
pub struct PosterizeChannels {
    pub r_levels: u32,
    pub g_levels: u32,
    pub b_levels: u32,
    /// If `Some`, quantise alpha to that many levels; otherwise alpha
    /// passes through untouched (RGBA only).
    pub alpha_levels: Option<u32>,
}

impl Default for PosterizeChannels {
    fn default() -> Self {
        Self::new(4, 4, 4)
    }
}

impl PosterizeChannels {
    pub fn new(r_levels: u32, g_levels: u32, b_levels: u32) -> Self {
        Self {
            r_levels: r_levels.max(2),
            g_levels: g_levels.max(2),
            b_levels: b_levels.max(2),
            alpha_levels: None,
        }
    }

    pub fn with_alpha_levels(mut self, alpha_levels: u32) -> Self {
        self.alpha_levels = Some(alpha_levels.max(2));
        self
    }
}

impl ImageFilter for PosterizeChannels {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, false),
            PixelFormat::Rgb24 => (3, false),
            PixelFormat::Rgba => (4, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: PosterizeChannels requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let lut_r = build_lut(self.r_levels);
        let lut_g = build_lut(self.g_levels);
        let lut_b = build_lut(self.b_levels);
        let lut_a = self.alpha_levels.map(build_lut);

        let w = params.width as usize;
        let h = params.height as usize;
        let row = w * bpp;
        let sp = &input.planes[0];
        let mut data = vec![0u8; row * h];
        for y in 0..h {
            let src_row = &sp.data[y * sp.stride..y * sp.stride + row];
            let dst_row = &mut data[y * row..(y + 1) * row];
            for x in 0..w {
                let base = x * bpp;
                if bpp == 1 {
                    // Gray8: use the R-channel LUT.
                    dst_row[base] = lut_r[src_row[base] as usize];
                } else {
                    dst_row[base] = lut_r[src_row[base] as usize];
                    dst_row[base + 1] = lut_g[src_row[base + 1] as usize];
                    dst_row[base + 2] = lut_b[src_row[base + 2] as usize];
                    if has_alpha {
                        dst_row[base + 3] = match &lut_a {
                            Some(la) => la[src_row[base + 3] as usize],
                            None => src_row[base + 3],
                        };
                    }
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride: row, data }],
        })
    }
}

fn build_lut(levels: u32) -> [u8; 256] {
    let n = levels.max(2);
    let mut lut = [0u8; 256];
    let nf = n as f32;
    let step = 255.0 / (nf - 1.0);
    for (i, slot) in lut.iter_mut().enumerate() {
        let bucket = ((i as f32) * nf / 256.0).floor().min(nf - 1.0);
        *slot = (bucket * step).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = f(x, y);
                data.extend_from_slice(&[r, g, b]);
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
    fn two_levels_collapses_each_channel_to_binary() {
        // 2-level posterise maps any v to either 0 or 255.
        let input = rgb(4, 1, |x, _| (x as u8 * 60, x as u8 * 60, x as u8 * 60));
        let out = PosterizeChannels::new(2, 2, 2)
            .apply(&input, p_rgb(4, 1))
            .unwrap();
        for c in 0..3 {
            for x in 0..4 {
                let v = out.planes[0].data[x * 3 + c];
                assert!(v == 0 || v == 255, "channel {c} x {x} = {v}");
            }
        }
    }

    #[test]
    fn distinct_per_channel_counts() {
        // B channel collapses to 2; R channel keeps 8 levels.
        let input = rgb(8, 1, |x, _| (x as u8 * 32, x as u8 * 32, x as u8 * 32));
        let out = PosterizeChannels::new(8, 8, 2)
            .apply(&input, p_rgb(8, 1))
            .unwrap();
        for x in 0..8 {
            let b = out.planes[0].data[x * 3 + 2];
            assert!(b == 0 || b == 255, "B channel at {x} = {b}");
        }
        // R channel can have more than two distinct values.
        let mut r_set = std::collections::BTreeSet::new();
        for x in 0..8 {
            r_set.insert(out.planes[0].data[x * 3]);
        }
        assert!(
            r_set.len() > 2,
            "expected >2 distinct R values, got {r_set:?}"
        );
    }

    #[test]
    fn rgba_alpha_passes_when_not_quantised() {
        let data: Vec<u8> = (0..4).flat_map(|i| [60u8, 60, 60, 20 + i * 30]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = PosterizeChannels::new(4, 4, 4)
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
            let a = out.planes[0].data[i * 4 + 3];
            assert_eq!(a, 20 + (i as u8) * 30, "alpha at pixel {i}");
        }
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
        let err = PosterizeChannels::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("PosterizeChannels"));
    }
}
