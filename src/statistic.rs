//! Rolling-window per-pixel statistic (`-statistic Op WxH`).
//!
//! For each output pixel and each colour channel, take a statistic
//! (median / minimum / maximum / mean) of the rectangular window
//! centred on the pixel. Border samples clamp to the nearest in-bounds
//! coordinate (the same edge-replication policy as
//! [`Despeckle`](crate::Despeckle)).
//!
//! The `Median` operator generalises the same logic as
//! [`Despeckle`](crate::Despeckle) to non-square windows and unifies
//! it with min/max/mean. Alpha (RGBA) and chroma (YUV) pass through
//! unchanged.
//!
//! # Pixel formats
//!
//! - `Gray8`, `Rgb24`: every channel is processed.
//! - `Rgba`: only R/G/B; alpha is preserved.
//! - `Yuv*P`: only the Y plane; chroma is preserved (chroma stats would
//!   smear hue across edges in a way that's almost never what callers
//!   want; convert to RGB first if needed).

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Which statistic to compute over the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticOp {
    /// Median sample of the window (robust to impulsive noise).
    Median,
    /// Minimum sample of the window (morphological erosion on greyscale).
    Min,
    /// Maximum sample of the window (morphological dilation on greyscale).
    Max,
    /// Mean (rounded to nearest integer).
    Mean,
}

impl StatisticOp {
    /// Public name used by the JSON registry.
    pub fn name(self) -> &'static str {
        match self {
            StatisticOp::Median => "median",
            StatisticOp::Min => "min",
            StatisticOp::Max => "max",
            StatisticOp::Mean => "mean",
        }
    }

    /// Parse the short operator name.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "median" => StatisticOp::Median,
            "min" | "minimum" => StatisticOp::Min,
            "max" | "maximum" => StatisticOp::Max,
            "mean" | "average" => StatisticOp::Mean,
            _ => return None,
        })
    }
}

/// Rolling-window statistic filter.
#[derive(Clone, Copy, Debug)]
pub struct Statistic {
    pub op: StatisticOp,
    /// Window width in pixels (must be ≥ 1; even values still work but
    /// the window is asymmetric — the right column extends `width/2`,
    /// the left `(width-1)/2`).
    pub width: u32,
    /// Window height in pixels (same semantics as `width`).
    pub height: u32,
}

impl Statistic {
    /// Build a windowed statistic with the given operator and window size.
    pub fn new(op: StatisticOp, width: u32, height: u32) -> Self {
        Self { op, width, height }
    }
}

impl ImageFilter for Statistic {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Statistic does not yet handle {:?}",
                params.format
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: Statistic window must have non-zero width and height",
            ));
        }

        let w = params.width as usize;
        let h = params.height as usize;

        let (bpp, channels): (usize, usize) = match params.format {
            PixelFormat::Gray8 => (1, 1),
            PixelFormat::Rgb24 => (3, 3),
            PixelFormat::Rgba => (4, 3),
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => (1, 1),
            _ => unreachable!(),
        };

        let mut out = input.clone();
        let src_plane = input.planes[0].clone();
        let row_bytes = w * bpp;
        let dst = &mut out.planes[0];
        let dst_data = &mut dst.data;
        let dst_stride = dst.stride;

        // 1×1 window ⇒ identity (each plane is copied through).
        if self.width == 1 && self.height == 1 {
            for y in 0..h {
                let src_row =
                    &src_plane.data[y * src_plane.stride..y * src_plane.stride + row_bytes];
                let dst_row = &mut dst_data[y * dst_stride..y * dst_stride + row_bytes];
                dst_row.copy_from_slice(src_row);
            }
            return Ok(out);
        }

        let half_x_left = ((self.width - 1) / 2) as i32;
        let half_x_right = (self.width / 2) as i32;
        let half_y_up = ((self.height - 1) / 2) as i32;
        let half_y_down = (self.height / 2) as i32;
        let window_n = (self.width * self.height) as usize;

        let mut scratch: Vec<u8> = Vec::with_capacity(window_n);

        for y in 0..h {
            let dst_row = &mut dst_data[y * dst_stride..y * dst_stride + row_bytes];
            for x in 0..w {
                let base = x * bpp;
                for ch in 0..channels {
                    scratch.clear();
                    for dy in -half_y_up..=half_y_down {
                        let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let sy_off = sy * src_plane.stride;
                        for dx in -half_x_left..=half_x_right {
                            let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            scratch.push(src_plane.data[sy_off + sx * bpp + ch]);
                        }
                    }
                    dst_row[base + ch] = combine(self.op, &mut scratch);
                }
            }
        }

        Ok(out)
    }
}

fn combine(op: StatisticOp, samples: &mut [u8]) -> u8 {
    match op {
        StatisticOp::Median => {
            samples.sort_unstable();
            samples[samples.len() / 2]
        }
        StatisticOp::Min => *samples.iter().min().expect("non-empty window"),
        StatisticOp::Max => *samples.iter().max().expect("non-empty window"),
        StatisticOp::Mean => {
            let sum: u32 = samples.iter().map(|v| *v as u32).sum();
            let n = samples.len() as u32;
            ((sum + n / 2) / n).min(255) as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn pack(data: Vec<u8>, stride: usize) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        }
    }

    fn gray(w: u32, h: u32, vals: &[u8]) -> VideoFrame {
        assert_eq!(vals.len(), (w * h) as usize);
        pack(vals.to_vec(), w as usize)
    }

    fn run(op: StatisticOp, ww: u32, wh: u32, w: u32, h: u32, vals: &[u8]) -> Vec<u8> {
        let input = gray(w, h, vals);
        let out = Statistic::new(op, ww, wh)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: w,
                    height: h,
                },
            )
            .unwrap();
        out.planes[0].data.clone()
    }

    #[test]
    fn one_by_one_is_identity() {
        let v = vec![10, 20, 30, 40];
        assert_eq!(run(StatisticOp::Median, 1, 1, 2, 2, &v), v);
        assert_eq!(run(StatisticOp::Min, 1, 1, 2, 2, &v), v);
        assert_eq!(run(StatisticOp::Max, 1, 1, 2, 2, &v), v);
        assert_eq!(run(StatisticOp::Mean, 1, 1, 2, 2, &v), v);
    }

    #[test]
    fn min_3x3_picks_smallest_neighbour() {
        let v = vec![1, 100, 100, 100, 100, 100, 100, 100, 100];
        let out = run(StatisticOp::Min, 3, 3, 3, 3, &v);
        assert_eq!(out[4], 1);
        assert_eq!(out[0], 1);
        assert_eq!(out[8], 100);
    }

    #[test]
    fn max_3x3_picks_largest_neighbour() {
        let v = vec![100, 100, 100, 100, 100, 100, 100, 100, 200];
        let out = run(StatisticOp::Max, 3, 3, 3, 3, &v);
        assert_eq!(out[4], 200);
        assert_eq!(out[0], 100);
        assert_eq!(out[8], 200);
    }

    #[test]
    fn median_3x3_robust_to_single_spike() {
        let v = vec![100, 100, 100, 100, 200, 100, 100, 100, 100];
        let out = run(StatisticOp::Median, 3, 3, 3, 3, &v);
        assert_eq!(out[4], 100);
    }

    #[test]
    fn mean_3x3_average_window() {
        let v = vec![0, 100, 0, 100, 100, 100, 0, 100, 0];
        let out = run(StatisticOp::Mean, 3, 3, 3, 3, &v);
        assert!((out[4] as i16 - 56).abs() <= 1);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let mut data: Vec<u8> = Vec::new();
        for i in 0..4u8 {
            data.extend_from_slice(&[100, 100, 100, i]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Statistic::new(StatisticOp::Min, 3, 3)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for i in 0..4 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[3], i as u8, "alpha preserved");
        }
    }

    #[test]
    fn yuv_chroma_passthrough() {
        let y: Vec<u8> = vec![10, 200, 200, 10];
        let cb = vec![60u8; 1];
        let cr = vec![240u8; 1];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 2, data: y },
                VideoPlane {
                    stride: 1,
                    data: cb.clone(),
                },
                VideoPlane {
                    stride: 1,
                    data: cr.clone(),
                },
            ],
        };
        let out = Statistic::new(StatisticOp::Max, 3, 3)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // 3×3 window covers the entire 2×2 image at every position
        // (border-clamped), so every output Y is the global max = 200.
        assert_eq!(out.planes[0].data, vec![200, 200, 200, 200]);
        assert_eq!(out.planes[1].data, cb);
        assert_eq!(out.planes[2].data, cr);
    }

    #[test]
    fn name_round_trip() {
        for op in [
            StatisticOp::Median,
            StatisticOp::Min,
            StatisticOp::Max,
            StatisticOp::Mean,
        ] {
            assert_eq!(StatisticOp::from_name(op.name()), Some(op));
        }
        assert_eq!(StatisticOp::from_name("minimum"), Some(StatisticOp::Min));
        assert_eq!(StatisticOp::from_name("maximum"), Some(StatisticOp::Max));
        assert_eq!(StatisticOp::from_name("average"), Some(StatisticOp::Mean));
        assert_eq!(StatisticOp::from_name("nope"), None);
    }

    #[test]
    fn rejects_zero_window() {
        let input = gray(2, 2, &[0; 4]);
        let err = Statistic::new(StatisticOp::Median, 0, 3)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("non-zero"), "{err}");
    }
}
