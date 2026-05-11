//! Red/cyan anaglyph stereo composition (ImageMagick `-stereo`).
//!
//! Combines a left-eye view (`src`) and a right-eye view (`dst`) into a
//! single red/cyan anaglyph image suitable for viewing with red/cyan
//! glasses. The classic recipe (clean-room — every image-processing
//! textbook lists it):
//!
//! ```text
//! out.R = src.R                 // red channel from the left eye
//! out.G = dst.G                 // green channel from the right eye
//! out.B = dst.B                 // blue channel from the right eye
//! out.A = src.A                 // alpha follows the left eye
//! ```
//!
//! Both inputs must share `(format, width, height)`. Output keeps the
//! same shape.
//!
//! # Pixel formats
//!
//! `Rgb24` / `Rgba`. Other formats return [`Error::unsupported`].

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Red/cyan anaglyph stereo composer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stereo;

impl Stereo {
    /// Build a stereo composer. There are no tunable parameters — the
    /// algebra is the canonical red/cyan recipe.
    pub fn new() -> Self {
        Self
    }
}

impl TwoInputImageFilter for Stereo {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Stereo requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if src.planes.is_empty() || dst.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Stereo requires non-empty src and dst planes",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let sp = &src.planes[0];
        let dp = &dst.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let s_row = &sp.data[y * sp.stride..y * sp.stride + row_bytes];
            let d_row = &dp.data[y * dp.stride..y * dp.stride + row_bytes];
            let o_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                // R from left eye, G + B from right eye.
                o_row[base] = s_row[base];
                o_row[base + 1] = d_row[base + 1];
                o_row[base + 2] = d_row[base + 2];
                if has_alpha {
                    o_row[base + 3] = s_row[base + 3];
                }
            }
        }
        Ok(VideoFrame {
            pts: src.pts.or(dst.pts),
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
    fn picks_red_from_src_and_green_blue_from_dst() {
        let src = rgb(2, 2, |_, _| [200, 10, 20]);
        let dst = rgb(2, 2, |_, _| [30, 150, 240]);
        let out = Stereo::new().apply_two(&src, &dst, p_rgb(2, 2)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[200, 150, 240]);
        }
    }

    #[test]
    fn identical_inputs_are_passthrough() {
        let frame = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 77]);
        let out = Stereo::new()
            .apply_two(&frame, &frame, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn alpha_follows_src_for_rgba() {
        // 2x2 RGBA → row stride = 2 * 4 = 8.
        let s_data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 100]).collect();
        let d_data: Vec<u8> = (0..4).flat_map(|_| [0u8, 255, 255, 200]).collect();
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: s_data,
            }],
        };
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: d_data,
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 2,
            height: 2,
        };
        let out = Stereo::new().apply_two(&src, &dst, p).unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[255, 255, 255, 100]);
        }
    }

    #[test]
    fn rejects_unsupported_format() {
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let err = Stereo::new()
            .apply_two(
                &f,
                &f,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Stereo"));
    }

    #[test]
    fn shape_preserved() {
        let src = rgb(3, 5, |_, _| [10, 20, 30]);
        let dst = rgb(3, 5, |_, _| [200, 100, 50]);
        let out = Stereo::new().apply_two(&src, &dst, p_rgb(3, 5)).unwrap();
        assert_eq!(out.planes[0].stride, 9);
        assert_eq!(out.planes[0].data.len(), 45);
    }
}
