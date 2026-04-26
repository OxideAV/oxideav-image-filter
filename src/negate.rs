//! Pixel-value negation. For RGB/RGBA/Gray inputs each colour channel is
//! inverted (`v' = 255 - v`, alpha untouched). For YUV inputs only the
//! luma channel is inverted — leaving chroma alone preserves hue and
//! saturation, which matches the intuitive "negative image" result for
//! YUV source material.
//!
//! Clean-room implementation from the standard definition of a photo
//! negative; parameter syntax mirrors ImageMagick's `-negate` flag so
//! the CLI layer can forward user input directly.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Invert pixel values ("photo negative").
///
/// - `Gray8`: `v' = 255 - v`.
/// - `Rgb24`: each of R/G/B is inverted independently.
/// - `Rgba`:  R/G/B inverted, alpha preserved.
/// - `Yuv*P`: only the Y plane is inverted; Cb/Cr are left alone so the
///   hue of the result follows the photographic-negative convention
///   rather than a colour-wheel rotation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Negate;

impl Negate {
    /// Build a new `Negate` filter.
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for Negate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Negate does not yet handle {:?}",
                params.format
            )));
        }
        let mut out = input.clone();
        match params.format {
            PixelFormat::Gray8 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let w = params.width as usize;
                let h = params.height as usize;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + w];
                    for v in row.iter_mut() {
                        *v = 255 - *v;
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let w = params.width as usize;
                let h = params.height as usize;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 3 * w];
                    for x in 0..w {
                        row[3 * x] = 255 - row[3 * x];
                        row[3 * x + 1] = 255 - row[3 * x + 1];
                        row[3 * x + 2] = 255 - row[3 * x + 2];
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                let w = params.width as usize;
                let h = params.height as usize;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        row[4 * x] = 255 - row[4 * x];
                        row[4 * x + 1] = 255 - row[4 * x + 1];
                        row[4 * x + 2] = 255 - row[4 * x + 2];
                        // alpha (index 3) preserved.
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let y_plane = &mut out.planes[0];
                let stride = y_plane.stride;
                let w = params.width as usize;
                let h = params.height as usize;
                for y in 0..h {
                    let row = &mut y_plane.data[y * stride..y * stride + w];
                    for v in row.iter_mut() {
                        *v = 255 - *v;
                    }
                }
                // Chroma planes are intentionally left as-is.
            }
            _ => unreachable!("is_supported gate already rejected this format"),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{TimeBase, VideoPlane};

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
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
    fn negate_gray_inverts_every_pixel() {
        let input = gray(4, 4, |x, y| (x * 10 + y) as u8);
        let out = Negate::new().apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 }).unwrap();
        for (i, v) in out.planes[0].data.iter().enumerate() {
            let orig = input.planes[0].data[i];
            assert_eq!(*v, 255 - orig);
        }
    }

    #[test]
    fn negate_twice_is_identity() {
        let p = VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 };
        let input = gray(4, 4, |x, y| ((x + y * 4) * 7) as u8);
        let once = Negate::new().apply(&input, p).unwrap();
        let twice = Negate::new().apply(&once, p).unwrap();
        assert_eq!(twice.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn negate_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..4 * 4)
            .flat_map(|i| [i as u8, (i + 10) as u8, (i + 20) as u8, 200u8])
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4 * 4,
                data,
            }],
        };
        let out = Negate::new()
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Rgba, width: 4, height: 4 },
            )
            .unwrap();
        for x in 0..16 {
            let src = &input.planes[0].data[x * 4..x * 4 + 4];
            let dst = &out.planes[0].data[x * 4..x * 4 + 4];
            assert_eq!(dst[0], 255 - src[0]);
            assert_eq!(dst[1], 255 - src[1]);
            assert_eq!(dst[2], 255 - src[2]);
            assert_eq!(dst[3], 200, "alpha must be preserved");
        }
    }

    #[test]
    fn negate_yuv_touches_only_luma() {
        let y: Vec<u8> = (0..4 * 4).map(|i| i as u8 * 10).collect();
        let u = vec![77u8; 2 * 2];
        let v = vec![128u8; 2 * 2];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: y.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: u.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: v.clone(),
                },
            ],
        };
        let out = Negate::new()
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Yuv420P, width: 4, height: 4 },
            )
            .unwrap();
        for (i, b) in out.planes[0].data.iter().enumerate() {
            assert_eq!(*b, 255 - y[i]);
        }
        assert_eq!(out.planes[1].data, u);
        assert_eq!(out.planes[2].data, v);
    }
}
