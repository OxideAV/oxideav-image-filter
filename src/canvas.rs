//! Fixed-colour canvas: replace every input pixel with a constant
//! `[R, G, B, A]` colour. Output shape (`format`, `width`, `height`) is
//! taken from the stream parameters — this is effectively `xc:`-style
//! `-canvas`/`-fill colour` from ImageMagick.
//!
//! Clean-room: trivial — for every output sample we emit the configured
//! colour. No input-pixel sampling at all (the input frame is consumed
//! only for its `pts`).
//!
//! Useful as a generator step at the top of a pipeline (a flat backdrop
//! for `composite-over`) or as a wipe / clear stage between montage
//! tiles. Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Constant-colour canvas filter.
#[derive(Clone, Copy, Debug)]
pub struct Canvas {
    /// Fill colour as `[R, G, B, A]`. For `Gray8` only `R` is used; for
    /// YUV the Y plane uses `R` and chroma planes are painted with the
    /// neutral value `128` (matching ImageMagick's behaviour when an
    /// RGB colour is laid down on a YUV pipeline).
    pub colour: [u8; 4],
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            colour: [0, 0, 0, 255],
        }
    }
}

impl Canvas {
    /// Build a canvas filter with the given fill colour.
    pub fn new(colour: [u8; 4]) -> Self {
        Self { colour }
    }
}

impl ImageFilter for Canvas {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        let [r, g, b, a] = self.colour;
        match params.format {
            PixelFormat::Gray8 => {
                let stride = w;
                let data = vec![r; stride * h];
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data }],
                })
            }
            PixelFormat::Rgb24 => {
                let stride = w * 3;
                let mut data = Vec::with_capacity(stride * h);
                for _ in 0..(w * h) {
                    data.extend_from_slice(&[r, g, b]);
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data }],
                })
            }
            PixelFormat::Rgba => {
                let stride = w * 4;
                let mut data = Vec::with_capacity(stride * h);
                for _ in 0..(w * h) {
                    data.extend_from_slice(&[r, g, b, a]);
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let (cw, ch) = match params.format {
                    PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
                    PixelFormat::Yuv422P => (w.div_ceil(2), h),
                    PixelFormat::Yuv444P => (w, h),
                    _ => unreachable!(),
                };
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![
                        VideoPlane {
                            stride: w,
                            data: vec![r; w * h],
                        },
                        VideoPlane {
                            stride: cw,
                            data: vec![128; cw * ch],
                        },
                        VideoPlane {
                            stride: cw,
                            data: vec![128; cw * ch],
                        },
                    ],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Canvas does not yet handle {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_in(w: u32, h: u32, fill: [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&fill);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    #[test]
    fn paints_uniform_colour_rgba() {
        let input = rgba_in(4, 3, [10, 20, 30, 40]);
        let out = Canvas::new([200, 100, 50, 255])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 3,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].stride, 16);
        assert_eq!(out.planes[0].data.len(), 48);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn gray8_uses_r_channel_only() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let out = Canvas::new([99, 200, 50, 255])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 99));
    }

    #[test]
    fn yuv_chroma_is_neutral_128() {
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![0; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 4],
                },
            ],
        };
        let out = Canvas::new([180, 0, 0, 255])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 3);
        assert!(out.planes[0].data.iter().all(|&v| v == 180));
        assert!(out.planes[1].data.iter().all(|&v| v == 128));
        assert!(out.planes[2].data.iter().all(|&v| v == 128));
    }

    #[test]
    fn pts_propagates_from_input() {
        let mut input = rgba_in(2, 2, [0, 0, 0, 0]);
        input.pts = Some(12345);
        let out = Canvas::new([1, 2, 3, 4])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.pts, Some(12345));
    }
}
