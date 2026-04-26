//! Desaturate an RGB / RGBA frame to grayscale using Rec. 601 luma
//! weights (`y = 0.299·r + 0.587·g + 0.114·b`).
//!
//! Output can either stay in the input's RGB layout (each channel set
//! to `y`) or collapse to `Gray8` when `output_gray8` is set.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// RGB → grayscale desaturation.
#[derive(Clone, Debug)]
pub struct Grayscale {
    /// Keep the alpha channel when the input is `Rgba`. Has no effect
    /// on `Rgb24` / `Gray8`.
    pub preserve_alpha: bool,
    /// Emit a `Gray8` single-plane frame instead of re-packing the
    /// grayscale value into RGB.
    pub output_gray8: bool,
}

impl Default for Grayscale {
    fn default() -> Self {
        Self {
            preserve_alpha: true,
            output_gray8: false,
        }
    }
}

impl Grayscale {
    /// Build with defaults (`preserve_alpha=true`, `output_gray8=false`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to keep alpha in the output when the input is RGBA.
    pub fn with_preserve_alpha(mut self, preserve: bool) -> Self {
        self.preserve_alpha = preserve;
        self
    }

    /// Set whether to produce a `Gray8` single-plane frame.
    pub fn with_output_gray8(mut self, gray8: bool) -> Self {
        self.output_gray8 = gray8;
        self
    }
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    let y = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
    y.round().clamp(0.0, 255.0) as u8
}

impl ImageFilter for Grayscale {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            PixelFormat::Gray8 => {
                // Already grey — just return a tight-stride copy.
                let w = params.width as usize;
                let h = params.height as usize;
                let src = &input.planes[0];
                let mut out = vec![0u8; w * h];
                for y in 0..h {
                    out[y * w..y * w + w]
                        .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
                }
                return Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: w,
                        data: out,
                    }],
                });
            }
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Grayscale requires Rgb24/Rgba/Gray8, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];

        if self.output_gray8 {
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + w * bpp];
                for x in 0..w {
                    out[y * w + x] = luma(row[x * bpp], row[x * bpp + 1], row[x * bpp + 2]);
                }
            }
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: w,
                    data: out,
                }],
            });
        }

        // Preserve layout — write y,y,y[,a] into each pixel.
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let yv = luma(src_row[x * bpp], src_row[x * bpp + 1], src_row[x * bpp + 2]);
                dst_row[x * bpp] = yv;
                dst_row[x * bpp + 1] = yv;
                dst_row[x * bpp + 2] = yv;
                if has_alpha {
                    dst_row[x * bpp + 3] = if self.preserve_alpha {
                        src_row[x * bpp + 3]
                    } else {
                        255
                    };
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
    use oxideav_core::TimeBase;

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

    #[test]
    fn pure_red_gives_rec601_luma() {
        // 0.299 * 255 = 76.245 -> 76
        let input = rgb(2, 2, |_, _| (255, 0, 0));
        let out = Grayscale::default().apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 2, height: 2 }).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 76);
            assert_eq!(chunk[1], 76);
            assert_eq!(chunk[2], 76);
        }
    }

    #[test]
    fn gray8_mode_returns_single_plane_gray() {
        let input = rgb(2, 2, |_, _| (255, 255, 255));
        let out = Grayscale::new()
            .with_output_gray8(true)
            .apply(&input, VideoStreamParams { format: PixelFormat::Rgb24, width: 2, height: 2 })
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        for b in &out.planes[0].data {
            assert_eq!(*b, 255);
        }
    }

    #[test]
    fn alpha_preserved_by_default() {
        let data = vec![255, 0, 0, 200, 0, 255, 0, 100];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Grayscale::default()
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Rgba, width: 2, height: 1 },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[3], 200);
        assert_eq!(out.planes[0].data[7], 100);
    }

    #[test]
    fn alpha_overwritten_when_preserve_off() {
        let data = vec![255, 0, 0, 50, 0, 255, 0, 1];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Grayscale::new()
            .with_preserve_alpha(false)
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Rgba, width: 2, height: 1 },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[3], 255);
        assert_eq!(out.planes[0].data[7], 255);
    }

    #[test]
    fn gray8_input_is_passed_through() {
        let data: Vec<u8> = (0..16).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: data.clone(),
            }],
        };
        let out = Grayscale::default().apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 }).unwrap();
        assert_eq!(out.planes[0].data, data);
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
        let err = Grayscale::default().apply(&input, VideoStreamParams { format: PixelFormat::Yuv420P, width: 4, height: 4 }).unwrap_err();
        assert!(format!("{err}").contains("Grayscale"));
    }
}
