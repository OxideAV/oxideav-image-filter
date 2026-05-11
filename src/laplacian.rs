//! Laplacian / second-derivative edge filter (`-laplacian`).
//!
//! Convolves the image with a 3×3 discrete Laplacian kernel, returning
//! a `Gray8` "edge" frame. Unlike Sobel ([`Edge`](crate::Edge)) which
//! is a first-derivative gradient magnitude, the Laplacian highlights
//! zero-crossings — fine isotropic edges and corners — which makes it
//! a useful precursor to Canny edge linking and a stand-alone
//! "second-derivative" detail map.
//!
//! Clean-room implementation: the standard 3×3 Laplacian kernel
//! (centre +4, 4-neighbours -1, corners 0). The output is the absolute
//! value of the response, clamped to `[0, 255]`. ImageMagick's
//! `-laplacian` accepts a `radius` parameter that picks among a small
//! family of fixed kernels; we expose the same radius knob with the
//! classic 3×3 stencil at radius 1 (the IM default).
//!
//! # Pixel formats
//!
//! Same conventions as [`Edge`]: any supported input is collapsed to
//! a luma plane first, then the Laplacian is applied. Output is
//! always `Gray8`.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Laplacian (second-derivative) filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Laplacian {
    /// Stencil radius. Currently only `1` is supported (3×3 kernel);
    /// reserved for a future 5×5 / 7×7 family.
    pub radius: u32,
}

impl Laplacian {
    /// Build the default 3×3 Laplacian.
    pub fn new() -> Self {
        Self { radius: 1 }
    }

    /// Pick a different stencil radius. Currently clamped to `1`; the
    /// API is kept open for future kernel families.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }
}

impl ImageFilter for Laplacian {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Laplacian does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let out = laplacian3(&luma, w, h);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

pub(crate) fn build_luma_plane(
    f: &VideoFrame,
    params: VideoStreamParams,
) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];
    match params.format {
        PixelFormat::Gray8 => {
            let src = &f.planes[0];
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        PixelFormat::Rgb24 => {
            let src = &f.planes[0];
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 3 * w];
                for x in 0..w {
                    let r = row[3 * x] as u16;
                    let g = row[3 * x + 1] as u16;
                    let b = row[3 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        PixelFormat::Rgba => {
            let src = &f.planes[0];
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                for x in 0..w {
                    let r = row[4 * x] as u16;
                    let g = row[4 * x + 1] as u16;
                    let b = row[4 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            let src = &f.planes[0];
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        _ => {
            return Err(Error::unsupported(format!(
                "Laplacian: cannot derive luma from {:?}",
                params.format
            )));
        }
    }
    Ok(out)
}

fn laplacian3(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    let px = |x: i32, y: i32| -> i32 {
        let cx = x.clamp(0, w as i32 - 1) as usize;
        let cy = y.clamp(0, h as i32 - 1) as usize;
        luma[cy * w + cx] as i32
    };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            // Standard 3x3 Laplacian:
            //   0 -1  0
            //  -1  4 -1
            //   0 -1  0
            let resp = 4 * px(x, y) - px(x - 1, y) - px(x + 1, y) - px(x, y - 1) - px(x, y + 1);
            out[y as usize * w + x as usize] = resp.unsigned_abs().min(255) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn flat_input_yields_zero() {
        let input = gray(8, 8, |_, _| 100);
        let out = Laplacian::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn single_bright_pixel_lights_up() {
        let input = gray(8, 8, |x, y| if x == 4 && y == 4 { 200 } else { 50 });
        let out = Laplacian::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        // Centre pixel: 4*200 - 4*50 = 600 → clamped to 255.
        assert_eq!(out.planes[0].data[4 * 8 + 4], 255);
        // Adjacent neighbours pick up a strong negative response too:
        // 4*50 - 50 - 50 - 200 - 50 = -150 → 150.
        assert!(out.planes[0].data[4 * 8 + 3] >= 100);
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = Laplacian::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 4);
    }
}
