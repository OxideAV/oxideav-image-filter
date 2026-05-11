//! Block-average mosaic ("pixelate"): collapse each `N×N` tile to its
//! mean colour.
//!
//! Visually identical to ImageMagick's `-scale W/N x H/N -scale W x H`
//! two-step (downscale → nearest-neighbour upscale) but implemented
//! as a single pass — for each output pixel we compute the floor-tile
//! coordinates `(tx, ty) = (x / block, y / block)`, average every
//! source pixel inside the tile, and stamp the mean back into every
//! pixel of that tile. The output shape matches the input shape so the
//! filter is stateless and chainable with other shape-preserving
//! filters.
//!
//! Distinct from [`Quantize`](crate::quantize::Quantize) which is a
//! purely colour-axis operation (channel rounding to N levels). Here
//! every channel is preserved bit-by-bit; what we coarsen is the
//! *spatial* axis.
//!
//! Operates on `Gray8` / `Rgb24` / `Rgba`. YUV inputs go through plane
//! 0 only (the luma plane) and the chroma planes are passed through
//! unchanged — pixelating chroma at the wrong subsampled grid would
//! look worse than not pixelating it at all, and the visible effect
//! comes from the luma blockiness.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Spatial block-average mosaic.
#[derive(Clone, Copy, Debug)]
pub struct Pixelate {
    /// Tile size in pixels. `block <= 1` is a no-op pass-through.
    pub block: u32,
}

impl Default for Pixelate {
    fn default() -> Self {
        Self { block: 8 }
    }
}

impl Pixelate {
    /// Build a pixelate filter with explicit `block` size in pixels.
    pub fn new(block: u32) -> Self {
        Self { block }
    }
}

fn pixelate_plane(src: &VideoPlane, w: usize, h: usize, bpp: usize, block: usize) -> Vec<u8> {
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    if block <= 1 {
        for y in 0..h {
            let s = &src.data[y * src.stride..y * src.stride + row_bytes];
            let d = &mut out[y * row_bytes..(y + 1) * row_bytes];
            d.copy_from_slice(s);
        }
        return out;
    }
    let mut sums = vec![0u32; bpp];
    let mut ty = 0;
    while ty < h {
        let mut tx = 0;
        while tx < w {
            for s in sums.iter_mut() {
                *s = 0;
            }
            let y_end = (ty + block).min(h);
            let x_end = (tx + block).min(w);
            let count = ((y_end - ty) * (x_end - tx)) as u32;
            for yy in ty..y_end {
                let row = &src.data[yy * src.stride..yy * src.stride + row_bytes];
                for xx in tx..x_end {
                    let base = xx * bpp;
                    for c in 0..bpp {
                        sums[c] += row[base + c] as u32;
                    }
                }
            }
            let mut mean = [0u8; 4];
            for c in 0..bpp {
                mean[c] = ((sums[c] + count / 2) / count).min(255) as u8;
            }
            for yy in ty..y_end {
                let row = &mut out[yy * row_bytes..yy * row_bytes + row_bytes];
                for xx in tx..x_end {
                    let base = xx * bpp;
                    row[base..base + bpp].copy_from_slice(&mean[..bpp]);
                }
            }
            tx += block;
        }
        ty += block;
    }
    out
}

impl ImageFilter for Pixelate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let block = self.block.max(1) as usize;
        match params.format {
            PixelFormat::Gray8 => {
                if input.planes.is_empty() {
                    return Err(Error::invalid(
                        "oxideav-image-filter: Pixelate requires a non-empty Gray8 plane",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let row_bytes = w;
                let data = pixelate_plane(&input.planes[0], w, h, 1, block);
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: row_bytes,
                        data,
                    }],
                })
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if params.format == PixelFormat::Rgb24 {
                    3
                } else {
                    4
                };
                if input.planes.is_empty() {
                    return Err(Error::invalid(
                        "oxideav-image-filter: Pixelate requires a non-empty packed RGB plane",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let row_bytes = w * bpp;
                let data = pixelate_plane(&input.planes[0], w, h, bpp, block);
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: row_bytes,
                        data,
                    }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                // Pixelate the Y plane; pass chroma through.
                if input.planes.len() < 3 {
                    return Err(Error::invalid(
                        "oxideav-image-filter: Pixelate on planar YUV requires 3 planes",
                    ));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let y_data = pixelate_plane(&input.planes[0], w, h, 1, block);
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![
                        VideoPlane {
                            stride: w,
                            data: y_data,
                        },
                        input.planes[1].clone(),
                        input.planes[2].clone(),
                    ],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Pixelate does not yet handle {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

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

    #[test]
    fn block_one_is_passthrough() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = Pixelate::new(1).apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn constant_input_constant_output() {
        let input = rgb(8, 8, |_, _| [128, 64, 32]);
        let out = Pixelate::new(4).apply(&input, p_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[128, 64, 32]);
        }
    }

    #[test]
    fn block_two_yields_pairs_of_identical_pixels() {
        // 4×2 with each row a horizontal gradient. With block = 2, the
        // top-left 2×2 should all share the same mean.
        let input = rgb(4, 2, |x, _| [(x * 60) as u8, (x * 30) as u8, 100]);
        let out = Pixelate::new(2).apply(&input, p_rgb(4, 2)).unwrap();
        // Pixel (0,0) and (1,0) and (0,1) and (1,1) should all match.
        let p00 = &out.planes[0].data[0..3];
        let p10 = &out.planes[0].data[3..6];
        let p01 = &out.planes[0].data[12..15];
        let p11 = &out.planes[0].data[15..18];
        assert_eq!(p00, p10);
        assert_eq!(p00, p01);
        assert_eq!(p00, p11);
        // And the mean of R values across the 2×2 region is (0+60)/2 = 30
        // (both rows have the same gradient).
        assert_eq!(p00[0], 30);
    }

    #[test]
    fn shape_preserved() {
        let input = rgb(5, 3, |_, _| [10, 20, 30]);
        let out = Pixelate::new(2).apply(&input, p_rgb(5, 3)).unwrap();
        assert_eq!(out.planes[0].stride, 15);
        assert_eq!(out.planes[0].data.len(), 45);
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
        let err = Pixelate::new(2)
            .apply(
                &f,
                VideoStreamParams {
                    format: PixelFormat::Yuyv422,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Pixelate"));
    }
}
