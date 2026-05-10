//! Set the output canvas to a fixed size, padding or cropping
//! (`-extent WxH+X+Y`).
//!
//! Unlike `Crop` (which only ever shrinks) or `Resize` (which scales),
//! `Extent` re-windows the input onto a brand-new canvas of arbitrary
//! size. The input is placed at offset `(x, y)` on the destination —
//! pixels outside the source rectangle receive a configurable
//! background colour. Negative offsets translate the input toward the
//! upper-left so the source's right / bottom edge can land inside the
//! window.
//!
//! Clean-room: trivial — for every output pixel `(ox, oy)` we look at
//! `(ox - x, oy - y)` in the input. If that's inside the source bounds
//! we copy the sample, otherwise we emit the background.
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV. Background
//! handling for YUV uses Y = `bg[0]`, U / V = 128 (neutral chroma) so
//! the padding region renders as a desaturated grey of the requested
//! luma — matching the way ImageMagick paints `-extent` on a YUV
//! pipeline.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, plane_dims};
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Re-window the input onto a `(width, height)` canvas with an
/// `(offset_x, offset_y)` shift. Padding outside the source rectangle
/// is filled with [`background`](Self::background).
#[derive(Clone, Debug)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
    pub offset_x: i32,
    pub offset_y: i32,
    /// Background colour as `[R, G, B, A]`. For `Gray8` only `R` is
    /// used; for YUV the Y plane uses `R` and the chroma planes are
    /// painted neutral 128.
    pub background: [u8; 4],
}

impl Extent {
    /// Build with the new canvas size and an offset of `(0, 0)`,
    /// background opaque black.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            offset_x: 0,
            offset_y: 0,
            background: [0, 0, 0, 255],
        }
    }

    /// Override the placement offset of the input on the new canvas.
    pub fn with_offset(mut self, x: i32, y: i32) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }

    /// Override the background fill.
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }
}

impl ImageFilter for Extent {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Extent does not yet handle {:?}",
                params.format
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: Extent width/height must be > 0",
            ));
        }

        let (cx, cy) = chroma_subsampling(params.format);
        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (src_w, src_h) =
                plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let (dst_w, dst_h) = plane_dims(self.width, self.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            // Plane-grid offsets — chroma planes need offsets scaled
            // down to the subsampled grid.
            let (ox, oy) = if idx > 0 {
                (self.offset_x / cx as i32, self.offset_y / cy as i32)
            } else {
                (self.offset_x, self.offset_y)
            };
            let bg = plane_fill(params.format, idx, &self.background);
            new_planes.push(extent_plane(
                plane, src_w, src_h, dst_w, dst_h, bpp, ox, oy, &bg,
            ));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

/// Per-plane fill colour. Returns a `bpp`-sized vec — for RGB / RGBA
/// it's the raw background; for Gray8 it's just R; for planar YUV the
/// luma plane uses R and chroma planes are forced to neutral 128.
fn plane_fill(fmt: PixelFormat, idx: usize, bg: &[u8; 4]) -> Vec<u8> {
    match fmt {
        PixelFormat::Gray8 => vec![bg[0]],
        PixelFormat::Rgb24 => vec![bg[0], bg[1], bg[2]],
        PixelFormat::Rgba => vec![bg[0], bg[1], bg[2], bg[3]],
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            if idx == 0 {
                vec![bg[0]]
            } else {
                vec![128]
            }
        }
        _ => vec![0],
    }
}

#[allow(clippy::too_many_arguments)]
fn extent_plane(
    plane: &VideoPlane,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    bpp: usize,
    ox: i32,
    oy: i32,
    fill: &[u8],
) -> VideoPlane {
    let dst_w = dst_w as usize;
    let dst_h = dst_h as usize;
    let src_w = src_w as usize;
    let src_h = src_h as usize;
    let row_bytes = dst_w * bpp;
    let mut out = vec![0u8; row_bytes * dst_h];

    // Pre-fill the entire canvas with the background. Faster than
    // checking bounds per-pixel and the source-region copy will
    // overwrite the in-bounds samples.
    for px in 0..dst_w {
        let base = px * bpp;
        out[base..base + bpp].copy_from_slice(&fill[..bpp]);
    }
    let template_row_bytes = row_bytes;
    let template = &out[..template_row_bytes].to_vec();
    for row in 1..dst_h {
        let off = row * row_bytes;
        out[off..off + row_bytes].copy_from_slice(template);
    }

    // Compute the rectangle of overlap between the source-at-offset and
    // the destination canvas, then bulk-copy the visible rows.
    let dst_x_start = ox.max(0);
    let dst_y_start = oy.max(0);
    let dst_x_end = (ox + src_w as i32).min(dst_w as i32);
    let dst_y_end = (oy + src_h as i32).min(dst_h as i32);
    if dst_x_end <= dst_x_start || dst_y_end <= dst_y_start {
        // No visible source rectangle — the canvas stays full of bg.
        return VideoPlane {
            stride: row_bytes,
            data: out,
        };
    }

    let copy_w = (dst_x_end - dst_x_start) as usize;
    let copy_h = (dst_y_end - dst_y_start) as usize;
    let src_x_start = (dst_x_start - ox) as usize;
    let src_y_start = (dst_y_start - oy) as usize;
    let copy_bytes = copy_w * bpp;
    for row in 0..copy_h {
        let src_off = (src_y_start + row) * plane.stride + src_x_start * bpp;
        let dst_off = (dst_y_start as usize + row) * row_bytes + dst_x_start as usize * bpp;
        out[dst_off..dst_off + copy_bytes]
            .copy_from_slice(&plane.data[src_off..src_off + copy_bytes]);
    }

    VideoPlane {
        stride: row_bytes,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

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

    fn params_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn same_size_zero_offset_is_identity() {
        let input = gray(4, 3, |x, y| (x + y * 5) as u8);
        let out = Extent::new(4, 3).apply(&input, params_gray(4, 3)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn larger_canvas_pads_with_background() {
        let input = gray(2, 2, |_, _| 99);
        let out = Extent::new(4, 4)
            .with_background([42, 0, 0, 255])
            .apply(&input, params_gray(2, 2))
            .unwrap();
        // 4×4 canvas, source placed at (0, 0). Top-left 2×2 is 99,
        // everything else is 42.
        assert_eq!(out.planes[0].data[0], 99);
        assert_eq!(out.planes[0].data[1], 99);
        assert_eq!(out.planes[0].data[2], 42);
        assert_eq!(out.planes[0].data[3], 42);
        assert_eq!(out.planes[0].data[4 * 4 - 1], 42);
    }

    #[test]
    fn offset_shifts_source_rect() {
        let input = gray(2, 2, |_, _| 99);
        let out = Extent::new(4, 4)
            .with_offset(1, 1)
            .with_background([0, 0, 0, 255])
            .apply(&input, params_gray(2, 2))
            .unwrap();
        // Top-left (0, 0) is now padding.
        assert_eq!(out.planes[0].data[0], 0);
        // Source pixel (0, 0) lands at (1, 1).
        assert_eq!(out.planes[0].data[5], 99);
        assert_eq!(out.planes[0].data[2 * 4 + 2], 99);
        // Bottom-right is padding again.
        assert_eq!(out.planes[0].data[3 * 4 + 3], 0);
    }

    #[test]
    fn negative_offset_clips_source_rect() {
        let input = gray(4, 4, |x, y| (x + y * 10) as u8);
        let out = Extent::new(2, 2)
            .with_offset(-2, -2)
            .with_background([0, 0, 0, 255])
            .apply(&input, params_gray(4, 4))
            .unwrap();
        // Source pixel (2, 2) = 22 lands at (0, 0).
        assert_eq!(out.planes[0].data[0], 22);
        // Source pixel (3, 3) = 33 lands at (1, 1).
        assert_eq!(out.planes[0].data[3], 33);
    }

    #[test]
    fn fully_off_canvas_returns_pure_background() {
        let input = gray(2, 2, |_, _| 99);
        let out = Extent::new(3, 3)
            .with_offset(10, 10)
            .with_background([7, 0, 0, 255])
            .apply(&input, params_gray(2, 2))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 7));
    }

    #[test]
    fn rgba_background_alpha_is_respected() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![10, 20, 30, 40],
            }],
        };
        let out = Extent::new(2, 1)
            .with_background([200, 150, 100, 50])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[..4], [10, 20, 30, 40]);
        assert_eq!(out.planes[0].data[4..], [200, 150, 100, 50]);
    }

    #[test]
    fn yuv420_chroma_uses_neutral_padding() {
        // 2×2 4:2:0 — chroma planes are 1×1.
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 2,
                    data: vec![100, 100, 100, 100],
                },
                VideoPlane {
                    stride: 1,
                    data: vec![200],
                },
                VideoPlane {
                    stride: 1,
                    data: vec![50],
                },
            ],
        };
        let out = Extent::new(4, 4)
            .with_background([10, 0, 0, 255])
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // Y plane: top-left 2×2 stays at 100, padding is 10.
        assert_eq!(out.planes[0].data[0], 100);
        assert_eq!(out.planes[0].data[3], 10);
        // U / V chroma padding lands as neutral 128 outside the source
        // rectangle (which is 1×1 at the corner).
        assert_eq!(out.planes[1].data[0], 200);
        assert!(out.planes[1].data[1..].iter().all(|&v| v == 128));
        assert_eq!(out.planes[2].data[0], 50);
        assert!(out.planes[2].data[1..].iter().all(|&v| v == 128));
    }
}
