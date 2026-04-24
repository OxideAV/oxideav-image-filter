//! Extract a rectangular sub-region of a frame. ImageMagick's
//! `-crop WxH+X+Y` is the canonical equivalent — the CLI layer parses
//! the geometry string and this filter just takes the resulting
//! `(x, y, w, h)` rectangle. Clean-room from the obvious definition:
//! the output pixel `(i, j)` is the input pixel `(x + i, y + j)`.

use crate::{is_supported, ImageFilter};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Crop a rectangular region out of a frame.
///
/// The rectangle is `width × height` starting at `(x, y)` in the input.
/// If any of `x + width > frame.width` or `y + height > frame.height`
/// (or the rectangle is empty) the filter returns
/// [`Error::invalid`](oxideav_core::Error). For planar YUV the filter
/// rounds chroma-plane crops up so odd-sized crops on 4:2:0 / 4:2:2
/// stay aligned — the luma plane stays exactly `width × height`.
#[derive(Clone, Debug)]
pub struct Crop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Crop {
    /// Build a crop with the given rectangle `(x, y, width, height)`.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl ImageFilter for Crop {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Crop does not yet handle {:?}",
                input.format
            )));
        }

        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: Crop width/height must be > 0",
            ));
        }
        let x_end = self
            .x
            .checked_add(self.width)
            .ok_or_else(|| Error::invalid("oxideav-image-filter: Crop x + width overflows u32"))?;
        let y_end = self
            .y
            .checked_add(self.height)
            .ok_or_else(|| Error::invalid("oxideav-image-filter: Crop y + height overflows u32"))?;
        if x_end > input.width || y_end > input.height {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: Crop ({},{})+{}x{} exceeds frame {}x{}",
                self.x, self.y, self.width, self.height, input.width, input.height
            )));
        }

        let chroma = crate::blur::chroma_subsampling(input.format);
        let rect = (self.x, self.y, self.width, self.height);
        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (plane_x, plane_y, plane_w, plane_h) = plane_rect(input.format, idx, chroma, rect);
            let bpp = crate::blur::bytes_per_plane_pixel(input.format, idx);
            new_planes.push(crop_plane(plane, plane_x, plane_y, plane_w, plane_h, bpp));
        }

        Ok(VideoFrame {
            format: input.format,
            width: self.width,
            height: self.height,
            pts: input.pts,
            time_base: input.time_base,
            planes: new_planes,
        })
    }
}

/// Compute the per-plane crop rectangle. For planar YUV we scale the
/// (x, y, w, h) rectangle by the chroma subsampling factor and round
/// outward so the chroma region fully contains the cropped luma region.
fn plane_rect(
    fmt: oxideav_core::PixelFormat,
    idx: usize,
    chroma: (u32, u32),
    rect: (u32, u32, u32, u32),
) -> (u32, u32, u32, u32) {
    use oxideav_core::PixelFormat;
    let (cx, cy) = chroma;
    let (x, y, w, h) = rect;
    match fmt {
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P if idx > 0 => {
            let px = x / cx;
            let py = y / cy;
            // For odd-aligned crops the chroma rect has to cover the
            // fractional samples on either side. Round end position up.
            let pw = (x + w).div_ceil(cx).saturating_sub(px);
            let ph = (y + h).div_ceil(cy).saturating_sub(py);
            (px, py, pw, ph)
        }
        _ => (x, y, w, h),
    }
}

fn crop_plane(src: &VideoPlane, x: u32, y: u32, w: u32, h: u32, bpp: usize) -> VideoPlane {
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    for row in 0..h {
        let src_off = (y + row) * src.stride + x * bpp;
        let dst_off = row * row_bytes;
        out[dst_off..dst_off + row_bytes].copy_from_slice(&src.data[src_off..src_off + row_bytes]);
    }
    VideoPlane {
        stride: row_bytes,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, TimeBase};

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    #[test]
    fn crop_extracts_dimensions_and_samples() {
        let input = gray(8, 6, |x, y| (x + y * 10) as u8);
        let out = Crop::new(2, 1, 4, 3).apply(&input).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 3);
        assert_eq!(out.planes[0].stride, 4);
        // Top-left of crop should be input(2, 1) = 2 + 10 = 12.
        assert_eq!(out.planes[0].data[0], 12);
        // Bottom-right of crop is input(5, 3) = 5 + 30 = 35.
        assert_eq!(out.planes[0].data[3 * 4 - 1], 35);
    }

    #[test]
    fn crop_rejects_overflow() {
        let input = gray(8, 6, |_, _| 0);
        // Width exceeds frame.
        let err = Crop::new(5, 0, 4, 4).apply(&input).unwrap_err();
        assert!(format!("{err}").contains("exceeds"));
        // Height exceeds frame.
        let err = Crop::new(0, 3, 4, 4).apply(&input).unwrap_err();
        assert!(format!("{err}").contains("exceeds"));
        // Zero size.
        let err = Crop::new(0, 0, 0, 4).apply(&input).unwrap_err();
        assert!(format!("{err}").contains("> 0"));
    }

    #[test]
    fn crop_full_frame_is_identity() {
        let input = gray(5, 4, |x, y| (x * 3 + y * 7) as u8);
        let out = Crop::new(0, 0, 5, 4).apply(&input).unwrap();
        assert_eq!(out.width, 5);
        assert_eq!(out.height, 4);
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn crop_yuv420_scales_chroma() {
        // 8×8 4:2:0 frame; chroma planes are 4×4.
        let y: Vec<u8> = (0..8 * 8).map(|i| (i % 256) as u8).collect();
        let u: Vec<u8> = (0..4 * 4).map(|i| 100 + i as u8).collect();
        let v: Vec<u8> = (0..4 * 4).map(|i| 200 + i as u8).collect();
        let input = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 8,
            height: 8,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![
                VideoPlane { stride: 8, data: y },
                VideoPlane { stride: 4, data: u },
                VideoPlane { stride: 4, data: v },
            ],
        };

        // Even-aligned 4×4 crop starting at (2, 2).
        let out = Crop::new(2, 2, 4, 4).apply(&input).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        // Luma is exactly 4×4.
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data.len(), 16);
        // Chroma planes are 2×2 (4 samples each).
        assert_eq!(out.planes[1].stride, 2);
        assert_eq!(out.planes[1].data.len(), 4);
        assert_eq!(out.planes[2].stride, 2);
        assert_eq!(out.planes[2].data.len(), 4);
    }
}
