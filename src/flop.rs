//! Mirror a frame horizontally — left column becomes right column.
//! ImageMagick calls this `-flop` (the vertical cousin is `Flip`).

use crate::{is_supported, ImageFilter};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Horizontal flop — reverses the order of pixels within every row.
///
/// For planar YUV formats each plane is flopped at its own resolution
/// (4:2:0 chroma planes at half-width, etc.). Packed formats (`Rgb24`,
/// `Rgba`) move whole pixels — every channel of a pixel stays together.
/// The output uses a tight stride that matches the plane width in bytes.
#[derive(Clone, Debug, Default)]
pub struct Flop;

impl Flop {
    /// Build the filter. `Flop` has no parameters.
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for Flop {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Flop does not yet handle {:?}",
                input.format
            )));
        }

        let (cx, cy) = crate::blur::chroma_subsampling(input.format);
        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (pw, ph) =
                crate::blur::plane_dims(input.width, input.height, input.format, idx, cx, cy);
            let bpp = crate::blur::bytes_per_plane_pixel(input.format, idx);
            new_planes.push(flop_plane(plane, pw as usize, ph as usize, bpp));
        }

        Ok(VideoFrame {
            format: input.format,
            width: input.width,
            height: input.height,
            pts: input.pts,
            time_base: input.time_base,
            planes: new_planes,
        })
    }
}

fn flop_plane(src: &VideoPlane, w: usize, h: usize, bpp: usize) -> VideoPlane {
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    for y in 0..h {
        let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
        let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..w {
            let dst_x = w - 1 - x;
            dst_row[dst_x * bpp..dst_x * bpp + bpp]
                .copy_from_slice(&src_row[x * bpp..x * bpp + bpp]);
        }
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
    fn flop_reverses_columns() {
        let input = gray(4, 2, |x, _| (x * 10) as u8);
        let out = Flop::new().apply(&input).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 2);
        // Row 0: [0, 10, 20, 30] -> [30, 20, 10, 0].
        assert_eq!(&out.planes[0].data[0..4], &[30u8, 20, 10, 0]);
        assert_eq!(&out.planes[0].data[4..8], &[30u8, 20, 10, 0]);
    }

    #[test]
    fn flop_flop_is_identity() {
        let input = gray(7, 5, |x, y| ((x * 19 + y * 31) % 251) as u8);
        let once = Flop::new().apply(&input).unwrap();
        let twice = Flop::new().apply(&once).unwrap();
        assert_eq!(twice.width, input.width);
        assert_eq!(twice.height, input.height);
        assert_eq!(twice.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flop_moves_rgb_pixels_as_a_whole() {
        // 2 pixels wide: (10, 20, 30) | (200, 210, 220)
        let data = vec![10u8, 20, 30, 200, 210, 220];
        let input = VideoFrame {
            format: PixelFormat::Rgb24,
            width: 2,
            height: 1,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane { stride: 6, data }],
        };
        let out = Flop::new().apply(&input).unwrap();
        // After flop: (200, 210, 220) | (10, 20, 30) — channel order inside a
        // pixel is preserved, only pixel positions swap.
        assert_eq!(out.planes[0].data, vec![200, 210, 220, 10, 20, 30]);
    }
}
