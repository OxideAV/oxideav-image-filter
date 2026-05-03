//! Mirror a frame vertically — top row becomes bottom row. ImageMagick
//! calls this `-flip` (flop is the horizontal cousin, see [`Flop`](crate::Flop)).

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Vertical flip — reverses the order of rows in every plane.
///
/// For planar YUV formats each plane is flipped at its own resolution
/// (the 4:2:0 chroma planes are half-height, so they're flipped
/// half-height too). The output uses a tight stride that matches the
/// plane width in bytes.
#[derive(Clone, Debug, Default)]
pub struct Flip;

impl Flip {
    /// Build the filter. `Flip` has no parameters.
    pub fn new() -> Self {
        Self
    }
}

impl ImageFilter for Flip {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Flip does not yet handle {:?}",
                params.format
            )));
        }

        let (cx, cy) = crate::blur::chroma_subsampling(params.format);
        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (pw, ph) =
                crate::blur::plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = crate::blur::bytes_per_plane_pixel(params.format, idx);
            new_planes.push(flip_plane(plane, pw as usize, ph as usize, bpp));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

fn flip_plane(src: &VideoPlane, w: usize, h: usize, bpp: usize) -> VideoPlane {
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    for y in 0..h {
        let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
        let dst_y = h - 1 - y;
        out[dst_y * row_bytes..dst_y * row_bytes + row_bytes].copy_from_slice(src_row);
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

    #[test]
    fn flip_swaps_rows() {
        let input = gray(4, 3, |_, y| (y * 10) as u8);
        let out = Flip::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 3,
                },
            )
            .unwrap();
        // Row 0 (was 0) -> becomes bottom; row 2 (was 20) -> top.
        assert_eq!(&out.planes[0].data[0..4], &[20u8, 20, 20, 20]);
        assert_eq!(&out.planes[0].data[4..8], &[10u8, 10, 10, 10]);
        assert_eq!(&out.planes[0].data[8..12], &[0u8, 0, 0, 0]);
    }

    #[test]
    fn flip_flip_is_identity() {
        let p = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 8,
            height: 7,
        };
        let input = gray(8, 7, |x, y| ((x * 13 + y * 29) % 251) as u8);
        let once = Flip::new().apply(&input, p).unwrap();
        let twice = Flip::new().apply(&once, p).unwrap();
        assert_eq!(twice.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flip_handles_yuv420_planes_separately() {
        let w = 8u32;
        let h = 4u32;
        // Y: row-distinct pattern.
        let y: Vec<u8> = (0..h)
            .flat_map(|row| std::iter::repeat(row as u8 * 10).take(w as usize))
            .collect();
        // U/V: also row-distinct, at half height.
        let u: Vec<u8> = (0..h / 2)
            .flat_map(|row| std::iter::repeat(100 + row as u8).take((w / 2) as usize))
            .collect();
        let v: Vec<u8> = (0..h / 2)
            .flat_map(|row| std::iter::repeat(200 + row as u8).take((w / 2) as usize))
            .collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: u,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: v,
                },
            ],
        };
        let out = Flip::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 8,
                    height: 4,
                },
            )
            .unwrap();
        // Top Y row (was row 3 with value 30) should now be 30.
        assert_eq!(out.planes[0].data[0], 30);
        assert_eq!(out.planes[0].data[(h as usize - 1) * w as usize], 0);
        // U plane: h/2 = 2 rows. Top was 100, bottom 101 → after flip top 101.
        assert_eq!(out.planes[1].data[0], 101);
        assert_eq!(out.planes[1].data[(w / 2) as usize], 100);
    }
}
