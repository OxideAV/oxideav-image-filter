//! Flat solid-coloured border around the image — distinct from the
//! existing [`Frame`](crate::frame::Frame), which paints a decorative
//! 3-D bevel.
//!
//! Adds a uniform border of `(left, top, right, bottom)` pixels filled
//! with a configurable `[R, G, B, A]` colour. The output canvas grows by
//! `(left + right, top + bottom)`. ImageMagick analogue: `-bordercolor
//! C -border WxH`.
//!
//! Operates on `Rgb24`, `Rgba`, and `Gray8`. YUV inputs return
//! `Unsupported` because painting a single RGB colour onto a YUV canvas
//! requires a real colourspace conversion (the planar-fill convention
//! used elsewhere in the crate paints chroma neutral 128, which loses
//! the user's hue).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Solid-colour border filter.
#[derive(Clone, Copy, Debug)]
pub struct BorderedFrame {
    /// Pixels of border on the left edge.
    pub left: u32,
    /// Pixels of border on the top edge.
    pub top: u32,
    /// Pixels of border on the right edge.
    pub right: u32,
    /// Pixels of border on the bottom edge.
    pub bottom: u32,
    /// Border fill colour as `[R, G, B, A]`. For `Gray8` only `R` is used.
    pub colour: [u8; 4],
}

impl Default for BorderedFrame {
    fn default() -> Self {
        Self {
            left: 4,
            top: 4,
            right: 4,
            bottom: 4,
            colour: [0, 0, 0, 255],
        }
    }
}

impl BorderedFrame {
    /// Build a uniform `width`-pixel border on all four sides.
    pub fn uniform(width: u32, colour: [u8; 4]) -> Self {
        Self {
            left: width,
            top: width,
            right: width,
            bottom: width,
            colour,
        }
    }

    /// Build with independent per-side widths.
    pub fn sides(left: u32, top: u32, right: u32, bottom: u32, colour: [u8; 4]) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            colour,
        }
    }
}

impl ImageFilter for BorderedFrame {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: BorderedFrame requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: BorderedFrame requires a non-empty input plane",
            ));
        }
        let iw = params.width as usize;
        let ih = params.height as usize;
        let ow = iw + self.left as usize + self.right as usize;
        let oh = ih + self.top as usize + self.bottom as usize;
        let stride = ow * bpp;
        let mut data = vec![0u8; stride * oh];

        // Fill background colour.
        let [r, g, b, a] = self.colour;
        for y in 0..oh {
            let row = &mut data[y * stride..(y + 1) * stride];
            for x in 0..ow {
                let base = x * bpp;
                match bpp {
                    1 => row[base] = r,
                    3 => {
                        row[base] = r;
                        row[base + 1] = g;
                        row[base + 2] = b;
                    }
                    4 => {
                        row[base] = r;
                        row[base + 1] = g;
                        row[base + 2] = b;
                        row[base + 3] = a;
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Copy input into the interior (alpha — when present — comes from src).
        let src = &input.planes[0];
        let src_row = iw * bpp;
        let x_off = self.left as usize;
        let y_off = self.top as usize;
        for y in 0..ih {
            let s = &src.data[y * src.stride..y * src.stride + src_row];
            let dst_y = y + y_off;
            let d = &mut data[dst_y * stride + x_off * bpp..dst_y * stride + x_off * bpp + src_row];
            d.copy_from_slice(s);
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data }],
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
    fn uniform_border_grows_canvas() {
        let input = rgb(4, 4, |_, _| [200, 0, 0]);
        let f = BorderedFrame::uniform(2, [0, 128, 0, 255]);
        let out = f.apply(&input, p_rgb(4, 4)).unwrap();
        // 4 + 2*2 = 8 pixels wide.
        assert_eq!(out.planes[0].stride, 8 * 3);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn interior_preserves_input_pixels() {
        let input = rgb(2, 2, |x, y| [(x * 50) as u8, (y * 50) as u8, 200]);
        let f = BorderedFrame::uniform(1, [0, 0, 0, 255]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        // Border is at y=0, y=3, x=0, x=3.
        // Interior pixel (1, 1) in output corresponds to (0, 0) of input.
        let stride = 4 * 3;
        // Output (1, 1) = input (0, 0) = [0, 0, 200]
        let base = stride + 3;
        assert_eq!(out.planes[0].data[base], 0);
        assert_eq!(out.planes[0].data[base + 1], 0);
        assert_eq!(out.planes[0].data[base + 2], 200);
        // Output (2, 1) = input (1, 0) = [50, 0, 200]
        let base = stride + 6;
        assert_eq!(out.planes[0].data[base], 50);
    }

    #[test]
    fn border_uses_specified_colour() {
        let input = rgb(2, 2, |_, _| [200, 0, 0]);
        let f = BorderedFrame::uniform(1, [10, 20, 30, 255]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        // Top-left corner of border, output (0, 0) = border colour.
        assert_eq!(out.planes[0].data[0], 10);
        assert_eq!(out.planes[0].data[1], 20);
        assert_eq!(out.planes[0].data[2], 30);
    }

    #[test]
    fn asymmetric_sides() {
        let input = rgb(2, 2, |_, _| [0, 0, 0]);
        let f = BorderedFrame::sides(3, 1, 5, 7, [255, 255, 255, 255]);
        let out = f.apply(&input, p_rgb(2, 2)).unwrap();
        // Width: 2 + 3 + 5 = 10; height: 2 + 1 + 7 = 10.
        assert_eq!(out.planes[0].stride, 10 * 3);
        assert_eq!(out.planes[0].data.len(), 10 * 10 * 3);
    }

    #[test]
    fn alpha_passes_through_in_interior() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 200, 50, 99]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let f = BorderedFrame::uniform(1, [0, 0, 0, 255]);
        let out = f
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // Interior (1, 1) input alpha is 99.
        let stride = 4 * 4;
        let base = stride + 4;
        assert_eq!(out.planes[0].data[base + 3], 99);
        // Border (0, 0) alpha is 255.
        assert_eq!(out.planes[0].data[3], 255);
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
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
        let err = BorderedFrame::uniform(1, [0, 0, 0, 255])
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("BorderedFrame"));
    }
}
