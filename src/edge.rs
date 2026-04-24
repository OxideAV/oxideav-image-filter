//! Sobel edge-magnitude filter. Returns a `Gray8` frame containing
//! `|Gx| + |Gy|` per pixel (clamped to 0..=255).

use crate::{is_supported, ImageFilter};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Sobel edge detector.
///
/// Computes per-pixel gradient magnitude using the 3×3 Sobel kernels
/// (Gx horizontal, Gy vertical). Output is a `Gray8` single-plane frame
/// sized to match the input. For YUV inputs the filter operates on the
/// luma (plane 0) directly. For `Rgb24` / `Rgba` the filter first
/// computes a luma proxy `Y = (R + 2*G + B) / 4` per pixel and then
/// applies Sobel.
#[derive(Clone, Debug, Default)]
pub struct Edge {
    // reserved for future: configurable radius / kernel size. 3x3 Sobel
    // is the classical baseline; larger kernels (5x5, 7x7) can be added
    // later without API churn.
    _kind: EdgeKind,
}

#[derive(Clone, Copy, Debug, Default)]
enum EdgeKind {
    #[default]
    Sobel3,
}

impl Edge {
    /// Build the default 3×3 Sobel edge detector.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ImageFilter for Edge {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Edge does not yet handle {:?}",
                input.format
            )));
        }

        let w = input.width as usize;
        let h = input.height as usize;
        let luma = build_luma_plane(input)?;

        let out = sobel3(&luma, w, h);

        Ok(VideoFrame {
            format: PixelFormat::Gray8,
            width: input.width,
            height: input.height,
            pts: input.pts,
            time_base: input.time_base,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

/// Build a tight-stride Gray8 buffer from any supported input.
fn build_luma_plane(f: &VideoFrame) -> Result<Vec<u8>, Error> {
    let w = f.width as usize;
    let h = f.height as usize;
    let mut out = vec![0u8; w * h];

    match f.format {
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
                "Edge: cannot derive luma from {:?}",
                f.format
            )));
        }
    }
    Ok(out)
}

fn sobel3(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
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
            let gx = -px(x - 1, y - 1) - 2 * px(x - 1, y) - px(x - 1, y + 1)
                + px(x + 1, y - 1)
                + 2 * px(x + 1, y)
                + px(x + 1, y + 1);
            let gy = -px(x - 1, y - 1) - 2 * px(x, y - 1) - px(x + 1, y - 1)
                + px(x - 1, y + 1)
                + 2 * px(x, y + 1)
                + px(x + 1, y + 1);
            // |Gx| + |Gy| is a fast approximation to sqrt(Gx² + Gy²) —
            // plenty accurate for edge visualisation, and branch-light.
            let mag = gx.unsigned_abs() + gy.unsigned_abs();
            out[y as usize * w + x as usize] = mag.min(255) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

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
    fn flat_input_has_zero_edges() {
        let input = gray(16, 16, |_, _| 120);
        let out = Edge::new().apply(&input).unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 0);
        }
    }

    #[test]
    fn vertical_step_produces_edge() {
        // Bright right half, dark left half → strong vertical edge at x=8.
        let input = gray(16, 16, |x, _| if x < 8 { 20 } else { 220 });
        let out = Edge::new().apply(&input).unwrap();
        // Interior pixel on the edge column must have large response.
        let idx = 8 * 16 + 8;
        assert!(out.planes[0].data[idx] > 200, "edge response weak: {}", out.planes[0].data[idx]);
        // Far from the edge → should be close to zero.
        assert!(out.planes[0].data[8 * 16 + 1] < 20);
        assert!(out.planes[0].data[8 * 16 + 14] < 20);
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..8 * 8)
            .flat_map(|i| {
                let v = (i * 3) as u8;
                [v, v, v]
            })
            .collect();
        let input = VideoFrame {
            format: PixelFormat::Rgb24,
            width: 8,
            height: 8,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane {
                stride: 8 * 3,
                data,
            }],
        };
        let out = Edge::new().apply(&input).unwrap();
        assert_eq!(out.format, PixelFormat::Gray8);
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
        assert_eq!(out.planes.len(), 1);
    }
}
