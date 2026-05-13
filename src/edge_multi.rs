//! Multi-kernel edge detector — pick between Sobel / Prewitt / Scharr /
//! Roberts at runtime instead of the fixed-Sobel [`Edge`](crate::Edge).
//!
//! Each kernel encodes a different trade-off between noise resistance,
//! orientation accuracy, and computational simplicity:
//!
//! - **Sobel** (3×3, weights `±1 ±2 ±1`) — the classical baseline,
//!   matches [`Edge`].
//! - **Prewitt** (3×3, weights `±1 ±1 ±1`) — slightly noisier than
//!   Sobel but no central-pixel bias.
//! - **Scharr** (3×3, weights `±3 ±10 ±3`) — better rotational symmetry
//!   than Sobel; the standard choice for accurate gradient orientation
//!   estimation.
//! - **Roberts** (2×2, cross kernel) — the cheapest classic operator;
//!   highly localised but very noise-sensitive.
//!
//! All four are clean-room transcriptions from Sonka, Hlavac & Boyle
//! ("Image Processing, Analysis, and Machine Vision", 4th ed.) — no
//! library reference.
//!
//! Output is always a single-plane `Gray8` frame; the magnitude is
//! computed as `|Gx| + |Gy|` (fast L1 approximation to L2), clamped to
//! `[0, 255]`. Same input format coverage as [`Edge`] (Gray8 / Rgb24 /
//! Rgba / planar YUV — RGB is reduced to a luma proxy first).

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Choice of gradient kernel for [`EdgeDetect`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EdgeKernel {
    /// 3×3 Sobel — the classical default (matches [`Edge`](crate::Edge)).
    #[default]
    Sobel,
    /// 3×3 Prewitt — flat ±1 weights, no central-pixel boost.
    Prewitt,
    /// 3×3 Scharr — `±3 ±10 ±3` for improved rotational symmetry.
    Scharr,
    /// 2×2 Roberts cross — cheap diagonal-pair difference.
    Roberts,
}

impl EdgeKernel {
    /// Parse the short kernel name (`"sobel"`, `"prewitt"`, …).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "sobel" => EdgeKernel::Sobel,
            "prewitt" => EdgeKernel::Prewitt,
            "scharr" => EdgeKernel::Scharr,
            "roberts" | "roberts-cross" | "roberts_cross" => EdgeKernel::Roberts,
            _ => return None,
        })
    }
}

/// Multi-kernel edge-magnitude filter.
#[derive(Clone, Debug, Default)]
pub struct EdgeDetect {
    pub kernel: EdgeKernel,
}

impl EdgeDetect {
    pub fn new(kernel: EdgeKernel) -> Self {
        Self { kernel }
    }
}

impl ImageFilter for EdgeDetect {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: EdgeDetect does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma(input, params)?;
        let out = match self.kernel {
            EdgeKernel::Sobel => sobel(&luma, w, h),
            EdgeKernel::Prewitt => prewitt(&luma, w, h),
            EdgeKernel::Scharr => scharr(&luma, w, h),
            EdgeKernel::Roberts => roberts(&luma, w, h),
        };
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

fn build_luma(f: &VideoFrame, params: VideoStreamParams) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];
    let src = &f.planes[0];
    match params.format {
        PixelFormat::Gray8 | PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        PixelFormat::Rgb24 => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 3 * w];
                for x in 0..w {
                    out[y * w + x] =
                        ((row[3 * x] as u16 + 2 * row[3 * x + 1] as u16 + row[3 * x + 2] as u16)
                            / 4) as u8;
                }
            }
        }
        PixelFormat::Rgba => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                for x in 0..w {
                    out[y * w + x] =
                        ((row[4 * x] as u16 + 2 * row[4 * x + 1] as u16 + row[4 * x + 2] as u16)
                            / 4) as u8;
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "EdgeDetect: cannot derive luma from {other:?}"
            )));
        }
    }
    Ok(out)
}

fn px(luma: &[u8], w: usize, h: usize, x: i32, y: i32) -> i32 {
    let cx = x.clamp(0, w as i32 - 1) as usize;
    let cy = y.clamp(0, h as i32 - 1) as usize;
    luma[cy * w + cx] as i32
}

fn sobel(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 2 || h < 2 {
        return out;
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = |a, b| px(luma, w, h, a, b);
            let gx = -p(x - 1, y - 1) - 2 * p(x - 1, y) - p(x - 1, y + 1)
                + p(x + 1, y - 1)
                + 2 * p(x + 1, y)
                + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)
                + p(x - 1, y + 1)
                + 2 * p(x, y + 1)
                + p(x + 1, y + 1);
            out[y as usize * w + x as usize] =
                (gx.unsigned_abs() + gy.unsigned_abs()).min(255) as u8;
        }
    }
    out
}

fn prewitt(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 2 || h < 2 {
        return out;
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = |a, b| px(luma, w, h, a, b);
            let gx = -p(x - 1, y - 1) - p(x - 1, y) - p(x - 1, y + 1)
                + p(x + 1, y - 1)
                + p(x + 1, y)
                + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1) - p(x, y - 1) - p(x + 1, y - 1)
                + p(x - 1, y + 1)
                + p(x, y + 1)
                + p(x + 1, y + 1);
            out[y as usize * w + x as usize] =
                (gx.unsigned_abs() + gy.unsigned_abs()).min(255) as u8;
        }
    }
    out
}

fn scharr(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 2 || h < 2 {
        return out;
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = |a, b| px(luma, w, h, a, b);
            let gx = -3 * p(x - 1, y - 1) - 10 * p(x - 1, y) - 3 * p(x - 1, y + 1)
                + 3 * p(x + 1, y - 1)
                + 10 * p(x + 1, y)
                + 3 * p(x + 1, y + 1);
            let gy = -3 * p(x - 1, y - 1) - 10 * p(x, y - 1) - 3 * p(x + 1, y - 1)
                + 3 * p(x - 1, y + 1)
                + 10 * p(x, y + 1)
                + 3 * p(x + 1, y + 1);
            // Scharr weights are 5× heavier than Sobel — divide by 4 so
            // the magnitude lands in a comparable 0..=255 band.
            let mag = (gx.unsigned_abs() + gy.unsigned_abs()) / 4;
            out[y as usize * w + x as usize] = mag.min(255) as u8;
        }
    }
    out
}

fn roberts(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 2 || h < 2 {
        return out;
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = |a, b| px(luma, w, h, a, b);
            let gx = p(x, y) - p(x + 1, y + 1);
            let gy = p(x + 1, y) - p(x, y + 1);
            out[y as usize * w + x as usize] =
                (gx.unsigned_abs() + gy.unsigned_abs()).min(255) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn kernel_name_round_trip() {
        assert_eq!(EdgeKernel::from_name("sobel"), Some(EdgeKernel::Sobel));
        assert_eq!(EdgeKernel::from_name("Prewitt"), Some(EdgeKernel::Prewitt));
        assert_eq!(EdgeKernel::from_name("scharr"), Some(EdgeKernel::Scharr));
        assert_eq!(EdgeKernel::from_name("roberts"), Some(EdgeKernel::Roberts));
        assert_eq!(
            EdgeKernel::from_name("roberts-cross"),
            Some(EdgeKernel::Roberts)
        );
        assert_eq!(EdgeKernel::from_name("nonsense"), None);
    }

    #[test]
    fn flat_image_has_no_edges_for_any_kernel() {
        let input = gray(8, 8, |_, _| 100);
        for k in [
            EdgeKernel::Sobel,
            EdgeKernel::Prewitt,
            EdgeKernel::Scharr,
            EdgeKernel::Roberts,
        ] {
            let out = EdgeDetect::new(k).apply(&input, p_gray(8, 8)).unwrap();
            for &v in &out.planes[0].data {
                assert_eq!(v, 0, "kernel {k:?} should emit zero edges on a flat input");
            }
        }
    }

    #[test]
    fn vertical_step_lights_up() {
        // Vertical step at x=4: left half 0, right half 255.
        let input = gray(8, 8, |x, _| if x >= 4 { 255 } else { 0 });
        for k in [EdgeKernel::Sobel, EdgeKernel::Prewitt, EdgeKernel::Scharr] {
            let out = EdgeDetect::new(k).apply(&input, p_gray(8, 8)).unwrap();
            // The middle column should have a strong response.
            let mid = out.planes[0].data[4 * 8 + 4];
            assert!(mid > 100, "kernel {k:?}: expected strong edge, got {mid}");
        }
    }

    #[test]
    fn roberts_responds_to_diagonal() {
        // A diagonal step pattern: lower-right quadrant 255, rest 0.
        let input = gray(8, 8, |x, y| if x >= 4 && y >= 4 { 255 } else { 0 });
        let out = EdgeDetect::new(EdgeKernel::Roberts)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        // Look for at least one non-zero sample along the diagonal seam.
        let sum: u32 = out.planes[0].data.iter().map(|&v| v as u32).sum();
        assert!(sum > 0, "Roberts on a diagonal step should produce edges");
    }

    #[test]
    fn output_shape_is_gray8() {
        let input = gray(5, 3, |_, _| 30);
        let out = EdgeDetect::new(EdgeKernel::Sobel)
            .apply(&input, p_gray(5, 3))
            .unwrap();
        assert_eq!(out.planes[0].stride, 5);
        assert_eq!(out.planes[0].data.len(), 15);
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 4,
            height: 4,
        };
        let err = EdgeDetect::new(EdgeKernel::Sobel)
            .apply(&input, p)
            .unwrap_err();
        assert!(format!("{err}").contains("EdgeDetect"));
    }
}
