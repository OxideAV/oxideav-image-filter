//! 3×3 emboss convolution producing a relief-style output.
//!
//! Kernel (clean-room):
//! ```text
//! -2 -1  0
//! -1  1  1
//!  0  1  2
//! ```
//! Each output pixel is the kernel sum plus a `+128` bias clamped to
//! `0..=255`. A uniform-colour input therefore produces the neutral
//! bias value everywhere.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, plane_dims};
use crate::{is_supported, ImageFilter, Planes};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Relief / "emboss" 3×3 convolution.
#[derive(Clone, Debug)]
pub struct Emboss {
    /// Reserved — the current implementation is fixed at 3×3; this
    /// field lets us grow to larger kernels later without an API break.
    pub radius: u32,
}

impl Default for Emboss {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl Emboss {
    /// Build with an explicit radius (only `1` is supported today).
    pub fn new(radius: u32) -> Self {
        Self {
            radius: radius.max(1),
        }
    }
}

const KERNEL: [[i32; 3]; 3] = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]];
const BIAS: i32 = 128;

impl ImageFilter for Emboss {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Emboss does not yet handle {:?}",
                input.format
            )));
        }

        let planes_selector = match input.format {
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => Planes::Luma,
            _ => Planes::All,
        };

        let (cx, cy) = chroma_subsampling(input.format);
        let mut out = input.clone();
        for (idx, plane) in out.planes.iter_mut().enumerate() {
            if !planes_selector.matches(idx, input.planes.len()) {
                continue;
            }
            let (pw, ph) = plane_dims(input.width, input.height, input.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(input.format, idx);
            *plane = emboss_plane(&input.planes[idx], pw as usize, ph as usize, bpp);
        }

        Ok(out)
    }
}

fn emboss_plane(src: &VideoPlane, w: usize, h: usize, bpp: usize) -> VideoPlane {
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    // Clamp-to-edge pixel lookup.
    let at = |x: i32, y: i32, ch: usize| -> i32 {
        let cx = x.clamp(0, w as i32 - 1) as usize;
        let cy = y.clamp(0, h as i32 - 1) as usize;
        src.data[cy * src.stride + cx * bpp + ch] as i32
    };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            for ch in 0..bpp {
                let mut acc: i32 = 0;
                for (ky, row) in KERNEL.iter().enumerate() {
                    for (kx, k) in row.iter().enumerate() {
                        acc += k * at(x + kx as i32 - 1, y + ky as i32 - 1, ch);
                    }
                }
                let v = (acc + BIAS).clamp(0, 255) as u8;
                out[y as usize * row_bytes + x as usize * bpp + ch] = v;
            }
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
    fn flat_input_gives_bias() {
        // Kernel sum = -2-1+0-1+1+1+0+1+2 = 1. For uniform c, acc = c.
        // Output = clamp(c + 128, 0, 255). For c=80, result = 208.
        let input = gray(8, 8, |_, _| 80);
        let out = Emboss::new(1).apply(&input).unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 208);
        }
    }

    #[test]
    fn zero_input_gives_bias_value() {
        let input = gray(8, 8, |_, _| 0);
        let out = Emboss::new(1).apply(&input).unwrap();
        for b in &out.planes[0].data {
            // Kernel sum = 1, so 0 * 1 + 128 = 128.
            assert_eq!(*b, 128);
        }
    }

    #[test]
    fn gradient_produces_relief() {
        // Vertical step dark/bright at x=4 → the +/- kernel highlights
        // one side and shadows the other.
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = Emboss::new(1).apply(&input).unwrap();
        // Interior pixel on bright side away from the edge: kernel sum
        // = 1, so value = 200 + 128 = 328 → clamped to 255.
        assert_eq!(out.planes[0].data[4 * 8 + 7], 255);
        // Interior pixel on dark side: 0*1 + 128 = 128.
        assert_eq!(out.planes[0].data[4 * 8], 128);
    }

    #[test]
    fn yuv_leaves_chroma_at_original() {
        let y: Vec<u8> = vec![50; 16 * 16];
        let u = vec![77u8; 8 * 8];
        let v = vec![188u8; 8 * 8];
        let input = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 16,
            height: 16,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y,
                },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };
        let out = Emboss::new(1).apply(&input).unwrap();
        // Luma should be bias value (50*1 + 128 = 178).
        for b in &out.planes[0].data {
            assert_eq!(*b, 178);
        }
        // Chroma is untouched.
        assert_eq!(out.planes[1].data, vec![77u8; 8 * 8]);
        assert_eq!(out.planes[2].data, vec![188u8; 8 * 8]);
    }
}
