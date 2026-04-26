//! Binary threshold: every sample at or above `threshold` becomes 255,
//! every sample below becomes 0.
//!
//! Clean-room implementation from the standard threshold definition;
//! parameter syntax mirrors ImageMagick's `-threshold N` flag (the CLI
//! is responsible for mapping percent syntax to a `u8`).

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Binary threshold.
///
/// - `Gray8` / `Rgb24` / `Rgba`: each colour channel is compared against
///   `threshold` and snapped to 0 or 255. On `Rgba` the alpha channel is
///   preserved unchanged.
/// - `Yuv*P`: the Y plane is thresholded; chroma planes are set to the
///   neutral value 128 so the result is an achromatic black-and-white
///   image (zeroing would introduce a green/magenta cast in YUV).
#[derive(Clone, Copy, Debug)]
pub struct Threshold {
    threshold: u8,
}

impl Default for Threshold {
    fn default() -> Self {
        Self::new(128)
    }
}

impl Threshold {
    /// Build a binariser with the given cut-off (0..=255).
    pub fn new(threshold: u8) -> Self {
        Self { threshold }
    }
}

impl ImageFilter for Threshold {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Threshold does not yet handle {:?}",
                params.format
            )));
        }
        let t = self.threshold;
        let mut out = input.clone();
        let w = params.width as usize;
        let h = params.height as usize;

        match params.format {
            PixelFormat::Gray8 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = if *v >= t { 255 } else { 0 };
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 3 * w];
                    for x in 0..w {
                        for ch in 0..3 {
                            let v = &mut row[3 * x + ch];
                            *v = if *v >= t { 255 } else { 0 };
                        }
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        for ch in 0..3 {
                            let v = &mut row[4 * x + ch];
                            *v = if *v >= t { 255 } else { 0 };
                        }
                        // alpha (index 3) preserved
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let y_plane = &mut out.planes[0];
                let stride = y_plane.stride;
                for y in 0..h {
                    for v in &mut y_plane.data[y * stride..y * stride + w] {
                        *v = if *v >= t { 255 } else { 0 };
                    }
                }
                for p in &mut out.planes[1..] {
                    p.data.fill(128);
                }
            }
            _ => unreachable!("is_supported gate already rejected this format"),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{TimeBase, VideoPlane};

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
    fn threshold_gray_snaps_to_black_or_white() {
        let input = gray(4, 4, |x, y| (x * 32 + y * 32) as u8);
        let out = Threshold::new(100).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 }).unwrap();
        for (i, v) in out.planes[0].data.iter().enumerate() {
            let orig = input.planes[0].data[i];
            let expected = if orig >= 100 { 255 } else { 0 };
            assert_eq!(*v, expected, "idx {i} orig {orig}");
        }
    }

    #[test]
    fn threshold_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|i| [50u8, 150, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Threshold::new(128).apply(&input, VideoStreamParams { format: PixelFormat::Rgba, width: 4, height: 4 }).unwrap();
        for i in 0..16 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[0], 0);
            assert_eq!(px[1], 255);
            assert_eq!(px[2], 255);
            assert_eq!(px[3], i as u8, "alpha preserved");
        }
    }

    #[test]
    fn threshold_yuv_sets_chroma_neutral() {
        let y: Vec<u8> = (0..16).map(|i| i as u8 * 16).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane {
                    stride: 2,
                    data: vec![200u8; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![50u8; 4],
                },
            ],
        };
        let out = Threshold::new(64).apply(&input, VideoStreamParams { format: PixelFormat::Yuv420P, width: 4, height: 4 }).unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255);
        }
        for p in &out.planes[1..] {
            for v in &p.data {
                assert_eq!(*v, 128);
            }
        }
    }
}
