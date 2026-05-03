//! Linear / directional motion blur.
//!
//! Builds a 1-D Gaussian-weighted kernel of length `2*radius + 1`,
//! then samples each output pixel along a line rotated by
//! `angle_degrees` from horizontal. Classical ImageMagick behaviour
//! for `-motion-blur RxS+A`.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, gaussian_kernel, plane_dims};
use crate::{is_supported_format, ImageFilter, Planes, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Directional motion blur.
#[derive(Clone, Debug)]
pub struct MotionBlur {
    pub radius: u32,
    pub sigma: f32,
    pub angle_degrees: f32,
}

impl Default for MotionBlur {
    fn default() -> Self {
        Self {
            radius: 3,
            sigma: 1.5,
            angle_degrees: 0.0,
        }
    }
}

impl MotionBlur {
    /// Build with explicit radius, sigma, and motion angle (degrees).
    pub fn new(radius: u32, sigma: f32, angle_degrees: f32) -> Self {
        Self {
            radius: radius.max(1),
            sigma: sigma.max(f32::EPSILON),
            angle_degrees,
        }
    }
}

impl ImageFilter for MotionBlur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: MotionBlur does not yet handle {:?}",
                params.format
            )));
        }

        let planes_selector = match params.format {
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => Planes::Luma,
            _ => Planes::All,
        };

        let kernel = gaussian_kernel(self.radius, self.sigma);
        let angle_rad = self.angle_degrees.to_radians();
        let dx = angle_rad.cos();
        let dy = angle_rad.sin();

        let (cx, cy) = chroma_subsampling(params.format);
        let mut out = input.clone();
        for (idx, plane) in out.planes.iter_mut().enumerate() {
            if !planes_selector.matches(idx, input.planes.len()) {
                continue;
            }
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            *plane = motion_blur_plane(
                &input.planes[idx],
                pw as usize,
                ph as usize,
                bpp,
                &kernel,
                dx,
                dy,
            );
        }

        Ok(out)
    }
}

fn motion_blur_plane(
    src: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    kernel: &[f32],
    dx: f32,
    dy: f32,
) -> VideoPlane {
    let row_bytes = w * bpp;
    let radius = (kernel.len() - 1) / 2;
    let mut out = vec![0u8; row_bytes * h];

    let at = |x: f32, y: f32, ch: usize| -> f32 {
        // Nearest-neighbour sample with clamp-to-edge.
        let cx = (x.round() as i32).clamp(0, w as i32 - 1) as usize;
        let cy = (y.round() as i32).clamp(0, h as i32 - 1) as usize;
        src.data[cy * src.stride + cx * bpp + ch] as f32
    };

    for y in 0..h {
        for x in 0..w {
            for ch in 0..bpp {
                let mut acc = 0.0f32;
                for (ki, kv) in kernel.iter().enumerate() {
                    let t = ki as f32 - radius as f32;
                    let sx = x as f32 + t * dx;
                    let sy = y as f32 + t * dy;
                    acc += *kv * at(sx, sy, ch);
                }
                out[y * row_bytes + x * bpp + ch] = acc.round().clamp(0.0, 255.0) as u8;
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
    fn flat_frame_stays_flat() {
        let input = gray(16, 16, |_, _| 123);
        let out = MotionBlur::new(3, 1.5, 30.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 123);
        }
    }

    #[test]
    fn horizontal_impulse_smears_horizontally() {
        // Single bright column at x=8 → horizontal motion blur keeps it
        // in the same row.
        let input = gray(16, 16, |x, _| if x == 8 { 255 } else { 0 });
        let out = MotionBlur::new(3, 1.0, 0.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        // Pixel to the right of the bright column should be lit.
        assert!(out.planes[0].data[8 * 16 + 9] > 0);
        // Pixel on a different row but same column should also carry
        // mass because the source column is vertical (stays on that
        // column at any y).
        let orig_col_other_row = out.planes[0].data[4 * 16 + 8];
        assert!(orig_col_other_row > 0);
    }

    #[test]
    fn angle_90_blurs_vertically() {
        // Bright row at y=8; 90° angle → samples along y axis.
        let input = gray(16, 16, |_, y| if y == 8 { 255 } else { 0 });
        let out = MotionBlur::new(3, 1.0, 90.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        // Pixel above the bright row should be lit.
        assert!(out.planes[0].data[7 * 16 + 8] > 0);
    }

    #[test]
    fn yuv_leaves_chroma_untouched() {
        let y: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let u = vec![77u8; 8 * 8];
        let v = vec![188u8; 8 * 8];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y,
                },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };
        let out = MotionBlur::new(3, 1.5, 45.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[1].data, vec![77u8; 8 * 8]);
        assert_eq!(out.planes[2].data, vec![188u8; 8 * 8]);
    }
}
