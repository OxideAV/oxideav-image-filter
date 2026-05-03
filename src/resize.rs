//! Rescale a frame to arbitrary dimensions.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// How samples are reconstructed during rescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest-neighbour — fast, blocky. Good for pixel art.
    Nearest,
    /// Bilinear — smooth, default for natural images.
    #[default]
    Bilinear,
}

/// Rescale to `target_width × target_height`.
#[derive(Clone, Debug)]
pub struct Resize {
    target_width: u32,
    target_height: u32,
    interp: Interpolation,
}

impl Resize {
    /// Create a resize filter targeting the given output dimensions.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width: target_width.max(1),
            target_height: target_height.max(1),
            interp: Interpolation::default(),
        }
    }

    /// Choose the interpolation kernel.
    pub fn with_interpolation(mut self, interp: Interpolation) -> Self {
        self.interp = interp;
        self
    }
}

impl ImageFilter for Resize {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Resize does not yet handle {:?}",
                params.format
            )));
        }

        let (cx, cy) = crate::blur::chroma_subsampling(params.format);
        let new_w = self.target_width;
        let new_h = self.target_height;

        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (src_w, src_h) =
                crate::blur::plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let (dst_w, dst_h) = crate::blur::plane_dims(new_w, new_h, params.format, idx, cx, cy);
            let bpp = crate::blur::bytes_per_plane_pixel(params.format, idx);
            new_planes.push(resize_plane(
                plane,
                src_w,
                src_h,
                dst_w,
                dst_h,
                bpp,
                self.interp,
            ));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

fn resize_plane(
    src: &VideoPlane,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    bpp: usize,
    interp: Interpolation,
) -> VideoPlane {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut out = vec![0u8; dw * dh * bpp];

    // Scale factors (source units per destination unit). Use a half-pixel
    // offset so the output samples centre on input pixels — prevents the
    // top-left bias that integer-math resize often shows.
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for y in 0..dh {
        let sy = (y as f32 + 0.5) * scale_y - 0.5;
        for x in 0..dw {
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            match interp {
                Interpolation::Nearest => {
                    let nx = sx.round().clamp(0.0, (sw - 1) as f32) as usize;
                    let ny = sy.round().clamp(0.0, (sh - 1) as f32) as usize;
                    let src_row_off = ny * src.stride + nx * bpp;
                    let dst_off = y * dw * bpp + x * bpp;
                    out[dst_off..dst_off + bpp]
                        .copy_from_slice(&src.data[src_row_off..src_row_off + bpp]);
                }
                Interpolation::Bilinear => {
                    let x0 = sx.floor().clamp(0.0, (sw - 1) as f32) as usize;
                    let y0 = sy.floor().clamp(0.0, (sh - 1) as f32) as usize;
                    let x1 = (x0 + 1).min(sw - 1);
                    let y1 = (y0 + 1).min(sh - 1);
                    let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                    let fy = (sy - y0 as f32).clamp(0.0, 1.0);

                    for ch in 0..bpp {
                        let p00 = src.data[y0 * src.stride + x0 * bpp + ch] as f32;
                        let p10 = src.data[y0 * src.stride + x1 * bpp + ch] as f32;
                        let p01 = src.data[y1 * src.stride + x0 * bpp + ch] as f32;
                        let p11 = src.data[y1 * src.stride + x1 * bpp + ch] as f32;
                        let v = p00 * (1.0 - fx) * (1.0 - fy)
                            + p10 * fx * (1.0 - fy)
                            + p01 * (1.0 - fx) * fy
                            + p11 * fx * fy;
                        out[y * dw * bpp + x * bpp + ch] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    VideoPlane {
        stride: dw * bpp,
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
    fn upscale_nearest_duplicates() {
        let input = gray(2, 2, |x, y| (y * 2 + x + 1) as u8 * 50); // 50, 100, 150, 200
        let out = Resize::new(4, 4)
            .with_interpolation(Interpolation::Nearest)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // Every 2×2 block should carry one source value.
        let v_tl = out.planes[0].data[0];
        let v_tr = out.planes[0].data[3];
        let v_bl = out.planes[0].data[4 * 3];
        let v_br = out.planes[0].data[4 * 3 + 3];
        assert!([50u8, 100, 150, 200].contains(&v_tl));
        assert!([50u8, 100, 150, 200].contains(&v_tr));
        assert!([50u8, 100, 150, 200].contains(&v_bl));
        assert!([50u8, 100, 150, 200].contains(&v_br));
    }

    #[test]
    fn downscale_bilinear_averages() {
        // 4×4 flat at 100 → 2×2 must also be flat at 100.
        let input = gray(4, 4, |_, _| 100);
        let out = Resize::new(2, 2)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 100);
        }
    }

    #[test]
    fn yuv420_resize_preserves_plane_layout() {
        let _w = 16u32;
        let _h = 16u32;
        let y: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let u = vec![77u8; 8 * 8];
        let v = vec![128u8; 8 * 8];
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
        let out = Resize::new(8, 8)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 3);
        // Y plane is 8×8 after resize, stride tight at 8.
        assert_eq!(out.planes[0].stride, 8);
        assert_eq!(out.planes[0].data.len(), 8 * 8);
        // Chroma planes are 4×4 after 4:2:0 subsampling of 8×8.
        assert_eq!(out.planes[1].stride, 4);
        assert_eq!(out.planes[1].data.len(), 4 * 4);
        assert_eq!(out.planes[2].data.len(), 4 * 4);
        // Flat chroma stays flat.
        for b in &out.planes[1].data {
            assert_eq!(*b, 77);
        }
        for b in &out.planes[2].data {
            assert_eq!(*b, 128);
        }
    }
}
