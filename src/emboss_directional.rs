//! Directional emboss — relief shading with a configurable light azimuth.
//!
//! Builds a 3×3 kernel whose weights are the cosine of the angle between
//! each neighbour offset and a light direction vector. The output is the
//! convolved gradient plus a `+128` bias, so flat areas land at neutral
//! grey while edges facing the light pop bright and edges facing away go
//! dark — the canonical "embossed in metal" look. The angle is
//! measured counter-clockwise from East (0° = light from the right;
//! 90° = light from the top; 180° = left; 270° = bottom).
//!
//! Clean-room derivation from the textbook gradient-shading
//! formulation:
//!
//! ```text
//! kernel[dy, dx] = ((dx, dy) · (cos a, -sin a))         # screen-y inverted
//! out[x, y]      = bias + Σ kernel[dy, dx] · src[x+dx, y+dy] · depth
//! ```
//!
//! Distinct from [`Emboss`](crate::Emboss), which is fixed at the
//! Documented CLI default kernel (south-east light).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Alpha passes through on RGBA. Planar YUV
//! is supported on the luma plane only — chroma is left untouched.
//! Other formats return [`Error::unsupported`].

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Relief filter with configurable light direction.
#[derive(Clone, Copy, Debug)]
pub struct EmbossDirectional {
    /// Light azimuth in degrees (CCW from East).
    pub azimuth_degrees: f32,
    /// Relief depth. `1.0` reproduces the classic look; higher numbers
    /// exaggerate contrast.
    pub depth: f32,
}

impl Default for EmbossDirectional {
    fn default() -> Self {
        Self {
            azimuth_degrees: 315.0, // south-east, matches the legacy Emboss.
            depth: 1.0,
        }
    }
}

impl EmbossDirectional {
    /// Build with a light azimuth (degrees). Depth defaults to `1.0`.
    pub fn new(azimuth_degrees: f32) -> Self {
        Self {
            azimuth_degrees,
            ..Self::default()
        }
    }

    /// Override the relief depth.
    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = depth;
        self
    }
}

impl ImageFilter for EmbossDirectional {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        // 3×3 kernel — weights = (dx, -dy) · (cos a, sin a). The minus
        // sign on dy converts from image / screen y (down-positive) to
        // math y (up-positive).
        let a = self.azimuth_degrees.to_radians();
        let lx = a.cos();
        let ly = a.sin();
        let depth = self.depth;
        let mut kernel = [[0f32; 3]; 3];
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let dot = (dx as f32) * lx + (-(dy as f32)) * ly;
                kernel[(dy + 1) as usize][(dx + 1) as usize] = dot * depth;
            }
        }
        match params.format {
            PixelFormat::Gray8 => emboss_packed(input, params, 1, &kernel),
            PixelFormat::Rgb24 => emboss_packed(input, params, 3, &kernel),
            PixelFormat::Rgba => emboss_packed(input, params, 4, &kernel),
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                emboss_yuv_luma(input, params, &kernel)
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: EmbossDirectional does not yet handle {other:?}"
            ))),
        }
    }
}

fn emboss_packed(
    input: &VideoFrame,
    params: VideoStreamParams,
    bpp: usize,
    kernel: &[[f32; 3]; 3],
) -> Result<VideoFrame, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    if w == 0 || h == 0 {
        return Ok(input.clone());
    }
    let src = &input.planes[0];
    let row_stride = w * bpp;
    let mut data = vec![0u8; row_stride * h];
    // tone channels: 1 for gray, 3 for RGB / RGBA.
    let tone = if bpp == 1 { 1 } else { 3 };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let dst_base = (y as usize) * row_stride + (x as usize) * bpp;
            for c in 0..tone {
                let mut acc = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let xx = (x + dx).clamp(0, w as i32 - 1) as usize;
                        let yy = (y + dy).clamp(0, h as i32 - 1) as usize;
                        let v = src.data[yy * src.stride + xx * bpp + c] as f32;
                        acc += kernel[(dy + 1) as usize][(dx + 1) as usize] * v;
                    }
                }
                let v = (acc + 128.0).round().clamp(0.0, 255.0) as u8;
                data[dst_base + c] = v;
            }
            if bpp == 4 {
                // alpha pass-through
                let yy = y.clamp(0, h as i32 - 1) as usize;
                let xx = x.clamp(0, w as i32 - 1) as usize;
                data[dst_base + 3] = src.data[yy * src.stride + xx * 4 + 3];
            }
        }
    }
    Ok(VideoFrame {
        pts: input.pts,
        planes: vec![VideoPlane {
            stride: row_stride,
            data,
        }],
    })
}

fn emboss_yuv_luma(
    input: &VideoFrame,
    params: VideoStreamParams,
    kernel: &[[f32; 3]; 3],
) -> Result<VideoFrame, Error> {
    if input.planes.len() < 3 {
        return Err(Error::invalid(
            "oxideav-image-filter: EmbossDirectional requires 3 YUV planes",
        ));
    }
    let w = params.width as usize;
    let h = params.height as usize;
    let src = &input.planes[0];
    let mut data = vec![0u8; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y + dy).clamp(0, h as i32 - 1) as usize;
                    let v = src.data[yy * src.stride + xx] as f32;
                    acc += kernel[(dy + 1) as usize][(dx + 1) as usize] * v;
                }
            }
            data[(y as usize) * w + x as usize] = (acc + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let planes = vec![
        VideoPlane { stride: w, data },
        input.planes[1].clone(),
        input.planes[2].clone(),
    ];
    Ok(VideoFrame {
        pts: input.pts,
        planes,
    })
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
    fn flat_input_lands_at_neutral_bias() {
        let input = gray(8, 8, |_, _| 100);
        let out = EmbossDirectional::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 128, "flat region should land at the bias");
        }
    }

    #[test]
    fn vertical_step_lights_with_east_azimuth() {
        // Light from east (az = 0°): a rising step in +x has a +x
        // gradient, so the 3×3 stencil weighted by (dx · 1) integrates
        // to a positive number near the edge → columns either side of
        // the step land ABOVE the bias. Flat regions further away
        // land exactly at the bias.
        let input = gray(16, 16, |x, _| if x < 8 { 0 } else { 200 });
        let out = EmbossDirectional::new(0.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let pre = out.planes[0].data[8 * 16 + 7] as i32;
        let post = out.planes[0].data[8 * 16 + 8] as i32;
        assert!(pre > 128, "pre-edge column should brighten, got {pre}");
        assert!(post > 128, "post-edge column should brighten, got {post}");
        let flat_left = out.planes[0].data[8 * 16 + 1] as i32;
        let flat_right = out.planes[0].data[8 * 16 + 14] as i32;
        assert_eq!(flat_left, 128);
        assert_eq!(flat_right, 128);
    }

    #[test]
    fn opposite_azimuth_flips_polarity() {
        // Light from west (180°) should invert the sign of every kernel
        // weight relative to az=0°. With a +x rising step a flat-region
        // pixel still lands at 128, but the edge response inverts sign:
        // sub-bias instead of super-bias.
        let input = gray(16, 16, |x, _| if x < 8 { 0 } else { 200 });
        let out0 = EmbossDirectional::new(0.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        let out180 = EmbossDirectional::new(180.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        // edge columns
        let a = out0.planes[0].data[8 * 16 + 7] as i32;
        let b = out180.planes[0].data[8 * 16 + 7] as i32;
        // (a - 128) and (b - 128) must have opposite signs (or both
        // saturate symmetrically against 0 / 255).
        let da = a - 128;
        let db = b - 128;
        assert!(
            (da > 0 && db < 0) || (a == 255 && b == 0),
            "polarity didn't flip: {a} vs {b}"
        );
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut data = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8 {
            for x in 0..8 {
                data.extend_from_slice(&[(x * 32) as u8, (y * 32) as u8, 100, 77]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 8,
            height: 8,
        };
        let out = EmbossDirectional::default().apply(&input, p).unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[3], 77, "alpha drifted");
        }
    }

    #[test]
    fn yuv_chroma_passthrough() {
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![100; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![200; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![50; 4],
                },
            ],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let out = EmbossDirectional::default().apply(&input, p).unwrap();
        assert_eq!(out.planes[1].data, vec![200; 4]);
        assert_eq!(out.planes[2].data, vec![50; 4]);
    }

    #[test]
    fn rejects_unsupported() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![0; 4],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 2,
            height: 2,
        };
        let err = EmbossDirectional::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("EmbossDirectional"));
    }
}
