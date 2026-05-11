//! Linear colour-matrix transform (`-color-matrix matrix`,
//! `-recolor matrix`).
//!
//! Multiplies every pixel by a 3×3 matrix (with an optional 3×1 offset
//! vector for the equivalent of an affine 3×4 matrix) and clamps the
//! result back into `[0, 255]`. Identical to ImageMagick's
//! `-color-matrix`/`-recolor` family — useful for hue rotations,
//! channel swaps, sepia recipes, and custom colour grading.
//!
//! Clean-room implementation: textbook matrix multiplication, no
//! library borrowed.
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba`: full 3-channel transform; alpha (RGBA) is
//!   preserved.
//! - `Gray8` / `Yuv*P`: returns [`Error::unsupported`] — the operator
//!   is intrinsically chromatic; callers wanting a per-channel scale
//!   on luma should use [`Evaluate`](crate::Evaluate) `Multiply`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// 3×3 colour matrix with an optional 3-vector offset.
///
/// Output:
///
/// ```text
/// R' = clamp(m[0]*R + m[1]*G + m[2]*B + offset[0])
/// G' = clamp(m[3]*R + m[4]*G + m[5]*B + offset[1])
/// B' = clamp(m[6]*R + m[7]*G + m[8]*B + offset[2])
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ColorMatrix {
    /// Row-major 3×3 matrix coefficients.
    pub matrix: [f32; 9],
    /// Per-channel additive offset in the 0..=255 input scale.
    pub offset: [f32; 3],
}

impl Default for ColorMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

impl ColorMatrix {
    /// Identity transform (every pixel passes through unchanged).
    pub fn identity() -> Self {
        Self {
            #[rustfmt::skip]
            matrix: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            offset: [0.0; 3],
        }
    }

    /// Build with a row-major 3×3 matrix.
    pub fn new(matrix: [f32; 9]) -> Self {
        Self {
            matrix,
            offset: [0.0; 3],
        }
    }

    /// Add a 3-element offset (in the 0..=255 input scale).
    pub fn with_offset(mut self, offset: [f32; 3]) -> Self {
        self.offset = offset;
        self
    }
}

impl ImageFilter for ColorMatrix {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ColorMatrix requires Rgb24 / Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        let mut out = input.clone();
        let p = &mut out.planes[0];
        let stride = p.stride;
        let m = &self.matrix;
        let o = &self.offset;
        for y in 0..h {
            let row = &mut p.data[y * stride..y * stride + bpp * w];
            for x in 0..w {
                let r = row[bpp * x] as f32;
                let g = row[bpp * x + 1] as f32;
                let b = row[bpp * x + 2] as f32;
                let r2 = m[0] * r + m[1] * g + m[2] * b + o[0];
                let g2 = m[3] * r + m[4] * g + m[5] * b + o[1];
                let b2 = m[6] * r + m[7] * g + m[8] * b + o[2];
                row[bpp * x] = clamp_u8(r2);
                row[bpp * x + 1] = clamp_u8(g2);
                row[bpp * x + 2] = clamp_u8(b2);
                // alpha untouched (Rgba) / no extra channel (Rgb24)
            }
        }
        Ok(out)
    }
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn rgba(w: u32, h: u32, pat: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&pat(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    fn rgb24(w: u32, h: u32, pat: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&pat(x, y));
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

    #[test]
    fn identity_is_identity() {
        let input = rgba(2, 2, |x, y| [x as u8 * 50, y as u8 * 50, 100, 80]);
        let out = ColorMatrix::identity()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn channel_swap_rgb_to_bgr() {
        // Map (R,G,B) → (B,G,R) by permuting rows.
        #[rustfmt::skip]
        let m = ColorMatrix::new([
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]);
        let input = rgb24(1, 1, |_, _| [10, 20, 30]);
        let out = m
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![30, 20, 10]);
    }

    #[test]
    fn offset_brightens_channels() {
        let m = ColorMatrix::identity().with_offset([20.0, 10.0, 0.0]);
        let input = rgba(1, 1, |_, _| [100, 100, 100, 200]);
        let out = m
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![120, 110, 100, 200]);
    }

    #[test]
    fn clamp_saturates() {
        let m = ColorMatrix::new([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
        let input = rgba(1, 1, |_, _| [200, 200, 200, 50]);
        let out = m
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![255, 255, 255, 50]);
    }

    #[test]
    fn rejects_yuv() {
        let input = rgba(1, 1, |_, _| [0, 0, 0, 0]);
        let err = ColorMatrix::identity()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap_err();
        let _ = err; // returns Error::unsupported
    }
}
