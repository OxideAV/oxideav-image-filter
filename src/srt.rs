//! Scale / Rotate / Translate composite warp (`-distort SRT`).
//!
//! Mirrors ImageMagick's most-common shorthand for combined geometric
//! transforms. The full argument signature is
//!
//! ```text
//!   -distort SRT  "ox,oy  sx[,sy]  angle  tx,ty"
//! ```
//!
//! where:
//!
//! - `(ox, oy)` is the centre of rotation / scaling (defaults to image
//!   centre).
//! - `(sx, sy)` are the X / Y scale factors. A single value is applied
//!   isotropically.
//! - `angle` is the rotation in degrees (positive = clockwise, matching
//!   ImageMagick and our [`Rotate`](crate::Rotate) convention).
//! - `(tx, ty)` is an additional post-rotation translation (the new
//!   coordinates of the original `(ox, oy)` after the warp).
//!
//! Internally we collapse the four steps into a single 2×3 affine
//! matrix and run the existing [`Affine`](crate::Affine) implementation.
//! The matrix is built as
//!
//! ```text
//!   T(tx, ty) · R(θ) · S(sx, sy) · T(-ox, -oy)
//! ```
//!
//! so that the source point `(ox, oy)` ends up at `(tx, ty)` after the
//! transform — exactly what IM's `-distort SRT` does.

use crate::affine::Affine;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Scale / Rotate / Translate warp wrapping [`Affine`].
#[derive(Clone, Copy, Debug)]
pub struct Srt {
    pub origin: (f32, f32),
    pub scale: (f32, f32),
    pub angle_degrees: f32,
    pub translate: (f32, f32),
    pub background: [u8; 4],
    pub output_size: Option<(u32, u32)>,
}

impl Srt {
    /// Build an SRT warp. `origin` defaults are not assumed here — the
    /// factory layer fills in the image centre when the JSON omits it.
    pub fn new(
        origin: (f32, f32),
        scale: (f32, f32),
        angle_degrees: f32,
        translate: (f32, f32),
    ) -> Self {
        Self {
            origin,
            scale,
            angle_degrees,
            translate,
            background: [0, 0, 0, 255],
            output_size: None,
        }
    }

    /// Convenience: pure rotation around `origin` by `angle` degrees.
    pub fn rotation(origin: (f32, f32), angle_degrees: f32) -> Self {
        Self::new(origin, (1.0, 1.0), angle_degrees, origin)
    }

    /// Override the background fill (RGBA).
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }

    /// Override the output canvas size (`(0, 0)` ⇒ disable).
    pub fn with_output_size(mut self, w: u32, h: u32) -> Self {
        self.output_size = if w == 0 || h == 0 { None } else { Some((w, h)) };
        self
    }

    /// Collapse the four-step transform into a single 2×3 affine
    /// matrix in `(sx, ry, rx, sy, tx, ty)` order.
    ///
    /// ```text
    ///   M = T(tx, ty) · R(θ) · S(sx, sy) · T(-ox, -oy)
    /// ```
    pub fn to_affine_matrix(&self) -> [f32; 6] {
        let (ox, oy) = self.origin;
        let (sx, sy) = self.scale;
        let (tx, ty) = self.translate;
        let theta = self.angle_degrees.to_radians();
        let (sin_t, cos_t) = theta.sin_cos();

        let a = cos_t * sx;
        let b = -sin_t * sy;
        let c = sin_t * sx;
        let d = cos_t * sy;
        let pre_tx = -(a * ox + b * oy);
        let pre_ty = -(c * ox + d * oy);
        let final_tx = pre_tx + tx;
        let final_ty = pre_ty + ty;
        [a, b, c, d, final_tx, final_ty]
    }

    /// Build the underlying [`Affine`] filter — visible for tests and
    /// for callers who want to compose further.
    pub fn to_affine(&self) -> Affine {
        let mut a = Affine::new(self.to_affine_matrix()).with_background(self.background);
        if let Some((w, h)) = self.output_size {
            a = a.with_output_size(w, h);
        }
        a
    }
}

impl ImageFilter for Srt {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        self.to_affine().apply(input, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = f(x, y);
                data.extend_from_slice(&[r, g, b]);
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
    fn identity_srt_passes_through() {
        let input = rgb(4, 4, |x, y| ((x * 32) as u8, (y * 32) as u8, 0));
        let f = Srt::new((1.5, 1.5), (1.0, 1.0), 0.0, (1.5, 1.5));
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn pure_rotation_180_inverts_corners() {
        let input = rgb(3, 3, |x, y| ((x + y * 3) as u8 * 30, 0, 0));
        let f = Srt::rotation((1.0, 1.0), 180.0);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 3,
                    height: 3,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data[0], 240);
        let br = (2 * 3 + 2) * 3;
        assert_eq!(out.planes[0].data[br], 0);
    }

    #[test]
    fn scale_two_zooms_in() {
        let input = rgb(3, 3, |x, y| ((x * 60) as u8, (y * 60) as u8, 0));
        let f =
            Srt::new((1.0, 1.0), (2.0, 2.0), 0.0, (1.0, 1.0)).with_background([255, 255, 255, 255]);
        let out = f
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 3,
                    height: 3,
                },
            )
            .unwrap();
        let centre = (3 + 1) * 3;
        assert_eq!(out.planes[0].data[centre], 60);
        assert_eq!(out.planes[0].data[centre + 1], 60);
    }

    #[test]
    fn matrix_decomposes_correctly() {
        let f = Srt::new((0.0, 0.0), (1.0, 1.0), 90.0, (0.0, 0.0));
        let m = f.to_affine_matrix();
        assert!(m[0].abs() < 1e-6, "sx_coef = {}", m[0]);
        assert!((m[1] - -1.0).abs() < 1e-6, "ry_coef = {}", m[1]);
        assert!((m[2] - 1.0).abs() < 1e-6, "rx_coef = {}", m[2]);
        assert!(m[3].abs() < 1e-6, "sy_coef = {}", m[3]);
        assert!(m[4].abs() < 1e-6 && m[5].abs() < 1e-6);
    }

    #[test]
    fn translation_after_rotation_preserves_origin() {
        let f = Srt::new((10.0, 20.0), (2.0, 0.5), 30.0, (100.0, 50.0));
        let m = f.to_affine_matrix();
        let dx = m[0] * 10.0 + m[1] * 20.0 + m[4];
        let dy = m[2] * 10.0 + m[3] * 20.0 + m[5];
        assert!((dx - 100.0).abs() < 1e-3, "dx = {dx}");
        assert!((dy - 50.0).abs() < 1e-3, "dy = {dy}");
    }
}
