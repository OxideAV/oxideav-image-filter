//! Radial-polynomial lens distortion (barrel / pincushion) with
//! optional tangential ("decentering") terms — the full Brown-Conrady
//! lens-distortion model.
//!
//! For each output pixel `(px, py)` compute its normalised offset from
//! the centre `(cx, cy)`:
//!
//! ```text
//!   nx = (px - cx) / r_norm
//!   ny = (py - cy) / r_norm
//!   r² = nx² + ny²
//!   scale = 1 + k1·r² + k2·r⁴            // radial component
//!   dx_t  = 2·p1·nx·ny + p2·(r² + 2·nx²)  // tangential component
//!   dy_t  = p1·(r² + 2·ny²) + 2·p2·nx·ny
//!   sx = cx + (nx · scale + dx_t) · r_norm
//!   sy = cy + (ny · scale + dy_t) · r_norm
//! ```
//!
//! `r_norm` defaults to half the smaller image side, so `r = 1` falls
//! at the closer edge and the corner sits at `r ≈ √2/2 · (long/short)`.
//! The bilinear sample at `(sx, sy)` becomes the output. Background fill
//! (default opaque black) is used for samples that fall outside
//! `[0, w-1] × [0, h-1]`.
//!
//! Convention (matches ImageMagick `-distort barrel`):
//!
//! - **`k1 > 0`** (and small `k2`) — *pincushion*: the centre is sampled
//!   from further out, so the image appears to bulge inward at the
//!   corners.
//! - **`k1 < 0`** — *barrel*: corners are sampled from inside the
//!   image, so the image appears to bulge outward.
//!
//! `k2` is a higher-order correction that mostly affects the corners;
//! pass `0.0` if you don't need it. `p1` and `p2` are the standard
//! tangential decentering coefficients (Brown-Conrady) — they model
//! the asymmetry produced when the lens elements are not perfectly
//! aligned with the sensor. Both default to `0.0` (pure-radial mode);
//! non-zero values shift samples off-axis (a `p1`-only model warps
//! the image vertically near the y-axis, `p2`-only warps it
//! horizontally near the x-axis).
//!
//! Operates on packed `Rgb24` / `Rgba`. Output dimensions match the
//! input (no canvas grow). Use [`Distort::with_background`] to set the
//! out-of-bounds fill colour.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Radial-polynomial lens distortion.
#[derive(Clone, Copy, Debug)]
pub struct Distort {
    /// Quadratic radial coefficient. Negative → barrel; positive →
    /// pincushion.
    pub k1: f32,
    /// Quartic radial coefficient. Pass 0.0 if not needed.
    pub k2: f32,
    /// First tangential (decentering) coefficient — Brown-Conrady
    /// `p1`. Defaults to `0.0` for the pure-radial model.
    pub p1: f32,
    /// Second tangential (decentering) coefficient — Brown-Conrady
    /// `p2`. Defaults to `0.0` for the pure-radial model.
    pub p2: f32,
    /// Optional centre override (in pixels). `None` means image centre.
    pub centre: Option<(f32, f32)>,
    /// Optional normalisation radius (in pixels). `None` means
    /// `min(w, h) / 2`.
    pub r_norm: Option<f32>,
    /// Out-of-bounds fill colour (`[R, G, B, A]`).
    pub background: [u8; 4],
}

impl Distort {
    /// Construct with `k1` and `k2`. Centre and normalisation radius
    /// default to image-derived values; tangential coefficients
    /// default to `0.0` (pure-radial mode); background defaults to
    /// opaque black.
    pub fn new(k1: f32, k2: f32) -> Self {
        Self {
            k1,
            k2,
            p1: 0.0,
            p2: 0.0,
            centre: None,
            r_norm: None,
            background: [0, 0, 0, 255],
        }
    }

    /// Convenience: pure barrel (negative quadratic).
    pub fn barrel(strength: f32) -> Self {
        Self::new(-strength.abs(), 0.0)
    }

    /// Convenience: pure pincushion (positive quadratic).
    pub fn pincushion(strength: f32) -> Self {
        Self::new(strength.abs(), 0.0)
    }

    /// Override the lens centre (in pixel coordinates).
    pub fn with_centre(mut self, cx: f32, cy: f32) -> Self {
        self.centre = Some((cx, cy));
        self
    }

    /// Override the normalisation radius (in pixels). Larger values
    /// reduce the apparent strength of `k1` and `k2`.
    pub fn with_r_norm(mut self, r_norm: f32) -> Self {
        self.r_norm = Some(r_norm);
        self
    }

    /// Override the out-of-bounds background fill.
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }

    /// Set the Brown-Conrady tangential ("decentering") coefficients
    /// `p1` and `p2`. Both default to `0.0` (pure-radial Distort);
    /// non-zero values introduce the standard off-axis lens-mis-alignment
    /// component. Pass both as `0.0` to revert to the pure-radial model.
    pub fn with_tangential(mut self, p1: f32, p2: f32) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self
    }
}

impl ImageFilter for Distort {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Distort requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if !self.k1.is_finite()
            || !self.k2.is_finite()
            || !self.p1.is_finite()
            || !self.p2.is_finite()
        {
            return Err(Error::invalid(
                "oxideav-image-filter: Distort coefficients must be finite",
            ));
        }

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        if w == 0 || h == 0 {
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        let (cx, cy) = self
            .centre
            .unwrap_or(((w as f32 - 1.0) * 0.5, (h as f32 - 1.0) * 0.5));
        let r_norm = self
            .r_norm
            .unwrap_or((w.min(h).max(1) as f32) * 0.5)
            .max(f32::EPSILON);
        let inv_r = 1.0 / r_norm;

        for py in 0..h {
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            let dy = (py as f32 - cy) * inv_r;
            for px in 0..w {
                let dx = (px as f32 - cx) * inv_r;
                let r2 = dx * dx + dy * dy;
                let scale = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
                // Brown-Conrady tangential ("decentering") component.
                // When p1 == p2 == 0 these terms vanish and the model
                // collapses back to the pure-radial form — keeping
                // existing zero-tangential outputs bit-exact.
                let dx_t = 2.0 * self.p1 * dx * dy + self.p2 * (r2 + 2.0 * dx * dx);
                let dy_t = self.p1 * (r2 + 2.0 * dy * dy) + 2.0 * self.p2 * dx * dy;
                let sx = cx + (dx * scale + dx_t) * r_norm;
                let sy = cy + (dy * scale + dy_t) * r_norm;
                let base = px * bpp;
                // A tiny epsilon absorbs floating-point round-off at
                // the source-rectangle boundary so an `sx` that
                // computes to -3e-7 for the leftmost column still
                // hits the inside branch (and clamps to 0.0 inside
                // `bilinear_sample_into`) instead of getting flushed
                // to background.
                const EPS: f32 = 1e-4;
                if sx < -EPS
                    || sx > (w as f32 - 1.0) + EPS
                    || sy < -EPS
                    || sy > (h as f32 - 1.0) + EPS
                {
                    write_background(&mut dst_row[base..base + bpp], &self.background, bpp);
                    continue;
                }
                bilinear_sample_into(
                    &src.data,
                    src.stride,
                    w,
                    h,
                    bpp,
                    sx,
                    sy,
                    &mut dst_row[base..base + bpp],
                );
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_bytes,
                data: out,
            }],
        })
    }
}

fn write_background(dst: &mut [u8], bg: &[u8; 4], bpp: usize) {
    for (i, slot) in dst.iter_mut().enumerate().take(bpp) {
        *slot = bg[i.min(3)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn zero_distortion_is_identity() {
        let input = rgb(16, 16, |x, y| (x as u8 * 16, y as u8 * 16, 50));
        let out = Distort::new(0.0, 0.0)
            .apply(&input, params_rgb(16, 16))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_invariant() {
        // Any non-trivial distortion of a flat image should still
        // produce a flat image (every interior sample is the same
        // colour, and the warp keeps every sample on the same colour).
        let input = rgb(16, 16, |_, _| (100, 150, 200));
        let out = Distort::barrel(0.3)
            .apply(&input, params_rgb(16, 16))
            .unwrap();
        // Background fill could replace corner samples that fall
        // outside; check the centre 8×8 only.
        for py in 4..12 {
            for px in 4..12 {
                let off = (py * 16 + px) * 3;
                assert_eq!(out.planes[0].data[off], 100);
                assert_eq!(out.planes[0].data[off + 1], 150);
                assert_eq!(out.planes[0].data[off + 2], 200);
            }
        }
    }

    #[test]
    fn centre_pixel_unchanged() {
        // The centre of the image has r = 0 ⇒ scale = 1 ⇒ output
        // centre matches input centre exactly.
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 0));
        let out = Distort::pincushion(0.4)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        let centre_off = (7 * 15 + 7) * 3;
        let in_centre = (7 * 15 + 7) * 3;
        assert_eq!(
            out.planes[0].data[centre_off],
            input.planes[0].data[in_centre]
        );
        assert_eq!(
            out.planes[0].data[centre_off + 1],
            input.planes[0].data[in_centre + 1]
        );
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
        let err = Distort::new(0.1, 0.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Distort"));
    }

    #[test]
    fn rejects_non_finite_coefficients() {
        let input = rgb(4, 4, |_, _| (0, 0, 0));
        let err = Distort::new(f32::NAN, 0.0)
            .apply(&input, params_rgb(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("finite"));
    }

    #[test]
    fn barrel_samples_corner_from_inside() {
        // A bright dot at the top-left corner of the input — barrel
        // distortion (k1 < 0) shrinks the radial coordinate, so the
        // output corner samples a point INSIDE the input image (near
        // the dim background) rather than picking up the bright corner.
        let input = rgb(15, 15, |x, y| {
            if x == 0 && y == 0 {
                (200, 0, 0)
            } else {
                (10, 10, 10)
            }
        });
        let out = Distort::barrel(0.5)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        // Output (0, 0) samples a non-corner pixel ⇒ R should be far
        // below the corner's 200.
        assert!(
            out.planes[0].data[0] < 50,
            "barrel should NOT bring corner colour to corner; got R = {}",
            out.planes[0].data[0]
        );
    }

    #[test]
    fn pincushion_pushes_corners_outward() {
        // A bright dot at the top-left corner — pincushion (k1 > 0)
        // expands the radial coordinate, so the output corner asks for
        // a sample beyond the input corner ⇒ falls back to background.
        let input = rgb(15, 15, |x, y| {
            if x == 0 && y == 0 {
                (200, 0, 0)
            } else {
                (10, 10, 10)
            }
        });
        let out = Distort::pincushion(0.5)
            .with_background([7, 7, 7, 255])
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        // Background fill should be visible at the corner.
        assert_eq!(&out.planes[0].data[0..3], &[7, 7, 7]);
    }

    #[test]
    fn tangential_zero_matches_radial_only() {
        // r9: with p1 = p2 = 0 the full Brown-Conrady model must
        // produce a bit-exact match against the previous radial-only
        // path for any radial-coefficient choice.
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 0));

        let radial = Distort::new(0.2, -0.05)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        let with_zeros = Distort::new(0.2, -0.05)
            .with_tangential(0.0, 0.0)
            .apply(&input, params_rgb(15, 15))
            .unwrap();

        assert_eq!(radial.planes[0].data, with_zeros.planes[0].data);
    }

    #[test]
    fn tangential_centre_pixel_unchanged() {
        // At the lens centre dx = dy = r² = 0 ⇒ both tangential
        // contributions vanish (every term has at least one of dx,
        // dy, r² as a factor). The centre pixel must therefore equal
        // the input centre pixel even with strong tangential terms.
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 5));
        let out = Distort::new(0.0, 0.0)
            .with_tangential(0.4, -0.3)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        let centre_off = (7 * 15 + 7) * 3;
        assert_eq!(out.planes[0].data[centre_off], 7 * 17);
        assert_eq!(out.planes[0].data[centre_off + 1], 7 * 17);
        assert_eq!(out.planes[0].data[centre_off + 2], 5);
    }

    #[test]
    fn tangential_p1_breaks_horizontal_symmetry() {
        // A purely tangential `p1`-only model adds dy_t = p1 * r²
        // along the y-axis (where dx = 0) and 2·p1·dx·dy elsewhere.
        // The resulting flow is anti-symmetric across the horizontal
        // axis, so a vertically-symmetric input becomes
        // vertically-asymmetric in the output.
        let input = rgb(15, 15, |_, y| {
            // Symmetric Y-axis pattern: bright band centred on y=7.
            let dy = (y as i32 - 7).unsigned_abs() as u8;
            (200u8.saturating_sub(dy * 20), 0, 0)
        });
        let out_radial = Distort::new(0.0, 0.0)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        let out_tang = Distort::new(0.0, 0.0)
            .with_tangential(0.5, 0.0)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        // Pure radial (k1=k2=0) is identity ⇒ output equals input.
        assert_eq!(out_radial.planes[0].data, input.planes[0].data);
        // Tangential output must differ from the input at at least
        // one pixel — the warp actually shifts samples.
        assert_ne!(out_tang.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rejects_non_finite_tangential() {
        let input = rgb(4, 4, |_, _| (0, 0, 0));
        let err = Distort::new(0.0, 0.0)
            .with_tangential(f32::NAN, 0.0)
            .apply(&input, params_rgb(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("finite"));
    }

    #[test]
    fn with_centre_override_keeps_overridden_centre_unchanged() {
        // Move the lens centre to (3, 3); that pixel must now equal
        // the input pixel at (3, 3).
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 0));
        let out = Distort::pincushion(0.3)
            .with_centre(3.0, 3.0)
            .apply(&input, params_rgb(15, 15))
            .unwrap();
        let centre_off = (3 * 15 + 3) * 3;
        assert_eq!(out.planes[0].data[centre_off], 3 * 17);
        assert_eq!(out.planes[0].data[centre_off + 1], 3 * 17);
    }
}
