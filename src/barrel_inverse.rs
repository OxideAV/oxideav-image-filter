//! Inverse barrel / pincushion distortion (ImageMagick `-distort BarrelInverse a,b,c,d`).
//!
//! Where the [`Distort`](crate::distort::Distort) filter applies the
//! polynomial radius scale `r_out = r · (a·r³ + b·r² + c·r + d)`
//! directly (output radius is a polynomial in the input radius), this
//! filter applies the **inverse** mapping — output samples are pulled
//! from `r_in = r / (a·r³ + b·r² + c·r + d)`. Together the two are the
//! `+barrel` / `-barrel-inverse` pair documented by ImageMagick.
//!
//! Practical effect:
//!
//! - `BarrelInverse` *expands* the centre and *compresses* the edges
//!   when `a`/`b`/`c` are positive — useful for correcting an image
//!   that was already barrel-distorted by a wide-angle lens.
//! - The constant term `d` acts as a uniform radial zoom (`d > 1.0`
//!   zooms out, `d < 1.0` zooms in).
//!
//! For each output pixel `(px, py)` the normalised radius `r = |p|`
//! (where `p` is the offset from the optical centre, normalised by
//! half-min-side) drives the inverse map:
//!
//! ```text
//! scale = a·r³ + b·r² + c·r + d
//! r_sample = r / scale
//! sample at (cx + (p.x / r) · r_sample · r_norm,
//!            cy + (p.y / r) · r_sample · r_norm)
//! ```
//!
//! `r == 0` maps to the centre unchanged (no division-by-zero — we
//! short-circuit). Samples outside the input bounds use the configured
//! background colour.
//!
//! Operates on packed `Rgb24` / `Rgba`.

use crate::implode::bilinear_sample_into;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Inverse-barrel polynomial distortion.
#[derive(Clone, Copy, Debug)]
pub struct BarrelInverse {
    /// Cubic coefficient.
    pub a: f32,
    /// Quadratic coefficient.
    pub b: f32,
    /// Linear coefficient.
    pub c: f32,
    /// Constant term (acts as a radial zoom factor — `1.0` is the
    /// identity zoom).
    pub d: f32,
    /// Optional centre override (pixels). `None` means image centre.
    pub centre: Option<(f32, f32)>,
    /// Normalisation radius (pixels). `None` means `min(w, h) / 2`.
    pub r_norm: Option<f32>,
    /// Out-of-bounds fill colour `[R, G, B, A]`.
    pub background: [u8; 4],
}

impl BarrelInverse {
    /// Build with the four polynomial coefficients (`a`, `b`, `c`, `d`).
    /// Centre / normalisation default to image-derived values. The
    /// background fill defaults to opaque black.
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            a,
            b,
            c,
            d,
            centre: None,
            r_norm: None,
            background: [0, 0, 0, 255],
        }
    }

    /// Override the centre (pixel coordinates).
    pub fn with_centre(mut self, cx: f32, cy: f32) -> Self {
        self.centre = Some((cx, cy));
        self
    }

    /// Override the normalisation radius (pixels).
    pub fn with_r_norm(mut self, r_norm: f32) -> Self {
        self.r_norm = Some(r_norm);
        self
    }

    /// Override the out-of-bounds fill colour.
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }
}

impl ImageFilter for BarrelInverse {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: BarrelInverse requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if !self.a.is_finite() || !self.b.is_finite() || !self.c.is_finite() || !self.d.is_finite()
        {
            return Err(Error::invalid(
                "oxideav-image-filter: BarrelInverse coefficients must be finite",
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
            let dy_n = (py as f32 - cy) * inv_r;
            for px in 0..w {
                let dx_n = (px as f32 - cx) * inv_r;
                let r = (dx_n * dx_n + dy_n * dy_n).sqrt();
                let scale = self.a * r * r * r + self.b * r * r + self.c * r + self.d;
                let (sx, sy) = if r > f32::EPSILON && scale.abs() > f32::EPSILON {
                    let r_sample = r / scale;
                    let ux = dx_n / r;
                    let uy = dy_n / r;
                    (cx + ux * r_sample * r_norm, cy + uy * r_sample * r_norm)
                } else {
                    // r ≈ 0 → centre stays unchanged. scale ≈ 0 → fall
                    // back to identity to keep the pixel defined.
                    (px as f32, py as f32)
                };
                let base = px * bpp;
                const EPS: f32 = 1e-4;
                if sx < -EPS
                    || sx > (w as f32 - 1.0) + EPS
                    || sy < -EPS
                    || sy > (h as f32 - 1.0) + EPS
                {
                    write_bg(&mut dst_row[base..base + bpp], &self.background, bpp);
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

fn write_bg(dst: &mut [u8], bg: &[u8; 4], bpp: usize) {
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn identity_coefficients_pass_through_centre() {
        // a=b=c=0, d=1 means scale = 1 everywhere → r_sample = r → the
        // output equals the input. Off-centre samples may pick up tiny
        // bilinear drift, so just check the exact centre pixel.
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 0));
        let out = BarrelInverse::new(0.0, 0.0, 0.0, 1.0)
            .apply(&input, p_rgb(15, 15))
            .unwrap();
        let centre = (7 * 15 + 7) * 3;
        assert_eq!(
            out.planes[0].data[centre], input.planes[0].data[centre],
            "centre R"
        );
    }

    #[test]
    fn flat_image_is_invariant() {
        // Non-trivial coefficients on a flat image — every interior
        // sample is the same colour and the warp keeps them so.
        let input = rgb(16, 16, |_, _| (100, 150, 200));
        let out = BarrelInverse::new(0.1, 0.05, 0.0, 1.0)
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        // Check the centre 8×8 (corners may fall outside source).
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
        // r=0 at the centre — short-circuit branch keeps `(sx, sy) =
        // (px, py)` → output equals input at the centre.
        let input = rgb(15, 15, |x, y| (x as u8 * 17, y as u8 * 17, 0));
        let out = BarrelInverse::new(0.1, 0.05, 0.02, 1.0)
            .apply(&input, p_rgb(15, 15))
            .unwrap();
        let off = (7 * 15 + 7) * 3;
        assert_eq!(out.planes[0].data[off], input.planes[0].data[off]);
        assert_eq!(out.planes[0].data[off + 1], input.planes[0].data[off + 1]);
        assert_eq!(out.planes[0].data[off + 2], input.planes[0].data[off + 2]);
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
        let err = BarrelInverse::new(0.0, 0.0, 0.0, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("BarrelInverse"));
    }

    #[test]
    fn rejects_non_finite_coefficients() {
        let input = rgb(4, 4, |_, _| (0, 0, 0));
        let err = BarrelInverse::new(f32::NAN, 0.0, 0.0, 1.0)
            .apply(&input, p_rgb(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("finite"));
    }

    #[test]
    fn rgba_alpha_sampled() {
        // Identity coefficients → centre pixel preserves its alpha.
        let mut data = Vec::with_capacity(64);
        for y in 0..4 {
            for x in 0..4 {
                let _ = y;
                data.extend_from_slice(&[10, 20, 30, 40 + x as u8]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = BarrelInverse::new(0.0, 0.0, 0.0, 1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Centre pixel (2, 2) — alpha at input is 40 + 2 = 42.
        let off = (2 * 4 + 2) * 4;
        assert_eq!(out.planes[0].data[off + 3], 42);
    }
}
