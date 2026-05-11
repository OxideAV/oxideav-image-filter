//! Cosine-weighted "soft" vignette.
//!
//! The existing [`Vignette`](crate::vignette::Vignette) uses a Gaussian
//! falloff, which leaves a sharp transition at the edge of the in-focus
//! disc when `radius` is large relative to `sigma`. This variant is the
//! cosine-squared falloff favoured by photographers for "soft" vignette
//! plates:
//!
//! ```text
//! cx = x_centre * width
//! cy = y_centre * height
//! d  = sqrt((px - cx)^2 + (py - cy)^2) / max_radius
//! t  = clamp((d - inner) / (outer - inner), 0, 1)
//! w  = 0.5 * (1 + cos(pi * t))   // raised cosine, 1 → 0
//! out = in * w
//! ```
//!
//! `inner` controls the boundary of the fully-bright disc (in
//! normalised radius units `[0, 1]` where `1` ≈ farthest corner from
//! the centre); `outer` controls where the image fades to black. The
//! cosine taper inside `[inner, outer]` is smoother than the
//! Gaussian version because both endpoints have zero derivative.
//!
//! Alpha is preserved on `Rgba`. Operates on `Rgb24` / `Rgba`; YUV
//! returns `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Cosine-weighted soft-vignette parameters.
///
/// `x` / `y` are normalised in `[0, 1]` (centre = `0.5 / 0.5`).
/// `inner` and `outer` are normalised against `0.5 * sqrt(w² + h²)`
/// (the farthest-corner distance), so `outer = 1.0` puts the corners
/// at full darkness. `inner >= outer` collapses to a hard cutoff
/// (constant `1` inside, constant `0` outside).
#[derive(Clone, Copy, Debug)]
pub struct VignetteSoft {
    pub x: f32,
    pub y: f32,
    pub inner: f32,
    pub outer: f32,
}

impl Default for VignetteSoft {
    fn default() -> Self {
        Self {
            x: 0.5,
            y: 0.5,
            inner: 0.4,
            outer: 1.0,
        }
    }
}

impl VignetteSoft {
    /// Build a soft-vignette filter with explicit centre + inner / outer.
    pub fn new(x: f32, y: f32, inner: f32, outer: f32) -> Self {
        Self {
            x,
            y,
            inner: inner.max(0.0),
            outer: outer.max(0.0),
        }
    }
}

impl ImageFilter for VignetteSoft {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: VignetteSoft requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: VignetteSoft requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let cx = self.x * params.width as f32;
        let cy = self.y * params.height as f32;
        // Farthest-corner distance from `(cx, cy)`.
        let corners = [
            ((cx).hypot(cy)),
            ((params.width as f32 - cx).hypot(cy)),
            ((cx).hypot(params.height as f32 - cy)),
            ((params.width as f32 - cx).hypot(params.height as f32 - cy)),
        ];
        let max_r = corners.iter().cloned().fold(0.0_f32, f32::max).max(1e-3);
        let row_bytes = w * bpp;
        let src = &input.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        let inner = self.inner;
        let outer = self.outer.max(inner + 1e-6);
        let span = outer - inner;
        for py in 0..h {
            let src_row = &src.data[py * src.stride..py * src.stride + row_bytes];
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            let dy = py as f32 + 0.5 - cy;
            for px in 0..w {
                let dx = px as f32 + 0.5 - cx;
                let d = (dx * dx + dy * dy).sqrt() / max_r;
                let weight = if d <= inner {
                    1.0
                } else if d >= outer {
                    0.0
                } else {
                    let t = (d - inner) / span;
                    0.5 * (1.0 + (std::f32::consts::PI * t).cos())
                };
                let base = px * bpp;
                dst_row[base] = (src_row[base] as f32 * weight).round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = (src_row[base + 1] as f32 * weight)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = (src_row[base + 2] as f32 * weight)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                if has_alpha {
                    dst_row[base + 3] = src_row[base + 3];
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    fn rgb_flat(w: u32, h: u32, v: u8) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data: vec![v; (w * h * 3) as usize],
            }],
        }
    }

    #[test]
    fn corners_darker_than_centre() {
        let input = rgb_flat(8, 8, 200);
        let out = VignetteSoft::default().apply(&input, p_rgb(8, 8)).unwrap();
        // Centre row 4, col 4 vs corner (0, 0).
        let centre = out.planes[0].data[(4 * 8 + 4) * 3];
        let corner = out.planes[0].data[0];
        assert!(
            centre > corner,
            "centre {centre} should be brighter than corner {corner}"
        );
    }

    #[test]
    fn outer_zero_is_no_op_inside_disc() {
        // inner = 0, outer = 0 collapses to inside == 1 / outside == 0.
        // For a 1x1 image the single pixel is at the centre so weight = 1.
        let input = rgb_flat(1, 1, 200);
        let out = VignetteSoft::new(0.5, 0.5, 0.0, 0.0)
            .apply(&input, p_rgb(1, 1))
            .unwrap();
        // The centre pixel sits exactly on the centre so d == 0; weight 1.
        assert_eq!(out.planes[0].data, vec![200, 200, 200]);
    }

    #[test]
    fn shape_preserved() {
        let input = rgb_flat(5, 3, 100);
        let out = VignetteSoft::default().apply(&input, p_rgb(5, 3)).unwrap();
        assert_eq!(out.planes[0].stride, 15);
        assert_eq!(out.planes[0].data.len(), 45);
    }

    #[test]
    fn alpha_passes_through_on_rgba() {
        let data: Vec<u8> = (0..16).flat_map(|_| [200u8, 200, 200, 222]).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = VignetteSoft::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn cosine_is_smoother_than_gaussian_at_seam() {
        // Probe near the inner boundary: the cosine derivative there
        // is exactly 0, so neighbouring weights should be very close.
        let input = rgb_flat(16, 16, 200);
        let out = VignetteSoft::new(0.5, 0.5, 0.3, 0.9)
            .apply(&input, p_rgb(16, 16))
            .unwrap();
        // Two pixels straddling the inner boundary should differ by at
        // most a few units, not a sharp step.
        // Inner radius 0.3 of farthest-corner ≈ 0.3 * 0.5*sqrt(2)*16 ≈ 3.4 px.
        // So pixels at offsets 3 and 4 from centre (col 8 row 8) are
        // around the seam.
        let a = out.planes[0].data[(8 * 16 + 8 + 3) * 3];
        let b = out.planes[0].data[(8 * 16 + 8 + 4) * 3];
        let diff = (a as i32 - b as i32).abs();
        assert!(diff < 30, "smooth-seam diff {diff} too large");
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let err = VignetteSoft::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("VignetteSoft"));
    }
}
