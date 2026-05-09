//! Gaussian "vignette" — darken the image radially away from a centre
//! point.
//!
//! Clean-room implementation of the textbook vignette operation:
//!
//! ```text
//! cx = x_centre * width
//! cy = y_centre * height
//! d  = sqrt((px - cx)^2 + (py - cy)^2)
//! e  = max(0, d - radius)
//! w  = exp(-(e^2) / (2 * sigma^2))
//! out = in * w   (clamped to [0, 255])
//! ```
//!
//! Pixels inside `radius` of the centre keep their full value; pixels
//! beyond the radius fall off as a Gaussian with standard deviation
//! `sigma`. Parameter semantics mirror ImageMagick's `-vignette
//! RxS{+x{+y}}` (with `radius`/`sigma` interpreted in pixels and
//! `(x, y)` interpreted as image-relative offsets that we encode as
//! normalised `[0.0, 1.0]` to keep the API resolution-independent).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Gaussian vignette parameters.
///
/// `x` / `y` are normalised in `[0.0, 1.0]` (centre = 0.5 / 0.5).
/// `radius` is in pixels; `sigma` is in pixels too. A sigma of 0 means
/// "no falloff" (the image stays untouched outside the radius), so for
/// any meaningful effect callers should pass a positive sigma.
#[derive(Clone, Copy, Debug)]
pub struct Vignette {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub sigma: f32,
}

impl Default for Vignette {
    fn default() -> Self {
        Self {
            x: 0.5,
            y: 0.5,
            radius: 0.0,
            sigma: 0.0, // resolved to 0.65 * min(w, h) in apply() when 0.
        }
    }
}

impl Vignette {
    /// Build a vignette with explicit centre + radius + sigma.
    pub fn new(x: f32, y: f32, radius: f32, sigma: f32) -> Self {
        Self {
            x,
            y,
            radius: radius.max(0.0),
            sigma: sigma.max(0.0),
        }
    }
}

impl ImageFilter for Vignette {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Vignette requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let cx = self.x * params.width as f32;
        let cy = self.y * params.height as f32;
        // ImageMagick default: sigma = 0.65 * min(w, h) when caller
        // passes 0 / unset.
        let sigma = if self.sigma > 0.0 {
            self.sigma
        } else {
            0.65 * (w.min(h)) as f32
        };
        let two_sigma_sq = 2.0 * sigma * sigma;
        let radius = self.radius.max(0.0);

        let mut out = vec![0u8; row_bytes * h];
        for py in 0..h {
            let src_row = &src.data[py * src.stride..py * src.stride + row_bytes];
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            let dy = py as f32 + 0.5 - cy;
            for px in 0..w {
                let dx = px as f32 + 0.5 - cx;
                let d = (dx * dx + dy * dy).sqrt();
                let excess = (d - radius).max(0.0);
                let weight = if two_sigma_sq > 0.0 {
                    (-(excess * excess) / two_sigma_sq).exp()
                } else if excess == 0.0 {
                    1.0
                } else {
                    0.0
                };
                let base = px * bpp;
                let r = src_row[base] as f32 * weight;
                let g = src_row[base + 1] as f32 * weight;
                let b = src_row[base + 2] as f32 * weight;
                dst_row[base] = r.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = g.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = b.round().clamp(0.0, 255.0) as u8;
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
    fn corners_darker_than_centre_on_default_8x8() {
        let input = rgb(8, 8, |_, _| (200, 200, 200));
        let out = Vignette::default().apply(&input, params_rgb(8, 8)).unwrap();
        // Centre pixel (4, 4) sits very near the Gaussian peak ⇒ ≈200.
        let centre_idx = (4 * 8 + 4) * 3;
        let corner_idx = 0; // (0, 0)
        let centre_r = out.planes[0].data[centre_idx];
        let corner_r = out.planes[0].data[corner_idx];
        assert!(
            centre_r > corner_r,
            "centre ({centre_r}) must stay brighter than corner ({corner_r})"
        );
        // Centre should be near-untouched (sigma = 0.65*8 = 5.2; d≈0.7).
        assert!(centre_r >= 195, "centre too dark: {centre_r}");
    }

    #[test]
    fn radius_covering_image_is_passthrough() {
        // Radius >= image diagonal → every pixel is inside, weight = 1.
        let input = rgb(8, 8, |x, y| (x as u8 * 32, y as u8 * 32, 100));
        let out = Vignette::new(0.5, 0.5, 100.0, 5.0)
            .apply(&input, params_rgb(8, 8))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let data: Vec<u8> = (0..16).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Vignette::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 222, "alpha at pixel {i}");
        }
    }

    #[test]
    fn rejects_yuv420p() {
        let input = VideoFrame {
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
        let err = Vignette::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Vignette"));
    }
}
