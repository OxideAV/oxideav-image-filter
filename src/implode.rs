//! "Implode" / "Explode" radial pinch effect.
//!
//! ImageMagick's `-implode N` operator: bend rays from the image centre
//! either inward (positive `factor`, the classic implode look) or
//! outward (negative `factor`, "explode"). Per-pixel inverse mapping
//! with bilinear sampling, leaving everything outside the inscribed
//! circle (`r >= 1.0`) untouched.
//!
//! ```text
//! For each output (px, py):
//!     dx = px - cx; dy = py - cy
//!     max_r = min(w/2, h/2)
//!     r = sqrt(dx*dx + dy*dy) / max_r
//!     if r >= 1.0: out = in (no change)
//!     else:
//!         scale = (1 - factor * (1 - r))^2
//!         sx = cx + dx * scale
//!         sy = cy + dy * scale
//!         out = bilinear_sample(in, sx, sy)
//! ```
//!
//! `factor = 0.0` is bit-exact identity. Only operates on packed
//! Rgb24/Rgba; alpha is sampled (and so survives) as just another channel.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Radial pinch / push transform.
///
/// `factor` controls the strength of the pinch: 0 is identity,
/// positive values implode toward the centre, negative values explode
/// outward. Typical range is roughly `-1.0..=1.0`; values outside that
/// can fold pixels and produce wrap-around artefacts.
#[derive(Clone, Copy, Debug)]
pub struct Implode {
    pub factor: f32,
}

impl Default for Implode {
    fn default() -> Self {
        Self { factor: 0.5 }
    }
}

impl Implode {
    /// Build an implode filter with explicit factor.
    pub fn new(factor: f32) -> Self {
        Self {
            factor: if factor.is_finite() { factor } else { 0.0 },
        }
    }
}

impl ImageFilter for Implode {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Implode requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let mut out = vec![0u8; row_bytes * h];

        // Identity fast-path (bit-exact pass-through).
        if self.factor.abs() < 1e-9 {
            for y in 0..h {
                let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
                let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
                dst_row.copy_from_slice(src_row);
            }
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        let cx = (w as f32 - 1.0) * 0.5;
        let cy = (h as f32 - 1.0) * 0.5;
        let max_r = (w.min(h) as f32) * 0.5;
        let factor = self.factor;

        for py in 0..h {
            let dy = py as f32 - cy;
            let dst_row = &mut out[py * row_bytes..(py + 1) * row_bytes];
            for px in 0..w {
                let dx = px as f32 - cx;
                let dist = (dx * dx + dy * dy).sqrt();
                let r = if max_r > 0.0 { dist / max_r } else { 0.0 };
                let base = px * bpp;
                if r >= 1.0 || max_r == 0.0 {
                    let src_off = py * src.stride + base;
                    dst_row[base..base + bpp].copy_from_slice(&src.data[src_off..src_off + bpp]);
                    continue;
                }
                // r' = r * (1 - factor * (1 - r))^2 ⇒ scale = r' / r =
                // (1 - factor * (1 - r))^2 (independent of r).
                let scale = {
                    let s = 1.0 - factor * (1.0 - r);
                    s * s
                };
                let sx = cx + dx * scale;
                let sy = cy + dy * scale;
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

/// Clamped bilinear sampler — coordinates outside `[0, w-1] × [0, h-1]`
/// snap to the nearest edge. Writes `bpp` bytes into `dst`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bilinear_sample_into(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    bpp: usize,
    sx: f32,
    sy: f32,
    dst: &mut [u8],
) {
    let x = sx.clamp(0.0, (w - 1) as f32);
    let y = sy.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    for ch in 0..bpp {
        let p00 = src[y0 * stride + x0 * bpp + ch] as f32;
        let p10 = src[y0 * stride + x1 * bpp + ch] as f32;
        let p01 = src[y1 * stride + x0 * bpp + ch] as f32;
        let p11 = src[y1 * stride + x1 * bpp + ch] as f32;
        let v = p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy;
        dst[ch] = v.round().clamp(0.0, 255.0) as u8;
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
    fn factor_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Implode::new(0.0).apply(&input, params_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_invariant_under_implode() {
        // A flat image has the same colour at every coordinate, so any
        // resampling-based transform should leave it unchanged.
        let input = rgb(8, 8, |_, _| (100, 150, 200));
        let out = Implode::new(0.5).apply(&input, params_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn outside_inscribed_circle_unchanged() {
        // Corner pixel of an 8×8 grid sits outside the inscribed circle
        // (r > 1). Implode must leave it alone.
        let input = rgb(8, 8, |x, y| {
            ((x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8)
        });
        let out = Implode::new(0.5).apply(&input, params_rgb(8, 8)).unwrap();
        // Top-left corner.
        let stride = out.planes[0].stride;
        for &x in &[0usize, 7] {
            for &y in &[0usize, 7] {
                let off_in = y * stride + x * 3;
                let off_out = y * stride + x * 3;
                assert_eq!(out.planes[0].data[off_out], input.planes[0].data[off_in]);
                assert_eq!(
                    out.planes[0].data[off_out + 1],
                    input.planes[0].data[off_in + 1]
                );
                assert_eq!(
                    out.planes[0].data[off_out + 2],
                    input.planes[0].data[off_in + 2]
                );
            }
        }
    }

    #[test]
    fn rgba_alpha_passes_through_for_uniform_alpha() {
        let data: Vec<u8> = (0..64).flat_map(|_| [200u8, 100, 50, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Implode::new(0.5)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        for i in 0..64 {
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
        let err = Implode::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Implode"));
    }
}
