//! Pencil-sketch stylise (ImageMagick `-sketch radius,sigma,angle`).
//!
//! Clean-room implementation of the documented ImageMagick pipeline:
//!
//! 1. Reduce the input to a tight `Gray8` luma plane.
//! 2. Apply a directional motion blur of `radius` / `sigma` along
//!    `angle_degrees`. The motion-blur kernel approximates a 1-D
//!    Gaussian sampled at integer offsets `t = -radius..=+radius`,
//!    each tap displaced by `(t * cos θ, t * sin θ)` from the centre.
//! 3. Take a Sobel edge-magnitude pass on the motion-blurred luma,
//!    then invert it (`out = 255 - clamp(mag, 0, 255)`).
//!
//! Step 3 is the same shape as [`Charcoal`](crate::charcoal::Charcoal) —
//! the difference is the directional pre-blur in step 2, which produces
//! the streaky "pencil hatch" look that gives the filter its name.
//!
//! Output is a single-plane `Gray8` frame of the same shape as the
//! input. `angle_degrees` is interpreted in the same sense as
//! [`MotionBlur`](crate::motion_blur::MotionBlur): 0° is horizontal
//! (rightward), positive angles rotate clockwise in screen space.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Pencil-sketch stylise.
#[derive(Clone, Copy, Debug)]
pub struct Sketch {
    /// Half-length of the directional blur kernel (samples on each
    /// side of the centre tap; total taps = `2 * radius + 1`).
    pub radius: u32,
    /// Standard deviation of the 1-D Gaussian used to weight the
    /// directional blur taps. `sigma <= 0` collapses to a centre-only
    /// kernel (no blur).
    pub sigma: f32,
    /// Direction of the hatch strokes in screen-space degrees. `0.0`
    /// produces horizontal strokes; `45.0` slants them.
    pub angle_degrees: f32,
}

impl Default for Sketch {
    fn default() -> Self {
        Self {
            radius: 3,
            sigma: 1.5,
            angle_degrees: 0.0,
        }
    }
}

impl Sketch {
    /// Build a sketch filter with explicit `(radius, sigma, angle)`.
    /// Non-finite `sigma` / `angle_degrees` are coerced to safe
    /// defaults so the per-pixel maths stays defined.
    pub fn new(radius: u32, sigma: f32, angle_degrees: f32) -> Self {
        Self {
            radius,
            sigma: if sigma.is_finite() && sigma > 0.0 {
                sigma
            } else {
                0.0
            },
            angle_degrees: if angle_degrees.is_finite() {
                angle_degrees
            } else {
                0.0
            },
        }
    }
}

impl ImageFilter for Sketch {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Sketch does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma(input, params)?;
        let blurred = motion_blur_luma(&luma, w, h, self.radius, self.sigma, self.angle_degrees);
        let out = sobel_invert(&blurred, w, h);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

/// Pack the input's luma plane into a tight-stride `Gray8` buffer.
fn build_luma(f: &VideoFrame, params: VideoStreamParams) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return Ok(out);
    }
    let src = &f.planes[0];
    match params.format {
        PixelFormat::Gray8 => {
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        PixelFormat::Rgb24 => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 3 * w];
                for x in 0..w {
                    let r = row[3 * x] as u16;
                    let g = row[3 * x + 1] as u16;
                    let b = row[3 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        PixelFormat::Rgba => {
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                for x in 0..w {
                    let r = row[4 * x] as u16;
                    let g = row[4 * x + 1] as u16;
                    let b = row[4 * x + 2] as u16;
                    out[y * w + x] = ((r + 2 * g + b) / 4) as u8;
                }
            }
        }
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "Sketch: cannot derive luma from {other:?}"
            )));
        }
    }
    Ok(out)
}

/// 1-D directional blur with Gaussian-weighted taps.
///
/// Per pixel `(x, y)` the output samples `2 * radius + 1` source
/// locations along `(cos θ, sin θ)`. Out-of-bounds taps clamp to the
/// nearest in-bounds sample. When `sigma <= 0` only the centre tap is
/// used (so the output is bit-identical to the input).
fn motion_blur_luma(
    luma: &[u8],
    w: usize,
    h: usize,
    radius: u32,
    sigma: f32,
    angle: f32,
) -> Vec<u8> {
    if w == 0 || h == 0 || radius == 0 || sigma <= 0.0 {
        return luma.to_vec();
    }
    let r = radius as i32;
    // Build a 1-D Gaussian whose centre tap is t = 0, normalised so the
    // weights sum to 1. (Clean-room — same closed-form as Wikipedia's
    // "Gaussian function".)
    let two_s2 = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0f32;
    for t in -r..=r {
        let w_t = (-(t as f32 * t as f32) / two_s2).exp();
        taps.push(w_t);
        sum += w_t;
    }
    if sum > 0.0 {
        for w_t in taps.iter_mut() {
            *w_t /= sum;
        }
    }
    let theta = angle.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, w_t) in taps.iter().enumerate() {
                let t = i as i32 - r;
                let sx = (x as f32 + t as f32 * cos_t).round() as i32;
                let sy = (y as f32 + t as f32 * sin_t).round() as i32;
                let sxc = sx.clamp(0, w as i32 - 1) as usize;
                let syc = sy.clamp(0, h as i32 - 1) as usize;
                acc += *w_t * luma[syc * w + sxc] as f32;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// 3×3 Sobel magnitude followed by inversion. Borders stay at the
/// "no edges" 255 value (matching Charcoal's convention).
fn sobel_invert(luma: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![255u8; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p00 = luma[(y - 1) * w + (x - 1)] as i32;
            let p01 = luma[(y - 1) * w + x] as i32;
            let p02 = luma[(y - 1) * w + (x + 1)] as i32;
            let p10 = luma[y * w + (x - 1)] as i32;
            let p12 = luma[y * w + (x + 1)] as i32;
            let p20 = luma[(y + 1) * w + (x - 1)] as i32;
            let p21 = luma[(y + 1) * w + x] as i32;
            let p22 = luma[(y + 1) * w + (x + 1)] as i32;
            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
            let mag = (gx.unsigned_abs() + gy.unsigned_abs()).min(255) as u8;
            out[y * w + x] = 255u8.saturating_sub(mag);
        }
    }
    out
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
    fn flat_image_yields_pure_white() {
        // Flat input → directional blur is identity → Sobel = 0 → invert → 255.
        let input = gray(8, 8, |_, _| 80);
        let out = Sketch::new(2, 1.0, 30.0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn vertical_edge_darkens_centre() {
        // Bright/dark vertical split — interior columns near the seam
        // pick up large Sobel magnitude after the (mild) directional
        // blur, so the inverted output is below 255 there.
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = Sketch::new(1, 0.5, 0.0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        let v = out.planes[0].data[3 * 8 + 3];
        assert!(v < 200, "seam should darken, got {v}");
    }

    #[test]
    fn sigma_zero_skips_blur() {
        // sigma=0 means no directional blur. The Sketch then collapses
        // to Charcoal-with-factor-1 over the same luma plane → vertical
        // edges still produce darkened seam pixels.
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = Sketch::new(3, 0.0, 0.0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        let v = out.planes[0].data[3 * 8 + 3];
        assert!(v < 255, "edge should still darken when blur disabled");
    }

    #[test]
    fn output_is_gray8_shape() {
        let input = gray(6, 6, |_, _| 100);
        let out = Sketch::default().apply(&input, p_gray(6, 6)).unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 6);
        assert_eq!(out.planes[0].data.len(), 36);
    }

    #[test]
    fn rgb_input_accepted_via_luma() {
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x < 4 { 0 } else { 200 };
                data.extend_from_slice(&[v, v, v]);
                let _ = y;
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 8,
            height: 8,
        };
        let out = Sketch::new(1, 0.5, 0.0).apply(&input, p).unwrap();
        assert_eq!(out.planes[0].data.len(), 64);
        assert_eq!(out.planes[0].stride, 8);
        // Seam between x=3 and x=4 → darker than 255 somewhere on row 3.
        assert!(out.planes[0].data[3 * 8 + 3] < 255);
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
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 4,
            height: 4,
        };
        let err = Sketch::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("Sketch"));
    }

    #[test]
    fn non_finite_sigma_coerced_to_zero() {
        let s = Sketch::new(3, f32::NAN, 30.0);
        assert_eq!(s.sigma, 0.0);
        let s2 = Sketch::new(3, -1.0, 30.0);
        assert_eq!(s2.sigma, 0.0);
    }
}
