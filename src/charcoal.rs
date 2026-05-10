//! Non-photorealistic "charcoal sketch" stylise (ImageMagick `-charcoal R`).
//!
//! Pipeline (per-pixel, all in 8-bit luma space):
//!
//! 1. Reduce input to a tight `Gray8` luma plane.
//! 2. Optionally Gaussian-blur the luma plane in place (when
//!    [`radius`](Charcoal::radius) > 0). The blur is separable with
//!    sigma = `radius / 2` — the same heuristic used by [`Blur`].
//!    A larger pre-blur softens fine texture so the Sobel pass picks
//!    up only the larger structural edges, producing thicker strokes.
//! 3. Apply a 3×3 Sobel edge magnitude (`|Gx| + |Gy|`).
//! 4. Scale the magnitude by the configured `factor` (≥ 0.0; 1.0 means
//!    "as-is", >1.0 sharpens contrast).
//! 5. Invert: `out = 255 - clamp(magnitude * factor, 0, 255)`.
//!
//! The output is always a single-plane `Gray8` frame the same size as
//! the input — same shape change as [`Edge`](crate::Edge), so the
//! registry adapter wires it up the same way.
//!
//! `factor = 0.0` produces a pure-white image (every output is `255`),
//! the limit case "no edges visible". `factor = 1.0` is the default.
//! `radius = 0` (the default) skips the pre-blur entirely.

use crate::blur::gaussian_kernel;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Charcoal-sketch stylise filter.
///
/// `factor` scales the Sobel edge magnitude before inversion. Larger
/// values darken the strokes; smaller values fade them out. `radius`
/// controls an optional Gaussian pre-blur on the luma plane: `0` (the
/// default) skips it, `> 0` thickens the strokes by suppressing fine
/// texture before the Sobel pass.
#[derive(Clone, Copy, Debug)]
pub struct Charcoal {
    pub factor: f32,
    pub radius: u32,
}

impl Default for Charcoal {
    fn default() -> Self {
        Self {
            factor: 1.0,
            radius: 0,
        }
    }
}

impl Charcoal {
    /// Build a charcoal filter with explicit edge-magnitude scale and
    /// no pre-blur (`radius = 0`). Use [`Self::with_radius`] to enable
    /// the Gaussian pre-blur.
    pub fn new(factor: f32) -> Self {
        Self {
            factor: if factor.is_finite() && factor >= 0.0 {
                factor
            } else {
                1.0
            },
            radius: 0,
        }
    }

    /// Set the Gaussian pre-blur radius (in samples on each side, so
    /// the kernel length is `2 * radius + 1`). Sigma is auto-derived
    /// from the radius (`sigma = radius / 2`). `radius = 0` skips the
    /// pre-blur entirely — exactly matching the r6 charcoal output.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }
}

impl ImageFilter for Charcoal {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Charcoal does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut luma = build_luma_plane(input, params)?;
        if self.radius > 0 {
            luma = gaussian_blur_luma(&luma, w, h, self.radius);
        }
        let out = sobel3_invert(&luma, w, h, self.factor);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

/// Separable Gaussian blur on a single-plane `Gray8` luma buffer.
///
/// Mirrors the two-pass row-then-column structure used by [`Blur`] but
/// keeps a tight stride throughout (the input is always a tight-stride
/// scratch buffer built by [`build_luma_plane`]). Edge taps clamp to
/// the nearest in-bounds sample. Sigma is derived as `radius / 2`,
/// matching `Blur::new(radius)`'s default.
fn gaussian_blur_luma(luma: &[u8], w: usize, h: usize, radius: u32) -> Vec<u8> {
    if w == 0 || h == 0 {
        return luma.to_vec();
    }
    let kernel = gaussian_kernel(radius.max(1), (radius as f32).max(1.0) * 0.5);
    let r = (kernel.len() - 1) / 2;
    // Pass 1: horizontal blur.
    let mut horiz = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                acc += *kv * luma[y * w + sx] as f32;
            }
            horiz[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    // Pass 2: vertical blur back into a fresh buffer.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
                acc += *kv * horiz[sy * w + x] as f32;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Build a tight-stride Gray8 buffer from any supported input. Mirrors
/// the helper in `edge.rs` — kept private to this module to avoid
/// changing `Edge`'s public surface.
fn build_luma_plane(f: &VideoFrame, params: VideoStreamParams) -> Result<Vec<u8>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let mut out = vec![0u8; w * h];

    match params.format {
        PixelFormat::Gray8 => {
            let src = &f.planes[0];
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        PixelFormat::Rgb24 => {
            let src = &f.planes[0];
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
            let src = &f.planes[0];
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
            let src = &f.planes[0];
            for y in 0..h {
                out[y * w..y * w + w]
                    .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
            }
        }
        _ => {
            return Err(Error::unsupported(format!(
                "Charcoal: cannot derive luma from {:?}",
                params.format
            )));
        }
    }
    Ok(out)
}

/// 3×3 Sobel magnitude, scaled by `factor`, then inverted into the
/// output buffer (`out = 255 - clamp(mag * factor)`).
fn sobel3_invert(luma: &[u8], w: usize, h: usize, factor: f32) -> Vec<u8> {
    let mut out = vec![255u8; w * h];
    if w < 3 || h < 3 {
        // Border-only frame: no interior pixels, the Sobel magnitude is
        // undefined. We leave the output at its 255-fill (pure white,
        // matching the inversion of "no edges").
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
            let mag = (gx.unsigned_abs() + gy.unsigned_abs()) as f32 * factor;
            let scaled = mag.round().clamp(0.0, 255.0) as u8;
            out[y * w + x] = 255u8.saturating_sub(scaled);
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

    fn params_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flat_image_yields_pure_white() {
        // Sobel of a flat image is 0 everywhere ⇒ invert ⇒ 255.
        let input = gray(8, 8, |_, _| 100);
        let out = Charcoal::new(1.0).apply(&input, params_gray(8, 8)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn vertical_step_edge_darkens() {
        // Half-and-half input: left = 0, right = 200. The Sobel response
        // at the seam is large ⇒ inverted output is darker than 255 in
        // the middle columns.
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = Charcoal::new(1.0).apply(&input, params_gray(8, 8)).unwrap();
        // Row 3, column 3 sits on the edge ⇒ should be dark.
        let v = out.planes[0].data[3 * 8 + 3];
        assert!(v < 100, "expected edge to darken below 100, got {v}");
    }

    #[test]
    fn factor_zero_yields_pure_white() {
        // factor=0 ⇒ scaled magnitude is 0 ⇒ inverted ⇒ 255 (white).
        let input = gray(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let out = Charcoal::new(0.0).apply(&input, params_gray(8, 8)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn output_is_gray8_shape() {
        let input = gray(8, 8, |_, _| 100);
        let out = Charcoal::new(1.0).apply(&input, params_gray(8, 8)).unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 64);
        assert_eq!(out.planes[0].stride, 8);
    }

    #[test]
    fn radius_zero_matches_no_preblur() {
        // radius = 0 must be byte-identical to the legacy r6 charcoal
        // (no pre-blur). Build the same input and compare.
        let input = gray(16, 16, |x, y| {
            ((x.wrapping_mul(13) ^ y.wrapping_mul(7)) & 0xFF) as u8
        });
        let plain = Charcoal::new(1.0)
            .apply(&input, params_gray(16, 16))
            .unwrap();
        let r0 = Charcoal::new(1.0)
            .with_radius(0)
            .apply(&input, params_gray(16, 16))
            .unwrap();
        assert_eq!(r0.planes[0].data, plain.planes[0].data);
    }

    #[test]
    fn radius_thickens_strokes_on_step_edge() {
        // Step edge between two halves of the image. With no pre-blur
        // the Sobel response is concentrated in a single column on
        // each side of the seam. With radius = 2 the pre-blur smears
        // the step over a wider neighbourhood, so the inverted (dark)
        // strokes occupy more columns ⇒ more dark pixels overall.
        let input = gray(32, 32, |x, _| if x < 16 { 0 } else { 220 });

        let no_blur = Charcoal::new(1.0)
            .with_radius(0)
            .apply(&input, params_gray(32, 32))
            .unwrap();
        let blurred = Charcoal::new(1.0)
            .with_radius(2)
            .apply(&input, params_gray(32, 32))
            .unwrap();

        // Count dark pixels (output < 200 ⇒ "stroke ink").
        let count = |p: &VideoPlane| p.data.iter().filter(|&&v| v < 200).count();
        let n0 = count(&no_blur.planes[0]);
        let n2 = count(&blurred.planes[0]);
        assert!(
            n2 > n0,
            "pre-blur radius=2 should thicken strokes; got n0={n0}, n2={n2}"
        );
    }

    #[test]
    fn radius_heavy_stylisation_keeps_white_outside_strokes() {
        // radius = 10 is heavy stylisation; verify the filter still
        // returns a Gray8 image of the right shape and that pixels far
        // from any edge stay close to white.
        let input = gray(48, 48, |x, _| if x == 24 { 200 } else { 60 });
        let out = Charcoal::new(1.0)
            .with_radius(10)
            .apply(&input, params_gray(48, 48))
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 48 * 48);
        // Top-left corner is far from the edge column 24 — should be
        // nearly white (inverted near-zero edge magnitude).
        assert!(
            out.planes[0].data[0] > 240,
            "corner should be nearly white under heavy pre-blur, got {}",
            out.planes[0].data[0]
        );
    }

    #[test]
    fn rgba_input_produces_gray8() {
        let data: Vec<u8> = (0..64).flat_map(|_| [50u8, 100, 150, 222]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Charcoal::new(1.0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data.len(), 64); // 8×8 single-channel
                                                  // Flat colour ⇒ flat luma ⇒ no edges ⇒ pure white.
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }
}
