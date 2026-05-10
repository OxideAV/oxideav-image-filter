//! Tilt-shift: selective Gaussian blur masked by a horizontal in-focus
//! band. Simulates the miniature-photography depth-of-field look where
//! a narrow horizontal slice of the frame is sharp and everything
//! above and below it falls off into a smoothly-blurred background.
//!
//! The mask is a piecewise-linear envelope of the **Gaussian-blurred
//! copy's** weight (1 − focus_weight):
//!
//! ```text
//!   focus_weight(y) =
//!       1.0                                    if  |y - centre| <= focus_height/2
//!       1.0 − (|y - centre| - focus_height/2)
//!              / falloff_height                otherwise (clamped to [0, 1])
//!
//!   out(x, y, c) = focus_weight(y)        · sharp(x, y, c)
//!                + (1 - focus_weight(y))  · blur (x, y, c)
//! ```
//!
//! `focus_centre` and `focus_height` are in normalised image
//! coordinates `[0, 1]` — `0.5` puts the focus band at the vertical
//! middle. `falloff_height` (also normalised) controls how quickly the
//! blur ramps in once you leave the focused band; `0.0` means an
//! abrupt step.
//!
//! The blurred copy uses the existing separable Gaussian helpers from
//! `blur.rs`. Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV
//! (every plane is processed; alpha on RGBA passes through unchanged).
//!
//! Defaults match a sensible "miniature street scene" look: focus band
//! occupies 20% of the frame height centred at the vertical middle,
//! with another 15% of falloff above and below, and a `radius = 4`,
//! `sigma = 2.0` Gaussian blur for the out-of-focus regions.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, gaussian_kernel, plane_dims};
use crate::{is_supported_format, Blur, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Tilt-shift filter: blends a sharp source with a Gaussian-blurred
/// copy according to a horizontal in-focus envelope.
#[derive(Clone, Debug)]
pub struct TiltShift {
    /// Vertical centre of the in-focus band, in normalised image
    /// coordinates `[0.0, 1.0]`. `0.5` is the middle.
    pub focus_centre: f32,
    /// Height of the fully in-focus band, normalised against image
    /// height.
    pub focus_height: f32,
    /// Vertical extent (normalised) of the linear ramp from "fully
    /// sharp" to "fully blurred" on each side of the focus band.
    /// `0.0` means an abrupt edge.
    pub falloff_height: f32,
    /// Gaussian radius applied to the blurred copy.
    pub blur_radius: u32,
    /// Gaussian sigma applied to the blurred copy.
    pub blur_sigma: f32,
}

impl Default for TiltShift {
    fn default() -> Self {
        Self {
            focus_centre: 0.5,
            focus_height: 0.20,
            falloff_height: 0.15,
            blur_radius: 4,
            blur_sigma: 2.0,
        }
    }
}

impl TiltShift {
    /// Convenience constructor with all-numeric parameters.
    pub fn new(
        focus_centre: f32,
        focus_height: f32,
        falloff_height: f32,
        blur_radius: u32,
        blur_sigma: f32,
    ) -> Self {
        Self {
            focus_centre,
            focus_height,
            falloff_height,
            blur_radius,
            blur_sigma,
        }
    }

    /// Builder: change the in-focus centre (normalised).
    pub fn with_focus_centre(mut self, c: f32) -> Self {
        self.focus_centre = c;
        self
    }

    /// Builder: change the in-focus band height (normalised).
    pub fn with_focus_height(mut self, h: f32) -> Self {
        self.focus_height = h;
        self
    }

    /// Builder: change the falloff height (normalised).
    pub fn with_falloff_height(mut self, h: f32) -> Self {
        self.falloff_height = h;
        self
    }

    /// Builder: override the Gaussian blur parameters.
    pub fn with_blur(mut self, radius: u32, sigma: f32) -> Self {
        self.blur_radius = radius;
        self.blur_sigma = sigma;
        self
    }

    /// Compute the in-focus weight for a row of plane height `ph`.
    /// Returned weight is in `[0.0, 1.0]` — `1.0` means fully sharp,
    /// `0.0` means fully blurred.
    fn focus_weight(&self, py: usize, ph: usize) -> f32 {
        if ph == 0 {
            return 1.0;
        }
        let y_norm = (py as f32 + 0.5) / ph as f32;
        let half_band = (self.focus_height.max(0.0) * 0.5).min(0.5);
        let dist = (y_norm - self.focus_centre).abs();
        if dist <= half_band {
            return 1.0;
        }
        let falloff = self.falloff_height.max(0.0);
        if falloff < 1e-6 {
            return 0.0;
        }
        let outside = (dist - half_band) / falloff;
        (1.0 - outside).clamp(0.0, 1.0)
    }
}

impl ImageFilter for TiltShift {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: TiltShift does not yet handle {:?}",
                params.format
            )));
        }
        if !self.focus_centre.is_finite()
            || !self.focus_height.is_finite()
            || !self.falloff_height.is_finite()
            || !self.blur_sigma.is_finite()
        {
            return Err(Error::invalid(
                "oxideav-image-filter: TiltShift parameters must be finite",
            ));
        }

        // Build the blurred reference frame using the existing Blur
        // filter — same separable-Gaussian path with edge clamp.
        let blurred = Blur::new(self.blur_radius)
            .with_sigma(self.blur_sigma)
            .apply(input, params)?;

        // We don't need the kernel here; just keep the symbol available
        // for future per-plane work. (Some compilers would otherwise
        // warn about the unused gaussian_kernel import.)
        let _ = gaussian_kernel(self.blur_radius.max(1), self.blur_sigma.max(f32::EPSILON));

        let (cx, cy) = chroma_subsampling(params.format);
        let mut out = input.clone();

        for (idx, plane) in out.planes.iter_mut().enumerate() {
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            let alpha_passthrough = matches!(params.format, PixelFormat::Rgba) && idx == 0;
            *plane = blend_plane(
                plane,
                &blurred.planes[idx],
                pw as usize,
                ph as usize,
                bpp,
                self,
                alpha_passthrough,
            );
        }

        Ok(out)
    }
}

fn blend_plane(
    sharp: &VideoPlane,
    blurred: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    f: &TiltShift,
    alpha_passthrough: bool,
) -> VideoPlane {
    let mut out = vec![0u8; w * bpp * h];
    for py in 0..h {
        let weight = f.focus_weight(py, h);
        let inv = 1.0 - weight;
        let sharp_row = &sharp.data[py * sharp.stride..py * sharp.stride + w * bpp];
        let blurred_row = &blurred.data[py * blurred.stride..py * blurred.stride + w * bpp];
        let dst_row = &mut out[py * w * bpp..(py + 1) * w * bpp];
        for px in 0..w {
            for ch in 0..bpp {
                let off = px * bpp + ch;
                if alpha_passthrough && bpp == 4 && ch == 3 {
                    dst_row[off] = sharp_row[off];
                    continue;
                }
                let s = sharp_row[off] as f32;
                let b = blurred_row[off] as f32;
                let v = weight * s + inv * b;
                dst_row[off] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    VideoPlane {
        stride: w * bpp,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_frame(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
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
    fn flat_image_is_invariant() {
        let input = gray_frame(16, 16, |_, _| 128);
        let out = TiltShift::default()
            .apply(&input, params_gray(16, 16))
            .unwrap();
        for &v in &out.planes[0].data {
            // Blur preserves a flat field (every row is 128); blending
            // 128 with 128 stays at 128.
            assert!((v as i16 - 128).abs() <= 1);
        }
    }

    #[test]
    fn focus_band_pixels_match_sharp() {
        // A vertical noise pattern; centre row must equal the input
        // row (focus weight = 1 at the centre band).
        let input = gray_frame(32, 32, |x, _| if x % 2 == 0 { 255 } else { 0 });
        let f = TiltShift::default()
            .with_focus_centre(0.5)
            .with_focus_height(0.20)
            .with_falloff_height(0.15);
        let out = f.apply(&input, params_gray(32, 32)).unwrap();
        // Row 16 sits inside the [0.40, 0.60] focus band ⇒ should be
        // bytewise identical to the input.
        let row = 16;
        let in_row = &input.planes[0].data[row * 32..(row + 1) * 32];
        let out_row = &out.planes[0].data[row * 32..(row + 1) * 32];
        assert_eq!(out_row, in_row);
    }

    #[test]
    fn out_of_focus_band_pixels_are_blurred() {
        // Same noise pattern; row 0 is well outside the focus band.
        // The blurred copy of an alternating 0/255 noise pattern is
        // close to 127 everywhere; the output row 0 should look like
        // the blur, NOT the noise.
        let input = gray_frame(32, 32, |x, _| if x % 2 == 0 { 255 } else { 0 });
        let f = TiltShift::default();
        let out = f.apply(&input, params_gray(32, 32)).unwrap();
        let row = 0;
        let avg: u32 = (0..32)
            .map(|x| out.planes[0].data[row * 32 + x] as u32)
            .sum();
        let mean = avg / 32;
        assert!(
            (mean as i32 - 127).abs() < 50,
            "expected row 0 to look ~grey (blurred), got mean = {mean}"
        );
        // Variance should be much lower than the input's (which alternates 0/255 ⇒ huge variance).
        let var: u32 = (0..32)
            .map(|x| {
                let v = out.planes[0].data[row * 32 + x] as i32 - mean as i32;
                (v * v) as u32
            })
            .sum::<u32>()
            / 32;
        assert!(var < 4000, "blurred row variance too high: {var}");
    }

    #[test]
    fn focus_weight_envelope_endpoints() {
        let f = TiltShift::default()
            .with_focus_centre(0.5)
            .with_focus_height(0.20)
            .with_falloff_height(0.15);
        // Row at exact centre: weight = 1.
        assert!((f.focus_weight(50, 100) - 1.0).abs() < 1e-3);
        // Row at the very top: well outside the focus band → fully blurred.
        assert!(f.focus_weight(0, 100) < 0.05);
        // Row halfway through the falloff: weight ≈ 0.5.
        // half_band = 0.10 ⇒ falloff starts at y_norm = 0.40; halfway
        // through the 0.15-tall ramp is y_norm = 0.40 - 0.075 = 0.325 ⇒
        // py_norm = 0.325 ⇒ py = round(0.325 * 100 - 0.5) = 32.
        let w = f.focus_weight(32, 100);
        assert!(
            (w - 0.5).abs() < 0.1,
            "halfway falloff weight should be ~0.5, got {w}"
        );
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![0; 32],
            }],
        };
        let err = TiltShift::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P10Le,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("TiltShift"));
    }

    #[test]
    fn rejects_non_finite_parameters() {
        let input = gray_frame(8, 8, |_, _| 0);
        let f = TiltShift::default().with_focus_centre(f32::NAN);
        let err = f.apply(&input, params_gray(8, 8)).unwrap_err();
        assert!(format!("{err}").contains("finite"));
    }

    #[test]
    fn rgba_alpha_passes_through() {
        let mut data = Vec::with_capacity(16 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let a = (x * 30 + y * 20) as u8;
                data.extend_from_slice(&[100, 150, 200, a]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = TiltShift::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let off = ((y * 4 + x) * 4 + 3) as usize;
                let expected = (x * 30 + y * 20) as u8;
                assert_eq!(out.planes[0].data[off], expected);
            }
        }
    }
}
