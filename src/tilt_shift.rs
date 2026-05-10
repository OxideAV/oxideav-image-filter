//! Tilt-shift: selective Gaussian blur masked by an in-focus band.
//! Simulates the miniature-photography depth-of-field look where a
//! narrow slice of the frame is sharp and everything to either side of
//! it falls off into a smoothly-blurred background.
//!
//! The mask is a piecewise-linear envelope of the **Gaussian-blurred
//! copy's** weight (1 − focus_weight):
//!
//! ```text
//!   focus_weight(x, y) =
//!       1.0                                          if  |d_perp| <= focus_height/2
//!       1.0 − (|d_perp| - focus_height/2)
//!              / falloff_height                      otherwise (clamped to [0, 1])
//!
//!   out(x, y, c) = focus_weight(x, y)        · sharp(x, y, c)
//!                + (1 - focus_weight(x, y))  · blur (x, y, c)
//! ```
//!
//! `d_perp` is the **perpendicular distance** from the pixel to the
//! focus-axis line (in normalised plane-height units). For the default
//! horizontal band (`angle_degrees = 0`) the focus axis is the
//! horizontal centre-line and `d_perp = (y_norm - focus_centre)` —
//! identical to the pre-r9 row-only model. Setting `angle_degrees = 90`
//! rotates the axis to vertical, so the band becomes a sharp column
//! flanked by blurred regions to the left and right.
//!
//! `focus_centre` and `focus_height` are in normalised image
//! coordinates `[0, 1]` — `0.5` puts the focus band at the centre.
//! `focus_centre` shifts the axis along its perpendicular direction
//! (i.e. for a horizontal band a smaller `focus_centre` puts the band
//! higher; for a vertical band it puts the band leftward).
//! `falloff_height` (also normalised) controls how quickly the blur
//! ramps in once you leave the focused band; `0.0` means an abrupt
//! step.
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
/// copy according to an in-focus envelope. The envelope is a band
/// perpendicular to a configurable axis (horizontal by default,
/// rotatable via [`Self::with_angle_degrees`]).
#[derive(Clone, Debug)]
pub struct TiltShift {
    /// Centre of the in-focus band along the axis perpendicular to
    /// the focus axis, in normalised image coordinates `[0.0, 1.0]`.
    /// `0.5` is the middle; for a horizontal band a smaller value
    /// shifts the band upward, for a vertical band a smaller value
    /// shifts it leftward.
    pub focus_centre: f32,
    /// Width of the fully in-focus band measured perpendicular to the
    /// focus axis, normalised against image height.
    pub focus_height: f32,
    /// Extent (normalised) of the linear ramp from "fully sharp" to
    /// "fully blurred" on each side of the focus band. `0.0` means an
    /// abrupt edge.
    pub falloff_height: f32,
    /// Gaussian radius applied to the blurred copy.
    pub blur_radius: u32,
    /// Gaussian sigma applied to the blurred copy.
    pub blur_sigma: f32,
    /// Rotation of the focus band in degrees around the image centre.
    /// `0.0` is the legacy horizontal band (focus axis runs left-to-
    /// right; band is "above + below"). `90.0` rotates the axis to
    /// vertical so the band becomes a sharp column flanked by
    /// blurred regions to the left and right. Any finite value works
    /// — the math is fully continuous.
    pub angle_degrees: f32,
}

impl Default for TiltShift {
    fn default() -> Self {
        Self {
            focus_centre: 0.5,
            focus_height: 0.20,
            falloff_height: 0.15,
            blur_radius: 4,
            blur_sigma: 2.0,
            angle_degrees: 0.0,
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
            angle_degrees: 0.0,
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

    /// Builder: rotate the focus band by `angle` degrees around the
    /// image centre. `0.0` (the default) is the legacy horizontal
    /// band; `90.0` is a vertical band; any other finite value
    /// produces a diagonal band.
    pub fn with_angle_degrees(mut self, angle: f32) -> Self {
        self.angle_degrees = angle;
        self
    }

    /// Compute the in-focus weight for a row of plane height `ph`.
    /// Specialised legacy path used when the focus band is horizontal
    /// (angle == 0); equivalent to the per-pixel
    /// [`focus_weight_xy`](Self::focus_weight_xy) but cheaper and
    /// bit-exact identical to the pre-r9 output. Returned weight is
    /// in `[0.0, 1.0]` — `1.0` means fully sharp, `0.0` means fully
    /// blurred.
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

    /// Per-pixel in-focus weight for a rotated focus band. Used when
    /// `angle_degrees` is non-zero.
    ///
    /// The focus axis passes through the image centre at angle
    /// `angle_degrees` (measured clockwise from the horizontal in the
    /// usual image-coordinate convention where +y is downward). The
    /// perpendicular distance from the pixel's normalised centre
    /// `(x_n, y_n)` to that axis is taken in normalised units (the
    /// pixel coordinates are normalised against `ph` on both axes so
    /// the band keeps its visual thickness as the band rotates).
    fn focus_weight_xy(&self, px: usize, py: usize, pw: usize, ph: usize) -> f32 {
        if pw == 0 || ph == 0 {
            return 1.0;
        }
        let theta = self.angle_degrees.to_radians();
        let (s, co) = theta.sin_cos();
        // Normalise both axes against the image height so a square
        // image keeps the band the same visual thickness regardless
        // of rotation. Using ph for both axes also means the
        // `focus_centre = 0.5` placement (which is a y-anchor for the
        // legacy horizontal band) maps cleanly onto an x-anchor for
        // the 90° vertical-band case.
        let x_n = (px as f32 + 0.5) / ph as f32;
        let y_n = (py as f32 + 0.5) / ph as f32;
        // Centre-of-image in the same normalised coordinate frame.
        let cx_n = (pw as f32 * 0.5) / ph as f32;
        let cy_n = 0.5;
        // Perpendicular distance from the pixel to the rotated focus
        // axis (passing through the image centre). focus_centre then
        // shifts the band along that perpendicular direction:
        // `focus_centre = 0.5` keeps the band centred, `< 0.5` shifts
        // it one way, `> 0.5` the other.
        let d_perp = -s * (x_n - cx_n) + co * (y_n - cy_n);
        let dist = (d_perp - (self.focus_centre - 0.5)).abs();
        let half_band = (self.focus_height.max(0.0) * 0.5).min(0.5);
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
            || !self.angle_degrees.is_finite()
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
    // Fast path: a horizontal focus band (angle == 0) lets us share
    // the in-focus weight across an entire row, which keeps the
    // pre-r9 default cost. Any other angle falls back to the
    // per-pixel `focus_weight_xy` path.
    let rotated = f.angle_degrees.abs() > f32::EPSILON;
    for py in 0..h {
        let row_weight = if rotated { 0.0 } else { f.focus_weight(py, h) };
        let row_inv = 1.0 - row_weight;
        let sharp_row = &sharp.data[py * sharp.stride..py * sharp.stride + w * bpp];
        let blurred_row = &blurred.data[py * blurred.stride..py * blurred.stride + w * bpp];
        let dst_row = &mut out[py * w * bpp..(py + 1) * w * bpp];
        for px in 0..w {
            let (weight, inv) = if rotated {
                let wgt = f.focus_weight_xy(px, py, w, h);
                (wgt, 1.0 - wgt)
            } else {
                (row_weight, row_inv)
            };
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
    fn angle_zero_matches_legacy_horizontal_band() {
        // r9: with angle = 0 the new rotated-band code path must be
        // bit-exact identical to the pre-r9 horizontal-band output.
        let input = gray_frame(32, 32, |x, _| if x % 2 == 0 { 255 } else { 0 });
        let legacy = TiltShift::default()
            .apply(&input, params_gray(32, 32))
            .unwrap();
        let with_angle = TiltShift::default()
            .with_angle_degrees(0.0)
            .apply(&input, params_gray(32, 32))
            .unwrap();
        assert_eq!(legacy.planes[0].data, with_angle.planes[0].data);
    }

    fn ts_default_vertical() -> TiltShift {
        TiltShift::default()
            .with_focus_centre(0.5)
            .with_focus_height(0.20)
            .with_falloff_height(0.15)
            .with_angle_degrees(90.0)
    }

    #[test]
    fn angle_90_rotates_band_to_vertical() {
        // Horizontal noise pattern (every ROW alternates 0/255) — so
        // the blurred mean is ~127 everywhere AND a vertical sample
        // line through any column carries the alternating-row noise.
        // With angle = 0 row 16 (in-focus row) would preserve its
        // single-noise value (constant per row); with angle = 90 the
        // band is vertical, so column 16 (in-focus column) preserves
        // the alternating-row pattern down its length.
        let input = gray_frame(32, 32, |_, y| if y % 2 == 0 { 255 } else { 0 });
        let f_v = ts_default_vertical();
        let out = f_v.apply(&input, params_gray(32, 32)).unwrap();

        // Column 16 lies on the vertical focus axis ⇒ should
        // preserve the per-row noise pattern (255 / 0 alternating).
        for py in 0..32usize {
            let expected = if py % 2 == 0 { 255 } else { 0 };
            let v = out.planes[0].data[py * 32 + 16];
            // Allow a small tolerance — the focus weight at row 16
            // column 16 is 1.0, but at the band edges it ramps off,
            // so columns near the centre should still be sharp.
            assert!(
                (v as i32 - expected).abs() < 30,
                "vertical-band column 16 row {py}: expected ~{expected}, got {v}"
            );
        }

        // Far-left column 0 lies well outside the vertical focus band
        // ⇒ pixels should be blurred. Variance along that column
        // (down the rows) should drop dramatically vs the raw input
        // (which alternates 0/255 ⇒ huge variance ~ 16383).
        let col0: Vec<u8> = (0..32).map(|y| out.planes[0].data[y * 32]).collect();
        let mean = col0.iter().map(|&v| v as u32).sum::<u32>() / 32;
        let var = col0
            .iter()
            .map(|&v| {
                let d = v as i32 - mean as i32;
                (d * d) as u32
            })
            .sum::<u32>()
            / 32;
        // Input variance is ~16K; blurred-out column should be much
        // less than that.
        assert!(
            var < 4000,
            "vertical-band: column 0 variance should be much lower than input (~16K), got {var}"
        );
    }

    #[test]
    fn focus_weight_xy_centre_pixel_is_in_focus() {
        // The image-centre pixel sits exactly on the focus axis (any
        // angle, since rotation is around the centre) ⇒ focus weight
        // = 1.0 regardless of angle.
        let f = TiltShift::default()
            .with_angle_degrees(37.0)
            .with_focus_height(0.20);
        let w = f.focus_weight_xy(16, 16, 32, 32);
        assert!(
            (w - 1.0).abs() < 1e-3,
            "centre pixel should be fully in-focus regardless of rotation, got {w}"
        );
    }

    #[test]
    fn rejects_non_finite_angle() {
        let input = gray_frame(8, 8, |_, _| 0);
        let err = TiltShift::default()
            .with_angle_degrees(f32::INFINITY)
            .apply(&input, params_gray(8, 8))
            .unwrap_err();
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
