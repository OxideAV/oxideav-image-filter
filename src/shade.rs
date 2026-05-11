//! Directional shading from a virtual light source (`-shade az,el`).
//!
//! Treats the input image (or its luma proxy) as a height field and
//! illuminates it with a single directional light. Each output sample
//! is a shaded intensity in `[0, 255]` that the caller can then mix
//! back with the original via [`Composite`](crate::Composite) or use
//! as a relief / bump-map preview.
//!
//! Clean-room implementation derived from the textbook
//! "Lambertian-shaded height field" formulation:
//!
//! 1. Compute the gradient `(gx, gy)` of the height field via central
//!    differences:
//!    ```text
//!    gx = (h(x+1, y) - h(x-1, y)) / 2
//!    gy = (h(x, y+1) - h(x, y-1)) / 2
//!    ```
//! 2. Construct the surface normal `n = (-gx, -gy, 1) / |...|`.
//! 3. Construct the light direction from `(azimuth, elevation)` in
//!    degrees: `L = (cos(el)*cos(az), cos(el)*sin(az), sin(el))`.
//! 4. Lambertian intensity `I = max(0, n · L)`.
//! 5. `out = round(I * 255)`.
//!
//! Edge samples reuse the row / column they would otherwise wrap into
//! (border-clamp).
//!
//! ImageMagick's `-shade` has a colour-pass-through mode (the
//! `+shade` form) where the shaded grayscale modulates the original
//! pixel; we expose that via [`Shade::with_color`].
//!
//! # Pixel formats
//!
//! - `Gray8`: shading is computed directly on the plane.
//! - `Rgb24` / `Rgba`: a luma proxy `Y = (R + 2G + B) / 4` is used as
//!   the height field; `gray_only = false` (the `+shade` mode)
//!   modulates the original RGB by the shaded intensity, otherwise the
//!   filter outputs a grayscale `Gray8` frame.
//! - `Yuv*P`: the Y plane is treated as the height field and modulated;
//!   chroma is preserved.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Directional shading from a virtual light source.
#[derive(Clone, Copy, Debug)]
pub struct Shade {
    /// Light azimuth in degrees (rotation around the surface normal,
    /// `0` = +X direction, `90` = +Y direction).
    pub azimuth_degrees: f32,
    /// Light elevation in degrees (`0` = parallel to the surface;
    /// `90` = straight down).
    pub elevation_degrees: f32,
    /// When `true` (the IM `-shade`/`gray_only` mode), the output is a
    /// pure grayscale shading map. When `false` (the `+shade` mode),
    /// each colour pixel is multiplied by the shading intensity so the
    /// original colour shows through.
    pub gray_only: bool,
}

impl Default for Shade {
    fn default() -> Self {
        Self::new(30.0, 30.0)
    }
}

impl Shade {
    /// Build a shading filter from azimuth / elevation degrees.
    pub fn new(azimuth_degrees: f32, elevation_degrees: f32) -> Self {
        Self {
            azimuth_degrees,
            elevation_degrees,
            gray_only: true,
        }
    }

    /// Toggle `gray_only`. With `gray_only = false` the shading map
    /// modulates the original colours instead of replacing them.
    pub fn with_color(mut self, color_pass_through: bool) -> Self {
        self.gray_only = !color_pass_through;
        self
    }
}

fn luma_proxy(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 + 2 * g as u32 + b as u32) / 4) as u8
}

fn build_height_field(input: &VideoFrame, params: VideoStreamParams) -> Vec<u8> {
    let w = params.width as usize;
    let h = params.height as usize;
    let p = &input.planes[0];
    let mut field = vec![0u8; w * h];
    match params.format {
        PixelFormat::Gray8 | PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + w];
                field[y * w..y * w + w].copy_from_slice(row);
            }
        }
        PixelFormat::Rgb24 => {
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + 3 * w];
                for x in 0..w {
                    field[y * w + x] = luma_proxy(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
                }
            }
        }
        PixelFormat::Rgba => {
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + 4 * w];
                for x in 0..w {
                    field[y * w + x] = luma_proxy(row[4 * x], row[4 * x + 1], row[4 * x + 2]);
                }
            }
        }
        _ => {}
    }
    field
}

fn shade_intensity_map(field: &[u8], w: usize, h: usize, az_deg: f32, el_deg: f32) -> Vec<u8> {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    let lx = el.cos() * az.cos();
    let ly = el.cos() * az.sin();
    let lz = el.sin();
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            let gx = (field[y * w + xp] as f32 - field[y * w + xm] as f32) / 2.0;
            let gy = (field[yp * w + x] as f32 - field[ym * w + x] as f32) / 2.0;
            // Surface normal: (-gx, -gy, 1). Normalise.
            let nx = -gx;
            let ny = -gy;
            let nz = 1.0_f32;
            let nlen = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            let dot = (nx * lx + ny * ly + nz * lz) / nlen;
            let i = dot.max(0.0);
            out[y * w + x] = (i * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

impl ImageFilter for Shade {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let supported = matches!(
            params.format,
            PixelFormat::Gray8
                | PixelFormat::Rgb24
                | PixelFormat::Rgba
                | PixelFormat::Yuv420P
                | PixelFormat::Yuv422P
                | PixelFormat::Yuv444P
        );
        if !supported {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Shade does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let field = build_height_field(input, params);
        let intensity =
            shade_intensity_map(&field, w, h, self.azimuth_degrees, self.elevation_degrees);

        if self.gray_only {
            // Always emit a Gray8 frame.
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: w,
                    data: intensity,
                }],
            });
        }

        // Colour-pass-through (`+shade`): modulate the original by the
        // shading intensity. Keep the input format.
        let src = &input.planes[0];
        let mut out = input.clone();
        match params.format {
            PixelFormat::Gray8 => {
                let dst = &mut out.planes[0];
                let dst_stride = dst.stride;
                for y in 0..h {
                    let row = &mut dst.data[y * dst_stride..y * dst_stride + w];
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = ((*v as u32 * intensity[y * w + x] as u32) / 255) as u8;
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let dst = &mut out.planes[0];
                let dst_stride = dst.stride;
                for y in 0..h {
                    let row = &mut dst.data[y * dst_stride..y * dst_stride + 3 * w];
                    for x in 0..w {
                        let i = intensity[y * w + x] as u32;
                        row[3 * x] = ((row[3 * x] as u32 * i) / 255) as u8;
                        row[3 * x + 1] = ((row[3 * x + 1] as u32 * i) / 255) as u8;
                        row[3 * x + 2] = ((row[3 * x + 2] as u32 * i) / 255) as u8;
                    }
                }
            }
            PixelFormat::Rgba => {
                let dst = &mut out.planes[0];
                let dst_stride = dst.stride;
                for y in 0..h {
                    let row = &mut dst.data[y * dst_stride..y * dst_stride + 4 * w];
                    for x in 0..w {
                        let i = intensity[y * w + x] as u32;
                        row[4 * x] = ((row[4 * x] as u32 * i) / 255) as u8;
                        row[4 * x + 1] = ((row[4 * x + 1] as u32 * i) / 255) as u8;
                        row[4 * x + 2] = ((row[4 * x + 2] as u32 * i) / 255) as u8;
                        // alpha preserved
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let dst = &mut out.planes[0];
                let dst_stride = dst.stride;
                for y in 0..h {
                    let row = &mut dst.data[y * dst_stride..y * dst_stride + w];
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = ((*v as u32 * intensity[y * w + x] as u32) / 255) as u8;
                    }
                }
                // chroma preserved
            }
            _ => unreachable!(),
        }
        let _ = src; // src is read above through input.planes
        Ok(out)
    }
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
    fn flat_field_with_overhead_light_is_max_bright() {
        // Constant height ⇒ surface normal = (0, 0, 1). Light straight
        // down (elevation = 90°) ⇒ I = 1 ⇒ out = 255.
        let input = gray(4, 4, |_, _| 100);
        let out = Shade::new(0.0, 90.0)
            .apply(&input, params_gray(4, 4))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn flat_field_horizontal_light_is_dark() {
        // Constant height; light horizontal (elevation = 0°) ⇒ dot
        // = 0 ⇒ out = 0.
        let input = gray(4, 4, |_, _| 100);
        let out = Shade::new(0.0, 0.0)
            .apply(&input, params_gray(4, 4))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn slope_along_x_lit_from_x_is_bright() {
        // Height ramps in +x: gx > 0 ⇒ normal points -x. Light from
        // +x direction at low elevation ⇒ dot is negative ⇒ clamped 0.
        // Light from -x direction (azimuth 180°) ⇒ dot positive ⇒
        // bright. We just verify gradient sign affects the result.
        let input = gray(4, 1, |x, _| (x as u8) * 50);
        let lit_from_neg_x = Shade::new(180.0, 30.0)
            .apply(&input, params_gray(4, 1))
            .unwrap();
        let lit_from_pos_x = Shade::new(0.0, 30.0)
            .apply(&input, params_gray(4, 1))
            .unwrap();
        // The interior pixel (x = 1 or x = 2) should be brighter when
        // lit from -x than from +x because gx > 0.
        assert!(
            lit_from_neg_x.planes[0].data[1] > lit_from_pos_x.planes[0].data[1],
            "expected -x lighting to be brighter ({} > {})",
            lit_from_neg_x.planes[0].data[1],
            lit_from_pos_x.planes[0].data[1]
        );
    }

    #[test]
    fn output_is_gray8_in_default_mode() {
        let input = gray(4, 4, |x, _| (x * 50) as u8);
        let out = Shade::new(45.0, 45.0)
            .apply(&input, params_gray(4, 4))
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 16);
    }

    #[test]
    fn color_mode_keeps_input_format() {
        let data: Vec<u8> = (0..16).flat_map(|_| [200u8, 100, 50, 220]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Shade::new(0.0, 90.0)
            .with_color(true)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Flat field, light overhead ⇒ intensity = 1 ⇒ output =
        // input. Alpha preserved.
        for i in 0..16 {
            assert_eq!(&out.planes[0].data[i * 4..i * 4 + 4], &[200, 100, 50, 220]);
        }
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
        let err = Shade::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Bgr24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Shade"));
    }
}
