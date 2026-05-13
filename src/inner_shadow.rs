//! Inner-shadow effect — soft offset shadow rendered *inside* the
//! opaque subject region instead of behind it.
//!
//! The filter takes an `Rgba` foreground (`input`), interprets its
//! alpha channel as a coverage mask, and darkens pixels just inside the
//! boundary along the `(offset_x, offset_y)` direction. Conceptually:
//!
//! 1. Build `inv_alpha = 255 - input.a` — the "outside" mask.
//! 2. Offset that mask by `(offset_x, offset_y)` and box-blur it
//!    (radius `radius`).
//! 3. For each pixel `p` inside the subject (`input.a > 0`), blend
//!    `p` toward the configured `colour` by the offset-and-blurred
//!    outside mask, scaled by `opacity`. Pixels far from any edge keep
//!    their original colour; pixels just inside the lit edge are
//!    darkened the most.
//!
//! Pixel formats: `Rgba` only — the algorithm depends on the alpha
//! channel as a coverage mask. Output keeps the same `(width, height)`.
//! Alpha is preserved.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Inner-shadow filter.
#[derive(Clone, Debug)]
pub struct InnerShadow {
    /// Horizontal shadow offset in pixels. Positive offsets darken the
    /// right-inside edge (light coming from the left).
    pub offset_x: i32,
    /// Vertical shadow offset in pixels. Positive offsets darken the
    /// bottom-inside edge (light coming from the top).
    pub offset_y: i32,
    /// Box-blur radius applied to the shadow mask. `0` is hard-edged.
    pub radius: u32,
    /// Shadow opacity in `[0, 1]`.
    pub opacity: f32,
    /// RGB colour of the shadow.
    pub colour: [u8; 3],
}

impl Default for InnerShadow {
    fn default() -> Self {
        Self {
            offset_x: 2,
            offset_y: 2,
            radius: 3,
            opacity: 0.6,
            colour: [0, 0, 0],
        }
    }
}

impl InnerShadow {
    pub fn new(offset_x: i32, offset_y: i32, radius: u32) -> Self {
        Self {
            offset_x,
            offset_y,
            radius,
            ..Default::default()
        }
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    pub fn with_colour(mut self, colour: [u8; 3]) -> Self {
        self.colour = colour;
        self
    }
}

impl ImageFilter for InnerShadow {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if params.format != PixelFormat::Rgba {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: InnerShadow requires Rgba, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: InnerShadow requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];

        // Invert the alpha channel: outside pixels become 255, inside
        // pixels become 0. This is the "outside" mask the shadow leaks
        // into the inside through the offset + blur.
        let mut inv_alpha = vec![0u8; w * h];
        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + 4 * w];
            for x in 0..w {
                inv_alpha[y * w + x] = 255 - row[4 * x + 3];
            }
        }
        // Box blur the inverted mask.
        let blurred = box_blur_gray(&inv_alpha, w, h, self.radius);

        let opacity = self.opacity.clamp(0.0, 1.0);
        let dx = self.offset_x;
        let dy = self.offset_y;

        let mut out = vec![0u8; 4 * w * h];
        for y in 0..h {
            let s_row = &src.data[y * src.stride..y * src.stride + 4 * w];
            let o_row = &mut out[y * 4 * w..(y + 1) * 4 * w];
            for x in 0..w {
                let s_r = s_row[4 * x] as f32;
                let s_g = s_row[4 * x + 1] as f32;
                let s_b = s_row[4 * x + 2] as f32;
                let s_a = s_row[4 * x + 3];

                // Sample the shadow mask at (x - dx, y - dy). Outside the
                // image counts as fully outside (255).
                let sx = x as i32 - dx;
                let sy = y as i32 - dy;
                let shadow_t = if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                    blurred[sy as usize * w + sx as usize] as f32 / 255.0
                } else {
                    1.0
                } * opacity;

                // Only darken pixels *inside* the subject (s_a > 0).
                let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
                let mut o_r = s_r;
                let mut o_g = s_g;
                let mut o_b = s_b;
                if s_a > 0 {
                    o_r = lerp(s_r, self.colour[0] as f32, shadow_t);
                    o_g = lerp(s_g, self.colour[1] as f32, shadow_t);
                    o_b = lerp(s_b, self.colour[2] as f32, shadow_t);
                }
                o_row[4 * x] = o_r.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 1] = o_g.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 2] = o_b.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 3] = s_a;
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: 4 * w,
                data: out,
            }],
        })
    }
}

fn box_blur_gray(src: &[u8], w: usize, h: usize, radius: u32) -> Vec<u8> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let r = radius as i32;
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dx in -r..=r {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + sx] as u32;
            }
            tmp[y * w + x] = (acc / (2 * r as u32 + 1)) as u8;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dy in -r..=r {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                acc += tmp[sy * w + x] as u32;
            }
            out[y * w + x] = (acc / (2 * r as u32 + 1)) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    fn p_rgba(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn interior_pixel_unchanged_when_far_from_edge() {
        // 8×8 fully opaque image, radius=1 blur, offset=(1, 1). The
        // canvas-edge "outside" leaks in along the top/left edge, but
        // the deep-interior pixel at (5, 5) is far enough from the
        // canvas edge that the blurred inverted-alpha sample is zero
        // there → no inner shadow → pixel unchanged.
        let input = rgba(8, 8, |_, _| [100, 150, 200, 255]);
        let out = InnerShadow::new(1, 1, 1)
            .apply(&input, p_rgba(8, 8))
            .unwrap();
        let off = (5 * 8 + 5) * 4;
        assert_eq!(&out.planes[0].data[off..off + 4], &[100, 150, 200, 255]);
    }

    #[test]
    fn transparent_pixels_unchanged() {
        // Transparent pixels (alpha==0) should keep their RGB intact —
        // inner shadow only darkens *inside* the subject.
        let input = rgba(4, 4, |_, _| [200, 100, 50, 0]);
        let out = InnerShadow::new(1, 1, 0)
            .with_opacity(1.0)
            .apply(&input, p_rgba(4, 4))
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[200, 100, 50, 0]);
        }
    }

    #[test]
    fn edge_interior_pixel_is_darker() {
        // Half-opaque-half-transparent split: opaque on x>=2 with
        // colour (200, 200, 200), transparent elsewhere. Inner shadow
        // offset (1, 0) → the inside-edge pixel at x=2 should be darker
        // than the deep-interior pixel at x=3.
        let input = rgba(4, 4, |x, _| {
            if x >= 2 {
                [200, 200, 200, 255]
            } else {
                [0, 0, 0, 0]
            }
        });
        let out = InnerShadow::new(1, 0, 0)
            .with_opacity(1.0)
            .with_colour([0, 0, 0])
            .apply(&input, p_rgba(4, 4))
            .unwrap();
        let edge_off = 2 * 4;
        let interior_off = 3 * 4;
        let edge_r = out.planes[0].data[edge_off];
        let interior_r = out.planes[0].data[interior_off];
        assert!(
            edge_r < interior_r,
            "edge {edge_r} should be darker than interior {interior_r}"
        );
    }

    #[test]
    fn alpha_is_preserved() {
        let input = rgba(3, 3, |_, _| [128, 64, 32, 200]);
        let out = InnerShadow::new(1, 1, 1)
            .apply(&input, p_rgba(3, 3))
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 200);
        }
    }

    #[test]
    fn rejects_non_rgba() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = InnerShadow::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("InnerShadow"));
    }
}
