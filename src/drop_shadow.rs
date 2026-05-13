//! Drop-shadow underlay — soft offset shadow composited behind opaque
//! subject pixels.
//!
//! The filter takes an `Rgba` foreground (`input`), interprets its
//! alpha channel as a coverage mask, and produces:
//!
//! 1. A blurred & offset copy of the mask (the shadow), darkened to a
//!    configurable `colour` and `opacity`.
//! 2. The original foreground composited (Porter–Duff `over`) on top of
//!    the shadow.
//!
//! Clean-room recipe — straight separable Gaussian blur + arithmetic
//! composite, all built from primitives this crate already ships:
//!
//! ```text
//! shadow_alpha[y, x] = blur(input.a, sigma)[y - dy, x - dx] * opacity
//! shadow_rgb         = colour                  // constant per pixel
//! out                = src over shadow         // straight-alpha over
//! ```
//!
//! Pixel formats: `Rgba` only. RGB / Gray / YUV inputs return
//! [`Error::unsupported`] (the algorithm depends on the alpha channel
//! as a coverage mask). The output keeps the same `(width, height)`
//! shape — the shadow is rendered onto the same canvas; consumers
//! wanting an outset canvas should pre-pad with [`Extent`](crate::Extent).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Drop-shadow filter.
#[derive(Clone, Debug)]
pub struct DropShadow {
    /// Horizontal shadow offset in pixels (`+` shifts the shadow right).
    pub offset_x: i32,
    /// Vertical shadow offset in pixels (`+` shifts the shadow down).
    pub offset_y: i32,
    /// Gaussian blur radius applied to the shadow's alpha mask. `0`
    /// produces a hard-edged offset duplicate.
    pub radius: u32,
    /// Shadow opacity in `[0, 1]`. Caller-friendly alias for an
    /// alpha multiplier on the blurred mask.
    pub opacity: f32,
    /// RGB colour of the shadow.
    pub colour: [u8; 3],
}

impl Default for DropShadow {
    fn default() -> Self {
        Self {
            offset_x: 4,
            offset_y: 4,
            radius: 6,
            opacity: 0.5,
            colour: [0, 0, 0],
        }
    }
}

impl DropShadow {
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

impl ImageFilter for DropShadow {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if params.format != PixelFormat::Rgba {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: DropShadow requires Rgba, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DropShadow requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];

        // Build a Gray8 alpha mask buffer.
        let mut mask = vec![0u8; w * h];
        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + 4 * w];
            for x in 0..w {
                mask[y * w + x] = row[4 * x + 3];
            }
        }
        // Blur the mask (separable Gaussian, edge-clamped). For
        // simplicity we use a box-radius-N triple pass which closely
        // approximates a Gaussian for small radii and is cheap to
        // implement without pulling in the Blur filter's plane bookkeeping.
        let blurred = box_blur_gray(&mask, w, h, self.radius);

        let opacity = self.opacity.clamp(0.0, 1.0);
        let dx = self.offset_x;
        let dy = self.offset_y;

        // Composite: shadow first (offset), foreground over it.
        let mut out = vec![0u8; 4 * w * h];
        for y in 0..h {
            let s_row = &src.data[y * src.stride..y * src.stride + 4 * w];
            let o_row = &mut out[y * 4 * w..(y + 1) * 4 * w];
            for x in 0..w {
                // Sample the shadow alpha at (x - dx, y - dy).
                let sx = x as i32 - dx;
                let sy = y as i32 - dy;
                let shadow_a = if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                    let m = blurred[sy as usize * w + sx as usize] as f32 / 255.0;
                    (m * opacity * 255.0).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                };

                let s_r = s_row[4 * x] as f32;
                let s_g = s_row[4 * x + 1] as f32;
                let s_b = s_row[4 * x + 2] as f32;
                let s_a = s_row[4 * x + 3] as f32 / 255.0;
                let sh_r = self.colour[0] as f32;
                let sh_g = self.colour[1] as f32;
                let sh_b = self.colour[2] as f32;
                let sh_a = shadow_a as f32 / 255.0;

                // straight-alpha "over": out = src + dst * (1 - src.a)
                let out_a = s_a + sh_a * (1.0 - s_a);
                let mix = |sc: f32, dc: f32| {
                    if out_a < 1e-6 {
                        0.0
                    } else {
                        (sc * s_a + dc * sh_a * (1.0 - s_a)) / out_a
                    }
                };
                let o_r = mix(s_r, sh_r);
                let o_g = mix(s_g, sh_g);
                let o_b = mix(s_b, sh_b);
                o_row[4 * x] = o_r.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 1] = o_g.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 2] = o_b.round().clamp(0.0, 255.0) as u8;
                o_row[4 * x + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
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

/// Cheap separable box blur, edge-clamped. Equivalent to a uniform
/// `(2r+1) × (2r+1)` window mean — close enough to a Gaussian for
/// soft-shadow purposes and avoids pulling in the full `Blur` plane
/// bookkeeping.
fn box_blur_gray(src: &[u8], w: usize, h: usize, radius: u32) -> Vec<u8> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let r = radius as i32;
    let mut tmp = vec![0u8; w * h];
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dx in -r..=r {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + sx] as u32;
            }
            let denom = (2 * r + 1) as u32;
            tmp[y * w + x] = (acc / denom) as u8;
        }
    }
    // Vertical pass.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dy in -r..=r {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                acc += tmp[sy * w + x] as u32;
            }
            let denom = (2 * r + 1) as u32;
            out[y * w + x] = (acc / denom) as u8;
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
    fn fully_opaque_input_is_visually_intact_at_top() {
        // 8×8 white opaque subject in the centre 4×4.
        let input = rgba(8, 8, |x, y| {
            if (2..=5).contains(&x) && (2..=5).contains(&y) {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 0]
            }
        });
        let out = DropShadow::new(2, 2, 1)
            .with_opacity(1.0)
            .with_colour([0, 0, 0])
            .apply(&input, p_rgba(8, 8))
            .unwrap();
        // Centre pixel of the subject (3, 3) keeps white.
        let off = (3 * 8 + 3) * 4;
        assert_eq!(&out.planes[0].data[off..off + 4], &[255, 255, 255, 255]);
    }

    #[test]
    fn transparent_input_produces_only_shadow() {
        // Fully-transparent input — shadow mask is 0 → output also fully
        // transparent.
        let input = rgba(4, 4, |_, _| [128, 64, 64, 0]);
        let out = DropShadow::new(1, 1, 1)
            .with_opacity(0.5)
            .apply(&input, p_rgba(4, 4))
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 0);
        }
    }

    #[test]
    fn shadow_visible_beneath_opaque_corner() {
        // Subject pixel at (0, 0); offset (2, 2) shadow blurred radius 1.
        // We expect a non-zero alpha somewhere in the (2-3, 2-3) band.
        let input = rgba(8, 8, |x, y| {
            if x == 0 && y == 0 {
                [200, 50, 50, 255]
            } else {
                [0, 0, 0, 0]
            }
        });
        let out = DropShadow::new(2, 2, 1)
            .with_opacity(1.0)
            .apply(&input, p_rgba(8, 8))
            .unwrap();
        let off = (2 * 8 + 2) * 4;
        let shadow_alpha = out.planes[0].data[off + 3];
        assert!(shadow_alpha > 0, "expected shadow alpha > 0 at (2,2)");
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
            format: PixelFormat::Rgb24,
            width: 4,
            height: 4,
        };
        let err = DropShadow::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("DropShadow"));
    }
}
