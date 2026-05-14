//! Watermark — overlay a secondary image onto the source at a given
//! position with a configurable opacity multiplier.
//!
//! Two-input filter:
//!
//! - `src`: the base image (input port 0).
//! - `dst`: the watermark / overlay (input port 1).
//!
//! Output is the same shape as `src`. The overlay is positioned at
//! `(offset_x, offset_y)` (top-left corner, in `src` coordinates); the
//! intersection with the `src` rectangle is composed with a straight
//! over operator scaled by `opacity ∈ [0, 1]`.
//!
//! Unlike the [`Composite`](crate::Composite) family, this filter does
//! **not** require shape parity between the two inputs — the overlay
//! can be smaller (the usual case) or larger (clipped to the source).
//!
//! # Pixel formats
//!
//! `Rgb24`: opacity scales the overlay-vs-base linear blend; the
//! overlay is treated as fully opaque (no alpha channel to read).
//!
//! `Rgba`: opacity multiplies the per-pixel alpha of the overlay; the
//! base alpha is left intact. Straight-alpha throughout.
//!
//! `Gray8`: same algebra as `Rgb24` on a single luminance channel.
//!
//! Both inputs must share the same `PixelFormat`. Other formats return
//! [`Error::unsupported`].

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Watermark overlay filter.
#[derive(Clone, Copy, Debug)]
pub struct Watermark {
    /// Top-left placement of the overlay inside the source image.
    pub offset_x: i32,
    pub offset_y: i32,
    /// `[0, 1]` strength multiplier.
    pub opacity: f32,
    /// Overlay's intrinsic width / height in pixels (the two-input
    /// adapter only carries `src`'s stream params, so the filter needs
    /// the overlay shape supplied up front).
    pub overlay_width: u32,
    pub overlay_height: u32,
}

impl Default for Watermark {
    fn default() -> Self {
        Self {
            offset_x: 0,
            offset_y: 0,
            opacity: 1.0,
            overlay_width: 0,
            overlay_height: 0,
        }
    }
}

impl Watermark {
    /// Construct a watermark with the overlay's intrinsic shape and
    /// a placement offset.
    pub fn new(offset_x: i32, offset_y: i32, overlay_width: u32, overlay_height: u32) -> Self {
        Self {
            offset_x,
            offset_y,
            opacity: 1.0,
            overlay_width,
            overlay_height,
        }
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

impl TwoInputImageFilter for Watermark {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, false),
            PixelFormat::Rgb24 => (3, false),
            PixelFormat::Rgba => (4, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Watermark requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if self.overlay_width == 0 || self.overlay_height == 0 {
            return Err(Error::invalid(
                "Watermark: overlay_width / overlay_height must both be > 0",
            ));
        }

        let sw = params.width as usize;
        let sh = params.height as usize;
        let ow = self.overlay_width as usize;
        let oh = self.overlay_height as usize;
        let row = sw * bpp;

        let sp = &src.planes[0];
        let dp = &dst.planes[0];
        let mut data = vec![0u8; row * sh];
        // Copy `src` into the output buffer first.
        for y in 0..sh {
            let s = &sp.data[y * sp.stride..y * sp.stride + row];
            data[y * row..(y + 1) * row].copy_from_slice(s);
        }

        let opacity = self.opacity.clamp(0.0, 1.0);
        if opacity == 0.0 {
            return Ok(VideoFrame {
                pts: src.pts,
                planes: vec![VideoPlane { stride: row, data }],
            });
        }

        // Compute the overlap rectangle in src coordinates and the
        // matching origin in dst coordinates.
        let dst_x0 = self.offset_x.max(0) as usize;
        let dst_y0 = self.offset_y.max(0) as usize;
        let ov_x0 = (-self.offset_x).max(0) as usize;
        let ov_y0 = (-self.offset_y).max(0) as usize;
        if dst_x0 >= sw || dst_y0 >= sh || ov_x0 >= ow || ov_y0 >= oh {
            return Ok(VideoFrame {
                pts: src.pts,
                planes: vec![VideoPlane { stride: row, data }],
            });
        }
        let cw = (sw - dst_x0).min(ow - ov_x0);
        let ch = (sh - dst_y0).min(oh - ov_y0);

        for y in 0..ch {
            let dst_row = &mut data[(dst_y0 + y) * row..(dst_y0 + y + 1) * row];
            let ov_row = &dp.data[(ov_y0 + y) * dp.stride..(ov_y0 + y + 1) * dp.stride];
            for x in 0..cw {
                let d_base = (dst_x0 + x) * bpp;
                let o_base = (ov_x0 + x) * bpp;
                let a = if has_alpha {
                    (ov_row[o_base + 3] as f32 / 255.0) * opacity
                } else {
                    opacity
                };
                let inv = 1.0 - a;
                for c in 0..bpp.min(3) {
                    let s = dst_row[d_base + c] as f32;
                    let o = ov_row[o_base + c] as f32;
                    dst_row[d_base + c] = (s * inv + o * a).round().clamp(0.0, 255.0) as u8;
                }
                // alpha (index 3) on the base is preserved by design.
            }
        }

        Ok(VideoFrame {
            pts: src.pts,
            planes: vec![VideoPlane { stride: row, data }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, fill: [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&fill);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }

    fn rgba(w: u32, h: u32, fill: [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&fill);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn opacity_zero_is_passthrough() {
        let s = rgb(4, 4, [100, 150, 200]);
        let d = rgb(2, 2, [0, 0, 0]);
        let out = Watermark::new(0, 0, 2, 2)
            .with_opacity(0.0)
            .apply_two(&s, &d, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, s.planes[0].data);
    }

    #[test]
    fn opacity_one_overwrites_in_overlap() {
        let s = rgb(4, 4, [100, 150, 200]);
        let d = rgb(2, 2, [0, 255, 0]);
        let out = Watermark::new(1, 1, 2, 2)
            .with_opacity(1.0)
            .apply_two(&s, &d, p_rgb(4, 4))
            .unwrap();
        // Overlap pixels (1,1) (2,1) (1,2) (2,2) ⇒ overlay colour.
        for &(x, y) in &[(1u32, 1u32), (2, 1), (1, 2), (2, 2)] {
            let base = ((y * 4 + x) * 3) as usize;
            let p = &out.planes[0].data[base..base + 3];
            assert_eq!(p, &[0, 255, 0], "overlay at ({x},{y})");
        }
        // Pixel (0, 0) is outside the overlay ⇒ stays at [100, 150, 200].
        assert_eq!(&out.planes[0].data[0..3], &[100, 150, 200]);
    }

    #[test]
    fn negative_offset_clips_overlay() {
        // Overlay is 4x4 starting at (-2, -2) ⇒ only its bottom-right
        // 2x2 lands inside a 4x4 src.
        let s = rgb(4, 4, [100, 100, 100]);
        let d = rgb(4, 4, [200, 0, 0]);
        let out = Watermark::new(-2, -2, 4, 4)
            .with_opacity(1.0)
            .apply_two(&s, &d, p_rgb(4, 4))
            .unwrap();
        // (0, 0) … (1, 1) ⇒ overlay; (2, 2) … (3, 3) ⇒ source.
        for y in 0..4 {
            for x in 0..4 {
                let base = ((y * 4 + x) * 3) as usize;
                let p = &out.planes[0].data[base..base + 3];
                if x < 2 && y < 2 {
                    assert_eq!(p, &[200, 0, 0], "overlay at ({x},{y})");
                } else {
                    assert_eq!(p, &[100, 100, 100], "source at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn rgba_alpha_blends() {
        let s = rgba(2, 2, [200, 0, 0, 255]);
        let d = rgba(2, 2, [0, 0, 200, 128]); // half-alpha overlay
        let out = Watermark::new(0, 0, 2, 2)
            .with_opacity(1.0)
            .apply_two(
                &s,
                &d,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        // a = 128/255 ≈ 0.5, so R ≈ 100, B ≈ 100, A preserved at 255.
        for i in 0..4 {
            let p = &out.planes[0].data[i * 4..i * 4 + 4];
            assert!((p[0] as i16 - 100).abs() <= 2);
            assert_eq!(p[1], 0);
            assert!((p[2] as i16 - 100).abs() <= 2);
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn rejects_zero_overlay_shape() {
        let s = rgb(4, 4, [0, 0, 0]);
        let d = rgb(2, 2, [0, 0, 0]);
        let err = Watermark::default()
            .apply_two(&s, &d, p_rgb(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("overlay_width"));
    }
}
