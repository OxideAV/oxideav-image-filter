//! `-polaroid` / `polaroidImage` (issue #12): the picture as an instant
//! print — a white card with a deeper bottom margin, turned by an
//! angle, casting a soft shadow on a transparent ground.
//!
//! 1. **Card.** A border `b` (a twentieth of the longer side, at least
//!    three pixels, unless given) is added on the left, top and right
//!    and four times that below.
//! 2. **Turn.** The card is rotated clockwise by the angle about its
//!    centre onto a transparent canvas just large enough to hold it and
//!    its shadow; every output pixel is read back through the inverse
//!    rotation with bilinear weights on premultiplied colour, so the
//!    card's edge is antialiased and nothing fringes.
//! 3. **Shadow.** The card's coverage, moved `b / 2` right and down and
//!    blurred with `σ = b / 3`, is laid under it as black at half
//!    opacity.
//!
//! The result is RGBA and larger than the input.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// Turn a picture into an instant-print card.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polaroid {
    /// Clockwise turn in degrees.
    pub angle_degrees: f32,
    /// Border width in pixels (`None`: a twentieth of the longer side).
    pub border: Option<u32>,
    /// Cast a shadow.
    pub shadow: bool,
}

impl Polaroid {
    /// Widest border, in pixels (a wider request is held to this).
    pub const MAX_BORDER: u32 = 512;

    /// A card turned by `angle_degrees`.
    pub fn new(angle_degrees: f32) -> Self {
        Self {
            angle_degrees,
            border: None,
            shadow: true,
        }
    }

    /// Set the border width.
    pub fn with_border(mut self, border: u32) -> Self {
        self.border = Some(border);
        self
    }

    /// Cast a shadow or not.
    pub fn with_shadow(mut self, shadow: bool) -> Self {
        self.shadow = shadow;
        self
    }

    fn border_for(&self, w: usize, h: usize) -> usize {
        self.border
            .unwrap_or(((w.max(h) / 20) as u32).max(3))
            .min(Self::MAX_BORDER) as usize
    }

    /// `(sin, cos)` of the turn, the card's extent and the canvas's.
    fn geometry(&self, w: usize, h: usize) -> (f32, f32, (usize, usize), (usize, usize)) {
        let b = self.border_for(w, h);
        let (cw, chh) = (w + 2 * b, h + 5 * b);
        let angle = if self.angle_degrees.is_finite() {
            self.angle_degrees.to_radians()
        } else {
            0.0
        };
        let (sin, cos) = angle.sin_cos();
        let margin = if self.shadow { b } else { 0 };
        let (fw, fh) = (cw as f32, chh as f32);
        let ow = (fw * cos.abs() + fh * sin.abs()).ceil() as usize + 2 + margin;
        let oh = (fw * sin.abs() + fh * cos.abs()).ceil() as usize + 2 + margin;
        (sin, cos, (cw, chh), (ow, oh))
    }

    /// The extent of the card made from a `width × height` picture.
    pub fn output_extent(&self, width: u32, height: u32) -> (u32, u32) {
        if width == 0 || height == 0 {
            return (width, height);
        }
        let (_, _, _, (ow, oh)) = self.geometry(width as usize, height as usize);
        (ow as u32, oh as u32)
    }

    /// The card for a float picture (RGBA out).
    pub fn apply_float(&self, img: &FloatImage) -> FloatImage {
        let src = img.to_rgba();
        let (w, h) = (src.width() as usize, src.height() as usize);
        if w == 0 || h == 0 {
            return src;
        }
        let b = self.border_for(w, h);
        let (sin, cos, (cw, chh), (ow, oh)) = self.geometry(w, h);
        // 1. The card, premultiplied.
        let mut card = vec![1.0f32; cw * chh * 4];
        for y in 0..h {
            let dst = &mut card[((y + b) * cw + b) * 4..][..w * 4];
            for (d, s) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.samples()[y * w * 4..].as_chunks::<4>().0)
            {
                // The picture over the white card.
                let a = if s[3].is_finite() {
                    s[3].clamp(0.0, 1.0)
                } else {
                    0.0
                };
                for (o, v) in d[..3].iter_mut().zip(s) {
                    *o = v * a + (1.0 - a);
                }
            }
        }
        // 2. The turn.
        let margin = if self.shadow { b } else { 0 };
        let (fw, fh) = (cw as f32, chh as f32);
        let (ocx, ocy) = ((ow - margin) as f32 / 2.0, (oh - margin) as f32 / 2.0);
        let tap = |x: i64, y: i64| -> [f32; 4] {
            if x < 0 || y < 0 || x >= cw as i64 || y >= chh as i64 {
                [0.0; 4]
            } else {
                let i = (y as usize * cw + x as usize) * 4;
                [card[i], card[i + 1], card[i + 2], card[i + 3]]
            }
        };
        let mut turned = vec![0.0f32; ow * oh * 4];
        crate::exec::par_rows(&mut turned, ow * 4, |y, row| {
            for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                // Output pixel centre back through the rotation.
                let (dx, dy) = (x as f32 + 0.5 - ocx, y as f32 + 0.5 - ocy);
                let u = dx * cos + dy * sin + fw / 2.0 - 0.5;
                let v = -dx * sin + dy * cos + fh / 2.0 - 0.5;
                let (x0, y0) = (u.floor(), v.floor());
                let (tx, ty) = (u - x0, v - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                for (wgt, sx, sy) in [
                    ((1.0 - tx) * (1.0 - ty), x0, y0),
                    (tx * (1.0 - ty), x0 + 1, y0),
                    ((1.0 - tx) * ty, x0, y0 + 1),
                    (tx * ty, x0 + 1, y0 + 1),
                ] {
                    for (o, s) in px.iter_mut().zip(tap(sx, sy)) {
                        *o += wgt * s;
                    }
                }
            }
        });
        // 3. The shadow under the card.
        if self.shadow {
            let alpha: Vec<f32> = turned.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
            let shift = b / 2;
            let mut moved = vec![0.0f32; ow * oh];
            for y in shift..oh {
                moved[y * ow + shift..(y + 1) * ow]
                    .copy_from_slice(&alpha[(y - shift) * ow..(y - shift + 1) * ow - shift]);
            }
            let soft = crate::tone_auto::blur_plane(&moved, ow as u32, oh as u32, b as f32 / 3.0);
            for (px, s) in turned.as_chunks_mut::<4>().0.iter_mut().zip(&soft) {
                // Premultiplied black at half opacity, under.
                px[3] += (1.0 - px[3]) * 0.5 * s;
            }
        }
        // Back to straight alpha.
        for px in turned.as_chunks_mut::<4>().0 {
            if px[3] > 1e-6 {
                let a = px[3];
                px[..3].iter_mut().for_each(|v| *v /= a);
            } else {
                px.fill(0.0);
            }
        }
        FloatImage::from_samples(ow as u32, oh as u32, 4, turned)
            .expect("canvas matches its extent")
    }
}

impl PixelFilter for Polaroid {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn output_format(&self, input: PixelFormat) -> PixelFormat {
        if deep::is_float(input) {
            PixelFormat::RgbaF32Le
        } else {
            PixelFormat::Rgba
        }
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| Ok(self.apply_float(&img)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red(w: u32, h: u32) -> FloatImage {
        FloatImage::from_samples(w, h, 3, [1.0, 0.0, 0.0].repeat((w * h) as usize)).unwrap()
    }

    #[test]
    fn an_unturned_card_has_the_stated_margins() {
        let out = Polaroid::new(0.0)
            .with_border(4)
            .with_shadow(false)
            .apply_float(&red(20, 10));
        // Card 28 × 30 on a canvas one pixel larger all round.
        assert_eq!((out.width(), out.height(), out.channels()), (30, 32, 4));
        assert_eq!(out.pixel(0, 0), &[0.0; 4]);
        assert_eq!(out.pixel(2, 2), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out.pixel(5, 5), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.pixel(24, 14), &[1.0, 0.0, 0.0, 1.0]);
        // Below the picture: four borders of white.
        assert_eq!(out.pixel(10, 15), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out.pixel(10, 30), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn turning_keeps_the_card_area_and_the_shadow_falls_down_right() {
        let flat = Polaroid::new(0.0)
            .with_border(4)
            .with_shadow(false)
            .apply_float(&red(20, 10));
        let coverage = |i: &FloatImage| -> f32 { i.samples().iter().skip(3).step_by(4).sum() };
        for angle in [10.0, -25.0, 90.0] {
            let out = Polaroid::new(angle)
                .with_border(4)
                .with_shadow(false)
                .apply_float(&red(20, 10));
            assert!((coverage(&out) - coverage(&flat)).abs() < 2.0, "{angle}");
            assert!(out.samples().iter().all(|v| (0.0..=1.0 + 1e-5).contains(v)));
        }
        let out = Polaroid::new(0.0).with_border(6).apply_float(&red(20, 10));
        let (w, h) = (out.width(), out.height());
        // Shadow only: dark, translucent, right of and below the card.
        let s = out.pixel(w - 5, h / 2);
        assert!(s[3] > 0.1 && s[3] < 0.6 && s[0] < 0.05, "{s:?}");
        assert!(out.pixel(w / 2, h - 5)[3] > 0.1);
        assert!(out.pixel(0, 0)[3] < 0.01);
        // The card itself is untouched by its shadow.
        assert_eq!(out.pixel(3, 3), &[1.0, 1.0, 1.0, 1.0]);
    }
}
