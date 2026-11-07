//! `-shear XxY` (`Imagick::shearImage`), issue #12: shear along the
//! rows by `x_degrees`, then along the columns by `y_degrees`, onto a
//! background-filled canvas just large enough for the sheared picture.
//!
//! The geometry was measured black-box on `magick` 7.1.2 (white
//! rectangles on black at many sizes and angles, then `rose:`; see
//! `tests/magick_ops.rs`). With `tx = tan(x_degrees)`, `ty =
//! tan(y_degrees)` and the picture `W × H`:
//!
//! - **row pass**: the canvas is `W₁ = round(W + H·|tx|)` wide; row `r`
//!   is shifted so its left edge lies at `kx + (H/2 − r)·tx`, `kx =
//!   round(H·|tx| / 2)` — positive angles lean the top to the right;
//! - **column pass** (on the `W₁ × H` result): the canvas is `H₂ =
//!   round(h₂)` tall, `h₂ = |ty|·W + |1 − tx·ty|·H` being the height of
//!   the sheared parallelogram, and column `c` is shifted so its top edge
//!   lies at `round((h₂ − H) / 2) + (c − p)·ty`, with the pivot `p =
//!   W₁/2`, or `W₁/2 − 1` when the row pass's offset `kx` sits left of
//!   centre (`kx + W/2 < W₁/2`); positive angles push the right side
//!   down;
//! - halves round half up; each shift resamples linearly (a pixel straddling two output pixels
//!   splits by its overlap) and the uncovered part of an output pixel is
//!   background.
//!
//! Either pass alone, and most angle pairs, are bit-exact against
//! `magick`; some pairs with both angles positive land a fraction of a
//! pixel off in the column pass (59–62 dB on `rose:`), a placement detail
//! the measurements did not pin down.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// `-shear XxY`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shear {
    /// Row shear angle in degrees (`-90 < x < 90`).
    pub x_degrees: f64,
    /// Column shear angle in degrees.
    pub y_degrees: f64,
    /// Fill colour, RGBA normalised (`-background`; white by default).
    pub background: [f32; 4],
}

impl Shear {
    /// Shear by the two angles, white background.
    pub fn new(x_degrees: f64, y_degrees: f64) -> Self {
        Self {
            x_degrees,
            y_degrees,
            background: [1.0; 4],
        }
    }

    /// Set the fill colour.
    pub fn with_background(mut self, background: [f32; 4]) -> Self {
        self.background = background;
        self
    }
}

/// Shift a line of `n` pixels (`get(i)` gives pixel `i`'s samples) so its
/// leading edge lands at `offset` on an `m`-long output line.
#[allow(clippy::too_many_arguments)]
fn shift_line(
    n: usize,
    m: usize,
    ch: usize,
    offset: f64,
    get: impl Fn(usize) -> usize,
    src: &[f32],
    bg: &[f32],
    mut put: impl FnMut(usize, &[f32]),
) {
    let mut acc = vec![0.0f64; ch];
    for o in 0..m {
        acc.iter_mut().for_each(|a| *a = 0.0);
        let (lo, hi) = (o as f64 - offset, o as f64 + 1.0 - offset);
        let mut covered = 0.0;
        let first = lo.floor().max(0.0) as usize;
        let last = (hi.ceil().max(0.0) as usize).min(n);
        for i in first..last {
            let w = (hi.min(i as f64 + 1.0) - lo.max(i as f64)).max(0.0);
            if w > 0.0 {
                covered += w;
                let base = get(i);
                for (a, s) in acc.iter_mut().zip(&src[base..base + ch]) {
                    *a += w * *s as f64;
                }
            }
        }
        let rest = (1.0 - covered).max(0.0);
        let px: Vec<f32> = acc
            .iter()
            .zip(bg)
            .map(|(a, b)| (a + rest * *b as f64) as f32)
            .collect();
        put(o, &px);
    }
}

impl Shear {
    fn run(&self, img: FloatImage) -> Result<FloatImage, Error> {
        let (tx, ty) = (
            self.x_degrees.to_radians().tan(),
            self.y_degrees.to_radians().tan(),
        );
        if !tx.is_finite() || !ty.is_finite() || self.x_degrees.abs() >= 90.0 {
            return Err(Error::invalid(
                "oxideav-image-filter: Shear angles must lie strictly between -90 and 90 degrees",
            ));
        }
        let (w, h, ch) = (img.width() as usize, img.height() as usize, img.channels());
        let b = self.background;
        let bg: Vec<f32> = match ch {
            1 => vec![0.212_656 * b[0] + 0.715_158 * b[1] + 0.072_186 * b[2]],
            3 => b[..3].to_vec(),
            _ => b.to_vec(),
        };
        let w1 = (w as f64 + h as f64 * tx.abs()).round() as usize;
        let h2 = (ty.abs() * w as f64 + (1.0 - tx * ty).abs() * h as f64).round() as usize;
        if w1 == 0 || h2 == 0 || w1 > 1 << 16 || h2 > 1 << 16 {
            return Err(Error::invalid(
                "oxideav-image-filter: Shear result extent out of range",
            ));
        }
        let src = img.samples();
        let mut mid = vec![0.0f32; w1 * h * ch];
        // Half the shear's spread, rounded half up (measured).
        let kx = (h as f64 * tx.abs() / 2.0 + 0.5).floor();
        for r in 0..h {
            let off = kx + (h as f64 / 2.0 - r as f64) * tx;
            shift_line(
                w,
                w1,
                ch,
                off,
                |i| (r * w + i) * ch,
                src,
                &bg,
                |o, px| {
                    mid[(r * w1 + o) * ch..][..ch].copy_from_slice(px);
                },
            );
        }
        let mut out = vec![0.0f32; w1 * h2 * ch];
        // Half the parallelogram's extra height, rounded half up; the
        // column pass pivots on the canvas centre, one column further
        // left when the row pass rounded its own half-spread down on an
        // odd spread (both measured).
        let h2_exact = ty.abs() * w as f64 + (1.0 - tx * ty).abs() * h as f64;
        let ky = ((h2_exact - h as f64) / 2.0 + 0.5).floor();
        let half = w1 as f64 / 2.0;
        let centre = if kx + (w as f64) / 2.0 < half {
            half - 1.0
        } else {
            half
        };
        for c in 0..w1 {
            let off = ky + (c as f64 - centre) * ty;
            shift_line(
                h,
                h2,
                ch,
                off,
                |i| (i * w1 + c) * ch,
                &mid,
                &bg,
                |o, px| {
                    out[(o * w1 + c) * ch..][..ch].copy_from_slice(px);
                },
            );
        }
        FloatImage::from_samples(w1 as u32, h2 as u32, ch, out)
    }
}

impl PixelFilter for Shear {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| self.run(img))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_pass_matches_the_measured_edges() {
        // 6 × 6 white on black, 20°: row 0 starts at 1 + 3·tan 20°.
        let img = FloatImage::from_samples(6, 6, 1, vec![1.0; 36]).unwrap();
        let out = Shear::new(20.0, 0.0)
            .with_background([0.0, 0.0, 0.0, 1.0])
            .run(img)
            .unwrap();
        assert_eq!((out.width(), out.height()), (8, 6));
        let row0 = &out.samples()[..8];
        assert!(row0[1].abs() < 1e-6 && (row0[2] - 0.9081).abs() < 1e-3);
        let row5 = &out.samples()[40..48];
        assert!((row5[0] - 0.7279).abs() < 1e-3 && (row5[6] - 0.2721).abs() < 1e-3);
    }
}
