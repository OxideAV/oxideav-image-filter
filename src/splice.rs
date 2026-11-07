//! `-splice` (insert rows and columns of background) and `-chop`
//! (remove them), with gravity (issue #12). Both are bit-exact against
//! `magick` 7.1.2 (`tests/magick_ops.rs`).
//!
//! The region is a `width × height` box placed by gravity, as measured
//! on `magick`:
//!
//! - **chop** removes the columns `[x, x + width)` and rows `[y, y +
//!   height)` of the picture, where `x` is `x_offset` (west gravities),
//!   `(W − width) / 2 + x_offset` (north / centre / south) or `W − width
//!   − x_offset` (east), and the same down the rows; the box is clipped
//!   to the picture (a box wholly outside it leaves the picture as it
//!   is).
//! - **splice** inserts `width` background columns before column `x` and
//!   `height` background rows before row `y`, with `x` = `x_offset`
//!   (west), `trunc((W − width) / 2) + width / 2 + x_offset` (north /
//!   centre / south) or `W − x_offset` (east), and `y` = `y_offset`
//!   (north), `trunc((H − height) / 2) + height / 2 + y_offset` (centre /
//!   east — west gravity adds `width / 2` instead of `height / 2`, as
//!   measured) or `H − y_offset` (south). The
//!   output is `(W + width) × (H + height)`. A negative position starts
//!   the band off the picture: only its visible part is background. The
//!   picture's columns then follow from its first column (the last one
//!   repeats to fill the width), while its rows resume `height` rows
//!   back (the first row repeats) — both measured. A position at or past
//!   the far edge of the result is an error, as it is for `magick`.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

use crate::gravity_translate::Gravity;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// The start of a `size`-long box anchored by `gravity` on a `total`-long
/// axis (the slack is halved with truncation toward zero).
fn anchor(gravity: Gravity, total: i64, size: i64, offset: i64, horizontal: bool) -> i64 {
    let (start, centre) = if horizontal {
        (
            matches!(
                gravity,
                Gravity::NorthWest | Gravity::West | Gravity::SouthWest
            ),
            matches!(gravity, Gravity::North | Gravity::Centre | Gravity::South),
        )
    } else {
        (
            matches!(
                gravity,
                Gravity::NorthWest | Gravity::North | Gravity::NorthEast
            ),
            matches!(gravity, Gravity::West | Gravity::Centre | Gravity::East),
        )
    };
    if start {
        offset
    } else if centre {
        (total - size) / 2 + offset
    } else {
        total - size - offset
    }
}

/// `-splice WxH±X±Y`: insert background rows and columns.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Splice {
    /// Columns inserted.
    pub width: u32,
    /// Rows inserted.
    pub height: u32,
    /// Horizontal offset (see the [module docs](self)).
    pub x: i64,
    /// Vertical offset.
    pub y: i64,
    /// Placement anchor (`NorthWest` by default).
    pub gravity: Gravity,
    /// Fill colour, RGBA normalised (`-background`; white by default).
    pub background: [f32; 4],
}

impl Splice {
    /// Insert `width` columns before column `x` and `height` rows before
    /// row `y`, white.
    pub fn new(width: u32, height: u32, x: i64, y: i64) -> Self {
        Self {
            width,
            height,
            x,
            y,
            gravity: Gravity::NorthWest,
            background: [1.0; 4],
        }
    }

    /// Set the anchor.
    pub fn with_gravity(mut self, gravity: Gravity) -> Self {
        self.gravity = gravity;
        self
    }

    /// Set the fill colour.
    pub fn with_background(mut self, background: [f32; 4]) -> Self {
        self.background = background;
        self
    }
}

impl Splice {
    /// The column and row the bands are inserted before, on a `w × h`
    /// picture (see the [module docs](self)).
    pub fn position(&self, w: i64, h: i64) -> (i64, i64) {
        let (sw, sh) = (self.width as i64, self.height as i64);
        let centred = |total: i64, size: i64, half: i64| (total - size) / 2 + half;
        let x = match self.gravity {
            Gravity::NorthWest | Gravity::West | Gravity::SouthWest => self.x,
            Gravity::North | Gravity::Centre | Gravity::South => centred(w, sw, sw / 2) + self.x,
            _ => w - self.x,
        };
        let y = match self.gravity {
            Gravity::NorthWest | Gravity::North | Gravity::NorthEast => self.y,
            // West centres the rows with half the band *width*, as
            // measured; centre and east with half the band height.
            Gravity::West => centred(h, sh, sw / 2) + self.y,
            Gravity::Centre | Gravity::East => centred(h, sh, sh / 2) + self.y,
            _ => h - self.y,
        };
        (x, y)
    }
}

impl PixelFilter for Splice {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let (w, h) = (img.width() as i64, img.height() as i64);
            let (sw, sh) = (self.width as i64, self.height as i64);
            let (x, y) = self.position(w, h);
            if x >= w + sw || y >= h + sh {
                return Err(Error::invalid(format!(
                    "oxideav-image-filter: Splice at ({x}, {y}) lies beyond the {}x{} result",
                    w + sw,
                    h + sh
                )));
            }
            let (ow, oh) = (w + sw, h + sh);
            let ch = img.channels();
            let b = self.background;
            let bg: Vec<f32> = match ch {
                1 => vec![0.212_656 * b[0] + 0.715_158 * b[1] + 0.072_186 * b[2]],
                3 => b[..3].to_vec(),
                _ => b.to_vec(),
            };
            // Columns after a band that starts off the picture resume at
            // the picture's first column; rows resume `band` rows back
            // (so the first picture row repeats) — both measured.
            let src_of = |c: i64, start: i64, band: i64, columns: bool| -> Option<i64> {
                if c >= start && c < start + band {
                    None
                } else if c < start {
                    Some(c)
                } else if columns {
                    Some(c - band - start.min(0))
                } else {
                    Some(c - band)
                }
            };
            let mut out = deep::FloatImage::new(ow as u32, oh as u32, ch);
            for r in 0..oh {
                let sr = src_of(r, y, sh, false);
                for c in 0..ow {
                    let px = out.pixel_mut(c as u32, r as u32);
                    match (sr, src_of(c, x, sw, true)) {
                        (Some(sr), Some(sc)) => {
                            for (k, v) in px.iter_mut().enumerate() {
                                *v = img.sample_clamped(sc, sr, k);
                            }
                        }
                        _ => px.copy_from_slice(&bg),
                    }
                }
            }
            Ok(out)
        })
    }
}

/// `-chop WxH±X±Y`: remove rows and columns.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chop {
    /// Columns removed.
    pub width: u32,
    /// Rows removed.
    pub height: u32,
    /// Horizontal offset (see the [module docs](self)).
    pub x: i64,
    /// Vertical offset.
    pub y: i64,
    /// Placement anchor (`NorthWest` by default).
    pub gravity: Gravity,
}

impl Chop {
    /// Remove `width` columns from column `x` and `height` rows from row
    /// `y`.
    pub fn new(width: u32, height: u32, x: i64, y: i64) -> Self {
        Self {
            width,
            height,
            x,
            y,
            gravity: Gravity::NorthWest,
        }
    }

    /// Set the anchor.
    pub fn with_gravity(mut self, gravity: Gravity) -> Self {
        self.gravity = gravity;
        self
    }

    /// The removed column and row ranges on a `w × h` picture, clipped
    /// (`None` when the box misses the picture).
    fn ranges(&self, w: i64, h: i64) -> Option<((i64, i64), (i64, i64))> {
        let (cw, chh) = (self.width as i64, self.height as i64);
        let x = anchor(self.gravity, w, cw, self.x, true);
        let y = anchor(self.gravity, h, chh, self.y, false);
        let (x0, x1) = (x.max(0), (x + cw).min(w));
        let (y0, y1) = (y.max(0), (y + chh).min(h));
        // A zero-size side removes nothing along that axis (`magick`'s
        // handling of a zero-size side is not pinned).
        let xs = if cw == 0 { (0, 0) } else { (x0, x1.max(x0)) };
        let ys = if chh == 0 { (0, 0) } else { (y0, y1.max(y0)) };
        let misses = (cw > 0 && x1 <= x0) || (chh > 0 && y1 <= y0) || (cw == 0 && chh == 0);
        (!misses).then_some((xs, ys))
    }
}

impl PixelFilter for Chop {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        crate::transpose::PACKED
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let g = layout::plane_geometry(params.format, 0, params.width, params.height).ok_or_else(
            || {
                Error::unsupported(format!(
                    "oxideav-image-filter: Chop does not take {:?}",
                    params.format
                ))
            },
        )?;
        let (w, h, bpp) = (g.width as i64, g.height as i64, g.position_bytes);
        let Some(((x0, x1), (y0, y1))) = self.ranges(w, h) else {
            return Ok(input.clone());
        };
        let (ow, oh) = (w - (x1 - x0), h - (y1 - y0));
        if ow <= 0 || oh <= 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: Chop would remove the whole picture",
            ));
        }
        let src = &input.planes[0];
        let stride = ow as usize * bpp;
        let mut data = Vec::with_capacity(stride * oh as usize);
        for r in (0..h).filter(|r| *r < y0 || *r >= y1) {
            let row = &src.data[r as usize * src.stride..];
            data.extend_from_slice(&row[..x0 as usize * bpp]);
            data.extend_from_slice(&row[x1 as usize * bpp..w as usize * bpp]);
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;

    fn row(data: &[u8]) -> (VideoFrame, VideoStreamParams) {
        (
            VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: data.len(),
                    data: data.to_vec(),
                }],
            },
            VideoStreamParams {
                format: PixelFormat::Gray8,
                width: data.len() as u32,
                height: 1,
            },
        )
    }

    #[test]
    fn chop_and_splice_a_row() {
        let (f, p) = row(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let out = Chop::new(3, 0, 0, 0)
            .with_gravity(Gravity::Centre)
            .apply(&f, p)
            .unwrap();
        assert_eq!(out.planes[0].data, [0, 1, 2, 6, 7, 8]);
        let out = Chop::new(3, 0, -1, 0)
            .with_gravity(Gravity::NorthEast)
            .apply(&f, p)
            .unwrap();
        assert_eq!(out.planes[0].data, [0, 1, 2, 3, 4, 5, 6]);
        let (f, p) = row(&[10, 20, 30, 40, 50]);
        let out = Splice::new(3, 0, -1, 0)
            .with_background([0.0; 4])
            .apply(&f, p)
            .unwrap();
        assert_eq!(out.planes[0].data, [0, 0, 10, 20, 30, 40, 50, 50]);
        let out = Splice::new(2, 0, 1, 0)
            .with_gravity(Gravity::East)
            .apply(&f, p)
            .unwrap();
        assert_eq!(out.planes[0].data, [10, 20, 30, 40, 255, 255, 50]);
    }
}
