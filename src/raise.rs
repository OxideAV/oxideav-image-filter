//! `-raise` / `+raise` (`Imagick::raiseImage`): lighten and darken a
//! bevel round the picture's edge so it looks raised (or sunk), issue
//! #12. Measured black-box on `magick` 7.1.2 and matched on `rose:`
//! (`tests/magick_ops.rs`).
//!
//! The bevel is `width` columns on the left and right and `height` rows
//! on the top and bottom; the corners are split on the diagonals (the
//! left band wins a tie). Every channel, alpha included, is shaded:
//!
//! | band | raised | sunk |
//! |---|---|---|
//! | top | `v + (1 − v) · 120/255` | `v · 135/255` |
//! | left | `v + (1 − v) · 65/255` | `v · 190/255` |
//! | right | `v · 190/255` | `v + (1 − v) · 65/255` |
//! | bottom | `v · 135/255` | `v + (1 − v) · 120/255` |
//!
//! The picture must be wider than twice the bevel width and taller than
//! twice its height.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// The four bevel bands and the untouched interior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Band {
    Top,
    Left,
    Right,
    Bottom,
    Inside,
}

/// `-raise WxH` (raised) or `+raise WxH` (sunk).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Raise {
    /// Bevel width (left and right bands).
    pub width: u32,
    /// Bevel height (top and bottom bands).
    pub height: u32,
    /// `true` for `-raise`, `false` for `+raise`.
    pub raised: bool,
}

impl Raise {
    /// A raised bevel.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            raised: true,
        }
    }

    /// A sunk bevel (`+raise`).
    pub fn sunk(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            raised: false,
        }
    }

    /// Choose raised or sunk.
    pub fn with_raised(mut self, raised: bool) -> Self {
        self.raised = raised;
        self
    }

    fn band(&self, x: i64, y: i64, w: i64, h: i64) -> Band {
        let (bw, bh) = (self.width as i64, self.height as i64);
        if y < bh {
            if x < y {
                Band::Left
            } else if x >= w - y {
                Band::Right
            } else {
                Band::Top
            }
        } else if y >= h - bh {
            let k = h - 1 - y;
            if x <= k {
                Band::Left
            } else if x >= w - 1 - k {
                Band::Right
            } else {
                Band::Bottom
            }
        } else if x < bw {
            Band::Left
        } else if x >= w - bw {
            Band::Right
        } else {
            Band::Inside
        }
    }
}

const HIGHLIGHT: f32 = 120.0 / 255.0;
const ACCENT: f32 = 65.0 / 255.0;

fn lighten(v: f32, k: f32) -> f32 {
    v + (1.0 - v) * k
}

fn darken(v: f32, k: f32) -> f32 {
    v * (1.0 - k)
}

impl PixelFilter for Raise {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let (w, h) = (params.width as i64, params.height as i64);
        if w <= 2 * self.width as i64 || h <= 2 * self.height as i64 {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: Raise {}x{} needs a picture larger than twice the bevel, not {w}x{h}",
                self.width, self.height
            )));
        }
        deep::run_any(input, params, |mut img| {
            let ch = img.channels();
            for (i, px) in img.samples_mut().chunks_exact_mut(ch).enumerate() {
                let (x, y) = (i as i64 % w, i as i64 / w);
                let f: fn(f32, f32) -> f32;
                let k;
                match (self.band(x, y, w, h), self.raised) {
                    (Band::Inside, _) => continue,
                    (Band::Top, true) | (Band::Bottom, false) => {
                        f = lighten;
                        k = HIGHLIGHT;
                    }
                    (Band::Left, true) | (Band::Right, false) => {
                        f = lighten;
                        k = ACCENT;
                    }
                    (Band::Right, true) | (Band::Left, false) => {
                        f = darken;
                        k = ACCENT;
                    }
                    (Band::Bottom, true) | (Band::Top, false) => {
                        f = darken;
                        k = HIGHLIGHT;
                    }
                }
                for v in px.iter_mut() {
                    *v = f(*v, k);
                }
            }
            Ok(img)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_partition_matches_the_measured_map() {
        let r = Raise::new(1, 4);
        let map: Vec<String> = (0..9)
            .map(|y| {
                (0..5)
                    .map(|x| match r.band(x, y, 5, 9) {
                        Band::Top => 'T',
                        Band::Left => 'L',
                        Band::Right => 'R',
                        Band::Bottom => 'B',
                        Band::Inside => '.',
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            map,
            [
                "TTTTT", "LTTTR", "LLTRR", "LLLRR", "L...R", "LLLLR", "LLLRR", "LLBRR", "LBBBR"
            ]
        );
    }
}
