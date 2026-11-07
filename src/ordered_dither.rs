//! Ordered dither against an arbitrary threshold map, with a level count
//! per channel — ImageMagick's `-ordered-dither map[,levels…]` (issue
//! #15).
//!
//! A threshold map is a `w × h` tile of cell indices `0 … divisor − 1`.
//! Quantising a normalised sample `v` to `L` levels, the cell `t` steps
//! up one level at every `v = (t + 1 + divisor · k) / ((L − 1) · divisor +
//! 1)`, `k = 0 … L − 2`: the `L − 1` level boundaries are spread over the
//! tile so that a flat area of value `v` lights the right fraction of
//! cells. For two levels that is the classic `v > (t + 1) / (divisor +
//! 1)`. The rule was fixed black-box against `magick` 7.1.2 (ramps at 2,
//! 3 and 6 levels through `threshold`, `checks`, `o2x2`, `o4x4`;
//! `tests/magick_ordered_dither.rs`).
//!
//! Built-in maps: `threshold` (1 × 1), `checks` (2 × 2) and the Bayer
//! dispersed-dot maps `o2x2` / `o4x4` from the recurrence of
//! `docs/image/filter/dithering-kernels.md` §2, plus every other map
//! `magick -list threshold` names — `o3x3`, `o8x8`, the halftones
//! `h4x4a` / `h6x6a` / `h8x8a` / `h4x4o` / `h6x6o` / `h8x8o` / `h16x16o`
//! and the circles `c5x5b` / `c5x5w` / `c6x6b` / `c6x6w` / `c7x7b` /
//! `c7x7w` — whose cells and divisors were recovered black-box: a flat
//! 16-bit picture at each of 4 097 levels, tiled by the map, run through
//! `magick -ordered-dither <map>`, gives each cell's two-level switching
//! value `(t + 1) / (divisor + 1)`, from which the smallest consistent
//! divisor and every cell follow. `tests/magick_ordered_dither.rs` then
//! holds each map bit-exact at 2, 3 and 6 levels. Any other map can be
//! supplied through [`ThresholdMap::new`].

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// `(name, alias, side, divisor, cells)` of the maps recovered black-box
/// (see the module docs).
#[rustfmt::skip]
const MEASURED_MAPS: &[(&str, &str, u32, u32, &[u16])] = &[
    ("o3x3", "3x3", 3, 9, &[
        2, 6, 3,
        5, 0, 8,
        1, 7, 4,
    ]),
    ("o8x8", "8x8", 8, 64, &[
        0, 48, 12, 60, 3, 51, 15, 63,
        32, 16, 44, 28, 35, 19, 47, 31,
        8, 56, 4, 52, 11, 59, 7, 55,
        40, 24, 36, 20, 43, 27, 39, 23,
        2, 50, 14, 62, 1, 49, 13, 61,
        34, 18, 46, 30, 33, 17, 45, 29,
        10, 58, 6, 54, 9, 57, 5, 53,
        42, 26, 38, 22, 41, 25, 37, 21,
    ]),
    ("h4x4a", "4x1", 4, 8, &[
        3, 1, 6, 4,
        2, 0, 7, 5,
        6, 4, 3, 1,
        7, 5, 2, 0,
    ]),
    ("h6x6a", "6x1", 6, 18, &[
        13, 12, 9, 7, 1, 2,
        15, 17, 11, 6, 0, 3,
        14, 16, 10, 8, 5, 4,
        7, 1, 2, 13, 12, 9,
        6, 0, 3, 15, 17, 11,
        8, 5, 4, 14, 16, 10,
    ]),
    ("h8x8a", "8x1", 8, 32, &[
        12, 6, 7, 13, 16, 20, 21, 17,
        5, 0, 2, 8, 27, 30, 28, 22,
        4, 1, 3, 9, 26, 31, 29, 23,
        15, 11, 10, 14, 19, 25, 24, 18,
        16, 20, 21, 17, 12, 6, 7, 13,
        27, 30, 28, 22, 5, 0, 2, 8,
        26, 31, 29, 23, 4, 1, 3, 9,
        19, 25, 24, 18, 15, 11, 10, 14,
    ]),
    ("h4x4o", "", 4, 16, &[
        6, 12, 10, 3,
        11, 15, 13, 7,
        9, 14, 5, 1,
        4, 8, 2, 0,
    ]),
    ("h6x6o", "", 6, 36, &[
        6, 16, 26, 13, 8, 3,
        20, 28, 32, 30, 17, 10,
        23, 31, 35, 33, 24, 21,
        18, 29, 34, 27, 19, 9,
        7, 14, 25, 15, 5, 1,
        4, 12, 22, 11, 2, 0,
    ]),
    ("h8x8o", "", 8, 64, &[
        6, 20, 32, 42, 35, 18, 8, 3,
        15, 26, 50, 54, 48, 28, 13, 10,
        30, 46, 56, 60, 58, 44, 34, 22,
        40, 52, 59, 63, 61, 51, 39, 37,
        36, 43, 57, 62, 55, 45, 29, 21,
        14, 27, 47, 53, 49, 25, 16, 9,
        7, 17, 33, 41, 31, 19, 5, 1,
        4, 12, 24, 38, 23, 11, 2, 0,
    ]),
    ("h16x16o", "", 16, 256, &[
        3, 11, 23, 43, 71, 99, 135, 151, 149, 133, 97, 69, 41, 22, 10, 2,
        6, 15, 31, 51, 75, 103, 143, 159, 157, 141, 101, 73, 49, 30, 14, 5,
        18, 26, 39, 59, 91, 131, 167, 179, 177, 165, 129, 89, 57, 38, 25, 17,
        35, 47, 55, 79, 123, 175, 187, 203, 202, 186, 174, 121, 78, 54, 46, 34,
        63, 67, 83, 115, 163, 199, 211, 223, 222, 210, 198, 161, 113, 82, 66, 62,
        87, 95, 111, 155, 191, 215, 231, 239, 238, 230, 213, 189, 153, 110, 94, 86,
        107, 119, 147, 183, 207, 227, 243, 251, 250, 242, 225, 205, 181, 146, 118, 106,
        127, 139, 171, 195, 218, 234, 246, 255, 254, 245, 233, 217, 193, 170, 138, 126,
        125, 137, 169, 194, 219, 235, 247, 252, 253, 244, 232, 216, 192, 168, 136, 124,
        105, 117, 145, 182, 206, 226, 241, 248, 249, 240, 224, 204, 180, 144, 116, 104,
        85, 93, 109, 154, 190, 214, 228, 237, 236, 229, 212, 188, 152, 108, 92, 84,
        61, 65, 81, 114, 162, 197, 209, 220, 221, 208, 196, 160, 112, 80, 64, 60,
        33, 45, 53, 77, 122, 173, 185, 201, 200, 184, 172, 120, 76, 52, 44, 32,
        19, 27, 36, 58, 90, 130, 166, 178, 176, 164, 128, 88, 56, 37, 24, 16,
        7, 12, 28, 50, 74, 102, 142, 158, 156, 140, 100, 72, 48, 29, 13, 4,
        0, 8, 20, 42, 70, 98, 134, 150, 148, 132, 96, 68, 40, 21, 9, 1,
    ]),
    ("c5x5b", "c5x5", 5, 25, &[
        0, 20, 15, 14, 3,
        4, 16, 19, 18, 13,
        5, 20, 24, 23, 11,
        6, 17, 21, 22, 10,
        1, 7, 8, 9, 2,
    ]),
    ("c5x5w", "", 5, 25, &[
        24, 20, 9, 10, 21,
        19, 8, 5, 6, 11,
        18, 4, 0, 1, 12,
        17, 7, 3, 2, 13,
        23, 16, 15, 14, 22,
    ]),
    ("c6x6b", "c6x6", 6, 36, &[
        0, 4, 13, 12, 11, 3,
        5, 21, 27, 26, 20, 10,
        14, 28, 34, 33, 25, 19,
        15, 29, 35, 32, 24, 18,
        6, 22, 30, 31, 23, 9,
        1, 7, 16, 17, 8, 2,
    ]),
    ("c6x6w", "", 6, 36, &[
        35, 31, 22, 23, 24, 32,
        30, 14, 8, 9, 15, 25,
        21, 7, 1, 2, 10, 16,
        20, 6, 0, 3, 11, 17,
        29, 13, 5, 4, 12, 26,
        34, 28, 19, 18, 27, 33,
    ]),
    ("c7x7b", "c7x7", 7, 49, &[
        2, 8, 17, 27, 16, 7, 1,
        9, 23, 32, 38, 31, 22, 6,
        18, 33, 43, 47, 42, 30, 15,
        24, 39, 44, 48, 46, 37, 26,
        19, 34, 40, 45, 41, 28, 14,
        10, 20, 35, 36, 27, 21, 5,
        3, 11, 12, 25, 13, 4, 0,
    ]),
    ("c7x7w", "", 7, 49, &[
        46, 40, 31, 21, 32, 41, 47,
        39, 25, 16, 10, 17, 26, 42,
        30, 15, 5, 1, 6, 18, 33,
        24, 9, 4, 0, 2, 11, 22,
        29, 14, 8, 3, 7, 19, 34,
        38, 28, 13, 12, 20, 27, 43,
        45, 37, 36, 23, 35, 44, 48,
    ]),
];

/// A tiled threshold map: cell indices `0 … divisor − 1`, row-major.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThresholdMap {
    /// Tile width.
    pub width: u32,
    /// Tile height.
    pub height: u32,
    /// Number of distinct cells (levels of the map).
    pub divisor: u32,
    /// `width × height` cell indices, each `< divisor`.
    pub cells: Vec<u32>,
}

impl ThresholdMap {
    /// A map from its cells. Fails on an empty tile, a cell count that is
    /// not `width × height`, or a cell `≥ divisor`.
    pub fn new(width: u32, height: u32, divisor: u32, cells: Vec<u32>) -> Result<Self, Error> {
        if width == 0 || height == 0 || divisor == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: ThresholdMap needs a non-empty tile and divisor",
            ));
        }
        if cells.len() != (width * height) as usize {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: ThresholdMap {width}x{height} needs {} cells, got {}",
                width * height,
                cells.len()
            )));
        }
        if cells.iter().any(|&c| c >= divisor) {
            return Err(Error::invalid(
                "oxideav-image-filter: ThresholdMap cell >= divisor",
            ));
        }
        Ok(Self {
            width,
            height,
            divisor,
            cells,
        })
    }

    /// The Bayer dispersed-dot map of order `2^k` (`docs/image/filter/
    /// dithering-kernels.md` §2 recurrence; `n` is rounded up to a power
    /// of two, 2 … 64).
    pub fn bayer(n: u32) -> Self {
        let n = n.clamp(2, 64).next_power_of_two();
        let mut m = vec![0u32, 2, 3, 1];
        let mut k = 2u32;
        while k < n {
            let k2 = k * 2;
            let mut next = vec![0u32; (k2 * k2) as usize];
            for y in 0..k2 {
                for x in 0..k2 {
                    let base = m[((y % k) * k + x % k) as usize];
                    let add = [0u32, 2, 3, 1][((y / k) * 2 + x / k) as usize];
                    next[(y * k2 + x) as usize] = 4 * base + add;
                }
            }
            m = next;
            k = k2;
        }
        Self {
            width: n,
            height: n,
            divisor: n * n,
            cells: m,
        }
    }

    /// A built-in map by its `-ordered-dither` name or alias (`1x1`,
    /// `2x1`, `2x2`, `3x3`, `4x4`, `8x8`, `4x1`, `6x1`, `8x1`, `c5x5`,
    /// `c6x6`, `c7x7`); see [`Self::names`]. `None` for an unknown name.
    pub fn named(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "threshold" | "1x1" => Self {
                width: 1,
                height: 1,
                divisor: 1,
                cells: vec![0],
            },
            "checks" | "2x1" => Self {
                width: 2,
                height: 2,
                divisor: 2,
                cells: vec![0, 1, 1, 0],
            },
            "o2x2" | "2x2" => Self::bayer(2),
            "o4x4" | "4x4" => Self::bayer(4),
            other => {
                let (_, _, side, divisor, cells) = MEASURED_MAPS
                    .iter()
                    .find(|(n, a, ..)| *n == other || (!a.is_empty() && *a == other))?;
                Self {
                    width: *side,
                    height: *side,
                    divisor: *divisor,
                    cells: cells.iter().map(|&c| c as u32).collect(),
                }
            }
        })
    }

    /// The names [`Self::named`] knows (aliases aside).
    pub fn names() -> impl Iterator<Item = &'static str> {
        ["threshold", "checks", "o2x2", "o4x4"]
            .into_iter()
            .chain(MEASURED_MAPS.iter().map(|(n, ..)| *n))
    }

    fn cell(&self, x: u32, y: u32) -> u32 {
        self.cells[((y % self.height) * self.width + x % self.width) as usize]
    }
}

/// Ordered dither of every colour channel (alpha too when its level
/// count is set) to a per-channel number of levels.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct OrderedDither {
    /// The threshold tile.
    pub map: ThresholdMap,
    /// Levels per channel `[R, G, B]` (grey uses the first), `≥ 2`.
    pub levels: [u32; 3],
    /// Levels for alpha (`None`: alpha untouched).
    pub alpha_levels: Option<u32>,
}

impl OrderedDither {
    /// Two-level dither against `map`.
    pub fn new(map: ThresholdMap) -> Self {
        Self {
            map,
            levels: [2; 3],
            alpha_levels: None,
        }
    }

    /// One level count for every colour channel (`map,L`).
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = [levels.max(2); 3];
        self
    }

    /// Levels per colour channel (`map,R,G,B`).
    pub fn with_channel_levels(mut self, r: u32, g: u32, b: u32) -> Self {
        self.levels = [r.max(2), g.max(2), b.max(2)];
        self
    }

    /// Dither alpha too.
    pub fn with_alpha_levels(mut self, levels: u32) -> Self {
        self.alpha_levels = Some(levels.max(2));
        self
    }

    /// Parse an `-ordered-dither` argument: `name[,L[,G,B[,A]]]`.
    pub fn parse(spec: &str) -> Result<Self, Error> {
        let mut parts = spec.split(',').map(str::trim);
        let name = parts.next().unwrap_or("");
        let map = ThresholdMap::named(name).ok_or_else(|| {
            Error::unsupported(format!(
                "oxideav-image-filter: ordered-dither map '{name}' is not built in \
                 (supply other maps with ThresholdMap::new)"
            ))
        })?;
        let nums: Vec<u32> = parts
            .map(|p| {
                p.parse::<u32>().map_err(|_| {
                    Error::invalid(format!("oxideav-image-filter: bad level count '{p}'"))
                })
            })
            .collect::<Result<_, _>>()?;
        let mut d = Self::new(map);
        match nums.as_slice() {
            [] => {}
            [l] => d = d.with_levels(*l),
            [r, g, b] => d = d.with_channel_levels(*r, *g, *b),
            [r, g, b, a] => d = d.with_channel_levels(*r, *g, *b).with_alpha_levels(*a),
            _ => {
                return Err(Error::invalid(
                    "oxideav-image-filter: ordered-dither takes 1, 3 or 4 level counts",
                ));
            }
        }
        Ok(d)
    }

    /// Quantise normalised `v` at cell `t` to one of `levels` levels.
    #[inline]
    fn quantise(&self, v: f32, t: u32, levels: u32) -> f32 {
        // Binary64: a sample on a boundary must land on the same side as
        // the exact rational comparison.
        let d = self.map.divisor as f64;
        let n = (levels - 1) as f64 * d + 1.0;
        let v = if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 1.0) as f64
        };
        // A sample that is a 16-bit code (every 8- and 16-bit input is)
        // within binary32 rounding is taken as that exact code.
        let q = (v * 65535.0).round();
        let v = if (v * 65535.0 - q).abs() < 2e-3 {
            q / 65535.0
        } else {
            v
        };
        let steps = ((v * n - (t as f64 + 1.0)) / d).floor() + 1.0;
        (steps.clamp(0.0, (levels - 1) as f64) / (levels - 1) as f64) as f32
    }
}

impl PixelFilter for OrderedDither {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let mut img = if deep::is_float(params.format) {
            deep::FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        let (cc, ch, has_alpha) = (img.color_channels(), img.channels(), img.has_alpha());
        let w = img.width();
        for (i, px) in img.samples_mut().chunks_exact_mut(ch).enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            let t = self.map.cell(x, y);
            for (c, v) in px.iter_mut().enumerate().take(cc) {
                *v = self.quantise(*v, t, self.levels[c.min(2)]);
            }
            if has_alpha && let Some(l) = self.alpha_levels {
                px[3] = self.quantise(px[3], t, l);
            }
        }
        Ok(if deep::is_float(params.format) {
            img.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&img);
            f.pts = input.pts;
            f
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_recurrence_matches_the_staged_4x4() {
        let m = ThresholdMap::bayer(4);
        assert_eq!(
            m.cells,
            vec![0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
        );
        assert_eq!(ThresholdMap::bayer(2).cells, vec![0, 2, 3, 1]);
        assert_eq!(ThresholdMap::bayer(8).divisor, 64);
    }

    #[test]
    fn flat_fields_light_the_right_fraction_and_parse() {
        let d = OrderedDither::parse("o4x4").unwrap();
        // v = 0.5 → (t+1)/17 < 0.5 for t ≤ 7: half the cells.
        let lit = (0..16u32).filter(|&t| d.quantise(0.5, t, 2) == 1.0).count();
        assert_eq!(lit, 8);
        assert_eq!(d.quantise(0.0, 0, 2), 0.0);
        assert_eq!(d.quantise(1.0, 15, 2), 1.0);
        let d = OrderedDither::parse("threshold,3").unwrap();
        assert_eq!(d.quantise(0.34, 0, 3), 0.5);
        assert_eq!(d.quantise(0.32, 0, 3), 0.0);
        assert!(OrderedDither::parse("h4x4a").is_ok());
        assert!(OrderedDither::parse("h5x5z").is_err());
        assert!(ThresholdMap::new(2, 1, 2, vec![0, 2]).is_err());
        let d = OrderedDither::parse("o2x2,6,4,2,3").unwrap();
        assert_eq!(d.levels, [6, 4, 2]);
        assert_eq!(d.alpha_levels, Some(3));
    }
}
