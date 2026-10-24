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
//! `docs/image/filter/dithering-kernels.md` §2. ImageMagick's other named
//! maps (`o3x3`, its own `o8x8`, the `h…` halftone and `c…` circle maps)
//! are data in its `thresholds.xml`, which is not staged; any of them can
//! be supplied through [`ThresholdMap::new`].

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

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

    /// A built-in map by its `-ordered-dither` name: `threshold`
    /// (`1x1`), `checks` (`2x2`), `o2x2` (`2x2`), `o4x4` (`4x4`); `None`
    /// for names whose tables are not staged.
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
            _ => return None,
        })
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
                 (threshold, checks, o2x2, o4x4; supply other maps with ThresholdMap::new)"
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
        let d = self.map.divisor as f32;
        let n = (levels - 1) as f32 * d + 1.0;
        let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        let steps = ((v * n - (t as f32 + 1.0)) / d).floor() + 1.0;
        steps.clamp(0.0, (levels - 1) as f32) / (levels - 1) as f32
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
        assert!(OrderedDither::parse("h4x4a").is_err());
        assert!(ThresholdMap::new(2, 1, 2, vec![0, 2]).is_err());
        let d = OrderedDither::parse("o2x2,6,4,2,3").unwrap();
        assert_eq!(d.levels, [6, 4, 2]);
        assert_eq!(d.alpha_levels, Some(3));
    }
}
