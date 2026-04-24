//! Posterisation — reduce each channel to `N` evenly spaced intensity
//! levels. Classic "poster" look.
//!
//! Clean-room formula from the textbook quantise-and-expand
//! definition:
//!
//! ```text
//! bucket = floor(v * N / 256)          // in [0, N-1]
//! v'     = bucket * 255 / (N - 1)      // remapped to full range
//! ```
//!
//! Matches ImageMagick's `-posterize N` parameter semantics.

use crate::{is_supported, tonal_lut::apply_tone_lut, ImageFilter};
use oxideav_core::{Error, VideoFrame};

/// Posterise each channel to `levels` intensity buckets.
///
/// `levels` must be at least 2; smaller values are clamped up to 2
/// (anything less degenerates to "one flat level" which is not a
/// useful operation).
///
/// Applied to Gray / R / G / B (alpha preserved for RGBA). For YUV
/// planar formats only the Y plane is posterised; chroma is left
/// alone so the colour content stays intact.
#[derive(Clone, Copy, Debug)]
pub struct Posterize {
    levels: u32,
}

impl Default for Posterize {
    fn default() -> Self {
        Self::new(4)
    }
}

impl Posterize {
    /// Build a posterise filter with `levels >= 2`.
    pub fn new(levels: u32) -> Self {
        Self {
            levels: levels.max(2),
        }
    }
}

impl ImageFilter for Posterize {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Posterize does not yet handle {:?}",
                input.format
            )));
        }
        let lut = build_posterize_lut(self.levels);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut);
        Ok(out)
    }
}

fn build_posterize_lut(levels: u32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let n = levels as f32;
    let step = 255.0 / (n - 1.0);
    for (i, entry) in lut.iter_mut().enumerate() {
        let bucket = ((i as f32) * n / 256.0).floor().min(n - 1.0);
        *entry = (bucket * step).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, TimeBase, VideoPlane};

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    #[test]
    fn posterize_two_levels_is_binary() {
        let input = gray(4, 4, |x, y| (x * 32 + y * 32) as u8);
        let out = Posterize::new(2).apply(&input).unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "got {v}");
        }
    }

    #[test]
    fn posterize_four_levels_matches_formula() {
        // With N=4 the step is 85; buckets are {0, 85, 170, 255}.
        let input = gray(4, 1, |x, _| match x {
            0 => 10,
            1 => 100,
            2 => 180,
            _ => 250,
        });
        let out = Posterize::new(4).apply(&input).unwrap();
        let vs = &out.planes[0].data;
        let allowed = [0u8, 85, 170, 255];
        for v in vs {
            assert!(allowed.contains(v), "got {v}");
        }
        // The smallest input should bucket to 0; the largest to 255.
        assert_eq!(vs[0], 0);
        assert_eq!(vs[3], 255);
    }

    #[test]
    fn posterize_256_levels_is_identity() {
        let input = gray(4, 4, |x, y| ((x + y * 4) * 15) as u8);
        let out = Posterize::new(256).apply(&input).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn posterize_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|_| [100u8, 100, 100, 200]).collect();
        let input = VideoFrame {
            format: PixelFormat::Rgba,
            width: 4,
            height: 4,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Posterize::new(3).apply(&input).unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 200);
        }
    }
}
