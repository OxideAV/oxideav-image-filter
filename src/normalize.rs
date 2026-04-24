//! Auto-levels: scan the image once to find its black and white point,
//! then stretch the luminance range to fill `[0, 255]`. A small clip
//! percentage can be used to ignore extreme outliers (sensor noise,
//! stuck pixels, etc.).
//!
//! Clean-room implementation of the standard two-pass normalise
//! operation — nothing borrowed from any other library. Parameter
//! semantics mirror ImageMagick's `-normalize`; the `low_clip` /
//! `high_clip` knobs are explicit here because IM's defaults aren't
//! universally useful and letting callers pick is clearer.

use crate::{is_supported, level::build_level_lut, tonal_lut::apply_tone_lut, ImageFilter};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Two-pass auto-levels.
///
/// 1. Scan the image's "tone" data (luma for YUV, R+G+B mean for RGB,
///    or the sample itself for Gray) to build a 256-bin histogram.
/// 2. Walk the histogram from both ends to find the cut-off samples
///    below `low_clip` of the total pixel count and above
///    `(1 - high_clip)`.
/// 3. Apply a [`Level`](crate::Level)-equivalent LUT with `gamma = 1.0`
///    so everything in `[lo, hi]` stretches to `[0, 255]`.
///
/// For RGB/RGBA the stretch is applied per channel with the same
/// `[lo, hi]` endpoints (common practice — gives perceptually correct
/// results without re-tinting the image).
#[derive(Clone, Copy, Debug)]
pub struct Normalize {
    low_clip: f32,
    high_clip: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Normalize {
    /// Build a normaliser. `low_clip` and `high_clip` are clamped to
    /// `0.0..0.5` (values ≥ 0.5 would meet in the middle and flatten
    /// the image entirely).
    pub fn new(low_clip: f32, high_clip: f32) -> Self {
        Self {
            low_clip: low_clip.clamp(0.0, 0.5),
            high_clip: high_clip.clamp(0.0, 0.5),
        }
    }
}

impl ImageFilter for Normalize {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Normalize does not yet handle {:?}",
                input.format
            )));
        }
        let (lo, hi) = scan_range(input, self.low_clip, self.high_clip);
        let lut = build_level_lut(lo, hi, 1.0);
        let mut out = input.clone();
        apply_tone_lut(&mut out, &lut);
        Ok(out)
    }
}

fn scan_range(frame: &VideoFrame, low_clip: f32, high_clip: f32) -> (u8, u8) {
    let mut hist = [0u32; 256];
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut total: u32 = 0;

    match frame.format {
        PixelFormat::Gray8 | PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            let p = &frame.planes[0];
            for y in 0..h {
                for v in &p.data[y * p.stride..y * p.stride + w] {
                    hist[*v as usize] += 1;
                    total += 1;
                }
            }
        }
        PixelFormat::Rgb24 => {
            let p = &frame.planes[0];
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + 3 * w];
                for x in 0..w {
                    // Use the per-pixel Rec. 601 luma proxy so the
                    // histogram reflects perceived brightness.
                    let r = row[3 * x] as u32;
                    let g = row[3 * x + 1] as u32;
                    let b = row[3 * x + 2] as u32;
                    let y_ = (r * 2 + g * 5 + b) / 8; // ≈ 0.25R+0.625G+0.125B
                    hist[y_.min(255) as usize] += 1;
                    total += 1;
                }
            }
        }
        PixelFormat::Rgba => {
            let p = &frame.planes[0];
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + 4 * w];
                for x in 0..w {
                    let r = row[4 * x] as u32;
                    let g = row[4 * x + 1] as u32;
                    let b = row[4 * x + 2] as u32;
                    let y_ = (r * 2 + g * 5 + b) / 8;
                    hist[y_.min(255) as usize] += 1;
                    total += 1;
                }
            }
        }
        _ => {}
    }

    if total == 0 {
        return (0, 255);
    }
    let lo_budget = (total as f32 * low_clip).round() as u32;
    let hi_budget = (total as f32 * high_clip).round() as u32;

    // Walk from low end.
    let mut acc = 0u32;
    let mut lo: u8 = 0;
    for (i, h_) in hist.iter().enumerate() {
        acc += *h_;
        if acc > lo_budget {
            lo = i as u8;
            break;
        }
    }

    // Walk from high end.
    acc = 0;
    let mut hi: u8 = 255;
    for (i, h_) in hist.iter().enumerate().rev() {
        acc += *h_;
        if acc > hi_budget {
            hi = i as u8;
            break;
        }
    }

    // Protect against a flat image that would give lo >= hi.
    if hi <= lo {
        if lo < 255 {
            hi = lo.saturating_add(1);
        } else {
            lo = 254;
            hi = 255;
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{TimeBase, VideoPlane};

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
    fn normalize_stretches_midrange_image() {
        // Samples sit in [64, 192]; after normalize the min must be 0
        // and the max must be 255 (approx).
        let input = gray(8, 8, |x, y| 64 + (x * 16 + y * 16) as u8 / 2);
        let out = Normalize::default().apply(&input).unwrap();
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0, "min = {min}");
        assert_eq!(max, 255, "max = {max}");
    }

    #[test]
    fn normalize_is_idempotent_on_already_stretched() {
        let input = gray(4, 4, |x, y| ((x + y * 4) * 17) as u8);
        let once = Normalize::default().apply(&input).unwrap();
        let twice = Normalize::default().apply(&once).unwrap();
        assert_eq!(once.planes[0].data, twice.planes[0].data);
    }

    #[test]
    fn normalize_flat_image_safe() {
        // Flat input → no useful range; result should still build without panic.
        let input = gray(4, 4, |_, _| 120);
        let _ = Normalize::default().apply(&input).unwrap();
    }

    #[test]
    fn normalize_rgba_preserves_alpha() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = 64 + (i as u8) * 2;
                [v, v, v, 77]
            })
            .collect();
        let input = VideoFrame {
            format: PixelFormat::Rgba,
            width: 4,
            height: 4,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Normalize::default().apply(&input).unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], 77);
        }
    }
}
