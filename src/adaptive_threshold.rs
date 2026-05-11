//! Local-mean-based adaptive threshold (ImageMagick `-thresh local`).
//!
//! For each sample, compare it against the mean of its `(2*radius+1)²`
//! local neighbourhood. If the sample is brighter than the local mean
//! by `offset` or more, emit `255`; otherwise emit `0`. The local mean
//! is computed with edge-clamped borders and a separable rolling-sum
//! pass per row, giving an `O(W*H)` cost regardless of radius.
//!
//! Use this instead of [`Threshold`](crate::threshold::Threshold) when
//! the input has spatially-varying illumination (e.g. a scanned page
//! with a dark left edge); a global cut-off would clip the bright
//! regions to all-white. The local mean tracks the lighting and keeps
//! the binarisation consistent across the frame.
//!
//! Output is always `Gray8` regardless of input. For `Rgb24` / `Rgba`
//! we collapse to a Rec. 601 luma proxy first; for planar YUV we use
//! plane 0 (the existing luma plane). YUV `Yuv420P` etc. work
//! directly.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Local-mean-based adaptive threshold.
///
/// `radius` is the half-extent of the square neighbourhood
/// (window side = `2*radius + 1`). `offset` is the threshold delta
/// applied above the local mean — bright biases the output toward
/// black, dark biases toward white.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveThreshold {
    pub radius: u32,
    pub offset: i32,
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self {
            radius: 3,
            offset: 0,
        }
    }
}

impl AdaptiveThreshold {
    pub fn new(radius: u32, offset: i32) -> Self {
        Self { radius, offset }
    }
}

fn to_luma(input: &VideoFrame, w: usize, h: usize, format: PixelFormat) -> Result<Vec<u8>, Error> {
    if input.planes.is_empty() {
        return Err(Error::invalid(
            "oxideav-image-filter: AdaptiveThreshold requires a non-empty input plane",
        ));
    }
    match format {
        PixelFormat::Gray8 => {
            let src = &input.planes[0];
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + w];
                out[y * w..(y + 1) * w].copy_from_slice(s);
            }
            Ok(out)
        }
        PixelFormat::Rgb24 | PixelFormat::Rgba => {
            let bpp = if format == PixelFormat::Rgb24 { 3 } else { 4 };
            let src = &input.planes[0];
            let row_bytes = w * bpp;
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + row_bytes];
                for x in 0..w {
                    let base = x * bpp;
                    // Rec. 601 luma proxy.
                    let r = s[base] as u32;
                    let g = s[base + 1] as u32;
                    let b = s[base + 2] as u32;
                    out[y * w + x] = ((r * 77 + g * 150 + b * 29 + 128) / 256).min(255) as u8;
                }
            }
            Ok(out)
        }
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            let src = &input.planes[0];
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                let s = &src.data[y * src.stride..y * src.stride + w];
                out[y * w..(y + 1) * w].copy_from_slice(s);
            }
            Ok(out)
        }
        other => Err(Error::unsupported(format!(
            "oxideav-image-filter: AdaptiveThreshold does not yet handle {other:?}"
        ))),
    }
}

/// Compute a per-pixel local mean using a box filter with edge clamp.
fn local_mean(src: &[u8], w: usize, h: usize, radius: usize) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    if radius == 0 {
        for (i, &v) in src.iter().enumerate() {
            out[i] = v as u16;
        }
        return out;
    }
    // Horizontal pass into a u32 accumulator buffer.
    let mut horiz = vec![0u32; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut sum: u32 = 0;
        for &v in row.iter().take(radius.min(w - 1) + 1) {
            sum += v as u32;
        }
        // Pre-pad the left side (clamp duplicates row[0]).
        sum += (row[0] as u32) * (radius as u32);
        for x in 0..w {
            // For x = 0..w-1, sum represents the window
            // [max(0, x - radius), min(w-1, x + radius)] with
            // out-of-bounds taps replaced by the nearest edge sample.
            horiz[y * w + x] = sum;
            // Advance window: add the sample entering on the right,
            // subtract the sample leaving on the left.
            let add_x = (x + radius + 1).min(w - 1);
            let drop_x = x.saturating_sub(radius);
            sum = sum + row[add_x] as u32 - row[drop_x] as u32;
        }
    }
    // Vertical pass over `horiz`.
    let win = 2 * radius + 1;
    for x in 0..w {
        let mut sum: u32 = 0;
        for y in 0..=radius.min(h - 1) {
            sum += horiz[y * w + x];
        }
        sum += horiz[x] * (radius as u32);
        for y in 0..h {
            let mean = (sum + (win * win / 2) as u32) / (win * win) as u32;
            out[y * w + x] = mean.min(255) as u16;
            let add_y = (y + radius + 1).min(h - 1);
            let drop_y = y.saturating_sub(radius);
            sum = sum + horiz[add_y * w + x] - horiz[drop_y * w + x];
        }
    }
    out
}

impl ImageFilter for AdaptiveThreshold {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: AdaptiveThreshold requires positive width and height",
            ));
        }
        let luma = to_luma(input, w, h, params.format)?;
        let mean = local_mean(&luma, w, h, self.radius as usize);
        let mut out = vec![0u8; w * h];
        for i in 0..w * h {
            let s = luma[i] as i32;
            let m = mean[i] as i32 + self.offset;
            out[i] = if s >= m { 255 } else { 0 };
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn constant_input_is_all_white_with_zero_offset() {
        // Constant image: every sample equals the local mean. With
        // offset = 0, the `>=` test wins for every pixel → 255.
        let input = gray(8, 8, |_, _| 128);
        let out = AdaptiveThreshold::new(2, 0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn constant_input_all_black_with_positive_offset() {
        // Sample equals local mean, but offset = +5 pushes the
        // threshold above the sample → 0 everywhere.
        let input = gray(8, 8, |_, _| 128);
        let out = AdaptiveThreshold::new(2, 5)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    #[test]
    fn local_mean_tracks_illumination() {
        // Bright region on the right (200) and dark region on the
        // left (50). Within each region the samples are uniform so
        // the local-mean threshold collapses to global (sample == mean,
        // offset = 0 → 255 in both regions). Bump offset to +10 to
        // get all-zero. With offset = 0 every pixel passes.
        let input = gray(16, 4, |x, _| if x < 8 { 50 } else { 200 });
        let out = AdaptiveThreshold::new(1, 0)
            .apply(&input, p_gray(16, 4))
            .unwrap();
        // First row sample equals local mean ⇒ ≥ ⇒ 255.
        assert_eq!(out.planes[0].data[0], 255);
        assert_eq!(out.planes[0].data[15], 255);
    }

    #[test]
    fn ink_against_local_mean_goes_black() {
        // 8×8 white background with a small black square. With
        // radius = 2, the local mean inside the square will be close
        // to the white background (because most of the window is white),
        // so the dark pixel falls below the mean ⇒ 0.
        let input = gray(8, 8, |x, y| {
            if (3..=4).contains(&x) && (3..=4).contains(&y) {
                30
            } else {
                200
            }
        });
        let out = AdaptiveThreshold::new(2, 0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        // The ink pixels should be black; the background should be white.
        assert_eq!(out.planes[0].data[3 * 8 + 3], 0);
        assert_eq!(out.planes[0].data[0], 255);
    }

    #[test]
    fn shape_preserved() {
        let input = gray(5, 3, |_, _| 100);
        let out = AdaptiveThreshold::default()
            .apply(&input, p_gray(5, 3))
            .unwrap();
        assert_eq!(out.planes[0].stride, 5);
        assert_eq!(out.planes[0].data.len(), 15);
    }

    #[test]
    fn works_on_rgb_via_luma_proxy() {
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for _ in 0..16 {
            data.extend_from_slice(&[100u8, 100, 100]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = AdaptiveThreshold::new(1, 0)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Single-plane Gray8 output regardless of input.
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 16);
    }
}
