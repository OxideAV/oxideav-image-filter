//! Auto-trim: detect the dominant background colour, then crop.
//!
//! Like [`Trim`](crate::Trim), but instead of trusting the `(0, 0)`
//! corner sample as the background, AutoTrim takes a histogram-vote over
//! the entire frame and crops away the bounding box of pixels that
//! disagree with the most-common colour. This is the right operator for
//! screenshots with a single dominant backdrop (white / black / a known
//! brand colour) where the corner pixel might happen to be a subject
//! pixel.
//!
//! Algorithm — clean-room, no library borrow:
//!
//! 1. Sample every pixel into a 4-bit-per-channel hash bucket (4096 RGB
//!    buckets for `Rgb24` / `Rgba`; 16 grey buckets for `Gray8` /
//!    luma-of-YUV). Pick the bucket with the highest vote.
//! 2. Re-scan the frame: the bucket centre is the inferred background.
//!    Compute the bounding box of pixels that exceed `fuzz` per-channel
//!    against this background (Chebyshev / L-infinity distance).
//! 3. Delegate the actual cropping to [`Crop`](crate::Crop) so YUV
//!    chroma alignment matches `Trim`.
//!
//! When the bounding box covers the whole image (no detectable
//! background border), the input is returned unchanged. When every
//! pixel votes for the background, the output is a single 1×1 pixel
//! frame holding the background colour (matching `Trim`'s behaviour).

use crate::blur::bytes_per_plane_pixel;
use crate::{is_supported_format, Crop, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Auto-trim filter — pick the dominant colour and crop to its
/// complement bounding box.
#[derive(Clone, Debug, Default)]
pub struct AutoTrim {
    /// Maximum per-channel absolute difference (`0..=255`) that still
    /// counts as "background". `0` means an exact-match background.
    pub fuzz: u8,
}

impl AutoTrim {
    /// Build an AutoTrim with `fuzz = 0` (exact match against the
    /// detected background).
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the per-channel fuzz tolerance.
    pub fn with_fuzz(mut self, fuzz: u8) -> Self {
        self.fuzz = fuzz;
        self
    }
}

impl ImageFilter for AutoTrim {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: AutoTrim does not yet handle {:?}",
                params.format
            )));
        }
        if params.width == 0 || params.height == 0 {
            return Ok(input.clone());
        }
        let bg = detect_background(input, params)?;
        let bbox = compute_bbox(input, params, bg, self.fuzz)?;
        let (x, y, w, h) = bbox.unwrap_or((0, 0, 1, 1));
        if x == 0 && y == 0 && w == params.width && h == params.height {
            return Ok(input.clone());
        }
        Crop::new(x, y, w, h).apply(input, params)
    }
}

/// Inspect `input` and return a 4-channel background colour estimate.
fn detect_background(input: &VideoFrame, params: VideoStreamParams) -> Result<[u8; 4], Error> {
    let plane = input
        .planes
        .first()
        .ok_or_else(|| Error::invalid("oxideav-image-filter: AutoTrim needs at least one plane"))?;
    let w = params.width as usize;
    let h = params.height as usize;
    let bpp = match params.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => 1,
        other => {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: AutoTrim cannot detect bg on {other:?}"
            )));
        }
    };
    debug_assert_eq!(bpp, bytes_per_plane_pixel(params.format, 0));

    if bpp == 1 {
        // 16 grey buckets (4-bit quantise).
        let mut votes = [0u32; 16];
        for y in 0..h {
            let row = &plane.data[y * plane.stride..y * plane.stride + w];
            for &v in row {
                votes[(v >> 4) as usize] += 1;
            }
        }
        let (best, _) = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .unwrap_or((0, &0));
        // Bucket centre: (best << 4) | 0x08.
        let grey = ((best as u8) << 4) | 0x08;
        Ok([grey, grey, grey, 255])
    } else {
        // 4096 RGB buckets (4-bit per channel).
        let mut votes = vec![0u32; 16 * 16 * 16];
        for y in 0..h {
            let row = &plane.data[y * plane.stride..y * plane.stride + w * bpp];
            for x in 0..w {
                let off = x * bpp;
                let r = row[off];
                let g = row[off + 1];
                let b = row[off + 2];
                let idx = ((r >> 4) as usize) * 256 + ((g >> 4) as usize) * 16 + (b >> 4) as usize;
                votes[idx] += 1;
            }
        }
        let (best, _) = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .unwrap_or((0, &0));
        let rh = (best / 256) as u8;
        let gh = ((best % 256) / 16) as u8;
        let bh = (best % 16) as u8;
        Ok([(rh << 4) | 0x08, (gh << 4) | 0x08, (bh << 4) | 0x08, 255])
    }
}

fn compute_bbox(
    input: &VideoFrame,
    params: VideoStreamParams,
    bg: [u8; 4],
    fuzz: u8,
) -> Result<Option<(u32, u32, u32, u32)>, Error> {
    let plane = input.planes.first().unwrap();
    let w = params.width as usize;
    let h = params.height as usize;
    let bpp = match params.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => 1,
        other => {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: AutoTrim cannot bbox {other:?}"
            )));
        }
    };

    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    let mut any = false;
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * bpp];
        for x in 0..w {
            let off = x * bpp;
            let mut differs = false;
            for ch in 0..bpp {
                let dv = row[off + ch].abs_diff(bg[ch]);
                if dv > fuzz {
                    differs = true;
                    break;
                }
            }
            if differs {
                any = true;
                if (x as u32) < min_x {
                    min_x = x as u32;
                }
                if (x as u32) > max_x {
                    max_x = x as u32;
                }
                if (y as u32) < min_y {
                    min_y = y as u32;
                }
                if (y as u32) > max_y {
                    max_y = y as u32;
                }
            }
        }
    }
    if !any {
        return Ok(None);
    }
    Ok(Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
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
    fn picks_dominant_background() {
        // 5×5: bright (250) frame around a single dark (10) middle pixel.
        // The (0, 0) corner sample is bright, AND the histogram majority
        // is bright, so the bbox shrinks to the dark pixel. Use a small
        // fuzz to absorb the 4-bit-per-channel quantisation of the
        // histogram bucket centre (bucket 15 centre = 248, value = 250 →
        // diff = 2).
        let input = gray(5, 5, |x, y| if x == 2 && y == 2 { 10 } else { 250 });
        let out = AutoTrim::new()
            .with_fuzz(8)
            .apply(&input, p_gray(5, 5))
            .unwrap();
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data.len(), 1);
        assert_eq!(out.planes[0].data[0], 10);
    }

    #[test]
    fn ignores_corner_outlier_for_background_pick() {
        // 5×5 image: corner is "noise" (different colour) but the
        // dominant colour is the bulk. AutoTrim should ignore the corner
        // outlier and still trim against the majority.
        // Build: corner=0, rest=200 except a 3×3 subject at (1,1).
        let input = gray(5, 5, |x, y| {
            if x == 0 && y == 0 {
                0
            } else if (1..=3).contains(&x) && (1..=3).contains(&y) {
                100
            } else {
                200
            }
        });
        let out = AutoTrim::new().apply(&input, p_gray(5, 5)).unwrap();
        // The detected background is in the (200 ± 8) bucket → output
        // bbox includes both the 100 subject pixels AND the (0,0) outlier.
        // That bbox spans (0,0)..(3,3) → 4×4 frame.
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data.len(), 16);
    }

    #[test]
    fn solid_image_collapses_to_1x1() {
        // Every pixel falls into the same histogram bucket → the detected
        // background equals the bucket centre. A small fuzz absorbs the
        // 4-bit quantisation gap; with fuzz=8 every pixel counts as
        // background and the bbox collapses to the 1×1 "any pixel" fallback.
        let input = gray(4, 4, |_, _| 77);
        let out = AutoTrim::new()
            .with_fuzz(8)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data.len(), 1);
    }

    #[test]
    fn rejects_unsupported_format() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 4,
            height: 4,
        };
        let err = AutoTrim::new().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("AutoTrim"));
    }
}
