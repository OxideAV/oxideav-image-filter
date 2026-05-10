//! Auto-crop to the bounding box of non-background pixels (`-trim`).
//!
//! Walks every pixel of the input and finds the smallest axis-aligned
//! rectangle that contains all samples differing from a reference
//! background colour by more than the configured `fuzz` tolerance. The
//! reference colour defaults to the corner pixel `(0, 0)` of the source
//! — this matches ImageMagick's behaviour when neither `-bordercolor`
//! nor `-background` is set on the command line.
//!
//! Clean-room: detect the bbox, then call [`Crop`](crate::crop::Crop).
//! Fuzz uses the maximum per-channel absolute difference (the
//! `Chebyshev` / L-infinity distance) — straightforward and matches
//! IM's `-fuzz N` semantics on `[0, 255]` channels.

use crate::blur::bytes_per_plane_pixel;
use crate::{is_supported_format, Crop, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Crop the frame down to the bounding box of pixels that differ from
/// a reference background colour by more than `fuzz` per channel.
#[derive(Clone, Debug, Default)]
pub struct Trim {
    /// Maximum per-channel absolute difference (`0..=255`) that still
    /// counts as "background". `0` means an exact-match background;
    /// `255` would match every pixel.
    pub fuzz: u8,
    /// Optional explicit reference colour `[R, G, B, A]`. When `None`
    /// the reference is the input's `(0, 0)` pixel.
    pub background: Option<[u8; 4]>,
}

impl Trim {
    /// Build a trim with `fuzz = 0` (exact match) and the auto-detected
    /// `(0, 0)` reference colour.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the per-channel fuzz tolerance.
    pub fn with_fuzz(mut self, fuzz: u8) -> Self {
        self.fuzz = fuzz;
        self
    }

    /// Override the reference background colour (defaults to the
    /// `(0, 0)` pixel of the input).
    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = Some(bg);
        self
    }
}

impl ImageFilter for Trim {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Trim does not yet handle {:?}",
                params.format
            )));
        }
        if params.width == 0 || params.height == 0 {
            return Ok(input.clone());
        }
        let bbox = compute_bbox(input, params, self.background, self.fuzz)?;
        // When the whole image matches the background, keep a 1×1 pixel
        // so the output frame stays representable.
        let (x, y, w, h) = bbox.unwrap_or((0, 0, 1, 1));
        if x == 0 && y == 0 && w == params.width && h == params.height {
            return Ok(input.clone());
        }
        Crop::new(x, y, w, h).apply(input, params)
    }
}

fn compute_bbox(
    input: &VideoFrame,
    params: VideoStreamParams,
    bg_override: Option<[u8; 4]>,
    fuzz: u8,
) -> Result<Option<(u32, u32, u32, u32)>, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let plane = input
        .planes
        .first()
        .ok_or_else(|| Error::invalid("oxideav-image-filter: Trim needs at least one plane"))?;
    let bpp = match params.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        // Planar YUV — operate on the Y plane only for bbox detection.
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => 1,
        other => {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Trim cannot bbox {other:?}"
            )));
        }
    };
    debug_assert_eq!(bpp, bytes_per_plane_pixel(params.format, 0));

    let bg = match bg_override {
        Some(c) => c,
        None => {
            // Read from (0, 0) of the first plane.
            let mut c = [0u8; 4];
            c[..bpp].copy_from_slice(&plane.data[..bpp]);
            c
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

    fn params_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn no_padding_is_identity() {
        let input = gray(4, 4, |x, y| (x + y * 5 + 50) as u8);
        let out = Trim::new().apply(&input, params_gray(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
        assert_eq!(out.planes[0].stride, 4);
    }

    #[test]
    fn trims_solid_border() {
        // 5×5: 0-bordered image with a 3×3 bright block at (1, 1).
        let input = gray(5, 5, |x, y| {
            if (1..=3).contains(&x) && (1..=3).contains(&y) {
                200
            } else {
                0
            }
        });
        let out = Trim::new().apply(&input, params_gray(5, 5)).unwrap();
        // Output is 3×3, all 200.
        assert_eq!(out.planes[0].stride, 3);
        assert_eq!(out.planes[0].data.len(), 9);
        assert!(out.planes[0].data.iter().all(|&v| v == 200));
    }

    #[test]
    fn fuzz_includes_close_pixels_in_background() {
        // Border noise in [0, 5]; subject at 200.
        let input = gray(5, 5, |x, y| {
            if x == 2 && y == 2 {
                200
            } else {
                ((x + y) % 6) as u8 // 0..5
            }
        });
        let out = Trim::new()
            .with_fuzz(10)
            .apply(&input, params_gray(5, 5))
            .unwrap();
        // Only the centre pixel exceeds the fuzz tolerance (which is
        // computed against the bg = (0, 0) sample of value 0). Output
        // is a 1×1 frame holding 200.
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data, vec![200]);
    }

    #[test]
    fn explicit_background_overrides_corner_sample() {
        let input = gray(4, 4, |_, _| 100);
        // Treat 100 as the background; everything else differs.
        let out = Trim::new()
            .with_background([100, 0, 0, 255])
            .apply(&input, params_gray(4, 4))
            .unwrap();
        // No pixels differ → whole image counts as bg → return a 1×1.
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data, vec![100]);
    }

    #[test]
    fn rgba_trims_around_subject() {
        // 4×4 RGBA: opaque-black background with a single red pixel at (1, 2).
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                if x == 1 && y == 2 {
                    data.extend_from_slice(&[255, 0, 0, 255]);
                } else {
                    data.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Trim::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Output is 1×1.
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data, vec![255, 0, 0, 255]);
    }

    #[test]
    fn empty_subject_returns_one_pixel() {
        let input = gray(3, 3, |_, _| 77);
        let out = Trim::new().apply(&input, params_gray(3, 3)).unwrap();
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data, vec![77]);
    }
}
