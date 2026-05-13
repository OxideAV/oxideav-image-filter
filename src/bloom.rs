//! Bloom / glow — additive blur of bright regions.
//!
//! Classic post-processing recipe:
//!
//! 1. Threshold the input to a "highlights" buffer: pixels whose
//!    luminance is at or above `threshold` survive; the rest become 0.
//! 2. Box-blur the highlights buffer with the configured `radius`
//!    (separable, edge-clamped — gives a soft halo).
//! 3. Additively blend the blurred highlights back on top of the
//!    original (`out = clamp(src + intensity * highlights, 255)`).
//!
//! Result: bright regions develop a soft glowing halo, mid-tones and
//! shadows are unchanged. Useful for stylising emissive UI elements,
//! sun glints, or to give a punchy HDR look on SDR footage.
//!
//! Pixel formats: `Gray8`, `Rgb24`, `Rgba`. RGBA alpha is preserved
//! (the glow stays inside the existing coverage). YUV inputs return
//! [`Error::unsupported`] — the glow is computed in display-RGB
//! space, so YUV callers should explicitly transcode first.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Bloom / glow filter.
#[derive(Clone, Debug)]
pub struct Bloom {
    /// Luminance cut-off (`0..=255`) — only pixels at or above this
    /// value contribute to the bloom.
    pub threshold: u8,
    /// Box-blur radius applied to the highlights buffer. Larger radii
    /// produce a softer / wider halo.
    pub radius: u32,
    /// Intensity multiplier for the additive composite. `1.0` adds the
    /// blurred highlights at full strength; `0.0` is identity.
    pub intensity: f32,
}

impl Default for Bloom {
    fn default() -> Self {
        Self {
            threshold: 200,
            radius: 6,
            intensity: 0.6,
        }
    }
}

impl Bloom {
    pub fn new(threshold: u8, radius: u32, intensity: f32) -> Self {
        Self {
            threshold,
            radius,
            intensity: if intensity.is_finite() {
                intensity.max(0.0)
            } else {
                0.0
            },
        }
    }
}

impl ImageFilter for Bloom {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        match params.format {
            PixelFormat::Gray8 => apply_gray(self, input, params),
            PixelFormat::Rgb24 => apply_rgb(self, input, params, false),
            PixelFormat::Rgba => apply_rgb(self, input, params, true),
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Bloom does not yet handle {other:?}"
            ))),
        }
    }
}

fn apply_gray(
    f: &Bloom,
    input: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    let src = &input.planes[0];
    let mut highlights = vec![0u8; w * h];
    for y in 0..h {
        let row = &src.data[y * src.stride..y * src.stride + w];
        for x in 0..w {
            let v = row[x];
            highlights[y * w + x] = if v >= f.threshold { v } else { 0 };
        }
    }
    let blurred = box_blur_gray(&highlights, w, h, f.radius);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let s_row = &src.data[y * src.stride..y * src.stride + w];
        for x in 0..w {
            let s = s_row[x] as f32;
            let g = blurred[y * w + x] as f32;
            out[y * w + x] = (s + f.intensity * g).clamp(0.0, 255.0).round() as u8;
        }
    }
    Ok(VideoFrame {
        pts: input.pts,
        planes: vec![VideoPlane {
            stride: w,
            data: out,
        }],
    })
}

fn apply_rgb(
    f: &Bloom,
    input: &VideoFrame,
    params: VideoStreamParams,
    has_alpha: bool,
) -> Result<VideoFrame, Error> {
    let bpp = if has_alpha { 4 } else { 3 };
    let w = params.width as usize;
    let h = params.height as usize;
    let src = &input.planes[0];

    // Build per-channel highlights buffer (R / G / B independently). We
    // gate on Rec. 601 luma `(R + 2G + B) / 4` for the threshold.
    let mut hi_r = vec![0u8; w * h];
    let mut hi_g = vec![0u8; w * h];
    let mut hi_b = vec![0u8; w * h];
    for y in 0..h {
        let row = &src.data[y * src.stride..y * src.stride + w * bpp];
        for x in 0..w {
            let off = x * bpp;
            let r = row[off];
            let g = row[off + 1];
            let b = row[off + 2];
            let luma = (r as u16 + 2 * g as u16 + b as u16) / 4;
            if luma >= f.threshold as u16 {
                hi_r[y * w + x] = r;
                hi_g[y * w + x] = g;
                hi_b[y * w + x] = b;
            }
        }
    }
    let br = box_blur_gray(&hi_r, w, h, f.radius);
    let bg = box_blur_gray(&hi_g, w, h, f.radius);
    let bb = box_blur_gray(&hi_b, w, h, f.radius);

    let mut out = vec![0u8; w * h * bpp];
    for y in 0..h {
        let s_row = &src.data[y * src.stride..y * src.stride + w * bpp];
        let o_row = &mut out[y * w * bpp..(y + 1) * w * bpp];
        for x in 0..w {
            let off = x * bpp;
            let r = s_row[off] as f32 + f.intensity * br[y * w + x] as f32;
            let g = s_row[off + 1] as f32 + f.intensity * bg[y * w + x] as f32;
            let b = s_row[off + 2] as f32 + f.intensity * bb[y * w + x] as f32;
            o_row[off] = r.clamp(0.0, 255.0).round() as u8;
            o_row[off + 1] = g.clamp(0.0, 255.0).round() as u8;
            o_row[off + 2] = b.clamp(0.0, 255.0).round() as u8;
            if has_alpha {
                o_row[off + 3] = s_row[off + 3];
            }
        }
    }
    Ok(VideoFrame {
        pts: input.pts,
        planes: vec![VideoPlane {
            stride: w * bpp,
            data: out,
        }],
    })
}

fn box_blur_gray(src: &[u8], w: usize, h: usize, radius: u32) -> Vec<u8> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let r = radius as i32;
    let denom = 2 * r as u32 + 1;
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dx in -r..=r {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + sx] as u32;
            }
            tmp[y * w + x] = (acc / denom) as u8;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dy in -r..=r {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                acc += tmp[sy * w + x] as u32;
            }
            out[y * w + x] = (acc / denom) as u8;
        }
    }
    out
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
    fn no_highlights_is_identity_for_gray() {
        // All pixels below the threshold → highlights empty → output ==
        // input.
        let input = gray(8, 8, |_, _| 100);
        let out = Bloom::new(200, 2, 1.0).apply(&input, p_gray(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn bright_pixel_glows_outward() {
        // Single bright pixel in the middle of a dark frame should
        // produce a halo of slightly-brighter neighbours.
        let input = gray(8, 8, |x, y| if x == 4 && y == 4 { 255 } else { 50 });
        let out = Bloom::new(200, 2, 1.0).apply(&input, p_gray(8, 8)).unwrap();
        // The neighbour pixel (4, 5) should be brighter than the
        // baseline 50 of the input thanks to the halo.
        let neighbour = out.planes[0].data[5 * 8 + 4];
        assert!(neighbour > 50, "expected glow, got {neighbour}");
    }

    #[test]
    fn zero_intensity_is_identity() {
        let input = gray(4, 4, |x, _| x as u8 * 60);
        let out = Bloom::new(0, 4, 0.0).apply(&input, p_gray(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rgb_bright_pixel_glows() {
        let mut data = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                if x == 4 && y == 4 {
                    data.extend_from_slice(&[255u8, 255, 255]);
                } else {
                    data.extend_from_slice(&[20u8, 20, 20]);
                }
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 8,
            height: 8,
        };
        let out = Bloom::new(200, 2, 1.0).apply(&input, p).unwrap();
        // Neighbour pixel at (4, 5) should have R > 20 from the halo.
        let off = (5 * 8 + 4) * 3;
        assert!(out.planes[0].data[off] > 20);
    }

    #[test]
    fn rgba_alpha_is_preserved() {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[220u8, 220, 220, 130]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 4,
            height: 4,
        };
        let out = Bloom::new(200, 1, 0.5).apply(&input, p).unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 130);
        }
    }

    #[test]
    fn rejects_yuv() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = Bloom::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("Bloom"));
    }
}
