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

use crate::{PixelFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Bloom / glow filter.
#[derive(Clone, Debug)]
#[non_exhaustive]
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

impl Bloom {
    /// Builder-style setter for [`threshold`](Self::threshold).
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }
    /// Builder-style setter for [`radius`](Self::radius).
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }
    /// Builder-style setter for [`intensity`](Self::intensity).
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }
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

impl PixelFilter for Bloom {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        // One kernel at every depth (8-bit pictures are widened to it
        // and rounded back, saturating; float pictures keep the glow's
        // headroom).
        crate::deep::run_any(input, params, |img| Ok(self.bloom(&img)))
    }
}

impl Bloom {
    /// `out = pixel + intensity · glow`, the glow being the box mean
    /// (`(2r + 1)²`, replicate edges) of the highlights: the pixels
    /// whose brightness — the sample on grey, `(R + 2G + B) / 4` on
    /// colour — reaches `threshold / 255`, black elsewhere. Alpha is
    /// untouched.
    fn bloom(&self, img: &crate::deep::FloatImage) -> crate::deep::FloatImage {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (ch, cc) = (img.channels(), img.color_channels());
        let threshold = self.threshold as f32;
        let mut highlights = img.samples().to_vec();
        for px in highlights.chunks_exact_mut(ch) {
            let level = if cc == 1 {
                px[0]
            } else {
                (px[0] + 2.0 * px[1] + px[2]) / 4.0
            };
            // Compared on the 8-bit scale, an 8-bit code landing exactly
            // on the threshold whatever the depth it is stored at.
            if crate::deep::code8(level) < threshold || level.is_nan() {
                px[..cc].fill(0.0);
            }
        }
        let glow = crate::kernel::box_mean(
            &highlights,
            w * ch,
            w,
            h,
            ch,
            cc,
            crate::kernel::Window::radius(self.radius as usize),
        );
        let mut out = img.clone();
        for (o, g) in out
            .samples_mut()
            .chunks_exact_mut(ch)
            .zip(glow.chunks_exact(ch))
        {
            for c in 0..cc {
                o[c] += self.intensity * g[c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    #[allow(unused_imports)]
    use oxideav_core::{PixelFormat, VideoPlane};

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
