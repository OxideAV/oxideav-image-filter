//! Oil-paint stylise (`-paint radius`).
//!
//! Replaces every pixel by the colour that occurs most frequently among
//! the samples in a `(2*radius + 1)²` window around it — quantised to
//! a coarse intensity bucket so similar shades count together. The
//! result looks like a painting where smooth gradients have been
//! collapsed into a small number of flat tone "strokes".
//!
//! Clean-room implementation derived from the textbook description.
//! For each output pixel `(x, y)`:
//!
//! 1. Walk the `(2r + 1)²` window centred on `(x, y)`.
//! 2. Compute each window-pixel's intensity bucket
//!    `b = min(⌊Y · levels⌋, levels − 1)` on the normalised luma
//!    `Y = (77 R + 150 G + 29 B) / 256` (default `levels = 20`).
//! 3. Tally a histogram of bucket counts and accumulate per-bucket
//!    channel sums.
//! 4. Pick the bucket with the highest count — break ties by picking
//!    the lower bucket index (deterministic).
//! 5. Output `(sum_R / count, sum_G / count, sum_B / count)`.
//!
//! 1×1 windows (`radius = 0`) are identity.
//!
//! # Pixel formats
//!
//! - `Rgb24` / `Rgba`: full effect; alpha is preserved on RGBA.
//! - `Gray8`: per-pixel intensity is the sample itself; output is the
//!   mean of the modal bucket.
//! - `GrayF32Le` / `RgbF32Le` / `RgbaF32Le` natively and every 16-bit
//!   layout through them: the same kernel at the picture's precision
//!   (the bucket mean is not rounded to 8 bits).
//!
//! The window slides along each row, so the cost per pixel grows with
//! the radius, not with the window area.

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Oil-paint stylise filter.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Paint {
    /// Window radius in pixels. The window is `(2 * radius + 1)²`.
    pub radius: u32,
    /// Number of intensity buckets used for the modal-bucket vote.
    /// Default `20`; smaller values create flatter / more cartoonish
    /// strokes, larger values approach an identity blur.
    pub levels: u32,
}

impl Paint {
    /// Builder-style setter for [`radius`](Self::radius).
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }
}

impl Default for Paint {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Paint {
    /// Build a paint filter with the given window radius and the
    /// default 20 intensity buckets.
    pub fn new(radius: u32) -> Self {
        Self { radius, levels: 20 }
    }

    /// Override the bucket count. Clamped to `[1, 256]`.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels.clamp(1, 256);
        self
    }
}

impl PixelFilter for Paint {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        // One kernel for every depth: an 8-bit picture is widened to it
        // (`v / 255`) and rounded back, float and 16-bit pictures run on
        // it directly.
        let (radius, levels) = (self.radius as usize, self.levels.clamp(1, 256) as usize);
        crate::deep::run_any(input, params, |img| Ok(paint(&img, radius, levels)))
    }
}

/// The oil-paint kernel on normalised samples.
///
/// Every pixel has an intensity bucket
/// `min(⌊Y · levels⌋, levels − 1)` with `Y = (77 R + 150 G + 29 B) / 256`
/// clamped to `[0, 1]` (grey: the sample). The output pixel is the mean
/// colour of the most populated bucket of its `(2r + 1)²` replicate-edge
/// window, the lowest such bucket on a tie; alpha is the pixel's own.
///
/// The window slides: moving one pixel right removes a column of
/// `2r + 1` pixels from the per-bucket counts and colour sums and adds
/// one, so the cost per pixel is `O(r + levels)`, not `O(r²)`. Sums are
/// binary64, so adding and removing a sample cancel to well below the
/// binary32 resolution of the result.
fn paint(img: &FloatImage, radius: usize, levels: usize) -> FloatImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let ch = img.channels();
    let nc = img.color_channels();
    if radius == 0 || w == 0 || h == 0 {
        return img.clone();
    }
    let src = img.samples();
    // The window holds `(2r + 1)²` samples; capping the radius at 2²⁴
    // keeps that count exact in the binary64 sums (no picture is wide
    // enough to tell the difference).
    let r = radius.min(1 << 24);
    let bucket_of = |px: &[f32]| -> u16 {
        let y = if nc == 1 {
            px[0]
        } else {
            (77.0 * px[0] + 150.0 * px[1] + 29.0 * px[2]) / 256.0
        };
        // NaN falls into bucket 0.
        let y = if y.is_nan() { 0.0 } else { y.clamp(0.0, 1.0) };
        ((y * levels as f32) as usize).min(levels - 1) as u16
    };
    let buckets: Vec<u16> = src.chunks_exact(ch).map(bucket_of).collect();
    let mut out = src.to_vec();
    let row = w * ch;
    // How many of the window offsets `−r..=r` around `centre` land on
    // position `at` of a line of `len` under replicate-edge addressing:
    // one, plus the offsets that overhang an edge when `at` is that
    // edge. A window larger than the picture therefore costs no more
    // than the picture.
    let copies = move |at: usize, centre: usize, len: usize| -> u64 {
        let mut m = 1u64;
        if at == 0 {
            m += r.saturating_sub(centre) as u64;
        }
        if at == len - 1 {
            m += centre.saturating_add(r).saturating_sub(len - 1) as u64;
        }
        m
    };
    crate::exec::par_bands(&mut out, row, |y0, band| {
        let mut counts = vec![0u64; levels];
        let mut sums = vec![0.0f64; levels * nc];
        for (i, line) in band.chunks_exact_mut(row).enumerate() {
            let y = y0 + i;
            counts.fill(0);
            sums.fill(0.0);
            let (top, bottom) = (y.saturating_sub(r), y.saturating_add(r).min(h - 1));
            // Add or remove `times` copies of the window column at `x`.
            let column = |x: usize, times: u64, add: bool, counts: &mut [u64], sums: &mut [f64]| {
                for sy in top..=bottom {
                    let m = times * copies(sy, y, h);
                    let p = sy * w + x;
                    let b = buckets[p] as usize;
                    let px = &src[p * ch..p * ch + nc];
                    if add {
                        counts[b] += m;
                        for c in 0..nc {
                            sums[b * nc + c] += m as f64 * px[c] as f64;
                        }
                    } else {
                        counts[b] -= m;
                        for c in 0..nc {
                            sums[b * nc + c] -= m as f64 * px[c] as f64;
                        }
                        if counts[b] == 0 {
                            // An empty bucket holds exactly nothing.
                            sums[b * nc..(b + 1) * nc].fill(0.0);
                        }
                    }
                }
            };
            for sx in 0..=r.min(w - 1) {
                column(sx, copies(sx, 0, w), true, &mut counts, &mut sums);
            }
            for x in 0..w {
                if x > 0 {
                    // One copy of the clamped column `x − 1 − r` leaves,
                    // one of the clamped column `x + r` enters.
                    let gone = (x - 1).saturating_sub(r);
                    let come = x.saturating_add(r).min(w - 1);
                    if gone != come {
                        column(gone, 1, false, &mut counts, &mut sums);
                        column(come, 1, true, &mut counts, &mut sums);
                    }
                }
                let mut best = 0usize;
                for (b, &n) in counts.iter().enumerate().skip(1) {
                    if n > counts[best] {
                        best = b;
                    }
                }
                let n = counts[best].max(1) as f64;
                for c in 0..nc {
                    line[x * ch + c] = (sums[best * nc + c] / n) as f32;
                }
            }
        }
    });
    FloatImage::from_samples(img.width(), img.height(), ch, out).unwrap_or_else(|_| img.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }

    fn params_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn radius_zero_is_identity() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 77]);
        let out = Paint::new(0).apply(&input, params_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn solid_image_passes_through() {
        let input = rgb(4, 4, |_, _| [100, 50, 200]);
        let out = Paint::new(1).apply(&input, params_rgb(4, 4)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[100, 50, 200]);
        }
    }

    #[test]
    fn picks_modal_colour_in_window() {
        // 3×3 input where 8 pixels are colour A and 1 is colour B at
        // the centre. With radius 1 every output pixel sees the same
        // window (border-clamped), so all output should be A.
        let input = rgb(3, 3, |x, y| {
            if x == 1 && y == 1 {
                [255, 0, 0]
            } else {
                [10, 10, 10]
            }
        });
        let out = Paint::new(1).apply(&input, params_rgb(3, 3)).unwrap();
        // Modal bucket (low intensity) wins ⇒ output is roughly the
        // mean of those 8 dark pixels.
        let centre = &out.planes[0].data[(3 + 1) * 3..(3 + 1) * 3 + 3];
        assert!(
            centre[0] < 30 && centre[1] < 30 && centre[2] < 30,
            "expected dark modal pixel, got {centre:?}"
        );
    }

    #[test]
    fn rgba_alpha_is_taken_from_source() {
        let data: Vec<u8> = (0..16).flat_map(|i| [10u8, 10, 10, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Paint::new(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8, "alpha at {i}");
        }
    }

    #[test]
    fn levels_clamped_to_1_256() {
        let p = Paint::new(1).with_levels(0);
        assert_eq!(p.levels, 1);
        let p = Paint::new(1).with_levels(1000);
        assert_eq!(p.levels, 256);
    }

    #[test]
    fn sixteen_bit_strokes_keep_sixteen_bit_means() {
        // Two flat halves at codes no 8-bit value holds: every window
        // has a clear majority, so each pixel is exactly its half's
        // code — an 8-bit round trip would return multiples of 257.
        let (w, h) = (8usize, 4usize);
        let (lo, hi) = (10_001u16, 50_003u16);
        let mut data = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                data.extend_from_slice(&(if x < w / 2 { lo } else { hi }).to_le_bytes());
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w * 2,
                data,
            }],
        };
        let out = Paint::new(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray16Le,
                    width: w as u32,
                    height: h as u32,
                },
            )
            .unwrap();
        for (i, c) in out.planes[0].data.as_chunks::<2>().0.iter().enumerate() {
            let want = if i % w < w / 2 { lo } else { hi };
            assert_eq!(u16::from_le_bytes(*c), want, "sample {i}");
        }
    }
}
