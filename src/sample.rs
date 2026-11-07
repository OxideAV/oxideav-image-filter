//! `-sample WxH` (`Imagick::sampleImage`) and `-scale WxH`
//! (`Imagick::scaleImage`), issue #10.
//!
//! **Sample** is point sampling with no filtering. Output pixel `x` copies source column
//! `⌈(x + ½) · W / w⌉ − 1` — the source pixel under the output pixel's
//! centre, a centre that falls exactly on a source pixel boundary taking
//! the left (upper) one — and the same down the rows. Measured on
//! `magick` 7.1.2 and bit-exact against it (`tests/magick_ops.rs`).
//! Pixels are copied whole, so every packed layout (palette indices,
//! 16-bit and float samples included) is taken as it is.
//!
//! **Scale** is the exact box average in both directions: output pixel
//! `x` covers the source interval `[x · W / w, (x + 1) · W / w)` and is
//! the coverage-weighted mean of the source pixels it overlaps (so an
//! enlargement repeats pixels and blends only across the boundaries).
//! With alpha the colour is weighted by coverage × alpha. Measured on
//! `magick` and matched within one 8-bit code.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// Point-sample to `width × height`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
}

impl Sample {
    /// Sample to `width × height` (each at least 1).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// The source index output index `x` of `out` copies from `src`.
    pub fn source_index(x: u32, out: u32, src: u32) -> u32 {
        let num = (2 * x as u64 + 1) * src as u64;
        (((num - 1) / (2 * out as u64)) as u32).min(src - 1)
    }
}

impl PixelFilter for Sample {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        crate::transpose::PACKED
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let g = layout::plane_geometry(params.format, 0, params.width, params.height).ok_or_else(
            || {
                Error::unsupported(format!(
                    "oxideav-image-filter: Sample does not take {:?}",
                    params.format
                ))
            },
        )?;
        let (sw, sh, bpp) = (g.width, g.height, g.position_bytes);
        let (w, h) = (self.width, self.height);
        let cols: Vec<usize> = (0..w)
            .map(|x| Self::source_index(x, w, sw) as usize * bpp)
            .collect();
        let src = &input.planes[0];
        let stride = w as usize * bpp;
        let mut data = Vec::with_capacity(stride * h as usize);
        for y in 0..h {
            let row = &src.data[Self::source_index(y, h, sh) as usize * src.stride..];
            for &c in &cols {
                data.extend_from_slice(&row[c..c + bpp]);
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data }],
        })
    }
}

/// `(first source index, weights)` of each output index along one axis.
fn box_weights(src: u32, out: u32) -> Vec<(usize, Vec<f64>)> {
    let scale = src as f64 / out as f64;
    (0..out)
        .map(|x| {
            let (a, b) = (x as f64 * scale, (x + 1) as f64 * scale);
            let first = a.floor() as usize;
            let last = ((b.ceil() as usize).max(first + 1)).min(src as usize);
            let w = (first..last)
                .map(|i| (b.min(i as f64 + 1.0) - a.max(i as f64)).max(0.0) / scale)
                .collect();
            (first, w)
        })
        .collect()
}

/// Box-average to `width × height` (`-scale`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
}

impl Scale {
    /// Scale to `width × height` (each at least 1).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }
}

impl PixelFilter for Scale {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| {
            let (sw, sh, ch) = (img.width(), img.height(), img.channels());
            let alpha = ch == 4;
            let (w, h) = (self.width, self.height);
            let xs = box_weights(sw, w);
            let ys = box_weights(sh, h);
            let s = img.samples();
            // Premultiplied accumulation when there is alpha.
            let at = |x: usize, y: usize, c: usize| -> f64 {
                let px = &s[(y * sw as usize + x) * ch..][..ch];
                if alpha && c < 3 {
                    px[c] as f64 * px[3] as f64
                } else {
                    px[c] as f64
                }
            };
            // Horizontal pass into a `w × sh` buffer.
            let mut mid = vec![0.0f64; w as usize * sh as usize * ch];
            for y in 0..sh as usize {
                for (x, (first, wts)) in xs.iter().enumerate() {
                    for c in 0..ch {
                        mid[(y * w as usize + x) * ch + c] = wts
                            .iter()
                            .enumerate()
                            .map(|(k, wt)| wt * at(first + k, y, c))
                            .sum();
                    }
                }
            }
            let mut out = deep::FloatImage::new(w, h, ch);
            for (y, (first, wts)) in ys.iter().enumerate() {
                for x in 0..w as usize {
                    let mut acc = [0.0f64; 4];
                    for (k, wt) in wts.iter().enumerate() {
                        let row = &mid[((first + k) * w as usize + x) * ch..][..ch];
                        for c in 0..ch {
                            acc[c] += wt * row[c];
                        }
                    }
                    let px = out.pixel_mut(x as u32, y as u32);
                    for c in 0..ch {
                        px[c] = if alpha && c < 3 {
                            if acc[3] > 0.0 { acc[c] / acc[3] } else { 0.0 }
                        } else {
                            acc[c]
                        } as f32;
                    }
                }
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_centres_take_the_lower_pixel() {
        let idx: Vec<u32> = (0..4).map(|x| Sample::source_index(x, 35, 70)).collect();
        assert_eq!(idx, [0, 2, 4, 6]);
        assert_eq!(Sample::source_index(9, 19, 70), 34);
        assert_eq!(Sample::source_index(0, 200, 1), 0);
        let w = box_weights(4, 9);
        assert_eq!(w[2].0, 0);
        assert!((w[2].1[0] - 0.25).abs() < 1e-12 && (w[2].1[1] - 0.75).abs() < 1e-12);
    }
}
