//! Unsharp-mask sharpening with built-in parameters.
//!
//! Formula (clean-room, standard unsharp mask):
//! `out = clamp(orig + (orig - blurred) * amount)`
//!
//! The blurred version is produced with the same separable Gaussian
//! kernel used by [`Blur`](crate::Blur). The filter touches the luma
//! plane on YUV inputs (chroma is left alone) and every RGB channel on
//! RGB/RGBA inputs.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, gaussian_kernel, plane_dims};
use crate::{is_supported, Blur, ImageFilter, Planes};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Sharpen an image via unsharp-mask. See the module docs for the
/// exact formula.
#[derive(Clone, Debug)]
pub struct Sharpen {
    pub radius: u32,
    pub sigma: f32,
    pub amount: f32,
}

impl Default for Sharpen {
    fn default() -> Self {
        Self {
            radius: 1,
            sigma: 0.5,
            amount: 1.0,
        }
    }
}

impl Sharpen {
    /// Build with explicit radius + sigma (amount defaults to 1.0).
    pub fn new(radius: u32, sigma: f32) -> Self {
        Self {
            radius: radius.max(1),
            sigma: sigma.max(f32::EPSILON),
            amount: 1.0,
        }
    }

    /// Override the sharpen strength (0.0 = identity, 1.0 = classic).
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.amount = amount;
        self
    }
}

impl ImageFilter for Sharpen {
    fn apply(&self, input: &VideoFrame) -> Result<VideoFrame, Error> {
        if !is_supported(input) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Sharpen does not yet handle {:?}",
                input.format
            )));
        }

        // For YUV, only sharpen luma (plane 0); for RGB/RGBA/Gray, blur
        // all channels.
        let planes_selector = match input.format {
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => Planes::Luma,
            _ => Planes::All,
        };

        let blurred = Blur::new(self.radius)
            .with_sigma(self.sigma)
            .with_planes(planes_selector)
            .apply(input)?;

        // Short-circuit if amount is effectively zero — avoid re-walking
        // the frame.
        if self.amount.abs() < f32::EPSILON {
            return Ok(input.clone());
        }

        let (cx, cy) = chroma_subsampling(input.format);
        let mut out = input.clone();
        for (idx, plane) in out.planes.iter_mut().enumerate() {
            if !planes_selector.matches(idx, input.planes.len()) {
                continue;
            }
            let (pw, ph) = plane_dims(input.width, input.height, input.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(input.format, idx);
            *plane = unsharp_mask_plane(
                &input.planes[idx],
                &blurred.planes[idx],
                pw as usize,
                ph as usize,
                bpp,
                self.amount,
                0,
            );
        }

        Ok(out)
    }
}

/// Shared unsharp-mask core used by both [`Sharpen`] and
/// [`Unsharp`](crate::Unsharp). `threshold` gates the mask on absolute
/// per-channel contrast — pixels whose `|orig - blurred|` is smaller
/// than the threshold are left untouched.
pub(crate) fn unsharp_mask_plane(
    orig: &VideoPlane,
    blurred: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    amount: f32,
    threshold: u8,
) -> VideoPlane {
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    for y in 0..h {
        let src_row = &orig.data[y * orig.stride..y * orig.stride + row_bytes];
        let blur_row = &blurred.data[y * blurred.stride..y * blurred.stride + row_bytes];
        let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
        for i in 0..row_bytes {
            let o = src_row[i] as f32;
            let b = blur_row[i] as f32;
            let diff = o - b;
            let abs = diff.abs();
            if abs < threshold as f32 {
                dst_row[i] = src_row[i];
            } else {
                let v = o + diff * amount;
                dst_row[i] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    VideoPlane {
        stride: row_bytes,
        data: out,
    }
}

/// Use the shared Gaussian kernel math from `blur.rs` — exposed here
/// for tests only.
#[cfg(test)]
#[allow(dead_code)]
fn kernel(radius: u32, sigma: f32) -> Vec<f32> {
    gaussian_kernel(radius, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

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
    fn amount_zero_is_identity() {
        let input = gray(8, 8, |x, y| ((x * 30 + y * 7) % 251) as u8);
        let out = Sharpen::new(2, 1.0).with_amount(0.0).apply(&input).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn sharpen_preserves_flat_frame() {
        let input = gray(16, 16, |_, _| 120);
        let out = Sharpen::new(2, 1.0).apply(&input).unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 120);
        }
    }

    #[test]
    fn sharpen_enhances_edge() {
        // Step from 60 to 200 at x=4 (horizontal).
        let input = gray(8, 1, |x, _| if x < 4 { 60 } else { 200 });
        let out = Sharpen::new(2, 1.0).with_amount(1.5).apply(&input).unwrap();
        // Pixel just inside bright side should be >= original 200.
        let right = out.planes[0].data[4];
        assert!(right >= 200, "right = {right}");
        // Pixel just inside dark side should be <= original 60.
        let left = out.planes[0].data[3];
        assert!(left <= 60, "left = {left}");
    }

    #[test]
    fn yuv_leaves_chroma_untouched() {
        let y: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let u = vec![77u8; 8 * 8];
        let v = vec![188u8; 8 * 8];
        let input = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 16,
            height: 16,
            pts: None,
            time_base: TimeBase::new(1, 1),
            planes: vec![
                VideoPlane { stride: 16, data: y },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };
        let out = Sharpen::new(2, 1.0).apply(&input).unwrap();
        assert_eq!(out.planes[1].data, vec![77u8; 8 * 8]);
        assert_eq!(out.planes[2].data, vec![188u8; 8 * 8]);
    }
}
