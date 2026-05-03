//! Gaussian blur — separable 1D kernel applied row-wise then column-wise
//! on each selected plane.

use crate::{is_supported_format, ImageFilter, Planes, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Separable Gaussian blur.
///
/// The kernel is generated from `sigma`; the kernel radius is `radius`
/// samples on each side (total kernel length = `2*radius + 1`).
/// Sensible values: sigma ≈ radius / 2, e.g. `Blur::new(3).with_sigma(1.5)`.
///
/// The filter supports [`Planes`] for per-plane control — `Luma` blurs
/// only the Y plane, `Chroma` blurs only Cb/Cr, etc.
#[derive(Clone, Debug)]
pub struct Blur {
    radius: u32,
    sigma: f32,
    planes: Planes,
}

impl Blur {
    /// Create a blur with the given kernel radius (>= 1). Sigma defaults
    /// to `radius / 2` (a common heuristic).
    pub fn new(radius: u32) -> Self {
        let radius = radius.max(1);
        Self {
            radius,
            sigma: (radius as f32) * 0.5,
            planes: Planes::default(),
        }
    }

    /// Override sigma (must be > 0).
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma.max(f32::EPSILON);
        self
    }

    /// Restrict the filter to specific planes.
    pub fn with_planes(mut self, planes: Planes) -> Self {
        self.planes = planes;
        self
    }
}

impl ImageFilter for Blur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Blur does not yet handle {:?}",
                params.format
            )));
        }

        let kernel = gaussian_kernel(self.radius, self.sigma);
        let (cx, cy) = chroma_subsampling(params.format);
        let mut out = input.clone();

        for (idx, plane) in out.planes.iter_mut().enumerate() {
            if !self.planes.matches(idx, input.planes.len()) {
                continue;
            }
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            *plane = blur_plane(plane, pw, ph, bpp, &kernel);
        }

        Ok(out)
    }
}

/// Produce a normalised 1D Gaussian kernel of length `2*radius + 1`.
pub(crate) fn gaussian_kernel(radius: u32, sigma: f32) -> Vec<f32> {
    let r = radius as i32;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0f32;
    for i in -r..=r {
        let v = (-(i as f32).powi(2) / two_sigma_sq).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

fn blur_plane(plane: &VideoPlane, w: u32, h: u32, bpp: usize, kernel: &[f32]) -> VideoPlane {
    let w = w as usize;
    let h = h as usize;
    let radius = (kernel.len() - 1) / 2;
    let stride = plane.stride;

    // Pass 1: horizontal blur into a scratch buffer (stride = w * bpp).
    let mut horiz = vec![0u8; w * bpp * h];
    for y in 0..h {
        let row = &plane.data[y * stride..y * stride + w * bpp];
        let out_row = &mut horiz[y * w * bpp..(y + 1) * w * bpp];
        for x in 0..w {
            for ch in 0..bpp {
                let mut acc = 0.0f32;
                for (ki, kv) in kernel.iter().enumerate() {
                    let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                    acc += *kv * row[sx * bpp + ch] as f32;
                }
                out_row[x * bpp + ch] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Pass 2: vertical blur back into an output plane with tight stride.
    let mut out_data = vec![0u8; w * bpp * h];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..bpp {
                let mut acc = 0.0f32;
                for (ki, kv) in kernel.iter().enumerate() {
                    let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                    acc += *kv * horiz[sy * w * bpp + x * bpp + ch] as f32;
                }
                out_data[y * w * bpp + x * bpp + ch] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    VideoPlane {
        stride: w * bpp,
        data: out_data,
    }
}

pub(crate) fn bytes_per_plane_pixel(fmt: PixelFormat, plane_idx: usize) -> usize {
    match fmt {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            // All three YUV planar formats use 1 byte per sample per plane.
            let _ = plane_idx;
            1
        }
        _ => 1,
    }
}

pub(crate) fn chroma_subsampling(fmt: PixelFormat) -> (u32, u32) {
    match fmt {
        PixelFormat::Yuv420P => (2, 2),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv444P => (1, 1),
        _ => (1, 1),
    }
}

pub(crate) fn plane_dims(
    w: u32,
    h: u32,
    fmt: PixelFormat,
    plane_idx: usize,
    cx: u32,
    cy: u32,
) -> (u32, u32) {
    match fmt {
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P if plane_idx > 0 => {
            (w.div_ceil(cx), h.div_ceil(cy))
        }
        _ => (w, h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_frame(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        use oxideav_core::TimeBase;
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

    #[test]
    fn kernel_sums_to_one() {
        let k = gaussian_kernel(3, 1.5);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        assert_eq!(k.len(), 7);
        // Symmetric.
        for i in 0..3 {
            assert!((k[i] - k[6 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn blur_preserves_flat_frame() {
        let input = gray_frame(16, 16, |_, _| 200);
        let out = Blur::new(3)
            .with_sigma(1.5)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();
        for b in &out.planes[0].data {
            assert_eq!(*b, 200, "flat input should stay flat after blur");
        }
    }

    #[test]
    fn blur_smooths_impulse() {
        // Isolated bright pixel in the middle of a dark field.
        let input = gray_frame(21, 21, |x, y| if x == 10 && y == 10 { 255 } else { 0 });
        let out = Blur::new(3)
            .with_sigma(1.5)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: 21,
                    height: 21,
                },
            )
            .unwrap();
        // The centre pixel must have dropped (some mass smeared outward).
        let centre = out.planes[0].data[10 * 21 + 10];
        assert!(centre < 200 && centre > 10, "centre = {centre}");
        // Neighbouring pixel must have picked up some intensity.
        let neighbour = out.planes[0].data[10 * 21 + 11];
        assert!(neighbour > 0, "neighbour = {neighbour}");
    }

    #[test]
    fn plane_selector_luma_leaves_chroma_untouched() {
        use oxideav_core::TimeBase;
        let y: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let u = vec![77u8; 8 * 8];
        let v = vec![128u8; 8 * 8];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y,
                },
                VideoPlane { stride: 8, data: u },
                VideoPlane { stride: 8, data: v },
            ],
        };

        let out = Blur::new(2)
            .with_sigma(1.0)
            .with_planes(Planes::Luma)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 16,
                    height: 16,
                },
            )
            .unwrap();

        // Chroma planes must be byte-identical.
        assert_eq!(out.planes[1].data, vec![77u8; 8 * 8]);
        assert_eq!(out.planes[2].data, vec![128u8; 8 * 8]);
        // Luma plane should still have the gradient pattern (not all-zero).
        let luma_sum: u64 = out.planes[0].data.iter().map(|b| *b as u64).sum();
        assert!(luma_sum > 0);
    }
}
