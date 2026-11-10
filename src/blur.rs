//! Gaussian blur — separable 1D kernel applied row-wise then column-wise
//! on each selected plane.

use crate::{PixelFilter, Planes, VideoStreamParams};
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

impl PixelFilter for Blur {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::CLASSIC_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        if crate::deep::is_float(params.format) {
            // Packed float: `Luma` / `All` / `Index(0)` mean the whole
            // picture, `Chroma` / `Index(n > 0)` nothing.
            if !self.planes.matches(0, 1) {
                return Ok(input.clone());
            }
            let kernel = gaussian_kernel(self.radius, self.sigma);
            return crate::deep::run(input, params, |img| {
                Ok(crate::deep::convolve_separable_covered(
                    &img, &kernel, &kernel,
                ))
            });
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
            // `Rgba` blurs colour under its alpha (premultiplied).
            let covered = params.format == PixelFormat::Rgba;
            *plane = blur_plane(plane, pw, ph, bpp, &kernel, covered);
        }

        Ok(out)
    }
}

/// Produce a normalised 1D Gaussian kernel of length `2*radius + 1`.
///
/// Taps further than `8σ` from the centre weigh less than `e⁻³²` of the
/// centre tap — nothing at binary32 precision — so the kernel is cut
/// there; a radius far beyond the Gaussian's reach therefore costs
/// nothing. The kernel never exceeds 2²⁰ taps a side.
pub(crate) fn gaussian_kernel(radius: u32, sigma: f32) -> Vec<f32> {
    let reach = (8.0 * sigma as f64).ceil().clamp(1.0, (1u32 << 20) as f64) as u32;
    let r = radius.min(reach) as i32;
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

fn blur_plane(
    plane: &VideoPlane,
    w: u32,
    h: u32,
    bpp: usize,
    kernel: &[f32],
    covered: bool,
) -> VideoPlane {
    let alpha = if covered {
        crate::kernel::Alpha::Premultiplied
    } else {
        crate::kernel::Alpha::Independent
    };
    // Rows then columns; the intermediate stays in binary32, so the
    // result is the 2-D Gaussian rounded once.
    let data = crate::kernel::convolve_separable(
        &plane.data,
        plane.stride,
        w as usize,
        h as usize,
        bpp,
        kernel,
        kernel,
        alpha,
    );
    VideoPlane {
        stride: w as usize * bpp,
        data,
    }
}

pub(crate) fn bytes_per_plane_pixel(fmt: PixelFormat, plane_idx: usize) -> usize {
    // Bytes one sample position of the plane occupies, from the format's
    // own row rule (3 for `Rgb24`, 2 for a 16-bit word, 2 for an `Nv12`
    // chroma pair, …). Side-channel entries and out-of-range planes
    // never reach here; fall back to one byte.
    fmt.plane_row_bytes(plane_idx, 1).unwrap_or(1)
}

pub(crate) fn chroma_subsampling(fmt: PixelFormat) -> (u32, u32) {
    // Divisors (`1 << shift`) of the format's chroma grid; `(1, 1)` for
    // every layout without a distinct chroma grid.
    match fmt.chroma_subsampling() {
        Some((sx, sy)) => (1 << sx, 1 << sy),
        None => (1, 1),
    }
}

pub(crate) fn plane_dims(
    w: u32,
    h: u32,
    fmt: PixelFormat,
    plane_idx: usize,
    _cx: u32,
    _cy: u32,
) -> (u32, u32) {
    // The plane's extent from the format's own geometry rule
    // (`ceil(w / sx) × ceil(h / sy)` on the chroma planes of a
    // subsampled layout, full size everywhere else — alpha planes
    // included).
    fmt.plane_dimensions(plane_idx, w, h).unwrap_or((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;

    fn gray_frame(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
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
