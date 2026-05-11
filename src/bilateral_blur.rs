//! Bilateral blur: edge-preserving Gaussian smoothing where each tap's
//! contribution is weighted by both spatial proximity (`sigma_spatial`)
//! and intensity proximity (`sigma_range`).
//!
//! Per output pixel `(x, y)` the formula is:
//!
//! ```text
//! out(x, y) = Σ_{i, j} src(x+i, y+j) * w(i, j) / Σ_{i, j} w(i, j)
//! w(i, j)  = exp(-(i² + j²) / (2 σ_s²)) ·
//!            exp(-(src(x+i, y+j) - src(x, y))² / (2 σ_r²))
//! ```
//!
//! Spatial taps within `[-radius, +radius]` are visited; the spatial
//! Gaussian is precomputed (`(2r+1)²` floats) but the range Gaussian is
//! evaluated per-tap because it depends on the centre intensity. Border
//! samples clamp to the nearest in-bounds coordinate.
//!
//! ImageMagick analogue: `convert in.png -filter Gaussian -define
//! convolve:bias=0 -define blur:bilateral=1 out.png` (the `bilateral=1`
//! marker switches IM's blur path to a bilateral kernel — there's no
//! standalone CLI flag, but the filter is well-known textbook material).
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba` (alpha pass-through). YUV
//! returns `Unsupported` — the range comparison only makes sense in a
//! gamma / luma-coherent space.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Edge-preserving bilateral blur.
#[derive(Clone, Copy, Debug)]
pub struct BilateralBlur {
    /// Spatial half-window in pixels. `radius == 0` is a no-op.
    pub radius: u32,
    /// Spatial Gaussian sigma. Defaults to `radius / 2`.
    pub sigma_spatial: f32,
    /// Range Gaussian sigma in 0..255 sample units. Smaller values
    /// preserve edges more aggressively.
    pub sigma_range: f32,
}

impl Default for BilateralBlur {
    fn default() -> Self {
        Self {
            radius: 3,
            sigma_spatial: 1.5,
            sigma_range: 30.0,
        }
    }
}

impl BilateralBlur {
    pub fn new(radius: u32, sigma_spatial: f32, sigma_range: f32) -> Self {
        Self {
            radius,
            sigma_spatial,
            sigma_range,
        }
    }

    fn spatial_kernel(&self) -> Vec<f32> {
        let r = self.radius as i32;
        let n = (2 * r + 1) as usize;
        let mut k = vec![0.0f32; n * n];
        let s = if self.sigma_spatial > 0.0 {
            self.sigma_spatial
        } else {
            (self.radius as f32 / 2.0).max(0.5)
        };
        let denom = 2.0 * s * s;
        for j in -r..=r {
            for i in -r..=r {
                let v = (-((i * i + j * j) as f32) / denom).exp();
                k[((j + r) as usize) * n + (i + r) as usize] = v;
            }
        }
        k
    }
}

fn clamp_idx(v: i32, max: i32) -> usize {
    v.clamp(0, max) as usize
}

#[allow(clippy::too_many_arguments)]
fn bilateral_plane(
    src: &VideoPlane,
    w: usize,
    h: usize,
    bpp: usize,
    channels: usize,
    radius: i32,
    spatial: &[f32],
    sigma_range: f32,
    has_alpha: bool,
) -> Vec<u8> {
    let stride_out = w * bpp;
    let mut out = vec![0u8; stride_out * h];
    let n = (2 * radius + 1) as usize;
    let inv_2sr2 = 1.0 / (2.0 * sigma_range.max(1e-3) * sigma_range.max(1e-3));
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let centre_base = (y as usize) * src.stride + (x as usize) * bpp;
            for c in 0..channels {
                let cv = src.data[centre_base + c] as f32;
                let mut sum = 0.0f32;
                let mut wsum = 0.0f32;
                for j in -radius..=radius {
                    let yy = clamp_idx(y + j, h as i32 - 1);
                    let row_off = yy * src.stride;
                    for i in -radius..=radius {
                        let xx = clamp_idx(x + i, w as i32 - 1);
                        let sample = src.data[row_off + xx * bpp + c] as f32;
                        let ws = spatial[((j + radius) as usize) * n + (i + radius) as usize];
                        let dr = sample - cv;
                        let wr = (-(dr * dr) * inv_2sr2).exp();
                        let wt = ws * wr;
                        sum += sample * wt;
                        wsum += wt;
                    }
                }
                let v = if wsum > 0.0 { sum / wsum } else { cv };
                out[(y as usize) * stride_out + (x as usize) * bpp + c] =
                    v.round().clamp(0.0, 255.0) as u8;
            }
            if has_alpha {
                // Pass alpha through unchanged.
                let a = src.data[centre_base + 3];
                out[(y as usize) * stride_out + (x as usize) * bpp + 3] = a;
            }
        }
    }
    out
}

impl ImageFilter for BilateralBlur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, channels, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, 1usize, false),
            PixelFormat::Rgb24 => (3, 3, false),
            PixelFormat::Rgba => (4, 3, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: BilateralBlur requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: BilateralBlur requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if self.radius == 0 {
            // Pass-through.
            let stride = w * bpp;
            let mut data = vec![0u8; stride * h];
            for y in 0..h {
                let s = &input.planes[0].data
                    [y * input.planes[0].stride..y * input.planes[0].stride + stride];
                let d = &mut data[y * stride..(y + 1) * stride];
                d.copy_from_slice(s);
            }
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane { stride, data }],
            });
        }
        let radius = self.radius as i32;
        let spatial = self.spatial_kernel();
        let data = bilateral_plane(
            &input.planes[0],
            w,
            h,
            bpp,
            channels,
            radius,
            &spatial,
            self.sigma_range,
            has_alpha,
        );
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w * bpp,
                data,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

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

    #[test]
    fn radius_zero_is_passthrough() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = BilateralBlur::new(0, 1.5, 30.0)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_input_stays_flat() {
        let input = rgb(6, 6, |_, _| [128, 64, 32]);
        let out = BilateralBlur::default().apply(&input, p_rgb(6, 6)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[128, 64, 32]);
        }
    }

    #[test]
    fn step_edge_is_preserved_more_than_box_blur() {
        // Vertical step edge: left half = 50, right half = 200.
        // A regular blur would smear the seam; bilateral with small
        // sigma_range should keep the boundary mostly sharp.
        let input = rgb(
            8,
            4,
            |x, _| {
                if x < 4 {
                    [50, 50, 50]
                } else {
                    [200, 200, 200]
                }
            },
        );
        let out = BilateralBlur::new(2, 1.5, 5.0)
            .apply(&input, p_rgb(8, 4))
            .unwrap();
        // Pixel just left of seam (3, 1) should still be near 50.
        let p_left = out.planes[0].data[(8 + 3) * 3];
        // Pixel just right of seam (4, 1) should still be near 200.
        let p_right = out.planes[0].data[(8 + 4) * 3];
        assert!(
            p_left < 90,
            "left of edge should stay dark with small sigma_range, got {p_left}"
        );
        assert!(
            p_right > 160,
            "right of edge should stay bright with small sigma_range, got {p_right}"
        );
    }

    #[test]
    fn alpha_pass_through_on_rgba() {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[100u8, 100, 100, 222]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = BilateralBlur::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn rejects_yuv() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let err = BilateralBlur::default()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("BilateralBlur"));
    }
}
