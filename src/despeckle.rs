//! Median-window despeckle: edge-preserving non-linear noise reduction.
//!
//! For each output pixel and each colour channel, take the median of
//! the `(2r+1) × (2r+1)` neighbourhood window centred on the pixel.
//! Border samples clamp to the nearest in-bounds coordinate. Alpha is
//! left intact on RGBA so transparency edges are not blurred. The
//! median is robust against impulsive (salt-and-pepper) noise while
//! preserving edges — a 5-sample noise spike inside an otherwise
//! constant window has no effect on the output, unlike a Gaussian blur.
//!
//! `radius = 0` is bit-exact identity (a 1×1 window is just the input).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Median-window despeckle filter.
///
/// `radius` is the half-window size in pixels: `0` is identity,
/// `1` is the standard 3×3 median, `2` is a 5×5 median, etc.
#[derive(Clone, Copy, Debug)]
pub struct Despeckle {
    pub radius: u32,
}

impl Default for Despeckle {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl Despeckle {
    /// Build a despeckle filter with explicit radius.
    pub fn new(radius: u32) -> Self {
        Self { radius }
    }
}

impl ImageFilter for Despeckle {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, denoise_channels) = match params.format {
            PixelFormat::Rgb24 => (3usize, 3usize),
            // RGBA: only despeckle R/G/B; alpha pass-through.
            PixelFormat::Rgba => (4usize, 3usize),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Despeckle requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };

        let w = params.width as usize;
        let h = params.height as usize;
        let src = &input.planes[0];
        let row_bytes = w * bpp;
        let r = self.radius as i32;

        let mut out = vec![0u8; row_bytes * h];
        // r == 0 ⇒ identity (window is 1×1, median is the pixel itself).
        if r == 0 {
            for y in 0..h {
                let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
                let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
                dst_row.copy_from_slice(src_row);
            }
            return Ok(VideoFrame {
                pts: input.pts,
                planes: vec![VideoPlane {
                    stride: row_bytes,
                    data: out,
                }],
            });
        }

        let window_side = (2 * r + 1) as usize;
        let window_n = window_side * window_side;
        // Fixed-capacity scratch (reused across pixels) to avoid an
        // allocation per output sample.
        let mut scratch: Vec<u8> = Vec::with_capacity(window_n);

        for y in 0..h {
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                for ch in 0..denoise_channels {
                    scratch.clear();
                    for dy in -r..=r {
                        let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let sy_off = sy * src.stride;
                        for dx in -r..=r {
                            let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            scratch.push(src.data[sy_off + sx * bpp + ch]);
                        }
                    }
                    // Quickselect via sort (window sizes are small —
                    // 9 / 25 / 49 — sort is fine).
                    scratch.sort_unstable();
                    dst_row[base + ch] = scratch[scratch.len() / 2];
                }
                if denoise_channels < bpp {
                    // Alpha pass-through (RGBA).
                    let src_off = y * src.stride + base + denoise_channels;
                    for ch in denoise_channels..bpp {
                        dst_row[base + ch] = src.data[src_off + ch - denoise_channels];
                    }
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_bytes,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = f(x, y);
                data.extend_from_slice(&[r, g, b]);
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
    fn radius_zero_is_passthrough() {
        let input = rgb(8, 8, |x, y| ((x * 30) as u8, (y * 30) as u8, 77));
        let out = Despeckle::new(0).apply(&input, params_rgb(8, 8)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn flat_image_is_unchanged() {
        let input = rgb(8, 8, |_, _| (50, 100, 200));
        let out = Despeckle::new(1).apply(&input, params_rgb(8, 8)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk[0], 50);
            assert_eq!(chunk[1], 100);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn isolated_speck_removed() {
        // 8×8 flat-grey image with one bright "speck" at (3, 3).
        let mut input = rgb(8, 8, |_, _| (100, 100, 100));
        let stride = input.planes[0].stride;
        let speck = 3 * stride + 3 * 3;
        input.planes[0].data[speck] = 255;
        input.planes[0].data[speck + 1] = 255;
        input.planes[0].data[speck + 2] = 255;

        let out = Despeckle::new(1).apply(&input, params_rgb(8, 8)).unwrap();
        // The 3×3 median of eight 100s and one 255 is 100 — speck gone.
        assert_eq!(out.planes[0].data[speck], 100);
        assert_eq!(out.planes[0].data[speck + 1], 100);
        assert_eq!(out.planes[0].data[speck + 2], 100);
    }

    #[test]
    fn edges_preserved_on_step_image() {
        // Vertical step: left half = 50, right half = 200. The median
        // filter must keep the discontinuity sharp (median of a window
        // straddling the boundary is still 50 or 200, never an
        // intermediate value).
        let input = rgb(
            8,
            8,
            |x, _| if x < 4 { (50, 50, 50) } else { (200, 200, 200) },
        );
        let out = Despeckle::new(1).apply(&input, params_rgb(8, 8)).unwrap();
        let stride = out.planes[0].stride;
        for y in 0..8usize {
            // Row interior: left pixel (1) should still be 50.
            assert_eq!(out.planes[0].data[y * stride + 3], 50);
            // Right pixel (6) should still be 200.
            assert_eq!(out.planes[0].data[y * stride + 4 * 3], 200);
        }
    }

    #[test]
    fn rgba_alpha_passes_through() {
        // 8×8 RGBA with a bright speck; alpha must be untouched.
        let mut data: Vec<u8> = (0..64).flat_map(|_| [100u8, 100, 100, 222]).collect();
        // Inject a speck at pixel (3, 3).
        let off = (3 * 8 + 3) * 4;
        data[off] = 255;
        data[off + 1] = 255;
        data[off + 2] = 255;
        // Tweak its alpha to confirm the filter copies it through.
        data[off + 3] = 17;
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let out = Despeckle::new(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 8,
                    height: 8,
                },
            )
            .unwrap();
        // RGB despeckled away.
        assert_eq!(out.planes[0].data[off], 100);
        assert_eq!(out.planes[0].data[off + 1], 100);
        assert_eq!(out.planes[0].data[off + 2], 100);
        // Alpha preserved (still 17 — not despeckled).
        assert_eq!(out.planes[0].data[off + 3], 17);
        // Other alphas remain 222.
        assert_eq!(out.planes[0].data[3], 222);
        assert_eq!(out.planes[0].data[4 * 4 + 3], 222);
    }

    #[test]
    fn rejects_yuv420p() {
        let input = VideoFrame {
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
        let err = Despeckle::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Despeckle"));
    }
}
