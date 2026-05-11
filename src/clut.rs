//! 1-D Colour Look-Up Table (`-clut`).
//!
//! Two-input filter: `src` is the image being recoloured and `dst`
//! is a 1-D CLUT image (a thin strip read row-major). For every input
//! pixel the per-channel intensity is treated as an index into the
//! CLUT (rescaled to the CLUT's length) and the result is the CLUT
//! sample at that index.
//!
//! Mirrors ImageMagick's `-clut`. The input order matches `Composite`:
//! `src` (foreground = image being recoloured) on input port 0, `dst`
//! (background = CLUT) on input port 1, output emits on port 0.
//!
//! Clean-room implementation derived from the textbook description of
//! a 1-D CLUT lookup:
//!
//! 1. Flatten the CLUT into a sequence of N samples by reading it
//!    row-major. For a `Wx1` CLUT this is just the row; for a `WxH`
//!    CLUT it concatenates rows top-to-bottom.
//! 2. For every input pixel:
//!    - per-channel: `idx = sample * (N - 1) / 255`.
//!    - `out_channel = clut[idx][channel]`.
//! 3. Alpha (RGBA) and chroma (YUV) pass through unchanged.
//!
//! # Pixel formats
//!
//! - `Gray8`: lookup uses the single channel; output samples are the
//!   CLUT's luma proxy.
//! - `Rgb24` / `Rgba`: per-channel lookup. The CLUT must be in the
//!   same format (Rgb24/Rgba) so per-channel samples line up.
//! - YUV / other formats return `Error::unsupported`.

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// 1-D colour look-up table filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Clut;

impl Clut {
    /// Construct a stateless CLUT applier.
    pub fn new() -> Self {
        Self
    }
}

fn clut_samples(
    dst: &VideoFrame,
    params: VideoStreamParams,
) -> Result<(Vec<[u8; 4]>, usize, bool), Error> {
    // Returns (palette, bpp, has_alpha) where `palette` is the flat
    // sequence of CLUT samples as `[R, G, B, A]` quads (alpha = 255 on
    // non-RGBA inputs).
    let (bpp, has_alpha) = match params.format {
        PixelFormat::Gray8 => (1usize, false),
        PixelFormat::Rgb24 => (3usize, false),
        PixelFormat::Rgba => (4usize, true),
        other => {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Clut requires Gray8/Rgb24/Rgba, got {other:?}"
            )));
        }
    };
    let w = params.width as usize;
    let h = params.height as usize;
    let plane = &dst.planes[0];
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * bpp];
        for x in 0..w {
            let base = x * bpp;
            let (r, g, b, a) = match bpp {
                1 => (row[base], row[base], row[base], 255),
                3 => (row[base], row[base + 1], row[base + 2], 255),
                4 => (row[base], row[base + 1], row[base + 2], row[base + 3]),
                _ => unreachable!(),
            };
            out.push([r, g, b, a]);
        }
    }
    Ok((out, bpp, has_alpha))
}

impl TwoInputImageFilter for Clut {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Gray8 => (1usize, false),
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: Clut requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let (palette, _, _) = clut_samples(dst, params)?;
        if palette.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Clut received an empty CLUT (dst frame has no samples)",
            ));
        }
        let n = palette.len();
        // Avoid division-by-zero when n == 1: every lookup snaps to that
        // single CLUT entry.
        let denom = (n - 1).max(1) as u32;

        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let src_plane = &src.planes[0];
        let mut out = vec![0u8; row_bytes * h];

        for y in 0..h {
            let src_row = &src_plane.data[y * src_plane.stride..y * src_plane.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                match bpp {
                    1 => {
                        let idx = (src_row[base] as u32 * denom / 255).min(denom) as usize;
                        // Project the CLUT sample to luma via the same
                        // luma proxy used elsewhere in the crate.
                        let p = palette[idx];
                        dst_row[base] =
                            ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8;
                    }
                    3 => {
                        let r = (src_row[base] as u32 * denom / 255).min(denom) as usize;
                        let g = (src_row[base + 1] as u32 * denom / 255).min(denom) as usize;
                        let b = (src_row[base + 2] as u32 * denom / 255).min(denom) as usize;
                        dst_row[base] = palette[r][0];
                        dst_row[base + 1] = palette[g][1];
                        dst_row[base + 2] = palette[b][2];
                    }
                    4 => {
                        let r = (src_row[base] as u32 * denom / 255).min(denom) as usize;
                        let g = (src_row[base + 1] as u32 * denom / 255).min(denom) as usize;
                        let b = (src_row[base + 2] as u32 * denom / 255).min(denom) as usize;
                        dst_row[base] = palette[r][0];
                        dst_row[base + 1] = palette[g][1];
                        dst_row[base + 2] = palette[b][2];
                        if has_alpha {
                            dst_row[base + 3] = src_row[base + 3];
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(VideoFrame {
            pts: src.pts,
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
    fn identity_clut_passes_input_through() {
        // CLUT is a 16x1 strip with sample i = (i*17, i*17, i*17).
        // Per-channel lookup of any RGB value should reproduce the
        // original. The bare TwoInputImageFilter contract takes the
        // params for the SHARED shape — so we test on a 16×1 fixture.
        let mut clut_data = Vec::new();
        for i in 0..16u32 {
            let v = (i * 17) as u8; // 0, 17, 34, ... 255
            clut_data.extend_from_slice(&[v, v, v]);
        }
        let clut = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16 * 3,
                data: clut_data,
            }],
        };
        let input = rgb(16, 1, |x, _| {
            let v = (x * 17) as u8;
            [v, v, v]
        });
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 16,
            height: 1,
        };
        let out = Clut::new().apply_two(&input, &clut, p).unwrap();
        // Each input v = i*17 maps to CLUT idx = (v * 15 / 255) = i,
        // and CLUT[i] = (i*17, i*17, i*17). ⇒ identity.
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn negative_clut_inverts() {
        // The Clut implementation reads the dst plane via its width *
        // height, so both ports agree on shape (the 4×4 case here).
        // CLUT = 4-sample ramp 255..0 repeated across 4 rows.
        let input = rgb(4, 4, |x, _| {
            let v = (x * 60) as u8;
            [v, v, v]
        });
        let mut clut_data = Vec::new();
        for _ in 0..4 {
            for v in [255, 170, 85, 0u8] {
                clut_data.extend_from_slice(&[v, v, v]);
            }
        }
        let clut4 = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4 * 3,
                data: clut_data,
            }],
        };
        let p4 = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 4,
            height: 4,
        };
        let out = Clut::new().apply_two(&input, &clut4, p4).unwrap();
        // 4-sample CLUT ⇒ idx = sample * 15 / 255 ... but denom is
        // (n-1) = 15 (since n = 16 from 4×4 = 16). The lookup returns
        // monotone-decreasing samples for monotone-increasing input.
        // First pixel (v = 0) ⇒ idx 0 ⇒ palette[0] = 255.
        // Last pixel (v = 180) ⇒ idx near 10 ⇒ palette[10] = small.
        let last_first = out.planes[0].data[0];
        let last_pixel = out.planes[0].data[(3 * 3) as usize];
        assert!(
            last_first >= last_pixel,
            "expected monotone decreasing CLUT to invert ordering ({last_first} >= {last_pixel})"
        );
    }

    #[test]
    fn rejects_empty_clut() {
        // 0×0 dst frame ⇒ empty palette ⇒ invalid.
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 0,
                data: vec![],
            }],
        };
        let input = rgb(1, 1, |_, _| [0, 0, 0]);
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 0,
            height: 0,
        };
        let err = Clut::new().apply_two(&input, &dst, p).unwrap_err();
        assert!(format!("{err}").contains("Clut"));
    }

    #[test]
    fn rgba_alpha_preserved() {
        let mut clut_data = Vec::new();
        for _ in 0..16 {
            clut_data.extend_from_slice(&[100u8, 50, 200, 0]);
        }
        let clut = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: clut_data,
            }],
        };
        let mut input_data = Vec::new();
        for i in 0..16 {
            input_data.extend_from_slice(&[10u8, 20, 30, i as u8]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: input_data,
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 4,
            height: 4,
        };
        let out = Clut::new().apply_two(&input, &clut, p).unwrap();
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4 + 3], i as u8, "alpha at {i}");
            // Single-colour CLUT ⇒ every output rgb should be (100, 50, 200).
            assert_eq!(out.planes[0].data[i * 4], 100);
            assert_eq!(out.planes[0].data[i * 4 + 1], 50);
            assert_eq!(out.planes[0].data[i * 4 + 2], 200);
        }
    }

    #[test]
    fn rejects_yuv() {
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
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
        let err = Clut::new().apply_two(&input, &dst, p).unwrap_err();
        assert!(format!("{err}").contains("Clut"));
    }
}
