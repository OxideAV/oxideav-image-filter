//! Hald CLUT image-as-LUT colour grading (`-hald-clut`).
//!
//! Two-input filter: `src` is the image to be colour-graded; `dst`
//! is a Hald CLUT image (a 3-D RGB LUT laid out in 2-D). The standard
//! Hald CLUT layout for level `L` is a square image `LxL × LxL` (so
//! `L=8` ⇒ 64×64, `L=16` ⇒ 256×256, etc.) where the cube of `L³`
//! distinct colours is unrolled by:
//!
//! ```text
//! r ∈ [0, L²),  g ∈ [0, L²),  b implicit
//! pixel(x = r, y = g) where the green and blue axes are split across
//! row/column blocks of size L:
//!
//! cell_r = r % L,  cell_g = g % L,  cell_b = (r / L) + (g / L) * L
//! ```
//!
//! For each input pixel `(R, G, B)` the filter:
//!
//! 1. Maps each channel to a continuous index in `[0, L)`:
//!    `r = R * (L - 1) / 255`, etc. (kept as `f32`).
//! 2. Trilinearly samples the cube at `(r, g, b)` to interpolate
//!    between the 8 enclosing CLUT cells.
//! 3. Uses the resulting RGB as the output. Alpha (RGBA) is preserved.
//!
//! Clean-room implementation derived from the published Hald CLUT
//! layout convention (Strange Banana, 2003 — public-domain
//! description).
//!
//! # CLUT shape
//!
//! The CLUT image must be `(L²) × (L²)` for some `L >= 2`. The filter
//! infers `L` as `(round(cbrt(width × height))).round()` and verifies
//! that `L² = width = height`. Mismatched shapes return
//! `Error::invalid`.
//!
//! # Pixel formats
//!
//! - `src` and `dst` must be the same format.
//! - `Rgb24` / `Rgba`: full effect; alpha is preserved on RGBA.
//! - YUV / Gray8 return `Error::unsupported` (the cube is RGB-only).

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Hald CLUT image-as-LUT colour grading filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct HaldClut;

impl HaldClut {
    /// Construct a stateless Hald CLUT applier.
    pub fn new() -> Self {
        Self
    }
}

fn infer_level(w: u32, h: u32) -> Option<u32> {
    if w != h {
        return None;
    }
    // L² = w. L = round(sqrt(w)). Verify L² = w.
    let l = (w as f64).sqrt().round() as u32;
    if l == 0 {
        return None;
    }
    if l * l != w {
        return None;
    }
    if l < 2 {
        return None;
    }
    Some(l)
}

fn fetch(plane: &VideoPlane, x: u32, y: u32, bpp: usize) -> [u8; 4] {
    let base = (y as usize) * plane.stride + (x as usize) * bpp;
    match bpp {
        3 => [
            plane.data[base],
            plane.data[base + 1],
            plane.data[base + 2],
            255,
        ],
        4 => [
            plane.data[base],
            plane.data[base + 1],
            plane.data[base + 2],
            plane.data[base + 3],
        ],
        _ => unreachable!(),
    }
}

/// Sample the Hald CLUT at integer cube coordinates `(cr, cg, cb)`
/// for cube level `L`. The Hald layout puts:
///   `x = cr + (cb % L) * L`
///   `y = cg + (cb / L) * L`
fn sample_cube(plane: &VideoPlane, l: u32, bpp: usize, cr: u32, cg: u32, cb: u32) -> [u8; 4] {
    let cr = cr.min(l - 1);
    let cg = cg.min(l - 1);
    let cb = cb.min(l - 1);
    let x = cr + (cb % l) * l;
    let y = cg + (cb / l) * l;
    fetch(plane, x, y, bpp)
}

impl TwoInputImageFilter for HaldClut {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: HaldClut requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let l = infer_level(params.width, params.height).ok_or_else(|| {
            Error::invalid(format!(
                "oxideav-image-filter: HaldClut: dst must be a square Hald CLUT image \
                 of size L²×L² for some L≥2; got {}×{}",
                params.width, params.height
            ))
        })?;

        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let src_plane = &src.planes[0];
        let dst_plane = &dst.planes[0];
        let mut out = vec![0u8; row_bytes * h];

        let lf = (l - 1) as f32;

        for y in 0..h {
            let src_row = &src_plane.data[y * src_plane.stride..y * src_plane.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32 * lf / 255.0;
                let g = src_row[base + 1] as f32 * lf / 255.0;
                let b = src_row[base + 2] as f32 * lf / 255.0;

                let r0 = r.floor() as u32;
                let g0 = g.floor() as u32;
                let b0 = b.floor() as u32;
                let r1 = (r0 + 1).min(l - 1);
                let g1 = (g0 + 1).min(l - 1);
                let b1 = (b0 + 1).min(l - 1);
                let dr = r - r0 as f32;
                let dg = g - g0 as f32;
                let db = b - b0 as f32;

                // Eight corners of the unit cube around (r, g, b).
                let c000 = sample_cube(dst_plane, l, bpp, r0, g0, b0);
                let c100 = sample_cube(dst_plane, l, bpp, r1, g0, b0);
                let c010 = sample_cube(dst_plane, l, bpp, r0, g1, b0);
                let c110 = sample_cube(dst_plane, l, bpp, r1, g1, b0);
                let c001 = sample_cube(dst_plane, l, bpp, r0, g0, b1);
                let c101 = sample_cube(dst_plane, l, bpp, r1, g0, b1);
                let c011 = sample_cube(dst_plane, l, bpp, r0, g1, b1);
                let c111 = sample_cube(dst_plane, l, bpp, r1, g1, b1);

                // Trilinear interpolation per channel.
                let lerp =
                    |a: u8, b: u8, t: f32| (a as f32 * (1.0 - t) + b as f32 * t).clamp(0.0, 255.0);
                let mut out_rgb = [0f32; 3];
                for ch in 0..3 {
                    let c00 = lerp(c000[ch], c100[ch], dr);
                    let c01 = lerp(c001[ch], c101[ch], dr);
                    let c10 = lerp(c010[ch], c110[ch], dr);
                    let c11 = lerp(c011[ch], c111[ch], dr);
                    let c0 = c00 * (1.0 - dg) + c10 * dg;
                    let c1 = c01 * (1.0 - dg) + c11 * dg;
                    out_rgb[ch] = c0 * (1.0 - db) + c1 * db;
                }
                dst_row[base] = out_rgb[0].round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = out_rgb[1].round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = out_rgb[2].round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    dst_row[base + 3] = src_row[base + 3];
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

/// Build an identity Hald CLUT image of level `L` in `Rgb24` /
/// `Rgba`. Exposed for tests + downstream identity-pipeline scaffolding.
pub fn build_identity_hald(l: u32, has_alpha: bool) -> (VideoFrame, u32) {
    let l = l.max(2);
    let side = l * l;
    let bpp = if has_alpha { 4 } else { 3 };
    let stride = (side as usize) * bpp;
    let mut data = vec![0u8; stride * side as usize];
    for cb in 0..l {
        for cg in 0..l {
            for cr in 0..l {
                let x = cr + (cb % l) * l;
                let y = cg + (cb / l) * l;
                let r = (cr * 255 / (l - 1)) as u8;
                let g = (cg * 255 / (l - 1)) as u8;
                let b = (cb * 255 / (l - 1)) as u8;
                let base = (y as usize) * stride + (x as usize) * bpp;
                data[base] = r;
                data[base + 1] = g;
                data[base + 2] = b;
                if has_alpha {
                    data[base + 3] = 255;
                }
            }
        }
    }
    let frame = VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    };
    (frame, side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hald_clut_passes_input_through() {
        let l = 8u32;
        let (clut, side) = build_identity_hald(l, false);
        // Make a small RGB input where each pixel hits a distinct cube
        // sample. Use a 4×4 input and the 64×64 identity CLUT; both
        // ports must agree on shape per the trait contract, so mock
        // both at the CLUT's 64×64 shape.
        let w = side;
        let h = side;
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[
                    ((x * 255) / (w - 1)) as u8,
                    ((y * 255) / (h - 1)) as u8,
                    (((x + y) % 256) as u8),
                ]);
            }
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        };
        let out = HaldClut::new().apply_two(&input, &clut, p).unwrap();
        // For an identity Hald CLUT every input ⇒ output (within
        // round-off).
        let mut max_err = 0i32;
        for (a, b) in input.planes[0].data.iter().zip(out.planes[0].data.iter()) {
            let d = (*a as i32 - *b as i32).abs();
            if d > max_err {
                max_err = d;
            }
        }
        assert!(
            max_err <= 2,
            "identity CLUT should be near-identity, max err = {max_err}"
        );
    }

    #[test]
    fn negative_clut_inverts_colour() {
        // Build an identity-then-inverted CLUT: every cube cell stores
        // the COMPLEMENT of its position. Output should be 255 - input.
        let l = 4u32;
        let side = l * l;
        let bpp = 3usize;
        let stride = (side as usize) * bpp;
        let mut data = vec![0u8; stride * side as usize];
        for cb in 0..l {
            for cg in 0..l {
                for cr in 0..l {
                    let x = cr + (cb % l) * l;
                    let y = cg + (cb / l) * l;
                    let r = 255 - (cr * 255 / (l - 1)) as u8;
                    let g = 255 - (cg * 255 / (l - 1)) as u8;
                    let b = 255 - (cb * 255 / (l - 1)) as u8;
                    let base = (y as usize) * stride + (x as usize) * bpp;
                    data[base] = r;
                    data[base + 1] = g;
                    data[base + 2] = b;
                }
            }
        }
        let clut = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        };
        // Test on a side×side input filled with mid-grey.
        let inp_data = vec![128u8; (side * side * 3) as usize];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (side * 3) as usize,
                data: inp_data,
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: side,
            height: side,
        };
        let out = HaldClut::new().apply_two(&input, &clut, p).unwrap();
        // For input 128, output ≈ 255 - 128 = 127 (within ±2).
        for chunk in out.planes[0].data.chunks(3) {
            for &v in chunk {
                assert!(
                    (v as i32 - 127).abs() <= 2,
                    "expected ~127 from inverted CLUT, got {v}"
                );
            }
        }
    }

    #[test]
    fn rejects_non_square_clut() {
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 9,
                data: vec![0; 9 * 6],
            }],
        };
        let input = dst.clone();
        let p = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 3,
            height: 6,
        };
        let err = HaldClut::new().apply_two(&input, &dst, p).unwrap_err();
        assert!(format!("{err}").contains("HaldClut"));
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
        let input = dst.clone();
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = HaldClut::new().apply_two(&input, &dst, p).unwrap_err();
        assert!(format!("{err}").contains("HaldClut"));
    }

    #[test]
    fn alpha_preserved_on_rgba() {
        let l = 4u32;
        let (clut, side) = build_identity_hald(l, true);
        // Input filled with A varying.
        let mut data = Vec::with_capacity((side * side * 4) as usize);
        for i in 0..(side * side) {
            data.extend_from_slice(&[100, 50, 200, (i % 256) as u8]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (side * 4) as usize,
                data,
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: side,
            height: side,
        };
        let out = HaldClut::new().apply_two(&input, &clut, p).unwrap();
        for i in 0..(side * side) as usize {
            assert_eq!(out.planes[0].data[i * 4 + 3], (i % 256) as u8);
        }
    }
}
