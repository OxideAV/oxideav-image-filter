//! LSB-plane steganography embed (ImageMagick `-stegano offset`).
//!
//! Two-input filter that hides the most-significant bits of one image
//! (`src` — the "watermark") inside the least-significant bit-plane of
//! another (`dst` — the "carrier"). On output the carrier visually
//! looks unchanged but the per-pixel LSBs of each channel now encode
//! the watermark's high bits.
//!
//! Per-channel embed (for each colour byte at each pixel):
//!
//! ```text
//! out_byte = (dst_byte & 0xFE) | (src_byte >> 7)
//! ```
//!
//! `src` and `dst` must agree on `(format, width, height)` — same
//! invariant as [`Composite`](crate::composite::Composite). Alpha is
//! preserved (alpha-channel byte is copied through from `dst`).
//!
//! `offset` is a scalar bit-position shift; the default of `7` puts
//! `src`'s MSB into `dst`'s LSB. `offset = 0` would embed `src`'s LSB
//! into `dst`'s LSB (identity for already-clean carriers).
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Other formats return
//! [`Error::unsupported`].

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// LSB-plane steganographic embed.
#[derive(Clone, Copy, Debug)]
pub struct Stegano {
    /// Bit position within each `src` byte that gets stamped into the
    /// LSB of the matching `dst` byte. Defaults to 7 (the MSB) — gives
    /// the most visible reconstruction of the watermark when later
    /// extracted, at the cost of mild perceptual artefacts in `dst`.
    pub offset: u8,
}

impl Default for Stegano {
    fn default() -> Self {
        Self { offset: 7 }
    }
}

impl Stegano {
    /// Build a Stegano filter that takes bit `offset` (0 = LSB, 7 = MSB)
    /// from each `src` channel byte and writes it into the LSB of the
    /// matching `dst` channel byte. `offset > 7` is clamped to 7.
    pub fn new(offset: u8) -> Self {
        Self {
            offset: offset.min(7),
        }
    }
}

impl TwoInputImageFilter for Stegano {
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
                    "oxideav-image-filter: Stegano requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if src.planes.is_empty() || dst.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Stegano requires non-empty src and dst planes",
            ));
        }
        let row_bytes = w * bpp;
        let src_plane = &src.planes[0];
        let dst_plane = &dst.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        let shift = self.offset.min(7);
        for y in 0..h {
            let s_row = &src_plane.data[y * src_plane.stride..y * src_plane.stride + row_bytes];
            let d_row = &dst_plane.data[y * dst_plane.stride..y * dst_plane.stride + row_bytes];
            let o_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                // Carry the colour channels through the embed; the
                // alpha byte (if any) copies straight from `dst` so we
                // don't corrupt transparency.
                let colour_lim = if has_alpha { bpp - 1 } else { bpp };
                for ch in 0..colour_lim {
                    let bit = (s_row[base + ch] >> shift) & 1;
                    o_row[base + ch] = (d_row[base + ch] & 0xFE) | bit;
                }
                if has_alpha {
                    o_row[base + bpp - 1] = d_row[base + bpp - 1];
                }
            }
        }
        Ok(VideoFrame {
            pts: dst.pts.or(src.pts),
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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn lsb_carries_src_msb() {
        // src = pure white (MSB=1), dst = pure 100 (binary 01100100, LSB=0).
        // After embed at offset 7, every output byte's LSB should be 1 —
        // so each byte is 101 (was 100, OR'd with bit 1).
        let src = rgb(2, 2, |_, _| [255, 255, 255]);
        let dst = rgb(2, 2, |_, _| [100, 100, 100]);
        let out = Stegano::default()
            .apply_two(&src, &dst, p_rgb(2, 2))
            .unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[101, 101, 101]);
        }
    }

    #[test]
    fn lsb_carries_src_zero() {
        // src = MSB-0 (e.g. 0), dst = 101 (LSB=1). After embed, the LSB
        // is forced to 0 → output = 100.
        let src = rgb(2, 2, |_, _| [0, 0, 0]);
        let dst = rgb(2, 2, |_, _| [101, 101, 101]);
        let out = Stegano::default()
            .apply_two(&src, &dst, p_rgb(2, 2))
            .unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[100, 100, 100]);
        }
    }

    #[test]
    fn alpha_passes_through_from_dst() {
        // 2x2 RGBA → row stride = 2 * 4 = 8.
        let src_data: Vec<u8> = (0..4).flat_map(|_| [255u8, 255, 255, 50]).collect();
        let dst_data: Vec<u8> = (0..4).flat_map(|_| [100u8, 100, 100, 200]).collect();
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: src_data,
            }],
        };
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: dst_data,
            }],
        };
        let out = Stegano::default()
            .apply_two(
                &src,
                &dst,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            // RGB bytes = 100 | 1 = 101; alpha straight from dst = 200.
            assert_eq!(chunk, &[101, 101, 101, 200]);
        }
    }

    #[test]
    fn output_visually_close_to_dst() {
        // The maximum perceptual change is 1 LSB per channel — so for a
        // random src/dst pair, no output byte may differ from dst by
        // more than 1.
        let src = rgb(4, 4, |x, y| [(x * 17 + y * 31) as u8, (y * 13) as u8, 42]);
        let dst = rgb(4, 4, |x, y| [(x * 11 + y * 7) as u8, (y * 5) as u8, 200]);
        let out = Stegano::default()
            .apply_two(&src, &dst, p_rgb(4, 4))
            .unwrap();
        for (o, d) in out.planes[0].data.iter().zip(dst.planes[0].data.iter()) {
            let delta = (*o as i32 - *d as i32).abs();
            assert!(delta <= 1, "delta {delta} > 1 (out={o}, dst={d})");
        }
    }

    #[test]
    fn rejects_unsupported_format() {
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let err = Stegano::default()
            .apply_two(
                &f,
                &f,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Stegano"));
    }

    #[test]
    fn offset_clamps_to_seven() {
        let s = Stegano::new(99);
        assert_eq!(s.offset, 7);
    }

    #[test]
    fn offset_zero_uses_src_lsb() {
        // offset = 0 embeds src LSB. dst = 100, src = 1 (LSB=1) →
        // out = (100 & 0xFE) | 1 = 101.
        let src = rgb(1, 1, |_, _| [1, 1, 1]);
        let dst = rgb(1, 1, |_, _| [100, 100, 100]);
        let out = Stegano::new(0).apply_two(&src, &dst, p_rgb(1, 1)).unwrap();
        assert_eq!(&out.planes[0].data[..3], &[101, 101, 101]);
    }
}
