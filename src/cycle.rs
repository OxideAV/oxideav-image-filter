//! Channel-value cycle: rotate every per-channel sample by `N` modulo 256.
//!
//! ImageMagick analogue: `-cycle N` (paletted images rotate the
//! colourmap; for direct-colour images the operator is defined on the
//! per-channel byte value with wrap-around). This crate produces direct
//! colour, so we apply the rotation to the raw sample bytes.
//!
//! The transform is `out = (src + amount) mod 256` per channel — a
//! pure 256-byte LUT applied to every R/G/B (and Y) byte. Alpha and
//! chroma are preserved.
//!
//! `amount = 0` is identity. `amount = 128` swaps the upper and lower
//! halves of the value range (a "negate-the-sign-bit" effect that
//! produces visually striking, periodic colour shifts on continuous-tone
//! images). Negative `amount` rotates downward.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame};

/// Per-channel modular value rotation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cycle {
    /// Signed rotation amount, taken modulo 256. `1` adds 1 to every
    /// channel byte (255 wraps to 0); `-1` subtracts.
    pub amount: i32,
}

impl Cycle {
    /// Build a cycle filter rotating by `amount` (signed; modular 256).
    pub fn new(amount: i32) -> Self {
        Self { amount }
    }
}

impl ImageFilter for Cycle {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Cycle does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let mut out = input.clone();
        let lut = build_lut(self.amount);

        match params.format {
            PixelFormat::Gray8 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgb24 => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    for v in &mut p.data[y * stride..y * stride + 3 * w] {
                        *v = lut[*v as usize];
                    }
                }
            }
            PixelFormat::Rgba => {
                let p = &mut out.planes[0];
                let stride = p.stride;
                for y in 0..h {
                    let row = &mut p.data[y * stride..y * stride + 4 * w];
                    for x in 0..w {
                        for ch in 0..3 {
                            let v = &mut row[4 * x + ch];
                            *v = lut[*v as usize];
                        }
                        // alpha preserved
                    }
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let yp = &mut out.planes[0];
                let stride = yp.stride;
                for y in 0..h {
                    for v in &mut yp.data[y * stride..y * stride + w] {
                        *v = lut[*v as usize];
                    }
                }
                // chroma untouched
            }
            _ => unreachable!("is_supported gate already rejected this format"),
        }
        Ok(out)
    }
}

fn build_lut(amount: i32) -> [u8; 256] {
    let a = amount.rem_euclid(256) as u32;
    let mut lut = [0u8; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = ((i as u32 + a) & 0xFF) as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn gray_strip(values: &[u8]) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: values.len(),
                data: values.to_vec(),
            }],
        }
    }

    fn run(amount: i32, values: &[u8]) -> Vec<u8> {
        let input = gray_strip(values);
        let out = Cycle::new(amount)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Gray8,
                    width: values.len() as u32,
                    height: 1,
                },
            )
            .unwrap();
        out.planes[0].data.clone()
    }

    #[test]
    fn zero_is_identity() {
        let v = vec![0, 50, 128, 200, 255];
        assert_eq!(run(0, &v), v);
    }

    #[test]
    fn positive_wraps_at_255() {
        assert_eq!(run(10, &[0, 100, 250, 255]), vec![10, 110, 4, 9]);
    }

    #[test]
    fn negative_wraps_through_zero() {
        assert_eq!(run(-5, &[0, 5, 100]), vec![251, 0, 95]);
    }

    #[test]
    fn full_period_is_identity() {
        let v = (0u8..=255u8).collect::<Vec<_>>();
        assert_eq!(run(256, &v), v);
        assert_eq!(run(-256, &v), v);
        assert_eq!(run(512, &v), v);
    }

    #[test]
    fn rgba_preserves_alpha() {
        let data: Vec<u8> = (0..4).flat_map(|i| [10u8, 20, 30, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Cycle::new(5)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        for i in 0..4 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert_eq!(px[0], 15);
            assert_eq!(px[1], 25);
            assert_eq!(px[2], 35);
            assert_eq!(px[3], i as u8, "alpha preserved");
        }
    }

    #[test]
    fn yuv_only_touches_y() {
        let y: Vec<u8> = vec![10, 20, 30, 40];
        let cb = vec![100u8; 1];
        let cr = vec![200u8; 1];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 2, data: y },
                VideoPlane {
                    stride: 1,
                    data: cb.clone(),
                },
                VideoPlane {
                    stride: 1,
                    data: cr.clone(),
                },
            ],
        };
        let out = Cycle::new(1)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![11, 21, 31, 41]);
        assert_eq!(out.planes[1].data, cb);
        assert_eq!(out.planes[2].data, cr);
    }
}
