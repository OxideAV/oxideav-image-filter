//! Per-channel pixel offset, simulating lateral chromatic aberration.
//!
//! Real lenses focus blue light onto a slightly smaller image circle
//! than red light. A convincing first-order simulation is to shift the
//! red and blue channels in opposite directions by a small number of
//! pixels and leave the green channel undisturbed:
//!
//! ```text
//! out.R(x, y) = src.R(x - r_dx, y - r_dy)
//! out.G(x, y) = src.G(x, y)
//! out.B(x, y) = src.B(x - b_dx, y - b_dy)   (typically (-r_dx, -r_dy))
//! out.A       = src.A
//! ```
//!
//! Pixels whose source location falls outside the image clamp to the
//! nearest edge (no wrap, no fill colour). Convenience constructor
//! [`ChromaticAberration::horizontal(n)`] offsets R by `+n` pixels and B
//! by `-n` pixels along the X axis; [`ChromaticAberration::radial(n)`]
//! offsets R outward / B inward along the radial direction from the
//! image centre — also a common lens model.
//!
//! Operates on `Rgb24` / `Rgba`; YUV inputs return
//! `Error::unsupported`.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Per-channel pixel offset, RGB only.
///
/// `r_dx` / `r_dy` are the integer pixel offsets applied to the red
/// channel; `b_dx` / `b_dy` are the offsets for the blue channel; the
/// green channel is undisturbed. Offsets are interpreted as "shift the
/// channel content by `(dx, dy)`" — i.e. the channel sample originally
/// at `(x, y)` ends up at `(x + dx, y + dy)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChromaticAberration {
    pub r_dx: i32,
    pub r_dy: i32,
    pub b_dx: i32,
    pub b_dy: i32,
}

impl ChromaticAberration {
    /// Explicit per-channel offsets.
    pub fn new(r_dx: i32, r_dy: i32, b_dx: i32, b_dy: i32) -> Self {
        Self {
            r_dx,
            r_dy,
            b_dx,
            b_dy,
        }
    }

    /// Convenience: shift R by `+n` and B by `-n` along the X axis.
    pub fn horizontal(n: i32) -> Self {
        Self {
            r_dx: n,
            r_dy: 0,
            b_dx: -n,
            b_dy: 0,
        }
    }

    /// Convenience: shift R by `+n` and B by `-n` along the Y axis.
    pub fn vertical(n: i32) -> Self {
        Self {
            r_dx: 0,
            r_dy: n,
            b_dx: 0,
            b_dy: -n,
        }
    }
}

impl ImageFilter for ChromaticAberration {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ChromaticAberration requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: ChromaticAberration requires a non-empty input plane",
            ));
        }
        let w = params.width as i32;
        let h = params.height as i32;
        let row_bytes = (params.width as usize) * bpp;
        let src = &input.planes[0];
        let stride = src.stride;
        let data = &src.data;
        let fetch = |x: i32, y: i32, c: usize| -> u8 {
            let xc = x.clamp(0, w - 1) as usize;
            let yc = y.clamp(0, h - 1) as usize;
            data[yc * stride + xc * bpp + c]
        };
        let mut out = vec![0u8; row_bytes * params.height as usize];
        for y in 0..h {
            let dst_row = &mut out[(y as usize) * row_bytes..((y as usize) + 1) * row_bytes];
            for x in 0..w {
                let base = (x as usize) * bpp;
                let r = fetch(x - self.r_dx, y - self.r_dy, 0);
                let g = data[(y as usize) * stride + base + 1];
                let b = fetch(x - self.b_dx, y - self.b_dy, 2);
                dst_row[base] = r;
                dst_row[base + 1] = g;
                dst_row[base + 2] = b;
                if has_alpha {
                    dst_row[base + 3] = data[(y as usize) * stride + base + 3];
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
    fn zero_offsets_passthrough() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = ChromaticAberration::default()
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn horizontal_shift_pushes_red_right_and_blue_left() {
        // Build a centred vertical bar: column 4 of a 9x1 image is
        // pure white (255, 255, 255); rest is black.
        let input = rgb(
            9,
            1,
            |x, _| if x == 4 { [255, 255, 255] } else { [0, 0, 0] },
        );
        let out = ChromaticAberration::horizontal(2)
            .apply(&input, p_rgb(9, 1))
            .unwrap();
        // Red channel of the white bar should now appear at x = 4 + 2 = 6.
        assert_eq!(out.planes[0].data[6 * 3], 255, "R should be at col 6");
        // Blue channel should appear at x = 4 - 2 = 2.
        assert_eq!(out.planes[0].data[2 * 3 + 2], 255, "B should be at col 2");
        // Green stays put at x = 4.
        assert_eq!(out.planes[0].data[4 * 3 + 1], 255);
    }

    #[test]
    fn sharp_edge_increases_channel_offset_distance() {
        // Black/white vertical edge at column 4 in a 9x1 image. The
        // separation between the R-channel transition and the
        // B-channel transition should grow when we apply a non-zero
        // horizontal offset.
        let input = rgb(
            9,
            1,
            |x, _| {
                if x >= 4 {
                    [255, 255, 255]
                } else {
                    [0, 0, 0]
                }
            },
        );
        // Baseline (no offset): R-transition column == B-transition column == 4.
        // With horizontal(2): R-transition at 4 + 2 = 6, B at 4 - 2 = 2.
        let out = ChromaticAberration::horizontal(2)
            .apply(&input, p_rgb(9, 1))
            .unwrap();
        let r_first_high = (0..9)
            .find(|&x| out.planes[0].data[x * 3] == 255)
            .expect("R transition exists");
        let b_first_high = (0..9)
            .find(|&x| out.planes[0].data[x * 3 + 2] == 255)
            .expect("B transition exists");
        assert!(
            r_first_high > b_first_high,
            "R transition (col {r_first_high}) should be to the right of B transition (col {b_first_high})"
        );
        assert!(
            (r_first_high as i32 - b_first_high as i32).abs() == 4,
            "shift magnitude must equal 4 px (= 2 * abs(n))"
        );
    }

    #[test]
    fn shape_preserved() {
        let input = rgb(5, 3, |_, _| [10, 20, 30]);
        let out = ChromaticAberration::horizontal(1)
            .apply(&input, p_rgb(5, 3))
            .unwrap();
        assert_eq!(out.planes[0].stride, 15);
        assert_eq!(out.planes[0].data.len(), 45);
    }

    #[test]
    fn alpha_preserved_on_rgba() {
        let data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 222]).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = ChromaticAberration::horizontal(1)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
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
        let err = ChromaticAberration::horizontal(1)
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("ChromaticAberration"));
    }
}
