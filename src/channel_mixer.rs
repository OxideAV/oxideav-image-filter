//! Linear combination of source channels into destination channels.
//!
//! Generalisation of [`ColorMatrix`](crate::color_matrix::ColorMatrix)
//! that also touches the alpha channel. Where `ColorMatrix` is a 3×3
//! (plus optional 3-vector offset) operation on `R / G / B`,
//! `ChannelMixer` carries a 4×4 weight matrix plus a 4-vector offset:
//!
//! ```text
//! [ R' ]   [ a b c d ] [ R ]   [ ox ]
//! [ G' ] = [ e f g h ] [ G ] + [ oy ]
//! [ B' ]   [ i j k l ] [ B ]   [ oz ]
//! [ A' ]   [ m n o p ] [ A ]   [ ow ]
//! ```
//!
//! The matrix is laid out in row-major order (first row controls the
//! red channel, second green, third blue, fourth alpha). On `Rgb24`
//! inputs the alpha row is ignored (output has no alpha channel) and
//! the alpha column of the source is treated as `255`.
//!
//! Convenience constructors:
//! - [`ChannelMixer::identity()`] — pass-through.
//! - [`ChannelMixer::sepia_classic()`] — a popular sepia recipe that
//!   touches the green and blue rows too.
//! - [`ChannelMixer::from_color_matrix(m)`] — lift a 3×3 colour matrix
//!   into the 4×4 layout (alpha = identity).
//!
//! Operates on `Rgb24` / `Rgba`; YUV inputs return
//! `Error::unsupported` (use [`ColorMatrix`] which is also RGB-only).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// 4×4 channel-mixing matrix (row-major) + 4-vector offset.
#[derive(Clone, Copy, Debug)]
pub struct ChannelMixer {
    pub matrix: [[f32; 4]; 4],
    pub offset: [f32; 4],
}

impl Default for ChannelMixer {
    fn default() -> Self {
        Self::identity()
    }
}

impl ChannelMixer {
    /// Build a channel mixer with explicit matrix + offset.
    pub fn new(matrix: [[f32; 4]; 4], offset: [f32; 4]) -> Self {
        Self { matrix, offset }
    }

    /// Identity pass-through.
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            offset: [0.0; 4],
        }
    }

    /// Classic sepia recipe (R / G / B from the well-known Microsoft
    /// formula, alpha pass-through). Useful as a worked example of
    /// what a 4×4 mixer can express.
    pub fn sepia_classic() -> Self {
        Self {
            matrix: [
                [0.393, 0.769, 0.189, 0.0],
                [0.349, 0.686, 0.168, 0.0],
                [0.272, 0.534, 0.131, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            offset: [0.0; 4],
        }
    }

    /// Lift a 3×3 colour matrix into 4×4 layout (alpha row = identity).
    pub fn from_color_matrix(m: [[f32; 3]; 3]) -> Self {
        Self {
            matrix: [
                [m[0][0], m[0][1], m[0][2], 0.0],
                [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            offset: [0.0; 4],
        }
    }
}

impl ImageFilter for ChannelMixer {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (bpp, has_alpha) = match params.format {
            PixelFormat::Rgb24 => (3usize, false),
            PixelFormat::Rgba => (4usize, true),
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ChannelMixer requires Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: ChannelMixer requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let row_bytes = w * bpp;
        let src = &input.planes[0];
        let mut out = vec![0u8; row_bytes * h];
        let m = &self.matrix;
        let o = &self.offset;
        for y in 0..h {
            let src_row = &src.data[y * src.stride..y * src.stride + row_bytes];
            let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..w {
                let base = x * bpp;
                let r = src_row[base] as f32;
                let g = src_row[base + 1] as f32;
                let b = src_row[base + 2] as f32;
                let a = if has_alpha {
                    src_row[base + 3] as f32
                } else {
                    255.0
                };
                let nr = m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3] * a + o[0];
                let ng = m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3] * a + o[1];
                let nb = m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3] * a + o[2];
                dst_row[base] = nr.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 1] = ng.round().clamp(0.0, 255.0) as u8;
                dst_row[base + 2] = nb.round().clamp(0.0, 255.0) as u8;
                if has_alpha {
                    let na = m[3][0] * r + m[3][1] * g + m[3][2] * b + m[3][3] * a + o[3];
                    dst_row[base + 3] = na.round().clamp(0.0, 255.0) as u8;
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
    fn identity_is_passthrough() {
        let input = rgb(4, 4, |x, y| [(x * 30) as u8, (y * 30) as u8, 100]);
        let out = ChannelMixer::identity().apply(&input, p_rgb(4, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn swap_r_and_b_via_mixer() {
        let m = ChannelMixer::new(
            [
                [0.0, 0.0, 1.0, 0.0], // R' = B
                [0.0, 1.0, 0.0, 0.0], // G' = G
                [1.0, 0.0, 0.0, 0.0], // B' = R
                [0.0, 0.0, 0.0, 1.0],
            ],
            [0.0; 4],
        );
        let input = rgb(2, 2, |_, _| [200, 100, 50]);
        let out = m.apply(&input, p_rgb(2, 2)).unwrap();
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[50, 100, 200]);
        }
    }

    #[test]
    fn sepia_classic_warms_neutral_grey() {
        // The Microsoft sepia recipe maps a neutral grey to a warm
        // (R > G > B) triple. Sanity-check the inequalities.
        let input = rgb(2, 2, |_, _| [128, 128, 128]);
        let out = ChannelMixer::sepia_classic()
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let chunk = &out.planes[0].data[..3];
        assert!(
            chunk[0] > chunk[1],
            "R={} should exceed G={}",
            chunk[0],
            chunk[1]
        );
        assert!(
            chunk[1] > chunk[2],
            "G={} should exceed B={}",
            chunk[1],
            chunk[2]
        );
    }

    #[test]
    fn from_color_matrix_matches_3x3_behaviour() {
        // Lifting a 3×3 colour matrix into the 4×4 layout should
        // produce identical RGB outputs to running the 3×3 directly.
        let m3 = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]];
        let mixer = ChannelMixer::from_color_matrix(m3);
        let input = rgb(2, 2, |_, _| [100, 150, 200]);
        let out = mixer.apply(&input, p_rgb(2, 2)).unwrap();
        let r = (m3[0][0] * 100.0 + m3[0][1] * 150.0 + m3[0][2] * 200.0).round() as u8;
        let g = (m3[1][0] * 100.0 + m3[1][1] * 150.0 + m3[1][2] * 200.0).round() as u8;
        let b = (m3[2][0] * 100.0 + m3[2][1] * 150.0 + m3[2][2] * 200.0).round() as u8;
        let chunk = &out.planes[0].data[..3];
        assert_eq!(chunk, &[r, g, b]);
    }

    #[test]
    fn shape_preserved() {
        let input = rgb(5, 3, |_, _| [10, 20, 30]);
        let out = ChannelMixer::identity().apply(&input, p_rgb(5, 3)).unwrap();
        assert_eq!(out.planes[0].stride, 15);
        assert_eq!(out.planes[0].data.len(), 45);
    }

    #[test]
    fn alpha_row_only_active_on_rgba() {
        // Build an RGBA input with alpha 100; matrix sets A' = 2 * R.
        let m = ChannelMixer::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            [0.0; 4],
        );
        let data: Vec<u8> = (0..4).flat_map(|_| [50u8, 60, 70, 100]).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = m
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
            assert_eq!(chunk[3], 100, "A' = 2*50 = 100");
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
        let err = ChannelMixer::identity()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("ChannelMixer"));
    }
}
