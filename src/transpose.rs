//! `-transpose` (mirror along the top-left → bottom-right diagonal) and
//! `-transverse` (along the top-right → bottom-left diagonal): a `w × h`
//! picture becomes `h × w`. Bit-exact against `magick`
//! (`tests/magick_ops.rs`). Part of issue #12.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

use crate::{PixelFilter, VideoStreamParams, layout};

/// Which diagonal to mirror along.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransposeKind {
    /// Top-left → bottom-right (`-transpose`): `out(x, y) = in(y, x)`.
    #[default]
    Transpose,
    /// Top-right → bottom-left (`-transverse`):
    /// `out(x, y) = in(h − 1 − y, w − 1 − x)`.
    Transverse,
}

/// Mirror a picture along a diagonal (the output is `height × width`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct Transpose {
    /// The diagonal.
    pub kind: TransposeKind,
}

impl Transpose {
    /// `-transpose`.
    pub fn new() -> Self {
        Self {
            kind: TransposeKind::Transpose,
        }
    }

    /// `-transverse`.
    pub fn transverse() -> Self {
        Self {
            kind: TransposeKind::Transverse,
        }
    }
}

/// Packed layouts with one plane whose positions can move freely.
const PACKED: &[PixelFormat] = &[
    PixelFormat::Gray8,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Bgr24,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Ya8,
    PixelFormat::Pal8,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Ya16Le,
    PixelFormat::Cmyk,
    PixelFormat::CmykInverted,
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
];

impl PixelFilter for Transpose {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        PACKED
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let g = layout::plane_geometry(params.format, 0, params.width, params.height).ok_or_else(
            || {
                Error::unsupported(format!(
                    "oxideav-image-filter: Transpose does not take {:?}",
                    params.format
                ))
            },
        )?;
        let (w, h, bpp) = (g.width as usize, g.height as usize, g.position_bytes);
        let src = &input.planes[0];
        let stride = h * bpp;
        let mut data = vec![0u8; stride * w];
        for y in 0..h {
            for x in 0..w {
                let (ox, oy) = match self.kind {
                    TransposeKind::Transpose => (y, x),
                    TransposeKind::Transverse => (h - 1 - y, w - 1 - x),
                };
                let s = y * src.stride + x * bpp;
                let d = oy * stride + ox * bpp;
                data[d..d + bpp].copy_from_slice(&src.data[s..s + bpp]);
            }
        }
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane { stride, data }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;

    #[test]
    fn transpose_and_transverse_of_a_3x2() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 3,
            height: 2,
        };
        let t = Transpose::new().apply(&input, p).unwrap();
        assert_eq!(t.planes[0].stride, 2);
        assert_eq!(t.planes[0].data, vec![1, 4, 2, 5, 3, 6]);
        let v = Transpose::transverse().apply(&input, p).unwrap();
        assert_eq!(v.planes[0].data, vec![6, 3, 5, 2, 4, 1]);
    }
}
