//! Pull a single channel out of a multi-channel frame as `Gray8`
//! (`-channel R -separate`).
//!
//! This is the simpler half of ImageMagick's `-separate` — instead of
//! emitting one frame per channel, the caller picks **one** channel
//! (R / G / B / A on packed RGB / RGBA, or Y / U / V on planar YUV) and
//! the filter returns a single-plane `Gray8` frame holding just that
//! channel's samples.
//!
//! Clean-room: index lookup + memcpy. Each output sample is just the
//! corresponding source sample at the requested channel offset.
//!
//! Output dimensions match the source plane the channel lives on:
//!
//! - For packed formats every channel sits on the same `width × height`
//!   grid as the input.
//! - For planar YUV the chroma channels live on the subsampled grid
//!   (`Yuv420P` → half on each axis, `Yuv422P` → half on the X axis,
//!   `Yuv444P` → full size). The `ImageFilterAdapter` rebuilds the
//!   downstream port spec from the produced plane's stride / row count
//!   so this is internally consistent.

use crate::blur::{chroma_subsampling, plane_dims};
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which channel to pull out as a `Gray8` frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// Red channel of `Rgb24` / `Rgba`.
    Red,
    /// Green channel of `Rgb24` / `Rgba`.
    Green,
    /// Blue channel of `Rgb24` / `Rgba`.
    Blue,
    /// Alpha channel of `Rgba`.
    Alpha,
    /// Luma plane of planar YUV. Equivalent to `Red` on `Gray8`.
    Y,
    /// First chroma plane of planar YUV (`U` / `Cb`).
    U,
    /// Second chroma plane of planar YUV (`V` / `Cr`).
    V,
}

impl Channel {
    /// Parse the JSON-friendly short name. Accepts the canonical
    /// `"r"` / `"g"` / `"b"` / `"a"` / `"y"` / `"u"` / `"v"` plus the
    /// long forms `"red"` / ... / `"alpha"` / `"luma"` / `"cb"` / `"cr"`
    /// (case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "r" | "red" => Some(Self::Red),
            "g" | "green" => Some(Self::Green),
            "b" | "blue" => Some(Self::Blue),
            "a" | "alpha" => Some(Self::Alpha),
            "y" | "luma" => Some(Self::Y),
            "u" | "cb" => Some(Self::U),
            "v" | "cr" => Some(Self::V),
            _ => None,
        }
    }
}

/// Extract one channel of the input as a single-plane `Gray8` frame.
#[derive(Clone, Copy, Debug)]
pub struct ChannelExtract {
    pub channel: Channel,
}

impl ChannelExtract {
    /// Build with the requested channel.
    pub fn new(channel: Channel) -> Self {
        Self { channel }
    }
}

impl ImageFilter for ChannelExtract {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: ChannelExtract does not yet handle {:?}",
                params.format
            )));
        }
        let (cx, cy) = chroma_subsampling(params.format);

        // Resolve `(plane index, byte offset within pixel)` for the
        // requested channel. Returns `Unsupported` when the channel
        // doesn't exist on the input format (e.g. Alpha on Rgb24, U on
        // Gray8, Red on Yuv).
        let (plane_idx, ch_offset) = match (params.format, self.channel) {
            (PixelFormat::Gray8, Channel::Red | Channel::Y) => (0usize, 0usize),
            (PixelFormat::Rgb24, Channel::Red) => (0, 0),
            (PixelFormat::Rgb24, Channel::Green) => (0, 1),
            (PixelFormat::Rgb24, Channel::Blue) => (0, 2),
            (PixelFormat::Rgba, Channel::Red) => (0, 0),
            (PixelFormat::Rgba, Channel::Green) => (0, 1),
            (PixelFormat::Rgba, Channel::Blue) => (0, 2),
            (PixelFormat::Rgba, Channel::Alpha) => (0, 3),
            (PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P, Channel::Y) => {
                (0, 0)
            }
            (PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P, Channel::U) => {
                (1, 0)
            }
            (PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P, Channel::V) => {
                (2, 0)
            }
            (fmt, ch) => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: ChannelExtract: format {fmt:?} has no {ch:?} channel"
                )));
            }
        };

        let bpp_packed = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba => 4,
            _ => 1,
        };

        let (pw, ph) = plane_dims(
            params.width,
            params.height,
            params.format,
            plane_idx,
            cx,
            cy,
        );
        let pw = pw as usize;
        let ph = ph as usize;

        let src = &input.planes[plane_idx];
        let mut out = vec![0u8; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                out[y * pw + x] = src.data[y * src.stride + x * bpp_packed + ch_offset];
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: pw,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_short_and_long_names() {
        assert_eq!(Channel::parse("r"), Some(Channel::Red));
        assert_eq!(Channel::parse("Red"), Some(Channel::Red));
        assert_eq!(Channel::parse("Alpha"), Some(Channel::Alpha));
        assert_eq!(Channel::parse("Cb"), Some(Channel::U));
        assert_eq!(Channel::parse("garbage"), None);
    }

    #[test]
    fn extracts_red_from_rgb24() {
        // 2×2 Rgb24: each pixel R = pixel index, G = 0, B = 0.
        let mut data = Vec::with_capacity(2 * 2 * 3);
        for i in 0..4u8 {
            data.extend_from_slice(&[i * 30, 0, 0]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 6, data }],
        };
        let out = ChannelExtract::new(Channel::Red)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(out.planes[0].data, vec![0, 30, 60, 90]);
    }

    #[test]
    fn extracts_alpha_from_rgba() {
        let mut data = Vec::with_capacity(4 * 4);
        for i in 0..4u8 {
            data.extend_from_slice(&[10, 20, 30, 40 + i * 10]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = ChannelExtract::new(Channel::Alpha)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 4,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![40, 50, 60, 70]);
    }

    #[test]
    fn extracts_chroma_from_yuv420() {
        // 4×4 4:2:0 — chroma planes are 2×2.
        let y: Vec<u8> = (0..16u8).collect();
        let u: Vec<u8> = vec![100, 110, 120, 130];
        let v: Vec<u8> = vec![200, 210, 220, 230];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane { stride: 2, data: u },
                VideoPlane { stride: 2, data: v },
            ],
        };
        let out = ChannelExtract::new(Channel::U)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Output is the 2×2 chroma plane.
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(out.planes[0].data, vec![100, 110, 120, 130]);
    }

    #[test]
    fn rejects_alpha_on_rgb24() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![0, 0, 0],
            }],
        };
        let err = ChannelExtract::new(Channel::Alpha)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Alpha"));
    }

    #[test]
    fn rejects_red_on_yuv() {
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 2,
                    data: vec![0, 0, 0, 0],
                },
                VideoPlane {
                    stride: 1,
                    data: vec![128],
                },
                VideoPlane {
                    stride: 1,
                    data: vec![128],
                },
            ],
        };
        let err = ChannelExtract::new(Channel::Red)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("Red"));
    }
}
