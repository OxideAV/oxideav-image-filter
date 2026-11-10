//! Pull a single channel out of a multi-channel frame as `Gray8`
//! (`-channel R -separate`).
//!
//! This is the simpler half of the documented `-separate` CLI — instead of
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
use crate::{PixelFilter, VideoStreamParams, is_supported_format};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which channel to pull out as a `Gray8` frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct ChannelExtract {
    pub channel: Channel,
}

impl ChannelExtract {
    /// Builder-style setter for [`channel`](Self::channel).
    pub fn with_channel(mut self, channel: Channel) -> Self {
        self.channel = channel;
        self
    }
}

impl ChannelExtract {
    /// Build with the requested channel.
    pub fn new(channel: Channel) -> Self {
        Self { channel }
    }
}

impl PixelFilter for ChannelExtract {
    /// The layouts that carry the requested channel: `Red` reads grey
    /// as well as packed RGB(A), `Green` / `Blue` need packed RGB(A),
    /// `Alpha` needs `Rgba`, `Y` reads grey or planar YUV, `U` / `V`
    /// need planar YUV.
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        use oxideav_core::PixelFormat as P;
        const UV: &[P] = &[
            P::Yuv420P,
            P::Yuv422P,
            P::Yuv444P,
            P::YuvJ420P,
            P::YuvJ422P,
            P::YuvJ444P,
        ];
        const Y_DEEP: &[P] = &[
            P::Gray8,
            P::Yuv420P,
            P::Yuv422P,
            P::Yuv444P,
            P::YuvJ420P,
            P::YuvJ422P,
            P::YuvJ444P,
            P::GrayF32Le,
        ];
        match self.channel {
            Channel::Red => crate::layout::PACKED_8BIT_DEEP,
            Channel::Green | Channel::Blue => crate::layout::PACKED_RGB_DEEP,
            Channel::Alpha => crate::layout::RGBA_DEEP,
            Channel::Y => Y_DEEP,
            Channel::U | Channel::V => UV,
        }
    }

    /// Collapses every input to a single-plane grey picture at the
    /// input's precision: `Gray8` for the 8-bit layouts, `GrayF32Le`
    /// for the float ones (and so for 16-bit pictures through the float
    /// hop — a 16-bit channel is never narrowed).
    fn output_format(&self, input: oxideav_core::PixelFormat) -> oxideav_core::PixelFormat {
        if crate::deep::is_float(input) {
            oxideav_core::PixelFormat::GrayF32Le
        } else {
            oxideav_core::PixelFormat::Gray8
        }
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        if crate::deep::is_float(params.format) {
            let img = crate::deep::FloatImage::from_frame(input, params)?;
            let c = match (img.channels(), self.channel) {
                (1, Channel::Red | Channel::Y) => 0,
                (3 | 4, Channel::Red) => 0,
                (3 | 4, Channel::Green) => 1,
                (3 | 4, Channel::Blue) => 2,
                (4, Channel::Alpha) => 3,
                (_, ch) => {
                    return Err(Error::unsupported(format!(
                        "oxideav-image-filter: ChannelExtract: format {:?} has no {ch:?} channel",
                        params.format
                    )));
                }
            };
            let data: Vec<f32> = crate::deep::channel_iter(&img, c).collect();
            let out = crate::deep::FloatImage::from_samples(params.width, params.height, 1, data)?;
            return Ok(out.into_frame(input.pts));
        }
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
            (
                PixelFormat::Yuv420P
                | PixelFormat::Yuv422P
                | PixelFormat::Yuv444P
                | PixelFormat::YuvJ420P
                | PixelFormat::YuvJ422P
                | PixelFormat::YuvJ444P,
                Channel::Y,
            ) => (0, 0),
            (
                PixelFormat::Yuv420P
                | PixelFormat::Yuv422P
                | PixelFormat::Yuv444P
                | PixelFormat::YuvJ420P
                | PixelFormat::YuvJ422P
                | PixelFormat::YuvJ444P,
                Channel::U,
            ) => (1, 0),
            (
                PixelFormat::Yuv420P
                | PixelFormat::Yuv422P
                | PixelFormat::Yuv444P
                | PixelFormat::YuvJ420P
                | PixelFormat::YuvJ422P
                | PixelFormat::YuvJ444P,
                Channel::V,
            ) => (2, 0),
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
    use crate::ImageFilter;

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
    fn sixteen_bit_channels_come_out_at_their_own_precision() {
        // One `Rgb48Le` pixel whose codes no 8-bit value holds: each
        // channel is returned as binary32 `code / 65535`, not narrowed.
        let codes = [12_345u16, 40_001, 65_535];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 6,
                data: codes.iter().flat_map(|c| c.to_le_bytes()).collect(),
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgb48Le,
            width: 1,
            height: 1,
        };
        for (channel, code) in [
            (Channel::Red, codes[0]),
            (Channel::Green, codes[1]),
            (Channel::Blue, codes[2]),
        ] {
            let f = ChannelExtract::new(channel);
            assert_eq!(
                crate::ImageFilter::output_pixel_format(&f, PixelFormat::Rgb48Le),
                Some(PixelFormat::GrayF32Le)
            );
            let out = f.apply(&input, p).unwrap();
            let d = &out.planes[0].data;
            let v = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
            assert_eq!(v, code as f32 / 65535.0, "{channel:?}");
        }
        // A picture without alpha is opaque.
        let out = ChannelExtract::new(Channel::Alpha)
            .apply(&input, p)
            .unwrap();
        let d = &out.planes[0].data;
        assert_eq!(f32::from_le_bytes([d[0], d[1], d[2], d[3]]), 1.0);
    }

    #[test]
    fn rejects_a_channel_no_route_reaches() {
        // `U` reads a chroma plane of 8-bit planar YUV only: a packed
        // float picture has none.
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 4],
            }],
        };
        let err = ChannelExtract::new(Channel::U)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::GrayF32Le,
                    width: 1,
                    height: 1,
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(format!("{err}").contains("GrayF32Le"));
    }
}
