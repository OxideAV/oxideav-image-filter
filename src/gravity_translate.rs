//! Gravity-anchored canvas placement: re-window the input on a new
//! canvas of `(width, height)` with the input pinned to a 9-point
//! cardinal compass anchor (`-gravity` in ImageMagick).
//!
//! Clean-room: this is `Extent` with the `(offset_x, offset_y)`
//! computed from a symbolic anchor instead of given numerically.
//!
//! For an input of size `(iw, ih)` placed on a `(ow, oh)` canvas,
//! anchor `North` ⇒ `x = (ow - iw) / 2, y = 0`; `South` ⇒
//! `x = (ow - iw) / 2, y = oh - ih`; `Centre` ⇒
//! `x = (ow - iw) / 2, y = (oh - ih) / 2`; etc. Padding outside the
//! source rectangle uses [`background`](Self::background).
//!
//! Operates on `Gray8`, `Rgb24`, `Rgba`, and planar YUV — backed by
//! the existing [`Extent`](crate::extent::Extent) pipeline so YUV
//! chroma subsampling alignment matches.

use crate::{extent::Extent, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// 9-point compass anchor mirroring ImageMagick's `-gravity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gravity {
    NorthWest,
    North,
    NorthEast,
    West,
    Centre,
    East,
    SouthWest,
    South,
    SouthEast,
}

impl Gravity {
    /// Parse the IM-style symbolic name (case-insensitive). `Center`
    /// is accepted as an alias for `Centre`.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "northwest" | "nw" => Gravity::NorthWest,
            "north" | "n" => Gravity::North,
            "northeast" | "ne" => Gravity::NorthEast,
            "west" | "w" => Gravity::West,
            "centre" | "center" | "c" => Gravity::Centre,
            "east" | "e" => Gravity::East,
            "southwest" | "sw" => Gravity::SouthWest,
            "south" | "s" => Gravity::South,
            "southeast" | "se" => Gravity::SouthEast,
            _ => return None,
        })
    }

    fn x_factor(self) -> f32 {
        match self {
            Gravity::NorthWest | Gravity::West | Gravity::SouthWest => 0.0,
            Gravity::North | Gravity::Centre | Gravity::South => 0.5,
            Gravity::NorthEast | Gravity::East | Gravity::SouthEast => 1.0,
        }
    }

    fn y_factor(self) -> f32 {
        match self {
            Gravity::NorthWest | Gravity::North | Gravity::NorthEast => 0.0,
            Gravity::West | Gravity::Centre | Gravity::East => 0.5,
            Gravity::SouthWest | Gravity::South | Gravity::SouthEast => 1.0,
        }
    }
}

/// Gravity-anchored canvas placement.
#[derive(Clone, Debug)]
pub struct GravityTranslate {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Compass anchor.
    pub gravity: Gravity,
    /// Background colour for the padded region (`[R, G, B, A]`). YUV
    /// chroma planes are painted neutral 128 — see [`Extent`].
    pub background: [u8; 4],
}

impl GravityTranslate {
    pub fn new(width: u32, height: u32, gravity: Gravity) -> Self {
        Self {
            width,
            height,
            gravity,
            background: [0, 0, 0, 255],
        }
    }

    pub fn with_background(mut self, bg: [u8; 4]) -> Self {
        self.background = bg;
        self
    }
}

impl ImageFilter for GravityTranslate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid(
                "oxideav-image-filter: GravityTranslate width/height must be > 0",
            ));
        }
        let iw = params.width as i32;
        let ih = params.height as i32;
        let ow = self.width as i32;
        let oh = self.height as i32;
        let off_x = ((ow - iw) as f32 * self.gravity.x_factor()).round() as i32;
        let off_y = ((oh - ih) as f32 * self.gravity.y_factor()).round() as i32;
        let extent = Extent::new(self.width, self.height)
            .with_offset(off_x, off_y)
            .with_background(self.background);
        extent.apply(input, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn rgba(w: u32, h: u32, fill: [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&fill);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    fn p_rgba(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn northwest_anchor_keeps_input_at_origin() {
        // Input 2×2 red on a 4×4 black canvas, NW gravity ⇒ red lands
        // at (0, 0).
        let input = rgba(2, 2, [255, 0, 0, 255]);
        let f = GravityTranslate::new(4, 4, Gravity::NorthWest);
        let out = f.apply(&input, p_rgba(2, 2)).unwrap();
        // Top-left pixel of output is red.
        assert_eq!(&out.planes[0].data[0..4], &[255, 0, 0, 255]);
        // Bottom-right pixel of output is background black.
        let last = out.planes[0].data.len() - 4;
        assert_eq!(&out.planes[0].data[last..], &[0, 0, 0, 255]);
    }

    #[test]
    fn southeast_anchor_pins_input_to_corner() {
        let input = rgba(2, 2, [0, 255, 0, 255]);
        let f = GravityTranslate::new(4, 4, Gravity::SouthEast);
        let out = f.apply(&input, p_rgba(2, 2)).unwrap();
        // Bottom-right 2×2 should be green; top-left should be background.
        assert_eq!(&out.planes[0].data[0..4], &[0, 0, 0, 255]);
        let stride = 16usize;
        let bottom_right_pixel_start = 3 * stride + 3 * 4;
        assert_eq!(
            &out.planes[0].data[bottom_right_pixel_start..bottom_right_pixel_start + 4],
            &[0, 255, 0, 255]
        );
    }

    #[test]
    fn centre_anchor_places_in_middle() {
        let input = rgba(2, 2, [10, 20, 30, 255]);
        let f = GravityTranslate::new(6, 6, Gravity::Centre);
        let out = f.apply(&input, p_rgba(2, 2)).unwrap();
        // Input lands at offset ((6-2)/2, (6-2)/2) = (2, 2). Pixel at
        // canvas (2, 2) should be input (0, 0).
        let stride = 24usize;
        let p = 2 * stride + 2 * 4;
        assert_eq!(&out.planes[0].data[p..p + 4], &[10, 20, 30, 255]);
        // Canvas (0, 0) is background.
        assert_eq!(&out.planes[0].data[0..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn rejects_zero_canvas() {
        let input = rgba(2, 2, [0; 4]);
        let f = GravityTranslate::new(0, 4, Gravity::Centre);
        assert!(f.apply(&input, p_rgba(2, 2)).is_err());
    }

    #[test]
    fn from_name_aliases() {
        assert_eq!(Gravity::from_name("c"), Some(Gravity::Centre));
        assert_eq!(Gravity::from_name("Center"), Some(Gravity::Centre));
        assert_eq!(Gravity::from_name("centre"), Some(Gravity::Centre));
        assert_eq!(Gravity::from_name("nonsense"), None);
    }
}
