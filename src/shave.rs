//! Strip a uniform border off every edge (`-shave XxY`).
//!
//! Drops `x_border` columns from the left **and** the right and
//! `y_border` rows from the top **and** the bottom. Output dimensions
//! become `(W - 2·x_border) × (H - 2·y_border)`. The operation fails
//! fast if either border swallows the whole image.
//!
//! Clean-room: this is a centred crop — the geometry alone determines
//! the result, no resampling. Backed by [`Crop`](crate::crop::Crop) so
//! YUV chroma alignment behaves identically.

use crate::{Crop, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Trim a `x_border × y_border` margin from every edge of the frame.
///
/// Equivalent to ImageMagick `-shave XxY`. Output size is
/// `(W - 2·x_border) × (H - 2·y_border)`. Returns
/// [`Error::invalid`](oxideav_core::Error) when the requested borders
/// would shrink the image to zero or negative width/height.
#[derive(Clone, Copy, Debug)]
pub struct Shave {
    pub x_border: u32,
    pub y_border: u32,
}

impl Shave {
    /// Build a shave with `x_border` columns trimmed from each side and
    /// `y_border` rows trimmed from top + bottom.
    pub fn new(x_border: u32, y_border: u32) -> Self {
        Self { x_border, y_border }
    }
}

impl ImageFilter for Shave {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let total_x = self
            .x_border
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("oxideav-image-filter: Shave x_border*2 overflows"))?;
        let total_y = self
            .y_border
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("oxideav-image-filter: Shave y_border*2 overflows"))?;
        if total_x >= params.width || total_y >= params.height {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: Shave {}x{} would zero out a {}x{} frame",
                self.x_border, self.y_border, params.width, params.height
            )));
        }
        let new_w = params.width - total_x;
        let new_h = params.height - total_y;
        Crop::new(self.x_border, self.y_border, new_w, new_h).apply(input, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, VideoPlane};

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    fn params_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn zero_border_is_identity() {
        let input = gray(5, 4, |x, y| (x + y * 7) as u8);
        let out = Shave::new(0, 0).apply(&input, params_gray(5, 4)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
        assert_eq!(out.planes[0].stride, 5);
    }

    #[test]
    fn shave_reduces_dimensions_symmetrically() {
        let input = gray(8, 6, |x, y| (x + y * 10) as u8);
        let out = Shave::new(2, 1).apply(&input, params_gray(8, 6)).unwrap();
        // Output is (8-4) × (6-2) = 4 × 4.
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data.len(), 16);
        // Top-left of output = input(2, 1) = 2 + 10 = 12.
        assert_eq!(out.planes[0].data[0], 12);
        // Bottom-right of output = input(5, 4) = 5 + 40 = 45.
        assert_eq!(out.planes[0].data[15], 45);
    }

    #[test]
    fn shave_rejects_oversize_border() {
        let input = gray(4, 4, |_, _| 0);
        let err = Shave::new(2, 0)
            .apply(&input, params_gray(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("zero"));
        // A border equal to half the dimension on either axis also collapses.
        let err = Shave::new(0, 3)
            .apply(&input, params_gray(4, 4))
            .unwrap_err();
        assert!(format!("{err}").contains("zero"));
    }
}
