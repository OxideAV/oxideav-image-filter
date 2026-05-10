//! Circular pixel shift (`-roll +X+Y`).
//!
//! Translates every pixel by `(dx, dy)` and wraps around the image
//! borders so no information is lost — the column that scrolls past the
//! right edge re-enters from the left, the row that scrolls off the
//! bottom re-enters from the top. Negative offsets shift left / up;
//! positive offsets shift right / down. Equivalent to the ImageMagick
//! `-roll WxH` / `-roll +X+Y` operation.
//!
//! Clean-room: the formula is `dst[(x + dx) mod W, (y + dy) mod H] =
//! src[x, y]`, or equivalently `src[(x' - dx) mod W, (y' - dy) mod H]`
//! when written destination-first. The whole frame is just a big modular
//! re-index — no resampling, no math beyond integer mod.
//!
//! Operates on every supported pixel format (`Gray8`, `Rgb24`, `Rgba`,
//! `Yuv420P`, `Yuv422P`, `Yuv444P`). For planar YUV the chroma offsets
//! are scaled by the subsampling factor so the U/V planes wrap on their
//! own (smaller) grid in lock-step with the Y plane — the visible image
//! shifts as a single rigid block.

use crate::blur::{bytes_per_plane_pixel, chroma_subsampling, plane_dims};
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Wrap-around translation of every pixel by `(dx, dy)`.
#[derive(Clone, Copy, Debug)]
pub struct Roll {
    /// Horizontal shift in pixels. Positive shifts right, negative
    /// shifts left. Wraps modulo image width.
    pub dx: i32,
    /// Vertical shift in pixels. Positive shifts down, negative shifts
    /// up. Wraps modulo image height.
    pub dy: i32,
}

impl Roll {
    /// Build a roll with the given pixel offsets.
    pub fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }
}

impl ImageFilter for Roll {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Roll does not yet handle {:?}",
                params.format
            )));
        }
        if params.width == 0 || params.height == 0 {
            return Ok(input.clone());
        }

        let (cx, cy) = chroma_subsampling(params.format);
        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (pw, ph) = plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let bpp = bytes_per_plane_pixel(params.format, idx);
            // Scale offsets to this plane's grid for chroma planes; the
            // Y / packed plane stays at full resolution.
            let (pdx, pdy) = if idx > 0 {
                (
                    scale_offset(self.dx, cx as i32),
                    scale_offset(self.dy, cy as i32),
                )
            } else {
                (self.dx, self.dy)
            };
            new_planes.push(roll_plane(plane, pw, ph, bpp, pdx, pdy));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

/// Scale a luma-grid offset onto a subsampled chroma grid. Rounds
/// toward zero so a 1-pixel luma shift stays a fractional chroma shift
/// that we conservatively round to no chroma motion (better to leave
/// chroma in place than to introduce a half-step misalignment).
fn scale_offset(d: i32, factor: i32) -> i32 {
    if factor <= 1 {
        return d;
    }
    d / factor
}

fn roll_plane(plane: &VideoPlane, w: u32, h: u32, bpp: usize, dx: i32, dy: i32) -> VideoPlane {
    let w = w as usize;
    let h = h as usize;
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    if w == 0 || h == 0 {
        return VideoPlane {
            stride: row_bytes,
            data: out,
        };
    }
    let dx = dx.rem_euclid(w as i32) as usize;
    let dy = dy.rem_euclid(h as i32) as usize;

    for y in 0..h {
        let src_y = (y + h - dy) % h;
        let src_row = &plane.data[src_y * plane.stride..src_y * plane.stride + row_bytes];
        let dst_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
        if dx == 0 {
            dst_row.copy_from_slice(src_row);
        } else {
            // Two contiguous segments: [0, w-dx) ↔ [dx, w) and [w-dx, w) ↔ [0, dx).
            let split_bytes = (w - dx) * bpp;
            dst_row[..dx * bpp].copy_from_slice(&src_row[split_bytes..]);
            dst_row[dx * bpp..].copy_from_slice(&src_row[..split_bytes]);
        }
    }

    VideoPlane {
        stride: row_bytes,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

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
    fn zero_offset_is_identity() {
        let input = gray(4, 3, |x, y| (x * 3 + y * 7) as u8);
        let out = Roll::new(0, 0).apply(&input, params_gray(4, 3)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn full_period_wraps_back_to_identity() {
        // A roll by (W, H) is the identity since each axis wraps modulo
        // its size.
        let input = gray(5, 3, |x, y| (x + y * 11) as u8);
        let out = Roll::new(5, 3).apply(&input, params_gray(5, 3)).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn positive_dx_shifts_columns_right() {
        // Row pattern: [0, 1, 2, 3]; +1 roll → [3, 0, 1, 2].
        let input = gray(4, 1, |x, _| x as u8);
        let out = Roll::new(1, 0).apply(&input, params_gray(4, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![3, 0, 1, 2]);
    }

    #[test]
    fn negative_dx_shifts_columns_left() {
        // Row pattern: [0, 1, 2, 3]; -1 roll → [1, 2, 3, 0].
        let input = gray(4, 1, |x, _| x as u8);
        let out = Roll::new(-1, 0).apply(&input, params_gray(4, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![1, 2, 3, 0]);
    }

    #[test]
    fn positive_dy_shifts_rows_down() {
        // Column pattern: [0, 1, 2]; +1 roll → [2, 0, 1].
        let input = gray(1, 3, |_, y| y as u8);
        let out = Roll::new(0, 1).apply(&input, params_gray(1, 3)).unwrap();
        assert_eq!(out.planes[0].data, vec![2, 0, 1]);
    }

    #[test]
    fn rgba_packed_roll_keeps_pixels_atomic() {
        // 2×1 RGBA: [R0,G0,B0,A0,  R1,G1,B1,A1]; +1 roll → [R1..A1, R0..A0].
        let data = vec![10, 20, 30, 255, 40, 50, 60, 128];
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = Roll::new(1, 0)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 1,
                },
            )
            .unwrap();
        assert_eq!(out.planes[0].data, vec![40, 50, 60, 128, 10, 20, 30, 255]);
    }

    #[test]
    fn yuv420p_chroma_offsets_track_luma() {
        // 4×4 4:2:0 frame; chroma planes are 2×2.
        // Luma: pattern 0..16. U: 0..4 +100. V: 0..4 +200.
        let y: Vec<u8> = (0..4 * 4).map(|i| i as u8).collect();
        let u: Vec<u8> = (0..2 * 2).map(|i| 100 + i as u8).collect();
        let v: Vec<u8> = (0..2 * 2).map(|i| 200 + i as u8).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane { stride: 2, data: u },
                VideoPlane { stride: 2, data: v },
            ],
        };
        // Roll the luma by (2, 2). Chroma offsets become (1, 1) on the
        // half-resolution grid.
        let out = Roll::new(2, 2)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Luma at (0, 0) was originally at (-2, -2) mod (4, 4) = (2, 2),
        // index 2*4 + 2 = 10.
        assert_eq!(out.planes[0].data[0], 10);
        // Chroma U at (0, 0) was originally at (1, 1), index 1*2 + 1 = 3.
        assert_eq!(out.planes[1].data[0], 100 + 3);
        assert_eq!(out.planes[2].data[0], 200 + 3);
    }
}
