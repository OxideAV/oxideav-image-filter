//! Shared helper for tonal-adjustment filters: apply an 8-bit lookup
//! table to the "tone" channels of a frame. For `Rgba` the alpha channel
//! is preserved. For YUV planar formats only the Y (luma) plane is
//! transformed; chroma is left untouched so hue/saturation stay put —
//! this matches the behaviour expected for gamma / brightness /
//! contrast / levels.
//!
//! Clean-room logic derived from the textbook definition of a LUT over
//! a raster image. Nothing here is copied from any other image library.

use oxideav_core::{PixelFormat, VideoFrame};

/// Apply `lut` to the tone channels of `frame` in place.
///
/// * `Gray8`: every sample is looked up.
/// * `Rgb24`: every R/G/B sample is looked up.
/// * `Rgba`: R/G/B looked up, alpha preserved.
/// * `Yuv420P` / `Yuv422P` / `Yuv444P`: Y plane looked up, chroma
///   planes untouched.
pub(crate) fn apply_tone_lut(frame: &mut VideoFrame, lut: &[u8; 256]) {
    let w = frame.width as usize;
    let h = frame.height as usize;
    match frame.format {
        PixelFormat::Gray8 => {
            let p = &mut frame.planes[0];
            let stride = p.stride;
            for y in 0..h {
                for v in &mut p.data[y * stride..y * stride + w] {
                    *v = lut[*v as usize];
                }
            }
        }
        PixelFormat::Rgb24 => {
            let p = &mut frame.planes[0];
            let stride = p.stride;
            for y in 0..h {
                let row = &mut p.data[y * stride..y * stride + 3 * w];
                for x in 0..w {
                    row[3 * x] = lut[row[3 * x] as usize];
                    row[3 * x + 1] = lut[row[3 * x + 1] as usize];
                    row[3 * x + 2] = lut[row[3 * x + 2] as usize];
                }
            }
        }
        PixelFormat::Rgba => {
            let p = &mut frame.planes[0];
            let stride = p.stride;
            for y in 0..h {
                let row = &mut p.data[y * stride..y * stride + 4 * w];
                for x in 0..w {
                    row[4 * x] = lut[row[4 * x] as usize];
                    row[4 * x + 1] = lut[row[4 * x + 1] as usize];
                    row[4 * x + 2] = lut[row[4 * x + 2] as usize];
                    // alpha (index 3) preserved
                }
            }
        }
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            let p = &mut frame.planes[0];
            let stride = p.stride;
            for y in 0..h {
                for v in &mut p.data[y * stride..y * stride + w] {
                    *v = lut[*v as usize];
                }
            }
            // chroma untouched
        }
        _ => {} // caller is expected to have rejected this via is_supported
    }
}
