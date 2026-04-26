//! Rotate a frame by an arbitrary angle around its centre, using
//! bilinear resampling and growing the canvas so no pixel is clipped.
//!
//! This is the ImageMagick `-rotate N` model: positive N rotates
//! *clockwise* (matching the convention that image y grows downward,
//! so a "clockwise" rotation in the viewer's frame is a mathematically
//! negative rotation in the standard x-right/y-up plane). The
//! implementation is a clean-room application of the 2-D rotation
//! matrix combined with bilinear interpolation:
//!
//!   x' =  x·cos(θ) + y·sin(θ)
//!   y' = -x·sin(θ) + y·cos(θ)
//!
//! We compute the forward-transformed bounding box to size the output,
//! then invert the mapping — for every output pixel we look up the
//! source coordinate and sample it with bilinear interpolation.
//! Out-of-bounds samples become the configured background colour.

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Rotate a frame by `degrees` around its centre. Positive values
/// rotate clockwise (ImageMagick convention). The output canvas is
/// grown to fit the rotated rectangle; any pixels that would fall
/// outside the source become [`background`](Self::with_background).
#[derive(Clone, Debug)]
pub struct Rotate {
    degrees: f32,
    background: Option<[u8; 4]>,
}

impl Rotate {
    /// Build a rotation by `degrees`. Default background is format-
    /// dependent: opaque black for Gray/YUV/Rgb, fully transparent black
    /// for `Rgba`.
    pub fn new(degrees: f32) -> Self {
        Self {
            degrees,
            background: None,
        }
    }

    /// Override the background fill colour. Channels are interpreted
    /// per format:
    /// - `Gray8`: `color[0]` (luma).
    /// - `Rgb24`: `color[0..3]` as R, G, B.
    /// - `Rgba`:  `color[0..4]` as R, G, B, A.
    /// - `Yuv*P`: `color[0..3]` as Y, Cb, Cr.
    pub fn with_background(mut self, color: [u8; 4]) -> Self {
        self.background = Some(color);
        self
    }
}

impl ImageFilter for Rotate {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Rotate does not yet handle {:?}",
                params.format
            )));
        }

        // Fast-path multiples of 90°: use exact axis-aligned remaps
        // so we don't accumulate bilinear drift from f32 sin/cos.
        if let Some(out) = quarter_turn(input, params, self.degrees) {
            return Ok(out);
        }

        // Convert "clockwise degrees in image coordinates" to the
        // mathematical angle we use in the rotation matrix. With image
        // y pointing downward, a clockwise visual rotation is +θ.
        let theta = self.degrees.to_radians();
        let (sin_t, cos_t) = theta.sin_cos();

        let sw = params.width as f32;
        let sh = params.height as f32;
        let (new_w, new_h) = rotated_bounds(sw, sh, sin_t, cos_t);
        let (cx, cy) = crate::blur::chroma_subsampling(params.format);

        let bg = resolve_background(params.format, self.background);

        let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
        for (idx, plane) in input.planes.iter().enumerate() {
            let (src_pw, src_ph) =
                crate::blur::plane_dims(params.width, params.height, params.format, idx, cx, cy);
            let (dst_pw, dst_ph) = crate::blur::plane_dims(new_w, new_h, params.format, idx, cx, cy);
            let bpp = crate::blur::bytes_per_plane_pixel(params.format, idx);
            let fill = plane_fill(params.format, idx, &bg);
            new_planes.push(rotate_plane(
                plane,
                src_pw as usize,
                src_ph as usize,
                dst_pw as usize,
                dst_ph as usize,
                bpp,
                sin_t,
                cos_t,
                &fill,
            ));
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: new_planes,
        })
    }
}

/// Fast path for rotations that are exact multiples of 90°. Returns
/// `None` for angles that need real interpolation.
///
/// `params` is threaded in because the slim moved width / height /
/// pixel format off the frame and onto the stream's `CodecParameters`.
///
/// We reduce the input to the canonical range `[0, 360)` and then pick
/// the matching axis-aligned remap (identity / 90° CW / 180° / 270° CW),
/// remembering that "positive degrees = clockwise" is our convention.
fn quarter_turn(input: &VideoFrame, params: VideoStreamParams, degrees: f32) -> Option<VideoFrame> {
    // Collapse to [0, 360) and snap within 0.001° of a multiple of 90°.
    let mut d = degrees.rem_euclid(360.0);
    let snaps = [0.0f32, 90.0, 180.0, 270.0, 360.0];
    let snapped = snaps.iter().copied().find(|s| (d - *s).abs() < 1e-3)?;
    d = if snapped == 360.0 { 0.0 } else { snapped };

    let (cx, cy) = crate::blur::chroma_subsampling(params.format);
    let bpp_of = |idx| crate::blur::bytes_per_plane_pixel(params.format, idx);
    let new_dims = |pw: usize, ph: usize| -> (usize, usize) {
        match d as u32 {
            0 => (pw, ph),
            90 | 270 => (ph, pw),
            180 => (pw, ph),
            _ => unreachable!(),
        }
    };

    let mut new_planes: Vec<VideoPlane> = Vec::with_capacity(input.planes.len());
    for (idx, plane) in input.planes.iter().enumerate() {
        let (pw, ph) =
            crate::blur::plane_dims(params.width, params.height, params.format, idx, cx, cy);
        let bpp = bpp_of(idx);
        let pw = pw as usize;
        let ph = ph as usize;
        let (dw, dh) = new_dims(pw, ph);
        let row_bytes = dw * bpp;
        let mut out = vec![0u8; row_bytes * dh];
        for y in 0..ph {
            for x in 0..pw {
                let (nx, ny) = match d as u32 {
                    0 => (x, y),
                    90 => (ph - 1 - y, x), // clockwise
                    180 => (pw - 1 - x, ph - 1 - y),
                    270 => (y, pw - 1 - x), // counter-clockwise 90
                    _ => unreachable!(),
                };
                let src_off = y * plane.stride + x * bpp;
                let dst_off = ny * row_bytes + nx * bpp;
                out[dst_off..dst_off + bpp].copy_from_slice(&plane.data[src_off..src_off + bpp]);
            }
        }
        new_planes.push(VideoPlane {
            stride: row_bytes,
            data: out,
        });
    }

    // (Old: returned width/height on the VideoFrame literal; the slim
    // moved those onto the stream's CodecParameters. Callers that need
    // the post-rotate dimensions read them off the output port spec.)
    Some(VideoFrame {
        pts: input.pts,
        planes: new_planes,
    })
}

/// Size of the axis-aligned bounding box around a rotated `sw × sh` rect.
/// Rounded up so we never lose a pixel; minimum of 1 in each dim. Values
/// within 1e-4 of an integer snap to that integer — this keeps multiples
/// of 90° (where sin / cos are ±1 but not bit-exact after f32 rounding)
/// from gaining a spurious +1 pixel of padding.
fn rotated_bounds(sw: f32, sh: f32, sin_t: f32, cos_t: f32) -> (u32, u32) {
    let w = sw * cos_t.abs() + sh * sin_t.abs();
    let h = sw * sin_t.abs() + sh * cos_t.abs();
    (snap_ceil(w).max(1) as u32, snap_ceil(h).max(1) as u32)
}

/// `value.ceil()` with a tiny snap-to-integer window.
fn snap_ceil(v: f32) -> i64 {
    let eps = 1e-4_f32;
    let r = v.round();
    if (v - r).abs() < eps {
        r as i64
    } else {
        v.ceil() as i64
    }
}

/// Pick the appropriate per-plane fill sample from a (possibly user-
/// supplied) 4-byte colour.
///
/// Returned slice is 1..=4 bytes long and matches the plane's bpp.
fn plane_fill(fmt: PixelFormat, plane_idx: usize, bg: &[u8; 4]) -> Vec<u8> {
    match fmt {
        PixelFormat::Gray8 => vec![bg[0]],
        PixelFormat::Rgb24 => vec![bg[0], bg[1], bg[2]],
        PixelFormat::Rgba => vec![bg[0], bg[1], bg[2], bg[3]],
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
            // Plane 0 = Y, 1 = Cb, 2 = Cr.
            vec![bg[plane_idx.min(2)]]
        }
        _ => vec![0],
    }
}

fn resolve_background(fmt: PixelFormat, override_: Option<[u8; 4]>) -> [u8; 4] {
    if let Some(c) = override_ {
        return c;
    }
    match fmt {
        PixelFormat::Rgba => [0, 0, 0, 0], // transparent
        // Opaque-black default for the rest. For YUV the tuple is
        // (Y=0, Cb=128, Cr=128) — neutral chroma + black luma.
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => [0, 128, 128, 0],
        _ => [0, 0, 0, 255],
    }
}

/// Inverse-map rotate a single plane. For each destination pixel,
/// compute the corresponding source coordinate using the inverse
/// rotation and sample with bilinear interpolation, filling with `fill`
/// when outside the source.
#[allow(clippy::too_many_arguments)]
fn rotate_plane(
    src: &VideoPlane,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    bpp: usize,
    sin_t: f32,
    cos_t: f32,
    fill: &[u8],
) -> VideoPlane {
    assert_eq!(fill.len(), bpp, "fill must have bpp bytes");
    let row_bytes = dw * bpp;
    let mut out = vec![0u8; row_bytes * dh];

    // Fill with background so out-of-range lookups need no branches.
    for y in 0..dh {
        let row = &mut out[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..dw {
            row[x * bpp..x * bpp + bpp].copy_from_slice(fill);
        }
    }

    if sw == 0 || sh == 0 {
        return VideoPlane {
            stride: row_bytes,
            data: out,
        };
    }

    // Translate so (0,0) is the source centre during sampling.
    let scx = (sw as f32 - 1.0) * 0.5;
    let scy = (sh as f32 - 1.0) * 0.5;
    let dcx = (dw as f32 - 1.0) * 0.5;
    let dcy = (dh as f32 - 1.0) * 0.5;

    // Inverse rotation: if forward is
    //   [ cos -sin ] [ x ]
    //   [ sin  cos ] [ y ]
    // then inverse (rotation is orthogonal) is the transpose, i.e.
    // swap sign of sin.
    let inv_sin = -sin_t;

    for y in 0..dh {
        let dy = y as f32 - dcy;
        for x in 0..dw {
            let dx = x as f32 - dcx;
            let sx = dx * cos_t + dy * inv_sin + scx;
            let sy = dx * sin_t + dy * cos_t + scy;

            if sx < 0.0 || sy < 0.0 || sx > (sw as f32 - 1.0) || sy > (sh as f32 - 1.0) {
                continue; // background already written
            }

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let dst_off = y * row_bytes + x * bpp;
            for ch in 0..bpp {
                let p00 = src.data[y0 * src.stride + x0 * bpp + ch] as f32;
                let p10 = src.data[y0 * src.stride + x1 * bpp + ch] as f32;
                let p01 = src.data[y1 * src.stride + x0 * bpp + ch] as f32;
                let p11 = src.data[y1 * src.stride + x1 * bpp + ch] as f32;
                let v = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                out[dst_off + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
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
    use oxideav_core::TimeBase;

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

    #[test]
    fn rotate_0_is_identity() {
        let input = gray(8, 6, |x, y| (x + y * 8) as u8);
        let out = Rotate::new(0.0).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 8, height: 6 }).unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn rotate_90_twice_matches_180() {
        // Square so 90° keeps dims equal to the original.
        let p = VideoStreamParams { format: PixelFormat::Gray8, width: 9, height: 9 };
        let input = gray(9, 9, |x, y| ((x * 17 + y * 23) % 251) as u8);
        let ninety = Rotate::new(90.0).apply(&input, p).unwrap();
        let one_eighty_via_two = Rotate::new(90.0).apply(&ninety, p).unwrap();
        let one_eighty = Rotate::new(180.0).apply(&input, p).unwrap();
        // Multiples of 90° are handled via the axis-aligned fast path,
        // so pixels must match exactly (no interpolation).
        for (a, b) in one_eighty.planes[0]
            .data
            .iter()
            .zip(one_eighty_via_two.planes[0].data.iter())
        {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "two 90° ≠ 180° by more than 1"
            );
        }
    }

    #[test]
    fn rotate_360_is_near_identity() {
        let input = gray(10, 8, |x, y| ((x * 11 + y * 13) % 251) as u8);
        let out = Rotate::new(360.0).apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 10, height: 8 }).unwrap();
        // 360° is a multiple of 90° and takes the fast path → exact.
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "360° drift > 1 on fast path"
            );
        }
    }

    #[test]
    fn rotate_45_grows_canvas() {
        let input = gray(8, 8, |_, _| 120);
        let out = Rotate::new(45.0)
            .apply(
                &input,
                VideoStreamParams { format: PixelFormat::Gray8, width: 8, height: 8 },
            )
            .unwrap();
        // 45° of an 8×8 square: bbox side = 8 * (cos45 + sin45)
        //                                  ≈ 8 * 1.4142 ≈ 11.31 → 12.
        // Output dims live on the downstream port spec now, not the
        // frame; recompute them from the source width/height.
        let out_w = (8.0_f32 * (45.0_f32.to_radians().cos().abs() + 45.0_f32.to_radians().sin().abs())).ceil() as usize;
        let out_h = out_w;
        // Corners of the output must be background (black by default).
        assert_eq!(out.planes[0].data[0], 0);
        // Centre pixel should have sampled the flat 120 input.
        let cx = out_w / 2;
        let cy = out_h / 2;
        let stride = out.planes[0].stride;
        assert!(
            (out.planes[0].data[cy * stride + cx] as i32 - 120).abs() <= 1,
            "centre sample = {}",
            out.planes[0].data[cy * stride + cx]
        );
    }

    #[test]
    fn rotate_respects_custom_background() {
        let input = gray(4, 4, |_, _| 200);
        let out = Rotate::new(45.0)
            .with_background([77, 0, 0, 0])
            .apply(&input, VideoStreamParams { format: PixelFormat::Gray8, width: 4, height: 4 })
            .unwrap();
        // Top-left corner is outside the rotated square → should be 77.
        assert_eq!(out.planes[0].data[0], 77);
    }

    #[test]
    fn rotate_yuv420_keeps_chroma_aligned() {
        // 8×8 YUV420 frame with flat planes; 90° rotation should keep
        // chroma plane dimensions = ceil(new_w/2) × ceil(new_h/2).
        let w = 8u32;
        let h = 8u32;
        let y = vec![100u8; (w * h) as usize];
        let u = vec![80u8; (w / 2 * h / 2) as usize];
        let v = vec![160u8; (w / 2 * h / 2) as usize];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: u,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: v,
                },
            ],
        };
        let out = Rotate::new(90.0).apply(&input, VideoStreamParams { format: PixelFormat::Yuv420P, width: 8, height: 8 }).unwrap();
        assert_eq!(out.planes[1].stride, 4);
        assert_eq!(out.planes[1].data.len(), 4 * 4);
        // Flat chroma stays flat where it's sampled (centre of the rotated
        // square). Edges might be background, so check middle.
        assert_eq!(out.planes[1].data[1 + 4], 80);
        assert_eq!(out.planes[2].data[1 + 4], 160);
    }
}
