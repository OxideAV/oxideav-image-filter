//! Porter–Duff and arithmetic composite operators.
//!
//! Implements the twelve standard two-input pixel composites that
//! mirror ImageMagick's `-compose <op>` family. Every operator takes
//! a foreground (`src`) and a background (`dst`) frame of identical
//! shape and pixel format, and produces a new frame.
//!
//! Reference algebra: Porter & Duff, "Compositing Digital Images",
//! SIGGRAPH 1984. The eight Porter–Duff operators (`over`, `in`,
//! `out`, `atop`, `xor` and their alpha-only siblings) are derived
//! from the per-pixel formula
//!
//! ```text
//! out_rgb = src_rgb * fa + dst_rgb * fb
//! out_a   = src_a   * fa + dst_a   * fb
//! ```
//!
//! where `fa` / `fb` are the operator-specific source / destination
//! coverage factors. The four arithmetic blends (`plus`,
//! `multiply`, `screen`, `darken`, `lighten`, `overlay`,
//! `difference`) follow the per-channel formulas listed in any
//! standard graphics reference (e.g. PDF 1.7 §11.3.5 — but
//! transcribed here from first principles, no library code).
//!
//! # Alpha convention
//!
//! All operators consume and produce **straight (non-premultiplied)**
//! alpha, matching every other RGBA filter in this crate. The
//! Porter–Duff coverage factors are written in straight-alpha form
//! (`src_rgb * src_a` etc.) so the formulas survive as-is.
//!
//! # Pixel formats
//!
//! - `Rgba` — full Porter–Duff + arithmetic; alpha is computed.
//! - `Rgb24` — alpha is treated as `1.0` everywhere (`fa` / `fb`
//!   collapse so `over` / `in` / `atop` all reduce to "src",
//!   `out` / `xor` to "dst", `plus` is additive, the per-channel
//!   blends still apply).
//! - `Gray8` — single-channel arithmetic blends only; alpha treated
//!   as `1.0`. Porter–Duff coverage operators reduce as for
//!   `Rgb24` (a Gray8 buffer has no alpha plane to drive coverage).
//!
//! Other formats return [`Error::unsupported`].

use crate::{ImageFilter, TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which composite operator to apply.
///
/// Names match ImageMagick's `-compose` argument exactly (so the
/// pipeline can map a JSON `op` string straight onto a variant).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeOp {
    /// `src` over `dst` — the default Porter–Duff "over".
    /// `out = src + dst * (1 - src.a)`.
    Over,
    /// `src` constrained to `dst`'s coverage.
    /// `out = src * dst.a`.
    In,
    /// `src` constrained to where `dst` is *not* covered.
    /// `out = src * (1 - dst.a)`.
    Out,
    /// `src` over `dst`, but result alpha follows `dst`.
    /// `out = src * dst.a + dst * (1 - src.a)`.
    Atop,
    /// Non-overlapping union of `src` and `dst`.
    /// `out = src * (1 - dst.a) + dst * (1 - src.a)`.
    Xor,
    /// Additive blend: `out = src + dst`, clamped to `255`.
    Plus,
    /// Per-channel multiplicative blend: `out = src * dst / 255`.
    Multiply,
    /// Inverse multiply: `out = 255 - (255 - src) * (255 - dst) / 255`.
    Screen,
    /// Combination of multiply (when `dst < 0.5`) and screen
    /// (when `dst >= 0.5`).
    Overlay,
    /// Per-channel minimum: `out = min(src, dst)`.
    Darken,
    /// Per-channel maximum: `out = max(src, dst)`.
    Lighten,
    /// Absolute per-channel difference: `out = |src - dst|`.
    Difference,
}

impl CompositeOp {
    /// Public name used by the JSON registry (`composite-<name>`).
    pub fn name(self) -> &'static str {
        match self {
            CompositeOp::Over => "over",
            CompositeOp::In => "in",
            CompositeOp::Out => "out",
            CompositeOp::Atop => "atop",
            CompositeOp::Xor => "xor",
            CompositeOp::Plus => "plus",
            CompositeOp::Multiply => "multiply",
            CompositeOp::Screen => "screen",
            CompositeOp::Overlay => "overlay",
            CompositeOp::Darken => "darken",
            CompositeOp::Lighten => "lighten",
            CompositeOp::Difference => "difference",
        }
    }

    /// Parse the short operator name (`"over"`, `"plus"`, …).
    /// Used by both the registry factories and any future schema
    /// reflection.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "over" => CompositeOp::Over,
            "in" => CompositeOp::In,
            "out" => CompositeOp::Out,
            "atop" => CompositeOp::Atop,
            "xor" => CompositeOp::Xor,
            "plus" => CompositeOp::Plus,
            "multiply" => CompositeOp::Multiply,
            "screen" => CompositeOp::Screen,
            "overlay" => CompositeOp::Overlay,
            "darken" => CompositeOp::Darken,
            "lighten" => CompositeOp::Lighten,
            "difference" => CompositeOp::Difference,
            _ => return None,
        })
    }
}

/// Two-input composite filter: combine `src` (foreground) and
/// `dst` (background) under the chosen [`CompositeOp`].
#[derive(Clone, Copy, Debug)]
pub struct Composite {
    pub op: CompositeOp,
}

impl Composite {
    /// Build a composite filter with the given operator.
    pub fn new(op: CompositeOp) -> Self {
        Self { op }
    }
}

impl TwoInputImageFilter for Composite {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        match params.format {
            PixelFormat::Rgba => composite_rgba(self.op, src, dst, params),
            PixelFormat::Rgb24 => composite_rgb_or_gray(self.op, src, dst, params, 3),
            PixelFormat::Gray8 => composite_rgb_or_gray(self.op, src, dst, params, 1),
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Composite requires Gray8/Rgb24/Rgba, got {other:?}"
            ))),
        }
    }
}

/// `Composite` is also a single-input filter — it composites the
/// frame *over itself* using the configured operator. This makes
/// it usable through the existing one-port `ImageFilterAdapter`
/// path purely as a convenience for tests and ad-hoc chaining;
/// the pipeline-facing factories use the two-input adapter.
impl ImageFilter for Composite {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        // Cheap clone of the planes — VideoFrame is shallow.
        let dst = VideoFrame {
            pts: input.pts,
            planes: input.planes.clone(),
        };
        self.apply_two(input, &dst, params)
    }
}

/// Normalise `(plane, params)` into a single packed RGBA buffer of
/// length `w * h * 4`. Used by the Porter–Duff path so we always
/// have a uniform `[u8; 4]` per pixel to plug into the algebra.
fn pack_rgba(plane: &VideoPlane, w: usize, h: usize) -> Vec<u8> {
    let row_bytes = w * 4;
    let mut out = Vec::with_capacity(row_bytes * h);
    for y in 0..h {
        let row_start = y * plane.stride;
        out.extend_from_slice(&plane.data[row_start..row_start + row_bytes]);
    }
    out
}

fn composite_rgba(
    op: CompositeOp,
    src: &VideoFrame,
    dst: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    if src.planes.is_empty() || dst.planes.is_empty() {
        return Err(Error::invalid(
            "oxideav-image-filter: Composite needs non-empty src and dst planes",
        ));
    }
    let src_buf = pack_rgba(&src.planes[0], w, h);
    let dst_buf = pack_rgba(&dst.planes[0], w, h);
    let row_bytes = w * 4;
    let mut out = vec![0u8; row_bytes * h];
    for i in 0..(w * h) {
        let s = [
            src_buf[i * 4],
            src_buf[i * 4 + 1],
            src_buf[i * 4 + 2],
            src_buf[i * 4 + 3],
        ];
        let d = [
            dst_buf[i * 4],
            dst_buf[i * 4 + 1],
            dst_buf[i * 4 + 2],
            dst_buf[i * 4 + 3],
        ];
        let r = composite_pixel_rgba(op, s, d);
        out[i * 4] = r[0];
        out[i * 4 + 1] = r[1];
        out[i * 4 + 2] = r[2];
        out[i * 4 + 3] = r[3];
    }
    Ok(VideoFrame {
        pts: src.pts,
        planes: vec![VideoPlane {
            stride: row_bytes,
            data: out,
        }],
    })
}

/// Per-pixel composite for straight-alpha RGBA. Returns straight-alpha
/// RGBA. All maths in `f32` for clarity; the result is rounded to
/// `u8`.
fn composite_pixel_rgba(op: CompositeOp, s: [u8; 4], d: [u8; 4]) -> [u8; 4] {
    let sa = s[3] as f32 / 255.0;
    let da = d[3] as f32 / 255.0;

    // Operator coverage factors and per-channel blend.
    match op {
        CompositeOp::Over => {
            // out_a   = sa + da * (1 - sa)
            // out_rgb = (src * sa + dst * da * (1 - sa)) / out_a
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            if out_a <= 0.0 {
                return [0, 0, 0, 0];
            }
            let mut out = [0u8; 4];
            for i in 0..3 {
                let num = (s[i] as f32 / 255.0) * sa + (d[i] as f32 / 255.0) * da * inv_sa;
                out[i] = u8_round(num / out_a * 255.0);
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::In => {
            // out_a = sa * da; rgb = src colour
            let out_a = sa * da;
            if out_a <= 0.0 {
                return [0, 0, 0, 0];
            }
            [s[0], s[1], s[2], u8_round(out_a * 255.0)]
        }
        CompositeOp::Out => {
            // out_a = sa * (1 - da); rgb = src colour
            let out_a = sa * (1.0 - da);
            if out_a <= 0.0 {
                return [0, 0, 0, 0];
            }
            [s[0], s[1], s[2], u8_round(out_a * 255.0)]
        }
        CompositeOp::Atop => {
            // out_a = da
            // out_rgb = (src * sa * da + dst * da * (1 - sa)) / out_a
            //         = src * sa + dst * (1 - sa)   (when da > 0)
            if da <= 0.0 {
                return [0, 0, 0, 0];
            }
            let inv_sa = 1.0 - sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                let num = (s[i] as f32 / 255.0) * sa + (d[i] as f32 / 255.0) * inv_sa;
                out[i] = u8_round(num * 255.0);
            }
            out[3] = d[3];
            out
        }
        CompositeOp::Xor => {
            // out_a   = sa * (1 - da) + da * (1 - sa)
            // out_rgb = (src * sa * (1 - da) + dst * da * (1 - sa)) / out_a
            let inv_sa = 1.0 - sa;
            let inv_da = 1.0 - da;
            let out_a = sa * inv_da + da * inv_sa;
            if out_a <= 0.0 {
                return [0, 0, 0, 0];
            }
            let mut out = [0u8; 4];
            for i in 0..3 {
                let num = (s[i] as f32 / 255.0) * sa * inv_da + (d[i] as f32 / 255.0) * da * inv_sa;
                out[i] = u8_round(num / out_a * 255.0);
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Plus => {
            // Saturating per-channel sum (incl. alpha).
            let mut out = [0u8; 4];
            for i in 0..4 {
                out[i] = (s[i] as u16 + d[i] as u16).min(255) as u8;
            }
            out
        }
        CompositeOp::Multiply => {
            // Per-channel product on RGB; alpha follows the standard
            // Porter–Duff "over" rebuild so transparent regions still
            // composite sensibly: out_a = sa + da * (1 - sa).
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                let m = (s[i] as u32 * d[i] as u32 + 127) / 255;
                out[i] = m.min(255) as u8;
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Screen => {
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                // out = 255 - (255 - s) * (255 - d) / 255
                let inv_s = 255 - s[i] as u32;
                let inv_d = 255 - d[i] as u32;
                let prod = (inv_s * inv_d + 127) / 255;
                out[i] = (255 - prod.min(255)) as u8;
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Overlay => {
            // For each channel, if dst < 128: 2 * src * dst / 255
            //                  else:           255 - 2 * (255 - src) * (255 - dst) / 255
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                let dv = d[i] as u32;
                let sv = s[i] as u32;
                let m = if dv < 128 {
                    (2 * sv * dv + 127) / 255
                } else {
                    let inv_s = 255 - sv;
                    let inv_d = 255 - dv;
                    let p = (2 * inv_s * inv_d + 127) / 255;
                    255 - p.min(255)
                };
                out[i] = m.min(255) as u8;
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Darken => {
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                out[i] = s[i].min(d[i]);
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Lighten => {
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                out[i] = s[i].max(d[i]);
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
        CompositeOp::Difference => {
            let inv_sa = 1.0 - sa;
            let out_a = sa + da * inv_sa;
            let mut out = [0u8; 4];
            for i in 0..3 {
                out[i] = (s[i] as i16 - d[i] as i16).unsigned_abs() as u8;
            }
            out[3] = u8_round(out_a * 255.0);
            out
        }
    }
}

/// Composite path for `Rgb24` (`bpp = 3`) and `Gray8` (`bpp = 1`).
/// Alpha is implicitly `1.0` for both inputs, so the Porter–Duff
/// coverage operators collapse:
/// - `Over`, `In`, `Atop`           → return `src` (fully opaque src
///   completely covers dst).
/// - `Out`, `Xor`                   → return all-zero (no visible
///   region survives).
/// - The arithmetic / per-channel blends survive unchanged.
fn composite_rgb_or_gray(
    op: CompositeOp,
    src: &VideoFrame,
    dst: &VideoFrame,
    params: VideoStreamParams,
    bpp: usize,
) -> Result<VideoFrame, Error> {
    let w = params.width as usize;
    let h = params.height as usize;
    if src.planes.is_empty() || dst.planes.is_empty() {
        return Err(Error::invalid(
            "oxideav-image-filter: Composite needs non-empty src and dst planes",
        ));
    }
    let row_bytes = w * bpp;
    let mut out = vec![0u8; row_bytes * h];
    let src_p = &src.planes[0];
    let dst_p = &dst.planes[0];
    for y in 0..h {
        let src_row = &src_p.data[y * src_p.stride..y * src_p.stride + row_bytes];
        let dst_row = &dst_p.data[y * dst_p.stride..y * dst_p.stride + row_bytes];
        let out_row = &mut out[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..(w * bpp) {
            let s = src_row[x];
            let d = dst_row[x];
            out_row[x] = match op {
                // Coverage operators with implicit fully-opaque alpha.
                CompositeOp::Over | CompositeOp::In | CompositeOp::Atop => s,
                CompositeOp::Out | CompositeOp::Xor => 0,
                // Arithmetic / per-channel blends.
                CompositeOp::Plus => (s as u16 + d as u16).min(255) as u8,
                CompositeOp::Multiply => ((s as u32 * d as u32 + 127) / 255).min(255) as u8,
                CompositeOp::Screen => {
                    let inv_s = 255 - s as u32;
                    let inv_d = 255 - d as u32;
                    let prod = (inv_s * inv_d + 127) / 255;
                    255 - prod.min(255) as u8
                }
                CompositeOp::Overlay => {
                    let dv = d as u32;
                    let sv = s as u32;
                    let m = if dv < 128 {
                        (2 * sv * dv + 127) / 255
                    } else {
                        let inv_s = 255 - sv;
                        let inv_d = 255 - dv;
                        let p = (2 * inv_s * inv_d + 127) / 255;
                        255 - p.min(255)
                    };
                    m.min(255) as u8
                }
                CompositeOp::Darken => s.min(d),
                CompositeOp::Lighten => s.max(d),
                CompositeOp::Difference => (s as i16 - d as i16).unsigned_abs() as u8,
            };
        }
    }
    Ok(VideoFrame {
        pts: src.pts,
        planes: vec![VideoPlane {
            stride: row_bytes,
            data: out,
        }],
    })
}

#[inline]
fn u8_round(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_pixel(s: [u8; 4], d: [u8; 4], op: CompositeOp) -> [u8; 4] {
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: s.to_vec(),
            }],
        };
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: d.to_vec(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 1,
            height: 1,
        };
        let out = Composite::new(op).apply_two(&src, &dst, params).unwrap();
        [
            out.planes[0].data[0],
            out.planes[0].data[1],
            out.planes[0].data[2],
            out.planes[0].data[3],
        ]
    }

    #[test]
    fn over_opaque_src_replaces_dst() {
        // Opaque red src over opaque blue dst ⇒ red.
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 255, 255], CompositeOp::Over);
        assert_eq!(r, [255, 0, 0, 255]);
    }

    #[test]
    fn over_half_alpha_blends_50_50() {
        // 50% red over opaque blue: out_a = 0.5 + 0.5 = 1.0
        // out_r = (255 * 0.5 + 0 * 1 * 0.5) / 1 = ~128
        // out_b = (0 * 0.5 + 255 * 1 * 0.5) / 1 = ~128
        let r = rgba_pixel([255, 0, 0, 128], [0, 0, 255, 255], CompositeOp::Over);
        assert!((r[0] as i16 - 128).abs() <= 2);
        assert_eq!(r[1], 0);
        assert!((r[2] as i16 - 127).abs() <= 2);
        assert_eq!(r[3], 255);
    }

    #[test]
    fn over_transparent_src_keeps_dst() {
        // Fully transparent src over opaque blue ⇒ blue.
        let r = rgba_pixel([255, 0, 0, 0], [0, 0, 255, 255], CompositeOp::Over);
        assert_eq!(r, [0, 0, 255, 255]);
    }

    #[test]
    fn over_opaque_src_over_transparent_keeps_src() {
        // Opaque red over transparent dst ⇒ red.
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 0, 0], CompositeOp::Over);
        assert_eq!(r, [255, 0, 0, 255]);
    }

    #[test]
    fn in_keeps_src_colour_with_dst_alpha() {
        let r = rgba_pixel([200, 100, 50, 255], [0, 0, 255, 200], CompositeOp::In);
        assert_eq!(r[0], 200);
        assert_eq!(r[1], 100);
        assert_eq!(r[2], 50);
        // sa * da = 1 * (200/255) ⇒ ~200
        assert!((r[3] as i16 - 200).abs() <= 1);
    }

    #[test]
    fn in_transparent_dst_clears_pixel() {
        let r = rgba_pixel([200, 100, 50, 255], [0, 0, 0, 0], CompositeOp::In);
        assert_eq!(r, [0, 0, 0, 0]);
    }

    #[test]
    fn out_keeps_src_where_dst_transparent() {
        let r = rgba_pixel([200, 100, 50, 255], [0, 0, 0, 0], CompositeOp::Out);
        assert_eq!(r, [200, 100, 50, 255]);
    }

    #[test]
    fn out_clears_where_dst_opaque() {
        let r = rgba_pixel([200, 100, 50, 255], [0, 0, 255, 255], CompositeOp::Out);
        assert_eq!(r, [0, 0, 0, 0]);
    }

    #[test]
    fn atop_keeps_dst_alpha() {
        // src * sa + dst * (1 - sa) with da preserved.
        // Opaque red atop opaque blue ⇒ red.
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 255, 255], CompositeOp::Atop);
        assert_eq!(r, [255, 0, 0, 255]);
        // 50% red atop opaque blue: rgb = (255*0.5 + 0*0.5, 0, 255*0.5) = (128, 0, 128)
        let r = rgba_pixel([255, 0, 0, 128], [0, 0, 255, 255], CompositeOp::Atop);
        assert!((r[0] as i16 - 128).abs() <= 2);
        assert_eq!(r[1], 0);
        assert!((r[2] as i16 - 127).abs() <= 2);
        assert_eq!(r[3], 255);
    }

    #[test]
    fn atop_transparent_dst_yields_zero_alpha() {
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 0, 0], CompositeOp::Atop);
        assert_eq!(r, [0, 0, 0, 0]);
    }

    #[test]
    fn xor_two_opaque_yields_zero_alpha() {
        // Both fully opaque ⇒ both coverage factors zero ⇒ all-zero.
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 255, 255], CompositeOp::Xor);
        assert_eq!(r, [0, 0, 0, 0]);
    }

    #[test]
    fn xor_opaque_src_over_transparent_keeps_src() {
        let r = rgba_pixel([255, 0, 0, 255], [0, 0, 0, 0], CompositeOp::Xor);
        assert_eq!(r, [255, 0, 0, 255]);
    }

    #[test]
    fn plus_clamps() {
        let r = rgba_pixel([200, 50, 0, 100], [100, 50, 200, 200], CompositeOp::Plus);
        assert_eq!(r[0], 255); // 200 + 100 ⇒ clamp
        assert_eq!(r[1], 100);
        assert_eq!(r[2], 200);
        assert_eq!(r[3], 255); // 100 + 200 ⇒ clamp
    }

    #[test]
    fn multiply_white_is_identity() {
        // White × any colour ⇒ the colour.
        let r = rgba_pixel(
            [255, 255, 255, 255],
            [128, 64, 200, 255],
            CompositeOp::Multiply,
        );
        assert_eq!(&r[0..3], &[128, 64, 200]);
        assert_eq!(r[3], 255);
    }

    #[test]
    fn multiply_black_is_zero() {
        let r = rgba_pixel([0, 0, 0, 255], [128, 64, 200, 255], CompositeOp::Multiply);
        assert_eq!(&r[0..3], &[0, 0, 0]);
    }

    #[test]
    fn screen_black_is_identity() {
        // Black screened with any colour ⇒ the colour.
        let r = rgba_pixel([0, 0, 0, 255], [128, 64, 200, 255], CompositeOp::Screen);
        assert_eq!(&r[0..3], &[128, 64, 200]);
    }

    #[test]
    fn screen_white_is_white() {
        let r = rgba_pixel(
            [255, 255, 255, 255],
            [128, 64, 200, 255],
            CompositeOp::Screen,
        );
        assert_eq!(&r[0..3], &[255, 255, 255]);
    }

    #[test]
    fn darken_picks_min() {
        let r = rgba_pixel(
            [200, 50, 100, 255],
            [100, 200, 100, 255],
            CompositeOp::Darken,
        );
        assert_eq!(&r[0..3], &[100, 50, 100]);
    }

    #[test]
    fn lighten_picks_max() {
        let r = rgba_pixel(
            [200, 50, 100, 255],
            [100, 200, 100, 255],
            CompositeOp::Lighten,
        );
        assert_eq!(&r[0..3], &[200, 200, 100]);
    }

    #[test]
    fn difference_is_abs_diff() {
        let r = rgba_pixel(
            [200, 50, 100, 255],
            [100, 200, 100, 255],
            CompositeOp::Difference,
        );
        assert_eq!(&r[0..3], &[100, 150, 0]);
    }

    #[test]
    fn overlay_dark_dst_uses_multiply() {
        // dst < 128 ⇒ 2 * src * dst / 255
        let r = rgba_pixel(
            [200, 200, 200, 255],
            [100, 100, 100, 255],
            CompositeOp::Overlay,
        );
        // 2 * 200 * 100 / 255 ≈ 156
        assert!((r[0] as i16 - 157).abs() <= 2);
    }

    #[test]
    fn overlay_bright_dst_uses_screen() {
        // dst >= 128 ⇒ 255 - 2 * (255-src) * (255-dst) / 255
        let r = rgba_pixel(
            [100, 100, 100, 255],
            [200, 200, 200, 255],
            CompositeOp::Overlay,
        );
        // 255 - 2 * 155 * 55 / 255 = 255 - 67 = 188
        assert!((r[0] as i16 - 188).abs() <= 2);
    }

    #[test]
    fn rgb24_path_works() {
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![200, 100, 50],
            }],
        };
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![50, 200, 100],
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 1,
            height: 1,
        };
        // Darken on Rgb24: per-channel min.
        let out = Composite::new(CompositeOp::Darken)
            .apply_two(&src, &dst, params)
            .unwrap();
        assert_eq!(out.planes[0].data, vec![50, 100, 50]);
        // Plus on Rgb24: per-channel sum, clamp.
        let out = Composite::new(CompositeOp::Plus)
            .apply_two(&src, &dst, params)
            .unwrap();
        assert_eq!(out.planes[0].data, vec![250, 255, 150]);
    }

    #[test]
    fn gray8_path_works() {
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![200, 100, 50, 0],
            }],
        };
        let dst = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![100, 100, 100, 100],
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 4,
            height: 1,
        };
        let out = Composite::new(CompositeOp::Difference)
            .apply_two(&src, &dst, params)
            .unwrap();
        assert_eq!(out.planes[0].data, vec![100, 0, 50, 100]);
    }

    #[test]
    fn name_round_trip() {
        for op in [
            CompositeOp::Over,
            CompositeOp::In,
            CompositeOp::Out,
            CompositeOp::Atop,
            CompositeOp::Xor,
            CompositeOp::Plus,
            CompositeOp::Multiply,
            CompositeOp::Screen,
            CompositeOp::Overlay,
            CompositeOp::Darken,
            CompositeOp::Lighten,
            CompositeOp::Difference,
        ] {
            assert_eq!(CompositeOp::from_name(op.name()), Some(op));
        }
        assert_eq!(CompositeOp::from_name("nope"), None);
    }

    #[test]
    fn rejects_unsupported_format() {
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let dst = src.clone();
        let params = VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = Composite::new(CompositeOp::Over)
            .apply_two(&src, &dst, params)
            .unwrap_err();
        assert!(err.to_string().contains("Composite"), "{err:?}");
    }
}
