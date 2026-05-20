//! Curves — per-channel monotone cubic curves adjustment.
//!
//! "Curves" is the per-channel tonal-mapping primitive in every
//! photo-editor (Photoshop, GIMP, darktable). The user specifies a
//! small set of control points `(x, y)` on `[0, 255]² → [0, 255]²` and
//! the editor fits a smooth monotone curve through them. Three classic
//! interpolants apply (Curves Photoshop docs, GIMP devel-docs §curves):
//!
//! 1. **Linear** — straight segments between control points. Cheapest;
//!    visible kinks at every knot.
//! 2. **Catmull-Rom** — cubic interpolation through the points with
//!    tangents inferred from neighbours (Catmull, Rom — "A class of
//!    local interpolating splines", Computer-Aided Geometric Design,
//!    1974). Smooth but **not** monotone-safe (can overshoot below `0`
//!    or above `1`).
//! 3. **Monotone-cubic (Fritsch-Carlson)** — Fritsch, Carlson —
//!    "Monotone Piecewise Cubic Interpolation", SIAM J. Numer. Anal.
//!    17 (1980), §3 algorithm. Cubic Hermite with tangents adjusted so
//!    the interpolant is monotone if the input data is monotone. This
//!    is what photo-editors actually want — sliders moving up should
//!    never invert a tone range.
//!
//! We implement all three under [`CurveInterpolation`] and apply the
//! curve as a per-channel 256-entry LUT.
//!
//! Channels supported: a single `master` curve always runs on every
//! tone channel; optional `red`/`green`/`blue` curves replace the
//! master on the corresponding RGB / RGBA channels. Alpha is
//! pass-through.
//!
//! Operates on `Gray8` / `Rgb24` / `Rgba` and planar YUV (master curve
//! on luma plane only — chroma untouched, matching ImageMagick's
//! `-channel RGB -function …` convention for tonal ops).

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Interpolation mode for [`Curves`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurveInterpolation {
    /// Straight segments between knots — no smoothing.
    Linear,
    /// Catmull-Rom cubic. Smooth but can overshoot.
    CatmullRom,
    /// Fritsch-Carlson monotone-cubic Hermite. Default — smooth and
    /// monotonicity-preserving.
    #[default]
    MonotoneCubic,
}

/// A single tonal curve: an ordered list of `(input, output)` control
/// points on `[0, 255]`.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    /// Control points sorted by `x` ascending. Endpoints `(0, …)` and
    /// `(255, …)` are auto-injected if missing.
    pub points: Vec<(u8, u8)>,
}

impl Curve {
    /// New curve through these control points (auto-sorts, dedups by
    /// `x`).
    pub fn new(points: impl IntoIterator<Item = (u8, u8)>) -> Self {
        let mut points: Vec<_> = points.into_iter().collect();
        points.sort_by_key(|p| p.0);
        points.dedup_by_key(|p| p.0);
        Self { points }
    }

    /// Identity curve `y = x` (single segment).
    pub fn identity() -> Self {
        Self {
            points: vec![(0, 0), (255, 255)],
        }
    }

    fn ensure_endpoints(&self) -> Vec<(f32, f32)> {
        let mut out: Vec<(f32, f32)> = self
            .points
            .iter()
            .map(|p| (p.0 as f32, p.1 as f32))
            .collect();
        if out.is_empty() || out[0].0 > 0.0 {
            // Inject the natural left endpoint at y = first y (held).
            let y0 = out.first().map(|p| p.1).unwrap_or(0.0);
            out.insert(0, (0.0, y0));
        }
        if out.last().is_none() || out.last().unwrap().0 < 255.0 {
            let yn = out.last().map(|p| p.1).unwrap_or(255.0);
            out.push((255.0, yn));
        }
        out
    }

    fn build_lut(&self, mode: CurveInterpolation) -> [u8; 256] {
        let pts = self.ensure_endpoints();
        let n = pts.len();
        if n < 2 {
            // Trivial: identity.
            let mut lut = [0u8; 256];
            for (v, slot) in lut.iter_mut().enumerate() {
                *slot = v as u8;
            }
            return lut;
        }
        // Per-segment tangents for the cubic-Hermite paths.
        let mut tangents = vec![0f32; n];
        match mode {
            CurveInterpolation::Linear => {}
            CurveInterpolation::CatmullRom => {
                // Standard Catmull-Rom tangents (centred differences;
                // one-sided at the boundaries).
                for i in 0..n {
                    let (xl, yl) = if i == 0 { pts[0] } else { pts[i - 1] };
                    let (xr, yr) = if i + 1 == n { pts[n - 1] } else { pts[i + 1] };
                    let dx = xr - xl;
                    tangents[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        (yr - yl) / dx
                    };
                }
            }
            CurveInterpolation::MonotoneCubic => {
                // Fritsch-Carlson §3. Step 1: secant slopes.
                let mut delta = vec![0f32; n - 1];
                for i in 0..n - 1 {
                    let dx = pts[i + 1].0 - pts[i].0;
                    delta[i] = if dx.abs() < 1.0e-6 {
                        0.0
                    } else {
                        (pts[i + 1].1 - pts[i].1) / dx
                    };
                }
                // Step 2: initial tangents = average of neighbouring
                // secants (Hermite default), one-sided at boundary.
                tangents[0] = delta[0];
                tangents[n - 1] = delta[n - 2];
                for i in 1..n - 1 {
                    tangents[i] = 0.5 * (delta[i - 1] + delta[i]);
                }
                // Step 3: enforce monotonicity (Fritsch-Carlson eq. 4).
                for i in 0..n - 1 {
                    if delta[i].abs() < 1.0e-9 {
                        tangents[i] = 0.0;
                        tangents[i + 1] = 0.0;
                        continue;
                    }
                    let a = tangents[i] / delta[i];
                    let b = tangents[i + 1] / delta[i];
                    if a < 0.0 {
                        tangents[i] = 0.0;
                    }
                    if b < 0.0 {
                        tangents[i + 1] = 0.0;
                    }
                    let s = a * a + b * b;
                    if s > 9.0 {
                        let t = 3.0 / s.sqrt();
                        tangents[i] = t * a * delta[i];
                        tangents[i + 1] = t * b * delta[i];
                    }
                }
            }
        }

        let mut lut = [0u8; 256];
        let mut seg = 0;
        for (v, slot) in lut.iter_mut().enumerate() {
            let x = v as f32;
            while seg + 1 < n && pts[seg + 1].0 < x {
                seg += 1;
            }
            let i = seg;
            let j = (i + 1).min(n - 1);
            let (x0, y0) = pts[i];
            let (x1, y1) = pts[j];
            let dx = x1 - x0;
            let t = if dx.abs() < 1.0e-6 {
                0.0
            } else {
                (x - x0) / dx
            };
            let y = match mode {
                CurveInterpolation::Linear => y0 + (y1 - y0) * t,
                _ => {
                    // Cubic Hermite (Wikipedia §"Cubic Hermite spline"
                    // / Fritsch-Carlson §2.1).
                    let h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
                    let h10 = t * (1.0 - t) * (1.0 - t);
                    let h01 = t * t * (3.0 - 2.0 * t);
                    let h11 = t * t * (t - 1.0);
                    h00 * y0 + h10 * dx * tangents[i] + h01 * y1 + h11 * dx * tangents[j]
                }
            };
            *slot = y.clamp(0.0, 255.0).round() as u8;
        }
        lut
    }
}

/// Per-channel curves filter.
#[derive(Clone, Debug, Default)]
pub struct Curves {
    /// Master curve applied to every tone channel by default.
    pub master: Curve,
    /// Per-channel overrides. `Some(c)` replaces the master on that
    /// channel; `None` falls back to the master.
    pub red: Option<Curve>,
    pub green: Option<Curve>,
    pub blue: Option<Curve>,
    /// Interpolation kind for all curves in this filter.
    pub interpolation: CurveInterpolation,
}

impl Curves {
    pub fn new(master: Curve) -> Self {
        Self {
            master,
            red: None,
            green: None,
            blue: None,
            interpolation: CurveInterpolation::default(),
        }
    }

    pub fn with_interpolation(mut self, i: CurveInterpolation) -> Self {
        self.interpolation = i;
        self
    }

    pub fn with_red(mut self, c: Curve) -> Self {
        self.red = Some(c);
        self
    }

    pub fn with_green(mut self, c: Curve) -> Self {
        self.green = Some(c);
        self
    }

    pub fn with_blue(mut self, c: Curve) -> Self {
        self.blue = Some(c);
        self
    }

    fn luts(&self) -> ([u8; 256], [u8; 256], [u8; 256]) {
        let master = self.master.build_lut(self.interpolation);
        let r = self
            .red
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        let g = self
            .green
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        let b = self
            .blue
            .as_ref()
            .map(|c| c.build_lut(self.interpolation))
            .unwrap_or(master);
        (r, g, b)
    }
}

impl ImageFilter for Curves {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let (lr, lg, lb) = self.luts();
        match params.format {
            PixelFormat::Gray8 => {
                if input.planes.is_empty() {
                    return Err(Error::invalid("Curves: missing plane"));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let mut out = vec![0u8; w * h];
                let s = &input.planes[0];
                let master = self.master.build_lut(self.interpolation);
                for y in 0..h {
                    for x in 0..w {
                        out[y * w + x] = master[s.data[y * s.stride + x] as usize];
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: w,
                        data: out,
                    }],
                })
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if matches!(params.format, PixelFormat::Rgb24) {
                    3
                } else {
                    4
                };
                if input.planes.is_empty() {
                    return Err(Error::invalid("Curves: missing plane"));
                }
                let w = params.width as usize;
                let h = params.height as usize;
                let stride = w * bpp;
                let mut out = vec![0u8; stride * h];
                let s = &input.planes[0];
                for y in 0..h {
                    let src = &s.data[y * s.stride..y * s.stride + stride];
                    let dst = &mut out[y * stride..(y + 1) * stride];
                    for x in 0..w {
                        let b = x * bpp;
                        dst[b] = lr[src[b] as usize];
                        dst[b + 1] = lg[src[b + 1] as usize];
                        dst[b + 2] = lb[src[b + 2] as usize];
                        if bpp == 4 {
                            dst[b + 3] = src[b + 3];
                        }
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane { stride, data: out }],
                })
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                // Master curve on luma only — chroma planes pass-through.
                let master = self.master.build_lut(self.interpolation);
                let w = params.width as usize;
                let h = params.height as usize;
                if input.planes.len() < 3 {
                    return Err(Error::invalid("Curves YUV: expected 3 planes"));
                }
                let s = &input.planes[0];
                let mut y_out = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        y_out[y * w + x] = master[s.data[y * s.stride + x] as usize];
                    }
                }
                let planes = vec![
                    VideoPlane {
                        stride: w,
                        data: y_out,
                    },
                    input.planes[1].clone(),
                    input.planes[2].clone(),
                ];
                Ok(VideoFrame {
                    pts: input.pts,
                    planes,
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Curves does not yet handle {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn identity_is_identity_linear() {
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Curves::new(Curve::identity())
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn identity_is_identity_cubic() {
        let input = rgb(4, 4, |x, y| [(x * 32) as u8, (y * 32) as u8, 100]);
        let out = Curves::new(Curve::identity())
            .with_interpolation(CurveInterpolation::MonotoneCubic)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Cubic on a 2-point straight line should still be identity
        // (the Hermite tangents collapse to the secant slope).
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1, "got {a} expected {b}");
        }
    }

    #[test]
    fn s_curve_brightens_midtones() {
        // Classic S-curve: (0,0), (64,32), (128,160), (192,224), (255,255).
        let curve = Curve::new([(0, 0), (64, 32), (128, 160), (192, 224), (255, 255)]);
        let input = rgb(4, 4, |_, _| [128, 128, 128]);
        let out = Curves::new(curve)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(4, 4))
            .unwrap();
        // Midtone 128 should map to 160.
        assert_eq!(out.planes[0].data[0], 160);
    }

    #[test]
    fn monotone_cubic_does_not_overshoot() {
        // Construct a curve with steep rises but request monotone cubic.
        // The Fritsch-Carlson tangents must prevent overshoot below
        // segment endpoints.
        let curve = Curve::new([(0, 0), (10, 0), (245, 255), (255, 255)]);
        let curves = Curves::new(curve).with_interpolation(CurveInterpolation::MonotoneCubic);
        // Apply on a synthetic ramp and verify monotone-non-decreasing
        // output.
        let input = rgb(8, 1, |x, _| {
            let v = (x * 32).min(255) as u8;
            [v, v, v]
        });
        let out = curves.apply(&input, p_rgb(8, 1)).unwrap();
        let mut prev = 0u8;
        for x in 0..8 {
            let v = out.planes[0].data[x * 3];
            assert!(v >= prev, "non-monotone at x={x}: {prev} → {v}");
            prev = v;
        }
    }

    #[test]
    fn per_channel_curve_overrides_master() {
        let master = Curve::new([(0, 0), (255, 255)]); // identity
        let red = Curve::new([(0, 0), (128, 64), (255, 128)]); // halve red
        let input = rgb(2, 2, |_, _| [200, 200, 200]);
        let out = Curves::new(master)
            .with_red(red)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(&input, p_rgb(2, 2))
            .unwrap();
        let pix = &out.planes[0].data[0..3];
        assert!(pix[0] < 128, "red should be halved, got {}", pix[0]);
        assert_eq!(pix[1], 200);
        assert_eq!(pix[2], 200);
    }

    #[test]
    fn alpha_pass_through_rgba() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[100u8, 100, 100, 88]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let curve = Curve::new([(0, 0), (255, 200)]);
        let out = Curves::new(curve)
            .with_interpolation(CurveInterpolation::Linear)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgba,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[3], 88);
        }
    }

    #[test]
    fn yuv_luma_only() {
        let y_data = vec![100u8; 16];
        let u_data = vec![64u8; 4];
        let v_data = vec![192u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: y_data,
                },
                VideoPlane {
                    stride: 2,
                    data: u_data.clone(),
                },
                VideoPlane {
                    stride: 2,
                    data: v_data.clone(),
                },
            ],
        };
        let out = Curves::new(Curve::new([(0, 0), (255, 128)]))
            .with_interpolation(CurveInterpolation::Linear)
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Luma 100 → curve(100) = 100 * 128 / 255 ≈ 50.
        assert!((out.planes[0].data[0] as i32 - 50).abs() <= 1);
        assert_eq!(out.planes[1].data, u_data);
        assert_eq!(out.planes[2].data, v_data);
    }
}
