//! Signed Distance Field (SDF) from a binary mask.
//!
//! A **signed** distance field stores, for every pixel, the Euclidean
//! distance to the nearest mask boundary, with a sign that records
//! which side of the boundary the pixel sits on: negative (or below the
//! neutral midpoint, once rendered to `Gray8`) **inside** the
//! foreground, positive **outside** it. It is the standard
//! intermediate for resolution-independent glyph / shape rendering,
//! soft feathering, outline / glow generation, and morphological
//! grow / shrink (a threshold at any iso-level erodes or dilates the
//! shape).
//!
//! ## Algorithm
//!
//! `docs/image/filter/distance-transform.md` §1 states the (generalised,
//! squared-Euclidean) distance transform
//!
//! ```text
//!   D_f(p) = min_{q} ( ‖p − q‖² + f(q) )
//! ```
//!
//! and notes in its intro that the transform backs "SDF generation".
//! The signed field is the difference of two unsigned exact-Euclidean
//! transforms (§2, Felzenszwalb–Huttenlocher):
//!
//! ```text
//!   d_out(p) = distance from p to the nearest FOREGROUND pixel
//!   d_in(p)  = distance from p to the nearest BACKGROUND pixel
//!   sdf(p)   = d_out(p) − d_in(p)
//! ```
//!
//! `d_out` is the EDT of the mask (`f(q) = 0` on foreground); `d_in` is
//! the EDT of the **inverted** mask (`f(q) = 0` on background). At a
//! foreground pixel `d_out = 0` so `sdf = −d_in ≤ 0`; at a background
//! pixel `d_in = 0` so `sdf = +d_out ≥ 0`. The zero level-set is the
//! shape boundary. Both component transforms are computed with the
//! exact separable lower-envelope march already used by
//! [`EuclideanDistanceTransform`](crate::EuclideanDistanceTransform);
//! the whole filter is `O(d · N)` for an **exact** signed field.
//!
//! Clean-room transcription from `docs/image/filter/distance-transform.md`
//! §1 (generalised DT + the stated SDF use-case) and §2 (the exact
//! Euclidean transform whose 1-D driver this filter reuses).
//!
//! ## Input / output
//!
//! Always single-plane `Gray8` input + single-plane `Gray8` output of
//! the same dimensions. The signed distance is in **input-pixel
//! units**, multiplied by [`scale`](Self::scale), then biased by the
//! neutral midpoint [`midpoint`](Self::midpoint) (default `128`) and
//! clamped to `[0, 255]`:
//!
//! ```text
//!   out = clamp( midpoint + scale · sdf, 0, 255 )
//! ```
//!
//! So an exactly-on-boundary pixel renders as `midpoint`; interior
//! pixels darken below it and exterior pixels brighten above it
//! (set [`invert`](Self::invert) to swap which intensity test is the
//! foreground, exactly as on the unsigned transform). A degenerate
//! mask (all-foreground or all-background) has no boundary, so one of
//! the two component distances saturates to `+∞`; the renderer clamps
//! it to the `Gray8` floor / ceiling accordingly.

use crate::euclidean_distance_transform::{dt_1d, INF};
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Signed-distance-field filter (difference of two exact-Euclidean
/// distance transforms; separable lower-envelope-of-parabolas per
/// dimension).
#[derive(Clone, Copy, Debug)]
pub struct SignedDistanceField {
    /// Foreground threshold. Pixels with `value >= threshold` count as
    /// foreground (negative side of the field); everything else is
    /// background (positive side).
    pub threshold: u8,
    /// If `true`, dark pixels (`value < threshold`) become the
    /// foreground. Useful when the source is a black drawing on a
    /// white background.
    pub invert: bool,
    /// Multiplier on the **signed** distance before the midpoint bias.
    /// `1.0` (default) maps one input-pixel of signed distance to one
    /// output-luminance unit; larger values steepen the falloff (the
    /// field saturates closer to the boundary), smaller values flatten
    /// it. Must be finite and positive.
    pub scale: f32,
    /// Output luminance assigned to an exactly-on-boundary pixel
    /// (signed distance `0`). Default `128` centres the field so
    /// interior and exterior have symmetric headroom. Clamped into
    /// `[0, 255]` at construction.
    pub midpoint: u8,
}

impl Default for SignedDistanceField {
    fn default() -> Self {
        Self {
            threshold: 128,
            invert: false,
            scale: 1.0,
            midpoint: 128,
        }
    }
}

impl SignedDistanceField {
    /// New filter with the given foreground `threshold` and the
    /// defaults `invert = false`, `scale = 1.0`, `midpoint = 128`.
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            invert: false,
            scale: 1.0,
            midpoint: 128,
        }
    }

    /// Flip the foreground/background test (dark pixels become FG).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Set the signed-distance scale. Non-finite or non-positive
    /// values are silently rejected (the previous value sticks).
    pub fn with_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
        }
        self
    }

    /// Set the on-boundary neutral output luminance.
    pub fn with_midpoint(mut self, midpoint: u8) -> Self {
        self.midpoint = midpoint;
        self
    }
}

/// Run the exact separable 2-D squared-Euclidean transform over the
/// seed array `f` (0 at feature sites, [`INF`] elsewhere), in place.
/// Mirrors the column-then-row schedule of
/// [`EuclideanDistanceTransform`]; both share the same `dt_1d` driver.
fn edt_2d(f: &mut [f64], w: usize, h: usize) {
    let n_max = w.max(h);
    let mut v_buf = vec![0_usize; n_max];
    let mut z_buf = vec![0.0_f64; n_max + 1];
    let mut col_in = vec![0.0_f64; h];
    let mut col_out = vec![0.0_f64; h];
    let mut row_in = vec![0.0_f64; w];
    let mut row_out = vec![0.0_f64; w];
    // Column pass.
    for x in 0..w {
        for y in 0..h {
            col_in[y] = f[y * w + x];
        }
        dt_1d(
            &col_in,
            &mut col_out[..h],
            &mut v_buf[..h],
            &mut z_buf[..h + 1],
        );
        for y in 0..h {
            f[y * w + x] = col_out[y];
        }
    }
    // Row pass.
    for y in 0..h {
        for x in 0..w {
            row_in[x] = f[y * w + x];
        }
        dt_1d(
            &row_in,
            &mut row_out[..w],
            &mut v_buf[..w],
            &mut z_buf[..w + 1],
        );
        for x in 0..w {
            f[y * w + x] = row_out[x];
        }
    }
}

impl ImageFilter for SignedDistanceField {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: SignedDistanceField requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: SignedDistanceField requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];

        // Two seeds: `fg` has 0 at foreground sites (its EDT is the
        // distance to the nearest foreground = d_out); `bg` has 0 at
        // background sites (its EDT is d_in). Build both in one scan.
        let mut fg = vec![INF; w * h];
        let mut bg = vec![INF; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = src.data[y * src.stride + x];
                let is_fg = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
                if is_fg {
                    fg[y * w + x] = 0.0;
                } else {
                    bg[y * w + x] = 0.0;
                }
            }
        }

        edt_2d(&mut fg, w, h);
        edt_2d(&mut bg, w, h);

        // Render: sdf = d_out − d_in = sqrt(fg) − sqrt(bg). When a side
        // has no feature its squared distance is the INF sentinel; the
        // sqrt + scale + clamp absorbs that into the Gray8 ceiling /
        // floor without any special-casing.
        let mut out = vec![0u8; w * h];
        let mid = self.midpoint as f64;
        let s = self.scale as f64;
        for i in 0..w * h {
            let d_out = fg[i].max(0.0).sqrt();
            let d_in = bg[i].max(0.0).sqrt();
            let sdf = d_out - d_in;
            let pix = mid + s * sdf;
            out[i] = pix.round().clamp(0.0, 255.0) as u8;
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    /// Brute-force signed Euclidean field: at every pixel walk every
    /// other pixel, find the nearest foreground and nearest background,
    /// return `sqrt(d_out²) − sqrt(d_in²)`. Quadratic but exact; the
    /// oracle for the separable F–H machinery.
    fn brute_force_sdf(mask: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for ty in 0..h {
            for tx in 0..w {
                let mut best_fg = f64::INFINITY;
                let mut best_bg = f64::INFINITY;
                for sy in 0..h {
                    for sx in 0..w {
                        let dx = tx as f64 - sx as f64;
                        let dy = ty as f64 - sy as f64;
                        let d2 = dx * dx + dy * dy;
                        if mask[sy * w + sx] {
                            if d2 < best_fg {
                                best_fg = d2;
                            }
                        } else if d2 < best_bg {
                            best_bg = d2;
                        }
                    }
                }
                out[ty * w + tx] = best_fg.sqrt() - best_bg.sqrt();
            }
        }
        out
    }

    /// Run the production `apply` on `mask`, decoding the signed field
    /// from a midpoint-0 / scale-1 render is lossy (clamps + rounds);
    /// instead replicate the two EDT passes the way `apply` does and
    /// hand back the raw signed field for an exact oracle comparison.
    fn run_sdf(mask: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut fg = vec![INF; w * h];
        let mut bg = vec![INF; w * h];
        for i in 0..w * h {
            if mask[i] {
                fg[i] = 0.0;
            } else {
                bg[i] = 0.0;
            }
        }
        edt_2d(&mut fg, w, h);
        edt_2d(&mut bg, w, h);
        let mut out = vec![0.0; w * h];
        for i in 0..w * h {
            out[i] = fg[i].max(0.0).sqrt() - bg[i].max(0.0).sqrt();
        }
        out
    }

    #[test]
    fn boundary_renders_at_midpoint() {
        // Two-column split: left half FG, right half BG. The pixels
        // straddling the boundary are at signed distance ±0.5·… so the
        // column just inside FG renders just below midpoint and the
        // column just outside renders just above. The midpoint bias is
        // exercised by checking symmetry around 128.
        let w = 8u32;
        let input = gray(w, 1, |x, _| if x < 4 { 255 } else { 0 });
        let out = SignedDistanceField::default()
            .apply(&input, p_gray(w, 1))
            .unwrap();
        let row = &out.planes[0].data;
        // Column 3 is the last FG pixel; nearest BG is column 4 at
        // distance 1, nearest FG is itself at distance 0 → sdf = -1 →
        // 127. Column 4 is the first BG pixel; nearest FG is column 3
        // at distance 1 → sdf = +1 → 129.
        assert_eq!(row[3], 127);
        assert_eq!(row[4], 129);
    }

    #[test]
    fn interior_is_below_exterior_is_above_midpoint() {
        // A solid 5×5 FG block centred in a 11×11 frame. The block
        // centre is deep inside → well below 128; a far corner is deep
        // outside → well above 128.
        let input = gray(11, 11, |x, y| {
            if (3..=7).contains(&x) && (3..=7).contains(&y) {
                255
            } else {
                0
            }
        });
        let out = SignedDistanceField::default()
            .apply(&input, p_gray(11, 11))
            .unwrap();
        let centre = out.planes[0].data[5 * 11 + 5];
        let corner = out.planes[0].data[0];
        assert!(
            centre < 128,
            "block centre should sit below midpoint: {centre}"
        );
        assert!(
            corner > 128,
            "far corner should sit above midpoint: {corner}"
        );
    }

    #[test]
    fn matches_brute_force_random_mask() {
        // Deterministic scatter of FG points; the separable signed
        // field must agree with the quadratic brute force everywhere.
        let w = 12usize;
        let h = 10usize;
        let mut mask = vec![false; w * h];
        for &(x, y) in &[(2, 1), (9, 2), (5, 5), (1, 7), (11, 8), (6, 0), (3, 9)] {
            mask[y * w + x] = true;
        }
        let oracle = brute_force_sdf(&mask, w, h);
        let got = run_sdf(&mask, w, h);
        for i in 0..w * h {
            let dg = (got[i] - oracle[i]).abs();
            assert!(
                dg < 1e-6,
                "SDF disagrees with brute force at i={i}: |Δ|={dg}"
            );
        }
    }

    #[test]
    fn matches_brute_force_solid_block() {
        // A solid rectangle: interior pixels carry negative distance
        // (to nearest edge), exterior positive. Stresses both EDT
        // passes simultaneously.
        let w = 9usize;
        let h = 9usize;
        let mut mask = vec![false; w * h];
        for y in 2..7 {
            for x in 2..7 {
                mask[y * w + x] = true;
            }
        }
        let oracle = brute_force_sdf(&mask, w, h);
        let got = run_sdf(&mask, w, h);
        for i in 0..w * h {
            let dg = (got[i] - oracle[i]).abs();
            assert!(dg < 1e-6, "block SDF mismatch at i={i}: |Δ|={dg}");
        }
    }

    #[test]
    fn sign_convention_fg_negative_bg_positive() {
        // With the raw (unrendered) field: every FG pixel must have
        // sdf ≤ 0 and every BG pixel sdf ≥ 0.
        let w = 10usize;
        let h = 8usize;
        let mut mask = vec![false; w * h];
        for y in 3..5 {
            for x in 4..6 {
                mask[y * w + x] = true;
            }
        }
        let field = run_sdf(&mask, w, h);
        for i in 0..w * h {
            if mask[i] {
                assert!(field[i] <= 1e-9, "FG pixel {i} not ≤ 0: {}", field[i]);
            } else {
                assert!(field[i] >= -1e-9, "BG pixel {i} not ≥ 0: {}", field[i]);
            }
        }
    }

    #[test]
    fn scale_steepens_the_falloff() {
        // Single FG pixel at (0,0) in a 1-row image. With scale 1 the
        // pixel at distance 3 renders 128+3=131; with scale 4 it
        // renders 128+12=140 (still below the clamp).
        let input = gray(8, 1, |x, _| if x == 0 { 255 } else { 0 });
        let out1 = SignedDistanceField::default()
            .apply(&input, p_gray(8, 1))
            .unwrap();
        assert_eq!(out1.planes[0].data[3], 131);
        let out4 = SignedDistanceField::default()
            .with_scale(4.0)
            .apply(&input, p_gray(8, 1))
            .unwrap();
        assert_eq!(out4.planes[0].data[3], 140);
    }

    #[test]
    fn midpoint_shifts_the_neutral_level() {
        // A boundary pixel (sdf = 0) renders at exactly `midpoint`.
        // Left-half / right-half split → column 3 is last FG (sdf=-1),
        // so with midpoint 200 it renders 199.
        let input = gray(8, 1, |x, _| if x < 4 { 255 } else { 0 });
        let out = SignedDistanceField::default()
            .with_midpoint(200)
            .apply(&input, p_gray(8, 1))
            .unwrap();
        assert_eq!(out.planes[0].data[3], 199);
        assert_eq!(out.planes[0].data[4], 201);
    }

    #[test]
    fn invert_swaps_inside_and_outside() {
        // Bright background, single dark dot at (3,3). Without invert
        // the bright field is FG so the dot is "outside" (positive,
        // above midpoint). With invert, the dot becomes FG → its
        // centre is "inside" (≤ midpoint).
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 0 } else { 255 });
        let plain = SignedDistanceField::new(128)
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert!(plain.planes[0].data[3 * 7 + 3] > 128);
        let inv = SignedDistanceField::new(128)
            .with_invert(true)
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert!(inv.planes[0].data[3 * 7 + 3] <= 128);
    }

    #[test]
    fn all_foreground_saturates_dark() {
        // No background at all → d_in = +∞ everywhere → sdf hugely
        // negative → every pixel clamps to the Gray8 floor 0.
        let input = gray(8, 8, |_, _| 255);
        let out = SignedDistanceField::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn all_background_saturates_bright() {
        // No foreground at all → d_out = +∞ everywhere → sdf hugely
        // positive → every pixel clamps to the Gray8 ceiling 255.
        let input = gray(8, 8, |_, _| 0);
        let out = SignedDistanceField::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn scale_silently_ignores_bad_input() {
        let f = SignedDistanceField::default()
            .with_scale(f32::NAN)
            .with_scale(-2.0)
            .with_scale(0.0);
        assert_eq!(f.scale, 1.0);
        let f2 = SignedDistanceField::default().with_scale(2.5);
        assert_eq!(f2.scale, 2.5);
    }

    #[test]
    fn empty_dimensions_clone_input() {
        let input = VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane {
                stride: 0,
                data: vec![],
            }],
        };
        let out = SignedDistanceField::default()
            .apply(&input, p_gray(0, 0))
            .unwrap();
        assert_eq!(out.pts, Some(7));
    }

    #[test]
    fn rejects_rgb_input() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: vec![0; 48],
            }],
        };
        let err = SignedDistanceField::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("SignedDistanceField"));
    }

    #[test]
    fn rejects_yuv_input() {
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![0; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 8],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 8],
                },
            ],
        };
        let err = SignedDistanceField::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("SignedDistanceField"));
    }
}
