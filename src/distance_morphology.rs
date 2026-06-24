//! Exact-Euclidean morphology via the distance transform.
//!
//! Classical binary morphology — dilation and erosion by a disc of
//! radius `r` — is a direct consequence of the **distance transform**
//! defined in `docs/image/filter/distance-transform.md` §1:
//!
//! ```text
//!   D_f(p) = min_{q feature} ‖p − q‖²        (squared distance to nearest feature)
//! ```
//!
//! A binary disc dilation grows the foreground set by every pixel within
//! Euclidean distance `r` of some original foreground pixel — i.e. every
//! pixel whose distance to the nearest **foreground** site is `≤ r`:
//!
//! ```text
//!   dilate_r(FG) = { p : D_FG(p) ≤ r }      ⇔   D_FG(p) ≤ r²  (squared)
//! ```
//!
//! Erosion is the dual: a foreground pixel survives only if it is at
//! least `r` away from the nearest **background** pixel (equivalently,
//! erosion is dilation of the complement):
//!
//! ```text
//!   erode_r(FG) = { p : D_BG(p) > r }       ⇔   D_BG(p) > r²  (squared)
//! ```
//!
//! Both reduce to a single exact §2.4 squared-Euclidean transform plus a
//! threshold against `r²`, so the disc is a **true Euclidean circle** —
//! not the octagon/diamond a fixed 3×3 structuring-element iteration
//! produces. Working in squared distance avoids any `sqrt` and keeps the
//! radius test exact for integer `r`.
//!
//! Clean-room transcription: the operators are pure thresholds of the
//! exact distance transform of `docs/image/filter/distance-transform.md`
//! §1 (the nearest-feature distance) computed by the §2.4 separable
//! Felzenszwalb–Huttenlocher driver. Nothing here is derived from any
//! image-library source tree.
//!
//! ## Input / output
//!
//! Single-plane `Gray8` in, single-plane `Gray8` out (same dimensions).
//! The input is binarised at `threshold` (foreground = `value ≥
//! threshold`, or `< threshold` when `invert`). Output is a binary mask:
//! `fg_value` (default `255`) on the result foreground, `0` elsewhere.

use crate::euclidean_distance_transform::edt_squared_2d;
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Which morphological operation a [`DistanceMorphology`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphOp {
    /// Grow the foreground by a Euclidean disc of the given radius
    /// (`D_FG ≤ r²`).
    Dilate,
    /// Shrink the foreground by a Euclidean disc of the given radius
    /// (`D_BG > r²`).
    Erode,
}

/// Exact-Euclidean binary morphology (dilation / erosion).
///
/// See the module docs for the distance-transform derivation. The disc
/// is a true Euclidean circle of radius [`radius`](Self::radius).
#[derive(Clone, Copy, Debug)]
pub struct DistanceMorphology {
    /// The operation to perform.
    pub op: MorphOp,
    /// Disc radius in pixels. `0` is the identity (the binarised mask is
    /// returned unchanged).
    pub radius: f32,
    /// Binarisation threshold: foreground = `value ≥ threshold`
    /// (or `< threshold` when `invert`).
    pub threshold: u8,
    /// Flip the foreground test (dark pixels become foreground).
    pub invert: bool,
    /// Output value written on result-foreground pixels (background is
    /// always `0`).
    pub fg_value: u8,
}

impl DistanceMorphology {
    /// New morphology filter with the given op + radius (threshold `128`,
    /// `invert = false`, `fg_value = 255`).
    pub fn new(op: MorphOp, radius: f32) -> Self {
        Self {
            op,
            radius,
            threshold: 128,
            invert: false,
            fg_value: 255,
        }
    }

    /// Shorthand for a dilation of the given radius.
    pub fn dilate(radius: f32) -> Self {
        Self::new(MorphOp::Dilate, radius)
    }

    /// Shorthand for an erosion of the given radius.
    pub fn erode(radius: f32) -> Self {
        Self::new(MorphOp::Erode, radius)
    }

    /// Set the binarisation threshold.
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Flip the foreground test (dark pixels become foreground).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Set the output foreground value.
    pub fn with_fg_value(mut self, fg_value: u8) -> Self {
        self.fg_value = fg_value;
        self
    }

    /// Binarise a Gray8 plane into a foreground mask per `threshold` /
    /// `invert`.
    fn build_mask(&self, data: &[u8], stride: usize, w: usize, h: usize) -> Vec<bool> {
        let mut mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = data[y * stride + x];
                mask[y * w + x] = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
            }
        }
        mask
    }
}

impl ImageFilter for DistanceMorphology {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: DistanceMorphology requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DistanceMorphology requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];
        let mask = self.build_mask(&src.data, src.stride, w, h);

        // Squared radius: the §1 distance transform yields squared
        // distances, so the disc test `D ≤ r²` (dilate) / `D > r²`
        // (erode) needs no sqrt. `r` is clamped non-negative.
        let r = self.radius.max(0.0) as f64;
        let r2 = r * r;

        // For dilation the seed set is the foreground; for erosion the
        // seed set is the background (we measure distance to the nearest
        // background pixel). Erosion of the complement equals dilation,
        // so a single transform + threshold serves both.
        let mut out = vec![0u8; w * h];
        match self.op {
            MorphOp::Dilate => {
                // dilate_r(FG) = { p : D_FG(p) ≤ r² }
                let d2 = edt_squared_2d(&mask, w, h);
                for (o, &dist) in out.iter_mut().zip(d2.iter()) {
                    if dist <= r2 {
                        *o = self.fg_value;
                    }
                }
            }
            MorphOp::Erode => {
                // erode_r(FG) = { p : D_BG(p) > r² }; seed the BG mask.
                let bg: Vec<bool> = mask.iter().map(|&m| !m).collect();
                let d2 = edt_squared_2d(&bg, w, h);
                for (o, &dist) in out.iter_mut().zip(d2.iter()) {
                    if dist > r2 {
                        *o = self.fg_value;
                    }
                }
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
}

/// Exact-Euclidean boundary band (outline / stroke) of a binary shape.
///
/// The signed Euclidean distance to a shape boundary is
///
/// ```text
///   s(p) = +D_FG(p)   for p outside the shape  (distance to nearest FG)
///   s(p) = −D_BG(p)   for p inside  the shape  (distance to nearest BG)
/// ```
///
/// both `D_FG` / `D_BG` being the exact §1 nearest-feature distance. A
/// pixel lies on the outline iff its signed distance falls in a band
/// straddling the contour:
///
/// ```text
///   outline = { p : −inner ≤ s(p) ≤ outer }
/// ```
///
/// i.e. up to `inner` pixels *inside* the boundary and up to `outer`
/// pixels *outside* it. `inner = 0, outer = w` traces a purely-outer
/// stroke of width `w`; `inner = w, outer = 0` a purely-inner stroke;
/// equal radii a centred stroke. Both comparisons run in squared
/// distance (`D_FG ≤ outer²` outside, `D_BG ≤ inner²` inside), so no
/// `sqrt` is taken and the stroke is a true Euclidean band — exactly the
/// set difference `dilate(outer) − erode(inner)` of [`DistanceMorphology`].
///
/// Clean-room from `docs/image/filter/distance-transform.md` §1 + §2.4.
///
/// Single-plane `Gray8` in / out. Output is `fg_value` on the band, `0`
/// elsewhere.
#[derive(Clone, Copy, Debug)]
pub struct DistanceOutline {
    /// Band half-width *inside* the boundary (pixels of the shape interior
    /// painted). `0` keeps the stroke entirely outside.
    pub inner: f32,
    /// Band half-width *outside* the boundary (pixels of the background
    /// painted). `0` keeps the stroke entirely inside.
    pub outer: f32,
    /// Binarisation threshold: foreground = `value ≥ threshold`
    /// (or `< threshold` when `invert`).
    pub threshold: u8,
    /// Flip the foreground test (dark pixels become the shape).
    pub invert: bool,
    /// Output value written on band pixels (background is always `0`).
    pub fg_value: u8,
}

impl DistanceOutline {
    /// New outline with the given inner / outer band half-widths
    /// (threshold `128`, `invert = false`, `fg_value = 255`).
    pub fn new(inner: f32, outer: f32) -> Self {
        Self {
            inner,
            outer,
            threshold: 128,
            invert: false,
            fg_value: 255,
        }
    }

    /// A centred stroke of total width `width` (half inside, half
    /// outside the contour).
    pub fn centred(width: f32) -> Self {
        let half = (width * 0.5).max(0.0);
        Self::new(half, half)
    }

    /// Set the binarisation threshold.
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Flip the foreground test (dark pixels become the shape).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Set the output band value.
    pub fn with_fg_value(mut self, fg_value: u8) -> Self {
        self.fg_value = fg_value;
        self
    }

    fn build_mask(&self, data: &[u8], stride: usize, w: usize, h: usize) -> Vec<bool> {
        let mut mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = data[y * stride + x];
                mask[y * w + x] = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
            }
        }
        mask
    }
}

impl ImageFilter for DistanceOutline {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: DistanceOutline requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DistanceOutline requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];
        let mask = self.build_mask(&src.data, src.stride, w, h);

        let inner = self.inner.max(0.0) as f64;
        let outer = self.outer.max(0.0) as f64;
        let inner2 = inner * inner;
        let outer2 = outer * outer;

        // Outside pixels (mask == false): on the band iff distance to the
        // nearest FG ≤ outer. Inside pixels (mask == true): on the band
        // iff distance to the nearest BG ≤ inner. Two exact §2.4
        // transforms — one seeded on FG, one on BG.
        let d_fg = edt_squared_2d(&mask, w, h);
        let bg: Vec<bool> = mask.iter().map(|&m| !m).collect();
        let d_bg = edt_squared_2d(&bg, w, h);

        let mut out = vec![0u8; w * h];
        for i in 0..w * h {
            let on = if mask[i] {
                // inside: paint up to `inner` pixels in from the edge.
                d_bg[i] <= inner2
            } else {
                // outside: paint up to `outer` pixels out from the edge.
                d_fg[i] <= outer2
            };
            if on {
                out[i] = self.fg_value;
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

    fn plane_of(f: &VideoFrame) -> &[u8] {
        &f.planes[0].data
    }

    /// Brute-force exact-Euclidean dilation oracle: a pixel is FG iff
    /// some FG seed lies within Euclidean radius `r`.
    fn brute_dilate(mask: &[bool], w: usize, h: usize, r: f64) -> Vec<bool> {
        let r2 = r * r;
        let mut out = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut hit = false;
                'scan: for yy in 0..h {
                    for xx in 0..w {
                        if mask[yy * w + xx] {
                            let dx = x as f64 - xx as f64;
                            let dy = y as f64 - yy as f64;
                            if dx * dx + dy * dy <= r2 {
                                hit = true;
                                break 'scan;
                            }
                        }
                    }
                }
                out[y * w + x] = hit;
            }
        }
        out
    }

    #[test]
    fn dilate_radius_zero_is_identity() {
        // Single centre pixel; radius 0 keeps exactly it.
        let input = gray(5, 5, |x, y| if x == 2 && y == 2 { 255 } else { 0 });
        let out = DistanceMorphology::dilate(0.0)
            .apply(&input, p_gray(5, 5))
            .unwrap();
        let p = plane_of(&out);
        for (i, &v) in p.iter().enumerate() {
            let want = if i == 2 * 5 + 2 { 255 } else { 0 };
            assert_eq!(v, want, "idx {i}");
        }
    }

    #[test]
    fn dilate_grows_a_disc_matching_brute_force() {
        let w = 11;
        let h = 11;
        // One central seed.
        let input = gray(
            w as u32,
            h as u32,
            |x, y| {
                if x == 5 && y == 5 {
                    255
                } else {
                    0
                }
            },
        );
        for &r in &[1.0_f64, 1.5, 2.0, 3.0, 4.5] {
            let out = DistanceMorphology::dilate(r as f32)
                .apply(&input, p_gray(w as u32, h as u32))
                .unwrap();
            let got = plane_of(&out);
            let mask: Vec<bool> = (0..w * h).map(|i| i == 5 * w + 5).collect();
            let want = brute_dilate(&mask, w, h, r);
            for i in 0..w * h {
                assert_eq!(
                    got[i] != 0,
                    want[i],
                    "dilate r={r} idx {i} ({},{})",
                    i % w,
                    i / w
                );
            }
        }
    }

    #[test]
    fn dilate_matches_brute_force_random_mask() {
        let w = 9;
        let h = 7;
        // Deterministic pseudo-random seeds (LCG), built directly into a
        // plane buffer so the generator's mutation is allowed.
        let mut state = 0x1234_5678u32;
        let mut data = Vec::with_capacity(w * h);
        for _ in 0..w * h {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push(if (state >> 24) & 1 == 0 { 255 } else { 0 });
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: w, data }],
        };
        let mask: Vec<bool> = plane_of(&input).iter().map(|&v| v >= 128).collect();
        for &r in &[0.0_f64, 1.0, 2.0, 3.5] {
            let out = DistanceMorphology::dilate(r as f32)
                .apply(&input, p_gray(w as u32, h as u32))
                .unwrap();
            let got = plane_of(&out);
            let want = brute_dilate(&mask, w, h, r);
            for i in 0..w * h {
                assert_eq!(got[i] != 0, want[i], "random dilate r={r} idx {i}");
            }
        }
    }

    #[test]
    fn erode_is_dilation_of_the_complement() {
        // Erosion of FG by r == complement of (dilation of BG by r).
        let w = 9;
        let h = 9;
        // A solid 5×5 block in the centre.
        let input = gray(w as u32, h as u32, |x, y| {
            if (2..7).contains(&x) && (2..7).contains(&y) {
                255
            } else {
                0
            }
        });
        let mask: Vec<bool> = plane_of(&input).iter().map(|&v| v >= 128).collect();
        for &r in &[1.0_f64, 2.0] {
            let eroded = DistanceMorphology::erode(r as f32)
                .apply(&input, p_gray(w as u32, h as u32))
                .unwrap();
            let got = plane_of(&eroded);
            // complement-of-dilation-of-complement
            let bg: Vec<bool> = mask.iter().map(|&m| !m).collect();
            let dil_bg = brute_dilate(&bg, w, h, r);
            for i in 0..w * h {
                let want = !dil_bg[i];
                assert_eq!(got[i] != 0, want, "erode r={r} idx {i}");
            }
        }
    }

    #[test]
    fn erode_then_smaller_than_original() {
        // Eroding a filled disc shrinks it strictly for r ≥ 1.
        let w = 11;
        let h = 11;
        let input = gray(w as u32, h as u32, |x, y| {
            let dx = x as f64 - 5.0;
            let dy = y as f64 - 5.0;
            if dx * dx + dy * dy <= 16.0 {
                255
            } else {
                0
            }
        });
        let orig_count = plane_of(&input).iter().filter(|&&v| v >= 128).count();
        let eroded = DistanceMorphology::erode(2.0)
            .apply(&input, p_gray(w as u32, h as u32))
            .unwrap();
        let er_count = plane_of(&eroded).iter().filter(|&&v| v != 0).count();
        assert!(
            er_count < orig_count,
            "erosion should shrink: {er_count} < {orig_count}"
        );
        assert!(
            er_count > 0,
            "erosion of a radius-4 disc by 2 should leave a core"
        );
    }

    #[test]
    fn invert_swaps_foreground() {
        // With invert, dark pixels seed the dilation.
        let input = gray(5, 5, |x, y| if x == 2 && y == 2 { 0 } else { 255 });
        let out = DistanceMorphology::dilate(1.0)
            .with_invert(true)
            .apply(&input, p_gray(5, 5))
            .unwrap();
        let p = plane_of(&out);
        // The single dark centre + its 4-neighbours within radius 1.
        assert_ne!(p[2 * 5 + 2], 0);
        assert_ne!(p[5 + 2], 0);
        assert_eq!(p[0], 0);
    }

    #[test]
    fn custom_fg_value_is_written() {
        let input = gray(3, 3, |x, y| if x == 1 && y == 1 { 255 } else { 0 });
        let out = DistanceMorphology::dilate(1.0)
            .with_fg_value(77)
            .apply(&input, p_gray(3, 3))
            .unwrap();
        let p = plane_of(&out);
        assert_eq!(p[3 + 1], 77);
    }

    #[test]
    fn no_foreground_dilates_to_empty() {
        let input = gray(4, 4, |_, _| 0);
        let out = DistanceMorphology::dilate(3.0)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        assert!(plane_of(&out).iter().all(|&v| v == 0));
    }

    #[test]
    fn fully_foreground_erodes_to_full() {
        // No background → nothing is within r of a BG pixel → all survive.
        let input = gray(4, 4, |_, _| 255);
        let out = DistanceMorphology::erode(2.0)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        assert!(plane_of(&out).iter().all(|&v| v == 255));
    }

    #[test]
    fn rejects_rgb_input() {
        let input = gray(2, 2, |_, _| 0);
        let params = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 2,
            height: 2,
        };
        let err = DistanceMorphology::dilate(1.0)
            .apply(&input, params)
            .unwrap_err();
        assert!(format!("{err}").contains("DistanceMorphology"));
    }

    // --- DistanceOutline ---

    #[test]
    fn outline_equals_dilate_minus_erode() {
        // The band {−inner ≤ s ≤ outer} is exactly the set difference of
        // dilate(outer) and erode(inner). Cross-check against the two
        // morphology ops on a solid block.
        let w = 11;
        let h = 11;
        let input = gray(w as u32, h as u32, |x, y| {
            if (3..8).contains(&x) && (3..8).contains(&y) {
                255
            } else {
                0
            }
        });
        let p = p_gray(w as u32, h as u32);
        for &(inner, outer) in &[(0.0_f32, 2.0_f32), (1.0, 0.0), (1.5, 1.5), (2.0, 3.0)] {
            let outline = DistanceOutline::new(inner, outer).apply(&input, p).unwrap();
            let dil = DistanceMorphology::dilate(outer).apply(&input, p).unwrap();
            let ero = DistanceMorphology::erode(inner).apply(&input, p).unwrap();
            let band = plane_of(&outline);
            let d = plane_of(&dil);
            let e = plane_of(&ero);
            for i in 0..w * h {
                // band = dilate AND NOT erode
                let want = (d[i] != 0) && (e[i] == 0);
                assert_eq!(
                    band[i] != 0,
                    want,
                    "outline(inner={inner},outer={outer}) idx {i}"
                );
            }
        }
    }

    #[test]
    fn outline_outer_only_is_outside_the_shape() {
        // inner=0 → no interior painted; the lone-pixel shape's own pixel
        // is interior, so with inner=0 it is NOT in the band, but its
        // neighbours within `outer` are.
        let input = gray(5, 5, |x, y| if x == 2 && y == 2 { 255 } else { 0 });
        let out = DistanceOutline::new(0.0, 1.0)
            .apply(&input, p_gray(5, 5))
            .unwrap();
        let p = plane_of(&out);
        assert_eq!(p[2 * 5 + 2], 0, "centre is interior, inner=0 excludes it");
        assert_ne!(p[5 + 2], 0, "the pixel above is within outer=1");
    }

    #[test]
    fn outline_inner_only_is_inside_the_shape() {
        // outer=0 → nothing outside painted; a solid block's rim (within
        // `inner` of the edge) is painted, but background stays 0.
        let w = 9;
        let h = 9;
        let input = gray(w as u32, h as u32, |x, y| {
            if (2..7).contains(&x) && (2..7).contains(&y) {
                255
            } else {
                0
            }
        });
        let out = DistanceOutline::new(1.0, 0.0)
            .apply(&input, p_gray(w as u32, h as u32))
            .unwrap();
        let p = plane_of(&out);
        // A corner of the block (2,2) is within 1 px of the edge → painted.
        assert_ne!(p[2 * w + 2], 0);
        // The block centre (4,4) is 2 px deep → NOT within inner=1.
        assert_eq!(p[4 * w + 4], 0);
        // Pure background well outside stays 0.
        assert_eq!(p[0], 0);
    }

    #[test]
    fn outline_centred_splits_width() {
        // centred(2.0) → inner = outer = 1.0.
        let f = DistanceOutline::centred(2.0);
        assert_eq!(f.inner, 1.0);
        assert_eq!(f.outer, 1.0);
    }

    #[test]
    fn outline_no_shape_is_empty() {
        let input = gray(4, 4, |_, _| 0);
        let out = DistanceOutline::new(2.0, 2.0)
            .apply(&input, p_gray(4, 4))
            .unwrap();
        // No FG → the whole frame is "outside"; outer band measures
        // distance to the nearest FG which is INF everywhere → empty.
        assert!(plane_of(&out).iter().all(|&v| v == 0));
    }

    #[test]
    fn outline_rejects_rgb_input() {
        let input = gray(2, 2, |_, _| 0);
        let params = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 2,
            height: 2,
        };
        let err = DistanceOutline::new(1.0, 1.0)
            .apply(&input, params)
            .unwrap_err();
        assert!(format!("{err}").contains("DistanceOutline"));
    }
}
