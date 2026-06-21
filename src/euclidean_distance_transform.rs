//! Exact Euclidean Distance Transform (Felzenszwalb–Huttenlocher).
//!
//! Computes, for every pixel of a thresholded binary mask, the
//! **exact** Euclidean distance to the nearest foreground pixel. The
//! companion [`DistanceTransform`](crate::DistanceTransform) filter in
//! this crate implements the cheaper two-pass Borgefors 3-4 chamfer
//! approximation; that one trades accuracy for speed and ships an
//! integer-metric whose worst case sits a few percent above the true
//! Euclidean metric. This filter is the **exact** counterpart — it
//! produces the true squared Euclidean distance in `O(d · N)` total
//! time by running a separable 1-D transform once per dimension.
//!
//! ## Algorithm
//!
//! Felzenszwalb–Huttenlocher (2012) — "Distance Transforms of Sampled
//! Functions", *Theory of Computing* 8(19), pp. 415–428 — observes
//! that the generalised squared-distance transform
//!
//! ```text
//!   D(p) = min_q ( (p − q)² + f(q) )
//! ```
//!
//! is the **lower envelope** of a family of upward parabolas, one per
//! sample, each rooted at `q` and lifted by `f(q)`. All parabolas
//! share the same curvature, so any two intersect at exactly one
//! abscissa
//!
//! ```text
//!   s = ( (f[q] + q²) − (f[v] + v²) ) / ( 2q − 2v )
//! ```
//!
//! and the envelope is built in a single left-to-right march that
//! pushes and pops envelope candidates at most once each — `O(n)` per
//! row. A second left-to-right pass fills `D(q)` by walking the
//! envelope. The 2-D transform runs the 1-D transform once per
//! column then once per row (using the column-pass output as `f`
//! for the row pass); the result is the **exact** squared Euclidean
//! distance.
//!
//! Clean-room transcription from `docs/image/filter/distance-transform.md`
//! §2 (which itself cites the original publication by DOI link only,
//! per the *Feist* uncopyrightable-facts posture).
//!
//! ## Input / output
//!
//! Always single-plane `Gray8` input + single-plane `Gray8` output of
//! the same dimensions. Foreground pixels (those that pass the
//! `value >= threshold` test, or the inverted test when
//! [`EuclideanDistanceTransform::invert`] is true) emit distance `0`;
//! every other pixel emits the rounded Euclidean distance to the
//! nearest foreground pixel, scaled by [`scale`](Self::scale) and
//! clamped to `[0, 255]`. Large empty regions saturate to `255`.
//!
//! Useful for stroke generation, contour rings, signed-distance-field
//! glyph atlases, feathering masks, and morphology effects that need
//! a metric better than the chamfer approximation.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Exact-Euclidean Distance-Transform filter (Felzenszwalb–Huttenlocher
/// lower-envelope-of-parabolas, separable per dimension).
#[derive(Clone, Copy, Debug)]
pub struct EuclideanDistanceTransform {
    /// Foreground threshold. Pixels with `value >= threshold` count as
    /// foreground and emit distance `0`; everything else is
    /// background and contributes a `+∞` sentinel to the transform.
    pub threshold: u8,
    /// If `true`, dark pixels (`value < threshold`) become the
    /// foreground. Useful when the source is a black drawing on a
    /// white background.
    pub invert: bool,
    /// Multiplier on the **rendered** distance. `1.0` (default) shows
    /// one input-pixel of Euclidean distance per output-luminance
    /// unit; `0.5` halves that so the visible falloff fits a smaller
    /// dynamic range; `4.0` quadruples it so a small detail's
    /// neighbourhood saturates quickly. Must be finite and positive.
    pub scale: f32,
}

impl Default for EuclideanDistanceTransform {
    fn default() -> Self {
        Self {
            threshold: 128,
            invert: false,
            scale: 1.0,
        }
    }
}

impl EuclideanDistanceTransform {
    /// New filter with the given foreground `threshold` and the
    /// default `invert = false`, `scale = 1.0`.
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            invert: false,
            scale: 1.0,
        }
    }

    /// Flip the foreground/background test (dark pixels become FG).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Set the output-distance scale. Non-finite or non-positive
    /// values are silently rejected (the previous value sticks).
    pub fn with_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
        }
        self
    }
}

/// Sentinel that stands in for `+∞` in the squared-distance transform.
/// We need it large enough that `INF + q²` never overflows for any
/// reasonable image size; `1e18` is well below the `f64::MAX` ceiling
/// and well above the worst-case squared distance (the diagonal of a
/// 32-bit image is `< 2^33 < 1e10`).
///
/// Exposed `pub(crate)` so the signed-distance-field filter can seed
/// its own `f` arrays with the same sentinel and reuse the [`dt_1d`]
/// driver (both transforms come straight out of §2 of the same
/// clean-room doc).
pub(crate) const INF: f64 = 1.0e18;

/// 1-D distance transform along a single line of `n` samples.
///
/// `f[i]` is the lifted sample value (`0` at a foreground site, `INF`
/// elsewhere — or, on the second pass, the column-pass output).
/// Writes the squared-distance envelope into `d`.
///
/// Implements §2.3 of `docs/image/filter/distance-transform.md` —
/// Felzenszwalb–Huttenlocher's two-pass lower-envelope march. Both
/// passes run in `O(n)`; the inner pop loop amortises because each
/// parabola is pushed and popped at most once.
///
/// Exposed `pub(crate)` so the signed-distance-field filter can drive
/// the exact same 1-D transform on its inside / outside masks without
/// re-transcribing the envelope march.
pub(crate) fn dt_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    debug_assert_eq!(d.len(), n);
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);
    if n == 0 {
        return;
    }
    // Build the lower envelope.
    let mut k: usize = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        // Pop envelope candidates that the new parabola hides.
        loop {
            // Parabola intersection abscissa
            //   s = ((f[q] + q²) − (f[v[k]] + v[k]²)) / (2q − 2·v[k])
            // straight out of §2.2 of the clean-room doc.
            let vq = v[k];
            let fq = f[q];
            let fvk = f[vq];
            // Compute in f64; both `q` and `v[k]` are < width/height so
            // squaring as f64 is exact.
            let qf = q as f64;
            let vf = vq as f64;
            let num = (fq + qf * qf) - (fvk + vf * vf);
            let den = 2.0 * (qf - vf);
            let s = num / den;
            if s <= z[k] {
                // This parabola hides v[k]; pop it. `k > 0` guard
                // because v[0] is the leftmost sample and always
                // anchors the envelope.
                if k == 0 {
                    // Replace the anchor outright — v[0] is fully
                    // hidden by q, which now anchors the envelope at
                    // `-∞`.
                    v[0] = q;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
                continue;
            }
            // q joins the envelope to the right of v[k] at s.
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    // Second pass: fill in distances by walking the envelope.
    let mut k2: usize = 0;
    for (q, slot) in d.iter_mut().enumerate().take(n) {
        while z[k2 + 1] < q as f64 {
            k2 += 1;
        }
        let vq = v[k2];
        let dx = q as f64 - vq as f64;
        *slot = dx * dx + f[vq];
    }
}

/// 1-D distance transform that **also records the nearest feature
/// index** (the argmin) for every output sample.
///
/// Identical lower-envelope march as [`dt_1d`] (§2.3 of
/// `docs/image/filter/distance-transform.md`), but the second pass
/// additionally writes `arg[q] = v[k]` — the sample index whose
/// parabola is lowest at `q`. For a 1-D feature seed (`f[i] = 0` at a
/// feature, `INF` elsewhere) `arg[q]` is the position of the nearest
/// feature along this line, and `d[q]` the squared distance to it. The
/// envelope already tracks `v[k]` for the distance pass, so surfacing
/// the nearest index is free — no extra march.
///
/// Run separably this builds the exact 2-D **feature transform**: the
/// column pass collapses each column to its nearest in-column feature
/// (storing that feature's row), and the row pass picks the column
/// whose per-column best is globally nearest. The resulting
/// `(arg_x, arg_y)` pair is the exact nearest feature pixel — the
/// argmin counterpart to the distance the unsigned transform already
/// computes. Exposed `pub(crate)` so the Voronoi / nearest-feature
/// filters can reuse the driver without re-transcribing the march.
pub(crate) fn dt_1d_arg(
    f: &[f64],
    d: &mut [f64],
    arg: &mut [usize],
    v: &mut [usize],
    z: &mut [f64],
) {
    let n = f.len();
    debug_assert_eq!(d.len(), n);
    debug_assert_eq!(arg.len(), n);
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);
    if n == 0 {
        return;
    }
    // Build the lower envelope (identical to dt_1d).
    let mut k: usize = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        loop {
            let vq = v[k];
            let fq = f[q];
            let fvk = f[vq];
            let qf = q as f64;
            let vf = vq as f64;
            let num = (fq + qf * qf) - (fvk + vf * vf);
            let den = 2.0 * (qf - vf);
            let s = num / den;
            if s <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
                continue;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    // Second pass: distance + nearest-feature index by walking the
    // envelope. `v[k2]` is the argmin sample at `q`.
    let mut k2: usize = 0;
    for q in 0..n {
        while z[k2 + 1] < q as f64 {
            k2 += 1;
        }
        let vq = v[k2];
        let dx = q as f64 - vq as f64;
        d[q] = dx * dx + f[vq];
        arg[q] = vq;
    }
}

impl ImageFilter for EuclideanDistanceTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: EuclideanDistanceTransform requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: EuclideanDistanceTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];

        // Seed: 0.0 at foreground pixels, INF elsewhere. We hold the
        // *squared* distances throughout — the sqrt only happens at
        // render time.
        let mut f = vec![INF; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = src.data[y * src.stride + x];
                let fg = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
                if fg {
                    f[y * w + x] = 0.0;
                }
            }
        }

        // Scratch buffers reused across rows / columns.
        let n_max = w.max(h);
        let mut col_in = vec![0.0_f64; h];
        let mut col_out = vec![0.0_f64; h];
        let mut row_in = vec![0.0_f64; w];
        let mut row_out = vec![0.0_f64; w];
        let mut v_buf = vec![0_usize; n_max];
        let mut z_buf = vec![0.0_f64; n_max + 1];

        // Column pass: write back into `f` once each column is done.
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
        // Row pass: write back row-by-row.
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

        // Render to Gray8: take sqrt to convert squared to actual
        // Euclidean distance, apply scale, round, clamp.
        let mut out = vec![0u8; w * h];
        for (i, &d2) in f.iter().enumerate() {
            // Negative / sub-zero values can't happen mathematically
            // but `max(0.0)` absorbs any FP rounding chaff.
            let d = d2.max(0.0).sqrt();
            let pix = d * self.scale as f64;
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

    /// Brute-force squared Euclidean DT: for every pixel walk every
    /// other pixel; quadratic but exact. Used as the oracle for the
    /// O(W·H) F–H transform under test.
    fn brute_force_sq_euclid(mask: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; w * h];
        for ty in 0..h {
            for tx in 0..w {
                if mask[ty * w + tx] {
                    out[ty * w + tx] = 0.0;
                    continue;
                }
                let mut best = f64::INFINITY;
                for sy in 0..h {
                    for sx in 0..w {
                        if !mask[sy * w + sx] {
                            continue;
                        }
                        let dx = tx as f64 - sx as f64;
                        let dy = ty as f64 - sy as f64;
                        let d2 = dx * dx + dy * dy;
                        if d2 < best {
                            best = d2;
                        }
                    }
                }
                out[ty * w + tx] = best;
            }
        }
        out
    }

    /// Build the same `f` seed the filter uses, then run the public
    /// `apply` and pull the squared distance back out of the rendered
    /// `Gray8` (we can't — sqrt + round + clamp is lossy). Instead, we
    /// call the internal `dt_1d` driver twice the same way `apply`
    /// does and compare to `brute_force_sq_euclid`. That way the
    /// oracle test exercises the actual numeric machinery used in
    /// production.
    fn run_dt_2d(mask: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut f = vec![INF; w * h];
        for i in 0..w * h {
            if mask[i] {
                f[i] = 0.0;
            }
        }
        let n_max = w.max(h);
        let mut v_buf = vec![0_usize; n_max];
        let mut z_buf = vec![0.0_f64; n_max + 1];
        let mut col_in = vec![0.0_f64; h];
        let mut col_out = vec![0.0_f64; h];
        let mut row_in = vec![0.0_f64; w];
        let mut row_out = vec![0.0_f64; w];
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
        f
    }

    #[test]
    fn foreground_pixel_has_zero_distance() {
        // Single bright pixel; that pixel itself must render as 0.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 255 } else { 0 });
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
    }

    #[test]
    fn single_source_matches_actual_euclidean() {
        // One FG pixel at (5,5) in an 11×11 grid: the squared distance
        // at (5+dx, 5+dy) is dx²+dy², so the rendered distance is
        // round(sqrt(dx²+dy²)).
        let input = gray(11, 11, |x, y| if x == 5 && y == 5 { 255 } else { 0 });
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(11, 11))
            .unwrap();
        for y in 0..11i32 {
            for x in 0..11i32 {
                let dx = (x - 5) as f64;
                let dy = (y - 5) as f64;
                let expect = (dx * dx + dy * dy).sqrt().round() as u8;
                let got = out.planes[0].data[(y as usize) * 11 + (x as usize)];
                assert_eq!(got, expect, "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn matches_brute_force_random_mask() {
        // 13×11 deterministic pattern with a handful of FG points; the
        // F–H output must agree with the brute-force quadratic DT.
        let w = 13usize;
        let h = 11usize;
        let mut mask = vec![false; w * h];
        for &(x, y) in &[(2, 1), (10, 3), (5, 5), (1, 8), (12, 9), (7, 0), (4, 10)] {
            mask[y * w + x] = true;
        }
        let oracle = brute_force_sq_euclid(&mask, w, h);
        let got = run_dt_2d(&mask, w, h);
        for i in 0..w * h {
            let dg = (got[i] - oracle[i]).abs();
            assert!(
                dg < 1e-6,
                "F–H disagrees with brute force at i={i}: got={got_v}, oracle={or}, |Δ|={dg}",
                got_v = got[i],
                or = oracle[i]
            );
        }
    }

    #[test]
    fn matches_brute_force_solid_border() {
        // A square ring of FG along the outer border; inner pixels
        // must report distance to the nearest border. Pathological
        // input for the lower-envelope march because every column /
        // row anchor produces many envelope candidates.
        let w = 9usize;
        let h = 9usize;
        let mut mask = vec![false; w * h];
        for x in 0..w {
            mask[x] = true;
            mask[(h - 1) * w + x] = true;
        }
        for y in 0..h {
            mask[y * w] = true;
            mask[y * w + (w - 1)] = true;
        }
        let oracle = brute_force_sq_euclid(&mask, w, h);
        let got = run_dt_2d(&mask, w, h);
        for i in 0..w * h {
            let dg = (got[i] - oracle[i]).abs();
            assert!(dg < 1e-6, "ring DT mismatch at i={i}: |Δ|={dg}");
        }
    }

    #[test]
    fn matches_brute_force_diagonal_line() {
        // FG on the main diagonal — tests separable-pass coupling.
        let w = 10usize;
        let h = 10usize;
        let mut mask = vec![false; w * h];
        for i in 0..w.min(h) {
            mask[i * w + i] = true;
        }
        let oracle = brute_force_sq_euclid(&mask, w, h);
        let got = run_dt_2d(&mask, w, h);
        for i in 0..w * h {
            let dg = (got[i] - oracle[i]).abs();
            assert!(dg < 1e-6, "diagonal DT mismatch at i={i}: |Δ|={dg}");
        }
    }

    #[test]
    fn distance_grows_monotonically_along_radius() {
        // Single FG pixel; row of pixels stepping away from it should
        // produce monotonically increasing Gray8 output up to the
        // saturation point.
        let input = gray(20, 1, |x, _| if x == 0 { 255 } else { 0 });
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(20, 1))
            .unwrap();
        let row = &out.planes[0].data;
        for i in 1..row.len() {
            assert!(
                row[i] >= row[i - 1],
                "non-monotone at {i}: {} → {}",
                row[i - 1],
                row[i]
            );
        }
    }

    #[test]
    fn no_foreground_saturates() {
        // No FG at all → every pixel sits at INF → renders as 255.
        let input = gray(8, 8, |_, _| 0);
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn fully_foreground_is_zero_everywhere() {
        // Every pixel ≥ threshold → all distances are 0.
        let input = gray(8, 8, |_, _| 255);
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn invert_swaps_fg_and_bg() {
        // Dark dot on a bright field; with invert=true the dot is FG.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 0 } else { 255 });
        let out = EuclideanDistanceTransform::new(128)
            .with_invert(true)
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
        // Bright pixels further out get positive distance.
        assert!(out.planes[0].data[3 * 7 + 5] > 0);
    }

    #[test]
    fn scale_compresses_or_expands_dynamic_range() {
        // Single FG at (0, 0). Without scaling, the pixel at (3, 4)
        // sits at exact Euclidean distance 5.0. Scale 0.5 → 2/3
        // (rounds 2.5 → 3 banker / half-to-even — depending on libc,
        // accept either 2 or 3); scale 10 → 50.
        let input = gray(8, 8, |x, y| if x == 0 && y == 0 { 255 } else { 0 });
        let out_half = EuclideanDistanceTransform::default()
            .with_scale(0.5)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        let v_half = out_half.planes[0].data[4 * 8 + 3];
        assert!(v_half == 2 || v_half == 3, "scale 0.5 → {v_half}");
        let out_ten = EuclideanDistanceTransform::default()
            .with_scale(10.0)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        // 5 × 10 = 50 — well below 255 so no clamp.
        assert_eq!(out_ten.planes[0].data[4 * 8 + 3], 50);
    }

    #[test]
    fn scale_silently_ignores_bad_input() {
        // NaN and ≤ 0 inputs leave the previous scale untouched.
        let f = EuclideanDistanceTransform::default()
            .with_scale(f32::NAN)
            .with_scale(-2.0)
            .with_scale(0.0);
        assert_eq!(f.scale, 1.0);
        let f2 = EuclideanDistanceTransform::default().with_scale(2.5);
        assert_eq!(f2.scale, 2.5);
    }

    #[test]
    fn far_distances_clamp_at_255() {
        // A 400×1 row with a single FG pixel at column 0; column 399
        // sits at Euclidean distance 399 — clips to 255 in the
        // rendered Gray8.
        let w = 400u32;
        let input = gray(w, 1, |x, _| if x == 0 { 255 } else { 0 });
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(w, 1))
            .unwrap();
        assert_eq!(out.planes[0].data[399], 255);
        // The clamp boundary: distance 255 lands at exactly 255 and
        // distance 256 also lands at 255.
        assert_eq!(out.planes[0].data[255], 255);
    }

    #[test]
    fn empty_dimensions_clone_input() {
        // 0×0 → bypass, returns the input verbatim.
        let input = VideoFrame {
            pts: Some(42),
            planes: vec![VideoPlane {
                stride: 0,
                data: vec![],
            }],
        };
        let out = EuclideanDistanceTransform::default()
            .apply(&input, p_gray(0, 0))
            .unwrap();
        assert_eq!(out.pts, Some(42));
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
        let err = EuclideanDistanceTransform::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("EuclideanDistanceTransform"));
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
        let err = EuclideanDistanceTransform::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("EuclideanDistanceTransform"));
    }

    #[test]
    fn one_d_transform_matches_brute_force() {
        // Direct 1-D oracle test for dt_1d (without the 2-D wrapper).
        // Feature pixels at positions 2 and 7 in a 12-sample row;
        // every output is min over all features of (q − p)².
        let mut f = vec![INF; 12];
        f[2] = 0.0;
        f[7] = 0.0;
        let mut d = vec![0.0; 12];
        let mut v = vec![0_usize; 12];
        let mut z = vec![0.0; 13];
        dt_1d(&f, &mut d, &mut v, &mut z);
        for (q, &got) in d.iter().enumerate() {
            let oracle: f64 = [(q as i32 - 2).pow(2), (q as i32 - 7).pow(2)]
                .iter()
                .min()
                .copied()
                .unwrap() as f64;
            assert!(
                (got - oracle).abs() < 1e-9,
                "1-D mismatch at q={q}: got={got}, oracle={oracle}"
            );
        }
    }

    #[test]
    fn one_d_all_inf_stays_inf() {
        // No features anywhere; every output stays at +∞.
        let f = vec![INF; 8];
        let mut d = vec![0.0; 8];
        let mut v = vec![0_usize; 8];
        let mut z = vec![0.0; 9];
        dt_1d(&f, &mut d, &mut v, &mut z);
        for (q, &got) in d.iter().enumerate() {
            assert!(got >= INF * 0.5, "expected INF at q={q}, got {got}");
        }
    }

    #[test]
    fn one_d_anchor_replacement_when_first_sample_dominates() {
        // Stress the `k == 0` early-out branch: f[0]=INF, f[1]=0.
        // The envelope march pops the anchor at q=1 because its
        // parabola is below v[0]'s everywhere.
        let mut f = vec![0.0; 5];
        f[0] = INF;
        // f[1..] = 0 → every position from 1 onwards is a feature.
        // Position 0 should report (0 - 1)² = 1.
        let mut d = vec![0.0; 5];
        let mut v = vec![0_usize; 5];
        let mut z = vec![0.0; 6];
        dt_1d(&f, &mut d, &mut v, &mut z);
        assert!(
            (d[0] - 1.0).abs() < 1e-9,
            "expected 1.0 at q=0, got {}",
            d[0]
        );
        for (q, &got) in d.iter().enumerate().skip(1) {
            assert!((got - 0.0).abs() < 1e-9, "expected 0 at q={q}, got {got}");
        }
    }
}
