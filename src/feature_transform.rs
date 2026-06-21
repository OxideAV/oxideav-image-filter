//! Exact nearest-feature (Voronoi) transform.
//!
//! Where [`EuclideanDistanceTransform`](crate::EuclideanDistanceTransform)
//! answers *"how far is the nearest feature pixel?"*, the **feature
//! transform** answers *"which feature pixel is nearest?"* — it labels
//! every pixel with the coordinates of its closest foreground site. The
//! cells of equal nearest-feature form the **Voronoi diagram** of the
//! feature set, so this is also the discrete Voronoi / nearest-neighbour
//! partition of the image grid.
//!
//! ## Algorithm
//!
//! `docs/image/filter/distance-transform.md` §1 states the generalised
//! squared-Euclidean transform
//!
//! ```text
//!   D_f(p) = min_{q} ( ‖p − q‖² + f(q) )
//! ```
//!
//! and §2 (Felzenszwalb–Huttenlocher 2012) computes it separably as the
//! lower envelope of upward parabolas, one per sample. The
//! envelope march of §2.3 already identifies, for every output position
//! `q`, the **sample `v[k]` whose parabola is lowest** — i.e. the
//! `argmin` of the 1-D transform, not just its value. Carrying that
//! `argmin` through both separable passes yields the exact 2-D nearest
//! feature:
//!
//! ```text
//!   column pass: for each (x, y) record the nearest in-column feature
//!                row  fy(x, y)  and the squared vertical distance.
//!   row pass:    for each (x, y) pick the column x' whose per-column
//!                best is globally nearest; the nearest feature is then
//!                ( x', fy(x', y) ).
//! ```
//!
//! Both passes reuse the exact [`dt_1d_arg`] driver (the argmin variant
//! of the distance march), so the whole transform stays `O(d · N)` for
//! an **exact** result. The `argmin` is exact for the same reason the
//! distance is: the column pass loses no information the row pass needs,
//! because the per-column best dominates every other feature in that
//! column for any off-column query.
//!
//! Clean-room transcription from `docs/image/filter/distance-transform.md`
//! §1 (generalised DT) and §2 (the exact Euclidean transform whose 1-D
//! argmin march this filter surfaces).
//!
//! ## Input / output
//!
//! Single-plane `Gray8` input + single-plane `Gray8` output of the same
//! dimensions. The output renders the nearest-feature **cell label** —
//! a deterministic hash of the nearest feature's `(x, y)` coordinate —
//! so adjacent Voronoi cells get visibly distinct grey levels (the
//! classic crystallised / cracked-glass mosaic). Pixels whose nearest
//! feature is themselves a feature share that feature's label, so a cell
//! is flat-coloured. With no feature pixels at all the output is flat
//! `0`.

use crate::euclidean_distance_transform::{dt_1d_arg, INF};
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Result of the exact 2-D feature transform: for every pixel, the
/// coordinate of the nearest foreground site. `has_feature` is `false`
/// when the input mask was empty (every pixel's nearest site is
/// undefined; `nearest_*` is left at `0`).
pub(crate) struct FeatureTransform {
    pub nearest_x: Vec<u32>,
    pub nearest_y: Vec<u32>,
    pub has_feature: bool,
}

/// Run the exact separable nearest-feature transform on a binary mask.
///
/// `mask[y*w + x] == true` marks a feature (foreground) site. Returns
/// the per-pixel nearest feature coordinate. `w`/`h` must be non-zero
/// and `mask.len() == w*h`.
pub(crate) fn feature_transform_2d(mask: &[bool], w: usize, h: usize) -> FeatureTransform {
    debug_assert_eq!(mask.len(), w * h);
    let has_feature = mask.iter().any(|&m| m);

    // Column pass: for each column, the nearest in-column feature row +
    // its squared vertical distance. We store the squared distance in
    // `col_d` (which doubles as the row pass's `f`) and the feature row
    // in `col_fy`.
    let mut col_d = vec![INF; w * h];
    let mut col_fy = vec![0u32; w * h];

    let n_max = w.max(h);
    let mut v_buf = vec![0usize; n_max];
    let mut z_buf = vec![0.0f64; n_max + 1];
    let mut line_in = vec![0.0f64; n_max];
    let mut line_out = vec![0.0f64; n_max];
    let mut line_arg = vec![0usize; n_max];

    for x in 0..w {
        for y in 0..h {
            line_in[y] = if mask[y * w + x] { 0.0 } else { INF };
        }
        dt_1d_arg(
            &line_in[..h],
            &mut line_out[..h],
            &mut line_arg[..h],
            &mut v_buf[..h],
            &mut z_buf[..h + 1],
        );
        for y in 0..h {
            col_d[y * w + x] = line_out[y];
            col_fy[y * w + x] = line_arg[y] as u32;
        }
    }

    // Row pass: for each row, the lower envelope of the per-column
    // bests. The argmin gives the nearest column `x'`; the nearest
    // feature row is `col_fy[y, x']`.
    let mut nearest_x = vec![0u32; w * h];
    let mut nearest_y = vec![0u32; w * h];

    for y in 0..h {
        for x in 0..w {
            line_in[x] = col_d[y * w + x];
        }
        dt_1d_arg(
            &line_in[..w],
            &mut line_out[..w],
            &mut line_arg[..w],
            &mut v_buf[..w],
            &mut z_buf[..w + 1],
        );
        for x in 0..w {
            let fx = line_arg[x];
            nearest_x[y * w + x] = fx as u32;
            nearest_y[y * w + x] = col_fy[y * w + fx];
        }
    }

    FeatureTransform {
        nearest_x,
        nearest_y,
        has_feature,
    }
}

/// Deterministic cell label in `[0, 255]` from a feature coordinate.
///
/// A cheap integer hash (multiply-xor-shift) so spatially-adjacent
/// Voronoi cells land on visibly different grey levels rather than a
/// smooth ramp. Pure function of `(x, y)` — every pixel in a cell maps
/// to the same label, so cells render flat.
#[inline]
pub(crate) fn cell_label(x: u32, y: u32) -> u8 {
    // 2-D integer hash: fold the coordinate into a 32-bit word then
    // avalanche it. Constants are odd to keep the map a bijection on the
    // low bits.
    let mut hsh = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
    hsh ^= hsh >> 15;
    hsh = hsh.wrapping_mul(0x2545_F491);
    hsh ^= hsh >> 13;
    (hsh & 0xFF) as u8
}

/// Exact nearest-feature (Voronoi cell-label) filter.
#[derive(Clone, Copy, Debug)]
pub struct VoronoiTransform {
    /// Foreground threshold. Pixels with `value >= threshold` are
    /// feature sites; everything else is partitioned to its nearest
    /// site.
    pub threshold: u8,
    /// If `true`, dark pixels (`value < threshold`) become the feature
    /// sites instead (black seeds on a white field).
    pub invert: bool,
}

impl Default for VoronoiTransform {
    fn default() -> Self {
        Self {
            threshold: 128,
            invert: false,
        }
    }
}

impl VoronoiTransform {
    /// New filter with the given feature `threshold` and `invert = false`.
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            invert: false,
        }
    }

    /// Flip the feature/background test (dark pixels become features).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Build the feature mask from a Gray8 plane per `threshold` /
    /// `invert`.
    fn build_mask(&self, data: &[u8], stride: usize, w: usize, h: usize) -> Vec<bool> {
        let mut mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = data[y * stride + x];
                let fg = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
                mask[y * w + x] = fg;
            }
        }
        mask
    }
}

impl ImageFilter for VoronoiTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: VoronoiTransform requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: VoronoiTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];
        let mask = self.build_mask(&src.data, src.stride, w, h);
        let ft = feature_transform_2d(&mask, w, h);

        let mut out = vec![0u8; w * h];
        if ft.has_feature {
            for (o, (&nx, &ny)) in out
                .iter_mut()
                .zip(ft.nearest_x.iter().zip(ft.nearest_y.iter()))
            {
                *o = cell_label(nx, ny);
            }
        }
        // No feature pixels → flat 0 (out is already zeroed).

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

    /// Brute-force nearest feature: for every pixel, scan all features
    /// and pick the one with the smallest squared Euclidean distance.
    /// Ties broken by the same total order the separable transform
    /// uses is *not* guaranteed, so the oracle only checks the squared
    /// distance to the reported nearest feature equals the global
    /// minimum (any valid argmin is accepted).
    fn brute_min_distsq(mask: &[bool], w: usize, h: usize, tx: usize, ty: usize) -> f64 {
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
        best
    }

    #[test]
    fn reported_nearest_is_a_true_argmin() {
        // Scattered features: the reported nearest feature's distance
        // must equal the brute-force global minimum at every pixel, and
        // the reported coordinate must itself be a feature.
        let w = 13usize;
        let h = 11usize;
        let mut mask = vec![false; w * h];
        for &(x, y) in &[(2, 1), (10, 3), (5, 5), (1, 8), (12, 9), (7, 0)] {
            mask[y * w + x] = true;
        }
        let ft = feature_transform_2d(&mask, w, h);
        for ty in 0..h {
            for tx in 0..w {
                let i = ty * w + tx;
                let fx = ft.nearest_x[i] as usize;
                let fy = ft.nearest_y[i] as usize;
                assert!(mask[fy * w + fx], "reported nearest is not a feature");
                let reported = {
                    let dx = tx as f64 - fx as f64;
                    let dy = ty as f64 - fy as f64;
                    dx * dx + dy * dy
                };
                let oracle = brute_min_distsq(&mask, w, h, tx, ty);
                assert!(
                    (reported - oracle).abs() < 1e-9,
                    "argmin mismatch at ({tx},{ty}): reported={reported}, oracle={oracle}"
                );
            }
        }
    }

    #[test]
    fn feature_pixel_points_to_itself() {
        let w = 7usize;
        let h = 7usize;
        let mut mask = vec![false; w * h];
        mask[3 * w + 3] = true;
        mask[w + 5] = true;
        let ft = feature_transform_2d(&mask, w, h);
        assert_eq!(ft.nearest_x[3 * w + 3], 3);
        assert_eq!(ft.nearest_y[3 * w + 3], 3);
        assert_eq!(ft.nearest_x[w + 5], 5);
        assert_eq!(ft.nearest_y[w + 5], 1);
    }

    #[test]
    fn single_feature_whole_grid_points_to_it() {
        let w = 9usize;
        let h = 6usize;
        let mut mask = vec![false; w * h];
        mask[2 * w + 4] = true;
        let ft = feature_transform_2d(&mask, w, h);
        for i in 0..w * h {
            assert_eq!(ft.nearest_x[i], 4);
            assert_eq!(ft.nearest_y[i], 2);
        }
    }

    #[test]
    fn cells_are_flat_and_distinct() {
        // Two well-separated seeds: every pixel renders one of exactly
        // two labels, and the two labels differ.
        let input = gray(16, 1, |x, _| if x == 2 || x == 13 { 255 } else { 0 });
        let out = VoronoiTransform::default()
            .apply(&input, p_gray(16, 1))
            .unwrap();
        let label_a = cell_label(2, 0);
        let label_b = cell_label(13, 0);
        assert_ne!(label_a, label_b, "two seeds should hash distinctly");
        for (x, &v) in out.planes[0].data.iter().enumerate() {
            let expect = if x <= 7 { label_a } else { label_b };
            assert_eq!(v, expect, "cell label mismatch at x={x}");
        }
    }

    #[test]
    fn no_feature_is_flat_zero() {
        let input = gray(8, 8, |_, _| 0);
        let out = VoronoiTransform::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn invert_swaps_feature_polarity() {
        // Dark dot on a bright field: with invert the dot is the only
        // feature, so the whole grid points to it.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 0 } else { 255 });
        let ft_mask = VoronoiTransform::new(128).with_invert(true);
        let out = ft_mask.apply(&input, p_gray(7, 7)).unwrap();
        let only = cell_label(3, 3);
        for &v in &out.planes[0].data {
            assert_eq!(v, only);
        }
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
        let out = VoronoiTransform::default()
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
        let err = VoronoiTransform::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("VoronoiTransform"));
    }
}
