//! Distance Transform — two-pass chamfer / Borgefors family.
//!
//! Computes the distance from every "background" pixel to the nearest
//! "foreground" pixel of a binary mask. Output is a `Gray8` image where
//! each pixel's value is its (clamped) distance to the closest
//! foreground pixel — useful for stroke generation, contour
//! visualisation, dilation by distance, signed-distance fields, etc.
//!
//! Algorithm (clean-room from
//! `docs/image/filter/distance-transform.md` §3, which transcribes
//! the published math facts from Gunilla Borgefors, "Distance
//! Transformations in Digital Images", Computer Vision, Graphics, and
//! Image Processing 34 (1986), and the earlier two-scan idea from
//! A. Rosenfeld & J. L. Pfaltz, "Sequential operations in digital
//! picture processing", J. ACM 13(4), 1966):
//!
//! 1. Threshold the input to a binary mask. Foreground pixels start at
//!    distance 0; background pixels at `+∞` (a sentinel).
//! 2. Forward pass (top-left → bottom-right): for every pixel, take
//!    the min of its current value and `d(neighbour) + w(dx, dy)` for
//!    every offset `(dx, dy)` in the forward half of the chosen
//!    chamfer kernel.
//! 3. Backward pass (bottom-right → top-left): same min update with
//!    the backward half (reflection of the forward half through the
//!    centre).
//! 4. Rescale the integer chamfer metric back to pixel-units by
//!    dividing through the kernel's orthogonal-step weight `a`
//!    (`distance ≈ chamfer / a`), then apply the user `scale`.
//!
//! r220 generalises the kernel choice from the original hard-coded
//! 3-4 mask to a runtime [`ChamferKind`]:
//!
//! - [`ChamferKind::Chamfer34`] — the default 3-4 mask
//!   (orthogonal = 3, diagonal = 4, `a = 3`). ~8% Euclidean error per
//!   `docs/image/filter/distance-transform.md` §4.
//! - [`ChamferKind::Chamfer5711`] — 5-7-11 mask (orthogonal = 5,
//!   diagonal = 7, knight-move (±1, ±2) / (±2, ±1) = 11, `a = 5`).
//!   ~2% Euclidean error per the doc's summary table; the best
//!   integer chamfer approximation listed in the reference.
//! - [`ChamferKind::CityBlock`] — orthogonal-only mask weighted `1`
//!   (`a = 1`). Exact L1 / Manhattan metric (Rosenfeld & Pfaltz
//!   1966).
//! - [`ChamferKind::Chessboard`] — 3×3 mask with orthogonal and
//!   diagonal both weighted `1` (`a = 1`). Exact L∞ / Chebyshev
//!   metric.
//!
//! Operates on `Gray8` input only; the mask is computed as
//! `value >= threshold ⇒ foreground` (or `value < threshold` when
//! `invert` is `true`). Output is always `Gray8`, clamped to
//! `[0, 255]` so unreachable regions saturate to white.

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Selectable chamfer kernel for [`DistanceTransform`].
///
/// Each variant corresponds to one of the kernels enumerated in
/// `docs/image/filter/distance-transform.md` §3.2. `divisor()` returns
/// the orthogonal step-weight `a` used to rescale the integer chamfer
/// metric back into pixel units (`pixels ≈ chamfer / a`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChamferKind {
    /// 3-4 chamfer: orthogonal step = 3, diagonal step = 4. `a = 3`.
    /// Cheapest two-pass chamfer with sub-10% Euclidean error.
    #[default]
    Chamfer34,
    /// 5-7-11 chamfer: orthogonal = 5, diagonal = 7, knight-move
    /// (±1, ±2) / (±2, ±1) = 11. `a = 5`. Closer Euclidean
    /// approximation than 3-4 at the cost of a larger neighbourhood
    /// in the propagation pass.
    Chamfer5711,
    /// City-block (L1 / Manhattan) — orthogonal-only mask weighted
    /// `1`. `a = 1`. Exact for the L1 metric.
    CityBlock,
    /// Chessboard (L∞ / Chebyshev) — orthogonal and diagonal both
    /// weighted `1`. `a = 1`. Exact for the L∞ metric.
    Chessboard,
}

impl ChamferKind {
    /// Orthogonal step-weight `a` (used to rescale chamfer →
    /// pixel-units). Returned as `f32` to keep the divide-and-round
    /// path branch-free.
    pub fn divisor(self) -> f32 {
        match self {
            ChamferKind::Chamfer34 => 3.0,
            ChamferKind::Chamfer5711 => 5.0,
            ChamferKind::CityBlock => 1.0,
            ChamferKind::Chessboard => 1.0,
        }
    }
}

/// Distance-Transform filter (configurable chamfer kernel).
#[derive(Clone, Copy, Debug)]
pub struct DistanceTransform {
    /// Foreground threshold. Pixels `>= threshold` count as foreground
    /// (distance `0`); others are background.
    pub threshold: u8,
    /// If `true`, invert the mask so dark pixels become the foreground
    /// (useful when the source has a black-on-white drawing).
    pub invert: bool,
    /// Multiplier on the output distance. `1.0` (default) is "show the
    /// raw distance up to 255"; `0.5` darkens / compresses the gradient
    /// so the visible falloff fits a smaller dynamic range.
    pub scale: f32,
    /// Which chamfer kernel to propagate. Defaults to
    /// [`ChamferKind::Chamfer34`] to preserve r1 behaviour.
    pub kind: ChamferKind,
}

impl Default for DistanceTransform {
    fn default() -> Self {
        Self {
            threshold: 128,
            invert: false,
            scale: 1.0,
            kind: ChamferKind::Chamfer34,
        }
    }
}

impl DistanceTransform {
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            invert: false,
            scale: 1.0,
            kind: ChamferKind::Chamfer34,
        }
    }

    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
        }
        self
    }

    /// Select a chamfer kernel. See [`ChamferKind`] for the catalogue.
    pub fn with_kind(mut self, kind: ChamferKind) -> Self {
        self.kind = kind;
        self
    }
}

/// One propagation offset (`(dx, dy)`, weight).
type Offset = (i32, i32, i32);

/// Forward-pass offsets for the chosen kernel. The mask is the
/// "top-left half" of the neighbourhood (every neighbour the raster
/// has already visited by the time it reaches the centre pixel).
fn forward_offsets(kind: ChamferKind) -> &'static [Offset] {
    match kind {
        ChamferKind::Chamfer34 => &[
            // Row above
            (-1, -1, 4),
            (0, -1, 3),
            (1, -1, 4),
            // Same row, left side
            (-1, 0, 3),
        ],
        ChamferKind::Chamfer5711 => &[
            // Two rows above — knight-moves (±1, -2) only (centre col
            // at dx = 0 is not part of the 5-7-11 mask; see doc §3.2).
            (-1, -2, 11),
            (1, -2, 11),
            // Row above — full 3-wide chamfer fringe + knight moves
            // (±2, -1).
            (-2, -1, 11),
            (-1, -1, 7),
            (0, -1, 5),
            (1, -1, 7),
            (2, -1, 11),
            // Same row, left side.
            (-1, 0, 5),
        ],
        ChamferKind::CityBlock => &[
            // L1: orthogonal-only, weight 1.
            (0, -1, 1),
            (-1, 0, 1),
        ],
        ChamferKind::Chessboard => &[
            // L∞: orthogonal + diagonal, all weight 1.
            (-1, -1, 1),
            (0, -1, 1),
            (1, -1, 1),
            (-1, 0, 1),
        ],
    }
}

/// Backward-pass offsets — mirror of the forward set (reflected
/// through the centre).
fn backward_offsets(kind: ChamferKind) -> &'static [Offset] {
    match kind {
        ChamferKind::Chamfer34 => &[
            // Same row, right side
            (1, 0, 3),
            // Row below
            (-1, 1, 4),
            (0, 1, 3),
            (1, 1, 4),
        ],
        ChamferKind::Chamfer5711 => &[
            // Same row, right side.
            (1, 0, 5),
            // Row below — full 3-wide fringe + knight moves (±2, 1).
            (-2, 1, 11),
            (-1, 1, 7),
            (0, 1, 5),
            (1, 1, 7),
            (2, 1, 11),
            // Two rows below — knight-moves (±1, 2).
            (-1, 2, 11),
            (1, 2, 11),
        ],
        ChamferKind::CityBlock => &[(1, 0, 1), (0, 1, 1)],
        ChamferKind::Chessboard => &[(1, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1)],
    }
}

impl ImageFilter for DistanceTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: DistanceTransform requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: DistanceTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];

        // Doc §3.1 sentinel: large enough that `INF + weight` never
        // wraps even for the 5-7-11 mask's knight-move weight 11. The
        // worst-case path through an N×M grid is bounded above by
        // `11·(N+M)`, comfortably under 1_000_000 for any sane image.
        const INF: i32 = 1_000_000;
        let mut d = vec![INF; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = src.data[y * src.stride + x];
                let fg = if self.invert {
                    v < self.threshold
                } else {
                    v >= self.threshold
                };
                if fg {
                    d[y * w + x] = 0;
                }
            }
        }

        let fwd = forward_offsets(self.kind);
        let bwd = backward_offsets(self.kind);

        // Forward pass — raster top-left → bottom-right.
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut best = d[i];
                for &(dx, dy, weight) in fwd {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || (nx as usize) >= w || (ny as usize) >= h {
                        continue;
                    }
                    let nb = d[(ny as usize) * w + (nx as usize)];
                    best = best.min(nb.saturating_add(weight));
                }
                d[i] = best;
            }
        }

        // Backward pass — raster bottom-right → top-left.
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut best = d[i];
                for &(dx, dy, weight) in bwd {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || (nx as usize) >= w || (ny as usize) >= h {
                        continue;
                    }
                    let nb = d[(ny as usize) * w + (nx as usize)];
                    best = best.min(nb.saturating_add(weight));
                }
                d[i] = best;
            }
        }

        // Render to Gray8. Per the doc §3.2 the integer chamfer metric
        // is `a × euclidean`, so divide by `a` to get pixel-distance,
        // then apply `scale`.
        let divisor = self.kind.divisor();
        let mut out = vec![0u8; w * h];
        for (i, &chamfer) in d.iter().enumerate() {
            let pix = (chamfer as f32 / divisor) * self.scale;
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

    #[test]
    fn foreground_pixel_has_zero_distance() {
        // Single bright pixel at centre; that pixel itself must be 0.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
    }

    #[test]
    fn distance_increases_with_radius() {
        // Single FG pixel — distance should increase monotonically as
        // we move away (chamfer 3-4 → euclidean approximation).
        let input = gray(11, 11, |x, y| if x == 5 && y == 5 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .apply(&input, p_gray(11, 11))
            .unwrap();
        // Manhattan-1 (4 cardinals) should be exactly 1 px after divide.
        let centre = (5usize, 5usize);
        let d1 = out.planes[0].data[(centre.1) * 11 + centre.0 + 1];
        let d2 = out.planes[0].data[(centre.1) * 11 + centre.0 + 2];
        let d3 = out.planes[0].data[(centre.1) * 11 + centre.0 + 3];
        assert!(d1 < d2, "d1={d1} d2={d2}");
        assert!(d2 < d3, "d2={d2} d3={d3}");
    }

    #[test]
    fn all_background_saturates() {
        // No foreground at all → every pixel stays at INF / 3 ⇒ clamps to 255.
        let input = gray(8, 8, |_, _| 0);
        let out = DistanceTransform::default()
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 255, "no foreground should saturate");
        }
    }

    #[test]
    fn invert_picks_dark_pixels_as_foreground() {
        // Dark dot on bright background.
        let input = gray(7, 7, |x, y| if x == 3 && y == 3 { 0 } else { 255 });
        let out = DistanceTransform::new(128)
            .with_invert(true)
            .apply(&input, p_gray(7, 7))
            .unwrap();
        assert_eq!(out.planes[0].data[3 * 7 + 3], 0);
    }

    #[test]
    fn rejects_rgb() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: vec![0; 48],
            }],
        };
        let err = DistanceTransform::default()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("DistanceTransform"));
    }

    // ---- r220: ChamferKind kernel selection --------------------------

    #[test]
    fn city_block_is_l1_metric() {
        // FG at (3, 3) on an 11×11 mask. L1 distance to (5, 5) is 4.
        // Default scale = 1, divisor = 1 → output pixel value = 4.
        let input = gray(11, 11, |x, y| if x == 3 && y == 3 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .with_kind(ChamferKind::CityBlock)
            .apply(&input, p_gray(11, 11))
            .unwrap();
        assert_eq!(out.planes[0].data[5 * 11 + 5], 4);
        // Same row, +3 columns from FG → 3.
        assert_eq!(out.planes[0].data[3 * 11 + 6], 3);
        // Same column, +2 rows → 2.
        assert_eq!(out.planes[0].data[5 * 11 + 3], 2);
    }

    #[test]
    fn chessboard_is_linf_metric() {
        // FG at (3, 3). L∞ distance to (5, 6) is max(|5-3|, |6-3|) = 3.
        let input = gray(11, 11, |x, y| if x == 3 && y == 3 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .with_kind(ChamferKind::Chessboard)
            .apply(&input, p_gray(11, 11))
            .unwrap();
        assert_eq!(out.planes[0].data[6 * 11 + 5], 3);
        // Pure diagonal (5, 5) → 2.
        assert_eq!(out.planes[0].data[5 * 11 + 5], 2);
        // Same row, +4 columns → 4.
        assert_eq!(out.planes[0].data[3 * 11 + 7], 4);
    }

    #[test]
    fn chamfer_5_7_11_diagonal_is_closer_to_euclid() {
        // For a pure diagonal at offset (k, k) the true Euclidean
        // distance is k·√2 ≈ 1.414 k.
        // Chamfer 3-4 reports diagonal = 4/3 ≈ 1.333 per step.
        // Chamfer 5-7-11 reports diagonal = 7/5 = 1.4 per step — the
        // doc summary table's "better than 3-4" claim.
        // At k = 3 the 5-7-11 estimate (3·7/5 = 21/5 = 4.2 → 4) should
        // be ≥ the 3-4 estimate (3·4/3 = 4) and ≤ the true 3√2 ≈ 4.24.
        let input = gray(15, 15, |x, y| if x == 1 && y == 1 { 255 } else { 0 });
        let out34 = DistanceTransform::default()
            .with_kind(ChamferKind::Chamfer34)
            .apply(&input, p_gray(15, 15))
            .unwrap();
        let out5711 = DistanceTransform::default()
            .with_kind(ChamferKind::Chamfer5711)
            .apply(&input, p_gray(15, 15))
            .unwrap();
        let d34 = out34.planes[0].data[4 * 15 + 4]; // (1,1) → (4,4) is k=3.
        let d5711 = out5711.planes[0].data[4 * 15 + 4];
        // True diagonal distance 3·√2 ≈ 4.24; both estimates must
        // round to ≤ 4 (3-4) and the 5-7-11 estimate must be ≥ the
        // 3-4 estimate (5-7-11 estimates diagonal ≥ 3-4 estimate).
        assert!(d34 <= 4, "3-4 estimate at k=3 = {d34}, expected ≤ 4");
        assert!(d5711 >= d34, "5-7-11 ({d5711}) should be ≥ 3-4 ({d34})");
        assert!(d5711 <= 5, "5-7-11 estimate at k=3 = {d5711}, expected ≤ 5");
    }

    #[test]
    fn chamfer_5_7_11_knight_move_uses_weight_11() {
        // FG at (4, 4) on a 13×13 grid; knight-move neighbour (5, 6) is
        // distance √5 ≈ 2.236 in true Euclid. 5-7-11 reports 11/5 = 2.2,
        // which rounds to 2. The 3-4 kernel reports the diagonal+ortho
        // path (4+3)/3 ≈ 2.33 → also 2 — but the 5-7-11 should be
        // exactly 2 from the direct knight edge.
        let input = gray(13, 13, |x, y| if x == 4 && y == 4 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .with_kind(ChamferKind::Chamfer5711)
            .apply(&input, p_gray(13, 13))
            .unwrap();
        // (5, 6): rounded(11/5) = 2.
        let v = out.planes[0].data[6 * 13 + 5];
        assert_eq!(v, 2, "knight-move (5,6) from FG (4,4) under 5-7-11 = {v}");
        // (6, 5): symmetric knight move (dx=2, dy=1).
        let v2 = out.planes[0].data[5 * 13 + 6];
        assert_eq!(v2, 2, "knight-move (6,5) from FG (4,4) under 5-7-11 = {v2}");
    }

    #[test]
    fn divisor_table() {
        assert_eq!(ChamferKind::Chamfer34.divisor(), 3.0);
        assert_eq!(ChamferKind::Chamfer5711.divisor(), 5.0);
        assert_eq!(ChamferKind::CityBlock.divisor(), 1.0);
        assert_eq!(ChamferKind::Chessboard.divisor(), 1.0);
    }

    #[test]
    fn foreground_pixel_zero_under_every_kernel() {
        // The seed FG pixel is `0` by definition, no matter the kernel.
        for kind in [
            ChamferKind::Chamfer34,
            ChamferKind::Chamfer5711,
            ChamferKind::CityBlock,
            ChamferKind::Chessboard,
        ] {
            let input = gray(9, 9, |x, y| if x == 4 && y == 4 { 255 } else { 0 });
            let out = DistanceTransform::default()
                .with_kind(kind)
                .apply(&input, p_gray(9, 9))
                .unwrap();
            assert_eq!(out.planes[0].data[4 * 9 + 4], 0, "kind = {kind:?}");
        }
    }

    #[test]
    fn city_block_distance_is_separable_sum() {
        // L1 distance is |dx| + |dy| — check a non-axial sample.
        let input = gray(9, 9, |x, y| if x == 1 && y == 1 { 255 } else { 0 });
        let out = DistanceTransform::default()
            .with_kind(ChamferKind::CityBlock)
            .apply(&input, p_gray(9, 9))
            .unwrap();
        // From (1,1) → (5,3): |5-1| + |3-1| = 6.
        assert_eq!(out.planes[0].data[3 * 9 + 5], 6);
    }
}
