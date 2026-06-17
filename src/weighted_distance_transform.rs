//! Weighted (generalised) distance transform.
//!
//! The exact-Euclidean filter
//! [`EuclideanDistanceTransform`](crate::EuclideanDistanceTransform)
//! is the *binary-mask* specialisation of a strictly more general
//! algorithm. `docs/image/filter/distance-transform.md` §1 states the
//! transform for an **arbitrary** sampled function
//! `f: G → ℝ ∪ {+∞}`:
//!
//! ```text
//!   D_f(p) = min_q ( ‖p − q‖² + f(q) )
//! ```
//!
//! and notes that "the generalized form (arbitrary `f`) is what makes
//! the algorithm reusable beyond binary masks." The binary
//! distance-transform filters in this crate all seed `f(q)` with the
//! two-valued mask `{0 at foreground, +∞ elsewhere}`. This filter
//! realises the **continuous** seed instead: every pixel is a soft
//! site whose `f(q)` cost is derived from its own intensity, so a
//! bright (or, inverted, a dark) pixel is "cheap" to reach and a faint
//! one is "expensive". The output at `p` is the smallest cost over the
//! whole image of *travelling* squared-Euclidean distance to a site
//! `q` *plus* paying that site's intrinsic cost `f(q)`.
//!
//! With the cost weight at its extreme this reproduces the binary
//! transform (every above-threshold pixel costs `0`, every other costs
//! `+∞`); with a finite weight the field is a smooth blend of "how far"
//! and "how faint", which is exactly the soft-seed behaviour the §1
//! generalisation is for (cost-weighted feature maps, intensity-aware
//! feathering, grey-weighted morphology).
//!
//! ## Algorithm
//!
//! The driver is the identical Felzenszwalb–Huttenlocher separable
//! lower-envelope-of-parabolas march of §2 — see
//! [`EuclideanDistanceTransform`](crate::EuclideanDistanceTransform)
//! for the full derivation. The **only** difference is the seed: where
//! the binary filter writes `0.0` or [`INF`], this filter writes a
//! finite per-pixel cost
//!
//! ```text
//!   f(q) = ( weight · c(q) )²
//! ```
//!
//! where `c(q) ∈ [0, 1]` is the normalised "faintness" of the source
//! pixel (`0` for a pixel that should behave like a free foreground
//! seed, `1` for one that should behave like distant background). The
//! cost is squared so it shares the units of the squared-Euclidean
//! travel term `‖p − q‖²` — i.e. a `weight` of `1.0` makes a
//! fully-faint pixel cost the same as travelling one pixel of distance.
//! Because the driver already minimises `‖p − q‖² + f(q)` over a
//! free-form `f`, no change to the march itself is needed.
//!
//! Clean-room transcription from `docs/image/filter/distance-transform.md`
//! §1 (generalised `D_f`) and §2 (the separable F–H driver), per the
//! *Feist* uncopyrightable-facts posture.
//!
//! ## Input / output
//!
//! Single-plane `Gray8` input + single-plane `Gray8` output of the same
//! dimensions. The rendered value is `sqrt(D_f) · scale`, rounded and
//! clamped to `[0, 255]`.

use crate::euclidean_distance_transform::{dt_1d, INF};
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Weighted (generalised) distance-transform filter — the continuous-
/// seed form of `D_f(p) = min_q(‖p − q‖² + f(q))`.
#[derive(Clone, Copy, Debug)]
pub struct WeightedDistanceTransform {
    /// Cost weight on the per-pixel seed. The seed cost is
    /// `f(q) = (weight · c(q))²` with `c(q) ∈ [0, 1]` the normalised
    /// faintness. `0.0` makes every pixel a free seed (output is all
    /// black); larger weights make faint pixels progressively more
    /// "expensive" so the field grows toward the binary transform.
    /// Must be finite and `>= 0`.
    pub weight: f32,
    /// If `true`, *dark* pixels are the cheap seeds (`c = value/255`);
    /// otherwise *bright* pixels are (`c = 1 − value/255`). Mirrors the
    /// `invert` flag on the binary transforms.
    pub invert: bool,
    /// Multiplier on the **rendered** distance, identical in meaning to
    /// [`EuclideanDistanceTransform::scale`](crate::EuclideanDistanceTransform::scale).
    /// Must be finite and positive.
    pub scale: f32,
}

impl Default for WeightedDistanceTransform {
    fn default() -> Self {
        Self {
            weight: 16.0,
            invert: false,
            scale: 1.0,
        }
    }
}

impl WeightedDistanceTransform {
    /// New filter with the given seed-cost `weight` and the defaults
    /// `invert = false`, `scale = 1.0`.
    pub fn new(weight: f32) -> Self {
        let mut f = Self::default();
        if weight.is_finite() && weight >= 0.0 {
            f.weight = weight;
        }
        f
    }

    /// Flip which pixels are the cheap seeds (dark vs bright).
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

    /// Set the seed-cost weight. Non-finite or negative values are
    /// silently rejected (the previous value sticks).
    pub fn with_weight(mut self, weight: f32) -> Self {
        if weight.is_finite() && weight >= 0.0 {
            self.weight = weight;
        }
        self
    }
}

impl ImageFilter for WeightedDistanceTransform {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !matches!(params.format, PixelFormat::Gray8) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: WeightedDistanceTransform requires Gray8, got {:?}",
                params.format
            )));
        }
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: WeightedDistanceTransform requires a non-empty input plane",
            ));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }
        let src = &input.planes[0];

        // Seed every pixel with its continuous cost f(q) = (weight·c)².
        // c ∈ [0, 1] is the normalised faintness; a cost of 0 behaves
        // like a free foreground site, a large cost like distant
        // background. We hold *squared* distances throughout.
        let weight = self.weight as f64;
        let mut f = vec![INF; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = src.data[y * src.stride + x] as f64 / 255.0;
                // Faintness: 0 = cheap seed, 1 = most expensive.
                let c = if self.invert { v } else { 1.0 - v };
                let cost = weight * c;
                // Square so the cost shares units with ‖p − q‖²; a
                // huge weight pushes faint pixels toward the binary
                // +∞ sentinel without ever exceeding it.
                let sq = (cost * cost).min(INF);
                f[y * w + x] = sq;
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

        // Render to Gray8: sqrt the squared cost, scale, round, clamp.
        let mut out = vec![0u8; w * h];
        for (i, &d2) in f.iter().enumerate() {
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

    /// Brute-force generalised DT oracle: for every target pixel walk
    /// every source pixel and take `min(‖p − q‖² + f(q))`. Quadratic
    /// but exact; mirrors the §1 definition directly.
    fn brute_force(seed: &[f64], w: usize, h: usize) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; w * h];
        for ty in 0..h {
            for tx in 0..w {
                let mut best = f64::INFINITY;
                for sy in 0..h {
                    for sx in 0..w {
                        let dx = tx as f64 - sx as f64;
                        let dy = ty as f64 - sy as f64;
                        let d2 = dx * dx + dy * dy + seed[sy * w + sx];
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

    /// Build the production seed the same way `apply` does, run the
    /// separable driver, and compare the squared-cost field to the
    /// brute-force oracle. Exercises the actual numeric machinery.
    fn run_2d(values: &[u8], w: usize, h: usize, weight: f64, invert: bool) -> Vec<f64> {
        let mut f = vec![0.0_f64; w * h];
        for i in 0..w * h {
            let v = values[i] as f64 / 255.0;
            let c = if invert { v } else { 1.0 - v };
            let cost = weight * c;
            f[i] = (cost * cost).min(INF);
        }
        let seed = f.clone();

        let n_max = w.max(h);
        let mut col_in = vec![0.0_f64; h];
        let mut col_out = vec![0.0_f64; h];
        let mut row_in = vec![0.0_f64; w];
        let mut row_out = vec![0.0_f64; w];
        let mut v_buf = vec![0_usize; n_max];
        let mut z_buf = vec![0.0_f64; n_max + 1];

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

        let oracle = brute_force(&seed, w, h);
        for i in 0..w * h {
            assert!(
                (f[i] - oracle[i]).abs() < 1e-6,
                "pixel {i}: separable {} vs brute {}",
                f[i],
                oracle[i]
            );
        }
        f
    }

    #[test]
    fn matches_brute_force_oracle_uniform_grey() {
        // A non-binary field: a diagonal grey ramp. The continuous seed
        // means no pixel is a free site, so the separable driver must
        // agree with the §1 brute-force minimisation everywhere.
        let w = 9;
        let h = 7;
        let mut vals = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                vals[y * w + x] = (((x + y) * 17) % 256) as u8;
            }
        }
        run_2d(&vals, w, h, 8.0, false);
    }

    #[test]
    fn matches_brute_force_oracle_single_bright_seed() {
        // One bright pixel in a dark field; everything else is faint
        // (expensive). The cheapest route is "travel to the one bright
        // pixel", so the field is the plain Euclidean distance to it
        // plus zero seed cost there.
        let w = 8;
        let h = 8;
        let mut vals = vec![0u8; w * h];
        vals[3 * w + 5] = 255;
        run_2d(&vals, w, h, 64.0, false);
    }

    #[test]
    fn weight_zero_makes_every_pixel_a_free_seed() {
        // weight = 0 ⇒ f(q) = 0 everywhere ⇒ D_f(p) = 0 everywhere
        // (each pixel is its own zero-cost site at distance 0).
        let frame = gray(6, 5, |x, y| ((x * 40 + y * 9) % 256) as u8);
        let out = WeightedDistanceTransform::new(0.0)
            .apply(&frame, p_gray(6, 5))
            .unwrap();
        assert!(out.planes[0].data.iter().all(|&p| p == 0));
    }

    #[test]
    fn bright_seed_grows_with_distance() {
        // Single bright pixel, large weight: rendered distance must
        // increase monotonically as we step away from the seed.
        let w = 11u32;
        let h = 1u32;
        let mut data = vec![0u8; (w * h) as usize];
        data[0] = 255; // seed at the left edge
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        };
        let out = WeightedDistanceTransform::new(1000.0)
            .apply(&frame, p_gray(w, h))
            .unwrap();
        let row = &out.planes[0].data;
        assert_eq!(row[0], 0, "seed pixel is zero-cost");
        for x in 1..w as usize {
            assert!(
                row[x] >= row[x - 1],
                "distance must be non-decreasing away from the seed: {row:?}"
            );
        }
    }

    #[test]
    fn invert_swaps_cheap_seed_polarity() {
        // A single dark pixel in a bright field. Without invert, bright
        // pixels are the cheap seeds, so the dark pixel is the *only*
        // costly spot and the field is near-zero everywhere. With
        // invert, the dark pixel becomes the cheap seed and the field
        // grows away from it.
        let w = 9u32;
        let h = 9u32;
        let frame = gray(w, h, |x, y| if x == 4 && y == 4 { 0 } else { 255 });

        let plain = WeightedDistanceTransform::new(40.0)
            .apply(&frame, p_gray(w, h))
            .unwrap();
        // Bright pixels are free seeds ⇒ everything but the dark pixel
        // is reachable at ~0 cost.
        let bright_corner = plain.planes[0].data[0];
        assert_eq!(bright_corner, 0);

        let inverted = WeightedDistanceTransform::new(40.0)
            .with_invert(true)
            .apply(&frame, p_gray(w, h))
            .unwrap();
        // Dark pixel is now the cheap seed; the centre is zero and the
        // corners are the farthest, hence the largest cost.
        let centre = inverted.planes[0].data[(4 * w + 4) as usize];
        let corner = inverted.planes[0].data[0];
        assert_eq!(centre, 0, "the dark seed is zero-cost when inverted");
        assert!(corner > centre, "cost grows away from the inverted seed");
    }

    #[test]
    fn scale_amplifies_rendered_distance() {
        let frame = gray(12, 1, |x, _| if x == 0 { 255 } else { 0 });
        let one = WeightedDistanceTransform::new(1000.0)
            .with_scale(1.0)
            .apply(&frame, p_gray(12, 1))
            .unwrap();
        let two = WeightedDistanceTransform::new(1000.0)
            .with_scale(2.0)
            .apply(&frame, p_gray(12, 1))
            .unwrap();
        // Where the ×1 render is unsaturated, the ×2 render is at least
        // as large (it saturates sooner).
        for x in 0..12usize {
            assert!(two.planes[0].data[x] >= one.planes[0].data[x]);
        }
        // And strictly larger somewhere before saturation.
        assert!(two.planes[0].data[1] > one.planes[0].data[1]);
    }

    #[test]
    fn rejects_non_gray8() {
        let frame = gray(2, 2, |_, _| 0);
        let params = VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: 2,
            height: 2,
        };
        assert!(WeightedDistanceTransform::new(8.0)
            .apply(&frame, params)
            .is_err());
    }

    #[test]
    fn empty_dimensions_passthrough() {
        let frame = VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane {
                stride: 0,
                data: vec![],
            }],
        };
        let out = WeightedDistanceTransform::new(8.0)
            .apply(&frame, p_gray(0, 4))
            .unwrap();
        assert_eq!(out.pts, Some(7));
    }

    #[test]
    fn builders_reject_bad_values() {
        let base = WeightedDistanceTransform::default();
        assert_eq!(base.with_weight(f32::NAN).weight, base.weight);
        assert_eq!(base.with_weight(-1.0).weight, base.weight);
        assert_eq!(base.with_scale(0.0).scale, base.scale);
        assert_eq!(base.with_scale(-2.0).scale, base.scale);
        assert_eq!(
            WeightedDistanceTransform::new(f32::INFINITY).weight,
            base.weight
        );
    }
}
