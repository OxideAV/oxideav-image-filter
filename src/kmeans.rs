//! `-kmeans` / `kmeansImage` (issue #12): reduce a picture to `k` colours
//! by Lloyd's algorithm in RGB.
//!
//! From seed colours (`kmeans:seed-colors`) each iteration assigns every
//! pixel to its nearest centroid (Euclidean, normalised RGB) and moves
//! each centroid to the mean of its pixels, until no centroid moves more
//! than the tolerance or the iteration limit is reached; every pixel is
//! then replaced by its centroid. From the same seeds the converged
//! result matches `magick` 7.1.2 exactly in one measured case (three
//! seeds on `rose:`) and to within 1 % of the pixels in another (four
//! seeds: `magick`'s assignment drifts from plain Lloyd's on a few
//! boundary pixels, not pinned). Without seeds
//! this crate seeds with the colours at the `(m + ½) / k` quantiles of
//! the pixels sorted by Rec. 709 luma — its own rule, as `magick`'s
//! default seeding is not documented in the staged material.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// Lloyd's k-means colour reduction.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Kmeans {
    /// Number of colours.
    pub k: usize,
    /// Seed colours (normalised RGB); `k` of them when given.
    pub seeds: Option<Vec<[f32; 3]>>,
    /// Iteration limit (default 300).
    pub max_iterations: u32,
    /// Largest centroid move that still counts as convergence.
    pub tolerance: f32,
}

impl Kmeans {
    /// `k` colours (at least 1).
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            seeds: None,
            max_iterations: 300,
            tolerance: 1e-6,
        }
    }

    /// Start from these colours (`k` becomes their count).
    pub fn with_seeds(mut self, seeds: Vec<[f32; 3]>) -> Self {
        if !seeds.is_empty() {
            self.k = seeds.len();
            self.seeds = Some(seeds);
        }
        self
    }

    /// `kmeans:max-iterations`.
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    /// The converged centroids for the colour samples `px` (RGB triples).
    pub fn centroids(&self, px: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut c: Vec<[f64; 3]> = match &self.seeds {
            Some(s) => s.iter().map(|v| v.map(f64::from)).collect(),
            None => {
                let mut sorted: Vec<[f32; 3]> = px.to_vec();
                sorted.sort_by(|a, b| luma(a).total_cmp(&luma(b)));
                (0..self.k)
                    .map(|m| {
                        let idx =
                            (((m as f64 + 0.5) / self.k as f64) * sorted.len() as f64) as usize;
                        sorted[idx.min(sorted.len().saturating_sub(1))].map(f64::from)
                    })
                    .collect()
            }
        };
        if px.is_empty() {
            return c.iter().map(|v| v.map(|x| x as f32)).collect();
        }
        for _ in 0..self.max_iterations {
            let mut sum = vec![[0.0f64; 3]; c.len()];
            let mut n = vec![0u64; c.len()];
            for p in px {
                let k = nearest(&c, p);
                for ch in 0..3 {
                    sum[k][ch] += p[ch] as f64;
                }
                n[k] += 1;
            }
            let mut shift = 0.0f64;
            for k in 0..c.len() {
                if n[k] == 0 {
                    continue;
                }
                let m = sum[k].map(|s| s / n[k] as f64);
                shift = shift.max(
                    (0..3)
                        .map(|ch| (m[ch] - c[k][ch]).abs())
                        .fold(0.0, f64::max),
                );
                c[k] = m;
            }
            if shift <= self.tolerance as f64 {
                break;
            }
        }
        c.iter().map(|v| v.map(|x| x as f32)).collect()
    }
}

fn luma(p: &[f32; 3]) -> f32 {
    0.212_656 * p[0] + 0.715_158 * p[1] + 0.072_186 * p[2]
}

fn nearest(c: &[[f64; 3]], p: &[f32; 3]) -> usize {
    let mut best = (f64::INFINITY, 0);
    for (k, q) in c.iter().enumerate() {
        let d: f64 = (0..3).map(|ch| (p[ch] as f64 - q[ch]).powi(2)).sum();
        if d < best.0 {
            best = (d, k);
        }
    }
    best.1
}

impl PixelFilter for Kmeans {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let float = deep::is_float(params.format);
        let mut img = if float {
            deep::FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        let (ch, cc) = (img.channels(), img.color_channels());
        let rgb: Vec<[f32; 3]> = img
            .samples()
            .chunks_exact(ch)
            .map(|p| {
                if cc >= 3 {
                    [p[0], p[1], p[2]]
                } else {
                    [p[0]; 3]
                }
            })
            .collect();
        let c64: Vec<[f64; 3]> = self
            .centroids(&rgb)
            .iter()
            .map(|v| v.map(f64::from))
            .collect();
        for (px, p) in img.samples_mut().chunks_exact_mut(ch).zip(&rgb) {
            let k = &c64[nearest(&c64, p)];
            if cc >= 3 {
                for c in 0..3 {
                    px[c] = k[c] as f32;
                }
            } else {
                px[0] = k[0] as f32;
            }
        }
        Ok(if float {
            img.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&img);
            f.pts = input.pts;
            f
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_converge_to_their_means() {
        let px = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0],
        ];
        let c = Kmeans::new(2)
            .with_seeds(vec![[0.2; 3], [0.7; 3]])
            .centroids(&px);
        assert!((c[0][0] - 0.05).abs() < 1e-6 && (c[1][0] - 0.95).abs() < 1e-6);
        let d = Kmeans::new(2).centroids(&px);
        assert!((d[0][0] - 0.05).abs() < 1e-6 && (d[1][0] - 0.95).abs() < 1e-6);
    }
}
