//! `-segment` / `segmentImage` (issue #12): split a picture into
//! regions of similar colour from the shape of its histograms.
//!
//! 1. Each colour channel's 256-bin histogram is smoothed by a Gaussian
//!    of `smoothing` bins, which leaves one peak per population of
//!    pixels (two peaks whose dip stays above half the lower one count
//!    as one population). The lowest point between two neighbouring
//!    peaks is a boundary, so each channel's range is cut into
//!    intervals, one per peak.
//! 2. A pixel's class is the combination of the intervals its channels
//!    fall in (a box in colour space).
//! 3. Classes holding less than `cluster_threshold` of the pixels are
//!    not regions of their own: their pixels go to the surviving class
//!    whose mean colour is nearest.
//! 4. Every pixel takes the mean colour of its class.
//!
//! The outcome does not depend on where a population sits, only on the
//! histogram having a valley between it and the next one.

use std::collections::HashMap;

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::tone_auto::gaussian;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

const BINS: usize = 256;

/// Segment a picture by histogram analysis.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// Smallest share of the pixels a class must hold, as a fraction
    /// (`0.01` = 1 %).
    pub cluster_threshold: f32,
    /// Standard deviation, in histogram bins, of the smoothing.
    pub smoothing: f32,
}

impl Segment {
    /// Segment with the given minimum class share and smoothing.
    pub fn new(cluster_threshold: f32, smoothing: f32) -> Self {
        Self {
            cluster_threshold,
            smoothing,
        }
    }
}

impl Default for Segment {
    fn default() -> Self {
        Self::new(0.01, 1.5)
    }
}

fn bin(v: f32) -> usize {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    (v * (BINS - 1) as f32 + 0.5) as usize
}

/// The interval index of every bin: boundaries at the lowest bin
/// between consecutive peaks of the smoothed histogram.
fn intervals(histogram: &[f64; BINS], sigma: f32) -> [u8; BINS] {
    let k = gaussian(if sigma.is_finite() {
        sigma.clamp(0.0, 64.0)
    } else {
        0.0
    });
    let r = k.len() as i64 / 2;
    let smooth: Vec<f64> = (0..BINS as i64)
        .map(|i| {
            k.iter()
                .enumerate()
                .map(|(j, &w)| {
                    let at = i + j as i64 - r;
                    // Nothing lies beyond the range.
                    if (0..BINS as i64).contains(&at) {
                        w as f64 * histogram[at as usize]
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect();
    // Peaks: bins higher than the bin before and not lower than the one
    // after (a plateau counts once, at its start).
    let floor = smooth.iter().cloned().fold(0.0, f64::max) * 1e-6;
    let mut peaks = Vec::new();
    for i in 0..BINS {
        let before = if i == 0 { -1.0 } else { smooth[i - 1] };
        let after = if i + 1 == BINS { -1.0 } else { smooth[i + 1] };
        if smooth[i] > floor && smooth[i] > before && smooth[i] >= after {
            peaks.push(i);
        }
    }
    // A dip is a valley only when it falls below half the lower of the
    // two peaks beside it; otherwise the two are one population and the
    // taller stands for both.
    let lowest = |a: usize, b: usize| (a..=b).map(|i| smooth[i]).fold(f64::INFINITY, f64::min);
    let mut distinct: Vec<usize> = Vec::new();
    for p in peaks {
        match distinct.last().copied() {
            Some(last) if lowest(last, p) >= 0.5 * smooth[last].min(smooth[p]) => {
                if smooth[p] > smooth[last] {
                    *distinct.last_mut().expect("non-empty") = p;
                }
            }
            _ => distinct.push(p),
        }
    }
    let peaks = distinct;
    let mut out = [0u8; BINS];
    let mut start = 0usize;
    for (n, pair) in peaks.windows(2).enumerate() {
        // The first lowest bin between the two peaks ends interval `n`.
        let cut = (pair[0]..=pair[1])
            .fold((f64::INFINITY, pair[0]), |b, i| {
                if smooth[i] < b.0 { (smooth[i], i) } else { b }
            })
            .1;
        out[start..=cut].fill(n.min(255) as u8);
        start = cut + 1;
    }
    out[start..].fill((peaks.len().saturating_sub(1)).min(255) as u8);
    out
}

impl PixelFilter for Segment {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let (ch, cc) = (img.channels(), img.color_channels());
            let pixels = img.samples().len() / ch.max(1);
            if pixels == 0 {
                return Ok(img);
            }
            let mut maps = Vec::with_capacity(cc);
            for c in 0..cc {
                let mut h = [0.0f64; BINS];
                img.samples()
                    .iter()
                    .skip(c)
                    .step_by(ch)
                    .for_each(|&v| h[bin(v)] += 1.0);
                maps.push(intervals(&h, self.smoothing));
            }
            let class_of = |px: &[f32]| -> u32 {
                px[..cc]
                    .iter()
                    .zip(&maps)
                    .fold(0u32, |k, (&v, m)| (k << 8) | m[bin(v)] as u32)
            };
            // Class -> (colour sum, count), then means in key order.
            let mut classes: HashMap<u32, ([f64; 3], f64)> = HashMap::new();
            for px in img.samples().chunks_exact(ch) {
                let e = classes.entry(class_of(px)).or_insert(([0.0; 3], 0.0));
                for (s, &v) in e.0.iter_mut().zip(&px[..cc]) {
                    *s += if v.is_finite() { v as f64 } else { 0.0 };
                }
                e.1 += 1.0;
            }
            let mut list: Vec<(u32, [f32; 3], f64)> = classes
                .into_iter()
                .map(|(k, (s, n))| (k, s.map(|v| (v / n) as f32), n))
                .collect();
            list.sort_by_key(|e| e.0);
            let need = if self.cluster_threshold.is_finite() {
                self.cluster_threshold.clamp(0.0, 1.0) as f64
            } else {
                0.0
            } * pixels as f64;
            let mut kept: Vec<usize> = (0..list.len()).filter(|&i| list[i].2 >= need).collect();
            if kept.is_empty() {
                // No class is large enough: the largest stands alone.
                let largest =
                    (0..list.len()).fold(0, |b, i| if list[i].2 > list[b].2 { i } else { b });
                kept.push(largest);
            }
            // Small classes join the nearest kept one; the kept means are
            // recomputed over everything they absorbed.
            let target: Vec<usize> = (0..list.len())
                .map(|i| {
                    if list[i].2 >= need && kept.contains(&i) {
                        return i;
                    }
                    *kept
                        .iter()
                        .min_by(|&&a, &&b| {
                            let d = |j: usize| -> f32 {
                                (0..cc).map(|c| (list[j].1[c] - list[i].1[c]).powi(2)).sum()
                            };
                            d(a).total_cmp(&d(b))
                        })
                        .expect("at least one class is kept")
                })
                .collect();
            let mut merged = vec![([0.0f64; 3], 0.0f64); list.len()];
            for (i, &t) in target.iter().enumerate() {
                for c in 0..cc {
                    merged[t].0[c] += list[i].1[c] as f64 * list[i].2;
                }
                merged[t].1 += list[i].2;
            }
            let colour: HashMap<u32, [f32; 3]> = list
                .iter()
                .zip(&target)
                .map(|(e, &t)| (e.0, merged[t].0.map(|v| (v / merged[t].1) as f32)))
                .collect();
            for px in img.samples_mut().chunks_exact_mut(ch) {
                let c = colour[&class_of(px)];
                px[..cc].copy_from_slice(&c[..cc]);
            }
            Ok(img)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    use crate::deep::FloatImage;

    fn run(f: &Segment, img: &FloatImage) -> FloatImage {
        let out = f
            .apply(&img.clone().into_frame(None), img.params())
            .unwrap();
        FloatImage::from_frame(&out, img.params()).unwrap()
    }

    #[test]
    fn two_populations_become_two_flat_regions() {
        // Left half around 0.25, right half around 0.7, each jittered.
        let data: Vec<f32> = (0..32 * 32)
            .map(|i| {
                let (x, y) = (i % 32, i / 32);
                let jitter = ((x * 7 + y * 13) % 9) as f32 / 255.0;
                if x < 16 { 0.25 + jitter } else { 0.7 - jitter }
            })
            .collect();
        let img = FloatImage::from_samples(32, 32, 1, data).unwrap();
        let out = run(&Segment::default(), &img);
        let mut values: Vec<u32> = out.samples().iter().map(|v| v.to_bits()).collect();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), 2, "two regions");
        assert!((out.pixel(3, 3)[0] - (0.25 + 4.0 / 255.0)).abs() < 0.01);
        assert!((out.pixel(30, 3)[0] - (0.7 - 4.0 / 255.0)).abs() < 0.01);
    }

    #[test]
    fn a_small_population_joins_its_nearest_neighbour() {
        // 2 % of the pixels sit at 0.3, between the modes at 0.2 and 0.9.
        let data: Vec<f32> = (0..2500)
            .map(|i| {
                if i < 50 {
                    0.3
                } else if i < 1300 {
                    0.2
                } else {
                    0.9
                }
            })
            .collect();
        let img = FloatImage::from_samples(50, 50, 1, data).unwrap();
        // Sharp histogram: three classes survive a 1 % threshold…
        let out = run(&Segment::new(0.01, 0.5), &img);
        assert_eq!(out.pixel(0, 0)[0], 0.3);
        // …and the small one is absorbed at 5 %, pulling the mean a little.
        let out = run(&Segment::new(0.05, 0.5), &img);
        let merged = out.pixel(0, 0)[0];
        assert_eq!(merged, out.pixel(10, 10)[0]);
        assert!((merged - (0.3 * 50.0 + 0.2 * 1250.0) / 1300.0).abs() < 1e-4);
        assert_eq!(out.pixel(49, 49)[0], 0.9);
    }

    #[test]
    fn interval_boundaries_sit_in_the_valleys() {
        let mut h = [0.0f64; BINS];
        h[40] = 100.0;
        h[120] = 80.0;
        h[200] = 60.0;
        let m = intervals(&h, 2.0);
        assert_eq!((m[40], m[120], m[200]), (0, 1, 2));
        assert_eq!(m[0], 0);
        assert_eq!(m[255], 2);
        assert!(m[70] == 0 || m[70] == 1);
        // A flat histogram is one interval.
        assert!(intervals(&[1.0; BINS], 2.0).iter().all(|&v| v == 0));
        assert!(intervals(&[0.0; BINS], 2.0).iter().all(|&v| v == 0));
    }
}
