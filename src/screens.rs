//! Generated threshold maps for [`ThresholdMap`]: clustered-dot screens
//! at any period and angle, and blue-noise masks (issue #15).
//!
//! # Clustered-dot screens
//!
//! A halftone screen is a lattice of cells; inside a cell the pixels
//! switch on in the order a *spot function* ranks them, so the dot
//! grows from the cell centre. For the screen to tile, the lattice must
//! be made of whole pixels: the cell edge is an integer vector
//! `u = (a, b)` and its quarter turn `v = (−b, a)`. A pixel `p = (x, y)`
//! then has cell coordinates
//!
//! ```text
//! s = (a·x + b·y) / D,   t = (−b·x + a·y) / D,   D = a² + b²
//! ```
//!
//! whose fractional parts depend only on the residues of the two
//! numerators modulo `D` — exact integers, so two pixels in the same
//! place of different cells get the same threshold. The screen has
//! period `√D` pixels at angle `atan2(b, a)`, `D` distinct thresholds,
//! and repeats every `D / gcd(a, b)` pixels along both axes (that
//! multiple of `u` and `v` is axis-aligned). A requested period and
//! angle are snapped to the integer pair that reproduces them best
//! among those whose tile is at most 256 pixels a side.
//!
//! # Blue noise
//!
//! A blue-noise mask is a threshold map whose every level set is a
//! uniformly spread point pattern — no clumps, no gaps, no lattice. It
//! is built by the void-and-cluster construction on a torus: every
//! point of a binary pattern spreads a Gaussian bump, and the summed
//! field measures crowding. The *tightest cluster* is the set point
//! where the field is largest, the *largest void* the unset point where
//! it is smallest.
//!
//! 1. Start from a tenth of the cells set at pseudo-random positions and
//!    move the tightest cluster into the largest void until that would
//!    undo itself — a uniformly spread prototype.
//! 2. Rank the prototype's points from last to first by removing the
//!    tightest cluster repeatedly.
//! 3. Rank every other cell by filling the largest void repeatedly. (On
//!    a torus the field of the unset points is a constant minus the
//!    field of the set ones, so "largest void of the set" and "tightest
//!    cluster of the unset" are the same cell; one rule serves the whole
//!    upper range.)
//!
//! The Gaussian is `c^(d²)` with `c = exp(−1 / (2σ²))`, `σ = 1.5`,
//! tabulated by repeated multiplication, and the field is updated in a
//! fixed order, so the mask is the same on every platform. Masks are
//! built on first use and kept for the life of the process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::ordered_dither::ThresholdMap;

/// The spot function of a clustered-dot screen: the order in which the
/// pixels of one cell switch on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpotShape {
    /// `cos 2πs + cos 2πt`: round dots that meet as a checkerboard at
    /// half tone and leave round holes above it.
    #[default]
    Round,
    /// `−(|s| + |t|)` on the centred cell: diamonds.
    Diamond,
    /// `cos 2πs`: parallel lines across the screen angle.
    Line,
    /// `−max(|s|, |t|)` on the centred cell: squares.
    Square,
}

impl SpotShape {
    /// A shape by name (`round` / `dot`, `diamond`, `line`, `square`).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "round" | "dot" | "circle" => Self::Round,
            "diamond" => Self::Diamond,
            "line" | "lines" => Self::Line,
            "square" => Self::Square,
            _ => return None,
        })
    }

    /// The spot value at cell coordinates `(s, t)` in `[0, 1)²`; larger
    /// switches on earlier.
    fn value(self, s: f64, t: f64) -> f64 {
        let centred = |v: f64| if v >= 0.5 { v - 1.0 } else { v };
        let tau = std::f64::consts::TAU;
        match self {
            Self::Round => (tau * s).cos() + (tau * t).cos(),
            Self::Diamond => -(centred(s).abs() + centred(t).abs()),
            Self::Line => (tau * s).cos(),
            Self::Square => -centred(s).abs().max(centred(t).abs()),
        }
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The integer cell vector `(a, b)` whose screen is closest to `period`
/// pixels at `angle` degrees, among those tiling within 256 pixels.
fn snap_cell(period: f64, angle_degrees: f64) -> (i64, i64) {
    let period = if period.is_finite() {
        period.clamp(1.0, 64.0)
    } else {
        4.0
    };
    let angle = if angle_degrees.is_finite() {
        angle_degrees.to_radians()
    } else {
        0.0
    };
    let (tx, ty) = (period * angle.cos(), period * angle.sin());
    let mut best = (1i64, 0i64);
    let mut best_cost = f64::INFINITY;
    for a in -64i64..=64 {
        for b in -64i64..=64 {
            let d = a * a + b * b;
            if d == 0 || d / gcd(a, b) > 256 {
                continue;
            }
            let cost = (a as f64 - tx).powi(2) + (b as f64 - ty).powi(2);
            if cost < best_cost {
                best_cost = cost;
                best = (a, b);
            }
        }
    }
    best
}

impl ThresholdMap {
    /// A clustered-dot (halftone) screen of about `period` pixels per
    /// cell (1 … 64) at about `angle_degrees`, with the given spot
    /// shape. The pair is snapped to the nearest screen that tiles
    /// exactly (see the [module docs](crate::screens)); the map has one
    /// threshold per pixel of a cell, so a period of `p` gives about
    /// `p²` tones.
    pub fn clustered_dot(period: f64, angle_degrees: f64, shape: SpotShape) -> Self {
        let (a, b) = snap_cell(period, angle_degrees);
        let d = a * a + b * b;
        let side = d / gcd(a, b);
        // Rank the `d` residue classes by spot value (ties by residue,
        // so the order is total and the same everywhere).
        // The spot value is held as a rounded fixed-point key so two
        // values that are equal in exact arithmetic compare equal.
        let mut classes: Vec<(i64, i64, i64)> = Vec::with_capacity(d as usize);
        let mut class_of: HashMap<(i64, i64), u32> = HashMap::with_capacity(d as usize);
        let residues =
            |x: i64, y: i64| ((a * x + b * y).rem_euclid(d), (a * y - b * x).rem_euclid(d));
        for y in 0..side {
            for x in 0..side {
                let (ra, rb) = residues(x, y);
                if let std::collections::hash_map::Entry::Vacant(v) = class_of.entry((ra, rb)) {
                    v.insert(0);
                    let v = shape.value(ra as f64 / d as f64, rb as f64 / d as f64);
                    classes.push(((v * (1u64 << 30) as f64).round() as i64, ra, rb));
                }
            }
        }
        classes.sort_by(|p, q| q.0.cmp(&p.0).then(p.1.cmp(&q.1)).then(p.2.cmp(&q.2)));
        for (rank, &(_, ra, rb)) in classes.iter().enumerate() {
            class_of.insert((ra, rb), rank as u32);
        }
        let mut cells = Vec::with_capacity((side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                cells.push(class_of[&residues(x, y)]);
            }
        }
        Self {
            width: side as u32,
            height: side as u32,
            divisor: classes.len() as u32,
            cells,
        }
    }

    /// A `size × size` blue-noise mask (`size` clamped to 8 … 128) with
    /// `size²` distinct thresholds, built by void-and-cluster (see the
    /// [module docs](crate::screens)) on first use and cached. The same
    /// mask on every platform and in every run.
    pub fn blue_noise(size: u32) -> Self {
        let size = size.clamp(8, 128);
        static CACHE: OnceLock<Mutex<HashMap<u32, Arc<Vec<u32>>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let hit = cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&size)
            .cloned();
        let ranks = hit.unwrap_or_else(|| {
            // Built outside the lock: two racing callers build the same
            // mask and one copy is kept.
            let built = Arc::new(void_and_cluster(size as usize));
            cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(size)
                .or_insert(built)
                .clone()
        });
        Self {
            width: size,
            height: size,
            divisor: size * size,
            cells: ranks.as_ref().clone(),
        }
    }
}

/// The crowding field of a binary pattern on an `n × n` torus.
struct Field {
    n: usize,
    reach: usize,
    /// `kernel[dy * (reach + 1) + dx]` for `0 ≤ dx, dy ≤ reach`.
    kernel: Vec<f64>,
    energy: Vec<f64>,
    set: Vec<bool>,
    ones: usize,
}

impl Field {
    fn new(n: usize) -> Self {
        // exp(−1 / (2 · 1.5²)); powers by repeated multiplication so the
        // table does not depend on the platform's `exp`.
        const C: f64 = 0.800_737_402_916_808_1;
        let reach = ((n - 1) / 2).min(8);
        let mut pow = vec![1.0f64; 2 * reach * reach + 1];
        for k in 1..pow.len() {
            pow[k] = pow[k - 1] * C;
        }
        let mut kernel = Vec::with_capacity((reach + 1) * (reach + 1));
        for dy in 0..=reach {
            for dx in 0..=reach {
                kernel.push(pow[dx * dx + dy * dy]);
            }
        }
        Self {
            n,
            reach,
            kernel,
            energy: vec![0.0; n * n],
            set: vec![false; n * n],
            ones: 0,
        }
    }

    fn splat(&mut self, at: usize, sign: f64) {
        let (n, r) = (self.n, self.reach);
        let (cx, cy) = (at % n, at / n);
        for dy in 0..=2 * r {
            let y = (cy + n + dy - r) % n;
            let ky = dy.abs_diff(r);
            let row = &mut self.energy[y * n..(y + 1) * n];
            for dx in 0..=2 * r {
                let x = (cx + n + dx - r) % n;
                row[x] += sign * self.kernel[ky * (r + 1) + dx.abs_diff(r)];
            }
        }
    }

    fn insert(&mut self, at: usize) {
        self.set[at] = true;
        self.ones += 1;
        self.splat(at, 1.0);
    }

    fn remove(&mut self, at: usize) {
        self.set[at] = false;
        self.ones -= 1;
        self.splat(at, -1.0);
    }

    /// The set cell with the largest field (first in raster order).
    fn tightest_cluster(&self) -> usize {
        let mut best = (f64::NEG_INFINITY, 0);
        for (i, (&e, &s)) in self.energy.iter().zip(&self.set).enumerate() {
            if s && e > best.0 {
                best = (e, i);
            }
        }
        best.1
    }

    /// The unset cell with the smallest field (first in raster order).
    fn largest_void(&self) -> usize {
        let mut best = (f64::INFINITY, 0);
        for (i, (&e, &s)) in self.energy.iter().zip(&self.set).enumerate() {
            if !s && e < best.0 {
                best = (e, i);
            }
        }
        best.1
    }
}

/// Void-and-cluster ranks `0 … n² − 1` of an `n × n` torus, row-major.
fn void_and_cluster(n: usize) -> Vec<u32> {
    let total = n * n;
    let mut field = Field::new(n);
    let mut rng = crate::noise::Rng::new(0x0B1E_5EED ^ n as u64);
    let seeds = (total / 10).max(1);
    while field.ones < seeds {
        let at = ((rng.unit() * total as f64) as usize).min(total - 1);
        if !field.set[at] {
            field.insert(at);
        }
    }
    // 1. Relax into a uniformly spread prototype.
    for _ in 0..10 * total {
        let cluster = field.tightest_cluster();
        field.remove(cluster);
        let void = field.largest_void();
        field.insert(void);
        if void == cluster {
            break;
        }
    }
    let prototype = field.set.clone();
    let mut rank = vec![0u32; total];
    // 2. The prototype's own points, last to first.
    while field.ones > 0 {
        let cluster = field.tightest_cluster();
        field.remove(cluster);
        rank[cluster] = field.ones as u32;
    }
    // 3. Everything else, filling the largest void.
    field.energy.fill(0.0);
    for (at, _) in prototype.iter().enumerate().filter(|(_, s)| **s) {
        field.insert(at);
    }
    while field.ones < total {
        let void = field.largest_void();
        rank[void] = field.ones as u32;
        field.insert(void);
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation(cells: &[u32]) -> bool {
        let mut seen = vec![false; cells.len()];
        cells
            .iter()
            .all(|&c| (c as usize) < seen.len() && !std::mem::replace(&mut seen[c as usize], true))
    }

    #[test]
    fn blue_noise_is_a_permutation_and_stable() {
        let m = ThresholdMap::blue_noise(16);
        assert_eq!((m.width, m.height, m.divisor), (16, 16, 256));
        assert!(is_permutation(&m.cells));
        assert_eq!(m, ThresholdMap::blue_noise(16));
        assert_eq!(m.cells, void_and_cluster(16));
    }

    /// Every level set of the mask is spread out: no two of the first
    /// `k` points are neighbours while the density allows it, and the
    /// low-frequency energy of the mask is far below white noise's.
    #[test]
    fn blue_noise_level_sets_are_spread() {
        let n = 32usize;
        let m = ThresholdMap::blue_noise(n as u32);
        let lit: Vec<(i64, i64)> = (0..n * n)
            .filter(|&i| m.cells[i] < (n * n / 16) as u32)
            .map(|i| ((i % n) as i64, (i / n) as i64))
            .collect();
        let wrap = |d: i64| d.abs().min(n as i64 - d.abs());
        for (i, p) in lit.iter().enumerate() {
            for q in &lit[i + 1..] {
                let d2 = wrap(p.0 - q.0).pow(2) + wrap(p.1 - q.1).pow(2);
                assert!(d2 >= 5, "points {p:?} and {q:?} are {d2} apart (squared)");
            }
        }
        // The mean threshold of 8×8 blocks scatters far less than white
        // noise's would (standard deviation 0.289 / 8 of the range).
        let mean = (n * n - 1) as f64 / 2.0;
        let mut squares = 0.0;
        for by in 0..n / 8 {
            for bx in 0..n / 8 {
                let mut s = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        s += m.cells[(by * 8 + y) * n + bx * 8 + x] as f64;
                    }
                }
                squares += ((s / 64.0 - mean) / (n * n) as f64).powi(2);
            }
        }
        let deviation = (squares / (n * n / 64) as f64).sqrt();
        assert!(deviation < 0.4 * 0.289 / 8.0, "block deviation {deviation}");
    }

    #[test]
    fn clustered_dot_tiles_exactly_and_grows_from_the_centre() {
        // Period 4 at 0°: a 4 × 4 tile, 16 thresholds, a permutation.
        let m = ThresholdMap::clustered_dot(4.0, 0.0, SpotShape::Round);
        assert_eq!((m.width, m.height, m.divisor), (4, 4, 16));
        assert!(is_permutation(&m.cells));
        assert_eq!(m.cells[0], 0, "the cell centre switches on first");
        // 45° with cell vector (3, 3): 18 thresholds on a 6 × 6 tile,
        // each used twice.
        let m = ThresholdMap::clustered_dot(18f64.sqrt(), 45.0, SpotShape::Round);
        assert_eq!((m.width, m.height, m.divisor), (6, 6, 18));
        for t in 0..18 {
            assert_eq!(m.cells.iter().filter(|&&c| c == t).count(), 2);
        }
        // Translating by the cell vector keeps the threshold.
        let at = |x: i64, y: i64| m.cells[(y.rem_euclid(6) * 6 + x.rem_euclid(6)) as usize];
        for y in 0..6 {
            for x in 0..6 {
                assert_eq!(at(x, y), at(x + 3, y + 3));
                assert_eq!(at(x, y), at(x - 3, y + 3));
            }
        }
        // An irrational request snaps to a tiling screen.
        for (p, a) in [(5.3, 15.0), (7.0, 75.0), (64.0, 33.0), (f64::NAN, f64::NAN)] {
            for shape in [
                SpotShape::Round,
                SpotShape::Diamond,
                SpotShape::Line,
                SpotShape::Square,
            ] {
                let m = ThresholdMap::clustered_dot(p, a, shape);
                assert!(m.width <= 256 && m.cells.len() == (m.width * m.height) as usize);
                assert!(m.cells.iter().all(|&c| c < m.divisor));
            }
        }
    }

    #[test]
    fn line_screen_is_constant_along_its_lines() {
        let m = ThresholdMap::clustered_dot(4.0, 0.0, SpotShape::Line);
        // s = x / 4 only matters up to the residue tie-break: the first
        // threshold of every column block is the column's.
        for x in 0..4u32 {
            let col: Vec<u32> = (0..4).map(|y| m.cells[(y * 4 + x) as usize] / 4).collect();
            assert!(col.iter().all(|&c| c == col[0]), "column {x}: {col:?}");
        }
    }
}
