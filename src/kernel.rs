//! Shared neighbourhood kernels: one implementation per algorithm,
//! generic over the sample type (`u8`, `u16`, `f32`), row-parallel under
//! the budget of [`crate::exec`], with no per-pixel allocation.
//!
//! - [`convolve_separable`]: two-pass separable convolution with
//!   replicate edges, the intermediate kept in binary32 (no rounding
//!   between the passes), optionally on alpha-premultiplied colour.
//! - [`rank_filter`]: sliding-window median — a running histogram for
//!   8-bit samples, a running sorted window for 16-bit and float.
//! - [`extreme_filter`]: windowed minimum / maximum in `O(1)` per sample
//!   and axis (block prefix / suffix scans).
//! - [`box_mean`]: windowed mean by running sums.
//!
//! Every kernel computes each output row from the same operations in
//! the same order whatever the band split, so results are identical at
//! every thread budget.

use crate::exec::{par_bands, par_rows};

/// A sample the kernels can average: `u8`, `u16` (native endian words)
/// or `f32`.
pub(crate) trait Sample: Copy + Send + Sync + PartialOrd + 'static {
    /// Nominal white (`255`, `65535`, `1.0`).
    const UNIT: f32;
    /// The least and greatest values of the type (the identities of a
    /// running maximum and minimum).
    const LOWEST: Self;
    const HIGHEST: Self;
    fn to_f32(self) -> f32;
    /// Round to nearest and saturate for the integer types; identity for
    /// float.
    fn from_f32(v: f32) -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl Sample for u8 {
    const UNIT: f32 = 255.0;
    const LOWEST: Self = 0;
    const HIGHEST: Self = 255;
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        // `as` saturates and maps NaN to 0.
        (v + 0.5).floor() as u8
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        (v + 0.5).floor() as u8
    }
}

impl Sample for u16 {
    const UNIT: f32 = 65535.0;
    const LOWEST: Self = 0;
    const HIGHEST: Self = 65535;
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        (v + 0.5).floor() as u16
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        (v + 0.5).floor() as u16
    }
}

impl Sample for f32 {
    const UNIT: f32 = 1.0;
    const LOWEST: Self = f32::NEG_INFINITY;
    const HIGHEST: Self = f32::INFINITY;
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// How [`convolve_separable`] treats the last channel of a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alpha {
    /// Every channel is convolved independently.
    Independent,
    /// The last channel is copied from the source, the others convolved.
    Keep,
    /// The last channel is alpha: colour is convolved premultiplied by
    /// it and divided by the convolved alpha, so the colour of a
    /// transparent pixel has no weight (a pixel whose convolved alpha is
    /// zero keeps its source colour).
    Premultiplied,
}

/// Separable convolution of a `w × h` picture of `ch` interleaved
/// samples per pixel held at `stride` samples per row: `kx` along rows,
/// then `ky` down columns, replicate-edge addressing, odd-length centred
/// kernels. Returns a tight (`w · ch` per row) buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve_separable<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    kx: &[f32],
    ky: &[f32],
    alpha: Alpha,
) -> Vec<T> {
    let row = w * ch;
    if row == 0 || h == 0 {
        return Vec::new();
    }
    let alpha = if ch < 2 { Alpha::Independent } else { alpha };
    let rx = kx.len() / 2;
    let ry = ky.len() / 2;
    let inv_unit = 1.0 / T::UNIT;

    // Pass 1: rows. Each band keeps one edge-padded scratch row.
    let mut mid = vec![0.0f32; row * h];
    par_bands(&mut mid, row, |y0, band| {
        let mut padded = vec![0.0f32; (w + 2 * rx) * ch];
        for (i, out) in band.chunks_exact_mut(row).enumerate() {
            let y = y0 + i;
            let line = &src[y * stride..y * stride + row];
            let body = &mut padded[rx * ch..rx * ch + row];
            for (d, s) in body.iter_mut().zip(line) {
                *d = s.to_f32();
            }
            if alpha == Alpha::Premultiplied {
                for px in body.chunks_exact_mut(ch) {
                    let a = px[ch - 1] * inv_unit;
                    for v in &mut px[..ch - 1] {
                        *v *= a;
                    }
                }
            }
            for k in 0..rx {
                for c in 0..ch {
                    padded[k * ch + c] = padded[rx * ch + c];
                    padded[(rx + w + k) * ch + c] = padded[(rx + w - 1) * ch + c];
                }
            }
            for (k, &kv) in kx.iter().enumerate() {
                let taps = &padded[k * ch..k * ch + row];
                for (o, t) in out.iter_mut().zip(taps) {
                    *o += kv * *t;
                }
            }
        }
    });

    // Pass 2: columns, whole rows at a time.
    let mut out: Vec<T> = vec![T::from_f32(0.0); row * h];
    par_bands(&mut out, row, |y0, band| {
        let mut acc = vec![0.0f32; row];
        for (i, dst) in band.chunks_exact_mut(row).enumerate() {
            let y = y0 + i;
            acc.fill(0.0);
            for (k, &kv) in ky.iter().enumerate() {
                let sy = (y + k).saturating_sub(ry).min(h - 1);
                let taps = &mid[sy * row..sy * row + row];
                for (o, t) in acc.iter_mut().zip(taps) {
                    *o += kv * *t;
                }
            }
            match alpha {
                Alpha::Independent => {
                    for (d, a) in dst.iter_mut().zip(&acc) {
                        *d = T::from_f32(*a);
                    }
                }
                Alpha::Keep => {
                    let line = &src[y * stride..y * stride + row];
                    for ((d, a), s) in dst
                        .chunks_exact_mut(ch)
                        .zip(acc.chunks_exact(ch))
                        .zip(line.chunks_exact(ch))
                    {
                        for c in 0..ch - 1 {
                            d[c] = T::from_f32(a[c]);
                        }
                        d[ch - 1] = s[ch - 1];
                    }
                }
                Alpha::Premultiplied => {
                    let line = &src[y * stride..y * stride + row];
                    for ((d, a), s) in dst
                        .chunks_exact_mut(ch)
                        .zip(acc.chunks_exact(ch))
                        .zip(line.chunks_exact(ch))
                    {
                        let cover = a[ch - 1] * inv_unit;
                        if cover > 0.0 {
                            for c in 0..ch - 1 {
                                d[c] = T::from_f32(a[c] / cover);
                            }
                        } else {
                            d[..ch - 1].copy_from_slice(&s[..ch - 1]);
                        }
                        d[ch - 1] = T::from_f32(a[ch - 1]);
                    }
                }
            }
        }
    });
    out
}

/// The window of a rank / extreme / mean filter: `left + 1 + right`
/// columns by `up + 1 + down` rows around each pixel, replicate edges.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub left: usize,
    pub right: usize,
    pub up: usize,
    pub down: usize,
}

impl Window {
    /// A `width × height` window anchored as the statistic filters
    /// anchor it: `(n − 1) / 2` before the pixel, `n / 2` after.
    pub(crate) fn sized(width: usize, height: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            left: (width - 1) / 2,
            right: width / 2,
            up: (height - 1) / 2,
            down: height / 2,
        }
    }

    /// The square `(2r + 1)²` window. The radius is capped at 2³⁰ so
    /// the window's sample count stays within 64 bits (no picture is
    /// large enough to tell).
    pub(crate) fn radius(r: usize) -> Self {
        let r = r.min(1 << 30);
        Self {
            left: r,
            right: r,
            up: r,
            down: r,
        }
    }

    fn columns(self) -> usize {
        self.left + 1 + self.right
    }

    fn rows(self) -> usize {
        self.up + 1 + self.down
    }

    /// Samples in the window.
    pub(crate) fn count(self) -> u64 {
        self.columns() as u64 * self.rows() as u64
    }

    /// The window with its reach cut to the picture: the same set of
    /// distinct replicate-edge positions (but not their multiplicities
    /// — for the extreme filters, which ignore them).
    fn within(self, w: usize, h: usize) -> Self {
        Self {
            left: self.left.min(w.saturating_sub(1)),
            right: self.right.min(w.saturating_sub(1)),
            up: self.up.min(h.saturating_sub(1)),
            down: self.down.min(h.saturating_sub(1)),
        }
    }
}

/// The replicate-edge source index of window offset `i` (`0..n`) around
/// position `p`: `clamp(p + i − before, 0, len − 1)`.
#[inline]
fn tap(p: usize, i: usize, before: usize, len: usize) -> usize {
    (p + i).saturating_sub(before).min(len - 1)
}

/// The geometry a rank row needs.
#[derive(Clone, Copy)]
pub(crate) struct RankJob {
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    rank: u64,
}

/// The rank filter's per-type sliding window. `rank` counts from the
/// smallest (`0`) sample of the window. `Scratch` lives for a band of
/// rows; a row leaves it as it found it (empty).
pub(crate) trait Ranked: Sample {
    type Scratch;
    fn scratch(job: &RankJob) -> Self::Scratch;
    fn rank_row(
        src: &[Self],
        job: &RankJob,
        y: usize,
        out: &mut [Self],
        scratch: &mut Self::Scratch,
    );
    /// The whole picture; a type may pick a cheaper equivalent window.
    #[allow(clippy::too_many_arguments)]
    fn rank_picture(
        src: &[Self],
        stride: usize,
        w: usize,
        h: usize,
        ch: usize,
        nc: usize,
        win: Window,
        rank: u64,
    ) -> Vec<Self> {
        rank_banded(src, stride, w, h, ch, nc, win, rank)
    }
}

/// How many of a window's offsets `−before..=after` around `centre`
/// land on position `at` of a line of `len` samples under
/// replicate-edge addressing: one, plus the offsets overhanging an edge
/// when `at` is that edge. With it a window is tallied over the
/// distinct positions it covers, so a window larger than the picture
/// costs no more than the picture.
#[inline]
fn copies(at: usize, centre: usize, before: usize, after: usize, len: usize) -> u64 {
    let mut m = 1u64;
    if at == 0 {
        m += before.saturating_sub(centre) as u64;
    }
    if at == len - 1 {
        m += centre.saturating_add(after).saturating_sub(len - 1) as u64;
    }
    m
}

/// A multiset of bin indices with order-statistic queries.
pub(crate) trait Counter {
    fn add(&mut self, bin: usize, n: u64);
    fn remove(&mut self, bin: usize, n: u64);
    /// The bin holding the sample of rank `rank` (0 = smallest).
    fn select(&self, rank: u64) -> usize;
}

/// A running histogram over `LEVELS` levels of 16-way fan-out
/// (`16^LEVELS` bins at the finest): a sample costs `LEVELS` counter
/// updates, a rank query at most `16 · LEVELS` additions. Level `l`
/// starts at `offsets[l]` and holds `16^(l + 1)` counters.
pub(crate) struct Histogram<const LEVELS: usize> {
    counts: Vec<u64>,
    offsets: [usize; LEVELS],
}

impl<const LEVELS: usize> Histogram<LEVELS> {
    fn new() -> Self {
        let mut offsets = [0usize; LEVELS];
        let mut total = 0usize;
        for (l, o) in offsets.iter_mut().enumerate() {
            *o = total;
            total += 16usize << (4 * l);
        }
        Self {
            counts: vec![0; total],
            offsets,
        }
    }
}

impl<const LEVELS: usize> Counter for Histogram<LEVELS> {
    #[inline]
    fn add(&mut self, v: usize, n: u64) {
        for l in 0..LEVELS {
            self.counts[self.offsets[l] + (v >> (4 * (LEVELS - 1 - l)))] += n;
        }
    }

    #[inline]
    fn remove(&mut self, v: usize, n: u64) {
        for l in 0..LEVELS {
            self.counts[self.offsets[l] + (v >> (4 * (LEVELS - 1 - l)))] -= n;
        }
    }

    #[inline]
    fn select(&self, rank: u64) -> usize {
        let mut seen = 0u64;
        let mut bin = 0usize;
        for l in 0..LEVELS {
            bin <<= 4;
            let level = &self.counts[self.offsets[l]..];
            while seen + level[bin] <= rank {
                seen += level[bin];
                bin += 1;
            }
        }
        bin
    }
}

/// A binary indexed tree over `len` bins: updates and rank queries in
/// `O(log len)` for an alphabet too large (or too sparse) to scan.
pub(crate) struct Fenwick {
    tree: Vec<u64>,
}

impl Fenwick {
    fn new(len: usize) -> Self {
        Self {
            tree: vec![0; len + 1],
        }
    }
}

impl Counter for Fenwick {
    fn add(&mut self, bin: usize, n: u64) {
        let mut i = bin + 1;
        while i < self.tree.len() {
            self.tree[i] += n;
            i += i & i.wrapping_neg();
        }
    }

    fn remove(&mut self, bin: usize, n: u64) {
        let mut i = bin + 1;
        while i < self.tree.len() {
            self.tree[i] -= n;
            i += i & i.wrapping_neg();
        }
    }

    fn select(&self, rank: u64) -> usize {
        // Descend by halving steps: the largest prefix whose count is
        // still `≤ rank` ends just before the wanted bin.
        let n = self.tree.len() - 1;
        let mut pos = 0usize;
        let mut left = rank;
        let mut step = n
            .checked_next_power_of_two()
            .unwrap_or(1 << (usize::BITS - 1));
        while step > 0 {
            let next = pos + step;
            if next <= n && self.tree[next] <= left {
                left -= self.tree[next];
                pos = next;
            }
            step >>= 1;
        }
        pos.min(n.saturating_sub(1))
    }
}

/// One row of a counting rank filter: `bin(x, sy)` is the bin of
/// channel `c`'s sample at column `x`, row `sy`; `value(bin)` the
/// sample a bin stands for. The window is tallied over its distinct
/// in-picture positions with their replicate-edge multiplicities, and
/// slides by swapping one copy of the leaving column for one of the
/// entering column.
fn rank_row_counted<T: Copy, C: Counter>(
    job: &RankJob,
    y: usize,
    out: &mut [T],
    counter: &mut C,
    bin: impl Fn(usize, usize, usize) -> usize,
    value: impl Fn(usize, usize) -> T,
) {
    let RankJob {
        w,
        h,
        ch,
        nc,
        win,
        rank,
        ..
    } = *job;
    let (top, bottom) = (
        y.saturating_sub(win.up),
        y.saturating_add(win.down).min(h - 1),
    );
    for c in 0..nc {
        let column = |counter: &mut C, x: usize, times: u64, add: bool| {
            for sy in top..=bottom {
                let n = times * copies(sy, y, win.up, win.down, h);
                if add {
                    counter.add(bin(x, sy, c), n);
                } else {
                    counter.remove(bin(x, sy, c), n);
                }
            }
        };
        for sx in 0..=win.right.min(w - 1) {
            column(counter, sx, copies(sx, 0, win.left, win.right, w), true);
        }
        for x in 0..w {
            if x > 0 {
                let gone = (x - 1).saturating_sub(win.left);
                let come = x.saturating_add(win.right).min(w - 1);
                if gone != come {
                    column(counter, gone, 1, false);
                    column(counter, come, 1, true);
                }
            }
            out[x * ch + c] = value(counter.select(rank), c);
        }
        // Leave the counter empty for the next channel / row.
        for sx in (w - 1).saturating_sub(win.left)..w {
            column(
                counter,
                sx,
                copies(sx, w - 1, win.left, win.right, w),
                false,
            );
        }
    }
}

impl Ranked for u8 {
    type Scratch = Histogram<2>;
    fn scratch(_: &RankJob) -> Self::Scratch {
        Histogram::new()
    }
    fn rank_row(src: &[u8], job: &RankJob, y: usize, out: &mut [u8], hist: &mut Histogram<2>) {
        let (stride, ch) = (job.stride, job.ch);
        rank_row_counted(
            job,
            y,
            out,
            hist,
            |x, sy, c| src[sy * stride + x * ch + c] as usize,
            |bin, _| bin as u8,
        );
    }
}

impl Ranked for u16 {
    type Scratch = Histogram<4>;
    fn scratch(_: &RankJob) -> Self::Scratch {
        Histogram::new()
    }
    fn rank_row(src: &[u16], job: &RankJob, y: usize, out: &mut [u16], hist: &mut Histogram<4>) {
        let (stride, ch) = (job.stride, job.ch);
        rank_row_counted(
            job,
            y,
            out,
            hist,
            |x, sy, c| src[sy * stride + x * ch + c] as usize,
            |bin, _| bin as u16,
        );
    }
}

/// A total order on binary32 as an unsigned key (the order of
/// `f32::total_cmp`: negative NaN lowest, positive NaN highest).
#[inline]
fn float_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

#[inline]
fn float_unkey(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 {
        k & 0x7fff_ffff
    } else {
        !k
    })
}

impl Ranked for f32 {
    /// The window's samples as ordered keys.
    type Scratch = Vec<u32>;
    fn scratch(job: &RankJob) -> Self::Scratch {
        Vec::with_capacity(job.win.count().min(1 << 16) as usize)
    }
    /// Sliding sorted window (arbitrary floats do not bin): each step
    /// swaps the leaving column's samples for the entering column's, one
    /// at a time, shifting only the keys between the two positions.
    fn rank_row(src: &[f32], job: &RankJob, y: usize, out: &mut [f32], sorted: &mut Vec<u32>) {
        let RankJob {
            stride,
            w,
            h,
            ch,
            nc,
            win,
            rank,
        } = *job;
        let (cols, rows) = (win.columns(), win.rows());
        for c in 0..nc {
            sorted.clear();
            let at =
                |x: usize, j: usize| float_key(src[tap(y, j, win.up, h) * stride + x * ch + c]);
            for i in 0..cols {
                let x = tap(0, i, win.left, w);
                sorted.extend((0..rows).map(|j| at(x, j)));
            }
            sorted.sort_unstable();
            for x in 0..w {
                if x > 0 {
                    let gone = tap(x - 1, 0, win.left, w);
                    let come = tap(x, cols - 1, win.left, w);
                    if gone != come {
                        for j in 0..rows {
                            let old = at(gone, j);
                            let new = at(come, j);
                            if old == new {
                                continue;
                            }
                            let p = sorted.partition_point(|&s| s < old);
                            if new > old {
                                // Slot of `new` among the keys after `p`.
                                let q = p + 1 + sorted[p + 1..].partition_point(|&s| s < new);
                                sorted.copy_within(p + 1..q, p);
                                sorted[q - 1] = new;
                            } else {
                                let q = sorted[..p].partition_point(|&s| s < new);
                                sorted.copy_within(q..p, q + 1);
                                sorted[q] = new;
                            }
                        }
                    }
                }
                out[x * ch + c] = float_unkey(sorted[rank as usize]);
            }
        }
        sorted.clear();
    }

    /// A picture whose samples are all exact 16-bit codes (the float
    /// hop of a 16-bit picture) is ranked on the codes with the running
    /// histogram and mapped back to the very same floats. Any other
    /// picture takes the sorted window while the window is small, and
    /// beyond that a binary indexed tree over the picture's distinct
    /// values.
    fn rank_picture(
        src: &[f32],
        stride: usize,
        w: usize,
        h: usize,
        ch: usize,
        nc: usize,
        win: Window,
        rank: u64,
    ) -> Vec<f32> {
        /// Largest window the sorted path materialises.
        const SORTED_MAX: u64 = 1024;
        let row = w * ch;
        if let Some(codes) = exact_codes(src, stride, row, h) {
            let ranked = rank_banded(&codes, row, w, h, ch, nc, win, rank);
            return ranked.iter().map(|&c| c as f32 / 65535.0).collect();
        }
        if win.count() <= SORTED_MAX {
            return rank_banded(src, stride, w, h, ch, nc, win, rank);
        }
        rank_dense(src, stride, w, h, ch, nc, win, rank)
    }
}

/// Rank filter on arbitrary floats through dense ranks: every channel's
/// samples are replaced by their index among the channel's distinct
/// values, the window is tallied in a binary indexed tree over those
/// indices, and the selected index is mapped back to its value.
#[allow(clippy::too_many_arguments)]
fn rank_dense(
    src: &[f32],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    rank: u64,
) -> Vec<f32> {
    let row = w * ch;
    let mut out = tight_copy(src, stride, row, h);
    if row == 0 || h == 0 {
        return out;
    }
    let mut tables: Vec<Vec<u32>> = Vec::with_capacity(nc);
    let mut bins = vec![0u32; w * h * nc];
    for c in 0..nc {
        let keys: Vec<u32> = (0..w * h)
            .map(|p| float_key(src[(p / w) * stride + (p % w) * ch + c]))
            .collect();
        let mut table = keys.clone();
        table.sort_unstable();
        table.dedup();
        for (p, k) in keys.iter().enumerate() {
            bins[p * nc + c] = table.partition_point(|t| t < k) as u32;
        }
        tables.push(table);
    }
    let widest = tables.iter().map(Vec::len).max().unwrap_or(0);
    let job = RankJob {
        stride,
        w,
        h,
        ch,
        nc,
        win,
        rank: rank.min(win.count() - 1),
    };
    let (bins, tables) = (&bins, &tables);
    par_bands(&mut out, row, |y0, band| {
        let mut tree = Fenwick::new(widest);
        for (i, line) in band.chunks_exact_mut(row).enumerate() {
            rank_row_counted(
                &job,
                y0 + i,
                line,
                &mut tree,
                |x, sy, c| bins[(sy * w + x) * nc + c] as usize,
                |bin, c| float_unkey(tables[c][bin]),
            );
        }
    });
    out
}

/// The 16-bit codes of a float picture whose every sample is exactly a
/// code over 65535 (a picture widened from 16-bit words): `Some` when
/// `code / 65535` reproduces each sample bit for bit, so a rank filter
/// may run on the codes and return the very same floats.
fn exact_codes(src: &[f32], stride: usize, row: usize, h: usize) -> Option<Vec<u16>> {
    let mut codes = Vec::with_capacity(row * h);
    for y in 0..h {
        for &v in &src[y * stride..y * stride + row] {
            let code = (v * 65535.0 + 0.5).floor();
            if !(0.0..=65535.0).contains(&code) || (code / 65535.0).to_bits() != v.to_bits() {
                return None;
            }
            codes.push(code as u16);
        }
    }
    Some(codes)
}

/// Rank filter: channel `c < nc` of every output pixel is the sample of
/// rank `rank` (0 = smallest) in the pixel's window; channels `≥ nc`
/// are copied. `rank` is clamped into the window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rank_filter<T: Ranked>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    rank: u64,
) -> Vec<T> {
    T::rank_picture(src, stride, w, h, ch, nc, win, rank)
}

/// The banded sliding-window rank filter on `T`'s own window type.
#[allow(clippy::too_many_arguments)]
fn rank_banded<T: Ranked>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    rank: u64,
) -> Vec<T> {
    let row = w * ch;
    let mut out = tight_copy(src, stride, row, h);
    if row == 0 || h == 0 {
        return out;
    }
    let job = RankJob {
        stride,
        w,
        h,
        ch,
        nc,
        win,
        rank: rank.min(win.count() - 1),
    };
    par_bands(&mut out, row, |y0, band| {
        let mut scratch = T::scratch(&job);
        for (i, line) in band.chunks_exact_mut(row).enumerate() {
            T::rank_row(src, &job, y0 + i, line, &mut scratch);
        }
    });
    out
}

/// The window median as the statistic filters define it: the upper
/// middle sample, rank `n / 2` of `n`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn median_filter<T: Ranked>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
) -> Vec<T> {
    rank_filter(src, stride, w, h, ch, nc, win, win.count() / 2)
}

fn tight_copy<T: Copy>(src: &[T], stride: usize, row: usize, h: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(row * h);
    for y in 0..h {
        out.extend_from_slice(&src[y * stride..y * stride + row]);
    }
    out
}

/// Running extreme of `n = before + 1 + after` consecutive samples of a
/// line with replicate edges, `O(1)` per sample: the line is cut in
/// blocks of `n`, a suffix scan `g` and a prefix scan `f` of each block
/// are kept, and the window `[i − before, i + after]` is
/// `pick(g[i − before], f[i + after])`. `line(i)` reads the padded line
/// (`len + n − 1` samples); the scratch vectors are reused by callers.
fn running_extreme<T: Copy>(
    padded: &[T],
    n: usize,
    f: &mut Vec<T>,
    g: &mut Vec<T>,
    out: &mut [T],
    pick: impl Fn(T, T) -> T,
) {
    let len = padded.len();
    f.clear();
    f.extend_from_slice(padded);
    g.clear();
    g.extend_from_slice(padded);
    for i in 0..len {
        if i % n != 0 {
            f[i] = pick(f[i - 1], padded[i]);
        }
    }
    for i in (0..len).rev() {
        if (i + 1) % n != 0 && i + 1 < len {
            g[i] = pick(g[i + 1], padded[i]);
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = pick(g[i], f[i + n - 1]);
    }
}

/// Windowed minimum (`max == false`) or maximum of channels `< nc`;
/// channels `≥ nc` are copied. Separable: rows then columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extreme_filter<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    max: bool,
) -> Vec<T> {
    // Up to this many taps per axis the direct whole-row passes (which
    // vectorise) beat the block scans (which do not).
    const DIRECT_TAPS: usize = 32;
    let win = win.within(w, h);
    if win.columns() <= DIRECT_TAPS && win.rows() <= DIRECT_TAPS {
        extreme_direct(src, stride, w, h, ch, nc, win, max)
    } else {
        extreme_scan(src, stride, w, h, ch, nc, win, max)
    }
}

/// [`extreme_filter`] by direct separable passes: each output row is
/// the running pick of the window's taps taken a whole row at a time.
/// Taps outside the picture are left out — for a window that contains
/// its pixel that is replicate-edge addressing.
#[allow(clippy::too_many_arguments)]
fn extreme_direct<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    max: bool,
) -> Vec<T> {
    let row = w * ch;
    if row == 0 || h == 0 {
        return Vec::new();
    }
    let identity = if max { T::LOWEST } else { T::HIGHEST };
    let fold = move |acc: &mut [T], taps: &[T]| {
        if max {
            for (a, t) in acc.iter_mut().zip(taps) {
                *a = if *t > *a { *t } else { *a };
            }
        } else {
            for (a, t) in acc.iter_mut().zip(taps) {
                *a = if *t < *a { *t } else { *a };
            }
        }
    };
    let cols = win.columns();
    let mut mid = vec![identity; row * h];
    par_bands(&mut mid, row, |y0, band| {
        let mut padded = vec![identity; (w + cols - 1) * ch];
        for (i, line) in band.chunks_exact_mut(row).enumerate() {
            let s = &src[(y0 + i) * stride..(y0 + i) * stride + row];
            padded[win.left * ch..win.left * ch + row].copy_from_slice(s);
            for k in 0..cols {
                fold(line, &padded[k * ch..k * ch + row]);
            }
        }
    });
    let rows = win.rows();
    let mut out = vec![identity; row * h];
    let mid = &mid;
    par_rows(&mut out, row, |y, line| {
        for j in 0..rows {
            if let Some(sy) = (y + j).checked_sub(win.up).filter(|sy| *sy < h) {
                fold(line, &mid[sy * row..sy * row + row]);
            }
        }
        if nc < ch {
            let s = &src[y * stride..y * stride + row];
            for (d, s) in line.chunks_exact_mut(ch).zip(s.chunks_exact(ch)) {
                d[nc..].copy_from_slice(&s[nc..]);
            }
        }
    });
    out
}

/// [`extreme_filter`] by block prefix / suffix scans: `O(1)` per sample
/// and axis whatever the window.
#[allow(clippy::too_many_arguments)]
fn extreme_scan<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
    max: bool,
) -> Vec<T> {
    let row = w * ch;
    let mut mid = tight_copy(src, stride, row, h);
    if row == 0 || h == 0 {
        return mid;
    }
    let pick = move |a: T, b: T| {
        if max {
            if b > a { b } else { a }
        } else if b < a {
            b
        } else {
            a
        }
    };
    let cols = win.columns();
    if cols > 1 {
        par_bands(&mut mid, row, |y0, band| {
            let mut padded: Vec<T> = Vec::with_capacity(w + cols - 1);
            let (mut f, mut g) = (Vec::new(), Vec::new());
            let mut line_out: Vec<T> = Vec::with_capacity(w);
            for (i, line) in band.chunks_exact_mut(row).enumerate() {
                let s = &src[(y0 + i) * stride..(y0 + i) * stride + row];
                for c in 0..nc {
                    padded.clear();
                    padded.extend((0..w + cols - 1).map(|i| s[tap(0, i, win.left, w) * ch + c]));
                    line_out.clear();
                    line_out.resize(w, padded[0]);
                    running_extreme(&padded, cols, &mut f, &mut g, &mut line_out, pick);
                    for (x, v) in line_out.iter().enumerate() {
                        line[x * ch + c] = *v;
                    }
                }
            }
        });
    }
    let rows = win.rows();
    if rows == 1 {
        return mid;
    }
    // Columns: the same scan run on whole rows at once — `f` and `g`
    // hold a row per padded line position.
    let mut out = mid.clone();
    let padded_at = |i: usize| tap(0, i, win.up, h);
    let len = h + rows - 1;
    let mut f = vec![mid[0]; len * row];
    let mut g = vec![mid[0]; len * row];
    for i in 0..len {
        let s = &mid[padded_at(i) * row..padded_at(i) * row + row];
        if i % rows == 0 {
            f[i * row..(i + 1) * row].copy_from_slice(s);
        } else {
            let (prev, cur) = f.split_at_mut(i * row);
            let prev = &prev[(i - 1) * row..];
            for ((d, p), v) in cur[..row].iter_mut().zip(prev).zip(s) {
                *d = pick(*p, *v);
            }
        }
    }
    for i in (0..len).rev() {
        let s = &mid[padded_at(i) * row..padded_at(i) * row + row];
        if (i + 1) % rows == 0 || i + 1 == len {
            g[i * row..(i + 1) * row].copy_from_slice(s);
        } else {
            let (cur, next) = g.split_at_mut((i + 1) * row);
            let cur = &mut cur[i * row..];
            for ((d, p), v) in cur.iter_mut().zip(&next[..row]).zip(s) {
                *d = pick(*p, *v);
            }
        }
    }
    let (f, g) = (&f, &g);
    par_rows(&mut out, row, |y, line| {
        let a = &g[y * row..(y + 1) * row];
        let b = &f[(y + rows - 1) * row..(y + rows) * row];
        for (x, px) in line.chunks_exact_mut(ch).enumerate() {
            for c in 0..nc {
                px[c] = pick(a[x * ch + c], b[x * ch + c]);
            }
        }
    });
    out
}

/// Flat greyscale dilation (`max`) or erosion of channels `< nc` by a
/// structuring element given as horizontal runs: `half_widths[i]` is
/// the half-width of the run on row offset `i − half_widths.len() / 2`
/// (`None`: the element has no pixel on that row). Pixels of the
/// element that fall outside the picture are left out. Channels `≥ nc`
/// are copied.
///
/// Each run is a row of whole-row compare-and-keep passes, so the cost
/// per pixel is the number of element pixels divided by the vector
/// width; a rectangular element is better served by [`extreme_filter`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn morph_runs<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    half_widths: &[Option<usize>],
    max: bool,
) -> Vec<T> {
    let row = w * ch;
    let mut out = tight_copy(src, stride, row, h);
    if row == 0 || h == 0 {
        return out;
    }
    let reach = half_widths.len() / 2;
    let widest = half_widths.iter().flatten().copied().max().unwrap_or(0);
    let identity = if max { T::LOWEST } else { T::HIGHEST };
    par_bands(&mut out, row, |y0, band| {
        let mut padded = vec![identity; (w + 2 * widest) * ch];
        let mut acc = vec![identity; row];
        for (i, line) in band.chunks_exact_mut(row).enumerate() {
            let y = y0 + i;
            acc.fill(identity);
            for (j, hw) in half_widths.iter().enumerate() {
                let Some(hw) = *hw else { continue };
                let Some(sy) = (y + j).checked_sub(reach).filter(|sy| *sy < h) else {
                    continue;
                };
                let s = &src[sy * stride..sy * stride + row];
                padded[widest * ch..widest * ch + row].copy_from_slice(s);
                for k in widest - hw..=widest + hw {
                    let taps = &padded[k * ch..k * ch + row];
                    if max {
                        for (a, t) in acc.iter_mut().zip(taps) {
                            *a = if *t > *a { *t } else { *a };
                        }
                    } else {
                        for (a, t) in acc.iter_mut().zip(taps) {
                            *a = if *t < *a { *t } else { *a };
                        }
                    }
                }
            }
            for (d, a) in line.chunks_exact_mut(ch).zip(acc.chunks_exact(ch)) {
                d[..nc].copy_from_slice(&a[..nc]);
            }
        }
    });
    out
}

/// Windowed mean of channels `< nc` by running sums (binary64
/// accumulators, so a sum over any window is exact for the integer
/// types and drift-free for float); channels `≥ nc` are copied. Integer
/// outputs round to nearest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_mean<T: Sample>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    ch: usize,
    nc: usize,
    win: Window,
) -> Vec<T> {
    let row = w * ch;
    let mut out = tight_copy(src, stride, row, h);
    if row == 0 || h == 0 {
        return out;
    }
    // A window sum under replicate-edge addressing is the sum over the
    // distinct positions the window covers plus the edge sample times
    // the window's overhang past that edge — prefix sums give the first
    // part as a difference, so no window is ever walked and a window
    // larger than the picture costs nothing extra.
    let span = |p: usize, before: usize, after: usize, len: usize| {
        (
            p.saturating_sub(before),
            p.saturating_add(after).min(len - 1),
            before.saturating_sub(p) as f64,
            p.saturating_add(after).saturating_sub(len - 1) as f64,
        )
    };
    // Rows: horizontal window sums per row.
    let mut sums = vec![0.0f64; w * nc * h];
    par_bands(&mut sums, w * nc, |y0, band| {
        let mut prefix: Vec<f64> = vec![0.0; w + 1];
        for (i, line) in band.chunks_exact_mut(w * nc).enumerate() {
            let s = &src[(y0 + i) * stride..(y0 + i) * stride + row];
            for c in 0..nc {
                let mut acc = 0.0f64;
                for x in 0..w {
                    acc += s[x * ch + c].to_f64();
                    prefix[x + 1] = acc;
                }
                let (first, last) = (s[c].to_f64(), s[(w - 1) * ch + c].to_f64());
                for x in 0..w {
                    let (lo, hi, over_lo, over_hi) = span(x, win.left, win.right, w);
                    line[x * nc + c] =
                        prefix[hi + 1] - prefix[lo] + over_lo * first + over_hi * last;
                }
            }
        }
    });
    // Columns: prefix sums of whole rows.
    let wn = w * nc;
    let mut prefix = vec![0.0f64; (h + 1) * wn];
    for y in 0..h {
        let (done, next) = prefix.split_at_mut((y + 1) * wn);
        let prev = &done[y * wn..];
        let s = &sums[y * wn..(y + 1) * wn];
        for ((d, p), v) in next[..wn].iter_mut().zip(prev).zip(s) {
            *d = *p + *v;
        }
    }
    let inv = 1.0 / win.count() as f64;
    let (prefix, sums) = (&prefix, &sums);
    par_rows(&mut out, row, |y, line| {
        let (lo, hi, over_lo, over_hi) = span(y, win.up, win.down, h);
        let lo = &prefix[lo * wn..(lo + 1) * wn];
        let hi = &prefix[(hi + 1) * wn..(hi + 2) * wn];
        let first = &sums[..wn];
        let last = &sums[(h - 1) * wn..];
        for (x, px) in line.chunks_exact_mut(ch).enumerate() {
            for (c, v) in px.iter_mut().enumerate().take(nc) {
                let i = x * nc + c;
                let total = hi[i] - lo[i] + over_lo * first[i] + over_hi * last[i];
                *v = T::from_f64(total * inv);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_execution_context;
    use oxideav_core::ExecutionContext;

    fn noise(n: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 32) as u32
            })
            .collect()
    }

    /// The definition, directly: gather the replicate-edge window, sort.
    #[allow(clippy::too_many_arguments)]
    fn window_of<T: Copy>(
        src: &[T],
        w: usize,
        h: usize,
        ch: usize,
        c: usize,
        x: usize,
        y: usize,
        win: Window,
    ) -> Vec<T> {
        let mut out = Vec::new();
        for j in 0..win.rows() {
            for i in 0..win.columns() {
                let sx = tap(x, i, win.left, w);
                let sy = tap(y, j, win.up, h);
                out.push(src[(sy * w + sx) * ch + c]);
            }
        }
        out
    }

    fn check_all<T: Ranked + std::fmt::Debug>(src: &[T], w: usize, h: usize, ch: usize, nc: usize) {
        for win in [
            Window::sized(1, 1),
            Window::sized(3, 3),
            Window::sized(4, 2),
            Window::sized(1, 5),
            Window::sized(7, 1),
            Window::radius(4),
            Window::sized(40, 40),
            Window::sized(3, 70),
            Window::radius(33),
        ] {
            let med = median_filter(src, w * ch, w, h, ch, nc, win);
            let mn = extreme_filter(src, w * ch, w, h, ch, nc, win, false);
            let mx = extreme_filter(src, w * ch, w, h, ch, nc, win, true);
            let mean = box_mean(src, w * ch, w, h, ch, nc, win);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..ch {
                        let i = (y * w + x) * ch + c;
                        if c >= nc {
                            assert_eq!(med[i], src[i]);
                            assert_eq!(mn[i], src[i]);
                            assert_eq!(mx[i], src[i]);
                            assert_eq!(mean[i], src[i]);
                            continue;
                        }
                        let mut win_s = window_of(src, w, h, ch, c, x, y, win);
                        win_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        assert_eq!(med[i], win_s[win_s.len() / 2], "median {win:?} {x},{y}");
                        assert_eq!(mn[i], win_s[0], "min {win:?} {x},{y}");
                        assert_eq!(mx[i], win_s[win_s.len() - 1], "max {win:?} {x},{y}");
                        let sum: f64 = win_s.iter().map(|v| v.to_f64()).sum();
                        let want = sum / win_s.len() as f64;
                        let got = mean[i].to_f64();
                        let tol = if T::UNIT == 1.0 { 1e-6 } else { 0.5 + 1e-9 };
                        assert!(
                            (got - want).abs() <= tol,
                            "mean {win:?} {x},{y}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn windows_match_their_definition_at_every_depth() {
        for (w, h, ch, nc) in [
            (9, 7, 1, 1),
            (5, 6, 4, 3),
            (1, 1, 3, 3),
            (13, 1, 1, 1),
            (1, 9, 3, 3),
        ] {
            let raw = noise(w * h * ch, (w * 31 + h) as u64);
            let b: Vec<u8> = raw.iter().map(|v| (v >> 24) as u8).collect();
            let s: Vec<u16> = raw.iter().map(|v| (v >> 16) as u16).collect();
            let f: Vec<f32> = raw
                .iter()
                .map(|v| *v as f32 / u32::MAX as f32 * 3.0 - 1.0)
                .collect();
            check_all(&b, w, h, ch, nc);
            check_all(&s, w, h, ch, nc);
            check_all(&f, w, h, ch, nc);
        }
    }

    #[test]
    fn floats_on_the_16_bit_grid_rank_through_their_codes_unchanged() {
        let (w, h) = (11usize, 9usize);
        let codes: Vec<u16> = noise(w * h, 5).iter().map(|v| (v >> 16) as u16).collect();
        let f: Vec<f32> = codes.iter().map(|&c| c as f32 / 65535.0).collect();
        assert_eq!(exact_codes(&f, w, w, h).as_deref(), Some(&codes[..]));
        let win = Window::radius(2);
        let via_codes = median_filter(&f, w, w, h, 1, 1, win);
        let direct = rank_banded(&f, w, w, h, 1, 1, win, 12);
        let dense = rank_dense(&f, w, w, h, 1, 1, win, 12);
        assert_eq!(dense, direct);
        assert_eq!(via_codes, direct);
        // Off-grid, out-of-range and non-finite samples take the sorted
        // window.
        for odd in [0.123_456_7f32, -0.25, 1.5, f32::NAN, f32::INFINITY] {
            let mut g = f.clone();
            g[7] = odd;
            assert!(exact_codes(&g, w, w, h).is_none(), "{odd}");
            let out = median_filter(&g, w, w, h, 1, 1, win);
            assert_eq!(out.len(), g.len());
        }
    }

    #[test]
    fn run_morphology_is_the_extreme_over_the_element_inside_the_picture() {
        let (w, h, ch, nc) = (9usize, 7usize, 4usize, 3usize);
        let src: Vec<u16> = noise(w * h * ch, 21)
            .iter()
            .map(|v| (v >> 16) as u16)
            .collect();
        // A radius-2 diamond and a lopsided element with an empty row.
        for runs in [
            vec![Some(0usize), Some(1), Some(2), Some(1), Some(0)],
            vec![Some(3), None, Some(0)],
            vec![Some(0)],
        ] {
            let reach = runs.len() / 2;
            for max in [true, false] {
                let out = morph_runs(&src, w * ch, w, h, ch, nc, &runs, max);
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..ch {
                            let i = (y * w + x) * ch + c;
                            if c >= nc {
                                assert_eq!(out[i], src[i]);
                                continue;
                            }
                            let mut vals = Vec::new();
                            for (j, hw) in runs.iter().enumerate() {
                                let Some(hw) = *hw else { continue };
                                let sy = y as isize + j as isize - reach as isize;
                                for dx in -(hw as isize)..=hw as isize {
                                    let sx = x as isize + dx;
                                    if sx >= 0 && sx < w as isize && sy >= 0 && sy < h as isize {
                                        vals.push(src[(sy as usize * w + sx as usize) * ch + c]);
                                    }
                                }
                            }
                            let want = if max {
                                vals.iter().max().copied()
                            } else {
                                vals.iter().min().copied()
                            };
                            // An element with no pixel inside the picture
                            // leaves the identity; every element here
                            // covers at least one in-picture pixel.
                            assert_eq!(Some(out[i]), want, "{runs:?} max={max} {x},{y},{c}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rank_selects_any_order_statistic() {
        let src: Vec<u8> = vec![9, 1, 8, 2, 7, 3, 6, 4, 5];
        for rank in 0..9u64 {
            let out = rank_filter(&src, 3, 3, 3, 1, 1, Window::radius(1), rank);
            assert_eq!(out[4], rank as u8 + 1);
        }
        // Out-of-range rank clamps to the maximum.
        assert_eq!(
            rank_filter(&src, 3, 3, 3, 1, 1, Window::radius(1), 99)[4],
            9
        );
    }

    #[test]
    fn separable_convolution_matches_the_direct_sum() {
        let (w, h, ch) = (7usize, 5usize, 3usize);
        let src: Vec<f32> = noise(w * h * ch, 7)
            .iter()
            .map(|v| *v as f32 / u32::MAX as f32)
            .collect();
        let kx = [0.25f32, 0.5, 0.25];
        let ky = [0.1f32, 0.2, 0.4, 0.2, 0.1];
        let out = convolve_separable(&src, w * ch, w, h, ch, &kx, &ky, Alpha::Independent);
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let mut want = 0.0f64;
                    for (j, kyv) in ky.iter().enumerate() {
                        for (i, kxv) in kx.iter().enumerate() {
                            let sx = tap(x, i, 1, w);
                            let sy = tap(y, j, 2, h);
                            want += (*kyv * *kxv) as f64 * src[(sy * w + sx) * ch + c] as f64;
                        }
                    }
                    let got = out[(y * w + x) * ch + c] as f64;
                    assert!((got - want).abs() < 1e-5, "{x},{y},{c}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn flat_pictures_are_fixed_points_of_a_unit_kernel() {
        let k = [0.2f32, 0.6, 0.2];
        let b = vec![77u8; 6 * 4 * 4];
        assert_eq!(
            convolve_separable(&b, 24, 6, 4, 4, &k, &k, Alpha::Premultiplied),
            b
        );
        let s = vec![40_001u16; 6 * 4 * 3];
        assert_eq!(
            convolve_separable(&s, 18, 6, 4, 3, &k, &k, Alpha::Independent),
            s
        );
    }

    #[test]
    fn premultiplied_convolution_gives_transparent_colour_no_weight() {
        // An opaque red pixel beside a fully transparent green one: the
        // blurred colour stays pure red wherever anything is covered,
        // and the coverage itself is the blurred alpha.
        let src: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let k = [0.25f32, 0.5, 0.25];
        let one = [1.0f32];
        let out = convolve_separable(&src, 8, 2, 1, 4, &k, &one, Alpha::Premultiplied);
        assert!((out[0] - 1.0).abs() < 1e-6 && out[1].abs() < 1e-6);
        assert!((out[3] - 0.75).abs() < 1e-6);
        assert!((out[4] - 1.0).abs() < 1e-6 && out[5].abs() < 1e-6);
        assert!((out[7] - 0.25).abs() < 1e-6);
        // Independent channels would have mixed green in.
        let mixed = convolve_separable(&src, 8, 2, 1, 4, &k, &one, Alpha::Independent);
        assert!((mixed[1] - 0.25).abs() < 1e-6);
        // Keep leaves alpha as it was.
        let kept = convolve_separable(&src, 8, 2, 1, 4, &k, &one, Alpha::Keep);
        assert_eq!((kept[3], kept[7]), (1.0, 0.0));
    }

    #[test]
    fn every_kernel_is_identical_at_every_thread_budget() {
        let (w, h, ch) = (23usize, 17usize, 4usize);
        let src: Vec<u16> = noise(w * h * ch, 99)
            .iter()
            .map(|v| (v >> 16) as u16)
            .collect();
        let k = [0.1f32, 0.2, 0.4, 0.2, 0.1];
        let run = || {
            (
                convolve_separable(&src, w * ch, w, h, ch, &k, &k, Alpha::Premultiplied),
                median_filter(&src, w * ch, w, h, ch, 3, Window::radius(2)),
                extreme_filter(&src, w * ch, w, h, ch, 3, Window::sized(5, 4), true),
                box_mean(&src, w * ch, w, h, ch, 3, Window::sized(6, 3)),
            )
        };
        let serial = run();
        for t in [2usize, 3, 8, 64] {
            let par = with_execution_context(&ExecutionContext::with_threads(t), run);
            assert_eq!(serial, par, "threads {t}");
        }
    }

    #[test]
    fn empty_pictures_do_not_panic() {
        let e: Vec<u8> = Vec::new();
        let k = [1.0f32];
        assert!(convolve_separable(&e, 0, 0, 0, 3, &k, &k, Alpha::Keep).is_empty());
        assert!(median_filter(&e, 0, 0, 5, 1, 1, Window::radius(1)).is_empty());
        assert!(extreme_filter(&e, 0, 0, 0, 1, 1, Window::radius(1), true).is_empty());
        assert!(box_mean(&e, 0, 0, 0, 1, 1, Window::radius(1)).is_empty());
    }
}
