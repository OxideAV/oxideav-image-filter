//! Colour quantisation (issue #15): reduce a picture to `N` colours
//! chosen for it, or to a given palette, with any dither of
//! [`crate::diffusion`] — the `-colors N`, `-remap` and `-dither`
//! family.
//!
//! # Colour distance
//!
//! A pixel is the vector `(R·A, G·A, B·A, A)` of its normalised stored
//! values (`A = 1` without alpha; grey is `R = G = B`), and the
//! distance between two colours is the Euclidean distance of those
//! vectors: a transparent pixel's colour does not count, and the
//! quantisation error a dither carries is a coverage-weighted colour.
//!
//! # Choosing `N` colours
//!
//! The picture's colours are gathered into a weighted set (binned at 8
//! bits a channel, coarser while there are more than 65 536 bins). A
//! palette of `N` colours that minimises the total squared distance of
//! every pixel to its palette colour is then sought in two steps:
//!
//! 1. **Variance splitting.** Starting from one box holding every
//!    colour, the box with the largest total squared deviation from its
//!    mean is split in two along its axis of largest variance, at the
//!    cut that leaves the smallest sum of the two parts' squared
//!    deviations (found exactly by a sweep over the sorted colours),
//!    until there are `N` boxes. The box means are the first palette.
//! 2. **Refinement.** A few rounds of Lloyd's iteration: every colour
//!    goes to its nearest palette entry and every entry moves to the
//!    mean of its colours. Each round lowers the total squared error.
//!
//! A picture with at most `N` distinct colours keeps them exactly. The
//! result does not depend on the thread budget or the platform.

use std::collections::HashMap;

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::diffusion::DitherMethod;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

type V4 = [f32; 4];

fn premultiplied(c: V4) -> V4 {
    [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
}

/// A sample the colour metric can work with: non-finite values count as
/// zero and far out-of-range ones are held at ±8 (squares stay finite).
fn tame(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-8.0, 8.0)
    } else {
        0.0
    }
}

fn dist2(a: &V4, b: &V4) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
}

/// A list of colours, straight (not premultiplied) normalised RGBA.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
    colors: Vec<V4>,
}

impl Palette {
    /// Most colours a palette holds.
    pub const MAX_COLORS: usize = 65_536;

    /// A palette of the given straight RGBA colours (1 …
    /// [`Self::MAX_COLORS`] finite entries).
    pub fn new(colors: Vec<[f32; 4]>) -> Result<Self, Error> {
        if colors.is_empty() || colors.len() > Self::MAX_COLORS {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: a palette holds 1 to {} colours, got {}",
                Self::MAX_COLORS,
                colors.len()
            )));
        }
        if colors.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::invalid(
                "oxideav-image-filter: a palette colour is not finite",
            ));
        }
        Ok(Self { colors })
    }

    /// A palette of opaque 8-bit RGB colours.
    pub fn from_rgb8(colors: &[[u8; 3]]) -> Result<Self, Error> {
        Self::new(
            colors
                .iter()
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                        1.0,
                    ]
                })
                .collect(),
        )
    }

    /// The palette of a picture used as a colour list: its distinct
    /// colours in raster order of first appearance (the second operand
    /// of `-remap`). Fails past [`Self::MAX_COLORS`] distinct colours.
    pub fn from_image_colors(img: &FloatImage) -> Result<Self, Error> {
        let mut seen: HashMap<[u32; 4], ()> = HashMap::new();
        let mut colors = Vec::new();
        for c in straight_pixels(img) {
            if seen.insert(c.map(f32::to_bits), ()).is_none() {
                if colors.len() == Self::MAX_COLORS {
                    return Err(Error::invalid(
                        "oxideav-image-filter: the palette picture has more than 65536 colours",
                    ));
                }
                colors.push(c);
            }
        }
        Self::new(colors)
    }

    /// At most `max_colors` colours chosen for `img` (see the
    /// [module docs](crate::palette)).
    pub fn adaptive(img: &FloatImage, max_colors: usize) -> Self {
        let n = max_colors.clamp(1, Self::MAX_COLORS);
        let entries = gather(img);
        if entries.is_empty() {
            return Self {
                colors: vec![[0.0, 0.0, 0.0, 1.0]],
            };
        }
        let mut centres = variance_split(&entries, n);
        if entries.len() > centres.len() {
            lloyd(&entries, &mut centres, 8);
        }
        Self {
            colors: centres
                .into_iter()
                .map(|c| {
                    if c[3] > 0.0 {
                        [c[0] / c[3], c[1] / c[3], c[2] / c[3], c[3]]
                    } else {
                        [0.0; 4]
                    }
                })
                .collect(),
        }
    }

    /// As [`Self::adaptive`], from a frame.
    pub fn adaptive_from_frame(
        frame: &VideoFrame,
        params: VideoStreamParams,
        max_colors: usize,
    ) -> Result<Self, Error> {
        Ok(Self::adaptive(
            &crate::measure::float_image_of(frame, params)?,
            max_colors,
        ))
    }

    /// The colours, straight normalised RGBA.
    pub fn colors(&self) -> &[[f32; 4]] {
        &self.colors
    }

    /// Number of colours.
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Never true: a palette holds at least one colour.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// The index of the colour nearest to a straight RGBA colour (the
    /// lowest index on a tie).
    pub fn nearest(&self, rgba: [f32; 4]) -> usize {
        let p = premultiplied(rgba);
        let mut best = (f32::INFINITY, 0);
        for (i, c) in self.colors.iter().enumerate() {
            let d = dist2(&premultiplied(*c), &p);
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }
}

/// The pixels of `img` as straight RGBA.
fn straight_pixels(img: &FloatImage) -> impl Iterator<Item = V4> + '_ {
    let (ch, cc) = (img.channels(), img.color_channels());
    let alpha = img.has_alpha();
    img.samples().chunks_exact(ch).map(move |px| {
        let a = if alpha { px[ch - 1] } else { 1.0 };
        if cc == 1 {
            [px[0], px[0], px[0], a]
        } else {
            [px[0], px[1], px[2], a]
        }
    })
}

/// A weighted colour: premultiplied sum and pixel count.
#[derive(Clone, Copy)]
struct Entry {
    sum: [f64; 4],
    weight: f64,
}

impl Entry {
    fn mean(&self) -> V4 {
        std::array::from_fn(|i| (self.sum[i] / self.weight) as f32)
    }
}

/// The picture's colours, binned, in ascending bin order.
fn gather(img: &FloatImage) -> Vec<Entry> {
    // An open-addressing table on the 32-bit bin key (the pixels of a
    // picture repeat their colours; one multiply finds the slot). When
    // the bins outgrow the limit the grid is coarsened at once — one low
    // bit a channel dropped, the bins so far merged — so the table never
    // holds more than the limit and stays in cache.
    const LIMIT: usize = 65_536;
    let mut bits = 8u32;
    let mut keep = u32::MAX;
    let mut mask = (1usize << 12) - 1;
    let mut slots: Vec<u32> = vec![u32::MAX; mask + 1];
    let mut keyed: Vec<(u32, Entry)> = Vec::new();
    let place = |key: u32, mask: usize| (key.wrapping_mul(0x9E37_79B1) as usize >> 7) & mask;
    let rebuild = |keyed: &[(u32, Entry)], mask: usize| {
        let mut slots = vec![u32::MAX; mask + 1];
        for (i, (k, _)) in keyed.iter().enumerate() {
            let mut at = place(*k, mask);
            while slots[at] != u32::MAX {
                at = (at + 1) & mask;
            }
            slots[at] = i as u32;
        }
        slots
    };
    for c in straight_pixels(img) {
        let p = premultiplied(c.map(tame));
        let key = p.iter().fold(0u32, |k, &v| {
            (k << 8) | (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
        }) & keep;
        let mut at = place(key, mask);
        let entry = loop {
            match slots[at] {
                u32::MAX => {
                    slots[at] = keyed.len() as u32;
                    keyed.push((
                        key,
                        Entry {
                            sum: [0.0; 4],
                            weight: 0.0,
                        },
                    ));
                    break keyed.len() - 1;
                }
                i if keyed[i as usize].0 == key => break i as usize,
                _ => at = (at + 1) & mask,
            }
        };
        let e = &mut keyed[entry].1;
        for (s, v) in e.sum.iter_mut().zip(p) {
            *s += v as f64;
        }
        e.weight += 1.0;
        if keyed.len() > LIMIT && bits > 3 {
            // Coarsen: mask the keys, merge equal ones.
            while keyed.len() > LIMIT && bits > 3 {
                bits -= 1;
                let drop = 8 - bits;
                keep = (0xFFu32 >> drop << drop) * 0x0101_0101;
                keyed.iter_mut().for_each(|(k, _)| *k &= keep);
                keyed.sort_by_key(|(k, _)| *k);
                let mut merged: Vec<(u32, Entry)> = Vec::with_capacity(keyed.len() / 2);
                for (k, e) in keyed {
                    match merged.last_mut() {
                        Some((lk, le)) if *lk == k => {
                            for (s, v) in le.sum.iter_mut().zip(e.sum) {
                                *s += v;
                            }
                            le.weight += e.weight;
                        }
                        _ => merged.push((k, e)),
                    }
                }
                keyed = merged;
            }
            slots = rebuild(&keyed, mask);
        } else if keyed.len() * 2 > mask {
            // Keep the table at most half full.
            mask = mask * 2 + 1;
            slots = rebuild(&keyed, mask);
        }
    }
    keyed.sort_unstable_by_key(|(k, _)| *k);
    keyed.into_iter().map(|(_, e)| e).collect()
}

/// Weight, sum and sum of squares of a set of entries.
#[derive(Clone, Copy, Default)]
struct Moments {
    w: f64,
    s: [f64; 4],
    q: f64,
}

impl Moments {
    fn add(&mut self, e: &Entry) {
        self.w += e.weight;
        let mut q = 0.0;
        for (s, v) in self.s.iter_mut().zip(e.sum) {
            *s += v;
            q += v * v;
        }
        // Σ w |p|² with p the entry mean.
        self.q += q / e.weight;
    }

    fn minus(&self, o: &Moments) -> Moments {
        Moments {
            w: self.w - o.w,
            s: std::array::from_fn(|i| self.s[i] - o.s[i]),
            q: self.q - o.q,
        }
    }

    /// Total squared deviation from the mean.
    fn sse(&self) -> f64 {
        if self.w <= 0.0 {
            return 0.0;
        }
        (self.q - self.s.iter().map(|v| v * v).sum::<f64>() / self.w).max(0.0)
    }
}

/// Split the colour set into at most `n` boxes; their means.
fn variance_split(entries: &[Entry], n: usize) -> Vec<V4> {
    struct Cell {
        members: Vec<u32>,
        moments: Moments,
    }
    let moments_of = |members: &[u32]| {
        let mut m = Moments::default();
        members.iter().for_each(|&i| m.add(&entries[i as usize]));
        m
    };
    let all: Vec<u32> = (0..entries.len() as u32).collect();
    let mut cells = vec![Cell {
        moments: moments_of(&all),
        members: all,
    }];
    // (sse, cell) of the cells that can still be split, largest first.
    let mut queue = std::collections::BinaryHeap::new();
    let key = |c: &Cell, id: usize| (c.moments.sse().to_bits(), std::cmp::Reverse(id));
    queue.push(key(&cells[0], 0));
    while cells.len() < n {
        let Some((_, std::cmp::Reverse(id))) = queue.pop() else {
            break;
        };
        if cells[id].members.len() < 2 {
            continue;
        }
        let total = cells[id].moments;
        // Axis of largest variance.
        let axis = (0..4)
            .map(|a| {
                let q: f64 = cells[id]
                    .members
                    .iter()
                    .map(|&i| {
                        let e = &entries[i as usize];
                        e.sum[a] * e.sum[a] / e.weight
                    })
                    .sum();
                q - total.s[a] * total.s[a] / total.w
            })
            .enumerate()
            .fold(
                (0, f64::NEG_INFINITY),
                |b, (a, v)| if v > b.1 { (a, v) } else { b },
            )
            .0;
        let mut members = std::mem::take(&mut cells[id].members);
        let coord = |i: u32| entries[i as usize].sum[axis] / entries[i as usize].weight;
        members.sort_by(|&p, &q| coord(p).total_cmp(&coord(q)).then(p.cmp(&q)));
        // The cut with the smallest summed deviation.
        let mut left = Moments::default();
        let mut best = (f64::INFINITY, 1usize);
        for (k, &i) in members[..members.len() - 1].iter().enumerate() {
            left.add(&entries[i as usize]);
            let cost = left.sse() + total.minus(&left).sse();
            if cost < best.0 {
                best = (cost, k + 1);
            }
        }
        let right = members.split_off(best.1);
        let (lm, rm) = (moments_of(&members), moments_of(&right));
        cells[id] = Cell {
            members,
            moments: lm,
        };
        cells.push(Cell {
            members: right,
            moments: rm,
        });
        let last = cells.len() - 1;
        queue.push(key(&cells[id], id));
        queue.push(key(&cells[last], last));
    }
    cells
        .iter()
        .map(|c| std::array::from_fn(|i| (c.moments.s[i] / c.moments.w) as f32))
        .collect()
}

/// Lloyd's iteration on the binned colours (`rounds` at most; stops
/// when no centre moves).
fn lloyd(entries: &[Entry], centres: &mut [V4], rounds: usize) {
    for _ in 0..rounds {
        let matcher = Matcher::new(centres);
        let mut acc = vec![
            Entry {
                sum: [0.0; 4],
                weight: 0.0
            };
            centres.len()
        ];
        // The entries are in colour order: the previous one's centre is
        // a close first candidate for the next.
        let mut last = 0usize;
        for e in entries {
            let mean = e.mean();
            last = matcher.nearest_from(&mean, (dist2(&centres[last], &mean), last as u32));
            let a = &mut acc[last];
            for (s, v) in a.sum.iter_mut().zip(e.sum) {
                *s += v;
            }
            a.weight += e.weight;
        }
        let mut moved = false;
        for (c, a) in centres.iter_mut().zip(&acc) {
            if a.weight > 0.0 {
                let m = a.mean();
                moved |= m != *c;
                *c = m;
            }
        }
        if !moved {
            break;
        }
    }
}

/// Exact nearest-colour search over premultiplied palette vectors: the
/// entries are sorted along their axis of largest spread and the search
/// walks outwards from the query's place on that axis until the axis
/// distance alone exceeds the best distance found.
pub(crate) struct Matcher {
    axis: usize,
    /// `(vector, index in the palette)`, ascending along `axis`.
    sorted: Vec<(V4, u32)>,
}

impl Matcher {
    pub(crate) fn new(vectors: &[V4]) -> Self {
        let axis = (0..4)
            .map(|a| {
                let (lo, hi) = vectors
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |m, v| {
                        (m.0.min(v[a]), m.1.max(v[a]))
                    });
                hi - lo
            })
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (a, v)| if v > b.1 { (a, v) } else { b },
            )
            .0;
        let mut sorted: Vec<(V4, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (*v, i as u32))
            .collect();
        sorted.sort_by(|p, q| p.0[axis].total_cmp(&q.0[axis]).then(p.1.cmp(&q.1)));
        Self { axis, sorted }
    }

    /// Index (in the palette) of the vector nearest to `p`; the lowest
    /// index on a tie.
    #[cfg(test)]
    pub(crate) fn nearest(&self, p: &V4) -> usize {
        self.nearest_from(p, (f32::INFINITY, u32::MAX))
    }

    /// [`Self::nearest`] given a first candidate `(squared distance,
    /// index)` — any palette entry, typically the one the previous
    /// pixel took. The answer is the same; a close candidate bounds the
    /// walk along the axis from its first step, and neighbouring pixels
    /// mostly are close.
    pub(crate) fn nearest_from(&self, p: &V4, first: (f32, u32)) -> usize {
        let k = p[self.axis];
        let start = self.sorted.partition_point(|e| e.0[self.axis] < k);
        let mut best = first;
        let consider = |best: &mut (f32, u32), e: &(V4, u32)| {
            let d = dist2(&e.0, p);
            if d < best.0 || (d == best.0 && e.1 < best.1) {
                *best = (d, e.1);
            }
        };
        let (mut lo, mut hi) = (start, start);
        let (mut lo_open, mut hi_open) = (true, true);
        while lo_open || hi_open {
            if hi_open {
                match self.sorted.get(hi) {
                    Some(e) if (e.0[self.axis] - k).powi(2) <= best.0 => {
                        consider(&mut best, e);
                        hi += 1;
                    }
                    _ => hi_open = false,
                }
            }
            if lo_open {
                match lo.checked_sub(1).map(|i| &self.sorted[i]) {
                    Some(e) if (e.0[self.axis] - k).powi(2) <= best.0 => {
                        consider(&mut best, e);
                        lo -= 1;
                    }
                    _ => lo_open = false,
                }
            }
        }
        if best.1 == u32::MAX {
            0
        } else {
            best.1 as usize
        }
    }
}

/// Where the palette of a [`PaletteQuantize`] comes from.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum PaletteSource {
    /// A given palette.
    Fixed(Palette),
    /// At most this many colours chosen for each picture
    /// ([`Palette::adaptive`]).
    Adaptive(usize),
}

/// A picture reduced to a palette: one index per pixel, row-major.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct IndexedImage {
    /// Picture width.
    pub width: u32,
    /// Picture height.
    pub height: u32,
    /// The palette the indices refer to.
    pub palette: Palette,
    /// `width × height` palette indices.
    pub indices: Vec<u32>,
}

/// Replace every pixel by a palette colour — the nearest one, or the
/// one a dither picks so that areas keep their average colour.
///
/// `-colors N` is [`PaletteQuantize::adaptive`], `-remap` is
/// [`PaletteQuantize::new`]; `-dither` is [`Self::with_dither`]. The
/// default dither is serpentine diffusion through the four-neighbour
/// stencil of `docs/image/filter/dithering-kernels.md` §3.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteQuantize {
    /// The palette, or how many colours to choose.
    pub source: PaletteSource,
    /// The dither.
    pub dither: DitherMethod,
}

impl PaletteQuantize {
    /// Remap to `palette`.
    pub fn new(palette: Palette) -> Self {
        Self {
            source: PaletteSource::Fixed(palette),
            dither: DitherMethod::floyd_steinberg(),
        }
    }

    /// Reduce to at most `colors` colours chosen for the picture.
    pub fn adaptive(colors: usize) -> Self {
        Self {
            source: PaletteSource::Adaptive(colors.max(1)),
            dither: DitherMethod::floyd_steinberg(),
        }
    }

    /// Select the dither ([`DitherMethod::None`] for nearest colour).
    pub fn with_dither(mut self, dither: DitherMethod) -> Self {
        self.dither = dither;
        self
    }

    /// The palette used for `img`.
    pub fn palette_for(&self, img: &FloatImage) -> Palette {
        match &self.source {
            PaletteSource::Fixed(p) => p.clone(),
            PaletteSource::Adaptive(n) => Palette::adaptive(img, *n),
        }
    }

    /// Quantise `img` to palette indices.
    pub fn indexed(&self, img: &FloatImage) -> IndexedImage {
        let palette = self.palette_for(img);
        let (w, h) = (img.width() as usize, img.height() as usize);
        let vectors: Vec<V4> = palette.colors.iter().map(|c| premultiplied(*c)).collect();
        let matcher = Matcher::new(&vectors);
        // Five samples a pixel: the premultiplied vector and, once the
        // quantiser has run, the index of the palette entry it chose.
        const SLOTS: usize = 5;
        let mut work: Vec<f32> = Vec::with_capacity(w * h * SLOTS);
        for c in straight_pixels(img) {
            work.extend(premultiplied(c.map(tame)));
            work.push(0.0);
        }
        // Ordered dither moves a pixel along the grey diagonal by the
        // threshold deviation times a spread. To carry a colour to its
        // nearest palette neighbour at offset `o`, the diagonal step
        // must cover `|o|² / |Σ o|` a channel (`|o|` at most, when the
        // neighbour lies across the diagonal); the spread is the mean
        // of that over the palette (at most 1024 entries probed).
        let spacing = if vectors.len() < 2 || !matches!(self.dither, DitherMethod::Ordered(_)) {
            0.0
        } else {
            let step = vectors.len().div_ceil(1024);
            let mut total = 0.0f64;
            let mut probes = 0.0f64;
            for v in vectors.iter().step_by(step) {
                let nearest = vectors
                    .iter()
                    .filter(|o| dist2(v, o) > 0.0)
                    .min_by(|p, q| dist2(v, p).total_cmp(&dist2(v, q)));
                if let Some(o) = nearest {
                    let len = (dist2(v, o) as f64).sqrt();
                    let along: f64 = (0..4).map(|i| (o[i] - v[i]) as f64).sum::<f64>().abs();
                    total += len * len / along.max(len);
                    probes += 1.0;
                }
            }
            if probes > 0.0 {
                (total / probes) as f32
            } else {
                0.0
            }
        };
        // Without alpha the coverage component is constant and takes no
        // part in the dither.
        let active = if img.has_alpha() { 4 } else { 3 };
        let spread = [spacing; 4];
        thread_local! {
            /// The palette entry the last pixel quantised on this thread took.
            static LAST: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let quant = |px: &mut [f32]| {
            let mut p = [0.0, 0.0, 0.0, 1.0];
            for (slot, v) in p[..active].iter_mut().zip(px.iter()) {
                *slot = tame(*v);
            }
            let hint = (LAST.get() as usize).min(vectors.len() - 1);
            let chosen = matcher.nearest_from(&p, (dist2(&vectors[hint], &p), hint as u32));
            LAST.set(chosen as u32);
            px[..active].copy_from_slice(&vectors[chosen][..active]);
            px[SLOTS - 1] = chosen as f32;
        };
        if matches!(self.dither, DitherMethod::None) && crate::exec::threads() <= 1 {
            // Without dither a colour always goes to the same entry, and
            // pictures repeat their colours: on one thread each distinct
            // pixel is matched once and remembered by its bit pattern
            // (the table stops growing on a picture of mostly distinct
            // pixels). With a thread budget the rows run in parallel
            // instead; the result is the same.
            const MEMO: usize = 1 << 17;
            let mut table: Vec<([u32; 4], u32)> = vec![([0; 4], u32::MAX); 2 * MEMO];
            let mut held = 0usize;
            for px in work.as_chunks_mut::<SLOTS>().0 {
                let key: [u32; 4] = std::array::from_fn(|i| px[i].to_bits());
                let hash = key.iter().fold(0u64, |k, v| {
                    (k ^ *v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                });
                let mut at = (hash >> 40) as usize & (2 * MEMO - 1);
                let hit = loop {
                    match table[at] {
                        (_, u32::MAX) => break None,
                        (k, i) if k == key => break Some(i),
                        _ => at = (at + 1) & (2 * MEMO - 1),
                    }
                };
                match hit {
                    Some(i) => {
                        px[..active].copy_from_slice(&vectors[i as usize][..active]);
                        px[SLOTS - 1] = i as f32;
                    }
                    None => {
                        quant(px);
                        if held < MEMO {
                            table[at] = (key, px[SLOTS - 1] as u32);
                            held += 1;
                        }
                    }
                }
            }
        } else {
            let job = crate::diffusion::Job {
                data: &mut work,
                w,
                h,
                ch: SLOTS,
                active,
                spread: &spread,
            };
            crate::diffusion::run(&self.dither, job, &quant);
        }
        let indices = work
            .as_chunks::<SLOTS>()
            .0
            .iter()
            .map(|px| px[SLOTS - 1] as u32)
            .collect();
        IndexedImage {
            width: img.width(),
            height: img.height(),
            palette,
            indices,
        }
    }

    /// Quantise `img`, keeping its layout: every pixel becomes its
    /// palette colour (a picture without alpha takes the colour only).
    pub fn apply_float(&self, img: &FloatImage) -> FloatImage {
        let indexed = self.indexed(img);
        let mut out = img.clone();
        let (ch, cc) = (out.channels(), out.color_channels());
        let alpha = out.has_alpha();
        for (px, &i) in out.samples_mut().chunks_exact_mut(ch).zip(&indexed.indices) {
            let c = indexed.palette.colors[i as usize];
            if cc == 1 {
                // The grey nearest to the palette colour in the metric.
                px[0] = (c[0] + c[1] + c[2]) / 3.0;
            } else {
                px[..3].copy_from_slice(&c[..3]);
            }
            if alpha {
                px[ch - 1] = c[3];
            }
        }
        out
    }
}

impl PixelFilter for PaletteQuantize {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |img| Ok(self.apply_float(&img)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> FloatImage {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend(f(x, y));
            }
        }
        FloatImage::from_samples(w, h, 3, data).unwrap()
    }

    fn sse(a: &FloatImage, b: &FloatImage) -> f64 {
        a.samples()
            .iter()
            .zip(b.samples())
            .map(|(p, q)| ((p - q) as f64).powi(2))
            .sum()
    }

    #[test]
    fn a_picture_of_few_colours_keeps_them_exactly() {
        let colours = [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.25],
            [0.2, 0.2, 0.9],
            [1.0, 1.0, 1.0],
        ];
        let img = rgb(16, 16, |x, y| colours[((x / 4 + y / 4) % 4) as usize]);
        for n in [4usize, 7, 256] {
            let p = Palette::adaptive(&img, n);
            assert_eq!(p.len(), 4);
            let out = PaletteQuantize::adaptive(n).apply_float(&img);
            assert_eq!(out.samples(), img.samples());
        }
        // Two colours of four: the split isolates clusters, never invents
        // a colour outside the hull.
        let p = Palette::adaptive(&img, 2);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn more_colours_never_fit_worse_and_refinement_helps() {
        let img = rgb(64, 64, |x, y| {
            let (u, v) = (x as f32 / 63.0, y as f32 / 63.0);
            [
                u,
                v * v,
                (0.5 + 0.5 * ((u * 9.0).sin() * (v * 7.0).cos())).clamp(0.0, 1.0),
            ]
        });
        let fit = |n: usize| {
            let q = PaletteQuantize::adaptive(n).with_dither(DitherMethod::None);
            sse(&q.apply_float(&img), &img)
        };
        let errors: Vec<f64> = [2usize, 4, 8, 16, 64, 256]
            .iter()
            .map(|&n| fit(n))
            .collect();
        for pair in errors.windows(2) {
            assert!(pair[1] < pair[0], "{errors:?}");
        }
        // 256 colours on a smooth picture: under one 8-bit code RMS a channel.
        let rms = (errors[5] / (64.0 * 64.0 * 3.0)).sqrt();
        assert!(rms < 0.03, "rms {rms}");
        // Lloyd's rounds lower the split's own error.
        let entries = gather(&img);
        let mut centres = variance_split(&entries, 16);
        let cost = |centres: &[V4]| -> f64 {
            let m = Matcher::new(centres);
            entries
                .iter()
                .map(|e| dist2(&centres[m.nearest(&e.mean())], &e.mean()) as f64 * e.weight)
                .sum()
        };
        let before = cost(&centres);
        lloyd(&entries, &mut centres, 8);
        assert!(cost(&centres) <= before);
    }

    #[test]
    fn a_picture_of_very_many_colours_is_binned_coarser_and_still_fits() {
        // 512 × 512 pixels, all distinct on the 8-bit grid in two
        // channels plus a third that varies: more bins than the limit.
        let img = rgb(512, 512, |x, y| {
            [
                x as f32 / 511.0,
                y as f32 / 511.0,
                ((x * 7 + y * 13) % 256) as f32 / 255.0,
            ]
        });
        let entries = gather(&img);
        assert!(
            entries.len() <= 65_536 && entries.len() > 256,
            "{}",
            entries.len()
        );
        let total: f64 = entries.iter().map(|e| e.weight).sum();
        assert_eq!(total, 512.0 * 512.0);
        let p = Palette::adaptive(&img, 64);
        assert_eq!(p.len(), 64);
        // Nearest-colour output is the same with and without a thread
        // budget (the serial path remembers colours it has matched).
        let q = PaletteQuantize::new(p).with_dither(DitherMethod::None);
        let serial = q.indexed(&img);
        let ctx = oxideav_core::ExecutionContext::with_threads(4);
        let parallel = crate::with_execution_context(&ctx, || q.indexed(&img));
        assert_eq!(serial.indices, parallel.indices);
        for (i, px) in serial
            .indices
            .iter()
            .zip(img.samples().as_chunks::<3>().0)
            .step_by(977)
        {
            assert_eq!(
                *i as usize,
                serial.palette.nearest([px[0], px[1], px[2], 1.0])
            );
        }
    }

    #[test]
    fn matcher_agrees_with_the_exhaustive_search() {
        let mut rng = crate::noise::Rng::new(7);
        let vectors: Vec<V4> = (0..300)
            .map(|_| std::array::from_fn(|_| rng.unit() as f32))
            .collect();
        let palette = Palette::new(vectors.clone()).unwrap();
        let m = Matcher::new(
            &vectors
                .iter()
                .map(|v| premultiplied(*v))
                .collect::<Vec<_>>(),
        );
        let premult: Vec<V4> = vectors.iter().map(|v| premultiplied(*v)).collect();
        for i in 0..2000usize {
            let p: V4 = std::array::from_fn(|_| rng.unit() as f32 * 1.4 - 0.2);
            let p = [p[0], p[1], p[2], p[3].clamp(0.0, 1.0)];
            let want = palette.nearest(p);
            let q = premultiplied(p);
            assert_eq!(m.nearest(&q), want);
            // Any first candidate gives the same answer.
            let hint = i % premult.len();
            assert_eq!(
                m.nearest_from(&q, (dist2(&premult[hint], &q), hint as u32)),
                want
            );
        }
    }

    #[test]
    fn dithers_keep_the_average_colour_on_a_fixed_palette() {
        let palette = Palette::from_rgb8(&[
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
        ])
        .unwrap();
        let target = [0.3f32, 0.55, 0.7];
        let img = rgb(64, 64, |_, _| target);
        for (which, dither) in [
            DitherMethod::floyd_steinberg(),
            DitherMethod::diffusion(crate::DiffusionKernel::Stucki),
            DitherMethod::hilbert(),
            DitherMethod::blue_noise(),
        ]
        .into_iter()
        .enumerate()
        {
            let out = PaletteQuantize::new(palette.clone())
                .with_dither(dither.clone())
                .apply_float(&img);
            for (c, want) in target.iter().enumerate() {
                let mean: f64 = out
                    .samples()
                    .iter()
                    .skip(c)
                    .step_by(3)
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (64.0 * 64.0);
                assert!(
                    (mean - *want as f64).abs() < 0.03,
                    "dither {which} channel {c}: {mean}"
                );
            }
            // Only palette colours come out.
            for px in out.samples().as_chunks::<3>().0 {
                assert!(px.iter().all(|&v| v == 0.0 || v == 1.0));
            }
        }
        // Without dither: the single nearest corner.
        let out = PaletteQuantize::new(palette)
            .with_dither(DitherMethod::None)
            .apply_float(&img);
        assert!(
            out.samples()
                .as_chunks::<3>()
                .0
                .iter()
                .all(|px| *px == [0.0, 1.0, 1.0])
        );
    }

    #[test]
    fn transparent_colour_does_not_count_and_alpha_is_quantised() {
        // Half the picture is fully transparent with a wild colour.
        let mut data = Vec::new();
        for i in 0..64 {
            if i % 2 == 0 {
                data.extend([0.9, 0.1, 0.1, 1.0]);
            } else {
                data.extend([0.0, 1.0, 0.0, 0.0]);
            }
        }
        let img = FloatImage::from_samples(8, 8, 4, data).unwrap();
        let p = Palette::adaptive(&img, 2);
        assert_eq!(p.len(), 2);
        let mut alphas: Vec<f32> = p.colors().iter().map(|c| c[3]).collect();
        alphas.sort_by(f32::total_cmp);
        assert_eq!(alphas, vec![0.0, 1.0]);
        let idx = PaletteQuantize::new(p)
            .with_dither(DitherMethod::None)
            .indexed(&img);
        assert_eq!(idx.indices.len(), 64);
        assert_ne!(idx.indices[0], idx.indices[1]);
        assert!(idx.indices.iter().step_by(2).all(|&i| i == idx.indices[0]));
    }

    #[test]
    fn refusals_and_the_colour_list_picture() {
        assert!(Palette::new(vec![]).is_err());
        assert!(Palette::new(vec![[f32::NAN, 0.0, 0.0, 1.0]]).is_err());
        let list = rgb(3, 1, |x, _| [[0.0; 3], [1.0; 3], [0.0; 3]][x as usize]);
        let p = Palette::from_image_colors(&list).unwrap();
        assert_eq!(p.colors(), &[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]);
    }
}
