//! Morphology with arbitrary kernels (issue #12): the `-morphology`
//! method family on a user-supplied or generated structuring element —
//! erode, dilate, open, close, smooth, the three edges, top hat, bottom
//! hat, hit-and-miss, thinning, thickening and the skeleton.
//!
//! A kernel is a small grid with an origin. A cell is *on* (value 1),
//! *off* (value 0) or *don't care* (NaN). On greyscale samples, per
//! colour channel:
//!
//! | method | result |
//! |---|---|
//! | erode `ε` | minimum over the on cells |
//! | dilate `δ` | maximum over the on cells of the reflected kernel |
//! | open | `δ(ε(f))` |
//! | close | `ε(δ(f))` |
//! | smooth | close of the open |
//! | edge in / out / edge | `f − ε(f)`, `δ(f) − f`, `δ(f) − ε(f)` |
//! | top hat / bottom hat | `f − open(f)`, `close(f) − f` |
//! | hit-and-miss | `max(0, min over on cells − max over off cells)`, outside the picture counting as background |
//! | thinning | `f − hit-and-miss(f)` |
//! | thicken | `max(f, f + hit-and-miss(f))` |
//!
//! Kernel cells that fall outside the picture take no part. With that
//! rule and the reflection, erosion and dilation are adjoint on the
//! picture's own domain, so opening never brightens, closing never
//! darkens and both are idempotent for any kernel, symmetric or not. With
//! several kernels, erode / dilate and their compounds use each in
//! turn, hit-and-miss takes the largest response, and thinning /
//! thickening apply one after another. `iterations` repeats a method;
//! [`KernelMorphology::until_stable`] repeats until the picture stops
//! changing.
//!
//! # Skeleton
//!
//! [`KernelMorphology::skeleton`] thins with two patterns in their four
//! quarter turns until nothing changes. A foreground pixel may be
//! removed without breaking or shortening a line when it sits on a
//! boundary and its foreground neighbours stay connected to each other
//! without it:
//!
//! - *flat boundary* — the three neighbours on one side are background
//!   and the three on the opposite side foreground: the opposite row
//!   connects everything that touched the pixel;
//! - *corner* — two adjacent sides and the corner between them are
//!   background, the two other sides foreground: those two sides touch
//!   each other diagonally.
//!
//! A line end (one foreground neighbour) matches neither, so lines are
//! not eaten from their ends.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// A structuring element: `width × height` cells that are on (`1`), off
/// (`0`) or don't care (NaN), with an origin.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct MorphKernel {
    width: usize,
    height: usize,
    origin: (usize, usize),
    cells: Vec<f32>,
}

impl PartialEq for MorphKernel {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.origin == other.origin
            && self
                .cells
                .iter()
                .zip(&other.cells)
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
    }
}

impl MorphKernel {
    /// Largest kernel side.
    pub const MAX_SIDE: usize = 255;

    /// A kernel from its cells, row-major, origin at the centre
    /// (`(width − 1) / 2`, `(height − 1) / 2`).
    pub fn new(width: usize, height: usize, cells: Vec<f32>) -> Result<Self, Error> {
        if width == 0 || height == 0 || width > Self::MAX_SIDE || height > Self::MAX_SIDE {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: a morphology kernel is 1 to {} cells a side",
                Self::MAX_SIDE
            )));
        }
        if cells.len() != width * height {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: a {width}x{height} kernel needs {} cells, got {}",
                width * height,
                cells.len()
            )));
        }
        if cells.iter().any(|v| v.is_infinite()) {
            return Err(Error::invalid(
                "oxideav-image-filter: a kernel cell is infinite",
            ));
        }
        Ok(Self {
            width,
            height,
            origin: ((width - 1) / 2, (height - 1) / 2),
            cells,
        })
    }

    /// Move the origin (must lie inside the kernel).
    pub fn with_origin(mut self, x: usize, y: usize) -> Result<Self, Error> {
        if x >= self.width || y >= self.height {
            return Err(Error::invalid(
                "oxideav-image-filter: the kernel origin lies outside the kernel",
            ));
        }
        self.origin = (x, y);
        Ok(self)
    }

    fn shaped(radius: u32, inside: impl Fn(i64, i64, i64) -> bool) -> Self {
        let r = radius.min(Self::MAX_SIDE as u32 / 2) as i64;
        let side = (2 * r + 1) as usize;
        let cells = (0..side * side)
            .map(|i| {
                let (dx, dy) = ((i % side) as i64 - r, (i / side) as i64 - r);
                if inside(dx, dy, r) { 1.0 } else { f32::NAN }
            })
            .collect();
        Self {
            width: side,
            height: side,
            origin: (r as usize, r as usize),
            cells,
        }
    }

    /// The `(2r + 1)²` square.
    pub fn square(radius: u32) -> Self {
        Self::shaped(radius, |_, _, _| true)
    }

    /// The diamond `|dx| + |dy| ≤ r`.
    pub fn diamond(radius: u32) -> Self {
        Self::shaped(radius, |dx, dy, r| dx.abs() + dy.abs() <= r)
    }

    /// The disk `dx² + dy² ≤ r²`.
    pub fn disk(radius: u32) -> Self {
        Self::shaped(radius, |dx, dy, r| dx * dx + dy * dy <= r * r)
    }

    /// The plus: the origin's row and column.
    pub fn plus(radius: u32) -> Self {
        Self::shaped(radius, |dx, dy, _| dx == 0 || dy == 0)
    }

    /// A `width × height` rectangle, all on.
    pub fn rectangle(width: usize, height: usize) -> Result<Self, Error> {
        Self::new(
            width,
            height,
            vec![1.0; width.saturating_mul(height).min(1 << 16)],
        )
    }

    /// Parse `"WxH: v, v, …"` (or `"W: …"` for a square): cells
    /// row-major, separated by commas or blanks; `-`, `nan` or `x` is a
    /// don't-care cell. A named shape is `square:r`, `diamond:r`,
    /// `disk:r`, `plus:r` (radius 1 by default).
    pub fn parse(spec: &str) -> Result<Self, Error> {
        let bad = || {
            Error::invalid(format!(
                "oxideav-image-filter: cannot read the kernel '{spec}'"
            ))
        };
        let (head, body) = match spec.split_once(':') {
            Some((h, b)) => (h.trim(), b.trim()),
            None => (spec.trim(), ""),
        };
        let radius = || -> Result<u32, Error> {
            if body.is_empty() {
                Ok(1)
            } else {
                body.parse().map_err(|_| bad())
            }
        };
        match head.to_ascii_lowercase().as_str() {
            "square" => return Ok(Self::square(radius()?)),
            "diamond" => return Ok(Self::diamond(radius()?)),
            "disk" => return Ok(Self::disk(radius()?)),
            "plus" | "cross" => return Ok(Self::plus(radius()?)),
            _ => {}
        }
        let (w, h) = match head.split_once(['x', 'X']) {
            Some((w, h)) => (
                w.trim().parse().map_err(|_| bad())?,
                h.trim().parse().map_err(|_| bad())?,
            ),
            None => {
                let n: usize = head.parse().map_err(|_| bad())?;
                (n, n)
            }
        };
        let cells = body
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .map(|t| match t.to_ascii_lowercase().as_str() {
                "-" | "nan" | "x" => Ok(f32::NAN),
                other => other.parse::<f32>().map_err(|_| bad()),
            })
            .collect::<Result<Vec<f32>, Error>>()?;
        Self::new(w, h, cells)
    }

    /// Width and height.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// The cells, row-major.
    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    /// The kernel turned a quarter turn clockwise (origin with it).
    pub fn rotated(&self) -> Self {
        let (w, h) = (self.width, self.height);
        let mut cells = vec![0.0; w * h];
        // (x, y) -> (h − 1 − y, x) on an h × w grid.
        for y in 0..h {
            for x in 0..w {
                cells[x * h + (h - 1 - y)] = self.cells[y * w + x];
            }
        }
        Self {
            width: h,
            height: w,
            origin: (h - 1 - self.origin.1, self.origin.0),
            cells,
        }
    }

    /// The offsets `(dx, dy)` from the origin of the cells `pick` takes.
    fn offsets(&self, pick: impl Fn(f32) -> bool) -> Vec<(i64, i64)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, v)| pick(**v))
            .map(|(i, _)| {
                (
                    (i % self.width) as i64 - self.origin.0 as i64,
                    (i / self.width) as i64 - self.origin.1 as i64,
                )
            })
            .collect()
    }

    fn on(&self) -> Vec<(i64, i64)> {
        self.offsets(|v| v >= 0.5)
    }

    fn off(&self) -> Vec<(i64, i64)> {
        self.offsets(|v| v < 0.5)
    }
}

/// A `-morphology` method.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelMorphMethod {
    /// Minimum over the kernel.
    Erode,
    /// Maximum over the reflected kernel.
    Dilate,
    /// Dilate of the erode.
    Open,
    /// Erode of the dilate.
    Close,
    /// Close of the open.
    Smooth,
    /// `f − erode`.
    EdgeIn,
    /// `dilate − f`.
    EdgeOut,
    /// `dilate − erode`.
    Edge,
    /// `f − open`.
    TopHat,
    /// `close − f`.
    BottomHat,
    /// Foreground under the on cells and background under the off ones.
    HitAndMiss,
    /// Remove what hit-and-miss finds.
    Thinning,
    /// Add what hit-and-miss finds.
    Thicken,
}

impl KernelMorphMethod {
    /// A method by its `-morphology` name.
    pub fn from_name(name: &str) -> Option<Self> {
        let n: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        Some(match n.as_str() {
            "erode" => Self::Erode,
            "dilate" => Self::Dilate,
            "open" => Self::Open,
            "close" => Self::Close,
            "smooth" => Self::Smooth,
            "edgein" => Self::EdgeIn,
            "edgeout" => Self::EdgeOut,
            "edge" => Self::Edge,
            "tophat" => Self::TopHat,
            "bottomhat" => Self::BottomHat,
            "hitandmiss" | "hmt" => Self::HitAndMiss,
            "thinning" | "thin" => Self::Thinning,
            "thicken" => Self::Thicken,
            _ => return None,
        })
    }
}

/// Morphology with arbitrary kernels (see the
/// [module docs](crate::morph_kernel)).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct KernelMorphology {
    /// The method.
    pub method: KernelMorphMethod,
    /// The kernels (at least one).
    pub kernels: Vec<MorphKernel>,
    /// How many times the method is applied; `None` repeats until the
    /// picture stops changing.
    pub iterations: Option<u32>,
}

impl KernelMorphology {
    /// `method` with one kernel, applied once.
    pub fn new(method: KernelMorphMethod, kernel: MorphKernel) -> Self {
        Self {
            method,
            kernels: vec![kernel],
            iterations: Some(1),
        }
    }

    /// Use several kernels.
    pub fn with_kernels(mut self, kernels: Vec<MorphKernel>) -> Result<Self, Error> {
        if kernels.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: morphology needs at least one kernel",
            ));
        }
        self.kernels = kernels;
        Ok(self)
    }

    /// Apply the method `n` times.
    pub fn with_iterations(mut self, n: u32) -> Self {
        self.iterations = Some(n);
        self
    }

    /// Repeat until the picture stops changing.
    pub fn until_stable(mut self) -> Self {
        self.iterations = None;
        self
    }

    /// Thin to a one-pixel-wide skeleton (see the
    /// [module docs](crate::morph_kernel)).
    pub fn skeleton() -> Self {
        let n = f32::NAN;
        let flat = MorphKernel::new(3, 3, vec![0., 0., 0., n, 1., n, 1., 1., 1.]).expect("3x3");
        let corner = MorphKernel::new(3, 3, vec![n, 0., 0., 1., 1., 0., n, 1., n]).expect("3x3");
        let mut kernels = Vec::with_capacity(8);
        let (mut a, mut b) = (flat, corner);
        for _ in 0..4 {
            kernels.push(a.clone());
            kernels.push(b.clone());
            (a, b) = (a.rotated(), b.rotated());
        }
        Self {
            method: KernelMorphMethod::Thinning,
            kernels,
            iterations: None,
        }
    }

    /// Run on a float picture (colour channels; alpha untouched).
    pub fn apply_float(&self, img: &FloatImage) -> FloatImage {
        let mut cur = img.clone();
        let limit = match self.iterations {
            Some(n) => n as usize,
            // Nothing can keep changing for longer than the picture is
            // large (each pass moves a boundary by at least a pixel).
            None => 2 * (img.width() as usize + img.height() as usize) + 16,
        };
        for _ in 0..limit {
            let next = self.once(&cur);
            let stable = next.samples() == cur.samples();
            cur = next;
            if stable {
                break;
            }
        }
        cur
    }

    fn once(&self, img: &FloatImage) -> FloatImage {
        use KernelMorphMethod as M;
        let through = |f: &dyn Fn(&FloatImage, &MorphKernel) -> FloatImage| {
            self.kernels.iter().fold(img.clone(), |acc, k| f(&acc, k))
        };
        let erode = |i: &FloatImage, k: &MorphKernel| extremum(i, &k.on(), false, false, None);
        let dilate = |i: &FloatImage, k: &MorphKernel| extremum(i, &k.on(), true, true, None);
        let open = |i: &FloatImage, k: &MorphKernel| dilate(&erode(i, k), k);
        let close = |i: &FloatImage, k: &MorphKernel| erode(&dilate(i, k), k);
        match self.method {
            M::Erode => through(&erode),
            M::Dilate => through(&dilate),
            M::Open => through(&open),
            M::Close => through(&close),
            M::Smooth => through(&|i, k| close(&open(i, k), k)),
            M::EdgeIn => combine(img, &through(&erode), |f, e| f - e),
            M::EdgeOut => combine(&through(&dilate), img, |d, f| d - f),
            M::Edge => combine(&through(&dilate), &through(&erode), |d, e| d - e),
            M::TopHat => combine(img, &through(&open), |f, o| f - o),
            M::BottomHat => combine(&through(&close), img, |c, f| c - f),
            M::HitAndMiss => {
                let mut best: Option<FloatImage> = None;
                for k in &self.kernels {
                    let hit = hit_and_miss(img, k);
                    best = Some(match best {
                        None => hit,
                        Some(b) => combine(&b, &hit, f32::max),
                    });
                }
                best.unwrap_or_else(|| img.clone())
            }
            M::Thinning => through(&|i, k| combine(i, &hit_and_miss(i, k), |f, h| f - h)),
            M::Thicken => through(&|i, k| combine(i, &hit_and_miss(i, k), |f, h| f.max(f + h))),
        }
    }
}

/// Per-sample combination of two pictures' colour channels.
fn combine(a: &FloatImage, b: &FloatImage, f: impl Fn(f32, f32) -> f32) -> FloatImage {
    let mut out = a.clone();
    let (ch, cc) = (a.channels(), a.color_channels());
    for (o, p) in out
        .samples_mut()
        .chunks_exact_mut(ch)
        .zip(b.samples().chunks_exact(ch))
    {
        for (v, q) in o[..cc].iter_mut().zip(p) {
            *v = f(*v, *q);
        }
    }
    out
}

/// Maximum (`max`) or minimum over `offsets` (negated when `reflect`);
/// cells outside the picture are skipped (a pixel none of whose cells
/// lands inside keeps its value) unless `outside` gives them a value.
/// Rows under the thread budget.
fn extremum(
    img: &FloatImage,
    offsets: &[(i64, i64)],
    max: bool,
    reflect: bool,
    outside: Option<f32>,
) -> FloatImage {
    let (w, h, ch) = (img.width() as i64, img.height() as i64, img.channels());
    let cc = img.color_channels();
    let src = img.samples();
    let mut out = img.clone();
    if offsets.is_empty() || w == 0 || h == 0 {
        return out;
    }
    let sign = if reflect { -1 } else { 1 };
    crate::exec::par_rows(out.samples_mut(), w as usize * ch, |y, line| {
        for (x, px) in line.chunks_exact_mut(ch).enumerate() {
            for (c, v) in px[..cc].iter_mut().enumerate() {
                let mut best = if max {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                for &(dx, dy) in offsets {
                    let (sx, sy) = (x as i64 + sign * dx, y as i64 + sign * dy);
                    let s = if sx < 0 || sy < 0 || sx >= w || sy >= h {
                        match outside {
                            Some(v) => v,
                            None => continue,
                        }
                    } else {
                        src[(sy * w + sx) as usize * ch + c]
                    };
                    best = if max { best.max(s) } else { best.min(s) };
                }
                if best.is_finite() {
                    *v = best;
                }
            }
        }
    });
    out
}

/// `max(0, min over on cells − max over off cells)`; outside the
/// picture is background.
fn hit_and_miss(img: &FloatImage, kernel: &MorphKernel) -> FloatImage {
    let (on, off) = (kernel.on(), kernel.off());
    let low = extremum(img, &on, false, false, Some(0.0));
    if off.is_empty() {
        return low;
    }
    let high = extremum(img, &off, true, false, Some(0.0));
    if on.is_empty() {
        return combine(&high, &high, |h, _| (1.0 - h).max(0.0));
    }
    combine(&low, &high, |l, h| (l - h).max(0.0))
}

impl PixelFilter for KernelMorphology {
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
    use KernelMorphMethod as M;

    fn bitmap(rows: &[&str]) -> FloatImage {
        let h = rows.len() as u32;
        let w = rows[0].len() as u32;
        let data = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| if b == b'#' { 1.0 } else { 0.0 }))
            .collect();
        FloatImage::from_samples(w, h, 1, data).unwrap()
    }

    fn noise(w: u32, h: u32, seed: u64) -> FloatImage {
        let mut rng = crate::noise::Rng::new(seed);
        FloatImage::from_samples(w, h, 1, (0..w * h).map(|_| rng.unit() as f32).collect()).unwrap()
    }

    fn le(a: &FloatImage, b: &FloatImage) -> bool {
        a.samples().iter().zip(b.samples()).all(|(p, q)| p <= q)
    }

    fn kernels() -> Vec<MorphKernel> {
        vec![
            MorphKernel::square(1),
            MorphKernel::diamond(2),
            MorphKernel::disk(2),
            MorphKernel::plus(1),
            // Asymmetric, off-centre origin.
            MorphKernel::parse("3x2: 1,1,0 0,1,1")
                .unwrap()
                .with_origin(0, 0)
                .unwrap(),
        ]
    }

    #[test]
    fn the_lattice_laws_hold_for_every_kernel() {
        let img = noise(24, 17, 5);
        for k in kernels() {
            let run = |m, i: &FloatImage| KernelMorphology::new(m, k.clone()).apply_float(i);
            let (e, d) = (run(M::Erode, &img), run(M::Dilate, &img));
            let (o, c) = (run(M::Open, &img), run(M::Close, &img));
            // Anti-extensive / extensive, idempotent.
            assert!(le(&o, &img) && le(&img, &c));
            assert_eq!(run(M::Open, &o).samples(), o.samples());
            assert_eq!(run(M::Close, &c).samples(), c.samples());
            // The derived methods are the stated differences.
            let diff = |a: &FloatImage, b: &FloatImage| combine(a, b, |p, q| p - q);
            assert_eq!(run(M::Edge, &img).samples(), diff(&d, &e).samples());
            assert_eq!(run(M::TopHat, &img).samples(), diff(&img, &o).samples());
            assert_eq!(run(M::BottomHat, &img).samples(), diff(&c, &img).samples());
            assert!(run(M::TopHat, &img).samples().iter().all(|&v| v >= 0.0));
            // Duality: eroding the negative is the negative of dilating
            // with the reflected kernel.
            let neg = |i: &FloatImage| combine(i, i, |p, _| 1.0 - p);
            let reflected = k.rotated().rotated();
            let d_reflected = KernelMorphology::new(M::Dilate, reflected).apply_float(&img);
            for (a, b) in run(M::Erode, &neg(&img))
                .samples()
                .iter()
                .zip(neg(&d_reflected).samples())
            {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn erode_and_dilate_on_a_bitmap() {
        let img = bitmap(&[".......", ".#####.", ".#####.", ".#####.", "......."]);
        let e = KernelMorphology::new(M::Erode, MorphKernel::square(1)).apply_float(&img);
        assert_eq!(
            e.samples(),
            bitmap(&[".......", ".......", "..###..", ".......", "......."]).samples()
        );
        let dot = bitmap(&[".....", ".....", "..#..", ".....", "....."]);
        let d = KernelMorphology::new(M::Dilate, MorphKernel::plus(1)).apply_float(&dot);
        assert_eq!(
            d.samples(),
            bitmap(&[".....", "..#..", ".###.", "..#..", "....."]).samples()
        );
        // Two iterations of the plus are the diamond of radius 2.
        let twice = KernelMorphology::new(M::Dilate, MorphKernel::plus(1))
            .with_iterations(2)
            .apply_float(&dot);
        let diamond = KernelMorphology::new(M::Dilate, MorphKernel::diamond(2)).apply_float(&dot);
        assert_eq!(twice.samples(), diamond.samples());
    }

    #[test]
    fn hit_and_miss_finds_its_pattern_only() {
        // Isolated pixels: on at the origin, off all round.
        let n = f32::NAN;
        let lone = MorphKernel::new(3, 3, vec![0., 0., 0., 0., 1., 0., 0., 0., 0.]).unwrap();
        let img = bitmap(&["........", ".#...##.", "........", "...#....", "........"]);
        let hit = KernelMorphology::new(M::HitAndMiss, lone.clone()).apply_float(&img);
        assert_eq!(
            hit.samples(),
            bitmap(&["........", ".#......", "........", "...#....", "........"]).samples()
        );
        let thin = KernelMorphology::new(M::Thinning, lone).apply_float(&img);
        assert_eq!(
            thin.samples(),
            bitmap(&["........", ".....##.", "........", "........", "........"]).samples()
        );
        // Thicken fills a one-pixel gap between two on cells, and only
        // a gap: the off cell under the origin must be background.
        let gap = MorphKernel::new(3, 1, vec![1., 0., 1.]).unwrap();
        let img = bitmap(&[".#.#.", ".###."]);
        let thick = KernelMorphology::new(M::Thicken, gap.clone()).apply_float(&img);
        assert_eq!(thick.samples(), bitmap(&[".###.", ".###."]).samples());
        let hit = KernelMorphology::new(M::HitAndMiss, gap).apply_float(&img);
        assert_eq!(hit.samples(), bitmap(&["..#..", "....."]).samples());
        // A don't-care cell takes no part.
        let any = MorphKernel::new(3, 1, vec![1., n, 1.]).unwrap();
        let hit = KernelMorphology::new(M::HitAndMiss, any).apply_float(&img);
        assert_eq!(hit.samples(), bitmap(&["..#..", "..#.."]).samples());
    }

    #[test]
    fn the_skeleton_of_a_bar_is_a_line_and_keeps_connectivity() {
        let img = bitmap(&[
            "............",
            ".##########.",
            ".##########.",
            ".##########.",
            ".##########.",
            ".##########.",
            "............",
        ]);
        let s = KernelMorphology::skeleton().apply_float(&img);
        let on: Vec<(usize, usize)> = (0..s.samples().len())
            .filter(|&i| s.samples()[i] > 0.5)
            .map(|i| (i % 12, i / 12))
            .collect();
        // The medial line plus a short branch into each corner.
        assert!((8..=20).contains(&on.len()), "{} pixels left", on.len());
        assert!(
            (3..=8).all(|x| on.contains(&(x, 3))),
            "the medial row is kept"
        );
        // Thin: no 2 × 2 block survives.
        for &(x, y) in &on {
            let has = |p: (usize, usize)| on.contains(&p);
            assert!(!(has((x + 1, y)) && has((x, y + 1)) && has((x + 1, y + 1))));
        }
        // Connected: a flood from the first pixel reaches all.
        let mut seen = vec![on[0]];
        let mut i = 0;
        while i < seen.len() {
            let (x, y) = seen[i];
            for p in &on {
                if p.0.abs_diff(x) <= 1 && p.1.abs_diff(y) <= 1 && !seen.contains(p) {
                    seen.push(*p);
                }
            }
            i += 1;
        }
        assert_eq!(seen.len(), on.len());
        // A one-pixel line is already its skeleton.
        let line = bitmap(&[".......", ".#####.", "......."]);
        assert_eq!(
            KernelMorphology::skeleton().apply_float(&line).samples(),
            line.samples()
        );
    }

    #[test]
    fn parsing_shapes_and_refusals() {
        let k = MorphKernel::parse("3: 0,1,0 1,1,1 0,1,0").unwrap();
        assert_eq!(k.size(), (3, 3));
        assert_eq!(k.on().len(), 5);
        assert_eq!(MorphKernel::parse("disk:2").unwrap(), MorphKernel::disk(2));
        assert_eq!(
            MorphKernel::parse("square").unwrap(),
            MorphKernel::square(1)
        );
        assert_eq!(MorphKernel::parse("2x1: 1 -").unwrap().off().len(), 0);
        assert!(MorphKernel::parse("3x3: 1,1").is_err());
        assert!(MorphKernel::parse("0x3:").is_err());
        assert!(MorphKernel::parse("blob:2").is_err());
        assert!(MorphKernel::square(1).with_origin(3, 0).is_err());
        assert_eq!(MorphKernel::disk(1).on().len(), 5);
        assert_eq!(MorphKernel::disk(2).on().len(), 13);
        let k = MorphKernel::parse("3x2: 1,2,3 4,5,6").unwrap();
        assert_eq!(k.rotated().cells(), &[4., 1., 5., 2., 6., 3.]);
        assert_eq!(k.rotated().rotated().rotated().rotated(), k);
        assert_eq!(
            KernelMorphMethod::from_name("Hit-And-Miss"),
            Some(M::HitAndMiss)
        );
        assert_eq!(KernelMorphMethod::from_name("top_hat"), Some(M::TopHat));
        assert_eq!(KernelMorphMethod::from_name("melt"), None);
    }
}
