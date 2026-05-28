//! Frei–Chen 3×3 orthonormal-basis edge detector.
//!
//! The Frei–Chen operator (Werner Frei & Chung-Ching Chen, "Fast
//! Boundary Detection: A Generalization and a New Algorithm", *IEEE
//! Transactions on Computers* C-26(10):988–998, October 1977) treats
//! each 3×3 image neighbourhood as a 9-dimensional vector and projects
//! it onto an orthonormal basis of 9 templates partitioned into three
//! sub-spaces:
//!
//! - **Edge sub-space** (4 templates `S1..S4`): the two gradient
//!   templates (`S1` = horizontal-difference Sobel-like, `S2` =
//!   vertical-difference Sobel-like, scaled by `1/(2√2)`) plus the
//!   two ripple templates (`S3`, `S4`).
//! - **Line sub-space** (4 templates `S5..S8`): two discrete-Laplacian
//!   variants plus two diagonal-line detectors.
//! - **Mean sub-space** (1 template `S9`): the flat-average kernel
//!   `1/3 · [[1,1,1],[1,1,1],[1,1,1]]`.
//!
//! The defining property of the basis is that the templates are
//! mutually orthogonal under the Frobenius (entry-wise dot product)
//! inner product and unit-norm when scaled by the divisors below; the
//! 9 templates therefore form a complete orthonormal basis for the
//! space of 3×3 patches. The Frei–Chen "edge strength" at a pixel is
//! the cosine of the angle between the local neighbourhood vector and
//! the 4-D edge sub-space, normalised to `[0, 1]`:
//!
//! ```text
//! M = sqrt( (p1² + p2² + p3² + p4²) / (p1² + ... + p9²) )
//! ```
//!
//! where `pi = <neighbourhood, Si>` is the projection coefficient onto
//! template `Si`. When the neighbourhood lies entirely in the edge
//! sub-space the metric is `1.0` (a "pure edge"); when it lies entirely
//! in the line or mean sub-space the metric is `0.0`. The output is
//! `round(M · 255)` clamped to `[0, 255]`.
//!
//! ## The 9 basis templates
//!
//! Each template is given as the integer pattern; the per-template
//! divisor is listed beside it. The projection coefficients are
//! computed in integer arithmetic and the divisor is applied at the
//! end so the ratio in `M` is numerator-divisor-free (both numerator
//! and denominator carry the same set of divisors, but since the
//! Frei–Chen ratio is *not* divisor-cancelling — different templates
//! have different divisors — each squared projection is scaled by
//! `1 / divisor²` before the sum).
//!
//! ```text
//!         [  1   √2   1 ]                  [  1   0  -1 ]
//! S1 = ── [  0    0   0 ]    1/(2√2)   S2 = ── [ √2   0  -√2 ]   1/(2√2)
//!         [ -1  -√2  -1 ]                  [  1   0  -1 ]
//!
//!         [  0   -1   √2 ]                 [ √2  -1   0 ]
//! S3 = ── [  1    0   -1 ]   1/(2√2)   S4 = ── [ -1   0   1 ]    1/(2√2)
//!         [ -√2   1    0 ]                 [  0   1   -√2 ]
//!
//!         [  0    1    0 ]                 [ -1   0    1 ]
//! S5 = ── [ -1    0   -1 ]   1/2       S6 = ── [  0   0    0 ]   1/2
//!         [  0    1    0 ]                 [  1   0   -1 ]
//!
//!         [  1   -2    1 ]                 [ -2    1   -2 ]
//! S7 = ── [ -2    4   -2 ]   1/6       S8 = ── [  1    4    1 ]  1/6
//!         [  1   -2    1 ]                 [ -2    1   -2 ]
//!
//!         [  1    1    1 ]
//! S9 = ── [  1    1    1 ]   1/3
//!         [  1    1    1 ]
//! ```
//!
//! `S1` / `S2` are the two cardinal gradient projections (orientations
//! at `0°` and `90°`), `S3` / `S4` the two diagonal gradients
//! (`45°` / `135°`). `S5` / `S6` are line / ripple detectors at the
//! cardinal axes; `S7` / `S8` are discrete-Laplacian variants. `S9`
//! captures the patch's DC component.
//!
//! ## Magnitude variants
//!
//! Two output metrics are offered:
//!
//! - [`FreiChenMode::Edge`] (default) — the canonical edge cosine
//!   `sqrt(E / T)` where `E = p1² + p2² + p3² + p4²` (edge energy)
//!   and `T` is the total `p1² + ... + p9²`.
//! - [`FreiChenMode::Line`] — analogous metric for the line sub-space:
//!   `sqrt(L / T)` with `L = p5² + p6² + p7² + p8²`. Useful for
//!   highlighting straight-line / ridge structures rather than
//!   step edges.
//!
//! When `T == 0` (a flat patch) the magnitude is taken to be `0`.
//!
//! ## Pixel formats and shape
//!
//! Like [`Roberts`](crate::roberts::Roberts) /
//! [`Prewitt`](crate::prewitt::Prewitt) /
//! [`Scharr`](crate::scharr::Scharr) / [`Edge`](crate::Edge), any
//! supported input is collapsed to a luma plane first (Gray8 / YUV use
//! the Y plane directly; RGB / RGBA use the `(R + 2G + B) / 4` quick
//! luma). The output is always `Gray8` of the same dimensions; the
//! one-pixel border samples that would fall outside the input are
//! clamped to the nearest in-bounds pixel.
//!
//! The filter is stateless, single-frame, and `O(W·H)`.
//!
//! ## Clean-room reference
//!
//! The 9 kernels and the cosine-based magnitude are the textbook
//! Frei–Chen operator as derived from the orthonormal-basis property
//! given in Frei & Chen 1977 and reproduced in Pratt, *Digital Image
//! Processing*, 4th ed. (chapter on edge detection); Gonzalez & Woods,
//! *Digital Image Processing*, 4th ed.; and Jähne, *Digital Image
//! Processing*, 6th ed. No external library source was consulted; the
//! orthonormality is checked algebraically in the [`tests`] module
//! (the integer cross-products of any two distinct templates, divided
//! by the product of their divisors squared, sum to zero, and each
//! template's self-product divided by its divisor squared sums to one).
//! The expected outputs in the tests are hand-derived from the formulae
//! above.

use crate::laplacian::build_luma_plane;
use crate::{is_supported_format, ImageFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame, VideoPlane};

/// Which sub-space the Frei–Chen magnitude projects onto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FreiChenMode {
    /// Cosine angle to the **edge** sub-space (templates `S1..S4`).
    /// The classical Frei–Chen output; sensitive to step edges /
    /// gradients. This is the default.
    #[default]
    Edge,
    /// Cosine angle to the **line** sub-space (templates `S5..S8`).
    /// Sensitive to ripples / lines / Laplacian-style ridges; the
    /// complementary projection to [`FreiChenMode::Edge`].
    Line,
}

/// Frei–Chen edge / line detector.
///
/// See the [module-level docs](self) for the 9 templates and the
/// cosine-based magnitude.
#[derive(Clone, Copy, Debug, Default)]
pub struct FreiChen {
    /// Which sub-space projection to report.
    pub mode: FreiChenMode,
}

impl FreiChen {
    /// Build a default Frei–Chen detector ([`FreiChenMode::Edge`]).
    pub fn new() -> Self {
        Self {
            mode: FreiChenMode::Edge,
        }
    }

    /// Build a Frei–Chen detector projecting onto the line sub-space.
    pub fn line() -> Self {
        Self {
            mode: FreiChenMode::Line,
        }
    }

    /// Override the projection mode.
    pub fn with_mode(mut self, mode: FreiChenMode) -> Self {
        self.mode = mode;
        self
    }
}

impl ImageFilter for FreiChen {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: FreiChen does not yet handle {:?}",
                params.format
            )));
        }
        let w = params.width as usize;
        let h = params.height as usize;
        let luma = build_luma_plane(input, params)?;
        let out = frei_chen(&luma, w, h, self.mode);
        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: w,
                data: out,
            }],
        })
    }
}

// ----------------------------------------------------------------------
// Kernel definitions and projection.
// ----------------------------------------------------------------------

// We store integer numerators for each kernel and separately remember
// each kernel's overall divisor `d_i` so that the actual template entry
// is `(numerator entry) / d_i`. Working in integers keeps the
// projection bit-exact for small images.
//
// The "√2" entries in `S1..S4` are represented in floating point at
// projection time; everything else is integer.

/// Project a 3×3 neighbourhood onto the 9 Frei–Chen templates and
/// return the cosine of the angle to the selected sub-space, in
/// `[0.0, 1.0]`.
///
/// `n[i][j]` is the neighbourhood value at row `i`, column `j` with
/// `i, j ∈ {0, 1, 2}`.
fn project_cosine(n: &[[i32; 3]; 3], mode: FreiChenMode) -> f64 {
    // Convert the integer neighbourhood to f64 once.
    let p: [[f64; 3]; 3] = [
        [n[0][0] as f64, n[0][1] as f64, n[0][2] as f64],
        [n[1][0] as f64, n[1][1] as f64, n[1][2] as f64],
        [n[2][0] as f64, n[2][1] as f64, n[2][2] as f64],
    ];
    let s2 = std::f64::consts::SQRT_2;

    // S1: 1/(2√2) * [[1, √2, 1], [0, 0, 0], [-1, -√2, -1]]
    let c1 = (p[0][0] + s2 * p[0][1] + p[0][2] - p[2][0] - s2 * p[2][1] - p[2][2]) / (2.0 * s2);
    // S2: 1/(2√2) * [[1, 0, -1], [√2, 0, -√2], [1, 0, -1]]
    let c2 = (p[0][0] - p[0][2] + s2 * p[1][0] - s2 * p[1][2] + p[2][0] - p[2][2]) / (2.0 * s2);
    // S3: 1/(2√2) * [[0, -1, √2], [1, 0, -1], [-√2, 1, 0]]
    let c3 = (-p[0][1] + s2 * p[0][2] + p[1][0] - p[1][2] - s2 * p[2][0] + p[2][1]) / (2.0 * s2);
    // S4: 1/(2√2) * [[√2, -1, 0], [-1, 0, 1], [0, 1, -√2]]
    let c4 = (s2 * p[0][0] - p[0][1] - p[1][0] + p[1][2] + p[2][1] - s2 * p[2][2]) / (2.0 * s2);
    // S5: 1/2 * [[0, 1, 0], [-1, 0, -1], [0, 1, 0]]
    let c5 = (p[0][1] - p[1][0] - p[1][2] + p[2][1]) / 2.0;
    // S6: 1/2 * [[-1, 0, 1], [0, 0, 0], [1, 0, -1]]
    let c6 = (-p[0][0] + p[0][2] + p[2][0] - p[2][2]) / 2.0;
    // S7: 1/6 * [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    let c7 = (p[0][0] - 2.0 * p[0][1] + p[0][2] - 2.0 * p[1][0] + 4.0 * p[1][1] - 2.0 * p[1][2]
        + p[2][0]
        - 2.0 * p[2][1]
        + p[2][2])
        / 6.0;
    // S8: 1/6 * [[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]
    let c8 = (-2.0 * p[0][0] + p[0][1] - 2.0 * p[0][2] + p[1][0] + 4.0 * p[1][1] + p[1][2]
        - 2.0 * p[2][0]
        + p[2][1]
        - 2.0 * p[2][2])
        / 6.0;
    // S9: 1/3 * [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    let c9 =
        (p[0][0] + p[0][1] + p[0][2] + p[1][0] + p[1][1] + p[1][2] + p[2][0] + p[2][1] + p[2][2])
            / 3.0;

    let e = c1 * c1 + c2 * c2 + c3 * c3 + c4 * c4;
    let l = c5 * c5 + c6 * c6 + c7 * c7 + c8 * c8;
    let t = e + l + c9 * c9;
    if t <= 0.0 {
        return 0.0;
    }
    let num = match mode {
        FreiChenMode::Edge => e,
        FreiChenMode::Line => l,
    };
    (num / t).sqrt()
}

fn frei_chen(luma: &[u8], w: usize, h: usize, mode: FreiChenMode) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    let px = |x: i64, y: i64| -> i32 {
        let cx = x.clamp(0, w as i64 - 1) as usize;
        let cy = y.clamp(0, h as i64 - 1) as usize;
        luma[cy * w + cx] as i32
    };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let n = [
                [px(x - 1, y - 1), px(x, y - 1), px(x + 1, y - 1)],
                [px(x - 1, y), px(x, y), px(x + 1, y)],
                [px(x - 1, y + 1), px(x, y + 1), px(x + 1, y + 1)],
            ];
            let m = project_cosine(&n, mode);
            let q = (m * 255.0).round().clamp(0.0, 255.0) as u8;
            out[y as usize * w + x as usize] = q;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat;

    fn gray(w: u32, h: u32, pat: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pat(x, y));
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

    // ----- Algebraic identities on the basis itself ---------------
    //
    // These tests verify orthonormality directly from the projection
    // formula by constructing the 9 templates as neighbourhood patterns
    // and confirming that each template projects fully onto itself
    // (cosine to its own sub-space = 1) and is orthogonal to the other
    // sub-spaces.

    /// Build the integer entries of template `S_i` scaled by `divisor`
    /// (so each entry is exactly representable without fractions for
    /// `i ∈ {5..9}`; for `i ∈ {1..4}` the √2 entries are float-valued
    /// and we approximate them as the nearest integer ×100 — only used
    /// in tests that explicitly normalise).
    fn template_integer(i: usize) -> [[f64; 3]; 3] {
        let s2 = std::f64::consts::SQRT_2;
        match i {
            1 => [[1.0, s2, 1.0], [0.0, 0.0, 0.0], [-1.0, -s2, -1.0]],
            2 => [[1.0, 0.0, -1.0], [s2, 0.0, -s2], [1.0, 0.0, -1.0]],
            3 => [[0.0, -1.0, s2], [1.0, 0.0, -1.0], [-s2, 1.0, 0.0]],
            4 => [[s2, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -s2]],
            5 => [[0.0, 1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            6 => [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]],
            7 => [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            8 => [[-2.0, 1.0, -2.0], [1.0, 4.0, 1.0], [-2.0, 1.0, -2.0]],
            9 => [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            _ => panic!("template index out of range"),
        }
    }

    fn divisor(i: usize) -> f64 {
        match i {
            1..=4 => 2.0 * std::f64::consts::SQRT_2,
            5 | 6 => 2.0,
            7 | 8 => 6.0,
            9 => 3.0,
            _ => panic!("template index out of range"),
        }
    }

    /// Inner product `<A, B>` of two 3×3 patches.
    fn dot(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += a[i][j] * b[i][j];
            }
        }
        s
    }

    #[test]
    fn templates_are_pairwise_orthogonal() {
        // <Si/di, Sj/dj> = 0 for i != j (orthogonal basis); since the
        // divisors are scalar, equivalent to <Si, Sj> = 0.
        for i in 1..=9 {
            for j in (i + 1)..=9 {
                let ti = template_integer(i);
                let tj = template_integer(j);
                let p = dot(&ti, &tj);
                assert!(
                    p.abs() < 1e-9,
                    "templates S{} and S{} are not orthogonal: <Si,Sj> = {}",
                    i,
                    j,
                    p
                );
            }
        }
    }

    #[test]
    fn templates_are_unit_norm_after_divisor() {
        // <Si, Si> / di² = 1 (each template is unit-norm once the
        // overall scalar divisor is applied).
        for i in 1..=9 {
            let t = template_integer(i);
            let n2 = dot(&t, &t) / (divisor(i).powi(2));
            assert!(
                (n2 - 1.0).abs() < 1e-9,
                "template S{} not unit-norm: ||Si/di||² = {}",
                i,
                n2
            );
        }
    }

    // ----- Projection-level tests ---------------------------------

    #[test]
    fn flat_patch_yields_zero() {
        // A constant neighbourhood projects entirely onto S9 (the
        // mean kernel); edge and line metrics are both zero.
        let input = gray(5, 5, |_, _| 100);
        let out_edge = FreiChen::new().apply(&input, p_gray(5, 5)).unwrap();
        for v in &out_edge.planes[0].data {
            assert_eq!(*v, 0, "flat input should give zero edge response");
        }
        let out_line = FreiChen::line().apply(&input, p_gray(5, 5)).unwrap();
        for v in &out_line.planes[0].data {
            assert_eq!(*v, 0, "flat input should give zero line response");
        }
    }

    #[test]
    fn horizontal_step_gives_nonzero_edge_response_hand_derived() {
        // 3×3 neighbourhood: top two rows = 200, bottom row = 0.
        // Hand-derive each projection at the centre pixel (1, 1):
        //   c1 = (200 + √2·200 + 200 - 0 - 0 - 0) / (2√2)
        //      = (400 + 200√2) / (2√2) = 100√2 + 100 ≈ 241.421
        //   c2 = (200 - 200 + √2·200 - √2·200 + 0 - 0) / (2√2) = 0
        //   c3 = (-200 + √2·200 + 200 - 200 - 0 + 0) / (2√2)
        //      = (-200 + 200√2) / (2√2) = 100 - 50√2 ≈ 29.289
        //   c4 = (√2·200 - 200 - 200 + 200 + 0 - 0) / (2√2)
        //      = (200√2 - 200) / (2√2) = 100 - 50√2 ≈ 29.289
        //   c5 = (200 - 200 - 200 + 0) / 2 = -100
        //   c6 = (-200 + 200 + 0 - 0) / 2 = 0
        //   c7 = (200 - 400 + 200 - 400 + 800 - 400 + 0 - 0 + 0) / 6 = 0
        //   c8 = (-400 + 200 - 400 + 200 + 800 + 200 + 0 + 0 + 0) / 6 = 100
        //   c9 = (200·6 + 0) / 3 = 400
        //   E = 241.421² + 0 + 29.289² + 29.289² ≈ 60000
        //   L = 100² + 0 + 0 + 100² = 20000
        //   T = E + L + 400² = 60000 + 20000 + 160000 = 240000
        //   edge cosine = sqrt(60000 / 240000) = 0.5  → 128
        //   line cosine = sqrt(20000 / 240000) ≈ 0.2887 → 74
        let input = gray(3, 3, |_, y| if y == 2 { 0 } else { 200 });
        let out_e = FreiChen::new().apply(&input, p_gray(3, 3)).unwrap();
        let out_l = FreiChen::line().apply(&input, p_gray(3, 3)).unwrap();
        let e = out_e.planes[0].data[4];
        let l = out_l.planes[0].data[4];
        // Allow ±1 LSB for f64-rounding slack.
        assert!((127..=129).contains(&e), "expected edge ≈128, got {}", e);
        assert!((73..=75).contains(&l), "expected line ≈74, got {}", l);
    }

    #[test]
    fn mean_zero_horizontal_step_is_pure_edge() {
        // The horizontal step becomes a "pure edge" only after the DC
        // is removed. Constructing the neighbourhood directly:
        //   [[+100, +100, +100], [0, 0, 0], [-100, -100, -100]]
        // gives c5..c9 all zero and projects entirely onto S1, so the
        // edge cosine is exactly 1.0 → 255.
        let n: [[i32; 3]; 3] = [[100, 100, 100], [0, 0, 0], [-100, -100, -100]];
        let m = project_cosine(&n, FreiChenMode::Edge);
        assert!(
            (m - 1.0).abs() < 1e-9,
            "mean-zero step is not a pure edge: cos = {}",
            m
        );
        let l = project_cosine(&n, FreiChenMode::Line);
        assert!(l < 1e-9, "mean-zero step has line content: cos = {}", l);
    }

    #[test]
    fn s2_template_projects_purely_onto_edge_subspace() {
        // Plugging S2 itself as the neighbourhood — its projection onto
        // S2 is ||S2||² / d2 = 8 / (2√2) = 2√2 ≈ 2.828, and onto every
        // other Si is zero (orthogonality). Sub-space mass therefore
        // sits entirely in E, so the edge cosine is exactly 1.
        let n: [[i32; 3]; 3] = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]; // ≈ S2·√2 (integer-encoded)
        let _ = n; // we'll test the more general property below
                   // Use a clearly-integer mean-zero pattern aligned to S2 axes:
        let n: [[i32; 3]; 3] = [[40, 0, -40], [40, 0, -40], [40, 0, -40]];
        let m = project_cosine(&n, FreiChenMode::Edge);
        assert!(
            m > 0.99,
            "S2-shaped patch is not registered as a pure edge: cos = {}",
            m
        );
    }

    #[test]
    fn discrete_laplacian_pattern_is_line_not_edge() {
        // A pattern aligned with S7 (discrete Laplacian; centre +4,
        // 4-connected neighbours -2, diagonals +1) at the centre pixel:
        //   [[60, 0, 60], [0, 120, 0], [60, 0, 60]]
        // Hand-derive:
        //   c1..c4 = 0 (symmetric — no gradient component)
        //   c5 = c6 = 0; c8 = 0; c7 = 120; c9 = 120
        //   E = 0, L = 120² = 14400, T = 14400 + 120² = 28800
        //   edge cosine = 0
        //   line cosine = sqrt(14400 / 28800) = sqrt(1/2) ≈ 0.7071
        //              → round(0.7071 · 255) = 180
        let pattern = [[60u8, 0, 60], [0, 120, 0], [60, 0, 60]];
        let input = gray(3, 3, |x, y| pattern[y as usize][x as usize]);
        let out_edge = FreiChen::new().apply(&input, p_gray(3, 3)).unwrap();
        let out_line = FreiChen::line().apply(&input, p_gray(3, 3)).unwrap();
        let e = out_edge.planes[0].data[4];
        let l = out_line.planes[0].data[4];
        assert_eq!(
            e, 0,
            "Laplacian pattern is gradient-free — edge should be 0"
        );
        assert!(
            (179..=181).contains(&l),
            "expected line ≈180 (sqrt(1/2) · 255), got {}",
            l
        );
    }

    #[test]
    fn edge_plus_line_squared_le_one() {
        // E² + L² should never exceed T² (Pythagorean: E + L + p9² =
        // T), so the two cosines satisfy cos²(edge) + cos²(line) ≤ 1.
        // Spot-check at every pixel of a diagonal ramp.
        let input = gray(16, 16, |x, y| ((x + y) * 8) as u8);
        let e = FreiChen::new().apply(&input, p_gray(16, 16)).unwrap();
        let l = FreiChen::line().apply(&input, p_gray(16, 16)).unwrap();
        for (ev, lv) in e.planes[0].data.iter().zip(l.planes[0].data.iter()) {
            let ef = (*ev as f64) / 255.0;
            let lf = (*lv as f64) / 255.0;
            assert!(
                ef * ef + lf * lf <= 1.0 + 1e-9,
                "cos²(edge) + cos²(line) > 1: edge={}, line={}",
                ev,
                lv
            );
        }
    }

    #[test]
    fn rgb_input_emits_gray8() {
        let data: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = FreiChen::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Rgb24,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data.len(), 16);
    }

    #[test]
    fn yuv420_uses_luma_only() {
        // 4×4 YUV 4:2:0: luma plane is a 2-column step; chroma is flat
        // 128. The detector should react to the luma step.
        let mut y = Vec::with_capacity(16);
        for _row in 0..4 {
            for col in 0..4 {
                y.push(if col < 2 { 0 } else { 220 });
            }
        }
        let u = vec![128u8; 4];
        let v = vec![128u8; 4];
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane { stride: 2, data: u },
                VideoPlane { stride: 2, data: v },
            ],
        };
        let out = FreiChen::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuv420P,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        // Look at the centre column (x=2) — the column right after the
        // step. Edge magnitude there should be high.
        let v = out.planes[0].data[4 + 2];
        assert!(v > 100, "expected non-trivial edge response, got {}", v);
    }

    #[test]
    fn pts_pass_through() {
        let input = gray(3, 3, |x, _| if x == 2 { 50 } else { 0 });
        let mut frame = input;
        frame.pts = Some(7);
        let out = FreiChen::new().apply(&frame, p_gray(3, 3)).unwrap();
        assert_eq!(out.pts, Some(7));
    }

    #[test]
    fn one_by_one_is_zero() {
        let input = gray(1, 1, |_, _| 200);
        let out = FreiChen::new().apply(&input, p_gray(1, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![0]);
    }

    #[test]
    fn with_mode_setter_swaps_mode() {
        // Use the Laplacian-style pattern again; line metric should be
        // greater than edge metric.
        let pattern = [[60u8, 0, 60], [0, 120, 0], [60, 0, 60]];
        let input = gray(3, 3, |x, y| pattern[y as usize][x as usize]);
        let f = FreiChen::new().with_mode(FreiChenMode::Line);
        let line = f.apply(&input, p_gray(3, 3)).unwrap();
        let edge = FreiChen::new().apply(&input, p_gray(3, 3)).unwrap();
        assert!(line.planes[0].data[4] > edge.planes[0].data[4]);
    }

    #[test]
    fn rejects_unsupported_format() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0; 16],
            }],
        };
        let err = FreiChen::new()
            .apply(
                &frame,
                VideoStreamParams {
                    format: PixelFormat::Nv12,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(format!("{err}").contains("FreiChen"));
    }
}
