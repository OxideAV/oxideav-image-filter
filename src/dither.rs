//! Bit-depth-reduction dither: Bayer ordered-dither + the classic
//! error-diffusion kernel family (Floyd–Steinberg, Jarvis–Judice–Ninke,
//! Stucki, Sierra-3 / Sierra-2 / Sierra-Lite, Atkinson).
//!
//! Clean-room transcription of the matrices and stencils tabulated in
//! `docs/image/filter/dithering-kernels.md`, which itself is sourced from
//! the original publications: Bayer 1973 (IEEE ICC), Floyd & Steinberg
//! 1976 (SID), Jarvis–Judice–Ninke 1976 (CGIP), Stucki 1981 (IBM
//! RZ1060), the Sierra family (community-attributed, c. 1989–1990), and
//! Atkinson's late-1980s Macintosh 1-bit dither. The kernel coefficients
//! are uncopyrightable published facts (cf. *Feist v. Rural*); no prose
//! from any source is reproduced and nothing here is derived from any
//! image-library source tree.
//!
//! # What the filter does
//!
//! Quantises every "tone" sample (grey / R / G / B; luma of YUV) to one
//! of `levels` evenly-spaced output values at the picture's own depth —
//! 8-bit, 16-bit and float pictures run the same binary32 kernel, a
//! 16-bit picture's output levels are 16-bit codes — applying
//! either:
//!
//! * an **ordered dither** with a tiled Bayer threshold map (2×2, 4×4,
//!   8×8 or 16×16), or
//! * an **error-diffusion** kernel that pushes the per-pixel
//!   quantisation residue forward to not-yet-quantised neighbours along
//!   a left-to-right raster scan — or, optionally, a **serpentine
//!   (boustrophedon)** scan that alternates row direction and mirrors
//!   the stencil horizontally on the reversed rows (`docs/image/filter/
//!   dithering-kernels.md` §1.3). Serpentine scanning breaks up the
//!   directional "worm"/hysteresis artifacts a pure raster scan
//!   produces, at no extra cost; the kernel coefficients are unchanged.
//!
//! With `levels = 2` (the default) the result is a 1-bit black-and-white
//! halftone — the classic use case. Higher `levels` give multi-tone
//! posterise-with-dither output: the colour count is reduced without
//! visible banding.
//!
//! # Pixel format handling
//!
//! * `Gray8` / `Rgb24` / `Rgba` — every tone channel is dithered
//!   independently; alpha on `Rgba` is preserved verbatim.
//! * `Yuv420P` / `Yuv422P` / `Yuv444P` — only the luma (Y) plane is
//!   dithered; chroma planes are left untouched so hue/saturation stay
//!   put. Matches the behaviour of every other "tone" filter in this
//!   crate (`Posterize`, `Gamma`, `BrightnessContrast`, …).
//!
//! * `GrayF32Le` / `RgbF32Le` / `RgbaF32Le` natively, and through them
//!   every 16-bit layout (`Gray16Le`, `Rgb48Le`, `Rgba64Le`, deep
//!   planar YUV / GBR): no 8-bit round trip, alpha untouched.
//!
//! Every other layout arrives through the routes of the crate's
//! wrapper.

use crate::{PixelFilter, VideoStreamParams, is_supported_format};
use oxideav_core::{Error, PixelFormat, VideoFrame};

// ----------------------------------------------------------------------
// Bayer ordered-dither threshold maps
// ----------------------------------------------------------------------
//
// `BAYER_*` are the recursively-constructed Bayer index matrices from
// the original publication (Bayer 1973). Each entry is in the range
// `0 ..= n² - 1`; to use them as a fractional threshold, divide by `n²`
// (the matrix has the textbook half-pixel `+0.5` offset folded in by
// the caller — see `bayer_threshold_q8`).

/// Bayer 2×2 index matrix.
#[rustfmt::skip]
const BAYER_2: [[u8; 2]; 2] = [
    [0, 2],
    [3, 1],
];

/// Bayer 4×4 index matrix.
#[rustfmt::skip]
const BAYER_4: [[u8; 4]; 4] = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
];

/// Bayer 8×8 index matrix.
///
/// Constructed by the standard `M_2n = 4·M_n + recurrence` rule
/// (`docs/image/filter/dithering-kernels.md` §2).
#[rustfmt::skip]
const BAYER_8: [[u8; 8]; 8] = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// Bayer 16×16 index matrix.
///
/// One more turn of the `M_2n = 4·M_n + recurrence` crank past
/// [`BAYER_8`] (`docs/image/filter/dithering-kernels.md` §2). Entries
/// span `0 ..= 255` (a full permutation), giving the finest
/// dispersed-dot ordered dither — 256 distinct thresholds tiled across
/// the 16×16 cell, so smooth gradients reproduce with the least
/// visible cross-hatch texture of the family.
#[rustfmt::skip]
const BAYER_16: [[u8; 16]; 16] = [
    [  0,128, 32,160,  8,136, 40,168,  2,130, 34,162, 10,138, 42,170],
    [192, 64,224, 96,200, 72,232,104,194, 66,226, 98,202, 74,234,106],
    [ 48,176, 16,144, 56,184, 24,152, 50,178, 18,146, 58,186, 26,154],
    [240,112,208, 80,248,120,216, 88,242,114,210, 82,250,122,218, 90],
    [ 12,140, 44,172,  4,132, 36,164, 14,142, 46,174,  6,134, 38,166],
    [204, 76,236,108,196, 68,228,100,206, 78,238,110,198, 70,230,102],
    [ 60,188, 28,156, 52,180, 20,148, 62,190, 30,158, 54,182, 22,150],
    [252,124,220, 92,244,116,212, 84,254,126,222, 94,246,118,214, 86],
    [  3,131, 35,163, 11,139, 43,171,  1,129, 33,161,  9,137, 41,169],
    [195, 67,227, 99,203, 75,235,107,193, 65,225, 97,201, 73,233,105],
    [ 51,179, 19,147, 59,187, 27,155, 49,177, 17,145, 57,185, 25,153],
    [243,115,211, 83,251,123,219, 91,241,113,209, 81,249,121,217, 89],
    [ 15,143, 47,175,  7,135, 39,167, 13,141, 45,173,  5,133, 37,165],
    [207, 79,239,111,199, 71,231,103,205, 77,237,109,197, 69,229,101],
    [ 63,191, 31,159, 55,183, 23,151, 61,189, 29,157, 53,181, 21,149],
    [255,127,223, 95,247,119,215, 87,253,125,221, 93,245,117,213, 85],
];

/// The Bayer threshold deviation at pixel `(x, y)`: cell index `m` of
/// an `n²`-entry map is the fractional threshold `(m + 0.5) / n²`
/// (`docs/image/filter/dithering-kernels.md` §2); the value returned is
/// its distance from the middle, `(m + 0.5) / n² − 0.5`, in
/// `(−0.5, 0.5)`.
fn bayer_deviation(matrix: BayerMatrix, x: usize, y: usize) -> f32 {
    let (m, cells) = match matrix {
        BayerMatrix::M2 => (BAYER_2[y & 1][x & 1], 4.0),
        BayerMatrix::M4 => (BAYER_4[y & 3][x & 3], 16.0),
        BayerMatrix::M8 => (BAYER_8[y & 7][x & 7], 64.0),
        BayerMatrix::M16 => (BAYER_16[y & 15][x & 15], 256.0),
    };
    (m as f32 + 0.5) / cells - 0.5
}

/// Which Bayer matrix size the ordered dither tiles with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BayerMatrix {
    /// 2×2 — the coarsest dither, very visible structure.
    M2,
    /// 4×4 — the classic 16-level Bayer dither.
    #[default]
    M4,
    /// 8×8 — 64-level dither, the standard choice for 1-bit output.
    M8,
    /// 16×16 — 256-level dither, the finest dispersed-dot map; smooth
    /// gradients show the least cross-hatch texture.
    M16,
}

// ----------------------------------------------------------------------
// Error-diffusion kernels
// ----------------------------------------------------------------------

/// Which error-diffusion kernel the filter pushes residues through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DiffusionKernel {
    /// Floyd–Steinberg, ÷16 (Floyd & Steinberg 1976).
    #[default]
    FloydSteinberg,
    /// Jarvis–Judice–Ninke, ÷48 (Jarvis et al. 1976).
    Jjn,
    /// Stucki, ÷42 (Stucki 1981).
    Stucki,
    /// Sierra-3 (three-row), ÷32.
    Sierra3,
    /// Sierra-2 (two-row), ÷16.
    Sierra2,
    /// Sierra-Lite / "Filter Lite" / Sierra-2-4A, ÷4.
    SierraLite,
    /// Atkinson, ÷8 with coefficient sum 6 — 2/8 of the error is
    /// intentionally discarded (Atkinson, Apple, late 1980s).
    Atkinson,
}

/// Scan order for the error-diffusion path (`docs/image/filter/
/// dithering-kernels.md` §1.3).
///
/// The kernel coefficients are identical for both orders; only the
/// per-row traversal direction (and, on the reversed rows, the
/// horizontal mirroring of the stencil) differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ScanOrder {
    /// Every scanline is processed left→right (top→bottom). The stencil
    /// is used exactly as drawn. This is the classic ordering and the
    /// default, preserving the pre-existing behaviour.
    #[default]
    Raster,
    /// Boustrophedon: odd scanlines go left→right, even scanlines go
    /// right→left with the stencil **mirrored horizontally** (every
    /// `dx` negated). Recommended by Ulichney to suppress the
    /// directional "worm"/hysteresis artifacts a pure raster scan
    /// produces. Has no effect on the [`DitherMode::Ordered`] (Bayer)
    /// path, which is a feedback-free point process whose result is
    /// scan-order-independent.
    Serpentine,
}

/// One entry in a diffusion stencil: `(dx, dy, weight)`.
///
/// `dy = 0` is the current row, `dy > 0` is below. `dx > 0` is right
/// of the current pixel, `dx < 0` is left. The current pixel itself
/// (`dx = 0`, `dy = 0`) never appears — the stencil only lists
/// neighbours that receive a fraction of the error.
type DiffusionStencil = &'static [(i32, i32, i32)];

/// Stencil + divisor for one diffusion kernel.
struct KernelTable {
    stencil: DiffusionStencil,
    divisor: i32,
}

// Floyd–Steinberg: ÷16, neighbours (+1,0)=7  (-1,+1)=3  (0,+1)=5  (+1,+1)=1.
const KERN_FS: KernelTable = KernelTable {
    stencil: &[(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)],
    divisor: 16,
};

// Jarvis–Judice–Ninke: ÷48.
//   *  7  5
// 3 5  7 5 3
// 1 3  5 3 1
const KERN_JJN: KernelTable = KernelTable {
    stencil: &[
        (1, 0, 7),
        (2, 0, 5),
        (-2, 1, 3),
        (-1, 1, 5),
        (0, 1, 7),
        (1, 1, 5),
        (2, 1, 3),
        (-2, 2, 1),
        (-1, 2, 3),
        (0, 2, 5),
        (1, 2, 3),
        (2, 2, 1),
    ],
    divisor: 48,
};

// Stucki: ÷42.
//   *  8  4
// 2 4  8 4 2
// 1 2  4 2 1
const KERN_STUCKI: KernelTable = KernelTable {
    stencil: &[
        (1, 0, 8),
        (2, 0, 4),
        (-2, 1, 2),
        (-1, 1, 4),
        (0, 1, 8),
        (1, 1, 4),
        (2, 1, 2),
        (-2, 2, 1),
        (-1, 2, 2),
        (0, 2, 4),
        (1, 2, 2),
        (2, 2, 1),
    ],
    divisor: 42,
};

// Sierra-3 (three-row): ÷32.
//     *  5  3
// 2 4 5  4  2
//   2 3  2
const KERN_SIERRA3: KernelTable = KernelTable {
    stencil: &[
        (1, 0, 5),
        (2, 0, 3),
        (-2, 1, 2),
        (-1, 1, 4),
        (0, 1, 5),
        (1, 1, 4),
        (2, 1, 2),
        (-1, 2, 2),
        (0, 2, 3),
        (1, 2, 2),
    ],
    divisor: 32,
};

// Sierra-2 (two-row): ÷16.
//   *  4  3
// 1 2 3 2 1
const KERN_SIERRA2: KernelTable = KernelTable {
    stencil: &[
        (1, 0, 4),
        (2, 0, 3),
        (-2, 1, 1),
        (-1, 1, 2),
        (0, 1, 3),
        (1, 1, 2),
        (2, 1, 1),
    ],
    divisor: 16,
};

// Sierra-Lite: ÷4. Cheapest of the family.
//   * 2
// 1 1
const KERN_SIERRA_LITE: KernelTable = KernelTable {
    stencil: &[(1, 0, 2), (-1, 1, 1), (0, 1, 1)],
    divisor: 4,
};

// Atkinson: ÷8 with Σ = 6 (2/8 of the error is discarded).
//   * 1 1
// 1 1 1
//   1
const KERN_ATKINSON: KernelTable = KernelTable {
    stencil: &[
        (1, 0, 1),
        (2, 0, 1),
        (-1, 1, 1),
        (0, 1, 1),
        (1, 1, 1),
        (0, 2, 1),
    ],
    divisor: 8,
};

fn kernel_table(k: DiffusionKernel) -> KernelTable {
    match k {
        DiffusionKernel::FloydSteinberg => KERN_FS,
        DiffusionKernel::Jjn => KERN_JJN,
        DiffusionKernel::Stucki => KERN_STUCKI,
        DiffusionKernel::Sierra3 => KERN_SIERRA3,
        DiffusionKernel::Sierra2 => KERN_SIERRA2,
        DiffusionKernel::SierraLite => KERN_SIERRA_LITE,
        DiffusionKernel::Atkinson => KERN_ATKINSON,
    }
}

// ----------------------------------------------------------------------
// Filter mode + struct
// ----------------------------------------------------------------------

/// Which dithering algorithm the filter runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DitherMode {
    /// Tiled Bayer ordered dither — no state, perfectly parallel,
    /// produces the characteristic crosshatch texture.
    Ordered(BayerMatrix),
    /// Error-diffusion dither — pushes the per-pixel residue forward
    /// along a left-to-right raster scan into the kernel's neighbours.
    /// Produces a more naturalistic, less-structured result than
    /// ordered dither at the cost of a serial dependency along the
    /// scan order.
    ErrorDiffusion(DiffusionKernel),
}

impl Default for DitherMode {
    fn default() -> Self {
        DitherMode::ErrorDiffusion(DiffusionKernel::FloydSteinberg)
    }
}

/// Quantise + dither each tone channel to `levels` evenly-spaced output
/// values.
///
/// * `mode` — ordered Bayer or one of the error-diffusion kernels.
///   Defaults to Floyd–Steinberg.
/// * `levels` — number of output values per channel (≥ 2). Clamped up
///   to `2` to keep at least black + white. With `levels = 2` the
///   output is the classic 1-bit halftone; higher values give multi-
///   tone dithered posterisation.
#[derive(Clone, Copy, Debug)]
pub struct Dither {
    mode: DitherMode,
    levels: u32,
    scan: ScanOrder,
}

impl Default for Dither {
    fn default() -> Self {
        Self::new()
    }
}

impl Dither {
    /// Default Floyd–Steinberg at `levels = 2` (1-bit B&W halftone),
    /// raster scan.
    pub fn new() -> Self {
        Self {
            mode: DitherMode::default(),
            levels: 2,
            scan: ScanOrder::Raster,
        }
    }

    /// Bayer ordered dither with the given matrix size at `levels = 2`.
    pub fn ordered(matrix: BayerMatrix) -> Self {
        Self {
            mode: DitherMode::Ordered(matrix),
            levels: 2,
            scan: ScanOrder::Raster,
        }
    }

    /// Error-diffusion dither with the given kernel at `levels = 2`,
    /// raster scan.
    pub fn error_diffusion(kernel: DiffusionKernel) -> Self {
        Self {
            mode: DitherMode::ErrorDiffusion(kernel),
            levels: 2,
            scan: ScanOrder::Raster,
        }
    }

    /// Override the output bit-depth. `levels = 2` is 1-bit B&W
    /// (the default); `levels = 4` is 2 bits per channel; etc. Clamped
    /// up to `2` — `levels = 1` would collapse the whole tone scale
    /// to a single flat colour.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels.max(2);
        self
    }

    /// Select the error-diffusion scan order (`docs/image/filter/
    /// dithering-kernels.md` §1.3). [`ScanOrder::Serpentine`] alternates
    /// row direction (mirroring the stencil on reversed rows) to break
    /// up directional artifacts; it has no effect on the ordered (Bayer)
    /// mode, which is scan-order-independent. Defaults to
    /// [`ScanOrder::Raster`].
    pub fn with_scan(mut self, scan: ScanOrder) -> Self {
        self.scan = scan;
        self
    }

    /// Returns the dither's mode.
    pub fn mode(&self) -> DitherMode {
        self.mode
    }

    /// Returns the error-diffusion scan order.
    pub fn scan(&self) -> ScanOrder {
        self.scan
    }

    /// Returns the output bit-depth (in distinct levels per channel).
    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Dither channels `0..nc` of a tight `width × height` picture of
    /// `ch` interleaved normalised samples (`1.0` = white) in place;
    /// channels `nc..ch` (alpha) are untouched. One kernel for every
    /// depth: 8-bit pictures are widened to it, 16-bit and float
    /// pictures run on it directly.
    ///
    /// A sample `v` is quantised to the nearest of `levels` evenly
    /// spaced values, `q(v) = round(clamp(v, 0, 1) · (L − 1)) / (L − 1)`.
    ///
    /// * Ordered: `out = q(v + d / (L − 1))` with `d` the cell's
    ///   threshold deviation in `(−½, ½)` — the deviation is scaled to
    ///   the spacing of the output levels, so black stays black and
    ///   white stays white at any level count.
    /// * Error diffusion: `out = q(v + e)` with `e` the error diffused
    ///   onto the pixel so far; the residue `(v + e) − out` is pushed
    ///   onto the stencil's not-yet-visited neighbours with weights
    ///   `w / divisor`, in binary32 (nothing is truncated on the way).
    ///   Error leaving the picture is dropped. A non-finite sample
    ///   quantises as it clamps (NaN to black) and diffuses nothing.
    fn dither_samples(&self, data: &mut [f32], width: usize, height: usize, ch: usize, nc: usize) {
        if width == 0 || height == 0 {
            return;
        }
        let span = (self.levels.max(2) - 1) as f32;
        let quantise = |v: f32| -> f32 {
            if v.is_nan() {
                return 0.0;
            }
            (v.clamp(0.0, 1.0) * span + 0.5).floor() / span
        };
        let row_len = width * ch;
        match self.mode {
            DitherMode::Ordered(matrix) => {
                // A point process: rows are independent.
                crate::exec::par_rows(data, row_len, |y, row| {
                    for (x, px) in row.chunks_exact_mut(ch).enumerate() {
                        let bias = bayer_deviation(matrix, x, y) / span;
                        for v in &mut px[..nc] {
                            *v = quantise(*v + bias);
                        }
                    }
                });
            }
            DitherMode::ErrorDiffusion(kernel) => {
                let table = kernel_table(kernel);
                let inv = 1.0 / table.divisor as f32;
                // The stencils reach two columns either side and two
                // rows down: three error rows, padded by two columns so
                // no tap needs a bounds test, rotated as the scan
                // descends.
                const PAD: usize = 2;
                let padded = width + 2 * PAD;
                let mut err = vec![0.0f32; 3 * padded];
                for c in 0..nc {
                    err.fill(0.0);
                    for y in 0..height {
                        // §1.3 serpentine: odd rows run right→left with
                        // the stencil mirrored, so error still flows
                        // into pixels the scan has not reached.
                        let reverse = matches!(self.scan, ScanOrder::Serpentine) && (y % 2 == 1);
                        let base = (y % 3) * padded;
                        for col in 0..width {
                            let x = if reverse { width - 1 - col } else { col };
                            let i = y * row_len + x * ch + c;
                            let v = data[i] + err[base + PAD + x];
                            let q = quantise(v);
                            data[i] = q;
                            let e = v - q;
                            if !e.is_finite() {
                                continue;
                            }
                            for &(dx, dy, w) in table.stencil {
                                let sdx = if reverse { -dx } else { dx };
                                let row = ((y + dy as usize) % 3) * padded;
                                let at = (row + PAD + x).wrapping_add_signed(sdx as isize);
                                err[at] += e * (w as f32 * inv);
                            }
                        }
                        // This row's slot becomes row `y + 3`'s.
                        err[base..base + padded].fill(0.0);
                    }
                }
            }
        }
    }
}

impl PixelFilter for Dither {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::CLASSIC_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if crate::deep::is_float(params.format) {
            let me = *self;
            return crate::deep::run(input, params, move |mut img| {
                let (ch, nc) = (img.channels(), img.color_channels());
                me.dither_samples(img.samples_mut(), w, h, ch, nc);
                Ok(img)
            });
        }
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Dither does not yet handle {:?}",
                params.format
            )));
        }
        // 8-bit: the tone plane (the packed picture, or luma on YUV —
        // chroma is left alone) is widened to `v / 255`, dithered by the
        // same kernel and rounded back; the output levels land on
        // `round(255 · k / (L − 1))`.
        let (ch, nc) = match params.format {
            PixelFormat::Rgb24 => (3, 3),
            PixelFormat::Rgba => (4, 3),
            _ => (1, 1),
        };
        let mut out = input.clone();
        let plane = &mut out.planes[0];
        let row = w * ch;
        let mut samples = Vec::with_capacity(row * h);
        for y in 0..h {
            samples.extend(
                plane.data[y * plane.stride..y * plane.stride + row]
                    .iter()
                    .map(|&b| b as f32 / 255.0),
            );
        }
        self.dither_samples(&mut samples, w, h, ch, nc);
        for y in 0..h {
            let dst = &mut plane.data[y * plane.stride..y * plane.stride + row];
            for (d, s) in dst
                .chunks_exact_mut(ch)
                .zip(samples[y * row..(y + 1) * row].chunks_exact(ch))
            {
                for c in 0..nc {
                    d[c] = (s[c] * 255.0 + 0.5) as u8;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_dither_on_tight_packed_rows_touches_every_channel() {
        // A tightly-strided Rgb24 / Rgba picture: the last row's green,
        // blue and alpha channels must be reachable without slicing
        // past the buffer.
        for (fmt, bpp) in [(PixelFormat::Rgb24, 3usize), (PixelFormat::Rgba, 4)] {
            let (w, h) = (5usize, 3usize);
            let input = VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: w * bpp,
                    data: (0..w * h * bpp).map(|i| (i * 37 % 256) as u8).collect(),
                }],
            };
            let out = Dither::ordered(BayerMatrix::M4)
                .with_levels(2)
                .apply(
                    &input,
                    VideoStreamParams {
                        format: fmt,
                        width: w as u32,
                        height: h as u32,
                    },
                )
                .unwrap();
            assert_eq!(out.planes[0].data.len(), w * h * bpp);
            // Two-level quantisation: every colour byte is 0 or 255.
            for px in out.planes[0].data.chunks_exact(bpp) {
                for &c in &px[..3] {
                    assert!(c == 0 || c == 255, "{fmt:?}: {px:?}");
                }
            }
        }
    }
    use crate::ImageFilter;
    use oxideav_core::VideoPlane;

    fn gray(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
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

    fn params(w: u32, h: u32, fmt: PixelFormat) -> VideoStreamParams {
        VideoStreamParams {
            format: fmt,
            width: w,
            height: h,
        }
    }

    // ---- Bayer matrix self-checks ----------------------------------

    #[test]
    fn bayer_2_is_the_self_similar_seed() {
        // M2 must be {0, 2, 3, 1} — the recurrence's own constants.
        assert_eq!(BAYER_2, [[0, 2], [3, 1]]);
    }

    #[test]
    fn bayer_4_recurrence_matches_doc() {
        // Verifies the published 4×4 matrix entry-for-entry.
        let expected: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
        assert_eq!(BAYER_4, expected);
    }

    #[test]
    fn bayer_4_recurrence_derives_from_bayer_2() {
        // From `docs/image/filter/dithering-kernels.md` §2 the recurrence
        // is `M_2n = 4·M_n + M_2[hi, lo]` where the four constants in
        // the 2×2 block layout are the flattened M2. The high-block
        // index `(i / 2, j / 2)` therefore picks the *constant* (i.e.
        // a row-major-flattened M2 entry), and the low-block index
        // `(i mod 2, j mod 2)` picks the entry of the smaller matrix.
        for i in 0..4 {
            for j in 0..4 {
                let hi = BAYER_2[i / 2][j / 2] as u32;
                let lo = BAYER_2[i % 2][j % 2] as u32;
                let expected = 4 * lo + hi;
                assert_eq!(
                    BAYER_4[i][j] as u32, expected,
                    "Bayer recurrence broken at ({i},{j})",
                );
            }
        }
    }

    #[test]
    fn bayer_8_is_a_permutation_of_0_to_63() {
        let mut seen = [false; 64];
        for row in BAYER_8 {
            for v in row {
                assert!(v < 64, "Bayer 8 entry out of range: {v}");
                assert!(!seen[v as usize], "Bayer 8 duplicate: {v}");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn bayer_16_is_a_permutation_of_0_to_255() {
        let mut seen = [false; 256];
        for row in BAYER_16 {
            for v in row {
                assert!(!seen[v as usize], "Bayer 16 duplicate: {v}");
                seen[v as usize] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "Bayer 16 is not a full permutation"
        );
    }

    #[test]
    fn bayer_16_recurrence_derives_from_bayer_8() {
        // `docs/image/filter/dithering-kernels.md` §2: one more turn of
        // the `M_2n = 4·M_n + M_2[hi, lo]` crank. The high-block index
        // `(i / 8, j / 8)` selects the flattened-M2 constant and the
        // low-block index `(i mod 8, j mod 8)` indexes the 8×8 map.
        for i in 0..16 {
            for j in 0..16 {
                let hi = BAYER_2[i / 8][j / 8] as u32;
                let lo = BAYER_8[i % 8][j % 8] as u32;
                let expected = 4 * lo + hi;
                assert_eq!(
                    BAYER_16[i][j] as u32, expected,
                    "Bayer 16 recurrence broken at ({i},{j})",
                );
            }
        }
    }

    #[test]
    fn bayer_deviations_are_the_centred_cell_thresholds() {
        // Each of the n² cells maps to the distinct deviation
        // `(m + 0.5) / n² − 0.5`: symmetric about zero, strictly inside
        // (−½, ½), summing to zero over a tile.
        for (matrix, n) in [
            (BayerMatrix::M2, 2usize),
            (BayerMatrix::M4, 4),
            (BayerMatrix::M8, 8),
            (BayerMatrix::M16, 16),
        ] {
            let cells = (n * n) as f32;
            let mut all: Vec<f32> = Vec::new();
            for y in 0..n {
                for x in 0..n {
                    // The map tiles.
                    assert_eq!(
                        bayer_deviation(matrix, x, y),
                        bayer_deviation(matrix, x + n, y + 3 * n)
                    );
                    all.push(bayer_deviation(matrix, x, y));
                }
            }
            all.sort_by(f32::total_cmp);
            for (m, d) in all.iter().enumerate() {
                assert_eq!(*d, (m as f32 + 0.5) / cells - 0.5, "{matrix:?} cell {m}");
            }
            assert_eq!(all[0], 0.5 / cells - 0.5);
            assert_eq!(all[n * n - 1], 0.5 - 0.5 / cells);
            assert!(all.iter().sum::<f32>().abs() < 1e-5);
        }
    }

    // ---- Error-diffusion kernel divisor checks ---------------------

    #[test]
    fn floyd_steinberg_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::FloydSteinberg);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, table.divisor);
        assert_eq!(sum, 16);
    }

    #[test]
    fn jjn_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::Jjn);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, 48);
        assert_eq!(table.divisor, 48);
    }

    #[test]
    fn stucki_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::Stucki);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, 42);
        assert_eq!(table.divisor, 42);
    }

    #[test]
    fn sierra3_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::Sierra3);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, 32);
        assert_eq!(table.divisor, 32);
    }

    #[test]
    fn sierra2_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::Sierra2);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, 16);
        assert_eq!(table.divisor, 16);
    }

    #[test]
    fn sierra_lite_coefficient_sum_matches_divisor() {
        let table = kernel_table(DiffusionKernel::SierraLite);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, 4);
        assert_eq!(table.divisor, 4);
    }

    #[test]
    fn atkinson_divisor_is_8_but_coefficients_sum_to_6() {
        // Atkinson is the lone non-conserving kernel: divisor 8,
        // coefficients sum to 6 → 2/8 of the error is discarded.
        let table = kernel_table(DiffusionKernel::Atkinson);
        let sum: i32 = table.stencil.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(table.divisor, 8);
        assert_eq!(sum, 6);
    }

    // ---- Behavioural tests on Gray8 --------------------------------

    #[test]
    fn floyd_steinberg_two_levels_only_emits_0_or_255() {
        // Flat-ramp input: every pixel must end up as either 0 or 255
        // with the default `levels = 2`.
        let input = gray(8, 8, |x, _| (x * 32).min(255) as u8);
        let out = Dither::new()
            .apply(&input, params(8, 8, PixelFormat::Gray8))
            .unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "FS produced non-binary value {v}");
        }
    }

    #[test]
    fn floyd_steinberg_average_tracks_input_mean() {
        // Error-diffusion conserves the mean to within a few %
        // — that's the whole point. Use a 50 % flat field as the
        // hardest test (lots of 50/50 quantisation decisions).
        let input = gray(32, 32, |_, _| 128);
        let out = Dither::new()
            .apply(&input, params(32, 32, PixelFormat::Gray8))
            .unwrap();
        let sum: u32 = out.planes[0].data.iter().map(|&v| v as u32).sum();
        let mean = sum as f32 / (32.0 * 32.0);
        assert!((mean - 128.0).abs() < 6.0, "FS mean {mean} drifted off 128",);
    }

    #[test]
    fn atkinson_loses_more_error_than_floyd_steinberg() {
        // Atkinson diffuses only 6/8 of the error each pixel — the
        // remaining 2/8 is intentionally discarded. Floyd–Steinberg
        // conserves the full error. On the same flat-grey field the
        // *cumulative discarded magnitude* under Atkinson must
        // therefore be strictly larger than under FS (which is zero
        // by construction). The output mean for each is checked
        // against the input mean to confirm.
        let input = gray(32, 32, |x, _| (x * 8).min(255) as u8);
        let in_sum: u32 = input.planes[0].data.iter().map(|&v| v as u32).sum();
        let in_mean = in_sum as f32 / (32.0 * 32.0);
        let atk = Dither::error_diffusion(DiffusionKernel::Atkinson)
            .apply(&input, params(32, 32, PixelFormat::Gray8))
            .unwrap();
        let fs = Dither::new()
            .apply(&input, params(32, 32, PixelFormat::Gray8))
            .unwrap();
        let atk_mean =
            atk.planes[0].data.iter().map(|&v| v as u32).sum::<u32>() as f32 / (32.0 * 32.0);
        let fs_mean =
            fs.planes[0].data.iter().map(|&v| v as u32).sum::<u32>() as f32 / (32.0 * 32.0);
        // FS conserves the mean within ~5 codes; Atkinson is allowed
        // to drift further but must stay within the [0,255] range.
        assert!(
            (fs_mean - in_mean).abs() < 6.0,
            "FS mean {fs_mean} drifted from input {in_mean}",
        );
        // Atkinson's mean differs from FS's by a non-trivial margin
        // — it isn't an error-conserving kernel. We don't pin the
        // exact direction (it depends on the input distribution) but
        // we do require that it isn't bit-identical to FS.
        assert!(
            (atk_mean - fs_mean).abs() > 0.1,
            "Atkinson ({atk_mean}) and FS ({fs_mean}) produced identical means \
             — error discard isn't happening",
        );
    }

    #[test]
    fn ordered_dither_pure_black_stays_black() {
        let input = gray(8, 8, |_, _| 0);
        let out = Dither::ordered(BayerMatrix::M4)
            .apply(&input, params(8, 8, PixelFormat::Gray8))
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 0, "Bayer threshold mishandled pure black");
        }
    }

    #[test]
    fn ordered_dither_pure_white_stays_white() {
        let input = gray(8, 8, |_, _| 255);
        let out = Dither::ordered(BayerMatrix::M4)
            .apply(&input, params(8, 8, PixelFormat::Gray8))
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 255, "Bayer threshold mishandled pure white");
        }
    }

    #[test]
    fn ordered_dither_50_percent_mid_grey_dithers_to_balanced_tile() {
        // 50 % grey ordered-dithered with 2×2 Bayer = `[[0,2],[3,1]]`
        // → q8 thresholds are 32 / 159 / 96 / 223 at the four cells.
        // With levels = 2 the bias is `t - 127`, so cell-by-cell the
        // biased sample is 33 / 160 / 97 / 224. Two values land below
        // the midpoint (snap to 0) and two land above (snap to 255).
        // The result is therefore a balanced 50/50 tile — half black
        // half white — which is exactly what a 50 % flat field should
        // dither to.
        let input = gray(4, 4, |_, _| 128);
        let out = Dither::ordered(BayerMatrix::M2)
            .apply(&input, params(4, 4, PixelFormat::Gray8))
            .unwrap();
        let d = &out.planes[0].data;
        // Tile pattern (row 0 = matrix row 0 with thresholds 32, 159;
        // row 1 = matrix row 1 with thresholds 96, 223):
        //   row 0: t=32  → 33 → 0;    t=159 → 160 → 255
        //   row 1: t=223 → 224 → 255; t=96  → 97 → 0
        // i.e. a 2×2 checker, repeated.
        for y in 0..4 {
            if y % 2 == 0 {
                assert_eq!(d[y * 4], 0, "row {y} col 0");
                assert_eq!(d[y * 4 + 1], 255, "row {y} col 1");
                assert_eq!(d[y * 4 + 2], 0, "row {y} col 2");
                assert_eq!(d[y * 4 + 3], 255, "row {y} col 3");
            } else {
                assert_eq!(d[y * 4], 255, "row {y} col 0");
                assert_eq!(d[y * 4 + 1], 0, "row {y} col 1");
                assert_eq!(d[y * 4 + 2], 255, "row {y} col 2");
                assert_eq!(d[y * 4 + 3], 0, "row {y} col 3");
            }
        }
        // Half black + half white by count.
        let black = d.iter().filter(|&&v| v == 0).count();
        let white = d.iter().filter(|&&v| v == 255).count();
        assert_eq!(black, 8);
        assert_eq!(white, 8);
    }

    #[test]
    fn ordered_dither_is_stateless_across_invocations() {
        let input = gray(16, 16, |x, y| ((x * 16 + y * 16) & 0xff) as u8);
        let f = Dither::ordered(BayerMatrix::M4);
        let a = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        let b = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn error_diffusion_is_stateless_across_invocations() {
        // The internal error buffer is allocated per `apply` call.
        let input = gray(16, 16, |x, _| (x * 16) as u8);
        let f = Dither::error_diffusion(DiffusionKernel::Stucki);
        let a = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        let b = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }

    // ---- Multi-level dither ---------------------------------------

    #[test]
    fn four_level_dither_emits_only_quartile_codes() {
        // levels = 4 → output must come from {0, 85, 170, 255}.
        let input = gray(16, 16, |x, y| ((x * 16 + y * 16) & 0xff) as u8);
        let out = Dither::new()
            .with_levels(4)
            .apply(&input, params(16, 16, PixelFormat::Gray8))
            .unwrap();
        for v in &out.planes[0].data {
            assert!(
                matches!(*v, 0 | 85 | 170 | 255),
                "4-level dither produced {v}",
            );
        }
    }

    // ---- RGB / RGBA --------------------------------------------------

    #[test]
    fn rgba_dither_preserves_alpha() {
        let data: Vec<u8> = (0..16).flat_map(|i| [50u8, 150, 200, i as u8]).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        };
        let out = Dither::new()
            .apply(&input, params(4, 4, PixelFormat::Rgba))
            .unwrap();
        for i in 0..16 {
            let px = &out.planes[0].data[i * 4..i * 4 + 4];
            assert!(px[0] == 0 || px[0] == 255);
            assert!(px[1] == 0 || px[1] == 255);
            assert!(px[2] == 0 || px[2] == 255);
            assert_eq!(px[3], i as u8, "alpha preserved at pixel {i}");
        }
    }

    #[test]
    fn rgb_dither_treats_each_channel_independently() {
        // Solid R=128 G=0 B=255 — only R can dither, the other two
        // stay pinned.
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3 * 8,
                data: (0..64).flat_map(|_| [128u8, 0, 255]).collect(),
            }],
        };
        let out = Dither::new()
            .apply(&input, params(8, 8, PixelFormat::Rgb24))
            .unwrap();
        for px in out.planes[0].data.as_chunks::<3>().0 {
            assert!(px[0] == 0 || px[0] == 255, "R should dither, got {}", px[0]);
            assert_eq!(px[1], 0, "G channel should stay 0");
            assert_eq!(px[2], 255, "B channel should stay 255");
        }
    }

    // ---- YUV ---------------------------------------------------------

    #[test]
    fn yuv_dither_only_touches_luma() {
        let y: Vec<u8> = (0..16).map(|i| i as u8 * 16).collect();
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: 4, data: y },
                VideoPlane {
                    stride: 2,
                    data: vec![200u8; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![50u8; 4],
                },
            ],
        };
        let out = Dither::new()
            .apply(&input, params(4, 4, PixelFormat::Yuv420P))
            .unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "luma should be 1-bit, got {v}");
        }
        // Chroma must be untouched.
        assert_eq!(out.planes[1].data, vec![200u8; 4]);
        assert_eq!(out.planes[2].data, vec![50u8; 4]);
    }

    // ---- Unsupported formats ---------------------------------------

    #[test]
    fn unsupported_format_errors() {
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![0u8; 16],
            }],
        };
        let err = Dither::new()
            .apply(
                &input,
                VideoStreamParams {
                    format: PixelFormat::Yuyv422,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("Dither"));
    }

    // ---- Serpentine scan (§1.3) ------------------------------------

    #[test]
    fn scan_order_default_is_raster() {
        assert_eq!(Dither::new().scan(), ScanOrder::Raster);
        assert_eq!(
            Dither::error_diffusion(DiffusionKernel::Stucki).scan(),
            ScanOrder::Raster,
        );
        assert_eq!(Dither::ordered(BayerMatrix::M4).scan(), ScanOrder::Raster,);
        // The builder records the requested order.
        assert_eq!(
            Dither::new().with_scan(ScanOrder::Serpentine).scan(),
            ScanOrder::Serpentine,
        );
    }

    #[test]
    fn serpentine_first_row_matches_raster() {
        // Row 0 is even → left→right in both scan orders, and no prior
        // row has fed error into it, so the first scanline must be
        // byte-identical between raster and serpentine. (Lower rows can
        // and do differ because the row-1 traversal reverses.)
        let input = gray(24, 8, |x, y| ((x * 11 + y * 7) & 0xff) as u8);
        let raster = Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
            .apply(&input, params(24, 8, PixelFormat::Gray8))
            .unwrap();
        let serp = Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
            .with_scan(ScanOrder::Serpentine)
            .apply(&input, params(24, 8, PixelFormat::Gray8))
            .unwrap();
        assert_eq!(
            raster.planes[0].data[0..24],
            serp.planes[0].data[0..24],
            "first scanline must match before the scan ever reverses",
        );
    }

    #[test]
    fn serpentine_differs_from_raster_on_lower_rows() {
        // Once the scan reverses on odd rows the two orders diverge —
        // that divergence is the whole point of serpentine (it breaks
        // the directional "worm" the raster scan accumulates). A flat
        // field dithers to a symmetric checker that happens to coincide
        // under both scans, so use a smooth diagonal gradient whose
        // residue trail is genuinely direction-dependent.
        let input = gray(24, 24, |x, y| (((x + y) * 5) & 0xff) as u8);
        let raster = Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
            .apply(&input, params(24, 24, PixelFormat::Gray8))
            .unwrap();
        let serp = Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
            .with_scan(ScanOrder::Serpentine)
            .apply(&input, params(24, 24, PixelFormat::Gray8))
            .unwrap();
        assert_ne!(
            raster.planes[0].data, serp.planes[0].data,
            "serpentine should produce a different pattern than raster",
        );
    }

    #[test]
    fn serpentine_still_emits_only_two_levels() {
        // The quantiser is unchanged; serpentine only affects which
        // neighbour each residue lands in, so the output is still 1-bit.
        let input = gray(16, 16, |x, y| ((x * 16 + y * 9) & 0xff) as u8);
        let out = Dither::error_diffusion(DiffusionKernel::Jjn)
            .with_scan(ScanOrder::Serpentine)
            .apply(&input, params(16, 16, PixelFormat::Gray8))
            .unwrap();
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "serpentine produced non-binary {v}");
        }
    }

    #[test]
    fn serpentine_conserves_mean_for_conserving_kernel() {
        // Floyd–Steinberg conserves the full error regardless of scan
        // order, so a flat field's mean must still track the input mean.
        let input = gray(32, 32, |_, _| 128);
        let out = Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
            .with_scan(ScanOrder::Serpentine)
            .apply(&input, params(32, 32, PixelFormat::Gray8))
            .unwrap();
        let mean = out.planes[0].data.iter().map(|&v| v as u32).sum::<u32>() as f32 / (32.0 * 32.0);
        assert!(
            (mean - 128.0).abs() < 6.0,
            "serpentine FS mean {mean} drifted off 128",
        );
    }

    #[test]
    fn serpentine_has_no_effect_on_ordered_dither() {
        // Bayer is a feedback-free point process — scan order cannot
        // change its output (doc §1.3 / §2 normalisation note).
        let input = gray(16, 16, |x, y| ((x * 16 + y * 16) & 0xff) as u8);
        let raster = Dither::ordered(BayerMatrix::M4)
            .apply(&input, params(16, 16, PixelFormat::Gray8))
            .unwrap();
        let serp = Dither::ordered(BayerMatrix::M4)
            .with_scan(ScanOrder::Serpentine)
            .apply(&input, params(16, 16, PixelFormat::Gray8))
            .unwrap();
        assert_eq!(raster.planes[0].data, serp.planes[0].data);
    }

    #[test]
    fn serpentine_is_stateless_across_invocations() {
        let input = gray(16, 16, |x, _| (x * 16) as u8);
        let f = Dither::error_diffusion(DiffusionKernel::Stucki).with_scan(ScanOrder::Serpentine);
        let a = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        let b = f.apply(&input, params(16, 16, PixelFormat::Gray8)).unwrap();
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }
}
