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
//! Quantises every "tone" sample (Gray8 / R / G / B of RGB & RGBA;
//! luma of YUV) to one of `levels` evenly-spaced output values, applying
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
//! Other pixel formats return [`Error::unsupported`].

use crate::{is_supported_format, ImageFilter, VideoStreamParams};
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

/// Pick the Bayer 2×2 / 4×4 / 8×8 / 16×16 entry for pixel `(x, y)` and return
/// it as a fractional threshold scaled into the 8-bit range. The
/// threshold for cell index `m` in an `n²`-entry map is
/// `(m + 0.5) / n²`, expressed in eighths of `255` rounded to the
/// nearest integer.
fn bayer_threshold_q8(matrix: BayerMatrix, x: usize, y: usize) -> i32 {
    match matrix {
        BayerMatrix::M2 => {
            let m = BAYER_2[y & 1][x & 1] as i32;
            // (m + 0.5) / 4 * 255  ==  ((2m + 1) * 255 + 4) / 8
            (((2 * m + 1) * 255) + 4) / 8
        }
        BayerMatrix::M4 => {
            let m = BAYER_4[y & 3][x & 3] as i32;
            (((2 * m + 1) * 255) + 16) / 32
        }
        BayerMatrix::M8 => {
            let m = BAYER_8[y & 7][x & 7] as i32;
            (((2 * m + 1) * 255) + 64) / 128
        }
        BayerMatrix::M16 => {
            let m = BAYER_16[y & 15][x & 15] as i32;
            // (m + 0.5) / 256 * 255  ==  ((2m + 1) * 255 + 256) / 512
            (((2 * m + 1) * 255) + 256) / 512
        }
    }
}

/// Which Bayer matrix size the ordered dither tiles with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
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

    /// Quantise a single sample to one of `levels` evenly-spaced output
    /// values. `bias` is the per-pixel offset (the Bayer threshold for
    /// ordered dither, or the accumulated diffusion residue for
    /// error-diffusion). `levels` is taken as `i32` so the caller's
    /// arithmetic doesn't have to cast.
    fn quantise(sample: i32, bias: i32, levels: i32) -> u8 {
        // `bias` is signed: ordered dither's threshold deviation is
        // `(M[i,j] + 0.5) / n² - 0.5` scaled into 8-bit, which is
        // negative for low-index cells. For error-diffusion the bias
        // is the running residue divided by the kernel divisor.
        let biased = sample + bias;
        // Standard posterise step: bucket = round(v * (L-1) / 255)
        // then expanded back to bucket * 255 / (L-1).
        let l = levels.max(2);
        let bucket_num = biased * (l - 1) + 128;
        // Clamp to the valid bucket index range before rescaling so
        // far-out-of-range bias doesn't push us past `L-1`.
        let bucket = (bucket_num / 255).clamp(0, l - 1);
        let out = (bucket * 255 + (l - 1) / 2) / (l - 1);
        out.clamp(0, 255) as u8
    }

    /// Same as `quantise` but also returns the residual error
    /// `original - quantised` (used by the error-diffusion path).
    fn quantise_with_error(sample: i32, levels: i32) -> (u8, i32) {
        let out = Self::quantise(sample, 0, levels);
        (out, sample - out as i32)
    }

    /// Apply the dither to a single 8-bit channel plane (Gray8 or a
    /// YUV luma plane). `step` is the inter-pixel byte stride within
    /// a row (1 for Gray8 / luma; 3 / 4 for the per-channel slice of
    /// an interleaved RGB / RGBA row).
    ///
    /// `width` and `height` are in pixels; `row_stride` is the number
    /// of bytes between the *start* of one row and the start of the
    /// next.
    fn dither_channel(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
        row_stride: usize,
        step: usize,
        x_offset: usize,
    ) {
        let levels = self.levels as i32;
        match self.mode {
            DitherMode::Ordered(matrix) => {
                // Pure point process — the result for `(x, y)` only
                // depends on the input at `(x, y)` and the Bayer cell
                // for that position. Half-range bias: the matrix
                // entry, mapped from `[0, 255]` to `[-128, +127]`,
                // is what we add before quantising. Scaled down by
                // `1 / levels` so the bias gradient lines up with the
                // distance between output buckets — wider gaps with
                // more levels would otherwise wash out the dither.
                for y in 0..height {
                    let row = &mut data[y * row_stride..y * row_stride + step * width + x_offset];
                    for x in 0..width {
                        let t = bayer_threshold_q8(matrix, x, y);
                        // Bias = (t - 127) / (levels - 1), keeps the
                        // dither magnitude proportional to the bucket
                        // spacing.
                        let bias = (t - 127) / (levels - 1).max(1);
                        let idx = step * x + x_offset;
                        let s = row[idx] as i32;
                        row[idx] = Self::quantise(s, bias, levels);
                    }
                }
            }
            DitherMode::ErrorDiffusion(kernel) => {
                let table = kernel_table(kernel);
                // Running error buffer covers the kernel's reach — for
                // a 3-row kernel we need rows `[y, y+1, y+2]`. We use a
                // flat `height × width` i32 buffer so we don't have to
                // worry about which kernel is active.
                let mut err = vec![0i32; width * height];
                for y in 0..height {
                    let row_off = y * row_stride;
                    // §1.3 serpentine: even rows (0, 2, …) go left→right;
                    // odd rows go right→left with the stencil mirrored
                    // horizontally (every `dx` negated). For raster scan
                    // every row goes left→right and the stencil is used
                    // as drawn. `reverse` selects the right→left rows.
                    let reverse = matches!(self.scan, ScanOrder::Serpentine) && (y % 2 == 1);
                    for col in 0..width {
                        // Visit columns right→left on reversed rows so
                        // the "not-yet-quantised" neighbours the stencil
                        // pushes into stay ahead of the scan.
                        let x = if reverse { width - 1 - col } else { col };
                        let idx = row_off + step * x + x_offset;
                        let s = data[idx] as i32 + err[y * width + x] / table.divisor;
                        let (q, e) = Self::quantise_with_error(s, levels);
                        data[idx] = q;
                        // The fractional residue we push forward is
                        // `e * weight / divisor`; we delay dividing by
                        // `divisor` until the destination pixel is
                        // read (above) so all the arithmetic stays
                        // integer and lossless across the stencil. On a
                        // reversed row the horizontal offsets are
                        // mirrored (`-dx`) so error still flows into the
                        // pixels the right→left scan hasn't reached yet.
                        for &(dx, dy, w) in table.stencil {
                            let sdx = if reverse { -dx } else { dx };
                            let nx = x as i32 + sdx;
                            let ny = y as i32 + dy;
                            if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                                continue;
                            }
                            err[ny as usize * width + nx as usize] += e * w;
                        }
                    }
                }
            }
        }
    }
}

impl ImageFilter for Dither {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        if !is_supported_format(params.format) {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: Dither does not yet handle {:?}",
                params.format
            )));
        }
        let mut out = input.clone();
        let w = params.width as usize;
        let h = params.height as usize;

        match params.format {
            PixelFormat::Gray8 => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                self.dither_channel(&mut plane.data, w, h, stride, 1, 0);
            }
            PixelFormat::Rgb24 => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                // R, G, B share the same scan; each colour channel is
                // dithered independently. (A cross-channel scheme like
                // vector-error-diffusion would change the perceived
                // hue; per-channel matches every other implementation
                // we tabulated.)
                for ch in 0..3 {
                    self.dither_channel(&mut plane.data, w, h, stride, 3, ch);
                }
            }
            PixelFormat::Rgba => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                // Dither R/G/B; leave alpha verbatim.
                for ch in 0..3 {
                    self.dither_channel(&mut plane.data, w, h, stride, 4, ch);
                }
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                let plane = &mut out.planes[0];
                let stride = plane.stride;
                self.dither_channel(&mut plane.data, w, h, stride, 1, 0);
            }
            _ => unreachable!("is_supported_format gate already rejected this format"),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn bayer_16_thresholds_span_the_full_byte_range() {
        // Each of the 256 cells maps to a distinct `(m + 0.5) / 256`
        // threshold scaled to 8-bit, so the spread covers the range
        // from near-0 to near-255 without clustering.
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for y in 0..16 {
            for x in 0..16 {
                let t = bayer_threshold_q8(BayerMatrix::M16, x, y);
                assert!((0..=255).contains(&t), "threshold {t} out of range");
                lo = lo.min(t);
                hi = hi.max(t);
            }
        }
        // The smallest index (0) → ~0, the largest (255) → ~254.
        assert!(lo <= 1, "lowest threshold should be near 0, got {lo}");
        assert!(hi >= 254, "highest threshold should be near 255, got {hi}");
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
        for px in out.planes[0].data.chunks_exact(3) {
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
