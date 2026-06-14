//! Edge feathering — a soft coverage falloff at a shape boundary driven
//! by the **exact Euclidean distance transform**.
//!
//! Feathering replaces a hard mask edge with a smooth ramp: pixels deep
//! inside the shape keep full coverage, pixels within `radius` of the
//! boundary fade linearly to zero coverage at the edge, and pixels
//! outside stay fully transparent. It is the standard soft-matte /
//! soft-selection primitive.
//!
//! ## Algorithm
//!
//! `docs/image/filter/distance-transform.md` §1 states the (generalised,
//! squared-Euclidean) distance transform
//!
//! ```text
//!   D_f(p) = min_{q} ( ‖p − q‖² + f(q) )
//! ```
//!
//! and its intro names **feathering** as a backed use-case. The
//! feathering ramp needs, for every interior pixel, the Euclidean
//! distance to the nearest *exterior* (background) pixel — i.e. the
//! "inner" distance `d_in` of the signed field. That is exactly the EDT
//! of the **inverted** mask (`f(q) = 0` on the background): §2,
//! Felzenszwalb–Huttenlocher, the separable lower-envelope-of-parabolas
//! transform whose 1-D driver this filter reuses (the same driver behind
//! [`EuclideanDistanceTransform`](crate::EuclideanDistanceTransform) and
//! [`SignedDistanceField`](crate::SignedDistanceField)).
//!
//! With `d_in(p)` in input-pixel units, the coverage ramp is
//!
//! ```text
//!   cov(p) = clamp( d_in(p) / radius, 0, 1 )          (background → 0)
//! ```
//!
//! a pixel exactly on the boundary (`d_in = 0`) maps to `0`, a pixel a
//! full `radius` inside reaches `1`, and the interior beyond the band is
//! saturated at `1`. `radius` is the feather width in pixels. The whole
//! filter is `O(d · N)` for an **exact** distance field.
//!
//! Clean-room transcription from `docs/image/filter/distance-transform.md`
//! §1 (generalised DT + the stated feathering use-case) and §2 (the exact
//! Euclidean transform whose 1-D driver this filter reuses).
//!
//! ## Input / output
//!
//! - **`Rgba`** — the alpha channel is the coverage mask. A pixel is
//!   foreground when `alpha >= threshold` (or, with [`invert`](Self::invert),
//!   `alpha < threshold`). Output alpha becomes `round(255 · cov)`; the
//!   RGB channels pass through unchanged. This feathers the matte edge of
//!   a cut-out subject.
//! - **`Gray8`** — the luma value is the mask. Foreground is
//!   `value >= threshold`; output is the single-plane feathered coverage
//!   ramp `round(255 · cov)`.
//!
//! Output keeps the input `(width, height)` and format. Other formats
//! return [`Error::unsupported`].

use crate::euclidean_distance_transform::{dt_1d, INF};
use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Edge-feathering filter (soft coverage falloff from an exact
/// Euclidean inner-distance transform).
#[derive(Clone, Copy, Debug)]
pub struct Feather {
    /// Feather band width in pixels. Interior pixels within `radius` of
    /// the boundary fade linearly from full coverage (a full `radius`
    /// inside) to zero coverage (on the boundary). `0` disables the
    /// ramp (every foreground pixel stays at full coverage). Must be
    /// finite and `> 0` to have any effect.
    pub radius: f32,
    /// Foreground threshold on the mask channel (alpha for `Rgba`, luma
    /// for `Gray8`). Pixels at or above this value count as foreground.
    pub threshold: u8,
    /// If `true`, the test is flipped so pixels *below* `threshold`
    /// become the foreground (useful for a dark-on-light mask).
    pub invert: bool,
}

impl Default for Feather {
    fn default() -> Self {
        Self {
            radius: 4.0,
            threshold: 128,
            invert: false,
        }
    }
}

impl Feather {
    /// New feather filter with the given band `radius` (pixels) and the
    /// defaults `threshold = 128`, `invert = false`.
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            threshold: 128,
            invert: false,
        }
    }

    /// Set the foreground threshold on the mask channel.
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Flip the foreground/background test (mask values *below*
    /// `threshold` become foreground).
    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }
}

/// Run the exact separable 2-D squared-Euclidean transform over the
/// seed array `f` (0 at feature sites, [`INF`] elsewhere), in place.
/// Mirrors the column-then-row schedule shared with
/// [`EuclideanDistanceTransform`] and [`SignedDistanceField`].
fn edt_2d(f: &mut [f64], w: usize, h: usize) {
    let n_max = w.max(h);
    let mut v_buf = vec![0_usize; n_max];
    let mut z_buf = vec![0.0_f64; n_max + 1];
    let mut col_in = vec![0.0_f64; h];
    let mut col_out = vec![0.0_f64; h];
    let mut row_in = vec![0.0_f64; w];
    let mut row_out = vec![0.0_f64; w];
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
}

impl Feather {
    /// Build the per-pixel coverage ramp `cov ∈ [0, 1]` from a binary
    /// foreground mask. Background pixels are `0.0`; foreground pixels
    /// take `clamp(d_in / radius, 0, 1)` where `d_in` is the exact
    /// Euclidean distance to the nearest background pixel.
    fn coverage(&self, is_fg: &[bool], w: usize, h: usize) -> Vec<f32> {
        // Seed the EDT of the *inverted* mask: feature sites are the
        // background pixels, so the resulting distance at each pixel is
        // the distance to the nearest background (= d_in inside the
        // shape, 0 on the background itself).
        let mut bg = vec![INF; w * h];
        for i in 0..w * h {
            if !is_fg[i] {
                bg[i] = 0.0;
            }
        }
        edt_2d(&mut bg, w, h);

        let r = self.radius as f64;
        let mut cov = vec![0.0_f32; w * h];
        for i in 0..w * h {
            if !is_fg[i] {
                // Background stays fully transparent.
                continue;
            }
            // `bg[i]` is the squared distance to the nearest background.
            let d_in = bg[i].max(0.0).sqrt();
            let c = if r > 0.0 {
                (d_in / r).clamp(0.0, 1.0)
            } else {
                // No feather band: every foreground pixel is opaque.
                1.0
            };
            cov[i] = c as f32;
        }
        cov
    }
}

impl ImageFilter for Feather {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let w = params.width as usize;
        let h = params.height as usize;
        if input.planes.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: Feather requires a non-empty input plane",
            ));
        }
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }

        match params.format {
            PixelFormat::Rgba => {
                let src = &input.planes[0];
                // Foreground = alpha channel test.
                let mut is_fg = vec![false; w * h];
                for y in 0..h {
                    let row = &src.data[y * src.stride..y * src.stride + 4 * w];
                    for x in 0..w {
                        let a = row[4 * x + 3];
                        is_fg[y * w + x] = if self.invert {
                            a < self.threshold
                        } else {
                            a >= self.threshold
                        };
                    }
                }
                let cov = self.coverage(&is_fg, w, h);

                let mut out = vec![0u8; 4 * w * h];
                for y in 0..h {
                    let s_row = &src.data[y * src.stride..y * src.stride + 4 * w];
                    let o_row = &mut out[y * 4 * w..(y + 1) * 4 * w];
                    for x in 0..w {
                        let i = y * w + x;
                        // RGB pass through; alpha becomes the feathered
                        // coverage. (`cov` is already 0 on background.)
                        o_row[4 * x] = s_row[4 * x];
                        o_row[4 * x + 1] = s_row[4 * x + 1];
                        o_row[4 * x + 2] = s_row[4 * x + 2];
                        o_row[4 * x + 3] = (cov[i] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: 4 * w,
                        data: out,
                    }],
                })
            }
            PixelFormat::Gray8 => {
                let src = &input.planes[0];
                let mut is_fg = vec![false; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let v = src.data[y * src.stride + x];
                        is_fg[y * w + x] = if self.invert {
                            v < self.threshold
                        } else {
                            v >= self.threshold
                        };
                    }
                }
                let cov = self.coverage(&is_fg, w, h);

                let mut out = vec![0u8; w * h];
                for i in 0..w * h {
                    out[i] = (cov[i] * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                Ok(VideoFrame {
                    pts: input.pts,
                    planes: vec![VideoPlane {
                        stride: w,
                        data: out,
                    }],
                })
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Feather requires Rgba or Gray8, got {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(format: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format,
            width: w,
            height: h,
        }
    }

    /// Build a Gray8 frame from a closure.
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

    /// Build an Rgba frame from a closure returning `[r, g, b, a]`.
    fn rgba(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        }
    }

    /// Brute-force exact inner distance: at every foreground pixel, the
    /// minimum Euclidean distance to any background pixel. Quadratic but
    /// exact — the oracle for the separable F–H machinery.
    fn brute_force_din(is_fg: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for ty in 0..h {
            for tx in 0..w {
                if !is_fg[ty * w + tx] {
                    continue;
                }
                let mut best = f64::INFINITY;
                for sy in 0..h {
                    for sx in 0..w {
                        if is_fg[sy * w + sx] {
                            continue;
                        }
                        let dx = tx as f64 - sx as f64;
                        let dy = ty as f64 - sy as f64;
                        let d2 = dx * dx + dy * dy;
                        if d2 < best {
                            best = d2;
                        }
                    }
                }
                out[ty * w + tx] = best.sqrt();
            }
        }
        out
    }

    #[test]
    fn unsupported_format_errors() {
        let f = Feather::new(3.0);
        let frame = gray(2, 2, |_, _| 255);
        let err = f.apply(&frame, p(PixelFormat::Yuv420P, 2, 2));
        assert!(err.is_err());
    }

    #[test]
    fn empty_plane_errors() {
        let f = Feather::new(3.0);
        let frame = VideoFrame {
            pts: None,
            planes: vec![],
        };
        assert!(f.apply(&frame, p(PixelFormat::Gray8, 2, 2)).is_err());
    }

    #[test]
    fn zero_dims_pass_through() {
        let f = Feather::new(3.0);
        let frame = gray(0, 0, |_, _| 0);
        let out = f.apply(&frame, p(PixelFormat::Gray8, 0, 0)).unwrap();
        assert_eq!(out.planes[0].data.len(), 0);
    }

    #[test]
    fn background_stays_zero_coverage() {
        // A single foreground dot in a sea of background. Every
        // background pixel must render 0; the lone foreground pixel sits
        // exactly on the boundary (its nearest background is a 4-
        // neighbour at distance 1), so with radius 4 it is 1/4 → 64.
        let f = Feather::new(4.0);
        let frame = gray(5, 5, |x, y| if x == 2 && y == 2 { 255 } else { 0 });
        let out = f.apply(&frame, p(PixelFormat::Gray8, 5, 5)).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                let v = out.planes[0].data[y * 5 + x];
                if x == 2 && y == 2 {
                    // d_in = 1 (4-neighbour). 1/4 * 255 = 63.75 → 64.
                    assert_eq!(v, 64, "centre coverage");
                } else {
                    assert_eq!(v, 0, "background at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn deep_interior_is_full_coverage() {
        // A big solid block: the centre is far from any edge, so with a
        // small radius it saturates to full coverage (255).
        let w = 21;
        let h = 21;
        let f = Feather::new(3.0);
        let frame = gray(w as u32, h as u32, |_, _| 255);
        // An all-foreground frame has no background → d_in saturates to
        // the INF sentinel everywhere, so every pixel is full coverage.
        let out = f
            .apply(&frame, p(PixelFormat::Gray8, w as u32, h as u32))
            .unwrap();
        for v in &out.planes[0].data {
            assert_eq!(*v, 255);
        }
    }

    #[test]
    fn ramp_is_monotone_into_the_shape() {
        // Left half background, right half foreground. Coverage along a
        // row must be 0 in the background and increase monotonically as
        // we march into the foreground, saturating at 255.
        let w = 16usize;
        let h = 4usize;
        let f = Feather::new(5.0);
        let frame = gray(w as u32, h as u32, |x, _| if x >= 8 { 255 } else { 0 });
        let out = f
            .apply(&frame, p(PixelFormat::Gray8, w as u32, h as u32))
            .unwrap();
        // Read row index 1.
        let row: Vec<u8> = (0..w).map(|x| out.planes[0].data[w + x]).collect();
        // Background columns are all zero.
        for (x, &v) in row.iter().enumerate().take(8) {
            assert_eq!(v, 0, "bg col {x}");
        }
        // Foreground columns are non-decreasing and end saturated.
        for x in 9..w {
            assert!(row[x] >= row[x - 1], "monotone at col {x}: {:?}", row);
        }
        assert_eq!(row[w - 1], 255, "deep interior saturates");
    }

    #[test]
    fn coverage_matches_brute_force_exact_din() {
        // Random-ish mask; verify the separable EDT-derived coverage
        // matches the brute-force exact inner distance ramp pixel-exact.
        let w = 13usize;
        let h = 11usize;
        // A deterministic pseudo-random foreground pattern.
        let is_fg: Vec<bool> = (0..w * h)
            .map(|i| {
                let x = (i % w) as i32;
                let y = (i / w) as i32;
                ((x * 7 + y * 13 + x * y) % 5) > 1
            })
            .collect();
        let radius = 6.0f64;
        let f = Feather::new(radius as f32);

        let frame = gray(w as u32, h as u32, |x, y| {
            if is_fg[(y as usize) * w + x as usize] {
                255
            } else {
                0
            }
        });
        let out = f
            .apply(&frame, p(PixelFormat::Gray8, w as u32, h as u32))
            .unwrap();

        let din = brute_force_din(&is_fg, w, h);
        for i in 0..w * h {
            let expect = if is_fg[i] {
                let c = (din[i] / radius).clamp(0.0, 1.0);
                (c * 255.0).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
            assert_eq!(out.planes[0].data[i], expect, "pixel {i}");
        }
    }

    #[test]
    fn rgba_feathers_alpha_keeps_rgb() {
        // A solid opaque block on a transparent field; RGB must survive,
        // alpha must be feathered (edge alpha < interior alpha).
        let w = 16usize;
        let h = 16usize;
        let f = Feather::new(4.0);
        let frame = rgba(w as u32, h as u32, |x, y| {
            let inside = (4..12).contains(&x) && (4..12).contains(&y);
            if inside {
                [10, 20, 30, 255]
            } else {
                [10, 20, 30, 0]
            }
        });
        let out = f
            .apply(&frame, p(PixelFormat::Rgba, w as u32, h as u32))
            .unwrap();
        let at = |x: usize, y: usize| {
            let o = (y * w + x) * 4;
            &out.planes[0].data[o..o + 4]
        };
        // RGB preserved everywhere.
        for px in out.planes[0].data.chunks_exact(4) {
            assert_eq!(&px[0..3], &[10, 20, 30]);
        }
        // Centre of the block is deep interior → full alpha.
        assert_eq!(at(7, 7)[3], 255, "centre alpha");
        // Edge pixel just inside the block (one pixel in) is feathered
        // below full.
        let edge_alpha = at(4, 7)[3];
        assert!(edge_alpha < 255, "edge alpha feathered: {edge_alpha}");
        assert!(edge_alpha > 0, "edge alpha non-zero: {edge_alpha}");
        // Background stays transparent.
        assert_eq!(at(0, 0)[3], 0, "bg alpha");
    }

    #[test]
    fn invert_flips_foreground() {
        // With invert, dark pixels become foreground. A dark dot on a
        // bright field should produce coverage at the dot, zero
        // elsewhere — mirror of the non-inverted bright-dot case.
        let f = Feather::new(4.0).with_invert(true);
        let frame = gray(5, 5, |x, y| if x == 2 && y == 2 { 0 } else { 255 });
        let out = f.apply(&frame, p(PixelFormat::Gray8, 5, 5)).unwrap();
        assert_eq!(out.planes[0].data[2 * 5 + 2], 64, "dot coverage");
        assert_eq!(out.planes[0].data[0], 0, "bright corner is bg");
    }

    #[test]
    fn zero_radius_is_hard_mask() {
        // radius 0 disables the ramp: every foreground pixel is full
        // coverage, background zero (a plain binary mask).
        let f = Feather::new(0.0);
        let frame = gray(6, 1, |x, _| if x >= 3 { 255 } else { 0 });
        let out = f.apply(&frame, p(PixelFormat::Gray8, 6, 1)).unwrap();
        assert_eq!(out.planes[0].data, vec![0, 0, 0, 255, 255, 255]);
    }
}
