//! Anisotropic diffusion — edge-preserving smoothing via the
//! Perona-Malik 1990 update rule.
//!
//! Iteratively replaces every pixel by the explicit-Euler update
//!
//! ```text
//!   I_{n+1}(x, y) = I_n(x, y) + λ · Σ_{d ∈ N,S,E,W} g(|∇_d I|) · ∇_d I
//! ```
//!
//! where the four directional gradients are first differences against
//! the 4-neighbours and the *edge-stopping* function is the Lorentzian
//!
//! ```text
//!   g(s) = 1 / (1 + (s / κ)²)
//! ```
//!
//! Small gradients (smooth regions) get `g ≈ 1` and diffuse freely;
//! large gradients (edges) get `g → 0` and freeze, so edges are
//! preserved even at high iteration counts. `λ ≤ 0.25` guarantees
//! stability for the 4-neighbour scheme.
//!
//! Distinct from:
//! - [`Blur`](crate::Blur) — full Gaussian; oblivious to edges.
//! - [`BilateralBlur`](crate::BilateralBlur) — single-shot range +
//!   space joint Gaussian; no iterative diffusion.
//! - [`Kuwahara`](crate::Kuwahara) — quadrant-variance selection;
//!   piecewise-constant output.
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba`. Alpha pass-through on RGBA. Other
//! formats return [`Error::unsupported`].

use crate::{PixelFilter, VideoStreamParams};
use oxideav_core::{Error, VideoFrame};

/// Perona-Malik anisotropic diffusion filter.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct AnisotropicBlur {
    /// Number of diffusion iterations (≥ 1). More iterations =
    /// smoother flat areas, but edges still stop the flow.
    pub iterations: u32,
    /// Edge-stopping threshold (κ in the Lorentzian). Gradients far
    /// below `kappa` diffuse freely; far above are preserved. Typical
    /// `10..50` for 8-bit imagery.
    pub kappa: f32,
    /// Time step λ in `(0, 0.25]`. The 4-neighbour explicit scheme
    /// requires `λ ≤ 1/4` for stability.
    pub lambda: f32,
}

impl AnisotropicBlur {
    /// Builder-style setter for [`iterations`](Self::iterations).
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }
}

impl Default for AnisotropicBlur {
    fn default() -> Self {
        Self {
            iterations: 5,
            kappa: 20.0,
            lambda: 0.2,
        }
    }
}

impl AnisotropicBlur {
    /// Build with the given iteration count; κ defaults to 20, λ to 0.2.
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations: iterations.max(1),
            ..Self::default()
        }
    }

    /// Override κ (edge-stopping threshold). Clamped to `> 0`.
    pub fn with_kappa(mut self, kappa: f32) -> Self {
        self.kappa = kappa.max(f32::EPSILON);
        self
    }

    /// Override λ (time step). Clamped to `(0, 0.25]`.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(f32::EPSILON, 0.25);
        self
    }
}

impl PixelFilter for AnisotropicBlur {
    fn accepted_formats(&self) -> &'static [oxideav_core::PixelFormat] {
        crate::layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        // One kernel at every depth: the diffusion runs in binary32 on
        // the 8-bit scale (`κ` is a contrast in 8-bit codes), so a
        // float or 16-bit picture is scaled by 255 on the way in and
        // back on the way out without being rounded.
        let me = *self;
        crate::deep::run_any(input, params, move |mut img| {
            let (w, h) = (img.width() as usize, img.height() as usize);
            if w == 0 || h == 0 {
                return Ok(img);
            }
            let (ch, cc) = (img.channels(), img.color_channels());
            let mut chan = vec![0.0f32; w * h];
            let mut scratch = vec![0.0f32; w * h];
            for c in 0..cc {
                for (d, px) in chan.iter_mut().zip(img.samples().chunks_exact(ch)) {
                    *d = px[c] * 255.0;
                }
                me.diffuse(&mut chan, &mut scratch, w, h);
                for (px, v) in img.samples_mut().chunks_exact_mut(ch).zip(&chan) {
                    px[c] = *v / 255.0;
                }
            }
            Ok(img)
        })
    }
}

impl AnisotropicBlur {
    /// `iterations` explicit-Euler steps of Perona–Malik diffusion on
    /// one channel (8-bit scale), edges replicated: each step adds
    /// `λ · Σ g(∇) ∇` over the four neighbours with the Lorentzian
    /// conductance `g(s) = 1 / (1 + (s / κ)²)`.
    fn diffuse(&self, chan: &mut Vec<f32>, scratch: &mut Vec<f32>, w: usize, h: usize) {
        let kappa_sq = self.kappa * self.kappa;
        let lambda = self.lambda;
        for _ in 0..self.iterations {
            let cur = &*chan;
            // A step reads only the previous state: rows run under the
            // thread budget.
            crate::exec::par_rows(scratch, w, |y, out| {
                for (x, o) in out.iter_mut().enumerate() {
                    let i = y * w + x;
                    let v = cur[i];
                    let n = if y > 0 { cur[i - w] } else { v };
                    let s = if y + 1 < h { cur[i + w] } else { v };
                    let east = if x + 1 < w { cur[i + 1] } else { v };
                    let west = if x > 0 { cur[i - 1] } else { v };
                    let (dn, ds, de, dw) = (n - v, s - v, east - v, west - v);
                    // g(s) = 1 / (1 + (s/κ)²)  ⇒ g(s) · s
                    let gn = dn / (1.0 + dn * dn / kappa_sq);
                    let gs = ds / (1.0 + ds * ds / kappa_sq);
                    let ge = de / (1.0 + de * de / kappa_sq);
                    let gw = dw / (1.0 + dw * dw / kappa_sq);
                    *o = v + lambda * (gn + gs + ge + gw);
                }
            });
            std::mem::swap(chan, scratch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    #[allow(unused_imports)]
    use oxideav_core::{PixelFormat, VideoPlane};

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

    fn p_gray(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Gray8,
            width: w,
            height: h,
        }
    }

    #[test]
    fn constant_input_is_fixed_point() {
        let input = gray(8, 8, |_, _| 100);
        let out = AnisotropicBlur::new(10)
            .apply(&input, p_gray(8, 8))
            .unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 100, "smooth region drifted");
        }
    }

    #[test]
    fn iterations_clamped() {
        assert_eq!(AnisotropicBlur::new(0).iterations, 1);
    }

    #[test]
    fn lambda_clamped_to_quarter() {
        let f = AnisotropicBlur::default().with_lambda(1.0);
        assert!(f.lambda <= 0.25 + 1e-6);
    }

    #[test]
    fn strong_step_edge_survives() {
        // 16-wide bipartite image; κ small enough that step gradient
        // (255) is treated as an edge.
        let input = gray(16, 16, |x, _| if x < 8 { 0 } else { 255 });
        let out = AnisotropicBlur::new(20)
            .with_kappa(15.0)
            .apply(&input, p_gray(16, 16))
            .unwrap();
        // Centre column on each side should still be saturated.
        for y in 1..15 {
            assert!(
                out.planes[0].data[y * 16 + 1] < 30,
                "left dark side drifted"
            );
            assert!(
                out.planes[0].data[y * 16 + 14] > 225,
                "right bright side drifted"
            );
        }
    }

    #[test]
    fn rgb_alpha_preserved() {
        let mut data = Vec::with_capacity(8 * 8 * 4);
        for _ in 0..(8 * 8) {
            data.extend_from_slice(&[100u8, 100, 100, 55]);
        }
        let input = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: 8,
            height: 8,
        };
        let out = AnisotropicBlur::new(3).apply(&input, p).unwrap();
        for px in out.planes[0].data.chunks(4) {
            assert_eq!(px[3], 55, "alpha not preserved");
        }
    }

    #[test]
    fn rejects_yuv420p() {
        // Every 8-bit layout outside the native set is taken through
        // the packed-RGB hop; a deep layout never is (no silent
        // narrowing) and is refused with a typed error.
        let input = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![128; 16],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![128; 4],
                },
            ],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Yuv420P10Le,
            width: 4,
            height: 4,
        };
        let err = AnisotropicBlur::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("AnisotropicBlur"));
    }
}
