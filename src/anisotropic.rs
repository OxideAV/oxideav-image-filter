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

use crate::{ImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Perona-Malik anisotropic diffusion filter.
#[derive(Clone, Copy, Debug)]
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

impl ImageFilter for AnisotropicBlur {
    fn apply(&self, input: &VideoFrame, params: VideoStreamParams) -> Result<VideoFrame, Error> {
        let bpp = match params.format {
            PixelFormat::Gray8 => 1usize,
            PixelFormat::Rgb24 => 3usize,
            PixelFormat::Rgba => 4usize,
            other => {
                return Err(Error::unsupported(format!(
                    "oxideav-image-filter: AnisotropicBlur requires Gray8/Rgb24/Rgba, got {other:?}"
                )));
            }
        };
        let w = params.width as usize;
        let h = params.height as usize;
        if w == 0 || h == 0 {
            return Ok(input.clone());
        }

        // Work in f32 for stability across iterations. Per-colour-channel
        // independent diffusion. Alpha (channel 3 on RGBA) is held
        // constant.
        let tone_ch = match bpp {
            1 => 1,
            _ => 3,
        };
        let src = &input.planes[0];
        let row_stride_in = src.stride;

        // Build per-channel f32 buffers.
        let mut buf: Vec<Vec<f32>> = (0..tone_ch).map(|_| vec![0.0; w * h]).collect();
        for y in 0..h {
            let row = &src.data[y * row_stride_in..y * row_stride_in + w * bpp];
            for x in 0..w {
                let p = x * bpp;
                for c in 0..tone_ch {
                    buf[c][y * w + x] = row[p + c] as f32;
                }
            }
        }

        let kappa_sq = self.kappa * self.kappa;
        let lambda = self.lambda;
        let mut scratch = vec![0.0f32; w * h];
        for chan in buf.iter_mut().take(tone_ch) {
            for _ in 0..self.iterations {
                // North/South/East/West first differences + Lorentzian
                // weights folded into one explicit-Euler step.
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let v = chan[i];
                        let n = if y > 0 { chan[i - w] } else { v };
                        let s = if y + 1 < h { chan[i + w] } else { v };
                        let east = if x + 1 < w { chan[i + 1] } else { v };
                        let west = if x > 0 { chan[i - 1] } else { v };
                        let dn = n - v;
                        let ds = s - v;
                        let de = east - v;
                        let dw = west - v;
                        // g(s) = 1 / (1 + (s/κ)²)  ⇒ g(s) · s
                        let gn = dn / (1.0 + dn * dn / kappa_sq);
                        let gs = ds / (1.0 + ds * ds / kappa_sq);
                        let ge = de / (1.0 + de * de / kappa_sq);
                        let gw = dw / (1.0 + dw * dw / kappa_sq);
                        scratch[i] = v + lambda * (gn + gs + ge + gw);
                    }
                }
                std::mem::swap(chan, &mut scratch);
            }
        }

        // Re-pack into bytes (alpha preserved from input).
        let row_stride = w * bpp;
        let mut data = vec![0u8; row_stride * h];
        for y in 0..h {
            let src_row = &src.data[y * row_stride_in..y * row_stride_in + w * bpp];
            let dst_row = &mut data[y * row_stride..(y + 1) * row_stride];
            for x in 0..w {
                let p = x * bpp;
                for c in 0..tone_ch {
                    dst_row[p + c] = buf[c][y * w + x].round().clamp(0.0, 255.0) as u8;
                }
                if bpp == 4 {
                    dst_row[p + 3] = src_row[p + 3];
                }
            }
        }

        Ok(VideoFrame {
            pts: input.pts,
            planes: vec![VideoPlane {
                stride: row_stride,
                data,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
        };
        let err = AnisotropicBlur::default().apply(&input, p).unwrap_err();
        assert!(format!("{err}").contains("AnisotropicBlur"));
    }
}
