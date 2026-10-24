//! `+noise` / `addNoiseImage` (issue #12): seeded, deterministic noise.
//!
//! The generator is this crate's own (SplitMix64), so pixels do not
//! match `magick`'s; the distributions do. Measured on `magick` 7.1.2
//! Q16-HDRI (400 × 400 flat pictures at 25 / 50 / 75 % grey, attenuate 0.5
//! / 1 / 2), in normalised units with attenuate `a`:
//!
//! | type | model |
//! |---|---|
//! | Uniform | `v + U(−a/128, a/128)` |
//! | Gaussian | `v + N(0, 0.0779 a)` |
//! | Multiplicative | `v · (1 + N(0, 0.25 a))` |
//! | Impulse | with probability `0.1 a` the sample becomes 0 or 1 (equally) |
//! | Laplacian | `v + Laplace(0, a · 10/256)` |
//! | Poisson | `Poisson(v · 12.5 a) / (12.5 a)` |
//! | Random | `U(0, a)` |
//!
//! Float output keeps out-of-range values (as HDRI does); integer output
//! saturates. Alpha is untouched.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// A noise distribution.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Additive uniform.
    Uniform,
    /// Additive Gaussian.
    Gaussian,
    /// Multiplicative Gaussian.
    Multiplicative,
    /// Salt and pepper.
    Impulse,
    /// Additive Laplacian.
    Laplacian,
    /// Poisson (shot) noise.
    Poisson,
    /// Replace by uniform random.
    Random,
}

impl NoiseType {
    /// Parse a `+noise` type name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "uniform" => Self::Uniform,
            "gaussian" => Self::Gaussian,
            "multiplicative" => Self::Multiplicative,
            "impulse" => Self::Impulse,
            "laplacian" => Self::Laplacian,
            "poisson" => Self::Poisson,
            "random" => Self::Random,
            _ => return None,
        })
    }
}

/// Add noise.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct AddNoise {
    /// The distribution.
    pub kind: NoiseType,
    /// Amplitude multiplier (`-attenuate`).
    pub attenuate: f64,
    /// Generator seed (`-seed` / `setSeed`).
    pub seed: u64,
}

impl AddNoise {
    /// `kind` at attenuate 1, seed 0.
    pub fn new(kind: NoiseType) -> Self {
        Self {
            kind,
            attenuate: 1.0,
            seed: 0,
        }
    }

    /// `-attenuate`.
    pub fn with_attenuate(mut self, a: f64) -> Self {
        self.attenuate = if a.is_finite() && a >= 0.0 { a } else { 1.0 };
        self
    }

    /// `-seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn sample(&self, v: f32, rng: &mut Rng) -> f32 {
        let a = self.attenuate;
        let v = v as f64;
        (match self.kind {
            NoiseType::Uniform => v + (rng.unit() * 2.0 - 1.0) * a / 128.0,
            NoiseType::Gaussian => v + rng.normal() * 0.0779 * a,
            NoiseType::Multiplicative => v * (1.0 + rng.normal() * 0.25 * a),
            NoiseType::Impulse => {
                let u = rng.unit();
                let q = 0.1 * a;
                if u < q / 2.0 {
                    0.0
                } else if u < q {
                    1.0
                } else {
                    v
                }
            }
            NoiseType::Laplacian => {
                let u = rng.unit() - 0.5;
                let b = a * 10.0 / 256.0;
                v - b * u.signum() * (1.0 - 2.0 * u.abs()).max(1e-300).ln()
            }
            NoiseType::Poisson => {
                let k = 12.5 * a.max(1e-9);
                rng.poisson(v.max(0.0) * k) / k
            }
            NoiseType::Random => rng.unit() * a,
        }) as f32
    }
}

/// SplitMix64 with Box–Muller and Knuth / normal-approximation Poisson.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        if lambda > 60.0 {
            return (lambda + lambda.sqrt() * self.normal()).round().max(0.0);
        }
        let l = (-lambda).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.unit();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

impl PixelFilter for AddNoise {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let float = deep::is_float(params.format);
        let mut img = if float {
            deep::FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        let mut rng = Rng(self.seed ^ 0x6A09_E667_F3BC_C908);
        let me = *self;
        img.map_pixels_in_place(|px| {
            for v in px.iter_mut() {
                *v = me.sample(*v, &mut rng);
            }
        });
        Ok(if float {
            img.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&img);
            f.pts = input.pts;
            f
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed() {
        let n = AddNoise::new(NoiseType::Gaussian).with_seed(3);
        let (mut a, mut b) = (Rng(3), Rng(3));
        assert_eq!(n.sample(0.5, &mut a), n.sample(0.5, &mut b));
        let mut c = Rng(4);
        assert_ne!(n.sample(0.5, &mut Rng(3)), n.sample(0.5, &mut c));
    }
}
