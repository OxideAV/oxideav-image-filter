//! `-sparse-color` / `sparseColorImage` (issue #12): fill the colour
//! channels from a few coloured points.
//!
//! Measured on `magick` 7.1.2: a point `x,y` sits at pixel index `(x, y)`
//! (no half-pixel shift); `Barycentric` is the least-squares plane
//! through the points (a gradient along the line for two, a constant for
//! one), `Bilinear` the least-squares `a + bx + cy + dxy` (four or more
//! points), `Shepards` / `Inverse` the inverse-distance weighted mean with
//! power 2 / 1 and the squared distance floored at 1, `Voronoi` /
//! `Manhattan` the nearest point by Euclidean / L1 distance (first point
//! on ties). Colours are linear blends of the normalised samples, clamped
//! to `[0, 1]` on integer output. Alpha is left as it is.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// A sparse-colour interpolation method.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseMethod {
    /// Least-squares plane.
    Barycentric,
    /// Least-squares bilinear surface.
    Bilinear,
    /// Inverse squared distance.
    Shepards,
    /// Inverse distance.
    Inverse,
    /// Nearest point (Euclidean).
    Voronoi,
    /// Nearest point (city block).
    Manhattan,
}

impl SparseMethod {
    /// Parse a `-sparse-color` method name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "barycentric" => Self::Barycentric,
            "bilinear" => Self::Bilinear,
            "shepards" => Self::Shepards,
            "inverse" => Self::Inverse,
            "voronoi" => Self::Voronoi,
            "manhattan" => Self::Manhattan,
            _ => return None,
        })
    }
}

/// A coloured point (normalised RGB).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparsePoint {
    /// Column (pixel index units).
    pub x: f64,
    /// Row.
    pub y: f64,
    /// Colour, normalised RGB.
    pub color: [f32; 3],
}

impl SparsePoint {
    /// A point.
    pub fn new(x: f64, y: f64, color: [f32; 3]) -> Self {
        Self { x, y, color }
    }
}

/// Fill the colour channels from sparse points.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SparseColor {
    /// The method.
    pub method: SparseMethod,
    /// The points.
    pub points: Vec<SparsePoint>,
    /// Override of the Shepards power (`shepards:power`).
    pub power: Option<f64>,
}

impl SparseColor {
    /// A method over `points` (at least one).
    pub fn new(method: SparseMethod, points: Vec<SparsePoint>) -> Result<Self, Error> {
        if points.is_empty() {
            return Err(Error::invalid(
                "oxideav-image-filter: sparse-color needs at least one point",
            ));
        }
        if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err(Error::invalid(
                "oxideav-image-filter: sparse-color point is not finite",
            ));
        }
        Ok(Self {
            method,
            points,
            power: None,
        })
    }

    /// `shepards:power`.
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = (power.is_finite() && power > 0.0).then_some(power);
        self
    }

    /// The colour at `(x, y)`.
    pub fn color_at(&self, x: f64, y: f64) -> [f32; 3] {
        let pts = &self.points;
        let mut out = [0.0f32; 3];
        match self.method {
            SparseMethod::Voronoi | SparseMethod::Manhattan => {
                let mut best = (f64::INFINITY, 0usize);
                for (k, p) in pts.iter().enumerate() {
                    let d = if self.method == SparseMethod::Voronoi {
                        (x - p.x).powi(2) + (y - p.y).powi(2)
                    } else {
                        (x - p.x).abs() + (y - p.y).abs()
                    };
                    if d < best.0 {
                        best = (d, k);
                    }
                }
                return pts[best.1].color;
            }
            SparseMethod::Shepards | SparseMethod::Inverse => {
                let pw = self
                    .power
                    .unwrap_or(if self.method == SparseMethod::Shepards {
                        2.0
                    } else {
                        1.0
                    });
                let mut acc = [0.0f64; 3];
                let mut wsum = 0.0f64;
                for p in pts {
                    let d2 = ((x - p.x).powi(2) + (y - p.y).powi(2)).max(1.0);
                    let w = 1.0 / d2.powf(pw / 2.0);
                    for c in 0..3 {
                        acc[c] += w * p.color[c] as f64;
                    }
                    wsum += w;
                }
                for c in 0..3 {
                    out[c] = (acc[c] / wsum) as f32;
                }
            }
            SparseMethod::Barycentric | SparseMethod::Bilinear => {
                let coef = self.fit();
                let basis = if coef[0].len() == 4 {
                    vec![1.0, x, y, x * y]
                } else {
                    vec![1.0, x, y]
                };
                for c in 0..3 {
                    out[c] = basis.iter().zip(&coef[c]).map(|(a, b)| a * b).sum::<f64>() as f32;
                }
            }
        }
        out
    }

    /// Per-channel surface coefficients over `[1, x, y(, xy)]`.
    fn fit(&self) -> Vec<Vec<f64>> {
        let pts = &self.points;
        let bilinear = self.method == SparseMethod::Bilinear && pts.len() >= 4;
        (0..3)
            .map(|c| {
                if pts.len() == 1 {
                    return vec![pts[0].color[c] as f64, 0.0, 0.0];
                }
                if pts.len() == 2 && !bilinear {
                    // Gradient along the segment.
                    let (a, b) = (pts[0], pts[1]);
                    let (dx, dy) = (b.x - a.x, b.y - a.y);
                    let l2 = (dx * dx + dy * dy).max(1e-12);
                    let dc = (b.color[c] - a.color[c]) as f64;
                    let (gx, gy) = (dc * dx / l2, dc * dy / l2);
                    return vec![a.color[c] as f64 - gx * a.x - gy * a.y, gx, gy];
                }
                let rows: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|p| {
                        if bilinear {
                            vec![1.0, p.x, p.y, p.x * p.y]
                        } else {
                            vec![1.0, p.x, p.y]
                        }
                    })
                    .collect();
                let n = rows[0].len();
                let mut a = vec![vec![0.0; n]; n];
                let mut b = vec![0.0; n];
                for (r, p) in rows.iter().zip(pts) {
                    for i in 0..n {
                        b[i] += r[i] * p.color[c] as f64;
                        for k in 0..n {
                            a[i][k] += r[i] * r[k];
                        }
                    }
                }
                solve(a, b).unwrap_or_else(|| {
                    let mean =
                        pts.iter().map(|p| p.color[c] as f64).sum::<f64>() / pts.len() as f64;
                    let mut v = vec![0.0; n];
                    v[0] = mean;
                    v
                })
            })
            .collect()
    }
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (dst, src) in a[row][col..n].iter_mut().zip(&pivot_row[col..n]) {
                *dst -= f * src;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

impl PixelFilter for SparseColor {
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
        let (w, ch, cc) = (img.width(), img.channels(), img.color_channels());
        for (i, px) in img.samples_mut().chunks_exact_mut(ch).enumerate() {
            let (x, y) = ((i as u32 % w) as f64, (i as u32 / w) as f64);
            let c = self.color_at(x, y);
            if cc == 1 {
                px[0] = 0.212_656 * c[0] + 0.715_158 * c[1] + 0.072_186 * c[2];
            } else {
                px[..3].copy_from_slice(&c);
            }
        }
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
    fn methods_hit_their_points() {
        let pts = vec![
            SparsePoint::new(0.0, 0.0, [1.0, 0.0, 0.0]),
            SparsePoint::new(10.0, 0.0, [0.0, 1.0, 0.0]),
            SparsePoint::new(0.0, 10.0, [0.0, 0.0, 1.0]),
        ];
        let b = SparseColor::new(SparseMethod::Barycentric, pts.clone()).unwrap();
        let c = b.color_at(10.0, 0.0);
        assert!((c[1] - 1.0).abs() < 1e-6 && c[0].abs() < 1e-6);
        let v = SparseColor::new(SparseMethod::Voronoi, pts.clone()).unwrap();
        assert_eq!(v.color_at(1.0, 8.0), [0.0, 0.0, 1.0]);
        let s = SparseColor::new(SparseMethod::Shepards, pts).unwrap();
        // d² floored at 1: a point's own pixel is mostly, not exactly, it.
        let c = s.color_at(0.0, 0.0);
        assert!(c[0] > 0.98 && c[0] < 1.0);
        assert!(SparseColor::new(SparseMethod::Inverse, vec![]).is_err());
    }
}
