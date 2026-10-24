//! `-fft` / `-ift` (issue #12): the 2-D discrete Fourier transform of
//! every channel as a magnitude / phase (or real / imaginary) pair, and
//! its exact inverse.
//!
//! Conventions (this crate's; the `magick` on the validation host is
//! built without its FFT delegate, so they are not pinned against it):
//! the forward transform is
//! `F(k, l) = (1 / (W·H)) Σ f(x, y) e^{−2πi (kx/W + ly/H)}`
//! with the zero frequency moved to `(⌊W/2⌋, ⌊H/2⌋)`; the
//! phase is mapped from `[−π, π]` to `[0, 1]`; magnitudes are not
//! log-scaled. [`inverse`] undoes [`forward`] to floating-point
//! precision for every extent (radix-2 where a side is a power of two, a
//! direct transform otherwise).

use oxideav_core::{Error, VideoFrame};

use crate::VideoStreamParams;
use crate::deep::FloatImage;

/// The frequency-domain pictures.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    /// Magnitude (or real part), one channel per input channel.
    pub first: FloatImage,
    /// Phase in `[0, 1]` (or imaginary part).
    pub second: FloatImage,
    /// `true` for magnitude / phase, `false` for real / imaginary.
    pub polar: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C(f64, f64);

impl C {
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}

/// In-place 1-D DFT of `v` (sign −1 forward, +1 inverse), unnormalised.
fn dft(v: &mut [C], sign: f64) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        // Iterative radix-2.
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                v.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
            let wl = C(ang.cos(), ang.sin());
            for start in (0..n).step_by(len) {
                let mut w = C(1.0, 0.0);
                for k in 0..len / 2 {
                    let a = v[start + k];
                    let b = v[start + k + len / 2].mul(w);
                    v[start + k] = C(a.0 + b.0, a.1 + b.1);
                    v[start + k + len / 2] = C(a.0 - b.0, a.1 - b.1);
                    w = w.mul(wl);
                }
            }
            len <<= 1;
        }
        return;
    }
    let src = v.to_vec();
    for (k, out) in v.iter_mut().enumerate() {
        let mut acc = C::default();
        for (x, &s) in src.iter().enumerate() {
            let ang = sign * 2.0 * std::f64::consts::PI * ((k * x) % n) as f64 / n as f64;
            acc = C(
                acc.0 + s.0 * ang.cos() - s.1 * ang.sin(),
                acc.1 + s.0 * ang.sin() + s.1 * ang.cos(),
            );
        }
        *out = acc;
    }
}

fn dft2(data: &mut [C], w: usize, h: usize, sign: f64) {
    let mut row = vec![C::default(); w];
    for y in 0..h {
        row.copy_from_slice(&data[y * w..(y + 1) * w]);
        dft(&mut row, sign);
        data[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    let mut col = vec![C::default(); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = data[y * w + x];
        }
        dft(&mut col, sign);
        for y in 0..h {
            data[y * w + x] = col[y];
        }
    }
}

/// `-fft` (`polar`: magnitude / phase) or `+fft` (real / imaginary) of a
/// float picture.
pub fn forward_float(img: &FloatImage, polar: bool) -> Spectrum {
    let (w, h, ch) = (img.width() as usize, img.height() as usize, img.channels());
    let mut first = FloatImage::new(w as u32, h as u32, ch);
    let mut second = FloatImage::new(w as u32, h as u32, ch);
    let norm = 1.0 / (w * h) as f64;
    for c in 0..ch {
        let mut data: Vec<C> = (0..w * h)
            .map(|i| C(img.samples()[i * ch + c] as f64, 0.0))
            .collect();
        dft2(&mut data, w, h, -1.0);
        for y in 0..h {
            for x in 0..w {
                // Shift the zero frequency to (w/2, h/2).
                let (sx, sy) = ((x + w / 2) % w, (y + h / 2) % h);
                let v = data[y * w + x];
                let (re, im) = (v.0 * norm, v.1 * norm);
                let (a, b) = if polar {
                    (
                        (re * re + im * im).sqrt(),
                        im.atan2(re) / (2.0 * std::f64::consts::PI) + 0.5,
                    )
                } else {
                    (re, im)
                };
                first.pixel_mut(sx as u32, sy as u32)[c] = a as f32;
                second.pixel_mut(sx as u32, sy as u32)[c] = b as f32;
            }
        }
    }
    Spectrum {
        first,
        second,
        polar,
    }
}

/// [`forward_float`] on a frame (any layout the float hop reaches).
pub fn forward(
    frame: &VideoFrame,
    params: VideoStreamParams,
    polar: bool,
) -> Result<Spectrum, Error> {
    Ok(forward_float(
        &crate::measure::float_image_of(frame, params)?,
        polar,
    ))
}

/// `-ift`: the picture back from a [`Spectrum`].
pub fn inverse(s: &Spectrum) -> Result<FloatImage, Error> {
    let (a, b) = (&s.first, &s.second);
    if a.width() != b.width() || a.height() != b.height() || a.channels() != b.channels() {
        return Err(Error::invalid(
            "oxideav-image-filter: ift needs two pictures of one shape",
        ));
    }
    let (w, h, ch) = (a.width() as usize, a.height() as usize, a.channels());
    let mut out = FloatImage::new(w as u32, h as u32, ch);
    for c in 0..ch {
        let mut data = vec![C::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = ((x + w / 2) % w, (y + h / 2) % h);
                let (p, q) = (
                    a.pixel(sx as u32, sy as u32)[c] as f64,
                    b.pixel(sx as u32, sy as u32)[c] as f64,
                );
                data[y * w + x] = if s.polar {
                    let ang = (q - 0.5) * 2.0 * std::f64::consts::PI;
                    C(p * ang.cos(), p * ang.sin())
                } else {
                    C(p, q)
                };
            }
        }
        dft2(&mut data, w, h, 1.0);
        for (i, v) in data.iter().enumerate() {
            out.samples_mut()[i * ch + c] = v.0 as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pic(w: u32, h: u32) -> FloatImage {
        let mut img = FloatImage::new(w, h, 1);
        for (i, v) in img.samples_mut().iter_mut().enumerate() {
            *v = ((i * 37 % 11) as f32) / 10.0;
        }
        img
    }

    #[test]
    fn round_trip_power_of_two_and_odd() {
        for (w, h) in [(8u32, 4u32), (7, 5), (6, 9)] {
            let img = pic(w, h);
            for polar in [true, false] {
                let back = inverse(&forward_float(&img, polar)).unwrap();
                for (a, b) in back.samples().iter().zip(img.samples()) {
                    assert!((a - b).abs() < 1e-5, "{w}x{h} polar={polar}");
                }
            }
        }
    }

    #[test]
    fn flat_picture_has_only_dc_and_impulse_is_flat() {
        let mut img = FloatImage::new(8, 8, 1);
        img.samples_mut().fill(0.4);
        let s = forward_float(&img, true);
        assert!((s.first.pixel(4, 4)[0] - 0.4).abs() < 1e-6);
        assert!(
            s.first
                .samples()
                .iter()
                .enumerate()
                .all(|(i, &v)| i == 4 * 8 + 4 || v.abs() < 1e-6)
        );
        let mut imp = FloatImage::new(8, 8, 1);
        imp.pixel_mut(0, 0)[0] = 1.0;
        let s = forward_float(&imp, true);
        assert!(
            s.first
                .samples()
                .iter()
                .all(|&v| (v - 1.0 / 64.0).abs() < 1e-6)
        );
    }
}
