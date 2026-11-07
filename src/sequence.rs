//! Operations over an image list (issue #12): `-evaluate-sequence`,
//! `-separate` and `-combine`.
//!
//! Pictures are [`FloatImage`]s (normalised binary32 — any frame reaches
//! one through [`crate::measure::float_image_of`]), so a 16-bit or float
//! sequence is combined without quantisation. The per-operator rules
//! were measured on `magick` 7.1.2 Q16-HDRI with three- and four-picture
//! lists (`tests/magick_ops.rs` matches them on `rose:` variants). Every
//! channel, alpha included, is combined independently, except `Median`:
//!
//! | op | result over `x₀ … xₙ₋₁` |
//! |---|---|
//! | `Mean` | arithmetic mean |
//! | `Median` | the whole pixel ranked `n / 2` by the sum of its samples |
//! | `Min` / `Max` | extremes |
//! | `Add` (`Sum`) | sum |
//! | `Abs` | sum of absolute values |
//! | `Subtract` | running `acc ← xᵢ − acc` from `acc = x₀` |
//! | `Multiply` | product |
//! | `Divide` | running `acc ← xᵢ / acc` in quantum units (65 535 = 1) |
//! | `RootMeanSquare` | `√(Σᵢ₌₁ xᵢ² / n)` — `magick` leaves the first picture out of the sum |
//! | `AddModulus` | the sum in quantum units modulo 65 536 |
//! | `And` / `Or` / `Xor` | bitwise on the 16-bit quantum codes |
//! | `Set` | the first picture |
//!
//! Float results keep values outside `[0, 1]` (as HDRI does).

use oxideav_core::Error;

use crate::deep::FloatImage;

/// Quantum range of the reference build.
const Q: f64 = 65535.0;

/// An `-evaluate-sequence` operator.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SequenceOp {
    /// Arithmetic mean.
    Mean,
    /// Median (upper median for an even count).
    Median,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Sum (`Add` / `Sum`).
    Add,
    /// Sum of absolute values.
    Abs,
    /// Running `xᵢ − acc`.
    Subtract,
    /// Product.
    Multiply,
    /// Running `xᵢ / acc` in quantum units.
    Divide,
    /// Root mean square (first picture excluded from the sum).
    RootMeanSquare,
    /// Sum modulo the quantum range + 1.
    AddModulus,
    /// Bitwise AND of the quantum codes.
    And,
    /// Bitwise OR of the quantum codes.
    Or,
    /// Bitwise XOR of the quantum codes.
    Xor,
    /// The first picture.
    Set,
}

impl SequenceOp {
    /// Parse a `magick -list evaluate` name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "mean" => Self::Mean,
            "median" => Self::Median,
            "min" => Self::Min,
            "max" => Self::Max,
            "add" | "sum" => Self::Add,
            "abs" => Self::Abs,
            "subtract" => Self::Subtract,
            "multiply" => Self::Multiply,
            "divide" => Self::Divide,
            "rms" | "rootmeansquare" => Self::RootMeanSquare,
            "addmodulus" => Self::AddModulus,
            "and" => Self::And,
            "or" => Self::Or,
            "xor" => Self::Xor,
            "set" => Self::Set,
            _ => return None,
        })
    }
}

/// The 16-bit quantum code of a normalised sample (snapped to the code
/// when binary32 rounding put it a hair below).
fn code(v: f32) -> u32 {
    let c = v as f64 * Q;
    let r = c.round();
    let c = if (c - r).abs() < 1e-3 { r } else { c.floor() };
    c.clamp(0.0, Q) as u32
}

fn combine(op: SequenceOp, xs: &mut [f64]) -> f64 {
    let n = xs.len() as f64;
    match op {
        SequenceOp::Mean => xs.iter().sum::<f64>() / n,
        SequenceOp::Median => {
            xs.sort_by(f64::total_cmp);
            xs[xs.len() / 2]
        }
        SequenceOp::Min => xs.iter().copied().fold(f64::INFINITY, f64::min),
        SequenceOp::Max => xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        SequenceOp::Add => xs.iter().sum(),
        SequenceOp::Abs => xs.iter().map(|v| v.abs()).sum(),
        SequenceOp::Subtract => xs[1..].iter().fold(xs[0], |acc, x| x - acc),
        SequenceOp::Multiply => xs.iter().product(),
        SequenceOp::Divide => {
            let q = xs[1..].iter().fold(
                xs[0] * Q,
                |acc, x| {
                    if acc == 0.0 { 0.0 } else { x * Q / acc }
                },
            );
            q / Q
        }
        SequenceOp::RootMeanSquare => (xs[1..].iter().map(|v| v * v).sum::<f64>() / n).sqrt(),
        SequenceOp::AddModulus => {
            let s: f64 = xs.iter().map(|v| v * Q).sum();
            s.rem_euclid(Q + 1.0) / Q
        }
        SequenceOp::And | SequenceOp::Or | SequenceOp::Xor => {
            let mut acc = code(xs[0] as f32);
            for x in &xs[1..] {
                let c = code(*x as f32);
                acc = match op {
                    SequenceOp::And => acc & c,
                    SequenceOp::Or => acc | c,
                    _ => acc ^ c,
                };
            }
            acc as f64 / Q
        }
        SequenceOp::Set => xs[0],
    }
}

/// Bring every picture to the widest channel layout of the list (grey →
/// RGB → RGBA) and check the extents agree.
fn harmonise(images: &[FloatImage]) -> Result<Vec<FloatImage>, Error> {
    let first = images
        .first()
        .ok_or_else(|| Error::invalid("oxideav-image-filter: an image sequence needs a picture"))?;
    if images
        .iter()
        .any(|i| i.width() != first.width() || i.height() != first.height())
    {
        return Err(Error::invalid(
            "oxideav-image-filter: the pictures of a sequence must share one extent",
        ));
    }
    let ch = images.iter().map(|i| i.channels()).max().unwrap_or(1);
    Ok(images
        .iter()
        .map(|i| {
            if i.channels() == ch {
                i.clone()
            } else if ch == 4 {
                i.to_rgba()
            } else {
                // Grey widened to RGB.
                i.to_rgba().to_channels(3)
            }
        })
        .collect())
}

/// `-evaluate-sequence op`: combine the pictures sample by sample.
pub fn evaluate_sequence(images: &[FloatImage], op: SequenceOp) -> Result<FloatImage, Error> {
    let imgs = harmonise(images)?;
    let first = &imgs[0];
    let len = first.samples().len();
    let mut out = vec![0.0f32; len];
    if op == SequenceOp::Median {
        // Whole pixels, ranked by the sum of their samples.
        let ch = first.channels();
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(imgs.len());
        for (p, o) in out.chunks_exact_mut(ch).enumerate() {
            order.clear();
            for (k, img) in imgs.iter().enumerate() {
                let px = &img.samples()[p * ch..(p + 1) * ch];
                order.push((px.iter().map(|&v| v as f64).sum(), k));
            }
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let k = order[order.len() / 2].1;
            o.copy_from_slice(&imgs[k].samples()[p * ch..(p + 1) * ch]);
        }
        return FloatImage::from_samples(first.width(), first.height(), ch, out);
    }
    let mut xs = vec![0.0f64; imgs.len()];
    for (i, o) in out.iter_mut().enumerate() {
        for (x, img) in xs.iter_mut().zip(&imgs) {
            *x = img.samples()[i] as f64;
        }
        *o = combine(op, &mut xs) as f32;
    }
    FloatImage::from_samples(first.width(), first.height(), first.channels(), out)
}

/// `-separate`: one grey picture per channel (alpha last when present).
pub fn separate(image: &FloatImage) -> Vec<FloatImage> {
    let ch = image.channels();
    (0..ch)
        .map(|c| {
            let data = image.samples().chunks_exact(ch).map(|px| px[c]).collect();
            FloatImage::from_samples(image.width(), image.height(), 1, data)
                .expect("one sample per pixel")
        })
        .collect()
}

/// `-combine`: grey pictures as the channels of one picture — one gives
/// grey, two grey + alpha (returned as RGBA), three RGB, four RGBA. A
/// colour input contributes its first channel.
pub fn combine_channels(channels: &[FloatImage]) -> Result<FloatImage, Error> {
    let n = channels.len();
    if n == 0 || n > 4 {
        return Err(Error::invalid(
            "oxideav-image-filter: combine takes one to four channel pictures",
        ));
    }
    let (w, h) = (channels[0].width(), channels[0].height());
    if channels.iter().any(|c| c.width() != w || c.height() != h) {
        return Err(Error::invalid(
            "oxideav-image-filter: the channel pictures must share one extent",
        ));
    }
    let take = |c: &FloatImage, i: usize| c.samples()[i * c.channels()];
    let px = w as usize * h as usize;
    let out_ch = match n {
        1 => 1,
        3 => 3,
        _ => 4,
    };
    let mut data = Vec::with_capacity(px * out_ch);
    for i in 0..px {
        match n {
            1 => data.push(take(&channels[0], i)),
            2 => {
                let g = take(&channels[0], i);
                data.extend_from_slice(&[g, g, g, take(&channels[1], i)]);
            }
            _ => {
                for c in channels {
                    data.push(take(c, i));
                }
            }
        }
    }
    FloatImage::from_samples(w, h, out_ch, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: f32) -> FloatImage {
        FloatImage::from_samples(1, 1, 1, vec![v]).unwrap()
    }

    #[test]
    fn measured_sequence_rules() {
        let three = [flat(0.2), flat(0.5), flat(0.4)];
        let four = [flat(0.2), flat(0.5), flat(0.4), flat(0.9)];
        let at = |imgs: &[FloatImage], op| evaluate_sequence(imgs, op).unwrap().samples()[0];
        let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(near(at(&three, SequenceOp::Mean), 0.366_667));
        assert!(near(at(&three, SequenceOp::Median), 0.4));
        assert!(near(at(&four, SequenceOp::Median), 0.5));
        assert!(near(at(&three, SequenceOp::Subtract), 0.1));
        assert!(near(at(&four, SequenceOp::Subtract), 0.8));
        assert!(near(at(&four, SequenceOp::Multiply), 0.036));
        assert!(near(at(&three, SequenceOp::RootMeanSquare), 0.369_686));
        assert!(near(at(&four, SequenceOp::RootMeanSquare), 0.552_269));
        assert!(near(at(&three, SequenceOp::AddModulus), 0.099_985));
        assert!(near(at(&three, SequenceOp::And), 0.133_333));
        assert!(at(&[flat(0.2), flat(0.5)], SequenceOp::Divide) * 65535.0 - 2.5 < 1e-3);
        let rgb = FloatImage::from_samples(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
        let parts = separate(&rgb);
        assert_eq!(parts.len(), 3);
        assert_eq!(combine_channels(&parts).unwrap(), rgb);
    }
}
