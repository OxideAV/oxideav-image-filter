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

/// `-poly`: the weighted sum of powers `Σ wᵢ · imageᵢ^eᵢ`, one
/// `(weight, exponent)` term per picture, on every channel.
///
/// The pictures must share one shape and there must be one term each.
/// An exponent of 1 is the plain weighted sum (so `[(0.5, 1), (0.5, 1)]`
/// is the mean of two pictures); a negative base under a fractional
/// exponent contributes nothing.
pub fn poly(images: &[FloatImage], terms: &[(f64, f64)]) -> Result<FloatImage, Error> {
    let first = images
        .first()
        .ok_or_else(|| Error::invalid("oxideav-image-filter: poly needs at least one picture"))?;
    if terms.len() != images.len() {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: poly needs one (weight, exponent) term per picture: \
             {} pictures, {} terms",
            images.len(),
            terms.len()
        )));
    }
    if images.iter().any(|i| {
        i.width() != first.width()
            || i.height() != first.height()
            || i.channels() != first.channels()
    }) {
        return Err(Error::invalid(
            "oxideav-image-filter: the pictures of a poly must share one shape",
        ));
    }
    let mut out = vec![0.0f64; first.samples().len()];
    for (img, &(weight, exponent)) in images.iter().zip(terms) {
        for (o, &v) in out.iter_mut().zip(img.samples()) {
            let p = if exponent == 1.0 {
                v as f64
            } else {
                (v as f64).powf(exponent)
            };
            if p.is_finite() {
                *o += weight * p;
            }
        }
    }
    FloatImage::from_samples(
        first.width(),
        first.height(),
        first.channels(),
        out.into_iter().map(|v| v as f32).collect(),
    )
}

/// `-channel-fx`: move channels about by a short expression.
///
/// The expression is a comma-separated list of steps. Every step reads
/// the *original* picture and writes the result, so the steps do not
/// see each other's writes:
///
/// - `a<=>b` — exchange channels `a` and `b`;
/// - `a=>b` — copy channel `a` into `b`;
/// - `a=value` — set channel `a` to a constant: a fraction of the range
///   as `50%`, or a normalised number;
/// - `a` on its own — copy channel `a` into the next channel of the
///   result (the first bare name fills the first channel, the second
///   the second, …). An expression that is one bare name gives that
///   channel as a grey picture.
///
/// Channels are `red`, `green`, `blue`, `alpha` (or `r`, `g`, `b`, `a`;
/// `gray` is the first channel). Naming `alpha` on a picture without
/// one adds an opaque alpha channel first; a grey picture is widened to
/// colour when a colour channel other than the first is named.
pub fn channel_fx(image: &FloatImage, expression: &str) -> Result<FloatImage, Error> {
    let bad = |what: &str| {
        Error::invalid(format!(
            "oxideav-image-filter: channel-fx: {what} in '{expression}'"
        ))
    };
    let channel = |name: &str| -> Result<usize, Error> {
        Ok(match name.trim().to_ascii_lowercase().as_str() {
            "red" | "r" | "gray" | "grey" | "0" => 0,
            "green" | "g" | "1" => 1,
            "blue" | "b" | "2" => 2,
            "alpha" | "a" | "opacity" | "3" => 3,
            other => return Err(bad(&format!("unknown channel '{other}'"))),
        })
    };
    enum Step {
        Swap(usize, usize),
        Copy(usize, usize),
        Set(usize, f32),
        Take(usize),
    }
    let mut steps = Vec::new();
    for step in expression
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        steps.push(if let Some((a, b)) = step.split_once("<=>") {
            Step::Swap(channel(a)?, channel(b)?)
        } else if let Some((a, b)) = step.split_once("=>") {
            Step::Copy(channel(a)?, channel(b)?)
        } else if let Some((a, value)) = step.split_once('=') {
            let value = value.trim();
            let v = match value.strip_suffix('%') {
                Some(pc) => pc.trim().parse::<f32>().map(|p| p / 100.0),
                None => value.parse::<f32>(),
            }
            .map_err(|_| bad(&format!("bad value '{value}'")))?;
            Step::Set(channel(a)?, v)
        } else {
            Step::Take(channel(step)?)
        });
    }
    // The layout that holds every channel named (layouts only grow).
    let mut needs_alpha = image.has_alpha();
    let mut needs_colour = image.color_channels() == 3;
    let mut bare = 0usize;
    for s in &steps {
        let named: &[usize] = match s {
            Step::Swap(a, b) | Step::Copy(a, b) => &[*a, *b],
            Step::Set(a, _) => &[*a],
            Step::Take(a) => {
                bare += 1;
                &[*a, bare - 1]
            }
        };
        needs_alpha |= named.contains(&3);
        needs_colour |= named.iter().any(|c| (1..3).contains(c));
    }
    let single_take = matches!(steps.as_slice(), [Step::Take(_)]);
    if single_take {
        // One bare name: that channel as grey (opaque when there is no
        // alpha to take).
        let Step::Take(a) = steps[0] else {
            unreachable!()
        };
        let ch = image.channels();
        let data = image
            .samples()
            .chunks_exact(ch)
            .map(|px| match a {
                3 if image.has_alpha() => px[ch - 1],
                3 => 1.0,
                c => px[c.min(image.color_channels() - 1)],
            })
            .collect();
        return FloatImage::from_samples(image.width(), image.height(), 1, data);
    }
    let src = match (needs_colour, needs_alpha) {
        (true, true) => image.to_rgba(),
        (true, false) => image.to_channels(3),
        (false, true) if !image.has_alpha() => image.to_rgba(),
        _ => image.clone(),
    };
    let ch = src.channels();
    let cc = src.color_channels();
    let slot = |c: usize| if c == 3 { ch - 1 } else { c.min(cc - 1) };
    let mut out = src.clone();
    let mut next = 0usize;
    for s in &steps {
        let moves: Vec<(usize, usize)> = match *s {
            Step::Swap(a, b) => vec![(a, b), (b, a)],
            Step::Copy(a, b) => vec![(a, b)],
            Step::Take(a) => {
                next += 1;
                vec![(a, next - 1)]
            }
            Step::Set(a, v) => {
                let sa = slot(a);
                out.samples_mut()
                    .chunks_exact_mut(ch)
                    .for_each(|px| px[sa] = v);
                continue;
            }
        };
        for (from, to) in moves {
            let (sf, st) = (slot(from), slot(to));
            for (o, i) in out
                .samples_mut()
                .chunks_exact_mut(ch)
                .zip(src.samples().chunks_exact(ch))
            {
                o[st] = i[sf];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod poly_and_channel_fx_tests {
    use super::*;

    fn rgb(px: &[[f32; 3]]) -> FloatImage {
        FloatImage::from_samples(
            px.len() as u32,
            1,
            3,
            px.iter().flatten().copied().collect(),
        )
        .unwrap()
    }

    #[test]
    fn poly_is_the_weighted_sum_of_powers() {
        let a = rgb(&[[0.25, 0.5, 1.0]]);
        let b = rgb(&[[0.5, 0.5, 0.0]]);
        let out = poly(&[a.clone(), b.clone()], &[(0.5, 1.0), (2.0, 2.0)]).unwrap();
        assert_eq!(out.samples(), &[0.125 + 0.5, 0.25 + 0.5, 0.5]);
        let mean = poly(&[a.clone(), b.clone()], &[(0.5, 1.0), (0.5, 1.0)]).unwrap();
        assert_eq!(mean.samples(), &[0.375, 0.5, 0.5]);
        assert!(poly(std::slice::from_ref(&a), &[]).is_err());
        assert!(poly(&[], &[]).is_err());
        let gray = FloatImage::from_samples(1, 1, 1, vec![0.5]).unwrap();
        assert!(poly(&[a, gray], &[(1.0, 1.0), (1.0, 1.0)]).is_err());
    }

    #[test]
    fn channel_fx_swaps_copies_sets_and_selects() {
        let img = rgb(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let swapped = channel_fx(&img, "red<=>blue").unwrap();
        assert_eq!(swapped.samples(), &[0.3, 0.2, 0.1, 0.6, 0.5, 0.4]);
        let copied = channel_fx(&img, "red=>green").unwrap();
        assert_eq!(copied.samples(), &[0.1, 0.1, 0.3, 0.4, 0.4, 0.6]);
        let set = channel_fx(&img, "green=50%, blue=0").unwrap();
        assert_eq!(set.samples(), &[0.1, 0.5, 0.0, 0.4, 0.5, 0.0]);
        let green = channel_fx(&img, "green").unwrap();
        assert_eq!((green.channels(), green.samples()), (1, &[0.2f32, 0.5][..]));
        // Every step reads the original; alpha is added on demand.
        let out = channel_fx(&img, "red<=>green, green=>alpha").unwrap();
        assert_eq!(out.channels(), 4);
        assert_eq!(&out.samples()[..4], &[0.2, 0.1, 0.3, 0.2]);
        // Bare names fill the result's channels in turn.
        let out = channel_fx(&img, "red, blue").unwrap();
        assert_eq!(&out.samples()[..3], &[0.1, 0.3, 0.3]);
        assert!(channel_fx(&img, "mauve=>red").is_err());
        assert!(channel_fx(&img, "red=lots").is_err());
        assert_eq!(channel_fx(&img, "alpha").unwrap().samples(), &[1.0, 1.0]);
    }
}
