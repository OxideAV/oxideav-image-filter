//! Skew estimation by the projection-profile (discrete Radon) method
//! (issue #15).
//!
//! A page of text lines, ruled lines or a table has a direction along
//! which its ink lines up. Summing the ink along parallel lines of a
//! trial slope gives a *projection profile*: at the right slope the
//! lines of text fall into few sums and the gaps between them into
//! others, so the profile is as jagged as it can be; at a wrong slope
//! every sum crosses ink and gap alike and the profile is smooth. The
//! estimate is the slope whose profile has the largest energy of
//! differences, `Σ (P[i + 1] − P[i])²`.
//!
//! The profiles of every slope `s / W` (`s = −(W − 1) … W − 1`, `W` the
//! width rounded up to a power of two) come from one recursive pass:
//! the sum along a digital line across a strip `2w` columns wide that
//! drops `s` rows is the sum across the left half dropping `⌊s / 2⌋`
//! rows plus the sum across the right half, started `⌈s / 2⌉` rows
//! lower, dropping `⌊s / 2⌋` rows again:
//!
//! ```text
//! R₂w(2j,     y) = R_w(left, j, y) + R_w(right, j, y + j)
//! R₂w(2j + 1, y) = R_w(left, j, y) + R_w(right, j, y + j + 1)
//! ```
//!
//! starting from single columns. That is `log₂ W` passes of `W · (H +
//! W)` additions for all `W` slopes at once, against `W² · H` for
//! summing each slope on its own. Rising slopes are the falling slopes
//! of the picture turned upside down. The angle found is
//! `atan(s / W)`, so its resolution is `1 / W` in slope (0.06° for a
//! width of 1 000) and its range ±45°.

/// The sharpness of the projection profile of every slope `s / W`,
/// `s = 0 … limit` (rows dropping to the right), of a binary picture
/// `ink` (`width × height`, row-major). `W` is `width` rounded up to a
/// power of two. Only what the slopes up to `limit` need is computed: a
/// strip `w` columns wide takes part with its slopes up to `⌈limit · w /
/// W⌉ + 1`, and a line starts at most `limit + 1` rows above the picture.
fn profile_sharpness(ink: &[bool], width: usize, height: usize, limit: usize) -> Vec<f64> {
    let big = width.next_power_of_two();
    let limit = limit.min(big - 1);
    let top = (limit + 2).min(big);
    let rows = height + top;
    let mut cur = vec![0u32; big * rows];
    for x in 0..width {
        let col = &mut cur[x * rows..(x + 1) * rows];
        for (y, slot) in col[top..].iter_mut().enumerate() {
            *slot = ink[y * width + x] as u32;
        }
    }
    // Slopes a strip of `w` columns must provide.
    let needed = |w: usize| (w - 1).min((limit * w).div_ceil(big) + 1);
    let mut next = vec![0u32; big * rows];
    let mut strip = 1usize;
    while strip < big {
        // `cur[(i · strip + j) · rows + y]`: strip `i`, slope `j`.
        let pairs = needed(2 * strip) / 2;
        for (pair, out) in next.chunks_exact_mut(2 * strip * rows).enumerate() {
            let left = &cur[2 * pair * strip * rows..][..strip * rows];
            let right = &cur[(2 * pair + 1) * strip * rows..][..strip * rows];
            for j in 0..=pairs.min(strip - 1) {
                let (l, r) = (&left[j * rows..][..rows], &right[j * rows..][..rows]);
                let (even, odd) = out[2 * j * rows..][..2 * rows].split_at_mut(rows);
                let below = r.iter().skip(j).chain(std::iter::repeat(&0));
                for (((e, o), &a), (&b0, &b1)) in even
                    .iter_mut()
                    .zip(odd.iter_mut())
                    .zip(l)
                    .zip(below.clone().zip(below.skip(1)))
                {
                    *e = a + b0;
                    *o = a + b1;
                }
            }
        }
        std::mem::swap(&mut cur, &mut next);
        strip *= 2;
    }
    cur.chunks_exact(rows)
        .take(limit + 1)
        .map(|profile| {
            profile
                .windows(2)
                .map(|p| {
                    let d = p[1] as f64 - p[0] as f64;
                    d * d
                })
                .sum()
        })
        .collect()
}

/// The skew of a binary picture in degrees, positive when its lines
/// drop to the right (rotate by the negative to level them), searched
/// over `|angle| ≤ max_degrees` (at most 45). `0` for an empty picture
/// or one with no ink.
pub fn skew_degrees(ink: &[bool], width: usize, height: usize, max_degrees: f64) -> f64 {
    if width == 0 || height == 0 || ink.len() < width * height || !ink.iter().any(|&b| b) {
        return 0.0;
    }
    let big = width.next_power_of_two() as f64;
    let limit = max_degrees.clamp(0.0, 45.0).to_radians().tan() * big;
    let steps = (limit + 1e-9).floor() as usize;
    let falling = profile_sharpness(ink, width, height, steps);
    let flipped: Vec<bool> = ink[..width * height]
        .chunks_exact(width)
        .rev()
        .flatten()
        .copied()
        .collect();
    let rising = profile_sharpness(&flipped, width, height, steps);
    // Level first, then by increasing tilt, falling before rising: a tie
    // goes to the smaller angle.
    let mut best = (falling[0], 0.0f64);
    for (s, (&fall, &rise)) in falling.iter().zip(&rising).enumerate().skip(1) {
        if s as f64 > limit + 1e-9 {
            break;
        }
        for (score, sign) in [(fall, 1.0), (rise, -1.0)] {
            if score > best.0 {
                best = (score, sign * s as f64);
            }
        }
    }
    (best.1 / big).atan().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ruled(w: usize, h: usize, slope: f64, period: i64) -> Vec<bool> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let line = (y - slope * (x - w as f64 / 2.0)).round() as i64;
                line.rem_euclid(period) < 2
            })
            .collect()
    }

    #[test]
    fn recovers_the_slope_of_ruled_lines_to_the_grid_step() {
        for (w, h) in [(256usize, 200usize), (300, 180), (97, 150)] {
            let step = 1.0 / w.next_power_of_two() as f64;
            for slope in [0.0, 0.03125, -0.0625, 0.11, -0.2, 0.4] {
                let got = skew_degrees(&ruled(w, h, slope, 14), w, h, 45.0);
                let want = slope.atan().to_degrees();
                let tolerance = (1.5 * step).atan().to_degrees() + 1e-9;
                assert!(
                    (got - want).abs() <= tolerance,
                    "{w}x{h} slope {slope}: got {got}°, want {want}° ± {tolerance}°"
                );
            }
        }
    }

    #[test]
    fn the_search_range_is_honoured_and_empty_pictures_are_level() {
        let ink = ruled(128, 128, 0.3, 12);
        let limited = skew_degrees(&ink, 128, 128, 5.0);
        assert!(limited.abs() <= 5.0 + 1e-9, "{limited}");
        assert_eq!(skew_degrees(&[false; 64], 8, 8, 45.0), 0.0);
        assert_eq!(skew_degrees(&[], 0, 0, 45.0), 0.0);
        assert_eq!(skew_degrees(&ink, 128, 128, 0.0), 0.0);
    }

    #[test]
    fn digital_line_sums_are_exact_for_level_and_unit_slopes() {
        // One full row of ink: the level profile has a single spike of
        // `w`, so its sharpness is 2 w²; the 45° profile is flat.
        let (w, h) = (16usize, 16usize);
        let ink: Vec<bool> = (0..w * h).map(|i| i / w == 5).collect();
        let s = profile_sharpness(&ink, w, h, w - 1);
        // A limited search computes the same sums for the slopes it keeps.
        for limit in [0usize, 1, 3, 7] {
            assert_eq!(profile_sharpness(&ink, w, h, limit), s[..=limit]);
        }
        assert_eq!(s[0], 2.0 * (w * w) as f64);
        assert!(s[w - 1] <= 2.0 * w as f64 + 2.0);
    }
}
