//! The threshold family of issue #12: `-white-threshold`,
//! `-black-threshold`, `-range-threshold`, `-color-threshold` and
//! `-random-threshold`.
//!
//! Every rule below was measured black-box on `magick` 7.1.2 Q16-HDRI
//! (`tests/magick_ops.rs` holds the comparison on `rose:`). Thresholds
//! are in normalised units (`1.0` = the quantum range); a 16-bit or
//! float picture is thresholded in binary32 through the float hop.
//!
//! - **White / black threshold.** With no channel mask (the default),
//!   the pixel's intensity — Rec. 709 luma `0.212656 R + 0.715158 G +
//!   0.072186 B` of the stored values, or the grey itself — is compared
//!   with each channel's own threshold, and that channel becomes `1`
//!   (white: intensity strictly above) or `0` (black: strictly below).
//!   An unspecified green / blue threshold repeats red's; an unspecified
//!   alpha threshold is `1.0`, so a default black threshold clears the
//!   alpha of every pixel whose intensity is below full white and a
//!   default white threshold never touches alpha. With an explicit mask
//!   each selected channel is compared with its own value instead.
//! - **Range threshold** `lb, lw, hw, hb`: `0` below `lb`, a linear ramp
//!   up to `1` at `lw`, `1` up to `hw`, a ramp down to `0` at `hb`, `0`
//!   above. With no mask the intensity drives every channel, alpha
//!   included; with a mask each selected channel ramps on its own value.
//! - **Colour threshold** `start-stop`: a pixel whose every channel
//!   (alpha included when the picture has one) lies in `[start, stop]`
//!   becomes opaque white, every other pixel opaque black.
//! - **Random threshold** `low, high`: each selected sample (every
//!   channel, alpha included, by default) is `0` below `low`, `1` above
//!   `high`, and between them compared with a fresh uniform threshold
//!   in `[low, high]`. The generator is this crate's own (seeded), so
//!   pixels do not match `magick`'s; the proportions do.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::composite::ChannelMask;
use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// Rec. 709 luma of a pixel's colour channels (the grey itself for a
/// one-channel pixel).
fn intensity(px: &[f32], cc: usize) -> f32 {
    if cc == 1 {
        px[0]
    } else {
        0.212_656 * px[0] + 0.715_158 * px[1] + 0.072_186 * px[2]
    }
}

/// Whether channel `c` of a `channels`-sample picture is selected.
fn selected(mask: &ChannelMask, c: usize, channels: usize) -> bool {
    let alpha = channels == 2 || channels == 4;
    if alpha && c == channels - 1 {
        return mask.alpha;
    }
    match c {
        0 => mask.red,
        1 => mask.green,
        _ => mask.blue,
    }
}

/// The threshold that applies to channel `c` (alpha takes index 3).
fn threshold_of(t: &[f32; 4], c: usize, channels: usize) -> f32 {
    let alpha = channels == 2 || channels == 4;
    if alpha && c == channels - 1 {
        t[3]
    } else {
        t[c.min(2)]
    }
}

/// Which way a level threshold pushes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    White,
    Black,
}

fn level_threshold(
    img: &mut FloatImage,
    side: Side,
    thresholds: &[f32; 4],
    mask: Option<&ChannelMask>,
) {
    let (ch, cc) = (img.channels(), img.color_channels());
    for px in img.samples_mut().chunks_exact_mut(ch) {
        let lum = intensity(px, cc);
        for (c, s) in px.iter_mut().enumerate() {
            let t = threshold_of(thresholds, c, ch);
            let v = match mask {
                None => lum,
                Some(m) => {
                    if !selected(m, c, ch) {
                        continue;
                    }
                    *s
                }
            };
            match side {
                Side::White if v > t => *s = 1.0,
                Side::Black if v < t => *s = 0.0,
                _ => {}
            }
        }
    }
}

macro_rules! level_threshold_type {
    ($name:ident, $side:expr, $doc:literal, $flag:literal) => {
        #[doc = $doc]
        #[non_exhaustive]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            /// Thresholds for red (grey), green, blue and alpha, normalised.
            pub thresholds: [f32; 4],
            /// `None` (the default): compare the pixel's intensity with
            /// each channel's threshold. `Some(mask)`: compare each
            /// selected channel's own value.
            pub channels: Option<ChannelMask>,
        }

        impl $name {
            #[doc = concat!("`", $flag, " t`: one threshold for the colour channels (alpha's is `1.0`).")]
            pub fn new(threshold: f32) -> Self {
                Self {
                    thresholds: [threshold, threshold, threshold, 1.0],
                    channels: None,
                }
            }

            #[doc = concat!("`", $flag, " r,g,b,a`: a threshold per channel.")]
            pub fn per_channel(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
                Self {
                    thresholds: [red, green, blue, alpha],
                    channels: None,
                }
            }

            /// Set the thresholds.
            pub fn with_thresholds(mut self, thresholds: [f32; 4]) -> Self {
                self.thresholds = thresholds;
                self
            }

            /// Compare the selected channels' own values (`-channel …`).
            pub fn with_channels(mut self, channels: ChannelMask) -> Self {
                self.channels = Some(channels);
                self
            }
        }

        impl PixelFilter for $name {
            fn accepted_formats(&self) -> &'static [PixelFormat] {
                layout::PACKED_8BIT_DEEP
            }

            fn is_pointwise(&self) -> bool {
                true
            }

            fn apply_pixels(
                &self,
                input: &VideoFrame,
                params: VideoStreamParams,
            ) -> Result<VideoFrame, Error> {
                deep::run_any(input, params, |mut img| {
                    level_threshold(&mut img, $side, &self.thresholds, self.channels.as_ref());
                    Ok(img)
                })
            }
        }
    };
}

level_threshold_type!(
    WhiteThreshold,
    Side::White,
    "`-white-threshold`: push to white every channel whose test value is strictly above its threshold (see the [module docs](self)).",
    "-white-threshold"
);
level_threshold_type!(
    BlackThreshold,
    Side::Black,
    "`-black-threshold`: push to black every channel whose test value is strictly below its threshold (see the [module docs](self)).",
    "-black-threshold"
);

/// `-range-threshold low-black,low-white,high-white,high-black`: a soft
/// band-pass threshold (hard when the ramp ends meet).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangeThreshold {
    /// Below this the result is `0`.
    pub low_black: f32,
    /// The ramp up reaches `1` here.
    pub low_white: f32,
    /// `1` up to here.
    pub high_white: f32,
    /// The ramp down reaches `0` here; above it the result is `0`.
    pub high_black: f32,
    /// `None` (the default): the intensity drives every channel, alpha
    /// included. `Some(mask)`: each selected channel on its own value.
    pub channels: Option<ChannelMask>,
}

impl RangeThreshold {
    /// Build from the four normalised edges.
    pub fn new(low_black: f32, low_white: f32, high_white: f32, high_black: f32) -> Self {
        Self {
            low_black,
            low_white,
            high_white,
            high_black,
            channels: None,
        }
    }

    /// Ramp the selected channels on their own values.
    pub fn with_channels(mut self, channels: ChannelMask) -> Self {
        self.channels = Some(channels);
        self
    }

    /// The band-pass response at `v`.
    pub fn response(&self, v: f32) -> f32 {
        if v < self.low_black {
            0.0
        } else if v < self.low_white {
            (v - self.low_black) / (self.low_white - self.low_black)
        } else if v <= self.high_white {
            1.0
        } else if v < self.high_black {
            (self.high_black - v) / (self.high_black - self.high_white)
        } else {
            0.0
        }
    }
}

impl PixelFilter for RangeThreshold {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn is_pointwise(&self) -> bool {
        true
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let (ch, cc) = (img.channels(), img.color_channels());
            for px in img.samples_mut().chunks_exact_mut(ch) {
                match &self.channels {
                    None => {
                        let r = self.response(intensity(px, cc));
                        px.fill(r);
                    }
                    Some(m) => {
                        for (c, s) in px.iter_mut().enumerate() {
                            if selected(m, c, ch) {
                                *s = self.response(*s);
                            }
                        }
                    }
                }
            }
            Ok(img)
        })
    }
}

/// `-color-threshold start-stop`: opaque white where every channel lies
/// in `[start, stop]`, opaque black elsewhere.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorThreshold {
    /// Lower corner (RGBA, normalised).
    pub start: [f32; 4],
    /// Upper corner (RGBA, normalised).
    pub stop: [f32; 4],
}

impl ColorThreshold {
    /// Build from the two RGBA corners (alpha `1.0` for an opaque colour).
    pub fn new(start: [f32; 4], stop: [f32; 4]) -> Self {
        Self { start, stop }
    }

    fn inside(&self, px: &[f32], ch: usize) -> bool {
        let within = |v: f32, c: usize| v >= self.start[c] && v <= self.stop[c];
        match ch {
            1 | 2 => {
                let lo = intensity(&self.start, 3);
                let hi = intensity(&self.stop, 3);
                px[0] >= lo && px[0] <= hi && (ch == 1 || within(px[1], 3))
            }
            _ => (0..ch).all(|c| within(px[c], c)),
        }
    }
}

impl PixelFilter for ColorThreshold {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn is_pointwise(&self) -> bool {
        true
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let (ch, cc) = (img.channels(), img.color_channels());
            for px in img.samples_mut().chunks_exact_mut(ch) {
                let v = if self.inside(px, ch) { 1.0 } else { 0.0 };
                px[..cc].fill(v);
                if ch > cc {
                    px[ch - 1] = 1.0;
                }
            }
            Ok(img)
        })
    }
}

/// `-random-threshold low,high` with a seed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomThreshold {
    /// Below this a sample becomes `0`.
    pub low: f32,
    /// Above this a sample becomes `1`.
    pub high: f32,
    /// Generator seed.
    pub seed: u64,
    /// The channels thresholded (every channel, alpha included, by
    /// default).
    pub channels: ChannelMask,
}

impl RandomThreshold {
    /// Build from the normalised bounds.
    pub fn new(low: f32, high: f32) -> Self {
        Self {
            low,
            high,
            seed: 0,
            channels: ChannelMask::ALL,
        }
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Restrict the channels thresholded.
    pub fn with_channels(mut self, channels: ChannelMask) -> Self {
        self.channels = channels;
        self
    }
}

impl PixelFilter for RandomThreshold {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let mut rng = crate::noise::Rng::new(self.seed);
        deep::run_any(input, params, |mut img| {
            let ch = img.channels();
            let (lo, hi) = (self.low.min(self.high), self.low.max(self.high));
            for px in img.samples_mut().chunks_exact_mut(ch) {
                for (c, s) in px.iter_mut().enumerate() {
                    if !selected(&self.channels, c, ch) {
                        continue;
                    }
                    let v = *s;
                    *s = if v < lo {
                        0.0
                    } else if v > hi {
                        1.0
                    } else {
                        let t = lo + (hi - lo) * rng.unit() as f32;
                        if v >= t { 1.0 } else { 0.0 }
                    };
                }
            }
            Ok(img)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageFilter;
    use oxideav_core::VideoPlane;

    fn px(rgba: [u8; 4]) -> (VideoFrame, VideoStreamParams) {
        (
            VideoFrame {
                pts: None,
                planes: vec![VideoPlane {
                    stride: 4,
                    data: rgba.to_vec(),
                }],
            },
            VideoStreamParams {
                format: PixelFormat::Rgba,
                width: 1,
                height: 1,
            },
        )
    }

    fn run(f: &dyn ImageFilter, rgba: [u8; 4]) -> Vec<u8> {
        let (i, p) = px(rgba);
        f.apply(&i, p).unwrap().planes[0].data.clone()
    }

    #[test]
    fn level_thresholds_follow_the_measured_rules() {
        // Intensity of (100, 150, 200) is 56.07 %.
        assert_eq!(
            run(&WhiteThreshold::new(0.55), [100, 150, 200, 128]),
            [255, 255, 255, 128]
        );
        assert_eq!(
            run(&WhiteThreshold::new(0.57), [100, 150, 200, 128]),
            [100, 150, 200, 128]
        );
        assert_eq!(
            run(&BlackThreshold::new(0.1), [100, 150, 200, 128]),
            [100, 150, 200, 0]
        );
        assert_eq!(
            run(
                &BlackThreshold::new(0.6)
                    .with_channels(ChannelMask::new(true, false, false, false)),
                [100, 150, 200, 255]
            ),
            [0, 150, 200, 255]
        );
    }

    #[test]
    fn range_color_and_random() {
        let r = RangeThreshold::new(0.2, 0.4, 0.6, 0.8);
        assert!((r.response(0.3) - 0.5).abs() < 1e-6);
        assert_eq!(r.response(0.2), 0.0);
        assert_eq!(r.response(0.6), 1.0);
        assert_eq!(r.response(0.8), 0.0);
        let c = ColorThreshold::new([0.3, 0.5, 0.7, 0.4], [0.5, 0.7, 0.9, 0.6]);
        assert_eq!(run(&c, [100, 150, 200, 128]), [255, 255, 255, 255]);
        assert_eq!(run(&c, [100, 150, 200, 255]), [0, 0, 0, 255]);
        // A flat 30 % grey between 20 % and 40 %: about half turns white.
        let grey = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 100,
                data: vec![77; 100 * 100],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 100,
            height: 100,
        };
        let out = RandomThreshold::new(0.2, 0.4)
            .with_seed(9)
            .apply(&grey, p)
            .unwrap();
        let white = out.planes[0].data.iter().filter(|&&v| v == 255).count();
        assert!((4_500..5_500).contains(&white), "{white}");
        let t = RandomThreshold::new(0.5, 0.5);
        assert_eq!(run(&t, [100, 150, 200, 77]), [0, 255, 255, 0]);
    }
}
