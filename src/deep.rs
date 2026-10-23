//! The float working picture behind every filter's deep path.
//!
//! A filter that takes the packed float layouts (`GrayF32Le` /
//! `RgbF32Le` / `RgbaF32Le`, [`layout::FLOAT`]) natively gets every
//! other deep layout — 16-bit words (`Rgb48Le`, `Rgba64Le`, `Gray16Le`,
//! `Ya16Le`, the 10/12/14/16-bit planar YUV(A) and GBR(A) families) and
//! the planar floats — through the **float hop** in [`crate::route`]:
//! the picture is converted to the first of `RgbaF32Le` / `RgbF32Le` /
//! `GrayF32Le` the filter takes (a pure normalisation for the integer
//! RGB / grey families, `code / (2^bits − 1)`; the matrix for YUV), the
//! filter runs on binary32 samples with no intermediate quantisation,
//! and the result is converted back to the source layout (saturating to
//! the word's full scale). A 16-bit picture therefore never rounds
//! through 8 bits, and a float picture keeps out-of-range intermediates
//! the way a scene-referred (HDRI) pipeline does.
//!
//! Inside the filter the picture is a [`FloatImage`]: `channels`
//! interleaved binary32 samples per pixel, row-major and tight, with
//! `1.0` the nominal white of the integer families and alpha (when
//! present, always the last channel) straight in `[0, 1]`. Tone values
//! that parameterise a filter in 8-bit units (`Threshold::new(128)`,
//! `Level::new(16, 235, 1.0)`) map onto this scale as `v / 255`.

use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

use crate::{VideoStreamParams, layout};

/// `true` for the three packed float layouts a filter's deep path takes
/// natively (`GrayF32Le`, `RgbF32Le`, `RgbaF32Le`).
pub fn is_float(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::GrayF32Le | PixelFormat::RgbF32Le | PixelFormat::RgbaF32Le
    )
}

/// Channels per pixel of a packed float layout (`None` otherwise).
pub fn float_channels(fmt: PixelFormat) -> Option<usize> {
    Some(match fmt {
        PixelFormat::GrayF32Le => 1,
        PixelFormat::RgbF32Le => 3,
        PixelFormat::RgbaF32Le => 4,
        _ => return None,
    })
}

/// The packed float layout holding `channels` samples per pixel.
pub fn float_format(channels: usize) -> Option<PixelFormat> {
    Some(match channels {
        1 => PixelFormat::GrayF32Le,
        3 => PixelFormat::RgbF32Le,
        4 => PixelFormat::RgbaF32Le,
        _ => return None,
    })
}

/// A picture as binary32 samples: `channels` interleaved per pixel,
/// row-major, tight. Grey is one channel, RGB three, RGBA four with
/// alpha last.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FloatImage {
    width: u32,
    height: u32,
    channels: usize,
    data: Vec<f32>,
}

impl FloatImage {
    /// A zero-filled `width × height` picture of `channels` samples per
    /// pixel (`1`, `3` or `4`).
    pub fn new(width: u32, height: u32, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0.0; width as usize * height as usize * channels],
        }
    }

    /// Wrap existing samples (`data.len()` must be `width × height ×
    /// channels`).
    pub fn from_samples(
        width: u32,
        height: u32,
        channels: usize,
        data: Vec<f32>,
    ) -> Result<Self, Error> {
        let need = width as usize * height as usize * channels;
        if data.len() != need {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: FloatImage holds {} samples, {need} needed for {width}x{height}x{channels}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Unpack a validated `params`-shaped frame in one of the packed
    /// float layouts.
    pub fn from_frame(frame: &VideoFrame, params: VideoStreamParams) -> Result<Self, Error> {
        let channels = float_channels(params.format).ok_or_else(|| {
            Error::unsupported(format!(
                "oxideav-image-filter: {:?} is not a packed float layout",
                params.format
            ))
        })?;
        let w = params.width as usize;
        let h = params.height as usize;
        let plane = frame.planes.first().ok_or_else(|| {
            Error::invalid("oxideav-image-filter: float picture carries no plane")
        })?;
        let row_bytes = w * channels * 4;
        if plane.stride < row_bytes || plane.data.len() < plane.stride * h {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: float plane holds {} bytes at stride {}, {}x{} rows of {row_bytes} needed",
                plane.data.len(),
                plane.stride,
                w,
                h
            )));
        }
        let mut data = Vec::with_capacity(w * h * channels);
        for y in 0..h {
            let row = &plane.data[y * plane.stride..y * plane.stride + row_bytes];
            data.extend(
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            );
        }
        Ok(Self {
            width: params.width,
            height: params.height,
            channels,
            data,
        })
    }

    /// Pack into a tight single-plane frame in [`Self::format`].
    pub fn into_frame(self, pts: Option<i64>) -> VideoFrame {
        let stride = self.width as usize * self.channels * 4;
        let mut data = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        VideoFrame {
            pts,
            planes: vec![VideoPlane { stride, data }],
        }
    }

    /// The packed float layout of this picture (by channel count).
    pub fn format(&self) -> PixelFormat {
        float_format(self.channels).unwrap_or(PixelFormat::RgbaF32Le)
    }

    /// The stream parameters of this picture.
    pub fn params(&self) -> VideoStreamParams {
        VideoStreamParams {
            format: self.format(),
            width: self.width,
            height: self.height,
        }
    }

    /// Pixels per row.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Rows.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Samples per pixel (alpha included).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// `true` when the last channel is alpha (four channels).
    pub fn has_alpha(&self) -> bool {
        self.channels == 4
    }

    /// Colour (tone) channels per pixel: `channels` minus alpha.
    pub fn color_channels(&self) -> usize {
        if self.has_alpha() {
            self.channels - 1
        } else {
            self.channels
        }
    }

    /// All samples, row-major, `channels` per pixel.
    pub fn samples(&self) -> &[f32] {
        &self.data
    }

    /// All samples, mutable.
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Take the samples out.
    pub fn into_samples(self) -> Vec<f32> {
        self.data
    }

    /// The samples of the pixel at `(x, y)` (`channels` of them).
    pub fn pixel(&self, x: u32, y: u32) -> &[f32] {
        let i = (y as usize * self.width as usize + x as usize) * self.channels;
        &self.data[i..i + self.channels]
    }

    /// The samples of the pixel at `(x, y)`, mutable.
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut [f32] {
        let i = (y as usize * self.width as usize + x as usize) * self.channels;
        &mut self.data[i..i + self.channels]
    }

    /// Sample `c` of the pixel at `(x, y)` with the coordinates clamped
    /// into the picture (replicate-edge addressing for neighbourhood
    /// filters).
    pub fn sample_clamped(&self, x: i64, y: i64, c: usize) -> f32 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        self.data[(y * self.width as usize + x) * self.channels + c]
    }

    /// Bilinear sample at the continuous position `(sx, sy)` (clamped
    /// into the picture), every channel, written to `dst`.
    pub fn bilinear_into(&self, sx: f32, sy: f32, dst: &mut [f32]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let x = if sx.is_nan() {
            0.0
        } else {
            sx.clamp(0.0, (w - 1) as f32)
        };
        let y = if sy.is_nan() {
            0.0
        } else {
            sy.clamp(0.0, (h - 1) as f32)
        };
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let ch = self.channels;
        let p00 = &self.data[(y0 * w + x0) * ch..(y0 * w + x0) * ch + ch];
        let p10 = &self.data[(y0 * w + x1) * ch..(y0 * w + x1) * ch + ch];
        let p01 = &self.data[(y1 * w + x0) * ch..(y1 * w + x0) * ch + ch];
        let p11 = &self.data[(y1 * w + x1) * ch..(y1 * w + x1) * ch + ch];
        for c in 0..ch.min(dst.len()) {
            dst[c] = p00[c] * (1.0 - fx) * (1.0 - fy)
                + p10[c] * fx * (1.0 - fy)
                + p01[c] * (1.0 - fx) * fy
                + p11[c] * fx * fy;
        }
    }

    /// A `width × height` picture of this picture's channel count
    /// filled with `fill` (one value per channel, alpha last).
    pub fn filled_like(&self, width: u32, height: u32, fill: &[f32]) -> Self {
        let mut out = Self::new(width, height, self.channels);
        for px in out.data.chunks_exact_mut(self.channels) {
            for (v, f) in px.iter_mut().zip(fill.iter()) {
                *v = *f;
            }
        }
        out
    }

    /// Apply `tone` to every colour channel of every pixel, leaving
    /// alpha untouched.
    pub fn map_tone(mut self, tone: impl Fn(f32) -> f32) -> Self {
        self.map_tone_in_place(tone);
        self
    }

    /// [`Self::map_tone`] in place.
    pub fn map_tone_in_place(&mut self, tone: impl Fn(f32) -> f32) {
        let cc = self.color_channels();
        let ch = self.channels;
        for px in self.data.chunks_exact_mut(ch) {
            for v in &mut px[..cc] {
                *v = tone(*v);
            }
        }
    }

    /// Apply `tone` to channel `c` of every pixel.
    pub fn map_channel_in_place(&mut self, c: usize, tone: impl Fn(f32) -> f32) {
        let ch = self.channels;
        if c >= ch {
            return;
        }
        for px in self.data.chunks_exact_mut(ch) {
            px[c] = tone(px[c]);
        }
    }

    /// Apply `f` to every pixel's colour channels as a slice (alpha
    /// untouched; a grey picture passes a one-sample slice).
    pub fn map_pixels_in_place(&mut self, mut f: impl FnMut(&mut [f32])) {
        let cc = self.color_channels();
        let ch = self.channels;
        for px in self.data.chunks_exact_mut(ch) {
            f(&mut px[..cc]);
        }
    }

    /// Mean of the colour channels of a pixel slice (`rec601`-free
    /// plain average, the grey of an RGB triple when no luma weighting
    /// is wanted).
    pub fn pixel_mean(px: &[f32]) -> f32 {
        if px.is_empty() {
            0.0
        } else {
            px.iter().sum::<f32>() / px.len() as f32
        }
    }

    /// Rec. 601 luma of a colour slice (`0.299 R + 0.587 G + 0.114 B`),
    /// the grey sample itself for a one-channel picture.
    pub fn luma(px: &[f32]) -> f32 {
        match px.len() {
            0 => 0.0,
            1 | 2 => px[0],
            _ => 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2],
        }
    }

    /// Widen to RGBA: grey replicated to three channels, alpha `1.0`
    /// added when missing.
    pub fn to_rgba(&self) -> Self {
        if self.channels == 4 {
            return self.clone();
        }
        let n = self.width as usize * self.height as usize;
        let mut data = Vec::with_capacity(n * 4);
        for px in self.data.chunks_exact(self.channels) {
            match self.channels {
                1 => data.extend_from_slice(&[px[0], px[0], px[0], 1.0]),
                3 => data.extend_from_slice(&[px[0], px[1], px[2], 1.0]),
                _ => data.extend_from_slice(&[px[0], px[0], px[0], 1.0]),
            }
        }
        Self {
            width: self.width,
            height: self.height,
            channels: 4,
            data,
        }
    }

    /// Narrow an RGBA picture back to `channels` (grey takes the Rec.
    /// 601 luma, RGB drops alpha).
    pub fn to_channels(&self, channels: usize) -> Self {
        if channels == self.channels {
            return self.clone();
        }
        let n = self.width as usize * self.height as usize;
        let mut data = Vec::with_capacity(n * channels);
        for px in self.data.chunks_exact(self.channels) {
            let rgba = match self.channels {
                1 => [px[0], px[0], px[0], 1.0],
                3 => [px[0], px[1], px[2], 1.0],
                _ => [px[0], px[1], px[2], px[3]],
            };
            match channels {
                1 => data.push(Self::luma(&rgba[..3])),
                3 => data.extend_from_slice(&rgba[..3]),
                _ => data.extend_from_slice(&rgba),
            }
        }
        Self {
            width: self.width,
            height: self.height,
            channels,
            data,
        }
    }
}

/// The 8-bit code of a normalised tone, `v · 255`, snapped onto the
/// integer grid when within `1e-4` of it. Discontinuous operators
/// (threshold, solarize, posterize) compare on this so a picture widened
/// from 8 bits (`code / 255` is not exact in binary32) lands on the same
/// side of the step as its 8-bit original; the snap is four orders of
/// magnitude below 16-bit resolution.
#[inline]
pub fn code8(v: f32) -> f32 {
    let c = v * 255.0;
    let r = c.round();
    if (c - r).abs() < 1e-4 { r } else { c }
}

/// An 8-bit RGBA background colour as the fill of a `channels`-channel
/// float picture (grey takes the red component, alpha `1.0` when the
/// picture has none).
pub fn background(bg: [u8; 4], channels: usize) -> Vec<f32> {
    let f = bg.map(|v| v as f32 / 255.0);
    match channels {
        1 => vec![f[0]],
        3 => vec![f[0], f[1], f[2]],
        _ => vec![f[0], f[1], f[2], f[3]],
    }
}

/// Run `kernel` on a packed-float frame: unpack, apply, repack. The
/// one-liner every filter's deep path is built on.
pub fn run(
    input: &VideoFrame,
    params: VideoStreamParams,
    kernel: impl FnOnce(FloatImage) -> Result<FloatImage, Error>,
) -> Result<VideoFrame, Error> {
    let img = FloatImage::from_frame(input, params)?;
    Ok(kernel(img)?.into_frame(input.pts))
}

/// Run a per-sample tone function on a packed-float frame (alpha
/// untouched).
pub fn map_tone(
    input: &VideoFrame,
    params: VideoStreamParams,
    tone: impl Fn(f32) -> f32,
) -> Result<VideoFrame, Error> {
    run(input, params, |img| Ok(img.map_tone(tone)))
}

/// Separable convolution with replicate-edge addressing: `kx` along
/// rows then `ky` down columns, every channel (alpha included when
/// `with_alpha`). Both kernels are odd-length and centred.
pub fn convolve_separable(
    img: &FloatImage,
    kx: &[f32],
    ky: &[f32],
    with_alpha: bool,
) -> FloatImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let ch = img.channels;
    let nc = if with_alpha { ch } else { img.color_channels() };
    let rx = (kx.len() / 2) as i64;
    let ry = (ky.len() / 2) as i64;
    let mut mid = img.data.clone();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = 0.0f32;
                for (k, &kv) in kx.iter().enumerate() {
                    let sx = (x as i64 + k as i64 - rx).clamp(0, w as i64 - 1) as usize;
                    acc += kv * img.data[(y * w + sx) * ch + c];
                }
                mid[(y * w + x) * ch + c] = acc;
            }
        }
    }
    let mut out = mid.clone();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = 0.0f32;
                for (k, &kv) in ky.iter().enumerate() {
                    let sy = (y as i64 + k as i64 - ry).clamp(0, h as i64 - 1) as usize;
                    acc += kv * mid[(sy * w + x) * ch + c];
                }
                out[(y * w + x) * ch + c] = acc;
            }
        }
    }
    FloatImage {
        width: img.width,
        height: img.height,
        channels: ch,
        data: out,
    }
}

/// Full 2-D convolution with a `size × size` row-major kernel and
/// replicate-edge addressing on the colour channels (alpha untouched).
pub fn convolve_2d(img: &FloatImage, kernel: &[f32], size: usize) -> FloatImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let ch = img.channels;
    let nc = img.color_channels();
    let r = (size / 2) as i64;
    let mut out = img.data.clone();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = 0.0f32;
                for ky in 0..size {
                    let sy = (y as i64 + ky as i64 - r).clamp(0, h as i64 - 1) as usize;
                    for kx in 0..size {
                        let sx = (x as i64 + kx as i64 - r).clamp(0, w as i64 - 1) as usize;
                        acc += kernel[ky * size + kx] * img.data[(sy * w + sx) * ch + c];
                    }
                }
                out[(y * w + x) * ch + c] = acc;
            }
        }
    }
    FloatImage {
        width: img.width,
        height: img.height,
        channels: ch,
        data: out,
    }
}

/// Per-pixel neighbourhood reduction: for every pixel and colour
/// channel, `reduce` sees the `(2·rx+1) × (2·ry+1)` window of samples
/// (replicate-edge, row-major) and returns the output sample. Alpha is
/// untouched.
pub fn neighbourhood(
    img: &FloatImage,
    rx: usize,
    ry: usize,
    mut reduce: impl FnMut(&[f32]) -> f32,
) -> FloatImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let ch = img.channels;
    let nc = img.color_channels();
    let mut out = img.data.clone();
    let mut window = Vec::with_capacity((2 * rx + 1) * (2 * ry + 1));
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                window.clear();
                for dy in -(ry as i64)..=ry as i64 {
                    let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    for dx in -(rx as i64)..=rx as i64 {
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        window.push(img.data[(sy * w + sx) * ch + c]);
                    }
                }
                out[(y * w + x) * ch + c] = reduce(&window);
            }
        }
    }
    FloatImage {
        width: img.width,
        height: img.height,
        channels: ch,
        data: out,
    }
}

/// Bins of the float histograms the statistics-driven tonal filters
/// build (`65 536`: one per 16-bit code of the clamped tone).
pub const HIST_BINS: usize = 65536;

/// Histogram of `values` clamped to `[0, 1]` over [`HIST_BINS`] bins.
pub fn histogram(values: impl Iterator<Item = f32>) -> Vec<u32> {
    let mut hist = vec![0u32; HIST_BINS];
    let full = (HIST_BINS - 1) as f32;
    for v in values {
        let bin = if v.is_nan() {
            0
        } else {
            (v.clamp(0.0, 1.0) * full).round() as usize
        };
        hist[bin.min(HIST_BINS - 1)] += 1;
    }
    hist
}

/// The `[lo, hi]` of a tone population after discarding the darkest
/// `low_clip` and brightest `high_clip` fractions (same budget rule as
/// the 8-bit histogram scans: the first bin whose running count exceeds
/// the budget), as normalised endpoints at least one bin apart.
pub fn clip_range(values: impl Iterator<Item = f32>, low_clip: f32, high_clip: f32) -> (f32, f32) {
    let hist = histogram(values);
    let total: u32 = hist.iter().sum();
    if total == 0 {
        return (0.0, 1.0);
    }
    let lo_budget = (total as f32 * low_clip).round() as u32;
    let hi_budget = (total as f32 * high_clip).round() as u32;
    let mut acc = 0u32;
    let mut lo = 0usize;
    for (i, h) in hist.iter().enumerate() {
        acc += *h;
        if acc > lo_budget {
            lo = i;
            break;
        }
    }
    acc = 0;
    let mut hi = HIST_BINS - 1;
    for (i, h) in hist.iter().enumerate().rev() {
        acc += *h;
        if acc > hi_budget {
            hi = i;
            break;
        }
    }
    if hi <= lo {
        if lo < HIST_BINS - 1 {
            hi = lo + 1;
        } else {
            lo = HIST_BINS - 2;
            hi = HIST_BINS - 1;
        }
    }
    let full = (HIST_BINS - 1) as f32;
    (lo as f32 / full, hi as f32 / full)
}

/// Channel `c` of every pixel.
pub fn channel_iter(img: &FloatImage, c: usize) -> impl Iterator<Item = f32> + '_ {
    img.data.chunks_exact(img.channels).map(move |px| px[c])
}

/// Widen an 8-bit packed picture (`Gray8` / `Rgb24` / `Rgba`) to its
/// float counterpart (`v / 255`). Test and bridging helper.
pub fn widen_u8(frame: &VideoFrame, params: VideoStreamParams) -> Result<FloatImage, Error> {
    let channels = match params.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        other => {
            return Err(Error::unsupported(format!(
                "oxideav-image-filter: widen_u8 takes Gray8 / Rgb24 / Rgba, not {other:?}"
            )));
        }
    };
    layout::validate_frame(frame, params, "widen_u8")?;
    let w = params.width as usize;
    let h = params.height as usize;
    let plane = &frame.planes[0];
    let mut data = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * channels];
        data.extend(row.iter().map(|&b| b as f32 / 255.0));
    }
    Ok(FloatImage {
        width: params.width,
        height: params.height,
        channels,
        data,
    })
}

/// Quantise a float picture to its 8-bit packed counterpart
/// (`round(clamp(v, 0, 1) · 255)`). Test and bridging helper.
pub fn narrow_u8(img: &FloatImage) -> VideoFrame {
    let stride = img.width as usize * img.channels;
    let data = img
        .data
        .iter()
        .map(|&v| {
            if v.is_nan() {
                0
            } else {
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            }
        })
        .collect();
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_is_exact() {
        let mut img = FloatImage::new(3, 2, 4);
        for (i, v) in img.samples_mut().iter_mut().enumerate() {
            *v = i as f32 * 0.37 - 1.5;
        }
        let frame = img.clone().into_frame(Some(4));
        assert_eq!(frame.planes[0].stride, 3 * 16);
        let back = FloatImage::from_frame(&frame, img.params()).unwrap();
        assert_eq!(back, img);
        assert_eq!(img.format(), PixelFormat::RgbaF32Le);
        assert_eq!(FloatImage::new(1, 1, 1).format(), PixelFormat::GrayF32Le);
        assert_eq!(FloatImage::new(1, 1, 3).format(), PixelFormat::RgbF32Le);
    }

    #[test]
    fn map_tone_leaves_alpha_alone() {
        let mut img = FloatImage::new(2, 1, 4);
        img.samples_mut()
            .copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let out = img.map_tone(|v| v * 2.0);
        assert_eq!(out.samples(), &[0.2, 0.4, 0.6, 0.4, 1.0, 1.2, 1.4, 0.8]);
    }

    #[test]
    fn widen_narrow_round_trip_every_code() {
        let data: Vec<u8> = (0..=255).collect();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 256,
                data: data.clone(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 256,
            height: 1,
        };
        let img = widen_u8(&frame, params).unwrap();
        assert_eq!(narrow_u8(&img).planes[0].data, data);
    }

    #[test]
    fn separable_convolution_of_flat_is_flat_and_edges_replicate() {
        let mut img = FloatImage::new(4, 3, 3);
        img.samples_mut().fill(0.4);
        let k = [0.25, 0.5, 0.25];
        let out = convolve_separable(&img, &k, &k, false);
        assert!(out.samples().iter().all(|&v| (v - 0.4).abs() < 1e-6));
        let out2 = convolve_2d(&img, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(out2, img);
    }

    #[test]
    fn neighbourhood_sees_the_window() {
        let mut img = FloatImage::new(3, 3, 1);
        for (i, v) in img.samples_mut().iter_mut().enumerate() {
            *v = i as f32;
        }
        let out = neighbourhood(&img, 1, 1, |w| w.iter().cloned().fold(f32::MIN, f32::max));
        assert_eq!(out.pixel(0, 0)[0], 4.0);
        assert_eq!(out.pixel(2, 2)[0], 8.0);
    }

    #[test]
    fn rgba_widening_and_narrowing() {
        let mut g = FloatImage::new(1, 1, 1);
        g.samples_mut()[0] = 0.25;
        let rgba = g.to_rgba();
        assert_eq!(rgba.samples(), &[0.25, 0.25, 0.25, 1.0]);
        assert_eq!(rgba.to_channels(1).samples(), &[0.25]);
        assert_eq!(rgba.to_channels(3).samples(), &[0.25, 0.25, 0.25]);
    }
}
