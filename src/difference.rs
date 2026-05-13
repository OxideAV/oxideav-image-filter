//! Pixel-wise absolute-difference of two inputs — `out = |src - dst|`.
//!
//! Two-input filter useful for change detection / motion masks /
//! before-vs-after visualisation. Identical inputs produce a fully
//! black frame; the brighter / more saturated the disagreement, the
//! larger the resulting magnitude.
//!
//! For RGBA the alpha channel is also differenced (so a fully-opaque
//! pair with identical pixels lands at `alpha = 0`); callers wanting
//! "alpha pass-through" can pre-multiply or post-restore alpha
//! themselves. The [`Composite`](crate::Composite) family already
//! includes a `difference` operator that preserves `src.a`; this
//! standalone filter avoids the Porter–Duff coverage algebra and is
//! pixel-arithmetic only.
//!
//! Both inputs must share `(format, width, height)`. Output keeps the
//! same shape.
//!
//! # Pixel formats
//!
//! `Gray8` / `Rgb24` / `Rgba` / planar YUV (Y / U / V planes are each
//! differenced independently). Other formats return
//! [`Error::unsupported`].

use crate::{TwoInputImageFilter, VideoStreamParams};
use oxideav_core::{Error, PixelFormat, VideoFrame, VideoPlane};

/// Pixel-wise absolute-difference filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Difference;

impl Difference {
    pub fn new() -> Self {
        Self
    }
}

impl TwoInputImageFilter for Difference {
    fn apply_two(
        &self,
        src: &VideoFrame,
        dst: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        match params.format {
            PixelFormat::Gray8 | PixelFormat::Rgb24 | PixelFormat::Rgba => {
                diff_packed(src, dst, params)
            }
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {
                diff_planar_yuv(src, dst, params)
            }
            other => Err(Error::unsupported(format!(
                "oxideav-image-filter: Difference does not yet handle {other:?}"
            ))),
        }
    }
}

fn diff_packed(
    src: &VideoFrame,
    dst: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    let bpp = match params.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Rgba => 4,
        _ => unreachable!(),
    };
    if src.planes.is_empty() || dst.planes.is_empty() {
        return Err(Error::invalid(
            "oxideav-image-filter: Difference requires non-empty src/dst planes",
        ));
    }
    let w = params.width as usize;
    let h = params.height as usize;
    let row = w * bpp;
    let sp = &src.planes[0];
    let dp = &dst.planes[0];
    let mut out = vec![0u8; row * h];
    for y in 0..h {
        let s = &sp.data[y * sp.stride..y * sp.stride + row];
        let d = &dp.data[y * dp.stride..y * dp.stride + row];
        let o = &mut out[y * row..(y + 1) * row];
        for i in 0..row {
            o[i] = s[i].abs_diff(d[i]);
        }
    }
    Ok(VideoFrame {
        pts: src.pts.or(dst.pts),
        planes: vec![VideoPlane {
            stride: row,
            data: out,
        }],
    })
}

fn diff_planar_yuv(
    src: &VideoFrame,
    dst: &VideoFrame,
    params: VideoStreamParams,
) -> Result<VideoFrame, Error> {
    if src.planes.len() < 3 || dst.planes.len() < 3 {
        return Err(Error::invalid(
            "oxideav-image-filter: Difference requires 3 YUV planes on src/dst",
        ));
    }
    let (xsub, ysub) = match params.format {
        PixelFormat::Yuv420P => (2, 2),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv444P => (1, 1),
        _ => unreachable!(),
    };
    let w = params.width as usize;
    let h = params.height as usize;
    let cw = w.div_ceil(xsub);
    let ch = h.div_ceil(ysub);
    let mut planes = Vec::with_capacity(3);
    for i in 0..3 {
        let (pw, ph) = if i == 0 { (w, h) } else { (cw, ch) };
        let sp = &src.planes[i];
        let dp = &dst.planes[i];
        let stride = pw;
        let mut data = vec![0u8; stride * ph];
        for y in 0..ph {
            let s = &sp.data[y * sp.stride..y * sp.stride + pw];
            let d = &dp.data[y * dp.stride..y * dp.stride + pw];
            let o = &mut data[y * stride..(y + 1) * stride];
            for x in 0..pw {
                o[x] = s[x].abs_diff(d[x]);
            }
        }
        planes.push(VideoPlane { stride, data });
    }
    Ok(VideoFrame {
        pts: src.pts.or(dst.pts),
        planes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: u32, h: u32, fill: [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&fill);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }

    fn p_rgb(w: u32, h: u32) -> VideoStreamParams {
        VideoStreamParams {
            format: PixelFormat::Rgb24,
            width: w,
            height: h,
        }
    }

    #[test]
    fn identical_inputs_yield_black() {
        let f = rgb(4, 4, [100, 50, 200]);
        let out = Difference::new().apply_two(&f, &f, p_rgb(4, 4)).unwrap();
        for &v in &out.planes[0].data {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn absolute_difference_is_signless() {
        let a = rgb(2, 2, [100, 100, 100]);
        let b = rgb(2, 2, [40, 200, 100]);
        let out_ab = Difference::new().apply_two(&a, &b, p_rgb(2, 2)).unwrap();
        let out_ba = Difference::new().apply_two(&b, &a, p_rgb(2, 2)).unwrap();
        // |a - b| == |b - a|.
        assert_eq!(out_ab.planes[0].data, out_ba.planes[0].data);
        // First pixel: |100-40| = 60, |100-200| = 100, |100-100| = 0.
        let p = &out_ab.planes[0].data[0..3];
        assert_eq!(p, &[60, 100, 0]);
    }

    #[test]
    fn gray8_diff() {
        let mk = |fill: u8| VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![fill; 16],
            }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 4,
            height: 4,
        };
        let out = Difference::new().apply_two(&mk(200), &mk(50), p).unwrap();
        assert!(out.planes[0].data.iter().all(|&v| v == 150));
    }

    #[test]
    fn rejects_unsupported() {
        let f = rgb(2, 2, [0, 0, 0]);
        let p = VideoStreamParams {
            format: PixelFormat::Bgr24,
            width: 2,
            height: 2,
        };
        let err = Difference::new().apply_two(&f, &f, p).unwrap_err();
        assert!(format!("{err}").contains("Difference"));
    }
}
