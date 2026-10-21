//! Pixel-layout classes and per-plane geometry shared by every filter.
//!
//! A filter declares the layouts it takes natively through
//! [`PixelFilter::accepted_formats`](crate::PixelFilter::accepted_formats)
//! by returning one of the named sets below (or its own slice). The
//! geometry helpers derive every plane's extent and bytes-per-position
//! from [`PixelFormat`] itself, so a filter that moves whole sample
//! positions around (flips, crops, rolls, resizes) is correct for any
//! byte-addressable layout without knowing its chroma siting or depth.

use oxideav_core::{Error, PixelFormat, VideoFrame};

/// The three packed 8-bit layouts the first filters were written for:
/// `Gray8`, `Rgb24`, `Rgba`.
pub const PACKED_8BIT: &[PixelFormat] =
    &[PixelFormat::Gray8, PixelFormat::Rgb24, PixelFormat::Rgba];

/// Packed 8-bit colour only: `Rgb24`, `Rgba`.
pub const PACKED_RGB: &[PixelFormat] = &[PixelFormat::Rgb24, PixelFormat::Rgba];

/// `Rgba` only (filters whose semantics need a straight alpha channel).
pub const RGBA_ONLY: &[PixelFormat] = &[PixelFormat::Rgba];

/// `Gray8` only (binary-image and distance-field operators).
pub const GRAY8: &[PixelFormat] = &[PixelFormat::Gray8];

/// `Gray8` and `Rgba` (mask-shaped operators that also take a straight
/// alpha channel).
pub const GRAY8_RGBA: &[PixelFormat] = &[PixelFormat::Gray8, PixelFormat::Rgba];

/// The 8-bit planar YUV layouts whose planes are one byte per sample:
/// the limited-range `Yuv420P` / `Yuv422P` / `Yuv444P` trio and the
/// full-range `YuvJ*` labels (identical storage; the range lives in the
/// label).
pub const YUV8_CLASSIC: &[PixelFormat] = &[
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
];

/// The original six-layout set (`Gray8`, `Rgb24`, `Rgba`, `Yuv420P`,
/// `Yuv422P`, `Yuv444P`) exactly as the first filters gated it.
pub const LEGACY: &[PixelFormat] = &[
    PixelFormat::Gray8,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
];

/// The original six-layout set (`Gray8`, `Rgb24`, `Rgba`, `Yuv420P`,
/// `Yuv422P`, `Yuv444P`) the per-plane filters were written for, plus
/// the full-range `YuvJ*` trio whose storage is identical.
pub const CLASSIC: &[PixelFormat] = &[
    PixelFormat::Gray8,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
];

/// Every layout whose planes are addressable one sample position at a
/// time with a fixed byte cost per position: everything in
/// [`PixelFormat`] except the bit-packed `MonoBlack` / `MonoWhite` and
/// the macropixel-interleaved `Yuyv422` / `Uyvy422`. Geometry-only
/// filters (flip, flop, crop, roll, shave, rotate, resize) take all of
/// these natively.
pub const BYTE_ADDRESSABLE: &[PixelFormat] = &[
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Gray8,
    PixelFormat::Pal8,
    PixelFormat::Bgr24,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv422P10Le,
    PixelFormat::Yuv444P10Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv444P12Le,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
    PixelFormat::Nv12,
    PixelFormat::Nv21,
    PixelFormat::Ya8,
    PixelFormat::Yuva420P,
    PixelFormat::Cmyk,
    PixelFormat::Yuv411P,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap14Le,
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuva422P,
    PixelFormat::Yuva444P,
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P16Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
    PixelFormat::Yuva420P10Le,
    PixelFormat::Yuva420P12Le,
    PixelFormat::Yuva420P16Le,
    PixelFormat::Gbrap8,
    PixelFormat::Ya16Le,
    PixelFormat::CmykInverted,
    PixelFormat::Yuv440P,
    PixelFormat::Yuv440P10Le,
    PixelFormat::Yuv440P12Le,
    PixelFormat::Yuv440P16Le,
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
    PixelFormat::GbrpF32Le,
    PixelFormat::GbrapF32Le,
];

/// Every layout the crate knows how to describe, in discriminant order
/// — the probe set for the filter × layout matrix tests and the fuzz
/// target. Includes the layouts no filter takes (bit-packed mono,
/// macropixel 4:2:2) so refusals are exercised too.
pub const ALL: &[PixelFormat] = &[
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Gray8,
    PixelFormat::Pal8,
    PixelFormat::Bgr24,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv422P10Le,
    PixelFormat::Yuv444P10Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv444P12Le,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
    PixelFormat::Nv12,
    PixelFormat::Nv21,
    PixelFormat::Ya8,
    PixelFormat::Yuva420P,
    PixelFormat::MonoBlack,
    PixelFormat::MonoWhite,
    PixelFormat::Yuyv422,
    PixelFormat::Uyvy422,
    PixelFormat::Cmyk,
    PixelFormat::Yuv411P,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap14Le,
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuva422P,
    PixelFormat::Yuva444P,
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P16Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
    PixelFormat::Yuva420P10Le,
    PixelFormat::Yuva420P12Le,
    PixelFormat::Yuva420P16Le,
    PixelFormat::Gbrap8,
    PixelFormat::Ya16Le,
    PixelFormat::CmykInverted,
    PixelFormat::Yuv440P,
    PixelFormat::Yuv440P10Le,
    PixelFormat::Yuv440P12Le,
    PixelFormat::Yuv440P16Le,
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
    PixelFormat::GbrpF32Le,
    PixelFormat::GbrapF32Le,
];

/// `true` when every sample position of every plane of `fmt` occupies a
/// fixed whole number of bytes (see [`BYTE_ADDRESSABLE`]).
pub fn is_byte_addressable(fmt: PixelFormat) -> bool {
    !matches!(
        fmt,
        PixelFormat::MonoBlack
            | PixelFormat::MonoWhite
            | PixelFormat::Yuyv422
            | PixelFormat::Uyvy422
    )
}

/// `true` for the planar and semi-planar YUV(A) carriages (limited or
/// full range, any depth).
pub fn is_yuv(fmt: PixelFormat) -> bool {
    fmt.chroma_subsampling().is_some()
}

/// Sample storage: how one sample of `fmt` is stored and how far its
/// codes reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleKind {
    /// One byte per sample, full scale 255.
    U8,
    /// Little-endian 16-bit word per sample; `bits` of it significant
    /// (full scale `2^bits − 1`).
    U16 { bits: u8 },
    /// Little-endian IEEE 754 binary32 per sample; no full scale.
    F32,
}

impl SampleKind {
    /// Bytes one sample occupies.
    pub fn bytes(self) -> usize {
        match self {
            SampleKind::U8 => 1,
            SampleKind::U16 { .. } => 2,
            SampleKind::F32 => 4,
        }
    }

    /// Largest code the storage carries (`None` for float).
    pub fn max_code(self) -> Option<u32> {
        match self {
            SampleKind::U8 => Some(255),
            SampleKind::U16 { bits } => Some((1u32 << bits) - 1),
            SampleKind::F32 => None,
        }
    }
}

/// The sample storage of `fmt` (`None` for the bit-packed and
/// macropixel layouts, whose "sample" is not individually addressable).
pub fn sample_kind(fmt: PixelFormat) -> Option<SampleKind> {
    use PixelFormat as P;
    Some(match fmt {
        P::MonoBlack | P::MonoWhite | P::Yuyv422 | P::Uyvy422 => return None,
        P::GrayF32Le | P::RgbF32Le | P::RgbaF32Le | P::GbrpF32Le | P::GbrapF32Le => SampleKind::F32,
        P::Gray10Le
        | P::Yuv420P10Le
        | P::Yuv422P10Le
        | P::Yuv444P10Le
        | P::Yuv440P10Le
        | P::Yuva420P10Le
        | P::Yuva422P10Le
        | P::Yuva444P10Le
        | P::Gbrp10Le
        | P::Gbrap10Le => SampleKind::U16 { bits: 10 },
        P::Gray12Le
        | P::Yuv420P12Le
        | P::Yuv422P12Le
        | P::Yuv444P12Le
        | P::Yuv440P12Le
        | P::Yuva420P12Le
        | P::Yuva422P12Le
        | P::Yuva444P12Le
        | P::Gbrp12Le
        | P::Gbrap12Le => SampleKind::U16 { bits: 12 },
        P::Gbrp14Le | P::Gbrap14Le => SampleKind::U16 { bits: 14 },
        P::Gray16Le
        | P::Rgb48Le
        | P::Rgba64Le
        | P::Ya16Le
        | P::Yuv420P16Le
        | P::Yuv422P16Le
        | P::Yuv444P16Le
        | P::Yuv440P16Le
        | P::Yuva420P16Le
        | P::Yuva422P16Le
        | P::Yuva444P16Le
        | P::Gbrp16Le
        | P::Gbrap16Le => SampleKind::U16 { bits: 16 },
        _ => SampleKind::U8,
    })
}

/// Geometry of one image plane of a picture under a pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlaneGeometry {
    /// Sample positions per row (a position holds every component
    /// stored interleaved at that site: 1 for a planar sample, 3 for an
    /// `Rgb24` pixel, 2 for an `Nv12` chroma pair).
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Bytes one sample position occupies.
    pub position_bytes: usize,
    /// Bytes of one row's visible content (`width × position_bytes`).
    pub row_bytes: usize,
}

impl PlaneGeometry {
    /// Build a geometry record by hand.
    pub fn new(width: u32, height: u32, position_bytes: usize) -> Self {
        Self {
            width,
            height,
            position_bytes,
            row_bytes: width as usize * position_bytes,
        }
    }
}

/// Geometry of plane `plane` of a `width × height` picture in `fmt`
/// (`None` when `fmt` has no such plane or is not byte-addressable).
pub fn plane_geometry(
    fmt: PixelFormat,
    plane: usize,
    width: u32,
    height: u32,
) -> Option<PlaneGeometry> {
    if !is_byte_addressable(fmt) {
        return None;
    }
    let (w, h) = fmt.plane_dimensions(plane, width, height)?;
    let position_bytes = fmt.plane_row_bytes(plane, 1)?;
    Some(PlaneGeometry::new(w, h, position_bytes))
}

/// Geometry of every image plane of a `width × height` picture in
/// `fmt` (empty when `fmt` is not byte-addressable).
pub fn plane_geometries(fmt: PixelFormat, width: u32, height: u32) -> Vec<PlaneGeometry> {
    (0..fmt.plane_count())
        .filter_map(|p| plane_geometry(fmt, p, width, height))
        .collect()
}

/// Check that `frame` is a well-formed `params`-shaped picture for the
/// filter called `name` (used in the error text only): a
/// non-empty extent, exactly the format's image planes (side-channel
/// records excluded), and every plane holding `stride × rows` bytes
/// with a stride no shorter than its visible row. Every filter runs
/// this before touching a sample, so a short, mis-strided or
/// plane-count-mismatched frame is a typed [`Error::invalid`] rather
/// than an index panic.
pub fn validate_frame(
    frame: &VideoFrame,
    params: crate::VideoStreamParams,
    name: &str,
) -> Result<(), Error> {
    let fmt = params.format;
    if params.width == 0 || params.height == 0 {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: {name}: empty picture {}x{}",
            params.width, params.height
        )));
    }
    let expected = fmt.plane_count();
    let have = frame.image_plane_count();
    if have != expected {
        return Err(Error::invalid(format!(
            "oxideav-image-filter: {name}: {fmt:?} needs {expected} image plane(s), frame carries {have}"
        )));
    }
    for (idx, plane) in frame.image_planes().iter().enumerate() {
        let rows = fmt
            .plane_dimensions(idx, params.width, params.height)
            .map(|(_, h)| h as usize)
            .unwrap_or(0);
        let row_bytes = fmt.plane_row_bytes(idx, params.width).unwrap_or(0);
        if plane.stride < row_bytes {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: {name}: {fmt:?} plane {idx} stride {} is shorter than its {row_bytes}-byte row",
                plane.stride
            )));
        }
        let need = plane.stride.checked_mul(rows).ok_or_else(|| {
            Error::invalid(format!(
                "oxideav-image-filter: {name}: {fmt:?} plane {idx} extent overflows"
            ))
        })?;
        if plane.data.len() < need {
            return Err(Error::invalid(format!(
                "oxideav-image-filter: {name}: {fmt:?} plane {idx} holds {} bytes, {need} needed for {rows} rows of stride {}",
                plane.data.len(),
                plane.stride
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    #[test]
    fn byte_addressable_set_is_everything_but_mono_and_macropixel() {
        for &f in ALL {
            assert_eq!(
                BYTE_ADDRESSABLE.contains(&f),
                is_byte_addressable(f),
                "{f:?}"
            );
            assert_eq!(sample_kind(f).is_some(), is_byte_addressable(f), "{f:?}");
        }
        assert_eq!(ALL.len(), 70);
        assert_eq!(BYTE_ADDRESSABLE.len(), 66);
    }

    #[test]
    fn geometry_follows_core_plane_rules() {
        let g = plane_geometries(PixelFormat::Yuv420P, 5, 3);
        assert_eq!(g[0], PlaneGeometry::new(5, 3, 1));
        assert_eq!(g[1], PlaneGeometry::new(3, 2, 1));
        assert_eq!(g[2], PlaneGeometry::new(3, 2, 1));
        let g = plane_geometries(PixelFormat::Nv12, 5, 3);
        assert_eq!(g[1], PlaneGeometry::new(3, 2, 2));
        let g = plane_geometries(PixelFormat::Rgb48Le, 2, 2);
        assert_eq!(g[0], PlaneGeometry::new(2, 2, 6));
        let g = plane_geometries(PixelFormat::Yuv440P12Le, 5, 3);
        assert_eq!(g[1], PlaneGeometry::new(5, 2, 2));
        assert!(plane_geometries(PixelFormat::MonoBlack, 8, 8).is_empty());
        assert_eq!(
            sample_kind(PixelFormat::Gbrp14Le),
            Some(SampleKind::U16 { bits: 14 })
        );
        assert_eq!(SampleKind::U16 { bits: 10 }.max_code(), Some(1023));
        assert_eq!(SampleKind::F32.max_code(), None);
    }

    #[test]
    fn validation_rejects_malformed_frames_with_typed_errors() {
        let params = crate::VideoStreamParams {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 2,
        };
        let good = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: 4,
                    data: vec![0; 8],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 2],
                },
                VideoPlane {
                    stride: 2,
                    data: vec![0; 2],
                },
            ],
        };
        assert!(validate_frame(&good, params, "t").is_ok());

        let mut short = good.clone();
        short.planes[2].data.pop();
        assert!(matches!(
            validate_frame(&short, params, "t"),
            Err(Error::InvalidData(_))
        ));

        let mut missing = good.clone();
        missing.planes.pop();
        assert!(matches!(
            validate_frame(&missing, params, "t"),
            Err(Error::InvalidData(_))
        ));

        let mut thin = good.clone();
        thin.planes[0].stride = 3;
        assert!(matches!(
            validate_frame(&thin, params, "t"),
            Err(Error::InvalidData(_))
        ));

        let empty = crate::VideoStreamParams { width: 0, ..params };
        let err = validate_frame(&good, empty, "Probe").unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
        assert!(err.to_string().contains("Probe"));

        // Side-channel records do not count as image planes.
        let tagged = good.clone().with_blob(oxideav_core::MetadataBlob::new(
            oxideav_core::BlobKind::XMP,
            b"x".to_vec(),
        ));
        assert_eq!(tagged.planes.len(), 4);
        assert!(validate_frame(&tagged, params, "t").is_ok());
    }
}
