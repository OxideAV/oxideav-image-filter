//! The single place in the crate that knows a [`VideoFrame`] may carry
//! side-channel records behind its image planes.
//!
//! `oxideav-core` 0.1.40 lets a frame end with records that are not
//! pixels — a palette, a palette-alpha table, per-plane significant
//! bits, a colour signal, a layer identity, a metadata-blob list —
//! each tagged by a `stride` no image plane can have (`0` or above
//! `isize::MAX`). A filter that walks `frame.planes` as pixels would
//! read those records as rows and index out of bounds. Every filter in
//! this crate therefore runs behind [`image_only`], which hands it a
//! frame holding nothing but image planes, and [`carry`], which
//! re-attaches the input's records to the output under one ruling:
//!
//! | record | carried when |
//! |---|---|
//! | metadata blobs | always — the picture's provenance is unchanged by filtering |
//! | layer identity | always |
//! | colour signal | the output stays in the input's carriage (YUV → YUV or RGB/grey → RGB/grey); dropped across a matrix hop |
//! | per-plane significant bits | the output layout equals the input layout (the depth claim still describes every plane) |
//! | palette + palette alpha | the output layout is still `Pal8` (indices were moved, not recoloured) |
//!
//! A filter that recolours a palette (the pointwise palette route of
//! [`crate::PixelFilter`]) replaces the palette records itself after
//! `carry` has run.

use std::borrow::Cow;

use oxideav_core::{PixelFormat, VideoFrame};

/// `frame` with every side-channel record removed. Borrows when the
/// frame carries none (the common case — no copy); otherwise clones
/// the image planes into an owned frame.
pub fn image_only(frame: &VideoFrame) -> Cow<'_, VideoFrame> {
    let n = frame.image_plane_count();
    if n == frame.planes.len() {
        Cow::Borrowed(frame)
    } else {
        Cow::Owned(VideoFrame {
            pts: frame.pts,
            planes: frame.planes[..n].to_vec(),
        })
    }
}

/// `true` when `frame` ends with at least one side-channel record.
pub fn has_side_channels(frame: &VideoFrame) -> bool {
    frame.image_plane_count() != frame.planes.len()
}

/// Re-attach `input`'s side-channel records to `output` (a frame that
/// holds only image planes, in layout `output_format`) under the
/// module-level ruling. `input_format` is the layout `input` was in.
pub fn carry(
    input: &VideoFrame,
    output: &mut VideoFrame,
    input_format: PixelFormat,
    output_format: PixelFormat,
) {
    let n = input.image_plane_count();
    if n == input.planes.len() {
        return;
    }
    // Copy the whole trailing run byte-for-byte (unknown future record
    // kinds included, in their original order), then remove the records
    // the ruling says no longer describe the output.
    output.planes.extend(input.planes[n..].iter().cloned());
    let same_layout = input_format == output_format;
    if !same_layout {
        output.take_significant_bits();
    }
    if output_format != PixelFormat::Pal8 {
        output.take_palette();
        output.take_palette_alpha();
    }
    if crate::layout::is_yuv(input_format) != crate::layout::is_yuv(output_format) {
        output.take_color_signal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{BlobKind, ColorSignal, LayerIdentity, MetadataBlob, VideoPlane};

    fn gray_with_records() -> VideoFrame {
        VideoFrame {
            pts: Some(3),
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![1; 8],
            }],
        }
        .with_blob(MetadataBlob::new(BlobKind::ICC, b"icc".to_vec()))
        .with_significant_bits(vec![7])
        .with_color_signal(ColorSignal::default())
        .with_layer(LayerIdentity::default())
    }

    #[test]
    fn image_only_strips_and_borrows_when_clean() {
        let f = gray_with_records();
        assert!(has_side_channels(&f));
        let stripped = image_only(&f);
        assert!(matches!(stripped, Cow::Owned(_)));
        assert_eq!(stripped.planes.len(), 1);
        assert_eq!(stripped.pts, Some(3));
        let clean = image_only(&stripped);
        assert!(matches!(clean, Cow::Borrowed(_)));
        assert!(!has_side_channels(&stripped));
    }

    #[test]
    fn carry_same_layout_keeps_everything() {
        let f = gray_with_records();
        let mut out = image_only(&f).into_owned();
        carry(&f, &mut out, PixelFormat::Gray8, PixelFormat::Gray8);
        assert_eq!(out.blobs().len(), 1);
        assert_eq!(out.significant_bits(), Some(&[7u8][..]));
        assert!(out.color_signal().is_some());
        assert!(out.layer().is_some());
        // Byte-identical record run.
        assert_eq!(out.planes.len(), f.planes.len());
        for (a, b) in out.planes.iter().zip(&f.planes) {
            assert_eq!(a.stride, b.stride);
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn carry_layout_change_drops_depth_claim_and_matrix_hop_drops_signal() {
        let f = gray_with_records();
        let mut out = image_only(&f).into_owned();
        carry(&f, &mut out, PixelFormat::Gray8, PixelFormat::Rgb24);
        assert_eq!(out.blobs().len(), 1);
        assert!(out.significant_bits().is_none());
        assert!(
            out.color_signal().is_some(),
            "grey → RGB stays in RGB carriage"
        );
        assert!(out.layer().is_some());

        let mut out = image_only(&f).into_owned();
        carry(&f, &mut out, PixelFormat::Gray8, PixelFormat::Yuv444P);
        assert!(out.color_signal().is_none(), "grey → YUV crosses a matrix");
        assert_eq!(out.blobs().len(), 1);
    }

    #[test]
    fn carry_palette_only_survives_into_pal8() {
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![0, 1, 1, 0],
            }],
        }
        .with_palette(vec![0, 0, 0, 255, 255, 255])
        .with_palette_alpha(vec![255, 0])
        .with_blob(MetadataBlob::new(BlobKind::EXIF, b"e".to_vec()));
        let mut out = image_only(&f).into_owned();
        carry(&f, &mut out, PixelFormat::Pal8, PixelFormat::Pal8);
        assert_eq!(out.palette(), Some(&[0u8, 0, 0, 255, 255, 255][..]));
        assert_eq!(out.palette_alpha(), Some(&[255u8, 0][..]));
        assert_eq!(out.blobs().len(), 1);

        let mut out = image_only(&f).into_owned();
        carry(&f, &mut out, PixelFormat::Pal8, PixelFormat::Rgba);
        assert!(out.palette().is_none());
        assert!(out.palette_alpha().is_none());
        assert_eq!(out.blobs().len(), 1);
    }
}
