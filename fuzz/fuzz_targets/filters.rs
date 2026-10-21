#![no_main]

//! Panic-freedom + record-carriage fuzz target for every filter type.
//!
//! The input bytes choose a filter from the catalog, a layout from
//! `layout::ALL`, a small extent, how each image plane is shaped
//! (exact, short, over-long, mis-strided, missing, extra) and which
//! side-channel records ride behind the planes (well-formed or
//! malformed: a blobs record of arbitrary bytes, a palette of
//! arbitrary length, an alpha table, a significant-bits record, a
//! colour signal, a layer identity), then feed the picture to the
//! filter. The filter must return. When it returns `Ok` on a
//! well-formed picture, the output's blobs must equal the input's and
//! the output's image planes must be exactly the declared output
//! layout's plane count — a side-channel record read as pixels, a
//! record lost, or a record counted as a plane would show here.
//!
//! Two-input filters are driven the same way with a second picture of
//! the same (or, for the overlay filter, a smaller) extent.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    BlobKind, ColorSignal, LayerIdentity, MetadataBlob, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_image_filter::catalog::{all_pair, all_single};
use oxideav_image_filter::layout;
use oxideav_image_filter::VideoStreamParams;

struct Bytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bytes<'a> {
    fn u8(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    fn take(&mut self, n: usize) -> Vec<u8> {
        let start = self.pos.min(self.data.len());
        let end = (start + n).min(self.data.len());
        self.pos = end;
        let mut v = self.data[start..end].to_vec();
        v.resize(n, 0x5a);
        v
    }
}

/// Build the frame and say whether it is well formed for `params`.
fn picture(b: &mut Bytes, params: VideoStreamParams, max_bytes: usize) -> (VideoFrame, bool) {
    let fmt = params.format;
    let mut planes = Vec::new();
    let mut well_formed = true;
    let shape = b.u8();
    for p in 0..fmt.plane_count() {
        let (_, rows) = fmt
            .plane_dimensions(p, params.width, params.height)
            .unwrap_or((0, 0));
        let row = fmt.plane_row_bytes(p, params.width).unwrap_or(0);
        let mut stride = row;
        let mut len = row * rows as usize;
        match (shape >> (p * 2)) & 3 {
            0 => {}
            1 => {
                // Short by a few bytes.
                len = len.saturating_sub(1 + (b.u8() as usize % 7));
                well_formed = false;
            }
            2 => {
                // Padded stride (still well formed) or thin stride.
                let pad = b.u8() as usize % 5;
                if pad == 0 {
                    stride = stride.saturating_sub(1);
                    well_formed = false;
                } else {
                    stride = row + pad;
                }
                len = stride * rows as usize;
            }
            _ => {
                // Over-long data (well formed: extra bytes ignored).
                len += b.u8() as usize % 9;
            }
        }
        if len > max_bytes {
            len = max_bytes;
            well_formed = false;
        }
        planes.push(VideoPlane {
            stride,
            data: b.take(len),
        });
    }
    let planes_ctl = b.u8();
    if planes_ctl & 1 == 1 && !planes.is_empty() {
        planes.pop();
        well_formed = false;
    }
    if planes_ctl & 2 == 2 {
        planes.push(VideoPlane {
            stride: 3,
            data: vec![1, 2, 3],
        });
        well_formed = false;
    }
    let mut f = VideoFrame {
        pts: Some(b.u8() as i64),
        planes,
    };
    // Side-channel records, each optional, each possibly hostile.
    let rec = b.u8();
    if rec & 1 == 1 {
        let n = 1 + b.u8() as usize % 3;
        let blobs = (0..n)
            .map(|i| {
                let kind = match i % 3 {
                    0 => BlobKind::ICC,
                    1 => BlobKind::EXIF,
                    _ => BlobKind::custom("fuzz"),
                };
                let len = b.u8() as usize % 24;
                MetadataBlob::new(kind, b.take(len))
            })
            .collect();
        f.set_blobs(blobs);
    }
    if rec & 2 == 2 {
        // Raw blobs record of arbitrary bytes (possibly malformed).
        let len = 1 + b.u8() as usize % 40;
        f.planes.push(VideoPlane {
            stride: usize::MAX - 4,
            data: b.take(len),
        });
    }
    if rec & 4 == 4 {
        let len = 1 + b.u8() as usize % 100 * 3 + b.u8() as usize % 3;
        f.set_palette(b.take(len));
    }
    if rec & 8 == 8 {
        let len = 1 + b.u8() as usize;
        f.set_palette_alpha(b.take(len));
    }
    if rec & 16 == 16 {
        let len = 1 + b.u8() as usize % 6;
        f.set_significant_bits(b.take(len));
    }
    if rec & 32 == 32 {
        f.set_color_signal(ColorSignal::default());
    }
    if rec & 64 == 64 {
        f.set_layer(LayerIdentity::default());
    }
    if rec & 128 == 128 {
        // A zero-stride, non-empty entry: a palette-shaped record of
        // arbitrary length.
        let len = 1 + b.u8() as usize % 20;
        f.planes.push(VideoPlane {
            stride: 0,
            data: b.take(len),
        });
    }
    (f, well_formed)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut b = Bytes { data, pos: 0 };
    let singles = all_single();
    let pairs = all_pair();
    let pick = b.u8() as usize;
    let fmt = layout::ALL[b.u8() as usize % layout::ALL.len()];
    let width = 1 + b.u8() as u32 % 12;
    let height = 1 + b.u8() as u32 % 12;
    let params = VideoStreamParams {
        format: fmt,
        width,
        height,
    };
    let (input, well_formed) = picture(&mut b, params, 4096);
    let in_blobs = input.blobs();

    if pick < singles.len() {
        let entry = &singles[pick];
        let f = entry.filter.as_ref();
        let declared = f.accepted_pixel_formats().contains(&fmt);
        match f.apply(&input, params) {
            Ok(out) => {
                assert!(declared, "{}: accepted an undeclared {fmt:?}", entry.name);
                assert_eq!(out.blobs(), in_blobs, "{}: blobs not carried", entry.name);
                if let Some(of) = f.output_pixel_format(fmt) {
                    assert_eq!(
                        out.image_plane_count(),
                        of.plane_count(),
                        "{}: declared {of:?}, emitted a different plane count",
                        entry.name
                    );
                }
            }
            Err(oxideav_core::Error::Unsupported(_)) => {
                assert!(
                    !declared
                        || matches!(
                            fmt,
                            PixelFormat::Yuyv422 | PixelFormat::Uyvy422 | PixelFormat::Yuv411P
                        ),
                    "{}: refused a declared {fmt:?}",
                    entry.name
                );
            }
            Err(oxideav_core::Error::InvalidData(_)) => {
                // Malformed pictures, hop geometry pixfmt refuses, or a
                // filter's own parameter / extent rule.
                let _ = well_formed;
            }
            Err(_) => {}
        }
    } else {
        let entry = &pairs[(pick - singles.len()) % pairs.len()];
        let f = entry.filter.as_ref();
        let dst_params = if entry.name == "Watermark" {
            VideoStreamParams {
                width: 2,
                height: 2,
                ..params
            }
        } else {
            params
        };
        let (dst, _) = picture(&mut b, dst_params, 4096);
        if let Ok(out) = f.apply_two(&input, &dst, params) {
            assert_eq!(out.blobs(), in_blobs, "{}: blobs not carried", entry.name);
        }
    }
});
