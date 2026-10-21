//! Pixel-exact checks for the geometry filters (flip, flop, crop, roll,
//! rotate by a quarter turn) on layouts beyond the classic 8-bit six:
//! 16-bit words must move as words, palette indices as indices with the
//! table riding along, semi-planar chroma pairs as pairs, full-resolution
//! alpha planes at picture resolution, and a quarter turn of a 4:2:2
//! picture must land its chroma on the rotated picture's own grid.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::{Crop, Flip, Flop, ImageFilter, Roll, Rotate, VideoStreamParams};

fn params(fmt: PixelFormat, w: u32, h: u32) -> VideoStreamParams {
    VideoStreamParams {
        format: fmt,
        width: w,
        height: h,
    }
}

/// A `w × h` picture whose every plane holds a position-dependent
/// pattern: byte `i` of plane `p` is `(p * 50 + i * 7) % 256`.
fn picture(fmt: PixelFormat, w: u32, h: u32) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            VideoPlane {
                stride,
                data: (0..stride * rows as usize)
                    .map(|i| ((p * 50 + i * 7) % 256) as u8)
                    .collect(),
            }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

fn u16_at(plane: &VideoPlane, x: usize, y: usize) -> u16 {
    let o = y * plane.stride + 2 * x;
    u16::from_le_bytes([plane.data[o], plane.data[o + 1]])
}

#[test]
fn flip_and_flop_move_16_bit_words_whole() {
    let (w, h) = (5u32, 4u32);
    let input = picture(PixelFormat::Gray16Le, w, h);
    let flipped = Flip::new()
        .apply(&input, params(PixelFormat::Gray16Le, w, h))
        .unwrap();
    let flopped = Flop::new()
        .apply(&input, params(PixelFormat::Gray16Le, w, h))
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let v = u16_at(&input.planes[0], x, y);
            assert_eq!(u16_at(&flipped.planes[0], x, h as usize - 1 - y), v);
            assert_eq!(u16_at(&flopped.planes[0], w as usize - 1 - x, y), v);
        }
    }
    // Rgb48Le: a 6-byte pixel moves intact.
    let input = picture(PixelFormat::Rgb48Le, 3, 2);
    let flopped = Flop::new()
        .apply(&input, params(PixelFormat::Rgb48Le, 3, 2))
        .unwrap();
    assert_eq!(&flopped.planes[0].data[0..6], &input.planes[0].data[12..18]);
    assert_eq!(&flopped.planes[0].data[12..18], &input.planes[0].data[0..6]);
}

#[test]
fn crop_on_pal8_keeps_indices_and_table() {
    let mut input = picture(PixelFormat::Pal8, 6, 4);
    for v in &mut input.planes[0].data {
        *v %= 3;
    }
    let table = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    let input = input
        .with_palette(table.clone())
        .with_palette_alpha(vec![255, 128, 0]);
    let out = Crop::new(2, 1, 3, 2)
        .apply(&input, params(PixelFormat::Pal8, 6, 4))
        .unwrap();
    assert_eq!(out.image_plane_count(), 1);
    assert_eq!(out.planes[0].stride, 3);
    for y in 0..2 {
        for x in 0..3 {
            assert_eq!(
                out.planes[0].data[y * 3 + x],
                input.planes[0].data[(y + 1) * 6 + x + 2]
            );
        }
    }
    assert_eq!(out.palette(), Some(table.as_slice()));
    assert_eq!(out.palette_alpha(), Some(&[255u8, 128, 0][..]));
}

#[test]
fn crop_chroma_lands_on_the_cropped_pictures_grid() {
    // 4:2:2, odd origin: the chroma plane is ceil(w/2) wide, taken from
    // chroma column floor(x/2).
    let (w, h) = (9u32, 3u32);
    let input = picture(PixelFormat::YuvJ422P, w, h);
    let out = Crop::new(3, 1, 5, 2)
        .apply(&input, params(PixelFormat::YuvJ422P, w, h))
        .unwrap();
    assert_eq!(out.planes[0].stride, 5);
    assert_eq!(out.planes[0].data.len(), 10);
    assert_eq!(out.planes[1].stride, 3);
    assert_eq!(out.planes[1].data.len(), 6);
    for y in 0..2 {
        for cx in 0..3 {
            assert_eq!(
                out.planes[1].data[y * 3 + cx],
                input.planes[1].data[(y + 1) * input.planes[1].stride + 1 + cx]
            );
        }
    }
    // 4:2:0 at 16 bits: same rule, two bytes per sample.
    let input = picture(PixelFormat::Yuv420P10Le, 7, 5);
    let out = Crop::new(1, 1, 5, 3)
        .apply(&input, params(PixelFormat::Yuv420P10Le, 7, 5))
        .unwrap();
    assert_eq!(out.planes[0].stride, 10);
    assert_eq!(out.planes[1].stride, 6);
    assert_eq!(out.planes[1].data.len(), 6 * 2);
    assert_eq!(u16_at(&out.planes[1], 0, 0), u16_at(&input.planes[1], 0, 0));
    assert_eq!(u16_at(&out.planes[2], 2, 1), u16_at(&input.planes[2], 2, 1));
}

#[test]
fn roll_moves_semi_planar_pairs_and_full_resolution_alpha() {
    // Nv12: the chroma plane holds (Cb, Cr) pairs; a roll by (2, 0)
    // moves each pair by one position.
    let (w, h) = (6u32, 4u32);
    let input = picture(PixelFormat::Nv12, w, h);
    let out = Roll::new(2, 0)
        .apply(&input, params(PixelFormat::Nv12, w, h))
        .unwrap();
    let c = &input.planes[1];
    let oc = &out.planes[1];
    assert_eq!(oc.stride, 6);
    for y in 0..2 {
        for px in 0..3 {
            let src = (px + 2) % 3;
            assert_eq!(
                &oc.data[y * 6 + 2 * px..y * 6 + 2 * px + 2],
                &c.data[y * 6 + 2 * src..y * 6 + 2 * src + 2]
            );
        }
    }
    // Yuva420P: the alpha plane (index 3) is full resolution and rolls
    // by the full offset; chroma rolls by half.
    let input = picture(PixelFormat::Yuva420P, w, h);
    let out = Roll::new(2, 2)
        .apply(&input, params(PixelFormat::Yuva420P, w, h))
        .unwrap();
    let a = &input.planes[3];
    let oa = &out.planes[3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let (sx, sy) = (
                (x + w as usize - 2) % w as usize,
                (y + h as usize - 2) % h as usize,
            );
            assert_eq!(oa.data[y * oa.stride + x], a.data[sy * a.stride + sx]);
        }
    }
    let c = &input.planes[1];
    let oc = &out.planes[1];
    for y in 0..2 {
        for x in 0..3 {
            let (sx, sy) = ((x + 3 - 1) % 3, (y + 2 - 1) % 2);
            assert_eq!(oc.data[y * oc.stride + x], c.data[sy * c.stride + sx]);
        }
    }
}

#[test]
fn quarter_turn_of_422_lands_chroma_on_the_rotated_grid() {
    let (w, h) = (8u32, 6u32);
    let input = picture(PixelFormat::Yuv422P, w, h);
    let out = Rotate::new(90.0)
        .apply(&input, params(PixelFormat::Yuv422P, w, h))
        .unwrap();
    // Rotated picture is 6 × 8: luma 6×8, chroma ceil(6/2) × 8 = 3 × 8.
    assert_eq!(out.planes[0].stride, 6);
    assert_eq!(out.planes[0].data.len(), 48);
    assert_eq!(out.planes[1].stride, 3);
    assert_eq!(out.planes[1].data.len(), 24);
    assert_eq!(out.planes[2].stride, 3);
    // Luma is an exact clockwise turn.
    for y in 0..h as usize {
        for x in 0..w as usize {
            assert_eq!(
                out.planes[0].data[x * 6 + (h as usize - 1 - y)],
                input.planes[0].data[y * input.planes[0].stride + x]
            );
        }
    }
    // The same picture at 4:4:0 turns into a 4:2:2-shaped chroma plane of
    // the rotated picture too: 8 × 6 at 4:4:0 has chroma 8 × 3; the 6 × 8
    // result keeps its own 4:4:0 rule, chroma 6 × 4.
    let input = picture(PixelFormat::Yuv440P, w, h);
    let out = Rotate::new(270.0)
        .apply(&input, params(PixelFormat::Yuv440P, w, h))
        .unwrap();
    assert_eq!(out.planes[1].stride, 6);
    assert_eq!(out.planes[1].data.len(), 24);
}

#[test]
fn four_quarter_turns_are_the_identity_on_deep_planar_and_palette() {
    for fmt in [
        PixelFormat::Yuv444P16Le,
        PixelFormat::Yuv420P12Le,
        PixelFormat::Gbrap8,
        PixelFormat::Pal8,
        PixelFormat::Rgba64Le,
    ] {
        let (w, h) = (6u32, 4u32);
        let input = picture(fmt, w, h).with_palette(vec![1, 2, 3]);
        let mut cur = input.clone();
        let (mut cw, mut ch) = (w, h);
        for _ in 0..4 {
            cur = Rotate::new(90.0).apply(&cur, params(fmt, cw, ch)).unwrap();
            std::mem::swap(&mut cw, &mut ch);
        }
        for (a, b) in cur.image_planes().iter().zip(input.image_planes()) {
            assert_eq!(a.stride, b.stride, "{fmt:?}");
            assert_eq!(a.data, b.data, "{fmt:?}");
        }
        if fmt == PixelFormat::Pal8 {
            assert_eq!(cur.palette(), Some(&[1u8, 2, 3][..]));
        } else {
            assert!(cur.palette().is_none());
        }
    }
}

#[test]
fn arbitrary_angle_takes_classic_layouts_only() {
    let input = picture(PixelFormat::Gray16Le, 4, 4);
    let err = Rotate::new(30.0)
        .apply(&input, params(PixelFormat::Gray16Le, 4, 4))
        .unwrap_err();
    assert!(matches!(err, oxideav_core::Error::Unsupported(_)));
    assert!(err.to_string().contains("Gray16Le"));
    let input = picture(PixelFormat::YuvJ420P, 4, 4);
    assert!(Rotate::new(30.0)
        .apply(&input, params(PixelFormat::YuvJ420P, 4, 4))
        .is_ok());
}
