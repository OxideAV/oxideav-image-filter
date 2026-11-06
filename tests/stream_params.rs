//! `Frame::StreamParams` through the registry: each registered filter
//! re-plans for a mid-stream input shape change, announces its own new
//! output shape before the first frame of that shape, or refuses with
//! `Error::Unsupported` and keeps its previous plan.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, PixelFormat, PortParams, PortSpec, Result,
    RuntimeContext, StreamFilter, TimeBase, VideoFrame, VideoPlane, filter::FilterContext,
};
use oxideav_image_filter::registry::register;
use serde_json::{Value, json};

struct Collect(Vec<(usize, Frame)>);

impl FilterContext for Collect {
    fn emit(&mut self, port: usize, frame: Frame) -> Result<()> {
        self.0.push((port, frame));
        Ok(())
    }
}

fn ctx() -> RuntimeContext {
    let mut c = RuntimeContext::new();
    register(&mut c);
    c
}

fn port(name: &str, fmt: PixelFormat, w: u32, h: u32) -> PortSpec {
    PortSpec::video(name, w, h, fmt, TimeBase::new(1, 30))
}

fn make(name: &str, params: Value, inputs: &[PortSpec]) -> Box<dyn StreamFilter> {
    ctx()
        .filters
        .make(name, &params, inputs)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A decoder-style event announcing `w`x`h` in `fmt`.
fn event(fmt: PixelFormat, w: u32, h: u32) -> Frame {
    let mut p = CodecParameters::video(CodecId::new("rawvideo"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(fmt);
    Frame::StreamParams(Box::new(p))
}

fn picture(fmt: PixelFormat, w: u32, h: u32, seed: u32) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            let data = (0..stride * rows as usize)
                .map(|i| ((i as u32 * 37 + p as u32 * 11 + seed) % 251) as u8)
                .collect();
            VideoPlane { stride, data }
        })
        .collect();
    VideoFrame {
        pts: Some(seed as i64),
        planes,
    }
}

fn shape(spec: &PortSpec) -> (u32, u32, PixelFormat) {
    match spec.params {
        PortParams::Video {
            width,
            height,
            format,
            ..
        } => (width, height, format),
        _ => panic!("not a video port"),
    }
}

fn event_shape(f: &Frame) -> (u32, u32, PixelFormat) {
    let p = f.stream_params().expect("a stream-params event");
    (p.width.unwrap(), p.height.unwrap(), p.pixel_format.unwrap())
}

/// Width / height of a packed or planar picture in `fmt`, read off plane 0.
fn picture_extent(fmt: PixelFormat, v: &VideoFrame) -> (u32, u32) {
    let bytes = fmt.plane_row_bytes(0, 1).unwrap();
    let p = &v.planes[0];
    ((p.stride / bytes) as u32, (p.data.len() / p.stride) as u32)
}

fn pixels(v: &VideoFrame) -> Vec<(usize, Vec<u8>)> {
    v.planes
        .iter()
        .map(|p| (p.stride, p.data.clone()))
        .collect()
}

fn only_video(c: &Collect) -> &VideoFrame {
    assert_eq!(c.0.len(), 1, "exactly one frame expected");
    match &c.0[0] {
        (0, Frame::Video(v)) => v,
        other => panic!("expected a video frame on port 0, got {other:?}"),
    }
}

#[test]
fn scale_fixed_target_keeps_its_output_through_a_size_change() {
    let mut f = make(
        "resize",
        json!({"width": 3, "height": 2}),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    let mut c = Collect(Vec::new());
    // Size change only: the output is still 3x2 RGBA — nothing to announce.
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 8, 6)).unwrap();
    assert!(c.0.is_empty(), "no output change, no event: {:?}", c.0);
    assert_eq!(shape(&f.input_ports()[0]), (8, 6, PixelFormat::Rgba));
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 8, 6, 1)),
    )
    .unwrap();
    assert_eq!(picture_extent(PixelFormat::Rgba, only_video(&c)), (3, 2));

    // Layout change: the output follows the layout, so it is announced.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgb24, 8, 6)).unwrap();
    assert_eq!(c.0.len(), 1);
    assert_eq!(c.0[0].0, 0);
    assert_eq!(event_shape(&c.0[0].1), (3, 2, PixelFormat::Rgb24));
    assert_eq!(shape(&f.output_ports()[0]), (3, 2, PixelFormat::Rgb24));
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgb24, 8, 6, 2)),
    )
    .unwrap();
    assert!(c.0[0].1.is_stream_params(), "event before the frame");
    let Frame::Video(v) = &c.0[1].1 else {
        panic!("frame expected")
    };
    assert_eq!(v.planes[0].stride, 9);
    assert_eq!(v.planes[0].data.len(), 18);
}

#[test]
fn scale_output_matches_a_filter_planned_for_the_new_shape() {
    let params = json!({"width": 5, "height": 3, "interpolation": "lanczos"});
    let mut f = make(
        "resize",
        params.clone(),
        &[port("in", PixelFormat::Rgb24, 4, 4)],
    );
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgb24, 9, 7)).unwrap();
    let frame = Frame::Video(picture(PixelFormat::Rgb24, 9, 7, 3));
    f.push(&mut c, 0, &frame).unwrap();
    let mut fresh = make("resize", params, &[port("in", PixelFormat::Rgb24, 9, 7)]);
    let mut c2 = Collect(Vec::new());
    fresh.push(&mut c2, 0, &frame).unwrap();
    assert_eq!(pixels(only_video(&c)), pixels(only_video(&c2)));
}

#[test]
fn crop_replans_or_refuses_when_the_window_no_longer_fits() {
    let mut f = make(
        "crop",
        json!({"x": 1, "y": 1, "width": 2, "height": 2}),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 6, 6)).unwrap();
    assert!(c.0.is_empty(), "the 2x2 window is unchanged");
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 6, 6, 4)),
    )
    .unwrap();
    assert_eq!(picture_extent(PixelFormat::Rgba, only_video(&c)), (2, 2));

    // A 2x2 input no longer holds the (1,1)+2x2 window.
    let mut c = Collect(Vec::new());
    let err = f
        .push(&mut c, 0, &event(PixelFormat::Rgba, 2, 2))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("crop"), "{err}");
    assert!(c.0.is_empty());
    // The previous plan stands.
    assert_eq!(shape(&f.input_ports()[0]), (6, 6, PixelFormat::Rgba));
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 6, 6, 5)),
    )
    .unwrap();
    assert_eq!(picture_extent(PixelFormat::Rgba, only_video(&c)), (2, 2));
}

#[test]
fn colour_convert_announces_its_collapsed_layout() {
    let mut f = make(
        "grayscale",
        json!({"output_gray8": true}),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    assert_eq!(shape(&f.output_ports()[0]), (4, 4, PixelFormat::Gray8));
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgb24, 8, 2)).unwrap();
    assert_eq!(c.0.len(), 1);
    assert_eq!(event_shape(&c.0[0].1), (8, 2, PixelFormat::Gray8));
    assert_eq!(shape(&f.output_ports()[0]), (8, 2, PixelFormat::Gray8));
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgb24, 8, 2, 6)),
    )
    .unwrap();
    let Frame::Video(v) = &c.0[1].1 else {
        panic!("frame expected")
    };
    assert_eq!((v.planes[0].stride, v.planes[0].data.len()), (8, 16));

    // A layout change to one that keeps the Gray8 output at the same
    // extent announces nothing.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 8, 2)).unwrap();
    assert!(c.0.is_empty(), "{:?}", c.0);
}

#[test]
fn colour_convert_refuses_a_layout_it_does_not_take() {
    let mut f = make(
        "grayscale",
        json!({"output_gray8": true}),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    let mut c = Collect(Vec::new());
    let err = f
        .push(&mut c, 0, &event(PixelFormat::Rgb565Le, 4, 4))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("Rgb565Le"), "{err}");
    assert!(c.0.is_empty());
    assert_eq!(shape(&f.input_ports()[0]), (4, 4, PixelFormat::Rgba));
}

#[test]
fn convolution_follows_its_input_and_matches_a_fresh_plan() {
    let params = json!({"size": 3, "kernel": [1, 2, 1, 2, 4, 2, 1, 2, 1]});
    let mut f = make(
        "convolve",
        params.clone(),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 7, 5)).unwrap();
    assert_eq!(c.0.len(), 1);
    assert_eq!(event_shape(&c.0[0].1), (7, 5, PixelFormat::Rgba));
    let frame = Frame::Video(picture(PixelFormat::Rgba, 7, 5, 7));
    f.push(&mut c, 0, &frame).unwrap();
    let Frame::Video(got) = &c.0[1].1 else {
        panic!("frame expected")
    };
    let mut fresh = make("convolve", params, &[port("in", PixelFormat::Rgba, 7, 5)]);
    let mut c2 = Collect(Vec::new());
    fresh.push(&mut c2, 0, &frame).unwrap();
    assert_eq!(pixels(got), pixels(only_video(&c2)));

    // An event that repeats the current shape is forwarded unchanged by
    // a filter whose output follows its input.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 7, 5)).unwrap();
    assert_eq!(c.0.len(), 1);
    assert_eq!(event_shape(&c.0[0].1), (7, 5, PixelFormat::Rgba));
}

#[test]
fn convolution_event_keeps_the_upstream_fields() {
    let mut f = make(
        "convolve",
        json!({"kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0]}),
        &[port("in", PixelFormat::Rgba, 4, 4)],
    );
    let mut p = CodecParameters::video(CodecId::new("rawvideo"));
    p.width = Some(6);
    p.bit_rate = Some(1234);
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &Frame::StreamParams(Box::new(p)))
        .unwrap();
    let out = c.0[0].1.stream_params().unwrap();
    // Unset fields keep the port's current value; others pass through.
    assert_eq!(out.width, Some(6));
    assert_eq!(out.height, Some(4));
    assert_eq!(out.pixel_format, Some(PixelFormat::Rgba));
    assert_eq!(out.bit_rate, Some(1234));
}

#[test]
fn transpose_announces_swapped_extent() {
    let mut f = make(
        "transpose",
        json!({}),
        &[port("in", PixelFormat::Gray8, 4, 2)],
    );
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Gray8, 6, 3)).unwrap();
    assert_eq!(event_shape(&c.0[0].1), (3, 6, PixelFormat::Gray8));
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Gray8, 6, 3, 8)),
    )
    .unwrap();
    let Frame::Video(v) = &c.0[1].1 else {
        panic!("frame expected")
    };
    assert_eq!(picture_extent(PixelFormat::Gray8, v), (3, 6));
}

#[test]
fn compose_refuses_a_change_on_one_input_and_keeps_its_buffer() {
    let inputs = [
        port("src", PixelFormat::Rgba, 4, 4),
        port("dst", PixelFormat::Rgba, 4, 4),
    ];
    let mut f = make("composite-over", json!({}), &inputs);
    let mut c = Collect(Vec::new());
    f.push(
        &mut c,
        1,
        &Frame::Video(picture(PixelFormat::Rgba, 4, 4, 9)),
    )
    .unwrap();
    let err = f
        .push(&mut c, 0, &event(PixelFormat::Rgba, 8, 8))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("must agree"), "{err}");
    assert!(c.0.is_empty());
    assert_eq!(shape(&f.input_ports()[0]), (4, 4, PixelFormat::Rgba));
    // The buffered dst still pairs with the next src.
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 4, 4, 10)),
    )
    .unwrap();
    assert_eq!(picture_extent(PixelFormat::Rgba, only_video(&c)), (4, 4));

    // An event that does not change the shape is accepted and forwarded.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 1, &event(PixelFormat::Rgba, 4, 4)).unwrap();
    assert_eq!(c.0.len(), 1);
    assert_eq!(event_shape(&c.0[0].1), (4, 4, PixelFormat::Rgba));
}

#[test]
fn stateful_overlay_buffer_survives_a_replan_of_the_other_input() {
    // `watermark` holds the overlay frame (port 1) until a base frame
    // (port 0) arrives. A base-size change rebuilds the plan: the held
    // overlay is carried into it, the stale base frame is dropped.
    let inputs = [
        port("src", PixelFormat::Rgba, 8, 8),
        port("dst", PixelFormat::Rgba, 2, 2),
    ];
    let mut f = make("watermark", json!({"x": 1, "y": 1}), &inputs);
    let mut c = Collect(Vec::new());
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 8, 8, 11)),
    )
    .unwrap();
    let overlay = Frame::Video(picture(PixelFormat::Rgba, 2, 2, 12));
    f.push(&mut c, 1, &overlay).unwrap();
    assert_eq!(c.0.len(), 1, "the pair emitted");

    // A stale old-shape base frame is held, then the base size changes:
    // the stale frame is dropped with the old plan.
    let mut c = Collect(Vec::new());
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 8, 8, 13)),
    )
    .unwrap();
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 16, 16))
        .unwrap();
    assert_eq!(c.0.len(), 1, "only the announcement: {:?}", c.0);
    f.push(&mut c, 1, &overlay).unwrap();
    assert_eq!(c.0.len(), 1, "the stale base frame did not pair");

    // The overlay is now held; the next base change carries it over.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 12, 10))
        .unwrap();
    assert_eq!(c.0.len(), 1, "only the announcement: {:?}", c.0);
    assert_eq!(event_shape(&c.0[0].1), (12, 10, PixelFormat::Rgba));
    let base = Frame::Video(picture(PixelFormat::Rgba, 12, 10, 14));
    f.push(&mut c, 0, &base).unwrap();
    assert_eq!(c.0.len(), 2, "the held overlay paired with the new base");
    let Frame::Video(got) = &c.0[1].1 else {
        panic!("frame expected")
    };
    assert_eq!(picture_extent(PixelFormat::Rgba, got), (12, 10));

    // Same pixels as a filter planned for 12x10 from the start.
    let mut fresh = make(
        "watermark",
        json!({"x": 1, "y": 1}),
        &[
            port("src", PixelFormat::Rgba, 12, 10),
            port("dst", PixelFormat::Rgba, 2, 2),
        ],
    );
    let mut c2 = Collect(Vec::new());
    fresh.push(&mut c2, 0, &base).unwrap();
    fresh.push(&mut c2, 1, &overlay).unwrap();
    assert_eq!(pixels(got), pixels(only_video(&c2)));

    // A reset clears the held state: an overlay pushed before it is gone.
    let mut c = Collect(Vec::new());
    f.push(&mut c, 1, &overlay).unwrap();
    f.reset().unwrap();
    f.push(&mut c, 0, &event(PixelFormat::Rgba, 10, 10))
        .unwrap();
    f.push(
        &mut c,
        0,
        &Frame::Video(picture(PixelFormat::Rgba, 10, 10, 15)),
    )
    .unwrap();
    assert_eq!(c.0.len(), 1, "announcement only, no frame: {:?}", c.0);
}

#[test]
fn non_video_event_on_a_video_port_is_unsupported() {
    let mut f = make("negate", json!({}), &[port("in", PixelFormat::Rgba, 4, 4)]);
    let p = CodecParameters::audio(CodecId::new("pcm_s16le"));
    let mut c = Collect(Vec::new());
    let err = f
        .push(&mut c, 0, &Frame::StreamParams(Box::new(p)))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}
