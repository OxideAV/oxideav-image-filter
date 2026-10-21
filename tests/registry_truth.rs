//! Registry truthfulness: for every factory name the crate registers,
//! build the filter through `RuntimeContext::filters` on a video port
//! of each probe layout, push a picture carrying every side-channel
//! record, and check that what comes out of port 0 matches the output
//! port the filter *declared* — plane count and exact plane geometry
//! for the declared format and extent — and still carries the input's
//! blobs. A factory that refuses a layout must do so with a typed
//! error, and a filter that was built must not panic on push.

use oxideav_core::{
    filter::FilterContext, BlobKind, ColorSignal, Frame, LayerIdentity, MetadataBlob, PixelFormat,
    PortParams, PortSpec, Result, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_image_filter::registry::{register, registered_names};
use serde_json::{json, Value};

struct Collect(Vec<(usize, Frame)>);

impl FilterContext for Collect {
    fn emit(&mut self, port: usize, frame: Frame) -> Result<()> {
        self.0.push((port, frame));
        Ok(())
    }
}

fn picture(fmt: PixelFormat, w: u32, h: u32, seed: u8) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (_, rows) = fmt.plane_dimensions(p, w, h).unwrap();
            let stride = fmt.plane_row_bytes(p, w).unwrap();
            let data = (0..stride * rows as usize)
                .map(|i| ((i as u32 * 29 + p as u32 * 7 + seed as u32) % 251) as u8)
                .collect();
            VideoPlane { stride, data }
        })
        .collect();
    let mut f = VideoFrame {
        pts: Some(1),
        planes,
    };
    if fmt == PixelFormat::Pal8 {
        for v in &mut f.planes[0].data {
            *v %= 4;
        }
    }
    f.with_blob(MetadataBlob::new(BlobKind::ICC, vec![1, 2, 3]))
        .with_palette(vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255])
        .with_palette_alpha(vec![0, 255, 255, 255])
        .with_color_signal(ColorSignal::default())
        .with_layer(LayerIdentity::default())
}

/// Parameters for the factories that need some.
fn params_for(name: &str) -> Value {
    match name {
        "resize" => json!({"width": 11, "height": 7}),
        "color-matrix" | "recolor" => json!({"matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}),
        "crop" => json!({"x": 1, "y": 2, "width": 9, "height": 7}),
        "extent" => json!({"width": 20, "height": 20}),
        "shave" => json!({"x": 1, "y": 1}),
        "roll" => json!({"dx": 3, "dy": -2}),
        "rotate" => json!({"degrees": 90}),
        "convolve" => json!({"size": 3, "kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0]}),
        "gravity-translate" | "gravity" => json!({"width": 20, "height": 20, "gravity": "center"}),
        "channel-extract" => json!({"channel": "red"}),
        "evaluate" => json!({"op": "add", "value": 3}),
        "function" => json!({"op": "polynomial", "args": [1.0, 0.0]}),
        "statistic" => json!({"op": "median", "width": 3, "height": 3}),
        "gradient-map" | "duotone" => {
            json!({"stops": [{"position": 0.0, "color": [0, 0, 0]}, {"position": 1.0, "color": [255, 255, 255]}]})
        }
        "perspective" => {
            json!({"src": [[0, 0], [1, 0], [1, 1], [0, 1]], "dst": [[0, 0], [1, 0], [1, 1], [0, 1]]})
        }
        "affine" => json!({"matrix": [1, 0, 0, 1, 0, 0]}),
        "srt" => json!({"angle": 0}),
        "distort" => json!({"k1": 0.0, "k2": 0.0}),
        "barrel-inverse" => json!({"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0}),
        "flood-fill" => json!({"x": 0, "y": 0, "color": [1, 2, 3, 255]}),
        "watermark" => json!({"overlay_width": 16, "overlay_height": 16}),
        "frame" => json!({"width": 2, "height": 2}),
        "canvas" => json!({"color": [1, 2, 3, 255]}),
        "curves" => json!({"master": [[0, 0], [255, 255]]}),
        "morphology-dilate" | "morphology-erode" | "morphology-open" | "morphology-close" => {
            json!({"element": "square"})
        }
        "stegano" => json!({"offset": 0}),
        "tilt-shift" => json!({}),
        _ => json!({}),
    }
}

fn video_port(name: &str, fmt: PixelFormat, w: u32, h: u32) -> PortSpec {
    PortSpec::video(name, w, h, fmt, TimeBase::new(1, 30))
}

const TWO_INPUT: &[&str] = &[
    "clut",
    "hald-clut",
    "stegano",
    "stereo",
    "difference",
    "watermark",
    "displacement-map",
];

fn is_two_input(name: &str) -> bool {
    name.starts_with("composite-") || TWO_INPUT.contains(&name)
}

#[test]
fn every_factory_emits_what_its_output_port_declares() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let names = registered_names();
    assert!(names.len() >= 200, "{} names", names.len());
    let (w, h) = (16u32, 16u32);
    let mut built = 0usize;
    let mut refused = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for name in &names {
        for &fmt in &[
            PixelFormat::Rgba,
            PixelFormat::Rgb24,
            PixelFormat::Gray8,
            PixelFormat::Yuv420P,
            PixelFormat::YuvJ422P,
            PixelFormat::Gray16Le,
            PixelFormat::Pal8,
        ] {
            let inputs: Vec<PortSpec> = if is_two_input(name) {
                vec![video_port("src", fmt, w, h), video_port("dst", fmt, w, h)]
            } else {
                vec![video_port("in", fmt, w, h)]
            };
            let mut filter = match ctx.filters.make(name, &params_for(name), &inputs) {
                Ok(f) => f,
                Err(e) => {
                    failures.push(format!("{name} on {fmt:?}: factory failed: {e}"));
                    continue;
                }
            };
            let out_port = filter.output_ports()[0].clone();
            let PortParams::Video {
                width: ow,
                height: oh,
                format: of,
                ..
            } = out_port.params
            else {
                failures.push(format!("{name}: output port is not video"));
                continue;
            };
            let mut sink = Collect(Vec::new());
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut r = filter.push(&mut sink, 0, &Frame::Video(picture(fmt, w, h, 3)));
                if r.is_ok() && is_two_input(name) {
                    let dst_dims = if name == "watermark" {
                        (16, 16)
                    } else {
                        (w, h)
                    };
                    r = filter.push(
                        &mut sink,
                        1,
                        &Frame::Video(picture(fmt, dst_dims.0, dst_dims.1, 5)),
                    );
                }
                r
            }));
            match r {
                Err(_) => failures.push(format!("{name} on {fmt:?}: PANICKED on push")),
                Ok(Err(oxideav_core::Error::Unsupported(msg))) => {
                    if !msg.contains(&format!("{fmt:?}")) {
                        failures.push(format!(
                            "{name} on {fmt:?}: refusal does not name the layout: {msg}"
                        ));
                    }
                    refused += 1;
                }
                Ok(Err(e)) => failures.push(format!("{name} on {fmt:?}: push failed: {e}")),
                Ok(Ok(())) => {
                    let Some((_, Frame::Video(out))) = sink.0.first() else {
                        failures.push(format!("{name} on {fmt:?}: nothing emitted"));
                        continue;
                    };
                    built += 1;
                    if out.image_plane_count() != of.plane_count() {
                        failures.push(format!(
                            "{name} on {fmt:?}: port says {of:?} ({} planes), emitted {} planes",
                            of.plane_count(),
                            out.image_plane_count()
                        ));
                        continue;
                    }
                    // Extent: trust the declared one, except for the
                    // filters whose extent depends on the picture content
                    // (a trim's bounding box, a deskew's rotated bounds):
                    // their factories cannot know it up front and mirror
                    // the input extent — derive the real one from plane 0
                    // and check the planes against it.
                    let content_extent = matches!(name.as_str(), "trim" | "auto-trim" | "deskew");
                    let (ew, eh) = if ow == 0 || oh == 0 || content_extent {
                        let pb = of.plane_row_bytes(0, 1).unwrap();
                        let p0 = &out.planes[0];
                        ((p0.stride / pb) as u32, (p0.data.len() / p0.stride) as u32)
                    } else {
                        (ow, oh)
                    };
                    for (i, p) in out.image_planes().iter().enumerate() {
                        let rb = of.plane_row_bytes(i, ew).unwrap();
                        let rows = of.plane_dimensions(i, ew, eh).unwrap().1 as usize;
                        if p.stride != rb || p.data.len() != rb * rows {
                            failures.push(format!(
                                "{name} on {fmt:?}: port says {of:?} {ew}x{eh}; plane {i} is stride {} / {} bytes (expected {rb} / {})",
                                p.stride,
                                p.data.len(),
                                rb * rows
                            ));
                            break;
                        }
                    }
                    if out.blobs().len() != 1 {
                        failures.push(format!("{name} on {fmt:?}: blobs lost through the adapter"));
                    }
                }
            }
        }
    }
    eprintln!(
        "registry: {} names, {built} built+emitted cells, {refused} typed refusals",
        names.len()
    );
    assert!(
        failures.is_empty(),
        "{} failing cells:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
