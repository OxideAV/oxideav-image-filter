#![no_main]

//! Panic-freedom fuzz target for the registry seam: every factory
//! name the crate registers, built through `RuntimeContext::filters`
//! with attacker-chosen JSON parameters on an attacker-chosen input
//! port (layout, extent, including the 0×0 placeholder port), then
//! pushed arbitrary frames on every port. Construction may fail with a
//! typed error; a built filter must survive every push.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    filter::FilterContext, Frame, PixelFormat, PortSpec, Result, RuntimeContext, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_image_filter::layout;
use oxideav_image_filter::registry::{register, registered_names};
use serde_json::{json, Value};

struct Sink;

impl FilterContext for Sink {
    fn emit(&mut self, _port: usize, _frame: Frame) -> Result<()> {
        Ok(())
    }
}

fn next(data: &[u8], pos: &mut usize) -> u8 {
    let b = data.get(*pos).copied().unwrap_or(0);
    *pos += 1;
    b
}

/// A JSON object whose values are a few arbitrary numbers / strings
/// under the keys the factories look for.
fn params(data: &[u8], pos: &mut usize) -> Value {
    const KEYS: &[&str] = &[
        "width", "height", "x", "y", "dx", "dy", "radius", "sigma", "amount", "degrees",
        "threshold", "levels", "factor", "angle", "low", "high", "gamma", "brightness",
        "contrast", "planes", "interpolation", "channel", "op", "mode", "kernel", "matrix",
        "size", "color", "element", "stops", "value", "args", "overlay_width", "overlay_height",
        "iterations", "key", "bias", "offset", "scale", "strength", "cell", "block",
    ];
    let mut m = serde_json::Map::new();
    let n = next(data, pos) as usize % 6;
    for _ in 0..n {
        let key = KEYS[next(data, pos) as usize % KEYS.len()];
        let v = match next(data, pos) % 5 {
            0 => json!(next(data, pos) as i64 - 64),
            1 => json!(next(data, pos) as f64 / 17.0 - 3.0),
            2 => json!(["luma", "all", "chroma", "red", "nearest", "add", "square", "median", "center"]
                [next(data, pos) as usize % 9]),
            3 => json!([next(data, pos), next(data, pos), next(data, pos), next(data, pos)]),
            _ => json!({"position": 0.5, "color": [1, 2, 3]}),
        };
        m.insert(key.to_string(), v);
    }
    Value::Object(m)
}

fn frame(data: &[u8], pos: &mut usize, fmt: PixelFormat, w: u32, h: u32) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let rows = fmt.plane_dimensions(p, w, h).map(|d| d.1).unwrap_or(0) as usize;
            let stride = fmt.plane_row_bytes(p, w).unwrap_or(0);
            let want = (stride * rows).min(4096);
            let len = match next(data, pos) % 4 {
                0 => want.saturating_sub(next(data, pos) as usize % 5),
                _ => want,
            };
            let start = (*pos).min(data.len());
            let end = (start + len).min(data.len());
            *pos = end;
            let mut d = data[start..end].to_vec();
            d.resize(len, 0x33);
            VideoPlane { stride, data: d }
        })
        .collect();
    let mut f = VideoFrame { pts: None, planes };
    if next(data, pos) & 1 == 1 {
        f.set_blobs(vec![oxideav_core::MetadataBlob::new(
            oxideav_core::BlobKind::XMP,
            vec![next(data, pos)],
        )]);
    }
    f
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let names = registered_names();
    let mut pos = 0usize;
    let name = &names[next(data, &mut pos) as usize % names.len()];
    let fmt = layout::ALL[next(data, &mut pos) as usize % layout::ALL.len()];
    let w = next(data, &mut pos) as u32 % 20;
    let h = next(data, &mut pos) as u32 % 20;
    let inputs = [
        PortSpec::video("src", w, h, fmt, TimeBase::new(1, 30)),
        PortSpec::video("dst", w, h, fmt, TimeBase::new(1, 30)),
    ];
    let p = params(data, &mut pos);
    let Ok(mut filter) = ctx.filters.make(name, &p, &inputs) else {
        return;
    };
    let ports = filter.input_ports().len();
    let mut sink = Sink;
    for _ in 0..2 {
        for port in 0..ports {
            let fw = if next(data, &mut pos) & 1 == 1 { w } else { 1 + w % 7 };
            let fr = frame(data, &mut pos, fmt, fw, h);
            let _ = filter.push(&mut sink, port, &Frame::Video(fr));
        }
    }
    let _ = filter.flush(&mut sink);
    let _ = filter.reset();
});
