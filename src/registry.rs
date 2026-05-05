//! Factory glue + `register` entry point.
//!
//! Lifts the image-filter factories (blur / edge / resize) and the
//! legacy `ImageFilterAdapter` shim out of the old
//! `oxideav-pipeline::filter_registry` module — they live with the
//! concrete filters now so that `oxideav-pipeline` itself doesn't
//! need to depend on this crate.

use oxideav_core::{
    filter::FilterContext, Error, Frame, PixelFormat, PortParams, PortSpec, Result, RuntimeContext,
    StreamFilter, TimeBase,
};
use serde_json::Value;

use crate::{ImageFilter, Planes, VideoStreamParams};

/// Install Blur, Edge, and Resize into the runtime context's filter
/// registry. Idempotent — last write wins per filter name.
///
/// Also auto-registered into [`oxideav_core::REGISTRARS`] via the
/// [`oxideav_core::register!`] macro below so consumers calling
/// [`oxideav_core::RuntimeContext::with_all_features`] pick the
/// image filters up without any explicit umbrella plumbing.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.filters.register("blur", Box::new(make_blur));
    ctx.filters.register("edge", Box::new(make_edge));
    ctx.filters.register("resize", Box::new(make_resize));
}

oxideav_core::register!("image_filter", register);

/// Wraps a legacy [`ImageFilter`] in the [`StreamFilter`] contract.
/// Single video port in, single video port out. The stream-level video
/// shape ([`VideoStreamParams`]) is cached once at construction off
/// the input port and threaded into every `apply()` call — the trait
/// used to read these off the frame, but they live on the stream's
/// `CodecParameters` now.
struct ImageFilterAdapter {
    inner: Box<dyn ImageFilter>,
    inp: [PortSpec; 1],
    outp: [PortSpec; 1],
    params: VideoStreamParams,
}

impl ImageFilterAdapter {
    fn new(inner: Box<dyn ImageFilter>, in_port: PortSpec, out_port: PortSpec) -> Self {
        let params = match &in_port.params {
            PortParams::Video {
                format,
                width,
                height,
                ..
            } => VideoStreamParams {
                format: *format,
                width: *width,
                height: *height,
            },
            _ => VideoStreamParams {
                format: PixelFormat::Yuv420P,
                width: 0,
                height: 0,
            },
        };
        Self {
            inner,
            inp: [in_port],
            outp: [out_port],
            params,
        }
    }
}

impl StreamFilter for ImageFilterAdapter {
    fn input_ports(&self) -> &[PortSpec] {
        &self.inp
    }
    fn output_ports(&self) -> &[PortSpec] {
        &self.outp
    }
    fn push(&mut self, ctx: &mut dyn FilterContext, port: usize, frame: &Frame) -> Result<()> {
        if port != 0 {
            return Err(Error::invalid(format!(
                "image-filter adapter: unknown input port {port}"
            )));
        }
        let Frame::Video(v) = frame else {
            return Err(Error::invalid(
                "image-filter adapter: input port 0 only accepts video frames",
            ));
        };
        let out = self.inner.apply(v, self.params)?;
        ctx.emit(0, Frame::Video(out))?;
        Ok(())
    }
}

fn video_in_port(inputs: &[PortSpec]) -> PortSpec {
    inputs
        .iter()
        .find(|p| matches!(p.params, PortParams::Video { .. }))
        .cloned()
        .unwrap_or_else(|| PortSpec::video("in", 0, 0, PixelFormat::Yuv420P, TimeBase::new(1, 30)))
}

fn make_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Blur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let radius = get_u64("radius").unwrap_or(3) as u32;
    let planes = match get_str("planes") {
        Some("luma") => Planes::Luma,
        Some("chroma") => Planes::Chroma,
        Some(other) if other != "all" => {
            return Err(Error::invalid(format!(
                "job: filter 'blur': unknown planes '{other}' (expected 'all', 'luma', or 'chroma')"
            )));
        }
        _ => Planes::All,
    };
    let mut f = Blur::new(radius).with_planes(planes);
    if let Some(s) = get_f64("sigma") {
        f = f.with_sigma(s as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = PortSpec {
        name: "video".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_edge(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Edge;
    let f = Edge::new();
    let in_port = video_in_port(inputs);
    // Edge collapses any input to single-plane luma.
    let out_port = match &in_port.params {
        PortParams::Video {
            width,
            height,
            time_base,
            ..
        } => PortSpec::video("video", *width, *height, PixelFormat::Gray8, *time_base),
        _ => PortSpec::video("video", 0, 0, PixelFormat::Gray8, TimeBase::new(1, 30)),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_resize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Interpolation, Resize};
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let w = get_u64("width")
        .ok_or_else(|| Error::invalid("job: filter 'resize' needs unsigned `width`"))?
        as u32;
    let h = get_u64("height")
        .ok_or_else(|| Error::invalid("job: filter 'resize' needs unsigned `height`"))?
        as u32;
    let interp = match get_str("interpolation") {
        Some("nearest") => Interpolation::Nearest,
        None | Some("bilinear") => Interpolation::Bilinear,
        Some(other) => {
            return Err(Error::invalid(format!(
                "job: filter 'resize': unknown interpolation '{other}' \
                 (expected 'nearest' or 'bilinear')"
            )));
        }
    };
    let f = Resize::new(w, h).with_interpolation(interp);
    let in_port = video_in_port(inputs);
    let out_port = match &in_port.params {
        PortParams::Video {
            format, time_base, ..
        } => PortSpec::video("video", w, h, *format, *time_base),
        _ => PortSpec::video("video", w, h, PixelFormat::Yuv420P, TimeBase::new(1, 30)),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}
