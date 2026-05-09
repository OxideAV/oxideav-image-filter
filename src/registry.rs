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

/// Install Blur, Edge, Resize, Sharpen, Gamma, and BrightnessContrast
/// (plus the `brightness` / `contrast` single-axis aliases) into the
/// runtime context's filter registry. Idempotent — last write wins per
/// filter name.
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.filters.register("blur", Box::new(make_blur));
    ctx.filters.register("edge", Box::new(make_edge));
    ctx.filters.register("resize", Box::new(make_resize));
    ctx.filters.register("sharpen", Box::new(make_sharpen));
    ctx.filters.register("gamma", Box::new(make_gamma));
    ctx.filters
        .register("brightness-contrast", Box::new(make_brightness_contrast));
    ctx.filters
        .register("brightness", Box::new(make_brightness));
    ctx.filters.register("contrast", Box::new(make_contrast));
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

fn passthrough_out_port(in_port: &PortSpec) -> PortSpec {
    PortSpec {
        name: "video".to_string(),
        ..in_port.clone()
    }
}

fn make_sharpen(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Sharpen;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let radius = get_u64("radius").unwrap_or(1) as u32;
    let sigma = get_f64("sigma").unwrap_or(0.5) as f32;
    let mut f = Sharpen::new(radius, sigma);
    if let Some(a) = get_f64("amount") {
        f = f.with_amount(a as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_gamma(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Gamma;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let gamma = get_f64("gamma").or_else(|| get_f64("value")).unwrap_or(1.0) as f32;
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(Error::invalid(format!(
            "job: filter 'gamma': value must be a positive finite number (got {gamma})"
        )));
    }
    let f = Gamma::new(gamma);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_brightness_contrast(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BrightnessContrast;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let brightness = get_f64("brightness").unwrap_or(0.0) as f32;
    let contrast = get_f64("contrast").unwrap_or(0.0) as f32;
    let f = BrightnessContrast::new(brightness, contrast);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_brightness(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BrightnessContrast;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let brightness = get_f64("brightness")
        .or_else(|| get_f64("value"))
        .or_else(|| get_f64("delta"))
        .unwrap_or(0.0) as f32;
    let f = BrightnessContrast::new(brightness, 0.0);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_contrast(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BrightnessContrast;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let contrast = get_f64("contrast")
        .or_else(|| get_f64("value"))
        .or_else(|| get_f64("factor"))
        .unwrap_or(0.0) as f32;
    let f = BrightnessContrast::new(0.0, contrast);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> RuntimeContext {
        let mut c = RuntimeContext::new();
        register(&mut c);
        c
    }

    fn yuv_in_port() -> PortSpec {
        PortSpec::video("in", 16, 16, PixelFormat::Yuv420P, TimeBase::new(1, 30))
    }

    #[test]
    fn registers_round_next_filters() {
        let c = ctx();
        for name in [
            "blur",
            "edge",
            "resize",
            "sharpen",
            "gamma",
            "brightness-contrast",
            "brightness",
            "contrast",
        ] {
            assert!(c.filters.contains(name), "missing factory: {name}");
        }
    }

    #[test]
    fn sharpen_factory_builds() {
        let c = ctx();
        let inputs = [yuv_in_port()];
        let f = c
            .filters
            .make(
                "sharpen",
                &json!({"radius": 1, "sigma": 0.8, "amount": 1.2}),
                &inputs,
            )
            .expect("sharpen factory");
        assert_eq!(f.input_ports().len(), 1);
        assert_eq!(f.output_ports().len(), 1);
    }

    #[test]
    fn gamma_factory_rejects_non_positive() {
        let c = ctx();
        let inputs = [yuv_in_port()];
        assert!(
            c.filters
                .make("gamma", &json!({"value": 0.0}), &inputs)
                .is_err(),
            "gamma must reject value = 0"
        );
        assert!(
            c.filters
                .make("gamma", &json!({"value": -1.5}), &inputs)
                .is_err(),
            "gamma must reject negative value"
        );
    }

    #[test]
    fn gamma_factory_accepts_value_or_gamma_key() {
        let c = ctx();
        let inputs = [yuv_in_port()];
        c.filters
            .make("gamma", &json!({"value": 1.8}), &inputs)
            .expect("gamma {value}");
        c.filters
            .make("gamma", &json!({"gamma": 2.2}), &inputs)
            .expect("gamma {gamma}");
        c.filters
            .make("gamma", &json!({}), &inputs)
            .expect("gamma defaults to identity");
    }

    #[test]
    fn brightness_contrast_factory_builds_and_aliases() {
        let c = ctx();
        let inputs = [yuv_in_port()];
        c.filters
            .make(
                "brightness-contrast",
                &json!({"brightness": 10.0, "contrast": -5.0}),
                &inputs,
            )
            .expect("brightness-contrast");
        c.filters
            .make("brightness", &json!({"value": 25.0}), &inputs)
            .expect("brightness alias");
        c.filters
            .make("contrast", &json!({"value": 25.0}), &inputs)
            .expect("contrast alias");
    }

    #[test]
    fn factories_resolve_via_video_prefix() {
        let c = ctx();
        let inputs = [yuv_in_port()];
        // The schema allows `video.<name>` / `v:<name>` — the registry
        // strips the prefix before lookup.
        c.filters
            .make("video.sharpen", &json!({}), &inputs)
            .expect("video.sharpen");
        c.filters
            .make("v:gamma", &json!({"value": 1.5}), &inputs)
            .expect("v:gamma");
    }
}
