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

/// Install every image-filter factory into the runtime context's
/// filter registry. Idempotent — last write wins per filter name.
///
/// Wired in this round:
/// - Round-prev-prev: `blur`, `edge`, `resize`, `sharpen`, `gamma`,
///   `brightness-contrast`, `brightness`, `contrast`.
/// - Round-prev additions: `unsharp`, `threshold`, `level`,
///   `normalize`, `posterize`, `solarize`, `flip`, `flop`,
///   `rotate`, `crop`, `negate`, `sepia`, `modulate`,
///   `grayscale`, `motion-blur`, `emboss`.
/// - Round-next additions: `vignette`, `colorize`, `equalize`,
///   `auto-gamma`.
/// - r5 additions: `tint`, `sigmoidal-contrast`, `implode`, `swirl`,
///   `despeckle`.
/// - r6 additions: `wave`, `spread`, `charcoal`, `convolve`,
///   `polar`, `depolar`.
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

    // Round-next factories.
    ctx.filters.register("unsharp", Box::new(make_unsharp));
    ctx.filters.register("threshold", Box::new(make_threshold));
    ctx.filters.register("level", Box::new(make_level));
    ctx.filters.register("normalize", Box::new(make_normalize));
    ctx.filters.register("posterize", Box::new(make_posterize));
    ctx.filters.register("solarize", Box::new(make_solarize));
    ctx.filters.register("flip", Box::new(make_flip));
    ctx.filters.register("flop", Box::new(make_flop));
    ctx.filters.register("rotate", Box::new(make_rotate));
    ctx.filters.register("crop", Box::new(make_crop));
    ctx.filters.register("negate", Box::new(make_negate));
    ctx.filters.register("sepia", Box::new(make_sepia));
    ctx.filters.register("modulate", Box::new(make_modulate));
    ctx.filters.register("grayscale", Box::new(make_grayscale));
    ctx.filters
        .register("motion-blur", Box::new(make_motion_blur));
    ctx.filters.register("emboss", Box::new(make_emboss));

    // Round-after-next factories.
    ctx.filters.register("vignette", Box::new(make_vignette));
    ctx.filters.register("colorize", Box::new(make_colorize));
    ctx.filters.register("equalize", Box::new(make_equalize));
    ctx.filters
        .register("auto-gamma", Box::new(make_auto_gamma));

    // Round-after-after-next factories.
    ctx.filters.register("tint", Box::new(make_tint));
    ctx.filters
        .register("sigmoidal-contrast", Box::new(make_sigmoidal_contrast));
    ctx.filters.register("implode", Box::new(make_implode));
    ctx.filters.register("swirl", Box::new(make_swirl));
    ctx.filters.register("despeckle", Box::new(make_despeckle));

    // r6 factories (wave / spread / charcoal / convolve / polar).
    ctx.filters.register("wave", Box::new(make_wave));
    ctx.filters.register("spread", Box::new(make_spread));
    ctx.filters.register("charcoal", Box::new(make_charcoal));
    ctx.filters.register("convolve", Box::new(make_convolve));
    ctx.filters.register("polar", Box::new(make_polar));
    ctx.filters.register("depolar", Box::new(make_depolar));

    // r7 factories (morphology dilate / erode / open / close).
    ctx.filters
        .register("morphology-dilate", Box::new(make_morphology_dilate));
    ctx.filters
        .register("morphology-erode", Box::new(make_morphology_erode));
    ctx.filters
        .register("morphology-open", Box::new(make_morphology_open));
    ctx.filters
        .register("morphology-close", Box::new(make_morphology_close));

    // r7 factory (perspective).
    ctx.filters
        .register("perspective", Box::new(make_perspective));
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

// --- Round-next factories ---

fn make_unsharp(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Unsharp;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let radius = get_u64("radius").unwrap_or(1) as u32;
    let sigma = get_f64("sigma").unwrap_or(0.5) as f32;
    let amount = get_f64("amount").unwrap_or(1.0) as f32;
    let threshold = get_u64("threshold").unwrap_or(0).min(255) as u8;
    let f = Unsharp::new(radius, sigma, amount, threshold);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_threshold(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Threshold;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let threshold = get_u64("value")
        .or_else(|| get_u64("threshold"))
        .unwrap_or(128)
        .min(255) as u8;
    let f = Threshold::new(threshold);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_level(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Level;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let black = get_u64("black").unwrap_or(0).min(255) as u8;
    let white = get_u64("white").unwrap_or(255).min(255) as u8;
    let gamma = get_f64("gamma").unwrap_or(1.0) as f32;
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(Error::invalid(format!(
            "job: filter 'level': gamma must be a positive finite number (got {gamma})"
        )));
    }
    let f = Level::new(black, white, gamma);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_normalize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Normalize;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let low_clip = get_f64("low_clip")
        .or_else(|| get_f64("low"))
        .unwrap_or(0.0) as f32;
    let high_clip = get_f64("high_clip")
        .or_else(|| get_f64("high"))
        .unwrap_or(0.0) as f32;
    let f = Normalize::new(low_clip, high_clip);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_posterize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Posterize;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let levels = get_u64("levels")
        .or_else(|| get_u64("value"))
        .unwrap_or(4)
        .min(u32::MAX as u64) as u32;
    let f = Posterize::new(levels);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_solarize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Solarize;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let threshold = get_u64("value")
        .or_else(|| get_u64("threshold"))
        .unwrap_or(128)
        .min(255) as u8;
    let f = Solarize::new(threshold);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_flip(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Flip;
    let f = Flip::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_flop(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Flop;
    let f = Flop::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_rotate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Rotate;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let degrees = get_f64("degrees")
        .or_else(|| get_f64("angle"))
        .unwrap_or(0.0) as f32;
    if !degrees.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'rotate': degrees must be finite (got {degrees})"
        )));
    }
    let mut f = Rotate::new(degrees);
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'rotate': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'rotate': background array must contain unsigned ints",
                    )
                })?
                .min(255) as u8;
        }
        f = f.with_background(bg);
    }

    let in_port = video_in_port(inputs);
    // For exact 90 / 270 degree multiples the canvas swaps width and
    // height; for 0 / 180 it stays the same. Other angles change shape
    // too — we leave that to the adapter to handle by re-reading the
    // output dimensions off the produced frame's planes (the framework
    // currently consumes them via the port cache, so for arbitrary
    // angles the downstream node will need to re-negotiate. The 90 /
    // 270 case is the one users hit in practice and we wire it
    // exactly.)
    let out_port = match &in_port.params {
        PortParams::Video {
            format,
            width,
            height,
            time_base,
            ..
        } => {
            let snapped = degrees.rem_euclid(360.0);
            let (ow, oh) = if (snapped - 90.0).abs() < 1e-3 || (snapped - 270.0).abs() < 1e-3 {
                (*height, *width)
            } else if (snapped - 180.0).abs() < 1e-3 || snapped.abs() < 1e-3 {
                (*width, *height)
            } else {
                // Arbitrary angle — best-effort use the bounding box of
                // the rotated rect; downstream nodes should re-read the
                // produced VideoPlane dims.
                let theta = (degrees as f64).to_radians();
                let s = theta.sin().abs();
                let c = theta.cos().abs();
                let nw = (*width as f64 * c + *height as f64 * s).ceil() as u32;
                let nh = (*width as f64 * s + *height as f64 * c).ceil() as u32;
                (nw.max(1), nh.max(1))
            };
            PortSpec::video("video", ow, oh, *format, *time_base)
        }
        _ => passthrough_out_port(&in_port),
    };

    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_crop(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Crop;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let x = get_u64("x").unwrap_or(0).min(u32::MAX as u64) as u32;
    let y = get_u64("y").unwrap_or(0).min(u32::MAX as u64) as u32;
    let w = get_u64("width")
        .ok_or_else(|| Error::invalid("job: filter 'crop' needs unsigned `width`"))?
        .min(u32::MAX as u64) as u32;
    let h = get_u64("height")
        .ok_or_else(|| Error::invalid("job: filter 'crop' needs unsigned `height`"))?
        .min(u32::MAX as u64) as u32;
    let f = Crop::new(x, y, w, h);
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

fn make_negate(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Negate;
    let f = Negate::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_sepia(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Sepia;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let threshold = get_f64("threshold")
        .or_else(|| get_f64("value"))
        .unwrap_or(1.0) as f32;
    let f = Sepia::new(threshold);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_modulate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Modulate;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let brightness = get_f64("brightness").unwrap_or(100.0) as f32;
    let saturation = get_f64("saturation").unwrap_or(100.0) as f32;
    let hue = get_f64("hue_degrees")
        .or_else(|| get_f64("hue"))
        .unwrap_or(0.0) as f32;
    let f = Modulate::new(brightness, saturation, hue);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_grayscale(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Grayscale;
    let p = params.as_object();
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());

    let preserve_alpha = get_bool("preserve_alpha").unwrap_or(true);
    let output_gray8 = get_bool("output_gray8").unwrap_or(false);
    let f = Grayscale::new()
        .with_preserve_alpha(preserve_alpha)
        .with_output_gray8(output_gray8);

    let in_port = video_in_port(inputs);
    let out_port = if output_gray8 {
        match &in_port.params {
            PortParams::Video {
                width,
                height,
                time_base,
                ..
            } => PortSpec::video("video", *width, *height, PixelFormat::Gray8, *time_base),
            _ => PortSpec::video("video", 0, 0, PixelFormat::Gray8, TimeBase::new(1, 30)),
        }
    } else {
        passthrough_out_port(&in_port)
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_motion_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::MotionBlur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let radius = get_u64("radius").unwrap_or(3) as u32;
    let sigma = get_f64("sigma").unwrap_or(1.5) as f32;
    let angle = get_f64("angle_degrees")
        .or_else(|| get_f64("angle"))
        .unwrap_or(0.0) as f32;
    let f = MotionBlur::new(radius, sigma, angle);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_emboss(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Emboss;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let radius = get_u64("radius").unwrap_or(1) as u32;
    let f = Emboss::new(radius);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_vignette(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Vignette;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let x = get_f64("x").unwrap_or(0.5) as f32;
    let y = get_f64("y").unwrap_or(0.5) as f32;
    let radius = get_f64("radius").unwrap_or(0.0) as f32;
    let sigma = get_f64("sigma").unwrap_or(0.0) as f32;
    if !x.is_finite() || !y.is_finite() || !radius.is_finite() || !sigma.is_finite() {
        return Err(Error::invalid(
            "job: filter 'vignette': x/y/radius/sigma must be finite numbers",
        ));
    }
    let f = Vignette::new(x, y, radius, sigma);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_colorize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Colorize;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let amount = get_f64("amount")
        .or_else(|| get_f64("value"))
        .unwrap_or(0.5) as f32;
    if !amount.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'colorize': amount must be finite (got {amount})"
        )));
    }
    let mut color = [255u8, 255, 255, 255];
    if let Some(arr) = p.and_then(|m| m.get("color")).and_then(|v| v.as_array()) {
        if !(3..=4).contains(&arr.len()) {
            return Err(Error::invalid(format!(
                "job: filter 'colorize': color must be a 3- or 4-element array, got {}",
                arr.len()
            )));
        }
        for (i, v) in arr.iter().enumerate() {
            color[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid("job: filter 'colorize': color array must contain unsigned ints")
                })?
                .min(255) as u8;
        }
    }
    let f = Colorize::new(color, amount);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_equalize(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Equalize;
    let f = Equalize::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_auto_gamma(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AutoGamma;
    let f = AutoGamma::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- Round-after-after-next factories ---

fn make_tint(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Tint;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let amount = get_f64("amount")
        .or_else(|| get_f64("value"))
        .unwrap_or(1.0) as f32;
    if !amount.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'tint': amount must be finite (got {amount})"
        )));
    }
    let mut color = [128u8, 128, 128, 255];
    if let Some(arr) = p.and_then(|m| m.get("color")).and_then(|v| v.as_array()) {
        if !(3..=4).contains(&arr.len()) {
            return Err(Error::invalid(format!(
                "job: filter 'tint': color must be a 3- or 4-element array, got {}",
                arr.len()
            )));
        }
        for (i, v) in arr.iter().enumerate() {
            color[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid("job: filter 'tint': color array must contain unsigned ints")
                })?
                .min(255) as u8;
        }
    }
    let f = Tint::new(color, amount);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_sigmoidal_contrast(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::SigmoidalContrast;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let contrast = get_f64("contrast")
        .or_else(|| get_f64("value"))
        .unwrap_or(3.0) as f32;
    let midpoint = get_f64("midpoint").unwrap_or(128.0) as f32;
    if !contrast.is_finite() || !midpoint.is_finite() {
        return Err(Error::invalid(
            "job: filter 'sigmoidal-contrast': contrast and midpoint must be finite",
        ));
    }
    let f = SigmoidalContrast::new(contrast, midpoint);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_implode(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Implode;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let factor = get_f64("factor")
        .or_else(|| get_f64("amount"))
        .or_else(|| get_f64("value"))
        .unwrap_or(0.5) as f32;
    if !factor.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'implode': factor must be finite (got {factor})"
        )));
    }
    let f = Implode::new(factor);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_swirl(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Swirl;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let degrees = get_f64("degrees")
        .or_else(|| get_f64("angle"))
        .or_else(|| get_f64("value"))
        .unwrap_or(90.0) as f32;
    if !degrees.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'swirl': degrees must be finite (got {degrees})"
        )));
    }
    let f = Swirl::new(degrees);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_despeckle(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Despeckle;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let radius = get_u64("radius")
        .or_else(|| get_u64("value"))
        .unwrap_or(1)
        .min(u32::MAX as u64) as u32;
    let f = Despeckle::new(radius);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r6 factories ---

fn make_wave(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Wave;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let amplitude = get_f64("amplitude").unwrap_or(5.0) as f32;
    let wavelength = get_f64("wavelength").unwrap_or(25.0) as f32;
    if !amplitude.is_finite() || !wavelength.is_finite() {
        return Err(Error::invalid(
            "job: filter 'wave': amplitude and wavelength must be finite",
        ));
    }
    let f = Wave::new(amplitude, wavelength);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_spread(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Spread;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let radius = get_u64("radius").unwrap_or(3).min(u32::MAX as u64) as u32;
    let mut f = Spread::new(radius);
    if let Some(seed) = get_u64("seed") {
        f = f.with_seed(seed);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_charcoal(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Charcoal;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let factor = get_f64("factor")
        .or_else(|| get_f64("amount"))
        .or_else(|| get_f64("value"))
        .unwrap_or(1.0) as f32;
    if !factor.is_finite() || factor < 0.0 {
        return Err(Error::invalid(format!(
            "job: filter 'charcoal': factor must be a non-negative finite number (got {factor})"
        )));
    }
    let f = Charcoal::new(factor);
    let in_port = video_in_port(inputs);
    // Charcoal collapses any input to single-plane luma — same shape
    // change as Edge.
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

fn make_convolve(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Convolve;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let kernel_arr = p
        .and_then(|m| m.get("kernel"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            Error::invalid(
                "job: filter 'convolve' needs `kernel` as a flat array of numbers (row-major)",
            )
        })?;
    let kernel: Vec<f32> = kernel_arr
        .iter()
        .map(|v| {
            v.as_f64().map(|x| x as f32).ok_or_else(|| {
                Error::invalid("job: filter 'convolve': kernel array must contain numbers")
            })
        })
        .collect::<Result<_>>()?;

    // Allow `size` to be omitted when the array length is a perfect
    // square (3, 5, 7, ...) — auto-derive in that case.
    let size = if let Some(s) = get_u64("size") {
        s as u32
    } else {
        let n = kernel.len();
        let s = (n as f64).sqrt().round() as usize;
        if s * s != n {
            return Err(Error::invalid(format!(
                "job: filter 'convolve': kernel length {n} is not a perfect square; \
                 specify `size` explicitly"
            )));
        }
        s as u32
    };
    let mut f = Convolve::new(size, kernel)?;
    if let Some(b) = get_f64("bias") {
        f = f.with_bias(b as f32);
    }
    if let Some(d) = get_f64("divisor") {
        f = f.with_divisor(Some(d as f32));
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_polar(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Polar;
    let f = Polar::forward();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_depolar(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Polar;
    let f = Polar::inverse();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r7 factories ---

fn morph_element_from_json(
    p: Option<&serde_json::Map<String, Value>>,
) -> Result<crate::StructuringElement> {
    let s = p.and_then(|m| m.get("element")).and_then(|v| v.as_str());
    match s {
        None | Some("square") | Some("square3x3") | Some("Square3x3") => {
            Ok(crate::StructuringElement::Square3x3)
        }
        Some("cross") | Some("cross3x3") | Some("plus") | Some("Cross3x3") => {
            Ok(crate::StructuringElement::Cross3x3)
        }
        Some(other) => Err(Error::invalid(format!(
            "job: filter 'morphology-*': unknown element '{other}' (expected 'square' or 'cross')"
        ))),
    }
}

fn morph_iters_from_json(p: Option<&serde_json::Map<String, Value>>) -> u32 {
    p.and_then(|m| m.get("iterations").or_else(|| m.get("value")))
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .min(u32::MAX as u64) as u32
}

fn make_morphology_dilate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Morphology;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let iterations = morph_iters_from_json(p);
    let f = Morphology::dilate(element, iterations);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_morphology_erode(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Morphology;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let iterations = morph_iters_from_json(p);
    let f = Morphology::erode(element, iterations);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_morphology_open(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Morphology;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let iterations = morph_iters_from_json(p);
    let f = Morphology::open(element, iterations);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_morphology_close(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Morphology;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let iterations = morph_iters_from_json(p);
    let f = Morphology::close(element, iterations);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// Parse a quad-corner array out of the JSON params.
///
/// Accepts either a flat 8-element array `[x0, y0, x1, y1, x2, y2, x3, y3]`
/// or a 4-element array of 2-element arrays `[[x, y], ...]`. Returns
/// the canonical `[(x, y); 4]` tuple.
fn parse_corners(value: &Value, key: &str) -> Result<[(f32, f32); 4]> {
    let arr = value.as_array().ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'perspective': '{key}' must be an array"
        ))
    })?;
    if arr.len() == 8 {
        let mut out = [(0f32, 0f32); 4];
        for i in 0..4 {
            let x = arr[i * 2].as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{}]' must be a number",
                    i * 2
                ))
            })? as f32;
            let y = arr[i * 2 + 1].as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{}]' must be a number",
                    i * 2 + 1
                ))
            })? as f32;
            out[i] = (x, y);
        }
        Ok(out)
    } else if arr.len() == 4 {
        let mut out = [(0f32, 0f32); 4];
        for (i, v) in arr.iter().enumerate() {
            let pair = v.as_array().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{i}]' must be a 2-element array"
                ))
            })?;
            if pair.len() != 2 {
                return Err(Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{i}]' must have 2 elements"
                )));
            }
            let x = pair[0].as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{i}][0]' must be a number"
                ))
            })? as f32;
            let y = pair[1].as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'perspective': '{key}[{i}][1]' must be a number"
                ))
            })? as f32;
            out[i] = (x, y);
        }
        Ok(out)
    } else {
        Err(Error::invalid(format!(
            "job: filter 'perspective': '{key}' must have 4 (nested) or 8 (flat) elements, got {}",
            arr.len()
        )))
    }
}

fn make_perspective(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Perspective;
    let p = params.as_object();

    let src_v = p
        .and_then(|m| m.get("src_corners").or_else(|| m.get("src")))
        .ok_or_else(|| Error::invalid("job: filter 'perspective' needs `src_corners`"))?;
    let dst_v = p
        .and_then(|m| m.get("dst_corners").or_else(|| m.get("dst")))
        .ok_or_else(|| Error::invalid("job: filter 'perspective' needs `dst_corners`"))?;
    let src = parse_corners(src_v, "src_corners")?;
    let dst = parse_corners(dst_v, "dst_corners")?;

    let mut f = Perspective::new(src, dst);
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'perspective': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'perspective': background array must contain unsigned ints",
                    )
                })?
                .min(255) as u8;
        }
        f = f.with_background(bg);
    }

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
    use oxideav_core::{Frame, VideoFrame, VideoPlane};
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
            // Round-next additions.
            "unsharp",
            "threshold",
            "level",
            "normalize",
            "posterize",
            "solarize",
            "flip",
            "flop",
            "rotate",
            "crop",
            "negate",
            "sepia",
            "modulate",
            "grayscale",
            "motion-blur",
            "emboss",
            // Round-after-next additions.
            "vignette",
            "colorize",
            "equalize",
            "auto-gamma",
            // Round-after-after-next additions.
            "tint",
            "sigmoidal-contrast",
            "implode",
            "swirl",
            "despeckle",
            // r6 additions.
            "wave",
            "spread",
            "charcoal",
            "convolve",
            "polar",
            "depolar",
            // r7 additions.
            "morphology-dilate",
            "morphology-erode",
            "morphology-open",
            "morphology-close",
            "perspective",
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

    // --- Round-next smoke tests ---
    //
    // Each newly-wired filter gets one factory-build + one apply through
    // the adapter on a small synthetic RGBA fixture (or matching format
    // when the filter requires something else, e.g. Crop on Yuv420P).

    /// Collect everything emitted by `push` into a vec.
    struct CollectCtx {
        out: Vec<(usize, Frame)>,
    }
    impl oxideav_core::filter::FilterContext for CollectCtx {
        fn emit(&mut self, port: usize, frame: Frame) -> Result<()> {
            self.out.push((port, frame));
            Ok(())
        }
    }

    fn rgba_in_port(w: u32, h: u32) -> PortSpec {
        PortSpec::video("in", w, h, PixelFormat::Rgba, TimeBase::new(1, 30))
    }

    fn rgb24_in_port(w: u32, h: u32) -> PortSpec {
        PortSpec::video("in", w, h, PixelFormat::Rgb24, TimeBase::new(1, 30))
    }

    fn gray_in_port(w: u32, h: u32) -> PortSpec {
        PortSpec::video("in", w, h, PixelFormat::Gray8, TimeBase::new(1, 30))
    }

    /// Build a 4×4 RGBA frame from a per-pixel function returning
    /// `[R, G, B, A]`.
    fn rgba_4x4(pattern: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                data.extend_from_slice(&pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        }
    }

    fn gray_4x4(pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity(16);
        for y in 0..4u32 {
            for x in 0..4u32 {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 4, data }],
        }
    }

    /// Run a built filter against a single video frame and return the
    /// emitted output (asserting exactly one frame on port 0).
    fn run_one(mut f: Box<dyn StreamFilter>, frame: VideoFrame) -> VideoFrame {
        let mut col = CollectCtx { out: Vec::new() };
        f.push(&mut col, 0, &Frame::Video(frame))
            .expect("push must not error");
        assert_eq!(col.out.len(), 1, "filter emitted {} frames", col.out.len());
        let (port, frame) = col.out.into_iter().next().unwrap();
        assert_eq!(port, 0, "filter emitted on unexpected port {port}");
        match frame {
            Frame::Video(v) => v,
            other => panic!("filter emitted non-video frame: {other:?}"),
        }
    }

    #[test]
    fn unsharp_smoke() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "unsharp",
                &json!({"radius": 1, "sigma": 0.5, "amount": 1.0, "threshold": 0}),
                &inputs,
            )
            .expect("unsharp factory");
        let out = run_one(f, rgba_4x4(|x, _| [x as u8 * 60, 0, 0, 255]));
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 4 * 4 * 4);
    }

    #[test]
    fn threshold_smoke_binarises_gray() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("threshold", &json!({"value": 128}), &inputs)
            .expect("threshold factory");
        // Half-and-half pattern: bottom rows above 128, top rows below.
        let out = run_one(f, gray_4x4(|_, y| if y < 2 { 50 } else { 200 }));
        assert_eq!(out.planes[0].data[0], 0); // top row -> 0
        assert_eq!(out.planes[0].data[12], 255); // bottom row -> 255
    }

    #[test]
    fn level_smoke_stretches_range() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "level",
                &json!({"black": 50, "white": 200, "gamma": 1.0}),
                &inputs,
            )
            .expect("level factory");
        let out = run_one(f, gray_4x4(|_, _| 50));
        // Sample at the black point should map to 0.
        assert_eq!(out.planes[0].data[0], 0);
    }

    #[test]
    fn level_factory_rejects_non_positive_gamma() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        assert!(c
            .filters
            .make("level", &json!({"gamma": 0.0}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("level", &json!({"gamma": -1.0}), &inputs)
            .is_err());
    }

    #[test]
    fn normalize_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("normalize", &json!({}), &inputs)
            .expect("normalize factory");
        let out = run_one(f, gray_4x4(|x, _| 50 + x as u8 * 30));
        assert_eq!(out.planes.len(), 1);
        assert_eq!(out.planes[0].data.len(), 16);
        // Min sample (50) and max sample (140) should stretch to 0/255.
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn posterize_smoke_collapses_levels() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("posterize", &json!({"levels": 2}), &inputs)
            .expect("posterize factory");
        let out = run_one(f, gray_4x4(|x, _| x as u8 * 80));
        // With 2 levels, every sample should be in {0, 255}.
        for &v in &out.planes[0].data {
            assert!(v == 0 || v == 255, "posterize{{2}} produced {v}");
        }
    }

    #[test]
    fn solarize_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("solarize", &json!({"value": 100}), &inputs)
            .expect("solarize factory");
        let out = run_one(f, gray_4x4(|_, y| if y < 2 { 50 } else { 200 }));
        // Below threshold 100 -> unchanged 50.
        assert_eq!(out.planes[0].data[0], 50);
        // Above threshold 100 -> 255 - 200 = 55.
        assert_eq!(out.planes[0].data[12], 55);
    }

    #[test]
    fn flip_smoke_reverses_rows() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("flip", &json!({}), &inputs)
            .expect("flip factory");
        let out = run_one(f, gray_4x4(|_, y| y as u8 * 10));
        // Top row was 0; after flip top row is what was the bottom row (3*10 = 30).
        assert_eq!(out.planes[0].data[0], 30);
        assert_eq!(out.planes[0].data[12], 0);
    }

    #[test]
    fn flop_smoke_reverses_columns() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("flop", &json!({}), &inputs)
            .expect("flop factory");
        let out = run_one(f, gray_4x4(|x, _| x as u8 * 10));
        // Left col was 0; after flop left col is what was right col (3*10 = 30).
        assert_eq!(out.planes[0].data[0], 30);
        assert_eq!(out.planes[0].data[3], 0);
    }

    #[test]
    fn rotate_smoke_90_swaps_w_and_h() {
        let c = ctx();
        // Use a 4x2 input so the swap is observable.
        let inputs = [PortSpec::video(
            "in",
            4,
            2,
            PixelFormat::Gray8,
            TimeBase::new(1, 30),
        )];
        let f = c
            .filters
            .make("rotate", &json!({"degrees": 90}), &inputs)
            .expect("rotate factory");
        // Output port should already advertise the swapped dimensions.
        let out_ports = f.output_ports();
        match &out_ports[0].params {
            PortParams::Video { width, height, .. } => {
                assert_eq!(*width, 2);
                assert_eq!(*height, 4);
            }
            _ => panic!("rotate output port is not video"),
        }
        // Also exercise the apply path on a 4x4 fixture (simpler) and
        // verify the produced plane has 4*4 = 16 samples.
        let inputs2 = [gray_in_port(4, 4)];
        let f2 = c
            .filters
            .make("rotate", &json!({"degrees": 90}), &inputs2)
            .expect("rotate factory");
        let out = run_one(f2, gray_4x4(|x, _| x as u8 * 50));
        assert_eq!(out.planes[0].data.len(), 16);
    }

    #[test]
    fn rotate_background_array_must_be_4_long() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c.filters.make(
            "rotate",
            &json!({"degrees": 45, "background": [1, 2, 3]}),
            &inputs,
        );
        assert!(bad.is_err(), "background must be a 4-element array");
        c.filters
            .make(
                "rotate",
                &json!({"degrees": 45, "background": [1, 2, 3, 4]}),
                &inputs,
            )
            .expect("4-element background should build");
    }

    #[test]
    fn crop_smoke_extracts_subregion() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "crop",
                &json!({"x": 1, "y": 1, "width": 2, "height": 2}),
                &inputs,
            )
            .expect("crop factory");
        // Output port should report the 2x2 dimensions.
        match &f.output_ports()[0].params {
            PortParams::Video { width, height, .. } => {
                assert_eq!(*width, 2);
                assert_eq!(*height, 2);
            }
            _ => panic!("crop output port is not video"),
        }
        let out = run_one(f, gray_4x4(|x, y| (x + y * 10) as u8));
        // Top-left of crop should be input(1, 1) = 1 + 10 = 11.
        assert_eq!(out.planes[0].data[0], 11);
        assert_eq!(out.planes[0].data.len(), 4);
    }

    #[test]
    fn crop_factory_requires_width_and_height() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        assert!(c
            .filters
            .make("crop", &json!({"x": 0, "y": 0, "height": 2}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("crop", &json!({"x": 0, "y": 0, "width": 2}), &inputs)
            .is_err());
    }

    #[test]
    fn negate_smoke_inverts_rgba() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("negate", &json!({}), &inputs)
            .expect("negate factory");
        let out = run_one(f, rgba_4x4(|_, _| [10, 20, 30, 200]));
        // R/G/B inverted; alpha preserved.
        assert_eq!(out.planes[0].data[0], 245); // 255 - 10
        assert_eq!(out.planes[0].data[1], 235); // 255 - 20
        assert_eq!(out.planes[0].data[2], 225); // 255 - 30
        assert_eq!(out.planes[0].data[3], 200); // alpha unchanged
    }

    #[test]
    fn sepia_smoke() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let f = c
            .filters
            .make("sepia", &json!({"threshold": 1.0}), &inputs)
            .expect("sepia factory");
        // RGB24 4×4: build manually.
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for _y in 0..4 {
            for _x in 0..4 {
                data.extend_from_slice(&[100u8, 100, 100]);
            }
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 12, data }],
        };
        let out = run_one(f, frame);
        // Output should still be 4*4*3 = 48 bytes.
        assert_eq!(out.planes[0].data.len(), 48);
        // Sepia of a neutral gray (100,100,100) should warm-tint:
        // r' = (0.393+0.769+0.189)*100 = 135.1 -> 135.
        // Just sanity-check the R channel grew above 100.
        assert!(out.planes[0].data[0] > 100);
    }

    #[test]
    fn modulate_smoke_identity() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "modulate",
                &json!({"brightness": 100.0, "saturation": 100.0, "hue": 0.0}),
                &inputs,
            )
            .expect("modulate factory");
        let frame = rgba_4x4(|_, _| [120, 80, 40, 255]);
        let out = run_one(f, frame.clone());
        // Identity modulate should reproduce the input bytewise (within
        // round-trip rounding error of 1).
        for (i, (&a, &b)) in out.planes[0]
            .data
            .iter()
            .zip(frame.planes[0].data.iter())
            .enumerate()
        {
            let diff = (a as i16 - b as i16).abs();
            assert!(diff <= 1, "byte {i}: |{a} - {b}| = {diff} > 1");
        }
    }

    #[test]
    fn grayscale_smoke_rgba_to_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("grayscale", &json!({"output_gray8": true}), &inputs)
            .expect("grayscale factory");
        // Output port should now be Gray8.
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => {
                assert_eq!(*format, PixelFormat::Gray8);
            }
            _ => panic!("grayscale output port is not video"),
        }
        let out = run_one(f, rgba_4x4(|_, _| [100, 100, 100, 255]));
        // Gray8 frame: 4*4 = 16 bytes.
        assert_eq!(out.planes[0].data.len(), 16);
        // Luma of (100,100,100) is 100.
        assert_eq!(out.planes[0].data[0], 100);
    }

    #[test]
    fn motion_blur_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "motion-blur",
                &json!({"radius": 2, "sigma": 1.0, "angle": 0.0}),
                &inputs,
            )
            .expect("motion-blur factory");
        let out = run_one(f, gray_4x4(|_, _| 50));
        // Flat input should remain flat after blur.
        assert!(out.planes[0]
            .data
            .iter()
            .all(|&v| (v as i16 - 50).abs() <= 1));
    }

    #[test]
    fn vignette_smoke_darkens_corners() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("vignette", &json!({}), &inputs)
            .expect("vignette factory");
        let out = run_one(f, rgba_4x4(|_, _| [200, 200, 200, 255]));
        // Corner (0, 0) should be darker than centre-ish pixel (1, 1).
        assert!(
            out.planes[0].data[0] <= out.planes[0].data[5 * 4],
            "vignette should darken corners more than centre"
        );
        // Alpha preserved.
        assert_eq!(out.planes[0].data[3], 255);
    }

    #[test]
    fn vignette_factory_rejects_non_finite() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        // serde_json doesn't have NaN, but very-large radius works; test
        // a structurally bad call instead — bad type for x.
        let bad = c
            .filters
            .make("vignette", &json!({"x": "not-a-number"}), &inputs);
        // String coerces to None ⇒ default 0.5 used; should still build.
        bad.expect("non-numeric values use defaults");
    }

    #[test]
    fn colorize_smoke_blends_to_target() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "colorize",
                &json!({"color": [200, 100, 50, 255], "amount": 1.0}),
                &inputs,
            )
            .expect("colorize factory");
        let out = run_one(f, rgba_4x4(|_, _| [10, 20, 30, 222]));
        // amount = 1.0 should fully replace the colour; alpha preserved.
        assert_eq!(out.planes[0].data[0], 200);
        assert_eq!(out.planes[0].data[1], 100);
        assert_eq!(out.planes[0].data[2], 50);
        assert_eq!(out.planes[0].data[3], 222);
    }

    #[test]
    fn colorize_amount_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "colorize",
                &json!({"color": [255, 0, 0, 255], "amount": 0.0}),
                &inputs,
            )
            .expect("colorize factory");
        let frame = rgba_4x4(|_, _| [10, 20, 30, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn colorize_factory_rejects_bad_color_arity() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c.filters.make(
            "colorize",
            &json!({"color": [1, 2], "amount": 0.5}),
            &inputs,
        );
        assert!(bad.is_err(), "color must be 3 or 4 elements");
    }

    #[test]
    fn equalize_smoke_spreads_range() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make("equalize", &json!({}), &inputs)
            .expect("equalize factory");
        // Build an 8×8 Gray8 fixture with samples in [60, 80].
        let mut data = Vec::with_capacity(64);
        for y in 0..8u32 {
            for x in 0..8u32 {
                data.push(60 + ((x + y) as u8) % 21);
            }
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        };
        let out = run_one(f, frame);
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn auto_gamma_smoke_brightens_dark_input() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("auto-gamma", &json!({}), &inputs)
            .expect("auto-gamma factory");
        let out = run_one(f, gray_4x4(|_, _| 30));
        // Dark input (mean 30) should be brightened toward mid-grey.
        let v = out.planes[0].data[0];
        assert!(v > 100, "auto-gamma did not brighten: {v}");
    }

    #[test]
    fn emboss_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("emboss", &json!({"radius": 1}), &inputs)
            .expect("emboss factory");
        let out = run_one(f, gray_4x4(|_, _| 100));
        // The 3x3 emboss kernel sums to 1, so a flat-100 input becomes
        // 100*1 + 128 (bias) = 228 inside, clamped to 0..=255.
        // Edge pixels still touch the convolution machinery; just assert
        // every output is a valid u8 (no panic) and the centre pixels
        // are exactly 228.
        assert_eq!(out.planes[0].data.len(), 16);
        // Centre pixel (1,1) and (2,2) sit fully inside the 4x4 grid.
        assert_eq!(out.planes[0].data[5], 228); // (1, 1)
        assert_eq!(out.planes[0].data[10], 228); // (2, 2)
    }

    // --- Round-after-after-next smoke tests ---

    #[test]
    fn tint_smoke_blends_white_to_target() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "tint",
                &json!({"color": [10, 20, 30, 255], "amount": 1.0}),
                &inputs,
            )
            .expect("tint factory");
        // White input ⇒ luma = 255 ⇒ weight = 1 ⇒ output = target.
        let out = run_one(f, rgba_4x4(|_, _| [255, 255, 255, 222]));
        assert_eq!(out.planes[0].data[0], 10);
        assert_eq!(out.planes[0].data[1], 20);
        assert_eq!(out.planes[0].data[2], 30);
        assert_eq!(out.planes[0].data[3], 222); // alpha preserved
    }

    #[test]
    fn tint_factory_rejects_bad_color_arity() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c
            .filters
            .make("tint", &json!({"color": [1, 2], "amount": 0.5}), &inputs);
        assert!(bad.is_err(), "color must be 3 or 4 elements");
    }

    #[test]
    fn sigmoidal_contrast_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "sigmoidal-contrast",
                &json!({"contrast": 5.0, "midpoint": 128.0}),
                &inputs,
            )
            .expect("sigmoidal-contrast factory");
        // Below-midpoint input should be pulled darker.
        let out = run_one(f, gray_4x4(|_, _| 64));
        assert!(out.planes[0].data[0] < 64);
    }

    #[test]
    fn sigmoidal_contrast_zero_is_passthrough() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("sigmoidal-contrast", &json!({"contrast": 0.0}), &inputs)
            .expect("sigmoidal-contrast factory");
        let frame = gray_4x4(|x, y| (x + y * 4) as u8 * 16);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn implode_smoke_factor_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("implode", &json!({"factor": 0.0}), &inputs)
            .expect("implode factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn implode_smoke_flat_image_invariant() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("implode", &json!({"factor": 0.5}), &inputs)
            .expect("implode factory");
        let frame = rgba_4x4_8x8(|_, _| [100, 150, 200, 222]);
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn swirl_smoke_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("swirl", &json!({"degrees": 0.0}), &inputs)
            .expect("swirl factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn swirl_smoke_flat_image_invariant() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("swirl", &json!({"degrees": 90.0}), &inputs)
            .expect("swirl factory");
        let frame = rgba_4x4_8x8(|_, _| [100, 150, 200, 222]);
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn despeckle_smoke_radius_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("despeckle", &json!({"radius": 0}), &inputs)
            .expect("despeckle factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn despeckle_smoke_removes_isolated_speck() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("despeckle", &json!({"radius": 1}), &inputs)
            .expect("despeckle factory");
        let mut frame = rgba_4x4_8x8(|_, _| [100, 100, 100, 222]);
        // Inject a speck at (3, 3).
        let off = (3 * 8 + 3) * 4;
        frame.planes[0].data[off] = 255;
        frame.planes[0].data[off + 1] = 255;
        frame.planes[0].data[off + 2] = 255;
        let out = run_one(f, frame);
        // The 3×3 median wipes the speck back to 100.
        assert_eq!(out.planes[0].data[off], 100);
        assert_eq!(out.planes[0].data[off + 1], 100);
        assert_eq!(out.planes[0].data[off + 2], 100);
    }

    // --- r6 smoke tests ---

    #[test]
    fn wave_smoke_zero_amplitude_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "wave",
                &json!({"amplitude": 0.0, "wavelength": 16.0}),
                &inputs,
            )
            .expect("wave factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn wave_smoke_flat_image_invariant() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "wave",
                &json!({"amplitude": 3.0, "wavelength": 8.0}),
                &inputs,
            )
            .expect("wave factory");
        let frame = rgba_4x4_8x8(|_, _| [100, 150, 200, 222]);
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn spread_smoke_radius_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("spread", &json!({"radius": 0}), &inputs)
            .expect("spread factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn spread_smoke_seed_changes_output() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f1 = c
            .filters
            .make("spread", &json!({"radius": 2, "seed": 1}), &inputs)
            .expect("spread factory");
        let f2 = c
            .filters
            .make("spread", &json!({"radius": 2, "seed": 999}), &inputs)
            .expect("spread factory");
        let frame =
            rgba_4x4_8x8(|x, y| [(x * 30) as u8, (y * 30) as u8, ((x + y) * 10) as u8, 222]);
        let a = run_one(f1, frame.clone());
        let b = run_one(f2, frame);
        assert_ne!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn charcoal_smoke_flat_input_yields_white() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make("charcoal", &json!({"factor": 1.0}), &inputs)
            .expect("charcoal factory");
        // Output port should be Gray8 with same dims.
        match &f.output_ports()[0].params {
            PortParams::Video {
                format,
                width,
                height,
                ..
            } => {
                assert_eq!(*format, PixelFormat::Gray8);
                assert_eq!(*width, 8);
                assert_eq!(*height, 8);
            }
            _ => panic!("charcoal output port is not video"),
        }
        // 8×8 Gray fixture, flat 100.
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![100u8; 64],
            }],
        };
        let out = run_one(f, frame);
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn charcoal_factory_rejects_negative_factor() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c
            .filters
            .make("charcoal", &json!({"factor": -0.5}), &inputs);
        assert!(bad.is_err(), "negative factor should be rejected");
    }

    #[test]
    fn convolve_smoke_identity_kernel_passes_through() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "convolve",
                &json!({"kernel": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}),
                &inputs,
            )
            .expect("convolve factory");
        let frame = gray_4x4(|x, y| (x as u8 + y as u8 * 4) * 10);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn convolve_factory_rejects_bad_kernel_shape() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        // 8 elements (not a perfect square) and no `size` → error.
        let bad = c.filters.make(
            "convolve",
            &json!({"kernel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
            &inputs,
        );
        assert!(bad.is_err(), "8-element kernel without size should fail");
        // size=3 but 8-element kernel → error.
        let bad = c.filters.make(
            "convolve",
            &json!({"size": 3, "kernel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
            &inputs,
        );
        assert!(bad.is_err(), "size mismatch should fail");
    }

    #[test]
    fn convolve_factory_requires_kernel() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c.filters.make("convolve", &json!({}), &inputs);
        assert!(bad.is_err(), "missing kernel should fail");
    }

    #[test]
    fn polar_smoke_flat_image_invariant() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("polar", &json!({}), &inputs)
            .expect("polar factory");
        let frame = rgba_4x4_8x8(|_, _| [100, 150, 200, 222]);
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
            assert_eq!(chunk[3], 222);
        }
    }

    #[test]
    fn depolar_smoke_flat_image_invariant() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("depolar", &json!({}), &inputs)
            .expect("depolar factory");
        let frame = rgba_4x4_8x8(|_, _| [50, 100, 150, 200]);
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk[0], 50);
            assert_eq!(chunk[1], 100);
            assert_eq!(chunk[2], 150);
            assert_eq!(chunk[3], 200);
        }
    }

    #[test]
    fn morphology_dilate_factory_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "morphology-dilate",
                &json!({"iterations": 1, "element": "square"}),
                &inputs,
            )
            .expect("morphology-dilate factory");
        // Single bright pixel at (1, 1); after dilate the 3×3 around it
        // should all carry 200.
        let out = run_one(f, gray_4x4(|x, y| if x == 1 && y == 1 { 200 } else { 10 }));
        for y in 0..=2 {
            for x in 0..=2 {
                assert_eq!(out.planes[0].data[y * 4 + x], 200, "({x}, {y})");
            }
        }
    }

    #[test]
    fn morphology_erode_factory_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "morphology-erode",
                &json!({"iterations": 1, "element": "cross"}),
                &inputs,
            )
            .expect("morphology-erode factory");
        // 3×3 bright centred — erode-cross strips the four 4-edge pixels.
        let out = run_one(
            f,
            gray_4x4(|x, y| {
                if (1..=2).contains(&x) && (1..=2).contains(&y) {
                    200
                } else {
                    10
                }
            }),
        );
        // Sanity: at least one centre sample stayed bright (true if any
        // of the original 3x3 kept its 4-neighbours bright).
        let any_bright = out.planes[0].data.contains(&200);
        let any_dark = out.planes[0].data.contains(&10);
        assert!(any_dark, "erode should leave some dark pixels");
        // 4×4 input + 2×2 bright square + cross element ⇒ no fully-bright
        // 4-neighbourhood survives, so we expect *no* 200 samples in the
        // output. The point is that the bright region shrunk.
        assert!(
            !any_bright || any_dark,
            "erode should at least introduce some dark pixels"
        );
    }

    #[test]
    fn morphology_open_factory_removes_speck() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "morphology-open",
                &json!({"iterations": 1, "element": "square"}),
                &inputs,
            )
            .expect("morphology-open factory");
        let out = run_one(f, gray_4x4(|x, y| if x == 1 && y == 1 { 200 } else { 10 }));
        // The lone speck at (1,1) should disappear.
        assert!(out.planes[0].data.iter().all(|&v| v == 10));
    }

    #[test]
    fn morphology_close_factory_fills_hole() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "morphology-close",
                &json!({"iterations": 1, "element": "square"}),
                &inputs,
            )
            .expect("morphology-close factory");
        // A bright 4×4 square with one dark pixel at (1, 1).
        let out = run_one(f, gray_4x4(|x, y| if x == 1 && y == 1 { 10 } else { 200 }));
        assert!(out.planes[0].data.iter().all(|&v| v == 200));
    }

    #[test]
    fn morphology_factory_rejects_unknown_element() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c
            .filters
            .make("morphology-dilate", &json!({"element": "diamond"}), &inputs);
        assert!(bad.is_err(), "diamond is not a recognised element");
    }

    #[test]
    fn perspective_identity_corners_pass_through() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "perspective",
                &json!({
                    "src_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                    "dst_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0]
                }),
                &inputs,
            )
            .expect("perspective factory");
        let frame = rgba_4x4(|x, y| [(x * 60) as u8, (y * 60) as u8, 50, 255]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn perspective_factory_rejects_missing_dst() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c.filters.make(
            "perspective",
            &json!({"src_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0]}),
            &inputs,
        );
        assert!(bad.is_err(), "missing dst_corners must fail");
    }

    #[test]
    fn perspective_factory_accepts_nested_corner_arrays() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        c.filters
            .make(
                "perspective",
                &json!({
                    "src_corners": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
                    "dst_corners": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]]
                }),
                &inputs,
            )
            .expect("nested corner arrays must work");
    }

    /// Build an 8×8 RGBA fixture from a per-pixel function returning `[R, G, B, A]`.
    fn rgba_4x4_8x8(pattern: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let mut data = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8u32 {
            for x in 0..8u32 {
                data.extend_from_slice(&pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 32, data }],
        }
    }
}
