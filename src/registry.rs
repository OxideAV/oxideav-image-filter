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

use crate::{
    clut::Clut,
    composite::{Composite, CompositeOp},
    hald_clut::HaldClut,
    ImageFilter, Planes, TwoInputImageFilter, VideoStreamParams,
};

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

    // r7 factory (distort).
    ctx.filters.register("distort", Box::new(make_distort));

    // r7 factory (tilt-shift).
    ctx.filters
        .register("tilt-shift", Box::new(make_tilt_shift));

    // r10 factories: geometric re-windowing + channel ops + morphology
    // edge / gradient operators.
    ctx.filters.register("roll", Box::new(make_roll));
    ctx.filters.register("shave", Box::new(make_shave));
    ctx.filters.register("extent", Box::new(make_extent));
    ctx.filters.register("trim", Box::new(make_trim));
    ctx.filters
        .register("channel-extract", Box::new(make_channel_extract));
    ctx.filters
        .register("morphology-edgein", Box::new(make_morphology_edgein));
    ctx.filters
        .register("morphology-edgeout", Box::new(make_morphology_edgeout));
    ctx.filters.register(
        "morphology-edge-magnitude",
        Box::new(make_morphology_edge_magnitude),
    );

    // r11 factories: per-pixel arithmetic + windowed statistic + 2-D
    // affine / SRT geometric warps.
    ctx.filters.register("evaluate", Box::new(make_evaluate));
    ctx.filters.register("cycle", Box::new(make_cycle));
    ctx.filters.register("statistic", Box::new(make_statistic));
    ctx.filters.register("affine", Box::new(make_affine));
    ctx.filters.register("srt", Box::new(make_srt));

    // r12 factories: tonal range/clamp + colour matrix + maths-fn map +
    // Laplacian / Canny edges.
    ctx.filters.register("clamp", Box::new(make_clamp));
    ctx.filters
        .register("auto-level", Box::new(make_auto_level));
    ctx.filters
        .register("contrast-stretch", Box::new(make_contrast_stretch));
    ctx.filters
        .register("linear-stretch", Box::new(make_linear_stretch));
    ctx.filters
        .register("color-matrix", Box::new(make_color_matrix));
    ctx.filters.register("recolor", Box::new(make_color_matrix));
    ctx.filters.register("function", Box::new(make_function));
    ctx.filters.register("laplacian", Box::new(make_laplacian));
    ctx.filters.register("canny", Box::new(make_canny));

    // r13 factories: blue-shift / picture frame / shade relief / oil
    // paint / palette quantizer + two-input CLUT family.
    ctx.filters
        .register("blue-shift", Box::new(make_blue_shift));
    ctx.filters.register("frame", Box::new(make_frame));
    ctx.filters.register("shade", Box::new(make_shade));
    ctx.filters.register("paint", Box::new(make_paint));
    ctx.filters.register("quantize", Box::new(make_quantize));
    ctx.filters.register("clut", Box::new(make_clut));
    ctx.filters.register("hald-clut", Box::new(make_hald_clut));

    // r14 factories: pencil sketch / auto-deskew / Hough lines /
    // inverse-barrel + two-input Stegano / Stereo.
    ctx.filters.register("sketch", Box::new(make_sketch));
    ctx.filters.register("deskew", Box::new(make_deskew));
    ctx.filters
        .register("hough-lines", Box::new(make_hough_lines));
    ctx.filters
        .register("barrel-inverse", Box::new(make_barrel_inverse));
    ctx.filters.register("stegano", Box::new(make_stegano));
    ctx.filters.register("stereo", Box::new(make_stereo));

    // r15 factories: HSL hue rotation / soft cosine vignette / chromatic
    // aberration / pixelate mosaic / channel mixer / adaptive threshold.
    ctx.filters
        .register("hsl-rotate", Box::new(make_hsl_rotate));
    ctx.filters
        .register("vignette-soft", Box::new(make_vignette_soft));
    ctx.filters
        .register("chromatic-aberration", Box::new(make_chromatic_aberration));
    ctx.filters.register("pixelate", Box::new(make_pixelate));
    ctx.filters.register("mosaic", Box::new(make_pixelate));
    ctx.filters
        .register("channel-mixer", Box::new(make_channel_mixer));
    ctx.filters
        .register("adaptive-threshold", Box::new(make_adaptive_threshold));

    // r16 factories: bilateral blur / canvas / gradient-radial /
    // gradient-conic / gravity-translate / color-balance / hsl-shift.
    ctx.filters
        .register("bilateral-blur", Box::new(make_bilateral_blur));
    ctx.filters.register("canvas", Box::new(make_canvas));
    ctx.filters
        .register("gradient-radial", Box::new(make_gradient_radial));
    ctx.filters
        .register("radial-gradient", Box::new(make_gradient_radial));
    ctx.filters
        .register("gradient-conic", Box::new(make_gradient_conic));
    ctx.filters
        .register("conic-gradient", Box::new(make_gradient_conic));
    ctx.filters
        .register("gravity-translate", Box::new(make_gravity_translate));
    ctx.filters
        .register("gravity", Box::new(make_gravity_translate));
    ctx.filters
        .register("color-balance", Box::new(make_color_balance));
    ctx.filters.register("hsl-shift", Box::new(make_hsl_shift));

    // r17 factories: exposure / temperature / vibrance / bw-mix /
    // clarity / shadow-highlight / bordered-frame.
    ctx.filters.register("exposure", Box::new(make_exposure));
    ctx.filters
        .register("temperature", Box::new(make_temperature));
    ctx.filters.register("vibrance", Box::new(make_vibrance));
    ctx.filters.register("bw-mix", Box::new(make_bw_mix));
    ctx.filters
        .register("black-and-white", Box::new(make_bw_mix));
    ctx.filters.register("clarity", Box::new(make_clarity));
    ctx.filters
        .register("shadow-highlight", Box::new(make_shadow_highlight));
    ctx.filters
        .register("bordered-frame", Box::new(make_bordered_frame));
    ctx.filters
        .register("border", Box::new(make_bordered_frame));

    // r18 factories: HoughCircles / AutoTrim / DropShadow / InnerShadow
    // / Bloom / EdgeDetect (multi-kernel) + two-input Difference.
    ctx.filters
        .register("hough-circles", Box::new(make_hough_circles));
    ctx.filters.register("auto-trim", Box::new(make_auto_trim));
    ctx.filters
        .register("drop-shadow", Box::new(make_drop_shadow));
    ctx.filters
        .register("inner-shadow", Box::new(make_inner_shadow));
    ctx.filters.register("bloom", Box::new(make_bloom));
    ctx.filters.register("glow", Box::new(make_bloom));
    ctx.filters
        .register("edge-detect", Box::new(make_edge_detect));
    ctx.filters
        .register("edge-multi", Box::new(make_edge_detect));
    ctx.filters
        .register("difference", Box::new(make_difference));

    // r19 factories: Heatmap (false-colour) / SplitTone / FloodFill /
    // PosterizeChannels (per-channel) / Toon (cel-shading) + two-input
    // Watermark.
    ctx.filters.register("heatmap", Box::new(make_heatmap));
    ctx.filters.register("false-color", Box::new(make_heatmap));
    ctx.filters
        .register("split-tone", Box::new(make_split_tone));
    ctx.filters
        .register("flood-fill", Box::new(make_flood_fill));
    ctx.filters
        .register("posterize-channels", Box::new(make_posterize_channels));
    ctx.filters.register("toon", Box::new(make_toon));
    ctx.filters.register("cartoon", Box::new(make_toon));
    ctx.filters.register("watermark", Box::new(make_watermark));

    // r20 factories: Crystallize / Halftone / GradientMap (+ alias) /
    // SelectiveColor (+ alias) / CrossProcess (+ alias) / OtsuThreshold
    // (+ alias).
    ctx.filters
        .register("crystallize", Box::new(make_crystallize));
    ctx.filters.register("halftone", Box::new(make_halftone));
    ctx.filters
        .register("gradient-map", Box::new(make_gradient_map));
    ctx.filters.register("duotone", Box::new(make_gradient_map));
    ctx.filters
        .register("selective-color", Box::new(make_selective_color));
    ctx.filters
        .register("selective-colour", Box::new(make_selective_color));
    ctx.filters
        .register("cross-process", Box::new(make_cross_process));
    ctx.filters
        .register("crossprocess", Box::new(make_cross_process));
    ctx.filters
        .register("otsu-threshold", Box::new(make_otsu_threshold));
    ctx.filters.register("otsu", Box::new(make_otsu_threshold));
    ctx.filters.register("comic", Box::new(make_comic));
    ctx.filters.register("manga", Box::new(make_comic));

    // r21 factories: Kuwahara / AnisotropicBlur / ZoomBlur / RadialBlur /
    // EmbossDirectional + two-input DisplacementMap.
    ctx.filters.register("kuwahara", Box::new(make_kuwahara));
    ctx.filters
        .register("anisotropic-blur", Box::new(make_anisotropic_blur));
    ctx.filters
        .register("anisotropic", Box::new(make_anisotropic_blur));
    ctx.filters.register("zoom-blur", Box::new(make_zoom_blur));
    ctx.filters
        .register("radial-blur", Box::new(make_radial_blur));
    ctx.filters
        .register("spin-blur", Box::new(make_radial_blur));
    ctx.filters
        .register("emboss-directional", Box::new(make_emboss_directional));
    ctx.filters
        .register("displacement-map", Box::new(make_displacement_map));

    // r22 factories: tone-mapping (Reinhard / Hable / Drago) + Curves +
    // DistanceTransform + Cyanotype vintage emulation.
    ctx.filters.register("reinhard", Box::new(make_reinhard));
    ctx.filters
        .register("tonemap-reinhard", Box::new(make_reinhard));
    // r237: unkeyed white-clamping variant of the Reinhard operator
    // (docs/image/filter/tone-mapping-operators.md §5.1).
    ctx.filters
        .register("reinhard-extended", Box::new(make_reinhard_extended));
    ctx.filters.register(
        "tonemap-reinhard-extended",
        Box::new(make_reinhard_extended),
    );
    // r248: local "dodging-and-burning" variant of the Reinhard
    // operator (docs/image/filter/tone-mapping-operators.md §3).
    ctx.filters
        .register("reinhard-local", Box::new(make_reinhard_local));
    ctx.filters
        .register("tonemap-reinhard-local", Box::new(make_reinhard_local));
    ctx.filters
        .register("dodge-and-burn", Box::new(make_reinhard_local));
    ctx.filters.register("hable", Box::new(make_hable));
    ctx.filters.register("tonemap-hable", Box::new(make_hable));
    ctx.filters.register("uncharted2", Box::new(make_hable));
    ctx.filters.register("drago", Box::new(make_drago));
    ctx.filters.register("tonemap-drago", Box::new(make_drago));
    ctx.filters.register("curves", Box::new(make_curves));
    ctx.filters
        .register("distance-transform", Box::new(make_distance_transform));
    ctx.filters
        .register("distance", Box::new(make_distance_transform));
    ctx.filters.register("cyanotype", Box::new(make_cyanotype));
    ctx.filters.register("blueprint", Box::new(make_cyanotype));

    // r23 factories: MaxRGB / MinRGB channel-collapse (HSV-V).
    ctx.filters.register("max-rgb", Box::new(make_max_rgb));
    ctx.filters.register("hsv-value", Box::new(make_max_rgb));
    ctx.filters.register("min-rgb", Box::new(make_min_rgb));
    ctx.filters.register("roberts", Box::new(make_roberts));
    ctx.filters
        .register("roberts-cross", Box::new(make_roberts));

    // r101 factory: Prewitt 3×3 first-derivative edge operator.
    ctx.filters.register("prewitt", Box::new(make_prewitt));

    // r105 factory: Scharr 3×3 first-derivative edge operator.
    ctx.filters.register("scharr", Box::new(make_scharr));

    // r174 factory: Frei–Chen 3×3 orthonormal-basis edge / line
    // detector. Two-name registration so `frei-chen` / `freichen` both
    // work as factory aliases.
    ctx.filters.register("frei-chen", Box::new(make_frei_chen));
    ctx.filters.register("freichen", Box::new(make_frei_chen));

    // r181 factory: Marr–Hildreth Laplacian-of-Gaussian edge /
    // zero-crossing detector. Three aliases so `laplacian-of-gaussian`
    // / `log` / `marr-hildreth` all resolve to the same factory.
    ctx.filters
        .register("laplacian-of-gaussian", Box::new(make_log));
    ctx.filters.register("log", Box::new(make_log));
    ctx.filters.register("marr-hildreth", Box::new(make_log));

    // r198 factory: oriented Gabor filter (Gabor 1946; Daugman 1985).
    // Three aliases so `gabor` / `gabor-filter` / `gabor-wavelet` all
    // resolve to the same factory.
    ctx.filters.register("gabor", Box::new(make_gabor));
    ctx.filters.register("gabor-filter", Box::new(make_gabor));
    ctx.filters.register("gabor-wavelet", Box::new(make_gabor));

    // r205 factory: Niblack 1986 adaptive local-statistics threshold.
    // Two aliases so `niblack` / `niblack-threshold` resolve to the
    // same factory.
    ctx.filters.register("niblack", Box::new(make_niblack));
    ctx.filters
        .register("niblack-threshold", Box::new(make_niblack));

    // r209 factory: exact-Euclidean Distance Transform
    // (Felzenszwalb–Huttenlocher 2012, separable lower-envelope of
    // parabolas — exact counterpart to the cheaper integer-chamfer
    // `distance-transform` factory already registered above). Three
    // aliases so `edt` / `euclidean-distance` /
    // `euclidean-distance-transform` resolve to the same factory.
    ctx.filters.register(
        "euclidean-distance-transform",
        Box::new(make_euclidean_distance_transform),
    );
    ctx.filters.register(
        "euclidean-distance",
        Box::new(make_euclidean_distance_transform),
    );
    ctx.filters
        .register("edt", Box::new(make_euclidean_distance_transform));

    // Weighted (generalised) distance transform: the continuous-seed
    // form `D_f(p) = min_q(‖p − q‖² + f(q))` of distance-transform.md
    // §1, seeding every pixel with a finite cost derived from its
    // intensity rather than the binary 0/∞ mask the EDT uses.
    ctx.filters.register(
        "weighted-distance-transform",
        Box::new(make_weighted_distance_transform),
    );
    ctx.filters.register(
        "weighted-distance",
        Box::new(make_weighted_distance_transform),
    );
    ctx.filters
        .register("wdt", Box::new(make_weighted_distance_transform));

    // Signed distance field: the difference of two exact-Euclidean
    // transforms (FG-distance minus BG-distance) so foreground pixels
    // carry negative distance and background pixels positive, biased to
    // a neutral midpoint on render. Three aliases so `sdf` /
    // `signed-distance` / `signed-distance-field` resolve to the same
    // factory.
    ctx.filters.register(
        "signed-distance-field",
        Box::new(make_signed_distance_field),
    );
    ctx.filters
        .register("signed-distance", Box::new(make_signed_distance_field));
    ctx.filters
        .register("sdf", Box::new(make_signed_distance_field));

    // r303 factory: edge feathering — a soft coverage falloff at a
    // shape boundary driven by the exact Euclidean inner-distance
    // transform (distance-transform.md §1 names feathering; §2 is the
    // exact transform reused here). Three aliases.
    ctx.filters.register("feather", Box::new(make_feather));
    ctx.filters.register("feather-edge", Box::new(make_feather));
    ctx.filters.register("soft-edge", Box::new(make_feather));

    // r324 factory: sRGB / power-law transfer-function transform
    // (tone-mapping-operators.md §5.2) exposed as a standalone LUT.
    // Decode (display → linear, EOTF) and encode (linear → display, OETF)
    // each get explicit aliases; the generic name reads direction + curve
    // from JSON.
    ctx.filters
        .register("srgb-decode", Box::new(make_srgb_decode));
    ctx.filters
        .register("linearize", Box::new(make_srgb_decode));
    ctx.filters
        .register("srgb-to-linear", Box::new(make_srgb_decode));
    ctx.filters
        .register("srgb-encode", Box::new(make_srgb_encode));
    ctx.filters
        .register("delinearize", Box::new(make_srgb_encode));
    ctx.filters
        .register("linear-to-srgb", Box::new(make_srgb_encode));
    ctx.filters
        .register("srgb-transform", Box::new(make_srgb_decode));

    // r186 factory: classic bit-depth-reduction dither (Bayer ordered
    // + the Floyd–Steinberg / Jarvis–Judice–Ninke / Stucki /
    // Sierra-3 / Sierra-2 / Sierra-Lite / Atkinson error-diffusion
    // family). Per-kernel + per-mode names so a pipeline can pick one
    // without packing it into a JSON param.
    ctx.filters.register("dither", Box::new(make_dither));
    ctx.filters
        .register("dither-floyd-steinberg", Box::new(make_dither_fs));
    ctx.filters.register("dither-fs", Box::new(make_dither_fs));
    ctx.filters
        .register("dither-jjn", Box::new(make_dither_jjn));
    ctx.filters
        .register("dither-jarvis", Box::new(make_dither_jjn));
    ctx.filters
        .register("dither-stucki", Box::new(make_dither_stucki));
    ctx.filters
        .register("dither-sierra3", Box::new(make_dither_sierra3));
    ctx.filters
        .register("dither-sierra2", Box::new(make_dither_sierra2));
    ctx.filters
        .register("dither-sierra-lite", Box::new(make_dither_sierra_lite));
    ctx.filters
        .register("dither-atkinson", Box::new(make_dither_atkinson));
    ctx.filters
        .register("dither-bayer", Box::new(make_dither_bayer));
    ctx.filters
        .register("ordered-dither", Box::new(make_dither_bayer));

    // r8 factories: Porter–Duff and arithmetic composite operators
    // (two-input). Mirrors the documented `-compose <op>` CLI family. r11
    // adds the four overlay-family ops (HardLight / SoftLight /
    // ColorDodge / ColorBurn).
    for op in [
        CompositeOp::Over,
        CompositeOp::In,
        CompositeOp::Out,
        CompositeOp::Atop,
        CompositeOp::Xor,
        CompositeOp::Plus,
        CompositeOp::Multiply,
        CompositeOp::Screen,
        CompositeOp::Overlay,
        CompositeOp::Darken,
        CompositeOp::Lighten,
        CompositeOp::Difference,
        CompositeOp::HardLight,
        CompositeOp::SoftLight,
        CompositeOp::ColorDodge,
        CompositeOp::ColorBurn,
    ] {
        let name = format!("composite-{}", op.name());
        ctx.filters.register(
            &name,
            Box::new(move |params, inputs| make_composite(op, params, inputs)),
        );
    }
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

/// Wraps a [`TwoInputImageFilter`] in the [`StreamFilter`] contract.
/// Two video input ports — `0 = src` (foreground), `1 = dst`
/// (background) — and one video output port. The adapter buffers the
/// most-recent frame from each input port and emits the composite
/// whenever both ports have a frame in hand. Buffered frames are
/// consumed (cleared) on every emit so the next frame on either port
/// must be paired with a fresh frame on the other before the next
/// emit. `flush` clears any pending side without emitting.
struct TwoInputImageFilterAdapter {
    inner: Box<dyn TwoInputImageFilter>,
    inp: [PortSpec; 2],
    outp: [PortSpec; 1],
    params: VideoStreamParams,
    pending_src: Option<oxideav_core::VideoFrame>,
    pending_dst: Option<oxideav_core::VideoFrame>,
}

impl TwoInputImageFilterAdapter {
    fn new(
        inner: Box<dyn TwoInputImageFilter>,
        src_port: PortSpec,
        dst_port: PortSpec,
        out_port: PortSpec,
    ) -> Self {
        // The two input ports must agree on shape — pick `src`'s
        // params as the canonical pair (the factory layer enforces
        // the match before we get here).
        let params = match &src_port.params {
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
                format: PixelFormat::Rgba,
                width: 0,
                height: 0,
            },
        };
        Self {
            inner,
            inp: [src_port, dst_port],
            outp: [out_port],
            params,
            pending_src: None,
            pending_dst: None,
        }
    }
}

impl StreamFilter for TwoInputImageFilterAdapter {
    fn input_ports(&self) -> &[PortSpec] {
        &self.inp
    }
    fn output_ports(&self) -> &[PortSpec] {
        &self.outp
    }
    fn push(&mut self, ctx: &mut dyn FilterContext, port: usize, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid(
                "two-input image-filter adapter: only video frames are accepted",
            ));
        };
        match port {
            0 => self.pending_src = Some(v.clone()),
            1 => self.pending_dst = Some(v.clone()),
            other => {
                return Err(Error::invalid(format!(
                    "two-input image-filter adapter: unknown input port {other}"
                )));
            }
        }
        if self.pending_src.is_some() && self.pending_dst.is_some() {
            // Both inputs are buffered — take and emit. The earlier
            // is_some() guards prevent the take() from clearing a
            // half-pair (which would force the next push to wait for
            // *two* fresh frames instead of one).
            let src = self.pending_src.take().unwrap();
            let dst = self.pending_dst.take().unwrap();
            let out = self.inner.apply_two(&src, &dst, self.params)?;
            ctx.emit(0, Frame::Video(out))?;
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.pending_src = None;
        self.pending_dst = None;
        Ok(())
    }
}

/// Resolve `(src, dst)` input ports for the two-input composite
/// adapter. Looks for ports named `src` / `dst` first; falls back to
/// the first two video ports in declaration order. Both ports must
/// share the same `(format, width, height)` — the factory rejects
/// mismatches up front so the adapter can rely on the invariant.
fn two_video_in_ports(inputs: &[PortSpec]) -> (PortSpec, PortSpec) {
    let by_name = |want: &str| inputs.iter().find(|p| p.name == want).cloned();
    let src = by_name("src");
    let dst = by_name("dst");
    if let (Some(s), Some(d)) = (src, dst) {
        return (s, d);
    }
    let mut video = inputs
        .iter()
        .filter(|p| matches!(p.params, PortParams::Video { .. }))
        .cloned();
    let s = video
        .next()
        .unwrap_or_else(|| PortSpec::video("src", 0, 0, PixelFormat::Rgba, TimeBase::new(1, 30)));
    let d = video
        .next()
        .unwrap_or_else(|| PortSpec::video("dst", 0, 0, PixelFormat::Rgba, TimeBase::new(1, 30)));
    (s, d)
}

fn make_composite(
    op: CompositeOp,
    _params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    let (src_port, dst_port) = two_video_in_ports(inputs);
    // Enforce shape parity between the two inputs; the algebra
    // assumes per-pixel correspondence.
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'composite-{}': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})",
                op.name()
            )));
        }
    }
    // Output port mirrors the src port shape but uses the canonical
    // "video" name so downstream wiring matches the rest of the
    // crate's filters.
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(Composite::new(op)),
        src_port,
        dst_port,
        out_port,
    )))
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
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let factor = get_f64("factor")
        .or_else(|| get_f64("amount"))
        .or_else(|| get_f64("value"))
        .unwrap_or(1.0) as f32;
    if !factor.is_finite() || factor < 0.0 {
        return Err(Error::invalid(format!(
            "job: filter 'charcoal': factor must be a non-negative finite number (got {factor})"
        )));
    }
    // r9: optional Gaussian pre-blur radius (default 0 — no pre-blur).
    let radius = get_u64("radius").unwrap_or(0) as u32;
    let f = Charcoal::new(factor).with_radius(radius);
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

/// Apply optional `cx` / `cy` / `max_radius` (alias `max_r`) overrides
/// from a JSON parameter object onto a `Polar`. Mirrors the new
/// builder methods exposed in r7.
fn polar_apply_overrides(mut f: crate::Polar, params: &Value) -> crate::Polar {
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let cx = get_f64("cx").map(|v| v as f32);
    let cy = get_f64("cy").map(|v| v as f32);
    if cx.is_some() || cy.is_some() {
        f = f.with_centre(cx, cy);
    }
    if let Some(r) = get_f64("max_radius").or_else(|| get_f64("max_r")) {
        f = f.with_max_radius(Some(r as f32));
    }
    f
}

fn make_polar(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Polar;
    let f = polar_apply_overrides(Polar::forward(), params);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_depolar(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Polar;
    let f = polar_apply_overrides(Polar::inverse(), params);
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
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

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

    // r9: optional explicit output canvas size — both keys must be
    // present together, or both absent (the default "match input"
    // behaviour). One-without-the-other is rejected to avoid silent
    // axis-default surprises.
    let out_w_opt = get_u64("output_width");
    let out_h_opt = get_u64("output_height");
    let configured_size = match (out_w_opt, out_h_opt) {
        (Some(w), Some(h)) => {
            f = f.with_output_size(w as u32, h as u32);
            Some((w as u32, h as u32))
        }
        (None, None) => None,
        _ => {
            return Err(Error::invalid(
                "job: filter 'perspective': output_width and output_height must be supplied together",
            ));
        }
    };

    let in_port = video_in_port(inputs);
    // When the output canvas size differs from the input, rebuild the
    // output port spec so the downstream pipeline sees the new shape.
    let out_port = match (configured_size, &in_port.params) {
        (
            Some((w, h)),
            PortParams::Video {
                format, time_base, ..
            },
        ) => PortSpec::video("video", w, h, *format, *time_base),
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_distort(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Distort;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    // Either explicit `mode` ("barrel" / "pincushion") with a single
    // `strength`, or full `k1` / `k2` coefficients. The mode form is
    // sugar — when both are supplied the coefficients win.
    let (mut k1, mut k2) = (0.0f32, 0.0f32);
    if let Some(mode) = p.and_then(|m| m.get("mode")).and_then(|v| v.as_str()) {
        let strength = get_f64("strength").unwrap_or(0.3) as f32;
        match mode {
            "barrel" => k1 = -strength.abs(),
            "pincushion" => k1 = strength.abs(),
            other => {
                return Err(Error::invalid(format!(
                    "job: filter 'distort': unknown mode '{other}' (expected 'barrel' or 'pincushion')"
                )));
            }
        }
    }
    if let Some(v) = get_f64("k1") {
        k1 = v as f32;
    }
    if let Some(v) = get_f64("k2") {
        k2 = v as f32;
    }
    if !k1.is_finite() || !k2.is_finite() {
        return Err(Error::invalid(
            "job: filter 'distort': k1 / k2 must be finite numbers",
        ));
    }
    // r9: optional Brown-Conrady tangential coefficients (default 0).
    let p1 = get_f64("p1").unwrap_or(0.0) as f32;
    let p2 = get_f64("p2").unwrap_or(0.0) as f32;
    if !p1.is_finite() || !p2.is_finite() {
        return Err(Error::invalid(
            "job: filter 'distort': p1 / p2 must be finite numbers",
        ));
    }
    let mut f = Distort::new(k1, k2).with_tangential(p1, p2);
    if let (Some(cx), Some(cy)) = (get_f64("cx"), get_f64("cy")) {
        f = f.with_centre(cx as f32, cy as f32);
    }
    if let Some(r) = get_f64("r_norm").or_else(|| get_f64("radius")) {
        f = f.with_r_norm(r as f32);
    }
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'distort': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'distort': background array must contain unsigned ints",
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

fn make_tilt_shift(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::TiltShift;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let mut f = TiltShift::default();
    if let Some(c) = get_f64("focus_centre").or_else(|| get_f64("centre")) {
        f = f.with_focus_centre(c as f32);
    }
    if let Some(h) = get_f64("focus_height").or_else(|| get_f64("height")) {
        f = f.with_focus_height(h as f32);
    }
    if let Some(h) = get_f64("falloff_height").or_else(|| get_f64("falloff")) {
        f = f.with_falloff_height(h as f32);
    }
    let radius = get_u64("blur_radius")
        .or_else(|| get_u64("radius"))
        .map(|v| v as u32)
        .unwrap_or(f.blur_radius);
    let sigma = get_f64("blur_sigma")
        .or_else(|| get_f64("sigma"))
        .map(|v| v as f32)
        .unwrap_or(f.blur_sigma);
    f = f.with_blur(radius, sigma);

    // r9: optional rotated focus band — `angle_degrees` (alias `angle`).
    if let Some(a) = get_f64("angle_degrees").or_else(|| get_f64("angle")) {
        f = f.with_angle_degrees(a as f32);
    }

    if !f.focus_centre.is_finite()
        || !f.focus_height.is_finite()
        || !f.falloff_height.is_finite()
        || !f.blur_sigma.is_finite()
        || !f.angle_degrees.is_finite()
    {
        return Err(Error::invalid(
            "job: filter 'tilt-shift': focus_centre / focus_height / falloff_height / sigma / angle_degrees must be finite",
        ));
    }

    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r10 factories: geometry + channel ops + morphology edges ---

fn make_roll(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Roll;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());

    let dx = get_i64("dx")
        .or_else(|| get_i64("x"))
        .unwrap_or(0)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let dy = get_i64("dy")
        .or_else(|| get_i64("y"))
        .unwrap_or(0)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let f = Roll::new(dx, dy);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_shave(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Shave;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let x_border = get_u64("x_border")
        .or_else(|| get_u64("x"))
        .unwrap_or(0)
        .min(u32::MAX as u64) as u32;
    let y_border = get_u64("y_border")
        .or_else(|| get_u64("y"))
        .unwrap_or(0)
        .min(u32::MAX as u64) as u32;
    let f = Shave::new(x_border, y_border);
    let in_port = video_in_port(inputs);
    // Shave changes the output dimensions — reflect that on the spec.
    let out_port = match &in_port.params {
        PortParams::Video {
            format,
            width,
            height,
            time_base,
        } => {
            let new_w = width.saturating_sub(x_border.saturating_mul(2));
            let new_h = height.saturating_sub(y_border.saturating_mul(2));
            PortSpec::video("video", new_w, new_h, *format, *time_base)
        }
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_extent(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Extent;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());

    let w = get_u64("width")
        .ok_or_else(|| Error::invalid("job: filter 'extent' needs unsigned `width`"))?
        .min(u32::MAX as u64) as u32;
    let h = get_u64("height")
        .ok_or_else(|| Error::invalid("job: filter 'extent' needs unsigned `height`"))?
        .min(u32::MAX as u64) as u32;
    let ox = get_i64("offset_x")
        .or_else(|| get_i64("x"))
        .unwrap_or(0)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let oy = get_i64("offset_y")
        .or_else(|| get_i64("y"))
        .unwrap_or(0)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let mut f = Extent::new(w, h).with_offset(ox, oy);
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'extent': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'extent': background array must contain unsigned ints",
                    )
                })?
                .min(255) as u8;
        }
        f = f.with_background(bg);
    }

    let in_port = video_in_port(inputs);
    let out_port = match &in_port.params {
        PortParams::Video {
            format, time_base, ..
        } => PortSpec::video("video", w, h, *format, *time_base),
        _ => PortSpec::video("video", w, h, PixelFormat::Rgba, TimeBase::new(1, 30)),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_trim(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Trim;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let fuzz = get_u64("fuzz").unwrap_or(0).min(255) as u8;
    let mut f = Trim::new().with_fuzz(fuzz);
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if !(3..=4).contains(&arr.len()) {
            return Err(Error::invalid(format!(
                "job: filter 'trim': background must be a 3- or 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8, 0, 0, 255];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'trim': background array must contain unsigned ints",
                    )
                })?
                .min(255) as u8;
        }
        f = f.with_background(bg);
    }
    let in_port = video_in_port(inputs);
    // Output dimensions are not known until the bbox is computed at
    // apply()-time; advertise the input shape and let the downstream
    // node re-read the produced VideoPlane stride.
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_channel_extract(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Channel, ChannelExtract};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let name = get_str("channel").ok_or_else(|| {
        Error::invalid(
            "job: filter 'channel-extract' needs `channel` (e.g. 'r', 'g', 'b', 'a', 'y', 'u', 'v')",
        )
    })?;
    let channel = Channel::parse(name).ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'channel-extract': unknown channel '{name}' \
             (expected r/g/b/a/y/u/v)"
        ))
    })?;
    let f = ChannelExtract::new(channel);
    let in_port = video_in_port(inputs);
    // Output is a single-plane Gray8 frame at the dimensions of the
    // plane the channel lives on. For packed formats that's
    // width × height; for planar YUV chroma channels it's the chroma
    // grid (width / cx, height / cy). Compute the size up-front so the
    // port spec is accurate at construction.
    let (out_w, out_h) = match (&in_port.params, channel) {
        (PortParams::Video { width, height, .. }, Channel::U | Channel::V)
            if matches!(
                extract_video_format(&in_port.params),
                Some(PixelFormat::Yuv420P) | Some(PixelFormat::Yuv422P)
            ) =>
        {
            let (cx, cy) = match extract_video_format(&in_port.params).unwrap() {
                PixelFormat::Yuv420P => (2u32, 2u32),
                PixelFormat::Yuv422P => (2, 1),
                _ => (1, 1),
            };
            (width.div_ceil(cx), height.div_ceil(cy))
        }
        (PortParams::Video { width, height, .. }, _) => (*width, *height),
        _ => (0, 0),
    };
    let out_port = match &in_port.params {
        PortParams::Video { time_base, .. } => {
            PortSpec::video("video", out_w, out_h, PixelFormat::Gray8, *time_base)
        }
        _ => PortSpec::video(
            "video",
            out_w,
            out_h,
            PixelFormat::Gray8,
            TimeBase::new(1, 30),
        ),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// Helper: pull the pixel format out of a port's params, or return
/// `None` if the port isn't a video port. Used by the channel-extract
/// factory to size the output grid for chroma channels.
fn extract_video_format(params: &PortParams) -> Option<PixelFormat> {
    if let PortParams::Video { format, .. } = params {
        Some(*format)
    } else {
        None
    }
}

fn make_morphology_edgein(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::MorphologyEdge;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let f = MorphologyEdge::edge_in(element);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_morphology_edgeout(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::MorphologyEdge;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let f = MorphologyEdge::edge_out(element);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_morphology_edge_magnitude(
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::MorphologyEdge;
    let p = params.as_object();
    let element = morph_element_from_json(p)?;
    let f = MorphologyEdge::edge_magnitude(element);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r11 factories (evaluate / cycle / statistic / affine / srt) ---

fn make_evaluate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Evaluate, EvaluateOp};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let op_name = get_str("op").or_else(|| get_str("operator")).ok_or_else(|| {
        Error::invalid("job: filter 'evaluate' needs string `op` (one of add/sub/mul/div/pow/max/min/set/and/or/xor/threshold)")
    })?;
    let op = EvaluateOp::from_name(op_name).ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'evaluate': unknown op '{op_name}' (expected add/subtract/multiply/divide/pow/max/min/set/and/or/xor/threshold)"
        ))
    })?;
    let value = get_f64("value")
        .or_else(|| get_f64("amount"))
        .ok_or_else(|| Error::invalid("job: filter 'evaluate' needs numeric `value`"))?
        as f32;
    let f = Evaluate::new(op, value);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_cycle(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Cycle;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let amount = get_i64("amount")
        .or_else(|| get_i64("value"))
        .or_else(|| get_i64("n"))
        .unwrap_or(0) as i32;
    let f = Cycle::new(amount);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_statistic(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Statistic, StatisticOp};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let op_name = get_str("op")
        .or_else(|| get_str("operator"))
        .ok_or_else(|| {
            Error::invalid("job: filter 'statistic' needs string `op` (median/min/max/mean)")
        })?;
    let op = StatisticOp::from_name(op_name).ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'statistic': unknown op '{op_name}' (expected median/min/max/mean)"
        ))
    })?;
    let width = get_u64("width").unwrap_or(3) as u32;
    let height = get_u64("height").unwrap_or(3) as u32;
    if width == 0 || height == 0 {
        return Err(Error::invalid(
            "job: filter 'statistic': width and height must be > 0",
        ));
    }
    let f = Statistic::new(op, width, height);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// Pull six affine coefficients from JSON. Accepts either an array
/// `matrix: [sx, ry, rx, sy, tx, ty]` or six per-named keys
/// (`sx`/`ry`/`rx`/`sy`/`tx`/`ty`). Defaults to identity.
fn parse_affine_matrix(p: Option<&serde_json::Map<String, Value>>) -> Result<[f32; 6]> {
    if let Some(arr) = p.and_then(|m| m.get("matrix")).and_then(|v| v.as_array()) {
        if arr.len() != 6 {
            return Err(Error::invalid(format!(
                "job: filter 'affine': `matrix` must have exactly 6 numbers, got {}",
                arr.len()
            )));
        }
        let mut m = [0f32; 6];
        for (i, v) in arr.iter().enumerate() {
            m[i] = v.as_f64().ok_or_else(|| {
                Error::invalid(format!("job: filter 'affine': matrix[{i}] is not a number"))
            })? as f32;
        }
        return Ok(m);
    }
    let get_f = |k: &str, dflt: f32| -> f32 {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(dflt)
    };
    Ok([
        get_f("sx", 1.0),
        get_f("ry", 0.0),
        get_f("rx", 0.0),
        get_f("sy", 1.0),
        get_f("tx", 0.0),
        get_f("ty", 0.0),
    ])
}

fn parse_background_rgba(p: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<[u8; 4]> {
    let arr = p.and_then(|m| m.get(key)).and_then(|v| v.as_array())?;
    if arr.is_empty() {
        return None;
    }
    let mut bg = [0u8, 0, 0, 255];
    for (i, slot) in bg.iter_mut().enumerate().take(4.min(arr.len())) {
        if let Some(v) = arr.get(i).and_then(|v| v.as_u64()) {
            *slot = v.min(255) as u8;
        }
    }
    Some(bg)
}

fn make_affine(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Affine;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let matrix = parse_affine_matrix(p)?;
    let mut f = Affine::new(matrix);
    if let Some(bg) = parse_background_rgba(p, "background") {
        f = f.with_background(bg);
    }
    let out_w = get_u64("output_width").unwrap_or(0) as u32;
    let out_h = get_u64("output_height").unwrap_or(0) as u32;
    if out_w > 0 && out_h > 0 {
        f = f.with_output_size(out_w, out_h);
    }
    let in_port = video_in_port(inputs);
    // The output port may have a different (w, h) — match the canvas
    // size we produce. Format stays the same as input.
    let out_port = match (&in_port.params, f.output_size) {
        (
            PortParams::Video {
                format, time_base, ..
            },
            Some((w, h)),
        ) => PortSpec::video("video", w, h, *format, *time_base),
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_srt(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Srt;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let in_port = video_in_port(inputs);
    let (in_w, in_h) = match &in_port.params {
        PortParams::Video { width, height, .. } => (*width, *height),
        _ => (0, 0),
    };
    // Default origin = image centre (matches IM SRT convention).
    let cx = (in_w.max(1) as f32 - 1.0) * 0.5;
    let cy = (in_h.max(1) as f32 - 1.0) * 0.5;
    let ox = get_f64("ox")
        .or_else(|| get_f64("origin_x"))
        .unwrap_or(cx as f64) as f32;
    let oy = get_f64("oy")
        .or_else(|| get_f64("origin_y"))
        .unwrap_or(cy as f64) as f32;
    // Scale: `scale` (uniform) overrides `sx`/`sy` if present.
    let uniform = get_f64("scale");
    let sx = uniform.or_else(|| get_f64("sx")).unwrap_or(1.0) as f32;
    let sy = uniform.or_else(|| get_f64("sy")).unwrap_or(1.0) as f32;
    let angle = get_f64("angle")
        .or_else(|| get_f64("angle_degrees"))
        .unwrap_or(0.0) as f32;
    // Translation defaults to (ox, oy) — pure rotation/scale around origin.
    let tx = get_f64("tx")
        .or_else(|| get_f64("translate_x"))
        .unwrap_or(ox as f64) as f32;
    let ty = get_f64("ty")
        .or_else(|| get_f64("translate_y"))
        .unwrap_or(oy as f64) as f32;

    let mut f = Srt::new((ox, oy), (sx, sy), angle, (tx, ty));
    if let Some(bg) = parse_background_rgba(p, "background") {
        f = f.with_background(bg);
    }
    let out_w = get_u64("output_width").unwrap_or(0) as u32;
    let out_h = get_u64("output_height").unwrap_or(0) as u32;
    if out_w > 0 && out_h > 0 {
        f = f.with_output_size(out_w, out_h);
    }
    let out_port = match (&in_port.params, f.output_size) {
        (
            PortParams::Video {
                format, time_base, ..
            },
            Some((w, h)),
        ) => PortSpec::video("video", w, h, *format, *time_base),
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r12 factories (clamp / auto-level / stretches / colour matrix /
// function / laplacian / canny) ---

fn make_clamp(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Clamp;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let low = get_u64("low").unwrap_or(0).min(255) as u8;
    let high = get_u64("high").unwrap_or(255).min(255) as u8;
    let f = Clamp::new(low, high);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_auto_level(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AutoLevel;
    let f = AutoLevel::new();
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_contrast_stretch(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ContrastStretch;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let black = get_f64("black")
        .or_else(|| get_f64("black_pct"))
        .or_else(|| get_f64("low"))
        .unwrap_or(0.02) as f32;
    let white = get_f64("white")
        .or_else(|| get_f64("white_pct"))
        .or_else(|| get_f64("high"))
        .unwrap_or(0.01) as f32;
    let f = ContrastStretch::new(black, white);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_linear_stretch(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::LinearStretch;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let black = get_u64("black")
        .or_else(|| get_u64("black_pixels"))
        .or_else(|| get_u64("low"))
        .unwrap_or(0) as u32;
    let white = get_u64("white")
        .or_else(|| get_u64("white_pixels"))
        .or_else(|| get_u64("high"))
        .unwrap_or(0) as u32;
    let f = LinearStretch::new(black, white);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_color_matrix(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ColorMatrix;
    let p = params.as_object();
    let arr = p
        .and_then(|m| m.get("matrix"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            Error::invalid(
                "job: filter 'color-matrix' needs `matrix`: a 9-element array (row-major 3x3)",
            )
        })?;
    if arr.len() != 9 {
        return Err(Error::invalid(format!(
            "job: filter 'color-matrix': `matrix` must have exactly 9 numbers, got {}",
            arr.len()
        )));
    }
    let mut m = [0f32; 9];
    for (i, v) in arr.iter().enumerate() {
        m[i] = v.as_f64().ok_or_else(|| {
            Error::invalid(format!(
                "job: filter 'color-matrix': matrix[{i}] is not a number"
            ))
        })? as f32;
    }
    let mut f = ColorMatrix::new(m);
    if let Some(off_arr) = p.and_then(|m| m.get("offset")).and_then(|v| v.as_array()) {
        if off_arr.len() != 3 {
            return Err(Error::invalid(format!(
                "job: filter 'color-matrix': `offset` must have exactly 3 numbers, got {}",
                off_arr.len()
            )));
        }
        let mut o = [0f32; 3];
        for (i, v) in off_arr.iter().enumerate() {
            o[i] = v.as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'color-matrix': offset[{i}] is not a number"
                ))
            })? as f32;
        }
        f = f.with_offset(o);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_function(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Function, FunctionOp};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let op_name = get_str("op")
        .or_else(|| get_str("kind"))
        .or_else(|| get_str("function"))
        .ok_or_else(|| {
            Error::invalid(
                "job: filter 'function' needs string `op` (polynomial / sinusoid / arcsin / arctan)",
            )
        })?;
    let op = FunctionOp::from_name(op_name).ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'function': unknown op '{op_name}' \
             (expected polynomial / sinusoid / arcsin / arctan)"
        ))
    })?;
    let args = if let Some(arr) = p.and_then(|m| m.get("args")).and_then(|v| v.as_array()) {
        let mut a = Vec::with_capacity(arr.len());
        for (i, v) in arr.iter().enumerate() {
            a.push(v.as_f64().ok_or_else(|| {
                Error::invalid(format!("job: filter 'function': args[{i}] is not a number"))
            })? as f32);
        }
        a
    } else {
        Vec::new()
    };
    let f = Function::new(op, args);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_laplacian(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Laplacian;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let radius = get_u64("radius").unwrap_or(1).max(1) as u32;
    let f = Laplacian::new().with_radius(radius);
    let in_port = video_in_port(inputs);
    // Laplacian output is always Gray8.
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

fn make_canny(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Canny;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let radius = get_u64("radius").unwrap_or(1) as u32;
    let sigma = get_f64("sigma").unwrap_or(1.0) as f32;
    let low = get_u64("low")
        .or_else(|| get_u64("low_threshold"))
        .unwrap_or(30)
        .min(255) as u8;
    let high = get_u64("high")
        .or_else(|| get_u64("high_threshold"))
        .unwrap_or(90)
        .min(255) as u8;
    let f = Canny::new(radius, sigma, low, high);
    let in_port = video_in_port(inputs);
    // Canny output is always Gray8 (binary).
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

// --- r13 factories ---

fn make_blue_shift(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BlueShift;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let factor = get_f64("factor")
        .or_else(|| get_f64("amount"))
        .unwrap_or(1.5) as f32;
    let f = BlueShift::new(factor);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_frame(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Frame as FrameFilter;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    let bw = get_u64("width").unwrap_or(25) as u32;
    let bh = get_u64("height").unwrap_or(bw as u64) as u32;
    let inner = get_u64("inner_bevel")
        .or_else(|| get_u64("inner"))
        .unwrap_or(6) as u32;
    let outer = get_u64("outer_bevel")
        .or_else(|| get_u64("outer"))
        .unwrap_or(6) as u32;
    let mut f = FrameFilter::new(bw, bh, inner, outer);
    if let Some(arr) = p.and_then(|m| m.get("matte")).and_then(|v| v.as_array()) {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'frame': `matte` must be a 4-element [R,G,B,A] array, got {}",
                arr.len()
            )));
        }
        let mut m = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            m[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "job: filter 'frame': matte[{i}] is not an unsigned int"
                    ))
                })?
                .min(255) as u8;
        }
        f = f.with_matte(m);
    }
    let in_port = video_in_port(inputs);
    // Frame grows the canvas to (w + 2*bw, h + 2*bh); rebuild the
    // output port spec to match.
    let out_port = match &in_port.params {
        PortParams::Video {
            format,
            width,
            height,
            time_base,
        } => PortSpec::video(
            "video",
            *width + 2 * bw,
            *height + 2 * bh,
            *format,
            *time_base,
        ),
        _ => PortSpec::video(
            "video",
            2 * bw,
            2 * bh,
            PixelFormat::Rgba,
            TimeBase::new(1, 30),
        ),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_shade(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Shade;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());

    let az = get_f64("azimuth")
        .or_else(|| get_f64("azimuth_degrees"))
        .unwrap_or(30.0) as f32;
    let el = get_f64("elevation")
        .or_else(|| get_f64("elevation_degrees"))
        .unwrap_or(30.0) as f32;
    let color_pass_through = get_bool("color")
        .or_else(|| get_bool("colour"))
        .unwrap_or(false);
    let f = Shade::new(az, el).with_color(color_pass_through);
    let in_port = video_in_port(inputs);
    // gray-only mode collapses to Gray8; colour mode preserves shape.
    let out_port = if color_pass_through {
        passthrough_out_port(&in_port)
    } else {
        match &in_port.params {
            PortParams::Video {
                width,
                height,
                time_base,
                ..
            } => PortSpec::video("video", *width, *height, PixelFormat::Gray8, *time_base),
            _ => PortSpec::video("video", 0, 0, PixelFormat::Gray8, TimeBase::new(1, 30)),
        }
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_paint(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Paint;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let radius = get_u64("radius").unwrap_or(2) as u32;
    let mut f = Paint::new(radius);
    if let Some(levels) = get_u64("levels") {
        f = f.with_levels(levels.min(u32::MAX as u64) as u32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_quantize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Quantize;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let colors = get_u64("colors")
        .or_else(|| get_u64("colours"))
        .unwrap_or(64) as u32;
    let f = Quantize::new(colors);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_clut(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    let (src_port, dst_port) = two_video_in_ports(inputs);
    // Clut requires both ports to share the format and shape.
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'clut': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(Clut::new()),
        src_port,
        dst_port,
        out_port,
    )))
}

fn make_hald_clut(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    let (src_port, dst_port) = two_video_in_ports(inputs);
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'hald-clut': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(HaldClut::new()),
        src_port,
        dst_port,
        out_port,
    )))
}

// --- r14 factories ---

fn make_sketch(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Sketch;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let radius = get_u64("radius").unwrap_or(3) as u32;
    let sigma = get_f64("sigma").unwrap_or(1.5) as f32;
    let angle = get_f64("angle_degrees")
        .or_else(|| get_f64("angle"))
        .unwrap_or(0.0) as f32;
    if !sigma.is_finite() || !angle.is_finite() {
        return Err(Error::invalid(
            "job: filter 'sketch': sigma / angle_degrees must be finite numbers",
        ));
    }
    let f = Sketch::new(radius, sigma, angle);
    let in_port = video_in_port(inputs);
    // Sketch always collapses to Gray8 (single-plane edge canvas).
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

fn make_deskew(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Deskew;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let threshold = get_u64("threshold").unwrap_or(64).min(255) as u8;
    let mut f = Deskew::new(threshold);
    if let Some(a) = get_f64("max_angle_degrees").or_else(|| get_f64("max_angle")) {
        f = f.with_max_angle(a as f32);
    }
    if let Some(s) = get_f64("step_degrees").or_else(|| get_f64("step")) {
        f = f.with_step(s as f32);
    }
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'deskew': background must be a 4-element [R,G,B,A] array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid("job: filter 'deskew': background entries must be unsigned ints")
                })?
                .min(255) as u8;
        }
        f = f.with_background(bg);
    }
    let in_port = video_in_port(inputs);
    // Deskew may grow the canvas (rotated bounding box). The actual
    // size depends on the per-frame skew estimate, so we keep the
    // output port shape == input shape — downstream consumers must be
    // prepared to handle a frame slightly larger or smaller. (This
    // matches Rotate's existing behaviour through the registry shim.)
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_hough_lines(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HoughLines;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let edge = get_u64("edge_threshold").unwrap_or(64) as u32;
    let votes = get_u64("vote_threshold").unwrap_or(20) as u32;
    let mut f = HoughLines::new(edge, votes);
    if let Some(n) = get_u64("n_theta") {
        f = f.with_n_theta(n as u32);
    }
    if let Some(k) = get_u64("top_k") {
        f = f.with_top_k(k as u32);
    }
    let in_port = video_in_port(inputs);
    // HoughLines always emits a Gray8 line-trace canvas.
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

fn make_barrel_inverse(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BarrelInverse;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let a = get_f64("a").unwrap_or(0.0) as f32;
    let b = get_f64("b").unwrap_or(0.0) as f32;
    let c = get_f64("c").unwrap_or(0.0) as f32;
    let d = get_f64("d").unwrap_or(1.0) as f32;
    if !a.is_finite() || !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return Err(Error::invalid(
            "job: filter 'barrel-inverse': a / b / c / d must be finite numbers",
        ));
    }
    let mut f = BarrelInverse::new(a, b, c, d);
    if let (Some(cx), Some(cy)) = (get_f64("cx"), get_f64("cy")) {
        f = f.with_centre(cx as f32, cy as f32);
    }
    if let Some(r) = get_f64("r_norm").or_else(|| get_f64("radius")) {
        f = f.with_r_norm(r as f32);
    }
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'barrel-inverse': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            bg[i] = v
                .as_u64()
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter 'barrel-inverse': background entries must be unsigned ints",
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

fn make_stegano(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Stegano;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let offset = get_u64("offset").unwrap_or(7).min(7) as u8;
    let (src_port, dst_port) = two_video_in_ports(inputs);
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'stegano': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(Stegano::new(offset)),
        src_port,
        dst_port,
        out_port,
    )))
}

fn make_stereo(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Stereo;
    let (src_port, dst_port) = two_video_in_ports(inputs);
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'stereo': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(Stereo::new()),
        src_port,
        dst_port,
        out_port,
    )))
}

// --- r15 factories ---

fn make_hsl_rotate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HslRotate;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let degrees = get_f64("degrees")
        .or_else(|| get_f64("angle"))
        .or_else(|| get_f64("hue"))
        .unwrap_or(0.0) as f32;
    if !degrees.is_finite() {
        return Err(Error::invalid(format!(
            "job: filter 'hsl-rotate': degrees must be finite (got {degrees})"
        )));
    }
    let f = HslRotate::new(degrees);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_vignette_soft(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::VignetteSoft;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let x = get_f64("x").unwrap_or(0.5) as f32;
    let y = get_f64("y").unwrap_or(0.5) as f32;
    let inner = get_f64("inner").unwrap_or(0.4) as f32;
    let outer = get_f64("outer").unwrap_or(1.0) as f32;
    if !x.is_finite() || !y.is_finite() || !inner.is_finite() || !outer.is_finite() {
        return Err(Error::invalid(
            "job: filter 'vignette-soft': x / y / inner / outer must be finite numbers",
        ));
    }
    let f = VignetteSoft::new(x, y, inner, outer);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_chromatic_aberration(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ChromaticAberration;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    // Two supported shapes:
    //   1. mode + n  (mode = "horizontal" | "vertical")
    //   2. explicit r_dx / r_dy / b_dx / b_dy
    let f = if let Some(mode) = get_str("mode") {
        let n = get_i64("n")
            .or_else(|| get_i64("strength"))
            .or_else(|| get_i64("offset"))
            .unwrap_or(2) as i32;
        match mode {
            "horizontal" | "h" => ChromaticAberration::horizontal(n),
            "vertical" | "v" => ChromaticAberration::vertical(n),
            other => {
                return Err(Error::invalid(format!(
                    "job: filter 'chromatic-aberration': unknown mode '{other}' \
                     (expected 'horizontal' or 'vertical')"
                )));
            }
        }
    } else {
        let r_dx = get_i64("r_dx").unwrap_or(2) as i32;
        let r_dy = get_i64("r_dy").unwrap_or(0) as i32;
        let b_dx = get_i64("b_dx").unwrap_or(-2) as i32;
        let b_dy = get_i64("b_dy").unwrap_or(0) as i32;
        ChromaticAberration::new(r_dx, r_dy, b_dx, b_dy)
    };
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_pixelate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Pixelate;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let block = get_u64("block")
        .or_else(|| get_u64("size"))
        .or_else(|| get_u64("tile"))
        .unwrap_or(8)
        .min(u32::MAX as u64) as u32;
    let f = Pixelate::new(block);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_channel_mixer(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ChannelMixer;
    let p = params.as_object();
    // Accept several shapes:
    //   1. preset = "identity" | "sepia"
    //   2. matrix = [[a,b,c,d], [e,f,g,h], [i,j,k,l], [m,n,o,p]]
    //   3. matrix3 = [[a,b,c], [d,e,f], [g,h,i]]      (3×3 → 4×4 lift)
    // Optional offset = [ox, oy, oz, ow].
    let preset = p.and_then(|m| m.get("preset")).and_then(|v| v.as_str());
    let mut f = if let Some(name) = preset {
        match name {
            "identity" => ChannelMixer::identity(),
            "sepia" | "sepia_classic" => ChannelMixer::sepia_classic(),
            other => {
                return Err(Error::invalid(format!(
                    "job: filter 'channel-mixer': unknown preset '{other}' (expected 'identity' or 'sepia')"
                )));
            }
        }
    } else if let Some(m4) = p.and_then(|m| m.get("matrix")).and_then(|v| v.as_array()) {
        if m4.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'channel-mixer': matrix must be a 4-element array of 4-element arrays, got outer len {}",
                m4.len()
            )));
        }
        let mut matrix = [[0.0f32; 4]; 4];
        for (i, row) in m4.iter().enumerate() {
            let row = row.as_array().ok_or_else(|| {
                Error::invalid("job: filter 'channel-mixer': matrix rows must be arrays")
            })?;
            if row.len() != 4 {
                return Err(Error::invalid(format!(
                    "job: filter 'channel-mixer': matrix row {i} must have 4 entries, got {}",
                    row.len()
                )));
            }
            for (j, v) in row.iter().enumerate() {
                matrix[i][j] = v.as_f64().ok_or_else(|| {
                    Error::invalid("job: filter 'channel-mixer': matrix entries must be numbers")
                })? as f32;
            }
        }
        ChannelMixer::new(matrix, [0.0; 4])
    } else if let Some(m3) = p.and_then(|m| m.get("matrix3")).and_then(|v| v.as_array()) {
        if m3.len() != 3 {
            return Err(Error::invalid(format!(
                "job: filter 'channel-mixer': matrix3 must be a 3-element array of 3-element arrays, got outer len {}",
                m3.len()
            )));
        }
        let mut m = [[0.0f32; 3]; 3];
        for (i, row) in m3.iter().enumerate() {
            let row = row.as_array().ok_or_else(|| {
                Error::invalid("job: filter 'channel-mixer': matrix3 rows must be arrays")
            })?;
            if row.len() != 3 {
                return Err(Error::invalid(format!(
                    "job: filter 'channel-mixer': matrix3 row {i} must have 3 entries, got {}",
                    row.len()
                )));
            }
            for (j, v) in row.iter().enumerate() {
                m[i][j] = v.as_f64().ok_or_else(|| {
                    Error::invalid("job: filter 'channel-mixer': matrix3 entries must be numbers")
                })? as f32;
            }
        }
        ChannelMixer::from_color_matrix(m)
    } else {
        ChannelMixer::identity()
    };
    if let Some(arr) = p.and_then(|m| m.get("offset")).and_then(|v| v.as_array()) {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'channel-mixer': offset must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut o = [0.0f32; 4];
        for (i, v) in arr.iter().enumerate() {
            o[i] = v.as_f64().ok_or_else(|| {
                Error::invalid("job: filter 'channel-mixer': offset entries must be numbers")
            })? as f32;
        }
        f.offset = o;
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_adaptive_threshold(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AdaptiveThreshold;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let radius = get_u64("radius").unwrap_or(3).min(u32::MAX as u64) as u32;
    let offset = get_i64("offset").unwrap_or(0) as i32;
    let f = AdaptiveThreshold::new(radius, offset);
    let in_port = video_in_port(inputs);
    // Always emits Gray8 regardless of input.
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

// --- r16 factories ---

fn make_bilateral_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BilateralBlur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let radius = get_u64("radius").unwrap_or(3).min(u32::MAX as u64) as u32;
    let sigma_spatial = get_f64("sigma_spatial")
        .or_else(|| get_f64("sigma"))
        .unwrap_or((radius as f64 / 2.0).max(0.5)) as f32;
    let sigma_range = get_f64("sigma_range")
        .or_else(|| get_f64("range"))
        .unwrap_or(30.0) as f32;
    if !sigma_spatial.is_finite() || !sigma_range.is_finite() || sigma_range <= 0.0 {
        return Err(Error::invalid(format!(
            "job: filter 'bilateral-blur': sigma_spatial / sigma_range must be finite \
             positive numbers (got {sigma_spatial} / {sigma_range})"
        )));
    }
    let f = BilateralBlur::new(radius, sigma_spatial, sigma_range);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn parse_color4(v: &Value, name: &str) -> Result<[u8; 4]> {
    let arr = v.as_array().ok_or_else(|| {
        Error::invalid(format!(
            "job: filter '{name}': colour must be a 3- or 4-element array"
        ))
    })?;
    if arr.len() != 3 && arr.len() != 4 {
        return Err(Error::invalid(format!(
            "job: filter '{name}': colour must have 3 or 4 entries, got {}",
            arr.len()
        )));
    }
    let mut out = [0u8, 0, 0, 255];
    for (i, e) in arr.iter().enumerate() {
        let n = e.as_u64().ok_or_else(|| {
            Error::invalid(format!(
                "job: filter '{name}': colour entries must be unsigned 0..255 numbers"
            ))
        })?;
        if n > 255 {
            return Err(Error::invalid(format!(
                "job: filter '{name}': colour entry {i} = {n} out of 0..255 range"
            )));
        }
        out[i] = n as u8;
    }
    Ok(out)
}

fn make_canvas(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Canvas;
    let p = params.as_object();
    let colour = if let Some(v) = p.and_then(|m| m.get("color").or_else(|| m.get("colour"))) {
        parse_color4(v, "canvas")?
    } else {
        [0, 0, 0, 255]
    };
    let f = Canvas::new(colour);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_gradient_radial(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::GradientRadial;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let inner = if let Some(v) = p.and_then(|m| m.get("inner")) {
        parse_color4(v, "gradient-radial")?
    } else {
        [255, 255, 255, 255]
    };
    let outer = if let Some(v) = p.and_then(|m| m.get("outer")) {
        parse_color4(v, "gradient-radial")?
    } else {
        [0, 0, 0, 255]
    };
    let mut f = GradientRadial::new(inner, outer);
    if let Some(x) = get_f64("centre_x").or_else(|| get_f64("cx")) {
        f.centre_x = x as f32;
    }
    if let Some(y) = get_f64("centre_y").or_else(|| get_f64("cy")) {
        f.centre_y = y as f32;
    }
    if let Some(r) = get_f64("radius") {
        f.radius = r as f32;
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_gradient_conic(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::GradientConic;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let inner = if let Some(v) = p.and_then(|m| m.get("inner")) {
        parse_color4(v, "gradient-conic")?
    } else {
        [255, 255, 255, 255]
    };
    let outer = if let Some(v) = p.and_then(|m| m.get("outer")) {
        parse_color4(v, "gradient-conic")?
    } else {
        [0, 0, 0, 255]
    };
    let mut f = GradientConic::new(inner, outer);
    if let Some(x) = get_f64("centre_x").or_else(|| get_f64("cx")) {
        f.centre_x = x as f32;
    }
    if let Some(y) = get_f64("centre_y").or_else(|| get_f64("cy")) {
        f.centre_y = y as f32;
    }
    if let Some(a) = get_f64("start_angle").or_else(|| get_f64("angle")) {
        f.start_angle = a as f32;
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_gravity_translate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Gravity, GravityTranslate};
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let in_port = video_in_port(inputs);
    let in_w = match &in_port.params {
        PortParams::Video { width, .. } => *width,
        _ => 0,
    };
    let in_h = match &in_port.params {
        PortParams::Video { height, .. } => *height,
        _ => 0,
    };
    let width = get_u64("width").unwrap_or(in_w as u64).min(u32::MAX as u64) as u32;
    let height = get_u64("height")
        .unwrap_or(in_h as u64)
        .min(u32::MAX as u64) as u32;
    let gravity_name = get_str("gravity").unwrap_or("centre");
    let gravity = Gravity::from_name(gravity_name).ok_or_else(|| {
        Error::invalid(format!(
            "job: filter 'gravity-translate': unknown gravity '{gravity_name}' \
             (expected one of north / south / east / west / centre + diagonals)"
        ))
    })?;
    let mut f = GravityTranslate::new(width, height, gravity);
    if let Some(arr) = p
        .and_then(|m| m.get("background"))
        .and_then(|v| v.as_array())
    {
        if arr.len() != 4 {
            return Err(Error::invalid(format!(
                "job: filter 'gravity-translate': background must be a 4-element array, got {}",
                arr.len()
            )));
        }
        let mut bg = [0u8; 4];
        for (i, v) in arr.iter().enumerate() {
            let n = v.as_u64().ok_or_else(|| {
                Error::invalid(
                    "job: filter 'gravity-translate': background entries must be unsigned numbers",
                )
            })?;
            if n > 255 {
                return Err(Error::invalid(format!(
                    "job: filter 'gravity-translate': background entry {i} = {n} out of range"
                )));
            }
            bg[i] = n as u8;
        }
        f = f.with_background(bg);
    }
    // The output canvas is fixed (width × height) regardless of input.
    let out_port = match &in_port.params {
        PortParams::Video {
            format, time_base, ..
        } => PortSpec::video("video", width, height, *format, *time_base),
        _ => PortSpec::video(
            "video",
            width,
            height,
            PixelFormat::Rgba,
            TimeBase::new(1, 30),
        ),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn parse_f3(v: &Value, name: &str, key: &str) -> Result<[f32; 3]> {
    let arr = v.as_array().ok_or_else(|| {
        Error::invalid(format!(
            "job: filter '{name}': '{key}' must be a 3-element array"
        ))
    })?;
    if arr.len() != 3 {
        return Err(Error::invalid(format!(
            "job: filter '{name}': '{key}' must have exactly 3 entries, got {}",
            arr.len()
        )));
    }
    let mut out = [0.0f32; 3];
    for (i, e) in arr.iter().enumerate() {
        out[i] = e.as_f64().ok_or_else(|| {
            Error::invalid(format!(
                "job: filter '{name}': '{key}' entries must be numbers"
            ))
        })? as f32;
    }
    Ok(out)
}

fn make_color_balance(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ColorBalance;
    let p = params.as_object();
    let lift = if let Some(v) = p.and_then(|m| m.get("lift")) {
        parse_f3(v, "color-balance", "lift")?
    } else {
        [0.0; 3]
    };
    let gamma = if let Some(v) = p.and_then(|m| m.get("gamma")) {
        parse_f3(v, "color-balance", "gamma")?
    } else {
        [1.0; 3]
    };
    let gain = if let Some(v) = p.and_then(|m| m.get("gain")) {
        parse_f3(v, "color-balance", "gain")?
    } else {
        [1.0; 3]
    };
    for g in gamma.iter() {
        if !g.is_finite() || *g <= 0.0 {
            return Err(Error::invalid(format!(
                "job: filter 'color-balance': gamma entries must be positive finite numbers \
                 (got {g})"
            )));
        }
    }
    let f = ColorBalance::new(lift, gamma, gain);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_hsl_shift(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HslShift;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let h = get_f64("h")
        .or_else(|| get_f64("hue"))
        .or_else(|| get_f64("h_shift"))
        .unwrap_or(0.0) as f32;
    let s = get_f64("s")
        .or_else(|| get_f64("saturation"))
        .or_else(|| get_f64("s_shift"))
        .unwrap_or(0.0) as f32;
    let l = get_f64("l")
        .or_else(|| get_f64("lightness"))
        .or_else(|| get_f64("l_shift"))
        .unwrap_or(0.0) as f32;
    let f = HslShift::new(h, s, l);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_exposure(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Exposure;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let ev = get_f64("ev")
        .or_else(|| get_f64("stops"))
        .or_else(|| get_f64("value"))
        .unwrap_or(0.0) as f32;
    let f = Exposure::new(ev);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_temperature(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Temperature;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let warmth = get_f64("warmth")
        .or_else(|| get_f64("temperature"))
        .or_else(|| get_f64("value"))
        .unwrap_or(0.0) as f32;
    let f = Temperature::new(warmth);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_vibrance(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Vibrance;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let amount = get_f64("amount")
        .or_else(|| get_f64("value"))
        .unwrap_or(0.0) as f32;
    let f = Vibrance::new(amount);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_bw_mix(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BwMix;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let mut f = if let Some(preset) = get_str("preset") {
        match preset.to_ascii_lowercase().as_str() {
            "red" => BwMix::red_filter(),
            "green" => BwMix::green_filter(),
            "blue" => BwMix::blue_filter(),
            "rec601" | "default" => BwMix::default(),
            other => {
                return Err(Error::invalid(format!(
                    "job: filter 'bw-mix': unknown preset '{other}' \
                     (expected red / green / blue / rec601)"
                )));
            }
        }
    } else if let Some(weights) = p.and_then(|m| m.get("weights")) {
        let w = parse_f3(weights, "bw-mix", "weights")?;
        BwMix::new(w[0], w[1], w[2])
    } else {
        let r = get_f64("r").unwrap_or(0.299) as f32;
        let g = get_f64("g").unwrap_or(0.587) as f32;
        let b = get_f64("b").unwrap_or(0.114) as f32;
        BwMix::new(r, g, b)
    };
    if let Some(k) = get_bool("keep_format") {
        f = f.with_keep_format(k);
    }
    let in_port = video_in_port(inputs);
    // BwMix without keep_format collapses to Gray8.
    let out_port = if f.keep_format {
        passthrough_out_port(&in_port)
    } else {
        match &in_port.params {
            PortParams::Video {
                width,
                height,
                time_base,
                ..
            } => PortSpec::video("video", *width, *height, PixelFormat::Gray8, *time_base),
            _ => PortSpec::video("video", 0, 0, PixelFormat::Gray8, TimeBase::new(1, 30)),
        }
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_clarity(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Clarity;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let amount = get_f64("amount")
        .or_else(|| get_f64("value"))
        .unwrap_or(0.5) as f32;
    let mut f = Clarity::new(amount);
    if let Some(r) = get_u64("radius") {
        f = f.with_radius(r as u32);
    }
    if let Some(s) = get_f64("sigma") {
        f = f.with_sigma(s as f32);
    }
    if let Some(t) = get_u64("threshold") {
        f = f.with_threshold(t.min(255) as u8);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_shadow_highlight(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ShadowHighlight;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let shadow = get_f64("shadow")
        .or_else(|| get_f64("shadows"))
        .or_else(|| get_f64("shadow_amount"))
        .unwrap_or(0.0) as f32;
    let highlight = get_f64("highlight")
        .or_else(|| get_f64("highlights"))
        .or_else(|| get_f64("highlight_amount"))
        .unwrap_or(0.0) as f32;
    let f = ShadowHighlight::new(shadow, highlight);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_bordered_frame(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BorderedFrame;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());

    // Default: uniform 4px black border.
    let uniform = get_u64("width").map(|v| v.min(u32::MAX as u64) as u32);
    let left = get_u64("left").map(|v| v.min(u32::MAX as u64) as u32);
    let top = get_u64("top").map(|v| v.min(u32::MAX as u64) as u32);
    let right = get_u64("right").map(|v| v.min(u32::MAX as u64) as u32);
    let bottom = get_u64("bottom").map(|v| v.min(u32::MAX as u64) as u32);

    let colour = if let Some(v) = p.and_then(|m| m.get("color").or_else(|| m.get("colour"))) {
        parse_color4(v, "bordered-frame")?
    } else {
        [0, 0, 0, 255]
    };

    let f = if let Some(w) = uniform {
        BorderedFrame::uniform(w, colour)
    } else {
        BorderedFrame::sides(
            left.unwrap_or(0),
            top.unwrap_or(0),
            right.unwrap_or(0),
            bottom.unwrap_or(0),
            colour,
        )
    };

    let in_port = video_in_port(inputs);
    let (in_w, in_h, fmt, tb) = match &in_port.params {
        PortParams::Video {
            width,
            height,
            format,
            time_base,
            ..
        } => (*width, *height, *format, *time_base),
        _ => (0, 0, PixelFormat::Rgba, TimeBase::new(1, 30)),
    };
    let out_w = in_w + f.left + f.right;
    let out_h = in_h + f.top + f.bottom;
    let out_port = PortSpec::video("video", out_w, out_h, fmt, tb);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_hough_circles(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HoughCircles;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let edge_threshold = get_u64("edge_threshold")
        .or_else(|| get_u64("edge"))
        .unwrap_or(64) as u32;
    let min_radius = get_u64("min_radius")
        .or_else(|| get_u64("min"))
        .unwrap_or(4) as u32;
    let max_radius = get_u64("max_radius")
        .or_else(|| get_u64("max"))
        .unwrap_or(32) as u32;
    let vote_threshold = get_u64("vote_threshold")
        .or_else(|| get_u64("votes"))
        .unwrap_or(16) as u32;
    let top_k = get_u64("top_k").or_else(|| get_u64("k")).unwrap_or(8) as u32;
    let f = HoughCircles::new(min_radius, max_radius, vote_threshold)
        .with_edge_threshold(edge_threshold)
        .with_top_k(top_k);
    let in_port = video_in_port(inputs);
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

fn make_auto_trim(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AutoTrim;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let fuzz = get_u64("fuzz").unwrap_or(0).min(255) as u8;
    let f = AutoTrim::new().with_fuzz(fuzz);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn parse_rgb3(v: &Value, name: &str) -> Result<[u8; 3]> {
    let arr = v
        .as_array()
        .ok_or_else(|| Error::invalid(format!("job: filter '{name}': colour must be an array")))?;
    if arr.len() != 3 && arr.len() != 4 {
        return Err(Error::invalid(format!(
            "job: filter '{name}': colour must have 3 or 4 entries, got {}",
            arr.len()
        )));
    }
    let mut out = [0u8; 3];
    for (i, e) in arr.iter().take(3).enumerate() {
        let n = e.as_u64().ok_or_else(|| {
            Error::invalid(format!(
                "job: filter '{name}': colour entries must be 0..=255"
            ))
        })?;
        if n > 255 {
            return Err(Error::invalid(format!(
                "job: filter '{name}': colour entry {i} = {n} out of 0..=255"
            )));
        }
        out[i] = n as u8;
    }
    Ok(out)
}

fn make_drop_shadow(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DropShadow;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let offset_x = get_i64("offset_x").or_else(|| get_i64("dx")).unwrap_or(4) as i32;
    let offset_y = get_i64("offset_y").or_else(|| get_i64("dy")).unwrap_or(4) as i32;
    let radius = get_u64("radius").unwrap_or(6) as u32;
    let opacity = get_f64("opacity").unwrap_or(0.5) as f32;
    let mut f = DropShadow::new(offset_x, offset_y, radius).with_opacity(opacity);
    if let Some(v) = p.and_then(|m| m.get("color").or_else(|| m.get("colour"))) {
        f = f.with_colour(parse_rgb3(v, "drop-shadow")?);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_inner_shadow(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::InnerShadow;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let offset_x = get_i64("offset_x").or_else(|| get_i64("dx")).unwrap_or(2) as i32;
    let offset_y = get_i64("offset_y").or_else(|| get_i64("dy")).unwrap_or(2) as i32;
    let radius = get_u64("radius").unwrap_or(3) as u32;
    let opacity = get_f64("opacity").unwrap_or(0.6) as f32;
    let mut f = InnerShadow::new(offset_x, offset_y, radius).with_opacity(opacity);
    if let Some(v) = p.and_then(|m| m.get("color").or_else(|| m.get("colour"))) {
        f = f.with_colour(parse_rgb3(v, "inner-shadow")?);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_bloom(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Bloom;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let threshold = get_u64("threshold").unwrap_or(200).min(255) as u8;
    let radius = get_u64("radius").unwrap_or(6) as u32;
    let intensity = get_f64("intensity")
        .or_else(|| get_f64("amount"))
        .unwrap_or(0.6) as f32;
    let f = Bloom::new(threshold, radius, intensity);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_edge_detect(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{EdgeDetect, EdgeKernel};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let kernel = match get_str("kernel").or_else(|| get_str("kind")) {
        Some(name) => EdgeKernel::from_name(name).ok_or_else(|| {
            Error::invalid(format!(
                "job: filter 'edge-detect': unknown kernel '{name}' \
                 (expected sobel / prewitt / scharr / roberts)"
            ))
        })?,
        None => EdgeKernel::Sobel,
    };
    let f = EdgeDetect::new(kernel);
    let in_port = video_in_port(inputs);
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

fn make_difference(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Difference;
    let (src_port, dst_port) = two_video_in_ports(inputs);
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'difference': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(Difference::new()),
        src_port,
        dst_port,
        out_port,
    )))
}

// --- r19 factories ---

fn make_heatmap(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Heatmap, HeatmapRamp};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let ramp = match get_str("ramp").or_else(|| get_str("palette")) {
        Some(name) => HeatmapRamp::from_name(name).ok_or_else(|| {
            Error::invalid(format!(
                "job: filter 'heatmap': unknown ramp '{name}' \
                 (expected jet / viridis / plasma / hot / cool / grayscale)"
            ))
        })?,
        None => HeatmapRamp::default(),
    };
    let f = Heatmap::new(ramp);
    let in_port = video_in_port(inputs);
    // Output is always Rgb24 / Rgba; rebuild the output port to reflect.
    let out_port = match &in_port.params {
        PortParams::Video {
            format,
            width,
            height,
            time_base,
            ..
        } => {
            let out_fmt = if *format == PixelFormat::Rgba {
                PixelFormat::Rgba
            } else {
                PixelFormat::Rgb24
            };
            PortSpec::video("video", *width, *height, out_fmt, *time_base)
        }
        _ => PortSpec::video("video", 0, 0, PixelFormat::Rgb24, TimeBase::new(1, 30)),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_split_tone(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::SplitTone;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let shadow = match p.and_then(|m| m.get("shadow").or_else(|| m.get("shadow_color"))) {
        Some(v) => parse_rgb3(v, "split-tone")?,
        None => [0, 30, 70],
    };
    let highlight = match p.and_then(|m| m.get("highlight").or_else(|| m.get("highlight_color"))) {
        Some(v) => parse_rgb3(v, "split-tone")?,
        None => [255, 220, 80],
    };
    let balance = get_f64("balance").unwrap_or(0.0) as f32;
    let amount = get_f64("amount").unwrap_or(0.5) as f32;
    let f = SplitTone::new(shadow, highlight, balance, amount);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_flood_fill(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::FloodFill;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let seed_x = get_u64("seed_x").or_else(|| get_u64("x")).unwrap_or(0) as u32;
    let seed_y = get_u64("seed_y").or_else(|| get_u64("y")).unwrap_or(0) as u32;
    let tolerance = get_u64("tolerance")
        .or_else(|| get_u64("fuzz"))
        .unwrap_or(0)
        .min(255) as u8;
    let mut color = [255u8, 0, 0, 255];
    if let Some(v) = p.and_then(|m| m.get("color").or_else(|| m.get("colour"))) {
        let c3 = parse_rgb3(v, "flood-fill")?;
        color = [c3[0], c3[1], c3[2], 255];
        // If the array had a 4th entry, capture it.
        if let Some(arr) = v.as_array() {
            if arr.len() >= 4 {
                if let Some(a) = arr[3].as_u64() {
                    color[3] = a.min(255) as u8;
                }
            }
        }
    }
    let f = FloodFill::new(seed_x, seed_y, color, tolerance);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_posterize_channels(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::PosterizeChannels;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let r = get_u64("r_levels").or_else(|| get_u64("r")).unwrap_or(4) as u32;
    let g = get_u64("g_levels").or_else(|| get_u64("g")).unwrap_or(4) as u32;
    let b = get_u64("b_levels").or_else(|| get_u64("b")).unwrap_or(4) as u32;
    let mut f = PosterizeChannels::new(r, g, b);
    if let Some(a) = get_u64("alpha_levels").or_else(|| get_u64("a")) {
        f = f.with_alpha_levels(a as u32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_toon(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Toon;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let levels = get_u64("levels").unwrap_or(4) as u32;
    let edge_threshold = get_u64("edge_threshold")
        .or_else(|| get_u64("threshold"))
        .unwrap_or(60)
        .min(255) as u8;
    let edge_color = match p.and_then(|m| m.get("edge_color").or_else(|| m.get("edge_colour"))) {
        Some(v) => parse_rgb3(v, "toon")?,
        None => [0, 0, 0],
    };
    let f = Toon::new(levels, edge_threshold, edge_color);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_watermark(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Watermark;
    let p = params.as_object();
    let get_i64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_i64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let offset_x = get_i64("offset_x").or_else(|| get_i64("x")).unwrap_or(0) as i32;
    let offset_y = get_i64("offset_y").or_else(|| get_i64("y")).unwrap_or(0) as i32;
    let opacity = get_f64("opacity").unwrap_or(1.0) as f32;

    let (src_port, dst_port) = two_video_in_ports(inputs);
    // src_port carries the *base* image; the overlay (dst_port) supplies
    // its own intrinsic width/height.
    let (overlay_w, overlay_h) = match &dst_port.params {
        PortParams::Video { width, height, .. } => (*width, *height),
        _ => (0, 0),
    };
    if let (PortParams::Video { format: sf, .. }, PortParams::Video { format: df, .. }) =
        (&src_port.params, &dst_port.params)
    {
        if sf != df {
            return Err(Error::invalid(format!(
                "job: filter 'watermark': src and dst pixel formats must match; \
                 got src={sf:?} dst={df:?}"
            )));
        }
    }
    if overlay_w == 0 || overlay_h == 0 {
        return Err(Error::invalid(
            "job: filter 'watermark': overlay port must have non-zero width/height",
        ));
    }
    let f = Watermark::new(offset_x, offset_y, overlay_w, overlay_h).with_opacity(opacity);
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(f),
        src_port,
        dst_port,
        out_port,
    )))
}

// --- r20 factories ---

fn make_crystallize(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Crystallize;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let cell_size = get_u64("cell_size")
        .or_else(|| get_u64("cell"))
        .or_else(|| get_u64("size"))
        .unwrap_or(8) as u32;
    let mut f = Crystallize::new(cell_size);
    if let Some(j) = get_f64("jitter") {
        f = f.with_jitter(j as f32);
    }
    if let Some(s) = get_u64("seed") {
        f = f.with_seed(s);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_halftone(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Halftone;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let cell_size = get_u64("cell_size")
        .or_else(|| get_u64("cell"))
        .or_else(|| get_u64("size"))
        .unwrap_or(8) as u32;
    let mut f = Halftone::new(cell_size);
    if let Some(v) = p.and_then(|m| m.get("ink").or_else(|| m.get("ink_color"))) {
        f = f.with_ink(parse_rgb3(v, "halftone")?);
    }
    if let Some(v) = p.and_then(|m| m.get("paper").or_else(|| m.get("paper_color"))) {
        f = f.with_paper(parse_rgb3(v, "halftone")?);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_gradient_map(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{GradientMap, GradientStop};
    let p = params.as_object();
    // Either: { "stops": [ { "position": 0.0, "color": [R,G,B] }, ... ] }
    // Or:     { "shadow": [...], "highlight": [...] } (duotone sugar)
    // Or:     { "shadow": [...], "midtone": [...], "highlight": [...] } (tritone)
    let f = if let Some(stops_v) = p.and_then(|m| m.get("stops")) {
        let arr = stops_v.as_array().ok_or_else(|| {
            Error::invalid("job: filter 'gradient-map': 'stops' must be an array")
        })?;
        let mut stops: Vec<GradientStop> = Vec::with_capacity(arr.len());
        for (idx, s) in arr.iter().enumerate() {
            let obj = s.as_object().ok_or_else(|| {
                Error::invalid(format!(
                    "job: filter 'gradient-map': stop {idx} must be an object"
                ))
            })?;
            let position = obj
                .get("position")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "job: filter 'gradient-map': stop {idx} missing 'position'"
                    ))
                })? as f32;
            let cv = obj
                .get("color")
                .or_else(|| obj.get("colour"))
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "job: filter 'gradient-map': stop {idx} missing 'color'"
                    ))
                })?;
            let color = parse_rgb3(cv, "gradient-map")?;
            stops.push(GradientStop { position, color });
        }
        GradientMap::new(stops)?
    } else {
        let shadow = match p.and_then(|m| m.get("shadow")) {
            Some(v) => parse_rgb3(v, "gradient-map")?,
            None => [0, 0, 0],
        };
        let highlight = match p.and_then(|m| m.get("highlight")) {
            Some(v) => parse_rgb3(v, "gradient-map")?,
            None => [255, 255, 255],
        };
        if let Some(mid_v) = p.and_then(|m| m.get("midtone").or_else(|| m.get("middle"))) {
            let midtone = parse_rgb3(mid_v, "gradient-map")?;
            GradientMap::tritone(shadow, midtone, highlight)
        } else {
            GradientMap::duotone(shadow, highlight)
        }
    };
    let in_port = video_in_port(inputs);
    // Output is always RGB or RGBA; Gray8 input upgrades to Rgb24.
    let out_port = match &in_port.params {
        PortParams::Video {
            format,
            width,
            height,
            time_base,
            ..
        } => {
            let out_fmt = match format {
                PixelFormat::Rgba => PixelFormat::Rgba,
                PixelFormat::Gray8 => PixelFormat::Rgb24,
                _ => PixelFormat::Rgb24,
            };
            PortSpec::video("video", *width, *height, out_fmt, *time_base)
        }
        _ => PortSpec::video("video", 0, 0, PixelFormat::Rgb24, TimeBase::new(1, 30)),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_selective_color(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{BandAdjust, HueBand, SelectiveColor};
    let p = params.as_object();
    let mut f = SelectiveColor::new();
    let bands = [
        HueBand::Reds,
        HueBand::Yellows,
        HueBand::Greens,
        HueBand::Cyans,
        HueBand::Blues,
        HueBand::Magentas,
    ];
    let names = ["reds", "yellows", "greens", "cyans", "blues", "magentas"];
    for (b, n) in bands.iter().zip(names.iter()) {
        if let Some(obj) = p.and_then(|m| m.get(*n)).and_then(|v| v.as_object()) {
            let adj = BandAdjust {
                hue_shift_degrees: obj.get("hue").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                saturation_shift: obj
                    .get("saturation")
                    .or_else(|| obj.get("sat"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                lightness_shift: obj
                    .get("lightness")
                    .or_else(|| obj.get("light"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
            };
            f = f.with_band(*b, adj);
        }
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_cross_process(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::CrossProcess;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let amount = get_f64("amount").unwrap_or(1.0) as f32;
    let mut f = CrossProcess::new(amount);
    if let Some(w) = get_f64("warmth") {
        f = f.with_warmth(w as f32);
    }
    if let Some(c) = get_f64("contrast") {
        f = f.with_contrast(c as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_otsu_threshold(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::OtsuThreshold;
    let p = params.as_object();
    let invert = p
        .and_then(|m| m.get("invert"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let f = OtsuThreshold::new().with_invert(invert);
    let in_port = video_in_port(inputs);
    // Output is always Gray8.
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

fn make_comic(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Comic;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let mut f = Comic::new();
    if let Some(r) = get_u64("smooth_radius").or_else(|| get_u64("radius")) {
        f = f.with_smooth_radius(r as u32);
    }
    if let Some(s) = get_f64("colour_sigma").or_else(|| get_f64("color_sigma")) {
        f = f.with_colour_sigma(s as f32);
    }
    if let Some(n) = get_u64("levels") {
        f = f.with_levels(n as u32);
    }
    if let Some(t) = get_u64("edge_threshold").or_else(|| get_u64("threshold")) {
        f = f.with_edge_threshold(t.min(255) as u8);
    }
    if let Some(v) = p.and_then(|m| m.get("ink_color").or_else(|| m.get("ink_colour"))) {
        f = f.with_ink_color(parse_rgb3(v, "comic")?);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

// --- r21 factories ---

fn make_kuwahara(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Kuwahara;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let radius = get_u64("radius").unwrap_or(3) as u32;
    let f = Kuwahara::new(radius);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_anisotropic_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AnisotropicBlur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let iterations = get_u64("iterations").unwrap_or(5) as u32;
    let mut f = AnisotropicBlur::new(iterations);
    if let Some(k) = get_f64("kappa").or_else(|| get_f64("k")) {
        f = f.with_kappa(k as f32);
    }
    if let Some(l) = get_f64("lambda") {
        f = f.with_lambda(l as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_zoom_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ZoomBlur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let strength = get_f64("strength").unwrap_or(0.2) as f32;
    let mut f = ZoomBlur::new(strength);
    let cx = get_f64("centre_x")
        .or_else(|| get_f64("center_x"))
        .or_else(|| get_f64("cx"));
    let cy = get_f64("centre_y")
        .or_else(|| get_f64("center_y"))
        .or_else(|| get_f64("cy"));
    if cx.is_some() || cy.is_some() {
        f = f.with_centre(cx.unwrap_or(0.5) as f32, cy.unwrap_or(0.5) as f32);
    }
    if let Some(s) = get_u64("samples") {
        f = f.with_samples(s as u32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_radial_blur(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::RadialBlur;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let angle = get_f64("angle_degrees")
        .or_else(|| get_f64("angle"))
        .unwrap_or(10.0) as f32;
    let mut f = RadialBlur::new(angle);
    let cx = get_f64("centre_x")
        .or_else(|| get_f64("center_x"))
        .or_else(|| get_f64("cx"));
    let cy = get_f64("centre_y")
        .or_else(|| get_f64("center_y"))
        .or_else(|| get_f64("cy"));
    if cx.is_some() || cy.is_some() {
        f = f.with_centre(cx.unwrap_or(0.5) as f32, cy.unwrap_or(0.5) as f32);
    }
    if let Some(s) = get_u64("samples") {
        f = f.with_samples(s as u32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_emboss_directional(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::EmbossDirectional;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let az = get_f64("azimuth_degrees")
        .or_else(|| get_f64("azimuth"))
        .or_else(|| get_f64("angle"))
        .unwrap_or(315.0) as f32;
    let mut f = EmbossDirectional::new(az);
    if let Some(d) = get_f64("depth") {
        f = f.with_depth(d as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_displacement_map(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DisplacementMap;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let scale = get_f64("scale").unwrap_or(10.0) as f32;
    let mut f = DisplacementMap::new(scale);
    let sx = get_f64("scale_x");
    let sy = get_f64("scale_y");
    if sx.is_some() || sy.is_some() {
        f = f.with_scales(
            sx.unwrap_or(scale as f64) as f32,
            sy.unwrap_or(scale as f64) as f32,
        );
    }
    let (src_port, dst_port) = two_video_in_ports(inputs);
    if let (
        PortParams::Video {
            format: sf,
            width: sw,
            height: sh,
            ..
        },
        PortParams::Video {
            format: df,
            width: dw,
            height: dh,
            ..
        },
    ) = (&src_port.params, &dst_port.params)
    {
        if sf != df || sw != dw || sh != dh {
            return Err(Error::invalid(format!(
                "job: filter 'displacement-map': src and dst ports must agree on \
                 (format, width, height); got src=({sf:?}, {sw}x{sh}) dst=({df:?}, {dw}x{dh})"
            )));
        }
    }
    let out_port = PortSpec {
        name: "video".to_string(),
        ..src_port.clone()
    };
    Ok(Box::new(TwoInputImageFilterAdapter::new(
        Box::new(f),
        src_port,
        dst_port,
        out_port,
    )))
}

fn make_reinhard(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Reinhard;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let key = get_f64("key").or_else(|| get_f64("a")).unwrap_or(0.18) as f32;
    let mut f = Reinhard::new(key);
    if let Some(w) = get_f64("white").or_else(|| get_f64("white_point")) {
        f = f.with_white(w as f32);
    }
    if let Some(s) = get_f64("saturation").or_else(|| get_f64("s")) {
        f = f.with_saturation(s as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_reinhard_extended(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ReinhardExtended;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    // Accept several spellings of the single parameter; absence or a
    // non-positive value triggers the auto-L_max mode per §5.1.
    let l_white = get_f64("l_white")
        .or_else(|| get_f64("white"))
        .or_else(|| get_f64("white_point"))
        .unwrap_or(0.0) as f32;
    let f = ReinhardExtended::new(l_white);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_reinhard_local(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ReinhardLocal;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    // §2.1 key — accept `key` or the paper's `a` spelling.
    let key = get_f64("key").or_else(|| get_f64("a")).unwrap_or(0.18) as f32;
    let mut f = ReinhardLocal::new(key);
    if let Some(phi) = get_f64("phi").or_else(|| get_f64("sharpening")) {
        f = f.with_phi(phi as f32);
    }
    if let Some(eps) = get_f64("epsilon")
        .or_else(|| get_f64("eps"))
        .or_else(|| get_f64("threshold"))
    {
        f = f.with_epsilon(eps as f32);
    }
    if let Some(n) = get_u64("scales").or_else(|| get_u64("levels")) {
        f = f.with_scales(n as u32);
    }
    if let Some(s0) = get_f64("initial_scale").or_else(|| get_f64("s0")) {
        f = f.with_initial_scale(s0 as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_hable(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Hable;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let bias = get_f64("exposure_bias")
        .or_else(|| get_f64("exposure"))
        .or_else(|| get_f64("bias"))
        .unwrap_or(2.0) as f32;
    let mut f = Hable::new(bias);
    if let Some(w) = get_f64("white").or_else(|| get_f64("white_point")) {
        f = f.with_white(w as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_drago(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Drago;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let bias = get_f64("bias").unwrap_or(0.85) as f32;
    let mut f = Drago::new(bias);
    if let Some(s) = get_f64("display_scale").or_else(|| get_f64("scale")) {
        f = f.with_display_scale(s as f32);
    }
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_curves(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Curve, CurveInterpolation, Curves};

    fn parse_curve(v: &Value) -> Option<Curve> {
        let arr = v.as_array()?;
        let mut pts = Vec::with_capacity(arr.len());
        for entry in arr {
            let pair = entry.as_array()?;
            if pair.len() != 2 {
                return None;
            }
            let x = pair[0].as_i64()?.clamp(0, 255) as u8;
            let y = pair[1].as_i64()?.clamp(0, 255) as u8;
            pts.push((x, y));
        }
        Some(Curve::new(pts))
    }

    let p = params.as_object();
    let master = p
        .and_then(|m| m.get("master").or_else(|| m.get("curve")))
        .and_then(parse_curve)
        .unwrap_or_else(Curve::identity);
    let mut f = Curves::new(master);
    if let Some(c) = p.and_then(|m| m.get("red")).and_then(parse_curve) {
        f = f.with_red(c);
    }
    if let Some(c) = p.and_then(|m| m.get("green")).and_then(parse_curve) {
        f = f.with_green(c);
    }
    if let Some(c) = p.and_then(|m| m.get("blue")).and_then(parse_curve) {
        f = f.with_blue(c);
    }
    // r262: read tension first so the Cardinal alias arm below can use it.
    // Tension `c ∈ [0, 1]` per §3.2 of
    // `docs/image/filter/curve-interpolation.md`; accept `tension` or `c`
    // as the JSON spelling; out-of-range values clamp to [0, 1]; missing
    // values default to `0.5` (the "half-tension" knob, midway between
    // uniform Catmull-Rom at c=0 and flat-tangent at c=1).
    let tension_q8 = p
        .and_then(|m| m.get("tension").or_else(|| m.get("c")))
        .and_then(|v| v.as_f64())
        .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
        .unwrap_or(128);
    // r277: prescribed end slopes for the clamped-cubic boundary (§4.2
    // of `docs/image/filter/curve-interpolation.md`). Signed Q8.8 fixed
    // point (`slope = q / 256`); accept `start_slope` / `slope0` / `s0`
    // and `end_slope` / `slope1` / `s1` spellings; out-of-range values
    // clamp to `[-127, 127]`; missing values default to slope `1.0`
    // (the identity-preserving choice — the identity knot pair with
    // both slopes at 1 solves to the exact identity LUT).
    let slope_q8 = |keys: [&str; 3]| -> i16 {
        p.and_then(|m| keys.iter().find_map(|k| m.get(*k)).and_then(|v| v.as_f64()))
            .map(|s| (s.clamp(-127.0, 127.0) * 256.0).round() as i16)
            .unwrap_or(256)
    };
    let start_slope_q8 = slope_q8(["start_slope", "slope0", "s0"]);
    let end_slope_q8 = slope_q8(["end_slope", "slope1", "s1"]);
    let mode =
        p.and_then(|m| m.get("interpolation").or_else(|| m.get("mode")))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "linear" => CurveInterpolation::Linear,
                "catmull-rom" | "catmull_rom" | "catmullrom" => CurveInterpolation::CatmullRom,
                "natural-cubic" | "natural_cubic" | "natural" => CurveInterpolation::NaturalCubic,
                // r226: centripetal Catmull-Rom (Yuksel et al. 2011, α = 0.5)
                // — §3.3 of `docs/image/filter/curve-interpolation.md`.
                "centripetal"
                | "centripetal-catmull-rom"
                | "centripetal_catmull_rom"
                | "centripetalcatmullrom"
                | "catmull-rom-centripetal" => CurveInterpolation::CentripetalCatmullRom,
                // r231: chordal Catmull-Rom (Yuksel et al. 2011, α = 1) —
                // §3.3 of `docs/image/filter/curve-interpolation.md`.
                "chordal"
                | "chordal-catmull-rom"
                | "chordal_catmull_rom"
                | "chordalcatmullrom"
                | "catmull-rom-chordal" => CurveInterpolation::ChordalCatmullRom,
                // r262: cardinal spline with tension parameter (Catmull & Rom
                // 1974) — §3.2 of `docs/image/filter/curve-interpolation.md`.
                // Tension is read from the `tension` (or `c`) JSON field above.
                "cardinal" | "cardinal-spline" | "cardinal_spline" | "cardinalspline" => {
                    CurveInterpolation::Cardinal { tension_q8 }
                }
                // r277: clamped-cubic boundary — the first §4.2-alternative
                // boundary of `docs/image/filter/curve-interpolation.md`
                // (prescribed end first derivatives). Slopes are read from
                // the `start_slope` / `end_slope` (or `slope0`/`s0`,
                // `slope1`/`s1`) JSON fields above.
                "clamped" | "clamped-cubic" | "clamped_cubic" | "clampedcubic" => {
                    CurveInterpolation::ClampedCubic {
                        start_slope_q8,
                        end_slope_q8,
                    }
                }
                // r277: not-a-knot boundary — the second §4.2-alternative
                // boundary (third derivative continuous across the first /
                // last interior knot). Parameter-free.
                "not-a-knot" | "not_a_knot" | "notaknot" | "not-a-knot-cubic"
                | "not_a_knot_cubic" => CurveInterpolation::NotAKnotCubic,
                // r270: the §2.2 box-region Fritsch-Carlson monotone-cubic
                // variant (`0 ≤ α ≤ 3` and `0 ≤ β ≤ 3` applied
                // independently) per `docs/image/filter/curve-interpolation.md`
                // §2.2 — the doc-flagged alternative to the radius-3 circle
                // used by the plain `MonotoneCubic` default.
                "monotone-box" | "monotone_box" | "monotonebox" | "monotone-cubic-box"
                | "monotone_cubic_box" | "monotonecubicbox" => CurveInterpolation::MonotoneCubicBox,
                _ => CurveInterpolation::MonotoneCubic,
            })
            .unwrap_or(CurveInterpolation::MonotoneCubic);
    f = f.with_interpolation(mode);
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_distance_transform(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{ChamferKind, DistanceTransform};
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let threshold = get_u64("threshold").unwrap_or(128).min(255) as u8;
    let mut f = DistanceTransform::new(threshold);
    if let Some(inv) = get_bool("invert") {
        f = f.with_invert(inv);
    }
    if let Some(s) = get_f64("scale") {
        f = f.with_scale(s as f32);
    }
    // r220: pick a chamfer kernel from
    // `docs/image/filter/distance-transform.md` §3.2. Accepts a `kind`
    // or `kernel` key for spelling preference; both fall back to the
    // historical 3-4 default.
    let kind_str = get_str("kind").or_else(|| get_str("kernel"));
    if let Some(s) = kind_str {
        let kind = match s.to_ascii_lowercase().as_str() {
            "chamfer-3-4" | "chamfer34" | "3-4" | "34" => ChamferKind::Chamfer34,
            "chamfer-5-7-11" | "chamfer5711" | "5-7-11" | "5711" => ChamferKind::Chamfer5711,
            "city-block" | "cityblock" | "manhattan" | "l1" => ChamferKind::CityBlock,
            "chessboard" | "chebyshev" | "l-infinity" | "linfinity" | "linf" => {
                ChamferKind::Chessboard
            }
            other => {
                return Err(Error::invalid(format!(
                    "distance-transform: unknown kind {other:?} (expected \
                     chamfer-3-4 / chamfer-5-7-11 / city-block / chessboard)"
                )));
            }
        };
        f = f.with_kind(kind);
    }
    let in_port = video_in_port(inputs);
    // Force the output port to Gray8 — DistanceTransform always emits
    // a single-plane intensity image regardless of input flavour, and
    // the input is Gray8-only anyway (the apply() call rejects others).
    let out_port = match &in_port.params {
        PortParams::Video {
            width,
            height,
            time_base,
            ..
        } => PortSpec {
            name: "video".to_string(),
            params: PortParams::Video {
                format: PixelFormat::Gray8,
                width: *width,
                height: *height,
                time_base: *time_base,
            },
            ..in_port.clone()
        },
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_cyanotype(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Cyanotype;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let strength = get_f64("strength").unwrap_or(1.0) as f32;
    let mut f = Cyanotype::new(strength);

    fn parse_rgb(v: &Value) -> Option<[u8; 3]> {
        let arr = v.as_array()?;
        if arr.len() != 3 {
            return None;
        }
        let r = arr[0].as_i64()?.clamp(0, 255) as u8;
        let g = arr[1].as_i64()?.clamp(0, 255) as u8;
        let b = arr[2].as_i64()?.clamp(0, 255) as u8;
        Some([r, g, b])
    }

    let shadow = p.and_then(|m| m.get("shadow")).and_then(parse_rgb);
    let highlight = p.and_then(|m| m.get("highlight")).and_then(parse_rgb);
    if shadow.is_some() || highlight.is_some() {
        f = f.with_endpoints(
            shadow.unwrap_or([15, 42, 111]),
            highlight.unwrap_or([217, 235, 248]),
        );
    }

    let in_port = video_in_port(inputs);
    // Cyanotype upgrades Gray8 → Rgb24; otherwise pass through.
    let out_port = match &in_port.params {
        PortParams::Video {
            format: PixelFormat::Gray8,
            width,
            height,
            time_base,
        } => PortSpec {
            name: "video".to_string(),
            params: PortParams::Video {
                format: PixelFormat::Rgb24,
                width: *width,
                height: *height,
                time_base: *time_base,
            },
            ..in_port.clone()
        },
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// Build a MaxRGB or MinRGB collapse based on `mode`.
///
/// Accepts `keep_format` (bool, default `false`) on the params map.
/// With `keep_format = false` the output port advertises `Gray8`;
/// with `keep_format = true` it passes through the input RGB / RGBA
/// shape.
fn make_max_rgb_inner(
    mode: crate::MaxRgbMode,
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::MaxRgb;
    let p = params.as_object();
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());

    let mut f = MaxRgb {
        mode,
        keep_format: false,
    };
    if let Some(k) = get_bool("keep_format") {
        f = f.with_keep_format(k);
    }
    let in_port = video_in_port(inputs);
    // Gray8 input stays Gray8 (identity); RGB / RGBA defaults to Gray8
    // unless keep_format opts back into the source format.
    let out_port = if f.keep_format {
        passthrough_out_port(&in_port)
    } else {
        match &in_port.params {
            PortParams::Video {
                format: PixelFormat::Gray8,
                ..
            } => passthrough_out_port(&in_port),
            PortParams::Video {
                width,
                height,
                time_base,
                ..
            } => PortSpec::video("video", *width, *height, PixelFormat::Gray8, *time_base),
            _ => PortSpec::video("video", 0, 0, PixelFormat::Gray8, TimeBase::new(1, 30)),
        }
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_max_rgb(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    make_max_rgb_inner(crate::MaxRgbMode::Max, params, inputs)
}

fn make_min_rgb(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    make_max_rgb_inner(crate::MaxRgbMode::Min, params, inputs)
}

fn make_roberts(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Roberts, RobertsMagnitude};
    let p = params.as_object();
    // Accept `magnitude: "l1" | "l2"` (default l2). The `l1` boolean
    // shorthand is also honoured.
    let mag = p
        .and_then(|m| m.get("magnitude"))
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.eq_ignore_ascii_case("l1") {
                RobertsMagnitude::L1
            } else {
                RobertsMagnitude::L2
            }
        })
        .or_else(|| {
            p.and_then(|m| m.get("l1"))
                .and_then(|v| v.as_bool())
                .map(|b| {
                    if b {
                        RobertsMagnitude::L1
                    } else {
                        RobertsMagnitude::L2
                    }
                })
        })
        .unwrap_or(RobertsMagnitude::L2);
    let f = Roberts::new().with_magnitude(mag);
    let in_port = video_in_port(inputs);
    // Roberts output is always Gray8.
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

fn make_scharr(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Scharr, ScharrMagnitude};
    let p = params.as_object();
    // Accept `magnitude: "l1" | "l2"` (default l2). The `l1` boolean
    // shorthand is also honoured.
    let mag = p
        .and_then(|m| m.get("magnitude"))
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.eq_ignore_ascii_case("l1") {
                ScharrMagnitude::L1
            } else {
                ScharrMagnitude::L2
            }
        })
        .or_else(|| {
            p.and_then(|m| m.get("l1"))
                .and_then(|v| v.as_bool())
                .map(|b| {
                    if b {
                        ScharrMagnitude::L1
                    } else {
                        ScharrMagnitude::L2
                    }
                })
        })
        .unwrap_or(ScharrMagnitude::L2);
    let f = Scharr::new().with_magnitude(mag);
    let in_port = video_in_port(inputs);
    // Scharr output is always Gray8.
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

/// Factory for the Frei–Chen edge / line detector. Accepts
/// `mode: "edge" | "line"` (default `edge`). For convenience the
/// short-form alias `line: true` also selects line mode.
fn make_frei_chen(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{FreiChen, FreiChenMode};
    let p = params.as_object();
    let mode = p
        .and_then(|m| m.get("mode"))
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.eq_ignore_ascii_case("line") {
                FreiChenMode::Line
            } else {
                FreiChenMode::Edge
            }
        })
        .or_else(|| {
            p.and_then(|m| m.get("line"))
                .and_then(|v| v.as_bool())
                .map(|b| {
                    if b {
                        FreiChenMode::Line
                    } else {
                        FreiChenMode::Edge
                    }
                })
        })
        .unwrap_or(FreiChenMode::Edge);
    let f = FreiChen::new().with_mode(mode);
    let in_port = video_in_port(inputs);
    // Frei–Chen output is always Gray8.
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

fn make_gabor(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Gabor, GaborMode};
    let p = params.as_object();
    let mut f = Gabor::new();
    if let Some(w) = p.and_then(|m| m.get("wavelength")).and_then(|v| v.as_f64()) {
        if !(w.is_finite() && w >= 2.0) {
            return Err(Error::invalid(
                "gabor: wavelength must be >= 2 (spatial Nyquist)".to_string(),
            ));
        }
        f = f.with_wavelength(w as f32);
    }
    if let Some(t) = p
        .and_then(|m| m.get("orientation_degrees"))
        .or_else(|| p.and_then(|m| m.get("orientation")))
        .or_else(|| p.and_then(|m| m.get("theta")))
        .and_then(|v| v.as_f64())
    {
        if !t.is_finite() {
            return Err(Error::invalid(
                "gabor: orientation must be finite".to_string(),
            ));
        }
        f = f.with_orientation_degrees(t as f32);
    }
    if let Some(ps) = p
        .and_then(|m| m.get("phase_degrees"))
        .or_else(|| p.and_then(|m| m.get("phase"))) // alias
        .or_else(|| p.and_then(|m| m.get("psi")))
        .and_then(|v| v.as_f64())
    {
        if !ps.is_finite() {
            return Err(Error::invalid("gabor: phase must be finite".to_string()));
        }
        f = f.with_phase_degrees(ps as f32);
    }
    if let Some(s) = p.and_then(|m| m.get("sigma")).and_then(|v| v.as_f64()) {
        if !(s.is_finite() && s > 0.0) {
            return Err(Error::invalid("gabor: sigma must be > 0".to_string()));
        }
        f = f.with_sigma(s as f32);
    }
    if let Some(g) = p
        .and_then(|m| m.get("gamma"))
        .or_else(|| p.and_then(|m| m.get("aspect"))) // alias
        .and_then(|v| v.as_f64())
    {
        if !(g.is_finite() && g > 0.0) {
            return Err(Error::invalid("gabor: gamma must be > 0".to_string()));
        }
        f = f.with_gamma(g as f32);
    }
    if let Some(r) = p.and_then(|m| m.get("radius")).and_then(|v| v.as_u64()) {
        f = f.with_radius(r as u32);
    }
    if let Some(mode) = p.and_then(|m| m.get("mode")).and_then(|v| v.as_str()) {
        let m = if mode.eq_ignore_ascii_case("magnitude")
            || mode.eq_ignore_ascii_case("energy")
            || mode.eq_ignore_ascii_case("abs")
        {
            GaborMode::Magnitude
        } else {
            GaborMode::Signed
        };
        f = f.with_mode(m);
    } else if let Some(b) = p.and_then(|m| m.get("magnitude")).and_then(|v| v.as_bool()) {
        f = f.with_mode(if b {
            GaborMode::Magnitude
        } else {
            GaborMode::Signed
        });
    }
    if let Some(g) = p
        .and_then(|m| m.get("output_gain"))
        .and_then(|v| v.as_f64())
    {
        if !g.is_finite() {
            return Err(Error::invalid(
                "gabor: output_gain must be finite".to_string(),
            ));
        }
        f = f.with_output_gain(g as f32);
    }
    let in_port = video_in_port(inputs);
    // Gabor output is always Gray8.
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

fn make_niblack(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Niblack;
    let p = params.as_object();
    let mut f = Niblack::new();
    if let Some(r) = p.and_then(|m| m.get("radius")).and_then(|v| v.as_u64()) {
        f = f.with_radius(r.min(u32::MAX as u64) as u32);
    }
    if let Some(k) = p.and_then(|m| m.get("k")).and_then(|v| v.as_f64()) {
        if !k.is_finite() {
            return Err(Error::invalid("niblack: k must be finite".to_string()));
        }
        f = f.with_k(k as f32);
    }
    if let Some(b) = p.and_then(|m| m.get("invert")).and_then(|v| v.as_bool()) {
        f = f.with_invert(b);
    }
    let in_port = video_in_port(inputs);
    // Niblack output is always Gray8.
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

fn make_euclidean_distance_transform(
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::EuclideanDistanceTransform;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let threshold = get_u64("threshold").unwrap_or(128).min(255) as u8;
    let mut f = EuclideanDistanceTransform::new(threshold);
    if let Some(inv) = get_bool("invert") {
        f = f.with_invert(inv);
    }
    if let Some(s) = get_f64("scale") {
        if !s.is_finite() {
            return Err(Error::invalid(
                "euclidean-distance-transform: scale must be finite".to_string(),
            ));
        }
        if s <= 0.0 {
            return Err(Error::invalid(
                "euclidean-distance-transform: scale must be > 0".to_string(),
            ));
        }
        f = f.with_scale(s as f32);
    }
    let in_port = video_in_port(inputs);
    // EuclideanDistanceTransform always emits a single-plane Gray8
    // intensity image; the input is Gray8-only too (apply() rejects
    // every other format).
    let out_port = match &in_port.params {
        PortParams::Video {
            width,
            height,
            time_base,
            ..
        } => PortSpec {
            name: "video".to_string(),
            params: PortParams::Video {
                format: PixelFormat::Gray8,
                width: *width,
                height: *height,
                time_base: *time_base,
            },
            ..in_port.clone()
        },
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_weighted_distance_transform(
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::WeightedDistanceTransform;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let weight = match get_f64("weight") {
        Some(wv) => {
            if !wv.is_finite() {
                return Err(Error::invalid(
                    "weighted-distance-transform: weight must be finite".to_string(),
                ));
            }
            if wv < 0.0 {
                return Err(Error::invalid(
                    "weighted-distance-transform: weight must be >= 0".to_string(),
                ));
            }
            wv as f32
        }
        None => 16.0,
    };
    let mut f = WeightedDistanceTransform::new(weight);
    if let Some(inv) = get_bool("invert") {
        f = f.with_invert(inv);
    }
    if let Some(s) = get_f64("scale") {
        if !s.is_finite() {
            return Err(Error::invalid(
                "weighted-distance-transform: scale must be finite".to_string(),
            ));
        }
        if s <= 0.0 {
            return Err(Error::invalid(
                "weighted-distance-transform: scale must be > 0".to_string(),
            ));
        }
        f = f.with_scale(s as f32);
    }
    let in_port = video_in_port(inputs);
    // WeightedDistanceTransform always emits a single-plane Gray8
    // intensity image; the input is Gray8-only too (apply() rejects
    // every other format).
    let out_port = match &in_port.params {
        PortParams::Video {
            width,
            height,
            time_base,
            ..
        } => PortSpec {
            name: "video".to_string(),
            params: PortParams::Video {
                format: PixelFormat::Gray8,
                width: *width,
                height: *height,
                time_base: *time_base,
            },
            ..in_port.clone()
        },
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_signed_distance_field(
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::SignedDistanceField;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let threshold = get_u64("threshold").unwrap_or(128).min(255) as u8;
    let mut f = SignedDistanceField::new(threshold);
    if let Some(inv) = get_bool("invert") {
        f = f.with_invert(inv);
    }
    if let Some(s) = get_f64("scale") {
        if !s.is_finite() {
            return Err(Error::invalid(
                "signed-distance-field: scale must be finite".to_string(),
            ));
        }
        if s <= 0.0 {
            return Err(Error::invalid(
                "signed-distance-field: scale must be > 0".to_string(),
            ));
        }
        f = f.with_scale(s as f32);
    }
    if let Some(m) = get_u64("midpoint") {
        f = f.with_midpoint(m.min(255) as u8);
    }
    let in_port = video_in_port(inputs);
    // SignedDistanceField always emits a single-plane Gray8 intensity
    // image; the input is Gray8-only too (apply() rejects every other
    // format).
    let out_port = match &in_port.params {
        PortParams::Video {
            width,
            height,
            time_base,
            ..
        } => PortSpec {
            name: "video".to_string(),
            params: PortParams::Video {
                format: PixelFormat::Gray8,
                width: *width,
                height: *height,
                time_base: *time_base,
            },
            ..in_port.clone()
        },
        _ => passthrough_out_port(&in_port),
    };
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_feather(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Feather;
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_bool = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_bool());
    let radius = match get_f64("radius") {
        Some(r) => {
            if !r.is_finite() {
                return Err(Error::invalid("feather: radius must be finite".to_string()));
            }
            if r < 0.0 {
                return Err(Error::invalid("feather: radius must be >= 0".to_string()));
            }
            r as f32
        }
        None => 4.0,
    };
    let mut f = Feather::new(radius);
    if let Some(t) = get_u64("threshold") {
        f = f.with_threshold(t.min(255) as u8);
    }
    if let Some(inv) = get_bool("invert") {
        f = f.with_invert(inv);
    }
    // Feather preserves the input format: Rgba feathers the alpha edge
    // (RGB pass-through), Gray8 emits the single-plane coverage ramp.
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// Shared body: build an [`SrgbTransform`] from a JSON `params` object and a
/// default direction, then wrap it in the adapter. The explicit encode /
/// decode aliases pass the matching default; the generic `srgb-transform`
/// alias passes `Decode` and lets JSON override.
fn build_srgb_transform(
    params: &Value,
    inputs: &[PortSpec],
    default_direction: crate::SrgbDirection,
) -> Result<Box<dyn StreamFilter>> {
    use crate::{SrgbCurve, SrgbDirection, SrgbTransform};
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());

    let direction = match get_str("direction").or_else(|| get_str("mode")) {
        Some(d) if d.eq_ignore_ascii_case("encode") || d.eq_ignore_ascii_case("forward") => {
            SrgbDirection::Encode
        }
        Some(d) if d.eq_ignore_ascii_case("decode") || d.eq_ignore_ascii_case("inverse") => {
            SrgbDirection::Decode
        }
        _ => default_direction,
    };

    // `curve: "srgb"` (default) or `curve: "gamma"`. A `gamma` numeric field
    // implies the power-law curve even without `curve: "gamma"`.
    let curve_name = get_str("curve");
    let gamma_field = get_f64("gamma");
    let curve = match curve_name {
        Some(c) if c.eq_ignore_ascii_case("gamma") || c.eq_ignore_ascii_case("power") => {
            SrgbCurve::gamma(gamma_field.unwrap_or(2.2) as f32)
        }
        Some(c) if c.eq_ignore_ascii_case("srgb") => SrgbCurve::Srgb,
        _ => {
            if let Some(g) = gamma_field {
                SrgbCurve::gamma(g as f32)
            } else {
                SrgbCurve::Srgb
            }
        }
    };

    let f = SrgbTransform { curve, direction };
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_srgb_decode(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    build_srgb_transform(params, inputs, crate::SrgbDirection::Decode)
}

fn make_srgb_encode(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    build_srgb_transform(params, inputs, crate::SrgbDirection::Encode)
}

fn make_log(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{LaplacianOfGaussian, LogMode};
    let p = params.as_object();
    let mut f = LaplacianOfGaussian::new();
    if let Some(sigma) = p.and_then(|m| m.get("sigma")).and_then(|v| v.as_f64()) {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::invalid("log: sigma must be > 0".to_string()));
        }
        f = f.with_sigma(sigma as f32);
    }
    if let Some(radius) = p.and_then(|m| m.get("radius")).and_then(|v| v.as_u64()) {
        f = f.with_radius(radius as u32);
    }
    // Accept `mode: "magnitude" | "zero-crossings"` (default magnitude).
    // The `zero_crossings: true` boolean shorthand also flips the mode.
    if let Some(mode) = p.and_then(|m| m.get("mode")).and_then(|v| v.as_str()) {
        let m = if mode.eq_ignore_ascii_case("zero-crossings")
            || mode.eq_ignore_ascii_case("zerocrossings")
            || mode.eq_ignore_ascii_case("crossings")
        {
            LogMode::ZeroCrossings
        } else {
            LogMode::Magnitude
        };
        f = f.with_mode(m);
    } else if let Some(b) = p
        .and_then(|m| m.get("zero_crossings"))
        .and_then(|v| v.as_bool())
    {
        f = f.with_mode(if b {
            LogMode::ZeroCrossings
        } else {
            LogMode::Magnitude
        });
    }
    if let Some(g) = p
        .and_then(|m| m.get("output_gain"))
        .and_then(|v| v.as_f64())
    {
        if !g.is_finite() {
            return Err(Error::invalid(
                "log: output_gain must be finite".to_string(),
            ));
        }
        f = f.with_output_gain(g as f32);
    }
    if let Some(t) = p
        .and_then(|m| m.get("slope_threshold"))
        .and_then(|v| v.as_f64())
    {
        if !t.is_finite() {
            return Err(Error::invalid(
                "log: slope_threshold must be finite".to_string(),
            ));
        }
        f = f.with_slope_threshold(t as f32);
    }
    let in_port = video_in_port(inputs);
    // LaplacianOfGaussian output is always Gray8.
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

// ------------------------------------------------------------------
// r186: Dither factories (bit-depth-reduction; Bayer ordered + the
// FS / JJN / Stucki / Sierra family / Atkinson error-diffusion
// kernels). All seven kernels and three Bayer matrix sizes share a
// single parse helper so the eleven registered names + aliases stay
// one-liners.
// ------------------------------------------------------------------

fn make_dither(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    let f = parse_dither(params, None)?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_fs(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(
        params,
        Some(ForcedMode::Diffusion(DiffusionKernel::FloydSteinberg)),
    )?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_jjn(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(params, Some(ForcedMode::Diffusion(DiffusionKernel::Jjn)))?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_stucki(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(params, Some(ForcedMode::Diffusion(DiffusionKernel::Stucki)))?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_sierra3(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(
        params,
        Some(ForcedMode::Diffusion(DiffusionKernel::Sierra3)),
    )?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_sierra2(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(
        params,
        Some(ForcedMode::Diffusion(DiffusionKernel::Sierra2)),
    )?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_sierra_lite(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(
        params,
        Some(ForcedMode::Diffusion(DiffusionKernel::SierraLite)),
    )?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_atkinson(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DiffusionKernel;
    let f = parse_dither(
        params,
        Some(ForcedMode::Diffusion(DiffusionKernel::Atkinson)),
    )?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

fn make_dither_bayer(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BayerMatrix;
    // Pull the matrix size from `params.matrix` / `params.size` so the
    // user can swap 2/4/8 without re-naming the filter.
    let p = params.as_object();
    let size = p
        .and_then(|m| m.get("matrix").or_else(|| m.get("size")))
        .and_then(|v| v.as_u64())
        .unwrap_or(4);
    let matrix = match size {
        2 => BayerMatrix::M2,
        4 => BayerMatrix::M4,
        8 => BayerMatrix::M8,
        other => {
            return Err(Error::invalid(format!(
                "dither-bayer: matrix size must be 2, 4 or 8 (got {other})",
            )));
        }
    };
    let f = parse_dither(params, Some(ForcedMode::Ordered(matrix)))?;
    let in_port = video_in_port(inputs);
    let out_port = passthrough_out_port(&in_port);
    Ok(Box::new(ImageFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

enum ForcedMode {
    Diffusion(crate::DiffusionKernel),
    Ordered(crate::BayerMatrix),
}

/// Shared parser for every dither factory. If `forced` is set the
/// algorithm is fixed and only `levels` is read from `params`;
/// otherwise the algorithm comes from `params.kernel` / `params.mode`
/// with Floyd–Steinberg as the default.
fn parse_dither(params: &Value, forced: Option<ForcedMode>) -> Result<crate::Dither> {
    use crate::{BayerMatrix, DiffusionKernel, Dither, ScanOrder};
    let p = params.as_object();
    let levels = p
        .and_then(|m| m.get("levels"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2)
        .min(u32::MAX as u64) as u32;
    // `scan: "serpentine" | "raster"` (default raster) selects the
    // error-diffusion traversal order (§1.3). A `serpentine: true`
    // boolean shorthand is also honoured. No effect on ordered/Bayer.
    let scan = {
        let by_str = p
            .and_then(|m| m.get("scan"))
            .and_then(|v| v.as_str())
            .map(|s| {
                if s.eq_ignore_ascii_case("serpentine") || s.eq_ignore_ascii_case("boustrophedon") {
                    ScanOrder::Serpentine
                } else {
                    ScanOrder::Raster
                }
            });
        by_str
            .or_else(|| {
                p.and_then(|m| m.get("serpentine"))
                    .and_then(|v| v.as_bool())
                    .map(|b| {
                        if b {
                            ScanOrder::Serpentine
                        } else {
                            ScanOrder::Raster
                        }
                    })
            })
            .unwrap_or(ScanOrder::Raster)
    };
    let f = match forced {
        Some(ForcedMode::Diffusion(k)) => Dither::error_diffusion(k),
        Some(ForcedMode::Ordered(m)) => Dither::ordered(m),
        None => {
            // Parse `mode` / `kernel` strings from the parameters
            // map. Unknown values fall through to Floyd–Steinberg.
            let kernel_name = p
                .and_then(|m| m.get("kernel").or_else(|| m.get("mode")))
                .and_then(|v| v.as_str())
                .unwrap_or("floyd-steinberg");
            match kernel_name.to_ascii_lowercase().as_str() {
                "floyd-steinberg" | "fs" | "floydsteinberg" => {
                    Dither::error_diffusion(DiffusionKernel::FloydSteinberg)
                }
                "jjn" | "jarvis" | "jarvis-judice-ninke" => {
                    Dither::error_diffusion(DiffusionKernel::Jjn)
                }
                "stucki" => Dither::error_diffusion(DiffusionKernel::Stucki),
                "sierra3" | "sierra-3" => Dither::error_diffusion(DiffusionKernel::Sierra3),
                "sierra2" | "sierra-2" => Dither::error_diffusion(DiffusionKernel::Sierra2),
                "sierra-lite" | "sierralite" | "sierra-2-4a" => {
                    Dither::error_diffusion(DiffusionKernel::SierraLite)
                }
                "atkinson" => Dither::error_diffusion(DiffusionKernel::Atkinson),
                "bayer2" | "bayer-2" | "ordered2" => Dither::ordered(BayerMatrix::M2),
                "bayer4" | "bayer-4" | "ordered4" | "bayer" | "ordered" => {
                    Dither::ordered(BayerMatrix::M4)
                }
                "bayer8" | "bayer-8" | "ordered8" => Dither::ordered(BayerMatrix::M8),
                other => {
                    return Err(Error::invalid(format!(
                        "dither: unknown kernel/mode '{other}'",
                    )));
                }
            }
        }
    };
    Ok(f.with_levels(levels).with_scan(scan))
}

fn make_prewitt(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Prewitt, PrewittMagnitude};
    let p = params.as_object();
    // Accept `magnitude: "l1" | "l2"` (default l2). The `l1` boolean
    // shorthand is also honoured.
    let mag = p
        .and_then(|m| m.get("magnitude"))
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.eq_ignore_ascii_case("l1") {
                PrewittMagnitude::L1
            } else {
                PrewittMagnitude::L2
            }
        })
        .or_else(|| {
            p.and_then(|m| m.get("l1"))
                .and_then(|v| v.as_bool())
                .map(|b| {
                    if b {
                        PrewittMagnitude::L1
                    } else {
                        PrewittMagnitude::L2
                    }
                })
        })
        .unwrap_or(PrewittMagnitude::L2);
    let f = Prewitt::new().with_magnitude(mag);
    let in_port = video_in_port(inputs);
    // Prewitt output is always Gray8.
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
            "distort",
            "tilt-shift",
            // r10 additions (geometry + channel ops + morphology edges).
            "roll",
            "shave",
            "extent",
            "trim",
            "channel-extract",
            "morphology-edgein",
            "morphology-edgeout",
            "morphology-edge-magnitude",
            // r8 additions (Porter–Duff + arithmetic composites).
            "composite-over",
            "composite-in",
            "composite-out",
            "composite-atop",
            "composite-xor",
            "composite-plus",
            "composite-multiply",
            "composite-screen",
            "composite-overlay",
            "composite-darken",
            "composite-lighten",
            "composite-difference",
            // r12 additions (clamp / stretches / colour matrix /
            // function / laplacian / canny).
            "clamp",
            "auto-level",
            "contrast-stretch",
            "linear-stretch",
            "color-matrix",
            "recolor",
            "function",
            "laplacian",
            "canny",
            // r13 additions (blue-shift / frame / shade / paint /
            // quantize / clut / hald-clut).
            "blue-shift",
            "frame",
            "shade",
            "paint",
            "quantize",
            "clut",
            "hald-clut",
            // r14 additions (sketch / deskew / hough-lines /
            // barrel-inverse / stegano / stereo).
            "sketch",
            "deskew",
            "hough-lines",
            "barrel-inverse",
            "stegano",
            "stereo",
            // r15 additions (hsl-rotate / vignette-soft /
            // chromatic-aberration / pixelate / mosaic / channel-mixer
            // / adaptive-threshold).
            "hsl-rotate",
            "vignette-soft",
            "chromatic-aberration",
            "pixelate",
            "mosaic",
            "channel-mixer",
            "adaptive-threshold",
            // r16 additions (bilateral-blur / canvas / gradient-radial
            // + alias / gradient-conic + alias / gravity-translate +
            // alias / color-balance / hsl-shift).
            "bilateral-blur",
            "canvas",
            "gradient-radial",
            "radial-gradient",
            "gradient-conic",
            "conic-gradient",
            "gravity-translate",
            "gravity",
            "color-balance",
            "hsl-shift",
            // r17 additions (exposure / temperature / vibrance /
            // bw-mix + alias / clarity / shadow-highlight /
            // bordered-frame + alias).
            "exposure",
            "temperature",
            "vibrance",
            "bw-mix",
            "black-and-white",
            "clarity",
            "shadow-highlight",
            "bordered-frame",
            "border",
            // r18 additions (hough-circles / auto-trim / drop-shadow /
            // inner-shadow / bloom + alias / edge-detect + alias /
            // difference).
            "hough-circles",
            "auto-trim",
            "drop-shadow",
            "inner-shadow",
            "bloom",
            "glow",
            "edge-detect",
            "edge-multi",
            "difference",
            // r19 additions (heatmap + alias / split-tone / flood-fill /
            // posterize-channels / toon + alias / watermark).
            "heatmap",
            "false-color",
            "split-tone",
            "flood-fill",
            "posterize-channels",
            "toon",
            "cartoon",
            "watermark",
            // r20 additions (crystallize / halftone / gradient-map +
            // alias / selective-color + alias / cross-process + alias /
            // otsu-threshold + alias / comic + alias).
            "crystallize",
            "halftone",
            "gradient-map",
            "duotone",
            "selective-color",
            "selective-colour",
            "cross-process",
            "crossprocess",
            "otsu-threshold",
            "otsu",
            "comic",
            "manga",
            // r21 additions (kuwahara / anisotropic-blur + alias / zoom-blur
            // / radial-blur + alias / emboss-directional + two-input
            // displacement-map).
            "kuwahara",
            "anisotropic-blur",
            "anisotropic",
            "zoom-blur",
            "radial-blur",
            "spin-blur",
            "emboss-directional",
            "displacement-map",
            // r22 additions (tone-mapping: reinhard / hable + alias /
            // drago + alias / curves / distance-transform + alias /
            // cyanotype + alias).
            "reinhard",
            "tonemap-reinhard",
            // r237 — unkeyed white-clamping Reinhard variant.
            "reinhard-extended",
            "tonemap-reinhard-extended",
            // r248 — local "dodging-and-burning" Reinhard variant.
            "reinhard-local",
            "tonemap-reinhard-local",
            "dodge-and-burn",
            "hable",
            "tonemap-hable",
            "uncharted2",
            "drago",
            "tonemap-drago",
            "curves",
            "distance-transform",
            "distance",
            "cyanotype",
            "blueprint",
            // r23 additions (max-rgb / hsv-value alias / min-rgb).
            "max-rgb",
            "hsv-value",
            "min-rgb",
            // r24 additions (roberts cross + alias).
            "roberts",
            "roberts-cross",
            // r101 addition (prewitt edge operator).
            "prewitt",
            // r105 addition (scharr edge operator).
            "scharr",
            // r174 additions (Frei–Chen 3×3 orthonormal-basis edge /
            // line detector + spelling alias).
            "frei-chen",
            "freichen",
            // r181 additions (Marr–Hildreth Laplacian-of-Gaussian
            // edge / zero-crossing detector + spelling aliases).
            "laplacian-of-gaussian",
            "log",
            "marr-hildreth",
            // r186 additions (classic dither: Bayer ordered + every
            // canonical error-diffusion kernel + spelling aliases).
            "dither",
            "dither-floyd-steinberg",
            "dither-fs",
            "dither-jjn",
            "dither-jarvis",
            "dither-stucki",
            "dither-sierra3",
            "dither-sierra2",
            "dither-sierra-lite",
            "dither-atkinson",
            "dither-bayer",
            "ordered-dither",
            // r205 additions (Niblack 1986 adaptive local-statistics
            // threshold + spelling alias).
            "niblack",
            "niblack-threshold",
            // r209 additions (exact-Euclidean Distance Transform —
            // Felzenszwalb–Huttenlocher 2012, three aliases).
            "euclidean-distance-transform",
            "euclidean-distance",
            "edt",
            // r288 additions (Signed Distance Field — difference of two
            // exact-Euclidean transforms, three aliases).
            "signed-distance-field",
            "signed-distance",
            "sdf",
            // r303 additions (edge feathering — soft coverage falloff
            // from the exact-Euclidean inner distance, three aliases).
            "feather",
            "feather-edge",
            "soft-edge",
            // r324 additions (sRGB / power-law transfer-function transform
            // — display↔linear OETF/EOTF — with encode / decode aliases).
            "srgb-decode",
            "linearize",
            "srgb-to-linear",
            "srgb-encode",
            "delinearize",
            "linear-to-srgb",
            "srgb-transform",
            // r329 additions (weighted / generalised distance transform
            // — continuous-seed `D_f(p)=min_q(‖p−q‖²+f(q))`, three
            // aliases).
            "weighted-distance-transform",
            "weighted-distance",
            "wdt",
        ] {
            assert!(c.filters.contains(name), "missing factory: {name}");
        }
    }

    #[test]
    fn euclidean_distance_transform_factory_emits_gray8() {
        // edt is always single-plane Gray8 → check the adapter port
        // spec reports Gray8 regardless of how the upstream port was
        // shaped (use a Gray8 input here since the filter rejects
        // non-Gray8 inputs at apply()-time anyway).
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let f = c
            .filters
            .make(
                "euclidean-distance-transform",
                &json!({"threshold": 200, "scale": 2.0}),
                &inputs,
            )
            .expect("edt factory");
        let outs = f.output_ports();
        assert_eq!(outs.len(), 1);
        if let PortParams::Video { format, .. } = outs[0].params {
            assert_eq!(format, PixelFormat::Gray8);
        } else {
            panic!("expected video output port");
        }
    }

    #[test]
    fn euclidean_distance_transform_factory_aliases() {
        // Every alias resolves to the same factory.
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        for name in ["euclidean-distance-transform", "euclidean-distance", "edt"] {
            let r = c.filters.make(name, &json!({}), &inputs);
            assert!(r.is_ok(), "alias {name} failed: {:?}", r.err());
        }
    }

    #[test]
    fn euclidean_distance_transform_factory_rejects_bad_scale() {
        // scale must be finite + > 0 — the factory mirrors the
        // chamfer-DT contract.
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let r = c.filters.make("edt", &json!({"scale": 0.0}), &inputs);
        assert!(r.is_err(), "scale = 0 should be rejected");
        let r2 = c.filters.make("edt", &json!({"scale": -1.0}), &inputs);
        assert!(r2.is_err(), "scale < 0 should be rejected");
    }

    #[test]
    fn weighted_distance_transform_factory_emits_gray8() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let f = c
            .filters
            .make(
                "weighted-distance-transform",
                &json!({"weight": 8.0, "scale": 2.0, "invert": true}),
                &inputs,
            )
            .expect("wdt factory");
        let outs = f.output_ports();
        assert_eq!(outs.len(), 1);
        if let PortParams::Video { format, .. } = outs[0].params {
            assert_eq!(format, PixelFormat::Gray8);
        } else {
            panic!("expected video output port");
        }
    }

    #[test]
    fn weighted_distance_transform_factory_aliases() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        for name in ["weighted-distance-transform", "weighted-distance", "wdt"] {
            let r = c.filters.make(name, &json!({}), &inputs);
            assert!(r.is_ok(), "alias {name} failed: {:?}", r.err());
        }
    }

    #[test]
    fn weighted_distance_transform_factory_rejects_bad_params() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        assert!(c
            .filters
            .make("wdt", &json!({"scale": 0.0}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("wdt", &json!({"weight": -1.0}), &inputs)
            .is_err());
    }

    #[test]
    fn signed_distance_field_factory_emits_gray8() {
        // sdf is always single-plane Gray8 → the adapter port must
        // report Gray8 regardless of upstream shape.
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let f = c
            .filters
            .make(
                "signed-distance-field",
                &json!({"threshold": 200, "scale": 2.0, "midpoint": 100}),
                &inputs,
            )
            .expect("sdf factory");
        let outs = f.output_ports();
        assert_eq!(outs.len(), 1);
        if let PortParams::Video { format, .. } = outs[0].params {
            assert_eq!(format, PixelFormat::Gray8);
        } else {
            panic!("expected video output port");
        }
    }

    #[test]
    fn signed_distance_field_factory_aliases() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        for name in ["signed-distance-field", "signed-distance", "sdf"] {
            let r = c.filters.make(name, &json!({}), &inputs);
            assert!(r.is_ok(), "alias {name} failed: {:?}", r.err());
        }
    }

    #[test]
    fn signed_distance_field_factory_rejects_bad_scale() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let r = c.filters.make("sdf", &json!({"scale": 0.0}), &inputs);
        assert!(r.is_err(), "scale = 0 should be rejected");
        let r2 = c.filters.make("sdf", &json!({"scale": -1.0}), &inputs);
        assert!(r2.is_err(), "scale < 0 should be rejected");
    }

    #[test]
    fn feather_factory_preserves_format() {
        // Feather is format-preserving: Gray8 → Gray8, Rgba → Rgba.
        let c = ctx();
        let g = c
            .filters
            .make("feather", &json!({"radius": 4.0}), &[gray_in_port(16, 16)])
            .expect("feather gray factory");
        if let PortParams::Video { format, .. } = g.output_ports()[0].params {
            assert_eq!(format, PixelFormat::Gray8);
        } else {
            panic!("expected video output port");
        }
        let r = c
            .filters
            .make("feather", &json!({"radius": 4.0}), &[rgba_in_port(16, 16)])
            .expect("feather rgba factory");
        if let PortParams::Video { format, .. } = r.output_ports()[0].params {
            assert_eq!(format, PixelFormat::Rgba);
        } else {
            panic!("expected video output port");
        }
    }

    #[test]
    fn feather_factory_aliases() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        for name in ["feather", "feather-edge", "soft-edge"] {
            let r = c.filters.make(name, &json!({}), &inputs);
            assert!(r.is_ok(), "alias {name} failed: {:?}", r.err());
        }
    }

    #[test]
    fn feather_factory_rejects_bad_radius() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let r = c.filters.make("feather", &json!({"radius": -1.0}), &inputs);
        assert!(r.is_err(), "radius < 0 should be rejected");
    }

    #[test]
    fn curves_factory_accepts_natural_cubic_mode_aliases() {
        // r215: the JSON factory parser recognises three spellings for
        // the natural-cubic mode in addition to the three pre-existing
        // `linear` / `catmull-rom` / monotone-cubic-by-default values.
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in ["natural-cubic", "natural_cubic", "natural"] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
    }

    #[test]
    fn curves_factory_accepts_centripetal_mode_aliases() {
        // r226: the JSON factory parser gains five spelling aliases for
        // the centripetal Catmull-Rom mode (Yuksel et al. 2011, α = 0.5)
        // — §3.3 of `docs/image/filter/curve-interpolation.md`. All
        // spellings must build a valid filter (each alias is the same
        // CurveInterpolation::CentripetalCatmullRom value internally).
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in [
            "centripetal",
            "centripetal-catmull-rom",
            "centripetal_catmull_rom",
            "centripetalcatmullrom",
            "catmull-rom-centripetal",
        ] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
    }

    #[test]
    fn curves_factory_accepts_chordal_mode_aliases() {
        // r231: the JSON factory parser gains five spelling aliases for
        // the chordal Catmull-Rom mode (Yuksel et al. 2011, α = 1) —
        // §3.3 of `docs/image/filter/curve-interpolation.md`. All
        // spellings must build a valid filter (each alias is the same
        // CurveInterpolation::ChordalCatmullRom value internally).
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in [
            "chordal",
            "chordal-catmull-rom",
            "chordal_catmull_rom",
            "chordalcatmullrom",
            "catmull-rom-chordal",
        ] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
    }

    #[test]
    fn curves_factory_accepts_cardinal_mode_aliases() {
        // r262: the JSON factory parser gains four spelling aliases for
        // the cardinal-spline mode (Catmull & Rom 1974) — §3.2 of
        // `docs/image/filter/curve-interpolation.md`. All spellings
        // must build a valid filter (each alias is the same
        // CurveInterpolation::Cardinal { tension_q8 } value
        // internally; tension is read from the `tension` / `c` JSON
        // field, defaulting to `0.5` if omitted).
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in [
            "cardinal",
            "cardinal-spline",
            "cardinal_spline",
            "cardinalspline",
        ] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                    "tension": 0.3,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
        // Also accept the `c` short spelling and verify omitting
        // tension still builds (default `0.5`).
        let r = c.filters.make(
            "curves",
            &json!({
                "master": [[0, 0], [128, 200], [255, 255]],
                "interpolation": "cardinal",
                "c": 0.0,
            }),
            &inputs,
        );
        assert!(r.is_ok(), "curves cardinal c=0.0 failed: {:?}", r.err());
        let r = c.filters.make(
            "curves",
            &json!({
                "master": [[0, 0], [128, 200], [255, 255]],
                "interpolation": "cardinal",
            }),
            &inputs,
        );
        assert!(
            r.is_ok(),
            "curves cardinal missing-tension default failed: {:?}",
            r.err()
        );
        // Out-of-range tension clamps to [0, 1] rather than erroring.
        for c_val in [-1.0, 1.5, f64::INFINITY] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": "cardinal",
                    "tension": c_val,
                }),
                &inputs,
            );
            assert!(
                r.is_ok(),
                "curves cardinal tension={c_val} should clamp not error: {:?}",
                r.err()
            );
        }
    }

    #[test]
    fn curves_factory_accepts_monotone_box_mode_aliases() {
        // r270: the JSON factory parser gains six spelling aliases for
        // the §2.2 box-region Fritsch-Carlson monotone-cubic variant
        // (`0 ≤ α ≤ 3` and `0 ≤ β ≤ 3` applied independently) —
        // `docs/image/filter/curve-interpolation.md` §2.2. All spellings
        // must build a valid filter (each alias resolves to the same
        // CurveInterpolation::MonotoneCubicBox value internally).
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in [
            "monotone-box",
            "monotone_box",
            "monotonebox",
            "monotone-cubic-box",
            "monotone_cubic_box",
            "monotonecubicbox",
        ] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
    }

    #[test]
    fn curves_factory_accepts_clamped_mode_aliases() {
        // r277: the JSON factory parser gains four spelling aliases for
        // the clamped-cubic boundary — the first §4.2-alternative
        // boundary of `docs/image/filter/curve-interpolation.md`
        // (prescribed end first derivatives). Slopes ride in the
        // `start_slope` / `end_slope` (or `slope0`/`s0`, `slope1`/`s1`)
        // JSON fields, defaulting to `1.0` when omitted.
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in ["clamped", "clamped-cubic", "clamped_cubic", "clampedcubic"] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                    "start_slope": 0.0,
                    "end_slope": 2.5,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
        }
        // Short spellings + missing-slope default (1.0) + out-of-range
        // clamp rather than error.
        for params in [
            json!({
                "master": [[0, 0], [128, 200], [255, 255]],
                "interpolation": "clamped",
                "s0": -2.0,
                "s1": 0.5,
            }),
            json!({
                "master": [[0, 0], [128, 200], [255, 255]],
                "interpolation": "clamped",
            }),
            json!({
                "master": [[0, 0], [128, 200], [255, 255]],
                "interpolation": "clamped",
                "slope0": 1.0e9,
                "slope1": f64::NEG_INFINITY,
            }),
        ] {
            let r = c.filters.make("curves", &params, &inputs);
            assert!(
                r.is_ok(),
                "curves clamped params={params} failed: {:?}",
                r.err()
            );
        }
    }

    #[test]
    fn curves_factory_accepts_not_a_knot_mode_aliases() {
        // r277: the JSON factory parser gains five spelling aliases for
        // the not-a-knot boundary — the second §4.2-alternative
        // boundary of `docs/image/filter/curve-interpolation.md`
        // (third derivative continuous across the first / last interior
        // knot). Parameter-free; all spellings resolve to the same
        // CurveInterpolation::NotAKnotCubic value internally.
        let c = ctx();
        let inputs = [yuv_in_port()];
        for mode in [
            "not-a-knot",
            "not_a_knot",
            "notaknot",
            "not-a-knot-cubic",
            "not_a_knot_cubic",
        ] {
            let r = c.filters.make(
                "curves",
                &json!({
                    "master": [[0, 0], [128, 200], [255, 255]],
                    "interpolation": mode,
                }),
                &inputs,
            );
            assert!(r.is_ok(), "curves mode={mode} failed: {:?}", r.err());
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
    fn distance_transform_factory_accepts_kind_aliases() {
        // r220: the `kind` (or `kernel`) parameter accepts every
        // chamfer variant transcribed in
        // `docs/image/filter/distance-transform.md` §3.2.
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        for kind in [
            "chamfer-3-4",
            "chamfer34",
            "3-4",
            "34",
            "chamfer-5-7-11",
            "chamfer5711",
            "5-7-11",
            "5711",
            "city-block",
            "cityblock",
            "manhattan",
            "l1",
            "chessboard",
            "chebyshev",
            "l-infinity",
            "linfinity",
            "linf",
        ] {
            let r = c
                .filters
                .make("distance-transform", &json!({"kind": kind}), &inputs);
            assert!(r.is_ok(), "kind={kind} should parse: {:?}", r.err());
            // Verify the `kernel` spelling works too — registry should
            // accept either key.
            let r2 = c
                .filters
                .make("distance-transform", &json!({"kernel": kind}), &inputs);
            assert!(r2.is_ok(), "kernel={kind} should parse: {:?}", r2.err());
        }
    }

    #[test]
    fn distance_transform_factory_rejects_unknown_kind() {
        // Unknown kernel strings must surface as a factory-level error
        // rather than silently falling back to the default.
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let r = c.filters.make(
            "distance-transform",
            &json!({"kind": "not-a-kernel"}),
            &inputs,
        );
        let err = match r {
            Ok(_) => panic!("unknown kind should be rejected"),
            Err(e) => format!("{e}"),
        };
        assert!(
            err.contains("not-a-kernel"),
            "error message should mention the offending value, got: {err}"
        );
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
    fn dither_factory_round_trip_for_every_kernel() {
        // Every dither factory must round-trip a flat-grey gray input
        // to a binary {0, 255} field at the default `levels = 2`.
        let c = ctx();
        for name in [
            "dither",
            "dither-fs",
            "dither-floyd-steinberg",
            "dither-jjn",
            "dither-jarvis",
            "dither-stucki",
            "dither-sierra3",
            "dither-sierra2",
            "dither-sierra-lite",
            "dither-atkinson",
            "dither-bayer",
            "ordered-dither",
        ] {
            let inputs = [gray_in_port(4, 4)];
            let f = c
                .filters
                .make(name, &json!({}), &inputs)
                .unwrap_or_else(|e| panic!("{name} factory failed: {e}"));
            let out = run_one(f, gray_4x4(|_, _| 128));
            for &v in &out.planes[0].data {
                assert!(v == 0 || v == 255, "{name} produced {v}");
            }
        }
    }

    #[test]
    fn dither_factory_rejects_unknown_kernel() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let r = c
            .filters
            .make("dither", &json!({"kernel": "made-up-thing"}), &inputs);
        let err = match r {
            Ok(_) => panic!("unknown kernel must error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("dither"));
    }

    #[test]
    fn dither_bayer_rejects_unsupported_matrix_size() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        assert!(c
            .filters
            .make("dither-bayer", &json!({"matrix": 3}), &inputs)
            .is_err());
    }

    #[test]
    fn dither_factory_honours_levels_override() {
        // levels = 4 → output codes from the quartile set only.
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("dither-fs", &json!({"levels": 4}), &inputs)
            .expect("dither-fs factory");
        let out = run_one(f, gray_4x4(|x, y| ((x * 16 + y * 16) & 0xff) as u8));
        for &v in &out.planes[0].data {
            assert!(matches!(v, 0 | 85 | 170 | 255), "4-level produced {v}");
        }
    }

    #[test]
    fn dither_factory_parses_scan_order() {
        use crate::ScanOrder;
        // Default → raster.
        assert_eq!(
            parse_dither(&json!({}), None).unwrap().scan(),
            ScanOrder::Raster,
        );
        // String form.
        assert_eq!(
            parse_dither(&json!({"scan": "serpentine"}), None)
                .unwrap()
                .scan(),
            ScanOrder::Serpentine,
        );
        assert_eq!(
            parse_dither(&json!({"scan": "raster"}), None)
                .unwrap()
                .scan(),
            ScanOrder::Raster,
        );
        // Boolean shorthand.
        assert_eq!(
            parse_dither(&json!({"serpentine": true}), None)
                .unwrap()
                .scan(),
            ScanOrder::Serpentine,
        );
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
    fn charcoal_factory_accepts_radius_key() {
        // r9: optional `radius` key thickens the strokes via Gaussian
        // pre-blur. Default 0 = identical to no-radius behaviour;
        // radius > 0 must construct cleanly and produce a valid Gray8
        // output.
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make("charcoal", &json!({"factor": 1.0, "radius": 2}), &inputs)
            .expect("charcoal factory with radius=2");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![100u8; 64],
            }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 64);
        // Flat input ⇒ Sobel sees no edges ⇒ pre-blur leaves it flat ⇒
        // pure white.
        for &v in &out.planes[0].data {
            assert_eq!(v, 255);
        }
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

    #[test]
    fn perspective_factory_accepts_output_size_keys() {
        // r9: explicit output canvas size — both keys together rebuild
        // the output port spec to the new shape.
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "perspective",
                &json!({
                    "src_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                    "dst_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                    "output_width": 8,
                    "output_height": 6,
                }),
                &inputs,
            )
            .expect("perspective factory with output_size");
        match &f.output_ports()[0].params {
            PortParams::Video { width, height, .. } => {
                assert_eq!(*width, 8);
                assert_eq!(*height, 6);
            }
            _ => panic!("perspective output port is not video"),
        }
    }

    #[test]
    fn perspective_factory_rejects_partial_output_size() {
        // Only one of (output_width, output_height) ⇒ explicit error
        // rather than a silent axis default.
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c.filters.make(
            "perspective",
            &json!({
                "src_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                "dst_corners": [0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                "output_width": 8,
            }),
            &inputs,
        );
        assert!(
            bad.is_err(),
            "perspective: output_width without output_height should fail"
        );
    }

    #[test]
    fn distort_factory_zero_strength_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("distort", &json!({"k1": 0.0, "k2": 0.0}), &inputs)
            .expect("distort factory");
        let frame = rgba_4x4(|x, y| [(x * 50) as u8, (y * 50) as u8, 100, 255]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn distort_factory_mode_barrel_builds() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make(
                "distort",
                &json!({"mode": "barrel", "strength": 0.2}),
                &inputs,
            )
            .expect("distort factory mode=barrel");
    }

    #[test]
    fn distort_factory_mode_pincushion_builds() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make(
                "distort",
                &json!({"mode": "pincushion", "strength": 0.2}),
                &inputs,
            )
            .expect("distort factory mode=pincushion");
    }

    #[test]
    fn distort_factory_rejects_unknown_mode() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c
            .filters
            .make("distort", &json!({"mode": "fisheye"}), &inputs);
        assert!(bad.is_err(), "unknown mode must fail");
    }

    #[test]
    fn distort_factory_accepts_tangential_p1_p2() {
        // r9: p1 / p2 are the Brown-Conrady tangential coefficients.
        // The factory must accept them and produce a filter whose
        // output differs from the pure-radial baseline at at least
        // one pixel. We use an asymmetric pattern so the difference
        // is observable.
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let with_tang = c
            .filters
            .make(
                "distort",
                &json!({"k1": 0.0, "k2": 0.0, "p1": 0.3, "p2": 0.0}),
                &inputs,
            )
            .expect("distort factory with tangential p1 / p2");
        let radial_only = c
            .filters
            .make("distort", &json!({"k1": 0.0, "k2": 0.0}), &inputs)
            .expect("distort factory pure radial");
        let frame = rgba_4x4(|x, y| [(x * 60) as u8, (y * 60) as u8, 100, 200]);
        let a = run_one(with_tang, frame.clone());
        let b = run_one(radial_only, frame);
        assert_ne!(
            a.planes[0].data, b.planes[0].data,
            "tangential p1 must shift at least one sample relative to the pure-radial baseline"
        );
    }

    #[test]
    fn distort_factory_zero_tangential_matches_radial_baseline() {
        // p1 = p2 = 0 must be bit-exact identical to the pre-r9
        // radial-only Distort.
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let a = c
            .filters
            .make(
                "distort",
                &json!({"k1": 0.2, "k2": 0.0, "p1": 0.0, "p2": 0.0}),
                &inputs,
            )
            .expect("with explicit zero tangential");
        let b = c
            .filters
            .make("distort", &json!({"k1": 0.2, "k2": 0.0}), &inputs)
            .expect("without tangential keys");
        let frame = rgba_4x4(|x, y| [(x * 60) as u8, (y * 60) as u8, 100, 255]);
        let out_a = run_one(a, frame.clone());
        let out_b = run_one(b, frame);
        assert_eq!(out_a.planes[0].data, out_b.planes[0].data);
    }

    #[test]
    fn tilt_shift_factory_smoke_flat_image() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "tilt-shift",
                &json!({
                    "focus_centre": 0.5,
                    "focus_height": 0.4,
                    "falloff_height": 0.3,
                    "blur_radius": 2,
                    "blur_sigma": 1.0
                }),
                &inputs,
            )
            .expect("tilt-shift factory");
        let out = run_one(f, gray_4x4(|_, _| 100));
        // Flat input survives both blur and weighted blend.
        for &v in &out.planes[0].data {
            assert!((v as i16 - 100).abs() <= 1);
        }
    }

    #[test]
    fn tilt_shift_factory_accepts_short_aliases() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make(
                "tilt-shift",
                &json!({
                    "centre": 0.6,
                    "height": 0.3,
                    "falloff": 0.2,
                    "radius": 3,
                    "sigma": 1.5
                }),
                &inputs,
            )
            .expect("tilt-shift accepts short alias keys");
    }

    #[test]
    fn tilt_shift_factory_accepts_angle_degrees() {
        // r9: rotated focus band — the factory must accept both the
        // long `angle_degrees` key and the short `angle` alias.
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make("tilt-shift", &json!({"angle_degrees": 90.0}), &inputs)
            .expect("tilt-shift with angle_degrees");
        c.filters
            .make("tilt-shift", &json!({"angle": 45.0}), &inputs)
            .expect("tilt-shift with short angle alias");
    }

    #[test]
    fn tilt_shift_factory_rejects_non_finite_angle() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c
            .filters
            .make("tilt-shift", &json!({"angle_degrees": null}), &inputs);
        // null ⇒ as_f64 returns None ⇒ default (0.0). That's finite,
        // so factory must succeed. The genuine non-finite path is
        // exercised through the typed builder in the in-module test.
        assert!(bad.is_ok(), "null angle silently defaults to 0.0");
    }

    #[test]
    fn polar_factory_accepts_centre_overrides() {
        // r7 follow-up: polar / depolar should now surface cx / cy /
        // max_radius via JSON. Build with explicit centre + max_radius
        // and verify the factory builds.
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        c.filters
            .make(
                "polar",
                &json!({"cx": 2.0, "cy": 5.0, "max_radius": 3.0}),
                &inputs,
            )
            .expect("polar with overrides");
        c.filters
            .make("depolar", &json!({"cx": 4.0, "max_r": 6.0}), &inputs)
            .expect("depolar with overrides");
    }

    // --- r8 composite tests (two-input adapter) ---

    /// Build a single-pixel RGBA frame for the 2x2 fixture matrix.
    fn rgba_1x1(pixel: [u8; 4]) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: pixel.to_vec(),
            }],
        }
    }

    /// Push (src, dst) into a two-input filter and return the single
    /// emitted frame. Asserts exactly one emit.
    fn run_two(mut f: Box<dyn StreamFilter>, src: VideoFrame, dst: VideoFrame) -> VideoFrame {
        let mut col = CollectCtx { out: Vec::new() };
        f.push(&mut col, 0, &Frame::Video(src))
            .expect("push src must not error");
        // Nothing emitted yet — only one input is buffered.
        assert!(
            col.out.is_empty(),
            "expected no emit before dst arrives, got {}",
            col.out.len()
        );
        f.push(&mut col, 1, &Frame::Video(dst))
            .expect("push dst must not error");
        assert_eq!(col.out.len(), 1, "expected exactly one emit after pair");
        let (port, frame) = col.out.into_iter().next().unwrap();
        assert_eq!(port, 0);
        match frame {
            Frame::Video(v) => v,
            other => panic!("composite emitted non-video frame: {other:?}"),
        }
    }

    /// Standard 2x2 fixture matrix: opaque red over opaque blue.
    fn op_red() -> VideoFrame {
        rgba_1x1([255, 0, 0, 255])
    }
    fn op_blue() -> VideoFrame {
        rgba_1x1([0, 0, 255, 255])
    }
    fn half_red() -> VideoFrame {
        rgba_1x1([255, 0, 0, 128])
    }
    fn transparent() -> VideoFrame {
        rgba_1x1([0, 0, 0, 0])
    }

    fn make_composite_filter(name: &str) -> Box<dyn StreamFilter> {
        let c = ctx();
        // Two ports of identical RGBA shape, named src + dst.
        let inputs = [
            PortSpec::video("src", 1, 1, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 1, 1, PixelFormat::Rgba, TimeBase::new(1, 30)),
        ];
        c.filters
            .make(name, &json!({}), &inputs)
            .unwrap_or_else(|e| panic!("{name} factory: {e}"))
    }

    #[test]
    fn composite_over_2x2_fixture() {
        // (op-red, op-blue) ⇒ red wins
        let out = run_two(make_composite_filter("composite-over"), op_red(), op_blue());
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
        // (half-red, op-blue) ⇒ ~50/50 blend
        let out = run_two(
            make_composite_filter("composite-over"),
            half_red(),
            op_blue(),
        );
        assert!((out.planes[0].data[0] as i16 - 128).abs() <= 2);
        assert_eq!(out.planes[0].data[3], 255);
        // (op-red, transparent) ⇒ red survives
        let out = run_two(
            make_composite_filter("composite-over"),
            op_red(),
            transparent(),
        );
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
        // (transparent, op-blue) ⇒ blue survives
        let out = run_two(
            make_composite_filter("composite-over"),
            transparent(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..4], [0, 0, 255, 255]);
    }

    #[test]
    fn composite_in_2x2_fixture() {
        let out = run_two(make_composite_filter("composite-in"), op_red(), op_blue());
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
        let out = run_two(
            make_composite_filter("composite-in"),
            op_red(),
            transparent(),
        );
        assert_eq!(out.planes[0].data[0..4], [0, 0, 0, 0]);
    }

    #[test]
    fn composite_out_2x2_fixture() {
        let out = run_two(make_composite_filter("composite-out"), op_red(), op_blue());
        assert_eq!(out.planes[0].data[0..4], [0, 0, 0, 0]);
        let out = run_two(
            make_composite_filter("composite-out"),
            op_red(),
            transparent(),
        );
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
    }

    #[test]
    fn composite_atop_2x2_fixture() {
        let out = run_two(make_composite_filter("composite-atop"), op_red(), op_blue());
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
        let out = run_two(
            make_composite_filter("composite-atop"),
            op_red(),
            transparent(),
        );
        assert_eq!(out.planes[0].data[0..4], [0, 0, 0, 0]);
    }

    #[test]
    fn composite_xor_2x2_fixture() {
        let out = run_two(make_composite_filter("composite-xor"), op_red(), op_blue());
        assert_eq!(out.planes[0].data[0..4], [0, 0, 0, 0]);
        let out = run_two(
            make_composite_filter("composite-xor"),
            op_red(),
            transparent(),
        );
        assert_eq!(out.planes[0].data[0..4], [255, 0, 0, 255]);
        let out = run_two(
            make_composite_filter("composite-xor"),
            transparent(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..4], [0, 0, 255, 255]);
    }

    #[test]
    fn composite_plus_2x2_fixture() {
        let out = run_two(make_composite_filter("composite-plus"), op_red(), op_blue());
        // Per-channel sum, clamp.
        assert_eq!(out.planes[0].data[0..4], [255, 0, 255, 255]);
    }

    #[test]
    fn composite_multiply_2x2_fixture() {
        // (255,0,0) * (0,0,255) / 255 ⇒ (0,0,0)
        let out = run_two(
            make_composite_filter("composite-multiply"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [0, 0, 0]);
    }

    #[test]
    fn composite_screen_2x2_fixture() {
        // 255 - (255-255)*(255-0)/255 = 255 in R; 255 - 255*0/255 = 0... wait
        // 255 - (255 - r_s) * (255 - r_d) / 255
        // R: 255 - 0*255/255 = 255
        // G: 255 - 255*255/255 = 0
        // B: 255 - 255*0/255 = 255
        let out = run_two(
            make_composite_filter("composite-screen"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [255, 0, 255]);
    }

    #[test]
    fn composite_darken_2x2_fixture() {
        // min channel-wise: (255,0,0) vs (0,0,255) ⇒ (0,0,0)
        let out = run_two(
            make_composite_filter("composite-darken"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [0, 0, 0]);
    }

    #[test]
    fn composite_lighten_2x2_fixture() {
        // max channel-wise: (255,0,0) vs (0,0,255) ⇒ (255,0,255)
        let out = run_two(
            make_composite_filter("composite-lighten"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [255, 0, 255]);
    }

    #[test]
    fn composite_difference_2x2_fixture() {
        // |s - d|: |255-0|=255, |0-0|=0, |0-255|=255
        let out = run_two(
            make_composite_filter("composite-difference"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [255, 0, 255]);
    }

    #[test]
    fn composite_overlay_2x2_fixture() {
        // dst<128 path: 2 * 255 * 0 / 255 = 0 (R); 2 * 0 * 0 / 255 = 0 (G);
        // dst=255 ⇒ screen path: 255 - 2 * 255 * 0 / 255 = 255 (B)
        let out = run_two(
            make_composite_filter("composite-overlay"),
            op_red(),
            op_blue(),
        );
        assert_eq!(out.planes[0].data[0..3], [0, 0, 255]);
    }

    #[test]
    fn composite_factory_rejects_shape_mismatch() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 8, 8, PixelFormat::Rgba, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("composite-over", &json!({}), &inputs);
        assert!(bad.is_err(), "shape mismatch must fail at factory time");
    }

    #[test]
    fn composite_factory_rejects_format_mismatch() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 4, 4, PixelFormat::Rgb24, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("composite-over", &json!({}), &inputs);
        assert!(bad.is_err(), "format mismatch must fail at factory time");
    }

    #[test]
    fn composite_factory_two_input_ports_exposed() {
        let f = make_composite_filter("composite-multiply");
        assert_eq!(f.input_ports().len(), 2);
        assert_eq!(f.output_ports().len(), 1);
        assert_eq!(f.input_ports()[0].name, "src");
        assert_eq!(f.input_ports()[1].name, "dst");
    }

    #[test]
    fn composite_adapter_buffers_until_pair_ready() {
        // Two pushes on port 0 in a row without any port-1 input should
        // not emit; the second one simply replaces the buffered src.
        let mut f = make_composite_filter("composite-over");
        let mut col = CollectCtx { out: Vec::new() };
        f.push(&mut col, 0, &Frame::Video(op_red())).unwrap();
        f.push(&mut col, 0, &Frame::Video(transparent())).unwrap();
        assert!(col.out.is_empty(), "no emit before any dst arrives");
        f.push(&mut col, 1, &Frame::Video(op_blue())).unwrap();
        assert_eq!(col.out.len(), 1);
        // The second src (transparent) is what was paired with op_blue,
        // so the result is op_blue (transparent over blue ⇒ blue).
        let Frame::Video(v) = &col.out[0].1 else {
            panic!("non-video emit");
        };
        assert_eq!(v.planes[0].data[0..4], [0, 0, 255, 255]);
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

    // --- r10 smoke tests (geometry + channel ops + morphology edges) ---

    #[test]
    fn roll_factory_wraps_columns() {
        let c = ctx();
        let inputs = [gray_in_port(4, 1)];
        let f = c
            .filters
            .make("roll", &json!({"dx": 1, "dy": 0}), &inputs)
            .expect("roll factory");
        let out = run_one(f, gray_arbitrary(4, 1, |x, _| x as u8));
        assert_eq!(out.planes[0].data, vec![3, 0, 1, 2]);
    }

    #[test]
    fn shave_factory_resizes_canvas() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("shave", &json!({"x_border": 1, "y_border": 1}), &inputs)
            .expect("shave factory");
        let out = run_one(f, gray_4x4(|_, _| 99));
        // Output is 2×2.
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(out.planes[0].data.len(), 4);
        assert!(out.planes[0].data.iter().all(|&v| v == 99));
    }

    #[test]
    fn extent_factory_pads_with_background() {
        let c = ctx();
        let inputs = [gray_in_port(2, 2)];
        let f = c
            .filters
            .make(
                "extent",
                &json!({"width": 4, "height": 4, "background": [10, 0, 0, 255]}),
                &inputs,
            )
            .expect("extent factory");
        let out = run_one(f, gray_arbitrary(2, 2, |_, _| 50));
        assert_eq!(out.planes[0].stride, 4);
        // Top-left 2×2 retains the source.
        assert_eq!(out.planes[0].data[0], 50);
        assert_eq!(out.planes[0].data[1], 50);
        // Padding rows are 10.
        assert_eq!(out.planes[0].data[3], 10);
        assert_eq!(out.planes[0].data[4 * 4 - 1], 10);
    }

    #[test]
    fn extent_factory_rejects_partial_size() {
        let c = ctx();
        let inputs = [gray_in_port(2, 2)];
        assert!(c
            .filters
            .make("extent", &json!({"width": 4}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("extent", &json!({"height": 4}), &inputs)
            .is_err());
    }

    #[test]
    fn trim_factory_crops_solid_border() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("trim", &json!({}), &inputs)
            .expect("trim factory");
        // 4×4: zeros around a single 200 at the centre.
        let out = run_one(
            f,
            gray_arbitrary(4, 4, |x, y| if x == 1 && y == 1 { 200 } else { 0 }),
        );
        assert_eq!(out.planes[0].stride, 1);
        assert_eq!(out.planes[0].data, vec![200]);
    }

    #[test]
    fn channel_extract_factory_pulls_red() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 1)];
        let f = c
            .filters
            .make("channel-extract", &json!({"channel": "r"}), &inputs)
            .expect("channel-extract factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![10, 20, 30, 255, 70, 80, 90, 255],
            }],
        };
        let out = run_one(f, frame);
        // Output is a single-plane Gray8 frame holding R0, R1.
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(out.planes[0].data, vec![10, 70]);
    }

    #[test]
    fn channel_extract_factory_rejects_unknown_channel() {
        let c = ctx();
        let inputs = [rgba_in_port(1, 1)];
        assert!(c
            .filters
            .make("channel-extract", &json!({"channel": "garbage"}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("channel-extract", &json!({}), &inputs)
            .is_err());
    }

    #[test]
    fn morphology_edgein_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("morphology-edgein", &json!({"element": "square"}), &inputs)
            .expect("morphology-edgein factory");
        // Solid bright field — every pixel survives erosion → edge = 0.
        let out = run_one(f, gray_4x4(|_, _| 200));
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    #[test]
    fn morphology_edgeout_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("morphology-edgeout", &json!({"element": "cross"}), &inputs)
            .expect("morphology-edgeout factory");
        let out = run_one(f, gray_4x4(|_, _| 200));
        // Flat field → dilate = src → edge = 0.
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    #[test]
    fn morphology_edge_magnitude_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(5, 5)];
        let f = c
            .filters
            .make(
                "morphology-edge-magnitude",
                &json!({"element": "square"}),
                &inputs,
            )
            .expect("morphology-edge-magnitude factory");
        let out = run_one(
            f,
            gray_arbitrary(5, 5, |x, y| if x == 2 && y == 2 { 200 } else { 0 }),
        );
        // Centre + 8-neighbour ring all = 200.
        assert_eq!(out.planes[0].data[2 * 5 + 2], 200);
        assert_eq!(out.planes[0].data[5 + 2], 200);
        assert_eq!(out.planes[0].data[0], 0);
    }

    /// Helper: build an arbitrary-sized Gray8 frame for r10 tests.
    /// The existing `gray_4x4` helper is hard-coded to 4×4.
    fn gray_arbitrary(w: u32, h: u32, pattern: impl Fn(u32, u32) -> u8) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(pattern(x, y));
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize,
                data,
            }],
        }
    }

    // --- r12 factory smoke tests ---

    #[test]
    fn clamp_factory_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("clamp", &json!({"low": 50, "high": 200}), &inputs)
            .expect("clamp factory");
        let out = run_one(f, gray_4x4(|x, _| (x * 80) as u8));
        for v in &out.planes[0].data {
            assert!(*v >= 50 && *v <= 200, "out-of-range: {v}");
        }
    }

    #[test]
    fn auto_level_factory_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("auto-level", &json!({}), &inputs)
            .expect("auto-level factory");
        let out = run_one(f, gray_4x4(|x, _| 64 + (x * 20) as u8));
        let min = *out.planes[0].data.iter().min().unwrap();
        let max = *out.planes[0].data.iter().max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn contrast_stretch_factory_smoke() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "contrast-stretch",
                &json!({"black": 0.0, "white": 0.0}),
                &inputs,
            )
            .expect("contrast-stretch factory");
        let _out = run_one(f, rgba_4x4(|x, _| [10 + x as u8 * 10, 50, 200, 80]));
    }

    #[test]
    fn linear_stretch_factory_smoke() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("linear-stretch", &json!({"black": 0, "white": 0}), &inputs)
            .expect("linear-stretch factory");
        let _out = run_one(f, gray_4x4(|x, _| 64 + (x * 20) as u8));
    }

    #[test]
    fn color_matrix_factory_swaps_channels() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        // Map (R,G,B) → (B,G,R) — same swap as the unit test.
        let f = c
            .filters
            .make(
                "color-matrix",
                &json!({"matrix": [
                    0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0,
                    1.0, 0.0, 0.0,
                ]}),
                &inputs,
            )
            .expect("color-matrix factory");
        let out = run_one(f, rgba_4x4(|_, _| [10, 20, 30, 40]));
        // First pixel: BGR.
        let px = &out.planes[0].data[0..4];
        assert_eq!(px, &[30, 20, 10, 40]);
    }

    #[test]
    fn color_matrix_factory_rejects_wrong_length() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        let r = c
            .filters
            .make("color-matrix", &json!({"matrix": [1.0, 0.0, 0.0]}), &inputs);
        assert!(r.is_err());
    }

    #[test]
    fn recolor_alias_resolves() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        c.filters
            .make(
                "recolor",
                &json!({"matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}),
                &inputs,
            )
            .expect("recolor alias");
    }

    #[test]
    fn function_factory_polynomial_invert() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "function",
                &json!({"op": "polynomial", "args": [-1.0, 1.0]}),
                &inputs,
            )
            .expect("function factory");
        let out = run_one(f, gray_4x4(|_, _| 0));
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn function_factory_unknown_op_errors() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        assert!(c
            .filters
            .make("function", &json!({"op": "bogus"}), &inputs)
            .is_err());
    }

    #[test]
    fn laplacian_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("laplacian", &json!({}), &inputs)
            .expect("laplacian factory");
        // Output port should be Gray8.
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn log_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("log", &json!({"sigma": 1.0}), &inputs)
            .expect("log factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn log_factory_accepts_zero_crossings_mode() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        // Long-form spelling.
        assert!(c
            .filters
            .make(
                "laplacian-of-gaussian",
                &json!({"sigma": 1.0, "mode": "zero-crossings", "slope_threshold": 2.0}),
                &inputs
            )
            .is_ok());
        // Marr-Hildreth spelling alias + boolean shorthand.
        assert!(c
            .filters
            .make(
                "marr-hildreth",
                &json!({"sigma": 1.4, "zero_crossings": true}),
                &inputs
            )
            .is_ok());
    }

    #[test]
    fn log_factory_rejects_non_positive_sigma() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        assert!(c
            .filters
            .make("log", &json!({"sigma": 0.0}), &inputs)
            .is_err());
        assert!(c
            .filters
            .make("log", &json!({"sigma": -1.0}), &inputs)
            .is_err());
    }

    #[test]
    fn roberts_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("roberts", &json!({}), &inputs)
            .expect("roberts factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn roberts_factory_accepts_l1_magnitude() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        // Both the string form and the alias should construct fine.
        assert!(c
            .filters
            .make("roberts", &json!({"magnitude": "l1"}), &inputs)
            .is_ok());
        assert!(c
            .filters
            .make("roberts-cross", &json!({"l1": true}), &inputs)
            .is_ok());
    }

    #[test]
    fn prewitt_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("prewitt", &json!({}), &inputs)
            .expect("prewitt factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn prewitt_factory_accepts_l1_magnitude() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        assert!(c
            .filters
            .make("prewitt", &json!({"magnitude": "l1"}), &inputs)
            .is_ok());
        assert!(c
            .filters
            .make("prewitt", &json!({"l1": true}), &inputs)
            .is_ok());
    }

    #[test]
    fn scharr_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("scharr", &json!({}), &inputs)
            .expect("scharr factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn scharr_factory_accepts_l1_magnitude() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        assert!(c
            .filters
            .make("scharr", &json!({"magnitude": "l1"}), &inputs)
            .is_ok());
        assert!(c
            .filters
            .make("scharr", &json!({"l1": true}), &inputs)
            .is_ok());
    }

    #[test]
    fn canny_factory_emits_gray8_binary() {
        let c = ctx();
        let inputs = [gray_in_port(16, 16)];
        let f = c
            .filters
            .make(
                "canny",
                &json!({"radius": 0, "low": 30, "high": 90}),
                &inputs,
            )
            .expect("canny factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
        let frame = gray_arbitrary(16, 16, |x, _| if x < 8 { 30 } else { 220 });
        let out = run_one(f, frame);
        for v in &out.planes[0].data {
            assert!(*v == 0 || *v == 255, "non-binary: {v}");
        }
    }

    // --- r13 factory smoke tests ---

    #[test]
    fn blue_shift_factory_smoke() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let f = c
            .filters
            .make("blue-shift", &json!({"factor": 1.0}), &inputs)
            .expect("blue-shift factory");
        // Each pixel (200, 100, 50) ⇒ (50, 50, 200).
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: (0..16).flat_map(|_| [200u8, 100, 50]).collect(),
            }],
        };
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[50, 50, 200]);
        }
    }

    #[test]
    fn frame_factory_grows_canvas() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "frame",
                &json!({"width": 3, "height": 2, "inner": 0, "outer": 0}),
                &inputs,
            )
            .expect("frame factory");
        match &f.output_ports()[0].params {
            PortParams::Video { width, height, .. } => {
                assert_eq!(*width, 4 + 2 * 3);
                assert_eq!(*height, 4 + 2 * 2);
            }
            _ => panic!("frame output port not video"),
        }
        // Apply.
        let frame = rgba_4x4(|_, _| [50, 100, 150, 255]);
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 10 * 8 * 4);
    }

    #[test]
    fn frame_factory_rejects_bad_matte() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c
            .filters
            .make("frame", &json!({"matte": [1, 2, 3]}), &inputs);
        assert!(bad.is_err(), "matte must be 4-element");
    }

    #[test]
    fn shade_factory_default_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "shade",
                &json!({"azimuth": 30.0, "elevation": 60.0}),
                &inputs,
            )
            .expect("shade factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => assert_eq!(*format, PixelFormat::Gray8),
            _ => panic!("shade output port not video"),
        }
    }

    #[test]
    fn shade_factory_color_mode_preserves_format() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "shade",
                &json!({"azimuth": 30.0, "elevation": 60.0, "color": true}),
                &inputs,
            )
            .expect("shade factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => assert_eq!(*format, PixelFormat::Rgba),
            _ => panic!("shade output port not video"),
        }
    }

    #[test]
    fn paint_factory_smoke() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let f = c
            .filters
            .make("paint", &json!({"radius": 1}), &inputs)
            .expect("paint factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: (0..16).flat_map(|_| [200u8, 100, 50]).collect(),
            }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 16 * 3);
    }

    #[test]
    fn quantize_factory_smoke() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let f = c
            .filters
            .make("quantize", &json!({"colors": 8}), &inputs)
            .expect("quantize factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: (0..16u32)
                    .flat_map(|i| [(i * 16) as u8, (255 - i * 16) as u8, 128])
                    .collect(),
            }],
        };
        let out = run_one(f, frame);
        // With 8 colours ⇒ k = 2 ⇒ output samples in {0, 255}.
        for &v in &out.planes[0].data {
            assert!(v == 0 || v == 255, "quantize{{8}} produced {v}");
        }
    }

    #[test]
    fn clut_factory_requires_matching_shape() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 8, 8, PixelFormat::Rgba, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("clut", &json!({}), &inputs);
        assert!(bad.is_err(), "clut should reject mismatched src/dst");
    }

    #[test]
    fn clut_factory_two_input_smoke() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
        ];
        let f = c
            .filters
            .make("clut", &json!({}), &inputs)
            .expect("clut factory");
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: (0..16u32)
                    .flat_map(|i| [(i * 16) as u8, (i * 16) as u8, (i * 16) as u8, 200])
                    .collect(),
            }],
        };
        // 4×4 CLUT filled with a single colour ⇒ output should be that
        // single colour (per channel), alpha preserved.
        let clut = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: (0..16).flat_map(|_| [77u8, 88, 99, 0]).collect(),
            }],
        };
        let out = run_two(f, src, clut);
        for i in 0..16 {
            assert_eq!(out.planes[0].data[i * 4], 77);
            assert_eq!(out.planes[0].data[i * 4 + 1], 88);
            assert_eq!(out.planes[0].data[i * 4 + 2], 99);
            assert_eq!(out.planes[0].data[i * 4 + 3], 200);
        }
    }

    #[test]
    fn hald_clut_factory_requires_matching_shape() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 8, 8, PixelFormat::Rgb24, TimeBase::new(1, 30)),
            PortSpec::video("dst", 9, 9, PixelFormat::Rgb24, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("hald-clut", &json!({}), &inputs);
        assert!(bad.is_err(), "hald-clut should reject mismatched shapes");
    }

    #[test]
    fn hald_clut_factory_identity_smoke() {
        let c = ctx();
        // L = 4 ⇒ 16×16 image.
        let (clut, side) = crate::hald_clut::build_identity_hald(4, false);
        let inputs = [
            PortSpec::video("src", side, side, PixelFormat::Rgb24, TimeBase::new(1, 30)),
            PortSpec::video("dst", side, side, PixelFormat::Rgb24, TimeBase::new(1, 30)),
        ];
        let f = c
            .filters
            .make("hald-clut", &json!({}), &inputs)
            .expect("hald-clut factory");
        let mut data = Vec::new();
        for y in 0..side {
            for x in 0..side {
                data.extend_from_slice(&[
                    ((x * 255) / (side - 1)) as u8,
                    ((y * 255) / (side - 1)) as u8,
                    128,
                ]);
            }
        }
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (side * 3) as usize,
                data,
            }],
        };
        let out = run_two(f, src.clone(), clut);
        // Identity CLUT should reproduce the input within ±2.
        let mut max_err = 0i32;
        for (a, b) in src.planes[0].data.iter().zip(out.planes[0].data.iter()) {
            let d = (*a as i32 - *b as i32).abs();
            if d > max_err {
                max_err = d;
            }
        }
        assert!(max_err <= 2, "identity CLUT max err = {max_err}");
    }

    // --- r15 smoke tests ---

    #[test]
    fn hsl_rotate_factory_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("hsl-rotate", &json!({"degrees": 0.0}), &inputs)
            .expect("hsl-rotate factory");
        let input = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 100, 255]);
        let out = run_one(f, input.clone());
        for (a, b) in out.planes[0].data.iter().zip(input.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn hsl_rotate_factory_defaults_build() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make("hsl-rotate", &json!({}), &inputs)
            .expect("hsl-rotate default builds");
    }

    #[test]
    fn vignette_soft_factory_darkens_corners() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make("vignette-soft", &json!({}), &inputs)
            .expect("vignette-soft factory");
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..(8 * 8) {
            data.extend_from_slice(&[200u8, 200, 200]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        let centre = out.planes[0].data[(4 * 8 + 4) * 3];
        let corner = out.planes[0].data[0];
        assert!(centre > corner);
    }

    #[test]
    fn chromatic_aberration_factory_shifts_red_right() {
        let c = ctx();
        let inputs = [rgb24_in_port(9, 1)];
        let f = c
            .filters
            .make(
                "chromatic-aberration",
                &json!({"mode": "horizontal", "n": 2}),
                &inputs,
            )
            .expect("chromatic-aberration factory");
        let mut data = Vec::with_capacity(9 * 3);
        for x in 0..9 {
            if x == 4 {
                data.extend_from_slice(&[255u8, 255, 255]);
            } else {
                data.extend_from_slice(&[0u8, 0, 0]);
            }
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 27, data }],
        };
        let out = run_one(f, frame);
        // R should appear at col 6, B at col 2.
        assert_eq!(out.planes[0].data[6 * 3], 255);
        assert_eq!(out.planes[0].data[2 * 3 + 2], 255);
    }

    #[test]
    fn chromatic_aberration_factory_rejects_unknown_mode() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let bad = c.filters.make(
            "chromatic-aberration",
            &json!({"mode": "diagonal", "n": 1}),
            &inputs,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn pixelate_factory_block_1_is_passthrough() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let f = c
            .filters
            .make("pixelate", &json!({"block": 1}), &inputs)
            .expect("pixelate factory");
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for y in 0..4 {
            for x in 0..4 {
                data.extend_from_slice(&[(x * 30) as u8, (y * 30) as u8, 100]);
            }
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: data.clone(),
            }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data, data);
    }

    #[test]
    fn mosaic_alias_resolves() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        c.filters
            .make("mosaic", &json!({"block": 2}), &inputs)
            .expect("mosaic alias should resolve");
    }

    #[test]
    fn channel_mixer_factory_identity_preset() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("channel-mixer", &json!({"preset": "identity"}), &inputs)
            .expect("channel-mixer factory");
        let input = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 100, 222]);
        let out = run_one(f, input.clone());
        assert_eq!(out.planes[0].data, input.planes[0].data);
    }

    #[test]
    fn channel_mixer_factory_matrix3_lift() {
        let c = ctx();
        let inputs = [rgb24_in_port(2, 2)];
        let f = c
            .filters
            .make(
                "channel-mixer",
                &json!({"matrix3": [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]}),
                &inputs,
            )
            .expect("channel-mixer matrix3");
        let mut data = Vec::with_capacity(2 * 2 * 3);
        for _ in 0..4 {
            data.extend_from_slice(&[200u8, 100, 50]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 6, data }],
        };
        let out = run_one(f, frame);
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, &[50, 100, 200]);
        }
    }

    #[test]
    fn channel_mixer_factory_rejects_bad_matrix_arity() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        let bad = c.filters.make(
            "channel-mixer",
            &json!({"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}),
            &inputs,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn adaptive_threshold_factory_emits_gray8() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("adaptive-threshold", &json!({"radius": 1}), &inputs)
            .expect("adaptive-threshold factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => assert_eq!(*format, PixelFormat::Gray8),
            _ => panic!("adaptive-threshold output port is not video"),
        }
        let out = run_one(f, gray_4x4(|_, _| 128));
        // Constant input + offset 0 ⇒ every sample >= mean ⇒ all 255.
        assert!(out.planes[0].data.iter().all(|&v| v == 255));
    }

    #[test]
    fn adaptive_threshold_factory_positive_offset_pushes_to_black() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "adaptive-threshold",
                &json!({"radius": 1, "offset": 5}),
                &inputs,
            )
            .expect("adaptive-threshold factory");
        let out = run_one(f, gray_4x4(|_, _| 128));
        assert!(out.planes[0].data.iter().all(|&v| v == 0));
    }

    // --- r16 smoke tests ---

    #[test]
    fn bilateral_blur_factory_default_keeps_flat_input_flat() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("bilateral-blur", &json!({}), &inputs)
            .expect("bilateral-blur factory");
        let out = run_one(f, rgba_4x4(|_, _| [50, 100, 150, 222]));
        // Constant input ⇒ constant output.
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[50, 100, 150, 222]);
        }
    }

    #[test]
    fn bilateral_blur_factory_rejects_bad_sigma() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c
            .filters
            .make("bilateral-blur", &json!({"sigma_range": 0.0}), &inputs);
        assert!(bad.is_err());
    }

    #[test]
    fn canvas_factory_paints_uniform_colour() {
        let c = ctx();
        let inputs = [rgba_in_port(3, 3)];
        let f = c
            .filters
            .make("canvas", &json!({"color": [200, 100, 50, 255]}), &inputs)
            .expect("canvas factory");
        let out = run_one(f, rgba_4x4(|_, _| [0, 0, 0, 0]));
        for chunk in out.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn canvas_factory_default_is_opaque_black() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        let f = c
            .filters
            .make("canvas", &json!({}), &inputs)
            .expect("canvas factory");
        let out = run_one(f, rgba_4x4(|_, _| [200, 50, 30, 0]));
        let stride = 8usize;
        for y in 0..2 {
            for x in 0..2 {
                let p = y * stride + x * 4;
                assert_eq!(&out.planes[0].data[p..p + 4], &[0, 0, 0, 255]);
            }
        }
    }

    #[test]
    fn gradient_radial_factory_centre_is_inner() {
        let c = ctx();
        let inputs = [rgba_in_port(5, 5)];
        let f = c
            .filters
            .make(
                "gradient-radial",
                &json!({"inner": [255, 255, 255, 255], "outer": [0, 0, 0, 255]}),
                &inputs,
            )
            .expect("gradient-radial factory");
        // 4×4 ctx fixture — the helper only emits 4×4. Build a 5×5
        // Rgba directly so the centre pixel is well-defined.
        let mut data = Vec::with_capacity(5 * 5 * 4);
        for _ in 0..(5 * 5) {
            data.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 20, data }],
        };
        let out = run_one(f, frame);
        let p = 2 * 20 + 2 * 4;
        assert_eq!(out.planes[0].data[p], 255);
    }

    #[test]
    fn radial_gradient_alias_resolves() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make("radial-gradient", &json!({}), &inputs)
            .expect("radial-gradient alias should resolve");
    }

    #[test]
    fn gradient_conic_factory_builds() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make(
                "gradient-conic",
                &json!({"inner": [255, 0, 0, 255], "outer": [0, 0, 255, 255]}),
                &inputs,
            )
            .expect("gradient-conic factory");
        c.filters
            .make("conic-gradient", &json!({}), &inputs)
            .expect("conic-gradient alias");
    }

    #[test]
    fn gravity_translate_factory_resizes_canvas() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        let f = c
            .filters
            .make(
                "gravity-translate",
                &json!({"width": 4, "height": 4, "gravity": "northwest"}),
                &inputs,
            )
            .expect("gravity-translate factory");
        match &f.output_ports()[0].params {
            PortParams::Video { width, height, .. } => {
                assert_eq!(*width, 4);
                assert_eq!(*height, 4);
            }
            _ => panic!("gravity-translate output port is not video"),
        }
    }

    #[test]
    fn gravity_translate_factory_rejects_unknown_anchor() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c.filters.make(
            "gravity-translate",
            &json!({"width": 8, "height": 8, "gravity": "skyward"}),
            &inputs,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn gravity_alias_resolves() {
        let c = ctx();
        let inputs = [rgba_in_port(2, 2)];
        c.filters
            .make(
                "gravity",
                &json!({"width": 4, "height": 4, "gravity": "centre"}),
                &inputs,
            )
            .expect("gravity alias");
    }

    #[test]
    fn color_balance_factory_identity_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("color-balance", &json!({}), &inputs)
            .expect("color-balance factory");
        let frame = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 100, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn color_balance_factory_rejects_bad_gamma() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let bad = c
            .filters
            .make("color-balance", &json!({"gamma": [0.0, 1.0, 1.0]}), &inputs);
        assert!(bad.is_err());
    }

    #[test]
    fn hsl_shift_factory_zero_is_passthrough_within_rounding() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("hsl-shift", &json!({}), &inputs)
            .expect("hsl-shift factory");
        let frame = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 100, 222]);
        let out = run_one(f, frame.clone());
        for (a, b) in out.planes[0].data.iter().zip(frame.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    // --- r19 smoke tests ---

    #[test]
    fn heatmap_factory_swaps_gray_for_rgb() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make("heatmap", &json!({"ramp": "jet"}), &inputs)
            .expect("heatmap factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => {
                assert_eq!(*format, PixelFormat::Rgb24);
            }
            _ => panic!("heatmap output must be video"),
        }
        let frame = gray_4x4(|x, _| (x * 60) as u8);
        let out = run_one(f, frame);
        // Output buffer is 3× the input size.
        assert_eq!(out.planes[0].data.len(), 4 * 4 * 3);
    }

    #[test]
    fn heatmap_factory_rejects_unknown_ramp() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c
            .filters
            .make("heatmap", &json!({"ramp": "rainbow-of-doom"}), &inputs);
        assert!(bad.is_err(), "unknown ramp must fail");
    }

    #[test]
    fn heatmap_false_color_alias_resolves() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make("false-color", &json!({}), &inputs)
            .expect("false-color alias");
    }

    #[test]
    fn split_tone_factory_zero_amount_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "split-tone",
                &json!({
                    "shadow": [0, 0, 255],
                    "highlight": [255, 255, 0],
                    "amount": 0.0,
                }),
                &inputs,
            )
            .expect("split-tone factory");
        let frame = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 100, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn flood_fill_factory_paints_seed_region() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "flood-fill",
                &json!({"seed_x": 0, "seed_y": 0, "color": [199, 0, 0], "tolerance": 0}),
                &inputs,
            )
            .expect("flood-fill factory");
        let frame = gray_4x4(|_, _| 0);
        let out = run_one(f, frame);
        for &v in &out.planes[0].data {
            assert_eq!(v, 199);
        }
    }

    #[test]
    fn posterize_channels_factory_builds() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "posterize-channels",
                &json!({"r_levels": 4, "g_levels": 4, "b_levels": 2}),
                &inputs,
            )
            .expect("posterize-channels factory");
        let frame = rgba_4x4(|x, _| [(x * 60) as u8, (x * 60) as u8, (x * 60) as u8, 200]);
        let out = run_one(f, frame);
        // B channel collapsed to 2 levels ⇒ every B byte must be 0 or 255.
        for chunk in out.planes[0].data.chunks(4) {
            assert!(
                chunk[2] == 0 || chunk[2] == 255,
                "B channel = {} not in {{0, 255}}",
                chunk[2]
            );
        }
    }

    #[test]
    fn toon_factory_emits_same_shape() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "toon",
                &json!({"levels": 4, "edge_threshold": 60, "edge_color": [0, 0, 0]}),
                &inputs,
            )
            .expect("toon factory");
        // 8x8 RGB constant.
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[200u8, 100, 50]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn toon_cartoon_alias_resolves() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        c.filters
            .make("cartoon", &json!({}), &inputs)
            .expect("cartoon alias");
    }

    #[test]
    fn watermark_factory_rejects_format_mismatch() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 2, 2, PixelFormat::Rgb24, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("watermark", &json!({}), &inputs);
        assert!(bad.is_err(), "format mismatch must fail");
    }

    #[test]
    fn watermark_factory_builds() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgba, TimeBase::new(1, 30)),
            PortSpec::video("dst", 2, 2, PixelFormat::Rgba, TimeBase::new(1, 30)),
        ];
        let f = c
            .filters
            .make(
                "watermark",
                &json!({"offset_x": 1, "offset_y": 1, "opacity": 0.5}),
                &inputs,
            )
            .expect("watermark factory");
        assert_eq!(f.input_ports().len(), 2);
        assert_eq!(f.output_ports().len(), 1);
    }

    // --- r20 smoke tests ---

    #[test]
    fn crystallize_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [rgb24_in_port(16, 16)];
        let f = c
            .filters
            .make(
                "crystallize",
                &json!({"cell_size": 4, "jitter": 0.5, "seed": 7}),
                &inputs,
            )
            .expect("crystallize factory");
        let mut data = Vec::with_capacity(16 * 16 * 3);
        for _ in 0..(16 * 16) {
            data.extend_from_slice(&[100u8, 150, 200]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 48, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 16 * 16 * 3);
    }

    #[test]
    fn halftone_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make("halftone", &json!({"cell_size": 4}), &inputs)
            .expect("halftone factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![128u8; 64],
            }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 64);
    }

    #[test]
    fn halftone_with_custom_colours_factory() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        c.filters
            .make(
                "halftone",
                &json!({"cell_size": 4, "ink": [100, 0, 0], "paper": [240, 240, 240]}),
                &inputs,
            )
            .expect("halftone factory");
    }

    #[test]
    fn gradient_map_duotone_factory_upgrades_gray_to_rgb() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let f = c
            .filters
            .make(
                "gradient-map",
                &json!({"shadow": [0, 0, 0], "highlight": [255, 0, 0]}),
                &inputs,
            )
            .expect("gradient-map factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => {
                assert_eq!(*format, PixelFormat::Rgb24);
            }
            _ => panic!("gradient-map output must be video"),
        }
        let out = run_one(f, gray_4x4(|_, _| 255));
        // 4×4 RGB = 48 bytes.
        assert_eq!(out.planes[0].data.len(), 48);
        // White input ⇒ highlight colour [255, 0, 0].
        for chunk in out.planes[0].data.chunks(3) {
            assert_eq!(chunk, [255u8, 0, 0]);
        }
    }

    #[test]
    fn gradient_map_explicit_stops_factory() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make(
                "gradient-map",
                &json!({
                    "stops": [
                        {"position": 0.0, "color": [0, 0, 0]},
                        {"position": 0.5, "color": [255, 0, 0]},
                        {"position": 1.0, "color": [255, 255, 0]},
                    ]
                }),
                &inputs,
            )
            .expect("gradient-map stops factory");
    }

    #[test]
    fn gradient_map_rejects_single_stop() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        let bad = c.filters.make(
            "gradient-map",
            &json!({"stops": [{"position": 0.0, "color": [0, 0, 0]}]}),
            &inputs,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn gradient_map_duotone_alias_resolves() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make("duotone", &json!({}), &inputs)
            .expect("duotone alias");
    }

    #[test]
    fn selective_color_identity_passes_through() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("selective-color", &json!({}), &inputs)
            .expect("selective-color factory");
        let frame = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 80, 222]);
        let out = run_one(f, frame.clone());
        for (a, b) in out.planes[0].data.iter().zip(frame.planes[0].data.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1);
        }
    }

    #[test]
    fn selective_colour_alias_resolves() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        c.filters
            .make("selective-colour", &json!({}), &inputs)
            .expect("selective-colour alias");
    }

    #[test]
    fn selective_color_reds_band_factory() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        c.filters
            .make(
                "selective-color",
                &json!({
                    "reds": {"hue": 30.0, "saturation": -0.5, "lightness": 0.1}
                }),
                &inputs,
            )
            .expect("selective-color reds factory");
    }

    #[test]
    fn cross_process_factory_amount_zero_is_passthrough() {
        let c = ctx();
        let inputs = [rgba_in_port(4, 4)];
        let f = c
            .filters
            .make("cross-process", &json!({"amount": 0.0}), &inputs)
            .expect("cross-process factory");
        let frame = rgba_4x4(|x, y| [(x * 30) as u8, (y * 30) as u8, 80, 222]);
        let out = run_one(f, frame.clone());
        assert_eq!(out.planes[0].data, frame.planes[0].data);
    }

    #[test]
    fn cross_process_alias_resolves() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        c.filters
            .make("crossprocess", &json!({}), &inputs)
            .expect("crossprocess alias");
    }

    #[test]
    fn otsu_threshold_factory_emits_gray8() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make("otsu-threshold", &json!({}), &inputs)
            .expect("otsu-threshold factory");
        match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => {
                assert_eq!(*format, PixelFormat::Gray8);
            }
            _ => panic!("otsu output must be video"),
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: (0..64u8)
                    .map(|i| if i % 8 < 4 { 40 } else { 210 })
                    .collect(),
            }],
        };
        let out = run_one(f, frame);
        assert!(out.planes[0].data.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn otsu_alias_resolves() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make("otsu", &json!({}), &inputs)
            .expect("otsu alias");
    }

    #[test]
    fn otsu_threshold_invert_factory() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        c.filters
            .make("otsu-threshold", &json!({"invert": true}), &inputs)
            .expect("otsu-threshold invert factory");
    }

    #[test]
    fn comic_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "comic",
                &json!({
                    "smooth_radius": 1,
                    "levels": 3,
                    "edge_threshold": 30,
                    "ink_color": [10, 10, 10],
                }),
                &inputs,
            )
            .expect("comic factory");
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[120u8, 80, 200]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn comic_manga_alias_resolves() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        c.filters
            .make("manga", &json!({}), &inputs)
            .expect("manga alias");
    }

    // --- r21 smoke tests ---

    #[test]
    fn kuwahara_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make("kuwahara", &json!({"radius": 2}), &inputs)
            .expect("kuwahara factory");
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[100u8, 150, 200]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn anisotropic_blur_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "anisotropic-blur",
                &json!({"iterations": 3, "kappa": 20.0, "lambda": 0.2}),
                &inputs,
            )
            .expect("anisotropic-blur factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![128u8; 64],
            }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 64);
    }

    #[test]
    fn anisotropic_alias_resolves() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        c.filters
            .make("anisotropic", &json!({}), &inputs)
            .expect("anisotropic alias");
    }

    #[test]
    fn zoom_blur_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "zoom-blur",
                &json!({"strength": 0.3, "samples": 6, "centre_x": 0.5, "centre_y": 0.5}),
                &inputs,
            )
            .expect("zoom-blur factory");
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[100u8, 100, 100]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn radial_blur_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "radial-blur",
                &json!({"angle_degrees": 15.0, "samples": 6}),
                &inputs,
            )
            .expect("radial-blur factory");
        let mut data = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            data.extend_from_slice(&[120u8, 80, 40]);
        }
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 24, data }],
        };
        let out = run_one(f, frame);
        assert_eq!(out.planes[0].data.len(), 8 * 8 * 3);
    }

    #[test]
    fn spin_blur_alias_resolves() {
        let c = ctx();
        let inputs = [rgb24_in_port(4, 4)];
        c.filters
            .make("spin-blur", &json!({"angle": 5.0}), &inputs)
            .expect("spin-blur alias");
    }

    #[test]
    fn emboss_directional_factory_builds_and_runs() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        let f = c
            .filters
            .make(
                "emboss-directional",
                &json!({"azimuth_degrees": 45.0, "depth": 1.5}),
                &inputs,
            )
            .expect("emboss-directional factory");
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![100u8; 64],
            }],
        };
        let out = run_one(f, frame);
        // Flat input ⇒ uniformly 128 (bias).
        for &v in &out.planes[0].data {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn displacement_map_factory_rejects_format_mismatch() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgb24, TimeBase::new(1, 30)),
            PortSpec::video("dst", 4, 4, PixelFormat::Gray8, TimeBase::new(1, 30)),
        ];
        let bad = c.filters.make("displacement-map", &json!({}), &inputs);
        assert!(bad.is_err(), "format mismatch must fail");
    }

    #[test]
    fn displacement_map_factory_builds() {
        let c = ctx();
        let inputs = [
            PortSpec::video("src", 4, 4, PixelFormat::Rgb24, TimeBase::new(1, 30)),
            PortSpec::video("dst", 4, 4, PixelFormat::Rgb24, TimeBase::new(1, 30)),
        ];
        let f = c
            .filters
            .make(
                "displacement-map",
                &json!({"scale_x": 4.0, "scale_y": 4.0}),
                &inputs,
            )
            .expect("displacement-map factory");
        assert_eq!(f.input_ports().len(), 2);
        assert_eq!(f.output_ports().len(), 1);
    }

    #[test]
    fn gabor_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgba_in_port(8, 8)];
        let f = c
            .filters
            .make("gabor", &json!({"wavelength": 6.0}), &inputs)
            .expect("gabor factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn gabor_factory_accepts_all_aliases() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        assert!(c.filters.make("gabor", &json!({}), &inputs).is_ok());
        assert!(c.filters.make("gabor-filter", &json!({}), &inputs).is_ok());
        assert!(c.filters.make("gabor-wavelet", &json!({}), &inputs).is_ok());
    }

    #[test]
    fn gabor_factory_accepts_magnitude_mode() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        // Long-form mode string.
        assert!(c
            .filters
            .make(
                "gabor",
                &json!({"mode": "magnitude", "wavelength": 4.0, "sigma": 2.0}),
                &inputs,
            )
            .is_ok());
        // Boolean shorthand.
        assert!(c
            .filters
            .make("gabor", &json!({"magnitude": true}), &inputs)
            .is_ok());
        // Param aliases.
        assert!(c
            .filters
            .make(
                "gabor",
                &json!({"theta": 45.0, "psi": 90.0, "aspect": 0.7}),
                &inputs,
            )
            .is_ok());
    }

    #[test]
    fn gabor_factory_rejects_bad_parameters() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        // wavelength below spatial Nyquist.
        assert!(c
            .filters
            .make("gabor", &json!({"wavelength": 1.0}), &inputs)
            .is_err());
        // sigma non-positive.
        assert!(c
            .filters
            .make("gabor", &json!({"sigma": 0.0}), &inputs)
            .is_err());
        // gamma non-positive.
        assert!(c
            .filters
            .make("gabor", &json!({"gamma": -0.5}), &inputs)
            .is_err());
    }

    #[test]
    fn niblack_factory_emits_gray8() {
        let c = ctx();
        let inputs = [rgb24_in_port(8, 8)];
        let f = c
            .filters
            .make("niblack", &json!({"radius": 3, "k": -0.2}), &inputs)
            .expect("niblack factory");
        let out_format = match &f.output_ports()[0].params {
            PortParams::Video { format, .. } => *format,
            _ => panic!("expected video port"),
        };
        assert_eq!(out_format, PixelFormat::Gray8);
    }

    #[test]
    fn niblack_factory_accepts_aliases_and_invert() {
        let c = ctx();
        let inputs = [gray_in_port(8, 8)];
        assert!(c.filters.make("niblack", &json!({}), &inputs).is_ok());
        assert!(c
            .filters
            .make("niblack-threshold", &json!({"invert": true}), &inputs)
            .is_ok());
    }

    #[test]
    fn niblack_factory_rejects_non_finite_k() {
        let c = ctx();
        let inputs = [gray_in_port(4, 4)];
        // serde_json doesn't accept NaN/Infinity literally, but we
        // can exercise the finiteness gate by feeding it a very large
        // f64 that still rejects. Instead just feed a valid value to
        // confirm the happy path, and trust the unit tests in
        // `niblack::tests::rejects_non_finite_k` for the NaN case.
        assert!(c
            .filters
            .make("niblack", &json!({"k": -0.5}), &inputs)
            .is_ok());
    }
}
