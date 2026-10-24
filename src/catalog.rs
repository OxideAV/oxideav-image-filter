//! One instance of every filter type in the crate, for the
//! filter × layout matrix tests and the fuzz target. Not part of the
//! public API (`#[doc(hidden)]`): the constructor arguments are test
//! defaults chosen so every filter runs on a small picture, not
//! recommended settings.

use crate::*;

/// A named, boxed single-input filter.
pub struct Entry {
    /// The filter's type name.
    pub name: &'static str,
    /// The filter, with representative parameters.
    pub filter: Box<dyn ImageFilter>,
}

/// A named, boxed two-input filter.
pub struct PairEntry {
    /// The filter's type name.
    pub name: &'static str,
    /// The filter, with representative parameters.
    pub filter: Box<dyn TwoInputImageFilter>,
}

fn e(name: &'static str, filter: impl ImageFilter + 'static) -> Entry {
    Entry {
        name,
        filter: Box::new(filter),
    }
}

fn p(name: &'static str, filter: impl TwoInputImageFilter + 'static) -> PairEntry {
    PairEntry {
        name,
        filter: Box::new(filter),
    }
}

/// Every single-input filter type (one instance each), in source order.
/// Geometry filters that need a target size are built for an 8×8
/// picture or smaller (crop / shave / extent fit inside 8×8; resize
/// targets 5×3 so odd sizes and both directions of scaling are hit).
pub fn all_single() -> Vec<Entry> {
    vec![
        e("AdaptiveThreshold", AdaptiveThreshold::new(1, 0)),
        e("Affine", Affine::identity()),
        e("AnisotropicBlur", AnisotropicBlur::new(1)),
        e("AutoGamma", AutoGamma::new()),
        e("AutoLevel", AutoLevel::new()),
        e("AutoTrim", AutoTrim::new()),
        e("BarrelInverse", BarrelInverse::new(0.0, 0.0, 0.0, 1.0)),
        e("BilateralBlur", BilateralBlur::new(1, 1.0, 20.0)),
        e("Bloom", Bloom::new(128, 1, 0.5)),
        e("BlueShift", BlueShift::new(1.5)),
        e("Blur", Blur::new(1)),
        e("BorderedFrame", BorderedFrame::uniform(1, [0, 0, 0, 255])),
        e("BrightnessContrast", BrightnessContrast::new(10.0, 10.0)),
        e("BwMix", BwMix::new(0.3, 0.6, 0.1)),
        e("Canny", Canny::new(1, 1.0, 20, 60)),
        e("Canvas", Canvas::new([1, 2, 3, 255])),
        e("ChannelExtract", ChannelExtract::new(Channel::Red)),
        e("ChannelExtract(U)", ChannelExtract::new(Channel::U)),
        e("ChannelMixer", ChannelMixer::identity()),
        e("Charcoal", Charcoal::new(0.5)),
        e("ChromaticAberration", ChromaticAberration::new(1, 0, -1, 0)),
        e("Clamp", Clamp::new(10, 200)),
        e("Clarity", Clarity::new(0.5)),
        e(
            "ColorBalance",
            ColorBalance::new([0.0; 3], [1.0; 3], [1.0; 3]),
        ),
        e("ColorMatrix", ColorMatrix::identity()),
        e("Colorize", Colorize::new([200, 100, 50, 255], 0.5)),
        e("Comic", Comic::new()),
        e("Composite(self)", Composite::new(CompositeOp::Over)),
        e(
            "ConnectedComponents",
            ConnectedComponents::new(4).with_area_threshold(3.0),
        ),
        e("ContrastStretch", ContrastStretch::new(1.0, 1.0)),
        e(
            "Convolve",
            Convolve::new(3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                .expect("3x3 identity kernel"),
        ),
        e("Crop", Crop::new(1, 1, 3, 3)),
        e("CrossProcess", CrossProcess::new(0.5)),
        e("Crystallize", Crystallize::new(2)),
        e("Curves", Curves::default()),
        e("Cyanotype", Cyanotype::new(0.5)),
        e("Cycle", Cycle::new(10)),
        e("Deskew", Deskew::new(128)),
        e("Despeckle", Despeckle::new(1)),
        e(
            "DistanceMorphology",
            DistanceMorphology::new(MorphOp::Dilate, 1.0),
        ),
        e("DistanceOutline", DistanceOutline::new(1.0, 2.0)),
        e("DistanceTransform", DistanceTransform::new(128)),
        e("Distort", Distort::new(0.0, 0.0)),
        e(
            "DistortImage",
            DistortImage::new(DistortMethod::ScaleRotateTranslate, vec![10.0])
                .with_virtual_pixel(VirtualPixel::Mirror),
        ),
        e("Dither", Dither::new()),
        e("Drago", Drago::new(0.85)),
        e("DropShadow", DropShadow::new(1, 1, 1)),
        e("Edge", Edge::new()),
        e("EdgeDetect", EdgeDetect::default()),
        e("Emboss", Emboss::new(1)),
        e("EmbossDirectional", EmbossDirectional::new(45.0)),
        e("Equalize", Equalize::new()),
        e(
            "EuclideanDistanceTransform",
            EuclideanDistanceTransform::new(128),
        ),
        e("Evaluate", Evaluate::new(EvaluateOp::Add, 10.0)),
        e("Exposure", Exposure::new(0.5)),
        e("Extent", Extent::new(8, 8)),
        e("Feather", Feather::new(1.0)),
        e("VoronoiTransform", VoronoiTransform::new(128)),
        e("ProximityFill", ProximityFill::new(128)),
        e("Flip", Flip::new()),
        e("FloodFill", FloodFill::new(0, 0, [9, 9, 9, 255], 10)),
        e("Flop", Flop::new()),
        e("Frame", Frame::new(1, 1, 0, 0)),
        e("FreiChen", FreiChen::new()),
        e(
            "Function",
            Function::new(FunctionOp::Polynomial, vec![1.0, 0.0]),
        ),
        e("Fx", Fx::new("u*0.8+0.1*(i%3)").expect("expression")),
        e("Gabor", Gabor::new()),
        e("Gamma", Gamma::new(2.2)),
        e(
            "GradientConic",
            GradientConic::new([0, 0, 0, 255], [255, 255, 255, 255]),
        ),
        e(
            "GradientMap",
            GradientMap::new(vec![
                crate::gradient_map::GradientStop {
                    position: 0.0,
                    color: [0, 0, 0],
                },
                crate::gradient_map::GradientStop {
                    position: 1.0,
                    color: [255, 255, 255],
                },
            ])
            .expect("two-stop ramp"),
        ),
        e(
            "GradientRadial",
            GradientRadial::new([0, 0, 0, 255], [255, 255, 255, 255]),
        ),
        e(
            "GravityTranslate",
            GravityTranslate::new(8, 8, Gravity::NorthWest),
        ),
        e("Grayscale", Grayscale::new()),
        e("Grayscale(gray8)", Grayscale::new().with_output_gray8(true)),
        e("Hable", Hable::new(1.0)),
        e("Halftone", Halftone::new(2)),
        e("Heatmap", Heatmap::default()),
        e("HoughCircles", HoughCircles::new(1, 2, 1)),
        e("HoughLines", HoughLines::new(10, 1)),
        e("HslRotate", HslRotate::new(90.0)),
        e("HslShift", HslShift::new(10.0, 0.1, 0.1)),
        e("Implode", Implode::new(0.3)),
        e("InnerShadow", InnerShadow::new(1, 1, 1)),
        e("Kmeans", Kmeans::new(3)),
        e("Kuwahara", Kuwahara::new(1)),
        e("Laplacian", Laplacian::new()),
        e("LaplacianOfGaussian", LaplacianOfGaussian::new()),
        e("Level", Level::new(10, 240, 1.0)),
        e("LinearStretch", LinearStretch::new(1, 1)),
        e("MaxRgb", MaxRgb::new()),
        e("Modulate", Modulate::new(1.1, 1.1, 10.0)),
        e(
            "Morphology",
            Morphology::new(MorphologyOp::Dilate, StructuringElement::Square3x3, 1),
        ),
        e(
            "MorphologyChain",
            MorphologyChain {
                first: Morphology::new(MorphologyOp::Dilate, StructuringElement::Cross3x3, 1),
                second: Morphology::new(MorphologyOp::Erode, StructuringElement::Cross3x3, 1),
            },
        ),
        e(
            "MorphologyEdge",
            MorphologyEdge::edge_in(StructuringElement::Square3x3),
        ),
        e("MotionBlur", MotionBlur::new(1, 1.0, 0.0)),
        e("Negate", Negate::new()),
        e("Niblack", Niblack::new()),
        e("AddNoise", AddNoise::new(NoiseType::Gaussian).with_seed(1)),
        e("Normalize", Normalize::new(0.0, 0.0)),
        e(
            "OrderedDither",
            OrderedDither::new(ThresholdMap::bayer(4)).with_levels(3),
        ),
        e("OtsuThreshold", OtsuThreshold::new()),
        e("Paint", Paint::new(1)),
        e(
            "Perspective",
            Perspective::new(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            ),
        ),
        e("Pixelate", Pixelate::new(2)),
        e("Polar", Polar::forward()),
        e("Posterize", Posterize::new(4)),
        e("PosterizeChannels", PosterizeChannels::new(4, 4, 4)),
        e("Prewitt", Prewitt::new()),
        e("Quantize", Quantize::new(4)),
        e("RadialBlur", RadialBlur::new(5.0)),
        e("Reinhard", Reinhard::new(0.18)),
        e("ReinhardExtended", ReinhardExtended::new(2.0)),
        e("ReinhardLocal", ReinhardLocal::new(0.18)),
        e("Resize", Resize::new(5, 3)),
        e("Roberts", Roberts::new()),
        e("Roll", Roll::new(1, -1)),
        e("Rotate", Rotate::new(90.0)),
        e("Scharr", Scharr::new()),
        e("SelectiveColor", SelectiveColor::new()),
        e("Sepia", Sepia::new(0.8)),
        e("Shade", Shade::new(30.0, 30.0)),
        e("Shade(color)", Shade::new(30.0, 30.0).with_color(true)),
        e("ShadowHighlight", ShadowHighlight::new(0.3, 0.3)),
        e("Sharpen", Sharpen::new(1, 1.0)),
        e("Shave", Shave::new(1, 1)),
        e("SigmoidalContrast", SigmoidalContrast::new(3.0, 0.5)),
        e("SignedDistanceField", SignedDistanceField::new(128)),
        e("Sketch", Sketch::new(1, 1.0, 45.0)),
        e("Solarize", Solarize::new(128)),
        e(
            "SplitTone",
            SplitTone::new([20, 30, 90], [250, 200, 100], 0.0, 0.5),
        ),
        e(
            "SparseColor",
            SparseColor::new(
                SparseMethod::Shepards,
                vec![
                    SparsePoint::new(1.0, 1.0, [1.0, 0.0, 0.0]),
                    SparsePoint::new(5.0, 4.0, [0.0, 0.5, 1.0]),
                ],
            )
            .expect("two points"),
        ),
        e("Spread", Spread::new(1)),
        e("SrgbTransform", SrgbTransform::decode()),
        e("Srt", Srt::new((0.5, 0.5), (1.0, 1.0), 0.0, (0.0, 0.0))),
        e("Statistic", Statistic::new(StatisticOp::Median, 3, 3)),
        e("Swirl", Swirl::new(45.0)),
        e("Temperature", Temperature::new(0.3)),
        e("Threshold", Threshold::new(128)),
        e("TiltShift", TiltShift::new(0.5, 0.3, 0.2, 1, 1.0)),
        e("Tint", Tint::new([200, 100, 50, 255], 0.5)),
        e("Toon", Toon::new(4, 64, [0, 0, 0])),
        e("Transpose", Transpose::new()),
        e("Transverse", Transpose::transverse()),
        e("Trim", Trim::new()),
        e("Unsharp", Unsharp::new(1, 1.0, 1.0, 0)),
        e("Vibrance", Vibrance::new(0.5)),
        e("Vignette", Vignette::new(0.5, 0.5, 0.5, 0.3)),
        e("VignetteSoft", VignetteSoft::new(0.5, 0.5, 0.3, 0.8)),
        e("Wave", Wave::new(1.0, 4.0)),
        e(
            "WeightedDistanceTransform",
            WeightedDistanceTransform::new(1.0),
        ),
        e("ZoomBlur", ZoomBlur::new(0.1)),
    ]
}

/// Every two-input filter type (one instance each), in source order.
pub fn all_pair() -> Vec<PairEntry> {
    vec![
        p("Clut", Clut::new()),
        p("Composite", Composite::new(CompositeOp::Over)),
        p("Difference", Difference::new()),
        p("DisplacementMap", DisplacementMap::new(1.0)),
        p("HaldClut", HaldClut::new()),
        p("Stegano", Stegano::new(0)),
        p("Stereo", Stereo::new()),
        p("Watermark", Watermark::new(0, 0, 2, 2)),
    ]
}
