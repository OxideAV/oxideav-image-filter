#![no_main]

//! Panic-freedom fuzz target for the distortion engine: an arbitrary
//! method with arbitrary (finite or not) arguments, edge rule, point
//! interpolation or area filter with arbitrary overrides, viewport,
//! best fit and output scaling, over a small picture of arbitrary
//! samples. Construction of the mapping may fail with a typed error; a
//! built mapping must render, and the output must have the extent the
//! filter declares. The cylindrical resize path runs on the same bytes.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::deep::FloatImage;
use oxideav_image_filter::{
    DistortImage, DistortMethod, FilterOptions, ImageFilter, PixelInterpolate, ResampleFilter,
    Resize, VideoStreamParams, VirtualPixel,
};

const METHODS: &[&str] = &[
    "Affine",
    "AffineProjection",
    "SRT",
    "Perspective",
    "PerspectiveProjection",
    "BilinearForward",
    "BilinearReverse",
    "Polynomial",
    "Arc",
    "Polar",
    "DePolar",
    "Barrel",
    "BarrelInverse",
    "Shepards",
    "Cylinder2Plane",
    "Plane2Cylinder",
    "Resize",
];

const EDGES: &[&str] = &[
    "edge",
    "background",
    "black",
    "white",
    "gray",
    "transparent",
    "tile",
    "mirror",
    "horizontaltile",
    "verticaltile",
    "horizontaltileedge",
    "verticaltileedge",
    "checkertile",
    "random",
    "dither",
];

const POINTS: &[&str] = &[
    "integer",
    "nearest",
    "average",
    "average9",
    "average16",
    "bilinear",
    "catrom",
    "spline",
    "mesh",
    "background",
];

struct Bytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bytes<'_> {
    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// A number: mostly small, sometimes huge, zero, negative or not
    /// finite.
    fn number(&mut self) -> f64 {
        let v = self.byte() as f64;
        match self.byte() % 12 {
            0 => 0.0,
            1 => -v,
            2 => v * 1e6,
            3 => v / 255.0,
            4 => f64::NAN,
            5 => f64::INFINITY,
            6 => v * 1e-9,
            7 => -v / 16.0,
            _ => v / 4.0,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut b = Bytes { data, pos: 0 };
    let w = 1 + b.byte() as u32 % 12;
    let h = 1 + b.byte() as u32 % 12;
    let channels = [1usize, 3, 4][b.byte() as usize % 3];
    let method = DistortMethod::from_name(METHODS[b.byte() as usize % METHODS.len()]).unwrap();
    let n_args = b.byte() as usize % 26;
    let args: Vec<f64> = (0..n_args).map(|_| b.number()).collect();
    let mut filter = DistortImage::new(method, args)
        .with_virtual_pixel(VirtualPixel::from_name(EDGES[b.byte() as usize % EDGES.len()]).unwrap())
        .with_bestfit(b.byte() & 1 == 1)
        .with_scale(b.number())
        .with_shepards_power(b.number())
        .with_background([0.2, 0.4, 0.6, (b.byte() % 3) as f32 / 2.0]);
    let filter_kind = ResampleFilter::ALL[b.byte() as usize % ResampleFilter::ALL.len()];
    let mut opts = FilterOptions::new();
    if b.byte() & 1 == 1 {
        opts = opts.with_blur(b.number() as f32);
    }
    if b.byte() & 1 == 1 {
        opts = opts.with_support(b.number() as f32);
    }
    if b.byte() & 1 == 1 {
        opts = opts.with_lobes(b.byte() as u32);
    }
    if b.byte() & 1 == 1 {
        opts = opts
            .with_window(ResampleFilter::ALL[b.byte() as usize % ResampleFilter::ALL.len()]);
    }
    if b.byte() & 1 == 1 {
        opts = opts.with_sigma(b.number() as f32).with_b(b.number() as f32);
    }
    filter = if b.byte() & 1 == 1 {
        filter.with_interpolate(
            PixelInterpolate::from_name(POINTS[b.byte() as usize % POINTS.len()]).unwrap(),
        )
    } else {
        filter.with_ewa(filter_kind).with_filter_options(opts)
    };
    if b.byte() & 1 == 1 {
        filter = filter.with_viewport(
            1 + b.byte() as u32 % 24,
            1 + b.byte() as u32 % 24,
            b.byte() as i64 - 128,
            (b.byte() as i64 - 128) * 1000,
        );
    }
    let samples: Vec<f32> = (0..w as usize * h as usize * channels)
        .map(|_| match b.byte() {
            255 => f32::NAN,
            254 => f32::INFINITY,
            253 => -1e30,
            v => v as f32 / 200.0,
        })
        .collect();
    let img = FloatImage::from_samples(w, h, channels, samples).unwrap();

    // Keep the render small: the contract is panic-freedom, not load.
    if let Ok((ow, oh)) = filter.output_extent(w, h)
        && (ow as u64) * (oh as u64) <= 96 * 96
    {
        match filter.distort(&img) {
            Ok(out) => {
                assert_eq!((out.image.width(), out.image.height()), (ow, oh));
            }
            Err(_) => {}
        }
        // The same through the filter contract on 8-bit storage.
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize * 4,
                data: data
                    .iter()
                    .copied()
                    .cycle()
                    .take(w as usize * h as usize * 4)
                    .collect(),
            }],
        };
        let params = VideoStreamParams {
            format: PixelFormat::Rgba,
            width: w,
            height: h,
        };
        if let Ok(out) = filter.apply(&frame, params) {
            assert_eq!(out.planes[0].data.len(), ow as usize * oh as usize * 4);
        }
        // The cylindrical resize with the same filter and overrides.
        let (rw, rh) = (1 + ow % 40, 1 + oh % 40);
        let out = Resize::new(rw, rh)
            .with_filter(filter_kind)
            .with_filter_options(opts)
            .with_cylindrical(true)
            .with_linear_light(b.byte() & 1 == 1)
            .apply(&frame, params)
            .expect("a well-formed Rgba picture resizes");
        assert_eq!(out.planes[0].data.len(), rw as usize * rh as usize * 4);
    }
});
