//! Factories for the dither / quantise family and the operator set of
//! the r490 round, written against one small parameter reader. Every
//! factory builds with an empty parameter object (documented defaults).

use oxideav_core::{Error, PortSpec, Result, StreamFilter};
use serde_json::{Map, Value};

use super::{ImageFilterAdapter, passthrough_out_port, video_in_port};
use crate::ImageFilter;

/// A filter's JSON parameters.
pub(super) struct Params<'a> {
    filter: &'static str,
    map: Option<&'a Map<String, Value>>,
}

impl<'a> Params<'a> {
    fn get(&self, key: &str) -> Option<&'a Value> {
        self.map.and_then(|m| m.get(key))
    }

    pub(super) fn f64(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| self.bad(key, "a finite number")),
        }
    }

    pub(super) fn u32(&self, key: &str, default: u32) -> Result<u32> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| self.bad(key, "an unsigned integer")),
        }
    }

    pub(super) fn str(&self, key: &str) -> Result<Option<&'a str>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(Some)
                .ok_or_else(|| self.bad(key, "a string")),
        }
    }

    /// A list of numbers (`None` when the key is absent).
    pub(super) fn numbers(&self, key: &str) -> Result<Option<Vec<f64>>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => self.number_list(v, key).map(Some),
        }
    }

    fn number_list(&self, v: &Value, key: &str) -> Result<Vec<f64>> {
        v.as_array()
            .ok_or_else(|| self.bad(key, "a list of numbers"))?
            .iter()
            .map(|n| {
                n.as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| self.bad(key, "a list of finite numbers"))
            })
            .collect()
    }

    /// A list of number lists (`None` when the key is absent).
    pub(super) fn rows(&self, key: &str) -> Result<Option<Vec<Vec<f64>>>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_array()
                .ok_or_else(|| self.bad(key, "a list of number lists"))?
                .iter()
                .map(|row| self.number_list(row, key))
                .collect::<Result<Vec<_>>>()
                .map(Some),
        }
    }

    pub(super) fn bad(&self, key: &str, want: &str) -> Error {
        Error::invalid(format!(
            "job: filter '{}': `{key}` must be {want}",
            self.filter
        ))
    }
}

type Build = fn(&Params) -> Result<Box<dyn ImageFilter>>;

/// Register `build` under every name of `names`; the output port keeps
/// the input's extent (a filter that changes it documents so).
fn add(
    sink: &mut dyn FnMut(&str, oxideav_core::FilterFactory),
    names: &[&'static str],
    build: Build,
) {
    for &name in names {
        sink(
            name,
            Box::new(
                move |params: &Value, inputs: &[PortSpec]| -> Result<Box<dyn StreamFilter>> {
                    let filter = build(&Params {
                        filter: name,
                        map: params.as_object(),
                    })?;
                    let in_port = video_in_port(inputs);
                    let out_port = passthrough_out_port(&in_port);
                    Ok(Box::new(ImageFilterAdapter::new(filter, in_port, out_port)))
                },
            ),
        );
    }
}

/// The dither of `{"dither" | "method": name, "scan", "stencil": {"rows",
/// "origin", "divisor"}, "history", "ratio", "period", "angle", "shape",
/// "size"}`; `fallback` when no method is named.
fn dither_method(p: &Params, fallback: &str) -> Result<crate::DitherMethod> {
    use crate::{DiffusionStencil, DitherMethod, ScanOrder, SpotShape, ThresholdMap};
    let name = p.str("dither")?.or(p.str("method")?).unwrap_or(fallback);
    let scan = match p.str("scan")? {
        None | Some("serpentine") => ScanOrder::Serpentine,
        Some("raster") => ScanOrder::Raster,
        Some(_) => return Err(p.bad("scan", "'raster' or 'serpentine'")),
    };
    let mut method = match name {
        "stencil" | "custom" => {
            let rows = p
                .rows("rows")?
                .ok_or_else(|| p.bad("rows", "the stencil rows (current line first)"))?;
            let rows: Vec<Vec<f32>> = rows
                .iter()
                .map(|r| r.iter().map(|&v| v as f32).collect())
                .collect();
            let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
            let divisor = match p.get("divisor") {
                Some(_) => p.f64("divisor", 1.0)? as f32,
                None => rows.iter().flatten().sum(),
            };
            DitherMethod::Diffusion {
                stencil: DiffusionStencil::from_rows(&refs, p.u32("origin", 0)? as usize, divisor)?,
                scan,
            }
        }
        "clustered-dot" | "halftone" | "screen" => {
            let shape = match p.str("shape")? {
                None => SpotShape::default(),
                Some(s) => SpotShape::from_name(s)
                    .ok_or_else(|| p.bad("shape", "'round', 'diamond', 'line' or 'square'"))?,
            };
            DitherMethod::Ordered(ThresholdMap::clustered_dot(
                p.f64("period", 6.0)?,
                p.f64("angle", 45.0)?,
                shape,
            ))
        }
        "blue-noise" | "bluenoise" => {
            DitherMethod::Ordered(ThresholdMap::blue_noise(p.u32("size", 64)?))
        }
        other => DitherMethod::from_name(other).ok_or_else(|| {
            p.bad(
                "dither",
                "'none', a diffusion kernel, 'hilbert', 'stencil', 'clustered-dot', \
                 'blue-noise' or a threshold-map name",
            )
        })?,
    };
    match &mut method {
        DitherMethod::Diffusion { scan: s, .. } => *s = scan,
        DitherMethod::Hilbert { history, ratio } => {
            *history = p.u32("history", *history)?;
            *ratio = p.f64("ratio", *ratio as f64)? as f32;
        }
        _ => {}
    }
    Ok(method)
}

fn channel_dither(p: &Params, fallback: &str) -> Result<Box<dyn ImageFilter>> {
    let mut f = crate::ChannelDither::new(dither_method(p, fallback)?);
    match p.get("levels") {
        None => {}
        Some(Value::Array(_)) => {
            let l = p.numbers("levels")?.unwrap_or_default();
            if l.len() != 3 {
                return Err(p.bad("levels", "one count or three (R, G, B)"));
            }
            f = f.with_channel_levels(l[0] as u32, l[1] as u32, l[2] as u32);
        }
        Some(_) => f = f.with_levels(p.u32("levels", 2)?),
    }
    if p.get("alpha_levels").is_some() {
        f = f.with_alpha_levels(p.u32("alpha_levels", 2)?);
    }
    Ok(Box::new(f))
}

fn palette_quantize(p: &Params) -> Result<Box<dyn ImageFilter>> {
    use crate::{Palette, PaletteQuantize};
    let f = match p.rows("palette")? {
        Some(rows) => {
            let mut colors = Vec::with_capacity(rows.len());
            for row in &rows {
                if !(3..=4).contains(&row.len()) {
                    return Err(p.bad("palette", "a list of [R, G, B] or [R, G, B, A] (0 to 255)"));
                }
                let mut c = [0.0, 0.0, 0.0, 1.0];
                for (slot, v) in c.iter_mut().zip(row) {
                    *slot = (*v / 255.0) as f32;
                }
                colors.push(c);
            }
            PaletteQuantize::new(Palette::new(colors)?)
        }
        None => PaletteQuantize::adaptive(p.u32("colors", 256)? as usize),
    };
    Ok(Box::new(
        f.with_dither(dither_method(p, "floyd-steinberg")?),
    ))
}

/// Install the factories.
pub(super) fn register_ops(sink: &mut dyn FnMut(&str, oxideav_core::FilterFactory)) {
    add(sink, &["channel-dither", "dither-levels"], |p| {
        channel_dither(p, "floyd-steinberg")
    });
    add(sink, &["dither-hilbert", "dither-riemersma"], |p| {
        channel_dither(p, "hilbert")
    });
    add(sink, &["dither-blue-noise"], |p| {
        channel_dither(p, "blue-noise")
    });
    add(sink, &["dither-clustered-dot", "dither-halftone"], |p| {
        channel_dither(p, "clustered-dot")
    });
    add(
        sink,
        &["palette-quantize", "quantize-colors", "colors", "remap"],
        palette_quantize,
    );
}

#[cfg(test)]
mod tests {
    use oxideav_core::{PixelFormat, PortSpec, RuntimeContext, TimeBase};
    use serde_json::json;

    fn make(name: &str, params: serde_json::Value) -> oxideav_core::Result<()> {
        let mut ctx = RuntimeContext::new();
        crate::registry::register(&mut ctx);
        let inputs = [PortSpec::video(
            "in",
            16,
            16,
            PixelFormat::Rgb24,
            TimeBase::new(1, 30),
        )];
        ctx.filters.make(name, &params, &inputs).map(|_| ())
    }

    #[test]
    fn dither_and_palette_factories_take_their_parameters() {
        make("channel-dither", json!({})).unwrap();
        make(
            "channel-dither",
            json!({"method": "stucki", "scan": "raster", "levels": [4, 8, 4]}),
        )
        .unwrap();
        make(
            "channel-dither",
            json!({"method": "stencil", "rows": [[0, 0, 8, 4], [2, 4, 8, 4]], "origin": 1}),
        )
        .unwrap();
        make(
            "dither-hilbert",
            json!({"history": 32, "ratio": 8.0, "levels": 3}),
        )
        .unwrap();
        make("dither-blue-noise", json!({"size": 32, "alpha_levels": 2})).unwrap();
        make(
            "dither-halftone",
            json!({"period": 5, "angle": 15, "shape": "diamond"}),
        )
        .unwrap();
        make("colors", json!({"colors": 16, "dither": "none"})).unwrap();
        make(
            "remap",
            json!({"palette": [[0, 0, 0], [255, 255, 255, 128]], "dither": "hilbert"}),
        )
        .unwrap();
        make("palette-quantize", json!({"dither": "dot6@45"})).unwrap();
        assert!(make("channel-dither", json!({"method": "nope"})).is_err());
        assert!(make("channel-dither", json!({"levels": [2, 2]})).is_err());
        assert!(make("remap", json!({"palette": [[0, 0]]})).is_err());
        assert!(make("channel-dither", json!({"method": "stencil"})).is_err());
        assert!(make("channel-dither", json!({"scan": "diagonal"})).is_err());
    }
}
