//! `-cdl` / `Imagick::colorDecisionListImage` (issue #12): the ASC
//! colour decision list — per-channel slope, offset and power, then a
//! saturation.
//!
//! As measured on `magick` 7.1.2 Q16-HDRI (`tests/magick_ops.rs`):
//!
//! - `v' = clamp((v · slope + offset) ^ power, 0, 1)` per colour
//!   channel — clamped after the power, not before (a negative base with
//!   an integral power follows the sign rule; with a fractional power it
//!   has no real value, and that channel stores as `0`);
//! - `out = L + saturation · (v' − L)`, where `L = 0.21267 R + 0.71526 G
//!   + 0.07217 B` is the luma of the **input** pixel;
//! - alpha is untouched; a saturation above 1 can still leave `[0, 1]`
//!   (integer layouts saturate, float layouts keep the value).

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// An ASC colour decision list.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cdl {
    /// Per-channel slope (R, G, B).
    pub slope: [f32; 3],
    /// Per-channel offset.
    pub offset: [f32; 3],
    /// Per-channel power.
    pub power: [f32; 3],
    /// Saturation (`1.0` = unchanged).
    pub saturation: f32,
}

impl Default for Cdl {
    fn default() -> Self {
        Self::new()
    }
}

impl Cdl {
    /// The identity list.
    pub fn new() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        }
    }

    /// Set the slopes.
    pub fn with_slope(mut self, slope: [f32; 3]) -> Self {
        self.slope = slope;
        self
    }

    /// Set the offsets.
    pub fn with_offset(mut self, offset: [f32; 3]) -> Self {
        self.offset = offset;
        self
    }

    /// Set the powers.
    pub fn with_power(mut self, power: [f32; 3]) -> Self {
        self.power = power;
        self
    }

    /// Set the saturation.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation;
        self
    }

    /// Read the `<Slope>`, `<Offset>`, `<Power>` and `<Saturation>`
    /// elements of a colour-correction document (the first of each; a
    /// missing element keeps its identity value).
    pub fn from_xml(xml: &str) -> Result<Self, Error> {
        fn element<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
            let open = format!("<{tag}");
            let start = xml.find(&open)?;
            let body = &xml[start + open.len()..];
            let body = &body[body.find('>')? + 1..];
            let end = body.find(&format!("</{tag}>"))?;
            Some(&body[..end])
        }
        fn numbers(s: &str, tag: &str) -> Result<Vec<f32>, Error> {
            s.split_whitespace()
                .map(|t| {
                    t.parse::<f32>().map_err(|_| {
                        Error::invalid(format!("oxideav-image-filter: CDL <{tag}> value '{t}'"))
                    })
                })
                .collect()
        }
        let mut cdl = Self::new();
        for (tag, dst) in [
            ("Slope", &mut cdl.slope),
            ("Offset", &mut cdl.offset),
            ("Power", &mut cdl.power),
        ] {
            if let Some(body) = element(xml, tag) {
                let v = numbers(body, tag)?;
                if v.len() != 3 {
                    return Err(Error::invalid(format!(
                        "oxideav-image-filter: CDL <{tag}> needs three values"
                    )));
                }
                dst.copy_from_slice(&v);
            }
        }
        if let Some(body) = element(xml, "Saturation") {
            let v = numbers(body, "Saturation")?;
            cdl.saturation = *v
                .first()
                .ok_or_else(|| Error::invalid("oxideav-image-filter: CDL <Saturation> is empty"))?;
        }
        Ok(cdl)
    }
}

impl PixelFilter for Cdl {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn is_pointwise(&self) -> bool {
        true
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        deep::run_any(input, params, |mut img| {
            let cc = img.color_channels();
            img.map_pixels_in_place(|px| {
                let rgb = if cc == 1 {
                    [px[0]; 3]
                } else {
                    [px[0], px[1], px[2]]
                };
                let l = 0.21267 * rgb[0] + 0.71526 * rgb[1] + 0.07217 * rgb[2];
                for (c, v) in px.iter_mut().enumerate() {
                    let sop = (rgb[c] * self.slope[c] + self.offset[c]).powf(self.power[c]);
                    // `clamp` keeps NaN (a fractional power of a negative).
                    let sop = sop.clamp(0.0, 1.0);
                    *v = l + self.saturation * (sop - l);
                }
            });
            Ok(img)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_correction_document() {
        let c = Cdl::from_xml(
            "<ColorCorrection id=\"x\"><SOPNode><Slope> 0.9 1.2 0.5 </Slope>\
             <Offset>0.4 -0.5 0.6</Offset><Power>1 0.8 1.5</Power></SOPNode>\
             <SATNode><Saturation>0.85</Saturation></SATNode></ColorCorrection>",
        )
        .unwrap();
        assert_eq!(c.slope, [0.9, 1.2, 0.5]);
        assert_eq!(c.offset, [0.4, -0.5, 0.6]);
        assert_eq!(c.power, [1.0, 0.8, 1.5]);
        assert_eq!(c.saturation, 0.85);
        assert!(Cdl::from_xml("<Slope>1 2</Slope>").is_err());
    }
}
