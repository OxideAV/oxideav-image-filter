//! `-connected-components` / `connectedComponentsImage` (issue #12):
//! label the regions of equal colour.
//!
//! Measured on `magick` 7.1.2: a component is a maximal 4- or
//! 8-connected set of pixels of exactly one colour; ids are assigned in
//! raster order of each component's first pixel; the object list is
//! sorted by area (largest first, then id). Centroids are the mean pixel
//! index. The labelling and object list match `magick` exactly.
//!
//! With an area threshold, every object smaller than it is absorbed by
//! its lowest-id neighbouring object, whose area, bounding box and
//! centroid grow while its mean colour stays. That reproduces `magick`
//! on simple layouts but not on every picture: which neighbour `magick`
//! picks is not documented in the staged material (it is neither the
//! lowest id, the first found, the most adjacent nor the largest in
//! general — measured).

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::{PixelFilter, VideoStreamParams, deep, layout};

/// One labelled object.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentObject {
    /// The object's id (its label value).
    pub id: u32,
    /// Bounding box `(x, y, width, height)`.
    pub bbox: (u32, u32, u32, u32),
    /// Mean pixel index `(x, y)`.
    pub centroid: (f64, f64),
    /// Pixel count.
    pub area: u64,
    /// Mean colour of the object's own pixels (normalised, the picture's
    /// channels).
    pub mean_color: Vec<f32>,
}

/// Labels and objects of a picture.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ComponentMap {
    /// Picture width.
    pub width: u32,
    /// Picture height.
    pub height: u32,
    /// Object id of every pixel, row-major.
    pub labels: Vec<u32>,
    /// The surviving objects, largest first (then by id).
    pub objects: Vec<ComponentObject>,
}

/// Label `frame` with `connectivity` (4 or 8) and an optional area
/// threshold.
pub fn connected_components(
    frame: &VideoFrame,
    params: VideoStreamParams,
    connectivity: u8,
    area_threshold: Option<f64>,
) -> Result<ComponentMap, Error> {
    if connectivity != 4 && connectivity != 8 {
        return Err(Error::invalid(
            "oxideav-image-filter: connected-components connectivity is 4 or 8",
        ));
    }
    let img = crate::measure::float_image_of(frame, params)?;
    let (w, h, ch) = (img.width() as usize, img.height() as usize, img.channels());
    let px = |i: usize| &img.samples()[i * ch..i * ch + ch];
    let mut labels = vec![u32::MAX; w * h];
    let mut next = 0u32;
    let mut stack = Vec::new();
    let neigh: &[(i64, i64)] = if connectivity == 4 {
        &[(1, 0), (-1, 0), (0, 1), (0, -1)]
    } else {
        &[
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    };
    for start in 0..w * h {
        if labels[start] != u32::MAX {
            continue;
        }
        let id = next;
        next += 1;
        labels[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for &(dx, dy) in neigh {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if labels[j] == u32::MAX && px(j) == px(start) {
                    labels[j] = id;
                    stack.push(j);
                }
            }
        }
    }
    let n = next as usize;
    let mut stats: Vec<Stat> = (0..n).map(|_| Stat::new(ch)).collect();
    for (i, &l) in labels.iter().enumerate() {
        stats[l as usize].add(i % w, i / w, px(i));
    }
    let mut alive = vec![true; n];
    if let Some(t) = area_threshold.filter(|t| t.is_finite() && *t > 0.0) {
        for id in 0..n {
            if !alive[id] || (stats[id].area as f64) >= t {
                continue;
            }
            // Lowest-id neighbouring object (after earlier merges).
            let mut best: Option<u32> = None;
            for (i, &l) in labels.iter().enumerate() {
                if l as usize != id {
                    continue;
                }
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                for &(dx, dy) in neigh {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let other = labels[ny as usize * w + nx as usize];
                    if other as usize != id {
                        best = Some(best.map_or(other, |b| b.min(other)));
                    }
                }
            }
            let Some(target) = best else { continue };
            for l in labels.iter_mut() {
                if *l as usize == id {
                    *l = target;
                }
            }
            let absorbed = std::mem::replace(&mut stats[id], Stat::new(ch));
            stats[target as usize].absorb(&absorbed);
            alive[id] = false;
        }
    }
    let mut objects: Vec<ComponentObject> = (0..n)
        .filter(|&id| alive[id])
        .map(|id| stats[id].object(id as u32))
        .collect();
    objects.sort_by(|a, b| b.area.cmp(&a.area).then(a.id.cmp(&b.id)));
    Ok(ComponentMap {
        width: w as u32,
        height: h as u32,
        labels,
        objects,
    })
}

struct Stat {
    area: u64,
    sx: f64,
    sy: f64,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color_sum: Vec<f64>,
    color_n: u64,
}

impl Stat {
    fn new(ch: usize) -> Self {
        Self {
            area: 0,
            sx: 0.0,
            sy: 0.0,
            x0: usize::MAX,
            y0: usize::MAX,
            x1: 0,
            y1: 0,
            color_sum: vec![0.0; ch],
            color_n: 0,
        }
    }

    fn add(&mut self, x: usize, y: usize, px: &[f32]) {
        self.area += 1;
        self.sx += x as f64;
        self.sy += y as f64;
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
        for (s, &v) in self.color_sum.iter_mut().zip(px) {
            *s += v as f64;
        }
        self.color_n += 1;
    }

    /// Geometry grows; the mean colour stays the absorber's (measured).
    fn absorb(&mut self, o: &Stat) {
        if o.area == 0 {
            return;
        }
        self.area += o.area;
        self.sx += o.sx;
        self.sy += o.sy;
        self.x0 = self.x0.min(o.x0);
        self.y0 = self.y0.min(o.y0);
        self.x1 = self.x1.max(o.x1);
        self.y1 = self.y1.max(o.y1);
    }

    fn object(&self, id: u32) -> ComponentObject {
        let n = self.area.max(1) as f64;
        ComponentObject {
            id,
            bbox: (
                self.x0 as u32,
                self.y0 as u32,
                (self.x1 + 1 - self.x0) as u32,
                (self.y1 + 1 - self.y0) as u32,
            ),
            centroid: (self.sx / n, self.sy / n),
            area: self.area,
            mean_color: self
                .color_sum
                .iter()
                .map(|s| (s / self.color_n.max(1) as f64) as f32)
                .collect(),
        }
    }
}

/// The picture recoloured with each object's mean colour
/// (`connected-components:mean-color=true`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct ConnectedComponents {
    /// 4 or 8.
    pub connectivity: u8,
    /// Objects smaller than this are absorbed.
    pub area_threshold: Option<f64>,
}

impl ConnectedComponents {
    /// `connectivity` 4 or 8 (anything else is 8).
    pub fn new(connectivity: u8) -> Self {
        Self {
            connectivity: if connectivity == 4 { 4 } else { 8 },
            area_threshold: None,
        }
    }

    /// `connected-components:area-threshold`.
    pub fn with_area_threshold(mut self, t: f64) -> Self {
        self.area_threshold = Some(t);
        self
    }
}

impl PixelFilter for ConnectedComponents {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let map = connected_components(input, params, self.connectivity, self.area_threshold)?;
        let float = deep::is_float(params.format);
        let mut img = if float {
            deep::FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        let mut color: Vec<Option<&[f32]>> = vec![
            None;
            map.labels
                .iter()
                .copied()
                .max()
                .map_or(0, |m| m as usize + 1)
        ];
        for o in &map.objects {
            color[o.id as usize] = Some(&o.mean_color);
        }
        let ch = img.channels();
        for (px, &l) in img.samples_mut().chunks_exact_mut(ch).zip(&map.labels) {
            if let Some(c) = color[l as usize] {
                px.copy_from_slice(c);
            }
        }
        Ok(if float {
            img.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&img);
            f.pts = input.pts;
            f
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    #[test]
    fn four_and_eight_connectivity_and_merging() {
        // Two diagonal white pixels on black.
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 0];
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 3, data }],
        };
        let p = VideoStreamParams {
            format: PixelFormat::Gray8,
            width: 3,
            height: 3,
        };
        let m4 = connected_components(&f, p, 4, None).unwrap();
        assert_eq!(m4.objects.len(), 3);
        let m8 = connected_components(&f, p, 8, None).unwrap();
        assert_eq!(m8.objects.len(), 2);
        assert_eq!(m8.objects[0].area, 7);
        let merged = connected_components(&f, p, 8, Some(3.0)).unwrap();
        assert_eq!(merged.objects.len(), 1);
        assert_eq!(merged.objects[0].area, 9);
        assert!(merged.labels.iter().all(|&l| l == merged.objects[0].id));
    }
}
