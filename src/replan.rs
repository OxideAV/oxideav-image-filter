//! Mid-stream re-planning for every registered filter.
//!
//! A [`Frame::StreamParams`] event announces that the upstream shape
//! (width / height / pixel format) changes from the next frame on. Every
//! factory in [`crate::registry`] already derives its whole plan — the
//! concrete filter, its output port shape, its refusal of an input it
//! cannot take — from the input [`PortSpec`]s it is handed. So one
//! wrapper serves all of them: on an event it re-runs the filter's own
//! factory with the same parameters and the updated input port, swaps
//! the freshly planned filter in, and announces the new output shape on
//! every output port whose shape moved (or which follows its input's
//! shape, the pass-through case).
//!
//! A factory that cannot plan for the new input (a compositor whose two
//! inputs no longer agree, a crop window that no longer fits, a pixel
//! layout the filter does not take) makes the event fail with
//! [`Error::Unsupported`], and the filter keeps its previous plan.

use std::sync::Arc;

use oxideav_core::{
    CodecParameters, Error, FilterFactory, Frame, MediaType, PixelFormat, PortParams, PortSpec,
    Result, StreamFilter, filter::FilterContext,
};
use serde_json::Value;

/// `(width, height, format)` of a video port, `None` for other kinds.
pub(crate) fn video_shape(spec: &PortSpec) -> Option<(u32, u32, PixelFormat)> {
    match spec.params {
        PortParams::Video {
            width,
            height,
            format,
            ..
        } => Some((width, height, format)),
        _ => None,
    }
}

/// The video shape `event` announces for a port currently shaped
/// `current`: fields the event leaves unset keep their current value.
pub(crate) fn merged_shape(
    current: (u32, u32, PixelFormat),
    event: &CodecParameters,
) -> (u32, u32, PixelFormat) {
    (
        event.width.unwrap_or(current.0),
        event.height.unwrap_or(current.1),
        event.pixel_format.unwrap_or(current.2),
    )
}

/// The event to emit on an output port now shaped `shape`: the incoming
/// event's parameters with the picture shape replaced by the port's.
fn output_event(incoming: &CodecParameters, shape: (u32, u32, PixelFormat)) -> Frame {
    let mut p = incoming.clone();
    p.media_type = MediaType::Video;
    p.width = Some(shape.0);
    p.height = Some(shape.1);
    p.pixel_format = Some(shape.2);
    Frame::StreamParams(Box::new(p))
}

/// Wrap `factory` so every filter it builds re-plans on
/// [`Frame::StreamParams`].
pub(crate) fn replannable(name: &str, factory: FilterFactory) -> FilterFactory {
    let factory = Arc::new(factory);
    let name: Arc<str> = Arc::from(name);
    Box::new(move |params: &Value, inputs: &[PortSpec]| {
        let inner = factory(params, inputs)?;
        let pending = if inner.input_ports().len() > 1 {
            vec![None; inner.input_ports().len()]
        } else {
            Vec::new()
        };
        Ok(Box::new(Replanning {
            name: name.clone(),
            factory: factory.clone(),
            params: params.clone(),
            inner,
            pending,
        }) as Box<dyn StreamFilter>)
    })
}

/// A registered filter plus what it takes to plan it again.
struct Replanning {
    name: Arc<str>,
    factory: Arc<FilterFactory>,
    params: Value,
    inner: Box<dyn StreamFilter>,
    /// Multi-input filters only: the frame each input port holds that has
    /// not yet been consumed by an emit. A re-plan rebuilds the filter,
    /// so the frames buffered on the ports whose shape did not change are
    /// handed to the new plan (the changed port's old-shape frame is
    /// dropped — it would have been replaced by the next frame anyway).
    pending: Vec<Option<Frame>>,
}

/// Forwards to the real context and records whether anything was emitted.
struct EmitProbe<'a> {
    ctx: &'a mut dyn FilterContext,
    emitted: bool,
}

impl FilterContext for EmitProbe<'_> {
    fn emit(&mut self, output_port: usize, frame: Frame) -> Result<()> {
        self.emitted = true;
        self.ctx.emit(output_port, frame)
    }
}

/// Swallows everything (the plan check emits nothing that matters).
struct Discard;

impl FilterContext for Discard {
    fn emit(&mut self, _output_port: usize, _frame: Frame) -> Result<()> {
        Ok(())
    }
}

impl Replanning {
    fn push_tracked(
        &mut self,
        ctx: &mut dyn FilterContext,
        port: usize,
        frame: &Frame,
    ) -> Result<()> {
        if port >= self.pending.len() {
            return self.inner.push(ctx, port, frame);
        }
        let mut probe = EmitProbe {
            ctx,
            emitted: false,
        };
        self.inner.push(&mut probe, port, frame)?;
        if probe.emitted {
            self.pending.iter_mut().for_each(|p| *p = None);
        } else {
            self.pending[port] = Some(frame.clone());
        }
        Ok(())
    }

    fn replan(
        &mut self,
        ctx: &mut dyn FilterContext,
        port: usize,
        event: &CodecParameters,
    ) -> Result<()> {
        let name = &*self.name;
        let old_in: Vec<PortSpec> = self.inner.input_ports().to_vec();
        let Some(spec) = old_in.get(port) else {
            return Err(Error::invalid(format!(
                "filter '{name}': stream-params event on unknown input port {port}"
            )));
        };
        let (Some(old_shape), PortParams::Video { time_base, .. }) =
            (video_shape(spec), &spec.params)
        else {
            return Err(Error::unsupported(format!(
                "filter '{name}': input port {port} is not a video port"
            )));
        };
        if event.media_type != MediaType::Video {
            return Err(Error::unsupported(format!(
                "filter '{name}': input port {port} carries video, the stream-params event \
                 announces {:?}",
                event.media_type
            )));
        }
        let new_shape = merged_shape(old_shape, event);
        let mut new_in = old_in.clone();
        new_in[port] = PortSpec::video(
            spec.name.clone(),
            new_shape.0,
            new_shape.1,
            new_shape.2,
            *time_base,
        );

        let mut merged = event.clone();
        merged.width = Some(new_shape.0);
        merged.height = Some(new_shape.1);
        merged.pixel_format = Some(new_shape.2);
        let check = Frame::StreamParams(Box::new(merged));

        let mut rebuilt = if new_shape == old_shape {
            None
        } else {
            Some((self.factory)(&self.params, &new_in).map_err(|e| {
                Error::unsupported(format!(
                    "filter '{name}': cannot re-plan input port {port} from {}x{} {:?} to \
                     {}x{} {:?}: {e}",
                    old_shape.0, old_shape.1, old_shape.2, new_shape.0, new_shape.1, new_shape.2
                ))
            })?)
        };
        // The planned filter checks the layout it is now asked to take.
        match rebuilt.as_mut() {
            Some(f) => f.push(&mut Discard, port, &check)?,
            None => self.inner.push(&mut Discard, port, &check)?,
        }

        let old_out: Vec<PortSpec> = self.inner.output_ports().to_vec();
        if let Some(f) = rebuilt.take() {
            self.inner = f;
        }
        for (i, out) in self.inner.output_ports().to_vec().iter().enumerate() {
            let Some(now) = video_shape(out) else {
                continue;
            };
            let before = old_out.get(i).and_then(video_shape);
            let changed = before != Some(now);
            let follows_input = before == Some(old_shape) && now == new_shape;
            if changed || follows_input {
                ctx.emit(i, output_event(event, now))?;
            }
        }

        if new_shape != old_shape && !self.pending.is_empty() {
            // Hand the still-unpaired frames of the unchanged ports to the
            // new plan, after the announcement of its output shape.
            self.pending[port] = None;
            let held: Vec<(usize, Frame)> = self
                .pending
                .iter_mut()
                .enumerate()
                .filter_map(|(q, f)| f.take().map(|f| (q, f)))
                .collect();
            for (q, f) in held {
                self.push_tracked(ctx, q, &f)?;
            }
        }
        Ok(())
    }
}

impl StreamFilter for Replanning {
    fn input_ports(&self) -> &[PortSpec] {
        self.inner.input_ports()
    }
    fn output_ports(&self) -> &[PortSpec] {
        self.inner.output_ports()
    }
    fn push(&mut self, ctx: &mut dyn FilterContext, port: usize, frame: &Frame) -> Result<()> {
        match frame {
            Frame::StreamParams(p) => self.replan(ctx, port, p),
            _ => self.push_tracked(ctx, port, frame),
        }
    }
    fn flush(&mut self, ctx: &mut dyn FilterContext) -> Result<()> {
        self.inner.flush(ctx)
    }
    fn reset(&mut self) -> Result<()> {
        self.pending.iter_mut().for_each(|p| *p = None);
        self.inner.reset()
    }
}
