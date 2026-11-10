//! Thread budget for the row-parallel filter kernels.
//!
//! Every filter is serial unless the caller grants a budget: wrap the
//! call in [`with_execution_context`] and the kernels that split by
//! rows fan out over at most
//! [`ExecutionContext::effective_workers`] scoped threads. The split is
//! by whole output rows, each row computed exactly as the serial loop
//! computes it, so the output is identical at every budget.
//!
//! The budget is scoped to the closure and to the calling thread; it
//! does not leak into other threads or later calls.

use std::cell::Cell;

use oxideav_core::ExecutionContext;

thread_local! {
    static BUDGET: Cell<usize> = const { Cell::new(1) };
}

struct Restore(usize);

impl Drop for Restore {
    fn drop(&mut self) {
        BUDGET.with(|b| b.set(self.0));
    }
}

/// Run `f` with `ctx` as the thread budget of every filter call made
/// inside it on this thread (`apply`, `apply_two`, the free functions).
///
/// ```
/// # use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
/// use oxideav_image_filter::{Blur, ImageFilter, VideoStreamParams, with_execution_context};
/// # let frame = VideoFrame { pts: None, planes: vec![VideoPlane { stride: 16, data: vec![7; 64] }] };
/// let params = VideoStreamParams { format: PixelFormat::Rgba, width: 4, height: 4 };
/// let serial = Blur::new(1).apply(&frame, params)?;
/// let ctx = ExecutionContext::with_threads(4);
/// let parallel = with_execution_context(&ctx, || Blur::new(1).apply(&frame, params))?;
/// assert_eq!(serial.planes[0].data, parallel.planes[0].data);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn with_execution_context<R>(ctx: &ExecutionContext, f: impl FnOnce() -> R) -> R {
    let previous = BUDGET.with(|b| b.replace(ctx.threads.max(1)));
    let _restore = Restore(previous);
    f()
}

/// The budget in force on this thread (1 outside
/// [`with_execution_context`]).
pub(crate) fn threads() -> usize {
    BUDGET.with(|b| b.get())
}

/// Fill `out` row by row: `out` is `rows × row_len` items and
/// `row(y, slice)` writes row `y`. Rows are independent, so the bands
/// run on scoped threads under the budget; with a budget of 1 (or a
/// single row) this is the plain serial loop.
pub(crate) fn par_rows<T: Send>(
    out: &mut [T],
    row_len: usize,
    row: impl Fn(usize, &mut [T]) + Sync,
) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    par_bands(out, row_len, |y0, band| {
        for (i, r) in band.chunks_exact_mut(row_len).enumerate() {
            row(y0 + i, r);
        }
    });
    debug_assert_eq!(rows * row_len, out.len());
}

/// As [`par_rows`], but `band(y0, slice)` receives a whole run of
/// consecutive rows starting at `y0` — for kernels that keep per-band
/// scratch (a sliding window, an accumulator) they would otherwise
/// allocate per row.
pub(crate) fn par_bands<T: Send>(
    out: &mut [T],
    row_len: usize,
    band: impl Fn(usize, &mut [T]) + Sync,
) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let workers = ExecutionContext::with_threads(threads()).effective_workers(rows);
    if workers <= 1 {
        band(0, out);
        return;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|s| {
        let band = &band;
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            s.spawn(move || band(i * per, chunk));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_scoped_and_restored() {
        assert_eq!(threads(), 1);
        let n = with_execution_context(&ExecutionContext::with_threads(5), || {
            let inner = with_execution_context(&ExecutionContext::serial(), threads);
            assert_eq!(inner, 1);
            threads()
        });
        assert_eq!(n, 5);
        assert_eq!(threads(), 1);
    }

    #[test]
    fn rows_see_their_own_index_at_every_budget() {
        for t in [1usize, 2, 3, 7, 64] {
            let mut out = vec![0u32; 13 * 5];
            with_execution_context(&ExecutionContext::with_threads(t), || {
                par_rows(&mut out, 5, |y, row| {
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = (y * 100 + x) as u32;
                    }
                });
            });
            for y in 0..13 {
                for x in 0..5 {
                    assert_eq!(out[y * 5 + x], (y * 100 + x) as u32, "threads {t}");
                }
            }
        }
    }

    #[test]
    fn empty_and_degenerate_are_no_ops() {
        let mut out: Vec<u8> = Vec::new();
        par_rows(&mut out, 4, |_, _| panic!("no rows"));
        let mut out = vec![1u8; 3];
        par_rows(&mut out, 0, |_, _| panic!("no row length"));
        assert_eq!(out, vec![1; 3]);
    }
}
