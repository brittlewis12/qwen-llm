//! Fixed eight-row layer-zero observer for the frontier scheduling diagnostic.
use super::*;
use crate::metal::{DispatchCensusTagGuard, dispatch_census_tag_scope};
use std::cell::{Cell, RefCell};

pub(crate) const COPY_TAG: &str = "frontier.capture.";
pub(crate) const GDN_TAG: &str = "frontier.layer0.";
const ROWS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Progress {
    candidate: bool,
    calls: usize,
}

impl Progress {
    fn next(&mut self, tokens: usize) -> Result<(usize, usize), Qwen4ExpGdnError> {
        let (expected, start, count) = match (self.candidate, self.calls) {
            (false, 0) => (3, 0, 3),
            (false, 1) => (2045, 3, 5),
            (true, 0) => (2048, 0, 8),
            _ => return invalid("frontier observer received an extra layer-zero call"),
        };
        if tokens != expected {
            return invalid(format!(
                "frontier observer expected N={expected}, got N={tokens}"
            ));
        }
        self.calls += 1;
        Ok((start, count))
    }

    fn complete(self) -> bool {
        self.calls == if self.candidate { 1 } else { 2 }
    }
}

thread_local! {
    static PROGRESS: Cell<Option<Progress>> = const { Cell::new(None) };
    static BANK: RefCell<Option<FrontierGdnCapture>> = const { RefCell::new(None) };
}

fn with_progress<R>(candidate: bool, work: impl FnOnce() -> R) -> (R, Progress) {
    struct Restore(Option<Progress>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRESS.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(PROGRESS.with(|p| {
        p.replace(Some(Progress {
            candidate,
            calls: 0,
        }))
    }));
    let result = work();
    (result, PROGRESS.with(|p| p.get().unwrap()))
}

pub(crate) struct FrontierGdnCapture {
    identity: MetalTensor,
    geometry: GatedDeltaNetMetalGeometry,
    pub(crate) candidate: bool,
    pub(crate) complete: bool,
    pub(crate) buffers: Vec<(&'static str, MetalTensor)>,
}

impl FrontierGdnCapture {
    pub(crate) fn specs(g: GatedDeltaNetMetalGeometry) -> Vec<(&'static str, Vec<u64>)> {
        let mut result: Vec<_> = [
            ("input", g.hidden_size()),
            ("qkv", g.conv_width()),
            ("beta", g.value_heads()),
            ("alpha", g.value_heads()),
            ("decay", g.value_heads()),
            ("query", g.key_width()),
            ("key", g.key_width()),
            ("query_norm", g.key_width()),
            ("key_norm", g.key_width()),
            ("value", g.value_width()),
            ("recurrent", g.value_width()),
            ("gate", g.value_width()),
            ("normalized", g.value_width()),
            ("output", g.hidden_size()),
        ]
        .into_iter()
        .map(|(name, width)| (name, vec![width as u64, ROWS as u64]))
        .collect();
        result.extend([
            (
                "initial_delta",
                vec![
                    g.head_dim() as u64,
                    g.head_dim() as u64,
                    g.value_heads() as u64,
                ],
            ),
            ("initial_conv", vec![g.conv_width() as u64, 3]),
            (
                "state_checkpoints",
                vec![
                    g.head_dim() as u64,
                    g.head_dim() as u64,
                    g.value_heads() as u64,
                    3,
                ],
            ),
        ]);
        result
    }

    /// The diagnostic must price/admit every spec before constructing this bank.
    pub(crate) fn new(
        ctx: &MetalContext,
        weights: GatedDeltaNetMetalWeights<'_>,
    ) -> Result<Self, Qwen4ExpGdnError> {
        let buffers = Self::specs(weights.geometry)
            .into_iter()
            .map(|(name, shape)| Ok((name, MetalTensor::zeros_f32(ctx, shape)?)))
            .collect::<Result<_, MetalError>>()?;
        Ok(Self {
            identity: weights.qkv.clone(),
            geometry: weights.geometry,
            candidate: false,
            complete: false,
            buffers,
        })
    }

    fn buffer(&self, name: &str) -> &MetalTensor {
        &self.buffers.iter().find(|(n, _)| *n == name).unwrap().1
    }
}

/// Owns all observer buffers for the synchronous work; no CPU readback here.
pub(crate) fn with_frontier_gdn_capture<R>(
    mut capture: FrontierGdnCapture,
    work: impl FnOnce() -> R,
) -> (R, FrontierGdnCapture) {
    struct Restore(Option<FrontierGdnCapture>);
    impl Drop for Restore {
        fn drop(&mut self) {
            BANK.with(|b| {
                b.replace(self.0.take());
            });
        }
    }
    capture.complete = false;
    let candidate = capture.candidate;
    let _restore = Restore(BANK.with(|b| b.replace(Some(capture))));
    let (result, progress) = with_progress(candidate, work);
    let mut capture = BANK.with(|b| b.borrow_mut().take().unwrap());
    capture.complete = progress.complete();
    (result, capture)
}

pub(super) struct Visit {
    start: usize,
    count: usize,
    tokens: usize,
    checkpoints: Option<MetalTensor>,
}

fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    name: &str,
) -> Result<(), Qwen4ExpGdnError> {
    let _tag = dispatch_census_tag_scope(|| format!("{COPY_TAG}{name}"));
    encode_copy_offset_f32(ctx, enc, src, 0, dst, dst.n_elements() as usize)?;
    Ok(())
}

pub(super) fn before(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: GatedDeltaNetMetalWeights<'_>,
    conv: &MetalTensor,
    delta: &MetalTensor,
    tokens: usize,
) -> Result<Option<Visit>, Qwen4ExpGdnError> {
    BANK.with(|b| {
        let b = b.borrow();
        let Some(bank) = b.as_ref() else {
            return Ok(None);
        };
        if Retained::as_ptr(&bank.identity.buffer) != Retained::as_ptr(&weights.qkv.buffer)
            || bank.identity.offset != weights.qkv.offset
        {
            return Ok(None);
        }
        if bank.geometry != weights.geometry {
            return invalid("frontier layer-zero geometry changed");
        }
        let (start, count) = PROGRESS.with(|p| {
            let mut progress = p.get().unwrap();
            let visit = progress.next(tokens)?;
            p.set(Some(progress));
            Ok::<_, Qwen4ExpGdnError>(visit)
        })?;
        let checkpoints = if start == 0 {
            copy(
                ctx,
                enc,
                delta,
                bank.buffer("initial_delta"),
                "initial_delta",
            )?;
            copy(ctx, enc, conv, bank.buffer("initial_conv"), "initial_conv")?;
            Some(bank.buffer("state_checkpoints").clone())
        } else {
            None
        };
        Ok(Some(Visit {
            start,
            count,
            tokens,
            checkpoints,
        }))
    })
}

pub(super) fn tag(visit: Option<&Visit>) -> Option<DispatchCensusTagGuard> {
    visit.and_then(|v| {
        dispatch_census_tag_scope(|| format!("{GDN_TAG}absolute{}.N{}", 2048 + v.start, v.tokens))
    })
}

pub(super) fn checkpoint(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visit: Option<&Visit>,
    state: &MetalTensor,
    scratch: &GatedDeltaNetPackedScratch,
    tokens: usize,
) -> Result<bool, Qwen4ExpGdnError> {
    let Some(tape) = visit.and_then(|v| v.checkpoints.as_ref()) else {
        return Ok(false);
    };
    let g = scratch.geometry;
    let prefix =
        |t: &MetalTensor, width| scratch.prefix_view("frontier recurrence", t, width, tokens);
    crate::metal::encode_gdn_step_decay_packed_ckpt_f32(
        ctx,
        enc,
        &prefix(&scratch.query_norm, g.key_width())?,
        &prefix(&scratch.key_norm, g.key_width())?,
        &prefix(&scratch.value, g.value_width())?,
        &prefix(&scratch.decay, g.value_heads())?,
        &prefix(&scratch.beta, g.value_heads())?,
        state,
        &prefix(&scratch.recurrent, g.value_width())?,
        tape,
        3,
        tokens,
        g.value_heads(),
        g.key_heads(),
        g.head_dim(),
    )?;
    Ok(true)
}

pub(super) fn after(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visit: Option<&Visit>,
    input: &MetalTensor,
    scratch: &GatedDeltaNetPackedScratch,
) -> Result<(), Qwen4ExpGdnError> {
    let Some(v) = visit else {
        return Ok(());
    };
    BANK.with(|b| {
        let b = b.borrow();
        let bank = b.as_ref().unwrap();
        for (name, source) in [
            ("input", input),
            ("qkv", &scratch.qkv),
            ("beta", &scratch.beta),
            ("alpha", &scratch.alpha),
            ("decay", &scratch.decay),
            ("query", &scratch.query),
            ("key", &scratch.key),
            ("query_norm", &scratch.query_norm),
            ("key_norm", &scratch.key_norm),
            ("value", &scratch.value),
            ("recurrent", &scratch.recurrent),
            ("gate", &scratch.gate),
            ("normalized", &scratch.normalized),
            ("output", &scratch.output),
        ] {
            let target = bank.buffer(name);
            let width = target.shape[0] as usize;
            let view =
                target.view_subrange((v.start * width) as u64, vec![(v.count * width) as u64]);
            copy(ctx, enc, source, &view, name)?;
        }
        Ok(())
    })
}

#[test]
fn frontier_capture_schedule_bounds() {
    for (candidate, widths, windows) in [
        (false, vec![3, 2045], vec![(0, 3), (3, 5)]),
        (true, vec![2048], vec![(0, 8)]),
    ] {
        let mut p = Progress {
            candidate,
            calls: 0,
        };
        assert!(!p.complete());
        assert!(p.next(8).is_err());
        assert_eq!(p.calls, 0);
        for (n, window) in widths.into_iter().zip(windows) {
            assert_eq!(p.next(n).unwrap(), window);
        }
        assert!(p.complete());
        assert!(p.next(2048).is_err());
    }
}

#[test]
fn frontier_capture_scope_restores_nested_unwind_and_thread() {
    assert!(PROGRESS.with(|p| p.get()).is_none());
    with_progress(false, || {
        PROGRESS.with(|p| {
            let mut progress = p.get().unwrap();
            progress.next(3).unwrap();
            p.set(Some(progress));
        });
        let parent = PROGRESS.with(|p| p.get());
        with_progress(true, || {
            assert!(PROGRESS.with(|p| p.get().unwrap().candidate))
        });
        assert_eq!(PROGRESS.with(|p| p.get()), parent);
        assert!(std::panic::catch_unwind(|| with_progress(true, || panic!("scope"))).is_err());
        assert_eq!(PROGRESS.with(|p| p.get()), parent);
        assert!(
            std::thread::spawn(|| PROGRESS.with(|p| p.get()).is_none())
                .join()
                .unwrap()
        );
    });
    assert!(PROGRESS.with(|p| p.get()).is_none());
}

#[test]
fn frontier_capture_specs_bound_rows_and_checkpoint_tape() {
    let g = GatedDeltaNetMetalGeometry::new(2560, 16, 48, 128, 4, 1e-6).unwrap();
    let specs = FrontierGdnCapture::specs(g);
    assert_eq!(specs.len(), 17);
    for (_, shape) in &specs[..14] {
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1], 8);
    }
    assert_eq!(specs[16], ("state_checkpoints", vec![128, 128, 48, 3]));
    let bytes: u64 = specs
        .iter()
        .map(|(_, s)| 4 * s.iter().product::<u64>())
        .sum();
    assert_eq!(bytes, 14_250_496);
}
