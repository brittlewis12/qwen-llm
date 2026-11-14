//! Diagnostic dump of the production causal schedule: per wave, the kernel
//! names and the GPU timestamp span between the buffer barriers that already
//! separate waves.
//!
//! M5 rejects `sampleCountersInBuffer:` inside a compute encoder, so a sampled
//! dump encodes each wave in its own stage-boundary-sampled concurrent encoder.
//! Encoder transitions can allow independent passes to overlap, so wave-mode
//! spans are diagnostic windows and are not an ordinary token-wall price.
//! Node mode uses explicit same-queue events between sampled encoders to measure
//! nonoverlapping individual dispatches. The production encoder is suspended
//! for the scope and reopened afterwards.
use super::*;
use objc2_metal::{MTLCommandBuffer, MTLDispatchType};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

struct PendingWave {
    scope: usize,
    wave: usize,
    kernels: Vec<String>,
    start: usize,
    end: usize,
    node: Option<usize>,
}

struct Active {
    samples: Vec<MetalTimestampSampleBuffer>,
    pending: Vec<PendingWave>,
    scope: usize,
    next_pair: usize,
    truncated: usize,
    nodes: bool,
    plans: Vec<serde_json::Value>,
    isolation_event: Option<Retained<ProtocolObject<dyn objc2_metal::MTLEvent>>>,
    isolation_value: u64,
}

thread_local! {
    static GRAPH: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Marker that clears the thread-local dump when dropped without `finish`.
#[must_use]
pub struct CausalGraphGuard {
    _thread: PhantomData<Rc<()>>,
}

impl Drop for CausalGraphGuard {
    fn drop(&mut self) {
        GRAPH.with(|g| {
            g.borrow_mut().take();
        });
    }
}

/// Start recording wave boundaries on this thread. Up to `max_waves` waves
/// can be sampled; a one-round B=3 V4.1 decode needs well under 2,000.
pub fn causal_graph_begin(
    ctx: &MetalContext,
    max_waves: usize,
) -> Result<CausalGraphGuard, MetalError> {
    begin(ctx, max_waves, false)
}

/// Capture exact span-derived predecessor edges and one isolated GPU timestamp
/// pair per node. This diagnostic serializes dispatches; use an uninstrumented
/// run for token wall time. Scope entry/exit ordering remains intact.
pub fn causal_node_graph_begin(
    ctx: &MetalContext,
    max_nodes: usize,
) -> Result<CausalGraphGuard, MetalError> {
    begin(ctx, max_nodes, true)
}

fn begin(
    ctx: &MetalContext,
    max_waves: usize,
    nodes: bool,
) -> Result<CausalGraphGuard, MetalError> {
    if max_waves == 0 {
        return Err(MetalError::Counter("empty causal graph dump".into()));
    }
    // One Metal counter sample buffer is capped at 4096 timestamps.
    let samples = (0..max_waves.div_ceil(2048))
        .map(|i| ctx.timestamp_sample_buffer(2 * (max_waves - i * 2048).min(2048)))
        .collect::<Result<Vec<_>, _>>()?;
    let isolation_event = if nodes {
        Some(
            ctx.device
                .newEvent()
                .ok_or_else(|| MetalError::Counter("no node isolation event".into()))?,
        )
    } else {
        None
    };
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        if g.is_some() {
            return Err(MetalError::Counter(
                "nested causal graph dump on one thread".into(),
            ));
        }
        *g = Some(Active {
            samples,
            pending: Vec::with_capacity(max_waves),
            scope: 0,
            next_pair: 0,
            truncated: 0,
            nodes,
            plans: Vec::new(),
            isolation_event,
            isolation_value: 0,
        });
        Ok(CausalGraphGuard {
            _thread: PhantomData,
        })
    })
}

pub(crate) fn active() -> bool {
    GRAPH.with(|g| g.borrow().is_some())
}

pub(crate) fn nodes_active() -> bool {
    GRAPH.with(|g| g.borrow().as_ref().is_some_and(|a| a.nodes))
}

/// Encoder boundaries alone may overlap independent passes. The event wait is
/// outside the next sampled encoder, so each node's start/end pair excludes
/// the preceding pass and the synchronization command itself.
pub(crate) fn isolate_node(enc: &KernelEncoder) {
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let a = g.as_mut().expect("node graph");
        let event = a.isolation_event.as_ref().expect("node isolation event");
        a.isolation_value += 1;
        enc.parent.encodeSignalEvent_value(event, a.isolation_value);
        enc.parent
            .encodeWaitForEvent_value(event, a.isolation_value);
    });
}

pub(crate) fn record_plan(nodes: &[Node], spans: &[Vec<Access>], waves: &[Vec<usize>]) {
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let Some(a) = g.as_mut() else { return };
        let rows: Vec<_> = nodes.iter().enumerate().map(|(i,n)| {
            let predecessors: Vec<_> = (0..i).filter(|&j| spans[j].iter().any(|&x|spans[i].iter().any(|&y|conflict(x,y)))).collect();
            // An in-place (read-write) binding becomes one read span and one
            // write span so the waste analyzer can see both edges. Read-only
            // and write-only bindings emit one span each.
            let encoded: Vec<_> = spans[i].iter().flat_map(|s| {
                let mut out = Vec::with_capacity(2);
                if s.read { out.push(serde_json::json!({"buffer":s.buffer,"lo":s.lo,"hi":s.hi,"write":false})); }
                if s.write { out.push(serde_json::json!({"buffer":s.buffer,"lo":s.lo,"hi":s.hi,"write":true})); }
                out
            }).collect();
            let constants: Vec<_> = n.state.slots.iter().filter_map(|(slot,b)| match b {
                Binding::Bytes(bytes) => Some((*slot, bytes.to_vec())), _ => None,
            }).collect();
            serde_json::json!({"id":i,"kernel":n.name(),"label":n.label,"grid":[n.grid.width,n.grid.height,n.grid.depth],"threads":[n.threads.width,n.threads.height,n.threads.depth],"predecessors":predecessors,"spans":encoded,"constants":constants,"threadgroup_memory":n.state.threadgroup,"pipeline":Retained::as_ptr(&n.state.pipeline.as_ref().unwrap().raw) as usize})
        }).collect();
        a.plans.push(serde_json::json!({"scope":a.scope,"nodes":rows,"waves":waves}));
    });
}

/// Begin a new causal scope (one per layer). Scopes are encoded in order, so
/// their wave spans are also serialized.
pub(crate) fn begin_scope() {
    GRAPH.with(|g| {
        if let Some(a) = g.borrow_mut().as_mut() {
            a.scope += 1;
        }
    });
}

pub(crate) fn scope() -> usize {
    GRAPH.with(|g| g.borrow().as_ref().map_or(0, |a| a.scope))
}

/// End the production encoder so the scope can be re-encoded wave by wave into
/// sampled encoders on the same command buffer.
pub(crate) fn suspend(enc: &KernelEncoder) {
    enc.raw.borrow().endEncoding();
}

/// Open a sampled encoder for one wave. Falls back to a plain concurrent
/// encoder (no timing) once the sample buffer is exhausted.
pub(crate) fn begin_wave(enc: &KernelEncoder) -> (KernelEncoder, Option<(usize, usize)>) {
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let Some(a) = g.as_mut() else {
            return (KernelEncoder::begin_concurrent(&enc.parent), None);
        };
        let start = 2 * a.next_pair;
        let end = start + 1;
        let buffer = a.next_pair / 2048;
        let local_start = (a.next_pair % 2048) * 2;
        if buffer >= a.samples.len() || local_start + 1 >= a.samples[buffer].sample_count() {
            a.truncated += 1;
            return (KernelEncoder::begin_concurrent(&enc.parent), None);
        }
        a.next_pair += 1;
        match KernelEncoder::try_begin_sampled(
            &enc.parent,
            &a.samples[buffer],
            local_start,
            local_start + 1,
            true,
        ) {
            Ok(wave) => (wave, Some((start, end))),
            Err(_) => {
                a.truncated += 1;
                (KernelEncoder::begin_concurrent(&enc.parent), None)
            }
        }
    })
}

/// Close one wave. `span` is the sample pair from [`begin_wave`] when sampled.
pub(crate) fn finish_wave(
    scope: usize,
    wave: usize,
    kernels: Vec<String>,
    span: Option<(usize, usize)>,
) {
    let Some((start, end)) = span else { return };
    GRAPH.with(|g| {
        if let Some(a) = g.borrow_mut().as_mut() {
            a.pending.push(PendingWave {
                scope,
                wave,
                kernels,
                start,
                end,
                node: None,
            });
        }
    });
}

pub(crate) fn finish_node(
    scope: usize,
    wave: usize,
    node: usize,
    kernel: String,
    span: Option<(usize, usize)>,
) {
    finish_wave(scope, wave, vec![kernel], span);
    if span.is_some() {
        GRAPH.with(|g| {
            if let Some(a) = g.borrow_mut().as_mut() {
                a.pending.last_mut().unwrap().node = Some(node);
            }
        });
    }
}

/// Reopen the production concurrent encoder after the sampled waves.
pub(crate) fn resume(enc: &KernelEncoder) {
    let raw = enc
        .parent
        .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        .expect("concurrent compute encoder");
    *enc.raw.borrow_mut() = raw;
}

/// Resolve the recorded wave spans. Call only after the command that carried
/// them has completed.
pub fn causal_graph_finish(
    ctx: &MetalContext,
    guard: CausalGraphGuard,
) -> Result<serde_json::Value, MetalError> {
    let active = GRAPH.with(|g| g.borrow_mut().take());
    // The slot is already empty, so the guard's drop is a no-op.
    drop(guard);
    let Some(active) = active else {
        return Err(MetalError::Counter(
            "causal graph dump is not active".into(),
        ));
    };
    if active.next_pair == 0 {
        return Err(MetalError::Counter(
            "causal graph dump recorded no waves".into(),
        ));
    }
    let mut times = Vec::with_capacity(2 * active.next_pair);
    for (i, samples) in active.samples.iter().enumerate() {
        let count = active.next_pair.saturating_sub(i * 2048).min(2048);
        if count > 0 {
            times.extend(ctx.resolve_timestamp_samples(samples, 2 * count)?);
        }
    }
    let mut rows = Vec::with_capacity(active.pending.len());
    let mut scopes = std::collections::BTreeMap::<usize, Vec<serde_json::Value>>::new();
    for w in &active.pending {
        let (Some(&start), Some(&end)) = (times.get(w.start), times.get(w.end)) else {
            continue;
        };
        let ns = end.saturating_sub(start);
        let row = serde_json::json!({
            "scope": w.scope,
            "wave": w.wave,
            "kernels": w.kernels,
            "gpu_ns": ns,
            "gpu_start": start,
            "gpu_end": end,
            "node": w.node,
        });
        scopes.entry(w.scope).or_default().push(row.clone());
        rows.push(row);
    }
    let layers: Vec<_> = scopes
        .iter()
        .map(|(scope, waves)| {
            serde_json::json!({
                "scope": scope,
                "waves": waves.len(),
                "gpu_ms": waves.iter().map(|r| r["gpu_ns"].as_u64().unwrap_or(0) as f64).sum::<f64>() / 1e6,
            })
        })
        .collect();
    Ok(serde_json::json!({
        "kind": "causal_graph",
        "mode": if active.nodes {"nodes"} else {"waves"},
        "node_isolation": if active.nodes {"same-queue MTLEvent signal/wait between sampled encoders"} else {"wave encoders"},
        "plans": active.plans,
        "waves": rows.len(),
        "scope_count": scopes.len(),
        "truncated_waves": active.truncated,
        "gpu_ms": rows.iter().map(|r| r["gpu_ns"].as_u64().unwrap_or(0) as f64).sum::<f64>() / 1e6,
        "layers": layers,
        "rows": rows,
    }))
}
