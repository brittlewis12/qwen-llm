//! PRIVATE restored decode diagnostic on production 1388793d.
//! Reuses 9e163c52 observation/parity/bracket gates and 634cdc97 instrumented warmup.
use super::*;
use qwen_llm::deepseek_v41::forward::{self, profile};
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "decode_private_tests.rs"]
mod tests;

const LABEL: &str = "v41_private_restored_decode_ffn_not_canonical";
const SCHEMA: &str = "v41-private-restored-decode-ffn-v1";
const CAPACITY: usize = 2080;
const TG: usize = 32;
const DEPTHS: [usize; 1] = [2048];
const ROLES: [&str; 5] = [
    "model_warmup_reference",
    "instrumented_warmup",
    "control_before",
    "profile",
    "control_after",
];

pub(super) fn validate_recipe(args: &SuiteArgs, extent: FamilyBenchExtent) -> Result<()> {
    ensure!(
        args.pp.is_empty()
            && args.tg == [TG]
            && args.depth == DEPTHS
            && args.runs == 3
            && !args.no_warmup
            && args.seed == 3
            && args.prefill_chunk.is_none()
            && extent.forwards == CAPACITY
            && extent.prefill_endpoint == 2048,
        "PRIVATE requires --tg 32 --depth 2048 --runs 3 --seed 3; capacity 2080; no pp/no-warmup/prefill-chunk"
    );
    Ok(())
}

pub(super) struct Bench<'a, 'w> {
    forward: &'a V41Forward<'w>,
    ctx: &'a MetalContext,
    vocab: usize,
    engrams: Vec<usize>,
    next_depth: usize,
    rejected_depths: Vec<usize>,
}

impl<'a, 'w> Bench<'a, 'w> {
    pub(super) fn new(
        forward: &'a V41Forward<'w>,
        ctx: &'a MetalContext,
        vocab: usize,
        engrams: Vec<usize>,
    ) -> Self {
        eprintln!(
            "{}",
            serde_json::json!({
                "diagnostic": LABEL, "schema": SCHEMA, "event": "recipe", "production_base": "1388793dc6bce0de79aaf2df136085f47c02e9a3",
                "capacity": CAPACITY, "depths": DEPTHS, "tg": TG, "seed": 3,
                "ffn_split_layers": profile::decode::FFN_LAYERS,
                "ffn_split_stages": profile::decode::FFN_STAGES,
                "sample_budget": profile::decode::MAX_SAMPLES,
                "capacity_warning": "2080 differs from predecessor capacity 8224; old rates are not matched causal controls",
                "ffn_coarse_view": "selected layers use min-start/max-end enclosing envelope of four samples, NOT their sum; includes internal gaps; alternative nonadditive view excluded from global geometry",
                "depth_seed": 3 ^ DEPTH_SEED_SALT, "roles": ROLES,
                "engram_layers": engrams, "census": false, "trace_capture": false,
                "scope": "one prefix prefill per nonzero depth, snapshot once, restore before all five windows outside timing and I/O scope",
                "wall_scope": "fixed 32-step loop including full-vocab parity; excludes restore, snapshots, profile begin/finish and JSON",
                "io_scope": "process envelope includes profile begin/finish, excludes restore",
                "oracle": "all 32 finite full-vocabulary rows bitwise vs same-candidate restored model warmup; not quality or cross-revision parity",
                "timing": "INCLUSIVE encoder-envelope durations are nonadditive; sampled interval union/envelope/overlap/gaps are not command coverage; command GPU and CPU encode/wait/wall remain separate",
                "overlap": "valid inclusive observation with exclusive=false; not proof of concurrent execution, hazard or clock defect",
                "limits": "attention operator composite unseparated; only selected FFNs split; routed includes routing/weight/sum and is not pure expert cost; no bandwidth or speedup claims",
                "overhead_limit_fraction": 0.10, "canonical": false
            })
        );
        Self {
            forward,
            ctx,
            vocab,
            engrams,
            next_depth: 0,
            rejected_depths: Vec::new(),
        }
    }

    pub(super) fn finish(&self, error: Option<&anyhow::Error>) -> Result<()> {
        let accepted =
            error.is_none() && self.next_depth == DEPTHS.len() && self.rejected_depths.is_empty();
        eprintln!(
            "{}",
            serde_json::json!({"diagnostic": LABEL, "schema": SCHEMA, "event": "complete",
            "observation_accepted": accepted, "completed_depths": self.next_depth,
            "rejected_depths": self.rejected_depths,
            "error": error.map(|e| format!("{e:#}"))})
        );
        ensure!(
            accepted,
            "PRIVATE decode diagnostic rejected; retain all raw attempts"
        );
        Ok(())
    }
}

impl FamilyBench for Bench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        bail!("PRIVATE use restored rows")
    }
    fn prefill(&mut self, _: &[u32]) -> Result<()> {
        bail!("PRIVATE use restored rows")
    }
    fn decode(&mut self, _: u32) -> Result<()> {
        bail!("PRIVATE use restored rows")
    }
    fn private_restored_decode(&self) -> bool {
        true
    }
    fn restored_decode_row(&mut self, depth: usize) -> Result<Vec<f64>> {
        ensure!(
            DEPTHS.get(self.next_depth) == Some(&depth),
            "PRIVATE unexpected depth order"
        );
        let mut session = RealSession {
            forward: self.forward,
            state: self.forward.new_state_with_gpu_step(CAPACITY)?,
        };
        let attempt = run_depth(&mut session, depth, self.vocab, 40, &self.engrams, || {
            DecodeObservation::now(self.ctx)
        })?;
        if !attempt.accepted {
            self.rejected_depths.push(depth);
        }
        self.next_depth += 1;
        Ok(attempt.wall)
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: crate::family_options::V41_PREFILL_MODE,
            decode_mode: "private_restored_seed3_gpu_step+logits_copy+ffn_observer_not_canonical",
            prefill_chunk: Some(crate::family_options::V41_PREFILL_ROWS),
        }
    }
}

trait Session {
    type Snapshot;
    fn position(&self) -> usize;
    fn capacity(&self) -> usize;
    fn prefill(&mut self, ids: &[u32]) -> Result<()>;
    fn snapshot(&self) -> Result<Self::Snapshot>;
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()>;
    fn step(&mut self, token: u32) -> Result<Vec<f32>>;
    fn counts(&self) -> Option<(usize, usize)>;
}

struct RealSession<'a, 'w> {
    forward: &'a V41Forward<'w>,
    state: forward::State,
}

impl Session for RealSession<'_, '_> {
    type Snapshot = forward::Snapshot;
    fn position(&self) -> usize {
        self.state.position()
    }
    fn capacity(&self) -> usize {
        self.state.capacity()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        self.forward
            .prefill(&mut self.state, ids, Some(shutdown::cancellation_flag()))?;
        Ok(())
    }
    fn snapshot(&self) -> Result<Self::Snapshot> {
        Ok(self.forward.capture_snapshot(&self.state)?)
    }
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()> {
        Ok(self.forward.restore_snapshot(&mut self.state, snapshot)?)
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.forward
            .step(&mut self.state, token, Some(shutdown::cancellation_flag()))
            .map_err(anyhow::Error::msg)
    }
    fn counts(&self) -> Option<(usize, usize)> {
        self.state.diagnostic_gpu_counts()
    }
}

#[derive(serde::Serialize)]
struct Step {
    position: usize,
    token: u32,
    forward_wall_ms: f64,
    logits_len: usize,
    finite: bool,
    bitwise_equal: Option<bool>,
    first_mismatch: Option<(usize, u32, u32)>,
    commands: Option<usize>,
    trace_copies: Option<usize>,
}

#[derive(serde::Serialize)]
struct Window {
    rep: usize,
    role: &'static str,
    start_position: usize,
    end_position: usize,
    capacity: usize,
    window_wall_ms: f64,
    before: DecodeObservation,
    after: DecodeObservation,
    physical_read_bytes: Option<u64>,
    pageins: Option<u64>,
    steps: Vec<Step>,
    profile: Option<profile::Report>,
    error: Option<String>,
}

#[derive(Default, serde::Serialize)]
struct StageStats {
    count: usize,
    sum_ms: f64,
    min_ms: Option<f64>,
    max_ms: Option<f64>,
}

impl StageStats {
    fn push(&mut self, ms: f64) {
        self.count += 1;
        self.sum_ms += ms;
        self.min_ms = Some(self.min_ms.map_or(ms, |old| old.min(ms)));
        self.max_ms = Some(self.max_ms.map_or(ms, |old| old.max(ms)));
    }
}

fn raw_stage_stats(report: &profile::Report) -> Option<BTreeMap<&'static str, StageStats>> {
    let mut stats = BTreeMap::<_, StageStats>::new();
    for op in report
        .operations
        .iter()
        .filter(|o| o.operation.starts_with("decode_"))
    {
        let ms = op.gpu_ms.filter(|ms| ms.is_finite() && *ms > 0.0)?;
        if op.invalid_gpu_timestamps != 0 || op.count != 1 {
            return None;
        }
        stats.entry(op.operation).or_default().push(ms);
    }
    (!stats.is_empty()).then_some(stats)
}

#[derive(serde::Serialize)]
struct PositionCost {
    position: usize,
    command_gpu_ms: f64,
    sampled_envelopes: profile::decode::EnvelopeSummary,
    encode_ms: f64,
    commit_wait_ms: f64,
    command_wall_ms: f64,
}

#[derive(serde::Serialize)]
struct Costs {
    positions: Vec<PositionCost>,
    inclusive_stage_stats_nonadditive: BTreeMap<&'static str, StageStats>,
    ffn_enclosing_views_nonadditive: Vec<FfnView>,
    exclusive: bool,
    expected_spans_per_token: usize,
    expected_samples_per_token: usize,
}

#[derive(serde::Serialize)]
struct FfnView {
    position: usize,
    layer: usize,
    reconstructed_from_split: bool,
    start_ns: u64,
    end_ns: u64,
    enclosing_ms: f64,
}

fn costs(
    report: &profile::Report,
    depth: usize,
    layers: usize,
    engrams: &[usize],
) -> Result<Costs> {
    let layout =
        profile::decode::layout(layers, engrams, false, true).map_err(anyhow::Error::msg)?;
    let by_key: BTreeMap<_, _> = report
        .operations
        .iter()
        .map(|o| ((o.operation, o.dimensions), o))
        .collect();
    ensure!(
        report.token_wall_ms.is_finite()
            && report.token_wall_ms > 0.0
            && report.operations.len() == TG * (layout.len() + 1)
            && by_key.len() == report.operations.len(),
        "missing, duplicate or unexpected profile records"
    );
    let mut result = Costs {
        positions: Vec::with_capacity(TG),
        inclusive_stage_stats_nonadditive: BTreeMap::new(),
        ffn_enclosing_views_nonadditive: Vec::with_capacity(TG * layers),
        exclusive: true,
        expected_spans_per_token: layout.len(),
        expected_samples_per_token: layout.len() * 2,
    };
    let valid_ms = |o: &profile::Operation| -> Result<f64> {
        ensure!(
            o.count == 1 && o.invalid_gpu_timestamps == 0,
            "invalid timestamp/count"
        );
        let ms = o.gpu_ms.context("unavailable GPU timestamps")?;
        ensure!(ms.is_finite() && ms > 0.0, "nonpositive/nonfinite GPU time");
        Ok(ms)
    };
    let intervals: BTreeMap<_, _> = report
        .decode_intervals
        .iter()
        .map(|r| ((r.operation, r.position, r.layer), r))
        .collect();
    ensure!(
        report.decode_intervals.len() == TG * layout.len()
            && intervals.len() == report.decode_intervals.len(),
        "missing or duplicate raw timestamp coverage"
    );
    for position in depth..depth + TG {
        let command = by_key
            .get(&("gpu_step", [position, 0, layers, 0]))
            .context("missing command")?;
        let command_gpu_ms = valid_ms(command)?;
        ensure!(
            command.completed_submits == 1
                && [command.encode_ms, command.commit_wait_ms, command.wall_ms]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0),
            "invalid command timing/submits"
        );
        let mut envelopes = Vec::with_capacity(layout.len());
        for &(name, layer) in &layout {
            let op = by_key
                .get(&(name, [position, layer, 1, 0]))
                .context("missing expected stage")?;
            let ms = valid_ms(op)?;
            ensure!(
                op.completed_submits == 0
                    && op.encode_ms == 0.0
                    && op.commit_wait_ms == 0.0
                    && op.wall_ms == 0.0,
                "stage includes unexpected CPU timing/submits"
            );
            let raw = intervals
                .get(&(name, position, layer))
                .context("missing expected raw interval")?;
            ensure!(
                raw.start_ns > 0 && raw.end_ns > raw.start_ns && raw.end_ns != u64::MAX,
                "invalid raw timestamp endpoints"
            );
            ensure!(
                ms == (raw.end_ns - raw.start_ns) as f64 / 1e6,
                "raw interval/operation mismatch"
            );
            envelopes.push((raw.start_ns, raw.end_ns));
            let stats = result
                .inclusive_stage_stats_nonadditive
                .entry(name)
                .or_default();
            stats.push(ms);
        }
        for layer in 0..layers {
            let split = profile::decode::FFN_LAYERS.contains(&layer);
            let names: &[&str] = if split {
                &profile::decode::FFN_STAGES
            } else {
                &["decode_ffn_composite_including_mhc"]
            };
            let mut start = u64::MAX;
            let mut end = 0;
            for name in names {
                let raw = intervals
                    .get(&(*name, position, layer))
                    .context("missing FFN view interval")?;
                start = start.min(raw.start_ns);
                end = end.max(raw.end_ns);
            }
            result.ffn_enclosing_views_nonadditive.push(FfnView {
                position,
                layer,
                reconstructed_from_split: split,
                start_ns: start,
                end_ns: end,
                enclosing_ms: (end - start) as f64 / 1e6,
            });
        }
        let sampled_envelopes =
            profile::decode::summarize_envelopes(&envelopes).map_err(anyhow::Error::msg)?;
        result.exclusive &= sampled_envelopes.exclusive;
        result.positions.push(PositionCost {
            position,
            command_gpu_ms,
            sampled_envelopes,
            encode_ms: command.encode_ms,
            commit_wait_ms: command.commit_wait_ms,
            command_wall_ms: command.wall_ms,
        });
    }
    Ok(result)
}

fn window_valid(w: &Window, depth: usize, vocab: usize, layers: usize, engrams: &[usize]) -> bool {
    w.rep < 5
        && w.role == ROLES[w.rep]
        && w.error.is_none()
        && w.start_position == depth
        && w.end_position == depth + TG
        && w.capacity == CAPACITY
        && w.steps.len() == TG
        && w.window_wall_ms.is_finite()
        && w.window_wall_ms > 0.0
        && w.physical_read_bytes.is_some()
        && w.pageins.is_some()
        && w.physical_read_bytes
            == counter_delta(w.before.physical_read_bytes, w.after.physical_read_bytes)
        && w.pageins == counter_delta(w.before.pageins, w.after.pageins)
        && (w.rep < 2 || (w.physical_read_bytes == Some(0) && w.pageins == Some(0)))
        && w.steps
            .iter()
            .zip(synthetic_ids(TG, vocab, 3))
            .enumerate()
            .all(|(i, (s, token))| {
                s.position == depth + i
                    && s.token == token
                    && s.logits_len == vocab
                    && s.finite
                    && s.forward_wall_ms.is_finite()
                    && s.forward_wall_ms > 0.0
                    && s.commands == Some(1)
                    && s.trace_copies == Some(0)
                    && (if w.rep == 0 {
                        s.bitwise_equal.is_none()
                    } else {
                        s.bitwise_equal == Some(true)
                    })
                    && s.first_mismatch.is_none()
            })
        && if matches!(w.rep, 1 | 3) {
            w.profile.as_ref().is_some_and(|r| {
                costs(r, depth, layers, engrams).is_ok()
                    && r.process_pageins.is_some()
                    && r.process_disk_read_bytes.is_some()
                    && (w.rep == 1
                        || (r.process_pageins == Some(0) && r.process_disk_read_bytes == Some(0)))
            })
        } else {
            w.profile.is_none()
        }
}

fn verdict(
    windows: &[Window],
    depth: usize,
    vocab: usize,
    layers: usize,
    engrams: &[usize],
) -> (Option<f64>, Vec<String>) {
    let mut reasons = Vec::new();
    if windows.iter().map(|w| w.rep).collect::<Vec<_>>() != [0, 1, 2, 3, 4] {
        reasons.push("incomplete warmups/control/profile/control".into());
    }
    for w in windows {
        if !window_valid(w, depth, vocab, layers, engrams) {
            reasons.push(format!("window {} failed admission", w.rep));
        }
    }
    let overhead = if windows.len() == 5 {
        let mean = (windows[2].window_wall_ms + windows[4].window_wall_ms) * 0.5;
        let value = windows[3].window_wall_ms / mean - 1.0;
        if mean.is_finite() && mean > 0.0 && value.is_finite() {
            if windows[3].window_wall_ms > 1.10 * mean {
                reasons.push("observer overhead exceeds 10%".into());
            }
            Some(value)
        } else {
            None
        }
    } else {
        None
    };
    if overhead.is_none() {
        reasons.push("observer overhead unavailable".into());
    }
    (overhead, reasons)
}

fn window<S: Session>(
    session: &mut S,
    depth: usize,
    vocab: usize,
    rep: usize,
    reference: &mut [u32],
    observe: &impl Fn() -> DecodeObservation,
) -> Window {
    let ids = synthetic_ids(TG, vocab, 3);
    let mut steps = Vec::with_capacity(TG);
    let start_position = session.position();
    let before = observe();
    let scope = if matches!(rep, 1 | 3) {
        profile::Scope::begin_stages().map(Some)
    } else {
        Ok(None)
    };
    let (scope, scope_error) = match scope {
        Ok(scope) => (scope, None),
        Err(e) => (None, Some(e)),
    };
    let start = Instant::now();
    let result = (|| -> Result<()> {
        if let Some(e) = scope_error {
            bail!(e);
        }
        ensure!(
            !qwen_llm::metal::dispatch_census_is_active(),
            "PRIVATE census must be disabled"
        );
        for (i, token) in ids.into_iter().enumerate() {
            shutdown::checkpoint()?;
            ensure!(
                session.position() == depth + i,
                "PRIVATE step position mismatch"
            );
            let timer = Instant::now();
            let logits = session.step(token)?;
            let forward_wall_ms = timer.elapsed().as_secs_f64() * 1e3;
            let counts = session.counts();
            let reference = &mut reference[i * vocab..(i + 1) * vocab];
            let first_mismatch = (rep > 0)
                .then(|| first_logits_mismatch(reference, &logits))
                .flatten();
            let finite = logits.iter().all(|v| v.is_finite());
            let bitwise_equal =
                (rep > 0).then_some(logits.len() == vocab && first_mismatch.is_none());
            if rep == 0 && logits.len() == vocab {
                for (dst, src) in reference.iter_mut().zip(&logits) {
                    *dst = src.to_bits();
                }
            }
            steps.push(Step {
                position: depth + i,
                token,
                forward_wall_ms,
                logits_len: logits.len(),
                finite,
                bitwise_equal,
                first_mismatch,
                commands: counts.map(|v| v.0),
                trace_copies: counts.map(|v| v.1),
            });
            ensure!(
                logits.len() == vocab && finite && session.position() == depth + i + 1,
                "PRIVATE invalid logits or position"
            );
        }
        Ok(())
    })();
    let window_wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let profile = scope.map(profile::Scope::finish);
    let after = observe();
    Window {
        rep,
        role: ROLES[rep],
        start_position,
        end_position: session.position(),
        capacity: session.capacity(),
        window_wall_ms,
        physical_read_bytes: counter_delta(before.physical_read_bytes, after.physical_read_bytes),
        pageins: counter_delta(before.pageins, after.pageins),
        before,
        after,
        steps,
        profile,
        error: result.err().map(|e| format!("{e:#}")),
    }
}

struct DepthAttempt {
    wall: Vec<f64>,
    accepted: bool,
}

fn run_depth<S: Session>(
    session: &mut S,
    depth: usize,
    vocab: usize,
    layers: usize,
    engrams: &[usize],
    observe: impl Fn() -> DecodeObservation,
) -> Result<DepthAttempt> {
    ensure!(
        session.position() == 0 && session.capacity() == CAPACITY && DEPTHS.contains(&depth),
        "PRIVATE initial state"
    );
    let prefix_start = Instant::now();
    if depth > 0 {
        session.prefill(&synthetic_ids(depth, vocab, 3 ^ DEPTH_SEED_SALT))?;
    }
    ensure!(session.position() == depth, "PRIVATE prefix endpoint");
    let snapshot = session.snapshot()?;
    eprintln!(
        "{}",
        serde_json::json!({"diagnostic": LABEL, "schema": SCHEMA, "event": "snapshot", "depth": depth,
        "capacity": CAPACITY, "prefix_prefills": usize::from(depth > 0),
        "prefix_and_snapshot_wall_ms": prefix_start.elapsed().as_secs_f64() * 1e3,
        "excluded_from_decode_timing_and_io": true})
    );
    let mut reference = vec![0u32; TG.checked_mul(vocab).context("reference extent overflow")?];
    let mut windows = Vec::with_capacity(5);
    let result = (|| -> Result<()> {
        for rep in 0..5 {
            session.restore(&snapshot)?;
            ensure!(
                session.position() == depth && session.capacity() == CAPACITY,
                "PRIVATE restored endpoint/capacity"
            );
            let w = window(session, depth, vocab, rep, &mut reference, &observe);
            let costs = w.profile.as_ref().map(|p| costs(p, depth, layers, engrams));
            eprintln!(
                "{}",
                serde_json::json!({"diagnostic": LABEL, "schema": SCHEMA, "event": "raw_window", "depth": depth,
                "window": w, "costs": costs.as_ref().and_then(|c| c.as_ref().ok()),
                "raw_stage_stats_not_exclusive": w.profile.as_ref().and_then(raw_stage_stats),
                "raw_stage_stats_warning": "INCLUSIVE encoder-envelope durations, nonadditive; observation acceptance requires complete bracket; overlap only disables exclusive partition",
                "costs_error": costs.as_ref().and_then(|c| c.as_ref().err()).map(|e| e.to_string()),
                "observation_status": "pending_complete_bracket"})
            );
            let failed = w.error.is_some();
            windows.push(w);
            ensure!(!failed, "PRIVATE failed decode window; no retry");
        }
        Ok(())
    })();
    let (overhead, mut reasons) = verdict(&windows, depth, vocab, layers, engrams);
    if let Err(e) = result {
        reasons.push(format!("{e:#}"));
    }
    let exclusive = windows
        .iter()
        .find(|w| w.rep == 3)
        .and_then(|w| w.profile.as_ref())
        .and_then(|r| costs(r, depth, layers, engrams).ok())
        .map(|c| c.exclusive);
    eprintln!(
        "{}",
        serde_json::json!({"diagnostic": LABEL, "schema": SCHEMA, "event": "observation_verdict", "depth": depth,
        "observation_accepted": reasons.is_empty(), "exclusive": exclusive, "rejection_reasons": reasons,
        "profile_overhead_fraction_vs_mean_controls": overhead,
        "warning": "inclusive encoder envelopes are nonadditive; exclusive describes sampled envelopes only, not command coverage; restored fixed diagnostic, not canonical throughput"})
    );
    ensure!(
        windows.len() == 5 && windows.iter().all(|w| w.error.is_none()),
        "PRIVATE incomplete depth {depth}; raw attempts retained"
    );
    Ok(DepthAttempt {
        wall: windows[2..].iter().map(|w| w.window_wall_ms).collect(),
        accepted: reasons.is_empty(),
    })
}

#[derive(serde::Serialize)]
struct DecodeObservation {
    unix_s: Option<f64>,
    pid: u32,
    physical_read_bytes: Option<u64>,
    pageins: Option<u64>,
    rss_bytes: Option<u64>,
    footprint_bytes: Option<u64>,
    metal_allocated_bytes: u64,
    thermal_state: isize,
    error: Option<String>,
}

impl DecodeObservation {
    fn now(ctx: &MetalContext) -> Self {
        let snapshot = qwen_llm::pid_metrics::PidSnapshot::now();
        let pid = snapshot.as_ref().ok();
        Self {
            unix_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs_f64()),
            pid: std::process::id(),
            physical_read_bytes: pid.map(|p| p.diskio_bytesread),
            pageins: pid.map(|p| p.pageins),
            rss_bytes: pid.map(|p| p.resident_size),
            footprint_bytes: pid.map(|p| p.phys_footprint),
            metal_allocated_bytes: ctx.current_allocated_size(),
            thermal_state: objc2_foundation::NSProcessInfo::processInfo()
                .thermalState()
                .0,
            error: snapshot.err().map(|e| e.to_string()),
        }
    }
}

fn counter_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    after?.checked_sub(before?)
}

fn first_logits_mismatch(reference: &[u32], logits: &[f32]) -> Option<(usize, u32, u32)> {
    reference
        .iter()
        .zip(logits)
        .enumerate()
        .find_map(|(i, (&expected, actual))| {
            (expected != actual.to_bits()).then_some((i, expected, actual.to_bits()))
        })
}
