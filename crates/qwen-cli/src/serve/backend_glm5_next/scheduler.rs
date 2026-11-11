//! GLM adapter for the family-neutral owner scheduler. Cache and checkpoint
//! decisions stay in the same resumable phases used by serial serving.
use super::super::{
    batching::driver::{Begun, Driver, Runtime},
    decode_cursor::{Advance, DecodeCursor},
};
use super::*;
use qwen_llm::glm5_next_metal::{GLM5_NEXT_MAX_BATCH_SIZE, Glm5NextBatch};
use std::collections::VecDeque;

/// Positive startup limit; default four on supported M5 Fast/Half. One keeps
/// the original serial owner path and its single live session.
const MAX_ACTIVE: &str = "QWEN_SERVE_GLM_MAX_ACTIVE";
/// Idle live sessions retained for exact-prefix reuse; default four when
/// batching is enabled. Optional states are released after snapshot relief.
const LIVE_STATES: &str = "QWEN_SERVE_GLM_LIVE_STATES";
/// Keep the admitted normal prefill geometry under load; 0 selects 512-row geometry. Layer slicing has its own rollback.
const PREFILL_FAIR: &str = "QWEN_SERVE_GLM_PREFILL_FAIR";
/// Target decode service percentage under mixed load for unsliced appends (1..=90, default 10).
const DECODE_PERCENT: &str = "QWEN_SERVE_GLM_DECODE_PERCENT";
/// Nominal GPU milliseconds per row-preserving layer slice under mixed load (default 0; 200 opts into selective slicing). Zero restores whole-row chunks and the original decode-percentage policy.
const LAYER_SLICE_MS: &str = "QWEN_SERVE_GLM_LAYER_SLICE_MS";
// Keep opt-in: the selective 500ms/200ms packet also regressed global p99.
const DEFAULT_LAYER_SLICE_MS: usize = 0;
/// Slice only appends whose predicted original prefill wall exceeds this many milliseconds (default 500). Zero restores the all-append slicing policy; LAYER_SLICE_MS=0 still disables slicing entirely.
const LAYER_SLICE_MIN_MS: &str = "QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS";
const DEFAULT_LAYER_SLICE_MIN_MS: usize = 500;
/// Persistent threads that sample a multi-member step beside the owner; 0
/// samples serially on the owner (rollback). Default: max_active - 1.
const SAMPLE_WORKERS: &str = "QWEN_SERVE_GLM_SAMPLE_WORKERS";
qwen_llm::env_flag!(
    /// Run the whole active GLM batch through the one-ahead device-feedback token loop: GPU sampling and pipelined forwarding. 0 keeps per-round host sampling.
    default_on batch_loop_enabled,
    "QWEN_SERVE_GLM_BATCH_GPU_LOOP"
);
#[derive(Clone, Copy)]
pub(super) struct Config {
    pub active: usize,
    pub live: usize,
    pub fair: bool,
    pub decode_percent: usize,
    pub layer_slice_ms: usize,
    pub layer_slice_min_ms: usize,
    pub sample_workers: usize,
}
fn parse(key: &str, value: Option<&str>) -> Result<usize> {
    let n = match value {
        None => 4,
        Some(s) => s
            .parse::<usize>()
            .with_context(|| format!("{key} must be 1..=4"))?,
    };
    anyhow::ensure!(
        (1..=GLM5_NEXT_MAX_BATCH_SIZE).contains(&n),
        "{key} must be 1..=4"
    );
    Ok(n)
}
fn parse_percent(value: Option<&str>) -> Result<usize> {
    let n = value
        .map_or(Ok(10), |s| s.parse::<usize>())
        .context("QWEN_SERVE_GLM_DECODE_PERCENT must be 1..=90")?;
    anyhow::ensure!(
        (1..=90).contains(&n),
        "QWEN_SERVE_GLM_DECODE_PERCENT must be 1..=90"
    );
    Ok(n)
}
fn parse_slice_ms(value: Option<&str>) -> Result<usize> {
    let ms = value
        .map_or(Ok(DEFAULT_LAYER_SLICE_MS), str::parse::<usize>)
        .context("QWEN_SERVE_GLM_LAYER_SLICE_MS must be 0..=10000")?;
    anyhow::ensure!(
        ms <= 10000,
        "QWEN_SERVE_GLM_LAYER_SLICE_MS must be 0..=10000"
    );
    Ok(ms)
}
fn parse_slice_min_ms(value: Option<&str>) -> Result<usize> {
    let ms = value
        .map_or(Ok(DEFAULT_LAYER_SLICE_MIN_MS), str::parse::<usize>)
        .context("QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS must be 0..=1000000")?;
    anyhow::ensure!(
        ms <= 1_000_000,
        "QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS must be 0..=1000000"
    );
    Ok(ms)
}
fn parse_workers(key: &str, value: Option<&str>) -> Result<usize> {
    let n = match value {
        None => return Ok(usize::MAX), // resolved to max_active - 1 by the caller
        Some(s) => s
            .parse::<usize>()
            .with_context(|| format!("{key} must be a non-negative integer"))?,
    };
    Ok(n)
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            fair: qwen_llm::env_flag::read_default_on(PREFILL_FAIR),
            decode_percent: parse_percent(std::env::var(DECODE_PERCENT).ok().as_deref())?,
            layer_slice_ms: parse_slice_ms(std::env::var(LAYER_SLICE_MS).ok().as_deref())?,
            layer_slice_min_ms: parse_slice_min_ms(
                std::env::var(LAYER_SLICE_MIN_MS).ok().as_deref(),
            )?,
            active: parse(MAX_ACTIVE, std::env::var(MAX_ACTIVE).ok().as_deref())?,
            live: parse(LIVE_STATES, std::env::var(LIVE_STATES).ok().as_deref())?,
            sample_workers: parse_workers(
                SAMPLE_WORKERS,
                std::env::var(SAMPLE_WORKERS).ok().as_deref(),
            )?,
        })
    }
}

pub(super) struct Idle<'w> {
    pub session: Glm5NextSession<'w>,
    pub history: Vec<u32>,
    pub canonical: bool,
    pub origin: Option<chunk_checkpoint::ChunkOrigin>,
}

impl<'w, 'g> Glm5NextBackend<'w, 'g> {
    pub(super) fn enable_batching(&mut self) -> Result<()> {
        let config = self.prepared.batching;
        if config.active == 1
            || self.lineage != PackedLineage::Fast
            || self.prepared.fast_precision != FastPrecision::Half
            || self.drafter.is_some()
            || self.prepared.drafter.is_some()
            || !Glm5NextBatch::supported(self.ctx, self.weights)
        {
            return Ok(());
        }
        let mut batch = match Glm5NextBatch::new(self.ctx, self.weights, config.active) {
            Ok(b) => b,
            Err(Glm5NextMetalError::MemoryAdmission { denied, .. })
                if denied.reason().is_pressure() =>
            {
                tracing::warn!(target:"qwen_diag","serve: glm5_next batching declined by memory admission; keeping serial owner");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        // Two columns compile the same few-row PSOs as widths 3/4. Tiny
        // throwaway caches avoid reserving four full-context sessions at boot.
        let mut a = Glm5NextSession::with_prefill_rows(self.ctx, self.weights, 16, 4)?;
        let mut b = Glm5NextSession::with_prefill_rows(self.ctx, self.weights, 16, 4)?;
        let tokens = self
            .prepared
            .artifact
            .tokenizer()
            .encode("[gMASK]<sop>", false)?;
        let token = u32::try_from(tokens[0])?;
        batch.step(self.ctx, &mut [&mut a, &mut b], &[token, token])?;
        a.forward(self.ctx, token)?;
        self.batch = Some(batch);
        let sample_workers = if config.sample_workers == usize::MAX {
            config.active.saturating_sub(1)
        } else {
            config.sample_workers
        };
        let driver = Driver::new(config.active)
            .with_prefill_policy(config.fair, config.decode_percent)
            .with_sample_workers(sample_workers)?;
        let sample_workers = driver.sample_workers();
        self.scheduler = Some(driver);
        if config.layer_slice_ms > 0 {
            tracing::info!(target:"qwen_diag", "serve layer prefill: family=glm5_next target_gpu_ms={} minimum_predicted_ms={} rows=original", config.layer_slice_ms, config.layer_slice_min_ms);
        }
        tracing::info!(target:"qwen_diag","serve batch: family=glm5_next max_active={} live_states={} sample_workers={} arithmetic=fast_half",config.active,config.live,sample_workers);
        Ok(())
    }
    pub(super) fn take_idle(&mut self, tokens: &[u32], lineage: PackedLineage) {
        if self.session.is_some() {
            return;
        }
        let best = self
            .idle_sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.session.packed_lineage() == Some(lineage)
                    && (lineage != PackedLineage::Fast
                        || s.session.fast_precision() == Some(self.prepared.fast_precision))
            })
            .max_by_key(|(i, s)| {
                (
                    decode_loop::extending_prefix(
                        &s.history,
                        tokens,
                        s.session.position(),
                        self.prefix_reuse,
                    ),
                    std::cmp::Reverse(*i),
                )
            })
            .map(|(i, _)| i);
        if let Some(i) = best {
            let s = self.idle_sessions.remove(i).expect("idle index");
            self.session = Some(s.session);
            self.history = s.history;
            self.live_canonical = s.canonical;
            self.chunk_origin = s.origin;
        }
    }
    fn keep_idle(&mut self) {
        if let Some(session) = self.session.take() {
            let history = std::mem::take(&mut self.history);
            if !session.is_poisoned() && !history.is_empty() && history.len() == session.position()
            {
                self.idle_sessions.push_back(Idle {
                    session,
                    history,
                    canonical: self.live_canonical,
                    origin: self.chunk_origin.take(),
                });
                while self.idle_sessions.len() > self.prepared.batching.live {
                    self.idle_sessions.pop_front();
                }
            }
        }
        self.history.clear();
        self.live_canonical = false;
        self.chunk_origin = None;
    }
    fn attach(&mut self, run: &mut execution::Run<'w>) {
        debug_assert!(self.session.is_none());
        self.session = run.session.take();
    }
    fn detach(&mut self, run: &mut execution::Run<'w>) {
        run.session = self.session.take();
        self.history.clear();
        self.live_canonical = false;
        self.chunk_origin = None;
    }
    pub(super) fn pump_batch(&mut self) -> bool {
        let Some(mut driver) = self.scheduler.take() else {
            return false;
        };
        let progress = driver.pump(&mut Adapter(self));
        if driver.take_native_failure() {
            self.idle_residency.request_failed_on_server();
        }
        if progress {
            self.idle_residency.note_activity();
        }
        self.scheduler = Some(driver);
        progress
    }
}
struct Adapter<'a, 'w, 'g>(&'a mut Glm5NextBackend<'w, 'g>);
impl<'w> Runtime for Adapter<'_, 'w, '_> {
    type Run = execution::Run<'w>;
    type State = Glm5NextSession<'w>;
    fn extra_reserve(&self) -> u64 {
        self.0
            .batch
            .as_ref()
            .map_or(0, Glm5NextBatch::cpu_reserve_bytes)
    }
    fn begin(
        &mut self,
        p: &PreparedResponse,
        sink: &mut dyn GenerationSink,
    ) -> Result<Begun<Self::Run>, BackendFailure> {
        match self.0.begin_generation(
            p.request(),
            &p.prompt,
            p.boundaries,
            p.tokens.as_deref(),
            sink,
        ) {
            Ok(execution::Begun::Complete(o)) => Ok(Begun::Complete(o)),
            Ok(execution::Begun::Run(mut r)) => {
                r.session = self.0.session.take();
                Ok(Begun::Run(r))
            }
            Err(e) => {
                self.0.keep_idle();
                Err(e)
            }
        }
    }
    fn rows(&self, run: &Self::Run) -> usize {
        run.selected_rows
    }
    fn prefill(
        &mut self,
        run: &mut Self::Run,
        rows: usize,
        sink: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Result<Option<DecodeCursor>, BackendFailure> {
        run.reserve = reserve;
        if let Some(c) = &mut run.chunk_capture {
            c.reserve = reserve;
        }
        self.0.attach(run);
        let result = (|| {
            let start = Instant::now();
            let before = run.capture_ms + run.chunk_capture.as_ref().map_or(0.0, |c| c.elapsed_ms);
            let result = self.0.prefill_run(run, rows, sink);
            let after = run.capture_ms + run.chunk_capture.as_ref().map_or(0.0, |c| c.elapsed_ms);
            if self.0.prepared.batching.layer_slice_ms > 0 {
                run.layer_work
                    .note(start.elapsed().as_secs_f64() * 1e3 - (after - before));
            }
            if !result? {
                return Ok(None);
            }
            self.0.finish_prefill(run, sink)?;
            DecodeCursor::new(decode_loop::DecodeRequest {
                family: FAMILY,
                logits: std::mem::take(&mut run.logits),
                max_tokens: run.maximum,
                stop_tokens: &CHAT_STOPS,
                vocab_size: run.vocab_size,
            })
            .map(Some)
        })();
        self.0.detach(run);
        result
    }
    fn prefill_sliced(
        &mut self,
        run: &mut Self::Run,
        rows: usize,
        sink: &mut dyn GenerationSink,
        reserve: u64,
        peers: bool,
    ) -> Option<Result<Option<DecodeCursor>, BackendFailure>> {
        let ms = self.0.prepared.batching.layer_slice_ms;
        if ms == 0 || (!peers && !run.layer_work.armed) {
            return None;
        }
        if !run.layer_work.armed {
            let predicted_ms = layer_prefill::predicted_append_ms(
                run.matched,
                &run.ends,
                rows.min(run.selected_rows).max(1),
            );
            if !layer_prefill::qualifies(predicted_ms, self.0.prepared.batching.layer_slice_min_ms)
            {
                return None;
            }
        }
        // Batching already requires M5 Fast/Half; a serial/unequal workspace
        // that cannot hold the original row quantum keeps the legacy path.
        if !run.layer_work.armed
            && run.session.as_ref().is_none_or(|s| {
                s.prefill_rows() == 0
                    || s.packed_lineage() != Some(PackedLineage::Fast)
                    || s.fast_precision() != Some(FastPrecision::Half)
            })
        {
            return None;
        }
        let _qos = qwen_llm::metal::GpuThreadQos::enter();
        run.reserve = reserve;
        if let Some(c) = &mut run.chunk_capture {
            c.reserve = reserve;
        }
        self.0.attach(run);
        let result = (|| {
            if !self
                .0
                .prefill_slice_run(run, rows, sink, ms as f64, peers)?
            {
                return Ok(None);
            }
            self.0.finish_prefill(run, sink)?;
            DecodeCursor::new(decode_loop::DecodeRequest {
                family: FAMILY,
                logits: std::mem::take(&mut run.logits),
                max_tokens: run.maximum,
                stop_tokens: &CHAT_STOPS,
                vocab_size: run.vocab_size,
            })
            .map(Some)
        })();
        self.0.detach(run);
        Some(result)
    }
    fn advance(
        &mut self,
        run: &mut Self::Run,
        cursor: &mut DecodeCursor,
        sink: &mut dyn GenerationSink,
        reserve: u64,
        presampled: Option<crate::serve::backend_v41::sample_pool::Sampled>,
    ) -> Result<Advance, BackendFailure> {
        run.reserve = reserve;
        cursor.advance(
            &mut run.sampler,
            presampled,
            self.0.prepared.artifact.tokenizer(),
            sink,
            crate::shutdown::checkpoint,
        )
    }
    fn sampler_mut(run: &mut Self::Run) -> Option<&mut qwen_llm::sampling::Sampler> {
        Some(&mut run.sampler)
    }
    fn advance_single(
        &mut self,
        run: &mut Self::Run,
        cursor: &mut DecodeCursor,
        sink: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Option<Result<Advance, BackendFailure>> {
        run.reserve = reserve;
        gpu::advance(
            self.0.ctx,
            self.0.prepared.artifact.tokenizer(),
            run.session.as_mut().expect("request owns session"),
            &mut run.sampler,
            cursor,
            sink,
            16,
        )
    }
    fn advance_group(
        &mut self,
        runs: &mut [&mut Self::Run],
        cursors: &mut [&mut DecodeCursor],
        sinks: &mut [&mut dyn GenerationSink],
        tokens: &[u32],
        reserves: &[u64],
    ) -> Option<Result<Vec<Advance>, BackendFailure>> {
        let b = runs.len();
        if !batch_loop_enabled()
            || b < 2
            || self.0.batch.is_none()
            || cursors.len() != b
            || sinks.len() != b
            || tokens.len() != b
            || reserves.len() != b
        {
            return None;
        }
        let vocab = self.0.weights.config.vocab_size as usize;
        for (i, run) in runs.iter().enumerate() {
            let Some(session) = run.session.as_ref() else {
                return None;
            };
            if session.is_poisoned()
                || session.position() >= session.ledger().capacity() as usize
                || tokens[i] as usize >= vocab
            {
                return None;
            }
        }
        for (run, reserve) in runs.iter_mut().zip(reserves) {
            run.reserve = *reserve;
        }
        // Whole burst forwards, samples and waits on this thread; nested with
        // the driver guard.
        let _qos = qwen_llm::metal::GpuThreadQos::enter();
        let offset: Vec<f64> = cursors.iter().map(|c| c.elapsed_ms()).collect();
        let counts: Vec<usize> = cursors
            .iter()
            .map(|c| c.remaining_tokens().min(8))
            .collect();
        let eos: Vec<&[i32]> = vec![&CHAT_STOPS[..]; b];
        let tokenizer = self.0.prepared.artifact.tokenizer();
        let mut burst: Vec<Vec<i32>> = vec![Vec::new(); b];
        let mut callback_error: Option<BackendFailure> = None;
        let result = {
            let mut sessions: Vec<&mut Glm5NextSession<'_>> = Vec::with_capacity(b);
            let mut samplers: Vec<&mut qwen_llm::sampling::Sampler> = Vec::with_capacity(b);
            for run in runs.iter_mut() {
                sessions.push(run.session.as_mut().expect("request owns session"));
                samplers.push(&mut run.sampler);
            }
            self.0.batch.as_mut().unwrap().generate_gpu_tokens(
                self.0.ctx,
                &mut sessions,
                &mut samplers,
                tokens,
                &counts,
                &eos,
                |member, token| {
                    burst[member].push(token as i32);
                    let result = (|| -> Result<bool, BackendFailure> {
                        crate::shutdown::checkpoint()
                            .map_err(|e| ServeError::server_error(e.to_string()))?;
                        let bytes = tokenizer
                            .try_decode_piece_bytes_exact(token as i32)
                            .map_err(|e| {
                                ServeError::server_error(format!("decode {FAMILY}: {e}"))
                            })?;
                        sinks[member]
                            .piece(bytes)
                            .and_then(|()| sinks[member].tick())
                            .map_err(BackendFailure::Aborted)?;
                        let ready = sinks[member]
                            .flush_ready()
                            .map_err(BackendFailure::Aborted)?;
                        crate::shutdown::checkpoint()
                            .map_err(|e| ServeError::server_error(e.to_string()))?;
                        Ok(ready)
                    })();
                    match result {
                        Ok(ready) => ready,
                        Err(e) => {
                            callback_error = Some(e);
                            false
                        }
                    }
                },
            )
        };
        if let Some(error) = callback_error {
            return Some(Err(error));
        }
        let stats = match result {
            Ok(stats) => stats,
            Err(error) => {
                return Some(Err(ServeError::server_error(format!(
                    "{FAMILY} GPU batch loop: {error}"
                ))
                .into()));
            }
        };
        let sampled = stats.sampled_tokens / b;
        let forwarded = stats.forwarded_tokens / b;
        let per_round = stats.wall_ms / sampled.max(1) as f64;
        // The first frame forwarded the pending tokens and sampled on the GPU,
        // so complete the pending transition without retaining logits.
        for cursor in cursors.iter_mut() {
            if let Err(error) = cursor.on_token(stats.first_frame_ms, crate::shutdown::checkpoint) {
                return Some(Err(error));
            }
        }
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let terminal = stats.terminal_tokens.get(i).copied().unwrap_or(-1);
            let stopped_on_eos = CHAT_STOPS.contains(&terminal);
            if stopped_on_eos {
                burst[i].push(terminal);
            }
            let member = qwen_llm::metal::GpuTokenLoopStats {
                sampled_tokens: burst[i].len(),
                forwarded_tokens: burst[i].len().saturating_sub(1),
                emitted_tokens: burst[i].len() - usize::from(stopped_on_eos),
                encode_cpu_ms: stats.encode_cpu_ms / b as f64,
                completion_wait_ms: stats.completion_wait_ms / b as f64,
                gpu_ms: stats.gpu_ms / b as f64,
                wall_ms: stats.wall_ms,
                terminal_token: Some(terminal),
                stopped_on_eos,
                first_token_ready_ms: None,
                first_token_callback_ms: None,
                first_token_selection_ms: 0.,
                transition_ms: stats.wall_ms * forwarded as f64 / sampled.max(1) as f64,
                first_transition_ms: (forwarded > 0).then_some(per_round),
                ..Default::default()
            };
            out.push(cursors[i].append_gpu_burst(&burst[i], member, offset[i]));
        }
        Some(Ok(out))
    }
    fn state(run: &mut Self::Run) -> &mut Self::State {
        run.session.as_mut().expect("request owns session")
    }
    fn can_step(&self, state: &Self::State) -> bool {
        !state.is_poisoned() && state.position() < (state.ledger().capacity() as usize)
    }
    fn poisoned(state: &Self::State) -> bool {
        state.is_poisoned()
    }
    fn step(&mut self, state: &mut Self::State, token: u32) -> Result<Vec<f32>, String> {
        crate::shutdown::checkpoint().map_err(|e| e.to_string())?;
        let v = state
            .forward(self.0.ctx, token)
            .map_err(|e| e.to_string())?;
        if v.len() != self.0.weights.config.vocab_size as usize || v.iter().any(|x| !x.is_finite())
        {
            return Err("GLM returned malformed logits".into());
        }
        Ok(v)
    }
    fn step_batch(
        &mut self,
        states: &mut [&mut Self::State],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        crate::shutdown::checkpoint().map_err(|e| e.to_string())?;
        self.0
            .batch
            .as_mut()
            .expect("batch enabled")
            .step(self.0.ctx, states, tokens)
            .map_err(|e| e.to_string())
    }
    fn finish(
        &mut self,
        mut run: Self::Run,
        generation: &crate::GenerationResult,
        batch: serde_json::Value,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.0.attach(&mut run);
        let phases = serde_json::json!({"prefill_ms":run.prefill_ms,"decode_ms":generation.wall_ms,"reused_tokens":run.matched,"prefill_tokens":run.tokens.len()-run.matched,"restore_ms":run.restore_ms,"reuse_source":run.reuse_source,"restore_origin":run.restore_origin,"lineage":lineage_name(run.lineage),"layer_slices":run.layer_work.commands,"layer_gpu_ms":run.layer_work.gpu_ms,"batch":batch});
        let result = self.0.finish_run(run, generation);
        if result.is_ok() {
            self.0.keep_idle();
        } else {
            self.0.session = None;
            self.0.history.clear();
            self.0.chunk_origin = None;
        }
        result.map(|mut o| {
            if let Some(s) = &mut o.stats {
                s.phases = Some(phases);
            }
            o
        })
    }
}

/// Snapshot relief comes first; then only unused idle sessions are evicted.
/// Active sessions live in their runs and cannot be released by admission.
pub(super) fn with_session_relief<'w, T, V>(
    cache: &mut SnapshotCache<V, CacheNamespace>,
    idle: &mut VecDeque<Idle<'w>>,
    keep: Option<EntryId>,
    mut attempt: impl FnMut() -> Result<T, Glm5NextMetalError>,
) -> Result<T, Glm5NextMetalError> {
    loop {
        let result = with_snapshot_release(cache, keep, &mut attempt);
        let pressure = matches!(&result,Err(Glm5NextMetalError::MemoryAdmission{denied,..})if denied.reason().is_pressure());
        if pressure && idle.pop_front().is_some() {
            continue;
        }
        return result;
    }
}
pub(super) fn with_transport_relief<'w, T, V>(
    cache: &mut SnapshotCache<V, CacheNamespace>,
    idle: &mut VecDeque<Idle<'w>>,
    keep: Option<EntryId>,
    mut attempt: impl FnMut() -> Result<T, super::super::transport_memory::CpuMemoryRefusal>,
) -> Result<T, ServeError> {
    loop {
        let result = with_transport_snapshot_release(cache, keep, &mut attempt);
        if result.as_ref().err().is_some_and(is_pressure_refusal) && idle.pop_front().is_some() {
            continue;
        }
        return result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_slice_ms_has_an_exact_zero_rollback_and_bounded_input() {
        assert_eq!(parse_slice_ms(None).unwrap(), DEFAULT_LAYER_SLICE_MS);
        for n in ["0", "1", "100", "10000"] {
            assert!(parse_slice_ms(Some(n)).is_ok());
        }
        for n in ["10001", "-1", "x", "1.5", ""] {
            assert!(parse_slice_ms(Some(n)).is_err());
        }
    }
    #[test]
    fn selective_slice_threshold_is_bounded_and_has_an_all_append_rollback() {
        assert_eq!(parse_slice_min_ms(None).unwrap(), 500);
        assert_eq!(parse_slice_min_ms(Some("0")).unwrap(), 0);
        assert_eq!(parse_slice_ms(Some("0")).unwrap(), 0);
        for n in ["1", "500", "1000000"] {
            assert!(parse_slice_min_ms(Some(n)).is_ok());
        }
        for n in ["1000001", "-1", "1.5", "x", ""] {
            assert!(parse_slice_min_ms(Some(n)).is_err());
        }
    }
    #[test]
    fn glm_batch_limits_are_bounded_and_have_an_explicit_serial_rollback() {
        assert_eq!(parse_percent(None).unwrap(), 10);
        for n in ["1", "10", "90"] {
            assert!(parse_percent(Some(n)).is_ok());
        }
        for n in ["0", "91", "x", "-1", "1.5", ""] {
            assert!(parse_percent(Some(n)).is_err());
        }
        assert_eq!(parse(MAX_ACTIVE, None).unwrap(), 4);
        assert_eq!(parse(MAX_ACTIVE, Some("1")).unwrap(), 1);
        for value in ["0", "5", "-1", "x", ""] {
            assert!(parse(MAX_ACTIVE, Some(value)).is_err());
        }
    }
}
