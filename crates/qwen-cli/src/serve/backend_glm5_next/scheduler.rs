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
/// Keep the admitted normal prefill geometry under load; 0 restores 512-row alternation.
const PREFILL_FAIR: &str = "QWEN_SERVE_GLM_PREFILL_FAIR";
/// Target decode service percentage under mixed load (1..=90, default 10).
const DECODE_PERCENT: &str = "QWEN_SERVE_GLM_DECODE_PERCENT";
#[derive(Clone, Copy)]
pub(super) struct Config {
    pub active: usize,
    pub live: usize,
    pub fair: bool,
    pub decode_percent: usize,
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
impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            fair: qwen_llm::env_flag::read_default_on(PREFILL_FAIR),
            decode_percent: parse_percent(std::env::var(DECODE_PERCENT).ok().as_deref())?,
            active: parse(MAX_ACTIVE, std::env::var(MAX_ACTIVE).ok().as_deref())?,
            live: parse(LIVE_STATES, std::env::var(LIVE_STATES).ok().as_deref())?,
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
        self.scheduler = Some(
            Driver::new(config.active).with_prefill_policy(config.fair, config.decode_percent),
        );
        tracing::info!(target:"qwen_diag","serve batch: family=glm5_next max_active={} live_states={} arithmetic=fast_half",config.active,config.live);
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
        match self
            .0
            .begin_generation(p.request(), &p.prompt, p.boundaries, sink)
        {
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
            if !self.0.prefill_run(run, rows, sink)? {
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
    fn advance(
        &mut self,
        run: &mut Self::Run,
        cursor: &mut DecodeCursor,
        sink: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Result<Advance, BackendFailure> {
        run.reserve = reserve;
        cursor.advance(
            &mut run.sampler,
            None,
            self.0.prepared.artifact.tokenizer(),
            sink,
            crate::shutdown::checkpoint,
        )
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
        let phases = serde_json::json!({"prefill_ms":run.prefill_ms,"decode_ms":generation.wall_ms,"reused_tokens":run.matched,"prefill_tokens":run.tokens.len()-run.matched,"restore_ms":run.restore_ms,"reuse_source":run.reuse_source,"restore_origin":run.restore_origin,"lineage":lineage_name(run.lineage),"batch":batch});
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
