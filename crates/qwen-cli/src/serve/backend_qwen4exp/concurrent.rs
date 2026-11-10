//! Flash-Next adapter for the shared owner driver. Ordinary forwards are the
//! initial decode-group seam; every request retains its own causal runner.
use super::*;
use crate::serve::{
    batching::{
        self,
        driver::{Begun, Driver, Runtime},
    },
    decode_cursor::{Advance, DecodeCursor},
    http::PreparedResponse,
};
use qwen_llm::qwen4exp_runtime::{Qwen4ExpDecodeBatch, Qwen4ExpSessionFactory};
qwen_llm::env_flag!(
    /// Share independent Flash-Next M5 decode HC/dense/head work. 0 retains the fair owner with ordinary member forwards. MAX_ACTIVE=1 restores serial serving.
    default_on native_enabled, "QWEN_SERVE_FN_NATIVE_BATCH"
);
use std::collections::VecDeque;

type Runner<'a> = Qwen4ExpTextRunner<'a, 'a, 'static>;

/// Maximum concurrent requests; 1 retains the original single-live-state owner.
const MAX_ACTIVE: &str = "QWEN_SERVE_FN_MAX_ACTIVE";
/// Optional idle live states, default four; active states are never evicted.
const LIVE_STATES: &str = "QWEN_SERVE_FN_LIVE_STATES";
/// Decode service share under mixed prefill load, default ten percent.
const DECODE_PERCENT: &str = "QWEN_SERVE_FN_DECODE_PERCENT";
#[derive(Clone, Copy)]
pub(super) struct Config {
    pub active: usize,
    live: usize,
    decode_percent: usize,
}
fn bounded(
    key: &str,
    value: Option<&str>,
    default: usize,
    maximum: usize,
) -> anyhow::Result<usize> {
    let n = value
        .map_or(Ok(default), str::parse::<usize>)
        .with_context(|| format!("{key} must be 1..={maximum}"))?;
    anyhow::ensure!((1..=maximum).contains(&n), "{key} must be 1..={maximum}");
    Ok(n)
}
impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            active: bounded(MAX_ACTIVE, std::env::var(MAX_ACTIVE).ok().as_deref(), 4, 4)?,
            live: bounded(
                LIVE_STATES,
                std::env::var(LIVE_STATES).ok().as_deref(),
                4,
                4,
            )?,
            decode_percent: bounded(
                DECODE_PERCENT,
                std::env::var(DECODE_PERCENT).ok().as_deref(),
                10,
                90,
            )?,
        })
    }
}

pub(super) struct State<'a> {
    runner: Runner<'a>,
    failed: bool,
}
struct Idle<'a> {
    state: State<'a>,
    live: Live,
}
pub(super) struct Run<'a> {
    state: State<'a>,
    tokens: Vec<u32>,
    sampler: Sampler,
    maximum: usize,
    conversation: Option<super::super::checkpoint_ladder::Conversation>,
    matched: usize,
    reuse_source: &'static str,
    restore_ms: f64,
    tokenize_ms: f64,
    departing_ms: f64,
    required: usize,
    transcript_at: Option<usize>,
    transcript_split: Option<usize>,
    transcript_entry: Option<qwen_llm::snapshot_policy::EntryId>,
    endpoint: bool,
    ends: Vec<usize>,
    commands: VecDeque<std::ops::Range<usize>>,
    segment: usize,
    rows: usize,
    position: usize,
    restored_logits: Option<Vec<f32>>,
    prefill_ms: f64,
    capture_ms: f64,
    prefill_calls: usize,
    reserve: u64,
    ready: bool,
}

pub(super) struct Owner<'a> {
    ctx: &'a MetalContext,
    tokenizer: &'a Tokenizer,
    factory: Qwen4ExpSessionFactory<'a, 'static>,
    cache: &'a mut SnapshotCache<Checkpoint>,
    residency: &'a mut super::super::idle_residency::IdleResidency,
    idle: VecDeque<Idle<'a>>,
    driver: Option<Driver<Run<'a>>>,
    config: Config,
    model_id: &'a str,
    default_max_tokens: usize,
    max_piece_bytes: usize,
    sampling: crate::release_sampling::SamplingDefaults,
    style: super::super::items::TemplateStyle,
    turns: TurnConfig,
    forward_limit: usize,
    vocab: u32,
    stops: &'a [i32],
    im_start: Option<u32>,
    control_reserve: u64,
    head_reserve: u64,
    native: Option<Qwen4ExpDecodeBatch>,
}
impl<'a> Owner<'a> {
    pub fn new(base: &'a mut FlashNextBackend) -> anyhow::Result<Self> {
        let (factory, runner) = base.loaded.session_factory(&base.ctx)?;
        let mut native = if native_enabled() {
            match Qwen4ExpDecodeBatch::new(&runner, base.batching.active) {
                Ok(batch) => Some(batch),
                Err(e) => {
                    tracing::info!(target:"qwen_diag","serve native batch declined: family=qwen4exp reason={e}");
                    None
                }
            }
        } else {
            None
        };
        let mut idle = VecDeque::from([Idle {
            state: State {
                runner,
                failed: false,
            },
            live: base.live.take().unwrap_or(Live {
                tokens: Vec::new(),
                conversation: None,
            }),
        }]);
        // Warm the actual independent path before any client can measure a
        // first-use pipeline compile. The legacy startup runner is empty here.
        if native.is_some()
            && idle[0].state.runner.next_position() == 0
            && idle[0].live.tokens.is_empty()
        {
            match factory.create_runner(&base.ctx) {
                Ok(mut second) => {
                    let first = &mut idle[0].state.runner;
                    let result = native
                        .as_mut()
                        .unwrap()
                        .step(&mut [&mut *first, &mut second], &[0, 0]);
                    first.reset()?;
                    second.reset()?;
                    if let Err(e) = result {
                        tracing::info!(target:"qwen_diag","serve native warmup declined: family=qwen4exp reason={e}");
                        native = None;
                    }
                    if base.batching.live > 1 {
                        idle.push_back(Idle {
                            state: State {
                                runner: second,
                                failed: false,
                            },
                            live: Live {
                                tokens: Vec::new(),
                                conversation: None,
                            },
                        });
                    }
                }
                Err(e) => {
                    tracing::info!(target:"qwen_diag","serve native warmup admission declined: family=qwen4exp reason={e}");
                    native = None;
                }
            }
        }
        tracing::info!(target:"qwen_diag", "serve batch: family=qwen4exp max_active={} live_states={} native={} prefills=normal",base.batching.active,base.batching.live,native.is_some());
        Ok(Self {
            native,
            ctx: &base.ctx,
            tokenizer: &base.tokenizer,
            factory,
            cache: &mut base.cache,
            residency: &mut base.idle_residency,
            idle,
            driver: Some(
                Driver::new(base.batching.active)
                    .with_prefill_policy(true, base.batching.decode_percent),
            ),
            config: base.batching,
            model_id: &base.model_id,
            default_max_tokens: base.default_max_tokens,
            max_piece_bytes: base.max_piece_bytes,
            sampling: base.release_sampling,
            style: base.template_style,
            turns: base.turns,
            forward_limit: base.forward_limit,
            vocab: base.vocab_size,
            stops: &base.stop_tokens,
            im_start: base.im_start,
            control_reserve: base.control_cpu_reserve,
            head_reserve: base.vocab_size as u64 * 4 * base.batching.active as u64 * 2
                + base.forward_limit as u64 * 64 * base.batching.active as u64,
        })
    }
    fn limits(&self) -> super::super::request_profile::OutputLimits {
        super::super::request_profile::OutputLimits {
            default_max_tokens: self.default_max_tokens,
            max_piece_bytes: self.max_piece_bytes,
        }
    }
    fn allocate_state(&mut self, reserve: u64) -> Result<State<'a>, BackendFailure> {
        loop {
            match self.factory.create_runner_with_reserve(self.ctx, reserve) {
                Ok(runner) => {
                    return Ok(State {
                        runner,
                        failed: false,
                    });
                }
                Err(error) => {
                    let denied = match &error {
                        Qwen4ExpRuntimeError::MemoryAdmission(d)
                        | Qwen4ExpRuntimeError::Session(qwen_llm::qwen4exp_text_session::Qwen4ExpTextSessionError::MemoryAdmission(d)) => Some(d),
                        _ => None,
                    };
                    if let Some(d) = denied {
                        if d.reason().is_pressure() {
                            if !self.cache.evict_for(u64::MAX).is_empty() {
                                continue;
                            }
                            if self.idle.pop_front().is_some() {
                                continue;
                            }
                        }
                        return Err(super::super::transport_memory::memory_refusal(
                            "Flash-Next live state",
                            d,
                        )
                        .into());
                    }
                    return Err(ServeError::server_error(format!(
                        "allocate {FAMILY} live state: {error}"
                    ))
                    .into());
                }
            }
        }
    }
    fn admit_transport(&mut self, reserve: u64) -> Result<(), BackendFailure> {
        loop {
            match super::super::transport_memory::admit_resident_transport(
                reserve,
                qwen_llm::metal::effective_host_memory_remaining_bytes(),
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if e.status == 503 {
                        if !self.cache.evict_for(u64::MAX).is_empty() {
                            continue;
                        }
                        if self.idle.pop_front().is_some() {
                            continue;
                        }
                    }
                    return Err(e.into());
                }
            }
        }
    }
    fn begin_run(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<Run<'a>, BackendFailure> {
        sink.tick().map_err(BackendFailure::Aborted)?;
        let maximum = self.limits().max_tokens(request);
        let sampler = Sampler::new(super::super::request_profile::sampling_with_defaults(
            request,
            self.sampling,
        ))
        .map_err(|e| ServeError::invalid_request(None, format!("sampling: {e}")))?;
        let t = Instant::now();
        let tokens =
            decode_loop::encode_checked(self.tokenizer, prompt, false, self.vocab, FAMILY)?;
        let tokenize_ms = t.elapsed().as_secs_f64() * 1e3;
        let required =
            decode_loop::required_forwards(FAMILY, tokens.len(), maximum, self.forward_limit)?;
        let reserve = sink.transport_reserve_bytes();
        let hit = checkpoint_candidate(
            self.cache,
            &tokens,
            self.turns.live || self.turns.endpoint_checkpoint,
            |c| c.logits.len() == self.vocab as usize && c.logits.iter().all(|v| v.is_finite()),
        );
        let pinned = hit.as_ref().map(|h| h.id).filter(|&id| self.cache.pin(id));
        if let Err(e) = self.admit_transport(reserve) {
            if let Some(id) = pinned {
                self.cache.unpin(id);
            }
            return Err(e);
        }
        let best = self
            .idle
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                live_prefix_len(
                    &s.live.tokens,
                    &tokens,
                    s.state.runner.next_position(),
                    !s.state.failed
                        && !s.state.runner.is_poisoned()
                        && s.state.runner.logits().is_ok(),
                )
                .filter(|_| self.turns.live)
                .map(|n| (i, n))
            })
            .max_by_key(|(i, n)| (*n, std::cmp::Reverse(*i)))
            .map(|(i, _)| i);
        let empty = self.idle.iter().position(|s| {
            s.live.tokens.is_empty()
                && s.state.runner.next_position() == 0
                && !s.state.failed
                && !s.state.runner.is_poisoned()
        });
        let candidate = match idle_choice(best, empty, self.idle.len(), self.config.live) {
            Some(i) => self.idle.remove(i).expect("idle index"),
            None => Idle {
                state: match self.allocate_state(reserve) {
                    Ok(state) => state,
                    Err(e) => {
                        if let Some(id) = pinned {
                            self.cache.unpin(id);
                        }
                        return Err(e);
                    }
                },
                live: Live {
                    tokens: Vec::new(),
                    conversation: None,
                },
            },
        };
        let mut idle = candidate;
        let runner = &mut idle.state.runner;
        let restore_t = Instant::now();
        let live = live_prefix_len(
            &idle.live.tokens,
            &tokens,
            runner.next_position(),
            !idle.state.failed && !runner.is_poisoned() && runner.logits().is_ok(),
        )
        .filter(|_| self.turns.live);
        let use_live = live.is_some_and(|n| hit.as_ref().is_none_or(|h| n >= h.prefix_len));
        let mut restored_logits = None;
        let mut departing_ms = 0.;
        let (matched, reuse_source) = if use_live {
            if let Some(id) = pinned {
                self.cache.unpin(id);
            }
            (live.unwrap(), "live_session")
        } else {
            if self.turns.live
                && !idle.live.tokens.is_empty()
                && idle.live.tokens.len() == runner.next_position()
                && !runner.is_poisoned()
            {
                if let Ok(logits) = runner.logits() {
                    let started = Instant::now();
                    let head = logits.to_vec();
                    capture_with_logits(
                        self.cache,
                        self.ctx,
                        runner,
                        &idle.live.tokens,
                        "departing_live",
                        reserve,
                        &head,
                    );
                    self.cache
                        .record_tip(idle.live.conversation, &(), &idle.live.tokens);
                    departing_ms = started.elapsed().as_secs_f64() * 1e3;
                }
            }
            if let Some(id) = pinned {
                self.cache.unpin(id);
            }
            let n = restore_prefix_hit(
                self.cache,
                hit,
                runner,
                |r, c: &Checkpoint| {
                    if c.logits.is_empty() {
                        r.restore_snapshot(&c.snapshot)?;
                    } else {
                        r.restore_snapshot_with_logits(&c.snapshot, &c.logits)?;
                    }
                    restored_logits = (!c.logits.is_empty()).then(|| c.logits.clone());
                    Ok(r.next_position())
                },
                |r| r.reset(),
            )
            .map_err(|e| ServeError::server_error(format!("restore {FAMILY}: {e}")))?;
            if n == 0 {
                runner
                    .reset()
                    .map_err(|e| ServeError::server_error(e.to_string()))?;
            }
            (n, if n == 0 { "cold" } else { "snapshot_ram" })
        };
        let restore_ms = restore_t.elapsed().as_secs_f64() * 1e3;
        let transcript_at = self
            .im_start
            .and_then(|id| transcript_boundary(&tokens, id));
        let split = transcript_split(transcript_at, matched);
        let endpoint = endpoint_checkpoint(
            self.turns.endpoint_checkpoint,
            matched,
            tokens.len(),
            transcript_at,
            runner.packed_prefill_capacity(),
        );
        let transcript_entry = transcript_at
            .filter(|&n| n == matched)
            .and_then(|n| self.cache.entry_for(&tokens[..n]));
        let mut ends = Vec::new();
        if let Some(n) = split {
            ends.push(n);
        }
        if matched < tokens.len() {
            ends.push(tokens.len());
        }
        ends.dedup();
        let mut commands = VecDeque::new();
        let mut start = matched;
        for &end in &ends {
            commands.extend(
                runner
                    .prefill_command_ranges(start..end)
                    .map_err(|e| ServeError::server_error(e.to_string()))?,
            );
            start = end;
        }
        let rows = runner.packed_prefill_capacity().unwrap_or(1).max(1);
        Ok(Run {
            state: idle.state,
            tokens,
            sampler,
            maximum,
            conversation: super::super::checkpoint_ladder::conversation(request),
            matched,
            reuse_source,
            restore_ms,
            tokenize_ms,
            departing_ms,
            required,
            transcript_at,
            transcript_split: split,
            transcript_entry,
            endpoint,
            ends,
            commands,
            segment: 0,
            rows,
            position: matched,
            restored_logits,
            prefill_ms: 0.,
            capture_ms: 0.,
            prefill_calls: 0,
            reserve,
            ready: false,
        })
    }
    fn prefill_run(
        &mut self,
        run: &mut Run<'a>,
        rows: usize,
        sink: &mut dyn GenerationSink,
    ) -> Result<bool, BackendFailure> {
        sink.tick().map_err(BackendFailure::Aborted)?;
        if let Some(&end) = run.ends.get(run.segment) {
            let command = run
                .commands
                .pop_front()
                .ok_or_else(|| ServeError::server_error("missing native prefill command"))?;
            if command.start != run.position || command.end > end || command.len() > rows {
                return Err(ServeError::server_error(
                    "owner quantum differs from the native prefill plan",
                )
                .into());
            }
            let stop = command.end;
            let mut abort = None;
            let mut captured = 0.;
            let started = Instant::now();
            let result = run
                .state
                .runner
                .prefill_continuation_with_observer(
                    &run.tokens[run.position..stop],
                    || {
                        sink.tick().map_err(|e| {
                            abort = Some(e);
                            Qwen4ExpRuntimeError::Checkpoint("concurrent transport aborted".into())
                        })
                    },
                    &mut |runner, chunk| {
                        if self.cache.ladder_enabled()
                            && run.conversation.is_some()
                            && chunk.len() == run.rows
                            && chunk.end < end
                            && end - chunk.end <= run.rows
                        {
                            let t = Instant::now();
                            capture_snapshot_impl(
                                self.cache,
                                self.ctx,
                                runner,
                                &run.tokens[..chunk.end],
                                "ladder",
                                run.reserve,
                                run.conversation,
                                &[],
                            );
                            captured += t.elapsed().as_secs_f64() * 1e3;
                        }
                    },
                )
                .map(|_| ());
            map_prefill_result(result, abort)?;
            run.prefill_ms += started.elapsed().as_secs_f64() * 1e3 - captured;
            run.capture_ms += captured;
            run.prefill_calls += 1;
            run.position = run.state.runner.next_position();
            if run.position != stop {
                return Err(ServeError::server_error("concurrent prefill extent differs").into());
            }
            if stop == end {
                if run.transcript_split == Some(end) && !run.endpoint {
                    let t = Instant::now();
                    if capture_snapshot(
                        self.cache,
                        self.ctx,
                        &run.state.runner,
                        &run.tokens[..end],
                        "transcript",
                        run.reserve,
                    ) {
                        run.transcript_entry = self.cache.entry_for(&run.tokens[..end]);
                    }
                    run.capture_ms += t.elapsed().as_secs_f64() * 1e3;
                }
                run.segment += 1;
            }
        }
        if run.segment < run.ends.len() {
            return Ok(false);
        }
        if !run.ready {
            if run.transcript_entry.is_none() {
                let logits = if run.endpoint {
                    run.state
                        .runner
                        .logits()
                        .map_err(|e| ServeError::server_error(e.to_string()))?
                        .to_vec()
                } else {
                    Vec::new()
                };
                let t = Instant::now();
                capture_with_logits(
                    self.cache,
                    self.ctx,
                    &run.state.runner,
                    &run.tokens,
                    "prompt",
                    run.reserve,
                    &logits,
                );
                run.capture_ms += t.elapsed().as_secs_f64() * 1e3;
            }
            run.ready = true;
        }
        Ok(true)
    }
    fn cursor(&self, run: &mut Run<'a>) -> Result<DecodeCursor, BackendFailure> {
        let logits = match run
            .restored_logits
            .take()
            .filter(|_| run.matched == run.tokens.len())
        {
            Some(v) => v,
            None => run
                .state
                .runner
                .logits()
                .map_err(|e| ServeError::server_error(e.to_string()))?
                .to_vec(),
        };
        DecodeCursor::new(decode_loop::DecodeRequest {
            family: FAMILY,
            logits,
            max_tokens: run.maximum,
            stop_tokens: self.stops,
            vocab_size: self.vocab,
        })
    }
    fn finish_run(
        &mut self,
        mut run: Run<'a>,
        generation: &crate::GenerationResult,
        mut batch: serde_json::Value,
    ) -> Result<GenerationOutcome, BackendFailure> {
        batch["sampler_version"] = serde_json::json!(run.sampler.algorithm_version());
        crate::derive_completed_checkpoint_boundary(
            run.tokens.len(),
            &generation.tokens,
            generation.transitions,
            run.state.runner.next_position(),
        )
        .map_err(|e| ServeError::server_error(e.to_string()))?;
        let mut consumed = run.tokens.clone();
        for &token in &generation.tokens[..generation.transitions] {
            consumed.push(
                crate::checked_token_id(token, self.vocab, "consumed")
                    .map_err(|e| ServeError::server_error(e.to_string()))?,
            );
        }
        let retain = self.turns.live
            && !run.state.failed
            && !run.state.runner.is_poisoned()
            && run.state.runner.logits().is_ok();
        if !retain {
            capture_snapshot(
                self.cache,
                self.ctx,
                &run.state.runner,
                &consumed,
                "completed",
                run.reserve,
            );
        }
        if let Some(cut) = run.transcript_at.filter(|_| run.transcript_entry.is_some()) {
            self.cache
                .record_tip(run.conversation, &(), &run.tokens[..cut]);
        } else {
            self.cache.record_tip(run.conversation, &(), &run.tokens);
        }
        self.cache.record_tip(run.conversation, &(), &consumed);
        let phases = serde_json::json!({"prefill_ms":run.prefill_ms,"decode_ms":generation.wall_ms,"reused_tokens":run.matched,"prefill_tokens":run.tokens.len()-run.matched,"restore_ms":run.restore_ms,"reuse_source":run.reuse_source,"batch":batch,"decode_impl":if self.native.is_some(){"native_hc_dense_head"}else{"ordinary_members"}});
        tracing::info!(target:"qwen_diag", "serve phases: family=qwen4exp tokenize_ms={:.1} restore_ms={:.1} reuse_source={} departing_capture_ms={:.1} endpoint_checkpoint={} prefill_calls={} matched_tokens={} prefill_ms={:.1} prefill_tokens={} snapshot_capture_ms={:.1} decode_ms={:.1} required_forwards={} forward_limit={} transitions={} batch={}",run.tokenize_ms,run.restore_ms,run.reuse_source,run.departing_ms,run.endpoint,run.prefill_calls,run.matched,run.prefill_ms,run.tokens.len()-run.matched,run.capture_ms,generation.wall_ms,run.required,self.forward_limit,generation.transitions,phases["batch"]);
        let mut out = super::super::outcome::finish_generation(
            run.tokens.len(),
            generation,
            run.matched,
            run.restore_ms,
        );
        if let Some(s) = &mut out.stats {
            s.phases = Some(phases);
        }
        if !run.state.failed && !run.state.runner.is_poisoned() {
            let live = if retain {
                Live {
                    tokens: consumed,
                    conversation: run.conversation,
                }
            } else {
                run.state
                    .runner
                    .reset()
                    .map_err(|e| ServeError::server_error(e.to_string()))?;
                Live {
                    tokens: Vec::new(),
                    conversation: None,
                }
            };
            self.idle.push_back(Idle {
                state: run.state,
                live,
            });
            while self.idle.len() > self.config.live {
                if let Some(old) = self.idle.pop_front() {
                    self.publish_idle(&old, run.reserve);
                }
            }
        }
        Ok(out)
    }
    fn publish_idle(&mut self, old: &Idle<'a>, reserve: u64) {
        if self.turns.live
            && !old.state.failed
            && !old.state.runner.is_poisoned()
            && !old.live.tokens.is_empty()
            && old.live.tokens.len() == old.state.runner.next_position()
            && let Ok(logits) = old.state.runner.logits()
        {
            capture_with_logits(
                self.cache,
                self.ctx,
                &old.state.runner,
                &old.live.tokens,
                "departing_idle",
                reserve,
                &logits.to_vec(),
            );
            self.cache
                .record_tip(old.live.conversation, &(), &old.live.tokens);
        }
    }
    fn track_failure<T>(&mut self, result: Result<T, BackendFailure>) -> Result<T, BackendFailure> {
        if matches!(&result,Err(BackendFailure::Serve(e)) if e.status>=500) {
            self.residency.request_failed_on_server();
        }
        result
    }
    fn ordinary_step(&mut self, state: &mut State<'a>, token: u32) -> Result<Vec<f32>, String> {
        crate::shutdown::checkpoint().map_err(|e| e.to_string())?;
        let v = state
            .runner
            .forward_token(token)
            .map_err(|e| e.to_string())?
            .to_vec();
        if v.len() != self.vocab as usize || v.iter().any(|x| !x.is_finite()) {
            state.failed = true;
            return Err("malformed Flash-Next logits".into());
        }
        Ok(v)
    }
    fn decode_group(
        &mut self,
        states: &mut [&mut State<'a>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let Some(native) = self.native.as_mut() else {
            return self.ordinary_group(states, tokens);
        };
        crate::shutdown::checkpoint().map_err(|e| e.to_string())?;
        if states.is_empty()
            || states.len() > self.config.active
            || states.len() != tokens.len()
            || states.iter().zip(tokens).any(|(s, &t)| {
                s.failed
                    || s.runner.is_poisoned()
                    || s.runner.next_position() >= self.forward_limit
                    || t >= self.vocab
            })
        {
            return Err("invalid native Flash-Next group".into());
        }
        for s in states.iter_mut() {
            s.failed = true;
        }
        let mut runners: Vec<_> = states.iter_mut().map(|s| &mut s.runner).collect();
        let result = native.step(&mut runners, tokens).map_err(|e| e.to_string());
        drop(runners);
        if let Ok(v) = &result
            && (v.len() != states.len() || v.iter().any(|h| h.len() != self.vocab as usize))
        {
            return Err("malformed native Flash-Next heads".into());
        }
        if result.is_ok() {
            for s in states {
                s.failed = false;
            }
        }
        result
    }
    /// Replaceable native decode seam. An error after the first member advances
    /// marks the entire group failed, preventing the driver's safe-refusal replay.
    fn ordinary_group(
        &mut self,
        states: &mut [&mut State<'a>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let limit = self.forward_limit;
        let vocab = self.vocab;
        step_members(
            states,
            tokens,
            |s, t| {
                !s.failed
                    && !s.runner.is_poisoned()
                    && s.runner.next_position() < limit
                    && t < vocab
            },
            |s, failed| s.failed = failed,
            |s, t| self.ordinary_step(s, t),
        )
    }
}

fn idle_choice(
    best: Option<usize>,
    empty: Option<usize>,
    len: usize,
    limit: usize,
) -> Option<usize> {
    best.or(empty).or_else(|| (len >= limit).then_some(0))
}

/// Mark the whole group before any member advances. Safe refusal remains
/// replayable; begun errors and unwinds cannot repeat already consumed tokens.
fn step_members<S>(
    states: &mut [&mut S],
    tokens: &[u32],
    valid: impl Fn(&S, u32) -> bool,
    mut mark: impl FnMut(&mut S, bool),
    mut forward: impl FnMut(&mut S, u32) -> Result<Vec<f32>, String>,
) -> Result<Vec<Vec<f32>>, String> {
    if states.is_empty() || states.len() > 4 || states.len() != tokens.len() {
        return Err("invalid Flash-Next decode group".into());
    }
    if states.iter().zip(tokens).any(|(s, &t)| !valid(s, t)) {
        return Err("invalid Flash-Next decode member".into());
    }
    for s in states.iter_mut() {
        mark(s, true);
    }
    let mut out = Vec::with_capacity(states.len());
    for (s, &t) in states.iter_mut().zip(tokens) {
        out.push(forward(s, t)?);
    }
    for s in states.iter_mut() {
        mark(s, false);
    }
    Ok(out)
}

impl<'a> Runtime for Owner<'a> {
    type Run = Run<'a>;
    type State = State<'a>;
    fn extra_reserve(&self) -> u64 {
        self.control_reserve.saturating_add(self.head_reserve)
    }
    fn begin(
        &mut self,
        p: &PreparedResponse,
        sink: &mut dyn GenerationSink,
    ) -> Result<Begun<Self::Run>, BackendFailure> {
        let result = self
            .begin_run(p.request(), &p.prompt, sink)
            .map(|r| Begun::Run(Box::new(r)));
        self.track_failure(result)
    }
    fn rows(&self, run: &Self::Run) -> usize {
        run.rows
    }
    fn prefill(
        &mut self,
        run: &mut Self::Run,
        rows: usize,
        sink: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Result<Option<DecodeCursor>, BackendFailure> {
        let result = (|| {
            run.reserve = reserve;
            if !self.prefill_run(run, rows, sink)? {
                return Ok(None);
            }
            self.cursor(run).map(Some)
        })();
        self.track_failure(result)
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
            self.tokenizer,
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
        if !run.state.runner.gpu_token_loop_enabled()
            || (!run.state.runner.gpu_token_loop_warm() && cursor.remaining_tokens() < 128)
        {
            return None;
        }
        run.reserve = reserve;
        // A newly arriving peer waits at most one bounded burst. Existing
        // peers/prefills use native rounds, and backpressure yields at a token.
        let count = cursor.remaining_tokens().min(16);
        let offset = cursor.elapsed_ms();
        let draws = run.sampler.draws();
        let mut tokens = Vec::with_capacity(count);
        let mut callback_error = None;
        let result =
            run.state
                .runner
                .generate_gpu_tokens(&mut run.sampler, count, self.stops, |token| {
                    tokens.push(token);
                    let result =
                        (|| -> Result<bool, BackendFailure> {
                            crate::shutdown::checkpoint()
                                .map_err(|e| ServeError::server_error(e.to_string()))?;
                            let bytes =
                                self.tokenizer.try_decode_piece_bytes_exact(token).map_err(
                                    |e| ServeError::server_error(format!("decode {FAMILY}: {e}")),
                                )?;
                            sink.piece(bytes)
                                .and_then(|()| sink.tick())
                                .map_err(BackendFailure::Aborted)?;
                            let ready = sink.flush_ready().map_err(BackendFailure::Aborted)?;
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
                });
        if let Some(e) = callback_error {
            return Some(Err(e));
        }
        match result {
            Err(_)
                if !run.state.runner.is_poisoned()
                    && tokens.is_empty()
                    && run.sampler.draws() == draws =>
            {
                None
            }
            Err(e) => {
                run.state.failed = true;
                Some(self.track_failure(Err(
                    ServeError::server_error(format!("{FAMILY} GPU feedback: {e}")).into(),
                )))
            }
            Ok(stats) => {
                if stats.stopped_on_eos {
                    tokens.push(stats.terminal_token.expect("GPU EOS token"));
                }
                Some(Ok(cursor.append_gpu_burst(&tokens, stats, offset)))
            }
        }
    }
    fn state(run: &mut Self::Run) -> &mut Self::State {
        &mut run.state
    }
    fn can_step(&self, s: &Self::State) -> bool {
        !s.failed && !s.runner.is_poisoned() && s.runner.next_position() < self.forward_limit
    }
    fn poisoned(s: &Self::State) -> bool {
        s.failed || s.runner.is_poisoned()
    }
    fn step(&mut self, s: &mut Self::State, token: u32) -> Result<Vec<f32>, String> {
        self.ordinary_step(s, token)
    }
    fn step_batch(
        &mut self,
        s: &mut [&mut Self::State],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.decode_group(s, tokens)
    }
    fn finish(
        &mut self,
        run: Self::Run,
        generation: &crate::GenerationResult,
        batch: serde_json::Value,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let result = self.finish_run(run, generation, batch);
        self.track_failure(result)
    }
}
impl GenerationBackend for Owner<'_> {
    fn model_id(&self) -> &str {
        self.model_id
    }
    fn set_control_memory_reserve(&mut self, n: u64) {
        self.control_reserve = n;
    }
    fn request_profile(&self) -> super::super::request_profile::RequestProfile {
        super::super::request_profile::RequestProfile::FlashNext {
            sampling: self.sampling,
            style: self.style,
            limits: self.limits(),
        }
    }
    fn max_active(&self) -> usize {
        self.config.active
    }
    fn submit(&mut self, job: batching::Job) {
        self.driver.as_mut().expect("driver").submit(job);
    }
    fn pump(&mut self) -> bool {
        let mut d = self.driver.take().expect("driver");
        // Attribute encoder activity to one synchronous owner quantum. HTTP
        // connection completions overlap and cannot own this observation.
        self.residency.before_request();
        let progress = d.pump(self);
        if d.take_native_failure() {
            self.residency.request_failed_on_server();
        }
        self.residency.request_finished();
        self.driver = Some(d);
        progress
    }
    fn abort_all(&mut self) {
        if let Some(d) = &mut self.driver {
            d.abort_all();
        }
    }
    fn idle(&mut self) {
        super::super::log_expired_snapshots("qwen4exp", &self.cache.sweep());
        self.residency
            .on_idle(self.ctx, || self.factory.retained_buffers());
    }
    fn request_finished(&mut self) {
        self.residency.request_finished();
    }
    fn request_failed_on_server(&mut self) {
        self.residency.request_failed_on_server();
    }
    fn shutdown(&mut self) {
        self.abort_all();
        self.residency.shutdown();
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.residency.before_request();
        let mut run = self.begin_run(request, prompt, sink)?;
        while !self.prefill_run(&mut run, usize::MAX, sink)? {}
        let mut cursor = self.cursor(&mut run)?;
        loop {
            match cursor.advance(
                &mut run.sampler,
                None,
                self.tokenizer,
                sink,
                crate::shutdown::checkpoint,
            )? {
                Advance::Done(g) => {
                    return self.finish_run(run, &g, serde_json::json!({"max_active":1}));
                }
                Advance::Forward(t) => {
                    let started = Instant::now();
                    let result = self.ordinary_step(&mut run.state, t);
                    let elapsed = started.elapsed().as_secs_f64() * 1e3;
                    cursor.on_logits(
                        result.map_err(anyhow::Error::msg),
                        elapsed,
                        crate::shutdown::checkpoint,
                    )?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits_default_to_four_and_allow_a_single_owner_rollback() {
        assert_eq!(bounded(MAX_ACTIVE, None, 4, 4).unwrap(), 4);
        assert_eq!(bounded(MAX_ACTIVE, Some("1"), 4, 4).unwrap(), 1);
        for value in ["0", "5", "-1", "x", ""] {
            assert!(bounded(MAX_ACTIVE, Some(value), 4, 4).is_err());
        }
        assert_eq!(bounded(DECODE_PERCENT, None, 10, 90).unwrap(), 10);
        assert!(bounded(DECODE_PERCENT, Some("90"), 10, 90).is_ok());
        assert!(bounded(DECODE_PERCENT, Some("91"), 10, 90).is_err());
    }
    #[test]
    fn ordinary_group_does_not_replay_partial_progress() {
        #[derive(Default)]
        struct Fake {
            position: usize,
            failed: bool,
        }
        let (mut a, mut b) = (Fake::default(), Fake::default());
        let result = step_members(
            &mut [&mut a, &mut b],
            &[7, 8],
            |_, _| true,
            |s, f| s.failed = f,
            |s, t| {
                if t == 8 {
                    return Err("injected failure".into());
                }
                s.position += 1;
                Ok(vec![t as f32])
            },
        );
        assert!(result.is_err());
        assert_eq!(a.position, 1);
        assert_eq!(b.position, 0);
        assert!(a.failed && b.failed);
        let mut calls = 0;
        let result = step_members(
            &mut [&mut a, &mut b],
            &[7, 8],
            |s, _| !s.failed,
            |s, f| s.failed = f,
            |_, _| {
                calls += 1;
                Ok(vec![])
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, 0);
        let (mut a, mut b) = (Fake::default(), Fake::default());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = step_members(
                &mut [&mut a, &mut b],
                &[7, 8],
                |_, _| true,
                |s, f| s.failed = f,
                |s, t| {
                    if t == 8 {
                        panic!("injected owner unwind");
                    }
                    s.position += 1;
                    Ok(vec![])
                },
            );
        }));
        assert!(panic.is_err());
        assert_eq!((a.position, b.position), (1, 0));
        assert!(a.failed && b.failed);
    }
    #[test]
    fn ordinary_group_keeps_member_order_and_refuses_before_mutation() {
        let (mut a, mut b) = ((0usize, false), (0usize, false));
        let out = step_members(
            &mut [&mut a, &mut b],
            &[7, 9],
            |_, t| t < 10,
            |s, f| s.1 = f,
            |s, t| {
                s.0 += 1;
                Ok(vec![t as f32])
            },
        )
        .unwrap();
        assert_eq!(out, vec![vec![7.], vec![9.]]);
        assert_eq!(a, (1, false));
        assert_eq!(b, (1, false));
        assert!(
            step_members(
                &mut [&mut a, &mut b],
                &[7, 10],
                |_, t| t < 10,
                |s, f| s.1 = f,
                |s, _| {
                    s.0 += 1;
                    Ok(vec![])
                }
            )
            .is_err()
        );
        assert_eq!(a, (1, false));
        assert_eq!(b, (1, false));
    }
    #[test]
    fn idle_choice_retains_unrelated_warm_states_until_the_limit() {
        assert_eq!(idle_choice(None, Some(0), 1, 4), Some(0));
        assert_eq!(idle_choice(None, None, 1, 4), None);
        assert_eq!(idle_choice(None, None, 3, 4), None);
        assert_eq!(idle_choice(Some(2), None, 4, 4), Some(2));
        assert_eq!(idle_choice(None, None, 4, 4), Some(0));
    }
}
