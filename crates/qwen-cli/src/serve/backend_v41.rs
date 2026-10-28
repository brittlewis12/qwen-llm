//! Exact-prefix GGUF HTTP backend. Forward borrows the caller's resident
//! weights/context; neither they nor the backend move to an HTTP worker.
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
#[cfg(test)]
use super::output_partition::GenerationEnd;
use super::request_profile::{OutputLimits, RequestProfile, sampling_with_defaults};
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::{Forward, RequestMemoryEstimate, StateError};
use qwen_llm::metal::{HostMemoryHeadroom, MetalContext, MetalMemorySignals};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qwen_llm::tokenizer::NativeTokenizer;

/// Concurrent V4.1 requests `qwen serve` keeps in flight on the model owner
/// (default `DEFAULT_MAX_ACTIVE`; 1 is one request at a time, exactly the
/// serial path). Above 1, every decoding request advances one token per
/// `Forward::step_batch` step (plain `step` when only one decodes), and a new
/// request's prefill interleaves with their decode in 512-row calls (full
/// chunks when nobody else decodes), which changes that prompt's prefill
/// partition and so F32 rounding. The acceptor's `QWEN_SERVE_QUEUE_DEPTH`
/// still bounds active plus waiting connections. Positive integer; read at
/// startup.
const MAX_ACTIVE_ENV: &str = "QWEN_SERVE_V41_MAX_ACTIVE";
/// Idle V4.1 live sessions `qwen serve` keeps for exact-prefix reuse (default
/// `DEFAULT_LIVE_STATES`; 1 is the single live session). More than one keeps
/// other conversations' sessions resident; the least recently used is released when the count or
/// `QWEN_SERVE_V41_LIVE_STATE_MIB` budget is exceeded and, under memory
/// pressure, before a request is refused. Positive integer; read at startup.
const LIVE_STATES_ENV: &str = "QWEN_SERVE_V41_LIVE_STATES";
/// Optional MiB budget for idle V4.1 live sessions, each measured by the
/// engine's request-memory estimate at its capacity. Unset: count only.
/// Positive integer; read at startup.
const LIVE_STATE_MIB_ENV: &str = "QWEN_SERVE_V41_LIVE_STATE_MIB";

/// Agent workloads keep about three conversations busy at once. M5 Ultra, GSQ
/// IQ2_XS, three agents x three turns (greedy, 128 tokens per turn, fresh
/// server): 26.9 s with one request in flight and one live session, 21.6 s
/// with four of each (1.25x); three concurrent single requests 1.18x.
const DEFAULT_MAX_ACTIVE: usize = 4;
const DEFAULT_LIVE_STATES: usize = 4;

fn positive_env(name: &str) -> Result<Option<usize>> {
    let value = match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => anyhow::bail!("read {name}: {error}"),
        Ok(value) => value,
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n >= 1)
        .map(Some)
        .with_context(|| format!("{name} must be a positive integer, got {value:?}"))
}

pub(super) struct Prepared {
    settings: Settings,
    snapshot_mib: Option<u64>,
    snapshot_policy: qwen_llm::snapshot_policy::SnapshotPolicyConfig,
    live_states: usize,
    live_state_bytes: Option<u64>,
    max_active: usize,
}
impl Prepared {
    pub(super) fn new(
        gguf: &qwen_llm::gguf::GgufFile,
        invocation: &crate::cli::ServeInvocation,
    ) -> Result<Self> {
        let plan = qwen_llm::deepseek_v41::gguf::plan(gguf)?;
        ensure!(
            invocation.lens_data_dir.is_none() && invocation.web_root.is_none(),
            "DeepSeek V4.1 has no Lens backend"
        );
        let limits = super::resolve_serve_limits(
            super::profile(qwen_llm::model_family::ModelFamily::DeepSeekV41),
            Some(usize::try_from(plan.engram_hash.max_position)?),
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?
        .context("V4.1 requires fixed request limits")?;
        let tokenizer = NativeTokenizer::from_gguf(gguf)?;
        ensure!(
            tokenizer.n_vocab() as usize == plan.config.vocab,
            "V4.1 tokenizer/model vocabulary mismatch"
        );
        let eos = tokenizer.eos().context("V4.1 GGUF has no EOS")? as u32;
        let live_state_bytes = positive_env(LIVE_STATE_MIB_ENV)?
            .map(|mib| (mib as u64).checked_mul(1 << 20))
            .map(|bytes| bytes.context("V4.1 live-state budget overflows"))
            .transpose()?;
        Ok(Self {
            live_states: positive_env(LIVE_STATES_ENV)?.unwrap_or(DEFAULT_LIVE_STATES),
            max_active: positive_env(MAX_ACTIVE_ENV)?.unwrap_or(DEFAULT_MAX_ACTIVE),
            live_state_bytes,
            snapshot_mib: invocation.snapshot_cache_mib,
            snapshot_policy: invocation.snapshot_policy,
            settings: Settings::new(
                tokenizer,
                limits.context_tokens,
                limits.max_tokens,
                crate::deepseek_v41::default_sampling(crate::release_sampling::fresh_seed()),
                eos,
                String::new(),
            )?,
        })
    }
}

pub(super) fn start(
    prepared: Prepared,
    gguf: qwen_llm::gguf::GgufFile,
    listening: super::Listening,
) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let started = Instant::now();
    let weights = crate::deepseek_v41::load(&ctx, std::sync::Arc::new(gguf))?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let plan = super::SnapshotCachePlan::resolve(
        prepared.snapshot_mib,
        prepared.snapshot_policy,
        ctx.memory_signals(),
    )?;
    tracing::info!("serve: V4.1 snapshot cache: {plan}");
    if prepared.max_active > 1 {
        warm_batched_decode(&forward, prepared.max_active, prepared.settings.eos)?;
    }
    let mut settings = prepared.settings;
    settings.model_id = listening.model_id.clone();
    let mut backend = Backend::new(&ctx, forward, settings);
    tracing::info!(
        "serve: V4.1 idle live sessions: {} (budget {})",
        prepared.live_states,
        prepared.live_state_bytes.map_or_else(
            || "count only".to_owned(),
            |bytes| format!("{} MiB", bytes >> 20)
        )
    );
    backend.reuse = reuse::Store::new(plan.bytes, plan.policy)
        .with_live_budget(prepared.live_states, prepared.live_state_bytes);
    if prepared.max_active > 1 {
        tracing::info!(
            "serve: V4.1 continuous batching: up to {} requests in flight (prefill {} rows per call while others decode)",
            prepared.max_active,
            scheduler::LOADED_PREFILL_ROWS
        );
        backend.scheduler = Some(scheduler::Scheduler::new(prepared.max_active));
    }
    listening.serve(started.elapsed().as_secs_f64() * 1e3, &mut backend)
}

/// Compile the batched decode path's pipelines before serving: on M5 Ultra
/// the first `step_batch` of a process otherwise spends ~9 s compiling them
/// while the first concurrent requests wait. One step of `width` (at most 8)
/// throwaway two-position states, plus one plain step; results are discarded
/// and the states dropped.
fn warm_batched_decode(forward: &Forward<'_>, width: usize, token: u32) -> Result<()> {
    let started = Instant::now();
    let width = width.clamp(2, 8);
    let cancel = Some(crate::shutdown::cancellation_flag());
    let mut states = (0..=width)
        .map(|_| forward.new_state(2))
        .collect::<std::result::Result<Vec<_>, StateError>>()?;
    let (single, batch) = states.split_first_mut().expect("width >= 2");
    forward
        .step(single, token, cancel)
        .map_err(anyhow::Error::msg)?;
    let mut members: Vec<_> = batch.iter_mut().collect();
    forward
        .step_batch(&mut members, &vec![token; width], cancel)
        .map_err(anyhow::Error::msg)?;
    tracing::info!(
        "serve: V4.1 decode warm-up: one step and one {width}-wide batched step in {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

mod cursor;
mod execution;
#[path = "backend_v41/io.rs"]
mod memory_io;
mod reuse;
mod scheduler;

pub(crate) struct Backend<'a> {
    ctx: &'a MetalContext,
    forward: Forward<'a>,
    settings: Settings,
    reuse: reuse::Store,
    scheduler: Option<scheduler::Scheduler<qwen_llm::deepseek_v41::forward::State>>,
}

/// Caller-retained logits and sampler candidates/weights/sort storage, separate
/// from the forward's request estimate. CPU-only bytes never consume Metal headroom.
pub(crate) fn caller_cpu_reserve(output_peak: u64, transport: u64, vocab: usize) -> Option<u64> {
    output_peak
        .checked_add(transport)?
        .checked_add(u64::try_from(vocab).ok()?.checked_mul(64)?)
}

pub(crate) fn admit_request_memory(
    estimate: RequestMemoryEstimate,
    extra_cpu: u64,
    signals: MetalMemorySignals,
) -> std::result::Result<(), qwen_llm::metal::MemoryAdmissionDenied> {
    let cpu = estimate.cpu_bytes.checked_add(extra_cpu).ok_or_else(|| {
        qwen_llm::metal::MemoryAdmissionDenied::new(
            qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
            None,
            signals,
            signals
                .recommended_max_bytes
                .checked_sub(signals.current_allocated_bytes),
        )
        .expect("required byte overflow refuses admission")
    })?;
    let decision = qwen_llm::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        estimate.metal_bytes,
        cpu,
        0,
        signals,
        true,
    );
    match decision.refusal() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(crate) struct Settings {
    tokenizer: NativeTokenizer,
    context: usize,
    limits: OutputLimits,
    sampling: SamplingConfig,
    eos: u32,
    model_id: String,
    headroom: fn() -> Option<HostMemoryHeadroom>,
}

impl Settings {
    pub(crate) fn new(
        tokenizer: NativeTokenizer,
        context: usize,
        default_max: usize,
        sampling: SamplingConfig,
        eos: u32,
        model_id: String,
    ) -> Result<Self> {
        ensure!(
            context > 0 && default_max > 0 && default_max <= context,
            "V4.1 default output limit must fit positive context"
        );
        sampling.validate()?;
        ensure!(
            tokenizer.encode(deepseek_recipe_encoding::v4::EOS_TOKEN, false)? == [eos as i32],
            "V4.1 tokenizer/config EOS mismatch"
        );
        ensure!(eos <= i32::MAX as u32, "EOS ID exceeds server token range");
        let max_piece_bytes = tokenizer.max_decoded_piece_bytes();
        Ok(Self {
            tokenizer,
            context,
            limits: OutputLimits {
                default_max_tokens: default_max,
                max_piece_bytes,
            },
            sampling,
            eos,
            model_id,
            headroom: qwen_llm::metal::host_memory_headroom,
        })
    }

    fn profile(&self) -> RequestProfile {
        RequestProfile::DeepSeekV41 {
            limits: self.limits,
            sampling: Some(self.sampling.into()),
        }
    }
}

#[cfg(test)]
fn finite_headroom(remaining: u64) -> Option<HostMemoryHeadroom> {
    HostMemoryHeadroom::from_budget(
        qwen_llm::metal::HostMemoryBudget {
            physical_bytes: 5 << 40,
            conservative_available_bytes: remaining,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        },
        0,
        0,
    )
}

impl<'a> Backend<'a> {
    pub(crate) fn new(ctx: &'a MetalContext, forward: Forward<'a>, settings: Settings) -> Self {
        Self {
            ctx,
            forward,
            settings,
            reuse: reuse::Store::new(0, Default::default()),
            scheduler: None,
        }
    }
}

/// The resident forward as the batch scheduler's engine.
struct ForwardEngine<'a, 'b> {
    forward: &'a Forward<'b>,
    ctx: &'a MetalContext,
    cold_plan: Option<qwen_llm::deepseek_v41::forward::RequestPlan>,
    epoch: Instant,
}

impl execution::Engine<qwen_llm::deepseek_v41::forward::State> for ForwardEngine<'_, '_> {
    fn admit(
        &mut self,
        positions: usize,
        prefill_tokens: usize,
        reserve: u64,
    ) -> std::result::Result<(), StateError> {
        let plan = self
            .forward
            .plan_request(positions, prefill_tokens, reserve)?;
        admit_request_memory(plan.memory(), reserve, self.ctx.memory_signals())
            .map_err(StateError::MemoryAdmissionDenied)?;
        self.cold_plan = Some(plan);
        Ok(())
    }
    fn new_state(
        &mut self,
        _: usize,
    ) -> std::result::Result<qwen_llm::deepseek_v41::forward::State, ServeError> {
        self.forward
            .new_state_from_plan(
                &self
                    .cold_plan
                    .take()
                    .expect("cold allocation follows admission"),
            )
            .map_err(state_error)
    }
    fn prefill(
        &mut self,
        state: &mut qwen_llm::deepseek_v41::forward::State,
        tokens: &[u32],
        cancel: &AtomicBool,
        with_logits: bool,
    ) -> std::result::Result<Vec<f32>, ServeError> {
        if with_logits {
            self.forward.prefill(state, tokens, Some(cancel))
        } else {
            self.forward
                .advance_prefill(state, tokens, Some(cancel))
                .map(|()| Vec::new())
        }
        .map_err(state_error)
    }
    fn step(
        &mut self,
        state: &mut qwen_llm::deepseek_v41::forward::State,
        token: u32,
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String> {
        self.forward.step(state, token, Some(cancel))
    }
    fn now(&mut self) -> Duration {
        self.epoch.elapsed()
    }
}

impl scheduler::BatchEngine<qwen_llm::deepseek_v41::forward::State> for ForwardEngine<'_, '_> {
    fn step_batch(
        &mut self,
        states: &mut [&mut qwen_llm::deepseek_v41::forward::State],
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        self.forward.step_batch(states, tokens, Some(cancel))
    }
    fn can_step(&self, state: &qwen_llm::deepseek_v41::forward::State) -> bool {
        state.is_ready() && state.position() < state.capacity()
    }
    fn poisoned(&self, state: &qwen_llm::deepseek_v41::forward::State) -> bool {
        state.is_poisoned()
    }
    fn position(&self, state: &qwen_llm::deepseek_v41::forward::State) -> usize {
        state.position()
    }
}

impl GenerationBackend for Backend<'_> {
    fn model_id(&self) -> &str {
        &self.settings.model_id
    }
    fn request_profile(&self) -> RequestProfile {
        self.settings.profile()
    }
    fn idle(&mut self) {
        self.reuse.sweep();
    }
    fn max_active(&self) -> usize {
        self.scheduler
            .as_ref()
            .map_or(1, scheduler::Scheduler::max_active)
    }
    fn submit(&mut self, job: super::batching::Job) {
        match &mut self.scheduler {
            Some(scheduler) => scheduler.submit(job),
            None => job.sink.complete(Err(ServeError::server_error(
                "V4.1 continuous batching is not enabled",
            )
            .into())),
        }
    }
    fn pump(&mut self) -> bool {
        let Backend {
            ctx,
            forward,
            settings,
            reuse,
            scheduler,
        } = self;
        let Some(scheduler) = scheduler else {
            return false;
        };
        let (forward, ctx) = (&*forward, *ctx);
        if scheduler.admitting() {
            reuse.sweep();
        }
        let mut engine = ForwardEngine {
            forward,
            ctx,
            cold_plan: None,
            epoch: scheduler.epoch(),
        };
        let progressed = scheduler.pump(
            settings,
            &mut engine,
            &mut reuse::Session::new(forward, reuse, ctx),
        );
        for _ in 0..scheduler.take_server_failures() {
            self.request_failed_on_server();
        }
        progressed
    }
    fn abort_all(&mut self) {
        if let Some(scheduler) = &mut self.scheduler {
            scheduler.abort_all();
        }
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let protocol = self
            .settings
            .profile()
            .resolve_output(request)
            .map_err(BackendFailure::Serve)?;
        let super::output_partition::OutputProtocol::DeepSeekV41 { max_bytes, .. } = protocol
        else {
            return Err(BackendFailure::Serve(ServeError::server_error(
                "V4.1 request resolved to a non-V4.1 output protocol",
            )));
        };
        self.generate_prompt(request, prompt, None, max_bytes, sink)
    }

    fn generate_prepared(
        &mut self,
        prepared: std::sync::Arc<super::http::PreparedResponse>,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let super::output_partition::OutputProtocol::DeepSeekV41 { max_bytes, .. } =
            prepared.validated.output()
        else {
            return Err(BackendFailure::Serve(ServeError::server_error(
                "V4.1 prepared request has a non-V4.1 output protocol",
            )));
        };
        self.generate_prompt(
            prepared.request(),
            &prepared.prompt,
            prepared.boundaries,
            *max_bytes,
            sink,
        )
    }
}

impl Backend<'_> {
    fn generate_prompt(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::http::PromptBoundaries>,
        ceiling: super::output_memory::OutputByteCeiling,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let epoch = Instant::now();
        self.reuse.sweep();
        let mut observations = request.echo_stats.then(memory_io::Observations::default);
        let mut prefill_observations = request
            .echo_stats
            .then(memory_io::ChunkObservations::default);
        let cold_plan = std::cell::RefCell::new(None);
        let mut outcome = self.settings.execute_with_ceiling(
            ceiling,
            request,
            prompt,
            boundaries,
            sink,
            |positions, prefill_tokens, reserve| {
                let plan = self
                    .forward
                    .plan_request(positions, prefill_tokens, reserve)?;
                admit_request_memory(plan.memory(), reserve, self.ctx.memory_signals())
                    .map_err(StateError::MemoryAdmissionDenied)?;
                *cold_plan.borrow_mut() = Some(plan);
                Ok(())
            },
            |_| {
                self.forward
                    .new_state_from_plan(
                        &cold_plan
                            .borrow_mut()
                            .take()
                            .expect("cold allocation follows admission"),
                    )
                    .map_err(state_error)
            },
            &mut reuse::Session::new(&self.forward, &mut self.reuse, self.ctx),
            |state, tokens, cancel, with_logits| {
                memory_io::observe_chunk(
                    prefill_observations.as_mut(),
                    state.position(),
                    tokens.len(),
                    || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
                    || {
                        if with_logits {
                            self.forward.prefill(state, tokens, Some(cancel))
                        } else {
                            self.forward
                                .advance_prefill(state, tokens, Some(cancel))
                                .map(|()| Vec::new())
                        }
                    },
                )
                .map_err(state_error)
            },
            |state, token, cancel| {
                let position = state.position();
                memory_io::observe(
                    observations.as_mut(),
                    position,
                    || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
                    || self.forward.step(state, token, Some(cancel)),
                )
            },
            || epoch.elapsed(),
        )?;
        if let Some(observations) = observations {
            let phases = outcome
                .stats
                .as_mut()
                .expect("shared stats")
                .phases
                .as_mut()
                .expect("V4.1 phases");
            phases["process_token_io"] = observations.json();
            phases["process_prefill_io"] = prefill_observations.expect("stats enabled").json();
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod reuse_tests;
#[cfg(test)]
mod scheduler_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tokenizer_fixture;

fn state_error(error: qwen_llm::deepseek_v41::forward::StateError) -> ServeError {
    match error.memory_admission() {
        Some(denied) => super::transport_memory::memory_refusal("V4.1 state", denied),
        None => ServeError::server_error(format!("V4.1 state: {error}")),
    }
}
