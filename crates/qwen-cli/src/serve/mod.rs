//! `qwen serve` — Open Responses subset facade (S1).
//!
//! Contract: docs/SERVE.md. This module tree is layered so everything below
//! the HTTP loop is pure and unit-testable without a model:
//!
//! - [`items`]: wire types, request parsing, item-sequence validation
//!   (review defect 1: item-list grammar, not turn grammar), and the spec
//!   error envelope.
//! - [`render`]: validated transcript → prompt bytes for the generic Qwen
//!   family, satisfying the normative cases in
//!   `tests/fixtures/serve_render_fixtures_v1.json` (fixture-before-renderer).
//!
//! The serial HTTP/SSE loop lands in the next slice and consumes these.
#![allow(dead_code)] // consumed incrementally; the HTTP slice wires the rest

pub(crate) mod backend;
pub(crate) mod backend_ds4;
pub(crate) mod backend_k2;
pub(crate) mod backend_muse;
pub(crate) mod backend_qwen4exp;
pub(crate) mod decode_loop;
pub(crate) mod durable;
pub(crate) mod events;
pub(crate) mod http;
pub(crate) mod outcome;
pub(crate) mod output_partition;
pub(crate) mod partition;
pub(crate) mod partition_k2;
pub(crate) mod partition_muse;
pub(crate) mod render_ds4;
pub(crate) mod render_k2;
pub(crate) mod render_muse;
pub(crate) mod snapshot_cache;
pub(crate) mod utf8;

pub(crate) use crate::open_responses::{items, render, tool_parse};

use crate::family_profile::profile;
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalMemorySignals;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::Runtime;
use qwen_llm::snapshot_policy::{Evicted, SnapshotPolicyConfig};
use std::net::{TcpListener, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
const ADMISSION_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) const DEFAULT_SERVE_MAX_CONTEXT_TOKENS: usize = 262_144;
pub(crate) const DEFAULT_SERVE_MAX_TOKENS: usize = 65_536;
const GIB: u64 = 1 << 30;
/// Auto budget when neither physical-memory nor Metal signals are readable.
const FALLBACK_SNAPSHOT_CACHE_BYTES: u64 = 4 * GIB;
const MIN_AUTO_SNAPSHOT_CACHE_BYTES: u64 = GIB;
const SNAPSHOT_CAPTURE_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// Resolved RAM snapshot-cache budget and eviction policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotCachePlan {
    pub(crate) bytes: u64,
    pub(crate) policy: SnapshotPolicyConfig,
    /// Inputs of an auto-sized budget; `None` when set explicitly.
    auto: Option<AutoBudgetInputs>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AutoBudgetInputs {
    physical_bytes: Option<u64>,
    metal_recommended_bytes: u64,
    metal_resident_bytes: u64,
}

impl SnapshotCachePlan {
    /// `mib: None` is auto: min(25% of physical RAM, 50% of the Metal working
    /// set left after the resident model), at least 1 GiB. Resolve after the
    /// model is loaded so `signals` include its residency.
    pub(crate) fn resolve(
        mib: Option<u64>,
        policy: SnapshotPolicyConfig,
        signals: MetalMemorySignals,
    ) -> Result<Self> {
        Self::resolve_with(
            mib,
            policy,
            qwen_llm::metal::host_physical_memory_bytes(),
            signals,
        )
    }

    fn resolve_with(
        mib: Option<u64>,
        policy: SnapshotPolicyConfig,
        physical_bytes: Option<u64>,
        signals: MetalMemorySignals,
    ) -> Result<Self> {
        if let Some(mib) = mib {
            let bytes = mib
                .checked_mul(1024 * 1024)
                .context("--snapshot-cache-mib byte conversion overflow")?;
            return Ok(Self {
                bytes,
                policy,
                auto: None,
            });
        }
        let quarter_ram = physical_bytes.map(|bytes| bytes / 4);
        let half_free_working_set = (signals.recommended_max_bytes != 0)
            .then(|| {
                signals
                    .recommended_max_bytes
                    .checked_sub(signals.current_allocated_bytes)
            })
            .flatten()
            .map(|bytes| bytes / 2);
        let bytes = [quarter_ram, half_free_working_set]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(FALLBACK_SNAPSHOT_CACHE_BYTES)
            .max(MIN_AUTO_SNAPSHOT_CACHE_BYTES);
        Ok(Self {
            bytes,
            policy,
            auto: Some(AutoBudgetInputs {
                physical_bytes,
                metal_recommended_bytes: signals.recommended_max_bytes,
                metal_resident_bytes: signals.current_allocated_bytes,
            }),
        })
    }
}

impl std::fmt::Display for SnapshotCachePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "snapshot_cache_bytes={}", self.bytes)?;
        match self.auto {
            None => write!(f, " snapshot_cache_source=explicit")?,
            Some(inputs) => write!(
                f,
                " snapshot_cache_source=auto physical_bytes={} metal_recommended_bytes={} metal_resident_bytes={}",
                inputs
                    .physical_bytes
                    .map_or_else(|| "unknown".to_owned(), |bytes| bytes.to_string()),
                inputs.metal_recommended_bytes,
                inputs.metal_resident_bytes,
            )?,
        }
        write!(
            f,
            " snapshot_half_life_secs={} snapshot_idle_ttl_secs={} snapshot_max_age_secs={}",
            self.policy.half_life.as_secs(),
            self.policy.idle_ttl.as_secs(),
            self.policy.max_age.as_secs(),
        )
    }
}

pub(super) fn log_expired_snapshots(family: &str, expired: &Evicted) {
    if !expired.is_empty() {
        tracing::info!(
            target: "qwen_diag",
            "serve: {family} snapshot cache expired entries={} freed_bytes={}",
            expired.ids.len(),
            expired.bytes,
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SnapshotCaptureDenial {
    EstimateOverflow,
    InvalidMetalSignal,
    MetalHeadroom,
    ProcessSignalUnavailable,
    ProcessHeadroom { deficit_bytes: u64 },
}

/// [`snapshot_capture_admission`], first relieving a process-footprint
/// deficit by evicting cached snapshots and re-checking once. A Metal
/// headroom denial is not retried: snapshots live in CPU arenas, so evicting
/// them does not change the Metal allocation signal.
pub(super) fn admit_snapshot_capture(
    bytes: u64,
    signals: impl Fn() -> MetalMemorySignals,
    evict_for: impl FnOnce(u64) -> Evicted,
) -> std::result::Result<(), (SnapshotCaptureDenial, MetalMemorySignals)> {
    let observed = signals();
    let deficit_bytes = match snapshot_capture_admission(bytes, observed) {
        Ok(()) => return Ok(()),
        Err(SnapshotCaptureDenial::ProcessHeadroom { deficit_bytes }) => deficit_bytes,
        Err(reason) => return Err((reason, observed)),
    };
    let evicted = evict_for(deficit_bytes);
    if evicted.is_empty() {
        return Err((
            SnapshotCaptureDenial::ProcessHeadroom { deficit_bytes },
            observed,
        ));
    }
    tracing::info!(
        target: "qwen_diag",
        "serve: snapshot cache evicted for process headroom; entries={} freed_bytes={} deficit_bytes={deficit_bytes}",
        evicted.ids.len(),
        evicted.bytes,
    );
    let observed = signals();
    snapshot_capture_admission(bytes, observed).map_err(|reason| (reason, observed))
}

pub(super) fn snapshot_capture_admission(
    bytes: u64,
    signals: MetalMemorySignals,
) -> Result<(), SnapshotCaptureDenial> {
    let required = bytes
        .checked_add(SNAPSHOT_CAPTURE_HEADROOM_BYTES)
        .ok_or(SnapshotCaptureDenial::EstimateOverflow)?;
    let metal_available = signals
        .recommended_max_bytes
        .checked_sub(signals.current_allocated_bytes)
        .filter(|_| signals.recommended_max_bytes != 0)
        .ok_or(SnapshotCaptureDenial::InvalidMetalSignal)?;
    if required > metal_available {
        return Err(SnapshotCaptureDenial::MetalHeadroom);
    }
    match signals.process_limit_remaining_bytes {
        // Darwin reports zero when this advisory budget is omitted. Match the
        // engine admission contract: Metal headroom remains authoritative.
        Some(0) => {}
        Some(process_available) if required > process_available => {
            return Err(SnapshotCaptureDenial::ProcessHeadroom {
                deficit_bytes: required - process_available,
            });
        }
        Some(_) => {}
        None => return Err(SnapshotCaptureDenial::ProcessSignalUnavailable),
    }
    Ok(())
}

/// Recognised is not served: a family is served when its profile declares a
/// `GenerationBackend`.
fn supports_serve_family(family: Option<ModelFamily>) -> bool {
    family.is_some_and(|family| profile(family).serve_backend)
}

/// Limits for a family whose resident session capacity is fixed at load
/// (Muse Glimmer, Flash-Next): both ceilings must be explicit.
fn fixed_session_limits(
    family: ModelFamily,
    model_context: usize,
    max_context_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> Result<(usize, usize)> {
    let family = profile(family).display;
    let context_limit = max_context_tokens.with_context(|| {
        format!("{family} serve requires --max-context-tokens because its resident session capacity is fixed at startup")
    })?;
    ensure!(
        context_limit <= model_context,
        "{family} --max-context-tokens {context_limit} exceeds model context {model_context}",
    );
    let default_max_tokens = max_tokens.with_context(|| {
        format!("{family} serve requires explicit --max-tokens; the generic 65536-token default exceeds its session capacity")
    })?;
    ensure!(
        default_max_tokens <= context_limit,
        "{family} --max-tokens {default_max_tokens} exceeds --max-context-tokens {context_limit}"
    );
    Ok((context_limit, default_max_tokens))
}

/// `qwen serve` entry: resident model, serial accept loop.
pub(crate) fn run_serve(invocation: crate::cli::ServeInvocation) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let gguf = GgufFile::open(&invocation.model)
        .with_context(|| format!("open model {}", invocation.model.display()))?;
    let Some(family) = ModelFamily::detect(&gguf) else {
        bail!(
            "qwen serve does not support model architecture {:?}; recognised families: {} (docs/SERVE.md)",
            gguf.architecture(),
            ModelFamily::ALL
                .iter()
                .map(|family| family.architecture_name())
                .collect::<Vec<_>>()
                .join(", ")
        );
    };
    ensure!(
        supports_serve_family(Some(family)),
        "qwen serve has no backend for {} (docs/SERVE.md)",
        family.architecture_name()
    );
    // Keep K2 out of the generic serve admission and listener setup. Its
    // resident plan and raw request contract are owned by the K2 lane.
    match family {
        ModelFamily::K2Horizon => {
            let prepared = backend_k2::Prepared::new(&gguf, &invocation)?;
            // Fail cheap on a busy/invalid loopback address, after K2-only admission
            // but before taking the GPU lease or loading weights.
            let listener = bind_loopback(&invocation.addr)?;
            let mut trace = invocation
                .trace_sse
                .as_deref()
                .map(http::TraceLog::open)
                .transpose()?;
            let model_id = invocation
                .model
                .file_stem()
                .and_then(|stem| stem.to_str())
                .context("model path has no printable file stem")?
                .to_owned();
            crate::shutdown::checkpoint()?;
            let ctx = qwen_llm::metal::MetalContext::new()?;
            let started = Instant::now();
            let model = qwen_llm::k2_horizon_runtime::K2LoadedModel::load(
                &ctx,
                &gguf,
                u32::try_from(prepared.capacity)?,
            )?;
            let load_ms = started.elapsed().as_secs_f64() * 1e3;
            tracing::info!(target: "qwen_diag", "serve limits: family=k2_horizon raw_input_string_only capacity={} snapshot_cache_bytes=0", prepared.capacity);
            let mut backend = backend_k2::K2Backend::new(&model, prepared, model_id.clone());
            return accept_loop(listener, &model_id, load_ms, &mut backend, &mut trace);
        }
        ModelFamily::Qwen35
        | ModelFamily::Qwen35Moe
        | ModelFamily::Qwen4Exp
        | ModelFamily::DeepSeek4
        | ModelFamily::MuseGlimmer => {}
    }
    // Drafter admission is a header-level decision: refuse unsupported
    // family/shape combinations and bind the drafter's metadata before the
    // target's weights are loaded. `EngineBackend::new` still performs the
    // GPU copy from the path; consolidating that reuse waits for the serve
    // backend to settle.
    let drafter = crate::drafter_policy::PreparedDrafter::prepare(
        invocation.drafter.as_deref(),
        &gguf,
        Some(family),
        crate::drafter_policy::Lane::Serve,
    )?;
    drop(drafter);
    // Bind before loading weights: an unresolvable, non-loopback, or busy
    // address is a startup error, not something to discover after a
    // multi-gigabyte load. Connections arriving during load queue in the
    // kernel backlog and are answered once the accept loop starts.
    let listener = bind_loopback(&invocation.addr)?;
    let mut trace = invocation
        .trace_sse
        .as_deref()
        .map(http::TraceLog::open)
        .transpose()
        .context("open --trace-sse log")?;
    let model_id = invocation
        .model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("model path has no printable file stem")?
        .to_owned();

    match family {
        ModelFamily::K2Horizon => unreachable!("K2 Horizon returned above"),
        ModelFamily::MuseGlimmer => {
            let math_options = backend_muse::read_math_options()?;
            let config = qwen_llm::muse_glimmer::MuseGlimmerConfig::from_gguf(&gguf)
                .context("bind Muse Glimmer serve contract")?;
            let (context_limit, default_max_tokens) = fixed_session_limits(
                ModelFamily::MuseGlimmer,
                config.context_length as usize,
                invocation.max_context_tokens,
                invocation.max_tokens,
            )?;
            crate::shutdown::checkpoint()?;
            let ctx = qwen_llm::metal::MetalContext::new().context("initialize Metal context")?;
            let load_t0 = Instant::now();
            let mut backend = backend_muse::MuseGlimmerBackend::new_with_options(
                ctx,
                gguf,
                &invocation.model,
                model_id.clone(),
                default_max_tokens,
                context_limit,
                math_options,
            )?;
            let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
            let math_options = backend.math_options();
            tracing::info!(target: "qwen_diag", "serve limits: family=muse_glimmer max_context_tokens={} default_max_tokens={} snapshot_cache_bytes=0 matrix_prefill={} split_decode={}", context_limit, default_max_tokens, math_options.matrix_prefill, math_options.split_decode);
            crate::shutdown::checkpoint()?;
            accept_loop(listener, &model_id, load_ms, &mut backend, &mut trace)
        }
        ModelFamily::DeepSeek4 => {
            // DS4 sizes its session from a forward budget fixed at startup, so
            // serve must be told the context ceiling up front (the CLI's stdin
            // JSONL lane has the same requirement).
            let context_limit = invocation.max_context_tokens.context(
                "DeepSeek V4 serve requires --max-context-tokens: the session forward budget is fixed at startup",
            )?;
            let forward_limit = crate::deepseek_v4_forward_budget_for_context_limit(context_limit)?;
            crate::shutdown::checkpoint()?;
            let ctx = qwen_llm::metal::MetalContext::new().context("initialize Metal context")?;
            let mut backend = backend_ds4::DeepSeekV4Backend::new(
                ctx,
                gguf,
                model_id.clone(),
                invocation.max_tokens.unwrap_or(DEFAULT_SERVE_MAX_TOKENS),
                forward_limit,
                crate::DeepSeekV4MultigroupSelectorArg::Auto,
                invocation.snapshot_cache_mib,
                invocation.snapshot_policy,
            )?;
            tracing::info!(target: "qwen_diag", "serve limits: family=deepseek_v4 max_context_tokens={} {}", context_limit, backend.snapshot_cache_plan);
            match invocation.durable.resolve("deepseek_v4") {
                Ok(Some(plan)) => {
                    if let Err(error) = backend.attach_durable(plan, &invocation.model) {
                        tracing::warn!(target: "qwen_diag", "serve durable: family=deepseek_v4 tier disabled: {error:#}");
                    }
                }
                Ok(None) => {
                    tracing::info!(target: "qwen_diag", "serve durable: family=deepseek_v4 tier off")
                }
                Err(error) => {
                    tracing::warn!(target: "qwen_diag", "serve durable: family=deepseek_v4 tier disabled: {error:#}")
                }
            }
            crate::shutdown::checkpoint()?;
            accept_loop(listener, &model_id, 0.0, &mut backend, &mut trace)
        }
        ModelFamily::Qwen4Exp => {
            if let Some(failure) =
                crate::qwen4exp_prompt_capability_failure(ModelFamily::Qwen4Exp, &gguf)
            {
                bail!(
                    "Qwen3.8-Flash-Next serve does not support the declared {}",
                    failure.as_str()
                );
            }
            let config = qwen_llm::qwen4exp::Qwen4ExpConfig::from_gguf(&gguf)
                .context("bind Qwen3.8-Flash-Next serve geometry")?;
            let (context_limit, default_max_tokens) = fixed_session_limits(
                ModelFamily::Qwen4Exp,
                config.context_length as usize,
                invocation.max_context_tokens,
                invocation.max_tokens,
            )?;
            crate::shutdown::checkpoint()?;
            let ctx = qwen_llm::metal::MetalContext::new().context("initialize Metal context")?;
            let load_t0 = Instant::now();
            let mut backend = backend_qwen4exp::FlashNextBackend::new(
                ctx,
                Box::leak(Box::new(gguf)),
                model_id.clone(),
                default_max_tokens,
                context_limit,
                invocation.snapshot_cache_mib,
                invocation.snapshot_policy,
            )?;
            let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
            tracing::info!(target: "qwen_diag", "serve limits: family=qwen4exp max_context_tokens={context_limit} default_max_tokens={default_max_tokens} {}", backend.snapshot_cache_plan);
            crate::shutdown::checkpoint()?;
            accept_loop(listener, &model_id, load_ms, &mut backend, &mut trace)
        }
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {
            // The same release identity `qwen run` resolves; serve must not render
            // a Qwen3.8 model with the generic contract.
            let identity = crate::prompt_template::identify_qwen_release_for_gguf(&gguf)
                .context("identify the loaded model's Qwen release")?;
            if let Some(warning) = identity.warning() {
                tracing::warn!(target: "qwen_diag", "serve: {warning}");
            }
            let template = identity.template.serve_template();
            let no_thinking_supported = template.verified();
            // Deriving the default ceiling from the model requires readable context
            // metadata; an explicit --max-context-tokens does not.
            let declared_context =
                match invocation.max_context_tokens {
                    Some(_) => None,
                    None => Some(gguf.declared_context_length().context(
                        "read the model's declared context length for the serve ceiling",
                    )?),
                };
            drop(gguf);
            crate::shutdown::checkpoint()?;
            let runtime = Runtime::metal().context("initialize Metal runtime")?;
            let load_t0 = Instant::now();
            let loaded = runtime
                .load_model_with_config(
                    &invocation.model,
                    qwen_llm::runtime::LoadedModelConfig::default(),
                )
                .with_context(|| format!("load model {}", invocation.model.display()))?;
            let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
            // Sized after load so auto budgets see the resident model.
            let snapshot_cache_plan = SnapshotCachePlan::resolve(
                invocation.snapshot_cache_mib,
                invocation.snapshot_policy,
                loaded.context().memory_signals(),
            )?;
            loaded.set_prefix_cache_max_bytes(snapshot_cache_plan.bytes);
            loaded.set_prefix_cache_policy(snapshot_cache_plan.policy);
            // Admission ceiling: explicit, else the smaller of the hard default and
            // the model's declared context length.
            let (context_ceiling, context_source) =
                match (invocation.max_context_tokens, declared_context) {
                    (Some(explicit), _) => (explicit, "explicit"),
                    (None, Some(declared)) if declared < DEFAULT_SERVE_MAX_CONTEXT_TOKENS => {
                        (declared, "declared_context_length")
                    }
                    (None, _) => (DEFAULT_SERVE_MAX_CONTEXT_TOKENS, "default_hard_ceiling"),
                };
            let mut backend = backend::EngineBackend::new(
                loaded,
                model_id.clone(),
                invocation.max_tokens.unwrap_or(DEFAULT_SERVE_MAX_TOKENS),
                invocation.max_context_tokens,
                context_ceiling,
                invocation.drafter.as_deref(),
                template,
                no_thinking_supported,
            )?;
            tracing::info!(target: "qwen_diag", "serve limits: family=qwen max_context_tokens={context_ceiling} context_source={context_source} {snapshot_cache_plan}");
            match invocation.durable.resolve("qwen") {
                Ok(Some(plan)) => {
                    if let Err(error) =
                        backend.attach_durable(plan, &invocation.model, snapshot_cache_plan.bytes)
                    {
                        tracing::warn!(target: "qwen_diag", "serve durable: family=qwen tier disabled: {error:#}");
                    }
                }
                Ok(None) => {
                    tracing::info!(target: "qwen_diag", "serve durable: family=qwen tier off")
                }
                Err(error) => {
                    tracing::warn!(target: "qwen_diag", "serve durable: family=qwen tier disabled: {error:#}")
                }
            }
            crate::shutdown::checkpoint()?;

            accept_loop(listener, &model_id, load_ms, &mut backend, &mut trace)
        }
    }
}

fn bind_loopback(addr: &str) -> Result<TcpListener> {
    let addresses = addr
        .to_socket_addrs()
        .with_context(|| format!("resolve listen address {addr}"))?
        .collect::<Vec<_>>();
    ensure!(
        !addresses.is_empty(),
        "listen address {addr} resolved to nothing"
    );
    ensure!(
        addresses.iter().all(|address| address.ip().is_loopback()),
        "qwen serve requires a loopback listen address; rejected {addr}"
    );
    TcpListener::bind(addresses.as_slice()).with_context(|| format!("bind {addr}"))
}

fn wait_for_connection(listener: &TcpListener) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let mut descriptor = libc::pollfd {
        fd: listener.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // The listener owns the descriptor throughout this bounded wait. Readiness
    // wakes immediately; the timeout only bounds cooperative shutdown latency.
    let result = unsafe { libc::poll(&mut descriptor, 1, ACCEPT_POLL_INTERVAL.as_millis() as i32) };
    if result < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    } else if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
        return Err(std::io::Error::other("HTTP listener readiness failed"));
    }
    Ok(())
}

fn spawn_acceptor(
    listener: TcpListener,
    sender: SyncSender<std::net::TcpStream>,
    ready: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
) -> Result<JoinHandle<()>> {
    listener
        .set_nonblocking(true)
        .context("configure nonblocking HTTP acceptor")?;
    std::thread::Builder::new()
        .name("qwen-http-accept".into())
        .spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(error) = http::configure_stream(&stream) {
                            tracing::warn!("serve: configure accepted socket failed: {error}");
                            continue;
                        }
                        if ready.swap(false, Ordering::AcqRel) {
                            if sender.send(stream).is_err() {
                                break;
                            }
                        } else if let Err(error) = http::write_busy_response(&stream) {
                            tracing::info!(target: "qwen_diag", "serve: busy response aborted: {error}");
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if let Err(error) = wait_for_connection(&listener) {
                            tracing::warn!("serve: listener wait failed: {error}");
                            std::thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                    }
                    Err(error) => {
                        tracing::warn!("serve: accept failed: {error}");
                        std::thread::sleep(ACCEPT_POLL_INTERVAL);
                    }
                }
            }
        })
        .context("spawn HTTP acceptor")
}

/// Serial generation loop shared by every family backend. Acceptance runs on
/// a separate thread so busy clients can be rejected without moving backend.
fn accept_loop(
    listener: TcpListener,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
) -> Result<()> {
    accept_loop_with_checkpoint(listener, model_id, load_ms, backend, trace, || {
        crate::shutdown::checkpoint()
    })
}

fn accept_loop_with_checkpoint(
    listener: TcpListener,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<()> {
    let local_addr = listener
        .local_addr()
        .map_or_else(|_| "<unknown>".to_owned(), |addr| addr.to_string());
    tracing::info!(
        target: "qwen_diag",
        "serve: listening on http://{} model={} load_ms={:.1} (serial; POST /v1/responses, GET /v1/models)",
        local_addr,
        model_id,
        load_ms,
    );
    let (sender, receiver) = sync_channel(0);
    // Publish initial readiness before the acceptor can observe a connection;
    // otherwise an idle server has a startup window that returns a false 503.
    let ready = Arc::new(AtomicBool::new(true));
    let accept_ready = Arc::clone(&ready);
    let stopping = Arc::new(AtomicBool::new(false));
    let accept_stopping = Arc::clone(&stopping);
    let acceptor = spawn_acceptor(listener, sender, accept_ready, accept_stopping)?;

    let result = (|| -> Result<()> {
        loop {
            checkpoint()?;
            ready.store(true, Ordering::Release);
            let stream = match receiver.recv_timeout(ADMISSION_POLL_INTERVAL) {
                Ok(stream) => stream,
                Err(RecvTimeoutError::Timeout) => {
                    backend.idle();
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            // A signal may arrive while admission is parked in recv_timeout.
            // Never start an admitted request without checking it again.
            checkpoint()?;
            if let Err(error) = http::handle_connection(&stream, backend, trace.as_mut()) {
                tracing::info!(target: "qwen_diag", "serve: connection aborted: {error}");
            }
        }
        Ok(())
    })();

    ready.store(false, Ordering::Release);
    stopping.store(true, Ordering::Release);
    drop(receiver);
    let acceptor_result = acceptor.join();
    // Stop accepting before the bounded durable flush, so clients see a
    // closed port rather than a stalled server during shutdown.
    backend.shutdown();
    if acceptor_result.is_err() {
        return Err(anyhow::anyhow!("HTTP acceptor panicked"));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREAD_EXIT_TIMEOUT: Duration = Duration::from_secs(2);

    fn assert_thread_finishes<T>(thread: &JoinHandle<T>, context: &str) {
        let started = Instant::now();
        while !thread.is_finished() && started.elapsed() < THREAD_EXIT_TIMEOUT {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(thread.is_finished(), "{context}");
    }

    struct UnreachableBackend;

    impl http::GenerationBackend for UnreachableBackend {
        fn model_id(&self) -> &str {
            "shutdown-test"
        }

        fn generate(
            &mut self,
            _request: &items::ServeRequest,
            _prompt: &str,
            _sink: &mut dyn http::GenerationSink,
        ) -> std::result::Result<http::GenerationOutcome, http::BackendFailure> {
            panic!("pre-loop shutdown must not dispatch a request")
        }
    }

    #[test]
    fn binding_rejects_non_loopback() {
        let error = bind_loopback("0.0.0.0:0").expect_err("wildcard must be rejected");
        assert!(error.to_string().contains("requires a loopback"));
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        assert!(listener.local_addr().unwrap().ip().is_loopback());
    }

    #[test]
    fn serve_family_gate_lists_backends_explicitly() {
        for family in ModelFamily::ALL {
            assert_eq!(
                supports_serve_family(Some(*family)),
                profile(*family).serve_backend,
                "{family:?}"
            );
            assert!(profile(*family).serve_backend, "{family:?}");
        }
        assert!(!supports_serve_family(None));
    }

    #[test]
    fn muse_limits_require_explicit_bounded_capacity_and_output_default() {
        assert_eq!(
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, Some(7_168), Some(2_048),)
                .unwrap(),
            (7168, 2048)
        );
        assert_eq!(
            fixed_session_limits(
                ModelFamily::MuseGlimmer,
                131_072,
                Some(131_072),
                Some(16_384),
            )
            .unwrap(),
            (131_072, 16_384)
        );
        assert!(
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, None, Some(2_048)).is_err()
        );
        assert!(
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, Some(7_168), None).is_err()
        );
        assert!(
            fixed_session_limits(
                ModelFamily::MuseGlimmer,
                131_072,
                Some(131_073),
                Some(2_048)
            )
            .is_err()
        );
        assert!(
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, Some(1_024), Some(2_048))
                .is_err()
        );
    }

    #[test]
    fn acceptor_stop_flag_releases_listener_and_joins() {
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = sync_channel(0);
        let ready = Arc::new(AtomicBool::new(false));
        let stopping = Arc::new(AtomicBool::new(false));
        let acceptor =
            spawn_acceptor(listener, sender, Arc::clone(&ready), Arc::clone(&stopping)).unwrap();
        stopping.store(true, Ordering::Release);
        drop(receiver);
        assert_thread_finishes(
            &acceptor,
            "acceptor did not observe its stop flag without a wake connection",
        );
        assert!(acceptor.join().is_ok());
        drop(TcpListener::bind(address).unwrap());
    }

    #[test]
    fn shutdown_pending_before_accept_loop_never_needs_a_wake_connection() {
        let server = std::thread::spawn(|| {
            let mut backend = UnreachableBackend;
            let mut trace = None;
            accept_loop_with_checkpoint(
                bind_loopback("127.0.0.1:0").unwrap(),
                "shutdown-test",
                0.0,
                &mut backend,
                &mut trace,
                || Err(anyhow::anyhow!("termination already requested")),
            )
        });
        assert_thread_finishes(
            &server,
            "pending shutdown did not unwind without a wake connection",
        );
        let error = server
            .join()
            .expect("shutdown regression thread panicked")
            .expect_err("pending shutdown must unwind before admission");
        assert!(error.to_string().contains("termination already requested"));
    }

    #[test]
    fn initially_ready_acceptor_admits_the_first_connection() {
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = sync_channel(0);
        let ready = Arc::new(AtomicBool::new(true));
        let stopping = Arc::new(AtomicBool::new(false));
        let acceptor =
            spawn_acceptor(listener, sender, Arc::clone(&ready), Arc::clone(&stopping)).unwrap();
        let client = std::net::TcpStream::connect(address).unwrap();
        let admitted = receiver
            .recv_timeout(ACCEPT_POLL_INTERVAL * 4)
            .expect("first connection should be admitted, not rejected as busy");
        stopping.store(true, Ordering::Release);
        drop(receiver);
        drop(admitted);
        drop(client);
        acceptor.join().unwrap();
    }

    #[test]
    fn idle_acceptor_wakes_and_preserves_busy_rejection() {
        use std::io::Read;
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = sync_channel(0);
        let ready = Arc::new(AtomicBool::new(true));
        let stopping = Arc::new(AtomicBool::new(false));
        let acceptor =
            spawn_acceptor(listener, sender, Arc::clone(&ready), Arc::clone(&stopping)).unwrap();
        std::thread::sleep(ACCEPT_POLL_INTERVAL + Duration::from_millis(5));
        let client = std::net::TcpStream::connect(address).unwrap();
        let admitted = receiver
            .recv_timeout(THREAD_EXIT_TIMEOUT)
            .expect("idle listener must wake");
        assert!(!ready.load(Ordering::Acquire));
        let mut busy = std::net::TcpStream::connect(address).unwrap();
        busy.set_read_timeout(Some(THREAD_EXIT_TIMEOUT)).unwrap();
        let mut response = String::new();
        busy.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"));
        assert!(response.to_ascii_lowercase().contains("retry-after: 1"));
        stopping.store(true, Ordering::Release);
        drop(receiver);
        drop(admitted);
        drop(client);
        assert_thread_finishes(&acceptor, "readiness wait did not stop");
        acceptor.join().unwrap();
    }

    #[test]
    fn snapshot_capture_admission_fails_closed() {
        let signals = |metal, process| MetalMemorySignals {
            recommended_max_bytes: metal,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: process,
        };
        assert_eq!(
            snapshot_capture_admission(1, signals(600_000_000, None)),
            Err(SnapshotCaptureDenial::ProcessSignalUnavailable)
        );
        assert_eq!(
            snapshot_capture_admission(600_000_000, signals(10_000, Some(10_000))),
            Err(SnapshotCaptureDenial::MetalHeadroom)
        );
        assert!(snapshot_capture_admission(1, signals(600_000_000, Some(600_000_000))).is_ok());
        assert!(snapshot_capture_admission(1, signals(600_000_000, Some(0))).is_ok());
        assert_eq!(
            snapshot_capture_admission(600_000_000, signals(10_000, Some(0))),
            Err(SnapshotCaptureDenial::MetalHeadroom)
        );
    }

    #[test]
    fn auto_snapshot_budget_scales_with_machine_and_resident_model() {
        let policy = SnapshotPolicyConfig::default();
        let signals = |recommended, resident| MetalMemorySignals {
            recommended_max_bytes: recommended,
            current_allocated_bytes: resident,
            process_limit_remaining_bytes: Some(0),
        };
        let auto = |physical, recommended, resident| {
            SnapshotCachePlan::resolve_with(None, policy, physical, signals(recommended, resident))
                .unwrap()
                .bytes
        };
        // 128 GiB host, ~96 GiB working set: RAM quarter vs. free half.
        assert_eq!(auto(Some(128 * GIB), 96 * GIB, 20 * GIB), 32 * GIB);
        assert_eq!(auto(Some(128 * GIB), 96 * GIB, 60 * GIB), 18 * GIB);
        // Nearly full working set clamps up to the floor.
        assert_eq!(auto(Some(128 * GIB), 96 * GIB, 95 * GIB), GIB);
        // Missing signals fall back to whichever remains, then the constant.
        assert_eq!(auto(Some(16 * GIB), 0, 0), 4 * GIB);
        assert_eq!(auto(None, 96 * GIB, 90 * GIB), 3 * GIB);
        assert_eq!(auto(None, 0, 0), FALLBACK_SNAPSHOT_CACHE_BYTES);
        // An explicit value is honored verbatim, including zero.
        let explicit = |mib| {
            SnapshotCachePlan::resolve_with(Some(mib), policy, Some(GIB), signals(1, 0))
                .unwrap()
                .bytes
        };
        assert_eq!(explicit(0), 0);
        assert_eq!(explicit(8192), 8 * GIB);
        assert!(
            SnapshotCachePlan::resolve_with(Some(u64::MAX), policy, None, signals(0, 0)).is_err()
        );
    }

    #[test]
    fn capture_admission_evicts_only_for_process_deficit() {
        let signals = |process| MetalMemorySignals {
            recommended_max_bytes: 10 * GIB,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(process),
        };
        let required = 100 + SNAPSHOT_CAPTURE_HEADROOM_BYTES;
        let mut cache = qwen_llm::snapshot_policy::SnapshotPolicy::new(
            u64::MAX,
            SnapshotPolicyConfig::default(),
        );
        cache.insert(64, 1);
        // Eviction is asked for exactly the deficit; the re-check sees relief.
        let observed = std::cell::Cell::new(required - 40);
        let result = admit_snapshot_capture(
            100,
            || signals(observed.get()),
            |deficit| {
                assert_eq!(deficit, 40);
                observed.set(required);
                cache.evict_for(deficit)
            },
        );
        assert!(result.is_ok());
        assert!(cache.is_empty());
        // Nothing evictable: the original denial stands.
        let result = admit_snapshot_capture(
            100,
            || signals(required - 40),
            |deficit| cache.evict_for(deficit),
        );
        assert_eq!(
            result.unwrap_err().0,
            SnapshotCaptureDenial::ProcessHeadroom { deficit_bytes: 40 }
        );
        // Metal headroom is never retried by evicting CPU snapshots.
        let result = admit_snapshot_capture(
            20 * GIB,
            || signals(0),
            |_| panic!("metal denial must not evict"),
        );
        assert_eq!(result.unwrap_err().0, SnapshotCaptureDenial::MetalHeadroom);
    }
}
