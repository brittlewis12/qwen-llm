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
//! [`request_profile`] owns CPU request semantics; [`transport`] connects one
//! admitted HTTP worker to the resident owner. [`owner_activity`] gates owner
//! maintenance on complete request lifetimes, and [`trace`] owns the shared log.
#![allow(dead_code)] // consumed incrementally; the HTTP slice wires the rest

mod assets;
pub(crate) mod backend;
pub(crate) mod backend_ds4;
mod backend_glm5_next;
pub(crate) mod backend_k2;
pub(crate) mod backend_muse;
pub(crate) mod backend_qwen4exp;
mod control;
pub(crate) mod decode_loop;
pub(crate) mod durable;
pub(crate) mod events;
pub(crate) mod http;
mod idle_residency;
mod jobs;
pub(crate) mod lens_http;
mod native;
pub(crate) mod outcome;
mod output_memory;
pub(crate) mod output_partition;
mod owner_activity;
pub(crate) mod partition;
pub(crate) mod partition_glm5_next;
pub(crate) mod partition_k2;
pub(crate) mod partition_muse;
pub(crate) mod partition_preopened;
pub(crate) mod render_ds4;
pub(crate) mod render_glm5_next;
pub(crate) mod render_k2;
pub(crate) mod render_muse;
pub(crate) mod request_profile;
pub(crate) mod snapshot_cache;
mod trace;
mod transport;
mod transport_memory;
pub(crate) mod utf8;

pub(crate) use crate::open_responses::{items, render, tool_parse};

use crate::family_profile::profile;
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalMemorySignals;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::snapshot_policy::{Evicted, SnapshotPolicyConfig};
use std::net::{TcpListener, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;
#[cfg(test)]
use std::time::Instant;

const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
const ADMISSION_POLL_INTERVAL: Duration = Duration::from_millis(100);

struct Workbench {
    store: Option<Arc<jobs::store::JobStore>>,
    assets: Option<Arc<assets::WebAssets>>,
    access: lens_http::access::BrowserAccess,
}

pub(super) struct Listening {
    listener: TcpListener,
    trace: Option<http::TraceLog>,
    model_id: String,
    workbench: Option<Workbench>,
}

impl Listening {
    fn open(
        invocation: &crate::cli::ServeInvocation,
        workbench: Option<Workbench>,
    ) -> Result<Self> {
        let listener = bind_loopback(&invocation.addr)?;
        let trace = invocation
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
        Ok(Self {
            listener,
            trace,
            model_id,
            workbench,
        })
    }

    pub(super) fn serve(
        mut self,
        load_ms: f64,
        backend: &mut dyn http::GenerationBackend,
    ) -> Result<()> {
        accept_loop(
            self.listener,
            &self.model_id,
            load_ms,
            backend,
            &mut self.trace,
            self.workbench,
        )
    }
}

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

/// Limits for a family whose resident session capacity is fixed at load:
/// both ceilings must be explicit and positive.
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
        context_limit > 0,
        "{family} --max-context-tokens must be greater than 0"
    );
    ensure!(
        context_limit <= model_context,
        "{family} --max-context-tokens {context_limit} exceeds model context {model_context}",
    );
    let default_max_tokens = max_tokens.with_context(|| {
        format!("{family} serve requires explicit --max-tokens; the generic 65536-token default exceeds its session capacity")
    })?;
    ensure!(
        default_max_tokens > 0,
        "{family} --max-tokens must be greater than 0"
    );
    ensure!(
        default_max_tokens <= context_limit,
        "{family} --max-tokens {default_max_tokens} exceeds --max-context-tokens {context_limit}"
    );
    Ok((context_limit, default_max_tokens))
}

/// `qwen serve` entry: resident model, serial accept loop.
/// Refuse an explicit durable-tier request on a family that has none (the
/// user would wait through a cold re-prefill after restart believing it was
/// on); say once that the defaulted tier does not apply; warn that a
/// snapshot budget means nothing to a live-session family.
/// Rejects warmth settings a family cannot honour, so none is silently
/// ignored. Values are judged, not flag presence: an explicit default is
/// indistinguishable from omission (clap `ValueSource` could tell them
/// apart if that ever matters), and values that switch a feature off are
/// always accepted.
fn check_warmth_flags(
    family: &crate::family_profile::FamilyProfile,
    durable: &durable::DurableSnapshotConfig,
    snapshot_cache_mib: Option<u64>,
    snapshot_policy: &qwen_llm::snapshot_policy::SnapshotPolicyConfig,
) -> Result<()> {
    use crate::family_profile::ServeWarmth;
    let durable_off = matches!(durable.dir, durable::DurableDir::Off) || durable.max_mib == Some(0);
    if family.serve_warmth != ServeWarmth::SnapshotsDurable && !durable_off {
        let explicit = matches!(durable.dir, durable::DurableDir::Path(_))
            || durable.max_mib.is_some()
            || durable.min_tokens != durable::DEFAULT_MIN_TOKENS
            || durable.shutdown_secs != durable::DEFAULT_SHUTDOWN_SECS
            || durable.idle_publish_secs != durable::DEFAULT_IDLE_PUBLISH_SECS;
        ensure!(
            !explicit,
            "{} serve has no durable snapshot tier, so --durable-* flags cannot keep prefixes across restarts; drop the flags or pass --durable-snapshot-dir off",
            family.display
        );
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family={} tier unsupported; prefixes do not survive a restart",
            family.display
        );
    }
    if family.serve_warmth == ServeWarmth::SnapshotsDurable
        && !durable_off
        && !family.durable_idle_publish
    {
        ensure!(
            matches!(
                durable.idle_publish_secs,
                0 | durable::DEFAULT_IDLE_PUBLISH_SECS
            ),
            "{} serve's durable tier writes snapshots as it captures them and has no idle publication, so --durable-idle-publish-secs cannot take effect; drop the flag or pass 0",
            family.display
        );
    }
    if family.serve_warmth == ServeWarmth::LiveSession {
        let defaults = qwen_llm::snapshot_policy::SnapshotPolicyConfig::default();
        let unsupported = [
            (
                "--snapshot-cache-mib",
                snapshot_cache_mib.is_some_and(|mib| mib != 0),
            ),
            (
                "--snapshot-idle-ttl-secs",
                !(snapshot_policy.idle_ttl.is_zero()
                    || snapshot_policy.idle_ttl == defaults.idle_ttl),
            ),
            (
                "--snapshot-max-age-secs",
                !(snapshot_policy.max_age.is_zero() || snapshot_policy.max_age == defaults.max_age),
            ),
            // Zero is pure LRU ranking, a policy, not off.
            (
                "--snapshot-half-life-secs",
                snapshot_policy.half_life != defaults.half_life,
            ),
        ];
        if let Some((flag, _)) = unsupported.iter().find(|(_, set)| *set) {
            bail!(
                "{} serve reuses its live session's prefix and keeps no snapshots, so {flag} cannot take effect; drop the flag (`auto` or 0 for --snapshot-cache-mib, 0 for the expiry flags, are accepted)",
                family.display
            );
        }
    }
    Ok(())
}

fn open_workbench(invocation: &crate::cli::ServeInvocation) -> Result<Option<Workbench>> {
    let assets = invocation
        .web_root
        .as_deref()
        .map(assets::WebAssets::open)
        .transpose()
        .context("open prebuilt Lens client")?
        .map(Arc::new);
    let store = invocation
        .lens_data_dir
        .as_deref()
        .map(|root| jobs::store::JobStore::open(root, jobs::store::Limits::default()).map(Arc::new))
        .transpose()
        .context("open durable Lens history")?;
    Ok((store.is_some() || assets.is_some()).then_some(Workbench {
        store,
        assets,
        access: lens_http::access::BrowserAccess::new(invocation.lens_allowed_origin.clone()),
    }))
}

/// Startup stages are header admission, family preparation, workbench files,
/// listener setup, then family start and acceptance. Header admission maps the
/// target GGUF and may open a drafter header; it must not read weight payloads,
/// bind, or initialize Metal. Family preparation inspects GGUF metadata and
/// may build CPU tokenizers; it must not read large payloads, bind, initialize
/// Metal, or open workbench files. Workbench setup reads web assets and may
/// create job-store state; it must not bind or initialize Metal. Listener setup
/// binds the socket and opens the trace log without initializing Metal. Family
/// start initializes Metal and loads weights. Qwen opens and hashes fitted
/// Lens payloads after bind but before Metal because the registry reads large
/// payloads; Muse opens its llama.cpp tokenizer after Metal context creation
/// because llama.cpp initializes its Metal backend there.
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
    check_warmth_flags(
        profile(family),
        &invocation.durable,
        invocation.snapshot_cache_mib,
        &invocation.snapshot_policy,
    )?;
    // Idle residency needs a backend whose weights are no-copy GGUF windows.
    ensure!(
        invocation.idle_residency_secs.unwrap_or(0) == 0 || profile(family).idle_residency_eligible,
        "--idle-residency-secs is not implemented for {} serve: its backend does not name its weight buffers (its default weights are Metal-allocated copies, which stay wired; QWEN_GGUF_NO_COPY storage is not yet covered)",
        family.architecture_name()
    );
    let idle_window = idle_residency::configured_window(
        invocation.idle_residency_secs,
        idle_residency::DEFAULT_WINDOW,
    );
    let template_style = invocation.template_style;
    ensure!(
        invocation.lens_allowed_origin.is_empty()
            || invocation.lens_data_dir.is_some()
            || invocation.web_root.is_some(),
        "--lens-allowed-origin requires --lens-data-dir or --web-root"
    );
    ensure!(
        invocation.lens_config.is_none()
            || matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "--lens-config requires the ordinary Qwen native executor"
    );
    ensure!(
        invocation.lens_config.is_none() || invocation.lens_data_dir.is_some(),
        "--lens-config requires --lens-data-dir"
    );
    ensure!(
        invocation.lens_config.is_none() || template_style == items::TemplateStyle::House,
        "--lens-config requires qualified House native generation"
    );
    if !profile(family).upstream_template_style {
        ensure!(
            template_style == items::TemplateStyle::House,
            "--template-style upstream is defined for Qwen and DeepSeek V4 serve; {} serve renders its release format",
            family.architecture_name()
        );
    } else {
        tracing::info!(target: "qwen_diag", "serve: template_style={}", template_style.as_str());
    }
    // Drafter admission checks the model header before family preparation.
    // EngineBackend::new performs the GPU copy from the path.
    let drafter = crate::drafter_policy::PreparedDrafter::prepare(
        invocation.drafter.as_deref(),
        &gguf,
        Some(family),
        crate::drafter_policy::Lane::Serve,
    )?;
    drop(drafter);
    match family {
        ModelFamily::K2Horizon => {
            let prepared = backend_k2::Prepared::new(&gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend_k2::start(prepared, &gguf, &invocation, listening, idle_window)
        }
        ModelFamily::Glm5Next => {
            let prepared = backend_glm5_next::Prepared::new(&gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend_glm5_next::start(prepared, &gguf, &invocation, listening, idle_window)
        }
        ModelFamily::MuseGlimmer => {
            let prepared = backend_muse::Prepared::new(&gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend_muse::start(prepared, gguf, &invocation, listening, idle_window)
        }
        ModelFamily::DeepSeek4 => {
            let prepared = backend_ds4::Prepared::new(&gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend_ds4::start(prepared, gguf, &invocation, listening, idle_window)
        }
        ModelFamily::Qwen4Exp => {
            let prepared = backend_qwen4exp::Prepared::new(&gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend_qwen4exp::start(prepared, gguf, &invocation, listening, idle_window)
        }
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {
            let prepared = backend::Prepared::new(family, &gguf, &invocation)?;
            let workbench = open_workbench(&invocation)?;
            let listening = Listening::open(&invocation, workbench)?;
            backend::start(prepared, gguf, &invocation, listening)
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

/// Owner-side completion of finished connections: a server-side failure
/// the connection answered after generation (a partition failure) reaches
/// the backend before the completion itself.
fn drain_completions(
    activity: &mut owner_activity::OwnerActivity,
    backend: &mut dyn http::GenerationBackend,
) {
    activity.drain_finished_with_failures(|failed| {
        if failed {
            backend.request_failed_on_server();
        }
        backend.request_finished();
    });
}

fn spawn_acceptor(
    listener: TcpListener,
    sender: SyncSender<control::Event>,
    ready: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
) -> Result<JoinHandle<Result<()>>> {
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
                            if sender.send(control::Event::Incoming(stream)).is_err() {
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
            Ok(())
        })
        .context("spawn HTTP acceptor")
}

/// Single-admission owner loop shared by every family backend. Acceptance and
/// HTTP handling run on CPU threads without moving the resident backend.
fn accept_loop(
    listener: TcpListener,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
    workbench: Option<Workbench>,
) -> Result<()> {
    accept_loop_with_workbench(
        listener,
        model_id,
        load_ms,
        backend,
        trace,
        workbench,
        |_| crate::shutdown::checkpoint(),
    )
}

#[derive(Clone, Copy, PartialEq)]
enum OwnerCheckpoint {
    BeforeAdmission,
    BeforeHandling,
    DuringHandling,
}

fn accept_loop_with_checkpoint(
    listener: TcpListener,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
    checkpoint: impl FnMut(OwnerCheckpoint) -> Result<()>,
) -> Result<()> {
    accept_loop_with_workbench(
        listener, model_id, load_ms, backend, trace, None, checkpoint,
    )
}

fn accept_loop_with_workbench(
    listener: TcpListener,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
    workbench: Option<Workbench>,
    mut checkpoint: impl FnMut(OwnerCheckpoint) -> Result<()>,
) -> Result<()> {
    let local_addr = listener
        .local_addr()
        .map_or_else(|_| "<unknown>".to_owned(), |addr| addr.to_string());
    tracing::info!(
        target: "qwen_diag",
        "serve: listening on http://{} model={} load_ms={:.1} lens_history={} (serial; POST /v1/responses, GET /v1/models)",
        local_addr,
        model_id,
        load_ms,
        workbench.as_ref().is_some_and(|w| w.store.is_some()),
    );
    let control_enabled = workbench.is_some();
    let (sender, receiver) = sync_channel(usize::from(control_enabled));
    // Publish initial readiness before the acceptor can observe a connection;
    // otherwise an idle server has a startup window that returns a false 503.
    let ready = Arc::new(AtomicBool::new(true));
    let accept_ready = Arc::clone(&ready);
    let stopping = Arc::new(AtomicBool::new(false));
    let accept_stopping = Arc::clone(&stopping);
    let mut activity = owner_activity::OwnerActivity::default();
    let admission = activity.admission();
    let gate = control::ExecutionGate::default();
    let acceptor = if let Some(Workbench {
        store,
        assets,
        access,
    }) = workbench
    {
        let native = if let Some(store) = &store {
            backend.native_profile()?.map(|profile| {
                Arc::new(native::NativeAdmission {
                    profile,
                    store: Arc::clone(store),
                    sender: sender.clone(),
                    gate: gate.clone(),
                    activity: admission.clone(),
                }) as Arc<dyn lens_http::Admission>
            })
        } else {
            None
        };
        let lens =
            Arc::new(lens_http::LensApi::new(model_id.into(), store, native).with_access(access));
        backend.set_control_memory_reserve(control::cpu_reserve(lens.history_enabled()));
        match control::spawn(
            listener,
            control::Profile {
                model_id: model_id.into(),
                request: backend.request_profile(),
                lens,
                assets,
                gate: gate.clone(),
                activity: admission.clone(),
                sender,
                trace: trace.as_ref().map(http::TraceLog::factory),
                #[cfg(test)]
                classified: None,
            },
            accept_stopping,
        ) {
            Ok(acceptor) => acceptor,
            Err(cause) => {
                backend.set_control_memory_reserve(0);
                backend.shutdown();
                return Err(cause);
            }
        }
    } else {
        spawn_acceptor(listener, sender, accept_ready, accept_stopping)?
    };
    let mut connection: Option<transport::Connection> = None;

    let result = (|| -> Result<()> {
        loop {
            if control_enabled {
                gate.checkpoint()?;
            }
            if let Some(active) = &mut connection {
                let settled =
                    match active.advance(backend, || checkpoint(OwnerCheckpoint::DuringHandling)) {
                        Ok(settled) => settled,
                        Err(cause) if cause.is::<transport::WorkerPanicked>() => {
                            tracing::warn!("serve: HTTP worker failed: {cause:#}");
                            true
                        }
                        Err(cause) => return Err(cause),
                    };
                if settled {
                    connection.take();
                    drain_completions(&mut activity, backend);
                }
                continue;
            }
            checkpoint(OwnerCheckpoint::BeforeAdmission)?;
            if !control_enabled {
                ready.store(true, Ordering::Release);
            }
            let event = match receiver.recv_timeout(ADMISSION_POLL_INTERVAL) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => {
                    drain_completions(&mut activity, backend);
                    activity.idle_if_quiet(|| backend.idle());
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            // A signal may arrive while admission is parked in recv_timeout.
            // Never start an admitted request without checking it again.
            checkpoint(OwnerCheckpoint::BeforeHandling)?;
            if control_enabled {
                gate.checkpoint()?;
            }
            match event {
                control::Event::Incoming(stream) => {
                    let guard = admission
                        .try_admit()
                        .context("HTTP owner admission is closed")?;
                    let subscriber = trace.as_ref().map(http::TraceLog::subscriber);
                    match transport::Connection::start(stream, backend, subscriber, guard) {
                        Ok(started) => connection = Some(started),
                        Err(cause) => {
                            tracing::warn!("serve: HTTP connection setup failed: {cause:#}");
                            std::thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                    }
                }
                control::Event::Prepared(prepared) => connection = Some(prepared),
                control::Event::Native(job) => {
                    if let Err(cause) = job.run(backend) {
                        tracing::error!("native job publication failed: {cause:#}");
                    }
                }
            }
            drain_completions(&mut activity, backend);
        }
        Ok(())
    })();

    admission.close();
    gate.close();
    ready.store(false, Ordering::Release);
    stopping.store(true, Ordering::Release);
    drop(receiver);
    let worker_result = connection.map_or(Ok(()), transport::Connection::stop_and_join);
    let acceptor_result = acceptor.join();
    drain_completions(&mut activity, backend);
    // Stop accepting before the bounded durable flush, so clients see a
    // closed port rather than a stalled server during shutdown.
    let settlement = shutdown_owner(backend, &activity);
    acceptor_result.map_err(|_| anyhow::anyhow!("HTTP acceptor panicked"))??;
    worker_result?;
    settlement?;
    result
}

fn shutdown_owner(
    backend: &mut dyn http::GenerationBackend,
    activity: &owner_activity::OwnerActivity,
) -> Result<()> {
    let settled = activity.is_settled();
    backend.set_control_memory_reserve(0);
    backend.shutdown();
    ensure!(settled, "HTTP owner activity did not settle");
    Ok(())
}

#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod signal_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_flags_are_refused_where_no_tier_exists_and_off_is_always_fine() {
        use crate::family_profile::profile;
        use durable::{DEFAULT_MIN_TOKENS, DurableDir, DurableSnapshotConfig};
        let config = |dir, max_mib, min_tokens| DurableSnapshotConfig {
            dir,
            max_mib,
            min_tokens,
            ..DurableSnapshotConfig::off()
        };
        let defaulted = config(DurableDir::Default, None, DEFAULT_MIN_TOKENS);
        let explicit = [
            config(DurableDir::Path("/tmp/x".into()), None, DEFAULT_MIN_TOKENS),
            config(DurableDir::Default, Some(512), DEFAULT_MIN_TOKENS),
            config(DurableDir::Default, None, 64),
        ];
        let off = [
            config(DurableDir::Off, None, DEFAULT_MIN_TOKENS),
            config(DurableDir::Off, Some(512), 64),
            config(DurableDir::Default, Some(0), DEFAULT_MIN_TOKENS),
        ];
        for family in ModelFamily::ALL {
            let family = profile(*family);
            let has_tier =
                family.serve_warmth == crate::family_profile::ServeWarmth::SnapshotsDurable;
            let policy = qwen_llm::snapshot_policy::SnapshotPolicyConfig::default();
            assert!(check_warmth_flags(family, &defaulted, None, &policy).is_ok());
            assert_eq!(
                check_warmth_flags(family, &defaulted, Some(1024), &policy).is_ok(),
                family.serve_warmth != crate::family_profile::ServeWarmth::LiveSession,
                "{}",
                family.display
            );
            for config in &explicit {
                assert_eq!(
                    check_warmth_flags(family, config, None, &policy).is_ok(),
                    has_tier,
                    "{} {config:?}",
                    family.display
                );
            }
            for config in &off {
                assert!(check_warmth_flags(family, config, None, &policy).is_ok());
            }
        }
    }

    /// Snapshot-policy settings on families that keep no snapshots are
    /// refused rather than ignored; omission, the defaults and the values
    /// that switch a feature off are accepted. Idle publication is refused
    /// where the durable tier has none (DeepSeek V4), unless the tier is off.
    #[test]
    fn unsupported_snapshot_policy_settings_are_refused_not_ignored() {
        use crate::family_profile::{ServeWarmth, profile};
        use durable::{DEFAULT_IDLE_PUBLISH_SECS, DurableDir, DurableSnapshotConfig};
        use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
        let defaults = SnapshotPolicyConfig::default();
        let durable = |dir, idle_publish_secs| DurableSnapshotConfig {
            dir,
            max_mib: None,
            idle_publish_secs,
            ..DurableSnapshotConfig::off()
        };
        let default_tier = durable(DurableDir::Default, DEFAULT_IDLE_PUBLISH_SECS);
        let secs = Duration::from_secs;
        let accepted = [
            (None, defaults),
            (Some(0), defaults),
            (
                None,
                SnapshotPolicyConfig {
                    idle_ttl: Duration::ZERO,
                    max_age: Duration::ZERO,
                    ..defaults
                },
            ),
        ];
        let refused_on_live = [
            (Some(512), defaults),
            (
                None,
                SnapshotPolicyConfig {
                    idle_ttl: secs(60),
                    ..defaults
                },
            ),
            (
                None,
                SnapshotPolicyConfig {
                    max_age: secs(60),
                    ..defaults
                },
            ),
            (
                None,
                SnapshotPolicyConfig {
                    half_life: Duration::ZERO,
                    ..defaults
                },
            ),
        ];
        for family in ModelFamily::ALL {
            let family = profile(*family);
            assert!(
                !family.durable_idle_publish
                    || family.serve_warmth == ServeWarmth::SnapshotsDurable,
                "{}: idle publication without a durable tier",
                family.display
            );
            let live = family.serve_warmth == ServeWarmth::LiveSession;
            for (mib, policy) in &accepted {
                assert!(
                    check_warmth_flags(family, &default_tier, *mib, policy).is_ok(),
                    "{} {mib:?} {policy:?}",
                    family.display
                );
            }
            for (mib, policy) in &refused_on_live {
                let error = check_warmth_flags(family, &default_tier, *mib, policy);
                assert_eq!(
                    error.is_err(),
                    live,
                    "{} {mib:?} {policy:?}",
                    family.display
                );
                if let Err(error) = error {
                    assert!(error.to_string().contains("keeps no snapshots"), "{error}");
                }
            }
            if family.serve_warmth == ServeWarmth::SnapshotsDurable {
                let publish = durable(DurableDir::Default, DEFAULT_IDLE_PUBLISH_SECS * 3);
                assert_eq!(
                    check_warmth_flags(family, &publish, None, &defaults).is_ok(),
                    family.durable_idle_publish,
                    "{}",
                    family.display
                );
                for config in [
                    durable(DurableDir::Default, 0),
                    durable(DurableDir::Off, DEFAULT_IDLE_PUBLISH_SECS * 3),
                ] {
                    assert!(check_warmth_flags(family, &config, None, &defaults).is_ok());
                }
            }
        }
    }

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

    fn serve_invocation(model: std::path::PathBuf, addr: String) -> crate::cli::ServeInvocation {
        crate::cli::ServeInvocation {
            model,
            addr,
            max_tokens: None,
            max_context_tokens: None,
            snapshot_cache_mib: None,
            snapshot_policy: SnapshotPolicyConfig::default(),
            durable: durable::DurableSnapshotConfig::off(),
            drafter: None,
            trace_sse: None,
            lens_data_dir: None,
            lens_config: None,
            web_root: None,
            lens_allowed_origin: Vec::new(),
            template_style: items::TemplateStyle::House,
            idle_residency_secs: None,
        }
    }

    #[test]
    fn preparation_refusal_precedes_listener_bind() {
        let fixture = crate::linear_transport::tests::fixture("serve-prebind-refusal", 2, 19);
        let model = fixture.0.join("model.gguf");
        crate::linear_transport::cpu_fixture::write_cpu_gguf(
            &model, "qwen35", 2, "cpu-test", false,
        );
        let held = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = held.local_addr().unwrap();
        let mut invocation = serve_invocation(model, address.to_string());
        invocation.template_style = items::TemplateStyle::Upstream;
        let error = run_serve(invocation).expect_err("generic Qwen cannot use upstream style");
        assert_eq!(
            error.to_string(),
            "--template-style upstream requires an identified Qwen release; this model uses the generic ChatML contract"
        );
        drop(held);
        let rebound = TcpListener::bind(address).expect("preparation refusal must not bind");
        drop(rebound);
    }

    fn fail_before_serve(_listening: Listening) -> Result<()> {
        Err(anyhow::anyhow!("injected family-start failure"))
    }

    #[test]
    fn listening_releases_socket_when_start_fails() {
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = probe.local_addr().unwrap();
        drop(probe);
        let invocation = serve_invocation("synthetic-model.gguf".into(), address.to_string());
        let listening = Listening::open(&invocation, None).unwrap();
        let error = fail_before_serve(listening).unwrap_err();
        assert_eq!(error.to_string(), "injected family-start failure");
        let rebound = TcpListener::bind(address).expect("failed start must release listener");
        drop(rebound);
    }

    #[test]
    fn serve_family_gate_lists_backends_explicitly() {
        for family in ModelFamily::ALL {
            assert_eq!(
                supports_serve_family(Some(*family)),
                profile(*family).serve_backend,
                "{family:?}"
            );
            // Every recognised family has a backend.
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
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, Some(0), Some(2_048)).is_err()
        );
        assert!(
            fixed_session_limits(ModelFamily::MuseGlimmer, 131_072, Some(7_168), Some(0)).is_err()
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
        acceptor.join().unwrap().unwrap();
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
                |_| Err(anyhow::anyhow!("termination already requested")),
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
        acceptor.join().unwrap().unwrap();
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
        acceptor.join().unwrap().unwrap();
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
