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
pub(crate) mod backend_muse;
pub(crate) mod events;
pub(crate) mod http;
pub(crate) mod output_partition;
pub(crate) mod partition;
pub(crate) mod partition_muse;
pub(crate) mod render_ds4;
pub(crate) mod render_muse;
pub(crate) mod utf8;

pub(crate) use crate::open_responses::{items, render, tool_parse};

use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalMemorySignals;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::Runtime;
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
pub(crate) const DEFAULT_SNAPSHOT_CACHE_MIB: u64 = 4096;
pub(crate) const DEFAULT_SNAPSHOT_CACHE_BYTES: u64 = DEFAULT_SNAPSHOT_CACHE_MIB * 1024 * 1024;
const SNAPSHOT_CAPTURE_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SnapshotCaptureDenial {
    EstimateOverflow,
    InvalidMetalSignal,
    MetalHeadroom,
    ProcessSignalUnavailable,
    ProcessHeadroom,
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
            return Err(SnapshotCaptureDenial::ProcessHeadroom);
        }
        Some(_) => {}
        None => return Err(SnapshotCaptureDenial::ProcessSignalUnavailable),
    }
    Ok(())
}

fn snapshot_cache_bytes(mib: u64) -> Result<u64> {
    mib.checked_mul(1024 * 1024)
        .context("--snapshot-cache-mib byte conversion overflow")
}

fn supports_serve_family(family: Option<ModelFamily>) -> bool {
    matches!(
        family,
        Some(
            ModelFamily::Qwen35
                | ModelFamily::Qwen35Moe
                | ModelFamily::DeepSeek4
                | ModelFamily::MuseGlimmer
        )
    )
}

fn muse_serve_limits(
    model_context: usize,
    has_drafter: bool,
    max_context_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> Result<(usize, usize)> {
    ensure!(
        !has_drafter,
        "--drafter is not supported for Muse Glimmer serve"
    );
    let context_limit = max_context_tokens.context(
        "Muse Glimmer serve requires --max-context-tokens because its resident session capacity is fixed at startup",
    )?;
    ensure!(
        context_limit <= model_context,
        "Muse Glimmer --max-context-tokens {context_limit} exceeds model context {model_context}",
    );
    let default_max_tokens = max_tokens.context(
        "Muse Glimmer serve requires explicit --max-tokens; the generic 65536-token default exceeds its reference session capacity",
    )?;
    ensure!(
        default_max_tokens <= context_limit,
        "Muse Glimmer --max-tokens {default_max_tokens} exceeds --max-context-tokens {context_limit}"
    );
    Ok((context_limit, default_max_tokens))
}

/// `qwen serve` entry: resident model, serial accept loop.
pub(crate) fn run_serve(invocation: crate::cli::ServeInvocation) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let snapshot_cache_bytes = snapshot_cache_bytes(invocation.snapshot_cache_mib)?;
    let gguf = GgufFile::open(&invocation.model)
        .with_context(|| format!("open model {}", invocation.model.display()))?;
    let family = ModelFamily::detect(&gguf);
    let muse_glimmer = family == Some(ModelFamily::MuseGlimmer);
    ensure!(
        supports_serve_family(family),
        "qwen serve supports Qwen3.5/3.6-family, DeepSeek V4, and Muse Glimmer models (docs/SERVE.md)"
    );
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

    if muse_glimmer {
        let config = qwen_llm::muse_glimmer::MuseGlimmerConfig::from_gguf(&gguf)
            .context("bind Muse Glimmer serve contract")?;
        let (context_limit, default_max_tokens) = muse_serve_limits(
            config.context_length as usize,
            invocation.drafter.is_some(),
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?;
        crate::shutdown::checkpoint()?;
        let ctx = qwen_llm::metal::MetalContext::new().context("initialize Metal context")?;
        let load_t0 = Instant::now();
        let mut backend = backend_muse::MuseGlimmerBackend::new(
            ctx,
            gguf,
            &invocation.model,
            model_id.clone(),
            default_max_tokens,
            context_limit,
        )?;
        let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
        tracing::info!(target: "qwen_diag", "serve limits: family=muse_glimmer max_context_tokens={} default_max_tokens={} snapshot_cache_bytes=0", context_limit, default_max_tokens);
        crate::shutdown::checkpoint()?;
        return accept_loop(
            &invocation.addr,
            &model_id,
            load_ms,
            &mut backend,
            &mut trace,
        );
    }

    if family == Some(ModelFamily::DeepSeek4) {
        ensure!(
            invocation.drafter.is_none(),
            "--drafter is not supported for DeepSeek V4 serve"
        );
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
            snapshot_cache_bytes,
        )?;
        tracing::info!(target: "qwen_diag", "serve limits: family=deepseek_v4 max_context_tokens={} snapshot_cache_bytes={}", context_limit, snapshot_cache_bytes);
        crate::shutdown::checkpoint()?;
        return accept_loop(&invocation.addr, &model_id, 0.0, &mut backend, &mut trace);
    }
    // Resolve the rendering protocol once from the loaded metadata -- the
    // same gate `qwen run` applies. Without this a Qwen3.8 model renders
    // with the generic ChatML contract (no effort instruction, no
    // preclosed history), silently diverging from upstream.
    let template =
        crate::prompt_template::serve_qwen_template(family.expect("family checked above"), &gguf);
    let no_thinking_supported =
        crate::supports_qwen_no_thinking_prompt(family.expect("family checked above"), &gguf);
    drop(gguf);
    crate::shutdown::checkpoint()?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let load_t0 = Instant::now();
    let loaded = runtime
        .load_model_with_config(
            &invocation.model,
            qwen_llm::runtime::LoadedModelConfig {
                prefix_cache_max_bytes: snapshot_cache_bytes,
                ..qwen_llm::runtime::LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", invocation.model.display()))?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    let mut backend = backend::EngineBackend::new(
        loaded,
        model_id.clone(),
        invocation.max_tokens.unwrap_or(DEFAULT_SERVE_MAX_TOKENS),
        invocation.max_context_tokens,
        invocation.drafter.as_deref(),
        template,
        no_thinking_supported,
    )?;
    tracing::info!(target: "qwen_diag", "serve limits: family=qwen max_context_tokens={} context_source={} snapshot_cache_bytes={}", invocation.max_context_tokens.unwrap_or(DEFAULT_SERVE_MAX_CONTEXT_TOKENS), if invocation.max_context_tokens.is_some() { "explicit" } else { "default_hard_ceiling" }, snapshot_cache_bytes);
    crate::shutdown::checkpoint()?;

    accept_loop(
        &invocation.addr,
        &model_id,
        load_ms,
        &mut backend,
        &mut trace,
    )
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
                        std::thread::sleep(ACCEPT_POLL_INTERVAL);
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
    addr: &str,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
) -> Result<()> {
    accept_loop_with_checkpoint(addr, model_id, load_ms, backend, trace, || {
        crate::shutdown::checkpoint()
    })
}

fn accept_loop_with_checkpoint(
    addr: &str,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<()> {
    let listener = bind_loopback(addr)?;
    let local_addr = listener
        .local_addr()
        .map_or_else(|_| addr.to_owned(), |addr| addr.to_string());
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
                Err(RecvTimeoutError::Timeout) => continue,
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
    if acceptor.join().is_err() {
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
    fn serve_family_gate_rejects_flash_next_until_it_has_a_backend() {
        assert!(supports_serve_family(Some(ModelFamily::Qwen35)));
        assert!(supports_serve_family(Some(ModelFamily::Qwen35Moe)));
        assert!(supports_serve_family(Some(ModelFamily::DeepSeek4)));
        assert!(!supports_serve_family(Some(ModelFamily::Qwen4Exp)));
        assert!(!supports_serve_family(None));
    }

    #[test]
    fn muse_limits_require_explicit_bounded_capacity_and_output_default() {
        assert_eq!(
            muse_serve_limits(131_072, false, Some(7_168), Some(2_048)).unwrap(),
            (7168, 2048)
        );
        assert_eq!(
            muse_serve_limits(131_072, false, Some(131_072), Some(16_384)).unwrap(),
            (131_072, 16_384)
        );
        assert!(muse_serve_limits(131_072, true, Some(7_168), Some(2_048)).is_err());
        assert!(muse_serve_limits(131_072, false, None, Some(2_048)).is_err());
        assert!(muse_serve_limits(131_072, false, Some(7_168), None).is_err());
        assert!(muse_serve_limits(131_072, false, Some(131_073), Some(2_048)).is_err());
        assert!(muse_serve_limits(131_072, false, Some(1_024), Some(2_048)).is_err());
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
                "127.0.0.1:0",
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
    fn default_snapshot_budget_covers_a3b_not_dense_snapshot() {
        let configured = snapshot_cache_bytes(DEFAULT_SNAPSHOT_CACHE_MIB).unwrap();
        assert!(configured > 2_700_000_000);
        assert!(configured < 12_000_000_000);
    }
}
