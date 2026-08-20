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
pub(crate) mod events;
pub(crate) mod http;
pub(crate) mod items;
pub(crate) mod partition;
pub(crate) mod render;
pub(crate) mod render_ds4;
pub(crate) mod tool_parse;
pub(crate) mod utf8;

use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::Runtime;
use std::net::TcpListener;
use std::time::Instant;

/// `qwen serve` entry: resident model, serial accept loop.
pub(crate) fn run_serve(invocation: crate::cli::ServeInvocation) -> Result<()> {
    let mut trace = invocation
        .trace_sse
        .as_deref()
        .map(http::TraceLog::open)
        .transpose()
        .context("open --trace-sse log")?;
    let gguf = GgufFile::open(&invocation.model)
        .with_context(|| format!("open model {}", invocation.model.display()))?;
    let family = ModelFamily::detect(&gguf);
    ensure!(
        matches!(
            family,
            Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::DeepSeek4)
        ),
        "qwen serve supports Qwen3.5/3.6-family and DeepSeek V4 models (docs/SERVE.md)"
    );
    let model_id = invocation
        .model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("model path has no printable file stem")?
        .to_owned();

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
        let ctx = qwen_llm::metal::MetalContext::new().context("initialize Metal context")?;
        let mut backend = backend_ds4::DeepSeekV4Backend::new(
            ctx,
            gguf,
            model_id.clone(),
            invocation.max_tokens,
            forward_limit,
            crate::DeepSeekV4MultigroupSelectorArg::Auto,
        )?;
        return accept_loop(&invocation.addr, &model_id, 0.0, &mut backend, &mut trace);
    }
    // Resolve the rendering family once, from the loaded identity — the
    // same gate `qwen run` applies. Without this a Qwen3.8 model renders
    // with the generic ChatML contract (no effort instruction, no
    // preclosed history), silently diverging from upstream.
    let template =
        if crate::validated_qwen38_prompt_model(family.expect("family checked above"), &gguf) {
            items::QwenTemplate::Qwen38
        } else {
            items::QwenTemplate::Generic
        };
    drop(gguf);
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let load_t0 = Instant::now();
    let loaded = runtime
        .load_model(&invocation.model)
        .with_context(|| format!("load model {}", invocation.model.display()))?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    let mut backend = backend::EngineBackend::new(
        loaded,
        model_id.clone(),
        invocation.max_tokens,
        invocation.max_context_tokens,
        invocation.drafter.as_deref(),
        template,
    )?;

    accept_loop(
        &invocation.addr,
        &model_id,
        load_ms,
        &mut backend,
        &mut trace,
    )
}

/// Serial accept loop shared by every family backend.
fn accept_loop(
    addr: &str,
    model_id: &str,
    load_ms: f64,
    backend: &mut dyn http::GenerationBackend,
    trace: &mut Option<http::TraceLog>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    tracing::info!(
        target: "qwen_diag",
        "serve: listening on http://{} model={} load_ms={:.1} (serial; POST /v1/responses, GET /v1/models)",
        listener
            .local_addr()
            .map_or_else(|_| addr.to_owned(), |addr| addr.to_string()),
        model_id,
        load_ms,
    );
    for stream in listener.incoming() {
        crate::shutdown::checkpoint()?;
        match stream {
            Ok(stream) => {
                if let Err(error) = http::handle_connection(&stream, backend, trace.as_mut()) {
                    tracing::info!(target: "qwen_diag", "serve: connection aborted: {error}");
                }
            }
            Err(error) => tracing::warn!("serve: accept failed: {error}"),
        }
    }
    Ok(())
}
