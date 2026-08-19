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
pub(crate) mod events;
pub(crate) mod http;
pub(crate) mod items;
pub(crate) mod partition;
pub(crate) mod render;

use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::Runtime;
use std::net::TcpListener;
use std::time::Instant;

/// `qwen serve` entry: resident model, serial accept loop.
pub(crate) fn run_serve(invocation: crate::cli::ServeInvocation) -> Result<()> {
    let gguf = GgufFile::open(&invocation.model)
        .with_context(|| format!("open model {}", invocation.model.display()))?;
    let family = ModelFamily::detect(&gguf);
    ensure!(
        matches!(family, Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)),
        "qwen serve currently supports Qwen3.5/3.6-family models only (docs/SERVE.md; DeepSeek V4 serve lands in S3)"
    );
    drop(gguf);
    let model_id = invocation
        .model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("model path has no printable file stem")?
        .to_owned();

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
    )?;

    let listener =
        TcpListener::bind(&invocation.addr).with_context(|| format!("bind {}", invocation.addr))?;
    tracing::info!(
        target: "qwen_diag",
        "serve: listening on http://{} model={} load_ms={:.1} (serial; POST /v1/responses, GET /v1/models)",
        listener.local_addr().map_or_else(|_| invocation.addr.clone(), |addr| addr.to_string()),
        model_id,
        load_ms,
    );
    for stream in listener.incoming() {
        crate::shutdown::checkpoint()?;
        match stream {
            Ok(stream) => {
                if let Err(error) = http::handle_connection(&stream, &mut backend) {
                    tracing::info!(
                        target: "qwen_diag",
                        "serve: connection aborted: {error}"
                    );
                }
            }
            Err(error) => {
                tracing::warn!("serve: accept failed: {error}");
            }
        }
    }
    Ok(())
}
