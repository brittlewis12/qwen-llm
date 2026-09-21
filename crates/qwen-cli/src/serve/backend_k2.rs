//! Borrowed resident model; each request owns and drops its entire fresh session.
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::OutputProtocol;
use super::render_k2;
use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::k2_horizon_runtime::{K2LoadedModel, K2PreparedArtifact, K2RuntimePlan};
use qwen_llm::metal::host_page_size_bytes;
use qwen_llm::sampling::Sampler;
use qwen_llm::tokenizer::NativeTokenizer;

pub(super) struct Prepared {
    tokenizer: NativeTokenizer,
    pub capacity: usize,
    default_max: usize,
    chat_profile: Option<render_k2::ChatCapability>,
}

fn limits(
    context: u32,
    capacity: Option<usize>,
    maximum: Option<usize>,
    snapshots: u64,
    drafter: bool,
) -> Result<(usize, usize)> {
    ensure!(!drafter, "K2 serve does not support a drafter");
    ensure!(
        snapshots == 0,
        "K2 serve requires --snapshot-cache-mib 0; snapshots are unsupported"
    );
    let capacity = capacity
        .context("K2 serve requires explicit --max-context-tokens for resident memory planning")?;
    let maximum = maximum.context("K2 serve requires explicit --max-tokens")?;
    ensure!(
        capacity > 0 && capacity <= context as usize,
        "K2 serve capacity must fit declared context {context}"
    );
    ensure!(
        maximum > 0 && maximum <= capacity,
        "K2 serve --max-tokens must be in 1..=capacity"
    );
    Ok((capacity, maximum))
}

impl Prepared {
    pub(super) fn new(gguf: &GgufFile, invocation: &crate::cli::ServeInvocation) -> Result<Self> {
        let config = qwen_llm::k2_horizon::K2HorizonConfig::from_gguf(gguf)?;
        let (capacity, default_max) = limits(
            config.context_length,
            invocation.max_context_tokens,
            invocation.max_tokens,
            invocation.snapshot_cache_mib,
            invocation.drafter.is_some(),
        )?;
        let artifact = K2PreparedArtifact::inspect(gguf)?;
        artifact.generation_stops()?;
        K2RuntimePlan::inspect(
            gguf,
            u32::try_from(capacity)?,
            host_page_size_bytes()?,
            usize::MAX,
        )?;
        let tokenizer = artifact.into_tokenizer();
        let chat_profile = render_k2::ChatCapability::verify(gguf);
        // A termination request must not become a raw-only server fallback.
        crate::shutdown::checkpoint()?;
        Ok(Self {
            tokenizer,
            capacity,
            default_max,
            chat_profile: match chat_profile {
                Ok(profile) => Some(profile),
                Err(error) => {
                    eprintln!("K2 serve chat unavailable; raw requests remain supported: {error}");
                    None
                }
            },
        })
    }
}

pub(super) struct K2Backend<'model, 'ctx> {
    model: &'model K2LoadedModel<'ctx>,
    prepared: Prepared,
    model_id: String,
}

impl<'model, 'ctx> K2Backend<'model, 'ctx> {
    pub(super) fn new(
        model: &'model K2LoadedModel<'ctx>,
        prepared: Prepared,
        model_id: String,
    ) -> Self {
        Self {
            model,
            prepared,
            model_id,
        }
    }
}

impl GenerationBackend for K2Backend<'_, '_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn parse_request(&self, body: &serde_json::Value) -> Result<ServeRequest, ServeError> {
        render_k2::parse_with_profile(body, self.prepared.chat_profile.as_ref())
    }
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        render_k2::normalize_with_profile(
            request,
            self.prepared.default_max,
            self.prepared.capacity,
            self.prepared.chat_profile.as_ref(),
        )
    }
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_k2::render_with_profile(request, self.prepared.chat_profile.as_ref())
    }
    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        match (&self.prepared.chat_profile, &request.k2_chat) {
            (Some(_), Some(chat)) => OutputProtocol::K2Chat {
                effort: chat.effort,
            },
            _ => OutputProtocol::RawText,
        }
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        render_k2::render_with_profile(request, self.prepared.chat_profile.as_ref())?;
        let stops: &[i32] = if request.k2_chat.is_some() {
            &qwen_llm::k2_horizon_chat::CHAT_STOPS
        } else {
            &[1]
        };
        let maximum = request
            .max_output_tokens
            .unwrap_or(self.prepared.default_max);
        let mut sampler = Sampler::new(render_k2::sampling(request))
            .map_err(|e| ServeError::invalid_request(None, format!("K2 sampler: {e}")))?;
        let tokens = decode_loop::encode_checked(
            &self.prepared.tokenizer,
            prompt,
            request.k2_add_special_tokens.unwrap_or(true),
            self.model.config().vocab_size,
            "K2",
        )?;
        let required =
            decode_loop::required_forwards("K2", tokens.len(), maximum, self.prepared.capacity)?;
        sink.tick().map_err(BackendFailure::Aborted)?;
        let prefill_t0 = std::time::Instant::now();
        let mut session = self
            .model
            .create_session(0)
            .map_err(|e| ServeError::server_error(format!("K2 fresh session: {e}")))?;
        let mut logits = Vec::new();
        // Genuine per-token cancellation boundaries. Every append is synchronous;
        // no staged prefix survives an aborted request's session drop.
        for (index, token) in tokens.iter().enumerate() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            if index + 1 == tokens.len() {
                logits = session
                    .append(&[*token])
                    .map_err(|e| ServeError::server_error(format!("K2 prefill: {e}")))?;
            } else {
                session
                    .advance(&[*token])
                    .map_err(|e| ServeError::server_error(format!("K2 prefill: {e}")))?;
            }
        }
        sink.tick().map_err(BackendFailure::Aborted)?;
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
        let generation = decode_loop::decode_serial(
            decode_loop::DecodeRequest {
                family: "K2",
                logits,
                max_tokens: maximum,
                stop_tokens: stops,
                vocab_size: self.model.config().vocab_size,
            },
            &mut sampler,
            &self.prepared.tokenizer,
            sink,
            |token| Ok(session.append(&[token])?),
        )?;
        if session.committed_len() as usize != tokens.len() + generation.transitions {
            return Err(ServeError::server_error("K2 consumed-prefix accounting mismatch").into());
        }
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=k2_horizon prefill_ms={prefill_ms:.1} prefill_tokens={} decode_ms={:.1} required_forwards={required} capacity={} transitions={}",
            tokens.len(),
            generation.wall_ms,
            self.prepared.capacity,
            generation.transitions,
        );
        Ok(super::outcome::finish_generation(
            tokens.len(),
            &generation,
            0,
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests;
