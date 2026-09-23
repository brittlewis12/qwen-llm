//! Borrowed resident model with one live session kept across requests. Full
//! attention truncates at any token, so multi-turn reuse is the longest common
//! prefix of the consumed history, rewound in O(1); no snapshots.
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::OutputProtocol;
use super::render_k2;
use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::k2_horizon_runtime::{K2LoadedModel, K2PreparedArtifact, K2RuntimePlan, K2Session};
use qwen_llm::metal::host_page_size_bytes;
use qwen_llm::sampling::Sampler;
use qwen_llm::tokenizer::NativeTokenizer;

pub(super) struct Prepared {
    tokenizer: NativeTokenizer,
    pub capacity: usize,
    default_max: usize,
    chat_profile: Option<render_k2::ChatCapability>,
    max_piece_bytes: usize,
}

/// Default-on rollback lever for live-session prefix reuse.
const PREFIX_REUSE_ENV: &str = "QWEN_K2_PREFIX_REUSE";
/// Prompt tokens per synchronous append; transport ticks (cancellation and
/// heartbeat) fall between spans. Packed Q8 splits each span into 32-token
/// commands; serial weights run one command per token inside the span.
const PREFILL_SPAN: usize = 64;

fn limits(
    context: u32,
    capacity: Option<usize>,
    maximum: Option<usize>,
    drafter: bool,
) -> Result<(usize, usize)> {
    ensure!(!drafter, "K2 serve does not support a drafter");
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
            invocation.drafter.is_some(),
        )?;
        if invocation.snapshot_cache_mib.is_some_and(|mib| mib != 0) {
            eprintln!(
                "K2 serve ignores --snapshot-cache-mib; prefix reuse rewinds the live session"
            );
        }
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
            max_piece_bytes: tokenizer.max_decoded_piece_bytes(),
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
    session: Option<K2Session<'model, 'ctx>>,
    /// Exactly the tokens `session` has committed, or empty when unknown.
    history: Vec<u32>,
    prefix_reuse: bool,
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
            session: None,
            history: Vec::new(),
            prefix_reuse: qwen_llm::env_flag::read_default_on(PREFIX_REUSE_ENV),
        }
    }
}

impl GenerationBackend for K2Backend<'_, '_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn decode_request_json(&self, body: &[u8]) -> Result<serde_json::Value, ServeError> {
        render_k2::tools::decode_request_json(body)
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
        )?;
        if request.k2_tools.is_some() {
            render_k2::tools::byte_budget(
                request.max_output_tokens.unwrap(),
                self.prepared.max_piece_bytes,
            )?;
        }
        Ok(())
    }
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_k2::render_with_profile(request, self.prepared.chat_profile.as_ref())
    }
    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        if self.prepared.chat_profile.is_some() {
            render_k2::tools::output_protocol(request, self.prepared.max_piece_bytes)
        } else {
            OutputProtocol::RawText
        }
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        render_k2::render_with_profile(request, self.prepared.chat_profile.as_ref())?;
        let stops: &[i32] = if request.k2_chat.is_some() || request.k2_tools.is_some() {
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
        // Taken before the session moves and republished only on success, so any
        // error or abort below leaves no history and the next request rewinds to
        // zero. A late HTTP write failure after success keeps it, which is sound:
        // it records exactly what the session committed.
        let mut history = std::mem::take(&mut self.history);
        // A poisoned session (failed submitted command) is dropped, releasing its
        // one-per-model permit, before a replacement is created.
        let session = match self.session.take().filter(|s| !s.is_poisoned()) {
            Some(session) => session,
            None => self
                .model
                .create_session(0)
                .map_err(|e| ServeError::server_error(format!("K2 session: {e}")))?,
        };
        let session = self.session.insert(session);
        let reused = decode_loop::reusable_prefix(
            &history,
            &tokens,
            session.committed_len() as usize,
            self.prefix_reuse,
        );
        session
            .rewind(reused as u32)
            .map_err(|e| ServeError::server_error(format!("K2 rewind: {e}")))?;
        // Each span is one synchronous append that commits whole or poisons;
        // ticks between spans are the cancellation boundaries.
        let suffix = &tokens[reused..];
        let spans = suffix.len().div_ceil(PREFILL_SPAN);
        let mut logits = Vec::new();
        for (index, span) in suffix.chunks(PREFILL_SPAN).enumerate() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            if index + 1 == spans {
                logits = session
                    .append(span)
                    .map_err(|e| ServeError::server_error(format!("K2 prefill: {e}")))?;
            } else {
                session
                    .advance(span)
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
        history.clear();
        history.extend_from_slice(&tokens);
        history.extend(
            generation.tokens[..generation.transitions]
                .iter()
                .map(|&token| token as u32),
        );
        self.history = history;
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=k2_horizon prefill_ms={prefill_ms:.1} reused_tokens={reused} prefill_tokens={} decode_ms={:.1} required_forwards={required} capacity={} transitions={}",
            tokens.len() - reused,
            generation.wall_ms,
            self.prepared.capacity,
            generation.transitions,
        );
        Ok(super::outcome::finish_generation(
            tokens.len(),
            &generation,
            reused,
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests;
