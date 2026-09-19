//! Borrowed resident model; each request owns and drops its entire fresh session.
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::OutputProtocol;
use super::render_k2;
use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::k2_horizon_runtime::GUARDED_APPLICATION_FORWARD_CEILING as MAX_FORWARDS;
use qwen_llm::k2_horizon_runtime::{K2LoadedModel, K2RuntimePlan};
use qwen_llm::metal::host_page_size_bytes;
use qwen_llm::sampling::Sampler;
use qwen_llm::tokenizer::NativeTokenizer;

pub(super) struct Prepared {
    tokenizer: NativeTokenizer,
    pub capacity: usize,
    default_max: usize,
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
    let capacity = capacity.with_context(|| {
        format!("K2 serve requires explicit --max-context-tokens in 1..={MAX_FORWARDS}")
    })?;
    let maximum = maximum.context("K2 serve requires explicit --max-tokens")?;
    ensure!(
        capacity > 0 && capacity <= MAX_FORWARDS && capacity <= context as usize,
        "K2 serve capacity must fit 1..={MAX_FORWARDS} and declared context"
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
        ensure!(
            gguf.stop_token_ids()? == [1],
            "K2 serve requires exact EOS 1 only"
        );
        K2RuntimePlan::inspect(
            gguf,
            capacity as u32,
            host_page_size_bytes()?,
            8 * 1024 * 1024 * 1024,
        )?;
        let tokenizer = NativeTokenizer::from_gguf(gguf)?;
        Ok(Self {
            tokenizer,
            capacity,
            default_max,
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
        render_k2::parse_request(body)
    }
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        render_k2::normalize(request, self.prepared.default_max, self.prepared.capacity)
    }
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_k2::render(request)
    }
    fn output_protocol(&self, _: &ServeRequest) -> OutputProtocol {
        OutputProtocol::RawText
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let maximum = request
            .max_output_tokens
            .unwrap_or(self.prepared.default_max);
        let mut sampler = Sampler::new(render_k2::sampling(request))
            .map_err(|e| ServeError::invalid_request(None, format!("K2 sampler: {e}")))?;
        let ids = self
            .prepared
            .tokenizer
            .encode(prompt, request.k2_add_special_tokens.unwrap_or(true))
            .map_err(|e| {
                ServeError::invalid_request(Some("input"), format!("K2 tokenization: {e}"))
            })?;
        if ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "K2 input tokenized to no tokens",
            )
            .into());
        }
        let tokens = ids
            .iter()
            .map(|&id| crate::checked_token_id(id, self.model.config().vocab_size, "K2 prompt"))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| ServeError::invalid_request(Some("input"), e.to_string()))?;
        super::backend_muse::required_forwards(
            "K2",
            tokens.len(),
            maximum,
            self.prepared.capacity,
        )?;
        sink.tick().map_err(BackendFailure::Aborted)?;
        let mut session = self
            .model
            .create_session(0)
            .map_err(|e| ServeError::server_error(format!("K2 fresh session: {e}")))?;
        let mut logits = Vec::new();
        // Genuine per-token cancellation boundaries. Every append is synchronous;
        // no staged prefix survives an aborted request's session drop.
        for token in &tokens {
            sink.tick().map_err(BackendFailure::Aborted)?;
            logits = session
                .append(&[*token])
                .map_err(|e| ServeError::server_error(format!("K2 prefill: {e}")))?;
        }
        sink.tick().map_err(BackendFailure::Aborted)?;
        let mut abort = None;
        let generation = crate::generate_serial(
            logits,
            maximum,
            &[1],
            &mut sampler,
            |token| {
                let bytes = self
                    .prepared
                    .tokenizer
                    .try_decode_piece_bytes_exact(token)?;
                sink.piece(bytes)
                    .and_then(|()| sink.tick())
                    .map_err(|error| {
                        abort = Some(error);
                        anyhow::anyhow!("K2 transport aborted")
                    })
            },
            |token| {
                let id =
                    crate::checked_token_id(token, self.model.config().vocab_size, "K2 generated")?;
                Ok(session.append(&[id])?)
            },
        );
        let generation = generation.map_err(|error| match abort {
            Some(error) => BackendFailure::Aborted(error),
            None => ServeError::server_error(format!("K2 decode: {error:#}")).into(),
        })?;
        if session.committed_len() as usize != tokens.len() + generation.transitions {
            return Err(ServeError::server_error("K2 consumed-prefix accounting mismatch").into());
        }
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
