//! Resident Muse Glimmer [`GenerationBackend`].

use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::OutputProtocol;
use super::render_muse;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{MuseGlimmerChatTemplateProfile, MuseGlimmerConfig};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::LlamaCppTokenizer;
use std::io;
use std::path::Path;
use std::time::Instant;

pub(crate) struct MuseGlimmerBackend {
    ctx: MetalContext,
    _gguf: GgufFile,
    tokenizer: LlamaCppTokenizer,
    loaded: MuseGlimmerLoadedModel,
    model_id: String,
    default_max_tokens: usize,
    capacity: usize,
    vocab_size: u32,
    eos_token_id: i32,
    eot_token_id: i32,
    profile: MuseGlimmerChatTemplateProfile,
}

impl MuseGlimmerBackend {
    pub(crate) fn new(
        ctx: MetalContext,
        gguf: GgufFile,
        model_path: &Path,
        model_id: String,
        default_max_tokens: usize,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let config = MuseGlimmerConfig::from_gguf(&gguf)
            .context("bind Muse Glimmer release contract for serve")?;
        let tokenizer =
            LlamaCppTokenizer::open(model_path).context("load Muse Glimmer serve tokenizer")?;
        config
            .validate_tokenizer(&tokenizer)
            .context("Muse Glimmer serve tokenizer contract")?;
        let eos_token_id = config.eos_token_id as i32;
        let eot_token_id = config.eot_token_id as i32;
        config
            .validate_stop_tokens(&gguf.stop_token_ids()?)
            .context("Muse Glimmer serve stop-token contract")?;
        let vocab_size = config.vocab_size;
        let profile = config.chat_template_profile;
        let loaded = MuseGlimmerLoadedModel::load(&ctx, &gguf, capacity)
            .context("load resident Muse Glimmer serve model")?;
        Ok(Self {
            ctx,
            _gguf: gguf,
            tokenizer,
            loaded,
            model_id,
            default_max_tokens,
            capacity,
            vocab_size,
            eos_token_id,
            eot_token_id,
            profile,
        })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, ServeError> {
        self.tokenizer
            .encode(prompt, false)
            .map_err(|error| ServeError::server_error(format!("tokenize Muse prompt: {error}")))?
            .into_iter()
            .enumerate()
            .map(|(index, token)| {
                crate::checked_token_id(token, self.vocab_size, &format!("prompt[{index}]"))
                    .map_err(|error| ServeError::server_error(error.to_string()))
            })
            .collect()
    }
}

impl GenerationBackend for MuseGlimmerBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        render_muse::normalize_request(request, self.default_max_tokens)
    }

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::MuseAtem {
            eos_token_id: self.eos_token_id,
            eot_token_id: self.eot_token_id,
            declared_tools: request
                .model_request
                .tools
                .iter()
                .map(|tool| tool.name.clone())
                .collect(),
        }
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_muse::render_muse_glimmer_serve_prompt(request, self.profile)
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let max_tokens = request.max_output_tokens.unwrap_or(self.default_max_tokens);
        let sampling = SamplingConfig {
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(64),
            top_p: request.top_p.unwrap_or(0.95),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed.unwrap_or(42),
        };
        let mut sampler = Sampler::new(sampling)
            .map_err(|error| ServeError::invalid_request(None, format!("sampling: {error}")))?;
        let tokenize_t0 = Instant::now();
        let prompt_ids = self.encode(prompt)?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        if prompt_ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "Muse prompt tokenized to zero tokens",
            )
            .into());
        }
        let required = required_forwards(prompt_ids.len(), max_tokens, self.capacity)?;
        let stop_tokens = [self.eos_token_id, self.eot_token_id];
        let tokenizer = &self.tokenizer;
        let vocab_size = self.vocab_size;
        let prefill_t0 = Instant::now();
        let mut runner = self
            .loaded
            .create_runner(&self.ctx)
            .map_err(|error| ServeError::server_error(format!("bind Muse runner: {error}")))?;
        runner
            .reset()
            .map_err(|error| ServeError::server_error(format!("reset Muse session: {error}")))?;
        let mut checkpoint_abort: Option<io::Error> = None;
        let logits = runner.prefill_with_command_checkpoint(&prompt_ids, || {
            sink.tick().map_err(|error| {
                checkpoint_abort = Some(error);
                "transport aborted during Muse prefill".into()
            })
        });
        let logits = match logits {
            Ok(logits) => logits,
            Err(error) => {
                return Err(match checkpoint_abort {
                    Some(error) => BackendFailure::Aborted(error),
                    None => {
                        ServeError::server_error(format!("prefill Muse prompt: {error}")).into()
                    }
                });
            }
        };
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

        // Canonical serial loop (same shape as DeepSeek serve): the stop
        // token is counted but never written, and a client disconnect is
        // observed through the piece write that precedes every forward.
        let mut abort: Option<io::Error> = None;
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    let bytes = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode Muse token {token}"))?;
                    sink.piece(&bytes).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let token = crate::checked_token_id(token, vocab_size, "generated")?;
                    runner.forward_token(token).context("forward Muse token")
                },
            )
        };
        let generation = match generation {
            Ok(generation) => generation,
            Err(error) => {
                return Err(match abort {
                    Some(io_error) => BackendFailure::Aborted(io_error),
                    None => ServeError::server_error(format!("decode: {error:#}")).into(),
                });
            }
        };
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=muse_glimmer tokenize_ms={tokenize_ms:.1} prefill_ms={prefill_ms:.1} decode_ms={:.1} required_forwards={required} capacity={}",
            generation.wall_ms,
            self.capacity,
        );
        Ok(super::outcome::finish_generation(prompt_ids.len(), &generation, 0, 0.0))
    }
}

fn required_forwards(
    prompt_tokens: usize,
    max_tokens: usize,
    capacity: usize,
) -> Result<usize, ServeError> {
    if max_tokens == 0 {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            "max_output_tokens must be >= 1",
        ));
    }
    let required = prompt_tokens
        .checked_add(max_tokens - 1)
        .ok_or_else(|| ServeError::invalid_request(None, "Muse forward count overflow"))?;
    if required > capacity {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            format!(
                "request needs {required} Muse forwards, beyond this server's capacity {capacity} (raise --max-context-tokens at startup)"
            ),
        ));
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_admission_counts_only_required_transitions() {
        assert_eq!(required_forwards(10, 1, 10).unwrap(), 10);
        assert_eq!(required_forwards(10, 3, 12).unwrap(), 12);
        assert!(required_forwards(10, 0, 10).is_err());
        assert!(required_forwards(10, 3, 11).is_err());
    }
}
