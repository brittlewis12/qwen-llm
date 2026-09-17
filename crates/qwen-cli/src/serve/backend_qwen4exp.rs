//! Resident Qwen3.8-Flash-Next [`GenerationBackend`].
//!
//! The text session's workspace is allocated once at load for a fixed
//! forward limit; each request takes it as a runner and hands it back reset
//! (`Qwen4ExpLoadedModel::restore_workspace`), so no per-request allocation
//! and no history carries across requests.

use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{QwenTemplate, ServeError, ServeRequest};
use super::output_partition::{OutputProtocol, ToolGrammar};
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLoadedModel, Qwen4ExpRuntimeError, Qwen4ExpSessionCapacity,
};
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

pub(crate) struct FlashNextBackend {
    ctx: MetalContext,
    /// Serve holds the mapping for the process lifetime; the loaded model's
    /// CPU-resident PLE table borrows it.
    gguf: &'static GgufFile,
    tokenizer: Tokenizer,
    loaded: Qwen4ExpLoadedModel<'static>,
    model_id: String,
    default_max_tokens: usize,
    forward_limit: usize,
    vocab_size: u32,
    stop_tokens: Vec<i32>,
}

impl FlashNextBackend {
    pub(crate) fn new(
        ctx: MetalContext,
        gguf: &'static GgufFile,
        model_id: String,
        default_max_tokens: usize,
        context_limit: usize,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(gguf).context("load Qwen3.8-Flash-Next tokenizer")?;
        let config =
            Qwen4ExpConfig::from_gguf(gguf).context("bind Qwen3.8-Flash-Next request geometry")?;
        anyhow::ensure!(
            config == Qwen4ExpConfig::flash_next_reference(),
            "Qwen3.8-Flash-Next runtime requires the released architecture contract"
        );
        let vocab_size = tokenizer.n_vocab();
        anyhow::ensure!(
            vocab_size == config.vocab_size,
            "Qwen3.8-Flash-Next tokenizer vocabulary {vocab_size} differs from model vocabulary {}",
            config.vocab_size
        );
        let stop_tokens = gguf
            .stop_token_ids()
            .context("load producer-declared Qwen3.8-Flash-Next stop tokens")?;
        crate::qwen4exp::validate_qwen4exp_stop_tokens(&stop_tokens, vocab_size)?;
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, context_limit)
            .context("derive Qwen3.8-Flash-Next resident session capacity")?;
        let decode_options = crate::qwen4exp::qwen4exp_decode_options_from_env()?;
        // Packed prefill sized to the whole context; when that allocation
        // is refused, prompts prefill token by token as on the run lane.
        let (loaded, packed_fallback) = match Qwen4ExpLoadedModel::load_with_decode_options(
            &ctx,
            gguf,
            capacity,
            Some(context_limit),
            decode_options,
        ) {
            Ok(loaded) => (loaded, None),
            Err(packed_error) => (
                Qwen4ExpLoadedModel::load_with_decode_options(
                    &ctx,
                    gguf,
                    capacity,
                    None,
                    decode_options,
                )
                .with_context(|| {
                    format!(
                        "load Qwen3.8-Flash-Next serve session (packed prefill refused: {packed_error})"
                    )
                })?,
                Some(packed_error.to_string()),
            ),
        };
        tracing::info!(
            target: "qwen_diag",
            "serve: qwen4exp resident forward_limit={} qsa_physical_capacity={} packed_prefill_capacity={:?} packed_fallback={:?} guarded_topk={} hc_up_mix={}",
            capacity.forward_limit(),
            capacity.qsa_physical_capacity(),
            loaded.packed_prefill_capacity(),
            packed_fallback,
            loaded.guarded_topk_enabled(),
            loaded.hc_up_mix_enabled(),
        );
        Ok(Self {
            ctx,
            gguf,
            tokenizer,
            loaded,
            model_id,
            default_max_tokens,
            forward_limit: capacity.forward_limit(),
            vocab_size,
            stop_tokens,
        })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, ServeError> {
        self.tokenizer
            .encode(prompt, false)
            .map_err(|error| {
                ServeError::server_error(format!("tokenize Qwen3.8-Flash-Next prompt: {error}"))
            })?
            .into_iter()
            .enumerate()
            .map(|(index, token)| {
                crate::checked_token_id(token, self.vocab_size, &format!("prompt[{index}]"))
                    .map_err(|error| ServeError::server_error(error.to_string()))
            })
            .collect()
    }
}

impl GenerationBackend for FlashNextBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Bind once through the family table so the bound request is what
    /// renders, selects the output protocol, and echoes.
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        *request = crate::open_responses::bind_qwen_request(request, QwenTemplate::Qwen38, true)?;
        Ok(())
    }

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: super::render::qwen_generation(request)
                == super::render::QwenGeneration::PreOpen,
            parse_tools: true,
            tool_grammar: ToolGrammar::QwenXml,
        }
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        Ok(super::render::render_qwen_serve_prompt(request))
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let max_tokens = request.max_output_tokens.unwrap_or(self.default_max_tokens);
        let mut sampler = super::backend::request_sampler(request)?;
        let tokenize_t0 = Instant::now();
        let prompt_ids = self.encode(prompt)?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        if prompt_ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "Qwen3.8-Flash-Next prompt tokenized to zero tokens",
            )
            .into());
        }
        let required = super::backend_muse::required_forwards(
            "Qwen3.8-Flash-Next",
            prompt_ids.len(),
            max_tokens,
            self.forward_limit,
        )?;
        let stop_tokens = self.stop_tokens.clone();
        let tokenizer = &self.tokenizer;
        let vocab_size = self.vocab_size;
        let mut runner = self.loaded.create_runner(&self.ctx).map_err(|error| {
            ServeError::server_error(format!("bind Qwen3.8-Flash-Next runner: {error}"))
        })?;

        let prefill_t0 = Instant::now();
        let mut checkpoint_abort: Option<io::Error> = None;
        let prefill = runner.prefill_with_command_checkpoint(&prompt_ids, || {
            sink.tick().map_err(|error| {
                checkpoint_abort = Some(error);
                Qwen4ExpRuntimeError::Checkpoint(
                    "transport aborted during Qwen3.8-Flash-Next prefill".into(),
                )
            })
        });
        let outcome = match prefill {
            Err(error) => Err(match checkpoint_abort {
                Some(error) => BackendFailure::Aborted(error),
                None => {
                    ServeError::server_error(format!("prefill Qwen3.8-Flash-Next prompt: {error}"))
                        .into()
                }
            }),
            Ok(logits) => {
                let logits = logits.to_vec();
                let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
                // Canonical serial loop (same shape as DeepSeek and Muse
                // serve): the stop token is counted but never written;
                // `tick` runs on every emitted token because `piece` only
                // touches the socket when the partition emits.
                let mut abort: Option<io::Error> = None;
                let generation = {
                    let abort = &mut abort;
                    crate::generate_serial(
                        logits,
                        max_tokens,
                        &stop_tokens,
                        &mut sampler,
                        |token| {
                            let bytes =
                                tokenizer.try_decode_piece_bytes_exact(token).with_context(
                                    || format!("decode Qwen3.8-Flash-Next token {token}"),
                                )?;
                            sink.piece(bytes)
                                .and_then(|()| sink.tick())
                                .map_err(|error| {
                                    *abort = Some(error);
                                    anyhow::anyhow!("client disconnected during decode")
                                })
                        },
                        |token| {
                            let token = crate::checked_token_id(token, vocab_size, "generated")?;
                            Ok(runner
                                .forward_token(token)
                                .context("forward Qwen3.8-Flash-Next token")?
                                .to_vec())
                        },
                    )
                };
                match generation {
                    Ok(generation) => {
                        tracing::info!(
                            target: "qwen_diag",
                            "serve phases: family=qwen4exp tokenize_ms={tokenize_ms:.1} prefill_ms={prefill_ms:.1} prefill_tokens={} decode_ms={:.1} required_forwards={required} forward_limit={} transitions={}",
                            prompt_ids.len(),
                            generation.wall_ms,
                            self.forward_limit,
                            generation.transitions,
                        );
                        Ok(super::outcome::finish_generation(
                            prompt_ids.len(),
                            &generation,
                            0,
                            0.0,
                        ))
                    }
                    Err(error) => Err(match abort {
                        Some(io_error) => BackendFailure::Aborted(io_error),
                        None => ServeError::server_error(format!("decode: {error:#}")).into(),
                    }),
                }
            }
        };
        // The workspace goes back reset whatever the outcome; a failed reset
        // leaves the backend without a session, which the next request
        // reports as a server error rather than serving stale state.
        let workspace = runner.into_workspace();
        self.loaded.restore_workspace(workspace).map_err(|error| {
            ServeError::server_error(format!("reset Qwen3.8-Flash-Next session: {error}"))
        })?;
        outcome
    }
}
