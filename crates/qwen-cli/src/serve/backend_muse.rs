//! Resident Muse Glimmer [`GenerationBackend`].

use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::OutputProtocol;
use super::render_muse;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{MuseGlimmerChatTemplateProfile, MuseGlimmerConfig};
use qwen_llm::muse_glimmer_runtime::{MuseGlimmerLoadedModel, MuseGlimmerRuntimeOptions};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::LlamaCppTokenizer;
use std::io;
use std::path::Path;
use std::time::Instant;

#[cfg(test)]
#[path = "muse_prefix_pilot.rs"]
mod prefix_pilot;

pub(crate) const MATRIX_PREFILL_ENV: &str = "QWEN_SERVE_MUSE_MATRIX_PREFILL";
pub(crate) const SPLIT_DECODE_ENV: &str = "QWEN_SERVE_MUSE_SPLIT_DECODE";
/// Default-on rollback lever for live-session prefix reuse.
const PREFIX_REUSE_ENV: &str = "QWEN_MUSE_PREFIX_REUSE";

pub(crate) fn read_math_options() -> anyhow::Result<MuseGlimmerRuntimeOptions> {
    Ok(MuseGlimmerRuntimeOptions {
        matrix_prefill: crate::family_options::read_math_flag(MATRIX_PREFILL_ENV)?,
        split_decode: crate::family_options::read_math_flag(SPLIT_DECODE_ENV)?,
    })
}

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
    consumed_tokens: Vec<u32>,
    prefix_reuse: bool,
    math_options: MuseGlimmerRuntimeOptions,
}

impl MuseGlimmerBackend {
    #[cfg(test)]
    pub(crate) fn new(
        ctx: MetalContext,
        gguf: GgufFile,
        model_path: &Path,
        model_id: String,
        default_max_tokens: usize,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_options(
            ctx,
            gguf,
            model_path,
            model_id,
            default_max_tokens,
            capacity,
            MuseGlimmerRuntimeOptions::REFERENCE,
        )
    }

    pub(crate) fn new_with_options(
        ctx: MetalContext,
        gguf: GgufFile,
        model_path: &Path,
        model_id: String,
        default_max_tokens: usize,
        capacity: usize,
        math_options: MuseGlimmerRuntimeOptions,
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
        let loaded = MuseGlimmerLoadedModel::load_with_options(&ctx, &gguf, capacity, math_options)
            .context("load resident Muse Glimmer serve model")?;
        let math_options = loaded.math_options();
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
            consumed_tokens: Vec::new(),
            prefix_reuse: qwen_llm::env_flag::read_default_on(PREFIX_REUSE_ENV),
            math_options,
        })
    }

    pub(crate) fn math_options(&self) -> MuseGlimmerRuntimeOptions {
        self.math_options
    }

    #[cfg(test)]
    pub(crate) fn encode(&self, prompt: &str) -> Result<Vec<u32>, ServeError> {
        decode_loop::encode_checked(&self.tokenizer, prompt, false, self.vocab_size, "Muse")
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
        let prompt_ids =
            decode_loop::encode_checked(&self.tokenizer, prompt, false, self.vocab_size, "Muse")?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        let required =
            decode_loop::required_forwards("Muse", prompt_ids.len(), max_tokens, self.capacity)?;
        let stop_tokens = [self.eos_token_id, self.eot_token_id];
        let prefill_t0 = Instant::now();
        // Taken before the session moves and republished only on success, so
        // any error or abort below leaves no history. A late HTTP write failure
        // after success keeps it, which is sound: it records exactly what the
        // session consumed.
        let mut history = std::mem::take(&mut self.consumed_tokens);
        let mut runner = self
            .loaded
            .create_runner(&self.ctx)
            .map_err(|error| ServeError::server_error(format!("bind Muse runner: {error}")))?;
        let reused_tokens = decode_loop::reusable_prefix(
            &history,
            &prompt_ids,
            runner.next_position(),
            self.prefix_reuse,
        );
        runner
            .rewind_prefix(reused_tokens)
            .map_err(|error| ServeError::server_error(format!("rewind Muse session: {error}")))?;
        let mut checkpoint_abort: Option<io::Error> = None;
        let logits = runner.prefill_with_command_checkpoint(&prompt_ids[reused_tokens..], || {
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

        let generation = decode_loop::decode_serial(
            decode_loop::DecodeRequest {
                family: "Muse",
                logits,
                max_tokens,
                stop_tokens: &stop_tokens,
                vocab_size: self.vocab_size,
            },
            &mut sampler,
            &self.tokenizer,
            sink,
            |token| Ok(runner.forward_token(token)?),
        )?;
        let consumed_end = prompt_ids.len() + generation.transitions;
        if runner.next_position() != consumed_end {
            return Err(ServeError::server_error("Muse consumed-history frontier mismatch").into());
        }
        history.clear();
        history.extend_from_slice(&prompt_ids);
        history.extend(
            generation.tokens[..generation.transitions]
                .iter()
                .map(|&token| token as u32),
        );
        self.consumed_tokens = history;
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=muse_glimmer tokenize_ms={tokenize_ms:.1} prefill_ms={prefill_ms:.1} reused_tokens={reused_tokens} prefill_tokens={} decode_ms={:.1} required_forwards={required} capacity={} transitions={} matrix_prefill={} split_decode={} planned_tiled_tokens={} planned_online_tokens={}",
            prompt_ids.len() - reused_tokens,
            generation.wall_ms,
            self.capacity,
            generation.transitions,
            self.math_options.matrix_prefill,
            self.math_options.split_decode,
            if self.math_options.matrix_prefill { (prompt_ids.len()-reused_tokens)/128*128 } else { 0 },
            if self.math_options.matrix_prefill { (prompt_ids.len()-reused_tokens)%128/16*16 } else { 0 },
        );
        Ok(super::outcome::finish_generation(
            prompt_ids.len(),
            &generation,
            reused_tokens,
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family_options::parse_math_flag;

    #[test]
    fn serve_math_switches_are_strict_and_separate_from_run() {
        assert_eq!(MATRIX_PREFILL_ENV, "QWEN_SERVE_MUSE_MATRIX_PREFILL");
        assert_eq!(SPLIT_DECODE_ENV, "QWEN_SERVE_MUSE_SPLIT_DECODE");
        for name in [
            MATRIX_PREFILL_ENV,
            SPLIT_DECODE_ENV,
            "QWEN_MUSE_MATRIX_PREFILL",
            "QWEN_MUSE_SPLIT_DECODE",
        ] {
            assert!(parse_math_flag(name, None).unwrap());
            assert!(!parse_math_flag(name, Some("0")).unwrap());
            assert!(parse_math_flag(name, Some("1")).unwrap());
            for invalid in ["", "true", "yes", " 1", "2"] {
                assert!(
                    parse_math_flag(name, Some(invalid))
                        .unwrap_err()
                        .to_string()
                        .contains(name)
                );
            }
        }
        let defaults = MuseGlimmerRuntimeOptions::default();
        assert!(defaults.matrix_prefill && defaults.split_decode);
    }
}
