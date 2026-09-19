//! Native plain/data-only readout. No Qwen runtime, local fitting, or chat rendering.
use super::*;
use qwen_llm::checkpoint_identity::CheckpointContentReport;
use qwen_llm::k2_horizon::{K2HorizonConfig, K2HorizonModel};
use qwen_llm::k2_horizon_runtime::K2LoadedModel;
use qwen_llm::metal::MetalContext;
use qwen_llm::tokenizer::NativeTokenizer;

mod imported;
pub(crate) use imported::read as read_transport;

use qwen_llm::k2_horizon_runtime::GUARDED_APPLICATION_FORWARD_CEILING as MAX_FORWARDS;

pub(super) struct Prepared {
    config: K2HorizonConfig,
    tokenizer: NativeTokenizer,
    tokens: Vec<i32>,
    position: usize,
    layers: Vec<u32>,
    capture_layers: Vec<u32>,
    head_dtype: String,
}

fn preflight(
    args: &ReadFullArgs,
    config: &K2HorizonConfig,
    encoded: Option<Vec<i32>>,
) -> Result<(Vec<i32>, usize, Vec<u32>, Vec<u32>)> {
    preflight_mode(args, config, encoded, true)
}

fn preflight_mode(
    args: &ReadFullArgs,
    config: &K2HorizonConfig,
    encoded: Option<Vec<i32>>,
    plain: bool,
) -> Result<(Vec<i32>, usize, Vec<u32>, Vec<u32>)> {
    ensure!(
        !plain || !args.allow_unvalidated_transfer,
        "K2 plain observations do not accept --allow-unvalidated-transfer"
    );
    let (tokens, position) = input(
        args,
        encoded,
        config.vocab_size,
        config.context_length as usize,
    )?;
    ensure!(
        position < MAX_FORWARDS,
        "K2 lens readout is bounded to {MAX_FORWARDS} executed tokens; selected position must be below {MAX_FORWARDS}"
    );
    let layers = layers(&args.layers, config.layer_count)?;
    capture_budget(layers.len(), config.hidden_size as usize)?;
    metadata_budget(
        args,
        layers.len(),
        config.hidden_size as usize,
        tokens.len(),
        config.vocab_size as usize,
    )?;
    let mut capture_layers = layers.clone();
    capture_layers.sort_unstable();
    Ok((tokens, position, layers, capture_layers))
}

impl Prepared {
    pub(super) fn new(args: &ReadFullArgs, gguf: &GgufFile) -> Result<Self> {
        Self::new_mode(args, gguf, true)
    }

    fn new_mode(args: &ReadFullArgs, gguf: &GgufFile, plain: bool) -> Result<Self> {
        ensure!(
            !plain || !args.allow_unvalidated_transfer,
            "K2 plain observations do not accept --allow-unvalidated-transfer"
        );
        let bound = K2HorizonModel::from_gguf(gguf)?;
        let tokenizer = NativeTokenizer::from_gguf(gguf)?;
        let encoded = args
            .prompt
            .as_ref()
            .map(|p| tokenizer.encode(p, !args.no_special_tokens))
            .transpose()?;
        let (tokens, position, layers, capture_layers) = if plain {
            preflight(args, &bound.config, encoded)?
        } else {
            preflight_mode(args, &bound.config, encoded, false)?
        };
        Ok(Self {
            config: bound.config.clone(),
            tokenizer,
            tokens,
            position,
            layers,
            capture_layers,
            head_dtype: format!("{:?}", bound.output.dtype),
        })
    }

    pub(super) fn execute(
        self,
        args: &ReadFullArgs,
        gguf: &GgufFile,
        content: &CheckpointContentReport,
        mut transport: Option<&mut crate::linear_transport::VerifiedTransport>,
    ) -> Result<(Vec<i32>, usize, Value, Vec<Value>, Option<Bundle>)> {
        crate::shutdown::checkpoint()?;
        let context = MetalContext::new()?;
        let model = K2LoadedModel::load_unqualified(&context, gguf, (self.position + 1) as u32)?;
        let mut session = model.create_session(0)?;
        for &token in &self.tokens[..self.position] {
            crate::shutdown::checkpoint()?;
            session.append(&[token as u32])?;
        }
        crate::shutdown::checkpoint()?;
        let capture = session
            .append_with_captures(&[self.tokens[self.position] as u32], &self.capture_layers)?;
        ensure!(
            capture.absolute_position as usize == self.position
                && capture.post_block_layers == self.capture_layers,
            "K2 capture metadata mismatch"
        );
        let mut bundle = Bundle::optional(
            args.full_output.as_deref(),
            &self.layers,
            self.config.vocab_size as usize,
        )?;
        let mut results = Vec::with_capacity(self.layers.len());
        for (slot, &layer) in self.layers.iter().enumerate() {
            crate::shutdown::checkpoint()?;
            let index = self
                .capture_layers
                .binary_search(&layer)
                .ok()
                .context("missing K2 capture site")?;
            let source_residual = &capture.residuals[index * 4096..(index + 1) * 4096];
            let (residual, logits) = if let Some(data) = transport.as_mut() {
                let matrix = qwen_llm::k2_horizon_runtime::K2LinearF16::from_target_source_le(
                    data.read_matrix(layer)?,
                )?;
                let readout = session.readout_linear_f16(&matrix, source_residual)?;
                (std::borrow::Cow::Owned(readout.residual), readout.logits)
            } else {
                (
                    std::borrow::Cow::Borrowed(source_residual),
                    session.readout(source_residual)?,
                )
            };
            if transport.is_none() && layer + 1 == self.config.layer_count {
                ensure!(
                    logits
                        .iter()
                        .zip(&capture.logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "K2 final-block readout differs from ordinary logits"
                );
            }
            if let Some(bundle) = &mut bundle {
                bundle.row(slot, &logits)?;
            }
            let mut result = row(
                args,
                layer,
                self.position,
                self.tokens[self.position],
                &residual,
                &logits,
                |id| Ok(self.tokenizer.try_decode_piece_bytes_exact(id)?.to_vec()),
            )?;
            if transport.is_some() && args.include_vector {
                result["transported_vector"]["operation"] = json!("post_block_linear");
                result["transported_vector"]["hidden_coordinate"] =
                    json!("target_final_post_block_residual");
            }
            results.push(result);
        }
        // The lightweight metadata hash includes all tokenizer.* keys (including
        // pair-SEP settings); architecture and the native implementation are named
        // separately. It is not a cryptographic tokenizer/artifact authentication.
        let metadata = json!({
            "architecture": "k2-horizon", "n_layers": self.config.layer_count,
            "hidden_size": self.config.hidden_size, "vocab_size": self.config.vocab_size,
            "lm_head_dtype": self.head_dtype, "output_tail": "native_four_group_rmsnorm_untied_head",
            "checkpoint_context_length": self.config.context_length, "rope_theta": self.config.rope_theta,
            "executed_capacity": self.position + 1, "start_position": 0,
            "requested_layer_order": self.layers, "runtime_capture_layer_order": self.capture_layers,
            "kv_storage": "f16", "execution_topology": "serial_single_token",
            "qualification_scope": "final_q8_m4max_short_context_only",
            "tokenizer_metadata_id": format!("{:016x}", qwen_llm::runtime::tokenizer_metadata_identity(gguf)),
            "tokenizer": {"implementation": "native_k2_horizon", "metadata_model": gguf.get_str("tokenizer.ggml.model"),
                "metadata_pre": gguf.get_str("tokenizer.ggml.pre"), "normalization": "NFC_after_added_token_partition",
                "bos": 0, "eos": 1, "single_sequence_auto_bos": true, "pair_separator_in_single_sequence": false},
            "content_identity": {"scheme": "native_ordered_gguf_content_identity_v1",
                "outcome": format!("{:?}", content.outcome), "bytes_hashed": content.bytes_hashed,
                "trust": "native_local_identity_cache_and_downloader_policy_not_asset_authentication"}
        });
        Ok((self.tokens, self.position, metadata, results, bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ReadFullArgs,
    }

    fn args() -> ReadFullArgs {
        Cli::parse_from([
            "read-full",
            "--model",
            "unused.gguf",
            "--token-ids",
            "0,42,17",
            "--logit-lens",
            "--identity-cache",
            "unused",
        ])
        .args
    }

    fn config() -> K2HorizonConfig {
        K2HorizonConfig {
            layer_count: 36,
            context_length: 524288,
            hidden_size: 4096,
            feed_forward_size: 12288,
            vocab_size: 250624,
            query_head_count: 32,
            kv_head_count: 8,
            key_head_dim: 128,
            value_head_dim: 128,
            norm_groups: 4,
            rms_epsilon: 1e-6,
            rope_dimension_count: 128,
            rope_theta: 10000000.0,
        }
    }

    #[test]
    fn k2_plain_preflight_bounds_executed_prefix_without_discarding_input() {
        let mut args = args();
        args.max_tokens = 512;
        args.token_ids = vec![0; 256];
        assert_eq!(preflight(&args, &config(), None).unwrap().1, 255);
        assert_eq!(
            preflight_mode(&args, &config(), None, false).unwrap().1,
            255
        );
        args.token_ids.push(42);
        assert!(preflight(&args, &config(), None).is_err());
        assert!(preflight_mode(&args, &config(), None, false).is_err());
        args.position = Some(5);
        let (tokens, position, _, _) = preflight(&args, &config(), None).unwrap();
        assert_eq!(
            tokens,
            args.token_ids
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(position, 5);
        args.max_tokens = 256;
        assert!(preflight(&args, &config(), None).is_err());
        args.max_tokens = 512;
        args.token_ids[256] = 250624;
        assert!(preflight(&args, &config(), None).is_err());
    }

    #[test]
    fn k2_plain_preflight_sorts_capture_not_output_and_rejects_transfer() {
        let mut args = args();
        let (_, _, requested, captured) = preflight(&args, &config(), None).unwrap();
        assert_eq!(requested, (0..36).collect::<Vec<_>>());
        assert_eq!(requested, captured);
        args.layers = vec![35, 0, 17];
        let (_, _, requested, captured) = preflight(&args, &config(), None).unwrap();
        assert_eq!(requested, [35, 0, 17]);
        assert_eq!(captured, [0, 17, 35]);
        for sites in [vec![0, 0], vec![36]] {
            args.layers = sites;
            assert!(preflight(&args, &config(), None).is_err());
        }
        args.layers = vec![35];
        args.allow_unvalidated_transfer = true;
        assert!(preflight(&args, &config(), None).is_err());
        args.allow_unvalidated_transfer = false;
        args.no_special_tokens = true;
        assert!(preflight(&args, &config(), None).is_err());
    }

    #[test]
    fn k2_unsupported_trace_gate_precedes_asset_or_identity_access() {
        let fixture = crate::linear_transport::tests::fixture("gate-only", 2, 123);
        let path = fixture.0.join("model.gguf");
        crate::full_lens::write_cpu_gguf(&path, "k2-horizon", 2, "unused", false);
        let command = crate::Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            path.to_str().unwrap(),
            "--token-ids",
            "0",
            "--full-lens",
            fixture.0.join("missing-asset").to_str().unwrap(),
            "--identity-cache",
            fixture.0.join("must-not-exist").to_str().unwrap(),
        ])
        .unwrap()
        .command;
        let error = crate::validate_k2_command(&command).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("K2 Horizon supports only read-full here")
        );
        assert!(!fixture.0.join("must-not-exist").exists());
    }
}
