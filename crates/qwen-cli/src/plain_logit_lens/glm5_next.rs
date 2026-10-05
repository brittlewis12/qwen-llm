//! GLM-5.3-Flash plain logit lens: native post-block residual streams read
//! out through the deployed tail (the head's fixed four-stream mean, output
//! norm, untied head). Raw prompts or token ids only (`[gMASK]<sop>` belongs
//! in the input); no chat rendering, transport, fitting or interventions.
//!
//! Content identity is not computed: hashing the 117 GB artifact is outside
//! this lane's policy. The artifact is identified by its admitted tensor
//! layout and its tokenizer metadata.
use super::*;
use qwen_llm::glm5_next::{Glm5NextConfig, Glm5NextPreparedArtifact};
use qwen_llm::glm5_next_metal::{
    DEFAULT_PREFILL_ROWS, Glm5NextSession, Glm5NextWeights, prefetch_retained_with_cancel,
    preflight_session,
};
use qwen_llm::metal::MetalContext;

const FAMILY: &str = "GLM-5.3-Flash";

pub(super) struct Prepared<'g> {
    artifact: Glm5NextPreparedArtifact<'g>,
    tokens: Vec<i32>,
    position: usize,
    layers: Vec<u32>,
    capture_layers: Vec<u32>,
}

type Selection = (Vec<i32>, usize, Vec<u32>, Vec<u32>);

/// What lens preflight needs from the configuration.
#[derive(Clone, Copy)]
struct Geometry {
    vocab: u32,
    context: usize,
    /// Executed blocks (the NextN block is never run).
    blocks: u32,
    /// Values per site: every stream of the post-block residual.
    width: usize,
}

impl From<&Glm5NextConfig> for Geometry {
    fn from(config: &Glm5NextConfig) -> Self {
        Self {
            vocab: config.vocab_size,
            context: config.context_length as usize,
            blocks: config.executed_block_count(),
            width: config.hc_width() as usize,
        }
    }
}

fn preflight(
    args: &ReadFullArgs,
    geometry: Geometry,
    encoded: Option<Vec<i32>>,
) -> Result<Selection> {
    ensure!(
        !args.allow_unvalidated_transfer,
        "{FAMILY} plain observations do not accept --allow-unvalidated-transfer"
    );
    let (tokens, position) = input(args, encoded, geometry.vocab, geometry.context)?;
    let layers = layers(&args.layers, geometry.blocks)?;
    capture_budget(layers.len(), geometry.width)?;
    metadata_budget(
        args,
        layers.len(),
        geometry.width,
        tokens.len(),
        geometry.vocab as usize,
    )?;
    let mut capture_layers = layers.clone();
    capture_layers.sort_unstable();
    Ok((tokens, position, layers, capture_layers))
}

/// Packed-prefill rows for the prefix before the captured position.
fn prefill_rows(packed: bool, position: usize) -> usize {
    if packed {
        position.min(DEFAULT_PREFILL_ROWS)
    } else {
        0
    }
}

impl<'g> Prepared<'g> {
    pub(super) fn new(args: &ReadFullArgs, gguf: &'g GgufFile) -> Result<Self> {
        ensure!(
            !args.allow_unvalidated_transfer,
            "{FAMILY} plain observations do not accept --allow-unvalidated-transfer"
        );
        let artifact = Glm5NextPreparedArtifact::inspect(gguf)
            .with_context(|| format!("admit {FAMILY} artifact"))?;
        // The glm4 tokenizer inserts nothing; the flag only governs parsing.
        let encoded = args
            .prompt
            .as_ref()
            .map(|p| artifact.tokenizer().encode(p, !args.no_special_tokens))
            .transpose()?;
        let (tokens, position, layers, capture_layers) =
            preflight(args, artifact.config().into(), encoded)?;
        Ok(Self {
            artifact,
            tokens,
            position,
            layers,
            capture_layers,
        })
    }

    pub(super) fn execute(self, args: &ReadFullArgs, gguf: &GgufFile) -> Result<Readout> {
        let config = self.artifact.config();
        let blocks = config.executed_block_count();
        let (streams, hidden) = (config.hc_streams as usize, config.hidden_size as usize);
        let capacity = self.position + 1;
        let rows = prefill_rows(self.artifact.packed_prefill(), self.position);
        crate::shutdown::checkpoint()?;
        let ctx = MetalContext::new()?;
        preflight_session(&ctx, gguf, self.artifact.model(), capacity, rows)
            .context("admit GLM-5.3 lens session")?;
        prefetch_retained_with_cancel(&ctx, gguf, 0.98, &|| crate::shutdown::checkpoint().is_err())
            .context("prefetch GLM-5.3 retained windows")?;
        let weights = Glm5NextWeights::load(&ctx, gguf).context("load GLM-5.3 weights")?;
        let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, rows)
            .context("create GLM-5.3 lens session")?;
        let executed = self.tokens[..=self.position]
            .iter()
            .map(|&id| id as u32)
            .collect::<Vec<_>>();
        if self.position > 0 {
            session.prefill_packed_with_checkpoint(
                &ctx,
                &executed[..self.position],
                &mut || crate::shutdown::checkpoint().map_err(|e| e.to_string()),
            )?;
        }
        crate::shutdown::checkpoint()?;
        let capture = session.forward_with_post_block_captures(
            &ctx,
            executed[self.position],
            &self.capture_layers,
        )?;
        ensure!(
            capture.position == self.position
                && capture.layers == self.capture_layers
                && (capture.streams, capture.hidden) == (streams, hidden),
            "{FAMILY} capture metadata mismatch"
        );
        let mut bundle = Bundle::optional(
            args.full_output.as_deref(),
            &self.layers,
            config.vocab_size as usize,
        )?;
        let tokenizer = self.artifact.tokenizer();
        let mut results = Vec::with_capacity(self.layers.len());
        for (slot, &layer) in self.layers.iter().enumerate() {
            crate::shutdown::checkpoint()?;
            let index = self
                .capture_layers
                .binary_search(&layer)
                .ok()
                .context("missing GLM-5.3 capture site")?;
            let residual = capture.site(index).context("missing GLM-5.3 residual")?;
            let logits = session.readout(&ctx, residual)?;
            if layer + 1 == blocks {
                ensure!(
                    logits
                        .iter()
                        .zip(&capture.logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{FAMILY} final-block readout differs from ordinary logits"
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
                residual,
                &logits,
                |id| Ok(tokenizer.try_decode_piece_bytes_exact(id)?.to_vec()),
            )?;
            if args.include_vector {
                label_streams(&mut result, streams, hidden);
            }
            results.push(result);
        }
        let metadata = json!({
            "architecture": "glm5-next", "n_layers": blocks, "hidden_size": hidden,
            "residual_streams": streams, "vocab_size": config.vocab_size,
            "lm_head_dtype": format!("{:?}", self.artifact.model().output.dtype),
            "output_tail": "native_four_stream_mean_rmsnorm_untied_head",
            "checkpoint_context_length": config.context_length, "sparse_frontier": config.sparse_frontier(),
            "executed_capacity": capacity, "start_position": 0,
            "requested_layer_order": self.layers, "runtime_capture_layer_order": self.capture_layers,
            "latent_cache": "f16",
            "prefill": {"mode": if rows > 0 { "packed_fast" } else { "serial" }, "rows": rows,
                "prefix_tokens": self.position},
            "capture": "serial_decode_of_the_selected_position",
            "qualification_scope": "ud_iq3_xxs_m4max",
            "tokenizer_metadata_id": format!("{:016x}", qwen_llm::runtime::tokenizer_metadata_identity(gguf)),
            "tokenizer": {"implementation": "native_glm4", "metadata_model": gguf.get_str("tokenizer.ggml.model"),
                "metadata_pre": gguf.get_str("tokenizer.ggml.pre"), "bos_insertion": false,
                "release_prefix": "[gMASK]<sop> belongs in the raw input"},
            "content_identity": {"scheme": "not_computed",
                "reason": "artifact content is not hashed; identified by admitted tensor layout and tokenizer metadata"},
        });
        Ok((self.tokens, self.position, metadata, results, bundle))
    }
}

/// The captured vector is every stream of the post-block residual,
/// stream-major, before the head's collapse.
fn label_streams(result: &mut Value, streams: usize, hidden: usize) {
    let vector = &mut result["transported_vector"];
    vector["hidden_coordinate"] = json!("native_source_post_block_residual_streams");
    vector["hidden_size"] = json!(hidden);
    vector["residual_streams"] = json!(streams);
    vector["shape"] = json!([streams, hidden]);
    vector["stream_layout"] = json!("stream_major");
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

    fn args(extra: &[&str]) -> ReadFullArgs {
        Cli::parse_from(
            [
                "read-full",
                "--model",
                "unused.gguf",
                "--token-ids",
                "154822,154824,42,17",
                "--logit-lens",
                "--identity-cache",
                "unused",
            ]
            .into_iter()
            .chain(extra.iter().copied()),
        )
        .args
    }

    /// The release geometry: 45 executed blocks, four 4096-wide streams.
    const RELEASE: Geometry = Geometry {
        vocab: 154_880,
        context: 1 << 20,
        blocks: 45,
        width: 4 * 4096,
    };

    #[test]
    fn preflight_selects_executed_blocks_and_four_stream_sites() {
        let config = RELEASE;
        let (tokens, position, requested, captured) = preflight(&args(&[]), config, None).unwrap();
        assert_eq!(tokens, [154_822, 154_824, 42, 17]);
        assert_eq!(position, 3);
        assert_eq!(requested, (0..45).collect::<Vec<_>>());
        assert_eq!(captured, requested);
        let (_, _, requested, captured) =
            preflight(&args(&["--layers", "44,0,20"]), config, None).unwrap();
        assert_eq!((requested, captured), (vec![44, 0, 20], vec![0, 20, 44]));
        // Block 45 is the never-executed NextN block.
        for bad in [["--layers", "45"], ["--layers", "3,3"]] {
            assert!(preflight(&args(&bad), config, None).is_err(), "{bad:?}");
        }
        assert!(preflight(&args(&["--allow-unvalidated-transfer"]), config, None).is_err());
        assert!(preflight(&args(&["--position", "4"]), config, None).is_err());
        assert_eq!(prefill_rows(true, 0), 0);
        assert_eq!(prefill_rows(true, 3), 3);
        assert_eq!(prefill_rows(true, 4096), DEFAULT_PREFILL_ROWS);
        assert_eq!(prefill_rows(false, 9), 0);
    }

    #[test]
    fn only_plain_read_full_passes_the_family_gate_before_any_asset_access() {
        let fixture = crate::linear_transport::tests::fixture("glm-gate", 2, 123);
        let path = fixture.0.join("model.gguf");
        crate::full_lens::write_cpu_gguf(&path, "glm5-next", 2, "unused", false);
        let (model, missing, cache) = (
            path.to_str().unwrap(),
            fixture.0.join("missing-asset"),
            fixture.0.join("must-not-exist"),
        );
        let (missing, cache) = (missing.to_str().unwrap(), cache.to_str().unwrap());
        let command = |argv: &[&str]| crate::Cli::try_parse_from(argv).unwrap().command;
        for refused in [
            command(&[
                "qwen-lens",
                "trace-full",
                "--model",
                model,
                "--token-ids",
                "0",
                "--full-lens",
                missing,
                "--identity-cache",
                cache,
            ]),
            command(&[
                "qwen-lens",
                "read-full",
                "--model",
                model,
                "--token-ids",
                "0",
                "--full-lens",
                missing,
                "--identity-cache",
                cache,
            ]),
        ] {
            let error = crate::validate_glm5_next_command(&refused).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("supports only read-full --logit-lens"),
                "{error}"
            );
        }
        let plain = command(&[
            "qwen-lens",
            "read-full",
            "--model",
            model,
            "--token-ids",
            "0",
            "--logit-lens",
            "--identity-cache",
            cache,
        ]);
        crate::validate_glm5_next_command(&plain).unwrap();
        assert!(!fixture.0.join("must-not-exist").exists());
    }

    #[test]
    fn stream_vectors_name_their_layout() {
        let args = args(&["--include-vector", "--top-k", "1"]);
        let mut result = row(&args, 44, 3, 17, &[0.5; 8], &[1.0, 2.0], |_| {
            Ok(b"x".to_vec())
        })
        .unwrap();
        label_streams(&mut result, 4, 2);
        let vector = &result["transported_vector"];
        assert_eq!(vector["shape"], json!([4, 2]));
        assert_eq!(vector["hidden_size"], 2);
        assert_eq!(
            vector["hidden_coordinate"],
            "native_source_post_block_residual_streams"
        );
        assert_eq!(vector["values"].as_array().unwrap().len(), 8);
    }
}
