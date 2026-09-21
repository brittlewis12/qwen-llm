use super::full_lens::ReadFullArgs;
use super::full_output::{Bundle, ranked};
use anyhow::{Context, Result, ensure};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::{LoadedModelConfig, ModelLoadIntent, Runtime, SequenceConfig};
use qwen_llm::tokenizer::{LlamaCppTokenizer, Tokenizer};
use serde_json::{Value, json};

mod k2;
pub(crate) use k2::read_transport as read_k2_transport;

fn layers(requested: &[u32], count: u32) -> Result<Vec<u32>> {
    ensure!(
        count > 0 && count <= 4096,
        "model layer count exceeds capture bound"
    );
    let layers = if requested.is_empty() {
        (0..count).collect()
    } else {
        requested.to_vec()
    };
    ensure!(
        !layers.is_empty()
            && layers.len() <= 4096
            && layers.iter().all(|&l| l < count)
            && layers
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == layers.len(),
        "--layers must be unique model blocks"
    );
    Ok(layers)
}

fn input(
    args: &ReadFullArgs,
    encoded: Option<Vec<i32>>,
    vocab: u32,
    context: usize,
) -> Result<(Vec<i32>, usize)> {
    ensure!(
        args.prompt.as_ref().is_none_or(|prompt| !prompt.is_empty()),
        "--prompt must not be empty"
    );
    ensure!(
        args.prompt.is_some() ^ !args.token_ids.is_empty(),
        "specify exactly one of --prompt or --token-ids"
    );
    ensure!(
        args.prompt.is_some() || !args.no_special_tokens,
        "--no-special-tokens requires --prompt"
    );
    let tokens = if let Some(tokens) = encoded {
        tokens
    } else {
        args.token_ids
            .iter()
            .map(|&id| i32::try_from(id).context("token ID overflow"))
            .collect::<Result<Vec<_>>>()?
    };
    ensure!(
        !tokens.is_empty()
            && tokens.len() <= args.max_tokens
            && tokens.iter().all(|&t| t >= 0 && (t as u32) < vocab),
        "invalid or oversized input token sequence"
    );
    let position = args.position.unwrap_or(tokens.len() - 1);
    ensure!(
        position < tokens.len() && position < context,
        "selected position exceeds input or context"
    );
    Ok((tokens, position))
}

fn row(
    args: &ReadFullArgs,
    layer: u32,
    position: usize,
    token: i32,
    residual: &[f32],
    logits: &[f32],
    decode: impl Fn(i32) -> Result<Vec<u8>>,
) -> Result<Value> {
    ensure!(
        residual.iter().all(|v| v.is_finite()),
        "nonfinite native residual"
    );
    let top = ranked(logits, args.top_k)?.into_iter().enumerate().map(|(rank, (id, logit))| {
        let piece = decode(i32::try_from(id)?)?;
        Ok(json!({"rank": rank, "token_id": id, "logit": logit, "token_display_lossy": String::from_utf8_lossy(&piece), "token_piece_hex": super::hex(&piece)}))
    }).collect::<Result<Vec<_>>>()?;
    let mut row = json!({"source_layer": layer, "source_position": position, "source_token_id": token, "predicts_position": position + 1, "top_k": top});
    if args.include_vector {
        row["transported_vector"] = json!({"operation": "identity", "stage": "before_architectural_output_norm", "value_dtype": "f32", "hidden_coordinate": "native_source_post_block_residual", "hidden_size": residual.len(), "shape": [residual.len()], "values": residual});
    }
    Ok(row)
}

pub(super) fn read(args: ReadFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    ensure!(
        args.logit_lens && args.full_lens.is_none(),
        "--logit-lens conflicts with --full-lens"
    );
    ensure!(
        args.output.is_none() || args.full_output.is_none(),
        "--output conflicts with --full-output"
    );
    ensure!(
        (1..=1024).contains(&args.top_k) && args.max_tokens > 0,
        "invalid top-k or max-tokens"
    );
    let gguf = GgufFile::open(&args.model)?;
    let family = ModelFamily::detect(&gguf);
    let k2 = if family == Some(ModelFamily::K2Horizon) {
        Some(k2::Prepared::new(&args, &gguf)?)
    } else {
        None
    };
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(&args.identity_cache))?;
    let (tokens, position, model, results, bundle) = if let Some(prepared) = k2 {
        prepared.execute(&args, &gguf, &content, None)?
    } else if family == Some(ModelFamily::MuseGlimmer) {
        use qwen_llm::metal::MetalContext;
        use qwen_llm::muse_glimmer::MuseGlimmerModel;
        use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
        let bound = MuseGlimmerModel::from_gguf(&gguf)?;
        let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)?;
        let encoded = args
            .prompt
            .as_ref()
            .map(|p| tokenizer.encode(p, !args.no_special_tokens))
            .transpose()?;
        let (tokens, position) = input(
            &args,
            encoded,
            bound.config.vocab_size,
            bound.config.context_length as usize,
        )?;
        let layers = layers(&args.layers, bound.config.layer_count)?;
        capture_budget(layers.len(), bound.config.hidden_size as usize)?;
        metadata_budget(
            &args,
            layers.len(),
            bound.config.hidden_size as usize,
            tokens.len(),
            bound.config.vocab_size as usize,
        )?;
        let context = MetalContext::new()?;
        let mut loaded = MuseGlimmerLoadedModel::load_reference(&context, &gguf, position + 1)?;
        let mut runner = loaded.create_runner(&context)?;
        for &token in &tokens[..position] {
            crate::shutdown::checkpoint()?;
            runner.forward_token(token as u32)?;
        }
        let mut capture_layers = layers.clone();
        capture_layers.sort_unstable();
        let capture =
            runner.forward_token_capture_post_blocks(tokens[position] as u32, &capture_layers)?;
        ensure!(
            capture.position == position
                && capture.token_id == tokens[position] as u32
                && capture.layer_ids == capture_layers
                && capture.hidden_size == bound.config.hidden_size as usize,
            "native Muse capture metadata mismatch"
        );
        let mut bundle = Bundle::optional(
            args.full_output.as_deref(),
            &layers,
            bound.config.vocab_size as usize,
        )?;
        let mut results = Vec::new();
        for (slot, &layer) in layers.iter().enumerate() {
            crate::shutdown::checkpoint()?;
            let residual = capture
                .layer_values(capture_layers.binary_search(&layer).unwrap())
                .context("missing native Muse residual")?;
            let logits = runner.deployed_logits_from_post_block_residual(residual)?;
            if let Some(bundle) = &mut bundle {
                bundle.row(slot, &logits)?;
            }
            results.push(row(
                &args,
                layer,
                position,
                tokens[position],
                residual,
                &logits,
                |id| Ok(tokenizer.try_decode_piece_bytes_exact(id)?),
            )?);
        }
        (
            tokens,
            position,
            json!({"architecture": ModelFamily::MuseGlimmer.architecture_name(), "n_layers": bound.config.layer_count, "hidden_size": bound.config.hidden_size, "vocab_size": bound.config.vocab_size, "output_tail": "native_rmsnorm_output_projection_logit_scale_final_softcap"}),
            results,
            bundle,
        )
    } else {
        ensure!(
            matches!(family, Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)),
            "plain logit lens has no native single-post-block-residual/deployed-head adapter for this architecture; multistream geometry requires an explicit residual-to-head adapter"
        );
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let encoded = args
            .prompt
            .as_ref()
            .map(|p| tokenizer.encode(p, !args.no_special_tokens))
            .transpose()?;
        let (tokens, position) = input(
            &args,
            encoded,
            tokenizer.n_vocab(),
            gguf.declared_context_length()?,
        )?;
        let runtime = Runtime::metal()?;
        let loaded = runtime.load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )?;
        let arch = loaded.arch();
        let layers = layers(&args.layers, arch.n_layer)?;
        capture_budget(layers.len(), arch.hidden_size as usize)?;
        metadata_budget(
            &args,
            layers.len(),
            arch.hidden_size as usize,
            tokens.len(),
            arch.vocab_size as usize,
        )?;
        crate::lens_run::ensure_qwen_sequence_admitted(&loaded, position + 1)?;
        let mut sequence = loaded.create_sequence(SequenceConfig::new(position + 1))?;
        let mut session = loaded.passive_workspace_lens_session(&mut sequence)?;
        let capture =
            session.forward_prompt_last_post_block_residuals(&tokens[..=position], &layers)?;
        ensure!(
            capture.position == position
                && capture.token_id == tokens[position]
                && capture.capture.layer_ids == layers
                && capture.capture.hidden_size == arch.hidden_size as usize,
            "native Qwen capture metadata mismatch"
        );
        let mut bundle = Bundle::optional(
            args.full_output.as_deref(),
            &layers,
            arch.vocab_size as usize,
        )?;
        let mut results = Vec::new();
        for (slot, &layer) in layers.iter().enumerate() {
            crate::shutdown::checkpoint()?;
            let residual = capture
                .capture
                .values
                .get(slot * arch.hidden_size as usize..(slot + 1) * arch.hidden_size as usize)
                .context("missing native Qwen residual")?;
            let readout = session.deployed_logits_from_post_block_residual(residual)?;
            let logits = readout.logits;
            if let Some(bundle) = &mut bundle {
                bundle.row(slot, &logits)?;
            }
            results.push(row(
                &args,
                layer,
                position,
                tokens[position],
                residual,
                &logits,
                |id| Ok(tokenizer.try_decode_piece_bytes_exact(id)?.to_vec()),
            )?);
        }
        let identity = loaded.workspace_lens_identity();
        (
            tokens,
            position,
            json!({"architecture": family.unwrap().architecture_name(), "n_layers": arch.n_layer, "hidden_size": arch.hidden_size, "vocab_size": arch.vocab_size, "lm_head_dtype": format!("{:?}", loaded.metal_model().lm_head.dtype), "model_locator_id": format!("{:016x}", identity.model_locator_id), "tokenizer_metadata_id": format!("{:016x}", identity.tokenizer_metadata_id), "output_tail": "native_output_rmsnorm_and_lm_head"}),
            results,
            bundle,
        )
    };
    let input = input_metadata(&args, &tokens, position, family);
    let observer = json!({"method": "plain_logit_lens", "transport": "identity", "source_site": "native_post_block_residual", "fitted_artifact": null, "transfer_acknowledgement_required": false});
    let mut document = json!({"schema": "llm.lens.readout", "schema_version": 1, "readout": "native_plain_logit_lens", "score_semantics": "deployed_pre_softmax_logits_after_architectural_output_norm_scaling_and_softcap", "ranking_scope": "full_vocabulary", "input_blake3": super::digest_json(&input)?, "input": input, "observer_blake3": super::digest_json(&observer)?, "observer": observer, "deployed_model": model, "reader": {"build_commit": env!("QWEN_BUILD_COMMIT"), "build_dirty": env!("QWEN_BUILD_DIRTY"), "build_source_state": env!("QWEN_BUILD_SOURCE_STATE"), "build_stamp_source": env!("QWEN_BUILD_STAMP_SOURCE"), "build_stamp_error": env!("QWEN_BUILD_STAMP_ERROR")}});
    document["results"] = Value::Array(results);
    document["deployed_model"]["content_blake3"] = json!(super::hex(&content.content_id));
    document["deployed_model"]["path"] = json!(args.model);
    document["execution_provenance"] = super::full_output::execution_provenance();
    let bytes = serialize_then_publish(&document, bundle)?;
    if let Some(output) = &args.output {
        super::publish_immutable(&super::full_lens::resolve_output_file(output)?, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}

fn input_metadata(
    args: &ReadFullArgs,
    tokens: &[i32],
    position: usize,
    family: Option<ModelFamily>,
) -> Value {
    let mut input = json!({"source": if args.prompt.is_some() { "prompt" } else { "token_ids" }, "add_special_tokens": args.prompt.as_ref().map(|_| !args.no_special_tokens), "token_ids": tokens, "selected_position": position, "captured_token_id": tokens[position], "predicts_position": position + 1});
    // Existing families' input objects are digest contracts; do not extend them
    // as a side effect of adding K2 provenance.
    if family == Some(ModelFamily::K2Horizon) {
        input["input_token_count"] = json!(tokens.len());
        input["executed_token_count"] = json!(position + 1);
    }
    input
}

fn capture_budget(layers: usize, hidden: usize) -> Result<()> {
    let bytes = layers
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(4))
        .context("native capture dimensions overflow")?;
    ensure!(
        hidden > 0 && bytes <= 512 * 1024 * 1024,
        "native capture exceeds 512 MiB"
    );
    Ok(())
}

fn serialize_then_publish(document: &Value, bundle: Option<Bundle>) -> Result<Vec<u8>> {
    let bytes = super::serialize_json_pretty_bounded(document, "plain logit lens")?;
    if let Some(bundle) = bundle {
        bundle.publish(document)?;
    }
    Ok(bytes)
}

fn metadata_budget(
    args: &ReadFullArgs,
    layers: usize,
    hidden: usize,
    tokens: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(
        (1..=1024).contains(&args.top_k) && args.top_k <= vocab,
        "plain --top-k must be in 1..=min(1024, vocabulary)"
    );
    super::full_output::retained_metadata_budget(
        layers,
        hidden,
        tokens,
        args.top_k,
        args.include_vector,
    )
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

    fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(
            [
                "read-full",
                "--model",
                "model.gguf",
                "--token-ids",
                "1,2",
                "--identity-cache",
                "cache",
            ]
            .into_iter()
            .chain(extra.iter().copied()),
        )
    }

    #[test]
    fn plain_logit_lens_flags_and_defaults() {
        let cli = parse(&[
            "--logit-lens",
            "--full-output",
            "bundle",
            "--include-vector",
        ])
        .unwrap();
        assert!(cli.args.logit_lens && cli.args.full_lens.is_none());
        assert!(!cli.args.allow_unvalidated_transfer);
        assert_eq!(layers(&cli.args.layers, 4).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(layers(&[3, 1], 4).unwrap(), vec![3, 1]);
        assert!(layers(&[1, 1], 4).is_err());
        assert!(layers(&[4], 4).is_err());
        assert!(parse(&[]).is_err());
        assert!(parse(&["--logit-lens", "--full-lens", "fit"]).is_err());
        assert!(
            parse(&[
                "--logit-lens",
                "--full-output",
                "bundle",
                "--output",
                "out.json"
            ])
            .is_err()
        );
        assert!(parse(&["--full-lens", "fit", "--full-output", "bundle"]).is_ok());
    }

    #[test]
    fn plain_logit_lens_vector_is_native_identity() {
        let args = parse(&["--logit-lens", "--include-vector", "--top-k", "1"])
            .unwrap()
            .args;
        let result = row(&args, 3, 1, 2, &[0.25, -0.5], &[1.0, 2.0], |_| {
            Ok(b"piece".to_vec())
        })
        .unwrap();
        assert_eq!(result["transported_vector"]["operation"], "identity");
        assert_eq!(
            result["transported_vector"]["hidden_coordinate"],
            "native_source_post_block_residual"
        );
        assert_eq!(result["transported_vector"]["values"], json!([0.25, -0.5]));
        assert_eq!(result["top_k"][0]["token_id"], 1);
    }

    #[test]
    fn k2_input_counts_do_not_change_existing_family_digest_contracts() {
        let args = parse(&["--logit-lens"]).unwrap().args;
        let original = json!({"source":"token_ids", "add_special_tokens":null, "token_ids":[1,2],
            "selected_position":0, "captured_token_id":1, "predicts_position":1});
        for family in [
            ModelFamily::Qwen35,
            ModelFamily::Qwen35Moe,
            ModelFamily::MuseGlimmer,
        ] {
            let input = input_metadata(&args, &[1, 2], 0, Some(family));
            assert_eq!(input, original);
            assert_eq!(
                crate::digest_json(&input).unwrap(),
                crate::digest_json(&original).unwrap()
            );
        }
        let k2 = input_metadata(&args, &[1, 2], 0, Some(ModelFamily::K2Horizon));
        assert_eq!(k2["input_token_count"], 2);
        assert_eq!(k2["executed_token_count"], 1);
    }

    #[test]
    fn plain_top_k_uses_general_metadata_and_vocabulary_bounds() {
        let mut args = parse(&["--logit-lens", "--top-k", "1024"]).unwrap().args;
        metadata_budget(&args, 1, 5120, 256, 1024).unwrap();
        assert!(metadata_budget(&args, 1, 5120, 256, 1023).is_err());
        args.top_k = 1025;
        assert!(metadata_budget(&args, 1, 5120, 256, 2000).is_err());
        args.top_k = 64;
        metadata_budget(&args, 1, 5120, 256, 2000).unwrap();
    }

    #[test]
    fn compact_serialization_failure_precedes_bundle_publication() {
        let output = std::env::temp_dir().join(format!(
            "qwen-compact-order-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut bundle = Bundle::optional(Some(&output), &[0], 1).unwrap().unwrap();
        bundle.row(0, &[1.0]).unwrap();
        let document = json!({"large": "x".repeat(crate::JSON_FILE_MAX_BYTES)});
        assert!(serialize_then_publish(&document, Some(bundle)).is_err());
        assert!(!output.exists());
    }
}
