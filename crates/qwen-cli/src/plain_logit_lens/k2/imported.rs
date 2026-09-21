//! K2 data-only transport policy. Never falls through to legacy Qwen/Muse assets.
use super::*;
use crate::linear_transport::{ExpectedProfile, VerifiedTransport};
use qwen_llm::checkpoint_identity::verified_checkpoint_content_identity;

pub(crate) fn read(mut args: ReadFullArgs, gguf: GgufFile) -> Result<()> {
    crate::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    ensure!(
        !args.logit_lens && args.full_lens.is_some(),
        "K2 imported readout requires --full-lens, not --logit-lens"
    );
    ensure!(
        args.output.is_none() || args.full_output.is_none(),
        "--output conflicts with --full-output"
    );
    let config = K2HorizonConfig::from_gguf(&gguf)?;
    let directory = args.full_lens.as_ref().unwrap().clone();
    let mut data = VerifiedTransport::open_with_expected_profile(
        &directory,
        ExpectedProfile {
            architecture: "k2-horizon",
            n_layers: config.layer_count,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            target_layer: config.layer_count - 1,
        },
    )
    .context("open K2 data-only linear transport")?;
    select_layers(&mut args, &data)?;
    let prepared = Prepared::new_mode(&args, &gguf, false)?;
    ensure!(
        data.manifest().model.exact_binding.is_some() || args.allow_unvalidated_transfer,
        "source/deployment equivalence unverified; require --allow-unvalidated-transfer"
    );
    crate::shutdown::checkpoint()?;
    // Exact imported bindings never inherit cached/downloader-declared roots.
    // Even explicit unverified transfer records the actual deployed byte identity.
    let content = verified_checkpoint_content_identity(&gguf)?;
    crate::shutdown::checkpoint()?;
    let tokenizer = format!(
        "{:016x}",
        qwen_llm::runtime::tokenizer_metadata_identity(&gguf)
    );
    let status = data.bind(
        "k2-horizon",
        config.layer_count,
        config.hidden_size,
        config.vocab_size,
        &crate::hex(&content.content_id),
        &tokenizer,
        true,
        args.allow_unvalidated_transfer,
    )?;
    let artifact = artifact_summary(&data, &directory)?;
    let exact = data.manifest().model.exact_binding.is_some();
    let method = data.manifest().transport.method.clone();
    let transfer = json!({"status": status, "requested_override": args.allow_unvalidated_transfer,
        "override_applied": !exact, "policy": if exact { "exact_binding_required_even_with_override" } else { "explicit_unvalidated_transfer" },
        "qualification": "producer_claims_only_not_revalidated"});
    let (tokens, position, mut model, results, bundle) =
        prepared.execute(&args, &gguf, &content, Some(&mut data))?;
    model["content_blake3"] = json!(crate::hex(&content.content_id));
    model["path"] = json!(args.model);
    model["content_identity"]["trust"] =
        json!("retained_bytes_hashed_no_cache_or_downloader_declarations");
    let input =
        super::super::input_metadata(&args, &tokens, position, Some(ModelFamily::K2Horizon));
    let observer = json!({"method":method, "transport":"post_block_linear", "source_site":"native_post_block_residual",
        "target_layer":35, "coefficient_dtype":"f16_le", "orientation":"target_source", "output":"deployed_native",
        "artifact_manifest_blake3":artifact["manifest_canonical_json_blake3"], "payload_sha256":artifact["payload_sha256"],
        "transfer":transfer});
    let document = json!({"schema":"llm.lens.readout", "schema_version":1, "readout":"native_linear_transport",
        "score_semantics":"deployed_pre_softmax_logits_after_architectural_output_norm_scaling_and_softcap",
        "ranking_scope":"full_vocabulary", "source_site":"native_post_block_residual", "scoring":"transport_then_deployed_output_tail",
        "input_blake3":crate::digest_json(&input)?, "input":input,
        "observer_blake3":crate::digest_json(&observer)?, "observer":observer,
        "artifact":artifact, "transfer":transfer, "deployed_model":model, "results":results,
        "reader":{"build_commit":env!("QWEN_BUILD_COMMIT"), "build_dirty":env!("QWEN_BUILD_DIRTY"),
            "build_source_state":env!("QWEN_BUILD_SOURCE_STATE"), "build_stamp_source":env!("QWEN_BUILD_STAMP_SOURCE"), "build_stamp_error":env!("QWEN_BUILD_STAMP_ERROR")},
        "execution_provenance":crate::full_output::execution_provenance()});
    let bytes = super::super::serialize_then_publish(&document, bundle)?;
    if let Some(output) = &args.output {
        crate::publish_immutable(&crate::full_lens::resolve_output_file(output)?, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}

fn select_layers(args: &mut ReadFullArgs, data: &VerifiedTransport) -> Result<()> {
    let sources = &data.manifest().transport.source_layers;
    if args.layers.is_empty() {
        args.layers = sources.clone();
    }
    ensure!(
        args.layers.iter().all(|layer| sources.contains(layer)),
        "--layers must select source layers present in the K2 transport"
    );
    // The common preflight also rejects duplicates and preserves requested order.
    Ok(())
}

fn artifact_summary(data: &VerifiedTransport, directory: &std::path::Path) -> Result<Value> {
    let mut canonical = data.original_manifest().clone();
    canonical.sort_all_objects();
    Ok(
        json!({"kind":"linear_transport", "manifest":directory.join("lens.json"),
        "manifest_canonical_json_blake3":crate::digest_json(&canonical)?,
        "payload_sha256":data.manifest().payload.sha256, "matrix_sha256":data.manifest().payload.matrix_sha256,
        "payload_blake3":data.payload_blake3(), "producer_contract":data.original_manifest(),
        "verification":"whole_payload_observed_before_execution_selected_matrix_rehashed_before_each_upload",
        "qualification":"producer_claims_only"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn imported_selection_preserves_manifest_or_explicit_order_and_digest_labels() {
        let fixture = crate::linear_transport::tests::fixture("synthetic-host-only", 2, 17);
        let data = VerifiedTransport::open(&fixture.0).unwrap();
        let command = crate::Cli::try_parse_from([
            "qwen-lens",
            "read-full",
            "--model",
            "unused.gguf",
            "--token-ids",
            "0",
            "--full-lens",
            fixture.0.to_str().unwrap(),
            "--identity-cache",
            "unused-cache",
        ])
        .unwrap()
        .command;
        let crate::Command::ReadFull(mut args) = command else {
            panic!("read-full");
        };
        select_layers(&mut args, &data).unwrap();
        assert_eq!(args.layers, [2, 0]);
        args.layers = vec![0, 2];
        select_layers(&mut args, &data).unwrap();
        assert_eq!(args.layers, [0, 2]);
        args.layers = vec![1];
        assert!(select_layers(&mut args, &data).is_err());
        let summary = artifact_summary(&data, &fixture.0).unwrap();
        assert_eq!(summary["payload_sha256"], data.manifest().payload.sha256);
        assert_eq!(summary["payload_blake3"], data.payload_blake3());
        assert_ne!(summary["payload_sha256"], summary["payload_blake3"]);
        assert_eq!(
            summary["matrix_sha256"],
            json!(data.manifest().payload.matrix_sha256)
        );
        assert_eq!(summary["producer_contract"], *data.original_manifest());
    }
}
