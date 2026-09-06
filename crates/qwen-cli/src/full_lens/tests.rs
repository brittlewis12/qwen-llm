use super::*;
use sha2::{Digest, Sha256};
use std::io::Cursor;
use zip::CompressionMethod;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

fn test_archive(matrix: &[u8], data_pickle: &[u8]) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    {
        let mut writer = ZipWriter::new(&mut output);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for (name, bytes) in [
            ("lens/data.pkl", data_pickle),
            ("lens/.format_version", b"1" as &[u8]),
            ("lens/.storage_alignment", b"64" as &[u8]),
            ("lens/byteorder", b"little" as &[u8]),
            ("lens/version", b"3" as &[u8]),
            ("lens/.data/serialization_id", b"fixture" as &[u8]),
            ("lens/data/0", matrix),
        ] {
            writer.start_file(name, options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap();
    }
    output.into_inner()
}

fn canonical_payload() -> FullPayload {
    FullPayload {
        path: FULL_PAYLOAD_NAME.into(),
        dtype: "f16_le".into(),
        shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
        byte_length: PAYLOAD_BYTES,
        blake3: PUBLISHED_PROFILES[0].expected_payload_blake3.into(),
    }
}

fn test_read_full_args() -> ReadFullArgs {
    ReadFullArgs {
        model: "model.gguf".into(),
        full_lens: "full-lens".into(),
        prompt: Some("hello".into()),
        token_ids: Vec::new(),
        no_special_tokens: false,
        position: None,
        layers: vec![0, 31, 62],
        top_k: 10,
        max_tokens: 256,
        identity_cache: "identity-cache".into(),
        allow_unvalidated_transfer: true,
        include_vector: false,
        output: None,
    }
}

fn test_trace_full_args() -> TraceFullArgs {
    TraceFullArgs {
        model: "model.gguf".into(),
        full_lens: "full-lens".into(),
        prompt: Some("hello".into()),
        token_ids: None,
        user: None,
        system: None,
        messages: None,
        open_responses: None,
        requests_jsonl: None,
        message_mode: None,
        no_special_tokens: false,
        layers: vec![0, 31, 62],
        top_k: 8,
        max_tokens: None,
        vectors: Vec::new(),
        identity_cache: None,
        allow_unvalidated_transfer: false,
        output: None,
        output_dir: None,
        format: None,
    }
}

#[test]
fn trace_batch_args_replace_single_input_without_changing_shared_bounds() {
    let mut args = test_trace_full_args();
    args.prompt = None;
    args.requests_jsonl = Some("requests.jsonl".into());
    args.output_dir = Some("traces".into());
    validate_trace_full_args(&args).unwrap();

    args.output = Some("single.json".into());
    assert!(validate_trace_full_args(&args).is_err());
}

#[test]
fn trace_batch_jsonl_is_bounded_strict_and_resolves_structured_inputs() {
    let root = std::env::temp_dir().join(format!(
        "qwen-trace-batch-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("messages.json"), "[]").unwrap();
    let requests_path = root.join("requests.jsonl");
    let mut cohort = String::from(concat!(
        "{\"id\":\"literal\",\"token_ids\":[1,2],\"vectors\":[{\"source_layer\":0,\"source_position\":1}]}\n",
        "{\"id\":\"chat\",\"messages\":\"messages.json\",\"message_mode\":\"thinking\"}\n"
    ));
    for index in 2..32 {
        cohort.push_str(&format!(
            "{{\"id\":\"literal-{index}\",\"token_ids\":[1,2]}}\n"
        ));
    }
    std::fs::write(&requests_path, cohort).unwrap();
    let requests = read_trace_full_batch_requests(&requests_path).unwrap();
    assert_eq!(requests.len(), 32);
    assert_eq!(requests[0].1.input.id, "literal");
    assert_eq!(
        requests[1].1.input.messages.as_deref(),
        Some(root.join("messages.json").as_path())
    );

    std::fs::write(
        &requests_path,
        "{\"id\":\"bad\",\"prompt\":\"x\",\"unknown\":true}\n",
    )
    .unwrap();
    assert!(read_trace_full_batch_requests(&requests_path).is_err());
    std::fs::write(&requests_path, "# comments are not JSON records\n").unwrap();
    assert!(read_trace_full_batch_requests(&requests_path).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn trace_batch_publication_exposes_only_one_complete_fresh_generation() {
    let root = std::env::temp_dir().join(format!(
        "qwen-trace-publish-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let output = root.join("cohort");
    let documents = vec![
        ("trace-0000.json", serde_json::json!({"value": "first"})),
        ("trace-0001.json", serde_json::json!({"value": "second"})),
    ];
    let mut publication = TraceFullBatchPublication::create(&output, 1024, 128).unwrap();
    for (path, document) in &documents {
        publication.stage_json_document(path, document).unwrap();
    }
    publication.publish(b"manifest").unwrap();
    assert_eq!(
        std::fs::read(output.join("trace-0000.json")).unwrap(),
        serde_json::to_vec(&documents[0].1).unwrap()
    );
    assert_eq!(
        std::fs::read(output.join(TRACE_FULL_BATCH_MANIFEST_NAME)).unwrap(),
        b"manifest"
    );
    assert!(TraceFullBatchPublication::create(&output, 1024, 128).is_err());
    assert_eq!(
        std::fs::read(output.join(TRACE_FULL_BATCH_MANIFEST_NAME)).unwrap(),
        b"manifest"
    );

    let rejected = root.join("rejected");
    {
        let mut publication = TraceFullBatchPublication::create(&rejected, 16, 8).unwrap();
        publication
            .stage_json_document("trace-0000.json", &"first")
            .unwrap();
        assert!(
            publication
                .stage_json_document("trace-0001.json", &"second")
                .is_err()
        );
    }
    assert!(!rejected.exists());

    let manifest_rejected = root.join("manifest-rejected");
    {
        let mut publication = TraceFullBatchPublication::create(&manifest_rejected, 64, 8).unwrap();
        publication
            .stage_json_document("trace-0000.json", &serde_json::json!({}))
            .unwrap();
        assert!(publication.publish(b"manifest-too-large").is_err());
    }
    assert!(!manifest_rejected.exists());

    let raced = root.join("raced");
    let mut publication = TraceFullBatchPublication::create(&raced, 64, 8).unwrap();
    publication
        .stage_json_document("trace-0000.json", &serde_json::json!({}))
        .unwrap();
    std::fs::create_dir(&raced).unwrap();
    assert!(publication.publish(b"manifest").is_err());
    assert!(raced.is_dir());
    assert_eq!(std::fs::read_dir(&raced).unwrap().count(), 0);
    std::fs::remove_dir_all(&raced).unwrap();

    let mut bounded_bytes = Vec::new();
    let mut bounded = ByteLimitedWriter::new(&mut bounded_bytes, 5);
    assert!(serde_json::to_writer(&mut bounded, &"too large").is_err());
    drop(bounded);
    assert!(bounded_bytes.len() <= 5);

    assert!(std::fs::read_dir(&root).unwrap().all(|entry| {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        !name.contains(".stage.")
    }));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn trace_stdout_format_defaults_follow_output_presence() {
    assert_eq!(
        effective_trace_stdout_format(None, false),
        TraceFullStdoutFormat::Json
    );
    assert_eq!(
        effective_trace_stdout_format(None, true),
        TraceFullStdoutFormat::Summary
    );
    assert_eq!(
        effective_trace_stdout_format(Some(TraceFullStdoutFormat::Summary), false),
        TraceFullStdoutFormat::Summary
    );
    assert_eq!(
        effective_trace_stdout_format(Some(TraceFullStdoutFormat::Json), true),
        TraceFullStdoutFormat::Json
    );
}

fn trace_score(rank: usize, token_id: u32) -> TraceFullTokenScore {
    TraceFullTokenScore {
        rank,
        token_id,
        token_display_lossy: format!("token-{token_id}"),
        token_piece_hex: format!("{token_id:02x}"),
        logit: -(rank as f32),
    }
}

#[test]
fn extracts_pinned_inventory_without_executing_pickle() {
    let matrix = [0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c];
    let pickle = b"inert fixture";
    let bytes = test_archive(&matrix, pickle);
    let digest = hex(&Sha256::digest(pickle));
    let spec = ArchiveSpec {
        root: "lens",
        layout: ArchiveLayout::LayerStorages,
        layer_count: 1,
        hidden_size: 2,
        matrix_bytes: matrix.len() as u64,
        data_pickle_sha256: &digest,
        serialization_id: None,
        identity_layer_index: Some(0),
    };
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    validate_archive(&mut archive, spec).unwrap();
    let mut output = Vec::new();
    let payload = extract_payload(&mut archive, spec, &mut output).unwrap();
    assert_eq!(output, matrix);
    assert_eq!(payload.byte_length, matrix.len() as u64);
}

#[test]
fn rejects_non_finite_half_storage() {
    let matrix = [0x00, 0x7c];
    let pickle = b"inert fixture";
    let bytes = test_archive(&matrix, pickle);
    let digest = hex(&Sha256::digest(pickle));
    let spec = ArchiveSpec {
        root: "lens",
        layout: ArchiveLayout::LayerStorages,
        layer_count: 1,
        hidden_size: 1,
        matrix_bytes: matrix.len() as u64,
        data_pickle_sha256: &digest,
        serialization_id: None,
        identity_layer_index: None,
    };
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    validate_archive(&mut archive, spec).unwrap();
    let error = extract_payload(&mut archive, spec, &mut Vec::new()).unwrap_err();
    assert!(error.to_string().contains("non-finite F16"));
}

#[test]
fn full_manifest_binds_every_published_claim_and_payload_digest() {
    let profile = PUBLISHED_PROFILES[0];
    let manifest = published_manifest(profile, canonical_payload());
    validate_manifest(&manifest).unwrap();
    assert_eq!(manifest.fit.skip_first, 16);
    assert_eq!(manifest.fit.accumulator_dtype.as_deref(), Some("float32"));

    let mut wrong_fit = published_manifest(profile, canonical_payload());
    wrong_fit.fit.accumulator_dtype = Some("bfloat16".into());
    assert!(validate_manifest(&wrong_fit).is_err());

    let mut wrong_payload = published_manifest(profile, canonical_payload());
    wrong_payload.payload.blake3 = "00".repeat(32);
    assert!(validate_manifest(&wrong_payload).is_err());
}

#[test]
fn released_qwen36_pair_has_matched_recipe_and_distinct_methods() {
    let j_profile = *PUBLISHED_PROFILES
        .iter()
        .find(|profile| profile.id == PublishedProfileId::Qwen36J)
        .unwrap();
    let r_profile = *PUBLISHED_PROFILES
        .iter()
        .find(|profile| profile.id == PublishedProfileId::Qwen36R)
        .unwrap();
    let j = published_manifest(
        j_profile,
        FullPayload {
            blake3: j_profile.expected_payload_blake3.into(),
            ..canonical_payload()
        },
    );
    let r = published_manifest(
        r_profile,
        FullPayload {
            blake3: r_profile.expected_payload_blake3.into(),
            ..canonical_payload()
        },
    );
    validate_manifest(&j).unwrap();
    validate_manifest(&r).unwrap();
    assert_eq!(j.transport.method, "j");
    assert_eq!(r.transport.method, "r");
    assert_eq!(j.transport.target_layer, 62);
    assert_eq!(j.fit.n_prompts, 25);
    assert_eq!(j.fit.max_sequence_length, 128);
    assert_eq!(j.fit.skip_first, 4);
    assert_eq!(j.fit.dataset, r.fit.dataset);
}

#[test]
fn neuronpedia_qwen36_j_preserves_its_distinct_fit_identity() {
    let profile = *PUBLISHED_PROFILES
        .iter()
        .find(|profile| profile.id == PublishedProfileId::Qwen36NeuronpediaJ1000)
        .unwrap();
    let manifest = published_manifest(
        profile,
        FullPayload {
            blake3: profile.expected_payload_blake3.into(),
            ..canonical_payload()
        },
    );
    validate_manifest(&manifest).unwrap();
    assert_eq!(manifest.transport.target_layer, 63);
    assert_eq!(manifest.fit.n_prompts, 1_000);
    assert_eq!(manifest.fit.dataset, "Salesforce/wikitext");
    assert_eq!(
        manifest.source.sha256,
        "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
    );
    assert_ne!(
        manifest.payload.blake3,
        PUBLISHED_PROFILES
            .iter()
            .find(|candidate| candidate.id == PublishedProfileId::Qwen36J)
            .unwrap()
            .expected_payload_blake3
    );
}

#[test]
fn trace_manifest_validates_geometry_and_pinned_digest_without_rescanning_payload() {
    let mut manifest = published_manifest(PUBLISHED_PROFILES[0], canonical_payload());
    manifest.provenance.build_source_state = "not-used-by-trace-full".into();
    validate_trace_full_manifest(&manifest).unwrap();

    manifest.payload.blake3 = "00".repeat(32);
    assert!(validate_trace_full_manifest(&manifest).is_err());
    manifest.payload.blake3 = PUBLISHED_PROFILES[0].expected_payload_blake3.into();
    manifest.payload.byte_length -= 2;
    assert!(validate_trace_full_manifest(&manifest).is_err());
}

#[test]
fn archive_validation_binds_pickle_digest() {
    let matrix = [0x00, 0x3c];
    let bytes = test_archive(&matrix, b"unexpected pickle");
    let digest = "00".repeat(32);
    let spec = ArchiveSpec {
        root: "lens",
        layout: ArchiveLayout::LayerStorages,
        layer_count: 1,
        hidden_size: 1,
        matrix_bytes: matrix.len() as u64,
        data_pickle_sha256: &digest,
        serialization_id: None,
        identity_layer_index: None,
    };
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    assert!(validate_archive(&mut archive, spec).is_err());
}

#[test]
fn layer_aggregates_preserve_layer_heterogeneity() {
    let directions = vec![
        TransferDirection {
            source_layer: 0,
            token_id: 1,
            cosine_similarity: 0.25,
            relative_l2_error: 1.0,
            norm_ratio_public_over_native: 0.5,
            public_norm: 1.0,
            native_norm: 2.0,
            maximum_absolute_difference: 1.0,
        },
        TransferDirection {
            source_layer: 62,
            token_id: 1,
            cosine_similarity: 0.99,
            relative_l2_error: 0.1,
            norm_ratio_public_over_native: 1.0,
            public_norm: 2.0,
            native_norm: 2.0,
            maximum_absolute_difference: 0.1,
        },
    ];
    let layers = aggregate_layer_metrics(&directions).unwrap();
    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].source_layer, 0);
    assert_eq!(layers[0].mean_cosine_similarity, 0.25);
    assert_eq!(layers[1].source_layer, 62);
    assert_eq!(layers[1].mean_cosine_similarity, 0.99);
}

#[test]
fn full_readout_requires_explicit_safe_input_contract() {
    let mut args = test_read_full_args();
    validate_read_full_args(&args).unwrap();

    args.max_tokens = 262_144;
    validate_read_full_args(&args).unwrap();
    args.max_tokens = 0;
    assert!(validate_read_full_args(&args).is_err());
    args.max_tokens = 256;

    args.allow_unvalidated_transfer = false;
    assert!(
        validate_read_full_args(&args)
            .unwrap_err()
            .to_string()
            .contains("unvalidated")
    );
    args.allow_unvalidated_transfer = true;

    args.top_k = 26;
    assert!(validate_read_full_args(&args).is_err());
    args.top_k = 10;

    args.token_ids = vec![1];
    assert!(validate_read_full_args(&args).is_err());
    args.prompt = None;
    validate_read_full_args(&args).unwrap();

    args.no_special_tokens = true;
    assert!(validate_read_full_args(&args).is_err());
    args.no_special_tokens = false;
    args.token_ids.clear();
    assert!(validate_read_full_args(&args).is_err());

    args.prompt = Some(String::new());
    assert!(validate_read_full_args(&args).is_err());
    args.prompt = Some("hello".into());
    args.position = Some(args.max_tokens);
    assert!(validate_read_full_args(&args).is_err());
    args.prompt = None;
    args.token_ids = vec![1, 2];
    args.position = Some(2);
    assert!(validate_read_full_args(&args).is_err());
}

#[test]
fn trace_full_arguments_enforce_the_bounded_exact_input_contract() {
    let mut args = test_trace_full_args();
    validate_trace_full_args(&args).unwrap();

    args.top_k = 26;
    assert!(validate_trace_full_args(&args).is_err());
    args.top_k = 8;
    args.max_tokens = Some(262_144);
    validate_trace_full_args(&args).unwrap();
    args.max_tokens = Some(0);
    assert!(validate_trace_full_args(&args).is_err());
    args.max_tokens = Some(MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS);

    args.prompt = None;
    args.messages = Some("messages.json".into());
    args.no_special_tokens = true;
    assert!(validate_trace_full_args(&args).is_err());
    args.no_special_tokens = false;
    validate_trace_full_args(&args).unwrap();

    args.messages = None;
    args.token_ids = Some(Vec::new());
    assert!(validate_trace_full_args(&args).is_err());
    args.token_ids = Some(vec![1, 2]);
    validate_trace_full_args(&args).unwrap();
    args.prompt = Some("also set".into());
    assert!(validate_trace_full_args(&args).is_err());
}

#[test]
fn trace_position_tiles_cover_logical_context_without_exposing_tile_width() {
    assert!(trace_position_tiles(0, 128).is_err());
    assert!(trace_position_tiles(1, 0).is_err());
    for (positions, expected) in [
        (1, vec![0..1]),
        (128, vec![0..128]),
        (129, vec![0..128, 128..129]),
        (493, vec![0..128, 128..256, 256..384, 384..493]),
    ] {
        assert_eq!(trace_position_tiles(positions, 128).unwrap(), expected);
    }
}

#[test]
fn trace_document_budget_accepts_tool_transcripts_and_rejects_impossible_artifacts_early() {
    let tool_trace_bytes = ensure_trace_document_budget(
        493,
        51,
        8,
        32,
        6_656,
        MAX_TRACE_DOCUMENT_BYTES,
        "test trace",
    )
    .unwrap();
    assert!(
        tool_trace_bytes
            .checked_mul(32)
            .is_some_and(|bytes| bytes <= MAX_TRACE_FULL_BATCH_DOCUMENT_BYTES)
    );
    assert!(
        ensure_trace_document_budget(
            8_192,
            51,
            8,
            0,
            6_656,
            MAX_TRACE_DOCUMENT_BYTES,
            "test trace",
        )
        .is_err()
    );
    let vector_count = MAX_TRACE_DOCUMENT_BYTES / (6_656 * TRACE_VECTOR_VALUE_RESERVE_BYTES) + 1;
    assert!(
        ensure_trace_document_budget(
            1,
            1,
            1,
            vector_count,
            6_656,
            MAX_TRACE_DOCUMENT_BYTES,
            "test trace",
        )
        .is_err()
    );
    assert_eq!(
        trace_host_result_reserve_bytes(2, MAX_TRACE_DOCUMENT_BYTES).unwrap(),
        3 * MAX_TRACE_DOCUMENT_BYTES as u64
    );
}

#[test]
fn projected_full_token_banks_are_priced_by_shape() {
    let logical_bytes = 82_575_360 + 64 * 4 + 63 * 4 + 128;
    let page_size = host_page_size_bytes().unwrap();
    let priced = projected_full_token_direction_retained_bytes(63, 64, 5_120).unwrap();
    assert!(priced >= logical_bytes && priced <= logical_bytes + 4 * (page_size - 1));
    assert!(projected_full_token_direction_retained_bytes(usize::MAX, 2, 5_120).is_err());
}

#[test]
fn projected_full_token_tiles_preserve_layer_major_token_order() {
    let mut values = vec![f32::NAN; 2 * 5 * 2];
    write_projected_full_token_tile(&mut values, 0, 5, 0, 2, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    write_projected_full_token_tile(&mut values, 0, 5, 3, 2, &[6.0, 7.0, 8.0, 9.0]).unwrap();
    write_projected_full_token_tile(
        &mut values,
        1,
        5,
        0,
        2,
        &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    )
    .unwrap();
    write_projected_full_token_tile(&mut values, 1, 5, 3, 2, &[16.0, 17.0, 18.0, 19.0]).unwrap();
    assert_eq!(
        values,
        (0..20).map(|value| value as f32).collect::<Vec<_>>()
    );
    assert!(write_projected_full_token_tile(&mut values, 1, 5, 4, 2, &[0.0; 4]).is_err());
}

#[test]
fn trace_vector_cells_parse_strict_unsigned_coordinates() {
    assert_eq!(
        "31:127".parse::<TraceFullVectorCell>().unwrap(),
        TraceFullVectorCell {
            source_layer: 31,
            source_position: 127,
        }
    );
    for invalid in ["", "31", "31:", ":1", "31:1:2", "-1:2", "1:+2", "a:2"] {
        assert!(
            invalid.parse::<TraceFullVectorCell>().is_err(),
            "unexpectedly accepted {invalid:?}"
        );
    }
}

#[test]
fn trace_vector_cells_require_selected_layers_and_valid_positions() {
    let cells = [
        TraceFullVectorCell {
            source_layer: 31,
            source_position: 4,
        },
        TraceFullVectorCell {
            source_layer: 0,
            source_position: 2,
        },
        TraceFullVectorCell {
            source_layer: 31,
            source_position: 1,
        },
    ];
    let grouped = group_trace_full_vector_cells(&cells, &[0, 31], 5).unwrap();
    assert_eq!(grouped[&0], [2]);
    assert_eq!(grouped[&31], [1, 4]);

    assert!(group_trace_full_vector_cells(&cells, &[0], 5).is_err());
    assert!(group_trace_full_vector_cells(&cells, &[0, 31], 4).is_err());
}

#[test]
fn trace_vector_cells_are_resource_admitted_and_unique() {
    let mut args = test_trace_full_args();
    args.vectors = vec![
        TraceFullVectorCell {
            source_layer: 31,
            source_position: 2,
        };
        2
    ];
    assert!(validate_trace_full_args(&args).is_err());

    args.vectors = (0..33)
        .map(|source_position| TraceFullVectorCell {
            source_layer: 31,
            source_position,
        })
        .collect();
    validate_trace_full_args(&args).unwrap();
}

#[test]
fn trace_occurrences_count_each_token_once_per_cell_and_sort_deterministically() {
    let cells = vec![
        TraceFullCell {
            source_layer: 2,
            source_position: 0,
            source_token_id: 10,
            predicts_position: 1,
            top_k: vec![trace_score(0, 7), trace_score(1, 5)],
        },
        TraceFullCell {
            source_layer: 2,
            source_position: 1,
            source_token_id: 11,
            predicts_position: 2,
            top_k: vec![trace_score(0, 5), trace_score(1, 7), trace_score(2, 7)],
        },
        TraceFullCell {
            source_layer: 0,
            source_position: 0,
            source_token_id: 10,
            predicts_position: 1,
            top_k: vec![trace_score(0, 7), trace_score(1, 9)],
        },
    ];
    let occurrences = aggregate_trace_full_occurrences(&cells, &[2, 0]);
    assert_eq!(
        occurrences.global,
        [
            TraceFullOccurrence {
                token_id: 7,
                count: 3,
                top1_count: 2,
                best_rank: 0,
            },
            TraceFullOccurrence {
                token_id: 5,
                count: 2,
                top1_count: 1,
                best_rank: 0,
            },
            TraceFullOccurrence {
                token_id: 9,
                count: 1,
                top1_count: 0,
                best_rank: 1,
            },
        ]
    );
    assert_eq!(occurrences.per_layer[0].source_layer, 2);
    assert_eq!(
        occurrences.per_layer[0]
            .tokens
            .iter()
            .map(|occurrence| occurrence.token_id)
            .collect::<Vec<_>>(),
        [5, 7]
    );
    assert_eq!(occurrences.per_layer[1].source_layer, 0);
    assert_eq!(
        occurrences.per_layer[1]
            .tokens
            .iter()
            .map(|occurrence| occurrence.token_id)
            .collect::<Vec<_>>(),
        [7, 9]
    );
}
