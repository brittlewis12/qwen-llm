use super::*;

pub(crate) fn generic_bound_trace_json() -> serde_json::Value {
    let fixture = crate::linear_transport::tests::fixture("unknown-cpu-fixture-method", 2, 91);
    let path = fixture.0.join("model.gguf");
    write_cpu_gguf(&path, "qwen35", 2, "test-tokenizer", false);
    let gguf = GgufFile::open(&path).unwrap();
    let cache = fixture.0.join("cache");
    let content = hex(
        &checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(&cache))
            .unwrap()
            .content_id,
    );
    let (locator, tokenizer) =
        qwen_llm::runtime::opened_gguf_lightweight_identity_parts(&gguf).unwrap();
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["exact_binding"] = serde_json::json!({
        "gguf_content_blake3":content, "tokenizer_metadata_id":format!("{tokenizer:016x}")})
    });
    let bound = FullAccess::open(&fixture.0, true)
        .unwrap()
        .bind_opened(&gguf, FullExecutionMode::Packed, Some(&cache), false)
        .unwrap();
    let occurrence = serde_json::json!({"token_id":2,"count":1,"top1_count":1,"best_rank":0});
    serde_json::json!({"schema":"qwen.lens.trace","schema_version":3,
        "producer":{"build_commit":"fixture-build","build_dirty":"dirty","build_source_state":"fixture-source"},
        "deployed_model":{"path":path,"locator_scheme":WORKSPACE_LENS_IDENTITY_SCHEME,"locator_id":format!("{locator:016x}"),
            "content_authenticated":true,"content_blake3":content,"architecture":"qwen35","n_layers":3,"hidden_size":2,"vocab_size":32},
        "tokenizer":{"metadata_id":format!("{tokenizer:016x}"),"model":"test-tokenizer","pretokenizer":null},
        "lens":bound.trace_summary(),
        "score_semantics":{"kind":"logit","normalization":"deployed_output_rmsnorm_and_lm_head","candidate_universe":"full_model_vocabulary","softmax_applied":false},
        "execution_mode":"cpu_fixture_no_inference","input_source":"token_ids","add_special_tokens":null,
        "input_token_ids":[1],"input_tokens":[{"position":0,"token_id":1,"token_display_lossy":"one","token_piece_hex":"6f6e65"}],
        "rendering":{"renderer":"literal_token_ids","generation_mode":null,"spans":[]},"selected_layers":[0],"top_k":1,
        "cells":[{"source_layer":0,"source_position":0,"source_token_id":1,"predicts_position":1,
            "top_k":[{"rank":0,"token_id":2,"token_display_lossy":"two","token_piece_hex":"74776f","logit":1.0}]}],
        "timing":{"matrix_read_wall_ms":0.0,"readout_gpu_ms":0.0,"readout_command_wall_ms":0.0,"trace_execution_wall_ms":0.0},
        "occurrences":{"global":[occurrence.clone()],"per_layer":[{"source_layer":0,"tokens":[occurrence]}]},
        "batch":{"batch_schema":"qwen.lens.trace_batch","request_id":"fixture-request","request_index":1,"request_count":2,"aggregate_rows":2,
            "shared_timing_fields":["matrix_read_wall_ms","readout_gpu_ms","readout_command_wall_ms","trace_execution_wall_ms"]}})
}

/// Small real GGUF descriptors for CPU preflight; never loaded onto Metal.
pub(crate) fn write_cpu_gguf(
    path: &Path,
    family: &str,
    hidden: u64,
    token_claim: &str,
    bad_head: bool,
) {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    let mut metadata = Vec::new();
    for (key, text) in [
        ("general.architecture", family),
        ("tokenizer.ggml.model", token_claim),
    ] {
        let mut entry = Vec::new();
        string(&mut entry, key);
        entry.extend_from_slice(&8u32.to_le_bytes());
        string(&mut entry, text);
        metadata.push(entry);
    }
    for (key, value) in [
        ("block_count", 3),
        ("embedding_length", hidden),
        ("feed_forward_length", 4),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("full_attention_interval", 1),
        ("ssm.state_size", 2),
        ("ssm.time_step_rank", 1),
        ("ssm.group_count", 1),
        ("ssm.conv_kernel", 4),
        ("expert_count", 2),
        ("expert_used_count", 1),
        ("expert_feed_forward_length", 4),
        ("expert_shared_feed_forward_length", 4),
    ] {
        let mut entry = Vec::new();
        string(&mut entry, &format!("{family}.{key}"));
        entry.extend_from_slice(&10u32.to_le_bytes());
        entry.extend_from_slice(&value.to_le_bytes());
        metadata.push(entry);
    }
    let mut tensors = vec![
        ("token_embd.weight".to_owned(), vec![hidden, 32]),
        ("output_norm.weight".into(), vec![hidden]),
        (
            "output.weight".into(),
            vec![hidden + u64::from(bad_head), 32],
        ),
    ];
    for layer in 0..3 {
        for (name, shape) in [
            ("attn_norm", vec![hidden]),
            ("post_attention_norm", vec![hidden]),
            ("attn_q", vec![hidden, 4]),
            ("attn_k", vec![hidden, 2]),
            ("attn_v", vec![hidden, 2]),
            ("attn_output", vec![2, hidden]),
            ("attn_q_norm", vec![2]),
            ("attn_k_norm", vec![2]),
        ] {
            tensors.push((format!("blk.{layer}.{name}.weight"), shape));
        }
        let ffn = if family == "qwen35moe" {
            vec![
                ("ffn_gate_inp", vec![hidden, 2]),
                ("ffn_gate_exps", vec![hidden, 4, 2]),
                ("ffn_up_exps", vec![hidden, 4, 2]),
                ("ffn_down_exps", vec![4, hidden, 2]),
                ("ffn_gate_inp_shexp", vec![hidden]),
                ("ffn_gate_shexp", vec![hidden, 4]),
                ("ffn_up_shexp", vec![hidden, 4]),
                ("ffn_down_shexp", vec![4, hidden]),
            ]
        } else {
            vec![
                ("ffn_gate", vec![hidden, 4]),
                ("ffn_up", vec![hidden, 4]),
                ("ffn_down", vec![4, hidden]),
            ]
        };
        for (name, shape) in ffn {
            tensors.push((format!("blk.{layer}.{name}.weight"), shape));
        }
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for entry in metadata {
        out.extend(entry);
    }
    let mut payload = Vec::new();
    for (name, shape) in tensors {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut out, &name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in &shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        for _ in 0..shape.iter().product::<u64>() {
            payload.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    out.extend(payload);
    std::fs::write(path, out).unwrap();
}

#[test]
fn cpu_binding_rejects_geometry_head_and_family_before_metal() {
    let fixture = crate::linear_transport::tests::fixture("cpu-only-fit", 2, 11);
    let model = fixture.0.join("model.gguf");
    for (family, hidden, bad_head, succeeds) in [
        ("qwen35", 2, false, true),
        ("qwen35", 3, false, false),
        ("qwen35", 2, true, false),
        ("unsupported", 2, false, false),
    ] {
        write_cpu_gguf(&model, family, hidden, "first", bad_head);
        let gguf = GgufFile::open(&model).unwrap();
        let access = FullAccess::open(&fixture.0, false).unwrap();
        let result = access.bind_opened(&gguf, FullExecutionMode::Scalar, None, true);
        assert_eq!(
            result.is_ok(),
            succeeds,
            "{family} {hidden} {bad_head}: {:?}",
            result.err()
        );
    }
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["architecture"] = "qwen35moe".into()
    });
    write_cpu_gguf(&model, "qwen35moe", 2, "first", false);
    let gguf = GgufFile::open(&model).unwrap();
    for mode in [
        FullExecutionMode::Scalar,
        FullExecutionMode::Projection,
        FullExecutionMode::Packed,
    ] {
        let result = FullAccess::open(&fixture.0, false)
            .unwrap()
            .bind_opened(&gguf, mode, None, true);
        assert_eq!(
            result.is_ok(),
            mode != FullExecutionMode::Packed,
            "{mode:?}: {:?}",
            result.err()
        );
    }
}

#[test]
fn cpu_exact_binding_uses_native_identity_and_rejects_override_mismatches() {
    let fixture = crate::linear_transport::tests::fixture("exact-cpu-fit", 2, 12);
    let model = fixture.0.join("model.gguf");
    write_cpu_gguf(&model, "qwen35", 2, "first", false);
    let gguf = GgufFile::open(&model).unwrap();
    let cache = fixture.0.join("identity-cache");
    let content = hex(
        &checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(&cache))
            .unwrap()
            .content_id,
    );
    let tokenizer = format!(
        "{:016x}",
        qwen_llm::runtime::tokenizer_metadata_identity(&gguf)
    );
    assert_eq!(
        qwen_llm::runtime::opened_gguf_lightweight_identity_parts(&gguf)
            .unwrap()
            .1,
        qwen_llm::runtime::tokenizer_metadata_identity(&gguf)
    );
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["exact_binding"] =
            serde_json::json!({"gguf_content_blake3":content,"tokenizer_metadata_id":tokenizer})
    });
    assert!(
        FullAccess::open(&fixture.0, false)
            .unwrap()
            .bind_opened(&gguf, FullExecutionMode::Scalar, None, true)
            .is_ok()
    );
    let bound = FullAccess::open(&fixture.0, false)
        .unwrap()
        .bind_opened(&gguf, FullExecutionMode::Scalar, Some(&cache), false)
        .unwrap();
    assert_eq!(
        bound.runtime_binding.as_ref().unwrap()["binding_phase"],
        "cpu_before_metal"
    );
    for key in ["gguf_content_blake3", "tokenizer_metadata_id"] {
        let original = crate::linear_transport::VerifiedTransport::open(&fixture.0)
            .unwrap()
            .original_manifest()["model"]["exact_binding"][key]
            .clone();
        crate::linear_transport::tests::modify(&fixture, |v| {
            v["model"]["exact_binding"][key] = if key == "gguf_content_blake3" {
                "0".repeat(64).into()
            } else {
                "0".repeat(16).into()
            }
        });
        assert!(
            FullAccess::open(&fixture.0, false)
                .unwrap()
                .bind_opened(&gguf, FullExecutionMode::Scalar, Some(&cache), true)
                .is_err()
        );
        crate::linear_transport::tests::modify(&fixture, |v| {
            v["model"]["exact_binding"][key] = original
        });
    }
    let changed_path = fixture.0.join("changed-tokenizer.gguf");
    write_cpu_gguf(&changed_path, "qwen35", 2, "changed-tokenizer", false);
    let changed = GgufFile::open(&changed_path).unwrap();
    let changed_content = hex(&checkpoint_content_identity(
        &changed,
        &CheckpointIdentityCache::new(&cache),
    )
    .unwrap()
    .content_id);
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["exact_binding"]["gguf_content_blake3"] = changed_content.into()
    });
    assert!(
        FullAccess::open(&fixture.0, false)
            .unwrap()
            .bind_opened(&changed, FullExecutionMode::Scalar, Some(&cache), true)
            .is_err()
    );
}

#[test]
fn shared_data_access_preserves_unknown_recipes_geometry_order_and_claims() {
    for (method, h, seed) in [
        ("future-recipe-2026-a", 2, 100),
        ("independent-fit-b", 5, 300),
    ] {
        let fixture = crate::linear_transport::tests::fixture(method, h, seed);
        for trace in [false, true] {
            let mut access = FullAccess::open(&fixture.0, trace).unwrap();
            assert!(access.is_data());
            assert_eq!(access.transport.source_layers, [2, 0]);
            assert_eq!(access.transport.hidden_size as usize, h);
            assert_eq!(access.transport.method, method);
            assert!(access.acknowledge_transfer(false).is_err());
            access.acknowledge_transfer(true).unwrap();
            for layer in [0, 2] {
                assert_eq!(access.read_matrix(layer).unwrap().len(), h * h * 2);
            }
            assert!(access.read_matrix(1).is_err());
            let artifact = access.readout_artifact(&fixture.0).unwrap();
            assert_eq!(artifact["method"], method);
            assert!(artifact.get("fit_n_prompts").is_none());
            assert!(artifact.get("fitted_checkpoint").is_none());
            assert!(artifact.get("source_repository").is_none());
            assert_eq!(
                artifact["producer_contract"]["qualification"]["validated"],
                true
            );
            let trace = serde_json::to_value(access.trace_summary()).unwrap();
            assert_eq!(trace["method"], method);
            assert!(trace.get("source_repository").is_none());
            assert_eq!(trace["producer_contract"], artifact["producer_contract"]);
        }
    }
}

#[test]
fn exact_binding_hashes_retained_bytes_despite_sidecars_and_poisoned_auto_cache() {
    use qwen_llm::checkpoint_identity::{
        checkpoint_content_identity_without_weight_hashing, verified_checkpoint_content_identity,
    };
    use std::io::Write;
    let fixture = crate::linear_transport::tests::fixture("unregistered-byte-bound-fit", 2, 73);
    let path = fixture.0.join("model.gguf");
    write_cpu_gguf(&path, "qwen35", 2, "test-tokenizer", false);
    let gguf = GgufFile::open(&path).unwrap();
    let reference = checkpoint_content_identity(
        &gguf,
        &CheckpointIdentityCache::new(fixture.0.join("reference-cache")),
    )
    .unwrap();
    let actual = verified_checkpoint_content_identity(&gguf).unwrap();
    assert_eq!(actual.content_id, reference.content_id);
    assert_eq!(actual.bytes_hashed, std::fs::metadata(&path).unwrap().len());
    let sidecars = fixture.0.join(".cache/huggingface/download");
    std::fs::create_dir_all(&sidecars).unwrap();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        + 1.0;
    std::fs::write(
        sidecars.join("model.gguf.metadata"),
        format!("{}\n{}\n{timestamp}\n", "a".repeat(40), "b".repeat(64)),
    )
    .unwrap();
    let cache_path = fixture.0.join("declared-cache");
    let cache = CheckpointIdentityCache::new(&cache_path);
    let declared = checkpoint_content_identity_without_weight_hashing(&gguf, &cache).unwrap();
    assert_eq!(declared.bytes_hashed, 0);
    assert_ne!(declared.content_id, actual.content_id);
    assert_eq!(
        checkpoint_content_identity(&gguf, &cache)
            .unwrap()
            .content_id,
        declared.content_id
    );
    let tokenizer = format!(
        "{:016x}",
        qwen_llm::runtime::tokenizer_metadata_identity(&gguf)
    );
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["exact_binding"] = serde_json::json!({
        "gguf_content_blake3":hex(&declared.content_id),"tokenizer_metadata_id":tokenizer})
    });
    assert!(
        FullAccess::open(&fixture.0, false)
            .unwrap()
            .bind_opened(&gguf, FullExecutionMode::Scalar, Some(&cache_path), true)
            .is_err()
    );
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"]["exact_binding"]["gguf_content_blake3"] = hex(&actual.content_id).into()
    });
    let bound = FullAccess::open(&fixture.0, false)
        .unwrap()
        .bind_opened(&gguf, FullExecutionMode::Scalar, Some(&cache_path), false)
        .unwrap();
    let report = bound.runtime_binding.as_ref().unwrap();
    assert_eq!(
        report["content_identity_provenance"],
        "retained_bytes_hashed"
    );
    assert_eq!(report["content_bytes_hashed"], actual.bytes_hashed);
    assert_eq!(report["gguf_content_blake3"], hex(&actual.content_id));
    crate::linear_transport::tests::modify(&fixture, |v| {
        v["model"].as_object_mut().unwrap().remove("exact_binding");
    });
    let unbound = FullAccess::open(&fixture.0, false)
        .unwrap()
        .bind_opened(&gguf, FullExecutionMode::Scalar, Some(&cache_path), true)
        .unwrap();
    assert!(unbound.bound_content_blake3().is_none());
    assert_eq!(
        unbound.runtime_binding.as_ref().unwrap()["content_bytes_hashed"],
        0
    );
    assert_eq!(
        unbound.validation_status(),
        "source_deployment_equivalence_unverified"
    );
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(verified_checkpoint_content_identity(&gguf).is_err());
}

#[test]
fn independent_python_f16_fixtures_share_the_native_consumer_adapter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear_transport");
    for (name, method, hidden, sources) in [
        ("seed41_h2", "experimental/seed41", 2, vec![2, 0]),
        ("seed7_h3", "orthogonal-regression+beta", 3, vec![3, 1, 2]),
    ] {
        let directory = root.join(name);
        let mut access = FullAccess::open(&directory, false).unwrap();
        assert_eq!(access.transport.method, method);
        assert_eq!(access.transport.hidden_size, hidden);
        assert_eq!(access.transport.source_layers, sources);
        for &layer in sources.iter().rev() {
            assert_eq!(
                access.read_matrix(layer).unwrap().len(),
                hidden as usize * hidden as usize * 2
            );
        }
        assert!(access.acknowledge_transfer(false).is_err());
        let trace: crate::lens_inspect::LensSummary =
            serde_json::from_value(serde_json::to_value(access.trace_summary()).unwrap()).unwrap();
        assert_eq!(trace.method, method);
    }
}

#[test]
fn shared_access_rejects_corrupted_unused_matrix_before_runtime() {
    use std::io::{Seek, SeekFrom, Write};
    let fixture = crate::linear_transport::tests::fixture("unused-corrupt", 2, 12);
    let mut access = FullAccess::open(&fixture.0, false).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.0.join("transport.f16le"))
        .unwrap();
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&[0, 0]).unwrap();
    assert!(FullAccess::open(&fixture.0, false).is_err());
    assert!(FullAccess::open(&fixture.0, true).is_err());
    assert!(access.read_matrix(0).is_err());
}

#[test]
fn shared_access_dispatch_keeps_published_manifest_legacy() {
    let fixture = crate::linear_transport::tests::fixture("replace-manifest", 2, 12);
    let manifest = published_manifest(PUBLISHED_PROFILES[0], canonical_payload());
    std::fs::write(
        fixture.0.join(FULL_MANIFEST_NAME),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(!FullAccess::is_data_directory(&fixture.0).unwrap());
    validate_manifest(&manifest).unwrap();
    let summary = serde_json::to_value(trace_full_lens_summary(&manifest)).unwrap();
    assert!(summary.get("producer_contract").is_none());
    assert!(summary.get("runtime_binding").is_none());
    assert_eq!(summary["source_repository"], manifest.source.repository);
}

#[test]
#[ignore = "requires QWEN_LINEAR_TRANSPORT_SMALL_GGUF and QWEN_LINEAR_TRANSPORT_SMALL_ARTIFACT; uses Metal"]
fn generic_unknown_small_geometry_scalar_native_readout() -> Result<()> {
    let model = PathBuf::from(std::env::var("QWEN_LINEAR_TRANSPORT_SMALL_GGUF")?);
    let directory = PathBuf::from(std::env::var("QWEN_LINEAR_TRANSPORT_SMALL_ARTIFACT")?);
    let access = FullAccess::open(&directory, false)?;
    ensure!(access.is_data() && access.transport.hidden_size < 5120);
    ensure!(!matches!(
        access.transport.method.as_str(),
        "J" | "R" | "j" | "r"
    ));
    let layers = access
        .transport
        .source_layers
        .iter()
        .rev()
        .copied()
        .collect();
    read_full(ReadFullArgs {
        model,
        full_lens: Some(directory),
        logit_lens: false,
        prompt: None,
        token_ids: vec![0],
        no_special_tokens: false,
        position: None,
        layers,
        top_k: 1,
        max_tokens: 1,
        identity_cache: PathBuf::from("unused-for-data-contract"),
        allow_unvalidated_transfer: true,
        include_vector: true,
        full_output: None,
        output: None,
    })
}
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
        full_lens: Some("full-lens".into()),
        logit_lens: false,
        full_output: None,
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
        distribution_summaries: false,
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
    args.allow_unvalidated_transfer = true;
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
        false,
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
            false,
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
            false,
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
fn distribution_summary_trace_default_compatibility_and_budget() {
    assert!(!test_trace_full_args().distribution_summaries);
    let mut cell = TraceFullCell {
        source_layer: 0,
        source_position: 0,
        source_token_id: 1,
        predicts_position: 1,
        top_k: vec![],
        distribution_summary: None,
    };
    let old = serde_json::to_value(&cell).unwrap();
    assert_eq!(
        old,
        serde_json::json!({
            "source_layer":0,"source_position":0,"source_token_id":1,"predicts_position":1,"top_k":[]
        })
    );
    let old_size = serde_json::to_vec_pretty(&cell).unwrap().len();
    cell.distribution_summary = Some(qwen_llm::workspace_lens::WorkspaceLensDistributionSummary {
        vocab_size: u32::MAX,
        entropy_nats: 1.2345678901234567,
        logsumexp: f64::MAX,
        top_k_mass: 0.12345678901234567,
        score_max: -f64::MAX,
        score_mean: -f64::MAX,
        score_variance_population: f64::MAX,
    });
    assert!(
        serde_json::to_vec_pretty(&cell).unwrap().len() - old_size
            <= TRACE_DISTRIBUTION_SUMMARY_RESERVE_BYTES
    );
    assert!(
        serde_json::to_value(&cell)
            .unwrap()
            .get("distribution_summary")
            .is_some()
    );
    let base =
        ensure_trace_document_budget(128, 63, 8, false, 0, 4096, usize::MAX, "test").unwrap();
    let with = ensure_trace_document_budget(128, 63, 8, true, 0, 4096, usize::MAX, "test").unwrap();
    assert_eq!(
        with - base,
        128 * 63 * TRACE_DISTRIBUTION_SUMMARY_RESERVE_BYTES
    );
    assert!(ensure_trace_document_budget(128, 63, 8, true, 0, 4096, base, "test").is_err());
    assert!(
        ensure_trace_document_budget(usize::MAX, 63, 8, true, 0, 4096, usize::MAX, "test").is_err()
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
            distribution_summary: None,
        },
        TraceFullCell {
            source_layer: 2,
            source_position: 1,
            source_token_id: 11,
            predicts_position: 2,
            top_k: vec![trace_score(0, 5), trace_score(1, 7), trace_score(2, 7)],
            distribution_summary: None,
        },
        TraceFullCell {
            source_layer: 0,
            source_position: 0,
            source_token_id: 10,
            predicts_position: 1,
            top_k: vec![trace_score(0, 7), trace_score(1, 9)],
            distribution_summary: None,
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
