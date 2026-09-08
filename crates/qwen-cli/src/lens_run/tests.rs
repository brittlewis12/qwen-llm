use super::*;

#[test]
fn all_plan_artifacts_are_cpu_bound_before_any_projection_can_start() {
    let first = crate::linear_transport::tests::fixture("first-valid", 2, 1);
    let second = crate::linear_transport::tests::fixture("second-wrong-geometry", 3, 2);
    let model = first.0.join("model.gguf");
    crate::full_lens::write_cpu_gguf(&model, "qwen35", 2, "first", false);
    let gguf = GgufFile::open(&model).unwrap();
    let plan: LensPlan = serde_json::from_value(serde_json::json!({"version":1,
        "lenses":[{"kind":"linear_transport","id":"first","artifact":first.0,"token_ids":[1],"allow_unvalidated_transfer":true},
            {"kind":"linear_transport","id":"second","artifact":second.0,"token_ids":[1],"allow_unvalidated_transfer":true}],
        "directions":[],"operations":[],"readouts":[
            {"id":"a","lens":"first","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"all"}},"top_k":1},
            {"id":"b","lens":"second","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"all"}},"top_k":1}]})).unwrap();
    validate_plan(&plan).unwrap();
    let opened = open_full_transports(&plan, Path::new(".")).unwrap();
    let mut projection_started = false;
    let result: Result<()> = (|| {
        let _bound = bind_full_transports(opened, &plan, Path::new("."), &gguf, None)?;
        projection_started = true;
        Ok(())
    })();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("geometry mismatch")
    );
    assert!(!projection_started);
    let mut mixed = plan.clone();
    for legacy in [
        LensDefinition::NativeSelected {
            id: "second".into(),
            artifact: second.0.join("missing-native"),
        },
        LensDefinition::WorkspaceTemplate {
            id: "second".into(),
            weights: second.0.join("missing-weights"),
            labels: second.0.join("missing-labels"),
        },
    ] {
        mixed.lenses[1] = legacy;
        let opened = open_full_transports(&mixed, Path::new(".")).unwrap();
        let mut projection_started = false;
        let result: Result<()> = (|| {
            let _bound = bind_full_transports(opened, &mixed, Path::new("."), &gguf, None)?;
            projection_started = true;
            Ok(())
        })();
        assert!(result.is_err());
        assert!(!projection_started);
    }
}

#[test]
fn generic_plan_preflight_is_data_driven_and_retains_verified_handles() {
    use crate::linear_transport::tests::{fixture, modify};
    for (method, hidden, seed) in [("new-qwen-recipe", 2, 3), ("future-fit", 5, 7)] {
        let f = fixture(method, hidden, seed);
        let mut plan: LensPlan = serde_json::from_value(serde_json::json!({
            "version":1,
            "lenses":[{"kind":"linear_transport", "id":"arbitrary", "artifact":f.0, "token_ids":[7,1], "allow_unvalidated_transfer":true}],
            "directions":[], "operations":[],
            "readouts":[{"id":"scores", "lens":"arbitrary", "scope":{"layers":{"kind":"values","values":[0,2]},"prefill":{"kind":"all"}},"top_k":2}]
        })).unwrap();
        validate_plan(&plan).unwrap();
        validate_ordinary_plan(&plan).unwrap();
        assert_eq!(
            open_full_transports(&plan, Path::new(".")).unwrap().len(),
            1
        );
        if let LensDefinition::PublishedFullTransport {
            allow_unvalidated_transfer,
            ..
        } = &mut plan.lenses[0]
        {
            *allow_unvalidated_transfer = false;
        }
        assert!(open_full_transports(&plan, Path::new(".")).is_err());
        modify(
            &f,
            |v| v["model"]["exact_binding"] = serde_json::json!({"gguf_content_blake3":"a".repeat(64),"tokenizer_metadata_id":"b".repeat(16)}),
        );
        assert_eq!(
            open_full_transports(&plan, Path::new(".")).unwrap().len(),
            1
        );
        std::fs::write(f.0.join("transport.f16le"), vec![0u8; hidden * hidden * 4]).unwrap();
        assert!(open_full_transports(&plan, Path::new(".")).is_err());
    }
}
use clap::Parser;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Parser)]
struct RunArgsParser {
    #[command(flatten)]
    args: LensRunArgs,
}

#[derive(Debug, Parser)]
struct SweepArgsParser {
    #[command(flatten)]
    args: CoefficientSweepArgs,
}

#[test]
fn request_capacity_follows_model_context_not_cli_constants() {
    assert_eq!(required_forward_count(1, 1).unwrap(), 1);
    assert_eq!(required_forward_count(493, 16_384).unwrap(), 16_876);
    assert_eq!(
        ensure_request_fits_context(493, 16_384, 16_876).unwrap(),
        16_876
    );
    assert!(ensure_request_fits_context(493, 16_384, 16_875).is_err());
    assert!(required_forward_count(0, 1).is_err());
    assert!(required_forward_count(1, 0).is_err());
}

fn minimal_plan() -> LensPlan {
    serde_json::from_value(json!({
        "version": 1,
        "lenses": [{"kind":"native_selected","id":"j","artifact":"relative/j"}],
        "directions": [],
        "operations": [],
        "readouts": [{
            "id":"live",
            "lens":"j",
            "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
            "top_k":1
        }]
    }))
    .unwrap()
}

fn sweep_plan() -> LensPlan {
    serde_json::from_value(json!({
        "version": 1,
        "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
        "directions": [{
            "id":"d",
            "lens":"t",
            "row":{"kind":"template_row_id","template_row_id":0},
            "normalization":"unit_l2"
        }],
        "operations": [
            {
                "id":"swept",
                "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
                "action":{"kind":"residual_l2_fraction","direction":"d","coefficient":0.25}
            },
            {
                "id":"fixed",
                "scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"all"}},
                "action":{"kind":"fixed_add","direction":"d","coefficient":0.5}
            }
        ],
        "readouts": []
    }))
    .unwrap()
}

fn test_args() -> LensRunArgs {
    RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--prompt",
        "hello",
    ])
    .unwrap()
    .args
}

#[test]
fn run_cli_uses_contextual_default_and_accepts_explicit_json_output() {
    let defaults = test_args();
    assert_eq!(defaults.format, None);
    assert!(defaults.output.is_none());
    assert_eq!(defaults.prefill_execution, PrefillExecution::Auto);

    let parsed = RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--token-ids",
        "1,2",
        "--output",
        "run.json",
        "--format",
        "json",
        "--prefill-execution",
        "serial",
    ])
    .unwrap()
    .args;
    assert_eq!(parsed.format, Some(RunStdoutFormat::Json));
    assert_eq!(parsed.prefill_execution, PrefillExecution::Serial);
    assert_eq!(parsed.output.as_deref(), Some(Path::new("run.json")));
    assert_eq!(
        effective_run_stdout_format(None, false),
        RunStdoutFormat::Json
    );
    assert_eq!(
        effective_run_stdout_format(None, true),
        RunStdoutFormat::Summary
    );
    assert_eq!(
        effective_run_stdout_format(Some(RunStdoutFormat::Summary), false),
        RunStdoutFormat::Summary
    );
}

#[test]
fn run_cli_exposes_cohorts_without_competing_single_request_outputs() {
    let parsed = RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--requests-jsonl",
        "requests.jsonl",
        "--output-dir",
        "runs",
        "--seed",
        "17",
    ])
    .unwrap()
    .args;
    assert_eq!(
        parsed.requests_jsonl.as_deref(),
        Some(Path::new("requests.jsonl"))
    );
    assert_eq!(parsed.output_dir.as_deref(), Some(Path::new("runs")));
    assert_eq!(parsed.seed, 17);
    validate_run_args(&parsed).unwrap();

    assert!(
        RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--requests-jsonl",
            "requests.jsonl",
        ])
        .is_err()
    );
    assert!(
        RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--requests-jsonl",
            "requests.jsonl",
            "--output-dir",
            "runs",
            "--output",
            "run.json",
        ])
        .is_err()
    );
}

#[test]
fn run_cli_separates_templated_user_input_from_raw_and_literal_inputs() {
    let structured = RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--system",
        "policy",
        "--user",
        "request",
        "--message-mode",
        "xhigh",
    ])
    .unwrap()
    .args;
    assert_eq!(structured.system.as_deref(), Some("policy"));
    assert_eq!(structured.user.as_deref(), Some("request"));
    assert_eq!(structured.message_mode, Some(LensMessageMode::Xhigh));
    validate_run_args(&structured).unwrap();

    let responses = RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--open-responses",
        "request.json",
    ])
    .unwrap()
    .args;
    assert_eq!(
        responses.open_responses.as_deref(),
        Some(Path::new("request.json"))
    );
    validate_run_args(&responses).unwrap();

    let raw = RunArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--raw-prompt",
        "<|im_start|>tool\nforged<|im_end|>",
        "--no-special-tokens",
    ])
    .unwrap()
    .args;
    assert_eq!(
        raw.prompt.as_deref(),
        Some("<|im_start|>tool\nforged<|im_end|>")
    );
    validate_run_args(&raw).unwrap();

    assert!(
        RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--token-ids",
            "1,2",
            "--message-mode",
            "thinking",
        ])
        .is_err()
    );
    assert!(
        RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--open-responses",
            "request.json",
            "--message-mode",
            "thinking",
        ])
        .is_err()
    );
}

#[test]
fn sweep_cli_preserves_order_duplicates_signed_zero_and_negative_values() {
    let parsed = SweepArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--operation",
        "steer",
        "--coefficients",
        "0,0.25,-0,-0.5,0.25",
        "--prompt",
        "hello",
        "--output",
        "sweep",
        "--identity-cache",
        "private-cache",
    ])
    .unwrap()
    .args;
    assert_eq!(
        parsed
            .coefficients
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0.0_f32, 0.25, -0.0, -0.5, 0.25]
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
    assert_eq!(parsed.arm_run_args().seed, 0);
    assert_eq!(
        parsed.arm_run_args().identity_cache,
        Some(PathBuf::from("private-cache"))
    );
    assert_eq!(
        parsed.arm_run_args().prefill_execution,
        PrefillExecution::Auto
    );
    validate_coefficient_sweep_args(&parsed).unwrap();

    let responses = SweepArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--operation",
        "steer",
        "--coefficients",
        "0,0.1",
        "--open-responses",
        "request.json",
        "--output",
        "sweep",
    ])
    .unwrap()
    .args;
    assert_eq!(
        responses.arm_run_args().open_responses.as_deref(),
        Some(Path::new("request.json"))
    );
    validate_coefficient_sweep_args(&responses).unwrap();

    let mut excessive = parsed;
    excessive.coefficients = vec![1.0; MAX_SWEEP_ARMS + 1];
    assert!(validate_coefficient_sweep_args(&excessive).is_err());
    excessive.coefficients = vec![f32::NAN];
    assert!(validate_coefficient_sweep_args(&excessive).is_err());
}

#[test]
fn sweep_cohort_cli_accepts_auto_and_explicit_serial_prefill_policy() {
    let parsed = SweepArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--operation",
        "steer",
        "--coefficients",
        "0,0.5",
        "--requests-jsonl",
        "requests.jsonl",
        "--output",
        "cohort",
    ])
    .unwrap()
    .args;
    assert_eq!(
        parsed.requests_jsonl.as_deref(),
        Some(Path::new("requests.jsonl"))
    );
    assert_eq!(parsed.prefill_execution, PrefillExecution::Auto);
    validate_coefficient_sweep_args(&parsed).unwrap();

    let explicit_serial = SweepArgsParser::try_parse_from([
        "test",
        "--model",
        "model.gguf",
        "--plan",
        "plan.json",
        "--operation",
        "steer",
        "--coefficients",
        "0",
        "--requests-jsonl",
        "requests.jsonl",
        "--prefill-execution",
        "serial",
        "--output",
        "cohort",
    ])
    .unwrap()
    .args;
    validate_coefficient_sweep_args(&explicit_serial).unwrap();
    assert!(
        SweepArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--operation",
            "steer",
            "--coefficients",
            "0",
            "--requests-jsonl",
            "requests.jsonl",
            "--messages",
            "one.json",
            "--prefill-execution",
            "serial",
            "--output",
            "cohort",
        ])
        .is_err()
    );
}

#[test]
fn run_cohort_jsonl_reuses_the_complete_strict_lens_input_grammar() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-run-cohort-jsonl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let requests_path = root.join("requests.jsonl");
    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"raw\",\"prompt\":\"hello\"}\n",
            "{\"id\":\"tokens\",\"token_ids\":[1,2]}\n",
            "{\"id\":\"user\",\"user\":\"hello\",\"system\":\"policy\",\"message_mode\":\"no_thinking\"}\n",
            "{\"id\":\"messages\",\"messages\":\"messages.json\",\"message_mode\":\"thinking\"}\n",
            "{\"id\":\"responses\",\"open_responses\":\"request.json\"}\n"
        ),
    )
    .unwrap();
    let loaded = read_run_cohort_requests(&requests_path).unwrap();
    assert!(loaded.canonical_path.is_absolute());
    assert_eq!(loaded.records.len(), 5);
    assert_eq!(loaded.records[0].1.prompt.as_deref(), Some("hello"));
    assert_eq!(
        loaded.records[1].1.token_ids.as_deref(),
        Some([1, 2].as_slice())
    );
    assert_eq!(loaded.records[2].1.system.as_deref(), Some("policy"));
    assert_eq!(
        loaded.records[3].1.messages.as_deref(),
        Some(
            loaded
                .canonical_path
                .parent()
                .unwrap()
                .join("messages.json")
                .as_path()
        )
    );
    assert_eq!(
        loaded.records[4].1.open_responses.as_deref(),
        Some(
            loaded
                .canonical_path
                .parent()
                .unwrap()
                .join("request.json")
                .as_path()
        )
    );

    for invalid in [
        concat!(
            "{\"id\":\"one\",\"prompt\":\"x\",\"unknown\":true}\n",
            "{\"id\":\"two\",\"prompt\":\"x\"}\n"
        ),
        concat!(
            "{\"id\":\"one\",\"prompt\":\"x\",\"token_ids\":[1]}\n",
            "{\"id\":\"two\",\"prompt\":\"x\"}\n"
        ),
        concat!(
            "{\"id\":\"one\",\"user\":\"-\"}\n",
            "{\"id\":\"two\",\"prompt\":\"x\"}\n"
        ),
        concat!(
            "{\"id\":\"same\",\"prompt\":\"x\"}\n",
            "{\"id\":\"same\",\"prompt\":\"x\"}\n"
        ),
    ] {
        std::fs::write(&requests_path, invalid).unwrap();
        assert!(read_run_cohort_requests(&requests_path).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn run_cohort_manifest_roundtrips_and_recomputes_all_aggregate_bounds() {
    let source_plan = minimal_plan();
    let prepared = |tokens: Vec<i32>| PreparedLensInput {
        source: "token_ids",
        add_special_tokens: None,
        token_ids: tokens,
        rendering: LensInputRendering {
            renderer: "literal_token_ids".into(),
            generation_mode: None,
            spans: Vec::new(),
        },
    };
    let requests = vec![
        PreflightRunCohortRequest {
            source_line: 1,
            id: "one".into(),
            prepared_input: prepared(vec![1, 2]),
        },
        PreflightRunCohortRequest {
            source_line: 2,
            id: "two".into(),
            prepared_input: prepared(vec![1, 2, 3]),
        },
    ];
    let basis = RunCohortManifestBasis {
        producer: current_sweep_producer(),
        requests_jsonl_path: PathBuf::from("/tmp/requests.jsonl"),
        canonical_plan_path: PathBuf::from("/tmp/plan.json"),
        source_plan,
        model_path: PathBuf::from("model.gguf"),
        sampler: RunSampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        },
        max_new_tokens: 4,
        prefill_execution: PrefillExecution::Serial,
        bounds: RunCohortBounds {
            request_count: 2,
            aggregate_prompt_tokens: 5,
            work_upper_bound: 13,
            forward_upper_bound: 11,
            sample_upper_bound: 8,
            max_request_forwards: 6,
        },
    };
    let children = requests
        .iter()
        .enumerate()
        .map(|(index, request)| RunCohortChild {
            index,
            id: request.id.clone(),
            source_line: request.source_line,
            path: format!("run-{index:06}.json"),
            input_source: request.prepared_input.source.into(),
            prompt_token_count: request.prepared_input.token_ids.len(),
            artifact_byte_length: 10 * (index as u64 + 1),
        })
        .collect::<Vec<_>>();
    let manifest = basis.build(30, children);
    let bytes = serialize_run_cohort_manifest(&manifest).unwrap();
    let decoded: RunCohortManifest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, manifest);

    let mut invalid = manifest;
    invalid.work_upper_bound += 1;
    assert!(serialize_run_cohort_manifest(&invalid).is_err());
}

#[test]
fn sweep_cohort_jsonl_is_strict_bounded_and_resolves_relative_message_paths() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-jsonl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("one.json"), "[]").unwrap();
    std::fs::write(root.join("two.json"), "[]").unwrap();
    let requests_path = root.join("requests.jsonl");
    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"first-safe\",\"messages\":\"one.json\"}\n",
            "\n",
            "{\"id\":\"second_safe\",\"messages\":\"two.json\",\"message_mode\":\"thinking\"}\n"
        ),
    )
    .unwrap();
    let loaded = read_sweep_cohort_requests(&requests_path, 2, 64).unwrap();
    assert!(loaded.canonical_path.is_absolute());
    assert_eq!(loaded.records.len(), 2);
    assert_eq!(loaded.records[0].0, 1);
    assert_eq!(loaded.records[1].0, 3);
    assert_eq!(
        loaded.records[0].1.messages,
        loaded.canonical_path.parent().unwrap().join("one.json")
    );
    assert_eq!(
        loaded.records[1].1.message_mode,
        Some(LensMessageMode::Thinking)
    );

    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"first\",\"messages\":\"one.json\",\"unknown\":true}\n",
            "{\"id\":\"second\",\"messages\":\"two.json\"}\n"
        ),
    )
    .unwrap();
    assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"../unsafe\",\"messages\":\"one.json\"}\n",
            "{\"id\":\"second\",\"messages\":\"two.json\"}\n"
        ),
    )
    .unwrap();
    assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sweep_cohort_jsonl_rejects_invalid_records_and_accepts_resource_bounded_counts() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-bounds-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("messages.json"), "[]").unwrap();
    let requests_path = root.join("requests.jsonl");
    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"same\",\"messages\":\"messages.json\"}\n",
            "{\"id\":\"same\",\"messages\":\"messages.json\"}\n"
        ),
    )
    .unwrap();
    assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
    std::fs::write(
        &requests_path,
        concat!(
            "{\"id\":\"one\",\"messages\":\"-\"}\n",
            "{\"id\":\"two\",\"messages\":\"messages.json\"}\n"
        ),
    )
    .unwrap();
    assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
    std::fs::write(
        &requests_path,
        "{\"id\":\"one\",\"messages\":\"messages.json\"}\n",
    )
    .unwrap();
    assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
    assert!(checked_sweep_cohort_child_count(usize::MAX, 2).is_err());

    let expanded = (0..97)
        .map(|index| format!("{{\"id\":\"request-{index}\",\"messages\":\"messages.json\"}}\n"))
        .collect::<String>();
    std::fs::write(&requests_path, expanded).unwrap();
    let loaded = read_sweep_cohort_requests(&requests_path, 2, 64).unwrap();
    assert_eq!(loaded.records.len(), 97);
    assert_eq!(checked_sweep_cohort_child_count(97, 2).unwrap(), 194);
    assert!(read_sweep_cohort_requests(&requests_path, 1, 499_999).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sweep_cohort_messages_are_single_read_no_follow_and_content_bound() {
    static READS: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-capture-{}-{}",
        std::process::id(),
        READS.load(Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("messages.json");
    let original = br#"[{"role":"user","content":"original"}]"#;
    std::fs::write(&path, original).unwrap();
    READS.store(0, Ordering::Relaxed);
    let captured = capture_sweep_cohort_messages_with(&path, |path| {
        READS.fetch_add(1, Ordering::Relaxed);
        crate::read_regular_file_bounded(path, MAX_SWEEP_COHORT_MESSAGES_BYTES)
    })
    .unwrap();
    std::fs::write(&path, br#"[{"role":"user","content":"replacement"}]"#).unwrap();
    assert_eq!(READS.load(Ordering::Relaxed), 1);
    assert_eq!(captured.bytes, original);
    assert_eq!(captured.blake3, blake3::hash(original).to_hex().to_string());
    let parsed = crate::messages::parse_strict_messages_input(
        std::str::from_utf8(&captured.bytes).unwrap(),
        &captured.path.display().to_string(),
    )
    .unwrap()
    .messages;
    assert_eq!(parsed[0].content, "original");

    let symlink = root.join("messages-link.json");
    std::os::unix::fs::symlink(&path, &symlink).unwrap();
    assert!(capture_sweep_cohort_messages(&symlink).is_err());
    let oversized = root.join("oversized.json");
    std::fs::File::create(&oversized)
        .unwrap()
        .set_len((MAX_SWEEP_COHORT_MESSAGES_BYTES + 1) as u64)
        .unwrap();
    assert!(capture_sweep_cohort_messages(&oversized).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sweep_cohort_aggregate_bounds_admit_campaign_and_reject_pathological_work() {
    let admitted = plan_sweep_cohort_bounds(&vec![4_096; 64], 3, 64).unwrap();
    assert_eq!(admitted.request_count, 64);
    assert_eq!(admitted.total_arm_count, 192);
    assert_eq!(admitted.transition_upper_bound, 798_720);

    assert!(plan_sweep_cohort_bounds(&vec![4_096; 64], 64, 64).is_err());
    assert!(plan_sweep_cohort_bounds(&vec![11_000; 32], 3, 64).is_err());
}

#[test]
fn sweep_cohort_output_budget_overflow_cleans_staging() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-output-budget-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let output = root.join("cohort");
    assert!(
        stage_and_publish_bundle(&output, |staging| -> Result<()> {
            write_new_bundle_file(&staging.join("partial"), b"partial")?;
            let mut budget = BundleOutputBudget::new(0)?;
            budget.consumed = MAX_SWEEP_BUNDLE_BYTES - 1;
            budget.charge(2)
        })
        .is_err()
    );
    assert!(!output.exists());
    assert!(std::fs::read_dir(&root).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".stage.")
    }));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sweep_output_budget_reserves_then_reconciles_outer_manifest() {
    let mut budget = BundleOutputBudget::new(8).unwrap();
    budget.consumed = MAX_SWEEP_BUNDLE_BYTES - 8;
    assert!(budget.charge(1).is_err());
    assert_eq!(budget.consumed, MAX_SWEEP_BUNDLE_BYTES - 8);
    budget.release_reservation();
    budget.charge(8).unwrap();
    assert_eq!(budget.consumed, MAX_SWEEP_BUNDLE_BYTES);
}

#[test]
#[ignore = "requires a compatible local GGUF, Lens plan, operation ID, and two message fixtures"]
fn sweep_cohort_model_bound_children_are_inspectable() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let model = std::env::var_os("QWEN_LENS_COHORT_MODEL").unwrap();
    let plan = std::env::var_os("QWEN_LENS_COHORT_PLAN").unwrap();
    let operation = std::env::var("QWEN_LENS_COHORT_OPERATION").unwrap();
    let first_messages =
        PathBuf::from(std::env::var_os("QWEN_LENS_COHORT_MESSAGES_FIRST").unwrap());
    let second_messages =
        PathBuf::from(std::env::var_os("QWEN_LENS_COHORT_MESSAGES_SECOND").unwrap());
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-model-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let requests_path = root.join("requests.jsonl");
    let records = [
        json!({"id":"first","messages":first_messages}),
        json!({"id":"second","messages":second_messages}),
    ];
    std::fs::write(
        &requests_path,
        records
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let output = root.join("cohort");
    run_coefficient_sweep(CoefficientSweepArgs {
        identity_cache: None,
        model: model.into(),
        plan: plan.into(),
        operation,
        coefficients: vec![0.0, 0.5],
        prompt: None,
        token_ids: None,
        user: None,
        system: None,
        messages: None,
        open_responses: None,
        requests_jsonl: Some(requests_path),
        message_mode: None,
        no_special_tokens: false,
        max_new_tokens: 1,
        prefill_execution: PrefillExecution::Auto,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0,
        output: output.clone(),
    })
    .unwrap();
    let manifest_bytes = std::fs::read(output.join(SWEEP_MANIFEST_NAME)).unwrap();
    let manifest = parse_sweep_cohort_manifest_bytes(&manifest_bytes).unwrap();
    for child in &manifest.sweeps {
        verify_sweep_cohort_child_manifest(&output, child).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn every_action_coefficient_can_be_overridden() {
    let mut actions = vec![
        Action::FixedAdd {
            direction: "a".into(),
            coefficient: 1.0,
        },
        Action::ResidualL2Fraction {
            direction: "a".into(),
            coefficient: 1.0,
        },
        Action::ProjectionAblate {
            direction: "a".into(),
            coefficient: 1.0,
        },
        Action::SourceToTarget {
            source: "a".into(),
            target: "b".into(),
            coefficient: 1.0,
        },
        Action::CoordinateSwap {
            source: "a".into(),
            target: "b".into(),
            coefficient: 1.0,
        },
    ];
    for action in &mut actions {
        action.set_coefficient(-0.125);
        assert_eq!(action.coefficient().to_bits(), (-0.125_f32).to_bits());
    }
}

#[test]
fn sweep_changes_only_the_selected_operation_and_supports_zero_controls() {
    let source = sweep_plan();
    validate_plan(&source).unwrap();
    let zero = plan_with_operation_coefficient(&source, "swept", -0.0).unwrap();
    assert_eq!(source.operations[0].action.coefficient(), 0.25);
    assert_eq!(
        zero.operations[0].action.coefficient().to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(zero.operations[1], source.operations[1]);
    assert!(!operation_enabled(&zero.operations[0]));
    assert!(operation_enabled(&zero.operations[1]));
    validate_plan(&zero).unwrap();
    assert!(validate_sweep_source_operation(&zero, "swept").is_err());
    validate_sweep_source_operation(&source, "swept").unwrap();

    let mut wrong_zero = source.clone();
    wrong_zero.operations[1].action.set_coefficient(0.0);
    validate_plan(&wrong_zero).unwrap();
    assert!(!operation_enabled(&wrong_zero.operations[1]));
    assert!(plan_with_operation_coefficient(&source, "missing", 1.0).is_err());

    let first = plan_with_operation_coefficient(&source, "swept", 0.0).unwrap();
    let middle = plan_with_operation_coefficient(&source, "swept", 0.75).unwrap();
    let last = plan_with_operation_coefficient(&source, "swept", 0.0).unwrap();
    assert_eq!(first, last);
    assert_ne!(first, middle);
}

#[test]
fn sweep_manifest_is_ordered_bounded_and_preserves_signed_zero() {
    let source_plan = sweep_plan();
    let coefficients = vec![0.0, 0.5, -0.0];
    let arms = coefficients
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| CoefficientSweepArm {
            index,
            coefficient,
            artifact: format!("arms/{index:06}/run.json"),
            byte_length: 10,
            blake3: "0".repeat(64),
        })
        .collect();
    let manifest = CoefficientSweepManifest {
        schema: SWEEP_SCHEMA.into(),
        schema_version: SWEEP_SCHEMA_VERSION,
        producer: SweepProducer {
            build_commit: "a".repeat(40),
            build_dirty: "0".into(),
            build_source_state: format!("git-source-sha256-v2:{}", "b".repeat(64)),
        },
        canonical_source_plan_path: "/tmp/plan.json".into(),
        source_plan: Some(source_plan.clone()),
        source_plan_canonical_json_blake3: Some(canonical_plan_blake3(&source_plan).unwrap()),
        operation_id: "swept".into(),
        coefficients,
        arms,
    };
    let bytes = serialize_sweep_manifest(&manifest).unwrap();
    let decoded = parse_sweep_manifest_bytes(&bytes).unwrap();
    assert_eq!(decoded.coefficients[2].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(decoded.arms[2].coefficient.to_bits(), (-0.0_f32).to_bits());
    let arm_bytes = decoded.arms.iter().map(|arm| arm.byte_length).sum::<u64>();
    let largest_manifest = usize::try_from(MAX_SWEEP_BUNDLE_BYTES - arm_bytes).unwrap();
    validate_sweep_bundle_size(&decoded, largest_manifest).unwrap();
    assert!(validate_sweep_bundle_size(&decoded, largest_manifest + 1).is_err());

    let mut malformed = decoded.clone();
    malformed.arms[1].artifact = "../run.json".into();
    assert!(validate_sweep_manifest(&malformed).is_err());

    let mut malformed_producer = decoded;
    malformed_producer.producer.build_commit = "not-a-commit".into();
    assert!(validate_sweep_manifest(&malformed_producer).is_err());
}

#[test]
fn sweep_cohort_manifest_roundtrips_and_verifies_child_manifest_integrity() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let source_plan = sweep_plan();
    let child_manifest = CoefficientSweepManifest {
        schema: SWEEP_SCHEMA.into(),
        schema_version: SWEEP_SCHEMA_VERSION,
        producer: SweepProducer {
            build_commit: "a".repeat(40),
            build_dirty: "0".into(),
            build_source_state: format!("git-source-sha256-v2:{}", "b".repeat(64)),
        },
        canonical_source_plan_path: "/tmp/plan.json".into(),
        source_plan: Some(source_plan.clone()),
        source_plan_canonical_json_blake3: Some(canonical_plan_blake3(&source_plan).unwrap()),
        operation_id: "swept".into(),
        coefficients: vec![0.0],
        arms: vec![CoefficientSweepArm {
            index: 0,
            coefficient: 0.0,
            artifact: "arms/000000/run.json".into(),
            byte_length: 1,
            blake3: "c".repeat(64),
        }],
    };
    let child_bytes = serialize_sweep_manifest(&child_manifest).unwrap();
    let child_digest = blake3::hash(&child_bytes).to_hex().to_string();
    let children = (0..2)
        .map(|index| SweepCohortChild {
            index,
            id: format!("request-{index}"),
            source_line: index + 1,
            path: format!("sweeps/{index:06}"),
            prompt_token_count: 4,
            messages_path: format!("/tmp/messages-{index}.json").into(),
            messages_blake3: "e".repeat(64),
            serialized_byte_length: child_bytes.len() as u64 + 1,
            manifest_byte_length: child_bytes.len() as u64,
            manifest_blake3: child_digest.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = SweepCohortManifest {
        schema: SWEEP_COHORT_SCHEMA.into(),
        schema_version: SWEEP_COHORT_SCHEMA_VERSION,
        producer: child_manifest.producer.clone(),
        requests_jsonl_path: "/tmp/requests.jsonl".into(),
        requests_jsonl_blake3: "d".repeat(64),
        canonical_source_plan_path: "/tmp/plan.json".into(),
        source_plan: source_plan.clone(),
        source_plan_canonical_json_blake3: canonical_plan_blake3(&source_plan).unwrap(),
        model_path: "model.gguf".into(),
        operation_id: "swept".into(),
        coefficients: vec![0.0],
        sampler: RunSampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        },
        max_new_tokens: 1,
        prefill_execution: PrefillExecution::Serial,
        execution_policy:
            "serial_prompts_serial_arms_fresh_sequence_and_sampler_no_batched_generation".into(),
        planned_request_count: 2,
        planned_total_arm_count: 2,
        transition_upper_bound: 10,
        cumulative_serialized_child_bytes: 2 * (child_bytes.len() as u64 + 1),
        sweeps: children,
    };
    let bytes = serialize_sweep_cohort_manifest(&manifest).unwrap();
    assert_eq!(parse_sweep_cohort_manifest_bytes(&bytes).unwrap(), manifest);
    let largest_manifest =
        usize::try_from(MAX_SWEEP_BUNDLE_BYTES - manifest.cumulative_serialized_child_bytes)
            .unwrap();
    validate_sweep_cohort_bundle_size(&manifest, largest_manifest).unwrap();
    assert!(validate_sweep_cohort_bundle_size(&manifest, largest_manifest + 1).is_err());
    let mut malformed_aggregate = manifest.clone();
    malformed_aggregate.cumulative_serialized_child_bytes += 1;
    assert!(validate_sweep_cohort_manifest(&malformed_aggregate).is_err());
    let mut malformed_binding = manifest.clone();
    malformed_binding.sweeps[0].messages_blake3 = "not-a-digest".into();
    assert!(validate_sweep_cohort_manifest(&malformed_binding).is_err());

    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-cohort-integrity-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(root.join("sweeps/000000")).unwrap();
    std::fs::write(root.join("sweeps/000000/manifest.json"), &child_bytes).unwrap();
    verify_sweep_cohort_child_manifest(&root, &manifest.sweeps[0]).unwrap();
    let mut drifted = manifest.sweeps[0].clone();
    drifted.manifest_blake3 = "e".repeat(64);
    assert!(verify_sweep_cohort_child_manifest(&root, &drifted).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sweep_cohort_binds_semantic_positions_independently_per_prompt() {
    let plan = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[{"span_kind":"message_content","role":"user","edge":"end"}]
    }));
    let rendering = |range| LensInputRendering {
        renderer: "qwen_chatml_messages_v1".into(),
        generation_mode: Some("auto".into()),
        spans: vec![rendered_span(
            "message_content",
            Some(0),
            "user",
            None,
            Some(range),
        )],
    };
    let first = bind_plan_positions(&plan, &rendering((1, 3)), 3).unwrap();
    let second = bind_plan_positions(&plan, &rendering((4, 8)), 8).unwrap();
    assert_eq!(first.position_bindings[0].resolved_index, 2);
    assert_eq!(second.position_bindings[0].resolved_index, 7);
    assert_ne!(first.resolved, second.resolved);
    assert_eq!(first.authored, second.authored);
}

#[test]
fn sweep_staging_publishes_exclusively_and_cleans_its_failures() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-sweep-stage-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    DirBuilder::new().mode(0o700).create(&root).unwrap();

    let published = root.join("published");
    stage_and_publish_bundle(&published, |staging| {
        write_new_bundle_file(&staging.join("marker"), b"complete")?;
        crate::sync_directory(staging)?;
        Ok(())
    })
    .unwrap();
    assert_eq!(
        std::fs::read(published.join("marker")).unwrap(),
        b"complete"
    );

    let failed = root.join("failed");
    assert!(
        stage_and_publish_bundle(&failed, |staging| -> Result<()> {
            let nested = staging.join("sweeps");
            create_bundle_directory(&nested)?;
            write_new_bundle_file(&nested.join("partial"), b"partial")?;
            bail!("injected failure")
        })
        .is_err()
    );
    assert!(!failed.exists());

    let raced = root.join("raced");
    assert!(
        stage_and_publish_bundle(&raced, |staging| {
            write_new_bundle_file(&staging.join("marker"), b"ours")?;
            create_bundle_directory(&raced)?;
            Ok(())
        })
        .is_err()
    );
    assert!(raced.is_dir());
    let leftovers = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains(".stage."))
        .count();
    assert_eq!(leftovers, 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn run_artifact_serializes_envelope_exact_plan_and_score_semantics() {
    let plan = minimal_plan();
    let plan_digest = canonical_plan_blake3(&plan).unwrap();
    let artifact = RunOutput {
        linear_transports: Vec::new(),
        schema: RUN_SCHEMA,
        schema_version: RUN_SCHEMA_VERSION,
        runtime_kind: "ordinary_qwen",
        model_path: "model.gguf".into(),
        canonical_plan_path: "/canonical/plan.json".into(),
        authored_plan: plan.clone(),
        authored_plan_canonical_json_blake3: plan_digest,
        requested_live_readouts: plan.readouts.clone(),
        plan,
        position_bindings: Vec::new(),
        input_source: "prompt",
        add_special_tokens: Some(true),
        rendering: LensInputRendering {
            renderer: "tokenizer_text".into(),
            generation_mode: None,
            spans: Vec::new(),
        },
        prompt_token_ids: vec![1, 2],
        generated_token_ids: vec![3],
        sampler: RunSampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        },
        max_new_tokens: 1,
        decoded_text: "done".into(),
        stop_reason: "max_new_tokens".into(),
        execution: RunExecution::runtime_serial(
            PrefillExecution::Auto,
            RunSerialReason::NoEligiblePassiveSpan,
        ),
        operation_applications: Vec::new(),
        live_readouts: vec![LiveReadout {
            id: "live".into(),
            lens: "j".into(),
            method: "J".into(),
            score_kind: "selected_row_projection_numerator",
            candidate_universe: "lens_artifact_selected_token_rows",
            source_layer: 1,
            target_layer: Some(2),
            phase: "prefill",
            index: 0,
            scores: Vec::new(),
        }],
        native_hyper_captures: Vec::new(),
        execution_binding: None,
    };
    let value = serde_json::to_value(&artifact).unwrap();
    assert_eq!(value["schema"], RUN_SCHEMA);
    assert_eq!(value["schema_version"], 5);
    assert_eq!(value["input_source"], "prompt");
    assert_eq!(value["add_special_tokens"], true);
    assert_eq!(value["rendering"]["renderer"], "tokenizer_text");
    assert_eq!(value["plan"]["lenses"][0]["artifact"], "relative/j");
    assert_eq!(value["requested_live_readouts"], value["plan"]["readouts"]);
    assert_eq!(
        value["live_readouts"][0]["score_kind"],
        "selected_row_projection_numerator"
    );
    assert_eq!(
        value["live_readouts"][0]["candidate_universe"],
        "lens_artifact_selected_token_rows"
    );
    assert!(value["live_readouts"][0].get("probability").is_none());

    let expected = serialize_run_output(&artifact).unwrap();
    let root = std::env::temp_dir().join(format!(
        "qwen-lens-run-output-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let output = root.join("run.json");
    let length = write_new_run_output(&output, &artifact, MAX_RUN_ARTIFACT_BYTES).unwrap();
    assert_eq!(length, expected.len());
    assert_eq!(std::fs::read(&output).unwrap(), expected);

    let rejected = root.join("rejected.json");
    assert!(write_new_run_output(&rejected, &artifact, expected.len() - 1).is_err());
    assert!(!rejected.exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn run_execution_metadata_distinguishes_serial_controls_from_packed_spans() {
    let automatic_serial = RunExecution::runtime_serial(
        PrefillExecution::Auto,
        RunSerialReason::NoEligiblePassiveSpan,
    );
    automatic_serial.validate("ordinary_qwen", 8).unwrap();
    assert_eq!(
        automatic_serial.serial_reason(),
        Some(RunSerialReason::NoEligiblePassiveSpan)
    );

    let explicit_serial = RunExecution::runtime_serial(
        PrefillExecution::Serial,
        RunSerialReason::MusePackedNotImplemented,
    );
    explicit_serial.validate("ordinary_qwen", 8).unwrap();
    assert_eq!(
        explicit_serial.serial_reason(),
        Some(RunSerialReason::RequestedSerial)
    );

    let cohort_auto = RunExecution::serial(
        PrefillExecution::Auto,
        RunExecutionScheduleBasis::SweepSourcePlan,
        RunSerialReason::CohortSerialPolicy,
    );
    cohort_auto.validate("ordinary_qwen", 8).unwrap();
    assert_eq!(
        cohort_auto.serial_reason(),
        Some(RunSerialReason::CohortSerialPolicy)
    );

    let packed = RunExecution::dense_packed(
        RunExecutionScheduleBasis::EffectivePlan,
        70,
        141,
        1,
        vec![
            RunPackedPrefillSpan { start: 0, end: 65 },
            RunPackedPrefillSpan {
                start: 70,
                end: 140,
            },
        ],
    );
    packed.validate("ordinary_qwen", 141).unwrap();

    let mut final_token_overlap = packed.clone();
    final_token_overlap.packed_spans[1].end = 141;
    assert!(final_token_overlap.validate("ordinary_qwen", 141).is_err());
    let mut wrong_block = packed;
    wrong_block.block_tokens = Some(65);
    assert!(wrong_block.validate("ordinary_qwen", 141).is_err());

    let mut scheduled_plan = minimal_plan();
    scheduled_plan.readouts[0].scope.prefill = Some(Selector::Values { values: vec![70] });
    let scheduled = RunExecution::dense_packed(
        RunExecutionScheduleBasis::EffectivePlan,
        70,
        0,
        1,
        vec![
            RunPackedPrefillSpan { start: 0, end: 70 },
            RunPackedPrefillSpan {
                start: 71,
                end: 140,
            },
        ],
    );
    scheduled
        .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
        .unwrap();
    let mut undersized_matrix = scheduled.clone();
    undersized_matrix.attention_matrix_max_position = Some(1);
    assert!(undersized_matrix.validate("ordinary_qwen", 141).is_err());
    assert!(
        automatic_serial
            .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
            .is_err()
    );
    let mut false_schedule = scheduled;
    false_schedule.packed_spans[0].end = 69;
    false_schedule.block_tokens = Some(69);
    assert!(
        false_schedule
            .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
            .is_err()
    );
}

#[test]
fn summary_contains_required_counts_text_and_artifact_path() {
    let plan = minimal_plan();
    let plan_digest = canonical_plan_blake3(&plan).unwrap();
    let artifact = RunOutput {
        linear_transports: Vec::new(),
        schema: RUN_SCHEMA,
        schema_version: RUN_SCHEMA_VERSION,
        runtime_kind: "muse_glimmer",
        model_path: "muse.gguf".into(),
        canonical_plan_path: "/canonical/plan.json".into(),
        authored_plan: plan.clone(),
        authored_plan_canonical_json_blake3: plan_digest,
        requested_live_readouts: plan.readouts.clone(),
        plan,
        position_bindings: Vec::new(),
        input_source: "token_ids",
        add_special_tokens: None,
        rendering: LensInputRendering {
            renderer: "literal_token_ids".into(),
            generation_mode: None,
            spans: Vec::new(),
        },
        prompt_token_ids: vec![1],
        generated_token_ids: vec![2],
        sampler: RunSampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        },
        max_new_tokens: 1,
        decoded_text: "line\nbreak".into(),
        stop_reason: "stop_token".into(),
        execution: RunExecution::runtime_serial(
            PrefillExecution::Auto,
            RunSerialReason::MusePackedNotImplemented,
        ),
        operation_applications: vec![OperationApplication {
            id: "op".into(),
            layer: 1,
            phase: "prefill",
            index: 0,
        }],
        live_readouts: Vec::new(),
        native_hyper_captures: Vec::new(),
        execution_binding: None,
    };
    assert_eq!(
        run_summary(&artifact, Some(Path::new("/tmp/run.json"))),
        concat!(
            "runtime=muse_glimmer model=muse.gguf\n",
            "prefill_execution=serial serial_reason=muse_packed_not_implemented\n",
            "generated_text=\"line\\nbreak\"\n",
            "stop_reason=stop_token\n",
            "operation_applications=1 live_readouts=0\n",
            "artifact=/tmp/run.json\n"
        )
    );
}

fn temporary_direction_path() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "qwen-lens-native-hyper-{}-{}.f32le",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn native_hyper_plan(path: &Path, operations: serde_json::Value) -> LensPlan {
    serde_json::from_value(json!({
        "version": 1,
        "lenses": [],
        "directions": [{
            "id": "hyper",
            "source": {
                "kind": "native_hyper_f32",
                "path": path,
                "layer": 23
            }
        }],
        "operations": operations,
        "readouts": []
    }))
    .unwrap()
}

#[test]
fn plan_selectors_expand_sorted_unique_and_inclusive() {
    assert_eq!(Selector::All.expand(4, "layers").unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(
        Selector::Values { values: vec![1, 3] }
            .expand(4, "layers")
            .unwrap(),
        vec![1, 3]
    );
    assert_eq!(
        Selector::Range { start: 1, end: 3 }
            .expand(4, "layers")
            .unwrap(),
        vec![1, 2, 3]
    );
    assert!(
        Selector::Values { values: vec![2, 1] }
            .expand(4, "layers")
            .is_err()
    );
}

fn rendered_span(
    kind: &str,
    message_index: Option<usize>,
    role: &str,
    channel: Option<&str>,
    token_range: Option<(usize, usize)>,
) -> LensRenderedSpan {
    LensRenderedSpan {
        kind: kind.into(),
        message_index,
        tool_call_index: None,
        role: Some(role.into()),
        channel: channel.map(str::to_owned),
        label: None,
        byte_start: token_range.map_or(0, |range| range.0),
        byte_end: token_range.map_or(1, |range| range.1),
        token_start: token_range.map(|range| range.0),
        token_end: token_range.map(|range| range.1),
    }
}

fn semantic_readout_plan(prefill: serde_json::Value) -> LensPlan {
    serde_json::from_value(json!({
        "version": 2,
        "lenses": [{"kind":"native_selected","id":"j","artifact":"j"}],
        "directions": [],
        "operations": [],
        "readouts": [{
            "id":"live",
            "lens":"j",
            "scope":{"layers":{"kind":"values","values":[1]},"prefill":prefill},
            "top_k":1
        }]
    }))
    .unwrap()
}

#[test]
fn rendered_span_selectors_bind_exact_content_edges_and_generated_markers() {
    let plan = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[
            {"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"},
            {"span_kind":"generated_assistant_start_marker","edge":"start"}
        ]
    }));
    validate_plan(&plan).unwrap();
    let rendering = LensInputRendering {
        renderer: "qwen_open_responses_annotated_v1".into(),
        generation_mode: Some("auto".into()),
        spans: vec![
            rendered_span("message_content", Some(0), "user", None, Some((1, 3))),
            rendered_span("message_content", Some(2), "user", None, Some((5, 7))),
            rendered_span(
                "generated_assistant_start_marker",
                None,
                "assistant",
                None,
                Some((8, 9)),
            ),
        ],
    };
    let bound = bind_plan_positions(&plan, &rendering, 9).unwrap();
    assert_eq!(
        bound.resolved.readouts[0].scope.prefill,
        Some(Selector::Values { values: vec![6, 8] })
    );
    assert_eq!(bound.position_bindings.len(), 2);
    assert_eq!(bound.position_bindings[0].rendering_span_index, 1);
    assert_eq!(bound.position_bindings[0].resolved_index, 6);
    assert_eq!(bound.position_bindings[1].resolved_index, 8);
    assert_eq!(
        bound.authored_plan_canonical_json_blake3,
        canonical_plan_blake3(&plan).unwrap()
    );
}

#[test]
fn tool_result_selectors_are_channel_portable_but_preserve_actual_roles() {
    let plan = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[
            {"span_kind":"tool_result_content","channel":"tool_result","edge":"end"},
            {"span_kind":"message_end_marker","channel":"tool_result","edge":"start"}
        ]
    }));
    for role in ["user", "tool"] {
        let rendering = LensInputRendering {
            renderer: if role == "user" {
                "qwen_open_responses_annotated_v1"
            } else {
                "muse_glimmer_atem_annotated_v1"
            }
            .into(),
            generation_mode: Some("auto".into()),
            spans: vec![
                rendered_span(
                    "tool_result_content",
                    Some(3),
                    role,
                    Some("tool_result"),
                    Some((4, 6)),
                ),
                rendered_span(
                    "message_end_marker",
                    Some(3),
                    role,
                    Some("tool_result"),
                    Some((6, 7)),
                ),
            ],
        };
        let bound = bind_plan_positions(&plan, &rendering, 7).unwrap();
        assert_eq!(
            bound.resolved.readouts[0].scope.prefill,
            Some(Selector::Values { values: vec![5, 6] })
        );
        assert_eq!(
            bound.position_bindings[0].matched_span.role.as_deref(),
            Some(role)
        );
    }
}

#[test]
fn rendered_span_selectors_fail_closed_on_ambiguity_missing_ranges_and_raw_input() {
    let rendering = LensInputRendering {
        renderer: "qwen_chatml_messages_v1".into(),
        generation_mode: Some("auto".into()),
        spans: vec![
            rendered_span("message_content", Some(0), "user", None, Some((1, 2))),
            rendered_span("message_content", Some(2), "user", None, Some((3, 4))),
        ],
    };
    let unique = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[{"span_kind":"message_content","role":"user","edge":"end"}]
    }));
    assert!(bind_plan_positions(&unique, &rendering, 4).is_err());

    let last = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[{"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"}]
    }));
    assert_eq!(
        bind_plan_positions(&last, &rendering, 4)
            .unwrap()
            .resolved
            .readouts[0]
            .scope
            .prefill,
        Some(Selector::Values { values: vec![3] })
    );

    let mut missing_range = rendering.clone();
    missing_range.spans[1].token_start = None;
    missing_range.spans[1].token_end = None;
    assert!(bind_plan_positions(&last, &missing_range, 4).is_err());

    let raw = LensInputRendering {
        renderer: "tokenizer_text".into(),
        generation_mode: None,
        spans: Vec::new(),
    };
    assert!(bind_plan_positions(&last, &raw, 4).is_err());

    let duplicate_position = semantic_readout_plan(json!({
        "kind":"rendered_spans",
        "selectors":[
            {"span_kind":"message_content","message_index":2,"edge":"end"},
            {"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"}
        ]
    }));
    assert!(bind_plan_positions(&duplicate_position, &rendering, 4).is_err());
}

#[test]
fn rendered_span_binding_has_global_selector_and_match_work_bounds() {
    let make_selectors = |count: usize| {
        (0..count)
            .map(|message_index| RenderedSpanSelector {
                span_kind: "message_content".into(),
                message_index: Some(message_index),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                occurrence: RenderedSpanOccurrence::Unique,
                edge: RenderedSpanEdge::End,
            })
            .collect::<Vec<_>>()
    };
    let mut excessive = minimal_plan();
    excessive.version = 2;
    excessive.readouts[0].scope.prefill = Some(Selector::RenderedSpans {
        selectors: make_selectors(MAX_RENDERED_SELECTORS_PER_PLAN + 1),
    });
    let one_span = LensInputRendering {
        renderer: "qwen_chatml_messages_v1".into(),
        generation_mode: Some("auto".into()),
        spans: vec![rendered_span(
            "message_content",
            Some(0),
            "user",
            None,
            Some((0, 1)),
        )],
    };
    assert!(bind_plan_positions(&excessive, &one_span, 1).is_err());

    let mut expensive = minimal_plan();
    expensive.version = 2;
    expensive.readouts[0].scope.prefill = Some(Selector::RenderedSpans {
        selectors: make_selectors(1000),
    });
    let rendering = LensInputRendering {
        renderer: "qwen_chatml_messages_v1".into(),
        generation_mode: Some("auto".into()),
        spans: (0..1001)
            .map(|index| {
                rendered_span(
                    "message_content",
                    Some(index),
                    "user",
                    None,
                    Some((index, index + 1)),
                )
            })
            .collect(),
    };
    assert!(bind_plan_positions(&expensive, &rendering, 1001).is_err());
}

#[test]
fn plan_v2_limits_semantic_selectors_to_prefill_and_keeps_numeric_v1_exact() {
    let semantic_prefill = json!({
        "kind":"rendered_spans",
        "selectors":[{"span_kind":"message_content","edge":"end"}]
    });
    let mut v1 = semantic_readout_plan(semantic_prefill.clone());
    v1.version = 1;
    assert!(validate_plan(&v1).is_err());

    let mut decode = semantic_readout_plan(json!({"kind":"values","values":[0]}));
    decode.readouts[0].scope.prefill = None;
    decode.readouts[0].scope.decode = Some(serde_json::from_value(semantic_prefill).unwrap());
    assert!(validate_plan(&decode).is_err());

    let numeric = minimal_plan();
    let raw = LensInputRendering {
        renderer: "literal_token_ids".into(),
        generation_mode: None,
        spans: Vec::new(),
    };
    let bound = bind_plan_positions(&numeric, &raw, 2).unwrap();
    assert_eq!(bound.authored, numeric);
    assert_eq!(bound.resolved, numeric);
    assert!(bound.position_bindings.is_empty());
}

#[test]
fn plan_rejects_missing_phase_and_retains_operation_order() {
    let plan: LensPlan = serde_json::from_value(json!({
        "version": 1,
        "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
        "directions": [],
        "operations": [
            {"id":"second","scope":{"layers":{"kind":"values","values":[2]},"decode":{"kind":"values","values":[0]}},"action":{"kind":"projection_ablate","direction":"d","coefficient":1.0}},
            {"id":"first","scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"fixed_add","direction":"d","coefficient":1.0}}
        ],
        "readouts": []
    }))
    .unwrap();
    assert!(validate_plan(&plan).is_err());

    let plan: LensPlan = serde_json::from_value(json!({
        "version": 1,
        "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
        "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"unit_l2"}],
        "operations": [
            {"id":"second","scope":{"layers":{"kind":"values","values":[2]},"decode":{"kind":"values","values":[0]}},"action":{"kind":"projection_ablate","direction":"d","coefficient":1.0}},
            {"id":"first","scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"fixed_add","direction":"d","coefficient":1.0}}
        ],
        "readouts": []
    }))
    .unwrap();
    validate_plan(&plan).unwrap();
    assert_eq!(plan.operations[0].id, "second");
    assert_eq!(plan.operations[1].id, "first");
}

#[test]
fn plan_scope_requires_reachable_prefill_or_decode() {
    let scope = Scope {
        layers: Selector::All,
        prefill: None,
        decode: Some(Selector::Values { values: vec![1] }),
    };
    assert!(validate_scope_reachable(&scope, 2, 1, "x").is_err());
    assert!(validate_scope_reachable(&scope, 2, 2, "x").is_ok());
    let scope = Scope {
        layers: Selector::All,
        prefill: None,
        decode: Some(Selector::Values { values: vec![2] }),
    };
    assert!(validate_scope_reachable(&scope, 2, 2, "x").is_err());
    let scope = Scope {
        layers: Selector::All,
        prefill: Some(Selector::Values { values: vec![1] }),
        decode: None,
    };
    assert!(validate_scope_reachable(&scope, 2, 1, "x").is_ok());
}

#[test]
fn only_the_final_prefill_event_and_decode_events_need_logits() {
    assert!(phase_needs_logits(Phase::Prefill(0), 1));
    assert!(!phase_needs_logits(Phase::Prefill(0), 3));
    assert!(!phase_needs_logits(Phase::Prefill(1), 3));
    assert!(phase_needs_logits(Phase::Prefill(2), 3));
    assert!(phase_needs_logits(Phase::Decode(0), 3));
    assert!(phase_needs_logits(Phase::Decode(99), 3));
}

#[test]
fn event_forward_routes_preserve_serial_and_production_topology() {
    use EventForwardRoute::*;

    assert_eq!(
        event_forward_route(true, false, false, false),
        ProductionFullTail
    );
    assert_eq!(
        event_forward_route(false, false, false, false),
        ProductionFullTailDiscardLogits
    );
    assert_eq!(
        event_forward_route(true, true, true, false),
        SerialFullTailCapture
    );
    assert_eq!(
        event_forward_route(false, true, true, true),
        SerialNoTailCapture
    );
    assert_eq!(
        event_forward_route(true, true, false, false),
        SerialFullTailNoCapture
    );
    assert_eq!(
        event_forward_route(false, true, false, false),
        SerialNoTailNoCapture
    );
    assert_eq!(
        event_forward_route(true, false, false, true),
        SerialFullTailNoCapture
    );
    assert_eq!(
        event_forward_route(false, false, false, true),
        SerialNoTailNoCapture
    );
}

#[test]
fn plan_relative_paths_resolve_against_the_plan_directory() {
    let base = Path::new("/tmp/lens-plan");
    assert_eq!(
        resolve_plan_path(base, Path::new("artifacts/readouts.json")),
        PathBuf::from("/tmp/lens-plan/artifacts/readouts.json")
    );
    assert_eq!(
        resolve_plan_path(base, Path::new("/absolute/weights.safetensors")),
        PathBuf::from("/absolute/weights.safetensors")
    );
}

#[test]
fn plan_file_parser_accepts_fractional_coefficients() {
    let plan = parse_plan_bytes(
        br#"{
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"as_stored"}],
            "operations": [{"id":"add","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"all"}},"action":{"kind":"fixed_add","direction":"d","coefficient":0.0001}}],
            "readouts": []
        }"#,
    )
    .unwrap();
    assert_eq!(plan.operations[0].action.coefficient(), 0.0001);
}

#[test]
fn coordinate_swap_requires_distinct_unit_directions() {
    let plan = |source: &str, target: &str, target_normalization: &str| {
        serde_json::from_value::<LensPlan>(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [
                {"id":"source","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"unit_l2"},
                {"id":"target","lens":"t","row":{"kind":"template_row_id","template_row_id":1},"normalization":target_normalization}
            ],
            "operations": [{
                "id":"swap",
                "scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"all"}},
                "action":{"kind":"coordinate_swap","source":source,"target":target,"coefficient":1.0}
            }],
            "readouts": []
        }))
        .unwrap()
    };

    validate_plan(&plan("source", "target", "unit_l2")).unwrap();
    assert!(validate_plan(&plan("source", "source", "unit_l2")).is_err());
    assert!(validate_plan(&plan("source", "target", "as_stored")).is_err());
}

#[test]
fn coordinate_swap_reflection_exchanges_two_lens_coordinates() {
    let source = [1.0_f32, 0.0, 0.0];
    let target = [0.6_f32, 0.8, 0.0];
    let mut activation = [2.0_f32, -1.0, 5.0];
    let source_before = activation
        .iter()
        .zip(source)
        .map(|(&x, v)| x * v)
        .sum::<f32>();
    let target_before = activation
        .iter()
        .zip(target)
        .map(|(&x, v)| x * v)
        .sum::<f32>();
    let reflection = coordinate_swap_reflection_direction(&source, &target, "test").unwrap();
    let projection = activation
        .iter()
        .zip(&reflection)
        .map(|(&x, &u)| x * u)
        .sum::<f32>();
    for (value, &direction) in activation.iter_mut().zip(&reflection) {
        *value -= 2.0 * projection * direction;
    }
    let source_after = activation
        .iter()
        .zip(source)
        .map(|(&x, v)| x * v)
        .sum::<f32>();
    let target_after = activation
        .iter()
        .zip(target)
        .map(|(&x, v)| x * v)
        .sum::<f32>();
    assert!((source_after - target_before).abs() < 1e-5);
    assert!((target_after - source_before).abs() < 1e-5);
    assert!((activation[2] - 5.0).abs() < 1e-6);
    assert!(coordinate_swap_reflection_direction(&source, &[2.0, 0.0, 0.0], "bad").is_err());
}

#[test]
fn published_full_transport_plan_requires_explicit_transfer_and_unique_tokens() {
    let plan = |token_ids: serde_json::Value, allow_unvalidated_transfer: bool| {
        serde_json::from_value::<LensPlan>(json!({
            "version": 1,
            "lenses": [{
                "kind": "published_full_transport",
                "id": "j",
                "artifact": "published",
                "token_ids": token_ids,
                "allow_unvalidated_transfer": allow_unvalidated_transfer
            }],
            "directions": [{
                "id": "concept",
                "lens": "j",
                "row": {"kind": "token_id", "token_id": 42},
                "normalization": "unit_l2"
            }],
            "operations": [{
                "id": "add",
                "scope": {
                    "layers": {"kind": "values", "values": [31]},
                    "prefill": {"kind": "all"}
                },
                "action": {
                    "kind": "residual_l2_fraction",
                    "direction": "concept",
                    "coefficient": 0.01
                }
            }],
            "readouts": []
        }))
        .unwrap()
    };

    let valid = plan(json!([42, 43]), true);
    validate_plan(&valid).unwrap();
    validate_ordinary_plan(&valid).unwrap();
    assert!(matches!(
        &valid.lenses[0],
        LensDefinition::PublishedFullTransport { token_ids, .. } if token_ids == &[42, 43]
    ));
    let expanded = plan(json!((0..64).collect::<Vec<_>>()), true);
    validate_plan(&expanded).unwrap();

    // Transfer authority is checked against the opened artifact, not syntax:
    // an exact-bound data artifact needs no unvalidated-transfer override.
    assert!(validate_plan(&plan(json!([42, 43]), false)).is_ok());
    assert!(validate_plan(&plan(json!([42, 42]), true)).is_err());
    assert!(validate_plan(&plan(json!([]), true)).is_err());
    assert!(validate_plan(&plan(json!([43]), true)).is_err());

    let legacy: LensPlan = serde_json::from_value(json!({
        "version": 1,
        "lenses": [{
            "kind": "published_full_j",
            "id": "j",
            "artifact": "published",
            "token_ids": [42],
            "allow_unvalidated_transfer": true
        }],
        "directions": [],
        "operations": [],
        "readouts": [{
            "id": "read",
            "lens": "j",
            "scope": {"layers":{"kind":"values","values":[31]},"prefill":{"kind":"all"}},
            "top_k": 1
        }]
    }))
    .unwrap();
    assert!(matches!(
        legacy.lenses[0],
        LensDefinition::PublishedFullTransport { .. }
    ));

    let mut required = HashMap::new();
    required.insert("j", (0..63).collect::<BTreeSet<_>>());
    let raw = HashMap::new();
    ensure_projected_full_direction_bank_budget(&expanded, &required, &raw, 5_120).unwrap();
    let mut raw = HashMap::new();
    raw.insert("j", (0..63).collect::<BTreeSet<_>>());
    assert!(
        ensure_projected_full_direction_bank_budget(&expanded, &required, &raw, 5_120).is_err()
    );
}

#[test]
fn published_transport_direction_target_covectors_are_explicit_and_scoped() {
    let plan = |target_covector: Option<&str>, lens: serde_json::Value| {
        let mut direction = json!({
            "id": "concept",
            "lens": "j",
            "row": {"kind": "token_id", "token_id": 42},
            "normalization": "unit_l2"
        });
        if let Some(target_covector) = target_covector {
            direction["target_covector"] = json!(target_covector);
        }
        serde_json::from_value::<LensPlan>(json!({
            "version": 1,
            "lenses": [lens],
            "directions": [direction],
            "operations": [{
                "id": "add",
                "scope": {"layers":{"kind":"values","values":[31]},"prefill":{"kind":"all"}},
                "action": {"kind":"residual_l2_fraction","direction":"concept","coefficient":0.1}
            }],
            "readouts": []
        }))
        .unwrap()
    };
    let published = || {
        json!({
            "kind": "published_full_transport",
            "id": "j",
            "artifact": "published",
            "token_ids": [42],
            "allow_unvalidated_transfer": true
        })
    };

    let implicit = plan(None, published());
    validate_plan(&implicit).unwrap();
    let implicit_direction = implicit.directions[0].lens_row().unwrap();
    assert_eq!(implicit_direction.target_covector, None);
    assert_eq!(
        implicit_direction.effective_target_covector(),
        DirectionTargetCovector::DeployedLogitNumerator
    );
    assert!(
        !serde_json::to_value(&implicit).unwrap()["directions"][0]
            .as_object()
            .unwrap()
            .contains_key("target_covector")
    );

    for (serialized, expected) in [
        (
            "deployed_logit_numerator",
            DirectionTargetCovector::DeployedLogitNumerator,
        ),
        ("raw_lm_head", DirectionTargetCovector::RawLmHead),
        (
            "raw_lm_head_orthogonal_to_deployed_logit_numerator",
            DirectionTargetCovector::RawLmHeadOrthogonalToDeployedLogitNumerator,
        ),
    ] {
        let explicit = plan(Some(serialized), published());
        validate_plan(&explicit).unwrap();
        assert_eq!(
            explicit.directions[0]
                .lens_row()
                .unwrap()
                .effective_target_covector(),
            expected
        );
    }

    let unsupported = plan(
        Some("raw_lm_head"),
        json!({"kind":"native_selected","id":"j","artifact":"native"}),
    );
    assert!(validate_plan(&unsupported).is_err());
}

#[test]
fn raw_direction_orthogonalization_removes_the_readout_axis() {
    let orthogonal = orthogonal_component(&[2.0, 3.0, 4.0], &[1.0, 0.0, 0.0], "test").unwrap();
    assert_eq!(orthogonal, [0.0, 3.0, 4.0]);
    let dot = orthogonal
        .iter()
        .zip([1.0_f32, 0.0, 0.0])
        .map(|(&left, right)| left * right)
        .sum::<f32>();
    assert_eq!(dot, 0.0);

    let axis = [0.3_f32, -0.7, 1.1, 0.2];
    let raw = [1.2_f32, 0.4, -0.5, 2.0];
    let orthogonal = normalize_direction(
        orthogonal_component(&raw, &axis, "test").unwrap(),
        Normalization::UnitL2,
        "test",
    )
    .unwrap();
    let axis = normalize_direction(axis.to_vec(), Normalization::UnitL2, "axis").unwrap();
    let normalized_dot = orthogonal
        .iter()
        .zip(axis)
        .map(|(&left, right)| left * right)
        .sum::<f32>();
    assert!(normalized_dot.abs() < 1.0e-6, "dot={normalized_dot}");

    assert!(orthogonal_component(&[1.0, 2.0], &[2.0, 4.0], "test").is_err());
    assert!(orthogonal_component(&[f32::NAN], &[1.0], "test").is_err());
}

#[test]
fn native_payload_indexing_is_layer_token_hidden_major() {
    let values = (0..2 * 3 * 4).map(|value| value as f32).collect::<Vec<_>>();
    let offset = native_payload_offset(1, 2, 3, 4);
    assert_eq!(&values[offset..offset + 4], &[20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn native_hyper_direction_syntax_is_additive_and_fail_closed() {
    let native = native_hyper_plan(
        Path::new("direction.f32le"),
        json!([{
            "id": "add",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "prefill": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
        }]),
    );
    validate_plan(&native).unwrap();
    assert!(native.directions[0].native_hyper().is_some());
    assert!(validate_ordinary_plan(&native).is_err());

    let legacy: LensPlan = serde_json::from_value(json!({
        "version": 1,
        "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
        "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"as_stored"}],
        "operations": [],
        "readouts": []
    }))
    .unwrap();
    validate_plan(&legacy).unwrap();
    validate_ordinary_plan(&legacy).unwrap();
    assert!(legacy.directions[0].lens_row().is_some());

    let mixed = serde_json::from_value::<LensPlan>(json!({
        "version": 1,
        "lenses": [],
        "directions": [{
            "id": "bad",
            "lens": "x",
            "row": {"kind": "token_id", "token_id": 1},
            "normalization": "as_stored",
            "source": {"kind": "native_hyper_f32", "path": "x", "layer": 23}
        }],
        "operations": [],
        "readouts": []
    }));
    assert!(mixed.is_err());
}

#[test]
fn native_hyper_payload_requires_exact_finite_nonzero_f32() {
    const WIDTH: usize = 10_240;
    let path = temporary_direction_path();
    let mut values = vec![0.0_f32; WIDTH];
    values[17] = 1.0;
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    let loaded = load_native_hyper_direction(&path, WIDTH, "hyper").unwrap();
    assert_eq!(loaded, values);

    std::fs::write(&path, bytemuck::cast_slice(&values[..WIDTH - 1])).unwrap();
    assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());

    values.fill(0.0);
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());

    values[0] = f32::NAN;
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn flash_plan_rejects_wrong_actions_layers_overlap_and_capture_excess() {
    const WIDTH: usize = 10_240;
    let path = temporary_direction_path();
    let mut values = vec![0.0_f32; WIDTH];
    values[0] = 1.0;
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();

    let valid = native_hyper_plan(
        &path,
        json!([{
            "id": "add",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "prefill": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
        }]),
    );
    validate_plan(&valid).unwrap();
    let execution = prepare_qwen4exp_execution_plan(valid, Path::new("/"), &config).unwrap();
    validate_qwen4exp_event_schedule(&execution, 1, 1).unwrap();
    assert_eq!(execution.directions["hyper"].layer, 23);

    let disabled = native_hyper_plan(
        &path,
        json!([{
            "id": "disabled",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "prefill": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": -0.0}
        }]),
    );
    validate_plan(&disabled).unwrap();
    let disabled = prepare_qwen4exp_execution_plan(disabled, Path::new("/"), &config).unwrap();
    validate_qwen4exp_event_schedule(&disabled, 1, 1).unwrap();
    assert!(
        qwen4exp_matching_operation(&disabled, Phase::Prefill(0))
            .unwrap()
            .is_none()
    );

    let wrong_layer = native_hyper_plan(
        &path,
        json!([{
            "id": "add",
            "scope": {
                "layers": {"kind": "values", "values": [22]},
                "prefill": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
        }]),
    );
    assert!(prepare_qwen4exp_execution_plan(wrong_layer, Path::new("/"), &config).is_err());

    let wrong_action = native_hyper_plan(
        &path,
        json!([{
            "id": "project",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "prefill": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "projection_ablate", "direction": "hyper", "coefficient": 1.0}
        }]),
    );
    validate_plan(&wrong_action).unwrap();
    assert!(prepare_qwen4exp_execution_plan(wrong_action, Path::new("/"), &config).is_err());

    let overlapping = native_hyper_plan(
        &path,
        json!([
            {
                "id": "first",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            },
            {
                "id": "second",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.5}
            }
        ]),
    );
    let execution = prepare_qwen4exp_execution_plan(overlapping, Path::new("/"), &config).unwrap();
    assert!(validate_qwen4exp_event_schedule(&execution, 1, 1).is_err());

    let excessive = native_hyper_plan(
        &path,
        json!([{
            "id": "many",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "prefill": {"kind": "all"}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
        }]),
    );
    let execution = prepare_qwen4exp_execution_plan(excessive, Path::new("/"), &config).unwrap();
    validate_qwen4exp_event_schedule(&execution, 32, 1).unwrap();
    assert!(validate_qwen4exp_event_schedule(&execution, 33, 1).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn flash_decode_schedule_and_capture_metadata_are_explicit() {
    const WIDTH: usize = 10_240;
    let path = temporary_direction_path();
    let mut values = vec![0.0_f32; WIDTH];
    values[0] = 1.0;
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    let plan = native_hyper_plan(
        &path,
        json!([{
            "id": "decode-add",
            "scope": {
                "layers": {"kind": "values", "values": [23]},
                "decode": {"kind": "values", "values": [0]}
            },
            "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
        }]),
    );
    validate_reachable_scopes(&plan, 1, 2).unwrap();
    let execution = prepare_qwen4exp_execution_plan(
        plan,
        Path::new("/"),
        &Qwen4ExpConfig::flash_next_reference(),
    )
    .unwrap();
    validate_qwen4exp_event_schedule(&execution, 1, 2).unwrap();
    assert!(
        qwen4exp_matching_operation(&execution, Phase::Prefill(0))
            .unwrap()
            .is_none()
    );
    let (operation, layer) = qwen4exp_matching_operation(&execution, Phase::Decode(0))
        .unwrap()
        .unwrap();
    assert_eq!(operation.id, "decode-add");
    assert_eq!(layer, 23);

    let capture = NativeHyperCapture {
        operation_id: "decode-add".into(),
        layer: 23,
        phase: "decode",
        index: 0,
        position: 1,
        coordinate: "qwen4exp_persistent_post_layer_hyper_state",
        capture_stage: "after_fixed_add",
        shape: [4, 2_560],
        flattening: "branch_major_hidden_minor",
        direction_normalization: "as_stored",
        coefficient: 0.25,
        values: vec![1.0, 2.0],
    };
    let encoded = serde_json::to_value(capture).unwrap();
    assert_eq!(encoded["capture_stage"], "after_fixed_add");
    assert_eq!(encoded["direction_normalization"], "as_stored");
    assert!(encoded.get("normalization").is_none());
    assert_eq!(encoded["shape"], json!([4, 2560]));
    std::fs::remove_file(path).unwrap();
}
