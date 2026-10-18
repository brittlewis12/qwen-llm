//! Opt-in real-weight replay using named fixtures under QWEN_MODEL_SEARCH_PATH.
//! Bounds logits and complete inventories; hidden/pre tensors are characterization only.
//! The model config marker is a copy/symlink beside config.json, index, and tokenizer.
//! The optional base Engram marker is a symlink to shard 47 beside actual shards 47/48.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

const TOKENS: [u32; 7] = [0, 5, 223, 939, 23, 695, 22];
const POLICY: &str = "lower_expert_id_then_id_sorted";
const CPU_REFERENCE: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu",
    env: &[],
    filename: "deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu.json",
};
const MODEL_CONFIG: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-config",
    env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-config.json",
};
const ENGRAM_SHARD: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-flash-base-engram-47-of-48",
    env: &[],
    filename: "DeepSeek-V4.1-Flash-base-engram-00047-of-00048.safetensors",
};
const ENGRAM_FILES: [&str; 2] = [
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
];

fn model_root(marker: &Path) -> Result<PathBuf> {
    let root = marker.parent().context("model marker has no parent")?;
    ensure!(
        manifest(marker)? == manifest(&root.join("config.json"))?,
        "model marker differs from adjacent config.json"
    );
    for file in ["model.safetensors.index.json", "tokenizer.json"] {
        ensure!(
            root.join(file).is_file(),
            "model marker missing adjacent {file}"
        );
    }
    Ok(root.to_owned())
}

fn engram_root(model: &Path, resolve_marker: impl FnOnce() -> PathBuf) -> Result<PathBuf> {
    if ENGRAM_FILES.iter().all(|file| model.join(file).is_file()) {
        return Ok(model.to_owned());
    }
    let marker = resolve_marker();
    let root = marker.parent().context("Engram marker has no parent")?;
    for file in ENGRAM_FILES {
        ensure!(
            root.join(file).is_file(),
            "Engram marker missing adjacent {file}"
        );
    }
    ensure!(
        marker.canonicalize()? == root.join(ENGRAM_FILES[0]).canonicalize()?,
        "Engram marker must alias adjacent shard 47"
    );
    Ok(root.to_owned())
}

fn replay_command(model: &Path, engram: &Path, trace: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen"));
    command
        .arg("deepseek-v41")
        .arg("--model-dir")
        .arg(model)
        .arg("--engram-dir")
        .arg(engram)
        .args([
            "--token-ids",
            "0,5,223,939,23,695,22",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--context",
            "16",
            "--reserve-mib",
            "1024",
        ])
        .arg("--trace-dir")
        .arg(trace);
    command
}

fn manifest(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

const LONG_TOKENS: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-native-f32-10k-attention",
    env: &[],
    filename: "deepseek-v4.1-reap-272e-native-f32-10k-attention.json",
};
const LONG_POSITIONS: [usize; 4] = [2046, 2047, 9998, 9999];
const LONG_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

fn long_tokens(meta: &Value) -> Result<String> {
    ensure!(
        meta["format"] == "dsv41-long-context-v1",
        "wrong capture format"
    );
    ensure!(
        meta["mode"] == "NativeF32" && meta["status"] == "success",
        "incomplete NativeF32 fixture"
    );
    ensure!(
        meta["requested_checkpoints"] == json!(LONG_POSITIONS),
        "wrong checkpoint request"
    );
    let tokens = meta["token_ids"].as_array().context("missing token_ids")?;
    ensure!(tokens.len() == 10000, "expected exactly 10000 tokens");
    tokens
        .iter()
        .map(|token| {
            let id = token
                .as_u64()
                .context("token must be an unsigned integer")?;
            ensure!(id < 129280, "token outside vocabulary");
            Ok(id.to_string())
        })
        .collect::<Result<Vec<_>>>()
        .map(|ids| ids.join(","))
}

fn long_capture_command(model: &Path, engram: &Path, capture: &Path, tokens: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen"));
    command
        .args(["deepseek-v41", "--gpu-step", "--model-dir"])
        .arg(model)
        .arg("--engram-dir")
        .arg(engram)
        .args([
            "--token-ids",
            tokens,
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--context",
            "10000",
            "--reserve-mib",
            "1024",
            "--attention-capture-positions",
            "2046,2047,9998,9999",
        ])
        .arg("--attention-capture-dir")
        .arg(capture);
    command
}

fn long_compare_command(model: &Path, capture: &Path) -> Command {
    let mut command = Command::new("uv");
    command
        .args(["run", "--offline", "--script"])
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/reference/deepseek_v41/cpu/long_context.py"),
        )
        .arg("--model")
        .arg(model)
        .arg("--capture")
        .arg(capture)
        .args(["--atol", "0.0001", "--rtol", "0"])
        .env("CUDA_VISIBLE_DEVICES", "");
    command
}

fn check_long_report(report: &Value) -> Result<()> {
    let rows = report.as_array().context("expected comparison rows")?;
    ensure!(rows.len() == 32, "expected exactly 32 checkpoints");
    for (row, (position, layer)) in rows.iter().zip(
        LONG_POSITIONS
            .into_iter()
            .flat_map(|p| LONG_LAYERS.into_iter().map(move |l| (p, l))),
    ) {
        ensure!(
            row["position"] == position && row["layer"] == layer,
            "wrong checkpoint inventory"
        );
        ensure!(
            row["owner"] == layer.min(20) && row["index_source"] == layer,
            "wrong checkpoint owners"
        );
        for key in ["selection_exact", "owner_exact"] {
            ensure!(row[key] == true, "{key} mismatch");
        }
        for key in ["selection_boundary_tie", "candidate_boundary_tie"] {
            ensure!(row[key] == false, "{key} missing or tied");
        }
        if layer >= 20 {
            ensure!(row["candidates_exact"] == true, "candidate mismatch");
            // This 10k witness does not exercise candidate pruning.
            ensure!(
                row["candidate_discard"] == 0,
                "unexpected candidate pruning"
            );
        }
        ensure!(row["output"]["pass"] == true, "attention comparison failed");
        bounded(&row["output"], "max_abs", 1e-4)?;
        bounded(&row["output"], "relative_l2", 3e-5)?;
    }
    Ok(())
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-native-f32-10k-attention"]
fn gpu_step_ten_thousand_token_attention_oracle() -> Result<()> {
    let tokens = long_tokens(&manifest(LONG_TOKENS.required())?)?;
    let model = model_root(MODEL_CONFIG.required())?;
    let engram = engram_root(&model, || ENGRAM_SHARD.required().clone())?;
    let output = std::env::temp_dir().join(format!(
        "qwen-v41-long-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&output)?;
    eprintln!("long oracle evidence retained at {}", output.display());
    let capture = output.join("capture");
    // Inherit output rather than retaining 10k forward logs in memory.
    ensure!(
        long_capture_command(&model, &engram, &capture, &tokens)
            .status()?
            .success(),
        "GPU capture failed"
    );
    ensure!(
        long_tokens(&manifest(&capture.join("manifest.json"))?)? == tokens,
        "fresh capture tokens differ from fixture"
    );
    let report = output.join("comparison.json");
    ensure!(
        long_compare_command(&model, &capture)
            .stdout(std::fs::File::create(&report)?)
            .status()?
            .success(),
        "CPU replay failed"
    );
    check_long_report(&manifest(&report)?)
}

#[test]
fn long_context_commands_are_fresh_gpu_and_independent_cpu() {
    let gpu = long_capture_command(
        Path::new("model"),
        Path::new("engram"),
        Path::new("fresh"),
        "1,2",
    );
    assert_eq!(gpu.get_program(), env!("CARGO_BIN_EXE_qwen"));
    let args: Vec<_> = gpu.get_args().map(|a| a.to_str().unwrap()).collect();
    assert_eq!(
        args,
        [
            "deepseek-v41",
            "--gpu-step",
            "--model-dir",
            "model",
            "--engram-dir",
            "engram",
            "--token-ids",
            "1,2",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--context",
            "10000",
            "--reserve-mib",
            "1024",
            "--attention-capture-positions",
            "2046,2047,9998,9999",
            "--attention-capture-dir",
            "fresh"
        ]
    );
    let cpu = long_compare_command(Path::new("model"), Path::new("fresh"));
    let args: Vec<_> = cpu.get_args().map(|a| a.to_str().unwrap()).collect();
    assert_eq!(&args[..3], ["run", "--offline", "--script"]);
    assert!(args[3].ends_with("/cpu/long_context.py"));
    assert_eq!(
        &args[4..],
        [
            "--model",
            "model",
            "--capture",
            "fresh",
            "--atol",
            "0.0001",
            "--rtol",
            "0"
        ]
    );
}

#[test]
fn long_context_manifest_validation() -> Result<()> {
    let valid = json!({"format":"dsv41-long-context-v1", "mode":"NativeF32", "status":"success",
        "requested_checkpoints":LONG_POSITIONS, "token_ids":vec![0;10000]});
    assert_eq!(long_tokens(&valid)?.split(',').count(), 10000);
    for (key, bad) in [
        ("format", json!("other")),
        ("mode", json!("BF16")),
        ("status", json!("incomplete")),
        ("requested_checkpoints", json!([9999])),
        ("token_ids", json!([0])),
    ] {
        let mut meta = valid.clone();
        meta[key] = bad;
        assert!(long_tokens(&meta).is_err(), "accepted {key}");
    }
    for bad in [json!(-1), json!(1.5), json!(129280), Value::Null] {
        let mut meta = valid.clone();
        meta["token_ids"][9999] = bad;
        assert!(long_tokens(&meta).is_err());
    }
    Ok(())
}

#[test]
fn long_context_report_negative_controls() -> Result<()> {
    let valid = Value::Array(
        LONG_POSITIONS
            .into_iter()
            .flat_map(|p| {
                LONG_LAYERS.into_iter().map(move |l| {
                    json!({"position":p,"layer":l,"owner":l.min(20),"index_source":l,
            "selection_exact":true,"owner_exact":true,"candidates_exact":true,"candidate_discard":0,
            "selection_boundary_tie":false,"candidate_boundary_tie":false,
            "output":{"pass":true,"max_abs":2.8848648e-5,"relative_l2":1.00598e-5}})
                })
            })
            .collect(),
    );
    check_long_report(&valid)?;
    for key in [
        "position",
        "layer",
        "owner",
        "index_source",
        "selection_exact",
        "owner_exact",
        "candidates_exact",
        "candidate_discard",
        "selection_boundary_tie",
        "candidate_boundary_tie",
    ] {
        let mut report = valid.clone();
        report[31][key] = Value::Null;
        assert!(
            check_long_report(&report).is_err(),
            "accepted missing {key}"
        );
    }
    for (key, bad) in [
        ("pass", json!(false)),
        ("max_abs", json!(0.000101)),
        ("relative_l2", json!(0.000031)),
        ("max_abs", json!(-1)),
        ("relative_l2", Value::Null),
    ] {
        let mut report = valid.clone();
        report[31]["output"][key] = bad;
        assert!(check_long_report(&report).is_err(), "accepted {key}");
    }
    let mut report = valid.clone();
    report[31] = report[0].clone();
    assert!(check_long_report(&report).is_err());
    let mut rows = valid.as_array().unwrap().clone();
    rows.pop();
    assert!(check_long_report(&json!(rows)).is_err());
    rows.extend([valid[31].clone(), valid[31].clone()]);
    assert!(check_long_report(&json!(rows)).is_err());
    Ok(())
}

fn check_policy(reference: &Value, runtime: &Value) -> Result<()> {
    for value in [reference, runtime] {
        ensure!(value["mode"] == "NativeF32", "expected NativeF32");
        ensure!(
            value["router_tie_policy"] == POLICY,
            "router policy mismatch"
        );
    }
    ensure!(runtime["status"] == "success", "incomplete runtime trace");
    ensure!(runtime["dtype"] == "float32", "runtime dtype mismatch");
    ensure!(
        runtime["byte_order"] == "little",
        "runtime byte order mismatch"
    );
    let positions = runtime["positions"]
        .as_array()
        .context("missing positions")?;
    ensure!(
        positions.len() == TOKENS.len(),
        "wrong consumed token count"
    );
    for (i, token) in TOKENS.iter().enumerate() {
        ensure!(
            positions[i]["position"] == i && positions[i]["token_id"] == *token,
            "wrong consumed token at {i}"
        );
    }
    Ok(())
}

fn bounded(value: &Value, key: &str, maximum: f64) -> Result<()> {
    let number = value[key]
        .as_f64()
        .with_context(|| format!("missing {key}"))?;
    ensure!(
        number.is_finite() && number >= 0.0 && number <= maximum,
        "{key}={number} exceeds {maximum}"
    );
    Ok(())
}

fn check_report(report: &Value) -> Result<()> {
    ensure!(report["mode"] == "NativeF32", "comparison mode mismatch");
    ensure!(
        report["positions"] == 7 && report["layers"] == 40 && report["tensors_compared"] == 567,
        "incomplete comparison"
    );
    ensure!(
        report["token_check"] == "complete" && report["token_positions_verified"] == 7,
        "incomplete token agreement"
    );
    ensure!(
        report["shapes"] == json!({"hidden":[4,5120],"pre":[4],"logits":[129280]}),
        "unexpected trace geometry"
    );
    let logits = report["logits"]
        .as_array()
        .context("missing logits metrics")?;
    ensure!(logits.len() == 7, "missing logits positions");
    for (position, metrics) in logits.iter().enumerate() {
        ensure!(metrics["position"] == position, "wrong logits position");
        let reference = metrics["reference_top1"]
            .as_u64()
            .context("missing reference top1")?;
        ensure!(
            reference < 129280
                && metrics["runtime_top1"].as_u64() == Some(reference)
                && metrics["top1_match"] == true,
            "top1 mismatch at {position}"
        );
        // Baseline regression ceilings: observed 5.49e-5 maxabs and 4.82e-11 KL.
        // These bound logits only, not layerwise certification or source arithmetic.
        bounded(metrics, "maxabs", 1e-4)?;
        bounded(metrics, "kl_reference_to_runtime", 1e-10)?;
    }
    Ok(())
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu"]
fn native_f32_seven_token_cpu_oracle() -> Result<()> {
    replay(false)
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu"]
fn gpu_step_seven_token_cpu_oracle() -> Result<()> {
    replay(true)
}

fn replay(gpu_step: bool) -> Result<()> {
    let marker = CPU_REFERENCE.required();
    let reference = marker.parent().context("reference has no parent")?;
    let reference_meta = manifest(&reference.join("run.json"))?;
    ensure!(
        manifest(marker)? == reference_meta,
        "named fixture manifest differs from run.json"
    );
    ensure!(
        reference_meta["mode"] == "NativeF32" && reference_meta["router_tie_policy"] == POLICY,
        "reference must use stable NativeF32 routing (not the upstream-tie negative control)"
    );
    let model = model_root(MODEL_CONFIG.required())?;
    let engram = engram_root(&model, || ENGRAM_SHARD.required().clone())?;
    let output = std::env::temp_dir().join(format!(
        "qwen-v41-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&output)?;
    eprintln!("oracle evidence retained at {}", output.display());
    let trace = output.join("trace");
    let mut command = replay_command(&model, &engram, &trace);
    if gpu_step {
        command.arg("--gpu-step");
    }
    let status = command.status()?;
    ensure!(status.success(), "qwen replay failed: {status}");
    let runtime = manifest(&trace.join("run.json"))?;
    check_policy(&reference_meta, &runtime)?;
    ensure!(
        runtime["execution"]
            == if gpu_step {
                "gpu_step"
            } else {
                "cpu_boundaries"
            },
        "wrong execution mode"
    );
    let compare = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/reference/deepseek_v41/cpu/compare.py");
    let result = Command::new("uv")
        .args(["run", "--offline", "--script"])
        .arg(compare)
        .arg("--reference-dir")
        .arg(reference)
        .arg("--runtime-dir")
        .arg(&trace)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()?;
    std::fs::write(output.join("comparison.json"), &result.stdout)?;
    ensure!(
        result.status.success(),
        "compare failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout)?;
    check_report(&report)?;
    ensure!(
        runtime["generated_token_ids"] == json!([report["logits"][6]["reference_top1"]]),
        "final sampled token differs from CPU argmax"
    );
    Ok(())
}

#[test]
fn comparison_gate_rejects_characterization_failures() {
    let mut report = json!({"mode":"NativeF32", "positions":7,"layers":40,
        "tensors_compared":567,"token_check":"complete","token_positions_verified":7,
        "shapes":{"hidden":[4,5120],"pre":[4],"logits":[129280]},
        "logits": (0..7).map(|position| json!({"position":position,"reference_top1":5,
            "runtime_top1":5,"top1_match":true,"maxabs":5.49e-5,
            "kl_reference_to_runtime":4.82e-11})).collect::<Vec<_>>()});
    check_report(&report).unwrap();
    for (key, value) in [
        ("maxabs", json!(1.01e-4)),
        ("kl_reference_to_runtime", json!(1.01e-10)),
        ("top1_match", json!(false)),
        ("runtime_top1", json!(6)),
        ("maxabs", Value::Null),
    ] {
        let mut bad = report.clone();
        bad["logits"][0][key] = value;
        assert!(check_report(&bad).is_err(), "accepted {key}");
    }
    report["token_positions_verified"] = json!(6);
    assert!(check_report(&report).is_err());
}

#[test]
fn replay_uses_bounded_reserve_and_default_cache_policy() {
    let command = replay_command(Path::new("model"), Path::new("base"), Path::new("trace"));
    let args: Vec<_> = command
        .get_args()
        .map(|arg| arg.to_str().unwrap())
        .collect();
    assert_eq!(
        args,
        [
            "deepseek-v41",
            "--model-dir",
            "model",
            "--engram-dir",
            "base",
            "--token-ids",
            "0,5,223,939,23,695,22",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--context",
            "16",
            "--reserve-mib",
            "1024",
            "--trace-dir",
            "trace"
        ]
    );
}

#[test]
fn named_root_fixtures_and_colocated_engrams() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "qwen-v41-root-fixtures-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    let result = (|| -> Result<()> {
        let model = root.join("checkpoint");
        let base = root.join("base");
        std::fs::create_dir(&model)?;
        std::fs::create_dir(&base)?;
        let marker = model.join(MODEL_CONFIG.filename());
        std::fs::write(&marker, b"{}")?;
        std::fs::write(model.join("config.json"), b"{}")?;
        assert!(model_root(&marker).is_err());
        for file in ["model.safetensors.index.json", "tokenizer.json"] {
            std::fs::write(model.join(file), b"{}")?;
        }
        assert_eq!(model_root(&marker)?, model);
        std::fs::write(&marker, b"{\"wrong\":true}")?;
        assert!(model_root(&marker).is_err());
        std::fs::write(&marker, b"{}")?;
        let alias = base.join(ENGRAM_SHARD.filename());
        std::fs::write(base.join(ENGRAM_FILES[0]), b"synthetic header")?;
        std::os::unix::fs::symlink(ENGRAM_FILES[0], &alias)?;
        assert!(engram_root(&model, || alias.clone()).is_err());
        std::fs::write(base.join(ENGRAM_FILES[1]), b"synthetic header")?;
        assert_eq!(engram_root(&model, || alias.clone())?, base);
        // Resolve through the real shared resolver in a child: no process-global env edits.
        let child = Command::new(std::env::current_exe()?)
            .args([
                "named_root_fixture_lookup_child",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env("QWEN_MODEL_SEARCH_PATH", &root)
            .output()?;
        ensure!(
            child.status.success(),
            "fixture lookup child failed: {}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        for file in ENGRAM_FILES {
            std::fs::write(model.join(file), b"synthetic header")?;
        }
        assert_eq!(
            engram_root(&model, || panic!(
                "co-located shards need no second fixture"
            ))?,
            model
        );
        Ok(())
    })();
    std::fs::remove_dir_all(&root)?;
    result
}

#[test]
#[ignore = "subprocess fixture; run by named_root_fixtures_and_colocated_engrams"]
fn named_root_fixture_lookup_child() -> Result<()> {
    let model = model_root(MODEL_CONFIG.required())?;
    let base = engram_root(&model, || ENGRAM_SHARD.required().clone())?;
    ensure!(
        model.file_name().unwrap() == "checkpoint" && base.file_name().unwrap() == "base",
        "named fixtures resolved to unexpected roots"
    );
    Ok(())
}

#[test]
fn upstream_tie_negative_control_is_not_an_agreement_fixture() {
    let mut reference = json!({"mode":"NativeF32","router_tie_policy":POLICY});
    let runtime = json!({"mode":"NativeF32","router_tie_policy":POLICY,
        "status":"success","dtype":"float32","byte_order":"little",
        "positions":TOKENS.iter().enumerate().map(|(position, token)|
            json!({"position":position,"token_id":token})).collect::<Vec<_>>()});
    check_policy(&reference, &runtime).unwrap();
    reference["router_tie_policy"] = json!("upstream_torch_topk");
    assert!(check_policy(&reference, &runtime).is_err());
    for (key, value) in [
        ("status", json!("incomplete")),
        ("dtype", json!("bfloat16")),
        ("byte_order", json!("big")),
        ("mode", json!("ReferenceFp8")),
    ] {
        let mut bad = runtime.clone();
        bad[key] = value;
        assert!(check_policy(&runtime, &bad).is_err(), "accepted {key}");
    }
}
