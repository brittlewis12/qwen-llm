//! Separate GGUF execution-only logit acceptance and HF conversion characterization.
//! Hidden states and long teacher-forced attention remain characterization only.
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::{forward::Forward, gguf::ResidentWeights};
use qwen_llm::{gguf::GgufFile, metal::MetalContext};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[path = "support/v41_capture.rs"]
mod capture;
#[path = "support/v41_characterization.rs"]
mod characterization;
#[path = "support/v41_execution.rs"]
mod execution;
#[path = "support/v41_prefill.rs"]
mod prefill;
#[path = "support/v41_trace.rs"]
mod trace;
use characterization::{
    check_characterization, check_long_characterization, emit_characterization,
};

const TOKENS: [u32; 7] = [0, 5, 223, 939, 23, 695, 22];
const POLICY: &str = "lower_expert_id_then_id_sorted";
const CPU_REFERENCE: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu",
    env: &[],
    filename: "deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu.json",
};
const GGUF_CPU_REFERENCE: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-gguf-native-f32-7t-40l-cpu",
    env: &[],
    filename: "deepseek-v4.1-reap-272e-gguf-native-f32-7t-40l-cpu.json",
};
const GGUF_MODEL: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-mxfp4-q8engram",
    env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf",
};
const OFFLINE_CONFIG: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-cpu-source-config",
    env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-config.json",
};

fn offline_source_root(marker: &Path) -> Result<PathBuf> {
    let root = marker.parent().context("source marker has no parent")?;
    ensure!(
        manifest(marker)? == manifest(&root.join("config.json"))?,
        "source marker differs from config.json"
    );
    ensure!(
        root.join("model.safetensors.index.json").is_file(),
        "CPU source index missing"
    );
    Ok(root.to_owned())
}

fn load_characterization_model(ctx: &MetalContext, path: &Path) -> Result<ResidentWeights> {
    let gguf = Arc::new(GgufFile::open(path)?);
    let plan = ResidentWeights::plan(&gguf)?;
    ensure!(
        plan.layers == 40
            && plan.config.hidden == 5120
            && plan.config.vocab == 129280
            && plan.config.experts == 272,
        "GGUF does not match the HF comparison's geometry"
    );
    Ok(ResidentWeights::load(ctx, gguf)?)
}

fn replay_trace(model: &Path, path: &Path, execution_only: bool) -> Result<()> {
    qwen_cli::shutdown::install()?;
    qwen_cli::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let weights = load_characterization_model(&ctx, model)?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let mut state = forward.new_state_with_gpu_step(TOKENS.len())?;
    let mut trace = if execution_only {
        trace::Trace::create_execution(path, &weights.config, 40, &TOKENS)?
    } else {
        trace::Trace::create(path, &weights.config, 40, &TOKENS)?
    };
    let mut logits = Vec::new();
    for (position, token) in TOKENS.into_iter().enumerate() {
        trace.begin_step(position, token)?;
        logits = forward
            .step_with_trace(
                &mut state,
                token,
                Some(qwen_cli::shutdown::cancellation_flag()),
                Some(&mut |layer| trace.layer(layer).map_err(|e| e.to_string())),
            )
            .map_err(anyhow::Error::msg)?;
        trace.finish_step(&logits)?;
    }
    qwen_cli::shutdown::checkpoint()?;
    let token = qwen_llm::sampling::Sampler::new(Default::default())?
        .sample(&logits)?
        .token;
    trace.finish(&[u32::try_from(token)?], None)
}

fn capture_attention(model: &Path, path: &Path, tokens: &[u32]) -> Result<()> {
    qwen_cli::shutdown::install()?;
    qwen_cli::shutdown::checkpoint()?;
    ensure!(
        tokens.len() == 10000,
        "attention oracle needs exactly 10000 tokens"
    );
    let ctx = MetalContext::new()?;
    let weights = load_characterization_model(&ctx, model)?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let mut state = forward.new_state_with_gpu_step(tokens.len())?;
    let mut capture = capture::Writer::create(
        path,
        &LONG_POSITIONS,
        json!({"gguf":model,"execution":"gpu_step","provenance_verified":false}),
    )?;
    for (position, &token) in tokens.iter().enumerate() {
        capture.begin_step(position, token)?;
        if let Err(error) = forward.step_with_capture(
            &mut state,
            token,
            Some(qwen_cli::shutdown::cancellation_flag()),
            None,
            Some(&mut capture),
        ) {
            capture.failure(&error, qwen_cli::shutdown::checkpoint().is_err());
            anyhow::bail!(error);
        }
    }
    qwen_cli::shutdown::checkpoint()?;
    capture.finish()
}

fn manifest(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn compare_command(reference: &Path, trace: &Path) -> Command {
    let mut command = Command::new("uv");
    command
        .args(["run", "--offline", "--script"])
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/reference/deepseek_v41/cpu/compare.py"),
        )
        .arg("--reference-dir")
        .arg(reference)
        .arg("--runtime-dir")
        .arg(trace)
        .env("CUDA_VISIBLE_DEVICES", "");
    command
}

#[test]
fn seven_token_comparison_has_no_numerical_threshold_arguments() {
    let command = compare_command(Path::new("hf-reference"), Path::new("gguf-trace"));
    let args: Vec<_> = command.get_args().map(|a| a.to_str().unwrap()).collect();
    assert_eq!(&args[..3], ["run", "--offline", "--script"]);
    assert!(args[3].ends_with("/cpu/compare.py"));
    assert_eq!(
        &args[4..],
        [
            "--reference-dir",
            "hf-reference",
            "--runtime-dir",
            "gguf-trace"
        ]
    );
    assert_eq!(
        GGUF_MODEL.filename(),
        "DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf"
    );
}

const LONG_TOKENS: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-native-f32-10k-attention",
    env: &[],
    filename: "deepseek-v4.1-reap-272e-native-f32-10k-attention.json",
};
const LONG_POSITIONS: [usize; 4] = capture::DEFAULT_CHECKPOINTS;
const LONG_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

fn long_tokens(meta: &Value) -> Result<Vec<u32>> {
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
            Ok(id as u32)
        })
        .collect::<Result<Vec<_>>>()
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

fn check_legacy_tight_long_report(report: &Value) -> Result<()> {
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
#[ignore = "model characterization: deepseek-v4.1-reap-272e-mxfp4-q8engram"]
fn hf_conversion_ten_thousand_token_attention_characterization() -> Result<()> {
    let tokens = long_tokens(&manifest(LONG_TOKENS.required())?)?;
    let model = GGUF_MODEL.required();
    let output = std::env::temp_dir().join(format!(
        "qwen-v41-long-characterization-{}-{}",
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
    eprintln!(
        "long HF conversion characterization (not acceptance) retained at {}",
        output.display()
    );
    let capture = output.join("capture");
    capture_attention(model, &capture, &tokens)?;
    ensure!(
        long_tokens(&manifest(&capture.join("manifest.json"))?)? == tokens,
        "fresh capture tokens differ from fixture"
    );
    let report = output.join("comparison.json");
    let status = long_compare_command(&offline_source_root(OFFLINE_CONFIG.required())?, &capture)
        .stdout(std::fs::File::create(&report)?)
        .status()?;
    // The existing CPU tool exits 1 on numerical/selection differences. Those
    // differences are measurements here; malformed or nonfinite reports still fail.
    ensure!(
        matches!(status.code(), Some(0 | 1)),
        "CPU diagnostic failed: {status}"
    );
    let report = manifest(&report)?;
    check_long_characterization(&report)?;
    emit_characterization(
        &output,
        "teacher_forced_attention_hf_source_vs_converted_gguf",
        &report,
    )
}

#[test]
fn long_context_comparison_remains_independent_cpu() {
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
    assert_eq!(long_tokens(&valid)?.len(), 10000);
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
    check_legacy_tight_long_report(&valid)?;
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
            check_legacy_tight_long_report(&report).is_err(),
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
        assert!(
            check_legacy_tight_long_report(&report).is_err(),
            "accepted {key}"
        );
    }
    let mut report = valid.clone();
    report[31] = report[0].clone();
    assert!(check_legacy_tight_long_report(&report).is_err());
    let mut rows = valid.as_array().unwrap().clone();
    rows.pop();
    assert!(check_legacy_tight_long_report(&json!(rows)).is_err());
    rows.extend([valid[31].clone(), valid[31].clone()]);
    assert!(check_legacy_tight_long_report(&json!(rows)).is_err());
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

fn check_legacy_tight_report(report: &Value) -> Result<()> {
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
#[ignore = "model characterization: deepseek-v4.1-reap-272e-mxfp4-q8engram"]
fn hf_conversion_seven_token_characterization() -> Result<()> {
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
    let model = GGUF_MODEL.required();
    let output = std::env::temp_dir().join(format!(
        "qwen-v41-characterization-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&output)?;
    eprintln!(
        "HF conversion characterization (not acceptance) retained at {}",
        output.display()
    );
    let trace = output.join("trace");
    replay_trace(model, &trace, false)?;
    let runtime = manifest(&trace.join("run.json"))?;
    check_policy(&reference_meta, &runtime)?;
    ensure!(
        runtime["execution"] == "gpu_step",
        "wrong production execution mode"
    );
    let result = compare_command(reference, &trace).output()?;
    std::fs::write(output.join("comparison.json"), &result.stdout)?;
    ensure!(
        result.status.success(),
        "compare failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout)?;
    check_characterization(&report)?;
    ensure!(
        runtime["generated_token_ids"] == json!([report["logits"][6]["runtime_top1"]]),
        "final sampled token differs from runtime argmax"
    );
    emit_characterization(
        &output,
        "seven_token_hf_source_vs_converted_gguf_conversion_and_execution",
        &report,
    )
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-mxfp4-q8engram"]
fn independent_gguf_seven_token_execution_acceptance() -> Result<()> {
    execution::run()
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-mxfp4-q8engram"]
fn gguf_native_f32_prefill_65_token_serial_snapshot_parity() -> Result<()> {
    use prefill::compare;

    qwen_cli::shutdown::install()?;
    qwen_cli::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let weights = load_characterization_model(&ctx, GGUF_MODEL.required())?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    // Reuse the oracle prompt IDs: 65 rows cross two full 32-row chunks and a tail.
    let prompt: Vec<_> = TOKENS.into_iter().cycle().take(65).collect();
    let continuation = [223, 939, 23];
    let cancel = Some(qwen_cli::shutdown::cancellation_flag());
    let mut serial = forward.new_state_with_gpu_step(prompt.len() + continuation.len())?;
    let mut batch = forward.new_state_with_gpu_step(prompt.len() + continuation.len())?;
    let mut failures = Vec::new();
    for &token in &prompt[..prompt.len() - 1] {
        forward
            .advance(&mut serial, token, cancel)
            .map_err(anyhow::Error::msg)?;
    }
    let expected = forward
        .step(&mut serial, *prompt.last().unwrap(), cancel)
        .map_err(anyhow::Error::msg)?;
    let actual = forward.prefill(&mut batch, &prompt, cancel)?;
    ensure!(
        serial.position() == prompt.len() && batch.position() == prompt.len(),
        "prompt position mismatch"
    );
    compare("65-row prompt", &actual, &expected, &mut failures)?;

    let snapshot = forward.capture_snapshot(&batch)?;
    ensure!(
        snapshot.position() == prompt.len(),
        "snapshot position mismatch"
    );
    let mut expected_continuation = Vec::new();
    let mut packed_continuation = Vec::new();
    for (i, &token) in continuation.iter().enumerate() {
        let expected = forward
            .step(&mut serial, token, cancel)
            .map_err(anyhow::Error::msg)?;
        let actual = forward
            .step(&mut batch, token, cancel)
            .map_err(anyhow::Error::msg)?;
        compare(
            &format!("uninterrupted packed-prefix scalar continuation {i}"),
            &actual,
            &expected,
            &mut failures,
        )?;
        expected_continuation.push(expected);
        packed_continuation.push(actual);
    }
    for grow in [false, true] {
        forward.restore_snapshot(&mut batch, &snapshot)?;
        ensure!(
            batch.position() == prompt.len(),
            "restore position mismatch"
        );
        if grow {
            let old_capacity = batch.capacity();
            forward.ensure_capacity(&mut batch, old_capacity + 1)?;
            ensure!(
                batch.capacity() > old_capacity && batch.position() == prompt.len(),
                "growth did not preserve prefix"
            );
        }
        for (i, &token) in continuation.iter().enumerate() {
            let actual = forward
                .step(&mut batch, token, cancel)
                .map_err(anyhow::Error::msg)?;
            let label = format!("restored scalar continuation {i} growth={grow}");
            compare(&label, &actual, &packed_continuation[i], &mut failures)?;
            if !actual
                .iter()
                .zip(&packed_continuation[i])
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                failures.push(format!(
                    "{label}: not bitwise equal to uninterrupted packed-prefix scalar continuation"
                ));
            }
        }
        ensure!(
            batch.position() == 68,
            "scalar continuation position mismatch"
        );
    }
    // Restore over a used state, so stale continuation rows must not survive.
    forward.restore_snapshot(&mut batch, &snapshot)?;
    ensure!(
        batch.position() == prompt.len(),
        "restore position mismatch"
    );
    let actual = forward.prefill(&mut batch, &continuation, cancel)?;
    compare(
        "restored continuation prefill",
        &actual,
        expected_continuation.last().unwrap(),
        &mut failures,
    )?;
    ensure!(
        batch.position() == serial.position() && batch.position() == 68,
        "continuation position mismatch"
    );
    qwen_cli::shutdown::checkpoint()?;
    ensure!(
        failures.is_empty(),
        "numerical comparison failures:\n{}",
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn tight_execution_report_bounds_remain_unchanged() {
    let mut report = json!({"mode":"NativeF32", "positions":7,"layers":40,
        "tensors_compared":567,"token_check":"complete","token_positions_verified":7,
        "shapes":{"hidden":[4,5120],"pre":[4],"logits":[129280]},
        "logits": (0..7).map(|position| json!({"position":position,"reference_top1":5,
            "runtime_top1":5,"top1_match":true,"maxabs":5.49e-5,
            "kl_reference_to_runtime":4.82e-11})).collect::<Vec<_>>()});
    check_legacy_tight_report(&report).unwrap();
    report["first_divergence"] = json!({"position":0,"layer":15,"tensor":"hidden",
        "maxabs":0.00390625,"normrelative":2.9849533440885867e-7});
    report["worst_maxabs"] = json!({"position":1,"layer":36,"tensor":"hidden","maxabs":0.3125});
    check_legacy_tight_report(&report).unwrap();
    for (key, value) in [
        ("tensors_compared", json!(566)),
        ("layers", json!(39)),
        ("positions", json!(6)),
        ("token_check", json!("partial")),
        (
            "shapes",
            json!({"hidden":[4,5121],"pre":[4],"logits":[129280]}),
        ),
    ] {
        let mut bad = report.clone();
        bad[key] = value;
        assert!(check_legacy_tight_report(&bad).is_err(), "accepted {key}");
    }
    for (key, value) in [
        ("maxabs", json!(1.01e-4)),
        ("kl_reference_to_runtime", json!(1.01e-10)),
        ("top1_match", json!(false)),
        ("runtime_top1", json!(6)),
        ("maxabs", Value::Null),
    ] {
        let mut bad = report.clone();
        bad["logits"][0][key] = value;
        assert!(check_legacy_tight_report(&bad).is_err(), "accepted {key}");
    }
    report["token_positions_verified"] = json!(6);
    assert!(check_legacy_tight_report(&report).is_err());
}

#[test]
fn named_gguf_fixture_is_resolved_without_runtime_source_directory() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "qwen-v41-gguf-fixture-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    let result = (|| -> Result<()> {
        std::fs::write(root.join(GGUF_MODEL.filename()), b"fixture lookup only")?;
        let child = Command::new(std::env::current_exe()?)
            .args([
                "named_gguf_fixture_lookup_child",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env("QWEN_MODEL_SEARCH_PATH", &root)
            .output()?;
        ensure!(
            child.status.success(),
            "GGUF fixture lookup failed: {}",
            String::from_utf8_lossy(&child.stderr)
        );
        Ok(())
    })();
    std::fs::remove_dir_all(&root)?;
    result
}

#[test]
#[ignore = "subprocess fixture; run by named_gguf_fixture_is_resolved_without_runtime_source_directory"]
fn named_gguf_fixture_lookup_child() -> Result<()> {
    let path = GGUF_MODEL.required();
    ensure!(
        path.extension().is_some_and(|ext| ext == "gguf"),
        "not a GGUF fixture"
    );
    let root = path.parent().unwrap();
    ensure!(
        !root.join("config.json").exists() && !root.join("tokenizer.json").exists(),
        "runtime source directory is not needed"
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
