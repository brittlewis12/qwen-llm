use super::*;

fn check_reference(reference: &Value, tokens: &Value) -> Result<()> {
    ensure!(
        reference["weight_backend"] == "independent_gguf",
        "CPU reference must decode GGUF independently, not HF weights"
    );
    ensure!(
        reference["mode"] == "NativeF32" && reference["router_tie_policy"] == POLICY,
        "CPU arithmetic/router configuration differs"
    );
    ensure!(
        reference["max_seq_len"] == 7 && reference["projection_rows"] == 128,
        "CPU sequence/projection configuration differs from retained witness"
    );
    ensure!(
        reference["source_sha256"]
            == "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
            && reference["engram_source_sha256"]
                == "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
        "CPU inference source differs from pinned reference"
    );
    ensure!(
        reference["gguf"].as_str().is_some_and(|p| !p.is_empty()),
        "CPU GGUF path missing"
    );
    ensure!(
        tokens["prompt"] == json!(TOKENS),
        "CPU consumed token sequence differs"
    );
    let generated = tokens["generated"]
        .as_array()
        .context("CPU generated tokens missing")?;
    ensure!(
        generated.len() == 1 && generated[0].as_u64().is_some_and(|t| t < 129280),
        "CPU generated token metadata invalid"
    );
    Ok(())
}

pub(super) fn run() -> Result<()> {
    let marker = GGUF_CPU_REFERENCE.required();
    let reference = marker.parent().context("reference has no parent")?;
    let meta = manifest(&reference.join("run.json"))?;
    ensure!(
        manifest(marker)? == meta,
        "named GGUF CPU marker differs from run.json"
    );
    let tokens = manifest(&reference.join("tokens.json"))?;
    check_reference(&meta, &tokens)?;
    // Fixture names select the retained pair; the recorded source path is
    // provenance, not a restriction on where fixtures may be installed.
    let model = GGUF_MODEL.required();
    let output = std::env::temp_dir().join(format!(
        "qwen-v41-gguf-execution-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&output)?;
    eprintln!(
        "GGUF execution-only acceptance retained at {}",
        output.display()
    );
    let trace = output.join("trace");
    replay_trace(model, &trace, true)?;
    let runtime = manifest(&trace.join("run.json"))?;
    check_policy(&meta, &runtime)?;
    ensure!(
        runtime["execution"] == "gpu_step"
            && runtime["comparison_scope"] == "independent_gguf_execution_only"
            && runtime["context"] == 7,
        "runtime execution configuration differs"
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
    check_legacy_tight_report(&report)?;
    ensure!(
        tokens["generated"] == json!([report["logits"][6]["reference_top1"]])
            && runtime["generated_token_ids"] == json!([report["logits"][6]["runtime_top1"]]),
        "sampled tokens differ from compared argmax"
    );
    std::fs::write(
        output.join("acceptance.json"),
        serde_json::to_vec_pretty(&json!({
            "kind":"acceptance", "scope":"independent_gguf_execution_only",
            "logit_maxabs_bound":1e-4, "kl_reference_to_runtime_bound":1e-10,
            "required_top1_matches":7, "hidden_state_acceptance":"not_evaluated",
            "metrics":report
        }))?,
    )?;
    Ok(())
}

#[test]
fn independent_gguf_reference_negative_controls() {
    let good = json!({"weight_backend":"independent_gguf", "mode":"NativeF32",
        "router_tie_policy":POLICY, "max_seq_len":7, "projection_rows":128,
        "source_sha256":"4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
        "engram_source_sha256":"11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
        "gguf":"fixture.gguf"});
    let tokens = json!({"prompt":TOKENS,"generated":[1266]});
    check_reference(&good, &tokens).unwrap();
    for (key, value) in [
        ("weight_backend", json!("hf")),
        ("weight_backend", Value::Null),
        ("mode", json!("ReferenceFp8")),
        ("router_tie_policy", json!("upstream_torch_topk")),
        ("max_seq_len", json!(8)),
        ("projection_rows", json!(64)),
        ("source_sha256", json!("different")),
        ("engram_source_sha256", Value::Null),
        ("gguf", json!("")),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        assert!(check_reference(&bad, &tokens).is_err(), "accepted {key}");
    }
    for (key, value) in [
        ("prompt", json!([0])),
        ("prompt", json!([0, 5, 223, 939, 23, 695, 23])),
        ("generated", json!([])),
        ("generated", json!([129280])),
        ("generated", json!([1, 2])),
    ] {
        let mut bad = tokens.clone();
        bad[key] = value;
        assert!(check_reference(&good, &bad).is_err(), "accepted {key}");
    }
}
