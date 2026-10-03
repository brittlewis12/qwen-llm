//! Opt-in original-forward oracle; synthetic matrices are not fitted-quality evidence.

use super::*;
use crate::serve::http::GenerationBackend;
use anyhow::{Context, Result, ensure};
use qwen_llm::{
    gguf::GgufFile,
    runtime::{LoadedModelConfig, Runtime},
};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

pub(super) fn synthetic_assets(root: &Path, gguf: &GgufFile) -> Result<(u32, u32)> {
    let arch = qwen_llm::loader::Model::from_gguf(gguf)?.arch;
    ensure!(
        arch.n_layer > 2 && arch.hidden_size > 2,
        "oracle geometry too small"
    );
    let h = arch.hidden_size as usize;
    let bytes = h
        .checked_mul(h)
        .and_then(|n| n.checked_mul(2))
        .context("fixture overflow")?;
    ensure!(
        bytes as u64 <= registry::MAX_STAGED_BYTES / 2,
        "fixture exceeds staging bound"
    );
    let middle = (arch.n_layer - 1) / 2;
    let last = arch.n_layer - 1;
    let mut rotated = vec![0u8; bytes];
    let mut identity = vec![0u8; bytes];
    for row in 0..h {
        let value = [0.5f32, 1., 2.][row % 3];
        let offset = (row * h + (row + 1) % h) * 2;
        rotated[offset..offset + 2]
            .copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        let offset = (row * h + row) * 2;
        identity[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let digest = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    let hashes = [digest(&rotated), digest(&identity)];
    rotated.extend_from_slice(&identity);
    let manifest = json!({"schema":"llm.lens.linear_transport","schema_version":1,"status":"complete",
        "transport":{"operator":"post_block_linear","method":"synthetic_rotated_diagonal_cpu_oracle","source_layers":[middle,last],
            "target_layer":last,"orientation":"target_source","bias":"none","output":"deployed_native","identity_layers":[last]},
        "model":{"architecture":gguf.get_str("general.architecture"),"n_layers":arch.n_layer,"hidden_size":h,"vocab_size":arch.vocab_size},
        "payload":{"path":"transport.f16le","dtype":"f16_le","shape":[2,h,h],"byte_length":rotated.len(),"sha256":digest(&rotated),"matrix_sha256":hashes},
        "provenance":{"purpose":"test_only_not_fit_quality"},"qualification":{}});
    std::fs::write(root.join("lens.json"), serde_json::to_vec(&manifest)?)?;
    std::fs::write(root.join("transport.f16le"), rotated)?;
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({"schema_version":1,"assets":[
        {"alias":"oracle","path":".","allow_unvalidated_transfer":true}]}))?,
    )?;
    Ok((middle, last))
}

#[test]
#[ignore = "Metal: run scripts/serve/lens_fitted_check.ts with QWEN_LENS_TEST_MODEL"]
fn fitted_original_forward_cpu_oracle() -> Result<()> {
    crate::shutdown::install()?;
    let model = std::path::PathBuf::from(std::env::var("QWEN_LENS_TEST_MODEL")?);
    let output = std::path::PathBuf::from(std::env::var("QWEN_LENS_TEST_OUTPUT")?);
    ensure!(!output.exists(), "oracle output must be fresh");
    std::fs::create_dir(&output)?;
    let gguf = GgufFile::open(&model)?;
    let (middle, last) = synthetic_assets(&output, &gguf)?;
    let protocol = crate::prompt_template::identify_qwen_release_for_gguf(&gguf)?
        .template
        .serve_template();
    ensure!(
        matches!(
            protocol,
            crate::serve::items::QwenTemplate::Qwen36 | crate::serve::items::QwenTemplate::Qwen38
        ),
        "qualified native protocol required"
    );
    let registry = Arc::new(registry::Registry::open(
        &output.join("config.json"),
        &gguf,
        &mut crate::shutdown::checkpoint,
    )?);
    let store = Arc::new(JobStore::open(
        &output.join("jobs"),
        crate::serve::jobs::store::Limits::default(),
    )?);
    let runtime = Runtime::metal()?;
    let started = Instant::now();
    let loaded = runtime.load_opened_gguf_with_config(
        gguf,
        model,
        LoadedModelConfig {
            prefix_cache_max_bytes: 0,
            ..Default::default()
        },
    )?;
    let mut backend = crate::serve::backend::EngineBackend::new(
        loaded,
        "fitted-oracle".into(),
        3,
        Some(256),
        256,
        None,
        protocol,
        true,
    )?;
    backend.attach_lens_registry(Some(registry))?;
    let profile = backend.native_profile()?.context("native profile")?;
    let mut value = json!({"schema_version":1,"idempotency_key":"baseline",
        "input":{"kind":"messages","messages":[{"role":"user","content":"Name an animal."}],"generation_mode":"thinking","assistant_prefill":{"channel":"reasoning","text":"Let me"}},
        "generation":{"max_new_tokens":3,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0,"seed":7}}});
    let mut run = |value: &Value| -> Result<String> {
        let key = value["idempotency_key"].as_str().context("key")?;
        let prepared = profile
            .prepare(&Request::parse(value)?)
            .map_err(|e| anyhow::anyhow!("{}", e.error.message))?;
        let id = store
            .accept(key, value, value.get("diagnostics").is_some())?
            .status
            .id;
        let writer = writer::Writer::spawn(
            store.clone(),
            id.clone(),
            store.control(&id)?,
            &prepared,
            Default::default(),
        )?;
        ensure!(
            matches!(writer.wait_ready()?, writer::Readiness::Execute),
            "unexpected settled job"
        );
        let outcome = backend.generate_native(&prepared, writer.sink());
        writer.finish(outcome)?;
        Ok(id)
    };
    let baseline = run(&value)?;
    let readout = |id, lens, layers| {
        json!({"id":id,"lens":lens,"mode":"full_vocabulary","top_k":5,
        "scope":{"layers":{"kind":"values","values":layers},"prefill":{"kind":"values","values":[0]},"decode":{"kind":"values","values":[0]}}})
    };
    value["idempotency_key"] = json!("observed");
    value["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[
        readout("fit","oracle",vec![middle,last]),readout("identity","plain",vec![last])]});
    let observed = run(&value)?;
    drop(backend);
    drop(runtime);
    let residency_ms = started.elapsed().as_millis();
    println!("oracle GPU resources released after {residency_ms}ms");
    for id in [&baseline, &observed] {
        let status = store.status(id)?;
        ensure!(
            status.state == crate::serve::jobs::state::JobState::Completed
                && status.result.complete
                && status.result.error.is_none(),
            "oracle failed: {status:?}"
        );
        ensure!(
            status.generation.counters.consumed_generated_tokens > 0,
            "require decode consumption"
        );
    }
    let baseline = store.result(&baseline, None, 128)?.records;
    let records = store.result(&observed, None, 128)?.records;
    let samples = |records: &[Value]| {
        records
            .iter()
            .filter(|r| r["kind"] == "sampled_token")
            .cloned()
            .map(|mut r| {
                r.as_object_mut().unwrap().remove("seq");
                r
            })
            .collect::<Vec<_>>()
    };
    ensure!(
        samples(&baseline) == samples(&records),
        "fitted observation changed sampling"
    );
    let rows = records
        .iter()
        .filter(|r| r["kind"] == "readout")
        .collect::<Vec<_>>();
    ensure!(rows.len() == 6, "require six distinct rows");
    let mut expected = std::collections::BTreeSet::new();
    for phase in ["prefill", "decode"] {
        for (id, layer) in [("fit", middle), ("fit", last), ("identity", last)] {
            expected.insert((id, phase, layer));
        }
    }
    let prompt = records[0]["token_ids"].as_array().context("tokens")?;
    let mut witnesses = 0;
    let mut generation_witnesses = 0;
    for row in &rows {
        let phase = row["phase"].as_str().context("phase")?;
        let layer = u32::try_from(row["source_layer"].as_u64().context("layer")?)?;
        let id = row["readout_id"].as_str().context("readout id")?;
        ensure!(
            expected.remove(&(id, phase, layer)),
            "unexpected/duplicate row"
        );
        let position = if phase == "prefill" { 0 } else { prompt.len() };
        ensure!(
            row["index"] == 0
                && row["position"] == position
                && row["predicts_position"] == position + 1,
            "oracle site mismatch"
        );
        ensure!(
            row["scores"].as_array().context("scores")?.len() == 5,
            "top-k mismatch"
        );
        if id == "fit" {
            let witness = &row["test_transport_witness"];
            ensure!(
                witness["basis"] == "cpu_f64_row_major_f16_matrix_times_original_forward_residual"
                    && witness["within_tolerance"] == true
                    && witness["hidden_size"] == profile.hidden,
                "missing independent transport witness"
            );
            ensure!(
                row["generation_logit_witness"].is_null(),
                "fitted generation witness forbidden"
            );
            witnesses += 1;
            if layer == last {
                let plain = rows
                    .iter()
                    .find(|r| r["lens"] == "plain" && r["position"] == position)
                    .context("identity control")?;
                for (a, b) in row["scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(plain["scores"].as_array().unwrap())
                {
                    ensure!(a["token_id"] == b["token_id"], "identity ranking mismatch");
                    ensure!(
                        (a["score"].as_f64().unwrap() - b["score"].as_f64().unwrap()).abs() <= 1e-4,
                        "identity logits mismatch"
                    );
                }
            }
        } else if phase == "decode" {
            let witness = &row["generation_logit_witness"];
            ensure!(
                witness["basis"] == "same_original_forward_generation_logits"
                    && witness["vocabulary_size"] == profile.tokenizer.n_vocab()
                    && witness["within_tolerance"] == true
                    && witness["max_abs_error"]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v >= 0.),
                "missing/failing original-generation witness"
            );
            generation_witnesses += 1;
        } else {
            ensure!(
                row["generation_logit_witness"].is_null(),
                "no-tail prompt cannot have a generation witness"
            );
        }
    }
    ensure!(
        expected.is_empty() && witnesses == 4 && generation_witnesses == 1,
        "oracle incomplete"
    );
    std::fs::write(
        output.join("witnesses.json"),
        serde_json::to_vec_pretty(
            &json!({"witness_count":witnesses,"identity_controls":2,"passing_generation_witnesses":generation_witnesses,
        "unchanged_sampling":true,"nonidentity_middle_layer":middle,"identity_last_layer":last,"job_id":observed,
        "residency_ms":residency_ms,"readouts":rows}),
        )?,
    )?;
    Ok(())
}
