//! Q8 versus native F16 cache diagnostic. Quality failures are reported, not retuned.
use super::*;
use std::io::{BufWriter, Write};

struct Control {
    file: PathBuf,
    digest: String,
    captures: Vec<f32>,
    trajectory: PathBuf,
    trajectory_digest: String,
    tail: Vec<u32>,
}

fn save(directory: &Path, name: &str, data: &serde_json::Value) {
    fs::write(
        directory.join(name),
        serde_json::to_vec_pretty(data).unwrap(),
    )
    .unwrap();
}

fn cache_digest(session: &K2Session<'_, '_>) -> String {
    let t = &session.buffers.cache;
    assert_eq!(t.offset, 0);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            t.buffer.contents().as_ptr().cast::<u8>(),
            t.n_bytes() as usize,
        )
    };
    format!("{:x}", Sha256::digest(bytes))
}

fn poison_empty(session: &K2Session<'_, '_>) {
    assert_eq!(session.committed_len(), 0);
    let t = &session.buffers.cache;
    unsafe {
        std::slice::from_raw_parts_mut(
            t.buffer.contents().as_ptr().cast::<u8>(),
            t.n_bytes() as usize,
        )
        .fill(255);
    }
}

fn memory_record(model: &K2LoadedModel<'_>, ctx: &MetalContext) -> serde_json::Value {
    let before = ctx.current_allocated_size();
    let session = model.create_session(0).unwrap();
    let cache = &session.buffers.cache;
    let expected = model
        .config()
        .kv_storage_bytes(256, model.plan.session.storage)
        .unwrap();
    assert_eq!(cache.n_bytes(), expected);
    assert_eq!(
        model.plan.session.buffer_bytes().iter().sum::<u64>(),
        expected + 1_240_068
    );
    json!({"cache_storage":format!("{:?}",model.plan.session.storage),"tensor_dtype":format!("{:?}",cache.dtype),
        "actual_logical_cache_bytes":cache.n_bytes(),"cache_buffer_length":cache.buffer.length(),
        "cache_allocated_bytes":cache.buffer.allocatedSize(),
        "planned_session_logical_bytes":model.plan.session.buffer_bytes().iter().sum::<u64>(),
        "observed_session_allocation_delta_bytes":ctx.current_allocated_size().saturating_sub(before)})
}

fn prepare_control(
    model: &K2LoadedModel<'_>,
    directory: &Path,
    name: &str,
    base: u32,
    tokens: &[u32],
    sites: &[u32],
) -> Control {
    assert_eq!(model.plan.session.storage, K2KvStorage::F16);
    assert_eq!(model.attention, AttentionBackend::Online);
    assert_eq!(model.prefill, PrefillMode::Serial);
    let file = directory.join(format!("{name}-{base}-native-f16.rows"));
    let mut writer = BufWriter::new(fs::File::create_new(&file).unwrap());
    online::write_header(&mut writer, base, 256, 250624);
    let mut session = model.create_session(base).unwrap();
    let mut captures = Vec::new();
    for (index, &token) in tokens.iter().enumerate() {
        let logits = if index == 255 {
            let result = session.append_with_captures(&[token], sites).unwrap();
            assert_eq!(result.absolute_position, base + 255);
            assert_eq!(result.post_block_layers, sites);
            captures = result.residuals;
            result.logits
        } else {
            session.append(&[token]).unwrap()
        };
        assert_eq!(logits.len(), 250624);
        online::write_row(&mut writer, base + index as u32, token, &logits);
    }
    writer.flush().unwrap();
    drop(writer);
    drop(session);
    assert_eq!(captures.len(), sites.len() * 4096);
    fs::write(
        file.with_extension("captures.f32le"),
        captures
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let trajectory = directory.join(format!("{name}-{base}-native-f16-tail.rows"));
    let mut writer = BufWriter::new(fs::File::create_new(&trajectory).unwrap());
    online::write_header(&mut writer, base + 240, 16, 250624);
    let mut session = model.create_session(base).unwrap();
    let mut logits = session.append(&tokens[..241]).unwrap();
    let mut token = tokens[240];
    let mut tail = Vec::new();
    for index in 0..16 {
        tail.push(token);
        online::write_row(&mut writer, base + 240 + index, token, &logits);
        if index < 15 {
            token = runner::argmax(&logits);
            logits = session.append(&[token]).unwrap();
        }
    }
    assert_eq!(session.committed_len(), 256);
    writer.flush().unwrap();
    drop(writer);
    Control {
        digest: file_sha256(&file),
        trajectory_digest: file_sha256(&trajectory),
        file,
        captures,
        trajectory,
        tail,
    }
}

fn record_row(
    p: &serde_json::Value,
    name: &str,
    base: u32,
    visible: usize,
    token: u32,
    actual: &[f32],
    expected: &[f32],
    trajectory: bool,
) -> serde_json::Value {
    let metrics = runner::row_metrics(actual, expected);
    let failed = v2::row_failures(p, &metrics, trajectory);
    let mismatch = metrics["logits"]["actual_top1"] != metrics["logits"]["reference_top1"];
    json!({"corpus":name,"base":base,"visible_length":visible,"token":token,
        "lane":if trajectory {"f16_derived_trajectory"} else {"teacher_forced"},
        "metrics":metrics,"v2_failed_gates":failed,"exact_top1_mismatch":mismatch,
        "exact_predictor_required":trajectory,"accepted_ranking_indeterminate":mismatch && failed.is_empty()})
}

#[test]
#[ignore = "GPU compact-cache quality diagnostic; production lease; about 1.1 GiB native controls; no promotion"]
fn gpu_compact_cache_256_quality_and_packed_controls_diagnostic() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let p = v2::policy();
    let cases = inputs_for(&source, &p)
        .into_iter()
        .filter(|c| c.3)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 4);
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let sites = [0, 11, 23, 35];
    let boundaries = p["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!(
            "k2-compact-diagnostic-{}-{stamp}",
            std::process::id()
        ));
    fs::create_dir(&directory).unwrap();
    save(
        &directory,
        "manifest.json",
        &json!({
            "schema":"k2.compact_cache_diagnostic.v1","claim":"native_q8_vs_native_f16_not_independent_oracle_or_new_holdout",
            "policy":p,"policy_sha256":v2::SHA256,"compact_policy_commit":"4842876c",
            "compact_policy_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../../../../scripts/reference/k2/COMPACT-KV-POLICY.md"))),
            "model_sha256":p["model_sha256"],"tokenizer_metadata_id":p["tokenizer_metadata_id"],
            "inputs":cases,"sites":sites,"boundaries":boundaries,"capacity":256,
            "control":"native_online_F16_KV","candidate":"native_online_Q8_0_KV",
            "trajectory":"F16-derived 241+15 EOS-inclusive mathematical inputs; a Q8 predictor disagreement invalidates free-generation agreement",
            "native_test_binary_sha256":file_sha256(&std::env::current_exe().unwrap()),
            "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
            "host_build_note":std::env::var("K2_HOST_BUILD_NOTE").ok(),"metal_api_validation":"1",
            "quality_failures":"reported_under_unchanged_v2_gates_not_hard_test_failures",
            "hard_failures":"storage_layout_finiteness_sentinels_transactions_within_Q8_bitwise_controls",
            "public_default_changed":false,"performance_claim":false,
        }),
    );
    eprintln!(
        "compact cache diagnostic artifacts: {}",
        directory.display()
    );
    let ctx = MetalContext::new().unwrap();
    let (controls, f16_memory) = {
        let model = K2LoadedModel::load_with_attention_unqualified(
            &ctx,
            &source,
            256,
            AttentionBackend::Online,
        )
        .unwrap();
        let memory = memory_record(&model, &ctx);
        let controls = cases
            .iter()
            .map(|(name, base, tokens, _)| {
                prepare_control(&model, &directory, name, *base, tokens, &sites)
            })
            .collect::<Vec<_>>();
        (controls, memory)
    };
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    save(
        &directory,
        "controls.json",
        &json!(
            controls
                .iter()
                .zip(&cases)
                .map(|(c, (name, base, _, _))| json!({
                    "corpus":name,"base":base,"file":c.file,"sha256":c.digest,
                    "captures_sha256":file_sha256(&c.file.with_extension("captures.f32le")),
                    "tail_file":c.trajectory,"tail_sha256":c.trajectory_digest,"tail_ids":c.tail,
                }))
                .collect::<Vec<_>>()
        ),
    );
    let mut model = K2LoadedModel::load_with_storage_unqualified(
        &ctx,
        &source,
        256,
        AttentionBackend::Online,
        K2KvStorage::Q8_0,
    )
    .unwrap();
    let q8_memory = memory_record(&model, &ctx);
    assert_eq!(f16_memory["actual_logical_cache_bytes"], 147456u64 * 256);
    assert_eq!(q8_memory["actual_logical_cache_bytes"], 78336u64 * 256);
    assert!(
        q8_memory["cache_allocated_bytes"].as_u64().unwrap()
            < f16_memory["cache_allocated_bytes"].as_u64().unwrap()
    );
    save(
        &directory,
        "memory.json",
        &json!({"F16":f16_memory,"Q8_0":q8_memory,"nonoverlapping_model_lifetimes":true,
        "logical_bytes_saved_per_token":69120,"logical_KV_savings_fraction":0.46875}),
    );
    let mut identity = vec![0u8; 4096 * 4096 * 2];
    for i in 0..4096 {
        identity[(i * 4096 + i) * 2..(i * 4096 + i) * 2 + 2]
            .copy_from_slice(&half::f16::ONE.to_le_bytes());
    }
    let identity = K2LinearF16::from_target_source_le(identity).unwrap();
    let mut rows = Vec::new();
    let mut captures = Vec::new();
    let mut partitions = 0;
    for ((name, base, tokens, _), control) in cases.iter().zip(controls) {
        assert_eq!(file_sha256(&control.file), control.digest);
        let mut reference = OracleRows::new(
            BufReader::new(fs::File::open(&control.file).unwrap()),
            *base,
            tokens,
            250624,
        );
        model.prefill = PrefillMode::Serial;
        let mut session = model.create_session(*base).unwrap();
        poison_empty(&session);
        let mut checkpoints = Vec::new();
        let mut final_capture = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            let expected = reference.row();
            let actual = if index == 255 {
                let result = session.append_with_captures(&[token], &sites).unwrap();
                assert_eq!(result.absolute_position, *base + 255);
                assert_eq!(result.post_block_layers, sites);
                final_capture = result.residuals;
                for ((&layer, a), b) in sites
                    .iter()
                    .zip(final_capture.chunks_exact(4096))
                    .zip(control.captures.chunks_exact(4096))
                {
                    let mut metrics = online::capture_metrics(&p, a, b);
                    metrics["corpus"] = json!(name);
                    metrics["base"] = json!(base);
                    metrics["layer"] = json!(layer);
                    captures.push(metrics);
                }
                result.logits
            } else {
                session.append(&[token]).unwrap()
            };
            rows.push(record_row(
                &p,
                name,
                *base,
                index + 1,
                token,
                &actual,
                &expected,
                false,
            ));
            if boundaries.contains(&(index + 1)) {
                checkpoints.push((actual, cache_digest(&session)));
            }
        }
        reference.finish();
        let cache = cache_digest(&session);
        let identity_result = session
            .readout_linear_f16(&identity, &final_capture[3 * 4096..])
            .unwrap();
        assert!(runner::bitwise_equal(
            &identity_result.residual,
            &final_capture[3 * 4096..]
        ));
        assert!(runner::bitwise_equal(
            &identity_result.logits,
            &checkpoints.last().unwrap().0
        ));
        assert_eq!(cache_digest(&session), cache);
        assert!(session.append(&[42]).is_err());
        assert_eq!(session.committed_len(), 256);
        assert!(!session.is_poisoned());
        drop(session);
        save(
            &directory,
            "metrics.json",
            &json!({"rows":rows,"captures":captures}),
        );
        model.prefill = PrefillMode::BatchQ8;
        let mut split = model.create_session(*base).unwrap();
        poison_empty(&split);
        let mut start = 0;
        for (&end, (expected, cache)) in boundaries.iter().zip(&checkpoints) {
            assert!(
                runner::bitwise_equal(&split.append(&tokens[start..end]).unwrap(), expected),
                "Q8 split mismatch {name} {end}"
            );
            assert_eq!(cache_digest(&split), *cache, "Q8 split cache mismatch");
            start = end;
        }
        drop(split);
        let mut whole = model.create_session(*base).unwrap();
        poison_empty(&whole);
        let actual = whole.append_with_captures(tokens, &sites).unwrap();
        assert!(runner::bitwise_equal(
            &actual.logits,
            &checkpoints.last().unwrap().0
        ));
        assert!(runner::bitwise_equal(&actual.residuals, &final_capture));
        assert_eq!(cache_digest(&whole), checkpoints.last().unwrap().1);
        drop(whole);
        partitions += 1;
        model.prefill = PrefillMode::Serial;
        assert_eq!(file_sha256(&control.trajectory), control.trajectory_digest);
        assert_eq!(control.tail[0], tokens[240]);
        let mut reference = OracleRows::new(
            BufReader::new(fs::File::open(&control.trajectory).unwrap()),
            base + 240,
            &control.tail,
            250624,
        );
        let mut session = model.create_session(*base).unwrap();
        let mut actual = session.append(&tokens[..241]).unwrap();
        let mut previous = None;
        for (index, &token) in control.tail.iter().enumerate() {
            let expected = reference.row();
            if index > 0 {
                assert_eq!(previous, Some(token));
                actual = session.append(&[token]).unwrap();
            }
            previous = Some(runner::argmax(&expected));
            rows.push(record_row(
                &p,
                name,
                *base,
                index + 241,
                token,
                &actual,
                &expected,
                true,
            ));
        }
        reference.finish();
        assert_eq!(session.committed_len(), 256);
        save(
            &directory,
            "metrics.json",
            &json!({"rows":rows,"captures":captures}),
        );
        eprintln!(
            "{name}: compact storage/packed/identity controls pass; diagnostic metrics recorded"
        );
    }
    assert_eq!(rows.len(), 1088);
    assert_eq!(captures.len(), 16);
    assert_eq!(partitions, 4);
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let failed_rows = rows
        .iter()
        .filter(|r| !r["v2_failed_gates"].as_array().unwrap().is_empty())
        .count();
    let failed_captures = captures
        .iter()
        .filter(|r| !r["failed_gates"].as_array().unwrap().is_empty())
        .count();
    save(
        &directory,
        "summary.json",
        &json!({
            "invariants_passed":true,"v2_quality_envelope_passed":failed_rows+failed_captures==0,
            "rows":rows.len(),"capture_sites":captures.len(),"failed_quality_rows":failed_rows,"failed_quality_capture_sites":failed_captures,
            "top1_mismatches":rows.iter().filter(|r|r["exact_top1_mismatch"]==true).count(),
            "exact_trajectory_predictors":64,"trajectory_predictor_mismatches":rows.iter().filter(|r|r["exact_predictor_required"]==true && r["exact_top1_mismatch"]==true).count(),
            "q8_bitwise_partition_controls":partitions,"q8_identity_transport_controls":4,
            "logical_cache_bytes_per_token":{"F16":147456,"Q8_0":78336},
            "public_default_changed":false,"performance_claim":false,
        }),
    );
}
