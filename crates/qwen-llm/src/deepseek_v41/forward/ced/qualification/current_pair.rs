//! Private qualification, not a throughput protocol or an original-runtime oracle.
//! Uses the loader's storage plan and combined host policy before weight import.
use super::*;
use std::io::{Seek, SeekFrom};

const SCHEMA: &str = "v41-ced-current-pair-v1";
const SOURCE_PATH: &str =
    "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/current_pair.rs";
const INPUT_LIMIT: u64 = 16 * 1024 * 1024;

#[test]
#[ignore = "PRIVATE: central preload/host-budget repair and bounded pilot controller required before any full-model run"]
fn current_full_vs_ced_paired_full_model() {
    run_pair().unwrap();
}

fn prompt_n(text: &str) -> AResult<usize> {
    match text {
        "128" => Ok(128),
        "2048" => Ok(2048),
        "8192" => Ok(8192),
        _ => anyhow::bail!("V41_CED_PAIR_N must be explicitly 128, 2048, or 8192"),
    }
}

fn select_prefix(source: &Input, n: usize, vocab: usize) -> AResult<Input> {
    prompt_n(&n.to_string())?;
    ensure!(source.prompt.len() >= n, "frozen prompt shorter than N");
    ensure!(
        source.continuation.len() == STEPS,
        "frozen continuation must have 256 IDs"
    );
    ensure!(
        source
            .prompt
            .iter()
            .chain(&source.continuation)
            .all(|&t| (t as usize) < vocab),
        "frozen token outside vocabulary"
    );
    let continuation = source.prompt[n..]
        .iter()
        .chain(&source.continuation)
        .take(STEPS)
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        continuation.len() == STEPS,
        "not enough true continuation IDs"
    );
    Ok(Input {
        prompt: source.prompt[..n].to_vec(),
        continuation,
    })
}

fn product(values: &[u64]) -> AResult<u64> {
    values.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b).context("pair budget overflow")
    })
}

#[derive(Debug, serde::Serialize)]
struct CaptureCpuPlan {
    reference_logits_bytes: u64,
    observation_and_report_reserve_bytes: u64,
    input_parse_reserve_bytes: u64,
    total_bytes: u64,
}

fn capture_cpu_plan(vocab: usize) -> AResult<CaptureCpuPlan> {
    ensure!(vocab > 0, "empty vocabulary");
    let reference_logits_bytes = product(&[(STEPS + 1) as u64, vocab as u64, 4])?;
    let observation_and_report_reserve_bytes = 128 * 1024 * 1024;
    let input_parse_reserve_bytes = 64 * 1024 * 1024;
    Ok(CaptureCpuPlan {
        reference_logits_bytes,
        observation_and_report_reserve_bytes,
        input_parse_reserve_bytes,
        total_bytes: add(&[
            reference_logits_bytes,
            observation_and_report_reserve_bytes,
            input_parse_reserve_bytes,
        ])?,
    })
}

#[derive(Debug, serde::Serialize)]
struct RuntimeUpper {
    request_cpu_bytes: u64,
    request_metal_bytes: u64,
    chunk_metal_bytes: u64,
    half_chunk_metal_bytes: u64,
    retained_and_seed_metal_bytes: u64,
    binding_upload_hash_cpu_bytes: u64,
    cpu_bytes: u64,
    metal_bytes: u64,
}

#[derive(Debug, serde::Serialize)]
struct ResidentPreloadUpper {
    storage_view_metal_bytes: u64,
    private_shadow_cpu_bytes: u64,
    bookkeeping_cpu_bytes: u64,
    planner_evidence: Value,
}

fn preload_report(cold: &ColdPrefillPreflight) -> AResult<(RuntimeUpper, ResidentPreloadUpper)> {
    let storage = &cold.resident_storage;
    let half_chunk_metal_bytes = cold
        .ced
        .as_ref()
        .context("missing CED preload ledger")?
        .priced_half_chunk_bytes;
    let runtime = RuntimeUpper {
        request_cpu_bytes: cold.request.cpu_bytes,
        request_metal_bytes: cold.request.metal_bytes,
        chunk_metal_bytes: cold
            .full_incremental_metal_bytes
            .checked_sub(half_chunk_metal_bytes)
            .context("Full half ledger exceeds chunk ledger")?,
        half_chunk_metal_bytes,
        retained_and_seed_metal_bytes: cold.ced.as_ref().map_or(0, |ced| {
            ced.incremental_metal_bytes - ced.priced_chunk_bytes
        }),
        binding_upload_hash_cpu_bytes: cold.runtime_cpu_bytes - cold.request.cpu_bytes,
        cpu_bytes: cold.runtime_cpu_bytes,
        metal_bytes: cold.runtime_metal_bytes,
    };
    let windows: Vec<_> = storage
        .windows
        .iter()
        .map(|planned| {
            json!({"shard_idx": planned.window.shard_idx, "mmap_offset": planned.window.mmap_offset,
            "length": planned.window.length, "mapping_kind": format!("{:?}", planned.mapping_kind),
            "private_shadow_bytes": planned.mapping_kind.private_bytes()})
        })
        .collect();
    let resident = ResidentPreloadUpper {
        storage_view_metal_bytes: storage.metal_view_bytes,
        private_shadow_cpu_bytes: storage.private_shadow_bytes,
        bookkeeping_cpu_bytes: cold.binding_cpu_bytes,
        planner_evidence: json!({"page_size": storage.page_size,
            "max_buffer_length": storage.max_buffer_length, "windows": windows,
            "shared_eof_window_count": storage.shared_eof_window_count,
            "private_window_count": storage.private_window_count,
            "disk_engram_bytes": storage.bytes.disk_engram,
            "source_mapping_charge": "shared views overlap; only full private shadow exposure is additional"}),
    };
    Ok((runtime, resident))
}

fn admission(ctx: &MetalContext, metal_bytes: u64, cpu_bytes: u64) -> AResult<Value> {
    let s = ctx.memory_signals();
    let a =
        metal::evaluate_metal_memory_admission_with_cpu_bytes(metal_bytes, cpu_bytes, 0, s, true);
    if let Some(reason) = a.refusal() {
        return Err(reason.into());
    }
    Ok(admission_report(&a, metal_bytes, cpu_bytes))
}

fn admission_report(a: &metal::MetalMemoryAdmission, metal_bytes: u64, cpu_bytes: u64) -> Value {
    let s = a.signals;
    json!({"metal_bytes": metal_bytes, "cpu_bytes": cpu_bytes, "reason": a.reason.as_str(),
        "recommended_max_bytes": s.recommended_max_bytes, "current_allocated_bytes": s.current_allocated_bytes,
        "process_limit_remaining_bytes": s.process_limit_remaining_bytes,
        "host_budget": format!("{:?}", s.host_budget), "host_remaining_bytes": s.host_remaining_bytes(),
        "raw_zero_process_limit_uses_validated_host_budget": true})
}

fn geometry(gguf: &crate::gguf::GgufFile, p: &v41_gguf::Plan, n: usize) -> AResult<Geometry> {
    ensure!(
        gguf.architecture().as_deref() == Some("deepseek41")
            && format!("{:?}", p.config) == format!("{:?}", Config::default())
            && p.layers == p.schedule.layers
            && p.schedule == v41_gguf::Schedule::default()
            && p.engram_hash == engram::HashConfig::default()
            && p.raw_pad == 2,
        "metadata-validated canonical full V4.1 geometry/schedule/hash/pad required"
    );
    p.schedule.validate()?;
    p.engram_hash
        .validate()
        .map_err(|e| anyhow::anyhow!("hash profile: {e:?}"))?;
    ensure!(
        n >= p.config.window,
        "physical rings must be entirely causal"
    );
    Ok(Geometry {
        n,
        layers: p.schedule.layers,
        cut: p.schedule.candidate_source_layer.context("missing cut")?,
        window: p.config.window,
        hidden: p.config.hidden,
        head_dim: p.config.head_dim,
        index_dim: p.config.index_dim,
        vocab: p.config.vocab,
        topk: p.config.index_topk,
        block: p.config.candidate_block,
        owners: p
            .schedule
            .kv_source_layers
            .iter()
            .map(|&l| (l, p.schedule.ratio(l)))
            .collect(),
        producers: p.schedule.index_source_layers.clone(),
    })
}

fn checked_compiled_source(root: &Path, path: &str, compiled: &[u8]) -> AResult<Value> {
    let hash = archive::sha(compiled);
    ensure!(
        archive::file_sha(&mut archive::open(&root.join(path))?)? == hash,
        "current-pair binary/source mismatch: {path}; rebuild after editing"
    );
    Ok(json!({"path": path, "sha256": hash}))
}

fn source_identity() -> AResult<(PathBuf, Value)> {
    let (root, mut source) = provenance::source()?;
    let sources: &[(&str, &[u8])] = &[
        (SOURCE_PATH, include_bytes!("current_pair.rs")),
        (
            "crates/qwen-llm/src/deepseek_v41/forward.rs",
            include_bytes!("../../../forward.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/preflight.rs",
            include_bytes!("../../preflight.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/memory.rs",
            include_bytes!("../../memory.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/prefill.rs",
            include_bytes!("../../prefill.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs",
            include_bytes!("../../gpu_step.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced.rs",
            include_bytes!("../../gpu_step/ced.rs"),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced.rs",
            include_bytes!("../../ced.rs"),
        ),
        (
            "crates/qwen-llm/src/metal/host_budget.rs",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/metal/host_budget.rs"
            )),
        ),
        (
            "crates/qwen-llm/src/metal/context.rs",
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/context.rs")),
        ),
        (
            "crates/qwen-llm/src/metal/tensor.rs",
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/tensor.rs")),
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/gguf/resident_storage.rs",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/deepseek_v41/gguf/resident_storage.rs"
            )),
        ),
    ];
    source["current_pair_compiled_sources"] = json!(
        sources
            .iter()
            .map(|&(path, bytes)| { checked_compiled_source(&root, path, bytes) })
            .collect::<AResult<Vec<_>>>()?
    );
    Ok((root, source))
}

fn artifact_identity(gguf: &crate::gguf::GgufFile, p: &v41_gguf::Plan) -> AResult<Value> {
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    let shards = stamps
        .iter()
        .map(|s| {
            json!({"shard_idx": s.shard_idx, "supplied_path": s.path,
        "device": s.device, "inode": s.inode, "size": s.size, "mtime_sec": s.mtime_sec,
        "mtime_nsec": s.mtime_nsec, "ctime_sec": s.ctime_sec, "ctime_nsec": s.ctime_nsec})
        })
        .collect::<Vec<_>>();
    let layout = gguf.tensors.iter().map(|t| json!({"name": t.name, "dtype": format!("{:?}", t.dtype),
        "shape": t.shape, "shard_idx": t.shard_idx, "data_offset": t.data_offset, "n_bytes": t.n_bytes})).collect::<Vec<_>>();
    Ok(
        json!({"shards": shards, "tensor_layout_sha256": archive::json_sha(&json!(layout))?,
        "metadata_sha256": archive::json_sha(&serde_json::to_value(gguf.model.metadata())?)?,
        "config_debug": format!("{:?}", p.config), "schedule_debug": format!("{:?}", p.schedule),
        "hash_profile_debug": format!("{:?}", p.engram_hash), "dtype_inventory_bytes": p.dtype_inventory,
        "token_map_sha256_u32_le": archive::sha(bytemuck::cast_slice(&p.token_map)), "raw_pad": p.raw_pad,
        "identity_method": "same loaded ResidentWeights; retained file stamps and metadata/layout hashes, NOT fresh shard-content hashes"}),
    )
}

fn observe_state(
    model: &Forward<'_>,
    state: &State,
    g: &Geometry,
    position: usize,
    origin: PrefillExecution,
) -> AResult<Value> {
    ensure!(
        state.is_ready() && state.position() == position && state.prefill_origin() == origin,
        "state readiness/position/origin at {position}"
    );
    // Reuse snapshot validation and the same causal views/keys/hash as the frozen
    // observer. Never capture_snapshot: that would clone the entire payload.
    let payload_bytes = model.snapshot_payload_bytes(state)?;
    let attention = state
        .gpu_step
        .as_ref()
        .context("missing GPU state")?
        .attention();
    let regions = attention
        .causal_regions(position)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        regions.len() == 1 + 4 * g.owners.len() && model.layers.len() == g.layers,
        "state inventory"
    );
    let raw = causal_slice(regions[0])?;
    let width = g.window * g.head_dim;
    ensure!(
        position >= g.window && raw.len() == g.layers * width,
        "causal physical ring extent"
    );
    let mut hashes = Vec::new();
    for (i, (role, layer)) in g.state_keys().into_iter().enumerate() {
        let values = if i < g.layers {
            &raw[layer * width..(layer + 1) * width]
        } else {
            causal_slice(regions[1 + i - g.layers])?
        };
        let shape = if i < g.layers {
            vec![g.window, g.head_dim]
        } else {
            let ratio = g.owners[(i - g.layers) / 4].1;
            match (i - g.layers) % 4 {
                0 => vec![position / ratio, g.head_dim],
                1 => vec![position / ratio, g.index_dim],
                _ => vec![g.head_dim],
            }
        };
        ensure!(
            shape.iter().product::<usize>() == values.len() && values.iter().all(|v| v.is_finite()),
            "causal shape/nonfinite {position}/{role}/{layer}"
        );
        hashes.push(json!({"role": role, "layer": layer, "shape": shape,
            "sha256": archive::sha(bytemuck::cast_slice(values))}));
    }
    let address = json!({"position": position, "compressed_pad": state.addresses.pad(),
        "compressed_history_oldest_first": state.addresses.history().iter().map(|t| match t {
            engram::Token::Compressed(id) => json!(id), engram::Token::Dead => Value::Null }).collect::<Vec<_>>()});
    // Origin is checked per arm, not made equal between different executions.
    Ok(
        json!({"position": position, "ready": true, "address_sha256": archive::json_sha(&address)?,
        "address": address, "physical_ring_next_slot": position % g.window,
        "pending_ratio2_active": position % 2 == 1, "causal_payload_bytes": payload_bytes,
        "attention_cursor": {"next_position": position, "next_layer": 0, "batch": null,
            "validated_by": "snapshot_payload_bytes/validate_snapshot_position"}, "regions": hashes}),
    )
}

fn validate_execution(model: &Forward<'_>, r: &PrefillReport, p: &CedPlan) -> AResult<Value> {
    ensure!(
        r.execution == PrefillExecution::ShrinkingCed && r.fallback.is_none(),
        "CED fallback: {r:?}"
    );
    let counters = r.counters.as_ref().context("missing CED counters")?;
    counters.validate(p).map_err(anyhow::Error::msg)?;
    let commands = production_command_count(p)?;
    ensure!(
        counters.encoded_commands == commands && counters.completed_commands == commands,
        "production command coverage"
    );
    for l in &counters.layers {
        let rows = if l.layer < p.publication_layer {
            p.prompt_rows
        } else {
            p.decoder[l.layer - p.publication_layer].output_rows.len()
        };
        ensure!(
            l.selection_producer_rows
                == if model.schedule.produces_index(l.layer) {
                    rows
                } else {
                    0
                },
            "index producer coverage"
        );
    }
    let memory = r.memory.as_ref().context("missing accepted CED memory")?;
    let estimated =
        super::super::estimate(model, p.prompt_rows + STEPS, p).map_err(anyhow::Error::msg)?;
    ensure!(
        admission_ledger(memory) == admission_ledger(&estimated)
            && memory.incremental_cpu_bytes > 0
            && memory.incremental_metal_bytes > 0
            && r.observed_incremental_metal_bytes
                .context("missing observed allocation")?
                <= memory.incremental_metal_bytes,
        "CED memory ledger/observation mismatch"
    );
    ensure!(
        memory
            .process_ru_maxrss_before_bytes
            .zip(memory.process_ru_maxrss_after_bytes)
            .is_some_and(|(before, after)| after >= before),
        "missing or nonmonotonic process peak observations"
    );
    Ok(
        json!({"report": r, "publication_completed_rows": counters.publication_rows,
        "publication_validation": "completion-gated runtime CED cursor plus exact plan counters; committed snapshot requires no active CED",
        "dependency_boundary_range": [p.boundary_rows.start, p.boundary_rows.end],
        "output_rows": p.work.total_output_rows, "full_output_rows": p.work.full_forward_output_rows,
        "actually_reduces_output_rows": p.work.total_output_rows < p.work.full_forward_output_rows}),
    )
}

struct ReferenceRow {
    logits: Vec<f32>,
    state: Value,
}

fn full_reference(
    model: &Forward<'_>,
    input: &Input,
    g: &Geometry,
) -> AResult<(Vec<ReferenceRow>, Value, f64)> {
    let mut state = model.new_state(g.n + STEPS)?;
    let started = std::time::Instant::now();
    let (mut logits, execution) = model.prefill_with_execution(
        &mut state,
        &input.prompt,
        None,
        PrefillExecution::FullForward,
    )?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    ensure!(
        execution.execution == PrefillExecution::FullForward
            && execution.fallback.is_none()
            && execution.counters.is_none(),
        "current Full execution"
    );
    let mut reference = Vec::with_capacity(STEPS + 1);
    for i in 0..=STEPS {
        ensure!(
            logits.len() == g.vocab && logits.iter().all(|v| v.is_finite()),
            "Full logits at {i}"
        );
        let observation = observe_state(model, &state, g, g.n + i, PrefillExecution::FullForward)?;
        reference.push(ReferenceRow {
            logits,
            state: observation,
        });
        if i == STEPS {
            break;
        }
        logits = model
            .step(&mut state, input.continuation[i], None)
            .map_err(anyhow::Error::msg)?;
    }
    drop(state);
    Ok((reference, serde_json::to_value(execution)?, prefill_seconds))
}

fn compare_ced(
    model: &Forward<'_>,
    input: &Input,
    g: &Geometry,
    plan: &CedPlan,
    reference: Vec<ReferenceRow>,
) -> AResult<(Vec<Value>, Value, f64)> {
    ensure!(reference.len() == STEPS + 1, "reference position coverage");
    let mut state = model.new_state(g.n + STEPS)?;
    let started = std::time::Instant::now();
    let (mut logits, execution) = model.prefill_with_execution(
        &mut state,
        &input.prompt,
        None,
        PrefillExecution::ShrinkingCed,
    )?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let execution = validate_execution(model, &execution, plan)?;
    let mut observations = Vec::with_capacity(STEPS + 1);
    for (i, expected) in reference.into_iter().enumerate() {
        let pos = g.n + i;
        ensure!(
            logits.len() == g.vocab && logits.iter().all(|v| v.is_finite()),
            "CED logits at {pos}"
        );
        let observed = observe_state(model, &state, g, pos, PrefillExecution::ShrinkingCed)?;
        ensure!(
            expected.state == observed,
            "strict causal state/hash mismatch at {pos}: expected_sha256={} actual_sha256={}",
            archive::json_sha(&expected.state)?,
            archive::json_sha(&observed)?
        );
        let mut difference = Difference::default();
        difference.add(
            bytemuck::cast_slice(&expected.logits),
            bytemuck::cast_slice(&logits),
            true,
            &json!({"position": pos, "role": "full_vocabulary_logits"}),
            true,
        )?;
        observations.push(
            json!({"position": pos, "state": observed, "state_bitwise_hash_equal": true,
            "logits": {"elements": g.vocab, "sha256": archive::sha(bytemuck::cast_slice(&logits)),
                "reference_sha256": archive::sha(bytemuck::cast_slice(&expected.logits)),
                "difference": difference.json(), "bitwise_equal": true},
            "next_teacher_forced_id": input.continuation.get(i), "comparison_only": i == STEPS}),
        );
        // Consume each Full vector and hash record before the next decode step;
        // there is no CED archive and never a second live KV state.
        drop(expected);
        if i == STEPS {
            break;
        }
        logits = model
            .step(&mut state, input.continuation[i], None)
            .map_err(anyhow::Error::msg)?;
    }
    Ok((observations, execution, prefill_seconds))
}

fn write_report(file: &mut File, report: &Value) -> AResult<()> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    serde_json::to_writer(&mut *file, report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn run_pair() -> AResult<()> {
    provenance::environment()?;
    ensure!(
        ACTIVE.with(|slot| slot.borrow().is_none()),
        "legacy frozen comparator already active"
    );
    let n =
        prompt_n(&std::env::var("V41_CED_PAIR_N").context("explicit V41_CED_PAIR_N required")?)?;
    let gguf_path = env_path("V41_CED_GGUF")?;
    let input_path = env_path("V41_CED_TOKENS")?;
    let report_path = env_path("V41_CED_CURRENT_PAIR_REPORT")?;
    let (checkout, source) = source_identity()?;
    ensure!(
        !input_path.canonicalize()?.starts_with(&checkout),
        "private input must be outside checkout"
    );
    let mut report_file = private_report(&report_path, &checkout)?;
    let mut report = json!({"schema": SCHEMA, "complete": false, "current_pair": true,
        "arm_order": ["current_full_production", "shrinking_ced_production"], "candidate_provenance": source,
        "N": n, "capacity": n + STEPS, "continuation_steps": STEPS,
        "strict_contract": "bitwise all full-vocabulary logits and SHA256 of every existing causal state region at all 257 positions; exact address/history, readiness, per-arm origin, completion/publication/counters",
        "performance_qualified": false, "formal_quality_verdict": Value::Null,
        "environment": {"MTL_DEBUG_LAYER": "1", "QWEN_METAL_LEASE_WAIT": "1",
            "production_lease_required": true, "production_lease_acquired": false, "memory_bypasses": false},
        "stage": "preload_not_authorized"});
    write_report(&mut report_file, &report)?;
    File::open(report_path.parent().context("report parent")?)?.sync_all()?;
    let outcome = (|| -> AResult<()> {
        // Test contexts skip automatic production-lease acquisition.
        let _lease = metal::acquire_metal_benchmark_lease()?;
        report["environment"]["production_lease_acquired"] = json!(true);
        let ctx = MetalContext::new()?;
        admission(&ctx, 0, 128 * 1024 * 1024)?;
        let mut bytes = Vec::new();
        archive::open(&input_path)?
            .take(INPUT_LIMIT + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() as u64 <= INPUT_LIMIT, "input exceeds 16 MiB");
        let frozen: Input = serde_json::from_slice(&bytes)?;
        let gguf = Arc::new(crate::gguf::GgufFile::open(&gguf_path)?);
        let planned = v41_gguf::plan(&gguf)?;
        let g = geometry(&gguf, &planned, n)?;
        let input = select_prefix(&frozen, n, g.vocab)?;
        let plan = CedPlan::shrinking(
            &planned.config,
            &planned.schedule,
            &planned.engram_hash,
            &CedRequest::cold(n, n + STEPS),
        )?;
        let capture = capture_cpu_plan(g.vocab)?;
        let cold = preflight_cold_prefill(
            &ctx,
            &gguf,
            n,
            n + STEPS,
            &[
                PrefillExecution::FullForward,
                PrefillExecution::ShrinkingCed,
            ],
            capture.total_bytes,
        )?;
        ensure!(
            cold.resident_storage.bytes == planned.bytes,
            "resident storage payload inventory"
        );
        ensure!(
            cold.ced.is_some(),
            "current pair requires supported CED geometry"
        );
        let (runtime, resident) = preload_report(&cold)?;
        report["input"] = json!({"path": input_path, "source_raw_json_sha256": archive::sha(&bytes),
            "source_prompt_ids": frozen.prompt.len(), "source_continuation_ids": frozen.continuation.len(),
            "selected_ids_sha256": archive::json_sha(&serde_json::to_value(&input)?)?,
            "prompt_source_stream_range": [0, n], "continuation_source_stream_range": [n, n + STEPS],
            "source_stream": "frozen.prompt concatenated with frozen.continuation; no retokenization, sampling or argmax feedback"});
        report["memory"] = json!({"runtime_upper": runtime, "capture_cpu_plan": capture,
            "resident_storage_plan": null, "combined_preweight_admission": null,
            "core_preflight": {"full_incremental_metal_bytes": cold.full_incremental_metal_bytes,
                "full_incremental_cpu_bytes": cold.full_incremental_cpu_bytes, "ced": cold.ced,
                "runtime_metal_bytes": cold.runtime_metal_bytes, "runtime_cpu_bytes": cold.runtime_cpu_bytes,
                "binding_cpu_bytes": cold.binding_cpu_bytes, "caller_cpu_bytes": cold.caller_cpu_bytes,
                "aggregation": "request plus max(Full, CED), one state at a time; not a sum over policies or repetitions",
                "legacy_report_fields": "resident bookkeeping includes all binding CPU; runtime binding_upload_hash_cpu is incremental runtime CPU only",
                "reservation": "none; all later loader, state and prefill gates remain required"},
            "strategy": "one resident import, one live KV state; retain Full logits+hashes only, consume during CED; no full-state archives"});
        report["artifact_before"] = artifact_identity(&gguf, &planned)?;
        report["device_facts"] = serde_json::to_value(ctx.device_facts())?;
        report["tensor_ops_status"] = json!(format!("{:?}", ctx.tensor_ops_status()));
        write_report(&mut report_file, &report)?;
        report["memory"]["resident_storage_plan"] = serde_json::to_value(&resident)?;
        report["memory"]["combined_upper"] =
            json!({"metal_bytes": cold.required_metal_bytes, "cpu_bytes": cold.required_cpu_bytes});
        report["memory"]["combined_preweight_admission"] = admission_report(
            &cold.admission,
            cold.required_metal_bytes,
            cold.required_cpu_bytes,
        );
        report["stage"] = json!("preweight_admitted_not_complete");
        write_report(&mut report_file, &report)?;
        drop(bytes);
        drop(frozen);
        // This is the only weight import. Keep it downstream of every combined gate.
        let weights = ResidentWeights::load(&ctx, Arc::clone(&gguf))?;
        let model =
            Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
                .map_err(anyhow::Error::msg)?;
        report["stage"] = json!("current_full");
        write_report(&mut report_file, &report)?;
        let (reference, full_execution, full_seconds) =
            objc2::rc::autoreleasepool(|_| full_reference(&model, &input, &g))?;
        report["stage"] = json!("shrinking_ced");
        write_report(&mut report_file, &report)?;
        let (observations, ced_execution, ced_seconds) =
            objc2::rc::autoreleasepool(|_| compare_ced(&model, &input, &g, &plan, reference))?;
        ensure!(observations.len() == STEPS + 1, "paired position coverage");
        let artifact_after = artifact_identity(&gguf, &planned)?;
        ensure!(
            report["artifact_before"] == artifact_after,
            "GGUF identity changed during pair"
        );
        ensure!(
            source_identity()?.1 == source,
            "source/binary provenance changed during pair"
        );
        let mut final_bytes = Vec::new();
        archive::open(&input_path)?
            .take(INPUT_LIMIT + 1)
            .read_to_end(&mut final_bytes)?;
        ensure!(
            report["input"]["source_raw_json_sha256"] == archive::sha(&final_bytes),
            "input changed during pair"
        );
        report["artifact_after"] = artifact_after;
        report["execution"] = json!({"full": full_execution, "ced": ced_execution,
            "resident_load_count": 1, "maximum_simultaneous_kv_states": 1,
            "prefill_origins_checked_each_position": {"full": "full_forward", "ced": "shrinking_ced"},
            "full_state_dropped_before_ced_allocation": true});
        report["coverage"] = json!({"state_positions_per_arm": observations.len(),
            "full_vocabulary_vectors_per_arm": observations.len(), "logit_elements_per_vector": g.vocab,
            "causal_regions_per_position": g.state_keys().len(), "first_position": n, "last_position": n + STEPS,
            "teacher_forced_steps_per_arm": STEPS});
        report["observations"] = json!(observations);
        report["diagnostic_timings"] = json!({"full_native_prefill_seconds": full_seconds,
            "ced_native_prefill_seconds": ced_seconds, "descriptive_pilot_only": true,
            "not_performance_evidence": "single ordered pair; no throughput protocol, repetitions, thermal control, or performance verdict; elapsed prefill excludes post-completion state observations"});
        report["limitations"] = json!([
            "not the frozen original runtime and not an original-golden pass",
            "no boundary activation or per-query control archive comparison; production completion/publication checks and exact counters remain active",
            "no fresh model shard-content hash; same resident object and retained stamps/layout identify this pair",
            "no sampled continuation, snapshot/restore/growth, failure/cancellation or throughput qualification",
            "requires parent bounded pilot controller and central combined host/import budget repair before actual execution"
        ]);
        report["stage"] = json!("complete");
        report["complete"] = json!(true);
        Ok(())
    })();
    if let Err(error) = &outcome {
        report["failure"] = json!(format!("{error:#}"));
    }
    write_report(&mut report_file, &report)?;
    outcome
}

#[test]
fn prefix_selection_uses_true_next_256_ids_and_rejects_bad_inputs() {
    let frozen = Input {
        prompt: (0..8192).collect(),
        continuation: (8192..8448).collect(),
    };
    for n in [128, 2048, 8192] {
        let selected = select_prefix(&frozen, n, 9000).unwrap();
        assert_eq!(selected.prompt, (0..n as u32).collect::<Vec<_>>());
        assert_eq!(
            selected.continuation,
            (n as u32..n as u32 + 256).collect::<Vec<_>>()
        );
    }
    let crossing = Input {
        prompt: (0..256).collect(),
        continuation: (256..512).collect(),
    };
    assert_eq!(
        select_prefix(&crossing, 128, 512).unwrap().continuation,
        (128..384).collect::<Vec<_>>()
    );
    assert!(select_prefix(&crossing, 2048, 9000).is_err());
    assert!(select_prefix(&frozen, 128, 8447).is_err());
    let short = Input {
        prompt: frozen.prompt,
        continuation: vec![0; 255],
    };
    assert!(select_prefix(&short, 128, 9000).is_err());
    for text in [
        "", "0", "1", "127", "129", "2049", "8193", "16384", "0128", "128 ", "-128",
    ] {
        assert!(prompt_n(text).is_err(), "accepted N={text:?}");
    }
}

#[test]
fn capture_and_combined_budget_charge_all_257_full_vocab_vectors() {
    let capture = capture_cpu_plan(Config::default().vocab).unwrap();
    assert_eq!(capture.reference_logits_bytes, 132_899_840);
    assert_eq!(capture.total_bytes, 132_899_840 + 192 * 1024 * 1024);
    assert!(capture_cpu_plan(0).is_err());
    assert!(capture_cpu_plan(usize::MAX).is_err());
}

#[test]
fn compiled_safety_source_refuses_a_stale_body_before_import() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("qwen-pair-source-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("host_budget.rs");
    let compiled = b"compiled safety body\n";
    std::fs::write(&path, compiled).unwrap();
    assert!(checked_compiled_source(&root, "host_budget.rs", compiled).is_ok());
    std::fs::write(&path, b"edited safety body\n").unwrap();
    let refusal = checked_compiled_source(&root, "host_budget.rs", compiled).unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("binary/source mismatch: host_budget.rs")
    );
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&root).unwrap();
}
