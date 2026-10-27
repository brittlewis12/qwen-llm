//! Same-resident exact CED host-scheduling discriminator, never a speed qualification.
use super::*;

const DECODE: usize = 2;
const RESERVE: u64 = 64 << 20;
const FULL: PrefillExecution = PrefillExecution::FullForward;
const CED: PrefillExecution = PrefillExecution::ShrinkingCed;
const SELECTOR: &str = "deepseek_v41::forward::ced::qualification::current_pair::range_pilot::range_host_full_model_native_abba";
const ORDER: [&str; 7] = [
    "warm_Full",
    "warm_A_oldhost",
    "warm_B_range",
    "primary_A_oldhost",
    "primary_B_range",
    "primary_B_range",
    "primary_A_oldhost",
];

#[test]
#[ignore = "PRIVATE release native aarch64 macOS, actual NAX, fail-fast self-lease; one GSQ import, frozen IDs/private report; parent <=60s N2048/default or <=90s explicitly selected N8192 process cap"]
fn range_host_full_model_native_abba() {
    run().unwrap();
}

fn selected_rows(text: Option<&str>) -> AResult<usize> {
    match text {
        None | Some("2048") => Ok(2048),
        Some("8192") => Ok(8192),
        _ => anyhow::bail!(
            "private V41_CED_RANGE_ROWS must be absent, 2048 or 8192; no automatic escalation"
        ),
    }
}

fn limit_seconds(n: usize) -> u64 {
    if n == 2048 { 60 } else { 90 }
}

fn budget(start: std::time::Instant, n: usize) -> AResult<()> {
    ensure!(
        start.elapsed() < std::time::Duration::from_secs(limit_seconds(n)),
        "cooperative process budget exhausted; no next operation authorized (not an interrupt of blocking load/GPU calls)"
    );
    Ok(())
}

fn controls() -> AResult<Value> {
    ensure!(
        cfg!(all(
            target_arch = "aarch64",
            target_os = "macos",
            target_endian = "little"
        )) && !cfg!(debug_assertions),
        "aarch64 macOS little-endian release required"
    );
    ensure!(
        crate::enabled_diagnostic_features().is_empty(),
        "diagnostic features refused"
    );
    let mut stable = BTreeMap::new();
    for (name, value) in std::env::vars_os() {
        let name = name.to_string_lossy();
        let expected = match name.as_ref() {
            "QWEN_PRODUCT_METALLIB_TARGET" => Some(crate::PRODUCT_METALLIB_TARGET),
            "QWEN_RESEARCH_METALLIB_TARGET" => Some(crate::RESEARCH_METALLIB_TARGET),
            "QWEN_TENSOR_METALLIB_TARGET" => Some(crate::TENSOR_METALLIB_TARGET),
            "QWEN_TENSOR_METALLIB_TIERS" => Some(crate::TENSOR_METALLIB_TIERS),
            "QWEN_TENSOR_METAL40_PROBE" => Some(crate::TENSOR_METAL40_PROBE),
            "QWEN_TENSOR_METAL41_PROBE" => Some(crate::TENSOR_METAL41_PROBE),
            "QWEN_METALLIB_COMPILER_IDENTITY" => Some(crate::metallib_build_identity().0),
            "QWEN_METALLIB_SDK_VERSION" => Some(crate::metallib_build_identity().1),
            _ => None,
        };
        if let Some(expected) = expected {
            ensure!(
                value == std::ffi::OsStr::new(expected),
                "build override: {name}"
            );
        } else if name == "QWEN_METAL_LEASE_WAIT" {
            ensure!(
                value == "0",
                "WAIT must be absent or 0 for fail-fast self-lease"
            );
        } else {
            ensure!(
                ![
                    "QWEN_",
                    "MTL_",
                    "DYLD_",
                    "METAL_",
                    "GGML_METAL_",
                    "SHADER_VALIDATION",
                    "INSTRUMENTS_",
                    "DT_",
                    "LLVM_PROFILE_",
                    "CPUPROFILE",
                    "Malloc",
                    "NSZombie"
                ]
                .iter()
                .any(|p| name.starts_with(p)),
                "runtime/validation/injection/profiler control refused: {name}"
            );
        }
        if expected.is_some() || name.starts_with("QWEN_") || name == "V41_CED_RANGE_ROWS" {
            stable.insert(name.into_owned(), value.to_string_lossy().into_owned());
        }
    }
    ensure!(
        metal::diagnostics_observer_active_counts() == [0; 3]
            && profile::start().is_none()
            && !profile::stages()
            && !profile::dispatches()
            && ACTIVE.with(|s| s.borrow().is_none())
            && !prefill::F32_LINEAR_ROWS.with(|f| f.get())
            && !prefill::ROWWISE_LINEAR_ROWS.with(|f| f.get()),
        "observer/arithmetic override active"
    );
    Ok(json!(stable))
}

fn identity() -> AResult<(PathBuf, Value)> {
    let (root, mut source) = source_identity()?;
    macro_rules! embedded {
        ($($p:literal),+ $(,)?) => { &[$(($p, include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../", $p)).as_slice())),+] };
    }
    let files: &[(&str, &[u8])] = embedded![
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/current_pair/range_pilot.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/current_pair.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/provenance.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/tests/query_batch/range_execution.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/tests/query_batch.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/tests.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/query_batch/range.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/planning.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/preflight.rs",
        "scripts/reference/acceptance_inventory.json",
    ];
    source["range_pilot_compiled_sources"] = json!(
        files
            .iter()
            .map(|(p, b)| checked_compiled_source(&root, p, b))
            .collect::<AResult<Vec<_>>>()?
    );
    Ok((root, source))
}

fn select(source: &Input, n: usize, vocab: usize) -> AResult<Input> {
    ensure!(
        [2048, 8192].contains(&n) && source.prompt.len() >= n,
        "selected frozen prefix unavailable"
    );
    let continuation: Vec<_> = source.prompt[n..]
        .iter()
        .chain(&source.continuation)
        .take(DECODE)
        .copied()
        .collect();
    ensure!(
        continuation.len() == DECODE
            && source.prompt[..n]
                .iter()
                .chain(&continuation)
                .all(|&t| (t as usize) < vocab),
        "two true next IDs and valid selected prefix required"
    );
    Ok(Input {
        prompt: source.prompt[..n].to_vec(),
        continuation,
    })
}

fn validate_ced(model: &Forward<'_>, r: &PrefillReport, p: &CedPlan) -> AResult<()> {
    ensure!(
        r.execution == CED && r.fallback.is_none(),
        "exact CED/no fallback required"
    );
    let c = r.counters.as_ref().context("missing CED counters")?;
    c.validate(p).map_err(anyhow::Error::msg)?;
    let commands = production_command_count(p)?;
    ensure!(
        c.encoded_commands == commands && c.completed_commands == commands,
        "encoded/completed production command coverage"
    );
    for l in &c.layers {
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
            "selection producer coverage"
        );
    }
    let memory = r.memory.as_ref().context("missing CED memory")?;
    let estimated = super::super::super::estimate(model, p.prompt_rows + DECODE, p)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        admission_ledger(memory) == admission_ledger(&estimated)
            && memory.incremental_cpu_bytes > 0
            && memory.incremental_metal_bytes > 0
            && r.observed_incremental_metal_bytes
                .context("missing observed allocation")?
                <= memory.incremental_metal_bytes
            && memory
                .process_ru_maxrss_before_bytes
                .zip(memory.process_ru_maxrss_after_bytes)
                .is_some_and(|(a, b)| b >= a),
        "CED ledger/allocation/RSS contract"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn arm(
    model: &Forward<'_>,
    request: &RequestPlan,
    input: &Input,
    g: &Geometry,
    plan: &CedPlan,
    sample: usize,
    full: Option<&[ReferenceRow]>,
    old: Option<&[ReferenceRow]>,
    raw: &mut File,
    start: std::time::Instant,
    mut publish: impl FnMut(&Value) -> AResult<()>,
) -> AResult<Vec<ReferenceRow>> {
    let n = g.n;
    let execution = if sample == 0 { FULL } else { CED };
    let mut run = || -> AResult<Vec<ReferenceRow>> {
        budget(start, n)?;
        let mut state = model.new_state_from_plan(request)?;
        ensure!(state.capacity() == n + DECODE, "exact capacity required");
        budget(start, n)?;
        let began = std::time::Instant::now();
        let result = model.prefill_with_execution(&mut state, &input.prompt, None, execution);
        let seconds = began.elapsed().as_secs_f64();
        let mut evidence = json!({"arm": ORDER[sample], "primary": sample >= 3, "complete": false,
            "prefill_seconds": seconds, "prefill_returned": true, "observations": [], "teacher_forced_seconds": [],
            "timing_scope": "host prefill API including command completion/status and endpoint logits copy; excludes fixture state allocation, TF, external causal/logit hashing, raw staging/report I/O and teardown; production API-internal allocation/hashing remains included"});
        publish(&evidence)?;
        let (mut logits, r) = result?;
        evidence["prefill_report"] = serde_json::to_value(&r)?;
        publish(&evidence)?;
        budget(start, n)?;
        let admitted = r
            .admission
            .as_ref()
            .context("missing production admission")?;
        ensure!(
            r.execution == execution
                && r.fallback.is_none()
                && r.geometry == request.prefill()
                && admitted.fallback.is_none()
                && admitted.intent == PrefillIntent::RequiredFullCed,
            "execution/admission/geometry contract"
        );
        if sample == 0 {
            ensure!(r.counters.is_none(), "Full counters must be absent");
        } else {
            validate_ced(model, &r, plan)?;
        }
        let mut retained = Vec::new();
        for i in 0..=DECODE {
            budget(start, n)?;
            ensure!(
                logits.len() == g.vocab && logits.iter().all(|v| v.is_finite()),
                "complete finite vocabulary at {i}"
            );
            let observed = observe_state(model, &state, g, n + i, execution)?;
            ensure!(
                observed["regions"]
                    .as_array()
                    .is_some_and(|r| r.len() == 56),
                "canonical causal inventory"
            );
            let bits: &[u8] = bytemuck::cast_slice(&logits);
            let offset = raw.stream_position()?;
            raw.write_all(bits)?;
            raw.sync_data()?;
            let mut difference = Difference::default();
            if let Some(full) = full {
                difference.add(
                    bytemuck::cast_slice(&full[i].logits),
                    bits,
                    true,
                    &json!({"position": n+i, "role": "complete_vocabulary"}),
                    false,
                )?;
            }
            let full_equal = full.map(|f| bytemuck::cast_slice::<f32, u8>(&f[i].logits) == bits);
            let old_equal = old.map(|o| o[i].state == observed);
            evidence["observations"].as_array_mut().unwrap().push(json!({"position": n+i,
                "state": observed, "ready": state.is_ready(), "origin": state.prefill_origin(),
                "status": "production completion/status checks plus snapshot_payload_bytes/validate_snapshot_position succeeded",
                "logits_sha256_f32le": archive::sha(bits), "logit_elements": g.vocab, "raw_offset": offset, "raw_bytes": bits.len(),
                "full_logits_bitwise_equal": full_equal, "full_logits_difference": full.map(|_| difference.json()),
                "full_causal_diagnostic_equal": full.map(|f| f[i].state == observed),
                "old_ced_causal_equal": old_equal, "next_teacher_forced_id": input.continuation.get(i)}));
            publish(&evidence)?;
            ensure!(
                full_equal != Some(false),
                "Full/CED complete-vocabulary FLOATBITS mismatch at {}",
                n + i
            );
            ensure!(
                old_equal != Some(false),
                "old-CED/range causal/readiness/address/hash mismatch at {}",
                n + i
            );
            if sample <= 1 {
                retained.push(ReferenceRow {
                    logits: std::mem::take(&mut logits),
                    state: observed,
                });
            }
            if i < DECODE {
                budget(start, n)?;
                let began = std::time::Instant::now();
                let next = model.step(&mut state, input.continuation[i], None);
                evidence["teacher_forced_seconds"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!(began.elapsed().as_secs_f64()));
                publish(&evidence)?;
                logits = next.map_err(anyhow::Error::msg)?;
            }
        }
        budget(start, n)?;
        evidence["complete"] = json!(true);
        publish(&evidence)?;
        Ok(retained)
    };
    if [1, 3, 6].contains(&sample) {
        attention_gpu::with_old_host_loop(run)
    } else {
        run()
    }
}

fn run() -> AResult<()> {
    let start = std::time::Instant::now();
    let rows_control = std::env::var("V41_CED_RANGE_ROWS")
        .map(Some)
        .or_else(|e| match e {
            std::env::VarError::NotPresent => Ok(None),
            e => Err(e),
        })?;
    let n = selected_rows(rows_control.as_deref())?;
    let gguf_path = env_path("V41_CED_GGUF")?;
    let input_path = env_path("V41_CED_TOKENS")?;
    let report_path = env_path("V41_CED_CURRENT_PAIR_REPORT")?;
    let (checkout, source) = identity()?;
    ensure!(
        !input_path.canonicalize()?.starts_with(&checkout)
            && !gguf_path.canonicalize()?.starts_with(&checkout),
        "private artifact/input must be outside checkout"
    );
    let mut file = private_report(&report_path, &checkout)?;
    let mut raw_name = report_path.as_os_str().to_os_string();
    raw_name.push(".logits.f32le");
    let raw_path = PathBuf::from(raw_name);
    let mut report = json!({"schema": "v41-range-native-pilot-v1", "test_selector": SELECTOR,
        "complete": false, "performance_qualified": false, "source_before": source, "arms": [], "arm_order": ORDER,
        "N": n, "requested_rows_control": rows_control, "private_rows_choice_only": true, "automatic_escalation": false,
        "capacity": n+DECODE, "teacher_forced_steps_per_arm": DECODE,
        "predeclared_totals": {"prefills": 7, "prompt_tokens": 7*n, "scalar_steps": 14, "full_vocab_vectors": 21},
        "protocol": "warm Full then warm A=oldhost exactCED, warm B=range exactCED; primary ABBA; single import/binding, one live KV state, default None mapping/common chunk settings; all samples retained, no retry/filter/feedback",
        "correctness_contract": "all CED complete-vocabulary vectors FLOATBITS equal Full; oldCED vs range causal hashes/address/ready/status/origin checked at all three positions; Full/CED state equality diagnostic only, no inactive-state equivalence demand",
        "raw_logits": {"path": raw_path, "encoding": "IEEE F32 little endian, complete vocabulary, declared arm then position order"},
        "parent_process_cap_seconds": limit_seconds(n), "supervision": "parent MUST enforce selected 60s/90s whole-process cap; first launch N2048 only; internal checks cannot interrupt blocking import/GPU/report/hash",
        "limitations": ["same resident actual-model discriminator, not original-golden/quality/authenticity or global model content hash", "no performance verdict or filesystem-cold/warmup-sufficiency claim", "publisher eligibility repair out of scope: inherited unconditional CDF publication rollbacks remain; N8192 is explicit private follow-up only, not a new qualification claim"],
        "stage": "controls_and_self_lease_pending"});
    write_report(&mut file, &report)?;
    File::open(report_path.parent().context("report parent")?)?.sync_all()?;
    let mut lease = None;
    let mut raw_file = None;
    let outcome = (|| -> AResult<()> {
        let stable = controls()?;
        report["stable_controls"] = stable.clone();
        budget(start, n)?;
        lease = Some(metal::acquire_metal_benchmark_lease()?);
        report["stage"] = json!("self_leased_before_context");
        write_report(&mut file, &report)?;
        raw_file = Some(private_report(&raw_path, &checkout)?);
        let raw = raw_file.as_mut().unwrap();
        File::open(report_path.parent().context("report parent")?)?.sync_all()?;
        let ctx = MetalContext::new()?;
        ensure!(
            ctx.tensor_ops_status().is_available() && metal::moe_grouped_tensor_selected(&ctx),
            "actual-device NAX/TensorOps capability and production tensor route required; no device-name gate"
        );
        admission(&ctx, 0, 96 << 20)?;
        budget(start, n)?;
        let mut bytes = Vec::new();
        archive::open(&input_path)?
            .take(INPUT_LIMIT + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= INPUT_LIMIT,
            "frozen input exceeds 16 MiB"
        );
        let frozen: Input = serde_json::from_slice(&bytes)?;
        budget(start, n)?;
        let gguf = Arc::new(crate::gguf::GgufFile::open(&gguf_path)?);
        let planned = v41_gguf::plan(&gguf)?;
        let g = geometry(&gguf, &planned, n)?;
        let input = select(&frozen, n, g.vocab)?;
        report["input"] = json!({"path": input_path, "source_json_sha256": archive::sha(&bytes),
            "selected_ids_sha256": archive::json_sha(&serde_json::to_value(&input)?)?, "N": n, "prompt_range": [0,n], "continuation_range": [n,n+DECODE],
            "stream": "frozen.prompt then frozen.continuation; needed IDs only, no retokenization or adaptive choice"});
        drop(bytes);
        drop(frozen);
        let references = product(&[2, 3, g.vocab as u64, 4])?;
        let caller = add(&[references, RESERVE])?;
        let preferred = prefill_rows_for_resident(planned.bytes.resident_bytes());
        report["encoder_geometry"] = json!({"resident_bytes": planned.bytes.resident_bytes(),
            "preferred_rows": preferred, "selected_rows": null, "actual_chunk_count": null,
            "expected_chunk_count": n.div_ceil(preferred),
            "contract": "normal strict joint admission; refuse pressure-demoted width, never pin or bypass planner"});
        write_report(&mut file, &report)?;
        budget(start, n)?;
        let cold =
            plan_cold_prefill_with_target(&ctx, &gguf, n, n + DECODE, &[FULL, CED], caller, None)?
                .into_strict_for_test(ctx.memory_signals())?;
        report["encoder_geometry"]["selected_rows"] = json!(cold.full_chunk_rows);
        report["encoder_geometry"]["actual_chunk_count"] = json!(cold.geometry.chunk_count());
        report["encoder_geometry"]["admitted_input_geometry"] =
            serde_json::to_value(cold.geometry)?;
        write_report(&mut file, &report)?;
        ensure!(
            cold.executions == [FULL, CED]
                && cold.ced.is_some()
                && cold.full_chunk_rows == preferred
                && cold.geometry.chunk_count() == n.div_ceil(preferred)
                && cold.resident_storage.soft_coalescing_target.is_none()
                && cold.resident_storage.bytes == planned.bytes,
            "strict joint Full+CED, resident-byte-derived preferred encoder width/default None storage required; pressure-demoted geometry refused"
        );
        let plan = CedPlan::shrinking(
            &planned.config,
            &planned.schedule,
            &planned.engram_hash,
            &CedRequest::cold(n, n + DECODE).with_full_chunk_rows(cold.full_chunk_rows),
        )?;
        let (runtime, resident) = preload_report(&cold)?;
        report["memory"] = json!({"retained_reference_vectors": 6, "retained_reference_bytes": references,
            "cpu_reserve_bytes": RESERVE, "caller_cpu_bytes": caller, "preparse_cpu_admission_bytes": 96u64 << 20,
            "reserve_scope": "current vector, input parse/final recheck, causal hashes/report, artifact descriptor staging and raw file/page-cache staging; all raw vectors streamed, not retained in heap",
            "raw_file_bytes": product(&[7,3,g.vocab as u64,4])?, "runtime_upper": runtime, "resident": resident,
            "cold_peak": admission_report(&cold.admission, cold.required_metal_bytes, cold.required_cpu_bytes),
            "core_preflight": {"executions": cold.executions, "admission": cold.prefill_admission,
                "capacity": cold.capacity, "full_chunk_rows": cold.full_chunk_rows, "ced": cold.ced,
                "runtime_metal_bytes": cold.runtime_metal_bytes, "runtime_cpu_bytes": cold.runtime_cpu_bytes,
                "runtime_gate_cpu_bytes": cold.runtime_gate_cpu_bytes, "binding_cpu_bytes": cold.binding_cpu_bytes,
                "binding_metal_bytes": cold.binding_metal_bytes, "loader_control_cpu_bytes": cold.loader_control_cpu_bytes,
                "descriptor_window_cpu_bytes": cold.descriptor_window_cpu_bytes, "retained_summary_cpu_bytes": cold.retained_summary_cpu_bytes,
                "loader_controls": {"payload_bytes_read": cold.loader_controls.payload_bytes_read,
                    "cold_cpu_bytes": cold.loader_controls.cold_cpu_bytes, "retained_cpu_bytes": cold.loader_controls.retained_cpu_bytes,
                    "source_bookkeeping_bytes": cold.loader_controls.source_bookkeeping_bytes}},
            "accounting": "strict RequiredFullCed request + max(Full,CED), not sum over arms; binding/loader/retained resources included, refresh runtime before each state; no reservation"});
        report["artifact_before"] = artifact_identity(&gguf, &planned)?;
        report["device_facts"] = serde_json::to_value(ctx.device_facts())?;
        report["tensor_ops_status"] = json!(format!("{:?}", ctx.tensor_ops_status()));
        report["stage"] = json!("strict_joint_admitted_before_single_import");
        write_report(&mut file, &report)?;
        budget(start, n)?;
        let began = std::time::Instant::now();
        let loaded = ResidentWeights::load_with_storage_plan(
            &ctx,
            Arc::clone(&gguf),
            &cold.resident_storage,
        );
        report["resident_import_seconds"] = json!(began.elapsed().as_secs_f64());
        write_report(&mut file, &report)?;
        let weights = loaded?;
        budget(start, n)?;
        validate_loaded_sources(&gguf, &planned, &weights.engram)?;
        ensure!(
            weights.token_map == planned.token_map
                && weights.raw_pad == planned.raw_pad
                && weights.engram_hash == planned.engram_hash,
            "loaded normalized controls changed"
        );
        report["retained_resources"] = serde_json::to_value(&weights.retained_resources)?;
        let model =
            Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
                .map_err(anyhow::Error::msg)?;
        let request = model.plan_request_from_cold(&cold)?;
        let mut full = None;
        let mut old = None;
        for (sample, name) in ORDER.iter().enumerate() {
            budget(start, n)?;
            ensure!(controls()? == stable, "controls changed before arm");
            let gate = cold.assess_runtime(ctx.memory_signals())?;
            let runtime_gate =
                admission_report(&gate, cold.runtime_metal_bytes, cold.runtime_gate_cpu_bytes);
            report["stage"] = json!(name);
            report["arms"].as_array_mut().unwrap().push(json!({"arm": name, "sample": sample, "complete": false, "runtime_admission": runtime_gate}));
            write_report(&mut file, &report)?;
            let save = |e: &Value| {
                report["arms"][sample] = e.clone();
                report["arms"][sample]["sample"] = json!(sample);
                report["arms"][sample]["runtime_admission"] = runtime_gate.clone();
                write_report(&mut file, &report)
            };
            let captured = objc2::rc::autoreleasepool(|_| {
                arm(
                    &model,
                    &request,
                    &input,
                    &g,
                    &plan,
                    sample,
                    full.as_deref(),
                    old.as_deref(),
                    raw,
                    start,
                    save,
                )
            })?;
            if sample == 0 {
                full = Some(captured);
            } else if sample == 1 {
                old = Some(captured);
            }
            report["arms"][sample]["memory_after_arm"] = admission_report(
                &cold.assess_runtime(ctx.memory_signals())?,
                cold.runtime_metal_bytes,
                cold.runtime_gate_cpu_bytes,
            );
            write_report(&mut file, &report)?;
        }
        budget(start, n)?;
        validate_loaded_sources(&gguf, &planned, &weights.engram)?;
        report["artifact_after"] = artifact_identity(&gguf, &planned)?;
        report["source_after"] = identity()?.1;
        report["controls_after"] = controls()?;
        ensure!(
            report["artifact_before"] == report["artifact_after"]
                && report["source_after"] == source
                && report["controls_after"] == stable,
            "source/artifact/controls changed"
        );
        budget(start, n)?;
        let mut final_bytes = Vec::new();
        archive::open(&input_path)?
            .take(INPUT_LIMIT + 1)
            .read_to_end(&mut final_bytes)?;
        ensure!(
            final_bytes.len() as u64 <= INPUT_LIMIT
                && report["input"]["source_json_sha256"] == archive::sha(&final_bytes),
            "frozen IDs changed"
        );
        report["raw_logits"]["retained_bytes"] = json!(raw.metadata()?.len());
        ensure!(
            raw.metadata()?.len() == product(&[7, 3, g.vocab as u64, 4])?,
            "21 raw complete vectors required"
        );
        budget(start, n)?;
        report["execution"] = json!({"resident_imports": 1, "forward_bindings": 1, "maximum_live_kv_states": 1,
            "warmup_arms": 3, "primary_arms": 4, "prefill_prompt_tokens": 7*n, "scalar_steps": 14,
            "vectors": 21, "positions_per_arm": 3, "causal_hashes_per_position": 56,
            "all_full_ced_vectors_bitwise_equal": true, "all_old_ced_range_causal_observations_equal": true,
            "all_samples_retained_without_filter_or_retry": true, "no_fallback": true});
        report["stage"] = json!("complete");
        report["complete"] = json!(true);
        Ok(())
    })();
    if let Err(e) = &outcome {
        report["failure"] = json!(format!("{e:#}"));
    }
    report["aggregate_test_elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    if let Some(raw) = raw_file.as_mut() {
        report["raw_logits"]["retained_bytes"] = json!(raw.metadata()?.len());
        raw.sync_all()?;
        raw.set_permissions(std::fs::Permissions::from_mode(0o400))?;
    }
    write_report(&mut file, &report)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o400))?;
    file.sync_all()?;
    drop(lease);
    outcome
}

#[test]
fn preferred_encoder_geometry_uses_actual_chunk_ceiling() {
    for (resident_bytes, width, chunks) in [
        (217u64 << 30, 4096, [1, 2, 2]),
        (241u64 << 30, 2048, [1, 4, 3]),
    ] {
        let preferred = prefill_rows_for_resident(resident_bytes);
        assert_eq!(preferred, width);
        for (n, expected) in [2048usize, 8192, 4097].into_iter().zip(chunks) {
            let geometry = PrefillGeometry {
                start: 0,
                tokens: n,
                selected_rows: preferred,
            };
            assert_eq!(geometry.chunk_count(), expected);
            assert_eq!(geometry.chunk_count(), n.div_ceil(preferred));
        }
    }
}

#[test]
fn frozen_prefix_uses_two_true_next_ids_only() {
    assert_eq!(selected_rows(None).unwrap(), 2048);
    for text in ["", "128", "4096", "8193", " 8192"] {
        assert!(selected_rows(Some(text)).is_err());
    }
    for n in [2048, 8192] {
        assert_eq!(selected_rows(Some(&n.to_string())).unwrap(), n);
        assert_eq!(limit_seconds(n), if n == 2048 { 60 } else { 90 });
        let mut input = Input {
            prompt: (0..n as u32 + 2).collect(),
            continuation: vec![u32::MAX],
        };
        let next = [n as u32, n as u32 + 1];
        assert_eq!(select(&input, n, n + 2).unwrap().continuation, next);
        assert!(select(&input, n, n + 1).is_err());
        input.prompt.truncate(n + 1);
        input.continuation = vec![n as u32 + 1, u32::MAX];
        assert_eq!(select(&input, n, n + 2).unwrap().continuation, next);
        input.continuation.clear();
        assert!(select(&input, n, n + 2).is_err());
        input.prompt.truncate(n - 1);
        assert!(select(&input, n, n + 2).is_err());
    }
}
