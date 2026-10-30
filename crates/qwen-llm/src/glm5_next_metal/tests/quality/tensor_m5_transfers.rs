//! Separate actual-Q4 transfer screen, not the stock four-shard qualification.
use super::*;
use crate::glm5_next_metal::tensor_q5_down;

const ARMS: [(&str, bool, bool); 3] = [
    ("CURRENT", false, false),
    ("Q8WIDE", true, false),
    ("Q5DOWN", false, true),
];
const Q4_PIN: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-16-glm-quality-comparison/stock-q4.json");

fn selected<R>(arm: usize, body: impl FnOnce() -> R) -> R {
    let _dense = tensor_prefill::Scope::new(true);
    let _down = tensor_down::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(ARMS[arm].1);
    let _q5 = tensor_q5_down::Scope::new(ARMS[arm].2);
    body()
}

fn difference(reference: &[f32], logits: &[f32]) -> Value {
    let max_abs = max_abs_diff(logits, reference);
    let kl = kl_divergence(reference, logits);
    assert!(max_abs.is_finite() && kl.is_finite());
    json!({"changed_bits_components":reference.iter().zip(logits)
        .filter(|(a,b)| a.to_bits() != b.to_bits()).count(),
        "max_abs":max_abs,"kl_reference_to_arm":kl,"compared_vocab":reference.len(),
        "all_finite":true})
}

fn profile(ctx: &MetalContext, w: &Glm5NextWeights, tokens: &[u32], cpu: u64, out: &mut Report) {
    for (arm, &(label, wide, q5)) in ARMS.iter().enumerate() {
        let mut reference = None;
        let mut plain = Vec::new();
        for phase in ["warm_census", "plain1", "plain2", "profiled"] {
            let mut s = fresh(ctx, w, ROWS, cpu, label, phase, out);
            assert_plain();
            let census = (phase == "warm_census").then(Census::begin);
            let (logits, wall_ms, report) = selected(arm, || {
                let start = Instant::now();
                let (logits, report) = if phase == "profiled" {
                    let (logits, report) = s.prefill_chunk_stage_profiled(ctx, tokens).unwrap();
                    (logits, Some(report))
                } else {
                    (s.prefill_packed(ctx, tokens).unwrap(), None)
                };
                (logits, start.elapsed().as_secs_f64() * 1e3, report)
            });
            let census = census.map(Census::finish);
            assert_plain();
            check_session(&s, ROWS);
            check_logits(&logits, w.config.vocab_size as usize);
            assert!(wall_ms.is_finite() && wall_ms > 0.0);
            if let Some(census) = census {
                let heap = census.capacity() * size_of::<DispatchCensusRow>()
                    + census
                        .iter()
                        .map(|r| r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity))
                        .sum::<usize>();
                assert!((heap as u64).checked_mul(2).unwrap() <= CENSUS_BYTES);
                let mut kernels = BTreeMap::<&str, usize>::new();
                for row in &census {
                    *kernels.entry(&row.kernel).or_default() += 1;
                }
                let count = |name: &str| kernels.get(name).copied().unwrap_or(0);
                let wide_calls = count("kernel_mat_mat_q8_0_f32_tensor_wide");
                assert_eq!(wide_calls > 0, wide);
                if !wide {
                    assert!(count("kernel_mat_mat_q8_0_f32_tensor") > 0);
                }
                assert_eq!(
                    count("kernel_moe_down_q5_K_f32_grouped_slots_tensor"),
                    if q5 { 42 } else { 0 }
                );
                out.emit(json!({"event":"census","arm":label,"prompt_tokens":ROWS,
                    "expected_chunks":chunks(tokens.len()),"dispatches":census.len(),"kernel_counts":kernels,
                    "census_heap_bytes":heap,"census_upper_bytes":heap*2,"timed":false}));
            }
            let delta = reference
                .as_ref()
                .map(|r: &Vec<f32>| difference(r, &logits));
            out.emit(
                json!({"event":"profile_run","arm":label,"phase":phase,"wall_ms":wall_ms,
                "vs_same_arm_warm":delta,"logits_sha256":sha256_hex(bytemuck::cast_slice(&logits)),
                "ordinary_plain":phase.starts_with("plain"),"logits_finite":true}),
            );
            if phase == "warm_census" {
                reference = Some(logits);
            }
            if phase.starts_with("plain") {
                plain.push(wall_ms);
            }
            if let Some(report) = report {
                let mut stages = BTreeMap::<&str, f64>::new();
                for span in &report.spans {
                    *stages.entry(span.stage.as_str()).or_default() += span.gpu_ms;
                }
                out.emit(json!({"event":"profile","arm":label,"plain_wall_ms":plain,
                    "profiled_wall_ms":wall_ms,"profiled_over_plain_mean":wall_ms/((plain[0]+plain[1])*0.5),
                    "command_gpu_ms":report.command_gpu_ms,"scaled_span_sum_ms":report.span_sum_ms,
                    "stages_ms":stages,"warning":"observer diagnostic: encoder splitting and sample resolution perturb execution; scaled spans are not additive ordinary-wall shares or independent leaf costs"}));
            }
        }
    }
}

fn plain(ctx: &MetalContext, w: &Glm5NextWeights, long: &Document, cpu: u64, out: &mut Report) {
    for n in [2048, 4096] {
        for candidate in [1, 2] {
            for (sample, arm) in [0, candidate, candidate, 0, candidate, 0, 0, candidate]
                .into_iter()
                .enumerate()
            {
                let mut s = fresh(ctx, w, n, cpu, ARMS[arm].0, "plain_abba", out);
                assert_plain();
                let (logits, wall_ms) = selected(arm, || {
                    let start = Instant::now();
                    let logits = s.prefill_packed(ctx, &long.tokens[..n]).unwrap();
                    (logits, start.elapsed().as_secs_f64() * 1e3)
                });
                assert_plain();
                check_session(&s, n);
                check_logits(&logits, w.config.vocab_size as usize);
                assert!(wall_ms.is_finite() && wall_ms > 0.0);
                out.emit(json!({"event":"plain_sample","candidate":ARMS[candidate].0,"arm":ARMS[arm].0,
                    "prompt_tokens":n,"sample":sample,"block":if sample < 4 { "ABBA" } else { "BAAB" },
                    "wall_ms":wall_ms,"expected_prefill_chunks":chunks(n),"final_position":s.position(),
                    "actual_prefill_rows":s.ledger().prefill_rows(),"logits_finite":true}));
            }
        }
    }
}

fn quality(ctx: &MetalContext, w: &Glm5NextWeights, docs: &[Document], cpu: u64, out: &mut Report) {
    for d in docs {
        let mut baseline = Vec::<Vec<f32>>::with_capacity(CONTINUATION);
        for (arm, &(label, _, _)) in ARMS.iter().enumerate() {
            let mut s = fresh(ctx, w, d.prefix + CONTINUATION, cpu, label, "quality", out);
            assert_plain();
            let (mut logits, commits) = selected(arm, || {
                observed_prefill(ctx, &mut s, &d.tokens[..d.prefix], DENSE_AND_DOWN)
            });
            for (i, &target) in d.tokens[d.prefix..].iter().enumerate() {
                check_session(&s, d.prefix + i);
                check_logits(&logits, w.config.vocab_size as usize);
                if arm == 0 {
                    baseline.push(logits.clone());
                }
                let reference = &baseline[i];
                let nll = -log_prob(&logits, target);
                let reference_nll = -log_prob(reference, target);
                assert!(nll.is_finite() && reference_nll.is_finite());
                let prediction = argmax(&logits);
                let reference_top1 = argmax(reference);
                out.emit(json!({"event":"quality_token","arm":label,"path":d.path,"stratum":d.stratum,
                    "prefix":d.prefix,"index":i,"target_position":d.prefix+i,"target":target,
                    "difference":difference(reference, &logits),"nll":nll,"current_nll":reference_nll,
                    "nll_delta":nll-reference_nll,"top1":prediction,"current_top1":reference_top1,
                    "top1_hit":prediction == target as usize,"current_top1_hit":reference_top1 == target as usize,
                    "top1_agrees_current":prediction == reference_top1}));
                if i + 1 < CONTINUATION {
                    logits = s.forward(ctx, target).unwrap();
                }
            }
            assert_plain();
            assert_eq!(baseline.len(), CONTINUATION);
            check_session(&s, d.prefix + CONTINUATION - 1);
            out.emit(
                json!({"event":"document_arm","arm":label,"path":d.path,"prefix":d.prefix,
                "logit_rows":CONTINUATION,"vocab":w.config.vocab_size,"prefill_chunks":commits,
                "decode_calls":CONTINUATION-1,"final_position":s.position()}),
            );
        }
    }
}

#[test]
#[ignore = "release actual six-shard UD-Q4_K_XL M5 transfer screen; GLM53_GGUF, fresh GLM53_M5_TRANSFER_REPORT; MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 QWEN_MATMAT_Q6_K_N64_MIN_N=1024"]
fn tensor_m5_transfer_screen() {
    environment();
    let _lease = production_lease();
    let mut out = Report::new("GLM53_M5_TRANSFER_REPORT", true);
    let docs = documents();
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").unwrap());
    let gguf = GgufFile::open(&path).unwrap();
    let pin: Value = serde_json::from_slice(Q4_PIN).unwrap();
    assert_eq!(gguf.shards.len(), 6);
    assert_eq!(gguf.tensors.len(), 1412);
    for (shard, expected) in gguf.shards.iter().zip(pin["shards"].as_array().unwrap()) {
        assert_eq!(
            shard.path.file_name().unwrap().to_str().unwrap(),
            expected["name"].as_str().unwrap()
        );
        assert_eq!(shard.mmap_len() as u64, expected["size"].as_u64().unwrap());
    }
    let header_sha = sha256_hex(gguf.primary_shard().mmap_bytes());
    assert_eq!(header_sha, pin["shards"][0]["sha256"].as_str().unwrap());
    let layout = artifact_layout(&gguf);
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_ne!(layout, fixture["artifact_layout"]);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    assert!(artifact.packed_prefill());
    let config = artifact.config();
    assert_eq!(config.blocks.len(), 45);
    assert_eq!(config.leading_dense_block_count, 3);
    for block in 3..45 {
        for (suffix, dtype, shape) in [
            ("down", GgmlType::Q5_K, [2048, 4096, 288]),
            ("gate", GgmlType::Q4_K, [4096, 2048, 288]),
            ("up", GgmlType::Q4_K, [4096, 2048, 288]),
        ] {
            let t = gguf
                .tensors
                .iter()
                .find(|t| t.name == format!("blk.{block}.ffn_{suffix}_exps.weight"))
                .unwrap();
            assert_eq!(t.dtype, dtype);
            assert_eq!(t.shape, shape);
        }
    }
    let vocab = config.vocab_size;
    assert!(docs.iter().all(|d| d.tokens.iter().all(|&t| t < vocab)));
    let capacity = docs.iter().map(|d| d.prefix + CONTINUATION).max().unwrap();
    assert_eq!(capacity, 4416);
    assert!(capacity <= config.context_length as usize);
    let terms = cpu_terms(vocab, false);
    let cpu = terms
        .iter()
        .try_fold(0u64, |n, (_, bytes)| n.checked_add(*bytes))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.tensor_ops_status().is_available(),
        "actual M5 tensor tier required"
    );
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    let mut shapes = BTreeMap::<String, usize>::new();
    for t in &gguf.tensors {
        *shapes
            .entry(format!("{:?} {:?}", t.dtype, t.shape))
            .or_default() += 1;
    }
    out.emit(json!({"event":"header","schema":"glm53.m5_transfers.v1","model":path,
        "artifact_layout":layout,"stock_fixture_artifact_layout":fixture["artifact_layout"],
        "artifact_layout_matches_frozen_fixture":false,"artifact_pin":pin,"metadata_shard_sha256":header_sha,
        "artifact_identity_scope":"pinned shard names/sizes and metadata-only primary SHA; actual tensor-table digest and retained stamps recorded; weight payload hashes not verified",
        "shards":stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"bytes":s.size,
            "device":s.device,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,
            "ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>(),
        "config":format!("{config:?}"),"tensor_shape_counts":shapes,
        "executed_expert_inventory":{"main_blocks":45,"q5_down":42,"q4_gate_up":84},
        "binary":std::env::current_exe().unwrap(),"binary_sha256":binary_hash(),
        "fixture_sha256":sha256_hex(FROZEN),"probe_sha256":sha256_hex(include_bytes!("tensor_m5_transfers.rs")),
        "dense_source_sha256":sha256_hex(include_bytes!("../../tensor_prefill.rs")),
        "q5_source_sha256":sha256_hex(include_bytes!("../../tensor_q5_down.rs")),
        "packed_source_sha256":sha256_hex(include_bytes!("../../packed.rs")),
        "arms":ARMS,"arm_columns":["label","q8_wide","q5_down"],"dense_scope":true,"down_scope":true,
        "scope":"independent thread-local overrides during whole prefill only; no combined candidate; ordinary forward continuation unchanged",
        "documents":docs.iter().map(|d|json!({"path":d.path,"stratum":d.stratum,"prefix":d.prefix,
            "tokens":d.tokens,"tokens_sha256":token_hash(&d.tokens)})).collect::<Vec<_>>(),
        "token_source":"frozen first-in-stratum quality-v1 IDs, u32 little-endian hashes; no retokenization or stock-artifact equivalence claim",
        "capacity":capacity,"workspace_rows":ROWS,"cpu_reserve_bytes":cpu,"cpu_terms":terms,
        "retained_baselines":1,"baseline_rows":CONTINUATION,"vocab":vocab,"targets_per_arm":192,
        "lineage":"Fast","precision":"Half","tensor_tier":format!("{:?}",ctx.tensor_ops_status()),
        "environment": ["MTL_DEBUG_LAYER","QWEN_METAL_LEASE_WAIT","QWEN_MATMAT_Q6_K_N64_MIN_N",
            "QWEN_MATMAT_Q6_K_N64","QWEN_EXPERT_TENSOR_TILE","QWEN_GLM53_DENSE_Q8_WIDE","QWEN_GLM53_TENSOR_Q5_DOWN"]
            .map(|key| (key,std::env::var(key).ok())),
        "plain_prompts":"first frozen long document, prefixes 2048/4096; ABBA then BAAB CURRENT vs each candidate",
        "timing":"diagnostic only, validation ON; ordinary prefill includes GPU wait and normal logits readback, excludes allocation/checks/JSON; no census/profiler in plain samples; parent qwen-bench same binary without DEBUG supplies canonical performance evidence",
        "policy":"descriptive full-vocab arithmetic/NLL/KL/top1 screen, no equality/quality threshold or promotion; incomplete without complete event"}));
    let weights = {
        let _allocation = ctx.begin_allocation_transaction();
        let preflight = preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
        assert_eq!(preflight.ledger.prefill_rows(), ROWS as u64);
        let admission = crate::metal::evaluate_metal_memory_admission(
            preflight.ledger.peak_bytes().checked_add(cpu).unwrap(),
            0,
            ctx.memory_signals(),
            true,
        );
        out.emit(json!({"event":"preload_admission","capacity":capacity,"workspace_rows":ROWS,
            "full_preflight_peak_bytes":preflight.ledger.peak_bytes(),"ledger":preflight.ledger.terms(),
            "cpu_reserve_bytes":cpu,"admitted":admission.is_admitted(),"reason":admission.reason.as_str(),
            "required_bytes":admission.required_bytes,"allow_zero_process_budget":true,
            "cpu_charged_to_nonzero_process_budget":true,"cpu_charged_to_working_set_headroom":true,
            "current_gpu_bytes":admission.signals.current_allocated_bytes,
            "process_limit_remaining_bytes":admission.signals.process_limit_remaining_bytes,
            "working_set_headroom_bytes":admission.working_set_headroom_bytes}));
        assert!(
            admission.is_admitted(),
            "full model + session + CPU refused: {:?}",
            admission.reason
        );
        Glm5NextWeights::load(&ctx, &gguf).unwrap()
    };
    assert_eq!(weights.config, *config);
    profile(&ctx, &weights, &docs[0].tokens[..ROWS], cpu, &mut out);
    plain(&ctx, &weights, &docs[2], cpu, &mut out);
    quality(&ctx, &weights, &docs, cpu, &mut out);
    assert_plain();
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    out.emit(
        json!({"event":"complete","execution_complete":true,"arms":3,"documents_per_arm":3,
        "targets_per_arm":192,"quality_rows":576,"plain_samples":32,"profiled_calls":3,
        "quality_verdict":null,"performance_verdict":null}),
    );
    out.sync();
}
