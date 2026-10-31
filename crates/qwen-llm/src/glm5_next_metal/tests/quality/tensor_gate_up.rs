//! Bounded actual-six-shard Q4 screen. No tools, promotion or stock equivalence.
use super::*;
use crate::glm5_next::memory;
use crate::glm5_next_metal::tensor_q5_down;

const PIN: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-16-glm-quality-comparison/stock-q4.json");
const LABELS: [&str; 2] = ["CURRENT", "TENSOR_GATE_UP"];

fn selected<T>(candidate: bool, body: impl FnOnce() -> T) -> T {
    let _dense = tensor_prefill::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(true);
    let _down = tensor_q5_down::Scope::new(true);
    let _kda = memory::KdaHalfInputsScope::new(false);
    let _gate_up = memory::TensorGateUpScope::new(candidate);
    body()
}

fn prefill(
    ctx: &MetalContext,
    s: &mut Glm5NextSession<'_>,
    tokens: &[u32],
    candidate: bool,
) -> Vec<f32> {
    selected(candidate, || {
        let mut commits = Vec::new();
        let logits = s
            .prefill_packed_with_observer(ctx, tokens, &mut || Ok(()), &mut |s, n| {
                commits.push(n);
                check_session(s, commits.iter().sum());
            })
            .unwrap();
        assert_eq!(commits, chunks(tokens.len()));
        logits
    })
}

#[test]
#[ignore = "actual M5, release binary, validation ON, production lease, fresh private report; three docs only"]
fn m5_gate_up_actual_q4_three_document_smoke_profile_plain_abba() {
    environment();
    assert!(
        crate::env_flag::read_default_off("QWEN_GLM53_TENSOR_GATE_UP"),
        "global opt-in owns identical scratch in BOTH arms"
    );
    assert!(!crate::env_flag::read_default_off(
        "QWEN_GLM53_KDA_HALF_INPUTS"
    ));
    for key in ["QWEN_GLM53_DENSE_Q8_WIDE", "QWEN_GLM53_TENSOR_Q5_DOWN"] {
        assert!(
            crate::env_flag::read_default_on(key),
            "baseline control {key} must be ON"
        );
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let mut out = Report::new("GLM53_TENSOR_GATE_UP_REPORT", true);
    let docs = documents();
    assert_eq!((docs.len(), CONTINUATION), (3, 64));
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").unwrap());
    let gguf = GgufFile::open(&path).unwrap();
    let pin: Value = serde_json::from_slice(PIN).unwrap();
    assert_eq!((gguf.shards.len(), gguf.tensors.len()), (6, 1412));
    assert_eq!(pin["shards"].as_array().unwrap().len(), 6);
    for (shard, expected) in gguf.shards.iter().zip(pin["shards"].as_array().unwrap()) {
        assert_eq!(
            shard.path.file_name().unwrap().to_str().unwrap(),
            expected["name"]
        );
        assert_eq!(shard.mmap_len() as u64, expected["size"].as_u64().unwrap());
    }
    let primary_sha = sha256_hex(gguf.primary_shard().mmap_bytes());
    assert_eq!(primary_sha, pin["shards"][0]["sha256"]);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    let config = artifact.config();
    assert_eq!(
        (config.blocks.len(), config.leading_dense_block_count),
        (45, 3)
    );
    for block in 3..45 {
        let gate = if block == 11 {
            GgmlType::Q5_K
        } else {
            GgmlType::Q4_K
        };
        let down = if [11, 12, 44].contains(&block) {
            GgmlType::Q6_K
        } else {
            GgmlType::Q5_K
        };
        for (suffix, dtype, shape) in [
            ("gate", gate, [4096, 2048, 288]),
            ("up", gate, [4096, 2048, 288]),
            ("down", down, [2048, 4096, 288]),
        ] {
            let t = gguf
                .tensors
                .iter()
                .find(|t| t.name == format!("blk.{block}.ffn_{suffix}_exps.weight"))
                .unwrap();
            assert_eq!((t.dtype, t.shape.as_slice()), (dtype, shape.as_slice()));
        }
    }
    let vocab = config.vocab_size as usize;
    assert!(
        docs.iter()
            .all(|d| d.tokens.iter().all(|&id| (id as usize) < vocab))
    );
    let capacity = docs.iter().map(|d| d.prefix + CONTINUATION).max().unwrap();
    assert_eq!(capacity, 4416);
    let terms = cpu_terms(config.vocab_size, false);
    let cpu = terms
        .iter()
        .try_fold(0u64, |n, (_, b)| n.checked_add(*b))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    out.emit(json!({"event":"header","schema":"glm53.tensor_gate_up_smoke.v1",
        "model":path,"artifact_layout":artifact_layout(&gguf),"pin_sha256":sha256_hex(PIN),"primary_sha256":primary_sha,
        "identity_scope":"pinned names/sizes, primary SHA, tensor inventory and retained stamps; payload hashes not reverified",
        "shards":stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"bytes":s.size,"device":s.device,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>(),
        "binary":std::env::current_exe().unwrap(),"binary_sha256":binary_hash(),
        "fixture_sha256":sha256_hex(FROZEN),"probe_sha256":sha256_hex(include_bytes!("tensor_gate_up.rs")),
        "selector_sha256":sha256_hex(include_bytes!("../../tensor_gate_up.rs")),
        "adapter_sha256":sha256_hex(include_bytes!("../../../metal/expert_tensor_gate_up.rs")),
        "packed_sha256":sha256_hex(include_bytes!("../../packed.rs")),
        "memory_sha256":sha256_hex(include_bytes!("../../../glm5_next/memory.rs")),
        "arms":LABELS,"q8_wide":true,"q5_down":true,"kda_execution":false,
        "scope":"only gate/up TLS differs during prefill; global gate/up opt-in allocates one shared 16MiB Half buffer in both arms; ordinary decode outside scopes",
        "documents":docs.iter().map(|d|json!({"path":d.path,"stratum":d.stratum,"prefix":d.prefix,"tokens_sha256":token_hash(&d.tokens)})).collect::<Vec<_>>(),
        "capacity":capacity,"workspace_rows":ROWS,"cpu_terms":terms,"cpu_reserve_bytes":cpu,
        "tensor_tier":format!("{:?}",ctx.tensor_ops_status()),
        "environment":(["MTL_DEBUG_LAYER","QWEN_METAL_LEASE_WAIT","QWEN_EXPERT_TENSOR_TILE","QWEN_GLM53_TENSOR_GATE_UP","QWEN_GLM53_KDA_HALF_INPUTS","QWEN_GLM53_DENSE_Q8_WIDE","QWEN_GLM53_TENSOR_Q5_DOWN","QWEN_MATMAT_Q6_K_N64_MIN_N"].map(|key|(key,std::env::var(key).ok()))),
        "required":{"document_arms":6,"quality_rows":384,"targets_per_arm":192,"census_calls":2,"profile_calls":2,"plain_samples":8},
        "policy":"descriptive finite full-vocab NLL/KL/top1; no quality threshold, promotion or performance verdict; complete event required"}));
    let weights = {
        let _allocation = ctx.begin_allocation_transaction();
        let plan = preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
        assert!(plan.ledger.scratch_options().tensor_gate_up);
        assert!(!plan.ledger.scratch_options().kda_half_inputs);
        let half = memory::packed_normalized_half_specs(
            config,
            ROWS as u64,
            plan.ledger.scratch_options(),
        );
        assert_eq!((half.len(), half[0].bytes()), (1, Some(16 << 20)));
        let admission = crate::metal::evaluate_metal_memory_admission(
            plan.ledger.peak_bytes().checked_add(cpu).unwrap(),
            0,
            ctx.memory_signals(),
            true,
        );
        out.emit(json!({"event":"preload_admission","ledger":plan.ledger.terms(),"cpu_reserve_bytes":cpu,"required_bytes":admission.required_bytes,"admitted":admission.is_admitted(),"reason":admission.reason.as_str()}));
        assert!(
            admission.is_admitted(),
            "full GPU+CPU admission refused: {admission:?}"
        );
        Glm5NextWeights::load(&ctx, &gguf).unwrap()
    };
    assert_eq!(weights.config, *config);
    let mut allocation = None;
    let (
        mut census_calls,
        mut profile_calls,
        mut plain_samples,
        mut quality_rows,
        mut document_arms,
    ) = (0, 0, 0, 0, 0);
    for candidate in [false, true] {
        let label = LABELS[usize::from(candidate)];
        for profiled in [false, true] {
            let mut s = fresh(
                &ctx,
                &weights,
                ROWS,
                cpu,
                label,
                if profiled { "profile" } else { "warm_census" },
                &mut out,
            );
            let owned = (
                s.ledger().session_buffer_bytes(),
                s.observed_allocation_delta(),
            );
            if let Some(expected) = allocation {
                assert_eq!(owned, expected);
            } else {
                allocation = Some(owned);
            }
            assert_plain();
            if profiled {
                let start = Instant::now();
                let (logits, profile) = selected(candidate, || {
                    s.prefill_chunk_stage_profiled(&ctx, &docs[0].tokens[..ROWS])
                        .unwrap()
                });
                let wall = start.elapsed().as_secs_f64() * 1e3;
                check_session(&s, ROWS);
                check_logits(&logits, vocab);
                assert!(wall.is_finite() && wall > 0.0);
                let mut stages = BTreeMap::<&str, f64>::new();
                for span in &profile.spans {
                    assert!(span.gpu_ms.is_finite());
                    *stages.entry(span.stage.as_str()).or_default() += span.gpu_ms;
                }
                out.emit(json!({"event":"profile","arm":label,"wall_ms":wall,"command_gpu_ms":profile.command_gpu_ms,"scaled_span_sum_ms":profile.span_sum_ms,"stages_ms":stages,"warning":"observer perturbs execution; scaled spans are not recoverable ordinary wall time"}));
                profile_calls += 1;
            } else {
                let census = Census::begin();
                let logits = prefill(&ctx, &mut s, &docs[0].tokens[..ROWS], candidate);
                let rows = census.finish();
                check_session(&s, ROWS);
                check_logits(&logits, vocab);
                let heap = rows.capacity() * size_of::<DispatchCensusRow>()
                    + rows
                        .iter()
                        .map(|r| r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity))
                        .sum::<usize>();
                assert!((heap as u64).checked_mul(2).unwrap() <= CENSUS_BYTES);
                let mut kernels = BTreeMap::<&str, usize>::new();
                for row in &rows {
                    *kernels.entry(&row.kernel).or_default() += 1;
                }
                let count = |name: &str| kernels.get(name).copied().unwrap_or(0);
                assert_eq!(count("kernel_moe_down_q5_K_f32_grouped_slots_tensor"), 39);
                assert_eq!(count("kernel_moe_down_q6_K_f32_grouped_slots_generic"), 3);
                for (dtype, pairs) in [("q4_K", 41), ("q5_K", 1)] {
                    for role in ["down", "up_clamped"] {
                        assert_eq!(
                            count(&format!(
                                "kernel_moe_{role}_{dtype}_f16act_grouped_slots_tensor"
                            )),
                            if candidate { pairs } else { 0 }
                        );
                    }
                }
                assert_eq!(
                    kernels
                        .iter()
                        .filter(|(n, _)| n.contains("_f16act_tensor"))
                        .map(|(_, n)| *n)
                        .sum::<usize>(),
                    0,
                    "KDA half execution must be OFF"
                );
                out.emit(json!({"event":"census","arm":label,"dispatches":rows.len(),"kernel_counts":kernels,"heap_upper_bytes":heap*2,"timed":false}));
                census_calls += 1;
            }
            assert_plain();
        }
    }
    for (sample, candidate) in [false, true, true, false, true, false, false, true]
        .into_iter()
        .enumerate()
    {
        let label = LABELS[usize::from(candidate)];
        let mut s = fresh(
            &ctx,
            &weights,
            ROWS,
            cpu,
            label,
            "plain_abba_baab",
            &mut out,
        );
        assert_plain();
        let start = Instant::now();
        let logits = selected(candidate, || {
            s.prefill_packed(&ctx, &docs[0].tokens[..ROWS]).unwrap()
        });
        let wall = start.elapsed().as_secs_f64() * 1e3;
        assert_plain();
        check_session(&s, ROWS);
        check_logits(&logits, vocab);
        assert!(wall.is_finite() && wall > 0.0);
        out.emit(json!({"event":"plain_sample","arm":label,"sample":sample,"block":if sample<4 {"ABBA"} else {"BAAB"},"rows":ROWS,"wall_ms":wall,"scope":"validation ON, normal GPU wait/readback; excludes allocation, checks and JSON; no observer"}));
        plain_samples += 1;
    }
    for d in &docs {
        let mut baseline = Vec::with_capacity(CONTINUATION);
        for candidate in [false, true] {
            let label = LABELS[usize::from(candidate)];
            let mut s = fresh(
                &ctx,
                &weights,
                d.prefix + CONTINUATION,
                cpu,
                label,
                "quality",
                &mut out,
            );
            assert_plain();
            let mut logits = prefill(&ctx, &mut s, &d.tokens[..d.prefix], candidate);
            let mut totals = QualityTotals::default();
            for (i, &target) in d.tokens[d.prefix..].iter().enumerate() {
                check_session(&s, d.prefix + i);
                check_logits(&logits, vocab);
                if !candidate {
                    baseline.push(logits.clone());
                }
                totals.token(
                    Arm {
                        label,
                        dense: candidate,
                        down: false,
                    },
                    d,
                    i,
                    &logits,
                    &baseline[i],
                    None,
                    &mut out,
                );
                quality_rows += 1;
                if i + 1 < CONTINUATION {
                    logits = s.forward(&ctx, target).unwrap();
                }
            }
            assert_eq!(baseline.len(), CONTINUATION);
            check_session(&s, d.prefix + CONTINUATION - 1);
            let mut event = json!({"event":"document_arm","arm":label,"path":d.path,"prefix":d.prefix,"rows":CONTINUATION,"vocab":vocab,"final_position":s.position(),"prefill_chunks":chunks(d.prefix),"all_finite":true});
            totals.finish(&mut event, false);
            out.emit(event);
            document_arms += 1;
        }
    }
    assert_eq!(
        (
            census_calls,
            profile_calls,
            plain_samples,
            document_arms,
            quality_rows
        ),
        (2, 2, 8, 6, 384)
    );
    assert_plain();
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    out.emit(json!({"event":"complete","execution_complete":true,"census_calls":census_calls,"profile_calls":profile_calls,"plain_samples":plain_samples,"document_arms":document_arms,"quality_rows":quality_rows,"targets_per_arm":192,"retained_shard_stamps_unchanged":true,"observers_clear":true,"quality_verdict":null,"performance_verdict":null}));
    out.sync();
}
