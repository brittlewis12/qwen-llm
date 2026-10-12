//! Captured IQ2_S gate/up replay plus same-binary whole-model ABBA.
//! GLM53_GGUF selects Maya/UD; GLM53_EXPERT_GATE_UP_OUT must be a new JSONL path.
//! No promotion threshold; the candidate remains test-only research.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 GLM53_GGUF=/path/to/model.gguf \
//!   GLM53_EXPERT_GATE_UP_OUT=/path/to/new.jsonl /path/to/release/test-binary \
//!   glm5_next_metal::packed::expert_down::gate_up::expert_gate_up_packet \
//!   --ignored --exact --nocapture --test-threads=1

use super::*;
use crate::metal::{
    dispatch_census_begin, dispatch_census_take, encode_moe_down_f32_grouped_slots_generic,
    encode_moe_swiglu_clamped_f32_grouped_slots_split, with_grouped_down_wide_override,
    with_grouped_gate_up_m32_override,
};

const BLOCK: usize = 4;
const STEPS: usize = 3;

fn record(out: &mut File, event: Value) {
    emit(out, event);
    out.sync_data().unwrap();
}

fn arm<T>(candidate: bool, body: impl FnOnce() -> T) -> T {
    // Disable the independent wide-DOWN experiment, retaining real SmallCounts.
    with_production(|| {
        with_grouped_down_wide_override(false, || {
            with_grouped_gate_up_m32_override(candidate, body)
        })
    })
    .0
}

fn mean(times: &[f64]) -> f64 {
    times.iter().sum::<f64>() / times.len() as f64
}

// Only warmups trace. Down launch signatures must remain identical between arms.
fn census() -> (Value, Value) {
    let mut gates = std::collections::BTreeMap::new();
    let mut downs = std::collections::BTreeMap::new();
    for r in dispatch_census_take() {
        let table = if r.kernel.starts_with("kernel_moe_swiglu_clamped_") {
            &mut gates
        } else if r.kernel.starts_with("kernel_moe_down_") {
            &mut downs
        } else {
            continue;
        };
        *table
            .entry((r.kernel, r.grid_width, r.grid_height, r.grid_depth))
            .or_insert(0usize) += 1;
    }
    let encode = |table: std::collections::BTreeMap<(String, u64, u64, u64), usize>| {
        json!(table.into_iter()
        .map(|((kernel,x,y,z),calls)|json!({"kernel":kernel,"grid":[x,y,z],"calls":calls})).collect::<Vec<_>>())
    };
    (encode(gates), encode(downs))
}

fn check_gate_census(gates: &Value, candidate: bool) {
    let entries = gates.as_array().unwrap();
    assert!(
        !entries.is_empty(),
        "gate/up warmup must exercise actual grouped dispatch"
    );
    assert_eq!(
        entries
            .iter()
            .any(|r| r["kernel"].as_str().unwrap().contains("_m32")),
        candidate
    );
}

fn production_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    c: &Capture,
    output: &MetalTensor,
) -> std::result::Result<(), MetalError> {
    if grouped_down_policy(PackedLineage::Fast, weight.dtype, K, H, E, c.rows)
        == GroupedDownPolicy::Iq3SSmallCounts
    {
        if let Some(prepared) = crate::metal::iq3_s_down_retile::prepare_small_counts(
            ctx, enc, weight, &c.inner, &c.counts, &c.slots, output, K, H, E, c.rows,
        )? {
            return prepared.encode();
        }
    }
    encode_moe_down_f32_grouped_slots_generic(
        ctx, enc, weight, &c.inner, &c.counts, &c.slots, output, K, H, E, c.rows,
    )
}

fn leaf(ctx: &MetalContext, session: &Glm5NextSession<'_>, c: &Capture, out: &mut File) {
    let p = session.packed.as_ref().unwrap();
    let FfnTensors::Moe(moe) = &session.weights.blocks[BLOCK].ffn else {
        unreachable!()
    };
    assert_eq!(
        (moe.gate_experts.dtype, moe.up_experts.dtype),
        (GgmlType::IQ2_S, GgmlType::IQ2_S)
    );
    let input = c
        .normalized_input
        .as_ref()
        .expect("normalized block input requested");
    let output = flat(&p.inner, K * TOP * c.rows);
    let down_out = flat(&p.slot_out, H * TOP * c.rows);
    let counts = read_i32(&c.counts).unwrap();
    let routes = read_i32(&flat(&p.routes[BLOCK].as_ref().unwrap().ids, TOP * c.rows)).unwrap();
    let inputs = [input, &c.inner, &c.counts, &c.slots];
    let hashes: Vec<_> = inputs.iter().map(|t| tensor_hash(t)).collect();
    record(
        out,
        json!({"event":"leaf_binding","rows":c.rows,"block":BLOCK,
        "gate_dtype":"IQ2_S","up_dtype":"IQ2_S","gate_shape":moe.gate_experts.shape,
        "down_dtype":format!("{:?}",c.dtype),"clamp":session.weights.config.swiglu_clamp,
        "input_hashes_normalized_inner_counts_slots":hashes,
        "coverage":coverage(&c.counts,&c.slots,c.rows,false,Some(&routes)),
        "active_experts":counts.iter().filter(|&&n|n>0).count(),
        "S16":counts.iter().map(|&n|usize::try_from(n).unwrap().div_ceil(16)).sum::<usize>(),
        "S32":counts.iter().map(|&n|usize::try_from(n).unwrap().div_ceil(32)).sum::<usize>(),
        "counts_above16":counts.iter().filter(|&&n|n>16).count(),
        "down_policy":format!("{:?}",grouped_down_policy(PackedLineage::Fast,c.dtype,K,H,E,c.rows))}),
    );
    drop(routes);
    drop(counts);
    let mut reference = vec![0.125; K * TOP * c.rows];
    save_reference(&c.inner, &mut reference);
    let mut gate_wall = [Vec::new(), Vec::new()];
    let mut gate_gpu = [Vec::new(), Vec::new()];
    for (label, candidate) in ARMS {
        let warm = label.starts_with("warm");
        fill_f32(&output, f32::NAN);
        if warm {
            dispatch_census_begin();
        }
        let (result, wall, gpu) = arm(candidate, || {
            command(ctx, |enc| {
                encode_moe_swiglu_clamped_f32_grouped_slots_split(
                    ctx,
                    enc,
                    &moe.gate_experts,
                    &moe.up_experts,
                    input,
                    &c.counts,
                    &c.slots,
                    &output,
                    H,
                    K,
                    E,
                    TOP,
                    c.rows,
                    session.weights.config.swiglu_clamp,
                    17,
                )
            })
        });
        let trace = warm.then(census);
        record(
            out,
            json!({"event":"gate_up_attempt","rows":c.rows,"label":label,
            "wall_ms":wall,"gpu_ms":gpu,"census":trace,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        result.unwrap();
        if let Some((gates, downs)) = &trace {
            check_gate_census(gates, candidate);
            let launches = gates.as_array().unwrap();
            assert_eq!(launches.len(), 2);
            let wide_name = if candidate {
                crate::metal::moe_swiglu_m32n32_pipeline_name(GgmlType::IQ2_S).unwrap()
            } else {
                "kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_n32"
            };
            for launch in launches {
                let wide = launch["kernel"].as_str().unwrap() == wide_name;
                let m = if wide && candidate { 32 } else { 64 };
                let n = if wide { 32 } else { 16 };
                assert_eq!(
                    launch["grid"],
                    json!([c.rows.div_ceil(n), K.div_ceil(m), E])
                );
                assert_eq!(launch["calls"], json!(1));
            }
            assert!(
                launches
                    .iter()
                    .any(|r| r["kernel"].as_str().unwrap() == wide_name)
            );
            assert!(downs.as_array().unwrap().is_empty());
        }
        let (mut values, finite) = metrics(&output, Some(reference.as_slice()), None);
        let comparison = values
            .as_object_mut()
            .unwrap()
            .remove("comparison_to_A1")
            .unwrap();
        values["comparison_to_captured_production"] = comparison;
        record(
            out,
            json!({"event":"gate_up_output","rows":c.rows,"label":label,"metrics":values,
            "hash_equals_capture":values["sha256_f32_le"]==json!(hashes[1]),"finite":finite}),
        );
        assert!(finite, "nonfinite or incomplete gate/up output");
        if !warm {
            gate_wall[usize::from(candidate)].push(wall);
            gate_gpu[usize::from(candidate)].extend(gpu);
        }
    }
    // Same captured production inner for every down sample: gate/up variants do
    // not feed this cost reference. There is no experimental down substitution.
    let mut down_wall = Vec::new();
    let mut down_gpu = Vec::new();
    let mut down_signature = None;
    for (label, _) in ARMS {
        let warm = label.starts_with("warm");
        fill_f32(&down_out, f32::NAN);
        if warm {
            dispatch_census_begin();
        }
        let (result, wall, gpu) = arm(false, || {
            command(ctx, |enc| {
                production_down(ctx, enc, &moe.down_experts, c, &down_out)
            })
        });
        let trace = warm.then(census);
        record(
            out,
            json!({"event":"production_down_attempt","rows":c.rows,"label":label,
            "wall_ms":wall,"gpu_ms":gpu,"census":trace,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        result.unwrap();
        if let Some((gates, downs)) = trace {
            assert!(gates.as_array().unwrap().is_empty());
            if let Some(a) = &down_signature {
                assert_eq!(a, &downs);
            } else {
                down_signature = Some(downs);
            }
        }
        let (values, finite) = metrics(&down_out, None, None);
        record(
            out,
            json!({"event":"production_down_output","rows":c.rows,"label":label,"metrics":values,"finite":finite}),
        );
        assert!(finite, "nonfinite or incomplete production down");
        if !warm {
            down_wall.push(wall);
            down_gpu.extend(gpu);
        }
    }
    let a = mean(&gate_wall[0]);
    let b = mean(&gate_wall[1]);
    let d = mean(&down_wall);
    let gpu_valid = gate_gpu.iter().all(|v| v.len() == 2) && down_gpu.len() == 4;
    record(
        out,
        json!({"event":"leaf_summary","rows":c.rows,"gate_a_mean_wall_ms":a,"gate_b_mean_wall_ms":b,
        "down_mean_wall_ms":d,"gate_a_over_b_wall_ratio":a/b,"gate_a_over_down_wall_ratio":a/d,
        "gate_b_over_down_wall_ratio":b/d,"gpu_summary":gpu_valid.then(||json!({
            "gate_a_ms":mean(&gate_gpu[0]),"gate_b_ms":mean(&gate_gpu[1]),"down_ms":mean(&down_gpu),
            "gate_a_over_down":mean(&gate_gpu[0])/mean(&down_gpu),"gate_b_over_down":mean(&gate_gpu[1])/mean(&down_gpu)}))}),
    );
    eprintln!(
        "gate/up N{}: A={a:.3}ms B={b:.3}ms unchanged down={d:.3}ms A/B={:.4}x",
        c.rows,
        a / b
    );
    for (t, hash) in inputs.iter().zip(hashes) {
        assert_eq!(tensor_hash(t), hash, "capture mutated");
    }
}

fn whole(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokens: &[u32],
    rows: usize,
    cpu: u64,
    out: &mut File,
) {
    assert!(OBSERVER.with(|s| !s.borrow().active));
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut times = [Vec::new(), Vec::new()];
    let mut down_signature = None;
    for (label, candidate) in ARMS {
        let warm = label.starts_with("warm");
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            ctx,
            weights,
            rows + STEPS,
            rows,
            cpu,
        )
        .unwrap();
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        if warm {
            dispatch_census_begin();
        }
        let (result, ms) = arm(candidate, || {
            let start = Instant::now();
            let result = session.prefill_packed(ctx, &tokens[..rows]);
            (result, start.elapsed().as_secs_f64() * 1e3)
        });
        let trace = warm.then(census);
        record(
            out,
            json!({"event":"whole_attempt","rows":rows,"label":label,
            "wall_ms":(!warm).then_some(ms),"census":trace,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let mut logits = vec![result.unwrap()];
        if let Some((gates, downs)) = trace {
            check_gate_census(&gates, candidate);
            if let Some(a) = &down_signature {
                assert_eq!(a, &downs, "whole down selection changed");
            } else {
                down_signature = Some(downs);
            }
        }
        for &token in &tokens[rows..rows + STEPS] {
            logits.push(session.forward(ctx, token).unwrap());
        }
        assert_eq!(session.position(), rows + STEPS);
        assert!(
            logits
                .iter()
                .all(|r| r.len() == weights.config.vocab_size as usize
                    && r.iter().all(|v| v.is_finite()))
        );
        let comparisons: Vec<_>=logits.iter().enumerate().map(|(step,row)|json!({"position":rows-1+step,
            "sha256_f32_le":sha(bytemuck::cast_slice(row)),"vs_a1":reference.as_ref().map(|a|domain_comparison(&a[step],row))})).collect();
        record(
            out,
            json!({"event":"whole_output","rows":rows,"label":label,"steps":comparisons,
            "teacher_forced_tokens":&tokens[rows..rows+STEPS],"finite":true}),
        );
        if !warm {
            times[usize::from(candidate)].push(ms);
        }
        if label == "A1" {
            reference = Some(logits);
        }
    }
    let a = mean(&times[0]);
    let b = mean(&times[1]);
    record(
        out,
        json!({"event":"whole_summary","rows":rows,"a_mean_ms":a,"b_mean_ms":b,"a_over_b_wall_ratio":a/b,
        "a_tokens_per_second":rows as f64*1000.0/a,"b_tokens_per_second":rows as f64*1000.0/b}),
    );
    eprintln!(
        "whole gate/up N{rows}: A={a:.2}ms B={b:.2}ms A/B={:.4}x",
        a / b
    );
}

#[test]
#[ignore = "gate/up M32 research: release, GLM53_GGUF Maya/UD, new GLM53_EXPERT_GATE_UP_OUT, production lease, no MTL_DEBUG_LAYER"]
fn expert_gate_up_packet() {
    assert!(!cfg!(debug_assertions), "requires release test binary");
    let _lease = perf_lease();
    let path = std::path::PathBuf::from(std::env::var_os("GLM53_GGUF").expect("Maya or UD GGUF"));
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("GLM53_EXPERT_GATE_UP_OUT").expect("new JSONL path"))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    assert!(2048 + STEPS <= artifact.model().config.context_length as usize);
    for rows in [2048, 512] {
        preflight_session(&ctx, &gguf, artifact.model(), rows + STEPS, rows).unwrap();
    }
    let text = long_qualification_text();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{text}\n\n{text}"), false)
        .unwrap()
        .into_iter()
        .take(2048 + STEPS)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), 2048 + STEPS);
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    assert_eq!(
        (
            weights.config.hidden_size as usize,
            weights.config.expert_ffn_size as usize,
            weights.config.expert_count as usize,
            weights.config.expert_used_count as usize,
            weights.blocks.len()
        ),
        (H, K, E, TOP, 45)
    );
    assert_eq!(super::super::router_prefill::requested_override(), None);
    let vocab = u64::from(weights.config.vocab_size);
    let token_bytes = tokens.capacity() as u64 * 4;
    let whole_cpu = (2 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + token_bytes;
    record(
        &mut out,
        json!({"event":"packet","schema":"glm53.expert_gate_up.v1","model":path,
        "test_binary":std::env::current_exe().unwrap(),"device":ctx.describe(),"tokens":tokens,"block":BLOCK,
        "widths":[2048,512],"precision":"default Fast Half","clamp":weights.config.swiglu_clamp,
        "arms":{"A":"production N16<=16 / M64N32>=17","B":"test-only N16<=16 / M32N32>=17"},
        "down":"production in every arm, including SmallCounts at N512; leaf down uses identical captured production inner",
        "timing":"warm A B then A1 B1 B2 A2; leaf excludes readback; whole excludes allocation and 3 common teacher-forced steps; warmup-only census, no timed captures",
        "whole_cpu_reserve_bytes":whole_cpu,"whole_cpu_terms":{"f32_rows":2*(STEPS+1)+1,"f64_kl_vectors":2,"vocab":vocab,"token_bytes":token_bytes},
        "numeric_promotion_gate":null,"performance_assertion":false}),
    );
    for rows in [2048, 512] {
        let FfnTensors::Moe(moe) = &weights.blocks[BLOCK].ffn else {
            panic!("MoE witness")
        };
        assert_eq!(
            (moe.gate_experts.dtype, moe.up_experts.dtype),
            (GgmlType::IQ2_S, GgmlType::IQ2_S)
        );
        let terms = [
            K * TOP * rows * 4,
            READ_FLOATS * 4,
            vocab as usize * 4,
            E * 4 * 2,
            E * rows * 4,
            TOP * rows * 4,
            TOP * rows,
            TOP * rows * 4,
        ];
        let cpu = terms.iter().sum::<usize>() as u64 + token_bytes;
        let ledger = Glm5NextSession::plan_memory(&ctx, &weights, rows + STEPS, rows).unwrap();
        record(
            &mut out,
            json!({"event":"memory","rows":rows,"capacity":rows+STEPS,"workspace_rows":rows,
            "ledger_peak_bytes":ledger.peak_bytes(),"session_buffer_bytes":ledger.session_buffer_bytes(),
            "capture_cpu_terms":["inner_reference","read_chunk","endpoint_logits","two_count_vectors","coverage_slots","route_ids","coverage_seen_upper","coverage_live_prefixes"],
            "capture_cpu_term_bytes":terms,"token_bytes":token_bytes,"capture_cpu_reserve_bytes":cpu,
            "capture_gpu":"inner[K*TOP*N], counts[E], slots[E*N], normalized_input[H*N]; individually priced at admission; replay outputs reuse session scratch"}),
        );
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            rows + STEPS,
            rows,
            cpu,
        )
        .unwrap();
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        let capture = Capture::allocate_with_input(
            &ctx,
            BLOCK,
            rows,
            moe.down_experts.dtype,
            cpu,
            true,
            &mut out,
        );
        let (result, obs) = arm(false, || {
            observe(Some(capture.clone()), || {
                session.prefill_packed(&ctx, &tokens[..rows])
            })
        });
        record(
            &mut out,
            json!({"event":"capture","rows":rows,"captures":obs.captures,"commands":obs.completed,
            "error":result.as_ref().err().map(ToString::to_string),"timing_evidence":false}),
        );
        let logits = result.unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!((obs.captures, obs.completed), (1, 1));
        drop(logits);
        leaf(&ctx, &session, &capture, &mut out);
    }
    // No live capture allocations/references during full-model timing.
    for rows in [2048, 512] {
        whole(&ctx, &weights, &tokens, rows, whole_cpu, &mut out);
    }
}
