//! Compact same-binary down diagnostic, sharing the existing capture/replay helpers.
//! GLM53_GGUF selects UD or Maya; GLM53_EXPERT_DOWN_WIDE_OUT must be a new JSONL path.
//! Optional GLM53_EXPERT_DOWN_WIDE_BLOCK21=1 adds that leaf witness;
//! GLM53_EXPERT_DOWN_WIDE_DEPTH=4096 adds a 512-token suffix after a common prefix.
//! Release test binary, production perf lease, no MTL_DEBUG_LAYER. No promotion gate.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 GLM53_GGUF=/path/to/model.gguf \
//!   GLM53_EXPERT_DOWN_WIDE_OUT=/path/to/new.jsonl /path/to/release/test-binary \
//!   glm5_next_metal::packed::expert_down::wide::expert_down_wide_packet \
//!   --ignored --exact --nocapture --test-threads=1

use super::*;
use crate::metal::{
    dispatch_census_begin, dispatch_census_take, encode_moe_down_f32_grouped_slots_generic,
    encode_moe_down_f32_grouped_slots_generic_capped, encode_moe_down_f32_grouped_slots_n64_split,
    with_grouped_down_wide_override,
};

const STEPS: usize = 3;

fn record(out: &mut File, event: Value) {
    emit(out, event);
    out.sync_data().unwrap();
}

fn summary(out: &mut File, phase: &str, rows: usize, block: Option<usize>, times: &[Vec<f64>; 2]) {
    assert!(times.iter().all(|t| t.len() == 2));
    let mean = |t: &[f64]| t.iter().sum::<f64>() / t.len() as f64;
    let (a, b) = (mean(&times[0]), mean(&times[1]));
    eprintln!(
        "wide down {phase} N{rows} block{block:?}: A={a:.3}ms B={b:.3}ms A/B={:.4}x",
        a / b
    );
    record(
        out,
        json!({"event":"summary","phase":phase,"rows":rows,"block":block,
        "a_mean_wall_ms":a,"b_mean_wall_ms":b,"a_over_b_wall_ratio":a/b,
        "a_rows_per_second":rows as f64*1000.0/a,"b_rows_per_second":rows as f64*1000.0/b,
        "performance_assertion":false}),
    );
}

// The exact runtime policy, including the admitted IQ3_S SmallCounts composition
// at N512 and its capability fallback. This leaf never recomputes gate/up.
fn baseline_down(
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

fn take_down_census() -> Value {
    let mut counts = std::collections::BTreeMap::new();
    for r in dispatch_census_take() {
        if r.kernel.starts_with("kernel_moe_down_") {
            *counts
                .entry((r.kernel, r.grid_width, r.grid_height, r.grid_depth))
                .or_insert(0usize) += 1;
        }
    }
    json!(
        counts
            .into_iter()
            .map(|((kernel, x, y, z), calls)| json!({"kernel":kernel,"grid":[x,y,z],"calls":calls}))
            .collect::<Vec<_>>()
    )
}

fn leaf(ctx: &MetalContext, session: &Glm5NextSession<'_>, c: &Capture, out: &mut File) {
    let p = session.packed.as_ref().unwrap();
    let FfnTensors::Moe(moe) = &session.weights.blocks[c.block].ffn else {
        unreachable!()
    };
    let slots_out = flat(&p.slot_out, H * TOP * c.rows);
    let weighted = flat(&p.routed, H * c.rows);
    let route = p.routes[c.block].as_ref().unwrap();
    let route_weights = flat(&route.weights, TOP * c.rows);
    let ids = read_i32(&flat(&route.ids, TOP * c.rows)).unwrap();
    let counts = read_i32(&c.counts).unwrap();
    let count = |n: &i32| usize::try_from(*n).expect("nonnegative count");
    let max_count = counts.iter().map(count).max().unwrap();
    let inputs = [&c.inner, &c.counts, &c.slots, &route_weights];
    let hashes: Vec<_> = inputs.iter().map(|t| tensor_hash(t)).collect();
    record(
        out,
        json!({"event":"leaf_binding","rows":c.rows,"block":c.block,
        "dtype":format!("{:?}",moe.down_experts.dtype),"input_hashes_inner_counts_slots_weights":hashes,
        "coverage":coverage(&c.counts,&c.slots,c.rows,false,Some(&ids)),
        "active_experts":counts.iter().filter(|&&n|n>0).count(),"max_count":max_count,
        "S32":counts.iter().map(|n|count(n).div_ceil(32)).sum::<usize>(),
        "S64":counts.iter().map(|n|count(n).div_ceil(64)).sum::<usize>(),
        "counts_above32":counts.iter().filter(|&&n|n>32).count(),
        "capped_n32_grid_width":max_count.div_ceil(32),
        "baseline_policy":format!("{:?}",grouped_down_policy(PackedLineage::Fast,c.dtype,K,H,E,c.rows))}),
    );
    drop(ids);
    drop(counts);
    let mut slot_reference = vec![0.125; H * TOP * c.rows];
    let mut weighted_reference = vec![0.125; H * c.rows];
    for capped in [false, true] {
        let phase = if capped {
            "leaf_n32_capped"
        } else {
            "leaf_production_vs_split"
        };
        let mut reference_ready = false;
        let mut times = [Vec::new(), Vec::new()];
        for (label, candidate) in ARMS {
            let warm = label.starts_with("warm");
            fill_f32(&slots_out, f32::NAN);
            fill_f32(&weighted, f32::NAN);
            if warm {
                dispatch_census_begin();
            }
            let (result, wall, gpu) = command(ctx, |enc| {
                if capped {
                    if candidate {
                        encode_moe_down_f32_grouped_slots_generic_capped(
                            ctx,
                            enc,
                            &moe.down_experts,
                            &c.inner,
                            &c.counts,
                            &c.slots,
                            &slots_out,
                            K,
                            H,
                            E,
                            c.rows,
                            max_count,
                        )
                    } else {
                        encode_moe_down_f32_grouped_slots_generic(
                            ctx,
                            enc,
                            &moe.down_experts,
                            &c.inner,
                            &c.counts,
                            &c.slots,
                            &slots_out,
                            K,
                            H,
                            E,
                            c.rows,
                        )
                    }
                } else if candidate {
                    encode_moe_down_f32_grouped_slots_n64_split(
                        ctx,
                        enc,
                        &moe.down_experts,
                        &c.inner,
                        &c.counts,
                        &c.slots,
                        &slots_out,
                        K,
                        H,
                        E,
                        c.rows,
                    )
                } else {
                    baseline_down(ctx, enc, &moe.down_experts, c, &slots_out)
                }
            });
            let census = warm.then(take_down_census);
            record(
                out,
                json!({"event":"leaf_attempt","phase":phase,"rows":c.rows,"block":c.block,"label":label,
            "wall_ms":wall,"gpu_ms":gpu,"census":census,"error":result.as_ref().err().map(ToString::to_string)}),
            );
            result.unwrap();
            if let Some(census) = &census {
                let launches = census.as_array().unwrap();
                if capped {
                    assert_eq!(launches.len(), 1);
                    assert_eq!(
                        launches[0]["grid"][0],
                        json!(if candidate {
                            max_count.div_ceil(32)
                        } else {
                            c.rows.div_ceil(32)
                        })
                    );
                    assert!(!launches[0]["kernel"].as_str().unwrap().ends_with("_n64"));
                } else if candidate {
                    assert_eq!(launches.len(), 2);
                    let wide = launches
                        .iter()
                        .find(|r| r["kernel"].as_str().unwrap().ends_with("_n64"))
                        .expect("candidate N64 launch");
                    assert_eq!(wide["grid"][0], json!(c.rows.div_ceil(64)));
                    let narrow = launches
                        .iter()
                        .find(|r| !r["kernel"].as_str().unwrap().ends_with("_n64"))
                        .unwrap();
                    assert_eq!(narrow["grid"][0], json!(1));
                } else {
                    assert!(
                        launches
                            .iter()
                            .all(|r| !r["kernel"].as_str().unwrap().ends_with("_n64"))
                    );
                }
            }
            // Weighted reduction and all readback/hash work are outside down timing.
            command(ctx, |enc| {
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    ctx,
                    enc,
                    &slots_out,
                    &route_weights,
                    &weighted,
                    H,
                    TOP,
                    c.rows,
                )
            })
            .0
            .unwrap();
            let (slot_metrics, a) = metrics(
                &slots_out,
                reference_ready.then_some(slot_reference.as_slice()),
                None,
            );
            let (weighted_metrics, b) = metrics(
                &weighted,
                reference_ready.then_some(weighted_reference.as_slice()),
                None,
            );
            record(
                out,
                json!({"event":"leaf_output","phase":phase,"rows":c.rows,"block":c.block,"label":label,
            "slots":slot_metrics,"weighted":weighted_metrics,"finite":a&&b}),
            );
            assert!(a && b, "nonfinite or incomplete leaf/weighted output");
            if label == "A1" {
                save_reference(&slots_out, &mut slot_reference);
                save_reference(&weighted, &mut weighted_reference);
                reference_ready = true;
            }
            if !warm {
                times[usize::from(candidate)].push(wall);
            }
        }
        summary(out, phase, c.rows, Some(c.block), &times);
    }
    for (t, hash) in inputs.iter().zip(hashes) {
        assert_eq!(tensor_hash(t), hash, "capture mutated");
    }
}

fn whole(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokens: &[u32],
    prefix: usize,
    rows: usize,
    cpu_bytes: u64,
    out: &mut File,
) {
    let end = prefix + rows;
    assert!(
        OBSERVER.with(|s| !s.borrow().active),
        "whole timing must not capture or observe"
    );
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut times = [Vec::new(), Vec::new()];
    for (label, candidate) in ARMS {
        let warm = label.starts_with("warm");
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            ctx,
            weights,
            end + STEPS,
            rows,
            cpu_bytes,
        )
        .unwrap();
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        if prefix > 0 {
            let logits = with_grouped_down_wide_override(false, || {
                session.prefill_packed(ctx, &tokens[..prefix])
            })
            .unwrap();
            assert!(logits.iter().all(|v| v.is_finite()));
        }
        assert_eq!(session.position(), prefix);
        if warm {
            dispatch_census_begin();
        }
        // No observer, capture copies, profiler, or dispatch census during timed runs.
        let (result, wall_ms) = with_grouped_down_wide_override(candidate, || {
            let start = Instant::now();
            let result = session.prefill_packed(ctx, &tokens[prefix..end]);
            (result, start.elapsed().as_secs_f64() * 1e3)
        });
        let census = warm.then(take_down_census);
        record(
            out,
            json!({"event":"whole_attempt","prefix":prefix,"rows":rows,"label":label,
            "wall_ms":(!warm).then_some(wall_ms),"census":census,
            "error":result.as_ref().err().map(ToString::to_string)}),
        );
        let mut logits = vec![result.unwrap()];
        if let Some(census) = &census {
            let wide_calls: u64 = census
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["kernel"].as_str().unwrap().ends_with("_n64"))
                .map(|r| r["calls"].as_u64().unwrap())
                .sum();
            let moe_layers = weights
                .blocks
                .iter()
                .filter(|b| matches!(&b.ffn, FfnTensors::Moe(_)))
                .count();
            assert_eq!(
                wide_calls,
                if candidate { moe_layers as u64 } else { 0 },
                "whole-model warmup dispatch"
            );
        }
        for &token in &tokens[end..end + STEPS] {
            logits.push(session.forward(ctx, token).unwrap());
        }
        assert_eq!(session.position(), end + STEPS);
        assert!(
            logits
                .iter()
                .all(|r| r.len() == weights.config.vocab_size as usize
                    && r.iter().all(|v| v.is_finite()))
        );
        let comparisons: Vec<_> = logits
            .iter()
            .enumerate()
            .map(|(step, row)| {
                json!({
            "position":end-1+step,"sha256_f32_le":sha(bytemuck::cast_slice(row)),
            "vs_a1":reference.as_ref().map(|r|domain_comparison(&r[step],row))})
            })
            .collect();
        record(
            out,
            json!({"event":"whole_output","prefix":prefix,"rows":rows,"label":label,
            "teacher_forced_tokens":&tokens[end..end+STEPS],"finite":true,"steps":comparisons}),
        );
        if !warm {
            times[usize::from(candidate)].push(wall_ms);
        }
        if label == "A1" {
            reference = Some(logits);
        }
    }
    summary(out, &format!("whole_prefix{prefix}"), rows, None, &times);
}

#[test]
#[ignore = "UD/Maya wide down ABBA: release, GLM53_GGUF, new GLM53_EXPERT_DOWN_WIDE_OUT, no MTL_DEBUG_LAYER; optional BLOCK21=1 and DEPTH=4096"]
fn expert_down_wide_packet() {
    assert!(!cfg!(debug_assertions), "requires release test binary");
    let _lease = perf_lease();
    let path = std::path::PathBuf::from(
        std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF selects UD or Maya"),
    );
    let output =
        std::env::var_os("GLM53_EXPERT_DOWN_WIDE_OUT").expect("new JSONL output path required");
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let blocks = if enabled("GLM53_EXPERT_DOWN_WIDE_BLOCK21") {
        vec![4, 21]
    } else {
        vec![4]
    };
    let deep = std::env::var_os("GLM53_EXPERT_DOWN_WIDE_DEPTH")
        .map(|v| {
            assert_eq!(v.to_str(), Some("4096"));
            4096
        })
        .unwrap_or(0);
    let mut cells = vec![(0usize, 2048usize), (0, 512)];
    if deep > 0 {
        cells.push((deep, 512));
    }
    let needed = cells.iter().map(|(p, n)| p + n + STEPS).max().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    assert!(needed <= artifact.model().config.context_length as usize);
    for &(prefix, rows) in &cells {
        preflight_session(&ctx, &gguf, artifact.model(), prefix + rows + STEPS, rows).unwrap();
    }
    let text = long_qualification_text();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{text}\n\n{text}"), false)
        .unwrap()
        .into_iter()
        .take(needed)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), needed);
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
    for block in &weights.blocks {
        if let FfnTensors::Moe(moe) = &block.ffn {
            assert_eq!(
                expert_mode(
                    Arith {
                        lineage: PackedLineage::Fast,
                        precision: FastPrecision::Half
                    },
                    moe.gate_experts.dtype,
                    moe.up_experts.dtype,
                    moe.down_experts.dtype
                ),
                StageMode::Fast
            );
        }
    }
    for &(prefix, rows) in &cells {
        let ledger =
            Glm5NextSession::plan_memory(&ctx, &weights, prefix + rows + STEPS, rows).unwrap();
        record(
            &mut out,
            json!({"event":"session_memory","prefix":prefix,"rows":rows,
            "capacity":prefix+rows+STEPS,"workspace_rows":rows,"ledger_peak_bytes":ledger.peak_bytes(),
            "session_buffer_bytes":ledger.session_buffer_bytes()}),
        );
    }
    let down_banks: Vec<_> = weights.blocks.iter().enumerate().filter_map(|(block,b)| {
        if let FfnTensors::Moe(moe) = &b.ffn {
            Some(json!({"block":block,"dtype":format!("{:?}",moe.down_experts.dtype),"shape":moe.down_experts.shape}))
        } else { None }
    }).collect();
    let token_bytes = tokens.capacity() as u64 * 4;
    let vocab = u64::from(weights.config.vocab_size);
    let whole_cpu = (2 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + token_bytes;
    record(
        &mut out,
        json!({"event":"packet","schema":"glm53.expert_down_wide.v1","model":path,
        "test_binary":std::env::current_exe().unwrap(),"device":ctx.describe(),"blocks":blocks,"cells":cells,"down_banks":down_banks,
        "arms":{"A":"current production including IQ3_S SmallCounts through N512",
        "B":"test-only composed N32 counts<=32 / N64 counts>=33"},
        "capped_leaf_arms":{"A":"generic N32 incumbent full grid","B":"same generic N32, grid ceil(max_count/32), same actual N stride"},
        "precision":"default Fast Half",
        "order":"warm A B then A1 B1 B2 A2, one resident model, fresh full-model sessions",
        "tokens":tokens,"whole_cpu_reserve_bytes":whole_cpu,
        "whole_cpu_terms":{"trajectory_and_readback_f32_rows":2*(STEPS+1)+1,"f64_kl_vectors":2,"vocab":vocab,"tokens_bytes":token_bytes},
        "timing":"leaf excludes weighted reduction/readback; full prefill excludes allocation/prefix/continuation; no capture or tracing in full timed runs",
        "numeric_promotion_gate":null,"performance_assertion":false}),
    );
    for rows in [2048usize, 512] {
        for &block in &blocks {
            let FfnTensors::Moe(moe) = &weights.blocks[block].ffn else {
                panic!("MoE witness")
            };
            assert!(matches!(
                moe.down_experts.dtype,
                GgmlType::IQ3_S | GgmlType::IQ3_XXS
            ));
            // Named buffers only: two leaf references, bounded readback, endpoint
            // logits, coverage counts/ids/seen/live prefixes, and prompt storage.
            let terms = [
                H * TOP * rows * 4,
                H * rows * 4,
                READ_FLOATS * 4,
                vocab as usize * 4,
                E * 4 * 2,
                E * rows * 4,
                TOP * rows * 4,
                TOP * rows,
                TOP * rows * 4,
            ];
            let cpu_bytes = terms.iter().sum::<usize>() as u64 + token_bytes;
            record(
                &mut out,
                json!({"event":"leaf_memory","rows":rows,"block":block,
                "cpu_term_names":["slot_reference","weighted_reference","read_chunk","endpoint_logits","two_count_vectors",
                    "coverage_slot_ids","route_ids","coverage_seen_upper","coverage_live_prefixes"],
                "cpu_terms_bytes":terms,"token_bytes":token_bytes,"cpu_reserve_bytes":cpu_bytes,
                "extra_gpu_buffers":"inner[K*TOP*N], counts[E], slots[E*N]; priced individually at allocation; outputs reuse session scratch"}),
            );
            let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                rows + STEPS,
                rows,
                cpu_bytes,
            )
            .unwrap();
            let capture = Capture::allocate_for(
                &ctx,
                block,
                rows,
                moe.down_experts.dtype,
                cpu_bytes,
                &mut out,
            );
            let (logits, obs) = with_grouped_down_wide_override(false, || {
                observe(Some(capture.clone()), || {
                    session.prefill_packed(&ctx, &tokens[..rows])
                })
            });
            record(
                &mut out,
                json!({"event":"capture_attempt","rows":rows,"block":block,
                "captures":obs.captures,"commands":obs.completed,
                "error":logits.as_ref().err().map(ToString::to_string)}),
            );
            assert_eq!(obs.captures, 1);
            assert_eq!(
                obs.completed, 1,
                "capture must be one actual packed command"
            );
            let logits = logits.unwrap();
            assert!(logits.iter().all(|v| v.is_finite()));
            record(
                &mut out,
                json!({"event":"capture","rows":rows,"block":block,"iq3_s_layers":obs.iq3_layers,
                "iq3_xxs_layers":obs.iq3_xxs_layers,"iq4_xs_layers":obs.iq4_layers,
                "logits_sha256_f32_le":sha(bytemuck::cast_slice(&logits)),"timing_evidence":false}),
            );
            drop(logits);
            leaf(&ctx, &session, &capture, &mut out);
        }
    }
    // All capture buffers and leaf references are dropped before whole-model ABBA.
    for (prefix, rows) in cells {
        whole(&ctx, &weights, &tokens, prefix, rows, whole_cpu, &mut out);
    }
}
