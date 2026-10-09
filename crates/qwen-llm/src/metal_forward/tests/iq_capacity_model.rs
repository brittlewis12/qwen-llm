//! Whole Saluki native-IQ qualification, not a throughput benchmark.
//! One admitted model, four sequential trajectories; no inflated-IQ baseline load.
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   IQ_CAPACITY_MODEL_OUT=/tmp/saluki-native-model.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   metal_forward::tests::iq_capacity_model::iq_capacity_model_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! IQ_CAPACITY_GGUF optionally overrides the documented artifact. Owner supplies
//! the parent module declaration. Requires IQ2/IQ1 native projections and IQ1_M gather.

use super::super::*;
use crate::metal::{
    DispatchCensusRow, acquire_metal_benchmark_lease, dispatch_census_begin,
    dispatch_census_is_active, dispatch_census_take,
    evaluate_metal_memory_admission_with_cpu_bytes,
};
use crate::metal_dflash::{
    MetalDFlashLayerMajorScratch, PrefillScratchConfig,
    plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
};
use crate::pid_metrics::PidSnapshot;
use crate::qwen_queue2::{QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES, qwen_queue2_session_upper_bytes};
use crate::runtime::{LoadedModelConfig, prefetch_opened_gguf};
use crate::tokenizer::Tokenizer;
use objc2_metal::MTLBuffer;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, io::Write, time::Instant};

const ARTIFACT: &str =
    "/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf";
const CHUNK: u32 = 128;
const CAPACITY: usize = 135;
const CONTINUATIONS: usize = 4;
const NATIVE_COHORTS: [(GgmlType, &str); 4] = [
    (GgmlType::IQ2_XS, "iq2_xs"),
    (GgmlType::IQ2_XXS, "iq2_xxs"),
    (GgmlType::IQ1_S, "iq1_s"),
    (GgmlType::IQ1_M, "iq1_m"),
];
const PROSE: &str = "A coastal town is replacing its old public library. The planning committee has three proposals: renovate the existing building, convert an empty school, or construct a smaller library beside the railway station. Residents want quiet reading rooms, reliable internet access, space for children's activities, and somewhere to meet during winter evenings. The existing building is central but has a leaking roof and narrow staircases. The school has large rooms and a garden, although its heating system is expensive to operate. The station site is easy to reach by bus but has less outdoor space. Explain how the committee should compare these options without assuming that the cheapest initial price gives the best result. Include accessibility, recurring costs, uncertainty about future attendance, and the inconvenience caused during construction. Suggest what information volunteers could collect in one month and what would require a professional survey. Finally, describe a fair way to publish the findings so residents can distinguish measured facts from estimates and express their priorities before the final decision.";
const CODE: &str = include_str!("../../../../qwen-cli/src/serve/backend_glm5_next.rs");

// Both whole trajectories and primitive gather probes use this tokenization.
// Vocabulary loading happens before Metal context creation; no weights load.
fn qualification_streams(gguf: &GgufFile) -> Vec<(&'static str, usize, Vec<i32>, String)> {
    let tokenizer = Tokenizer::from_gguf(gguf).unwrap();
    let texts = [
        ("prose", 129, PROSE.to_owned()),
        (
            "code_review",
            131,
            format!(
                "Review this Rust inference backend for a cache-state or prefill bug. Explain the failure and suggest a focused fix.\n\n{CODE}"
            ),
        ),
    ];
    texts
        .into_iter()
        .map(|(name, n, text)| {
            let tokens: Vec<_> = tokenizer
                .encode(&text, false)
                .unwrap()
                .into_iter()
                .take(n + CONTINUATIONS)
                .collect();
            assert_eq!(tokens.len(), n + CONTINUATIONS, "natural stream too short");
            (name, n, tokens, sha(text.as_bytes()))
        })
        .collect()
}

pub(super) fn qualification_gather_ids(gguf: &GgufFile) -> (Vec<i32>, Vec<Value>) {
    let mut ids = Vec::with_capacity(6);
    let mut origins = Vec::with_capacity(6);
    for (stream, n, tokens, text_hash) in qualification_streams(gguf) {
        let token_hash = sha(bytemuck::cast_slice(&tokens));
        for (role, position) in [
            ("first_prompt", 0),
            ("last_prompt", n - 1),
            ("first_continuation", n),
        ] {
            ids.push(tokens[position]);
            origins.push(json!({"stream":stream,"prompt_rows":n,"role":role,"position_zero_based":position,
                "token_id":tokens[position],"text_sha256":text_hash,"stream_tokens_sha256_i32_le":token_hash,
                "special_token_policy":"encode(false), plain natural text; no extra template"}));
        }
    }
    (ids, origins)
}

fn emit(out: &mut File, event: Value) {
    serde_json::to_writer(&mut *out, &event).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn footprint(ctx: &MetalContext, out: &mut File, phase: &str) {
    let pid = match PidSnapshot::now() {
        Ok(p) => json!({"resident_bytes":p.resident_size,"phys_footprint_bytes":p.phys_footprint,
            "pageins":p.pageins,"disk_read_bytes":p.diskio_bytesread}),
        Err(e) => json!({"error":e.to_string()}),
    };
    emit(
        out,
        json!({"event":"footprint","phase":phase,"metal_current_allocated_bytes":ctx.current_allocated_size(),
        "process":pid,"measurement":"point samples, not peaks; Metal allocation, RSS and physical footprint are distinct"}),
    );
}
fn price(ctx: &MetalContext, bytes: u64) -> u64 {
    if bytes == 0 {
        return 0;
    }
    ctx.price_shared_buffer_upper(bytes)
        .expect("device allocation pricing")
        .priced_upper_bytes
}
fn admit(
    ctx: &MetalContext,
    out: &mut File,
    phase: &str,
    metal: u64,
    cpu: u64,
    reserve: u64,
    details: Value,
) {
    let signals = ctx.memory_signals();
    let a = evaluate_metal_memory_admission_with_cpu_bytes(metal, cpu, reserve, signals, true);
    emit(
        out,
        json!({"event":"admission","phase":phase,"metal_upper_bytes":metal,"cpu_upper_bytes":cpu,
        "dynamic_reserve_bytes":reserve,"components":details,"admitted":a.admitted,"reason":a.reason.as_str(),
        "current_metal_allocated_bytes":signals.current_allocated_bytes,"recommended_working_set_bytes":signals.recommended_max_bytes,
        "process_limit_remaining_bytes":signals.process_limit_remaining_bytes}),
    );
    assert!(
        a.admitted,
        "normal memory admission refused; record flushed"
    );
}
fn request_json(r: &ModelWeightStorageRequest<'_>) -> Value {
    json!({"name":r.desc.name,"source_dtype":format!("{:?}",r.desc.dtype),"shape":r.desc.shape,
        "shard":r.desc.shard_idx,"offset":r.desc.data_offset,"source_bytes":r.desc.n_bytes,
        "kind":format!("{:?}",r.kind),"resident_logical_bytes":r.resident_bytes})
}
fn plan_summary(requests: &[ModelWeightStorageRequest<'_>]) -> Value {
    let mut groups = BTreeMap::<(String, String), (usize, u64, u64)>::new();
    for r in requests {
        let g = groups
            .entry((format!("{:?}", r.desc.dtype), format!("{:?}", r.kind)))
            .or_default();
        g.0 += 1;
        g.1 += r.desc.n_bytes;
        g.2 += r.resident_bytes;
    }
    json!(groups.into_iter().map(|((dtype,kind),(count,source,resident))|
        json!({"source_dtype":dtype,"kind":kind,"requests":count,"source_bytes":source,"resident_logical_bytes":resident})).collect::<Vec<_>>())
}

// Enumerate the actual bindings in exactly the planner's source-request order.
// Derived fused QKV is separate; no assumption that file tensor count == requests.
fn bindings(model: &MetalModel) -> (Vec<&MetalTensor>, Vec<&MetalTensor>) {
    assert_eq!(model.arch.kind, ArchKind::Dense);
    let mut source = vec![&model.token_embd, &model.output_norm, &model.lm_head];
    let mut derived = Vec::new();
    for block in &model.blocks {
        match block {
            MetalBlock::Gdn(g) => {
                assert!(g.ffn_moe.is_none());
                source.extend([
                    &g.attn_norm,
                    &g.post_attn_norm,
                    &g.ffn_gate,
                    &g.ffn_up,
                    &g.ffn_down,
                    &g.in_proj_qkv,
                    &g.in_proj_z,
                    &g.beta_proj,
                    &g.alpha_proj,
                    &g.a_log,
                    &g.dt_bias,
                    &g.conv1d,
                    &g.norm,
                    &g.out_proj,
                ]);
            }
            MetalBlock::Attn(a) => {
                assert!(a.ffn_moe.is_none());
                source.extend([
                    &a.q,
                    &a.k,
                    &a.v,
                    &a.attn_norm,
                    &a.post_attn_norm,
                    &a.ffn_gate,
                    &a.ffn_up,
                    &a.ffn_down,
                    &a.o,
                    &a.q_norm,
                    &a.k_norm,
                ]);
                if let Some(qkv) = &a.qkv_fused {
                    derived.push(qkv);
                }
            }
        }
    }
    (source, derived)
}
fn parse_ledger(line: &str) -> BTreeMap<String, Vec<u64>> {
    assert!(line.starts_with("[metal-load-ledger] "));
    line.split_whitespace()
        .skip(1)
        .map(|field| {
            let (name, values) = field.split_once('=').expect("ledger field");
            (
                name.to_owned(),
                values
                    .split('/')
                    .map(|n| n.parse().expect("ledger number"))
                    .collect(),
            )
        })
        .collect()
}
fn audit(
    model: &MetalModel,
    requests: &[ModelWeightStorageRequest<'_>],
    lines: &[String],
    expected_derived: &[u64],
    out: &mut File,
) {
    let (source, derived) = bindings(model);
    assert_eq!(source.len(), requests.len());
    let mut actual = BTreeMap::<String, (usize, u64)>::new();
    for (t, r) in source.iter().zip(requests) {
        let dtype = match r.kind {
            ModelWeightStorageKind::Direct => r.desc.dtype,
            ModelWeightStorageKind::ConvertedF32 => GgmlType::F32,
            ModelWeightStorageKind::ConvertedF16 => GgmlType::F16,
        };
        emit(
            out,
            json!({"event":"realized_binding","name":r.desc.name,"source_dtype":format!("{:?}",r.desc.dtype),
            "planned_kind":format!("{:?}",r.kind),"actual_dtype":format!("{:?}",t.dtype),"shape":t.shape,
            "logical_bytes":t.n_bytes(),"buffer_bytes":t.buffer.length(),"offset":t.offset,"provenance":format!("{:?}",t.provenance())}),
        );
        assert_eq!(t.dtype, dtype, "realized dtype: {}", r.desc.name);
        assert_eq!(t.shape, r.desc.shape, "realized shape: {}", r.desc.name);
        assert_eq!(t.n_bytes(), r.resident_bytes);
        let a = actual.entry(format!("{:?}", t.dtype)).or_default();
        a.0 += 1;
        a.1 += t.n_bytes();
    }
    assert_eq!(
        derived.iter().map(|t| t.n_bytes()).collect::<Vec<_>>(),
        expected_derived
    );
    let raw: Vec<_> = lines
        .iter()
        .filter(|l| l.starts_with("[metal-load-ledger] "))
        .collect();
    assert_eq!(raw.len(), 1, "one realization ledger");
    let ledger = parse_ledger(raw[0]);
    let direct: Vec<_> = requests
        .iter()
        .filter(|r| r.kind == ModelWeightStorageKind::Direct)
        .collect();
    let converted: Vec<_> = requests
        .iter()
        .filter(|r| r.kind != ModelWeightStorageKind::Direct)
        .collect();
    assert_eq!(
        ledger["source"],
        vec![
            requests.len() as u64,
            requests.iter().map(|r| r.desc.n_bytes).sum()
        ]
    );
    assert_eq!(
        ledger["direct_copy"],
        vec![
            direct.len() as u64,
            direct.iter().map(|r| r.desc.n_bytes).sum()
        ]
    );
    for key in ["direct_view", "direct_alias", "tail_fallback"] {
        assert_eq!(ledger[key], vec![0, 0]);
    }
    assert_eq!(
        ledger["converted"],
        vec![
            converted.len() as u64,
            converted.iter().map(|r| r.desc.n_bytes).sum(),
            converted.iter().map(|r| r.resident_bytes).sum()
        ]
    );
    assert_eq!(
        ledger["derived"],
        vec![
            derived.len() as u64,
            derived.iter().map(|t| t.n_bytes()).sum()
        ]
    );
    emit(
        out,
        json!({"event":"realized_summary","ledger":ledger,"actual_source_bindings_by_dtype":actual,
        "planned_by_source_dtype_and_kind":plan_summary(requests),"derived_logical_bytes":expected_derived,
        "reconciled":true,"note":"request counts include any source aliases; buffer_bytes per binding must not be summed as unique physical allocation"}),
    );
}

fn census<R>(f: impl FnOnce() -> R) -> (R, Vec<DispatchCensusRow>) {
    assert!(!dispatch_census_is_active(), "nested/external census");
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            let _ = dispatch_census_take();
        }
    }
    dispatch_census_begin();
    let _reset = Reset;
    let result = f();
    (result, dispatch_census_take())
}
// IQ1 S/M share projection PSOs. Only an observed dtype tag can distinguish
// their dispatches; source inventory or launch geometry alone is not evidence.
fn census_dtype(kernel: &str, tag: Option<&str>) -> Option<&'static str> {
    for (_, name) in NATIVE_COHORTS {
        if kernel.contains(&format!("{name}_")) {
            return Some(name);
        }
    }
    if kernel.contains("mat_vec_iq1_f32") || kernel.contains("mat_mat_iq1_f32") {
        return match tag {
            Some("iq1_s") => Some("iq1_s"),
            Some("iq1_m") => Some("iq1_m"),
            _ => None,
        };
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn witness(
    model: &MetalModel,
    out: &mut File,
    stream: &str,
    phase: &str,
    rows: Vec<DispatchCensusRow>,
    require_matrix: bool,
    require_concurrent: bool,
) {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut kernels = BTreeMap::<(String, Option<String>, bool), usize>::new();
    for r in &rows {
        *kernels
            .entry((r.kernel.clone(), r.tag.clone(), r.encoder_concurrent))
            .or_default() += 1;
        if let Some(dtype) = census_dtype(&r.kernel, r.tag.as_deref()) {
            *counts.entry(dtype.into()).or_default() += 1;
            if r.kernel.contains("mat_mat") {
                *counts.entry(format!("{dtype}_matrix")).or_default() += 1;
            }
            if r.kernel.contains("mat_vec") {
                *counts.entry(format!("{dtype}_vector")).or_default() += 1;
            }
            if r.kernel.contains("get_rows") {
                *counts.entry(format!("{dtype}_gather")).or_default() += 1;
            }
            if r.encoder_concurrent {
                *counts.entry(format!("{dtype}_concurrent")).or_default() += 1;
            }
        } else if r.kernel.contains("mat_vec_iq1_f32") || r.kernel.contains("mat_mat_iq1_f32") {
            *counts
                .entry("iq1_projection_without_dtype_tag".into())
                .or_default() += 1;
        }
    }
    // Only GDN front projections run in concurrent encoders in ordinary dense
    // decode. FFN-only cohorts must not acquire an artificial concurrency gate.
    let expected_concurrent: Vec<_> = NATIVE_COHORTS
        .iter()
        .filter(|(dtype, _)| {
            require_concurrent
                && model.blocks.iter().any(|block| match block {
                    MetalBlock::Gdn(g) => {
                        [&g.in_proj_qkv, &g.in_proj_z, &g.beta_proj, &g.alpha_proj]
                            .iter()
                            .any(|w| w.dtype == *dtype)
                    }
                    MetalBlock::Attn(_) => false,
                })
        })
        .map(|(_, name)| *name)
        .collect();
    let concurrent = rows.iter().filter(|r| r.encoder_concurrent).count();
    let topology:Vec<_>=kernels.into_iter().map(|((kernel,tag,concurrent),dispatches)|json!({"kernel":kernel,"tag":tag,"concurrent":concurrent,"dispatches":dispatches})).collect();
    emit(
        out,
        json!({"event":"kernel_witness","stream":stream,"phase":phase,"counts":counts,
        "dispatches":rows.len(),"concurrent_dispatches":concurrent,"kernels":topology,
        "expected_concurrent_dtypes_from_realized_gdn_front":expected_concurrent,"benchmark":false,
        "iq1_dtype_attribution":"observed iq1_s/iq1_m census tag for shared projection kernels; gather has a distinct IQ1_M kernel"}),
    );
    assert_eq!(
        counts
            .get("iq1_projection_without_dtype_tag")
            .copied()
            .unwrap_or(0),
        0,
        "shared IQ1 projection kernels require observed iq1_s/iq1_m census tags; raw witness flushed"
    );
    for (_, dtype) in NATIVE_COHORTS {
        assert!(
            counts.get(dtype).copied().unwrap_or(0) > 0,
            "missing native {dtype} witness"
        );
        let operation = if require_matrix { "matrix" } else { "vector" };
        assert!(
            counts
                .get(&format!("{dtype}_{operation}"))
                .copied()
                .unwrap_or(0)
                > 0,
            "missing native {dtype} {operation} witness"
        );
    }
    assert!(
        counts.get("iq1_m_gather").copied().unwrap_or(0) > 0,
        "missing IQ1_M embedding gather witness"
    );
    for dtype in expected_concurrent {
        assert!(
            counts
                .get(&format!("{dtype}_concurrent"))
                .copied()
                .unwrap_or(0)
                > 0,
            "missing concurrent native {dtype} GDN front witness"
        );
    }
    if require_concurrent {
        assert!(concurrent > 0, "ordinary decode concurrency missing");
    }
}
fn top(a: &[f32]) -> usize {
    assert!(!a.is_empty());
    (1..a.len()).fold(0, |best, i| if a[i] > a[best] { i } else { best })
}
fn log_z(a: &[f32]) -> f64 {
    let max = f64::from(a[top(a)]);
    max + a
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}
fn comparison(reference: &[f32], actual: &[f32]) -> Value {
    assert_eq!(reference.len(), actual.len());
    assert!(reference.iter().chain(actual).all(|v| v.is_finite()));
    let (za, zb) = (log_z(reference), log_z(actual));
    let (mut max, mut err, mut norm, mut ab, mut ba) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in reference.iter().zip(actual) {
        let (a, b) = (f64::from(a), f64::from(b));
        let d = b - a;
        max = max.max(d.abs());
        err += d * d;
        norm += a * a;
        let (la, lb) = (a - za, b - zb);
        ab += la.exp() * (la - lb);
        ba += lb.exp() * (lb - la);
    }
    assert!(ab.is_finite() && ba.is_finite());
    let (a, b) = (top(reference), top(actual));
    json!({"max_abs":max,"relative_l2":(err/norm.max(1e-30)).sqrt(),"kl_reference_packed":ab,"kl_packed_reference":ba,
        "reference_top1":a,"packed_top1":b,"reference_choice_regret":f64::from(reference[a])-f64::from(reference[b]),
        "packed_choice_regret":f64::from(actual[b])-f64::from(actual[a])})
}
fn output(
    out: &mut File,
    stream: &str,
    lineage: &str,
    position: usize,
    step: usize,
    logits: &[f32],
    reference: Option<&[f32]>,
) {
    let nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
    emit(
        out,
        json!({"event":"output","stream":stream,"lineage":lineage,"position":position,"teacher_forced_steps":step,
        "elements":logits.len(),"nonfinite":nonfinite,"sha256_f32_le":sha(bytemuck::cast_slice(logits))}),
    );
    assert_eq!(nonfinite, 0, "nonfinite logits; record flushed");
    assert!(!logits.is_empty());
    if let Some(r) = reference {
        emit(
            out,
            json!({"event":"comparison","stream":stream,"position":position,"teacher_forced_steps":step,
            "reference":"serial_tokens_native_gemv","actual":"ordinary_packed_native",
            "metrics":comparison(r,logits),"decision":"unscored_no_bit_or_quality_gate"}),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn trajectory(
    ctx: &MetalContext,
    model: &MetalModel,
    out: &mut File,
    stream: &str,
    tokens: &[i32],
    n: usize,
    reference: Option<&[Vec<f32>]>,
) -> Vec<Vec<f32>> {
    let packed = reference.is_some();
    let lineage = if packed {
        "ordinary_packed_native"
    } else {
        "serial_tokens_native_gemv"
    };
    let mut saved = Vec::new();
    {
        let allocation = ctx.begin_allocation_transaction();
        let session_bytes = qwen_queue2_session_upper_bytes(ctx, model, CAPACITY).unwrap();
        let plan = packed.then(|| {
            plan_prefill_scratch_with_matrix_max_pos_configured(
                model,
                CHUNK,
                CAPACITY,
                PrefillScratchConfig::default(),
            )
            .unwrap()
        });
        let scratch_bytes = plan
            .as_ref()
            .map_or(0, |p| p.priced_upper_bound(|n| Ok(price(ctx, n))).unwrap());
        let ids_bytes = if packed {
            price(ctx, u64::from(CHUNK) * 4)
        } else {
            0
        };
        let cpu_bytes = u64::from(model.arch.vocab_size) * 6 * 4 + (16 << 20);
        admit(
            ctx,
            out,
            &format!("{stream}/{lineage}"),
            session_bytes + scratch_bytes + ids_bytes,
            cpu_bytes,
            QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES,
            json!({"session_pricer":"qwen_queue2_session_upper_bytes","session_bytes":session_bytes,
                "prefill_scratch_upper_bytes":scratch_bytes,"prefill_ids_upper_bytes":ids_bytes,
                "capacity":CAPACITY,"chunk_rows":CHUNK,"cpu_logits_slots":6,"cpu_bookkeeping_bytes":16<<20,
                "session_count":1,"scratch_deferred_allocations_included":true}),
        );
        let mut session = MetalSession::fresh(ctx, model, CAPACITY).unwrap();
        let mut scratch = plan
            .map(|p| MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, model, p).unwrap());
        drop(allocation);
        footprint(ctx, out, &format!("{stream}/{lineage}/allocated"));
        let forward = MetalForward::new(ctx, model);
        let start = Instant::now();
        let endpoint = if let Some(s) = scratch.as_mut() {
            let (result, rows) = census(|| {
                prefill_tokens_with_multi_hidden(
                    &forward,
                    &tokens[..n],
                    0,
                    &mut session,
                    s,
                    &[],
                    None,
                )
            });
            emit(
                out,
                json!({"event":"trajectory_attempt","stream":stream,"lineage":lineage,"rows":n,
                "wall_ms":start.elapsed().as_secs_f64()*1e3,"benchmark":false,"census_observer":true,
                "error":result.as_ref().err().map(ToString::to_string)}),
            );
            let logits = result.expect("ordinary packed prefill failed; raw attempt flushed");
            witness(model, out, stream, "packed_prefill", rows, true, false);
            logits
        } else {
            for (i, &token) in tokens[..n - 1].iter().enumerate() {
                if i == 0 {
                    let (result, rows) =
                        census(|| forward.single_token_no_tail(token, i as u32, &mut session));
                    emit(
                        out,
                        json!({"event":"reference_first_token","stream":stream,"error":result.as_ref().err().map(ToString::to_string)}),
                    );
                    result.expect("ordinary GEMV reference failed");
                    witness(model, out, stream, "reference_decode", rows, false, true);
                } else {
                    let result = forward.single_token_no_tail(token, i as u32, &mut session);
                    if let Err(error) = &result {
                        emit(
                            out,
                            json!({"event":"reference_prefix_error","stream":stream,
                            "position":i,"token":token,"error":error.to_string()}),
                        );
                    }
                    result.expect("GEMV reference prefix; error record flushed");
                }
            }
            let result = forward.single_token(tokens[n - 1], (n - 1) as u32, &mut session);
            emit(
                out,
                json!({"event":"trajectory_attempt","stream":stream,"lineage":lineage,"rows":n,
                "wall_ms":start.elapsed().as_secs_f64()*1e3,"benchmark":false,"census_observer":"first token only",
                "error":result.as_ref().err().map(ToString::to_string)}),
            );
            result.expect("GEMV reference endpoint failed")
        };
        assert!(session.kv_n_pos.iter().all(|&p| p == n));
        output(
            out,
            stream,
            lineage,
            n,
            0,
            &endpoint,
            reference.map(|r| r[0].as_slice()),
        );
        if !packed {
            saved.push(endpoint);
        } else {
            drop(endpoint);
        }
        for step in 1..=CONTINUATIONS {
            let position = n + step - 1;
            let token = tokens[position];
            let (result, rows) = if step == 1 {
                census(|| forward.single_token(token, position as u32, &mut session))
            } else {
                (
                    forward.single_token(token, position as u32, &mut session),
                    Vec::new(),
                )
            };
            emit(
                out,
                json!({"event":"continuation","stream":stream,"lineage":lineage,"step":step,"token":token,"position":position,
                "error":result.as_ref().err().map(ToString::to_string)}),
            );
            let logits = result.expect("ordinary continuation failed; raw record flushed");
            if step == 1 {
                witness(
                    model,
                    out,
                    stream,
                    &format!("{lineage}/continuation"),
                    rows,
                    false,
                    true,
                );
            }
            assert!(session.kv_n_pos.iter().all(|&p| p == position + 1));
            output(
                out,
                stream,
                lineage,
                position + 1,
                step,
                &logits,
                reference.map(|r| r[step].as_slice()),
            );
            if !packed {
                saved.push(logits);
            }
        }
        footprint(ctx, out, &format!("{stream}/{lineage}/completed"));
    }
    footprint(ctx, out, &format!("{stream}/{lineage}/session_dropped"));
    saved
}

#[test]
fn iq_capacity_model_cpu_metrics_and_ledger() {
    assert_eq!(census_dtype("kernel_mat_vec_iq1_f32", None), None);
    assert_eq!(
        census_dtype("kernel_mat_vec_iq1_f32", Some("iq1_s")),
        Some("iq1_s")
    );
    assert_eq!(
        census_dtype("kernel_mat_mat_iq1_f32_mma", Some("iq1_m")),
        Some("iq1_m")
    );
    assert_eq!(
        census_dtype("kernel_get_rows_iq1_m_f32", None),
        Some("iq1_m")
    );
    assert_eq!(
        census_dtype("kernel_mat_vec_iq2_xxs_f32", None),
        Some("iq2_xxs")
    );
    let a = [-2.0, 1.0, 0.0];
    let same = comparison(&a, &a);
    assert_eq!(same["max_abs"], json!(0.0));
    assert_eq!(same["kl_reference_packed"], json!(0.0));
    let shifted = comparison(&a, &[-1.0, 2.0, 1.0]);
    assert!(shifted["kl_reference_packed"].as_f64().unwrap().abs() < 1e-12);
    let ledger = parse_ledger("[metal-load-ledger] source=2/100 converted=1/20/80 derived=0/0");
    assert_eq!(ledger["converted"], vec![1, 20, 80]);
}

#[test]
#[ignore = "whole Saluki native IQ model; normal lease/admission; release; new IQ_CAPACITY_MODEL_OUT"]
fn iq_capacity_model_packet() {
    assert!(!cfg!(debug_assertions), "release qualification required");
    assert!(std::env::var_os("MTL_DEBUG_LAYER").is_none());
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("IQ_CAPACITY_MODEL_OUT").expect("IQ_CAPACITY_MODEL_OUT required"))
        .expect("new output file required");
    let path = std::env::var_os("IQ_CAPACITY_GGUF").unwrap_or_else(|| ARTIFACT.into());
    let gguf = GgufFile::open(&path).unwrap();
    let bound = Model::from_gguf(&gguf).unwrap();
    assert_eq!(bound.arch.kind, ArchKind::Dense);
    let streams = qualification_streams(&gguf);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let _lease = acquire_metal_benchmark_lease().expect("production adaptive-wait lease");
    let ctx = MetalContext::new().unwrap();
    assert!(
        concurrent_gdn_dense_decode_enabled(),
        "keep ordinary decode concurrency enabled"
    );
    // Reject diagnostic splitting/no-op knobs instead of changing process policy.
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        assert!(
            !name.starts_with("QWEN_PREFILL_TRACE_")
                && !name.starts_with("QWEN_PREFILL_NOOP_")
                && !name.starts_with("QWEN_DECODE_GDN_NOOP_")
                && !name.starts_with("QWEN_PHASE_GDN_"),
            "remove diagnostic override {name}"
        );
    }
    let shards:Vec<_>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,"bytes":s.size,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(name, _)| {
            [
                "QWEN_PREFILL_",
                "QWEN_DECODE_",
                "QWEN_GGUF_",
                "QWEN_DENSE_GDN_",
                "QWEN_METAL_LEASE_",
                "QWEN_MATVEC_",
                "QWEN_MATMAT_",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
                || name == "QWEN_KV_Q8"
                || name == "QWEN_NATIVE_QUANT_EMBED"
        })
        .collect();
    let source_binding: Value = json!({"packet":sha(include_bytes!("iq_capacity_model.rs")),"loader":sha(include_bytes!("../mod.rs")),
            "residency":sha(include_bytes!("../residency.rs")),"dispatch":sha(include_bytes!("../dispatch.rs")),
            "embedding_policy":sha(include_bytes!("../support.rs")),"gather_dispatch":sha(include_bytes!("../../metal/elementwise.rs")),
            "token":sha(include_bytes!("../token.rs")),"gdn":sha(include_bytes!("../gdn.rs")),
            "prefill":sha(include_bytes!("../../metal_dflash.rs")),"codec":sha(include_bytes!("../../codec.rs")),
            "xxs":sha(include_bytes!("../../metal/iq2_xxs.rs")),"xxs_metal":sha(include_bytes!("../../../../../kernels/iq2_xxs.metal")),
            "xxs_grid":sha(include_bytes!("../../../../../kernels/iq2_xxs_grid.metalh")),
            "iq1_dispatch":sha(include_bytes!("../../metal/iq1.rs")),
            "iq1_kernel_decoder":sha(include_bytes!("../../../../../kernels/iq1.metal")),
            "iq1_grid":sha(include_bytes!("../../../../../kernels/iq1_grid.metalh")),
            "gemv_dispatch":sha(include_bytes!("../../metal/mat_vec.rs")),"gemm_dispatch":sha(include_bytes!("../../metal/mat_mat.rs")),
            "xs_gemm":sha(include_bytes!("../../../../../kernels/mat_mat_iq2_xs.metal")),"code_corpus":sha(CODE.as_bytes()),
            "metallib":sha(crate::KERNELS_METALLIB)});
    emit(
        &mut out,
        json!({"event":"header","schema":"iq_capacity.model.v2","device":ctx.describe(),"shards":shards,
        "cases":[{"stream":"prose","rows":129},{"stream":"code_review","rows":131}],"chunk_rows":CHUNK,"capacity":CAPACITY,
        "continuations":CONTINUATIONS,"trajectories":4,"weight_loads":1,
         "environment":environment,"gpu_timing_collected":false,
         "environment_absence":"unlisted variables matching captured QWEN_PREFILL/DECODE/GGUF/DENSE_GDN/METAL_LEASE/MATVEC/MATMAT prefixes and QWEN_KV_Q8/QWEN_NATIVE_QUANT_EMBED are unset",
        "policy":"same native weights; reference serial tokens through ordinary GEMV, packed ordinary prefill; default decode concurrency remains enabled",
        "scope":"numerical and capacity qualification; not benchmark, not inflated-baseline A/B, not language-quality proof",
        "source_binding":source_binding}),
    );
    for (name, n, ids, text_hash) in &streams {
        emit(
            &mut out,
            json!({"event":"stream","name":name,"prompt_rows":n,"token_ids":ids,"text_sha256":text_hash,
            "tokens_sha256_i32_le":sha(bytemuck::cast_slice(ids)),"special_token_policy":"encode(false), plain natural text; no extra template"}),
        );
    }
    footprint(&ctx, &mut out, "before_prepare");
    let prepared = MetalModel::prepare_load_with_options(
        &ctx,
        &gguf,
        &bound,
        MetalModelLoadOptions::default(),
    )
    .unwrap();
    let copied = prepared.storage.no_copy_mode == GgufNoCopyMode::Disabled
        && prepared.storage.owned_mode == GgufOwnedArenaMode::Disabled
        && !prepared.storage.parallel_mode.is_forced()
        && !matches!(&prepared.auto, PreparedAutoSelection::Selected(_))
        && !matches!(
            &prepared.auto_retained,
            PreparedAutoRetainedSelection::Selected(_)
        );
    emit(
        &mut out,
        json!({"event":"prepared_policy","copied_topology_supported":copied,"no_copy":format!("{:?}",prepared.storage.no_copy_mode),
        "owned":format!("{:?}",prepared.storage.owned_mode),"parallel":format!("{:?}",prepared.storage.parallel_mode),
        "native_embedding_selection":prepared.choices.embedding_selection.label(),
        "embedding_source_dtype":format!("{:?}",bound.token_embd.dtype),"tied_embeddings":bound.tied_embeddings,
        "prefetch_advice":format!("{:?}",prepared.prefetch_advice()),"plan":prepared.expected.iter().map(request_json).collect::<Vec<_>>(),
        "plan_summary":plan_summary(&prepared.expected)}),
    );
    assert!(
        copied,
        "prepared topology is not canonical copied; refuse before prefetch/load"
    );
    assert_eq!(
        prepared.prefetch_advice(),
        MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
    );
    let requests = prepared.expected.clone();
    assert_eq!(
        bound.token_embd.dtype,
        GgmlType::IQ1_M,
        "released Saluki embedding cohort"
    );
    assert!(
        !bound.tied_embeddings,
        "released untied embedding/head policy"
    );
    assert_eq!(
        prepared.choices.embedding_selection,
        NativeQuantEmbeddingSelection::AutoPromoted,
        "qualification requires the default native IQ1_M embedding selection; policy flushed"
    );
    for (dtype, _) in NATIVE_COHORTS {
        let cohort: Vec<_> = gguf.tensors.iter().filter(|t| t.dtype == dtype).collect();
        assert!(!cohort.is_empty());
        let native_bytes: u64 = cohort.iter().map(|d| d.n_bytes).sum();
        let f32_bytes: u64 = cohort.iter().map(|d| d.n_elements() * 4).sum();
        emit(
            &mut out,
            json!({"event":"native_cohort_plan","dtype":format!("{dtype:?}"),
            "file_tensors":cohort.len(),"native_logical_bytes":native_bytes,
            "hypothetical_converted_f32_logical_bytes":f32_bytes,
            "avoided_inflation_logical_bytes":f32_bytes-native_bytes,
            "comparison_kind":"descriptor arithmetic, no inflated baseline allocated"}),
        );
        for d in cohort {
            assert!(
                requests
                    .iter()
                    .any(|r| r.desc.name == d.name && r.kind == ModelWeightStorageKind::Direct),
                "{} not planned native; no inflated fallback load",
                d.name
            );
        }
    }
    let remaining_conversions: Vec<_> = requests
        .iter()
        .filter(|r| r.kind != ModelWeightStorageKind::Direct)
        .map(request_json)
        .collect();
    emit(
        &mut out,
        json!({"event":"conversion_plan","remaining_converted_requests":remaining_conversions,
        "all_conversions_eliminated":remaining_conversions.is_empty(),
        "expectation":"released artifact formerly converted only IQ1_S/IQ1_M; source F32 norms are direct, not conversions"}),
    );
    assert!(
        remaining_conversions.is_empty(),
        "released-artifact zero-conversion target not met; plan flushed"
    );
    // Match the actual optional derived-QKV loader predicate. Price its GPU
    // destination and its two simultaneous compressed CPU concatenation copies.
    let derived: Vec<u64> = bound
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Attn(a)
                if prepared.choices.fused_qkv_g8
                    && a.q.dtype == GgmlType::Q8_0
                    && a.k.dtype == a.q.dtype
                    && a.v.dtype == a.q.dtype
                    && a.q.shape.len() == 2
                    && a.k.shape.len() == 2
                    && a.v.shape.len() == 2
                    && a.q.shape[0] == a.k.shape[0]
                    && a.q.shape[0] == a.v.shape[0] =>
            {
                Some(a.q.n_bytes + a.k.n_bytes + a.v.n_bytes)
            }
            _ => None,
        })
        .collect();
    let config = LoadedModelConfig::default();
    let workers = if config.prefetch_workers == 0 {
        crate::prefetch::DEFAULT_WORKERS
    } else {
        config.prefetch_workers
    };
    let chunk = if config.prefetch_chunk_bytes == 0 {
        crate::prefetch::DEFAULT_CHUNK_BYTES
    } else {
        config.prefetch_chunk_bytes
    };
    let source_pages: u64 = stamps.iter().map(|s| s.size).sum();
    let metal_bytes: u64 = requests
        .iter()
        .map(|r| price(&ctx, r.resident_bytes))
        .chain(derived.iter().map(|&n| price(&ctx, n)))
        .sum();
    let concat_bytes = derived.iter().max().copied().unwrap_or(0) * 2;
    let f16_conversion_cpu = requests
        .iter()
        .filter(|r| r.kind == ModelWeightStorageKind::ConvertedF16)
        .map(|r| r.desc.n_elements() * 6) // F32 decode plus simultaneous F16 staging.
        .max()
        .unwrap_or(0);
    let codec_staging = 256 * 1024u64; // canonical codec calls at most 65,536 elements.
    let headroom = config.effective_prefetch_min_headroom_bytes();
    let cpu_bytes = source_pages
        + (workers * chunk) as u64
        + concat_bytes
        + f16_conversion_cpu
        + codec_staging
        + headroom;
    let allocation = ctx.begin_allocation_transaction();
    admit(
        &ctx,
        &mut out,
        "before_prefetch_and_load",
        metal_bytes,
        cpu_bytes,
        0,
        json!({
        "priced_copied_destinations_and_derived":metal_bytes,"source_mmap_residency_upper_bytes":source_pages,
        "prefetch_worker_bytes":workers*chunk,"derived_concat_cpu_upper_bytes":concat_bytes,
        "converted_f16_cpu_upper_bytes":f16_conversion_cpu,"codec_staging_upper_bytes":codec_staging,
        "canonical_prefetch_headroom_bytes":headroom,"full_cpu_f32_model_bytes":0,
        "note":"whole source-file residency conservatively includes already resident pages; converted F32 fills Metal storage directly"}),
    );
    // Bind metadata via retained descriptors in bounded reads, not a full model hash.
    let mut bytes = vec![0u8; 1 << 20];
    for (index, shard) in gguf.shards.iter().enumerate() {
        let mut h = Sha256::new();
        let mut offset = 0u64;
        while offset < shard.tensor_data_start {
            let n = (shard.tensor_data_start - offset).min(bytes.len() as u64) as usize;
            gguf.read_shard_exact_at(index, offset, &mut bytes[..n])
                .unwrap();
            h.update(&bytes[..n]);
            offset += n as u64;
        }
        emit(
            &mut out,
            json!({"event":"header_binding","shard":index,"bytes_including_padding":offset,"sha256":format!("{:x}",h.finalize()),"full_payload_hash":false}),
        );
    }
    drop(bytes);
    let prefetch = prefetch_opened_gguf(&gguf, &config);
    emit(
        &mut out,
        json!({"event":"prefetch","policy":format!("{:?}",prefetch.policy),"action":format!("{:?}",prefetch.action),
        "wall_ms":prefetch.total_wall.as_secs_f64()*1e3,"bytes_returned":prefetch.bytes_returned_total(),"details":format!("{:?}",prefetch.shards)}),
    );
    footprint(&ctx, &mut out, "after_prefetch_before_load");
    // Prefetch warms the file cache, not necessarily this process's mmap pages.
    // Keep the full source-residency reservation through conversion; only the
    // joined prefetch workers' temporary buffers can be removed here.
    admit(
        &ctx,
        &mut out,
        "before_load_recheck",
        metal_bytes,
        source_pages + concat_bytes + f16_conversion_cpu + codec_staging + headroom,
        0,
        json!({"source_mmap_residency_upper_bytes":source_pages,
            "derived_concat_cpu_upper_bytes":concat_bytes,"converted_f16_cpu_upper_bytes":f16_conversion_cpu,
            "codec_staging_upper_bytes":codec_staging,"canonical_prefetch_headroom_bytes":headroom,
            "prefetch_worker_bytes":0,"note":"source reservation retained conservatively; file cache is not process mmap residency"}),
    );
    let start = Instant::now();
    let (result, lines) = capture_metal_load_lines(|| MetalModel::load_prepared(prepared));
    emit(
        &mut out,
        json!({"event":"load_result","wall_ms":start.elapsed().as_secs_f64()*1e3,"raw_loader_lines":lines,
        "error":result.as_ref().err().map(ToString::to_string)}),
    );
    let metal = result.expect("normal prepared load failed; raw record flushed");
    drop(allocation);
    footprint(&ctx, &mut out, "loaded");
    audit(&metal, &requests, &lines, &derived, &mut out);
    for (name, n, tokens, _) in streams {
        eprintln!("IQ model {name} N{n}: native serial-token reference then ordinary packed");
        let reference = trajectory(&ctx, &metal, &mut out, name, &tokens, n, None);
        assert_eq!(reference.len(), CONTINUATIONS + 1);
        trajectory(&ctx, &metal, &mut out, name, &tokens, n, Some(&reference));
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    footprint(&ctx, &mut out, "qualification_complete_weights_resident");
    emit(
        &mut out,
        json!({"event":"complete","schema":"iq_capacity.model.v2","trajectories":4,"comparisons":10,
        "decision":"unscored_capacity_and_numerical_qualification_no_promotion"}),
    );
}
