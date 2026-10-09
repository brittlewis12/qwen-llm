//! Same-model native IQ2_XS whole-prefill diagnostic: forced Scalar vs MMA or production.
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   IQ2_XS_MMA_MODEL_OUT=/tmp/saluki-iq2-xs-mma-model.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   metal_forward::tests::iq2_xs_mma_model::iq2_xs_mma_model_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! Optional: IQ2_XS_MMA_GGUF, IQ2_XS_MMA_MODEL_ROUNDS (default 2),
//! IQ2_XS_MMA_MODEL_CHUNK (default 1024; actual scratch width is min(chunk,N)).
//! IQ2_XS_MMA_MODEL_WIDTHS (comma-separated, default 128,4096),
//! IQ2_XS_MMA_MODEL_CANDIDATE (mma by default, or production for actual None scope).
//! Each of two corpora runs fresh sessions: warm A/B, then ABBA.
//! Post-promotion check: WIDTHS=129, CHUNK=128, ROUNDS=1, CANDIDATE=production
//! (all four variables use the IQ2_XS_MMA_MODEL_ prefix).
//! Four common teacher-forced tokens follow every prefill, outside its timing.

use super::super::*;
use super::iq_capacity_model::{
    admit, census, comparison, emit, footprint, load_copied_native, price, sha,
};
use crate::metal::{
    Iq2XsMatMatVariant, acquire_metal_benchmark_lease, dispatch_census_is_active,
    with_iq2_xs_matmat_variant,
};
use crate::metal_dflash::{
    MetalDFlashLayerMajorScratch, PrefillScratchConfig,
    plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden_profiled,
};
use crate::qwen_queue2::{QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES, qwen_queue2_session_upper_bytes};
use crate::tokenizer::Tokenizer;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, time::Instant};

const ARTIFACT: &str =
    "/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf";
const WIDTHS: [usize; 2] = [128, 4096];
const NEXT: usize = 4;
const PROSE: &str = include_str!("../../../../../docs/GLM53-FLASH-PLAN.md");
const CODE: &str = include_str!("../../../../qwen-cli/src/serve/backend_glm5_next.rs");
const RENDER: &str = include_str!("../../../../qwen-cli/src/serve/render_glm5_next.rs");
const ABBA: [(&str, bool); 4] = [("A1", false), ("B1", true), ("B2", true), ("A2", false)];

fn positive_env(name: &str, default: usize) -> usize {
    let n = match std::env::var(name) {
        Ok(s) => s
            .parse::<usize>()
            .expect("positive integer environment value"),
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => panic!("{name}: {e}"),
    };
    assert!(n > 0, "{name} must be positive");
    n
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Mma,
    Production,
}
impl Candidate {
    fn parse(value: &str) -> Self {
        match value {
            "mma" => Self::Mma,
            "production" => Self::Production,
            _ => panic!("IQ2_XS_MMA_MODEL_CANDIDATE must be mma or production"),
        }
    }
    fn scope(self) -> Option<Iq2XsMatMatVariant> {
        match self {
            Self::Mma => Some(Iq2XsMatMatVariant::Mma),
            Self::Production => None,
        }
    }
    fn arm(self) -> &'static str {
        match self {
            Self::Mma => "mma_native",
            Self::Production => "production_native",
        }
    }
}
fn parse_widths(value: &str) -> Vec<usize> {
    let mut widths = Vec::new();
    for field in value.split(',') {
        let n = field
            .trim()
            .parse::<usize>()
            .expect("comma-separated positive widths");
        assert!(n > 0, "width must be positive");
        let capacity = n
            .checked_add(NEXT)
            .expect("width plus continuations overflows");
        u32::try_from(capacity).expect("width plus continuations fits token positions");
        assert!(!widths.contains(&n), "duplicate width {n}");
        widths.push(n);
    }
    widths
}
fn streams(gguf: &GgufFile, max_width: usize) -> Vec<(&'static str, Vec<i32>, String)> {
    let tokenizer = Tokenizer::from_gguf(gguf).unwrap();
    let texts = [
        (
            "technical_prose",
            format!(
                "Review this engineering plan and explain the main performance tradeoffs.\n\n{PROSE}"
            ),
        ),
        (
            "code_review",
            format!(
                "Review this Rust inference backend for a prefill or cache-state bug. Explain the likely failure and suggest a focused fix.\n\n{CODE}\n\nRelated prompt-rendering implementation:\n\n{RENDER}"
            ),
        ),
    ];
    texts
        .into_iter()
        .map(|(name, text)| {
            let ids: Vec<_> = tokenizer
                .encode(&text, false)
                .unwrap()
                .into_iter()
                .take(max_width + NEXT)
                .collect();
            assert_eq!(
                ids.len(),
                max_width + NEXT,
                "natural corpus too short; no padding or repeated templates"
            );
            (name, ids, sha(text.as_bytes()))
        })
        .collect()
}
// Derive the expected substitution count from realized projection bindings,
// excluding norms, embedding and head. No hard-coded 39-dispatch assumption.
fn xs_projection_inventory(model: &MetalModel) -> Vec<Value> {
    let mut inventory = Vec::new();
    for (layer, block) in model.blocks.iter().enumerate() {
        let projections: Vec<(&str, &MetalTensor)> = match block {
            MetalBlock::Gdn(g) => vec![
                ("gate", &g.ffn_gate),
                ("up", &g.ffn_up),
                ("down", &g.ffn_down),
                ("qkv", &g.in_proj_qkv),
                ("z", &g.in_proj_z),
                ("beta", &g.beta_proj),
                ("alpha", &g.alpha_proj),
                ("out", &g.out_proj),
            ],
            MetalBlock::Attn(a) => vec![
                ("gate", &a.ffn_gate),
                ("up", &a.ffn_up),
                ("down", &a.ffn_down),
                ("q", &a.q),
                ("k", &a.k),
                ("v", &a.v),
                ("out", &a.o),
            ],
        };
        for (role, w) in projections {
            if w.dtype == GgmlType::IQ2_XS {
                inventory.push(
                    json!({"layer":layer,"role":role,"shape":w.shape,"native_bytes":w.n_bytes()}),
                );
            }
        }
    }
    inventory
}
fn expected_mma(n: usize, chunk: usize, projections: usize) -> usize {
    (0..n)
        .step_by(chunk)
        .filter(|&start| (n - start).min(chunk) > 1)
        .count()
        * projections
}

fn log_output(out: &mut File, mut identity: Value, logits: &[f32], reference: Option<&[f32]>) {
    let nonfinite = logits.iter().filter(|x| !x.is_finite()).count();
    let fields = json!({"event":"output","elements":logits.len(),"nonfinite":nonfinite,
        "sha256_f32_le":sha(bytemuck::cast_slice(logits))});
    identity
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    emit(out, identity.clone());
    assert!(
        !logits.is_empty() && nonfinite == 0,
        "invalid logits; raw record flushed"
    );
    if let Some(reference) = reference {
        let mut metrics = comparison(reference, logits);
        for (old, new) in [
            ("kl_reference_packed", "kl_reference_actual"),
            ("kl_packed_reference", "kl_actual_reference"),
            ("packed_top1", "actual_top1"),
            ("packed_choice_regret", "actual_choice_regret"),
        ] {
            let value = metrics.as_object_mut().unwrap().remove(old).unwrap();
            metrics.as_object_mut().unwrap().insert(new.into(), value);
        }
        let record = identity.as_object_mut().unwrap();
        record.insert("event".into(), json!("comparison"));
        record.insert("reference".into(), json!("A1_scalar_same_round_or_warm_A"));
        record.insert("metrics".into(), metrics);
        record.insert(
            "decision".into(),
            json!("unscored_no_bit_or_quality_threshold_gate"),
        );
        emit(out, identity);
    }
}
#[derive(Clone, Copy)]
struct Timing {
    gpu_ms: f64,
    wall_ms: f64,
    substitutions: usize,
}
struct Run {
    timing: Option<Timing>,
    saved: Vec<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn trajectory(
    ctx: &MetalContext,
    model: &MetalModel,
    out: &mut File,
    stream: &str,
    tokens: &[i32],
    n: usize,
    configured_chunk: usize,
    projections: usize,
    round: usize,
    label: &str,
    candidate: bool,
    candidate_mode: Candidate,
    warm: bool,
    reference: Option<&[Vec<f32>]>,
) -> Run {
    let chunk = configured_chunk.min(n);
    let capacity = n + NEXT;
    assert!(tokens.len() >= capacity);
    let variant = if candidate {
        candidate_mode.scope()
    } else {
        Some(Iq2XsMatMatVariant::Scalar)
    };
    let expected = if candidate {
        expected_mma(n, chunk, projections)
    } else {
        0
    };
    let keep_reference = reference.is_none() && !candidate;
    let identity = |step: usize| {
        json!({"stream":stream,"N":n,"round":round,"label":label,
        "arm":if candidate {candidate_mode.arm()} else {"scalar_native"},
        "selector_scope":format!("{variant:?}"),"teacher_forced_steps":step,"position":n+step})
    };
    let allocation = ctx.begin_allocation_transaction();
    let session_bytes = qwen_queue2_session_upper_bytes(ctx, model, capacity).unwrap();
    let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        model,
        u32::try_from(chunk).unwrap(),
        capacity,
        PrefillScratchConfig::default(),
    )
    .unwrap();
    let scratch_bytes = plan.priced_upper_bound(|n| Ok(price(ctx, n))).unwrap();
    let ids_bytes = price(ctx, (chunk * 4) as u64);
    let cpu_bytes = u64::from(model.arch.vocab_size) * 6 * 4 + (16 << 20);
    admit(
        ctx,
        out,
        &format!("{stream}/N{n}/{round}/{label}"),
        session_bytes + scratch_bytes + ids_bytes,
        cpu_bytes,
        QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES,
        json!({"session_bytes":session_bytes,"prefill_scratch_upper_bytes":scratch_bytes,
        "per_call_ids_bytes":ids_bytes,"capacity":capacity,"configured_chunk":configured_chunk,"actual_chunk":chunk,
        "cpu_logits_slots":6,"cpu_metadata_bytes":16<<20,"scratch_deferred_allocations_included":true,
        "model_and_source_mmap":"already loaded; represented in current allocation/process signals"}),
    );
    let mut session = MetalSession::fresh(ctx, model, capacity).unwrap();
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, model, plan).unwrap();
    drop(allocation);
    assert!(session.kv_n_pos.iter().all(|&p| p == 0));
    assert!(
        !dispatch_census_is_active(),
        "external census must not enter timed path"
    );
    let forward = MetalForward::new(ctx, model);
    let mut run = || {
        with_iq2_xs_matmat_variant(variant, || {
            prefill_tokens_with_multi_hidden_profiled(
                &forward,
                &tokens[..n],
                0,
                &mut session,
                &mut scratch,
                &[],
                None,
            )
        })
    };
    let started = Instant::now();
    let ((result, substitutions), census_rows) = if warm {
        census(run)
    } else {
        (run(), Vec::new())
    };
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let gpu_ms = result.as_ref().ok().map(|(_, gpu)| *gpu);
    let gpu_valid = gpu_ms.is_some_and(|gpu| gpu.is_finite() && gpu > 0.0);
    let singleton_chunks = if chunk == 1 {
        n
    } else {
        usize::from(n % chunk == 1)
    };
    let details = json!({"event":"whole_attempt","measured":!warm,"census_observer":warm,
        "wall_ms":(!warm).then_some(wall_ms),"aggregate_gpu_ms_raw":gpu_ms,
        "aggregate_gpu_valid":gpu_valid,"gpu_ms":if !warm&&gpu_valid {gpu_ms} else {None},
        "individual_timestamp_validity":"not exposed by existing ordinary prefill API",
        "chunks":n.div_ceil(chunk),"chunk_rows":chunk,"mma_substitutions":substitutions,"expected_mma_substitutions":expected,
        "singleton_chunks":singleton_chunks,"expected_singleton_mma_substitutions":0,
        "error":result.as_ref().err().map(ToString::to_string)});
    let mut event = identity(0);
    event
        .as_object_mut()
        .unwrap()
        .extend(details.as_object().unwrap().clone());
    emit(out, event);
    let (endpoint, gpu_ms) = result.expect("ordinary prefill failed; raw attempt flushed");
    assert!(
        gpu_valid,
        "invalid aggregate GPU timing; raw attempt flushed"
    );
    assert_eq!(
        substitutions, expected,
        "IQ2_XS substitutions differ from realized projection inventory"
    );
    assert!(session.kv_n_pos.iter().all(|&p| p == n));
    if warm {
        let mut kernels = BTreeMap::<(String, bool), usize>::new();
        for row in &census_rows {
            *kernels
                .entry((row.kernel.clone(), row.encoder_concurrent))
                .or_default() += 1;
        }
        let scalar = census_rows
            .iter()
            .filter(|r| r.kernel == "kernel_mat_mat_iq2_xs_f32")
            .count();
        let mma = census_rows
            .iter()
            .filter(|r| r.kernel == "kernel_mat_mat_iq2_xs_f32_mma")
            .count();
        let gemv = census_rows
            .iter()
            .filter(|r| {
                matches!(
                    r.kernel.as_str(),
                    "kernel_mat_vec_iq2_xs_f32" | "kernel_mat_vec_iq2_xs_f32_fast"
                )
            })
            .count();
        let topology:Vec<_>=kernels.into_iter().map(|((kernel,concurrent),count)|json!({"kernel":kernel,"concurrent":concurrent,"count":count})).collect();
        let mut event = identity(0);
        event.as_object_mut().unwrap().extend(
            json!({"event":"warm_kernel_witness","kernels":topology,
            "iq2_xs_scalar_gemm":scalar,"iq2_xs_mma_gemm":mma,"mma_substitutions":substitutions,
            "iq2_xs_gemv":gemv,"singleton_chunks":singleton_chunks,
            "expected_singleton_gemv":singleton_chunks*projections,
            "singleton_attribution":"aggregate GEMV census and total MMA count; no command splitting"})
            .as_object()
            .unwrap()
            .clone(),
        );
        emit(out, event);
        assert_eq!(mma, substitutions);
        assert_eq!(
            gemv,
            singleton_chunks * projections,
            "singleton tail must stay GEMV"
        );
        if candidate {
            assert_eq!(scalar, 0);
        } else {
            assert_eq!(scalar, expected_mma(n, chunk, projections));
        }
    }
    let mut saved = Vec::new();
    log_output(
        out,
        identity(0),
        &endpoint,
        reference.map(|r| r[0].as_slice()),
    );
    if keep_reference {
        saved.push(endpoint);
    } else {
        drop(endpoint);
    }
    // Preserve the actual prefill handoff; continue this session using common
    // corpus tokens. No restore/reset or independent sampling between arms.
    for step in 1..=NEXT {
        assert!(!dispatch_census_is_active());
        let position = n + step - 1;
        let token = tokens[position];
        let start = Instant::now();
        let (result, substitutions) = with_iq2_xs_matmat_variant(variant, || {
            forward.single_token(token, position as u32, &mut session)
        });
        let mut event = identity(step);
        event.as_object_mut().unwrap().extend(json!({"event":"continuation","token_id":token,
            "wall_ms":start.elapsed().as_secs_f64()*1e3,"measured":false,"mma_substitutions":substitutions,"expected_mma_substitutions":0,
            "error":result.as_ref().err().map(ToString::to_string)}).as_object().unwrap().clone());
        emit(out, event);
        let logits = result.expect("ordinary continuation failed; raw record flushed");
        assert_eq!(substitutions, 0, "ordinary decode must stay GEMV");
        assert!(session.kv_n_pos.iter().all(|&p| p == position + 1));
        log_output(
            out,
            identity(step),
            &logits,
            reference.map(|r| r[step].as_slice()),
        );
        if keep_reference {
            saved.push(logits);
        }
    }
    drop(scratch);
    drop(session);
    footprint(
        ctx,
        out,
        &format!("{stream}/N{n}/{round}/{label}/session_dropped"),
    );
    Run {
        timing: (!warm).then_some(Timing {
            gpu_ms,
            wall_ms,
            substitutions,
        }),
        saved,
    }
}

#[test]
fn iq2_xs_mma_model_cpu_chunk_counts() {
    assert_eq!(expected_mma(128, 128, 39), 39);
    assert_eq!(expected_mma(4096, 1024, 39), 156);
    assert_eq!(expected_mma(129, 128, 39), 39);
    assert_eq!(expected_mma(128, 1, 39), 0);
}

#[test]
fn iq2_xs_mma_model_cpu_options() {
    assert_eq!(parse_widths("128,4096"), WIDTHS);
    assert_eq!(parse_widths("129"), vec![129]);
    assert_eq!(parse_widths(" 1, 129, 512 "), vec![1, 129, 512]);
    assert_eq!(Candidate::parse("production").scope(), None);
    assert_eq!(
        Candidate::parse("mma").scope(),
        Some(Iq2XsMatMatVariant::Mma)
    );
    // Reject malformed inputs before model allocation, without changing process env.
    for bad in ["", "0", "129,", "129,129", "-1", "4294967295"] {
        assert!(
            std::panic::catch_unwind(|| parse_widths(bad)).is_err(),
            "{bad}"
        );
    }
}

#[test]
#[ignore = "whole Saluki native IQ2_XS scalar/MMA ABBA; release; lease; new IQ2_XS_MMA_MODEL_OUT"]
fn iq2_xs_mma_model_packet() {
    assert!(!cfg!(debug_assertions), "release diagnostic required");
    assert!(
        std::env::var_os("MTL_DEBUG_LAYER").is_none(),
        "timing requires no debug layer"
    );
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("IQ2_XS_MMA_MODEL_OUT").expect("IQ2_XS_MMA_MODEL_OUT required"))
        .expect("new output file required");
    let rounds = positive_env("IQ2_XS_MMA_MODEL_ROUNDS", 2);
    let chunk = positive_env("IQ2_XS_MMA_MODEL_CHUNK", 1024);
    u32::try_from(chunk).expect("chunk fits u32");
    let widths = match std::env::var("IQ2_XS_MMA_MODEL_WIDTHS") {
        Ok(value) => parse_widths(&value),
        Err(std::env::VarError::NotPresent) => WIDTHS.to_vec(),
        Err(e) => panic!("IQ2_XS_MMA_MODEL_WIDTHS: {e}"),
    };
    let candidate_mode = match std::env::var("IQ2_XS_MMA_MODEL_CANDIDATE") {
        Ok(value) => Candidate::parse(&value),
        Err(std::env::VarError::NotPresent) => Candidate::Mma,
        Err(e) => panic!("IQ2_XS_MMA_MODEL_CANDIDATE: {e}"),
    };
    let path = std::env::var_os("IQ2_XS_MMA_GGUF").unwrap_or_else(|| ARTIFACT.into());
    let gguf = GgufFile::open(path).unwrap();
    let bound = Model::from_gguf(&gguf).unwrap();
    assert_eq!(bound.arch.kind, ArchKind::Dense);
    let streams = streams(&gguf, *widths.iter().max().unwrap());
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let _lease = acquire_metal_benchmark_lease().expect("production adaptive-wait lease");
    let ctx = MetalContext::new().unwrap();
    assert!(
        concurrent_gdn_dense_decode_enabled(),
        "ordinary decode concurrency required"
    );
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        assert!(
            !name.starts_with("QWEN_PREFILL_TRACE_")
                && !name.starts_with("QWEN_PREFILL_NOOP_")
                && !name.starts_with("QWEN_DECODE_GDN_NOOP_")
                && !name.starts_with("QWEN_PHASE_")
                && !(name.starts_with("QWEN_") && name.contains("ORACLE"))
                && !name.starts_with("QWEN_PREFILL_GDN_MATVEC_"),
            "remove diagnostic override {name}"
        );
    }
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(name, _)| {
            name.starts_with("QWEN_")
                && [
                    "PREFILL",
                    "DECODE",
                    "GGUF",
                    "DENSE_GDN",
                    "METAL_LEASE",
                    "MATVEC",
                    "MATMAT",
                    "KV_Q8",
                    "NATIVE_QUANT_EMBED",
                ]
                .iter()
                .any(|part| name.starts_with(&format!("QWEN_{part}")))
        })
        .collect();
    let shards:Vec<_>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,
        "bytes":s.size,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let source_binding = json!({"packet":sha(include_bytes!("iq2_xs_mma_model.rs")),"shared_loader_audit":sha(include_bytes!("iq_capacity_model.rs")),
        "model_loader":sha(include_bytes!("../mod.rs")),"residency":sha(include_bytes!("../residency.rs")),"dispatch":sha(include_bytes!("../dispatch.rs")),
        "prefill":sha(include_bytes!("../../metal_dflash.rs")),"session_pricer":sha(include_bytes!("../../qwen_queue2.rs")),
        "selector":sha(include_bytes!("../../metal/iq2_xs.rs")),"mat_mat":sha(include_bytes!("../../metal/mat_mat.rs")),
        "scalar_kernel":sha(include_bytes!("../../../../../kernels/mat_vec.metal")),"mma_kernel":sha(include_bytes!("../../../../../kernels/iq2_xs_dense.metal")),
        "iq2_xs_grid":sha(include_bytes!("../../../../../kernels/iq2_xs_grid.metalh")),
        "decode":sha(include_bytes!("../token.rs")),"gdn":sha(include_bytes!("../gdn.rs")),
        "tokenizer":sha(include_bytes!("../../tokenizer.rs")),"metallib":sha(crate::KERNELS_METALLIB)});
    let timing_policy = json!({"api":"prefill_tokens_with_multi_hidden_profiled; ordinary wrapper calls this same function",
        "gpu":"sum of GPUEndTime-GPUStartTime after normal completion checks; includes final norm/LM head command",
        "validity":"aggregate finite and positive required; individual raw command timestamps unavailable",
        "wall":"ordinary prefill call including per-call IDs allocation, encoding, waits and final logits readback; excludes session/scratch construction, census and continuations",
        "census":"warm calls only; diagnostic phase splitting rejected; no new command splitting"});
    let selection_policy = json!({"candidate":if candidate_mode == Candidate::Production {"production"} else {"mma"},
        "A_scope":"Some(Scalar)","B_scope":format!("{:?}",candidate_mode.scope()),
        "production":"None follows guarded Auto: aligned multirow MMA when pipeline/device support permits, otherwise scalar",
        "witness":"requires one MMA substitution per eligible chunk/projection in B; production fallback remains legal but does not pass this MMA confirmation",
        "singleton_and_decode":"zero MMA substitutions; warm prefill census checks singleton GEMV counts"});
    emit(
        &mut out,
        json!({"event":"header","schema":"iq2_xs.native_mma.model.v1","packet_revision":2,"device":ctx.describe(),"shards":shards,
        "widths":widths,"configured_chunk":chunk,"rounds":rounds,"continuations":NEXT,"environment":environment,
        "order":"fresh warm A/B then fresh A1/B1/B2/A2 per round, per corpus/width; one copied native model",
        "source_binding":source_binding,"timing_policy":timing_policy,"selection_policy":selection_policy,
        "production":"None follows guarded Auto; A forces Scalar and B uses the recorded selection_policy",
        "decision":"unscored_diagnostic_no_performance_or_quality_promotion"}),
    );
    for (stream, ids, text_hash) in &streams {
        emit(
            &mut out,
            json!({"event":"stream","name":stream,"token_ids":ids,"text_sha256":text_hash,
            "tokens_sha256_i32_le":sha(bytemuck::cast_slice(ids)),"special_token_policy":"encode(false), plain natural text, no model-specific template"}),
        );
    }
    let model = load_copied_native(&ctx, &gguf, &bound, &stamps, &mut out);
    let inventory = xs_projection_inventory(&model);
    assert!(!inventory.is_empty());
    emit(
        &mut out,
        json!({"event":"xs_projection_inventory","per_eligible_chunk":inventory.len(),"bindings":inventory,
        "count_policy":"one IQ2_XS matrix dispatch per realized dense projection per chunk with more than one token"}),
    );
    for (stream, tokens, _) in &streams {
        for &n in &widths {
            eprintln!("IQ2_XS whole {stream} N{n}: warm pair then {rounds} fresh ABBA rounds");
            let warm_a = trajectory(
                &ctx,
                &model,
                &mut out,
                stream,
                tokens,
                n,
                chunk,
                inventory.len(),
                0,
                "warm_A",
                false,
                candidate_mode,
                true,
                None,
            );
            trajectory(
                &ctx,
                &model,
                &mut out,
                stream,
                tokens,
                n,
                chunk,
                inventory.len(),
                0,
                "warm_B",
                true,
                candidate_mode,
                true,
                Some(&warm_a.saved),
            );
            drop(warm_a);
            for round in 1..=rounds {
                let mut reference = Vec::new();
                let mut times = Vec::with_capacity(4);
                for (label, candidate) in ABBA {
                    let run = trajectory(
                        &ctx,
                        &model,
                        &mut out,
                        stream,
                        tokens,
                        n,
                        chunk,
                        inventory.len(),
                        round,
                        label,
                        candidate,
                        candidate_mode,
                        false,
                        if label == "A1" {
                            None
                        } else {
                            Some(&reference)
                        },
                    );
                    times.push(run.timing.unwrap());
                    if label == "A1" {
                        reference = run.saved;
                    }
                }
                let [a1, b1, b2, a2]: [Timing; 4] = times.try_into().ok().unwrap();
                emit(
                    &mut out,
                    json!({"event":"abba_summary","stream":stream,"N":n,"round":round,"candidate_arm":candidate_mode.arm(),
                    "paired_gpu_savings":[1.0-b1.gpu_ms/a1.gpu_ms,1.0-b2.gpu_ms/a2.gpu_ms],
                    "gpu_mean_saving":1.0-(b1.gpu_ms+b2.gpu_ms)/(a1.gpu_ms+a2.gpu_ms),
                    "wall_mean_saving":1.0-(b1.wall_ms+b2.wall_ms)/(a1.wall_ms+a2.wall_ms),
                    "scalar_A2_over_A1_gpu":a2.gpu_ms/a1.gpu_ms,"candidate_mma_substitutions":b1.substitutions+b2.substitutions,
                    "decision":"unscored_no_promotion"}),
                );
            }
        }
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    footprint(&ctx, &mut out, "whole_packet_complete_weights_resident");
    let cells = streams.len() * widths.len();
    emit(
        &mut out,
        json!({"event":"complete","schema":"iq2_xs.native_mma.model.v1","cells":cells,
        "timed_prefills":cells*rounds*4,"warm_prefills":cells*2,"teacher_forced_tokens":cells*(rounds*4+2)*NEXT,
        "decision":"unscored_no_production_change"}),
    );
}
