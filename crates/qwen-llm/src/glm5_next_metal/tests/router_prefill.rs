//! Diagnostic only: actual GLM Fast router substitution, with no promotion gate.
//!
//! Owner invocation (uses the released fixture; GLM53_GGUF can override it):
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_ROUTER_OUT=/tmp/glm53-router-abba.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   glm5_next_metal::packed::router_prefill::router_prefill_abba \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! Each width warms A and B, then records A1/B1/B2/A2 on fresh sessions.
//! A forces generic even at production-selected widths; B forces the candidate
//! at diagnostic widths (including N32), subject to the production capability
//! guard. Outside this scope, Fast uses E8P32 only at N128/N512.
//! Only prefill is timed; allocation and CPU diagnostics are outside it.
//! Ordinary command timestamps are read after completion, without splitting
//! encoders. GPU duration includes execution stalls; wall minus GPU duration
//! is not an attribution of CPU work or memory wiring in isolation.
//! The final-block leaf reuses A2's actual input and existing scratch. Its
//! repeated binding is cache-hot and is NOT an attribution of all 42 routers.
//! JSONL is flushed after every attempt, including execution errors. Numerical
//! differences are observations, never a bitwise or quality acceptance gate.

use super::super::tests::{
    argmax, choice_regret, kl_divergence, long_qualification_text, perf_lease,
};
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Override {
    active: bool,
    enabled: bool,
    substitutions: usize,
    completed_commands: usize,
    valid_gpu_commands: usize,
    gpu_ms: f64,
}

thread_local! {
    static ROUTER_OVERRIDE: Cell<Override> = Cell::new(Override::default());
}

/// Restores the calling thread's previous state even on error or unwinding.
fn with_override<T>(enabled: bool, body: impl FnOnce() -> T) -> (T, Override) {
    struct Restore(Override);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROUTER_OVERRIDE.set(self.0);
        }
    }
    let _restore = Restore(ROUTER_OVERRIDE.replace(Override {
        active: true,
        enabled,
        ..Override::default()
    }));
    let result = body();
    (result, ROUTER_OVERRIDE.get())
}

/// Called only after a successful wait. Read timestamps only in this packet's
/// scope, including the incumbent arm; unrelated tests do not query them.
pub(super) fn record_completed_command_gpu_time(timestamps: impl FnOnce() -> (f64, f64)) {
    let mut value = ROUTER_OVERRIDE.get();
    if !value.active {
        return;
    }
    let (start, end) = timestamps();
    let ms = (end - start) * 1e3;
    value.completed_commands += 1;
    if start.is_finite() && start > 0.0 && end.is_finite() && ms.is_finite() && ms > 0.0 {
        value.valid_gpu_commands += 1;
        value.gpu_ms += ms;
    }
    ROUTER_OVERRIDE.set(value);
}

pub(super) fn requested_override() -> Option<bool> {
    let value = ROUTER_OVERRIDE.get();
    value.active.then_some(value.enabled)
}

/// Count successful candidate encodes only while a diagnostic arm is active.
pub(super) fn record_substitution() {
    let mut value = ROUTER_OVERRIDE.get();
    if !value.active {
        return;
    }
    value.substitutions += 1;
    ROUTER_OVERRIDE.set(value);
}

#[test]
fn router_cpu_production_selection_is_narrow() {
    assert_eq!(requested_override(), None);
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for rows in [
            0, 1, 31, 32, 33, 64, 127, 128, 129, 255, 256, 511, 512, 513, 1024,
        ] {
            let expected = lineage == PackedLineage::Fast && matches!(rows, 128 | 512);
            let queried = Cell::new(false);
            assert_eq!(
                router_e8p32_selected(lineage, 4096, 288, GgmlType::F32, rows, || {
                    queried.set(true);
                    true
                }),
                expected,
                "{lineage:?} N{rows}"
            );
            assert_eq!(
                queried.get(),
                expected,
                "ineligible calls must not query Metal"
            );
            assert!(!router_e8p32_selected(
                lineage,
                4096,
                288,
                GgmlType::F32,
                rows,
                || false
            ));
        }
    }
    for (hidden, experts, dtype) in [
        (2048, 288, GgmlType::F32),
        (4096, 256, GgmlType::F32),
        (4096, 288, GgmlType::Q8_0),
    ] {
        for rows in [128, 512] {
            assert!(!router_e8p32_selected(
                PackedLineage::Fast,
                hidden,
                experts,
                dtype,
                rows,
                || panic!("unqualified geometry must not query Metal")
            ));
        }
    }
}

#[test]
fn router_cpu_pipeline_capacity_matches_dispatch() {
    assert!(router_e8p32_capacity_supported(32, 32, 0, 0));
    assert!(router_e8p32_capacity_supported(32, 1024, 32768, 32768));
    for width in [0, 16, 64] {
        assert!(!router_e8p32_capacity_supported(width, 1024, 0, 32768));
    }
    assert!(!router_e8p32_capacity_supported(32, 31, 0, 32768));
    assert!(!router_e8p32_capacity_supported(32, 1024, 32769, 32768));
}

#[test]
fn router_cpu_scoped_arms_override_widths_but_not_guards() {
    let before = ROUTER_OVERRIDE.get();
    for candidate in [false, true] {
        with_override(candidate, || {
            assert_eq!(requested_override(), Some(candidate));
            for rows in [32, 128, 512] {
                assert_eq!(
                    router_e8p32_selected(
                        PackedLineage::Fast,
                        4096,
                        288,
                        GgmlType::F32,
                        rows,
                        || true
                    ),
                    candidate
                );
                assert!(!router_e8p32_selected(
                    PackedLineage::Fast,
                    4096,
                    288,
                    GgmlType::F32,
                    rows,
                    || false
                ));
                assert!(!router_e8p32_selected(
                    PackedLineage::Exact,
                    4096,
                    288,
                    GgmlType::F32,
                    rows,
                    || panic!("Exact must never query E8P32 capability")
                ));
            }
            for (hidden, experts, dtype, rows) in [
                (2048, 288, GgmlType::F32, 128),
                (4096, 256, GgmlType::F32, 128),
                (4096, 288, GgmlType::Q8_0, 128),
                (4096, 288, GgmlType::F32, 0),
            ] {
                assert!(!router_e8p32_selected(
                    PackedLineage::Fast,
                    hidden,
                    experts,
                    dtype,
                    rows,
                    || panic!("override must not bypass geometry guards")
                ));
            }
            if !candidate {
                assert!(!router_e8p32_selected(
                    PackedLineage::Fast,
                    4096,
                    288,
                    GgmlType::F32,
                    512,
                    || panic!("forced incumbent must not query E8P32 capability")
                ));
            }
        });
        assert_eq!(ROUTER_OVERRIDE.get(), before);
    }
}

#[test]
fn router_cpu_override_restores_nested_error_and_unwind() {
    let before = ROUTER_OVERRIDE.get();
    let (result, report) = with_override(false, || {
        record_completed_command_gpu_time(|| (1.0, 1.25));
        let outer = ROUTER_OVERRIDE.get();
        let ((), inner) = with_override(true, || {
            record_substitution();
            record_completed_command_gpu_time(|| (2.0, 2.5));
        });
        assert_eq!(inner.substitutions, 1);
        assert_eq!(inner.gpu_ms, 500.0);
        assert_eq!(ROUTER_OVERRIDE.get(), outer);
        assert!(
            std::panic::catch_unwind(|| {
                with_override(true, || {
                    record_substitution();
                    panic!("diagnostic unwind");
                });
            })
            .is_err()
        );
        assert_eq!(ROUTER_OVERRIDE.get(), outer);
        Err::<(), &str>("diagnostic error")
    });
    assert_eq!(result, Err("diagnostic error"));
    assert_eq!(report.substitutions, 0);
    assert_eq!(report.completed_commands, 1);
    assert_eq!(report.valid_gpu_commands, 1);
    assert_eq!(report.gpu_ms, 250.0);
    assert_eq!(ROUTER_OVERRIDE.get(), before);
    assert!(
        std::panic::catch_unwind(|| {
            with_override(true, || panic!("outer diagnostic unwind"));
        })
        .is_err()
    );
    assert_eq!(ROUTER_OVERRIDE.get(), before);
}

#[test]
fn router_cpu_inactive_capture_does_not_read_gpu_timestamps() {
    let before = ROUTER_OVERRIDE.get();
    assert!(!before.active);
    record_completed_command_gpu_time(|| panic!("inactive capture queried GPU timestamps"));
    record_substitution();
    assert_eq!(ROUTER_OVERRIDE.get(), before);
}

#[test]
fn router_cpu_capture_rejects_invalid_timestamps_in_both_arms() {
    for candidate in [false, true] {
        let ((), report) = with_override(candidate, || {
            record_completed_command_gpu_time(|| (1.0, 1.25));
            for timestamps in [
                (0.0, 1.0),
                (f64::NAN, 1.0),
                (1.0, f64::INFINITY),
                (2.0, 1.0),
                (1.0, 1.0),
                (1.0, f64::MAX),
            ] {
                record_completed_command_gpu_time(|| timestamps);
            }
        });
        assert_eq!(report.completed_commands, 7);
        assert_eq!(report.valid_gpu_commands, 1);
        assert_eq!(report.gpu_ms, 250.0);
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn emit(out: &mut std::fs::File, event: Value) {
    serde_json::to_writer(&mut *out, &event).expect("write router diagnostic JSONL");
    writeln!(out).unwrap();
    out.flush().unwrap();
}

fn vector_diff(reference: &[f32], candidate: &[f32]) -> Value {
    assert!(!reference.is_empty() && reference.len() == candidate.len());
    let nonfinite = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
    let finite = nonfinite(reference) == 0 && nonfinite(candidate) == 0;
    let metrics = finite.then(|| {
        let mut max_abs = 0.0f64;
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        for (&a, &b) in reference.iter().zip(candidate) {
            let error = f64::from(a) - f64::from(b);
            max_abs = max_abs.max(error.abs());
            squared_error += error * error;
            squared_reference += f64::from(a).powi(2);
        }
        json!({"max_abs": max_abs,
            "relative_l2": (squared_error / squared_reference.max(1e-30)).sqrt()})
    });
    json!({
        "elements": reference.len(),
        "reference_sha256_f32_le": sha256(bytemuck::cast_slice(reference)),
        "candidate_sha256_f32_le": sha256(bytemuck::cast_slice(candidate)),
        "different_bits": reference.iter().zip(candidate).filter(|(a,b)| a.to_bits() != b.to_bits()).count(),
        "reference_nonfinite": nonfinite(reference), "candidate_nonfinite": nonfinite(candidate),
        "finite_metrics": metrics,
    })
}

fn logit_diff(reference: &[f32], candidate: &[f32]) -> Value {
    let vectors = vector_diff(reference, candidate);
    let distribution = reference.iter().chain(candidate).all(|v| v.is_finite()).then(|| {
        let (reference_regret, candidate_regret) = choice_regret(reference, candidate);
        json!({"reference_top1": argmax(reference), "candidate_top1": argmax(candidate),
            "kl_reference_candidate": kl_divergence(reference, candidate),
            "kl_candidate_reference": kl_divergence(candidate, reference),
            "reference_choice_regret": reference_regret, "candidate_choice_regret": candidate_regret})
    });
    json!({"vectors": vectors, "distribution": distribution})
}

struct RouteSnapshot {
    block: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

struct Snapshot {
    logits: Vec<f32>,
    routes: Vec<RouteSnapshot>,
}

fn snapshot(session: &Glm5NextSession<'_>, logits: Vec<f32>) -> Snapshot {
    let p = session.packed.as_ref().unwrap();
    let routes = p
        .routes
        .iter()
        .enumerate()
        .filter_map(|(block, route)| {
            route.as_ref().map(|route| RouteSnapshot {
                block,
                ids: read_i32(&route.ids).unwrap(),
                weights: read_f32(&route.weights).unwrap(),
            })
        })
        .collect();
    Snapshot { logits, routes }
}

fn route_occupancy(ids: &[i32]) -> Value {
    let mut counts = vec![0usize; 288];
    for &id in ids {
        counts[usize::try_from(id).expect("nonnegative expert id")] += 1;
    }
    json!({"counts_by_expert": counts,
        "active_experts": counts.iter().filter(|&&n| n > 0).count(),
        "active_n16_panels": counts.iter().map(|n| n.div_ceil(16)).sum::<usize>(),
        "active_n32_panels": counts.iter().map(|n| n.div_ceil(32)).sum::<usize>()})
}

fn compare(reference: &Snapshot, candidate: &Snapshot, top_k: usize) -> Value {
    assert_eq!(reference.routes.len(), candidate.routes.len());
    let routes: Vec<Value> = reference
        .routes
        .iter()
        .zip(&candidate.routes)
        .map(|(a, b)| {
            assert_eq!(a.block, b.block);
            assert_eq!(a.ids.len(), b.ids.len());
            let mut ordered_changed = 0usize;
            let mut set_changed = 0usize;
            for (a, b) in a.ids.chunks_exact(top_k).zip(b.ids.chunks_exact(top_k)) {
                ordered_changed += usize::from(a != b);
                let (mut a, mut b) = (a.to_vec(), b.to_vec());
                a.sort_unstable();
                b.sort_unstable();
                set_changed += usize::from(a != b);
            }
            json!({"block": a.block, "rows": a.ids.len() / top_k,
            "ordered_route_rows_changed": ordered_changed, "route_set_rows_changed": set_changed,
            "reference_ids_sha256_i32_le": sha256(bytemuck::cast_slice(&a.ids)),
            "candidate_ids_sha256_i32_le": sha256(bytemuck::cast_slice(&b.ids)),
            "reference_occupancy": route_occupancy(&a.ids),
            "candidate_occupancy": route_occupancy(&b.ids),
            "weights_by_slot": vector_diff(&a.weights, &b.weights)})
        })
        .collect();
    json!({"endpoint_logits": logit_diff(&reference.logits, &candidate.logits), "routes": routes})
}

/// Same final-block binding in every leaf arm; scratch output is no longer
/// needed by the completed prefill. No additional GPU allocations or copies.
fn leaf_screen(ctx: &MetalContext, session: &Glm5NextSession<'_>, out: &mut std::fs::File) {
    const REPEATS: usize = 8;
    let p = session.packed.as_ref().unwrap();
    let block = session.weights.blocks.len() - 1;
    let FfnTensors::Moe(moe) = &session.weights.blocks[block].ffn else {
        panic!("last GLM block must have a routed FFN");
    };
    let (h, e, rows) = (
        session.weights.config.hidden_size as usize,
        session.weights.config.expert_count as usize,
        p.rows,
    );
    let input_sha = sha256(bytemuck::cast_slice(&read_f32(&p.normed).unwrap()));
    let mut baseline = None;
    for (label, candidate) in [
        ("warm_A", false),
        ("warm_B", true),
        ("A1", false),
        ("B1", true),
        ("B2", true),
        ("A2", false),
    ] {
        let started = Instant::now();
        let command = ctx.queue.commandBuffer().expect("leaf command buffer");
        let enc = KernelEncoder::begin(&command);
        let result = (|| -> Result<()> {
            for _ in 0..REPEATS {
                if candidate {
                    crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                        ctx,
                        &enc,
                        &moe.router,
                        &p.normed,
                        &p.router,
                        h,
                        e,
                        rows,
                    )?;
                } else {
                    matmat(
                        ctx,
                        &enc,
                        super::super::packed::StageMode::Fast,
                        &moe.router,
                        &p.normed,
                        &p.router,
                        h,
                        e,
                        rows,
                    )?;
                }
            }
            Ok(())
        })();
        enc.end();
        let result = result.and_then(|()| {
            command.commit();
            wait_completed(&command)?;
            Ok(())
        });
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = result
            .is_ok()
            .then(|| (command.GPUEndTime() - command.GPUStartTime()) * 1e3)
            .filter(|ms| ms.is_finite() && *ms > 0.0);
        emit(
            out,
            json!({"event": "leaf_attempt", "rows": rows, "block": block,
            "label": label, "candidate": candidate, "repeated_dispatches": REPEATS,
            "input_sha256_f32_le": input_sha, "wall_ms": wall_ms,
            "command_gpu_ms": gpu_ms, "gpu_ms_per_dispatch": gpu_ms.map(|v| v / REPEATS as f64),
            "error": result.as_ref().err().map(ToString::to_string)}),
        );
        result.expect("router leaf failed; raw attempt was flushed");
        let values = read_f32(&p.router).unwrap();
        if label == "A1" {
            baseline = Some(values);
        } else if let Some(reference) = &baseline {
            emit(
                out,
                json!({"event": "leaf_comparison", "rows": rows, "label": label,
                "reference": "A1", "projection": vector_diff(reference, &values)}),
            );
        }
    }
}

#[test]
#[ignore = "diagnostic timing: GLM53_GGUF, GLM53_ROUTER_OUT (new JSONL), no MTL_DEBUG_LAYER; loads 109.5 GiB under production lease"]
fn router_prefill_abba() {
    assert!(!cfg!(debug_assertions), "timing packet requires --release");
    let _lease = perf_lease();
    let output = std::env::var_os("GLM53_ROUTER_OUT").expect("GLM53_ROUTER_OUT");
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("GLM53_ROUTER_OUT must be a new writable JSONL path");
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), 512, 512)
        .expect("production memory preflight");
    let text = format!("[gMASK]<sop>{}", long_qualification_text());
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .take(512)
        .collect();
    assert_eq!(
        tokens.len(),
        512,
        "natural corpus must supply all requested rows"
    );
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("released GLM weights");
    assert_eq!(
        (
            weights.config.hidden_size,
            weights.config.expert_count,
            weights.config.expert_used_count,
            weights.blocks.len()
        ),
        (4096, 288, 8, 45)
    );
    let routers: Vec<Value> = weights
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(block, weights)| {
            let FfnTensors::Moe(moe) = &weights.ffn else {
                return None;
            };
            assert_eq!(moe.router.dtype, GgmlType::F32);
            let name = format!("blk.{block}.ffn_gate_inp.weight");
            let tensor = gguf
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .expect("router descriptor");
            Some(
                json!({"block": block, "name": name, "dtype": "F32", "shape": tensor.shape,
            "shard": tensor.shard_idx, "offset": tensor.data_offset, "bytes": tensor.n_bytes,
            "sha256": sha256(gguf.slice(tensor))}),
            )
        })
        .collect();
    assert_eq!(routers.len(), 42);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let shards: Vec<Value> = stamps
        .iter()
        .map(|s| {
            json!({"path": s.path, "shard": s.shard_idx,
        "device": s.device, "inode": s.inode, "bytes": s.size,
        "mtime_sec": s.mtime_sec, "mtime_nsec": s.mtime_nsec,
        "ctime_sec": s.ctime_sec, "ctime_nsec": s.ctime_nsec})
        })
        .collect();
    emit(
        &mut out,
        json!({"event": "header", "schema": "glm53.router_prefill_abba.v3",
            "decision": "diagnostic_only_no_promotion", "device": ctx.describe(),
            "fixture": crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.id, "shards": shards,
            "routers": routers, "widths": [32,128,512], "lineage": "Fast",
            "incumbent": "scoped A forces matmat Fast -> generic F32, including production E8P32 widths",
            "candidate": "scoped B requests existing E8P32 strict at all diagnostic widths, including N32; capability fallback retained; 42 substitutions required",
            "production_scope": {"lineage": "Fast", "rows": ROUTER_E8P32_ROWS,
                "hidden": 4096, "experts": 288, "dtype": "F32",
                "fallback": "generic on all other widths/geometry/dtypes or unavailable/incapable pipeline; Exact unchanged"},
            "router_pipeline_supported": router_e8p32_supported(&ctx),
            "order_per_width": ["warm_A", "warm_B", "A1", "B1", "B2", "A2"],
            "whole_model_metric": "prefill wall including encode, wait, route checks and final logits copy; excludes session allocation and diagnostics",
            "command_gpu_metric": "sum of ordinary (GPUEndTime-GPUStartTime)*1000 after successful encode_chunk wait; no encoder splitting; includes GPU execution stalls",
            "command_gpu_validity": "prefill succeeded, exactly one completed command, finite positive start and duration; otherwise command_gpu_ms is null (never a partial sum)",
            "leaf_metric": "final-block router on identical A2 input, repeated cache-hot binding; not whole-model stage attribution",
            "source_binding": {
                "packed_rs_sha256": sha256(include_bytes!("../packed.rs")),
                "packet_rs_sha256": sha256(include_bytes!("router_prefill.rs")),
                "test_helpers_rs_sha256": sha256(include_bytes!("../tests.rs")),
                "session_rs_sha256": sha256(include_bytes!("../../glm5_next_metal.rs")),
                "dispatch_rs_sha256": sha256(include_bytes!("../../metal_forward/dispatch.rs")),
                "matrix_encoder_rs_sha256": sha256(include_bytes!("../../metal/mat_mat.rs")),
                "router_encoder_rs_sha256": sha256(include_bytes!("../../metal/moe.rs")),
                "metallib_sha256": sha256(crate::KERNELS_METALLIB),
                "binding_note": "hashes embedded at compilation, not a claim about runtime checkout HEAD"},
            "prompt": {"corpus": "long_qualification_text, prefixed [gMASK]<sop>; first N tokens",
                "text_sha256": sha256(text.as_bytes()), "token_ids": tokens},
            "cpu_diagnostic_reserve_bytes": 16 << 20,
        }),
    );

    let packet_started = Instant::now();
    for rows in [32usize, 128, 512] {
        let mut baseline_a: Option<Snapshot> = None;
        let mut baseline_b: Option<Snapshot> = None;
        for (label, candidate) in [
            ("warm_A", false),
            ("warm_B", true),
            ("A1", false),
            ("B1", true),
            ("B2", true),
            ("A2", false),
        ] {
            let allocation_started = Instant::now();
            let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                rows,
                rows,
                16 << 20,
            )
            .expect("admit one diagnostic session");
            session.set_packed_lineage(PackedLineage::Fast).unwrap();
            let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;
            let start_ms = packet_started.elapsed().as_secs_f64() * 1e3;
            let ((result, wall_ms), timing) = with_override(candidate, || {
                let started = Instant::now();
                let result = session.prefill_packed(&ctx, &tokens[..rows]);
                (result, started.elapsed().as_secs_f64() * 1e3)
            });
            // Each requested width fits one fresh-session chunk. Missing or
            // unexpected commands must not masquerade as a complete GPU sample.
            let command_gpu_valid = result.is_ok()
                && timing.completed_commands == 1
                && timing.valid_gpu_commands == 1
                && timing.gpu_ms.is_finite()
                && timing.gpu_ms > 0.0;
            let command_gpu_ms = command_gpu_valid.then_some(timing.gpu_ms);
            let substitutions = timing.substitutions;
            emit(
                &mut out,
                json!({"event": "prefill_attempt", "rows": rows,
                "capacity": rows, "chunk_rows": rows, "start_position": 0,
                "label": label, "candidate": candidate, "allocation_ms": allocation_ms,
                "prefill_start_ms_since_packet": start_ms,
                "wall_ms": wall_ms, "router_substitutions": substitutions,
                "command_gpu_ms": command_gpu_ms, "command_gpu_valid": command_gpu_valid,
                "command_gpu_expected_commands": 1,
                "command_gpu_completed_commands": timing.completed_commands,
                "command_gpu_valid_commands": timing.valid_gpu_commands,
                "error": result.as_ref().err().map(ToString::to_string)}),
            );
            let logits = result.expect("prefill failed; raw attempt was flushed");
            assert_eq!(
                substitutions,
                if candidate { 42 } else { 0 },
                "wrong substitution topology"
            );
            assert_eq!(session.position(), rows);
            emit(
                &mut out,
                json!({"event": "prefill_output", "rows": rows, "label": label,
                "logits_sha256_f32_le": sha256(bytemuck::cast_slice(&logits)),
                "nonfinite_logits": logits.iter().filter(|v| !v.is_finite()).count(),
                "top1": logits.iter().all(|v| v.is_finite()).then(|| argmax(&logits))}),
            );
            if !label.starts_with("warm") {
                let current = snapshot(&session, logits);
                if let Some(reference) = &baseline_a {
                    emit(
                        &mut out,
                        json!({"event": "prefill_comparison", "rows": rows,
                        "reference": "A1", "label": label, "comparison": compare(reference, &current, 8)}),
                    );
                }
                if label == "B2" {
                    emit(
                        &mut out,
                        json!({"event": "prefill_comparison", "rows": rows,
                        "reference": "B1", "label": label,
                        "comparison": compare(baseline_b.as_ref().unwrap(), &current, 8)}),
                    );
                }
                match label {
                    "A1" => baseline_a = Some(current),
                    "B1" => baseline_b = Some(current),
                    "A2" => leaf_screen(&ctx, &session, &mut out),
                    _ => {}
                }
            }
            eprintln!(
                "GLM router N{rows} {label}: {wall_ms:.3} ms; command_gpu_ms={command_gpu_ms:?}; command_gpu_valid={command_gpu_valid}; substitutions={substitutions}"
            );
        }
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        &mut out,
        json!({"event": "complete", "decision": "unscored_diagnostic"}),
    );
}
