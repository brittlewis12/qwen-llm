//! Bounded diagnostic, not a planner/router promotion or a quality gate.
//!
//! Owner invocation (no other Metal work; output must not already exist):
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   FLASH_PREFILL_MODEL=/path/to/first-shard.gguf \
//!   FLASH_PREFILL_OUT=/tmp/flash-prefill-map.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   qwen4exp_runtime::tests::prefill_map::native_prefill_map \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! One native session admits the full 4097-position extent. Natural text is
//! repeated as whole text, then tokenized; both workloads use prefixes of the
//! same token stream. First/warm/profiled passes use the incumbent planner.
//! Historical N2045 generic routing is explicitly pinned by this entry point.
//! The historical frontier planner is also pinned for the entire invocation;
//! manually supplied mixed schedules remain explicit and unchanged.
//! Suffix ABBA compares 3+2045 with 2048 from the same position-2048 checkpoint,
//! once with the configured default router and once with strict routing off.
//! Allocation, restore, hashes, JSONL and the one-token handoff are not timed.
//! Each command's executor wall/GPU interval is retained; their sum excludes
//! the diagnostic gaps between commands and is NOT a request TTFT measurement.

#[path = "router_2045.rs"]
mod router_2045;

#[path = "prefill_map/frontier_schedule.rs"]
mod frontier_schedule;

use super::*;
use serde_json::{Value, json};
use std::io::Write;

type PacketResult<T> = Result<T, Box<dyn std::error::Error>>;
const EXTENT: usize = 4097;
const PREFIX: usize = 2048;
const END: usize = 4096;
const CPU_MARGIN: u64 = 16 << 20;
const CORPUS: &str =
    include_str!("../../../../../docs/bench/2026-08-29-qwen4exp-packed-natural-n512/prompt.txt");

fn require(ok: bool, message: &str) -> PacketResult<()> {
    if ok {
        Ok(())
    } else {
        Err(std::io::Error::other(message).into())
    }
}

fn emit(out: &mut std::fs::File, event: Value) {
    serde_json::to_writer(&mut *out, &event).expect("write Flash packet JSONL");
    writeln!(out).expect("terminate Flash packet row");
    out.flush().expect("flush Flash packet row");
}

fn timing_json(t: Qwen4ExpTokenTiming) -> Value {
    json!({"position": t.position, "encode_cpu_ms": t.encode_cpu_ms,
        "completion_wait_ms": t.completion_wait_ms, "gpu_ms": t.gpu_ms,
        "executor_wall_ms": t.total_wall_ms, "outside_gpu_ms": t.outside_gpu_ms()})
}

fn profile_json(outcome: &Qwen4ExpPackedProfileOutcome) -> Value {
    let profile = match &outcome.profile {
        Err(error) => json!({"error": error.to_string(), "stages": null,
            "raw_samples_available": false,
            "note": "existing private API returns resolution error, not rejected raw timestamp samples"}),
        Ok(p) => {
            let stages: Vec<Value> = p
                .stages
                .iter()
                .map(|s| {
                    let parent = p
                        .stages
                        .iter()
                        .filter(|other| {
                            other.depth < s.depth
                                && other.start_sample <= s.start_sample
                                && other.end_sample >= s.end_sample
                        })
                        .max_by_key(|other| other.depth);
                    json!({"scope": s.label.scope.as_str(), "label": s.label.name,
                    "layer": s.label.layer, "mixer": s.label.mixer.map(|m| format!("{m:?}")),
                    "depth": s.depth, "parent_label": parent.map(|v| v.label.name),
                    "start_sample": s.start_sample, "end_sample": s.end_sample,
                    "duration_ticks": s.duration_ticks, "gpu_ms": s.gpu_ms,
                    "fraction_of_command_gpu": s.fraction_of_gpu})
                })
                .collect();
            json!({"error": null, "sample_count": p.sample_count,
                "sampled_span_ticks": p.sampled_span_ticks,
                "raw_span_ms_assuming_ns": p.raw_span_ms_assuming_ns,
                "raw_coverage_assuming_ns": p.raw_coverage_assuming_ns,
                "raw_coverage_accepted": (0.995..=1.005).contains(&p.raw_coverage_assuming_ns),
                "stages": stages,
                "accounting": "inclusive nested spans; never add parents to children; layers 5/7 are representatives, not all dtype cohorts"})
        }
    };
    json!({"sampling": outcome.sampling.as_str(), "sampling_fallback": outcome.sampling_fallback,
        "encode": {"preflight_ms": outcome.encode.preflight_ms,
            "stage_inputs_ms": outcome.encode.stage_inputs_ms,
            "graph_encode_ms": outcome.encode.graph_encode_ms,
            "unattributed_ms": outcome.encode.unattributed_ms},
        "release": {"commit_return_ms": outcome.command.commit_return_ms,
            "root_wait_ms": outcome.command.root_wait_ms,
            "child_publication_ms": outcome.command.child_publication_ms,
            "root_publish_ms": outcome.command.root_publish_ms,
            "release_total_ms": outcome.command.release_total_ms},
        "profile": profile})
}

fn normal_plan(
    r: &Qwen4ExpTextRunner<'_, '_, '_>,
    n: usize,
) -> Result<Qwen4ExpPrefillExecutionPlan, Qwen4ExpRuntimeError> {
    plan_qwen4exp_prefill_execution_from(
        r.next_position(),
        n,
        r.workspace.packed_prefill_capacity(),
        r.workspace.packed_selected_capable() && qwen4exp_packed_selected_qsa_enabled(),
        r.packed_qsa_dense_end()?,
    )
}

struct Run {
    commands: Vec<Qwen4ExpTokenTiming>,
}

impl Run {
    fn gpu_ms(&self) -> Option<f64> {
        self.commands.iter().map(|t| t.gpu_ms).sum()
    }
    fn wall_ms(&self) -> f64 {
        self.commands.iter().map(|t| t.total_wall_ms).sum()
    }
}

/// Calls the existing private executors in precisely the supplied schedule.
/// Output is flushed between commands, outside each executor's time interval.
fn run_schedule(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    plan: &Qwen4ExpPrefillExecutionPlan,
    profiled: bool,
    out: &mut std::fs::File,
    label: &str,
) -> PacketResult<Run> {
    r.validate_prefill_request(tokens)?;
    let base = r.next_position();
    let end = base + tokens.len();
    let mut cursor = base;
    for range in &plan.packed_ranges {
        require(
            range.start == cursor && range.end > range.start && range.end <= end,
            "diagnostic schedule must be contiguous and in the request",
        )?;
        cursor = range.end;
    }
    require(
        cursor == plan.scalar_start && cursor - base == plan.packed_token_count,
        "diagnostic schedule scalar boundary/count mismatch",
    )?;
    emit(
        out,
        json!({"event": "run_begin", "label": label, "profiled": profiled,
        "start": base, "end": end,
        "packed_ranges": plan.packed_ranges.iter().map(|v| [v.start, v.end]).collect::<Vec<_>>(),
        "scalar_start": plan.scalar_start, "contains_selection": plan.contains_selection}),
    );
    let mut commands = Vec::new();
    for range in &plan.packed_ranges {
        emit(
            out,
            json!({"event": "command_begin", "label": label,
            "absolute_range": [range.start, range.end], "profiled": profiled}),
        );
        let ids = &tokens[range.start - base..range.end - base];
        let result = if profiled {
            execute_qwen4exp_text_packed_profiled_sync(
                r.ctx,
                ids,
                r.ple_table,
                &r.weights,
                &mut r.workspace,
            )
            .map(|p| (p.token, Some(profile_json(&p))))
        } else {
            execute_qwen4exp_text_packed_sync(r.ctx, ids, r.ple_table, &r.weights, &mut r.workspace)
                .map(|t| (t, None))
        };
        match result {
            Ok((timing, profile)) => {
                emit(
                    out,
                    json!({"event": "command_complete", "label": label,
                    "absolute_range": [range.start, range.end], "timing": timing_json(timing),
                    "profile": profile, "published_position": r.next_position()}),
                );
                require(
                    r.next_position() == range.end,
                    "packed publication differs from plan",
                )?;
                commands.push(timing);
            }
            Err(error) => {
                emit(
                    out,
                    json!({"event": "command_error", "label": label,
                    "absolute_range": [range.start, range.end], "error": error.to_string(),
                    "published_position": r.next_position()}),
                );
                return Err(error.into());
            }
        }
    }
    for &id in &tokens[plan.scalar_start - base..] {
        let position = r.next_position();
        emit(
            out,
            json!({"event": "command_begin", "label": label,
            "absolute_range": [position, position + 1], "scalar": true}),
        );
        let result =
            execute_qwen4exp_text_token_sync(r.ctx, id, r.ple_table, &r.weights, &mut r.workspace);
        emit(
            out,
            json!({"event": "scalar_command", "label": label,
            "absolute_range": [position, position + 1],
            "timing": result.as_ref().ok().map(|t| timing_json(*t)),
            "error": result.as_ref().err().map(ToString::to_string)}),
        );
        commands.push(result?);
    }
    require(r.next_position() == end, "run did not publish its endpoint")?;
    let run = Run { commands };
    emit(
        out,
        json!({"event": "run_complete", "label": label,
        "commands": run.commands.len(), "executor_wall_ms_sum": run.wall_ms(),
        "command_gpu_ms_sum": run.gpu_ms(), "published_position": r.next_position()}),
    );
    Ok(run)
}

struct Endpoint {
    logits: Vec<f32>,
    state: Value,
}

fn endpoint(
    r: &Qwen4ExpTextRunner<'_, '_, '_>,
    packed_rows: Option<usize>,
) -> PacketResult<Endpoint> {
    let logits = r.logits()?.to_vec();
    let tensors: Vec<Value> = r
        .workspace
        .persistent_state_tensors()
        .iter()
        .enumerate()
        .map(|(index, t)| {
            // SAFETY: private synchronous execution has released the workspace;
            // these shared buffers remain alive and no command runs while hashing.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    t.buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(t.offset as usize),
                    t.n_bytes() as usize,
                )
            };
            json!({"index": index, "dtype": format!("{:?}", t.dtype),
                "shape": t.shape, "bytes": bytes.len(), "sha256": sha256_bytes(bytes)})
        })
        .collect();
    let hyper = match packed_rows {
        Some(rows) => r.workspace.final_packed_hyper_for_tests(rows),
        None => r.workspace.final_hyper_for_tests(),
    };
    Ok(Endpoint {
        logits,
        state: json!({"position": r.next_position(),
        "scope": "full persistent allocations, including inactive cache tails; suffix arms restore identical storage",
        "qsa_lengths": r.workspace.qsa_committed_lengths(),
        "ple_prior_tokens": r.workspace.ple_prior_tokens(), "tensors": tensors,
        "hyper_sha256_f32_le": sha256_f32_bits(b"", &hyper)}),
    })
}

fn comparison(a: &Endpoint, b: &Endpoint) -> Value {
    let same_bits = a.logits.len() == b.logits.len()
        && a.logits
            .iter()
            .zip(&b.logits)
            .all(|(x, y)| x.to_bits() == y.to_bits());
    let finite = a.logits.iter().chain(&b.logits).all(|v| v.is_finite());
    let metrics = (finite && a.logits.len() == b.logits.len()).then(|| {
        let mut error = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (&x, &y) in a.logits.iter().zip(&b.logits) {
            let d = f64::from(x) - f64::from(y);
            error += d * d;
            scale += f64::from(x).powi(2);
            max_abs = max_abs.max(d.abs());
        }
        json!({"relative_l2": (error / scale.max(1e-30)).sqrt(), "max_abs": max_abs,
            "reference_top1": argmax(&a.logits), "candidate_top1": argmax(&b.logits)})
    });
    json!({"logits_bits_equal": same_bits, "state_digests_equal": a.state == b.state,
        "finite": finite, "metrics": metrics,
        "endpoint_and_state_concordant": same_bits && a.state == b.state})
}

fn emit_endpoint(out: &mut std::fs::File, label: &str, e: &Endpoint) {
    emit(
        out,
        json!({"event": "endpoint", "label": label, "state": e.state,
        "logits_sha256_f32_le": sha256_f32_bits(b"", &e.logits),
        "logit_count": e.logits.len(),
        "nonfinite_logits": e.logits.iter().filter(|v| !v.is_finite()).count()}),
    );
}

fn observer(out: &mut std::fs::File, width: usize, warm: &Run, profiled: &Run) {
    emit(
        out,
        json!({"event": "observer_shape", "width": width,
        "same_command_endpoints": warm.commands.len() == profiled.commands.len()
            && warm.commands.iter().zip(&profiled.commands).all(|(a, b)| a.position == b.position)}),
    );
    for (index, (a, b)) in warm.commands.iter().zip(&profiled.commands).enumerate() {
        let gpu = a.gpu_ms.zip(b.gpu_ms).map(|(a, b)| b / a);
        let wall = b.total_wall_ms / a.total_wall_ms;
        emit(
            out,
            json!({"event": "observer", "width": width, "command_index": index,
            "gpu_ratio": gpu, "wall_ratio": wall,
            "accepted": gpu.is_some_and(|v| (0.985..=1.015).contains(&v))
                && (0.98..=1.02).contains(&wall)}),
        );
    }
}

fn census_json(rows: &[DispatchCensusRow]) -> Value {
    let mut kernels = BTreeMap::<String, usize>::new();
    for row in rows {
        *kernels.entry(row.kernel.clone()).or_default() += 1;
    }
    json!({"dispatches": rows.len(), "kernel_counts": kernels,
        "timing_eligible": false, "note": "warm witness only; measured ABBA has no census"})
}

fn suffix_packet(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    out: &mut std::fs::File,
) -> PacketResult<()> {
    r.reset()?;
    let prefix_plan = normal_plan(r, PREFIX)?;
    run_schedule(
        r,
        &tokens[..PREFIX],
        &prefix_plan,
        false,
        out,
        "suffix_prefix",
    )?;
    let checkpoint = r.workspace.checkpoint_for_tests();
    let normal = normal_plan(r, END - PREFIX)?;
    require(
        normal.packed_ranges == vec![PREFIX..2051, 2051..END] && normal.scalar_start == END,
        "expected incumbent 3+2045 suffix plan",
    )?;
    let mixed = Qwen4ExpPrefillExecutionPlan {
        packed_ranges: vec![PREFIX..END],
        packed_token_count: END - PREFIX,
        scalar_start: END,
        contains_selection: true,
    };
    let mut effects = Vec::new();
    let mut default_a: Option<Endpoint> = None;
    for disabled in [false, true] {
        let policy = if disabled {
            "strict_disabled"
        } else {
            "router_default"
        };
        let mut a_reference: Option<Endpoint> = None;
        let mut b_reference: Option<Endpoint> = None;
        let mut measurements = Vec::new();
        for (arm, merge) in [
            ("warm_A", false),
            ("warm_B", true),
            ("A1", false),
            ("B1", true),
            ("B2", true),
            ("A2", false),
        ] {
            let label = format!("suffix/{policy}/{arm}");
            let restore_start = Instant::now();
            r.workspace.restore_checkpoint_for_tests(&checkpoint);
            emit(
                out,
                json!({"event": "restore", "label": label,
                "position": r.next_position(), "outside_timing_ms": restore_start.elapsed().as_secs_f64() * 1e3}),
            );
            let warm = arm.starts_with("warm");
            if warm {
                crate::metal::dispatch_census_begin();
            }
            let mut execute = || {
                run_schedule(
                    r,
                    &tokens[PREFIX..END],
                    if merge { &mixed } else { &normal },
                    false,
                    out,
                    &label,
                )
            };
            let result = if disabled {
                with_qwen4exp_packed_router_e8p32_strict_override(false, execute)
            } else {
                execute()
            };
            if warm {
                emit(
                    out,
                    json!({"event": "census", "label": label,
                    "witness": census_json(&crate::metal::dispatch_census_take())}),
                );
            }
            let run = result?;
            let current = endpoint(r, Some(if merge { PREFIX } else { END - 2051 }))?;
            emit_endpoint(out, &label, &current);
            if !warm {
                if disabled && arm == "A1" {
                    emit(
                        out,
                        json!({"event": "comparison", "label": label,
                        "reference": "suffix/router_default/A1", "same_schedule": true,
                        "note": "3/2045 are outside strict scope, so this is the unchanged-router control across policy blocks",
                        "comparison": comparison(default_a.as_ref().unwrap(), &current)}),
                    );
                }
                if let Some(reference) = &a_reference {
                    emit(
                        out,
                        json!({"event": "comparison", "label": label,
                        "reference": format!("suffix/{policy}/A1"),
                        "same_schedule": !merge, "comparison": comparison(reference, &current)}),
                    );
                }
                if arm == "B2" {
                    emit(
                        out,
                        json!({"event": "comparison", "label": label,
                        "reference": format!("suffix/{policy}/B1"), "same_schedule": true,
                        "comparison": comparison(b_reference.as_ref().unwrap(), &current)}),
                    );
                }
                measurements.push((run.gpu_ms(), run.wall_ms()));
                if arm == "A1" {
                    a_reference = Some(current);
                } else if arm == "B1" {
                    b_reference = Some(current);
                }
            }
            // One original corpus token; diagnostic handoff, outside suffix timing.
            let result = r.forward_token(tokens[END]).map(|v| v.to_vec());
            emit(
                out,
                json!({"event": "handoff", "label": label,
                "token_id": tokens[END], "published_position": r.next_position(),
                "logits_sha256_f32_le": result.as_ref().ok().map(|v| sha256_f32_bits(b"", v)),
                "nonfinite_logits": result.as_ref().ok().map(|v| v.iter().filter(|x| !x.is_finite()).count()),
                "error": result.as_ref().err().map(ToString::to_string)}),
            );
            result?;
        }
        let gpu: Option<Vec<f64>> = measurements.iter().map(|v| v.0).collect();
        let gpu_saved = gpu.as_ref().map(|v| (v[0] + v[3] - v[1] - v[2]) * 0.5);
        let gpu_pairs = gpu.as_ref().map(|v| [v[0] - v[1], v[3] - v[2]]);
        let control_spread = gpu
            .as_ref()
            .map(|v| 2.0 * (v[3] - v[0]).abs() / (v[3] + v[0]));
        let wall_saved =
            (measurements[0].1 + measurements[3].1 - measurements[1].1 - measurements[2].1) * 0.5;
        emit(
            out,
            json!({"event": "suffix_abba_summary", "router_policy": policy,
            "order": ["A1", "B1", "B2", "A2"], "gpu_ms": gpu,
            "executor_wall_ms": measurements.iter().map(|v| v.1).collect::<Vec<_>>(),
            "mean_gpu_ms_saved_by_merge": gpu_saved, "mean_executor_wall_ms_saved_by_merge": wall_saved,
            "paired_gpu_ms_saved": gpu_pairs, "control_gpu_spread_fraction": control_spread,
            "scope": "suffix only; checkpoint/logging/census/hash/handoff excluded; unscored"}),
        );
        effects.push(gpu_saved);
        if !disabled {
            default_a = a_reference;
        }
    }
    emit(
        out,
        json!({"event": "factorial", "default_merge_gpu_ms_saved": effects[0],
        "strict_disabled_merge_gpu_ms_saved": effects[1],
        "difference_in_differences_ms": effects[0].zip(effects[1]).map(|(a, b)| a - b),
        "interpretation": "router-policy interaction, not an independently additive saving or pure router attribution; regrouping, bucket density and arithmetic may change; policy blocks are not interleaved"}),
    );
    Ok(())
}

fn with_native_runner(
    ctx: &MetalContext,
    gguf: &GgufFile,
    extent: usize,
    cpu_margin: u64,
    out: &mut std::fs::File,
    work: impl FnOnce(&mut Qwen4ExpTextRunner<'_, '_, '_>, &mut std::fs::File) -> PacketResult<()>,
) -> PacketResult<()> {
    let config = Qwen4ExpConfig::from_gguf(gguf)?;
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, extent)?;
    emit(
        out,
        json!({"event": "load_begin", "forward_limit": extent,
        "packed_admission_extent": extent, "qsa_physical_capacity": capacity.qsa_physical_capacity()}),
    );
    // Production native combined admission precedes realization; never retry scalar
    // or bypass admission if it refuses. No test-only strict UD binder is used.
    let mut loaded = Qwen4ExpLoadedModel::load_with_packed_prefill(ctx, gguf, capacity, extent)?;
    emit(
        out,
        json!({"event": "loaded", "admission": format!("{:?}", loaded.admission()),
        "weight_allocation_bytes": loaded.observed_weight_bytes(),
        "session_allocation_bytes": loaded.observed_session_bytes(),
        "packed_capacity": loaded.packed_prefill_capacity(),
        "selected_capable": loaded.packed_selected_capable(),
        "selected_active": loaded.packed_selected_active()}),
    );
    require(
        loaded.packed_prefill_capacity() == Some(PREFIX) && loaded.packed_selected_active(),
        "packet needs the existing 2048-row selected-capable Flash plan",
    )?;
    let mut r = loaded.create_runner(ctx)?;
    // One checkpoint plus small logits/digest/census records; no retained full
    // endpoint state copies. Bound and admit this CPU addition before model work.
    let checkpoint_bytes = r
        .workspace
        .persistent_state_tensors()
        .iter()
        .map(|t| t.n_bytes())
        .sum::<u64>()
        + (config.vocab_size as u64 + r.weights.geometry.hyper_width() as u64) * 4;
    emit(
        out,
        json!({"event": "diagnostic_memory_bound", "checkpoint_bytes": checkpoint_bytes,
        "cpu_margin_bytes": cpu_margin, "bounded_cpu_bytes": checkpoint_bytes + cpu_margin,
        "maximum_cpu_bytes": 512u64 << 20}),
    );
    require(
        checkpoint_bytes + cpu_margin <= (512 << 20),
        "diagnostic CPU bound exceeds 512 MiB",
    )?;
    let cpu_gate = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        0,
        checkpoint_bytes + cpu_margin,
        crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    emit(
        out,
        json!({"event": "diagnostic_memory_gate", "checkpoint_bytes": checkpoint_bytes,
        "cpu_margin_bytes": cpu_margin, "admitted": cpu_gate.admitted,
        "admission": format!("{cpu_gate:?}")}),
    );
    require(
        cpu_gate.admitted,
        "diagnostic memory gate refused; no model execution",
    )?;
    let mut moe = vec![
        r.weights.zero_one.layer_zero.moe,
        r.weights.zero_one.layer_one_moe,
    ];
    moe.extend(r.weights.post_ple.iter().map(|b| b.moe));
    emit(
        out,
        json!({"event": "moe_cohorts", "layers": moe.iter().enumerate().map(|(layer, w)|
        json!({"layer": layer, "router": format!("{:?}", w.router.dtype),
            "gate": format!("{:?}", w.routed_gate.dtype), "up": format!("{:?}", w.routed_up.dtype),
            "down": format!("{:?}", w.routed_down.dtype)})).collect::<Vec<_>>() }),
    );

    work(&mut r, out)
}

fn model_packet(
    ctx: &MetalContext,
    gguf: &GgufFile,
    tokens: &[u32],
    out: &mut std::fs::File,
) -> PacketResult<()> {
    with_native_runner(ctx, gguf, EXTENT, CPU_MARGIN, out, |r, out| {
        for width in [512usize, END] {
            let mut baseline: Option<Endpoint> = None;
            let mut warm_run: Option<Run> = None;
            for pass in ["first", "warm", "profiled"] {
                r.reset()?;
                let label = format!("normal/{width}/{pass}");
                let plan = normal_plan(&r, width)?;
                let run =
                    run_schedule(r, &tokens[..width], &plan, pass == "profiled", out, &label)?;
                let packed_rows =
                    (plan.scalar_start == width).then(|| plan.packed_ranges.last().unwrap().len());
                let current = endpoint(&r, packed_rows)?;
                emit_endpoint(out, &label, &current);
                if let Some(reference) = &baseline {
                    emit(
                        out,
                        json!({"event": "comparison", "label": label,
                    "reference": format!("normal/{width}/first"), "same_schedule": true,
                    "comparison": comparison(reference, &current)}),
                    );
                } else {
                    baseline = Some(current);
                }
                if pass == "profiled" {
                    observer(out, width, warm_run.as_ref().unwrap(), &run);
                }
                if pass == "warm" {
                    warm_run = Some(run);
                }
            }
        }
        suffix_packet(r, tokens, out)
    })
}

fn with_native_artifact(
    out: &mut std::fs::File,
    work: impl FnOnce(&MetalContext, &GgufFile, &mut std::fs::File) -> PacketResult<()>,
) -> PacketResult<()> {
    require(!cfg!(debug_assertions), "timing packet requires --release")?;
    require(
        std::env::var_os("MTL_DEBUG_LAYER").is_none(),
        "remove MTL_DEBUG_LAYER for timing",
    )?;
    let executable = std::env::current_exe()?;
    emit(
        out,
        json!({"event": "executable_binding", "path": executable,
        "sha256": sha256_file(&executable), "bytes": std::fs::metadata(&executable)?.len()}),
    );
    // This function calls the real wired-memory gate even in cfg(test).
    let _lease = crate::metal::acquire_metal_benchmark_lease()?;
    emit(
        out,
        json!({"event": "production_lease_acquired", "wired_gate_passed": true}),
    );
    let path = std::env::var_os("FLASH_PREFILL_MODEL").ok_or("FLASH_PREFILL_MODEL is required")?;
    let gguf = GgufFile::open(std::path::PathBuf::from(path))?;
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    emit(
        out,
        json!({"event": "artifact", "shards": stamps.iter().map(|s| json!({
        "path": s.path, "shard": s.shard_idx, "device": s.device, "inode": s.inode,
        "bytes": s.size, "mtime_sec": s.mtime_sec, "mtime_nsec": s.mtime_nsec,
        "ctime_sec": s.ctime_sec, "ctime_nsec": s.ctime_nsec})).collect::<Vec<_>>() }),
    );
    let ctx = MetalContext::new()?;
    emit(
        out,
        json!({"event": "device", "description": ctx.describe(),
        "selected_requested": qwen4exp_packed_selected_qsa_enabled(),
        "router_default_enabled": crate::env_flag::read_default_on("QWEN4EXP_PACKED_ROUTER_E8P32_STRICT"),
        "strict_router_pipeline_supported": crate::qwen4exp_moe::packed_router_e8p32_strict_supported(&ctx)}),
    );
    require(
        crate::env_flag::read_default_on("QWEN4EXP_PACKED_ROUTER_E8P32_STRICT"),
        "router diagnostic requires configured strict router enabled; rollback remains authoritative",
    )?;
    // Retain the final shard check even when a private diagnostic assertion
    // panics. The outer new-file envelope still records the original panic.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&ctx, &gguf, out)));
    match &result {
        Ok(Err(error)) => emit(
            out,
            json!({"event": "model_error", "error": error.to_string(),
            "model_work_aborted": true}),
        ),
        Err(_) => emit(
            out,
            json!({"event": "model_error", "error": "diagnostic panic; see final error",
            "model_work_aborted": true}),
        ),
        _ => {}
    }
    let final_stamps = gguf.revalidate_retained_shard_stamps();
    emit(
        out,
        json!({"event": "artifact_revalidation", "unchanged": final_stamps.as_ref().is_ok_and(|s| *s == stamps),
        "error": final_stamps.as_ref().err().map(ToString::to_string)}),
    );
    match result {
        Ok(result) => result?,
        Err(payload) => std::panic::resume_unwind(payload),
    }
    require(final_stamps? == stamps, "retained artifact stamps changed")
}

fn packet(out: &mut std::fs::File) -> PacketResult<()> {
    with_native_artifact(out, |ctx, gguf, out| {
        let tokenizer = Tokenizer::from_gguf(gguf)?;
        let source_tokens = tokenizer
            .encode(CORPUS, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let mut text = String::new();
        let mut tokens = Vec::new();
        let mut repetitions = 0;
        while tokens.len() < EXTENT && repetitions < 32 {
            if repetitions > 0 {
                text.push_str("\n\n");
            }
            text.push_str(CORPUS);
            repetitions += 1;
            tokens = tokenizer
                .encode(&text, false)?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
        }
        require(
            tokens.len() >= EXTENT,
            "32 whole-corpus repetitions did not supply 4097 tokens",
        )?;
        emit(
            out,
            json!({"event": "prompt", "source": "docs/bench/2026-08-29-qwen4exp-packed-natural-n512/prompt.txt",
        "source_text_sha256": sha256_bytes(CORPUS.as_bytes()), "whole_corpus_repetitions": repetitions,
        "separator": "two newlines between exact whole corpus copies", "add_special_tokens": false,
        "source_token_ids": source_tokens, "source_token_count": source_tokens.len(),
        "source_token_ids_sha256_u32_le": sha256_u32_le(b"", &source_tokens),
        "prefix512_matches_standalone_source": (source_tokens.len() >= 512)
            .then(|| source_tokens[..512] == tokens[..512]),
        "full_text": text, "full_text_sha256": sha256_bytes(text.as_bytes()),
        "full_token_ids": tokens, "full_token_ids_sha256_u32_le": sha256_u32_le(b"", &tokens),
        "prefixes": ([512usize, END].iter().map(|&n| json!({"tokens": n,
            "sha256_u32_le": sha256_u32_le(b"", &tokens[..n])})).collect::<Vec<_>>()),
        "first_label": "first for this width; not a controlled placement-cold measurement"}),
        );
        require(
            source_tokens.len() < 512 || source_tokens[..512] == tokens[..512],
            "corpus repetition changed the standalone natural512 prefix",
        )?;
        model_packet(ctx, gguf, &tokens, out)
    })
}

fn with_historical_prefill_map_policies<R>(work: impl FnOnce() -> R) -> R {
    with_qwen4exp_frontier_schedule_override(Some(false), || {
        crate::qwen4exp_moe::with_qwen4exp_packed_router_n2045_override(false, work)
    })
}

#[test]
fn prefill_map_historical_schedule_pin_restores_production() {
    let plan = |start, count| {
        plan_qwen4exp_prefill_execution_from(start, count, Some(2048), true, 2051).unwrap()
    };
    let production_whole = plan(0, END);
    let production_suffix = plan(PREFIX, END - PREFIX);
    assert_eq!(production_whole.packed_ranges, vec![0..PREFIX, PREFIX..END]);
    assert_eq!(production_suffix.packed_ranges, vec![PREFIX..END]);
    with_historical_prefill_map_policies(|| {
        let whole = plan(0, END);
        let suffix = plan(PREFIX, END - PREFIX);
        assert_eq!(
            whole.packed_ranges,
            vec![0..PREFIX, PREFIX..2051, 2051..END]
        );
        assert_eq!(suffix.packed_ranges, vec![PREFIX..2051, 2051..END]);
        assert_eq!(whole.packed_token_count, END);
        assert_eq!(suffix.packed_token_count, END - PREFIX);
        assert_eq!(whole.scalar_start, END);
        assert_eq!(suffix.scalar_start, END);
        assert!(whole.contains_selection && suffix.contains_selection);
        assert_eq!(plan(0, 512).packed_ranges, vec![0..512]);
    });
    assert_eq!(plan(0, END), production_whole);
    assert_eq!(plan(PREFIX, END - PREFIX), production_suffix);
}

#[test]
#[ignore = "production lease; FLASH_PREFILL_MODEL and new FLASH_PREFILL_OUT JSONL; release timing only"]
fn native_prefill_map() {
    run_packet(
        "flash.prefill_map.v1",
        include_bytes!("prefill_map.rs"),
        json!({
            "historical_baseline": "N2045 explicitly pinned to generic router; other default widths unchanged",
            "frontier_schedule_override": false,
            "planner_scope": "explicit Some(false) for the entire historical invocation; planner-derived whole2048+3+2045 and suffix3+2045; manually supplied mixed suffix2048 unchanged; shared native scaffolding is unpinned",
            "measurement": "sum of private executor command intervals, excludes diagnostic gaps; not request TTFT",
            "allocation_scope": "both widths share full 4097-position admission",
            "bounded_work": "512/4096 first,warm,profiled; 2048 prefix; two router policies warm_A,warm_B,A1,B1,B2,A2 plus one handoff"
        }),
        |out| with_historical_prefill_map_policies(|| packet(out)),
    );
}

fn run_packet(
    schema: &str,
    packet_source: &[u8],
    details: Value,
    work: impl FnOnce(&mut std::fs::File) -> PacketResult<()>,
) {
    let path =
        std::env::var_os("FLASH_PREFILL_OUT").expect("FLASH_PREFILL_OUT must name a new JSONL");
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("FLASH_PREFILL_OUT must not already exist");
    emit(
        &mut out,
        json!({"event": "header", "schema": schema, "details": details,
                "decision": "diagnostic_only_no_promotion", "pid": std::process::id(),
                "source_binding": {
                    "runtime_rs": sha256_bytes(include_bytes!("../../qwen4exp_runtime.rs")),
                    "packet_rs": sha256_bytes(packet_source),
                    "shared_packet_rs": sha256_bytes(include_bytes!("prefill_map.rs")),
                    "session_rs": sha256_bytes(include_bytes!("../../qwen4exp_text_session.rs")),
                    "checkpoint_rs": sha256_bytes(include_bytes!("../../qwen4exp_text_session/checkpoint.rs")),
                    "qsa_rs": sha256_bytes(include_bytes!("../../qwen4exp_qsa.rs")),
                    "moe_rs": sha256_bytes(include_bytes!("../../qwen4exp_moe.rs")),
                    "gdn_rs": sha256_bytes(include_bytes!("../../qwen4exp_gdn.rs")),
                    "dispatch_rs": sha256_bytes(include_bytes!("../../metal_forward/dispatch.rs")),
                    "profile_rs": sha256_bytes(include_bytes!("../../qwen4exp_profile.rs")),
                    "metallib": sha256_bytes(crate::KERNELS_METALLIB),
                    "note": "SHA256 of compiled inputs; not a claim about runtime checkout HEAD"},
                "environment": (["QWEN4EXP_PACKED_SELECTED_QSA", "QWEN4EXP_PACKED_ROUTER_E8P32_STRICT",
                    "QWEN_MATMAT_BF16_BFLOAT_ACT", "QWEN4EXP_MOE_IQ4_DOWN_M128_N16"]
                    .iter().map(|&key| (key, std::env::var(key).ok())).collect::<BTreeMap<_, _>>()),
        }),
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut out)));
    // Do not leave a warm witness active if an assertion unwound its packet.
    let _ = crate::metal::dispatch_census_take();
    let error = match result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error.to_string()),
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "non-string panic".into()),
        ),
    };
    if let Some(error) = &error {
        emit(
            &mut out,
            json!({"event": "error", "error": error, "model_work_aborted": true}),
        );
    }
    emit(
        &mut out,
        json!({"event": "complete", "execution_complete": error.is_none(),
        "decision": "unscored_diagnostic", "error": error,
        "measurement_validity": "inspect profile errors, raw coverage, observer ratios, concordance and ABBA drift; execution completion is not measurement acceptance"}),
    );
    assert!(
        error.is_none(),
        "Flash packet failed; error and completion flushed: {error:?}"
    );
}
