//! N2045 router qualification; A pins the pre-promotion generic route.
//! Run only under the production lease, in release, with a NEW output path:
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   FLASH_PREFILL_MODEL=/path/to/first-shard.gguf \
//!   FLASH_PREFILL_OUT=/tmp/flash-router-2045.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   qwen4exp_runtime::tests::prefill_map::router_2045::native_router_n2045 \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! A pins only N2045 to generic; B admits only N2045 to strict E8P32, subject
//! to the existing global rollback, geometry, dtype, and pipeline guards.
//! Fresh means reset state in one resident session, not placement-cold TTFT.
//! Fresh-arm persistent storage is zeroed outside timing so inactive cache tails
//! cannot masquerade as arithmetic differences after a preceding continuation.
//! Warm census arms are never performance evidence. Timed arms use ordinary
//! prefill, no census/profiler and no diagnostic callback between commands.

use super::*;
use crate::qwen4exp_moe::with_qwen4exp_packed_router_n2045_override as candidate;

const SUFFIX_START: usize = 2051;
const WIDE: usize = 8192;
const CONTINUATION: usize = 4;
const ADMISSION_EXTENT: usize = WIDE + CONTINUATION;
const DIAGNOSTIC_MARGIN: u64 = 96 << 20;
const SSH: &[u8] = include_bytes!(
    "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
);

#[test]
fn router_n2045_normal_planner_reachability() {
    let plan = |start, count, selected| {
        plan_qwen4exp_prefill_execution_from(start, count, Some(2048), selected, 2051).unwrap()
    };
    for enabled in [false, true] {
        candidate(enabled, || {
            assert_eq!(
                plan(0, 4096, true).packed_ranges,
                vec![0..2048, 2048..2051, 2051..4096]
            );
            assert_eq!(plan(2051, 2045, true).packed_ranges, vec![2051..4096]);
            assert_eq!(
                plan(0, 8192, true).packed_ranges,
                vec![0..2048, 2048..2051, 2051..4099, 4099..6147, 6147..8192]
            );
            for end in [4095, 4097, 4098, 4099] {
                let p = plan(0, end, true);
                assert_eq!(p.packed_ranges.last().unwrap().len(), end - 2051);
                assert!(!p.packed_ranges.iter().any(|r| r.len() == 2045));
            }
            assert_eq!(plan(0, 512, true).packed_ranges, vec![0..512]);
            // Capability=false stops packed traversal at the frontier; it does
            // not disable scalar attention sparsity. Do not fake reachability.
            let fallback = plan(0, 4096, false);
            assert_eq!(fallback.packed_ranges, vec![0..2048, 2048..2051]);
            assert_eq!(fallback.scalar_start, 2051);
            assert!(plan(2051, 2045, false).packed_ranges.is_empty());
        });
    }
}

fn prompt(
    tokenizer: &Tokenizer,
    source: &str,
    id: &str,
    out: &mut std::fs::File,
) -> PacketResult<Vec<u32>> {
    let mut text = String::new();
    let mut tokens = Vec::new();
    let mut repetitions = 0;
    while tokens.len() < ADMISSION_EXTENT && repetitions < 32 {
        if repetitions != 0 {
            text.push_str("\n\n");
        }
        text.push_str(source);
        repetitions += 1;
        tokens = tokenizer
            .encode(&text, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
    }
    require(
        tokens.len() >= ADMISSION_EXTENT,
        "natural corpus is too short after 32 repetitions",
    )?;
    emit(
        out,
        json!({"event": "prompt", "id": id,
        "source_text_sha256": sha256_bytes(source.as_bytes()),
        "whole_text_repetitions": repetitions, "separator": "two newlines", "add_special_tokens": false,
        "full_text": text, "full_text_sha256": sha256_bytes(text.as_bytes()),
        "full_token_ids": tokens, "full_token_ids_sha256_u32_le": sha256_u32_le(b"", &tokens),
        "prefixes": ([4096usize, 8192].map(|n| json!({"tokens": n,
            "sha256_u32_le": sha256_u32_le(b"", &tokens[..n])})))}),
    );
    Ok(tokens)
}

fn prompts(
    gguf: &GgufFile,
    out: &mut std::fs::File,
) -> PacketResult<Vec<(&'static str, Vec<u32>)>> {
    let tokenizer = Tokenizer::from_gguf(gguf)?;
    let identity = crate::tokenizer::qwen4exp_tokenizer_identity_sha256(gguf)?;
    emit(
        out,
        json!({"event": "tokenizer_binding", "identity_sha256": identity.iter()
        .map(|b| format!("{b:02x}")).collect::<String>(),
        "matches_retained_ssh_tokenizer": identity == crate::tokenizer::QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256,
        "note": "a different coherent tokenizer is allowed; its reconstructed SSH text needs source review"}),
    );
    require(
        sha256_bytes(SSH) == "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97",
        "retained SSH fixture binding changed",
    )?;
    let ids: Vec<u32> = SSH
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut bytes = Vec::new();
    for &id in &ids {
        bytes.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(i32::try_from(id)?)?);
    }
    let ssh = String::from_utf8(bytes)?;
    let roundtrip: Vec<u32> = tokenizer
        .encode(&ssh, false)?
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    emit(
        out,
        json!({"event": "ssh_source", "fixture": "2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le",
        "sha256_u32_le": sha256_bytes(SSH), "retained_token_count": ids.len(),
        "artifact_tokenizer_exact_roundtrip": ids == roundtrip,
        "extent_note": "retained 2211-token text is repeated whole; not an original uninterrupted 4096-token SSH excerpt"}),
    );
    require(
        ids == roundtrip,
        "artifact tokenizer does not exactly round-trip retained SSH fixture",
    )?;
    Ok(vec![
        ("prose", prompt(&tokenizer, CORPUS, "prose", out)?),
        (
            "ssh_repeated",
            prompt(&tokenizer, &ssh, "ssh_repeated", out)?,
        ),
    ])
}

/// Stable KL(reference softmax || candidate softmax), evaluated in f64.
fn logit_kl(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return None;
    }
    let norm = |v: &[f32]| {
        let max = v
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let z = v
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
        (max, z)
    };
    let (am, az) = norm(a);
    let (bm, bz) = norm(b);
    Some(
        a.iter()
            .zip(b)
            .map(|(&a, &b)| {
                let la = (f64::from(a) - am) - az;
                let lb = (f64::from(b) - bm) - bz;
                la.exp() * (la - lb)
            })
            .sum::<f64>()
            .max(0.0),
    )
}

#[test]
fn router_n2045_kl_diagnostic_sanity() {
    assert_eq!(logit_kl(&[1000.0, 999.0], &[1000.0, 999.0]), Some(0.0));
    assert!(logit_kl(&[1.0, 0.0], &[0.0, 1.0]).unwrap() > 0.4);
    assert!(logit_kl(&[1.0, 0.0], &[11.0, 10.0]).unwrap() < 1e-12);
    assert_eq!(logit_kl(&[f32::NAN], &[0.0]), None);
}

struct Arm {
    label: String,
    timing: Qwen4ExpPrefillTiming,
    call_wall_ms: f64,
    endpoints: Vec<Endpoint>,
    census: Vec<DispatchCensusRow>,
}

fn arm(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    end: usize,
    strict: bool,
    witness: bool,
    label: &str,
    out: &mut std::fs::File,
) -> PacketResult<Arm> {
    candidate(strict, || {
        let start = r.next_position();
        let plan = normal_plan(r, end - start)?;
        require(
            plan.packed_token_count == end - start
                && plan.packed_ranges.last().is_some_and(|v| v.len() == 2045),
            "normal production plan must end in the N2045 command",
        )?;
        emit(
            out,
            json!({"event": "arm_begin", "label": label, "candidate_n2045": strict,
            "absolute_ranges": plan.packed_ranges.iter().map(|v| [v.start, v.end]).collect::<Vec<_>>(),
            "timing_eligible": !witness, "census": witness, "profiled": false,
            "start": start, "end": end}),
        );
        if witness {
            crate::metal::dispatch_census_begin();
        }
        let clock = std::time::Instant::now();
        let result = if start == 0 {
            r.prefill(&tokens[..end])
        } else {
            r.prefill_continuation_with_command_checkpoint(&tokens[start..end], || Ok(()))
        }
        .map(|_| ());
        let call_wall_ms = clock.elapsed().as_secs_f64() * 1e3;
        let census = if witness {
            crate::metal::dispatch_census_take()
        } else {
            Vec::new()
        };
        if let Err(error) = result {
            emit(
                out,
                json!({"event": "arm_error", "label": label, "error": error.to_string(),
                "published_position": r.next_position(), "census": census_json(&census)}),
            );
            return Err(error.into());
        }
        let timing = r
            .last_prefill_timing()
            .ok_or("missing ordinary prefill timing")?;
        emit(
            out,
            json!({"event": "arm_complete", "label": label, "timing_eligible": !witness,
            "gpu_ms": timing.complete_gpu_ms(), "gpu_samples": timing.gpu_samples,
            "command_count": timing.command_count, "packed_tokens": timing.packed_token_count,
            "contains_selection": timing.contains_selection, "encode_cpu_ms": timing.encode_cpu_ms,
            "completion_wait_ms": timing.completion_wait_ms, "executor_wall_ms": timing.total_wall_ms,
            "ordinary_call_wall_ms": call_wall_ms, "census": witness.then(|| census_json(&census))}),
        );
        require(
            r.next_position() == end && timing.command_count == plan.packed_ranges.len(),
            "ordinary planner publication/command count changed",
        )?;
        let first = endpoint(r, plan.packed_ranges.last().map(|range| range.len()))?;
        emit_endpoint(out, label, &first);
        require(
            first.logits.iter().all(|v| v.is_finite()),
            "nonfinite prefill logits",
        )?;
        let mut endpoints = vec![first];
        // Teacher-forced natural handoff, outside every measured prefill. Choices
        // are compared independently; this is not a free-generation quality gate.
        if !witness {
            for step in 0..CONTINUATION {
                let token = tokens[end + step];
                let choice = argmax(&endpoints.last().unwrap().logits);
                r.forward_token(token)?;
                let e = endpoint(r, None)?;
                emit(
                    out,
                    json!({"event": "continuation", "label": label, "step": step,
                    "input_position": end + step, "teacher_token": token, "prior_greedy_choice": choice,
                    "timing_eligible": false}),
                );
                emit_endpoint(out, &format!("{label}/continuation{step}"), &e);
                require(
                    e.logits.iter().all(|v| v.is_finite()),
                    "nonfinite continuation logits",
                )?;
                endpoints.push(e);
            }
        }
        Ok(Arm {
            label: label.to_owned(),
            timing,
            call_wall_ms,
            endpoints,
            census,
        })
    })
}

fn compare(out: &mut std::fs::File, a: &Arm, b: &Arm) -> bool {
    let mut finite_and_metadata = a.endpoints.len() == b.endpoints.len();
    for (step, (a_end, b_end)) in a.endpoints.iter().zip(&b.endpoints).enumerate() {
        let choice_equal = argmax(&a_end.logits) == argmax(&b_end.logits);
        let metadata_equal = ["position", "qsa_lengths", "ple_prior_tokens"]
            .iter()
            .all(|key| a_end.state[*key] == b_end.state[*key]);
        let kl = logit_kl(&a_end.logits, &b_end.logits);
        finite_and_metadata &= metadata_equal && kl.is_some();
        emit(
            out,
            json!({"event": "comparison", "reference": a.label, "candidate": b.label,
            "step": step, "step_zero_is_prefill": true, "comparison": comparison(a_end, b_end),
            "metadata_equal": metadata_equal, "greedy_choice_equal": choice_equal,
            "logit_kl_reference_candidate": kl,
            "quality_policy": "report actual numerical/state differences; no new bit-identity or KL threshold; owner reviews differences"}),
        );
    }
    finite_and_metadata
}

fn dispatch_row(r: &DispatchCensusRow) -> Value {
    json!({"family": r.family, "tag": r.tag, "encoder": r.encoder_ordinal,
        "concurrent": r.encoder_concurrent, "kernel": r.kernel,
        "grid": [r.grid_width, r.grid_height, r.grid_depth],
        "threads": [r.threads_width, r.threads_height, r.threads_depth],
        "grid_tgs": r.grid_tgs, "tg_threads": r.tg_threads})
}

/// Only the last ordinary planner command may acquire 48 strict calls. Already
/// strict N2048 calls and all other dispatch records must remain unchanged.
fn witness(out: &mut std::fs::File, a: &Arm, b: &Arm) -> PacketResult<()> {
    let mut substitutions = Vec::new();
    let mut unexpected = Vec::new();
    let mut valid = !a.census.is_empty() && a.census.len() == b.census.len();
    for (index, (ar, br)) in a.census.iter().zip(&b.census).enumerate() {
        if dispatch_row(ar) == dispatch_row(br) {
            continue;
        }
        let only_router = ar.kernel == "kernel_mat_mat_f32_f32"
            && br.kernel == "kernel_mat_mat_f32_f32_router_e8p32_strict"
            && ar.family == br.family
            && ar.tag == br.tag
            && ar.encoder_ordinal == br.encoder_ordinal
            && ar.encoder_ordinal == (a.timing.command_count - 1) as u64
            && !ar.encoder_concurrent
            && !br.encoder_concurrent
            && (ar.grid_width, br.grid_width) == (512, 64)
            && ar.grid_height == 64
            && br.grid_height == 64
            && ar.grid_depth == 1
            && br.grid_depth == 1
            && (ar.threads_width, ar.threads_height, ar.threads_depth) == (32, 1, 1)
            && (br.threads_width, br.threads_height, br.threads_depth) == (32, 1, 1)
            && ar.grid_tgs == 512 * 64
            && br.grid_tgs == 64 * 64
            && ar.tg_threads == 32
            && br.tg_threads == 32;
        valid &= only_router;
        let row = json!({"dispatch_index": index, "baseline": dispatch_row(ar), "candidate": dispatch_row(br)});
        if only_router {
            substitutions.push(row);
        } else if unexpected.len() < 16 {
            unexpected.push(row);
        }
    }
    valid &= substitutions.len() == 48;
    emit(
        out,
        json!({"event": "dispatch_witness", "reference": a.label, "candidate": b.label,
        "baseline_dispatches": a.census.len(), "candidate_dispatches": b.census.len(),
        "new_strict_calls": substitutions.len(), "expected_new_strict_calls": 48,
        "only_last_command_changed": valid, "substitutions": substitutions,
        "unexpected_first16": unexpected, "valid": valid, "timing_eligible": false}),
    );
    require(
        valid,
        "dispatch witness failed: expected exactly 48 router substitutions in final N2045 command",
    )
}

#[derive(serde::Serialize)]
struct AbbaScreen {
    label: String,
    whole4096_performance_gate_met: Option<bool>,
    finite_and_metadata_preserved: bool,
}

fn abba(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    label: &str,
    out: &mut std::fs::File,
    mut prepare: impl FnMut(&mut Qwen4ExpTextRunner<'_, '_, '_>) -> PacketResult<()>,
) -> PacketResult<AbbaScreen> {
    let mut warm = Vec::new();
    for (name, strict) in [("warm_A", false), ("warm_B", true)] {
        prepare(r)?;
        warm.push(arm(
            r,
            tokens,
            END,
            strict,
            true,
            &format!("{label}/{name}"),
            out,
        )?);
    }
    witness(out, &warm[0], &warm[1])?;
    let mut quality_observations_ok = compare(out, &warm[0], &warm[1]);
    drop(warm);
    let mut timed = Vec::new();
    for (name, strict) in [("A1", false), ("B1", true), ("B2", true), ("A2", false)] {
        let before = std::time::Instant::now();
        prepare(r)?;
        emit(
            out,
            json!({"event": "prepare", "label": format!("{label}/{name}"),
            "reset_or_restore_wall_ms": before.elapsed().as_secs_f64() * 1e3,
            "outside_measured_prefill": true}),
        );
        let current = arm(
            r,
            tokens,
            END,
            strict,
            false,
            &format!("{label}/{name}"),
            out,
        )?;
        if let Some(reference) = timed.first() {
            quality_observations_ok &= compare(out, reference, &current);
        }
        timed.push(current);
    }
    quality_observations_ok &= compare(out, &timed[1], &timed[2]);
    let gpu: Option<Vec<f64>> = timed.iter().map(|v| v.timing.complete_gpu_ms()).collect();
    let wall: Vec<f64> = timed.iter().map(|v| v.call_wall_ms).collect();
    let pairs = |v: &[f64]| {
        json!({"A1": v[0], "B1": v[1], "B2": v[2], "A2": v[3],
        "mean_fraction_saved": 1.0 - (v[1] + v[2]) / (v[0] + v[3]),
        "pair1_fraction_saved": 1.0 - v[1] / v[0], "pair2_fraction_saved": 1.0 - v[2] / v[3],
        "A2_over_A1": v[3] / v[0], "B2_over_B1": v[2] / v[1]})
    };
    let gpu_gate = gpu.as_ref().is_some_and(|v| {
        v.iter().all(|x| x.is_finite() && *x > 0.0)
            && (v[1] + v[2]) <= 0.97 * (v[0] + v[3])
            && v[1] < v[0]
            && v[2] < v[3]
    });
    let wall_gate =
        wall.iter().all(|x| x.is_finite() && *x > 0.0) && wall[1] <= wall[0] && wall[2] <= wall[3];
    let whole = timed[0].timing.token_count == END;
    emit(
        out,
        json!({"event": "abba", "label": label, "gpu_ms": gpu.as_ref().map(|v| pairs(v)),
        "ordinary_call_wall_ms": pairs(&wall), "whole4096_performance_gate_met": whole.then_some(gpu_gate && wall_gate),
        "observed_finite_and_metadata_preserved": quality_observations_ok,
        "interpretation": "suffix saving is included in whole4096 saving; do not add them; no placement or profiled-arm claim",
        "decision": "owner_qualification_required_even_if_performance_gate_met"}),
    );
    Ok(AbbaScreen {
        label: label.into(),
        whole4096_performance_gate_met: whole.then_some(gpu_gate && wall_gate),
        finite_and_metadata_preserved: quality_observations_ok,
    })
}

fn packet(out: &mut std::fs::File) -> PacketResult<()> {
    with_native_artifact(out, |ctx, gguf, out| {
        require(
            crate::qwen4exp_moe::packed_router_e8p32_strict_supported(ctx),
            "strict router pipeline/device guard declined; no candidate measurement",
        )?;
        let prompts = prompts(gguf, out)?;
        with_native_runner(
            ctx,
            gguf,
            ADMISSION_EXTENT,
            DIAGNOSTIC_MARGIN,
            out,
            |r, out| {
                let mut whole_pass = Vec::new();
                let mut suffix_observations = Vec::new();
                for (id, tokens) in &prompts {
                    whole_pass.push(abba(r, tokens, &format!("{id}/fresh4096"), out, |r| {
                        r.reset()?;
                        zero_persistent_state(r);
                        Ok(())
                    })?);
                    r.reset()?;
                    zero_persistent_state(r);
                    candidate(false, || r.prefill(&tokens[..SUFFIX_START]).map(|_| ()))?;
                    let clock = std::time::Instant::now();
                    let checkpoint = r.workspace.checkpoint_for_tests();
                    emit(
                        out,
                        json!({"event": "checkpoint", "prompt": id, "position": SUFFIX_START,
                    "capture_wall_ms": clock.elapsed().as_secs_f64() * 1e3, "outside_timed_arms": true}),
                    );
                    suffix_observations.push(abba(
                        r,
                        tokens,
                        &format!("{id}/suffix2045_at2051"),
                        out,
                        |r| {
                            r.workspace.restore_checkpoint_for_tests(&checkpoint);
                            Ok(())
                        },
                    )?);
                }
                // One wider extent, one stream, census/semantic sample only. This
                // verifies repeated full chunks still leave exactly one N2045 tail;
                // it is intentionally not an extra wide performance matrix.
                let mut wide = Vec::new();
                for (name, strict) in [("A", false), ("B", true)] {
                    r.reset()?;
                    zero_persistent_state(r);
                    wide.push(arm(
                        r,
                        &prompts[0].1,
                        WIDE,
                        strict,
                        true,
                        &format!("prose/wide8192_witness/{name}"),
                        out,
                    )?);
                }
                witness(out, &wide[0], &wide[1])?;
                let wide_quality = compare(out, &wide[0], &wide[1]);
                emit(
                    out,
                    json!({"event": "qualification_summary", "whole4096_streams": whole_pass,
                 "whole4096_performance_gate_met_on_both": whole_pass.iter().all(|v| v.whole4096_performance_gate_met == Some(true)),
                 "suffix_observations": suffix_observations,
                 "finite_and_metadata_preserved_all_screens": whole_pass.iter().chain(&suffix_observations).all(|v| v.finite_and_metadata_preserved),
                 "wide_sample_finite_and_metadata_preserved": wide_quality,
                 "promotion": false, "decision": "owner reviews raw differences and drift; this diagnostic does not decide promotion"}),
                );
                Ok(())
            },
        )
    })
}

#[test]
#[ignore = "production lease; arbitrary FLASH_PREFILL_MODEL and new FLASH_PREFILL_OUT; release ABBA only"]
fn native_router_n2045() {
    run_packet(
        "flash.router_n2045.v2",
        include_bytes!("router_2045.rs"),
        json!({
            "candidate": "production N2045 strict router versus explicit historical generic override; planner unchanged",
            "hyper_observation": "v2 observes the actual final packed row; v1 incorrectly hashed stale singleton scratch at packed endpoints",
            "rollback": "QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0 remains authoritative; packet requires it enabled to measure candidate",
            "performance_predeclared": "each natural stream: >=3% mean whole4096 GPU saving, both ABBA GPU pairs improve, both ordinary-call wall pairs nonregress",
            "continuation_predeclared": "four teacher-forced natural tokens; finite outputs, greedy choices and causal metadata reported; numerical/state differences and KL retained for owner review, no invented bit-identity gate",
            "allocation": "one resident native session, full8196 admission; production gate plus one full checkpoint and 96MiB diagnostic CPU margin; no scalar retry on refusal",
            "bounded_work": "two streams, fresh4096 and suffix2045 each warm A/B census then unprofiled ABBA with four continuations; one prose8192 A/B census sample",
        "measurement": "ordinary prefill call wall plus summed command GPU; reset, fresh-arm persistent zeroing, checkpoint, hashing, output and continuation excluded; warm census timing ineligible; not placement-cold TTFT"
        }),
        packet,
    );
}
