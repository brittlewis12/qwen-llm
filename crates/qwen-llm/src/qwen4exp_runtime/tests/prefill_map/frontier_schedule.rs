//! Ordinary-driver frontier scheduling diagnostic, both arms production router.
//! FLASH_FRONTIER_STAGE=suffix (default), whole, or both;
//! FLASH_FRONTIER_CORPORA=prose (default) or both. Uses FLASH_PREFILL_MODEL and
//! a NEW FLASH_PREFILL_OUT. Two ABBA rounds follow an untimed census warm pair.
//! No phase profiling, no manual packed-command driver, no production promotion.

use super::*;

#[path = "frontier_layer0.rs"]
mod frontier_layer0;

const CAPACITY: usize = END + 4;
const MARGIN: u64 = 96 << 20;
const SSH: &[u8] = include_bytes!(
    "../../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
);

fn ranges(start: usize, candidate: bool) -> Vec<std::ops::Range<usize>> {
    match (start, candidate) {
        (0, false) => vec![0..2048, 2048..2051, 2051..4096],
        (0, true) => vec![0..2048, 2048..4096],
        (2048, false) => vec![2048..2051, 2051..4096],
        (2048, true) => vec![2048..4096],
        _ => panic!("unsupported diagnostic start"),
    }
}

#[test]
fn frontier_schedule_scope_and_plans() {
    let plan = |start, n, capacity, selected, frontier| {
        plan_qwen4exp_prefill_execution_from(start, n, capacity, selected, frontier).unwrap()
    };
    with_qwen4exp_frontier_schedule(false, || {
        for start in [0, 2048] {
            let a = plan(start, END - start, Some(2048), true, 2051);
            assert_eq!(a.packed_ranges, ranges(start, false));
            with_qwen4exp_frontier_schedule(true, || {
                let b = plan(start, END - start, Some(2048), true, 2051);
                assert_eq!(b.packed_ranges, ranges(start, true));
                assert_eq!(b.packed_token_count, END - start);
                assert_eq!(b.scalar_start, END);
                assert!(b.contains_selection);
                with_qwen4exp_frontier_schedule(false, || {
                    assert_eq!(plan(start, END - start, Some(2048), true, 2051), a);
                });
                assert_eq!(plan(start, END - start, Some(2048), true, 2051), b);
            });
            assert_eq!(plan(start, END - start, Some(2048), true, 2051), a);
        }
        for (start, n, cap, selected, frontier) in [
            (0, 4095, Some(2048), true, 2051),
            (0, 4097, Some(2048), true, 2051),
            (2048, 2047, Some(2048), true, 2051),
            (2049, 2048, Some(2048), true, 2051),
            (2051, 2045, Some(2048), true, 2051),
            (0, 4096, Some(1024), true, 2051),
            (0, 4096, Some(4096), true, 2051),
            (0, 4096, Some(2048), false, 2051),
            (0, 4096, None, false, 2051),
            (0, 4096, Some(2048), true, 2050),
            (0, 4096, Some(2048), true, 2052),
        ] {
            let a = plan(start, n, cap, selected, frontier);
            assert_eq!(
                with_qwen4exp_frontier_schedule(true, || plan(start, n, cap, selected, frontier)),
                a
            );
        }
        let unwind = std::panic::catch_unwind(|| {
            with_qwen4exp_frontier_schedule(true, || panic!("scope probe"))
        });
        assert!(unwind.is_err());
        assert!(!QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get()));
        with_qwen4exp_frontier_schedule(true, || {
            assert!(
                std::panic::catch_unwind(|| with_qwen4exp_frontier_schedule(false, || panic!(
                    "nested scope probe"
                )))
                .is_err()
            );
            assert!(QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get()));
            assert!(
                !std::thread::spawn(|| QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get()))
                    .join()
                    .unwrap()
            );
        });
    });
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
    while tokens.len() < CAPACITY && repetitions < 32 {
        if repetitions != 0 {
            text.push_str("\n\n");
        }
        text.push_str(source);
        repetitions += 1;
        tokens = tokenizer
            .encode(&text, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
    }
    require(
        tokens.len() >= CAPACITY,
        "natural text too short after 32 whole-text repetitions",
    )?;
    tokens.truncate(CAPACITY);
    emit(
        out,
        json!({"event":"prompt", "id":id, "source_text_sha256":sha256_bytes(source.as_bytes()),
        "full_text":text,"full_text_sha256":sha256_bytes(text.as_bytes()),"whole_text_repetitions":repetitions,
        "separator":"two newlines","add_special_tokens":false,"used_token_ids":tokens,
        "used_token_ids_sha256_u32_le":sha256_u32_le(b"", &tokens),
        "used_extent":CAPACITY,"prefix2048_sha256":sha256_u32_le(b"", &tokens[..PREFIX]),
        "prompt4096_sha256":sha256_u32_le(b"", &tokens[..END])}),
    );
    Ok(tokens)
}

fn prompts(
    gguf: &GgufFile,
    both: bool,
    out: &mut std::fs::File,
) -> PacketResult<Vec<(&'static str, Vec<u32>)>> {
    let tokenizer = Tokenizer::from_gguf(gguf)?;
    let identity = crate::tokenizer::qwen4exp_tokenizer_identity_sha256(gguf)?;
    let identity_matches = identity == crate::tokenizer::QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256;
    emit(
        out,
        json!({"event":"tokenizer_binding","identity_sha256":identity.iter().map(|b|format!("{b:02x}")).collect::<String>(),
        "matches_retained_ssh_tokenizer":identity_matches}),
    );
    let mut result = vec![("prose", prompt(&tokenizer, CORPUS, "prose", out)?)];
    if both {
        require(
            identity_matches,
            "SSH fixture requires the released tokenizer identity",
        )?;
        require(
            sha256_bytes(SSH) == "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97",
            "SSH fixture hash changed",
        )?;
        let ids: Vec<u32> = SSH
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut bytes = Vec::new();
        for &id in &ids {
            bytes.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(i32::try_from(id)?)?);
        }
        let text = String::from_utf8(bytes)?;
        let roundtrip: Vec<u32> = tokenizer
            .encode(&text, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        emit(
            out,
            json!({"event":"ssh_source","fixture_sha256":sha256_bytes(SSH),"retained_tokens":ids.len(),
            "artifact_tokenizer_exact_roundtrip":ids==roundtrip,
            "label":"ssh_repeated","extent_note":"whole retained natural SSH text repeated; not an uninterrupted 4096-token source excerpt"}),
        );
        require(ids == roundtrip, "SSH fixture tokenizer roundtrip failed")?;
        result.push((
            "ssh_repeated",
            prompt(&tokenizer, &text, "ssh_repeated", out)?,
        ));
    }
    Ok(result)
}

fn kl(a: &[f32], b: &[f32]) -> f64 {
    let logz = |x: &[f32]| {
        let m = x
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        (
            m,
            x.iter()
                .map(|&v| (f64::from(v) - m).exp())
                .sum::<f64>()
                .ln(),
        )
    };
    let ((am, az), (bm, bz)) = (logz(a), logz(b));
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let a = (f64::from(a) - am) - az;
            a.exp() * (a - ((f64::from(b) - bm) - bz))
        })
        .sum::<f64>()
        .max(0.0)
}

struct Arm {
    label: String,
    gpu_ms: f64,
    wall_ms: f64,
    endpoints: Vec<Endpoint>,
}

fn compare(out: &mut std::fs::File, a: &Arm, b: &Arm) -> PacketResult<()> {
    require(
        a.endpoints.len() == b.endpoints.len(),
        "endpoint count differs",
    )?;
    for (step, (x, y)) in a.endpoints.iter().zip(&b.endpoints).enumerate() {
        require(
            !x.logits.is_empty()
                && x.logits.len() == y.logits.len()
                && x.logits.iter().chain(&y.logits).all(|v| v.is_finite()),
            "comparison requires equal nonempty finite logits",
        )?;
        let metadata_equal = ["position", "qsa_lengths", "ple_prior_tokens"]
            .iter()
            .all(|k| x.state[*k] == y.state[*k]);
        let (at, bt) = (argmax(&x.logits), argmax(&y.logits));
        let (ab, ba) = (kl(&x.logits, &y.logits), kl(&y.logits, &x.logits));
        emit(
            out,
            json!({"event":"comparison","reference":a.label,"candidate":b.label,"step":step,
            "comparison":comparison(x,y),"causal_metadata_equal":metadata_equal,
            "kl_reference_candidate":ab,"kl_candidate_reference":ba,
            "candidate_choice_regret_reference_logits":f64::from(x.logits[at])-f64::from(x.logits[bt]),
            "reference_choice_regret_candidate_logits":f64::from(y.logits[bt])-f64::from(y.logits[at]),
            "quality_policy":"finite and causal metadata checked; logits and persistent hashes reported without bit or numerical threshold gate"}),
        );
        require(
            metadata_equal && ab.is_finite() && ba.is_finite(),
            "nonfinite comparison or causal metadata mismatch",
        )?;
    }
    Ok(())
}

fn witness(
    out: &mut std::fs::File,
    label: &str,
    plan: &Qwen4ExpPrefillExecutionPlan,
    rows: &[DispatchCensusRow],
) -> PacketResult<()> {
    let mut valid = true;
    let mut commands = Vec::new();
    for (index, range) in plan.packed_ranges.iter().enumerate() {
        let strict: Vec<_> = rows
            .iter()
            .filter(|r| {
                r.encoder_ordinal == index as u64
                    && r.kernel == "kernel_mat_mat_f32_f32_router_e8p32_strict"
            })
            .collect();
        let generic_n3 = rows
            .iter()
            .filter(|r| {
                r.encoder_ordinal == index as u64
                    && r.kernel == "kernel_mat_mat_f32_f32"
                    && (r.grid_width, r.grid_height, r.grid_depth) == (512, 1, 1)
                    && r.tg_threads == 32
            })
            .count();
        let expected = if range.len() == 3 { 0 } else { 48 };
        valid &= strict.len() == expected
            && strict.iter().all(|r| {
                (r.grid_width, r.grid_height, r.grid_depth) == (64, 64, 1)
                    && r.tg_threads == 32
                    && !r.encoder_concurrent
            });
        if range.len() == 3 {
            valid &= generic_n3 == 48;
        }
        commands.push(json!({"absolute_range":[range.start,range.end],"rows":range.len(),"encoder":index,
            "strict_router_calls":strict.len(),"expected_strict_router_calls":expected,"generic_n3_router_shape_calls":generic_n3}));
    }
    let total = rows
        .iter()
        .filter(|r| r.kernel == "kernel_mat_mat_f32_f32_router_e8p32_strict")
        .count();
    let expected = if plan.packed_token_count == END {
        96
    } else {
        48
    };
    valid &= total == expected;
    emit(
        out,
        json!({"event":"warm_dispatch_witness","label":label,"commands":commands,"strict_router_calls":total,
        "expected_strict_router_calls":expected,"valid":valid,"all_kernels":census_json(rows),
        "router_policy":"both arms current production; no N2045 or global router override",
        "generic_n3_identification":"generic F32 kernel, output512, one token tile, 32 threads, in the three-row command"}),
    );
    require(valid, "production router census mismatch")
}

fn arm(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    candidate: bool,
    warm: bool,
    label: &str,
    out: &mut std::fs::File,
) -> PacketResult<Arm> {
    with_qwen4exp_frontier_schedule(candidate, || {
        let start = r.next_position();
        let plan = normal_plan(r, END - start)?;
        require(
            plan.packed_ranges == ranges(start, candidate)
                && plan.scalar_start == END
                && plan.contains_selection,
            "ordinary plan differs from scoped schedule",
        )?;
        emit(
            out,
            json!({"event":"arm_begin","label":label,"candidate_schedule":candidate,"start":start,"end":END,
            "absolute_ranges":plan.packed_ranges.iter().map(|r|[r.start,r.end]).collect::<Vec<_>>(),
            "final_packed_rows":plan.packed_ranges.last().unwrap().len(),"census":warm,"timing_eligible":!warm}),
        );
        require(
            !crate::metal::dispatch_census_is_active(),
            "external census active",
        )?;
        if warm {
            crate::metal::dispatch_census_begin();
        }
        let clock = std::time::Instant::now();
        let result = if start == 0 {
            r.prefill(&tokens[..END]).map(|_| ())
        } else {
            r.prefill_continuation_with_command_checkpoint(&tokens[start..END], || Ok(()))
                .map(|_| ())
        };
        let wall_ms = clock.elapsed().as_secs_f64() * 1e3;
        let census = if warm {
            crate::metal::dispatch_census_take()
        } else {
            Vec::new()
        };
        if let Err(error) = result {
            emit(
                out,
                json!({"event":"arm_error","label":label,"error":error.to_string(),"published_position":r.next_position()}),
            );
            return Err(error.into());
        }
        let timing = r.last_prefill_timing().ok_or("missing ordinary timing")?;
        let gpu = timing.complete_gpu_ms();
        let valid = gpu.is_some_and(|v| v.is_finite() && v > 0.0)
            && timing.gpu_samples == plan.packed_ranges.len();
        emit(
            out,
            json!({"event":"arm_complete","label":label,"timing_eligible":!warm,
            "ordinary_call_wall_ms":wall_ms,"complete_gpu_ms":gpu,"gpu_sum_ms_raw":timing.gpu_ms,
            "gpu_valid":valid,"gpu_samples":timing.gpu_samples,"command_count":timing.command_count,
            "packed_tokens":timing.packed_token_count,"contains_selection":timing.contains_selection,
            "encode_cpu_ms":timing.encode_cpu_ms,"completion_wait_ms":timing.completion_wait_ms,
            "executor_wall_ms":timing.total_wall_ms}),
        );
        require(
            valid && wall_ms.is_finite() && wall_ms > 0.0,
            "invalid complete GPU or wall timing; raw attempt flushed",
        )?;
        require(
            r.next_position() == END
                && timing.command_count == plan.packed_ranges.len()
                && timing.token_count == END - start
                && timing.packed_token_count == END - start
                && timing.contains_selection,
            "publication or command count mismatch",
        )?;
        if warm {
            witness(out, label, &plan, &census)?;
        }
        let first = endpoint(r, Some(plan.packed_ranges.last().unwrap().len()))?;
        emit_endpoint(out, label, &first);
        require(
            !first.logits.is_empty() && first.logits.iter().all(|v| v.is_finite()),
            "invalid prefill logits",
        )?;
        let mut endpoints = vec![first];
        if !warm {
            for step in 0..4 {
                r.forward_token(tokens[END + step])?;
                let current = endpoint(r, None)?;
                emit(
                    out,
                    json!({"event":"continuation","label":label,"step":step+1,"input_position":END+step,
                    "teacher_token":tokens[END+step],"timing_eligible":false}),
                );
                emit_endpoint(out, &format!("{label}/continuation{}", step + 1), &current);
                require(
                    r.next_position() == END + step + 1
                        && current.logits.iter().all(|v| v.is_finite()),
                    "invalid continuation",
                )?;
                endpoints.push(current);
            }
        }
        Ok(Arm {
            label: label.into(),
            gpu_ms: gpu.unwrap(),
            wall_ms,
            endpoints,
        })
    })
}

fn screen(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    label: &str,
    out: &mut std::fs::File,
    mut prepare: impl FnMut(&mut Qwen4ExpTextRunner<'_, '_, '_>) -> PacketResult<()>,
) -> PacketResult<()> {
    let mut warm = Vec::new();
    for (name, candidate) in [("warm_A", false), ("warm_B", true)] {
        prepare(r)?;
        warm.push(arm(
            r,
            tokens,
            candidate,
            true,
            &format!("{label}/{name}"),
            out,
        )?);
    }
    compare(out, &warm[0], &warm[1])?;
    drop(warm);
    for round in 1..=2 {
        let mut attempts = Vec::new();
        for (name, candidate) in [("A1", false), ("B1", true), ("B2", true), ("A2", false)] {
            let clock = std::time::Instant::now();
            prepare(r)?;
            let arm_label = format!("{label}/round{round}/{name}");
            emit(
                out,
                json!({"event":"prepare","label":arm_label,"wall_ms":clock.elapsed().as_secs_f64()*1e3,"outside_prefill_timing":true}),
            );
            let current = arm(r, tokens, candidate, false, &arm_label, out)?;
            if let Some(reference) = attempts.first() {
                compare(out, reference, &current)?;
            }
            attempts.push(current);
        }
        compare(out, &attempts[1], &attempts[2])?;
        let pairs = |v: Vec<f64>| {
            json!({"A1":v[0],"B1":v[1],"B2":v[2],"A2":v[3],
            "pair_savings":[1.0-v[1]/v[0],1.0-v[2]/v[3]],"mean_saving":1.0-(v[1]+v[2])/(v[0]+v[3]),
            "A2_over_A1":v[3]/v[0],"B2_over_B1":v[2]/v[1]})
        };
        emit(
            out,
            json!({"event":"abba","label":label,"round":round,
            "gpu_ms":pairs(attempts.iter().map(|a|a.gpu_ms).collect()),
            "ordinary_call_wall_ms":pairs(attempts.iter().map(|a|a.wall_ms).collect()),
            "decision":"unscored; suffix and whole savings overlap, do not add"}),
        );
    }
    Ok(())
}

fn packet(out: &mut std::fs::File, stage: &str, both: bool) -> PacketResult<()> {
    with_native_artifact(out, |ctx, gguf, out| {
        require(
            crate::qwen4exp_moe::packed_router_e8p32_strict_supported(ctx),
            "production strict router pipeline unsupported",
        )?;
        let prompts = prompts(gguf, both, out)?;
        with_native_runner(ctx, gguf, CAPACITY, MARGIN, out, |r, out| {
            require(
                r.packed_qsa_dense_end()? == 2051,
                "packet requires dense frontier2051",
            )?;
            for (id, tokens) in &prompts {
                if stage != "whole" {
                    r.reset()?;
                    zero_persistent_state(r);
                    with_qwen4exp_frontier_schedule(false, || {
                        r.prefill(&tokens[..PREFIX]).map(|_| ())
                    })?;
                    let initial = endpoint(r, Some(PREFIX))?;
                    emit_endpoint(out, &format!("{id}/checkpoint2048"), &initial);
                    drop(initial);
                    let clock = std::time::Instant::now();
                    let checkpoint = r.workspace.checkpoint_for_tests();
                    emit(
                        out,
                        json!({"event":"checkpoint","prompt":id,"position":PREFIX,
                        "capture_wall_ms":clock.elapsed().as_secs_f64()*1e3,"outside_timing":true}),
                    );
                    screen(r, tokens, &format!("{id}/suffix_at2048"), out, |r| {
                        r.workspace.restore_checkpoint_for_tests(&checkpoint);
                        require(r.next_position() == PREFIX, "checkpoint frontier mismatch")
                    })?;
                }
                // The suffix checkpoint has dropped before any whole request.
                if stage != "suffix" {
                    screen(r, tokens, &format!("{id}/whole4096"), out, |r| {
                        r.reset()?;
                        zero_persistent_state(r);
                        Ok(())
                    })?;
                }
            }
            Ok(())
        })
    })
}

#[test]
#[ignore = "release; normal production lease/admission; FLASH_PREFILL_MODEL and NEW FLASH_PREFILL_OUT"]
fn native_frontier_schedule() {
    let stage = std::env::var("FLASH_FRONTIER_STAGE").unwrap_or_else(|_| "suffix".into());
    let corpora = std::env::var("FLASH_FRONTIER_CORPORA").unwrap_or_else(|_| "prose".into());
    assert!(["suffix", "whole", "both"].contains(&stage.as_str()));
    assert!(["prose", "both"].contains(&corpora.as_str()));
    run_packet(
        "flash.frontier_schedule.v1",
        include_bytes!("frontier_schedule.rs"),
        json!({
            "stage":stage,"corpora":corpora,"rounds":2,"capacity":CAPACITY,"packed_capacity":2048,"frontier":2051,
            "A":"production planner: suffix3+2045 / whole2048+3+2045",
            "B":"scoped planner only: suffix2048 / whole2048+2048",
            "router":"both arms current production, including strict N2045; no router override",
            "scope":"cfg(test), thread-local, unwind-safe; only selected-capable capacity2048 frontier2051 requests (2048,2048) and (0,4096)",
            "admission":"native combined load includes selected scratch for4100; normal CPU gate for one checkpoint plus96MiB diagnostic margin",
            "measurement":"ordinary driver, no command splitting; complete valid command GPU sum and ordinary-call wall; reset/restore, endpoint readback/hashes, JSON and four continuations outside timing; warm census excluded",
            "timestamp_limit":"existing aggregate and sample count; individual raw timestamps not exposed; incomplete or invalid aggregate fails",
            "quality":"finite logits, bidirectional KL, relative L2, choice regrets, QSA/PLE/position metadata and persistent allocation hashes; no bit gate; final packed rows2045 versus2048",
            "source_prompt":"retained roadmap prose (GSQ is selected by artifact path); optional identity-checked SSH repeated whole and labeled; actual4100 token IDs bound",
            "production_change":false
        }),
        |out| packet(out, &stage, corpora == "both"),
    );
}
