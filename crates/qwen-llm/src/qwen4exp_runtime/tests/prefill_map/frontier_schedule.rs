//! Ordinary-driver frontier scheduling diagnostic, both arms production router.
//! FLASH_FRONTIER_STAGE=suffix (default), whole, or both;
//! FLASH_FRONTIER_CORPORA=prose (default) or both. Uses FLASH_PREFILL_MODEL and
//! a NEW FLASH_PREFILL_OUT. FLASH_FRONTIER_CANDIDATE=forced (default) or production;
//! FLASH_FRONTIER_ROUNDS is positive (default 2), after an untimed census warm pair.
//! A always forces the incumbent; production B uses the landed production policy.
//! No phase profiling or manual packed-command driver; this packet changes no defaults.

use super::*;

#[path = "frontier_layer0.rs"]
mod frontier_layer0;

#[path = "frontier_quality.rs"]
mod frontier_quality;

const CAPACITY: usize = END + 4;
const MARGIN: u64 = 96 << 20;
const SSH: &[u8] = include_bytes!(
    "../../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrontierCandidateMode {
    Forced,
    Production,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrontierOptions {
    candidate: FrontierCandidateMode,
    rounds: usize,
}

impl FrontierOptions {
    fn parse(candidate: Option<&str>, rounds: Option<&str>) -> PacketResult<Self> {
        let candidate = match candidate.unwrap_or("forced") {
            "forced" => FrontierCandidateMode::Forced,
            "production" => FrontierCandidateMode::Production,
            _ => return Err("FLASH_FRONTIER_CANDIDATE must be forced or production".into()),
        };
        let rounds = rounds.unwrap_or("2").parse::<usize>()?;
        require(rounds > 0, "FLASH_FRONTIER_ROUNDS must be positive")?;
        Ok(Self { candidate, rounds })
    }

    fn candidate_label(self) -> &'static str {
        match self.candidate {
            FrontierCandidateMode::Forced => "forced",
            FrontierCandidateMode::Production => "production",
        }
    }

    fn arm_override(self, candidate: bool) -> Option<bool> {
        if !candidate {
            Some(false)
        } else {
            match self.candidate {
                FrontierCandidateMode::Forced => Some(true),
                FrontierCandidateMode::Production => None,
            }
        }
    }

    fn with_arm<R>(self, candidate: bool, work: impl FnOnce() -> R) -> R {
        with_qwen4exp_frontier_schedule_override(self.arm_override(candidate), work)
    }
}

#[test]
fn frontier_schedule_options_parse_and_defaults() {
    let defaults = FrontierOptions::parse(None, None).unwrap();
    assert_eq!(defaults.candidate, FrontierCandidateMode::Forced);
    assert_eq!(defaults.rounds, 2);
    assert_eq!(defaults.candidate_label(), "forced");
    assert_eq!(defaults.arm_override(false), Some(false));
    assert_eq!(defaults.arm_override(true), Some(true));
    let production = FrontierOptions::parse(Some("production"), Some("1")).unwrap();
    assert_eq!(production.rounds, 1);
    assert_eq!(production.candidate_label(), "production");
    assert_eq!(production.arm_override(false), Some(false));
    assert_eq!(production.arm_override(true), None);
    assert_eq!(
        FrontierOptions::parse(Some("forced"), None).unwrap(),
        defaults
    );
    assert_eq!(
        FrontierOptions::parse(Some("production"), None)
            .unwrap()
            .rounds,
        2
    );
    assert_eq!(FrontierOptions::parse(None, Some("3")).unwrap().rounds, 3);
    for mode in ["", "auto", "true", "Production", " production"] {
        assert!(FrontierOptions::parse(Some(mode), None).is_err());
    }
    for rounds in ["", "0", "-1", "1.5", "one", " 1", "1 "] {
        assert!(FrontierOptions::parse(None, Some(rounds)).is_err());
    }
    let overflow = format!("{}0", usize::MAX);
    assert!(FrontierOptions::parse(None, Some(&overflow)).is_err());
}

#[test]
fn frontier_schedule_options_scope_and_plans() {
    let state = || QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get());
    let initial = state();
    for mode in ["forced", "production"] {
        let options = FrontierOptions::parse(Some(mode), None).unwrap();
        for outer in [None, Some(false), Some(true)] {
            with_qwen4exp_frontier_schedule_override(outer, || {
                for start in [0, 2048] {
                    for candidate in [false, true] {
                        options.with_arm(candidate, || {
                            assert_eq!(state(), options.arm_override(candidate));
                            let plan = plan_qwen4exp_prefill_execution_from(
                                start,
                                END - start,
                                Some(2048),
                                true,
                                2051,
                            )
                            .unwrap();
                            assert_eq!(plan.packed_ranges, ranges(start, candidate));
                        });
                        assert_eq!(state(), outer);
                    }
                }
            });
            assert_eq!(state(), initial);
        }
    }
}

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
fn frontier_schedule_production_and_forced_plans() {
    assert_eq!(QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get()), None);
    for start in [0, 2048] {
        let plan = || {
            plan_qwen4exp_prefill_execution_from(start, END - start, Some(2048), true, 2051)
                .unwrap()
        };
        let production = plan();
        assert_eq!(production.packed_ranges, ranges(start, true));
        assert_eq!(production.packed_token_count, END - start);
        assert_eq!(production.scalar_start, END);
        assert!(production.contains_selection);
        assert_eq!(
            with_qwen4exp_frontier_schedule_override(None, plan),
            production
        );
        assert_eq!(with_qwen4exp_frontier_schedule(true, plan), production);
        let incumbent = with_qwen4exp_frontier_schedule(false, plan);
        assert_eq!(incumbent.packed_ranges, ranges(start, false));
        assert_eq!(incumbent.packed_token_count, production.packed_token_count);
        assert_eq!(incumbent.scalar_start, production.scalar_start);
        assert_eq!(incumbent.contains_selection, production.contains_selection);
        assert_ne!(incumbent, production);
        assert_eq!(plan(), production);
    }
}

#[test]
fn frontier_schedule_excludes_other_geometries() {
    for start in [0, 1, 2047, 2048, 2049, 2051] {
        for n in [1, 2045, 2047, 2048, 2049, 4095, 4096, 4097] {
            for cap in [
                None,
                Some(1024),
                Some(2047),
                Some(2048),
                Some(2049),
                Some(4096),
            ] {
                for selected in [false, true] {
                    for frontier in [2050, 2051, 2052] {
                        if (cap.is_none() && selected)
                            || (cap == Some(2048)
                                && selected
                                && frontier == 2051
                                && matches!((start, n), (0, 4096) | (2048, 2048)))
                        {
                            continue;
                        }
                        let plan = || {
                            plan_qwen4exp_prefill_execution_from(start, n, cap, selected, frontier)
                                .unwrap()
                        };
                        let incumbent = with_qwen4exp_frontier_schedule(false, plan);
                        for mode in [None, Some(true)] {
                            assert_eq!(
                                with_qwen4exp_frontier_schedule_override(mode, plan),
                                incumbent,
                                "mode={mode:?} start={start} n={n} cap={cap:?} selected={selected} frontier={frontier}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn frontier_schedule_preserves_validation_order_and_overflow_checks() {
    for (start, n, cap, selected, frontier, message) in [
        (usize::MAX, 0, None, true, 0, "at least one token"),
        (usize::MAX, 1, None, true, 0, "nonzero QSA dense end"),
        (usize::MAX, 1, None, true, 2051, "requires packed scratch"),
        (usize::MAX, 1, None, false, 2051, "position range overflow"),
        (
            usize::MAX,
            1,
            Some(0),
            true,
            2051,
            "position range overflow",
        ),
        (
            usize::MAX - 1,
            2,
            Some(2048),
            true,
            2051,
            "position range overflow",
        ),
        (0, 0, Some(2048), true, 2051, "at least one token"),
        (2048, 0, Some(2048), true, 2051, "at least one token"),
        (0, 4096, Some(0), true, 2051, "smaller than two tokens"),
        (2048, 2048, Some(1), true, 2051, "smaller than two tokens"),
        (0, 4096, Some(2048), true, 0, "nonzero QSA dense end"),
    ] {
        for mode in [None, Some(false), Some(true)] {
            let error = with_qwen4exp_frontier_schedule_override(mode, || {
                plan_qwen4exp_prefill_execution_from(start, n, cap, selected, frontier)
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains(message), "mode={mode:?}: {error}");
        }
    }
    let last =
        plan_qwen4exp_prefill_execution_from(usize::MAX - 2, 2, Some(2048), true, 2051).unwrap();
    assert_eq!(last.packed_ranges, vec![usize::MAX - 2..usize::MAX]);
    assert_eq!(last.packed_token_count, 2);
    assert_eq!(last.scalar_start, usize::MAX);
    assert!(last.contains_selection);
}

#[test]
fn frontier_schedule_scope_restores_none_nested_unwind_and_thread() {
    let state = || QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get());
    let plan = || plan_qwen4exp_prefill_execution_from(2048, 2048, Some(2048), true, 2051).unwrap();
    let production = plan();
    assert_eq!(state(), None);
    for outer in [None, Some(false), Some(true)] {
        with_qwen4exp_frontier_schedule_override(outer, || {
            let expected = plan();
            assert_eq!(state(), outer);
            assert_eq!(expected.packed_ranges, ranges(2048, outer.unwrap_or(true)));
            for inner in [None, Some(false), Some(true)] {
                with_qwen4exp_frontier_schedule_override(inner, || {
                    assert_eq!(state(), inner);
                    assert_eq!(plan().packed_ranges, ranges(2048, inner.unwrap_or(true)));
                });
                assert_eq!(state(), outer);
                assert_eq!(plan(), expected);
                assert!(
                    std::panic::catch_unwind(|| {
                        with_qwen4exp_frontier_schedule_override(inner, || {
                            panic!("nested scope probe")
                        })
                    })
                    .is_err()
                );
                assert_eq!(state(), outer);
                assert_eq!(plan(), expected);
            }
            let (child_state, child_plan) = std::thread::spawn(|| {
                (
                    QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s| s.get()),
                    plan_qwen4exp_prefill_execution_from(2048, 2048, Some(2048), true, 2051)
                        .unwrap(),
                )
            })
            .join()
            .unwrap();
            assert_eq!(child_state, None);
            assert_eq!(child_plan, production);
        });
        assert_eq!(state(), None);
        assert_eq!(plan(), production);
        assert!(
            std::panic::catch_unwind(|| {
                with_qwen4exp_frontier_schedule_override(outer, || panic!("outer scope probe"))
            })
            .is_err()
        );
        assert_eq!(state(), None);
        assert_eq!(plan(), production);
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
        "schedule_override":QWEN4EXP_FRONTIER_SCHEDULE_OVERRIDE.with(|s|s.get()),
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
    options: FrontierOptions,
    warm: bool,
    label: &str,
    out: &mut std::fs::File,
) -> PacketResult<Arm> {
    // This innermost scope covers plan inspection, ordinary execution and census.
    // In production mode B must explicitly clear any enclosing override.
    options.with_arm(candidate, || {
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
            "candidate_mode":options.candidate_label(),"schedule_override":options.arm_override(candidate),
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
            "candidate_mode":options.candidate_label(),"schedule_override":options.arm_override(candidate),
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
    options: FrontierOptions,
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
            options,
            true,
            &format!("{label}/{name}"),
            out,
        )?);
    }
    compare(out, &warm[0], &warm[1])?;
    drop(warm);
    for round in 1..=options.rounds {
        let mut attempts = Vec::new();
        for (name, candidate) in [("A1", false), ("B1", true), ("B2", true), ("A2", false)] {
            let clock = std::time::Instant::now();
            prepare(r)?;
            let arm_label = format!("{label}/round{round}/{name}");
            emit(
                out,
                json!({"event":"prepare","label":arm_label,"wall_ms":clock.elapsed().as_secs_f64()*1e3,"outside_prefill_timing":true}),
            );
            let current = arm(r, tokens, candidate, options, false, &arm_label, out)?;
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
            "rounds":options.rounds,"candidate_mode":options.candidate_label(),
            "gpu_ms":pairs(attempts.iter().map(|a|a.gpu_ms).collect()),
            "ordinary_call_wall_ms":pairs(attempts.iter().map(|a|a.wall_ms).collect()),
            "decision":"unscored; suffix and whole savings overlap, do not add"}),
        );
    }
    Ok(())
}

fn packet(
    out: &mut std::fs::File,
    stage: &str,
    both: bool,
    options: FrontierOptions,
) -> PacketResult<()> {
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
                    screen(
                        r,
                        tokens,
                        &format!("{id}/suffix_at2048"),
                        options,
                        out,
                        |r| {
                            r.workspace.restore_checkpoint_for_tests(&checkpoint);
                            require(r.next_position() == PREFIX, "checkpoint frontier mismatch")
                        },
                    )?;
                }
                // The suffix checkpoint has dropped before any whole request.
                if stage != "suffix" {
                    screen(r, tokens, &format!("{id}/whole4096"), options, out, |r| {
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
    let candidate = std::env::var_os("FLASH_FRONTIER_CANDIDATE");
    let rounds = std::env::var_os("FLASH_FRONTIER_ROUNDS");
    let options = FrontierOptions::parse(
        candidate
            .as_deref()
            .map(|v| v.to_str().expect("FLASH_FRONTIER_CANDIDATE must be UTF-8")),
        rounds
            .as_deref()
            .map(|v| v.to_str().expect("FLASH_FRONTIER_ROUNDS must be UTF-8")),
    )
    .expect("invalid frontier schedule packet options");
    run_packet(
        "flash.frontier_schedule.v1",
        include_bytes!("frontier_schedule.rs"),
        json!({
            "stage":stage,"corpora":corpora,"rounds":options.rounds,"capacity":CAPACITY,"packed_capacity":2048,"frontier":2051,
            "candidate_mode":options.candidate_label(),"A_schedule_override":false,"B_schedule_override":options.arm_override(true),
            "A":"explicit forced incumbent: suffix3+2045 / whole2048+3+2045",
            "B":if options.candidate == FrontierCandidateMode::Production {
                "production policy (None): suffix2048 / whole2048+2048"
            } else {
                "explicit guarded candidate (Some(true)); historical forced B: suffix2048 / whole2048+2048"
            },
            "router":"both arms current production, including strict N2045; no router override",
            "scope":"cfg(test), thread-local, unwind-safe; only selected-capable capacity2048 frontier2051 requests (2048,2048) and (0,4096)",
            "admission":"native combined load includes selected scratch for4100; normal CPU gate for one checkpoint plus96MiB diagnostic margin",
            "measurement":"ordinary driver, no command splitting; complete valid command GPU sum and ordinary-call wall; reset/restore, endpoint readback/hashes, JSON and four continuations outside timing; warm census excluded",
            "timestamp_limit":"existing aggregate and sample count; individual raw timestamps not exposed; incomplete or invalid aggregate fails",
            "quality":"finite logits, bidirectional KL, relative L2, choice regrets, QSA/PLE/position metadata and persistent allocation hashes; no bit gate; final packed rows2045 versus2048",
            "source_prompt":"retained roadmap prose (GSQ is selected by artifact path); optional identity-checked SSH repeated whole and labeled; actual4100 token IDs bound",
            "production_change":false
        }),
        |out| packet(out, &stage, corpora == "both", options),
    );
}
