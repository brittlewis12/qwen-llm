use super::*;

fn capture(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    token: u32,
    profiled: bool,
) -> (
    Observation,
    Option<Qwen4ExpLayerProfileOutcome>,
    Vec<DispatchCensusRow>,
) {
    crate::metal::dispatch_census_begin();
    let (observation, profile) = if profiled {
        let profile = runner.forward_token_layer_profiled(token).unwrap();
        let observation = Observation {
            logits: vec![runner.logits().unwrap().to_vec()],
            hyper: vec![runner.workspace.final_hyper_for_tests()],
            state: snapshot_persistent_state(runner),
            timing: vec![profile.token],
        };
        (observation, Some(profile))
    } else {
        (observe_product_at(runner, &[token], true, PREFIX), None)
    };
    (observation, profile, crate::metal::dispatch_census_take())
}

#[test]
#[ignore = "production lease; single settled-default whole-forward parent ledger, no kernel candidate"]
fn settled_default_parent_ledger() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!(
        "qwen4exp-default-parent-ledger-{}",
        std::process::id()
    ));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("parent_ledger artifacts={}", artifact.display());
    let bytes = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97"
    );
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, PREFIX + 1).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, PREFIX).unwrap();
    assert!(
        loaded.guarded_topk_enabled()
            && loaded.hc_up_mix_enabled()
            && !loaded.split_decode_enabled()
    );
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert!(
        !runner
            .workspace
            .memory_plan()
            .allocations()
            .iter()
            .any(|a| a.name == "session.qsa_split")
    );
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    runner.prefill(&tokens[..PREFIX]).unwrap();
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        PREFIX
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let state_bytes: usize = runner
        .workspace
        .persistent_state_tensors()
        .iter()
        .map(|t| t.n_bytes() as usize)
        .sum();
    assert!(state_bytes * 5 < 1024 * 1024 * 1024);
    let mut observations = Vec::new();
    let mut profile = None;
    let mut native_shape = None;
    for (name, profiled, measured) in [
        ("warm-ordinary", false, false),
        ("warm-profiled", true, false),
        ("before", false, true),
        ("profiled", true, true),
        ("after", false, true),
    ] {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        assert_eq!(runner.next_position(), PREFIX);
        let (observed, outcome, census) = capture(&mut runner, tokens[PREFIX], profiled);
        defaults::save(&artifact, name, &observed);
        std::fs::write(
            artifact.join(format!("{name}.txt")),
            format!(
                "timing={:?}\nprofile={outcome:#?}\ncensus={census:#?}\n",
                observed.timing
            ),
        )
        .unwrap();
        if let Some(outcome) = &outcome {
            if let Ok(timestamps) = &outcome.raw_timestamps {
                std::fs::write(
                    artifact.join(format!("{name}.timestamps.u64le")),
                    timestamps
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        defaults::witness(&census, 1, [true, false, true]);
        let shapes: Vec<_> = census
            .iter()
            .map(|r| {
                (
                    r.kernel.clone(),
                    r.grid_width,
                    r.grid_height,
                    r.grid_depth,
                    r.threads_width,
                    r.threads_height,
                    r.threads_depth,
                    r.encoder_concurrent,
                )
            })
            .collect();
        if let Some(native) = &native_shape {
            assert_eq!(&shapes, native);
        } else {
            native_shape = Some(shapes);
        }
        assert_eq!(runner.next_position(), PREFIX + 1);
        assert!(
            runner
                .workspace
                .qsa_committed_lengths()
                .iter()
                .all(|(_, n)| *n == PREFIX + 1)
        );
        assert_eq!(
            *runner.workspace.ple_prior_tokens().last().unwrap(),
            tokens[PREFIX]
        );
        if measured {
            if let Some(first) = observations.first() {
                assert_replay(name, first, &observed);
            }
            observations.push(observed);
            if let Some(outcome) = outcome {
                profile = Some(outcome.profile.unwrap());
            }
        } else if let Some(outcome) = outcome {
            outcome.profile.unwrap();
        }
    }
    let profile = profile.unwrap();
    assert_eq!(profile.stages.len(), 48);
    let mut accepted = true;
    for (axis, observer_floor) in [("GPU", 0.05), ("wall", 0.10)] {
        let times: Vec<_> = observations
            .iter()
            .map(|o| {
                if axis == "GPU" {
                    o.timing[0].gpu_ms.unwrap()
                } else {
                    o.timing[0].total_wall_ms
                }
            })
            .collect();
        assert!(times.iter().all(|v| v.is_finite() && *v > 0.0));
        let baseline = (times[0] + times[2]) * 0.5;
        let drift = (times[0] - times[2]).abs() / baseline;
        let observer = times[1] / baseline - 1.0;
        let valid = drift <= 0.05 && observer.abs() <= observer_floor;
        accepted &= valid;
        eprintln!(
            "parent_ledger {axis} before_profiled_after_ms={times:?} control_drift={drift} observer_delta={observer} accepted={valid}"
        );
    }
    let mut groups: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for row in &profile.stages {
        assert!(row.duration_ticks > 0);
        let name = match row.stage {
            Qwen4ExpLayerStage::LayersZeroOne => "bootstrap_layers0_1",
            Qwen4ExpLayerStage::PostPle {
                mixer: MixerKind::GatedDeltaNet,
                ..
            } => "complete_GDN_containing_blocks",
            Qwen4ExpLayerStage::PostPle {
                mixer: MixerKind::QwenSparseAttention,
                ..
            } => "complete_QSA_containing_blocks",
            Qwen4ExpLayerStage::Tail => "final_HC_and_logits",
        };
        let group = groups.entry(name).or_default();
        group.0 += 1;
        group.1 += row.gpu_ms;
        eprintln!(
            "parent_ledger stage={:?} normalized_ms={} ticks={}",
            row.stage, row.gpu_ms, row.duration_ticks
        );
    }
    assert_eq!(groups["complete_GDN_containing_blocks"].0, 34);
    assert_eq!(groups["complete_QSA_containing_blocks"].0, 12);
    eprintln!(
        "parent_ledger groups={groups:?} boundary_ms={} sampled_span_ticks={} verdict={}",
        profile.encoder_boundary_ms,
        profile.sampled_span_ticks,
        if accepted {
            "QUALIFIED_PARENT_OBSERVATION"
        } else {
            "INCONCLUSIVE_NO_RETRY"
        }
    );
}
