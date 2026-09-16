use super::*;
use crate::qwen4exp_child_profile::{self as child, Bank, Record};

fn shapes(rows: &[DispatchCensusRow]) -> Vec<(String, [u64; 6], bool)> {
    rows.iter()
        .map(|r| {
            (
                r.kernel.clone(),
                [
                    r.grid_width,
                    r.grid_height,
                    r.grid_depth,
                    r.threads_width,
                    r.threads_height,
                    r.threads_depth,
                ],
                r.encoder_concurrent,
            )
        })
        .collect()
}

fn report(records: &[Record], raw: &[u64], command_ms: f64) -> bool {
    const LEAVES: [&str; 16] = [
        "attention_hc",
        "index_projection_norm_pending",
        "pool_publish",
        "index_score_select_expand",
        "qkv_norm_publish",
        "attention_logits_value",
        "qsa_output_projection",
        "mixer_output_copy",
        "attention_combine",
        "ffn_hc",
        "moe_router",
        "moe_routed",
        "moe_shared",
        "moe_accumulate",
        "moe_output_copy",
        "ffn_combine",
    ];
    assert_eq!(raw.len(), records.len() * 2);
    assert_eq!(records.len(), LEAVES.len() + 2);
    assert!(records.iter().all(|r| r.completed && r.host_ms.is_finite()));
    let pair = |name| {
        let rows: Vec<_> = records.iter().filter(|r| r.name == name).collect();
        assert_eq!(rows.len(), 1, "{name}");
        let row = rows[0];
        (raw[row.start], raw[row.end])
    };
    let command = pair("command");
    let block = pair("block");
    assert!(command.1 > command.0 && block.0 >= command.0 && block.1 <= command.1);
    let pairs: Vec<_> = LEAVES.iter().map(|name| pair(*name)).collect();
    let coverage = child::coverage(block, &pairs).expect("valid positive in-envelope leaf spans");
    let scale = command_ms / (command.1 - command.0) as f64;
    let mut largest = ("", 0.0);
    for name in LEAVES {
        let (a, b) = pair(name);
        let ms = (b - a) as f64 * scale;
        let row = records.iter().find(|r| r.name == name).unwrap();
        eprintln!(
            "child_ledger group={name} raw_start={a} raw_end={b} ticks={} normalized_ms={ms:.9} host_encode_ms={:.9}",
            b - a,
            row.host_ms
        );
        if ms > largest.1 {
            largest = (name, ms);
        }
    }
    let block_host = records.iter().find(|r| r.name == "block").unwrap().host_ms;
    let leaf_host: f64 = records
        .iter()
        .filter(|r| LEAVES.contains(&r.name))
        .map(|r| r.host_ms)
        .sum();
    eprintln!(
        "child_ledger coverage={coverage:?} envelope_ms={} union_ms={} sum_ms={} overlap_ms={} gap_ms={} block_host_ms={block_host} leaf_host_sum_ms={leaf_host} largest_inclusive={largest:?}",
        coverage.envelope as f64 * scale,
        coverage.union as f64 * scale,
        coverage.sum as f64 * scale,
        coverage.overlap as f64 * scale,
        coverage.gaps as f64 * scale
    );
    // Overlap is retained as a telemetry finding, not silently added as exclusive work.
    coverage.overlap == 0 && coverage.gaps as f64 / coverage.envelope as f64 <= 0.05
}

#[test]
#[ignore = "production lease; single native layer39 child ledger on settled defaults, no candidate or replay"]
fn native_qsa39_child_ledger() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!(
        "qwen4exp-native-child-ledger-{}",
        std::process::id()
    ));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("child_ledger artifacts={}", artifact.display());
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
    let bank = Bank::new(&ctx);
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
    let cohort: BTreeMap<_, _> = (3..48)
        .step_by(4)
        .map(|layer| {
            let prefix = format!("blk.{layer}.");
            let mut tensors: Vec<_> = loaded
                .weights
                .tensors()
                .filter_map(|(name, t)| {
                    name.strip_prefix(&prefix)
                        .map(|name| (name.to_string(), format!("{:?}", t.dtype), t.shape.clone()))
                })
                .collect();
            tensors.sort();
            assert!(!tensors.is_empty(), "weight cohort {layer}");
            (layer, tensors)
        })
        .collect();
    std::fs::write(
        artifact.join("qsa-weight-cohort.txt"),
        format!("{cohort:#?}"),
    )
    .unwrap();
    for (layer, types) in &cohort {
        eprintln!(
            "child_ledger weight_cohort layer={layer} matches39={}",
            types == &cohort[&39]
        );
    }
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
    let tensors = runner.workspace.persistent_state_tensors();
    assert_eq!(tensors.len(), 121);
    assert!(tensors.iter().map(|t| t.n_bytes() as usize).sum::<usize>() * 6 < 1024 * 1024 * 1024);
    let mut timings = Vec::new();
    let mut reference = None;
    let mut reference_shapes = None;
    let mut profiles = Vec::new();
    for (name, profiled, measured) in [
        ("warm-ordinary", false, false),
        ("warm-profiled", true, false),
        ("before", false, true),
        ("profiled", true, true),
        ("after", false, true),
    ] {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        crate::metal::dispatch_census_begin();
        let mut run = || observe_product_at(&mut runner, &tokens[PREFIX..PREFIX + 1], true, PREFIX);
        let observed = if profiled {
            child::with_bank(&bank, run)
        } else {
            run()
        };
        let census = crate::metal::dispatch_census_take();
        let profile = profiled.then(|| bank.resolve(&ctx));
        defaults::save(&artifact, name, &observed);
        std::fs::write(
            artifact.join(format!("{name}.hyper.f32le")),
            observed
                .hyper
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            artifact.join(format!("{name}.txt")),
            format!(
                "timing={:?}\nprofile={profile:#?}\ncensus={census:#?}\n",
                observed.timing
            ),
        )
        .unwrap();
        if let Some((_, Ok(raw))) = &profile {
            std::fs::write(
                artifact.join(format!("{name}.timestamps.u64le")),
                raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )
            .unwrap();
        }
        defaults::witness(&census, 1, [true, false, true]);
        let rows = shapes(&census);
        if let Some(first) = &reference_shapes {
            assert_eq!(&rows, first);
        } else {
            let target = shapes(
                &census
                    .iter()
                    .filter(|r| r.tag.as_deref() == Some("native.layer39"))
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            assert!(!target.is_empty());
            for layer in (3..48).step_by(4) {
                let tag = format!("native.layer{layer}");
                let rows = shapes(
                    &census
                        .iter()
                        .filter(|r| r.tag.as_deref() == Some(tag.as_str()))
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                eprintln!(
                    "child_ledger census_cohort layer={layer} dispatches={} matches39={}",
                    rows.len(),
                    rows == target
                );
            }
            reference_shapes = Some(rows);
        }
        if let Some(first) = &reference {
            assert_replay(name, first, &observed);
        }
        if measured {
            timings.push(observed.timing[0]);
        }
        if let Some(profile) = profile {
            profiles.push((name, observed.timing[0], profile));
        }
        if reference.is_none() {
            reference = Some(observed);
        }
        assert_eq!(runner.next_position(), PREFIX + 1);
    }
    let mut accepted = true;
    for (axis, limit) in [("GPU", 0.05), ("wall", 0.10)] {
        let values: Vec<_> = timings
            .iter()
            .map(|t| {
                if axis == "GPU" {
                    t.gpu_ms.unwrap()
                } else {
                    t.total_wall_ms
                }
            })
            .collect();
        assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
        let mean = (values[0] + values[2]) * 0.5;
        let drift = (values[0] - values[2]).abs() / mean;
        let observer = values[1] / mean - 1.0;
        let valid = drift <= 0.05 && observer.abs() <= limit;
        accepted &= valid;
        eprintln!(
            "child_ledger {axis} A_P_A_ms={values:?} control_drift={drift} observer_delta={observer} accepted={valid}"
        );
    }
    for (name, timing, (records, raw)) in profiles {
        eprintln!("child_ledger profile={name}");
        let valid = report(&records, &raw.unwrap(), timing.gpu_ms.unwrap());
        if name == "profiled" {
            accepted &= valid;
        }
    }
    eprintln!(
        "child_ledger verdict={}",
        if accepted {
            "QUALIFIED_NATIVE_CHILD_OBSERVATION"
        } else {
            "INCONCLUSIVE_NO_RETRY"
        }
    );
}
