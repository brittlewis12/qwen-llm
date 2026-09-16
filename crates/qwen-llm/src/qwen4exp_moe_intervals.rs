use super::*;

#[derive(Debug, PartialEq)]
struct Coverage {
    envelope: u64,
    union: u64,
    sum: u64,
    concurrent: u64,
}

fn coverage(raw: &[u64]) -> Result<Coverage, &'static str> {
    if raw.is_empty() || raw.len() % 2 != 0 {
        return Err("missing or incomplete timestamp pairs");
    }
    let mut events = std::collections::BTreeMap::<u64, i32>::new();
    let mut sum = 0u64;
    for p in raw.chunks_exact(2) {
        if p[0] == 0 || p[1] == u64::MAX || p[1] <= p[0] {
            return Err("invalid timestamp pair");
        }
        sum = sum
            .checked_add(p[1] - p[0])
            .ok_or("timestamp sum overflow")?;
        *events.entry(p[0]).or_default() += 1;
        *events.entry(p[1]).or_default() -= 1;
    }
    let first = *events.first_key_value().unwrap().0;
    let last = *events.last_key_value().unwrap().0;
    let (mut previous, mut active, mut union, mut concurrent) = (first, 0, 0, 0);
    for (tick, delta) in events {
        if active > 0 {
            union += tick - previous;
        }
        if active > 1 {
            concurrent += tick - previous;
        }
        active += delta;
        previous = tick;
    }
    assert_eq!(active, 0);
    Ok(Coverage {
        envelope: last - first,
        union,
        sum,
        concurrent,
    })
}

pub(super) fn summarize(raw: &[u64], repeats: usize, gpu_ms: f64) -> serde_json::Value {
    assert_eq!(raw.len(), repeats * 12);
    let c = coverage(raw).expect("invalid stage intervals");
    let scale = gpu_ms / c.envelope as f64;
    let normalized = |ticks: u64| ticks as f64 * scale / repeats as f64;
    let stages: serde_json::Map<String, serde_json::Value> = STAGES.iter().enumerate().map(|(index, name)| {
        let ticks: u64 = (0..repeats).map(|r| raw[r*12+index*2+1]-raw[r*12+index*2]).sum();
        (name.to_string(), serde_json::json!({"inclusive_ticks":ticks,"normalized_inclusive_ms_per_moe":normalized(ticks),"fraction_of_envelope":ticks as f64/c.envelope as f64}))
    }).collect();
    serde_json::json!({"normalization_ms_per_tick":scale,"interpretation":"command-span normalization, not clock calibration or exclusive costs","envelope_ticks":c.envelope,"union_ticks":c.union,"sum_inclusive_ticks":c.sum,"overlap_multiplicity_ticks":c.sum-c.union,"concurrent_wall_ticks":c.concurrent,"uncovered_envelope_ticks":c.envelope-c.union,"normalized_union_ms_per_moe":normalized(c.union),"normalized_envelope_ms_per_moe":normalized(c.envelope),"stages":stages})
}

#[test]
fn interval_coverage_handles_reordering_overlap_and_invalid_samples() {
    assert_eq!(
        coverage(&[20, 30, 5, 10]).unwrap(),
        Coverage {
            envelope: 25,
            union: 15,
            sum: 15,
            concurrent: 0
        }
    );
    assert_eq!(
        coverage(&[5, 20, 10, 15, 12, 18]).unwrap(),
        Coverage {
            envelope: 15,
            union: 15,
            sum: 26,
            concurrent: 8
        }
    );
    assert_eq!(
        coverage(&[10, 20, 5, 10]).unwrap(),
        Coverage {
            envelope: 15,
            union: 15,
            sum: 15,
            concurrent: 0
        }
    );
    for bad in [
        &[][..],
        &[1][..],
        &[0, 2][..],
        &[2, 2][..],
        &[3, 2][..],
        &[1, u64::MAX][..],
        &[u64::MAX, 2][..],
    ] {
        assert!(coverage(bad).is_err());
    }
}

fn expected_census() -> Vec<serde_json::Value> {
    [
        ("kernel_mat_vec_f32_f32_lcpp_r2", [256,1,1],128),
        ("kernel_topk_logits_softmax_f32", [1,1,1],1),
        ("kernel_dot_sigmoid_f32", [1,1,1],32),
        ("kernel_moe_swiglu_iq4_xs_f32", [160,10,1],128),
        ("kernel_moe_down_weighted_sum_q8_0_f32", [1280,1,1],128),
        ("kernel_shared_swiglu_q8_0_f32_lcpp", [320,1,1],128),
        ("kernel_mat_vec_q8_0_f32_lcpp", [1280,1,1],128),
        ("kernel_axpy_scalar_f32", [3,1,1],1024),
    ].into_iter().map(|(kernel,grid,threads)| serde_json::json!({"kernel":kernel,"grid":grid,"threads":[threads,1,1]})).collect()
}

pub(crate) fn observe_saved_layer2(
    ctx: &MetalContext,
    weights: &Qwen4ExpMetalWeights,
    source: &std::path::Path,
    artifact: &std::path::Path,
) {
    let c = load_saved_layer2(ctx, weights, source);
    let frozen = bytes(&c.input);
    let ids: Vec<i32> = bytes(&c.ids)
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(
        ids.iter()
            .all(|&id| id >= 0 && (id as usize) < c.geometry.expert_count)
    );
    assert_eq!(
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        ids.len()
    );
    let w = Qwen4ExpMoeMetalWorkspace::new(ctx, c.geometry).unwrap();
    validate_contract(ctx, &c.input, c.weights(), &w).unwrap();
    preflight(ctx, c.weights()).unwrap();
    let expected = expected_census();
    let run = |name, repeats, sampled, census| {
        let (result, rows) =
            replay_packet(ctx, &c, &w, repeats, sampled, census, artifact, name, true);
        assert_eq!(bytes(&c.input), frozen);
        if census {
            assert_eq!(rows.len(), repeats * expected.len());
            for group in rows.chunks_exact(expected.len()) {
                assert_eq!(group.iter().map(row_json).collect::<Vec<_>>(), expected);
            }
        }
        result
    };
    run("census", 1, false, true);
    run("warm", 3, false, false);
    let before = run("before", REPEATS, false, true);
    let sampled = run("sampled", REPEATS, true, true);
    let after = run("after", REPEATS, false, true);
    let b = before["gpu_ms_per_moe"].as_f64().unwrap();
    let a = after["gpu_ms_per_moe"].as_f64().unwrap();
    let drift = (a - b).abs() / ((a + b) * 0.5);
    let result = serde_json::json!({"protocol":"saved-layer2-interval-v2","source":source.display().to_string(),"layer":2,"gate_dtype":"IQ4_XS","down_dtype":"Q8_0","selected_experts":ids,"before":before,"sampled":sampled,"after":after,"unsampled_drift_fraction":drift,"stability":if drift<=0.05 {"bounded"}else{"INCONCLUSIVE"},"interpretation":"warm isolated complete MoE; inclusive intervals, no end-to-end speedup or all-layer attribution","all_packets_native_output_routes_bitwise_and_input_immutable":true,"all_recorded_censuses_match_frozen_route":true});
    std::fs::write(
        artifact.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!(
        "moe_intervals_v2 artifacts={} before_ms={b} after_ms={a} drift={drift} intervals={}",
        artifact.display(),
        sampled["intervals"]
    );
}
