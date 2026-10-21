use super::*;

fn observation() -> DecodeObservation {
    DecodeObservation {
        unix_s: Some(1.0),
        pid: 1,
        physical_read_bytes: Some(4096),
        pageins: Some(7),
        rss_bytes: Some(123),
        footprint_bytes: Some(456),
        metal_allocated_bytes: 789,
        thermal_state: 0,
        error: None,
    }
}

fn report(depth: usize, engrams: &[usize]) -> profile::Report {
    let mut operations = Vec::new();
    let mut decode_intervals = Vec::new();
    for position in depth..depth + TG {
        let mut add = |name, dims, submits, gpu_ms| {
            operations.push(profile::Operation {
                operation: name,
                dimensions: dims,
                count: 1,
                completed_submits: submits,
                invalid_gpu_timestamps: 0,
                gpu_ms: Some(gpu_ms),
                encode_ms: 0.0,
                commit_wait_ms: 0.0,
                wall_ms: 0.0,
            })
        };
        add("gpu_step", [position, 0, 40, 0], 1, 1.0);
        for (i, (name, layer)) in profile::decode::layout(40, engrams, false, true)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            add(name, [position, layer, 1, 0], 0, 0.001);
            let start_ns = 1_000_000 * (position + 1) as u64 + 2000 * i as u64;
            decode_intervals.push(profile::decode::Interval {
                operation: name,
                position,
                layer,
                start_ns,
                end_ns: start_ns + 1000,
            });
        }
    }
    profile::Report {
        decode_intervals,
        token_wall_ms: 100.0,
        process_pageins: Some(0),
        process_disk_read_bytes: Some(0),
        operations,
    }
}

fn fixture(depth: usize) -> Vec<Window> {
    (0..5)
        .map(|rep| Window {
            rep,
            role: ROLES[rep],
            start_position: depth,
            end_position: depth + TG,
            capacity: CAPACITY,
            window_wall_ms: 100.0,
            before: observation(),
            after: observation(),
            physical_read_bytes: Some(0),
            pageins: Some(0),
            error: None,
            profile: matches!(rep, 1 | 3).then(|| report(depth, &[0, 7])),
            steps: synthetic_ids(TG, 7, 3)
                .into_iter()
                .enumerate()
                .map(|(i, token)| Step {
                    position: depth + i,
                    token,
                    forward_wall_ms: 1.0,
                    logits_len: 7,
                    finite: true,
                    bitwise_equal: (rep > 0).then_some(true),
                    first_mismatch: None,
                    commands: Some(1),
                    trace_copies: Some(0),
                })
                .collect(),
        })
        .collect()
}

#[test]
fn protocol_gates_keep_raw_evidence_and_exact_overhead_limit() {
    for depth in DEPTHS {
        let mut windows = fixture(depth);
        assert!(verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
        windows[3].window_wall_ms = 110.0;
        assert!(verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
        windows[3].window_wall_ms = 110.001;
        assert!(!verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
    }
    for rep in 0..5 {
        for change in 0..16 {
            let mut windows = fixture(2048);
            let w = &mut windows[rep];
            match change {
                0 => w.steps.pop().map(|_| ()).unwrap(),
                1 => w.steps[31].bitwise_equal = Some(false),
                2 => w.steps[0].finite = false,
                3 => w.steps[0].logits_len = 6,
                4 => w.steps[0].token ^= 1,
                5 => w.steps[0].position += 1,
                6 => w.steps[0].commands = Some(2),
                7 => w.steps[0].trace_copies = Some(1),
                8 => w.window_wall_ms = f64::NAN,
                9 => w.capacity -= 1,
                10 => w.end_position -= 1,
                11 => w.after.pageins = None,
                12 => w.after.physical_read_bytes = Some(4095),
                13 => w.error = Some("injected".into()),
                14 => w.steps[31].commands = None,
                _ => w.steps[31].first_mismatch = Some((6, 0, 1)),
            }
            let raw = serde_json::to_value(&windows).unwrap();
            assert!(
                !verdict(&windows, 2048, 7, 40, &[0, 7]).1.is_empty(),
                "{rep}/{change}"
            );
            assert_eq!(raw, serde_json::to_value(&windows).unwrap());
        }
    }
    for rep in 2..5 {
        for physical in [false, true] {
            let mut w = fixture(0);
            if physical {
                w[rep].after.physical_read_bytes = Some(8192);
                w[rep].physical_read_bytes = Some(4096);
            } else {
                w[rep].after.pageins = Some(8);
                w[rep].pageins = Some(1);
            }
            assert!(!verdict(&w, 0, 7, 40, &[0, 7]).1.is_empty());
        }
    }
    for n in 0..5 {
        assert!(!verdict(&fixture(0)[..n], 0, 7, 40, &[0, 7]).1.is_empty());
    }
    assert_eq!(counter_delta(Some(5), Some(4)), None);
    assert_eq!(counter_delta(None, Some(4)), None);
    assert_eq!(counter_delta(Some(4), None), None);
    assert!(first_logits_mismatch(&[0.0f32.to_bits()], &[-0.0]).is_some());
}

#[test]
fn timestamps_and_exact_position_coverage_fail_closed_not_overlap() {
    for engrams in [vec![], vec![0, 7]] {
        let mut r = report(8192, &engrams);
        r.operations.reverse();
        r.decode_intervals.reverse();
        let c = costs(&r, 8192, 40, &engrams).unwrap();
        assert_eq!(c.positions.len(), TG);
        assert!(c.exclusive);
        assert!(
            c.positions
                .iter()
                .all(|p| p.sampled_envelopes.uncovered_sampled_envelope_ns > 0)
        );
        assert_eq!(
            c.inclusive_stage_stats_nonadditive
                .contains_key("decode_engram_projections"),
            !engrams.is_empty()
        );
        assert_eq!(
            c.inclusive_stage_stats_nonadditive["decode_ffn_composite_including_mhc"].count,
            TG * 30
        );
        for name in profile::decode::FFN_STAGES {
            assert_eq!(c.inclusive_stage_stats_nonadditive[name].count, TG * 10);
        }
        assert_eq!(c.ffn_enclosing_views_nonadditive.len(), TG * 40);
        for view in &c.ffn_enclosing_views_nonadditive {
            assert_eq!(
                view.reconstructed_from_split,
                profile::decode::FFN_LAYERS.contains(&view.layer)
            );
            assert_eq!(
                view.end_ns - view.start_ns,
                if view.reconstructed_from_split {
                    7000
                } else {
                    1000
                }
            );
        }
    }
    for change in 0..15 {
        let mut r = report(0, &[0, 7]);
        match change {
            0 => {
                r.operations.pop();
            }
            1 => r.operations[2].dimensions = r.operations[1].dimensions,
            2 => r.operations[1].gpu_ms = None,
            3 => r.operations[1].gpu_ms = Some(f64::NAN),
            4 => r.operations[1].gpu_ms = Some(0.0),
            5 => r.operations[1].invalid_gpu_timestamps = 1,
            6 => r.operations[0].completed_submits = 2,
            7 => r.operations[1].completed_submits = 1,
            8 => r.operations[1].count = 2,
            9 => r.operations[1].dimensions[0] = 32,
            10 => r.operations[1].operation = "unknown",
            11 => r.operations[1].gpu_ms = Some(2.0),
            12 => r.operations[1].wall_ms = 1.0,
            13 => r.operations[0].encode_ms = f64::NAN,
            _ => r.operations[1].dimensions[1] = 39,
        }
        assert!(costs(&r, 0, 40, &[0, 7]).is_err(), "change {change}");
    }
    let r = report(0, &[0, 7]);
    assert!(costs(&r, 0, 40, &[]).is_err());
    assert!(costs(&report(0, &[]), 0, 40, &[0]).is_err());
    for change in 0..4 {
        let mut r = report(0, &[0, 7]);
        match change {
            0 => {
                r.decode_intervals.pop();
            }
            1 => r.decode_intervals[1].start_ns = r.decode_intervals[0].end_ns - 1,
            2 => r.decode_intervals[1].position = 1,
            _ => r.decode_intervals[1].end_ns += 1,
        }
        assert!(costs(&r, 0, 40, &[0, 7]).is_err());
    }
    let mut w = fixture(0);
    w[3].profile.as_mut().unwrap().process_pageins = None;
    assert!(!verdict(&w, 0, 7, 40, &[0, 7]).1.is_empty());
}

fn set_interval(r: &mut profile::Report, index: usize, start: u64, end: u64) {
    let raw = &mut r.decode_intervals[index];
    raw.start_ns = start;
    raw.end_ns = end;
    r.operations
        .iter_mut()
        .find(|o| o.operation == raw.operation && o.dimensions == [raw.position, raw.layer, 1, 0])
        .unwrap()
        .gpu_ms = Some((end - start) as f64 / 1e6);
}

#[test]
fn overlapping_observations_pass_without_an_exclusive_or_command_coverage_claim() {
    for depth in DEPTHS {
        let mut windows = fixture(depth);
        for rep in [1, 3] {
            let r = windows[rep].profile.as_mut().unwrap();
            let start = r.decode_intervals[0].start_ns;
            // The first contains both following disjoint intervals: adjacency misses one overlap.
            set_interval(r, 0, start + 100, start + 200);
            set_interval(r, 1, start + 110, start + 130);
            set_interval(r, 2, start + 140, start + 150);
            for op in r
                .operations
                .iter_mut()
                .filter(|o| o.operation == "gpu_step")
            {
                op.gpu_ms = Some(0.000001);
            }
            r.operations.reverse();
            r.decode_intervals.reverse();
            let c = costs(r, depth, 40, &[0, 7]).unwrap();
            assert!(!c.exclusive);
            let summary = &c.positions[0].sampled_envelopes;
            assert!(!summary.exclusive);
            assert_eq!(summary.overlap_multiplicity_ns, 30);
            assert_eq!(summary.covered_by_two_or_more_ns, 30);
            let json = serde_json::to_value(&c).unwrap();
            assert_eq!(json["exclusive"], false);
            assert!(json.get("inclusive_stage_stats_nonadditive").is_some());
            assert!(json["positions"][0].get("remainder_ms").is_none());
            assert!(json["positions"][0].get("exclusive_stage_sum_ms").is_none());
            assert_eq!(json["positions"][0]["command_gpu_ms"], 0.000001);
        }
        assert!(verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
        windows[3].window_wall_ms = 110.001;
        assert!(!verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
        windows[3].window_wall_ms = 100.0;
        windows[3].steps[31].bitwise_equal = Some(false);
        assert!(!verdict(&windows, depth, 7, 40, &[0, 7]).1.is_empty());
    }
}

#[test]
fn duplicate_wrong_position_and_corrupt_raw_samples_remain_invalid() {
    for change in 0..7 {
        let mut r = report(2048, &[0, 7]);
        match change {
            0 => r.decode_intervals[0].start_ns = 0,
            1 => r.decode_intervals[0].end_ns = u64::MAX,
            2 => r.decode_intervals[0].end_ns = r.decode_intervals[0].start_ns,
            3 => r.decode_intervals[0].end_ns = r.decode_intervals[0].start_ns - 1,
            4 => r.decode_intervals[0].position = 0,
            5 => r.decode_intervals[0].layer = 39,
            _ => {
                r.decode_intervals[1].operation = r.decode_intervals[0].operation;
                r.decode_intervals[1].layer = r.decode_intervals[0].layer;
            }
        }
        assert!(costs(&r, 2048, 40, &[0, 7]).is_err(), "{change}");
    }
}

#[test]
fn recipe_requires_common_capacity_exact_depths_and_seed() {
    let parse = |extra: &[&str]| {
        let mut args = vec!["qwen-bench", "suite", "-m", "fixture.gguf"];
        args.extend_from_slice(extra);
        let Cmd::Suite(args) = Args::try_parse_from(args).unwrap().cmd else {
            unreachable!()
        };
        args
    };
    let valid = [
        "--tg", "32", "--depth", "2048", "--runs", "3", "--seed", "3",
    ];
    let extent = FamilyBenchExtent::for_shapes(&[], &[32], &DEPTHS, (16, 16)).unwrap();
    validate_recipe(&parse(&valid), extent).unwrap();
    for index in [1, 3, 5, 7] {
        let mut invalid = valid;
        invalid[index] = "1";
        assert!(validate_recipe(&parse(&invalid), extent).is_err());
    }
    for extra in [
        vec!["--pp", "16"],
        vec!["--no-warmup"],
        vec!["--prefill-chunk", "256"],
    ] {
        let mut invalid = valid.to_vec();
        invalid.extend(extra);
        assert!(validate_recipe(&parse(&invalid), extent).is_err());
    }
    assert!(
        validate_recipe(
            &parse(&valid),
            FamilyBenchExtent {
                forwards: 8224,
                ..extent
            }
        )
        .is_err()
    );
    let mut old = valid;
    old[3] = "0,2048,8192";
    assert!(validate_recipe(&parse(&old), extent).is_err());
    assert_eq!((extent.forwards, extent.prefill_endpoint), (2080, 2048));
    assert_eq!(SCHEMA, "v41-private-restored-decode-ffn-v1");
}

#[test]
fn ffn_enclosing_view_uses_sorted_endpoints_not_sum_or_adjacency() {
    let mut r = report(2048, &[1, 14]);
    let indices: Vec<_> = r
        .decode_intervals
        .iter()
        .enumerate()
        .filter(|(_, i)| {
            i.position == 2048 && i.layer == 0 && profile::decode::FFN_STAGES.contains(&i.operation)
        })
        .map(|(i, _)| i)
        .collect();
    for (index, (start, end)) in
        indices
            .into_iter()
            .zip([(100, 150), (90, 200), (120, 140), (160, 180)])
    {
        set_interval(&mut r, index, start, end);
    }
    r.decode_intervals.reverse();
    let c = costs(&r, 2048, 40, &[1, 14]).unwrap();
    assert_eq!(
        (c.expected_spans_per_token, c.expected_samples_per_token),
        (196, 392)
    );
    let v = &c.ffn_enclosing_views_nonadditive[0];
    assert_eq!((v.start_ns, v.end_ns), (90, 200));
    assert_eq!(v.enclosing_ms, 110.0 / 1e6);
    assert!(!c.exclusive);
}

#[derive(Default)]
struct Mock {
    position: usize,
    prefilled: Vec<u32>,
    restores: usize,
    tokens: Vec<u32>,
    fail_at: Option<usize>,
}
impl Session for Mock {
    type Snapshot = usize;
    fn position(&self) -> usize {
        self.position
    }
    fn capacity(&self) -> usize {
        CAPACITY
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        assert!(self.prefilled.is_empty());
        self.prefilled.extend_from_slice(ids);
        self.position += ids.len();
        Ok(())
    }
    fn snapshot(&self) -> Result<usize> {
        Ok(self.position)
    }
    fn restore(&mut self, snapshot: &usize) -> Result<()> {
        self.position = *snapshot;
        self.restores += 1;
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        ensure!(self.fail_at != Some(self.position), "injected step failure");
        self.tokens.push(token);
        self.position += 1;
        Ok(vec![self.position as f32 + token as f32; 7])
    }
    fn counts(&self) -> Option<(usize, usize)> {
        Some((1, 0))
    }
}

#[test]
fn one_prefix_five_restores_fixed_tokens_no_cpu_fallback_profile() {
    for depth in DEPTHS {
        let mut session = Mock::default();
        // CPU fixtures deliberately have no GPU timestamps: the complete attempt must be rejected.
        assert!(
            !run_depth(&mut session, depth, 7, 40, &[0, 7], observation)
                .unwrap()
                .accepted
        );
        assert_eq!(
            session.prefilled,
            synthetic_ids(depth, 7, 3 ^ DEPTH_SEED_SALT)
        );
        assert_eq!(session.restores, 5);
        assert_eq!(session.tokens, synthetic_ids(TG, 7, 3).repeat(5));
        assert_eq!(session.position, depth + TG);
        profile::Scope::begin().unwrap().finish();
    }
}

#[test]
fn restored_windows_compare_all_rows_and_errors_release_scope() {
    let mut session = Mock::default();
    let mut reference = vec![0; TG * 7];
    let first = window(&mut session, 0, 7, 0, &mut reference, &observation);
    assert!(first.error.is_none());
    session.position = 0;
    let profiled = window(&mut session, 0, 7, 1, &mut reference, &observation);
    assert!(profiled.steps.iter().all(|s| s.bitwise_equal == Some(true)));
    reference[31 * 7 + 6] ^= 1;
    session.position = 0;
    let bad = window(&mut session, 0, 7, 3, &mut reference, &observation);
    assert_eq!(bad.steps[31].bitwise_equal, Some(false));
    assert_eq!(bad.steps[31].first_mismatch.unwrap().0, 6);
    session.position = 0;
    session.fail_at = Some(2);
    let failed = window(&mut session, 0, 7, 3, &mut reference, &observation);
    assert_eq!(failed.steps.len(), 2);
    assert!(failed.error.unwrap().contains("injected"));
    assert!(failed.profile.is_some());
    profile::Scope::begin().unwrap().finish();
}
