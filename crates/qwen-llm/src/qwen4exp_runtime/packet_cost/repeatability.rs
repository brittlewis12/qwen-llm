//! No warmup exemption: retain first-packet evidence and gate candidate attribution.
use super::*;
use sha2::{Digest, Sha256};

pub(super) fn widths(value: &str) -> Vec<usize> {
    let rows: Vec<usize> = value
        .split(',')
        .map(|v| v.trim().parse().expect("invalid packet width"))
        .collect();
    assert!(!rows.is_empty());
    for (i, row) in rows.iter().enumerate() {
        assert!(
            [1, 2, 4, 8].contains(row) && !rows[..i].contains(row),
            "widths must be distinct members of 1,2,4,8"
        );
    }
    rows
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) struct State {
    position: usize,
    history: String,
    qsa_lengths: Vec<(u32, usize)>,
    active: BTreeMap<String, String>,
    tails: BTreeMap<String, String>,
}

impl State {
    fn same_active(&self, other: &Self) -> bool {
        self.position == other.position
            && self.history == other.history
            && self.qsa_lengths == other.qsa_lengths
            && self.active == other.active
    }
}

fn changes(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> Vec<serde_json::Value> {
    assert!(a.keys().eq(b.keys()));
    a.iter()
        .filter(|(name, hash)| b[*name] != **hash)
        .map(|(name, hash)| json!({"region": name, "before": hash, "after": b[name]}))
        .collect()
}

pub(super) fn state(runner: &Qwen4ExpTextRunner<'_, '_, '_>, prefix: usize) -> State {
    assert!(!runner.workspace.is_poisoned());
    let mut qsa_regions = Vec::new();
    for (layer, tensors) in runner.workspace.qsa_persistent_state_tensors() {
        assert_eq!(tensors.len(), 4);
        let ratio = runner
            .weights
            .geometry
            .post_ple()
            .iter()
            .find(|block| block.layer() == layer)
            .unwrap()
            .mixer()
            .qsa()
            .unwrap()
            .compression_ratio();
        for (part, tensor) in tensors.into_iter().enumerate() {
            assert_eq!(
                tensor.dtype,
                if part == 0 {
                    GgmlType::F32
                } else {
                    GgmlType::F16
                }
            );
            assert_eq!(tensor.shape.len(), if part < 2 { 2 } else { 3 });
            let active = match part {
                0 => tensor.n_bytes() as usize,
                1 => tensor.shape[0] as usize * (prefix / ratio) * 2,
                2 | 3 => tensor.shape[0] as usize * tensor.shape[1] as usize * prefix * 2,
                _ => unreachable!(),
            };
            let name = ["pending", "compressed", "key", "value"][part];
            qsa_regions.push((tensor, format!("qsa.{layer}.{name}"), active));
        }
    }
    let mut active = BTreeMap::new();
    let mut tails = BTreeMap::new();
    let mut active_bytes = 0;
    for (i, tensor) in runner
        .workspace
        .persistent_state_tensors()
        .into_iter()
        .enumerate()
    {
        let matched = qsa_regions.iter().find(|(q, _, _)| {
            std::ptr::addr_eq(
                Retained::as_ptr(&q.buffer),
                Retained::as_ptr(&tensor.buffer),
            ) && q.offset == tensor.offset
        });
        let (name, n) = matched
            .map(|(_, name, n)| (name.clone(), *n))
            .unwrap_or_else(|| (format!("recurrent_or_ple.{i}"), tensor.n_bytes() as usize));
        assert!(n <= tensor.n_bytes() as usize);
        // All callers observe completed commands or a synchronous host restore.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                tensor.n_bytes() as usize,
            )
        };
        active_bytes += n;
        assert!(active.insert(name.clone(), sha(&bytes[..n])).is_none());
        tails.insert(name, sha(&bytes[n..]));
    }
    if prefix == runner.next_position() {
        assert_eq!(
            active_bytes as u64,
            runner.snapshot_bytes().unwrap(),
            "snapshot-owned coverage"
        );
    }
    State {
        position: runner.next_position(),
        history: sha(bytemuck::cast_slice(runner.workspace.ple_prior_tokens())),
        qsa_lengths: runner.workspace.qsa_committed_lengths(),
        active,
        tails,
    }
}

pub(super) fn restore(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    width: usize,
    label: &str,
) -> bool {
    let before = state(runner, base.length());
    runner.restore_snapshot(base).unwrap();
    let after = state(runner, base.length());
    let same = frozen.same_active(&after);
    eprintln!(
        "packet_restore {}",
        json!({
            "depth": base.length(), "width": width, "label": label,
            "position_before": before.position, "position_after": after.position,
            "active_matches_base": same, "active_changed_by_restore": changes(&before.active, &after.active),
            "active_diff_from_base": changes(&frozen.active, &after.active),
            "tails_changed_by_restore": changes(&before.tails, &after.tails),
            "tails_diff_from_base": changes(&frozen.tails, &after.tails),
            "history_matches_base": frozen.history == after.history,
            "qsa_lengths": after.qsa_lengths, "timing_eligible": false,
            "scope": "snapshot-owned active regions versus unused allocation tails; scratch not included",
        })
    );
    same
}

pub(super) struct Observation {
    pub rows: Vec<u32>,
    pub future: Vec<u32>,
    pub(super) packet_state: State,
    pub(super) future_state: State,
}

pub(super) struct Audit {
    pub baseline: Observation,
    pub candidate: Observation,
    pub stable: bool,
}

fn observe(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    next: u32,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) -> Observation {
    let (values, _) = run_arm(runner, tokens, "verify_all", captures, logits);
    let rows = bits(&values);
    let packet_state = state(runner, runner.next_position());
    let future = bits(runner.forward_token(next).unwrap().as_slice());
    let future_state = state(runner, runner.next_position());
    Observation {
        rows,
        future,
        packet_state,
        future_state,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_logits_comparison(
    depth: usize,
    width: usize,
    mode: Mode,
    label: &str,
    a: &[u32],
    b: &[u32],
    af: &[u32],
    bf: &[u32],
    vocab: usize,
) {
    assert_eq!(a.len(), width * vocab);
    assert_eq!(a.len(), b.len());
    eprintln!(
        "packet_audit {}",
        json!({
            "depth": depth, "width": width, "hc_mode": mode.name(), "label": label,
            "allrow_bits_equal": a == b, "future_bits_equal": af == bf,
            "reference_logits_sha256": sha(bytemuck::cast_slice(a)),
            "actual_logits_sha256": sha(bytemuck::cast_slice(b)),
            "reference_future_sha256": sha(bytemuck::cast_slice(af)),
            "actual_future_sha256": sha(bytemuck::cast_slice(bf)),
            "rows": a.chunks_exact(vocab).zip(b.chunks_exact(vocab))
                .map(|(a,b)| distribution_difference(a,b)).collect::<Vec<_>>(),
            "continuation": distribution_difference(af,bf), "timing_eligible": false,
            "scope": "each observation includes one scalar continuation before the next restore; no scratch clearing",
        })
    );
}

pub(super) fn compare(
    depth: usize,
    width: usize,
    mode: Mode,
    label: &str,
    a: &Observation,
    b: &Observation,
    vocab: usize,
) -> bool {
    emit_logits_comparison(
        depth, width, mode, label, &a.rows, &b.rows, &a.future, &b.future, vocab,
    );
    eprintln!(
        "packet_state {}",
        json!({
            "depth": depth, "width": width, "label": label, "hc_mode": mode.name(),
            "packet_active_equal": a.packet_state.same_active(&b.packet_state),
            "future_active_equal": a.future_state.same_active(&b.future_state),
            "packet_changed": changes(&a.packet_state.active, &b.packet_state.active),
            "future_changed": changes(&a.future_state.active, &b.future_state.active),
            "timing_eligible": false,
        })
    );
    a.rows == b.rows
        && a.future == b.future
        && a.packet_state.same_active(&b.packet_state)
        && a.future_state.same_active(&b.future_state)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn audit(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    tokens: &[u32],
    next: u32,
    mode: Mode,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) -> Audit {
    let (depth, width, vocab) = (
        base.length(),
        tokens.len(),
        runner.weights.geometry.vocab_size(),
    );
    let _baseline = Scope::new(Mode::Baseline);
    let mut stable = restore(runner, base, frozen, width, "A_first");
    let first = observe(runner, tokens, next, captures, logits);
    eprintln!(
        "packet_observation {}",
        json!({"depth":depth,"width":width,
        "label":"A_first", "hc_mode":"baseline", "timing_eligible":false,
        "logits_sha256":sha(bytemuck::cast_slice(&first.rows)),
        "future_sha256":sha(bytemuck::cast_slice(&first.future))})
    );
    // Retain every first-use result. No candidate executes until all controls finish.
    for label in [
        "A_repeat_1",
        "A_repeat_2",
        "A_after_serial_N",
        "A_after_packed_last",
    ] {
        if let Some(arm) = match label {
            "A_after_serial_N" => Some("serial"),
            "A_after_packed_last" => Some("packed_last"),
            _ => None,
        } {
            stable &= restore(runner, base, frozen, width, arm);
            run_arm(runner, tokens, arm, captures, logits);
        }
        stable &= restore(runner, base, frozen, width, label);
        let current = observe(runner, tokens, next, captures, logits);
        stable &= compare(depth, width, Mode::Baseline, label, &first, &current, vocab);
    }
    // A/B/A is allowed only after the baseline is repeatable, even for a no-op B.
    stable &= restore(runner, base, frozen, width, "A_pre_candidate");
    let before = observe(runner, tokens, next, captures, logits);
    stable &= compare(
        depth,
        width,
        Mode::Baseline,
        "A_pre_candidate",
        &first,
        &before,
        vocab,
    );
    stable &= restore(runner, base, frozen, width, "B_candidate_entry");
    if !stable {
        eprintln!(
            "packet_audit_gate {}",
            json!({"depth":depth,"width":width,
            "hc_mode":mode.name(),"baseline_stable":false,"candidate_executed":false,"timings_blocked":true})
        );
        return Audit {
            baseline: first,
            candidate: before,
            stable: false,
        };
    }
    let candidate = {
        let _candidate = Scope::new(mode);
        observe(runner, tokens, next, captures, logits)
    };
    compare(
        depth,
        width,
        mode,
        "A_before_vs_B",
        &before,
        &candidate,
        vocab,
    );
    stable &= restore(runner, base, frozen, width, "A_post_candidate");
    let after = observe(runner, tokens, next, captures, logits);
    stable &= compare(
        depth,
        width,
        Mode::Baseline,
        "A_before_vs_A_after_B",
        &before,
        &after,
        vocab,
    );
    compare(
        depth,
        width,
        mode,
        "A_after_vs_B",
        &after,
        &candidate,
        vocab,
    );
    if mode == Mode::Baseline {
        stable &= compare(depth, width, mode, "noop_B", &before, &candidate, vocab);
    }
    eprintln!(
        "packet_audit_gate {}",
        json!({"depth":depth,"width":width,
        "baseline_stable":stable,"candidate_executed":true,"hc_mode":mode.name(),
        "timings_blocked":!stable,"quality_qualified":false})
    );
    Audit {
        baseline: first,
        candidate,
        stable,
    }
}

fn feature_difference(a: &[f32], b: &[f32]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty() && a.iter().chain(b).all(|v| v.is_finite()));
    let (mut aa, mut bb, mut ab, mut error2, mut max_abs) = (0.0, 0.0, 0.0, 0.0, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        aa += x * x;
        bb += y * y;
        ab += x * y;
        error2 += (x - y) * (x - y);
        max_abs = max_abs.max((x - y).abs());
    }
    json!({"bits_equal": a.iter().zip(b).all(|(x,y)| x.to_bits() == y.to_bits()),
        "max_abs": max_abs, "rms_error": (error2/a.len() as f64).sqrt(),
        "baseline_rms": (aa/a.len() as f64).sqrt(),
        "relative_rms": (aa > 0.0).then(|| (error2/aa).sqrt()),
        "cosine": (aa > 0.0 && bb > 0.0).then(|| ab/(aa*bb).sqrt())})
}

pub(super) fn completed_values<T: bytemuck::Pod>(tensor: &MetalTensor) -> Vec<T> {
    assert_eq!(
        tensor.n_bytes() as usize,
        tensor.n_elements() as usize * size_of::<T>()
    );
    // Only used after synchronous native execution; these are shared F32/I32 buffers.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<T>(),
            tensor.n_elements() as usize,
        )
        .to_vec()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn localize(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    tokens: &[u32],
    next: u32,
    mode: Mode,
    audit: &Audit,
    logits: &mut Qwen4ExpRowLogitsBank,
) {
    use crate::metal::{
        evaluate_metal_memory_admission, host_page_size_bytes, price_shared_buffer_upper,
    };
    use crate::qwen4exp_moe::with_qwen4exp_moe_route_count_capture;
    use crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES as RESERVE;
    const LAYERS: usize = 48;
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    assert!(audit.stable && matches!(tokens.len(), 2 | 4));
    assert_eq!(runner.weights.geometry.layer_count(), LAYERS);
    let (depth, width, hidden) = (
        base.length(),
        tokens.len(),
        runner.weights.geometry.hidden_size(),
    );
    let taps: Vec<u32> = (1..LAYERS as u32).collect();
    let mut captures = runner.allocate_capture_bank(taps.clone(), width).unwrap();
    let counts = {
        let ctx = runner.ctx;
        let logical = (LAYERS * EXPERTS * size_of::<i32>()) as u64;
        let price = price_shared_buffer_upper(
            logical,
            ctx.shared_buffer_size_and_align(logical).unwrap(),
            host_page_size_bytes().unwrap() as u64,
            ctx.max_buffer_length() as u64,
        )
        .unwrap();
        let _allocation = ctx.begin_allocation_transaction();
        assert!(
            evaluate_metal_memory_admission(
                price.priced_upper_bytes,
                RESERVE,
                ctx.memory_signals(),
                true
            )
            .is_admitted(),
            "route capture allocation admission denied"
        );
        let counts = MetalTensor::zeros_i32(ctx, vec![EXPERTS as u64, LAYERS as u64]).unwrap();
        assert!(
            evaluate_metal_memory_admission(0, RESERVE, ctx.memory_signals(), true).is_admitted(),
            "route capture post-allocation admission denied"
        );
        counts
    };
    let mut records = Vec::new();
    for (selected, expected, label) in [
        (Mode::Baseline, &audit.baseline, "localize_baseline"),
        (mode, &audit.candidate, "localize_candidate"),
    ] {
        assert!(restore(runner, base, frozen, width, label));
        let _scope = Scope::new(selected);
        // Sentinel distinguishes an absent layer from a legitimate zero expert count.
        unsafe {
            std::slice::from_raw_parts_mut(
                counts.buffer.contents().as_ptr().cast::<i32>(),
                LAYERS * EXPERTS,
            )
            .fill(-1);
        }
        let ((features, rows), captured_layers) =
            with_qwen4exp_moe_route_count_capture(&counts, || {
                let output = runner
                    .forward_rows_with_captures_and_logits(tokens, &mut captures, logits)
                    .unwrap();
                (
                    completed_values::<f32>(output.features().tensor()),
                    bits(output.logits().as_slice()),
                )
            });
        let route_counts = completed_values::<i32>(&counts);
        eprintln!(
            "packet_localize_capture {}",
            json!({"depth":depth,"width":width,"label":label,
            "hc_mode":selected.name(),"feature_taps":taps,"route_layers":captured_layers,
            "capture_priced_bytes":captures.priced_upper_bytes(),"timing_eligible":false})
        );
        assert_eq!(
            captured_layers, LAYERS,
            "plain packed MoE includes both bootstrap layers"
        );
        for row in route_counts.chunks_exact(EXPERTS) {
            assert!(row.iter().all(|&n| (0..=width as i32).contains(&n)));
            assert_eq!(row.iter().sum::<i32>(), (width * TOP_K) as i32);
        }
        assert_eq!(features.len(), width * taps.len() * hidden);
        let packet_state = state(runner, runner.next_position());
        let future = bits(runner.forward_token(next).unwrap().as_slice());
        let observed = Observation {
            rows,
            future,
            packet_state,
            future_state: state(runner, runner.next_position()),
        };
        assert!(
            compare(
                depth,
                width,
                selected,
                label,
                expected,
                &observed,
                runner.weights.geometry.vocab_size()
            ),
            "localization hooks changed same-mode output/state"
        );
        records.push((features, route_counts));
    }
    let (a, ac) = &records[0];
    let (b, bc) = &records[1];
    let mut first_feature = None;
    let mut first_route = None;
    let mut previous = vec![None; width];
    let mut largest_increase = (0.0, None);
    for layer in 0..LAYERS {
        let count_start = layer * EXPERTS;
        let changed: Vec<_> = (0..EXPERTS)
            .filter(|&e| ac[count_start + e] != bc[count_start + e])
            .map(|e| json!({"expert":e,"baseline":ac[count_start+e],"candidate":bc[count_start+e]}))
            .collect();
        if !changed.is_empty() {
            first_route.get_or_insert(layer);
        }
        let mut features = Vec::new();
        if layer > 0 {
            for row in 0..width {
                let start = (row * taps.len() + layer - 1) * hidden;
                let diff = feature_difference(&a[start..start + hidden], &b[start..start + hidden]);
                if diff["bits_equal"] == false {
                    first_feature.get_or_insert(layer);
                }
                let rel = diff["relative_rms"].as_f64();
                let increase = rel.zip(previous[row]).map(|(value, prior)| value - prior);
                if let Some(increase) = increase {
                    if increase > largest_increase.0 {
                        largest_increase = (increase, Some((layer, row)));
                    }
                }
                previous[row] = rel;
                features
                    .push(json!({"row":row,"difference":diff,"relative_rms_increase":increase}));
            }
        }
        eprintln!(
            "packet_localize_layer {}",
            json!({"depth":depth,"width":width,"hc_mode":mode.name(),
            "layer":layer,"feature_rows":features,"changed_experts":changed.len(),"count_deltas":changed,
            "timing_eligible":false})
        );
    }
    eprintln!(
        "packet_localize_summary {}",
        json!({"depth":depth,"width":width,"hc_mode":mode.name(),
        "first_feature_bit_difference_layer":first_feature,"first_route_count_difference_layer":first_route,
        "largest_positive_relative_rms_increase_between_taps":largest_increase.0,"largest_increase_layer_row":largest_increase.1,
        "timing_eligible":false,"materiality_threshold":null,
        "limits":"47 completed means: layer 0 is inside fused bootstrap, first tap is layer 1; 48 MoE histograms include bootstrap. Counts aggregate token rows: equal counts do not imply equal per-row routes/weights. Means can hide stream differences. No QSA-selection or near-tie mechanism is proven."})
    );
}

#[test]
fn feature_metrics_are_not_distribution_metrics() {
    let same = feature_difference(&[1.0, -2.0], &[1.0, -2.0]);
    assert_eq!(same["relative_rms"], 0.0);
    assert!(same.get("kl_baseline_candidate").is_none());
    let scaled = feature_difference(&[1.0, -2.0], &[2.0, -4.0]);
    assert_eq!(scaled["relative_rms"], 1.0);
    assert_eq!(scaled["cosine"], 1.0);
    let zero = feature_difference(&[0.0], &[1.0]);
    assert!(zero["relative_rms"].is_null());
    assert_eq!(zero["rms_error"], 1.0);
}

#[test]
fn width_selection_fails_closed() {
    assert_eq!(widths("2, 4"), [2, 4]);
    for bad in ["", "0", "3", "2,2", "2,", "-1"] {
        assert!(std::panic::catch_unwind(|| widths(bad)).is_err());
    }
}

#[test]
fn state_equality_excludes_unused_tails_but_not_metadata_or_active_bytes() {
    let make = || State {
        position: 2179,
        history: "history".into(),
        qsa_lengths: vec![(7, 2179)],
        active: BTreeMap::from([("qsa.7.key".into(), "active".into())]),
        tails: BTreeMap::from([("qsa.7.key".into(), "tail".into())]),
    };
    let a = make();
    let mut b = make();
    b.tails.insert("qsa.7.key".into(), "different tail".into());
    assert!(a.same_active(&b));
    assert_eq!(changes(&a.tails, &b.tails).len(), 1);
    b.active
        .insert("qsa.7.key".into(), "different active".into());
    assert!(!a.same_active(&b));
    let mut b = make();
    b.position += 1;
    assert!(!a.same_active(&b));
    let mut b = make();
    b.qsa_lengths[0].1 += 1;
    assert!(!a.same_active(&b));
    let mut b = make();
    b.history.push('x');
    assert!(!a.same_active(&b));
}
