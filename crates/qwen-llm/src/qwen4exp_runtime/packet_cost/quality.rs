//! Teacher-forced characterization, not a task-quality or serial-identity gate.
use super::*;
use crate::qwen4exp_moe::router_capture::{self, Bank, EXPERTS, LAYERS, TOP_K};
use repeatability::{Audit, Observation, State, completed_values, restore, state};

fn nll(logits: &[u32], target: u32) -> f64 {
    assert!((target as usize) < logits.len());
    let values: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(f32::from_bits(x)))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()));
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln() - values[target as usize]
}

fn score(reference: &[u32], actual: &[u32], labels: &[u32], vocab: usize) -> serde_json::Value {
    assert!(!labels.is_empty());
    assert_eq!(reference.len(), labels.len() * vocab);
    assert_eq!(reference.len(), actual.len());
    let (mut a_nll, mut b_nll, mut kl, mut reverse_kl, mut top1, mut max_kl) =
        (0.0, 0.0, 0.0, 0.0, 0, 0.0_f64);
    for ((a, b), &target) in reference
        .chunks_exact(vocab)
        .zip(actual.chunks_exact(vocab))
        .zip(labels)
    {
        a_nll += nll(a, target);
        b_nll += nll(b, target);
        let d = distribution_difference(a, b);
        let k = d["kl_baseline_candidate"].as_f64().unwrap();
        kl += k;
        max_kl = max_kl.max(k);
        reverse_kl += d["kl_candidate_baseline"].as_f64().unwrap();
        top1 += usize::from(d["top1_agrees"] == true);
    }
    let n = labels.len() as f64;
    json!({"rows":labels.len(),"reference_mean_nll":a_nll/n,"mean_nll":b_nll/n,
        "mean_nll_delta":(b_nll-a_nll)/n,"mean_kl_reference_actual":kl/n,
        "mean_kl_actual_reference":reverse_kl/n,"max_kl_reference_actual":max_kl,
        "top1_agreements":top1})
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cheap_report(
    depth: usize,
    width: usize,
    mode: Mode,
    serial: &[u32],
    serial_future: &[u32],
    baseline: &[u32],
    baseline_future: &[u32],
    candidate: &[u32],
    candidate_future: &[u32],
    labels: &[u32],
    vocab: usize,
) {
    assert_eq!(labels.len(), width + 1);
    eprintln!(
        "packet_serial_quality {}",
        json!({"depth":depth,"width":width,"hc_mode":mode.name(),
        "baseline_vs_serial":score(serial,baseline,&labels[..width],vocab),
        "candidate_vs_serial":score(serial,candidate,&labels[..width],vocab),
        "rows":(0..width).map(|row| {
            let span = row*vocab..(row+1)*vocab;
            json!({"row":row,"serial_nll":nll(&serial[span.clone()],labels[row]),
                "baseline_nll":nll(&baseline[span.clone()],labels[row]),
                "candidate_nll":nll(&candidate[span.clone()],labels[row]),
                "baseline_vs_serial":distribution_difference(&serial[span.clone()],&baseline[span.clone()]),
                "candidate_vs_serial":distribution_difference(&serial[span.clone()],&candidate[span])})
        }).collect::<Vec<_>>(),
        "baseline_continuation_vs_serial":score(serial_future,baseline_future,&labels[width..],vocab),
        "candidate_continuation_vs_serial":score(serial_future,candidate_future,&labels[width..],vocab),
        "scope":"same prefix; held next-token labels; native scalar precision is a reference, not author truth",
        "timing_eligible":false,"quality_threshold":null})
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn serial_witness(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    tokens: &[u32],
    labels: &[u32],
    mode: Mode,
    audit: &Audit,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) {
    assert!(restore(
        runner,
        base,
        frozen,
        tokens.len(),
        "serial_quality"
    ));
    let _scope = Scope::new(Mode::Baseline);
    let (serial, _) = run_arm(runner, tokens, "serial", captures, logits);
    let future = bits(
        runner
            .forward_token(labels[tokens.len() - 1])
            .unwrap()
            .as_slice(),
    );
    cheap_report(
        base.length(),
        tokens.len(),
        mode,
        &bits(&serial),
        &future,
        &audit.baseline.rows,
        &audit.baseline.future,
        &audit.candidate.rows,
        &audit.candidate.future,
        labels,
        runner.weights.geometry.vocab_size(),
    );
}

fn ranked(scores: &[f32]) -> Vec<usize> {
    assert!(scores.iter().all(|x| x.is_finite()));
    let mut order: Vec<_> = (0..scores.len()).collect();
    // moe_better_pair: descending raw score, smaller expert ID on ties (including signed zero).
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    order
}

fn route_row(scores: &[f32], ids: &[i32], weights: &[f32]) -> serde_json::Value {
    assert_eq!(scores.len(), EXPERTS);
    assert_eq!(ids.len(), TOP_K);
    assert_eq!(weights.len(), TOP_K);
    let order = ranked(scores);
    assert!(
        ids.iter()
            .zip(&order)
            .all(|(&id, &expected)| id == expected as i32),
        "captured top-k disagrees with raw router scores"
    );
    assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    let max = f64::from(scores[order[0]]);
    let mass: Vec<_> = order[..TOP_K]
        .iter()
        .map(|&id| (f64::from(scores[id]) - max).exp())
        .collect();
    let sum: f64 = mass.iter().sum();
    let weight_error = weights
        .iter()
        .zip(mass)
        .map(|(&w, m)| (f64::from(w) - m / sum).abs())
        .fold(0.0, f64::max);
    json!({"ids":ids,"weights":weights,"weight_sum":weights.iter().map(|&w|f64::from(w)).sum::<f64>(),
        "softmax_f64_max_abs":weight_error,"cut_selected":order[TOP_K-1],"cut_excluded":order[TOP_K],
        "cut_margin":f64::from(scores[order[TOP_K-1]])-f64::from(scores[order[TOP_K]]),
        "top11":order[..=TOP_K].iter().map(|&id|json!({"expert":id,"score":scores[id]})).collect::<Vec<_>>()})
}

fn route_counts(reference: &[i32], actual: &[i32], width: usize) -> serde_json::Value {
    assert_eq!(reference.len(), LAYERS.len() * width * TOP_K);
    assert_eq!(reference.len(), actual.len());
    json!(
        LAYERS
            .iter()
            .enumerate()
            .map(|(index, &layer)| {
                let span = index * width * TOP_K..(index + 1) * width * TOP_K;
                let (a, b) = (&reference[span.clone()], &actual[span]);
                let histogram = |ids: &[i32]| {
                    let mut counts = [0_i32; EXPERTS];
                    for &id in ids {
                        assert!((0..EXPERTS as i32).contains(&id));
                        counts[id as usize] += 1;
                    }
                    counts
                };
                let (ha, hb) = (histogram(a), histogram(b));
                json!({"layer":layer,"selected_routes":width*TOP_K,
            "rows_changed":a.chunks_exact(TOP_K).zip(b.chunks_exact(TOP_K))
                .filter(|(a,b)|a.iter().any(|id|!b.contains(id))).count(),
            "changed_experts":ha.iter().zip(&hb).filter(|(a,b)|a!=b).count(),
            "histogram_l1":ha.iter().zip(&hb).map(|(a,b)|(a-b).abs()).sum::<i32>()})
            })
            .collect::<Vec<_>>()
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn router_witness(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    tokens: &[u32],
    next: u32,
    mode: Mode,
    audit: &Audit,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) {
    let bank = Bank::new(runner.ctx, tokens.len());
    let mut records = Vec::new();
    for (m, expected, label) in [
        (Mode::Baseline, &audit.baseline, "router_baseline"),
        (mode, &audit.candidate, "router_candidate"),
    ] {
        assert!(restore(runner, base, frozen, tokens.len(), label));
        let _scope = Scope::new(m);
        let ((values, _), seen) = router_capture::capture(&bank, || {
            run_arm(runner, tokens, "verify_all", captures, logits)
        });
        assert_eq!(seen, [true; 2]);
        records.push((
            completed_values::<f32>(&bank.scores),
            completed_values::<i32>(&bank.ids),
            completed_values::<f32>(&bank.weights),
        ));
        let packet_state = state(runner, runner.next_position());
        let future = bits(runner.forward_token(next).unwrap().as_slice());
        let observed = Observation {
            rows: bits(&values),
            future,
            packet_state,
            future_state: state(runner, runner.next_position()),
        };
        assert!(
            repeatability::compare(
                base.length(),
                tokens.len(),
                m,
                label,
                expected,
                &observed,
                runner.weights.geometry.vocab_size()
            ),
            "raw router capture changed execution"
        );
    }
    for (layer_index, &layer) in LAYERS.iter().enumerate() {
        for row in 0..tokens.len() {
            let r = layer_index * tokens.len() + row;
            let (a, b) = (&records[0], &records[1]);
            let (sa, sb) = (
                &a.0[r * EXPERTS..(r + 1) * EXPERTS],
                &b.0[r * EXPERTS..(r + 1) * EXPERTS],
            );
            let (ia, ib) = (
                &a.1[r * TOP_K..(r + 1) * TOP_K],
                &b.1[r * TOP_K..(r + 1) * TOP_K],
            );
            let switched: Vec<_> = (0..EXPERTS).filter(|&id| ia.contains(&(id as i32)) != ib.contains(&(id as i32)))
                .map(|id|json!({"expert":id,"baseline_score":sa[id],"candidate_score":sb[id],"baseline_selected":ia.contains(&(id as i32))})).collect();
            eprintln!(
                "packet_router_margin {}",
                json!({"depth":base.length(),"width":tokens.len(),"hc_mode":mode.name(),"layer":layer,"row":row,
                "baseline":route_row(sa,ia,&a.2[r*TOP_K..(r+1)*TOP_K]),"candidate":route_row(sb,ib,&b.2[r*TOP_K..(r+1)*TOP_K]),
                "max_score_delta":sa.iter().zip(sb).map(|(&a,&b)|(f64::from(a)-f64::from(b)).abs()).fold(0.0,f64::max),
                "ordered_ids_equal":ia==ib,"switched":switched,"timing_eligible":false,
                "scope":"same model weights, raw packed router logits, selected-softmax weights; no near-tie threshold"})
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rolling(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    base: &Qwen4ExpTextSnapshot,
    frozen: &State,
    stream: &[u32],
    width: usize,
    mode: Mode,
    audit: &Audit,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) {
    const N: usize = 64;
    assert_eq!(stream.len(), N + 2);
    assert!(matches!(width, 2 | 4 | 8));
    let vocab = runner.weights.geometry.vocab_size();
    assert!(restore(runner, base, frozen, width, "quality64_serial"));
    let serial = {
        let _scope = Scope::new(Mode::Baseline);
        let (values, _) = run_arm(runner, &stream[..N], "serial", captures, logits);
        bits(&values)
    };
    let serial_mean_nll = serial
        .chunks_exact(vocab)
        .zip(&stream[1..=N])
        .map(|(row, &label)| nll(row, label))
        .sum::<f64>()
        / N as f64;
    let serial_future = bits(runner.forward_token(stream[N]).unwrap().as_slice());
    let bank = Bank::new(runner.ctx, width);
    let mut baseline_routes = Vec::<Vec<i32>>::new();
    for (m, label, expected) in [
        (Mode::Baseline, "baseline_packed", &audit.baseline),
        (mode, "candidate_packed", &audit.candidate),
    ] {
        assert!(restore(runner, base, frozen, width, label));
        let _scope = Scope::new(m);
        let (mut total_nll, mut total_kl, mut reverse_kl, mut agreements, mut route_switched_rows) =
            (0.0, 0.0, 0.0, 0, 0);
        for (packet, input) in stream[..N].chunks_exact(width).enumerate() {
            let start = packet * width;
            let ((values, _), seen) = router_capture::capture(&bank, || {
                run_arm(runner, input, "verify_all", captures, logits)
            });
            assert_eq!(seen, [true; 2]);
            assert_eq!(runner.next_position(), base.length() + start + width);
            let values = bits(&values);
            if packet == 0 {
                assert!(
                    values == expected.rows,
                    "quality capture first packet changed same-mode logits"
                );
            }
            let ids = completed_values::<i32>(&bank.ids);
            let switched = if label == "baseline_packed" {
                baseline_routes.push(ids.clone());
                0
            } else {
                ids.chunks_exact(TOP_K)
                    .zip(baseline_routes[packet].chunks_exact(TOP_K))
                    .filter(|(a, b)| a.iter().any(|id| !b.contains(id)))
                    .count()
            };
            route_switched_rows += switched;
            let counts = route_counts(&baseline_routes[packet], &ids, width);
            let metrics = score(
                &serial[start * vocab..(start + width) * vocab],
                &values,
                &stream[start + 1..start + width + 1],
                vocab,
            );
            total_nll += metrics["mean_nll"].as_f64().unwrap() * width as f64;
            total_kl += metrics["mean_kl_reference_actual"].as_f64().unwrap() * width as f64;
            reverse_kl += metrics["mean_kl_actual_reference"].as_f64().unwrap() * width as f64;
            agreements += metrics["top1_agreements"].as_u64().unwrap();
            eprintln!(
                "packet_quality_chunk {}",
                json!({"depth":base.length(),"width":width,"hc_mode":m.name(),"arm":label,"packet":packet,
                "consumed_start":base.length()+start,"metrics":metrics,"route_layers":LAYERS,"route_rows_changed_vs_baseline":switched,
                "route_counts_vs_baseline":counts,
                "timing_eligible":false,"state_restored_between_packets":false})
            );
        }
        assert_eq!(runner.next_position(), base.length() + N);
        let future = bits(runner.forward_token(stream[N]).unwrap().as_slice());
        eprintln!(
            "packet_quality_summary {}",
            json!({"depth":base.length(),"width":width,"hc_mode":m.name(),"arm":label,"held_tokens":N,
            "mean_nll":total_nll/N as f64,"serial_mean_nll":serial_mean_nll,"mean_nll_delta":total_nll/N as f64-serial_mean_nll,
            "mean_kl_serial_actual":total_kl/N as f64,"mean_kl_actual_serial":reverse_kl/N as f64,"top1_agreements":agreements,
            "continuation":score(&serial_future,&future,&stream[N+1..],vocab),"route_layers":LAYERS,
            "route_rows_changed_vs_baseline":route_switched_rows,"timing_eligible":false,"quality_threshold":null,
            "scope":"64 common corpus inputs, one prefix restore per arm, native scalar reference not author truth; no sampled rollout claim"})
        );
    }
}

#[test]
fn held_label_alignment_and_router_ties() {
    let a = bits(&[0.0, 0.0, 0.0, 0.0]);
    let result = score(&a, &a, &[0, 1], 2);
    assert_eq!(result["rows"], 2);
    assert_eq!(result["mean_nll"], 2.0_f64.ln());
    assert_eq!(result["mean_nll_delta"], 0.0);
    assert_eq!(ranked(&[0.0, -0.0, 2.0, 2.0]), [2, 3, 0, 1]);
    assert!(std::panic::catch_unwind(|| nll(&a, 4)).is_err());
    let scores = vec![0.0; EXPERTS];
    let ids: Vec<i32> = (0..TOP_K as i32).collect();
    let route = route_row(&scores, &ids, &[0.1; TOP_K]);
    assert_eq!(route["cut_margin"], 0.0);
    assert_eq!(route["cut_selected"], 9);
    assert_eq!(route["cut_excluded"], 10);
    let routes: Vec<_> = ids.repeat(4);
    let mut changed = routes.clone();
    changed[0] = 20;
    let counts = route_counts(&routes, &changed, 2);
    assert_eq!(counts[0]["rows_changed"], 1);
    assert_eq!(counts[0]["changed_experts"], 2);
    assert_eq!(counts[0]["histogram_l1"], 2);
    assert_eq!(counts[1]["rows_changed"], 0);
}
