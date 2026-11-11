//! GLM append numerical audit; build with `--features perf-ablations`.
//!
//! For each real prompt, a 48K prefix is captured once. Appends of 8..=64 rows
//! are then processed three ways from that snapshot:
//!   - reference: the 1-row decode path, one token at a time (the qualified
//!     path);
//!   - grouped: the production packed-prefill path (grouped expert buckets);
//!   - packet: the half-operand per-slot expert packet.
//!
//! For every appended position we compare the next-token distribution against
//! the reference (signed teacher-forced dNLL and both KL directions), then feed
//! one common token through ordinary decode to compare the resulting state.
//! Output is JSON lines. No production change.
//!
//! Usage: `glm_append_audit [--layers|--layers-f32] <depth> <prompt> [prompt ...]`
//! `--layers` compares every post-block activation for sequential decode,
//! one-row packed control, and the 8-row packed schedule. `--layers-f32`
//! compares an F32 packed suffix against the same Fast/Half prefix and
//! sequential reference.

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}

#[cfg(feature = "perf-ablations")]
mod probe {
    use objc2_metal::MTLBuffer;
    use qwen_llm::{
        gguf::GgufFile,
        glm5_next_metal::{
            FastPrecision, Glm5NextCapturedRows, Glm5NextRowBank, Glm5NextRowBankPlan,
            Glm5NextSession, Glm5NextWeights, PackedLineage, set_expert_packet_max_rows,
        },
        metal::*,
        tokenizer::Tokenizer,
    };
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{io::Write, time::Instant};

    const CAPACITY: usize = 65536;
    const PREFILL_ROWS: usize = 2048;
    const MAX_WIDTH: usize = 64;
    const WIDTHS: [usize; 7] = [8, 12, 16, 24, 32, 48, 64];
    const GROUPED_MAX_ROWS: usize = 8;
    const PACKET_MAX_ROWS: usize = 128;

    fn emit(v: Value) {
        println!("{v}");
        std::io::stdout().flush().unwrap();
    }

    fn pressure() -> i32 {
        std::process::Command::new("notifyutil")
            .args(["-g", "com.apple.system.thermalpressurelevel"])
            .output()
            .ok()
            .and_then(|x| {
                String::from_utf8(x.stdout)
                    .ok()?
                    .split_whitespace()
                    .last()?
                    .parse()
                    .ok()
            })
            .unwrap_or(-1)
    }

    fn loadavg() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|x| String::from_utf8(x.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }

    fn gate() {
        while pressure() > 1 {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    fn read_f32(t: &MetalTensor) -> Vec<f32> {
        let bytes = t.n_bytes() as usize;
        assert_eq!(bytes % 4, 0, "F32 view");
        unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<f32>(),
                bytes / 4,
            )
            .to_vec()
        }
    }

    fn log_softmax(x: &[f32]) -> Vec<f64> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = x.iter().map(|v| (*v as f64 - max).exp()).sum();
        let logsum = max + sum.ln();
        x.iter().map(|v| *v as f64 - logsum).collect()
    }

    fn kl(reference: &[f32], other: &[f32]) -> f64 {
        let (p, q) = (log_softmax(reference), log_softmax(other));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
    }

    fn log_prob(logits: &[f32], token: u32) -> f64 {
        log_softmax(logits)[token as usize]
    }

    fn top1(logits: &[f32]) -> usize {
        let mut best = 0;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        best
    }

    struct PositionMetrics {
        dnll: Vec<f64>,
        kl_path_ref: Vec<f64>,
        kl_ref_path: Vec<f64>,
        top1: Vec<bool>,
    }

    fn position_metrics(
        reference: &[Vec<f32>],
        path: &[Vec<f32>],
        actual: &[u32],
    ) -> PositionMetrics {
        assert_eq!(reference.len(), path.len());
        let mut m = PositionMetrics {
            dnll: Vec::with_capacity(path.len()),
            kl_path_ref: Vec::with_capacity(path.len()),
            kl_ref_path: Vec::with_capacity(path.len()),
            top1: Vec::with_capacity(path.len()),
        };
        for (i, (r, p)) in reference.iter().zip(path).enumerate() {
            m.dnll.push(log_prob(r, actual[i]) - log_prob(p, actual[i]));
            m.kl_path_ref.push(kl(p, r));
            m.kl_ref_path.push(kl(r, p));
            m.top1.push(top1(r) == top1(p));
        }
        m
    }

    fn aggregates(values: &[f64]) -> (f64, f64, usize) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let (worst, &max) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("nonempty");
        (mean, max, worst)
    }

    fn state_metrics(reference: &[f32], path: &[f32], next2: u32) -> (f64, f64, f64, bool) {
        (
            log_prob(reference, next2) - log_prob(path, next2),
            kl(reference, path),
            kl(path, reference),
            top1(reference) == top1(path),
        )
    }

    fn capture_logits(rows: &Glm5NextCapturedRows<'_>) -> Vec<f32> {
        read_f32(rows.logits().expect("captured logits"))
    }

    fn capture_features(rows: &Glm5NextCapturedRows<'_>) -> Vec<f32> {
        read_f32(rows.features().expect("captured features"))
    }

    fn layer_deltas(
        reference: &[f32],
        path: &[f32],
        rows: usize,
        taps: &[u32],
        hidden: usize,
        config: &qwen_llm::glm5_next::Glm5NextConfig,
    ) -> Vec<Value> {
        let stride = taps.len() * hidden;
        assert_eq!(reference.len(), rows * stride);
        assert_eq!(path.len(), rows * stride);
        taps.iter()
            .enumerate()
            .map(|(tap_index, &layer)| {
                let mut sum_sq = 0.0f64;
                let mut ref_sq = 0.0f64;
                let mut max_abs = 0.0f32;
                let mut differing = 0usize;
                let mut values = 0usize;
                for row in 0..rows {
                    let base = row * stride + tap_index * hidden;
                    for i in 0..hidden {
                        let a = reference[base + i];
                        let b = path[base + i];
                        let delta = (b - a).abs();
                        max_abs = max_abs.max(delta);
                        sum_sq += (delta as f64) * (delta as f64);
                        ref_sq += (a as f64) * (a as f64);
                        differing += usize::from(a.to_bits() != b.to_bits());
                        values += 1;
                    }
                }
                let rms = (sum_sq / values as f64).sqrt();
                let relative_rms = (sum_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt();
                let block = config.blocks[layer as usize];
                json!({
                    "layer": layer,
                    "mixer": format!("{:?}", block.mixer),
                    "ffn": format!("{:?}", block.ffn),
                    "different_values": differing,
                    "values": values,
                    "max_abs": max_abs,
                    "rms": rms,
                    "relative_rms": relative_rms,
                })
            })
            .collect()
    }

    fn route_rows(
        routes: Vec<(usize, Vec<i32>)>,
        rows: usize,
        top_k: usize,
    ) -> Vec<Vec<(usize, Vec<i32>)>> {
        let mut result = vec![Vec::new(); rows];
        for (block, ids) in routes {
            assert_eq!(ids.len(), rows * top_k, "route block {block} size");
            for row in 0..rows {
                result[row].push((block, ids[row * top_k..(row + 1) * top_k].to_vec()));
            }
        }
        result
    }

    fn route_deltas(
        reference: &[Vec<(usize, Vec<i32>)>],
        path: &[Vec<(usize, Vec<i32>)>],
    ) -> Vec<Value> {
        assert_eq!(reference.len(), path.len());
        let mut blocks: Vec<usize> = reference
            .iter()
            .flatten()
            .map(|(block, _)| *block)
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        blocks
            .into_iter()
            .map(|block| {
                let mut changed_slots = 0usize;
                let mut changed_rows = 0usize;
                let mut total_slots = 0usize;
                for row in 0..reference.len() {
                    let reference_ids = reference[row]
                        .iter()
                        .find(|(found, _)| *found == block)
                        .map(|(_, ids)| ids.as_slice())
                        .expect("reference route layer");
                    let path_ids = path[row]
                        .iter()
                        .find(|(found, _)| *found == block)
                        .map(|(_, ids)| ids.as_slice())
                        .expect("path route layer");
                    assert_eq!(reference_ids.len(), path_ids.len());
                    let changed = reference_ids
                        .iter()
                        .zip(path_ids)
                        .filter(|(a, b)| a != b)
                        .count();
                    changed_slots += changed;
                    changed_rows += usize::from(changed > 0);
                    total_slots += reference_ids.len();
                }
                json!({
                    "layer": block,
                    "changed_rows": changed_rows,
                    "rows": reference.len(),
                    "changed_slots": changed_slots,
                    "slots": total_slots,
                })
            })
            .collect()
    }

    fn audit_prompt(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        tokenizer: &Tokenizer,
        path: &str,
        text: &str,
        depth: usize,
        layer_audit: bool,
        f32_suffix: bool,
    ) -> anyhow::Result<()> {
        let escaped = text.replace("<|", "‹|").replace("<｜", "‹｜");
        let mut ids: Vec<u32> = tokenizer
            .encode(&escaped, false)?
            .into_iter()
            .map(|t| t as u32)
            .collect();
        anyhow::ensure!(!ids.is_empty(), "empty prompt {path}");
        while ids.len() < depth + MAX_WIDTH + 2 {
            let mut next = ids.clone();
            ids.append(&mut next);
        }
        let preview: String = escaped
            .chars()
            .take(120)
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        emit(json!({
            "kind": "prompt", "path": path, "tokens": ids.len(), "depth": depth,
            "preview": preview, "pressure": pressure(), "loadavg": loadavg(),
        }));

        let mut session = Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, PREFILL_ROWS)?;
        for part in ids[..depth].chunks(PREFILL_ROWS) {
            gate();
            session.prefill_packed(ctx, part)?;
        }
        anyhow::ensure!(session.position() == depth, "prefix position");
        let snapshot = session.capture_snapshot()?;
        let taps: Vec<u32> = if layer_audit {
            (0..weights.config.blocks.len() as u32).collect()
        } else {
            Vec::new()
        };
        let feature_capacity = if layer_audit { 8 } else { 0 };
        let plan = Glm5NextRowBankPlan::new(&weights.config, &taps, feature_capacity, MAX_WIDTH)?;
        let mut bank = Glm5NextRowBank::new(ctx, weights, plan)?;
        let vocab = weights.config.vocab_size as usize;

        let widths: &[usize] = if layer_audit { &[8] } else { &WIDTHS };
        for &width in widths {
            let append = &ids[depth..depth + width];
            let actual: Vec<u32> = ids[depth + 1..depth + width + 1].to_vec();
            let next = ids[depth + width];
            let next2 = ids[depth + width + 1];

            // Reference: the qualified 1-row decode path, one token at a time.
            gate();
            set_expert_packet_max_rows(GROUPED_MAX_ROWS);
            session.restore_snapshot(&snapshot)?;
            let started = Instant::now();
            let mut reference = Vec::with_capacity(width);
            let mut reference_features = Vec::new();
            let mut reference_routes = Vec::new();
            for &token in append {
                if layer_audit {
                    let captured = session.forward_with_captures(ctx, token, &mut bank)?;
                    reference.push(capture_logits(&captured));
                    reference_features.extend(capture_features(&captured));
                    reference_routes.push(session.perf_decode_route_ids()?);
                } else {
                    reference.push(session.forward(ctx, token)?);
                }
            }
            let reference_state = if layer_audit {
                let captured = session.forward_with_captures(ctx, next, &mut bank)?;
                capture_logits(&captured)
            } else {
                session.forward(ctx, next)?
            };
            let reference_ms = started.elapsed().as_secs_f64() * 1e3;
            emit(json!({
                "kind": "reference", "path": path, "width": width,
                "wall_ms": reference_ms, "pressure": pressure(),
            }));

            let mut control_logits = Vec::new();
            let mut control_features = Vec::new();
            let mut control_routes = Vec::new();
            if layer_audit {
                // Same-lineage one-row packed calls are a legal alternative
                // schedule to ordinary one-token decode. This control
                // measures the observed schedule delta before comparing an
                // 8-row packed call with the sequential reference.
                gate();
                set_expert_packet_max_rows(PACKET_MAX_ROWS);
                session.restore_snapshot(&snapshot)?;
                for &token in append {
                    let captured = session.verify_packed_with_captures(ctx, &[token], &mut bank)?;
                    control_logits.extend(capture_logits(&captured));
                    control_features.extend(capture_features(&captured));
                    control_routes.push(
                        route_rows(
                            session.perf_packed_route_ids(1)?,
                            1,
                            weights.config.expert_used_count as usize,
                        )[0]
                        .clone(),
                    );
                }
                let control_rows: Vec<Vec<f32>> =
                    control_logits.chunks(vocab).map(<[f32]>::to_vec).collect();
                let m = position_metrics(&reference, &control_rows, &actual);
                let abs_dnll: Vec<f64> = m.dnll.iter().map(|x| x.abs()).collect();
                let (mean_abs_dnll, max_abs_dnll, _) = aggregates(&abs_dnll);
                let (mean_kl, max_kl, _) = aggregates(&m.kl_path_ref);
                let control_state = session.forward(ctx, next)?;
                let (state_dnll, state_kl_path_ref, state_kl_ref_path, state_top1_agree) =
                    state_metrics(&reference_state, &control_state, next2);
                emit(json!({
                    "kind": "serial_control", "prompt": path, "width": width,
                    "schedule": "one_row_packed_vs_forward_decode",
                    "mean_abs_dnll": mean_abs_dnll, "max_abs_dnll": max_abs_dnll,
                    "mean_kl_path_ref": mean_kl, "max_kl_path_ref": max_kl,
                    "state_dnll": state_dnll,
                    "state_kl_path_ref": state_kl_path_ref,
                    "state_kl_ref_path": state_kl_ref_path,
                    "state_top1_agree": state_top1_agree,
                    "layer_deltas": layer_deltas(
                        &reference_features,
                        &control_features,
                        width,
                        &taps,
                        weights.config.hidden_size as usize,
                        &weights.config,
                    ),
                    "route_deltas": route_deltas(&reference_routes, &control_routes),
                    "pressure": pressure(),
                }));
            }

            for (label, max_rows) in [("grouped", GROUPED_MAX_ROWS), ("packet", PACKET_MAX_ROWS)] {
                gate();
                set_expert_packet_max_rows(max_rows);
                session.restore_snapshot(&snapshot)?;
                if f32_suffix {
                    session.perf_override_fast_precision(FastPrecision::F32)?;
                }
                let started = Instant::now();
                let captured = session.verify_packed_with_captures(ctx, append, &mut bank)?;
                let logits = captured.logits().expect("all-row logits");
                let flat = read_f32(logits);
                anyhow::ensure!(
                    flat.len() == width * vocab,
                    "logit view {} != {}",
                    flat.len(),
                    width * vocab
                );
                let packed_features = if layer_audit {
                    Some(capture_features(&captured))
                } else {
                    None
                };
                let packed_routes = if layer_audit {
                    Some(route_rows(
                        session.perf_packed_route_ids(width)?,
                        width,
                        weights.config.expert_used_count as usize,
                    ))
                } else {
                    None
                };
                drop(captured);
                let path_logits: Vec<Vec<f32>> = (0..width)
                    .map(|r| flat[r * vocab..(r + 1) * vocab].to_vec())
                    .collect();
                if f32_suffix {
                    session.perf_override_fast_precision(FastPrecision::Half)?;
                }
                let path_state = if layer_audit {
                    let captured = session.forward_with_captures(ctx, next, &mut bank)?;
                    capture_logits(&captured)
                } else {
                    session.forward(ctx, next)?
                };
                let wall_ms = started.elapsed().as_secs_f64() * 1e3;

                let m = position_metrics(&reference, &path_logits, &actual);
                let (mean_dnll, max_dnll, worst_dnll) = aggregates(&m.dnll);
                let (mean_kl_pr, max_kl_pr, worst_kl_pr) = aggregates(&m.kl_path_ref);
                let (mean_kl_rp, max_kl_rp, worst_kl_rp) = aggregates(&m.kl_ref_path);
                let (final_dnll, final_kl_pr, final_kl_rp, final_top1) =
                    state_metrics(&reference[width - 1], &path_logits[width - 1], next);
                let (state_dnll, state_kl_pr, state_kl_rp, state_top1) =
                    state_metrics(&reference_state, &path_state, next2);
                let top1_agree = m.top1.iter().filter(|v| **v).count();
                emit(json!({
                    "kind": "audit", "prompt": path, "width": width, "path": label,
                    "selection_max_rows": max_rows,
                    "wall_ms": wall_ms,
                    "positions": width,
                    "mean_dnll": mean_dnll, "max_dnll": max_dnll,
                    "worst_dnll_index": worst_dnll,
                    "mean_kl_path_ref": mean_kl_pr, "max_kl_path_ref": max_kl_pr,
                    "worst_kl_path_ref_index": worst_kl_pr,
                    "mean_kl_ref_path": mean_kl_rp, "max_kl_ref_path": max_kl_rp,
                    "worst_kl_ref_path_index": worst_kl_rp,
                    "top1_agree": top1_agree,
                    "dnll": m.dnll, "kl_path_ref": m.kl_path_ref,
                    "kl_ref_path": m.kl_ref_path,
                    "final_dnll": final_dnll, "final_kl_path_ref": final_kl_pr,
                    "final_kl_ref_path": final_kl_rp, "final_top1_agree": final_top1,
                    "state_dnll": state_dnll, "state_kl_path_ref": state_kl_pr,
                    "state_kl_ref_path": state_kl_rp, "state_top1_agree": state_top1,
                    "layer_deltas": if let Some(features) = packed_features {
                        json!(layer_deltas(
                            &reference_features,
                            &features,
                            width,
                            &taps,
                            weights.config.hidden_size as usize,
                            &weights.config,
                        ))
                    } else {
                        Value::Null
                    },
                    "route_deltas": packed_routes
                        .map(|routes| json!(route_deltas(&reference_routes, &routes)))
                        .unwrap_or(Value::Null),
                    "next_token": next, "next2_token": next2,
                    "pressure": pressure(), "loadavg": loadavg(),
                }));
            }
        }
        Ok(())
    }

    fn route_set_deltas(
        reference: &[Vec<(usize, Vec<i32>)>],
        path: &[Vec<(usize, Vec<i32>)>],
    ) -> Vec<Value> {
        assert_eq!(reference.len(), path.len());
        let mut blocks: Vec<usize> = reference
            .iter()
            .flatten()
            .map(|(block, _)| *block)
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        blocks
            .into_iter()
            .map(|block| {
                let mut changed_rows = 0usize;
                let mut changed_slots = 0usize;
                for row in 0..reference.len() {
                    let mut a = reference[row]
                        .iter()
                        .find(|(found, _)| *found == block)
                        .map(|(_, ids)| ids.clone())
                        .expect("reference route block");
                    let mut b = path[row]
                        .iter()
                        .find(|(found, _)| *found == block)
                        .map(|(_, ids)| ids.clone())
                        .expect("path route block");
                    a.sort_unstable();
                    b.sort_unstable();
                    let changed = a.iter().zip(&b).filter(|(x, y)| x != y).count();
                    changed_rows += usize::from(changed > 0);
                    changed_slots += changed;
                }
                json!({
                    "layer": block,
                    "changed_rows": changed_rows,
                    "rows": reference.len(),
                    "changed_slots": changed_slots,
                    "slots": reference.len() * 8,
                })
            })
            .collect()
    }

    fn quality_result(
        prompt: &str,
        stratum: &str,
        arm: &str,
        width: usize,
        exact: &[Vec<f32>],
        sequential_fast: &[Vec<f32>],
        candidate: &[Vec<f32>],
        actual: &[u32],
        exact_routes: &[Vec<(usize, Vec<i32>)>],
        candidate_routes: &[Vec<(usize, Vec<i32>)>],
        wall_ms: Option<f64>,
    ) {
        let exact = &exact[..width];
        let sequential_fast = &sequential_fast[..width];
        let candidate = &candidate[..width];
        let actual = &actual[..width];
        let exact_m = position_metrics(exact, exact, actual);
        let seq_m = position_metrics(exact, sequential_fast, actual);
        let m = position_metrics(exact, candidate, actual);
        let seq_dnll: Vec<f64> = m
            .dnll
            .iter()
            .zip(&seq_m.dnll)
            .map(|(candidate, sequential)| candidate - sequential)
            .collect();
        emit(json!({
            "kind": "quality_result",
            "prompt": prompt,
            "stratum": stratum,
            "arm": arm,
            "width": width,
            "positions": width,
            "mean_dnll_vs_exact": m.dnll.iter().sum::<f64>() / width as f64,
            "dnll_vs_exact": m.dnll,
            "mean_dnll_vs_sequential_fast": seq_dnll.iter().sum::<f64>() / width as f64,
            "dnll_vs_sequential_fast": seq_dnll,
            "mean_kl_exact_to_arm": m.kl_ref_path.iter().sum::<f64>() / width as f64,
            "mean_kl_arm_to_exact": m.kl_path_ref.iter().sum::<f64>() / width as f64,
            "max_kl_exact_to_arm": m.kl_ref_path.iter().copied().fold(0.0, f64::max),
            "top1_agree": m.top1.iter().filter(|agree| **agree).count(),
            "top1_rate": m.top1.iter().filter(|agree| **agree).count() as f64 / width as f64,
            "exact_repeat_mean_dnll": exact_m.dnll.iter().sum::<f64>() / width as f64,
            "router_set_changes_vs_exact": route_set_deltas(
                &exact_routes[..width],
                &candidate_routes[..width],
            ),
            "wall_ms": wall_ms,
        }));
    }

    fn audit_quality_prompt(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        tokenizer: &Tokenizer,
        path: &str,
        text: &str,
        depth: usize,
    ) -> anyhow::Result<()> {
        const MAX_ROWS: usize = 64;
        const WIDTHS: [usize; 4] = [8, 16, 32, 64];
        let escaped = text.replace("<|", "‹|").replace("<｜", "‹｜");
        let ids: Vec<u32> = tokenizer
            .encode(&escaped, false)?
            .into_iter()
            .map(|token| token as u32)
            .collect();
        anyhow::ensure!(
            ids.len() >= depth + MAX_ROWS + 1,
            "{path} has {} tokens; need at least {} real prefix + continuation tokens",
            ids.len(),
            depth + MAX_ROWS + 1
        );
        let source_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        let mut token_bytes = Vec::with_capacity(ids.len() * 4);
        for token in &ids {
            token_bytes.extend_from_slice(&token.to_le_bytes());
        }
        let tokens_sha256 = format!("{:x}", Sha256::digest(&token_bytes));
        let append = &ids[depth..depth + MAX_ROWS];
        let actual = &ids[depth + 1..depth + MAX_ROWS + 1];
        let stratum = if path.ends_with("PERF-LOG.md") {
            "prose"
        } else {
            "code"
        };
        emit(json!({
            "kind": "quality_prompt",
            "prompt": path,
            "stratum": stratum,
            "tokens": ids.len(),
            "source_sha256": source_sha256,
            "tokens_sha256": tokens_sha256,
            "depth": depth,
            "append_rows": MAX_ROWS,
            "preview": escaped.chars().take(120).collect::<String>(),
            "pressure": pressure(),
        }));

        let mut exact_session =
            Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, PREFILL_ROWS)?;
        exact_session.set_packed_lineage(PackedLineage::Exact)?;
        let started = Instant::now();
        for part in ids[..depth].chunks(PREFILL_ROWS) {
            gate();
            exact_session.prefill_packed(ctx, part)?;
        }
        let exact_prefix_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut exact_logits = Vec::with_capacity(MAX_ROWS);
        let mut exact_routes = Vec::with_capacity(MAX_ROWS);
        for &token in append {
            gate();
            exact_logits.push(exact_session.forward(ctx, token)?);
            exact_routes.push(exact_session.perf_decode_route_ids()?);
        }
        drop(exact_session);
        emit(json!({
            "kind": "quality_prefix",
            "prompt": path,
            "lineage": "Exact",
            "wall_ms": exact_prefix_ms,
            "pressure": pressure(),
        }));

        let mut fast = Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, PREFILL_ROWS)?;
        fast.set_packed_lineage(PackedLineage::Fast)?;
        fast.set_fast_precision(FastPrecision::Half)?;
        let started = Instant::now();
        for part in ids[..depth].chunks(PREFILL_ROWS) {
            gate();
            fast.prefill_packed(ctx, part)?;
        }
        let fast_prefix_ms = started.elapsed().as_secs_f64() * 1e3;
        let snapshot = fast.capture_snapshot()?;
        let plan = Glm5NextRowBankPlan::new(&weights.config, &[], 0, MAX_ROWS)?;
        let mut bank = Glm5NextRowBank::new(ctx, weights, plan)?;

        fast.restore_snapshot(&snapshot)?;
        let mut sequential_fast = Vec::with_capacity(MAX_ROWS);
        let mut sequential_routes = Vec::with_capacity(MAX_ROWS);
        for &token in append {
            gate();
            sequential_fast.push(fast.forward(ctx, token)?);
            sequential_routes.push(fast.perf_decode_route_ids()?);
        }
        emit(json!({
            "kind": "quality_prefix",
            "prompt": path,
            "lineage": "Fast/Half",
            "wall_ms": fast_prefix_ms,
            "pressure": pressure(),
        }));

        for width in WIDTHS {
            let exact_rows = &exact_logits[..width];
            quality_result(
                path,
                stratum,
                "Exact",
                width,
                &exact_logits,
                &sequential_fast,
                exact_rows,
                actual,
                &exact_routes,
                &exact_routes,
                None,
            );
            quality_result(
                path,
                stratum,
                "sequential_fast_half",
                width,
                &exact_logits,
                &sequential_fast,
                &sequential_fast,
                actual,
                &exact_routes,
                &sequential_routes,
                None,
            );
            for (arm, precision) in [
                ("packed_fast_half", FastPrecision::Half),
                ("packed_dense_f32_suffix", FastPrecision::DenseF32),
                ("packed_f32_suffix", FastPrecision::F32),
            ] {
                fast.restore_snapshot(&snapshot)?;
                if precision != FastPrecision::Half {
                    fast.perf_override_fast_precision(precision)?;
                }
                let started = Instant::now();
                let captured =
                    fast.verify_packed_with_captures(ctx, &append[..width], &mut bank)?;
                let wall_ms = started.elapsed().as_secs_f64() * 1e3;
                let logits = read_f32(captured.logits().expect("packed all-row logits"));
                let path_logits: Vec<Vec<f32>> = logits
                    .chunks(weights.config.vocab_size as usize)
                    .map(<[f32]>::to_vec)
                    .collect();
                let route_ids = fast.perf_packed_route_ids(width)?;
                let routes =
                    route_rows(route_ids, width, weights.config.expert_used_count as usize);
                drop(captured);
                if precision != FastPrecision::Half {
                    fast.perf_override_fast_precision(FastPrecision::Half)?;
                }
                quality_result(
                    path,
                    stratum,
                    arm,
                    width,
                    &exact_logits,
                    &sequential_fast,
                    &path_logits,
                    actual,
                    &exact_routes,
                    &routes,
                    Some(wall_ms),
                );
            }
        }
        Ok(())
    }

    pub fn run() -> anyhow::Result<()> {
        assert!(
            std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
            "canonical lease required"
        );
        let args: Vec<String> = std::env::args().collect();
        let mode = args.get(1).map(String::as_str).unwrap_or_default();
        let layer_audit = matches!(mode, "--layers" | "--layers-f32");
        let quality_cohort = mode == "--cohort";
        let f32_suffix = mode == "--layers-f32";
        let base = if layer_audit || quality_cohort { 2 } else { 1 };
        anyhow::ensure!(
            args.len() >= base + 2,
            "usage: glm_append_audit [--layers|--layers-f32|--cohort] <depth> <prompt> [prompt ...]"
        );
        let depth: usize = args[base].parse()?;
        let ctx = MetalContext::new()?;
        let model = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
            .path_or_skip()
            .ok_or_else(|| anyhow::anyhow!("GLM fixture unavailable"))?;
        let gguf = GgufFile::open(model)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = Glm5NextWeights::load(&ctx, &gguf)?;
        emit(json!({
            "kind": "header", "depth": depth,
            "widths": if quality_cohort { json!([8, 16, 32, 64]) } else { json!(WIDTHS) },
            "layer_audit": layer_audit,
            "quality_cohort": quality_cohort,
            "prefix_precision": "Half",
            "packed_suffix_precision": if f32_suffix { "F32" } else { "Half" },
            "capacity": CAPACITY, "prefill_rows": PREFILL_ROWS,
            "device": ctx.device_facts(), "pressure": pressure(), "loadavg": loadavg(),
        }));
        for path in &args[base + 1..] {
            let text = std::fs::read_to_string(path)?;
            if quality_cohort {
                audit_quality_prompt(&ctx, &weights, &tokenizer, path, &text, depth)?;
            } else {
                audit_prompt(
                    &ctx,
                    &weights,
                    &tokenizer,
                    path,
                    &text,
                    depth,
                    layer_audit,
                    f32_suffix,
                )?;
            }
        }
        emit(json!({
            "kind": "done", "pressure": pressure(), "loadavg": loadavg(),
        }));
        Ok(())
    }
}

#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
