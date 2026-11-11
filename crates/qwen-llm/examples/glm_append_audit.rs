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
//! Usage: `glm_append_audit <depth> <prompt> [prompt ...]`

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
            Glm5NextRowBank, Glm5NextRowBankPlan, Glm5NextSession, Glm5NextWeights,
            set_expert_packet_max_rows,
        },
        metal::*,
        tokenizer::Tokenizer,
    };
    use serde_json::{Value, json};
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

    fn audit_prompt(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        tokenizer: &Tokenizer,
        path: &str,
        text: &str,
        depth: usize,
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
        let plan = Glm5NextRowBankPlan::new(&weights.config, &[], 0, MAX_WIDTH)?;
        let mut bank = Glm5NextRowBank::new(ctx, weights, plan)?;
        let vocab = weights.config.vocab_size as usize;

        for width in WIDTHS {
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
            for &token in append {
                reference.push(session.forward(ctx, token)?);
            }
            let reference_state = session.forward(ctx, next)?;
            let reference_ms = started.elapsed().as_secs_f64() * 1e3;
            emit(json!({
                "kind": "reference", "path": path, "width": width,
                "wall_ms": reference_ms, "pressure": pressure(),
            }));

            for (label, max_rows) in [("grouped", GROUPED_MAX_ROWS), ("packet", PACKET_MAX_ROWS)] {
                gate();
                set_expert_packet_max_rows(max_rows);
                session.restore_snapshot(&snapshot)?;
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
                drop(captured);
                let path_logits: Vec<Vec<f32>> = (0..width)
                    .map(|r| flat[r * vocab..(r + 1) * vocab].to_vec())
                    .collect();
                let path_state = session.forward(ctx, next)?;
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
                    "next_token": next, "next2_token": next2,
                    "pressure": pressure(), "loadavg": loadavg(),
                }));
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
        anyhow::ensure!(
            args.len() >= 3,
            "usage: glm_append_audit <depth> <prompt> [prompt ...]"
        );
        let depth: usize = args[1].parse()?;
        let ctx = MetalContext::new()?;
        let model = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
            .path_or_skip()
            .ok_or_else(|| anyhow::anyhow!("GLM fixture unavailable"))?;
        let gguf = GgufFile::open(model)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = Glm5NextWeights::load(&ctx, &gguf)?;
        emit(json!({
            "kind": "header", "depth": depth, "widths": WIDTHS,
            "capacity": CAPACITY, "prefill_rows": PREFILL_ROWS,
            "device": ctx.device_facts(), "pressure": pressure(), "loadavg": loadavg(),
        }));
        for path in &args[2..] {
            let text = std::fs::read_to_string(path)?;
            audit_prompt(&ctx, &weights, &tokenizer, path, &text, depth)?;
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
