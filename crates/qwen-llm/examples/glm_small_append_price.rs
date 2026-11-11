//! GLM small-append pricing probe; build with `--features perf-ablations`.
//!
//! One model load, two questions:
//!  1. `census`: at 48K, where does a single packed-prefill chunk's wall go?
//!     Ordinary wall, sampled stage spans, and a per-dispatch GPU census for
//!     widths 1..=128. The dispatch profile serializes every dispatch, so its
//!     sums are diagnostic attribution, not wall time.
//!  2. `batch`: at 48K, time the independent decode batch at B=1..=4 and
//!     profile its kernel families. The B=1 arm is the serial decode forward.
//!     This prices the row-independent dense/MoE/head work a middle path would
//!     borrow, and shows whether the per-row cost is linear over the only
//!     widths the shipped batch plumbing admits.
//!  3. `threshold`: A/B the half-operand per-slot routed-expert packet against
//!     grouped buckets at widths 8..=96, with full-vocab KL / signed dNLL and
//!     the selected expert kernels at 32 rows.
//!
//! Output is JSON lines. No production scheduling change.
//!
//! Usage: `glm_small_append_price <corpus> [all|census|batch|threshold] [depth]`

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}

#[cfg(feature = "perf-ablations")]
mod probe {
    use qwen_llm::{
        gguf::GgufFile,
        glm5_next_metal::{
            GLM5_NEXT_MAX_BATCH_SIZE, Glm5NextBatch, Glm5NextSession, Glm5NextStageReport,
            Glm5NextWeights, set_expert_packet_max_rows,
        },
        metal::*,
        tokenizer::Tokenizer,
    };
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, io::Write, time::Instant};

    const CAPACITY: usize = 65536;
    const PREFILL_ROWS: usize = 2048;
    const CENSUS_WIDTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 48, 64, 128];
    const THRESHOLD_WIDTHS: [usize; 8] = [8, 12, 16, 24, 32, 48, 64, 96];
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

    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(f64::total_cmp);
        xs[xs.len() / 2]
    }

    fn family(kernel: &str) -> &'static str {
        if kernel.contains("q6_K") || kernel.contains("q6_k") {
            "q6_projection"
        } else if kernel.contains("q8_0") || kernel.contains("q8_0") {
            "q8_projection"
        } else if kernel.contains("mat_mat") || kernel.contains("mat_vec") {
            "matmul"
        } else if kernel.contains("expert_native_gate_up") || kernel.contains("glm_batch_gate_up") {
            "routed_gate_up"
        } else if kernel.contains("expert_native_down") || kernel.contains("glm_batch_down") {
            "routed_down"
        } else if kernel.contains("weighted_sum") || kernel.contains("finalizer") {
            "routed_reduce"
        } else if kernel.contains("indexer_append") || kernel.contains("scatter_offset") {
            "index_cache"
        } else if kernel.contains("indexer") || kernel.contains("index_score") {
            "index_scores"
        } else if kernel.contains("select_top_k") || kernel.contains("expand_selection") {
            "selection"
        } else if kernel.contains("selected_attention")
            || kernel.contains("latent_attention")
            || kernel.contains("attention")
        {
            "attention"
        } else if kernel.contains("kda") {
            "kda"
        } else if kernel.contains("hc_") || kernel.contains("mhc") {
            "hc"
        } else if kernel.contains("route_learned") {
            "router"
        } else if kernel.contains("rms_norm")
            || kernel.contains("l2_norm")
            || kernel.contains("swiglu")
            || kernel.contains("silu")
            || kernel.contains("argmax")
            || kernel.contains("copy")
            || kernel.contains("fill")
            || kernel.contains("add")
            || kernel.contains("scale")
        {
            "pointwise"
        } else {
            "other"
        }
    }

    fn stage_totals(report: &Glm5NextStageReport) -> BTreeMap<&'static str, f64> {
        let mut out = BTreeMap::new();
        for span in &report.spans {
            *out.entry(span.stage.as_str()).or_insert(0.0) += span.gpu_ms;
        }
        out
    }

    fn kernel_totals(
        report: &DispatchProfileReport,
    ) -> (Vec<Value>, BTreeMap<&'static str, (usize, f64)>, usize) {
        let mut by_kernel = BTreeMap::<String, (usize, f64)>::new();
        let mut by_family = BTreeMap::<&'static str, (usize, f64)>::new();
        let mut counted = 0usize;
        for row in &report.rows {
            let ms = row.gpu_ns.unwrap_or(0) as f64 / 1e6;
            let e = by_kernel.entry(row.kernel.clone()).or_default();
            e.0 += 1;
            e.1 += ms;
            let f = by_family.entry(family(&row.kernel)).or_default();
            f.0 += 1;
            f.1 += ms;
            counted += 1;
        }
        let kernels = by_kernel
            .iter()
            .map(|(k, (c, ms))| json!({"kernel": k, "family": family(k), "calls": c, "gpu_ms": ms}))
            .collect();
        (kernels, by_family, counted)
    }

    fn new_session<'a>(
        ctx: &MetalContext,
        weights: &'a Glm5NextWeights,
        ids: &[u32],
        depth: usize,
    ) -> anyhow::Result<Glm5NextSession<'a>> {
        let mut session = Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, PREFILL_ROWS)?;
        for part in ids[..depth].chunks(PREFILL_ROWS) {
            gate();
            session.prefill_packed(ctx, part)?;
        }
        anyhow::ensure!(session.position() == depth, "prefix position mismatch");
        Ok(session)
    }

    fn census(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depth: usize,
    ) -> anyhow::Result<()> {
        let mut session = new_session(ctx, weights, ids, depth)?;
        let snapshot = session.capture_snapshot()?;

        // A decode row is the middle path's floor anchor.
        let mut decode_walls = Vec::new();
        for _ in 0..5 {
            gate();
            session.restore_snapshot(&snapshot)?;
            let t = Instant::now();
            let logits = session.forward(ctx, ids[depth])?;
            decode_walls.push(t.elapsed().as_secs_f64() * 1e3);
            anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite decode");
        }
        let decode_wall = median(decode_walls);
        gate();
        session.restore_snapshot(&snapshot)?;
        let (decode_wall_profiled, decode_report) =
            session.forward_stage_profiled(ctx, ids[depth])?;
        anyhow::ensure!(
            decode_report.spans.iter().all(|s| s.gpu_ms.is_finite()),
            "non-finite decode stage"
        );
        emit(json!({
            "kind": "decode_anchor", "depth": depth,
            "wall_ms": decode_wall, "profiled_wall_ms": decode_wall_profiled,
            "gpu_ms": decode_report.command_gpu_ms,
            "stages": stage_totals(&decode_report),
            "pressure": pressure(), "loadavg": loadavg(),
        }));

        for width in CENSUS_WIDTHS {
            let tokens = &ids[depth..depth + width];
            let mut walls = Vec::new();
            for _ in 0..3 {
                gate();
                session.restore_snapshot(&snapshot)?;
                let t = Instant::now();
                let logits = session.prefill_packed(ctx, tokens)?;
                walls.push(t.elapsed().as_secs_f64() * 1e3);
                anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
            }
            let wall = median(walls);

            gate();
            session.restore_snapshot(&snapshot)?;
            let t = Instant::now();
            let (_logits, staged) = session.prefill_chunk_stage_profiled(ctx, tokens)?;
            let staged_wall = t.elapsed().as_secs_f64() * 1e3;

            gate();
            session.restore_snapshot(&snapshot)?;
            let guard = dispatch_profile_begin(ctx, 16384)?;
            let t = Instant::now();
            let logits = session.prefill_packed(ctx, tokens)?;
            let profiled_wall = t.elapsed().as_secs_f64() * 1e3;
            let report = dispatch_profile_finish(ctx, guard)?;
            anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
            let (kernels, families, counted) = kernel_totals(&report);

            let mut family_json = serde_json::Map::new();
            for (name, (calls, ms)) in &families {
                family_json.insert(
                    (*name).into(),
                    json!({"calls": calls, "gpu_ms": ms, "us_per_row": ms * 1e3 / width as f64}),
                );
            }
            emit(json!({
                "kind": "census", "width": width, "depth": depth,
                "wall_ms": wall, "staged_wall_ms": staged_wall,
                "staged_gpu_ms": staged.command_gpu_ms,
                "staged_span_sum_ms": staged.span_sum_ms,
                "stages": stage_totals(&staged),
                "profile_wall_ms": profiled_wall,
                "profile_sum_ms": families.values().map(|(_, ms)| *ms).sum::<f64>(),
                "dispatches": counted, "overflow": report.overflow,
                "families": family_json,
                "kernels": kernels,
                "pressure": pressure(), "loadavg": loadavg(),
            }));
        }
        Ok(())
    }

    fn batch(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depth: usize,
    ) -> anyhow::Result<()> {
        // Member 0 carries the shared packed activation scratch; the others
        // only need decode state. All four restore the same 48K prefix.
        let mut sessions: Vec<Glm5NextSession> = (0..GLM5_NEXT_MAX_BATCH_SIZE)
            .map(|i| {
                Glm5NextSession::with_prefill_rows(
                    ctx,
                    weights,
                    CAPACITY,
                    if i == 0 { PREFILL_ROWS } else { 4 },
                )
            })
            .collect::<Result<_, _>>()?;
        let seed = new_session(ctx, weights, ids, depth)?;
        let bank = seed.capture_snapshot()?;
        drop(seed);
        for session in sessions.iter_mut() {
            gate();
            session.restore_snapshot(&bank)?;
        }
        let mut batch = Glm5NextBatch::new(ctx, weights, GLM5_NEXT_MAX_BATCH_SIZE)?;
        let token = |row: usize, step: usize| ids[depth + (row * 7 + step) % 128];

        // Warm every admitted width so the first profile is not a first-touch.
        for width in 1..=GLM5_NEXT_MAX_BATCH_SIZE {
            for step in 0..3 {
                batch.step(
                    ctx,
                    &mut sessions[..width].iter_mut().collect::<Vec<_>>(),
                    &(0..width).map(|r| token(r, step)).collect::<Vec<_>>(),
                )?;
            }
        }

        for width in 1..=GLM5_NEXT_MAX_BATCH_SIZE {
            gate();
            for session in sessions[..width].iter_mut() {
                session.restore_snapshot(&bank)?;
            }
            for step in 0..2 {
                batch.step(
                    ctx,
                    &mut sessions[..width].iter_mut().collect::<Vec<_>>(),
                    &(0..width).map(|r| token(r, 16 + step)).collect::<Vec<_>>(),
                )?;
            }
            gate();
            for session in sessions[..width].iter_mut() {
                session.restore_snapshot(&bank)?;
            }
            let tokens: Vec<u32> = (0..width).map(|r| token(r, 32)).collect();
            let guard = dispatch_profile_begin(ctx, 16384)?;
            let t = Instant::now();
            let logits = batch.step(
                ctx,
                &mut sessions[..width].iter_mut().collect::<Vec<_>>(),
                &tokens,
            )?;
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let report = dispatch_profile_finish(ctx, guard)?;
            anyhow::ensure!(
                logits.iter().flatten().all(|v| v.is_finite()),
                "non-finite batch logits"
            );
            let (kernels, families, counted) = kernel_totals(&report);
            let mut family_json = serde_json::Map::new();
            for (name, (calls, ms)) in &families {
                family_json.insert(
                    (*name).into(),
                    json!({"calls": calls, "gpu_ms": ms, "us_per_row": ms * 1e3 / width as f64}),
                );
            }
            emit(json!({
                "kind": "batch", "width": width, "depth": depth,
                "wall_ms": wall, "dispatches": counted, "overflow": report.overflow,
                "profile_sum_ms": families.values().map(|(_, ms)| *ms).sum::<f64>(),
                "families": family_json,
                "kernels": kernels,
                "pressure": pressure(), "loadavg": loadavg(),
            }));
        }
        Ok(())
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

    /// A/B the per-slot half packet against grouped buckets by row count.
    /// Both arms use the same session, snapshot and token prefix; only the
    /// packet row threshold differs.
    fn threshold(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depth: usize,
    ) -> anyhow::Result<()> {
        let mut session = new_session(ctx, weights, ids, depth)?;
        let snapshot = session.capture_snapshot()?;
        for width in THRESHOLD_WIDTHS {
            let tokens = &ids[depth..depth + width];
            // At 8 rows neither threshold selects grouped, so force it (0) to
            // measure the two path's inherent arithmetic delta as a control.
            let grouped_rows = if width == 8 { 0 } else { GROUPED_MAX_ROWS };
            // Warm both selections so neither arm pays a first-touch.
            for max_rows in [grouped_rows, PACKET_MAX_ROWS] {
                set_expert_packet_max_rows(max_rows);
                gate();
                session.restore_snapshot(&snapshot)?;
                let logits = session.prefill_packed(ctx, tokens)?;
                anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite warm");
            }

            let mut walls: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut logits: [Option<Vec<f32>>; 2] = [None, None];
            for rep in 0..3 {
                let order: [usize; 2] = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
                for arm in order {
                    let max_rows = if arm == 0 {
                        grouped_rows
                    } else {
                        PACKET_MAX_ROWS
                    };
                    set_expert_packet_max_rows(max_rows);
                    gate();
                    session.restore_snapshot(&snapshot)?;
                    let t = Instant::now();
                    let out = session.prefill_packed(ctx, tokens)?;
                    walls[arm].push(t.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(out.iter().all(|v| v.is_finite()), "non-finite logits");
                    if rep == 2 {
                        logits[arm] = Some(out);
                    }
                }
            }

            // Name the selected expert kernels once per arm at 8/32/48 rows.
            let mut selected = [Vec::new(), Vec::new()];
            if matches!(width, 8 | 32 | 48) {
                for arm in 0..2 {
                    set_expert_packet_max_rows(if arm == 0 {
                        grouped_rows
                    } else {
                        PACKET_MAX_ROWS
                    });
                    gate();
                    session.restore_snapshot(&snapshot)?;
                    let guard = dispatch_profile_begin(ctx, 16384)?;
                    let out = session.prefill_packed(ctx, tokens)?;
                    anyhow::ensure!(out.iter().all(|v| v.is_finite()), "non-finite profile");
                    let report = dispatch_profile_finish(ctx, guard)?;
                    let mut by_name = BTreeMap::<String, (usize, f64)>::new();
                    for row in &report.rows {
                        if row.kernel.contains("expert") || row.kernel.contains("moe_") {
                            let e = by_name.entry(row.kernel.clone()).or_default();
                            e.0 += 1;
                            e.1 += row.gpu_ns.unwrap_or(0) as f64 / 1e6;
                        }
                    }
                    selected[arm] = by_name
                        .into_iter()
                        .map(|(k, (c, ms))| json!({"kernel": k, "calls": c, "gpu_ms": ms}))
                        .collect();
                }
            }

            let grouped = median(walls[0].clone());
            let packet = median(walls[1].clone());
            let g = logits[0].as_ref().expect("grouped logits");
            let p = logits[1].as_ref().expect("packet logits");
            let next = ids[depth + width];
            let max_abs =
                qwen_llm::compare::report_max_abs_diff_f32(g.iter().copied(), p.iter().copied());
            emit(json!({
                "kind": "threshold", "width": width, "depth": depth,
                "grouped_ms": grouped, "packet_ms": packet,
                "delta_ms": packet - grouped,
                "saving_pct": 100.0 * (grouped - packet) / grouped,
                "grouped_logits_sha": blake3::hash(bytemuck::cast_slice(g)).to_hex().as_str(),
                "packet_logits_sha": blake3::hash(bytemuck::cast_slice(p)).to_hex().as_str(),
                "max_abs_logit_diff": max_abs,
                "top1_grouped": top1(g), "top1_packet": top1(p),
                "top1_agree": top1(g) == top1(p),
                "kl_grouped_packet": kl(g, p), "kl_packet_grouped": kl(p, g),
                "next_token": next,
                "dnll_signed": log_prob(g, next) - log_prob(p, next),
                "grouped_kernels": selected[0], "packet_kernels": selected[1],
                "pressure": pressure(), "loadavg": loadavg(),
            }));
        }
        set_expert_packet_max_rows(GROUPED_MAX_ROWS);
        Ok(())
    }

    pub fn run() -> anyhow::Result<()> {
        assert!(
            std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
            "canonical lease required"
        );
        let args: Vec<String> = std::env::args().collect();
        let corpus = args.get(1).ok_or_else(|| {
            anyhow::anyhow!(
                "usage: glm_small_append_price <corpus> [all|census|batch|threshold] [depth]"
            )
        })?;
        let mode = args.get(2).map(String::as_str).unwrap_or("all");
        let depth: usize = args
            .get(3)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(48 * 1024);
        anyhow::ensure!(
            matches!(mode, "all" | "census" | "batch" | "threshold"),
            "unknown mode {mode}"
        );

        let ctx = MetalContext::new()?;
        let path = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
            .path_or_skip()
            .ok_or_else(|| anyhow::anyhow!("GLM fixture unavailable"))?;
        let gguf = GgufFile::open(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let text = std::fs::read_to_string(corpus)?
            .replace("<|", "‹|")
            .replace("<｜", "‹｜");
        let ids: Vec<u32> = tokenizer
            .encode(&text, false)?
            .into_iter()
            .map(|t| t as u32)
            .collect();
        anyhow::ensure!(
            ids.len() >= depth + 256,
            "corpus has {} tokens, need {}",
            ids.len(),
            depth + 256
        );
        let weights = Glm5NextWeights::load(&ctx, &gguf)?;
        emit(json!({
            "kind": "header", "mode": mode, "depth": depth,
            "capacity": CAPACITY, "prefill_rows": PREFILL_ROWS,
            "corpus_tokens": ids.len(), "device": ctx.device_facts(),
            "pressure": pressure(), "loadavg": loadavg(),
        }));

        if mode == "all" || mode == "census" {
            census(&ctx, &weights, &ids, depth)?;
        }
        if mode == "all" || mode == "batch" {
            batch(&ctx, &weights, &ids, depth)?;
        }
        if mode == "threshold" {
            threshold(&ctx, &weights, &ids, depth)?;
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
