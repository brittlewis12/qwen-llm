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
//!  4. `routes`: at a chosen depth, force every legal routed-expert route
//!     (incumbent, per-slot native, grouped generic, expanded tensor, tensor
//!     IQ) at the real append-width grid, with interleaved warm repetitions,
//!     ordinary wall, the routed-expert stage, kernel names and logit deltas.
//!
//! Output is JSON lines. No production scheduling change.
//!
//!  5. `occupancy`: capture real per-expert row buckets and per-kernel bytes
//!     at one or more context depths, using the production route selection.
//!
//! Usage: `glm_small_append_price <corpus> [all|census|batch|threshold|routes|route-quality|occupancy] [depth]`

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}

#[cfg(feature = "perf-ablations")]
mod probe {
    use objc2_metal::MTLBuffer;
    use qwen_llm::{
        gguf::GgufFile,
        glm5_next::{FfnTensors, MixerTensors},
        glm5_next_metal::{
            GLM5_NEXT_MAX_BATCH_SIZE, Glm5NextBatch, Glm5NextCapturedRows, Glm5NextExpertRoute,
            Glm5NextRowBank, Glm5NextRowBankPlan, Glm5NextSession, Glm5NextSnapshot,
            Glm5NextStageReport, Glm5NextWeights, set_expert_packet_max_rows, set_expert_route,
        },
        metal::*,
        tokenizer::Tokenizer,
    };
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, io::Write, time::Instant};

    const CAPACITY: usize = 65536;
    /// Deep appends need room above the 48K/80K roots.
    const ROUTE_CAPACITY: usize = 98304;
    const PREFILL_ROWS: usize = 2048;
    const CENSUS_WIDTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 48, 64, 128];
    const THRESHOLD_WIDTHS: [usize; 8] = [8, 12, 16, 24, 32, 48, 64, 96];
    const ROUTE_WIDTHS: [usize; 17] = [
        8, 16, 24, 32, 48, 64, 96, 112, 128, 161, 192, 256, 384, 512, 768, 1023, 1024,
    ];
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
        let level = pressure();
        assert!(
            (0..=2).contains(&level),
            "thermal protocol abort at {level}"
        );
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
        capacity: usize,
    ) -> anyhow::Result<Glm5NextSession<'a>> {
        let mut session = Glm5NextSession::with_prefill_rows(ctx, weights, capacity, PREFILL_ROWS)?;
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
        let mut session = new_session(ctx, weights, ids, depth, CAPACITY)?;
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
        let seed = new_session(ctx, weights, ids, depth, CAPACITY)?;
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
        let mut session = new_session(ctx, weights, ids, depth, CAPACITY)?;
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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RouteArm {
        Incumbent,
        PerSlotNative,
        GroupedGeneric,
        ExpandedTensor,
        TensorIqExpanded,
    }

    impl RouteArm {
        fn name(self) -> &'static str {
            match self {
                Self::Incumbent => "incumbent",
                Self::PerSlotNative => "per_slot_native",
                Self::GroupedGeneric => "grouped_generic",
                Self::ExpandedTensor => "expanded_tensor",
                Self::TensorIqExpanded => "tensor_iq",
            }
        }

        fn apply(self) {
            set_expert_route(match self {
                Self::Incumbent => None,
                Self::PerSlotNative => Some(Glm5NextExpertRoute::PerSlotNative),
                Self::GroupedGeneric => Some(Glm5NextExpertRoute::GroupedGeneric),
                Self::ExpandedTensor => Some(Glm5NextExpertRoute::ExpandedTensor),
                Self::TensorIqExpanded => Some(Glm5NextExpertRoute::TensorIqExpanded),
            });
        }
    }

    fn route_widths() -> Vec<usize> {
        std::env::var("GLM_ROUTE_WIDTHS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|item| item.trim().parse().ok())
                    .filter(|width| *width > 0)
                    .collect::<Vec<_>>()
            })
            .filter(|widths| !widths.is_empty())
            .unwrap_or_else(|| ROUTE_WIDTHS.to_vec())
    }

    fn occupancy_cells(depths: &[usize], widths: &[usize]) -> anyhow::Result<Vec<(usize, usize)>> {
        let mut cells = if let Ok(value) = std::env::var("GLM_ROUTE_CELLS") {
            value
                .split(',')
                .map(|item| {
                    let (depth, width) = item
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("invalid route cell {item:?}"))?;
                    Ok((depth.parse()?, width.parse()?))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            depths
                .iter()
                .flat_map(|depth| widths.iter().map(|width| (*depth, *width)))
                .collect()
        };
        for (depth, width) in &cells {
            anyhow::ensure!(
                depths.contains(depth),
                "route cell has uncaptured depth {depth}"
            );
            anyhow::ensure!(
                widths.contains(width),
                "route cell has unconfigured width {width}"
            );
        }
        cells.sort_unstable();
        cells.dedup();
        anyhow::ensure!(!cells.is_empty(), "occupancy needs at least one route cell");
        Ok(cells)
    }

    /// Every route the shipped kernels admit at `width`. The per-slot packet
    /// validates 1..=128 rows; the others span the whole grid.
    fn route_arms(width: usize) -> Vec<RouteArm> {
        let mut arms = vec![RouteArm::Incumbent];
        if width <= PACKET_MAX_ROWS {
            arms.push(RouteArm::PerSlotNative);
        }
        arms.extend([
            RouteArm::GroupedGeneric,
            RouteArm::ExpandedTensor,
            RouteArm::TensorIqExpanded,
        ]);
        arms
    }

    fn distribution(counts: &[usize]) -> Value {
        let mut sorted: Vec<usize> = counts.iter().copied().filter(|&n| n > 0).collect();
        sorted.sort_unstable();
        let mut histogram = BTreeMap::<usize, usize>::new();
        for &count in counts {
            *histogram.entry(count).or_default() += 1;
        }
        let quantile = |p: f64| -> usize {
            if sorted.is_empty() {
                0
            } else {
                sorted[((sorted.len() - 1) as f64 * p).round() as usize]
            }
        };
        json!({
            "expert_bins": counts.len(),
            "active_experts": sorted.len(),
            "empty_experts": counts.len() - sorted.len(),
            "routed_rows": counts.iter().sum::<usize>(),
            "active_rows_per_expert": {
                "min": sorted.first().copied().unwrap_or(0),
                "p50": quantile(0.50), "p90": quantile(0.90),
                "p99": quantile(0.99), "max": sorted.last().copied().unwrap_or(0),
            },
            "histogram_rows_per_expert": histogram,
        })
    }

    fn route_weight_bytes(weights: &Glm5NextWeights, block: usize) -> anyhow::Result<(u64, u64)> {
        let experts = u64::from(weights.config.expert_count);
        let FfnTensors::Moe(moe) = &weights.blocks[block].ffn else {
            anyhow::bail!("route capture block {block} is not MoE")
        };
        let gate_up_bytes = moe.gate_experts.n_bytes() + moe.up_experts.n_bytes();
        let down_bytes = moe.down_experts.n_bytes();
        anyhow::ensure!(
            gate_up_bytes.is_multiple_of(experts) && down_bytes.is_multiple_of(experts),
            "expert weight banks are not divisible by expert count in block {block}"
        );
        Ok((gate_up_bytes / experts, down_bytes / experts))
    }

    #[derive(Clone, Copy, Default)]
    struct StageCost {
        flops: f64,
        logical_bytes: u64,
        q6_bytes: u64,
        q8_bytes: u64,
    }

    fn add_matrix_cost(
        costs: &mut BTreeMap<&'static str, StageCost>,
        stage: &'static str,
        tensor: &qwen_llm::metal::MetalTensor,
        rows: u64,
    ) {
        let cost = costs.entry(stage).or_default();
        cost.logical_bytes += tensor.n_bytes();
        if tensor.shape.len() >= 2 {
            cost.flops += 2.0 * tensor.n_elements() as f64 * rows as f64;
        }
        let dtype = format!("{:?}", tensor.dtype);
        if dtype.contains("Q6_K") {
            cost.q6_bytes += tensor.n_bytes();
        } else if dtype.contains("Q8_0") {
            cost.q8_bytes += tensor.n_bytes();
        }
    }

    fn add_dense_cost(
        costs: &mut BTreeMap<&'static str, StageCost>,
        stage: &'static str,
        dense: &qwen_llm::glm5_next::DenseFfnTensors<qwen_llm::metal::MetalTensor>,
        rows: u64,
    ) {
        for tensor in [&dense.gate, &dense.up, &dense.down] {
            add_matrix_cost(costs, stage, tensor, rows);
        }
    }

    fn stage_cost_model(
        weights: &Glm5NextWeights,
        rows: usize,
        depth: usize,
        routed_bytes: u64,
    ) -> BTreeMap<&'static str, StageCost> {
        let c = &weights.config;
        let (h, rows) = (u64::from(c.hidden_size), rows as u64);
        let mut costs = BTreeMap::<&'static str, StageCost>::new();
        for block in &weights.blocks {
            for hc in [&block.attention_hc, &block.ffn_hc] {
                let cost = costs.entry("mhc").or_default();
                for tensor in [&hc.mix, &hc.base, &hc.scale] {
                    cost.logical_bytes += tensor.n_bytes();
                }
            }
            for norm in [&block.attention_norm, &block.ffn_norm] {
                costs.entry("mhc").or_default().logical_bytes += norm.n_bytes();
            }

            match &block.mixer {
                MixerTensors::Kda(kda) => {
                    for tensor in [
                        &kda.query,
                        &kda.key,
                        &kda.value,
                        &kda.query_conv,
                        &kda.key_conv,
                        &kda.value_conv,
                        &kda.decay_a,
                        &kda.decay_b,
                        &kda.decay_bias,
                        &kda.neg_exp_a_log,
                        &kda.beta,
                        &kda.gate_a,
                        &kda.gate_b,
                        &kda.output_norm,
                        &kda.output,
                    ] {
                        add_matrix_cost(&mut costs, "kda", tensor, rows);
                    }
                    let heads = h / u64::from(c.kda_head_dim.max(1));
                    let dim = u64::from(c.kda_head_dim);
                    // The recurrent estimate includes state decay, rank-one
                    // update, and state-vector output for each row.
                    costs.entry("kda").or_default().flops +=
                        5.0 * rows as f64 * (heads * dim * dim) as f64;
                }
                MixerTensors::Mla(mla) => {
                    for tensor in [&mla.query_a, &mla.query_b, &mla.latent, &mla.key_absorb] {
                        add_matrix_cost(&mut costs, "mla_projection", tensor, rows);
                    }
                    for tensor in [&mla.value_expand, &mla.output] {
                        add_matrix_cost(&mut costs, "mla_output", tensor, rows);
                    }
                    for tensor in [&mla.indexer.key, &mla.indexer.pool_gate] {
                        add_matrix_cost(&mut costs, "mla_indexer", tensor, rows);
                    }
                    for tensor in [&mla.indexer.query, &mla.indexer.head_weights] {
                        add_matrix_cost(&mut costs, "sparse_query", tensor, rows);
                    }
                }
            }

            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    add_dense_cost(&mut costs, "dense_ffn", dense, rows);
                }
                FfnTensors::Moe(moe) => {
                    add_matrix_cost(&mut costs, "router", &moe.router, rows);
                    add_dense_cost(&mut costs, "shared_expert", &moe.shared, rows);
                }
            }
        }

        let moe_blocks = weights
            .blocks
            .iter()
            .filter(|block| matches!(&block.ffn, FfnTensors::Moe(_)))
            .count() as u64;
        let route_slots = rows * u64::from(c.expert_used_count) * moe_blocks;
        let routed = costs.entry("routed_experts").or_default();
        routed.flops = 6.0 * h as f64 * f64::from(c.expert_ffn_size) * route_slots as f64;
        routed.logical_bytes = routed_bytes;

        let context = (depth as u64 + rows / 2).min(u64::from(c.context_length));
        let pools = context.div_ceil(u64::from(c.indexer_pool.max(1)));
        let score_flops = 2.0
            * rows as f64
            * pools as f64
            * f64::from(c.indexer_head_count)
            * f64::from(c.indexer_head_dim);
        let score = costs.entry("sparse_scores").or_default();
        score.flops = score_flops;
        score.logical_bytes = rows * pools * u64::from(c.indexer_head_dim) * 2;

        let select = costs.entry("sparse_select").or_default();
        select.logical_bytes = rows * pools * std::mem::size_of::<f32>() as u64;

        let sparse = costs.entry("sparse_attention").or_default();
        let selected_rows = u64::from(c.indexer_top_k);
        let heads = u64::from(c.head_count);
        let kv = u64::from(c.kv_lora_rank);
        let head_dim = u64::from(c.mla_key_head_dim);
        sparse.flops =
            2.0 * rows as f64 * selected_rows as f64 * heads as f64 * (kv + head_dim) as f64;
        sparse.logical_bytes = rows * selected_rows * (kv + head_dim) * 2;

        let mhc = costs.entry("mhc").or_default();
        mhc.flops = weights.blocks.len() as f64
            * rows as f64
            * h as f64
            * f64::from(c.hc_streams)
            * f64::from(c.hc_streams)
            * 4.0;
        costs
    }

    fn stage_time_ms(stages: &BTreeMap<&'static str, f64>, stage: &str) -> f64 {
        if stage == "mhc" {
            return ["attention_pre", "attention_post", "ffn_pre", "ffn_post"]
                .iter()
                .map(|name| stages.get(name).copied().unwrap_or(0.0))
                .sum();
        }
        if stage == "kda" {
            return stages
                .iter()
                .filter(|(name, _)| **name == "kda" || name.starts_with("packed_kda_"))
                .map(|(_, ms)| *ms)
                .sum();
        }
        stages.get(stage).copied().unwrap_or(0.0)
    }

    fn stage_metrics(
        costs: &BTreeMap<&'static str, StageCost>,
        stages: &BTreeMap<&'static str, f64>,
    ) -> Value {
        let mut metrics = serde_json::Map::new();
        for (name, cost) in costs {
            let ms = stage_time_ms(stages, name);
            let seconds = ms / 1e3;
            let tflops = (seconds > 0.0).then_some(cost.flops / seconds / 1e12);
            let logical_gbs = (seconds > 0.0).then_some(cost.logical_bytes as f64 / seconds / 1e9);
            metrics.insert(
                (*name).into(),
                json!({
                    "stage_ms": ms,
                    "estimated_flops": cost.flops,
                    "logical_weight_or_scan_bytes": cost.logical_bytes,
                    "q6_weight_bytes": cost.q6_bytes,
                    "q8_weight_bytes": cost.q8_bytes,
                    "estimated_tflops": tflops,
                    "logical_gbs": logical_gbs,
                    "q6_logical_gbs": (seconds > 0.0).then_some(cost.q6_bytes as f64 / seconds / 1e9),
                    "q8_logical_gbs": (seconds > 0.0).then_some(cost.q8_bytes as f64 / seconds / 1e9),
                    "tensor_roof_util_pct_of_131_7_tflops": tflops.map(|rate| 100.0 * rate / 131.7),
                }),
            );
        }
        Value::Object(metrics)
    }

    /// Capture the routed union and distribution from ordinary production
    /// selection. Route ids are read from the completed packed call; a second
    /// serialized dispatch census attributes gate/up and down bandwidth.
    fn occupancy(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depths: &[usize],
        capacity: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!depths.is_empty(), "occupancy needs at least one depth");
        let mut sorted_depths = depths.to_vec();
        sorted_depths.sort_unstable();
        sorted_depths.dedup();
        let widths = route_widths();
        let cells = occupancy_cells(&sorted_depths, &widths)?;
        let max_depth = *sorted_depths.last().expect("nonempty depths");
        let max_width = *widths.iter().max().expect("route widths");
        anyhow::ensure!(
            max_depth + max_width < ids.len(),
            "corpus has {} tokens, needs {}",
            ids.len(),
            max_depth + max_width + 1
        );

        let mut session = Glm5NextSession::with_prefill_rows(ctx, weights, capacity, PREFILL_ROWS)?;
        let mut snapshots = Vec::new();
        let mut depth_cursor = 0usize;
        while session.position() < max_depth {
            gate();
            let start = session.position();
            let end = (start + PREFILL_ROWS).min(max_depth);
            session.prefill_packed(ctx, &ids[start..end])?;
            while sorted_depths.get(depth_cursor).copied() == Some(session.position()) {
                snapshots.push((session.position(), session.capture_snapshot()?));
                depth_cursor += 1;
            }
        }
        anyhow::ensure!(
            depth_cursor == sorted_depths.len(),
            "failed to capture all depth snapshots"
        );

        emit(json!({
            "kind": "occupancy_header", "mode": "production_route_occupancy",
            "depths": sorted_depths, "widths": widths, "cells": cells, "capacity": capacity,
            "hidden_size": weights.config.hidden_size,
            "expert_count": weights.config.expert_count,
            "expert_used_count": weights.config.expert_used_count,
            "expert_ffn_size": weights.config.expert_ffn_size,
            "moe_blocks": weights.blocks.iter().filter(|b| matches!(&b.ffn, FfnTensors::Moe(_))).count(),
            "selection": "production route selection, no expert route override",
        }));

        for (depth, snapshot) in &snapshots {
            for width in &widths {
                let width = *width;
                if !cells.contains(&(*depth, width)) {
                    continue;
                }
                let tokens = &ids[*depth..*depth + width];
                gate();
                session.restore_snapshot(snapshot)?;
                let logits = session.prefill_packed(ctx, tokens)?;
                anyhow::ensure!(
                    logits.iter().all(|v| v.is_finite()),
                    "non-finite occupancy warm logits"
                );
                let route_ids = session.perf_packed_route_ids(width)?;

                let mut ordinary_wall_samples = Vec::new();
                for _ in 0..5 {
                    gate();
                    session.restore_snapshot(snapshot)?;
                    let start = Instant::now();
                    let logits = session.prefill_packed(ctx, tokens)?;
                    ordinary_wall_samples.push(start.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(
                        logits.iter().all(|v| v.is_finite()),
                        "non-finite ordinary occupancy logits"
                    );
                }

                let mut stage_profile_wall_samples = Vec::new();
                let mut command_gpu_samples = Vec::new();
                let mut span_sum_samples = Vec::new();
                let mut stage_samples = BTreeMap::<&'static str, Vec<f64>>::new();
                for _ in 0..3 {
                    gate();
                    session.restore_snapshot(snapshot)?;
                    let start = Instant::now();
                    let (logits, report) = session.prefill_chunk_stage_profiled(ctx, tokens)?;
                    stage_profile_wall_samples.push(start.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(
                        logits.iter().all(|v| v.is_finite()),
                        "non-finite stage-profile logits"
                    );
                    command_gpu_samples.push(report.command_gpu_ms);
                    span_sum_samples.push(report.span_sum_ms);
                    for (name, ms) in stage_totals(&report) {
                        stage_samples.entry(name).or_default().push(ms);
                    }
                }
                let stages: BTreeMap<&'static str, f64> = stage_samples
                    .iter()
                    .map(|(name, samples)| (*name, median(samples.clone())))
                    .collect();

                let mut all_counts = Vec::new();
                let mut per_layer = Vec::with_capacity(route_ids.len());
                let (mut gate_up_union_bytes, mut down_union_bytes) = (0u64, 0u64);
                for (block, routed_ids) in route_ids {
                    let mut counts = vec![0usize; weights.config.expert_count as usize];
                    for expert in routed_ids {
                        let expert = usize::try_from(expert)?;
                        anyhow::ensure!(
                            expert < counts.len(),
                            "route id {expert} exceeds expert count"
                        );
                        counts[expert] += 1;
                    }
                    let (gate_up_per_expert, down_per_expert) = route_weight_bytes(weights, block)?;
                    let active = counts.iter().filter(|&&n| n > 0).count() as u64;
                    let gate_up_bytes = active * gate_up_per_expert;
                    let down_bytes = active * down_per_expert;
                    gate_up_union_bytes += gate_up_bytes;
                    down_union_bytes += down_bytes;
                    all_counts.extend_from_slice(&counts);
                    per_layer.push(json!({
                        "block": block, "counts": counts, "distribution": distribution(&counts),
                        "gate_up_bytes_per_expert": gate_up_per_expert,
                        "down_bytes_per_expert": down_per_expert,
                        "gate_up_union_bytes": gate_up_bytes,
                        "down_union_bytes": down_bytes,
                    }));
                }

                // Profile the exact selected kernels for this same input and
                // state; this sum is attribution, not an ordinary wall time.
                gate();
                session.restore_snapshot(snapshot)?;
                let guard = dispatch_profile_begin(ctx, 16384)?;
                let profiled_logits = session.prefill_packed(ctx, tokens)?;
                let report = dispatch_profile_finish(ctx, guard)?;
                anyhow::ensure!(
                    profiled_logits.iter().all(|v| v.is_finite()),
                    "non-finite profiled logits"
                );
                let (kernels, families, dispatches) = kernel_totals(&report);
                let mut family_json = serde_json::Map::new();
                for (name, (calls, ms)) in &families {
                    family_json.insert(
                        (*name).into(),
                        json!({
                            "calls": calls,
                            "serialized_gpu_ms": ms,
                            "serialized_us_per_row": ms * 1e3 / width as f64,
                        }),
                    );
                }
                let mut gate_up_ms = 0.0;
                let mut down_ms = 0.0;
                for row in &report.rows {
                    let ms = row.gpu_ns.unwrap_or(0) as f64 / 1e6;
                    // The gate pass uses a "moe_down" pipeline name. Classify
                    // projection roles rather than assigning it to down.
                    let name = row.kernel.as_str();
                    let is_gate_up = name.contains("gate_up")
                        || name.contains("moe_swiglu")
                        || name.contains("moe_up_")
                        || name.contains("moe_wide_probe")
                        || name.contains("moe_down_iq2_s_")
                        || name.contains("moe_down_iq3_s_f16act");
                    if is_gate_up {
                        gate_up_ms += ms;
                    } else if name.contains("moe_down_")
                        || name.contains("glm_iq3_down")
                        || name.contains("expert_native_down")
                        || name.contains("glm_batch_down")
                    {
                        down_ms += ms;
                    }
                }
                let routed_ms = stages.get("routed_experts").copied();
                let cost_model = stage_cost_model(
                    weights,
                    width,
                    *depth,
                    gate_up_union_bytes + down_union_bytes,
                );
                let ordinary_wall_ms = median(ordinary_wall_samples.clone());
                let stage_profile_wall_ms = median(stage_profile_wall_samples.clone());
                let command_gpu_ms = median(command_gpu_samples.clone());
                let span_sum_ms = median(span_sum_samples.clone());
                emit(json!({
                    "kind": "occupancy", "depth": depth, "width": width,
                    "union_gate_up_bytes": gate_up_union_bytes,
                    "union_down_bytes": down_union_bytes,
                    "union_total_bytes": gate_up_union_bytes + down_union_bytes,
                    "logical_slot_bytes": (width as u64 * u64::from(weights.config.expert_used_count))
                        * per_layer.iter().map(|layer| layer["gate_up_bytes_per_expert"].as_u64().unwrap_or(0)
                            + layer["down_bytes_per_expert"].as_u64().unwrap_or(0)).sum::<u64>(),
                    "rows_per_expert_distribution": distribution(&all_counts),
                    "per_layer": per_layer,
                    "routed_experts_ms": routed_ms,
                    "ordinary_wall_ms": ordinary_wall_ms,
                    "ordinary_wall_samples_ms": ordinary_wall_samples,
                    "stage_profile_wall_ms": stage_profile_wall_ms,
                    "stage_profile_wall_samples_ms": stage_profile_wall_samples,
                    "stage_profile_overhead_ratio": stage_profile_wall_ms / ordinary_wall_ms,
                    "command_gpu_ms": command_gpu_ms,
                    "command_gpu_ms_samples": command_gpu_samples,
                    "stage_profile_wall_minus_command_gpu_ms": stage_profile_wall_ms - command_gpu_ms,
                    "stage_span_sum_ms": span_sum_ms,
                    "stage_unassigned_gap_ms": command_gpu_ms - span_sum_ms,
                    "ordinary_wall_minus_profiled_gpu_ms": ordinary_wall_ms - command_gpu_ms,
                    "stages_ms": stages,
                    "stage_samples_ms": stage_samples,
                    "stage_metrics": stage_metrics(&cost_model, &stages),
                    "stage_cost_semantics": "estimated multiply-add FLOPs from logical weight shapes; logical unique-weight and modeled cache-scan bytes, not physical bus counters",
                    "gate_up_kernel_ms": gate_up_ms,
                    "down_kernel_ms": down_ms,
                    "serialized_dispatch_total_ms": families.values().map(|(_, ms)| *ms).sum::<f64>(),
                    "dispatches": dispatches, "kernels": kernels,
                    "serialized_dispatch_families": family_json,
                    "pressure": pressure(), "loadavg": loadavg(),
                }));
            }
        }
        Ok(())
    }

    /// Force each legal routed-expert route at every width, with interleaved
    /// warm repetitions. The ordinary wall is the timing number; the stage
    /// profile and kernel census attribute the routed-expert time.
    fn routes(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depth: usize,
        capacity: usize,
    ) -> anyhow::Result<()> {
        let mut session = new_session(ctx, weights, ids, depth, capacity)?;
        let snapshot = session.capture_snapshot()?;
        let widths = route_widths();
        for (width_index, width) in widths.into_iter().enumerate() {
            let tokens = &ids[depth..depth + width];
            let arms = route_arms(width);

            // Warm each route once so the first timed call is not first-touch.
            for &arm in &arms {
                arm.apply();
                gate();
                session.restore_snapshot(&snapshot)?;
                let logits = session.prefill_packed(ctx, tokens)?;
                anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite warm");
            }

            let mut walls: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
            let mut last_logits: Vec<Option<Vec<f32>>> = arms.iter().map(|_| None).collect();
            let mut last_states: Vec<Option<Glm5NextSnapshot>> =
                arms.iter().map(|_| None).collect();
            for rep in 0..3 {
                let n = arms.len();
                let shift = (rep + width_index) % n;
                let order: Vec<usize> = (0..n).map(|i| (i + shift) % n).collect();
                for &index in &order {
                    arms[index].apply();
                    gate();
                    session.restore_snapshot(&snapshot)?;
                    let t = Instant::now();
                    let out = session.prefill_packed(ctx, tokens)?;
                    walls[index].push(t.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(out.iter().all(|v| v.is_finite()), "non-finite logits");
                    if rep == 2 {
                        last_logits[index] = Some(out);
                        last_states[index] = Some(session.capture_snapshot()?);
                    }
                }
            }

            let incumbent = last_logits[0].clone().expect("incumbent logits");
            let incumbent_state = last_states[0].as_ref().expect("incumbent state");
            let next = ids[depth + width];

            // Three interleaved stage-profile repetitions per arm.
            let mut staged_walls: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
            let mut routed_samples: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
            let mut last_stages: Vec<Option<BTreeMap<&'static str, f64>>> =
                arms.iter().map(|_| None).collect();
            let mut last_staged_gpu: Vec<Option<f64>> = arms.iter().map(|_| None).collect();
            for rep in 0..3 {
                let n = arms.len();
                let shift = (rep + width_index) % n;
                let order: Vec<usize> = (0..n).map(|i| (i + shift) % n).collect();
                for &index in &order {
                    arms[index].apply();
                    gate();
                    session.restore_snapshot(&snapshot)?;
                    let t = Instant::now();
                    let (_out, staged) = session.prefill_chunk_stage_profiled(ctx, tokens)?;
                    staged_walls[index].push(t.elapsed().as_secs_f64() * 1e3);
                    let stages = stage_totals(&staged);
                    routed_samples[index]
                        .push(stages.get("routed_experts").copied().unwrap_or(0.0));
                    if rep == 2 {
                        last_stages[index] = Some(stages);
                        last_staged_gpu[index] = Some(staged.command_gpu_ms);
                    }
                }
            }

            for (index, &arm) in arms.iter().enumerate() {
                let stages = last_stages[index].clone().expect("stage totals");
                let routed_ms = median(routed_samples[index].clone());
                let staged_wall = median(staged_walls[index].clone());

                // Name the selected expert kernels at representative widths.
                let mut kernels = Vec::new();
                if matches!(width, 32 | 512) {
                    gate();
                    session.restore_snapshot(&snapshot)?;
                    let guard = dispatch_profile_begin(ctx, 16384)?;
                    let out = session.prefill_packed(ctx, tokens)?;
                    let report = dispatch_profile_finish(ctx, guard)?;
                    anyhow::ensure!(out.iter().all(|v| v.is_finite()), "non-finite profile");
                    let mut by_name = BTreeMap::<String, (usize, f64)>::new();
                    for row in &report.rows {
                        if row.kernel.contains("expert") || row.kernel.contains("moe_") {
                            let e = by_name.entry(row.kernel.clone()).or_default();
                            e.0 += 1;
                            e.1 += row.gpu_ns.unwrap_or(0) as f64 / 1e6;
                        }
                    }
                    kernels = by_name
                        .into_iter()
                        .map(|(k, (c, ms))| json!({"kernel": k, "calls": c, "gpu_ms": ms}))
                        .collect();
                }

                let logits = last_logits[index].as_ref().expect("route logits");
                let max_abs = qwen_llm::compare::report_max_abs_diff_f32(
                    incumbent.iter().copied(),
                    logits.iter().copied(),
                );
                emit(json!({
                    "kind": "route", "depth": depth, "capacity": capacity,
                    "width": width, "arm": arm.name(),
                    "wall_ms": median(walls[index].clone()), "walls_ms": walls[index],
                    "staged_wall_ms": staged_wall,
                    "staged_walls_ms": staged_walls[index],
                    "staged_gpu_ms": last_staged_gpu[index].unwrap_or(0.0),
                    "routed_experts_ms": routed_ms,
                    "routed_experts_ms_samples": routed_samples[index],
                    "stages": stages,
                    "max_abs_logit_diff": max_abs,
                    "state_bitwise": last_states[index]
                        .as_ref()
                        .is_some_and(|state| state.same_state(incumbent_state)),
                    "top1_incumbent": top1(&incumbent), "top1_arm": top1(logits),
                    "top1_agree": top1(&incumbent) == top1(logits),
                    "kl_incumbent_arm": kl(&incumbent, logits),
                    "dnll_signed_arm": log_prob(&incumbent, next) - log_prob(logits, next),
                    "kernels": kernels,
                    "pressure": pressure(), "loadavg": loadavg(),
                }));
            }
        }
        set_expert_route(None);
        Ok(())
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

    fn route_quality_widths() -> Vec<usize> {
        let default = [16usize, 24, 32, 48, 64, 96, 112, 128, 161, 192, 256];
        std::env::var("GLM_ROUTE_QUALITY_WIDTHS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|item| item.trim().parse().ok())
                    .filter(|width| *width > 0)
                    .collect::<Vec<_>>()
            })
            .filter(|widths| !widths.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }

    /// Paired per-position dNLL of each forced route against the incumbent at
    /// the widths whose policy changes. One prompt per invocation; the cohort
    /// analysis bootstraps over prompt files.
    fn route_quality(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        ids: &[u32],
        depth: usize,
        capacity: usize,
    ) -> anyhow::Result<()> {
        let widths = route_quality_widths();
        let max_width = *widths.iter().max().expect("quality widths");
        anyhow::ensure!(
            ids.len() >= depth + max_width + 1,
            "prompt has {} tokens, need {}",
            ids.len(),
            depth + max_width + 1
        );
        let mut session = new_session(ctx, weights, ids, depth, capacity)?;
        let snapshot = session.capture_snapshot()?;
        let plan = Glm5NextRowBankPlan::new(&weights.config, &[], 0, max_width)?;
        let mut bank = Glm5NextRowBank::new(ctx, weights, plan)?;
        let vocab = weights.config.vocab_size as usize;
        let routes = [
            RouteArm::PerSlotNative,
            RouteArm::GroupedGeneric,
            RouteArm::ExpandedTensor,
            RouteArm::TensorIqExpanded,
        ];

        for width in widths {
            let append = &ids[depth..depth + width];
            let actual: Vec<u32> = ids[depth + 1..depth + width + 1].to_vec();

            let capture = |session: &mut Glm5NextSession,
                           bank: &mut Glm5NextRowBank|
             -> anyhow::Result<Vec<Vec<f32>>> {
                let captured: Glm5NextCapturedRows<'_> =
                    session.verify_packed_with_captures(ctx, append, bank)?;
                let flat = read_f32(captured.logits().expect("all-row logits"));
                anyhow::ensure!(
                    flat.len() == width * vocab,
                    "logit view {} != {}",
                    flat.len(),
                    width * vocab
                );
                Ok((0..width)
                    .map(|row| flat[row * vocab..(row + 1) * vocab].to_vec())
                    .collect())
            };

            set_expert_route(None);
            gate();
            session.restore_snapshot(&snapshot)?;
            let incumbent = capture(&mut session, &mut bank)?;

            for &route in &routes {
                if route == RouteArm::PerSlotNative && width > PACKET_MAX_ROWS {
                    continue;
                }
                route.apply();
                gate();
                session.restore_snapshot(&snapshot)?;
                let candidate = capture(&mut session, &mut bank)?;
                let mut dnll = Vec::with_capacity(width);
                let mut kl_both = Vec::with_capacity(width);
                let mut top1_agree = 0usize;
                for row in 0..width {
                    let (i, c) = (&incumbent[row], &candidate[row]);
                    dnll.push(log_prob(i, actual[row]) - log_prob(c, actual[row]));
                    kl_both.push(kl(i, c));
                    if top1(i) == top1(c) {
                        top1_agree += 1;
                    }
                }
                let mean_dnll = dnll.iter().sum::<f64>() / width as f64;
                emit(json!({
                    "kind": "route_quality", "depth": depth, "capacity": capacity,
                    "width": width, "arm": route.name(),
                    "positions": width,
                    "mean_dnll_vs_incumbent": mean_dnll,
                    "max_abs_dnll": dnll.iter().map(|v| v.abs()).fold(0.0, f64::max),
                    "mean_kl_incumbent_arm": kl_both.iter().sum::<f64>() / width as f64,
                    "top1_agree": top1_agree,
                    "dnll": dnll,
                    "pressure": pressure(), "loadavg": loadavg(),
                }));
            }
        }
        set_expert_route(None);
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
                "usage: glm_small_append_price <corpus> [all|census|batch|threshold|routes|route-quality] [depth]"
            )
        })?;
        let mode = args.get(2).map(String::as_str).unwrap_or("all");
        let depth: usize = args
            .get(3)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(48 * 1024);
        let occupancy_depths = if mode == "occupancy" {
            std::env::var("GLM_ROUTE_DEPTHS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|item| item.trim().parse().ok())
                        .collect()
                })
                .unwrap_or_else(|| vec![depth])
        } else {
            Vec::new()
        };
        anyhow::ensure!(
            matches!(
                mode,
                "all" | "census" | "batch" | "threshold" | "routes" | "route-quality" | "occupancy"
            ),
            "unknown mode {mode}"
        );
        let quality_widths = if mode == "route-quality" {
            route_quality_widths()
        } else {
            Vec::new()
        };
        let capacity = if matches!(mode, "routes" | "route-quality" | "occupancy") {
            std::env::var("GLM_APPEND_CAPACITY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(ROUTE_CAPACITY)
        } else {
            CAPACITY
        };
        let needed_width = if mode == "routes" || mode == "occupancy" {
            *route_widths().iter().max().expect("route widths")
        } else if mode == "route-quality" {
            *quality_widths.iter().max().expect("quality widths")
        } else {
            256
        };
        let max_depth = if mode == "occupancy" {
            occupancy_depths.iter().copied().max().unwrap_or(depth)
        } else {
            depth
        };
        anyhow::ensure!(
            capacity >= max_depth + needed_width + 1,
            "capacity {capacity} cannot hold depth {max_depth} plus {needed_width} rows"
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
            ids.len() >= max_depth + needed_width + 1,
            "corpus has {} tokens, need {}",
            ids.len(),
            max_depth + needed_width + 1
        );
        let weights = Glm5NextWeights::load(&ctx, &gguf)?;
        emit(json!({
            "kind": "header", "mode": mode, "depth": depth,
            "occupancy_depths": if mode == "occupancy" { Some(&occupancy_depths) } else { None },
            "capacity": capacity, "prefill_rows": PREFILL_ROWS,
            "route_widths": if matches!(mode, "routes" | "occupancy") { Some(route_widths()) } else { None },
            "quality_widths": if mode == "route-quality" { Some(quality_widths) } else { None },
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
        if mode == "routes" {
            routes(&ctx, &weights, &ids, depth, capacity)?;
        }
        if mode == "route-quality" {
            route_quality(&ctx, &weights, &ids, depth, capacity)?;
        }
        if mode == "occupancy" {
            occupancy(&ctx, &weights, &ids, &occupancy_depths, capacity)?;
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
