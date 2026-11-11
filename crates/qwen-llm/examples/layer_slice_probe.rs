//! Layer-boundary append probe, not a serving policy. Requires perf-ablations.
//! layer_slice_probe <corpus> [reps=8] [decoders=3] [slices=1,2,4,8] [runtime] [idle_ms=0]
//! Paired orders alternate. Teacher-forced decoder inputs keep GPU work equal.
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::{
        Glm5NextBatch, Glm5NextMetalError, Glm5NextSession, Glm5NextSnapshot, Glm5NextWeights,
        take_last_step_timing,
    },
    metal::MetalContext,
    tokenizer::Tokenizer,
};
use std::{io::Write, time::Instant};

const DEPTH: usize = 48 * 1024;
const ROWS: usize = 2048;
const CAPACITY: usize = 65536;

fn emit(value: serde_json::Value) {
    println!("{value}");
    std::io::stdout().flush().unwrap();
}
fn pressure() -> i32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|x| String::from_utf8(x.stdout).ok())
        .and_then(|x| x.split_whitespace().last()?.parse().ok())
        .unwrap_or(-1)
}
fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
}
fn p99(xs: &[f64]) -> f64 {
    let mut xs = xs.to_vec();
    xs.sort_by(f64::total_cmp);
    xs[(xs.len() * 99).div_ceil(100) - 1]
}

struct Rounds<'a, 'w> {
    ctx: &'a MetalContext,
    batch: &'a mut Glm5NextBatch,
    members: &'a mut [Glm5NextSession<'w>],
    ids: &'a [u32],
    count: usize,
    last: Instant,
    gaps: Vec<f64>,
    gpu: Vec<f64>,
    logits: Vec<Vec<Vec<f32>>>,
}
impl Rounds<'_, '_> {
    fn step(&mut self) -> Result<(), Glm5NextMetalError> {
        let tokens: Vec<_> = (0..self.members.len())
            .map(|i| self.ids[DEPTH + i * 64 + self.count])
            .collect();
        let mut refs: Vec<_> = self.members.iter_mut().collect();
        let logits = self.batch.step(self.ctx, &mut refs, &tokens)?;
        self.gaps.push(ms(self.last));
        self.last = Instant::now();
        self.gpu
            .push(take_last_step_timing().expect("batch GPU timing").1);
        self.logits.push(logits);
        self.count += 1;
        Ok(())
    }
}
struct Arm {
    ttft: f64,
    work_wall: f64,
    append_gpu: f64,
    rounds_gpu: f64,
    commands: Vec<(usize, f64)>,
    gaps: Vec<f64>,
    logits: Vec<f32>,
    state: Glm5NextSnapshot,
    next_logits: Vec<f32>,
    decoder_logits: Vec<Vec<Vec<f32>>>,
    decoder_states: Vec<Glm5NextSnapshot>,
}
#[allow(clippy::too_many_arguments)]
fn arm(
    ctx: &MetalContext,
    app: &mut Glm5NextSession<'_>,
    members: &mut [Glm5NextSession<'_>],
    batch: &mut Glm5NextBatch,
    root: &Glm5NextSnapshot,
    ids: &[u32],
    n: usize,
    ends: &[usize],
    interleave: bool,
) -> anyhow::Result<Arm> {
    app.restore_snapshot(root)?;
    for member in members.iter_mut() {
        member.restore_snapshot(root)?;
    }
    let mut rounds = Rounds {
        ctx,
        batch,
        members,
        ids,
        count: 0,
        last: Instant::now(),
        gaps: Vec::new(),
        gpu: Vec::new(),
        logits: Vec::new(),
    };
    // Complete one real round before time zero. Its completion is the left
    // endpoint of the first measured gap; no restore/capture is in the gaps.
    rounds.step()?;
    rounds.gaps.clear();
    rounds.gpu.clear();
    let start = Instant::now();
    let unsplit = [*ends.last().unwrap()];
    let (logits, commands) = app.prefill_layer_sliced_probe(
        ctx,
        &ids[DEPTH..DEPTH + n],
        if interleave { ends } else { &unsplit },
        &mut |_| rounds.step(),
    )?;
    let ttft = ms(start);
    if !interleave {
        for _ in 1..ends.len() {
            rounds.step()?;
        }
    }
    let work_wall = ms(start);
    let rounds_gpu = rounds.gpu.iter().sum();
    // Include the last slice in the gap measurement. This common tail round
    // is excluded from work GPU/wall and appender TTFT in BOTH arms.
    rounds.step()?;
    let gaps = rounds.gaps;
    let decoder_logits = rounds.logits;
    let decoder_states = members
        .iter()
        .map(|s| s.capture_snapshot())
        .collect::<Result<Vec<_>, _>>()?;
    let state = app.capture_snapshot()?;
    let next_logits = app.forward(ctx, ids[DEPTH + n])?;
    Ok(Arm {
        ttft,
        work_wall,
        append_gpu: commands.iter().map(|x| x.1).sum(),
        rounds_gpu,
        commands,
        gaps,
        logits,
        state,
        next_logits,
        decoder_logits,
        decoder_states,
    })
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let corpus = args.get(1).ok_or_else(|| {
        anyhow::anyhow!("usage: layer_slice_probe <corpus> [reps=8] [decoders=3] [slices=1,2,4,8]")
    })?;
    let reps: usize = args.get(2).map_or(Ok(8), |x| x.parse())?;
    let width: usize = args.get(3).map_or(Ok(3), |x| x.parse())?;
    let slices: Vec<usize> = args
        .get(4)
        .map_or("1,2,4,8", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(reps > 0 && (2..=3).contains(&width), "reps/width");
    let ctx = MetalContext::new()?;
    let path = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
        .path_or_skip()
        .ok_or_else(|| anyhow::anyhow!("resolve GLM via QWEN_MODEL_SEARCH_PATH"))?;
    let gguf = GgufFile::open(path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let text = std::fs::read_to_string(corpus)?
        .replace("<|", "‹|")
        .replace("<｜", "‹｜");
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    anyhow::ensure!(ids.len() > DEPTH + ROWS, "short corpus");
    let weights = Glm5NextWeights::load(&ctx, &gguf)?;
    let layers = weights.blocks.len();
    anyhow::ensure!(slices.iter().all(|s| *s > 0 && *s <= layers), "slice count");
    emit(
        serde_json::json!({"kind":"header", "depth":DEPTH,"reps":reps,
        "decoders":width,"layers":layers,"hidden":weights.config.hidden_size,
        "slices":slices,"pressure":pressure(),
        "live_residual_bytes_per_row": weights.config.hidden_size as u64 * 4 * 4,
        "parked_extra_allocation_bytes":0}),
    );
    let mut app = Glm5NextSession::with_prefill_rows(&ctx, &weights, CAPACITY, ROWS)?;
    for part in ids[..DEPTH].chunks(ROWS) {
        app.prefill_packed(&ctx, part)?;
    }
    let root = app.capture_snapshot()?;
    let mut members = (0..width)
        .map(|_| Glm5NextSession::with_prefill_rows(&ctx, &weights, CAPACITY, 4))
        .collect::<Result<Vec<_>, _>>()?;
    let mut batch = Glm5NextBatch::new(&ctx, &weights, width)?;
    // Warm the exact decoder geometry and compile it outside measurements.
    for member in &mut members {
        member.restore_snapshot(&root)?;
    }
    batch.step(
        &ctx,
        &mut members.iter_mut().collect::<Vec<_>>(),
        &vec![ids[DEPTH]; width],
    )?;
    if args.get(5).map(String::as_str) == Some("runtime") {
        let idle_ms: u64 = args.get(6).map_or(Ok(0), |s| s.parse())?;
        return runtime_gate(
            &ctx,
            &weights,
            &ids,
            &mut app,
            &mut members,
            &mut batch,
            &root,
            reps,
            idle_ms,
        );
    }
    // Independent ordinary one-command reference for the instrumented s=1.
    for n in [32, 342, 1509] {
        app.restore_snapshot(&root)?;
        let reference = app.prefill_packed(&ctx, &ids[DEPTH..DEPTH + n])?;
        let reference_state = app.capture_snapshot()?;
        let reference_next = app.forward(&ctx, ids[DEPTH + n])?;
        for &s in &slices {
            let ends: Vec<_> = (1..=s).map(|i| i * layers / s).collect();
            let mut all_gaps = Vec::new();
            let mut total_gpu_ratios = Vec::new();
            for rep in 0..reps {
                let thermal = pressure();
                let (base, split) = if rep % 2 == 0 {
                    let base = arm(
                        &ctx,
                        &mut app,
                        &mut members,
                        &mut batch,
                        &root,
                        &ids,
                        n,
                        &ends,
                        false,
                    )?;
                    let split = arm(
                        &ctx,
                        &mut app,
                        &mut members,
                        &mut batch,
                        &root,
                        &ids,
                        n,
                        &ends,
                        true,
                    )?;
                    (base, split)
                } else {
                    let split = arm(
                        &ctx,
                        &mut app,
                        &mut members,
                        &mut batch,
                        &root,
                        &ids,
                        n,
                        &ends,
                        true,
                    )?;
                    let base = arm(
                        &ctx,
                        &mut app,
                        &mut members,
                        &mut batch,
                        &root,
                        &ids,
                        n,
                        &ends,
                        false,
                    )?;
                    (base, split)
                };
                let bitwise = same(&split.logits, &reference) && same(&base.logits, &reference);
                let state_exact = split.state.same_state(&reference_state)
                    && base.state.same_state(&reference_state);
                let next_exact = same(&split.next_logits, &reference_next)
                    && same(&base.next_logits, &reference_next);
                let decoders_exact = split
                    .decoder_states
                    .iter()
                    .zip(&base.decoder_states)
                    .all(|(a, b)| a.same_state(b))
                    && split
                        .decoder_logits
                        .iter()
                        .zip(&base.decoder_logits)
                        .all(|(a, b)| a.iter().zip(b).all(|(a, b)| same(a, b)));
                let ratio =
                    (split.append_gpu + split.rounds_gpu) / (base.append_gpu + base.rounds_gpu);
                emit(
                    serde_json::json!({"kind":"pair","n":n,"s":s,"rep":rep,"decoders":width,
                    "ends":ends,"base_ttft_ms":base.ttft,"split_ttft_ms":split.ttft,
                    "base_work_wall_ms":base.work_wall,"split_work_wall_ms":split.work_wall,
                    "base_append_gpu_ms":base.append_gpu,"split_append_gpu_ms":split.append_gpu,
                    "base_rounds_gpu_ms":base.rounds_gpu,"split_rounds_gpu_ms":split.rounds_gpu,
                    "total_gpu_ratio":ratio,"commands":split.commands,
                    "base_gaps_ms":base.gaps,"split_gaps_ms":split.gaps,
                    "bitwise_logits":bitwise,"bitwise_state":state_exact,
                    "bitwise_next":next_exact,"bitwise_decoders":decoders_exact,
                    "pressure_start":thermal,"pressure_end":pressure()}),
                );
                anyhow::ensure!(
                    bitwise && state_exact && next_exact && decoders_exact,
                    "exactness gate failed at n={n} s={s} rep={rep}"
                );
                all_gaps.extend(split.gaps);
                total_gpu_ratios.push(ratio);
            }
            emit(
                serde_json::json!({"kind":"summary","n":n,"s":s,"decoders":width,
                "gap_samples":all_gaps.len(),"gap_p99_ms":p99(&all_gaps),
                "gap_max_ms":all_gaps.iter().copied().fold(0.0,f64::max),
                "mean_total_gpu_ratio":total_gpu_ratios.iter().sum::<f64>()/reps as f64,
                "pressure":pressure()}),
            );
        }
    }
    emit(serde_json::json!({"kind":"done","pressure":pressure()}));
    Ok(())
}

/// Regression gate for the actual resumable production API, including fresh
/// prefixes, idle residency, all persistent bytes, exclusivity and disposal.
#[allow(clippy::too_many_arguments)]
fn runtime_gate<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    ids: &[u32],
    app: &mut Glm5NextSession<'w>,
    members: &mut [Glm5NextSession<'w>],
    batch: &mut Glm5NextBatch,
    root: &Glm5NextSnapshot,
    reps: usize,
    idle_ms: u64,
) -> anyhow::Result<()> {
    app.reset()?;
    let empty = app.capture_snapshot()?;
    for (base, snap) in [(DEPTH, root), (0, &empty)] {
        for n in [32, 342, 1509] {
            app.restore_snapshot(snap)?;
            let reference = app.prefill_packed(ctx, &ids[base..base + n])?;
            let state = app.capture_snapshot()?;
            let next = app.forward(ctx, ids[base + n])?;
            for rep in 0..reps {
                app.restore_snapshot(snap)?;
                for member in members.iter_mut() {
                    member.restore_snapshot(snap)?;
                }
                let mut rounds = Rounds {
                    ctx,
                    batch,
                    members,
                    ids,
                    count: 0,
                    last: Instant::now(),
                    gaps: Vec::new(),
                    gpu: Vec::new(),
                    logits: Vec::new(),
                };
                rounds.step()?;
                rounds.gaps.clear();
                rounds.gpu.clear();
                if idle_ms > 0 && base == DEPTH && n == 32 && rep == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(idle_ms));
                    // Idle itself is not an inter-token gap during append.
                    rounds.last = Instant::now();
                }
                let start = Instant::now();
                app.begin_prefill_chunk(ctx, &ids[base..base + n], 100.0)?;
                let mut gpu_ms = 0.0;
                let mut commands = 0;
                let logits = loop {
                    let _qos = qwen_llm::metal::GpuThreadQos::enter();
                    let slice = app.resume_prefill_chunk(ctx, true)?;
                    gpu_ms += slice.gpu_ms;
                    commands += 1;
                    if let Some(logits) = slice.logits {
                        break logits;
                    }
                    anyhow::ensure!(
                        app.is_poisoned() && app.position() == base,
                        "partial position published"
                    );
                    anyhow::ensure!(
                        app.capture_snapshot().is_err()
                            && app.prepare_decode(ctx).is_err()
                            && app.reset().is_err()
                            && app.forward(ctx, ids[base]).is_err(),
                        "partial state escaped"
                    );
                    rounds.step()?;
                };
                let ttft_ms = ms(start);
                rounds.step()?;
                let total_rounds = rounds.count;
                let gaps = rounds.gaps;
                let decoder_logits = rounds.logits;
                let decoder_states = members
                    .iter()
                    .map(|s| s.capture_snapshot())
                    .collect::<Result<Vec<_>, _>>()?;
                let bitwise_logits = same(&logits, &reference);
                let bitwise_state = app.capture_snapshot()?.same_state(&state);
                let bitwise_next = same(&app.forward(ctx, ids[base + n])?, &next);
                for member in members.iter_mut() {
                    member.restore_snapshot(snap)?;
                }
                let mut reference_rounds = Rounds {
                    ctx,
                    batch,
                    members,
                    ids,
                    count: 0,
                    last: Instant::now(),
                    gaps: Vec::new(),
                    gpu: Vec::new(),
                    logits: Vec::new(),
                };
                for _ in 0..total_rounds {
                    reference_rounds.step()?;
                }
                let bitwise_decoders = reference_rounds
                    .logits
                    .iter()
                    .zip(&decoder_logits)
                    .all(|(a, b)| a.iter().zip(b).all(|(a, b)| same(a, b)))
                    && members
                        .iter()
                        .zip(&decoder_states)
                        .all(|(s, b)| s.capture_snapshot().unwrap().same_state(b));
                emit(
                    serde_json::json!({"kind":"runtime_gate","base":base,"n":n,"rep":rep,
                    "idle_ms":if base==DEPTH&&n==32&&rep==0 {idle_ms}else{0},
                    "commands":commands,"gpu_ms":gpu_ms,"ttft_ms":ttft_ms,"gaps_ms":gaps,
                    "bitwise_logits":bitwise_logits,"bitwise_state":bitwise_state,
                    "bitwise_next":bitwise_next,"bitwise_decoders":bitwise_decoders,"pressure":pressure()}),
                );
                anyhow::ensure!(
                    bitwise_logits && bitwise_state && bitwise_next && bitwise_decoders,
                    "runtime exactness failed base={base} n={n} rep={rep}"
                );
            }
        }
    }
    // Cross the dense/sparse frontier inside the original row chunk, with
    // peers at unequal depths. Layer resumes must retain the MLA ordinal and
    // original per-row visibility, including the three dense rows at 2048.
    app.restore_snapshot(&empty)?;
    app.prefill_packed(ctx, &ids[..2048])?;
    let frontier = app.capture_snapshot()?;
    let reference = app.prefill_packed(ctx, &ids[2048..2390])?;
    let state = app.capture_snapshot()?;
    let next = app.forward(ctx, ids[2390])?;
    app.restore_snapshot(&frontier)?;
    members[0].restore_snapshot(root)?;
    for member in &mut members[1..] {
        member.restore_snapshot(&empty)?;
    }
    app.begin_prefill_chunk(ctx, &ids[2048..2390], 100.0)?;
    let logits = loop {
        let slice = app.resume_prefill_chunk(ctx, true)?;
        if let Some(logits) = slice.logits {
            break logits;
        }
        let tokens = vec![ids[DEPTH]; members.len()];
        batch.step(ctx, &mut members.iter_mut().collect::<Vec<_>>(), &tokens)?;
    };
    anyhow::ensure!(
        same(&logits, &reference)
            && app.capture_snapshot()?.same_state(&state)
            && same(&app.forward(ctx, ids[2390])?, &next),
        "dense/sparse frontier changed state"
    );
    // An explicit full snapshot restore may replace a partial cursor. It must
    // clear the cursor as well as restore every recurrent/cache region.
    app.restore_snapshot(&frontier)?;
    app.begin_prefill_chunk(ctx, &ids[2048..2390], 100.0)?;
    app.resume_prefill_chunk(ctx, true)?;
    app.restore_snapshot(&frontier)?;
    anyhow::ensure!(
        !app.prefill_chunk_pending() && !app.is_poisoned(),
        "snapshot kept a stale cursor"
    );
    anyhow::ensure!(
        same(&app.prefill_packed(ctx, &ids[2048..2390])?, &reference),
        "partial restore changed append"
    );
    emit(
        serde_json::json!({"kind":"runtime_frontier","bitwise":true,"partial_snapshot_restore":true}),
    );
    let mut cancelled = Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, ROWS)?;
    cancelled.restore_snapshot(root)?;
    cancelled.begin_prefill_chunk(ctx, &ids[DEPTH..DEPTH + 32], 100.0)?;
    anyhow::ensure!(
        cancelled.resume_prefill_chunk(ctx, true)?.logits.is_none(),
        "expected partial slice"
    );
    anyhow::ensure!(
        cancelled.is_poisoned() && cancelled.capture_snapshot().is_err(),
        "partial cancellation state escaped"
    );
    drop(cancelled); // Every submitted command has completed; no reusable state escapes.
    let mut tiny = Glm5NextSession::with_prefill_rows(ctx, weights, CAPACITY, 1)?;
    tiny.restore_snapshot(root)?;
    members[0].restore_snapshot(root)?;
    anyhow::ensure!(
        batch
            .step(ctx, &mut [&mut tiny, &mut members[0]], &[ids[DEPTH]; 2])
            .is_err(),
        "unequal geometry must decline"
    );
    anyhow::ensure!(
        !tiny.is_poisoned() && !members[0].is_poisoned(),
        "refusal mutated a member"
    );
    tiny.forward(ctx, ids[DEPTH])?;
    members[0].forward(ctx, ids[DEPTH])?;
    emit(
        serde_json::json!({"kind":"runtime_done","cancellation_discard":true,"unequal_geometry_fallback":true,"pressure":pressure()}),
    );
    Ok(())
}
