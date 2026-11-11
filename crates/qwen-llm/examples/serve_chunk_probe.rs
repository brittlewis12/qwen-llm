//! Serve chunk-price probe; build with `--features perf-ablations`.
//!
//! Two questions, one model load:
//!  1. `stage`: at 48K, where does the per-append wall go (prepare/admission,
//!     kernel, finish/release), first append versus repeats, idle versus a
//!     CPU-loaded host?
//!  2. `chunk`: at 48K, does a single C-row prefill chunk cost
//!     `floor + per_row*C` with one floor per forward, or does the floor come
//!     back per chunk? Also times a 1509-row append split into C-row chunks and
//!     the marginal 3 rows in a prefill forward.
//!
//! `all` runs both. Output is JSON lines; no production scheduling is changed.
//!
//! Usage: `serve_chunk_probe <corpus> [all|stage|chunk] [first_n] [load_threads]`
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
    metal::*,
    tokenizer::Tokenizer,
};
use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const DEPTH: usize = 48 * 1024;
const CAPACITY: usize = 65536;
const PREFILL_ROWS: usize = 2048;
const CHUNKS: [usize; 6] = [1024, 512, 256, 128, 64, 32];
const STAGE_N: [usize; 4] = [1, 32, 85, 342];
const SPLIT_ROWS: usize = 1509;
const REPS: usize = 5;

fn emit(v: serde_json::Value) {
    println!("{v}");
    let _ = std::io::stdout().flush();
}

fn pressure() -> i32 {
    let x = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok();
    x.and_then(|x| {
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

fn clock_note() -> String {
    // Unprivileged machines cannot read GPU clocks; record whatever the
    // system will tell us without prompting.
    for (program, args) in [
        (
            "sudo",
            vec![
                "-n",
                "powermetrics",
                "--samplers",
                "gpu_power",
                "-n",
                "1",
                "-i",
                "100",
                "-f",
                "text",
            ],
        ),
        (
            "ioreg",
            vec!["-r", "-d", "1", "-c", "AGXAccelerator", "-w", "0"],
        ),
    ] {
        let Ok(x) = std::process::Command::new(program).args(&args).output() else {
            continue;
        };
        if !x.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&x.stdout);
        if let Some(line) = text
            .lines()
            .find(|l| l.contains("GPU") && (l.contains("Hz") || l.contains("clock")))
        {
            return line.trim().chars().take(200).collect();
        }
    }
    "unavailable".into()
}

fn gate() {
    while pressure() > 1 {
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

struct HostLoad {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl HostLoad {
    fn start(threads: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..threads)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        x = std::hint::black_box(
                            x.wrapping_mul(6364136223846793005).wrapping_add(1),
                        );
                    }
                })
            })
            .collect();
        Self { stop, handles }
    }
}

impl Drop for HostLoad {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let corpus = args.get(1).ok_or_else(|| {
        anyhow::anyhow!("usage: serve_chunk_probe <corpus> [mode] [first_n] [load_threads] [capacity] [decode_steps]")
    })?;
    let mode = args.get(2).map(String::as_str).unwrap_or("all");
    let first_n: usize = args.get(3).map_or(Ok(32), |s| s.parse())?;
    let load_threads: usize = args.get(4).map_or(Ok(0), |s| s.parse())?;
    let capacity: usize = args.get(5).map_or(Ok(CAPACITY), |s| s.parse())?;
    let decode_steps: usize = args.get(6).map_or(Ok(0), |s| s.parse())?;

    let c = MetalContext::new()?;
    let path = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
        .path_or_skip()
        .ok_or_else(|| anyhow::anyhow!("resolve GLM via QWEN_MODEL_SEARCH_PATH"))?;
    let g = GgufFile::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let text = std::fs::read_to_string(corpus)?
        .replace("<|", "‹|")
        .replace("<｜", "‹｜");
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    anyhow::ensure!(
        ids.len() >= DEPTH + SPLIT_ROWS + 64,
        "corpus has {} tokens, need {}",
        ids.len(),
        DEPTH + SPLIT_ROWS + 64
    );

    let w = Glm5NextWeights::load(&c, &g)?;
    emit(serde_json::json!({
        "kind": "header", "mode": mode, "depth": DEPTH, "first_n": first_n,
        "load_threads": load_threads, "capacity": capacity,
        "decode_steps": decode_steps, "pressure": pressure(), "loadavg": loadavg(),
        "clock": clock_note(), "corpus_tokens": ids.len(),
    }));

    if mode == "all" || mode == "stage" {
        stage(&c, &w, &ids, first_n, load_threads, capacity, decode_steps)?;
    }
    if mode == "all" || mode == "chunk" {
        chunk(&c, &w, &ids)?;
    }
    if mode == "edges" {
        edges(&c, &w, &ids)?;
    }
    emit(serde_json::json!({
        "kind": "done", "pressure": pressure(), "loadavg": loadavg(),
        "clock": clock_note(),
    }));
    Ok(())
}

fn new_session<'a>(
    c: &MetalContext,
    w: &'a Glm5NextWeights,
    ids: &[u32],
    capacity: usize,
    decode_steps: usize,
) -> anyhow::Result<Glm5NextSession<'a>> {
    let mut s = Glm5NextSession::with_prefill_rows(c, w, capacity, PREFILL_ROWS)?;
    for part in ids[..DEPTH].chunks(PREFILL_ROWS) {
        gate();
        s.prefill_packed(c, part)?;
    }
    // Serve's first measured append follows the warm-up reply's decode, not
    // the root prefill itself. Advance the same way before time zero.
    for step in 0..decode_steps {
        gate();
        let logits = s.forward(c, ids[DEPTH + step])?;
        anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite decode");
    }
    let base = DEPTH + decode_steps;
    anyhow::ensure!(s.position() == base, "prefill position {}", s.position());
    let released = s.prepare_decode(c)?;
    emit(serde_json::json!({
        "kind": "session", "depth": DEPTH, "base": base, "capacity": capacity,
        "decode_steps": decode_steps, "prefill_rows": PREFILL_ROWS,
        "released_bytes": released, "pressure": pressure(), "loadavg": loadavg(),
    }));
    Ok(s)
}

fn measure(
    s: &mut Glm5NextSession<'_>,
    c: &MetalContext,
    tokens: &[u32],
) -> anyhow::Result<(f64, f64, f64, u64, String)> {
    let t = Instant::now();
    s.prepare_prefill(c, 0)?;
    let prepare_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let logits = s.prefill_packed(c, tokens)?;
    let kernel_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let released = s.prepare_decode(c)?;
    let finish_ms = t.elapsed().as_secs_f64() * 1e3;
    anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    let hash = blake3::hash(bytemuck::cast_slice(&logits))
        .to_hex()
        .to_string();
    Ok((prepare_ms, kernel_ms, finish_ms, released, hash))
}

fn profiled(
    s: &mut Glm5NextSession<'_>,
    c: &MetalContext,
    tokens: &[u32],
) -> anyhow::Result<(f64, f64)> {
    let t = Instant::now();
    let (logits, report) = s.prefill_chunk_stage_profiled(c, tokens)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    Ok((wall_ms, report.command_gpu_ms))
}

fn stage(
    c: &MetalContext,
    w: &Glm5NextWeights,
    ids: &[u32],
    first_n: usize,
    load_threads: usize,
    capacity: usize,
    decode_steps: usize,
) -> anyhow::Result<()> {
    let base = DEPTH + decode_steps;
    let mut s = new_session(c, w, ids, capacity, decode_steps)?;
    let snap = s.capture_snapshot()?;
    let load = (load_threads > 0).then(|| HostLoad::start(load_threads));
    if load.is_some() {
        // Let the busy threads saturate before the first append.
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    emit(serde_json::json!({
        "kind": "load_start", "threads": load_threads, "loadavg": loadavg(),
        "pressure": pressure(),
    }));

    let (p, k, f, rel, hash) = measure(&mut s, c, &ids[base..base + first_n])?;
    emit(serde_json::json!({
        "kind": "stage", "first": true, "n": first_n, "rep": 0,
        "prepare_ms": p, "kernel_ms": k, "finish_ms": f, "total_ms": p + k + f,
        "released_bytes": rel, "hash": hash,
        "pressure": pressure(), "loadavg": loadavg(), "clock": clock_note(),
    }));
    gate();
    s.restore_snapshot(&snap)?;
    let (pwall, pgpu) = profiled(&mut s, c, &ids[base..base + first_n])?;
    emit(serde_json::json!({
        "kind": "stage_profiled", "first": true, "n": first_n,
        "wall_ms": pwall, "gpu_ms": pgpu, "wall_minus_gpu_ms": pwall - pgpu,
    }));

    for n in STAGE_N {
        let mut prepares = Vec::new();
        let mut kernels = Vec::new();
        let mut finishes = Vec::new();
        let mut releases = Vec::new();
        for rep in 0..REPS {
            gate();
            s.restore_snapshot(&snap)?;
            let (p, k, f, rel, hash) = measure(&mut s, c, &ids[base..base + n])?;
            prepares.push(p);
            kernels.push(k);
            finishes.push(f);
            releases.push(rel);
            emit(serde_json::json!({
                "kind": "stage", "first": false, "n": n, "rep": rep,
                "prepare_ms": p, "kernel_ms": k, "finish_ms": f, "total_ms": p + k + f,
                "released_bytes": rel, "hash": hash,
                "pressure": pressure(), "loadavg": loadavg(),
            }));
        }
        gate();
        s.restore_snapshot(&snap)?;
        let (pwall, pgpu) = profiled(&mut s, c, &ids[base..base + n])?;
        let prepare = median(prepares);
        let kernel = median(kernels);
        let finish = median(finishes);
        emit(serde_json::json!({
            "kind": "stage_warm", "n": n,
            "prepare_ms": prepare, "kernel_ms": kernel,
            "finish_ms": finish, "total_ms": prepare + kernel + finish,
            "released_bytes": releases.iter().copied().max().unwrap_or(0),
            "profiled_wall_ms": pwall, "gpu_ms": pgpu,
        }));
    }
    drop(load);
    emit(serde_json::json!({
        "kind": "load_end", "threads": load_threads, "loadavg": loadavg(),
    }));
    Ok(())
}

fn edges(c: &MetalContext, w: &Glm5NextWeights, ids: &[u32]) -> anyhow::Result<()> {
    let mut s = new_session(c, w, ids, CAPACITY, 0)?;
    let snap = s.capture_snapshot()?;
    for width in [
        64usize, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1025,
        1026, 1027, 1032,
    ] {
        let mut walls = Vec::new();
        for _ in 0..3 {
            gate();
            s.restore_snapshot(&snap)?;
            let t = Instant::now();
            let logits = s.prefill_packed(c, &ids[DEPTH..DEPTH + width])?;
            walls.push(t.elapsed().as_secs_f64() * 1e3);
            anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
        }
        emit(serde_json::json!({
            "kind": "edge", "width": width, "wall_ms": median(walls),
            "pressure": pressure(), "loadavg": loadavg(),
        }));
    }
    for base in [512usize, 1024] {
        for extra in [1usize, 2, 3] {
            let mut base_ms = Vec::new();
            let mut plus_ms = Vec::new();
            for _ in 0..3 {
                gate();
                s.restore_snapshot(&snap)?;
                let t = Instant::now();
                s.prefill_packed(c, &ids[DEPTH..DEPTH + base])?;
                base_ms.push(t.elapsed().as_secs_f64() * 1e3);
                gate();
                s.restore_snapshot(&snap)?;
                let t = Instant::now();
                s.prefill_packed(c, &ids[DEPTH..DEPTH + base + extra])?;
                plus_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let b = median(base_ms);
            let p = median(plus_ms);
            emit(serde_json::json!({
                "kind": "edge_marginal", "base": base, "extra": extra,
                "base_ms": b, "plus_ms": p, "delta_ms": p - b,
            }));
        }
    }
    Ok(())
}

fn chunk(c: &MetalContext, w: &Glm5NextWeights, ids: &[u32]) -> anyhow::Result<()> {
    let mut s = new_session(c, w, ids, CAPACITY, 0)?;
    let snap = s.capture_snapshot()?;

    let mut fit = Vec::new();
    for chunk in CHUNKS {
        let mut walls = Vec::new();
        for rep in 0..REPS {
            gate();
            s.restore_snapshot(&snap)?;
            let t = Instant::now();
            let logits = s.prefill_packed(c, &ids[DEPTH..DEPTH + chunk])?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
            walls.push(ms);
            emit(serde_json::json!({
                "kind": "chunk", "rows": chunk, "rep": rep, "ms": ms,
                "pressure": pressure(), "loadavg": loadavg(),
            }));
        }
        gate();
        s.restore_snapshot(&snap)?;
        let (pwall, pgpu) = profiled(&mut s, c, &ids[DEPTH..DEPTH + chunk])?;
        let warm = median(walls);
        fit.push((chunk as f64, warm));
        emit(serde_json::json!({
            "kind": "chunk_warm", "rows": chunk, "wall_ms": warm,
            "profiled_wall_ms": pwall, "gpu_ms": pgpu,
            "wall_minus_gpu_ms": pwall - pgpu,
        }));
    }

    // Least squares of warm single-chunk wall on rows: wall = floor + per_row*rows.
    let n = fit.len() as f64;
    let sx: f64 = fit.iter().map(|(x, _)| x).sum();
    let sy: f64 = fit.iter().map(|(_, y)| y).sum();
    let sxx: f64 = fit.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = fit.iter().map(|(x, y)| x * y).sum();
    let per_row = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let floor = (sy - per_row * sx) / n;
    emit(serde_json::json!({
        "kind": "chunk_fit", "floor_ms": floor, "per_row_ms": per_row,
    }));

    // Split the same 1509-row append into C-row chunks and charge the result
    // against the single-forward fit.
    let tokens = &ids[DEPTH..DEPTH + SPLIT_ROWS];
    for chunk in CHUNKS {
        let mut walls = Vec::new();
        for _ in 0..REPS {
            gate();
            s.restore_snapshot(&snap)?;
            let t = Instant::now();
            let mut logits = Vec::new();
            for part in tokens.chunks(chunk) {
                logits = s.prefill_packed(c, part)?;
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            walls.push(ms);
            anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
        }
        let warm = median(walls);
        emit(serde_json::json!({
            "kind": "split", "rows": SPLIT_ROWS, "chunk": chunk,
            "calls": SPLIT_ROWS.div_ceil(chunk), "total_ms": warm,
            "predicted_ms": floor + per_row * SPLIT_ROWS as f64,
            "per_chunk_predicted_ms": floor + per_row * chunk as f64,
            "mean_chunk_ms": warm / SPLIT_ROWS.div_ceil(chunk) as f64,
        }));
    }

    // Marginal cost of a few more rows inside one prefill forward. This is the
    // same-sequence price; cross-sequence decode rows additionally need a
    // per-row KV/attention mapping.
    for base in [512usize, 128] {
        let mut one = Vec::new();
        let mut plus = Vec::new();
        for _ in 0..REPS {
            gate();
            s.restore_snapshot(&snap)?;
            let t = Instant::now();
            s.prefill_packed(c, &ids[DEPTH..DEPTH + base])?;
            one.push(t.elapsed().as_secs_f64() * 1e3);
            gate();
            s.restore_snapshot(&snap)?;
            let t = Instant::now();
            s.prefill_packed(c, &ids[DEPTH..DEPTH + base + 3])?;
            plus.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let base_ms = median(one);
        let plus_ms = median(plus);
        emit(serde_json::json!({
            "kind": "marginal", "base": base, "extra": 3,
            "base_ms": base_ms, "plus3_ms": plus_ms,
            "delta_ms": plus_ms - base_ms,
        }));
    }
    Ok(())
}
