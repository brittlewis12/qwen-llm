//! Row-preserving V4.1 layer-boundary price and exactness probe.
//! v41_layer_slice_probe <model> <corpus> [reps=2] [sizes=32,342,1509]
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, Snapshot, State, profile},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, MetalContext},
    tokenizer::NativeTokenizer,
};
use serde_json::json;
use std::{io::Write, sync::Arc, time::Instant};
const DEPTH: usize = 48 * 1024;
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.is_finite() && x.to_bits() == y.to_bits())
}
fn thermal() -> i32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|x| String::from_utf8(x.stdout).ok())
        .and_then(|x| x.split_whitespace().last()?.parse().ok())
        .unwrap_or(-1)
}
struct Arm {
    ttft: f64,
    wall: f64,
    gpu: f64,
    gaps: Vec<f64>,
    commands: Vec<(usize, f64)>,
    logits: Vec<f32>,
    digest: String,
    next: Vec<f32>,
    peers: Vec<Vec<Vec<f32>>>,
    peer_digests: Vec<String>,
}
fn arm(
    f: &Forward<'_>,
    root: &Snapshot,
    ids: &[u32],
    n: usize,
    ends: &[usize],
    split: bool,
) -> Result<Arm> {
    let mut app = f.new_state(98_304)?;
    f.restore_snapshot(&mut app, root)?;
    let mut peers: Vec<State> = (0..2)
        .map(|_| f.new_state(98_304))
        .collect::<std::result::Result<_, _>>()?;
    for peer in &mut peers {
        f.restore_snapshot(peer, root)?;
    }
    let mut count = 0;
    let mut gaps = Vec::new();
    let mut peer_logits = Vec::new();
    let mut last = Instant::now();
    let mut round = || -> std::result::Result<(), String> {
        let tokens = vec![ids[DEPTH + count], ids[DEPTH + count + 64]];
        let logits = f.step_batch(&mut peers.iter_mut().collect::<Vec<_>>(), &tokens, None)?;
        gaps.push(ms(last));
        last = Instant::now();
        count += 1;
        peer_logits.push(logits);
        Ok(())
    };
    round().map_err(anyhow::Error::msg)?;
    let scope = profile::Scope::begin().map_err(anyhow::Error::msg)?;
    let started = Instant::now();
    let unsplit = [40];
    let (logits, commands) = f.prefill_layer_sliced_probe(
        &mut app,
        &ids[DEPTH..DEPTH + n],
        None,
        if split { ends } else { &unsplit },
        &mut round,
    )?;
    let ttft = ms(started);
    if !split {
        for _ in 1..ends.len() {
            round().map_err(anyhow::Error::msg)?;
        }
    }
    let wall = ms(started);
    let report = scope.finish();
    let gpu = report
        .operations
        .iter()
        .filter(|x| x.completed_submits > 0)
        .map(|x| x.gpu_ms.unwrap_or(0.0))
        .sum();
    round().map_err(anyhow::Error::msg)?;
    drop(round);
    gaps.remove(0);
    peer_logits.remove(0);
    let snapshot = f.capture_snapshot(&app)?;
    let digest = snapshot.diagnostic_digest();
    let next = f
        .step(&mut app, ids[DEPTH + n], None)
        .map_err(anyhow::Error::msg)?;
    let peer_digests = peers
        .iter()
        .map(|s| f.capture_snapshot(s).map(|s| s.diagnostic_digest()))
        .collect::<std::result::Result<_, _>>()?;
    Ok(Arm {
        ttft,
        wall,
        gpu,
        gaps,
        commands,
        logits,
        digest,
        next,
        peers: peer_logits,
        peer_digests,
    })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() >= 3, "model corpus [reps] [sizes]");
    let reps: usize = args.get(3).map_or(Ok(2), |s| s.parse())?;
    let sizes: Vec<usize> = args
        .get(4)
        .map_or("32,342,1509", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let ctx = MetalContext::new()?;
    let g = Arc::new(GgufFile::open(&args[1])?);
    let storage = resident_storage_plan_with_target(
        &g,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(4_294_967_296),
    )?;
    let w = ResidentWeights::load_with_storage_plan(&ctx, g.clone(), &storage)?;
    let f = Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .map_err(anyhow::Error::msg)?;
    let tok = NativeTokenizer::from_gguf(&g)?;
    let mut text = std::fs::read_to_string(&args[2])?;
    while tok.encode(&text, false)?.len() < DEPTH + sizes.iter().max().unwrap() + 100 {
        let extra = text.clone();
        text.push_str(&extra);
    }
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    emit(json!({"kind":"load", "depth":DEPTH,"thermal":thermal(),"sizes":sizes}));
    let mut prefix = f.new_state(98_304)?;
    ensure!(
        f.begin_cold_prefill_episode(&mut prefix, DEPTH)
            .map_err(anyhow::Error::msg)?,
        "CED prefix"
    );
    loop {
        if f.cold_prefill_episode_slice(&mut prefix, &ids[..DEPTH], None, 4096)
            .map_err(anyhow::Error::msg)?
            .complete
        {
            break;
        }
    }
    let root = f.capture_snapshot(&prefix)?;
    drop(prefix);
    emit(json!({"kind":"snapshot","thermal":thermal(),"digest":root.diagnostic_digest()}));
    for n in sizes {
        let mut ordinary = f.new_state(98_304)?;
        f.restore_snapshot(&mut ordinary, &root)?;
        let reference = f.prefill(&mut ordinary, &ids[DEPTH..DEPTH + n], None)?;
        let reference_digest = f.capture_snapshot(&ordinary)?.diagnostic_digest();
        drop(ordinary);
        for s in [1, 2, 4, 8] {
            let ends: Vec<_> = (1..=s).map(|i| i * 40 / s).collect();
            for rep in 0..reps {
                let (a, b) = if rep % 2 == 0 {
                    (
                        arm(&f, &root, &ids, n, &ends, false)?,
                        arm(&f, &root, &ids, n, &ends, true)?,
                    )
                } else {
                    let b = arm(&f, &root, &ids, n, &ends, true)?;
                    let a = arm(&f, &root, &ids, n, &ends, false)?;
                    (a, b)
                };
                let exact = same(&a.logits, &b.logits)
                    && same(&reference, &b.logits)
                    && a.digest == b.digest
                    && b.digest == reference_digest
                    && same(&a.next, &b.next)
                    && a.peer_digests == b.peer_digests
                    && a.peers
                        .iter()
                        .flatten()
                        .zip(b.peers.iter().flatten())
                        .all(|(x, y)| same(x, y));
                emit(
                    json!({"kind":"pair","rows":n,"slices":s,"rep":rep,"bitwise":exact,"thermal":thermal(),
                    "base_ttft_ms":a.ttft,"slice_ttft_ms":b.ttft,"base_wall_ms":a.wall,"slice_wall_ms":b.wall,
                    "base_gpu_ms":a.gpu,"slice_gpu_ms":b.gpu,"gpu_ratio":b.gpu/a.gpu,
                    "base_gaps_ms":a.gaps,"slice_gaps_ms":b.gaps,"commands":b.commands}),
                );
                ensure!(exact, "slice exactness gate failed");
            }
        }
    }
    Ok(())
}
