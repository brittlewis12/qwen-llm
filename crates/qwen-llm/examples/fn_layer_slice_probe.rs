//! Row-preserving Flash-Next paired layer checkpoints with two ready peers.
//! MODEL CORPUS [REPS=2] [SIZES=32,342,1383,1798,2048]
use anyhow::{Result, ensure};
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
    },
    qwen4exp_text_session::Qwen4ExpTextSnapshot,
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{io::Write, time::Instant};
const DEPTH: usize = 49152;
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
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
    logits: Vec<f32>,
    state: String,
    next: Vec<f32>,
    peers: Vec<Vec<Vec<f32>>>,
    peer_states: Vec<String>,
}
#[allow(clippy::too_many_arguments)]
fn arm(
    app: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    peers: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    batch: &mut Qwen4ExpDecodeBatch,
    root: &Qwen4ExpTextSnapshot,
    root_logits: &[f32],
    ids: &[u32],
    n: usize,
    s: usize,
    split: bool,
    cold: bool,
) -> Result<Arm> {
    if cold {
        app.reset()?;
    } else {
        app.restore_snapshot_with_logits(root, root_logits)?;
    }
    let append_start = if cold { 0 } else { DEPTH };
    for p in peers.iter_mut() {
        p.restore_snapshot_with_logits(root, root_logits)?;
    }
    let mut count = 0;
    let mut last = Instant::now();
    let mut gaps = Vec::new();
    let mut peer_logits = Vec::new();
    let mut peer_gpu = Vec::new();
    let mut round = || -> Result<()> {
        let tokens = [ids[DEPTH + count], ids[DEPTH + count + 64]];
        let out = batch.step(&mut peers.iter_mut().collect::<Vec<_>>(), &tokens)?;
        gaps.push(ms(last));
        last = Instant::now();
        count += 1;
        peer_logits.push(out);
        peer_gpu.push(batch.diagnostic_ms[1]);
        Ok(())
    };
    round()?;
    let start = Instant::now();
    let mut append_gpu = 0.;
    let logits = if split {
        for i in 1..=s {
            let (done, gpu) =
                app.prefill_layer_slice(&ids[append_start..append_start + n], i * 48 / s)?;
            append_gpu += gpu;
            ensure!(done == (i == s), "completion boundary");
            if i < s {
                round()?;
            }
        }
        app.logits()?.to_vec()
    } else {
        let out = app
            .prefill_continuation_with_command_checkpoint(
                &ids[append_start..append_start + n],
                || Ok(()),
            )?
            .to_vec();
        append_gpu = app.last_prefill_timing().unwrap().gpu_ms;
        out
    };
    let ttft = ms(start);
    if !split {
        for _ in 1..s {
            round()?;
        }
    }
    let wall = ms(start);
    round()?;
    drop(round);
    gaps.remove(0);
    peer_logits.remove(0);
    peer_gpu.remove(0);
    peer_gpu.pop();
    let state = app.capture_snapshot()?.diagnostic_digest();
    let next = app.forward_token(ids[append_start + n])?.to_vec();
    let peer_states = peers
        .iter()
        .map(|p| p.capture_snapshot().map(|x| x.diagnostic_digest()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Arm {
        ttft,
        wall,
        gpu: append_gpu + peer_gpu.iter().sum::<f64>(),
        gaps,
        logits,
        state,
        next,
        peers: peer_logits,
        peer_states,
    })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let reps: usize = args.get(3).map_or(Ok(2), |x| x.parse())?;
    let sizes: Vec<usize> = args
        .get(4)
        .map_or("32,342,1383,1798,2048", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let cold = args.get(5).map(String::as_str) == Some("cold");
    let append_start = if cold { 0 } else { DEPTH };
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&args[1])?;
    let cfg = Qwen4ExpConfig::from_gguf(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut text = std::fs::read_to_string(&args[2])?;
    while tok.encode(&text, false)?.len() < DEPTH + 4096 {
        let extra = text.clone();
        text.push_str(&extra);
    }
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&cfg, 98304)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, 98304)?;
    let (factory, mut app) = model.session_factory(&ctx)?;
    app.prefill(&ids[..DEPTH])?;
    let root = app.capture_snapshot()?;
    let root_logits = app.logits()?.to_vec();
    let mut peers = vec![factory.create_runner(&ctx)?, factory.create_runner(&ctx)?];
    for p in &mut peers {
        p.restore_snapshot_with_logits(&root, &root_logits)?;
    }
    let mut batch = Qwen4ExpDecodeBatch::new(&peers[0], 2)?;
    batch.step(&mut peers.iter_mut().collect::<Vec<_>>(), &[ids[DEPTH]; 2])?;
    emit(json!({"kind":"header","depth":DEPTH,"sizes":sizes,"reps":reps,"thermal":thermal()}));
    for n in sizes {
        if cold {
            app.reset()?;
        } else {
            app.restore_snapshot_with_logits(&root, &root_logits)?;
        }
        let reference = app
            .prefill_continuation_with_command_checkpoint(
                &ids[append_start..append_start + n],
                || Ok(()),
            )?
            .to_vec();
        let reference_state = app.capture_snapshot()?.diagnostic_digest();
        for s in [1, 2, 4, 8] {
            for rep in 0..reps {
                let (a, b) = if rep % 2 == 0 {
                    (
                        arm(
                            &mut app,
                            &mut peers,
                            &mut batch,
                            &root,
                            &root_logits,
                            &ids,
                            n,
                            s,
                            false,
                            cold,
                        )?,
                        arm(
                            &mut app,
                            &mut peers,
                            &mut batch,
                            &root,
                            &root_logits,
                            &ids,
                            n,
                            s,
                            true,
                            cold,
                        )?,
                    )
                } else {
                    let b = arm(
                        &mut app,
                        &mut peers,
                        &mut batch,
                        &root,
                        &root_logits,
                        &ids,
                        n,
                        s,
                        true,
                        cold,
                    )?;
                    let a = arm(
                        &mut app,
                        &mut peers,
                        &mut batch,
                        &root,
                        &root_logits,
                        &ids,
                        n,
                        s,
                        false,
                        cold,
                    )?;
                    (a, b)
                };
                let exact = same(&reference, &b.logits)
                    && same(&a.logits, &b.logits)
                    && reference_state == b.state
                    && a.state == b.state
                    && same(&a.next, &b.next)
                    && a.peer_states == b.peer_states
                    && a.peers.len() == b.peers.len()
                    && a.peers
                        .iter()
                        .flatten()
                        .zip(b.peers.iter().flatten())
                        .all(|(a, b)| same(a, b));
                emit(
                    json!({"kind":"pair","cold":cold,"rows":n,"slices":s,"rep":rep,"bitwise":exact,"thermal":thermal(),"base_ttft_ms":a.ttft,"slice_ttft_ms":b.ttft,"base_wall_ms":a.wall,"slice_wall_ms":b.wall,"base_gpu_ms":a.gpu,"slice_gpu_ms":b.gpu,"base_gaps_ms":a.gaps,"slice_gaps_ms":b.gaps}),
                );
                ensure!(exact, "native slice gate failed");
            }
        }
        if cold {
            app.reset()?;
        } else {
            app.restore_snapshot_with_logits(&root, &root_logits)?;
        }
        let (done, _) = app.prefill_layer_slice(&ids[append_start..append_start + n], 6)?;
        ensure!(
            !done && app.capture_snapshot().is_err(),
            "provisional snapshot refused"
        );
        app.cancel_layer_prefill();
        ensure!(
            app.capture_snapshot().is_err(),
            "cancelled snapshot refused"
        );
        drop(app);
        app = factory.create_runner(&ctx)?;
        if cold {
            app.reset()?;
        } else {
            app.restore_snapshot_with_logits(&root, &root_logits)?;
        }
        let resumed = app
            .prefill_continuation_with_command_checkpoint(
                &ids[append_start..append_start + n],
                || Ok(()),
            )?
            .to_vec();
        ensure!(
            same(&reference, &resumed)
                && reference_state == app.capture_snapshot()?.diagnostic_digest(),
            "cancel resume exact"
        );
        emit(json!({"kind":"cancel_resume","cold":cold,"rows":n,"bitwise":true}));
    }
    Ok(())
}
