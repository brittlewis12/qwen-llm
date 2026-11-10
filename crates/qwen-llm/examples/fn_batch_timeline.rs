#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! GPU-timestamp timeline for the one-ahead Flash-Next token loops.
//!
//! Runs three arms from the same prefilled state:
//! - `serial`: B=3 `step_tokens`, one commit-and-wait per round (no pipelining);
//! - `batch`: B=3 pipelined device-feedback loop;
//! - `single`: B=1 pipelined loop on one member.
//!
//! Each pipelined command reports its host encode span, commit time, wait span
//! and Metal `GPUStartTime`/`GPUEndTime` on one clock, so a probe can separate
//! queue handoff latency from encode overrun and from execution.
//!
//! MODEL CODE PROSE DEPTH [STEPS=64] [WIDTH=3] [WARM=8]
//! Requires `--features perf-ablations`; build with `--profile iter`.
use qwen_llm::qwen4exp_text_session::Qwen4ExpTextSnapshot;
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
        Qwen4ExpTextRunner,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use std::{
    fs,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

fn restore(
    runners: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    snapshots: &[Qwen4ExpTextSnapshot],
    logits: &[Vec<f32>],
) {
    for (r, (snap, logits)) in runners.iter_mut().zip(snapshots.iter().zip(logits)) {
        r.restore_snapshot_with_logits(snap, logits).unwrap();
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}
const QOS_CLASS_DEFAULT: u32 = 0x15;
const QOS_CLASS_USER_INITIATED: u32 = 0x19;
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

struct StopOnDrop<'a>(&'a AtomicBool);
impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[cfg(feature = "perf-ablations")]
fn timeline_json(frames: &[qwen_llm::metal::TokenFrameTimeline]) -> serde_json::Value {
    let mut encode = Vec::new();
    let mut gpu = Vec::new();
    let mut commit_to_start = Vec::new();
    let mut gpu_gap = Vec::new();
    let mut commit_before_prev_end = Vec::new();
    let mut end_to_wait = Vec::new();
    let mut rows = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let prev_end = if i > 0 {
            Some(frames[i - 1].gpu_end_s)
        } else {
            None
        };
        let gap = prev_end.map(|e| (f.gpu_start_s - e) * 1e3);
        let before = prev_end.map(|e| (e - f.commit_s) * 1e3);
        encode.push(f.encode_ms());
        gpu.push(f.gpu_ms());
        commit_to_start.push(f.commit_to_gpu_start_ms());
        end_to_wait.push(f.gpu_end_to_wait_return_ms());
        if let Some(g) = gap {
            gpu_gap.push(g);
        }
        if let Some(b) = before {
            commit_before_prev_end.push(b);
        }
        rows.push(serde_json::json!({
            "frame": f.frame,
            "forward": f.forward,
            "encode_ms": f.encode_ms(),
            "commit_to_gpu_start_ms": f.commit_to_gpu_start_ms(),
            "gpu_ms": f.gpu_ms(),
            "gpu_end_to_wait_return_ms": f.gpu_end_to_wait_return_ms(),
            "wait_ms": f.wait_ms(),
            "commit_before_prev_gpu_end_ms": before,
            "gpu_gap_from_prev_end_ms": gap,
        }));
    }
    let forwarded = frames.iter().filter(|f| f.forward).count();
    let submitted_busy = commit_before_prev_end.iter().filter(|b| **b > 0.).count();
    serde_json::json!({
        "frames": frames.len(),
        "forward_frames": forwarded,
        "mean_encode_ms": mean(&encode),
        "median_encode_ms": median(encode.clone()),
        "mean_gpu_ms": mean(&gpu),
        "median_gpu_ms": median(gpu.clone()),
        "mean_commit_to_gpu_start_ms": mean(&commit_to_start),
        "median_commit_to_gpu_start_ms": median(commit_to_start.clone()),
        "mean_gpu_gap_from_prev_end_ms": mean(&gpu_gap),
        "median_gpu_gap_from_prev_end_ms": median(gpu_gap.clone()),
        "mean_commit_before_prev_gpu_end_ms": mean(&commit_before_prev_end),
        "median_commit_before_prev_gpu_end_ms": median(commit_before_prev_end.clone()),
        "frames_committed_while_gpu_busy": submitted_busy,
        "mean_gpu_end_to_wait_return_ms": mean(&end_to_wait),
        "median_gpu_end_to_wait_return_ms": median(end_to_wait.clone()),
        "rows": rows,
    })
}

#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[4].parse()?;
    let steps: usize = a.get(5).map_or(Ok(64), |s| s.parse())?;
    let width: usize = a.get(6).map_or(Ok(3), |s| s.parse())?;
    let warm: usize = a.get(7).map_or(Ok(8), |s| s.parse())?;
    let load: usize = a.get(8).map_or(Ok(0), |s| s.parse())?;
    let qos: u32 = a.get(9).map_or(Ok(0), |s| s.parse())?;
    let qos_class = match qos {
        1 => QOS_CLASS_USER_INTERACTIVE,
        2 => QOS_CLASS_USER_INITIATED,
        _ => 0,
    };
    assert!(steps > 1 && warm > 0 && (2..=4).contains(&width));
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&gg)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let texts = [fs::read_to_string(&a[2])?, fs::read_to_string(&a[3])?];
    let mut docs = Vec::new();
    for i in 0..width {
        let ids = tok.encode(&texts[i % 2], false)?;
        assert!(!ids.is_empty());
        docs.push(
            (0..depth + 512 + steps + warm)
                .map(|j| ids[(j + i * 157) % ids.len()] as u32)
                .collect::<Vec<_>>(),
        );
    }
    let extent = depth + 4096 + steps + warm;
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gg,
        Qwen4ExpSessionCapacity::for_forward_limit(&c, extent)?,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )?;
    let (factory, first) = loaded.session_factory(&ctx)?;
    let mut runners = vec![first];
    for _ in 1..width {
        runners.push(factory.create_runner(&ctx)?);
    }
    let mut snapshots = Vec::new();
    let mut prefill_logits = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        let logits = r.prefill(&docs[i][..depth + i * 17])?;
        assert_eq!(logits.as_slice().len(), c.vocab_size as usize);
        prefill_logits.push(logits.as_slice().to_vec());
        snapshots.push(r.capture_snapshot()?);
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners[0], width)?;
    let config = SamplingConfig::qwen38_flash_next(1);
    for r in runners.iter_mut() {
        r.configure_gpu_token_loop(true);
    }

    // Warm both shapes so first-touch pipeline and command allocation are not
    // part of the measured timelines.
    restore(&mut runners, &snapshots, &prefill_logits);
    let mut samplers: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
    {
        let mut members: Vec<_> = runners.iter_mut().collect();
        let mut refs: Vec<_> = samplers.iter_mut().collect();
        let _ = native.step_tokens(
            &mut members,
            &mut refs,
            &docs.iter().map(|d| d[depth]).collect::<Vec<_>>(),
        )?;
    }
    restore(&mut runners, &snapshots, &prefill_logits);
    for s in samplers.iter_mut() {
        *s = Sampler::new(config).unwrap();
    }
    {
        let mut members: Vec<_> = runners.iter_mut().collect();
        let mut refs: Vec<_> = samplers.iter_mut().collect();
        let _ = native.generate_gpu_tokens(
            &mut members,
            &mut refs,
            &vec![warm; width],
            &vec![&[][..]; width],
            |_, _, _| true,
        )?;
    }
    restore(&mut runners[..1], &snapshots[..1], &prefill_logits[..1]);
    {
        let mut s = Sampler::new(config).unwrap();
        let _ = runners[0].generate_gpu_tokens(&mut s, warm, &[], |_| true)?;
    }

    // Serial B=3: one commit-and-wait per round, host samples between rounds.
    restore(&mut runners, &snapshots, &prefill_logits);
    for s in samplers.iter_mut() {
        *s = Sampler::new(config).unwrap();
    }
    let mut round_ms = Vec::new();
    let (mut diag_encode, mut diag_gpu, mut diag_wall) = (0., 0., 0.);
    let mut serial_tokens: Vec<u32> = prefill_logits
        .iter()
        .zip(samplers.iter_mut())
        .map(|(l, s)| s.sample(l).unwrap().token as u32)
        .collect();
    for _ in 0..steps {
        let tokens = serial_tokens.clone();
        let t = Instant::now();
        let mut members: Vec<_> = runners.iter_mut().collect();
        let mut refs: Vec<_> = samplers.iter_mut().collect();
        serial_tokens = native.step_tokens(&mut members, &mut refs, &tokens)?;
        round_ms.push(t.elapsed().as_secs_f64() * 1e3);
        diag_encode += native.diagnostic_ms[0];
        diag_gpu += native.diagnostic_ms[1];
        diag_wall += native.diagnostic_ms[2];
    }

    // Pipelined B=3.
    restore(&mut runners, &snapshots, &prefill_logits);
    for s in samplers.iter_mut() {
        *s = Sampler::new(config).unwrap();
    }
    let batch_started = Instant::now();
    let batch_stats = {
        let mut members: Vec<_> = runners.iter_mut().collect();
        let mut refs: Vec<_> = samplers.iter_mut().collect();
        native.generate_gpu_tokens(
            &mut members,
            &mut refs,
            &vec![steps; width],
            &vec![&[][..]; width],
            |_, _, _| true,
        )?
    };
    let batch_wall = batch_started.elapsed().as_secs_f64() * 1e3;

    // Pipelined B=1.
    restore(&mut runners[..1], &snapshots[..1], &prefill_logits[..1]);
    let mut single_sampler = Sampler::new(config).unwrap();
    let single_started = Instant::now();
    let single_stats = runners[0].generate_gpu_tokens(&mut single_sampler, steps, &[], |_| true)?;
    let single_wall = single_started.elapsed().as_secs_f64() * 1e3;

    // Pipelined B=3 with the host CPU saturated, reproducing the build-load
    // conditions under which the earlier A/B regressed.
    let loaded = if load > 0 {
        restore(&mut runners, &snapshots, &prefill_logits);
        for s in samplers.iter_mut() {
            *s = Sampler::new(config).unwrap();
        }
        let stop = AtomicBool::new(false);
        let (stats, wall) = std::thread::scope(|scope| {
            let _guard = StopOnDrop(&stop);
            let mut spins = Vec::with_capacity(load);
            for _ in 0..load {
                let stop = &stop;
                spins.push(scope.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                }));
            }
            if qos_class != 0 {
                // Raise only the loop thread after the load threads have been
                // spawned, so they keep the default class.
                unsafe {
                    pthread_set_qos_class_self_np(qos_class, 0);
                }
            }
            let started = Instant::now();
            let mut members: Vec<_> = runners.iter_mut().collect();
            let mut refs: Vec<_> = samplers.iter_mut().collect();
            let stats = native
                .generate_gpu_tokens(
                    &mut members,
                    &mut refs,
                    &vec![steps; width],
                    &vec![&[][..]; width],
                    |_, _, _| true,
                )
                .unwrap();
            let wall = started.elapsed().as_secs_f64() * 1e3;
            if qos_class != 0 {
                unsafe {
                    pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
                }
            }
            stop.store(true, Ordering::Relaxed);
            for spin in spins {
                let _ = spin.join();
            }
            (stats, wall)
        });
        Some((stats, wall))
    } else {
        None
    };

    let serial_round = mean(&round_ms);
    let loaded_json = loaded.as_ref().map(|(stats, wall)| {
        serde_json::json!({
            "wall_ms": wall,
            "stats_wall_ms": stats.wall_ms,
            "gpu_ms_total": stats.gpu_ms,
            "encode_ms_total": stats.encode_cpu_ms,
            "completion_wait_ms_total": stats.completion_wait_ms,
            "round_ms": wall / steps as f64,
            "timeline": timeline_json(&stats.timeline),
        })
    });
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_batch_timeline",
            "depth": depth,
            "steps": steps,
            "width": width,
            "warm": warm,
            "serial": {
                "mean_round_ms": serial_round,
                "median_round_ms": median(round_ms.clone()),
                "mean_encode_ms": diag_encode / steps as f64,
                "mean_gpu_ms": diag_gpu / steps as f64,
                "mean_commit_readback_ms": (diag_wall - diag_gpu - diag_encode) / steps as f64,
                "round_minus_gpu_ms": serial_round - diag_gpu / steps as f64,
            },
            "batch_pipelined": {
                "wall_ms": batch_wall,
                "stats_wall_ms": batch_stats.wall_ms,
                "gpu_ms_total": batch_stats.gpu_ms,
                "encode_ms_total": batch_stats.encode_cpu_ms,
                "completion_wait_ms_total": batch_stats.completion_wait_ms,
                "round_ms": batch_wall / steps as f64,
                "timeline": timeline_json(&batch_stats.timeline),
            },
            "single_pipelined": {
                "wall_ms": single_wall,
                "stats_wall_ms": single_stats.wall_ms,
                "gpu_ms_total": single_stats.gpu_ms,
                "encode_ms_total": single_stats.encode_cpu_ms,
                "completion_wait_ms_total": single_stats.completion_wait_ms,
                "round_ms": single_wall / steps as f64,
                "timeline": timeline_json(&single_stats.timeline),
            },
            "batch_pipelined_loaded": loaded_json,
        })
    );
    Ok(())
}
