//! Saved-prefix decode/profile witness; run under build.lock + the Metal lease.
#![allow(dead_code)]
#[path = "../src/qwen/prefill_plan.rs"]
mod prefill_plan;

use anyhow::{Result, ensure};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::{
    checkpoint_store::DurableCheckpointStore,
    metal::*,
    runtime::{LoadedModel, Runtime, Sequence, SequenceConfig},
    tensor::GgmlType,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long, default_value = "profile")]
    mode: String,
    #[arg(long, default_value_t = 32768)]
    depth: usize,
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    #[arg(long, default_value_t = 2)]
    reps: usize,
    #[arg(long, default_value_t = 24)]
    heads: usize,
    #[arg(long, default_value_t = 4)]
    kv_heads: usize,
    #[arg(long)]
    qreg: bool,
    #[arg(long)]
    automatic: bool,
}
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn ids(n: usize, vocab: u32, mut state: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(vocab)) as i32
        })
        .collect()
}
fn thermal(label: &str, where_: &str) -> Result<()> {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()?;
    let value = String::from_utf8(o.stdout)?;
    emit(json!({"record":"thermal","arm":label,"where":where_,"value":value.trim()}));
    ensure!(
        o.status.success() && value.split_whitespace().last() == Some("0"),
        "nonzero thermal pressure; stop"
    );
    Ok(())
}
fn floats(t: &MetalTensor) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
    }
}
fn run_decode(model: &LoadedModel, seq: &mut Sequence, tokens: &[i32]) -> Result<Vec<f32>> {
    let mut last = Vec::new();
    for (index, &id) in tokens.iter().enumerate() {
        qwen_cli::shutdown::checkpoint()?;
        let logits = model.decode_token(seq, id)?;
        if index + 1 == tokens.len() {
            last = logits;
        } else {
            std::hint::black_box(logits);
        }
    }
    Ok(last)
}
fn op(ctx: &MetalContext, a: &Args) -> Result<()> {
    ensure!(
        matches!((a.heads, a.kv_heads), (24, 4) | (16, 2)),
        "unsupported census heads"
    );
    let group = a.heads / a.kv_heads;
    let mut random_state = 0xd3c0de51u64;
    let mut random = || {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        ((random_state >> 40) as f32 / 16_777_216. * 2. - 1.) * 1.7320508
    };
    for pos in [32769usize, 131073] {
        let q: Vec<f32> = (0..a.heads * 256).map(|_| random()).collect();
        let k: Vec<u16> = (0..pos * a.kv_heads * 256)
            .map(|_| half::f16::from_f32(random()).to_bits())
            .collect();
        let v: Vec<u16> = (0..k.len())
            .map(|_| half::f16::from_f32(random()).to_bits())
            .collect();
        let make = |b: &[u8], n: usize, d| MetalTensor::from_bytes(ctx, b, vec![n as u64], d);
        let qt = make(bytemuck::cast_slice(&q), q.len(), GgmlType::F32)?;
        let kt = make(bytemuck::cast_slice(&k), k.len(), GgmlType::F16)?;
        let vt = make(bytemuck::cast_slice(&v), v.len(), GgmlType::F16)?;
        let out = MetalTensor::zeros_f32(ctx, vec![q.len() as u64])?;
        let partial = MetalTensor::zeros_f32(ctx, vec![(a.heads * 1024 * 256) as u64])?;
        let ml = MetalTensor::zeros_f32(ctx, vec![(a.heads * 1024 * 2) as u64])?;
        let base = (
            attn_v4_choose_nwg(pos, group),
            attn_v4_choose_tile_c(pos, group),
            false,
        );
        let one = |nwg, c, qreg| -> Result<f64> {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            with_attn_v4_register_q_override(qreg, || {
                encode_attn_decode_v4_f32(
                    ctx, &enc, &qt, &kt, &vt, &partial, &ml, &out, a.heads, a.kv_heads, 256, pos,
                    nwg, c,
                )
            })?;
            enc.end();
            cmd.commit();
            wait_completed(&cmd)?;
            ensure!(
                floats(&out).iter().all(|x| x.is_finite()),
                "nonfinite attention"
            );
            Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3)
        };
        one(base.0, base.1, base.2)?;
        let incumbent = floats(&out).to_vec();
        // Original F32 query and the actual stored half cache, independent F64 softmax.
        let mut reference = Vec::new();
        for h in [0, a.heads / 2, a.heads - 1] {
            let scores: Vec<f64> = (0..pos)
                .map(|p| {
                    (0..256)
                        .map(|d| {
                            f64::from(q[h * 256 + d])
                                * f64::from(
                                    half::f16::from_bits(
                                        k[p * a.kv_heads * 256 + (h / group) * 256 + d],
                                    )
                                    .to_f32(),
                                )
                        })
                        .sum::<f64>()
                        / 16.
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let sum = weights.iter().sum::<f64>();
            for d in 0..256 {
                reference.push((
                    h * 256 + d,
                    weights
                        .iter()
                        .enumerate()
                        .map(|(p, w)| {
                            w * f64::from(
                                half::f16::from_bits(
                                    v[p * a.kv_heads * 256 + (h / group) * 256 + d],
                                )
                                .to_f32(),
                            )
                        })
                        .sum::<f64>()
                        / sum,
                ));
            }
        }
        let error = |ys: &[f32]| {
            let (mut peak, mut e, mut r) = (0f64, 0f64, 0f64);
            for &(i, x) in &reference {
                let d = f64::from(ys[i]) - x;
                assert!(d.is_finite());
                peak = peak.max(d.abs());
                e += d * d;
                r += x * x;
            }
            (peak, (e / r).sqrt())
        };
        let inc = error(&incumbent);
        let mut config_inc = std::collections::BTreeMap::<(usize, usize), Vec<f32>>::new();
        let configs: Vec<_> = [64, 128, 256, 512, 1024]
            .into_iter()
            .flat_map(|n| {
                [16, 32, 64, 128].into_iter().flat_map(move |c| {
                    [false, true]
                        .into_iter()
                        .filter(move |q| !q || group == 6)
                        .map(move |q| (n, c, q))
                })
            })
            .collect();
        for (arm, candidate) in [false, true, true, false].into_iter().enumerate() {
            std::thread::sleep(std::time::Duration::from_secs(30));
            thermal(&format!("op-{pos}-{arm}"), "before")?;
            // Clock/cache priming is identical for A and the candidate sweep.
            // Pipeline compilation and first-touch costs stay outside samples.
            let priming = Instant::now();
            while priming.elapsed().as_millis() < 200 {
                one(base.0, base.1, base.2)?;
            }
            for &(nwg, c, qreg) in if candidate {
                configs.as_slice()
            } else {
                std::slice::from_ref(&base)
            } {
                one(nwg, c, qreg)?;
                one(nwg, c, qreg)?;
                let err = error(floats(&out));
                let eq = if qreg {
                    floats(&out)
                        .iter()
                        .zip(&config_inc[&(nwg, c)])
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                } else {
                    config_inc.insert((nwg, c), floats(&out).to_vec());
                    true
                };
                emit(
                    json!({"record":"accuracy","pos":pos,"heads":a.heads,"nwg":nwg,"c":c,"qreg":qreg,"equal_config_inc":eq,"max_abs":err.0,"relative_rms":err.1,"inc_max_abs":inc.0,"inc_relative_rms":inc.1,"max_ratio":err.0/inc.0,"rms_ratio":err.1/inc.1,"pass":err.0<=1.5*inc.0 && err.1<=1.5*inc.1}),
                );
                for sample in 0..8 {
                    let gpu_ms = one(nwg, c, qreg)?;
                    emit(
                        json!({"record":"op_sample","pos":pos,"heads":a.heads,"arm":arm,"candidate":candidate,"nwg":nwg,"c":c,"qreg":qreg,"sample":sample,"gpu_ms":gpu_ms}),
                    );
                }
            }
            thermal(&format!("op-{pos}-{arm}"), "after")?;
        }
    }
    Ok(())
}
fn run() -> Result<()> {
    qwen_cli::shutdown::install()?;
    let a = Args::parse();
    if a.automatic {
        run_args(a)
    } else {
        with_attn_v4_register_q_override(a.qreg, || run_args(a))
    }
}
fn run_args(a: Args) -> Result<()> {
    let rt = Runtime::metal()?;
    let ctx = rt.context();
    emit(
        json!({"record":"source","head":env!("QWEN_BUILD_COMMIT"),"device":ctx.device_facts().name,"mode":a.mode,"depth":a.depth,"qreg":a.qreg,"automatic":a.automatic}),
    );
    if a.mode == "op" {
        return op(ctx, &a);
    }
    let model = rt.load_model(
        a.model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--model required"))?,
    )?;
    ensure!(
        matches!(
            (
                model.arch().n_q_heads,
                model.arch().n_kv_heads,
                model.arch().attn_head_dim
            ),
            (24, 4, 256) | (16, 2, 256)
        ),
        "Qwen census geometry required"
    );
    model.set_prefix_cache_max_bytes(64 << 30);
    let capacity = a.depth + a.tokens.max(2048);
    let prefix = ids(a.depth, model.arch().vocab_size, 1 ^ 0x9e37_79b9_7f4a_7c15);
    let tokens = ids(a.tokens, model.arch().vocab_size, 1);
    let mut request = prefix.clone();
    request.extend_from_slice(&tokens);
    let store = DurableCheckpointStore::new(
        a.cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--cache required"))?,
        64 << 30,
    );
    if a.mode == "prepare" {
        thermal("prepare", "before")?;
        let mut p = prefill_plan::allocate_prefill_request_state(
            &model,
            prefill_plan::PrefillChunkArg::Auto,
            a.depth.max(1),
            capacity,
            true,
        )?;
        let start = Instant::now();
        model.prefill(&mut p.sequence, &mut p.scratch, &prefix)?;
        let checkpoint =
            model.prepare_checkpoint_boundary(&p.sequence, prefix, None, None, None, 0)?;
        let report = model.publish_prepared_checkpoint(&store, &checkpoint, 32 << 30)?;
        emit(
            json!({"record":"prepared","depth":a.depth,"seconds":start.elapsed().as_secs_f64(),"bytes":checkpoint.snapshot_bytes(),"report":format!("{report:?}")}),
        );
        thermal("prepare", "after")?;
        return Ok(());
    }
    if a.depth > 0 {
        let mut seq = model.create_sequence(SequenceConfig::new(capacity))?;
        let restored = model.restore_durable_prefix(&store, &mut seq, &request, 32 << 30)?;
        let hit = restored
            .hit
            .ok_or_else(|| anyhow::anyhow!("saved prefix miss: prepare first"))?;
        ensure!(
            hit.restored_prefix_len == a.depth,
            "wrong saved prefix depth"
        );
        model.cache_durable_restore(&hit)?;
    }
    let restore = || -> Result<Sequence> {
        let mut seq = model.create_sequence(SequenceConfig::new(capacity))?;
        if a.depth > 0 {
            ensure!(
                model.restore_cached_prefix(&mut seq, &request)?.is_some(),
                "RAM prefix miss"
            );
        }
        ensure!(seq.position() == a.depth, "wrong restored position");
        Ok(seq)
    };
    let mut warm = restore()?;
    run_decode(&model, &mut warm, &tokens)?;
    drop(warm);
    if a.mode == "profile" {
        thermal("profile", "before")?;
        let mut seq = restore()?;
        for (token_index, &id) in tokens.iter().take(2).enumerate() {
            let guard = dispatch_profile_begin(ctx, 8192)?;
            let logits = model.decode_token(&mut seq, id)?;
            ensure!(
                logits.iter().all(|x| x.is_finite()),
                "nonfinite profile logits"
            );
            let profile = dispatch_profile_finish(ctx, guard)?;
            ensure!(
                profile.overflow == 0 && profile.rows.iter().all(|r| r.gpu_ns.is_some()),
                "incomplete profile"
            );
            emit(
                json!({"record":"profile","depth":a.depth,"token_index":token_index,"rows":profile.rows}),
            );
        }
        thermal("profile", "after")?;
        return Ok(());
    }
    ensure!(a.mode == "bench", "unknown mode");
    std::thread::sleep(std::time::Duration::from_secs(30));
    thermal("bench", "before")?;
    for sample in 0..a.reps {
        let mut seq = restore()?;
        // Match suite's xorshift stream: every 64-token run consumes a final
        // unused token before the next run selects its first token.
        let skip = (sample + 1) * (a.tokens + 1);
        let sample_ids = ids(skip + a.tokens, model.arch().vocab_size, 1);
        let sample_ids = &sample_ids[skip..];
        let start = Instant::now();
        let logits = run_decode(&model, &mut seq, sample_ids)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite endpoint logits"
        );
        let sha = hash(bytemuck::cast_slice(&logits));
        emit(
            json!({"record":"sample","depth":a.depth,"sample":sample,"tokens":a.tokens,"ms":ms,"tok_s":a.tokens as f64*1000./ms,"logits_sha256":sha}),
        );
    }
    thermal("bench", "after")?;
    Ok(())
}
fn main() -> std::process::ExitCode {
    qwen_cli::shutdown::finish(run())
}
