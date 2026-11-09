//! Whole-model executor/feedback qualification. MODEL DEPTH [TOKENS].
//! Run under the Metal lease. Snapshots and full logits stay in RAM.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    sampling::{Sampler, SamplingConfig},
};
use serde_json::json;
use std::time::Instant;
fn ids(n: usize, v: u32, mut s: u64) -> Vec<u32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % u64::from(v)) as u32
        })
        .collect()
}
fn distribution(a: &[f32], b: &[f32], target: usize) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let logz = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v
            .iter()
            .map(|&x| (f64::from(x) - m).exp())
            .sum::<f64>()
            .ln()
    };
    let (la, lb) = (logz(a), logz(b));
    let kl = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - la).exp() * (f64::from(x) - la - f64::from(y) + lb))
        .sum::<f64>();
    let dnll = lb - f64::from(b[target]) - la + f64::from(a[target]);
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0., f64::max);
    json!({"dnll":dnll,"kl":kl,"max_abs":max_abs})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args[2].parse()?;
    let count: usize = args.get(3).map(|x| x.parse()).transpose()?.unwrap_or(64);
    assert!(count >= 4);
    let ctx = MetalContext::new()?;
    let gguf = GgufFile::open(&args[1])?;
    let config = Qwen4ExpConfig::from_gguf(&gguf)?;
    let limit = depth + count + 192;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&config, limit)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, cap, limit)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(false);
    runner.prefill(&ids(64, config.vocab_size, 1))?;
    runner.reset()?;
    runner.prefill(&ids(depth, config.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, config.vocab_size, 7) {
        runner.forward_token(t)?;
    }
    let start = runner.next_position();
    let entry = runner.capture_snapshot()?;
    let entry_logits = runner.logits()?.to_vec();
    // Import PLE banks and compile feedback pipelines before steady-state arms.
    let cold = runner.generate_gpu_tokens(
        &mut Sampler::new(SamplingConfig::default())?,
        2,
        &[],
        |_| true,
    )?;
    println!(
        "{}",
        json!({"kind":"gpu_loop_cold","depth":depth,"wall_ms":cold.wall_ms,"first_token_ms":cold.first_token_ready_ms})
    );
    for temperature in [0., 0.8] {
        let sampling = SamplingConfig {
            temperature,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.02,
            seed: 19,
        };
        for rep in 0..3 {
            let arms = if rep % 2 == 0 {
                [(false, false), (true, false), (true, true), (false, true)]
            } else {
                [(false, true), (true, true), (true, false), (false, false)]
            };
            for (causal, gpu) in arms {
                runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
                runner.configure_causal_executor(causal);
                runner.generate_gpu_tokens(&mut Sampler::new(sampling)?, 2, &[], |_| true)?;
                runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
                let mut sampler = Sampler::new(sampling)?;
                let now = Instant::now();
                let (wall_ms, gpu_ms, encode_ms, wait_ms, first_ms) = if gpu {
                    let stats = runner.generate_gpu_tokens(&mut sampler, count, &[], |_| true)?;
                    assert_eq!(stats.forwarded_tokens, count - 1);
                    (
                        stats.wall_ms,
                        stats.gpu_ms,
                        stats.encode_cpu_ms,
                        stats.completion_wait_ms,
                        stats.first_token_ready_ms.unwrap(),
                    )
                } else {
                    let first = sampler.sample(runner.logits()?.as_slice())?.token;
                    let first_ms = now.elapsed().as_secs_f64() * 1e3;
                    let mut token = first;
                    let (mut gpu_ms, mut encode_ms, mut wait_ms) = (0., 0., 0.);
                    for _ in 1..count {
                        token = sampler
                            .sample(runner.forward_token(token as u32)?.as_slice())?
                            .token;
                        let t = runner.last_token_timing().unwrap();
                        gpu_ms += t.gpu_ms.unwrap_or(0.);
                        encode_ms += t.encode_cpu_ms;
                        wait_ms += t.completion_wait_ms;
                    }
                    (
                        now.elapsed().as_secs_f64() * 1e3,
                        gpu_ms,
                        encode_ms,
                        wait_ms,
                        first_ms,
                    )
                };
                assert_eq!(runner.next_position(), start + count - 1);
                println!(
                    "{}",
                    json!({"kind":"whole_model","depth":depth,"temperature":temperature,"rep":rep,"causal":causal,"gpu_loop":gpu,"tokens":count,"transitions":count-1,"wall_ms":wall_ms,"transition_tps":(count-1) as f64*1e3/(wall_ms-first_ms),"gpu_ms":gpu_ms,"encode_ms":encode_ms,"wait_ms":wait_ms,"first_ms":first_ms,"sampler_version":sampler.algorithm_version()})
                );
            }
        }
    }
    // GPU autoregression supplies teacher inputs; replay exactly those inputs
    // on the incumbent twice to establish its repeat floor, then the executor.
    for causal in [false, true] {
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        runner.configure_causal_executor(causal);
        let sampling = SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.02,
            seed: 29,
        };
        let mut selected = Vec::new();
        let mut observed = Vec::new();
        runner.generate_gpu_tokens_observed(
            &mut Sampler::new(sampling)?,
            count,
            &[],
            |t, row| {
                selected.push(t);
                observed.push(row.to_vec());
                true
            },
        )?;
        let mut baseline = Vec::new();
        for replay in 0..3 {
            runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
            runner.configure_causal_executor(replay == 2);
            let mut rows = vec![runner.logits()?.to_vec()];
            let (mut wall_ms, mut gpu_ms, mut encode_ms) = (0., 0., 0.);
            for &t in &selected[..count - 1] {
                let now = Instant::now();
                rows.push(runner.forward_token(t as u32)?.to_vec());
                wall_ms += now.elapsed().as_secs_f64() * 1e3;
                let timing = runner.last_token_timing().unwrap();
                gpu_ms += timing.gpu_ms.unwrap_or(0.);
                encode_ms += timing.encode_cpu_ms;
            }
            println!(
                "{}",
                json!({"kind":"teacher_forced_timing","depth":depth,"gpu_causal":causal,"replay":replay,"causal":replay==2,"forwards":count-1,"wall_ms":wall_ms,"gpu_ms":gpu_ms,"encode_ms":encode_ms,"tps":(count-1) as f64*1e3/wall_ms})
            );
            if replay == 0 {
                baseline = rows;
                continue;
            }
            let results: Vec<_> = baseline
                .iter()
                .zip(&rows)
                .zip(&selected)
                .map(|((a, b), &t)| distribution(a, b, t as usize))
                .collect();
            println!(
                "{}",
                json!({"kind":"distribution","depth":depth,"gpu_causal":causal,"candidate":if replay==1{"incumbent_repeat"}else{"executor"},"rows":results})
            );
        }
        let results: Vec<_> = baseline
            .iter()
            .zip(&observed)
            .zip(&selected)
            .map(|((a, b), &t)| distribution(a, b, t as usize))
            .collect();
        println!(
            "{}",
            json!({"kind":"distribution","depth":depth,"gpu_causal":causal,"candidate":"gpu_loop","rows":results})
        );
        // A resumed ordinary step checks history, recurrence and QSA frontiers.
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        runner.configure_causal_executor(causal);
        let stats =
            runner.generate_gpu_tokens(&mut Sampler::new(sampling)?, count, &[], |_| true)?;
        let resumed = runner
            .forward_token(stats.terminal_token.unwrap() as u32)?
            .to_vec();
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        runner.configure_causal_executor(false);
        for &t in &selected {
            runner.forward_token(t as u32)?;
        }
        println!(
            "{}",
            json!({"kind":"continuation","depth":depth,"causal":causal,"comparison":distribution(runner.logits()?.as_slice(),&resumed,selected[0] as usize)})
        );
    }
    // A discarded pre-encoded frame must leave the completed state at entry.
    for (limit, eos, accept) in [(8, vec![], false), (1, vec![], true)] {
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let stats = runner.generate_gpu_tokens(
            &mut Sampler::new(SamplingConfig::default())?,
            limit,
            &eos,
            |_| accept,
        )?;
        assert_eq!(runner.next_position(), start);
        assert_eq!(stats.forwarded_tokens, 0);
        assert!(!runner.is_poisoned());
        runner.capture_snapshot()?;
    }
    runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
    let token = Sampler::new(SamplingConfig::default())?
        .sample(runner.logits()?.as_slice())?
        .token;
    let stats = runner.generate_gpu_tokens(
        &mut Sampler::new(SamplingConfig::default())?,
        8,
        &[token],
        |_| panic!("EOS reached callback"),
    )?;
    assert!(stats.stopped_on_eos);
    assert_eq!(runner.next_position(), start);
    println!(
        "{}",
        json!({"kind":"stop_contracts","depth":depth,"passed":true})
    );
    Ok(())
}
