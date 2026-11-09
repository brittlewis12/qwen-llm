//! GPU-loop A/B and handoff: MODEL DEPTH [TOKENS=128]. Canonical lease required.
use anyhow::{Result, ensure};
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::*,
    metal::*,
    sampling::{Sampler, SamplingConfig},
};
use serde_json::json;
use std::time::Instant;
fn config(temperature: f32, seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.02,
        seed,
    }
}
fn check(a: &[f32], b: &[f32], target: usize) -> serde_json::Value {
    let max = qwen_llm::compare::report_max_abs_diff_f64(a, b);
    ensure_finite(a);
    ensure_finite(b);
    let z = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (z(a), z(b));
    let dnll = lb - b[target] as f64 - la + a[target] as f64;
    let kl = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - la).exp() * (x as f64 - la - y as f64 + lb))
        .sum::<f64>();
    json!({"dnll":dnll,"kl":kl,"max_abs":max})
}
fn ensure_finite(v: &[f32]) {
    assert!(v.iter().all(|x| x.is_finite()));
}
fn conditions() -> serde_json::Value {
    let read = |cmd: &str, args: &[&str]| {
        String::from_utf8_lossy(
            &std::process::Command::new(cmd)
                .args(args)
                .output()
                .unwrap()
                .stdout,
        )
        .trim()
        .to_string()
    };
    json!({"thermal":read("notifyutil", &["-g","com.apple.system.thermalpressurelevel"]),
        "rustc":read("pgrep", &["-x","rustc"])})
}
fn arm_monitor() -> (
    std::sync::mpsc::Sender<()>,
    std::thread::JoinHandle<Vec<serde_json::Value>>,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    let (ready, wait) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let mut rows = vec![conditions()];
        ready.send(()).unwrap();
        while rx
            .recv_timeout(std::time::Duration::from_millis(250))
            .is_err()
        {
            rows.push(conditions());
        }
        rows.push(conditions());
        rows
    });
    wait.recv().unwrap();
    (tx, thread)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[2].parse()?;
    let n: usize = a.get(3).map(|v| v.parse()).transpose()?.unwrap_or(128);
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&a[1])?;
    let w = Glm5NextWeights::load(&ctx, &g)?;
    let f: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../scripts/reference/glm53/quality-v1.json"
    ))?;
    let ids: Vec<u32> = f["documents"][0]["tokens"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let prompt: Vec<_> = ids.iter().copied().cycle().take(depth).collect();
    let cap = depth + n + 1024;
    let mut s = Glm5NextSession::with_prefill_rows(&ctx, &w, cap, 2048)?;
    let mut head = s.prefill_packed(&ctx, &prompt)?;
    for &t in ids.iter().take(8) {
        head = s.forward(&ctx, t)?;
    }
    let entry = s.capture_snapshot()?;
    for temp in [0., 0.8] {
        s.generate_gpu_tokens(
            &ctx,
            &head,
            &mut Sampler::new(config(temp, 19))?,
            2,
            &[],
            |_| true,
        )?;
        s.restore_snapshot(&entry)?;
        for rep in 0..3 {
            for gpu in if rep % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                for attempt in 0..3 {
                    s.restore_snapshot(&entry)?;
                    let (stop_monitor, monitor) = arm_monitor();
                    let mut sampler = Sampler::new(config(temp, 19))?;
                    let rustc_before = std::process::Command::new("pgrep")
                        .args(["-x", "rustc"])
                        .output()?
                        .stdout;
                    let started_unix = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs_f64();
                    let started = Instant::now();
                    let (first, tokens, forwarded) = if gpu {
                        let st =
                            s.generate_gpu_tokens(&ctx, &head, &mut sampler, n, &[], |_| true)?;
                        (
                            st.first_token_ready_ms.unwrap(),
                            st.sampled_tokens,
                            st.forwarded_tokens,
                        )
                    } else {
                        let mut token = sampler.sample(&head)?.token;
                        let first = started.elapsed().as_secs_f64() * 1e3;
                        for _ in 1..n {
                            token = sampler.sample(&s.forward(&ctx, token as u32)?)?.token;
                        }
                        (first, n, n - 1)
                    };
                    let wall = started.elapsed().as_secs_f64() * 1e3;
                    stop_monitor.send(()).unwrap();
                    let observations = monitor.join().unwrap();
                    let pressure2 = observations
                        .iter()
                        .any(|r| r["thermal"].as_str().unwrap().ends_with(" 2"));
                    println!(
                        "{}",
                        json!({"attempt":attempt,"pressure2":pressure2,"conditions":observations,"kind":"timing","depth":depth,"temperature":temp,"rep":rep,"gpu":gpu,"wall_ms":wall,"first_ms":first,"tokens":tokens,"tps":forwarded as f64*1e3/(wall-first),"started_unix":started_unix,"finished_unix":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64(),"rustc_before":String::from_utf8_lossy(&rustc_before),"rustc_after":String::from_utf8_lossy(&std::process::Command::new("pgrep").args(["-x","rustc"]).output()?.stdout)})
                    );
                    if !pressure2 {
                        break;
                    }
                    ensure!(attempt < 2, "pressure2 on all three attempts");
                    // Only rejected level-2 arms wait, and only until level 1.
                    let cooling = Instant::now();
                    while conditions()["thermal"].as_str().unwrap().ends_with(" 2") {
                        ensure!(
                            cooling.elapsed().as_secs() < 180,
                            "pressure2 did not subside"
                        );
                        std::thread::sleep(std::time::Duration::from_secs(5));
                    }
                }
            }
        }
    }
    s.restore_snapshot(&entry)?;
    let mut tokens = Vec::new();
    let mut candidate = Vec::new();
    s.generate_gpu_tokens_observed(
        &ctx,
        &head,
        &mut Sampler::new(config(0.8, 29))?,
        32,
        &[],
        |token, row| {
            tokens.push(token);
            candidate.push(row.to_vec());
            true
        },
    )?;
    let mut reference = Vec::new();
    for repeat in 0..2 {
        s.restore_snapshot(&entry)?;
        let mut rows = vec![head.clone()];
        for &token in tokens.iter().take(tokens.len() - 1) {
            rows.push(s.forward(&ctx, token as u32)?);
        }
        if repeat == 0 {
            reference = rows;
        } else {
            println!(
                "{}",
                json!({"kind":"noise","rows":rows.iter().zip(&reference).zip(&tokens).map(|((a,b),&t)|check(b,a,t as usize)).collect::<Vec<_>>()})
            );
        }
    }
    println!(
        "{}",
        json!({"kind":"accuracy","rows":candidate.iter().zip(&reference).zip(&tokens).map(|((a,b),&t)|check(b,a,t as usize)).collect::<Vec<_>>()})
    );
    s.restore_snapshot(&entry)?;
    let stop = s.generate_gpu_tokens(
        &ctx,
        &head,
        &mut Sampler::new(config(0., 19))?,
        8,
        &[],
        |_| false,
    )?;
    ensure!(stop.sampled_tokens == 1 && stop.forwarded_tokens == 0);
    ensure!(s.position() == depth + 8);
    // Distinct histories after GPU warm-up; native membership changes widths.
    let mut states = vec![s];
    for _ in 1..4 {
        let mut peer = Glm5NextSession::with_prefill_rows(&ctx, &w, cap, 4)?;
        peer.restore_snapshot(&entry)?;
        states.push(peer);
    }
    for (i, state) in states.iter_mut().enumerate() {
        state.restore_snapshot(&entry)?;
        state.generate_gpu_tokens(
            &ctx,
            &head,
            &mut Sampler::new(config(0.8, 37 + i as u64))?,
            3,
            &[],
            |_| true,
        )?;
    }
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    for width in [2, 3, 4] {
        for step in 0..8 {
            let inputs: Vec<_> = (0..width)
                .map(|i| ids[(i * 11 + step) % ids.len()])
                .collect();
            batch.step(
                &ctx,
                &mut states[..width].iter_mut().collect::<Vec<_>>(),
                &inputs,
            )?;
        }
    }
    let snap = states[0].capture_snapshot()?;
    states[1].restore_snapshot(&snap)?;
    let mut last = Vec::new();
    for step in 0..32 {
        let t = ids[step % ids.len()];
        let mut pair: Vec<_> = states[..2].iter_mut().collect();
        if step % 2 != 0 {
            pair.reverse();
        }
        let rows = batch.step(&ctx, &mut pair, &[t, t])?;
        ensure!(rows[0] == rows[1]);
        last = rows[0].clone();
    }
    let mut selected = Vec::new();
    let mut seen = Vec::new();
    states[0].generate_gpu_tokens_observed(
        &ctx,
        &last,
        &mut Sampler::new(config(0., 19))?,
        4,
        &[],
        |token, row| {
            selected.push(token);
            seen.push(row.to_vec());
            true
        },
    )?;
    for (i, row) in seen.iter().enumerate() {
        if i > 0 {
            last = states[1].forward(&ctx, selected[i - 1] as u32)?;
        }
        ensure!(qwen_llm::compare::report_max_abs_diff_f64(row, &last) == 0.);
    }
    println!(
        "{}",
        json!({"kind":"interop","widths":[2,3,4],"fork_exact":32,"continuation_error":0,"stop_passed":true})
    );
    Ok(())
}
