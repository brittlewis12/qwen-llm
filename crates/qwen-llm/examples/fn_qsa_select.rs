//! Whole-token singleton QSA selector A/B through the production GPU token loop.
//! MODEL DEPTH [TOKENS=128] [REPS=4]. Mode 0 forces the existing decode selector,
//! mode 1 the wide selector; arms alternate order by repetition.
#[cfg(feature = "perf-ablations")]
mod probe {
    use anyhow::{Result, ensure};
    use qwen_llm::{
        gguf::GgufFile,
        metal::*,
        qwen4exp::Qwen4ExpConfig,
        qwen4exp_qsa::diagnostic_wide_decode_selector,
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
                (s % v as u64) as u32
            })
            .collect()
    }
    fn config() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.02,
            seed: 19,
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

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let depth: usize = args[2].parse()?;
        let count: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(128);
        let reps: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4);
        let modes: Vec<u8> = vec![1];
        let select = |mode: u8| diagnostic_wide_decode_selector(if mode == 0 { 1 } else { 2 });
        let ctx = MetalContext::new()?;
        let gguf = GgufFile::open(&args[1])?;
        let c = Qwen4ExpConfig::from_gguf(&gguf)?;
        let limit = depth + count + 256;
        let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, limit)?;
        let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, cap, limit)?;
        let mut runner = model.create_runner(&ctx)?;
        runner.configure_causal_executor(true);
        runner.prefill(&ids(64, c.vocab_size, 1))?;
        runner.reset()?;
        runner.prefill(&ids(depth, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
        for token in ids(64, c.vocab_size, 7) {
            runner.forward_token(token)?;
        }
        let entry = runner.capture_snapshot()?;
        let head = runner.logits()?.to_vec();
        for mode in std::iter::once(0).chain(modes.iter().copied()) {
            runner.restore_snapshot_with_logits(&entry, &head)?;
            let _select = select(mode);
            let warm =
                runner.generate_gpu_tokens(&mut Sampler::new(config())?, 16, &[], |_| true)?;
            println!(
                "{}",
                json!({"kind":"warm","depth":depth,"mode":mode,"wall_ms":warm.wall_ms,"first_ms":warm.first_token_ready_ms})
            );
        }
        for rep in 0..reps {
            let mut order = modes.clone();
            if rep % 2 != 0 {
                order.reverse();
            }
            for candidate in order {
                for mode in if rep % 2 == 0 {
                    [0, candidate]
                } else {
                    [candidate, 0]
                } {
                    runner.restore_snapshot_with_logits(&entry, &head)?;
                    let _select = select(mode);
                    runner.generate_gpu_tokens(&mut Sampler::new(config())?, 2, &[], |_| true)?;
                    runner.restore_snapshot_with_logits(&entry, &head)?;
                    let (stop, monitor) = arm_monitor();
                    let start = Instant::now();
                    let stats = runner.generate_gpu_tokens(
                        &mut Sampler::new(config())?,
                        count,
                        &[],
                        |_| true,
                    )?;
                    let wall = start.elapsed().as_secs_f64() * 1e3;
                    stop.send(()).unwrap();
                    let observations = monitor.join().unwrap();
                    println!(
                        "{}",
                        json!({"kind":"timing","depth":depth,"candidate":candidate,"rep":rep,"mode":mode,"tokens":count,"whole_tps":count as f64*1000./wall,"transition_tps":(count-1)as f64*1000./(wall-stats.first_token_ready_ms.unwrap()),"wall_ms":wall,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,"wait_ms":stats.completion_wait_ms,"first_ms":stats.first_token_ready_ms,"conditions":observations})
                    );
                }
            }
        }
        // One identical forced sequence, incumbent repeat establishes numeric noise.
        let targets = ids(16, c.vocab_size, 73);
        let mut reference = Vec::new();
        for mode in std::iter::once(0)
            .chain(std::iter::once(0))
            .chain(modes.iter().copied())
        {
            runner.restore_snapshot_with_logits(&entry, &head)?;
            let _select = select(mode);
            let mut rows = vec![head.clone()];
            for &token in targets.iter().take(targets.len() - 1) {
                rows.push(runner.forward_token(token)?.to_vec());
            }
            if reference.is_empty() {
                reference = rows;
                continue;
            }
            println!(
                "{}",
                json!({"kind":if mode==0{"noise"}else{"accuracy"},"depth":depth,"mode":mode,"rows":rows.iter().zip(&reference).zip(&targets).map(|((a,b),&t)|check(b,a,t as usize)).collect::<Vec<_>>()})
            );
        }
        ensure!(!runner.is_poisoned());
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations")
}
