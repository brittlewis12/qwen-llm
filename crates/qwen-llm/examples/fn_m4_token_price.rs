//! Output-preserving whole-token stage counterfactuals. MODEL DEPTH [REPS=4].
//! Requires --features perf-ablations and the canonical Metal lease.
#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use qwen_llm::{
        gguf::GgufFile,
        metal::*,
        qwen4exp::Qwen4ExpConfig,
        qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
        sampling::{Sampler, SamplingConfig},
    };
    use serde_json::json;
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
    fn conditions() -> serde_json::Value {
        let read = |name: &str, args: &[&str]| {
            String::from_utf8_lossy(
                &std::process::Command::new(name)
                    .args(args)
                    .output()
                    .unwrap()
                    .stdout,
            )
            .trim()
            .to_string()
        };
        json!({"thermal":read("notifyutil",&["-g","com.apple.system.thermalpressurelevel"]),"rustc":read("pgrep",&["-x","rustc"])})
    }
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[2].parse()?;
    let reps: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let experts = a.iter().any(|s| s == "--experts");
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let limit = depth + 512;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, limit)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, limit)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(true);
    runner.prefill(&ids(64, c.vocab_size, 1))?;
    runner.reset()?;
    runner.prefill(&ids(depth, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, c.vocab_size, 7) {
        runner.forward_token(t)?;
    }
    let entry = runner.capture_snapshot()?;
    let entry_logits = runner.logits()?.to_vec();
    let tokens = ids(4, c.vocab_size, 13);
    // Establish the actual resident feedback baseline independently of fixed-token substitutions.
    let before = conditions();
    let stats = runner.generate_gpu_tokens(
        &mut Sampler::new(SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.02,
            seed: 19,
        })?,
        64,
        &[],
        |_| true,
    )?;
    println!(
        "{}",
        json!({"kind":"gpu_loop_baseline","depth":depth,"before":before,"after":conditions(),"token_ms":(stats.wall_ms-stats.first_token_ready_ms.unwrap())/stats.forwarded_tokens as f64,"gpu_ms":(stats.gpu_ms-stats.first_token_selection_ms)/stats.forwarded_tokens as f64,"tokens":stats.sampled_tokens})
    );
    if a.iter().any(|s| s == "--q6-ab") {
        let pairs: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(12);
        let teacher = ids(64, c.vocab_size, 21);
        let greedy = || SamplingConfig {
            temperature: 0.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.02,
            seed: 19,
        };
        // Teacher-forced full-logit comparison on identical restored state.
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let mut incumbent = Vec::with_capacity(teacher.len());
        {
            let _g = qwen_llm::metal::diagnostic_q6k_r1(1);
            for &t in &teacher {
                incumbent.push(runner.forward_token(t)?.to_vec());
            }
        }
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let mut candidate = Vec::with_capacity(teacher.len());
        {
            let _g = qwen_llm::metal::diagnostic_q6k_r1(2);
            for &t in &teacher {
                candidate.push(runner.forward_token(t)?.to_vec());
            }
        }
        let (mut dnll, mut kl, mut max_abs, mut exact) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for (i, (x, y)) in incumbent
            .iter()
            .zip(&candidate)
            .take(teacher.len() - 1)
            .enumerate()
        {
            let equal = bytemuck::cast_slice::<f32, u8>(x) == bytemuck::cast_slice::<f32, u8>(y);
            exact += usize::from(equal);
            let target = teacher[i + 1] as usize;
            let lz = |v: &[f32]| {
                let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                m + v
                    .iter()
                    .map(|&q| (f64::from(q) - m).exp())
                    .sum::<f64>()
                    .ln()
            };
            let (zx, zy) = (lz(x), lz(y));
            dnll += (zy - f64::from(y[target])) - (zx - f64::from(x[target]));
            kl += x
                .iter()
                .zip(y)
                .map(|(&p, &q)| {
                    (f64::from(p) - zx).exp() * ((f64::from(p) - zx) - (f64::from(q) - zy))
                })
                .sum::<f64>();
            for (&p, &q) in x.iter().zip(y) {
                max_abs = max_abs.max((f64::from(p) - f64::from(q)).abs());
            }
        }
        let n = teacher.len() as f64;
        println!(
            "{}",
            json!({"kind":"q6_teacher","targets":teacher.len(),"signed_dNLL":dnll/(n-1.),"mean_KL":kl/(n-1.),"bitwise_rows":exact,"max_abs":max_abs})
        );
        for p in 0..pairs {
            let order: [(&str, u8); 2] = if p % 2 == 0 {
                [("incumbent", 1), ("two_row", 2)]
            } else {
                [("two_row", 2), ("incumbent", 1)]
            };
            for (name, mode) in order {
                let before = conditions();
                let _g = qwen_llm::metal::diagnostic_q6k_r1(mode);
                runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
                runner.generate_gpu_tokens(&mut Sampler::new(greedy())?, 2, &[], |_| true)?;
                runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
                let stats =
                    runner.generate_gpu_tokens(&mut Sampler::new(greedy())?, 16, &[], |_| true)?;
                let fwd = stats.forwarded_tokens as f64;
                println!(
                    "{}",
                    json!({"kind":"q6_pair","pair":p,"arm":name,"mode":mode,"wall_ms_per_token":stats.transition_ms/fwd,"gpu_ms_per_token":(stats.gpu_ms-stats.first_token_selection_ms)/fwd,"before":before,"after":conditions()})
                );
            }
        }
        return Ok(());
    }
    for phase in [0usize] {
        runner.restore_snapshot(&entry)?;
        for &t in &tokens[..phase] {
            runner.forward_token(t)?;
        }
        let snap = runner.capture_snapshot()?;
        let book = TokenPriceBook::default();
        let capture = book.scope(TokenPriceMode::Capture);
        let reference = runner.forward_token(tokens[phase])?.to_vec();
        capture.finish();
        println!(
            "{}",
            json!({"kind":"metadata","depth":depth,"phase":phase,"records":book.metadata()})
        );
        // Immediate controls: no idle gap or process-spawn between warm and timed steps.
        for rep in 0..reps {
            let mut masks = if experts {
                vec![2u32, 8192, 16384, 32768, 65536, 131072, 262144]
            } else {
                vec![1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            };
            if rep % 2 != 0 {
                masks.reverse();
            }
            for mask in masks {
                std::thread::sleep(std::time::Duration::from_millis(200));
                let before = conditions();
                let mut modes = if experts || mask == 64 {
                    vec![0, 1, 2]
                } else {
                    vec![0, 1]
                };
                if rep % 2 != 0 {
                    modes.reverse();
                }
                for mode in modes {
                    runner.restore_snapshot(&snap)?;
                    runner.forward_token(tokens[phase])?;
                    runner.restore_snapshot(&snap)?;
                    let guard = book.scope(match mode {
                        0 => TokenPriceMode::Hold(0),
                        1 => TokenPriceMode::Hold(mask),
                        _ => TokenPriceMode::Stream(mask),
                    });
                    let actual = runner.forward_token(tokens[phase])?.to_vec();
                    guard.finish();
                    let timing = runner.last_token_timing().unwrap();
                    let exact = reference
                        .iter()
                        .zip(&actual)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        exact,
                        "counterfactual changed full head mask {mask}, mode{mode}"
                    );
                    println!(
                        "{}",
                        json!({"kind":"price","depth":depth,"rep":rep,"group":mask,"mode":mode,"candidate":mode!=0,"before":before,"gpu_ms":timing.gpu_ms,"wall_ms":timing.total_wall_ms,"encode_ms":timing.encode_cpu_ms,"exact":exact})
                    );
                }
                println!(
                    "{}",
                    json!({"kind":"conditions_after","rep":rep,"group":mask,"after":conditions()})
                );
            }
        }
        if a.iter().any(|s| s == "--profile") {
            runner.restore_snapshot(&snap)?;
            let guard = qwen_llm::metal::dispatch_profile_begin(&ctx, 10_000)?;
            runner.forward_token(tokens[phase])?;
            let report = qwen_llm::metal::dispatch_profile_finish(&ctx, guard)?;
            let mut kernels = std::collections::BTreeMap::<String, (usize, u64)>::new();
            for row in report.rows {
                let e = kernels.entry(row.kernel).or_default();
                e.0 += 1;
                e.1 += row.gpu_ns.unwrap_or(0);
            }
            println!(
                "{}",
                json!({"kind":"dispatch_census","depth":depth,"phase":phase,"overflow":report.overflow,"kernels":kernels})
            );
        }
        // Isolated copy tax, same native wave barriers: no inference result claimed.
        for rep in 0..3 {
            for mask in [
                0u32, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
            ] {
                runner.restore_snapshot(&snap)?;
                runner.forward_token(tokens[phase])?;
                runner.restore_snapshot(&snap)?;
                let gpu = book.copy_tax(&ctx, mask)?;
                println!(
                    "{}",
                    json!({"kind":"copy_tax","rep":rep,"mask":mask,"gpu_ms":gpu})
                );
            }
        }
    }
    Ok(())
}
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}
