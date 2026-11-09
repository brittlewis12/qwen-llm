//! Whole-model A/B: MODEL DEPTH [STEPS]. Run under the canonical Metal lease.
//! Snapshots and logits remain in RAM; output contains compact metrics only.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
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
fn compare(reference: &[f32], candidate: &[f32], target: usize) -> (f64, f64, f64) {
    let max = |v: &[f32]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let (a, b) = (max(reference), max(candidate));
    let (za, zb) = (
        reference
            .iter()
            .map(|&x| (f64::from(x) - a).exp())
            .sum::<f64>(),
        candidate
            .iter()
            .map(|&x| (f64::from(x) - b).exp())
            .sum::<f64>(),
    );
    let (la, lb) = (a + za.ln(), b + zb.ln());
    let dnll = lb - f64::from(candidate[target]) - la + f64::from(reference[target]);
    let kl = reference
        .iter()
        .zip(candidate)
        .map(|(&x, &y)| (f64::from(x) - la).exp() * (f64::from(x) - la - f64::from(y) + lb))
        .sum();
    let max_abs = qwen_llm::compare::report_max_abs_diff_f64(reference, candidate);
    assert!(max_abs.is_finite(), "invalid diagnostic comparison");
    (dnll, kl, max_abs)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[2].parse()?;
    let steps: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let ctx = MetalContext::new()?;
    let gguf = GgufFile::open(&a[1])?;
    let config = Qwen4ExpConfig::from_gguf(&gguf)?;
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, depth + steps + 128)?;
    let mut model =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, depth + steps + 128)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(false);
    runner.prefill(&ids(64, config.vocab_size, 1))?;
    runner.reset()?;
    runner.prefill(&ids(depth, config.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for token in ids(64, config.vocab_size, 7) {
        runner.forward_token(token)?;
    }
    let entry = runner.capture_snapshot()?;
    let targets = ids(steps + 1, config.vocab_size, 13);
    let mut reference = Vec::<Vec<f32>>::new();
    for rep in 0..4 {
        let order = if rep % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for causal in order {
            runner.restore_snapshot(&entry)?;
            runner.configure_causal_executor(causal);
            // Excluded warm command compiles reflection and establishes each arm.
            qwen_llm::metal::dispatch_census_begin();
            runner.forward_token(targets[0])?;
            let census = qwen_llm::metal::dispatch_census_take();
            let fused = census
                .iter()
                .filter(|r| r.kernel == "kernel_qwen4exp_ffn_epilogue_f32")
                .count();
            assert_eq!(
                fused,
                if causal {
                    config.layer_count as usize - 2
                } else {
                    0
                },
                "executor dispatch witness"
            );
            assert_eq!(census.iter().any(|r| r.encoder_concurrent), causal);
            println!(
                "{}",
                json!({"kind":"executor_witness","depth":depth,"rep":rep,"causal":causal,"fused":fused,"dispatches":census.len()})
            );
            runner.restore_snapshot(&entry)?;
            let mut logits = Vec::with_capacity(steps);
            let mut samples = Vec::new();
            for &token in &targets[..steps] {
                let start = Instant::now();
                let row = runner.forward_token(token)?.to_vec();
                let wall = start.elapsed().as_secs_f64() * 1e3;
                assert!(row.iter().all(|v| v.is_finite()));
                let t = runner.last_token_timing().unwrap();
                samples.push(json!({"wall_ms":wall,"gpu_ms":t.gpu_ms,"encode_ms":t.encode_cpu_ms,"wait_ms":t.completion_wait_ms}));
                logits.push(row);
            }
            if reference.is_empty() {
                assert!(!causal);
                reference = logits.clone();
            }
            let comparison: Vec<_> = reference
                .iter()
                .zip(&logits)
                .enumerate()
                .map(|(i, (a, b))| {
                    let (dnll, kl, max_abs) = compare(a, b, targets[i + 1] as usize);
                    json!({"dnll":dnll,"kl":kl,"max_abs":max_abs})
                })
                .collect();
            println!(
                "{}",
                json!({"kind":"teacher_forced","depth":depth,"rep":rep,"causal":causal,"samples":samples,"distribution":comparison})
            );
        }
    }
    Ok(())
}
