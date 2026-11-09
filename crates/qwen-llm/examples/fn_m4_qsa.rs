//! Lean whole-token QSA pricing and selector A/B. MODEL DEPTH [TOKENS].
//! Build with perf-ablations; run under the canonical Metal lease.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_qsa::diagnostic_decode_price,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    sampling::{Sampler, SamplingConfig},
};
use serde_json::json;
use std::{io::Write, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
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
fn main() -> anyhow::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[2].parse()?;
    let count: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let limit = depth + count + 192;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, limit)?;
    let mut m = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, limit)?;
    let mut r = m.create_runner(&ctx)?;
    r.configure_causal_executor(true);
    r.prefill(&ids(64, c.vocab_size, 1))?;
    r.reset()?;
    r.prefill(&ids(depth, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, c.vocab_size, 7) {
        r.forward_token(t)?;
    }
    let entry = r.capture_snapshot()?;
    let entry_logits = r.logits()?.to_vec();
    let token = ids(1, c.vocab_size, 13)[0];
    let expected = {
        let _guard = diagnostic_decode_price(4);
        r.forward_token(token)?.to_vec()
    };
    for rep in 0..3 {
        let modes = if rep % 2 == 0 {
            [4, 1, 3, 2]
        } else {
            [2, 3, 1, 4]
        };
        for mode in modes {
            let mut wall = 0.;
            let mut gpu = 0.;
            for _ in 0..16 {
                r.restore_snapshot_with_logits(&entry, &entry_logits)?;
                {
                    let _native = diagnostic_decode_price(4);
                    r.forward_token(token)?;
                }
                r.restore_snapshot_with_logits(&entry, &entry_logits)?;
                let _guard = diagnostic_decode_price(mode);
                let t = Instant::now();
                let row = r.forward_token(token)?;
                wall += t.elapsed().as_secs_f64() * 1e3;
                anyhow::ensure!(
                    row.as_slice()
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "pricing substitute changes native logits: mode={mode}"
                );
                gpu += r.last_token_timing().unwrap().gpu_ms.unwrap();
            }
            emit(
                json!({"kind":"price","depth":depth,"rep":rep,"mode":mode,"wall_ms":wall/16.,"gpu_ms":gpu/16.,"native_outputs_exact":true}),
            );
        }
    }
    // Real autoregression: the candidate performs every selection, no cached substitution.
    for rep in 0..4 {
        for mode in if rep % 2 == 0 { [4, 2] } else { [2, 4] } {
            r.restore_snapshot_with_logits(&entry, &entry_logits)?;
            let _guard = diagnostic_decode_price(mode);
            r.generate_gpu_tokens(
                &mut Sampler::new(SamplingConfig::default())?,
                2,
                &[],
                |_| true,
            )?;
            r.restore_snapshot_with_logits(&entry, &entry_logits)?;
            let t = r.generate_gpu_tokens(
                &mut Sampler::new(SamplingConfig::default())?,
                count,
                &[],
                |_| true,
            )?;
            emit(
                json!({"kind":"selector_ab","depth":depth,"rep":rep,"mode":mode,"wall_ms":t.wall_ms,"first_ms":t.first_token_ready_ms,"transitions":count-1,"gpu_ms":t.gpu_ms,"tps":(count-1)as f64*1e3/(t.wall_ms-t.first_token_ready_ms.unwrap())}),
            );
        }
    }
    Ok(())
}
