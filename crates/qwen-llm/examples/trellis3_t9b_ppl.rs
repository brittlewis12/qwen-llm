//! T9b teacher-forced perplexity evaluator
//! (docs/bench/2026-07-20-trellis3-t9b-e2e-arbiter/).
//!
//! Frozen protocol: tokenize the eval text once, take the first
//! T9B_SEGMENTS segments of T9B_SEG_LEN tokens, fresh session per
//! segment, feed token i at position i and accumulate
//! nll_i = -ln softmax(logits)[token_{i+1}] in f64. Writes per-token
//! nll (f64 LE) for paired analysis and prints mean nll / PPL.
//!
//! Env: T9B_MODEL, T9B_TEXT, T9B_OUT_NLL, T9B_SEGMENTS (64),
//! T9B_SEG_LEN (512).

use objc2_metal::MTLBuffer as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::MetalContext;
use qwen_llm::metal_forward::{MetalForward, MetalModel, MetalSession};
use qwen_llm::tokenizer::Tokenizer;
use std::io::Write;

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let model_path = std::env::var("T9B_MODEL").expect("T9B_MODEL");
    let text_path = std::env::var("T9B_TEXT").expect("T9B_TEXT");
    let out_nll = std::env::var("T9B_OUT_NLL").expect("T9B_OUT_NLL");
    let n_segments = env_or("T9B_SEGMENTS", 64);
    let seg_len = env_or("T9B_SEG_LEN", 512);

    let g = GgufFile::open(&model_path).expect("open gguf");
    let m = Model::from_gguf(&g).expect("model");
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let ctx = MetalContext::new().expect("metal");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal model");
    let mf = MetalForward::new(&ctx, &mm);

    let text = std::fs::read_to_string(&text_path).expect("read text");
    let ids = tok.encode(&text, false).expect("tokenize");
    let need = n_segments * seg_len;
    assert!(
        ids.len() >= need,
        "eval text too short: {} < {need}",
        ids.len()
    );
    eprintln!(
        "[t9b-ppl] {model_path}: {} tokens available, using {need} ({n_segments}x{seg_len})",
        ids.len()
    );

    let t0 = std::time::Instant::now();
    let mut nlls: Vec<f64> = Vec::with_capacity(need - n_segments);
    for s in 0..n_segments {
        let seg = &ids[s * seg_len..(s + 1) * seg_len];
        let mut session = MetalSession::fresh(&ctx, &mm, seg_len + 8).expect("session");
        for i in 0..seg_len - 1 {
            let logits = mf.single_token(seg[i], i as u32, &mut session).expect("fw");
            let target = seg[i + 1] as usize;
            // f64 log-softmax.
            let mut mx = f64::NEG_INFINITY;
            for &v in &logits {
                let v = v as f64;
                if v > mx {
                    mx = v;
                }
            }
            let mut lse = 0f64;
            for &v in &logits {
                lse += ((v as f64) - mx).exp();
            }
            let lse = mx + lse.ln();
            nlls.push(lse - logits[target] as f64);
        }
        if s % 8 == 0 {
            eprintln!(
                "[t9b-ppl] segment {s}/{n_segments} ({:.0}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
    let ppl = mean.exp();
    println!(
        "[t9b-ppl] RESULT model={model_path} predicted_tokens={} mean_nll={mean:.6} ppl={ppl:.4} wall={:.0}s",
        nlls.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_nll).expect("nll out"));
    for v in &nlls {
        f.write_all(&v.to_le_bytes()).expect("w");
    }
    eprintln!("[t9b-ppl] wrote {out_nll}");
}
