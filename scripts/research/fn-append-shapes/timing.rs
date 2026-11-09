//! Positional FAMILY MODEL CORPUS [unused-controls], run through append-roofline/run.py.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::time::Instant;
fn hash(v: &[f32]) -> String {
    assert!(v.iter().all(|x| x.is_finite()));
    blake3::hash(bytemuck::cast_slice(v)).to_hex().to_string()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let g = GgufFile::open(&a[2])?;
    let tok = Tokenizer::from_gguf(&g)?;
    let text = std::fs::read_to_string(&a[3])?
        .replace("<|", "‹|")
        .replace("<｜", "‹｜");
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| u32::try_from(x).unwrap())
        .collect();
    let depth = 65536;
    assert!(ids.len() > depth + 2048);
    let ctx = MetalContext::new()?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, depth + 4096)?;
    let mut m = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, depth + 4096)?;
    let mut r = m.create_runner(&ctx)?;
    let start = Instant::now();
    let prefix = r.prefill(&ids[..depth])?.to_vec();
    println!(
        "{}",
        json!({"kind":"prefix","wall_ms":start.elapsed().as_secs_f64()*1e3,"hash":hash(&prefix),"input_hash":blake3::hash(bytemuck::cast_slice(&ids[..depth])).to_hex().to_string()})
    );
    let entry = r.capture_snapshot()?;
    for n in [400, 512, 2000, 2048] {
        let mut expected = String::new();
        for rep in 0..3 {
            r.restore_snapshot(&entry)?;
            let start = Instant::now();
            let v = r
                .prefill_continuation_with_command_checkpoint(&ids[depth..depth + n], || Ok(()))?
                .to_vec();
            let wall = start.elapsed().as_secs_f64() * 1e3;
            let h = hash(&v);
            if rep == 0 {
                expected = h.clone();
            } else {
                assert_eq!(h, expected);
            }
            let t = r.last_prefill_timing().unwrap();
            println!(
                "{}",
                json!({"kind":"native","rows":n,"rep":rep,"warm":rep==0,"wall_ms":wall,"gpu_ms":t.gpu_ms,"encode_ms":t.encode_cpu_ms,"commands":t.command_count,"hash":h})
            );
        }
    }
    Ok(())
}
