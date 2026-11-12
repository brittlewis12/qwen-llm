//! Flash-Next append wall/GPU price, without changing the native execution.
//! fn_append_price MODEL CORPUS [DEPTH=49152] [SIZES=32,342,1383,1798,2048,5479]
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{io::Write, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn pressure() -> i32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|x| String::from_utf8(x.stdout).ok())
        .and_then(|x| x.split_whitespace().last()?.parse().ok())
        .unwrap_or(-1)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args.get(3).map_or(Ok(49152), |x| x.parse())?;
    let sizes: Vec<usize> = args
        .get(4)
        .map_or("32,342,1383,1798,2048,5479", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&args[1])?;
    let config = Qwen4ExpConfig::from_gguf(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut text = std::fs::read_to_string(&args[2])?;
    while tok.encode(&text, false)?.len() < depth + sizes.iter().max().unwrap() + 1 {
        let extra = text.clone();
        text.push_str(&extra);
    }
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let limit = 98304;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&config, limit)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, limit)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(true);
    runner.configure_gpu_token_loop(true);
    runner.prefill(&ids[..depth])?;
    let root = runner.capture_snapshot()?;
    let logits = runner.logits()?.to_vec();
    emit(
        json!({"kind":"header","depth":depth,"packed_rows":runner.packed_prefill_capacity(),"pressure":pressure()}),
    );
    for n in sizes {
        for rep in 0..3 {
            runner.restore_snapshot_with_logits(&root, &logits)?;
            let before = qwen_llm::pid_metrics::PidSnapshot::now().ok();
            let start = Instant::now();
            runner
                .prefill_continuation_with_command_checkpoint(&ids[depth..depth + n], || Ok(()))?;
            let wall = start.elapsed().as_secs_f64() * 1000.;
            let t = runner.last_prefill_timing().ok_or("missing timing")?;
            let after = qwen_llm::pid_metrics::PidSnapshot::now().ok();
            emit(
                json!({"kind":"append","rows":n,"rep":rep,"wall_ms":wall,"encode_cpu_ms":t.encode_cpu_ms,
    "gpu_ms":t.gpu_ms,"wait_ms":t.completion_wait_ms,"gpu_samples":t.gpu_samples,"commands":t.command_count,
    "packed_tokens":t.packed_token_count,"pressure":pressure(),"disk_read_bytes":before.zip(after).map(|(a,b)|b.diskio_bytesread.saturating_sub(a.diskio_bytesread))}),
            );
        }
    }
    Ok(())
}
