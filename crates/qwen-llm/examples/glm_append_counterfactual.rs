//! Output-preserving whole-append counterfactuals; build with perf-ablations.
use qwen_llm::{
    gguf::GgufFile,
    glm_append_price as probe,
    glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
    metal::*,
    tokenizer::Tokenizer,
};
use std::{io::Write, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn pressure() -> i32 {
    let x = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    String::from_utf8(x.stdout)
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap()
}
fn compilers() -> String {
    let x = std::process::Command::new("ps")
        .args(["-axo", "comm="])
        .output()
        .unwrap();
    String::from_utf8(x.stdout)
        .unwrap()
        .lines()
        .filter(|s| s.ends_with("/rustc") || s.ends_with("/cargo") || s.ends_with("/metal"))
        .collect::<Vec<_>>()
        .join(";")
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let c = MetalContext::new()?;
    let model = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
        .path_or_skip()
        .ok_or_else(|| anyhow::anyhow!("QWEN_MODEL_SEARCH_PATH must resolve UD-IQ3_XXS"))?;
    let gg = GgufFile::open(model)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let text = std::fs::read_to_string(&args[1])?
        .replace("<|", "‹|")
        .replace("<｜", "‹｜");
    let ids = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    assert!(ids.len() > 68000);
    let w = Glm5NextWeights::load(&c, &gg)?;
    let mut s = Glm5NextSession::with_prefill_rows(&c, &w, 98304, 2048)?;
    for part in ids[..65536].chunks(2048) {
        while pressure() > 1 {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        s.prefill_packed(&c, part)?;
    }
    let snap = s.capture_snapshot()?;
    let ad = evaluate_metal_memory_admission(24u64 << 30, 512 << 20, c.memory_signals(), true);
    anyhow::ensure!(ad.is_admitted(), "counterfactual tape admission: {ad:?}");
    for n in args
        .get(2)
        .map(String::as_str)
        .unwrap_or("400,2000")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
    {
        probe::capture(&c);
        s.restore_snapshot(&snap)?;
        let baseline = s.prefill_packed(&c, &ids[65536..65536 + n])?;
        let hash = blake3::hash(bytemuck::cast_slice(&baseline));
        let (calls, bytes, _) = probe::finish();
        emit(
            serde_json::json!({"kind":"capture","rows":n,"calls":calls,"output_bytes":bytes,"hash":hash.to_hex().to_string()}),
        );
        let mut variants = vec![(1, false), (2, false), (3, false)];
        if args.get(3).is_some_and(|s| s == "stream") {
            variants.extend([(1, true), (2, true), (3, true)]);
        }
        for (mask, stream) in variants {
            for (rep, on) in [
                false, true, false, true, true, false, true, false, false, true,
            ]
            .into_iter()
            .enumerate()
            {
                probe::start(if on { mask } else { 0 }, stream);
                s.restore_snapshot(&snap)?;
                while pressure() > 1 {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
                let compile = compilers();
                let p0 = pressure();
                let t = Instant::now();
                let v = s.prefill_packed(&c, &ids[65536..65536 + n])?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let p1 = pressure();
                let compile_after = compilers();
                let (_, _, reads) = probe::finish();
                assert_eq!(s.position(), 65536 + n);
                assert!(!s.is_poisoned());
                assert!(v.iter().all(|v| v.is_finite()));
                assert_eq!(
                    hash,
                    blake3::hash(bytemuck::cast_slice(&v)),
                    "counterfactual changed endpoint"
                );
                emit(
                    serde_json::json!({"kind":"counterfactual","rows":n,"mask":mask,"stream":stream,"on":on,"rep":rep,"warm":rep<2,"ms":ms,"pressure_before":p0,"pressure_after":p1,"compile_processes":compile,"compile_processes_after":compile_after,"logical_stream_bytes_upper":reads}),
                );
                anyhow::ensure!(p0 <= 1 && p1 <= 1, "thermal pressure above1");
            }
        }
        probe::clear();
    }
    Ok(())
}
