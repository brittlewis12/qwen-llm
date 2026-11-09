//! B=4 GLM per-kernel census: commit plus this command, compact summaries only.
//! cargo run --release -p qwen-llm --example glm_batch_census -- MODEL 8192
//! Requires scripts/bench/with-metal-lease; no build lock during timing.
use qwen_llm::{gguf::GgufFile, glm5_next_metal::*, metal::*};
use std::{io::Write, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn main() -> anyhow::Result<()> {
    assert!(
        std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
        "canonical lease required"
    );
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args[2].parse()?;
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&args[1])?;
    let w = Glm5NextWeights::load(&ctx, &gg)?;
    let frozen = include_bytes!("../../../scripts/reference/glm53/quality-v1.json");
    let f: serde_json::Value = serde_json::from_slice(frozen)?;
    let docs: Vec<Vec<u32>> = f["documents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d["tokens"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        })
        .collect();
    let mut ss: Vec<_> = (0..4)
        .map(|i| {
            Glm5NextSession::with_prefill_rows(&ctx, &w, 73728, if i == 0 { 2048 } else { 4 })
                .unwrap()
        })
        .collect();
    assert!(
        evaluate_metal_memory_admission(8u64 << 30, 512 << 20, ctx.memory_signals(), true)
            .is_admitted()
    );
    let empty = ss[0].capture_snapshot()?;
    let mut banks = Vec::new();
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    emit(
        serde_json::json!({"event":"header","depth":depth,"device":ctx.device_facts(),"fixture_blake3":blake3::hash(frozen).to_hex().as_str(),"width":4}),
    );
    for r in 0..4 {
        ss[0].restore_snapshot(&empty)?;
        ss[0].prefill_packed(
            &ctx,
            &docs[[0, 8, 16, 9][r]]
                .iter()
                .copied()
                .cycle()
                .take(depth)
                .collect::<Vec<_>>(),
        )?;
        banks.push(ss[0].capture_snapshot()?);
        emit(serde_json::json!({"event":"prefix","seq":r}));
    }
    for r in 0..4 {
        ss[r].restore_snapshot(&banks[r])?;
    }
    for p in 0..8 {
        let input: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][p]).collect();
        batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &input)?;
    }
    for p in 8..24 {
        let input: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][p]).collect();
        let t = Instant::now();
        let vs = batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &input)?;
        assert!(vs.iter().flatten().all(|v| v.is_finite()));
        emit(
            serde_json::json!({"event":"ordinary","position":p,"wall_ms":t.elapsed().as_secs_f64()*1e3}),
        );
    }
    let mut totals = std::collections::BTreeMap::<String, (usize, u64)>::new();
    let mut walls = Vec::new();
    for p in 24..28 {
        let input: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][p]).collect();
        let guard = dispatch_profile_begin(&ctx, 4096)?;
        let t = Instant::now();
        batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &input)?;
        walls.push(t.elapsed().as_secs_f64() * 1e3);
        let report = dispatch_profile_finish(&ctx, guard)?;
        assert_eq!(report.overflow, 0);
        for row in report.rows {
            let v = totals.entry(row.kernel).or_default();
            v.0 += 1;
            v.1 += row.gpu_ns.unwrap_or(0);
        }
    }
    for (kernel, (calls, ns)) in totals {
        emit(
            serde_json::json!({"event":"family","kernel":kernel,"calls_per_round":calls/4,"sampled_span_ms":ns as f64/4e6}),
        );
    }
    emit(
        serde_json::json!({"event":"profile_wall","round_ms":walls,"warning":"Sampled encoders perturb execution; spans are not additive ordinary wall attribution."}),
    );
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
