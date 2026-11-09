//! Actual-model packet for same-binary batch-expert rollback timing and bits.
use super::*;
use std::{io::Write, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    assert!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "batch expert arithmetic differs"
    );
}
#[test]
#[ignore = "M5 UD-IQ3_XXS GLM53_GGUF; GLM53_BATCH_DEPTH=8192/32768/65536; internal lease; release"]
fn m5_batch_expert_abba_and_distribution() {
    run().unwrap();
}
fn run() -> anyhow::Result<()> {
    assert!(!cfg!(debug_assertions));
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let depth = std::env::var("GLM53_BATCH_DEPTH")
        .unwrap_or("32768".into())
        .parse::<usize>()?;
    assert!([8192, 32768, 65536].contains(&depth));
    let ctx = MetalContext::new()?;
    assert!(ctx.device_facts().gpu_families.apple10);
    let gg = crate::gguf::GgufFile::open(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF"))?;
    let w = Glm5NextWeights::load(&ctx, &gg)?;
    let f: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../scripts/reference/glm53/quality-v1.json"
    ))?;
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
            Glm5NextSession::with_prefill_rows(&ctx, &w, 73728, if i == 0 { 4096 } else { 4 })
                .unwrap()
        })
        .collect();
    assert!(
        evaluate_metal_memory_admission(4u64 << 30, 512 << 20, ctx.memory_signals(), true)
            .is_admitted()
    );
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    let empty = ss[0].capture_snapshot()?;
    let mut banks = Vec::new();
    emit(
        serde_json::json!({"event":"header","depth":depth,"device":ctx.device_facts(),"rows":[4096,4,4,4],"validation":std::env::var("MTL_DEBUG_LAYER").ok()}),
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
        emit(serde_json::json!({"event":"prefix","sequence":r}));
    }
    let mut reference = Vec::new();
    for enabled in [false, true] {
        batch.expert_rows = enabled;
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        for pos in 0..128 {
            let inputs: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
            let vs = batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &inputs)?;
            if !enabled {
                reference.push(vs);
            } else {
                for r in 0..4 {
                    eq(&reference[pos][r], &vs[r]);
                }
                emit(
                    serde_json::json!({"event":"quality","depth":depth,"position":pos,"vectors":4,"dnll":0,"kl":0,"bitwise":true}),
                );
            }
        }
    }
    drop(reference);
    for width in [2, 3, 4] {
        // Warm every route before any retained timing; same prefix every arm.
        for route in [0, 1, 2] {
            batch.expert_rows = route == 2;
            for r in 0..width {
                ss[r].restore_snapshot(&banks[r])?;
            }
            for pos in 0..8 {
                let inputs: Vec<_> = (0..width).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
                if route == 0 {
                    for r in 0..width {
                        std::hint::black_box(ss[r].forward(&ctx, inputs[r])?);
                    }
                } else {
                    std::hint::black_box(batch.step(
                        &ctx,
                        &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                        &inputs,
                    )?);
                }
            }
        }
        for (arm, route) in [0, 1, 2, 2, 1, 2, 1, 1, 2, 0].into_iter().enumerate() {
            batch.expert_rows = route == 2;
            for r in 0..width {
                ss[r].restore_snapshot(&banks[r])?;
            }
            let mut ms = Vec::new();
            for pos in 0..8 {
                let inputs: Vec<_> = (0..width).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
                let t = Instant::now();
                if route == 0 {
                    for r in 0..width {
                        std::hint::black_box(ss[r].forward(&ctx, inputs[r])?);
                    }
                } else {
                    std::hint::black_box(batch.step(
                        &ctx,
                        &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                        &inputs,
                    )?);
                }
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            emit(
                serde_json::json!({"event":"timing","depth":depth,"width":width,"arm":arm,"route":(["serial","rollback","default"][route]),"round_ms":ms}),
            );
        }
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
