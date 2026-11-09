//! Committed whole-round kill/likelihood driver. No exported snapshots or heads.
use super::*;
use crate::metal::{dispatch_census_begin, dispatch_census_take};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    time::Instant,
};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    assert!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "projection/MLA arithmetic differs"
    );
}
#[test]
#[ignore = "M5 UD-IQ3_XXS; GLM53_GGUF and GLM53_BATCH_DEPTH=8192/32768/65536; internal lease, release"]
fn m5_projection_abba_and_distribution() {
    run().unwrap();
}
fn run() -> anyhow::Result<()> {
    assert!(!cfg!(debug_assertions));
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let depth = std::env::var("GLM53_BATCH_DEPTH")
        .unwrap_or("8192".into())
        .parse::<usize>()?;
    assert!([8192, 32768, 65536].contains(&depth));
    let control = std::env::var("GLM53_BATCH_CONTROL")
        .unwrap_or("0".into())
        .parse::<u8>()?;
    let candidate = std::env::var("GLM53_BATCH_CANDIDATE")
        .unwrap_or("31".into())
        .parse::<u8>()?;
    assert_ne!(control, candidate);
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
                .map(|t| t.parse().unwrap())
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
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    let empty = ss[0].capture_snapshot()?;
    let mut banks = Vec::new();
    emit(
        serde_json::json!({"event":"header","depth":depth,"control":control,"candidate":candidate,"device":ctx.device_facts(),"validation":std::env::var("MTL_DEBUG_LAYER").ok()}),
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
        emit(serde_json::json!({"event":"prefix","member":r}));
    }
    drop(empty);
    // First census includes the actual routed unions, avoiding a fourfold
    // expert-byte assumption where members select overlapping experts.
    for variant in [control, candidate] {
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        for pos in 0..8 {
            let tokens: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
            projection::with_features(variant, || {
                batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &tokens)
            })?;
        }
        dispatch_census_begin();
        projection::with_features(variant, || {
            batch.step(
                &ctx,
                &mut ss.iter_mut().collect::<Vec<_>>(),
                &[1, 10, 17, 11],
            )
        })?;
        let census = dispatch_census_take();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for r in &census {
            *counts.entry(r.kernel.clone()).or_default() += 1;
        }
        let mut union_bytes = 0u64;
        let mut logical_bytes = 0u64;
        let mut unique = 0;
        let mut references = 0;
        let p = ss[0].packed.as_ref().unwrap();
        let k = w.config.expert_used_count as usize;
        for (i, block) in w.blocks.iter().enumerate() {
            if let FfnTensors::Moe(moe) = &block.ffn {
                let route = p.routes[i].as_ref().unwrap();
                let ids = read_i32(&route.ids.view_subrange(0, vec![(k * 4) as u64]))?;
                assert!(
                    ids.iter()
                        .all(|id| *id >= 0 && *id < w.config.expert_count as i32)
                );
                let u: BTreeSet<_> = ids.iter().copied().collect();
                let bytes = (moe.gate_experts.n_bytes()
                    + moe.up_experts.n_bytes()
                    + moe.down_experts.n_bytes())
                    / u64::from(w.config.expert_count);
                unique += u.len();
                references += ids.len();
                union_bytes += u.len() as u64 * bytes;
                logical_bytes += ids.len() as u64 * bytes;
            }
        }
        emit(
            serde_json::json!({"event":"census","variant":variant,"dispatches":census.len(),"counts":counts,"routed_union_bytes":union_bytes,"routed_logical_bytes":logical_bytes,"unique_layer_experts":unique,"layer_expert_references":references}),
        );
    }
    let mut reference = Vec::new();
    for variant in [control, candidate] {
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        for pos in 0..128 {
            let tokens: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
            let vs = projection::with_features(variant, || {
                batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &tokens)
            })?;
            if variant == control {
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
    // Changing first workspace, member order, width, and singleton fallback.
    let mut reference = Vec::new();
    let mut states = Vec::new();
    for variant in [control, candidate] {
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        let mut consumed = [0usize; 4];
        for pos in 0..32 {
            let order: &[usize] = match pos % 4 {
                0 => &[3, 1, 0, 2],
                1 => &[0, 2, 1],
                2 => &[1, 3],
                _ => &[2],
            };
            let tokens: Vec<_> = order
                .iter()
                .map(|r| docs[[1, 10, 17, 11][*r]][consumed[*r]])
                .collect();
            let mut refs: Vec<_> = ss
                .iter_mut()
                .enumerate()
                .filter(|(r, _)| order.contains(r))
                .collect();
            refs.sort_by_key(|(r, _)| order.iter().position(|i| i == r).unwrap());
            let mut refs: Vec<_> = refs.into_iter().map(|(_, s)| s).collect();
            let vs = projection::with_features(variant, || batch.step(&ctx, &mut refs, &tokens))?;
            for r in order {
                consumed[*r] += 1;
            }
            if variant == control {
                reference.push(vs);
            } else {
                for (a, b) in reference[pos].iter().zip(&vs) {
                    eq(a, b);
                }
            }
        }
        // Borrowed packed sparse/activation scratch must also survive the
        // transition back to a real append, including four-row workspaces.
        let mut appended = Vec::new();
        for r in 0..4 {
            appended.push(ss[r].prefill_packed(
                &ctx,
                &docs[[1, 10, 17, 11][r]][consumed[r]..consumed[r] + 13],
            )?);
        }
        if variant == control {
            reference.push(appended);
        } else {
            for (a, b) in reference[32].iter().zip(&appended) {
                eq(a, b);
            }
        }
        if variant == control {
            for s in &ss {
                states.push(s.capture_snapshot()?);
            }
        } else {
            for (s, old) in ss.iter().zip(&states) {
                assert!(s.capture_snapshot()?.same_state(old));
            }
        }
    }
    emit(
        serde_json::json!({"event":"membership","rounds":32,"append_after_batch":13,"state_and_logits_bitwise":true}),
    );
    drop(reference);
    drop(states);
    for width in [2, 3, 4] {
        for variant in [control, candidate] {
            for r in 0..width {
                ss[r].restore_snapshot(&banks[r])?;
            }
            for pos in 0..8 {
                let ts: Vec<_> = (0..width).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
                projection::with_features(variant, || {
                    batch.step(&ctx, &mut ss[..width].iter_mut().collect::<Vec<_>>(), &ts)
                })?;
            }
        }
        let arms = [
            control, candidate, candidate, control, candidate, control, control, candidate,
        ];
        for (arm, variant) in arms.into_iter().enumerate() {
            for r in 0..width {
                ss[r].restore_snapshot(&banks[r])?;
            }
            let mut wall = Vec::new();
            let mut encode = Vec::new();
            let mut gpu = Vec::new();
            for pos in 0..8 {
                let ts: Vec<_> = (0..width).map(|r| docs[[1, 10, 17, 11][r]][pos]).collect();
                let t = Instant::now();
                std::hint::black_box(projection::with_features(variant, || {
                    batch.step(&ctx, &mut ss[..width].iter_mut().collect::<Vec<_>>(), &ts)
                })?);
                wall.push(t.elapsed().as_secs_f64() * 1e3);
                let (e, g) = projection::timing();
                encode.push(e);
                gpu.push(g);
            }
            emit(
                serde_json::json!({"event":"timing","depth":depth,"width":width,"arm":arm,"variant":variant,"wall_ms":wall,"encode_ms":encode,"gpu_ms":gpu}),
            );
        }
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
