//! Private actual-model qualification driver; raw per-position likelihoods.
use crate::{gguf::GgufFile, glm5_next_metal::*, metal::*};
use std::io::Write;
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    assert!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "different arithmetic under width/permutation/replay"
    );
}
fn z(v: &[f32]) -> f64 {
    assert!(v.iter().all(|x| x.is_finite()));
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + v.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln()
}
fn dist(a: &[f32], b: &[f32], t: u32) -> (f64, f64, f64) {
    let za = z(a);
    let zb = z(b);
    let kl = a
        .iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - za).exp() * ((*a as f64 - za) - (*b as f64 - zb)))
        .sum();
    (za - a[t as usize] as f64, zb - b[t as usize] as f64, kl)
}
fn qualify(depth: usize) -> anyhow::Result<()> {
    assert!(!cfg!(debug_assertions));
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production Metal lease");
    let path = std::env::var_os("GLM53_GGUF").expect("actual UD-IQ3_XXS via GLM53_GGUF");
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(path)?;
    let w = Glm5NextWeights::load(&ctx, &gg)?;
    let frozen = include_bytes!("../../../../../scripts/reference/glm53/quality-v1.json");
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
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .collect();
    let mut ss: Vec<_> = (0..4)
        .map(|i| {
            Glm5NextSession::with_prefill_rows(&ctx, &w, 40960, if i == 0 { 2048 } else { 4 })
                .unwrap()
        })
        .collect();
    assert!(
        evaluate_metal_memory_admission(4u64 << 30, 512 << 20, ctx.memory_signals(), true)
            .is_admitted()
    );
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    // Invalid widths/tokens and non-Half modes refuse the whole group first.
    for mode in [FastPrecision::DenseF32, FastPrecision::F32] {
        ss[1].set_fast_precision(mode)?;
        let refs = &mut ss[..2].iter_mut().collect::<Vec<_>>();
        assert!(batch.step(&ctx, refs, &[1, 1]).is_err());
        assert!(refs.iter().all(|s| s.position() == 0 && !s.is_poisoned()));
        ss[1].set_fast_precision(FastPrecision::Half)?;
    }
    ss[1].set_packed_lineage(PackedLineage::Exact)?;
    assert!(
        batch
            .step(&ctx, &mut ss[..2].iter_mut().collect::<Vec<_>>(), &[1, 1])
            .is_err()
    );
    ss[1].set_packed_lineage(PackedLineage::Fast)?;
    assert!(
        batch
            .step(
                &ctx,
                &mut ss[..2].iter_mut().collect::<Vec<_>>(),
                &[1, u32::MAX]
            )
            .is_err()
    );
    assert!(
        batch
            .step(&ctx, &mut ss[..2].iter_mut().collect::<Vec<_>>(), &[1])
            .is_err()
    );
    assert!(ss.iter().all(|s| s.position() == 0 && !s.is_poisoned()));
    let instance = batch.weights_instance;
    batch.weights_instance = instance.wrapping_add(1);
    assert!(
        batch
            .step(&ctx, &mut ss[..2].iter_mut().collect::<Vec<_>>(), &[1, 1])
            .is_err()
    );
    batch.weights_instance = instance;
    let pos = ss[1].position;
    ss[1].position = ss[1].capacity;
    assert!(
        batch
            .step(&ctx, &mut ss[..2].iter_mut().collect::<Vec<_>>(), &[1, 1])
            .is_err()
    );
    ss[1].position = pos;
    let rows = ss[0].packed.as_ref().unwrap().rows;
    ss[0].packed.as_mut().unwrap().rows = 1;
    assert!(
        batch
            .step(
                &ctx,
                &mut ss[..3].iter_mut().collect::<Vec<_>>(),
                &[1, 1, 1]
            )
            .is_err()
    );
    ss[0].packed.as_mut().unwrap().rows = rows;
    assert!(ss.iter().all(|s| s.position() == 0 && !s.is_poisoned()));
    for (lineage, precision) in [
        (PackedLineage::Exact, FastPrecision::Half),
        (PackedLineage::Fast, FastPrecision::Half),
        (PackedLineage::Fast, FastPrecision::DenseF32),
        (PackedLineage::Fast, FastPrecision::F32),
    ] {
        ss[0].set_packed_lineage(lineage)?;
        ss[0].set_fast_precision(precision)?;
        let empty = ss[0].capture_snapshot()?;
        let a = ss[0].forward(&ctx, docs[0][0])?;
        ss[0].restore_snapshot(&empty)?;
        let b = batch.step(&ctx, &mut [&mut ss[0]], &[docs[0][0]])?;
        eq(&a, &b[0]);
        ss[0].reset()?;
    }
    ss[0].set_packed_lineage(PackedLineage::Fast)?;
    ss[0].set_fast_precision(FastPrecision::Half)?;
    let empty = ss[0].capture_snapshot()?;
    emit(
        serde_json::json!({"event":"header","depth":depth,"device":ctx.device_facts(),"fixture_blake3":blake3::hash(frozen).to_hex().as_str(),"b1_all_modes_bits":true,"invalid_group_unchanged":true,"targets":128,"documents":8}),
    );
    let mut likelihood_totals = Vec::new();
    for (group, (prefix_ids, suffix_ids)) in [
        ([0, 8, 16, 9], [1, 10, 17, 11]),
        ([2, 10, 18, 11], [3, 12, 19, 13]),
    ]
    .into_iter()
    .enumerate()
    {
        let mut banks = Vec::new();
        for r in 0..4 {
            ss[0].restore_snapshot(&empty)?;
            ss[0].prefill_packed(
                &ctx,
                &docs[prefix_ids[r]]
                    .iter()
                    .copied()
                    .cycle()
                    .take(depth)
                    .collect::<Vec<_>>(),
            )?;
            banks.push(ss[0].capture_snapshot()?);
            emit(
                serde_json::json!({"event":"prefix","depth":depth,"group":group,"seq":r,"document":prefix_ids[r]}),
            );
        }
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        dispatch_census_begin();
        let _ = batch.step(
            &ctx,
            &mut ss.iter_mut().collect::<Vec<_>>(),
            &(0..4).map(|r| docs[suffix_ids[r]][0]).collect::<Vec<_>>(),
        )?;
        let census = dispatch_census_take();
        let gemms = census
            .iter()
            .filter(|r| r.kernel.contains("fewrow_gen"))
            .count();
        let experts = census
            .iter()
            .filter(|r| r.kernel.starts_with("kernel_expert_native_") && r.grid_depth == 4)
            .count();
        assert_eq!(gemms, 508);
        assert!(experts >= 80);
        emit(
            serde_json::json!({"event":"census","group":group,"gemms":gemms,"native_expert_grids":experts,"dispatches":census.len()}),
        );
        let mut reference: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 4];
        let mut candidate: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 4];
        for arm in 0..3 {
            for r in 0..4 {
                ss[r].restore_snapshot(&banks[r])?;
            }
            for pos in 0..128 {
                let inputs: Vec<_> = (0..4).map(|r| docs[suffix_ids[r]][pos]).collect();
                let vs = if arm == 2 {
                    batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &inputs)?
                } else {
                    (0..4)
                        .map(|r| ss[r].forward(&ctx, inputs[r]).unwrap())
                        .collect()
                };
                for r in 0..4 {
                    if arm == 0 {
                        reference[r].push(vs[r].clone());
                    } else if arm == 1 {
                        eq(&reference[r][pos], &vs[r]);
                    } else {
                        let target = docs[suffix_ids[r]][pos + 1];
                        let (nll0, nll1, kl) = dist(&reference[r][pos], &vs[r], target);
                        likelihood_totals.push((nll1 - nll0, kl));
                        candidate[r].push(vs[r].clone());
                        emit(
                            serde_json::json!({"event":"quality","depth":depth,"group":group,"seq":r,"prefix_document":prefix_ids[r],"suffix_document":suffix_ids[r],"position":pos,"nll_serial":nll0,"nll_batch":nll1,"dnll":nll1-nll0,"kl":kl,"identical_arithmetic_bits":true}),
                        );
                    }
                }
            }
        }
        // Changing first workspace (2048 -> 4 rows), member order and active width
        // must not couple a sequence to its neighbours' activations or state.
        for order in [vec![3, 2, 0, 1], vec![0, 1], vec![0, 1, 2]] {
            for r in 0..4 {
                ss[r].restore_snapshot(&banks[r])?;
            }
            for pos in 0..16 {
                let inputs: Vec<_> = order.iter().map(|r| docs[suffix_ids[*r]][pos]).collect();
                let mut refs: Vec<_> = ss
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| order.contains(i))
                    .collect();
                refs.sort_by_key(|(i, _)| order.iter().position(|r| r == i).unwrap());
                let mut refs: Vec<_> = refs.into_iter().map(|(_, s)| s).collect();
                let vs = batch.step(&ctx, &mut refs, &inputs)?;
                for (i, r) in order.iter().enumerate() {
                    eq(&candidate[*r][pos], &vs[i]);
                }
            }
            emit(
                serde_json::json!({"event":"independence","depth":depth,"group":group,"order":order,"positions":16,"bitwise":true}),
            );
        }
    }
    // A malformed borrowed activation packet is an encoding failure, rather
    // than a preflight decline. Never retry any member of that started group.
    let unaffected = ss[3].position();
    let saved = ss[0].packed.as_ref().unwrap().q.clone();
    ss[0].packed.as_mut().unwrap().q = ss[0].packed.as_ref().unwrap().normed.clone();
    assert!(
        batch
            .step(
                &ctx,
                &mut ss[..3].iter_mut().collect::<Vec<_>>(),
                &[1, 1, 1]
            )
            .is_err()
    );
    assert!(ss[..3].iter().all(|s| s.is_poisoned()));
    assert_eq!(ss[3].position(), unaffected);
    assert!(!ss[3].is_poisoned());
    ss[0].packed.as_mut().unwrap().q = saved;
    let mean_dnll =
        likelihood_totals.iter().map(|v| v.0).sum::<f64>() / likelihood_totals.len() as f64;
    let mean_kl =
        likelihood_totals.iter().map(|v| v.1).sum::<f64>() / likelihood_totals.len() as f64;
    assert!(mean_dnll <= 0.005, "one-sided mean dNLL {mean_dnll}");
    assert!(mean_kl <= 1e-2, "mean KL {mean_kl}");
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}

#[test]
#[ignore = "M5 actual UD-IQ3_XXS, GLM53_GGUF, release, validation ON; internal production lease"]
fn m5_independent_likelihood_8k() {
    qualify(8192).unwrap();
}
#[test]
#[ignore = "M5 actual UD-IQ3_XXS, GLM53_GGUF, release, validation ON; internal production lease"]
fn m5_independent_likelihood_32k() {
    qualify(32768).unwrap();
}
