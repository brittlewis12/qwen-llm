#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Real independent states, full-vocabulary distribution witness and step costs.
//! MODEL CODE PROSE DEPTH [STEPS=128] 4 [pair|pair-hc|fine|bounds].
//! Requires perf-ablations. Pair modes are valid independent-state A/Bs; fine/bounds
//! hold selected kernel outputs as cost ceilings and intentionally invalidate numerics.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
    },
    tokenizer::Tokenizer,
};
use std::{fs, time::Instant};
fn lz(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + v.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln()
}
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}
#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[4].parse()?;
    let steps: usize = a.get(5).map_or(Ok(128), |s| s.parse())?;
    assert!(steps > 8, "need eight warm rounds plus retained rounds");
    let widths: Vec<usize> = a
        .get(6)
        .map(|s| s.split(',').map(str::parse).collect())
        .transpose()?
        .unwrap_or(vec![2, 3, 4]);
    assert_eq!(widths, [4], "whole-round counterfactuals require B4");
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&gg)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let texts = [fs::read_to_string(&a[2])?, fs::read_to_string(&a[3])?];
    let mut docs = Vec::new();
    for i in 0..4 {
        let ids = tok.encode(&texts[i % 2], false)?;
        assert!(!ids.is_empty());
        docs.push(
            (0..depth + 512 + steps)
                .map(|j| ids[(j + i * 157) % ids.len()] as u32)
                .collect::<Vec<_>>(),
        );
    }
    let extent = depth + 4096 + steps;
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gg,
        Qwen4ExpSessionCapacity::for_forward_limit(&c, extent)?,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )?;
    let (factory, first) = loaded.session_factory(&ctx)?;
    let mut runners = vec![
        first,
        factory.create_runner(&ctx)?,
        factory.create_runner(&ctx)?,
        factory.create_runner(&ctx)?,
    ];
    let mode = a.get(7).map(String::as_str).unwrap_or("pair");
    let mut checkpoints = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        r.prefill(&docs[i][..depth + i * 17])?;
        checkpoints.push(r.capture_snapshot()?);
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners[0], 4)?;
    let tokens = |j: usize| {
        (0..4)
            .map(|i| docs[i][depth + i * 17 + j])
            .collect::<Vec<_>>()
    };
    use qwen_llm::metal::fn_batch_probe;
    if mode.starts_with("pair") {
        fn_batch_probe::candidate(mode.strip_prefix("pair-").unwrap_or("formats"));
        let mut other = Vec::new();
        for snap in &checkpoints {
            let mut r = factory.create_runner(&ctx)?;
            r.restore_snapshot(snap)?;
            other.push(r);
        }
        let mut second = Qwen4ExpDecodeBatch::new(&other[0], 4)?;
        let mut ms = [Vec::new(), Vec::new()];
        let mut diagnostics = [[0.; 3]; 2];
        let mut quality = [(0., 0.); 4];
        let mut repeat = Vec::new();
        {
            let _scope = fn_batch_probe::scope(Some(false), "", false);
            for j in 0..8 {
                repeat.push(native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens(j))?);
            }
        }
        for (r, snap) in runners.iter_mut().zip(&checkpoints) {
            r.reset()?;
            r.restore_snapshot(snap)?;
        }
        let mut hashes = [blake3::Hasher::new(), blake3::Hasher::new()];
        for j in 0..steps {
            let mut values = [Vec::new(), Vec::new()];
            for arm in if j % 2 == 0 { [0, 1] } else { [1, 0] } {
                let _scope = fn_batch_probe::scope(Some(arm == 1), "", false);
                let t = Instant::now();
                let batch = if arm == 0 { &mut native } else { &mut second };
                let states = if arm == 0 { &mut runners } else { &mut other };
                values[arm] = batch.step(&mut states.iter_mut().collect::<Vec<_>>(), &tokens(j))?;
                let elapsed = t.elapsed().as_secs_f64() * 1000.;
                if mode == "pair-qsa" {
                    assert_eq!(
                        fn_batch_probe::observed("qsa_norm_grids"),
                        if arm == 1 { 36 } else { 0 },
                        "QSA candidate engagement"
                    );
                }
                if mode == "pair-direct" {
                    assert_eq!(
                        fn_batch_probe::observed("direct_outputs") > 0,
                        arm == 1,
                        "direct output route engagement"
                    );
                }
                if mode == "pair-moe_batch_projections" {
                    assert_eq!(
                        fn_batch_probe::observed("moe_batch_projections") > 0,
                        arm == 1,
                        "member FFN projection engagement"
                    );
                }
                if mode == "pair-moe_batch_shared" {
                    assert_eq!(
                        fn_batch_probe::observed("moe_batch_shared") > 0,
                        arm == 1,
                        "member shared-expert engagement"
                    );
                }
                if mode == "pair-zero_one_hc" {
                    assert_eq!(
                        fn_batch_probe::observed("zero_one_hc") > 0,
                        arm == 1,
                        "member bootstrap HC engagement"
                    );
                }
                if mode == "pair-bf16_mixer" {
                    assert_eq!(
                        fn_batch_probe::observed("bf16_mixer") > 0,
                        arm == 1,
                        "member BF16 mixer engagement"
                    );
                }
                if j >= 8 {
                    ms[arm].push(elapsed);
                    for k in 0..3 {
                        diagnostics[arm][k] += batch.diagnostic_ms[k];
                    }
                }
            }
            if j < 8 {
                for i in 0..4 {
                    assert_eq!(values[0][i], repeat[j][i], "repeat noise");
                }
            }
            for i in 0..4 {
                let (r, v) = (&values[0][i], &values[1][i]);
                assert!(r.iter().chain(v).all(|x| x.is_finite()));
                for arm in 0..2 {
                    hashes[arm].update(bytemuck::cast_slice(&values[arm][i]));
                }
                let (rz, vz) = (lz(r), lz(v));
                let target = docs[i][depth + i * 17 + j + 1] as usize;
                quality[i].0 += (vz - v[target] as f64) - (rz - r[target] as f64);
                quality[i].1 += r
                    .iter()
                    .zip(v)
                    .map(|(&p, &q)| (p as f64 - rz).exp() * ((p as f64 - rz) - (q as f64 - vz)))
                    .sum::<f64>();
            }
        }
        for q in &mut quality {
            q.0 /= steps as f64;
            q.1 /= steps as f64;
            assert!(q.0.abs() <= 0.005 && q.1 <= 0.01);
        }
        let means = ms
            .each_ref()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64);
        for a in &mut diagnostics {
            for v in a {
                *v /= (steps - 8) as f64;
            }
        }
        println!(
            "{}",
            serde_json::json!({"kind":"paired","depth":depth,"steps":steps,"ms":means,"gain":means[0]/means[1]-1.,"paired_ms":ms[0].iter().zip(&ms[1]).map(|(a,b)|a-b).collect::<Vec<_>>(),"diagnostic_encode_gpu_wall_ms":diagnostics,"member_dNLL_KL":quality,"noise_dNLL_KL":[0.,0.],"stream_hash":hashes.map(|h|h.finalize().to_hex().to_string())})
        );
        let fork = other[0].capture_snapshot()?;
        other[1].reset()?;
        other[1].restore_snapshot(&fork)?;
        let _scope = fn_batch_probe::scope(Some(true), "", false);
        for j in 0..32 {
            let token = docs[0][depth + steps + j];
            let v = second.step(
                &mut other[..2].iter_mut().collect::<Vec<_>>(),
                &[token, token],
            )?;
            assert_eq!(v[0], v[1], "native exact fork");
        }
        println!(
            "{}",
            serde_json::json!({"kind":"fork","exact_rounds":32,"depth":depth})
        );
    } else {
        let groups = if mode == "remaining" {
            vec![
                "",
                "qsa_norm",
                "qsa_publish",
                "gdn_prep",
                "routing",
                "copies",
                "",
            ]
        } else if mode == "fine" {
            vec![
                "",
                "hc_inject",
                "gdn_prep",
                "qsa_prepare",
                "routing",
                "copies",
                "causal",
                "",
            ]
        } else {
            vec![
                "",
                "hc",
                "gdn",
                "qsa_prepare",
                "routing",
                "experts",
                "transport",
                "norms",
                "causal",
                "",
            ]
        };
        {
            let _scope = fn_batch_probe::scope(None, "", true);
            native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens(0))?;
        }
        for rep in 0..2 {
            for &group in if rep == 0 {
                groups.iter().collect::<Vec<_>>()
            } else {
                groups.iter().rev().collect::<Vec<_>>()
            } {
                for (r, snap) in runners.iter_mut().zip(&checkpoints) {
                    r.reset()?;
                    r.restore_snapshot(snap)?;
                }
                {
                    let _scope = fn_batch_probe::scope(None, "", false);
                    for j in 0..2 {
                        native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens(j))?;
                    }
                }
                let _scope = fn_batch_probe::scope(None, group, group == "causal");
                let mut sum = [0.; 3];
                let mut wall = 0.;
                let mut hashes = blake3::Hasher::new();
                let n = steps.min(24);
                for j in 2..n + 2 {
                    let t = Instant::now();
                    let values =
                        native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens(j))?;
                    wall += t.elapsed().as_secs_f64() * 1000.;
                    for k in 0..3 {
                        sum[k] += native.diagnostic_ms[k];
                    }
                    for v in values {
                        hashes.update(bytemuck::cast_slice::<f32, u8>(&v));
                    }
                }
                for v in &mut sum {
                    *v /= n as f64;
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"bound","depth":depth,"rep":rep,"group":group,"rounds":n,"wall_ms":wall/n as f64,"encode_gpu_wall_ms":sum,"skipped":fn_batch_probe::skipped(),"hash":hashes.finalize().to_hex().to_string(),"valid_outputs":group.is_empty()||group=="causal"})
                );
            }
        }
    }
    Ok(())
}
