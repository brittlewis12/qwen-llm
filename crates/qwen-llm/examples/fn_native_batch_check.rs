//! Real independent states, full-vocabulary distribution witness and step costs.
//! MODEL CODE PROSE DEPTH [STEPS=128] [WIDTHS=2,3,4] [profile]. Keep all heads/checkpoints in RAM.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use std::{fs, time::Instant};
fn lz(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + v.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln()
}
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
    assert!(!widths.is_empty() && widths.iter().all(|b| (2..=4).contains(b)));
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
    let mut checkpoints = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        r.prefill(&docs[i][..depth + i * 17])?;
        // Leave cached GPU input/control banks on every independent owner,
        // then hand those live histories to native B>1 decode below.
        r.generate_gpu_tokens(
            &mut Sampler::new(SamplingConfig::default())?,
            3,
            &[],
            |_| true,
        )?;
        checkpoints.push(r.capture_snapshot()?);
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners[0], 4)?;
    for width in widths {
        for (r, s) in runners.iter_mut().zip(&checkpoints) {
            r.reset()?;
            r.restore_snapshot(s)?;
        }
        let mut reference = vec![Vec::with_capacity(steps); width];
        let mut serial_ms = 0.;
        for step in 0..steps {
            let t = Instant::now();
            for i in 0..width {
                reference[i].push(
                    runners[i]
                        .forward_token(docs[i][depth + i * 17 + step])?
                        .to_vec(),
                );
            }
            if step >= 8 {
                serial_ms += t.elapsed().as_secs_f64() * 1000.;
            }
        }
        for (r, s) in runners.iter_mut().zip(&checkpoints) {
            r.reset()?;
            r.restore_snapshot(s)?;
        }
        let mut noise_exact = 0;
        for step in 0..8.min(steps) {
            for i in 0..width {
                let v = runners[i].forward_token(docs[i][depth + i * 17 + step])?;
                assert_eq!(
                    bytemuck::cast_slice::<f32, u8>(v.as_slice()),
                    bytemuck::cast_slice::<f32, u8>(&reference[i][step])
                );
                noise_exact += 1;
            }
        }
        for (r, s) in runners.iter_mut().zip(&checkpoints) {
            r.reset()?;
            r.restore_snapshot(s)?;
        }
        let (mut dnll, mut kl, mut max_abs, mut exact, mut native_ms) = (0., 0., 0f64, 0, 0.);
        let mut per_member = vec![(0., 0.); width];
        let mut hash = blake3::Hasher::new();
        for step in 0..steps {
            let tokens: Vec<_> = (0..width).map(|i| docs[i][depth + i * 17 + step]).collect();
            let mut members: Vec<_> = runners[..width].iter_mut().collect();
            let t = Instant::now();
            let values = native.step(&mut members, &tokens)?;
            if step >= 8 {
                native_ms += t.elapsed().as_secs_f64() * 1000.;
            }
            for i in 0..width {
                let v = &values[i];
                let r = &reference[i][step];
                assert_eq!(v.len(), r.len());
                assert!(v.iter().all(|x| x.is_finite()));
                let equal =
                    bytemuck::cast_slice::<f32, u8>(v) == bytemuck::cast_slice::<f32, u8>(r);
                exact += usize::from(equal);
                hash.update(bytemuck::cast_slice(v));
                if equal {
                    continue;
                }
                let (z, bz) = (lz(v), lz(r));
                let target = docs[i][depth + i * 17 + step + 1] as usize;
                let delta = (z - v[target] as f64) - (bz - r[target] as f64);
                let divergence = r
                    .iter()
                    .zip(v)
                    .map(|(&p, &q)| (p as f64 - bz).exp() * ((p as f64 - bz) - (q as f64 - z)))
                    .sum::<f64>();
                dnll += delta;
                kl += divergence;
                per_member[i].0 += delta;
                per_member[i].1 += divergence;
                for (&p, &q) in r.iter().zip(v) {
                    max_abs = max_abs.max((p as f64 - q as f64).abs());
                }
            }
        }
        dnll /= (width * steps) as f64;
        kl /= (width * steps) as f64;
        for (i, (delta, divergence)) in per_member.iter_mut().enumerate() {
            *delta /= steps as f64;
            *divergence /= steps as f64;
            assert!(
                delta.abs() <= 0.005 && *divergence <= 1e-2,
                "member {i} distribution failed dNLL={delta} KL={divergence}"
            );
        }
        assert!(
            dnll.abs() <= 0.005 && kl <= 1e-2,
            "distribution failed dNLL={dnll} KL={kl}"
        );
        let mut fork_exact = 0;
        if width == 4 {
            let entry = runners[0].capture_snapshot()?;
            runners[1].reset()?;
            runners[1].restore_snapshot(&entry)?;
            for step in 0..32 {
                let token = docs[0][depth + steps + step];
                let mut pair: Vec<_> = runners[..2].iter_mut().collect();
                if step % 2 == 1 {
                    pair.reverse();
                }
                let v = native.step(&mut pair, &[token, token])?;
                assert_eq!(
                    bytemuck::cast_slice::<f32, u8>(&v[0]),
                    bytemuck::cast_slice::<f32, u8>(&v[1])
                );
                fork_exact += 1;
            }
            let mut tokens = Vec::new();
            let mut rows = Vec::new();
            runners[0].generate_gpu_tokens_observed(
                &mut Sampler::new(SamplingConfig::default())?,
                4,
                &[],
                |t, row| {
                    tokens.push(t);
                    rows.push(row.to_vec());
                    true
                },
            )?;
            let mut maximum = 0f64;
            for (i, row) in rows.iter().enumerate() {
                if i > 0 {
                    runners[1].forward_token(tokens[i - 1] as u32)?;
                }
                let error = qwen_llm::compare::report_max_abs_diff_f64(
                    row,
                    runners[1].logits()?.as_slice(),
                );
                assert!(error.is_finite(), "invalid GPU/native comparison");
                maximum = maximum.max(error);
            }
            assert_eq!(
                maximum, 0.,
                "GPU -> native batch -> GPU history/control round trip"
            );
            println!(
                "{}",
                serde_json::json!({"kind":"gpu_native_roundtrip","depth":depth,"max_logit_error":maximum})
            );
        }
        println!(
            "{}",
            serde_json::json!({"depth":depth,"width":width,"steps":steps,"serial_step_ms":serial_ms/(steps-8) as f64,"native_step_ms":native_ms/(steps-8) as f64,"aggregate_speedup":serial_ms/native_ms,"signed_dNLL":dnll,"mean_KL":kl,"member_dNLL_KL":per_member,"max_logit_error":max_abs,"bitwise_vectors":exact,"noise_exact_vectors":noise_exact,"noise_dNLL":0.,"noise_KL":0.,"fork_exact":fork_exact,"stream_hash":hash.finalize().to_hex().to_string()})
        );
    }
    if a.get(7).is_some_and(|v| v == "profile") {
        // Profile four independent histories, after the fork witness has
        // finished. Warm the restored states without including those dispatches.
        for (r, s) in runners.iter_mut().zip(&checkpoints) {
            r.reset()?;
            r.restore_snapshot(s)?;
        }
        for step in 0..8 {
            let tokens: Vec<_> = (0..4).map(|i| docs[i][depth + i * 17 + step]).collect();
            native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens)?;
        }
        let tokens: Vec<_> = runners
            .iter()
            .enumerate()
            .map(|(i, r)| docs[i][r.next_position()])
            .collect();
        let guard = qwen_llm::metal::dispatch_profile_begin(&ctx, 10000)?;
        native.step(&mut runners.iter_mut().collect::<Vec<_>>(), &tokens)?;
        let report = qwen_llm::metal::dispatch_profile_finish(&ctx, guard)?;
        assert_eq!(
            report.overflow, 0,
            "dispatch profile exceeded its admitted rows"
        );
        let mut kernels = std::collections::BTreeMap::<String, (usize, u64)>::new();
        for row in report.rows {
            let e = kernels.entry(row.kernel).or_default();
            e.0 += 1;
            e.1 += row.gpu_ns.unwrap_or(0);
        }
        println!(
            "{}",
            serde_json::json!({"dispatch_kernels":kernels,"overflow":report.overflow})
        );
    }
    Ok(())
}
