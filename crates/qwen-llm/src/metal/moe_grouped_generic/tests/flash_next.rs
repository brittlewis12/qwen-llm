//! Production Flash-Next routed dimensions; sampled independent F64 dots.
use super::*;

const H: usize = 2560;
const F: usize = 640;
const E: usize = 512;
const T: usize = 2048;
const TOP: usize = 10;

fn routes() -> Routing {
    let mut r = Routing {
        n_tokens: T,
        topk: TOP,
        n_expert: E,
        slot_expert: vec![0; T * TOP],
        counts: vec![0; E],
        ids: vec![-1; E * T],
    };
    // One full expert, one empty expert, and uneven loads in the remainder.
    for slot in 0..T * TOP {
        let e = if slot % TOP == 0 {
            0
        } else {
            1 + (slot * 17) % (E - 2)
        };
        r.slot_expert[slot] = e;
        r.ids[e * T + r.counts[e] as usize] = slot as i32;
        r.counts[e] += 1;
    }
    assert_eq!(r.counts[0], T as i32);
    assert_eq!(r.counts[E - 1], 0);
    r
}

fn dot64(w: &[f32], x: &[f32]) -> f64 {
    w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum()
}

fn check(label: &str, incumbent: &[f32], gpu: &[f32], reference: &[f64]) {
    let errors = |values: &[f32]| {
        let mut peak = 0.0f64;
        let mut squared = 0.0;
        let mut norm = 0.0;
        for (&v, &r) in values.iter().zip(reference) {
            assert!(v.is_finite() && r.is_finite());
            peak = peak.max((v as f64 - r).abs());
            squared += (v as f64 - r).powi(2);
            norm += r * r;
        }
        (peak, (squared / norm.max(1e-30)).sqrt())
    };
    let a = errors(incumbent);
    let b = errors(gpu);
    eprintln!(
        "flash_reference {label} sampled={} incumbent={a:?} tensor={b:?} peak_ratio={} rms_ratio={}",
        gpu.len(),
        b.0 / a.0.max(1e-30),
        b.1 / a.1.max(1e-30)
    );
    assert!(b.0 <= 1.5 * a.0 + 1e-7, "{label} F64 maximum error");
    assert!(b.1 <= 1.5 * a.1 + 1e-7, "{label} F64 relative RMS");
}

fn thermal() {
    let result = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    let raw = String::from_utf8(result.stdout).unwrap();
    eprintln!("flash_thermal {}", raw.trim());
    assert!(result.status.success() && raw.split_whitespace().last() == Some("0"));
}

fn screen(ctx: &MetalContext, label: &str, a: &dyn Fn(&KernelEncoder), b: &dyn Fn(&KernelEncoder)) {
    thermal();
    heat(ctx, 2, &[a, b]);
    for (arm, encode) in [("A", a), ("B", b), ("B", b), ("A", a)] {
        thermal();
        for sample in 0..6 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode(&enc);
            enc.end();
            commit_and_wait(&command).unwrap();
            eprintln!(
                "flash_kernel {label} arm={arm} sample={sample} gpu_ms={:.6}",
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.0
            );
        }
    }
    thermal();
}

#[test]
#[ignore = "production lease; FLASH_PREFILL_MODEL; real Flash-Next expert banks and M5 tensor tier"]
fn flash_next_tensor_routed_production_f64() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(moe_grouped_tensor_available(&ctx));
    let gguf =
        crate::gguf::GgufFile::open(std::env::var_os("FLASH_PREFILL_MODEL").unwrap()).unwrap();
    let bank = |dtype: GgmlType, role: &str| {
        let desc = gguf
            .tensors
            .iter()
            .find(|d| d.dtype == dtype && d.name.ends_with(role))
            .unwrap();
        eprintln!("flash_bank {} {:?} {:?}", desc.name, desc.dtype, desc.shape);
        gguf.try_slice(desc).unwrap().to_vec()
    };
    let r = routes();
    let counts = i32_typed_tensor(&ctx, &r.counts);
    let ids = i32_typed_tensor(&ctx, &r.ids);
    let sample_slots = [0, 1, 9, 10, 31, 32, 33, 511, T * TOP - 1];
    for dtype in [
        GgmlType::IQ2_S,
        GgmlType::IQ3_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ4_XS,
    ] {
        let gate = bank(dtype, "ffn_gate_exps.weight");
        let up = bank(dtype, "ffn_up_exps.weight");
        let g = bank_tensor(&ctx, dtype, &gate, H, F, E);
        let u = bank_tensor(&ctx, dtype, &up, H, F, E);
        let mut rng = Lcg(0x9300);
        let x: Vec<f32> = (0..T * H)
            .map(|i| {
                if i < H {
                    0.0
                } else if i < 2 * H {
                    rng.signed(1e-6)
                } else {
                    rng.signed(0.125)
                }
            })
            .collect();
        let xt = f32_tensor(&ctx, &x);
        let out = MetalTensor::zeros_f32(&ctx, vec![(T * TOP * F) as u64]).unwrap();
        let incumbent = |enc: &KernelEncoder| {
            match dtype {
                GgmlType::IQ3_XXS => crate::metal::encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
                    &ctx, enc, &g, &u, &xt, &counts, &ids, &out, H, F, E, TOP, T,
                ),
                GgmlType::IQ4_XS => crate::metal::encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
                    &ctx, enc, &g, &u, &xt, &counts, &ids, &out, H, F, E, TOP, T,
                ),
                _ => encode_moe_swiglu_f32_grouped_slots_generic(
                    &ctx, enc, &g, &u, &xt, &counts, &ids, &out, H, F, E, TOP, T,
                ),
            }
            .unwrap();
        };
        let tensor = |enc: &KernelEncoder| {
            encode_moe_swiglu_f32_grouped_slots_tensor(
                &ctx, enc, &g, &u, &xt, &counts, &ids, &out, H, F, E, TOP, T, None,
            )
            .unwrap()
        };
        screen(&ctx, &format!("gate_up {dtype:?}"), &incumbent, &tensor);
        one_shot(&ctx, |enc| {
            incumbent(enc);
            Ok(())
        })
        .unwrap();
        let baseline = tensor_f32_at_offset(&out);
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_f32_grouped_slots_tensor(
                &ctx, enc, &g, &u, &xt, &counts, &ids, &out, H, F, E, TOP, T, None,
            )
        })
        .unwrap();
        let values = tensor_f32_at_offset(&out);
        assert!(values.iter().all(|v| v.is_finite()));
        let mut old = Vec::new();
        let mut actual = Vec::new();
        let mut reference = Vec::new();
        for slot in sample_slots {
            let e = r.slot_expert[slot];
            let gw = cpu_expert(dtype, &gate, H, F, e);
            let uw = cpu_expert(dtype, &up, H, F, e);
            let input = &x[slot / TOP * H..(slot / TOP + 1) * H];
            for row in [0, 1, 31, 32, 63, 64, 127, 319, 639] {
                let a = dot64(&gw[row * H..(row + 1) * H], input);
                let b = dot64(&uw[row * H..(row + 1) * H], input);
                reference.push(a / (1.0 + (-a).exp()) * b);
                old.push(baseline[slot * F + row]);
                actual.push(values[slot * F + row]);
            }
        }
        check(&format!("gate_up {dtype:?}"), &old, &actual, &reference);
    }
    for dtype in [GgmlType::IQ4_NL, GgmlType::Q2_0] {
        let bank = bank(dtype, "ffn_down_exps.weight");
        let w = bank_tensor(&ctx, dtype, &bank, F, H, E);
        let mut rng = Lcg(0x9500);
        let x: Vec<f32> = (0..T * TOP * F)
            .map(|i| if i < F { 0.0 } else { rng.signed(0.25) })
            .collect();
        let xt = f32_tensor(&ctx, &x);
        let out = MetalTensor::zeros_f32(&ctx, vec![(T * TOP * H) as u64]).unwrap();
        let incumbent = |enc: &KernelEncoder| {
            if dtype == GgmlType::IQ4_NL {
                crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots(
                    &ctx, enc, &w, &xt, &counts, &ids, &out, F, H, E, T,
                )
            } else {
                encode_moe_down_f32_grouped_slots_generic(
                    &ctx, enc, &w, &xt, &counts, &ids, &out, F, H, E, T,
                )
            }
            .unwrap();
        };
        let tensor = |enc: &KernelEncoder| {
            encode_moe_down_f32_grouped_slots_tensor(
                &ctx, enc, &w, &xt, &counts, &ids, &out, F, H, E, T,
            )
            .unwrap()
        };
        screen(&ctx, &format!("down {dtype:?}"), &incumbent, &tensor);
        one_shot(&ctx, |enc| {
            incumbent(enc);
            Ok(())
        })
        .unwrap();
        let baseline = tensor_f32_at_offset(&out);
        one_shot(&ctx, |enc| {
            encode_moe_down_f32_grouped_slots_tensor(
                &ctx, enc, &w, &xt, &counts, &ids, &out, F, H, E, T,
            )
        })
        .unwrap();
        let values = tensor_f32_at_offset(&out);
        assert!(values.iter().all(|v| v.is_finite()));
        let mut old = Vec::new();
        let mut actual = Vec::new();
        let mut reference = Vec::new();
        for slot in sample_slots {
            let weights = cpu_expert(dtype, &bank, F, H, r.slot_expert[slot]);
            for row in [0, 1, 31, 32, 63, 64, 127, 1279, 2559] {
                reference.push(dot64(
                    &weights[row * F..(row + 1) * F],
                    &x[slot * F..(slot + 1) * F],
                ));
                old.push(baseline[slot * H + row]);
                actual.push(values[slot * H + row]);
            }
        }
        check(&format!("down {dtype:?}"), &old, &actual, &reference);
    }
}

#[test]
#[ignore = "production lease; FLASH_PREFILL_MODEL; real Flash-Next dense banks and M5 tensor tier"]
fn flash_next_tensor_dense_production_f64() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(moe_grouped_tensor_available(&ctx));
    let gguf =
        crate::gguf::GgufFile::open(std::env::var_os("FLASH_PREFILL_MODEL").unwrap()).unwrap();
    let mut cases = std::collections::BTreeMap::<String, Vec<&crate::tensor::TensorDesc>>::new();
    for d in &gguf.tensors {
        if d.name.starts_with("blk.")
            && d.shape.len() == 2
            && matches!(
                d.dtype,
                GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::IQ4_XS | GgmlType::IQ4_NL
            )
            && matches!(
                (d.shape[0], d.shape[1]),
                (2560, 10240 | 6144 | 12288 | 512 | 640) | (6144 | 640, 2560)
            )
        {
            cases
                .entry(format!("{:?}-{}-{}", d.dtype, d.shape[0], d.shape[1]))
                .or_default()
                .push(d);
        }
    }
    for (key, cases) in cases {
        let d = cases[0];
        let (k, n) = (d.shape[0] as usize, d.shape[1] as usize);
        eprintln!(
            "flash_dense_bank {} {key} occurrences={}",
            d.name,
            cases.len()
        );
        let w = MetalTensor::from_bytes(&ctx, gguf.try_slice(d).unwrap(), d.shape.clone(), d.dtype)
            .unwrap();
        let mut rng = Lcg(0x127654);
        let x: Vec<f32> = (0..T * k)
            .map(|i| if i < k { 0.0 } else { rng.signed(0.25) })
            .collect();
        let xt = f32_tensor(&ctx, &x);
        let out = MetalTensor::zeros_f32(&ctx, vec![n as u64, T as u64]).unwrap();
        let a = |enc: &KernelEncoder| {
            crate::metal_forward::encode_mat_mat_dispatch_without_fewrow(
                &ctx, enc, &w, &xt, &out, k, n, T,
            )
            .unwrap()
        };
        let b = |enc: &KernelEncoder| {
            encode_mat_mat_tensor_f32_narrow(&ctx, enc, &w, &xt, &out, k, n, T).unwrap()
        };
        screen(&ctx, &format!("dense {key}"), &a, &b);
        one_shot(&ctx, |enc| {
            a(enc);
            Ok(())
        })
        .unwrap();
        let old = tensor_f32_at_offset(&out);
        one_shot(&ctx, |enc| {
            b(enc);
            Ok(())
        })
        .unwrap();
        let new = tensor_f32_at_offset(&out);
        // Prove the owning production adapter reaches the screened operator.
        one_shot(&ctx, |enc| {
            crate::metal_forward::encode_qwen4exp_prefill_mat_mat(
                &ctx, enc, &w, &xt, &out, k, n, T,
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let production = tensor_f32_at_offset(&out);
        let expected = if crate::env_flag::read_default_on("QWEN_DENSE_PREFILL_TENSOR") {
            &new
        } else {
            &old
        };
        assert!(
            production
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{key} production dispatch"
        );
        assert!(old.iter().chain(&new).all(|x| x.is_finite()));
        let weights = crate::codec::dequant_to_f32(d, gguf.try_slice(d).unwrap()).unwrap();
        let mut av = Vec::new();
        let mut bv = Vec::new();
        let mut reference = Vec::new();
        for t in [0, 1, 31, 32, 63, 127, 1023, 2047] {
            for row in [0, 1, 31, 32, 63, n / 2, n - 1] {
                av.push(old[t * n + row]);
                bv.push(new[t * n + row]);
                reference.push(dot64(
                    &weights[row * k..(row + 1) * k],
                    &x[t * k..(t + 1) * k],
                ));
            }
        }
        check(&format!("dense {key}"), &av, &bv, &reference);
    }
}
