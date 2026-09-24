use super::topk_screen::{allocation, f32s, fixtures, guarded, guards, oracle};
use super::*;

#[test]
fn guarded_topk_geometry_is_narrow() {
    assert!(guarded_topk::eligible(512, 10));
    for (n, k) in [(256, 10), (512, 8), (513, 10), (0, 0)] {
        assert!(!guarded_topk::eligible(n, k));
    }
}

#[test]
#[ignore = "serial production lease; guarded selector finite and nonfinite compatibility"]
fn guarded_topk_compatibility() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let source =
        std::fs::read(parent.join("qwen4exp-native-topk-27023/incumbent-routes-logits.bin"))
            .unwrap();
    let native = source[..512 * 4]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let artifact = parent.join(format!("qwen4exp-topk-compat-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    let ctx = MetalContext::new().unwrap();
    guarded_topk::preflight(&ctx).unwrap();
    let mut cases: Vec<(String, Vec<f32>)> = fixtures(native)
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect();
    for (name, value) in [
        ("negative_inf", f32::NEG_INFINITY),
        ("positive_inf", f32::INFINITY),
        ("all_nan", f32::from_bits(0x7fc01234)),
    ] {
        cases.push((name.to_string(), vec![value; 512]));
    }
    let mut mixed = vec![f32::NEG_INFINITY; 512];
    for (i, value) in mixed[..7].iter_mut().enumerate() {
        *value = i as f32;
    }
    cases.push(("seven_finite".into(), mixed));
    for bits in [
        0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f801234, 0xff805678,
    ] {
        for position in [0, 9, 10, 255, 256, 511] {
            let mut values: Vec<f32> = (0..512).map(|i| (512 - i) as f32).collect();
            values[position] = f32::from_bits(bits);
            cases.push((format!("bits{bits:x}-at{position}"), values));
        }
    }
    let total = cases.len();
    let mut nan_payload_differences = 0;
    for (name, values) in cases {
        let logits = guarded(&ctx, bytemuck::cast_slice(&values), GgmlType::F32);
        let frozen = allocation(&logits);
        let finite = values.iter().all(|v| v.is_finite());
        let mut outputs = Vec::new();
        for candidate in [false, true] {
            let poison = if candidate { -23456.0f32 } else { -12345.0f32 };
            let ids = guarded(&ctx, bytemuck::cast_slice(&[-1i32; 10]), GgmlType::I32);
            let weights = guarded(&ctx, bytemuck::cast_slice(&[poison; 10]), GgmlType::F32);
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            if candidate {
                guarded_topk::encode(&ctx, &enc, &logits, &ids, &weights).unwrap();
            } else {
                crate::metal::encode_topk_logits_softmax_f32(
                    &ctx, &enc, &logits, &ids, &weights, 512, 10,
                )
                .unwrap();
            }
            enc.end();
            command.commit();
            crate::metal::wait_unchecked(&command);
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            std::fs::write(
                artifact.join(format!("{name}-{candidate}-ids.bin")),
                bytes(&ids),
            )
            .unwrap();
            std::fs::write(
                artifact.join(format!("{name}-{candidate}-weights.bin")),
                bytes(&weights),
            )
            .unwrap();
            if finite {
                oracle(&values, &ids, &weights);
            }
            assert!(
                f32s(&weights)
                    .iter()
                    .all(|v| v.to_bits() != poison.to_bits())
            );
            assert_eq!(allocation(&logits), frozen);
            guards(&ids);
            guards(&weights);
            outputs.push((bytes(&ids), f32s(&weights)));
        }
        assert_eq!(outputs[0].0, outputs[1].0, "{name} IDs");
        for (&a, &b) in outputs[0].1.iter().zip(&outputs[1].1) {
            if a.is_nan() && b.is_nan() {
                nan_payload_differences += usize::from(a.to_bits() != b.to_bits());
            } else {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} weights");
            }
        }
    }
    let logits = guarded(&ctx, bytemuck::cast_slice(&[1.0f32; 512]), GgmlType::F32);
    let ids = guarded(&ctx, bytemuck::cast_slice(&[-1i32; 10]), GgmlType::I32);
    let weights = guarded(&ctx, bytemuck::cast_slice(&[f32::NAN; 10]), GgmlType::F32);
    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let mut alias = logits.clone();
    alias.shape = vec![10];
    assert!(guarded_topk::encode(&ctx, &enc, &logits, &ids, &alias).is_err());
    let mut bad = ids.clone();
    bad.dtype = GgmlType::F32;
    assert!(guarded_topk::encode(&ctx, &enc, &logits, &bad, &weights).is_err());
    let mut bad = logits.clone();
    bad.offset += 1;
    assert!(guarded_topk::encode(&ctx, &enc, &bad, &ids, &weights).is_err());
    assert!(crate::metal::dispatch_census_take().is_empty());
    enc.end();
    eprintln!(
        "guarded_topk compatibility PASS cases={total} nan_payload_differences={nan_payload_differences} artifacts={}",
        artifact.display()
    );
}
