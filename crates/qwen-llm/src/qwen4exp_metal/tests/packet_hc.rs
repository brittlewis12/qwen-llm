use super::*;
use crate::metal::MetalTensorProvenance;
use objc2_metal::MTLCommandQueue;
use packet_hc::{Mode, Role, Scope};

#[test]
fn packet_hc_scope_is_thread_local_and_unwind_safe() {
    assert_eq!(packet_hc::active_mode(), Mode::Baseline);
    let outer = Scope::new(Mode::InjectRows);
    let failure = std::panic::catch_unwind(|| {
        let _inner = Scope::new(Mode::MixFewrow);
        assert_eq!(packet_hc::active_mode(), Mode::MixFewrow);
        assert_eq!(
            std::thread::spawn(packet_hc::active_mode).join().unwrap(),
            Mode::Baseline
        );
        panic!("scoped diagnostic unwind");
    });
    assert!(failure.is_err());
    assert_eq!(packet_hc::active_mode(), Mode::InjectRows);
    drop(outer);
    assert_eq!(packet_hc::active_mode(), Mode::Baseline);
}

#[test]
fn packet_hc_selection_and_pipeline_limits() {
    for rows in [1, 2, 3, 4, 8, 9, 128] {
        for role in [Role::Down, Role::Up, Role::Inject] {
            for dtype in [GgmlType::F32, GgmlType::Q8_0, GgmlType::BF16] {
                assert!(!packet_hc::selected(Mode::Baseline, role, dtype, rows));
                let eligible = matches!(rows, 2 | 4 | 8);
                assert_eq!(
                    packet_hc::selected(Mode::MixFewrowInjectRows, role, dtype, rows),
                    eligible
                        && dtype
                            == if role == Role::Inject {
                                GgmlType::F32
                            } else {
                                GgmlType::Q8_0
                            }
                );
            }
        }
    }
    assert!(packet_hc::pipeline_supported(32, 1024, 1024, 16384, 32768));
    assert!(!packet_hc::pipeline_supported(16, 1024, 1024, 16384, 32768));
    assert!(!packet_hc::pipeline_supported(32, 128, 256, 16384, 32768));
    assert!(!packet_hc::pipeline_supported(32, 1024, 1024, 32769, 32768));
}

fn guarded(ctx: &MetalContext, bytes: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let mut backing = vec![0xa5; 32];
    backing.extend_from_slice(bytes);
    backing.extend_from_slice(&[0x5a; 32]);
    MetalTensor {
        buffer: ctx.buffer_from(&backing).unwrap(),
        offset: 32,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn bytes(t: &MetalTensor) -> &[u8] {
    // These test allocations are shared and are read only after command completion.
    unsafe { std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast(), t.buffer.length()) }
}

#[test]
#[ignore = "synthetic HC leaves; exclusive production GPU lease; no model"]
fn packet_hc_offset_leaves_and_scope_restore() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    for (role, k, m, dtype) in [
        (Role::Inject, 10240, 4, GgmlType::F32),
        (Role::Down, 10240, 320, GgmlType::Q8_0),
        (Role::Up, 320, 10240, GgmlType::Q8_0),
    ] {
        let weights: Vec<f32> = (0..k * m).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        let mut quant = Vec::new();
        for block in weights.chunks_exact(32) {
            quant.extend_from_slice(&half::f16::from_f32(1.0 / 32.0).to_le_bytes());
            quant.extend(block.iter().map(|w| (w * 32.0) as i8 as u8));
        }
        let storage = if dtype == GgmlType::F32 {
            bytemuck::cast_slice(&weights)
        } else {
            &quant
        };
        let weight = guarded(&ctx, storage, vec![k as u64, m as u64], dtype);
        for rows in [2, 4, 8] {
            let input: Vec<f32> = (0..k * rows)
                .map(|i| ((i * 7 % 17) as f32 - 8.0) / 16.0)
                .collect();
            let x = guarded(
                &ctx,
                bytemuck::cast_slice(&input),
                vec![k as u64, rows as u64],
                GgmlType::F32,
            );
            let y = guarded(
                &ctx,
                bytemuck::cast_slice(&vec![f32::NAN; m * rows]),
                vec![m as u64, rows as u64],
                GgmlType::F32,
            );
            let w_before = bytes(&weight).to_vec();
            let x_before = bytes(&x).to_vec();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            let outer = Scope::new(Mode::MixFewrowInjectRows);
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _inner = Scope::new(Mode::Baseline);
                assert!(!packet_hc::encode(&ctx, &enc, role, &weight, &x, &y, k, m, rows).unwrap());
                panic!("scope restoration witness");
            }));
            assert!(panic.is_err());
            let short = y.view_subrange(0, vec![(m * rows - 1) as u64]);
            assert!(packet_hc::encode(&ctx, &enc, role, &weight, &x, &short, k, m, rows).is_err());
            assert!(packet_hc::encode(&ctx, &enc, role, &weight, &x, &y, k, m, rows).unwrap());
            drop(outer);
            assert!(!packet_hc::encode(&ctx, &enc, role, &weight, &x, &y, k, m, rows).unwrap());
            enc.end();
            command.commit();
            crate::metal::wait_completed(&command).unwrap();
            assert_eq!(bytes(&weight), w_before);
            assert_eq!(bytes(&x), x_before);
            let actual: &[f32] = bytemuck::cast_slice(&bytes(&y)[32..32 + m * rows * 4]);
            for row in 0..rows {
                for out in 0..m {
                    let expected: f64 = (0..k)
                        .map(|i| f64::from(input[row * k + i]) * f64::from(weights[out * k + i]))
                        .sum();
                    // Dyadic fixtures have exactly representable products and sums.
                    assert_eq!(
                        f64::from(actual[row * m + out]),
                        expected,
                        "{role:?}/{rows}/{row}/{out}"
                    );
                }
            }
            assert!(bytes(&y)[..32].iter().all(|&b| b == 0xa5));
            assert!(bytes(&y)[32 + m * rows * 4..].iter().all(|&b| b == 0x5a));
        }
    }
}
