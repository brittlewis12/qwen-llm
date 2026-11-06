use super::*;
use crate::metal::context::acquire_metal_benchmark_lease;
use crate::metal::test_support::{
    cpu_matrix_attn_reference, f32_differential, metal_test_context, tensor_f32_at_offset,
};

#[test]
fn depth_layout_keeps_unqualified_routes() {
    let eligible = |m5, heads, pos, nwg, c, half, scratch, tuning| {
        decode_attn_depth_layout(m5, heads, pos, nwg, c, half, scratch, tuning)
    };
    for pos in [32768, 32769, 65537, 98303] {
        assert!(eligible(
            true,
            (24, 4, 256),
            pos,
            128,
            32,
            true,
            true,
            false
        ));
    }
    for pos in [1, 32767, 98304, 131073] {
        assert!(!eligible(
            true,
            (24, 4, 256),
            pos,
            128,
            32,
            true,
            true,
            false
        ));
    }
    for (m5, heads, nwg, c, half, scratch, tuning) in [
        (false, (24, 4, 256), 128, 32, true, true, false),
        (true, (16, 2, 256), 128, 32, true, true, false),
        (true, (24, 4, 256), 512, 32, true, true, false),
        (true, (24, 4, 256), 128, 64, true, true, false),
        (true, (24, 4, 256), 128, 32, false, true, false),
        (true, (24, 4, 256), 128, 32, true, false, false),
        (true, (24, 4, 256), 128, 32, true, true, true),
    ] {
        assert!(!eligible(m5, heads, 32769, nwg, c, half, scratch, tuning));
    }
}

#[test]
fn depth_register_queries_and_fallback_are_bitwise() {
    let _lease = acquire_metal_benchmark_lease().unwrap();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let mut state = 0x57a8_c921u64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 16_777_216. * 2. - 1.) * 1.7320508
    };
    for pos in [129usize, 32769, 65537, 98303, 98304, 131073] {
        let mut q: Vec<f32> = (0..24 * 256).map(|_| random()).collect();
        // Uniform attention and peaked scores, plus unaligned KV/partition tails.
        q[..256].fill(0.);
        q[256] = 96.;
        q[257] = -96.;
        let k: Vec<u16> = (0..pos * 1024)
            .map(|_| half::f16::from_f32(random()).to_bits())
            .collect();
        let v: Vec<u16> = (0..k.len())
            .map(|_| half::f16::from_f32(random()).to_bits())
            .collect();
        let make = |bytes: &[u8], n: usize, dtype| {
            MetalTensor::from_bytes(&ctx, bytes, vec![n as u64], dtype).unwrap()
        };
        let qt = make(bytemuck::cast_slice(&q), q.len(), GgmlType::F32);
        let kt = make(bytemuck::cast_slice(&k), k.len(), GgmlType::F16);
        let vt = make(bytemuck::cast_slice(&v), v.len(), GgmlType::F16);
        let scratch = vec![f32::NAN; 24 * 1024 * 256];
        let partial = make(bytemuck::cast_slice(&scratch), scratch.len(), GgmlType::F32);
        let ml = make(
            bytemuck::cast_slice(&scratch[..24 * 1024 * 2]),
            24 * 1024 * 2,
            GgmlType::F32,
        );
        let out = MetalTensor::zeros_f32(&ctx, vec![q.len() as u64]).unwrap();
        let nwg = attn_v4_choose_nwg(pos, 6);
        let run = |register: bool, production: bool, partitions: usize| {
            with_attn_v4_register_q_override(register, || {
                one_shot(&ctx, |enc| {
                    let encode = if production {
                        encode_attn_decode_depth_f32
                    } else {
                        encode_attn_decode_v4_f32
                    };
                    encode(
                        &ctx, enc, &qt, &kt, &vt, &partial, &ml, &out, 24, 4, 256, pos, partitions,
                        32,
                    )
                })
                .unwrap();
            });
            let result = tensor_f32_at_offset(&out);
            assert!(result.iter().all(|x| x.is_finite()));
            result
        };
        let incumbent = run(false, false, nwg);
        let exact = run(true, false, nwg);
        assert!(
            incumbent
                .iter()
                .zip(&exact)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "register Q pos={pos}"
        );
        assert_eq!(incumbent, run(false, true, nwg));
        let promoted = run(true, true, nwg);
        let selected = ctx.device_facts().name == "Apple M5 Ultra"
            && (32768..98304).contains(&pos)
            && with_attn_v4_register_q_override(true, decode_attn_depth_enabled);
        // The production route equals its exact selected raw operator; all
        // unsupported devices/depths preserve the incumbent bit patterns.
        if pos == 32769 {
            // The shared CPU oracle accumulates QK, softmax and PV in F64.
            // Compare the changed partition order against the same stored KV.
            let kf: Vec<f32> = k
                .iter()
                .map(|&x| half::f16::from_bits(x).to_f32())
                .collect();
            let vf: Vec<f32> = v
                .iter()
                .map(|&x| half::f16::from_bits(x).to_f32())
                .collect();
            let reference = cpu_matrix_attn_reference(&q, &kf, &vf, 1, pos - 1, pos, 24, 4, 6, 256);
            let candidate = run(true, false, 256);
            let (inc_rms, inc_peak, _) = f32_differential(&incumbent, &reference);
            let (rms, peak, _) = f32_differential(&candidate, &reference);
            assert!(
                rms <= 1.5 * inc_rms && peak <= 1.5 * inc_peak,
                "F64 bound: RMS {rms}/{inc_rms}; peak {peak}/{inc_peak}"
            );
        }
        let expected = if selected {
            run(true, false, 256)
        } else {
            incumbent
        };
        assert!(
            promoted
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "route pos={pos}"
        );
    }
}

#[test]
fn depth_override_restores_after_unwind() {
    let before = ATTN_REGISTER_Q_OVERRIDE.with(|s| s.get());
    let _ = std::panic::catch_unwind(|| {
        with_attn_v4_register_q_override(true, || {
            with_attn_v4_register_q_override(false, || {
                assert_eq!(ATTN_REGISTER_Q_OVERRIDE.with(|s| s.get()), Some(false))
            });
            assert_eq!(ATTN_REGISTER_Q_OVERRIDE.with(|s| s.get()), Some(true));
            panic!("unwind witness");
        });
    });
    assert_eq!(ATTN_REGISTER_Q_OVERRIDE.with(|s| s.get()), before);
}
