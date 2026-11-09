use super::*;
use crate::metal::test_support::*;

#[test]
fn members_match_cooperative_bits_shifted_buffers_and_odd_outputs() {
    let _lease = crate::metal::acquire_metal_test_lease().expect("GPU test campaign lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (k, m) in [(12usize, 1usize), (36, 3), (4096, 288), (4096, 289)] {
        let w: Vec<_> = (0..k * m)
            .map(|i| (((i * 13 + 7) % 89) as f32 - 44.) * 0.007)
            .collect();
        let wt = offset_tensor(
            &ctx,
            32,
            bytemuck::cast_slice(&w),
            16,
            vec![k as u64, m as u64],
            GgmlType::F32,
        );
        for b in [2, 3, 4] {
            let x: Vec<_> = (0..k * b)
                .map(|i| (((i * 37 + 11) % 97) as f32 - 48.) * 0.01)
                .collect();
            let xt = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&x),
                16,
                vec![k as u64, b as u64],
                GgmlType::F32,
            );
            let a = offset_tensor(
                &ctx,
                12,
                bytemuck::cast_slice(&vec![f32::NAN; m * b]),
                20,
                vec![m as u64, b as u64],
                GgmlType::F32,
            );
            let y = offset_tensor(
                &ctx,
                12,
                bytemuck::cast_slice(&vec![f32::NAN; m * b]),
                20,
                vec![m as u64, b as u64],
                GgmlType::F32,
            );
            let immutable = [tensor_backing_bytes(&wt), tensor_backing_bytes(&xt)];
            one_shot(&ctx, |e| {
                for row in 0..b {
                    #[repr(C)]
                    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
                    struct A {
                        k: u32,
                        m: u32,
                    }
                    e.set_pipeline(&ctx.pipeline("kernel_mat_vec_f32_f32_lcpp_r2")?);
                    e.set_bytes(
                        0,
                        &A {
                            k: k as u32,
                            m: m as u32,
                        },
                    );
                    e.set_tensor(1, &wt);
                    e.set_tensor(2, &xt.view_subrange((row * k) as u64, vec![k as u64]));
                    e.set_tensor(3, &a.view_subrange((row * m) as u64, vec![m as u64]));
                    e.set_threadgroup_memory(0, 256);
                    e.dispatch(
                        MTLSize {
                            width: m.div_ceil(2),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                assert!(try_encode_mat_vec_f32_members(
                    &ctx, e, &wt, &xt, &y, k, m, b
                )?);
                Ok(())
            })
            .unwrap();
            let av = tensor_f32_at_offset(&a);
            let bv = tensor_f32_at_offset(&y);
            assert!(av.iter().chain(&bv).all(|v| v.is_finite()));
            assert!(av.iter().zip(&bv).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_offset_guards(&a, 12, 20);
            assert_offset_guards(&y, 12, 20);
            assert_eq!(
                immutable,
                [tensor_backing_bytes(&wt), tensor_backing_bytes(&xt)]
            );
            one_shot(&ctx, |e| {
                assert!(try_encode_mat_vec_f32_members(&ctx, e, &wt, &xt, &xt, k, m, b).is_err());
                assert!(try_encode_mat_vec_f32_members(&ctx, e, &wt, &xt, &y, k, m, 1).is_err());
                Ok(())
            })
            .unwrap();
        }
    }
}
