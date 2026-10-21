//! Bitwise witnesses for the fused decode expert chain. Gate/up share each
//! activation load and one epilogue applies SwiGLU and the route weight; down
//! sums slots in order and adds the shared branch. Both retain the native lane
//! partition, per-lane FMA order and SIMD reduction.

use super::*;

const ROUTES: [i32; 5] = [2, 0, 2, -1, EXPERTS as i32];
// The invalid slots exercise the zeroed native result times a negative weight.
const ROUTE_WEIGHTS: [f32; 5] = [-0.5, -0.25, 1.75, -1.5, 0.125];

fn chain_supported(dtype: GgmlType) -> bool {
    all_slots_gate_up_weight_native_supported(dtype, dtype)
        && all_slots_down_sum_native_supported(dtype)
}

#[test]
fn fused_gate_up_weight_matches_native_passes_bitwise() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in formats()
        .into_iter()
        .filter(|&dtype| chain_supported(dtype))
    {
        for &(k, m) in &[(256usize, 5usize), (5120, 3)] {
            let gate = bank(&ctx, &bank_bytes(dtype, k, m, 3), dtype, k, m);
            let up = bank(&ctx, &bank_bytes(dtype, k, m, 7), dtype, k, m);
            let x = values(k, 9);
            let input = f32_view(&ctx, &x, &[k], 0);
            let ids = i32_view(&ctx, &ROUTES);
            let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
            let weights = f32_view(&ctx, &ROUTE_WEIGHTS, &[ROUTES.len()], 8);
            let clamp = 0.37f32;

            let native = f32_view(
                &ctx,
                &vec![f32::NAN; m * ROUTES.len()],
                &[m, ROUTES.len()],
                20,
            );
            one_shot(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu_native(
                    &ctx,
                    enc,
                    &gate,
                    &up,
                    &input,
                    &ids,
                    &status,
                    &native,
                    k,
                    m,
                    EXPERTS,
                    ROUTES.len(),
                    clamp,
                )
            })
            .unwrap();
            let native = tensor_f32_at_offset(&native);

            let fused = f32_view(
                &ctx,
                &vec![f32::NAN; m * ROUTES.len()],
                &[m, ROUTES.len()],
                28,
            );
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu_weight_native(
                    &ctx,
                    enc,
                    &gate,
                    &up,
                    &input,
                    &ids,
                    &status,
                    &weights,
                    &fused,
                    k,
                    m,
                    EXPERTS,
                    ROUTES.len(),
                    clamp,
                )
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert!(census[0].kernel.contains("gate_up_weight"), "{dtype:?}");
            let fused = tensor_f32_at_offset(&fused);
            for slot in 0..ROUTES.len() {
                for row in 0..m {
                    let index = slot * m + row;
                    let expected = native[index] * ROUTE_WEIGHTS[slot];
                    assert_eq!(
                        fused[index].to_bits(),
                        expected.to_bits(),
                        "{dtype:?} k={k} slot={slot} row={row}"
                    );
                }
            }
        }
    }
}

#[test]
fn fused_down_sum_matches_native_down_then_sum_bitwise() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let sum = ctx.pipeline("kernel_deepseek_v41_sum").unwrap();
    for dtype in formats()
        .into_iter()
        .filter(|&dtype| chain_supported(dtype))
    {
        for &(k, m) in &[(256usize, 5usize), (5120, 3)] {
            let down = bank(&ctx, &bank_bytes(dtype, k, m, 11), dtype, k, m);
            let x = values(k * ROUTES.len(), 13);
            let input = f32_view(&ctx, &x, &[k, ROUTES.len()], 0);
            let ids = i32_view(&ctx, &ROUTES);
            let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
            let shared_v = values(m, 17);
            let shared = f32_view(&ctx, &shared_v, &[m], 4);

            let slots = f32_view(
                &ctx,
                &vec![f32::NAN; m * ROUTES.len()],
                &[m, ROUTES.len()],
                20,
            );
            one_shot(&ctx, |enc| {
                encode_all_slots_down_native(
                    &ctx,
                    enc,
                    &down,
                    &input,
                    &ids,
                    &status,
                    &slots,
                    k,
                    m,
                    EXPERTS,
                    ROUTES.len(),
                )
            })
            .unwrap();

            let native = f32_view(&ctx, &vec![f32::NAN; m], &[m], 28);
            one_shot(&ctx, |enc| {
                enc.set_pipeline(&sum);
                enc.set_bytes(0, &(m as u32));
                enc.set_bytes(1, &(ROUTES.len() as u32));
                enc.set_tensor(2, &slots);
                enc.set_tensor(3, &shared);
                enc.set_tensor(4, &native);
                enc.set_tensor(5, &status);
                enc.set_bytes(6, &1u32);
                enc.dispatch(
                    MTLSize {
                        width: m.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                Ok(())
            })
            .unwrap();
            let native = tensor_f32_at_offset(&native);

            let fused = f32_view(&ctx, &vec![f32::NAN; m], &[m], 36);
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                encode_all_slots_down_sum_native(
                    &ctx,
                    enc,
                    &down,
                    &input,
                    &ids,
                    &status,
                    &shared,
                    &fused,
                    k,
                    m,
                    EXPERTS,
                    ROUTES.len(),
                )
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert!(census[0].kernel.contains("down_sum"), "{dtype:?}");
            let fused = tensor_f32_at_offset(&fused);
            for row in 0..m {
                assert_eq!(
                    fused[row].to_bits(),
                    native[row].to_bits(),
                    "{dtype:?} k={k} row={row}"
                );
            }
        }
    }
}
