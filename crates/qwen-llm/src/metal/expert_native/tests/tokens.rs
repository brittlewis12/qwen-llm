//! The token-batched decode entrypoints
//! (`encode_all_slots_gate_up_swiglu_native_fused_tokens`,
//! `encode_all_slots_down_native_tokens`) equal one single-token dispatch per
//! token bitwise for every native storage type: ready and rejected tokens,
//! invalid IDs, and an odd channel count.

use super::*;

#[test]
fn native_token_rows_equal_one_dispatch_per_token_bitwise() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
    let top_k = IDS.len();
    let tokens = 3usize;
    let statuses = [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING, ROUTE_STATUS_READY];
    for dtype in formats() {
        let (block, _) = dtype.storage_layout().unwrap();
        let k = (block as usize).max(256) * 3;
        let m = 7usize;
        let gate = bank(&ctx, &bank_bytes(dtype, k, m, 3), dtype, k, m);
        let up = bank(&ctx, &bank_bytes(dtype, k, m, 4), dtype, k, m);
        let down = bank(&ctx, &bank_bytes(dtype, k, m, 5), dtype, k, m);
        let input = f32_view(&ctx, &values(k * tokens, 7), &[k, tokens], 0);
        let down_input = f32_view(
            &ctx,
            &values(k * top_k * tokens, 9),
            &[k, top_k * tokens],
            0,
        );
        // Different routes per token, including invalid IDs.
        let id_values: Vec<i32> = (0..tokens)
            .flat_map(|t| {
                IDS.iter()
                    .map(move |&id| if t == 1 { id } else { (id + t as i32) % 5 - 1 })
            })
            .collect();
        let ids = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&id_values),
            0,
            vec![top_k as u64, tokens as u64],
            GgmlType::I32,
        );
        let status = i32_view(&ctx, &statuses);
        let slots = top_k * tokens;
        let dims = (k, m, EXPERTS, top_k);
        for down_pass in [false, true] {
            let batched = f32_view(&ctx, &vec![f32::NAN; m * slots], &[m, slots], 0);
            let single = f32_view(&ctx, &vec![f32::NAN; m * slots], &[m, slots], 0);
            one_shot(&ctx, |enc| {
                if down_pass {
                    encode_all_slots_down_native_tokens(
                        &ctx,
                        enc,
                        &down,
                        &down_input,
                        &ids,
                        &status,
                        &batched,
                        dims,
                        tokens,
                    )?;
                } else {
                    encode_all_slots_gate_up_swiglu_native_fused_tokens(
                        &ctx, enc, &gate, &up, &input, &ids, &status, &batched, dims, tokens, 0.37,
                    )?;
                }
                for t in 0..tokens {
                    let ids_t = ids.view_bytes((t * top_k * 4) as u64, vec![top_k as u64]);
                    let status_t = status.view_bytes((t * 4) as u64, vec![1]);
                    let out_t =
                        single.view_bytes((t * m * top_k * 4) as u64, vec![m as u64, top_k as u64]);
                    if down_pass {
                        encode_all_slots_down_native(
                            &ctx,
                            enc,
                            &down,
                            &down_input.view_bytes(
                                (t * k * top_k * 4) as u64,
                                vec![k as u64, top_k as u64],
                            ),
                            &ids_t,
                            &status_t,
                            &out_t,
                            k,
                            m,
                            EXPERTS,
                            top_k,
                        )?;
                    } else {
                        encode_all_slots_gate_up_swiglu_native_fused(
                            &ctx,
                            enc,
                            &gate,
                            &up,
                            &input.view_bytes((t * k * 4) as u64, vec![k as u64]),
                            &ids_t,
                            &status_t,
                            &out_t,
                            k,
                            m,
                            EXPERTS,
                            top_k,
                            0.37,
                        )?;
                    }
                }
                Ok(())
            })
            .unwrap();
            let bits = |t: &MetalTensor| {
                tensor_f32_at_offset(t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            let (b, s) = (bits(&batched), bits(&single));
            assert!(b.iter().any(|&v| f32::from_bits(v) != 0.0), "{dtype:?}");
            assert_eq!(b, s, "{dtype:?} down {down_pass}");
        }
    }
}
