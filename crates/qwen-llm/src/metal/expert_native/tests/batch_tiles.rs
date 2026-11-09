use super::*;

#[test]
fn batch_expert_tile_scope_preserves_other_shapes_and_unaligned_inputs() {
    for (ty, k, m, fused) in [
        (GgmlType::IQ2_S, 4096, 2048, true),
        (GgmlType::IQ3_S, 2048, 4096, false),
        (GgmlType::IQ2_S, 2560, 640, true),
        (GgmlType::IQ3_S, 2560, 640, true),
        (GgmlType::IQ3_XXS, 2560, 640, true),
        (GgmlType::IQ4_XS, 2560, 640, true),
        (GgmlType::Q2_0, 640, 2560, false),
        (GgmlType::IQ4_NL, 640, 2560, false),
        (GgmlType::Q8_0, 640, 2560, false),
    ] {
        for offset in [0, 16, 32] {
            assert!(batch_tile_pipeline(ty, k, m, offset, fused).is_some());
        }
        for offset in [4, 12, 20] {
            assert!(batch_tile_pipeline(ty, k, m, offset, fused).is_none());
        }
        assert!(batch_tile_pipeline(ty, k, m - 1, 0, fused).is_none());
        assert!(batch_tile_pipeline(ty, k + 256, m, 0, fused).is_none());
        assert!(batch_tile_pipeline(ty, k, m, 0, !fused).is_none());
        assert!(batch_tile_pipeline(GgmlType::F32, k, m, 0, fused).is_none());
    }
}

#[test]
fn batch_expert_tiles_match_generic_bits_routes_offsets_and_guards() {
    let _lease = test_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (ty, k, m, fused) in [
        (GgmlType::IQ2_S, 4096, 2048, true),
        (GgmlType::IQ3_S, 2048, 4096, false),
        (GgmlType::IQ2_S, 2560, 640, true),
        (GgmlType::IQ3_S, 2560, 640, true),
        (GgmlType::IQ3_XXS, 2560, 640, true),
        (GgmlType::IQ4_XS, 2560, 640, true),
        (GgmlType::Q2_0, 640, 2560, false),
        (GgmlType::IQ4_NL, 640, 2560, false),
        (GgmlType::Q8_0, 640, 2560, false),
    ] {
        let wb = bank_bytes(ty, k, m, 3);
        let ub = bank_bytes(ty, k, m, 4);
        let w = bank(&ctx, &wb, ty, k, m);
        let u = bank(&ctx, &ub, ty, k, m);
        for n in [1, 2, 3, 4] {
            for offset in [16, 20] {
                let elements = if fused { k * n } else { k * IDS.len() * n };
                let x = values(elements, 7);
                let input = offset_tensor(
                    &ctx,
                    offset,
                    bytemuck::cast_slice(&x),
                    16,
                    vec![k as u64, if fused { n } else { IDS.len() * n } as u64],
                    GgmlType::F32,
                );
                let iv: Vec<_> = (0..n).flat_map(|_| IDS).collect();
                let ids = offset_tensor(
                    &ctx,
                    12,
                    bytemuck::cast_slice(&iv),
                    16,
                    vec![IDS.len() as u64, n as u64],
                    GgmlType::I32,
                );
                let sv: Vec<i32> = (0..n)
                    .map(|r| if r == 2 { -7 } else { ROUTE_STATUS_READY })
                    .collect();
                let status = i32_view(&ctx, &sv);
                let size = m * IDS.len() * n;
                let a = offset_tensor(
                    &ctx,
                    28,
                    bytemuck::cast_slice(&vec![f32::NAN; size]),
                    20,
                    vec![m as u64, (IDS.len() * n) as u64],
                    GgmlType::F32,
                );
                let b = offset_tensor(
                    &ctx,
                    28,
                    bytemuck::cast_slice(&vec![f32::NAN; size]),
                    20,
                    vec![m as u64, (IDS.len() * n) as u64],
                    GgmlType::F32,
                );
                let immutable = [&w, &u, &input, &ids, &status];
                let before: Vec<_> = immutable.iter().map(|t| tensor_backing_bytes(t)).collect();
                dispatch_census_begin();
                one_shot(&ctx, |enc| {
                    if fused {
                        encode_gate_up_with_batch_tile(
                            &ctx,
                            enc,
                            &w,
                            &u,
                            &input,
                            &ids,
                            &status,
                            &a,
                            k,
                            m,
                            EXPERTS,
                            IDS.len(),
                            7.0,
                            Some(n),
                            NativeOperands::F32,
                            true,
                            false,
                            false,
                        )?;
                        encode_gate_up_with_batch_tile(
                            &ctx,
                            enc,
                            &w,
                            &u,
                            &input,
                            &ids,
                            &status,
                            &b,
                            k,
                            m,
                            EXPERTS,
                            IDS.len(),
                            7.0,
                            Some(n),
                            NativeOperands::F32,
                            true,
                            false,
                            true,
                        )
                    } else {
                        encode_down_with_batch_tile(
                            &ctx,
                            enc,
                            &w,
                            &input,
                            &ids,
                            &status,
                            &a,
                            k,
                            m,
                            EXPERTS,
                            IDS.len(),
                            Some(n),
                            NativeOperands::F32,
                            false,
                            false,
                        )?;
                        encode_down_with_batch_tile(
                            &ctx,
                            enc,
                            &w,
                            &input,
                            &ids,
                            &status,
                            &b,
                            k,
                            m,
                            EXPERTS,
                            IDS.len(),
                            Some(n),
                            NativeOperands::F32,
                            false,
                            true,
                        )
                    }
                })
                .unwrap();
                let census = dispatch_census_take();
                assert_eq!(census.len(), 2);
                let selected = n > 1 && offset % 16 == 0;
                if selected {
                    let (name, _) = batch_tile_pipeline(ty, k, m, offset as u64, fused).unwrap();
                    assert_eq!(census[1].kernel, name);
                } else {
                    assert!(!census[1].kernel.contains("_r4"));
                }
                let av = tensor_f32_at_offset(&a);
                let bv = tensor_f32_at_offset(&b);
                assert!(av.iter().chain(&bv).all(|v| v.is_finite()));
                assert!(
                    av.iter().zip(&bv).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{ty:?} B={n} offset={offset}"
                );
                for r in 0..n {
                    for (slot, id) in IDS.iter().enumerate() {
                        if sv[r] != ROUTE_STATUS_READY || *id < 0 || *id as usize >= EXPERTS {
                            assert!(
                                bv[(r * IDS.len() + slot) * m..(r * IDS.len() + slot + 1) * m]
                                    .iter()
                                    .all(|v| v.to_bits() == 0)
                            );
                        }
                    }
                }
                assert_offset_guards(&a, 28, 20);
                assert_offset_guards(&b, 28, 20);
                for (t, before) in immutable.iter().zip(before) {
                    assert_eq!(tensor_backing_bytes(t), before);
                }
            }
        }
    }
}
