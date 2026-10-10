//! The 32-token grouped gate/up tile against the 16-token one: per slot,
//! bitwise equal outputs over bucket counts on both sides of every tile edge,
//! permuted slots and both GLM gate/up dtypes; untouched slots and guards
//! intact.

use super::super::test_support::{assert_offset_guards, offset_tensor};
use super::tests::{
    EXPERTS, FFN, HIDDEN, TOP_K, bank, f32_tensor, read_f32, run, synthetic_bank, values,
};
use super::*;

#[test]
fn wide_gate_up_tile_matches_the_16_token_tile_bitwise() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const ROWS: usize = 100;
    // Bucket sizes around the 16- and 32-token tile edges.
    let counts: [usize; EXPERTS] = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 0];
    let slot_limit = ROWS * TOP_K;
    // A fixed permutation of every slot; buckets take disjoint runs of it.
    let mut order: Vec<i32> = (0..slot_limit as i32).collect();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let mut ids = vec![-1i32; EXPERTS * ROWS];
    let mut next = 0;
    for (expert, &count) in counts.iter().enumerate() {
        ids[expert * ROWS..expert * ROWS + count].copy_from_slice(&order[next..next + count]);
        next += count;
    }
    let counts_i32: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
    let counts_t = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&counts_i32),
        16,
        vec![EXPERTS as u64],
        GgmlType::I32,
    );
    let ids_t = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&ids),
        16,
        vec![(EXPERTS * ROWS) as u64],
        GgmlType::I32,
    );
    let x = f32_tensor(
        &ctx,
        &values(HIDDEN * ROWS, 6),
        vec![(HIDDEN * ROWS) as u64],
    );
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
        let gate = bank(
            &ctx,
            &synthetic_bank(dtype, HIDDEN, FFN, 3),
            dtype,
            HIDDEN,
            FFN,
        );
        let up = bank(
            &ctx,
            &synthetic_bank(dtype, HIDDEN, FFN, 5),
            dtype,
            HIDDEN,
            FFN,
        );
        let run_with = |wide: Option<u32>| {
            let inner = f32_tensor(
                &ctx,
                &vec![7.0; FFN * slot_limit],
                vec![(FFN * slot_limit) as u64],
            );
            run(&ctx, |enc| {
                match wide {
                None => encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                    &ctx, enc, &gate, &up, &x, &counts_t, &ids_t, &inner, HIDDEN, FFN, EXPERTS,
                    TOP_K, ROWS, 10.0,
                )
                .unwrap(),
                Some(min) => super::super::moe_grouped_generic::encode_moe_swiglu_clamped_f32_grouped_slots_split(
                    &ctx, enc, &gate, &up, &x, &counts_t, &ids_t, &inner, HIDDEN, FFN, EXPERTS,
                    TOP_K, ROWS, 10.0, min,
                )
                .unwrap(),
            }
            });
            assert_offset_guards(&inner, 16, 20);
            read_f32(&inner)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let narrow = run_with(None);
        // Every bucket wide, buckets from 17 wide, and from 33 wide.
        for min in [1u32, 17, 33] {
            assert_eq!(run_with(Some(min)), narrow, "{dtype:?} wide from {min}");
        }
        // Slots no bucket names keep their initial value.
        let named: std::collections::HashSet<i32> =
            ids.iter().copied().filter(|&s| s >= 0).collect();
        let values = read_f32_bits(&narrow);
        for slot in 0..slot_limit as i32 {
            if !named.contains(&slot) {
                assert!(
                    values[slot as usize * FFN..(slot as usize + 1) * FFN]
                        .iter()
                        .all(|&v| v == 7.0)
                );
            }
        }
    }
}

fn read_f32_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}
