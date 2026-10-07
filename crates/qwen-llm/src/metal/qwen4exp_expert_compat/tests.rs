use super::*;
use crate::tensor::TensorDesc;
use objc2_metal::MTLCommandQueue;

fn bank(k: usize, rows: usize, experts: usize, mut state: u32) -> Vec<u8> {
    let mut bytes = vec![0; k / 256 * rows * experts * 82];
    for (ordinal, block) in bytes.chunks_exact_mut(82).enumerate() {
        let magnitude = (ordinal % 7 + 1) as f32 / 512.0;
        let scale = if ordinal % 3 == 0 {
            -magnitude
        } else {
            magnitude
        };
        block[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for byte in &mut block[2..] {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *byte = (state >> 24) as u8;
        }
    }
    bytes
}

fn tensor(
    ctx: &MetalContext,
    bytes: &[u8],
    prefix: usize,
    shape: Vec<u64>,
    dtype: GgmlType,
) -> MetalTensor {
    let mut backing = vec![0xA5; prefix];
    backing.extend_from_slice(bytes);
    backing.extend_from_slice(&[0x5A; 28]);
    MetalTensor {
        buffer: ctx.buffer_from(&backing).unwrap(),
        offset: prefix as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn backing(t: &MetalTensor) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    }
}

fn decoded_expert(bytes: &[u8], k: usize, rows: usize, expert: usize) -> Vec<f32> {
    let stride = k / 256 * rows * 82;
    let desc = TensorDesc {
        name: "iq2_s_independent_oracle".into(),
        shape: vec![k as u64, rows as u64],
        dtype: GgmlType::IQ2_S,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: stride as u64,
    };
    // GGML's CPU codec owns a separate decode implementation and lookup table.
    crate::codec::dequant_to_f32(&desc, &bytes[expert * stride..(expert + 1) * stride]).unwrap()
}

fn run_case(k: usize, rows: usize, experts: usize, routes: &[i32]) {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let gate_bytes = bank(k, rows, experts, 0x13572468);
    let up_bytes = bank(k, rows, experts, 0xFEDCBA98);
    let x: Vec<f32> = (0..k)
        .map(|j| ((j * 17 + j / 11) % 79) as f32 / 37.0 - 1.0)
        .collect();
    let gate = tensor(
        &ctx,
        &gate_bytes,
        18,
        vec![k as u64, rows as u64, experts as u64],
        GgmlType::IQ2_S,
    );
    let up = tensor(&ctx, &up_bytes, 34, gate.shape.clone(), GgmlType::IQ2_S);
    let input = tensor(
        &ctx,
        bytemuck::cast_slice(&x),
        12,
        vec![k as u64],
        GgmlType::F32,
    );
    let mut expected = Vec::new();
    let mut differs_from_clamp = false;
    for &expert in routes {
        if expert < 0 || expert as usize >= experts {
            expected.extend(std::iter::repeat_n(0.0, rows));
            continue;
        }
        let g = decoded_expert(&gate_bytes, k, rows, expert as usize);
        let u = decoded_expert(&up_bytes, k, rows, expert as usize);
        for row in 0..rows {
            let dot = |w: &[f32]| -> f64 {
                w[row * k..(row + 1) * k]
                    .iter()
                    .zip(&x)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            };
            let gv = dot(&g);
            let uv = dot(&u);
            let result = gv / (1.0 + (-gv).exp()) * uv;
            let clipped_g = gv.min(7.0);
            let clipped = clipped_g / (1.0 + (-clipped_g).exp()) * uv.clamp(-7.0, 7.0);
            differs_from_clamp |= (result - clipped).abs() > 0.1;
            expected.push(result);
        }
    }
    if k >= 1280 {
        assert!(
            differs_from_clamp,
            "fixture must distinguish the unclamped epilogue"
        );
    }
    for dtype in [GgmlType::I32, GgmlType::F32] {
        let ids = tensor(
            &ctx,
            bytemuck::cast_slice(routes),
            20,
            vec![routes.len() as u64],
            dtype,
        );
        let output = tensor(
            &ctx,
            bytemuck::cast_slice(&vec![f32::NAN; routes.len() * rows]),
            28,
            vec![rows as u64, routes.len() as u64],
            GgmlType::F32,
        );
        // Odd row counts deliberately exercise the helper's partial eight-row group.
        let geometry = Qwen4ExpMoeMetalGeometry {
            hidden_size: k,
            expert_count: experts,
            experts_per_token: routes.len(),
            routed_intermediate_size: rows,
            shared_intermediate_size: 32,
        };
        let weights = Qwen4ExpMoeMetalWeights {
            geometry,
            routed_gate: &gate,
            routed_up: &up,
            router: &input,
            routed_down: &input,
            shared_router: &input,
            shared_gate: &input,
            shared_up: &input,
            shared_down: &input,
        };
        let buffers = Qwen4ExpMoeSingletonBuffers {
            guarded_topk: false,
            topk_ids: &ids,
            routed_inner: &output,
            router_logits: &input,
            topk_weights: &input,
            shared_scale: &input,
            routed_expert_output: &input,
            shared_inner: &input,
            shared_output: &input,
            output: &input,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode(&ctx, &enc, &input, weights, buffers).unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        let raw = backing(&output);
        for (i, &e) in expected.iter().enumerate() {
            let start = output.offset as usize + i * 4;
            let actual = f32::from_le_bytes(raw[start..start + 4].try_into().unwrap()) as f64;
            assert!(
                actual.is_finite() && e.is_finite(),
                "nonfinite at {i}: {actual} vs {e}"
            );
            assert!(
                (actual - e).abs() <= 2e-4 * (1.0 + e.abs()),
                "K={k}, rows={rows}, dtype={dtype:?}, element={i}: {actual} vs {e}"
            );
        }
        assert!(raw[..output.offset as usize].iter().all(|&b| b == 0xA5));
        assert!(raw[raw.len() - 28..].iter().all(|&b| b == 0x5A));
        assert_eq!(
            &backing(&ids)[20..20 + routes.len() * 4],
            bytemuck::cast_slice::<i32, u8>(routes)
        );
    }
    for (t, bytes) in [
        (&gate, gate_bytes.as_slice()),
        (&up, up_bytes.as_slice()),
        (&input, bytemuck::cast_slice(&x)),
    ] {
        let raw = backing(t);
        assert_eq!(
            &raw[t.offset as usize..t.offset as usize + bytes.len()],
            bytes
        );
        assert!(raw[..t.offset as usize].iter().all(|&b| b == 0xA5));
        assert!(raw[raw.len() - 28..].iter().all(|&b| b == 0x5A));
    }
}

#[test]
fn iq2_s_singleton_f64_high_experts_tail_and_invalid_routes() {
    run_case(256, 9, 512, &[511, 257, 0, 511, -1, 512, i32::MAX]);
}

#[test]
fn iq2_s_singleton_f64_strided_k_loop_and_row_tails() {
    run_case(1280, 33, 7, &[6, 1, 4, 6, 0]);
}

#[test]
fn iq2_s_singleton_f64_flash_next_widths_unclamped() {
    run_case(2560, 640, 3, &[2, 0, 2]);
}

#[test]
fn iq2_s_grouped_swiglu_512_expert_dispatch_matches_f64() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const K: usize = 256;
    const ROWS: usize = 33;
    const EXPERTS: usize = 512;
    const TOKENS: usize = 2;
    const TOPK: usize = 2;
    let routes = [511usize, 257, 0, 511];
    let gate_bytes = bank(K, ROWS, EXPERTS, 0x12345678);
    let up_bytes = bank(K, ROWS, EXPERTS, 0xA9876543);
    let shape = vec![K as u64, ROWS as u64, EXPERTS as u64];
    let gate = tensor(&ctx, &gate_bytes, 18, shape.clone(), GgmlType::IQ2_S);
    let up = tensor(&ctx, &up_bytes, 34, shape, GgmlType::IQ2_S);
    let x: Vec<f32> = (0..K * TOKENS)
        .map(|j| ((j * 29 + j / 13) % 83) as f32 / 64.0 - 0.625)
        .collect();
    let input = tensor(
        &ctx,
        bytemuck::cast_slice(&x),
        16,
        vec![K as u64, TOKENS as u64],
        GgmlType::F32,
    );
    let mut counts = vec![0i32; EXPERTS];
    let mut slots = vec![-1i32; EXPERTS * TOKENS];
    for (slot, &expert) in routes.iter().enumerate().rev() {
        slots[expert * TOKENS + counts[expert] as usize] = slot as i32;
        counts[expert] += 1;
    }
    let counts_t = tensor(
        &ctx,
        bytemuck::cast_slice(&counts),
        12,
        vec![EXPERTS as u64],
        GgmlType::I32,
    );
    let slots_t = tensor(
        &ctx,
        bytemuck::cast_slice(&slots),
        20,
        vec![TOKENS as u64, EXPERTS as u64],
        GgmlType::I32,
    );
    let output = tensor(
        &ctx,
        bytemuck::cast_slice(&vec![f32::NAN; ROWS * TOKENS * TOPK]),
        28,
        vec![ROWS as u64, (TOKENS * TOPK) as u64],
        GgmlType::F32,
    );
    // Even this tiny payload launches 128 * 512 threads, crossing the ushort
    // builtin limit that aborted the real GSQ run under Metal validation.
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    crate::metal::encode_moe_swiglu_f32_grouped_slots_generic_range(
        &ctx,
        &enc,
        &gate,
        &up,
        &input,
        &counts_t,
        &slots_t,
        &output,
        K,
        ROWS,
        EXPERTS,
        TOPK,
        TOKENS,
        0,
        i32::MAX as u32,
    )
    .unwrap();
    enc.end();
    crate::metal::commit_and_wait(&command).unwrap();
    let raw = backing(&output);
    for (slot, &expert) in routes.iter().enumerate() {
        let g = decoded_expert(&gate_bytes, K, ROWS, expert);
        let u = decoded_expert(&up_bytes, K, ROWS, expert);
        let input_row = &x[(slot / TOPK) * K..(slot / TOPK + 1) * K];
        for row in 0..ROWS {
            let dot = |w: &[f32]| -> f64 {
                w[row * K..(row + 1) * K]
                    .iter()
                    .zip(input_row)
                    .map(|(&w, &x)| {
                        // Generic MMA stages both operands in half.
                        half::f16::from_f32(w).to_f64() * half::f16::from_f32(x).to_f64()
                    })
                    .sum()
            };
            let g = dot(&g);
            let u = dot(&u);
            let expected = g / (1.0 + (-g).exp()) * u;
            let start = output.offset as usize + (slot * ROWS + row) * 4;
            let actual = f32::from_le_bytes(raw[start..start + 4].try_into().unwrap()) as f64;
            assert!(actual.is_finite() && expected.is_finite());
            assert!(
                (actual - expected).abs() <= 3e-4 * (1.0 + expected.abs()),
                "expert={expert} slot={slot} row={row}: {actual} vs {expected}"
            );
        }
    }
    assert!(raw[..output.offset as usize].iter().all(|&b| b == 0xA5));
    assert!(raw[raw.len() - 28..].iter().all(|&b| b == 0x5A));
}
