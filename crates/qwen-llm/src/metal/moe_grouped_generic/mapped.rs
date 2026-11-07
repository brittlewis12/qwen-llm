use super::*;

/// Private transactional route: counts/IDs must own unique slots, with total
/// population <= tokens * topk. Dead routed-down output stores the tile map;
/// the serial encoder consumes it before routed-down overwrites that storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_moe_swiglu_mapped32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    input: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    dead: &MetalTensor,
    hidden: usize,
    ffn: usize,
    experts: usize,
    topk: usize,
    tokens: usize,
) -> Result<bool, MetalError> {
    const TILE_ROWS: usize = 32;
    let name = match gate.dtype {
        GgmlType::IQ2_S => "kernel_qwen4exp_expert_iq2_s_mapped32",
        GgmlType::IQ3_S => "kernel_qwen4exp_expert_iq3_s_mapped32",
        _ => return Ok(false),
    };
    if gate.dtype != up.dtype || experts != 512 {
        return Ok(false);
    }
    let prep_name = "kernel_qwen4exp_expert_tile_map32";
    for (name, threads, dynamic) in [(prep_name, 512, 0), (name, 128, 16384)] {
        match ctx.pipeline(name) {
            Ok(p)
                if p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= threads
                    && p.staticThreadgroupMemoryLength()
                        .checked_add(dynamic)
                        .is_some_and(|v| v <= ctx.device.maxThreadgroupMemoryLength()) => {}
            Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => {
                return Ok(false);
            }
            Err(e) => return Err(e),
        }
    }
    const K: &str = "fn_expert_mapped";
    use crate::metal::checks::{bad_shape, check_disjoint, check_physical, require_serial};
    require_serial(K, enc)?;
    let product = |a: usize, b: usize| {
        a.checked_mul(b)
            .filter(|&v| v > 0 && v <= i32::MAX as usize)
            .ok_or_else(|| bad_shape(K, "positive i32-sized dimensions/products required"))
    };
    let slots = product(tokens, topk)?;
    let input_len = product(tokens, hidden)?;
    let ids_len = product(tokens, experts)?;
    let out_len = product(slots, ffn)?;
    let bound = slots
        .div_ceil(TILE_ROWS)
        .checked_add(experts)
        .ok_or_else(|| bad_shape(K, "map bound overflow"))?;
    let words = product(bound, 2)?;
    let l = generic_layout_for(K, gate, hidden, ffn, experts)?;
    generic_layout_for(K, up, hidden, ffn, experts)?;
    if input.n_elements() != input_len as u64
        || counts.n_elements() != experts as u64
        || ids.n_elements() != ids_len as u64
        || out.n_elements() != out_len as u64
    {
        return Err(bad_shape(K, "input/counts/IDs/output size mismatch"));
    }
    check_f32x_bindings(
        ctx,
        K,
        name,
        16384,
        &[gate, up],
        input,
        counts,
        ids,
        out,
        &[
            (Some(slots), "slots"),
            (Some(ids_len), "ids"),
            (Some(out_len), "output"),
        ],
    )?;
    if dead.dtype != GgmlType::F32 || dead.n_elements() < words as u64 {
        return Err(bad_shape(K, "insufficient writable F32 dead scratch"));
    }
    check_physical(K, dead, 8, true, "tile map scratch")?;
    let map = dead.view_subrange(0, vec![words as u64]);
    check_disjoint(
        K,
        &map,
        &[
            (gate, "gate"),
            (up, "up"),
            (input, "input"),
            (counts, "counts"),
            (ids, "ids"),
            (out, "output"),
        ],
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        experts: u32,
        topk: u32,
        tokens: u32,
        row_bytes: u32,
        stride: u32,
        min: u32,
        max: u32,
    }
    let args = Args {
        ffn: ffn as u32,
        hidden: hidden as u32,
        experts: experts as u32,
        topk: topk as u32,
        tokens: tokens as u32,
        row_bytes: ((hidden / l.block_elems) * l.block_bytes) as u32,
        stride: hidden as u32,
        min: 0,
        max: i32::MAX as u32,
    };
    let prep = ctx.pipeline(prep_name)?;
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&prep);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, counts);
    enc.set_tensor(2, &map);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    for (i, t) in [
        (1, gate),
        (2, up),
        (3, input),
        (4, counts),
        (5, ids),
        (6, out),
        (7, &map),
    ] {
        enc.set_tensor(i, t);
    }
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: bound,
            height: ffn.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
    #[test]
    fn mapped_expert_tiles_preserve_sparse_hot_tail_and_empty_routes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let (h, f, e, t, top) = (256usize, 80usize, 512usize, 33usize, 3usize);
        for (dtype, bsize) in [(GgmlType::IQ2_S, 82), (GgmlType::IQ3_S, 110)] {
            let bank = |salt: usize| {
                let mut bytes: Vec<u8> = (0..f * e * bsize)
                    .map(|i| (i * 71 + i / 17 + salt) as u8)
                    .collect();
                for (i, b) in bytes.chunks_exact_mut(bsize).enumerate() {
                    b[..2].copy_from_slice(
                        &half::f16::from_f32((i % 7 + 1) as f32 / 4096.)
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
                offset_tensor(
                    &ctx,
                    16,
                    &bytes,
                    16,
                    vec![h as u64, f as u64, e as u64],
                    dtype,
                )
            };
            let gate = bank(13);
            let up = bank(97);
            let xv: Vec<f32> = (0..h * t).map(|i| (i % 43) as f32 / 64. - 0.3).collect();
            let x = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&xv),
                16,
                vec![(h * t) as u64],
                GgmlType::F32,
            );
            for empty in [false, true] {
                let tile_rows = 32;
                let mut c = vec![0i32; e];
                let mut ids = vec![-1i32; e * t];
                if !empty {
                    for slot in 0..t * top {
                        let expert = match slot % top {
                            0 => 0,
                            1 => 2 + slot % (e - 3),
                            _ => e - 1,
                        };
                        ids[expert * t + c[expert] as usize] = slot as i32;
                        c[expert] += 1;
                    }
                }
                c[1] = -1;
                c[e - 2] = t as i32 + 1;
                let expected: Vec<[i32; 2]> = c
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c > 0 && **c <= t as i32)
                    .flat_map(|(e, c)| {
                        (0..(*c as usize).div_ceil(tile_rows)).map(move |j| [e as i32, j as i32])
                    })
                    .collect();
                let c = offset_tensor(
                    &ctx,
                    16,
                    bytemuck::cast_slice(&c),
                    16,
                    vec![e as u64],
                    GgmlType::I32,
                );
                let ids = offset_tensor(
                    &ctx,
                    16,
                    bytemuck::cast_slice(&ids),
                    16,
                    vec![(e * t) as u64],
                    GgmlType::I32,
                );
                let output = || {
                    offset_tensor(
                        &ctx,
                        16,
                        bytemuck::cast_slice(&vec![f32::from_bits(0x7fc01234); t * top * f]),
                        16,
                        vec![(t * top * f) as u64],
                        GgmlType::F32,
                    )
                };
                let old = output();
                let new = output();
                let words = 2 * ((t * top).div_ceil(tile_rows) + e);
                let scratch = offset_tensor(
                    &ctx,
                    16,
                    bytemuck::cast_slice(&vec![f32::NAN; words]),
                    16,
                    vec![words as u64],
                    GgmlType::F32,
                );
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                encode_moe_swiglu_f32_grouped_slots_generic(
                    &ctx, &enc, &gate, &up, &x, &c, &ids, &old, h, f, e, top, t,
                )
                .unwrap();
                crate::metal::dispatch_census_begin();
                assert!(
                    try_encode_moe_swiglu_mapped32(
                        &ctx, &enc, &gate, &up, &x, &c, &ids, &new, &scratch, h, f, e, top, t,
                    )
                    .unwrap()
                );
                let calls = crate::metal::dispatch_census_take();
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[1].grid_width, (words / 2) as u64);
                assert_eq!(calls[1].grid_height, 2);
                assert_eq!(calls[1].grid_depth, 1);
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                let a = tensor_f32_at_offset(&old);
                let b = tensor_f32_at_offset(&new);
                assert_eq!(
                    bytemuck::cast_slice::<f32, u32>(&a),
                    bytemuck::cast_slice::<f32, u32>(&b)
                );
                assert!(b.iter().all(|v| if empty {
                    v.to_bits() == 0x7fc01234
                } else {
                    v.is_finite()
                }));
                let map = tensor_f32_at_offset(&scratch);
                let map: Vec<[i32; 2]> = bytemuck::cast_slice::<f32, i32>(&map)
                    .chunks_exact(2)
                    .map(|v| [v[0], v[1]])
                    .collect();
                assert_eq!(&map[..expected.len()], expected.as_slice());
                assert!(map[expected.len()..].iter().all(|v| *v == [-1, 0]));
                for z in [&gate, &up, &x, &c, &ids, &old, &new, &scratch] {
                    assert_offset_guards(z, 16, 16);
                }
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                crate::metal::dispatch_census_begin();
                let short = scratch.view_subrange(0, vec![(words - 1) as u64]);
                assert!(
                    try_encode_moe_swiglu_mapped32(
                        &ctx, &enc, &gate, &up, &x, &c, &ids, &new, &short, h, f, e, top, t,
                    )
                    .is_err()
                );
                let alias = new.view_subrange(0, vec![words as u64]);
                assert!(
                    try_encode_moe_swiglu_mapped32(
                        &ctx, &enc, &gate, &up, &x, &c, &ids, &new, &alias, h, f, e, top, t,
                    )
                    .is_err()
                );
                assert!(crate::metal::dispatch_census_take().is_empty());
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin_concurrent(&cmd);
                crate::metal::dispatch_census_begin();
                assert!(
                    try_encode_moe_swiglu_mapped32(
                        &ctx, &enc, &gate, &up, &x, &c, &ids, &new, &scratch, h, f, e, top, t,
                    )
                    .is_err()
                );
                assert!(crate::metal::dispatch_census_take().is_empty());
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
            }
        }
    }
}
