//! GLM-5.3-Flash DSA indexer cache maintenance: per-token LayerNorm'd key and
//! gate into an F16 4-slot pending ring, and an F16 pooled key whenever a pool
//! of four consecutive positions completes. Run from token zero so the sparse
//! switch at visible length 2052 needs no recompute.
//!
//! CPU contract: [`crate::glm5_next::oracle::indexer_append_step`].

use super::checks::{bad_shape, check_disjoint, check_tensor, require_serial};
use super::*;

pub const INDEXER_DIM: usize = 128;
pub const INDEXER_POOL: usize = 4;

/// Inputs are F32 `[128]` (raw key, raw gate, LayerNorm weight and bias) and
/// `ape` F32 `[128, 4]`. `pending` is F16 `[128, 2, 4]` (memory `[slot][key |
/// gate][128]`); `pooled` is F16 `[128, pools]` with `pools > position / 4`.
#[allow(clippy::too_many_arguments)]
pub fn encode_indexer_append(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    raw_key: &MetalTensor,
    raw_gate: &MetalTensor,
    norm_weight: &MetalTensor,
    norm_bias: &MetalTensor,
    ape: &MetalTensor,
    pending: &MetalTensor,
    pooled: &MetalTensor,
    position: usize,
    eps: f32,
) -> Result<(), MetalError> {
    encode(
        ctx,
        enc,
        &[
            raw_key,
            raw_gate,
            norm_weight,
            norm_bias,
            ape,
            pending,
            pooled,
        ],
        position,
        None,
        eps,
    )
}

/// [`encode_indexer_append`] for `rows` consecutive tokens starting at
/// `position`, in one dispatch: `raw_key` and `raw_gate` are `[128, rows]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_indexer_append_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    raw_key: &MetalTensor,
    raw_gate: &MetalTensor,
    norm_weight: &MetalTensor,
    norm_bias: &MetalTensor,
    ape: &MetalTensor,
    pending: &MetalTensor,
    pooled: &MetalTensor,
    position: usize,
    rows: usize,
    eps: f32,
) -> Result<(), MetalError> {
    encode(
        ctx,
        enc,
        &[
            raw_key,
            raw_gate,
            norm_weight,
            norm_bias,
            ape,
            pending,
            pooled,
        ],
        position,
        Some(rows),
        eps,
    )
}

fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bindings: &[&MetalTensor; 7],
    position: usize,
    rows: Option<usize>,
    eps: f32,
) -> Result<(), MetalError> {
    const K: &str = "indexer_append";
    let [
        raw_key,
        raw_gate,
        norm_weight,
        norm_bias,
        ape,
        pending,
        pooled,
    ] = *bindings;
    require_serial(K, enc)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(bad_shape(K, "epsilon must be finite and positive"));
    }
    let d = INDEXER_DIM as u64;
    let row_shape = match rows {
        None => vec![d],
        Some(0) => return Err(bad_shape(K, "rows must be positive")),
        Some(rows) => vec![d, rows as u64],
    };
    check_tensor(K, raw_key, GgmlType::F32, &row_shape, false, "raw key")?;
    check_tensor(K, raw_gate, GgmlType::F32, &row_shape, false, "raw gate")?;
    check_tensor(K, norm_weight, GgmlType::F32, &[d], false, "norm weight")?;
    check_tensor(K, norm_bias, GgmlType::F32, &[d], false, "norm bias")?;
    check_tensor(
        K,
        ape,
        GgmlType::F32,
        &[d, INDEXER_POOL as u64],
        false,
        "ape",
    )?;
    check_tensor(
        K,
        pending,
        GgmlType::F16,
        &[d, 2, INDEXER_POOL as u64],
        true,
        "pending",
    )?;
    if pooled.dtype != GgmlType::F16 || pooled.shape.len() != 2 || pooled.shape[0] != d {
        return Err(bad_shape(K, "pooled must be F16 [128, pools]"));
    }
    let pools = pooled.shape[1];
    check_tensor(K, pooled, GgmlType::F16, &[d, pools], true, "pooled")?;
    let last = position + rows.unwrap_or(1) - 1;
    if (last / INDEXER_POOL) as u64 >= pools {
        return Err(bad_shape(
            K,
            format!("position {last} beyond {pools} pools"),
        ));
    }
    let inputs = [
        (raw_key, "raw key"),
        (raw_gate, "raw gate"),
        (norm_weight, "norm weight"),
        (norm_bias, "norm bias"),
        (ape, "ape"),
    ];
    check_disjoint(K, pending, &inputs)?;
    check_disjoint(K, pooled, &inputs)?;
    check_disjoint(K, pooled, &[(pending, "pending")])?;
    let position = u32::try_from(position).map_err(|_| bad_shape(K, "position exceeds u32"))?;
    match rows {
        None => {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct Args {
                position: u32,
                eps: f32,
            }
            let pso = ctx.pipeline("kernel_glm53_indexer_append")?;
            enc.set_pipeline(&pso);
            enc.set_bytes(0, &Args { position, eps });
        }
        Some(rows) => {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct Args {
                position: u32,
                rows: u32,
                eps: f32,
            }
            let pso = ctx.pipeline("kernel_glm53_indexer_append_rows")?;
            enc.set_pipeline(&pso);
            enc.set_bytes(
                0,
                &Args {
                    position,
                    rows: u32::try_from(rows).map_err(|_| bad_shape(K, "rows exceed u32"))?,
                    eps,
                },
            );
        }
    }
    for (index, tensor) in bindings.iter().enumerate() {
        enc.set_tensor(index + 1, tensor);
    }
    enc.set_threadgroup_memory(0, 4 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: INDEXER_DIM,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::offset_tensor;
    use super::*;
    use crate::glm5_next::oracle::indexer_append_step;
    use half::f16;

    fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            16,
            shape,
            GgmlType::F32,
        )
    }

    fn read_f16(tensor: &MetalTensor) -> Vec<f16> {
        let n = tensor.n_elements() as usize;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                n * 2,
            )
        };
        bytemuck::cast_slice::<u8, u16>(bytes)
            .iter()
            .map(|&b| f16::from_bits(b))
            .collect()
    }

    /// Fifteen tokens (pools complete at 4, 8 and 12; tails 1-3 and continuation
    /// across partial pools), rounding-sensitive keys and gates: pending rows and
    /// pooled keys within one F16 ulp of the CPU contract.
    #[test]
    fn indexer_append_matches_contract_over_fifteen_tokens() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const TOKENS: usize = 15;
        let d = INDEXER_DIM;
        let weight: Vec<f32> = (0..d).map(|c| 0.6 + ((c * 7) % 13) as f32 * 0.05).collect();
        let bias: Vec<f32> = (0..d)
            .map(|c| ((c * 5) % 11) as f32 * 0.03 - 0.15)
            .collect();
        let ape: Vec<f32> = (0..d * 4)
            .map(|i| ((i * 13) % 17) as f32 * 0.21 - 1.6)
            .collect();
        let w_t = f32_tensor(&ctx, &weight, vec![d as u64]);
        let b_t = f32_tensor(&ctx, &bias, vec![d as u64]);
        let ape_t = f32_tensor(&ctx, &ape, vec![d as u64, 4]);
        let pending_t = offset_tensor(
            &ctx,
            16,
            &vec![0u8; d * 2 * 4 * 2],
            16,
            vec![d as u64, 2, 4],
            GgmlType::F16,
        );
        let pools = TOKENS.div_ceil(4) + 1;
        let pooled_t = offset_tensor(
            &ctx,
            16,
            &vec![0u8; d * pools * 2],
            16,
            vec![d as u64, pools as u64],
            GgmlType::F16,
        );
        let mut pending = vec![f16::ZERO; d * 2 * 4];
        let mut pooled = vec![f16::ZERO; d * pools];
        for t in 0..TOKENS {
            // Values straddling F16 rounding steps (1/1024 offsets around 1.0-4.0).
            let raw_key: Vec<f32> = (0..d)
                .map(|c| 1.0 + ((c * 31 + t * 17) % 97) as f32 / 1024.0 * (1.0 + (c % 3) as f32))
                .collect();
            let raw_gate: Vec<f32> = (0..d)
                .map(|c| ((c * 23 + t * 41) % 89) as f32 * 0.0731 - 3.0 + 1.0 / 4096.0)
                .collect();
            indexer_append_step(
                t,
                &raw_key,
                &raw_gate,
                &weight,
                &bias,
                &ape,
                1e-6,
                &mut pending,
                &mut pooled,
            );
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode_indexer_append(
                &ctx,
                &enc,
                &f32_tensor(&ctx, &raw_key, vec![d as u64]),
                &f32_tensor(&ctx, &raw_gate, vec![d as u64]),
                &w_t,
                &b_t,
                &ape_t,
                &pending_t,
                &pooled_t,
                t,
                1e-6,
            )
            .unwrap();
            enc.end();
            command.commit();
            wait_completed(&command).unwrap();
            let ulp_close = |a: f16, e: f16| (a.to_bits() as i32 - e.to_bits() as i32).abs() <= 1;
            let gpu_pending = read_f16(&pending_t);
            assert!(
                gpu_pending
                    .iter()
                    .zip(&pending)
                    .all(|(a, e)| ulp_close(*a, *e)),
                "token {t}: pending rows differ"
            );
            let gpu_pooled = read_f16(&pooled_t);
            let complete = (t + 1) / 4;
            assert!(
                gpu_pooled[..complete * d]
                    .iter()
                    .zip(&pooled)
                    .all(|(a, e)| ulp_close(*a, *e)),
                "token {t}: pooled keys differ"
            );
            assert!(
                gpu_pooled[complete * d..].iter().all(|v| v.to_bits() == 0),
                "token {t}: wrote an incomplete pool"
            );
        }
    }

    /// Rows variant: 6 then 9 tokens (crossing pools 1-3 and continuing a
    /// partial pool across calls) equals the single-token contract.
    #[test]
    fn indexer_append_rows_matches_contract_across_calls() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let d = INDEXER_DIM;
        let weight: Vec<f32> = (0..d).map(|c| 0.6 + ((c * 7) % 13) as f32 * 0.05).collect();
        let bias: Vec<f32> = (0..d)
            .map(|c| ((c * 5) % 11) as f32 * 0.03 - 0.15)
            .collect();
        let ape: Vec<f32> = (0..d * 4)
            .map(|i| ((i * 13) % 17) as f32 * 0.21 - 1.6)
            .collect();
        let w_t = f32_tensor(&ctx, &weight, vec![d as u64]);
        let b_t = f32_tensor(&ctx, &bias, vec![d as u64]);
        let ape_t = f32_tensor(&ctx, &ape, vec![d as u64, 4]);
        let pending_t = offset_tensor(
            &ctx,
            16,
            &vec![0u8; d * 2 * 4 * 2],
            16,
            vec![d as u64, 2, 4],
            GgmlType::F16,
        );
        let pools = 5;
        let pooled_t = offset_tensor(
            &ctx,
            16,
            &vec![0u8; d * pools * 2],
            16,
            vec![d as u64, pools as u64],
            GgmlType::F16,
        );
        let mut pending = vec![f16::ZERO; d * 2 * 4];
        let mut pooled = vec![f16::ZERO; d * pools];
        let key = |t: usize| -> Vec<f32> {
            (0..d)
                .map(|c| 1.0 + ((c * 31 + t * 17) % 97) as f32 / 1024.0 * (1.0 + (c % 3) as f32))
                .collect()
        };
        let gate = |t: usize| -> Vec<f32> {
            (0..d)
                .map(|c| ((c * 23 + t * 41) % 89) as f32 * 0.0731 - 3.0)
                .collect()
        };
        let mut position = 0;
        for rows in [6usize, 9] {
            let mut keys = Vec::new();
            let mut gates = Vec::new();
            for t in position..position + rows {
                let (k, g) = (key(t), gate(t));
                indexer_append_step(
                    t,
                    &k,
                    &g,
                    &weight,
                    &bias,
                    &ape,
                    1e-6,
                    &mut pending,
                    &mut pooled,
                );
                keys.extend(k);
                gates.extend(g);
            }
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode_indexer_append_rows(
                &ctx,
                &enc,
                &f32_tensor(&ctx, &keys, vec![d as u64, rows as u64]),
                &f32_tensor(&ctx, &gates, vec![d as u64, rows as u64]),
                &w_t,
                &b_t,
                &ape_t,
                &pending_t,
                &pooled_t,
                position,
                rows,
                1e-6,
            )
            .unwrap();
            enc.end();
            command.commit();
            wait_completed(&command).unwrap();
            position += rows;
            let ulp_close = |a: f16, e: f16| (a.to_bits() as i32 - e.to_bits() as i32).abs() <= 1;
            assert!(
                read_f16(&pending_t)
                    .iter()
                    .zip(&pending)
                    .all(|(a, e)| ulp_close(*a, *e)),
                "pending after {position}"
            );
            let complete = position / 4;
            assert!(
                read_f16(&pooled_t)[..complete * d]
                    .iter()
                    .zip(&pooled)
                    .all(|(a, e)| ulp_close(*a, *e)),
                "pooled after {position}"
            );
        }
    }
}
