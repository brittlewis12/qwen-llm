//! General batched K2 prefill: one command per chunk of consecutive tokens.
//!
//! Projections use the engine's tiled mat-mat dispatch (any admitted weight
//! dtype); grouped RMSNorm, RoPE, KV store and causal attention run once per
//! layer over all chunk rows. Row `r` of a chunk attends to the cache prefix
//! plus chunk rows `0..=r`, which are stored before the attention dispatch.
//! Numerics follow the serial graph except for the projection kernels, whose
//! tiled accumulation order differs from mat-vec (tolerance, not bitwise).

use super::*;
use crate::k2_horizon_metal::{
    encode_full_rope_rows, encode_grouped_norm_rows, encode_online_attention_rows,
    encode_store_kv_rows,
};
use crate::metal_forward::encode_mat_mat_dispatch;

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_chunk(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &ResidentWeights,
    attention: AttentionBackend,
    request: &K2ShortContextPlan,
    tokens: &[TokenPlan<'_>],
    b: &SessionBuffers,
    logits: bool,
    captures: Option<&CaptureArena>,
    interventions: Option<&InterventionArena>,
) -> Result<()> {
    let n = tokens.len();
    if !(2..=crate::k2_horizon_plan::MAX_CHUNK_TOKENS).contains(&n) {
        return Err(invalid("general K2 chunk must hold 2..=256 tokens"));
    }
    let first = &tokens[0];
    let last = &tokens[n - 1];
    if tokens
        .iter()
        .enumerate()
        .any(|(i, t)| t.visible_positions() != first.visible_positions() + i as u32)
    {
        return Err(invalid("general K2 chunk rows must be consecutive"));
    }
    let final_row = b.rows(n - 1, 1)?;
    let project = |weight: &MetalTensor,
                   x: &MetalTensor,
                   y: &MetalTensor,
                   n_in: usize,
                   n_out: usize|
     -> Result<()> {
        if weight.shape != [n_in as u64, n_out as u64]
            || !packed::general_projection_dtype(weight.dtype)
        {
            return Err(invalid(format!(
                "general K2 projection contract: {:?} {:?} for [{n_in}, {n_out}]",
                weight.dtype, weight.shape
            )));
        }
        encode_mat_mat_dispatch(ctx, enc, weight, x, y, n_in, n_out, n)?;
        Ok(())
    };
    let batched_attention = attention == AttentionBackend::Online
        && first.storage() == K2KvStorage::F16
        && last.storage() == K2KvStorage::F16;
    encode_get_rows_f32(ctx, enc, &weights.embedding, &b.id, &b.residual, n, 4096)?;
    for (index, layer) in weights.layers.iter().enumerate() {
        let layer_index = index as u32;
        encode_grouped_norm_rows(
            ctx,
            enc,
            request,
            &b.residual,
            &layer.attention_norm,
            &b.norm,
            n,
        )?;
        project(&layer.query, &b.norm, &b.query, 4096, 4096)?;
        project(&layer.key, &b.norm, &b.key, 4096, 1024)?;
        project(&layer.value, &b.norm, &b.value, 4096, 1024)?;
        encode_full_rope_rows(ctx, enc, first, n, &b.query, &b.key)?;
        if batched_attention {
            encode_store_kv_rows(
                ctx,
                enc,
                first,
                last,
                layer_index,
                &b.cache,
                &b.key,
                &b.value,
            )?;
            encode_online_attention_rows(
                ctx,
                enc,
                first,
                last,
                layer_index,
                &b.cache,
                &b.query,
                &b.attention,
            )?;
        } else {
            // Research-only cache/attention variants (compact Q8 cache,
            // materialized scores) have no row kernel: keep the serial
            // store/attend order per row while projections stay batched.
            encode_serial_attention(ctx, enc, attention, tokens, b, layer_index)?;
        }
        project(
            &layer.attention_output,
            &b.attention,
            &b.projection,
            4096,
            4096,
        )?;
        encode_add_inplace_f32(ctx, enc, &b.residual, &b.projection)?;
        encode_grouped_norm_rows(
            ctx,
            enc,
            request,
            &b.residual,
            &layer.feed_forward_norm,
            &b.norm,
            n,
        )?;
        project(&layer.gate, &b.norm, &b.gate, 4096, 12288)?;
        project(&layer.up, &b.norm, &b.up, 4096, 12288)?;
        encode_silu_mul_f32(ctx, enc, &b.gate, &b.up, &b.gated)?;
        project(&layer.down, &b.gated, &b.projection, 12288, 4096)?;
        encode_add_inplace_f32(ctx, enc, &b.residual, &b.projection)?;
        if let Some(interventions) = interventions {
            interventions.encode(ctx, enc, layer_index, &final_row.residual)?;
        }
        if let Some(captures) = captures {
            captures.encode(ctx, enc, layer_index, &final_row.residual)?;
        }
    }
    if logits {
        encode_readout(ctx, enc, weights, request, &final_row)?;
    }
    Ok(())
}

fn encode_serial_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    attention: AttentionBackend,
    tokens: &[TokenPlan<'_>],
    b: &SessionBuffers,
    layer: u32,
) -> Result<()> {
    let encode_attention = match attention {
        #[cfg(test)]
        AttentionBackend::Materialized => crate::k2_horizon_metal::encode_short_attention,
        AttentionBackend::Online => encode_online_attention,
    };
    for (index, token) in tokens.iter().enumerate() {
        let row = b.rows(index, 1)?;
        let (store, attend) = match token.storage() {
            K2KvStorage::F16 => (encode_store_kv as StoreFn, encode_attention as AttendFn),
            K2KvStorage::Q8_0 => (
                crate::k2_horizon_metal::compact::encode_store as StoreFn,
                crate::k2_horizon_metal::compact::encode_attention as AttendFn,
            ),
        };
        store(ctx, enc, token, layer, &b.cache, &row.key, &row.value)?;
        attend(ctx, enc, token, layer, &b.cache, &row.query, &row.attention)?;
    }
    Ok(())
}

type StoreFn = fn(
    &MetalContext,
    &KernelEncoder,
    &TokenPlan<'_>,
    u32,
    &MetalTensor,
    &MetalTensor,
    &MetalTensor,
) -> std::result::Result<(), MetalError>;
type AttendFn = StoreFn;

#[cfg(test)]
mod tests;
