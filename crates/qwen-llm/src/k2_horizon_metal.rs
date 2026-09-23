//! Unqualified K2 primitive bridge; no runtime dispatch or model ownership.
//!
//! Serial encoders only. Callers must discard the command buffer on encoding
//! failure, and must not commit session state until successful execution/readout.
//! These wrappers validate physical views, not cached-prefix identity/freshness,
//! value finiteness, or a complete forward graph. Primitive checks do not qualify
//! a checkpoint or promote a runtime dispatch path.

use crate::k2_horizon::K2KvStorage;
use crate::k2_horizon_plan::{FullNeoxRope, K2ShortContextPlan, TokenPlan};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, encode_attn_decode_f16kv_f32,
    encode_rms_norm_mul_f32, encode_rope_neox_pair_f32, encode_scatter_offset_f32_to_f16,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};
use std::ops::Range;

const KERNEL: &str = "k2_primitive_bridge";
type Result<T> = std::result::Result<T, MetalError>;

pub(crate) mod compact;

fn invalid(detail: impl Into<String>) -> MetalError {
    MetalError::BadShape {
        kernel: KERNEL,
        detail: detail.into(),
    }
}

struct View<'a> {
    allocation: usize,
    buffer_bytes: u64,
    offset: u64,
    shape: &'a [u64],
    dtype: GgmlType,
    writable: bool,
}

impl<'a> From<&'a MetalTensor> for View<'a> {
    fn from(tensor: &'a MetalTensor) -> Self {
        Self {
            allocation: Retained::as_ptr(&tensor.buffer) as *const () as usize,
            buffer_bytes: tensor.buffer.length() as u64,
            offset: tensor.offset,
            shape: &tensor.shape,
            dtype: tensor.dtype,
            writable: tensor.is_writable(),
        }
    }
}

struct CheckedView {
    allocation: usize,
    bytes: Range<u64>,
}

impl View<'_> {
    fn check(&self, shape: &[u64], dtype: GgmlType, write: bool) -> Result<CheckedView> {
        if self.shape != shape || self.dtype != dtype || (write && !self.writable) {
            return Err(invalid("wrong shape, dtype, or write access"));
        }
        let size = match dtype {
            GgmlType::I8 => 1,
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 => 2,
            _ => {
                return Err(invalid(
                    "only I32 IDs, F32 activations and F16 cache are supported",
                ));
            }
        };
        let elements = shape
            .iter()
            .try_fold(1u64, |count, &width| count.checked_mul(width));
        let bytes = elements
            .and_then(|elements| elements.checked_mul(size))
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| invalid("empty or overflowing shape"))?;
        let end = self
            .offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("byte range overflow"))?;
        if !self.offset.is_multiple_of(size) || end > self.buffer_bytes {
            return Err(invalid("unaligned or out-of-buffer view"));
        }
        Ok(CheckedView {
            allocation: self.allocation,
            bytes: self.offset..end,
        })
    }
}

fn disjoint(left: &CheckedView, right: &CheckedView) -> Result<()> {
    if left.allocation == right.allocation
        && left.bytes.start < right.bytes.end
        && right.bytes.start < left.bytes.end
    {
        return Err(invalid("overlapping tensor views"));
    }
    Ok(())
}

fn serial(enc: &KernelEncoder) -> Result<()> {
    if enc.is_concurrent() {
        return Err(invalid("K2 bridge requires an ordered serial encoder"));
    }
    Ok(())
}

pub(crate) fn checked_slice(
    tensor: &MetalTensor,
    offset: u64,
    shape: Vec<u64>,
    dtype: GgmlType,
) -> Result<MetalTensor> {
    View::from(tensor).check(&tensor.shape, dtype, true)?;
    let elements = shape
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("slice shape overflow"))?;
    if elements == 0
        || offset
            .checked_add(elements)
            .is_none_or(|end| end > tensor.n_elements())
    {
        return Err(invalid("slice outside owned activation"));
    }
    let view = tensor.view_subrange(offset, shape.clone());
    View::from(&view).check(&shape, dtype, true)?;
    Ok(view)
}

fn batch_projection_views(
    weight: View<'_>,
    x: View<'_>,
    y: View<'_>,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    if !(1..=crate::k2_horizon_plan::PACKED_CHUNK_TOKENS).contains(&rows)
        || !matches!(
            (n_in, n_out),
            (4096, 4096) | (4096, 1024) | (4096, 12288) | (12288, 4096)
        )
        || weight.dtype != GgmlType::Q8_0
        || weight.shape != [n_in as u64, n_out as u64]
        || !weight.offset.is_multiple_of(2)
    {
        return Err(invalid("unsupported K2 batched projection contract"));
    }
    let end = weight
        .offset
        .checked_add((n_in * n_out / 32 * 34) as u64)
        .filter(|&end| end <= weight.buffer_bytes)
        .ok_or_else(|| invalid("Q8 projection weight outside buffer"))?;
    let w = CheckedView {
        allocation: weight.allocation,
        bytes: weight.offset..end,
    };
    let x = x.check(&[n_in as u64, rows as u64], GgmlType::F32, false)?;
    let y = y.check(&[n_out as u64, rows as u64], GgmlType::F32, true)?;
    disjoint(&w, &x)?;
    disjoint(&w, &y)?;
    disjoint(&x, &y)
}

/// Token-axis Q8 GEMV, preserving singleton lcpp arithmetic, not half-staged GEMM.
pub(crate) fn encode_q8_projection_batch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    serial(enc)?;
    batch_projection_views(
        View::from(weight),
        View::from(x),
        View::from(y),
        n_in,
        n_out,
        rows,
    )?;
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return Err(invalid("K2 packed Q8 requires singleton lcpp arithmetic"));
    }
    enc.note_read(weight);
    enc.note_read(x);
    enc.note_write(y);
    crate::metal::encode_mat_vec_q8_0_batch_f32(ctx, enc, weight, x, y, n_in, n_out, rows)
}

/// Four zero-copy slices, with gamma indexed in full hidden coordinates.
/// Out-of-place only; no shared-gamma per-head normalization is substituted.
pub fn encode_grouped_norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    plan: &K2ShortContextPlan,
    x: &MetalTensor,
    gamma: &MetalTensor,
    out: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let x_view = View::from(x).check(&[4096], GgmlType::F32, false)?;
    let gamma_view = View::from(gamma).check(&[4096], GgmlType::F32, false)?;
    let out_view = View::from(out).check(&[4096], GgmlType::F32, true)?;
    disjoint(&x_view, &out_view)?;
    disjoint(&gamma_view, &out_view)?;
    enc.note_read(x);
    enc.note_read(gamma);
    enc.note_write(out);
    for group in plan.norm_groups() {
        let start = group.elements.start;
        let shape = vec![group.elements.end - start];
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &x.view_subrange(start, shape.clone()),
            &gamma.view_subrange(start, shape.clone()),
            &out.view_subrange(start, shape),
            group.epsilon,
        )?;
    }
    Ok(())
}

/// Q/K shape uses channel-fast [128, heads], with full NEOX rotation only.
pub fn encode_full_rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    query: &MetalTensor,
    key: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let q = View::from(query).check(&[128, 32], GgmlType::F32, true)?;
    let k = View::from(key).check(&[128, 8], GgmlType::F32, true)?;
    disjoint(&q, &k)?;
    enc.note_write(query);
    enc.note_write(key);
    let rope = token.rope();
    encode_rope_neox_pair_f32(
        ctx,
        enc,
        query,
        key,
        FullNeoxRope::QUERY_HEADS,
        FullNeoxRope::KV_HEADS,
        FullNeoxRope::HEAD_DIM,
        FullNeoxRope::ROTARY_DIM,
        rope.position(),
        rope.theta(),
    )
}

fn arena_view(arena: &MetalTensor, token: &TokenPlan<'_>, write: bool) -> Result<CheckedView> {
    if token.storage() != K2KvStorage::F16 {
        return Err(invalid("F16 encoder requires an F16 cache plan"));
    }
    View::from(arena).check(&[token.arena_bytes() / 2], GgmlType::F16, write)
}

fn plane(arena: &MetalTensor, bytes: Range<u64>) -> MetalTensor {
    arena.view_subrange(
        bytes.start / 2,
        vec![128, 8, (bytes.end - bytes.start) / 2048],
    )
}

/// Store post-RoPE K and projected V, rounded by the existing F32-to-F16 kernel.
/// A successful encode is not a committed cache row or proof of finite values.
pub fn encode_store_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let cache = arena_view(arena, token, true)?;
    let k = View::from(key).check(&[128, 8], GgmlType::F32, false)?;
    let v = View::from(value).check(&[128, 8], GgmlType::F32, false)?;
    disjoint(&cache, &k)?;
    disjoint(&cache, &v)?;
    let ranges = token
        .write_ranges(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let k_dst = plane(arena, ranges.key);
    let v_dst = plane(arena, ranges.value);
    enc.note_read(key);
    enc.note_read(value);
    enc.note_write(&k_dst);
    enc.note_write(&v_dst);
    encode_scatter_offset_f32_to_f16(ctx, enc, key, &k_dst, 0, 1024)?;
    encode_scatter_offset_f32_to_f16(ctx, enc, value, &v_dst, 0, 1024)
}

/// The caller must have stored the current row and all preceding rows for this
/// layer/model/session. Logical plans alone cannot establish that invariant.
pub fn encode_short_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    query: &MetalTensor,
    out: &MetalTensor,
) -> Result<()> {
    if token.visible_positions() > crate::k2_horizon_plan::MATERIALIZED_POSITION_CEILING {
        return Err(invalid(
            "materialized K2 attention exceeds 7168-position score scratch",
        ));
    }
    let (k, v) = attention_views(enc, token, layer, arena, query, out)?;
    encode_attn_decode_f16kv_f32(
        ctx,
        enc,
        query,
        &k,
        &v,
        out,
        32,
        8,
        128,
        token.visible_positions() as usize,
    )
}

fn attention_views(
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    query: &MetalTensor,
    out: &MetalTensor,
) -> Result<(MetalTensor, MetalTensor)> {
    serial(enc)?;
    let cache = arena_view(arena, token, false)?;
    let q = View::from(query).check(&[128, 32], GgmlType::F32, false)?;
    let y = View::from(out).check(&[128, 32], GgmlType::F32, true)?;
    disjoint(&cache, &q)?;
    disjoint(&cache, &y)?;
    disjoint(&q, &y)?;
    let ranges = token
        .read_ranges(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let k = plane(arena, ranges.key);
    let v = plane(arena, ranges.value);
    enc.note_read(query);
    enc.note_read(&k);
    enc.note_read(&v);
    enc.note_write(out);
    Ok((k, v))
}

fn online_alignment(query: u64, key: u64, value: u64, output: u64) -> Result<()> {
    if !query.is_multiple_of(16)
        || !output.is_multiple_of(16)
        || !key.is_multiple_of(8)
        || !value.is_multiple_of(8)
    {
        return Err(invalid(
            "online attention requires aligned float4/half4 views",
        ));
    }
    Ok(())
}

/// Register-only H128/GQA4 attention; no history-sized score scratch.
pub fn encode_online_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    query: &MetalTensor,
    out: &MetalTensor,
) -> Result<()> {
    let (key, value) = attention_views(enc, token, layer, arena, query, out)?;
    online_alignment(query.offset, key.offset, value.offset, out.offset)?;
    let pipeline = ctx.pipeline(if token.visible_positions() <= 256 {
        "kernel_k2_attn_online_f16kv_h128"
    } else {
        "kernel_k2_attn_online_blocked_f16kv_h128"
    })?;
    if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
        return Err(invalid("online attention requires a 32-thread SIMDgroup"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        scale: f32,
    }
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            n_pos: token.visible_positions(),
            kv_stride: 1024,
            scale: 128.0_f32.sqrt().recip(),
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, &key);
    enc.set_tensor(3, &value);
    enc.set_tensor(4, out);
    enc.dispatch(
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Row-major `[rows, hidden]` activation views for the general batched path.
fn rows_view(tensor: &MetalTensor, width: u64, rows: usize, write: bool) -> Result<CheckedView> {
    let elements = width
        .checked_mul(rows as u64)
        .ok_or_else(|| invalid("row view overflow"))?;
    View::from(tensor).check(&[elements], GgmlType::F32, write)
}

fn chunk_rows(first: &TokenPlan<'_>, last: &TokenPlan<'_>) -> Result<usize> {
    let rows = last
        .visible_positions()
        .checked_sub(first.visible_positions())
        .and_then(|span| span.checked_add(1))
        .filter(|&rows| rows >= 1)
        .ok_or_else(|| invalid("chunk rows must be contiguous and ascending"))?;
    if rows as usize > crate::k2_horizon_plan::MAX_CHUNK_TOKENS {
        return Err(invalid("chunk rows exceed the K2 batched prefill bound"));
    }
    Ok(rows as usize)
}

/// Grouped RMSNorm for `rows` contiguous hidden rows in one dispatch. Every
/// (group, row) threadgroup performs the serial group reduction; gamma is
/// indexed in full hidden coordinates exactly as [`encode_grouped_norm`].
pub(crate) fn encode_grouped_norm_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    plan: &K2ShortContextPlan,
    x: &MetalTensor,
    gamma: &MetalTensor,
    out: &MetalTensor,
    rows: usize,
) -> Result<()> {
    serial(enc)?;
    let x_view = rows_view(x, 4096, rows, false)?;
    let gamma_view = View::from(gamma).check(&[4096], GgmlType::F32, false)?;
    let out_view = rows_view(out, 4096, rows, true)?;
    disjoint(&x_view, &out_view)?;
    disjoint(&gamma_view, &out_view)?;
    let groups = plan.norm_groups();
    let width = groups[0].elements.end - groups[0].elements.start;
    let epsilon = groups[0].epsilon;
    if groups.iter().enumerate().any(|(index, group)| {
        group.elements.start != index as u64 * width
            || group.elements.end - group.elements.start != width
            || group.epsilon.to_bits() != epsilon.to_bits()
    }) || width * groups.len() as u64 != 4096
    {
        return Err(invalid(
            "grouped norm rows require uniform contiguous groups",
        ));
    }
    let pipeline = ctx.pipeline("kernel_k2_grouped_rms_norm_rows_f32")?;
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        groups: u32,
        rows: u32,
        threads: u32,
        eps: f32,
    }
    enc.note_read(x);
    enc.note_read(gamma);
    enc.note_write(out);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            width: width as u32,
            groups: groups.len() as u32,
            rows: rows as u32,
            threads: threads as u32,
            eps: epsilon,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, gamma);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, (threads.div_ceil(32) * 4).max(32));
    enc.dispatch(
        MTLSize {
            width: groups.len(),
            height: rows,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Full NEOX RoPE for `rows` consecutive positions starting at `first`. Uses
/// the packed-consecutive kernel, whose per-pair `pow`/`cos`/`sin` arithmetic
/// matches the singleton pair kernel used by [`encode_full_rope`].
pub(crate) fn encode_full_rope_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    first: &TokenPlan<'_>,
    rows: usize,
    query: &MetalTensor,
    key: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let q = rows_view(query, 4096, rows, true)?;
    let k = rows_view(key, 1024, rows, true)?;
    disjoint(&q, &k)?;
    enc.note_write(query);
    enc.note_write(key);
    let rope = first.rope();
    for (tensor, heads) in [
        (query, FullNeoxRope::QUERY_HEADS),
        (key, FullNeoxRope::KV_HEADS),
    ] {
        crate::metal::encode_rope_neox_f32_packed_consecutive(
            ctx,
            enc,
            tensor,
            rows,
            heads,
            FullNeoxRope::HEAD_DIM,
            FullNeoxRope::ROTARY_DIM,
            rope.position(),
            rope.theta(),
        )?;
    }
    Ok(())
}

/// Store `first..=last` post-RoPE K and projected V rows. Consecutive cache
/// rows are contiguous in each plane, so one F32-to-F16 conversion per plane
/// writes the whole chunk with the serial store's rounding.
pub(crate) fn encode_store_kv_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    first: &TokenPlan<'_>,
    last: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let rows = chunk_rows(first, last)?;
    let cache = arena_view(arena, last, true)?;
    let k = rows_view(key, 1024, rows, false)?;
    let v = rows_view(value, 1024, rows, false)?;
    disjoint(&cache, &k)?;
    disjoint(&cache, &v)?;
    let head = first
        .write_ranges(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let tail = last
        .write_ranges(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let span = rows as u64 * 2048;
    if tail.key.end - head.key.start != span || tail.value.end - head.value.start != span {
        return Err(invalid(
            "chunk cache rows are not one contiguous plane span",
        ));
    }
    let k_dst = plane(arena, head.key.start..tail.key.end);
    let v_dst = plane(arena, head.value.start..tail.value.end);
    enc.note_read(key);
    enc.note_read(value);
    enc.note_write(&k_dst);
    enc.note_write(&v_dst);
    encode_scatter_offset_f32_to_f16(ctx, enc, key, &k_dst, 0, 1024 * rows)?;
    encode_scatter_offset_f32_to_f16(ctx, enc, value, &v_dst, 0, 1024 * rows)
}

/// Causal chunk attention: row `r` of `first..=last` attends to positions
/// `0..first.visible_positions() + r`. The caller must have stored every
/// chunk row (and all earlier rows) for this layer before this dispatch.
pub(crate) fn encode_online_attention_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    first: &TokenPlan<'_>,
    last: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    query: &MetalTensor,
    out: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let rows = chunk_rows(first, last)?;
    let cache = arena_view(arena, last, false)?;
    let q = rows_view(query, 4096, rows, false)?;
    let y = rows_view(out, 4096, rows, true)?;
    disjoint(&cache, &q)?;
    disjoint(&cache, &y)?;
    disjoint(&q, &y)?;
    let ranges = last
        .read_ranges(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let key = plane(arena, ranges.key);
    let value = plane(arena, ranges.value);
    online_alignment(query.offset, key.offset, value.offset, out.offset)?;
    let pipeline = ctx.pipeline("kernel_k2_attn_online_rows_f16kv_h128")?;
    if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 512 {
        return Err(invalid(
            "chunk attention requires 32-lane SIMDgroups and 512-thread groups",
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        rows: u32,
        first_positions: u32,
        scale: f32,
    }
    enc.note_read(query);
    enc.note_read(&key);
    enc.note_read(&value);
    enc.note_write(out);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            rows: rows as u32,
            first_positions: first.visible_positions(),
            scale: 128.0_f32.sqrt().recip(),
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, &key);
    enc.set_tensor(3, &value);
    enc.set_tensor(4, out);
    // K2_TILE_POSITIONS (32) x 32 half4 lanes x {K, V} x 8 bytes.
    enc.set_threadgroup_memory(0, 32 * 32 * 2 * 8);
    enc.dispatch(
        MTLSize {
            width: 8,
            height: rows.div_ceil(4),
            depth: 1,
        },
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests;
