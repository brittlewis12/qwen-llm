//! Unqualified K2 primitive bridge; no runtime dispatch or model ownership.
//!
//! Serial encoders only. Callers must discard the command buffer on encoding
//! failure, and must not commit session state until successful execution/readout.
//! These wrappers validate physical views, not cached-prefix identity/freshness,
//! value finiteness, or a complete forward graph. GPU qualification is pending.

use crate::k2_horizon_plan::{FullNeoxRope, K2ShortContextPlan, TokenPlan};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, encode_attn_decode_f16kv_f32,
    encode_rms_norm_mul_f32, encode_rope_neox_pair_f32, encode_scatter_offset_f32_to_f16,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use std::ops::Range;

const KERNEL: &str = "k2_primitive_bridge";
type Result<T> = std::result::Result<T, MetalError>;

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
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            _ => return Err(invalid("only F32 activations and F16 cache are supported")),
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

#[cfg(test)]
mod tests;
