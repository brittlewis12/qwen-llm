//! Experimental K2 Q8 cache v1. Raw I8 arenas are byte storage, not weight tensors.
use super::*;

const ROW_BYTES: u64 = 1088;

fn cache_view(
    view: View<'_>,
    bytes: u64,
    storage: K2KvStorage,
    write: bool,
) -> Result<CheckedView> {
    if storage != K2KvStorage::Q8_0
        || !view.offset.is_multiple_of(2)
        || !bytes.is_multiple_of(ROW_BYTES)
    {
        return Err(invalid(
            "Q8 cache requires its own plan and aligned 34-byte blocks",
        ));
    }
    view.check(&[bytes], GgmlType::I8, write)
}

fn plane(arena: &MetalTensor, bytes: Range<u64>) -> MetalTensor {
    arena.view_bytes(bytes.start, vec![bytes.end - bytes.start])
}

fn pipeline(ctx: &MetalContext, name: &'static str) -> Result<crate::metal::Pipeline> {
    let pipeline = ctx.pipeline(name)?;
    if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
        return Err(invalid("Q8 K2 cache kernels require a 32-lane SIMDgroup"));
    }
    Ok(pipeline)
}

pub(crate) fn encode_store(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let cache = cache_view(
        View::from(arena),
        token.arena_bytes(),
        token.storage(),
        true,
    )?;
    let k = View::from(key).check(&[128, 8], GgmlType::F32, false)?;
    let v = View::from(value).check(&[128, 8], GgmlType::F32, false)?;
    disjoint(&cache, &k)?;
    disjoint(&cache, &v)?;
    let ranges = token
        .write_ranges(layer)
        .map_err(|e| invalid(e.to_string()))?;
    let key_dst = plane(arena, ranges.key);
    let value_dst = plane(arena, ranges.value);
    let pipeline = pipeline(ctx, "kernel_k2_store_q8_kv")?;
    enc.note_read(key);
    enc.note_read(value);
    enc.note_write(&key_dst);
    enc.note_write(&value_dst);
    enc.set_pipeline(&pipeline);
    enc.set_tensor(0, key);
    enc.set_tensor(1, value);
    enc.set_tensor(2, &key_dst);
    enc.set_tensor(3, &value_dst);
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

pub(crate) fn encode_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token: &TokenPlan<'_>,
    layer: u32,
    arena: &MetalTensor,
    query: &MetalTensor,
    output: &MetalTensor,
) -> Result<()> {
    serial(enc)?;
    let cache = cache_view(
        View::from(arena),
        token.arena_bytes(),
        token.storage(),
        false,
    )?;
    let q = View::from(query).check(&[128, 32], GgmlType::F32, false)?;
    let y = View::from(output).check(&[128, 32], GgmlType::F32, true)?;
    disjoint(&cache, &q)?;
    disjoint(&cache, &y)?;
    disjoint(&q, &y)?;
    if !query.offset.is_multiple_of(16) || !output.offset.is_multiple_of(16) {
        return Err(invalid(
            "Q8 attention query/output require float4 alignment",
        ));
    }
    let ranges = token
        .read_ranges(layer)
        .map_err(|e| invalid(e.to_string()))?;
    let key = plane(arena, ranges.key);
    let value = plane(arena, ranges.value);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        positions: u32,
        scale: f32,
    }
    let pipeline = pipeline(ctx, "kernel_k2_attn_online_q8kv_h128")?;
    enc.note_read(query);
    enc.note_read(&key);
    enc.note_read(&value);
    enc.note_write(output);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            positions: token.visible_positions(),
            scale: 128.0_f32.sqrt().recip(),
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, &key);
    enc.set_tensor(3, &value);
    enc.set_tensor(4, output);
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

/// Check committed candidates only. Does not dequantize or scan retained history.
pub(crate) fn validate_row(bytes: &[u8]) -> Result<()> {
    if bytes.len() != ROW_BYTES as usize {
        return Err(invalid("wrong Q8 cache row extent"));
    }
    for block in bytes.chunks_exact(34) {
        let bits = u16::from_le_bytes(block[..2].try_into().unwrap());
        if bits & 0x7c00 == 0x7c00 || bits & 0x8000 != 0 {
            return Err(invalid("nonfinite or negative Q8 cache scale"));
        }
        if block[2..].contains(&128) || (bits == 0 && block[2..].iter().any(|&q| q != 0)) {
            return Err(invalid("noncanonical Q8 cache payload"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
