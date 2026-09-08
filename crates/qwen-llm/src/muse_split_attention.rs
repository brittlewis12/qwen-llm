//! Bounded H128 split-position online attention with caller-owned scratch.

use super::*;

pub(crate) const PARTIAL_ELEMENTS: u64 = 32 * 32 * 132;

#[cfg(test)]
thread_local! {
    static SCRATCH: std::cell::RefCell<Option<MetalTensor>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn allocate(ctx: &MetalContext) -> MetalTensor {
    MetalTensor::zeros_f32(ctx, vec![PARTIAL_ELEMENTS]).unwrap()
}

#[cfg(test)]
pub(crate) fn with_scratch<R>(scratch: &MetalTensor, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<MetalTensor>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCRATCH.with(|slot| slot.replace(self.0.take()));
        }
    }
    let _restore = Restore(SCRATCH.with(|slot| slot.replace(Some(scratch.clone()))));
    run()
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    output: &MetalTensor,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    positions: usize,
) -> Result<bool, MetalError> {
    SCRATCH.with(|slot| {
        let slot = slot.borrow();
        let Some(partial) = slot.as_ref() else {
            return Ok(false);
        };
        encode(
            ctx, enc, query, key, value, output, partial, q_heads, kv_heads, dim, positions,
        )?;
        Ok(true)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    output: &MetalTensor,
    partial: &MetalTensor,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    positions: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_split_attention";
    if enc.is_concurrent() {
        return bad_shape(
            KERNEL,
            "dependent attention dispatches require a serial encoder".into(),
        );
    }
    if (q_heads, kv_heads, dim) != (32, 2, 128)
        || positions == 0
        || positions > u32::MAX as usize - 31
    {
        return bad_shape(
            KERNEL,
            "requires released G16/H128 and bounded nonempty KV".into(),
        );
    }
    let elements = checked_elements(KERNEL, positions, 256, "cache")?;
    validate_readable_f32(query, 4096, "query", KERNEL)?;
    validate_readable_f16(key, elements, "key", KERNEL)?;
    validate_readable_f16(value, elements, "value", KERNEL)?;
    validate_writable_f32(output, 4096, "output", KERNEL)?;
    validate_writable_f32(partial, 32 * 32 * 132, "partials", KERNEL)?;
    for (tensor, alignment) in [
        (query, 16),
        (key, 8),
        (value, 8),
        (output, 16),
        (partial, 16),
    ] {
        if !tensor.offset.is_multiple_of(alignment) {
            return bad_shape(KERNEL, "unaligned vector view".into());
        }
    }
    for source in [query, key, value] {
        if metal_tensor_ranges_overlap(source, output)
            || metal_tensor_ranges_overlap(source, partial)
        {
            return bad_shape(KERNEL, "input aliases mutable output".into());
        }
    }
    if metal_tensor_ranges_overlap(output, partial) {
        return bad_shape(KERNEL, "output aliases partial storage".into());
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        positions: u32,
        stride: u32,
        partitions: u32,
        scale: f32,
    }
    let partitions = positions.div_ceil(128).min(32);
    let args = Args {
        positions: checked_u32(KERNEL, positions, "positions")?,
        stride: 256,
        partitions: partitions as u32,
        scale: 128.0_f32.sqrt().recip(),
    };
    let main = ctx.pipeline("kernel_muse_split_attention_h128")?;
    let reduce = ctx.pipeline("kernel_muse_split_attention_reduce_h128")?;
    for source in [query, key, value] {
        enc.note_read(source);
    }
    enc.note_write(partial);
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    for (slot, tensor) in [query, key, value, partial].into_iter().enumerate() {
        enc.set_tensor(slot + 1, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: 32,
            height: partitions,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&reduce);
    enc.note_read(partial);
    enc.note_write(output);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partial);
    enc.set_tensor(2, output);
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
