//! Binding checks shared by family-neutral encoders: dtype, exact shape,
//! writability, element alignment and buffer range, plus expert-bank layout.

use super::*;

pub(crate) fn bad_shape(kernel: &'static str, detail: impl Into<String>) -> MetalError {
    MetalError::BadShape {
        kernel,
        detail: detail.into(),
    }
}

pub(crate) fn require_serial(kernel: &'static str, enc: &KernelEncoder) -> Result<(), MetalError> {
    if enc.is_concurrent() {
        return Err(bad_shape(kernel, "requires ordered serial dispatches"));
    }
    Ok(())
}

/// `tensor` is `dtype` with exactly `shape`, element-aligned, inside its buffer,
/// and writable when `writable`.
pub(crate) fn check_tensor(
    kernel: &'static str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), MetalError> {
    if tensor.dtype != dtype || tensor.shape != shape {
        return Err(bad_shape(
            kernel,
            format!(
                "{name} must be {dtype:?} {shape:?}, got {:?} {:?}",
                tensor.dtype, tensor.shape
            ),
        ));
    }
    if writable && !tensor.is_writable() {
        return Err(bad_shape(kernel, format!("{name} must be writable")));
    }
    if !tensor.offset.is_multiple_of(4) {
        return Err(bad_shape(
            kernel,
            format!("{name} offset is not 4-byte aligned"),
        ));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| bad_shape(kernel, format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return Err(bad_shape(kernel, format!("{name} exceeds its buffer")));
    }
    Ok(())
}

/// A binding's physical extent, whatever its logical shape: its offset is a
/// multiple of `align`, its bytes lie inside its buffer, and it is writable
/// when `writable`. For encoders that accept row views of several shapes.
pub(crate) fn check_physical(
    kernel: &'static str,
    tensor: &MetalTensor,
    align: u64,
    writable: bool,
    name: &str,
) -> Result<(), MetalError> {
    if !tensor.offset.is_multiple_of(align) {
        return Err(bad_shape(
            kernel,
            format!("{name} offset is not {align}-byte aligned"),
        ));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| bad_shape(kernel, format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return Err(bad_shape(kernel, format!("{name} exceeds its buffer")));
    }
    if writable && !tensor.is_writable() {
        return Err(bad_shape(kernel, format!("{name} must be writable")));
    }
    Ok(())
}

/// An expert bank `[n_in, n_out, experts]` of `experts` contiguous, block-aligned
/// expert matrices inside its buffer.
pub(crate) fn check_expert_bank(
    kernel: &'static str,
    bank: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    name: &str,
) -> Result<(), MetalError> {
    let shape = [n_in as u64, n_out as u64, experts as u64];
    if bank.shape != shape {
        return Err(bad_shape(
            kernel,
            format!("{name} must have shape {shape:?}, got {:?}", bank.shape),
        ));
    }
    let (block, block_bytes) = bank
        .dtype
        .storage_layout()
        .ok_or_else(|| bad_shape(kernel, format!("{name} has unknown layout")))?;
    if n_in == 0 || !(n_in as u64).is_multiple_of(block) {
        return Err(bad_shape(
            kernel,
            format!("{name} width {n_in} is not a multiple of block {block}"),
        ));
    }
    let bytes = (n_in as u64 / block)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(n_out as u64))
        .and_then(|expert| expert.checked_mul(experts as u64))
        .ok_or_else(|| bad_shape(kernel, format!("{name} size overflow")))?;
    let end = bank.offset.checked_add(bytes);
    if bank.n_bytes() != bytes || end.is_none_or(|end| end > bank.buffer.length() as u64) {
        return Err(bad_shape(
            kernel,
            format!("{name} is not {experts} contiguous expert slices inside its buffer"),
        ));
    }
    // Quant block structs are read through typed device pointers.
    check_alignment(kernel, bank, 16, name)
}

/// `tensor`'s byte offset is a multiple of `align` (e.g. 16 for `float4` access).
pub(crate) fn check_alignment(
    kernel: &'static str,
    tensor: &MetalTensor,
    align: u64,
    name: &str,
) -> Result<(), MetalError> {
    if !tensor.offset.is_multiple_of(align) {
        return Err(bad_shape(
            kernel,
            format!(
                "{name} offset {} is not {align}-byte aligned",
                tensor.offset
            ),
        ));
    }
    Ok(())
}

fn byte_range(tensor: &MetalTensor) -> (usize, u64, u64) {
    let start = tensor.offset;
    (
        Retained::as_ptr(&tensor.buffer) as *const _ as *const u8 as usize,
        start,
        start.saturating_add(tensor.n_bytes()),
    )
}

/// Whether two views share any byte of the same buffer.
pub(crate) fn overlaps(a: &MetalTensor, b: &MetalTensor) -> bool {
    let (buffer_a, start_a, end_a) = byte_range(a);
    let (buffer_b, start_b, end_b) = byte_range(b);
    buffer_a == buffer_b && start_a < end_b && start_b < end_a
}

/// Whether two views cover exactly the same bytes of the same buffer.
pub(crate) fn same_range(a: &MetalTensor, b: &MetalTensor) -> bool {
    byte_range(a) == byte_range(b)
}

/// `output` shares no byte with any of `inputs`.
pub(crate) fn check_disjoint(
    kernel: &'static str,
    output: &MetalTensor,
    inputs: &[(&MetalTensor, &str)],
) -> Result<(), MetalError> {
    for (input, name) in inputs {
        if overlaps(output, input) {
            return Err(bad_shape(kernel, format!("output aliases {name}")));
        }
    }
    Ok(())
}

pub(crate) fn to_u32(kernel: &'static str, value: usize, name: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| bad_shape(kernel, format!("{name} {value} exceeds u32")))
}
