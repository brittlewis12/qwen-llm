//! Column scratch and gather/scatter for batched decode projections: B
//! sequences' vectors side by side as the columns of one `[n, B]` F32 tensor
//! (each column contiguous), so a dense weight is read once for all of them
//! (`Linear::encode_columns`).
use super::*;

/// One allocation holding `[n, cols]` F32 regions of the given column
/// heights, each region starting on a 256-byte boundary; priced and admitted
/// before allocating. Returns the regions in order.
pub(super) fn allocate(
    ctx: &MetalContext,
    heights: &[usize],
    cols: usize,
    what: &str,
) -> Result<Vec<MetalTensor>> {
    allocate_typed(ctx, heights, cols, GgmlType::F32, what)
}

/// `allocate` with 4-byte `dtype` (F32 or I32) regions.
pub(super) fn allocate_typed(
    ctx: &MetalContext,
    heights: &[usize],
    cols: usize,
    dtype: GgmlType,
    what: &str,
) -> Result<Vec<MetalTensor>> {
    op::require(
        matches!(dtype, GgmlType::F32 | GgmlType::I32),
        &format!("{what} dtype"),
    )?;
    op::require(
        cols >= 1 && !heights.is_empty(),
        &format!("{what} needs columns"),
    )?;
    let sizes = heights
        .iter()
        .map(|&n| {
            n.checked_mul(cols)
                .filter(|&n| n > 0 && u32::try_from(n).is_ok())
                .ok_or_else(|| format!("{what} extent"))
        })
        .collect::<Result<Vec<_>>>()?;
    let strides: Vec<usize> = sizes.iter().map(|n| n.next_multiple_of(64)).collect();
    let total = strides
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .filter(|n| n.checked_mul(4).is_some())
        .ok_or_else(|| format!("{what} bytes"))?;
    let logical = total as u64 * 4;
    let priced = ctx
        .price_shared_buffer_upper(logical)
        .map_err(|e| e.to_string())?;
    op::require(
        priced.priced_upper_bytes <= logical + 16384,
        &format!("{what} exceeds estimated page slack"),
    )?;
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        priced.priced_upper_bytes,
        0,
        0,
        ctx.memory_signals(),
        true,
    );
    if let Some(reason) = admission.refusal() {
        return Err(format!("{what} refused: {reason:?}"));
    }
    let backing = MetalTensor::zeros_dtype_unstaged(ctx, vec![total as u64], dtype)
        .map_err(|e| e.to_string())?;
    let mut offset = 0;
    Ok(sizes
        .iter()
        .zip(strides)
        .map(|(&n, stride)| {
            let view = backing.view_subrange(offset as u64, vec![n as u64]);
            offset += stride;
            view
        })
        .collect())
}

fn height(columns: &MetalTensor, count: usize) -> std::result::Result<usize, metal::MetalError> {
    let n = columns.n_elements() as usize;
    if count == 0 || !n.is_multiple_of(count) {
        return Err(metal::MetalError::BadShape {
            kernel: "v41_columns",
            detail: format!("{count} columns do not divide {n} elements"),
        });
    }
    Ok(n / count)
}

/// `per_column` gathers and scatters batched decode columns with one copy
/// dispatch per sequence; otherwise up to 8 sequences take one gather or
/// scatter dispatch (`kernel_deepseek_v41_{gather,scatter}_columns`; same
/// values). M5 Ultra, GSQ, batched shared expert and attention: rounds
/// 0.8/1.5/2.3% faster than per-column copies at B = 2/4/8. A/B lever.
const BATCH_COPY_ENV: &str = "QWEN_V41_BATCH_COPY";

#[cfg(test)]
thread_local! {
    pub(super) static PER_COLUMN_COPY_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn per_column_copy() -> bool {
    #[cfg(test)]
    if let Some(per_column) = PER_COLUMN_COPY_OVERRIDE.with(|o| o.get()) {
        return per_column;
    }
    static PER_COLUMN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PER_COLUMN.get_or_init(|| std::env::var(BATCH_COPY_ENV).as_deref() == Ok("per_column"))
}

/// Vectors one column-copy dispatch binds.
const COPY_VECTORS: usize = 8;

/// One dispatch copying between `columns` (`[n * vectors.len()]`) and the
/// `[n]` vectors (`gather`: vectors -> columns).
fn encode_columns_copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    columns: &MetalTensor,
    vectors: &[&MetalTensor],
    n: usize,
    gather: bool,
) -> std::result::Result<(), metal::MetalError> {
    let bad = |detail: String| metal::MetalError::BadShape {
        kernel: "v41_columns_copy",
        detail,
    };
    if vectors.is_empty() || vectors.len() > COPY_VECTORS || u32::try_from(n).is_err() {
        return Err(bad(format!("{} vectors of {n}", vectors.len())));
    }
    for v in vectors.iter().copied().chain([columns]) {
        if v.dtype != columns.dtype
            || !matches!(v.dtype, GgmlType::F32 | GgmlType::I32)
            || !v.offset.is_multiple_of(4)
            || !v.is_writable()
        {
            return Err(bad(
                "writable 4-byte aligned bindings of one dtype (F32 or I32) required".into(),
            ));
        }
    }
    if vectors.iter().any(|v| v.n_elements() != n as u64)
        || columns.n_elements() != (n * vectors.len()) as u64
    {
        return Err(bad(format!("vectors of {n} and their columns required")));
    }
    let name = if gather {
        "kernel_deepseek_v41_gather_columns"
    } else {
        "kernel_deepseek_v41_scatter_columns"
    };
    enc.set_pipeline(&ctx.pipeline(name)?);
    enc.set_bytes(0, &[n as u32, vectors.len() as u32]);
    enc.set_tensor(1, columns);
    for slot in 0..COPY_VECTORS {
        enc.set_tensor(2 + slot, vectors[slot.min(vectors.len() - 1)]);
    }
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n.div_ceil(256),
            height: vectors.len(),
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The per-column copy of `columns`' dtype (F32 or I32).
fn copy_words(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    src_off: usize,
    dst: &MetalTensor,
    n: usize,
) -> std::result::Result<(), metal::MetalError> {
    if src.dtype == GgmlType::I32 {
        metal::encode_copy_offset_i32(ctx, enc, src, src_off, dst, n)
    } else {
        metal::encode_copy_offset_f32(ctx, enc, src, src_off, dst, n)
    }
}

/// Copy `inputs[b]` (`[n]` F32 or I32 each) into column b of `columns` (`[n * B]`).
pub(super) fn encode_gather(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    columns: &MetalTensor,
) -> std::result::Result<(), metal::MetalError> {
    let n = height(columns, inputs.len())?;
    if !per_column_copy() {
        for (chunk, vectors) in inputs.chunks(COPY_VECTORS).enumerate() {
            let at = (chunk * COPY_VECTORS * n) as u64;
            let view = columns.view_subrange(at, vec![(vectors.len() * n) as u64]);
            encode_columns_copy(ctx, enc, &view, vectors, n, true)?;
        }
        return Ok(());
    }
    for (column, input) in inputs.iter().enumerate() {
        let dst = columns.view_subrange((column * n) as u64, vec![n as u64]);
        copy_words(ctx, enc, input, 0, &dst, n)?;
    }
    Ok(())
}

/// Copy column b of `columns` (`[n * B]`) into `outputs[b]` (`[n]` F32 or
/// I32 each).
pub(super) fn encode_scatter(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    columns: &MetalTensor,
    outputs: &[&MetalTensor],
) -> std::result::Result<(), metal::MetalError> {
    let n = height(columns, outputs.len())?;
    if !per_column_copy() {
        for (chunk, vectors) in outputs.chunks(COPY_VECTORS).enumerate() {
            let at = (chunk * COPY_VECTORS * n) as u64;
            let view = columns.view_subrange(at, vec![(vectors.len() * n) as u64]);
            encode_columns_copy(ctx, enc, &view, vectors, n, false)?;
        }
        return Ok(());
    }
    for (column, output) in outputs.iter().enumerate() {
        copy_words(ctx, enc, columns, column * n, output, n)?;
    }
    Ok(())
}

/// `outputs[b] = linear(inputs[b])` for every sequence b as one column batch:
/// gather into `x` (`[input * B]`), `Linear::encode_columns` into `y`
/// (`[output * B]`), scatter. Bitwise per-sequence `Linear::encode` except
/// where `encode_columns` takes the skinny Q5_K kernel (F32 rounding).
pub(super) fn encode_linear(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    linear: &Linear<'_>,
    inputs: &[&MetalTensor],
    outputs: &[&MetalTensor],
    (x, y): (&MetalTensor, &MetalTensor),
) -> std::result::Result<(), metal::MetalError> {
    if inputs.len() != outputs.len() {
        return Err(metal::MetalError::BadShape {
            kernel: "v41_columns",
            detail: format!("{} inputs for {} outputs", inputs.len(), outputs.len()),
        });
    }
    encode_gather(ctx, enc, inputs, x)?;
    linear.encode_columns(enc, x, y, inputs.len())?;
    encode_scatter(ctx, enc, y, outputs)
}

/// `encode_linear_pair` of every sequence b (`inputs[b]` -> `outputs_a[b]`,
/// `outputs_b[b]`) as one column batch: gather into `x`,
/// `encode_linear_pair_columns` into `y_a` / `y_b`, scatter both. Bitwise
/// per-sequence except where the skinny Q8_0 pair kernel runs (F32
/// rounding).
pub(super) fn encode_linear_pair(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    (a, outputs_a): (&Linear<'_>, &[&MetalTensor]),
    (b, outputs_b): (&Linear<'_>, &[&MetalTensor]),
    inputs: &[&MetalTensor],
    (x, y_a, y_b): (&MetalTensor, &MetalTensor, &MetalTensor),
) -> std::result::Result<(), metal::MetalError> {
    if inputs.len() != outputs_a.len() || inputs.len() != outputs_b.len() {
        return Err(metal::MetalError::BadShape {
            kernel: "v41_columns",
            detail: format!(
                "{} inputs for {} and {} outputs",
                inputs.len(),
                outputs_a.len(),
                outputs_b.len()
            ),
        });
    }
    encode_gather(ctx, enc, inputs, x)?;
    super::encode_linear_pair_columns(enc, (a, y_a), (b, y_b), x, inputs.len())?;
    encode_scatter(ctx, enc, y_a, outputs_a)?;
    encode_scatter(ctx, enc, y_b, outputs_b)
}

#[cfg(test)]
mod tests;
