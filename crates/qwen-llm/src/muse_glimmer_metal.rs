//! Muse Glimmer-specific Metal kernels over the shared execution primitives.

use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    query_head_count: usize,
    key_head_count: usize,
    head_dim: usize,
    position: u32,
    theta: f32,
) -> Result<(), MetalError> {
    if query_head_count == 0 || key_head_count == 0 || head_dim == 0 || !head_dim.is_multiple_of(2)
    {
        return bad_shape(
            "muse_glimmer_rope",
            format!(
                "head counts and even head_dim must be nonzero, got q={query_head_count} k={key_head_count} dim={head_dim}"
            ),
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        return bad_shape(
            "muse_glimmer_rope",
            format!("theta must be finite and positive, got {theta}"),
        );
    }
    validate_writable_f32(
        query,
        checked_elements(
            "muse_glimmer_rope",
            query_head_count,
            head_dim,
            "query element count",
        )?,
        "query",
        "muse_glimmer_rope",
    )?;
    validate_writable_f32(
        key,
        checked_elements(
            "muse_glimmer_rope",
            key_head_count,
            head_dim,
            "key element count",
        )?,
        "key",
        "muse_glimmer_rope",
    )?;
    if metal_tensor_ranges_overlap(query, key) {
        return bad_shape(
            "muse_glimmer_rope",
            "query and key storage ranges overlap".into(),
        );
    }
    if position == 0 {
        return Ok(());
    }

    let pairs_per_head = head_dim / 2;
    let query_pair_count = checked_elements(
        "muse_glimmer_rope",
        query_head_count,
        pairs_per_head,
        "query pair count",
    )?;
    let key_pair_count = checked_elements(
        "muse_glimmer_rope",
        key_head_count,
        pairs_per_head,
        "key pair count",
    )?;
    let pair_count = query_pair_count
        .checked_add(key_pair_count)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "muse_glimmer_rope",
            detail: "combined pair count overflow".into(),
        })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_pair_count: u32,
        pair_count: u32,
        head_dim: u32,
        position: u32,
        theta: f32,
    }

    let pso = ctx.pipeline("kernel_muse_glimmer_rope_adjacent_pair_in_place_f32")?;
    enc.note_write(query);
    enc.note_write(key);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            query_pair_count: checked_u32(
                "muse_glimmer_rope",
                query_pair_count,
                "query pair count",
            )?,
            pair_count: checked_u32("muse_glimmer_rope", pair_count, "pair count")?,
            head_dim: checked_u32("muse_glimmer_rope", head_dim, "head_dim")?,
            position,
            theta,
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    let threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(threads),
            height: 1,
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

pub fn encode_muse_glimmer_logit_softcap_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    scale: f32,
    cap: f32,
) -> Result<(), MetalError> {
    let count = usize::try_from(input.n_elements()).map_err(|_| MetalError::BadShape {
        kernel: "muse_glimmer_logit_softcap",
        detail: "input element count exceeds usize".into(),
    })?;
    if count == 0 || output.n_elements() != input.n_elements() {
        return bad_shape(
            "muse_glimmer_logit_softcap",
            format!(
                "input/output must have the same nonzero element count, got {}/{}",
                input.n_elements(),
                output.n_elements()
            ),
        );
    }
    validate_readable_f32(input, count, "input", "muse_glimmer_logit_softcap")?;
    validate_writable_f32(output, count, "output", "muse_glimmer_logit_softcap")?;
    if !scale.is_finite() || !cap.is_finite() || cap <= 0.0 {
        return bad_shape(
            "muse_glimmer_logit_softcap",
            format!("scale must be finite and cap positive, got scale={scale} cap={cap}"),
        );
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        count: u32,
        scale: f32,
        cap: f32,
    }

    let pso = ctx.pipeline("kernel_muse_glimmer_logit_softcap_f32")?;
    enc.note_read(input);
    enc.note_write(output);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            count: checked_u32("muse_glimmer_logit_softcap", count, "logit count")?,
            scale,
            cap,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    let threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(threads),
            height: 1,
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

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_causal_gqa_vjp_bank_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    gate: &MetalTensor,
    attention_output: &MetalTensor,
    probabilities: &MetalTensor,
    grad_gated: &MetalTensor,
    grad_query: &MetalTensor,
    partial_grad_key: &MetalTensor,
    partial_grad_value: &MetalTensor,
    grad_key: &MetalTensor,
    grad_value: &MetalTensor,
    grad_gate: &MetalTensor,
    basis_count: usize,
    n_tokens: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_causal_gqa_vjp_bank";
    if enc.is_concurrent() {
        return bad_shape(
            KERNEL,
            "dependent VJP dispatches require a serial encoder".into(),
        );
    }
    if basis_count == 0
        || n_tokens == 0
        || n_tokens > 16
        || query_head_count == 0
        || kv_head_count == 0
        || head_dim == 0
        || !query_head_count.is_multiple_of(kv_head_count)
        || !head_dim.is_multiple_of(32)
        || head_dim > 256
    {
        return bad_shape(
            KERNEL,
            format!(
                "expected B>0, T in 1..=16, divisible heads, and head_dim in 32..=256; got B={basis_count} T={n_tokens} q={query_head_count} kv={kv_head_count} dim={head_dim}"
            ),
        );
    }

    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let kv_width = checked_elements(KERNEL, kv_head_count, head_dim, "KV width")?;
    let primal_query_elements =
        checked_elements(KERNEL, n_tokens, query_width, "primal query elements")?;
    let primal_kv_elements = checked_elements(KERNEL, n_tokens, kv_width, "primal KV elements")?;
    let bank_rows = checked_elements(KERNEL, basis_count, n_tokens, "bank rows")?;
    let bank_query_elements =
        checked_elements(KERNEL, bank_rows, query_width, "bank query elements")?;
    let bank_kv_elements = checked_elements(KERNEL, bank_rows, kv_width, "bank KV elements")?;
    let probability_rows =
        checked_elements(KERNEL, n_tokens, query_head_count, "probability rows")?;
    let probability_elements =
        checked_elements(KERNEL, probability_rows, n_tokens, "probability elements")?;
    let partial_heads = checked_elements(KERNEL, basis_count, query_head_count, "partial heads")?;
    let partial_rows = checked_elements(KERNEL, partial_heads, n_tokens, "partial rows")?;
    let partial_elements = checked_elements(KERNEL, partial_rows, head_dim, "partial elements")?;

    let primal_query_shape = [query_width as u64, n_tokens as u64];
    let primal_kv_shape = [kv_width as u64, n_tokens as u64];
    let probability_shape = [n_tokens as u64, query_head_count as u64, n_tokens as u64];
    let bank_query_shape = [query_width as u64, bank_rows as u64];
    let bank_kv_shape = [kv_width as u64, bank_rows as u64];
    let partial_shape = [
        head_dim as u64,
        n_tokens as u64,
        query_head_count as u64,
        basis_count as u64,
    ];

    for (tensor, elements, shape, name) in [
        (
            query,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "query",
        ),
        (key, primal_kv_elements, primal_kv_shape.as_slice(), "key"),
        (
            value,
            primal_kv_elements,
            primal_kv_shape.as_slice(),
            "value",
        ),
        (
            gate,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "gate",
        ),
        (
            attention_output,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "attention output",
        ),
        (
            probabilities,
            probability_elements,
            probability_shape.as_slice(),
            "probabilities",
        ),
        (
            grad_gated,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "gated cotangent bank",
        ),
    ] {
        validate_readable_f32_shape(tensor, elements, shape, name, KERNEL)?;
    }
    for (tensor, elements, shape, name) in [
        (
            grad_query,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "query cotangent bank",
        ),
        (
            partial_grad_key,
            partial_elements,
            partial_shape.as_slice(),
            "partial key bank",
        ),
        (
            partial_grad_value,
            partial_elements,
            partial_shape.as_slice(),
            "partial value bank",
        ),
        (
            grad_key,
            bank_kv_elements,
            bank_kv_shape.as_slice(),
            "key cotangent bank",
        ),
        (
            grad_value,
            bank_kv_elements,
            bank_kv_shape.as_slice(),
            "value cotangent bank",
        ),
        (
            grad_gate,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "gate cotangent bank",
        ),
    ] {
        validate_writable_f32_shape(tensor, elements, shape, name, KERNEL)?;
    }

    let reads = [
        query,
        key,
        value,
        gate,
        attention_output,
        probabilities,
        grad_gated,
    ];
    let writes = [
        grad_query,
        partial_grad_key,
        partial_grad_value,
        grad_key,
        grad_value,
        grad_gate,
    ];
    for (write_index, write) in writes.iter().enumerate() {
        if reads
            .iter()
            .any(|read| metal_tensor_ranges_overlap(write, read))
            || writes[..write_index]
                .iter()
                .any(|prior| metal_tensor_ranges_overlap(write, prior))
        {
            return bad_shape(KERNEL, "VJP output overlaps another tensor".into());
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        basis_count: u32,
        n_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
    }
    let args = Args {
        basis_count: checked_u32(KERNEL, basis_count, "basis count")?,
        n_tokens: checked_u32(KERNEL, n_tokens, "token count")?,
        query_heads: checked_u32(KERNEL, query_head_count, "query head count")?,
        kv_heads: checked_u32(KERNEL, kv_head_count, "KV head count")?,
        head_dim: checked_u32(KERNEL, head_dim, "head dimension")?,
        scale: (head_dim as f32).sqrt().recip(),
    };
    let partial_tgm_elements =
        checked_elements(KERNEL, n_tokens, head_dim, "partial TGM elements")?
            .checked_mul(2)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "partial TGM element count overflow".into(),
            })?;
    let partial_tgm_bytes = partial_tgm_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "partial TGM byte count overflow".into(),
        })?;

    let vjp = ctx.pipeline("kernel_muse_glimmer_causal_gqa_vjp_bank_f32")?;
    let reduce = ctx.pipeline("kernel_muse_glimmer_causal_gqa_vjp_reduce_kv_f32")?;
    for (name, pipeline) in [("VJP", &vjp), ("KV reduction", &reduce)] {
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
            return bad_shape(KERNEL, format!("{name} requires one 32-thread SIMDgroup"));
        }
    }
    if ctx.device.maxThreadgroupMemoryLength() < partial_tgm_bytes {
        return bad_shape(
            KERNEL,
            format!("VJP requires {partial_tgm_bytes} bytes of threadgroup memory"),
        );
    }

    for tensor in reads {
        enc.note_read(tensor);
    }
    for tensor in writes {
        enc.note_write(tensor);
    }
    enc.set_pipeline(&vjp);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    enc.set_tensor(3, value);
    enc.set_tensor(4, gate);
    enc.set_tensor(5, attention_output);
    enc.set_tensor(6, probabilities);
    enc.set_tensor(7, grad_gated);
    enc.set_tensor(8, grad_query);
    enc.set_tensor(9, partial_grad_key);
    enc.set_tensor(10, partial_grad_value);
    enc.set_tensor(11, grad_gate);
    enc.set_threadgroup_memory(0, partial_tgm_bytes);
    enc.dispatch(
        MTLSize {
            width: query_head_count,
            height: basis_count,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partial_grad_key);
    enc.set_tensor(2, partial_grad_value);
    enc.set_tensor(3, grad_key);
    enc.set_tensor(4, grad_value);
    enc.dispatch(
        MTLSize {
            width: kv_head_count,
            height: n_tokens,
            depth: basis_count,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_readable_f32(
    tensor: &MetalTensor,
    elements: usize,
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    if tensor.dtype != GgmlType::F32 || tensor.n_elements() != elements as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} must be F32 with {elements} elements, got {:?} and {}",
                tensor.dtype,
                tensor.n_elements()
            ),
        );
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<f32>() as u64)
    {
        return bad_shape(
            kernel,
            format!("{name} offset {} is not F32-aligned", tensor.offset),
        );
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: format!("{name} buffer range overflow"),
        })?;
    if end > tensor.buffer.length() as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} range ends at {end}, beyond buffer length {}",
                tensor.buffer.length()
            ),
        );
    }
    Ok(())
}

fn validate_writable_f32(
    tensor: &MetalTensor,
    elements: usize,
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_readable_f32(tensor, elements, name, kernel)?;
    if !tensor.is_writable() {
        return bad_shape(kernel, format!("{name} must be writable"));
    }
    Ok(())
}

fn validate_readable_f32_shape(
    tensor: &MetalTensor,
    elements: usize,
    shape: &[u64],
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_readable_f32(tensor, elements, name, kernel)?;
    if tensor.shape.as_slice() != shape {
        return bad_shape(
            kernel,
            format!("{name} shape {:?} differs from {shape:?}", tensor.shape),
        );
    }
    Ok(())
}

fn validate_writable_f32_shape(
    tensor: &MetalTensor,
    elements: usize,
    shape: &[u64],
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_writable_f32(tensor, elements, name, kernel)?;
    if tensor.shape.as_slice() != shape {
        return bad_shape(
            kernel,
            format!("{name} shape {:?} differs from {shape:?}", tensor.shape),
        );
    }
    Ok(())
}

fn metal_tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let Some(left_end) = left.offset.checked_add(left.n_bytes()) else {
        return true;
    };
    let Some(right_end) = right.offset.checked_add(right.n_bytes()) else {
        return true;
    };
    left.offset < right_end && right.offset < left_end
}

fn checked_elements(
    kernel: &'static str,
    left: usize,
    right: usize,
    label: &'static str,
) -> Result<usize, MetalError> {
    left.checked_mul(right).ok_or_else(|| MetalError::BadShape {
        kernel,
        detail: format!("{label} overflow"),
    })
}

fn checked_u32(kernel: &'static str, value: usize, label: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::BadShape {
        kernel,
        detail: format!("{label} {value} exceeds u32"),
    })
}

fn bad_shape<T>(kernel: &'static str, detail: String) -> Result<T, MetalError> {
    Err(MetalError::BadShape { kernel, detail })
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    fn tensor_from_f32(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        let mut values = vec![0.0; tensor.n_elements() as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(tensor.offset as usize / std::mem::size_of::<f32>()),
                values.as_mut_ptr(),
                values.len(),
            );
        }
        values
    }

    #[test]
    fn adjacent_rope_and_softcap_match_cpu_formulas() {
        let ctx = MetalContext::new().unwrap();
        let query_source = (0..16)
            .map(|value| value as f32 / 7.0 - 1.0)
            .collect::<Vec<_>>();
        let key_source = (0..8)
            .map(|value| value as f32 / 5.0 - 0.5)
            .collect::<Vec<_>>();
        let query = tensor_from_f32(&ctx, &query_source);
        let key = tensor_from_f32(&ctx, &key_source);
        let logits_source = [-100.0, -3.0, 0.0, 4.0, 100.0];
        let logits = tensor_from_f32(&ctx, &logits_source);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
            &ctx, &encoder, &query, &key, 2, 1, 8, 11, 500_000.0,
        )
        .unwrap();
        encode_muse_glimmer_logit_softcap_f32(&ctx, &encoder, &logits, &logits, 0.196_116_13, 20.0)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let mut expected_query = query_source;
        let mut expected_key = key_source;
        for values in [&mut expected_query[..], &mut expected_key[..]] {
            for head in values.chunks_exact_mut(8) {
                for pair in 0..4 {
                    let relative = pair * 2;
                    let angle = 11.0 * 500_000.0_f32.powf(-(relative as f32) / 8.0);
                    let (sine, cosine) = angle.sin_cos();
                    let first = head[relative];
                    let second = head[relative + 1];
                    head[relative] = first * cosine - second * sine;
                    head[relative + 1] = first * sine + second * cosine;
                }
            }
        }
        for (actual, expected) in read_f32(&query).into_iter().zip(expected_query) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        for (actual, expected) in read_f32(&key).into_iter().zip(expected_key) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        for (actual, raw) in read_f32(&logits).into_iter().zip(logits_source) {
            let expected = 20.0 * (raw * 0.196_116_13 / 20.0).tanh();
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn causal_gqa_vjp_bank_rejects_zero_width_and_malformed_layout() {
        let ctx = MetalContext::new().unwrap();
        const B: usize = 1;
        const T: usize = 3;
        const QH: usize = 4;
        const KVH: usize = 2;
        const D: usize = 32;
        let query_width = QH * D;
        let kv_width = KVH * D;
        let query = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let key = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, T as u64]).unwrap();
        let value = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, T as u64]).unwrap();
        let gate = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let attention = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let probabilities =
            MetalTensor::zeros_f32(&ctx, vec![T as u64, QH as u64, T as u64]).unwrap();
        let grad_gated =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();
        let grad_query =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();
        let partial_shape = vec![D as u64, T as u64, QH as u64, B as u64];
        let partial_key = MetalTensor::zeros_f32(&ctx, partial_shape.clone()).unwrap();
        let partial_value = MetalTensor::zeros_f32(&ctx, partial_shape).unwrap();
        let grad_key = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, (B * T) as u64]).unwrap();
        let grad_value =
            MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, (B * T) as u64]).unwrap();
        let grad_gate =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();

        let reject = |query: &MetalTensor, head_dim: usize| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let error = encode_muse_glimmer_causal_gqa_vjp_bank_f32(
                &ctx,
                &encoder,
                query,
                &key,
                &value,
                &gate,
                &attention,
                &probabilities,
                &grad_gated,
                &grad_query,
                &partial_key,
                &partial_value,
                &grad_key,
                &grad_value,
                &grad_gate,
                B,
                T,
                QH,
                KVH,
                head_dim,
            )
            .unwrap_err();
            encoder.end();
            error.to_string()
        };
        assert!(reject(&query, 0).contains("head_dim"));
        let mut malformed_query = query.clone();
        malformed_query.shape = vec![(query_width * T) as u64];
        assert!(reject(&malformed_query, D).contains("query shape"));
    }
}
