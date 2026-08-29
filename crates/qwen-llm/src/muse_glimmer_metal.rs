//! Muse Glimmer-specific Metal kernels over the shared execution primitives.

use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};

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
        checked_elements(query_head_count, head_dim, "query element count")?,
        "query",
        "muse_glimmer_rope",
    )?;
    validate_writable_f32(
        key,
        checked_elements(key_head_count, head_dim, "key element count")?,
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
    let query_pair_count = checked_elements(query_head_count, pairs_per_head, "query pair count")?;
    let key_pair_count = checked_elements(key_head_count, pairs_per_head, "key pair count")?;
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
            query_pair_count: checked_u32(query_pair_count, "query pair count")?,
            pair_count: checked_u32(pair_count, "pair count")?,
            head_dim: checked_u32(head_dim, "head_dim")?,
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
            count: checked_u32(count, "logit count")?,
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

fn checked_elements(left: usize, right: usize, label: &'static str) -> Result<usize, MetalError> {
    left.checked_mul(right).ok_or_else(|| MetalError::BadShape {
        kernel: "muse_glimmer_rope",
        detail: format!("{label} overflow"),
    })
}

fn checked_u32(value: usize, label: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::BadShape {
        kernel: "muse_glimmer",
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
}
