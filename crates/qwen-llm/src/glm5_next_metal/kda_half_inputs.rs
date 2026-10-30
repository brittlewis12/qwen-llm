//! One admitted half input shared only by the six KDA input projections.

use super::{memory, packed, tensor_prefill};
use crate::metal::checks::{bad_shape, check_alignment, check_disjoint, check_tensor, to_u32};
use crate::metal::{self, KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLResource};

const KERNEL: &str = "glm_kda_half_inputs";

pub(super) type Consumers<'a> = [(&'a MetalTensor, &'a MetalTensor, usize); 6];

pub(super) struct Prepared {
    src: MetalTensor,
    half: MetalTensor,
    consumers: [Option<tensor_prefill::PreparedHalf>; 6],
    elements: usize,
}

impl Prepared {
    pub(super) fn copy(&self, ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), MetalError> {
        metal::encode_scatter_offset_f32_to_f16(ctx, enc, &self.src, &self.half, 0, self.elements)
    }

    pub(super) fn encode(
        &self,
        index: usize,
        ctx: &MetalContext,
        enc: &KernelEncoder,
    ) -> Result<bool, MetalError> {
        if let Some(consumer) = &self.consumers[index] {
            consumer.encode(ctx, enc)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

fn backing_extent(
    shape: &[u64],
    offset: u64,
    physical: u64,
    hidden: usize,
    rows: usize,
) -> Result<(), MetalError> {
    if shape.len() != 2 || shape[0] != hidden as u64 || rows == 0 || shape[1] < rows as u64 {
        return Err(bad_shape(
            KERNEL,
            "half input must retain admitted [hidden, rows] backing",
        ));
    }
    let (_, bytes) = metal::checked_ggml_shape_bytes(shape, GgmlType::F16)?;
    if !offset.is_multiple_of(16)
        || offset
            .checked_add(bytes as u64)
            .is_none_or(|end| end > physical || end > isize::MAX as u64)
    {
        return Err(bad_shape(
            KERNEL,
            "half input backing outside physical extent/alignment",
        ));
    }
    Ok(())
}

/// No encoder mutation until converter, destination and every selected twin
/// have passed preflight. Ineligible consumers always retain the F32 input.
pub(super) fn prepare(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: packed::Arith,
    original: &MetalTensor,
    backing: Option<&MetalTensor>,
    consumers: &Consumers<'_>,
    hidden: usize,
    rows: usize,
) -> Result<Option<Prepared>, MetalError> {
    if !memory::kda_half_inputs_enabled() || !metal::moe_grouped_tensor_selected(ctx) {
        return Ok(None);
    }
    let Some(backing) = backing else {
        return Ok(None);
    };
    let mut selected = [false; 6];
    for (i, &(weight, _, width)) in consumers.iter().enumerate() {
        selected[i] = tensor_prefill::eligible(
            arith,
            packed::projection_mode(
                arith,
                packed::Stage::KdaProjection,
                weight.dtype,
                hidden,
                width,
            ),
            weight.dtype,
            hidden,
            width,
            rows,
        )?;
    }
    if !selected.iter().any(|&yes| yes) {
        return Ok(None);
    }
    backing_extent(
        &backing.shape,
        backing.offset,
        backing.buffer.length() as u64,
        hidden,
        rows,
    )?;
    check_tensor(
        KERNEL,
        backing,
        GgmlType::F16,
        &backing.shape,
        true,
        "half backing",
    )?;
    if backing.buffer.device().registryID() != ctx.device.registryID() {
        return Err(bad_shape(KERNEL, "half backing belongs to another device"));
    }
    let elements = hidden
        .checked_mul(rows)
        .ok_or_else(|| bad_shape(KERNEL, "conversion extent overflow"))?;
    to_u32(KERNEL, elements, "conversion elements")?;
    let half = backing.view_subrange(0, vec![hidden as u64, rows as u64]);
    // Check disjointness against even fallback consumers before the copy.
    for &(weight, output, _) in consumers {
        for t in [weight, output, original] {
            metal::checked_ggml_shape_bytes(&t.shape, t.dtype)?;
        }
        check_disjoint(
            KERNEL,
            &half,
            &[
                (weight, "weight"),
                (output, "output"),
                (original, "original"),
            ],
        )?;
    }
    check_alignment(KERNEL, original, 16, "converter input")?;
    let mut prepared = std::array::from_fn(|_| None);
    let mut supported = true;
    for (i, &(weight, output, width)) in consumers.iter().enumerate() {
        if selected[i] {
            prepared[i] = tensor_prefill::prepare_half(
                ctx,
                enc,
                arith,
                packed::projection_mode(
                    arith,
                    packed::Stage::KdaProjection,
                    weight.dtype,
                    hidden,
                    width,
                ),
                weight,
                original,
                &half,
                output,
                hidden,
                width,
                rows,
            )?;
            supported &= prepared[i].is_some();
        }
    }
    if !supported {
        return Ok(None);
    }
    let converter = ctx.pipeline("kernel_scatter_offset_f32_to_f16")?;
    if converter.maxTotalThreadsPerThreadgroup() == 0 {
        return Ok(None);
    }
    Ok(Some(Prepared {
        src: original.view_subrange(0, vec![elements as u64]),
        half,
        consumers: prepared,
        elements,
    }))
}

#[cfg(test)]
#[path = "tests/kda_half_inputs.rs"]
mod tests;
