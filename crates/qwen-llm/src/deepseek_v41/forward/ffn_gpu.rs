//! Opt-in one-command FFN boundary. No CPU routes or intermediate activations.
use super::*;
use objc2_metal::MTLSize;

#[cfg(test)]
mod tests;

pub(super) struct Workspace {
    // 0 X, 1 incoming pre, 2 projected mixes, 3 next pre, 4 post, 5 comb,
    // 6 collapse, 7 normalized input, 8 logits, 9 scores, 10 IDs, 11 weights,
    // 12 reserved legacy status, 13 activation, 14 weighted activation, 15 down slots,
    // 16/17 shared gate/up, 18 shared activation, 19 shared down, 20 sum, 21 X'.
    buffers: Vec<MetalTensor>,
    constants: Vec<[MetalTensor; 4]>,
    statuses: Vec<MetalTensor>,
}

fn shapes(c: &Config) -> Vec<Vec<u64>> {
    let (h, i, k, e) = (
        c.hidden as u64,
        c.intermediate as u64,
        c.topk as u64,
        c.experts as u64,
    );
    vec![
        vec![h, 4],
        vec![4],
        vec![24],
        vec![4],
        vec![4],
        vec![4, 4],
        vec![h],
        vec![h],
        vec![e],
        vec![e],
        vec![k],
        vec![k],
        vec![1],
        vec![i, k],
        vec![i, k],
        vec![h, k],
        vec![i],
        vec![i],
        vec![i],
        vec![h],
        vec![h],
        vec![h, 4],
    ]
}

fn validate(c: &Config) -> Result<()> {
    op::require(
        c.hidden > 0
            && c.hidden % 32 == 0
            && c.intermediate > 0
            && c.intermediate % 32 == 0
            && c.experts > 0
            && c.topk > 0
            && c.topk <= c.experts
            && c.topk <= 16
            && c.hc_iterations == 20
            && c.norm_eps.is_finite()
            && c.norm_eps > 0.0
            && c.hc_eps.is_finite()
            && c.hc_eps > 0.0
            && c.gate_temperature.is_finite()
            && c.gate_temperature > 0.0
            && c.route_scale.is_finite()
            && c.clamp.is_finite()
            && c.clamp > 0.0,
        "GPU FFN geometry/controls",
    )?;
    for n in [c.hidden, c.intermediate, c.experts] {
        op::require(n <= u32::MAX as usize / 64, "GPU FFN extent overflow")?;
    }
    Ok(())
}

// Includes per-buffer page slack, constants retained for all forty layers, and
// host-side tensor descriptors. Dense resident weights are not counted twice.
pub(super) fn bytes(c: &Config) -> Result<u64> {
    validate(c)?;
    allocation_shapes(c).iter().try_fold(0u64, |sum, shape| {
        shape
            .iter()
            .try_fold(4u64, |n, &d| n.checked_mul(d))
            .and_then(|n| n.checked_add(16384 + 1024))
            .and_then(|n| sum.checked_add(n))
            .ok_or_else(|| "GPU FFN memory overflow".into())
    })
}

fn allocation_shapes(c: &Config) -> Vec<Vec<u64>> {
    let mut sizes = shapes(c);
    sizes.extend((0..40).map(|_| vec![1]));
    for _ in 0..40 {
        sizes.extend([
            vec![3],
            vec![24],
            vec![c.hidden as u64],
            vec![c.experts as u64],
        ]);
    }
    sizes
}

fn dispatch(enc: &KernelEncoder, n: usize, width: usize) {
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteArgs {
    experts: u32,
    topk: u32,
    temperature: f32,
    scale: f32,
}

impl Workspace {
    pub(super) fn new(model: &Forward<'_>) -> Result<Self> {
        let c = &model.config;
        let size = bytes(c)?;
        op::require(model.layers.len() == 40, "GPU FFN layer count")?;
        for shape in allocation_shapes(c) {
            let logical = shape.iter().product::<u64>() * 4;
            let priced = model
                .ctx
                .price_shared_buffer_upper(logical)
                .map_err(|e| e.to_string())?;
            op::require(
                priced.priced_upper_bytes <= logical + 16384,
                "GPU FFN device allocation exceeds estimated page slack",
            )?;
        }
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            size,
            0,
            0,
            model.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(format!("GPU FFN admission: {reason:?}"));
        }
        let buffers = shapes(c)
            .into_iter()
            .enumerate()
            .map(|(index, shape)| {
                MetalTensor::zeros_dtype_unstaged(
                    model.ctx,
                    shape,
                    if index == 10 || index == 12 {
                        GgmlType::I32
                    } else {
                        GgmlType::F32
                    },
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        let constants = model
            .layers
            .iter()
            .map(|layer| {
                Ok([
                    upload(model.ctx, &layer.ffn_hc.scale, vec![3])?,
                    upload(model.ctx, &layer.ffn_hc.base, vec![24])?,
                    upload(model.ctx, &layer.ffn_hc.norm, vec![c.hidden as u64])?,
                    upload(model.ctx, &layer.bias, vec![c.experts as u64])?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let statuses = (0..40)
            .map(|_| {
                MetalTensor::zeros_dtype_unstaged(model.ctx, vec![1], GgmlType::I32)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            buffers,
            constants,
            statuses,
        })
    }

    pub(super) fn run(
        &mut self,
        ctx: &MetalContext,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &[f32],
        pre: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let _wall = profile::Wall(
            "ffn_gpu",
            [c.hidden, c.intermediate, c.experts, c.topk],
            profile::start(),
        );
        op::require(
            x.len() == 4 * c.hidden && pre.len() == 4 && x.iter().chain(pre).all(|v| v.is_finite()),
            "GPU FFN boundary input",
        )?;
        let b = &self.buffers;
        // Unique workspace ownership and synchronous completion exclude pending
        // GPU readers while replacing this token's boundary inputs.
        for (tensor, values) in [(&b[0], x), (&b[1], pre)] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    tensor.buffer.contents().as_ptr().cast::<f32>(),
                    values.len(),
                );
            }
        }
        submit(
            ctx,
            "ffn_gpu",
            [c.hidden, c.intermediate, c.experts, c.topk],
            |enc| self.encode(ctx, enc, c, layer, id, &b[0], &b[1], &b[21]),
        )?;
        let status = self.status(id);
        op::require(
            status == metal::ROUTE_STATUS_READY,
            &format!("GPU FFN route status {status}"),
        )?;
        let output = f32_slice(&b[21])?;
        let next = f32_slice(&b[3])?;
        op::require(
            output.iter().chain(next).all(|v| v.is_finite()),
            "GPU FFN non-finite output",
        )?;
        Ok((output.to_vec(), next.to_vec()))
    }
    pub(super) fn next_pre(&self) -> &MetalTensor {
        &self.buffers[3]
    }

    /// Inspect only after the command containing this layer has completed.
    pub(super) fn status(&self, id: usize) -> i32 {
        unsafe { *self.statuses[id].buffer.contents().as_ptr().cast::<i32>() }
    }

    /// Serial encoding only. Input, incoming pre and output must not alias.
    /// Incoming pre must also be distinct from next_pre(), which controls write
    /// before collapse. Consume next_pre() before encoding the following FFN.
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &MetalTensor,
        pre: &MetalTensor,
        output: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        let b = &self.buffers;
        let constants = &self.constants[id];
        metal::encode_mat_vec_f32(
            ctx,
            enc,
            &layer.ffn_hc.projection.tensor,
            x,
            &b[2],
            4 * c.hidden,
            24,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms")?);
        enc.set_bytes(0, &((4 * c.hidden) as u32));
        enc.set_bytes(1, &c.norm_eps);
        enc.set_tensor(2, x);
        enc.set_tensor(3, &b[2]);
        dispatch(enc, 256, 256);
        metal::encode_mhc4_controls(
            ctx,
            enc,
            c.hc_eps,
            &b[2],
            &constants[0],
            &constants[1],
            &b[3],
            &b[4],
            &b[5],
        )?;
        metal::encode_mhc4_collapse(ctx, enc, c.hidden, x, pre, &b[6])?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &b[6],
            &constants[2],
            &b[7],
            1,
            c.hidden,
            c.norm_eps,
        )?;
        metal::encode_mat_vec_f32(
            ctx,
            enc,
            &layer.router.tensor,
            &b[7],
            &b[8],
            c.hidden,
            c.experts,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_route")?);
        enc.set_bytes(
            0,
            &RouteArgs {
                experts: c.experts as u32,
                topk: c.topk as u32,
                temperature: c.gate_temperature,
                scale: c.route_scale,
            },
        );
        for (slot, tensor) in [
            &b[8],
            &constants[3],
            &b[9],
            &b[10],
            &b[11],
            &self.statuses[id],
        ]
        .iter()
        .enumerate()
        {
            enc.set_tensor(slot + 1, tensor);
        }
        dispatch(enc, 1, 1);
        metal::encode_all_slots_gate_up_swiglu_native(
            ctx,
            enc,
            layer.banks[0],
            layer.banks[1],
            &b[7],
            &b[10],
            &self.statuses[id],
            &b[13],
            c.hidden,
            c.intermediate,
            c.experts,
            c.topk,
            c.clamp,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight")?);
        enc.set_bytes(0, &((c.intermediate * c.topk) as u32));
        enc.set_bytes(1, &(c.intermediate as u32));
        for (slot, tensor) in [&b[13], &b[11], &b[14], &self.statuses[id]]
            .iter()
            .enumerate()
        {
            enc.set_tensor(slot + 2, tensor);
        }
        dispatch(enc, c.intermediate * c.topk, 256);
        metal::encode_all_slots_down_native(
            ctx,
            enc,
            layer.banks[2],
            &b[14],
            &b[10],
            &self.statuses[id],
            &b[15],
            c.intermediate,
            c.hidden,
            c.experts,
            c.topk,
        )?;
        for j in 0..2 {
            metal::encode_mat_vec_f32(
                ctx,
                enc,
                &layer.shared[j].tensor,
                &b[7],
                &b[16 + j],
                c.hidden,
                c.intermediate,
            )?;
        }
        metal::encode_clamped_swiglu(ctx, enc, &b[16], &b[17], &b[18], c.clamp)?;
        metal::encode_mat_vec_f32(
            ctx,
            enc,
            &layer.shared[2].tensor,
            &b[18],
            &b[19],
            c.intermediate,
            c.hidden,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        enc.set_bytes(0, &(c.hidden as u32));
        enc.set_bytes(1, &(c.topk as u32));
        for (slot, tensor) in [&b[15], &b[19], &b[20], &self.statuses[id]]
            .iter()
            .enumerate()
        {
            enc.set_tensor(slot + 2, tensor);
        }
        dispatch(enc, c.hidden, 256);
        metal::encode_mhc4_post(ctx, enc, c.hidden, &b[20], x, &b[4], &b[5], output)
    }
}
