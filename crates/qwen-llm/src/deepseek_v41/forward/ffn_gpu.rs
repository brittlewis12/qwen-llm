//! FFN component of the full GPU graph. No CPU routes or intermediate readbacks.
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
            && c.intermediate > 0
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

#[derive(Debug)]
struct CompositionExtents {
    slots: u32,
    weighted: u32,
    output: u32,
}

fn composition_extents(
    c: &Config,
    rows: usize,
) -> std::result::Result<CompositionExtents, metal::MetalError> {
    let bad = |detail| metal::MetalError::BadShape {
        kernel: "v41_ffn_composition",
        detail,
    };
    validate(c).map_err(bad)?;
    if rows == 0 {
        return Err(bad("nonempty rows required".into()));
    }
    // The native GEMV uses ulong offsets, but route/weight/sum and HC use uint.
    // Validate the entire composition before encoding, not just each projection.
    let extent = |name, factors: &[usize]| {
        factors
            .iter()
            .try_fold(1usize, |n, &factor| n.checked_mul(factor))
            .filter(|n| n.checked_mul(4).is_some())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad(format!("{name} extent exceeds host/GPU domains")))
    };
    let slots = extent("IDs/weights", &[rows, c.topk])?;
    extent("router", &[rows, c.experts])?;
    let weighted = extent("weighted activation", &[rows, c.topk, c.intermediate])?;
    extent("down slots", &[rows, c.topk, c.hidden])?;
    let output = extent("output", &[rows, c.hidden])?;
    extent("HC streams", &[rows, 4, c.hidden])?;
    extent("HC mixes", &[rows, 24])?;
    Ok(CompositionExtents {
        slots,
        weighted,
        output,
    })
}

// Includes per-buffer page slack, constants retained for the loaded layers, and
// host-side tensor descriptors. Dense resident weights are not counted twice.
pub(super) fn bytes_for_layers(c: &Config, layers: usize) -> Result<u64> {
    validate(c)?;
    allocation_shapes(c, layers)
        .iter()
        .try_fold(0u64, |sum, shape| {
            shape
                .iter()
                .try_fold(4u64, |n, &d| n.checked_mul(d))
                .and_then(|n| n.checked_add(16384 + 1024))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| "GPU FFN memory overflow".into())
        })
}

fn allocation_shapes(c: &Config, layers: usize) -> Vec<Vec<u64>> {
    let mut sizes = shapes(c);
    sizes.extend((0..layers).map(|_| vec![1]));
    for _ in 0..layers {
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

/// Threadgroup width of the parallel router; must match ROUTE_THREADS in
/// kernels/deepseek_v41_ffn.metal.
const ROUTE_THREADS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteArgs {
    experts: u32,
    topk: u32,
    temperature: f32,
    scale: f32,
}

fn encode_route_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    rows: usize,
    logits: &MetalTensor,
    bias: &MetalTensor,
    scores: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
    status: &MetalTensor,
) -> std::result::Result<(), metal::MetalError> {
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
    for (slot, tensor) in [logits, bias, scores, ids, weights, status]
        .iter()
        .enumerate()
    {
        enc.set_tensor(slot + 1, tensor);
    }
    // One ROUTE_THREADS-wide threadgroup per row (kernel_deepseek_v41_route).
    enc.dispatch(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ROUTE_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

impl Workspace {
    pub(super) fn new(model: &Forward<'_>) -> std::result::Result<Self, StateError> {
        let c = &model.config;
        let size = bytes_for_layers(c, model.layers.len())?;
        op::require(!model.layers.is_empty(), "GPU FFN layer count")?;
        for shape in allocation_shapes(c, model.layers.len()) {
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
            return Err(StateError::MemoryAdmissionDenied(reason));
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
        let statuses = (0..model.layers.len())
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

    #[cfg(test)]
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
        self.encode_mhc_pre(ctx, enc, c, layer, id, x, pre)?;
        self.encode_router_shared(ctx, enc, c, layer)?;
        self.encode_routed_stage(ctx, enc, c, layer, id)?;
        self.encode_mhc_post(ctx, enc, c, x, output)
    }

    pub(super) fn encode_mhc_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &MetalTensor,
        pre: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        composition_extents(c, 1)?;
        let b = &self.buffers;
        let constants = &self.constants[id];
        layer.ffn_hc.projection.encode(enc, x, &b[2])?;
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
        )
    }

    pub(super) fn encode_router_shared(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
    ) -> std::result::Result<(), metal::MetalError> {
        let b = &self.buffers;
        layer.router.encode(enc, &b[7], &b[8])?;
        for j in 0..2 {
            layer.shared[j].encode(enc, &b[7], &b[16 + j])?;
        }
        metal::encode_clamped_swiglu(ctx, enc, &b[16], &b[17], &b[18], c.clamp)?;
        layer.shared[2].encode(enc, &b[18], &b[19])
    }

    pub(super) fn encode_routed_stage(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        let b = &self.buffers;
        let constants = &self.constants[id];
        self.encode_routed(
            ctx,
            enc,
            c,
            layer,
            &constants[3],
            &b[7],
            &b[8],
            &b[19],
            &b[20],
            &self.statuses[id],
        )
    }

    pub(super) fn encode_mhc_post(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        x: &MetalTensor,
        output: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        let b = &self.buffers;
        metal::encode_mhc4_post(ctx, enc, c.hidden, &b[20], x, &b[4], &b[5], output)
    }

    fn encode_routed(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        bias: &MetalTensor,
        normalized: &MetalTensor,
        logits: &MetalTensor,
        shared: &MetalTensor,
        output: &MetalTensor,
        status: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        let extents = composition_extents(c, 1)?;
        let b = &self.buffers;
        encode_route_rows(ctx, enc, c, 1, logits, bias, &b[9], &b[10], &b[11], status)?;
        metal::encode_all_slots_gate_up_swiglu_native(
            ctx,
            enc,
            layer.banks[0],
            layer.banks[1],
            normalized,
            &b[10],
            status,
            &b[13],
            c.hidden,
            c.intermediate,
            c.experts,
            c.topk,
            c.clamp,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight")?);
        enc.set_bytes(0, &extents.weighted);
        enc.set_bytes(1, &(c.intermediate as u32));
        for (slot, tensor) in [&b[13], &b[11], &b[14], status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.set_bytes(6, &(c.topk as u32));
        dispatch(enc, extents.weighted as usize, 256);
        metal::encode_all_slots_down_native(
            ctx,
            enc,
            layer.banks[2],
            &b[14],
            &b[10],
            status,
            &b[15],
            c.intermediate,
            c.hidden,
            c.experts,
            c.topk,
        )?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        enc.set_bytes(0, &(c.hidden as u32));
        enc.set_bytes(1, &(c.topk as u32));
        for (slot, tensor) in [&b[15], shared, output, status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.set_bytes(6, &1u32);
        dispatch(enc, extents.output as usize, 256);
        Ok(())
    }

    pub(super) fn encode_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &MetalTensor,
        incoming: &MetalTensor,
        output: &MetalTensor,
        scratch: &Rows,
    ) -> std::result::Result<(), metal::MetalError> {
        let n = scratch.hc.rows;
        composition_extents(c, n)?;
        let b = &scratch.buffers;
        scratch.hc.encode(
            ctx,
            enc,
            c,
            &layer.ffn_hc.projection,
            &self.constants[id],
            x,
            incoming,
        )?;
        layer
            .router
            .encode_rows(enc, &scratch.hc.normalized, &b[0], n)?;
        for j in 0..2 {
            layer.shared[j].encode_rows(enc, &scratch.hc.normalized, &b[1 + j], n)?;
        }
        let shape = vec![(c.intermediate * n) as u64];
        metal::encode_clamped_swiglu(
            ctx,
            enc,
            &b[1].view_bytes(0, shape.clone()),
            &b[2].view_bytes(0, shape.clone()),
            &b[3].view_bytes(0, shape),
            c.clamp,
        )?;
        layer.shared[2].encode_rows(enc, &b[3], &b[4], n)?;
        if half_grouped_selected(c, layer, n)
            || grouped_selected(c, layer, n)
            || native_rows_supported(c, layer)
        {
            self.encode_routed_rows(ctx, enc, c, layer, id, scratch)?;
        } else {
            for row in 0..n {
                self.encode_routed(
                    ctx,
                    enc,
                    c,
                    layer,
                    &self.constants[id][3],
                    &prefill::row(&scratch.hc.normalized, c.hidden, row),
                    &prefill::row(&b[0], c.experts, row),
                    &prefill::row(&b[4], c.hidden, row),
                    &prefill::row(&b[5], c.hidden, row),
                    &prefill::row(&b[6], 1, id * n + row),
                )?;
            }
        }
        metal::encode_mhc4_post_rows(
            ctx,
            enc,
            c.hidden,
            n,
            &b[5],
            x,
            &scratch.hc.post,
            &scratch.hc.comb,
            output,
        )
    }

    fn encode_routed_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        scratch: &Rows,
    ) -> std::result::Result<(), metal::MetalError> {
        let n = scratch.hc.rows;
        let extents = composition_extents(c, n)?;
        let b = &scratch.buffers;
        let status = b[6].view_subrange((id * n) as u64, vec![n as u64]);
        let counts = b[10].view_subrange(0, vec![c.experts as u64]);
        encode_route_rows(
            ctx,
            enc,
            c,
            n,
            &b[0],
            &self.constants[id][3],
            &b[7],
            &b[8],
            &b[9],
            &status,
        )?;
        let half = half_grouped_selected(c, layer, n);
        let grouped = half || grouped_selected(c, layer, n);
        if grouped {
            metal::encode_moe_route_bucket_slots_f32(
                ctx, enc, &b[8], &counts, &b[11], c.experts, n, c.topk,
            )?;
            let gate_up = if half {
                metal::encode_moe_swiglu_clamped_f32_grouped_slots_generic
            } else {
                metal::encode_moe_swiglu_clamped_f32x_grouped_slots
            };
            gate_up(
                ctx,
                enc,
                layer.banks[0],
                layer.banks[1],
                &scratch.hc.normalized,
                &counts,
                &b[11],
                &b[12],
                c.hidden,
                c.intermediate,
                c.experts,
                c.topk,
                n,
                c.clamp,
            )?;
        } else {
            metal::encode_all_slots_gate_up_swiglu_native_rows(
                ctx,
                enc,
                layer.banks[0],
                layer.banks[1],
                &scratch.hc.normalized,
                &b[8],
                &status,
                &b[12].view_subrange(0, vec![c.intermediate as u64, extents.slots as u64]),
                c.hidden,
                c.intermediate,
                c.experts,
                c.topk,
                n,
                c.clamp,
            )?;
        }
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight")?);
        enc.set_bytes(0, &extents.weighted);
        enc.set_bytes(1, &(c.intermediate as u32));
        for (slot, tensor) in [&b[12], &b[9], &b[13], &status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.set_bytes(6, &(c.topk as u32));
        dispatch(enc, extents.weighted as usize, 256);
        if grouped {
            let down = if half {
                metal::encode_moe_down_f32_grouped_slots_generic
            } else {
                metal::encode_moe_down_f32x_grouped_slots
            };
            down(
                ctx,
                enc,
                layer.banks[2],
                &b[13],
                &counts,
                &b[11],
                &b[14],
                c.intermediate,
                c.hidden,
                c.experts,
                n,
            )?;
        } else {
            metal::encode_all_slots_down_native_rows(
                ctx,
                enc,
                layer.banks[2],
                &b[13].view_subrange(0, vec![c.intermediate as u64, extents.slots as u64]),
                &b[8],
                &status,
                &b[14].view_subrange(0, vec![c.hidden as u64, extents.slots as u64]),
                c.intermediate,
                c.hidden,
                c.experts,
                c.topk,
                n,
            )?;
        }
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        enc.set_bytes(0, &(c.hidden as u32));
        enc.set_bytes(1, &(c.topk as u32));
        for (slot, tensor) in [&b[14], &b[4], &b[5], &status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.set_bytes(6, &(n as u32));
        dispatch(enc, extents.output as usize, 256);
        Ok(())
    }
}

/// The half-staged grouped expert tiles GLM and DeepSeek V4 prefill use:
/// operands rounded to half inside the tile, F32 sums, weights read once per
/// tile rather than once per row. Covers every expert storage type with a
/// generic grouped layout (IQ2_S, IQ3_XXS, IQ3_S, Q2_K, Q3_K, ...); MXFP4
/// banks stay on the F32-operand tiles and IQ2_XXS/IQ2_XS on native rows.
pub(super) fn half_grouped_supported(c: &Config, layer: &Layer<'_>) -> bool {
    #[cfg(test)]
    if tests::NATIVE_ROWS_ONLY.with(|flag| flag.get()) {
        return false;
    }
    #[cfg(test)]
    if prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.get()) {
        return false;
    }
    metal::moe_grouped_clamped_recipe(layer.banks[0].dtype, layer.banks[1].dtype).is_some()
        && metal::moe_grouped_generic_supported(layer.banks[2].dtype)
        && [
            (c.hidden, layer.banks[0].dtype),
            (c.hidden, layer.banks[1].dtype),
            (c.intermediate, layer.banks[2].dtype),
        ]
        .iter()
        .all(|&(k, dtype)| {
            dtype
                .storage_layout()
                .is_some_and(|(block, _)| (k as u64).is_multiple_of(block.max(32)))
        })
}

pub(super) fn half_grouped_selected(c: &Config, layer: &Layer<'_>, rows: usize) -> bool {
    // Forcing the grouped route in tests means the F32-operand tiles, whose
    // frozen witnesses hold a 1e-5 contract the half tiles do not claim.
    #[cfg(test)]
    if tests::FORCE_GROUPED_ROWS.with(|flag| flag.get()) {
        return false;
    }
    // One row is a GEMV; the tiles only pay off once an expert sees several.
    rows > 1 && half_grouped_supported(c, layer)
}

/// Layers per routed-expert route for a chunk of `rows`: half-staged
/// grouped, F32-operand grouped, native rows, per-row replay.
pub(super) fn routed_route_census(c: &Config, layers: &[Layer<'_>], rows: usize) -> [usize; 4] {
    let mut census = [0; 4];
    for layer in layers
        .iter()
        .filter(|layer| layer.banks[0].n_elements() > 0)
    {
        let route = if half_grouped_selected(c, layer, rows) {
            0
        } else if grouped_selected(c, layer, rows) {
            1
        } else if native_rows_supported(c, layer) {
            2
        } else {
            3
        };
        census[route] += 1;
    }
    census
}

fn grouped_selected(c: &Config, layer: &Layer<'_>, rows: usize) -> bool {
    if !grouped_supported(c, layer) {
        return false;
    }
    #[cfg(test)]
    if tests::FORCE_GROUPED_ROWS.with(|flag| flag.get()) {
        return true;
    }
    // Paired small-chunk curves favor native rows below 128, not a precision fallback.
    !(rows < 128
        && matches!(layer.banks[0].dtype, GgmlType::IQ3_XXS | GgmlType::Q3_K)
        && native_rows_supported(c, layer))
}

fn grouped_supported(c: &Config, layer: &Layer<'_>) -> bool {
    #[cfg(test)]
    if tests::NATIVE_ROWS_ONLY.with(|flag| flag.get()) {
        return false;
    }
    #[cfg(test)]
    if prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.get()) {
        return false;
    }
    layer.banks[0].dtype == layer.banks[1].dtype
        && metal::moe_swiglu_clamped_f32x_grouped_slots_supported(layer.banks[0].dtype)
        && metal::moe_down_f32x_grouped_slots_supported(layer.banks[2].dtype)
        && [
            (c.hidden, layer.banks[0].dtype),
            (c.intermediate, layer.banks[2].dtype),
        ]
        .iter()
        .all(|&(k, dtype)| {
            dtype
                .storage_layout()
                .is_some_and(|(block, _)| (k as u64).is_multiple_of(block.max(32)))
        })
}

fn native_rows_supported(c: &Config, layer: &Layer<'_>) -> bool {
    #[cfg(test)]
    if prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.get()) {
        return false;
    }
    metal::all_slots_gate_up_native_rows_supported(layer.banks[0].dtype, layer.banks[1].dtype)
        && metal::all_slots_down_native_rows_supported(layer.banks[2].dtype)
        && [
            (c.hidden, layer.banks[0]),
            (c.hidden, layer.banks[1]),
            (c.intermediate, layer.banks[2]),
        ]
        .iter()
        .all(|&(k, bank)| {
            bank.dtype
                .storage_layout()
                .is_some_and(|(block, _)| (k as u64).is_multiple_of(block))
        })
}

pub(super) struct Rows {
    pub(super) hc: prefill::HcRows,
    // Activations are reused; buffer 6 retains layer-major [layer][row] statuses.
    buffers: Vec<MetalTensor>,
}
impl Rows {
    pub(super) fn widths(c: &Config, layers: usize) -> Vec<usize> {
        let mut widths = prefill::HcRows::widths(c);
        widths.extend([
            c.experts,
            c.intermediate,
            c.intermediate,
            c.intermediate,
            c.hidden,
            c.hidden,
            layers,
            c.experts,
            c.topk,
            c.topk,
            c.experts,
            c.experts,
            c.intermediate * c.topk,
            c.intermediate * c.topk,
            c.hidden * c.topk,
        ]);
        widths
    }
    pub(super) fn new(ctx: &MetalContext, c: &Config, rows: usize, layers: usize) -> Result<Self> {
        composition_extents(c, rows).map_err(|e| e.to_string())?;
        let hc = prefill::HcRows::new(ctx, c, rows)?;
        let buffers = [
            c.experts,
            c.intermediate,
            c.intermediate,
            c.intermediate,
            c.hidden,
            c.hidden,
            layers,
            c.experts,
            c.topk,
            c.topk,
            c.experts,
            c.experts,
            c.intermediate * c.topk,
            c.intermediate * c.topk,
            c.hidden * c.topk,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, width)| {
            prefill::allocate(
                ctx,
                width,
                rows,
                if matches!(i, 6 | 8 | 10 | 11) {
                    GgmlType::I32
                } else {
                    GgmlType::F32
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
        Ok(Self { hc, buffers })
    }
    pub(super) fn validate_status(&self) -> Result<()> {
        let t = &self.buffers[6];
        let values = unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add((t.offset / 4) as usize),
                t.n_elements() as usize,
            )
        };
        for (index, &status) in values.iter().enumerate() {
            if status != metal::ROUTE_STATUS_READY {
                return Err(format!(
                    "prefill FFN layer {} row {} route status {status}",
                    index / self.hc.rows,
                    index % self.hc.rows,
                ));
            }
        }
        Ok(())
    }
}
