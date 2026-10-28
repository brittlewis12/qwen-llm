//! FFN component of the full GPU graph. No CPU routes or intermediate readbacks.
use super::prefill::RowsArithmetic;
use super::*;
use objc2_metal::MTLSize;

#[cfg(test)]
pub(super) mod tests;

#[cfg(test)]
#[path = "ffn_gpu/tests/arithmetic.rs"]
mod arithmetic_tests;

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
    super::super::execution::validate_ffn(c).map_err(|e| e.to_string())
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

/// `split` keeps the decode FFN combine as routing-weight, sum and post
/// dispatches; otherwise wide sites apply the routing weights to the down
/// outputs inside one combine dispatch (A/B lever).
const FFN_COMBINE_ENV: &str = "QWEN_V41_FFN_COMBINE";

#[cfg(test)]
thread_local! {
    pub(super) static FOLDED_COMBINE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn folded_combine_enabled() -> bool {
    #[cfg(test)]
    if let Some(folded) = FOLDED_COMBINE_OVERRIDE.with(|o| o.get()) {
        return folded;
    }
    static FOLDED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FOLDED.get_or_init(|| std::env::var(FFN_COMBINE_ENV).as_deref() != Ok("split"))
}

/// `serial` buckets the prefill routes with one thread per expert scanning
/// every rows x topk id; otherwise one threadgroup per expert counts, scans
/// and scatters in parallel (bitwise-equal output; A/B lever).
const ROUTE_BUCKET_ENV: &str = "QWEN_V41_ROUTE_BUCKET";

fn route_bucket_parallel() -> bool {
    static PARALLEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PARALLEL.get_or_init(|| std::env::var(ROUTE_BUCKET_ENV).as_deref() != Ok("serial"))
}

/// One sequence's shared expert: gate and up projections of `x`, the clamped
/// SwiGLU, then the down projection. `scratch` is [gate, up, activation,
/// output].
fn encode_shared_vector(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    shared: &[Linear<'_>; 3],
    clamp: f32,
    x: &MetalTensor,
    [gate, up, activation, output]: [&MetalTensor; 4],
) -> std::result::Result<(), metal::MetalError> {
    shared[0].encode(enc, x, gate)?;
    shared[1].encode(enc, x, up)?;
    metal::encode_clamped_swiglu(ctx, enc, gate, up, activation, clamp)?;
    shared[2].encode(enc, activation, output)
}

/// Column scratch for the shared expert of `cols` sequences in one batched
/// decode step: input and output `[hidden, cols]`, gate/up/activation
/// `[intermediate, cols]`, each column contiguous, in one allocation whose
/// regions start on 256-byte boundaries. 4 * cols * (2 * hidden + 3 *
/// intermediate) bytes plus padding: about 69 KiB per sequence at GSQ
/// geometry.
pub(super) struct SharedColumns {
    cols: usize,
    x: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    activation: MetalTensor,
    output: MetalTensor,
}

impl SharedColumns {
    /// Prices and admits the scratch before allocating it.
    pub(super) fn new(ctx: &MetalContext, c: &Config, cols: usize) -> Result<Self> {
        validate(c)?;
        let (h, i) = (c.hidden, c.intermediate);
        let mut regions =
            columns::allocate(ctx, &[h, i, i, i, h], cols, "shared column scratch")?.into_iter();
        let mut next = || regions.next().expect("five regions");
        Ok(Self {
            cols,
            x: next(),
            gate: next(),
            up: next(),
            activation: next(),
            output: next(),
        })
    }
}

/// The shared expert of several sequences as one column batch: copies
/// sequence b's normalized input (`inputs[b]`, `[hidden]`) into column b,
/// projects gate and up with `Linear::encode_columns`, applies the clamped
/// SwiGLU over every column at once (an elementwise kernel, so each value is
/// `encode_shared`'s for the same gate/up values), projects down, and copies
/// column b to `outputs[b]` (`[hidden]`). With F32 (or any non-skinny)
/// weights every value is bitwise `encode_shared`'s; Q5_K weights go
/// through the skinny kernel and differ by F32 rounding.
pub(super) fn encode_shared_columns(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    shared: &[Linear<'_>; 3],
    id: usize,
    inputs: &[&MetalTensor],
    outputs: &[&MetalTensor],
    scratch: &SharedColumns,
) -> std::result::Result<(), metal::MetalError> {
    let cols = scratch.cols;
    if inputs.len() != cols || outputs.len() != cols {
        return Err(metal::MetalError::BadShape {
            kernel: "v41_shared_columns",
            detail: format!(
                "{} inputs and {} outputs for {cols} scratch columns",
                inputs.len(),
                outputs.len()
            ),
        });
    }
    metal::dispatch_profile_label("ffn_shared", Some(id));
    columns::encode_gather(ctx, enc, inputs, &scratch.x)?;
    shared[0].encode_columns(enc, &scratch.x, &scratch.gate, cols)?;
    shared[1].encode_columns(enc, &scratch.x, &scratch.up, cols)?;
    metal::encode_clamped_swiglu(
        ctx,
        enc,
        &scratch.gate,
        &scratch.up,
        &scratch.activation,
        c.clamp,
    )?;
    shared[2].encode_columns(enc, &scratch.activation, &scratch.output, cols)?;
    columns::encode_scatter(ctx, enc, &scratch.output, outputs)
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
        self.encode_hc_pre(ctx, enc, c, layer, id, x, pre)?;
        self.encode_shared(ctx, enc, c, layer, id)?;
        self.encode_routed_experts(ctx, enc, c, layer, id, x)?;
        self.encode_combine(ctx, enc, c, layer, id, x, output)
    }

    /// Whether this layer's FFN site runs the wide pre-block and the one-dispatch
    /// combine (routing weights and routed sum folded into the controls post).
    /// `x` is the site's residual input; the routed and combine parts must agree.
    fn folded_combine(&self, c: &Config, layer: &Layer<'_>, id: usize, x: &MetalTensor) -> bool {
        let b = &self.buffers;
        folded_combine_enabled()
            && hc_gpu::wide(
                &layer.ffn_hc,
                c.hidden,
                x,
                &b[6],
                &b[7],
                &self.constants[id][2],
            )
    }

    /// `encode` in four parts. The shared expert and the router plus routed
    /// experts only read the HC pre output and write disjoint buffers, so on
    /// separate encoders Metal's resource tracking may overlap them; the
    /// combine part reads both. In order on one encoder they are `encode`.
    pub(super) fn encode_hc_pre(
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
        metal::dispatch_profile_label("ffn_hc_pre", Some(id));
        if hc_gpu::fused_supported(&layer.ffn_hc) {
            hc_gpu::encode_pre(
                ctx,
                enc,
                c,
                &layer.ffn_hc,
                constants,
                x,
                hc_gpu::HcPreSite {
                    mix: &b[2],
                    pre: &b[3],
                    post: &b[4],
                    combination: &b[5],
                    incoming_pre: pre,
                    collapsed: &b[6],
                    normalized: &b[7],
                },
            )?;
        } else {
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
            )?;
        }
        Ok(())
    }

    pub(super) fn encode_shared(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        let b = &self.buffers;
        metal::dispatch_profile_label("ffn_shared", Some(id));
        encode_shared_vector(
            ctx,
            enc,
            &layer.shared,
            c.clamp,
            &b[7],
            [&b[16], &b[17], &b[18], &b[19]],
        )
    }

    /// The shared expert's input (the normalized FFN input `encode_hc_pre`
    /// writes) and output (read by `encode_combine`).
    pub(super) fn shared_io(&self) -> (&MetalTensor, &MetalTensor) {
        (&self.buffers[7], &self.buffers[19])
    }

    pub(super) fn encode_routed_experts(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        let folded = self.folded_combine(c, layer, id, x);
        let extents = composition_extents(c, 1)?;
        let b = &self.buffers;
        let constants = &self.constants[id];
        let status = &self.statuses[id];
        let label = |name| metal::dispatch_profile_label(name, Some(id));
        label("ffn_router");
        layer.router.encode(enc, &b[7], &b[8])?;
        label("ffn_routed");
        encode_route_rows(
            ctx,
            enc,
            c,
            1,
            &b[8],
            &constants[3],
            &b[9],
            &b[10],
            &b[11],
            status,
        )?;
        metal::encode_all_slots_gate_up_swiglu_native_fused(
            ctx,
            enc,
            layer.banks[0],
            layer.banks[1],
            &b[7],
            &b[10],
            status,
            &b[13],
            c.hidden,
            c.intermediate,
            c.experts,
            c.topk,
            c.clamp,
        )?;
        // Folded: the combine weights the down outputs, so down reads the
        // unweighted SwiGLU rows and the weight dispatch drops out.
        if !folded {
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight")?);
            enc.set_bytes(0, &extents.weighted);
            enc.set_bytes(1, &(c.intermediate as u32));
            for (slot, tensor) in [&b[13], &b[11], &b[14], status].iter().enumerate() {
                enc.set_tensor(slot + 2, tensor);
            }
            enc.set_bytes(6, &(c.topk as u32));
            dispatch(enc, extents.weighted as usize, 256);
        }
        metal::encode_all_slots_down_native(
            ctx,
            enc,
            layer.banks[2],
            if folded { &b[13] } else { &b[14] },
            &b[10],
            status,
            &b[15],
            c.intermediate,
            c.hidden,
            c.experts,
            c.topk,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_combine(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        x: &MetalTensor,
        output: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        let extents = composition_extents(c, 1)?;
        let b = &self.buffers;
        let status = &self.statuses[id];
        let constants = &self.constants[id];
        if self.folded_combine(c, layer, id, x) {
            metal::dispatch_profile_label("ffn_post", Some(id));
            return hc_gpu::encode_ffn_post(
                ctx,
                enc,
                c,
                constants,
                hc_gpu::FfnPostSite {
                    slots: &b[15],
                    weights: &b[11],
                    shared: &b[19],
                    status,
                    residual: x,
                    mix: &b[2],
                    pre: &b[3],
                    post: &b[4],
                    combination: &b[5],
                    output,
                },
            );
        }
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        enc.set_bytes(0, &(c.hidden as u32));
        enc.set_bytes(1, &(c.topk as u32));
        for (slot, tensor) in [&b[15], &b[19], &b[20], status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.set_bytes(6, &1u32);
        dispatch(enc, extents.output as usize, 256);
        metal::dispatch_profile_label("ffn_post", Some(id));
        let wide = hc_gpu::wide(&layer.ffn_hc, c.hidden, x, &b[6], &b[7], &constants[2]);
        hc_gpu::encode_post(
            ctx,
            enc,
            c,
            wide,
            constants,
            hc_gpu::HcPostSite {
                block: &b[20],
                residual: x,
                mix: &b[2],
                pre: &b[3],
                post: &b[4],
                combination: &b[5],
                output,
            },
        )
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
        let mut e = enc;
        self.encode_rows_with(ctx, &mut e, c, layer, id, x, incoming, output, scratch)
    }

    /// `encode_rows` with stage marks (`PREFILL_FFN_STAGES`) for a sampled
    /// prefill profile; a plain encoder ignores them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rows_with(
        &self,
        ctx: &MetalContext,
        e: &mut impl RowsEncoder,
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
        scratch.hc.encode_with_arithmetic(
            ctx,
            e.enc(),
            c,
            &layer.ffn_hc.projection,
            &self.constants[id],
            x,
            incoming,
            scratch.arithmetic,
        )?;
        e.mark(1);
        layer.router.encode_rows_with_arithmetic(
            e.enc(),
            &scratch.hc.normalized,
            &b[0],
            n,
            scratch.arithmetic,
        )?;
        // One half copy of the normalized input for the shared gate/up here
        // and the routed gate/up below.
        let input_half = match &scratch.half_input {
            Some(half)
                if layer.shared[..2]
                    .iter()
                    .any(|w| w.tensor_rows_selected_with_arithmetic(n, scratch.arithmetic))
                    || (half_grouped_selected_with_arithmetic(c, layer, n, scratch.arithmetic)
                        && metal::moe_grouped_tensor_selected(ctx)) =>
            {
                Some(half.convert(ctx, e.enc(), &scratch.hc.normalized, c.hidden * n)?)
            }
            _ => None,
        };
        e.mark(2);
        for j in 0..2 {
            layer.shared[j].encode_rows_with_arithmetic_and_half(
                e.enc(),
                &scratch.hc.normalized,
                input_half.as_ref(),
                &b[1 + j],
                n,
                scratch.arithmetic,
            )?;
        }
        let shape = vec![(c.intermediate * n) as u64];
        metal::encode_clamped_swiglu(
            ctx,
            e.enc(),
            &b[1].view_bytes(0, shape.clone()),
            &b[2].view_bytes(0, shape.clone()),
            &b[3].view_bytes(0, shape),
            c.clamp,
        )?;
        let shared_half = match &scratch.half_inner {
            Some(half)
                if layer.shared[2].tensor_rows_selected_with_arithmetic(n, scratch.arithmetic) =>
            {
                Some(half.convert(ctx, e.enc(), &b[3], c.intermediate * n)?)
            }
            _ => None,
        };
        layer.shared[2].encode_rows_with_arithmetic_and_half(
            e.enc(),
            &b[3],
            shared_half.as_ref(),
            &b[4],
            n,
            scratch.arithmetic,
        )?;
        if half_grouped_selected_with_arithmetic(c, layer, n, scratch.arithmetic)
            || grouped_selected(c, layer, n)
            || native_rows_supported(c, layer)
        {
            self.encode_routed_rows_with(ctx, e, c, layer, id, scratch, input_half.as_ref())?;
        } else {
            for stage in 3..=7 {
                e.mark(stage);
            }
            let status = scratch.status_view(id)?;
            for row in 0..n {
                self.encode_routed(
                    ctx,
                    e.enc(),
                    c,
                    layer,
                    &self.constants[id][3],
                    &prefill::row(&scratch.hc.normalized, c.hidden, row),
                    &prefill::row(&b[0], c.experts, row),
                    &prefill::row(&b[4], c.hidden, row),
                    &prefill::row(&b[5], c.hidden, row),
                    &prefill::row(&status, 1, row),
                )?;
            }
        }
        metal::encode_mhc4_post_rows(
            ctx,
            e.enc(),
            c.hidden,
            n,
            &b[5],
            x,
            &scratch.hc.post,
            &scratch.hc.comb,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_routed_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        scratch: &Rows,
        input_half: Option<&MetalTensor>,
    ) -> std::result::Result<(), metal::MetalError> {
        let mut e = enc;
        self.encode_routed_rows_with(ctx, &mut e, c, layer, id, scratch, input_half)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_routed_rows_with(
        &self,
        ctx: &MetalContext,
        e: &mut impl RowsEncoder,
        c: &Config,
        layer: &Layer<'_>,
        id: usize,
        scratch: &Rows,
        input_half: Option<&MetalTensor>,
    ) -> std::result::Result<(), metal::MetalError> {
        let n = scratch.hc.rows;
        let extents = composition_extents(c, n)?;
        let b = &scratch.buffers;
        let status = scratch.status_view(id)?;
        let counts = b[10].view_subrange(0, vec![c.experts as u64]);
        e.mark(3);
        encode_route_rows(
            ctx,
            e.enc(),
            c,
            n,
            &b[0],
            &self.constants[id][3],
            &b[7],
            &b[8],
            &b[9],
            &status,
        )?;
        let half = half_grouped_selected_with_arithmetic(c, layer, n, scratch.arithmetic);
        // The Metal 4 tensor tile is the half route's production form where
        // the tier exists: same operands and contract, matmul2d instead of
        // simdgroup MMA.
        let tensor = half && metal::moe_grouped_tensor_selected(ctx);
        let grouped = half || grouped_selected(c, layer, n);
        if grouped {
            let bucket = if route_bucket_parallel() {
                metal::encode_moe_route_bucket_slots_parallel_f32
            } else {
                metal::encode_moe_route_bucket_slots_f32
            };
            bucket(ctx, e.enc(), &b[8], &counts, &b[11], c.experts, n, c.topk)?;
            e.mark(4);
            if tensor {
                // The caller's half copy of the normalized input when it made
                // one (the shared gate/up read it too); otherwise convert here.
                let owned_half = match (input_half, &scratch.half_input) {
                    (None, Some(half)) => {
                        Some(half.convert(ctx, e.enc(), &scratch.hc.normalized, c.hidden * n)?)
                    }
                    _ => None,
                };
                let x_pack = input_half
                    .or(owned_half.as_ref())
                    .unwrap_or(&scratch.hc.normalized);
                metal::encode_moe_swiglu_f32_grouped_slots_tensor(
                    ctx,
                    e.enc(),
                    layer.banks[0],
                    layer.banks[1],
                    x_pack,
                    &counts,
                    &b[11],
                    &b[12],
                    c.hidden,
                    c.intermediate,
                    c.experts,
                    c.topk,
                    n,
                    Some(c.clamp),
                )?;
            } else {
                let gate_up = if half {
                    metal::encode_moe_swiglu_clamped_f32_grouped_slots_generic
                } else {
                    metal::encode_moe_swiglu_clamped_f32x_grouped_slots
                };
                gate_up(
                    ctx,
                    e.enc(),
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
            }
        } else {
            e.mark(4);
            metal::encode_all_slots_gate_up_swiglu_native_rows(
                ctx,
                e.enc(),
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
        e.mark(5);
        e.enc()
            .set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight")?);
        e.enc().set_bytes(0, &extents.weighted);
        e.enc().set_bytes(1, &(c.intermediate as u32));
        for (slot, tensor) in [&b[12], &b[9], &b[13], &status].iter().enumerate() {
            e.enc().set_tensor(slot + 2, tensor);
        }
        e.enc().set_bytes(6, &(c.topk as u32));
        dispatch(e.enc(), extents.weighted as usize, 256);
        if grouped {
            let down = if tensor {
                metal::encode_moe_down_f32_grouped_slots_tensor
            } else if half {
                metal::encode_moe_down_f32_grouped_slots_generic
            } else {
                metal::encode_moe_down_f32x_grouped_slots
            };
            let inner_half = match &scratch.half_inner {
                Some(half) if tensor => {
                    Some(half.convert(ctx, e.enc(), &b[13], c.intermediate * c.topk * n)?)
                }
                _ => None,
            };
            e.mark(6);
            down(
                ctx,
                e.enc(),
                layer.banks[2],
                inner_half.as_ref().unwrap_or(&b[13]),
                &counts,
                &b[11],
                &b[14],
                c.intermediate,
                c.hidden,
                c.experts,
                n,
            )?;
        } else {
            e.mark(6);
            metal::encode_all_slots_down_native_rows(
                ctx,
                e.enc(),
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
        e.mark(7);
        e.enc()
            .set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        e.enc().set_bytes(0, &(c.hidden as u32));
        e.enc().set_bytes(1, &(c.topk as u32));
        for (slot, tensor) in [&b[14], &b[4], &b[5], &status].iter().enumerate() {
            e.enc().set_tensor(slot + 2, tensor);
        }
        e.enc().set_bytes(6, &(n as u32));
        dispatch(e.enc(), extents.output as usize, 256);
        Ok(())
    }
}

/// Prefill FFN stage names in mark order (`QWEN_V41_PREFILL_STAGES`): HC pre,
/// router, shared expert, route + bucket, routed gate/up, routing weights +
/// half copy, routed down, sum + HC post.
pub(super) const PREFILL_FFN_STAGES: [&str; 8] = [
    "prefill_ffn_hc",
    "prefill_ffn_router",
    "prefill_ffn_shared",
    "prefill_ffn_route",
    "prefill_ffn_gate_up",
    "prefill_ffn_mid",
    "prefill_ffn_down",
    "prefill_ffn_sum",
];

/// The prefill FFN's encoder: plain encoding ignores stage marks; a sampled
/// profile ends the current encoder at each mark and begins the next.
pub(super) trait RowsEncoder {
    fn enc(&self) -> &KernelEncoder;
    fn mark(&mut self, _stage: usize) {}
}

impl RowsEncoder for &KernelEncoder {
    fn enc(&self) -> &KernelEncoder {
        self
    }
}

/// Sampled stage encoder: stage k (1..8) records samples base + 2(k - 1)
/// and base + 2(k - 1) + 1; stage 0 is the caller's current encoder.
pub(super) struct SampledRows<'a> {
    pub enc: &'a mut KernelEncoder,
    pub command:
        &'a objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    pub samples: &'a metal::MetalTimestampSampleBuffer,
    pub base: usize,
}

impl RowsEncoder for SampledRows<'_> {
    fn enc(&self) -> &KernelEncoder {
        self.enc
    }
    fn mark(&mut self, stage: usize) {
        self.enc.finish();
        let first = self.base + 2 * (stage - 1);
        *self.enc =
            KernelEncoder::begin_sampled(self.command, self.samples, first, first + 1, false);
    }
}

/// The half-staged grouped expert tiles GLM and DeepSeek V4 prefill use:
/// operands rounded to half inside the tile, F32 sums, weights read once per
/// tile rather than once per row. Covers every expert storage type with a
/// generic grouped layout (IQ2_S, IQ3_XXS, IQ3_S, Q2_K, Q3_K, MXFP4,
/// IQ2_XXS, IQ2_XS, ...). Where the Metal 4 tensor tier exists the route
/// runs the same operands through the tensor tile.
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
    half_grouped_selected_with_arithmetic(c, layer, rows, RowsArithmetic::Ordinary)
}

pub(super) fn half_grouped_selected_with_arithmetic(
    c: &Config,
    layer: &Layer<'_>,
    rows: usize,
    arithmetic: RowsArithmetic,
) -> bool {
    // Forcing the grouped route in tests means the F32-operand tiles, whose
    // frozen witnesses hold a 1e-5 contract the half tiles do not claim.
    #[cfg(test)]
    if tests::FORCE_GROUPED_ROWS.with(|flag| flag.get()) {
        return false;
    }
    arithmetic.packed_operands(rows) && half_grouped_supported(c, layer)
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
    arithmetic: RowsArithmetic,
    // Activations are reused; buffer 6 retains layer-major [layer][row] statuses.
    buffers: Vec<MetalTensor>,
    /// Half copy of the normalized input (shared and routed gate/up read
    /// it, with the shared down in between) and of the down inputs (the
    /// shared, then the routed slots); `None` when the tier is not selected.
    half_input: Option<prefill::HalfScratch>,
    half_inner: Option<prefill::HalfScratch>,
    // Constructor provenance; checked views must not silently lose half staging.
    half_required: [bool; 2],
    ced_status: Option<(usize, MetalTensor)>,
}
impl Rows {
    fn status_view(&self, id: usize) -> std::result::Result<MetalTensor, metal::MetalError> {
        if let Some((layer, status)) = &self.ced_status {
            if *layer != id {
                return Err(metal::MetalError::BadShape {
                    kernel: "v41_ffn_ced_status",
                    detail: "status producer mismatch".into(),
                });
            }
            Ok(status.clone())
        } else {
            Ok(
                self.buffers[6]
                    .view_subrange((id * self.hc.rows) as u64, vec![self.hc.rows as u64]),
            )
        }
    }

    fn prefix_widths(
        c: &Config,
        backing_rows: usize,
        rows: usize,
        layers: usize,
    ) -> Result<Vec<usize>> {
        prefill::checked_scratch_rows(backing_rows, rows)?;
        composition_extents(c, backing_rows).map_err(|e| e.to_string())?;
        composition_extents(c, rows).map_err(|e| e.to_string())?;
        op::require(layers > 0, "FFN prefix requires layers")?;
        let widths = Self::widths(c, layers);
        for &width in &widths {
            width
                .checked_mul(backing_rows)
                .and_then(|n| n.checked_mul(4))
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or("FFN prefix backing extent overflow")?;
        }
        Ok(widths)
    }

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
        Self::with_arithmetic(ctx, c, rows, layers, RowsArithmetic::Ordinary)
    }

    pub(super) fn with_arithmetic(
        ctx: &MetalContext,
        c: &Config,
        rows: usize,
        layers: usize,
        arithmetic: RowsArithmetic,
    ) -> Result<Self> {
        composition_extents(c, rows).map_err(|e| e.to_string())?;
        let [input_elements, inner_elements] = prefill::ffn_half_scratch_elements(c, rows)?;
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
        let half_input = prefill::HalfScratch::new(ctx, input_elements)?;
        let half_inner = prefill::HalfScratch::new(ctx, inner_elements)?;
        Ok(Self {
            hc,
            arithmetic,
            buffers,
            half_required: [half_input.is_some(), half_inner.is_some()],
            half_input,
            half_inner,
            ced_status: None,
        })
    }

    /// The driver must retire all users of the backing before this call: the
    /// compact [layer][current row] status slab is reset on the host.
    pub(super) fn checked_prefix(&self, c: &Config, rows: usize, layers: usize) -> Result<Self> {
        self.prefix(c, rows, layers, None)
    }

    pub(super) fn checked_ced_prefix(
        &self,
        c: &Config,
        rows: usize,
        layers: usize,
        layer: usize,
        status: &MetalTensor,
    ) -> Result<Self> {
        op::require(layer < layers, "CED FFN status layer")?;
        prefill::checked_scratch_view(status, &[rows], &[rows], GgmlType::I32)?;
        self.prefix(c, rows, layers, Some((layer, status.clone())))
    }

    fn prefix(
        &self,
        c: &Config,
        rows: usize,
        layers: usize,
        ced_status: Option<(usize, MetalTensor)>,
    ) -> Result<Self> {
        let widths = Self::prefix_widths(c, self.hc.rows, rows, layers)?;
        op::require(
            [self.half_input.is_some(), self.half_inner.is_some()] == self.half_required,
            "FFN prefix half backing presence differs from constructor",
        )?;
        op::require(
            self.buffers.len() == widths.len() - 6,
            "FFN scratch buffers",
        )?;
        let hc = self.hc.checked_prefix(c, rows)?;
        let buffers = self
            .buffers
            .iter()
            .zip(&widths[6..])
            .enumerate()
            .map(|(i, (tensor, &width))| {
                prefill::checked_scratch_view(
                    tensor,
                    &[width, self.hc.rows],
                    &[width, rows],
                    if matches!(i, 6 | 8 | 10 | 11) {
                        GgmlType::I32
                    } else {
                        GgmlType::F32
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let [input_elements, inner_elements] = prefill::ffn_half_scratch_elements(c, rows)?;
        let half_input = self
            .half_input
            .as_ref()
            .map(|half| half.checked_prefix(input_elements))
            .transpose()?;
        let half_inner = self
            .half_inner
            .as_ref()
            .map(|half| half.checked_prefix(inner_elements))
            .transpose()?;

        // Every view has passed validation. Reset only active statuses, not the
        // large overwrite-before-read activations or unused maximum-row tail.
        let status = &buffers[6];
        if ced_status.is_none() {
            unsafe {
                std::ptr::write_bytes(
                    status
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(status.offset as usize)
                        .cast::<i32>(),
                    0,
                    status.n_elements() as usize,
                );
            }
        }
        Ok(Self {
            hc,
            arithmetic: RowsArithmetic::Packed,
            buffers,
            half_input,
            half_inner,
            half_required: self.half_required,
            ced_status,
        })
    }

    pub(super) fn validate_status(&self) -> Result<()> {
        self.validate_status_layers(0..self.buffers[6].n_elements() as usize / self.hc.rows)
    }

    pub(super) fn validate_status_layers(&self, layers: std::ops::Range<usize>) -> Result<()> {
        if let Some((layer, status)) = &self.ced_status {
            op::require(layers == (*layer..*layer + 1), "CED FFN status scope")?;
            return validate_ced_status(status, *layer);
        }
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
        op::require(
            layers.start <= layers.end && layers.end <= values.len() / self.hc.rows,
            "prefill FFN status layer range",
        )?;
        for (index, &status) in values
            .iter()
            .enumerate()
            .take(layers.end * self.hc.rows)
            .skip(layers.start * self.hc.rows)
        {
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

pub(super) fn validate_ced_status(t: &MetalTensor, layer: usize) -> Result<()> {
    op::require(
        t.dtype == GgmlType::I32
            && t.offset.is_multiple_of(4)
            && t.n_elements() > 0
            && t.n_elements()
                .checked_mul(4)
                .and_then(|n| t.offset.checked_add(n))
                .is_some_and(|end| end <= t.buffer.length() as u64),
        "CED FFN status view",
    )?;
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
    for (row, &status) in values.iter().enumerate() {
        op::require(
            status == metal::ROUTE_STATUS_READY,
            &format!("prefill FFN layer {layer} row {row} route status {status}"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod scratch_prefix_geometry_tests {
    use super::*;

    #[test]
    #[ignore = "self-leased actual scratch buffers; MTL debug layer; no model or dispatch"]
    fn ffn_actual_prefix_resets_only_active_statuses_after_all_checks() {
        assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
        assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
        assert!(!crate::env_flag::read_default_off(
            "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
        ));
        let _lease = metal::acquire_metal_benchmark_lease().unwrap();
        let ctx = MetalContext::new().unwrap();
        let c = super::super::toy::config();
        let (maximum, layers) = (8, 3);
        let widths = Rows::widths(&c, layers);
        let half = prefill::ffn_half_scratch_elements(&c, maximum).unwrap();
        let bytes = widths
            .iter()
            .map(|&w| (w * maximum * 4) as u64)
            .chain(half.map(|n| (n * 2) as u64))
            .chain([64 * 4])
            .map(|n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
            .sum::<u64>();
        assert!(bytes < 16 * 1024 * 1024);
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            bytes,
            1024 * 1024,
            0,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.refusal().is_none());

        fn words(t: &MetalTensor) -> &[u32] {
            unsafe {
                std::slice::from_raw_parts(
                    t.buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(t.offset as usize)
                        .cast::<u32>(),
                    t.n_elements() as usize,
                )
            }
        }
        fn poison(t: &MetalTensor) {
            unsafe {
                std::slice::from_raw_parts_mut(
                    t.buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(t.offset as usize)
                        .cast::<u32>(),
                    t.n_elements() as usize,
                )
            }
            .iter_mut()
            .enumerate()
            .for_each(|(i, word)| *word = 0xdead0000 + i as u32);
        }
        fn activations(rows: &Rows) -> Vec<Vec<u32>> {
            rows.buffers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != 6)
                .map(|(_, t)| t)
                .chain([
                    &rows.hc.pre,
                    &rows.hc.post,
                    &rows.hc.comb,
                    &rows.hc.normalized,
                ])
                .map(|t| words(t).to_vec())
                .collect()
        }

        objc2::rc::autoreleasepool(|_| {
            let mut backing =
                Rows::with_arithmetic(&ctx, &c, maximum, layers, RowsArithmetic::Packed).unwrap();
            for t in backing.buffers.iter().chain([
                &backing.hc.pre,
                &backing.hc.post,
                &backing.hc.comb,
                &backing.hc.normalized,
            ]) {
                poison(t);
            }
            let before = activations(&backing);
            for n in [8, 1, 5] {
                poison(&backing.buffers[6]);
                let status_before = words(&backing.buffers[6]).to_vec();
                let view = backing.checked_prefix(&c, n, layers).unwrap();
                assert_eq!(view.hc.rows, n);
                assert_eq!(backing.hc.rows, maximum);
                assert_eq!(view.half_required, backing.half_required);
                for (index, (view, original)) in
                    view.buffers.iter().zip(&backing.buffers).enumerate()
                {
                    assert!(std::ptr::eq(&*view.buffer, &*original.buffer));
                    assert_eq!(view.offset, original.offset);
                    assert_eq!(view.n_elements(), (widths[index + 6] * n) as u64);
                }
                assert!(
                    words(&backing.buffers[6])[..layers * n]
                        .iter()
                        .all(|&word| word == 0)
                );
                assert_eq!(
                    &words(&backing.buffers[6])[layers * n..],
                    &status_before[layers * n..]
                );
                assert_eq!(activations(&backing), before);
            }

            poison(&backing.buffers[6]);
            let status_before = words(&backing.buffers[6]).to_vec();
            let ledger = prefill::allocate(&ctx, 64, 1, GgmlType::I32).unwrap();
            poison(&ledger);
            let ledger_before = words(&ledger).to_vec();
            for n in [8, 1, 5] {
                let scoped = ledger.view_subrange(16, vec![n as u64]);
                let view = backing
                    .checked_ced_prefix(&c, n, layers, 2, &scoped)
                    .unwrap();
                assert_eq!(view.status_view(2).unwrap().offset, scoped.offset);
                assert!(view.status_view(1).is_err());
                assert!(view.validate_status_layers(1..2).is_err());
                assert_eq!(
                    words(&backing.buffers[6]),
                    status_before,
                    "CED never CPU-clears the compact slab"
                );
                assert_eq!(
                    words(&ledger),
                    ledger_before,
                    "checked scoped views never mutate status or canaries"
                );
            }
            let scoped = ledger.view_subrange(16, vec![1]);
            let mut misaligned = scoped.clone();
            misaligned.offset += 1;
            let mut wrong_dtype = scoped.clone();
            wrong_dtype.dtype = GgmlType::F32;
            for (rows, layer, status) in [
                (0, 2, &scoped),
                (9, 2, &scoped),
                (1, 3, &scoped),
                (2, 2, &scoped),
                (1, 2, &misaligned),
                (1, 2, &wrong_dtype),
            ] {
                assert!(
                    backing
                        .checked_ced_prefix(&c, rows, layers, layer, status)
                        .is_err()
                );
                assert_eq!(words(&backing.buffers[6]), status_before);
                assert_eq!(words(&ledger), ledger_before);
            }
            let last = backing.buffers.pop().unwrap();
            backing
                .buffers
                .push(last.view_subrange(0, vec![last.shape[0], 7]));
            assert!(backing.checked_prefix(&c, 1, layers).is_err());
            assert_eq!(words(&backing.buffers[6]), status_before);
            *backing.buffers.last_mut().unwrap() = last;
            assert_eq!(activations(&backing), before);

            if let Some(original) = backing.half_input.take() {
                assert!(backing.half_required[0]);
                assert!(backing.checked_prefix(&c, 1, layers).is_err());
                assert_eq!(words(&backing.buffers[6]), status_before);
                assert_eq!(activations(&backing), before);
                backing.half_input = Some(original);
            }
            if let Some(original) = backing.half_inner.take() {
                assert!(backing.half_required[1]);
                assert!(backing.checked_prefix(&c, 1, layers).is_err());
                assert_eq!(words(&backing.buffers[6]), status_before);
                assert_eq!(activations(&backing), before);
                backing.half_inner = Some(original.checked_prefix(1).unwrap());
                assert!(backing.checked_prefix(&c, 1, layers).is_err());
                assert_eq!(words(&backing.buffers[6]), status_before);
                assert_eq!(activations(&backing), before);
                backing.half_inner = Some(original);
            }
        });
    }

    #[test]
    fn ffn_prefix_geometry_compacts_statuses_and_slot_strides() {
        let c = Config::default();
        let layers = 20;
        for rows in [1, 3, 127, 512, prefill::DEFAULT_CHUNK_ROWS] {
            let widths =
                Rows::prefix_widths(&c, prefill::DEFAULT_CHUNK_ROWS, rows, layers).unwrap();
            let active = widths[6..].iter().map(|w| w * rows).collect::<Vec<_>>();
            let extents = composition_extents(&c, rows).unwrap();
            assert_eq!(active[6], layers * rows);
            assert_eq!((layers - 1) * rows + rows, active[6]);
            assert_eq!(active[8], extents.slots as usize);
            assert_eq!(active[9], extents.slots as usize);
            assert_eq!(active[10], c.experts * rows);
            assert_eq!(active[11], c.experts * rows);
            assert_eq!(active[12], extents.weighted as usize);
            assert_eq!(active[13], extents.weighted as usize);
            assert_eq!(active[14], c.hidden * c.topk * rows);
            assert!(RowsArithmetic::Packed.packed_operands(rows));
        }
    }

    #[test]
    fn ffn_prefix_geometry_rejects_empty_oversize_and_overflow() {
        let mut c = Config::default();
        for (backing, rows, layers) in [(8, 0, 2), (8, 9, 2), (2049, 1, 2), (8, 1, 0)] {
            assert!(Rows::prefix_widths(&c, backing, rows, layers).is_err());
        }
        assert!(Rows::prefix_widths(&c, 8, 1, usize::MAX).is_err());
        c.intermediate = usize::MAX;
        assert!(Rows::prefix_widths(&c, 8, 1, 2).is_err());
    }
}
