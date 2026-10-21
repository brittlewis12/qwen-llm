//! Native GPU-indexed expert projections with F32 weights after decode,
//! activations and accumulation. No weight materialization, ID readback or
//! additional scratch. Exact refers to operand precision, not reduction order.
//!
//! Banks are contiguous `[K, M, experts]`; gate/up input is `[K]`, their
//! output is `[M, top_k]`. Down reads `[K, top_k]` and writes `[M, top_k]`.
//! Quantized K must be block-aligned; dense K and all M may be any positive
//! size. Dimensions fit u32, addresses/row byte strides use u64. Bank offsets
//! are 16-byte aligned; activations, IDs and status are 4-byte aligned.
//! Serial encoders are required. Output must be writable and disjoint from
//! every input. Invalid IDs and any status other than READY produce zeros.
//! Row-batched entrypoints add only a grid-depth/token axis, retaining the
//! native per-output reduction and the separate gate/up passes and F32 store.

use super::checks::{
    bad_shape, check_disjoint, check_expert_bank, check_tensor, require_serial, to_u32,
};
use super::*;

/// Actual native shader mapping, shared by all three projection modes.
/// Unsupported types return None; there is no implicit dtype fallback.
fn native_pipeline_name(dtype: GgmlType) -> Option<String> {
    let suffix = match dtype {
        GgmlType::Q2_0 => "q2_0",
        GgmlType::Q2_K => "q2_K",
        GgmlType::Q3_K => "q3_K",
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q6_K => "q6_K",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::IQ2_S => "iq2_s",
        GgmlType::IQ3_XXS => "iq3_xxs",
        GgmlType::IQ3_S => "iq3_s",
        GgmlType::IQ4_NL => "iq4_nl",
        GgmlType::IQ4_XS => "iq4_xs",
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::BF16 => "bf16",
        GgmlType::IQ2_XS => "iq2_xs",
        GgmlType::IQ2_XXS => "iq2_xxs",
        GgmlType::MXFP4 => "mxfp4",
        _ => return None,
    };
    Some(format!("kernel_expert_native_{suffix}_f32"))
}

fn fused_pipeline_name(dtype: GgmlType, family: &str) -> Option<String> {
    let native = native_pipeline_name(dtype)?;
    let suffix = native
        .strip_prefix("kernel_expert_native_")?
        .strip_suffix("_f32")?;
    Some(format!("kernel_expert_native_{family}_{suffix}_f32"))
}

fn gate_up_weight_pipeline_name(dtype: GgmlType) -> Option<String> {
    fused_pipeline_name(dtype, "gate_up_weight")
}

fn down_sum_pipeline_name(dtype: GgmlType) -> Option<String> {
    fused_pipeline_name(dtype, "down_sum")
}

/// Fused decode gate/up + SwiGLU + route weight. Equal storage types only;
/// MXFP4 keeps the native path because its subnormal rescaling is per bank.
pub fn all_slots_gate_up_weight_native_supported(gate: GgmlType, up: GgmlType) -> bool {
    gate == up && gate != GgmlType::MXFP4 && gate_up_weight_pipeline_name(gate).is_some()
}

/// Fused decode down + ordered top-k sum + shared branch.
pub fn all_slots_down_sum_native_supported(dtype: GgmlType) -> bool {
    dtype != GgmlType::MXFP4 && down_sum_pipeline_name(dtype).is_some()
}

/// Independently typed native gate/up capability. Shape constraints still apply.
pub fn all_slots_gate_up_native_supported(gate: GgmlType, up: GgmlType) -> bool {
    native_pipeline_name(gate).is_some() && native_pipeline_name(up).is_some()
}

/// Native down capability (the same storage formats as each gate/up bank).
pub fn all_slots_down_native_supported(dtype: GgmlType) -> bool {
    native_pipeline_name(dtype).is_some()
}

/// Row-batched gate/up coverage; single-token capability is unchanged.
pub fn all_slots_gate_up_native_rows_supported(gate: GgmlType, up: GgmlType) -> bool {
    [gate, up].into_iter().all(|t| {
        matches!(
            t,
            GgmlType::IQ3_XXS | GgmlType::Q3_K | GgmlType::IQ2_XS | GgmlType::IQ2_XXS
        )
    })
}

pub fn all_slots_down_native_rows_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::Q3_K | GgmlType::MXFP4 | GgmlType::IQ2_XXS
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NativeArgs {
    row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
    mode: u32,
    clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GateUpWeightArgs {
    gate_row_bytes: u64,
    up_row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
    clamp: f32,
    /// Matches the shader struct's trailing alignment padding.
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DownSumArgs {
    row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
}

fn dimensions(
    kernel: &'static str,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
) -> Result<NativeArgs, MetalError> {
    if n_in == 0 || n_out == 0 || experts == 0 || top_k == 0 || top_k > ROUTE_MAX_TOP_K {
        return Err(bad_shape(
            kernel,
            "positive dimensions and top-k in 1..=16 required",
        ));
    }
    Ok(NativeArgs {
        row_bytes: 0,
        n_in: to_u32(kernel, n_in, "n_in")?,
        n_out: to_u32(kernel, n_out, "n_out")?,
        experts: to_u32(kernel, experts, "experts")?,
        top_k: to_u32(kernel, top_k, "top_k")?,
        mode: 0,
        clamp: 0.0,
    })
}

fn bank_pipeline(
    ctx: &MetalContext,
    kernel: &'static str,
    bank: &MetalTensor,
    args: &NativeArgs,
    name: &str,
) -> Result<(Pipeline, u64), MetalError> {
    u64::from(args.n_in)
        .checked_mul(u64::from(args.n_out))
        .and_then(|n| n.checked_mul(u64::from(args.experts)))
        .ok_or_else(|| bad_shape(kernel, "expert bank element count overflow"))?;
    let pipeline = native_pipeline_name(bank.dtype).ok_or_else(|| {
        bad_shape(
            kernel,
            format!("no native expert kernel for {:?}", bank.dtype),
        )
    })?;
    check_expert_bank(
        kernel,
        bank,
        args.n_in as usize,
        args.n_out as usize,
        args.experts as usize,
        name,
    )?;
    let (block, bytes) = bank.dtype.storage_layout().expect("validated bank layout");
    let row_bytes = u64::from(args.n_in) / block * bytes;
    let pso = ctx.pipeline(&pipeline)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(
            kernel,
            "native expert kernel needs SIMD width 32 and 128 threads",
        ));
    }
    Ok((pso, row_bytes))
}

#[allow(clippy::too_many_arguments)]
fn check_views(
    kernel: &'static str,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    args: &NativeArgs,
    down: bool,
    rows: Option<usize>,
) -> Result<(), MetalError> {
    let n = to_u32(kernel, rows.unwrap_or(1), "rows")?;
    if n == 0 {
        return Err(bad_shape(kernel, "positive rows required"));
    }
    let slots = u64::from(args.top_k) * u64::from(n);
    for (width, count) in [
        (args.n_in, if down { slots } else { u64::from(n) }),
        (args.n_out, slots),
    ] {
        u64::from(width)
            .checked_mul(count)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| bad_shape(kernel, "row-batched byte extent overflow"))?;
    }
    let input_shape = if down {
        vec![u64::from(args.n_in), slots]
    } else if rows.is_some() {
        vec![u64::from(args.n_in), u64::from(n)]
    } else {
        vec![u64::from(args.n_in)]
    };
    let ids_shape = if rows.is_some() {
        vec![u64::from(args.top_k), u64::from(n)]
    } else {
        vec![u64::from(args.top_k)]
    };
    check_tensor(kernel, input, GgmlType::F32, &input_shape, false, "input")?;
    check_tensor(kernel, ids, GgmlType::I32, &ids_shape, false, "expert ids")?;
    check_tensor(
        kernel,
        status,
        GgmlType::I32,
        &[u64::from(n)],
        false,
        "route status",
    )?;
    check_tensor(
        kernel,
        output,
        GgmlType::F32,
        &[u64::from(args.n_out), slots],
        true,
        "output",
    )?;
    check_disjoint(
        kernel,
        output,
        &[
            (input, "input"),
            (ids, "expert ids"),
            (status, "route status"),
        ],
    )
}

fn check_device(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &'static str,
    tensors: &[&MetalTensor],
) -> Result<(), MetalError> {
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || tensors
            .iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            kernel,
            "encoder and bindings must belong to the context device",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    enc: &KernelEncoder,
    pso: &Pipeline,
    args: &NativeArgs,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    rows: usize,
) {
    enc.set_pipeline(pso);
    enc.set_bytes(0, args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: (args.n_out as usize).div_ceil(4),
            height: args.top_k as usize,
            depth: rows,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}

/// Two serial GPU-indexed projections, accepting independent gate/up types.
/// Gate first writes `output`; up then replaces each element with
/// `silu(min(g, clamp)) * clamp(u, -clamp, clamp)`. No extra scratch.
/// All bindings and both pipelines are validated before the first dispatch.
/// Clamp must be finite and positive. Route contents stay on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu_native(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    encode_gate_up(
        ctx, enc, gate_bank, up_bank, input, ids, status, output, n_in, n_out, experts, top_k,
        clamp, None,
    )
}

/// Token-batched native GEMV. Input `[K, rows]`, IDs `[top_k, rows]`, status
/// `[rows]`, output `[M, top_k * rows]`. Both banks/pipelines are preflighted
/// before either pass. No buckets, MMA, new scratch, or precision substitution.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu_native_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    if !all_slots_gate_up_native_rows_supported(gate_bank.dtype, up_bank.dtype) {
        return Err(bad_shape(
            "all_slots_gate_up_swiglu_native_rows",
            "unsupported row-batched gate/up storage",
        ));
    }
    encode_gate_up(
        ctx,
        enc,
        gate_bank,
        up_bank,
        input,
        ids,
        status,
        output,
        n_in,
        n_out,
        experts,
        top_k,
        clamp,
        Some(rows),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_gate_up(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    clamp: f32,
    rows: Option<usize>,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_gate_up_swiglu_native";
    require_serial(K, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(K, "clamp must be finite and positive"));
    }
    let mut args = dimensions(K, n_in, n_out, experts, top_k)?;
    check_views(K, input, ids, status, output, &args, false, rows)?;
    check_device(
        ctx,
        enc,
        K,
        &[gate_bank, up_bank, input, ids, status, output],
    )?;
    let (gate, gate_bytes) = bank_pipeline(ctx, K, gate_bank, &args, "gate bank")?;
    let (up, up_bytes) = bank_pipeline(ctx, K, up_bank, &args, "up bank")?;
    check_disjoint(K, output, &[(gate_bank, "gate bank"), (up_bank, "up bank")])?;
    args.clamp = clamp;
    args.row_bytes = gate_bytes;
    dispatch(
        enc,
        &gate,
        &args,
        gate_bank,
        input,
        ids,
        status,
        output,
        rows.unwrap_or(1),
    );
    args.mode = 1;
    args.row_bytes = up_bytes;
    dispatch(
        enc,
        &up,
        &args,
        up_bank,
        input,
        ids,
        status,
        output,
        rows.unwrap_or(1),
    );
    Ok(())
}

/// One GPU-indexed down projection per slot: `output[:,s] = bank[ids[s]] *
/// input[:,s]`. Input slots are independent, including duplicate expert IDs.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down_native(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
) -> Result<(), MetalError> {
    encode_down(
        ctx, enc, bank, input, ids, status, output, n_in, n_out, experts, top_k, None,
    )
}

/// Token/slot-batched down with the baseline native reduction. Input is
/// `[K, top_k * rows]`; other shapes match the row-batched gate/up entrypoint.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down_native_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
) -> Result<(), MetalError> {
    if !all_slots_down_native_rows_supported(bank.dtype) {
        return Err(bad_shape(
            "all_slots_down_native_rows",
            "unsupported row-batched down storage",
        ));
    }
    encode_down(
        ctx,
        enc,
        bank,
        input,
        ids,
        status,
        output,
        n_in,
        n_out,
        experts,
        top_k,
        Some(rows),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    rows: Option<usize>,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_down_native";
    require_serial(K, enc)?;
    let mut args = dimensions(K, n_in, n_out, experts, top_k)?;
    check_views(K, input, ids, status, output, &args, true, rows)?;
    check_device(ctx, enc, K, &[bank, input, ids, status, output])?;
    let (pso, row_bytes) = bank_pipeline(ctx, K, bank, &args, "down bank")?;
    check_disjoint(K, output, &[(bank, "down bank")])?;
    args.mode = 2;
    args.row_bytes = row_bytes;
    dispatch(
        enc,
        &pso,
        &args,
        bank,
        input,
        ids,
        status,
        output,
        rows.unwrap_or(1),
    );
    Ok(())
}

/// One fused decode pass: `output[:,s] = (silu(min(gate_s x, clamp)) *
/// clamp(up_s x, -clamp, clamp)) * weights[s]`. Gate and up decode share each
/// activation load; the arithmetic and SIMD reduction match the native gate,
/// up and weight passes. Equal storage types only. Route contents stay on GPU.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu_weight_native(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    weights: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_gate_up_swiglu_weight_native";
    require_serial(K, enc)?;
    if !all_slots_gate_up_weight_native_supported(gate_bank.dtype, up_bank.dtype) {
        return Err(bad_shape(
            K,
            format!(
                "unsupported fused gate/up storage {:?}/{:?}",
                gate_bank.dtype, up_bank.dtype
            ),
        ));
    }
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(K, "clamp must be finite and positive"));
    }
    let dims = dimensions(K, n_in, n_out, experts, top_k)?;
    check_tensor(
        K,
        input,
        GgmlType::F32,
        &[u64::from(dims.n_in)],
        false,
        "input",
    )?;
    check_tensor(
        K,
        ids,
        GgmlType::I32,
        &[u64::from(dims.top_k)],
        false,
        "expert ids",
    )?;
    check_tensor(K, status, GgmlType::I32, &[1], false, "route status")?;
    check_tensor(
        K,
        weights,
        GgmlType::F32,
        &[u64::from(dims.top_k)],
        false,
        "route weights",
    )?;
    check_tensor(
        K,
        output,
        GgmlType::F32,
        &[u64::from(dims.n_out), u64::from(dims.top_k)],
        true,
        "output",
    )?;
    check_disjoint(
        K,
        output,
        &[
            (gate_bank, "gate bank"),
            (up_bank, "up bank"),
            (input, "input"),
            (ids, "expert ids"),
            (status, "route status"),
            (weights, "route weights"),
        ],
    )?;
    check_device(
        ctx,
        enc,
        K,
        &[gate_bank, up_bank, input, ids, status, weights, output],
    )?;
    let (_, gate_bytes) = bank_pipeline(ctx, K, gate_bank, &dims, "gate bank")?;
    let (_, up_bytes) = bank_pipeline(ctx, K, up_bank, &dims, "up bank")?;
    if gate_bytes != up_bytes {
        return Err(bad_shape(K, "gate and up banks must share a row stride"));
    }
    let name = gate_up_weight_pipeline_name(gate_bank.dtype)
        .ok_or_else(|| bad_shape(K, "no fused gate/up kernel"))?;
    let pso = ctx.pipeline(&name)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(
            K,
            "fused gate/up kernel needs SIMD width 32 and 128 threads",
        ));
    }
    let args = GateUpWeightArgs {
        gate_row_bytes: gate_bytes,
        up_row_bytes: up_bytes,
        n_in: dims.n_in,
        n_out: dims.n_out,
        experts: dims.experts,
        top_k: dims.top_k,
        clamp,
        _padding: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, status);
    enc.set_tensor(6, weights);
    enc.set_tensor(7, output);
    enc.dispatch(
        MTLSize {
            width: dims.n_out as usize / 4 + usize::from(!dims.n_out.is_multiple_of(4)),
            height: dims.top_k as usize,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One fused decode pass: `output[:,r] = shared[:,r] + sum_s down_s input[:,s]`
/// with the top-k additions in slot order. Matches the native down reduction,
/// the ordered-sum kernel and the shared add. Input is `[K, top_k]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down_sum_native(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    shared: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_down_sum_native";
    require_serial(K, enc)?;
    if !all_slots_down_sum_native_supported(bank.dtype) {
        return Err(bad_shape(
            K,
            format!("unsupported fused down storage {:?}", bank.dtype),
        ));
    }
    let dims = dimensions(K, n_in, n_out, experts, top_k)?;
    check_tensor(
        K,
        input,
        GgmlType::F32,
        &[u64::from(dims.n_in), u64::from(dims.top_k)],
        false,
        "input",
    )?;
    check_tensor(
        K,
        ids,
        GgmlType::I32,
        &[u64::from(dims.top_k)],
        false,
        "expert ids",
    )?;
    check_tensor(K, status, GgmlType::I32, &[1], false, "route status")?;
    check_tensor(
        K,
        shared,
        GgmlType::F32,
        &[u64::from(dims.n_out)],
        false,
        "shared output",
    )?;
    check_tensor(
        K,
        output,
        GgmlType::F32,
        &[u64::from(dims.n_out)],
        true,
        "output",
    )?;
    check_disjoint(
        K,
        output,
        &[
            (bank, "down bank"),
            (input, "input"),
            (ids, "expert ids"),
            (status, "route status"),
            (shared, "shared output"),
        ],
    )?;
    check_device(ctx, enc, K, &[bank, input, ids, status, shared, output])?;
    let (_, row_bytes) = bank_pipeline(ctx, K, bank, &dims, "down bank")?;
    let name = down_sum_pipeline_name(bank.dtype)
        .ok_or_else(|| bad_shape(K, "no fused down/sum kernel"))?;
    let pso = ctx.pipeline(&name)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(
            K,
            "fused down/sum kernel needs SIMD width 32 and 128 threads",
        ));
    }
    let args = DownSumArgs {
        row_bytes,
        n_in: dims.n_in,
        n_out: dims.n_out,
        experts: dims.experts,
        top_k: dims.top_k,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.set_tensor(5, shared);
    enc.set_tensor(6, output);
    enc.dispatch(
        MTLSize {
            width: dims.n_out as usize / 4 + usize::from(!dims.n_out.is_multiple_of(4)),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests;
