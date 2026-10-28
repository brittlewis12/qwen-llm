//! Native GPU-indexed expert projections. Existing entrypoints use F32 weights
//! after decode, activations and accumulation. No weight materialization, ID
//! readback or additional scratch. Exact refers to operand precision, not reduction order.
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
//! Explicit `*_half_rows` entrypoints instead round each decoded coefficient and
//! activation to F16 before F32 FMA. The inner SwiGLU store stays F32; down rounds
//! it again on load. This matches Fast/Half operand storage, NOT grouped MMA bits:
//! each SIMD lane accumulates 16-element fragments, then SIMD-sums 32 partials.
//! No global precision policy is selected here and existing entrypoints stay F32.

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
        GgmlType::IQ1_S => "iq1_s",
        GgmlType::IQ1_M => "iq1_m",
        _ => return None,
    };
    if dtype == GgmlType::Q5_K && q5_k_split() {
        return Some("kernel_expert_native_q5_K_f32_split".to_string());
    }
    if dtype == GgmlType::IQ2_XXS && iq2_xxs_split() {
        return Some("kernel_expert_native_iq2_xxs_f32_split".to_string());
    }
    Some(format!("kernel_expert_native_{suffix}_f32"))
}

/// `one_row` keeps the one-row IQ2_XXS native decode kernels instead of
/// llama.cpp's mul_mv split (two channels per simdgroup); A/B lever.
const EXPERT_NATIVE_IQ2XXS_ENV: &str = "QWEN_EXPERT_NATIVE_IQ2XXS";

/// `one_row` keeps the one-row Q5_K native decode kernel instead of
/// llama.cpp's mul_mv work split; A/B lever.
const EXPERT_NATIVE_Q5K_ENV: &str = "QWEN_EXPERT_NATIVE_Q5K";

/// IQ2_XXS uses `kernel_expert_native_iq2_xxs_f32_split` (two channels per
/// simdgroup) unless `QWEN_EXPERT_NATIVE_IQ2XXS=one_row` (A/B).
fn iq2_xxs_split() -> bool {
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SPLIT.get_or_init(|| std::env::var(EXPERT_NATIVE_IQ2XXS_ENV).as_deref() != Ok("one_row"))
}

/// Output channels per 128-thread threadgroup of a single-projection native
/// dispatch (EXPERT_IQ2_ROWS = 2 channels per simdgroup for the IQ2_XXS split).
pub(crate) fn native_channels_per_threadgroup(dtype: GgmlType) -> usize {
    if dtype == GgmlType::IQ2_XXS && iq2_xxs_split() {
        8
    } else {
        4
    }
}

/// Q5_K uses `kernel_expert_native_q5_K_f32_split` (llama.cpp's work split)
/// unless `QWEN_EXPERT_NATIVE_Q5K=one_row` keeps the one-row kernel (A/B).
fn q5_k_split() -> bool {
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SPLIT.get_or_init(|| std::env::var(EXPERT_NATIVE_Q5K_ENV).as_deref() != Ok("one_row"))
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
            GgmlType::IQ3_XXS
                | GgmlType::Q3_K
                | GgmlType::IQ2_XS
                | GgmlType::IQ2_XXS
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
        )
    })
}

pub fn all_slots_down_native_rows_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::Q3_K
            | GgmlType::MXFP4
            | GgmlType::IQ2_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
    )
}

/// Bounded half-operand coverage for released GLM gate/up banks only.
pub fn all_slots_gate_up_native_half_rows_supported(gate: GgmlType, up: GgmlType) -> bool {
    [gate, up]
        .into_iter()
        .all(|t| matches!(t, GgmlType::IQ2_S | GgmlType::IQ3_S))
}

pub fn all_slots_down_native_half_rows_supported(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::IQ3_S | GgmlType::IQ4_XS)
}

// Shared with expert_native.metal. Projection values 0/1/2 and the 32-byte ABI
// remain unchanged; never replace the flags when advancing gate to up.
const NATIVE_HALF_OPERANDS: u32 = 1 << 8;

#[derive(Clone, Copy)]
enum NativeOperands {
    F32,
    Half,
}

impl NativeOperands {
    fn mode(self, projection: u32) -> u32 {
        projection
            | match self {
                Self::F32 => 0,
                Self::Half => NATIVE_HALF_OPERANDS,
            }
    }
}

fn check_half_rows(kernel: &'static str, rows: usize) -> Result<(), MetalError> {
    if !(1..=8).contains(&rows) {
        return Err(bad_shape(kernel, "half-operand packets require 1..=8 rows"));
    }
    Ok(())
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
            width: (args.n_out as usize).div_ceil(native_channels_per_threadgroup(bank.dtype)),
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
        None,
        NativeOperands::F32,
        false,
    )
}

/// `encode_all_slots_gate_up_swiglu_native` as one dispatch when gate and up
/// share a storage type with a fused kernel (`kernel_expert_native_gate_up_*`):
/// simdgroups 0-1 of each threadgroup compute two gate channels, 2-3 the same
/// up channels, and the epilogue reads the values the two-pass form stores
/// and reloads, so outputs are bitwise equal. Other storage pairs, or
/// `QWEN_EXPERT_NATIVE_FUSED_GATE_UP=0` (A/B), use the two dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu_native_fused(
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
        None,
        NativeOperands::F32,
        true,
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
        NativeOperands::F32,
        false,
    )
}

/// Two serial half-operand GEMVs, not a grouped MMA or an F32 fallback.
/// Accepts 1..=8 rows: input `[K, rows]`, IDs `[top_k, rows]`, status `[rows]`,
/// caller-provided inner/output `[M, top_k * rows]` (slot varies before token).
/// All bindings and both pipelines are preflighted before any dispatch. No
/// allocation, routing readback, buckets, or full-expert tiled grid is needed.
/// Clamp and SwiGLU use F32; results need not match grouped MMA bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu_native_half_rows(
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
    const K: &str = "all_slots_gate_up_swiglu_native_half_rows";
    check_half_rows(K, rows)?;
    if !all_slots_gate_up_native_half_rows_supported(gate_bank.dtype, up_bank.dtype) {
        return Err(bad_shape(K, "unsupported half-operand gate/up storage"));
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
        NativeOperands::Half,
        false,
    )
}

/// The fused pipeline for same-typed F32-operand banks, unless disabled by
/// `QWEN_EXPERT_NATIVE_FUSED_GATE_UP=0`.
fn fused_gate_up_pipeline(
    ctx: &MetalContext,
    gate: GgmlType,
    up: GgmlType,
    operands: NativeOperands,
) -> Result<Option<Pipeline>, MetalError> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED
        .get_or_init(|| std::env::var("QWEN_EXPERT_NATIVE_FUSED_GATE_UP").as_deref() != Ok("0"));
    if !enabled || gate != up || !matches!(operands, NativeOperands::F32) {
        return Ok(None);
    }
    let Some(name) = fused_gate_up_name(gate) else {
        return Ok(None);
    };
    ctx.pipeline(name).map(Some)
}

fn fused_gate_up_name(dtype: GgmlType) -> Option<&'static str> {
    Some(match dtype {
        GgmlType::Q2_K => "kernel_expert_native_gate_up_q2_K_f32",
        GgmlType::Q3_K => "kernel_expert_native_gate_up_q3_K_f32",
        GgmlType::Q4_K => "kernel_expert_native_gate_up_q4_K_f32",
        GgmlType::Q6_K => "kernel_expert_native_gate_up_q6_K_f32",
        GgmlType::Q8_0 => "kernel_expert_native_gate_up_q8_0_f32",
        GgmlType::IQ4_NL => "kernel_expert_native_gate_up_iq4_nl_f32",
        GgmlType::IQ4_XS => "kernel_expert_native_gate_up_iq4_xs_f32",
        GgmlType::IQ2_S => "kernel_expert_native_gate_up_iq2_s_f32",
        GgmlType::IQ3_XXS => "kernel_expert_native_gate_up_iq3_xxs_f32",
        GgmlType::IQ3_S => "kernel_expert_native_gate_up_iq3_s_f32",
        GgmlType::IQ2_XS => "kernel_expert_native_gate_up_iq2_xs_f32",
        GgmlType::IQ2_XXS if iq2_xxs_split() => "kernel_expert_native_gate_up_iq2_xxs_f32_split",
        GgmlType::IQ2_XXS => "kernel_expert_native_gate_up_iq2_xxs_f32",
        _ => return None,
    })
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
    operands: NativeOperands,
    fuse: bool,
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
    let fused = if fuse {
        fused_gate_up_pipeline(ctx, gate_bank.dtype, up_bank.dtype, operands)?
    } else {
        None
    };
    if let Some(fused) = fused {
        // One dispatch: gate and up channels side by side, then SwiGLU.
        args.mode = operands.mode(1);
        args.row_bytes = gate_bytes;
        enc.set_pipeline(&fused);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, gate_bank);
        enc.set_tensor(2, input);
        enc.set_tensor(3, ids);
        enc.set_tensor(4, status);
        enc.set_tensor(5, output);
        enc.set_tensor(6, up_bank);
        enc.dispatch(
            MTLSize {
                // Channels per threadgroup: two simdgroup pairs of one (or,
                // for the IQ2_XXS split, two) channels each.
                width: n_out.div_ceil(native_channels_per_threadgroup(gate_bank.dtype) / 2),
                height: top_k,
                depth: rows.unwrap_or(1),
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }
    args.mode = operands.mode(0);
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
    args.mode = operands.mode(1);
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
        None,
        NativeOperands::F32,
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
        NativeOperands::F32,
    )
}

/// One half-operand down GEMV for each token/slot, including duplicate IDs.
/// Input `[K, top_k * rows]`, output `[M, top_k * rows]`; routes and row bounds
/// match `encode_all_slots_gate_up_swiglu_native_half_rows`. Caller owns scratch.
/// Both decoded weights and F32 inner activations narrow to half before F32 FMA.
/// Slot weighting/reduction is deliberately left to the existing rows helper.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down_native_half_rows(
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
    const K: &str = "all_slots_down_native_half_rows";
    check_half_rows(K, rows)?;
    if !all_slots_down_native_half_rows_supported(bank.dtype) {
        return Err(bad_shape(K, "unsupported half-operand down storage"));
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
        NativeOperands::Half,
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
    operands: NativeOperands,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_down_native";
    require_serial(K, enc)?;
    let mut args = dimensions(K, n_in, n_out, experts, top_k)?;
    check_views(K, input, ids, status, output, &args, true, rows)?;
    check_device(ctx, enc, K, &[bank, input, ids, status, output])?;
    let (pso, row_bytes) = bank_pipeline(ctx, K, bank, &args, "down bank")?;
    check_disjoint(K, output, &[(bank, "down bank")])?;
    args.mode = operands.mode(2);
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

/// Most activation columns one skinny dispatch projects.
pub const DENSE_SKINNY_MAX_COLS: usize = 8;

/// Output channels per simdgroup of the skinny kernels (`DENSE_SKINNY_T8`
/// tiles of 8).
const DENSE_SKINNY_CHANNELS: usize = 16;

/// Simdgroups the skinny K split aims for (32 per core on an 80-core GPU;
/// extra splitting measured neutral on a 40-core one).
const DENSE_SKINNY_TARGET_SIMDGROUPS: usize = 2560;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseSkinnyArgs {
    row_bytes: u64,
    group_bytes: u64,
    n_in: u32,
    n_out: u32,
    cols: u32,
    in_col: u32,
    out_col: u32,
    in_group: u32,
    out_group: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DensePairSkinnyArgs {
    row_bytes: u64,
    n_in: u32,
    n_a: u32,
    n_b: u32,
    cols: u32,
}

/// K split of the skinny kernels: the smallest divisor of the K block count
/// (at most 8 simdgroups per threadgroup) that brings the grid to
/// `DENSE_SKINNY_TARGET_SIMDGROUPS`, else the largest divisor at most 8.
fn skinny_k_split(blocks: usize, channel_groups: usize) -> usize {
    let mut split = 1;
    for d in (1..=8).filter(|&d| blocks.is_multiple_of(d)) {
        split = d;
        if channel_groups * d >= DENSE_SKINNY_TARGET_SIMDGROUPS {
            break;
        }
    }
    split
}

/// Whether `encode_dense_skinny_q5_k` accepts this storage and geometry
/// (binding layout is checked separately when encoding).
pub fn dense_skinny_supported(dtype: GgmlType, n_in: usize, n_out: usize, cols: usize) -> bool {
    grouped_skinny_supported(dtype, n_in, n_out, 1, cols)
}

/// Whether `encode_grouped_skinny_q5_k` accepts this storage and geometry.
pub fn grouped_skinny_supported(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    groups: usize,
    cols: usize,
) -> bool {
    dtype == GgmlType::Q5_K
        && n_in > 0
        && n_in.is_multiple_of(256)
        && n_out > 0
        && groups > 0
        && (1..=DENSE_SKINNY_MAX_COLS).contains(&cols)
        && [n_in, n_out].iter().all(|&n| {
            n.checked_mul(groups * cols)
                .is_some_and(|n| u32::try_from(n).is_ok())
        })
}

fn skinny_pipeline(
    ctx: &MetalContext,
    kernel: &'static str,
    name: &str,
    split: usize,
) -> Result<Pipeline, MetalError> {
    let pso = ctx.pipeline(name)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 * split {
        return Err(bad_shape(
            kernel,
            format!(
                "skinny kernel needs SIMD width 32 and {} threads",
                32 * split
            ),
        ));
    }
    Ok(pso)
}

/// Dense Q5_K projection of `cols` (1..=8) F32 activation columns,
/// `output[:, c] = weight * input[:, c]`, in one dispatch that reads and
/// dequantizes each weight once for every column
/// (`kernel_dense_skinny_mma_q5_K_f32`, simdgroup MMA with F32 operands).
/// Weight is `[n_in, n_out]` Q5_K (rows of n_in / 256 blocks, 16-byte
/// aligned), input `[n_in, cols]` F32 (16-byte aligned), output `[n_out,
/// cols]` F32, disjoint from both. Values match the one-row native kernels to
/// F32 rounding (summation order differs), not bitwise. Invalid shapes,
/// storage or bindings return an error before anything is encoded.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_skinny_q5_k(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> Result<(), MetalError> {
    const K: &str = "dense_skinny_q5_k";
    let shape = [n_in as u64, n_out as u64];
    encode_skinny_q5_k(
        ctx,
        enc,
        K,
        (weight, &shape),
        input,
        output,
        n_in,
        n_out,
        1,
        cols,
    )
}

/// `groups` independent Q5_K projections `[n_in -> n_out]` of `cols`
/// (1..=8) F32 columns in one dispatch (each weight read once for every
/// column): bank `[n_in, n_out, groups]` (contiguous, 16-byte aligned), input
/// `[n_in * groups, cols]` with column c holding the groups' inputs back to
/// back, output `[n_out * groups, cols]` likewise; group g of column c is
/// `bank[g] * input[g * n_in.., c]`. Same kernel, rounding and validation as
/// `encode_dense_skinny_q5_k`.
#[allow(clippy::too_many_arguments)]
pub fn encode_grouped_skinny_q5_k(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    cols: usize,
) -> Result<(), MetalError> {
    const K: &str = "grouped_skinny_q5_k";
    let shape = [n_in as u64, n_out as u64, groups as u64];
    encode_skinny_q5_k(
        ctx,
        enc,
        K,
        (bank, &shape),
        input,
        output,
        n_in,
        n_out,
        groups,
        cols,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_skinny_q5_k(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &'static str,
    (weight, shape): (&MetalTensor, &[u64]),
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    cols: usize,
) -> Result<(), MetalError> {
    let k = kernel;
    require_serial(k, enc)?;
    if weight.dtype != GgmlType::Q5_K {
        return Err(bad_shape(
            k,
            format!("weight must be Q5_K, got {:?}", weight.dtype),
        ));
    }
    if !grouped_skinny_supported(weight.dtype, n_in, n_out, groups, cols) {
        return Err(bad_shape(
            k,
            format!(
                "n_in {n_in} must be a positive multiple of 256, n_out {n_out} and groups \
                 {groups} positive, cols {cols} in 1..={DENSE_SKINNY_MAX_COLS}"
            ),
        ));
    }
    if weight.shape != shape {
        return Err(bad_shape(
            k,
            format!("weight must have shape {shape:?}, got {:?}", weight.shape),
        ));
    }
    let bank = weight.view_bytes(0, vec![n_in as u64, n_out as u64, groups as u64]);
    check_expert_bank(k, &bank, n_in, n_out, groups, "weight")?;
    let cols64 = cols as u64;
    check_tensor(
        k,
        input,
        GgmlType::F32,
        &[(n_in * groups) as u64, cols64],
        false,
        "input",
    )?;
    super::checks::check_alignment(k, input, 16, "input")?;
    check_tensor(
        k,
        output,
        GgmlType::F32,
        &[(n_out * groups) as u64, cols64],
        true,
        "output",
    )?;
    check_disjoint(k, output, &[(input, "input"), (weight, "weight")])?;
    check_device(ctx, enc, k, &[weight, input, output])?;
    let split = skinny_k_split(n_in / 256, n_out.div_ceil(DENSE_SKINNY_CHANNELS) * groups);
    let pso = skinny_pipeline(ctx, k, "kernel_dense_skinny_mma_q5_K_f32", split)?;
    let (block, bytes) = GgmlType::Q5_K.storage_layout().expect("Q5_K layout");
    let row_bytes = n_in as u64 / block * bytes;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &DenseSkinnyArgs {
            row_bytes,
            group_bytes: row_bytes * n_out as u64,
            n_in: n_in as u32,
            n_out: n_out as u32,
            cols: cols as u32,
            in_col: (n_in * groups) as u32,
            out_col: (n_out * groups) as u32,
            in_group: n_in as u32,
            out_group: n_out as u32,
            _pad: 0,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(DENSE_SKINNY_CHANNELS),
            height: groups,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: split,
        },
    );
    Ok(())
}

/// Whether `encode_dense_pair_skinny_q8_0` accepts this storage and
/// geometry.
pub fn dense_pair_skinny_supported(
    a: GgmlType,
    b: GgmlType,
    n_in: usize,
    n_a: usize,
    n_b: usize,
    cols: usize,
) -> bool {
    a == GgmlType::Q8_0
        && b == GgmlType::Q8_0
        && n_in > 0
        && n_in.is_multiple_of(32)
        && n_a > 0
        && n_a.is_multiple_of(DENSE_SKINNY_CHANNELS)
        && n_b > 0
        && (1..=DENSE_SKINNY_MAX_COLS).contains(&cols)
        && [n_in, n_a + n_b].iter().all(|&n| {
            n.checked_mul(cols)
                .is_some_and(|n| u32::try_from(n).is_ok())
        })
}

/// Two dense Q8_0 projections of the same `cols` (1..=8) F32 columns in one
/// dispatch, `y_a[:, c] = a * input[:, c]` and `y_b[:, c] = b * input[:, c]`,
/// each weight read and dequantized once for every column
/// (`kernel_dense_pair_skinny_mma_q8_0_f32`; the skinny counterpart of the
/// decode rank pair). Weights `[n_in, n_a]` and `[n_in, n_b]` Q8_0 (2-byte
/// aligned), input `[n_in, cols]`, outputs `[n_a, cols]` and `[n_b, cols]`
/// F32, disjoint from every input and each other; n_a % 16 == 0. F32
/// rounding vs the GEMV kernels, not bitwise. Errors before encoding.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_pair_skinny_q8_0(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    b: &MetalTensor,
    input: &MetalTensor,
    y_a: &MetalTensor,
    y_b: &MetalTensor,
    (n_in, n_a, n_b): (usize, usize, usize),
    cols: usize,
) -> Result<(), MetalError> {
    const K: &str = "dense_pair_skinny_q8_0";
    require_serial(K, enc)?;
    if !dense_pair_skinny_supported(a.dtype, b.dtype, n_in, n_a, n_b, cols) {
        return Err(bad_shape(
            K,
            format!(
                "Q8_0 weights ({:?}, {:?}), n_in {n_in} a positive multiple of 32, \
                 n_a {n_a} a positive multiple of 16, n_b {n_b} positive, cols {cols} in \
                 1..={DENSE_SKINNY_MAX_COLS} required",
                a.dtype, b.dtype
            ),
        ));
    }
    let row_bytes = (n_in / 32 * 34) as u64;
    for (w, n, name) in [(a, n_a, "weight a"), (b, n_b, "weight b")] {
        if w.shape != [n_in as u64, n as u64]
            || w.n_bytes() != row_bytes * n as u64
            || w.offset
                .checked_add(w.n_bytes())
                .is_none_or(|end| end > w.buffer.length() as u64)
        {
            return Err(bad_shape(
                K,
                format!("{name} must be [{n_in}, {n}] Q8_0 inside its buffer"),
            ));
        }
        super::checks::check_alignment(K, w, 2, name)?;
    }
    let cols64 = cols as u64;
    check_tensor(
        K,
        input,
        GgmlType::F32,
        &[n_in as u64, cols64],
        false,
        "input",
    )?;
    check_tensor(
        K,
        y_a,
        GgmlType::F32,
        &[n_a as u64, cols64],
        true,
        "output a",
    )?;
    check_tensor(
        K,
        y_b,
        GgmlType::F32,
        &[n_b as u64, cols64],
        true,
        "output b",
    )?;
    for y in [y_a, y_b] {
        check_disjoint(K, y, &[(input, "input"), (a, "weight a"), (b, "weight b")])?;
    }
    check_disjoint(K, y_a, &[(y_b, "output b")])?;
    check_device(ctx, enc, K, &[a, b, input, y_a, y_b])?;
    let split = skinny_k_split(n_in / 32, (n_a + n_b).div_ceil(DENSE_SKINNY_CHANNELS));
    let pso = skinny_pipeline(ctx, K, "kernel_dense_pair_skinny_mma_q8_0_f32", split)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &DensePairSkinnyArgs {
            row_bytes,
            n_in: n_in as u32,
            n_a: n_a as u32,
            n_b: n_b as u32,
            cols: cols as u32,
        },
    );
    enc.set_tensor(1, a);
    enc.set_tensor(2, b);
    enc.set_tensor(3, input);
    enc.set_tensor(4, y_a);
    enc.set_tensor(5, y_b);
    enc.dispatch(
        MTLSize {
            width: (n_a + n_b).div_ceil(DENSE_SKINNY_CHANNELS),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: split,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests;
