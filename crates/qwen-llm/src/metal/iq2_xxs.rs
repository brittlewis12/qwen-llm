//! Dense rank-2 IQ2_XXS projections; weights remain compressed in every path.
use super::checks::{bad_shape, check_disjoint};
use super::moe::{checked_moe_product, validate_moe_decode_tensor};
use super::*;

const KERNEL: &str = "iq2_xxs_dense";
const GEMV: &str = "kernel_mat_vec_iq2_xxs_f32";
const SCALAR: &str = "kernel_mat_mat_iq2_xxs_f32_scalar";
const MMA: &str = "kernel_mat_mat_iq2_xxs_f32_mma";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    k: u32,
    m: u32,
    n: u32,
    row_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Gemv,
    Scalar,
    Mma,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Gemv => GEMV,
            Self::Scalar => SCALAR,
            Self::Mma => MMA,
        }
    }
    fn threads(self) -> usize {
        if self == Self::Scalar { 32 } else { 64 }
    }
    fn scratch(self) -> usize {
        match self {
            Self::Gemv => 0,
            Self::Scalar => 128,
            Self::Mma => 1024,
        }
    }
}

#[derive(Clone, Copy)]
enum Selection {
    Auto,
    #[cfg(test)]
    Scalar,
    #[cfg(test)]
    Mma,
}

fn check_devices(context: u64, encoder: u64, bindings: [u64; 3]) -> Result<(), MetalError> {
    if encoder != context || bindings.into_iter().any(|id| id != context) {
        return Err(bad_shape(
            KERNEL,
            "encoder and bindings must belong to the context device",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn checked_args(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
    matrix: bool,
) -> Result<Args, MetalError> {
    for (name, value) in [("K", k), ("M", m), ("N", n)] {
        if value == 0 || value > i32::MAX as usize {
            return Err(bad_shape(KERNEL, format!("{name} must fit positive i32")));
        }
    }
    if !k.is_multiple_of(256) {
        return Err(bad_shape(KERNEL, "K must be divisible by 256"));
    }
    if weight.shape != [k as u64, m as u64] {
        return Err(bad_shape(KERNEL, "weight must be a rank-2 [K, M] matrix"));
    }
    let x_shape = if matrix {
        vec![k as u64, n as u64]
    } else {
        vec![k as u64]
    };
    let y_shape = if matrix {
        vec![m as u64, n as u64]
    } else {
        vec![m as u64]
    };
    if x.shape != x_shape || y.shape != y_shape {
        return Err(bad_shape(
            KERNEL,
            "input/output must be [K]/[M] for GEMV or [K,N]/[M,N] for mat-mat",
        ));
    }
    let product = |label, a, b| checked_moe_product(KERNEL, label, &[a, b]);
    for (name, tensor, count, dtype, writable, alignment) in [
        (
            "weight",
            weight,
            product("weight", k, m)?,
            GgmlType::IQ2_XXS,
            false,
            2,
        ),
        ("input", x, product("input", k, n)?, GgmlType::F32, false, 4),
        (
            "output",
            y,
            product("output", m, n)?,
            GgmlType::F32,
            true,
            4,
        ),
    ] {
        validate_moe_decode_tensor(KERNEL, name, tensor, count, &[dtype], writable, alignment)?;
    }
    check_disjoint(KERNEL, y, &[(weight, "weight"), (x, "input")])?;
    check_devices(
        ctx.device.registryID(),
        enc.parent_command_buffer().device().registryID(),
        [
            weight.buffer.device().registryID(),
            x.buffer.device().registryID(),
            y.buffer.device().registryID(),
        ],
    )?;
    let row_bytes = product("row bytes", k / 256, 66)?;
    Ok(Args {
        k: k as u32,
        m: m as u32,
        n: n as u32,
        row_bytes: super::checks::to_u32(KERNEL, row_bytes, "row bytes")?,
    })
}

fn check_capacity(
    path: Path,
    width: usize,
    threads: usize,
    static_bytes: usize,
    limit: usize,
) -> Result<(), MetalError> {
    if width != 32
        || threads < path.threads()
        || static_bytes
            .checked_add(path.scratch())
            .is_none_or(|bytes| bytes > limit)
    {
        return Err(bad_shape(
            KERNEL,
            format!(
                "{} requires SIMD32, TG{} and {} dynamic TGM bytes",
                path.name(),
                path.threads(),
                path.scratch()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum PipelineFault {
    Missing,
    Width,
    Threads,
    Memory,
}

#[cfg(test)]
thread_local! {
    static PIPELINE_FAULT: Cell<Option<(Path, PipelineFault)>> = const { Cell::new(None) };
    static MMA_SUPPORT: Cell<Option<bool>> = const { Cell::new(None) };
}

fn check_mma_support(supported: bool) -> Result<(), MetalError> {
    if !supported {
        return Err(bad_shape(
            KERNEL,
            "register MMA requires Apple7 family support",
        ));
    }
    Ok(())
}

fn pipeline(ctx: &MetalContext, path: Path) -> Result<Pipeline, MetalError> {
    if path == Path::Mma {
        // Match upstream's SIMD-group matrix support contract, before pipeline lookup.
        let supported = ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7);
        #[cfg(test)]
        let supported = MMA_SUPPORT.with(|s| s.get()).unwrap_or(supported);
        check_mma_support(supported)?;
    }
    let name = path.name();
    #[cfg(test)]
    let fault = PIPELINE_FAULT
        .with(|f| f.get())
        .filter(|(target, _)| *target == path)
        .map(|(_, f)| f);
    #[cfg(test)]
    let name = if matches!(fault, Some(PipelineFault::Missing)) {
        "__iq2_xxs_missing_pipeline"
    } else {
        name
    };
    let pso = ctx.pipeline(name)?;
    let (width, threads, limit) = (
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    );
    #[cfg(test)]
    let (width, threads, limit) = match fault {
        Some(PipelineFault::Width) => (16, threads, limit),
        Some(PipelineFault::Threads) => (width, path.threads() - 1, limit),
        Some(PipelineFault::Memory) => (width, threads, 0),
        _ => (width, threads, limit),
    };
    check_capacity(
        path,
        width,
        threads,
        pso.staticThreadgroupMemoryLength(),
        limit,
    )?;
    Ok(pso)
}

fn select_pipeline(
    ctx: &MetalContext,
    input: &MetalTensor,
    n: usize,
    _selection: Selection,
) -> Result<(Path, Pipeline), MetalError> {
    if n == 1 {
        return Ok((Path::Gemv, pipeline(ctx, Path::Gemv)?));
    }
    #[cfg(test)]
    match _selection {
        Selection::Scalar => return Ok((Path::Scalar, pipeline(ctx, Path::Scalar)?)),
        Selection::Mma => {
            if !input.offset.is_multiple_of(16) {
                return Err(bad_shape(KERNEL, "MMA requires 16-byte input alignment"));
            }
            return Ok((Path::Mma, pipeline(ctx, Path::Mma)?));
        }
        Selection::Auto => {}
    }
    if input.offset.is_multiple_of(16) {
        if let Ok(pso) = pipeline(ctx, Path::Mma) {
            return Ok((Path::Mma, pso));
        }
    }
    Ok((Path::Scalar, pipeline(ctx, Path::Scalar)?))
}

fn encode(enc: &KernelEncoder, tensors: [&MetalTensor; 3], args: Args, path: Path, pso: &Pipeline) {
    enc.note_read(tensors[0]);
    enc.note_read(tensors[1]);
    enc.note_write(tensors[2]);
    enc.set_pipeline(pso);
    enc.set_bytes(0, &args);
    for (i, tensor) in tensors.into_iter().enumerate() {
        enc.set_tensor(i + 1, tensor);
    }
    if path.scratch() > 0 {
        enc.set_threadgroup_memory(0, path.scratch());
    }
    let (width, height) = match path {
        Path::Gemv => ((args.m as usize).div_ceil(8), 1),
        Path::Scalar => (args.m as usize, (args.n as usize).div_ceil(32)),
        Path::Mma => (
            (args.m as usize).div_ceil(16),
            (args.n as usize).div_ceil(8),
        ),
    };
    enc.dispatch(
        MTLSize {
            width,
            height,
            depth: 1,
        },
        MTLSize {
            width: path.threads(),
            height: 1,
            depth: 1,
        },
    );
}

/// Compressed IQ2_XXS `[K,M]` times F32 `[K]`, producing F32 `[M]`.
/// K must be a positive multiple of 256; dimensions fit positive i32. Weight
/// offsets are aligned to 2 bytes, activations/output to 4. No dequant buffer.
/// Encodes one dispatch on serial or concurrent encoders. Concurrent callers
/// must ensure disjoint outputs and no dependencies between dispatches; shared
/// read-only inputs are allowed. This primitive records its own hazard notes;
/// wrappers must not also record the same dispatch's write.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_vec_iq2_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    let args = checked_args(ctx, enc, weight, x, y, n_in, n_out, 1, false)?;
    let pso = pipeline(ctx, Path::Gemv)?;
    encode(enc, [weight, x, y], args, Path::Gemv, &pso);
    Ok(())
}

/// Compressed IQ2_XXS `[K,M]` times token-major F32 `[K,N]`, writing `[M,N]`.
/// Same bounds as GEMV, with arbitrary positive N. N=1 uses GEMV. Otherwise
/// F32 MMA requires Apple7 family support, 16-byte-aligned input, and a
/// SIMD32 pipeline supporting TG64 + 1 KiB TGM. Scalar F32 is the fallback.
/// Selection and validation finish before any dispatch; no full dequantization.
/// Encodes one dispatch on serial or concurrent encoders. Concurrent callers
/// must ensure disjoint outputs and no dependencies between dispatches; shared
/// read-only inputs are allowed. This primitive records its own hazard notes;
/// wrappers must not also record the same dispatch's write.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_iq2_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        Selection::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_mat_mat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
    selection: Selection,
) -> Result<(), MetalError> {
    let args = checked_args(ctx, enc, weight, x, y, k, m, n, true)?;
    let (path, pso) = select_pipeline(ctx, x, n, selection)?;
    encode(enc, [weight, x, y], args, path, &pso);
    Ok(())
}

/// Diagnostic path control; N=1 always follows GEMV. Explicit MMA is strict,
/// while Auto may fall back before encoding. No runtime environment override.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub enum Iq2XxsMatMatVariant {
    Auto,
    Scalar,
    Mma,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_iq2_xxs_f32_with_variant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    variant: Iq2XxsMatMatVariant,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    let selection = match variant {
        Iq2XxsMatMatVariant::Auto => Selection::Auto,
        Iq2XxsMatMatVariant::Scalar => Selection::Scalar,
        Iq2XxsMatMatVariant::Mma => Selection::Mma,
    };
    encode_mat_mat(ctx, enc, weight, x, y, n_in, n_out, n_query, selection)
}

#[cfg(test)]
mod tests;
