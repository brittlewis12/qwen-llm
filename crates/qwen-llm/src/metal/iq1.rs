//! Dense rank-2 IQ1_S/IQ1_M projections; weights remain compressed in every path.
use super::checks::{bad_shape, check_disjoint};
use super::moe::{checked_moe_product, validate_moe_decode_tensor};
use super::*;

const KERNEL: &str = "iq1_dense";
const GEMV: &str = "kernel_mat_vec_iq1_f32";
const SCALAR: &str = "kernel_mat_mat_iq1_f32_scalar";
const MMA: &str = "kernel_mat_mat_iq1_f32_mma";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    k: u32,
    m: u32,
    n: u32,
    row_bytes: u32,
    is_m: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Gemv,
    Scalar,
    Mma,
    Gather,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Gemv => GEMV,
            Self::Scalar => SCALAR,
            Self::Mma => MMA,
            Self::Gather => "kernel_get_rows_iq1_m_f32",
        }
    }
    fn threads(self) -> usize {
        match self {
            Self::Scalar => 32,
            Self::Gather => 256,
            _ => 64,
        }
    }
    fn scratch(self) -> usize {
        match self {
            Self::Gemv | Self::Gather => 0,
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

fn format_layout(dtype: GgmlType) -> Result<(usize, u32), MetalError> {
    match dtype {
        GgmlType::IQ1_S => Ok((50, 0)),
        GgmlType::IQ1_M => Ok((56, 1)),
        _ => Err(bad_shape(KERNEL, "weight must be IQ1_S or IQ1_M")),
    }
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
    let (block_bytes, is_m) = format_layout(weight.dtype)?;
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
            weight.dtype,
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
    let row_bytes = product("row bytes", k / 256, block_bytes)?;
    Ok(Args {
        k: k as u32,
        m: m as u32,
        n: n as u32,
        is_m,
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
        "__iq1_missing_pipeline"
    } else {
        name
    };
    let pso = ctx.pipeline(name)?;
    let (width, threads, static_bytes, limit) = (
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        ctx.device.maxThreadgroupMemoryLength(),
    );
    #[cfg(test)]
    let (width, threads, static_bytes, limit) = match fault {
        Some(PipelineFault::Width) => (16, threads, static_bytes, limit),
        Some(PipelineFault::Threads) => (width, path.threads() - 1, static_bytes, limit),
        Some(PipelineFault::Memory) => (width, threads, static_bytes.max(1), 0),
        _ => (width, threads, static_bytes, limit),
    };
    check_capacity(path, width, threads, static_bytes, limit)?;
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
        Path::Gather => ((args.k as usize).div_ceil(256), args.n as usize),
        Path::Mma => (
            (args.m as usize).div_ceil(16),
            (args.n as usize).div_ceil(8),
        ),
    };
    #[cfg(test)]
    let format_tag = if path != Path::Gather && dispatch_census_is_active() {
        let format = if args.is_m == 0 { "iq1_s" } else { "iq1_m" };
        // Diagnostics annotation only; the actual pipeline keeps its shared entry name.
        census_record_pso(&format!("{} [{format}]", path.name()));
        dispatch_census_tag_scope(|| format.to_owned())
    } else {
        None
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
    #[cfg(test)]
    if format_tag.is_some() {
        census_record_pso(path.name());
    }
}

/// Compressed IQ1_S/IQ1_M `[K,M]` times F32 `[K]`, producing F32 `[M]`.
/// K must be a positive multiple of 256; dimensions fit positive i32. Weight
/// offsets are aligned to 2 bytes, activations/output to 4. No dequant buffer.
/// Encodes one dispatch on serial or concurrent encoders. Concurrent callers
/// must ensure disjoint outputs and no dependencies between dispatches; shared
/// read-only inputs are allowed. This primitive records its own hazard notes;
/// wrappers must not also record the same dispatch's write.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_vec_iq1_f32(
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

/// Compressed IQ1_S/IQ1_M `[K,M]` times token-major F32 `[K,N]`, writing `[M,N]`.
/// Same bounds as GEMV, with arbitrary positive N. N=1 uses GEMV. Otherwise
/// F32 MMA requires Apple7 family support, 16-byte-aligned input, and a
/// SIMD32 pipeline supporting TG64 + 1 KiB TGM. Scalar F32 is the fallback.
/// Selection and validation finish before any dispatch; no full dequantization.
/// Encodes one dispatch on serial or concurrent encoders. Concurrent callers
/// must ensure disjoint outputs and no dependencies between dispatches; shared
/// read-only inputs are allowed. This primitive records its own hazard notes;
/// wrappers must not also record the same dispatch's write.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_iq1_f32(
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
pub enum Iq1MatMatVariant {
    Auto,
    Scalar,
    Mma,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_iq1_f32_with_variant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    variant: Iq1MatMatVariant,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    let selection = match variant {
        Iq1MatMatVariant::Auto => Selection::Auto,
        Iq1MatMatVariant::Scalar => Selection::Scalar,
        Iq1MatMatVariant::Mma => Selection::Mma,
    };
    encode_mat_mat(ctx, enc, weight, x, y, n_in, n_out, n_query, selection)
}

/// IQ1_M row lookup into F32 output, with aligned I32 IDs. As with
/// `encode_get_rows_f32`, shapes are interpreted by element count: the source
/// contains complete `n_embd`-element rows, IDs contain `n_tokens` elements,
/// and output contains `n_tokens * n_embd` elements. `n_embd` is a positive
/// multiple of 256; dimensions and vocabulary fit u32. Source alignment is
/// 2 bytes; IDs/output alignment is 4 bytes. Invalid IDs write a zero row.
/// One dispatch; concurrent callers must ensure independent outputs and no
/// cross-dispatch dependencies. This primitive records its own hazard notes.
pub fn encode_get_rows_iq1_m_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    embed: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_tokens: usize,
    n_embd: usize,
) -> Result<(), MetalError> {
    if n_tokens == 0 || n_embd == 0 || !n_embd.is_multiple_of(256) {
        return Err(bad_shape(
            KERNEL,
            "gather needs positive tokens and embedding width divisible by 256",
        ));
    }
    if embed.dtype != GgmlType::IQ1_M {
        return Err(bad_shape(KERNEL, "gather source must be IQ1_M"));
    }
    let (elements, _) = checked_ggml_shape_bytes(&embed.shape, embed.dtype)?;
    if elements == 0 || !elements.is_multiple_of(n_embd) {
        return Err(bad_shape(
            KERNEL,
            "gather source must contain complete embedding rows",
        ));
    }
    let vocab = elements / n_embd;
    let output_elements = checked_moe_product(KERNEL, "gather output", &[n_tokens, n_embd])?;
    for (name, tensor, count, dtype, writable, alignment) in [
        ("embedding", embed, elements, GgmlType::IQ1_M, false, 2),
        ("ids", ids, n_tokens, GgmlType::I32, false, 4),
        ("output", out, output_elements, GgmlType::F32, true, 4),
    ] {
        validate_moe_decode_tensor(KERNEL, name, tensor, count, &[dtype], writable, alignment)?;
    }
    check_disjoint(KERNEL, out, &[(embed, "embedding"), (ids, "ids")])?;
    check_devices(
        ctx.device.registryID(),
        enc.parent_command_buffer().device().registryID(),
        [
            embed.buffer.device().registryID(),
            ids.buffer.device().registryID(),
            out.buffer.device().registryID(),
        ],
    )?;
    let args = Args {
        k: super::checks::to_u32(KERNEL, n_embd, "embedding width")?,
        m: super::checks::to_u32(KERNEL, vocab, "vocabulary")?,
        n: super::checks::to_u32(KERNEL, n_tokens, "tokens")?,
        row_bytes: super::checks::to_u32(
            KERNEL,
            checked_moe_product(KERNEL, "row bytes", &[n_embd / 256, 56])?,
            "row bytes",
        )?,
        is_m: 1,
    };
    let pso = pipeline(ctx, Path::Gather)?;
    encode(enc, [embed, ids, out], args, Path::Gather, &pso);
    Ok(())
}

#[cfg(test)]
mod tests;
