//! Dense IQ2_XS F32 MMA for N>1, with checked scalar fallback.
use super::checks::bad_shape;
use super::*;

const KERNEL: &str = "iq2_xs_dense";
const SCALAR: &str = "kernel_mat_mat_iq2_xs_f32";
const MMA: &str = "kernel_mat_mat_iq2_xs_f32_mma";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Scalar,
    Mma,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => SCALAR,
            Self::Mma => MMA,
        }
    }
    fn threads(self) -> usize {
        match self {
            Self::Scalar => 32,
            Self::Mma => 64,
        }
    }
    fn scratch(self) -> usize {
        match self {
            Self::Scalar => 128,
            Self::Mma => 1024,
        }
    }
}

/// Test-only selection for N>=2. N=1 retains the existing scalar GEMM;
/// the generic dispatch's optional GEMV shortcut and its switches are untouched.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iq2XsMatMatVariant {
    Scalar,
    Auto,
    Mma,
}

#[cfg(test)]
thread_local! {
    static STATE: Cell<(Option<Iq2XsMatMatVariant>, usize)> = const { Cell::new((None, 0)) };
    static PIPELINE_FAULT: Cell<Option<(Path, PipelineFault)>> = const { Cell::new(None) };
    static MMA_SUPPORT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Scope dense IQ2_XS matrix selection to this thread; None selects production
/// guarded Auto for N>1. This scope does not change N=1/GEMV, dedicated grouped
/// kernels, or callers of the explicit scalar entry.
/// Auto tries guarded MMA, Scalar retains the incumbent, Mma requires support.
/// Returns (closure result, successful MMA dispatch count). Nested scopes isolate
/// their counts and restore the parent selection/count, including during unwind.
/// This counts encoded substitutions, not GPU completion. No environment override.
#[cfg(test)]
pub fn with_iq2_xs_matmat_variant<R>(
    variant: Option<Iq2XsMatMatVariant>,
    f: impl FnOnce() -> R,
) -> (R, usize) {
    struct Restore((Option<Iq2XsMatMatVariant>, usize));
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(STATE.with(|s| s.replace((variant, 0))));
    let result = f();
    (result, STATE.with(|s| s.get().1))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarArgs {
    k: u32,
    m: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MmaArgs {
    k: u32,
    m: u32,
    n: u32,
    row_bytes: u32,
}

fn check_devices(context: u64, encoder: u64, bindings: [u64; 3]) -> Result<(), MetalError> {
    if encoder != context || bindings.into_iter().any(|d| d != context) {
        return Err(bad_shape(
            KERNEL,
            "encoder and bindings must belong to the context device",
        ));
    }
    Ok(())
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

fn check_mma_support(supported: bool) -> Result<(), MetalError> {
    if supported {
        Ok(())
    } else {
        Err(bad_shape(
            KERNEL,
            "register MMA requires Apple7 family support",
        ))
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum PipelineFault {
    Missing,
    Width,
    Threads,
    Memory,
}

fn pipeline(ctx: &MetalContext, path: Path) -> Result<Pipeline, MetalError> {
    if path == Path::Mma {
        let supported = ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7);
        #[cfg(test)]
        let supported = MMA_SUPPORT.with(|s| s.get()).unwrap_or(supported);
        check_mma_support(supported)?;
    }
    let name = path.name();
    #[cfg(test)]
    let fault = PIPELINE_FAULT
        .with(|s| s.get())
        .filter(|(target, _)| *target == path)
        .map(|(_, f)| f);
    #[cfg(test)]
    let name = if matches!(fault, Some(PipelineFault::Missing)) {
        "__iq2_xs_missing_pipeline"
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

fn auto_enabled() -> bool {
    #[cfg(test)]
    {
        matches!(
            STATE.with(|s| s.get().0),
            None | Some(Iq2XsMatMatVariant::Auto)
        )
    }
    #[cfg(not(test))]
    {
        true
    }
}

fn select_pipeline(
    ctx: &MetalContext,
    input: &MetalTensor,
    n: usize,
) -> Result<(Path, Pipeline), MetalError> {
    if n > 1 {
        #[cfg(test)]
        if STATE.with(|s| s.get().0) == Some(Iq2XsMatMatVariant::Mma) {
            if !input.offset.is_multiple_of(16) {
                return Err(bad_shape(KERNEL, "MMA requires 16-byte input alignment"));
            }
            return Ok((Path::Mma, pipeline(ctx, Path::Mma)?));
        }
        if auto_enabled() && input.offset.is_multiple_of(16) {
            if let Ok(pso) = pipeline(ctx, Path::Mma) {
                return Ok((Path::Mma, pso));
            }
        }
    }
    Ok((Path::Scalar, pipeline(ctx, Path::Scalar)?))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_mat_mat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_selected(ctx, enc, weight, x, y, k, m, n, false)
}

/// Checked incumbent scalar GEMM for family-owned arithmetic contracts at every N.
/// Ignores dense MMA test overrides; records its own hazard notes exactly once.
/// Concurrent callers must ensure independent outputs and no cross-dispatch dependencies.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mat_mat_iq2_xs_f32_scalar(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_selected(ctx, enc, weight, x, y, k, m, n, true)
}

#[allow(clippy::too_many_arguments)]
fn encode_mat_mat_selected(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
    scalar_only: bool,
) -> Result<(), MetalError> {
    super::mat_mat::validate_iq2_xs_dense_bindings(weight, x, y, k, m, n)?;
    check_devices(
        ctx.device.registryID(),
        enc.parent_command_buffer().device().registryID(),
        [
            weight.buffer.device().registryID(),
            x.buffer.device().registryID(),
            y.buffer.device().registryID(),
        ],
    )?;
    let row_bytes = super::checks::to_u32(
        KERNEL,
        super::moe::checked_moe_product(KERNEL, "row bytes", &[k / 256, 74])?,
        "row bytes",
    )?;
    let (path, pso) = if scalar_only {
        (Path::Scalar, pipeline(ctx, Path::Scalar)?)
    } else {
        select_pipeline(ctx, x, n)?
    };
    // The GEMM seam owns these notes; the unchanged GEMV route notes in its wrapper.
    enc.note_read(weight);
    enc.note_read(x);
    enc.note_write(y);
    enc.set_pipeline(&pso);
    match path {
        Path::Scalar => enc.set_bytes(
            0,
            &ScalarArgs {
                k: k as u32,
                m: m as u32,
                n: n as u32,
            },
        ),
        Path::Mma => enc.set_bytes(
            0,
            &MmaArgs {
                k: k as u32,
                m: m as u32,
                n: n as u32,
                row_bytes,
            },
        ),
    }
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, path.scratch());
    let (width, height) = match path {
        Path::Scalar => (m, n.div_ceil(32)),
        Path::Mma => (m.div_ceil(16), n.div_ceil(8)),
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
    if path == Path::Mma {
        STATE.with(|s| {
            let (variant, count) = s.get();
            s.set((variant, count + 1));
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
