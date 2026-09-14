use super::*;

pub(crate) const SCRATCH_FLOATS: usize = 24 * 64 * 258;

pub(crate) fn eligible(g: QwenSparseAttentionMetalGeometry, ids: usize) -> bool {
    g.supports_split_decode() && (2048..=2051).contains(&ids)
}

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpQsaError> {
    for (name, threads) in [
        ("kernel_qwen4exp_qsa_split_f16", 128),
        ("kernel_qwen4exp_qsa_split_merge_f32", 32),
    ] {
        let pso = ctx.pipeline(name)?;
        validate_cooperative_pipeline_threads(
            name,
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            threads,
            pso.staticThreadgroupMemoryLength(),
            0,
            ctx.device.maxThreadgroupMemoryLength(),
        )?;
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    ids: u32,
    capacity: u32,
    splits: u32,
    keys_per_split: u32,
}

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &MetalTensor,
    ids: usize,
) -> Result<(), MetalError> {
    let splits = ids.div_ceil(32).min(64);
    let args = Args {
        ids: ids as u32,
        capacity: workspace.geometry.capacity as u32,
        splits: splits as u32,
        keys_per_split: ids.div_ceil(splits) as u32,
    };
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_split_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, &workspace.query);
    enc.set_tensor(2, &workspace.key_cache);
    enc.set_tensor(3, &workspace.value_cache);
    enc.set_tensor(4, &workspace.token_ids);
    enc.set_tensor(5, scratch);
    enc.dispatch(
        MTLSize {
            width: splits,
            height: 2,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_split_merge_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, scratch);
    enc.set_tensor(2, &workspace.raw_gate);
    enc.set_tensor(3, &workspace.attention);
    enc.dispatch(
        MTLSize {
            width: 24,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
