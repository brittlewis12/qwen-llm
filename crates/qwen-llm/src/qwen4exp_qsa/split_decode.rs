//! Decode attention with the value accumulation split across key ranges.
//!
//! The incumbent kernel runs one threadgroup per query head looping over
//! every selected key, so at the 2,048-key budget it occupies 24
//! threadgroups and reads each value row once per query head. The split
//! path keeps the incumbent logits and global softmax, then accumulates
//! values over `splits` key ranges per KV head (up to 128 threadgroups,
//! each value row read once per KV head) and merges the partials with the
//! sigmoid gate. Only the F32 summation order of the value sum changes.
use super::*;

/// Per head: `splits` partial records of `[denominator, pad, 256 dims]`.
pub(crate) const SCRATCH_FLOATS: usize = 24 * MAX_SPLITS * 258;
const MAX_SPLITS: usize = 64;
const KEYS_PER_SPLIT: usize = 32;
/// Below this many selected keys the incumbent single-pass kernel is used:
/// the split path costs two extra dispatches, and short key lists finish
/// inside one threadgroup's loop before that overhead pays back.
pub(crate) const MIN_IDS: usize = 256;

pub(crate) fn eligible(g: QwenSparseAttentionMetalGeometry, ids: usize) -> bool {
    g.supports_split_decode() && ids >= MIN_IDS
}

pub(crate) fn splits_for(ids: usize) -> usize {
    ids.div_ceil(KEYS_PER_SPLIT).clamp(1, MAX_SPLITS)
}

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpQsaError> {
    for (name, threads) in [
        ("kernel_qwen4exp_qsa_attention_logits_f16", 256),
        ("kernel_qwen4exp_qsa_split_softmax_f32", 256),
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
            if name == "kernel_qwen4exp_qsa_split_softmax_f32" {
                ATTENTION_SCRATCH_BYTES
            } else {
                0
            },
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
    row_stride: u32,
}

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &MetalTensor,
    ids: usize,
) -> Result<(), MetalError> {
    encode_to(ctx, enc, workspace, scratch, &workspace.attention, ids)
}

pub(super) fn encode_to(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &MetalTensor,
    output: &MetalTensor,
    ids: usize,
) -> Result<(), MetalError> {
    let splits = splits_for(ids);
    let args = Args {
        ids: ids as u32,
        capacity: workspace.geometry.capacity as u32,
        splits: splits as u32,
        keys_per_split: ids.div_ceil(splits).max(1) as u32,
        row_stride: workspace.geometry.output_width() as u32,
    };
    if ids > 0 {
        encode_attention_logits(ctx, enc, workspace, ids)?;
    }
    enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_qsa_split_softmax_f32")?);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, &workspace.attention_logits);
    enc.set_tensor(2, scratch);
    enc.set_threadgroup_memory(0, ATTENTION_SCRATCH_BYTES);
    enc.dispatch(
        MTLSize {
            width: 24,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_split_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, &workspace.attention_logits);
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
    enc.set_tensor(3, output);
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
