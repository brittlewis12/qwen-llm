use super::*;

const TREE_KERNEL: &str = "kernel_qwen4exp_topk_guarded_f32";
const SIMD_KERNEL: &str = "kernel_qwen4exp_topk_guarded_simd_f32";

/// `0` keeps the tree-reduction guarded top-k; the default takes each round's
/// arg-max with SIMD reductions (same selection, ties, nonfinite fallback and
/// weights; 3 instead of ~11 barriers per selected expert).
const TOPK_SIMD_ENV: &str = "QWEN_FN_TOPK_SIMD";

/// The guarded top-k kernel this process dispatches.
pub(crate) fn kernel() -> &'static str {
    static SIMD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SIMD.get_or_init(|| std::env::var(TOPK_SIMD_ENV).as_deref() != Ok("0")) {
        SIMD_KERNEL
    } else {
        TREE_KERNEL
    }
}

pub(crate) fn supported(ctx: &MetalContext) -> Result<bool, Qwen4ExpMoeError> {
    let p = ctx.pipeline(kernel())?;
    Ok(p.threadExecutionWidth() == 32
        && p.maxTotalThreadsPerThreadgroup() >= 512
        && p.staticThreadgroupMemoryLength()
            .checked_add(6144)
            .is_some_and(|bytes| bytes <= ctx.device.maxThreadgroupMemoryLength()))
}

pub(super) fn eligible(n: usize, k: usize) -> bool {
    (n, k) == (512, 10)
}

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpMoeError> {
    require_pipeline_capacity(ctx, kernel(), 512, 6144)?;
    if ctx.pipeline(kernel())?.threadExecutionWidth() != 32 {
        return invalid("guarded top-k requires SIMD width32");
    }
    Ok(())
}

pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
) -> Result<(), Qwen4ExpMoeError> {
    validate_encoder(ctx, enc)?;
    require_tensor("guarded top-k logits", logits, GgmlType::F32, &[512], false)?;
    require_tensor("guarded top-k IDs", ids, GgmlType::I32, &[10], true)?;
    require_tensor("guarded top-k weights", weights, GgmlType::F32, &[10], true)?;
    let tensors = [("logits", logits), ("IDs", ids), ("weights", weights)];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    preflight(ctx)?;
    enc.set_pipeline(&ctx.pipeline(kernel())?);
    enc.set_bytes_slice(0, &[512u32, 10]);
    enc.set_tensor(1, logits);
    enc.set_tensor(2, ids);
    enc.set_tensor(3, weights);
    for slot in 0..3 {
        enc.set_threadgroup_memory(slot, 2048);
    }
    enc.dispatch(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    fn run(ctx: &MetalContext, kernel: &str, logits: &[f32]) -> (Vec<i32>, Vec<u32>) {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = MetalTensor::from_bytes(ctx, &bytes, vec![512], GgmlType::F32).unwrap();
        let ids = MetalTensor::zeros_i32(ctx, vec![10]).unwrap();
        let weights = MetalTensor::zeros_f32(ctx, vec![10]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        enc.set_pipeline(&ctx.pipeline(kernel).unwrap());
        enc.set_bytes_slice(0, &[512u32, 10]);
        enc.set_tensor(1, &input);
        enc.set_tensor(2, &ids);
        enc.set_tensor(3, &weights);
        for slot in 0..3 {
            enc.set_threadgroup_memory(slot, 2048);
        }
        enc.dispatch(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        let read = |t: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<u32>(),
                10,
            )
            .to_vec()
        };
        let idx = read(&ids).into_iter().map(|v| v as i32).collect();
        (idx, read(&weights))
    }

    /// The SIMD guarded top-k selects the same experts in the same order with
    /// bit-identical weights as the tree kernel, including heavy ties, signed
    /// zeros, -inf entries and the nonfinite serial fallback.
    #[test]
    fn simd_guarded_topk_matches_tree_kernel() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut cases: Vec<Vec<f32>> = Vec::new();
        for levels in [3u64, 7, 40, 1 << 20] {
            for _ in 0..6 {
                cases.push(
                    (0..512)
                        .map(|_| (next() % levels) as f32 * 0.25 - 2.0)
                        .collect(),
                );
            }
        }
        let mut zeros = vec![-1.0f32; 512];
        for (i, v) in zeros.iter_mut().enumerate().step_by(37) {
            *v = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        cases.push(zeros);
        let mut minus_inf = vec![f32::NEG_INFINITY; 512];
        minus_inf[300] = 1.0;
        minus_inf[7] = 1.0;
        cases.push(minus_inf);
        let mut nan = cases[0].clone();
        nan[123] = f32::NAN;
        cases.push(nan);
        let mut inf = cases[1].clone();
        inf[400] = f32::INFINITY;
        cases.push(inf);
        for (case, logits) in cases.iter().enumerate() {
            assert_eq!(
                run(&ctx, SIMD_KERNEL, logits),
                run(&ctx, TREE_KERNEL, logits),
                "case {case}"
            );
        }
    }
}
