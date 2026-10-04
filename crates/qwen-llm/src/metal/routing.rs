//! Learned single-token expert routing shared by DeepSeek V4 and GLM-5.3-Flash.
//!
//! One threadgroup, one thread per expert: select the top-k of
//! `score(logit) + bias` (lowest expert id wins ties), then weight each
//! selected expert by its *unbiased* score over `max(sum, 2^-14)`, times
//! `routed_scale`. Results stay on the GPU in caller-owned views: `ids` (I32
//! `[top_k]`), `weights` (F32 `[top_k]`) and `status` (I32 `[1]`, see
//! [`ROUTE_STATUS_READY`]); consumers check `status` before reading ids.
//!
//! Tie policy: exact ties break toward the lowest expert id. llama.cpp's GLM
//! routing uses `ggml_argsort_top_k`, whose Metal bitonic sort has no id
//! tiebreak, so exactly tied selection scores can pick different experts;
//! exact-id parity is not claimed for ties (measure-zero for real logits).

use super::*;

pub const ROUTE_MAX_EXPERTS: usize = 512;
pub const ROUTE_MAX_TOP_K: usize = 16;
pub const ROUTE_STATUS_PENDING: i32 = 0;
pub const ROUTE_STATUS_READY: i32 = 1;
pub const ROUTE_STATUS_NONFINITE_LOGIT: i32 = -1;
pub const ROUTE_STATUS_NONFINITE_BIAS: i32 = -2;
pub const ROUTE_STATUS_NONFINITE_WEIGHT: i32 = -6;
pub const ROUTE_STATUS_INVALID_SHAPE: i32 = -7;

/// Per-expert score before the selection bias is added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteScore {
    /// DeepSeek V4: `sqrt(softplus(logit))`.
    SqrtSoftplus,
    /// GLM-5.3: `sigmoid(logit)`.
    Sigmoid,
}

impl RouteScore {
    fn kernel(self) -> &'static str {
        match self {
            Self::SqrtSoftplus => "kernel_deepseek_v4_route_learned",
            Self::Sigmoid => "kernel_route_learned_sigmoid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnedRoute {
    pub experts: usize,
    pub top_k: usize,
    pub score: RouteScore,
    pub routed_scale: f32,
}

impl LearnedRoute {
    pub fn validate(&self) -> Result<(), MetalError> {
        if self.experts == 0 || self.experts > ROUTE_MAX_EXPERTS {
            return Err(bad(format!(
                "expert count {} outside 1..={ROUTE_MAX_EXPERTS}",
                self.experts
            )));
        }
        if self.top_k == 0 || self.top_k > ROUTE_MAX_TOP_K || self.top_k > self.experts {
            return Err(bad(format!(
                "top-k {} outside 1..={}",
                self.top_k,
                ROUTE_MAX_TOP_K.min(self.experts)
            )));
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return Err(bad(format!(
                "routed scale must be finite and positive, got {}",
                self.routed_scale
            )));
        }
        Ok(())
    }

    /// Threads per routing threadgroup: one per expert, whole simdgroups.
    pub fn threads(&self) -> usize {
        self.experts.next_multiple_of(32)
    }
}

const KERNEL: &str = "route_learned";

fn bad(detail: impl Into<String>) -> MetalError {
    super::checks::bad_shape(KERNEL, detail)
}

fn check(
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), MetalError> {
    super::checks::check_tensor(KERNEL, tensor, dtype, shape, writable, name)
}

/// Encode one learned routing decision from `logits` and `bias` (F32
/// `[experts]`) into `ids`, `weights` and `status`.
#[allow(clippy::too_many_arguments)]
pub fn encode_route_learned(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    route: &LearnedRoute,
    logits: &MetalTensor,
    bias: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
    status: &MetalTensor,
) -> Result<(), MetalError> {
    if enc.is_concurrent() {
        return Err(bad("requires ordered serial dispatches"));
    }
    route.validate()?;
    let experts = route.experts as u64;
    let top_k = route.top_k as u64;
    check(logits, GgmlType::F32, &[experts], false, "logits")?;
    check(bias, GgmlType::F32, &[experts], false, "selection bias")?;
    check(ids, GgmlType::I32, &[top_k], true, "expert ids")?;
    check(weights, GgmlType::F32, &[top_k], true, "weights")?;
    check(status, GgmlType::I32, &[1], true, "status")?;
    for written in [ids, weights, status] {
        super::checks::check_disjoint(
            KERNEL,
            written,
            &[(logits, "logits"), (bias, "selection bias")],
        )?;
    }
    let kernel = route.score.kernel();
    let pso = ctx.pipeline(kernel)?;
    let threads = route.threads();
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads {
        return Err(bad(format!(
            "{kernel} needs 32-lane simdgroups and {threads} threads, pipeline has {} and {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        )));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        expert_count: u32,
        top_k: u32,
        token_id: u32,
        vocab_size: u32,
        routed_scale: f32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            expert_count: experts as u32,
            top_k: top_k as u32,
            token_id: 0,
            vocab_size: 0,
            routed_scale: route.routed_scale,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, bias);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, weights);
    enc.set_tensor(5, status);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::offset_tensor;
    use super::*;

    /// CPU contract: selection by score + bias with lowest-id ties; weights
    /// from unbiased scores over max(sum, 2^-14), times the routed scale.
    fn reference(route: &LearnedRoute, logits: &[f32], bias: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let score = |x: f32| match route.score {
            RouteScore::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            RouteScore::SqrtSoftplus => {
                let sp = if x > 20.0 {
                    x
                } else if x < -20.0 {
                    x.exp()
                } else {
                    (1.0 + x.exp()).ln()
                };
                sp.sqrt()
            }
        };
        let mut order: Vec<usize> = (0..route.experts).collect();
        order.sort_by(|&a, &b| {
            let (sa, sb) = (score(logits[a]) + bias[a], score(logits[b]) + bias[b]);
            sb.partial_cmp(&sa).unwrap().then(a.cmp(&b))
        });
        let ids: Vec<usize> = order[..route.top_k].to_vec();
        let sum: f32 = ids.iter().map(|&i| score(logits[i])).sum();
        let denominator = sum.max(6.103_515_6e-5);
        let weights = ids
            .iter()
            .map(|&i| score(logits[i]) / denominator * route.routed_scale)
            .collect();
        (ids.into_iter().map(|i| i as i32).collect(), weights)
    }

    fn f32_tensor(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            vec![values.len() as u64],
            GgmlType::F32,
        )
    }

    fn i32_tensor(ctx: &MetalContext, len: usize) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(&vec![-9i32; len]),
            20,
            vec![len as u64],
            GgmlType::I32,
        )
    }

    fn read<T: bytemuck::Pod>(tensor: &MetalTensor) -> Vec<T> {
        let n = tensor.n_elements() as usize;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                n * std::mem::size_of::<T>(),
            )
        };
        bytemuck::cast_slice(bytes).to_vec()
    }

    fn run(
        ctx: &MetalContext,
        route: &LearnedRoute,
        logits: &[f32],
        bias: &[f32],
    ) -> (i32, Vec<i32>, Vec<f32>) {
        let logits_t = f32_tensor(ctx, logits);
        let bias_t = f32_tensor(ctx, bias);
        let ids_t = i32_tensor(ctx, route.top_k);
        let weights_t = f32_tensor(ctx, &vec![0.0; route.top_k]);
        let status_t = i32_tensor(ctx, 1);
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&command);
        encode_route_learned(
            ctx, &enc, route, &logits_t, &bias_t, &ids_t, &weights_t, &status_t,
        )
        .unwrap();
        enc.end();
        command.commit();
        wait_completed(&command).expect("route command");
        (
            read::<i32>(&status_t)[0],
            read::<i32>(&ids_t),
            read::<f32>(&weights_t),
        )
    }

    fn assert_route(ctx: &MetalContext, route: &LearnedRoute, logits: &[f32], bias: &[f32]) {
        let (status, ids, weights) = run(ctx, route, logits, bias);
        assert_eq!(status, ROUTE_STATUS_READY);
        let (want_ids, want_weights) = reference(route, logits, bias);
        assert_eq!(ids, want_ids);
        for (a, e) in weights.iter().zip(&want_weights) {
            assert!(
                (a - e).abs() <= 2e-6 * (1.0 + e.abs()),
                "{weights:?} vs {want_weights:?}"
            );
        }
    }

    const GLM: LearnedRoute = LearnedRoute {
        experts: 288,
        top_k: 8,
        score: RouteScore::Sigmoid,
        routed_scale: 2.5,
    };

    #[test]
    fn sigmoid_route_matches_contract_at_glm_shape() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let base: Vec<f32> = (0..288)
            .map(|i| ((i * 37 + 11) % 101) as f32 * 0.05 - 2.5)
            .collect();
        let zero = vec![0.0; 288];
        assert_route(&ctx, &GLM, &base, &zero);

        // Winners in the ninth simdgroup (256..288), outside DS4's old range.
        let mut high = base.clone();
        for (k, i) in [256, 263, 270, 277, 284, 287, 260, 281]
            .into_iter()
            .enumerate()
        {
            high[i] = 8.0 - k as f32 * 0.5;
        }
        let (_, ids, _) = run(&ctx, &GLM, &high, &zero);
        assert!(ids.iter().all(|&i| i >= 256), "{ids:?}");
        assert_route(&ctx, &GLM, &high, &zero);

        // Bias changes selection only: weights use unbiased sigmoid values.
        let mut bias = zero.clone();
        bias[5] = 3.0;
        bias[200] = 2.0;
        let (_, ids, _) = run(&ctx, &GLM, &high, &bias);
        assert!(ids.contains(&5) && ids.contains(&200), "{ids:?}");
        assert_route(&ctx, &GLM, &high, &bias);

        // Exact ties resolve to the lowest id.
        let tied = vec![0.75; 288];
        let (_, ids, _) = run(&ctx, &GLM, &tied, &zero);
        assert_eq!(ids, [0, 1, 2, 3, 4, 5, 6, 7]);

        // Saturated and vanishing sigmoids, including the denominator floor.
        let mut extreme = vec![-90.0; 288];
        extreme[17] = 40.0;
        extreme[271] = 35.0;
        assert_route(&ctx, &GLM, &extreme, &zero);
        let tiny = vec![-30.0; 288];
        assert_route(&ctx, &GLM, &tiny, &zero);
    }

    #[test]
    fn route_reports_nonfinite_inputs_and_matches_ds4_score() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let mut logits: Vec<f32> = (0..288).map(|i| (i % 13) as f32 * 0.1).collect();
        let zero = vec![0.0; 288];
        logits[123] = f32::NAN;
        assert_eq!(
            run(&ctx, &GLM, &logits, &zero).0,
            ROUTE_STATUS_NONFINITE_LOGIT
        );
        logits[123] = 0.0;
        let mut bias = zero.clone();
        bias[3] = f32::INFINITY;
        assert_eq!(
            run(&ctx, &GLM, &logits, &bias).0,
            ROUTE_STATUS_NONFINITE_BIAS
        );

        // DS4 geometry through the same entry: 256 experts, top-6, sqrt-softplus.
        let ds4 = LearnedRoute {
            experts: 256,
            top_k: 6,
            score: RouteScore::SqrtSoftplus,
            routed_scale: 1.5,
        };
        let ds4_logits: Vec<f32> = (0..256)
            .map(|i| ((i * 29 + 3) % 97) as f32 * 0.21 - 9.0)
            .collect();
        let ds4_bias: Vec<f32> = (0..256).map(|i| ((i * 7) % 11) as f32 * 0.01).collect();
        assert_route(&ctx, &ds4, &ds4_logits, &ds4_bias);

        assert!(
            LearnedRoute {
                experts: 513,
                ..GLM
            }
            .validate()
            .is_err()
        );
        assert!(LearnedRoute { top_k: 17, ..GLM }.validate().is_err());
        assert!(
            LearnedRoute {
                routed_scale: 0.0,
                ..GLM
            }
            .validate()
            .is_err()
        );
    }

    /// Pins the native tie policy on a mixed tie (lowest id among the 0.5s
    /// fills slot 8). llama.cpp's bitonic argsort picks expert 6 here.
    #[test]
    fn route_mixed_ties_follow_the_lowest_id_policy() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let logits = vec![0.0; 288];
        let mut bias = vec![0.5f32; 288];
        for i in (0..28).step_by(4) {
            bias[i] = 0.75;
        }
        let (status, ids, _) = run(&ctx, &GLM, &logits, &bias);
        assert_eq!(status, ROUTE_STATUS_READY);
        assert_eq!(ids, [0, 4, 8, 12, 16, 20, 24, 1]);
    }

    /// Expert counts that are not a multiple of 32 leave a partially active
    /// final simdgroup (e.g. DS4 REAP variants).
    #[test]
    fn route_handles_expert_counts_not_multiple_of_32() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        for (experts, score) in [
            (200, RouteScore::Sigmoid),
            (160, RouteScore::SqrtSoftplus),
            (216, RouteScore::SqrtSoftplus),
        ] {
            let route = LearnedRoute {
                experts,
                top_k: 6,
                score,
                routed_scale: 1.5,
            };
            let logits: Vec<f32> = (0..experts)
                .map(|i| ((i * 41 + 7) % 89) as f32 * 0.17 - 7.0)
                .collect();
            let bias: Vec<f32> = (0..experts).map(|i| ((i * 3) % 7) as f32 * 0.02).collect();
            assert_route(&ctx, &route, &logits, &bias);
        }
    }

    #[test]
    fn route_refuses_outputs_aliasing_inputs() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let logits = f32_tensor(&ctx, &vec![0.0; 288]);
        let bias = f32_tensor(&ctx, &vec![0.0; 288]);
        let ids = i32_tensor(&ctx, 8);
        let status = i32_tensor(&ctx, 1);
        let aliased_weights = MetalTensor {
            shape: vec![8],
            ..logits.clone()
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let err = encode_route_learned(
            &ctx,
            &enc,
            &GLM,
            &logits,
            &bias,
            &ids,
            &aliased_weights,
            &status,
        )
        .unwrap_err();
        assert!(err.to_string().contains("aliases logits"), "{err}");
    }
}
