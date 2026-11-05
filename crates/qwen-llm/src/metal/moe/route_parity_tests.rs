//! Bitwise parity against the pre-guard routing kernels copied from main into
//! `kernels/research/qwen_moe_incumbent.metal`.

use super::*;
use crate::test_fixtures::metal_context_or_skip;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TopkArgs {
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DotArgs {
    n_expert: u32,
    topk: u32,
    hidden: u32,
    n_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BucketArgs {
    n_expert: u32,
    n_tokens: u32,
    topk: u32,
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn tensor_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, &bytes_f32(values), shape, GgmlType::F32).unwrap()
}

fn tensor_i32(ctx: &MetalContext, values: &[i32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::I32).unwrap()
}

fn read_i32(t: &MetalTensor, n: usize) -> Vec<i32> {
    unsafe {
        std::slice::from_raw_parts(
            (t.buffer.contents().as_ptr() as *const u8)
                .add(t.offset as usize)
                .cast::<i32>(),
            n,
        )
        .to_vec()
    }
}

fn read_u32(t: &MetalTensor, n: usize) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            (t.buffer.contents().as_ptr() as *const u8)
                .add(t.offset as usize)
                .cast::<u32>(),
            n,
        )
        .to_vec()
    }
}

fn run_incumbent_topk(
    ctx: &MetalContext,
    kernel: &str,
    logits: &MetalTensor,
    n: usize,
    k: usize,
    parallel: bool,
) -> (Vec<i32>, Vec<u32>) {
    let ids = MetalTensor::zeros_i32(ctx, vec![k as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![k as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    enc.set_pipeline(&ctx.pipeline(kernel).unwrap());
    enc.set_bytes(
        0,
        &TopkArgs {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, &ids);
    enc.set_tensor(3, &weights);
    if parallel {
        for slot in 0..3 {
            enc.set_threadgroup_memory(slot, 256 * 4);
        }
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end();
    crate::metal::commit_and_wait(&command).unwrap();
    (read_i32(&ids, k), read_u32(&weights, k))
}

fn run_branch_topk(
    ctx: &MetalContext,
    parallel: bool,
    logits: &MetalTensor,
    n: usize,
    k: usize,
) -> (Vec<i32>, Vec<u32>, i32) {
    let ids = MetalTensor::zeros_i32(ctx, vec![k as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![k as u64]).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    if parallel {
        encode_topk_logits_softmax_parallel_f32_status(
            ctx, &enc, logits, &ids, &weights, &status, n, k,
        )
        .unwrap();
    } else {
        encode_topk_logits_softmax_f32_status(ctx, &enc, logits, &ids, &weights, &status, n, k)
            .unwrap();
    }
    enc.end();
    crate::metal::commit_and_wait(&command).unwrap();
    (
        read_i32(&ids, k),
        read_u32(&weights, k),
        read_i32(&status, 1)[0],
    )
}

fn finite_cases(n: usize, _k: usize) -> Vec<Vec<f32>> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cases = Vec::new();
    // Dense random values, coarse ties, signed zeros, near ties and extremes.
    cases.push(
        (0..n)
            .map(|_| ((next() % 101) as f32 - 50.0) * 0.03125)
            .collect(),
    );
    cases.push((0..n).map(|i| ((i * 17) % 7) as f32 * 0.25).collect());
    let mut zeros = vec![-1.0f32; n];
    for (i, v) in zeros.iter_mut().enumerate().step_by(5) {
        *v = if i % 2 == 0 { 0.0 } else { -0.0 };
    }
    cases.push(zeros);
    let mut near = (0..n).map(|i| i as f32 * 1e-3).collect::<Vec<_>>();
    if n >= 2 {
        near[0] = 1.0;
        near[1] = f32::from_bits(1.0f32.to_bits() + 1);
    }
    cases.push(near);
    let mut extreme = (0..n).map(|i| i as f32 * 0.5).collect::<Vec<_>>();
    extreme[0] = f32::MAX;
    extreme[1 % n] = -f32::MAX;
    extreme[2 % n] = f32::MIN_POSITIVE;
    extreme[3 % n] = -f32::MIN_POSITIVE;
    cases.push(extreme);
    cases
}

#[test]
fn serial_and_parallel_topk_match_incumbent_bitwise() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    for parallel in [false, true] {
        for (case, values) in finite_cases(128, 6).into_iter().enumerate() {
            let logits = tensor_f32(&ctx, &values, vec![values.len() as u64]);
            let expected = run_incumbent_topk(
                &ctx,
                if parallel {
                    "kernel_incumbent_topk_logits_softmax_parallel_f32"
                } else {
                    "kernel_incumbent_topk_logits_softmax_f32"
                },
                &logits,
                values.len(),
                6,
                parallel,
            );
            let actual = run_branch_topk(&ctx, parallel, &logits, values.len(), 6);
            assert_eq!(actual.0, expected.0, "ids parallel={parallel} case={case}");
            assert_eq!(
                actual.1, expected.1,
                "weight bits parallel={parallel} case={case}"
            );
            assert_eq!(actual.2 & MOE_ROUTE_STATUS_NONFINITE, 0);
        }
    }
}

#[test]
fn dot_and_packed_topk_match_incumbent_bitwise() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    for (n_expert, topk, hidden) in [(256usize, 8usize, 32usize)] {
        for (case, values) in finite_cases(n_expert, topk).into_iter().enumerate() {
            let logits = tensor_f32(&ctx, &values, vec![n_expert as u64]);
            let shared_weight = tensor_f32(&ctx, &vec![0.25; hidden], vec![hidden as u64]);
            let x = tensor_f32(&ctx, &vec![0.5; hidden], vec![hidden as u64]);
            let ids = MetalTensor::zeros_i32(&ctx, vec![topk as u64]).unwrap();
            let weights = MetalTensor::zeros_f32(&ctx, vec![topk as u64]).unwrap();
            let shared_out = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
            let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode_topk_logits_softmax_dot_sigmoid_f32_status(
                &ctx,
                &enc,
                &logits,
                &shared_weight,
                &x,
                &ids,
                &weights,
                &shared_out,
                &status,
                n_expert,
                topk,
                hidden,
            )
            .unwrap();
            enc.end();
            crate::metal::commit_and_wait(&command).unwrap();

            let exp_ids = MetalTensor::zeros_i32(&ctx, vec![topk as u64]).unwrap();
            let exp_w = MetalTensor::zeros_f32(&ctx, vec![topk as u64]).unwrap();
            let exp_shared = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            enc.set_pipeline(
                &ctx.pipeline("kernel_incumbent_topk_logits_softmax_dot_sigmoid_f32")
                    .unwrap(),
            );
            enc.set_bytes(
                0,
                &DotArgs {
                    n_expert: n_expert as u32,
                    topk: topk as u32,
                    hidden: hidden as u32,
                    n_tokens: 1,
                },
            );
            enc.set_tensor(1, &logits);
            enc.set_tensor(2, &shared_weight);
            enc.set_tensor(3, &x);
            enc.set_tensor(4, &exp_ids);
            enc.set_tensor(5, &exp_w);
            enc.set_tensor(6, &exp_shared);
            for slot in 0..3 {
                enc.set_threadgroup_memory(slot, 256 * 4);
            }
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end();
            crate::metal::commit_and_wait(&command).unwrap();
            assert_eq!(
                read_i32(&ids, topk),
                read_i32(&exp_ids, topk),
                "ids case={case}"
            );
            assert_eq!(
                read_u32(&weights, topk),
                read_u32(&exp_w, topk),
                "weight bits case={case}"
            );
        }
    }

    for n_tokens in [1usize, 2, 7, 33] {
        let n_expert = 512usize;
        let topk = 10usize;
        let hidden = 32usize;
        let mut values = Vec::with_capacity(n_tokens * n_expert);
        for token in 0..n_tokens {
            let mut token_values = finite_cases(n_expert, topk).remove((token + 1) % 5);
            values.append(&mut token_values);
        }
        let logits = tensor_f32(&ctx, &values, vec![(n_tokens * n_expert) as u64]);
        let shared_weight = tensor_f32(&ctx, &vec![0.125; hidden], vec![hidden as u64]);
        let x = tensor_f32(
            &ctx,
            &vec![0.75; n_tokens * hidden],
            vec![(n_tokens * hidden) as u64],
        );
        let ids = MetalTensor::zeros_i32(&ctx, vec![(n_tokens * topk) as u64]).unwrap();
        let weights = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * topk) as u64]).unwrap();
        let shared_out = MetalTensor::zeros_f32(&ctx, vec![n_tokens as u64]).unwrap();
        let status = MetalTensor::zeros_i32(&ctx, vec![n_tokens as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode_topk_logits_softmax_dot_sigmoid_packed_f32_status(
            &ctx,
            &enc,
            &logits,
            &shared_weight,
            &x,
            &ids,
            &weights,
            &shared_out,
            &status,
            n_expert,
            topk,
            hidden,
            n_tokens,
        )
        .unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();

        let exp_ids = MetalTensor::zeros_i32(&ctx, vec![(n_tokens * topk) as u64]).unwrap();
        let exp_w = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * topk) as u64]).unwrap();
        let exp_shared = MetalTensor::zeros_f32(&ctx, vec![n_tokens as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        enc.set_pipeline(
            &ctx.pipeline("kernel_incumbent_topk_logits_softmax_dot_sigmoid_packed_f32")
                .unwrap(),
        );
        enc.set_bytes(
            0,
            &DotArgs {
                n_expert: n_expert as u32,
                topk: topk as u32,
                hidden: hidden as u32,
                n_tokens: n_tokens as u32,
            },
        );
        enc.set_tensor(1, &logits);
        enc.set_tensor(2, &shared_weight);
        enc.set_tensor(3, &x);
        enc.set_tensor(4, &exp_ids);
        enc.set_tensor(5, &exp_w);
        enc.set_tensor(6, &exp_shared);
        for slot in 0..3 {
            enc.set_threadgroup_memory(slot, 512 * 4);
        }
        enc.dispatch(
            MTLSize {
                width: 1,
                height: n_tokens,
                depth: 1,
            },
            MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        assert_eq!(
            read_i32(&ids, n_tokens * topk),
            read_i32(&exp_ids, n_tokens * topk),
            "packed ids tokens={n_tokens}"
        );
        assert_eq!(
            read_u32(&weights, n_tokens * topk),
            read_u32(&exp_w, n_tokens * topk),
            "packed weight bits tokens={n_tokens}"
        );
    }
}

#[test]
fn flash_next_guarded_topk_matches_incumbent_bitwise() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    for kernel in ["tree", "simd"] {
        for (case, values) in finite_cases(512, 10).into_iter().enumerate() {
            let logits = tensor_f32(&ctx, &values, vec![512]);
            let branch = if kernel == "tree" {
                "kernel_qwen4exp_topk_guarded_f32"
            } else {
                "kernel_qwen4exp_topk_guarded_simd_f32"
            };
            let incumbent = if kernel == "tree" {
                "kernel_incumbent_qwen4exp_topk_guarded_f32"
            } else {
                "kernel_incumbent_qwen4exp_topk_guarded_simd_f32"
            };
            let run = |name: &str, status: Option<&MetalTensor>| {
                let ids = MetalTensor::zeros_i32(&ctx, vec![10]).unwrap();
                let weights = MetalTensor::zeros_f32(&ctx, vec![10]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                enc.set_pipeline(&ctx.pipeline(name).unwrap());
                enc.set_bytes(0, &TopkArgs { n: 512, k: 10 });
                enc.set_tensor(1, &logits);
                enc.set_tensor(2, &ids);
                enc.set_tensor(3, &weights);
                if let Some(status) = status {
                    enc.set_tensor(4, status);
                }
                for slot in 0..3 {
                    enc.set_threadgroup_memory(slot, 2048);
                }
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 512,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.end();
                crate::metal::commit_and_wait(&command).unwrap();
                (read_i32(&ids, 10), read_u32(&weights, 10))
            };
            let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let actual = run(branch, Some(&status));
            let expected = run(incumbent, None);
            assert_eq!(actual.0, expected.0, "{kernel} ids case={case}");
            assert_eq!(actual.1, expected.1, "{kernel} weight bits case={case}");
        }
    }
}

#[test]
fn bucket_slots_match_incumbent_bitwise() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    let n_expert = 16usize;
    let n_tokens = 32usize;
    let topk = 4usize;
    let mut top_idx = Vec::with_capacity(n_tokens * topk);
    for token in 0..n_tokens {
        for slot in 0..topk {
            top_idx.push(((token * 7 + slot * 5) % n_expert) as i32);
        }
    }
    let top = tensor_i32(&ctx, &top_idx, vec![(n_tokens * topk) as u64]);
    for parallel in [false, true] {
        let counts = MetalTensor::zeros_i32(&ctx, vec![n_expert as u64]).unwrap();
        let ids = MetalTensor::zeros_i32(&ctx, vec![(n_expert * n_tokens) as u64]).unwrap();
        let status = MetalTensor::zeros_i32(&ctx, vec![n_tokens as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode_route_bucket(
            &ctx, &enc, &top, &counts, &ids, &status, n_expert, n_tokens, topk, parallel,
        )
        .unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();

        let exp_counts = MetalTensor::zeros_i32(&ctx, vec![n_expert as u64]).unwrap();
        let exp_ids = MetalTensor::zeros_i32(&ctx, vec![(n_expert * n_tokens) as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        enc.set_pipeline(
            &ctx.pipeline(if parallel {
                "kernel_incumbent_moe_route_bucket_slots_parallel_f32"
            } else {
                "kernel_incumbent_moe_route_bucket_slots_f32"
            })
            .unwrap(),
        );
        enc.set_bytes(
            0,
            &BucketArgs {
                n_expert: n_expert as u32,
                n_tokens: n_tokens as u32,
                topk: topk as u32,
            },
        );
        enc.set_tensor(1, &top);
        enc.set_tensor(2, &exp_counts);
        enc.set_tensor(3, &exp_ids);
        enc.dispatch(
            MTLSize {
                width: if parallel { n_expert } else { 1 },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        assert_eq!(
            read_i32(&counts, n_expert),
            read_i32(&exp_counts, n_expert),
            "parallel={parallel} counts"
        );
        assert_eq!(
            read_i32(&ids, n_expert * n_tokens),
            read_i32(&exp_ids, n_expert * n_tokens),
            "parallel={parallel} ids"
        );
    }
}

#[test]
fn routed_weighted_sum_output_bits_match_incumbent_route() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    let n = 64usize;
    let k = 6usize;
    let n_out = 32usize;
    for (case, values) in finite_cases(n, k).into_iter().enumerate() {
        let logits = tensor_f32(&ctx, &values, vec![n as u64]);
        let expected_route = run_incumbent_topk(
            &ctx,
            "kernel_incumbent_topk_logits_softmax_f32",
            &logits,
            n,
            k,
            false,
        );
        let actual_route = run_branch_topk(&ctx, false, &logits, n, k);
        let expert_out = |ids: &[i32]| {
            let mut values = Vec::with_capacity(k * n_out);
            for (slot, &id) in ids.iter().enumerate() {
                for col in 0..n_out {
                    values.push((id as f32 * 0.125) + (slot as f32 * 0.5) + (col as f32 * 1e-4));
                }
            }
            tensor_f32(&ctx, &values, vec![(k * n_out) as u64])
        };
        let weights_expected = MetalTensor::from_bytes(
            &ctx,
            unsafe { std::slice::from_raw_parts(expected_route.1.as_ptr().cast::<u8>(), k * 4) },
            vec![k as u64],
            GgmlType::F32,
        )
        .unwrap();
        let weights_actual = MetalTensor::from_bytes(
            &ctx,
            unsafe { std::slice::from_raw_parts(actual_route.1.as_ptr().cast::<u8>(), k * 4) },
            vec![k as u64],
            GgmlType::F32,
        )
        .unwrap();
        let out_expected = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let out_actual = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode_moe_weighted_sum_f32(
            &ctx,
            &enc,
            &expert_out(&expected_route.0),
            &weights_expected,
            &out_expected,
            n_out,
            k,
        )
        .unwrap();
        encode_moe_weighted_sum_f32(
            &ctx,
            &enc,
            &expert_out(&actual_route.0),
            &weights_actual,
            &out_actual,
            n_out,
            k,
        )
        .unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        assert_eq!(
            read_u32(&out_expected, n_out),
            read_u32(&out_actual, n_out),
            "layer output bits case={case}"
        );
    }
}
