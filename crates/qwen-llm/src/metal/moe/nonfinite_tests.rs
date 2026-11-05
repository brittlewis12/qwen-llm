//! Isolated single-logit non-finite status coverage for every Qwen route
//! kernel variant.

use super::*;
use crate::test_fixtures::metal_context_or_skip;

fn finite_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 101) as f32 * 0.013_671_875 - 0.683_593_75)
        .collect()
}

fn tensor_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
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

fn nonfinite_values(values: &mut [f32], index: usize, value: f32) {
    values[index] = value;
}

fn assert_nonfinite(word: i32, label: &str) {
    assert_ne!(
        word & MOE_ROUTE_STATUS_NONFINITE,
        0,
        "{label}: status {word} did not report NONFINITE"
    );
    assert_ne!(word & MOE_ROUTE_STATUS_READY, 0, "{label}: status {word}");
}

fn run_parallel(ctx: &MetalContext, n: usize, k: usize, values: &[f32]) -> i32 {
    let logits = tensor_f32(ctx, values, vec![n as u64]);
    let ids = MetalTensor::zeros_i32(ctx, vec![k as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![k as u64]).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode_topk_logits_softmax_parallel_f32_status(
        ctx, &enc, &logits, &ids, &weights, &status, n, k,
    )
    .unwrap();
    enc.end();
    crate::metal::commit_and_wait(&command).unwrap();
    read_i32(&status, 1)[0]
}

fn run_single_dot(
    ctx: &MetalContext,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    values: &[f32],
) -> i32 {
    let logits = tensor_f32(ctx, values, vec![n_expert as u64]);
    let shared_weight = tensor_f32(ctx, &vec![0.125; hidden], vec![hidden as u64]);
    let x = tensor_f32(ctx, &vec![0.75; hidden], vec![hidden as u64]);
    let ids = MetalTensor::zeros_i32(ctx, vec![topk as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![topk as u64]).unwrap();
    let shared_out = MetalTensor::zeros_f32(ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode_topk_logits_softmax_dot_sigmoid_f32_status(
        ctx,
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
    read_i32(&status, 1)[0]
}

fn run_packed_dot(
    ctx: &MetalContext,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
    values: &[f32],
) -> Vec<i32> {
    let logits = tensor_f32(ctx, values, vec![(n_tokens * n_expert) as u64]);
    let shared_weight = tensor_f32(ctx, &vec![0.125; hidden], vec![hidden as u64]);
    let x = tensor_f32(
        ctx,
        &vec![0.75; n_tokens * hidden],
        vec![(n_tokens * hidden) as u64],
    );
    let ids = MetalTensor::zeros_i32(ctx, vec![(n_tokens * topk) as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![(n_tokens * topk) as u64]).unwrap();
    let shared_out = MetalTensor::zeros_f32(ctx, vec![n_tokens as u64]).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![n_tokens as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode_topk_logits_softmax_dot_sigmoid_packed_f32_status(
        ctx,
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
    read_i32(&status, n_tokens)
}

fn run_fused(
    ctx: &MetalContext,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
    values: &[f32],
) -> Vec<i32> {
    let logits = tensor_f32(ctx, values, vec![(n_tokens * n_expert) as u64]);
    let shared_weight = tensor_f32(ctx, &vec![0.125; hidden], vec![hidden as u64]);
    let x = tensor_f32(
        ctx,
        &vec![0.75; n_tokens * hidden],
        vec![(n_tokens * hidden) as u64],
    );
    let ids = MetalTensor::zeros_i32(ctx, vec![(n_tokens * topk) as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(ctx, vec![(n_tokens * topk) as u64]).unwrap();
    let shared_out = MetalTensor::zeros_f32(ctx, vec![n_tokens as u64]).unwrap();
    let counts = MetalTensor::zeros_f32(ctx, vec![n_expert as u64]).unwrap();
    let slots = MetalTensor::zeros_f32(ctx, vec![(n_expert * n_tokens) as u64]).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![n_tokens as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32_status(
        ctx,
        &enc,
        &logits,
        &shared_weight,
        &x,
        &ids,
        &weights,
        &shared_out,
        &counts,
        &slots,
        &status,
        n_expert,
        topk,
        hidden,
        n_tokens,
    )
    .unwrap();
    enc.end();
    crate::metal::commit_and_wait(&command).unwrap();
    read_i32(&status, n_tokens)
}

#[test]
fn isolated_nonfinite_logit_raises_status_in_every_route_kernel() {
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let indices_256 = [0usize, 1, 31, 32, 255];
    let indices_250 = [0usize, 1, 31, 32, 249];
    let indices_512 = [0usize, 1, 31, 32, 255, 511];
    let indices_511 = [0usize, 1, 31, 32, 255, 510];

    for &value in &values {
        for &index in &indices_256 {
            let mut logits = finite_values(256);
            nonfinite_values(&mut logits, index, value);
            assert_nonfinite(
                run_parallel(&ctx, 256, 8, &logits),
                &format!("parallel n=256 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_single_dot(&ctx, 256, 8, 32, &logits),
                &format!("single-dot n=256 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_packed_dot(&ctx, 256, 8, 32, 1, &logits)[0],
                &format!("packed-dot width256 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_fused(&ctx, 256, 8, 32, 1, &logits)[0],
                &format!("fused n=256 idx={index} value={value}"),
            );
        }
        for &index in &indices_250 {
            let mut logits = finite_values(250);
            nonfinite_values(&mut logits, index, value);
            assert_nonfinite(
                run_parallel(&ctx, 250, 8, &logits),
                &format!("parallel tail n=250 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_single_dot(&ctx, 250, 8, 32, &logits),
                &format!("single-dot tail n=250 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_packed_dot(&ctx, 250, 8, 32, 1, &logits)[0],
                &format!("packed-dot tail n=250 idx={index} value={value}"),
            );
            assert_nonfinite(
                run_fused(&ctx, 250, 8, 32, 1, &logits)[0],
                &format!("fused tail n=250 idx={index} value={value}"),
            );
        }
        for &index in &indices_512 {
            let mut logits = finite_values(512 * 2);
            nonfinite_values(&mut logits, index, value);
            let statuses = run_packed_dot(&ctx, 512, 10, 32, 2, &logits);
            assert_nonfinite(
                statuses[0],
                &format!("packed-dot width512 idx={index} value={value}"),
            );
            assert_eq!(
                statuses[1] & MOE_ROUTE_STATUS_NONFINITE,
                0,
                "finite token 1 was marked non-finite"
            );
        }
        for &index in &indices_511 {
            let mut logits = finite_values(511);
            nonfinite_values(&mut logits, index, value);
            assert_nonfinite(
                run_packed_dot(&ctx, 511, 10, 32, 1, &logits)[0],
                &format!("packed-dot tail n=511 idx={index} value={value}"),
            );
        }
    }
}
