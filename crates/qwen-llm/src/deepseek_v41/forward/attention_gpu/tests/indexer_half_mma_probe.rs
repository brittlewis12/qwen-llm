use super::super::*;
use objc2_metal::MTLSize;

const HEADS: usize = 32;
const DIM: usize = 128;

fn sample(seed: usize) -> f32 {
    (((seed.wrapping_mul(2_654_435_761).wrapping_add(17) >> 9) % 4093) as f32 / 2048.0) - 1.0
}

fn tensor(ctx: &MetalContext, bytes: &[u8], shape: &[usize], dtype: GgmlType) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytes, shape.iter().map(|&n| n as u64).collect(), dtype)
        .expect("probe tensor")
}

fn score(
    ctx: &MetalContext,
    pipeline: &str,
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
    depth: usize,
    queries: usize,
) -> f64 {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let pso = ctx.pipeline(pipeline).unwrap();
    enc.set_pipeline(&pso);
    let prefix = depth - queries;
    enc.set_bytes(0, &[prefix as u32, 1, 512, depth as u32]);
    enc.set_bytes(
        1,
        &[DIM as u32, HEADS as u32, (DIM * HEADS) as u32, HEADS as u32],
    );
    enc.set_bytes(2, &[depth as u32, queries as u32, 0, 0]);
    enc.set_tensor(3, query);
    enc.set_tensor(4, weights);
    enc.set_tensor(5, keys);
    enc.set_tensor(6, scores);
    enc.set_bytes(7, &0u32);
    enc.dispatch(
        MTLSize {
            width: depth.div_ceil(128),
            height: queries,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    command.commit();
    crate::metal::wait_completed(&command).unwrap();
    assert!(command.error().is_none(), "{pipeline} command failed");
    (command.GPUEndTime() - command.GPUStartTime()) * 1e3
}

#[test]
fn v41_indexer_f16_mma_speed_bound() {
    if std::env::var_os("QWEN_V41_INDEXER_F16_MMA_PROBE").is_none() {
        return;
    }
    let ctx = MetalContext::new().expect("F16 MMA probe needs Metal");
    let max_depth = 128_000usize;
    let max_queries = 1_024usize;
    let raw_query = (0..max_queries * HEADS * DIM)
        .map(sample)
        .collect::<Vec<_>>();
    let raw_keys = (0..max_depth * DIM)
        .map(|i| sample(i + 991))
        .collect::<Vec<_>>();
    let half_query = raw_query
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let half_keys = raw_keys
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let raw_query_t = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_query),
        &[raw_query.len()],
        GgmlType::F32,
    );
    let half_query_t = tensor(
        &ctx,
        bytemuck::cast_slice(&half_query),
        &[half_query.len()],
        GgmlType::F16,
    );
    let raw_keys_t = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_keys),
        &[raw_keys.len()],
        GgmlType::F32,
    );
    let half_keys_t = tensor(
        &ctx,
        bytemuck::cast_slice(&half_keys),
        &[half_keys.len()],
        GgmlType::F16,
    );
    let weights = vec![1.0f32; max_queries * HEADS];
    let weights_t = tensor(
        &ctx,
        bytemuck::cast_slice(&weights),
        &[weights.len()],
        GgmlType::F32,
    );

    for depth in [80_000usize, 128_000] {
        for queries in [1usize, 2, 3, 256, 1_024] {
            let base_scores = MetalTensor::zeros_f32(&ctx, vec![(depth * queries) as u64]).unwrap();
            let half_scores = MetalTensor::zeros_f32(&ctx, vec![(depth * queries) as u64]).unwrap();
            for _ in 0..2 {
                score(
                    &ctx,
                    "kernel_deepseek_v41_attention_index_heads_rows_mma",
                    &raw_query_t,
                    &weights_t,
                    &raw_keys_t,
                    &base_scores,
                    depth,
                    queries,
                );
                score(
                    &ctx,
                    "kernel_deepseek_v41_attention_index_heads_rows_mma_f16_probe",
                    &half_query_t,
                    &weights_t,
                    &half_keys_t,
                    &half_scores,
                    depth,
                    queries,
                );
            }
            let mut base = Vec::with_capacity(3);
            let mut candidate = Vec::with_capacity(3);
            for pair in 0..3 {
                if pair % 2 == 0 {
                    base.push(score(
                        &ctx,
                        "kernel_deepseek_v41_attention_index_heads_rows_mma",
                        &raw_query_t,
                        &weights_t,
                        &raw_keys_t,
                        &base_scores,
                        depth,
                        queries,
                    ));
                    candidate.push(score(
                        &ctx,
                        "kernel_deepseek_v41_attention_index_heads_rows_mma_f16_probe",
                        &half_query_t,
                        &weights_t,
                        &half_keys_t,
                        &half_scores,
                        depth,
                        queries,
                    ));
                } else {
                    candidate.push(score(
                        &ctx,
                        "kernel_deepseek_v41_attention_index_heads_rows_mma_f16_probe",
                        &half_query_t,
                        &weights_t,
                        &half_keys_t,
                        &half_scores,
                        depth,
                        queries,
                    ));
                    base.push(score(
                        &ctx,
                        "kernel_deepseek_v41_attention_index_heads_rows_mma",
                        &raw_query_t,
                        &weights_t,
                        &raw_keys_t,
                        &base_scores,
                        depth,
                        queries,
                    ));
                }
            }
            let mut a = base.clone();
            let mut b = candidate.clone();
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            println!(
                "v41_indexer_f16_mma depth={depth} queries={queries} f32_ms={base:?} f16_ms={candidate:?} median_speedup={:.4}",
                a[1] / b[1]
            );
        }
    }
}
