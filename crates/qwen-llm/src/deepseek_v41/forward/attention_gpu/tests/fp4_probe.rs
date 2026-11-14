use super::super::*;
use crate::deepseek_v4_oracle::{
    DeepSeekV4IndexerFp4Row, INDEXER_FP4_VALUE_BYTES, pack_indexer_fp4_row, packed_indexer_scores,
    top_k_indices, unpack_indexer_fp4_row,
};
use objc2_metal::MTLSize;
use std::sync::Arc;
use std::time::Instant;

const HEADS: usize = 32;
const DIM: usize = 128;

fn e2m1_unit(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 == 0 { v } else { -v }
}

fn sample(seed: usize) -> f32 {
    (((seed.wrapping_mul(2_654_435_761).wrapping_add(17) >> 9) % 4093) as f32 / 2048.0) - 1.0
}

fn topk_overlap_and_margin(a: &[f32], b: &[f32], k: usize) -> (usize, f32, f32) {
    let a_top = top_k_indices(a, k + 1).unwrap();
    let b_top = top_k_indices(b, k + 1).unwrap();
    let a_set = a_top[..k]
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let overlap = b_top[..k].iter().filter(|id| a_set.contains(id)).count();
    (
        overlap,
        a[a_top[k - 1]] - a[a_top[k]],
        b[b_top[k - 1]] - b[b_top[k]],
    )
}

fn tensor(ctx: &MetalContext, bytes: &[u8], shape: &[usize], dtype: GgmlType) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytes, shape.iter().map(|&n| n as u64).collect(), dtype)
        .expect("probe tensor")
}

fn f32_values(tensor: &MetalTensor, first: usize, count: usize) -> Vec<f32> {
    assert_eq!(tensor.dtype, GgmlType::F32);
    assert!(
        first
            .checked_add(count)
            .is_some_and(|end| end <= tensor.n_elements() as usize)
    );
    let base = unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .add(tensor.offset as usize)
            .cast::<f32>()
    };
    unsafe { std::slice::from_raw_parts(base.add(first), count) }.to_vec()
}

fn gpu_pack(
    ctx: &MetalContext,
    input: &MetalTensor,
    rows: usize,
) -> (MetalTensor, MetalTensor, MetalTensor) {
    let values = MetalTensor::zeros_dtype(
        ctx,
        vec![(rows * INDEXER_FP4_VALUE_BYTES) as u64],
        GgmlType::I8,
    )
    .unwrap();
    let scales = MetalTensor::zeros_dtype(ctx, vec![(rows * 4) as u64], GgmlType::I8).unwrap();
    let status = MetalTensor::zeros_i32(ctx, vec![rows as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let pso = ctx
        .pipeline("kernel_deepseek_v4_pack_indexer_fp4_rows_shadow")
        .unwrap();
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &(rows as u32));
    enc.set_tensor(1, input);
    enc.set_tensor(2, &values);
    enc.set_tensor(3, &scales);
    enc.set_tensor(4, &status);
    enc.dispatch(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    command.commit();
    crate::metal::wait_completed(&command).unwrap();
    assert!(command.error().is_none(), "FP4 pack command failed");
    (values, scales, status)
}

fn gpu_unpack_f32(
    ctx: &MetalContext,
    values: &MetalTensor,
    scales: &MetalTensor,
    rows: usize,
) -> MetalTensor {
    let output = MetalTensor::zeros_f32(ctx, vec![(rows * DIM) as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let pso = ctx
        .pipeline("kernel_deepseek_v41_indexer_fp4_unpack_f32_contract")
        .unwrap();
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &(rows as u32));
    enc.set_tensor(1, values);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, &output);
    enc.dispatch(
        MTLSize {
            width: (rows * DIM).div_ceil(256),
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
    command.commit();
    crate::metal::wait_completed(&command).unwrap();
    assert!(command.error().is_none(), "FP4 unpack command failed");
    output
}

fn assert_gpu_pack_matches_contract(
    values: &MetalTensor,
    scales: &MetalTensor,
    status: &MetalTensor,
    expected: &[DeepSeekV4IndexerFp4Row],
) {
    let got_values = unsafe {
        std::slice::from_raw_parts(
            values.buffer.contents().as_ptr().cast::<u8>(),
            expected.len() * INDEXER_FP4_VALUE_BYTES,
        )
    };
    let got_scales = unsafe {
        std::slice::from_raw_parts(
            scales.buffer.contents().as_ptr().cast::<u8>(),
            expected.len() * 4,
        )
    };
    let got_status = unsafe {
        std::slice::from_raw_parts(
            status.buffer.contents().as_ptr().cast::<i32>(),
            expected.len(),
        )
    };
    for (row, expected_row) in expected.iter().enumerate() {
        assert_eq!(got_status[row], 0, "FP4 pack status row {row}");
        assert_eq!(
            &got_values[row * INDEXER_FP4_VALUE_BYTES..(row + 1) * INDEXER_FP4_VALUE_BYTES],
            &expected_row.as_bytes()[..INDEXER_FP4_VALUE_BYTES],
            "FP4 E2M1 codes row {row}"
        );
        assert_eq!(
            &got_scales[row * 4..(row + 1) * 4],
            &expected_row.as_bytes()[INDEXER_FP4_VALUE_BYTES..],
            "FP4 UE8M0 scales row {row}"
        );
    }
}

fn assert_gpu_dequant_matches_contract(actual: &MetalTensor, expected: &[DeepSeekV4IndexerFp4Row]) {
    let got = unsafe {
        std::slice::from_raw_parts(
            actual.buffer.contents().as_ptr().cast::<f32>(),
            expected.len() * DIM,
        )
    };
    for (row, packed) in expected.iter().enumerate() {
        let reference = unpack_indexer_fp4_row(packed).unwrap();
        for dim in 0..DIM {
            assert_eq!(
                got[row * DIM + dim].to_bits(),
                reference[dim].to_bits(),
                "FP4 dequantized value row {row} dimension {dim}"
            );
        }
    }
}

fn candidate(
    ctx: &MetalContext,
    query_units: &MetalTensor,
    query_scales: &MetalTensor,
    scaled_weights: &MetalTensor,
    key_values: &MetalTensor,
    key_scales: &MetalTensor,
    visible: &MetalTensor,
    scores: &MetalTensor,
    rows: usize,
    queries: usize,
) -> f64 {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let pso = ctx
        .pipeline("kernel_deepseek_v41_indexer_fp4_contract_score")
        .unwrap();
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &[HEADS as u32, DIM as u32, rows as u32, queries as u32]);
    for (slot, t) in [
        (1, query_units),
        (2, query_scales),
        (3, scaled_weights),
        (4, key_values),
        (5, key_scales),
        (6, visible),
        (7, scores),
    ] {
        enc.set_tensor(slot, t);
    }
    enc.set_threadgroup_memory(0, 8 * 32 * std::mem::size_of::<u16>());
    enc.set_threadgroup_memory(1, 32 * 8 * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, 32 * 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: rows.div_ceil(8),
            height: queries,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    command.commit();
    crate::metal::wait_completed(&command).unwrap();
    assert!(
        command.error().is_none(),
        "packed scorer GPU command failed"
    );
    (command.GPUEndTime() - command.GPUStartTime()) * 1e3
}

fn incumbent(
    ctx: &MetalContext,
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
    rows: usize,
    queries: usize,
    start: usize,
    ratio: usize,
    row0: usize,
) -> f64 {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let pso = ctx
        .pipeline("kernel_deepseek_v41_attention_index_heads_rows_mma")
        .unwrap();
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &[start as u32, ratio as u32, 512, rows as u32]);
    enc.set_bytes(
        1,
        &[DIM as u32, HEADS as u32, (DIM * HEADS) as u32, HEADS as u32],
    );
    enc.set_bytes(2, &[rows as u32, queries as u32, 0, 0]);
    enc.set_tensor(3, query);
    enc.set_tensor(4, weights);
    enc.set_tensor(5, keys);
    enc.set_tensor(6, scores);
    enc.set_bytes(7, &(row0 as u32));
    enc.dispatch(
        MTLSize {
            width: rows.div_ceil(128),
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
    assert!(
        command.error().is_none(),
        "incumbent score GPU command failed"
    );
    (command.GPUEndTime() - command.GPUStartTime()) * 1e3
}

#[test]
fn v41_fp4_reference_scorer_probe() {
    if std::env::var_os("QWEN_V41_FP4_INDEXER_PROBE").is_none() {
        return;
    }
    let ctx = MetalContext::new().expect("FP4 probe needs Metal");
    let max_rows = 128 * 1024usize;
    let max_queries = 1024usize;

    let raw_query = (0..max_queries * HEADS * DIM)
        .map(sample)
        .collect::<Vec<_>>();
    let mut query_rows = Vec::<DeepSeekV4IndexerFp4Row>::with_capacity(max_queries * HEADS);
    let mut query_units = Vec::<u16>::with_capacity(max_queries * HEADS * DIM);
    let mut query_scale_bytes = Vec::<u8>::with_capacity(max_queries * HEADS * 4);
    for q in 0..max_queries {
        for h in 0..HEADS {
            let start = (q * HEADS + h) * DIM;
            let row = pack_indexer_fp4_row(&raw_query[start..start + DIM]).unwrap();
            for d in 0..DIM {
                let byte = row.as_bytes()[d / 2];
                let code = if d % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                query_units.push(half::f16::from_f32(e2m1_unit(code)).to_bits());
            }
            query_scale_bytes.extend_from_slice(&row.as_bytes()[INDEXER_FP4_VALUE_BYTES..]);
            query_rows.push(row);
        }
    }

    let raw_keys = (0..max_rows * DIM)
        .map(|i| sample(i + 991))
        .collect::<Vec<_>>();
    let mut key_rows = Vec::<DeepSeekV4IndexerFp4Row>::with_capacity(max_rows);
    let mut key_values = Vec::<u8>::with_capacity(max_rows * INDEXER_FP4_VALUE_BYTES);
    let mut key_scale_bytes = Vec::<u8>::with_capacity(max_rows * 4);
    for row in 0..max_rows {
        let packed = pack_indexer_fp4_row(&raw_keys[row * DIM..(row + 1) * DIM]).unwrap();
        key_values.extend_from_slice(&packed.as_bytes()[..INDEXER_FP4_VALUE_BYTES]);
        key_scale_bytes.extend_from_slice(&packed.as_bytes()[INDEXER_FP4_VALUE_BYTES..]);
        key_rows.push(packed);
    }

    let raw_weights = vec![1.0f32; max_queries * HEADS];
    let scaled_weights = raw_weights
        .iter()
        .map(|w| *w / ((HEADS * DIM) as f32).sqrt())
        .collect::<Vec<_>>();
    let raw_query_t = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_query),
        &[max_queries * HEADS * DIM],
        GgmlType::F32,
    );
    let raw_keys_t = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_keys),
        &[max_rows * DIM],
        GgmlType::F32,
    );
    let (query_values_t, query_scales_t, query_status_t) =
        gpu_pack(&ctx, &raw_query_t, max_queries * HEADS);
    let (key_values_t, key_scales_t, key_status_t) = gpu_pack(&ctx, &raw_keys_t, max_rows);
    assert_gpu_pack_matches_contract(
        &query_values_t,
        &query_scales_t,
        &query_status_t,
        &query_rows,
    );
    assert_gpu_pack_matches_contract(&key_values_t, &key_scales_t, &key_status_t, &key_rows);
    println!(
        "v41_fp4_quant_bitwise query_rows={} key_rows={} payload_bytes_per_key=68",
        query_rows.len(),
        key_rows.len()
    );
    let query_dequantized =
        gpu_unpack_f32(&ctx, &query_values_t, &query_scales_t, max_queries * HEADS);
    assert_gpu_dequant_matches_contract(&query_dequantized, &query_rows);
    let key_dequantized = gpu_unpack_f32(&ctx, &key_values_t, &key_scales_t, max_rows);
    assert_gpu_dequant_matches_contract(&key_dequantized, &key_rows);
    println!(
        "v41_fp4_dequant_bitwise query_values={} key_values={}",
        query_rows.len() * DIM,
        key_rows.len() * DIM
    );

    let current_query = raw_query_t;
    let current_weights = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_weights),
        &[max_queries * HEADS],
        GgmlType::F32,
    );
    let current_keys = tensor(
        &ctx,
        bytemuck::cast_slice(&raw_keys),
        &[max_rows * DIM],
        GgmlType::F32,
    );
    let query_units_t = tensor(
        &ctx,
        bytemuck::cast_slice(&query_units),
        &[max_queries * HEADS * DIM],
        GgmlType::F16,
    );
    let scaled_weights_t = tensor(
        &ctx,
        bytemuck::cast_slice(&scaled_weights),
        &[max_queries * HEADS],
        GgmlType::F32,
    );
    let current_scores =
        MetalTensor::zeros_f32(&ctx, vec![(max_rows * max_queries) as u64]).unwrap();
    let packed_scores =
        MetalTensor::zeros_f32(&ctx, vec![(max_rows * max_queries) as u64]).unwrap();

    // Prove packed MSL scores stay within tight tolerance of the scalar oracle
    // with the target 32-head geometry and real packed operands.
    let visible3 = tensor(&ctx, bytemuck::cast_slice(&[3i32; 3]), &[3], GgmlType::I32);
    let small_scores = MetalTensor::zeros_f32(&ctx, vec![3 * 3]).unwrap();
    let small_query_units = tensor(
        &ctx,
        bytemuck::cast_slice(&query_units[..3 * HEADS * DIM]),
        &[3 * HEADS * DIM],
        GgmlType::F16,
    );
    let small_query_scales = tensor(
        &ctx,
        &query_scale_bytes[..3 * HEADS * 4],
        &[3 * HEADS * 4],
        GgmlType::I8,
    );
    let small_weights = tensor(
        &ctx,
        bytemuck::cast_slice(&scaled_weights[..3 * HEADS]),
        &[3 * HEADS],
        GgmlType::F32,
    );
    let small_key_values = tensor(
        &ctx,
        &key_values[..3 * INDEXER_FP4_VALUE_BYTES],
        &[3 * INDEXER_FP4_VALUE_BYTES],
        GgmlType::I8,
    );
    let small_key_scales = tensor(&ctx, &key_scale_bytes[..12], &[12], GgmlType::I8);
    let mut witness_max_abs = 0.0f32;
    let mut witness_max_rel = 0.0f32;
    for query_idx in 0..3 {
        let expected = packed_indexer_scores(
            &query_rows[query_idx * HEADS..(query_idx + 1) * HEADS],
            &scaled_weights[query_idx * HEADS..(query_idx + 1) * HEADS],
            &key_rows[..3],
        )
        .unwrap();
        let elapsed = candidate(
            &ctx,
            &small_query_units,
            &small_query_scales,
            &small_weights,
            &small_key_values,
            &small_key_scales,
            &visible3,
            &small_scores,
            3,
            3,
        );
        let got = unsafe {
            std::slice::from_raw_parts(small_scores.buffer.contents().as_ptr().cast::<f32>(), 9)
        };
        assert!(elapsed > 0.0);
        for (&actual, &reference) in got[query_idx * 3..query_idx * 3 + 3].iter().zip(&expected) {
            let error = (actual - reference).abs();
            witness_max_abs = witness_max_abs.max(error);
            witness_max_rel = witness_max_rel.max(error / reference.abs().max(1.0e-6));
        }
    }
    println!(
        "v41_fp4_contract_witness max_abs={witness_max_abs:.9e} max_rel={witness_max_rel:.9e}"
    );
    assert!(
        witness_max_rel <= 1.0e-6,
        "packed score error {witness_max_rel:e}"
    );

    // The full stage sweep is shape-only with real 32x128 operands; packing
    // remains resident between arms, matching a persistent key-cache probe.
    for depth in [80_000usize, 128_000] {
        for rows in [1usize, 2, 3, 256, 1024] {
            let start = depth - rows;
            let mut visible = (0..rows)
                .map(|r| (start + r + 1) as i32)
                .collect::<Vec<_>>();
            let visible_t = tensor(&ctx, bytemuck::cast_slice(&visible), &[rows], GgmlType::I32);
            // Warm both paths, then three interleaved pairs.
            for _ in 0..2 {
                incumbent(
                    &ctx,
                    &current_query,
                    &current_weights,
                    &current_keys,
                    &current_scores,
                    depth,
                    rows,
                    start,
                    1,
                    0,
                );
                candidate(
                    &ctx,
                    &query_units_t,
                    &query_scales_t,
                    &scaled_weights_t,
                    &key_values_t,
                    &key_scales_t,
                    &visible_t,
                    &packed_scores,
                    depth,
                    rows,
                );
            }
            let mut base = Vec::new();
            let mut packed = Vec::new();
            for pair in 0..3 {
                if pair % 2 == 0 {
                    base.push(incumbent(
                        &ctx,
                        &current_query,
                        &current_weights,
                        &current_keys,
                        &current_scores,
                        depth,
                        rows,
                        start,
                        1,
                        0,
                    ));
                    packed.push(candidate(
                        &ctx,
                        &query_units_t,
                        &query_scales_t,
                        &scaled_weights_t,
                        &key_values_t,
                        &key_scales_t,
                        &visible_t,
                        &packed_scores,
                        depth,
                        rows,
                    ));
                } else {
                    packed.push(candidate(
                        &ctx,
                        &query_units_t,
                        &query_scales_t,
                        &scaled_weights_t,
                        &key_values_t,
                        &key_scales_t,
                        &visible_t,
                        &packed_scores,
                        depth,
                        rows,
                    ));
                    base.push(incumbent(
                        &ctx,
                        &current_query,
                        &current_weights,
                        &current_keys,
                        &current_scores,
                        depth,
                        rows,
                        start,
                        1,
                        0,
                    ));
                }
            }
            println!(
                "v41_fp4_indexer depth={depth} rows={rows} f32_mma_ms={base:?} fp4_contract_ms={packed:?}"
            );
            visible.clear();
        }
    }
}

#[test]
fn v41_fp4_real_selection_screen() {
    if std::env::var_os("QWEN_V41_FP4_REAL_SELECTION").is_none() {
        return;
    }
    const PREFIX: usize = 80_000;
    const APPEND: usize = 256;
    const TOKENS: [u32; 7] = [0, 5, 223, 939, 23, 695, 22];
    let gguf_path = std::env::var("V41_FP4_GGUF").expect("V41_FP4_GGUF");
    println!("v41_fp4_real_model={gguf_path} prefix={PREFIX} append={APPEND}");
    let load_start = Instant::now();
    let ctx = MetalContext::new().expect("real V4.1 selection needs Metal");
    let gguf = Arc::new(crate::gguf::GgufFile::open(&gguf_path).expect("open production file A"));
    let resident = crate::deepseek_v41::gguf::ResidentWeights::load(&ctx, gguf)
        .expect("load production file A");
    let model = crate::deepseek_v41::forward::Forward::from_resident(
        &ctx,
        &resident,
        resident.token_map.clone(),
        resident.raw_pad,
    )
    .expect("bind production V4.1");
    println!(
        "v41_fp4_real_model_load_ms={:.3}",
        load_start.elapsed().as_secs_f64() * 1e3
    );
    let mut state = model
        .new_state(PREFIX + APPEND)
        .expect("allocate 80K state");
    let prefix = (0..PREFIX)
        .map(|i| TOKENS[i % TOKENS.len()])
        .collect::<Vec<_>>();
    let fill_start = Instant::now();
    model
        .prefill(&mut state, &prefix, None)
        .expect("fill production model to 80K");
    println!(
        "v41_fp4_real_prefix_fill_ms={:.3}",
        fill_start.elapsed().as_secs_f64() * 1e3
    );
    let append = (PREFIX..PREFIX + APPEND)
        .map(|i| TOKENS[i % TOKENS.len()])
        .collect::<Vec<_>>();
    let append_start = Instant::now();
    let (result, captures) =
        super::super::with_fp4_index_input_capture(PREFIX + APPEND - 1, || {
            model.prefill(&mut state, &append, None)
        });
    result.expect("real 256-row append");
    println!(
        "v41_fp4_real_append256_ms={:.3}",
        append_start.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(captures.len(), 8, "one score input per V4.1 index layer");

    let mut worst_overlap = (1.0f64, 0usize, 0usize);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut oracle_max_abs = 0.0f32;
    let mut oracle_max_rel = 0.0f32;
    let mut global_head_overlaps = Vec::<f64>::new();
    let mut global_current_head_margins = Vec::<f32>::new();
    let mut global_fp4_head_margins = Vec::<f32>::new();
    for (layer, input) in [2usize, 8, 14, 20, 24, 28, 32, 36]
        .into_iter()
        .zip(captures)
    {
        let key_rows = input.max_length;
        let raw_query = f32_values(&input.query, 0, HEADS * DIM);
        let raw_weights = f32_values(&input.weights, 0, HEADS);
        let raw_keys = f32_values(&input.keys, 0, key_rows * DIM);
        let query_rows = (0..HEADS)
            .map(|head| pack_indexer_fp4_row(&raw_query[head * DIM..(head + 1) * DIM]).unwrap())
            .collect::<Vec<_>>();
        let key_rows_packed = (0..key_rows)
            .map(|row| pack_indexer_fp4_row(&raw_keys[row * DIM..(row + 1) * DIM]).unwrap())
            .collect::<Vec<_>>();
        let quantized_keys = key_rows_packed
            .iter()
            .map(|row| unpack_indexer_fp4_row(row).unwrap())
            .collect::<Vec<_>>();
        let scaled_weights = raw_weights
            .iter()
            .map(|weight| *weight / ((HEADS * DIM) as f32).sqrt())
            .collect::<Vec<_>>();
        let mut query_units = Vec::<u16>::with_capacity(HEADS * DIM);
        for row in &query_rows {
            for dim in 0..DIM {
                let packed = row.as_bytes()[dim / 2];
                let code = if dim % 2 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                query_units.push(half::f16::from_f32(e2m1_unit(code)).to_bits());
            }
        }
        let mut query_scale_bytes = Vec::with_capacity(HEADS * 4);
        for row in &query_rows {
            query_scale_bytes.extend_from_slice(&row.as_bytes()[INDEXER_FP4_VALUE_BYTES..]);
        }
        let mut key_value_bytes = Vec::with_capacity(key_rows * INDEXER_FP4_VALUE_BYTES);
        let mut key_scale_bytes = Vec::with_capacity(key_rows * 4);
        for row in &key_rows_packed {
            key_value_bytes.extend_from_slice(&row.as_bytes()[..INDEXER_FP4_VALUE_BYTES]);
            key_scale_bytes.extend_from_slice(&row.as_bytes()[INDEXER_FP4_VALUE_BYTES..]);
        }

        let raw_query_t = tensor(
            &ctx,
            bytemuck::cast_slice(&raw_query),
            &[HEADS * DIM],
            GgmlType::F32,
        );
        let raw_keys_t = tensor(
            &ctx,
            bytemuck::cast_slice(&raw_keys),
            &[key_rows * DIM],
            GgmlType::F32,
        );
        let (query_values_t, query_scales_t, query_status_t) = gpu_pack(&ctx, &raw_query_t, HEADS);
        let (key_values_t, key_scales_t, key_status_t) = gpu_pack(&ctx, &raw_keys_t, key_rows);
        assert_gpu_pack_matches_contract(
            &query_values_t,
            &query_scales_t,
            &query_status_t,
            &query_rows,
        );
        assert_gpu_pack_matches_contract(
            &key_values_t,
            &key_scales_t,
            &key_status_t,
            &key_rows_packed,
        );

        let query_units_t = tensor(
            &ctx,
            bytemuck::cast_slice(&query_units),
            &[HEADS * DIM],
            GgmlType::F16,
        );
        let scaled_weights_t = tensor(
            &ctx,
            bytemuck::cast_slice(&scaled_weights),
            &[HEADS],
            GgmlType::F32,
        );
        let visible_t = tensor(
            &ctx,
            bytemuck::cast_slice(&[key_rows as i32]),
            &[1],
            GgmlType::I32,
        );
        let current_scores = MetalTensor::zeros_f32(&ctx, vec![key_rows as u64]).unwrap();
        let packed_scores = MetalTensor::zeros_f32(&ctx, vec![key_rows as u64]).unwrap();
        incumbent(
            &ctx,
            &input.query,
            &input.weights,
            &input.keys,
            &current_scores,
            key_rows,
            1,
            input.start,
            input.ratio,
            0,
        );
        candidate(
            &ctx,
            &query_units_t,
            &query_scales_t,
            &scaled_weights_t,
            &key_values_t,
            &key_scales_t,
            &visible_t,
            &packed_scores,
            key_rows,
            1,
        );
        let current = f32_values(&current_scores, 0, key_rows);
        let packed = f32_values(&packed_scores, 0, key_rows);
        let (overlap, current_margin, fp4_margin) = topk_overlap_and_margin(&current, &packed, 512);
        let overlap_ratio = overlap as f64 / 512.0;
        if overlap_ratio < worst_overlap.0 {
            worst_overlap = (overlap_ratio, layer, input.start);
        }
        for (&a, &b) in current.iter().zip(&packed) {
            let error = (a - b).abs();
            max_abs = max_abs.max(error);
            max_rel = max_rel.max(error / a.abs().max(1.0e-6));
        }
        let mut layer_head_overlaps = Vec::with_capacity(HEADS);
        let mut layer_current_margins = Vec::with_capacity(HEADS);
        let mut layer_fp4_margins = Vec::with_capacity(HEADS);
        let mut scalar_reference_scores = vec![0.0f32; key_rows];
        for head in 0..HEADS {
            let mut current_head = vec![0.0f32; key_rows];
            let mut fp4_head = vec![0.0f32; key_rows];
            let query_quantized = unpack_indexer_fp4_row(&query_rows[head]).unwrap();
            let scaled_weight = scaled_weights[head];
            for row in 0..key_rows {
                let mut current_dot = 0.0f32;
                let mut fp4_dot = 0.0f32;
                for dim in 0..DIM {
                    current_dot += raw_query[head * DIM + dim] * raw_keys[row * DIM + dim];
                    fp4_dot += query_quantized[dim] * quantized_keys[row][dim];
                }
                current_head[row] = current_dot.max(0.0) * scaled_weight;
                fp4_head[row] = fp4_dot.max(0.0) * scaled_weight;
                scalar_reference_scores[row] = scalar_reference_scores[row] + fp4_head[row];
            }
            let (head_overlap, current_head_margin, fp4_head_margin) =
                topk_overlap_and_margin(&current_head, &fp4_head, 512);
            let head_overlap = head_overlap as f64 / 512.0;
            layer_head_overlaps.push(head_overlap);
            layer_current_margins.push(current_head_margin);
            layer_fp4_margins.push(fp4_head_margin);
            global_head_overlaps.push(head_overlap);
            global_current_head_margins.push(current_head_margin);
            global_fp4_head_margins.push(fp4_head_margin);
        }
        let worst_head = layer_head_overlaps
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(head, &overlap)| (head, overlap))
            .unwrap();
        let mut layer_oracle_max_abs = 0.0f32;
        let mut layer_oracle_max_rel = 0.0f32;
        for (&actual, &reference) in packed.iter().zip(&scalar_reference_scores) {
            let error = (actual - reference).abs();
            layer_oracle_max_abs = layer_oracle_max_abs.max(error);
            layer_oracle_max_rel = layer_oracle_max_rel.max(error / reference.abs().max(1.0e-6));
        }
        oracle_max_abs = oracle_max_abs.max(layer_oracle_max_abs);
        oracle_max_rel = oracle_max_rel.max(layer_oracle_max_rel);
        println!(
            "v41_fp4_selection layer={layer} position={} ratio={} keys={} overlap={}/512 current_margin={:.8e} fp4_margin={:.8e} score_oracle_max_abs={layer_oracle_max_abs:.8e} score_oracle_max_rel={layer_oracle_max_rel:.8e} head_min_overlap={:.6} worst_head={} head_mean_overlap={:.6} head_current_margin_median={:.8e} head_fp4_margin_median={:.8e}",
            input.start,
            input.ratio,
            key_rows,
            overlap,
            current_margin,
            fp4_margin,
            worst_head.1,
            worst_head.0,
            layer_head_overlaps.iter().sum::<f64>() / HEADS as f64,
            median(&mut layer_current_margins),
            median(&mut layer_fp4_margins),
        );
    }
    println!(
        "v41_fp4_selection_summary min_overlap={:.6} worst_layer={} worst_position={} max_abs={max_abs:.9e} max_rel={max_rel:.9e} scalar_oracle_max_abs={oracle_max_abs:.9e} scalar_oracle_max_rel={oracle_max_rel:.9e} head_min_overlap={:.6} head_current_margin_median={:.8e} head_fp4_margin_median={:.8e}",
        worst_overlap.0,
        worst_overlap.1,
        worst_overlap.2,
        global_head_overlaps.iter().copied().fold(1.0f64, f64::min),
        median(&mut global_current_head_margins),
        median(&mut global_fp4_head_margins),
    );
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}
