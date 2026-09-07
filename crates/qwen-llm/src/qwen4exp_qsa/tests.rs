use super::*;
use crate::gguf::GgufFile;
use crate::qwen4exp_gdn::GatedDeltaNetMetalWeights;
use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
use half::{bf16, f16};
use objc2_metal::MTLCommandQueue;
use serde_json::Value;
use sha2::{Digest, Sha256};

const QSA_ORACLE_JSON: &str = include_str!("../../tests/fixtures/qwen4exp_qsa_text_f16_v1.json");
const QSA_ORACLE_F32: &[u8] = include_bytes!("../../tests/fixtures/qwen4exp_qsa_text_f16_v1.f32");

struct TestWeights {
    geometry: QwenSparseAttentionMetalGeometry,
    query: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    output: MetalTensor,
    query_norm: MetalTensor,
    key_norm: MetalTensor,
    index_query: MetalTensor,
    index_key: MetalTensor,
    index_query_norm: MetalTensor,
    index_key_norm: MetalTensor,
}

impl TestWeights {
    fn borrowed(&self) -> QwenSparseAttentionMetalWeights<'_> {
        QwenSparseAttentionMetalWeights {
            geometry: self.geometry,
            query: &self.query,
            key: &self.key,
            value: &self.value,
            output: &self.output,
            query_norm: &self.query_norm,
            key_norm: &self.key_norm,
            index_query: &self.index_query,
            index_key: &self.index_key,
            index_query_norm: &self.index_query_norm,
            index_key_norm: &self.index_key_norm,
        }
    }
}

fn context() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
        Err(error) => panic!("Metal initialization failed: {error}"),
    }
}

#[test]
fn decision_capture_rejects_nesting_without_losing_outer_binding() {
    let Some(ctx) = context() else { return };
    let config = Qwen4ExpConfig::flash_next_reference();
    let banks = Qwen4ExpQsaDecisionCaptureBanks::new(&ctx, &config, 2_052, 2_051, 1).unwrap();
    assert!(!qwen4exp_qsa_decision_capture_active());

    let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_qwen4exp_qsa_decision_capture(&banks, || {
            let nested = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_qwen4exp_qsa_decision_capture(&banks, || ());
            }));
            assert!(nested.is_err());
            assert!(qwen4exp_qsa_decision_capture_active());
            panic!("exercise outer QSA capture restoration");
        });
    }));
    assert!(outer.is_err());
    assert!(!qwen4exp_qsa_decision_capture_active());
}

fn test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 64;
    config.hidden_size = 16;
    config.attention.query_heads = 4;
    config.attention.kv_heads = 2;
    config.qsa.token_budget = 8;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn packed_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 128;
    config.hidden_size = 16;
    config.qsa.token_budget = 64;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn selected_attention_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 4_096;
    config.hidden_size = 16;
    config.qsa.token_budget = 2_048;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn selected_motor_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 64;
    config.hidden_size = 16;
    config.qsa.token_budget = 8;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn selected_bf16_motor_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 128;
    config.hidden_size = 16;
    config.qsa.token_budget = 32;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn selected_multi_band_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.context_length = 192;
    config.hidden_size = 16;
    config.qsa.token_budget = 64;
    config.ple = None;
    config.validate().unwrap();
    QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
}

fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let raw = ((index * 37 + seed * 19 + index / 7 * 3 + 5) % 101) as f32 - 50.0;
            raw * scale
        })
        .collect()
}

fn real_mat_vec(weight: &[f32], input: &[f32], n_in: usize) -> Vec<f32> {
    assert_eq!(weight.len() % n_in, 0);
    weight
        .chunks_exact(n_in)
        .map(|row| row.iter().zip(input).map(|(&w, &x)| w * x).sum())
        .collect()
}

fn rmsnorm_heads_position_zero(
    input: &[f32],
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), heads * head_dim);
    assert_eq!(weight.len(), head_dim);
    let mut output = vec![0.0; input.len()];
    for head in 0..heads {
        let start = head * head_dim;
        let row = &input[start..start + head_dim];
        let scale = 1.0
            / (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32 + eps).sqrt();
        for lane in 0..head_dim {
            output[start + lane] = row[lane] * scale * weight[lane];
        }
    }
    output
}

fn weight(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let mut tensor =
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap();
    tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    tensor
}

fn f16_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let bits = values
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&bits), shape, GgmlType::F16).unwrap()
}

fn bf16_weight(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let bits = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let mut tensor =
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&bits), shape, GgmlType::BF16).unwrap();
    tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    tensor
}

struct SelectedAttentionFixture {
    geometry: QwenSparseAttentionMetalGeometry,
    scratch: QwenSparseAttentionPackedScratch,
    index_query_norm: MetalTensor,
    compressed_keys: MetalTensor,
    key_cache: MetalTensor,
    value_cache: MetalTensor,
    compact_gate: MetalTensor,
}

fn selected_attention_fixture(ctx: &MetalContext, query_count: usize) -> SelectedAttentionFixture {
    let geometry = selected_attention_test_geometry(4_096);
    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        ctx,
        geometry,
        query_count,
        true,
    )
    .unwrap();
    let views = scratch.views(query_count).unwrap();
    let selected = scratch.selected.as_ref().unwrap();
    write_f32_tensor(
        &selected.index_query_raw,
        &values(
            geometry.index_query_width() * query_count,
            2_101 + query_count,
            0.002_1,
        ),
    );
    write_f32_tensor(
        &views.query,
        &values(
            geometry.query_width() * query_count,
            2_111 + query_count,
            0.003_3,
        ),
    );
    let projected = values(
        geometry.query_projection_width() * query_count,
        2_123 + query_count,
        0.007_1,
    );
    write_f32_tensor(&views.query_gate_projection, &projected);
    write_f32_tensor(
        &views.attention,
        &values(
            geometry.query_width() * query_count,
            2_129 + query_count,
            100.0,
        ),
    );
    let mut compact_gate = vec![0.0_f32; geometry.query_width() * query_count];
    for query in 0..query_count {
        for head in 0..geometry.query_heads {
            let source = query * geometry.query_projection_width()
                + head * 2 * geometry.head_dim
                + geometry.head_dim;
            let destination = query * geometry.query_width() + head * geometry.head_dim;
            compact_gate[destination..destination + geometry.head_dim]
                .copy_from_slice(&projected[source..source + geometry.head_dim]);
        }
    }
    let index_query_norm = weight(
        ctx,
        &(0..geometry.index_head_dim)
            .map(|lane| 0.69 + (lane % 13) as f32 * 0.017)
            .collect::<Vec<_>>(),
        vec![geometry.index_head_dim as u64],
    );
    let compressed_keys = f16_tensor(
        ctx,
        &values(
            geometry.index_head_dim * geometry.block_capacity(),
            2_137 + query_count,
            0.003_7,
        ),
        vec![
            geometry.index_head_dim as u64,
            geometry.block_capacity() as u64,
        ],
    );
    let key_cache = f16_tensor(
        ctx,
        &values(
            geometry.head_dim * geometry.kv_heads * geometry.capacity,
            2_143 + query_count,
            0.004_1,
        ),
        vec![
            geometry.head_dim as u64,
            geometry.kv_heads as u64,
            geometry.capacity as u64,
        ],
    );
    let value_cache = f16_tensor(
        ctx,
        &values(
            geometry.head_dim * geometry.kv_heads * geometry.capacity,
            2_147 + query_count,
            0.004_3,
        ),
        vec![
            geometry.head_dim as u64,
            geometry.kv_heads as u64,
            geometry.capacity as u64,
        ],
    );
    let compact_gate = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&compact_gate),
        vec![geometry.query_width() as u64, query_count as u64],
        GgmlType::F32,
    )
    .unwrap();
    SelectedAttentionFixture {
        geometry,
        scratch,
        index_query_norm,
        compressed_keys,
        key_cache,
        value_cache,
        compact_gate,
    }
}

fn test_weights(ctx: &MetalContext, geometry: QwenSparseAttentionMetalGeometry) -> TestWeights {
    let g = geometry;
    let query_cpu = values(g.hidden_size * g.query_projection_width(), 1, 0.0013);
    let key_cpu = values(g.hidden_size * g.kv_width(), 2, 0.0011);
    let value_cpu = values(g.hidden_size * g.kv_width(), 3, 0.0015);
    let output_cpu = values(g.query_width() * g.hidden_size, 4, 0.0009);
    let index_query_cpu = values(g.hidden_size * g.index_query_width(), 5, 0.0017);
    let index_key_cpu = values(g.hidden_size * g.index_head_dim, 6, 0.0019);
    let query_norm_cpu = (0..g.head_dim)
        .map(|lane| 0.75 + (lane % 13) as f32 * 0.025)
        .collect::<Vec<_>>();
    let key_norm_cpu = (0..g.head_dim)
        .map(|lane| 0.8 + (lane % 11) as f32 * 0.021)
        .collect::<Vec<_>>();
    let index_query_norm_cpu = (0..g.index_head_dim)
        .map(|lane| 0.7 + (lane % 9) as f32 * 0.031)
        .collect::<Vec<_>>();
    let index_key_norm_cpu = (0..g.index_head_dim)
        .map(|lane| 0.78 + (lane % 7) as f32 * 0.027)
        .collect::<Vec<_>>();
    TestWeights {
        geometry,
        query: weight(
            ctx,
            &query_cpu,
            vec![g.hidden_size as u64, g.query_projection_width() as u64],
        ),
        key: weight(
            ctx,
            &key_cpu,
            vec![g.hidden_size as u64, g.kv_width() as u64],
        ),
        value: weight(
            ctx,
            &value_cpu,
            vec![g.hidden_size as u64, g.kv_width() as u64],
        ),
        output: weight(
            ctx,
            &output_cpu,
            vec![g.query_width() as u64, g.hidden_size as u64],
        ),
        query_norm: weight(ctx, &query_norm_cpu, vec![g.head_dim as u64]),
        key_norm: weight(ctx, &key_norm_cpu, vec![g.head_dim as u64]),
        index_query: weight(
            ctx,
            &index_query_cpu,
            vec![g.hidden_size as u64, g.index_query_width() as u64],
        ),
        index_key: weight(
            ctx,
            &index_key_cpu,
            vec![g.hidden_size as u64, g.index_head_dim as u64],
        ),
        index_query_norm: weight(ctx, &index_query_norm_cpu, vec![g.index_head_dim as u64]),
        index_key_norm: weight(ctx, &index_key_norm_cpu, vec![g.index_head_dim as u64]),
    }
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize).to_vec()
    }
}

fn read_tensor_bytes(tensor: &MetalTensor) -> Vec<u8> {
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::slice::from_raw_parts(source, tensor.n_bytes() as usize).to_vec()
    }
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize).to_vec()
    }
}

fn read_f16(tensor: &MetalTensor) -> Vec<f32> {
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize)
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect()
    }
}

fn write_f32_tensor(tensor: &MetalTensor, values: &[f32]) {
    assert!(tensor.is_writable());
    assert_eq!(tensor.dtype, GgmlType::F32);
    assert_eq!(tensor.n_elements() as usize, values.len());
    let offset = tensor.offset as usize;
    let bytes = std::mem::size_of_val(values);
    let end = offset
        .checked_add(bytes)
        .expect("F32 test write range overflow");
    assert!(end <= tensor.buffer.length());
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<f32>(),
            values.len(),
        );
    }
}

fn write_i32_tensor(tensor: &MetalTensor, values: &[i32]) {
    assert!(tensor.is_writable());
    assert_eq!(tensor.dtype, GgmlType::I32);
    assert_eq!(tensor.n_elements() as usize, values.len());
    let offset = tensor.offset as usize;
    let bytes = std::mem::size_of_val(values);
    let end = offset
        .checked_add(bytes)
        .expect("I32 test write range overflow");
    assert!(end <= tensor.buffer.length());
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<i32>(),
            values.len(),
        );
    }
}

fn write_f16_prefix(tensor: &MetalTensor, values: &[f32]) {
    assert!(tensor.is_writable());
    assert_eq!(tensor.dtype, GgmlType::F16);
    assert!(values.len() <= tensor.n_elements() as usize);
    let offset = tensor.offset as usize;
    let bytes = values
        .len()
        .checked_mul(size_of::<u16>())
        .expect("F16 test write byte count overflow");
    let end = offset
        .checked_add(bytes)
        .expect("F16 test write range overflow");
    assert!(end <= tensor.buffer.length());
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(offset)
            .cast::<u16>();
        for (index, &value) in values.iter().enumerate() {
            destination.add(index).write(f16::from_f32(value).to_bits());
        }
    }
}

fn assert_dispatch_shape(
    census: &[crate::metal::DispatchCensusRow],
    tag: &str,
    kernel: &str,
    grid: [u64; 3],
    threads: [u64; 3],
) {
    let matches = census
        .iter()
        .filter(|row| row.tag.as_deref() == Some(tag) && row.kernel == kernel)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "{tag} {kernel}");
    let row = matches[0];
    assert_eq!(
        [row.grid_width, row.grid_height, row.grid_depth],
        grid,
        "{tag} {kernel} grid"
    );
    assert_eq!(
        [row.threads_width, row.threads_height, row.threads_depth,],
        threads,
        "{tag} {kernel} threads"
    );
    assert_eq!(row.grid_tgs, grid.iter().product::<u64>());
    assert_eq!(row.tg_threads, threads.iter().product::<u64>());
}

fn assert_qsa_rejected_without_dispatch(
    ctx: &MetalContext,
    label: &str,
    encode: impl FnOnce(&KernelEncoder) -> Result<(), Qwen4ExpQsaError>,
) {
    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let error = match encode(&encoder) {
        Ok(()) => panic!("{label} unexpectedly passed"),
        Err(error) => error,
    };
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    assert!(census.is_empty(), "{label} dispatched before {error}");
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: expected {expected}, got {actual}, tolerance={tolerance}"
        );
    }
}

fn assert_similarity(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    maximum_relative_rms: f64,
    minimum_cosine: f64,
    maximum_absolute: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(actual.iter().all(|value| value.is_finite()), "{label}");
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| *actual as f64 * *expected as f64)
        .sum::<f64>();
    let actual_square = actual
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let expected_square = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let difference_square = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual as f64 - *expected as f64).powi(2))
        .sum::<f64>();
    let relative_rms = (difference_square / expected_square.max(1e-30)).sqrt();
    let cosine = dot / (actual_square * expected_square).sqrt().max(1e-30);
    let observed_max = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "[{label}] relative_rms={relative_rms:.3e} cosine={cosine:.9} max_abs={observed_max:.3e}"
    );
    assert!(
        relative_rms <= maximum_relative_rms,
        "{label} relative_rms={relative_rms}"
    );
    assert!(cosine >= minimum_cosine, "{label} cosine={cosine}");
    assert!(
        observed_max <= maximum_absolute,
        "{label} max_abs={observed_max}"
    );
}

#[derive(Clone)]
struct DenseQsaStateSnapshot {
    committed_length: usize,
    pending_index_keys: Vec<f32>,
    compressed_index_keys: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    selected_count: i32,
}

struct DenseQsaSerialTrace {
    outputs: Vec<f32>,
    states: Vec<DenseQsaStateSnapshot>,
}

fn serial_dense_trace(
    ctx: &MetalContext,
    weights: &TestWeights,
    inputs: &[f32],
    tokens: usize,
) -> DenseQsaSerialTrace {
    let g = weights.geometry;
    assert_eq!(inputs.len(), g.hidden_size * tokens);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(ctx, g).unwrap();
    let mut outputs = Vec::with_capacity(g.hidden_size * tokens);
    let mut states = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let start = token * g.hidden_size;
        outputs.extend(encode_one(
            ctx,
            weights,
            &mut workspace,
            &inputs[start..start + g.hidden_size],
        ));
        let sequence_length = token + 1;
        let completed_index_elements = sequence_length / g.ratio * g.index_head_dim;
        let cache_elements = sequence_length * g.kv_width();
        states.push(DenseQsaStateSnapshot {
            committed_length: sequence_length,
            pending_index_keys: read_f32(&workspace.pending_index_keys),
            compressed_index_keys: read_f16(&workspace.compressed_index_keys)
                [..completed_index_elements]
                .to_vec(),
            key_cache: read_f16(&workspace.key_cache)[..cache_elements].to_vec(),
            value_cache: read_f16(&workspace.value_cache)[..cache_elements].to_vec(),
            selected_count: read_i32_scalar(&workspace.selected_count).unwrap(),
        });
    }
    DenseQsaSerialTrace { outputs, states }
}

fn encode_dense_packed_chunk(
    ctx: &MetalContext,
    weights: &TestWeights,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    inputs: &[f32],
    start_position: usize,
    tokens: usize,
) -> (Vec<f32>, Vec<crate::metal::DispatchCensusRow>) {
    let g = weights.geometry;
    assert_eq!(inputs.len(), g.hidden_size * tokens);
    let input = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(inputs),
        vec![g.hidden_size as u64, tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    let output = unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor(
            ctx,
            &encoder,
            &input,
            weights.borrowed(),
            workspace,
            scratch,
            start_position,
            tokens,
        )
    }
    .unwrap();
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    workspace.release_after().unwrap();
    (read_f32(&output), census)
}

fn encode_selected_packed_chunk(
    ctx: &MetalContext,
    weights: &TestWeights,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    inputs: &[f32],
    start_position: usize,
    tokens: usize,
) -> (Vec<f32>, Vec<crate::metal::DispatchCensusRow>) {
    let g = weights.geometry;
    assert_eq!(inputs.len(), g.hidden_size * tokens);
    let input = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(inputs),
        vec![g.hidden_size as u64, tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    let output = unsafe {
        encode_qwen_sparse_attention_text_packed_motor(
            ctx,
            &encoder,
            &input,
            weights.borrowed(),
            workspace,
            scratch,
            start_position,
            tokens,
        )
    }
    .unwrap();
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    workspace.release_after().unwrap();
    (read_f32(&output), census)
}

fn assert_dense_state_matches(
    label: &str,
    workspace: &QwenSparseAttentionMetalWorkspace,
    expected: &DenseQsaStateSnapshot,
) {
    let g = workspace.geometry;
    assert_eq!(workspace.committed_length(), expected.committed_length);
    assert_eq!(
        read_i32_scalar(&workspace.selected_count).unwrap(),
        expected.selected_count
    );
    assert_close(
        &read_f32(&workspace.pending_index_keys),
        &expected.pending_index_keys,
        1e-5,
        1e-4,
    );
    let completed_index_elements = expected.compressed_index_keys.len();
    assert_close(
        &read_f16(&workspace.compressed_index_keys)[..completed_index_elements],
        &expected.compressed_index_keys,
        5e-4,
        0.0,
    );
    let cache_elements = expected.committed_length * g.kv_width();
    assert_eq!(
        cache_elements,
        expected.key_cache.len(),
        "{label} key cache"
    );
    assert_close(
        &read_f16(&workspace.key_cache)[..cache_elements],
        &expected.key_cache,
        1e-3,
        0.0,
    );
    assert_close(
        &read_f16(&workspace.value_cache)[..cache_elements],
        &expected.value_cache,
        1e-3,
        0.0,
    );
}

fn oracle_values() -> Vec<f32> {
    assert!(QSA_ORACLE_F32.len().is_multiple_of(4));
    QSA_ORACLE_F32
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn oracle_section<'a>(fixture: &Value, values: &'a [f32], name: &str) -> &'a [f32] {
    let section = &fixture["binary"]["sections"][name];
    let offset = section["offset_f32"].as_u64().unwrap() as usize;
    let count = section["count_f32"].as_u64().unwrap() as usize;
    let shape_count = section["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|dimension| dimension.as_u64().unwrap() as usize)
        .product::<usize>();
    assert_eq!(count, shape_count, "bad oracle section shape for {name}");
    &values[offset..offset + count]
}

fn formula_from_json(formula: &Value, rows: usize, columns: usize) -> Vec<f32> {
    let multipliers = formula["multipliers"].as_array().unwrap();
    let m0 = multipliers[0].as_i64().unwrap();
    let m1 = multipliers[1].as_i64().unwrap();
    let add = formula["add"].as_i64().unwrap();
    let modulus = formula["modulus"].as_i64().unwrap();
    let center = formula["center"].as_i64().unwrap();
    let scale = formula["scale"].as_f64().unwrap() as f32;
    let mut output = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let raw = (row as i64 * m0 + column as i64 * m1 + add).rem_euclid(modulus) - center;
            output.push(raw as f32 * scale);
        }
    }
    output
}

fn norm_from_json(recipe: &Value, width: usize) -> Vec<f32> {
    let base = recipe["base"].as_f64().unwrap() as f32;
    let step = recipe["step"].as_f64().unwrap() as f32;
    let modulus = recipe["modulus"].as_u64().unwrap() as usize;
    (0..width)
        .map(|lane| base + (lane % modulus) as f32 * step)
        .collect()
}

fn oracle_weights(
    ctx: &MetalContext,
    geometry: QwenSparseAttentionMetalGeometry,
    fixture: &Value,
) -> TestWeights {
    let g = geometry;
    let formulas = &fixture["recipe"]["weights_and_inputs"];
    let norms = &fixture["recipe"]["norms"];
    let query = formula_from_json(
        &formulas["query_gate"],
        g.query_projection_width(),
        g.hidden_size,
    );
    let key = formula_from_json(&formulas["key"], g.kv_width(), g.hidden_size);
    let value = formula_from_json(&formulas["value"], g.kv_width(), g.hidden_size);
    let output = formula_from_json(&formulas["output"], g.hidden_size, g.query_width());
    let index_query = formula_from_json(
        &formulas["index_query"],
        g.index_query_width(),
        g.hidden_size,
    );
    let index_key = formula_from_json(&formulas["index_key"], g.index_head_dim, g.hidden_size);
    let query_norm = norm_from_json(&norms["query"], g.head_dim);
    let key_norm = norm_from_json(&norms["key"], g.head_dim);
    let index_query_norm = norm_from_json(&norms["index_query"], g.index_head_dim);
    let index_key_norm = norm_from_json(&norms["index_key"], g.index_head_dim);
    TestWeights {
        geometry,
        query: weight(
            ctx,
            &query,
            vec![g.hidden_size as u64, g.query_projection_width() as u64],
        ),
        key: weight(ctx, &key, vec![g.hidden_size as u64, g.kv_width() as u64]),
        value: weight(ctx, &value, vec![g.hidden_size as u64, g.kv_width() as u64]),
        output: weight(
            ctx,
            &output,
            vec![g.query_width() as u64, g.hidden_size as u64],
        ),
        query_norm: weight(ctx, &query_norm, vec![g.head_dim as u64]),
        key_norm: weight(ctx, &key_norm, vec![g.head_dim as u64]),
        index_query: weight(
            ctx,
            &index_query,
            vec![g.hidden_size as u64, g.index_query_width() as u64],
        ),
        index_key: weight(
            ctx,
            &index_key,
            vec![g.hidden_size as u64, g.index_head_dim as u64],
        ),
        index_query_norm: weight(ctx, &index_query_norm, vec![g.index_head_dim as u64]),
        index_key_norm: weight(ctx, &index_key_norm, vec![g.index_head_dim as u64]),
    }
}

fn assert_source_identity(
    fixture: &Value,
    name: &str,
    revision: &str,
    tree: &str,
    files: &[(&str, &str)],
) {
    let source = &fixture["sources"][name];
    assert_eq!(source["revision"], revision);
    assert_eq!(source["tree"], tree);
    let actual = source["files"].as_array().unwrap();
    assert_eq!(actual.len(), files.len());
    for &(path, digest) in files {
        let entry = actual
            .iter()
            .find(|entry| entry["path"] == path)
            .unwrap_or_else(|| panic!("missing pinned source {path}"));
        assert_eq!(entry["sha256"], digest);
    }
}

fn encode_one(
    ctx: &MetalContext,
    weights: &TestWeights,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    input_values: &[f32],
) -> Vec<f32> {
    let input = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(input_values),
        vec![weights.geometry.hidden_size as u64],
        GgmlType::F32,
    )
    .unwrap();
    let copied = MetalTensor::zeros_f32(ctx, vec![weights.geometry.hidden_size as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let read =
        encode_qwen_sparse_attention_text(ctx, &encoder, &input, weights.borrowed(), workspace)
            .unwrap();
    read.output()
        .encode_copy_to(ctx, &encoder, &copied)
        .unwrap();
    encoder.end();
    command.commit();
    drop(read);
    workspace.release_after().unwrap();
    read_f32(&copied)
}

#[test]
fn packed_range_plan_separates_dense_shoulder_and_selected_suffix() {
    let geometry = packed_test_geometry(128);
    assert_eq!(geometry.output_width(), 67);
    for (start, tokens, expected) in [
        (0, 64, (64, 64, 64, 0, 0)),
        (64, 1, (65, 1, 1, 0, 0)),
        (64, 3, (67, 3, 3, 0, 0)),
        (64, 4, (68, 3, 3, 1, 1)),
        (65, 2, (67, 2, 2, 0, 0)),
        (66, 2, (68, 1, 1, 1, 1)),
        (67, 33, (100, 0, 0, 33, 2)),
        (68, 2, (70, 0, 0, 2, 1)),
        (120, 8, (128, 0, 0, 8, 1)),
    ] {
        let plan = geometry.plan_packed_range(start, tokens).unwrap();
        assert_eq!(
            (
                plan.end_position,
                plan.dense_tokens,
                plan.selected_offset,
                plan.selected_tokens,
                plan.selected_bands,
            ),
            expected,
            "start={start} tokens={tokens}"
        );
    }
    assert!(geometry.plan_packed_range(0, 0).is_err());
    assert!(geometry.plan_packed_range(0, 65).is_err());
    assert!(geometry.plan_packed_range(120, 9).is_err());
    assert!(geometry.plan_packed_range(usize::MAX, 1).is_err());

    let production = QwenSparseAttentionMetalGeometry::from_config(
        &Qwen4ExpConfig::flash_next_reference(),
        3,
        4_096,
    )
    .unwrap();
    assert_eq!(
        production.plan_packed_range(2_048, 3).unwrap(),
        QwenSparseAttentionPackedRangePlan {
            end_position: 2_051,
            dense_tokens: 3,
            selected_offset: 3,
            selected_tokens: 0,
            selected_bands: 0,
        }
    );
    assert_eq!(
        production.plan_packed_range(2_048, 4).unwrap(),
        QwenSparseAttentionPackedRangePlan {
            end_position: 2_052,
            dense_tokens: 3,
            selected_offset: 3,
            selected_tokens: 1,
            selected_bands: 1,
        }
    );
    let mixed_n2 = production.plan_packed_range(2_050, 2).unwrap();
    let selected_n2 = production.plan_packed_range(2_051, 2).unwrap();
    assert!(requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        mixed_n2,
        2,
    ));
    assert!(requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        selected_n2,
        2,
    ));
    assert!(requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        production.plan_packed_range(2_051, 3).unwrap(),
        3,
    ));
    assert!(requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        production.plan_packed_range(2_051, 4).unwrap(),
        4,
    ));
    assert!(!requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        production.plan_packed_range(2_051, 5).unwrap(),
        5,
    ));
    assert!(!requires_exact_selected_q8_output(
        production,
        GgmlType::Q8_0,
        production.plan_packed_range(2_049, 2).unwrap(),
        2,
    ));
    assert!(!requires_exact_selected_q8_output(
        production,
        GgmlType::F32,
        selected_n2,
        2,
    ));
    let allocations = production.packed_scratch_logical_allocations(18).unwrap();
    let score_bytes = allocations[6];
    assert_eq!(score_bytes, 2_051 * 24 * 18 * size_of::<f32>());
    assert_eq!(score_bytes - 2_048 * 24 * 18 * size_of::<f32>(), 5_184);

    let maximum = QwenSparseAttentionMetalGeometry::from_config(
        &Qwen4ExpConfig::flash_next_reference(),
        3,
        262_144,
    )
    .unwrap();
    let selected = maximum
        .selected_packed_scratch_logical_allocations(2_048)
        .unwrap();
    assert_eq!(selected.len(), 9);
    assert_eq!(selected.iter().sum::<usize>(), 23_406_336);
}

#[test]
fn selected_packed_qsa_scratch_is_explicit_and_optional() {
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(16);
    let dense = QwenSparseAttentionPackedScratch::new(&ctx, geometry, 8).unwrap();
    assert!(!dense.selected_capable());
    let selected =
        QwenSparseAttentionPackedScratch::new_with_selected_capability(&ctx, geometry, 8, true)
            .unwrap();
    assert!(selected.selected_capable());
    selected
        .selected
        .as_ref()
        .unwrap()
        .validate(geometry, 8, 8)
        .unwrap();
}

#[test]
fn selected_packed_qsa_band_views_advance_only_raw_queries() {
    const CAPACITY: usize = 64;
    let Some(ctx) = context() else { return };
    let geometry = selected_multi_band_test_geometry(160);
    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, CAPACITY, true,
    )
    .unwrap();
    let selected = scratch.selected.as_ref().unwrap();
    let raw = values(geometry.index_query_width() * CAPACITY, 2_299, 0.002_1);
    write_f32_tensor(&selected.index_query_raw, &raw);
    let first = selected.band_views(geometry, CAPACITY, 32, 0, 32).unwrap();
    let second = selected.band_views(geometry, CAPACITY, 32, 32, 32).unwrap();
    assert_eq!(
        read_f32(&second.index_query_raw)[0],
        raw[32 * geometry.index_query_width()]
    );
    assert_eq!(
        second.index_query_raw.offset,
        first.index_query_raw.offset
            + (32 * geometry.index_query_width() * size_of::<f32>()) as u64
    );
    for (first, second) in [
        (&first.index_query, &second.index_query),
        (&first.scores, &second.scores),
        (&first.visible_blocks, &second.visible_blocks),
        (&first.selected_blocks, &second.selected_blocks),
        (&first.selected_count, &second.selected_count),
        (&first.selector_status, &second.selector_status),
        (&first.token_ids, &second.token_ids),
        (&first.attention_logits, &second.attention_logits),
    ] {
        assert_eq!(first.offset, second.offset);
    }
    assert_eq!(
        selected
            .raw_query_projection_view(geometry, CAPACITY, 32, CAPACITY)
            .unwrap()
            .shape,
        [geometry.index_query_width() as u64, CAPACITY as u64]
    );
    assert!(selected.band_views(geometry, CAPACITY, 32, 63, 2).is_err());
}

#[test]
fn selected_index_primitives_match_repeated_scalar_kernels() {
    const CAPACITY: usize = 8;
    const QUERIES: usize = 5;
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(40);
    let start_position = 31;
    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, CAPACITY, true,
    )
    .unwrap();
    let selected = scratch.selected.as_ref().unwrap();
    let raw_queries = values(geometry.index_query_width() * CAPACITY, 1_811, 0.002_4);
    write_f32_tensor(&selected.index_query_raw, &raw_queries);
    let norm_values = (0..geometry.index_head_dim)
        .map(|lane| 0.71 + (lane % 11) as f32 * 0.019)
        .collect::<Vec<_>>();
    let norm_weight = weight(&ctx, &norm_values, vec![geometry.index_head_dim as u64]);
    let key_values = values(
        geometry.index_head_dim * geometry.block_capacity(),
        1_823,
        0.003_1,
    );
    let key_bits = key_values
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let compressed_keys = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&key_bits),
        vec![
            geometry.index_head_dim as u64,
            geometry.block_capacity() as u64,
        ],
        GgmlType::F16,
    )
    .unwrap();

    let reference_queries = MetalTensor::zeros_f32(
        &ctx,
        vec![
            geometry.index_head_dim as u64,
            geometry.index_query_heads as u64,
            QUERIES as u64,
        ],
    )
    .unwrap();
    let reference_scores =
        MetalTensor::zeros_f32(&ctx, vec![geometry.block_capacity() as u64, QUERIES as u64])
            .unwrap();
    let reference_visible = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let reference_blocks =
        MetalTensor::zeros_i32(&ctx, vec![geometry.block_budget() as u64, QUERIES as u64]).unwrap();
    let reference_counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let reference_status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let reference_ids =
        MetalTensor::zeros_i32(&ctx, vec![geometry.output_width() as u64, QUERIES as u64]).unwrap();
    let visible = (0..QUERIES)
        .map(|query| ((start_position + query + 1) / geometry.ratio) as i32)
        .collect::<Vec<_>>();
    assert_eq!(visible, [8, 8, 8, 8, 9]);
    write_i32_tensor(&reference_visible, &visible);

    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let packed_tag =
        crate::metal::dispatch_census_tag_scope(|| "qwen4exp.qsa.selected_index.packet".into());
    let actual = encode_selected_index_primitives(
        &ctx,
        &encoder,
        &norm_weight,
        &compressed_keys,
        &scratch,
        start_position,
        0,
        QUERIES,
    )
    .unwrap();
    drop(packed_tag);
    for query in 0..QUERIES {
        let query_offset = query * geometry.index_query_width();
        let raw_query = selected.index_query_raw.view_subrange(
            query_offset as u64,
            vec![
                geometry.index_head_dim as u64,
                geometry.index_query_heads as u64,
            ],
        );
        let normalized_query = reference_queries.view_subrange(
            query_offset as u64,
            vec![
                geometry.index_head_dim as u64,
                geometry.index_query_heads as u64,
            ],
        );
        encode_norm_rope(
            &ctx,
            &encoder,
            &raw_query,
            &norm_weight,
            &normalized_query,
            geometry.index_query_heads,
            geometry.index_head_dim,
            geometry.rotary_dim,
            start_position + query,
            geometry.theta,
            geometry.eps,
        )
        .unwrap();
        let score_offset = query * geometry.block_capacity();
        let score = reference_scores.view_subrange(
            score_offset as u64,
            vec![geometry.block_capacity() as u64, 1],
        );
        encode_index_scores_tensors(
            &ctx,
            &encoder,
            &normalized_query,
            &compressed_keys,
            &score,
            visible[query] as usize,
        )
        .unwrap();
        let visible_view = reference_visible.view_subrange(query as u64, vec![1]);
        let block_offset = query * geometry.block_budget();
        let blocks = reference_blocks
            .view_subrange(block_offset as u64, vec![geometry.block_budget() as u64, 1]);
        let count = reference_counts.view_subrange(query as u64, vec![1]);
        let status = reference_status.view_subrange(query as u64, vec![1]);
        encode_select_blocks_tensors(
            &ctx,
            &encoder,
            &score,
            &visible_view,
            &blocks,
            &count,
            &status,
            geometry.block_capacity(),
            geometry.block_budget(),
            1,
        )
        .unwrap();
        let id_offset = query * geometry.output_width();
        let ids =
            reference_ids.view_subrange(id_offset as u64, vec![geometry.output_width() as u64]);
        encode_expand_ids_tensors(
            &ctx,
            &encoder,
            &blocks,
            &ids,
            geometry,
            visible[query] as usize,
            start_position + query + 1,
        )
        .unwrap();
    }
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());

    assert_eq!(
        read_tensor_bytes(&actual.index_query),
        read_tensor_bytes(&reference_queries)
    );
    assert_eq!(
        read_tensor_bytes(&actual.scores),
        read_tensor_bytes(&reference_scores)
    );
    assert_eq!(read_i32(&actual.visible_blocks), visible);
    assert_eq!(
        read_i32(&actual.selected_blocks),
        read_i32(&reference_blocks)
    );
    assert_eq!(read_i32(&actual.selected_count), vec![2; QUERIES]);
    assert_eq!(
        read_i32(&actual.selected_count),
        read_i32(&reference_counts)
    );
    assert_eq!(read_i32(&actual.selector_status), vec![0; QUERIES]);
    assert_eq!(
        read_i32(&actual.selector_status),
        read_i32(&reference_status)
    );
    assert_eq!(read_i32(&actual.token_ids), read_i32(&reference_ids));

    let packed_names = census
        .iter()
        .filter(|row| row.tag.as_deref() == Some("qwen4exp.qsa.selected_index.packet"))
        .map(|row| row.kernel.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        packed_names,
        [
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
        ]
    );
}

#[test]
fn selected_index_selector_ties_and_failures_expand_deterministically() {
    const QUERIES: usize = 2;
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(20);
    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, QUERIES, true,
    )
    .unwrap();
    let views = scratch
        .selected
        .as_ref()
        .unwrap()
        .views(geometry, QUERIES, QUERIES, QUERIES)
        .unwrap();
    write_f32_tensor(
        &views.scores,
        &[1.0, 1.0, 1.0, 1.0, -99.0, f32::NAN, 3.0, 2.0, 1.0, -99.0],
    );
    write_i32_tensor(&views.visible_blocks, &[4, 4]);

    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let packet_tag = crate::metal::dispatch_census_tag_scope(|| {
        "qwen4exp.qsa.selected_index.selector_faults".into()
    });
    encode_select_blocks_tensors(
        &ctx,
        &encoder,
        &views.scores,
        &views.visible_blocks,
        &views.selected_blocks,
        &views.selected_count,
        &views.selector_status,
        geometry.block_capacity(),
        geometry.block_budget(),
        QUERIES,
    )
    .unwrap();
    encode_expand_ids_packed(
        &ctx,
        &encoder,
        &views.visible_blocks,
        &views.selected_blocks,
        &views.selected_count,
        &views.selector_status,
        &views.token_ids,
        15,
        QUERIES,
        geometry.block_budget(),
        geometry.ratio,
        geometry.output_width(),
    )
    .unwrap();
    drop(packet_tag);
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());

    assert_eq!(read_i32(&views.selected_blocks), [0, 1, 0, 1]);
    assert_eq!(read_i32(&views.selected_count), [2, 2]);
    assert_eq!(read_i32(&views.selector_status), [0, 2]);
    let mut expected_ids = vec![0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1];
    expected_ids.extend(std::iter::repeat_n(-1, geometry.output_width()));
    assert_eq!(read_i32(&views.token_ids), expected_ids);
    assert_eq!(
        census
            .iter()
            .filter(|row| {
                row.tag.as_deref() == Some("qwen4exp.qsa.selected_index.selector_faults")
            })
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>(),
        [
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
        ]
    );
}

#[test]
fn selected_attention_logits_match_repeated_scalar_kernels() {
    const QUERIES: usize = 5;
    let Some(ctx) = context() else { return };
    let fixture = selected_attention_fixture(&ctx, QUERIES);
    let g = fixture.geometry;
    let start_position = g.output_width();
    let packed = fixture.scratch.views(QUERIES).unwrap();
    let reference_logits = MetalTensor::zeros_f32(
        &ctx,
        vec![
            g.output_width() as u64,
            g.query_heads as u64,
            QUERIES as u64,
        ],
    )
    .unwrap();
    write_f32_tensor(
        &reference_logits,
        &vec![f32::NEG_INFINITY; g.output_width() * g.query_heads * QUERIES],
    );

    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let packet_tag = crate::metal::dispatch_census_tag_scope(|| {
        "qwen4exp.qsa.selected_attention.logits_packet".into()
    });
    let selected = encode_selected_index_primitives(
        &ctx,
        &encoder,
        &fixture.index_query_norm,
        &fixture.compressed_keys,
        &fixture.scratch,
        start_position,
        0,
        QUERIES,
    )
    .unwrap();
    encode_attention_logits_packed(
        &ctx,
        &encoder,
        &packed.query,
        &fixture.key_cache,
        &selected.token_ids,
        &selected.selected_count,
        &selected.selector_status,
        &selected.attention_logits,
        g,
        start_position,
        QUERIES,
    )
    .unwrap();
    drop(packet_tag);
    for query in 0..QUERIES {
        let id_count = g.block_budget() * g.ratio + (start_position + query + 1) % g.ratio;
        let query_view = packed.query.view_subrange(
            (query * g.query_width()) as u64,
            vec![g.query_width() as u64],
        );
        let ids = selected.token_ids.view_subrange(
            (query * g.output_width()) as u64,
            vec![g.output_width() as u64],
        );
        let logits = reference_logits.view_subrange(
            (query * g.query_heads * g.output_width()) as u64,
            vec![g.output_width() as u64, g.query_heads as u64],
        );
        encode_attention_logits_tensors(
            &ctx,
            &encoder,
            &query_view,
            &fixture.key_cache,
            &ids,
            &logits,
            g,
            id_count,
        )
        .unwrap();
    }
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());

    assert_eq!(
        read_i32(&selected.selected_count),
        vec![g.block_budget() as i32; QUERIES]
    );
    assert_eq!(read_i32(&selected.selector_status), vec![0; QUERIES]);
    assert_eq!(
        read_tensor_bytes(&selected.attention_logits),
        read_tensor_bytes(&reference_logits)
    );
    assert_eq!(
        census
            .iter()
            .filter(|row| {
                row.tag.as_deref() == Some("qwen4exp.qsa.selected_attention.logits_packet")
            })
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>(),
        [
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
        ]
    );
    assert_dispatch_shape(
        &census,
        "qwen4exp.qsa.selected_attention.logits_packet",
        "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
        [
            ((g.query_heads / g.kv_heads / PACKED_ATTENTION_HEADS_PER_TG)
                * g.output_width().div_ceil(32)) as u64,
            g.kv_heads as u64,
            QUERIES as u64,
        ],
        [PACKED_ATTENTION_THREADS as u64, 1, 1],
    );
}

#[test]
fn selected_attention_packet_matches_repeated_scalar_kernels() {
    let Some(ctx) = context() else { return };
    for (query_count, repetitions) in [(1, 1), (32, 3)] {
        let fixture = selected_attention_fixture(&ctx, query_count);
        let g = fixture.geometry;
        let start_position = g.output_width();
        let packed = fixture.scratch.views(query_count).unwrap();
        for _ in 0..repetitions {
            let reference_logits = MetalTensor::zeros_f32(
                &ctx,
                vec![
                    g.output_width() as u64,
                    g.query_heads as u64,
                    query_count as u64,
                ],
            )
            .unwrap();
            write_f32_tensor(
                &reference_logits,
                &vec![f32::NEG_INFINITY; g.output_width() * g.query_heads * query_count],
            );
            let reference_attention =
                MetalTensor::zeros_f32(&ctx, vec![g.query_width() as u64, query_count as u64])
                    .unwrap();

            crate::metal::dispatch_census_begin();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let packet_tag = crate::metal::dispatch_census_tag_scope(|| {
                "qwen4exp.qsa.selected_attention.packet".into()
            });
            let actual = encode_selected_attention_packet(
                &ctx,
                &encoder,
                &fixture.index_query_norm,
                &fixture.compressed_keys,
                &packed.query,
                &packed.query_gate_projection,
                &fixture.key_cache,
                &fixture.value_cache,
                &packed.attention,
                &fixture.scratch,
                start_position,
                0,
                query_count,
            )
            .unwrap();
            drop(packet_tag);
            for query in 0..query_count {
                let id_count = g.block_budget() * g.ratio + (start_position + query + 1) % g.ratio;
                let query_view = packed.query.view_subrange(
                    (query * g.query_width()) as u64,
                    vec![g.query_width() as u64],
                );
                let gate = fixture.compact_gate.view_subrange(
                    (query * g.query_width()) as u64,
                    vec![g.query_width() as u64],
                );
                let ids = actual.token_ids.view_subrange(
                    (query * g.output_width()) as u64,
                    vec![g.output_width() as u64],
                );
                let logits = reference_logits.view_subrange(
                    (query * g.query_heads * g.output_width()) as u64,
                    vec![g.output_width() as u64, g.query_heads as u64],
                );
                let attention = reference_attention.view_subrange(
                    (query * g.query_width()) as u64,
                    vec![g.query_width() as u64],
                );
                encode_attention_logits_tensors(
                    &ctx,
                    &encoder,
                    &query_view,
                    &fixture.key_cache,
                    &ids,
                    &logits,
                    g,
                    id_count,
                )
                .unwrap();
                encode_attention_softmax_value_tensors(
                    &ctx,
                    &encoder,
                    &gate,
                    &fixture.value_cache,
                    &ids,
                    &logits,
                    &attention,
                    g,
                    id_count,
                )
                .unwrap();
            }
            let census = crate::metal::dispatch_census_take();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());

            assert_eq!(
                read_i32(&actual.selected_count),
                vec![g.block_budget() as i32; query_count]
            );
            assert_eq!(read_i32(&actual.selector_status), vec![0; query_count]);
            assert_eq!(
                read_tensor_bytes(&actual.attention_logits),
                read_tensor_bytes(&reference_logits)
            );
            assert_eq!(
                read_tensor_bytes(&packed.attention),
                read_tensor_bytes(&reference_attention)
            );
            assert_eq!(
                census
                    .iter()
                    .filter(|row| {
                        row.tag.as_deref() == Some("qwen4exp.qsa.selected_attention.packet")
                    })
                    .map(|row| row.kernel.as_str())
                    .collect::<Vec<_>>(),
                [
                    "kernel_qwen4exp_qsa_norm_rope_packed_f32",
                    "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
                    "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
                    "kernel_qwen4exp_qsa_expand_ids_packed_i32",
                    "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
                    "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
                ]
            );
            assert_dispatch_shape(
                &census,
                "qwen4exp.qsa.selected_attention.packet",
                "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
                [
                    ((g.query_heads / g.kv_heads / PACKED_ATTENTION_HEADS_PER_TG)
                        * g.output_width().div_ceil(32)) as u64,
                    g.kv_heads as u64,
                    query_count as u64,
                ],
                [PACKED_ATTENTION_THREADS as u64, 1, 1],
            );
            assert_dispatch_shape(
                &census,
                "qwen4exp.qsa.selected_attention.packet",
                "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
                [
                    (g.query_heads / g.kv_heads / PACKED_ATTENTION_HEADS_PER_TG) as u64,
                    g.kv_heads as u64,
                    query_count as u64,
                ],
                [ATTENTION_THREADS as u64, 1, 1],
            );
        }
    }
}

#[test]
fn selected_attention_packet_validation_fails_before_dispatch() {
    const QUERIES: usize = 2;
    let Some(ctx) = context() else { return };
    let fixture = selected_attention_fixture(&ctx, QUERIES);
    let g = fixture.geometry;
    let start_position = g.output_width();
    let packed = fixture.scratch.views(QUERIES).unwrap();
    assert_eq!(size_of::<PackedAttentionArgs>(), 40);

    assert_qsa_rejected_without_dispatch(&ctx, "aliased attention", |encoder| {
        encode_selected_attention_packet(
            &ctx,
            encoder,
            &fixture.index_query_norm,
            &fixture.compressed_keys,
            &packed.query,
            &packed.query_gate_projection,
            &fixture.key_cache,
            &fixture.value_cache,
            &packed.query,
            &fixture.scratch,
            start_position,
            0,
            QUERIES,
        )
        .map(|_| ())
    });
    assert_qsa_rejected_without_dispatch(&ctx, "crossing range", |encoder| {
        encode_selected_attention_packet(
            &ctx,
            encoder,
            &fixture.index_query_norm,
            &fixture.compressed_keys,
            &packed.query,
            &packed.query_gate_projection,
            &fixture.key_cache,
            &fixture.value_cache,
            &packed.attention,
            &fixture.scratch,
            start_position - 1,
            0,
            QUERIES,
        )
        .map(|_| ())
    });
    let short_query = packed
        .query
        .view_subrange(0, vec![g.query_width() as u64, 1]);
    assert_qsa_rejected_without_dispatch(&ctx, "malformed query shape", |encoder| {
        encode_selected_attention_packet(
            &ctx,
            encoder,
            &fixture.index_query_norm,
            &fixture.compressed_keys,
            &short_query,
            &packed.query_gate_projection,
            &fixture.key_cache,
            &fixture.value_cache,
            &packed.attention,
            &fixture.scratch,
            start_position,
            0,
            QUERIES,
        )
        .map(|_| ())
    });
    let dense_scratch = QwenSparseAttentionPackedScratch::new(&ctx, g, QUERIES).unwrap();
    let dense = dense_scratch.views(QUERIES).unwrap();
    assert_qsa_rejected_without_dispatch(&ctx, "missing selected scratch", |encoder| {
        encode_selected_attention_packet(
            &ctx,
            encoder,
            &fixture.index_query_norm,
            &fixture.compressed_keys,
            &dense.query,
            &dense.query_gate_projection,
            &fixture.key_cache,
            &fixture.value_cache,
            &dense.attention,
            &dense_scratch,
            start_position,
            0,
            QUERIES,
        )
        .map(|_| ())
    });
    let controls_before = fixture
        .scratch
        .selected
        .as_ref()
        .map(|selected| {
            (
                read_i32(&selected.visible_blocks),
                read_i32(&selected.selected_count),
                read_i32(&selected.selector_status),
            )
        })
        .unwrap();
    assert_qsa_rejected_without_dispatch(&ctx, "raw-query band overflow", |encoder| {
        encode_selected_attention_packet(
            &ctx,
            encoder,
            &fixture.index_query_norm,
            &fixture.compressed_keys,
            &packed.query,
            &packed.query_gate_projection,
            &fixture.key_cache,
            &fixture.value_cache,
            &packed.attention,
            &fixture.scratch,
            start_position,
            fixture.scratch.capacity - 1,
            QUERIES,
        )
        .map(|_| ())
    });
    let selected = fixture.scratch.selected.as_ref().unwrap();
    assert_eq!(
        (
            read_i32(&selected.visible_blocks),
            read_i32(&selected.selected_count),
            read_i32(&selected.selector_status),
        ),
        controls_before
    );

    assert!(
        validate_cooperative_pipeline_threads(
            "test selected logits",
            32,
            PACKED_ATTENTION_THREADS - 1,
            PACKED_ATTENTION_THREADS,
            0,
            MAIN_HEAD_DIM * size_of::<u16>(),
            usize::MAX,
        )
        .is_err()
    );
    assert!(
        validate_cooperative_pipeline_threads(
            "test selected logits",
            32,
            PACKED_ATTENTION_THREADS,
            PACKED_ATTENTION_THREADS,
            0,
            MAIN_HEAD_DIM * size_of::<u16>(),
            MAIN_HEAD_DIM * size_of::<u16>() - 1,
        )
        .is_err()
    );
    assert!(validate_selected_reset_pipeline(31).is_err());
}

#[test]
fn selected_attention_faults_overwrite_outputs_without_cache_reads() {
    const QUERIES: usize = 4;
    let Some(ctx) = context() else { return };
    let fixture = selected_attention_fixture(&ctx, QUERIES);
    let g = fixture.geometry;
    let start_position = g.output_width();
    let packed = fixture.scratch.views(QUERIES).unwrap();
    let selected = fixture
        .scratch
        .selected
        .as_ref()
        .unwrap()
        .views(g, QUERIES, QUERIES, QUERIES)
        .unwrap();
    let budget = g.block_budget() as i32;
    write_i32_tensor(
        &selected.selected_count,
        &[budget, budget - 1, budget, budget],
    );
    write_i32_tensor(&selected.selector_status, &[2, 0, 0, 0]);
    let mut ids = vec![-1_i32; g.output_width() * QUERIES];
    for query in 0..QUERIES {
        let start = query * g.output_width();
        for (slot, id) in ids[start..start + g.output_width()].iter_mut().enumerate() {
            *id = slot as i32;
        }
    }
    let fault_start = 2 * g.output_width();
    ids[fault_start + 6] = -1;
    ids[fault_start + 7] = g.capacity as i32;
    ids[fault_start + 8] = (start_position + 3) as i32;
    write_i32_tensor(&selected.token_ids, &ids);
    write_f32_tensor(
        &selected.attention_logits,
        &vec![123.5; g.output_width() * g.query_heads * QUERIES],
    );
    write_f32_tensor(&packed.attention, &vec![-456.25; g.query_width() * QUERIES]);

    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let packet_tag =
        crate::metal::dispatch_census_tag_scope(|| "qwen4exp.qsa.selected_attention.faults".into());
    encode_attention_logits_packed(
        &ctx,
        &encoder,
        &packed.query,
        &fixture.key_cache,
        &selected.token_ids,
        &selected.selected_count,
        &selected.selector_status,
        &selected.attention_logits,
        g,
        start_position,
        QUERIES,
    )
    .unwrap();
    encode_attention_softmax_value_packed(
        &ctx,
        &encoder,
        &packed.query_gate_projection,
        &fixture.value_cache,
        &selected.token_ids,
        &selected.selected_count,
        &selected.selector_status,
        &selected.attention_logits,
        &packed.attention,
        g,
        start_position,
        QUERIES,
    )
    .unwrap();
    drop(packet_tag);
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());

    assert_eq!(
        read_i32(&selected.selected_count),
        [budget, budget - 1, budget, budget]
    );
    assert_eq!(read_i32(&selected.selector_status), [2, 0, 0, 0]);
    let logits = read_f32(&selected.attention_logits);
    let attention = read_f32(&packed.attention);
    for query in 0..2 {
        for head in 0..g.query_heads {
            let start = (query * g.query_heads + head) * g.output_width();
            assert!(
                logits[start..start + g.output_width()]
                    .iter()
                    .all(|value| value.to_bits() == f32::NEG_INFINITY.to_bits())
            );
        }
        let start = query * g.query_width();
        assert!(
            attention[start..start + g.query_width()]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
    }
    for head in 0..g.query_heads {
        let start = (2 * g.query_heads + head) * g.output_width();
        for slot in [6, 7, 8] {
            assert_eq!(logits[start + slot].to_bits(), 0);
        }
        assert_eq!(
            logits[start + g.output_width() - 1].to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
    }
    assert!(
        attention[2 * g.query_width()..]
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(
        census
            .iter()
            .filter(|row| { row.tag.as_deref() == Some("qwen4exp.qsa.selected_attention.faults") })
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>(),
        [
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
        ]
    );
    assert_dispatch_shape(
        &census,
        "qwen4exp.qsa.selected_attention.faults",
        "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
        [
            ((g.query_heads / g.kv_heads / PACKED_ATTENTION_HEADS_PER_TG)
                * g.output_width().div_ceil(32)) as u64,
            g.kv_heads as u64,
            QUERIES as u64,
        ],
        [PACKED_ATTENTION_THREADS as u64, 1, 1],
    );
    assert_dispatch_shape(
        &census,
        "qwen4exp.qsa.selected_attention.faults",
        "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
        [
            (g.query_heads / g.kv_heads / PACKED_ATTENTION_HEADS_PER_TG) as u64,
            g.kv_heads as u64,
            QUERIES as u64,
        ],
        [ATTENTION_THREADS as u64, 1, 1],
    );
}

#[test]
fn selected_packed_motor_matches_scalar_rows_state_and_topology() {
    const TOTAL_TOKENS: usize = 13;
    let Some(ctx) = context() else { return };
    let geometry = selected_motor_test_geometry(16);
    assert_eq!(geometry.query_heads, 24);
    assert_eq!(geometry.kv_heads, 2);
    assert_eq!(geometry.output_width(), 11);
    let weights = test_weights(&ctx, geometry);
    let inputs = values(TOTAL_TOKENS * geometry.hidden_size, 2_303, 0.002_1);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, TOTAL_TOKENS);

    for (start_position, tokens, expected_dense) in [(8, 4, 3), (11, 2, 0)] {
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        for token in 0..start_position {
            let offset = token * geometry.hidden_size;
            encode_one(
                &ctx,
                &weights,
                &mut workspace,
                &inputs[offset..offset + geometry.hidden_size],
            );
        }
        let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
            &ctx, geometry, tokens, true,
        )
        .unwrap();
        let input_start = start_position * geometry.hidden_size;
        let input_end = (start_position + tokens) * geometry.hidden_size;
        let (actual, census) = encode_selected_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &inputs[input_start..input_end],
            start_position,
            tokens,
        );
        let expected = &serial.outputs[input_start..input_end];
        for token in 0..tokens {
            let row = token * geometry.hidden_size;
            assert_similarity(
                &format!("selected packed QSA start={start_position} token={token}"),
                &actual[row..row + geometry.hidden_size],
                &expected[row..row + geometry.hidden_size],
                1e-3,
                0.999999,
                1e-5,
            );
        }
        assert_dense_state_matches(
            &format!("selected packed QSA start={start_position}"),
            &workspace,
            &serial.states[start_position + tokens - 1],
        );
        assert_eq!(workspace.pending_selected_bands, None);
        assert_eq!(read_i32_scalar(&workspace.visible_blocks).unwrap(), 1);

        let names = census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        for required in [
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_audit_selected_i32",
        ] {
            assert_eq!(
                names.iter().filter(|&&name| name == required).count(),
                1,
                "start={start_position} {required}"
            );
        }
        let softmax = names
            .iter()
            .position(|&name| name == "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16")
            .unwrap();
        let audit = names
            .iter()
            .position(|&name| name == "kernel_qwen4exp_qsa_audit_selected_i32")
            .unwrap();
        assert_eq!(audit, softmax + 1);
        for absent in [
            "kernel_qwen4exp_qsa_attention_logits_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_f16",
        ] {
            assert!(!names.contains(&absent), "start={start_position} {absent}");
        }
        let dense_kq = names
            .iter()
            .filter(|&&name| {
                name == "kernel_attn_matrix_kq_f32"
                    || name == "kernel_attn_matrix_kq_f32_full_tiles"
            })
            .count();
        let dense_gate = names
            .iter()
            .filter(|&&name| name == "kernel_sigmoid_mul_gate_strided_f32")
            .count();
        assert_eq!(dense_kq > 0, expected_dense > 0);
        assert_eq!(dense_gate, usize::from(expected_dense > 0));
    }
}

#[test]
fn selected_packed_motor_keeps_bf16_index_queries_in_f32_activations() {
    const START_POSITION: usize = 35;
    const TOKENS: usize = 32;
    const TOTAL_TOKENS: usize = START_POSITION + TOKENS;
    let Some(ctx) = context() else { return };
    let geometry = selected_bf16_motor_test_geometry(96);
    assert_eq!(geometry.output_width(), START_POSITION);
    let mut weights = test_weights(&ctx, geometry);
    let index_query_values = read_f32(&weights.index_query);
    weights.index_query = bf16_weight(
        &ctx,
        &index_query_values,
        vec![
            geometry.hidden_size as u64,
            geometry.index_query_width() as u64,
        ],
    );
    let inputs = values(TOTAL_TOKENS * geometry.hidden_size, 2_311, 0.002_1);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, TOTAL_TOKENS);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    for token in 0..START_POSITION {
        let offset = token * geometry.hidden_size;
        encode_one(
            &ctx,
            &weights,
            &mut workspace,
            &inputs[offset..offset + geometry.hidden_size],
        );
    }
    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, TOKENS, true,
    )
    .unwrap();
    let input_start = START_POSITION * geometry.hidden_size;
    let (actual, census) = crate::metal_forward::with_matmat_bf16_bfloat_act_override(true, || {
        encode_selected_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &inputs[input_start..],
            START_POSITION,
            TOKENS,
        )
    });
    let expected = &serial.outputs[input_start..];
    for token in 0..TOKENS {
        let row = token * geometry.hidden_size;
        assert_similarity(
            &format!("selected packed QSA BF16 token={token}"),
            &actual[row..row + geometry.hidden_size],
            &expected[row..row + geometry.hidden_size],
            1e-3,
            0.999999,
            1e-5,
        );
    }
    assert_dense_state_matches(
        "selected packed QSA BF16",
        &workspace,
        &serial.states[TOTAL_TOKENS - 1],
    );
    let names = census
        .iter()
        .map(|row| row.kernel.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names
            .iter()
            .filter(|&&name| name == "kernel_mat_mat_bf16_f32")
            .count(),
        1,
        "BF16 index-query route: {names:?}"
    );
    assert_eq!(
        names
            .iter()
            .filter(|&&name| name == "kernel_qwen4exp_qsa_audit_selected_i32")
            .count(),
        1
    );
}

#[test]
fn selected_packed_motor_reuses_two_bands_in_order() {
    const TOTAL_TOKENS: usize = 131;
    let Some(ctx) = context() else { return };
    let geometry = selected_multi_band_test_geometry(160);
    assert_eq!(geometry.output_width(), 67);
    let inputs = values(TOTAL_TOKENS * geometry.hidden_size, 2_317, 0.002_3);

    let run_case = |weights: &TestWeights,
                    start_position: usize,
                    tokens: usize,
                    expected_dense: bool,
                    expected_projection: &str| {
        let serial_end = (start_position + tokens) * geometry.hidden_size;
        let serial = serial_dense_trace(
            &ctx,
            weights,
            &inputs[..serial_end],
            start_position + tokens,
        );
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        for token in 0..start_position {
            let offset = token * geometry.hidden_size;
            encode_one(
                &ctx,
                weights,
                &mut workspace,
                &inputs[offset..offset + geometry.hidden_size],
            );
        }
        let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
            &ctx, geometry, tokens, true,
        )
        .unwrap();
        let input_start = start_position * geometry.hidden_size;
        let input_end = (start_position + tokens) * geometry.hidden_size;
        let (actual, census) = encode_selected_packed_chunk(
            &ctx,
            weights,
            &mut workspace,
            &scratch,
            &inputs[input_start..input_end],
            start_position,
            tokens,
        );
        assert_similarity(
            &format!("selected packed QSA two-band start={start_position}"),
            &actual,
            &serial.outputs[input_start..input_end],
            1e-3,
            0.999999,
            1e-5,
        );
        assert_dense_state_matches(
            &format!("selected packed QSA two-band start={start_position}"),
            &workspace,
            &serial.states[start_position + tokens - 1],
        );
        assert_eq!(read_i32_scalar(&workspace.visible_blocks).unwrap(), 2);

        let projection = census
            .iter()
            .filter(|row| row.tag.as_deref() == Some("qwen4exp.qsa.selected_index_projection"))
            .collect::<Vec<_>>();
        assert_eq!(projection.len(), 1);
        assert_eq!(projection[0].kernel, expected_projection);
        assert_eq!(
            census
                .iter()
                .filter(|row| { row.tag.as_deref() == Some("qwen4exp.qsa.output_projection") })
                .count(),
            1
        );
        let packet = [
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_audit_selected_i32",
        ];
        for ordinal in 0..2 {
            let tag = format!("qwen4exp.qsa.selected_band.{ordinal}");
            let names = census
                .iter()
                .filter(|row| row.tag.as_deref() == Some(tag.as_str()))
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>();
            let mut expected = vec!["kernel_qwen4exp_qsa_reset_selected_controls_i32"];
            expected.extend(packet);
            assert_eq!(names, expected);
        }
        assert!(
            !census
                .iter()
                .any(|row| { row.tag.as_deref() == Some("qwen4exp.qsa.selected_band.2") })
        );
        let names = census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_scatter_offset_f32_to_f16_kv")
                .count(),
            1
        );
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_qwen4exp_qsa_audit_selected_i32")
                .count(),
            2
        );
        assert_eq!(
            names
                .iter()
                .filter(|&&name| { name == "kernel_qwen4exp_qsa_reset_selected_controls_i32" })
                .count(),
            2
        );
        assert_eq!(
            names
                .iter()
                .any(|&name| name == "kernel_sigmoid_mul_gate_strided_f32"),
            expected_dense
        );
        assert!(!names.contains(&"kernel_qwen4exp_qsa_attention_logits_f16"));
        assert!(!names.contains(&"kernel_qwen4exp_qsa_attention_softmax_value_f16"));
    };

    let weights = test_weights(&ctx, geometry);
    run_case(&weights, 64, 36, true, "kernel_mat_mat_f32_f32");

    let mut bf16_weights = test_weights(&ctx, geometry);
    bf16_weights.index_query = bf16_weight(
        &ctx,
        &read_f32(&bf16_weights.index_query),
        vec![
            geometry.hidden_size as u64,
            geometry.index_query_width() as u64,
        ],
    );
    crate::metal_forward::with_matmat_bf16_bfloat_act_override(true, || {
        run_case(
            &bf16_weights,
            geometry.output_width(),
            64,
            false,
            "kernel_mat_mat_bf16_f32",
        )
    });
}

#[test]
fn selected_packed_motor_preflight_rejects_missing_or_aliased_scratch() {
    const TOKENS: usize = 2;
    let Some(ctx) = context() else { return };
    let geometry = selected_motor_test_geometry(16);
    let weights = test_weights(&ctx, geometry);
    let start_position = geometry.output_width();
    let input_values = values(geometry.hidden_size * TOKENS, 2_319, 0.002_1);
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![geometry.hidden_size as u64, TOKENS as u64],
        GgmlType::F32,
    )
    .unwrap();

    let dense_scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, TOKENS).unwrap();
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    workspace.committed_length = start_position;
    assert_qsa_rejected_without_dispatch(&ctx, "missing selected motor scratch", |encoder| {
        unsafe {
            encode_qwen_sparse_attention_text_packed_motor(
                &ctx,
                encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
                &dense_scratch,
                start_position,
                TOKENS,
            )
        }
        .map(|_| ())
    });
    assert!(workspace.active_command.is_none());
    assert!(workspace.pending_length.is_none());
    assert!(workspace.pending_selected_bands.is_none());
    assert!(!workspace.is_poisoned());

    let mut aliased_scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, TOKENS, true,
    )
    .unwrap();
    let selected = aliased_scratch.selected.as_mut().unwrap();
    selected.index_query = selected.index_query_raw.clone();
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    workspace.committed_length = start_position;
    assert_qsa_rejected_without_dispatch(&ctx, "aliased selected motor scratch", |encoder| {
        unsafe {
            encode_qwen_sparse_attention_text_packed_motor(
                &ctx,
                encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
                &aliased_scratch,
                start_position,
                TOKENS,
            )
        }
        .map(|_| ())
    });
    assert!(workspace.active_command.is_none());
    assert!(!workspace.is_poisoned());

    let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx, geometry, TOKENS, true,
    )
    .unwrap();
    let mut writable_index_query = weights.index_query.clone();
    writable_index_query.provenance = MetalTensorProvenance::OwnedWritable;
    let bad_weights = QwenSparseAttentionMetalWeights {
        index_query: &writable_index_query,
        ..weights.borrowed()
    };
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    workspace.committed_length = start_position;
    assert_qsa_rejected_without_dispatch(&ctx, "writable selected index query", |encoder| {
        unsafe {
            encode_qwen_sparse_attention_text_packed_motor(
                &ctx,
                encoder,
                &input,
                bad_weights,
                &mut workspace,
                &scratch,
                start_position,
                TOKENS,
            )
        }
        .map(|_| ())
    });
    assert!(workspace.active_command.is_none());
    assert!(!workspace.is_poisoned());

    let incompatible_geometry = test_geometry(16);
    assert_eq!(
        incompatible_geometry.query_heads / incompatible_geometry.kv_heads,
        2
    );
    let incompatible_weights = test_weights(&ctx, incompatible_geometry);
    let incompatible_start = incompatible_geometry.output_width();
    let incompatible_input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![incompatible_geometry.hidden_size as u64, TOKENS as u64],
        GgmlType::F32,
    )
    .unwrap();
    let incompatible_scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
        &ctx,
        incompatible_geometry,
        TOKENS,
        true,
    )
    .unwrap();
    let mut incompatible_workspace =
        QwenSparseAttentionMetalWorkspace::new(&ctx, incompatible_geometry).unwrap();
    incompatible_workspace.committed_length = incompatible_start;
    let controls_before = (
        read_i32(&incompatible_workspace.visible_blocks),
        read_i32(&incompatible_workspace.selected_count),
        read_i32(&incompatible_workspace.selector_status),
    );
    assert_qsa_rejected_without_dispatch(&ctx, "incompatible selected GQA", |encoder| {
        unsafe {
            encode_qwen_sparse_attention_text_packed_motor(
                &ctx,
                encoder,
                &incompatible_input,
                incompatible_weights.borrowed(),
                &mut incompatible_workspace,
                &incompatible_scratch,
                incompatible_start,
                TOKENS,
            )
        }
        .map(|_| ())
    });
    assert_eq!(
        (
            read_i32(&incompatible_workspace.visible_blocks),
            read_i32(&incompatible_workspace.selected_count),
            read_i32(&incompatible_workspace.selector_status),
        ),
        controls_before
    );
    assert!(incompatible_workspace.active_command.is_none());
    assert!(incompatible_workspace.pending_length.is_none());
    assert!(incompatible_workspace.pending_selected_bands.is_none());
    assert!(!incompatible_workspace.is_poisoned());
}

#[test]
fn selected_audit_preserves_native_failures_and_detects_stale_rows() {
    const QUERIES: usize = 4;
    let Some(ctx) = context() else { return };
    assert_eq!(size_of::<SelectedAuditArgs>(), 20);
    let counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let workspace_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let workspace_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let audited = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();

    let run = |counts_values: &[i32], status_values: &[i32], band_ordinal: usize| {
        write_i32_tensor(&counts, counts_values);
        write_i32_tensor(&status, status_values);
        crate::metal::dispatch_census_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_selected_audit(
            &ctx,
            &encoder,
            &counts,
            &status,
            &workspace_count,
            &workspace_status,
            &audited,
            QUERIES,
            2,
            band_ordinal,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, "kernel_qwen4exp_qsa_audit_selected_i32");
        assert_eq!(
            (
                census[0].grid_width,
                census[0].grid_height,
                census[0].grid_depth,
                census[0].threads_width,
            ),
            (1, 1, 1, 1)
        );
    };

    run(&[2, 1, 2, 2], &[0, 7, 0, 0], 0);
    assert_eq!(read_i32_scalar(&workspace_status).unwrap(), 7);
    assert_eq!(read_i32_scalar(&workspace_count).unwrap(), 2);
    assert_eq!(read_i32_scalar(&audited).unwrap(), 1);
    run(&[2, 2, 2, 2], &[0, 0, 0, 0], 1);
    assert_eq!(read_i32_scalar(&workspace_status).unwrap(), 7);
    assert_eq!(read_i32_scalar(&audited).unwrap(), 2);

    write_i32_scalar(&workspace_status, 0).unwrap();
    write_i32_scalar(&audited, 0).unwrap();
    run(&[2, 1, 2, 2], &[0, 0, 0, 0], 0);
    assert_eq!(
        read_i32_scalar(&workspace_status).unwrap(),
        SELECTED_COUNT_MISMATCH_STATUS
    );
    assert_eq!(read_i32_scalar(&audited).unwrap(), 1);

    write_i32_scalar(&workspace_status, 0).unwrap();
    write_i32_scalar(&audited, 0).unwrap();
    run(&[2, 2, 2, 2], &[-1, 0, 0, 0], 0);
    assert_eq!(read_i32_scalar(&workspace_status).unwrap(), -1);
    assert_eq!(read_i32_scalar(&audited).unwrap(), 1);

    write_i32_scalar(&workspace_status, 0).unwrap();
    write_i32_scalar(&workspace_count, -1).unwrap();
    write_i32_scalar(&audited, 0).unwrap();
    run(&[2, 2, 2, 2], &[0, 0, 0, 0], 1);
    assert_eq!(
        read_i32_scalar(&workspace_status).unwrap(),
        SELECTED_AUDIT_ORDER_MISMATCH_STATUS
    );
    assert_eq!(read_i32_scalar(&workspace_count).unwrap(), -1);
    assert_eq!(read_i32_scalar(&audited).unwrap(), 0);

    write_i32_scalar(&workspace_status, 0).unwrap();
    write_i32_scalar(&workspace_count, -1).unwrap();
    write_i32_scalar(&audited, 0).unwrap();
    run(&[2, 2, 2, 2], &[0, 0, 0, 0], 0);
    run(&[2, 2, 2, 2], &[0, 0, 0, 0], 0);
    assert_eq!(
        read_i32_scalar(&workspace_status).unwrap(),
        SELECTED_AUDIT_ORDER_MISMATCH_STATUS
    );
    assert_eq!(read_i32_scalar(&workspace_count).unwrap(), 2);
    assert_eq!(read_i32_scalar(&audited).unwrap(), 1);
}

#[test]
fn selected_control_reset_clears_only_the_reused_band() {
    const CAPACITY: usize = 32;
    const USED: usize = 17;
    let Some(ctx) = context() else { return };
    let visible = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64]).unwrap();
    for tensor in [&visible, &counts, &status] {
        write_i32_tensor(tensor, &vec![7; CAPACITY]);
    }
    crate::metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_selected_control_reset(&ctx, &encoder, &visible, &counts, &status, USED).unwrap();
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    for tensor in [&visible, &counts, &status] {
        let values = read_i32(tensor);
        assert!(values[..USED].iter().all(|&value| value == -1));
        assert!(values[USED..].iter().all(|&value| value == 7));
    }
    assert_eq!(census.len(), 1);
    assert_eq!(
        census[0].kernel,
        "kernel_qwen4exp_qsa_reset_selected_controls_i32"
    );
    assert_eq!(
        (
            census[0].grid_width,
            census[0].threads_width,
            census[0].grid_tgs,
            census[0].tg_threads,
        ),
        (1, 32, 1, 32)
    );
}

#[test]
fn selected_release_requires_complete_audit_and_consistent_ownership() {
    let Some(ctx) = context() else { return };
    let geometry = selected_motor_test_geometry(16);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

    for audited_bands in [0, 2] {
        write_i32_scalar(&workspace.selector_status, 0).unwrap();
        write_i32_scalar(&workspace.selected_count, 2).unwrap();
        write_i32_scalar(&workspace.visible_blocks, audited_bands).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        reserve_command(&mut workspace, &encoder, 12, 1).unwrap();
        encoder.end();
        command.commit();
        assert!(workspace.release_after().is_err());
        assert!(workspace.is_poisoned());
        assert_eq!(workspace.committed_length(), 0);
        assert!(workspace.pending_length.is_none());
        assert!(workspace.pending_selected_bands.is_none());
        workspace.reset().unwrap();
    }

    write_i32_scalar(&workspace.selector_status, 0).unwrap();
    write_i32_scalar(&workspace.selected_count, 2).unwrap();
    write_i32_scalar(&workspace.visible_blocks, 1).unwrap();
    let poisoned_command = ctx.queue.commandBuffer().unwrap();
    let poisoned_encoder = KernelEncoder::begin(&poisoned_command);
    reserve_command(&mut workspace, &poisoned_encoder, 12, 1).unwrap();
    workspace.state_poisoned = true;
    poisoned_encoder.end();
    poisoned_command.commit();
    assert!(workspace.release_after().is_err());
    assert_eq!(workspace.committed_length(), 0);
    workspace.reset().unwrap();

    workspace.pending_length = Some(12);
    workspace.pending_selected_bands = Some(1);
    assert!(workspace.release_after().is_err());
    assert!(workspace.is_poisoned());
    assert!(workspace.pending_length.is_none());
    assert!(workspace.pending_selected_bands.is_none());
    workspace.reset().unwrap();

    write_i32_scalar(&workspace.selector_status, 0).unwrap();
    write_i32_scalar(&workspace.selected_count, 1).unwrap();
    write_i32_scalar(&workspace.visible_blocks, 99).unwrap();
    let dense_command = ctx.queue.commandBuffer().unwrap();
    let dense_encoder = KernelEncoder::begin(&dense_command);
    reserve_command(&mut workspace, &dense_encoder, 4, 0).unwrap();
    dense_encoder.end();
    dense_command.commit();
    workspace.release_after().unwrap();
    assert_eq!(workspace.committed_length(), 4);
    assert!(!workspace.is_poisoned());
}

#[test]
fn dense_packed_qsa_covers_the_preselection_shoulder() {
    const TOKEN_BUDGET: usize = 8;
    const TOKENS: usize = TOKEN_BUDGET + 3;
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(12);
    assert_eq!(geometry.token_budget(), TOKEN_BUDGET);
    assert_eq!(geometry.output_width(), TOKENS);
    let weights = test_weights(&ctx, geometry);
    let inputs = values(TOKENS * geometry.hidden_size, 1_663, 0.002_2);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, TOKENS);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    for token in 0..TOKEN_BUDGET {
        let start = token * geometry.hidden_size;
        encode_one(
            &ctx,
            &weights,
            &mut workspace,
            &inputs[start..start + geometry.hidden_size],
        );
    }
    let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, 3).unwrap();
    assert_eq!(
        scratch.attention_scores.shape,
        [TOKENS as u64, geometry.query_heads as u64, 3]
    );
    let input_start = TOKEN_BUDGET * geometry.hidden_size;
    let (actual, census) = encode_dense_packed_chunk(
        &ctx,
        &weights,
        &mut workspace,
        &scratch,
        &inputs[input_start..],
        TOKEN_BUDGET,
        3,
    );
    let expected = &serial.outputs[input_start..];
    for token in 0..3 {
        let row = token * geometry.hidden_size;
        assert_similarity(
            &format!("dense packed QSA shoulder token={token}"),
            &actual[row..row + geometry.hidden_size],
            &expected[row..row + geometry.hidden_size],
            1e-3,
            0.9999997,
            1e-5,
        );
    }
    assert_dense_state_matches(
        "dense packed QSA shoulder",
        &workspace,
        &serial.states[TOKENS - 1],
    );
    let names = census
        .iter()
        .map(|row| row.kernel.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"kernel_attn_matrix_softmax_f32"));
    for absent in [
        "qsa_index_scores",
        "select_top_k",
        "qsa_expand_ids",
        "qsa_attention_logits",
        "qsa_attention_softmax_value",
    ] {
        assert!(
            !names.iter().any(|name| name.contains(absent)),
            "shoulder unexpectedly dispatched {absent}"
        );
    }
}

#[test]
fn dense_packed_qsa_rejects_selected_suffix_before_dispatch() {
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(16);
    let weights = test_weights(&ctx, geometry);
    let all_inputs = values(14 * geometry.hidden_size, 1_727, 0.001_8);

    for (start_position, tokens, selected_offset) in
        [(8_usize, 4_usize, 3_usize), (11, 2, 0), (12, 2, 0)]
    {
        let plan = geometry.plan_packed_range(start_position, tokens).unwrap();
        assert_eq!(plan.selected_offset, selected_offset);
        assert!(plan.selected_tokens > 0);

        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        for token in 0..start_position {
            let start = token * geometry.hidden_size;
            encode_one(
                &ctx,
                &weights,
                &mut workspace,
                &all_inputs[start..start + geometry.hidden_size],
            );
        }
        let scratch = QwenSparseAttentionPackedScratch::new_with_selected_capability(
            &ctx, geometry, tokens, true,
        )
        .unwrap();
        let input_start = start_position * geometry.hidden_size;
        let input_end = (start_position + tokens) * geometry.hidden_size;
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&all_inputs[input_start..input_end]),
            vec![geometry.hidden_size as u64, tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let before = (
            read_tensor_bytes(&workspace.pending_index_keys),
            read_tensor_bytes(&workspace.compressed_index_keys),
            read_tensor_bytes(&workspace.key_cache),
            read_tensor_bytes(&workspace.value_cache),
            read_i32(&workspace.visible_blocks),
            read_i32(&workspace.selected_blocks),
            read_i32(&workspace.selected_count),
            read_i32(&workspace.selector_status),
            read_i32(&workspace.token_ids),
        );
        crate::metal::dispatch_census_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = match unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                &ctx,
                &encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
                &scratch,
                start_position,
                tokens,
            )
        } {
            Ok(_) => {
                panic!("selected suffix start={start_position} tokens={tokens} was accepted")
            }
            Err(error) => error,
        };
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        assert!(error.to_string().contains("dense limit"));
        assert!(census.is_empty());
        assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
        assert!(workspace.active_command.is_none());
        assert!(workspace.pending_length.is_none());
        assert!(workspace.pending_selected_bands.is_none());
        assert_eq!(workspace.committed_length(), start_position);
        assert!(!workspace.is_poisoned());
        assert_eq!(
            (
                read_tensor_bytes(&workspace.pending_index_keys),
                read_tensor_bytes(&workspace.compressed_index_keys),
                read_tensor_bytes(&workspace.key_cache),
                read_tensor_bytes(&workspace.value_cache),
                read_i32(&workspace.visible_blocks),
                read_i32(&workspace.selected_blocks),
                read_i32(&workspace.selected_count),
                read_i32(&workspace.selector_status),
                read_i32(&workspace.token_ids),
            ),
            before,
            "start={start_position} tokens={tokens}"
        );
    }
}

#[test]
fn dense_packed_qsa_matches_scalar_rows_state_and_topology() {
    const MAX_TOKENS: usize = 64;
    let Some(ctx) = context() else { return };
    let geometry = packed_test_geometry(64);
    assert_eq!(geometry.query_heads, 24);
    assert_eq!(geometry.kv_heads, 2);
    let weights = test_weights(&ctx, geometry);
    let inputs = values(MAX_TOKENS * geometry.hidden_size, 1_701, 0.002_1);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, MAX_TOKENS);

    for tokens in [1_usize, 2, 8, 16, 33, 64] {
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, tokens).unwrap();
        let (actual, census) = encode_dense_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &inputs[..tokens * geometry.hidden_size],
            0,
            tokens,
        );
        let expected = &serial.outputs[..tokens * geometry.hidden_size];
        if tokens == 1 {
            assert_eq!(actual, expected, "N=1 must delegate exactly");
            assert_eq!(
                read_f32(&workspace.pending_index_keys),
                serial.states[0].pending_index_keys
            );
            assert_eq!(
                read_f16(&workspace.key_cache)[..geometry.kv_width()],
                serial.states[0].key_cache
            );
            assert_eq!(
                read_f16(&workspace.value_cache)[..geometry.kv_width()],
                serial.states[0].value_cache
            );
        } else {
            for token in 0..tokens {
                let start = token * geometry.hidden_size;
                assert_similarity(
                    &format!("dense packed QSA N={tokens} token={token}"),
                    &actual[start..start + geometry.hidden_size],
                    &expected[start..start + geometry.hidden_size],
                    1e-3,
                    0.9999997,
                    1e-5,
                );
            }
        }
        assert_dense_state_matches(
            &format!("dense packed QSA N={tokens}"),
            &workspace,
            &serial.states[tokens - 1],
        );

        let names = census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        if tokens == 1 {
            assert!(names.contains(&"kernel_qwen4exp_qsa_write_pending_f32"));
            assert!(names.contains(&"kernel_qwen4exp_qsa_attention_logits_f16"));
            assert!(!names.iter().any(|name| name.contains("packed")));
            continue;
        }
        for required in [
            "kernel_qwen4exp_qsa_commit_pending_packed_f32",
            "kernel_qwen4exp_qsa_fill_block_ids_i32",
            "kernel_qk_rms_norm_rope_f32_packed_consecutive",
            "kernel_scatter_offset_f32_to_f16_kv",
            "kernel_attn_matrix_softmax_f32",
            "kernel_attn_matrix_kqv_direct_v_f32",
            "kernel_sigmoid_mul_gate_strided_f32",
        ] {
            assert!(names.contains(&required), "N={tokens} missing {required}");
        }
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_qwen4exp_qsa_pool_publish_packed_f16")
                .count(),
            usize::from(tokens >= geometry.ratio)
        );
        let attention_tiles = tokens.div_ceil(DENSE_PACKED_QUERY_TILE);
        assert_eq!(
            names
                .iter()
                .filter(|&&name| {
                    matches!(
                        name,
                        "kernel_attn_matrix_kq_f32" | "kernel_attn_matrix_kq_f32_full_tiles"
                    )
                })
                .count(),
            attention_tiles
        );
        if tokens == 64 {
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| name == "kernel_attn_matrix_kq_f32_full_tiles")
                    .count(),
                attention_tiles
            );
        }
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_attn_matrix_softmax_f32")
                .count(),
            attention_tiles
        );
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_attn_matrix_kqv_direct_v_f32")
                .count(),
            attention_tiles
        );
        for absent in [
            "qsa_index_scores",
            "select_top_k",
            "qsa_expand_ids",
            "qsa_attention_logits",
            "qsa_attention_softmax_value",
        ] {
            assert!(
                !names.iter().any(|name| name.contains(absent)),
                "N={tokens} unexpectedly dispatched {absent}"
            );
        }
    }
}

#[test]
fn dense_packed_qsa_cross_block_continuation_matches_scalar() {
    const TOKENS: usize = 33;
    let Some(ctx) = context() else { return };
    let geometry = packed_test_geometry(64);
    let weights = test_weights(&ctx, geometry);
    let inputs = values(TOKENS * geometry.hidden_size, 1_919, 0.001_9);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, TOKENS);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, TOKENS).unwrap();
    let mut start_position = 0;

    for chunk in [2_usize, 8, 16, 7] {
        let input_start = start_position * geometry.hidden_size;
        let input_end = (start_position + chunk) * geometry.hidden_size;
        let (actual, _) = encode_dense_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &inputs[input_start..input_end],
            start_position,
            chunk,
        );
        let expected = &serial.outputs[input_start..input_end];
        for token in 0..chunk {
            let row = token * geometry.hidden_size;
            assert_similarity(
                &format!("dense packed QSA continuation start={start_position} token={token}"),
                &actual[row..row + geometry.hidden_size],
                &expected[row..row + geometry.hidden_size],
                1e-3,
                0.9999997,
                1e-5,
            );
        }
        start_position += chunk;
        assert_dense_state_matches(
            &format!("dense packed QSA continuation end={start_position}"),
            &workspace,
            &serial.states[start_position - 1],
        );
    }
    assert_eq!(start_position, TOKENS);

    for (prefix, chunk) in [(0_usize, 4_usize), (1, 3), (2, 2), (3, 2)] {
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        for token in 0..prefix {
            let start = token * geometry.hidden_size;
            encode_one(
                &ctx,
                &weights,
                &mut workspace,
                &inputs[start..start + geometry.hidden_size],
            );
        }
        let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, chunk).unwrap();
        let input_start = prefix * geometry.hidden_size;
        let input_end = (prefix + chunk) * geometry.hidden_size;
        let (actual, _) = encode_dense_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &inputs[input_start..input_end],
            prefix,
            chunk,
        );
        let expected = &serial.outputs[input_start..input_end];
        for token in 0..chunk {
            let row = token * geometry.hidden_size;
            assert_similarity(
                &format!(
                    "dense packed QSA residue={} token={token}",
                    prefix % geometry.ratio
                ),
                &actual[row..row + geometry.hidden_size],
                &expected[row..row + geometry.hidden_size],
                1e-3,
                0.9999997,
                1e-5,
            );
        }
        assert_dense_state_matches(
            &format!("dense packed QSA residue={}", prefix % geometry.ratio),
            &workspace,
            &serial.states[prefix + chunk - 1],
        );
    }
}

#[test]
fn dense_packed_qsa_preflight_ownership_poison_and_reset_are_strict() {
    let Some(ctx) = context() else { return };
    let geometry = packed_test_geometry(64);
    let weights = test_weights(&ctx, geometry);
    let input_values = values(2 * geometry.hidden_size, 2_113, 0.002_3);
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![geometry.hidden_size as u64, 2],
        GgmlType::F32,
    )
    .unwrap();
    let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, 2).unwrap();
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

    let mut writable_query = weights.query.clone();
    writable_query.provenance = MetalTensorProvenance::OwnedWritable;
    let bad_weights = QwenSparseAttentionMetalWeights {
        query: &writable_query,
        ..weights.borrowed()
    };
    let bad_command = ctx.queue.commandBuffer().unwrap();
    let bad_encoder = KernelEncoder::begin(&bad_command);
    crate::metal::dispatch_census_begin();
    let error = match unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor(
            &ctx,
            &bad_encoder,
            &input,
            bad_weights,
            &mut workspace,
            &scratch,
            0,
            2,
        )
    } {
        Ok(_) => panic!("writable QSA weight was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("read-only weight provenance"));
    assert!(crate::metal::dispatch_census_take().is_empty());
    bad_encoder.end();
    assert!(workspace.active_command.is_none());
    assert_eq!(workspace.committed_length(), 0);
    assert!(!workspace.is_poisoned());

    let alias = scratch
        .output
        .view_subrange(0, vec![geometry.hidden_size as u64, 2]);
    let alias_command = ctx.queue.commandBuffer().unwrap();
    let alias_encoder = KernelEncoder::begin(&alias_command);
    crate::metal::dispatch_census_begin();
    let error = match unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor(
            &ctx,
            &alias_encoder,
            &alias,
            weights.borrowed(),
            &mut workspace,
            &scratch,
            0,
            2,
        )
    } {
        Ok(_) => panic!("aliased packed QSA input was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("overlap"));
    assert!(crate::metal::dispatch_census_take().is_empty());
    alias_encoder.end();
    assert!(workspace.active_command.is_none());
    assert!(!workspace.is_poisoned());

    let abandoned_command = ctx.queue.commandBuffer().unwrap();
    let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor(
            &ctx,
            &abandoned_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
            &scratch,
            0,
            2,
        )
    }
    .unwrap();
    abandoned_encoder.end();
    assert!(workspace.reset().is_err());
    unsafe { workspace.abandon_uncommitted().unwrap() };
    drop(abandoned_command);
    assert_eq!(workspace.committed_length(), 0);
    assert!(!workspace.is_poisoned());

    let poison_command = ctx.queue.commandBuffer().unwrap();
    let poison_encoder = KernelEncoder::begin(&poison_command);
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor(
            &ctx,
            &poison_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
            &scratch,
            0,
            2,
        )
    }
    .unwrap();
    poison_encoder.end();
    poison_command.commit();
    poison_command.waitUntilCompleted();
    write_i32_scalar(&workspace.selector_status, 17).unwrap();
    assert!(workspace.release_after().is_err());
    assert!(workspace.is_poisoned());
    workspace.reset().unwrap();
    let (output, _) = encode_dense_packed_chunk(
        &ctx,
        &weights,
        &mut workspace,
        &scratch,
        &input_values,
        0,
        2,
    );
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(workspace.committed_length(), 2);
}

#[test]
fn thirteen_tokens_match_independent_pytorch_oracle() {
    let fixture: Value = serde_json::from_str(QSA_ORACLE_JSON).unwrap();
    assert_eq!(fixture["schema"], "qwen4exp-qsa-text-f16-oracle");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["generator_version"], 1);
    assert_eq!(fixture["binary"]["file"], "qwen4exp_qsa_text_f16_v1.f32");
    assert_eq!(fixture["binary"]["dtype"], "f32");
    assert_eq!(fixture["binary"]["byte_order"], "little");
    assert_eq!(
        format!("{:x}", Sha256::digest(QSA_ORACLE_F32)),
        fixture["binary"]["sha256"].as_str().unwrap()
    );
    assert_source_identity(
        &fixture,
        "vllm",
        "02f2b4c15dd987d9436e125aab29604447c77405",
        "72eaeeeb9bfff19494a9d19ee85a6a83745d0602",
        &[
            (
                "vllm/models/qwen4_exp/nvidia/indexer_qsa.py",
                "668706c3a59c51e2c1ed51d19bd9a0e1564a0aad68c5e2bc20d5fb9e65cc2f98",
            ),
            (
                "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
                "faa8d358c79745f304edd363e4da21992e4cf015a22316b14980500bd199a0ad",
            ),
            (
                "tests/models/qwen4_exp/test_qsa_reference.py",
                "7396d4482c2e7a0529bd927b2922925a709916910b651b2ed5da790ec2d385f1",
            ),
        ],
    );
    assert_source_identity(
        &fixture,
        "sglang",
        "73a255206f916366c8d26d4022f82ddfb0ab558d",
        "dc134c86f21a7396d89bdb01a8019e8db81d763a",
        &[
            (
                "python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
                "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2",
            ),
            (
                "python/sglang/srt/layers/attention/qsa/kernel.py",
                "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8",
            ),
            (
                "python/sglang/srt/layers/attention/qsa/mqa.py",
                "af36d5c8f4fbda5b0e82b7f31046a95c9a709fcc57b3600c6473c49e87b7629f",
            ),
            (
                "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
                "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2",
            ),
        ],
    );
    assert!(fixture["minimum_sparse_score_margin"].as_f64().unwrap() > 0.05);
    for fault in [
        "modulo_gqa_mapping",
        "silu_gate",
        "partial_extra_block_tail_residue_1",
    ] {
        assert!(
            fixture["fault_sensitivity_max_abs_output"][fault]
                .as_f64()
                .unwrap()
                > 1e-2
        );
    }

    let Some(ctx) = context() else { return };
    let geometry = test_geometry(16);
    assert_eq!(geometry.query_heads, 4);
    assert_eq!(geometry.kv_heads, 2);
    assert_eq!(geometry.hidden_size, 16);
    let weights = oracle_weights(&ctx, geometry, &fixture);
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    let oracle = oracle_values();
    let inputs = formula_from_json(
        &fixture["recipe"]["weights_and_inputs"]["inputs"],
        13,
        geometry.hidden_size,
    );
    let mut observed_residues = [false; 4];
    let mut saw_true_sparse = false;
    for position in 0..13 {
        let input = &inputs[position * geometry.hidden_size..(position + 1) * geometry.hidden_size];
        let actual = encode_one(&ctx, &weights, &mut workspace, input);
        let row = |name: &str, width: usize| {
            let section = oracle_section(&fixture, &oracle, name);
            &section[position * width..(position + 1) * width]
        };
        assert_close(&actual, row("output", geometry.hidden_size), 4e-4, 5e-4);
        assert_close(
            &read_f32(&workspace.attention),
            row("attention", geometry.query_width()),
            4e-4,
            4e-4,
        );
        assert_close(
            &read_f32(&workspace.index_query),
            row("index_query", geometry.index_query_width()),
            8e-5,
            8e-5,
        );
        assert_close(
            &read_f32(&workspace.query),
            row("query", geometry.query_width()),
            8e-5,
            8e-5,
        );
        assert_close(
            &read_f32(&workspace.raw_gate),
            row("raw_gate", geometry.query_width()),
            2e-5,
            2e-5,
        );
        assert_close(
            &read_f32(&workspace.key),
            row("key", geometry.kv_width()),
            8e-5,
            8e-5,
        );
        assert_close(
            &read_f32(&workspace.value),
            row("value", geometry.kv_width()),
            2e-5,
            2e-5,
        );

        let step = &fixture["steps"][position];
        let expected_ids = step["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        assert_eq!(read_i32(&workspace.token_ids), expected_ids);
        let expected_blocks = step["selected_blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        assert_eq!(
            read_i32_scalar(&workspace.selected_count).unwrap(),
            expected_blocks.len() as i32
        );
        let mut expected_block_slots = expected_blocks.clone();
        expected_block_slots.resize(geometry.block_budget(), -1);
        assert_eq!(read_i32(&workspace.selected_blocks), expected_block_slots);
        let visible = step["visible_blocks"].as_u64().unwrap() as usize;
        if visible > 0 {
            assert_close(
                &read_f32(&workspace.scores)[..visible],
                &row("scores", geometry.block_capacity())[..visible],
                3e-4,
                3e-4,
            );
        }
        observed_residues[(position + 1) % geometry.ratio] = true;
        if visible > geometry.block_budget() {
            saw_true_sparse = true;
            assert!(expected_blocks.len() < visible);
        }
    }
    assert!(observed_residues.into_iter().all(|seen| seen));
    assert!(saw_true_sparse);
    assert_eq!(workspace.committed_length(), 13);
    assert_close(
        &read_f16(&workspace.compressed_index_keys),
        oracle_section(&fixture, &oracle, "compressed_cache"),
        5e-4,
        0.0,
    );
    assert_close(
        &read_f16(&workspace.key_cache),
        oracle_section(&fixture, &oracle, "key_cache"),
        1e-3,
        0.0,
    );
    assert_close(
        &read_f16(&workspace.value_cache),
        oracle_section(&fixture, &oracle, "value_cache"),
        1e-3,
        0.0,
    );
    assert_eq!(
        fixture["steps"][11]["newly_completed_third_block_selected"],
        true
    );
    assert_eq!(
        fixture["steps"][12]["newly_completed_third_block_selected"],
        true
    );
}

#[test]
fn command_ownership_abandon_poison_reset_and_capacity_are_strict() {
    let Some(ctx) = context() else { return };
    let geometry = test_geometry(4);
    let weights = test_weights(&ctx, geometry);
    let input_values = values(geometry.hidden_size, 700, 0.01);
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![geometry.hidden_size as u64],
        GgmlType::F32,
    )
    .unwrap();
    let destination = MetalTensor::zeros_f32(&ctx, vec![geometry.hidden_size as u64]).unwrap();
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

    let first_command = ctx.queue.commandBuffer().unwrap();
    let first_encoder = KernelEncoder::begin(&first_command);
    let read = encode_qwen_sparse_attention_text(
        &ctx,
        &first_encoder,
        &input,
        weights.borrowed(),
        &mut workspace,
    )
    .unwrap();
    let other_command = ctx.queue.commandBuffer().unwrap();
    let other_encoder = KernelEncoder::begin(&other_command);
    assert!(
        read.output()
            .encode_copy_to(&ctx, &other_encoder, &destination)
            .is_err()
    );
    other_encoder.end();
    drop(other_command);
    first_encoder.end();
    drop(read);
    assert!(workspace.release_after().is_err());
    workspace.state_poisoned = true;
    unsafe { workspace.abandon_uncommitted().unwrap() };
    assert_eq!(workspace.committed_length(), 0);
    assert!(!workspace.is_poisoned());

    let concurrent_command = ctx.queue.commandBuffer().unwrap();
    let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
    assert!(
        encode_qwen_sparse_attention_text(
            &ctx,
            &concurrent_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
        )
        .is_err()
    );
    concurrent_encoder.end();

    let owner_command = ctx.queue.commandBuffer().unwrap();
    let owner_encoder = KernelEncoder::begin(&owner_command);
    let owner_read = encode_qwen_sparse_attention_text(
        &ctx,
        &owner_encoder,
        &input,
        weights.borrowed(),
        &mut workspace,
    )
    .unwrap();
    owner_encoder.end();
    drop(owner_read);
    let second_command = ctx.queue.commandBuffer().unwrap();
    let second_encoder = KernelEncoder::begin(&second_command);
    assert!(
        encode_qwen_sparse_attention_text(
            &ctx,
            &second_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
        )
        .is_err()
    );
    second_encoder.end();
    unsafe { workspace.abandon_uncommitted().unwrap() };

    let poison_command = ctx.queue.commandBuffer().unwrap();
    let poison_encoder = KernelEncoder::begin(&poison_command);
    let poison_read = encode_qwen_sparse_attention_text(
        &ctx,
        &poison_encoder,
        &input,
        weights.borrowed(),
        &mut workspace,
    )
    .unwrap();
    poison_encoder.end();
    poison_command.commit();
    poison_command.waitUntilCompleted();
    drop(poison_read);
    write_i32_scalar(&workspace.selector_status, 17).unwrap();
    assert!(workspace.release_after().is_err());
    assert!(workspace.is_poisoned());
    assert!(encode_one_result(&ctx, &weights, &mut workspace, &input_values).is_err());
    workspace.reset().unwrap();
    assert!(!workspace.is_poisoned());

    for token in 0..geometry.capacity {
        let values = values(geometry.hidden_size, 900 + token, 0.01);
        encode_one(&ctx, &weights, &mut workspace, &values);
    }
    assert_eq!(workspace.committed_length(), geometry.capacity);
    assert!(encode_one_result(&ctx, &weights, &mut workspace, &input_values).is_err());
    assert_eq!(workspace.committed_length(), geometry.capacity);
}

fn encode_one_result(
    ctx: &MetalContext,
    weights: &TestWeights,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    input_values: &[f32],
) -> Result<(), Qwen4ExpQsaError> {
    let input = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(input_values),
        vec![weights.geometry.hidden_size as u64],
        GgmlType::F32,
    )
    .unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let result =
        encode_qwen_sparse_attention_text(ctx, &encoder, &input, weights.borrowed(), workspace);
    encoder.end();
    match result {
        Ok(read) => {
            command.commit();
            drop(read);
            workspace.release_after()
        }
        Err(error) => Err(error),
    }
}

#[test]
fn released_workspace_estimate_prices_large_f16_caches_and_logits() {
    let config = Qwen4ExpConfig::flash_next_reference();
    let geometry =
        QwenSparseAttentionMetalGeometry::from_config(&config, 3, config.context_length as usize)
            .unwrap();
    let estimate = geometry.checked_workspace_byte_estimate().unwrap();
    let allocations = geometry.workspace_logical_allocations().unwrap();
    assert_eq!(allocations.len(), 22);
    assert_eq!(
        allocations.iter().sum::<usize>(),
        estimate.total_workspace_bytes
    );
    let expected_cache = config.context_length as usize
        * config.attention.kv_heads as usize
        * config.attention.key_head_dim as usize
        * 2;
    assert_eq!(estimate.main_key_cache_bytes, expected_cache);
    assert_eq!(estimate.main_value_cache_bytes, expected_cache);
    assert_eq!(
        estimate.logits_bytes,
        geometry.output_width() * config.attention.query_heads as usize * 4
    );
    assert!(estimate.total_workspace_bytes > 2 * expected_cache);
}

#[test]
fn released_attention_launch_geometry_and_large_width_reduction_match_cpu() {
    let Some(ctx) = context() else { return };
    let config = Qwen4ExpConfig::flash_next_reference();
    let geometry = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 2_052).unwrap();
    assert_eq!(geometry.query_heads, 24);
    assert_eq!(geometry.kv_heads, 2);
    assert_eq!(geometry.head_dim, 256);
    assert_eq!(geometry.output_width(), 2_051);
    let workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

    let query = (0..geometry.query_width())
        .map(|index| ((index * 17 + 3) % 97) as f32 * 0.003 - 0.144)
        .collect::<Vec<_>>();
    let raw_gate = (0..geometry.query_width())
        .map(|index| ((index * 11 + 5) % 61) as f32 * 0.07 - 2.1)
        .collect::<Vec<_>>();
    let token_ids = (0..geometry.output_width())
        .map(|position| position as i32)
        .collect::<Vec<_>>();
    let cache_elements = geometry.output_width() * geometry.kv_width();
    let mut key_cache = Vec::with_capacity(cache_elements);
    let mut value_cache = Vec::with_capacity(cache_elements);
    for position in 0..geometry.output_width() {
        for kv_head in 0..geometry.kv_heads {
            for lane in 0..geometry.head_dim {
                let key =
                    ((position * 7 + kv_head * 13 + lane * 3 + 1) % 113) as f32 * 0.004 - 0.224;
                let value =
                    ((position * 5 + kv_head * 19 + lane * 11 + 2) % 127) as f32 * 0.003 - 0.189;
                key_cache.push(f16::from_f32(key).to_f32());
                value_cache.push(f16::from_f32(value).to_f32());
            }
        }
    }
    write_f32_tensor(&workspace.query, &query);
    write_f32_tensor(&workspace.raw_gate, &raw_gate);
    write_i32_tensor(&workspace.token_ids, &token_ids);
    write_f16_prefix(&workspace.key_cache, &key_cache);
    write_f16_prefix(&workspace.value_cache, &value_cache);

    let mut expected = vec![0.0; geometry.query_width()];
    let queries_per_kv = geometry.query_heads / geometry.kv_heads;
    let scale = 1.0 / (geometry.head_dim as f32).sqrt();
    for query_head in 0..geometry.query_heads {
        let kv_head = query_head / queries_per_kv;
        let query_row =
            &query[query_head * geometry.head_dim..(query_head + 1) * geometry.head_dim];
        let mut logits = Vec::with_capacity(geometry.output_width());
        for position in 0..geometry.output_width() {
            let key_start = position * geometry.kv_width() + kv_head * geometry.head_dim;
            let dot = query_row
                .iter()
                .zip(&key_cache[key_start..key_start + geometry.head_dim])
                .map(|(&q, &k)| q * k)
                .sum::<f32>();
            logits.push(dot * scale);
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let masses = logits
            .iter()
            .map(|&logit| (logit - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = masses.iter().sum::<f32>();
        for lane in 0..geometry.head_dim {
            let accumulator = masses
                .iter()
                .enumerate()
                .map(|(position, &mass)| {
                    let value_start = position * geometry.kv_width() + kv_head * geometry.head_dim;
                    value_cache[value_start + lane] * mass
                })
                .sum::<f32>();
            let index = query_head * geometry.head_dim + lane;
            expected[index] = accumulator / denominator / (1.0 + (-raw_gate[index]).exp());
        }
    }
    assert!(expected.iter().all(|value| value.is_finite()));

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    encode_attention_logits(&ctx, &encoder, &workspace, geometry.output_width()).unwrap();
    encode_attention_softmax_value(&ctx, &encoder, &workspace, geometry.output_width()).unwrap();
    let census = crate::metal::dispatch_census_take();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );

    assert_eq!(census.len(), 2);
    let logits_launch = &census[0];
    assert_eq!(
        logits_launch.kernel,
        "kernel_qwen4exp_qsa_attention_logits_f16"
    );
    assert_eq!(logits_launch.grid_width, 6_153);
    assert_eq!(logits_launch.grid_height, 1);
    assert_eq!(logits_launch.threads_width, 256);
    let value_launch = &census[1];
    assert_eq!(
        value_launch.kernel,
        "kernel_qwen4exp_qsa_attention_softmax_value_f16"
    );
    assert_eq!(value_launch.grid_width, 24);
    assert_eq!(value_launch.grid_height, 1);
    assert_eq!(value_launch.threads_width, 256);

    let actual = read_f32(&workspace.attention);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_close(&actual, &expected, 3e-5, 3e-5);
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_QSA_GGUF to the pinned full release"]
fn released_layer_three_dense_packed_matches_scalar_rows_and_state() {
    const MAX_TOKENS: usize = 33;
    let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = MetalContext::new().expect("initialize Metal");
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
    let admitted = plan.admit(ctx.memory_signals()).unwrap();
    let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
    let bound = QwenSparseAttentionMetalWeights::bind(realized.weights(), 3, 64).unwrap();
    assert_eq!(bound.index_key.dtype, GgmlType::BF16);
    for tensor in [bound.query, bound.key, bound.value, bound.output] {
        assert_eq!(tensor.dtype, GgmlType::Q8_0);
    }
    let geometry = bound.geometry;
    let weights = TestWeights {
        geometry,
        query: bound.query.clone(),
        key: bound.key.clone(),
        value: bound.value.clone(),
        output: bound.output.clone(),
        query_norm: bound.query_norm.clone(),
        key_norm: bound.key_norm.clone(),
        index_query: bound.index_query.clone(),
        index_key: bound.index_key.clone(),
        index_query_norm: bound.index_query_norm.clone(),
        index_key_norm: bound.index_key_norm.clone(),
    };
    let inputs = values(MAX_TOKENS * geometry.hidden_size, 2_311, 0.001_7);
    let serial = serial_dense_trace(&ctx, &weights, &inputs, MAX_TOKENS);

    for tokens in [2_usize, 8, 16, 33] {
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, tokens).unwrap();
        let (actual, census) =
            crate::metal_forward::with_matmat_bf16_bfloat_act_override(true, || {
                encode_dense_packed_chunk(
                    &ctx,
                    &weights,
                    &mut workspace,
                    &scratch,
                    &inputs[..tokens * geometry.hidden_size],
                    0,
                    tokens,
                )
            });
        let expected = &serial.outputs[..tokens * geometry.hidden_size];
        for token in 0..tokens {
            let start = token * geometry.hidden_size;
            assert_similarity(
                &format!("released dense packed QSA N={tokens} token={token}"),
                &actual[start..start + geometry.hidden_size],
                &expected[start..start + geometry.hidden_size],
                9e-4,
                0.99999965,
                1e-4,
            );
        }
        let expected_state = &serial.states[tokens - 1];
        assert_similarity(
            &format!("released dense packed QSA N={tokens} pending index state"),
            &read_f32(&workspace.pending_index_keys),
            &expected_state.pending_index_keys,
            2e-6,
            0.99999999,
            1e-6,
        );
        let completed_index_elements = expected_state.compressed_index_keys.len();
        assert_close(
            &read_f16(&workspace.compressed_index_keys)[..completed_index_elements],
            &expected_state.compressed_index_keys,
            3e-3,
            3e-3,
        );
        let cache_elements = expected_state.key_cache.len();
        assert_close(
            &read_f16(&workspace.key_cache)[..cache_elements],
            &expected_state.key_cache,
            3e-3,
            3e-3,
        );
        assert_close(
            &read_f16(&workspace.value_cache)[..cache_elements],
            &expected_state.value_cache,
            3e-3,
            3e-3,
        );
        assert_eq!(workspace.committed_length(), tokens);
        assert_eq!(
            read_i32_scalar(&workspace.selected_count).unwrap(),
            (tokens / geometry.ratio) as i32
        );

        let names = census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        let expected_q8 = match tokens {
            8 => "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
            16 => "kernel_mat_mat_q8_0_f32_n16",
            _ => "kernel_mat_mat_q8_0_f32",
        };
        assert_eq!(
            names.iter().filter(|&&name| name == expected_q8).count(),
            4,
            "released N={tokens} Q8 route: {names:?}"
        );
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "kernel_mat_mat_bf16_f32")
                .count(),
            1,
            "released N={tokens} BF16 route: {names:?}"
        );
    }
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_QSA_GGUF to the pinned full release"]
fn released_layer_three_position_zero_matches_cpu_quantized_oracle() {
    let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = MetalContext::new().expect("initialize Metal");
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
    let admitted = plan.admit(ctx.memory_signals()).unwrap();
    let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
    let metal_weights = realized.weights();

    let mut qsa_layers = 0;
    for layer in 0..48 {
        let qsa = QwenSparseAttentionMetalWeights::bind(metal_weights, layer, 4);
        if layer % 4 == 3 {
            qsa.unwrap();
            assert!(
                GatedDeltaNetMetalWeights::bind(metal_weights, layer).is_err(),
                "QSA layer {layer} accepted as GDN"
            );
            qsa_layers += 1;
        } else {
            assert!(qsa.is_err(), "GDN layer {layer} accepted as QSA");
        }
    }
    assert_eq!(qsa_layers, 12);

    let weights = QwenSparseAttentionMetalWeights::bind(metal_weights, 3, 4).unwrap();
    let geometry = weights.geometry;
    let input = values(geometry.hidden_size, 1_003, 0.01);
    let dequant = |name: &str| {
        let desc = gguf.find(name).unwrap();
        crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
    };

    let index_query_weight = dequant("blk.3.indexer.q_proj.weight");
    let index_query_raw = real_mat_vec(&index_query_weight, &input, geometry.hidden_size);
    drop(index_query_weight);
    let index_key_weight = dequant("blk.3.indexer.k_proj.weight");
    let index_key_raw = real_mat_vec(&index_key_weight, &input, geometry.hidden_size);
    drop(index_key_weight);
    let index_query_norm = dequant("blk.3.indexer.q_norm.weight");
    let index_query = rmsnorm_heads_position_zero(
        &index_query_raw,
        geometry.index_query_heads,
        geometry.index_head_dim,
        &index_query_norm,
        geometry.eps,
    );
    drop(index_query_norm);
    let index_key_norm = dequant("blk.3.indexer.k_norm.weight");
    assert_eq!(index_key_norm.len(), geometry.index_head_dim);
    drop(index_key_norm);

    let query_weight = dequant("blk.3.attn_q.weight");
    let query_gate = real_mat_vec(&query_weight, &input, geometry.hidden_size);
    drop(query_weight);
    let mut query_raw = Vec::with_capacity(geometry.query_width());
    let mut raw_gate = Vec::with_capacity(geometry.query_width());
    for head in 0..geometry.query_heads {
        let start = head * geometry.head_dim * 2;
        query_raw.extend_from_slice(&query_gate[start..start + geometry.head_dim]);
        raw_gate.extend_from_slice(
            &query_gate[start + geometry.head_dim..start + 2 * geometry.head_dim],
        );
    }
    drop(query_gate);
    let query_norm = dequant("blk.3.attn_q_norm.weight");
    let query = rmsnorm_heads_position_zero(
        &query_raw,
        geometry.query_heads,
        geometry.head_dim,
        &query_norm,
        geometry.eps,
    );
    drop(query_norm);

    let key_weight = dequant("blk.3.attn_k.weight");
    let key_raw = real_mat_vec(&key_weight, &input, geometry.hidden_size);
    drop(key_weight);
    let key_norm = dequant("blk.3.attn_k_norm.weight");
    let key = rmsnorm_heads_position_zero(
        &key_raw,
        geometry.kv_heads,
        geometry.head_dim,
        &key_norm,
        geometry.eps,
    );
    drop(key_norm);
    let value_weight = dequant("blk.3.attn_v.weight");
    let value = real_mat_vec(&value_weight, &input, geometry.hidden_size);
    drop(value_weight);

    let key_cache = key
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let value_cache = value
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let mut attention = vec![0.0; geometry.query_width()];
    let queries_per_kv = geometry.query_heads / geometry.kv_heads;
    for query_head in 0..geometry.query_heads {
        let kv_head = query_head / queries_per_kv;
        for lane in 0..geometry.head_dim {
            let query_index = query_head * geometry.head_dim + lane;
            attention[query_index] = value_cache[kv_head * geometry.head_dim + lane]
                / (1.0 + (-raw_gate[query_index]).exp());
        }
    }
    let output_weight = dequant("blk.3.attn_output.weight");
    let expected_output = real_mat_vec(&output_weight, &attention, geometry.query_width());
    drop(output_weight);

    let input_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input),
        vec![geometry.hidden_size as u64],
        GgmlType::F32,
    )
    .unwrap();
    let copied_output = MetalTensor::zeros_f32(&ctx, vec![geometry.hidden_size as u64]).unwrap();
    let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let read =
        encode_qwen_sparse_attention_text(&ctx, &encoder, &input_gpu, weights, &mut workspace)
            .unwrap();
    read.output()
        .encode_copy_to(&ctx, &encoder, &copied_output)
        .unwrap();
    encoder.end();
    command.commit();
    drop(read);
    workspace.release_after().unwrap();

    assert_close(&read_f32(&copied_output), &expected_output, 5e-2, 4e-3);
    assert_close(
        &read_f32(&workspace.index_query_raw),
        &index_query_raw,
        2e-2,
        3e-3,
    );
    assert_close(&read_f32(&workspace.index_query), &index_query, 3e-3, 3e-3);
    assert_close(
        &read_f32(&workspace.pending_index_keys)[..geometry.index_head_dim],
        &index_key_raw,
        2e-2,
        3e-3,
    );
    assert_close(&read_f32(&workspace.query), &query, 3e-3, 3e-3);
    assert_close(&read_f32(&workspace.raw_gate), &raw_gate, 2e-2, 3e-3);
    assert_close(&read_f32(&workspace.key), &key, 3e-3, 3e-3);
    assert_close(&read_f32(&workspace.value), &value, 2e-2, 3e-3);
    let expected_ids = std::iter::once(0)
        .chain(std::iter::repeat_n(-1, geometry.output_width() - 1))
        .collect::<Vec<_>>();
    assert_eq!(read_i32(&workspace.token_ids), expected_ids);
    assert_close(
        &read_f16(&workspace.key_cache)[..geometry.kv_width()],
        &key_cache,
        3e-3,
        3e-3,
    );
    assert_close(
        &read_f16(&workspace.value_cache)[..geometry.kv_width()],
        &value_cache,
        3e-3,
        3e-3,
    );
}
