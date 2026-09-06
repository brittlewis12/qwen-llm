//! Shared fixtures and helpers for the Metal unit tests.

#![allow(dead_code)]

use super::*;

pub(super) struct LeaseFixture(pub(super) PathBuf);

impl LeaseFixture {
    pub(super) fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(PathBuf::from(format!(
            "/tmp/qwen-metal-lease-test-{}-{id}.lock",
            std::process::id()
        )))
    }
}

impl Drop for LeaseFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) fn offset_tensor(
    ctx: &MetalContext,
    prefix: usize,
    data: &[u8],
    suffix: usize,
    shape: Vec<u64>,
    dtype: GgmlType,
) -> MetalTensor {
    let mut bytes = vec![0xA5; prefix];
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len() + suffix, 0x5A);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("offset tensor backing"),
        offset: prefix as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

pub(super) fn tensor_f32_at_offset(tensor: &MetalTensor) -> Vec<f32> {
    let n = tensor.n_elements() as usize;
    unsafe {
        let src = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(src, n).to_vec()
    }
}

pub(super) fn tensor_backing_bytes(tensor: &MetalTensor) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<u8>(),
            tensor.buffer.length(),
        )
        .to_vec()
    }
}

pub(super) fn assert_offset_guards(tensor: &MetalTensor, prefix: usize, suffix: usize) {
    let bytes = tensor_backing_bytes(tensor);
    assert!(bytes[..prefix].iter().all(|&byte| byte == 0xA5));
    assert!(
        bytes[bytes.len() - suffix..]
            .iter()
            .all(|&byte| byte == 0x5A)
    );
}

pub(super) fn synthetic_q8_0_bank(n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(n_in.is_multiple_of(32));
    let blocks_per_row = n_in / 32;
    let mut bytes = Vec::with_capacity(n_out * blocks_per_row * 34);
    let mut decoded = vec![0.0f32; n_in * n_out];
    for row in 0..n_out {
        for block_index in 0..blocks_per_row {
            let ordinal = row * blocks_per_row + block_index;
            let sign = if ordinal.is_multiple_of(3) { -1.0 } else { 1.0 };
            let scale = sign * (ordinal % 7 + 1) as f32 / 512.0;
            let stored_scale = half::f16::from_f32(scale);
            bytes.extend_from_slice(&stored_scale.to_bits().to_le_bytes());
            for lane in 0..32 {
                let quant = ((ordinal * 13 + lane * 7 + 5) % 63) as i8 - 31;
                bytes.push(quant as u8);
                decoded[row * n_in + block_index * 32 + lane] =
                    stored_scale.to_f32() * f32::from(quant);
            }
        }
    }
    (bytes, decoded)
}

pub(super) fn synthetic_q8_0_bytes(n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in.is_multiple_of(32));
    let blocks_per_row = n_in / 32;
    let mut bytes = Vec::with_capacity(n_out * blocks_per_row * 34);
    for row in 0..n_out {
        for block_index in 0..blocks_per_row {
            let ordinal = row * blocks_per_row + block_index;
            let sign = if ordinal.is_multiple_of(3) { -1.0 } else { 1.0 };
            let scale = sign * (ordinal % 7 + 1) as f32 / 512.0;
            let stored_scale = half::f16::from_f32(scale);
            bytes.extend_from_slice(&stored_scale.to_bits().to_le_bytes());
            for lane in 0..32 {
                let quant = ((ordinal * 13 + lane * 7 + 5) % 63) as i8 - 31;
                bytes.push(quant as u8);
            }
        }
    }
    bytes
}

pub(super) fn f32_f64_differential(actual: &[f32], reference: &[f64]) -> (f64, f64, f64) {
    assert_eq!(actual.len(), reference.len());
    let mut difference_sq = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_difference = 0.0f64;
    let mut max_reference = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual);
        let difference = actual - reference;
        difference_sq += difference * difference;
        actual_sq += actual * actual;
        reference_sq += reference * reference;
        dot += actual * reference;
        max_difference = max_difference.max(difference.abs());
        max_reference = max_reference.max(reference.abs());
    }
    (
        (difference_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt(),
        max_difference / max_reference.max(1.0),
        dot / (actual_sq * reference_sq).sqrt(),
    )
}

pub(super) fn f32_differential(actual: &[f32], reference: &[f32]) -> (f64, f64, f64) {
    assert_eq!(actual.len(), reference.len());
    let mut difference_sq = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_difference = 0.0f64;
    let mut max_reference = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let difference = actual - reference;
        difference_sq += difference * difference;
        actual_sq += actual * actual;
        reference_sq += reference * reference;
        dot += actual * reference;
        max_difference = max_difference.max(difference.abs());
        max_reference = max_reference.max(reference.abs());
    }
    (
        (difference_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt(),
        max_difference / max_reference.max(1.0),
        dot / (actual_sq * reference_sq).sqrt(),
    )
}

pub(super) fn synthetic_dense_linear_bank(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
) -> (Vec<u8>, Vec<f32>) {
    let source: Vec<f32> = (0..n_in * n_out)
        .map(|index| ((index * 23 + 7) % 97) as f32 * 0.003 - 0.14)
        .collect();
    let mut bytes = Vec::new();
    let mut decoded = Vec::with_capacity(source.len());
    for value in source {
        match dtype {
            GgmlType::F32 => {
                bytes.extend_from_slice(&value.to_le_bytes());
                decoded.push(value);
            }
            GgmlType::F16 => {
                let stored = half::f16::from_f32(value);
                bytes.extend_from_slice(&stored.to_bits().to_le_bytes());
                decoded.push(stored.to_f32());
            }
            GgmlType::BF16 => {
                let stored = half::bf16::from_f32(value);
                bytes.extend_from_slice(&stored.to_bits().to_le_bytes());
                decoded.push(stored.to_f32());
            }
            _ => panic!("unsupported synthetic dense dtype {dtype:?}"),
        }
    }
    (bytes, decoded)
}

pub(super) fn encode_q6_k_block(d: f32, seed: usize) -> ([u8; 210], [f32; 256]) {
    let mut block = [0u8; 210];
    let mut decoded = [0.0f32; 256];
    for scale_index in 0..16 {
        let scale = ((seed + scale_index * 3) % 15) as i8 - 7;
        block[192 + scale_index] = scale as u8;
    }
    block[208..210].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    let stored_d = half::f16::from_f32(d).to_f32();
    for (i, value) in decoded.iter_mut().enumerate() {
        let quant = ((seed * 11 + i * 7) % 64) as u8;
        let half_index = i / 128;
        let half_offset = i % 128;
        let ql_index = 64 * half_index + half_offset % 64;
        if half_offset < 64 {
            block[ql_index] = (block[ql_index] & 0xF0) | (quant & 0x0F);
        } else {
            block[ql_index] = (block[ql_index] & 0x0F) | ((quant & 0x0F) << 4);
        }
        let qh_index = 128 + 32 * half_index + half_offset % 32;
        let qh_shift = 2 * (half_offset / 32);
        block[qh_index] |= (quant >> 4) << qh_shift;
        let scale = block[192 + i / 16] as i8;
        *value = stored_d * f32::from(scale) * (f32::from(quant) - 32.0);
    }
    (block, decoded)
}

pub(super) fn encode_iq4_nl_block(d: f32, seed: usize) -> ([u8; 18], [f32; 32]) {
    const VALUES: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    let mut block = [0u8; 18];
    let mut decoded = [0.0f32; 32];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    let stored_d = half::f16::from_f32(d).to_f32();
    for lane in 0..16 {
        let low = (seed + lane * 3) % 16;
        let high = (seed * 5 + lane * 7 + 1) % 16;
        block[2 + lane] = low as u8 | ((high as u8) << 4);
        decoded[lane] = stored_d * VALUES[low];
        decoded[16 + lane] = stored_d * VALUES[high];
    }
    (block, decoded)
}

pub(super) fn encode_iq4_xs_block(d: f32, seed: usize) -> [u8; 136] {
    let mut block = [0u8; 136];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    let mut scales_h = 0u16;
    for subblock in 0..8 {
        let scale = (seed * 11 + subblock * 7 + 3) % 64;
        block[4 + subblock / 2] |= ((scale & 0x0f) as u8) << (4 * (subblock % 2));
        scales_h |= ((scale >> 4) as u16) << (2 * subblock);
        for lane in 0..16 {
            let low = (seed + subblock * 5 + lane * 3) % 16;
            let high = (seed * 7 + subblock * 3 + lane * 5 + 1) % 16;
            block[8 + subblock * 16 + lane] = low as u8 | ((high as u8) << 4);
        }
    }
    block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    block
}

pub(super) fn synthetic_iq4_xs_bank(
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    seed: usize,
) -> Vec<u8> {
    assert!(n_in.is_multiple_of(256));
    let blocks_per_row = n_in / 256;
    let mut bytes = Vec::with_capacity(n_expert * n_out * blocks_per_row * 136);
    for expert in 0..n_expert {
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let d = sign * (ordinal % 5 + 1) as f32 / 65_536.0;
                bytes.extend_from_slice(&encode_iq4_xs_block(d, ordinal));
            }
        }
    }
    bytes
}

pub(super) fn synthetic_iq4_nl_bank(
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    seed: usize,
) -> Vec<u8> {
    assert!(n_in.is_multiple_of(32));
    let blocks_per_row = n_in / 32;
    let mut bytes = Vec::with_capacity(n_expert * n_out * blocks_per_row * 18);
    for expert in 0..n_expert {
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let d = sign * (ordinal % 7 + 1) as f32 / 16_384.0;
                bytes.extend_from_slice(&encode_iq4_nl_block(d, ordinal).0);
            }
        }
    }
    bytes
}

pub(super) fn dequant_expert(
    bank: &[u8],
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    expert: usize,
) -> Vec<f32> {
    let (block_elements, block_bytes) = dtype.storage_layout().unwrap();
    let row_bytes = n_in / block_elements as usize * block_bytes as usize;
    let expert_bytes = n_out * row_bytes;
    let start = expert * expert_bytes;
    let desc = TensorDesc {
        name: format!("synthetic_{dtype:?}_expert_{expert}"),
        shape: vec![n_in as u64, n_out as u64],
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_bytes as u64,
    };
    crate::codec::dequant_to_f32(&desc, &bank[start..start + expert_bytes]).unwrap()
}

pub(super) fn assert_moe_oracle_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(candidate, reference)| (candidate - reference).abs())
        .fold(0.0f32, f32::max);
    let reference_scale = expected.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(candidate, reference)| *candidate as f64 * *reference as f64)
        .sum();
    let actual_norm = actual
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (actual_norm * expected_norm).max(1e-30);
    let relative_max = max_abs / reference_scale.max(1e-6);
    eprintln!("[{label}] max_abs={max_abs:.3e} relative_max={relative_max:.3e} cosine={cosine:.9}");
    assert!(cosine >= 0.999_999, "{label} cosine={cosine}");
    assert!(relative_max <= 2e-4, "{label} relative_max={relative_max}");
}

pub(super) fn run_iq4_moe_decode_case(
    ctx: &MetalContext,
    n_in: usize,
    n_ffn: usize,
    n_hidden: usize,
    n_expert: usize,
    top_idx: &[i32],
) {
    let topk = top_idx.len();
    let gate_bytes = synthetic_iq4_xs_bank(n_in, n_ffn, n_expert, 17);
    let up_bytes = synthetic_iq4_xs_bank(n_in, n_ffn, n_expert, 10_003);
    let down_bytes = synthetic_iq4_nl_bank(n_ffn, n_hidden, n_expert, 20_011);
    let input = (0..n_in)
        .map(|index| ((index * 37 + index / 11 * 5 + 3) % 257) as f32 * 0.0005 - 0.064)
        .collect::<Vec<_>>();

    let mut expected_inner = vec![0.0f32; topk * n_ffn];
    for (slot, &expert) in top_idx.iter().enumerate() {
        let expert = expert as usize;
        let gate_weight = dequant_expert(&gate_bytes, GgmlType::IQ4_XS, n_in, n_ffn, expert);
        let gate = crate::forward::mat_vec_pub(&gate_weight, n_in, n_ffn, &input);
        drop(gate_weight);
        let up_weight = dequant_expert(&up_bytes, GgmlType::IQ4_XS, n_in, n_ffn, expert);
        let up = crate::forward::mat_vec_pub(&up_weight, n_in, n_ffn, &input);
        for row in 0..n_ffn {
            let gate_value = gate[row];
            expected_inner[slot * n_ffn + row] = gate_value / (1.0 + (-gate_value).exp()) * up[row];
        }
    }

    let gate = MetalTensor::from_bytes(
        ctx,
        &gate_bytes,
        vec![n_in as u64, n_ffn as u64, n_expert as u64],
        GgmlType::IQ4_XS,
    )
    .unwrap();
    let up = MetalTensor::from_bytes(
        ctx,
        &up_bytes,
        vec![n_in as u64, n_ffn as u64, n_expert as u64],
        GgmlType::IQ4_XS,
    )
    .unwrap();
    let down = MetalTensor::from_bytes(
        ctx,
        &down_bytes,
        vec![n_ffn as u64, n_hidden as u64, n_expert as u64],
        GgmlType::IQ4_NL,
    )
    .unwrap();
    let input_gpu = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&input),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let top_idx_gpu = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(top_idx),
        vec![topk as u64],
        GgmlType::I32,
    )
    .unwrap();
    let inner_gpu = MetalTensor::zeros_f32(ctx, vec![(topk * n_ffn) as u64]).unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_swiglu_iq4_xs_f32(
            ctx,
            encoder,
            &gate,
            &up,
            &input_gpu,
            &top_idx_gpu,
            &inner_gpu,
            n_in,
            n_ffn,
            n_expert,
            topk,
        )
    })
    .unwrap();
    let actual_inner = read_back_f32(&inner_gpu.buffer, topk * n_ffn);
    assert_moe_oracle_close("moe-iq4-xs-swiglu", &actual_inner, &expected_inner);

    let mut expected_down = vec![0.0f32; topk * n_hidden];
    for (slot, &expert) in top_idx.iter().enumerate() {
        let down_weight = dequant_expert(
            &down_bytes,
            GgmlType::IQ4_NL,
            n_ffn,
            n_hidden,
            expert as usize,
        );
        let output = crate::forward::mat_vec_pub(
            &down_weight,
            n_ffn,
            n_hidden,
            &actual_inner[slot * n_ffn..(slot + 1) * n_ffn],
        );
        expected_down[slot * n_hidden..(slot + 1) * n_hidden].copy_from_slice(&output);
    }
    let expert_out = MetalTensor::zeros_f32(ctx, vec![(topk * n_hidden) as u64]).unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_down_iq4_nl_f32(
            ctx,
            encoder,
            &down,
            &inner_gpu,
            &top_idx_gpu,
            &expert_out,
            n_ffn,
            n_hidden,
            n_expert,
            topk,
        )
    })
    .unwrap();
    let actual_down = read_back_f32(&expert_out.buffer, topk * n_hidden);
    assert_moe_oracle_close("moe-iq4-nl-down", &actual_down, &expected_down);
    let fast_expert_out = MetalTensor::zeros_f32(ctx, vec![(topk * n_hidden) as u64]).unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_down_iq4_nl_f32_fast(
            ctx,
            encoder,
            &down,
            &inner_gpu,
            &top_idx_gpu,
            &fast_expert_out,
            n_ffn,
            n_hidden,
            n_expert,
            topk,
        )
    })
    .unwrap();
    let actual_fast_down = read_back_f32(&fast_expert_out.buffer, topk * n_hidden);
    assert_moe_oracle_close("moe-iq4-nl-down-fast", &actual_fast_down, &expected_down);

    let invalid_idx = [-1i32, n_expert as i32];
    let invalid_idx_gpu = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&invalid_idx),
        vec![invalid_idx.len() as u64],
        GgmlType::I32,
    )
    .unwrap();
    let invalid_inner_values = vec![7.0f32; invalid_idx.len() * n_ffn];
    let invalid_inner = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&invalid_inner_values),
        vec![invalid_inner_values.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_swiglu_iq4_xs_f32(
            ctx,
            encoder,
            &gate,
            &up,
            &input_gpu,
            &invalid_idx_gpu,
            &invalid_inner,
            n_in,
            n_ffn,
            n_expert,
            invalid_idx.len(),
        )
    })
    .unwrap();
    assert!(
        read_back_f32(&invalid_inner.buffer, invalid_inner_values.len())
            .iter()
            .all(|&value| value == 0.0)
    );

    let invalid_down_values = vec![7.0f32; invalid_idx.len() * n_hidden];
    let invalid_down = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&invalid_down_values),
        vec![invalid_down_values.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_down_iq4_nl_f32(
            ctx,
            encoder,
            &down,
            &invalid_inner,
            &invalid_idx_gpu,
            &invalid_down,
            n_ffn,
            n_hidden,
            n_expert,
            invalid_idx.len(),
        )
    })
    .unwrap();
    assert!(
        read_back_f32(&invalid_down.buffer, invalid_down_values.len())
            .iter()
            .all(|&value| value == 0.0)
    );
    let invalid_fast_down = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&invalid_down_values),
        vec![invalid_down_values.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(ctx, |encoder| {
        encode_moe_down_iq4_nl_f32_fast(
            ctx,
            encoder,
            &down,
            &invalid_inner,
            &invalid_idx_gpu,
            &invalid_fast_down,
            n_ffn,
            n_hidden,
            n_expert,
            invalid_idx.len(),
        )
    })
    .unwrap();
    assert!(
        read_back_f32(&invalid_fast_down.buffer, invalid_down_values.len())
            .iter()
            .all(|&value| value == 0.0)
    );
}

pub(super) fn encode_iq2_xs_block(d: f32, seed: usize) -> [u8; 74] {
    let mut block = [0u8; 74];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for word in 0..32usize {
        let grid = (seed * 97 + word * 53 + word * word * 3) % 512;
        let signs = (seed * 29 + word * 17 + 11) % 128;
        let packed = (grid | (signs << 9)) as u16;
        block[2 + 2 * word..4 + 2 * word].copy_from_slice(&packed.to_le_bytes());
    }
    for scale in 0..8usize {
        let low = (seed * 7 + scale * 3 + 1) % 16;
        let high = (seed * 11 + scale * 5 + 9) % 16;
        block[66 + scale] = (low | (high << 4)) as u8;
    }
    block
}

pub(super) fn mxfp4_scale(e: u8) -> f32 {
    let bits = match e {
        0 => 0x0020_0000,
        1 => 0x0040_0000,
        _ => u32::from(e - 1) << 23,
    };
    f32::from_bits(bits)
}

pub(super) fn encode_mxfp4_block(e: u8, indices: &[u8; 32]) -> [u8; 17] {
    let mut block = [0u8; 17];
    block[0] = e;
    for i in 0..16 {
        block[1 + i] = indices[i] | (indices[16 + i] << 4);
    }
    block
}

pub(super) fn run_mxfp4_f32_matrix_tile_k216_bucket_floor(
    columns: usize,
    bucket_counts: [usize; 2],
    max_batch: usize,
    min_conservative_gpu_saving: f64,
    min_conservative_wall_saving: f64,
    min_gpu_fraction: f64,
) {
    use std::time::Instant;

    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    let path = "/Users/tito/models/deepseek-v4-flash-0731-reap-k216/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[mxfp4-mm-floor] skipped: K216 fixture missing");
        return;
    }
    const N_IN: usize = 2_048;
    const N_OUT: usize = 4_096;
    const EXPERTS: usize = 216;
    let gguf = crate::gguf::GgufFile::open(path).expect("open K216 GGUF");
    let load_bank = |layer: usize| {
        let name = format!("blk.{layer}.ffn_down_exps.weight");
        let tensor = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(tensor.dtype, GgmlType::MXFP4);
        assert_eq!(
            tensor.shape,
            vec![N_IN as u64, N_OUT as u64, EXPERTS as u64]
        );
        MetalTensor::from_bytes(&ctx, gguf.slice(tensor), tensor.shape.clone(), tensor.dtype)
            .expect("copy MXFP4 expert bank")
    };
    let banks = [load_bank(26), load_bank(42)];
    let distributions = [(bucket_counts[0], columns), (bucket_counts[1], columns)].map(
        |(bucket_count, columns)| {
            let base = columns / bucket_count;
            let remainder = columns % bucket_count;
            (0..bucket_count)
                .map(|expert| base + usize::from(expert < remainder))
                .collect::<Vec<_>>()
        },
    );
    assert_eq!(distributions[0].iter().sum::<usize>(), columns);
    assert_eq!(distributions[1].iter().sum::<usize>(), columns);
    assert!(distributions.iter().flatten().all(|&count| count >= 16));

    let x_values = (0..max_batch * N_IN)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) / 19.0)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x_values),
        vec![N_IN as u64, max_batch as u64],
        GgmlType::F32,
    )
    .expect("MXFP4 floor input");
    let control_output = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, max_batch as u64])
        .expect("MXFP4 floor control output");
    let candidate_output = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, max_batch as u64])
        .expect("MXFP4 floor candidate output");
    let expert_bytes = banks[0].n_bytes() / EXPERTS as u64;
    assert_eq!(expert_bytes, (N_IN * N_OUT / 32 * 17) as u64);

    let check_weight = banks[0].view_bytes(0, vec![N_IN as u64, N_OUT as u64]);
    let check_command = ctx
        .queue
        .commandBuffer()
        .expect("MXFP4 production-K check command buffer");
    let check_encoder = KernelEncoder::begin(&check_command);
    for column in 0..max_batch {
        encode_mat_vec_mxfp4_f32(
            &ctx,
            &check_encoder,
            &check_weight,
            &input.view_subrange((column * N_IN) as u64, vec![N_IN as u64]),
            &control_output.view_subrange((column * N_OUT) as u64, vec![N_OUT as u64]),
            N_IN,
            N_OUT,
        )
        .expect("encode production-K scalar check");
    }
    encode_mat_mat_mxfp4_f32_mm64x32(
        &ctx,
        &check_encoder,
        &check_weight,
        &input,
        &candidate_output,
        N_IN,
        N_OUT,
        max_batch,
    )
    .expect("encode production-K matrix check");
    check_encoder.end();
    check_command.commit();
    check_command.waitUntilCompleted();
    assert!(check_command.error().is_none());
    let check_control = tensor_f32_at_offset(&control_output);
    let check_candidate = tensor_f32_at_offset(&candidate_output);
    let mut diff_sq = 0.0f64;
    let mut control_sq = 0.0f64;
    let mut candidate_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_control = 0.0f32;
    for (&got, &want) in check_candidate.iter().zip(&check_control) {
        let diff = f64::from(got) - f64::from(want);
        diff_sq += diff * diff;
        control_sq += f64::from(want) * f64::from(want);
        candidate_sq += f64::from(got) * f64::from(got);
        dot += f64::from(got) * f64::from(want);
        max_abs = max_abs.max((got - want).abs());
        max_control = max_control.max(want.abs());
    }
    let relative_rms = (diff_sq / control_sq).sqrt();
    let cosine = dot / (control_sq * candidate_sq).sqrt();
    let normalized_max = f64::from(max_abs / max_control.max(1.0));
    eprintln!(
        "[mxfp4-mm-floor] production-K relative_rms={relative_rms:.9} cosine={cosine:.9} normalized_max={normalized_max:.9}"
    );
    assert!(relative_rms <= 2.0e-5);
    assert!(cosine >= 0.999_999_9);
    assert!(normalized_max <= 1.0e-4);
    assert!(check_candidate.iter().all(|value| value.is_finite()));

    let run = |matrix: bool| {
        let started = Instant::now();
        let command = ctx
            .queue
            .commandBuffer()
            .expect("MXFP4 floor command buffer");
        let encoder = KernelEncoder::begin(&command);
        for (bank, counts) in banks.iter().zip(&distributions) {
            for (expert, &n_batch) in counts.iter().enumerate() {
                let weight = bank.view_bytes(
                    expert as u64 * expert_bytes,
                    vec![N_IN as u64, N_OUT as u64],
                );
                let x = input.view_subrange(0, vec![N_IN as u64, n_batch as u64]);
                if matrix {
                    let y = candidate_output.view_subrange(0, vec![N_OUT as u64, n_batch as u64]);
                    encode_mat_mat_mxfp4_f32_mm64x32(
                        &ctx, &encoder, &weight, &x, &y, N_IN, N_OUT, n_batch,
                    )
                    .expect("encode MXFP4 matrix floor");
                } else {
                    for column in 0..n_batch {
                        let x_row = x.view_subrange((column * N_IN) as u64, vec![N_IN as u64]);
                        let y_row = control_output
                            .view_subrange((column * N_OUT) as u64, vec![N_OUT as u64]);
                        encode_mat_vec_mxfp4_f32(
                            &ctx, &encoder, &weight, &x_row, &y_row, N_IN, N_OUT,
                        )
                        .expect("encode scalar MXFP4 floor");
                    }
                }
            }
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (
            started.elapsed().as_secs_f64() * 1e3,
            (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
        )
    };

    run(false);
    run(true);
    let a1 = run(false);
    let b1 = run(true);
    let b2 = run(true);
    let a2 = run(false);
    eprintln!(
        "[mxfp4-mm-floor] wall_ms A/B/B/A={:.3}/{:.3}/{:.3}/{:.3}",
        a1.0, b1.0, b2.0, a2.0
    );
    eprintln!(
        "[mxfp4-mm-floor] gpu_ms A/B/B/A={:.3}/{:.3}/{:.3}/{:.3}",
        a1.1, b1.1, b2.1, a2.1
    );
    let control_wall = (a1.0 + a2.0) / 2.0;
    let candidate_wall = (b1.0 + b2.0) / 2.0;
    let control_gpu = (a1.1 + a2.1) / 2.0;
    let candidate_gpu = (b1.1 + b2.1) / 2.0;
    let control_wall_spread = (a1.0 - a2.0).abs() / control_wall;
    let candidate_wall_spread = (b1.0 - b2.0).abs() / candidate_wall;
    let control_gpu_spread = (a1.1 - a2.1).abs() / control_gpu;
    let candidate_gpu_spread = (b1.1 - b2.1).abs() / candidate_gpu;
    let conservative_gpu_saving = a1.1.min(a2.1) - b1.1.max(b2.1);
    let conservative_wall_saving = a1.0.min(a2.0) - b1.0.max(b2.0);
    let gpu_fraction = 1.0 - candidate_gpu / control_gpu;
    eprintln!(
        "[mxfp4-mm-floor] medians wall={control_wall:.3}->{candidate_wall:.3} gpu={control_gpu:.3}->{candidate_gpu:.3} conservative_wall={conservative_wall_saving:.3} conservative_gpu={conservative_gpu_saving:.3} gpu_fraction={gpu_fraction:.4}"
    );
    assert!(control_wall_spread <= 0.05);
    assert!(candidate_wall_spread <= 0.05);
    assert!(control_gpu_spread <= 0.05);
    assert!(candidate_gpu_spread <= 0.05);
    assert!(b1.0 < a1.0.min(a2.0) && b2.0 < a1.0.min(a2.0));
    assert!(b1.1 < a1.1.min(a2.1) && b2.1 < a1.1.min(a2.1));
    assert!(conservative_gpu_saving >= min_conservative_gpu_saving);
    assert!(conservative_wall_saving >= min_conservative_wall_saving);
    assert!(gpu_fraction >= min_gpu_fraction);
}

pub(super) fn metal_test_context() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::EmptyLibrary | MetalError::NoDevice) => {
            let required = matches!(
                std::env::var("QWEN_REQUIRE_METAL_TESTS").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            );
            if required {
                panic!("Metal is required but unavailable");
            }
            None
        }
        Err(error) => panic!("Metal context: {error}"),
    }
}

pub(super) fn f32_desc(
    name: &str,
    shard_idx: usize,
    data_offset: u64,
    elements: u64,
) -> TensorDesc {
    TensorDesc {
        name: name.to_string(),
        shape: vec![elements],
        dtype: GgmlType::F32,
        shard_idx,
        data_offset,
        n_bytes: elements * 4,
    }
}

pub(super) fn assert_retained_plan_invariants(
    plan: &RetainedStoragePlan,
    requests: &[&TensorDesc],
    shard_mapped_lengths: &[usize],
) {
    for window in &plan.windows {
        assert_eq!(window.mmap_offset % plan.page_size as u64, 0);
        assert!(window.length > 0);
        assert_eq!(window.length % plan.page_size, 0);
        assert!(window.length <= plan.usable_window_length);
        let mapped_len = shard_mapped_lengths[window.shard_idx] as u64;
        assert!(window.mmap_offset + window.length as u64 <= mapped_len);
    }
    for entry in &plan.entries {
        assert_eq!(entry.n_bytes, requests[entry.request_index].n_bytes);
        match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let window = &plan.windows[window_index];
                assert_eq!(entry.shard_idx, window.shard_idx);
                assert_eq!(entry.data_offset, window.mmap_offset + buffer_offset);
                assert_eq!(buffer_offset % plan.required_alignment as u64, 0);
                assert!(buffer_offset + entry.n_bytes <= window.length as u64);
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                assert!(source_request_index < entry.request_index);
                let source = &plan.entries[source_request_index];
                assert_eq!(entry.shard_idx, source.shard_idx);
                assert_eq!(entry.data_offset, source.data_offset);
                assert_eq!(entry.n_bytes, source.n_bytes);
            }
            RetainedStorageDisposition::CopyFallback { .. } => {}
        }
    }
}

pub(super) fn memory_signals(
    recommended_max_bytes: u64,
    current_allocated_bytes: u64,
    process_limit_remaining_bytes: Option<u64>,
) -> MetalMemorySignals {
    MetalMemorySignals {
        recommended_max_bytes,
        current_allocated_bytes,
        process_limit_remaining_bytes,
    }
}

/// Helper: take an `encode_*` closure that produces a single F32
/// output buffer of length `n_out`, run it one-shot, and return the
/// readback. Common shape across the elementwise tests.
pub(super) fn one_shot_f32_out<F>(ctx: &MetalContext, n_out: usize, encode: F) -> Vec<f32>
where
    F: FnOnce(&KernelEncoder, &MetalTensor) -> Result<(), MetalError>,
{
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64]).unwrap();
    one_shot(ctx, |enc| encode(enc, &y_t)).unwrap();
    read_back_f32(&y_t.buffer, n_out)
}

#[allow(non_snake_case)]
pub(super) fn ffn_swiglu_q4_K_matches_unfused() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");

    // Find ffn_gate and ffn_up Q4_K tensors from layer 0.
    let gate = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
        .expect("no blk.0.ffn_gate.weight Q4_K");
    let up = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
        .expect("no blk.0.ffn_up.weight Q4_K");
    assert_eq!(gate.shape, up.shape, "gate/up shape mismatch");
    let n_in = gate.shape[0] as usize;
    let n_out = gate.shape[1] as usize;
    eprintln!(
        "[ffn-fused] gate={} up={} n_in={n_in} n_out={n_out}",
        gate.name, up.name
    );

    // Build inputs.
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let w_gate = MetalTensor::from_bytes(
        &ctx,
        g.slice(gate),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q4_K,
    )
    .unwrap();
    let w_up = MetalTensor::from_bytes(
        &ctx,
        g.slice(up),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q4_K,
    )
    .unwrap();

    // --- Unfused reference: 3 dispatches ---
    let gate_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
    let up_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
    let inner_ref_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_vec_q4_k_f32(&ctx, enc, &w_gate, &x_t, &gate_t, n_in, n_out)?;
        encode_mat_vec_q4_k_f32(&ctx, enc, &w_up, &x_t, &up_t, n_in, n_out)?;
        encode_silu_mul_f32(&ctx, enc, &gate_t, &up_t, &inner_ref_t)
    })
    .unwrap();
    let inner_ref = read_back_f32(&inner_ref_t.buffer, n_out);

    // --- Fused: 1 dispatch ---
    let inner_fused_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_ffn_swiglu_q4_K_f32(&ctx, enc, &w_gate, &w_up, &x_t, &inner_fused_t, n_in, n_out)
    })
    .unwrap();
    let inner_fused = read_back_f32(&inner_fused_t.buffer, n_out);

    let max_abs = inner_fused
        .iter()
        .zip(inner_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = inner_fused
        .iter()
        .zip(inner_ref.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = inner_fused
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = inner_ref
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    eprintln!("[ffn-fused] n_out={n_out} max|Δ|={max_abs:.2e} cos={cos:.6}");
    // The two paths do the same fp32 reductions in the same order
    // (both use simd_sum over the same 32 lanes per row, producing
    // identical totals). Difference should be ~0 except for the
    // silu compositional difference (silu computed before vs after
    // the float -> device write -> float read roundtrip; should
    // also be 0). Allow tiny tolerance for safety.
    assert!(
        cos > 0.9999,
        "fused FFN cos too low: {cos} (max|Δ|={max_abs})"
    );
    assert!(max_abs < 1e-3, "fused FFN diverged: max|Δ|={max_abs}");
}

pub(super) fn run_attn_v4_q8_kv_compare(
    label: &str,
    n_q: usize,
    n_kv: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
    group_tile: Option<usize>,
) {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let hd = 256usize;
    let kv_dim = n_kv * hd;
    let group = n_q / n_kv;
    assert_eq!(n_q % n_kv, 0);

    let q: Vec<f32> = (0..n_q * hd)
        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
        .collect();
    let k_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
        .collect();
    let v_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();

    let q_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&q),
        vec![(n_q * hd) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let k_f16 = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
    let v_f16 = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
    let k_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
    let v_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();

    let k_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k_f32),
        vec![k_f32.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let v_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v_f32),
        vec![v_f32.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(&ctx, |enc| {
        encode_scatter_offset_f32_to_f16_kv(
            &ctx,
            enc,
            &k_src_t,
            &v_src_t,
            &k_f16,
            &v_f16,
            0,
            k_f32.len(),
        )?;
        encode_scatter_offset_f32_to_q8_0_kv(
            &ctx,
            enc,
            &k_src_t,
            &v_src_t,
            &k_q8,
            &v_q8,
            0,
            k_f32.len(),
        )
    })
    .unwrap();

    let o_partial_f16 =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
    let ml_partial_f16 =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
    let out_f16 = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
    let o_partial_q8 =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
    let ml_partial_q8 =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
    let out_q8 = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

    let f16_result = || {
        one_shot(&ctx, |enc| {
            encode_attn_decode_v4_f32(
                &ctx,
                enc,
                &q_t,
                &k_f16,
                &v_f16,
                &o_partial_f16,
                &ml_partial_f16,
                &out_f16,
                n_q,
                n_kv,
                hd,
                n_pos,
                nwg,
                tile_c,
            )
        })
    };
    if let Some(tile) = group_tile {
        with_attn_v4_group_tile_override(tile, f16_result)
    } else {
        f16_result()
    }
    .unwrap();

    let q8_result = || {
        one_shot(&ctx, |enc| {
            encode_attn_decode_v4_f32(
                &ctx,
                enc,
                &q_t,
                &k_q8,
                &v_q8,
                &o_partial_q8,
                &ml_partial_q8,
                &out_q8,
                n_q,
                n_kv,
                hd,
                n_pos,
                nwg,
                tile_c,
            )
        })
    };
    if let Some(tile) = group_tile {
        with_attn_v4_group_tile_override(tile, q8_result)
    } else {
        q8_result()
    }
    .unwrap();

    let y_f16 = read_back_f32(&out_f16.buffer, n_q * hd);
    let y_q8 = read_back_f32(&out_q8.buffer, n_q * hd);
    let max_abs = y_f16
        .iter()
        .zip(y_q8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = y_f16
        .iter()
        .zip(y_q8.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = y_f16
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = y_q8.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-30);
    eprintln!(
        "[attn_v4_q8_kv {label}] group={group} n_pos={n_pos} nwg={nwg} C={tile_c} tile={:?} cos={cos:.6} max|Δ|={max_abs:.4}",
        group_tile,
    );
    assert!(cos > 0.9999, "q8 kv cos too low: {cos}");
    assert!(max_abs < 0.01, "q8 kv max|Δ| too high: {max_abs}");
}

pub(super) fn quantized_get_rows_fixture(path: &str, expected_dtype: GgmlType, bit_exact: bool) {
    assert!(
        std::path::Path::new(path).exists(),
        "required fixture missing: {path}"
    );
    let ctx = MetalContext::new().expect("metal context");
    let g = crate::gguf::GgufFile::open(path).expect("open fixture");
    let model = crate::loader::Model::from_gguf(&g).expect("load model");
    let desc = model.token_embd;
    assert_eq!(desc.dtype, expected_dtype);
    assert_eq!(desc.shape.len(), 2);
    let n_cols = desc.shape[0] as usize;
    let vocab = desc.shape[1] as usize;
    let ids = vec![0i32, (vocab - 1) as i32, 1, (vocab - 2) as i32, 17, 17];
    let n_rows = ids.len();
    let embed = MetalTensor::from_gguf_tensor(&ctx, desc, g.slice(desc)).expect("embedding");
    let ids_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![n_rows as u64],
        GgmlType::I32,
    )
    .expect("ids");
    let output = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).expect("output");
    one_shot(&ctx, |enc| {
        encode_get_rows_f32(&ctx, enc, &embed, &ids_t, &output, n_rows, n_cols)
    })
    .expect("get rows");
    let gpu = read_back_f32(&output.buffer, n_rows * n_cols);

    let tensor_bytes = g.slice(desc);
    let row_bytes = desc.n_bytes as usize / vocab;
    let mut expected = Vec::with_capacity(n_rows * n_cols);
    for &id in &ids {
        let row = id as usize;
        let mut row_desc = desc.clone();
        row_desc.name = format!("{}.row.{row}", desc.name);
        row_desc.shape = vec![n_cols as u64];
        row_desc.data_offset = 0;
        row_desc.n_bytes = row_bytes as u64;
        expected.extend(
            crate::codec::dequant_to_f32(
                &row_desc,
                &tensor_bytes[row * row_bytes..(row + 1) * row_bytes],
            )
            .expect("dequant row"),
        );
    }

    let mut max_abs = 0.0f32;
    let mut squared = 0.0f64;
    let mut dot = 0.0f64;
    let mut gpu_norm = 0.0f64;
    let mut cpu_norm = 0.0f64;
    for (&candidate, &reference) in gpu.iter().zip(expected.iter()) {
        assert!(candidate.is_finite());
        max_abs = max_abs.max((candidate - reference).abs());
        squared += ((candidate - reference) as f64).powi(2);
        dot += candidate as f64 * reference as f64;
        gpu_norm += (candidate as f64).powi(2);
        cpu_norm += (reference as f64).powi(2);
        if bit_exact {
            assert_eq!(candidate.to_bits(), reference.to_bits());
        }
    }
    let rmse = (squared / gpu.len() as f64).sqrt();
    let cosine = dot / (gpu_norm.sqrt() * cpu_norm.sqrt() + 1e-30);
    let repeated_a = &gpu[4 * n_cols..5 * n_cols];
    let repeated_b = &gpu[5 * n_cols..6 * n_cols];
    assert!(
        repeated_a
            .iter()
            .zip(repeated_b.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    eprintln!(
        concat!(
            "[quant-get-rows] dtype={:?} rows={} cols={} ",
            "max_abs={:.3e} rmse={:.3e} cos={:.10}"
        ),
        expected_dtype, n_rows, n_cols, max_abs, rmse, cosine,
    );
    assert!(max_abs <= 1e-6);
    assert!(rmse <= 1e-7);
    assert!(cosine >= 0.99999999);
}

/// CPU reference: forward::rope_in_place — applies NEOX-pairing partial
/// RoPE in place. We use it via the existing `forward::rope_in_place_pub`
/// helper added below.
pub(super) fn rope_neox_cpu_ref(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) {
    let pos = position as f32;
    let half = n_rot / 2;
    for hi in 0..n_heads {
        let h_off = hi * head_dim;
        for i in 0..half {
            let exponent = (2 * i) as f32 / n_rot as f32;
            let freq = pos / theta_base.powf(exponent);
            let (s, c) = freq.sin_cos();
            let a = buf[h_off + i];
            let b = buf[h_off + i + half];
            buf[h_off + i] = a * c - b * s;
            buf[h_off + i + half] = a * s + b * c;
        }
    }
}

pub(super) fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

pub(super) fn assert_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

pub(super) fn assert_bitwise_equal(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length mismatch");
    for (index, (expected, actual)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "{label} differs at [{index}]: expected={expected} actual={actual}"
        );
    }
}

pub(super) fn run_rope_pair_variant(
    ctx: &MetalContext,
    variant: &str,
    q_init: &[f32],
    k_init: &[f32],
    n_q: usize,
    n_k: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(q_init),
        vec![q_init.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let k_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(k_init),
        vec![k_init.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(ctx, |enc| match variant {
        "baseline" => encode_rope_neox_pair_f32(
            ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
        ),
        "sincos" => encode_rope_neox_pair_sincos_f32(
            ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
        ),
        "shared" => encode_rope_neox_pair_shared_f32(
            ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
        ),
        "minimax" => encode_rope_neox_pair_shared_minimax_f32(
            ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
        ),
        other => panic!("unknown RoPE pair variant {other}"),
    })
    .unwrap();
    (
        read_back_f32(&q_t.buffer, q_init.len()),
        read_back_f32(&k_t.buffer, k_init.len()),
    )
}

pub(super) fn run_rope_packed_pair_variant(
    ctx: &MetalContext,
    variant: &str,
    q_init: &[f32],
    k_init: &[f32],
    n_tokens: usize,
    n_q: usize,
    n_k: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(q_init),
        vec![q_init.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let k_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(k_init),
        vec![k_init.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(ctx, |enc| match variant {
        "baseline" => {
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                enc,
                &q_t,
                n_tokens,
                n_q,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            )?;
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                enc,
                &k_t,
                n_tokens,
                n_k,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            )
        }
        "paired" => encode_rope_neox_pair_f32_packed_consecutive(
            ctx,
            enc,
            &q_t,
            &k_t,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        ),
        "shared" => encode_rope_neox_pair_shared_f32_packed_consecutive(
            ctx,
            enc,
            &q_t,
            &k_t,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        ),
        "adaptive" => encode_rope_neox_pair_adaptive_f32_packed_consecutive(
            ctx,
            enc,
            &q_t,
            &k_t,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        ),
        "minimax" => encode_rope_neox_pair_shared_minimax_f32_packed_consecutive(
            ctx,
            enc,
            &q_t,
            &k_t,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        ),
        other => panic!("unknown packed RoPE pair variant {other}"),
    })
    .unwrap();
    (
        read_back_f32(&q_t.buffer, q_init.len()),
        read_back_f32(&k_t.buffer, k_init.len()),
    )
}

/// CPU reference for ssm_conv_silu — mirrors the conv block in
/// `forward::Forward::gdn_step`. Mutates `conv_buf` in place,
/// returns conv output (post-SiLU).
pub(super) fn ssm_conv_silu_cpu_ref(
    qkv_now: &[f32],
    conv_buf: &mut [f32],
    conv_w: &[f32],
    conv_dim: usize,
) -> Vec<f32> {
    const K: usize = 4;
    let kmin1 = K - 1;
    let mut conv_input = vec![0.0f32; K * conv_dim];
    for t in 0..kmin1 {
        conv_input[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&conv_buf[t * conv_dim..(t + 1) * conv_dim]);
    }
    conv_input[kmin1 * conv_dim..].copy_from_slice(qkv_now);

    let mut out = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let mut s = 0.0f32;
        for k in 0..K {
            s += conv_w[c * K + k] * conv_input[k * conv_dim + c];
        }
        out[c] = s / (1.0 + (-s).exp());
    }
    // Slide buffer (drop oldest, append current).
    for t in 0..kmin1 - 1 {
        for i in 0..conv_dim {
            conv_buf[t * conv_dim + i] = conv_buf[(t + 1) * conv_dim + i];
        }
    }
    conv_buf[(kmin1 - 1) * conv_dim..].copy_from_slice(qkv_now);
    out
}

/// CPU reference for rmsnorm_gated — per-head RMSNorm of `o` * silu(z).
pub(super) fn rmsnorm_gated_cpu_ref(
    o: &[f32],
    weight: &[f32],
    z: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n_heads * head_dim];
    for hi in 0..n_heads {
        let off = hi * head_dim;
        let sumsq: f32 = (0..head_dim).map(|i| o[off + i].powi(2)).sum();
        let scale = 1.0 / (sumsq / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            let zi = z[off + i];
            let silu_z = zi / (1.0 + (-zi).exp());
            y[off + i] = (o[off + i] * scale * weight[i]) * silu_z;
        }
    }
    y
}

/// CPU reference for the GDN-step kernel — mirrors the per-V-head
/// inner loop in `forward::Forward::gdn_step` exactly. Mutates
/// `state` in place and returns `out`.
pub(super) fn gdn_step_cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    n_v: usize,
    n_k: usize,
    hd: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_v * hd];
    for hi in 0..n_v {
        let hk = hi % n_k;
        let s_off = hi * hd * hd;
        let q_h = &q[hk * hd..(hk + 1) * hd];
        let k_h = &k[hk * hd..(hk + 1) * hd];
        let v_h = &v[hi * hd..(hi + 1) * hd];
        let g_h = g[hi].exp();
        let b_h = beta[hi];

        // Decay: S *= g_h
        for j in 0..hd * hd {
            state[s_off + j] *= g_h;
        }
        // s_k[dv] = sum_dk S[dv,dk] * k[dk]
        let mut sk = vec![0.0f32; hd];
        for dv in 0..hd {
            let mut s = 0.0f32;
            for dk in 0..hd {
                s += state[s_off + dv * hd + dk] * k_h[dk];
            }
            sk[dv] = s;
        }
        // Update: S[dv,dk] += beta * (v[dv] - sk[dv]) * k[dk]
        for dv in 0..hd {
            let coeff = b_h * (v_h[dv] - sk[dv]);
            for dk in 0..hd {
                state[s_off + dv * hd + dk] += coeff * k_h[dk];
            }
        }
        // Output is unscaled; the production GDN path folds the
        // 1/sqrt(head_dim) factor into RMSNormGated epsilon.
        for dv in 0..hd {
            let mut s = 0.0f32;
            for dk in 0..hd {
                s += state[s_off + dv * hd + dk] * q_h[dk];
            }
            out[hi * hd + dv] = s;
        }
    }
    out
}

pub(super) struct GdnStepVjpReference {
    pub(super) grad_q: Vec<f64>,
    pub(super) grad_k: Vec<f64>,
    pub(super) grad_v: Vec<f64>,
    pub(super) grad_decay: Vec<f64>,
    pub(super) grad_beta: Vec<f64>,
    pub(super) grad_state: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_step_decay_objective_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: &[f64],
    beta: &[f64],
    state: &[f64],
    grad_out: &[f64],
    grad_state_out: &[f64],
    n_v: usize,
    n_k: usize,
    head_dim: usize,
) -> f64 {
    let mut objective = 0.0f64;
    for hi in 0..n_v {
        let hk = hi % n_k;
        for dv in 0..head_dim {
            let vector_index = hi * head_dim + dv;
            let row_offset = vector_index * head_dim;
            let prediction: f64 = (0..head_dim)
                .map(|dk| decay[hi] * state[row_offset + dk] * k[hk * head_dim + dk])
                .sum();
            let residual = v[vector_index] - prediction;
            let correction = beta[hi] * residual;
            let mut output = 0.0f64;
            for dk in 0..head_dim {
                let state_out =
                    decay[hi] * state[row_offset + dk] + correction * k[hk * head_dim + dk];
                output += state_out * q[hk * head_dim + dk];
                objective += state_out * grad_state_out[row_offset + dk];
            }
            objective += output * grad_out[vector_index];
        }
    }
    objective
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_step_decay_vjp_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: &[f64],
    beta: &[f64],
    state: &[f64],
    grad_out: &[f64],
    grad_state_out: &[f64],
    n_v: usize,
    n_k: usize,
    head_dim: usize,
) -> GdnStepVjpReference {
    let mut result = GdnStepVjpReference {
        grad_q: vec![0.0; n_k * head_dim],
        grad_k: vec![0.0; n_k * head_dim],
        grad_v: vec![0.0; n_v * head_dim],
        grad_decay: vec![0.0; n_v],
        grad_beta: vec![0.0; n_v],
        grad_state: vec![0.0; n_v * head_dim * head_dim],
    };
    for hi in 0..n_v {
        let hk = hi % n_k;
        for dv in 0..head_dim {
            let vector_index = hi * head_dim + dv;
            let row_offset = vector_index * head_dim;
            let prediction: f64 = (0..head_dim)
                .map(|dk| decay[hi] * state[row_offset + dk] * k[hk * head_dim + dk])
                .sum();
            let residual = v[vector_index] - prediction;
            let correction = beta[hi] * residual;
            let mut grad_c = 0.0f64;
            for dk in 0..head_dim {
                let g = grad_state_out[row_offset + dk]
                    + grad_out[vector_index] * q[hk * head_dim + dk];
                grad_c += g * k[hk * head_dim + dk];
            }
            result.grad_v[vector_index] = beta[hi] * grad_c;
            result.grad_beta[hi] += grad_c * residual;
            for dk in 0..head_dim {
                let qk_index = hk * head_dim + dk;
                let state_in = state[row_offset + dk];
                let predicted = decay[hi] * state_in;
                let state_out = predicted + correction * k[qk_index];
                let g = grad_state_out[row_offset + dk] + grad_out[vector_index] * q[qk_index];
                let grad_predicted = g - beta[hi] * grad_c * k[qk_index];
                result.grad_q[qk_index] += state_out * grad_out[vector_index];
                result.grad_k[qk_index] += beta[hi] * (residual * g - grad_c * predicted);
                result.grad_decay[hi] += grad_predicted * state_in;
                result.grad_state[row_offset + dk] = decay[hi] * grad_predicted;
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_sequence_forward_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: &[f64],
    beta: &[f64],
    initial_state: &[f64],
    n_tokens: usize,
    n_v: usize,
    n_k: usize,
    head_dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let qk_elements = n_k * head_dim;
    let vector_elements = n_v * head_dim;
    let state_elements = vector_elements * head_dim;
    let mut outputs = vec![0.0f64; n_tokens * vector_elements];
    let mut checkpoints = vec![0.0f64; n_tokens * state_elements];
    let mut state = initial_state.to_vec();
    for token in 0..n_tokens {
        let q = &q[token * qk_elements..(token + 1) * qk_elements];
        let k = &k[token * qk_elements..(token + 1) * qk_elements];
        let v = &v[token * vector_elements..(token + 1) * vector_elements];
        let decay = &decay[token * n_v..(token + 1) * n_v];
        let beta = &beta[token * n_v..(token + 1) * n_v];
        let output = &mut outputs[token * vector_elements..(token + 1) * vector_elements];
        let mut next_state = vec![0.0f64; state_elements];
        for hi in 0..n_v {
            let hk = hi % n_k;
            for dv in 0..head_dim {
                let vector_index = hi * head_dim + dv;
                let row_offset = vector_index * head_dim;
                let prediction = (0..head_dim)
                    .map(|dk| decay[hi] * state[row_offset + dk] * k[hk * head_dim + dk])
                    .sum::<f64>();
                let correction = beta[hi] * (v[vector_index] - prediction);
                for dk in 0..head_dim {
                    let qk_index = hk * head_dim + dk;
                    let value = decay[hi] * state[row_offset + dk] + correction * k[qk_index];
                    next_state[row_offset + dk] = value;
                    output[vector_index] += value * q[qk_index];
                }
            }
        }
        checkpoints[token * state_elements..(token + 1) * state_elements]
            .copy_from_slice(&next_state);
        state = next_state;
    }
    (outputs, checkpoints)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_sequence_objective_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: &[f64],
    beta: &[f64],
    initial_state: &[f64],
    grad_out: &[f64],
    grad_final_state: &[f64],
    n_tokens: usize,
    n_v: usize,
    n_k: usize,
    head_dim: usize,
) -> f64 {
    let (outputs, checkpoints) = gdn_sequence_forward_f64(
        q,
        k,
        v,
        decay,
        beta,
        initial_state,
        n_tokens,
        n_v,
        n_k,
        head_dim,
    );
    let state_elements = n_v * head_dim * head_dim;
    outputs
        .iter()
        .zip(grad_out)
        .map(|(value, gradient)| value * gradient)
        .sum::<f64>()
        + checkpoints[(n_tokens - 1) * state_elements..]
            .iter()
            .zip(grad_final_state)
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_sequence_vjp_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: &[f64],
    beta: &[f64],
    initial_state: &[f64],
    grad_out: &[f64],
    grad_final_state: &[f64],
    n_tokens: usize,
    n_v: usize,
    n_k: usize,
    head_dim: usize,
) -> GdnStepVjpReference {
    let qk_elements = n_k * head_dim;
    let vector_elements = n_v * head_dim;
    let state_elements = vector_elements * head_dim;
    let (_, checkpoints) = gdn_sequence_forward_f64(
        q,
        k,
        v,
        decay,
        beta,
        initial_state,
        n_tokens,
        n_v,
        n_k,
        head_dim,
    );
    let mut result = GdnStepVjpReference {
        grad_q: vec![0.0; n_tokens * qk_elements],
        grad_k: vec![0.0; n_tokens * qk_elements],
        grad_v: vec![0.0; n_tokens * vector_elements],
        grad_decay: vec![0.0; n_tokens * n_v],
        grad_beta: vec![0.0; n_tokens * n_v],
        grad_state: grad_final_state.to_vec(),
    };
    for token in (0..n_tokens).rev() {
        let state = if token == 0 {
            initial_state
        } else {
            &checkpoints[(token - 1) * state_elements..token * state_elements]
        };
        let step = gdn_step_decay_vjp_f64(
            &q[token * qk_elements..(token + 1) * qk_elements],
            &k[token * qk_elements..(token + 1) * qk_elements],
            &v[token * vector_elements..(token + 1) * vector_elements],
            &decay[token * n_v..(token + 1) * n_v],
            &beta[token * n_v..(token + 1) * n_v],
            state,
            &grad_out[token * vector_elements..(token + 1) * vector_elements],
            &result.grad_state,
            n_v,
            n_k,
            head_dim,
        );
        result.grad_q[token * qk_elements..(token + 1) * qk_elements].copy_from_slice(&step.grad_q);
        result.grad_k[token * qk_elements..(token + 1) * qk_elements].copy_from_slice(&step.grad_k);
        result.grad_v[token * vector_elements..(token + 1) * vector_elements]
            .copy_from_slice(&step.grad_v);
        result.grad_decay[token * n_v..(token + 1) * n_v].copy_from_slice(&step.grad_decay);
        result.grad_beta[token * n_v..(token + 1) * n_v].copy_from_slice(&step.grad_beta);
        result.grad_state = step.grad_state;
    }
    result
}

pub(super) struct SsmConvSequenceVjpReference {
    pub(super) grad_qkv: Vec<f64>,
    pub(super) grad_state: Vec<f64>,
}

pub(super) fn ssm_conv_sequence_forward_f64(
    qkv: &[f64],
    initial_state: &[f64],
    weight: &[f64],
    n_tokens: usize,
    conv_dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let state_elements = 3 * conv_dim;
    let mut outputs = vec![0.0f64; n_tokens * conv_dim];
    let mut checkpoints = vec![0.0f64; n_tokens * state_elements];
    let mut state = initial_state.to_vec();
    for token in 0..n_tokens {
        let qkv = &qkv[token * conv_dim..(token + 1) * conv_dim];
        let output = &mut outputs[token * conv_dim..(token + 1) * conv_dim];
        for channel in 0..conv_dim {
            let preactivation = weight[4 * channel] * state[channel]
                + weight[4 * channel + 1] * state[conv_dim + channel]
                + weight[4 * channel + 2] * state[2 * conv_dim + channel]
                + weight[4 * channel + 3] * qkv[channel];
            output[channel] = preactivation / (1.0 + (-preactivation).exp());
        }
        let checkpoint = &mut checkpoints[token * state_elements..(token + 1) * state_elements];
        checkpoint[..2 * conv_dim].copy_from_slice(&state[conv_dim..]);
        checkpoint[2 * conv_dim..].copy_from_slice(qkv);
        state.copy_from_slice(checkpoint);
    }
    (outputs, checkpoints)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ssm_conv_sequence_objective_f64(
    qkv: &[f64],
    initial_state: &[f64],
    weight: &[f64],
    grad_q: &[f64],
    grad_k: &[f64],
    grad_v: &[f64],
    grad_final_state: &[f64],
    n_tokens: usize,
    qk_elements: usize,
    v_elements: usize,
) -> f64 {
    let conv_dim = 2 * qk_elements + v_elements;
    let state_elements = 3 * conv_dim;
    let (outputs, checkpoints) =
        ssm_conv_sequence_forward_f64(qkv, initial_state, weight, n_tokens, conv_dim);
    let mut objective = 0.0f64;
    for token in 0..n_tokens {
        let output = &outputs[token * conv_dim..(token + 1) * conv_dim];
        objective += output[..qk_elements]
            .iter()
            .zip(&grad_q[token * qk_elements..(token + 1) * qk_elements])
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>();
        objective += output[qk_elements..2 * qk_elements]
            .iter()
            .zip(&grad_k[token * qk_elements..(token + 1) * qk_elements])
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>();
        objective += output[2 * qk_elements..]
            .iter()
            .zip(&grad_v[token * v_elements..(token + 1) * v_elements])
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>();
    }
    objective
        + checkpoints[(n_tokens - 1) * state_elements..]
            .iter()
            .zip(grad_final_state)
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ssm_conv_sequence_vjp_f64(
    qkv: &[f64],
    initial_state: &[f64],
    weight: &[f64],
    grad_q: &[f64],
    grad_k: &[f64],
    grad_v: &[f64],
    grad_final_state: &[f64],
    n_tokens: usize,
    qk_elements: usize,
    v_elements: usize,
) -> SsmConvSequenceVjpReference {
    let conv_dim = 2 * qk_elements + v_elements;
    let state_elements = 3 * conv_dim;
    let (_, checkpoints) =
        ssm_conv_sequence_forward_f64(qkv, initial_state, weight, n_tokens, conv_dim);
    let mut grad_qkv = vec![0.0f64; n_tokens * conv_dim];
    let mut grad_state = grad_final_state.to_vec();
    for token in (0..n_tokens).rev() {
        let state = if token == 0 {
            initial_state
        } else {
            &checkpoints[(token - 1) * state_elements..token * state_elements]
        };
        let qkv = &qkv[token * conv_dim..(token + 1) * conv_dim];
        let mut next_grad_state = vec![0.0f64; state_elements];
        for channel in 0..conv_dim {
            let preactivation = weight[4 * channel] * state[channel]
                + weight[4 * channel + 1] * state[conv_dim + channel]
                + weight[4 * channel + 2] * state[2 * conv_dim + channel]
                + weight[4 * channel + 3] * qkv[channel];
            let sigmoid = 1.0 / (1.0 + (-preactivation).exp());
            let grad_output = if channel < qk_elements {
                grad_q[token * qk_elements + channel]
            } else if channel < 2 * qk_elements {
                grad_k[token * qk_elements + channel - qk_elements]
            } else {
                grad_v[token * v_elements + channel - 2 * qk_elements]
            };
            let grad_preactivation =
                grad_output * sigmoid * (1.0 + preactivation * (1.0 - sigmoid));
            grad_qkv[token * conv_dim + channel] =
                grad_preactivation * weight[4 * channel + 3] + grad_state[2 * conv_dim + channel];
            next_grad_state[channel] = grad_preactivation * weight[4 * channel];
            next_grad_state[conv_dim + channel] =
                grad_preactivation * weight[4 * channel + 1] + grad_state[channel];
            next_grad_state[2 * conv_dim + channel] =
                grad_preactivation * weight[4 * channel + 2] + grad_state[conv_dim + channel];
        }
        grad_state = next_grad_state;
    }
    SsmConvSequenceVjpReference {
        grad_qkv,
        grad_state,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_envelope_objective_f64(
    qkv_now: &[f64],
    conv_state: &[f64],
    conv_weight: &[f64],
    alpha_source: &[f64],
    dt_bias: &[f64],
    a_log: &[f64],
    beta_source: &[f64],
    recurrence_state: &[f64],
    z: &[f64],
    norm_weight: &[f64],
    grad_y: &[f64],
    grad_recurrence_state: &[f64],
    grad_conv_state: &[f64],
    n_v: usize,
    n_k: usize,
    head_dim: usize,
    l2_eps: f64,
    rms_eps: f64,
) -> f64 {
    let qk_elements = n_k * head_dim;
    let v_elements = n_v * head_dim;
    let conv_dim = 2 * qk_elements + v_elements;
    let mut conv_output = vec![0.0f64; conv_dim];
    let mut objective = 0.0f64;
    for channel in 0..conv_dim {
        let preactivation = (0..3)
            .map(|row| conv_weight[channel * 4 + row] * conv_state[row * conv_dim + channel])
            .sum::<f64>()
            + conv_weight[channel * 4 + 3] * qkv_now[channel];
        conv_output[channel] = preactivation / (1.0 + (-preactivation).exp());
        objective += grad_conv_state[channel] * conv_state[conv_dim + channel];
        objective += grad_conv_state[conv_dim + channel] * conv_state[2 * conv_dim + channel];
        objective += grad_conv_state[2 * conv_dim + channel] * qkv_now[channel];
    }
    let normalize = |rows: &[f64]| {
        let mut output = vec![0.0f64; rows.len()];
        for head in 0..n_k {
            let base = head * head_dim;
            let radius = rows[base..base + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                .max(l2_eps);
            for index in 0..head_dim {
                output[base + index] = rows[base + index] / radius;
            }
        }
        output
    };
    let q = normalize(&conv_output[..qk_elements]);
    let k = normalize(&conv_output[qk_elements..2 * qk_elements]);
    let v = &conv_output[2 * qk_elements..];
    let decay: Vec<f64> = (0..n_v)
        .map(|head| {
            let value = alpha_source[head] + dt_bias[head];
            let softplus = if value > 20.0 {
                value
            } else if value < -20.0 {
                value.exp()
            } else {
                (1.0 + value.exp()).ln()
            };
            (softplus * a_log[head]).exp()
        })
        .collect();
    let beta: Vec<f64> = beta_source
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let mut recurrence_output = vec![0.0f64; v_elements];
    for hi in 0..n_v {
        let hk = hi % n_k;
        for dv in 0..head_dim {
            let vector_index = hi * head_dim + dv;
            let row_offset = vector_index * head_dim;
            let prediction: f64 = (0..head_dim)
                .map(|dk| decay[hi] * recurrence_state[row_offset + dk] * k[hk * head_dim + dk])
                .sum();
            let correction = beta[hi] * (v[vector_index] - prediction);
            for dk in 0..head_dim {
                let state_out = decay[hi] * recurrence_state[row_offset + dk]
                    + correction * k[hk * head_dim + dk];
                recurrence_output[vector_index] += state_out * q[hk * head_dim + dk];
                objective += state_out * grad_recurrence_state[row_offset + dk];
            }
        }
    }
    for hi in 0..n_v {
        let base = hi * head_dim;
        let sumsq = recurrence_output[base..base + head_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let scale = (sumsq / head_dim as f64 + rms_eps).sqrt().recip();
        for index in 0..head_dim {
            let offset = base + index;
            let silu_z = z[offset] / (1.0 + (-z[offset]).exp());
            objective +=
                grad_y[offset] * recurrence_output[offset] * scale * norm_weight[index] * silu_z;
        }
    }
    objective
}

/// CPU f64 reference for the matrix-attention sidecar semantics: causal
/// multi-row attention over an F16 KV prefix with per-row visibility
/// `base_pos + row + 1`, softmax of `q·k / sqrt(head_dim)`.
///
/// Inputs must already be f16-representable (pre-rounded) so the GPU's
/// half demotion inside the GEMMs is exact and tolerances measure reduction
/// order + the probs half demotion, not input rounding.
#[allow(clippy::too_many_arguments)]
pub(super) fn cpu_matrix_attn_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    n_q: usize,
    n_kv: usize,
    group: usize,
    hd: usize,
) -> Vec<f32> {
    let kv_dim = n_kv * hd;
    let scale = 1.0f64 / (hd as f64).sqrt();
    let mut out = vec![0f32; n_rows * n_q * hd];
    for kvh in 0..n_kv {
        for row in 0..n_rows {
            for g in 0..group {
                let qh = kvh * group + g;
                let visible = (base_pos + row + 1).min(n_pos);
                let qv = &q[(row * n_q + qh) * hd..][..hd];
                let mut s = vec![0f64; visible];
                for (p, sp) in s.iter_mut().enumerate() {
                    let kb = &k[p * kv_dim + kvh * hd..][..hd];
                    *sp = qv
                        .iter()
                        .zip(kb)
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum::<f64>()
                        * scale;
                }
                let m = s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut probs: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
                let sum: f64 = probs.iter().sum();
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let mut acc = vec![0f64; hd];
                for (p, pr) in probs.iter_mut().enumerate() {
                    let w = *pr * inv;
                    let vb = &v[p * kv_dim + kvh * hd..][..hd];
                    for (a, x) in acc.iter_mut().zip(vb) {
                        *a += w * (*x as f64);
                    }
                }
                let ob = &mut out[(row * n_q + qh) * hd..][..hd];
                for (o, a) in ob.iter_mut().zip(&acc) {
                    *o = *a as f32;
                }
            }
        }
    }
    out
}

pub(super) fn read_back_u16(tensor: &MetalTensor) -> Vec<u16> {
    assert_eq!(tensor.dtype, GgmlType::F16);
    assert_eq!(tensor.buffer.storageMode(), MTLStorageMode::Shared);
    assert_eq!(tensor.offset as usize % std::mem::align_of::<u16>(), 0);
    let n = tensor.n_elements() as usize;
    let n_bytes = n.checked_mul(std::mem::size_of::<u16>()).unwrap();
    let end = (tensor.offset as usize).checked_add(n_bytes).unwrap();
    assert!(end <= tensor.buffer.length());
    let mut out = Vec::<u16>::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(
            (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize)
                as *const u16,
            out.as_mut_ptr(),
            n,
        );
        out.set_len(n);
    }
    out
}

pub(super) static ATTN_MATRIX_VT_SCOPE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn greedy_total_order_key(bits: u32) -> Option<u32> {
    ((bits & 0x7fff_ffff) <= 0x7f80_0000).then_some({
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    })
}

proptest::proptest! {
    #[test]
    fn greedy_total_order_key_matches_f32_total_cmp(a_bits: u32, b_bits: u32) {
        let Some(a_key) = greedy_total_order_key(a_bits) else {
            return Ok(());
        };
        let Some(b_key) = greedy_total_order_key(b_bits) else {
            return Ok(());
        };
        let a = f32::from_bits(a_bits);
        let b = f32::from_bits(b_bits);
        proptest::prop_assert_eq!(a_key.cmp(&b_key), a.total_cmp(&b));
    }
}

pub(super) fn fill_audit_f16(tensor: &MetalTensor, salt: usize) {
    assert_eq!(tensor.dtype, GgmlType::F16);
    let pattern: Vec<u16> = (0..4096)
        .map(|i| {
            let raw = ((i * 37 + salt * 101) % 257) as f32 - 128.0;
            half::f16::from_f32(raw * 0.00390625).to_bits()
        })
        .collect();
    let n = tensor.n_elements() as usize;
    let offset = tensor.offset as usize / std::mem::size_of::<u16>();
    let dst = unsafe { (tensor.buffer.contents().as_ptr() as *mut u16).add(offset) };
    for start in (0..n).step_by(pattern.len()) {
        let len = (n - start).min(pattern.len());
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), dst.add(start), len);
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub(super) fn dflash_attn_cpu_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos_k: &[i32],
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv_total: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    noncausal_noise: bool,
) -> Vec<f32> {
    let group = n_q_heads / n_kv_heads;
    let q_dim = n_q_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut o = vec![0.0_f32; n * q_dim];
    let full_attn = swa_window == 0;
    for q_idx in 0..n {
        let q_pos = noise_start_pos + q_idx as u32;
        for q_head in 0..n_q_heads {
            let kv_head = q_head / group;
            // Q vector for this (q_idx, q_head).
            let q_off = q_idx * q_dim + q_head * head_dim;
            // PASS 1: max.
            let mut m_run = f32::NEG_INFINITY;
            for kk in 0..n_kv_total {
                let allowed = if kk < ctx_len {
                    if full_attn {
                        true
                    } else {
                        let k_pos = pos_k[kk] as u32;
                        k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                    }
                } else {
                    noncausal_noise || (kk - ctx_len) <= q_idx
                };
                if !allowed {
                    continue;
                }
                let k_off = kk * kv_stride + kv_head * head_dim;
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[q_off + d] * k[k_off + d];
                }
                s *= scale;
                if s > m_run {
                    m_run = s;
                }
            }
            // PASS 2: sum.
            let mut l_sum = 0.0_f32;
            for kk in 0..n_kv_total {
                let allowed = if kk < ctx_len {
                    if full_attn {
                        true
                    } else {
                        let k_pos = pos_k[kk] as u32;
                        k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                    }
                } else {
                    noncausal_noise || (kk - ctx_len) <= q_idx
                };
                if !allowed {
                    continue;
                }
                let k_off = kk * kv_stride + kv_head * head_dim;
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[q_off + d] * k[k_off + d];
                }
                s *= scale;
                l_sum += (s - m_run).exp();
            }
            let inv_l = if l_sum > 0.0 { 1.0 / l_sum } else { 0.0 };
            // PASS 3: V agg.
            for kk in 0..n_kv_total {
                let allowed = if kk < ctx_len {
                    if full_attn {
                        true
                    } else {
                        let k_pos = pos_k[kk] as u32;
                        k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                    }
                } else {
                    noncausal_noise || (kk - ctx_len) <= q_idx
                };
                if !allowed {
                    continue;
                }
                let k_off = kk * kv_stride + kv_head * head_dim;
                let v_off = kk * kv_stride + kv_head * head_dim;
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[q_off + d] * k[k_off + d];
                }
                s *= scale;
                let w = (s - m_run).exp() * inv_l;
                for d in 0..head_dim {
                    o[q_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    o
}

/// One-shot helper: run kernel_dflash_attn_f32 against synthetic
/// CPU-staged buffers and read back o.
pub(super) fn dflash_attn_readback(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos_k: &[i32],
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv_total: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<Vec<f32>, MetalError> {
    let q_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(q),
        vec![(n * n_q_heads * head_dim) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let k_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(k),
        vec![(n_kv_total * n_kv_heads * head_dim) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let v_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v),
        vec![(n_kv_total * n_kv_heads * head_dim) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let pos_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(pos_k),
        vec![n_kv_total as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let o_t = MetalTensor::zeros_f32(ctx, vec![(n * n_q_heads * head_dim) as u64])?;
    one_shot(ctx, |enc| {
        encode_dflash_attn_f32(
            ctx,
            enc,
            &q_t,
            &k_t,
            &v_t,
            &pos_t,
            &o_t,
            n,
            n_q_heads,
            n_kv_heads,
            head_dim,
            n_kv_total,
            ctx_len,
            noise_start_pos,
            swa_window,
        )
    })?;
    Ok(read_back_f32(&o_t.buffer, n * n_q_heads * head_dim))
}

/// One-shot helper for the two-range DFlash attention sidecar.
pub(super) fn dflash_attn_two_range_readback(
    ctx: &MetalContext,
    q: &[f32],
    k_ctx: &[f32],
    v_ctx: &[f32],
    k_noise: &[f32],
    v_noise: &[f32],
    pos_ctx: &[i32],
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    online: bool,
    full_gqa_split4: bool,
    swa_split4: bool,
    noncausal_noise: bool,
    ctx_scan_start: usize,
) -> Result<Vec<f32>, MetalError> {
    let q_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(q),
        vec![(n * n_q_heads * head_dim) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let ctx_rows = ctx_len.max(1);
    let kv_stride = n_kv_heads * head_dim;
    let k_ctx_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(k_ctx),
        vec![(ctx_rows * kv_stride) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let v_ctx_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v_ctx),
        vec![(ctx_rows * kv_stride) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let k_noise_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(k_noise),
        vec![(n * kv_stride) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let v_noise_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v_noise),
        vec![(n * kv_stride) as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let pos_rows = ctx_len.max(1);
    let pos_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(pos_ctx),
        vec![pos_rows as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let o_t = MetalTensor::zeros_f32(ctx, vec![(n * n_q_heads * head_dim) as u64])?;
    let o_partial_len = n * 8 * 4 * 4 * 128;
    let ml_partial_len = n * 8 * 4 * 4 * 2;
    let o_partial_host = vec![f32::NAN; o_partial_len];
    let ml_partial_host = vec![f32::NAN; ml_partial_len];
    let o_partial_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&o_partial_host),
        vec![o_partial_len as u64],
        crate::tensor::GgmlType::F32,
    )?;
    let ml_partial_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&ml_partial_host),
        vec![ml_partial_len as u64],
        crate::tensor::GgmlType::F32,
    )?;
    one_shot(ctx, |enc| {
        if swa_split4 {
            encode_dflash_attn_swa_split4_with_noncausal_f32(
                ctx,
                enc,
                &q_t,
                &k_ctx_t,
                &v_ctx_t,
                &k_noise_t,
                &v_noise_t,
                &pos_t,
                &o_partial_t,
                &ml_partial_t,
                &o_t,
                n,
                ctx_len,
                noise_start_pos,
                swa_window,
                ctx_scan_start,
                noncausal_noise,
            )
        } else if full_gqa_split4 {
            encode_dflash_attn_full_gqa_split4_f32(
                ctx,
                enc,
                &q_t,
                &k_ctx_t,
                &v_ctx_t,
                &k_noise_t,
                &v_noise_t,
                &o_partial_t,
                &ml_partial_t,
                &o_t,
                n,
                n_q_heads,
                n_kv_heads,
                head_dim,
                ctx_len,
            )
        } else if online {
            encode_dflash_attn_online_two_range_scan_f32(
                ctx,
                enc,
                &q_t,
                &k_ctx_t,
                &v_ctx_t,
                &k_noise_t,
                &v_noise_t,
                &pos_t,
                &o_t,
                n,
                n_q_heads,
                n_kv_heads,
                head_dim,
                ctx_len,
                noise_start_pos,
                swa_window,
                ctx_scan_start,
            )
        } else {
            encode_dflash_attn_two_range_f32(
                ctx,
                enc,
                &q_t,
                &k_ctx_t,
                &v_ctx_t,
                &k_noise_t,
                &v_noise_t,
                &pos_t,
                &o_t,
                n,
                n_q_heads,
                n_kv_heads,
                head_dim,
                ctx_len,
                noise_start_pos,
                swa_window,
            )
        }
    })?;
    Ok(read_back_f32(&o_t.buffer, n * n_q_heads * head_dim))
}
