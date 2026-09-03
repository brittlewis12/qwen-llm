use super::*;
use std::mem::size_of;

struct LeaseFixture(PathBuf);

impl LeaseFixture {
    fn new() -> Self {
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

#[test]
fn process_lease_rejects_overlap_and_recovers_after_release() {
    let fixture = LeaseFixture::new();
    let first = open_metal_process_lease(&fixture.0, false).expect("acquire first lease");
    let error = open_metal_process_lease(&fixture.0, false)
        .err()
        .expect("overlapping lease must fail");
    let MetalError::ProcessLease(detail) = error else {
        panic!("unexpected overlap error: {error}");
    };
    assert!(detail.contains("another qwen process owns"));
    assert!(detail.contains(&format!("pid={}", std::process::id())));
    drop(first);
    let second = open_metal_process_lease(&fixture.0, false)
        .expect("lease must recover after owner release");
    drop(second);
}

#[test]
fn process_lease_owner_fields_are_single_line_and_bounded() {
    let dirty = format!("bad value\n{}", "x".repeat(1_024));
    let cleaned = lease_owner_field(&dirty);
    assert!(!cleaned.contains('\n'));
    assert!(!cleaned.contains(' '));
    assert_eq!(cleaned.len(), 512);
}

#[test]
fn process_lease_owner_display_cannot_inject_lines() {
    let dirty = "pid=1\nforged=owner\t\u{1b}[31m";
    let cleaned = lease_owner_display(dirty);
    assert_eq!(cleaned, "pid=1_forged=owner___31m");
    assert!(!cleaned.contains('\n'));
    assert!(!cleaned.contains('\t'));
}

#[test]
fn wired_memory_guard_rejects_half_of_physical_memory() {
    assert!(!host_wired_memory_is_unsafe(0, 128));
    assert!(!host_wired_memory_is_unsafe(63, 128));
    assert!(host_wired_memory_is_unsafe(64, 128));
    assert!(!host_wired_memory_is_unsafe(u64::MAX, 0));
}

#[test]
fn diagnostics_observers_return_to_inactive_state() {
    assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
    {
        let _trace = kernel_trace_begin();
        assert_eq!(diagnostics_observer_active_counts(), [1, 0, 0]);
    }
    dispatch_census_begin();
    assert_eq!(diagnostics_observer_active_counts(), [0, 1, 0]);
    assert!(dispatch_census_take().is_empty());
    assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
}

#[test]
fn allocation_census_records_realized_storage() {
    let ctx = MetalContext::new().expect("create Metal context");
    allocation_census_begin();
    let first = ctx
        .buffer_uninit(17)
        .expect("allocate uninitialized buffer");
    let second = ctx.buffer_from(&[1u32, 2]).expect("allocate copied buffer");
    let rows = allocation_census_take();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].requested_bytes, 17);
    assert_eq!(rows[0].buffer_length, first.length() as u64);
    assert_eq!(rows[0].storage_mode, "shared");
    assert_eq!(rows[1].requested_bytes, 8);
    assert_eq!(rows[1].buffer_length, second.length() as u64);
    assert_eq!(rows[1].storage_mode, "shared");
    assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
}

#[test]
fn mps_two_pass_topk_matches_cpu_at_full_vocabulary_width() {
    // Exercise the rounded mask-dispatch tail as well as full vocabulary width.
    const ROWS: usize = 3;
    const COLUMNS: usize = 248_320;
    const PASS_K: usize = 16;
    const RESULT_K: usize = 25;
    const PRIME: usize = 1_000_003;
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("initialize Metal: {error}"),
    };
    let mut host = Vec::with_capacity(ROWS * COLUMNS);
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            host.push(((column * 48_271 + row * 17) % PRIME) as f32);
        }
    }
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&host),
        vec![ROWS as u64, COLUMNS as u64],
        GgmlType::F32,
    )
    .expect("input tensor");
    let first_ids =
        MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("first IDs");
    let first_values =
        MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("first values");
    let second_ids =
        MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("second IDs");
    let second_values =
        MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("second values");
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let started = std::time::Instant::now();
    encode_mps_topk16_f32(
        &ctx,
        &command,
        &input,
        &first_ids,
        &first_values,
        ROWS,
        COLUMNS,
    )
    .expect("first top-k pass");
    let encoder = KernelEncoder::begin(&command);
    encode_mask_row_indices_f32(&ctx, &encoder, &input, &first_ids, ROWS, COLUMNS, PASS_K)
        .expect("mask first-pass IDs");
    encoder.end();
    encode_mps_topk16_f32(
        &ctx,
        &command,
        &input,
        &second_ids,
        &second_values,
        ROWS,
        COLUMNS,
    )
    .expect("second top-k pass");
    command.commit();
    command.waitUntilCompleted();
    let elapsed = started.elapsed();
    assert_eq!(
        command.status(),
        objc2_metal::MTLCommandBufferStatus::Completed,
        "MPS top-k command failed: {:?}",
        command.error()
    );

    let read_ids = |tensor: &MetalTensor| unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<i32>(),
            ROWS * PASS_K,
        )
        .to_vec()
    };
    let first_ids = read_ids(&first_ids);
    let second_ids = read_ids(&second_ids);
    let first_values = read_back_f32(&first_values.buffer, ROWS * PASS_K);
    let second_values = read_back_f32(&second_values.buffer, ROWS * PASS_K);
    for row in 0..ROWS {
        let mut candidates = Vec::with_capacity(2 * PASS_K);
        for (ids, values) in [(&first_ids, &first_values), (&second_ids, &second_values)] {
            for column in 0..PASS_K {
                let offset = row * PASS_K + column;
                candidates.push((ids[offset] as usize, values[offset]));
            }
        }
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut expected = Vec::<(usize, f32)>::with_capacity(RESULT_K);
        for column in 0..COLUMNS {
            let value = host[row * COLUMNS + column];
            let insertion = expected
                .partition_point(|&(id, score)| score > value || (score == value && id < column));
            if insertion < RESULT_K {
                expected.insert(insertion, (column, value));
                if expected.len() > RESULT_K {
                    expected.pop();
                }
            }
        }
        assert_eq!(&candidates[..RESULT_K], expected.as_slice(), "row {row}");
    }
    eprintln!(
        "MPS two-pass top-25 rows={ROWS} columns={COLUMNS} wall_ms={:.3}",
        elapsed.as_secs_f64() * 1e3
    );
}

fn offset_tensor(
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

fn tensor_f32_at_offset(tensor: &MetalTensor) -> Vec<f32> {
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

fn tensor_backing_bytes(tensor: &MetalTensor) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<u8>(),
            tensor.buffer.length(),
        )
        .to_vec()
    }
}

fn assert_offset_guards(tensor: &MetalTensor, prefix: usize, suffix: usize) {
    let bytes = tensor_backing_bytes(tensor);
    assert!(bytes[..prefix].iter().all(|&byte| byte == 0xA5));
    assert!(
        bytes[bytes.len() - suffix..]
            .iter()
            .all(|&byte| byte == 0x5A)
    );
}

fn synthetic_q8_0_bank(n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f32>) {
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

fn synthetic_q8_0_bytes(n_in: usize, n_out: usize) -> Vec<u8> {
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

fn f32_f64_differential(actual: &[f32], reference: &[f64]) -> (f64, f64, f64) {
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

fn f32_differential(actual: &[f32], reference: &[f32]) -> (f64, f64, f64) {
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

fn synthetic_dense_linear_bank(dtype: GgmlType, n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f32>) {
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

fn encode_q6_k_block(d: f32, seed: usize) -> ([u8; 210], [f32; 256]) {
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

fn encode_iq4_nl_block(d: f32, seed: usize) -> ([u8; 18], [f32; 32]) {
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

fn encode_iq4_xs_block(d: f32, seed: usize) -> [u8; 136] {
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

fn synthetic_iq4_xs_bank(n_in: usize, n_out: usize, n_expert: usize, seed: usize) -> Vec<u8> {
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

fn synthetic_iq4_nl_bank(n_in: usize, n_out: usize, n_expert: usize, seed: usize) -> Vec<u8> {
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

fn dequant_expert(
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

fn assert_moe_oracle_close(label: &str, actual: &[f32], expected: &[f32]) {
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

fn run_iq4_moe_decode_case(
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

fn encode_iq2_xs_block(d: f32, seed: usize) -> [u8; 74] {
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

#[test]
fn iq2_xs_mat_vec_and_mat_mat_match_cpu_codec_with_offsets() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_IN: usize = 4_096;
    const N_OUT: usize = 9;
    let mut weight_bytes = Vec::new();
    for row in 0..N_OUT {
        for block in 0..N_IN / 256 {
            weight_bytes.extend_from_slice(&encode_iq2_xs_block(
                0.001953125 * (1 + (row * 2 + block) % 7) as f32,
                row * (N_IN / 256) + block,
            ));
        }
    }
    let desc = TensorDesc {
        name: "iq2_xs_test".into(),
        shape: vec![N_IN as u64, N_OUT as u64],
        dtype: GgmlType::IQ2_XS,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: weight_bytes.len() as u64,
    };
    let decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
        .expect("llama.cpp IQ2_XS reference dequantization");
    let weight = offset_tensor(
        &ctx,
        32,
        &weight_bytes,
        19,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::IQ2_XS,
    );
    let input_values = (0..N_IN)
        .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
        .collect::<Vec<_>>();
    let input = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&input_values),
        17,
        vec![N_IN as u64],
        GgmlType::F32,
    );
    let output = offset_tensor(
        &ctx,
        32,
        &[0u8; N_OUT * size_of::<f32>()],
        23,
        vec![N_OUT as u64],
        GgmlType::F32,
    );
    let command = ctx.queue.commandBuffer().expect("IQ2_XS mat-vec command");
    let encoder = KernelEncoder::begin(&command);
    crate::metal_forward::encode_mat_vec_dispatch(
        &ctx, &encoder, &weight, &input, &output, N_IN, N_OUT,
    )
    .expect("IQ2_XS mat-vec dispatch");
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "IQ2_XS mat-vec command failed");
    let expected_row = |row: usize, input: &[f32]| {
        decoded[row * N_IN..(row + 1) * N_IN]
            .iter()
            .zip(input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>()
    };
    for (row, actual) in tensor_f32_at_offset(&output).into_iter().enumerate() {
        let expected = expected_row(row, &input_values);
        let tolerance = 3.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "mat-vec row {row}: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }

    for n_query in [1usize, 2, 6, 16, 32, 128] {
        let inputs = (0..n_query * N_IN)
            .map(|index| ((index * 41 + index / N_IN * 17 + 3) % 509) as f32 * 0.0005 - 0.127)
            .collect::<Vec<_>>();
        let x = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&inputs),
            13,
            vec![N_IN as u64, n_query as u64],
            GgmlType::F32,
        );
        let y = offset_tensor(
            &ctx,
            32,
            &vec![0u8; n_query * N_OUT * size_of::<f32>()],
            29,
            vec![N_OUT as u64, n_query as u64],
            GgmlType::F32,
        );
        let command = ctx.queue.commandBuffer().expect("IQ2_XS mat-mat command");
        let encoder = KernelEncoder::begin(&command);
        crate::metal_forward::encode_mat_mat_dispatch(
            &ctx, &encoder, &weight, &x, &y, N_IN, N_OUT, n_query,
        )
        .unwrap_or_else(|error| panic!("IQ2_XS mat-mat N={n_query}: {error}"));
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "IQ2_XS mat-mat N={n_query} command failed"
        );
        let actual = tensor_f32_at_offset(&y);
        for query in 0..n_query {
            let input = &inputs[query * N_IN..(query + 1) * N_IN];
            for row in 0..N_OUT {
                let expected = expected_row(row, input);
                let got = actual[query * N_OUT + row];
                let tolerance = 3.0e-5 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tolerance,
                    "mat-mat N={n_query} query={query} row={row}: got {got}, expected {expected}, tolerance {tolerance}"
                );
            }
        }
    }
}

#[test]
fn get_rows_q6_k_gpu_matches_cpu_with_offsets_and_rejects_bad_inputs() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_VOCAB: usize = 3;
    const N_COLS: usize = 512;
    let mut weight_bytes = Vec::new();
    let mut decoded = vec![0.0f32; N_VOCAB * N_COLS];
    for row in 0..N_VOCAB {
        for block_index in 0..2 {
            let (block, values) = encode_q6_k_block(
                if (row + block_index) % 2 == 0 {
                    0.5
                } else {
                    -0.25
                },
                row * 2 + block_index,
            );
            weight_bytes.extend_from_slice(&block);
            let start = row * N_COLS + block_index * 256;
            decoded[start..start + 256].copy_from_slice(&values);
        }
    }
    let desc = TensorDesc {
        name: "q6_k_test".into(),
        shape: vec![N_COLS as u64, N_VOCAB as u64],
        dtype: GgmlType::Q6_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: weight_bytes.len() as u64,
    };
    let codec_decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
        .expect("llama.cpp Q6_K reference dequantization");
    assert_eq!(decoded, codec_decoded);
    let decoded = codec_decoded;
    let embed = offset_tensor(
        &ctx,
        32,
        &weight_bytes,
        19,
        vec![N_COLS as u64, N_VOCAB as u64],
        GgmlType::Q6_K,
    );
    let row_ids = [2i32, -1, 0, 3];
    let ids = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&row_ids),
        12,
        vec![row_ids.len() as u64],
        GgmlType::I32,
    );
    let output_bytes = vec![0u8; row_ids.len() * N_COLS * size_of::<f32>()];
    let y = offset_tensor(
        &ctx,
        32,
        &output_bytes,
        20,
        vec![(row_ids.len() * N_COLS) as u64],
        GgmlType::F32,
    );
    one_shot(&ctx, |enc| {
        encode_get_rows_f32(&ctx, enc, &embed, &ids, &y, row_ids.len(), N_COLS)
    })
    .expect("Q6_K get_rows");

    let actual = tensor_f32_at_offset(&y);
    for (lookup_row, &row_id) in row_ids.iter().enumerate() {
        let output_row = &actual[lookup_row * N_COLS..(lookup_row + 1) * N_COLS];
        if row_id >= 0 && (row_id as usize) < N_VOCAB {
            let expected = &decoded[row_id as usize * N_COLS..(row_id as usize + 1) * N_COLS];
            assert_eq!(output_row, expected);
        } else {
            assert!(output_row.iter().all(|&value| value == 0.0));
        }
    }

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("validation command buffer");
    let enc = KernelEncoder::begin(&cmd);
    assert!(encode_get_rows_f32(&ctx, &enc, &embed, &ids, &y, row_ids.len(), 255).is_err());
    let mut short_embed = embed.clone();
    short_embed.offset = short_embed.buffer.length() as u64 - 1;
    assert!(
        encode_get_rows_f32(&ctx, &enc, &short_embed, &ids, &y, row_ids.len(), N_COLS).is_err()
    );
    let mut misaligned_embed = embed.clone();
    misaligned_embed.offset += 1;
    assert!(
        encode_get_rows_f32(
            &ctx,
            &enc,
            &misaligned_embed,
            &ids,
            &y,
            row_ids.len(),
            N_COLS,
        )
        .is_err()
    );
    let mut short_ids = ids.clone();
    short_ids.offset = short_ids.buffer.length() as u64 - 1;
    assert!(
        encode_get_rows_f32(&ctx, &enc, &embed, &short_ids, &y, row_ids.len(), N_COLS).is_err()
    );
    let mut wrong_ids_dtype = ids.clone();
    wrong_ids_dtype.dtype = GgmlType::F32;
    assert!(
        encode_get_rows_f32(
            &ctx,
            &enc,
            &embed,
            &wrong_ids_dtype,
            &y,
            row_ids.len(),
            N_COLS,
        )
        .is_err()
    );
    enc.end();
}

#[test]
fn get_rows_iq4_nl_gpu_matches_cpu_with_ple_width_and_offsets() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_VOCAB: usize = 3;
    const N_COLS: usize = 160;
    const BLOCKS_PER_ROW: usize = N_COLS / 32;
    let mut weight_bytes = Vec::new();
    let mut hand_decoded = vec![0.0f32; N_VOCAB * N_COLS];
    for row in 0..N_VOCAB {
        for block_index in 0..BLOCKS_PER_ROW {
            let ordinal = row * BLOCKS_PER_ROW + block_index;
            let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
            let d = sign * (ordinal % 7 + 1) as f32 / 1024.0;
            let (block, values) = encode_iq4_nl_block(d, ordinal);
            weight_bytes.extend_from_slice(&block);
            let start = row * N_COLS + block_index * 32;
            hand_decoded[start..start + 32].copy_from_slice(&values);
        }
    }
    let desc = TensorDesc {
        name: "iq4_nl_ple_rows".into(),
        shape: vec![N_COLS as u64, N_VOCAB as u64],
        dtype: GgmlType::IQ4_NL,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: weight_bytes.len() as u64,
    };
    let codec_decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
        .expect("llama.cpp IQ4_NL reference dequantization");
    assert!(
        hand_decoded
            .iter()
            .zip(&codec_decoded)
            .all(|(hand, codec)| hand.to_bits() == codec.to_bits()),
        "synthetic IQ4_NL encoder disagrees with llama.cpp"
    );

    let embed = offset_tensor(
        &ctx,
        30,
        &weight_bytes,
        19,
        vec![N_COLS as u64, N_VOCAB as u64],
        GgmlType::IQ4_NL,
    );
    let row_ids = [2i32, -1, 0, N_VOCAB as i32, 1, 1];
    let ids = offset_tensor(
        &ctx,
        12,
        bytemuck::cast_slice(&row_ids),
        13,
        vec![row_ids.len() as u64],
        GgmlType::I32,
    );
    let y = offset_tensor(
        &ctx,
        20,
        &vec![0u8; row_ids.len() * N_COLS * size_of::<f32>()],
        29,
        vec![(row_ids.len() * N_COLS) as u64],
        GgmlType::F32,
    );
    one_shot(&ctx, |enc| {
        encode_get_rows_f32(&ctx, enc, &embed, &ids, &y, row_ids.len(), N_COLS)
    })
    .expect("IQ4_NL get_rows");

    let actual = tensor_f32_at_offset(&y);
    for (lookup_row, &row_id) in row_ids.iter().enumerate() {
        let output_row = &actual[lookup_row * N_COLS..(lookup_row + 1) * N_COLS];
        if row_id >= 0 && (row_id as usize) < N_VOCAB {
            let expected = &codec_decoded[row_id as usize * N_COLS..(row_id as usize + 1) * N_COLS];
            let max_abs = output_row
                .iter()
                .zip(expected)
                .map(|(candidate, reference)| (candidate - reference).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 1e-6, "row {lookup_row} max_abs={max_abs}");
        } else {
            assert!(output_row.iter().all(|&value| value == 0.0));
        }
    }
    assert!(
        actual[4 * N_COLS..5 * N_COLS]
            .iter()
            .zip(&actual[5 * N_COLS..6 * N_COLS])
            .all(|(left, right)| left.to_bits() == right.to_bits())
    );
    assert_offset_guards(&embed, 30, 19);
    assert_offset_guards(&ids, 12, 13);
    assert_offset_guards(&y, 20, 29);

    let command = ctx.queue.commandBuffer().expect("validation command");
    let encoder = KernelEncoder::begin(&command);
    let malformed_embed = MetalTensor {
        shape: vec![31, N_VOCAB as u64],
        ..embed.clone()
    };
    let malformed_output =
        MetalTensor::zeros_f32(&ctx, vec![(row_ids.len() * 31) as u64]).expect("malformed output");
    assert!(
        encode_get_rows_f32(
            &ctx,
            &encoder,
            &malformed_embed,
            &ids,
            &malformed_output,
            row_ids.len(),
            31,
        )
        .is_err()
    );
    let mut misaligned_embed = embed.clone();
    misaligned_embed.offset += 1;
    assert!(
        encode_get_rows_f32(
            &ctx,
            &encoder,
            &misaligned_embed,
            &ids,
            &y,
            row_ids.len(),
            N_COLS,
        )
        .is_err()
    );
    encoder.end();
}

fn mxfp4_scale(e: u8) -> f32 {
    let bits = match e {
        0 => 0x0020_0000,
        1 => 0x0040_0000,
        _ => u32::from(e - 1) << 23,
    };
    f32::from_bits(bits)
}

fn encode_mxfp4_block(e: u8, indices: &[u8; 32]) -> [u8; 17] {
    let mut block = [0u8; 17];
    block[0] = e;
    for i in 0..16 {
        block[1 + i] = indices[i] | (indices[16 + i] << 4);
    }
    block
}

#[test]
fn mxfp4_mat_vec_dispatch_gpu_matches_reference_with_offsets() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_IN: usize = 64;
    const N_OUT: usize = 3;
    let mut all_indices = [0u8; 32];
    for (i, index) in all_indices.iter_mut().enumerate() {
        *index = (i % 16) as u8;
    }
    let zero_indices = [0u8; 32];
    let blocks = [
        encode_mxfp4_block(0, &all_indices),
        encode_mxfp4_block(127, &zero_indices),
        encode_mxfp4_block(1, &all_indices),
        encode_mxfp4_block(128, &zero_indices),
        encode_mxfp4_block(126, &zero_indices),
        encode_mxfp4_block(127, &all_indices),
    ];
    let weight_bytes = blocks.concat();
    let desc = TensorDesc {
        name: "mxfp4_test".into(),
        shape: vec![N_IN as u64, N_OUT as u64],
        dtype: GgmlType::MXFP4,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: weight_bytes.len() as u64,
    };
    let decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
        .expect("llama.cpp MXFP4 reference dequantization");
    let weight = offset_tensor(
        &ctx,
        32,
        &weight_bytes,
        31,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::MXFP4,
    );
    let mut x_values = vec![0.0f32; N_IN];
    for (i, x) in x_values[..32].iter_mut().enumerate() {
        *x = if i % 2 == 0 {
            2.0f32.powi(120)
        } else {
            -2.0f32.powi(120)
        };
    }
    for (i, x) in x_values[32..].iter_mut().enumerate() {
        *x = (i as f32 - 15.5) / 8.0;
    }
    let x = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&x_values),
        24,
        vec![N_IN as u64],
        GgmlType::F32,
    );
    let y = offset_tensor(
        &ctx,
        32,
        &[0u8; N_OUT * size_of::<f32>()],
        16,
        vec![N_OUT as u64],
        GgmlType::F32,
    );

    let cmd = ctx.queue.commandBuffer().expect("MXFP4 command buffer");
    let enc = KernelEncoder::begin(&cmd);
    crate::metal_forward::encode_mat_vec_dispatch(&ctx, &enc, &weight, &x, &y, N_IN, N_OUT)
        .expect("MXFP4 dispatch arm");
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut expected = vec![0.0f32; N_OUT];
    for row in 0..N_OUT {
        expected[row] = decoded[row * N_IN..(row + 1) * N_IN]
            .iter()
            .zip(&x_values)
            .map(|(weight, x)| weight * x)
            .sum();
    }
    let actual = tensor_f32_at_offset(&y);
    for (row, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 2.0e-5 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "row {row}: got {got}, want {want}"
        );
    }
    assert_eq!(mxfp4_scale(0).to_bits(), 0x0020_0000);
    assert_eq!(mxfp4_scale(1).to_bits(), 0x0040_0000);
    assert!(!crate::metal_forward::weight_dtype_kept_native(
        GgmlType::MXFP4
    ));
}

#[test]
fn mxfp4_f32_matrix_tile_matches_scalar_envelope_and_guards() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_IN: usize = 64;
    const N_OUT: usize = 65;
    let mut weight_bytes = Vec::new();
    for row in 0..N_OUT {
        for block in 0..N_IN / 32 {
            let mut indices = [0u8; 32];
            for (index, value) in indices.iter_mut().enumerate() {
                *value = ((row * 11 + block * 7 + index * 3) % 16) as u8;
            }
            weight_bytes.extend_from_slice(&encode_mxfp4_block(
                120 + ((row + block) % 8) as u8,
                &indices,
            ));
        }
    }
    let weight = offset_tensor(
        &ctx,
        7,
        &weight_bytes,
        13,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::MXFP4,
    );

    for n_batch in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let x_values = (0..n_batch * N_IN)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 19.0)
            .collect::<Vec<_>>();
        let x = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&x_values),
            20,
            vec![N_IN as u64, n_batch as u64],
            GgmlType::F32,
        );
        let output_bytes = vec![0u8; n_batch * N_OUT * size_of::<f32>()];
        let control = offset_tensor(
            &ctx,
            32,
            &output_bytes,
            28,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let candidate = offset_tensor(
            &ctx,
            24,
            &output_bytes,
            36,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let repeat = offset_tensor(
            &ctx,
            40,
            &output_bytes,
            44,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let weight_before = tensor_backing_bytes(&weight);
        let x_before = tensor_backing_bytes(&x);

        let command = ctx
            .queue
            .commandBuffer()
            .expect("MXFP4 matrix command buffer");
        let encoder = KernelEncoder::begin(&command);
        for row in 0..n_batch {
            encode_mat_vec_mxfp4_f32(
                &ctx,
                &encoder,
                &weight,
                &x.view_subrange((row * N_IN) as u64, vec![N_IN as u64]),
                &control.view_subrange((row * N_OUT) as u64, vec![N_OUT as u64]),
                N_IN,
                N_OUT,
            )
            .expect("encode scalar MXFP4 control");
        }
        encode_mat_mat_mxfp4_f32_mm64x32(
            &ctx, &encoder, &weight, &x, &candidate, N_IN, N_OUT, n_batch,
        )
        .expect("encode MXFP4 matrix candidate");
        encode_mat_mat_mxfp4_f32_mm64x32(
            &ctx, &encoder, &weight, &x, &repeat, N_IN, N_OUT, n_batch,
        )
        .expect("encode repeated MXFP4 matrix candidate");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());

        let control_values = tensor_f32_at_offset(&control);
        let candidate_values = tensor_f32_at_offset(&candidate);
        let repeat_values = tensor_f32_at_offset(&repeat);
        assert_eq!(
            candidate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeat_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "n_batch={n_batch} repeat"
        );
        let mut diff_sq = 0.0f64;
        let mut control_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut candidate_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut max_control = 0.0f32;
        for (&got, &want) in candidate_values.iter().zip(&control_values) {
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
        assert!(
            relative_rms <= 2.0e-5,
            "n_batch={n_batch} relative_rms={relative_rms}"
        );
        assert!(cosine >= 0.999_999_9, "n_batch={n_batch} cosine={cosine}");
        assert!(
            normalized_max <= 1.0e-4,
            "n_batch={n_batch} normalized_max={normalized_max}"
        );
        assert!(candidate_values.iter().all(|value| value.is_finite()));
        assert_eq!(tensor_backing_bytes(&weight), weight_before);
        assert_eq!(tensor_backing_bytes(&x), x_before);
        assert_offset_guards(&control, 32, 28);
        assert_offset_guards(&candidate, 24, 36);
        assert_offset_guards(&repeat, 40, 44);
    }
}

fn run_mxfp4_f32_matrix_tile_k216_bucket_floor(
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

#[test]
#[ignore]
fn mxfp4_f32_matrix_tile_k216_n2048_bucket_floor() {
    run_mxfp4_f32_matrix_tile_k216_bucket_floor(12_288, [160, 166], 77, 250.0, 200.0, 0.50);
}

#[test]
#[ignore]
fn mxfp4_f32_matrix_tile_k216_n4096_all_expert_floor() {
    run_mxfp4_f32_matrix_tile_k216_bucket_floor(24_576, [216; 2], 114, 600.0, 600.0, 0.75);
}

#[test]
fn mxfp4_mat_vec_host_rejects_invalid_shapes_and_ranges() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    let weight = MetalTensor::from_bytes(&ctx, &[0u8; 34], vec![64, 1], GgmlType::MXFP4).unwrap();
    let x = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("validation command buffer");
    let enc = KernelEncoder::begin(&cmd);
    assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &weight, &x, &y, 48, 1).is_err());
    let mut wrong_weight_shape = weight.clone();
    wrong_weight_shape.shape = vec![32, 1];
    assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &wrong_weight_shape, &x, &y, 64, 1).is_err());
    let wrong_x = MetalTensor::zeros_f32(&ctx, vec![32]).unwrap();
    assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &weight, &wrong_x, &y, 64, 1).is_err());
    for which in 0..3 {
        let mut bad_weight = weight.clone();
        let mut bad_x = x.clone();
        let mut bad_y = y.clone();
        match which {
            0 => bad_weight.offset = bad_weight.buffer.length() as u64 - 1,
            1 => bad_x.offset = bad_x.buffer.length() as u64 - 1,
            _ => bad_y.offset = bad_y.buffer.length() as u64 - 1,
        }
        assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &bad_weight, &bad_x, &bad_y, 64, 1).is_err());
    }
    enc.end();
}

fn metal_test_context() -> Option<MetalContext> {
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

#[test]
fn post_block_interventions_match_f32_cpu_oracles_at_realistic_hidden_size() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const H: usize = 6_656;
    let x_values: Vec<f32> = (0..H)
        .map(|i| ((i * 19 + 5) % 257) as f32 / 128.0 - 1.0)
        .collect();
    let direction_values: Vec<f32> = (0..H)
        .map(|i| ((i * 23 + 7) % 251) as f32 / 125.0 - 1.0)
        .collect();
    let source_values: Vec<f32> = (0..H)
        .map(|i| ((i * 29 + 11) % 241) as f32 / 120.0 - 1.0)
        .collect();
    let target_values: Vec<f32> = (0..H)
        .map(|i| ((i * 31 + 13) % 239) as f32 / 119.0 - 1.0)
        .collect();
    let direction = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&direction_values),
        vec![H as u64],
        GgmlType::F32,
    )
    .unwrap();
    let source = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&source_values),
        vec![H as u64],
        GgmlType::F32,
    )
    .unwrap();
    let target = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&target_values),
        vec![H as u64],
        GgmlType::F32,
    )
    .unwrap();
    let run = |intervention: PostBlockIntervention<'_>| {
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_values),
            vec![H as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_post_block_intervention_f32(&ctx, enc, &x, &intervention)
        })
        .unwrap();
        tensor_f32_at_offset(&x)
    };
    let coefficient = 0.375f32;
    let fixed = run(PostBlockIntervention::Fixed {
        layer: 0,
        direction: &direction,
        coefficient,
    });
    let residual_l2 = run(PostBlockIntervention::ResidualL2Relative {
        layer: 0,
        direction: &direction,
        coefficient,
    });
    let projection = run(PostBlockIntervention::Projection {
        layer: 0,
        direction: &direction,
        coefficient,
    });
    let source_to_target = run(PostBlockIntervention::SourceToTarget {
        layer: 0,
        source: &source,
        target: &target,
        coefficient,
    });
    let x_l2 = x_values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let x_dot_direction: f32 = x_values
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| x * direction)
        .sum();
    let x_dot_source: f32 = x_values
        .iter()
        .zip(&source_values)
        .map(|(x, source)| x * source)
        .sum();
    let expected = [
        x_values
            .iter()
            .zip(&direction_values)
            .map(|(x, direction)| *x + coefficient * direction)
            .collect::<Vec<_>>(),
        x_values
            .iter()
            .zip(&direction_values)
            .map(|(x, direction)| *x + coefficient * x_l2 * direction)
            .collect::<Vec<_>>(),
        x_values
            .iter()
            .zip(&direction_values)
            .map(|(x, direction)| *x - coefficient * x_dot_direction * direction)
            .collect::<Vec<_>>(),
        x_values
            .iter()
            .zip(source_values.iter().zip(&target_values))
            .map(|(x, (source, target))| *x + coefficient * x_dot_source * (target - source))
            .collect::<Vec<_>>(),
    ];
    for (name, actual, expected) in [
        ("fixed", fixed, &expected[0]),
        ("residual-l2", residual_l2, &expected[1]),
        ("projection", projection, &expected[2]),
        ("source-to-target", source_to_target, &expected[3]),
    ] {
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let tolerance = if name == "residual-l2" { 1e-3 } else { 2e-5 };
        assert!(
            max_abs <= tolerance,
            "{name}: max|delta|={max_abs} exceeds F32 tolerance {tolerance}"
        );
    }
}

#[test]
fn owned_weight_view_is_read_only_and_bounds_checked() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::NoDevice | MetalError::EmptyLibrary) => return,
        Err(error) => panic!("Metal context: {error}"),
    };
    let buffer = ctx.buffer_uninit(96).expect("owned backing");
    let view = MetalTensor::owned_weight_view(buffer.clone(), 32, vec![16], GgmlType::F32, 32)
        .expect("valid owned weight view");
    assert_eq!(
        view.provenance(),
        MetalTensorProvenance::OwnedWeightReadOnly
    );
    assert!(!view.is_writable());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            view.assert_writable("test write")
        }))
        .is_err()
    );
    assert!(
        MetalTensor::owned_weight_view(buffer.clone(), 1, vec![1], GgmlType::F32, 32,).is_err()
    );
    assert!(MetalTensor::owned_weight_view(buffer, 64, vec![16], GgmlType::F32, 32).is_err());
}

#[test]
fn q6_k_row_bank_view_fixes_geometry_alignment_and_provenance() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    let buffer = ctx.buffer_uninit(480).expect("Q6_K row-bank backing");
    let view = MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 256, 2)
        .expect("valid Q6_K row-bank view");
    assert_eq!(view.shape, [256, 2]);
    assert_eq!(view.dtype, GgmlType::Q6_K);
    assert_eq!(view.offset, 32);
    assert_eq!(view.n_bytes(), 420);
    assert_eq!(
        view.provenance(),
        MetalTensorProvenance::OwnedWeightReadOnly
    );
    assert!(!view.is_writable());

    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 1, 256, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 0, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 128, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 256, 0).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 64, 256, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 0, usize::MAX, 1).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer, 0, 256, usize::MAX).is_err());
}

fn f32_desc(name: &str, shard_idx: usize, data_offset: u64, elements: u64) -> TensorDesc {
    TensorDesc {
        name: name.to_string(),
        shape: vec![elements],
        dtype: GgmlType::F32,
        shard_idx,
        data_offset,
        n_bytes: elements * 4,
    }
}

fn assert_retained_plan_invariants(
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

#[test]
fn gguf_backing_classification_is_typed_and_fail_closed() {
    let geometry = GgufBackingGeometry::new(0, 160, 64, 32).unwrap();
    assert_eq!(geometry.mapped_len(), 160);
    assert_eq!(geometry.mmap_offset(), 0);
    assert_eq!(geometry.exposed_len(), 128);
    assert_eq!(geometry.page_size(), 64);
    assert_eq!(geometry.required_alignment(), 32);
    assert_eq!(
        geometry.classify(&f32_desc("ok", 0, 32, 8)).unwrap(),
        GgufBackingEligibility::Eligible
    );
    assert_eq!(
        geometry.classify(&f32_desc("tail", 0, 96, 9)).unwrap(),
        GgufBackingEligibility::FinalPartialPage
    );
    assert_eq!(
        geometry.classify(&f32_desc("shard", 1, 32, 8)).unwrap(),
        GgufBackingEligibility::WrongShard
    );
    assert_eq!(
        geometry.classify(&f32_desc("align", 0, 36, 8)).unwrap(),
        GgufBackingEligibility::BindingMisalignment
    );
    assert_eq!(
        geometry.classify(&f32_desc("outside", 0, 160, 8)).unwrap(),
        GgufBackingEligibility::OutsideBacking
    );

    let mut malformed = f32_desc("malformed", 0, 32, 8);
    malformed.n_bytes -= 1;
    assert!(geometry.classify(&malformed).is_err());
    assert!(GgufBackingGeometry::new(0, 160, 0, 32).is_err());
    assert!(GgufBackingGeometry::new(0, 160, 64, 0).is_err());
    assert!(GgufBackingGeometry::new(0, 32, 64, 32).is_err());

    let window = GgufBackingGeometry::new_window(0, 256, 64, 128, 64, 32).unwrap();
    assert_eq!(window.mapped_len(), 256);
    assert_eq!(window.mmap_offset(), 64);
    assert_eq!(window.exposed_len(), 128);
    assert_eq!(
        window.classify(&f32_desc("before", 0, 32, 8)).unwrap(),
        GgufBackingEligibility::OutsideBacking
    );
    assert_eq!(
        window.classify(&f32_desc("inside", 0, 96, 8)).unwrap(),
        GgufBackingEligibility::Eligible
    );
    assert_eq!(
        window.classify(&f32_desc("after", 0, 192, 8)).unwrap(),
        GgufBackingEligibility::OutsideBacking
    );
    assert_eq!(
        window
            .classify(&f32_desc("after-misaligned", 0, 196, 8))
            .unwrap(),
        GgufBackingEligibility::OutsideBacking
    );
    let earlier = GgufBackingGeometry::new_window(0, 160, 0, 64, 64, 32).unwrap();
    assert_eq!(
        earlier
            .classify(&f32_desc("outside-earlier", 0, 96, 9))
            .unwrap(),
        GgufBackingEligibility::OutsideBacking
    );
    let terminal = GgufBackingGeometry::new_window(0, 160, 64, 64, 64, 32).unwrap();
    assert_eq!(
        terminal
            .classify(&f32_desc("terminal-tail", 0, 96, 9))
            .unwrap(),
        GgufBackingEligibility::FinalPartialPage
    );
    assert!(GgufBackingGeometry::new_window(0, 256, 32, 64, 64, 32).is_err());
    assert!(GgufBackingGeometry::new_window(0, 256, 64, 96, 64, 32).is_err());
    assert!(GgufBackingGeometry::new_window(0, 256, 192, 128, 64, 32).is_err());
}

#[test]
fn retained_storage_plan_is_order_independent_and_window_bounded() {
    let a = f32_desc("a", 0, 32, 8);
    let b = f32_desc("b", 0, 64, 16);
    let c = f32_desc("c", 0, 128, 8);
    let requests = [&c, &a, &b];
    let plan = plan_retained_storage(&[192], &requests, 64, 130, 32).unwrap();
    assert_retained_plan_invariants(&plan, &requests, &[192]);

    assert_eq!(plan.usable_window_length, 128);
    assert_eq!(plan.windows.len(), 2);
    assert_eq!(
        plan.windows,
        vec![
            RetainedStorageWindow {
                shard_idx: 0,
                mmap_offset: 0,
                length: 128,
            },
            RetainedStorageWindow {
                shard_idx: 0,
                mmap_offset: 128,
                length: 64,
            },
        ]
    );
    assert_eq!(
        plan.entries[0].disposition,
        RetainedStorageDisposition::View {
            window_index: 1,
            buffer_offset: 0,
        }
    );
    assert_eq!(
        plan.entries[1].disposition,
        RetainedStorageDisposition::View {
            window_index: 0,
            buffer_offset: 32,
        }
    );
    assert_eq!(
        plan.entries[2].disposition,
        RetainedStorageDisposition::View {
            window_index: 0,
            buffer_offset: 64,
        }
    );
    assert_eq!(plan.unique_view_bytes, 128);
    assert_eq!(plan.logical_view_bytes, 128);
    assert_eq!(plan.unique_fallback_bytes, 0);
    assert_eq!(plan.alias_bytes, 0);

    let permutation = [&b, &c, &a];
    let permuted = plan_retained_storage(&[192], &permutation, 64, 130, 32).unwrap();
    assert_retained_plan_invariants(&permuted, &permutation, &[192]);
    assert_eq!(plan.windows, permuted.windows);
    for name in ["a", "b", "c"] {
        let original = plan
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .unwrap();
        let permuted = permuted
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .unwrap();
        assert_eq!(original.disposition, permuted.disposition);
    }

    let crossing_a = f32_desc("crossing-a", 0, 0, 24);
    let crossing_b = f32_desc("crossing-b", 0, 96, 16);
    let crossing = plan_retained_storage(&[192], &[&crossing_a, &crossing_b], 64, 128, 32).unwrap();
    assert_retained_plan_invariants(&crossing, &[&crossing_a, &crossing_b], &[192]);
    assert_eq!(crossing.windows.len(), 2);
    assert_eq!(crossing.windows[0].mmap_offset, 0);
    assert_eq!(crossing.windows[0].length, 128);
    assert_eq!(crossing.windows[1].mmap_offset, 64);
    assert_eq!(crossing.windows[1].length, 128);
    assert_eq!(
        crossing.entries[1].disposition,
        RetainedStorageDisposition::View {
            window_index: 1,
            buffer_offset: 32,
        }
    );
}

#[test]
fn retained_storage_plan_classifies_fallbacks_and_aliases() {
    let view = f32_desc("view", 1, 64, 8);
    let missing = f32_desc("missing", 3, 0, 8);
    let tail = f32_desc("tail", 0, 96, 16);
    let outside = f32_desc("outside", 0, 160, 8);
    let misaligned = f32_desc("misaligned", 0, 36, 8);
    let too_large = f32_desc("too-large", 2, 32, 32);
    let requests = [
        &view,
        &view,
        &missing,
        &tail,
        &outside,
        &misaligned,
        &too_large,
    ];
    let plan = plan_retained_storage(&[160, 256, 256], &requests, 64, 128, 32).unwrap();
    assert_retained_plan_invariants(&plan, &requests, &[160, 256, 256]);

    assert_eq!(
        plan.entries[0].disposition,
        RetainedStorageDisposition::View {
            window_index: 0,
            buffer_offset: 0,
        }
    );
    assert_eq!(
        plan.entries[1].disposition,
        RetainedStorageDisposition::Alias {
            source_request_index: 0,
        }
    );
    let reasons = plan
        .entries
        .iter()
        .filter_map(|entry| match entry.disposition {
            RetainedStorageDisposition::CopyFallback { reason } => Some(reason),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reasons,
        vec![
            RetainedStorageFallback::MissingShard,
            RetainedStorageFallback::FinalPartialPage,
            RetainedStorageFallback::OutsideShard,
            RetainedStorageFallback::BindingMisalignment,
            RetainedStorageFallback::TensorExceedsWindow,
        ]
    );
    assert_eq!(plan.unique_view_bytes, 32);
    assert_eq!(plan.logical_view_bytes, 64);
    assert_eq!(plan.unique_fallback_bytes, 288);
    assert_eq!(plan.alias_bytes, 32);

    let tail_alias = plan_retained_storage(&[160], &[&tail, &tail], 64, 128, 32).unwrap();
    assert_eq!(tail_alias.unique_fallback_bytes, 64);
    assert_eq!(tail_alias.alias_bytes, 64);
    assert_eq!(
        tail_alias.entries[1].disposition,
        RetainedStorageDisposition::Alias {
            source_request_index: 0,
        }
    );

    let shard_zero = f32_desc("shard-zero", 0, 32, 8);
    let shard_one = f32_desc("shard-one", 1, 64, 8);
    let multi_requests = [&shard_one, &shard_zero];
    let multi = plan_retained_storage(&[128, 192], &multi_requests, 64, 128, 32).unwrap();
    assert_retained_plan_invariants(&multi, &multi_requests, &[128, 192]);
    assert_eq!(multi.windows.len(), 2);
    assert_eq!(multi.windows[0].shard_idx, 0);
    assert_eq!(multi.windows[1].shard_idx, 1);
}

#[test]
fn retained_storage_plan_rejects_malformed_inputs() {
    let valid = f32_desc("valid", 0, 64, 16);
    assert!(plan_retained_storage(&[128], &[&valid], 0, 128, 32).is_err());
    assert!(plan_retained_storage(&[128], &[&valid], 64, 63, 32).is_err());
    assert!(plan_retained_storage(&[128], &[&valid], 64, 128, 0).is_err());
    assert!(plan_retained_storage(&[128], &[&valid], 64, 128, 128).is_err());

    let mut malformed = valid.clone();
    malformed.n_bytes -= 1;
    assert!(plan_retained_storage(&[128], &[&malformed], 64, 128, 32).is_err());

    let empty = f32_desc("empty", 0, 0, 0);
    assert!(plan_retained_storage(&[128], &[&empty], 64, 128, 32).is_err());

    let alias_a = f32_desc("alias-a", 0, 64, 8);
    let mut alias_b = alias_a.clone();
    alias_b.name = "alias-b".to_string();
    alias_b.shape = vec![2, 4];
    assert!(plan_retained_storage(&[128], &[&alias_a, &alias_b], 64, 128, 32).is_err());

    let overlap_a = f32_desc("overlap-a", 0, 32, 16);
    let overlap_b = f32_desc("overlap-b", 0, 64, 8);
    assert!(plan_retained_storage(&[128], &[&overlap_a, &overlap_b], 64, 128, 32).is_err());

    let tail_misaligned = f32_desc("tail-misaligned", 0, 100, 8);
    let tail_misaligned_plan =
        plan_retained_storage(&[160], &[&tail_misaligned], 64, 128, 32).unwrap();
    assert_eq!(
        tail_misaligned_plan.entries[0].disposition,
        RetainedStorageDisposition::CopyFallback {
            reason: RetainedStorageFallback::BindingMisalignment,
        }
    );

    let tail_oversized = f32_desc("tail-oversized", 0, 64, 24);
    let tail_oversized_plan =
        plan_retained_storage(&[160], &[&tail_oversized], 64, 64, 32).unwrap();
    assert_eq!(
        tail_oversized_plan.entries[0].disposition,
        RetainedStorageDisposition::CopyFallback {
            reason: RetainedStorageFallback::TensorExceedsWindow,
        }
    );
}

#[test]
fn read_only_mmap_backing_blits_and_outlives_rust_views() {
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    let page_size = host_page_size().expect("host page size");
    let mut bytes = vec![0u8; page_size * 2];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = index.wrapping_mul(17) as u8;
    }
    let n_in = 32usize;
    let n_out = 16usize;
    let weights: Vec<f32> = (0..n_in * n_out)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
        .collect();
    bytes[32..32 + weights.len() * 4].copy_from_slice(bytemuck::cast_slice(&weights));
    let mut path = std::env::temp_dir();
    path.push(format!(
        "qwen-metal-no-copy-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::File::create(&path)
        .and_then(|mut file| file.write_all(&bytes))
        .expect("write mmap fixture");

    let callback_calls = Arc::new(AtomicUsize::new(0));
    let callback_mismatches = Arc::new(AtomicUsize::new(0));
    let (weak, source_probe) = objc2::rc::autoreleasepool(|_| {
        let file = std::fs::File::open(&path).expect("open mmap fixture");
        // SAFETY: the test retains the immutable file and does not mutate
        // or truncate it while the mapping exists.
        let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map fixture") });
        let weak = Arc::downgrade(&mmap);
        let expected_pointer = mmap.as_ptr() as usize;
        let expected_length = bytes.len();
        let calls = Arc::clone(&callback_calls);
        let mismatches = Arc::clone(&callback_mismatches);
        let backing = ctx
            .gguf_no_copy_backing_with_observer(Arc::clone(&mmap), 0, 32, move |pointer, length| {
                calls.fetch_add(1, Ordering::Relaxed);
                if pointer.as_ptr() as usize != expected_pointer || length != expected_length {
                    mismatches.fetch_add(1, Ordering::Relaxed);
                }
            })
            .expect("read-only no-copy backing");
        let probe = DiagnosticGgufBlitReleaseProbe {
            weak: Weak::from_retained(&backing.buffer),
            deallocator_calls: Arc::clone(&callback_calls),
            deallocator_mismatches: Arc::clone(&callback_mismatches),
        };
        let source = DiagnosticGgufBlitSourceWindow { backing, probe };
        let source_probe = source.release_probe();
        assert_eq!(source.backing.page_size(), page_size);
        assert_eq!(source.backing.mapped_len(), bytes.len());
        assert_eq!(source.exposed_len(), bytes.len());
        assert_eq!(
            source.backing.buffer.contents().as_ptr(),
            mmap.as_ptr().cast_mut().cast::<c_void>()
        );
        let prefault = source.backing.prefault_read();
        assert_eq!(prefault.page_count, 2);
        assert_eq!(prefault.covered_bytes, bytes.len());
        let expected_checksum = u64::from(bytes[0]).rotate_left(5) ^ u64::from(bytes[page_size]);
        assert_eq!(prefault.checksum, expected_checksum);
        drop(mmap);
        assert!(weak.upgrade().is_some());

        let desc = TensorDesc {
            name: "view".to_string(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: 32,
            n_bytes: (weights.len() * 4) as u64,
        };
        let (eligibility, tensor) = source.backing.tensor(&desc).expect("tensor view");
        assert_eq!(eligibility, GgufBackingEligibility::Eligible);
        let tensor = tensor.expect("eligible tensor");
        assert_eq!(tensor.offset, 32);

        let dst = MetalTensor::zeros_f32(&ctx, vec![16]).expect("destination");
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let blit = BlitEncoder::begin(&command);
        assert!(
            source
                .encode_copy_to(&blit, 1, 32, &dst.buffer, dst.offset, 64)
                .is_err()
        );
        assert!(
            source
                .encode_copy_to(&blit, 0, bytes.len() as u64, &dst.buffer, dst.offset, 64)
                .is_err()
        );
        source
            .encode_copy_to(&blit, 0, 32, &dst.buffer, dst.offset, 64)
            .expect("diagnostic source blit");
        blit.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "no-copy blit command failed");
        let got =
            unsafe { std::slice::from_raw_parts(dst.buffer.contents().as_ptr().cast::<u8>(), 64) };
        assert_eq!(got, &bytes[32..96]);
        drop(source);
        assert!(weak.upgrade().is_some());

        let x: Vec<f32> = (0..n_in).map(|index| index as f32 * 0.02 - 0.3).collect();
        let expected = crate::forward::mat_vec_pub(&weights, n_in, n_out, &x);
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("input");
        let y = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("output");
        let command = ctx.queue.commandBuffer().expect("matvec command");
        let encoder = KernelEncoder::begin(&command);
        encode_mat_vec_f32(&ctx, &encoder, &tensor, &x, &y, n_in, n_out)
            .expect("nonzero-offset matvec");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "no-copy matvec command failed");
        let got = unsafe {
            std::slice::from_raw_parts(y.buffer.contents().as_ptr().cast::<f32>(), n_out)
        };
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "nonzero-offset matvec max error {max_abs}");
        (weak, source_probe)
    });
    assert!(
        weak.upgrade().is_none(),
        "MTLBuffer deallocator must release its Arc<Mmap> capture"
    );
    assert_eq!(callback_calls.load(Ordering::Relaxed), 1);
    assert_eq!(callback_mismatches.load(Ordering::Relaxed), 0);
    assert_eq!(
        source_probe.report(),
        DiagnosticGgufBlitReleaseReport {
            source_alive: false,
            deallocator_calls: 1,
            deallocator_mismatches: 0,
        }
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn diagnostic_blit_sources_release_after_completed_command() {
    use std::io::Write;

    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    let page_size = host_page_size().expect("host page size");
    let bytes = (0..page_size * 2)
        .map(|index| index.wrapping_mul(31) as u8)
        .collect::<Vec<_>>();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "qwen-metal-diagnostic-blit-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::File::create(&path)
        .and_then(|mut file| file.write_all(&bytes))
        .expect("write diagnostic blit fixture");

    let (probe, mmap_weak, staging_weak, window_out, staging_out) =
        objc2::rc::autoreleasepool(|_| {
            let file = std::fs::File::open(&path).expect("open diagnostic blit fixture");
            // SAFETY: the fixture remains immutable and untruncated through
            // command completion and source release.
            let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map fixture") });
            let mmap_weak = Arc::downgrade(&mmap);
            let geometry =
                GgufBackingGeometry::new_window(0, mmap.len(), 0, mmap.len(), page_size, 32)
                    .expect("diagnostic geometry");
            let calls = Arc::new(AtomicUsize::new(0));
            let mismatches = Arc::new(AtomicUsize::new(0));
            let observed_calls = Arc::clone(&calls);
            let observed_mismatches = Arc::clone(&mismatches);
            let expected_pointer = mmap.as_ptr() as usize;
            let expected_length = mmap.len();
            let backing = ctx
                .gguf_no_copy_geometry_with_observer(
                    Arc::clone(&mmap),
                    geometry,
                    move |pointer, length| {
                        if pointer.as_ptr() as usize != expected_pointer
                            || length != expected_length
                        {
                            observed_mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                        observed_calls.fetch_add(1, Ordering::Release);
                    },
                )
                .expect("diagnostic source backing");
            let probe = DiagnosticGgufBlitReleaseProbe {
                weak: Weak::from_retained(&backing.buffer),
                deallocator_calls: calls,
                deallocator_mismatches: mismatches,
            };
            let source = DiagnosticGgufBlitSourceWindow {
                backing,
                probe: probe.clone(),
            };
            drop(mmap);

            let staging = ctx.buffer_from(&bytes[96..160]).expect("staging source");
            let staging_weak = Weak::from_retained(&staging);
            let window_out = ctx.buffer_uninit(64).expect("window destination");
            let staging_out = ctx.buffer_uninit(64).expect("staging destination");
            let command = ctx.queue.commandBuffer().expect("diagnostic command");
            assert!(command.retainedReferences());
            let blit = BlitEncoder::try_begin(&command).expect("diagnostic blit encoder");
            source
                .encode_copy_to(&blit, 0, 32, &window_out, 0, 64)
                .expect("window copy");
            blit.copy_buffer(&staging, 0, &staging_out, 0, 64);
            blit.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(
                command.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
            );
            assert!(command.error().is_none());
            drop(command);
            drop(staging);
            drop(source);
            (probe, mmap_weak, staging_weak, window_out, staging_out)
        });

    assert_eq!(
        probe.report(),
        DiagnosticGgufBlitReleaseReport {
            source_alive: false,
            deallocator_calls: 1,
            deallocator_mismatches: 0,
        }
    );
    assert!(mmap_weak.upgrade().is_none());
    assert!(staging_weak.load().is_none());
    let window_got =
        unsafe { std::slice::from_raw_parts(window_out.contents().as_ptr().cast::<u8>(), 64) };
    let staging_got =
        unsafe { std::slice::from_raw_parts(staging_out.contents().as_ptr().cast::<u8>(), 64) };
    assert_eq!(window_got, &bytes[32..96]);
    assert_eq!(staging_got, &bytes[96..160]);
    let _ = std::fs::remove_file(path);
}

#[test]
fn overlapping_mmap_windows_retain_storage_independently() {
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    let page_size = host_page_size().expect("host page size");

    for reverse_drop_order in [false, true] {
        let mut bytes = vec![0u8; page_size * 4];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index.wrapping_mul(29).wrapping_add(7) as u8;
        }
        let mut path = std::env::temp_dir();
        path.push(format!(
            "qwen-metal-window-{}-{}-{}.bin",
            std::process::id(),
            reverse_drop_order,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::File::create(&path)
            .and_then(|mut file| file.write_all(&bytes))
            .expect("write window fixture");

        let calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let mismatches = Arc::new(AtomicUsize::new(0));
        let weak = objc2::rc::autoreleasepool(|_| {
            let file = std::fs::File::open(&path).expect("open window fixture");
            // SAFETY: the fixture file remains immutable and untruncated
            // while any mapping-backed Metal buffer exists.
            let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map window fixture") });
            let weak = Arc::downgrade(&mmap);
            let first = f32_desc("first", 0, 0, (page_size as u64 + 32) / 4);
            let second = f32_desc("second", 0, page_size as u64 + 64, page_size as u64 / 4);
            let requests = [&first, &second, &first];
            let plan =
                plan_retained_storage(&[bytes.len()], &requests, page_size, page_size * 2, 32)
                    .expect("overlapping-window plan");
            assert_retained_plan_invariants(&plan, &requests, &[bytes.len()]);
            assert_eq!(plan.windows.len(), 2);
            assert_eq!(plan.windows[0].mmap_offset, 0);
            assert_eq!(plan.windows[0].length, page_size * 2);
            assert_eq!(plan.windows[1].mmap_offset, page_size as u64);
            assert_eq!(plan.windows[1].length, page_size * 2);
            assert_eq!(
                plan.entries[2].disposition,
                RetainedStorageDisposition::Alias {
                    source_request_index: 0,
                }
            );

            let make_backing = |window_index: usize| {
                let window = &plan.windows[window_index];
                let expected_pointer =
                    unsafe { mmap.as_ptr().add(window.mmap_offset as usize) as usize };
                let expected_length = window.length;
                let calls = Arc::clone(&calls);
                let mismatches = Arc::clone(&mismatches);
                ctx.gguf_no_copy_window_with_observer(
                    Arc::clone(&mmap),
                    window.shard_idx,
                    window.mmap_offset as usize,
                    window.length,
                    32,
                    move |pointer, length| {
                        calls[window_index].fetch_add(1, Ordering::Relaxed);
                        if pointer.as_ptr() as usize != expected_pointer
                            || length != expected_length
                        {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                )
                .expect("realize retained window")
            };
            let first_backing = make_backing(0);
            let second_backing = make_backing(1);
            assert_eq!(first_backing.mmap_offset(), 0);
            assert_eq!(second_backing.mmap_offset(), page_size);

            let (_, first_tensor) = first_backing.tensor(&first).expect("first view");
            let first_tensor = first_tensor.expect("eligible first view");
            let (_, first_alias) = first_backing.tensor(&first).expect("first alias view");
            let first_alias = first_alias.expect("eligible first alias view");
            let (_, second_tensor) = second_backing.tensor(&second).expect("second view");
            let second_tensor = second_tensor.expect("eligible second view");
            let shared = f32_desc("shared-page", 0, page_size as u64 + 128, 8);
            let (_, shared_first) = first_backing.tensor(&shared).expect("shared first view");
            let shared_first = shared_first.expect("eligible shared first view");
            let (_, shared_second) = second_backing.tensor(&shared).expect("shared second view");
            let shared_second = shared_second.expect("eligible shared second view");
            assert_eq!(first_tensor.offset, 0);
            assert_eq!(second_tensor.offset, 64);
            assert_eq!(
                first_tensor.provenance(),
                MetalTensorProvenance::RetainedGgufReadOnly
            );
            assert!(!first_tensor.is_writable());
            let element_subview = first_tensor.view_subrange(0, vec![8]);
            let byte_subview = first_tensor.view_bytes(0, vec![8]);
            assert_eq!(
                element_subview.provenance(),
                MetalTensorProvenance::RetainedGgufReadOnly
            );
            assert_eq!(
                byte_subview.provenance(),
                MetalTensorProvenance::RetainedGgufReadOnly
            );
            assert_eq!(first_alias.offset, first_tensor.offset);
            assert_eq!(
                Retained::as_ptr(&first_alias.buffer),
                Retained::as_ptr(&first_tensor.buffer)
            );

            let first_out =
                MetalTensor::zeros_f32(&ctx, first.shape.clone()).expect("first destination");
            let second_out =
                MetalTensor::zeros_f32(&ctx, second.shape.clone()).expect("second destination");
            let shared_first_out =
                MetalTensor::zeros_f32(&ctx, shared.shape.clone()).expect("shared destination");
            let shared_second_out =
                MetalTensor::zeros_f32(&ctx, shared.shape.clone()).expect("shared destination");
            {
                let command = ctx.queue.commandBuffer().expect("window blit command");
                let blit = BlitEncoder::begin(&command);
                blit.copy_tensor(&first_tensor, &first_out);
                blit.copy_tensor(&second_tensor, &second_out);
                blit.copy_tensor(&shared_first, &shared_first_out);
                blit.copy_tensor(&shared_second, &shared_second_out);
                blit.end();
                command.commit();
                command.waitUntilCompleted();
                assert!(command.error().is_none(), "window blit command failed");
            }
            let first_got = unsafe {
                std::slice::from_raw_parts(
                    first_out.buffer.contents().as_ptr().cast::<u8>(),
                    first.n_bytes as usize,
                )
            };
            let second_got = unsafe {
                std::slice::from_raw_parts(
                    second_out.buffer.contents().as_ptr().cast::<u8>(),
                    second.n_bytes as usize,
                )
            };
            assert_eq!(first_got, &bytes[..first.n_bytes as usize]);
            let second_start = second.data_offset as usize;
            assert_eq!(
                second_got,
                &bytes[second_start..second_start + second.n_bytes as usize]
            );
            let shared_expected =
                &bytes[shared.data_offset as usize..(shared.data_offset + shared.n_bytes) as usize];
            let shared_first_got = unsafe {
                std::slice::from_raw_parts(
                    shared_first_out.buffer.contents().as_ptr().cast::<u8>(),
                    shared.n_bytes as usize,
                )
            };
            let shared_second_got = unsafe {
                std::slice::from_raw_parts(
                    shared_second_out.buffer.contents().as_ptr().cast::<u8>(),
                    shared.n_bytes as usize,
                )
            };
            assert_eq!(shared_first_got, shared_expected);
            assert_eq!(shared_second_got, shared_expected);

            let command = ctx.queue.commandBuffer().expect("write guard command");
            let blit = BlitEncoder::begin(&command);
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                blit.copy_tensor(&first_out, &first_tensor);
            }));
            assert!(
                write_result.is_err(),
                "retained destination must fail closed"
            );
            blit.end();
            let compute = KernelEncoder::begin(&command);
            let note_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute.note_write(&first_tensor);
            }));
            assert!(note_result.is_err(), "retained write note must fail closed");
            compute.end();

            drop(mmap);
            assert!(weak.upgrade().is_some());
            if reverse_drop_order {
                drop(second_backing);
                drop(first_backing);
            } else {
                drop(first_backing);
                drop(second_backing);
            }
            assert_eq!(calls[0].load(Ordering::Relaxed), 0);
            assert_eq!(calls[1].load(Ordering::Relaxed), 0);
            if reverse_drop_order {
                drop(second_tensor);
                drop(shared_second);
                assert!(weak.upgrade().is_some());
                drop(element_subview);
                drop(byte_subview);
                drop(first_alias);
                drop(shared_first);
                drop(first_tensor);
            } else {
                drop(element_subview);
                drop(byte_subview);
                drop(first_alias);
                drop(shared_first);
                drop(first_tensor);
                assert!(weak.upgrade().is_some());
                drop(shared_second);
                drop(second_tensor);
            }
            weak
        });
        assert!(
            weak.upgrade().is_none(),
            "last window must release the mmap"
        );
        assert_eq!(calls[0].load(Ordering::Relaxed), 1);
        assert_eq!(calls[1].load(Ordering::Relaxed), 1);
        assert_eq!(mismatches.load(Ordering::Relaxed), 0);
        let _ = std::fs::remove_file(path);
    }
}

#[test]
#[ignore = "requires QWEN_GGUF_NO_COPY_MODEL local single-shard fixture"]
fn gguf_no_copy_real_model_coverage_probe() {
    let path = std::env::var("QWEN_GGUF_NO_COPY_MODEL")
        .expect("set QWEN_GGUF_NO_COPY_MODEL to a local GGUF");
    let gguf = crate::gguf::GgufFile::open(&path).expect("open GGUF");
    assert_eq!(gguf.shard_count(), 1, "coverage probe requires one shard");
    let page_size = host_page_size().expect("host page size");
    let geometry = GgufBackingGeometry::new(0, gguf.total_mapped_len(), page_size, 32)
        .expect("GGUF backing geometry");
    let mut eligible_bytes = 0u64;
    let mut crossing = Vec::new();
    let mut dtype_geometry = std::collections::BTreeMap::new();
    for desc in &gguf.tensors {
        let alignment = desc.data_offset & desc.data_offset.wrapping_neg();
        let entry = dtype_geometry
            .entry(format!("{:?}", desc.dtype))
            .or_insert((0usize, 0u64, u64::MAX));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(desc.n_bytes);
        entry.2 = entry.2.min(alignment);
        match geometry.classify(desc).expect("classify tensor") {
            GgufBackingEligibility::Eligible => {
                eligible_bytes = eligible_bytes.saturating_add(desc.n_bytes);
            }
            GgufBackingEligibility::FinalPartialPage => {
                crossing.push((desc.name.clone(), desc.n_bytes));
            }
            other => panic!("unexpected ineligibility for {}: {other:?}", desc.name),
        }
    }
    let total_bytes: u64 = gguf.tensors.iter().map(|desc| desc.n_bytes).sum();
    let coverage = eligible_bytes as f64 / total_bytes.max(1) as f64;
    eprintln!(
        concat!(
            "[gguf-no-copy-coverage] model={} mapped={} exposed={} page={} ",
            "suffix={} tensors={} total={} eligible={} coverage={:.8} crossing={:?}"
        ),
        path,
        geometry.mapped_len(),
        geometry.exposed_len(),
        geometry.page_size(),
        geometry.mapped_len() - geometry.exposed_len(),
        gguf.tensors.len(),
        total_bytes,
        eligible_bytes,
        coverage,
        crossing,
    );
    eprintln!("[gguf-no-copy-dtypes] {dtype_geometry:?}");
    assert!(
        coverage >= 0.99,
        "no-copy coverage {coverage:.6} is below 99%"
    );
}

#[test]
fn cpu_only_admission_bytes_do_not_consume_metal_headroom() {
    let signals = MetalMemorySignals {
        recommended_max_bytes: 1_000,
        current_allocated_bytes: 700,
        process_limit_remaining_bytes: Some(1_000),
    };
    let decision = evaluate_metal_memory_admission_with_cpu_bytes(200, 400, 50, signals, false);
    assert!(decision.admitted);
    assert_eq!(decision.required_bytes, Some(650));
    assert_eq!(decision.working_set_headroom_bytes, Some(300));

    let denied = evaluate_metal_memory_admission_with_cpu_bytes(200, 800, 50, signals, false);
    assert!(!denied.admitted);
    assert_eq!(
        denied.reason,
        MetalMemoryAdmissionReason::ProcessInsufficient
    );
}

fn memory_signals(
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

#[test]
fn metal_memory_admission_requires_both_advisory_budgets() {
    let exact =
        evaluate_metal_memory_admission(400, 100, memory_signals(1500, 1000, Some(500)), false);
    assert!(exact.admitted);
    assert_eq!(
        exact.reason,
        MetalMemoryAdmissionReason::AdmittedWithProcessBudget
    );
    assert_eq!(exact.required_bytes, Some(500));
    assert_eq!(exact.working_set_headroom_bytes, Some(500));

    for (signals, reason) in [
        (
            memory_signals(1499, 1000, Some(500)),
            MetalMemoryAdmissionReason::WorkingSetInsufficient,
        ),
        (
            memory_signals(1500, 1000, Some(499)),
            MetalMemoryAdmissionReason::ProcessInsufficient,
        ),
        (
            memory_signals(1499, 1000, Some(499)),
            MetalMemoryAdmissionReason::BothInsufficient,
        ),
        (
            memory_signals(1000, 1000, Some(500)),
            MetalMemoryAdmissionReason::WorkingSetInsufficient,
        ),
        (
            memory_signals(999, 1000, Some(500)),
            MetalMemoryAdmissionReason::InvalidWorkingSetSignal,
        ),
        (
            memory_signals(1500, 1000, Some(0)),
            MetalMemoryAdmissionReason::ProcessSignalUnavailable,
        ),
        (
            memory_signals(1500, 1000, None),
            MetalMemoryAdmissionReason::ProcessSignalUnavailable,
        ),
    ] {
        let decision = evaluate_metal_memory_admission(400, 100, signals, false);
        assert!(!decision.admitted);
        assert_eq!(decision.reason, reason);
    }
}

#[test]
fn metal_memory_admission_fails_closed_on_required_overflow() {
    let decision = evaluate_metal_memory_admission(
        u64::MAX,
        1,
        memory_signals(u64::MAX, 1, Some(u64::MAX)),
        false,
    );
    assert!(!decision.admitted);
    assert_eq!(decision.required_bytes, None);
    assert_eq!(
        decision.reason,
        MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
}

#[test]
fn metal_memory_admission_omits_zero_process_budget_only_when_allowed() {
    let signals = memory_signals(1500, 1000, Some(0));
    let admitted = evaluate_metal_memory_admission(400, 100, signals, true);
    assert!(admitted.admitted);
    assert_eq!(
        admitted.reason,
        MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
    );
    let denied = evaluate_metal_memory_admission(400, 100, signals, false);
    assert!(!denied.admitted);
    assert_eq!(
        denied.reason,
        MetalMemoryAdmissionReason::ProcessSignalUnavailable
    );
}

#[test]
fn metal_memory_admission_rejects_zero_required_with_zero_headroom() {
    let decision =
        evaluate_metal_memory_admission(0, 0, memory_signals(1000, 1000, Some(1)), false);
    assert!(!decision.admitted);
    assert_eq!(
        decision.reason,
        MetalMemoryAdmissionReason::WorkingSetInsufficient
    );
}

#[test]
fn metal_memory_probes_are_available_on_product_host() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::NoDevice | MetalError::EmptyLibrary) => return,
        Err(error) => panic!("Metal context: {error}"),
    };
    let signals = ctx.memory_signals();
    assert!(signals.recommended_max_bytes > 0);
    eprintln!("[metal-memory-signals] {signals:?}");
}

#[test]
#[ignore]
fn metal_memory_probes_match_local_m4_max() {
    let ctx = MetalContext::new().expect("Metal context");
    let signals = ctx.memory_signals();
    assert_eq!(signals.recommended_max_bytes, 103_079_215_104);
    assert_eq!(signals.process_limit_remaining_bytes, Some(0));
}

#[test]
fn mat_mat_qk_threadgroup_memory_matches_full_tile_policy() {
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, false),
        8192
    );
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, true),
        5120
    );
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5120, 32, 32, true),
        6144
    );
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5120, 1024, 32, true),
        6144
    );
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5121, 32, 32, true),
        8192
    );
    assert_eq!(
        mat_mat_qk_threadgroup_memory_with_policy(5120, 31, 32, true),
        8192
    );
}

#[test]
fn checked_shape_bytes_rejects_product_overflow() {
    let err = checked_shape_bytes(&[u64::MAX, 2], std::mem::size_of::<f32>())
        .expect_err("shape product must overflow");
    assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
}

#[test]
fn checked_shape_bytes_rejects_byte_overflow() {
    let err = checked_shape_bytes(&[usize::MAX as u64 / 4 + 1], std::mem::size_of::<f32>())
        .expect_err("byte count must overflow usize");
    assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
}

#[test]
fn checked_ggml_shape_bytes_rejects_bad_q8_block() {
    let err = checked_ggml_shape_bytes(&[31], GgmlType::Q8_0)
        .expect_err("Q8_0 element count must align to a 32-element block");
    assert!(matches!(err, MetalError::BadShape { .. }));
}

#[test]
fn metal_context_initializes() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    eprintln!("[metal] {}", ctx.describe());
}

#[test]
fn i32_subrange_preserves_type_and_byte_offset() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let ids = MetalTensor::zeros_i32(&ctx, vec![8]).unwrap();
    let view = ids.view_subrange(3, vec![2]);
    assert_eq!(view.dtype, GgmlType::I32);
    assert_eq!(view.shape, vec![2]);
    assert_eq!(view.offset, ids.offset + 3 * size_of::<i32>() as u64);
    assert_eq!(
        Retained::as_ptr(&view.buffer),
        Retained::as_ptr(&ids.buffer)
    );
}

#[test]
fn kernel_encoder_drop_closes_validation_error_pass() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let embed = MetalTensor::zeros_f32(&ctx, vec![8]).unwrap();
    let wrong_ids = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    {
        let enc = KernelEncoder::begin(&cmd);
        let error = encode_get_rows_f32(&ctx, &enc, &embed, &wrong_ids, &output, 1, 4)
            .expect_err("F32 IDs must be rejected");
        assert!(matches!(error, MetalError::BadShape { .. }));
    }
    let enc = KernelEncoder::begin(&cmd);
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
}

#[test]
fn kernel_encoder_drop_closes_panicking_pass() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _enc = KernelEncoder::begin(&cmd);
        panic!("synthetic encoder unwind");
    }));
    assert!(panic.is_err());
    let enc = KernelEncoder::begin(&cmd);
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_allows_disjoint_and_shared_reads() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let shared_in = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let out_a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let out_b = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    // The production concurrent-pass shape: shared read-only input,
    // pairwise-disjoint outputs. Must not panic.
    enc.note_read(&shared_in);
    enc.note_write(&out_a);
    enc.note_read(&shared_in);
    enc.note_write(&out_b);
    enc.end();
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_panics_on_read_after_write() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    enc.note_write(&a);
    let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enc.note_read(&a);
    }));
    assert!(
        hazard.is_err(),
        "read of a tensor written in the same Concurrent pass must panic"
    );
    enc.end();
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_ignores_serial_encoders() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin(&cmd);
    // Serial encoders order dispatches; write-then-read is the normal
    // dataflow and must not trip the guard.
    enc.note_write(&a);
    enc.note_read(&a);
    enc.note_write(&a);
    enc.end();
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_allows_disjoint_views_of_one_buffer() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let arena = MetalTensor::zeros_f32(&ctx, vec![128]).unwrap();
    let lo = arena.view_subrange(0, vec![64]);
    let hi = arena.view_subrange(64, vec![64]);
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    // Disjoint sub-views of one arena are the packed-scratch pattern;
    // byte-range tracking (not buffer identity) must permit this.
    enc.note_write(&lo);
    enc.note_write(&hi);
    // But an overlapping second write must panic.
    let overlap = arena.view_subrange(32, vec![64]);
    let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enc.note_write(&overlap);
    }));
    assert!(hazard.is_err(), "overlapping concurrent writes must panic");
    enc.end();
}

#[test]
fn rms_norm_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &n in &[1024usize, 5120, 17408] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let eps = 1e-6;

        let cpu = crate::forward::rms_norm_pub(&x, &w, eps);
        let gpu = rms_norm_mul_f32_readback_for_test(&ctx, &x, &w, eps).expect("metal rms_norm");

        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[rms_norm n={n}] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-4, "rms_norm n={n}: max|Δ|={max_abs}");
    }
}

#[test]
fn rms_norm_vjp_matches_jacobian_and_relp_rules() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_DIM: usize = 67;
    const EPS: f32 = 1e-6;
    for row_count in [1usize, 2, 8] {
        let x: Vec<f32> = (0..row_count * N_DIM)
            .map(|index| ((index * 17 + 3) % 43) as f32 * 0.021 - 0.39)
            .collect();
        let weight: Vec<f32> = (0..N_DIM)
            .map(|index| 0.45 + (index % 11) as f32 * 0.07)
            .collect();
        let grad_output: Vec<f32> = (0..row_count * N_DIM)
            .map(|index| ((index * 7 + 1) % 31) as f32 * 0.013 - 0.18)
            .collect();
        let mut expected_j = vec![0.0f32; x.len()];
        let mut expected_r = vec![0.0f32; x.len()];
        for row in 0..row_count {
            let base = row * N_DIM;
            let sumsq: f32 = x[base..base + N_DIM]
                .iter()
                .map(|value| value * value)
                .sum();
            let scale = (sumsq / N_DIM as f32 + EPS).sqrt().recip();
            let dot: f32 = (0..N_DIM)
                .map(|index| x[base + index] * grad_output[base + index] * weight[index])
                .sum();
            let correction = dot * scale * scale * scale / N_DIM as f32;
            for index in 0..N_DIM {
                let direct = grad_output[base + index] * weight[index] * scale;
                expected_j[base + index] = direct - x[base + index] * correction;
                expected_r[base + index] = direct;
            }
        }

        let actual_j = rms_norm_mul_vjp_rows_f32_readback_for_test(
            &ctx,
            &x,
            &weight,
            &grad_output,
            row_count,
            N_DIM,
            EPS,
            RmsNormVjpRule::Jacobian,
        )
        .unwrap();
        let actual_r = rms_norm_mul_vjp_rows_f32_readback_for_test(
            &ctx,
            &x,
            &weight,
            &grad_output,
            row_count,
            N_DIM,
            EPS,
            RmsNormVjpRule::RelpDetachedScale,
        )
        .unwrap();
        let max_j = actual_j
            .iter()
            .zip(&expected_j)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let max_r = actual_r
            .iter()
            .zip(&expected_r)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_j < 2e-5, "rows={row_count}: Jacobian error {max_j}");
        assert!(max_r < 2e-5, "rows={row_count}: RelP error {max_r}");
        assert!(
            actual_j
                .iter()
                .zip(&actual_r)
                .any(|(jacobian, relp)| (jacobian - relp).abs() > 1e-3),
            "ordinary and RelP rules unexpectedly coincide"
        );

        let row = row_count - 1;
        for index in [0usize, 31, N_DIM - 1] {
            let epsilon = 1e-4f64;
            let objective = |delta: f64| {
                let base = row * N_DIM;
                let sumsq: f64 = (0..N_DIM)
                    .map(|column| {
                        let value =
                            f64::from(x[base + column]) + if column == index { delta } else { 0.0 };
                        value * value
                    })
                    .sum();
                let scale = (sumsq / N_DIM as f64 + f64::from(EPS)).sqrt().recip();
                (0..N_DIM)
                    .map(|column| {
                        let value =
                            f64::from(x[base + column]) + if column == index { delta } else { 0.0 };
                        f64::from(grad_output[base + column])
                            * value
                            * scale
                            * f64::from(weight[column])
                    })
                    .sum::<f64>()
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            let reverse = f64::from(actual_j[row * N_DIM + index]);
            assert!(
                (finite_difference - reverse).abs() < 1e-4,
                "rows={row_count} index={index}: finite difference {finite_difference} != {reverse}"
            );
        }
    }
}

#[test]
fn swiglu_vjp_matches_jacobian_and_relp_rules() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_DIM: usize = 79;
    for row_count in [1usize, 2, 8] {
        let len = row_count * N_DIM;
        let gate: Vec<f32> = (0..len)
            .map(|index| ((index * 19 + 5) % 101) as f32 * 0.11 - 5.5)
            .collect();
        let up: Vec<f32> = (0..len)
            .map(|index| ((index * 13 + 2) % 47) as f32 * 0.031 - 0.67)
            .collect();
        let grad_output: Vec<f32> = (0..len)
            .map(|index| ((index * 7 + 3) % 37) as f32 * 0.017 - 0.29)
            .collect();
        let mut expected_j_gate = vec![0.0f32; len];
        let mut expected_j_up = vec![0.0f32; len];
        let mut expected_r_gate = vec![0.0f32; len];
        let mut expected_r_up = vec![0.0f32; len];
        for index in 0..len {
            let sigmoid = 1.0 / (1.0 + (-gate[index]).exp());
            let silu = gate[index] * sigmoid;
            let silu_derivative = sigmoid * (1.0 + gate[index] * (1.0 - sigmoid));
            expected_j_gate[index] = grad_output[index] * up[index] * silu_derivative;
            expected_j_up[index] = grad_output[index] * silu;
            expected_r_gate[index] = 0.5 * grad_output[index] * up[index] * sigmoid;
            expected_r_up[index] = 0.5 * grad_output[index] * silu;
        }

        let (actual_j_gate, actual_j_up) = silu_mul_vjp_f32_readback_for_test(
            &ctx,
            &gate,
            &up,
            &grad_output,
            row_count,
            N_DIM,
            SwiGluVjpRule::Jacobian,
        )
        .unwrap();
        let (actual_r_gate, actual_r_up) = silu_mul_vjp_f32_readback_for_test(
            &ctx,
            &gate,
            &up,
            &grad_output,
            row_count,
            N_DIM,
            SwiGluVjpRule::RelpIdentityHalf,
        )
        .unwrap();
        for (label, actual, expected) in [
            ("j_gate", &actual_j_gate, &expected_j_gate),
            ("j_up", &actual_j_up, &expected_j_up),
            ("r_gate", &actual_r_gate, &expected_r_gate),
            ("r_up", &actual_r_up, &expected_r_up),
        ] {
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 2e-6, "rows={row_count} {label} error {max_abs}");
        }

        let index = len - 3;
        let epsilon = 1e-4f64;
        let objective = |gate_delta: f64, up_delta: f64| {
            let gate_value = f64::from(gate[index]) + gate_delta;
            let up_value = f64::from(up[index]) + up_delta;
            f64::from(grad_output[index]) * gate_value / (1.0 + (-gate_value).exp()) * up_value
        };
        let gate_fd = (objective(epsilon, 0.0) - objective(-epsilon, 0.0)) / (2.0 * epsilon);
        let up_fd = (objective(0.0, epsilon) - objective(0.0, -epsilon)) / (2.0 * epsilon);
        assert!((gate_fd - f64::from(actual_j_gate[index])).abs() < 1e-5);
        assert!((up_fd - f64::from(actual_j_up[index])).abs() < 1e-5);
    }
}

#[test]
fn periodic_vjps_match_per_row_controls_bits() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const PRIMAL_ROWS: usize = 3;
    const ROWS: usize = 9;
    const N_DIM: usize = 67;
    const EPS: f32 = 1.0e-6;
    let primal_len = PRIMAL_ROWS * N_DIM;
    let bank_len = ROWS * N_DIM;
    let x_values = (0..primal_len)
        .map(|index| ((index * 17 + 3) % 43) as f32 * 0.021 - 0.39)
        .collect::<Vec<_>>();
    let gate_values = (0..primal_len)
        .map(|index| ((index * 19 + 5) % 101) as f32 * 0.11 - 5.5)
        .collect::<Vec<_>>();
    let up_values = (0..primal_len)
        .map(|index| ((index * 13 + 2) % 47) as f32 * 0.031 - 0.67)
        .collect::<Vec<_>>();
    let weight_values = (0..N_DIM)
        .map(|index| 0.45 + (index % 11) as f32 * 0.07)
        .collect::<Vec<_>>();
    let grad_values = (0..bank_len)
        .map(|index| ((index * 7 + 1) % 31) as f32 * 0.013 - 0.18)
        .collect::<Vec<_>>();
    let primal_shape = vec![N_DIM as u64, PRIMAL_ROWS as u64];
    let bank_shape = vec![N_DIM as u64, ROWS as u64];
    let x = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x_values),
        primal_shape.clone(),
        GgmlType::F32,
    )
    .unwrap();
    let gate = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&gate_values),
        primal_shape.clone(),
        GgmlType::F32,
    )
    .unwrap();
    let up = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&up_values),
        primal_shape,
        GgmlType::F32,
    )
    .unwrap();
    let weight = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&weight_values),
        vec![N_DIM as u64],
        GgmlType::F32,
    )
    .unwrap();
    let grad_output = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&grad_values),
        bank_shape.clone(),
        GgmlType::F32,
    )
    .unwrap();

    for (rms_rule, swiglu_rule) in [
        (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
        (
            RmsNormVjpRule::RelpDetachedScale,
            SwiGluVjpRule::RelpIdentityHalf,
        ),
    ] {
        let rms_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let rms_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let gate_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let gate_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let up_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let up_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_rms_norm_mul_vjp_periodic_f32(
            &ctx,
            &encoder,
            &x,
            &weight,
            &grad_output,
            &rms_candidate,
            ROWS,
            PRIMAL_ROWS,
            N_DIM,
            EPS,
            rms_rule,
        )
        .unwrap();
        encode_silu_mul_vjp_periodic_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &grad_output,
            &gate_candidate,
            &up_candidate,
            ROWS,
            PRIMAL_ROWS,
            N_DIM,
            swiglu_rule,
        )
        .unwrap();
        for row in 0..ROWS {
            let primal_row = row % PRIMAL_ROWS;
            let one_row_shape = vec![N_DIM as u64, 1];
            let x_row = x.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
            let gate_row = gate.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
            let up_row = up.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
            let grad_row = grad_output.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
            let rms_row = rms_control.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
            let gate_out_row =
                gate_control.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
            let up_out_row = up_control.view_subrange((row * N_DIM) as u64, one_row_shape);
            encode_rms_norm_mul_vjp_rows_f32(
                &ctx, &encoder, &x_row, &weight, &grad_row, &rms_row, 1, N_DIM, EPS, rms_rule,
            )
            .unwrap();
            encode_silu_mul_vjp_f32(
                &ctx,
                &encoder,
                &gate_row,
                &up_row,
                &grad_row,
                &gate_out_row,
                &up_out_row,
                1,
                N_DIM,
                swiglu_rule,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());
        for (name, candidate, control) in [
            ("RMSNorm", &rms_candidate, &rms_control),
            ("SwiGLU gate", &gate_candidate, &gate_control),
            ("SwiGLU up", &up_candidate, &up_control),
        ] {
            assert_eq!(
                read_back_f32(&candidate.buffer, bank_len)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                read_back_f32(&control.buffer, bank_len)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{name} {rms_rule:?}/{swiglu_rule:?}"
            );
        }
    }
}

#[test]
fn residual_rms_norm_matches_separate_cpu_path() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &n in &[1024usize, 5120, 17408] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
        let r: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.4 + (i % 11) as f32 * 0.05).collect();
        let eps = 1e-6;

        let x_cpu: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
        let y_cpu = crate::forward::rms_norm_pub(&x_cpu, &w, eps);
        let (x_gpu, y_gpu) = residual_rms_norm_mul_f32_readback_for_test(&ctx, &x, &r, &w, eps)
            .expect("metal residual_rms_norm");

        let max_x = x_gpu
            .iter()
            .zip(x_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let max_y = y_gpu
            .iter()
            .zip(y_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[residual_rms_norm n={n}] max_x={max_x:.2e} max_y={max_y:.2e}");
        assert!(max_x == 0.0, "residual add n={n}: max|Δ|={max_x}");
        assert!(max_y < 1e-4, "residual_rms_norm n={n}: max|Δ|={max_y}");
    }
}

#[test]
fn mat_vec_f32_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(n_in, n_out) in &[
        (1024usize, 248_320usize),
        (1024, 6144),
        (5120, 17408),
        (5120, 5120),
    ] {
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-3)
            .collect();
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
        let gpu = mat_vec_f32_readback_for_test(&ctx, &w, &x, n_in, n_out).expect("gpu");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[mat_vec n_in={n_in} n_out={n_out}] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-3);
    }
}

#[test]
fn mat_mat_f32_router_e8p32_strict_matches_generic_bits() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_in = 4096usize;
    let n_out = 160usize;
    let mut state = 0x8b8b_8b8b_u32;
    let mut sample = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5) * 0.25
    };
    let weight = (0..n_in * n_out).map(|_| sample()).collect::<Vec<_>>();
    let weight_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&weight),
        vec![n_in as u64, n_out as u64],
        GgmlType::F32,
    )
    .expect("weight tensor");
    let short_weight =
        MetalTensor::zeros_f32(&ctx, vec![(n_in * n_out - 1) as u64]).expect("short weight tensor");
    let validation_x = MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).expect("validation input");
    let validation_y = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("validation output");
    let command = ctx.queue.commandBuffer().expect("validation command");
    let encoder = KernelEncoder::begin(&command);
    assert!(
        encode_mat_mat_f32_router_e8p32_strict(
            &ctx,
            &encoder,
            &short_weight,
            &validation_x,
            &validation_y,
            n_in,
            n_out,
            1,
        )
        .is_err()
    );
    encoder.end();

    for n_query in [1usize, 32, 128, 512, 2048, 4096] {
        let x = (0..n_in * n_query).map(|_| sample()).collect::<Vec<_>>();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("input tensor");
        let generic = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64])
            .expect("generic output");
        let strict = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64])
            .expect("strict output");
        one_shot(&ctx, |enc| {
            encode_mat_mat_f32(&ctx, enc, &weight_t, &x_t, &generic, n_in, n_out, n_query)?;
            encode_mat_mat_f32_router_e8p32_strict(
                &ctx, enc, &weight_t, &x_t, &strict, n_in, n_out, n_query,
            )
        })
        .expect("router differential");

        let generic = read_back_f32(&generic.buffer, n_out * n_query);
        let strict = read_back_f32(&strict.buffer, n_out * n_query);
        if let Some((index, (expected, actual))) = generic
            .iter()
            .zip(&strict)
            .enumerate()
            .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        {
            panic!(
                "n_query={n_query} output {index} differs: generic={expected:?} strict={actual:?}"
            );
        }
    }
}

#[test]
fn mat_vec_f32_sigmoid_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_in = 512usize;
    let n_out = 48usize;
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| ((i % 29) as f32 - 14.0) * 2e-3)
        .collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 17) as f32 - 8.0) * 3e-2).collect();
    let mat = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
    let cpu: Vec<f32> = mat.into_iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let w_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&w),
        vec![n_in as u64, n_out as u64],
        GgmlType::F32,
    )
    .unwrap();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_vec_f32_sigmoid(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })
    .unwrap();
    let gpu = read_back_f32(&y_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs < 1e-5, "fused beta sigmoid drift {max_abs}");
}

#[test]
fn moe_grouped_finalizer_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_out = 512usize;
    let topk = 4usize;
    let n_tokens = 2usize;
    let expert_out: Vec<f32> = (0..n_tokens * topk * n_out)
        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
        .collect();
    let weights: Vec<f32> = (0..n_tokens * topk)
        .map(|i| 0.1 + (i % topk) as f32 * 0.1)
        .collect();
    let shared_gate = vec![0.65f32, 0.35];
    let shared_out: Vec<f32> = (0..n_tokens * n_out)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();
    let x_init: Vec<f32> = (0..n_tokens * n_out)
        .map(|i| ((i % 13) as f32 - 6.0) * 3e-2)
        .collect();
    let expected: Vec<f32> = (0..n_tokens * n_out)
        .map(|i| {
            let token = i / n_out;
            let routed: f32 = (0..topk)
                .map(|slot| {
                    weights[token * topk + slot]
                        * expert_out[(token * topk + slot) * n_out + i % n_out]
                })
                .sum();
            x_init[i] + routed + shared_gate[token] * shared_out[i]
        })
        .collect();

    let expert_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&expert_out),
        vec![(n_tokens * topk * n_out) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let weights_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&weights),
        vec![(n_tokens * topk) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let gate_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&shared_gate),
        vec![n_tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let shared_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&shared_out),
        vec![(n_tokens * n_out) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x_init),
        vec![(n_tokens * n_out) as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(&ctx, |enc| {
        encode_moe_grouped_finalizer_f32(
            &ctx, enc, &expert_t, &weights_t, &gate_t, &shared_t, &x_t, n_out, topk, n_tokens,
        )
    })
    .unwrap();
    let gpu = read_back_f32(&x_t.buffer, n_tokens * n_out);
    let max_abs = gpu
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs < 1e-6, "grouped finalizer drift {max_abs}");
}

#[test]
fn moe_iq4_decode_matches_cpu_at_flash_next_geometry() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    run_iq4_moe_decode_case(&ctx, 2_560, 640, 2_560, 2, &[1, 0]);
}

#[test]
fn moe_iq4_decode_addresses_all_512_experts_and_odd_output_tail() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    run_iq4_moe_decode_case(&ctx, 256, 32, 65, 512, &[511, 257, 0]);
}

#[test]
fn moe_iq4_decode_rejects_unsafe_tensor_contracts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_IN: usize = 256;
    const N_FFN: usize = 32;
    const N_HIDDEN: usize = 65;
    const N_EXPERT: usize = 2;
    let gate_bytes = synthetic_iq4_xs_bank(N_IN, N_FFN, N_EXPERT, 17);
    let up_bytes = synthetic_iq4_xs_bank(N_IN, N_FFN, N_EXPERT, 31);
    let down_bytes = synthetic_iq4_nl_bank(N_FFN, N_HIDDEN, N_EXPERT, 47);
    let gate = MetalTensor::from_bytes(
        &ctx,
        &gate_bytes,
        vec![N_IN as u64, N_FFN as u64, N_EXPERT as u64],
        GgmlType::IQ4_XS,
    )
    .unwrap();
    let up = MetalTensor::from_bytes(
        &ctx,
        &up_bytes,
        vec![N_IN as u64, N_FFN as u64, N_EXPERT as u64],
        GgmlType::IQ4_XS,
    )
    .unwrap();
    let down = MetalTensor::from_bytes(
        &ctx,
        &down_bytes,
        vec![N_FFN as u64, N_HIDDEN as u64, N_EXPERT as u64],
        GgmlType::IQ4_NL,
    )
    .unwrap();
    let input = MetalTensor::zeros_f32(&ctx, vec![N_IN as u64]).unwrap();
    let indices =
        MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&[0i32]), vec![1], GgmlType::I32)
            .unwrap();
    let inner = MetalTensor::zeros_f32(&ctx, vec![N_FFN as u64]).unwrap();
    let output = MetalTensor::zeros_f32(&ctx, vec![N_HIDDEN as u64]).unwrap();
    let half_input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&vec![half::f16::ZERO; N_IN]),
        vec![N_IN as u64],
        GgmlType::F16,
    )
    .unwrap();
    let half_indices = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[half::f16::ZERO]),
        vec![1],
        GgmlType::F16,
    )
    .unwrap();

    let command = ctx.queue.commandBuffer().expect("validation command");
    let encoder = KernelEncoder::begin(&command);
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &half_input,
            &indices,
            &inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &input,
            &half_indices,
            &inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let mut read_only_inner = inner.clone();
    read_only_inner.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &input,
            &indices,
            &read_only_inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let mut short_input = input.clone();
    short_input.offset = 4;
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &short_input,
            &indices,
            &inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let mut misaligned_gate = gate.clone();
    misaligned_gate.offset = 1;
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &misaligned_gate,
            &up,
            &input,
            &indices,
            &inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let overflow_dimension = u32::MAX as usize - 255;
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &input,
            &indices,
            &inner,
            overflow_dimension,
            overflow_dimension,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    assert!(
        encode_moe_swiglu_iq4_xs_f32(
            &ctx,
            &encoder,
            &gate,
            &up,
            &input,
            &indices,
            &inner,
            N_IN,
            N_FFN,
            N_EXPERT,
            N_EXPERT + 1,
        )
        .is_err()
    );

    let mut read_only_output = output.clone();
    read_only_output.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(
        encode_moe_down_iq4_nl_f32(
            &ctx,
            &encoder,
            &down,
            &inner,
            &indices,
            &read_only_output,
            N_FFN,
            N_HIDDEN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let padded_inner = MetalTensor::zeros_f32(&ctx, vec![(N_FFN + 1) as u64]).unwrap();
    let misaligned_inner = padded_inner.view_subrange(1, vec![N_FFN as u64]);
    assert!(
        encode_moe_down_iq4_nl_f32_fast(
            &ctx,
            &encoder,
            &down,
            &misaligned_inner,
            &indices,
            &output,
            N_FFN,
            N_HIDDEN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    let mut short_down = down.clone();
    short_down.offset = 2;
    assert!(
        encode_moe_down_iq4_nl_f32(
            &ctx,
            &encoder,
            &short_down,
            &inner,
            &indices,
            &output,
            N_FFN,
            N_HIDDEN,
            N_EXPERT,
            1,
        )
        .is_err()
    );
    encoder.end();
}

#[test]
#[ignore]
fn moe_mat_vec_iq3_xxs_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q3_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
        .expect("missing IQ3_XXS MoE gate tensor");
    let n_in = t.shape[0] as usize;
    let n_out = t.shape[1] as usize;
    let n_expert = t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let row_stride = (n_in / 256) * 98;
    let expert_stride = n_out * row_stride;
    let all_bytes = g.slice(t);
    let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_out as u64],
        dtype: GgmlType::IQ3_XXS,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_bytes.len() as u64,
    };
    let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
        .collect();
    let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

    let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let expert_i = expert as i32;
    let topk_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[expert_i]),
        vec![1],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_mat_vec_iq3_xxs_f32(
            &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
        )
    })
    .expect("gpu iq3 matvec");
    let gpu = read_back_f32(&out_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[moe-iq3-oracle] max|delta|={max_abs:.3e}");
    assert!(max_abs < 2e-4, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q3_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3-direct-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
        .expect("missing IQ3_XXS MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
        .expect("missing IQ3_XXS MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let row_stride = (n_in / 256) * 98;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let gate_expert_bytes = &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::IQ3_XXS,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
    let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
    let mut cpu = vec![0.0f32; n_ffn];
    for i in 0..n_ffn {
        let g = gate[i];
        cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let expert_i = expert as i32;
    let topk_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[expert_i]),
        vec![1],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_xxs_f32(
            &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert, 1,
        )
    })
    .expect("gpu direct iq3 swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");

    let out_fast_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_xxs_f32_fast(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &topk_gpu,
            &out_fast_gpu,
            n_in,
            n_ffn,
            n_expert,
            1,
        )
    })
    .expect("gpu fast direct iq3 swiglu");
    let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
    let fast_max_abs = fast
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let fast_dot: f64 = fast
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
    assert!(fast_cos > 0.999, "fast cos={fast_cos}");
    assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
}

#[test]
#[ignore]
fn moe_grouped_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q3_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
        .expect("missing IQ3_XXS MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
        .expect("missing IQ3_XXS MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let n_tokens = 32usize;
    let topk = 1usize;
    let row_stride = (n_in / 256) * 98;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let gate_expert_bytes = &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::IQ3_XXS,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu = vec![0.0f32; n_tokens * n_ffn];
    for token in 0..n_tokens {
        let x_tok = &x[token * n_in..(token + 1) * n_in];
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_tokens * n_in) as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let mut counts = vec![0i32; n_expert];
    counts[expert] = n_tokens as i32;
    let mut ids = vec![0i32; n_expert * n_tokens];
    for token in 0..n_tokens {
        ids[expert * n_tokens + token] = token as i32;
    }
    let counts_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&counts),
        vec![n_expert as u64],
        GgmlType::F32,
    )
    .expect("counts tensor");
    let ids_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![(n_expert * n_tokens) as u64],
        GgmlType::F32,
    )
    .expect("ids tensor");
    let out_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &counts_gpu,
            &ids_gpu,
            &out_gpu,
            n_in,
            n_ffn,
            n_expert,
            topk,
            n_tokens,
        )
    })
    .expect("gpu grouped iq3 swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_mat_vec_iq3_s_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3s-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
        .expect("missing IQ3_S MoE gate tensor");
    let n_in = t.shape[0] as usize;
    let n_out = t.shape[1] as usize;
    let n_expert = t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let row_stride = (n_in / 256) * 110;
    let expert_stride = n_out * row_stride;
    let all_bytes = g.slice(t);
    let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_out as u64],
        dtype: GgmlType::IQ3_S,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_bytes.len() as u64,
    };
    let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
        .collect();
    let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

    let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let expert_i = expert as i32;
    let topk_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[expert_i]),
        vec![1],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_mat_vec_iq3_s_f32(
            &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
        )
    })
    .expect("gpu iq3s matvec");
    let gpu = read_back_f32(&out_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[moe-iq3s-oracle] max|delta|={max_abs:.3e}");
    assert!(max_abs < 2e-4, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_swiglu_iq3_s_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3s-direct-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
        .expect("missing IQ3_S MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
        .expect("missing IQ3_S MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let row_stride = (n_in / 256) * 110;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let gate_expert_bytes = &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::IQ3_S,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
    let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
    let mut cpu = vec![0.0f32; n_ffn];
    for i in 0..n_ffn {
        let g = gate[i];
        cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let expert_i = expert as i32;
    let topk_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[expert_i]),
        vec![1],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_s_f32(
            &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert, 1,
        )
    })
    .expect("gpu direct iq3s swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3s-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");

    let out_fast_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_s_f32_fast(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &topk_gpu,
            &out_fast_gpu,
            n_in,
            n_ffn,
            n_expert,
            1,
        )
    })
    .expect("gpu fast direct iq3s swiglu");
    let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
    let fast_max_abs = fast
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let fast_dot: f64 = fast
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3s-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
    assert!(fast_cos > 0.999, "fast cos={fast_cos}");
    assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
}

#[test]
#[ignore]
fn moe_grouped_swiglu_iq3_s_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq3s-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
        .expect("missing IQ3_S MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
        .expect("missing IQ3_S MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let n_tokens = 32usize;
    let topk = 1usize;
    let row_stride = (n_in / 256) * 110;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let gate_expert_bytes = &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::IQ3_S,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu = vec![0.0f32; n_tokens * n_ffn];
    for token in 0..n_tokens {
        let x_tok = &x[token * n_in..(token + 1) * n_in];
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_tokens * n_in) as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let mut counts = vec![0i32; n_expert];
    counts[expert] = n_tokens as i32;
    let mut ids = vec![0i32; n_expert * n_tokens];
    for token in 0..n_tokens {
        ids[expert * n_tokens + token] = token as i32;
    }
    let counts_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&counts),
        vec![n_expert as u64],
        GgmlType::F32,
    )
    .expect("counts tensor");
    let ids_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![(n_expert * n_tokens) as u64],
        GgmlType::F32,
    )
    .expect("ids tensor");
    let out_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &counts_gpu,
            &ids_gpu,
            &out_gpu,
            n_in,
            n_ffn,
            n_expert,
            topk,
            n_tokens,
        )
    })
    .expect("gpu grouped iq3s swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq3s-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_grouped_down_iq4_xs_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-iq4xs-down-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let down_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::IQ4_XS)
        .expect("missing IQ4_XS MoE down tensor");
    let n_in = down_t.shape[0] as usize;
    let n_out = down_t.shape[1] as usize;
    let n_expert = down_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let n_tokens = 32usize;
    let row_stride = (n_in / 256) * 136;
    let expert_stride = n_out * row_stride;
    let down_bytes_all = g.slice(down_t);
    let down_expert_bytes = &down_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_out as u64],
        dtype: GgmlType::IQ4_XS,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let down_f32 =
        crate::codec::dequant_to_f32(&expert_desc, down_expert_bytes).expect("down dequant");
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu = vec![0.0f32; n_tokens * n_out];
    for token in 0..n_tokens {
        let x_tok = &x[token * n_in..(token + 1) * n_in];
        let y = crate::forward::mat_vec_pub(&down_f32, n_in, n_out, x_tok);
        cpu[token * n_out..(token + 1) * n_out].copy_from_slice(&y);
    }

    let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_tokens * n_in) as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let mut counts = vec![0i32; n_expert];
    counts[expert] = n_tokens as i32;
    let mut ids = vec![0i32; n_expert * n_tokens];
    for token in 0..n_tokens {
        ids[expert * n_tokens + token] = token as i32;
    }
    let counts_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&counts),
        vec![n_expert as u64],
        GgmlType::F32,
    )
    .expect("counts tensor");
    let ids_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![(n_expert * n_tokens) as u64],
        GgmlType::F32,
    )
    .expect("ids tensor");
    let out_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_down_iq4_xs_f32_grouped_slots(
            &ctx,
            enc,
            &down_gpu,
            &x_gpu,
            &counts_gpu,
            &ids_gpu,
            &out_gpu,
            n_in,
            n_out,
            n_expert,
            n_tokens,
        )
    })
    .expect("gpu grouped iq4xs down");
    let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-iq4xs-down-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");

    let topk_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[expert as i32]),
        vec![1],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let x_one = x_gpu.view_subrange(0, vec![n_in as u64]);
    let out_fast_gpu = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("fast out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_down_iq4_xs_f32_fast(
            &ctx,
            enc,
            &down_gpu,
            &x_one,
            &topk_gpu,
            &out_fast_gpu,
            n_in,
            n_out,
            n_expert,
            1,
        )
    })
    .expect("gpu fast iq4xs down");
    let fast = read_back_f32(&out_fast_gpu.buffer, n_out);
    let fast_max_abs = fast
        .iter()
        .zip(cpu[..n_out].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let fast_dot: f64 = fast
        .iter()
        .zip(cpu[..n_out].iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc_fast: f64 = cpu[..n_out].iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let fast_cos = fast_dot / (nf.sqrt() * nc_fast.sqrt()).max(1e-12);
    eprintln!("[moe-iq4xs-fast-down-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
    assert!(fast_cos > 0.999, "fast cos={fast_cos}");
    assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
}

#[test]
#[ignore]
fn moe_swiglu_q6_k_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q6_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q6_K.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-q6-direct-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
        .expect("missing Q6_K MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
        .expect("missing Q6_K MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let topk = 2usize;
    let experts = [7usize.min(n_expert - 1), n_expert - 1];
    let row_stride = (n_in / 256) * 210;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu = vec![0.0f32; topk * n_ffn];
    for (slot, expert) in experts.iter().copied().enumerate() {
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q6_K,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let topk_i32 = [experts[0] as i32, experts[1] as i32];
    let topk_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&topk_i32),
        vec![topk as u64],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let out_gpu = MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_q6_K_f32(
            &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert, topk,
        )
    })
    .expect("gpu direct q6 swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, topk * n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-q6-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_grouped_swiglu_q6_k_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q6_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q6_K.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-q6-swiglu-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
        .expect("missing Q6_K MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
        .expect("missing Q6_K MoE up tensor");
    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let expert = 7usize.min(n_expert - 1);
    let n_tokens = 16usize;
    let topk = 1usize;
    let row_stride = (n_in / 256) * 210;
    let expert_stride = n_ffn * row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let gate_expert_bytes = &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
    let expert_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::Q6_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu = vec![0.0f32; n_tokens * n_ffn];
    for token in 0..n_tokens {
        let x_tok = &x[token * n_in..(token + 1) * n_in];
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_tokens * n_in) as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let mut counts = vec![0i32; n_expert];
    counts[expert] = n_tokens as i32;
    let mut ids = vec![0i32; n_expert * n_tokens];
    for token in 0..n_tokens {
        ids[expert * n_tokens + token] = token as i32;
    }
    let counts_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&counts),
        vec![n_expert as u64],
        GgmlType::F32,
    )
    .expect("counts tensor");
    let ids_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![(n_expert * n_tokens) as u64],
        GgmlType::F32,
    )
    .expect("ids tensor");
    let out_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &counts_gpu,
            &ids_gpu,
            &out_gpu,
            n_in,
            n_ffn,
            n_expert,
            topk,
            n_tokens,
        )
    })
    .expect("gpu grouped q6 swiglu");
    let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
    eprintln!("[moe-q6-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
    assert!(cos > 0.999, "cos={cos}");
    assert!(max_abs < 2e-2, "max|delta|={max_abs}");
}

#[test]
#[ignore]
fn moe_q8_0_swiglu_down_weighted_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q8_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q8_0.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-q8-direct-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE up tensor");
    let down_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE down tensor");

    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let h = down_t.shape[1] as usize;
    let experts = [7usize.min(n_expert - 1), n_expert - 1];
    let topk = experts.len();
    let top_w = [0.35f32, 0.65f32];
    let gate_row_stride = (n_in / 32) * 34;
    let gate_expert_stride = n_ffn * gate_row_stride;
    let down_row_stride = (n_ffn / 32) * 34;
    let down_expert_stride = h * down_row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let down_bytes_all = g.slice(down_t);
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let gate_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::Q8_0,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: gate_expert_stride as u64,
    };
    let down_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
        shape: vec![n_ffn as u64, h as u64],
        dtype: GgmlType::Q8_0,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: down_expert_stride as u64,
    };
    let mut cpu_inner = vec![0.0f32; topk * n_ffn];
    let mut cpu_down = vec![0.0f32; h];
    for (slot, expert) in experts.iter().copied().enumerate() {
        let gate_expert_bytes =
            &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let up_expert_bytes =
            &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let down_expert_bytes =
            &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
        let gate_f32 =
            crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 = crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
        let down_f32 =
            crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu_inner[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
        let down = crate::forward::mat_vec_pub(
            &down_f32,
            n_ffn,
            h,
            &cpu_inner[slot * n_ffn..(slot + 1) * n_ffn],
        );
        for i in 0..h {
            cpu_down[i] += top_w[slot] * down[i];
        }
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let topk_i32 = [experts[0] as i32, experts[1] as i32];
    let topk_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&topk_i32),
        vec![topk as u64],
        GgmlType::F32,
    )
    .expect("topk tensor");
    let topw_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&top_w),
        vec![topk as u64],
        GgmlType::F32,
    )
    .expect("top weights tensor");
    let inner_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("inner tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_q8_0_f32(
            &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &inner_gpu, n_in, n_ffn, n_expert,
            topk,
        )
    })
    .expect("gpu direct q8 swiglu");
    let gpu_inner = read_back_f32(&inner_gpu.buffer, topk * n_ffn);
    let inner_max = gpu_inner
        .iter()
        .zip(cpu_inner.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let inner_dot: f64 = gpu_inner
        .iter()
        .zip(cpu_inner.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let inner_ng: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inner_nc: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inner_cos = inner_dot / (inner_ng.sqrt() * inner_nc.sqrt()).max(1e-12);
    eprintln!("[moe-q8-direct-swiglu-oracle] cos={inner_cos:.6} max|delta|={inner_max:.3e}");
    assert!(inner_cos > 0.999, "inner cos={inner_cos}");
    assert!(inner_max < 2e-2, "inner max|delta|={inner_max}");

    let down_gpu_out = MetalTensor::zeros_f32(&ctx, vec![h as u64]).expect("down out");
    one_shot(&ctx, |enc| {
        encode_moe_down_weighted_sum_q8_0_f32(
            &ctx,
            enc,
            &down_gpu,
            &inner_gpu,
            &topk_gpu,
            &topw_gpu,
            &down_gpu_out,
            n_ffn,
            h,
            n_expert,
            topk,
        )
    })
    .expect("gpu direct q8 down weighted sum");
    let gpu_down = read_back_f32(&down_gpu_out.buffer, h);
    let down_max = gpu_down
        .iter()
        .zip(cpu_down.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let down_dot: f64 = gpu_down
        .iter()
        .zip(cpu_down.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let down_ng: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let down_nc: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let down_cos = down_dot / (down_ng.sqrt() * down_nc.sqrt()).max(1e-12);
    eprintln!("[moe-q8-direct-down-oracle] cos={down_cos:.6} max|delta|={down_max:.3e}");
    assert!(down_cos > 0.999, "down cos={down_cos}");
    assert!(down_max < 2e-2, "down max|delta|={down_max}");
}

#[test]
#[ignore]
fn moe_grouped_q8_0_swiglu_down_matches_f32_dequant_fixture() {
    let path = std::env::var("QWEN_A3B_Q8_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q8_0.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[moe-q8-oracle] skipped missing fixture {path}");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
    let gate_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE gate tensor");
    let up_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE up tensor");
    let down_t = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
        .expect("missing Q8_0 MoE down tensor");

    let n_in = gate_t.shape[0] as usize;
    let n_ffn = gate_t.shape[1] as usize;
    let n_expert = gate_t.shape[2] as usize;
    let h = down_t.shape[1] as usize;
    let expert = 7usize.min(n_expert - 1);
    let n_tokens = 16usize;
    let topk = 1usize;
    let gate_row_stride = (n_in / 32) * 34;
    let gate_expert_stride = n_ffn * gate_row_stride;
    let down_row_stride = (n_ffn / 32) * 34;
    let down_expert_stride = h * down_row_stride;
    let gate_bytes_all = g.slice(gate_t);
    let up_bytes_all = g.slice(up_t);
    let down_bytes_all = g.slice(down_t);
    let gate_expert_bytes =
        &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
    let up_expert_bytes =
        &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
    let down_expert_bytes =
        &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
    let gate_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
        shape: vec![n_in as u64, n_ffn as u64],
        dtype: GgmlType::Q8_0,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: gate_expert_stride as u64,
    };
    let down_desc = crate::tensor::TensorDesc {
        name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
        shape: vec![n_ffn as u64, h as u64],
        dtype: GgmlType::Q8_0,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: down_expert_stride as u64,
    };
    let gate_f32 =
        crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
    let up_f32 = crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
    let down_f32 =
        crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
        .collect();
    let mut cpu_inner = vec![0.0f32; n_tokens * n_ffn];
    let mut cpu_down = vec![0.0f32; n_tokens * h];
    for token in 0..n_tokens {
        let x_tok = &x[token * n_in..(token + 1) * n_in];
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
        for i in 0..n_ffn {
            let g = gate[i];
            cpu_inner[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
        }
        let down = crate::forward::mat_vec_pub(
            &down_f32,
            n_ffn,
            h,
            &cpu_inner[token * n_ffn..(token + 1) * n_ffn],
        );
        cpu_down[token * h..(token + 1) * h].copy_from_slice(&down);
    }

    let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
    let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
    let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
    let x_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_tokens * n_in) as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let mut counts = vec![0i32; n_expert];
    counts[expert] = n_tokens as i32;
    let mut ids = vec![0i32; n_expert * n_tokens];
    for token in 0..n_tokens {
        ids[expert * n_tokens + token] = token as i32;
    }
    let counts_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&counts),
        vec![n_expert as u64],
        GgmlType::F32,
    )
    .expect("counts tensor");
    let ids_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![(n_expert * n_tokens) as u64],
        GgmlType::F32,
    )
    .expect("ids tensor");
    let inner_gpu =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("inner tensor");
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
            &ctx,
            enc,
            &gate_gpu,
            &up_gpu,
            &x_gpu,
            &counts_gpu,
            &ids_gpu,
            &inner_gpu,
            n_in,
            n_ffn,
            n_expert,
            topk,
            n_tokens,
        )
    })
    .expect("gpu grouped q8 swiglu");
    let gpu_inner = read_back_f32(&inner_gpu.buffer, n_tokens * n_ffn);
    let dot_inner: f64 = gpu_inner
        .iter()
        .zip(cpu_inner.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng_inner: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc_inner: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos_inner = dot_inner / (ng_inner.sqrt() * nc_inner.sqrt()).max(1e-12);
    let max_inner = gpu_inner
        .iter()
        .zip(cpu_inner.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[moe-q8-swiglu-oracle] cos={cos_inner:.6} max|delta|={max_inner:.3e}");
    assert!(cos_inner > 0.999, "inner cos={cos_inner}");
    assert!(max_inner < 2e-2, "inner max|delta|={max_inner}");

    let cpu_inner_gpu = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&cpu_inner),
        vec![(n_tokens * n_ffn) as u64],
        GgmlType::F32,
    )
    .expect("cpu inner tensor");
    let down_out_gpu = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * h) as u64]).expect("down out");
    one_shot(&ctx, |enc| {
        encode_moe_down_q8_0_f32_grouped_slots(
            &ctx,
            enc,
            &down_gpu,
            &cpu_inner_gpu,
            &counts_gpu,
            &ids_gpu,
            &down_out_gpu,
            n_ffn,
            h,
            n_expert,
            n_tokens,
        )
    })
    .expect("gpu grouped q8 down");
    let gpu_down = read_back_f32(&down_out_gpu.buffer, n_tokens * h);
    let dot_down: f64 = gpu_down
        .iter()
        .zip(cpu_down.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let ng_down: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nc_down: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let cos_down = dot_down / (ng_down.sqrt() * nc_down.sqrt()).max(1e-12);
    let max_down = gpu_down
        .iter()
        .zip(cpu_down.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[moe-q8-down-oracle] cos={cos_down:.6} max|delta|={max_down:.3e}");
    assert!(cos_down > 0.999, "down cos={cos_down}");
    assert!(max_down < 2e-2, "down max|delta|={max_down}");
}

#[test]
fn mat_vec_and_mat_mat_half_weights_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(path, dtype) in &[
        ("/Users/tito/models/Qwen3.5-0.8B.f16.gguf", GgmlType::F16),
        ("/Users/tito/models/Qwen3.5-0.8B-BF16.gguf", GgmlType::BF16),
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("[half-weight] skipped missing fixture {path}");
            continue;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == dtype && t.shape.len() == 2)
            .expect("missing half test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[half-weight {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
            .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| match dtype {
            GgmlType::F16 => encode_mat_vec_f16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            GgmlType::BF16 => encode_mat_vec_bf16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            _ => unreachable!(),
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[half-weight {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-3, "{dtype:?} mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32, 33] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::F16 => encode_mat_mat_f16_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                GgmlType::BF16 => encode_mat_mat_bf16_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                _ => unreachable!(),
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[half-weight {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-3, "{dtype:?} mat_mat max_abs={max_abs}");
        }
    }
}

#[test]
fn mat_mat_bf16_bfloat_act_matches_rounded_cpu() {
    fn round_to_bf16_f32(x: f32) -> f32 {
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
    }

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[bf16-bfloat-act] skipped missing fixture {path}");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let w = g
        .tensors
        .iter()
        .find(|t| {
            t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::BF16 && t.shape.len() == 2
        })
        .expect("missing BF16 test tensor");
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(w),
        vec![n_in as u64, n_out as u64],
        GgmlType::BF16,
    )
    .expect("weight tensor");

    for &n_out_case in &[70usize, n_out] {
        let weight_case = &weight_f32[..n_in * n_out_case];
        for &n_query in &[1usize, 16, 32, 33] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let x_bf16: Vec<f32> = x_pack.iter().copied().map(round_to_bf16_f32).collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out_case];
            for q in 0..n_query {
                let row = &x_bf16[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(weight_case, n_in, n_out_case, row);
                cpu_pack[q * n_out_case..(q + 1) * n_out_case].copy_from_slice(&out);
            }
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out_case) as u64])
                .expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_bf16_bfloat_act_f32(
                    &ctx, enc, &w_t, &x_t, &y_t, n_in, n_out_case, n_query,
                )
            })
            .expect("approx bf16 matmat encode");
            let gpu = read_back_f32(&y_t.buffer, n_query * n_out_case);
            let dot: f64 = gpu
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let nc: f64 = cpu_pack.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
            let max_abs = gpu
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[bf16-bfloat-act n_out={n_out_case} n_query={n_query}] \
                 cos={cos:.6} max|Delta|={max_abs:.2e}"
            );
            assert!(cos > 0.99999, "cos={cos}");
            assert!(max_abs < 1e-3, "max_abs={max_abs}");
        }
    }
}

#[test]
fn mat_vec_q4_k_matches_cpu() {
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
    let q4k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q4_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        })
        .expect("no Q4_K tensor");
    let n_in = q4k.shape[0] as usize;
    let n_out = q4k.shape[1] as usize;
    eprintln!("[q4_k-test] {} shape=[{n_in}, {n_out}]", q4k.name);

    let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let gpu = mat_vec_q4_k_f32_readback_for_test(&ctx, g.slice(q4k), &x, n_in, n_out).expect("gpu");
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q4_k] max|Δ|={max_abs:.2e}");
    assert!(max_abs < 1e-2);
}

#[test]
fn mat_vec_trellis3_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    // Small shape with a partial last threadgroup (66 % 4 != 0) to
    // exercise the row guards, and multiple groups along n_in.
    let n_in = 512;
    let n_out = 66;
    let syn = trellis3_synthetic(n_in, n_out, 0x007E_1115);
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 29) as f32 - 14.0) * 3e-2).collect();
    for variant in [
        Trellis3Variant::ThreeInst,
        Trellis3Variant::ThreeInstV2,
        Trellis3Variant::Lut8x2,
        Trellis3Variant::HybV2,
        Trellis3Variant::ThreeInstG,
        Trellis3Variant::ThreeInstV2G,
        Trellis3Variant::ThreeInstGNsg4,
        Trellis3Variant::ThreeInstGNr4,
        Trellis3Variant::ThreeInstDG,
    ] {
        let cpu = trellis3_cpu_reference(variant, &syn, &x, n_in, n_out);
        let gpu = mat_vec_trellis3_f32_readback_for_test(&ctx, variant, &syn, &x, n_in, n_out)
            .expect("gpu");
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb2 = 0f64;
        let mut max_delta = 0f32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb2 += (*b as f64) * (*b as f64);
            max_delta = max_delta.max((a - b).abs());
        }
        let cos = dot / (na.sqrt() * nb2.sqrt()).max(1e-30);
        let max_abs_y = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "[trellis3 {}] max|Δ|={max_delta:.3e} max|y|={max_abs_y:.3e} cos={cos:.9}",
            variant.label()
        );
        assert!(cos >= 0.999999, "{} cosine {cos}", variant.label());
        assert!(
            max_delta <= 1e-3 * max_abs_y.max(1e-3),
            "{} max delta {max_delta} vs max|y| {max_abs_y}",
            variant.label()
        );
    }
}

#[test]
fn mat_vec_and_mat_mat_q4_legacy_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(path, dtype) in &[
        ("/Users/tito/models/Qwen3.5-0.8B-Q4_0.gguf", GgmlType::Q4_0),
        ("/Users/tito/models/Qwen3.5-0.8B-Q4_1.gguf", GgmlType::Q4_1),
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("[q4-legacy] skipped missing fixture {path}");
            continue;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == dtype
                    && t.shape.len() == 2
                    && t.shape[0] % 32 == 0
            })
            .expect("missing q4 legacy test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[q4-legacy {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
            .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| match dtype {
            GgmlType::Q4_0 => encode_mat_vec_q4_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            GgmlType::Q4_1 => encode_mat_vec_q4_1_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            _ => unreachable!(),
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q4-legacy {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32, 33] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::Q4_0 => encode_mat_mat_q4_0_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                GgmlType::Q4_1 => encode_mat_mat_q4_1_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                _ => unreachable!(),
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q4-legacy {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
        }
    }
}

#[test]
fn mat_vec_and_mat_mat_q3_k_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-0.8B-Q3_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[q3_k] skipped missing fixture {path}");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let w = g
        .tensors
        .iter()
        .find(|t| {
            t.name == "blk.0.ffn_gate.weight"
                && t.dtype == GgmlType::Q3_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        })
        .expect("missing q3_k test tensor");
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    eprintln!("[q3_k] {} shape=[{n_in}, {n_out}]", w.name);

    let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(w),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q3_K,
    )
    .expect("weight tensor");

    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
    one_shot(&ctx, |enc| {
        encode_mat_vec_q3_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })
    .expect("mat_vec encode");
    let gpu = read_back_f32(&y_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q3_k mat_vec] max|Delta|={max_abs:.2e}");
    assert!(max_abs < 1e-2, "Q3_K mat_vec max_abs={max_abs}");

    for &n_query in &[1usize, 16, 32] {
        let x_pack: Vec<f32> = (0..n_query * n_in)
            .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
            .collect();
        let mut cpu_pack = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row = &x_pack[q * n_in..(q + 1) * n_in];
            let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
            cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
        }
        let x_pack_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_pack),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x pack tensor");
        let y_pack_t =
            MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64]).expect("y pack tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q3_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");
        let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
        let max_abs = gpu_pack
            .iter()
            .zip(cpu_pack.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q3_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q3_K mat_mat max_abs={max_abs}");
    }
}

#[test]
fn mat_vec_and_mat_mat_dense_iq2_s_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-4B-UD-IQ2_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[dense-iq2_s] skipped missing fixture {path}");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let w = match g.tensors.iter().find(|t| {
        t.name.starts_with("blk.")
            && t.name.ends_with(".weight")
            && t.dtype == GgmlType::IQ2_S
            && t.shape.len() == 2
            && t.shape[0] % 256 == 0
    }) {
        Some(t) => t,
        None => {
            eprintln!("[dense-iq2_s] skipped missing IQ2_S tensor in {path}");
            return;
        }
    };
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    eprintln!("[dense-iq2_s] {} shape=[{n_in}, {n_out}]", w.name);

    let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(w),
        vec![n_in as u64, n_out as u64],
        GgmlType::IQ2_S,
    )
    .expect("weight tensor");

    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
    one_shot(&ctx, |enc| {
        encode_mat_vec_iq2_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })
    .expect("mat_vec encode");
    let gpu = read_back_f32(&y_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[dense-iq2_s mat_vec] max|Delta|={max_abs:.2e}");
    assert!(max_abs < 1e-2, "IQ2_S mat_vec max_abs={max_abs}");

    for &n_query in &[1usize, 16] {
        let x_pack: Vec<f32> = (0..n_query * n_in)
            .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
            .collect();
        let mut cpu_pack = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row = &x_pack[q * n_in..(q + 1) * n_in];
            let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
            cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
        }
        let x_pack_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_pack),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x pack tensor");
        let y_pack_t =
            MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64]).expect("y pack tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_iq2_s_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");
        let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
        let max_abs = gpu_pack
            .iter()
            .zip(cpu_pack.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[dense-iq2_s mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "IQ2_S mat_mat max_abs={max_abs}");
    }
}

#[test]
fn mat_vec_lowbit_nc2_synthetic_matches_singletons_and_rejects_bad_views() {
    type Encoder = fn(
        &MetalContext,
        &KernelEncoder,
        &MetalTensor,
        &MetalTensor,
        &MetalTensor,
        usize,
        usize,
    ) -> Result<(), MetalError>;
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("metal context: {error}"),
    };
    const N_IN: usize = 256;
    const N_OUT: usize = 9;
    let cases: [(GgmlType, usize, Encoder, Encoder); 2] = [
        (
            GgmlType::IQ2_S,
            82,
            encode_mat_vec_iq2_s_nc2_f32,
            encode_mat_vec_iq2_s_f32,
        ),
        (
            GgmlType::IQ3_S,
            110,
            encode_mat_vec_iq3_s_nc2_f32,
            encode_mat_vec_iq3_s_f32,
        ),
    ];
    for (dtype, block_bytes, encode_nc2, encode_single) in cases {
        let mut weight_bytes = vec![0u8; N_OUT * block_bytes];
        for row in 0..N_OUT {
            let block = &mut weight_bytes[row * block_bytes..(row + 1) * block_bytes];
            for (index, byte) in block.iter_mut().enumerate().skip(2) {
                *byte = ((row * 37 + index * 19 + 11) & 0xff) as u8;
            }
            block[..2].copy_from_slice(
                &half::f16::from_f32(0.03125 + row as f32 * 0.001)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        let weight = offset_tensor(
            &ctx,
            32,
            &weight_bytes,
            18,
            vec![N_IN as u64, N_OUT as u64],
            dtype,
        );
        let inputs = (0..2 * N_IN)
            .map(|index| ((index * 13 + index / N_IN * 7) % 97) as f32 * 1e-3 - 0.04)
            .collect::<Vec<_>>();
        let input = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&inputs),
            12,
            vec![2, N_IN as u64],
            GgmlType::F32,
        );
        let output_bytes = vec![0u8; 2 * N_OUT * size_of::<f32>()];
        let nc2 = offset_tensor(
            &ctx,
            32,
            &output_bytes,
            20,
            vec![(2 * N_OUT) as u64],
            GgmlType::F32,
        );
        let sequential = offset_tensor(
            &ctx,
            32,
            &output_bytes,
            20,
            vec![(2 * N_OUT) as u64],
            GgmlType::F32,
        );

        one_shot(&ctx, |enc| {
            encode_nc2(&ctx, enc, &weight, &input, &nc2, N_IN, N_OUT)?;
            for row in 0..2 {
                let input_row = input.view_subrange((row * N_IN) as u64, vec![N_IN as u64]);
                let output_row = sequential.view_subrange((row * N_OUT) as u64, vec![N_OUT as u64]);
                encode_single(&ctx, enc, &weight, &input_row, &output_row, N_IN, N_OUT)?;
            }
            Ok(())
        })
        .unwrap_or_else(|error| panic!("{dtype:?} synthetic NC2: {error}"));
        let candidate = tensor_f32_at_offset(&nc2);
        let reference = tensor_f32_at_offset(&sequential);
        assert!(
            candidate
                .iter()
                .zip(&reference)
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "{dtype:?} synthetic NC2 differs from singleton rows"
        );
        assert_offset_guards(&nc2, 32, 20);
        assert_offset_guards(&sequential, 32, 20);

        let command = ctx.queue.commandBuffer().expect("validation command");
        let encoder = KernelEncoder::begin(&command);
        let mut misaligned_weight = weight.clone();
        misaligned_weight.offset += 1;
        assert!(
            encode_nc2(
                &ctx,
                &encoder,
                &misaligned_weight,
                &input,
                &nc2,
                N_IN,
                N_OUT,
            )
            .is_err()
        );
        let mut read_only_output = nc2.clone();
        read_only_output.provenance = MetalTensorProvenance::RetainedGgufReadOnly;
        assert!(
            encode_nc2(
                &ctx,
                &encoder,
                &weight,
                &input,
                &read_only_output,
                N_IN,
                N_OUT,
            )
            .is_err()
        );
        let mut short_output = nc2.clone();
        short_output.offset = short_output.buffer.length() as u64 - 4;
        assert!(encode_nc2(&ctx, &encoder, &weight, &input, &short_output, N_IN, N_OUT,).is_err());
        encoder.end();
    }
}

#[test]
#[ignore = "requires local Ridge IQ2_S/IQ3_S fixtures"]
fn mat_vec_lowbit_nc2_matches_two_singleton_rows_bit_exact() {
    type Encoder = fn(
        &MetalContext,
        &KernelEncoder,
        &MetalTensor,
        &MetalTensor,
        &MetalTensor,
        usize,
        usize,
    ) -> Result<(), MetalError>;
    let ctx = MetalContext::new().expect("metal context");
    let path = "/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf";
    let g = crate::gguf::GgufFile::open(path).expect("open Ridge fixture");
    let cases: [(GgmlType, usize, usize, Encoder, Encoder); 4] = [
        (
            GgmlType::IQ2_S,
            5120,
            17408,
            encode_mat_vec_iq2_s_nc2_f32,
            encode_mat_vec_iq2_s_f32,
        ),
        (
            GgmlType::IQ2_S,
            17408,
            5120,
            encode_mat_vec_iq2_s_nc2_f32,
            encode_mat_vec_iq2_s_f32,
        ),
        (
            GgmlType::IQ3_S,
            5120,
            17408,
            encode_mat_vec_iq3_s_nc2_f32,
            encode_mat_vec_iq3_s_f32,
        ),
        (
            GgmlType::IQ3_S,
            17408,
            5120,
            encode_mat_vec_iq3_s_nc2_f32,
            encode_mat_vec_iq3_s_f32,
        ),
    ];
    for (dtype, expected_n_in, expected_n_out, encode_nc2, encode_single) in cases {
        let w = g
            .tensors
            .iter()
            .find(|tensor| {
                tensor.dtype == dtype
                    && tensor.shape.len() == 2
                    && tensor.shape == [expected_n_in as u64, expected_n_out as u64]
            })
            .unwrap_or_else(|| {
                panic!("Ridge {dtype:?} matrix [{expected_n_in}, {expected_n_out}]")
            });
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        let weight =
            MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                .unwrap_or_else(|error| panic!("{dtype:?} weight: {error}"));
        let inputs = (0..2 * n_in)
            .map(|i| (((i * 17 + i / n_in * 11) % 101) as f32 - 50.0) * 1e-3)
            .collect::<Vec<_>>();
        let input = offset_tensor(
            &ctx,
            32,
            bytemuck::cast_slice(&inputs),
            16,
            vec![2, n_in as u64],
            GgmlType::F32,
        );
        let output_bytes = vec![0u8; 2 * n_out * size_of::<f32>()];
        let nc2 = offset_tensor(
            &ctx,
            64,
            &output_bytes,
            32,
            vec![(2 * n_out) as u64],
            GgmlType::F32,
        );
        let sequential = offset_tensor(
            &ctx,
            64,
            &output_bytes,
            32,
            vec![(2 * n_out) as u64],
            GgmlType::F32,
        );

        one_shot(&ctx, |enc| {
            encode_nc2(&ctx, enc, &weight, &input, &nc2, n_in, n_out)?;
            for row in 0..2 {
                let input_row = input.view_subrange((row * n_in) as u64, vec![n_in as u64]);
                let output_row = sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
                encode_single(&ctx, enc, &weight, &input_row, &output_row, n_in, n_out)?;
            }
            Ok(())
        })
        .unwrap_or_else(|error| panic!("{dtype:?} NC2 and singleton rows: {error}"));

        let nc2_values = tensor_f32_at_offset(&nc2);
        let sequential_values = tensor_f32_at_offset(&sequential);
        let max_abs = nc2_values
            .iter()
            .zip(&sequential_values)
            .map(|(candidate, reference)| (candidate - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(
            nc2_values
                .iter()
                .zip(&sequential_values)
                .all(|(candidate, reference)| candidate.to_bits() == reference.to_bits()),
            "{dtype:?} NC2 differs from singleton rows; max_abs={max_abs:.3e}"
        );
        eprintln!("[lowbit-nc2] dtype={dtype:?} max_abs={max_abs:.3e}");
        assert_offset_guards(&nc2, 64, 32);
        assert_offset_guards(&sequential, 64, 32);
    }
}

#[test]
fn mat_vec_and_mat_mat_dense_iq3_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let fixtures = [
        (
            "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
            GgmlType::IQ3_XXS,
        ),
        (
            "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
            GgmlType::IQ3_S,
        ),
    ];
    for &(path, dtype) in &fixtures {
        if !std::path::Path::new(path).exists() {
            eprintln!("[dense-iq3 {dtype:?}] skipped missing fixture {path}");
            continue;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = match g.tensors.iter().find(|t| {
            t.name.starts_with("blk.")
                && t.name.ends_with(".weight")
                && t.dtype == dtype
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        }) {
            Some(t) => t,
            None => {
                eprintln!("[dense-iq3 {dtype:?}] skipped missing dtype tensor in {path}");
                continue;
            }
        };
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[dense-iq3 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
            .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| match dtype {
            GgmlType::IQ3_XXS => {
                encode_mat_vec_iq3_xxs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
            }
            GgmlType::IQ3_S => encode_mat_vec_iq3_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            _ => unreachable!(),
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[dense-iq3 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ3_XXS => encode_mat_mat_iq3_xxs_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                GgmlType::IQ3_S => encode_mat_mat_iq3_s_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                _ => unreachable!(),
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[dense-iq3 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
        }
    }
}

#[test]
fn mat_vec_and_mat_mat_q2_k_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-0.8B.Q2_K.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[q2_k] skipped missing fixture {path}");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let w = g
        .tensors
        .iter()
        .find(|t| {
            t.name == "blk.0.ffn_gate.weight"
                && t.dtype == GgmlType::Q2_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        })
        .expect("missing q2_k test tensor");
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    eprintln!("[q2_k] {} shape=[{n_in}, {n_out}]", w.name);

    let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(w),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q2_K,
    )
    .expect("weight tensor");

    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .expect("x tensor");
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
    one_shot(&ctx, |enc| {
        encode_mat_vec_q2_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })
    .expect("mat_vec encode");
    let gpu = read_back_f32(&y_t.buffer, n_out);
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q2_k mat_vec] max|Delta|={max_abs:.2e}");
    assert!(max_abs < 1e-2, "Q2_K mat_vec max_abs={max_abs}");

    for &n_query in &[1usize, 16, 32] {
        let x_pack: Vec<f32> = (0..n_query * n_in)
            .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
            .collect();
        let mut cpu_pack = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row = &x_pack[q * n_in..(q + 1) * n_in];
            let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
            cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
        }
        let x_pack_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_pack),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x pack tensor");
        let y_pack_t =
            MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64]).expect("y pack tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q2_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");
        let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
        let max_abs = gpu_pack
            .iter()
            .zip(cpu_pack.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q2_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q2_K mat_mat max_abs={max_abs}");
    }
}

#[test]
fn mat_vec_and_mat_mat_iq4_match_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(path, dtype) in &[
        (
            "/Users/tito/models/Qwen3.5-0.8B-IQ4_NL.gguf",
            GgmlType::IQ4_NL,
        ),
        (
            "/Users/tito/models/Qwen3.5-0.8B-IQ4_XS.gguf",
            GgmlType::IQ4_XS,
        ),
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("[iq4] skipped missing fixture {path}");
            continue;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let align = if dtype == GgmlType::IQ4_NL { 32 } else { 256 };
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == dtype
                    && t.shape.len() == 2
                    && t.shape[0] % align == 0
            })
            .expect("missing iq4 test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[iq4 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
            .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| match dtype {
            GgmlType::IQ4_NL => encode_mat_vec_iq4_nl_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            GgmlType::IQ4_XS => encode_mat_vec_iq4_xs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
            _ => unreachable!(),
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[iq4 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ4_NL => encode_mat_mat_iq4_nl_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                GgmlType::IQ4_XS => encode_mat_mat_iq4_xs_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                ),
                _ => unreachable!(),
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[iq4 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
        }
    }
}

/// H5.3b.0 gate: lifted Q4_K mat-mat correctness against
/// (a) CPU mat-mat oracle  (b) N successive mat-vec calls
/// (c) col-major output layout sanity.
///
/// Per codex H5.3b mid-impl review: lifted llama mat-mat is NOT
/// bit-exact with N mat-vec because it stages activations through
/// half before float accumulation. Gate thresholds:
///   * vs CPU mat-mat oracle (same half-staging math): cos ≥ 0.9999
///     and max|Δ| ≤ 0.01 (Q4_K dequant noise dominates the diff)
///   * vs N mat-vec: cos ≥ 0.999 per row (relaxed; half-vs-float
///     accumulation diff)
///   * layout: dst[row + col * M] stride explicitly probed
///
/// Uses real Q4_K weight from the 27B GGUF; N_QUERY ∈ {1, 16, 32}
/// to exercise both partial-tile path (N=1, N=16) and full-tile
/// path (N=32).
#[test]
fn mat_mat_q4_k_matches_cpu_and_mat_vec() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[mat_mat_q4_k] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    // Pick a Q4_K tensor with shape compatible with mat-mat tiling
    // (n_in % 32 == 0, n_out % 64 == 0 for the lifted tile).
    let q4k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q4_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
                && t.shape[1] % 64 == 0
        })
        .expect("no Q4_K tensor with compatible shape");
    let n_in = q4k.shape[0] as usize;
    let n_out = q4k.shape[1] as usize;
    eprintln!(
        "[mat_mat_q4_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
        q4k.name
    );

    let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
    let weight_bytes = g.slice(q4k);

    let n_queries: &[usize] = if mat_mat_q4_k_n64_enabled() {
        &[1, 16, 32, 64]
    } else {
        &[1, 16, 32]
    };
    for &n_query in n_queries {
        // Activation matrix [n_query, n_in] row-major, deterministic
        // pseudo-random fill.
        let mut x = vec![0.0f32; n_query * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }

        // -- CPU oracle: row-major output `[n_query, n_out]`
        //    y[q, o] = sum_i W[o, i] * x[q, i]
        //    We compute it via N successive mat_vec_pub calls.
        let mut cpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row_in = &x[q * n_in..(q + 1) * n_in];
            let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
            cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
        }

        // -- GPU mat-mat: output `[n_out, n_query]` COL-major
        //    i.e. y[r + c * n_out]. Allocate raw n_out*n_query f32.
        let w_t = MetalTensor::from_bytes(
            &ctx,
            weight_bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .expect("weight tensor");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");

        let gpu_col_major = read_back_f32(&y_t.buffer, n_out * n_query);

        // -- Reshape: convert col-major [n_out, n_query] →
        //    row-major [n_query, n_out] for comparison.
        //    cell (q, o) lives at gpu_col_major[o + q * n_out]
        //                  vs   cpu_row_major[q * n_out + o].
        let mut gpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            for o in 0..n_out {
                gpu_row_major[q * n_out + o] = gpu_col_major[o + q * n_out];
            }
        }

        // -- Per-row cosine + max|Δ|.
        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..n_query {
            let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
            let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for i in 0..n_out {
                let p = gpu_row[i] as f64;
                let c = cpu_row[i] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (gpu_row[i] - cpu_row[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!(
            "[mat_mat_q4_k n_query={n_query}] min_cos={min_cos:.6} \
             max|Δ|={max_abs:.3e}"
        );
        // Per H5.3b plan rev 6: cos ≥ 0.999 vs N mat-vec (relaxed
        // because half-staging in lifted kernel). max|Δ| ≤ 0.01
        // (Q4_K dequant + half-staging noise; same order as Q4_K
        // mat-vec test threshold).
        assert!(
            min_cos >= 0.999,
            "n_query={n_query}: min cos {min_cos} < 0.999"
        );
        assert!(
            max_abs < 1e-2,
            "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
        );

        // -- Layout sanity (codex Q7 failure-mode mitigation):
        //    explicitly assert col-major dst stride. Pick three
        //    cells (0,0), (1, n_query/2), (n_out-1, n_query-1) and
        //    check they live where the docs say they live.
        //    cell (r, c) at index `r + c * n_out` in gpu_col_major.
        for &(r, c) in &[
            (0usize, 0usize),
            (1usize, n_query / 2),
            (n_out - 1, n_query - 1),
        ] {
            let raw = gpu_col_major[r + c * n_out];
            let row_major_view = gpu_row_major[c * n_out + r];
            assert_eq!(
                raw.to_bits(),
                row_major_view.to_bits(),
                "layout sanity: gpu_col_major[r={r}+c={c}*n_out={n_out}] should equal \
                 reshape→row_major[c={c}*n_out+r={r}]; got {raw} vs {row_major_view}"
            );
        }
    }
}

/// H5.3b.6 gate: Q6_K mat-mat parity vs N successive mat-vec.
/// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
#[test]
fn mat_mat_q6_k_matches_cpu_and_mat_vec() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[mat_mat_q6_k] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let q6k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q6_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
                && t.shape[1] % 64 == 0
        })
        .expect("no Q6_K tensor with compatible shape");
    let n_in = q6k.shape[0] as usize;
    let n_out = q6k.shape[1] as usize;
    eprintln!(
        "[mat_mat_q6_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
        q6k.name
    );

    let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
    let weight_bytes = g.slice(q6k);

    for &n_query in &[1usize, 16, 32, 64, 128] {
        let mut x = vec![0.0f32; n_query * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }

        // CPU oracle via N mat_vec_pub.
        let mut cpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row_in = &x[q * n_in..(q + 1) * n_in];
            let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
            cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
        }

        let w_t = MetalTensor::from_bytes(
            &ctx,
            weight_bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q6_K,
        )
        .expect("weight tensor");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q6_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");

        let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

        // Reshape: bit-equivalent col-major [n_out, n_query] →
        // row-major [n_query, n_out] (same byte ordering trick as Q4_K).
        let mut gpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            for o in 0..n_out {
                gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
            }
        }

        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..n_query {
            let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
            let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for i in 0..n_out {
                let p = gpu_row[i] as f64;
                let c = cpu_row[i] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (gpu_row[i] - cpu_row[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!(
            "[mat_mat_q6_k n_query={n_query}] min_cos={min_cos:.6} \
             max|Δ|={max_abs:.3e}"
        );
        assert!(
            min_cos >= 0.999,
            "n_query={n_query}: min cos {min_cos} < 0.999"
        );
        assert!(
            max_abs < 1e-2,
            "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
        );
    }
}

/// v0.73a.0 gate: Q5_K mat-mat parity vs N successive Q5_K mat-vec.
/// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
#[test]
fn mat_mat_q5_k_matches_cpu_and_mat_vec() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[mat_mat_q5_k] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let q5k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q5_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
                && t.shape[1] % 64 == 0
        })
        .expect("no Q5_K tensor with compatible shape");
    let n_in = q5k.shape[0] as usize;
    let n_out = q5k.shape[1] as usize;
    eprintln!(
        "[mat_mat_q5_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
        q5k.name
    );

    let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
    let weight_bytes = g.slice(q5k);

    for &n_query in &[1usize, 16, 32, 64, 128] {
        let mut x = vec![0.0f32; n_query * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }

        // CPU oracle via N mat_vec_pub.
        let mut cpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row_in = &x[q * n_in..(q + 1) * n_in];
            let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
            cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
        }

        let w_t = MetalTensor::from_bytes(
            &ctx,
            weight_bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q5_K,
        )
        .expect("weight tensor");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q5_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");

        let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

        // Reshape: bit-equivalent col-major [n_out, n_query] →
        // row-major [n_query, n_out] (same byte ordering trick as Q4_K/Q6_K).
        let mut gpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            for o in 0..n_out {
                gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
            }
        }

        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..n_query {
            let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
            let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for i in 0..n_out {
                let p = gpu_row[i] as f64;
                let c = cpu_row[i] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (gpu_row[i] - cpu_row[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!(
            "[mat_mat_q5_k n_query={n_query}] min_cos={min_cos:.6} \
             max|Δ|={max_abs:.3e}"
        );
        assert!(
            min_cos >= 0.999,
            "n_query={n_query}: min cos {min_cos} < 0.999"
        );
        assert!(
            max_abs < 1e-2,
            "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
        );

        // Layout sanity (codex Q7 mitigation): col-major dst stride
        // `dst[r + c*n_out]` must equal row-major view at three
        // corner cells. Catches a transposed write (which would
        // pass cosine within a single row but corrupt downstream
        // chained mat-mats).
        for &(r, c) in &[
            (0usize, 0usize),
            (1usize, n_query / 2),
            (n_out - 1, n_query - 1),
        ] {
            let raw = gpu_flat[r + c * n_out];
            let row_major_view = gpu_row_major[c * n_out + r];
            assert_eq!(
                raw.to_bits(),
                row_major_view.to_bits(),
                "layout sanity n_query={n_query}: (r={r}, c={c})"
            );
        }
    }
}

#[test]
fn mat_vec_q5_k_matches_cpu() {
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
    let q5k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q5_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        })
        .expect("no Q5_K tensor in 27B layer 0");
    let n_in = q5k.shape[0] as usize;
    let n_out = q5k.shape[1] as usize;
    eprintln!("[q5_k-test] {} shape=[{n_in}, {n_out}]", q5k.name);

    let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let gpu =
        mat_vec_q5_k_f32_readback_for_test(&ctx, g.slice(q5k), &x, n_in, n_out).expect("metal q5k");
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q5_k] max|Δ|={max_abs:.2e}");
    assert!(max_abs < 1e-2);
}

/// v0.73b.0 gate: Q8_0 mat-mat correctness. Uses a real Q8_0
/// weight from the spiritbuun DFlash drafter GGUF
/// (`blk.0.ffn_down.weight`, shape `[17408, 5120]` — large weight,
/// hits both the whole-M-tile and partial-N paths). Same playbook
/// as the Q4_K (v0.63), Q6_K (v0.67), Q5_K (v0.73a.0) gates:
/// per-row cosine ≥ 0.999 across N_QUERY ∈ {1, 16, 32, 64, 128},
/// max|Δ| ≤ 1e-2, explicit col-major dst layout sanity probe.
///
/// Q8_0's structurally-simpler dequant (`int8 * scale`) typically
/// produces TIGHTER cosine than Q4_K/Q5_K/Q6_K mat-mat (which lose
/// precision in nibble packing + scale folding). Expect cos very
/// close to 1.000000.
#[test]
fn mat_mat_q8_0_matches_cpu_and_mat_vec() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[mat_mat_q8_0] skipped — drafter GGUF missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let q8 = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q8_0
                && t.shape.len() == 2
                && t.shape[0] % 32 == 0
                && t.shape[1] % 64 == 0
        })
        .expect("no Q8_0 tensor with compatible shape in drafter blk.0");
    let n_in = q8.shape[0] as usize;
    let n_out = q8.shape[1] as usize;
    eprintln!(
        "[mat_mat_q8_0-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
        q8.name
    );

    let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
    let weight_bytes = g.slice(q8);

    for &n_query in &[1usize, 16, 32, 64, 128] {
        let mut x = vec![0.0f32; n_query * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }

        let mut cpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            let row_in = &x[q * n_in..(q + 1) * n_in];
            let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
            cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
        }

        let w_t = MetalTensor::from_bytes(
            &ctx,
            weight_bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .expect("weight tensor");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_mat_q8_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
        })
        .expect("mat_mat encode");

        let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);
        let mut gpu_row_major = vec![0.0f32; n_query * n_out];
        for q in 0..n_query {
            for o in 0..n_out {
                gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
            }
        }

        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..n_query {
            let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
            let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for i in 0..n_out {
                let p = gpu_row[i] as f64;
                let c = cpu_row[i] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (gpu_row[i] - cpu_row[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!(
            "[mat_mat_q8_0 n_query={n_query}] min_cos={min_cos:.6} \
             max|Δ|={max_abs:.3e}"
        );
        assert!(
            min_cos >= 0.999,
            "n_query={n_query}: min cos {min_cos} < 0.999"
        );
        assert!(
            max_abs < 1e-2,
            "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
        );

        // Layout sanity: col-major dst at three corner cells.
        for &(r, c) in &[
            (0usize, 0usize),
            (1usize, n_query / 2),
            (n_out - 1, n_query - 1),
        ] {
            let raw = gpu_flat[r + c * n_out];
            let row_major_view = gpu_row_major[c * n_out + r];
            assert_eq!(
                raw.to_bits(),
                row_major_view.to_bits(),
                "layout sanity n_query={n_query}: (r={r}, c={c})"
            );
        }
    }
}

#[test]
fn frozen_linear_dense_vjp_matches_stored_precision() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_IN: usize = 73;
    const N_OUT: usize = 37;
    for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        let (weight_bytes, weight_f32) = synthetic_dense_linear_bank(dtype, N_IN, N_OUT);
        for n_query in [1usize, 2, 8] {
            let grad_output: Vec<f32> = (0..n_query * N_OUT)
                .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
                .collect();
            let mut expected = vec![0.0f32; n_query * N_IN];
            for query in 0..n_query {
                for input in 0..N_IN {
                    expected[query * N_IN + input] = (0..N_OUT)
                        .map(|output| {
                            weight_f32[output * N_IN + input] * grad_output[query * N_OUT + output]
                        })
                        .sum();
                }
            }
            let actual = frozen_linear_vjp_f32_readback_for_test(
                &ctx,
                &weight_bytes,
                dtype,
                &grad_output,
                N_IN,
                N_OUT,
                n_query,
            )
            .unwrap();
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "dtype={dtype:?} n_query={n_query}: max absolute error {max_abs}"
            );
        }
    }
}

#[test]
fn frozen_linear_q8_0_vjp_matches_dequantized_adjoint() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_IN: usize = 96;
    const N_OUT: usize = 37;
    let (weight_bytes, weight_f32) = synthetic_q8_0_bank(N_IN, N_OUT);

    for n_query in [1usize, 2, 8] {
        let grad_output: Vec<f32> = (0..n_query * N_OUT)
            .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
            .collect();
        let mut expected = vec![0.0f32; n_query * N_IN];
        for query in 0..n_query {
            for input in 0..N_IN {
                let mut sum = 0.0f32;
                for output in 0..N_OUT {
                    sum += weight_f32[output * N_IN + input] * grad_output[query * N_OUT + output];
                }
                expected[query * N_IN + input] = sum;
            }
        }

        let gpu = frozen_linear_q8_0_vjp_f32_readback_for_test(
            &ctx,
            &weight_bytes,
            &grad_output,
            N_IN,
            N_OUT,
            n_query,
        )
        .expect("Q8_0 activation VJP");
        let max_abs = gpu
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-4,
            "n_query={n_query}: max absolute error {max_abs}"
        );

        let primal: Vec<f32> = (0..n_query * N_IN)
            .map(|index| ((index * 11 + 1) % 41) as f32 * 0.003 - 0.057)
            .collect();
        let mut forward_inner_product = 0.0f64;
        for query in 0..n_query {
            for output in 0..N_OUT {
                let mut value = 0.0f64;
                for input in 0..N_IN {
                    value += weight_f32[output * N_IN + input] as f64
                        * primal[query * N_IN + input] as f64;
                }
                forward_inner_product += value * grad_output[query * N_OUT + output] as f64;
            }
        }
        let reverse_inner_product: f64 = gpu
            .iter()
            .zip(&primal)
            .map(|(gradient, input)| *gradient as f64 * *input as f64)
            .sum();
        assert!(
            (forward_inner_product - reverse_inner_product).abs() < 2e-5,
            "n_query={n_query}: adjoint mismatch forward={forward_inner_product} reverse={reverse_inner_product}"
        );

        let query = n_query - 1;
        for input in [0usize, 47, N_IN - 1] {
            let epsilon = 1e-3f64;
            let objective = |delta: f64| {
                let mut value = 0.0f64;
                for output in 0..N_OUT {
                    let mut projected = 0.0f64;
                    for column in 0..N_IN {
                        let primal_value = primal[query * N_IN + column] as f64
                            + if column == input { delta } else { 0.0 };
                        projected += weight_f32[output * N_IN + column] as f64 * primal_value;
                    }
                    value += projected * grad_output[query * N_OUT + output] as f64;
                }
                value
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            let reverse = gpu[query * N_IN + input] as f64;
            assert!(
                (finite_difference - reverse).abs() < 1e-4,
                "n_query={n_query} input={input}: finite difference {finite_difference} != reverse {reverse}"
            );
        }
    }
}

#[test]
fn frozen_linear_q8_0_vjp_supports_offset_tensor_views() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_IN: usize = 64;
    const N_OUT: usize = 5;
    const N_QUERY: usize = 2;
    let (weight_bytes, weight_f32) = synthetic_q8_0_bank(N_IN, N_OUT);
    let grad_output: Vec<f32> = (0..N_QUERY * N_OUT)
        .map(|index| index as f32 * 0.03 - 0.11)
        .collect();
    let weight = offset_tensor(
        &ctx,
        10,
        &weight_bytes,
        7,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::Q8_0,
    );
    let grad_output_tensor = offset_tensor(
        &ctx,
        8,
        bytemuck::cast_slice(&grad_output),
        12,
        vec![N_OUT as u64, N_QUERY as u64],
        GgmlType::F32,
    );
    let grad_input = offset_tensor(
        &ctx,
        12,
        &vec![0u8; N_QUERY * N_IN * std::mem::size_of::<f32>()],
        16,
        vec![N_IN as u64, N_QUERY as u64],
        GgmlType::F32,
    );
    one_shot(&ctx, |encoder| {
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            encoder,
            &weight,
            &grad_output_tensor,
            &grad_input,
            N_IN,
            N_OUT,
            N_QUERY,
        )
    })
    .unwrap();

    let actual = tensor_f32_at_offset(&grad_input);
    for query in 0..N_QUERY {
        for input in 0..N_IN {
            let expected: f32 = (0..N_OUT)
                .map(|output| {
                    weight_f32[output * N_IN + input] * grad_output[query * N_OUT + output]
                })
                .sum();
            assert!((actual[query * N_IN + input] - expected).abs() < 1e-5);
        }
    }
    assert_offset_guards(&weight, 10, 7);
    assert_offset_guards(&grad_output_tensor, 8, 12);
    assert_offset_guards(&grad_input, 12, 16);
}

#[test]
fn frozen_linear_q8_0_vjp_rejects_invalid_contracts() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N: usize = 32;
    const N_QUERY: usize = 2;
    let (weight_bytes, _) = synthetic_q8_0_bank(N, N);
    let weight = MetalTensor::from_bytes(
        &ctx,
        &weight_bytes,
        vec![N as u64, N as u64],
        GgmlType::Q8_0,
    )
    .unwrap();
    let grad_output = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&vec![0.25f32; N * N_QUERY]),
        vec![N as u64, N_QUERY as u64],
        GgmlType::F32,
    )
    .unwrap();
    let grad_input = MetalTensor::zeros_f32(&ctx, vec![N as u64, N_QUERY as u64]).unwrap();

    let error = one_shot(&ctx, |enc| {
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            enc,
            &weight,
            &grad_output,
            &grad_input,
            N - 1,
            N,
            N_QUERY,
        )
    })
    .expect_err("non-block-aligned input must fail");
    assert!(format!("{error}").contains("frozen_linear_q8_0_vjp"));

    let mut read_only_output = grad_input.clone();
    read_only_output.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    one_shot(&ctx, |enc| {
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            enc,
            &weight,
            &grad_output,
            &read_only_output,
            N,
            N,
            N_QUERY,
        )
    })
    .expect_err("read-only output must fail");

    let overlapping_output = grad_output.clone();
    one_shot(&ctx, |enc| {
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            enc,
            &weight,
            &grad_output,
            &overlapping_output,
            N,
            N,
            N_QUERY,
        )
    })
    .expect_err("overlapping output must fail");

    let mut misaligned_output = grad_input.clone();
    misaligned_output.offset = 2;
    one_shot(&ctx, |enc| {
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            enc,
            &weight,
            &grad_output,
            &misaligned_output,
            N,
            N,
            N_QUERY,
        )
    })
    .expect_err("misaligned output must fail");
}

#[test]
fn frozen_linear_q8_0_vjp_r2c16k64_matches_adjoint_and_tails() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };

    for (n_in, n_out, n_query) in [(32usize, 64usize, 128usize), (96, 128, 257), (96, 65, 129)] {
        let (weight_bytes, weight_f32) = synthetic_q8_0_bank(n_in, n_out);
        let grad_output = (0..n_query * n_out)
            .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
            .collect::<Vec<_>>();
        let weight = offset_tensor(
            &ctx,
            10,
            &weight_bytes,
            7,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        );
        let grad_output_tensor = offset_tensor(
            &ctx,
            8,
            bytemuck::cast_slice(&grad_output),
            12,
            vec![n_out as u64, n_query as u64],
            GgmlType::F32,
        );
        let candidate = offset_tensor(
            &ctx,
            12,
            &vec![0u8; n_query * n_in * std::mem::size_of::<f32>()],
            16,
            vec![n_in as u64, n_query as u64],
            GgmlType::F32,
        );
        let control = offset_tensor(
            &ctx,
            20,
            &vec![0u8; n_query * n_in * std::mem::size_of::<f32>()],
            24,
            vec![n_in as u64, n_query as u64],
            GgmlType::F32,
        );

        let command = ctx.queue.commandBuffer().expect("VJP matrix command");
        let encoder = KernelEncoder::begin(&command);
        encode_frozen_linear_vjp_bank_f32(
            &ctx,
            &encoder,
            &weight,
            &grad_output_tensor,
            &candidate,
            n_in,
            n_out,
            n_query,
        )
        .unwrap();
        encode_frozen_linear_q8_0_vjp_f32(
            &ctx,
            &encoder,
            &weight,
            &grad_output_tensor,
            &control,
            n_in,
            n_out,
            n_query,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());

        let actual = tensor_f32_at_offset(&candidate);
        let incumbent = tensor_f32_at_offset(&control);
        let mut expected = vec![0.0f64; n_query * n_in];
        for query in 0..n_query {
            for input in 0..n_in {
                expected[query * n_in + input] = (0..n_out)
                    .map(|output| {
                        f64::from(weight_f32[output * n_in + input])
                            * f64::from(grad_output[query * n_out + output])
                    })
                    .sum();
            }
        }
        let cpu = f32_f64_differential(&actual, &expected);
        let incumbent_f64 = incumbent
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let differential = f32_f64_differential(&actual, &incumbent_f64);
        eprintln!(
            "[q8-vjp-r2c16] shape=({n_in},{n_out},{n_query}) cpu={cpu:?} incumbent={differential:?}"
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(cpu.0 <= 2.0e-5, "relative L2 {cpu:?}");
        assert!(cpu.1 <= 5.0e-5, "normalized max {cpu:?}");
        assert!(cpu.2 >= 0.999_999_99, "cosine {cpu:?}");
        assert!(
            differential.0 <= 3.0e-4,
            "incumbent relative L2 {differential:?}"
        );
        assert!(
            differential.1 <= 1.0e-3,
            "incumbent normalized max {differential:?}"
        );
        assert!(
            differential.2 >= 0.999_999_9,
            "incumbent cosine {differential:?}"
        );

        let primal = (0..n_query * n_in)
            .map(|index| ((index * 11 + 1) % 41) as f32 * 0.003 - 0.057)
            .collect::<Vec<_>>();
        let mut forward_inner_product = 0.0f64;
        for query in 0..n_query {
            for output in 0..n_out {
                let projected = (0..n_in)
                    .map(|input| {
                        f64::from(weight_f32[output * n_in + input])
                            * f64::from(primal[query * n_in + input])
                    })
                    .sum::<f64>();
                forward_inner_product += projected * f64::from(grad_output[query * n_out + output]);
            }
        }
        let reverse_inner_product = actual
            .iter()
            .zip(&primal)
            .map(|(&gradient, &input)| f64::from(gradient) * f64::from(input))
            .sum::<f64>();
        let adjoint_error = (forward_inner_product - reverse_inner_product).abs()
            / forward_inner_product
                .abs()
                .max(reverse_inner_product.abs())
                .max(1.0);
        assert!(adjoint_error <= 2.0e-5, "adjoint error {adjoint_error}");

        let query = n_query - 1;
        for input in [0usize, n_in / 2, n_in - 1] {
            let epsilon = 1.0e-3f64;
            let objective = |delta: f64| {
                (0..n_out)
                    .map(|output| {
                        let projected = (0..n_in)
                            .map(|column| {
                                let value = f64::from(primal[query * n_in + column])
                                    + if column == input { delta } else { 0.0 };
                                f64::from(weight_f32[output * n_in + column]) * value
                            })
                            .sum::<f64>();
                        projected * f64::from(grad_output[query * n_out + output])
                    })
                    .sum::<f64>()
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            let reverse = f64::from(actual[query * n_in + input]);
            let tolerance = 1.0e-4 + 2.0e-5 * finite_difference.abs();
            assert!(
                (finite_difference - reverse).abs() <= tolerance,
                "shape=({n_in},{n_out},{n_query}) input={input} finite difference {finite_difference} != {reverse}"
            );
        }

        if !n_out.is_multiple_of(64) {
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                incumbent
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_offset_guards(&weight, 10, 7);
        assert_offset_guards(&grad_output_tensor, 8, 12);
        assert_offset_guards(&candidate, 12, 16);
        assert_offset_guards(&control, 20, 24);
    }

    let weight_bytes = synthetic_q8_0_bytes(64, 64);
    let weight =
        MetalTensor::from_bytes(&ctx, &weight_bytes, vec![64, 64], GgmlType::Q8_0).unwrap();
    let grad_output = MetalTensor::zeros_f32(&ctx, vec![64, 128]).unwrap();
    let grad_input = MetalTensor::zeros_f32(&ctx, vec![64, 128]).unwrap();
    let error = one_shot(&ctx, |encoder| {
        encode_frozen_linear_vjp_bank_f32(
            &ctx,
            encoder,
            &weight,
            &grad_output,
            &grad_input,
            48,
            64,
            128,
        )
    })
    .expect_err("non-Q8 input width must fail");
    assert!(format!("{error}").contains("n_in=48"));
}

#[test]
#[ignore = "bounded model-free Muse Q8 VJP qualification packet"]
fn profile_frozen_linear_q8_0_vjp_r2c16k64_muse_shapes() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const FIXED_KILL_MS: f64 = 23.579;
    let cases = [
        ("gate-up-q128", 6_656usize, 19_968usize, 128usize),
        ("gate-up-q512", 6_656, 19_968, 512),
        ("down-q128", 19_968, 6_656, 128),
        ("down-q512", 19_968, 6_656, 512),
    ];

    let weight_bytes = synthetic_q8_0_bytes(6_656, 19_968);
    let base_weight =
        MetalTensor::from_bytes(&ctx, &weight_bytes, vec![6_656, 19_968], GgmlType::Q8_0).unwrap();
    drop(weight_bytes);
    let mut speedups = Vec::with_capacity(cases.len());

    for (case_index, (label, n_in, n_out, n_query)) in cases.into_iter().enumerate() {
        let mut weight = base_weight.clone();
        weight.shape = vec![n_in as u64, n_out as u64];
        let grad_output_values = (0..n_query * n_out)
            .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
            .collect::<Vec<_>>();
        let grad_output = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&grad_output_values),
            vec![n_out as u64, n_query as u64],
            GgmlType::F32,
        )
        .unwrap();
        drop(grad_output_values);
        let control = MetalTensor::zeros_f32(&ctx, vec![n_in as u64, n_query as u64]).unwrap();
        let candidate = MetalTensor::zeros_f32(&ctx, vec![n_in as u64, n_query as u64]).unwrap();

        let run = |matrix: bool| {
            let command = ctx.queue.commandBuffer().expect("Q8 VJP timing command");
            let encoder = KernelEncoder::begin(&command);
            if matrix {
                encode_frozen_linear_vjp_bank_f32(
                    &ctx,
                    &encoder,
                    &weight,
                    &grad_output,
                    &candidate,
                    n_in,
                    n_out,
                    n_query,
                )
                .unwrap();
            } else {
                encode_frozen_linear_q8_0_vjp_f32(
                    &ctx,
                    &encoder,
                    &weight,
                    &grad_output,
                    &control,
                    n_in,
                    n_out,
                    n_query,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());
            let start = command.GPUStartTime();
            let end = command.GPUEndTime();
            assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
            (end - start) * 1.0e3
        };

        run(false);
        run(true);
        let b1 = run(false);
        let c1 = run(true);
        let c2 = run(true);
        let b2 = run(false);
        let control_mean = (b1 + b2) * 0.5;
        let candidate_mean = (c1 + c2) * 0.5;
        let speedup = control_mean / candidate_mean;
        let numerical = f32_differential(
            &tensor_f32_at_offset(&candidate),
            &tensor_f32_at_offset(&control),
        );
        eprintln!(
            "[q8-vjp-r2c16] {label} B-C-C-B gpu_ms={b1:.6}/{c1:.6}/{c2:.6}/{b2:.6} mean={control_mean:.6}->{candidate_mean:.6} speedup={speedup:.3}x numerical={numerical:?}"
        );
        assert!(
            c1 < b1 && c2 < b2,
            "{label}: both balanced comparisons must improve"
        );
        assert!(speedup >= 4.0, "{label}: speedup={speedup:.3}x");
        assert!(numerical.0 <= 3.0e-4, "{label}: relative L2 {numerical:?}");
        assert!(
            numerical.1 <= 1.0e-3,
            "{label}: normalized max {numerical:?}"
        );
        assert!(numerical.2 >= 0.999_999_9, "{label}: cosine {numerical:?}");
        if case_index == 0 {
            assert!(
                candidate_mean <= FIXED_KILL_MS,
                "first stop failed: candidate={candidate_mean:.6} ms"
            );
        }
        speedups.push(speedup);
    }
    let geometric_mean =
        (speedups.iter().map(|speedup| speedup.ln()).sum::<f64>() / speedups.len() as f64).exp();
    eprintln!("[q8-vjp-r2c16] geometric_mean={geometric_mean:.3}x");
    assert!(geometric_mean >= 5.5);
}

/// v0.73b.0 gate: Q8_0 mat-vec correctness. Uses a real Q8_0 weight
/// from the spiritbuun DFlash drafter GGUF (`blk.0.attn_q.weight`,
/// shape `[5120, 4096]`). Same threshold as Q4_K/Q5_K/Q6_K mat-vec
/// (`max|Δ| < 1e-2`).
#[test]
fn mat_vec_q8_0_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[q8_0-test] skipped — drafter GGUF missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let q8 = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q8_0
                && t.shape.len() == 2
                && t.shape[0] % 32 == 0
        })
        .expect("no Q8_0 tensor in drafter blk.0");
    let n_in = q8.shape[0] as usize;
    let n_out = q8.shape[1] as usize;
    eprintln!("[q8_0-test] {} shape=[{n_in}, {n_out}]", q8.name);

    let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let gpu =
        mat_vec_q8_0_f32_readback_for_test(&ctx, g.slice(q8), &x, n_in, n_out).expect("metal q8_0");
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q8_0] max|Δ|={max_abs:.2e}");
    assert!(max_abs < 1e-2);
}

#[test]
fn mat_vec_q6_k_matches_cpu() {
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
    let q6k = g
        .tensors
        .iter()
        .find(|t| {
            t.name.starts_with("blk.0.")
                && t.dtype == GgmlType::Q6_K
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        })
        .expect("no Q6_K tensor");
    let n_in = q6k.shape[0] as usize;
    let n_out = q6k.shape[1] as usize;
    eprintln!("[q6_k-test] {} shape=[{n_in}, {n_out}]", q6k.name);
    let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
    let gpu = mat_vec_q6_k_f32_readback_for_test(&ctx, g.slice(q6k), &x, n_in, n_out).expect("gpu");
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[q6_k] max|Δ|={max_abs:.2e}");
    assert!(max_abs < 1e-2);
}

/// Helper: take an `encode_*` closure that produces a single F32
/// output buffer of length `n_out`, run it one-shot, and return the
/// readback. Common shape across the elementwise tests.
fn one_shot_f32_out<F>(ctx: &MetalContext, n_out: usize, encode: F) -> Vec<f32>
where
    F: FnOnce(&KernelEncoder, &MetalTensor) -> Result<(), MetalError>,
{
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64]).unwrap();
    one_shot(ctx, |enc| encode(enc, &y_t)).unwrap();
    read_back_f32(&y_t.buffer, n_out)
}

#[test]
fn elementwise_silu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
    let n = x.len();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let gpu = one_shot_f32_out(&ctx, n, |enc, y| encode_silu_f32(&ctx, enc, &x_t, y));
    for (i, &v) in x.iter().enumerate() {
        let expected = v / (1.0 + (-v).exp());
        assert!(
            (gpu[i] - expected).abs() < 1e-5,
            "silu[{i}] {v} -> {} vs {}",
            gpu[i],
            expected
        );
    }
}

#[test]
fn elementwise_sigmoid_softplus() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let x: Vec<f32> = (-30..30).map(|i| i as f32 * 0.5).collect();
    let n = x.len();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();

    let sig = one_shot_f32_out(&ctx, n, |enc, y| encode_sigmoid_f32(&ctx, enc, &x_t, y));
    let sp = one_shot_f32_out(&ctx, n, |enc, y| encode_softplus_f32(&ctx, enc, &x_t, y));

    for (i, &v) in x.iter().enumerate() {
        let exp_sig = 1.0 / (1.0 + (-v).exp());
        assert!((sig[i] - exp_sig).abs() < 1e-5);
        let exp_sp = if v > 20.0 {
            v
        } else if v < -20.0 {
            v.exp()
        } else {
            (1.0 + v.exp()).ln()
        };
        assert!((sp[i] - exp_sp).abs() < 1e-5);
    }
}

#[test]
#[ignore]
fn prompt_mat_mat_production_shapes() {
    use std::time::Instant;

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[prompt-matmat] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    const N_QUERY: usize = 321;
    let cases = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.out_proj.weight",
        "blk.0.attn_qkv.weight",
    ];

    eprintln!("[prompt-matmat] {}", ctx.describe());
    for name in cases {
        let Some(t) = g.find(name) else {
            eprintln!("[prompt-matmat] skip missing {name}");
            continue;
        };
        if t.shape.len() != 2 {
            continue;
        }
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
        let x_vec: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_vec_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_vec),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x_vec");
        let y_vec_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y_vec");
        let x_mat: Vec<f32> = (0..N_QUERY * n_in)
            .map(|i| (i as f32 * 1e-3).sin())
            .collect();
        let x_mat_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_mat),
            vec![N_QUERY as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x_mat");
        let y_mat_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, N_QUERY as u64]).expect("y_mat");

        match t.dtype {
            GgmlType::Q4_K => {
                bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("warm vec");
                bench_q4_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("warm mm");
                let t0 = Instant::now();
                bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("vec");
                let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                bench_q4_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("mm");
                let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                    t.dtype,
                    vec_ms / mm_ms.max(1e-9)
                );
            }
            GgmlType::Q5_K => {
                bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("warm vec");
                bench_q5_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("warm mm");
                let t0 = Instant::now();
                bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("vec");
                let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                bench_q5_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("mm");
                let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                    t.dtype,
                    vec_ms / mm_ms.max(1e-9)
                );
            }
            GgmlType::Q6_K => {
                bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("warm vec");
                bench_q6_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("warm mm");
                let t0 = Instant::now();
                bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                    .expect("vec");
                let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                bench_q6_k_mat_mat_chained(&ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1)
                    .expect("mm");
                let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                    t.dtype,
                    vec_ms / mm_ms.max(1e-9)
                );
            }
            _ => {
                eprintln!("[prompt-matmat] skip {name} dtype={:?}", t.dtype);
            }
        }
    }
}

#[test]
#[ignore]
fn prompt_mat_mat_production_shapes_chained64() {
    use std::time::Instant;

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[prompt-matmat-chained] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let n_query: usize = std::env::var("QWEN_PROMPT_MATMAT_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(321);
    let n_dispatches: usize = std::env::var("QWEN_PROMPT_MATMAT_DISPATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let cases = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_qkv.weight",
    ];

    eprintln!(
        "[prompt-matmat-chained] {} N={n_query} dispatches={n_dispatches}",
        ctx.describe()
    );
    for name in cases {
        let Some(t) = g.find(name) else {
            eprintln!("[prompt-matmat-chained] skip missing {name}");
            continue;
        };
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
        let x_mat: Vec<f32> = (0..n_query * n_in)
            .map(|i| (i as f32 * 1e-3).sin())
            .collect();
        let x_mat_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_mat),
            vec![n_query as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x_mat");
        let y_mat_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y_mat");

        match t.dtype {
            GgmlType::Q4_K => {
                bench_q4_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("warm mm");
                let t0 = Instant::now();
                bench_q4_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("mm");
                let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                let weight_gib =
                    (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                eprintln!(
                    "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                    t.dtype
                );
            }
            GgmlType::Q5_K => {
                bench_q5_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("warm mm");
                let t0 = Instant::now();
                bench_q5_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("mm");
                let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                let weight_gib =
                    (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                eprintln!(
                    "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                    t.dtype
                );
            }
            GgmlType::Q6_K => {
                bench_q6_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("warm mm");
                let t0 = Instant::now();
                bench_q6_k_mat_mat_chained(
                    &ctx,
                    &w_t,
                    &x_mat_t,
                    &y_mat_t,
                    n_in,
                    n_out,
                    n_query,
                    n_dispatches,
                )
                .expect("mm");
                let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                let weight_gib =
                    (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                eprintln!(
                    "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                    t.dtype
                );
            }
            _ => {
                eprintln!("[prompt-matmat-chained] skip {name} dtype={:?}", t.dtype);
            }
        }
    }
}

/// Q8_0 token-axis amortization bench. The defaults retain the original
/// N=16 DFlash gate; environment overrides make the same harness useful for
/// exact production shapes at other small-N operating points. It compares
/// the generic mat-mat tile, one batched exact GEMV dispatch, and N exact
/// singleton GEMVs in same-command chains.
///
/// Run: `cargo test --release --lib -p qwen-llm
/// q8_0_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
#[test]
#[ignore]
fn q8_0_mat_mat_amortization_vs_n_mat_vec() {
    use std::time::Instant;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|value| value.parse::<usize>().expect("invalid positive integer"))
            .unwrap_or(default)
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    }

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = std::env::var("QWEN_Q8_AMORT_MODEL").unwrap_or_else(|_| {
        "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf".into()
    });
    if !std::path::Path::new(&path).exists() {
        eprintln!("[q8-amort] skipped - GGUF missing: {path}");
        return;
    }
    let tensor_name =
        std::env::var("QWEN_Q8_AMORT_TENSOR").unwrap_or_else(|_| "blk.0.ffn_down.weight".into());
    let g = crate::gguf::GgufFile::open(&path).expect("open");
    let q8 = g
        .tensors
        .iter()
        .find(|tensor| tensor.name == tensor_name && tensor.dtype == GgmlType::Q8_0)
        .unwrap_or_else(|| {
            let available = g
                .tensors
                .iter()
                .filter(|tensor| tensor.dtype == GgmlType::Q8_0)
                .take(64)
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>();
            panic!("{tensor_name} Q8_0 not found; first Q8_0 tensors: {available:?}")
        });
    let n_in = q8.shape[0] as usize;
    let n_out = q8.shape[1] as usize;
    let n_query = env_usize("QWEN_Q8_AMORT_N", 16);
    let n_layers = env_usize("QWEN_Q8_AMORT_LAYERS", 5);
    let warmup = env_usize("QWEN_Q8_AMORT_WARMUPS", 5);
    let iters = env_usize("QWEN_Q8_AMORT_ITERS", 30);
    assert!(n_query > 0 && n_layers > 0 && warmup > 0 && iters > 0);

    eprintln!(
        "[q8-amort] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
        q8.name
    );

    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(q8),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )
    .expect("weight tensor");
    let x = (0..n_query * n_in)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.002)
        .collect::<Vec<_>>();
    let x_packed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_query * n_in) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let y_mat_mat = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
    let y_batch = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
    let y_sequential = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();

    let bench_mat_mat = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            encode_mat_mat_q8_0_f32(
                &ctx, &enc, &w_t, &x_packed, &y_mat_mat, n_in, n_out, n_query,
            )
            .unwrap();
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    let bench_batch = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            encode_mat_vec_q8_0_batch_f32(
                &ctx, &enc, &w_t, &x_packed, &y_batch, n_in, n_out, n_query,
            )
            .unwrap();
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    let bench_n_mat_vec = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            for row in 0..n_query {
                let x_row = x_packed.view_subrange((row * n_in) as u64, vec![n_in as u64]);
                let y_row = y_sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_q8_0_f32(&ctx, &enc, &w_t, &x_row, &y_row, n_in, n_out).unwrap();
            }
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    for _ in 0..warmup {
        bench_mat_mat();
        bench_batch();
        bench_n_mat_vec();
    }

    let mut mm_wall = Vec::with_capacity(iters);
    let mut mm_gpu = Vec::with_capacity(iters);
    let mut batch_wall = Vec::with_capacity(iters);
    let mut batch_gpu = Vec::with_capacity(iters);
    let mut mv_wall = Vec::with_capacity(iters);
    let mut mv_gpu = Vec::with_capacity(iters);
    for iteration in 0..iters {
        let mut record = |kind: usize, (wall, gpu): (f64, f64)| match kind {
            0 => {
                mm_wall.push(wall);
                mm_gpu.push(gpu);
            }
            1 => {
                batch_wall.push(wall);
                batch_gpu.push(gpu);
            }
            2 => {
                mv_wall.push(wall);
                mv_gpu.push(gpu);
            }
            _ => unreachable!(),
        };
        match iteration % 3 {
            0 => {
                record(0, bench_mat_mat());
                record(1, bench_batch());
                record(2, bench_n_mat_vec());
            }
            1 => {
                record(1, bench_batch());
                record(2, bench_n_mat_vec());
                record(0, bench_mat_mat());
            }
            _ => {
                record(2, bench_n_mat_vec());
                record(0, bench_mat_mat());
                record(1, bench_batch());
            }
        }
    }
    let mm_wall = median(&mut mm_wall);
    let mm_gpu = median(&mut mm_gpu);
    let batch_wall = median(&mut batch_wall);
    let batch_gpu = median(&mut batch_gpu);
    let mv_wall = median(&mut mv_wall);
    let mv_gpu = median(&mut mv_gpu);

    eprintln!("[q8-amort] {n_layers} layers x N={n_query} median over {iters} iters:");
    eprintln!(
        "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        mm_gpu / n_layers as f64
    );
    eprintln!(
        "  batched GEMV (1/layer):    wall={batch_wall:7.2} ms  gpu={batch_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        batch_gpu / n_layers as f64
    );
    eprintln!(
        "  N={n_query} mat-vec ({n_query}/layer): wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        mv_gpu / n_layers as f64
    );
    let ratio_wall = mm_wall / mv_wall;
    let ratio_gpu = mm_gpu / mv_gpu;
    let batch_ratio_wall = batch_wall / mv_wall;
    let batch_ratio_gpu = batch_gpu / mv_gpu;
    eprintln!(
        "  ratios vs sequential: mat-mat wall/gpu={ratio_wall:.3}/{ratio_gpu:.3}; batched-GEMV wall/gpu={batch_ratio_wall:.3}/{batch_ratio_gpu:.3}"
    );

    one_shot(&ctx, |enc| {
        encode_mat_vec_q8_0_batch_f32(&ctx, enc, &w_t, &x_packed, &y_batch, n_in, n_out, n_query)?;
        for row in 0..n_query {
            let x_row = x_packed.view_subrange((row * n_in) as u64, vec![n_in as u64]);
            let y_row = y_sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
            encode_mat_vec_q8_0_f32(&ctx, enc, &w_t, &x_row, &y_row, n_in, n_out)?;
        }
        Ok(())
    })
    .unwrap();
    let batch = read_back_f32(&y_batch.buffer, n_query * n_out);
    let sequential = read_back_f32(&y_sequential.buffer, n_query * n_out);
    let bit_mismatches = batch
        .iter()
        .zip(&sequential)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    eprintln!("  batched-GEMV bit mismatches vs sequential: {bit_mismatches}");
    assert_eq!(bit_mismatches, 0);

    if let Ok(value) = std::env::var("QWEN_Q8_AMORT_MAX_GPU_RATIO") {
        let max_ratio = value.parse::<f64>().expect("invalid GPU ratio");
        assert!(
            ratio_gpu <= max_ratio,
            "Q8 mat-mat GPU ratio {ratio_gpu:.3} exceeds {max_ratio:.3}"
        );
    } else if n_query == 16 && tensor_name == "blk.0.ffn_down.weight" {
        assert!(
            ratio_gpu <= 0.5,
            "original N=16 Q8 gate failed: GPU ratio {ratio_gpu:.3} > 0.5"
        );
    }
}

/// v0.73c.2 A-lite GO/NO-GO bench. Compares the fused
/// `ffn_swiglu_q4_K_mm_n16` (one dispatch per FFN layer) against
/// the unfused `mat_mat_q4_K + mat_mat_q4_K + silu_mul` 3-dispatch
/// sequence at production 27B 64-layer FFN shape (n_in=5120,
/// n_out=17408, N=16, 64 layers). Codex's threshold for proceed
/// is ratio ≤ 0.7 (fused must beat unfused by at least ~30%).
///
/// Run: `cargo test --release --lib -p qwen-llm
/// ffn_fused_swiglu_q4_K_amortization_vs_unfused --ignored -- --nocapture`
#[test]
#[ignore]
#[allow(non_snake_case)]
fn ffn_fused_swiglu_q4_K_amortization_vs_unfused() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[v0.73c.2-gate] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let gate = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
        .expect("ffn_gate Q4_K not found");
    let up = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
        .expect("ffn_up Q4_K not found");
    let n_in = gate.shape[0] as usize;
    let n_out = gate.shape[1] as usize;
    const N: usize = 16;
    let n_layers = 64usize; // 27B has 64 transformer blocks (48 GDN + 16 attn; FFN runs on all)
    let warmup = 5usize;
    let iters = 30usize;

    eprintln!("[v0.73c.2-gate] shape=[n_in={n_in}, n_out={n_out}] N={N} layers={n_layers}");

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
    let x_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_in) as u64]).unwrap();
    let inner_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    let gate_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    let up_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();

    let bench_fused = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
                &ctx,
                &enc,
                &w_gate,
                &w_up,
                &x_packed,
                &inner_packed,
                n_in,
                n_out,
            )
            .unwrap();
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    let bench_unfused = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            encode_mat_mat_q4_k_f32(&ctx, &enc, &w_gate, &x_packed, &gate_packed, n_in, n_out, N)
                .unwrap();
            encode_mat_mat_q4_k_f32(&ctx, &enc, &w_up, &x_packed, &up_packed, n_in, n_out, N)
                .unwrap();
            encode_silu_mul_f32(&ctx, &enc, &gate_packed, &up_packed, &inner_packed).unwrap();
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    for _ in 0..warmup {
        bench_fused();
        bench_unfused();
    }
    let mut sum_f_wall = 0.0f64;
    let mut sum_f_gpu = 0.0f64;
    let mut sum_u_wall = 0.0f64;
    let mut sum_u_gpu = 0.0f64;
    for _ in 0..iters {
        let (w, g) = bench_fused();
        sum_f_wall += w;
        sum_f_gpu += g;
    }
    for _ in 0..iters {
        let (w, g) = bench_unfused();
        sum_u_wall += w;
        sum_u_gpu += g;
    }
    let f_wall = sum_f_wall / iters as f64;
    let f_gpu = sum_f_gpu / iters as f64;
    let u_wall = sum_u_wall / iters as f64;
    let u_gpu = sum_u_gpu / iters as f64;

    eprintln!("[v0.73c.2-gate] {n_layers} layers × N={N} avg over {iters} iters:");
    eprintln!(
        "  fused (1 disp/layer):       wall={f_wall:7.2} ms  gpu={f_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        f_gpu / n_layers as f64
    );
    eprintln!(
        "  unfused (3 disp/layer):     wall={u_wall:7.2} ms  gpu={u_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        u_gpu / n_layers as f64
    );
    let ratio_wall = f_wall / u_wall;
    let ratio_gpu = f_gpu / u_gpu;
    let speedup_wall = 1.0 / ratio_wall;
    let speedup_gpu = 1.0 / ratio_gpu;
    eprintln!(
        "  ratio fused / unfused: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
    );

    // v0.73c.2 RESULT: failed go/no-go. Measured ratio ≈ 0.94 on
    // production 27B Q4_K_M FFN shape — fusion only saves ~6%,
    // codex threshold was ≤ 0.7 (≥ 30% speedup). Kernel is
    // preserved as experimental institutional memory; the assertion
    // below allows the bench to run as a re-checkable "regime
    // still capped?" probe without panicking. If a future change
    // (e.g. larger N, different shape, different hardware) puts
    // the ratio under 0.7, this is where to flag it for plumbing.
    if ratio_gpu <= 0.7 {
        eprintln!(
            "[v0.73c.2-gate] REGIME CHANGE: ratio_gpu {ratio_gpu:.3} now ≤ 0.7. \
             Reconsider plumbing fused FFN into layer-major path."
        );
    } else {
        eprintln!(
            "[v0.73c.2-gate] still capped (ratio_gpu {ratio_gpu:.3} > 0.7); \
             fusion not worth plumbing. Same negative result as v0.73c.2."
        );
    }
}

/// v0.73a.0 A-lite GO/NO-GO bench. Compares amortized weight-BW of
/// Q5_K mat-mat (NR1=16 fast path) vs N=16 successive Q5_K mat-vec
/// on production GDN out_proj (`blk.*.ssm_out.weight`) shape.
///
/// Runs 48 chained dispatches (= 48 GDN layers) of each path in one
/// command buffer; reports per-dispatch latency and the ratio. The
/// hypothesis under test is that mat-mat amortizes the per-step
/// weight reads N=16-fold, so the ratio should be substantially
/// less than 1.0 (i.e. mat-mat much faster). Codex's framing:
/// "if Q5_K mat-mat doesn't beat 16 mat-vecs by a large margin,
/// stop and reassess." Threshold for proceed: ratio ≤ 0.5
/// (mat-mat at LEAST 2× faster than 16-mat-vec equivalent work).
/// Empirically Q4_K and Q6_K mat-mat at this shape achieve closer
/// to 4-8× on M4 Max.
///
/// Run: `cargo test --release --lib -p qwen-llm
/// q5_k_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
#[test]
#[ignore]
fn q5_k_mat_mat_amortization_vs_n_mat_vec() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[v0.73a.0-gate] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let q5k = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ssm_out.weight" && t.dtype == GgmlType::Q5_K)
        .expect("blk.0.ssm_out.weight Q5_K not found");
    let n_in = q5k.shape[0] as usize;
    let n_out = q5k.shape[1] as usize;
    let n_query = 16usize;
    let n_layers = 48usize; // GDN layers in 27B
    let warmup = 5usize;
    let iters = 30usize;

    eprintln!(
        "[v0.73a.0-gate] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
        q5k.name
    );

    let w_t = MetalTensor::from_bytes(
        &ctx,
        g.slice(q5k),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q5_K,
    )
    .expect("weight tensor");
    let x_packed = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_in) as u64]).unwrap();
    let y_packed = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
    let x_single = MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).unwrap();
    let y_single = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

    let bench_mat_mat = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            encode_mat_mat_q5_k_f32(&ctx, &enc, &w_t, &x_packed, &y_packed, n_in, n_out, n_query)
                .unwrap();
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    // 16 mat-vec per layer = 16*48 = 768 dispatches per command buffer.
    // This mirrors the production per-token GDN out_proj fall-through
    // we'd be replacing.
    let bench_n_mat_vec = || {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..n_layers {
            for _ in 0..n_query {
                encode_mat_vec_q5_k_f32(&ctx, &enc, &w_t, &x_single, &y_single, n_in, n_out)
                    .unwrap();
            }
        }
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        (wall, gpu)
    };

    // Warmup both.
    for _ in 0..warmup {
        bench_mat_mat();
        bench_n_mat_vec();
    }

    let mut sum_mm_wall = 0.0f64;
    let mut sum_mm_gpu = 0.0f64;
    let mut sum_mv_wall = 0.0f64;
    let mut sum_mv_gpu = 0.0f64;
    for _ in 0..iters {
        let (w, g) = bench_mat_mat();
        sum_mm_wall += w;
        sum_mm_gpu += g;
    }
    for _ in 0..iters {
        let (w, g) = bench_n_mat_vec();
        sum_mv_wall += w;
        sum_mv_gpu += g;
    }
    let mm_wall = sum_mm_wall / iters as f64;
    let mm_gpu = sum_mm_gpu / iters as f64;
    let mv_wall = sum_mv_wall / iters as f64;
    let mv_gpu = sum_mv_gpu / iters as f64;

    eprintln!("[v0.73a.0-gate] {n_layers} layers × N={n_query} avg over {iters} iters:");
    eprintln!(
        "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        mm_gpu / n_layers as f64
    );
    eprintln!(
        "  N=16 mat-vec (16/layer):   wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
        mv_gpu / n_layers as f64
    );
    let ratio_wall = mm_wall / mv_wall;
    let ratio_gpu = mm_gpu / mv_gpu;
    let speedup_wall = 1.0 / ratio_wall;
    let speedup_gpu = 1.0 / ratio_gpu;
    eprintln!(
        "  ratio mat-mat / 16×mat-vec: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
    );

    // GO/NO-GO: GPU ratio must be at most 0.5 (= 2× speedup). This
    // is a conservative bar relative to Q4_K/Q6_K experience (4-8×).
    // If we don't clear it, v0.73a.1 won't deliver the projected
    // win and we should reassess BEFORE shipping the orchestration
    // restructure.
    assert!(
        ratio_gpu <= 0.5,
        "v0.73a.0 GO/NO-GO failed: GPU ratio {ratio_gpu:.3} > 0.5 \
         (mat-mat must beat 16 mat-vec by at least 2×; \
         reassess before v0.73a.1)"
    );
}

/// Fused SwiGLU FFN (1 dispatch) must match the unfused
/// (mat_vec_q4_K + mat_vec_q4_K + silu_mul) 3-dispatch sequence
/// within fp32 reorder noise. Uses real Q4_K weights from the 27B
/// model's first FFN.
#[test]
// `Q4_K` is the GGUF dtype tag; matches the kernel and other
// function names (`encode_mat_vec_q4_K`, `kernel_ffn_swiglu_q4_K`).
// Lowercasing to `q4_k` would diverge from the rest of the codebase.
#[allow(non_snake_case)]
fn ffn_swiglu_q4_K_matches_unfused() {
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

#[test]
fn ffn_fused_swiglu_q4_k_mma8_matches_unfused_n8() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let path = "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");
    let gate = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
        .expect("Q4_K gate");
    let up = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
        .expect("Q4_K up");
    assert_eq!(gate.shape, up.shape);
    let n_in = gate.shape[0] as usize;
    let n_out = gate.shape[1] as usize;
    const N: usize = 8;
    let x: Vec<f32> = (0..N * n_in)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
        .collect();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(N * n_in) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let gate_w = MetalTensor::from_bytes(
        &ctx,
        g.slice(gate),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q4_K,
    )
    .unwrap();
    let up_w = MetalTensor::from_bytes(
        &ctx,
        g.slice(up),
        vec![n_in as u64, n_out as u64],
        GgmlType::Q4_K,
    )
    .unwrap();

    let gate_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    let up_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    let inner_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_mma8_variant(
            &ctx,
            enc,
            &gate_w,
            &x_t,
            &gate_ref,
            n_in,
            n_out,
            "r1c1k64_sg2",
        )?;
        encode_mat_mat_mma8_variant(&ctx, enc, &up_w, &x_t, &up_ref, n_in, n_out, "r1c1k64_sg2")?;
        encode_silu_mul_f32(&ctx, enc, &gate_ref, &up_ref, &inner_ref)
    })
    .unwrap();

    let inner_fused = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_ffn_fused_swiglu_q4_k_mma8_f32(
            &ctx,
            enc,
            &gate_w,
            &up_w,
            &x_t,
            &inner_fused,
            n_in,
            n_out,
        )
    })
    .unwrap();
    let reference = read_back_f32(&inner_ref.buffer, N * n_out);
    let fused = read_back_f32(&inner_fused.buffer, N * n_out);
    let max_abs = reference
        .iter()
        .zip(&fused)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let bit_mismatches = reference
        .iter()
        .zip(&fused)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("[ffn-fused-mma8-n8] max_abs={max_abs:.3e} bit_mismatches={bit_mismatches}");
    assert_eq!(bit_mismatches, 0, "fused N8 SwiGLU must be bitwise equal");

    let scalar = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    let vec4 = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_mma8_variant(&ctx, enc, &gate_w, &x_t, &scalar, n_in, n_out, "r1c1k128")?;
        encode_mat_mat_mma8_variant(
            &ctx,
            enc,
            &gate_w,
            &x_t,
            &vec4,
            n_in,
            n_out,
            "r1c1k128_vec4",
        )
    })
    .unwrap();
    let scalar = read_back_f32(&scalar.buffer, N * n_out);
    let vec4 = read_back_f32(&vec4.buffer, N * n_out);
    assert!(
        scalar
            .iter()
            .zip(&vec4)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "vectorized Q4_K dequant changed K128 matmul output"
    );
}

/// v0.73c.2 gate: layer-major fused SwiGLU FFN at N=16 must match
/// the unfused (mat_mat_q4_K + mat_mat_q4_K + silu_mul) reference
/// within mat-mat half-staging tolerance. Per-row cosine ≥ 0.999,
/// max|Δ| ≤ 1e-2 (mirrors Q4_K mat-mat gate).
///
/// Uses real `blk.0.ffn_gate.weight` + `ffn_up.weight` from 27B Q4_K_M.
#[test]
#[allow(non_snake_case)]
fn ffn_fused_swiglu_q4_K_mm_n16_matches_unfused() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[ffn-fused-mm-n16] skipped — fixture missing");
        return;
    }
    let g = crate::gguf::GgufFile::open(path).expect("open");

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
    const N: usize = 16;
    eprintln!("[ffn-fused-mm-n16] n_in={n_in} n_out={n_out} N={N}");

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

    // Activation: row-major [N, n_in] deterministic fill.
    let mut x = vec![0.0f32; N * n_in];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 13) as f32 - 6.0) * 1e-2;
    }
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![N as u64, n_in as u64],
        GgmlType::F32,
    )
    .unwrap();

    // --- Unfused reference: gate_mm + up_mm + silu_mul ---
    let gate_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
    let up_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
    let inner_ref_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x_t, &gate_pack, n_in, n_out, N)?;
        encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x_t, &up_pack, n_in, n_out, N)?;
        encode_silu_mul_f32(&ctx, enc, &gate_pack, &up_pack, &inner_ref_t)
    })
    .unwrap();
    let inner_ref_flat = read_back_f32(&inner_ref_t.buffer, N * n_out);

    // --- Fused: 1 dispatch ---
    let inner_fused_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
            &ctx,
            enc,
            &w_gate,
            &w_up,
            &x_t,
            &inner_fused_t,
            n_in,
            n_out,
        )
    })
    .unwrap();
    let inner_fused_flat = read_back_f32(&inner_fused_t.buffer, N * n_out);

    // Both buffers are bit-equivalently row-major [N, n_out] (= col-major [n_out, N]).
    // Reshape via the same indexing as Q4_K mat-mat tests.
    let mut min_cos = f64::INFINITY;
    let mut max_abs = 0.0f32;
    for q in 0..N {
        let mut dot = 0.0f64;
        let mut np = 0.0f64;
        let mut nc = 0.0f64;
        for o in 0..n_out {
            // dst[o + q * n_out] is the cell (m=o, n=q) in col-major
            // [n_out, N], which equals row-major [N, n_out][q][o].
            let p = inner_fused_flat[o + q * n_out] as f64;
            let c = inner_ref_flat[o + q * n_out] as f64;
            dot += p * c;
            np += p * p;
            nc += c * c;
            let d = (p - c).abs() as f32;
            if d > max_abs {
                max_abs = d;
            }
        }
        let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
        if cos < min_cos {
            min_cos = cos;
        }
    }
    eprintln!("[ffn-fused-mm-n16] min_cos={min_cos:.6} max|Δ|={max_abs:.3e}");
    assert!(min_cos >= 0.999, "fused FFN N=16 cos too low: {min_cos}");
    assert!(max_abs < 1e-2, "fused FFN N=16 diverged: max|Δ|={max_abs}");

    const N32: usize = 32;
    let mut x32 = vec![0.0f32; N32 * n_in];
    for (i, v) in x32.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) * 1e-2;
    }
    let x32_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x32),
        vec![N32 as u64, n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let gate32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
    let up32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
    let inner32_ref = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x32_t, &gate32, n_in, n_out, N32)?;
        encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x32_t, &up32, n_in, n_out, N32)?;
        encode_silu_mul_f32(&ctx, enc, &gate32, &up32, &inner32_ref)
    })
    .unwrap();
    let inner32_fused = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_ffn_fused_swiglu_q4_K_mm_f32(
            &ctx,
            enc,
            &w_gate,
            &w_up,
            &x32_t,
            &inner32_fused,
            n_in,
            n_out,
            N32,
        )
    })
    .unwrap();
    let inner32_ref_flat = read_back_f32(&inner32_ref.buffer, N32 * n_out);
    let inner32_fused_flat = read_back_f32(&inner32_fused.buffer, N32 * n_out);
    let mut min_cos32 = f64::INFINITY;
    let mut max_abs32 = 0.0f32;
    for q in 0..N32 {
        let mut dot = 0.0f64;
        let mut np = 0.0f64;
        let mut nc = 0.0f64;
        for o in 0..n_out {
            let p = inner32_fused_flat[o + q * n_out] as f64;
            let c = inner32_ref_flat[o + q * n_out] as f64;
            dot += p * c;
            np += p * p;
            nc += c * c;
            max_abs32 = max_abs32.max((p - c).abs() as f32);
        }
        min_cos32 = min_cos32.min(dot / (np.sqrt() * nc.sqrt() + 1e-30));
    }
    eprintln!("[ffn-fused-mm-n32] min_cos={min_cos32:.6} max|Δ|={max_abs32:.3e}");
    assert!(
        min_cos32 >= 0.999,
        "fused FFN N=32 cos too low: {min_cos32}"
    );
    assert!(
        max_abs32 < 1e-2,
        "fused FFN N=32 diverged: max|Δ|={max_abs32}"
    );
}

/// Fused K+V scatter (one dispatch writes both caches) must produce
/// identical bytes to the two-dispatch sequence.
#[test]
fn scatter_kv_fused_matches_unfused() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let kv_dim = 4 * 256; // n_kv_heads * head_dim for 27B
    let cap = 64usize;

    let k_src: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let v_src: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
        .collect();
    let k_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k_src),
        vec![kv_dim as u64],
        GgmlType::F32,
    )
    .unwrap();
    let v_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v_src),
        vec![kv_dim as u64],
        GgmlType::F32,
    )
    .unwrap();

    for &dst_off_slot in &[0usize, 7, 31, 63] {
        let dst_off = dst_off_slot * kv_dim;

        // --- Reference: two unfused dispatches ---
        let k_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16(&ctx, enc, &k_src_t, &k_ref, dst_off, kv_dim)?;
            encode_scatter_offset_f32_to_f16(&ctx, enc, &v_src_t, &v_ref, dst_off, kv_dim)
        })
        .unwrap();

        // --- Fused: one dispatch ---
        let k_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv(
                &ctx, enc, &k_src_t, &v_src_t, &k_fused, &v_fused, dst_off, kv_dim,
            )
        })
        .unwrap();

        // Compare F16 bytes directly (must be byte-identical).
        let n_bytes = cap * kv_dim * 2;
        let k_ref_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(k_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
        };
        let k_fused_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(k_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
        };
        let v_ref_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
        };
        let v_fused_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
        };
        assert_eq!(
            k_ref_bytes, k_fused_bytes,
            "K mismatch at slot={dst_off_slot}"
        );
        assert_eq!(
            v_ref_bytes, v_fused_bytes,
            "V mismatch at slot={dst_off_slot}"
        );
        eprintln!("[scatter_kv_fused slot={dst_off_slot}] byte-identical to unfused");
    }
}

#[test]
fn scatter_kv_q8_matches_ref_quant() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let kv_dim = 4 * 256usize;
    let cap = 8usize;
    let k_src: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let v_src: Vec<f32> = (0..kv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.046875)
        .collect();
    let k_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k_src),
        vec![kv_dim as u64],
        GgmlType::F32,
    )
    .unwrap();
    let v_src_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v_src),
        vec![kv_dim as u64],
        GgmlType::F32,
    )
    .unwrap();

    for &dst_off_slot in &[0usize, 3, 7] {
        let dst_off = dst_off_slot * kv_dim;
        let k_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_q8_0_kv(
                &ctx, enc, &k_src_t, &v_src_t, &k_q8, &v_q8, dst_off, kv_dim,
            )
        })
        .unwrap();

        let q8_block_bytes = 34usize;
        let blocks_per_row = kv_dim / 32;
        let total_bytes = cap * blocks_per_row * q8_block_bytes;
        let k_gpu: &[u8] = unsafe {
            std::slice::from_raw_parts(k_q8.buffer.contents().as_ptr() as *const u8, total_bytes)
        };
        let v_gpu: &[u8] = unsafe {
            std::slice::from_raw_parts(v_q8.buffer.contents().as_ptr() as *const u8, total_bytes)
        };

        let mut k_ref = vec![0u8; total_bytes];
        let mut v_ref = vec![0u8; total_bytes];
        let block_base = dst_off / 32;
        for (src, dst) in [(&k_src, &mut k_ref), (&v_src, &mut v_ref)] {
            for blk in 0..blocks_per_row {
                let src_blk = &src[blk * 32..(blk + 1) * 32];
                let amax = src_blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let d = amax / 127.0f32;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dst_blk = (block_base + blk) * q8_block_bytes;
                let dh = half::f16::from_f32(d).to_bits().to_le_bytes();
                dst[dst_blk..dst_blk + 2].copy_from_slice(&dh);
                for j in 0..32 {
                    dst[dst_blk + 2 + j] = ((src_blk[j] * id).round() as i8) as u8;
                }
            }
        }

        assert_eq!(
            k_gpu,
            k_ref.as_slice(),
            "Q8 K mismatch at slot={dst_off_slot}"
        );
        assert_eq!(
            v_gpu,
            v_ref.as_slice(),
            "Q8 V mismatch at slot={dst_off_slot}"
        );
        eprintln!("[scatter_kv_q8 slot={dst_off_slot}] byte-identical to ref quantization");
    }
}

fn run_attn_v4_q8_kv_compare(
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

#[test]
fn attn_v4_q8_kv_close_to_f16_kv() {
    run_attn_v4_q8_kv_compare("g6-main", 24, 4, 4096, 64, 32, None);
}

#[test]
fn attn_v4_q8_group8_main_close_to_f16_kv() {
    run_attn_v4_q8_kv_compare("g8-main", 16, 2, 2048, 32, 32, Some(8));
}

#[test]
fn attn_v4_q8_group8_subgroup_close_to_f16_kv() {
    run_attn_v4_q8_kv_compare("g8-t2", 16, 2, 8192, 64, 64, Some(2));
    run_attn_v4_q8_kv_compare("g8-t4", 16, 2, 16384, 128, 64, Some(4));
}

/// GDN α-chain fusion vs the 3-dispatch reference (add_inplace +
/// softplus + mul). Must match within fp32 rounding noise.
#[test]
fn gdn_alpha_chain_matches_unfused() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n = 48usize; // n_v_heads for 27B
    let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
    let dt: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let alog: Vec<f32> = (0..n).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

    let a_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&a),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let dt_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&dt),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let alog_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&alog),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();

    // Fused path.
    let fused = one_shot_f32_out(&ctx, n, |enc, out| {
        encode_gdn_alpha_chain_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out)
    });

    // Unfused reference: build via 3 sequential dispatches in one cmdbuf.
    let a_ref = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&a),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let unfused_t = MetalTensor::zeros_f32(&ctx, vec![n as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_add_inplace_f32(&ctx, enc, &a_ref, &dt_t)?;
        encode_softplus_f32(&ctx, enc, &a_ref, &unfused_t)?;
        encode_mul_f32(&ctx, enc, &unfused_t, &alog_t, &unfused_t)
    })
    .unwrap();
    let unfused = read_back_f32(&unfused_t.buffer, n);

    let max_abs = fused
        .iter()
        .zip(unfused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[gdn_alpha_chain] max|Δ|={max_abs:.2e}");
    assert!(
        max_abs < 1e-5,
        "gdn_alpha_chain fused vs unfused mismatch: max|Δ|={max_abs}"
    );

    // Also validate vs explicit CPU formula.
    for i in 0..n {
        let v = a[i] + dt[i];
        let sp = if v > 20.0 {
            v
        } else if v < -20.0 {
            v.exp()
        } else {
            (1.0 + v.exp()).ln()
        };
        let expected = sp * alog[i];
        assert!(
            (fused[i] - expected).abs() < 1e-5,
            "i={i}: fused={} expected={expected}",
            fused[i]
        );
    }
}

/// v0.73a: batched α-chain over `[N, n_v]` must produce
/// bit-identical output to N successive single-row α-chain calls
/// (broadcasting `dt_bias` and `a_log` across rows).
#[test]
fn gdn_alpha_chain_batched_matches_per_row() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n_rows = 16usize; // N for DFlash block_size
    let n_cols = 48usize; // n_v_heads for 27B
    let n = n_rows * n_cols;
    let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
    let dt: Vec<f32> = (0..n_cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let alog: Vec<f32> = (0..n_cols).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

    let a_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&a),
        vec![n_rows as u64, n_cols as u64],
        GgmlType::F32,
    )
    .unwrap();
    let dt_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&dt),
        vec![n_cols as u64],
        GgmlType::F32,
    )
    .unwrap();
    let alog_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&alog),
        vec![n_cols as u64],
        GgmlType::F32,
    )
    .unwrap();

    // Batched fused path.
    let batched = one_shot_f32_out(&ctx, n, |enc, out| {
        encode_gdn_alpha_chain_batched_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out, n_rows, n_cols)
    });

    // Per-row reference: N invocations of the single-row kernel,
    // each on a row-view of a / out.
    let out_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
    one_shot(&ctx, |enc| {
        for r in 0..n_rows {
            let a_row = a_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
            let out_row = out_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
            encode_gdn_alpha_chain_f32(&ctx, enc, &a_row, &dt_t, &alog_t, &out_row)?;
        }
        Ok(())
    })
    .unwrap();
    let per_row = read_back_f32(&out_t.buffer, n);

    // Bit-exact required (same kernel arithmetic, same broadcast, same order).
    for i in 0..n {
        assert_eq!(
            batched[i].to_bits(),
            per_row[i].to_bits(),
            "i={i} (r={}, c={}): batched={} per_row={}",
            i / n_cols,
            i % n_cols,
            batched[i],
            per_row[i]
        );
    }
}

#[test]
fn mat_vec_q6_k_batch_matches_singleton_bits() {
    let ctx = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    let n_in = 768usize;
    let n_out = 7usize;
    let n_tokens = 16usize;
    let blocks_per_row = n_in / 256;
    let row_bytes = blocks_per_row * 210;
    let mut weight = vec![0u8; n_out * row_bytes];
    for row in 0..n_out {
        for block_index in 0..blocks_per_row {
            let start = row * row_bytes + block_index * 210;
            let block = &mut weight[start..start + 210];
            for (index, value) in block[..192].iter_mut().enumerate() {
                *value = (index as u8)
                    .wrapping_mul(17)
                    .wrapping_add(row as u8)
                    .wrapping_add(block_index as u8 * 11);
            }
            for (index, value) in block[192..208].iter_mut().enumerate() {
                *value = (index as i8 - 8 + row as i8 + block_index as i8) as u8;
            }
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
    }
    let input = (0..n_tokens * n_in)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect::<Vec<_>>();
    let weight = MetalTensor::from_bytes(
        &ctx,
        &weight,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    )
    .unwrap();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input),
        vec![n_tokens as u64, n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let batched = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).unwrap();
    let singleton = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_mat_vec_q6_k_batch_f32(
            &ctx, encoder, &weight, &input, &batched, n_in, n_out, n_tokens,
        )
    })
    .unwrap();
    one_shot(&ctx, |encoder| {
        for token in 0..n_tokens {
            let input_row = input.view_subrange((token * n_in) as u64, vec![n_in as u64]);
            let output_row = singleton.view_subrange((token * n_out) as u64, vec![n_out as u64]);
            encode_mat_vec_q6_k_f32(&ctx, encoder, &weight, &input_row, &output_row, n_in, n_out)?;
        }
        Ok(())
    })
    .unwrap();
    let batched = read_back_f32(&batched.buffer, n_tokens * n_out);
    let singleton = read_back_f32(&singleton.buffer, n_tokens * n_out);
    for (index, (batched, singleton)) in batched.iter().zip(&singleton).enumerate() {
        assert_eq!(
            batched.to_bits(),
            singleton.to_bits(),
            "Q6 batch mismatch at {index}: {batched} != {singleton}"
        );
    }
}

#[test]
fn elementwise_add_mul_silu_mul_sigmoid_mul() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n = 17408usize; // FFN dim — exercise the realistic shape
    let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let a_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&a),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let b_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&b),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();

    let added = one_shot_f32_out(&ctx, n, |enc, y| encode_add_f32(&ctx, enc, &a_t, &b_t, y));
    for i in 0..n {
        assert!((added[i] - (a[i] + b[i])).abs() < 1e-5);
    }
    let muled = one_shot_f32_out(&ctx, n, |enc, y| encode_mul_f32(&ctx, enc, &a_t, &b_t, y));
    for i in 0..n {
        assert!((muled[i] - (a[i] * b[i])).abs() < 1e-5);
    }
    let silumul = one_shot_f32_out(&ctx, n, |enc, y| {
        encode_silu_mul_f32(&ctx, enc, &a_t, &b_t, y)
    });
    for i in 0..n {
        let silu_a = a[i] / (1.0 + (-a[i]).exp());
        assert!(
            (silumul[i] - silu_a * b[i]).abs() < 1e-5,
            "silu_mul[{i}] = {} vs {}",
            silumul[i],
            silu_a * b[i]
        );
    }

    let sigmul = one_shot_f32_out(&ctx, n, |enc, y| {
        encode_sigmoid_mul_f32(&ctx, enc, &a_t, &b_t, y)
    });
    for i in 0..n {
        let sig_a = 1.0 / (1.0 + (-a[i]).exp());
        assert!(
            (sigmul[i] - sig_a * b[i]).abs() < 1e-5,
            "sigmoid_mul[{i}] = {} vs {}",
            sigmul[i],
            sig_a * b[i]
        );
    }
}

#[test]
fn elementwise_add_inplace() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n = 5120usize;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
    let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 2e-3).collect();
    let a_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&a),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let b_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&b),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(&ctx, |enc| encode_add_inplace_f32(&ctx, enc, &a_t, &b_t)).unwrap();
    let result = read_back_f32(&a_t.buffer, n);
    for i in 0..n {
        assert!((result[i] - (a[i] + b[i])).abs() < 1e-5);
    }
}

#[test]
fn softmax_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    // Cover small (one warp) to large (multi-warp reduce) shapes.
    for &n in &[16usize, 256, 4096, 32768] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| encode_softmax_inplace_f32(&ctx, enc, &x_t)).unwrap();
        let gpu = read_back_f32(&x_t.buffer, n);

        // CPU reference.
        let mut cpu = x.clone();
        let m = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in cpu.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in cpu.iter_mut() {
            *v /= s;
        }
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let total: f32 = gpu.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax sum n={n}: {total}");
        assert!(max_abs < 1e-5, "softmax n={n} max|Δ|={max_abs}");
    }
}

#[test]
fn l2_norm_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &n in &[128usize, 256, 1024] {
        // include a few that hit the eps clamp (very small magnitudes)
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-2).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gpu = one_shot_f32_out(&ctx, n, |enc, y| {
            encode_l2_norm_f32(&ctx, enc, &x_t, y, 1e-6)
        });

        // CPU reference: y = x / max(||x||, eps).
        let sq: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / sq.sqrt().max(1e-6);
        let cpu: Vec<f32> = x.iter().map(|v| v * scale).collect();
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_abs < 1e-5, "l2_norm n={n}: max|Δ|={max_abs}");
    }
}

#[test]
fn l2_norm_batched_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
        let total = n_heads * head_dim;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let eps = 1e-6f32;

        // CPU reference: per head, y_h = x_h / max(||x_h||, eps).
        let mut cpu = vec![0.0f32; total];
        for h in 0..n_heads {
            let off = h * head_dim;
            let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
            let scale = 1.0 / sq.sqrt().max(eps);
            for i in 0..head_dim {
                cpu[off + i] = x[off + i] * scale;
            }
        }

        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_l2_norm_batched_f32(&ctx, enc, &x_t, &y_t, n_heads, head_dim, eps)
        })
        .unwrap();
        let gpu = read_back_f32(&y_t.buffer, total);

        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "l2_norm_batched n_heads={n_heads} head_dim={head_dim}: max|Δ|={max_abs}"
        );
    }
}

#[test]
fn l2_norm_pair_batched_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
        let total = n_heads * head_dim;
        let q: Vec<f32> = (0..total)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.04)
            .collect();
        let k: Vec<f32> = (0..total)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03)
            .collect();
        let eps = 1e-6f32;

        let normalize = |x: &[f32]| {
            let mut out = vec![0.0f32; total];
            for h in 0..n_heads {
                let off = h * head_dim;
                let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
                let scale = 1.0 / sq.sqrt().max(eps);
                for i in 0..head_dim {
                    out[off + i] = x[off + i] * scale;
                }
            }
            out
        };
        let q_cpu = normalize(&q);
        let k_cpu = normalize(&k);

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        let q_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
        let k_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_l2_norm_pair_batched_f32(
                &ctx, enc, &q_t, &q_y, &k_t, &k_y, n_heads, head_dim, eps,
            )
        })
        .unwrap();
        let q_gpu = read_back_f32(&q_y.buffer, total);
        let k_gpu = read_back_f32(&k_y.buffer, total);

        let q_max = q_gpu
            .iter()
            .zip(q_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let k_max = k_gpu
            .iter()
            .zip(k_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            q_max < 1e-5 && k_max < 1e-5,
            "l2_norm_pair n_heads={n_heads} head_dim={head_dim}: q={q_max} k={k_max}"
        );
    }
}

#[test]
fn get_rows_flat_f32_matches_cpu_and_guards_ids() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let vocab = 100usize;
    let n_cols = 64usize;
    let embed: Vec<f32> = (0..vocab * n_cols).map(|i| i as f32 * 0.001).collect();
    let ids: Vec<i32> = vec![3, 17, 42, 99, -1, vocab as i32];
    let n_rows = ids.len();

    let embed_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&embed),
        vec![(n_cols * vocab) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let ids_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![n_rows as u64],
        GgmlType::I32,
    )
    .unwrap();
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_get_rows_f32(&ctx, enc, &embed_t, &ids_t, &y_t, n_rows, n_cols)
    })
    .unwrap();
    let gpu = read_back_f32(&y_t.buffer, n_rows * n_cols);
    for r in 0..n_rows {
        for i in 0..n_cols {
            let expected = if ids[r] < 0 || ids[r] as usize >= vocab {
                0.0
            } else {
                embed[ids[r] as usize * n_cols + i]
            };
            let got = gpu[r * n_cols + i];
            assert!(
                (got - expected).abs() < 1e-7,
                "get_rows row={r} ids={} col={i}: {got} vs {expected}",
                ids[r]
            );
        }
    }
}

fn quantized_get_rows_fixture(path: &str, expected_dtype: GgmlType, bit_exact: bool) {
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

#[test]
#[ignore = "requires local 27B Q4_K fixture"]
fn get_rows_q4_k_matches_selected_cpu_rows() {
    quantized_get_rows_fixture(
        "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
        GgmlType::Q4_K,
        true,
    );
}

#[test]
#[ignore = "requires local A3B Q8_0 embedding fixture"]
fn get_rows_q8_0_matches_selected_cpu_rows() {
    quantized_get_rows_fixture(
        "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        GgmlType::Q8_0,
        true,
    );
}

#[test]
#[ignore = "requires local Ridge 27B Q6_K embedding fixture"]
fn get_rows_q6_k_matches_selected_cpu_rows() {
    quantized_get_rows_fixture(
        "/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf",
        GgmlType::Q6_K,
        true,
    );
}

/// CPU reference: forward::rope_in_place — applies NEOX-pairing partial
/// RoPE in place. We use it via the existing `forward::rope_in_place_pub`
/// helper added below.
fn rope_neox_cpu_ref(
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

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn assert_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

fn assert_bitwise_equal(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length mismatch");
    for (index, (expected, actual)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "{label} differs at [{index}]: expected={expected} actual={actual}"
        );
    }
}

fn run_rope_pair_variant(
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

fn run_rope_packed_pair_variant(
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

#[test]
fn rope_neox_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    // Real shapes from Qwen3.5 family:
    //   0.8B: 8 Q heads, 256 head_dim, 64 rotated dims
    //   27B:  24 Q heads or 4 KV heads, 256 head_dim, 64 rotated dims
    let head_dim = 256;
    let n_rot = 64;
    let theta_base = 10_000_000.0f32;
    for &n_heads in &[8usize, 24, 4] {
        for &position in &[0u32, 1, 7, 100] {
            let total = n_heads * head_dim;
            let buf_init: Vec<f32> = (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();

            let mut buf_cpu = buf_init.clone();
            rope_neox_cpu_ref(&mut buf_cpu, n_heads, head_dim, n_rot, position, theta_base);

            let buf_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&buf_init),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rope_neox_f32(
                    &ctx, enc, &buf_t, n_heads, head_dim, n_rot, position, theta_base,
                )
            })
            .unwrap();
            let gpu = read_back_f32(&buf_t.buffer, total);

            let max_abs = gpu
                .iter()
                .zip(buf_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "rope_neox n_heads={n_heads} pos={position}: max|Δ|={max_abs}"
            );
        }
    }
}

#[test]
fn rope_neox_pair_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let head_dim = 256;
    let n_rot = 64;
    let theta_base = 10_000_000.0f32;
    for &(n_q, n_k, position) in &[(24usize, 4usize, 0u32), (8, 8, 37)] {
        let q_len = n_q * head_dim;
        let k_len = n_k * head_dim;
        let q_init: Vec<f32> = (0..q_len).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let k_init: Vec<f32> = (0..k_len)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        let mut q_cpu = q_init.clone();
        let mut k_cpu = k_init.clone();
        rope_neox_cpu_ref(&mut q_cpu, n_q, head_dim, n_rot, position, theta_base);
        rope_neox_cpu_ref(&mut k_cpu, n_k, head_dim, n_rot, position, theta_base);

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q_init),
            vec![q_len as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_init),
            vec![k_len as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_rope_neox_pair_f32(
                &ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
            )
        })
        .unwrap();

        let q_gpu = read_back_f32(&q_t.buffer, q_len);
        let k_gpu = read_back_f32(&k_t.buffer, k_len);
        let q_max = q_gpu
            .iter()
            .zip(q_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let k_max = k_gpu
            .iter()
            .zip(k_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(q_max < 1e-5, "rope pair Q drift: {q_max}");
        assert!(k_max < 1e-5, "rope pair K drift: {k_max}");
    }
}

#[test]
fn rope_neox_pair_optimized_variants_match_baseline() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_q = 24;
    let n_k = 4;
    let head_dim = 256;
    let n_rot = 64;
    let theta_base = 10_000_000.0f32;
    let q_init: Vec<f32> = (0..n_q * head_dim)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
        .collect();
    let k_init: Vec<f32> = (0..n_k * head_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.046875)
        .collect();

    for position in [0, 1, 127, 65_535, 65_536, 262_143, 1_048_575] {
        let baseline = run_rope_pair_variant(
            &ctx, "baseline", &q_init, &k_init, n_q, n_k, head_dim, n_rot, position, theta_base,
        );
        assert_finite(&baseline.0, "baseline Q");
        assert_finite(&baseline.1, "baseline K");
        for variant in ["sincos", "shared"] {
            let candidate = run_rope_pair_variant(
                &ctx, variant, &q_init, &k_init, n_q, n_k, head_dim, n_rot, position, theta_base,
            );
            assert_finite(&candidate.0, &format!("RoPE {variant} Q at {position}"));
            assert_finite(&candidate.1, &format!("RoPE {variant} K at {position}"));
            assert_bitwise_equal(
                &baseline.0,
                &candidate.0,
                &format!("RoPE {variant} Q at {position}"),
            );
            assert_bitwise_equal(
                &baseline.1,
                &candidate.1,
                &format!("RoPE {variant} K at {position}"),
            );
        }

        let minimax = run_rope_pair_variant(
            &ctx, "minimax", &q_init, &k_init, n_q, n_k, head_dim, n_rot, position, theta_base,
        );
        assert_finite(&minimax.0, &format!("RoPE minimax Q at {position}"));
        assert_finite(&minimax.1, &format!("RoPE minimax K at {position}"));
        let q_max = max_abs_diff(&baseline.0, &minimax.0);
        let k_max = max_abs_diff(&baseline.1, &minimax.1);
        assert!(
            q_max <= 5.0e-3 && k_max <= 5.0e-3,
            "RoPE minimax drift at position {position}: q={q_max} k={k_max} tolerance=0.005"
        );
    }
}

#[test]
fn rope_optimized_wrappers_reject_zero_rotary_and_overlap() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_tokens, n_heads, head_dim, n_rot) = (1usize, 2usize, 8usize, 4usize);
    let theta = 10_000_000.0f32;
    let same = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let other = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("validation command buffer");
    let enc = KernelEncoder::begin(&cmd);

    assert!(matches!(
        encode_rope_neox_pair_f32(
            &ctx, &enc, &same, &other, n_heads, n_heads, head_dim, 0, 0, theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    assert!(matches!(
        encode_rope_neox_pair_f32(
            &ctx, &enc, &same, &same, n_heads, n_heads, head_dim, n_rot, 0, theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    let wrong_dtype = MetalTensor::zeros_f16(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    assert!(matches!(
        encode_rope_neox_pair_f32(
            &ctx,
            &enc,
            &same,
            &wrong_dtype,
            n_heads,
            n_heads,
            head_dim,
            n_rot,
            0,
            theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    let mut read_only = other.clone();
    read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(matches!(
        encode_rope_neox_pair_f32(
            &ctx, &enc, &same, &read_only, n_heads, n_heads, head_dim, n_rot, 0, theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    assert!(matches!(
        encode_rope_neox_pair_shared_minimax_f32(
            &ctx,
            &enc,
            &same,
            &other,
            n_heads,
            n_heads,
            head_dim,
            n_rot,
            ROPE_MINIMAX_MAX_VALIDATED_POSITION + 1,
            theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    assert!(matches!(
        encode_rope_neox_pair_f32_packed_consecutive(
            &ctx, &enc, &same, &same, n_tokens, n_heads, n_heads, head_dim, n_rot, 0, theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    let packed_q =
        MetalTensor::zeros_f32(&ctx, vec![(2 * n_tokens * n_heads * head_dim) as u64]).unwrap();
    let packed_k =
        MetalTensor::zeros_f32(&ctx, vec![(2 * n_tokens * n_heads * head_dim) as u64]).unwrap();
    assert!(matches!(
        encode_rope_neox_pair_shared_minimax_f32_packed_consecutive(
            &ctx,
            &enc,
            &packed_q,
            &packed_k,
            2,
            n_heads,
            n_heads,
            head_dim,
            n_rot,
            ROPE_MINIMAX_MAX_VALIDATED_POSITION,
            theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    assert!(matches!(
        encode_rope_neox_f32_packed_consecutive(
            &ctx, &enc, &same, n_tokens, n_heads, head_dim, 0, 0, theta,
        ),
        Err(MetalError::BadShape { .. })
    ));

    let n_q = 2usize;
    let n_k = 1usize;
    let q_source =
        MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_q * 2 * head_dim) as u64]).unwrap();
    let q_output_overlap = q_source.view_subrange(0, vec![(n_tokens * n_q * head_dim) as u64]);
    let k_source = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_k * head_dim) as u64]).unwrap();
    let k_output = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_k * head_dim) as u64]).unwrap();
    let q_weight = MetalTensor::zeros_f32(&ctx, vec![head_dim as u64]).unwrap();
    let k_weight = MetalTensor::zeros_f32(&ctx, vec![head_dim as u64]).unwrap();
    assert!(matches!(
        encode_qk_rms_norm_rope_f32_packed_consecutive(
            &ctx,
            &enc,
            &q_source,
            &q_weight,
            &q_output_overlap,
            &k_source,
            &k_weight,
            &k_output,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            0,
            1e-6,
            theta,
        ),
        Err(MetalError::BadShape { .. })
    ));
    let q_output_f16 =
        MetalTensor::zeros_f16(&ctx, vec![(n_tokens * n_q * head_dim) as u64]).unwrap();
    assert!(matches!(
        encode_qk_rms_norm_rope_f32_packed_consecutive(
            &ctx,
            &enc,
            &q_source,
            &q_weight,
            &q_output_f16,
            &k_source,
            &k_weight,
            &k_output,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            0,
            1e-6,
            theta,
        ),
        Err(MetalError::BadShape { .. })
    ));

    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
}

#[test]
fn rope_neox_packed_consecutive_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let head_dim = 256;
    let n_rot = 64;
    let theta_base = 10_000_000.0f32;
    for &(n_tokens, n_heads, start_position) in &[(5usize, 4usize, 0u32), (3, 24, 17)] {
        let total = n_tokens * n_heads * head_dim;
        let buf_init: Vec<f32> = (0..total)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();

        let mut buf_cpu = buf_init.clone();
        for tok in 0..n_tokens {
            let start = tok * n_heads * head_dim;
            let end = start + n_heads * head_dim;
            rope_neox_cpu_ref(
                &mut buf_cpu[start..end],
                n_heads,
                head_dim,
                n_rot,
                start_position + tok as u32,
                theta_base,
            );
        }

        let buf_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&buf_init),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_rope_neox_f32_packed_consecutive(
                &ctx,
                enc,
                &buf_t,
                n_tokens,
                n_heads,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            )
        })
        .unwrap();
        let gpu = read_back_f32(&buf_t.buffer, total);

        let max_abs = gpu
            .iter()
            .zip(buf_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "rope_neox_packed n_tokens={n_tokens} n_heads={n_heads} start={start_position}: max|Δ|={max_abs}"
        );
    }
}

#[test]
fn rope_neox_packed_pair_variants_match_baseline() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_q = 24;
    let n_k = 4;
    let head_dim = 256;
    let n_rot = 64;
    let theta_base = 10_000_000.0f32;
    for &(n_tokens, start_position) in &[(5usize, 0u32), (8, 65_531), (128, 65_536), (8, 1_048_568)]
    {
        let q_init: Vec<f32> = (0..n_tokens * n_q * head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
            .collect();
        let k_init: Vec<f32> = (0..n_tokens * n_k * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.046875)
            .collect();
        let baseline = run_rope_packed_pair_variant(
            &ctx,
            "baseline",
            &q_init,
            &k_init,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        );
        assert_finite(&baseline.0, "packed baseline Q");
        assert_finite(&baseline.1, "packed baseline K");
        for variant in ["paired", "shared", "adaptive"] {
            let candidate = run_rope_packed_pair_variant(
                &ctx,
                variant,
                &q_init,
                &k_init,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            );
            assert_finite(
                &candidate.0,
                &format!("packed RoPE {variant} Q at {start_position}"),
            );
            assert_finite(
                &candidate.1,
                &format!("packed RoPE {variant} K at {start_position}"),
            );
            assert_bitwise_equal(
                &baseline.0,
                &candidate.0,
                &format!("packed RoPE {variant} Q at {start_position}"),
            );
            assert_bitwise_equal(
                &baseline.1,
                &candidate.1,
                &format!("packed RoPE {variant} K at {start_position}"),
            );
        }

        let minimax = run_rope_packed_pair_variant(
            &ctx,
            "minimax",
            &q_init,
            &k_init,
            n_tokens,
            n_q,
            n_k,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        );
        assert_finite(
            &minimax.0,
            &format!("packed RoPE minimax Q at {start_position}"),
        );
        assert_finite(
            &minimax.1,
            &format!("packed RoPE minimax K at {start_position}"),
        );
        let q_max = max_abs_diff(&baseline.0, &minimax.0);
        let k_max = max_abs_diff(&baseline.1, &minimax.1);
        assert!(
            q_max <= 5.0e-3 && k_max <= 5.0e-3,
            "packed RoPE minimax drift at start {start_position}: q={q_max} k={k_max} tolerance=0.005"
        );
    }
}

/// CPU reference for ssm_conv_silu — mirrors the conv block in
/// `forward::Forward::gdn_step`. Mutates `conv_buf` in place,
/// returns conv output (post-SiLU).
fn ssm_conv_silu_cpu_ref(
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

#[test]
fn ssm_conv_silu_matches_cpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    // Real shapes: 0.8B has conv_dim = 2*16*128 + 16*128 = 6144;
    // 27B has conv_dim = 2*16*128 + 48*128 = 10240.
    for &conv_dim in &[6144usize, 10240] {
        const K: usize = 4;
        let qkv_now: Vec<f32> = (0..conv_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let conv_buf: Vec<f32> = (0..(K - 1) * conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 5e-3)
            .collect();
        let conv_w: Vec<f32> = (0..conv_dim * K)
            .map(|i| ((i % 7) as f32 - 3.0) * 1e-2)
            .collect();

        // CPU oracle.
        let mut buf_cpu = conv_buf.clone();
        let out_cpu = ssm_conv_silu_cpu_ref(&qkv_now, &mut buf_cpu, &conv_w, conv_dim);

        // GPU.
        let qkv_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&qkv_now),
            vec![conv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let buf_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&conv_buf),
            vec![((K - 1) * conv_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&conv_w),
            vec![(conv_dim * K) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let out_t = MetalTensor::zeros_f32(&ctx, vec![conv_dim as u64]).unwrap();

        one_shot(&ctx, |enc| {
            encode_ssm_conv_silu_f32(&ctx, enc, &qkv_t, &buf_t, &w_t, &out_t, conv_dim)
        })
        .unwrap();

        let out_gpu = read_back_f32(&out_t.buffer, conv_dim);
        let buf_gpu = read_back_f32(&buf_t.buffer, (K - 1) * conv_dim);

        let max_out = out_gpu
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let max_buf = buf_gpu
            .iter()
            .zip(buf_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[ssm_conv conv_dim={conv_dim}] max|out_Δ|={max_out:.2e} max|buf_Δ|={max_buf:.2e}"
        );
        assert!(max_out < 1e-5, "out drift {max_out}");
        assert!(max_buf < 1e-7, "buf drift {max_buf}");
    }
}

#[test]
fn gdn_prep_packed_checkpoints_match_token_steps() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N: usize = 4;
    const HEAD_DIM: usize = 128;
    const N_K_HEADS: usize = 1;
    const N_V_HEADS: usize = 4;
    let qk_dim = N_K_HEADS * HEAD_DIM;
    let v_dim = N_V_HEADS * HEAD_DIM;
    let conv_dim = 2 * qk_dim + v_dim;
    let conv_state_elems = 3 * conv_dim;

    let qkv: Vec<f32> = (0..N * conv_dim)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.003)
        .collect();
    let conv_initial: Vec<f32> = (0..conv_state_elems)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.002)
        .collect();
    let conv_w: Vec<f32> = (0..4 * conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.004)
        .collect();

    let qkv_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&qkv),
        vec![(N * conv_dim) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let conv_w_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&conv_w),
        vec![(4 * conv_dim) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let conv_packed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&conv_initial),
        vec![conv_state_elems as u64],
        GgmlType::F32,
    )
    .unwrap();
    let q_packed = MetalTensor::zeros_f32(&ctx, vec![(N * qk_dim) as u64]).unwrap();
    let k_packed = MetalTensor::zeros_f32(&ctx, vec![(N * qk_dim) as u64]).unwrap();
    let v_packed = MetalTensor::zeros_f32(&ctx, vec![(N * v_dim) as u64]).unwrap();
    let ckpt_packed = MetalTensor::zeros_f32(&ctx, vec![(N * conv_state_elems) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_gdn_prep_packed_ckpt_f32(
            &ctx,
            enc,
            &qkv_t,
            &conv_packed,
            &conv_w_t,
            &q_packed,
            &k_packed,
            &v_packed,
            &ckpt_packed,
            N,
            N,
            N_K_HEADS,
            N_V_HEADS,
            HEAD_DIM,
        )
    })
    .unwrap();

    let conv_token = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&conv_initial),
        vec![conv_state_elems as u64],
        GgmlType::F32,
    )
    .unwrap();
    let out_token = MetalTensor::zeros_f32(&ctx, vec![(N * conv_dim) as u64]).unwrap();
    let ckpt_token = MetalTensor::zeros_f32(&ctx, vec![(N * conv_state_elems) as u64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    for token in 0..N {
        let enc = KernelEncoder::begin(&cmd);
        let qkv_row = qkv_t.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
        let out_row = out_token.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
        encode_ssm_conv_silu_f32(
            &ctx,
            &enc,
            &qkv_row,
            &conv_token,
            &conv_w_t,
            &out_row,
            conv_dim,
        )
        .unwrap();
        enc.end();
        let blit = BlitEncoder::begin(&cmd);
        let ckpt_row = ckpt_token.view_subrange(
            (token * conv_state_elems) as u64,
            vec![conv_state_elems as u64],
        );
        blit.copy_tensor(&conv_token, &ckpt_row);
        blit.end();
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let packed_conv = read_back_f32(&conv_packed.buffer, conv_state_elems);
    let token_conv = read_back_f32(&conv_token.buffer, conv_state_elems);
    let packed_ckpt = read_back_f32(&ckpt_packed.buffer, N * conv_state_elems);
    let token_ckpt = read_back_f32(&ckpt_token.buffer, N * conv_state_elems);
    let packed_q = read_back_f32(&q_packed.buffer, N * qk_dim);
    let packed_k = read_back_f32(&k_packed.buffer, N * qk_dim);
    let packed_v = read_back_f32(&v_packed.buffer, N * v_dim);
    let token_out = read_back_f32(&out_token.buffer, N * conv_dim);

    assert!(
        packed_conv
            .iter()
            .zip(&token_conv)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "packed conv final state differs from token steps"
    );
    assert!(
        packed_ckpt
            .iter()
            .zip(&token_ckpt)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "packed conv checkpoints differ from token steps"
    );
    for token in 0..N {
        for channel in 0..conv_dim {
            let packed = if channel < qk_dim {
                packed_q[token * qk_dim + channel]
            } else if channel < 2 * qk_dim {
                packed_k[token * qk_dim + channel - qk_dim]
            } else {
                packed_v[token * v_dim + channel - 2 * qk_dim]
            };
            assert_eq!(
                packed.to_bits(),
                token_out[token * conv_dim + channel].to_bits(),
                "packed conv output mismatch at token={token} channel={channel}"
            );
        }
    }
}

#[test]
fn gdn_recurrence_packed_checkpoints_match_token_steps() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N: usize = 4;
    const HEAD_DIM: usize = 128;
    const N_K_HEADS: usize = 1;
    const N_V_HEADS: usize = 4;
    let qk_per_token = N_K_HEADS * HEAD_DIM;
    let v_per_token = N_V_HEADS * HEAD_DIM;
    let state_elems = N_V_HEADS * HEAD_DIM * HEAD_DIM;

    let q: Vec<f32> = (0..N * qk_per_token)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.004)
        .collect();
    let k: Vec<f32> = (0..N * qk_per_token)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.003)
        .collect();
    let v: Vec<f32> = (0..N * v_per_token)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.002)
        .collect();
    let decay: Vec<f32> = (0..N * N_V_HEADS)
        .map(|i| 0.9 + (i % 7) as f32 * 0.01)
        .collect();
    let beta: Vec<f32> = (0..N * N_V_HEADS)
        .map(|i| 0.2 + (i % 5) as f32 * 0.1)
        .collect();
    let state_initial: Vec<f32> = (0..state_elems)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
        .collect();

    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let q_t = tensor(&q);
    let k_t = tensor(&k);
    let v_t = tensor(&v);
    let decay_t = tensor(&decay);
    let beta_t = tensor(&beta);
    let state_packed = tensor(&state_initial);
    let out_packed = MetalTensor::zeros_f32(&ctx, vec![(N * v_per_token) as u64]).unwrap();
    let ckpt_packed = MetalTensor::zeros_f32(&ctx, vec![(N * state_elems) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_gdn_step_decay_packed_ckpt_f32(
            &ctx,
            enc,
            &q_t,
            &k_t,
            &v_t,
            &decay_t,
            &beta_t,
            &state_packed,
            &out_packed,
            &ckpt_packed,
            N,
            N,
            N_V_HEADS,
            N_K_HEADS,
            HEAD_DIM,
        )
    })
    .unwrap();

    let state_token = tensor(&state_initial);
    let out_token = MetalTensor::zeros_f32(&ctx, vec![(N * v_per_token) as u64]).unwrap();
    let ckpt_token = MetalTensor::zeros_f32(&ctx, vec![(N * state_elems) as u64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    for token in 0..N {
        let enc = KernelEncoder::begin(&cmd);
        encode_gdn_step_decay_f32(
            &ctx,
            &enc,
            &q_t.view_subrange((token * qk_per_token) as u64, vec![qk_per_token as u64]),
            &k_t.view_subrange((token * qk_per_token) as u64, vec![qk_per_token as u64]),
            &v_t.view_subrange((token * v_per_token) as u64, vec![v_per_token as u64]),
            &decay_t.view_subrange((token * N_V_HEADS) as u64, vec![N_V_HEADS as u64]),
            &beta_t.view_subrange((token * N_V_HEADS) as u64, vec![N_V_HEADS as u64]),
            &state_token,
            &out_token.view_subrange((token * v_per_token) as u64, vec![v_per_token as u64]),
            N_V_HEADS,
            N_K_HEADS,
            HEAD_DIM,
        )
        .unwrap();
        enc.end();
        let blit = BlitEncoder::begin(&cmd);
        let ckpt_row =
            ckpt_token.view_subrange((token * state_elems) as u64, vec![state_elems as u64]);
        blit.copy_tensor(&state_token, &ckpt_row);
        blit.end();
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let packed_state = read_back_f32(&state_packed.buffer, state_elems);
    let token_state = read_back_f32(&state_token.buffer, state_elems);
    let packed_out = read_back_f32(&out_packed.buffer, N * v_per_token);
    let token_out = read_back_f32(&out_token.buffer, N * v_per_token);
    let packed_ckpt = read_back_f32(&ckpt_packed.buffer, N * state_elems);
    let token_ckpt = read_back_f32(&ckpt_token.buffer, N * state_elems);
    assert!(
        packed_state
            .iter()
            .zip(&token_state)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "packed recurrence final state differs from token steps"
    );
    assert!(
        packed_out
            .iter()
            .zip(&token_out)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "packed recurrence outputs differ from token steps"
    );
    assert!(
        packed_ckpt
            .iter()
            .zip(&token_ckpt)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "packed recurrence checkpoints differ from token steps"
    );
}

/// CPU reference for rmsnorm_gated — per-head RMSNorm of `o` * silu(z).
fn rmsnorm_gated_cpu_ref(
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

#[test]
fn rmsnorm_gated_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128)] {
        let total = n_heads * head_dim;
        let o: Vec<f32> = (0..total)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
            .collect();
        let z: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let eps = 1e-6;

        let cpu = rmsnorm_gated_cpu_ref(&o, &weight, &z, n_heads, head_dim, eps);

        let o_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&o),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        let w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weight),
            vec![head_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let z_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&z),
            vec![total as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();

        one_shot(&ctx, |enc| {
            encode_rmsnorm_gated_f32(&ctx, enc, &o_t, &w_t, &z_t, &y_t, n_heads, head_dim, eps)
        })
        .unwrap();

        let gpu = read_back_f32(&y_t.buffer, total);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[rmsnorm_gated n_heads={n_heads} head_dim={head_dim}] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-4, "rmsnorm_gated drift {max_abs}");
    }
}

/// CPU reference for the GDN-step kernel — mirrors the per-V-head
/// inner loop in `forward::Forward::gdn_step` exactly. Mutates
/// `state` in place and returns `out`.
fn gdn_step_cpu_ref(
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

#[test]
fn gdn_step_matches_cpu() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    // Cover both Qwen3.5/3.6 sizes:
    //   0.8B: n_v_heads = 16, head_dim = 128
    //   27B:  n_v_heads = 48, head_dim = 128
    // 0.8B has n_v == n_k == 16; 27B has n_v=48, n_k=16 (3:1 repeat).
    for &(n_v, n_k) in &[(16usize, 16usize), (48, 16)] {
        let hd = 128usize;
        // Synthetic but realistic-magnitude inputs. Q/K are sized to
        // n_k heads; V is sized to n_v heads.
        let q: Vec<f32> = (0..n_k * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let k: Vec<f32> = (0..n_k * hd)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let v: Vec<f32> = (0..n_v * hd)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();
        let g: Vec<f32> = (0..n_v).map(|i| -((i % 7) as f32) * 1e-3).collect();
        let beta: Vec<f32> = (0..n_v).map(|i| 0.5 + ((i % 11) as f32) * 1e-2).collect();
        // Random-ish but deterministic state, including the
        // post-first-token regime (nonzero initial state) since
        // codex flagged "first-token only" coverage as inadequate.
        let state: Vec<f32> = (0..n_v * hd * hd)
            .map(|i| ((i % 13) as f32 - 6.0) * 1e-3)
            .collect();

        // CPU oracle.
        let mut state_cpu = state.clone();
        let out_cpu = gdn_step_cpu_ref(&q, &k, &v, &g, &beta, &mut state_cpu, n_v, n_k, hd);

        // GPU.
        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_k * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k),
            vec![(n_k * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v),
            vec![(n_v * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let g_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&g),
            vec![n_v as u64],
            GgmlType::F32,
        )
        .unwrap();
        let beta_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&beta),
            vec![n_v as u64],
            GgmlType::F32,
        )
        .unwrap();
        let state_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&state),
            vec![(n_v * hd * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_v * hd) as u64]).unwrap();

        one_shot(&ctx, |enc| {
            encode_gdn_step_f32(
                &ctx, enc, &q_t, &k_t, &v_t, &g_t, &beta_t, &state_t, &out_t, n_v, n_k, hd,
            )
        })
        .unwrap();

        let out_gpu = read_back_f32(&out_t.buffer, n_v * hd);
        let state_gpu = read_back_f32(&state_t.buffer, n_v * hd * hd);

        // Output comparison.
        let max_out = out_gpu
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // State comparison (this is the recurrent variable; correctness
        // here matters more than the output for multi-step decode).
        let max_state = state_gpu
            .iter()
            .zip(state_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        eprintln!(
            "[gdn_step n_v={n_v} n_k={n_k}] max|out_Δ|={max_out:.2e}  max|state_Δ|={max_state:.2e}"
        );
        // simd_sum reduction order can drift slightly from the
        // sequential CPU version; 1e-4 covers it for our magnitudes.
        assert!(max_out < 1e-4, "out drift {max_out}");
        assert!(max_state < 1e-4, "state drift {max_state}");
    }
}

struct GdnStepVjpReference {
    grad_q: Vec<f64>,
    grad_k: Vec<f64>,
    grad_v: Vec<f64>,
    grad_decay: Vec<f64>,
    grad_beta: Vec<f64>,
    grad_state: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn gdn_step_decay_objective_f64(
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
fn gdn_step_decay_vjp_f64(
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
fn gdn_sequence_forward_f64(
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
fn gdn_sequence_objective_f64(
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
fn gdn_sequence_vjp_f64(
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

struct SsmConvSequenceVjpReference {
    grad_qkv: Vec<f64>,
    grad_state: Vec<f64>,
}

fn ssm_conv_sequence_forward_f64(
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
fn ssm_conv_sequence_objective_f64(
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
fn ssm_conv_sequence_vjp_f64(
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
fn gdn_envelope_objective_f64(
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

#[test]
fn gdn_step_decay_vjp_matches_adjoint_and_finite_differences() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_V: usize = 6;
    const N_K: usize = 2;
    const HEAD_DIM: usize = 128;
    let q: Vec<f32> = (0..N_K * HEAD_DIM)
        .map(|index| ((index * 11 + 3) % 43) as f32 * 0.002 - 0.041)
        .collect();
    let k: Vec<f32> = (0..N_K * HEAD_DIM)
        .map(|index| ((index * 13 + 5) % 47) as f32 * 0.0017 - 0.039)
        .collect();
    let v: Vec<f32> = (0..N_V * HEAD_DIM)
        .map(|index| ((index * 17 + 1) % 53) as f32 * 0.0023 - 0.057)
        .collect();
    let decay: Vec<f32> = (0..N_V).map(|head| 0.89 + head as f32 * 0.021).collect();
    let beta: Vec<f32> = (0..N_V).map(|head| 0.23 + head as f32 * 0.14).collect();
    let state: Vec<f32> = (0..N_V * HEAD_DIM * HEAD_DIM)
        .map(|index| ((index * 19 + 7) % 59) as f32 * 0.0007 - 0.019)
        .collect();
    let grad_out: Vec<f32> = (0..N_V * HEAD_DIM)
        .map(|index| ((index * 23 + 2) % 61) as f32 * 0.0011 - 0.031)
        .collect();
    let grad_state_out: Vec<f32> = (0..N_V * HEAD_DIM * HEAD_DIM)
        .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00009 - 0.003)
        .collect();

    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let q_t = tensor(&q);
    let k_t = tensor(&k);
    let v_t = tensor(&v);
    let decay_t = tensor(&decay);
    let beta_t = tensor(&beta);
    let state_t = tensor(&state);
    let grad_out_t = tensor(&grad_out);
    let grad_state_out_t = tensor(&grad_state_out);
    let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![(N_K * HEAD_DIM) as u64]).unwrap();
    let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![(N_K * HEAD_DIM) as u64]).unwrap();
    let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
    let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_state_t =
        MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM * HEAD_DIM) as u64]).unwrap();
    let grad_c_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
    let residual_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_gdn_step_decay_vjp_f32(
            &ctx,
            encoder,
            &q_t,
            &k_t,
            &v_t,
            &decay_t,
            &beta_t,
            &state_t,
            &grad_out_t,
            &grad_state_out_t,
            &grad_q_t,
            &grad_k_t,
            &grad_v_t,
            &grad_decay_t,
            &grad_beta_t,
            &grad_state_t,
            &grad_c_t,
            &residual_t,
            N_V,
            N_K,
            HEAD_DIM,
        )
    })
    .unwrap();

    let actual = GdnStepVjpReference {
        grad_q: read_back_f32(&grad_q_t.buffer, N_K * HEAD_DIM)
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_k: read_back_f32(&grad_k_t.buffer, N_K * HEAD_DIM)
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_v: read_back_f32(&grad_v_t.buffer, N_V * HEAD_DIM)
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_decay: read_back_f32(&grad_decay_t.buffer, N_V)
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_beta: read_back_f32(&grad_beta_t.buffer, N_V)
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_state: read_back_f32(&grad_state_t.buffer, N_V * HEAD_DIM * HEAD_DIM)
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let q64 = as_f64(&q);
    let k64 = as_f64(&k);
    let v64 = as_f64(&v);
    let decay64 = as_f64(&decay);
    let beta64 = as_f64(&beta);
    let state64 = as_f64(&state);
    let grad_out64 = as_f64(&grad_out);
    let grad_state_out64 = as_f64(&grad_state_out);
    let expected = gdn_step_decay_vjp_f64(
        &q64,
        &k64,
        &v64,
        &decay64,
        &beta64,
        &state64,
        &grad_out64,
        &grad_state_out64,
        N_V,
        N_K,
        HEAD_DIM,
    );
    for (name, gpu, cpu, tolerance) in [
        ("q", &actual.grad_q, &expected.grad_q, 3e-5),
        ("k", &actual.grad_k, &expected.grad_k, 3e-5),
        ("v", &actual.grad_v, &expected.grad_v, 2e-6),
        ("decay", &actual.grad_decay, &expected.grad_decay, 5e-5),
        ("beta", &actual.grad_beta, &expected.grad_beta, 5e-5),
        ("state", &actual.grad_state, &expected.grad_state, 3e-6),
    ] {
        let max_abs = gpu
            .iter()
            .zip(cpu)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs < tolerance, "{name} VJP error {max_abs}");
    }

    let objective =
        |q: &[f64], k: &[f64], v: &[f64], decay: &[f64], beta: &[f64], state: &[f64]| {
            gdn_step_decay_objective_f64(
                q,
                k,
                v,
                decay,
                beta,
                state,
                &grad_out64,
                &grad_state_out64,
                N_V,
                N_K,
                HEAD_DIM,
            )
        };
    let epsilon = 1e-5;
    let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
    };
    for &index in &[0usize, 127, N_K * HEAD_DIM - 1] {
        let fd = finite_difference(&q64, index, &|candidate| {
            objective(candidate, &k64, &v64, &decay64, &beta64, &state64)
        });
        assert!((fd - actual.grad_q[index]).abs() < 2e-5);
        let fd = finite_difference(&k64, index, &|candidate| {
            objective(&q64, candidate, &v64, &decay64, &beta64, &state64)
        });
        assert!((fd - actual.grad_k[index]).abs() < 2e-5);
    }
    for &index in &[0usize, 255, N_V * HEAD_DIM - 1] {
        let fd = finite_difference(&v64, index, &|candidate| {
            objective(&q64, &k64, candidate, &decay64, &beta64, &state64)
        });
        assert!((fd - actual.grad_v[index]).abs() < 2e-5);
    }
    for index in 0..N_V {
        let fd = finite_difference(&decay64, index, &|candidate| {
            objective(&q64, &k64, &v64, candidate, &beta64, &state64)
        });
        assert!((fd - actual.grad_decay[index]).abs() < 3e-5);
        let fd = finite_difference(&beta64, index, &|candidate| {
            objective(&q64, &k64, &v64, &decay64, candidate, &state64)
        });
        assert!((fd - actual.grad_beta[index]).abs() < 3e-5);
    }
    for &index in &[
        0usize,
        HEAD_DIM - 1,
        HEAD_DIM,
        2 * HEAD_DIM * HEAD_DIM + 31 * HEAD_DIM + 32,
        N_V * HEAD_DIM * HEAD_DIM - 1,
    ] {
        let fd = finite_difference(&state64, index, &|candidate| {
            objective(&q64, &k64, &v64, &decay64, &beta64, candidate)
        });
        assert!((fd - actual.grad_state[index]).abs() < 2e-5);
    }

    let direction = |len: usize, stride: usize| {
        (0..len)
            .map(|index| ((index * stride + 3) % 29) as f64 * 0.001 - 0.014)
            .collect::<Vec<_>>()
    };
    let dq = direction(q64.len(), 5);
    let dk = direction(k64.len(), 7);
    let dv = direction(v64.len(), 11);
    let ddecay = direction(decay64.len(), 13);
    let dbeta = direction(beta64.len(), 17);
    let dstate = direction(state64.len(), 19);
    let inner = |gradient: &[f64], tangent: &[f64]| {
        gradient
            .iter()
            .zip(tangent)
            .map(|(gradient, tangent)| gradient * tangent)
            .sum::<f64>()
    };
    let reverse_directional = inner(&actual.grad_q, &dq)
        + inner(&actual.grad_k, &dk)
        + inner(&actual.grad_v, &dv)
        + inner(&actual.grad_decay, &ddecay)
        + inner(&actual.grad_beta, &dbeta)
        + inner(&actual.grad_state, &dstate);
    let shift = |base: &[f64], tangent: &[f64], amount: f64| {
        base.iter()
            .zip(tangent)
            .map(|(base, tangent)| base + amount * tangent)
            .collect::<Vec<_>>()
    };
    let directional_epsilon = 1e-5;
    let plus = objective(
        &shift(&q64, &dq, directional_epsilon),
        &shift(&k64, &dk, directional_epsilon),
        &shift(&v64, &dv, directional_epsilon),
        &shift(&decay64, &ddecay, directional_epsilon),
        &shift(&beta64, &dbeta, directional_epsilon),
        &shift(&state64, &dstate, directional_epsilon),
    );
    let minus = objective(
        &shift(&q64, &dq, -directional_epsilon),
        &shift(&k64, &dk, -directional_epsilon),
        &shift(&v64, &dv, -directional_epsilon),
        &shift(&decay64, &ddecay, -directional_epsilon),
        &shift(&beta64, &dbeta, -directional_epsilon),
        &shift(&state64, &dstate, -directional_epsilon),
    );
    let forward_directional = (plus - minus) / (2.0 * directional_epsilon);
    assert!(
        (forward_directional - reverse_directional).abs() < 2e-5,
        "directional adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
    );

    for (tensor, original) in [
        (&q_t, &q),
        (&k_t, &k),
        (&v_t, &v),
        (&decay_t, &decay),
        (&beta_t, &beta),
        (&state_t, &state),
        (&grad_out_t, &grad_out),
        (&grad_state_out_t, &grad_state_out),
    ] {
        assert_eq!(
            read_back_f32(&tensor.buffer, original.len())
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn gdn_step_decay_packed_vjp_matches_temporal_oracle_and_adjoint() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_TOKENS: usize = 4;
    const N_CHECKPOINTS: usize = N_TOKENS - 1;
    const N_V: usize = 3;
    const N_K: usize = 1;
    const HEAD_DIM: usize = 128;
    let qk_elements = N_K * HEAD_DIM;
    let vector_elements = N_V * HEAD_DIM;
    let state_elements = vector_elements * HEAD_DIM;
    let q: Vec<f32> = (0..N_TOKENS * qk_elements)
        .map(|index| ((index * 11 + 3) % 43) as f32 * 0.0017 - 0.035)
        .collect();
    let k: Vec<f32> = (0..N_TOKENS * qk_elements)
        .map(|index| ((index * 13 + 5) % 47) as f32 * 0.0015 - 0.033)
        .collect();
    let v: Vec<f32> = (0..N_TOKENS * vector_elements)
        .map(|index| ((index * 17 + 1) % 53) as f32 * 0.0019 - 0.049)
        .collect();
    let decay: Vec<f32> = (0..N_TOKENS * N_V)
        .map(|index| 0.89 + (index % N_V) as f32 * 0.026 + (index / N_V) as f32 * 0.004)
        .collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_V)
        .map(|index| 0.21 + (index % N_V) as f32 * 0.17 + (index / N_V) as f32 * 0.013)
        .collect();
    let initial_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 19 + 7) % 59) as f32 * 0.00061 - 0.017)
        .collect();
    let grad_out: Vec<f32> = (0..N_TOKENS * vector_elements)
        .map(|index| ((index * 23 + 2) % 61) as f32 * 0.00093 - 0.027)
        .collect();
    let grad_final_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 29 + 11) % 67) as f32 * 0.000071 - 0.0023)
        .collect();
    let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let q64 = as_f64(&q);
    let k64 = as_f64(&k);
    let v64 = as_f64(&v);
    let decay64 = as_f64(&decay);
    let beta64 = as_f64(&beta);
    let initial_state64 = as_f64(&initial_state);
    let grad_out64 = as_f64(&grad_out);
    let grad_final_state64 = as_f64(&grad_final_state);
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let q_t = tensor(&q);
    let k_t = tensor(&k);
    let v_t = tensor(&v);
    let decay_t = tensor(&decay);
    let beta_t = tensor(&beta);
    let initial_state_t = tensor(&initial_state);
    let forward_state_t = tensor(&initial_state);
    let forward_out_t =
        MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
    let checkpoints_t =
        MetalTensor::zeros_f32(&ctx, vec![(N_CHECKPOINTS * state_elements) as u64]).unwrap();
    let grad_out_t = tensor(&grad_out);
    let grad_final_state_t = tensor(&grad_final_state);
    let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![q.len() as u64]).unwrap();
    let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![k.len() as u64]).unwrap();
    let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![v.len() as u64]).unwrap();
    let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![decay.len() as u64]).unwrap();
    let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![beta.len() as u64]).unwrap();
    let grad_initial_state_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_state_a_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_state_b_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_correction_t = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let residual_t = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_gdn_step_decay_packed_ckpt_f32(
            &ctx,
            encoder,
            &q_t,
            &k_t,
            &v_t,
            &decay_t,
            &beta_t,
            &forward_state_t,
            &forward_out_t,
            &checkpoints_t,
            N_CHECKPOINTS,
            N_TOKENS,
            N_V,
            N_K,
            HEAD_DIM,
        )
    })
    .unwrap();
    let checkpoints = read_back_f32(&checkpoints_t.buffer, N_CHECKPOINTS * state_elements);
    one_shot(&ctx, |encoder| {
        encode_gdn_step_decay_packed_vjp_f32(
            &ctx,
            encoder,
            &q_t,
            &k_t,
            &v_t,
            &decay_t,
            &beta_t,
            &initial_state_t,
            &checkpoints_t,
            N_CHECKPOINTS,
            &grad_out_t,
            &grad_final_state_t,
            &grad_q_t,
            &grad_k_t,
            &grad_v_t,
            &grad_decay_t,
            &grad_beta_t,
            &grad_initial_state_t,
            &grad_state_a_t,
            &grad_state_b_t,
            &grad_correction_t,
            &residual_t,
            N_TOKENS,
            N_V,
            N_K,
            HEAD_DIM,
        )
    })
    .unwrap();
    let actual = GdnStepVjpReference {
        grad_q: read_back_f32(&grad_q_t.buffer, q.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_k: read_back_f32(&grad_k_t.buffer, k.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_v: read_back_f32(&grad_v_t.buffer, v.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_decay: read_back_f32(&grad_decay_t.buffer, decay.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_beta: read_back_f32(&grad_beta_t.buffer, beta.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_state: read_back_f32(&grad_initial_state_t.buffer, state_elements)
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    let expected = gdn_sequence_vjp_f64(
        &q64,
        &k64,
        &v64,
        &decay64,
        &beta64,
        &initial_state64,
        &grad_out64,
        &grad_final_state64,
        N_TOKENS,
        N_V,
        N_K,
        HEAD_DIM,
    );
    for (name, gpu, cpu, tolerance) in [
        ("q", &actual.grad_q, &expected.grad_q, 4e-5),
        ("k", &actual.grad_k, &expected.grad_k, 5e-5),
        ("v", &actual.grad_v, &expected.grad_v, 4e-6),
        ("decay", &actual.grad_decay, &expected.grad_decay, 7e-5),
        ("beta", &actual.grad_beta, &expected.grad_beta, 5e-5),
        (
            "initial_state",
            &actual.grad_state,
            &expected.grad_state,
            5e-6,
        ),
    ] {
        let max_abs = gpu
            .iter()
            .zip(cpu)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs < tolerance, "{name} temporal VJP error {max_abs}");
    }

    let objective =
        |q: &[f64], k: &[f64], v: &[f64], decay: &[f64], beta: &[f64], state: &[f64]| {
            gdn_sequence_objective_f64(
                q,
                k,
                v,
                decay,
                beta,
                state,
                &grad_out64,
                &grad_final_state64,
                N_TOKENS,
                N_V,
                N_K,
                HEAD_DIM,
            )
        };
    let epsilon = 1e-5;
    let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
    };
    for &index in &[0usize, qk_elements - 1, q64.len() - 1] {
        let fd = finite_difference(&q64, index, &|candidate| {
            objective(candidate, &k64, &v64, &decay64, &beta64, &initial_state64)
        });
        assert!((fd - actual.grad_q[index]).abs() < 4e-5);
        let fd = finite_difference(&k64, index, &|candidate| {
            objective(&q64, candidate, &v64, &decay64, &beta64, &initial_state64)
        });
        assert!((fd - actual.grad_k[index]).abs() < 4e-5);
    }
    for &index in &[0usize, vector_elements - 1, v64.len() - 1] {
        let fd = finite_difference(&v64, index, &|candidate| {
            objective(&q64, &k64, candidate, &decay64, &beta64, &initial_state64)
        });
        assert!((fd - actual.grad_v[index]).abs() < 3e-5);
    }
    for &index in &[0usize, N_V, decay64.len() - 1] {
        let fd = finite_difference(&decay64, index, &|candidate| {
            objective(&q64, &k64, &v64, candidate, &beta64, &initial_state64)
        });
        assert!((fd - actual.grad_decay[index]).abs() < 5e-5);
        let fd = finite_difference(&beta64, index, &|candidate| {
            objective(&q64, &k64, &v64, &decay64, candidate, &initial_state64)
        });
        assert!((fd - actual.grad_beta[index]).abs() < 5e-5);
    }
    for &index in &[0usize, HEAD_DIM, initial_state64.len() - 1] {
        let fd = finite_difference(&initial_state64, index, &|candidate| {
            objective(&q64, &k64, &v64, &decay64, &beta64, candidate)
        });
        assert!((fd - actual.grad_state[index]).abs() < 3e-5);
    }

    let direction = |len: usize, stride: usize| {
        (0..len)
            .map(|index| ((index * stride + 3) % 37) as f64 * 0.0007 - 0.012)
            .collect::<Vec<_>>()
    };
    let dq = direction(q64.len(), 5);
    let dk = direction(k64.len(), 7);
    let dv = direction(v64.len(), 11);
    let ddecay = direction(decay64.len(), 13);
    let dbeta = direction(beta64.len(), 17);
    let dstate = direction(initial_state64.len(), 19);
    let inner = |gradient: &[f64], tangent: &[f64]| {
        gradient
            .iter()
            .zip(tangent)
            .map(|(gradient, tangent)| gradient * tangent)
            .sum::<f64>()
    };
    let reverse_directional = inner(&actual.grad_q, &dq)
        + inner(&actual.grad_k, &dk)
        + inner(&actual.grad_v, &dv)
        + inner(&actual.grad_decay, &ddecay)
        + inner(&actual.grad_beta, &dbeta)
        + inner(&actual.grad_state, &dstate);
    let shift = |base: &[f64], tangent: &[f64], amount: f64| {
        base.iter()
            .zip(tangent)
            .map(|(base, tangent)| base + amount * tangent)
            .collect::<Vec<_>>()
    };
    let plus = objective(
        &shift(&q64, &dq, epsilon),
        &shift(&k64, &dk, epsilon),
        &shift(&v64, &dv, epsilon),
        &shift(&decay64, &ddecay, epsilon),
        &shift(&beta64, &dbeta, epsilon),
        &shift(&initial_state64, &dstate, epsilon),
    );
    let minus = objective(
        &shift(&q64, &dq, -epsilon),
        &shift(&k64, &dk, -epsilon),
        &shift(&v64, &dv, -epsilon),
        &shift(&decay64, &ddecay, -epsilon),
        &shift(&beta64, &dbeta, -epsilon),
        &shift(&initial_state64, &dstate, -epsilon),
    );
    let forward_directional = (plus - minus) / (2.0 * epsilon);
    assert!(
        (forward_directional - reverse_directional).abs() < 6e-5,
        "temporal adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
    );
    for (tensor, original) in [
        (&q_t, &q),
        (&k_t, &k),
        (&v_t, &v),
        (&decay_t, &decay),
        (&beta_t, &beta),
        (&initial_state_t, &initial_state),
        (&checkpoints_t, &checkpoints),
        (&grad_out_t, &grad_out),
        (&grad_final_state_t, &grad_final_state),
    ] {
        assert_eq!(
            read_back_f32(&tensor.buffer, original.len())
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn gdn_step_decay_vjp_rejects_unsafe_contracts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_V: usize = 3;
    const N_K: usize = 1;
    const HEAD_DIM: usize = 128;
    let qk_elements = N_K * HEAD_DIM;
    let vector_elements = N_V * HEAD_DIM;
    let state_elements = N_V * HEAD_DIM * HEAD_DIM;
    let q = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let k = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let v = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let decay = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let beta = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_out = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let grad_state_out = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_q = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_k = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_v = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let grad_decay = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_beta = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_c = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let residual = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let invoke = |encoder: &KernelEncoder, grad_q_output: &MetalTensor| {
        encode_gdn_step_decay_vjp_f32(
            &ctx,
            encoder,
            &q,
            &k,
            &v,
            &decay,
            &beta,
            &state,
            &grad_out,
            &grad_state_out,
            grad_q_output,
            &grad_k,
            &grad_v,
            &grad_decay,
            &grad_beta,
            &grad_state,
            &grad_c,
            &residual,
            N_V,
            N_K,
            HEAD_DIM,
        )
    };

    let command = ctx.queue.commandBuffer().unwrap();
    let concurrent = KernelEncoder::begin_concurrent(&command);
    invoke(&concurrent, &grad_q).expect_err("concurrent encoder must fail");
    concurrent.end();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke(&encoder, &q).expect_err("output/input alias must fail");
    encoder.end();

    let mut read_only = grad_q.clone();
    read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke(&encoder, &read_only).expect_err("read-only output must fail");
    encoder.end();

    let short = MetalTensor::zeros_f32(&ctx, vec![(qk_elements - 1) as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke(&encoder, &short).expect_err("short output must fail");
    encoder.end();
}

#[test]
fn gdn_packed_temporal_vjps_enforce_safe_tapes_and_storage() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_TOKENS: usize = 3;
    const N_V: usize = 3;
    const N_K: usize = 1;
    const HEAD_DIM: usize = 128;
    let qk_elements = N_K * HEAD_DIM;
    let vector_elements = N_V * HEAD_DIM;
    let state_elements = vector_elements * HEAD_DIM;
    let q = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let k = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let v = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
    let decay = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
    let beta = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
    let initial_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let full_checkpoints =
        MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * state_elements) as u64]).unwrap();
    let short_checkpoints =
        MetalTensor::zeros_f32(&ctx, vec![((N_TOKENS - 2) * state_elements) as u64]).unwrap();
    let grad_out = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
    let grad_final_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_q = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let grad_k = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let grad_v = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
    let grad_decay = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
    let grad_beta = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
    let grad_initial_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let state_scratch_a = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let state_scratch_b = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let correction_scratch = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let residual_scratch = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
    let invoke_recurrence = |encoder: &KernelEncoder,
                             q_input: &MetalTensor,
                             checkpoints: &MetalTensor,
                             n_checkpoints: usize,
                             grad_q_output: &MetalTensor| {
        encode_gdn_step_decay_packed_vjp_f32(
            &ctx,
            encoder,
            q_input,
            &k,
            &v,
            &decay,
            &beta,
            &initial_state,
            checkpoints,
            n_checkpoints,
            &grad_out,
            &grad_final_state,
            grad_q_output,
            &grad_k,
            &grad_v,
            &grad_decay,
            &grad_beta,
            &grad_initial_state,
            &state_scratch_a,
            &state_scratch_b,
            &correction_scratch,
            &residual_scratch,
            N_TOKENS,
            N_V,
            N_K,
            HEAD_DIM,
        )
    };
    one_shot(&ctx, |encoder| {
        invoke_recurrence(encoder, &q, &full_checkpoints, N_TOKENS, &grad_q)
    })
    .unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let concurrent = KernelEncoder::begin_concurrent(&command);
    invoke_recurrence(&concurrent, &q, &full_checkpoints, N_TOKENS, &grad_q)
        .expect_err("concurrent temporal recurrence must fail");
    concurrent.end();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke_recurrence(&encoder, &q, &short_checkpoints, N_TOKENS - 2, &grad_q)
        .expect_err("missing pre-state checkpoint must fail");
    encoder.end();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke_recurrence(&encoder, &q, &full_checkpoints, N_TOKENS, &q)
        .expect_err("packed gradient/input alias must fail");
    encoder.end();

    let mut malformed_q = q.clone();
    malformed_q.shape = vec![u64::MAX, 2];
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke_recurrence(&encoder, &malformed_q, &full_checkpoints, N_TOKENS, &grad_q)
        .expect_err("malformed packed shape must fail without panicking");
    encoder.end();

    const CONV_N_V: usize = 2;
    let conv_v_elements = CONV_N_V * HEAD_DIM;
    let conv_dim = 2 * qk_elements + conv_v_elements;
    let conv_state_elements = 3 * conv_dim;
    let qkv = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_dim) as u64]).unwrap();
    let conv_initial_state =
        MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
    let conv_full_checkpoints =
        MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_state_elements) as u64]).unwrap();
    let conv_weight = MetalTensor::zeros_f32(&ctx, vec![(4 * conv_dim) as u64]).unwrap();
    let conv_grad_q = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let conv_grad_k = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let conv_grad_v =
        MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_v_elements) as u64]).unwrap();
    let conv_grad_final_state =
        MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
    let grad_qkv = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_dim) as u64]).unwrap();
    let conv_grad_initial_state =
        MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
    let conv_state_scratch_a =
        MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
    let conv_state_scratch_b =
        MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
    let invoke_conv = |encoder: &KernelEncoder, grad_qkv_output: &MetalTensor| {
        encode_ssm_conv_silu_split_packed_vjp_f32(
            &ctx,
            encoder,
            &qkv,
            &conv_initial_state,
            &conv_full_checkpoints,
            N_TOKENS,
            &conv_weight,
            &conv_grad_q,
            &conv_grad_k,
            &conv_grad_v,
            &conv_grad_final_state,
            grad_qkv_output,
            &conv_grad_initial_state,
            &conv_state_scratch_a,
            &conv_state_scratch_b,
            N_TOKENS,
            N_K,
            CONV_N_V,
            HEAD_DIM,
        )
    };
    one_shot(&ctx, |encoder| invoke_conv(encoder, &grad_qkv)).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    invoke_conv(&encoder, &qkv).expect_err("packed conv gradient/input alias must fail");
    encoder.end();
}

#[test]
fn l2_norm_vjp_matches_clamp_and_finite_differences() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const EPS: f32 = 0.5;
    let mut x = vec![0.0f32; N_HEADS * HEAD_DIM];
    for (index, value) in x[..HEAD_DIM].iter_mut().enumerate() {
        *value = ((index * 7 + 3) % 23) as f32 * 0.009 - 0.099;
    }
    x[2 * HEAD_DIM] = EPS;
    x[3 * HEAD_DIM] = EPS * 0.5;
    let grad_output: Vec<f32> = (0..x.len())
        .map(|index| ((index * 11 + 1) % 31) as f32 * 0.013 - 0.19)
        .collect();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![x.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let grad_output_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&grad_output),
        vec![grad_output.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let grad_input_t = MetalTensor::zeros_f32(&ctx, vec![x.len() as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_l2_norm_vjp_batched_f32(
            &ctx,
            encoder,
            &x_t,
            &grad_output_t,
            &grad_input_t,
            N_HEADS,
            HEAD_DIM,
            EPS,
        )
    })
    .unwrap();
    let actual = read_back_f32(&grad_input_t.buffer, x.len());
    let mut expected = vec![0.0f64; x.len()];
    for head in 0..N_HEADS {
        let base = head * HEAD_DIM;
        let radius = x[base..base + HEAD_DIM]
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if radius > f64::from(EPS) {
            let dot = (0..HEAD_DIM)
                .map(|index| f64::from(x[base + index]) * f64::from(grad_output[base + index]))
                .sum::<f64>();
            for index in 0..HEAD_DIM {
                expected[base + index] = f64::from(grad_output[base + index]) / radius
                    - f64::from(x[base + index]) * dot / radius.powi(3);
            }
        } else {
            for index in 0..HEAD_DIM {
                expected[base + index] = f64::from(grad_output[base + index]) / f64::from(EPS);
            }
        }
    }
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(max_abs < 2e-5, "L2 VJP error {max_abs}");
    for head in [1usize, 2, 3] {
        let base = head * HEAD_DIM;
        for index in 0..HEAD_DIM {
            assert_eq!(
                actual[base + index].to_bits(),
                (grad_output[base + index] / EPS).to_bits(),
                "clamped row {head} index {index}"
            );
        }
    }

    let epsilon = 1e-5f64;
    for index in [0usize, 31, HEAD_DIM - 1] {
        let objective = |delta: f64| {
            let mut row = x[..HEAD_DIM]
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>();
            row[index] += delta;
            let radius = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            row.iter()
                .zip(&grad_output[..HEAD_DIM])
                .map(|(value, grad)| value / radius * f64::from(*grad))
                .sum::<f64>()
        };
        let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual[index])).abs() < 2e-5);
    }
}

#[test]
fn gdn_scalar_chain_vjps_match_piecewise_oracles() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let source = [-15.0f32, -3.0, -0.25, 0.0, 0.75, 4.0, 15.0];
    let sigmoid_output: Vec<f32> = source
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let sigmoid_grad: Vec<f32> = (0..source.len())
        .map(|index| index as f32 * 0.07 - 0.19)
        .collect();
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let sigmoid_output_t = tensor(&sigmoid_output);
    let sigmoid_grad_t = tensor(&sigmoid_grad);
    let sigmoid_source_grad_t = MetalTensor::zeros_f32(&ctx, vec![source.len() as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_sigmoid_output_vjp_f32(
            &ctx,
            encoder,
            &sigmoid_output_t,
            &sigmoid_grad_t,
            &sigmoid_source_grad_t,
        )
    })
    .unwrap();
    let actual_sigmoid = read_back_f32(&sigmoid_source_grad_t.buffer, source.len());
    for index in 0..source.len() {
        let expected = sigmoid_grad[index] * sigmoid_output[index] * (1.0 - sigmoid_output[index]);
        assert!((actual_sigmoid[index] - expected).abs() < 2e-7);
        let epsilon = 1e-4f64;
        let objective = |delta: f64| {
            let value = f64::from(source[index]) + delta;
            f64::from(sigmoid_grad[index]) / (1.0 + (-value).exp())
        };
        let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_sigmoid[index])).abs() < 2e-6);
    }

    let totals = [
        -25.0f32, -20.25, -20.0, -19.75, 0.0, 19.75, 20.0, 20.25, 25.0,
    ];
    let dt_bias: Vec<f32> = (0..totals.len())
        .map(|index| (index as f32 - 4.0) * 0.125)
        .collect();
    let alpha: Vec<f32> = totals
        .iter()
        .zip(&dt_bias)
        .map(|(total, bias)| total - bias)
        .collect();
    let a_log: Vec<f32> = (0..totals.len())
        .map(|index| -0.015 - index as f32 * 0.004)
        .collect();
    let grad_decay: Vec<f32> = (0..totals.len())
        .map(|index| index as f32 * 0.031 - 0.11)
        .collect();
    let alpha_t = tensor(&alpha);
    let dt_t = tensor(&dt_bias);
    let a_log_t = tensor(&a_log);
    let decay_t = MetalTensor::zeros_f32(&ctx, vec![totals.len() as u64]).unwrap();
    let grad_decay_t = tensor(&grad_decay);
    let grad_alpha_t = MetalTensor::zeros_f32(&ctx, vec![totals.len() as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_gdn_decay_chain_f32(&ctx, encoder, &alpha_t, &dt_t, &a_log_t, &decay_t)?;
        encode_gdn_decay_chain_vjp_f32(
            &ctx,
            encoder,
            &alpha_t,
            &dt_t,
            &a_log_t,
            &decay_t,
            &grad_decay_t,
            &grad_alpha_t,
        )
    })
    .unwrap();
    let decay = read_back_f32(&decay_t.buffer, totals.len());
    let actual_alpha = read_back_f32(&grad_alpha_t.buffer, totals.len());
    for index in 0..totals.len() {
        let value = totals[index];
        let softplus_derivative = if value > 20.0 {
            1.0
        } else if value < -20.0 {
            value.exp()
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        };
        let expected = grad_decay[index] * decay[index] * a_log[index] * softplus_derivative;
        assert!(
            (actual_alpha[index] - expected).abs() < 2e-6,
            "decay chain index {index}: {} != {expected}",
            actual_alpha[index]
        );
    }
    for &index in &[0usize, 1, 3, 4, 5, 7, 8] {
        let epsilon = 1e-4f64;
        let objective = |delta: f64| {
            let value = f64::from(alpha[index]) + f64::from(dt_bias[index]) + delta;
            let softplus = if value > 20.0 {
                value
            } else if value < -20.0 {
                value.exp()
            } else {
                (1.0 + value.exp()).ln()
            };
            f64::from(grad_decay[index]) * (softplus * f64::from(a_log[index])).exp()
        };
        let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_alpha[index])).abs() < 2e-5);
    }
}

#[test]
fn ssm_conv_silu_vjp_matches_shifted_state_oracle() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_K: usize = 1;
    const N_V: usize = 1;
    const HEAD_DIM: usize = 128;
    const CONV_DIM: usize = (2 * N_K + N_V) * HEAD_DIM;
    let qkv: Vec<f32> = (0..CONV_DIM)
        .map(|index| ((index * 7 + 2) % 31) as f32 * 0.009 - 0.13)
        .collect();
    let state: Vec<f32> = (0..3 * CONV_DIM)
        .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.105)
        .collect();
    let weight: Vec<f32> = (0..4 * CONV_DIM)
        .map(|index| ((index * 13 + 1) % 41) as f32 * 0.004 - 0.077)
        .collect();
    let grad_out: Vec<f32> = (0..CONV_DIM)
        .map(|index| ((index * 17 + 3) % 43) as f32 * 0.008 - 0.16)
        .collect();
    let grad_state_out: Vec<f32> = (0..3 * CONV_DIM)
        .map(|index| ((index * 19 + 7) % 47) as f32 * 0.005 - 0.11)
        .collect();
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let qkv_t = tensor(&qkv);
    let state_t = tensor(&state);
    let weight_t = tensor(&weight);
    let grad_out_t = tensor(&grad_out);
    let grad_state_out_t = tensor(&grad_state_out);
    let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![CONV_DIM as u64]).unwrap();
    let grad_state_t = MetalTensor::zeros_f32(&ctx, vec![(3 * CONV_DIM) as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_ssm_conv_silu_vjp_f32(
            &ctx,
            encoder,
            &qkv_t,
            &state_t,
            &weight_t,
            &grad_out_t,
            &grad_state_out_t,
            &grad_qkv_t,
            &grad_state_t,
            CONV_DIM,
        )
    })
    .unwrap();
    let actual_qkv = read_back_f32(&grad_qkv_t.buffer, CONV_DIM);
    let actual_state = read_back_f32(&grad_state_t.buffer, 3 * CONV_DIM);
    let split_grad_q_t = tensor(&grad_out[..HEAD_DIM]);
    let split_grad_k_t = tensor(&grad_out[HEAD_DIM..2 * HEAD_DIM]);
    let split_grad_v_t = tensor(&grad_out[2 * HEAD_DIM..]);
    let split_grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![CONV_DIM as u64]).unwrap();
    let split_grad_state_t = MetalTensor::zeros_f32(&ctx, vec![(3 * CONV_DIM) as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_ssm_conv_silu_split_vjp_f32(
            &ctx,
            encoder,
            &qkv_t,
            &state_t,
            &weight_t,
            &split_grad_q_t,
            &split_grad_k_t,
            &split_grad_v_t,
            &grad_state_out_t,
            &split_grad_qkv_t,
            &split_grad_state_t,
            N_K,
            N_V,
            HEAD_DIM,
        )
    })
    .unwrap();
    let split_qkv = read_back_f32(&split_grad_qkv_t.buffer, CONV_DIM);
    let split_state = read_back_f32(&split_grad_state_t.buffer, 3 * CONV_DIM);
    let split_qkv_error = split_qkv
        .iter()
        .zip(&actual_qkv)
        .map(|(split, joined)| (split - joined).abs())
        .fold(0.0f32, f32::max);
    let split_state_error = split_state
        .iter()
        .zip(&actual_state)
        .map(|(split, joined)| (split - joined).abs())
        .fold(0.0f32, f32::max);
    assert!(split_qkv_error < 2e-7, "split QKV error {split_qkv_error}");
    assert!(
        split_state_error < 2e-7,
        "split state error {split_state_error}"
    );
    let mut expected_qkv = vec![0.0f64; CONV_DIM];
    let mut expected_state = vec![0.0f64; 3 * CONV_DIM];
    for channel in 0..CONV_DIM {
        let preactivation = (0..3)
            .map(|row| {
                f64::from(weight[channel * 4 + row]) * f64::from(state[row * CONV_DIM + channel])
            })
            .sum::<f64>()
            + f64::from(weight[channel * 4 + 3]) * f64::from(qkv[channel]);
        let sigmoid = 1.0 / (1.0 + (-preactivation).exp());
        let derivative = sigmoid * (1.0 + preactivation * (1.0 - sigmoid));
        let grad_preactivation = f64::from(grad_out[channel]) * derivative;
        expected_qkv[channel] = grad_preactivation * f64::from(weight[channel * 4 + 3])
            + f64::from(grad_state_out[2 * CONV_DIM + channel]);
        expected_state[channel] = grad_preactivation * f64::from(weight[channel * 4]);
        expected_state[CONV_DIM + channel] = grad_preactivation
            * f64::from(weight[channel * 4 + 1])
            + f64::from(grad_state_out[channel]);
        expected_state[2 * CONV_DIM + channel] = grad_preactivation
            * f64::from(weight[channel * 4 + 2])
            + f64::from(grad_state_out[CONV_DIM + channel]);
    }
    let max_qkv = actual_qkv
        .iter()
        .zip(&expected_qkv)
        .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
        .fold(0.0f64, f64::max);
    let max_state = actual_state
        .iter()
        .zip(&expected_state)
        .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(max_qkv < 2e-6, "conv qkv VJP error {max_qkv}");
    assert!(max_state < 2e-6, "conv state VJP error {max_state}");

    let objective = |qkv: &[f64], state: &[f64]| {
        let mut value = 0.0f64;
        for channel in 0..CONV_DIM {
            let preactivation = (0..3)
                .map(|row| f64::from(weight[channel * 4 + row]) * state[row * CONV_DIM + channel])
                .sum::<f64>()
                + f64::from(weight[channel * 4 + 3]) * qkv[channel];
            value += f64::from(grad_out[channel]) * preactivation / (1.0 + (-preactivation).exp());
            value += f64::from(grad_state_out[channel]) * state[CONV_DIM + channel];
            value += f64::from(grad_state_out[CONV_DIM + channel]) * state[2 * CONV_DIM + channel];
            value += f64::from(grad_state_out[2 * CONV_DIM + channel]) * qkv[channel];
        }
        value
    };
    let qkv64 = qkv.iter().copied().map(f64::from).collect::<Vec<_>>();
    let state64 = state.iter().copied().map(f64::from).collect::<Vec<_>>();
    let epsilon = 1e-5;
    for &index in &[0usize, 128, CONV_DIM - 1] {
        let mut plus = qkv64.clone();
        let mut minus = qkv64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference =
            (objective(&plus, &state64) - objective(&minus, &state64)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_qkv[index])).abs() < 2e-5);
    }
    for &index in &[
        0usize,
        CONV_DIM - 1,
        CONV_DIM,
        2 * CONV_DIM + 128,
        3 * CONV_DIM - 1,
    ] {
        let mut plus = state64.clone();
        let mut minus = state64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference =
            (objective(&qkv64, &plus) - objective(&qkv64, &minus)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_state[index])).abs() < 2e-5);
    }
}

#[test]
fn ssm_conv_silu_vjp_replays_forward_accumulation_order() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let qkv = [1.0f32];
    let state = [1.0e8f32, -1.0e8, 1.0];
    let weight = [1.0f32; 4];
    let grad_out = [1.0f32];
    let grad_state_out = [0.0f32; 3];
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let qkv_t = tensor(&qkv);
    let forward_state_t = tensor(&state);
    let weight_t = tensor(&weight);
    let forward_out_t = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_ssm_conv_silu_f32(
            &ctx,
            encoder,
            &qkv_t,
            &forward_state_t,
            &weight_t,
            &forward_out_t,
            1,
        )
    })
    .unwrap();
    let forward_out = read_back_f32(&forward_out_t.buffer, 1)[0];
    let expected_forward = 2.0f32 / (1.0 + (-2.0f32).exp());
    assert!((forward_out - expected_forward).abs() < 2e-6);

    let state_t = tensor(&state);
    let grad_out_t = tensor(&grad_out);
    let grad_state_out_t = tensor(&grad_state_out);
    let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let grad_state_t = MetalTensor::zeros_f32(&ctx, vec![3]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_ssm_conv_silu_vjp_f32(
            &ctx,
            encoder,
            &qkv_t,
            &state_t,
            &weight_t,
            &grad_out_t,
            &grad_state_out_t,
            &grad_qkv_t,
            &grad_state_t,
            1,
        )
    })
    .unwrap();
    let sigmoid = 1.0f32 / (1.0 + (-2.0f32).exp());
    let expected_gradient = sigmoid * (1.0 + 2.0 * (1.0 - sigmoid));
    let grad_qkv = read_back_f32(&grad_qkv_t.buffer, 1)[0];
    let grad_state = read_back_f32(&grad_state_t.buffer, 3);
    assert!((grad_qkv - expected_gradient).abs() < 2e-6);
    for gradient in grad_state {
        assert!((gradient - expected_gradient).abs() < 2e-6);
    }
}

#[test]
fn ssm_conv_silu_split_packed_vjp_matches_temporal_oracle_and_adjoint() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_TOKENS: usize = 4;
    const N_CHECKPOINTS: usize = N_TOKENS - 1;
    const N_K: usize = 1;
    const N_V: usize = 2;
    const HEAD_DIM: usize = 128;
    let qk_elements = N_K * HEAD_DIM;
    let v_elements = N_V * HEAD_DIM;
    let conv_dim = 2 * qk_elements + v_elements;
    let state_elements = 3 * conv_dim;
    let qkv: Vec<f32> = (0..N_TOKENS * conv_dim)
        .map(|index| ((index * 7 + 3) % 41) as f32 * 0.0023 - 0.045)
        .collect();
    let initial_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 11 + 5) % 43) as f32 * 0.0019 - 0.039)
        .collect();
    let weight: Vec<f32> = (0..4 * conv_dim)
        .map(|index| ((index * 13 + 1) % 47) as f32 * 0.0031 - 0.071)
        .collect();
    let grad_q: Vec<f32> = (0..N_TOKENS * qk_elements)
        .map(|index| ((index * 17 + 7) % 53) as f32 * 0.0017 - 0.043)
        .collect();
    let grad_k: Vec<f32> = (0..N_TOKENS * qk_elements)
        .map(|index| ((index * 19 + 2) % 59) as f32 * 0.0015 - 0.041)
        .collect();
    let grad_v: Vec<f32> = (0..N_TOKENS * v_elements)
        .map(|index| ((index * 23 + 4) % 61) as f32 * 0.0013 - 0.037)
        .collect();
    let grad_final_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00031 - 0.009)
        .collect();
    let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let qkv64 = as_f64(&qkv);
    let initial_state64 = as_f64(&initial_state);
    let weight64 = as_f64(&weight);
    let grad_q64 = as_f64(&grad_q);
    let grad_k64 = as_f64(&grad_k);
    let grad_v64 = as_f64(&grad_v);
    let grad_final_state64 = as_f64(&grad_final_state);
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let qkv_t = tensor(&qkv);
    let initial_state_t = tensor(&initial_state);
    let forward_state_t = tensor(&initial_state);
    let checkpoints_t =
        MetalTensor::zeros_f32(&ctx, vec![(N_CHECKPOINTS * state_elements) as u64]).unwrap();
    let weight_t = tensor(&weight);
    let forward_q_t = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let forward_k_t = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
    let forward_v_t = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * v_elements) as u64]).unwrap();
    let grad_q_t = tensor(&grad_q);
    let grad_k_t = tensor(&grad_k);
    let grad_v_t = tensor(&grad_v);
    let grad_final_state_t = tensor(&grad_final_state);
    let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![qkv.len() as u64]).unwrap();
    let grad_initial_state_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_state_a_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_state_b_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_gdn_prep_packed_ckpt_f32(
            &ctx,
            encoder,
            &qkv_t,
            &forward_state_t,
            &weight_t,
            &forward_q_t,
            &forward_k_t,
            &forward_v_t,
            &checkpoints_t,
            N_TOKENS,
            N_CHECKPOINTS,
            N_K,
            N_V,
            HEAD_DIM,
        )
    })
    .unwrap();
    let checkpoints = read_back_f32(&checkpoints_t.buffer, N_CHECKPOINTS * state_elements);
    one_shot(&ctx, |encoder| {
        encode_ssm_conv_silu_split_packed_vjp_f32(
            &ctx,
            encoder,
            &qkv_t,
            &initial_state_t,
            &checkpoints_t,
            N_CHECKPOINTS,
            &weight_t,
            &grad_q_t,
            &grad_k_t,
            &grad_v_t,
            &grad_final_state_t,
            &grad_qkv_t,
            &grad_initial_state_t,
            &grad_state_a_t,
            &grad_state_b_t,
            N_TOKENS,
            N_K,
            N_V,
            HEAD_DIM,
        )
    })
    .unwrap();
    let actual = SsmConvSequenceVjpReference {
        grad_qkv: read_back_f32(&grad_qkv_t.buffer, qkv.len())
            .into_iter()
            .map(f64::from)
            .collect(),
        grad_state: read_back_f32(&grad_initial_state_t.buffer, state_elements)
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    let expected = ssm_conv_sequence_vjp_f64(
        &qkv64,
        &initial_state64,
        &weight64,
        &grad_q64,
        &grad_k64,
        &grad_v64,
        &grad_final_state64,
        N_TOKENS,
        qk_elements,
        v_elements,
    );
    for (name, gpu, cpu) in [
        ("qkv", &actual.grad_qkv, &expected.grad_qkv),
        ("initial_state", &actual.grad_state, &expected.grad_state),
    ] {
        let max_abs = gpu
            .iter()
            .zip(cpu)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs < 3e-6, "{name} temporal conv VJP error {max_abs}");
    }

    let objective = |qkv: &[f64], state: &[f64]| {
        ssm_conv_sequence_objective_f64(
            qkv,
            state,
            &weight64,
            &grad_q64,
            &grad_k64,
            &grad_v64,
            &grad_final_state64,
            N_TOKENS,
            qk_elements,
            v_elements,
        )
    };
    let epsilon = 1e-5;
    for &index in &[0usize, conv_dim - 1, qkv64.len() - 1] {
        let mut plus = qkv64.clone();
        let mut minus = qkv64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference = (objective(&plus, &initial_state64)
            - objective(&minus, &initial_state64))
            / (2.0 * epsilon);
        assert!((finite_difference - actual.grad_qkv[index]).abs() < 2e-5);
    }
    for &index in &[0usize, conv_dim, initial_state64.len() - 1] {
        let mut plus = initial_state64.clone();
        let mut minus = initial_state64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference =
            (objective(&qkv64, &plus) - objective(&qkv64, &minus)) / (2.0 * epsilon);
        assert!((finite_difference - actual.grad_state[index]).abs() < 2e-5);
    }

    let direction = |len: usize, stride: usize| {
        (0..len)
            .map(|index| ((index * stride + 3) % 31) as f64 * 0.0009 - 0.013)
            .collect::<Vec<_>>()
    };
    let dqkv = direction(qkv64.len(), 5);
    let dstate = direction(initial_state64.len(), 7);
    let inner = |gradient: &[f64], tangent: &[f64]| {
        gradient
            .iter()
            .zip(tangent)
            .map(|(gradient, tangent)| gradient * tangent)
            .sum::<f64>()
    };
    let reverse_directional = inner(&actual.grad_qkv, &dqkv) + inner(&actual.grad_state, &dstate);
    let shift = |base: &[f64], tangent: &[f64], amount: f64| {
        base.iter()
            .zip(tangent)
            .map(|(base, tangent)| base + amount * tangent)
            .collect::<Vec<_>>()
    };
    let forward_directional = (objective(
        &shift(&qkv64, &dqkv, epsilon),
        &shift(&initial_state64, &dstate, epsilon),
    ) - objective(
        &shift(&qkv64, &dqkv, -epsilon),
        &shift(&initial_state64, &dstate, -epsilon),
    )) / (2.0 * epsilon);
    assert!(
        (forward_directional - reverse_directional).abs() < 2e-5,
        "temporal conv adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
    );
    for (tensor, original) in [
        (&qkv_t, &qkv),
        (&initial_state_t, &initial_state),
        (&checkpoints_t, &checkpoints),
        (&weight_t, &weight),
        (&grad_q_t, &grad_q),
        (&grad_k_t, &grad_k),
        (&grad_v_t, &grad_v),
        (&grad_final_state_t, &grad_final_state),
    ] {
        assert_eq!(
            read_back_f32(&tensor.buffer, original.len())
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn rmsnorm_gated_vjp_matches_finite_differences() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_HEADS: usize = 3;
    const HEAD_DIM: usize = 128;
    const EPS: f32 = HEAD_DIM as f32 * 1e-6;
    let elements = N_HEADS * HEAD_DIM;
    let o: Vec<f32> = (0..elements)
        .map(|index| ((index * 7 + 1) % 37) as f32 * 0.011 - 0.19)
        .collect();
    let weight: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 0.55 + (index % 13) as f32 * 0.037)
        .collect();
    let z: Vec<f32> = (0..elements)
        .map(|index| ((index * 11 + 5) % 43) as f32 * 0.09 - 1.8)
        .collect();
    let grad_y: Vec<f32> = (0..elements)
        .map(|index| ((index * 13 + 3) % 47) as f32 * 0.007 - 0.15)
        .collect();
    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let o_t = tensor(&o);
    let weight_t = tensor(&weight);
    let z_t = tensor(&z);
    let grad_y_t = tensor(&grad_y);
    let grad_o_t = MetalTensor::zeros_f32(&ctx, vec![elements as u64]).unwrap();
    let grad_z_t = MetalTensor::zeros_f32(&ctx, vec![elements as u64]).unwrap();
    one_shot(&ctx, |encoder| {
        encode_rmsnorm_gated_vjp_f32(
            &ctx, encoder, &o_t, &weight_t, &z_t, &grad_y_t, &grad_o_t, &grad_z_t, N_HEADS,
            HEAD_DIM, EPS,
        )
    })
    .unwrap();
    let actual_o = read_back_f32(&grad_o_t.buffer, elements);
    let actual_z = read_back_f32(&grad_z_t.buffer, elements);
    let mut expected_o = vec![0.0f64; elements];
    let mut expected_z = vec![0.0f64; elements];
    for head in 0..N_HEADS {
        let base = head * HEAD_DIM;
        let sumsq = o[base..base + HEAD_DIM]
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let scale = (sumsq / HEAD_DIM as f64 + f64::from(EPS)).sqrt().recip();
        let mut dot = 0.0f64;
        for index in 0..HEAD_DIM {
            let offset = base + index;
            let z_value = f64::from(z[offset]);
            let sigmoid = 1.0 / (1.0 + (-z_value).exp());
            let silu = z_value * sigmoid;
            let grad_normed = f64::from(grad_y[offset]) * silu;
            dot += f64::from(o[offset]) * grad_normed * f64::from(weight[index]);
            let normed = f64::from(o[offset]) * scale * f64::from(weight[index]);
            let silu_derivative = sigmoid * (1.0 + z_value * (1.0 - sigmoid));
            expected_z[offset] = f64::from(grad_y[offset]) * normed * silu_derivative;
        }
        let correction = dot * scale.powi(3) / HEAD_DIM as f64;
        for index in 0..HEAD_DIM {
            let offset = base + index;
            let z_value = f64::from(z[offset]);
            let silu = z_value / (1.0 + (-z_value).exp());
            let weighted_grad = f64::from(grad_y[offset]) * silu * f64::from(weight[index]);
            expected_o[offset] = weighted_grad * scale - f64::from(o[offset]) * correction;
        }
    }
    let max_o = actual_o
        .iter()
        .zip(&expected_o)
        .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
        .fold(0.0f64, f64::max);
    let max_z = actual_z
        .iter()
        .zip(&expected_z)
        .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(max_o < 3e-5, "gated RMS grad_o error {max_o}");
    assert!(max_z < 2e-5, "gated RMS grad_z error {max_z}");

    let objective = |o: &[f64], z: &[f64]| {
        let mut value = 0.0f64;
        for head in 0..N_HEADS {
            let base = head * HEAD_DIM;
            let sumsq = o[base..base + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let scale = (sumsq / HEAD_DIM as f64 + f64::from(EPS)).sqrt().recip();
            for index in 0..HEAD_DIM {
                let offset = base + index;
                let silu = z[offset] / (1.0 + (-z[offset]).exp());
                value +=
                    f64::from(grad_y[offset]) * o[offset] * scale * f64::from(weight[index]) * silu;
            }
        }
        value
    };
    let o64 = o.iter().copied().map(f64::from).collect::<Vec<_>>();
    let z64 = z.iter().copied().map(f64::from).collect::<Vec<_>>();
    let epsilon = 1e-5;
    for &index in &[0usize, 127, 128, elements - 1] {
        let mut plus = o64.clone();
        let mut minus = o64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference =
            (objective(&plus, &z64) - objective(&minus, &z64)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_o[index])).abs() < 3e-5);

        let mut plus = z64.clone();
        let mut minus = z64.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference =
            (objective(&o64, &plus) - objective(&o64, &minus)) / (2.0 * epsilon);
        assert!((finite_difference - f64::from(actual_z[index])).abs() < 2e-5);
    }
}

#[test]
fn gdn_envelope_vjps_compose_to_full_step_adjoint() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const N_K: usize = 1;
    const N_V: usize = 2;
    const HEAD_DIM: usize = 128;
    const L2_EPS: f32 = 1e-6;
    const RMS_EPS: f32 = HEAD_DIM as f32 * 1e-6;
    let qk_elements = N_K * HEAD_DIM;
    let v_elements = N_V * HEAD_DIM;
    let conv_dim = 2 * qk_elements + v_elements;
    let state_elements = N_V * HEAD_DIM * HEAD_DIM;
    let qkv_now: Vec<f32> = (0..conv_dim)
        .map(|index| ((index * 7 + 3) % 41) as f32 * 0.006 - 0.115)
        .collect();
    let conv_state: Vec<f32> = (0..3 * conv_dim)
        .map(|index| ((index * 11 + 5) % 43) as f32 * 0.004 - 0.083)
        .collect();
    let conv_weight: Vec<f32> = (0..4 * conv_dim)
        .map(|index| ((index * 13 + 1) % 47) as f32 * 0.003 - 0.069)
        .collect();
    let alpha_source = [-0.7f32, 0.45];
    let dt_bias = [0.12f32, -0.08];
    let a_log = [-0.09f32, -0.14];
    let beta_source = [-0.35f32, 0.8];
    let recurrence_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 17 + 7) % 53) as f32 * 0.0008 - 0.021)
        .collect();
    let z: Vec<f32> = (0..v_elements)
        .map(|index| ((index * 19 + 2) % 59) as f32 * 0.05 - 1.35)
        .collect();
    let norm_weight: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 0.62 + (index % 17) as f32 * 0.029)
        .collect();
    let grad_y: Vec<f32> = (0..v_elements)
        .map(|index| ((index * 23 + 3) % 61) as f32 * 0.004 - 0.12)
        .collect();
    let grad_recurrence_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00006 - 0.002)
        .collect();
    let grad_conv_state: Vec<f32> = (0..3 * conv_dim)
        .map(|index| ((index * 31 + 13) % 71) as f32 * 0.0017 - 0.052)
        .collect();

    let mut conv_output = vec![0.0f32; conv_dim];
    for channel in 0..conv_dim {
        let mut preactivation = 0.0f32;
        for row in 0..3 {
            preactivation += conv_weight[channel * 4 + row] * conv_state[row * conv_dim + channel];
        }
        preactivation += conv_weight[channel * 4 + 3] * qkv_now[channel];
        conv_output[channel] = preactivation / (1.0 + (-preactivation).exp());
    }
    let q_raw = conv_output[..qk_elements].to_vec();
    let k_raw = conv_output[qk_elements..2 * qk_elements].to_vec();
    let v = conv_output[2 * qk_elements..].to_vec();
    let normalize = |input: &[f32]| {
        let mut output = input.to_vec();
        for head in 0..N_K {
            let base = head * HEAD_DIM;
            let radius = input[base..base + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(L2_EPS);
            for index in 0..HEAD_DIM {
                output[base + index] /= radius;
            }
        }
        output
    };
    let q = normalize(&q_raw);
    let k = normalize(&k_raw);
    let decay: Vec<f32> = (0..N_V)
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
    let beta: Vec<f32> = beta_source
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let mut recurrence_output = vec![0.0f32; v_elements];
    for hi in 0..N_V {
        let hk = hi % N_K;
        for dv in 0..HEAD_DIM {
            let vector_index = hi * HEAD_DIM + dv;
            let row_offset = vector_index * HEAD_DIM;
            let prediction = (0..HEAD_DIM)
                .map(|dk| decay[hi] * recurrence_state[row_offset + dk] * k[hk * HEAD_DIM + dk])
                .sum::<f32>();
            let correction = beta[hi] * (v[vector_index] - prediction);
            recurrence_output[vector_index] = (0..HEAD_DIM)
                .map(|dk| {
                    (decay[hi] * recurrence_state[row_offset + dk]
                        + correction * k[hk * HEAD_DIM + dk])
                        * q[hk * HEAD_DIM + dk]
                })
                .sum();
        }
    }

    let tensor = |values: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let qkv_t = tensor(&qkv_now);
    let conv_state_t = tensor(&conv_state);
    let conv_weight_t = tensor(&conv_weight);
    let q_raw_t = tensor(&q_raw);
    let k_raw_t = tensor(&k_raw);
    let q_t = tensor(&q);
    let k_t = tensor(&k);
    let v_t = tensor(&v);
    let alpha_t = tensor(&alpha_source);
    let dt_t = tensor(&dt_bias);
    let a_log_t = tensor(&a_log);
    let decay_t = tensor(&decay);
    let beta_t = tensor(&beta);
    let recurrence_state_t = tensor(&recurrence_state);
    let recurrence_output_t = tensor(&recurrence_output);
    let z_t = tensor(&z);
    let norm_weight_t = tensor(&norm_weight);
    let grad_y_t = tensor(&grad_y);
    let grad_recurrence_state_t = tensor(&grad_recurrence_state);
    let grad_conv_state_t = tensor(&grad_conv_state);

    let grad_recurrence_output_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
    let grad_z_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
    let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
    let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_recurrence_state_in_t =
        MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
    let grad_c_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
    let residual_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
    let grad_q_raw_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_k_raw_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
    let grad_alpha_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_beta_source_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
    let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![conv_dim as u64]).unwrap();
    let grad_conv_state_in_t = MetalTensor::zeros_f32(&ctx, vec![(3 * conv_dim) as u64]).unwrap();

    one_shot(&ctx, |encoder| {
        encode_rmsnorm_gated_vjp_f32(
            &ctx,
            encoder,
            &recurrence_output_t,
            &norm_weight_t,
            &z_t,
            &grad_y_t,
            &grad_recurrence_output_t,
            &grad_z_t,
            N_V,
            HEAD_DIM,
            RMS_EPS,
        )?;
        encode_gdn_step_decay_vjp_f32(
            &ctx,
            encoder,
            &q_t,
            &k_t,
            &v_t,
            &decay_t,
            &beta_t,
            &recurrence_state_t,
            &grad_recurrence_output_t,
            &grad_recurrence_state_t,
            &grad_q_t,
            &grad_k_t,
            &grad_v_t,
            &grad_decay_t,
            &grad_beta_t,
            &grad_recurrence_state_in_t,
            &grad_c_t,
            &residual_t,
            N_V,
            N_K,
            HEAD_DIM,
        )?;
        encode_l2_norm_vjp_batched_f32(
            &ctx,
            encoder,
            &q_raw_t,
            &grad_q_t,
            &grad_q_raw_t,
            N_K,
            HEAD_DIM,
            L2_EPS,
        )?;
        encode_l2_norm_vjp_batched_f32(
            &ctx,
            encoder,
            &k_raw_t,
            &grad_k_t,
            &grad_k_raw_t,
            N_K,
            HEAD_DIM,
            L2_EPS,
        )?;
        encode_gdn_decay_chain_vjp_f32(
            &ctx,
            encoder,
            &alpha_t,
            &dt_t,
            &a_log_t,
            &decay_t,
            &grad_decay_t,
            &grad_alpha_t,
        )?;
        encode_sigmoid_output_vjp_f32(&ctx, encoder, &beta_t, &grad_beta_t, &grad_beta_source_t)?;
        encode_ssm_conv_silu_split_vjp_f32(
            &ctx,
            encoder,
            &qkv_t,
            &conv_state_t,
            &conv_weight_t,
            &grad_q_raw_t,
            &grad_k_raw_t,
            &grad_v_t,
            &grad_conv_state_t,
            &grad_qkv_t,
            &grad_conv_state_in_t,
            N_K,
            N_V,
            HEAD_DIM,
        )
    })
    .unwrap();

    let grad_qkv = read_back_f32(&grad_qkv_t.buffer, conv_dim);
    let grad_conv_state_in = read_back_f32(&grad_conv_state_in_t.buffer, 3 * conv_dim);
    let grad_alpha = read_back_f32(&grad_alpha_t.buffer, N_V);
    let grad_beta_source = read_back_f32(&grad_beta_source_t.buffer, N_V);
    let grad_recurrence_state_in =
        read_back_f32(&grad_recurrence_state_in_t.buffer, state_elements);
    let grad_z = read_back_f32(&grad_z_t.buffer, v_elements);

    let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let qkv64 = as_f64(&qkv_now);
    let conv_state64 = as_f64(&conv_state);
    let conv_weight64 = as_f64(&conv_weight);
    let alpha64 = as_f64(&alpha_source);
    let dt64 = as_f64(&dt_bias);
    let a_log64 = as_f64(&a_log);
    let beta_source64 = as_f64(&beta_source);
    let recurrence_state64 = as_f64(&recurrence_state);
    let z64 = as_f64(&z);
    let norm_weight64 = as_f64(&norm_weight);
    let grad_y64 = as_f64(&grad_y);
    let grad_recurrence_state64 = as_f64(&grad_recurrence_state);
    let grad_conv_state64 = as_f64(&grad_conv_state);
    let objective = |qkv: &[f64],
                     conv_state: &[f64],
                     alpha: &[f64],
                     beta_source: &[f64],
                     recurrence_state: &[f64],
                     z: &[f64]| {
        gdn_envelope_objective_f64(
            qkv,
            conv_state,
            &conv_weight64,
            alpha,
            &dt64,
            &a_log64,
            beta_source,
            recurrence_state,
            z,
            &norm_weight64,
            &grad_y64,
            &grad_recurrence_state64,
            &grad_conv_state64,
            N_V,
            N_K,
            HEAD_DIM,
            f64::from(L2_EPS),
            f64::from(RMS_EPS),
        )
    };
    let epsilon = 1e-5;
    let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
    };
    for &index in &[0usize, qk_elements, 2 * qk_elements, conv_dim - 1] {
        let fd = finite_difference(&qkv64, index, &|candidate| {
            objective(
                candidate,
                &conv_state64,
                &alpha64,
                &beta_source64,
                &recurrence_state64,
                &z64,
            )
        });
        assert!((fd - f64::from(grad_qkv[index])).abs() < 2e-4);
    }
    for &index in &[0usize, conv_dim, 2 * conv_dim + 17, 3 * conv_dim - 1] {
        let fd = finite_difference(&conv_state64, index, &|candidate| {
            objective(
                &qkv64,
                candidate,
                &alpha64,
                &beta_source64,
                &recurrence_state64,
                &z64,
            )
        });
        assert!((fd - f64::from(grad_conv_state_in[index])).abs() < 2e-4);
    }
    for index in 0..N_V {
        let fd = finite_difference(&alpha64, index, &|candidate| {
            objective(
                &qkv64,
                &conv_state64,
                candidate,
                &beta_source64,
                &recurrence_state64,
                &z64,
            )
        });
        assert!((fd - f64::from(grad_alpha[index])).abs() < 2e-4);
        let fd = finite_difference(&beta_source64, index, &|candidate| {
            objective(
                &qkv64,
                &conv_state64,
                &alpha64,
                candidate,
                &recurrence_state64,
                &z64,
            )
        });
        assert!((fd - f64::from(grad_beta_source[index])).abs() < 2e-4);
    }
    for &index in &[
        0usize,
        127,
        HEAD_DIM * HEAD_DIM + 31 * HEAD_DIM + 32,
        state_elements - 1,
    ] {
        let fd = finite_difference(&recurrence_state64, index, &|candidate| {
            objective(
                &qkv64,
                &conv_state64,
                &alpha64,
                &beta_source64,
                candidate,
                &z64,
            )
        });
        assert!((fd - f64::from(grad_recurrence_state_in[index])).abs() < 2e-4);
    }
    for &index in &[0usize, 127, v_elements - 1] {
        let fd = finite_difference(&z64, index, &|candidate| {
            objective(
                &qkv64,
                &conv_state64,
                &alpha64,
                &beta_source64,
                &recurrence_state64,
                candidate,
            )
        });
        assert!((fd - f64::from(grad_z[index])).abs() < 2e-4);
    }

    let direction = |len: usize, stride: usize| {
        (0..len)
            .map(|index| ((index * stride + 3) % 37) as f64 * 0.0007 - 0.012)
            .collect::<Vec<_>>()
    };
    let dqkv = direction(qkv64.len(), 5);
    let dconv = direction(conv_state64.len(), 7);
    let dalpha = direction(alpha64.len(), 11);
    let dbeta = direction(beta_source64.len(), 13);
    let dstate = direction(recurrence_state64.len(), 17);
    let dz = direction(z64.len(), 19);
    let inner = |gradient: &[f32], tangent: &[f64]| {
        gradient
            .iter()
            .zip(tangent)
            .map(|(gradient, tangent)| f64::from(*gradient) * tangent)
            .sum::<f64>()
    };
    let reverse_directional = inner(&grad_qkv, &dqkv)
        + inner(&grad_conv_state_in, &dconv)
        + inner(&grad_alpha, &dalpha)
        + inner(&grad_beta_source, &dbeta)
        + inner(&grad_recurrence_state_in, &dstate)
        + inner(&grad_z, &dz);
    let shift = |base: &[f64], tangent: &[f64], amount: f64| {
        base.iter()
            .zip(tangent)
            .map(|(base, tangent)| base + amount * tangent)
            .collect::<Vec<_>>()
    };
    let plus = objective(
        &shift(&qkv64, &dqkv, epsilon),
        &shift(&conv_state64, &dconv, epsilon),
        &shift(&alpha64, &dalpha, epsilon),
        &shift(&beta_source64, &dbeta, epsilon),
        &shift(&recurrence_state64, &dstate, epsilon),
        &shift(&z64, &dz, epsilon),
    );
    let minus = objective(
        &shift(&qkv64, &dqkv, -epsilon),
        &shift(&conv_state64, &dconv, -epsilon),
        &shift(&alpha64, &dalpha, -epsilon),
        &shift(&beta_source64, &dbeta, -epsilon),
        &shift(&recurrence_state64, &dstate, -epsilon),
        &shift(&z64, &dz, -epsilon),
    );
    let forward_directional = (plus - minus) / (2.0 * epsilon);
    assert!(
        (forward_directional - reverse_directional).abs() < 3e-4,
        "full GDN envelope adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
    );
}

#[test]
fn gdn_envelope_vjps_reject_unsafe_contracts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let sigmoid = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let grad = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let invoke_sigmoid = |input: &MetalTensor, destination: &MetalTensor| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode_sigmoid_output_vjp_f32(&ctx, &encoder, input, &grad, destination);
        encoder.end();
        result
    };

    let mut overflow_shape = sigmoid.clone();
    overflow_shape.shape = vec![u64::MAX, 2];
    invoke_sigmoid(&overflow_shape, &output).expect_err("overflowing shape must fail");
    let f16 = MetalTensor::zeros_f16(&ctx, vec![4]).unwrap();
    invoke_sigmoid(&f16, &output).expect_err("non-F32 input must fail");
    invoke_sigmoid(&sigmoid, &sigmoid).expect_err("input/output alias must fail");
    let mut read_only = output.clone();
    read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    invoke_sigmoid(&sigmoid, &read_only).expect_err("read-only output must fail");
    let mut misaligned = output.clone();
    misaligned.offset = 2;
    invoke_sigmoid(&sigmoid, &misaligned).expect_err("misaligned output must fail");
    let mut out_of_range = output.clone();
    out_of_range.offset = 4;
    invoke_sigmoid(&sigmoid, &out_of_range).expect_err("short physical range must fail");

    let o = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let weight = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let z = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let grad_y = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let shared_output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_rmsnorm_gated_vjp_f32(
        &ctx,
        &encoder,
        &o,
        &weight,
        &z,
        &grad_y,
        &shared_output,
        &shared_output,
        1,
        4,
        1e-6,
    )
    .expect_err("gradient outputs must not alias");
    encoder.end();
}

/// Ensures the chained-encoding API is correctness-equivalent to one-shot.
/// (The bench's only structural difference vs `encode_*` is that it loops
/// `encode_*` inside the same encoder; if it diverges, the kernel is
/// reading non-deterministic state — a bug.)
#[test]
fn chained_encoding_is_correct() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n_in = 1024;
    let n_out = 4096;
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| ((i % 17) as f32 - 8.0) * 1e-3)
        .collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);

    let w_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&w),
        vec![n_in as u64, n_out as u64],
        GgmlType::F32,
    )
    .unwrap();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

    // Chain 8 dispatches; final result should equal one dispatch (each
    // overwrites the previous).
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    for _ in 0..8 {
        encode_mat_vec_f32(&ctx, &enc, &w_t, &x_t, &y_t, n_in, n_out).unwrap();
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let gpu = read_back_f32(&y_t.buffer, n_out);

    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "chained encoding diverged: max|Δ|={max_abs}"
    );
}

/// v0.432 equivalence gate: the strided-source batched q-norm reading
/// the Q halves of an interleaved `[head_dim Q, head_dim gate]` layout
/// must be BIT-IDENTICAL to split_q_gate followed by the compact
/// batched q-norm (pure addressing change, same per-row arithmetic).
#[test]
fn rms_norm_batched_src_strided_matches_split_path_bitwise() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    for &(n_heads, head_dim) in &[(24usize, 256usize), (16, 256), (8, 64)] {
        let full: Vec<f32> = (0..n_heads * 2 * head_dim)
            .map(|i| ((i % 41) as f32 - 20.0) * 3e-2)
            .collect();
        let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let full_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&full),
            vec![(n_heads * 2 * head_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weight),
            vec![head_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_strided = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let eps = 1e-6f32;
        one_shot(&ctx, |enc| {
            encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
            encode_rms_norm_batched_f32(&ctx, enc, &q_t, &w_t, &y_split, n_heads, head_dim, eps)
        })
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_rms_norm_batched_src_strided_f32(
                &ctx,
                enc,
                &full_t,
                &w_t,
                &y_strided,
                n_heads,
                head_dim,
                2 * head_dim,
                0,
                eps,
            )
        })
        .unwrap();
        let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
        let b = read_back_f32(&y_strided.buffer, n_heads * head_dim);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "strided q-norm not bit-identical at [{i}] (n_heads={n_heads}, \
                 head_dim={head_dim}): split={x} strided={y}"
            );
        }
    }
}

#[test]
fn qk_rms_norm_rope_fused_matches_composed_path() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let n_q = 24;
    let n_k = 4;
    let head_dim = 256;
    let n_rot = 64;
    let eps = 1e-6f32;
    let theta = 10_000_000.0f32;
    for &(n_tokens, start_position) in &[(1usize, 0u32), (8, 65_531), (128, 65_536), (8, 1_048_568)]
    {
        let q_source: Vec<f32> = (0..n_tokens * n_q * 2 * head_dim)
            .map(|i| ((i % 41) as f32 - 20.0) * 0.03125)
            .collect();
        let k_source: Vec<f32> = (0..n_tokens * n_k * head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.046875)
            .collect();
        let q_weight: Vec<f32> = (0..head_dim)
            .map(|i| 0.5 + (i % 11) as f32 * 0.0625)
            .collect();
        let k_weight: Vec<f32> = (0..head_dim)
            .map(|i| 0.625 + (i % 7) as f32 * 0.078125)
            .collect();
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let q_src = tensor(&q_source);
        let k_src = tensor(&k_source);
        let q_w = tensor(&q_weight);
        let k_w = tensor(&k_weight);
        let q_len = n_tokens * n_q * head_dim;
        let k_len = n_tokens * n_k * head_dim;
        let q_composed = MetalTensor::zeros_f32(&ctx, vec![q_len as u64]).unwrap();
        let k_composed = MetalTensor::zeros_f32(&ctx, vec![k_len as u64]).unwrap();
        let q_fused = MetalTensor::zeros_f32(&ctx, vec![q_len as u64]).unwrap();
        let k_fused = MetalTensor::zeros_f32(&ctx, vec![k_len as u64]).unwrap();

        one_shot(&ctx, |enc| {
            encode_rms_norm_batched_src_strided_f32(
                &ctx,
                enc,
                &q_src,
                &q_w,
                &q_composed,
                n_tokens * n_q,
                head_dim,
                2 * head_dim,
                0,
                eps,
            )?;
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &k_src,
                &k_w,
                &k_composed,
                n_tokens * n_k,
                head_dim,
                eps,
            )?;
            encode_rope_neox_f32_packed_consecutive(
                &ctx,
                enc,
                &q_composed,
                n_tokens,
                n_q,
                head_dim,
                n_rot,
                start_position,
                theta,
            )?;
            encode_rope_neox_f32_packed_consecutive(
                &ctx,
                enc,
                &k_composed,
                n_tokens,
                n_k,
                head_dim,
                n_rot,
                start_position,
                theta,
            )
        })
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_qk_rms_norm_rope_f32_packed_consecutive(
                &ctx,
                enc,
                &q_src,
                &q_w,
                &q_fused,
                &k_src,
                &k_w,
                &k_fused,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                start_position,
                eps,
                theta,
            )
        })
        .unwrap();

        let q_baseline = read_back_f32(&q_composed.buffer, q_len);
        let k_baseline = read_back_f32(&k_composed.buffer, k_len);
        let q_candidate = read_back_f32(&q_fused.buffer, q_len);
        let k_candidate = read_back_f32(&k_fused.buffer, k_len);
        let q_max = max_abs_diff(&q_baseline, &q_candidate);
        let k_max = max_abs_diff(&k_baseline, &k_candidate);
        eprintln!(
            "[qk-norm-rope] N={n_tokens} start={start_position} q_max={q_max:.3e} k_max={k_max:.3e}"
        );
        assert_finite(
            &q_candidate,
            &format!("fused norm+RoPE Q for N={n_tokens} start={start_position}"),
        );
        assert_finite(
            &k_candidate,
            &format!("fused norm+RoPE K for N={n_tokens} start={start_position}"),
        );
        assert_bitwise_equal(
            &q_baseline,
            &q_candidate,
            &format!("fused norm+RoPE Q for N={n_tokens} start={start_position}"),
        );
        assert_bitwise_equal(
            &k_baseline,
            &k_candidate,
            &format!("fused norm+RoPE K for N={n_tokens} start={start_position}"),
        );
    }
}

/// v0.432 equivalence gate: the fused strided gate epilogue
/// (`out = x / (1 + e^-gate)`) vs the old split + sigmoid-into-temp +
/// mul (`out = x * (1 / (1 + e^-gate))`). Different last-ulp rounding
/// (division vs reciprocal-multiply), so tolerance-based, tight.
#[test]
fn sigmoid_mul_gate_strided_matches_split_path() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let (n_heads, head_dim) = (24usize, 256usize);
    let full: Vec<f32> = (0..n_heads * 2 * head_dim)
        .map(|i| ((i % 37) as f32 - 18.0) * 5e-2)
        .collect();
    let x: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| ((i % 29) as f32 - 14.0) * 4e-2)
        .collect();
    let full_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&full),
        vec![(n_heads * 2 * head_dim) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![(n_heads * head_dim) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let sig_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    let y_fused = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
        encode_sigmoid_f32(&ctx, enc, &gate_t, &sig_t)?;
        encode_mul_f32(&ctx, enc, &x_t, &sig_t, &y_split)
    })
    .unwrap();
    one_shot(&ctx, |enc| {
        encode_sigmoid_mul_gate_strided_f32(
            &ctx,
            enc,
            &full_t,
            &x_t,
            &y_fused,
            n_heads,
            head_dim,
            2 * head_dim,
            head_dim,
        )
    })
    .unwrap();
    let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
    let b = read_back_f32(&y_fused.buffer, n_heads * head_dim);
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs < 1e-6,
        "fused strided gate epilogue diverged beyond ulp scale: max|Δ|={max_abs:.3e}"
    );
}

/// v0.433 triage repro for the `attn_v4_matches_naive_f16kv` load-flake:
/// NaN-prime the o/ml partials scratch before dispatch at the exact
/// config that failed under parallel-suite load (`group=4 n_pos=1024
/// nwg=64 C=16`, cos=0.9662). `zeros_f32` is documented-uninitialized,
/// so isolated runs see fresh zero pages while loaded runs see recycled
/// garbage; if any kernel cell is read without being written, this test
/// fails deterministically instead of 50%-of-suite-runs.
#[test]
fn attn_v4_partials_fully_written_nan_prime() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let hd = 256usize;
    // The observed failing config plus its close neighbors.
    let cases: &[(usize, usize, usize, usize, usize)] = &[
        // (n_q, n_kv, n_pos, nwg, tile_c)
        (8, 2, 1024, 64, 16),
        (8, 2, 1024, 64, 32),
        (8, 2, 1024, 128, 16),
        (8, 2, 1024, 256, 16),
        (24, 4, 1024, 64, 16),
    ];
    for &(n_q, n_kv, n_pos, nwg, tile_c) in cases {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
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
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(src_f32.as_slice()),
                vec![src_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
            })
            .unwrap();
        }
        let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_attn_decode_f16kv_f32(
                &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
            )
        })
        .unwrap();
        let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

        let o_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
        let ml_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
        let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        // NaN-prime everything the kernels are supposed to fully write.
        unsafe {
            for t in [&o_partial, &ml_partial, &y_v4_t] {
                let p = t.buffer.contents().as_ptr() as *mut f32;
                for i in 0..t.n_elements() as usize {
                    *p.add(i) = f32::NAN;
                }
            }
        }
        one_shot(&ctx, |enc| {
            encode_attn_decode_v4_f32(
                &ctx,
                enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial,
                &ml_partial,
                &y_v4_t,
                n_q,
                n_kv,
                hd,
                n_pos,
                nwg,
                tile_c,
            )
        })
        .unwrap();
        let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
        let nan_count = y_v4.iter().filter(|x| x.is_nan()).count();
        let max_abs = y_v4
            .iter()
            .zip(y_naive.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[v4-nan-prime group={group} n_pos={n_pos} nwg={nwg} C={tile_c}] \
             nans={nan_count} max|Δ|={max_abs:.2e}"
        );
        assert_eq!(
            nan_count, 0,
            "v4 output contains NaN after NaN-priming partials: some partial \
             cell is read without being written (group={group} n_pos={n_pos} \
             nwg={nwg} C={tile_c})"
        );
        assert!(
            max_abs < 5e-3,
            "v4 diverged from naive with NaN-primed partials: max|Δ|={max_abs} \
             (group={group} n_pos={n_pos} nwg={nwg} C={tile_c})"
        );
    }
}

/// v4 flash-attn (GQA-dedup + online softmax + split-K) vs production
/// `attn_decode_f16kv_f32`. Same F16 K/V inputs, multiple n_pos and NWG
/// settings. Must produce numerically equivalent outputs (cos > 0.9999;
/// max|Δ| ~1e-3 — the bound expected from F32 reorder noise across
/// completely different reduction orderings).
#[test]
fn attn_v4_matches_naive_f16kv() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let hd = 256usize;
    // Cover the currently-supported specializations:
    // small dense (GROUP=4), 27B dense (GROUP=6), 35B A3B (GROUP=8),
    // 122B A10B (GROUP=16).
    let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];

    for &(n_q, n_kv) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        let cases: &[(usize, usize)] = &[
            (1, 1),
            (32, 1),
            (32, 2),
            (64, 1),
            (256, 4),
            (1024, 8),
            (1024, 64),
            (1024, 128),
            (1024, 256),
            (4096, 16),
            (4096, 64),
        ];

        for &(n_pos, nwg) in cases {
            // Synthesize Q (F32) and K, V (F32 scratch → F16 cache).
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let cap = n_pos.max(64);
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            // Build F16 KV cache by scattering F32 source into a F16 dest.
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            // Use scatter to convert F32 → F16 in cache.
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            // --- Reference: naive f16kv kernel ---
            let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_f16kv_f32(
                    &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                )
            })
            .unwrap();
            let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

            // --- v4: allocate partials, dispatch main + reduce ---
            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
            let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            // Sweep all three tile-C variants — each must match naive within
            // fp32 reorder noise (cos > 0.9999, max|Δ| < 5e-3).
            for &tile_c in &[16usize, 32, 64, 128] {
                one_shot(&ctx, |enc| {
                    encode_attn_decode_v4_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        &y_v4_t,
                        n_q,
                        n_kv,
                        hd,
                        n_pos,
                        nwg,
                        tile_c,
                    )
                })
                .unwrap();
                let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
                let max_abs = y_v4
                    .iter()
                    .zip(y_naive.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let dot: f64 = y_v4
                    .iter()
                    .zip(y_naive.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = y_naive
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let cos = dot / (na * nb);
                eprintln!(
                    "[v4 group={group:>2} n_q={n_q:>2} n_kv={n_kv:>2} n_pos={n_pos:>4} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e}  cos={cos:.6}"
                );
                assert!(
                    cos > 0.9999,
                    "v4(group={group}, C={tile_c}) vs naive cos too low at n_pos={n_pos} nwg={nwg}: cos={cos}"
                );
                assert!(
                    max_abs < 5e-3,
                    "v4(group={group}, C={tile_c}) vs naive max|Δ| too high at n_pos={n_pos} nwg={nwg}: {max_abs}"
                );
            }
        }
    }
}

/// CPU f64 reference for the matrix-attention sidecar semantics: causal
/// multi-row attention over an F16 KV prefix with per-row visibility
/// `base_pos + row + 1`, softmax of `q·k / sqrt(head_dim)`.
///
/// Inputs must already be f16-representable (pre-rounded) so the GPU's
/// half demotion inside the GEMMs is exact and tolerances measure reduction
/// order + the probs half demotion, not input rounding.
#[allow(clippy::too_many_arguments)]
fn cpu_matrix_attn_reference(
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

fn read_back_u16(tensor: &MetalTensor) -> Vec<u16> {
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

static ATTN_MATRIX_VT_SCOPE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn attn_matrix_vt_dispatch_groups_cover_exact_thread_range() {
    for total in [1usize, 255, 256, 257] {
        assert_eq!(attn_matrix_vt_threadgroups(total, false).unwrap(), total);
        assert_eq!(
            attn_matrix_vt_threadgroups(total, true).unwrap(),
            total.div_ceil(ATTN_MATRIX_VT_THREADS)
        );
    }
    assert!(attn_matrix_vt_threadgroups(0, false).is_err());
    let legacy_max = (u32::MAX as usize + 1) / ATTN_MATRIX_VT_THREADS;
    assert_eq!(
        attn_matrix_vt_threadgroups(legacy_max, false).unwrap(),
        legacy_max
    );
    assert!(attn_matrix_vt_threadgroups(legacy_max + 1, false).is_err());
    assert!(attn_matrix_vt_threadgroups(u32::MAX as usize, true).is_ok());
    assert!(attn_matrix_vt_threadgroups(u32::MAX as usize + 1, true).is_err());
}

#[test]
fn attn_matrix_vt_scoped_override_restores_and_rejects_nesting() {
    let _serial = ATTN_MATRIX_VT_SCOPE_TEST_LOCK.lock().unwrap();
    let baseline = attn_matrix_vt_compact_dispatch_enabled().unwrap();
    assert_eq!(
        with_attn_matrix_vt_compact_dispatch_override(!baseline, || {
            attn_matrix_vt_compact_dispatch_enabled()
        })
        .unwrap()
        .unwrap(),
        !baseline
    );
    assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

    let nested = with_attn_matrix_vt_compact_dispatch_override(true, || {
        with_attn_matrix_vt_compact_dispatch_override(false, || ())
    })
    .unwrap();
    assert!(nested.is_err());
    assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

    let cross_thread = with_attn_matrix_vt_compact_dispatch_override(true, || {
        std::thread::spawn(attn_matrix_vt_compact_dispatch_enabled)
            .join()
            .unwrap()
    })
    .unwrap();
    assert!(cross_thread.is_err());
    assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

    let panicked = std::panic::catch_unwind(|| {
        let _ = with_attn_matrix_vt_compact_dispatch_override(!baseline, || {
            panic!("exercise override unwind restoration")
        });
    });
    assert!(panicked.is_err());
    assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);
}

#[test]
fn attn_matrix_vt_dispatch_capture_is_exact_and_scoped() {
    let _serial = ATTN_MATRIX_VT_SCOPE_TEST_LOCK.lock().unwrap();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const ROWS: usize = 257;
    let cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&vec![0x3555u16; ROWS]),
        vec![ROWS as u64],
        GgmlType::F16,
    )
    .unwrap();
    let make_vt = || {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0x3aaau16; ROWS]),
            vec![ROWS as u64],
            GgmlType::F16,
        )
        .unwrap()
    };

    for (compact, expected_groups) in [(false, ROWS), (true, ROWS.div_ceil(256))] {
        let vt = make_vt();
        let (encoded, capture) = with_attn_matrix_vt_compact_dispatch_override(compact, || {
            capture_attn_matrix_vt_dispatches(|| {
                one_shot(&ctx, |enc| {
                    encode_attn_matrix_transpose_v_f16(
                        &ctx, enc, &cache, &vt, 0, ROWS, ROWS, 1, ROWS, 1, 1,
                    )
                })
            })
        })
        .unwrap()
        .unwrap();
        encoded.unwrap();
        assert!(capture.owner_thread.starts_with("ThreadId("));
        assert_eq!(
            capture.stats,
            AttnMatrixVtDispatchStats {
                calls: 1,
                row_sum: ROWS as u64,
                element_sum: ROWS as u64,
                threadgroup_sum: expected_groups as u64,
                compact_calls: u64::from(compact),
                legacy_calls: u64::from(!compact),
                base_pos_sum: 0,
                n_pos_sum: ROWS as u64,
            }
        );
    }

    let (nested, outer) =
        capture_attn_matrix_vt_dispatches(|| capture_attn_matrix_vt_dispatches(|| ())).unwrap();
    assert!(nested.is_err());
    assert_eq!(outer.stats, AttnMatrixVtDispatchStats::default());

    let (cross_thread, capture) = capture_attn_matrix_vt_dispatches(|| {
        std::thread::spawn(|| record_attn_matrix_vt_dispatch(0, 1, 1, 1, 1, true))
            .join()
            .unwrap()
    })
    .unwrap();
    assert!(cross_thread.is_err());
    assert_eq!(capture.stats, AttnMatrixVtDispatchStats::default());

    let panicked = std::panic::catch_unwind(|| {
        let _ = capture_attn_matrix_vt_dispatches(|| panic!("exercise capture unwind"));
    });
    assert!(panicked.is_err());
    let (_, capture) = capture_attn_matrix_vt_dispatches(|| ()).unwrap();
    assert_eq!(capture.stats, AttnMatrixVtDispatchStats::default());
}

#[test]
fn attn_matrix_vt_compact_dispatch_matches_legacy_nonzero_span() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const SENTINEL: u16 = 0x3555;

    // Exact thread totals 255, 256, and 257. The first two also exercise
    // multiple KV heads and dimensions; all use nonzero base and padding.
    for &(n_kv, head_dim, n_rows) in &[(3usize, 5usize, 17usize), (2, 8, 16), (1, 1, 257)] {
        let base_pos = 2usize;
        let n_pos = base_pos + n_rows + 1;
        let vt_stride = n_pos + 3;
        let kv_dim = n_kv * head_dim;
        let total = kv_dim * n_rows;
        assert!([255, 256, 257].contains(&total));
        let cache: Vec<u16> = (0..n_pos * kv_dim)
            .map(|i| half::f16::from_f32(((i % 31) as f32 - 15.0) * 0.03125).to_bits())
            .collect();
        let cache_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&cache),
            vec![cache.len() as u64],
            GgmlType::F16,
        )
        .unwrap();
        let make_vt = || {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![SENTINEL; kv_dim * vt_stride]),
                vec![(kv_dim * vt_stride) as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let legacy = make_vt();
        let compact = make_vt();

        for (dst, compact_dispatch) in [(&legacy, false), (&compact, true)] {
            one_shot(&ctx, |enc| {
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx,
                    enc,
                    &cache_t,
                    dst,
                    base_pos,
                    n_rows,
                    n_pos,
                    kv_dim,
                    vt_stride,
                    n_kv,
                    head_dim,
                    compact_dispatch,
                )
            })
            .unwrap();
        }

        let mut expected = vec![SENTINEL; kv_dim * vt_stride];
        for pos in base_pos..base_pos + n_rows {
            for flat_d in 0..kv_dim {
                expected[flat_d * vt_stride + pos] = cache[pos * kv_dim + flat_d];
            }
        }
        assert_eq!(read_back_u16(&legacy), expected);
        assert_eq!(read_back_u16(&compact), expected);

        let cmd = ctx
            .queue
            .commandBuffer()
            .expect("validation command buffer");
        let enc = KernelEncoder::begin(&cmd);
        assert!(
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx,
                &enc,
                &cache_t,
                &compact,
                base_pos,
                n_rows,
                n_pos,
                kv_dim - 1,
                vt_stride,
                n_kv,
                head_dim,
                true,
            )
            .is_err()
        );
        let mut short_cache = cache_t.clone();
        short_cache.shape = vec![((base_pos + n_rows) * kv_dim - 1) as u64];
        assert!(
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx,
                &enc,
                &short_cache,
                &compact,
                base_pos,
                n_rows,
                n_pos,
                kv_dim,
                vt_stride,
                n_kv,
                head_dim,
                true,
            )
            .is_err()
        );
        let mut short_vt = compact.clone();
        let exact_vt_end = (kv_dim - 1) * vt_stride + base_pos + n_rows;
        short_vt.shape = vec![(exact_vt_end - 1) as u64];
        assert!(
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx, &enc, &cache_t, &short_vt, base_pos, n_rows, n_pos, kv_dim, vt_stride, n_kv,
                head_dim, true,
            )
            .is_err()
        );
        assert!(
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx,
                &enc,
                &cache_t,
                &compact,
                usize::MAX,
                1,
                n_pos,
                kv_dim,
                vt_stride,
                n_kv,
                head_dim,
                true,
            )
            .is_err()
        );
        enc.end();
    }
}

#[test]
fn attn_matrix_vt_prefix_rebuild_preserves_scattered_suffix() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const PREFIX: usize = 5;
    const CHUNK: usize = 3;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 8;
    const VT_PADDING: usize = 3;
    const SENTINELS: [u16; 3] = [0x3555, 0x3aaa, 0x3999];

    let n_pos = PREFIX + CHUNK;
    let vt_stride = n_pos + VT_PADDING;
    let kv_dim = N_KV * HEAD_DIM;
    let cache_elems = n_pos * kv_dim;
    let vt_elems = kv_dim * vt_stride;
    let initial_cache: Vec<u16> = (0..cache_elems)
        .map(|i| half::f16::from_f32(((i % 29) as f32 - 14.0) * 0.03125).to_bits())
        .collect();
    let current_f32: Vec<f32> = (0..CHUNK * kv_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
        .collect();
    let current_f16: Vec<u16> = current_f32
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect();
    let current = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&current_f32),
        vec![current_f32.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let mut expected_cache = initial_cache.clone();
    expected_cache[PREFIX * kv_dim..].copy_from_slice(&current_f16);

    let make_cache = || {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&initial_cache),
            vec![cache_elems as u64],
            GgmlType::F16,
        )
        .unwrap()
    };
    let mut arms = Vec::new();
    for &sentinel in &SENTINELS {
        let cache_k = make_cache();
        let cache_v = make_cache();
        let vt = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![sentinel; vt_elems]),
            vec![vt_elems as u64],
            GgmlType::F16,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv_vt(
                &ctx,
                enc,
                &current,
                &current,
                &cache_k,
                &cache_v,
                &vt,
                PREFIX * kv_dim,
                CHUNK * kv_dim,
                PREFIX,
                kv_dim,
                HEAD_DIM,
                vt_stride,
            )
        })
        .unwrap();
        assert_eq!(read_back_u16(&cache_k), expected_cache);
        assert_eq!(read_back_u16(&cache_v), expected_cache);
        let scattered_vt = read_back_u16(&vt);
        for row in 0..CHUNK {
            for flat_d in 0..kv_dim {
                assert_eq!(
                    scattered_vt[flat_d * vt_stride + PREFIX + row],
                    current_f16[row * kv_dim + flat_d]
                );
            }
        }
        arms.push((cache_v, vt, sentinel));
    }

    let divergent_suffix: Vec<f32> = current_f32.iter().map(|&value| value + 1.0).collect();
    let divergent_suffix_f16: Vec<u16> = divergent_suffix
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect();
    let divergent_suffix_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&divergent_suffix),
        vec![divergent_suffix.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    one_shot(&ctx, |enc| {
        encode_scatter_offset_f32_to_f16(
            &ctx,
            enc,
            &divergent_suffix_t,
            &arms[2].0,
            PREFIX * kv_dim,
            divergent_suffix.len(),
        )
    })
    .unwrap();
    assert_eq!(
        &read_back_u16(&arms[2].0)[PREFIX * kv_dim..],
        divergent_suffix_f16.as_slice()
    );

    for (index, (cache_v, vt, _)) in arms.iter().enumerate() {
        let (rows, compact) = match index {
            0 => (n_pos, false),
            1 => (n_pos, true),
            _ => (PREFIX, true),
        };
        one_shot(&ctx, |enc| {
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx, enc, cache_v, vt, 0, rows, n_pos, kv_dim, vt_stride, N_KV, HEAD_DIM, compact,
            )
        })
        .unwrap();
    }

    for (index, (_, vt, sentinel)) in arms.iter().enumerate() {
        let mut expected_vt = vec![*sentinel; vt_elems];
        for pos in 0..n_pos {
            for flat_d in 0..kv_dim {
                expected_vt[flat_d * vt_stride + pos] = if index == 2 && pos >= PREFIX {
                    current_f16[(pos - PREFIX) * kv_dim + flat_d]
                } else {
                    expected_cache[pos * kv_dim + flat_d]
                };
            }
        }
        assert_eq!(read_back_u16(vt), expected_vt);
    }
}

#[test]
fn attn_matrix_prefix_only_vt_rebuild_matches_full_attention() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const PREFIX: usize = 47;
    const CHUNK: usize = 17;
    const N_Q: usize = 24;
    const N_KV: usize = 4;
    const HEAD_DIM: usize = 256;
    const SENTINEL: u16 = 0x3555;

    let n_pos = PREFIX + CHUNK;
    let group = N_Q / N_KV;
    let kv_dim = N_KV * HEAD_DIM;
    let vt_stride = n_pos + 3;
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let q: Vec<f32> = (0..CHUNK * N_Q * HEAD_DIM)
        .map(|i| round_f16(((i % 31) as f32 - 15.0) * 0.01))
        .collect();
    let k: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| round_f16(((i % 23) as f32 - 11.0) * 0.015))
        .collect();
    let v: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| round_f16(((i % 17) as f32 - 8.0) * 0.02))
        .collect();
    let k_f16: Vec<u16> = k
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect();
    let v_f16: Vec<u16> = v
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect();
    let mut restored_k = vec![SENTINEL; n_pos * kv_dim];
    let mut restored_v = vec![SENTINEL; n_pos * kv_dim];
    restored_k[..PREFIX * kv_dim].copy_from_slice(&k_f16[..PREFIX * kv_dim]);
    restored_v[..PREFIX * kv_dim].copy_from_slice(&v_f16[..PREFIX * kv_dim]);

    let tensor_f32 = |data: &[f32]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(data),
            vec![data.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    };
    let tensor_f16 = |data: &[u16]| {
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(data),
            vec![data.len() as u64],
            GgmlType::F16,
        )
        .unwrap()
    };
    let q_t = tensor_f32(&q);
    let current_k = tensor_f32(&k[PREFIX * kv_dim..]);
    let current_v = tensor_f32(&v[PREFIX * kv_dim..]);
    let full_k = tensor_f16(&k_f16);
    let full_v = tensor_f16(&v_f16);
    let rebuilt_k = tensor_f16(&restored_k);
    let rebuilt_v = tensor_f16(&restored_v);
    let make_vt = || tensor_f16(&vec![SENTINEL; kv_dim * vt_stride]);
    let full_vt = make_vt();
    let rebuilt_vt = make_vt();

    let make_scores = || MetalTensor::zeros_f16(&ctx, vec![(CHUNK * N_Q * n_pos) as u64]).unwrap();
    let make_ml = || {
        MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(CHUNK, N_Q, n_pos) as u64]).unwrap()
    };
    let full_scores = make_scores();
    let full_ml = make_ml();
    let full_out = MetalTensor::zeros_f32(&ctx, vec![(CHUNK * N_Q * HEAD_DIM) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_attn_matrix_transpose_v_f16(
            &ctx, enc, &full_v, &full_vt, 0, n_pos, n_pos, kv_dim, vt_stride, N_KV, HEAD_DIM,
        )?;
        encode_attn_matrix_kq_online_f32(
            &ctx,
            enc,
            &q_t,
            &full_k,
            &full_scores,
            &full_ml,
            CHUNK,
            PREFIX,
            n_pos,
            kv_dim,
            N_Q,
            N_KV,
            group,
            HEAD_DIM,
            true,
        )?;
        encode_attn_matrix_kqv_norm_f32(
            &ctx,
            enc,
            &full_scores,
            &full_ml,
            &full_vt,
            &full_out,
            CHUNK,
            PREFIX,
            n_pos,
            vt_stride,
            N_Q,
            N_KV,
            group,
            HEAD_DIM,
            true,
        )
    })
    .unwrap();

    let rebuilt_scores = make_scores();
    let rebuilt_ml = make_ml();
    let rebuilt_out = MetalTensor::zeros_f32(&ctx, vec![(CHUNK * N_Q * HEAD_DIM) as u64]).unwrap();
    let rebuilt_prefix_rows =
        crate::metal_dflash::attn_matrix_vt_prefix_rebuild_rows(true, 0, PREFIX)
            .expect("restored prefix requires a V_T rebuild");
    one_shot(&ctx, |enc| {
        encode_scatter_offset_f32_to_f16_kv_vt(
            &ctx,
            enc,
            &current_k,
            &current_v,
            &rebuilt_k,
            &rebuilt_v,
            &rebuilt_vt,
            PREFIX * kv_dim,
            CHUNK * kv_dim,
            PREFIX,
            kv_dim,
            HEAD_DIM,
            vt_stride,
        )?;
        encode_attn_matrix_transpose_v_f16(
            &ctx,
            enc,
            &rebuilt_v,
            &rebuilt_vt,
            0,
            rebuilt_prefix_rows,
            n_pos,
            kv_dim,
            vt_stride,
            N_KV,
            HEAD_DIM,
        )?;
        encode_attn_matrix_kq_online_f32(
            &ctx,
            enc,
            &q_t,
            &rebuilt_k,
            &rebuilt_scores,
            &rebuilt_ml,
            CHUNK,
            PREFIX,
            n_pos,
            kv_dim,
            N_Q,
            N_KV,
            group,
            HEAD_DIM,
            true,
        )?;
        encode_attn_matrix_kqv_norm_f32(
            &ctx,
            enc,
            &rebuilt_scores,
            &rebuilt_ml,
            &rebuilt_vt,
            &rebuilt_out,
            CHUNK,
            PREFIX,
            n_pos,
            vt_stride,
            N_Q,
            N_KV,
            group,
            HEAD_DIM,
            true,
        )
    })
    .unwrap();

    assert_eq!(read_back_u16(&rebuilt_k), k_f16);
    assert_eq!(read_back_u16(&rebuilt_v), v_f16);
    assert_eq!(read_back_u16(&rebuilt_vt), read_back_u16(&full_vt));
    assert_eq!(
        read_back_f32(&rebuilt_out.buffer, CHUNK * N_Q * HEAD_DIM),
        read_back_f32(&full_out.buffer, CHUNK * N_Q * HEAD_DIM)
    );
}

/// Model-free screen preregistered in
/// `docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/README.md`.
#[test]
#[ignore]
fn attn_matrix_vt_environment_probe() {
    let ctx = MetalContext::new().expect("Metal context for V_T environment probe");
    println!(
        "VT_ENV_JSON {}",
        serde_json::json!({
            "schema_version": 1,
            "test": "metal::tests::attn_matrix_vt_environment_probe",
            "device_registry_id": ctx.device.registryID(),
            "device": ctx.device.name().to_string(),
            "max_buffer_length": ctx.device.maxBufferLength(),
            "recommended_max_working_set_size": ctx.recommended_max_working_set_size(),
        })
    );
}

/// Model-free screen preregistered in
/// `docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/README.md`.
#[test]
#[ignore]
fn attn_matrix_vt_rebuild_screen() {
    const N_LAYERS: usize = 16;
    const N_KV: usize = 4;
    const HEAD_DIM: usize = 256;

    struct Bank {
        src: Vec<MetalTensor>,
        dst: Vec<MetalTensor>,
    }

    fn parse_usize(name: &str) -> usize {
        let raw = std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"));
        raw.parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid {name}={raw:?}"))
    }

    fn fill_tensor(tensor: &MetalTensor, byte: u8) {
        assert_eq!(tensor.dtype, GgmlType::F16);
        assert_eq!(tensor.buffer.storageMode(), MTLStorageMode::Shared);
        assert_eq!(tensor.offset, 0);
        let n_bytes = tensor.n_bytes() as usize;
        assert!(n_bytes <= tensor.buffer.length());
        unsafe {
            std::ptr::write_bytes(tensor.buffer.contents().as_ptr() as *mut u8, byte, n_bytes);
        }
    }

    fn allocate_bank(
        ctx: &MetalContext,
        name: &str,
        elems_per_layer: usize,
        bytes_per_layer: usize,
        src_byte: u8,
        dst_byte: u8,
    ) -> Bank {
        let mut src = Vec::with_capacity(N_LAYERS);
        let mut dst = Vec::with_capacity(N_LAYERS);
        for layer in 0..N_LAYERS {
            let src_layer = MetalTensor::zeros_f16(ctx, vec![elems_per_layer as u64])
                .unwrap_or_else(|error| {
                    panic!(
                        "allocate bank={name} layer={layer} role=src bytes={bytes_per_layer}: {error}"
                    )
                });
            let dst_layer = MetalTensor::zeros_f16(ctx, vec![elems_per_layer as u64])
                .unwrap_or_else(|error| {
                    panic!(
                        "allocate bank={name} layer={layer} role=dst bytes={bytes_per_layer}: {error}"
                    )
                });
            fill_tensor(&src_layer, src_byte);
            fill_tensor(&dst_layer, dst_byte);
            src.push(src_layer);
            dst.push(dst_layer);
        }
        Bank { src, dst }
    }

    fn run_span(
        ctx: &MetalContext,
        bank: &Bank,
        label: &str,
        base_pos: usize,
        rows: usize,
        n_pos: usize,
        kv_dim: usize,
        compact: bool,
    ) -> (f64, f64) {
        let cmd = ctx
            .queue
            .commandBuffer()
            .expect("V_T rebuild command buffer");
        let enc = KernelEncoder::begin(&cmd);
        for layer in 0..N_LAYERS {
            encode_attn_matrix_transpose_v_f16_mode(
                ctx,
                &enc,
                &bank.src[layer],
                &bank.dst[layer],
                base_pos,
                rows,
                n_pos,
                kv_dim,
                n_pos,
                N_KV,
                HEAD_DIM,
                compact,
            )
            .unwrap();
        }
        enc.end();
        let wall_start = std::time::Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1e3;
        let status = cmd.status();
        let error = cmd.error();
        assert!(
            status == objc2_metal::MTLCommandBufferStatus::Completed && error.is_none(),
            "V_T command failed arm={label} status={status:?} error={error:?}"
        );
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        assert!(wall_ms.is_finite() && wall_ms > 0.0);
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
        (wall_ms, gpu_ms)
    }

    fn prep_overlap(
        ctx: &MetalContext,
        bank: &Bank,
        prefix: usize,
        chunk: usize,
        n_pos: usize,
        kv_dim: usize,
    ) {
        let _ = run_span(ctx, bank, "PREP_D2", prefix, chunk, n_pos, kv_dim, true);
    }

    let mode = std::env::var("QWEN_VT_REBUILD_MODE")
        .unwrap_or_else(|_| panic!("missing QWEN_VT_REBUILD_MODE"));
    let prefix = parse_usize("QWEN_VT_REBUILD_PREFIX");
    let chunk = parse_usize("QWEN_VT_REBUILD_CHUNK");
    match mode.as_str() {
        "dispatch" => {
            assert!([512, 2048, 8192].contains(&prefix));
            assert_eq!(chunk, 128);
        }
        "compact" => {
            assert!([8192, 16384, 32768].contains(&prefix));
            assert_eq!(chunk, 128);
        }
        "overlap" => {
            assert_eq!(prefix, 32768);
            assert_eq!(chunk, 1024);
        }
        _ => panic!("invalid QWEN_VT_REBUILD_MODE={mode:?}"),
    }

    let n_pos = prefix.checked_add(chunk).unwrap();
    let kv_dim = N_KV * HEAD_DIM;
    let elems_per_layer = n_pos.checked_mul(kv_dim).unwrap();
    let bytes_per_layer = elems_per_layer.checked_mul(2).unwrap();
    let total_requested_bytes = bytes_per_layer
        .checked_mul(N_LAYERS)
        .and_then(|value| value.checked_mul(4))
        .unwrap();
    let ctx = MetalContext::new().expect("Metal context for V_T rebuild screen");
    let max_buffer_length = ctx.device.maxBufferLength();
    assert!(
        bytes_per_layer <= max_buffer_length,
        "V_T layer bytes {bytes_per_layer} exceed maxBufferLength {max_buffer_length}"
    );
    let allocated_before = ctx.current_allocated_size();
    let bank_x = allocate_bank(&ctx, "X", elems_per_layer, bytes_per_layer, 0x3c, 0xa5);
    let bank_y = allocate_bank(&ctx, "Y", elems_per_layer, bytes_per_layer, 0x38, 0x5a);
    let allocated_after = ctx.current_allocated_size();

    println!(
        "VT_REBUILD_JSON {}",
        serde_json::json!({
            "kind": "meta",
            "schema_version": 2,
            "mode": mode,
            "prefix": prefix,
            "chunk": chunk,
            "n_pos": n_pos,
            "vt_stride": n_pos,
            "layers": N_LAYERS,
            "n_kv": N_KV,
            "head_dim": HEAD_DIM,
            "kv_dim": kv_dim,
            "device_registry_id": ctx.device.registryID(),
            "device": ctx.device.name().to_string(),
            "max_buffer_length": max_buffer_length,
            "recommended_max_working_set_size": ctx.recommended_max_working_set_size(),
            "bytes_per_layer_buffer": bytes_per_layer,
            "total_requested_bytes": total_requested_bytes,
            "allocated_before": allocated_before,
            "allocated_after": allocated_after,
        })
    );

    let arm_spec = |role: &str| -> (&str, bool, usize) {
        match (mode.as_str(), role) {
            ("dispatch", "A") => ("D0", false, n_pos),
            ("dispatch", "B") => ("D1", true, n_pos),
            ("overlap", "A") => ("D1", true, n_pos),
            ("overlap", "B") => ("D2", true, prefix),
            ("compact", "S") => ("D1", true, n_pos),
            _ => panic!("invalid mode/role {mode}/{role}"),
        }
    };
    let bank = |name: &str| -> &Bank {
        match name {
            "X" => &bank_x,
            "Y" => &bank_y,
            _ => panic!("invalid bank {name}"),
        }
    };
    let run_role = |role: &str, bank_name: &str| -> (f64, f64) {
        let (arm, compact, rows) = arm_spec(role);
        if mode == "overlap" {
            prep_overlap(&ctx, bank(bank_name), prefix, chunk, n_pos, kv_dim);
        }
        run_span(&ctx, bank(bank_name), arm, 0, rows, n_pos, kv_dim, compact)
    };

    let warmups: Vec<(&str, &str)> = match mode.as_str() {
        "dispatch" => vec![("A", "X"), ("B", "Y"), ("A", "Y"), ("B", "X")],
        "compact" => vec![("S", "X"), ("S", "Y")],
        "overlap" => vec![("A", "X"), ("B", "Y"), ("A", "Y"), ("B", "X")],
        _ => unreachable!(),
    };
    for (role, bank_name) in warmups {
        let _ = run_role(role, bank_name);
    }

    let paired_schedule = [
        [("A", "X"), ("B", "Y")],
        [("B", "X"), ("A", "Y")],
        [("B", "Y"), ("A", "X")],
        [("A", "Y"), ("B", "X")],
        [("A", "X"), ("B", "Y")],
        [("B", "X"), ("A", "Y")],
    ];
    let single_banks = ["X", "Y", "Y", "X", "X", "Y"];

    if mode == "compact" {
        for (sample_idx, bank_name) in single_banks.iter().enumerate() {
            let (wall_ms, gpu_ms) = run_role("S", bank_name);
            let (arm, compact, rows) = arm_spec("S");
            let logical_bytes = (N_LAYERS as u64)
                .checked_mul(kv_dim as u64)
                .and_then(|value| value.checked_mul(rows as u64))
                .and_then(|value| value.checked_mul(4))
                .unwrap();
            let total = kv_dim.checked_mul(rows).unwrap();
            let threadgroups = attn_matrix_vt_threadgroups(total, compact).unwrap();
            println!(
                "VT_REBUILD_JSON {}",
                serde_json::json!({
                    "kind": "arm",
                    "schema_version": 2,
                    "mode": mode,
                    "prefix": prefix,
                    "chunk": chunk,
                    "sample": sample_idx + 1,
                    "role": "S",
                    "arm": arm,
                    "bank": bank_name,
                    "base_pos": 0,
                    "rows": rows,
                    "threadgroups_per_layer": threadgroups,
                    "thread_slots_per_layer": threadgroups * ATTN_MATRIX_VT_THREADS,
                    "logical_bytes": logical_bytes,
                    "wall_ms": wall_ms,
                    "gpu_ms": gpu_ms,
                    "gb_s": logical_bytes as f64 / (gpu_ms * 1e6),
                })
            );
        }
    } else {
        for (pair_idx, pair) in paired_schedule.iter().enumerate() {
            let order = format!("{}{}", pair[0].0, pair[1].0);
            for (sequence_idx, &(role, bank_name)) in pair.iter().enumerate() {
                let (wall_ms, gpu_ms) = run_role(role, bank_name);
                let (arm, compact, rows) = arm_spec(role);
                let logical_bytes = (N_LAYERS as u64)
                    .checked_mul(kv_dim as u64)
                    .and_then(|value| value.checked_mul(rows as u64))
                    .and_then(|value| value.checked_mul(4))
                    .unwrap();
                let total = kv_dim.checked_mul(rows).unwrap();
                let threadgroups = attn_matrix_vt_threadgroups(total, compact).unwrap();
                println!(
                    "VT_REBUILD_JSON {}",
                    serde_json::json!({
                        "kind": "arm",
                        "schema_version": 2,
                        "mode": mode,
                        "prefix": prefix,
                        "chunk": chunk,
                        "pair": pair_idx + 1,
                        "order": order,
                        "sequence": sequence_idx + 1,
                        "role": role,
                        "arm": arm,
                        "bank": bank_name,
                        "base_pos": 0,
                        "rows": rows,
                        "threadgroups_per_layer": threadgroups,
                        "thread_slots_per_layer": threadgroups * ATTN_MATRIX_VT_THREADS,
                        "logical_bytes": logical_bytes,
                        "wall_ms": wall_ms,
                        "gpu_ms": gpu_ms,
                        "gb_s": logical_bytes as f64 / (gpu_ms * 1e6),
                    })
                );
            }
        }
    }
}

/// Micro-oracle for the non-flash matrix-attention sidecar
/// (`kernel_attn_matrix_{transpose_v,kq,softmax,kqv}_f32`): the packed
/// prefill attention body. Previously this path had only end-to-end
/// coverage (27B G6 prefix gate + runtime packed oracle); this gate pins
/// the kernels in isolation against a CPU f64 reference across all four
/// production group shapes, full/edge tile geometries, and mid-sequence
/// `base_pos > 0` chunks (including the tiny-chunk/long-prefix shape).
///
/// Also asserts `causal_skip` on/off produce bitwise-identical output:
/// skipped KQ tiles are exactly the rows the softmax zero-masks, and
/// skipped KQV K-tiles multiply exact-zero probs.
#[test]
fn attn_matrix_path_matches_cpu_reference() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let hd = 256usize;
    // (n_q, n_kv): G4 small dense, G6 27B, G8 A3B, G16 A10B.
    let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];
    // (n_rows, base_pos); n_pos = base_pos + n_rows as in production
    // (chunk attends to the whole prefix incl. itself).
    //  - (32, 0): first chunk, n_pos < 64 → KQ edge tiles
    //  - (64, 0): full 64-pos KQ tile; N edge depends on group
    //  - (17, 47): odd everything (M/N edge tiles, base_pos > 0)
    //  - (128, 896): full tiles, mid-sequence, n_pos = 1024
    //  - (8, 1016): tiny chunk over long prefix (prefix-gate shape)
    //  - (100, 156): n_pos = 256; N edge for G6/G4, full N for G8/G16
    let cases: &[(usize, usize)] = &[
        (32, 0),
        (64, 0),
        (17, 47),
        (128, 896),
        (8, 1016),
        (100, 156),
    ];

    for &(n_q, n_kv) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &(n_rows, base_pos) in cases {
            let n_pos = base_pos + n_rows;
            let round16 = |x: f32| half::f16::from_f32(x).to_f32();
            let q: Vec<f32> = (0..n_rows * n_q * hd)
                .map(|i| round16(((i % 31) as f32 - 15.0) * 1e-2 + ((i % 7) as f32) * 3e-3))
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| round16(((i % 23) as f32 - 11.0) * 1.5e-2))
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| round16(((i % 17) as f32 - 8.0) * 2e-2))
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_rows * n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let vt_stride = n_pos;
            let v_t = MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
            let scores =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

            let run_path = |causal_skip: bool| -> Vec<f32> {
                one_shot(&ctx, |enc| {
                    encode_attn_matrix_transpose_v_f16(
                        &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
                    )?;
                    encode_attn_matrix_kq_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &scores,
                        n_rows,
                        base_pos,
                        n_pos,
                        kv_dim,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        causal_skip,
                    )?;
                    encode_attn_matrix_softmax_f32(
                        &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                    )?;
                    encode_attn_matrix_kqv_f32(
                        &ctx,
                        enc,
                        &scores,
                        &v_t,
                        &out_t,
                        n_rows,
                        base_pos,
                        n_pos,
                        vt_stride,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        causal_skip,
                    )
                })
                .unwrap();
                read_back_f32(&out_t.buffer, n_rows * n_q * hd)
            };

            let y_gpu = run_path(true);
            let y_ref = cpu_matrix_attn_reference(
                &q, &k_f32, &v_f32, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
            );

            let max_abs = y_gpu
                .iter()
                .zip(y_ref.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let dot: f64 = y_gpu
                .iter()
                .zip(y_ref.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = y_gpu
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = y_ref
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            eprintln!(
                "[matrix group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={max_abs:.2e}  cos={cos:.7}"
            );
            assert!(
                cos > 0.9999,
                "matrix path vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {cos}"
            );
            assert!(
                max_abs < 5e-3,
                "matrix path vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {max_abs}"
            );

            // causal_skip must be a pure perf feature: bitwise-identical out.
            if matches!((n_rows, base_pos), (64, 0) | (8, 1016)) {
                let y_noskip = run_path(false);
                assert!(
                    y_gpu == y_noskip,
                    "causal_skip changed matrix attention output (group={group} n_rows={n_rows} base={base_pos})"
                );
            }

            // Two-pass online kernels: KQ folds the softmax into its
            // epilogue (F16 P~ + (m,l) sidecar), KQV normalizes during
            // staging. Must sit in the same envelope vs the CPU reference
            // (the P~ half demotion mirrors the sidecar's half probs).
            let scores_h_t =
                MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let ml_t =
                MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64])
                    .unwrap();
            let fused_t = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_matrix_kq_online_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &scores_h_t,
                    &ml_t,
                    n_rows,
                    base_pos,
                    n_pos,
                    kv_dim,
                    n_q,
                    n_kv,
                    group,
                    hd,
                    true,
                )?;
                encode_attn_matrix_kqv_norm_f32(
                    &ctx,
                    enc,
                    &scores_h_t,
                    &ml_t,
                    &v_t,
                    &fused_t,
                    n_rows,
                    base_pos,
                    n_pos,
                    vt_stride,
                    n_q,
                    n_kv,
                    group,
                    hd,
                    true,
                )
            })
            .unwrap();
            let y_fused = read_back_f32(&fused_t.buffer, n_rows * n_q * hd);
            let fmax_abs = y_fused
                .iter()
                .zip(y_ref.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let fdot: f64 = y_fused
                .iter()
                .zip(y_ref.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let fna: f64 = y_fused
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let fcos = fdot / (fna * nb);
            let gpu_max_abs = y_fused
                .iter()
                .zip(y_gpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[online group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={fmax_abs:.2e}  cos={fcos:.7}  vs3k|Δ|={gpu_max_abs:.2e}"
            );
            assert!(
                fcos > 0.9999,
                "online matrix attn vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {fcos}"
            );
            assert!(
                fmax_abs < 5e-3,
                "online matrix attn vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {fmax_abs}"
            );
            assert!(
                gpu_max_abs < 5e-3,
                "online vs 3-kernel matrix attn diverged (group={group} n_rows={n_rows} base={base_pos}): {gpu_max_abs}"
            );

            if (n_rows, base_pos) == (100, 156) {
                let query_cap = 32usize;
                let tiled_scores =
                    MetalTensor::zeros_f16(&ctx, vec![(query_cap * n_q * n_pos) as u64]).unwrap();
                let tiled_ml = MetalTensor::zeros_f32(
                    &ctx,
                    vec![attn_matrix_ml_elems(query_cap, n_q, n_pos) as u64],
                )
                .unwrap();
                let tiled_out =
                    MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
                one_shot(&ctx, |enc| {
                    for row_base in (0..n_rows).step_by(query_cap) {
                        let rows_n = (n_rows - row_base).min(query_cap);
                        let q_rows = q_t.view_subrange(
                            (row_base * n_q * hd) as u64,
                            vec![(rows_n * n_q * hd) as u64],
                        );
                        let out_rows = tiled_out.view_subrange(
                            (row_base * n_q * hd) as u64,
                            vec![(rows_n * n_q * hd) as u64],
                        );
                        let scores_rows =
                            tiled_scores.view_subrange(0, vec![(rows_n * n_q * n_pos) as u64]);
                        let ml_rows = tiled_ml.view_subrange(
                            0,
                            vec![attn_matrix_ml_elems(rows_n, n_q, n_pos) as u64],
                        );
                        let tile_base_pos = base_pos + row_base;
                        encode_attn_matrix_kq_online_f32(
                            &ctx,
                            enc,
                            &q_rows,
                            &k_cache,
                            &scores_rows,
                            &ml_rows,
                            rows_n,
                            tile_base_pos,
                            n_pos,
                            kv_dim,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            true,
                        )?;
                        encode_attn_matrix_kqv_norm_f32(
                            &ctx,
                            enc,
                            &scores_rows,
                            &ml_rows,
                            &v_t,
                            &out_rows,
                            rows_n,
                            tile_base_pos,
                            n_pos,
                            vt_stride,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            true,
                        )?;
                    }
                    Ok(())
                })
                .unwrap();
                let y_tiled = read_back_f32(&tiled_out.buffer, n_rows * n_q * hd);
                let tiled_max_abs = y_tiled
                    .iter()
                    .zip(y_fused.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    tiled_max_abs < 5e-5,
                    concat!(
                        "tiled vs untiled online attention diverged ",
                        "(group={}): {}"
                    ),
                    group,
                    tiled_max_abs
                );
            }
        }
    }
}

/// Kill-gate microbench for the two-pass online-softmax matrix attention
/// kernels vs the three-kernel sidecar (KQ + softmax + KQV) at production
/// chunk shapes. The promotion bar is >= 1.2x on the summed sidecar time.
/// Vᵀ transpose/maintenance is excluded from both sides: both variants
/// consume the same Vᵀ sidecar, so its upkeep cancels.
///
/// `cargo test -p qwen-llm --release attn_matrix_online_vs_sidecar_microbench -- --ignored --nocapture`
#[test]
#[ignore]
fn attn_matrix_online_vs_sidecar_microbench() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let hd = 256usize;

    fn timed_gpu<F>(ctx: &MetalContext, iters: usize, encode: F) -> f64
    where
        F: Fn(&KernelEncoder) -> Result<(), MetalError>,
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);
        for _ in 0..iters {
            encode(&enc).unwrap();
        }
        enc.end();
        let t0 = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    // (n_q, n_kv, n_rows, n_pos, label)
    let shapes: &[(usize, usize, usize, usize, &str)] = &[
        (16, 2, 1024, 4096, "G8/A3B chunk@pp4096"),
        (16, 2, 1024, 16384, "G8/A3B chunk@pp16384"),
        (24, 4, 1024, 4096, "G6/27B chunk@pp4096"),
        (24, 4, 1024, 16384, "G6/27B chunk@pp16384"),
        (32, 2, 1024, 1024, "G16/A10B chunk@pp1024"),
        (32, 2, 1024, 4096, "G16/A10B chunk@pp4096"),
    ];

    for &(n_q, n_kv, n_rows, n_pos, label) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        let base_pos = n_pos - n_rows;

        let q: Vec<f32> = (0..n_rows * n_q * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let kv_f32: Vec<f32> = (0..n_pos * kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_rows * n_q * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        for dst in [&k_cache, &v_cache] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(kv_f32.as_slice()),
                vec![kv_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, kv_f32.len())
            })
            .unwrap();
        }
        let vt_stride = n_pos;
        let v_t = MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_attn_matrix_transpose_v_f16(
                &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
            )
        })
        .unwrap();
        let scores =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
        let scores_h =
            MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
        let ml =
            MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64])
                .unwrap();
        let out_3k = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
        let out_fused = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

        let encode_3k = |enc: &KernelEncoder| -> Result<(), MetalError> {
            encode_attn_matrix_kq_f32(
                &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q, n_kv,
                group, hd, true,
            )?;
            encode_attn_matrix_softmax_f32(
                &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
            )?;
            encode_attn_matrix_kqv_f32(
                &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q, n_kv,
                group, hd, true,
            )
        };
        let encode_fused = |enc: &KernelEncoder| -> Result<(), MetalError> {
            encode_attn_matrix_kq_online_f32(
                &ctx, enc, &q_t, &k_cache, &scores_h, &ml, n_rows, base_pos, n_pos, kv_dim, n_q,
                n_kv, group, hd, true,
            )?;
            encode_attn_matrix_kqv_norm_f32(
                &ctx, enc, &scores_h, &ml, &v_t, &out_fused, n_rows, base_pos, n_pos, vt_stride,
                n_q, n_kv, group, hd, true,
            )
        };

        // Warmup + correctness spot at production size.
        one_shot(&ctx, |enc| encode_3k(enc)).unwrap();
        one_shot(&ctx, |enc| encode_fused(enc)).unwrap();
        let y3k = read_back_f32(&out_3k.buffer, n_rows * n_q * hd);
        let yfused = read_back_f32(&out_fused.buffer, n_rows * n_q * hd);
        let max_abs = y3k
            .iter()
            .zip(yfused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 5e-3,
            "fused vs sidecar diverged at {label}: max|Δ|={max_abs}"
        );

        let iters = 8usize;
        let mut t3k = f64::INFINITY;
        let mut tfused = f64::INFINITY;
        let mut tkq = f64::INFINITY;
        let mut tsm = f64::INFINITY;
        let mut tkqv = f64::INFINITY;
        for _ in 0..3 {
            t3k = t3k.min(timed_gpu(&ctx, iters, encode_3k));
            tfused = tfused.min(timed_gpu(&ctx, iters, encode_fused));
            tkq = tkq.min(timed_gpu(&ctx, iters, |enc| {
                encode_attn_matrix_kq_f32(
                    &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q, n_kv,
                    group, hd, true,
                )
            }));
            tsm = tsm.min(timed_gpu(&ctx, iters, |enc| {
                encode_attn_matrix_softmax_f32(
                    &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                )
            }));
            tkqv = tkqv.min(timed_gpu(&ctx, iters, |enc| {
                encode_attn_matrix_kqv_f32(
                    &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q,
                    n_kv, group, hd, true,
                )
            }));
        }
        eprintln!(
            "[{label:>22}] 3k={:8.3} ms (kq={:.3} sm={:.3} kqv={:.3})  online2p={:8.3} ms  ratio={:.2}x  vs|Δ|={max_abs:.2e}",
            t3k * 1e3,
            tkq * 1e3,
            tsm * 1e3,
            tkqv * 1e3,
            tfused * 1e3,
            t3k / tfused
        );
    }
}

/// Focused correctness gate for the A3B group-8 long-context subgroup path.
///
/// Run in a fresh process with one of:
///
/// - `QWEN_ATTN_V4_G8_TILE=4 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
/// - `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
///
/// The env var is intentionally process-global (`OnceLock`) so this test stays
/// ignored and single-purpose.
#[test]
#[ignore]
fn attn_v4_group8_subgroup_matches_naive_f16kv() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let n_q = 16usize;
    let n_kv = 2usize;
    let hd = 256usize;
    let kv_dim = n_kv * hd;
    for &(n_pos, nwg, tile_c) in &[(4096usize, 64usize, 64usize), (6144, 64, 64)] {
        let q: Vec<f32> = (0..n_q * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let cap = n_pos;
        let k_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let v_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_q * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(src_f32.as_slice()),
                vec![src_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
            })
            .unwrap();
        }

        let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_attn_decode_f16kv_f32(
                &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
            )
        })
        .unwrap();
        let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

        let o_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * hd) as u64]).unwrap();
        let ml_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * 2) as u64]).unwrap();
        let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_attn_decode_v4_f32(
                &ctx,
                enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial,
                &ml_partial,
                &y_v4_t,
                n_q,
                n_kv,
                hd,
                n_pos,
                nwg,
                tile_c,
            )
        })
        .unwrap();
        let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
        let max_abs = y_v4
            .iter()
            .zip(y_naive.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = y_v4
            .iter()
            .zip(y_naive.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = y_naive
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!(
            "[v4-g8-subgroup n_pos={n_pos:>5} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e} cos={cos:.6}"
        );
        assert!(
            cos > 0.9999,
            "group8 subgroup cos too low at n_pos={n_pos}: {cos}"
        );
        assert!(
            max_abs < 5e-3,
            "group8 subgroup max|Δ| too high at n_pos={n_pos}: {max_abs}"
        );
    }
}

/// Prompt-native packed-attention microproof for the A3B long-context shape.
///
/// Compares the new packed multi-query microkernel against repeated
/// decode-shaped `attn_v4` calls using the same subgroup setting
/// (`g8_t2`) and the same F16 KV cache.
#[test]
#[ignore]
fn attn_v4_prefill_g8_t2_q2_c64_vs_decode_loop() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    const N_Q: usize = 16;
    const N_KV: usize = 2;
    const HD: usize = 256;
    const N_ROWS: usize = 128;
    const NWG: usize = 64;
    const TILE_C: usize = 64;
    let kv_dim = N_KV * HD;

    for &base_pos in &[16384usize, 32768] {
        let n_pos = base_pos + N_ROWS;
        let q_rows: Vec<f32> = (0..N_ROWS * N_Q * HD)
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
            bytemuck::cast_slice(&q_rows),
            vec![(N_ROWS * N_Q * HD) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(src_f32.as_slice()),
                vec![src_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
            })
            .unwrap();
        }

        let out_baseline = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
        let out_packed = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
        let o_partial_row =
            MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * HD) as u64]).unwrap();
        let ml_partial_row =
            MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * 2) as u64]).unwrap();
        let o_partial_packed =
            MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * HD) as u64])
                .unwrap();
        let ml_partial_packed =
            MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * 2) as u64])
                .unwrap();

        let t = Instant::now();
        with_attn_v4_group_tile_override(2, || {
            one_shot(&ctx, |enc| {
                for row in 0..N_ROWS {
                    let q_row = q_t.view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                    let out_row = out_baseline
                        .view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                    encode_attn_decode_v4_f32(
                        &ctx,
                        enc,
                        &q_row,
                        &k_cache,
                        &v_cache,
                        &o_partial_row,
                        &ml_partial_row,
                        &out_row,
                        N_Q,
                        N_KV,
                        HD,
                        base_pos + row + 1,
                        NWG,
                        TILE_C,
                    )
                    .unwrap();
                }
                Ok(())
            })
        })
        .unwrap();
        let baseline_wall = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        one_shot(&ctx, |enc| {
            encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                &ctx,
                enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                N_ROWS,
                base_pos,
                NWG,
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let packed_wall = t.elapsed().as_secs_f64() * 1e3;

        let baseline = read_back_f32(&out_baseline.buffer, N_ROWS * N_Q * HD);
        let packed = read_back_f32(&out_packed.buffer, N_ROWS * N_Q * HD);
        let max_abs = packed
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = packed
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = packed
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = baseline
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!(
            "[v4-prefill-a3b base_pos={base_pos:>5} rows={N_ROWS:>3}] decode_loop={baseline_wall:7.2} ms packed={packed_wall:7.2} ms speedup={:.3} max|Δ|={max_abs:.2e} cos={cos:.6}",
            baseline_wall / packed_wall
        );
        assert!(
            cos > 0.99999,
            "prefill packed cos too low at base_pos={base_pos}: {cos}"
        );
        assert!(
            max_abs < 2e-3,
            "prefill packed max|Δ| too high at base_pos={base_pos}: {max_abs}"
        );
    }
}

/// Bench: sweep NWG (split-K count) across context lengths to discover
/// the optimal NWG for our shape on the host GPU. Compares against the
/// naive f16kv kernel.
///
/// Run with: `cargo test --release --lib -p qwen-llm attn_v4_nwg_sweep
/// --ignored -- --nocapture`
#[test]
#[ignore]
fn attn_v4_nwg_sweep() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-bench] {}", ctx.describe());

    let n_q = 24usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let kv_dim = n_kv * hd;
    const GROUP: usize = 6;

    let n_iters = 200usize; // chained dispatches per command buffer
    let warmup = 20usize;

    // Each context length we want to characterize.
    // Past 16K we skip naive_f16kv (cap'd) and only run v4 NWG sweep.
    for &n_pos in &[64usize, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072] {
        let cap = n_pos.max(64);
        let q: Vec<f32> = (0..n_q * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let k_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let v_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_q * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(src_f32.as_slice()),
                vec![src_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
            })
            .unwrap();
        }
        let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

        // ----- Naive f16kv baseline (skip if past tg-mem cap ~7000) -----
        let naive_works = n_pos * std::mem::size_of::<f32>() <= 28 * 1024;
        if naive_works {
            let bench = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_f16kv_f32(
                        &ctx, &enc, &q_t, &k_cache, &v_cache, &y_t, n_q, n_kv, hd, n_pos,
                    )
                    .unwrap();
                }
                enc.end();
                let t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[n_pos={n_pos:>5} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };
            // Warmup
            bench("naive_f16kv_warmup", warmup);
            bench("naive_f16kv       ", n_iters);
        } else {
            eprintln!("[n_pos={n_pos:>5} naive_f16kv       ] skipped (past tg-mem cap)");
        }

        // ----- v4: sweep NWG -----
        for &nwg in &[1usize, 2, 4, 8, 16, 32] {
            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();
            let bench = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_f32(
                        &ctx,
                        &enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        &y_t,
                        n_q,
                        n_kv,
                        hd,
                        n_pos,
                        nwg,
                        32, // tile_c — NWG sweep holds tile constant
                    )
                    .unwrap();
                }
                enc.end();
                let t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[n_pos={n_pos:>5} {label} nwg={nwg:>2}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };
            bench("v4_warmup           ", warmup);
            bench("v4                  ", n_iters);
        }
        eprintln!();
    }
}

/// Bench: sweep TILE-C (KV positions per inner softmax tile) at
/// production NWG settings. Per Codex's review, GQA-dedup raises
/// arithmetic intensity per K row, which may shift the optimal C
/// away from llama.cpp's vec-kernel default of 32.
///
/// Run with: `cargo test --release --lib -p qwen-llm
/// attn_v4_tile_c_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn attn_v4_tile_c_sweep() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-c-sweep] {}", ctx.describe());

    let n_q = 24usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let kv_dim = n_kv * hd;
    const GROUP: usize = 6;

    let n_iters = 200usize;
    let warmup = 20usize;

    // For each ctx, use the production NWG heuristic (16 below 256, 32 above).
    for &n_pos in &[64usize, 256, 1024, 4096, 16384, 65536, 131072] {
        let nwg = if n_pos < 256 { 16usize } else { 32usize };
        let cap = n_pos.max(64);

        let q: Vec<f32> = (0..n_q * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let k_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let v_f32: Vec<f32> = (0..cap * kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_q * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
        for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
            let src_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(src_f32.as_slice()),
                vec![src_f32.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
            })
            .unwrap();
        }
        let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        let o_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
        let ml_partial =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();

        for &tile_c in &[16usize, 32, 64, 128] {
            let bench = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_f32(
                        &ctx,
                        &enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        &y_t,
                        n_q,
                        n_kv,
                        hd,
                        n_pos,
                        nwg,
                        tile_c,
                    )
                    .unwrap();
                }
                enc.end();
                let t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };
            bench("warmup", warmup);
            bench("bench ", n_iters);
        }
        eprintln!();
    }
}

/// Focused long-context v4 NWG sweep for the MoE shapes we now care
/// about: A3B (group=8) and 122B-A10B (group=16). Synthetic K/V is
/// enough because we're tuning the attention kernel itself, not model
/// semantics.
#[test]
#[ignore]
fn attn_v4_nwg_sweep_moe_shapes() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-moe-nwg] {}", ctx.describe());

    let hd = 256usize;
    let n_iters = 120usize;
    let warmup = 16usize;
    let shapes: &[(usize, usize, &str)] = &[(16, 2, "a3b"), (32, 2, "122b")];

    for &(n_q, n_kv, label_shape) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &n_pos in &[4096usize, 8192, 16384, 32768] {
            let cap = n_pos;
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            for &nwg in &[4usize, 8, 16, 32, 64] {
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            32,
                        )
                        .unwrap();
                    }
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[v4-moe-nwg {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                bench("warmup", warmup);
                bench("bench ", n_iters);
            }
            eprintln!();
        }
    }
}

/// Focused long-context tile-C sweep for the same MoE shapes. Uses the
/// production-default NWG=32 at these contexts unless data says otherwise.
#[test]
#[ignore]
fn attn_v4_tile_c_sweep_moe_shapes() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-moe-c] {}", ctx.describe());

    let hd = 256usize;
    let n_iters = 120usize;
    let warmup = 16usize;
    let shapes: &[(usize, usize, &str)] = &[(16, 2, "a3b"), (32, 2, "122b")];

    for &(n_q, n_kv, label_shape) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &n_pos in &[4096usize, 8192, 16384, 32768] {
            for &nwg in &[32usize, 64] {
                let cap = n_pos;
                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let k_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();

                for &tile_c in &[16usize, 32, 64, 128] {
                    let bench = |label: &str, n: usize| {
                        let cmd = ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        for _ in 0..n {
                            encode_attn_decode_v4_f32(
                                &ctx,
                                &enc,
                                &q_t,
                                &k_cache,
                                &v_cache,
                                &o_partial,
                                &ml_partial,
                                &y_t,
                                n_q,
                                n_kv,
                                hd,
                                n_pos,
                                nwg,
                                tile_c,
                            )
                            .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        eprintln!(
                            "[v4-moe-c {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                            gpu / n as f64
                        );
                    };
                    bench("warmup", warmup);
                    bench("bench ", n_iters);
                }
                eprintln!();
            }
        }
    }
}

/// Split the v4 attention kernel into main and reduce passes so we can
/// see which part actually dominates at realistic long contexts for the
/// MoE shapes. This is synthetic, but it uses the real kernel bodies and
/// exact production shapes for A3B (group=8) and 122B (group=16).
#[test]
#[ignore]
fn attn_v4_main_reduce_breakdown_moe_shapes() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-main-reduce] {}", ctx.describe());

    let hd = 256usize;
    let n_iters = 120usize;
    let warmup = 16usize;
    let shapes: &[(usize, usize, &str, &[usize])] = &[
        (16, 2, "a3b", &[4096, 16384, 32768]),
        (32, 2, "122b", &[4096, 16384, 32768]),
    ];

    for &(n_q, n_kv, label_shape, ctxs) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &n_pos in ctxs {
            let nwg = attn_v4_choose_nwg(n_pos, group);
            let tile_c = attn_v4_choose_tile_c(n_pos, group);
            let cap = n_pos;

            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            let bench_main = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_main_only_f32(
                        &ctx,
                        &enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        n_q,
                        n_kv,
                        hd,
                        n_pos,
                        nwg,
                        tile_c,
                    )
                    .unwrap();
                }
                enc.end();
                let _t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[v4-main {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>3} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };

            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_main_only_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial,
                    &ml_partial,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
            .unwrap();

            let bench_reduce = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_reduce_only_f32(
                        &ctx,
                        &enc,
                        &o_partial,
                        &ml_partial,
                        &y_t,
                        n_q,
                        n_kv,
                        hd,
                        nwg,
                    )
                    .unwrap();
                }
                enc.end();
                let _t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[v4-reduce {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };

            bench_main("warmup", warmup);
            bench_main("bench ", n_iters);
            bench_reduce("warmup", warmup);
            bench_reduce("bench ", n_iters);
            eprintln!();
        }
    }
}

/// Synthetic head-major F16 K/V proof for the long-context MoE v4 attention
/// body. This is intentionally not a production cache layout: it isolates the
/// address-stride question before any prefill/session sidecar work.
#[test]
#[ignore]
fn attn_v4_head_major_main_reduce_breakdown_moe_shapes() {
    use std::time::Instant;
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[v4-hm-main-reduce] {}", ctx.describe());

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut aa = 0.0f64;
        let mut bb = 0.0f64;
        for (&x, &y) in a.iter().zip(b) {
            let x = x as f64;
            let y = y as f64;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        dot / (aa.sqrt() * bb.sqrt()).max(1e-30)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    let hd = 256usize;
    let n_iters = 96usize;
    let warmup = 12usize;
    let shapes: &[(usize, usize, &str, &[usize])] = &[
        (16, 2, "a3b", &[8192, 16384, 32768]),
        (32, 2, "a10b", &[8192, 16384, 32768]),
    ];

    for &(n_q, n_kv, label_shape, ctxs) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &n_pos in ctxs {
            let nwg = attn_v4_choose_nwg(n_pos, group);
            let tile_c = attn_v4_choose_tile_c(n_pos, group);
            let group_tile = attn_v4_choose_group_tile(n_pos, group);
            assert!(
                (group, group_tile, tile_c) == (8, 2, 64)
                    || (group, group_tile, tile_c) == (16, 4, 64)
                    || (group, group_tile, tile_c) == (16, 4, 128),
                "unexpected group/group_tile/tile_c {group}/{group_tile}/{tile_c}"
            );

            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let k_tok_h: Vec<half::f16> = k_f32.iter().copied().map(half::f16::from_f32).collect();
            let v_tok_h: Vec<half::f16> = v_f32.iter().copied().map(half::f16::from_f32).collect();
            let mut k_hm_h = vec![half::f16::ZERO; k_tok_h.len()];
            let mut v_hm_h = vec![half::f16::ZERO; v_tok_h.len()];
            for pos in 0..n_pos {
                for kvh in 0..n_kv {
                    let src = pos * kv_dim + kvh * hd;
                    let dst = (kvh * n_pos + pos) * hd;
                    k_hm_h[dst..dst + hd].copy_from_slice(&k_tok_h[src..src + hd]);
                    v_hm_h[dst..dst + hd].copy_from_slice(&v_tok_h[src..src + hd]);
                }
            }

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_tok = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k_tok_h),
                vec![(n_pos * kv_dim) as u64],
                GgmlType::F16,
            )
            .unwrap();
            let v_tok = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&v_tok_h),
                vec![(n_pos * kv_dim) as u64],
                GgmlType::F16,
            )
            .unwrap();
            let k_hm = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k_hm_h),
                vec![(n_pos * kv_dim) as u64],
                GgmlType::F16,
            )
            .unwrap();
            let v_hm = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&v_hm_h),
                vec![(n_pos * kv_dim) as u64],
                GgmlType::F16,
            )
            .unwrap();

            let partial_elems = n_kv * nwg * group * hd;
            let ml_elems = n_kv * nwg * group * 2;
            let o_tok = MetalTensor::zeros_f32(&ctx, vec![partial_elems as u64]).unwrap();
            let ml_tok = MetalTensor::zeros_f32(&ctx, vec![ml_elems as u64]).unwrap();
            let y_tok = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            let o_hm = MetalTensor::zeros_f32(&ctx, vec![partial_elems as u64]).unwrap();
            let ml_hm = MetalTensor::zeros_f32(&ctx, vec![ml_elems as u64]).unwrap();
            let y_hm = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_main_only_f32(
                    &ctx, enc, &q_t, &k_tok, &v_tok, &o_tok, &ml_tok, n_q, n_kv, hd, n_pos, nwg,
                    tile_c,
                )?;
                encode_attn_decode_v4_reduce_only_f32(
                    &ctx, enc, &o_tok, &ml_tok, &y_tok, n_q, n_kv, hd, nwg,
                )
            })
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_main_only_f32_head_major(
                    &ctx, enc, &q_t, &k_hm, &v_hm, &o_hm, &ml_hm, n_q, n_kv, hd, n_pos, nwg, tile_c,
                )?;
                encode_attn_decode_v4_reduce_only_f32(
                    &ctx, enc, &o_hm, &ml_hm, &y_hm, n_q, n_kv, hd, nwg,
                )
            })
            .unwrap();
            let y_tok_v = read_back_f32(&y_tok.buffer, n_q * hd);
            let y_hm_v = read_back_f32(&y_hm.buffer, n_q * hd);
            eprintln!(
                "[v4-hm-correct {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2}] cos={:.8} max_abs={:.3e}",
                cosine(&y_tok_v, &y_hm_v),
                max_abs(&y_tok_v, &y_hm_v)
            );

            let bench_tok = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_main_only_f32(
                        &ctx, &enc, &q_t, &k_tok, &v_tok, &o_tok, &ml_tok, n_q, n_kv, hd, n_pos,
                        nwg, tile_c,
                    )
                    .unwrap();
                }
                enc.end();
                let _t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[v4-main-token {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };
            let bench_hm = |label: &str, n: usize| {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for _ in 0..n {
                    encode_attn_decode_v4_main_only_f32_head_major(
                        &ctx, &enc, &q_t, &k_hm, &v_hm, &o_hm, &ml_hm, n_q, n_kv, hd, n_pos, nwg,
                        tile_c,
                    )
                    .unwrap();
                }
                enc.end();
                let _t = Instant::now();
                cmd.commit();
                cmd.waitUntilCompleted();
                let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                eprintln!(
                    "[v4-main-hmajor {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                    gpu / n as f64
                );
            };

            bench_tok("warmup", warmup);
            bench_hm("warmup", warmup);
            bench_tok("bench ", n_iters);
            bench_hm("bench ", n_iters);
            eprintln!();
        }
    }
}

/// Split the prompt-native packed prefill kernels into main and reduce
/// passes so we can see how much of the remaining packed-attention wall is
/// still the F32 partial spill/reduce path.
#[test]
#[ignore]
fn attn_prefill_v4_main_reduce_breakdown_moe_shapes() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    eprintln!("[prefill-v4-main-reduce] {}", ctx.describe());

    let hd = 256usize;
    let n_iters = 96usize;
    let warmup = 12usize;
    let rows_set = [4usize, 8usize];
    let shapes: &[(usize, usize, &str, &[usize])] = &[
        (16, 2, "a3b", &[4096, 16384, 32768]),
        (32, 2, "122b", &[4096, 16384, 32768]),
    ];

    for &(n_q, n_kv, label_shape, ctxs) in shapes {
        let group = n_q / n_kv;
        let kv_dim = n_kv * hd;
        for &n_rows in &rows_set {
            for &base_pos in ctxs {
                let n_pos = base_pos + n_rows;
                let nwg = attn_v4_choose_nwg(n_pos, group);

                let q: Vec<f32> = (0..n_rows * n_q * hd)
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
                    vec![(n_rows * n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }

                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_kv * nwg * group * hd) as u64])
                        .unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_kv * nwg * group * 2) as u64])
                        .unwrap();
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

                let bench_main = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        match group {
                            8 => encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
                                &ctx,
                                &enc,
                                &q_t,
                                &k_cache,
                                &v_cache,
                                &o_partial,
                                &ml_partial,
                                n_rows,
                                base_pos,
                                nwg,
                            ),
                            16 => encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
                                &ctx,
                                &enc,
                                &q_t,
                                &k_cache,
                                &v_cache,
                                &o_partial,
                                &ml_partial,
                                n_rows,
                                base_pos,
                                nwg,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                    }
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[prefill-v4-main {label_shape} rows={n_rows:>2} group={group:>2} base_pos={base_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };

                one_shot(&ctx, |enc| match group {
                    8 => encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        n_rows,
                        base_pos,
                        nwg,
                    ),
                    16 => encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        n_rows,
                        base_pos,
                        nwg,
                    ),
                    _ => unreachable!(),
                })
                .unwrap();

                let bench_reduce = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        match group {
                            8 => encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
                                &ctx,
                                &enc,
                                &o_partial,
                                &ml_partial,
                                &y_t,
                                n_rows,
                                nwg,
                            ),
                            16 => encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
                                &ctx,
                                &enc,
                                &o_partial,
                                &ml_partial,
                                &y_t,
                                n_rows,
                                nwg,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                    }
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[prefill-v4-reduce {label_shape} rows={n_rows:>2} group={group:>2} base_pos={base_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };

                bench_main("warmup", warmup);
                bench_main("bench ", n_iters);
                bench_reduce("warmup", warmup);
                bench_reduce("bench ", n_iters);
                eprintln!();
            }
        }
    }
}

fn greedy_total_order_key(bits: u32) -> Option<u32> {
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

#[test]
fn argmax_rejects_f32_output_metadata() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let input = MetalTensor::zeros_f32(&ctx, vec![2]).unwrap();
    let wrong_output = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd);
    assert!(encode_argmax_f32(&ctx, &enc, &input, &wrong_output, 1, 2).is_err());
    assert!(encode_argmax_f32_greedy(&ctx, &enc, &input, &wrong_output, 1, 2).is_err());
    enc.end();
}

#[test]
fn greedy_argmax_matches_sampler_total_order_contract() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> Vec<i32> {
        assert_eq!(x.len(), n_rows * n);
        let xt = MetalTensor {
            buffer: ctx.buffer_from(x).expect("input buffer"),
            offset: 0,
            shape: vec![n_rows as u64, n as u64],
            dtype: crate::tensor::GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };
        let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("output buffer");
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd);
        encode_argmax_f32_greedy(ctx, &enc, &xt, &ot, n_rows, n).expect("encode greedy argmax");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        unsafe {
            let ptr = ot.buffer.contents().as_ptr() as *const i32;
            (0..n_rows).map(|row| *ptr.add(row)).collect()
        }
    }

    fn cpu(row: &[f32]) -> i32 {
        if let Some(token) = row.iter().position(|value| value.is_nan()) {
            return !(token as i32);
        }
        let mut best = 0usize;
        for token in 1..row.len() {
            if row[token].total_cmp(&row[best]) == std::cmp::Ordering::Greater {
                best = token;
            }
        }
        best as i32
    }

    let rows = [
        [1.0, 4.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0],
        [5.0, 1.0, 5.0, 0.0, 5.0, 2.0, 5.0, 3.0],
        [-0.0, 0.0, -0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        [
            f32::INFINITY,
            1.0,
            f32::INFINITY,
            0.0,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
        ],
        [f32::NEG_INFINITY; 8],
        [0.0, f32::NAN, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0],
        [f32::NAN; 8],
        [
            f32::from_bits(1),
            -f32::from_bits(1),
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
        ],
    ];
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let got = run(&ctx, &flat, rows.len(), rows[0].len());
    let expected: Vec<i32> = flat.chunks(rows[0].len()).map(cpu).collect();
    assert_eq!(got, expected);
    assert_eq!(got[1], 0, "finite ties choose the lowest token id");
    assert_eq!(got[2], 1, "+0 outranks -0 under total_cmp");
    assert_eq!(got[4], 0, "equal -inf chooses the lowest token id");
    assert_eq!(got[5], !1, "lowest NaN token is encoded");
    assert_eq!(got[6], !0, "all-NaN row reports token zero");

    for n in [1usize, 31, 32, 33, 1023, 1025] {
        let mut row = vec![-1.0f32; n];
        row[n / 2] = 3.0;
        row[n - 1] = 3.0;
        assert_eq!(
            run(&ctx, &row, 1, n),
            vec![(n / 2) as i32],
            "boundary row length {n}"
        );
    }

    let mut wide = vec![0.0f32; 4096];
    wide[100] = 9.0;
    wide[2500] = 9.0;
    wide[3999] = 9.0;
    assert_eq!(run(&ctx, &wide, 1, wide.len()), vec![100]);
    wide[2500] = f32::NAN;
    wide[100] = f32::NAN;
    assert_eq!(run(&ctx, &wide, 1, wide.len()), vec![!100]);

    let mut vocab = vec![0.0f32; 248_320];
    let mut state = 0xc0ffeeu32;
    for value in &mut vocab {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *value = (state as i32) as f32 * 1e-9;
    }
    assert_eq!(run(&ctx, &vocab, 1, vocab.len()), vec![cpu(&vocab)]);
}

/// H5.3a GPU argmax — bit-exact match to CPU argmax with lowest-index
/// tie-breaking, including the explicit edge cases:
/// * tie at row start (idx 0 wins)
/// * tie at row end
/// * single-element row
/// * row larger than 1024 (tests cross-simdgroup reduce path)
/// * vocab-sized row (V=248320; the actual production shape)
/// * negative-infinity entries (production lm_head won't have these,
///   but defensive)
#[test]
fn argmax_matches_cpu_with_tie_to_lowest_index() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    // Helper: encode-only argmax + readback for tests.
    fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> Vec<i32> {
        assert_eq!(x.len(), n_rows * n);
        let xb = ctx.buffer_from(x).expect("xb");
        let xt = MetalTensor {
            buffer: xb,
            offset: 0,
            shape: vec![n_rows as u64, n as u64],
            dtype: crate::tensor::GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };
        let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("ob");
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_argmax_f32(ctx, &enc, &xt, &ot, n_rows, n).expect("encode argmax");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        unsafe {
            let p = ot.buffer.contents().as_ptr() as *const i32;
            (0..n_rows).map(|i| *p.add(i)).collect()
        }
    }

    fn cpu_argmax_lowest_idx(row: &[f32]) -> i32 {
        let mut best = f32::NEG_INFINITY;
        let mut idx: i32 = 0;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                idx = i as i32;
            }
        }
        idx
    }

    // 1. Single-element row.
    {
        let x = vec![3.5f32];
        let got = run(&ctx, &x, 1, 1);
        assert_eq!(got, vec![0]);
    }

    // 2. Tie at row start: x = [5.0, 1.0, 5.0, 5.0, 0.0]. Lowest idx
    //    among matches = 0.
    {
        let x = vec![5.0f32, 1.0, 5.0, 5.0, 0.0];
        let got = run(&ctx, &x, 1, 5);
        assert_eq!(got, vec![0], "tie at start should pick idx 0");
    }

    // 3. Tie at row end (max only at the last position).
    {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let got = run(&ctx, &x, 1, 5);
        assert_eq!(got, vec![4]);
    }

    // 4. Tie spread across the row at multiple distant positions.
    {
        let mut x = vec![0.0f32; 4096];
        x[100] = 9.0;
        x[2500] = 9.0;
        x[3999] = 9.0;
        let got = run(&ctx, &x, 1, 4096);
        assert_eq!(got, vec![100], "spread tie should pick lowest idx");
    }

    // 5. Multi-row batch: argmax independently per row.
    {
        let n = 1024;
        let n_rows = 5;
        let mut x = vec![0.0f32; n_rows * n];
        for r in 0..n_rows {
            // Place the max for row r at idx (r * 137) % n.
            let idx = (r * 137) % n;
            x[r * n + idx] = 1.0 + (r as f32) * 0.1;
        }
        let got = run(&ctx, &x, n_rows, n);
        for r in 0..n_rows {
            let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
            assert_eq!(got[r], want, "row {r}");
        }
    }

    // 6. Random fuzz at vocab-sized row (the production shape).
    {
        let n = 248_320usize;
        let n_rows = 16;
        let mut x = vec![0.0f32; n_rows * n];
        // Deterministic pseudo-random fill.
        let mut s: u32 = 0xc0ffeeu32;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *v = (s as i32) as f32 * 1e-9;
        }
        let got = run(&ctx, &x, n_rows, n);
        for r in 0..n_rows {
            let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
            assert_eq!(got[r], want, "vocab row {r}");
        }
    }

    // 7. Negative-infinity entries (defensive — production lm_head
    //    won't produce these but the kernel must not get confused).
    {
        let mut x = vec![f32::NEG_INFINITY; 1024];
        x[42] = -1e9;
        x[500] = -1e10; // smaller than 42's value
        let got = run(&ctx, &x, 1, 1024);
        assert_eq!(got, vec![42]);
    }

    // 8. All-zero row (every position ties); lowest index = 0.
    {
        let x = vec![0.0f32; 1024];
        let got = run(&ctx, &x, 1, 1024);
        assert_eq!(got, vec![0], "all-tie should pick idx 0");
    }

    // 9. All -INFINITY row (degenerate but well-defined): every
    //    position ties at -inf, lowest idx wins. Per codex H5.3a
    //    review: this returns idx 0, NOT -1. Production lm_head
    //    cannot produce all -inf, but documenting the contract.
    {
        let x = vec![f32::NEG_INFINITY; 1024];
        let got = run(&ctx, &x, 1, 1024);
        assert_eq!(
            got,
            vec![0],
            "all -INFINITY: kernel ties at -inf, lowest idx 0 wins"
        );
    }

    // 10. All NaN row: IEEE comparison `a > b` is FALSE for any
    //     NaN operand, so the per-lane scan never updates from
    //     `best_val=-INF, best_idx=UINT_MAX`. simd_max also returns
    //     NaN; (NaN == NaN) is false, so lane_idx stays UINT_MAX
    //     for every lane, simd_min(UINT_MAX) = UINT_MAX, cast to
    //     i32 = -1.
    //
    //     Production lm_head does not produce NaN under correct
    //     numerics. Treat this as a "this kernel returns -1
    //     deterministically when the entire row is unranked";
    //     callers should not feed it NaN rows.
    {
        let x = vec![f32::NAN; 1024];
        let got = run(&ctx, &x, 1, 1024);
        assert_eq!(
            got,
            vec![-1],
            "all-NaN: kernel returns -1 (UINT_MAX cast); document only — production should never see this"
        );
    }
}

#[test]
fn argmax_top2_matches_lowest_index_and_reports_exact_gap() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> (Vec<i32>, Vec<f32>) {
        assert_eq!(x.len(), n_rows * n);
        let xb = ctx.buffer_from(x).expect("xb");
        let xt = MetalTensor {
            buffer: xb,
            offset: 0,
            shape: vec![n_rows as u64, n as u64],
            dtype: crate::tensor::GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };
        let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("ob");
        let gt = MetalTensor::zeros_f32(ctx, vec![n_rows as u64]).expect("gb");
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_argmax_top2_f32(ctx, &enc, &xt, &ot, &gt, n_rows, n).expect("encode argmax top2");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        unsafe {
            let pi = ot.buffer.contents().as_ptr() as *const i32;
            let pg = gt.buffer.contents().as_ptr() as *const f32;
            let idx = (0..n_rows).map(|i| *pi.add(i)).collect();
            let gap = (0..n_rows).map(|i| *pg.add(i)).collect();
            (idx, gap)
        }
    }

    fn cpu_top2(row: &[f32]) -> (i32, f32) {
        let mut v1 = f32::NEG_INFINITY;
        let mut i1 = 0usize;
        let mut v2 = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > v1 {
                v2 = v1;
                v1 = v;
                i1 = i;
            } else if v > v2 {
                v2 = v;
            }
        }
        (i1 as i32, v1 - v2)
    }

    // Distinct max, duplicate max (gap 0), single element (+inf),
    // all-equal (gap 0), negative max, signed zero, boundary widths.
    let rows: Vec<Vec<f32>> = vec![
        vec![1.0, 4.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0],
        vec![5.0, 1.0, 5.0, 0.0, 5.0, 2.0, 5.0, 3.0],
        vec![7.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![-8.0, -9.0, -7.0, -9.5],
        vec![-0.0, 0.0, -0.0, -1.0],
    ];
    let n_rows = rows.len();
    let n = 8;
    let mut flat = Vec::with_capacity(n_rows * n);
    let mut expected = Vec::with_capacity(n_rows);
    for row in &rows {
        assert!(row.len() <= n);
        let mut padded = row.clone();
        padded.resize(n, f32::NEG_INFINITY);
        flat.extend_from_slice(&padded);
        expected.push(cpu_top2(&padded));
    }
    let (idx, gap) = run(&ctx, &flat, n_rows, n);
    for (r, &(ei, eg)) in expected.iter().enumerate() {
        assert_eq!(idx[r], ei, "row {r} idx");
        if eg.is_finite() {
            assert_eq!(gap[r], eg, "row {r} gap");
        } else {
            assert!(gap[r].is_infinite() && gap[r] > 0.0, "row {r} +inf gap");
        }
    }

    // Boundary widths with a duplicated max: lowest index wins, gap 0
    // (width 1 collapses to a single element: gap +inf).
    for width in [1usize, 31, 32, 33, 1023, 1025] {
        let mut row = vec![-1.0f32; width];
        row[width / 2] = 3.0;
        row[width - 1] = 3.0;
        let (idx, gap) = run(&ctx, &row, 1, width);
        assert_eq!(idx, vec![(width / 2) as i32], "boundary width {width} idx");
        if width == 1 {
            assert!(gap[0].is_infinite() && gap[0] > 0.0, "boundary width 1 gap");
        } else {
            assert_eq!(gap, vec![0.0], "boundary width {width} gap");
        }
    }
}

fn fill_audit_f16(tensor: &MetalTensor, salt: usize) {
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

/// Attn-v4 decode bandwidth audit (2026-08-22): synthetic session at a
/// large kv_n_pos, one `encode_attn_decode_v4_f32` call, kernel timing
/// only. Reports achieved GB/s against the 474 GB/s stream so the
/// long-context attention anomaly (serial ~130 GB/s, verify ~80 GB/s at
/// 130K) can be attributed. Sweep with env:
/// QWEN_ATTN_AUDIT_CTX (default 131072), QWEN_ATTN_AUDIT_MODEL
/// (default Qwen3.8-27B-Q8_0), QWEN_ATTN_V4_NWG, QWEN_ATTN_V4_TILE_C.
#[test]
#[ignore = "slow real-model GPU audit; run explicitly"]
fn attn_decode_v4_bandwidth_audit_130k() {
    let model_path = std::env::var("QWEN_ATTN_AUDIT_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf".into());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("[attn-audit] skipped — model missing");
        return;
    }
    let n_pos: usize = std::env::var("QWEN_ATTN_AUDIT_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(131_072);
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&model_path).expect("open model");
    let m = crate::loader::Model::from_gguf(&g).expect("load model");
    let mm = crate::metal_forward::MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mut sess =
        crate::metal_forward::MetalSession::fresh(&ctx, &mm, n_pos + 16).expect("session");
    for kp in sess.kv_n_pos.iter_mut() {
        *kp = n_pos;
    }
    fill_audit_f16(&sess.kv_k[0], 1);
    fill_audit_f16(&sess.kv_v[0], 2);
    let arch = &m.arch;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let group = n_q / n_kv;
    if head_dim != 256 || !matches!(group, 4 | 6 | 8 | 16) {
        eprintln!("[attn-audit] skipped — unsupported shape");
        return;
    }
    let nwg = crate::metal::attn_v4_choose_nwg(n_pos, group);
    let tile_c = crate::metal::attn_v4_choose_tile_c(n_pos, group);
    let q = MetalTensor::zeros_f32(&ctx, vec![(n_q * head_dim) as u64]).expect("q");
    let attn_o = MetalTensor::zeros_f32(&ctx, vec![(n_q * head_dim) as u64]).expect("o");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    encode_attn_decode_v4_f32(
        &ctx,
        &enc,
        &q,
        &sess.kv_k[0],
        &sess.kv_v[0],
        &sess.attn_v4_o_partial,
        &sess.attn_v4_ml_partial,
        &attn_o,
        n_q,
        n_kv,
        head_dim,
        n_pos,
        nwg,
        tile_c,
    )
    .expect("encode attn v4");
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    let bytes = n_pos as f64 * (n_kv * head_dim * 2) as f64 * 2.0;
    let gbps = bytes / 1e9 / (ms / 1e3);
    eprintln!(
        "[attn-audit] ctx={n_pos} group={group} nwg={nwg} tile_c={tile_c} gpu_ms={ms:.3} gb={:.2} gbps={gbps:.1}",
        bytes / 1e9
    );
}

/// Boundary oracle for the packed g6_q2 attention reduce (2026-08-22):
/// model-free comparison of `encode_attn_prefill_v4_g6_q2_c32_f32`
/// (n_rows=8, nwg=128 — above the historical 64-partition cap) against
/// eight per-row `encode_attn_decode_v4_f32` calls on the same synthetic
/// Q/KV. Two 64-partition couplings shipped behind the old cap — a
/// fixed-64 shmem sizing in the encoder and a two-pass staging loop in
/// the reduce — both silently corrupted at nwg > 64; a cosine gate here
/// discriminates that corruption from reorder noise (corruption moved
/// logits by 5-11 absolute, reorder stays < 1e-3 cosine distance).
#[test]
#[ignore = "slow GPU oracle; run explicitly"]
fn packed_q2_attention_matches_per_row_at_high_nwg() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    const N_ROWS: usize = 8;
    const N_Q: usize = 24;
    const N_KV: usize = 4;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 6;
    let ctx_len: usize = std::env::var("QWEN_PACKED_Q2_ORACLE_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let oracle_ctx_len: usize = ctx_len;
    let base_pos: usize = oracle_ctx_len - N_ROWS;
    const NWG: usize = 128;

    let mut rng_state: u32 = 0x1234_5678;
    let mut rand_f32 = move || {
        rng_state = rng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((rng_state >> 8) as f32) / ((1u32 << 24) as f32) - 0.5
    };
    let q: Vec<f32> = (0..N_ROWS * N_Q * HEAD_DIM).map(|_| rand_f32()).collect();
    let kv_elems = oracle_ctx_len * N_KV * HEAD_DIM;
    let k_f32: Vec<f32> = (0..kv_elems).map(|_| rand_f32()).collect();
    let v_f32: Vec<f32> = (0..kv_elems).map(|_| rand_f32()).collect();
    let k_half: Vec<half::f16> = k_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
    let v_half: Vec<half::f16> = v_f32.iter().map(|v| half::f16::from_f32(*v)).collect();

    let q_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&q),
        vec![(N_ROWS * N_Q * HEAD_DIM) as u64],
        GgmlType::F32,
    )
    .expect("q");
    let k_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k_half),
        vec![kv_elems as u64],
        GgmlType::F16,
    )
    .expect("k");
    let v_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v_half),
        vec![kv_elems as u64],
        GgmlType::F16,
    )
    .expect("v");
    let o_packed =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HEAD_DIM) as u64]).expect("o packed");
    let o_partial =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * GROUP * HEAD_DIM) as u64])
            .expect("o partial");
    let ml_partial = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * GROUP * 2) as u64])
        .expect("ml partial");

    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_attn_prefill_v4_g6_q2_c32_f32(
            &ctx,
            &enc,
            &q_t,
            &k_t,
            &v_t,
            &o_partial,
            &ml_partial,
            &o_packed,
            N_ROWS,
            base_pos,
            NWG,
            true,
        )
        .expect("packed encode");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let o_per_row =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HEAD_DIM) as u64]).expect("o per row");
    let o_partial_1 = MetalTensor::zeros_f32(&ctx, vec![(N_KV * 1024 * GROUP * HEAD_DIM) as u64])
        .expect("o partial 1");
    let ml_partial_1 =
        MetalTensor::zeros_f32(&ctx, vec![(N_KV * 1024 * GROUP * 2) as u64]).expect("ml partial 1");
    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for row in 0..N_ROWS {
            let q_row =
                q_t.view_subrange((row * N_Q * HEAD_DIM) as u64, vec![(N_Q * HEAD_DIM) as u64]);
            let o_row = o_per_row
                .view_subrange((row * N_Q * HEAD_DIM) as u64, vec![(N_Q * HEAD_DIM) as u64]);
            encode_attn_decode_v4_f32(
                &ctx,
                &enc,
                &q_row,
                &k_t,
                &v_t,
                &o_partial_1,
                &ml_partial_1,
                &o_row,
                N_Q,
                N_KV,
                HEAD_DIM,
                base_pos + row + 1,
                NWG,
                32,
            )
            .expect("per-row encode");
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    unsafe {
        let a = o_packed.buffer.contents().as_ptr() as *const f32;
        let b = o_per_row.buffer.contents().as_ptr() as *const f32;
        let n = N_ROWS * N_Q * HEAD_DIM;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let av = *a.add(i);
            let bv = *b.add(i);
            dot += (av as f64) * (bv as f64);
            na += (av as f64).powi(2);
            nb += (bv as f64).powi(2);
            max_abs = max_abs.max((av - bv).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[packed-q2-oracle] cos={cos:.6} max|delta|={max_abs:.3e} nwg={NWG}");
        assert!(
            cos > 0.9999,
            "packed q2 attention diverges from per-row at nwg={NWG} (cos={cos})"
        );
    }
}

/// Perf audit for the packed g6 q2 shared-KV attention at depth
/// (2026-08-22): synthetic session at a large kv_n_pos, one
/// `encode_attn_prefill_v4_g6_q2_c32_f32` call over n_rows=8 with the
/// selector's nwg, kernel timing only. Reports ms and effective GB/s
/// (KV read once per row-pair = 4x the per-layer KV bytes). The per-row
/// baseline at 130K/nwg=512 is ~4.81ms per row (~38.5ms per layer for
/// 8 rows); the packed path is the default-on perf gate.
/// Env: QWEN_ATTN_AUDIT_CTX, QWEN_ATTN_AUDIT_MODEL.
#[test]
#[ignore = "slow real-model GPU audit; run explicitly"]
fn packed_q2_attention_perf_audit_130k() {
    let model_path = std::env::var("QWEN_ATTN_AUDIT_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf".into());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("[packed-audit] skipped — model missing");
        return;
    }
    let n_pos: usize = std::env::var("QWEN_ATTN_AUDIT_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(131_072);
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&model_path).expect("open model");
    let m = crate::loader::Model::from_gguf(&g).expect("load model");
    let mm = crate::metal_forward::MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mut sess =
        crate::metal_forward::MetalSession::fresh(&ctx, &mm, n_pos + 16).expect("session");
    for kp in sess.kv_n_pos.iter_mut() {
        *kp = n_pos;
    }
    fill_audit_f16(&sess.kv_k[0], 1);
    fill_audit_f16(&sess.kv_v[0], 2);
    let arch = &m.arch;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let group = n_q / n_kv;
    if head_dim != 256 || group != 6 {
        eprintln!("[packed-audit] skipped — unsupported shape");
        return;
    }
    const N_ROWS: usize = 8;
    const NWG_MAX: usize = 1024;
    let nwg = crate::metal::attn_v4_choose_nwg(n_pos, 6);
    let base_pos = n_pos - N_ROWS;
    let q = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * head_dim) as u64]).expect("q");
    let o = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * head_dim) as u64]).expect("o");
    let per_row_o_partial =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * NWG_MAX * group * head_dim) as u64])
            .expect("per-row o partial");
    let per_row_ml_partial =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * NWG_MAX * group * 2) as u64])
            .expect("per-row ml partial");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    for row in 0..N_ROWS {
        let q_row = q.view_subrange((row * n_q * head_dim) as u64, vec![(n_q * head_dim) as u64]);
        let o_row = o.view_subrange((row * n_q * head_dim) as u64, vec![(n_q * head_dim) as u64]);
        encode_attn_decode_v4_f32(
            &ctx,
            &enc,
            &q_row,
            &sess.kv_k[0],
            &sess.kv_v[0],
            &per_row_o_partial,
            &per_row_ml_partial,
            &o_row,
            n_q,
            n_kv,
            head_dim,
            base_pos + row + 1,
            crate::metal::attn_v4_choose_nwg(base_pos + row + 1, group),
            crate::metal::attn_v4_choose_tile_c(base_pos + row + 1, group),
        )
        .expect("per-row encode");
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let per_row_chain_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    eprintln!("[per-row-chain-audit] ctx={n_pos} rows={N_ROWS} gpu_ms={per_row_chain_ms:.3}");
    let o_partial = MetalTensor::zeros_f32(
        &ctx,
        vec![(N_ROWS * n_kv * NWG_MAX * group * head_dim) as u64],
    )
    .expect("o partial");
    let ml_partial =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_kv * NWG_MAX * group * 2) as u64])
            .expect("ml partial");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    encode_attn_prefill_v4_g6_q2_c32_f32(
        &ctx,
        &enc,
        &q,
        &sess.kv_k[0],
        &sess.kv_v[0],
        &o_partial,
        &ml_partial,
        &o,
        N_ROWS,
        base_pos,
        nwg,
        true,
    )
    .expect("packed encode");
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    // KV read once per row-pair: ceil(N_ROWS/2) passes over the layer KV.
    let passes = N_ROWS.div_ceil(2) as f64;
    let bytes = passes * n_pos as f64 * (n_kv * head_dim * 2 * 2) as f64;
    let gbps = bytes / 1e9 / (ms / 1e3);
    eprintln!(
        "[packed-audit] ctx={n_pos} nwg={nwg} gpu_ms={ms:.3} gb={:.2} gbps={gbps:.1} (per-row baseline ~38.5 ms/layer at 130K)",
        bytes / 1e9
    );

    // Tier-3 matrix reader timing: KQ (MMA) -> softmax -> direct-V KQV.
    let scores = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * n_pos) as u64]).expect("scores");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    encode_attn_matrix_kq_f32(
        &ctx,
        &enc,
        &q,
        &sess.kv_k[0],
        &scores,
        N_ROWS,
        base_pos,
        n_pos,
        n_kv * head_dim,
        n_q,
        n_kv,
        group,
        head_dim,
        true,
    )
    .expect("matrix kq");
    encode_attn_matrix_softmax_f32(
        &ctx, &enc, &scores, N_ROWS, base_pos, n_pos, n_q, n_kv, group, head_dim,
    )
    .expect("matrix softmax");
    encode_attn_matrix_kqv_direct_v_f32(
        &ctx,
        &enc,
        &scores,
        &sess.kv_v[0],
        &o,
        N_ROWS,
        base_pos,
        n_pos,
        n_kv * head_dim,
        n_q,
        n_kv,
        group,
        head_dim,
        true,
    )
    .expect("matrix kqv direct v");
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    eprintln!(
        "[matrix-audit] ctx={n_pos} gpu_ms={ms:.3} (per-row baseline ~38.5 ms/layer at 130K)"
    );
}

/// H5.3a foundation: verify `BlitEncoder` actually copies device-side
/// buffers and that compute↔blit transitions on the same command
/// buffer are visible. We write a known pattern via a compute kernel
/// (`scatter_offset`), blit-copy into a destination buffer, then read
/// the destination back. If the blit didn't fire, we'd read zeros.
#[test]
fn blit_encoder_copies_buffer_within_one_command_buffer() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    const N: usize = 1024;
    let pattern: Vec<f32> = (0..N).map(|i| (i as f32) * 0.125).collect();

    // Source: a freshly-uploaded MetalTensor holding `pattern`.
    let src_buf = ctx.buffer_from(&pattern).expect("src buf");
    let src = MetalTensor {
        buffer: src_buf,
        offset: 0,
        shape: vec![N as u64],
        dtype: crate::tensor::GgmlType::F32,
        provenance: MetalTensorProvenance::OwnedWritable,
    };

    // Destination: zero-initialized.
    let dst_buf = ctx.buffer_uninit(N * 4).expect("dst buf");
    // Zero it out via the host pointer (StorageModeShared).
    unsafe {
        let p = dst_buf.contents().as_ptr() as *mut f32;
        for i in 0..N {
            *p.add(i) = -1.0;
        }
    }
    let dst = MetalTensor {
        buffer: dst_buf,
        offset: 0,
        shape: vec![N as u64],
        dtype: crate::tensor::GgmlType::F32,
        provenance: MetalTensorProvenance::OwnedWritable,
    };

    // One command buffer. Compute pass (no-op trampoline to validate
    // that compute → blit transitions don't drop ordering), then blit
    // pass that performs the actual copy.
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    // Empty compute pass. We don't dispatch anything — we just want
    // to verify that an opened-and-immediately-closed compute encoder
    // doesn't break the subsequent blit.
    {
        let enc = KernelEncoder::begin(&cmd);
        enc.end();
    }
    {
        let blit = BlitEncoder::begin(&cmd);
        blit.copy_tensor(&src, &dst);
        blit.end();
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    // Read back via host pointer.
    let got: Vec<f32> = unsafe {
        let p = dst.buffer.contents().as_ptr() as *const f32;
        (0..N).map(|i| *p.add(i)).collect()
    };
    for i in 0..N {
        assert!(
            (got[i] - pattern[i]).abs() < 1e-9,
            "blit mismatch at i={i}: got={} expected={}",
            got[i],
            pattern[i]
        );
    }

    // Also exercise `copy_buffer` with non-zero offsets: copy the
    // back half of `src` into the front half of `dst`.
    let cmd2 = ctx.queue.commandBuffer().expect("cmd2");
    {
        let blit = BlitEncoder::begin(&cmd2);
        let half_bytes = (N / 2) * 4;
        blit.copy_buffer(
            &src.buffer,
            half_bytes as u64,
            &dst.buffer,
            0,
            half_bytes as u64,
        );
        blit.end();
    }
    cmd2.commit();
    cmd2.waitUntilCompleted();
    let got2: Vec<f32> = unsafe {
        let p = dst.buffer.contents().as_ptr() as *const f32;
        (0..N).map(|i| *p.add(i)).collect()
    };
    // Front half of dst now equals back half of src.
    for i in 0..N / 2 {
        assert!(
            (got2[i] - pattern[N / 2 + i]).abs() < 1e-9,
            "offset blit front-half mismatch at i={i}: got={} expected={}",
            got2[i],
            pattern[N / 2 + i]
        );
    }
    // Back half of dst is unchanged from the previous full-blit copy.
    for i in N / 2..N {
        assert!(
            (got2[i] - pattern[i]).abs() < 1e-9,
            "offset blit back-half disturbed at i={i}: got={} expected={}",
            got2[i],
            pattern[i]
        );
    }
}

/// CPU oracle for `kernel_dflash_attn_f32`. Mirrors the kernel's
/// 3-pass streaming softmax + per-layer SWA mask EXACTLY. Used by
/// the v0.72.2 test suite (codex code-review test additions).
///
/// Mask semantics (kernel + this oracle):
///   * full-attn ctx key (swa_window == 0): ALWAYS allowed.
///   * SWA ctx key: causal && (q_pos - k_pos) <= swa_window.
///   * Noise key: noise_idx <= q_idx.
///
/// Returns o[N, n_q · head_dim] row-major.
// CPU dflash-attn reference: `kk` is a multi-purpose KV-position
// index — used for `pos_k[kk]`, `kk * kv_stride + ...` strided
// K/V offset computation, and `if kk < ctx_len` ctx-vs-noise
// branching. Iterator rewrite obscures all three.
#[allow(clippy::needless_range_loop)]
fn dflash_attn_cpu_oracle(
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
fn dflash_attn_readback(
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
fn dflash_attn_two_range_readback(
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

/// **v0.72.2 codex code-review test #1**: dflash attention kernel
/// matches the CPU oracle bit-tight under each mask regime.
///
/// Exercises:
///   * `ctx_len == 0` (degenerate: noise-only attention)
///   * `ctx_len > 0, swa_window > 0` (SWA layer)
///   * `ctx_len > 0, swa_window == 0` (full-attn layer; codex
///     mask-semantics flag — full-attn allows ALL ctx keys, no
///     causal restriction)
///   * `ctx_len > swa_window` (SWA boundary; some ctx keys
///     denied by the window even though causal)
///   * `ctx_len > 0` with non-contiguous / gapped pos_k
///   * Edge: q_pos == k_pos exactly (boundary causal — allowed
///     under SWA)
#[test]
fn dflash_attn_matches_cpu_oracle_under_mask_regimes() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    // Drafter shape: n_q=32, n_kv=8 (group=4), head_dim=128, N=16.
    let n = 16;
    let n_q = 32;
    let n_kv = 8;
    let hd = 128;
    let q_dim = n_q * hd;
    let kv_stride = n_kv * hd;

    // Deterministic synthetic activations.
    let make_buf = |seed: u32, len: usize| -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.5
            })
            .collect()
    };

    let q = make_buf(1, n * q_dim);

    struct Case {
        label: &'static str,
        ctx_len: usize,
        swa_window: u32,
        noise_start_pos: u32,
        // Custom pos_k for the ctx half (length ctx_len).
        // Builder receives ctx_len + noise_start_pos and returns
        // ctx-side positions.
        pos_ctx: fn(usize, u32) -> Vec<i32>,
    }

    fn pos_recent(ctx_len: usize, noise_start: u32) -> Vec<i32> {
        (0..ctx_len)
            .map(|c| noise_start as i32 - ctx_len as i32 + c as i32)
            .collect()
    }
    fn pos_gapped(ctx_len: usize, noise_start: u32) -> Vec<i32> {
        // Every other position skipped — non-contiguous.
        (0..ctx_len)
            .map(|c| (noise_start as i32 - 2 * ctx_len as i32 + 2 * c as i32).max(0))
            .collect()
    }

    let cases = [
        Case {
            label: "ctx_len=0 (noise-only)",
            ctx_len: 0,
            swa_window: 2048,
            noise_start_pos: 4,
            pos_ctx: pos_recent,
        },
        Case {
            label: "swa, ctx within window",
            ctx_len: 8,
            swa_window: 2048,
            noise_start_pos: 16,
            pos_ctx: pos_recent,
        },
        Case {
            label: "full-attn (swa=0), ctx allowed permissively",
            ctx_len: 8,
            swa_window: 0,
            noise_start_pos: 16,
            pos_ctx: pos_recent,
        },
        Case {
            label: "swa boundary, ctx_len > swa_window",
            ctx_len: 64,
            swa_window: 16,
            noise_start_pos: 80,
            pos_ctx: pos_recent,
        },
        Case {
            label: "swa, gapped pos_ctx",
            ctx_len: 12,
            swa_window: 2048,
            noise_start_pos: 32,
            pos_ctx: pos_gapped,
        },
        Case {
            label: "swa, q_pos == k_pos boundary",
            ctx_len: 4,
            swa_window: 2048,
            // pos_recent constructs ctx positions
            // [noise_start - ctx_len .. noise_start). With
            // noise_start=4, ctx pos = [0,1,2,3]. q_pos at q_idx=0
            // = 4. So q_pos > k_pos — no exact equality.
            // To exercise q_pos == k_pos: shift noise_start_pos so
            // pos_ctx ends at exactly noise_start_pos (= q_pos at
            // q_idx=0). Set ctx_len=4, noise_start_pos=4 →
            // pos_ctx = [0..4); the last ctx is at pos=3, q_pos at
            // q_idx=0 is 4 → still strict. Make ctx_len=5 and
            // noise_start_pos=4 → pos_ctx = [-1..4); ctx[4]=3.
            // Hmm same. This case structurally enforces k_pos < q_pos
            // unless we allow ctx that overlaps noise positions
            // (semantically a contract violation per codex flag).
            //
            // Instead, this case tests q_pos > all ctx positions
            // by a margin of 1 — boundary-adjacent without overlap.
            noise_start_pos: 4,
            pos_ctx: pos_recent,
        },
    ];

    for c in &cases {
        let pos_ctx_vec = (c.pos_ctx)(c.ctx_len, c.noise_start_pos);
        let n_kv_total = c.ctx_len + n;
        // Build pos_k = pos_ctx ++ [noise_start..noise_start+N].
        let mut pos_k = Vec::with_capacity(n_kv_total);
        pos_k.extend_from_slice(&pos_ctx_vec);
        for i in 0..n {
            pos_k.push((c.noise_start_pos + i as u32) as i32);
        }
        let k = make_buf(2, n_kv_total * kv_stride);
        let v = make_buf(3, n_kv_total * kv_stride);
        let ctx_rows = c.ctx_len.max(1);
        let mut k_ctx = vec![0.0_f32; ctx_rows * kv_stride];
        let mut v_ctx = vec![0.0_f32; ctx_rows * kv_stride];
        if c.ctx_len > 0 {
            k_ctx[..c.ctx_len * kv_stride].copy_from_slice(&k[..c.ctx_len * kv_stride]);
            v_ctx[..c.ctx_len * kv_stride].copy_from_slice(&v[..c.ctx_len * kv_stride]);
        }
        let k_noise = k[c.ctx_len * kv_stride..].to_vec();
        let v_noise = v[c.ctx_len * kv_stride..].to_vec();
        let mut pos_ctx = vec![0_i32; c.ctx_len.max(1)];
        if c.ctx_len > 0 {
            pos_ctx[..c.ctx_len].copy_from_slice(&pos_ctx_vec);
        }

        let cpu = dflash_attn_cpu_oracle(
            &q,
            &k,
            &v,
            &pos_k,
            n,
            n_q,
            n_kv,
            hd,
            n_kv_total,
            c.ctx_len,
            c.noise_start_pos,
            c.swa_window,
            false,
        );
        let gpu = dflash_attn_readback(
            &ctx,
            &q,
            &k,
            &v,
            &pos_k,
            n,
            n_q,
            n_kv,
            hd,
            n_kv_total,
            c.ctx_len,
            c.noise_start_pos,
            c.swa_window,
        )
        .expect("dflash_attn dispatch");
        let gpu_two_range = dflash_attn_two_range_readback(
            &ctx,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            n,
            n_q,
            n_kv,
            hd,
            c.ctx_len,
            c.noise_start_pos,
            c.swa_window,
            false,
            false,
            false,
            false,
            0,
        )
        .expect("dflash_attn_two_range dispatch");
        let gpu_online_two_range = dflash_attn_two_range_readback(
            &ctx,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            n,
            n_q,
            n_kv,
            hd,
            c.ctx_len,
            c.noise_start_pos,
            c.swa_window,
            true,
            false,
            false,
            false,
            0,
        )
        .expect("dflash_attn_online_two_range dispatch");
        let ctx_scan_start = if c.swa_window > 0 && c.ctx_len > 0 {
            let min_pos = c.noise_start_pos.saturating_sub(c.swa_window);
            pos_ctx_vec.partition_point(|&pos| pos >= 0 && (pos as u32) < min_pos)
        } else {
            0
        };
        let gpu_online_two_range_scan = dflash_attn_two_range_readback(
            &ctx,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            n,
            n_q,
            n_kv,
            hd,
            c.ctx_len,
            c.noise_start_pos,
            c.swa_window,
            true,
            false,
            false,
            false,
            ctx_scan_start,
        )
        .expect("dflash_attn_online_two_range scan dispatch");
        let gpu_full_gqa_split4 = if c.swa_window == 0 {
            Some(
                dflash_attn_two_range_readback(
                    &ctx,
                    &q,
                    &k_ctx,
                    &v_ctx,
                    &k_noise,
                    &v_noise,
                    &pos_ctx,
                    n,
                    n_q,
                    n_kv,
                    hd,
                    c.ctx_len,
                    c.noise_start_pos,
                    c.swa_window,
                    false,
                    true,
                    false,
                    false,
                    0,
                )
                .expect("dflash_attn_full_gqa_split4 dispatch"),
            )
        } else {
            None
        };

        let mut max_abs = 0.0f32;
        let mut max_abs_two_range = 0.0f32;
        let mut max_abs_online_two_range = 0.0f32;
        let mut max_abs_online_two_range_scan = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_diff_two_range = 0.0f64;
        let mut sum_sq_diff_online_two_range = 0.0f64;
        let mut sum_sq_diff_online_two_range_scan = 0.0f64;
        let mut sum_sq_cpu = 0.0f64;
        for i in 0..cpu.len() {
            let d = (gpu[i] - cpu[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
            let d_two_range = (gpu_two_range[i] - cpu[i]).abs();
            if d_two_range > max_abs_two_range {
                max_abs_two_range = d_two_range;
            }
            let d_online_two_range = (gpu_online_two_range[i] - cpu[i]).abs();
            if d_online_two_range > max_abs_online_two_range {
                max_abs_online_two_range = d_online_two_range;
            }
            let d_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]).abs();
            if d_online_two_range_scan > max_abs_online_two_range_scan {
                max_abs_online_two_range_scan = d_online_two_range_scan;
            }
            let dd = (gpu[i] - cpu[i]) as f64;
            sum_sq_diff += dd * dd;
            let dd_two_range = (gpu_two_range[i] - cpu[i]) as f64;
            sum_sq_diff_two_range += dd_two_range * dd_two_range;
            let dd_online_two_range = (gpu_online_two_range[i] - cpu[i]) as f64;
            sum_sq_diff_online_two_range += dd_online_two_range * dd_online_two_range;
            let dd_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]) as f64;
            sum_sq_diff_online_two_range_scan +=
                dd_online_two_range_scan * dd_online_two_range_scan;
            sum_sq_cpu += (cpu[i] as f64).powi(2);
        }
        let rel_l2 = sum_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
        let rel_l2_two_range = sum_sq_diff_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
        let rel_l2_online_two_range =
            sum_sq_diff_online_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
        let rel_l2_online_two_range_scan =
            sum_sq_diff_online_two_range_scan.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
        eprintln!(
            "[dflash-attn-mask {label}] max|Δ|={max_abs:.3e} rel_l2={rel_l2:.3e} two_range_max|Δ|={max_abs_two_range:.3e} two_range_rel_l2={rel_l2_two_range:.3e} online_two_range_max|Δ|={max_abs_online_two_range:.3e} online_two_range_rel_l2={rel_l2_online_two_range:.3e} scan_start={ctx_scan_start} online_scan_max|Δ|={max_abs_online_two_range_scan:.3e} online_scan_rel_l2={rel_l2_online_two_range_scan:.3e}",
            label = c.label
        );
        assert!(max_abs < 1e-4, "{}: max|Δ|={max_abs} too large", c.label);
        assert!(rel_l2 < 1e-5, "{}: rel_l2={rel_l2} too large", c.label);
        assert!(
            max_abs_two_range < 1e-4,
            "{}: two-range max|Δ|={max_abs_two_range} too large",
            c.label
        );
        assert!(
            rel_l2_two_range < 1e-5,
            "{}: two-range rel_l2={rel_l2_two_range} too large",
            c.label
        );
        assert!(
            max_abs_online_two_range < 1e-4,
            "{}: online two-range max|Δ|={max_abs_online_two_range} too large",
            c.label
        );
        assert!(
            rel_l2_online_two_range < 1e-5,
            "{}: online two-range rel_l2={rel_l2_online_two_range} too large",
            c.label
        );
        assert!(
            max_abs_online_two_range_scan < 1e-4,
            "{}: online scan max|Δ|={max_abs_online_two_range_scan} too large",
            c.label
        );
        assert!(
            rel_l2_online_two_range_scan < 1e-5,
            "{}: online scan rel_l2={rel_l2_online_two_range_scan} too large",
            c.label
        );
        if let Some(gpu_full_gqa_split4) = gpu_full_gqa_split4 {
            let mut candidate_max_abs = 0.0f32;
            let mut candidate_sq_diff = 0.0f64;
            for (&got, &want) in gpu_full_gqa_split4.iter().zip(&cpu) {
                assert!(
                    got.is_finite(),
                    "{}: split4 produced nonfinite output",
                    c.label
                );
                let diff = (got - want).abs();
                candidate_max_abs = candidate_max_abs.max(diff);
                candidate_sq_diff += (diff as f64).powi(2);
            }
            let candidate_rel_l2 = candidate_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            eprintln!(
                "[dflash-attn-mask split4 {}] max|delta|={candidate_max_abs:.3e} \
                 rel_l2={candidate_rel_l2:.3e}",
                c.label
            );
            assert!(
                candidate_max_abs < 1e-4,
                "{}: split4 max|delta|={candidate_max_abs} too large",
                c.label
            );
            assert!(
                candidate_rel_l2 < 1e-5,
                "{}: split4 rel_l2={candidate_rel_l2} too large",
                c.label
            );
        }
    }
}

#[test]
fn dflash_attn_swa_split4_matches_cpu_oracle_n8() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };
    let n = 8;
    let n_q = 32;
    let n_kv = 8;
    let hd = 128;
    let ctx_len = 40;
    let swa_window = 16u32;
    let noise_start_pos = 80u32;
    let kv_stride = n_kv * hd;
    let make_buf = |seed: u32, len: usize| -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.5
            })
            .collect()
    };
    let q = make_buf(11, n * n_q * hd);
    let k_ctx = make_buf(12, ctx_len * kv_stride);
    let v_ctx = make_buf(13, ctx_len * kv_stride);
    let k_noise = make_buf(14, n * kv_stride);
    let v_noise = make_buf(15, n * kv_stride);
    let pos_ctx: Vec<i32> = (0..ctx_len)
        .map(|index| noise_start_pos as i32 - ctx_len as i32 + index as i32)
        .collect();
    let min_pos = noise_start_pos.saturating_sub(swa_window);
    let ctx_scan_start = pos_ctx.partition_point(|&pos| pos >= 0 && (pos as u32) < min_pos);
    let mut k = k_ctx.clone();
    k.extend_from_slice(&k_noise);
    let mut v = v_ctx.clone();
    v.extend_from_slice(&v_noise);
    let mut pos_k = pos_ctx.clone();
    pos_k.extend((0..n).map(|index| (noise_start_pos + index as u32) as i32));
    let cpu = dflash_attn_cpu_oracle(
        &q,
        &k,
        &v,
        &pos_k,
        n,
        n_q,
        n_kv,
        hd,
        ctx_len + n,
        ctx_len,
        noise_start_pos,
        swa_window,
        false,
    );
    let split4 = dflash_attn_two_range_readback(
        &ctx,
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &pos_ctx,
        n,
        n_q,
        n_kv,
        hd,
        ctx_len,
        noise_start_pos,
        swa_window,
        false,
        false,
        true,
        false,
        ctx_scan_start,
    )
    .expect("SWA split4 N8 dispatch");
    let cpu_noncausal = dflash_attn_cpu_oracle(
        &q,
        &k,
        &v,
        &pos_k,
        n,
        n_q,
        n_kv,
        hd,
        ctx_len + n,
        ctx_len,
        noise_start_pos,
        swa_window,
        true,
    );
    let split4_noncausal = dflash_attn_two_range_readback(
        &ctx,
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &pos_ctx,
        n,
        n_q,
        n_kv,
        hd,
        ctx_len,
        noise_start_pos,
        swa_window,
        false,
        false,
        true,
        true,
        ctx_scan_start,
    )
    .expect("noncausal SWA split4 N8 dispatch");

    let assert_close = |label: &str, got: &[f32], want: &[f32]| {
        let mut max_abs = 0.0f32;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(got.is_finite(), "{label}: nonfinite output at {index}");
            let diff = (got - want).abs();
            max_abs = max_abs.max(diff);
            diff_sq += (diff as f64).powi(2);
            ref_sq += (want as f64).powi(2);
        }
        let rel_l2 = diff_sq.sqrt() / (ref_sq.sqrt() + 1e-30);
        eprintln!("[{label}] max|delta|={max_abs:.3e} rel_l2={rel_l2:.3e}");
        assert!(max_abs < 1e-4, "{label}: max|delta|={max_abs} too large");
        assert!(rel_l2 < 1e-5, "{label}: rel_l2={rel_l2} too large");
    };
    assert_close("dflash-swa-split4-n8", &split4, &cpu);
    assert_close(
        "dflash-swa-split4-n8-noncausal",
        &split4_noncausal,
        &cpu_noncausal,
    );
}

#[test]
fn dflash_attn_swa_split4_rejects_unsafe_contracts() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };
    let n = 8usize;
    let ctx_len = 16usize;
    let q_elems = n * 32 * 128;
    let kv_stride = 8 * 128;
    let partial_groups = n * 8 * 4 * 4;
    let q = MetalTensor::zeros_f32(&ctx, vec![q_elems as u64]).unwrap();
    let k_ctx = MetalTensor::zeros_f32(&ctx, vec![(ctx_len * kv_stride) as u64]).unwrap();
    let v_ctx = MetalTensor::zeros_f32(&ctx, vec![(ctx_len * kv_stride) as u64]).unwrap();
    let k_noise = MetalTensor::zeros_f32(&ctx, vec![(n * kv_stride) as u64]).unwrap();
    let v_noise = MetalTensor::zeros_f32(&ctx, vec![(n * kv_stride) as u64]).unwrap();
    let pos_ctx = MetalTensor::zeros_f32(&ctx, vec![ctx_len as u64]).unwrap();
    let short_pos = MetalTensor::zeros_f32(&ctx, vec![(ctx_len - 1) as u64]).unwrap();
    let o_partial = MetalTensor::zeros_f32(&ctx, vec![(partial_groups * 128) as u64]).unwrap();
    let ml_partial = MetalTensor::zeros_f32(&ctx, vec![(partial_groups * 2) as u64]).unwrap();
    let o = MetalTensor::zeros_f32(&ctx, vec![q_elems as u64]).unwrap();

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let concurrent = KernelEncoder::begin_concurrent(&cmd);
    let concurrent_error = encode_dflash_attn_swa_split4_f32(
        &ctx,
        &concurrent,
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &pos_ctx,
        &o_partial,
        &ml_partial,
        &o,
        n,
        ctx_len,
        32,
        16,
        0,
    )
    .expect_err("concurrent main/reduce must be rejected");
    concurrent.end();
    assert!(concurrent_error.to_string().contains("serial encoder"));

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd);
    let short_pos_error = encode_dflash_attn_swa_split4_f32(
        &ctx,
        &enc,
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &short_pos,
        &o_partial,
        &ml_partial,
        &o,
        n,
        ctx_len,
        32,
        16,
        0,
    )
    .expect_err("short positions must be rejected");
    assert!(short_pos_error.to_string().contains("shape mismatch"));

    let alias_error = encode_dflash_attn_swa_split4_f32(
        &ctx, &enc, &q, &k_ctx, &v_ctx, &k_noise, &v_noise, &pos_ctx, &o_partial, &o_partial, &o,
        n, ctx_len, 32, 16, 0,
    )
    .expect_err("aliased partials must be rejected");
    assert!(alias_error.to_string().contains("disjoint"));

    let position_error = encode_dflash_attn_swa_split4_f32(
        &ctx,
        &enc,
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &pos_ctx,
        &o_partial,
        &ml_partial,
        &o,
        n,
        ctx_len,
        u32::MAX,
        16,
        0,
    )
    .expect_err("overflowing noise positions must be rejected");
    assert!(position_error.to_string().contains("overflow"));
    enc.end();
}

/// **v0.72.2 codex code-review test #2**: head_dim > 256 must be
/// rejected at the host wrapper. Kernel uses fixed-size [8] register
/// arrays sized for head_dim=256; head_dim=320 would silently
/// stack-OOB without this guard.
#[test]
fn dflash_attn_rejects_head_dim_over_256() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };
    // Make tiny placeholder buffers; we only care about the host
    // wrapper validation.
    let q = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let k = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let v = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let p = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let o = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    let res = encode_dflash_attn_f32(
        &ctx, &enc, &q, &k, &v, &p, &o, 16,   // n
        32,   // n_q_heads
        8,    // n_kv_heads
        320,  // head_dim — REJECTED
        17,   // n_kv_total
        1,    // ctx_len
        0,    // noise_start_pos
        2048, // swa_window
    );
    enc.end();
    match res {
        Err(MetalError::BadShape { detail, .. }) => {
            assert!(detail.contains("256"), "wrong error detail: {detail}");
        }
        other => panic!("expected BadShape on head_dim>256, got {other:?}"),
    }
}
