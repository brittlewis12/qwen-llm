use super::*;
use crate::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use crate::deepseek_v4_census::PinnedDeepSeekV4AssetV1;
use crate::deepseek_v4_oracle::{
    CompressorState, DeepSeekV4IndexerFp4Row, INDEXER_FP4_ROW_BYTES, INDEXER_FP4_SCALE_BYTES,
    INDEXER_FP4_VALUE_BYTES, INDEXER_FP4_VALUES_PER_ROW, RopeDirection, RopeParameters,
    attention_fp8_nope_bf16_rope_roundtrip_in_place, grouped_low_rank_projection,
    hadamard_128_in_place, hyper_connection_head, hyper_connection_post, hyper_connection_pre,
    indexer_scores, mat_vec, pack_indexer_fp4_row, packed_indexer_scores, rms_norm,
    rope_tail_in_place, shared_kv_attention, shared_kv_projection, top_k_indices,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const LEGACY_CENSUS_MANIFEST: &str =
    include_str!("../../tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_census_v1.json");
#[cfg(feature = "dsv4-diagnostics")]
const CURRENT_CENSUS_MANIFEST: &str = include_str!(
    "../../tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_current_2026_08_04_census_v1.json"
);
const INDEXER_FP4_CONTRACT_FIXTURE: &str =
    include_str!("../../tests/fixtures/deepseek_v4_indexer_fp4_contract_v1.json");
const LEGACY_CHECKPOINT_CONTENT_ID: [u8; 32] = [
    0xaf, 0x65, 0xc3, 0x14, 0x59, 0xd1, 0xd2, 0x5e, 0xf9, 0xf3, 0xc7, 0x76, 0x6a, 0xf1, 0xf7, 0x41,
    0x57, 0xcb, 0x33, 0x0d, 0x9b, 0xde, 0xe5, 0x43, 0x2e, 0x32, 0x0f, 0xa1, 0xd8, 0xe4, 0x9e, 0x2a,
];

#[test]
fn model_residency_set_scope_is_exact() {
    let mut report = DeepSeekV4ResidencyReport {
        tensor_count: DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT,
        source_bytes: DEEPSEEK_V4_REAP_K160_SOURCE_BYTES,
        window_count: 4,
        window_bytes: 0,
        view_count: 0,
        unique_view_bytes: 0,
        logical_view_bytes: 0,
        alias_count: 0,
        alias_bytes: 0,
        fallback_count: 5,
        fallback_bytes: 0,
        resident_bytes: 0,
        page_size: 16 * 1024,
        max_buffer_length: 0,
        required_alignment: GGUF_BINDING_ALIGNMENT,
    };
    assert!(deepseek_v4_residency_set_scope_qualified(
        true,
        "Apple M4 Max",
        43,
        160,
        &report,
    ));
    for (enabled, device, layers, experts) in [
        (false, "Apple M4 Max", 43, 160),
        (true, "Apple M3 Max", 43, 160),
        (true, "Apple M4 Max", 42, 160),
        (true, "Apple M4 Max", 43, 216),
    ] {
        assert!(!deepseek_v4_residency_set_scope_qualified(
            enabled, device, layers, experts, &report,
        ));
    }

    let mut k216_report = report.clone();
    k216_report.source_bytes = DEEPSEEK_V4_REAP_K216_SOURCE_BYTES;
    assert!(deepseek_v4_residency_set_scope_qualified(
        true,
        "Apple M4 Max",
        43,
        216,
        &k216_report,
    ));

    let mut fresh_report = report.clone();
    fresh_report.source_bytes = DEEPSEEK_V4_FRESH_SOURCE_BYTES;
    assert!(deepseek_v4_residency_set_scope_qualified(
        true,
        "Apple M4 Max",
        43,
        256,
        &fresh_report,
    ));

    report.tensor_count -= 1;
    assert!(!deepseek_v4_residency_set_scope_qualified(
        true,
        "Apple M4 Max",
        43,
        160,
        &report,
    ));
    report.tensor_count += 1;
    report.source_bytes -= 1;
    assert!(!deepseek_v4_residency_set_scope_qualified(
        true,
        "Apple M4 Max",
        43,
        160,
        &report,
    ));
}

#[derive(serde::Deserialize)]
struct MetalFp4Fixture {
    invalid_rows: Vec<MetalFp4InvalidRowCase>,
    pack_rejections: Vec<MetalFp4PackRejectionCase>,
    rounding_cases: Vec<MetalFp4RoundingCase>,
    rows: Vec<MetalFp4RowCase>,
    scale_cases: Vec<MetalFp4ScaleCase>,
    score_cases: Vec<MetalFp4ScoreCase>,
}

#[derive(serde::Deserialize)]
struct MetalFp4RoundingCase {
    code: u8,
    input_bits: u32,
}

#[derive(serde::Deserialize)]
struct MetalFp4ScaleCase {
    code: u8,
    maximum_bits: u32,
}

#[derive(serde::Deserialize)]
struct MetalFp4RowCase {
    input_bits: Vec<u32>,
    name: String,
    packed_bytes: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct MetalFp4InvalidRowCase {
    error_category: String,
    mutations: Vec<MetalFp4ByteMutation>,
    name: String,
}

#[derive(serde::Deserialize)]
struct MetalFp4ByteMutation {
    byte_index: usize,
    byte_value: u8,
}

#[derive(serde::Deserialize)]
struct MetalFp4PackRejectionCase {
    dimension: usize,
    error_category: String,
    input_bits: u32,
    name: String,
}

#[derive(serde::Deserialize)]
struct MetalFp4ScoreCase {
    key_rows: Vec<Vec<u8>>,
    name: String,
    query_rows: Vec<Vec<u8>>,
    scaled_head_weight_bits: Vec<u32>,
    score_bits: Vec<u32>,
    top2: Vec<usize>,
}

fn metal_fp4_fixture() -> MetalFp4Fixture {
    serde_json::from_str(INDEXER_FP4_CONTRACT_FIXTURE).expect("valid strict indexer FP4 fixture")
}

fn fp4_row_from_bytes(bytes: &[u8]) -> DeepSeekV4IndexerFp4Row {
    DeepSeekV4IndexerFp4Row::from_bytes(
        bytes
            .try_into()
            .expect("indexer FP4 fixture row is 68 bytes"),
    )
    .expect("indexer FP4 fixture row is canonical")
}

fn split_fp4_rows(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
    let mut values = Vec::with_capacity(rows.len() * INDEXER_FP4_VALUE_BYTES);
    let mut scales = Vec::with_capacity(rows.len() * INDEXER_FP4_SCALE_BYTES);
    for row in rows {
        assert_eq!(row.len(), INDEXER_FP4_ROW_BYTES);
        values.extend_from_slice(&row[..INDEXER_FP4_VALUE_BYTES]);
        scales.extend_from_slice(&row[INDEXER_FP4_VALUE_BYTES..]);
    }
    (values, scales)
}

fn metal_context() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::NoDevice) => None,
        Err(error) => panic!("Metal context: {error}"),
    }
}

fn open_pinned_legacy_gguf(model_path: &Path) -> GgufFile {
    let gguf = GgufFile::open(model_path).expect("open legacy DS4 GGUF shards");
    PinnedDeepSeekV4AssetV1::parse(LEGACY_CENSUS_MANIFEST)
        .expect("parse legacy DS4 census")
        .validate_observed(&gguf)
        .expect("legacy DS4 schema and quant census match");
    let identity_cache_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/.qwen-dsv4-model-identity-v2");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve legacy DS4 ordered-content identity");
    assert_eq!(
        content.content_id, LEGACY_CHECKPOINT_CONTENT_ID,
        "DSV4_LEGACY_MODEL does not match the pinned 2026-07-31 asset"
    );
    gguf
}

#[cfg(feature = "dsv4-diagnostics")]
fn open_pinned_current_gguf(model_path: &Path) -> (GgufFile, DeepSeekV4ModelContentId) {
    let gguf = GgufFile::open(model_path).expect("open current DS4 GGUF shards");
    PinnedDeepSeekV4AssetV1::parse(CURRENT_CENSUS_MANIFEST)
        .expect("parse current DS4 census")
        .validate_observed(&gguf)
        .expect("current DS4 schema and quant census match");
    let identity_cache_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/.qwen-dsv4-model-identity-v2");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve current DS4 ordered-content identity");
    (gguf, DeepSeekV4ModelContentId::new(content.content_id))
}

fn offset_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let prefix = 16usize;
    let mut bytes = vec![0xA5u8; prefix];
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    bytes.extend_from_slice(&[0x5Au8; 20]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("offset F32 buffer"),
        offset: prefix as u64,
        shape,
        dtype: GgmlType::F32,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn offset_f16(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let prefix = 16usize;
    let mut bytes = vec![0xA5u8; prefix];
    bytes.extend(
        values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_ne_bytes()),
    );
    bytes.extend_from_slice(&[0x5Au8; 20]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("offset F16 buffer"),
        offset: prefix as u64,
        shape,
        dtype: GgmlType::F16,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn offset_i8(ctx: &MetalContext, values: &[u8], shape: Vec<u64>) -> MetalTensor {
    let prefix = 13usize;
    let mut bytes = vec![0x5Au8; prefix];
    bytes.extend_from_slice(values);
    bytes.extend_from_slice(&[0xA5u8; 19]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("offset raw-byte buffer"),
        offset: prefix as u64,
        shape,
        dtype: GgmlType::I8,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn encode_q6_k_test_block(d: f32, seed: usize) -> [u8; 210] {
    let mut block = [0u8; 210];
    for scale_index in 0..16 {
        let scale = ((seed + scale_index * 3) % 15) as i8 - 7;
        block[192 + scale_index] = scale as u8;
    }
    block[208..210].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for i in 0..256 {
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
    }
    block
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    unsafe {
        let pointer = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
    }
}

fn offset_weight(
    ctx: &MetalContext,
    bytes: &[u8],
    shape: Vec<u64>,
    dtype: GgmlType,
) -> MetalTensor {
    let prefix = 16usize;
    let mut backing = vec![0xA5u8; prefix];
    backing.extend_from_slice(bytes);
    backing.extend_from_slice(&[0x5Au8; 32]);
    MetalTensor {
        buffer: ctx.buffer_from(&backing).expect("offset weight buffer"),
        offset: prefix as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn q8_test_weight_bytes(n_in: usize, n_out: usize, salt: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n_out * (n_in / 32) * 34);
    for row in 0..n_out {
        for block in 0..n_in / 32 {
            let scale =
                half::f16::from_f32(0.0013 + ((row * 13 + block * 7 + salt) % 31) as f32 * 0.00019);
            bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            for index in 0..32 {
                bytes.push(((row * 29 + block * 17 + index * 23 + salt * 11) % 255) as u8);
            }
        }
    }
    bytes
}

fn prepare_test_weight(
    ctx: &MetalContext,
    n_in: usize,
    n_out: usize,
    dtype: GgmlType,
    salt: usize,
) -> MetalTensor {
    let bytes = match dtype {
        GgmlType::Q8_0 => q8_test_weight_bytes(n_in, n_out, salt),
        GgmlType::Q6_K => {
            let mut bytes = Vec::with_capacity(n_out * (n_in / 256) * 210);
            for row in 0..n_out {
                for block in 0..n_in / 256 {
                    bytes.extend_from_slice(&encode_q6_k_test_block(
                        0.0017 + ((row * 5 + block * 3 + salt) % 17) as f32 * 0.00023,
                        row * 37 + block * 19 + salt,
                    ));
                }
            }
            bytes
        }
        _ => panic!("unsupported prepare test weight dtype {dtype:?}"),
    };
    offset_weight(ctx, &bytes, vec![n_in as u64, n_out as u64], dtype)
}

fn assert_f32_tensor_bits_eq(label: &str, left: &MetalTensor, right: &MetalTensor) {
    let left = read_f32(left);
    let right = read_f32(right);
    assert_eq!(left.len(), right.len(), "{label} length");
    if let Some((index, (&left, &right))) = left
        .iter()
        .zip(&right)
        .enumerate()
        .find(|(_, (left, right))| left.to_bits() != right.to_bits())
    {
        panic!(
            "{label} mismatch at {index}: left={left:?} ({:08x}) right={right:?} ({:08x})",
            left.to_bits(),
            right.to_bits(),
        );
    }
}

#[test]
fn decode_prepare_projection_pairs_match_composed_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }

    for q_dtype in [GgmlType::Q8_0, GgmlType::Q6_K] {
        for (n_in, q_out, kv_out) in [(512usize, 67usize, 129usize), (4_096, 1_024, 512)] {
            let q_weight = prepare_test_weight(&ctx, n_in, q_out, q_dtype, 5);
            let kv_weight = prepare_test_weight(&ctx, n_in, kv_out, GgmlType::Q8_0, 23);
            let input_values = (0..n_in)
                .map(|index| ((index * 43 + index / 7 + 11) % 503) as f32 * 0.0091 - 2.27)
                .collect::<Vec<_>>();
            let input = offset_f32(&ctx, &input_values, vec![n_in as u64]);
            let composed_q = offset_f32(&ctx, &vec![0.0; q_out], vec![q_out as u64]);
            let composed_kv = offset_f32(&ctx, &vec![0.0; kv_out], vec![kv_out as u64]);
            let paired_q = offset_f32(&ctx, &vec![f32::NAN; q_out], vec![q_out as u64]);
            let paired_kv = offset_f32(&ctx, &vec![f32::NEG_INFINITY; kv_out], vec![kv_out as u64]);

            let _trace = crate::metal::kernel_trace_begin();
            let command = ctx
                .queue
                .commandBuffer()
                .expect("prepare projection command");
            let encoder = KernelEncoder::begin(&command);
            encode_projection(
                &ctx,
                &encoder,
                &q_weight,
                &input,
                &composed_q,
                n_in,
                q_out,
                "composed Q A differential",
            )
            .unwrap();
            encode_projection(
                &ctx,
                &encoder,
                &kv_weight,
                &input,
                &composed_kv,
                n_in,
                kv_out,
                "composed KV differential",
            )
            .unwrap();
            let composed_trace = crate::metal::kernel_trace_take_delta();
            encode_ds4_prepare_projection_pair(
                &ctx, &encoder, &q_weight, &kv_weight, &input, &paired_q, &paired_kv, n_in, q_out,
                kv_out,
            )
            .unwrap();
            let paired_trace = crate::metal::kernel_trace_take_delta();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "paired prepare projection command failed: {:?}",
                command.error()
            );

            assert_eq!(composed_trace.dispatches, 2);
            assert_eq!(paired_trace.dispatches, 1);
            assert_f32_tensor_bits_eq(
                &format!("{q_dtype:?} Q A {n_in}x{q_out}"),
                &composed_q,
                &paired_q,
            );
            assert_f32_tensor_bits_eq(&format!("Q8 KV {n_in}x{kv_out}"), &composed_kv, &paired_kv);
            assert!(read_f32(&paired_q).iter().any(|&value| value != 0.0));
            assert!(read_f32(&paired_kv).iter().any(|&value| value != 0.0));
        }
    }

    const N_IN: usize = 512;
    const N_OUT: usize = 8;
    let q_weight = prepare_test_weight(&ctx, N_IN, N_OUT, GgmlType::Q8_0, 7);
    let kv_weight = prepare_test_weight(&ctx, N_IN, N_OUT, GgmlType::Q8_0, 13);
    let input = offset_f32(&ctx, &[0.25; N_IN], vec![N_IN as u64]);
    let outputs = offset_f32(&ctx, &[0.0; N_OUT * 2], vec![(N_OUT * 2) as u64]);
    let q_output = outputs.view_subrange(0, vec![N_OUT as u64]);
    let overlapping_kv = outputs.view_subrange(4, vec![N_OUT as u64]);
    let adjacent_kv = outputs.view_subrange(N_OUT as u64, vec![N_OUT as u64]);
    let _trace = crate::metal::kernel_trace_begin();
    let command = ctx.queue.commandBuffer().expect("prepare overlap command");
    let encoder = KernelEncoder::begin(&command);
    let error = encode_ds4_prepare_projection_pair(
        &ctx,
        &encoder,
        &q_weight,
        &kv_weight,
        &input,
        &q_output,
        &overlapping_kv,
        N_IN,
        N_OUT,
        N_OUT,
    )
    .expect_err("partially overlapping projection outputs must fail closed");
    assert!(format!("{error}").contains("distinct F32 outputs"));
    assert_eq!(crate::metal::kernel_trace_take_delta().dispatches, 0);
    encode_ds4_prepare_projection_pair(
        &ctx,
        &encoder,
        &q_weight,
        &kv_weight,
        &input,
        &q_output,
        &adjacent_kv,
        N_IN,
        N_OUT,
        N_OUT,
    )
    .expect("adjacent projection outputs do not overlap");
    assert_eq!(crate::metal::kernel_trace_take_delta().dispatches, 1);
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
}

#[test]
fn decode_prepare_pair_routing_falls_back_for_ineligible_weights() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 512,
        q_lora_rank: 256,
        head_count: 2,
        head_dim: 128,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let mut scratch = DeepSeekV4PositionZeroAttentionScratch::new(&ctx, config).unwrap();
    scratch.set_prepare_test_policy(DeepSeekV4PrepareTestPolicy::Paired);
    let q8_q = prepare_test_weight(&ctx, 512, 256, GgmlType::Q8_0, 3);
    let q6_q = prepare_test_weight(&ctx, 512, 256, GgmlType::Q6_K, 5);
    let q8_kv = prepare_test_weight(&ctx, 512, 128, GgmlType::Q8_0, 7);
    let f32_q = offset_f32(&ctx, &[0.0; 512 * 256], vec![512, 256]);
    let f32_kv = offset_f32(&ctx, &[0.0; 512 * 128], vec![512, 128]);

    let eligible_capabilities = scratch.paired_prepare_capabilities;
    assert_eq!(
        scratch.use_paired_prepare(&q8_q, &q8_kv),
        eligible_capabilities.supports(GgmlType::Q8_0) && crate::metal::mat_vec_q8_0_lcpp_enabled()
    );
    assert_eq!(
        scratch.use_paired_prepare(&q6_q, &q8_kv),
        eligible_capabilities.supports(GgmlType::Q6_K) && crate::metal::mat_vec_q8_0_lcpp_enabled()
    );
    assert!(!scratch.use_paired_prepare(&f32_q, &q8_kv));
    assert!(!scratch.use_paired_prepare(&q8_q, &f32_kv));

    scratch.paired_prepare_capabilities = DeepSeekV4PairedPrepareCapabilities::default();
    assert!(!scratch.use_paired_prepare(&q8_q, &q8_kv));
    assert!(!scratch.use_paired_prepare(&q6_q, &q8_kv));
}

#[test]
fn decode_prepare_norm_pair_matches_composed_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const EPS: f32 = 1.0e-6;

    for (q_dim, kv_dim) in [(67usize, 129usize), (1_024, 512)] {
        let values = |n: usize, salt: usize| {
            (0..n)
                .map(|index| match index % 17 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(1),
                    3 => -f32::from_bits(1),
                    _ => ((index * 31 + salt) % 257) as f32 * 0.017 - 2.11,
                })
                .collect::<Vec<_>>()
        };
        let weights = |n: usize, salt: usize| {
            (0..n)
                .map(|index| 0.37 + ((index * 19 + salt) % 113) as f32 * 0.011)
                .collect::<Vec<_>>()
        };
        let q_input = offset_f32(&ctx, &values(q_dim, 3), vec![q_dim as u64]);
        let q_weight = offset_f32(&ctx, &weights(q_dim, 5), vec![q_dim as u64]);
        let kv_input = offset_f32(&ctx, &values(kv_dim, 17), vec![kv_dim as u64]);
        let kv_weight = offset_f32(&ctx, &weights(kv_dim, 29), vec![kv_dim as u64]);
        let composed_q = offset_f32(&ctx, &vec![0.0; q_dim], vec![q_dim as u64]);
        let composed_kv = offset_f32(&ctx, &vec![0.0; kv_dim], vec![kv_dim as u64]);
        let paired_q = offset_f32(&ctx, &vec![f32::NAN; q_dim], vec![q_dim as u64]);
        let paired_kv = offset_f32(&ctx, &vec![f32::NAN; kv_dim], vec![kv_dim as u64]);

        let _trace = crate::metal::kernel_trace_begin();
        let command = ctx.queue.commandBuffer().expect("prepare norm command");
        let encoder = KernelEncoder::begin(&command);
        encode_rms_norm_mul_f32(&ctx, &encoder, &q_input, &q_weight, &composed_q, EPS).unwrap();
        encode_rms_norm_mul_f32(&ctx, &encoder, &kv_input, &kv_weight, &composed_kv, EPS).unwrap();
        let composed_trace = crate::metal::kernel_trace_take_delta();
        encode_ds4_prepare_norm_pair(
            &ctx, &encoder, &q_input, &q_weight, &paired_q, &kv_input, &kv_weight, &paired_kv, EPS,
        )
        .unwrap();
        let paired_trace = crate::metal::kernel_trace_take_delta();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "paired prepare norm command failed: {:?}",
            command.error()
        );

        assert_eq!(composed_trace.dispatches, 2);
        assert_eq!(paired_trace.dispatches, 1);
        assert_f32_tensor_bits_eq("paired Q RMSNorm", &composed_q, &paired_q);
        assert_f32_tensor_bits_eq("paired KV RMSNorm", &composed_kv, &paired_kv);
    }

    let q_input = offset_f32(&ctx, &[0.25; 8], vec![8]);
    let q_weight = offset_f32(&ctx, &[1.0; 8], vec![8]);
    let kv_input = offset_f32(&ctx, &[0.5; 8], vec![8]);
    let kv_weight = offset_f32(&ctx, &[0.75; 8], vec![8]);
    let outputs = offset_f32(&ctx, &[0.0; 16], vec![16]);
    let q_output = outputs.view_subrange(0, vec![8]);
    let kv_output = outputs.view_subrange(4, vec![8]);
    let command = ctx
        .queue
        .commandBuffer()
        .expect("prepare norm overlap command");
    let encoder = KernelEncoder::begin(&command);
    let error = encode_ds4_prepare_norm_pair(
        &ctx, &encoder, &q_input, &q_weight, &q_output, &kv_input, &kv_weight, &kv_output, EPS,
    )
    .expect_err("partially overlapping norm outputs must fail closed");
    assert!(format!("{error}").contains("distinct F32 input/output rows"));
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
}

#[test]
fn decode_prepare_rope_pair_matches_composed_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 128;
    const Q_HEADS: usize = 3;
    let q_values = (0..HEAD_DIM * Q_HEADS)
        .map(|index| ((index * 37 + 11) % 401) as f32 * 0.0061 - 1.23)
        .collect::<Vec<_>>();
    let kv_values = (0..HEAD_DIM)
        .map(|index| ((index * 29 + 7) % 197) as f32 * 0.0093 - 0.91)
        .collect::<Vec<_>>();
    let ropes = [
        DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 10_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 0.0,
            beta_slow: 0.0,
        },
        DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 160_000.0,
            scaling_factor: 16.0,
            original_context_length: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    ];
    let positions = [
        0u32, 1, 127, 128, 129, 2_051, 2_052, 3_071, 65_535, 65_536, 1_048_575,
    ];
    let mut cases = Vec::new();
    let _trace = crate::metal::kernel_trace_begin();
    let command = ctx.queue.commandBuffer().expect("prepare RoPE command");
    let encoder = KernelEncoder::begin(&command);
    for (rope_index, rope) in ropes.into_iter().enumerate() {
        for position in positions {
            let composed_q = offset_f32(&ctx, &q_values, vec![HEAD_DIM as u64, Q_HEADS as u64]);
            let composed_kv = offset_f32(&ctx, &kv_values, vec![HEAD_DIM as u64]);
            let paired_q = offset_f32(&ctx, &q_values, vec![HEAD_DIM as u64, Q_HEADS as u64]);
            let paired_kv = offset_f32(&ctx, &kv_values, vec![HEAD_DIM as u64]);
            encode_ds4_rope_tail_adjacent_in_place(
                &ctx,
                &encoder,
                &composed_q,
                position,
                rope,
                false,
            )
            .unwrap();
            encode_ds4_rope_tail_adjacent_in_place(
                &ctx,
                &encoder,
                &composed_kv,
                position,
                rope,
                false,
            )
            .unwrap();
            let composed_trace = crate::metal::kernel_trace_take_delta();
            encode_ds4_rope_pair_in_place(&ctx, &encoder, &paired_q, &paired_kv, position, rope)
                .unwrap();
            let paired_trace = crate::metal::kernel_trace_take_delta();
            assert_eq!(composed_trace.dispatches, u64::from(position != 0) * 2);
            assert_eq!(paired_trace.dispatches, u64::from(position != 0));
            cases.push((
                rope_index,
                position,
                composed_q,
                composed_kv,
                paired_q,
                paired_kv,
            ));
        }
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "paired prepare RoPE command failed: {:?}",
        command.error()
    );

    for (rope_index, position, composed_q, composed_kv, paired_q, paired_kv) in cases {
        assert_f32_tensor_bits_eq(
            &format!("paired Q RoPE rope={rope_index} position={position}"),
            &composed_q,
            &paired_q,
        );
        assert_f32_tensor_bits_eq(
            &format!("paired KV RoPE rope={rope_index} position={position}"),
            &composed_kv,
            &paired_kv,
        );
        if position != 0 {
            assert!(
                read_f32(&paired_q)
                    .iter()
                    .zip(&q_values)
                    .any(|(&actual, &initial)| actual.to_bits() != initial.to_bits()),
                "nonzero RoPE case must rotate values"
            );
        }
    }

    let overlapping = offset_f32(&ctx, &[0.0; HEAD_DIM * 2], vec![(HEAD_DIM * 2) as u64]);
    let q = overlapping.view_subrange(0, vec![HEAD_DIM as u64]);
    let kv = overlapping.view_subrange(64, vec![HEAD_DIM as u64]);
    let command = ctx
        .queue
        .commandBuffer()
        .expect("prepare RoPE overlap command");
    let encoder = KernelEncoder::begin(&command);
    let error = encode_ds4_rope_pair_in_place(&ctx, &encoder, &q, &kv, 1, ropes[0])
        .expect_err("partially overlapping RoPE tensors must fail closed");
    assert!(format!("{error}").contains("distinct complete Q/KV head sets"));
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
}

#[test]
fn decode_prepare_paired_matches_composed_path_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }
    let config = deepseek_v4_session_attention_config();
    let c = config;
    let query_width = c.head_count * c.head_dim;
    let input_values = (0..c.hidden_size)
        .map(|index| ((index * 41 + 13) % 401) as f32 * 0.0087 - 1.73)
        .collect::<Vec<_>>();
    let attention_norm_values = (0..c.hidden_size)
        .map(|index| 0.43 + ((index * 17 + 3) % 127) as f32 * 0.0091)
        .collect::<Vec<_>>();
    let q_a_norm_values = (0..c.q_lora_rank)
        .map(|index| 0.51 + ((index * 23 + 7) % 113) as f32 * 0.0083)
        .collect::<Vec<_>>();
    let kv_norm_values = (0..c.head_dim)
        .map(|index| 0.61 + ((index * 29 + 11) % 97) as f32 * 0.0077)
        .collect::<Vec<_>>();
    let input = offset_f32(&ctx, &input_values, vec![c.hidden_size as u64]);
    let attention_norm = offset_f32(&ctx, &attention_norm_values, vec![c.hidden_size as u64]);
    let q_a_norm = offset_f32(&ctx, &q_a_norm_values, vec![c.q_lora_rank as u64]);
    let kv_norm = offset_f32(&ctx, &kv_norm_values, vec![c.head_dim as u64]);
    let q_b = prepare_test_weight(&ctx, c.q_lora_rank, query_width, GgmlType::Q8_0, 31);
    let kv_weight = prepare_test_weight(&ctx, c.hidden_size, c.head_dim, GgmlType::Q8_0, 47);
    let local_rope = DeepSeekV4RopeParameters {
        rotary_dim: c.rotary_dim,
        theta: 10_000.0,
        scaling_factor: 1.0,
        original_context_length: 0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };
    let yarn_rope = DeepSeekV4RopeParameters {
        rotary_dim: c.rotary_dim,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };

    for q_dtype in [GgmlType::Q8_0, GgmlType::Q6_K] {
        let q_a = prepare_test_weight(&ctx, c.hidden_size, c.q_lora_rank, q_dtype, 19);
        for (position, rope) in [(0u32, local_rope), (65_535, yarn_rope)] {
            let mut composed = DeepSeekV4PositionZeroAttentionScratch::new(&ctx, config).unwrap();
            composed.set_prepare_test_policy(DeepSeekV4PrepareTestPolicy::Composed);
            let mut paired = DeepSeekV4PositionZeroAttentionScratch::new(&ctx, config).unwrap();
            paired.set_prepare_test_policy(DeepSeekV4PrepareTestPolicy::Paired);
            let composed_cache = MetalTensor::zeros_f16(
                &ctx,
                vec![c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            )
            .unwrap();
            let paired_cache = MetalTensor::zeros_f16(
                &ctx,
                vec![c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            )
            .unwrap();
            const CACHE_SENTINEL_BYTE: u8 = 0x5A;
            const CACHE_SENTINEL_BITS: u16 = 0x5A5A;
            fill_tensor_bytes(&composed_cache, CACHE_SENTINEL_BYTE);
            fill_tensor_bytes(&paired_cache, CACHE_SENTINEL_BYTE);

            let run = |scratch: &DeepSeekV4PositionZeroAttentionScratch,
                       raw_cache: &MetalTensor,
                       label: &str| {
                let _trace = crate::metal::kernel_trace_begin();
                let command = ctx.queue.commandBuffer().expect("full prepare command");
                let encoder = KernelEncoder::begin(&command);
                scratch
                    .encode_prepare_local_f16(
                        &ctx,
                        &encoder,
                        &input,
                        &attention_norm,
                        &q_a,
                        &q_a_norm,
                        &q_b,
                        &kv_weight,
                        &kv_norm,
                        raw_cache,
                        position,
                        rope,
                        1.0e-6,
                    )
                    .unwrap();
                let trace = crate::metal::kernel_trace_snapshot();
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                assert!(
                    command.error().is_none(),
                    "{label} prepare command failed: {:?}",
                    command.error()
                );
                trace
            };
            let composed_trace = run(&composed, &composed_cache, "composed");
            let paired_trace = run(&paired, &paired_cache, "paired");
            let (expected_composed, expected_paired) = if position == 0 { (8, 6) } else { (10, 7) };
            assert_eq!(composed_trace.dispatches, expected_composed);
            assert_eq!(paired_trace.dispatches, expected_paired);

            let label = format!("{q_dtype:?} position={position}");
            for (name, left, right) in [
                (
                    "normalized input",
                    &composed.normalized_input,
                    &paired.normalized_input,
                ),
                ("Q LoRA raw", &composed.q_lora_raw, &paired.q_lora_raw),
                ("Q LoRA", &composed.q_lora, &paired.q_lora),
                ("queries raw", &composed.queries_raw, &paired.queries_raw),
                ("queries", &composed.queries, &paired.queries),
                ("KV raw", &composed.kv_raw, &paired.kv_raw),
                ("KV", &composed.kv, &paired.kv),
            ] {
                assert_f32_tensor_bits_eq(&format!("{label} {name}"), left, right);
            }
            let composed_cache_bits = read_f16_bits(&composed_cache);
            let paired_cache_bits = read_f16_bits(&paired_cache);
            assert_eq!(
                composed_cache_bits, paired_cache_bits,
                "{label} F16 publication differs"
            );
            let slot = position as usize % DEEPSEEK_V4_LOCAL_WINDOW;
            let row_start = slot * c.head_dim;
            let row_end = row_start + c.head_dim;
            let expected_row = read_f32(&paired.kv)
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                &paired_cache_bits[row_start..row_end],
                expected_row,
                "{label} F16 publication must round the authoritative KV row"
            );
            assert!(expected_row.iter().any(|&bits| bits != CACHE_SENTINEL_BITS));
            assert!(
                paired_cache_bits[..row_start]
                    .iter()
                    .chain(&paired_cache_bits[row_end..])
                    .all(|&bits| bits == CACHE_SENTINEL_BITS),
                "{label} F16 publication modified an untouched cache slot"
            );
        }
    }
}

#[test]
fn decode_grouped_output_gemv_matches_singleton_loop_bitwise() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }
    if !deepseek_v4_decode_compressor_fused_enabled() {
        return;
    }
    for (n_in, n_out, n_groups) in [(4_096usize, 1_024usize, 8usize), (64, 5, 3)] {
        let blocks_per_row = n_in / 32;
        let row_bytes = blocks_per_row * 34;
        let group_bytes = n_out * row_bytes;
        let mut bytes = vec![0u8; n_groups * group_bytes];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for (index, block) in bytes.chunks_mut(34).enumerate() {
            let scale = half::f16::from_f32(0.0035 + (index % 17) as f32 * 0.0004);
            block[..2].copy_from_slice(&scale.to_le_bytes());
            for quant in block[2..].iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *quant = (state >> 33) as u8;
            }
        }
        let weight = MetalTensor::from_bytes(
            &ctx,
            &bytes,
            vec![n_in as u64, (n_groups * n_out) as u64],
            GgmlType::Q8_0,
        )
        .expect("grouped Q8 weight");
        let x_values = (0..n_groups * n_in)
            .map(|i| ((i * 31 + 7) % 211) as f32 * 0.013 - 1.31)
            .collect::<Vec<_>>();
        let x = offset_f32(&ctx, &x_values, vec![(n_groups * n_in) as u64]);
        let y_loop = offset_f32(
            &ctx,
            &vec![0.0; n_groups * n_out],
            vec![(n_groups * n_out) as u64],
        );
        let y_grouped = offset_f32(
            &ctx,
            &vec![0.0; n_groups * n_out],
            vec![(n_groups * n_out) as u64],
        );

        let command = ctx.queue.commandBuffer().expect("grouped GEMV command");
        let encoder = KernelEncoder::begin(&command);
        for group in 0..n_groups {
            let weight_view =
                group_weight_view(&weight, n_in, n_out, group).expect("group weight view");
            let x_view = x.view_subrange((group * n_in) as u64, vec![n_in as u64]);
            let y_view = y_loop.view_subrange((group * n_out) as u64, vec![n_out as u64]);
            crate::metal::encode_mat_vec_q8_0_f32(
                &ctx,
                &encoder,
                &weight_view,
                &x_view,
                &y_view,
                n_in,
                n_out,
            )
            .expect("singleton GEMV");
        }
        crate::metal::encode_mat_vec_q8_0_grouped_f32(
            &ctx, &encoder, &weight, &x, &y_grouped, n_in, n_out, n_groups,
        )
        .expect("grouped GEMV");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "grouped GEMV command failed: {:?}",
            command.error()
        );

        let singleton = read_f32(&y_loop);
        let grouped = read_f32(&y_grouped);
        assert_eq!(singleton.len(), grouped.len());
        assert!(
            singleton
                .iter()
                .zip(&grouped)
                .all(|(&left, &right)| left.to_bits() == right.to_bits()),
            "grouped output must match the singleton loop bit-for-bit at n_in={n_in} n_out={n_out} groups={n_groups}"
        );
        assert!(
            singleton.iter().any(|&value| value != 0.0),
            "differential must exercise nonzero outputs"
        );
    }

    let bad = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let x = offset_f32(&ctx, &[0.0; 192], vec![192]);
    let y = offset_f32(&ctx, &[0.0; 15], vec![15]);
    let command = ctx.queue.commandBuffer().expect("rejection command");
    let encoder = KernelEncoder::begin(&command);
    let error =
        crate::metal::encode_mat_vec_q8_0_grouped_f32(&ctx, &encoder, &bad, &x, &y, 64, 5, 3)
            .expect_err("F32 weight must fail closed");
    assert!(
        format!("{error}").contains("mat_vec_q8_0_grouped"),
        "{error}"
    );
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
}

#[test]
fn decode_shared_swiglu_fusions_match_composed_paths_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const N_IN: usize = 512;
    const N_OUT: usize = 67;
    const CLAMP: f32 = 0.375;

    for dtype in [GgmlType::Q8_0, GgmlType::Q6_K] {
        let mut gate_bytes = Vec::new();
        let mut up_bytes = Vec::new();
        match dtype {
            GgmlType::Q8_0 => {
                for row in 0..N_OUT {
                    for block in 0..N_IN / 32 {
                        for (bytes, salt) in [(&mut gate_bytes, 11usize), (&mut up_bytes, 29)] {
                            let scale = half::f16::from_f32(
                                0.004 + ((row * 7 + block * 3 + salt) % 13) as f32 * 0.0007,
                            );
                            bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                            for index in 0..32 {
                                bytes.push(
                                    ((row * 19 + block * 23 + index * 17 + salt) % 255) as u8,
                                );
                            }
                        }
                    }
                }
            }
            GgmlType::Q6_K => {
                for row in 0..N_OUT {
                    for block in 0..N_IN / 256 {
                        let gate = encode_q6_k_test_block(
                            0.0025 + ((row + block) % 7) as f32 * 0.0004,
                            row * 31 + block * 7 + 5,
                        );
                        let up = encode_q6_k_test_block(
                            0.003 + ((row * 3 + block) % 5) as f32 * 0.0005,
                            row * 37 + block * 11 + 17,
                        );
                        gate_bytes.extend_from_slice(&gate);
                        up_bytes.extend_from_slice(&up);
                    }
                }
            }
            _ => unreachable!(),
        }
        let gate_weight =
            MetalTensor::from_bytes(&ctx, &gate_bytes, vec![N_IN as u64, N_OUT as u64], dtype)
                .expect("shared gate weight");
        let up_weight =
            MetalTensor::from_bytes(&ctx, &up_bytes, vec![N_IN as u64, N_OUT as u64], dtype)
                .expect("shared up weight");
        let x_values = (0..N_IN)
            .map(|index| ((index * 41 + 13) % 257) as f32 * 0.021 - 2.65)
            .collect::<Vec<_>>();
        let x = offset_f32(&ctx, &x_values, vec![N_IN as u64]);
        let gate = offset_f32(&ctx, &vec![0.0; N_OUT], vec![N_OUT as u64]);
        let up = offset_f32(&ctx, &vec![0.0; N_OUT], vec![N_OUT as u64]);
        let composed = offset_f32(&ctx, &vec![0.0; N_OUT], vec![N_OUT as u64]);
        let fused = offset_f32(&ctx, &vec![0.0; N_OUT], vec![N_OUT as u64]);

        let command = ctx.queue.commandBuffer().expect("shared fusion command");
        let encoder = KernelEncoder::begin(&command);
        encode_projection(
            &ctx,
            &encoder,
            &gate_weight,
            &x,
            &gate,
            N_IN,
            N_OUT,
            "shared gate differential",
        )
        .unwrap();
        encode_projection(
            &ctx,
            &encoder,
            &up_weight,
            &x,
            &up,
            N_IN,
            N_OUT,
            "shared up differential",
        )
        .unwrap();
        encode_ds4_clamped_swiglu(&ctx, &encoder, &gate, &up, &composed, CLAMP).unwrap();
        let fused_target = if dtype == GgmlType::Q6_K {
            offset_f32(&ctx, &vec![0.0; N_OUT * 3], vec![(N_OUT * 3) as u64])
        } else {
            fused
        };
        match dtype {
            GgmlType::Q8_0 => crate::metal::encode_ds4_shared_swiglu_q8_0_f32(
                &ctx,
                &encoder,
                &gate_weight,
                &up_weight,
                &x,
                &fused_target,
                N_IN,
                N_OUT,
                CLAMP,
            ),
            GgmlType::Q6_K => crate::metal::encode_ds4_shared_swiglu_q6_k_f32(
                &ctx,
                &encoder,
                &gate_weight,
                &up_weight,
                &x,
                &fused_target,
                N_IN,
                N_OUT,
                CLAMP,
            ),
            _ => unreachable!(),
        }
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{dtype:?} command failed");

        let gate = read_f32(&gate);
        let up = read_f32(&up);
        assert!(gate.iter().any(|&value| value > CLAMP));
        assert!(up.iter().any(|&value| value > CLAMP));
        assert!(up.iter().any(|&value| value < -CLAMP));
        let composed = read_f32(&composed);
        let fused_all = read_f32(&fused_target);
        if dtype == GgmlType::Q6_K {
            if let Some((index, (left, right))) = gate
                .iter()
                .zip(&fused_all[N_OUT..N_OUT * 2])
                .enumerate()
                .find(|(_, (left, right))| left.to_bits() != right.to_bits())
            {
                panic!(
                    "Q6 gate projection mismatch at {index}: composed={left:?} ({:08x}) fused={right:?} ({:08x})",
                    left.to_bits(),
                    right.to_bits()
                );
            }
            if let Some((index, (left, right))) = up
                .iter()
                .zip(&fused_all[N_OUT * 2..N_OUT * 3])
                .enumerate()
                .find(|(_, (left, right))| left.to_bits() != right.to_bits())
            {
                panic!(
                    "Q6 up projection mismatch at {index}: composed={left:?} ({:08x}) fused={right:?} ({:08x})",
                    left.to_bits(),
                    right.to_bits()
                );
            }
        }
        let fused = fused_all[..N_OUT].to_vec();
        assert!(
            composed
                .iter()
                .zip(&fused)
                .all(|(&left, &right)| left.to_bits() == right.to_bits()),
            "{dtype:?} fused shared expert must match composed gate/up/clamped-SwiGLU bit-for-bit"
        );
    }
}

#[test]
fn decode_compressor_q8_pair_matches_composed_frontier_write_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }

    for (n_in, n_out) in [
        (512usize, 67usize),
        (4_096, 256),
        (4_096, 512),
        (4_096, 1_024),
    ] {
        let make_weight = |salt: usize| {
            let mut bytes = Vec::with_capacity(n_out * (n_in / 32) * 34);
            for row in 0..n_out {
                for block in 0..n_in / 32 {
                    let scale = half::f16::from_f32(
                        0.0015 + ((row * 13 + block * 7 + salt) % 29) as f32 * 0.00017,
                    );
                    bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                    for index in 0..32 {
                        bytes.push(((row * 19 + block * 31 + index * 23 + salt * 11) % 255) as u8);
                    }
                }
            }
            MetalTensor::from_bytes(
                &ctx,
                &bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q8_0,
            )
            .expect("compressor Q8 weight")
        };
        let kv_weight = make_weight(3);
        let score_weight = make_weight(17);
        let x_values = (0..n_in)
            .map(|index| ((index * 43 + index / 7 + 11) % 503) as f32 * 0.0091 - 2.27)
            .collect::<Vec<_>>();
        let ape_values = (0..n_out)
            .map(|index| {
                let base = ((index * 37 + 5) % 211) as f32 * 0.00031 - 0.032;
                match index % 8 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(1),
                    3 => -f32::from_bits(1),
                    _ => base,
                }
            })
            .collect::<Vec<_>>();
        let x = offset_f32(&ctx, &x_values, vec![n_in as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![n_out as u64]);
        let composed_kv = offset_f32(&ctx, &vec![0.0; n_out], vec![n_out as u64]);
        let composed_score = offset_f32(&ctx, &vec![0.0; n_out], vec![n_out as u64]);
        let composed_kv_state = offset_f32(&ctx, &vec![f32::NAN; n_out], vec![n_out as u64]);
        let composed_score_state =
            offset_f32(&ctx, &vec![f32::NEG_INFINITY; n_out], vec![n_out as u64]);
        let fused_score = offset_f32(&ctx, &vec![0.0; n_out], vec![n_out as u64]);
        let fused_kv_state = offset_f32(&ctx, &vec![f32::NAN; n_out], vec![n_out as u64]);
        let fused_score_state =
            offset_f32(&ctx, &vec![f32::NEG_INFINITY; n_out], vec![n_out as u64]);

        let command = ctx.queue.commandBuffer().expect("compressor pair command");
        let encoder = KernelEncoder::begin(&command);
        encode_projection(
            &ctx,
            &encoder,
            &kv_weight,
            &x,
            &composed_kv,
            n_in,
            n_out,
            "compressor KV differential",
        )
        .unwrap();
        encode_projection(
            &ctx,
            &encoder,
            &score_weight,
            &x,
            &composed_score,
            n_in,
            n_out,
            "compressor score differential",
        )
        .unwrap();
        encode_compressor_frontier_write(
            &ctx,
            &encoder,
            &composed_kv,
            &composed_score,
            &ape,
            &composed_kv_state,
            &composed_score_state,
            n_out,
            0,
        )
        .unwrap();
        crate::metal::encode_ds4_compressor_pair_q8_0_f32(
            &ctx,
            &encoder,
            &kv_weight,
            &score_weight,
            &x,
            &fused_score,
            &ape,
            &fused_kv_state,
            &fused_score_state,
            n_in,
            n_out,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "compressor pair command failed");

        let assert_bits = |label: &str, composed: &MetalTensor, fused: &MetalTensor| {
            let composed = read_f32(composed);
            let fused = read_f32(fused);
            if let Some((index, (&left, &right))) = composed
                .iter()
                .zip(&fused)
                .enumerate()
                .find(|(_, (left, right))| left.to_bits() != right.to_bits())
            {
                panic!(
                    "{label} mismatch at {index} for {n_in}x{n_out}: composed={left:?} ({:08x}) fused={right:?} ({:08x})",
                    left.to_bits(),
                    right.to_bits(),
                );
            }
        };
        assert_bits("score projection", &composed_score, &fused_score);
        assert_bits("KV state", &composed_kv_state, &fused_kv_state);
        assert_bits("score state", &composed_score_state, &fused_score_state);
        assert!(read_f32(&fused_kv_state).iter().any(|&value| value != 0.0));
    }

    let q8_bytes = vec![0u8; 2 * 34];
    let weight = MetalTensor::from_bytes(&ctx, &q8_bytes, vec![32, 2], GgmlType::Q8_0)
        .expect("valid Q8 rejection weight");
    let x = offset_f32(&ctx, &[0.0; 32], vec![32]);
    let projected_score = offset_f32(&ctx, &[0.0; 2], vec![2]);
    let ape = offset_f32(&ctx, &[0.0; 2], vec![2]);
    let state = offset_f32(&ctx, &[0.0; 4], vec![4]);
    let kv_state = state.view_subrange(0, vec![2]);
    let score_state = state.view_subrange(1, vec![2]);
    let command = ctx
        .queue
        .commandBuffer()
        .expect("compressor rejection command");
    let encoder = KernelEncoder::begin(&command);
    let error = crate::metal::encode_ds4_compressor_pair_q8_0_f32(
        &ctx,
        &encoder,
        &weight,
        &weight,
        &x,
        &projected_score,
        &ape,
        &kv_state,
        &score_state,
        32,
        2,
    )
    .expect_err("partially overlapping state rows must fail closed");
    assert!(format!("{error}").contains("ds4_compressor_pair_q8_0"));
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
}

#[test]
fn decode_compressor_q8_pair_replaces_three_dispatches_with_one() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }

    const N_IN: usize = 512;
    const N_OUT: usize = 67;
    let mut bytes = vec![0u8; N_OUT * (N_IN / 32) * 34];
    for block in bytes.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        block[2..].fill(1);
    }
    let weight = MetalTensor::from_bytes(
        &ctx,
        &bytes,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::Q8_0,
    )
    .unwrap();
    let x = offset_f32(&ctx, &[0.25; N_IN], vec![N_IN as u64]);
    let ape = offset_f32(&ctx, &[0.125; N_OUT], vec![N_OUT as u64]);
    let projected_kv = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);
    let projected_score = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);
    let kv_state = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);
    let score_state = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);

    let composed = {
        let _trace = crate::metal::kernel_trace_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_projection(
            &ctx,
            &encoder,
            &weight,
            &x,
            &projected_kv,
            N_IN,
            N_OUT,
            "dispatch-count KV",
        )
        .unwrap();
        encode_projection(
            &ctx,
            &encoder,
            &weight,
            &x,
            &projected_score,
            N_IN,
            N_OUT,
            "dispatch-count score",
        )
        .unwrap();
        encode_compressor_frontier_write(
            &ctx,
            &encoder,
            &projected_kv,
            &projected_score,
            &ape,
            &kv_state,
            &score_state,
            N_OUT,
            0,
        )
        .unwrap();
        let trace = crate::metal::kernel_trace_snapshot();
        encoder.end();
        trace
    };
    let fused = {
        let _trace = crate::metal::kernel_trace_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::encode_ds4_compressor_pair_q8_0_f32(
            &ctx,
            &encoder,
            &weight,
            &weight,
            &x,
            &projected_score,
            &ape,
            &kv_state,
            &score_state,
            N_IN,
            N_OUT,
        )
        .unwrap();
        let trace = crate::metal::kernel_trace_snapshot();
        encoder.end();
        trace
    };
    assert_eq!(composed.dispatches, 3);
    assert_eq!(fused.dispatches, 1);
}

#[test]
fn decode_compressor_q8_pair_matches_composed_frontier_steps_bitwise() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !crate::metal::mat_vec_q8_0_lcpp_enabled() {
        return;
    }

    const N_IN: usize = 512;
    let run = |ratio: usize, head_dim: usize, positions: &[u32]| {
        let width = if ratio == 4 { 2 * head_dim } else { head_dim };
        let make_weight = |salt: usize| {
            let mut bytes = Vec::with_capacity(width * (N_IN / 32) * 34);
            for row in 0..width {
                for block in 0..N_IN / 32 {
                    let scale = half::f16::from_f32(
                        0.0012 + ((row * 11 + block * 17 + salt) % 31) as f32 * 0.00013,
                    );
                    bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                    for index in 0..32 {
                        bytes.push(((row * 29 + block * 19 + index * 7 + salt * 13) % 255) as u8);
                    }
                }
            }
            MetalTensor::from_bytes(
                &ctx,
                &bytes,
                vec![N_IN as u64, width as u64],
                GgmlType::Q8_0,
            )
            .expect("frontier-step Q8 weight")
        };
        let kv_weight = make_weight(5);
        let score_weight = make_weight(23);
        let ape_values = (0..ratio * width)
            .map(|index| ((index * 31 + 9) % 257) as f32 * 0.00021 - 0.027)
            .collect::<Vec<_>>();
        let ape = offset_f32(&ctx, &ape_values, vec![width as u64, ratio as u64]);
        let norm = offset_f32(&ctx, &vec![1.0; head_dim], vec![head_dim as u64]);
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: head_dim.min(64),
            theta: 10_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let composed = DeepSeekV4CompressorFrontier::new(
            &ctx,
            ratio,
            head_dim,
            DeepSeekV4CompressorPublication::Attention,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        )
        .unwrap();
        let fused = DeepSeekV4CompressorFrontier::new(
            &ctx,
            ratio,
            head_dim,
            DeepSeekV4CompressorPublication::Attention,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        )
        .unwrap();

        let command = ctx.queue.commandBuffer().expect("frontier-step command");
        let encoder = KernelEncoder::begin(&command);
        for &position in positions {
            let x_values = (0..N_IN)
                .map(|index| {
                    ((index * 41 + position as usize * 37 + 3) % 401) as f32 * 0.0083 - 1.67
                })
                .collect::<Vec<_>>();
            let x = offset_f32(&ctx, &x_values, vec![N_IN as u64]);
            encode_projection(
                &ctx,
                &encoder,
                &kv_weight,
                &x,
                &composed.projected_kv,
                N_IN,
                width,
                "composed frontier-step KV",
            )
            .unwrap();
            encode_projection(
                &ctx,
                &encoder,
                &score_weight,
                &x,
                &composed.projected_score,
                N_IN,
                width,
                "composed frontier-step score",
            )
            .unwrap();
            composed
                .encode_projected(
                    &ctx,
                    &encoder,
                    &composed.projected_kv,
                    &composed.projected_score,
                    &ape,
                    &norm,
                    position,
                    rope,
                    1e-5,
                )
                .unwrap();

            fused
                .encode(
                    &ctx,
                    &encoder,
                    &x,
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position,
                    N_IN,
                    rope,
                    1e-5,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "frontier-step command failed");

        let assert_f32_bits = |label: &str, left: &MetalTensor, right: &MetalTensor| {
            let left = read_f32(left);
            let right = read_f32(right);
            assert!(
                left.iter()
                    .zip(&right)
                    .all(|(&left, &right)| left.to_bits() == right.to_bits()),
                "{label} differs for ratio {ratio} head_dim {head_dim}"
            );
        };
        assert_f32_bits("KV state", &composed.kv_state, &fused.kv_state);
        assert_f32_bits("score state", &composed.score_state, &fused.score_state);
        assert_f32_bits("pooled row", &composed.pooled, &fused.pooled);
        assert_f32_bits("normalized row", &composed.normalized, &fused.normalized);
        assert_eq!(
            read_f16_bits(&composed.published),
            read_f16_bits(&fused.published),
            "publication differs for ratio {ratio} head_dim {head_dim}"
        );
    };

    run(4, 128, &[2, 3, 4, 6, 7, 8]);
    run(128, 128, &[126, 127, 128]);
}

fn read_u8(tensor: &MetalTensor) -> Vec<u8> {
    assert_eq!(tensor.dtype, GgmlType::I8);
    unsafe {
        let pointer = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
    }
}

fn read_f16_bits(tensor: &MetalTensor) -> Vec<u16> {
    assert_eq!(tensor.dtype, GgmlType::F16);
    unsafe {
        let pointer = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
    }
}

fn i8_prefix(tensor: &MetalTensor, shape: Vec<u64>) -> MetalTensor {
    assert_eq!(tensor.dtype, GgmlType::I8);
    let elements = crate::tensor::checked_shape_elements(&shape).expect("I8 prefix shape");
    assert!(elements <= tensor.n_elements());
    MetalTensor {
        buffer: tensor.buffer.clone(),
        offset: tensor.offset,
        shape,
        dtype: GgmlType::I8,
        provenance: tensor.provenance,
    }
}

fn zero_tensor_bytes(tensor: &MetalTensor) {
    let bytes = usize::try_from(tensor.n_bytes()).expect("test tensor byte count fits usize");
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::ptr::write_bytes(destination, 0, bytes);
    }
}

fn fill_tensor_bytes(tensor: &MetalTensor, byte: u8) {
    let bytes = usize::try_from(tensor.n_bytes()).expect("test tensor byte count fits usize");
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::ptr::write_bytes(destination, byte, bytes);
    }
}

fn initialize_zero_synthetic_causal_state(session: &DeepSeekV4Session) {
    fn zero_frontier(frontier: &DeepSeekV4CompressorFrontier) {
        zero_tensor_bytes(&frontier.kv_state);
        zero_tensor_bytes(&frontier.score_state);
        zero_tensor_bytes(&frontier.published);
    }

    // Scratch allocations are uninitialized; a direct phase jump must
    // define every causal byte and every phase-visible compressor score.
    zero_tensor_bytes(&session.raw_cache);
    for layer in &session.compressor_frontiers.layers {
        match layer {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                zero_frontier(attention);
                zero_frontier(indexer);
            }
            DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => {
                zero_frontier(attention);
            }
        }
    }
}

fn read_f16(tensor: &MetalTensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, GgmlType::F16);
    unsafe {
        let pointer = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::slice::from_raw_parts(pointer, tensor.n_elements() as usize)
            .iter()
            .map(|bits| half::f16::from_bits(*bits).to_f32())
            .collect()
    }
}

fn offset_i32(ctx: &MetalContext, values: &[i32], shape: Vec<u64>) -> MetalTensor {
    let prefix = 20usize;
    let mut bytes = vec![0xA5u8; prefix];
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    bytes.extend_from_slice(&[0x5Au8; 20]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("offset I32 buffer"),
        offset: prefix as u64,
        shape,
        dtype: GgmlType::I32,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    host_read_i32(tensor, "test I32 tensor").expect("read I32")
}

fn deployed_selector_order_key(value: f32) -> u32 {
    assert!(value.is_finite());
    let mut bits = value.to_bits();
    if (bits & 0x7f80_0000) == 0 {
        bits = 0;
    }
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn deployed_selector_top_k_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        deployed_selector_order_key(scores[right])
            .cmp(&deployed_selector_order_key(scores[left]))
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k.min(indices.len()));
    indices
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MultigroupSelectorThresholdOracle {
    status: u32,
    selected_count: u32,
    threshold_key: u32,
    threshold_take: u32,
    partition_plan: Vec<u32>,
}

fn multigroup_selector_threshold_oracle(
    scores: &[f32],
    row_capacity: usize,
    visible_count: i32,
    top_k: usize,
    group_count: usize,
) -> MultigroupSelectorThresholdOracle {
    let geometry_valid = visible_count > 0
        && visible_count as usize <= row_capacity
        && top_k > 0
        && top_k <= row_capacity
        && scores.len() == row_capacity;
    let visible = if geometry_valid {
        visible_count as usize
    } else {
        0
    };
    let selected_count = visible.min(top_k);
    let mut result = MultigroupSelectorThresholdOracle {
        status: u32::from(!geometry_valid),
        selected_count: selected_count as u32,
        threshold_key: 0,
        threshold_take: 0,
        partition_plan: vec![0; group_count * DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS],
    };
    if !geometry_valid {
        return result;
    }
    if scores[..visible].iter().any(|score| !score.is_finite()) {
        result.status = 2;
        return result;
    }
    let mut keys = scores[..visible]
        .iter()
        .map(|&score| deployed_selector_order_key(score))
        .collect::<Vec<_>>();
    keys.sort_unstable_by(|left, right| right.cmp(left));
    result.threshold_key = keys[selected_count - 1];
    let greater_total = keys
        .iter()
        .take_while(|&&key| key > result.threshold_key)
        .count();
    result.threshold_take = (selected_count - greater_total) as u32;
    let chunk = row_capacity.div_ceil(group_count);
    for group in 0..group_count {
        let plan_base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS;
        let start = (group * chunk).min(row_capacity).min(visible);
        let end = (start + chunk).min(row_capacity).min(visible);
        for &score in &scores[start..end] {
            let key = deployed_selector_order_key(score);
            if key > result.threshold_key {
                result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_GREATER] +=
                    1;
            } else if key == result.threshold_key {
                result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_EQUAL] += 1;
            }
        }
    }
    let mut remaining_ties = result.threshold_take;
    let mut output_offset = 0u32;
    for group in 0..group_count {
        let plan_base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS;
        let greater =
            result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_GREATER];
        let equal = result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_EQUAL];
        let quota = equal.min(remaining_ties);
        let selected = greater + quota;
        result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_TIE_QUOTA] = quota;
        result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_SELECTED] = selected;
        result.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_ID_OFFSET] =
            output_offset;
        remaining_ties -= quota;
        output_offset += selected;
    }
    assert_eq!(remaining_ties, 0);
    assert_eq!(output_offset, result.selected_count);
    result
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let allowed = tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "{label}[{index}] = {actual}, expected {expected}, tolerance {allowed}"
        );
    }
}

fn f32_desc(name: &str, offset: u64) -> TensorDesc {
    TensorDesc {
        name: name.into(),
        shape: vec![8],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: offset,
        n_bytes: 32,
    }
}

#[test]
fn session_position_guard_separates_evidence_from_physical_capacity() {
    let config = crate::deepseek_v4::flash_0731_config_fixture();
    assert_eq!(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 1_048_576);
    assert_eq!(config.context_length, 1_048_576);
    let capacity = DeepSeekV4SessionCapacity::for_forward_limit(
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
        config.context_length,
    )
    .unwrap();
    assert_eq!(capacity.csa_physical_rows(), 262_144);
    assert_eq!(capacity.hca_physical_rows(), 8_192);
    capacity.validate_position(1_048_575).unwrap();
    assert!(
        capacity
            .validate_position(1_048_576)
            .unwrap_err()
            .to_string()
            .contains("next position is 1048576")
    );

    let decimal_million =
        DeepSeekV4SessionCapacity::for_forward_limit(1_000_000, config.context_length).unwrap();
    assert_eq!(decimal_million.csa_physical_rows(), 250_112);
    assert_eq!(decimal_million.hca_physical_rows(), 7_936);

    let before_fourth =
        DeepSeekV4SessionCapacity::for_forward_limit(3_075, config.context_length).unwrap();
    let fourth =
        DeepSeekV4SessionCapacity::for_forward_limit(3_076, config.context_length).unwrap();
    assert_eq!(before_fourth.csa_physical_rows(), 768);
    assert_eq!(fourth.csa_physical_rows(), 1_024);
    for forward_limit in 1..=DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY {
        let derived =
            DeepSeekV4SessionCapacity::for_forward_limit(forward_limit, config.context_length)
                .unwrap();
        let required_csa = forward_limit / 4;
        let required_hca = forward_limit / 128;
        assert!(derived.csa_physical_rows() >= required_csa.max(768));
        assert!(derived.hca_physical_rows() >= required_hca.max(512));
        assert!(
            derived
                .csa_physical_rows()
                .is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        );
        assert!(
            derived
                .hca_physical_rows()
                .is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        );
        if required_csa > 768 {
            assert!(
                derived.csa_physical_rows() - required_csa
                    < DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS
            );
        }
        if required_hca > 512 {
            assert!(
                derived.hca_physical_rows() - required_hca
                    < DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS
            );
        }
    }
    assert!(
        DeepSeekV4SessionCapacity::for_forward_limit(1_048_577, config.context_length).is_err()
    );
    assert!(DeepSeekV4SessionCapacity::for_forward_limit(1_048_576, 1_048_575).is_err());
}

#[test]
fn multigroup_selector_crossover_predicate_is_frozen() {
    for (capacity, visible, expected) in [
        (196_352, 196_352, false),
        (196_608, 196_607, false),
        (196_608, 196_608, true),
        (250_112, 196_607, false),
        (250_112, 196_608, true),
        (250_112, 250_112, true),
        (262_144, 196_607, false),
        (262_144, 196_608, true),
        (262_144, 262_144, true),
        (262_145, 262_145, false),
        (262_144, 262_145, false),
    ] {
        assert_eq!(
            deepseek_v4_multigroup_selector_eligible(capacity, visible),
            expected,
            "capacity={capacity} visible={visible}"
        );
    }
}

#[test]
fn multigroup_selector_product_geometry_requires_reachable_visibility() {
    let rounded_but_unreachable =
        DeepSeekV4SessionCapacity::for_forward_limit(786_431, 1_048_576).unwrap();
    assert_eq!(rounded_but_unreachable.csa_physical_rows(), 196_608);
    assert_eq!(rounded_but_unreachable.forward_limit() / 4, 196_607);
    assert!(
        rounded_but_unreachable
            .qualify_multigroup_selector_experiment()
            .unwrap_err()
            .to_string()
            .contains("max_visible_rows=196607")
    );

    for (forwards, capacity, max_visible) in [
        (786_432, 196_608, 196_608),
        (1_000_000, 250_112, 250_000),
        (1_048_576, 262_144, 262_144),
    ] {
        let session = DeepSeekV4SessionCapacity::for_forward_limit(forwards, 1_048_576).unwrap();
        let geometry = session.qualify_multigroup_selector_experiment().unwrap();
        assert_eq!(geometry.forward_limit(), forwards);
        assert_eq!(geometry.physical_capacity_rows(), capacity);
        assert_eq!(geometry.max_visible_rows(), max_visible);
    }
}

#[test]
fn multigroup_selector_invocation_telemetry_distinguishes_fallback() {
    let counters = DeepSeekV4MultigroupSelectorInvocationCounters::default();
    assert_eq!(
        counters.telemetry(false),
        DeepSeekV4MultigroupSelectorTelemetry {
            sealed: false,
            multigroup_invocations: 0,
            ineligible_radix4_invocations: 0,
        }
    );

    let next = counters.next_ineligible_radix4().unwrap();
    counters.commit_ineligible_radix4(next);
    let next = counters.next_multigroup().unwrap();
    counters.commit_multigroup(next);
    let next = counters.next_multigroup().unwrap();
    counters.commit_multigroup(next);

    assert_eq!(
        counters.telemetry(true),
        DeepSeekV4MultigroupSelectorTelemetry {
            sealed: true,
            multigroup_invocations: 2,
            ineligible_radix4_invocations: 1,
        }
    );
}

#[test]
fn multigroup_selector_generation_fails_before_reuse() {
    let generation =
        DeepSeekV4MultigroupSelectorGeneration::from_next(NonZeroU32::new(u32::MAX - 1).unwrap());
    assert_eq!(generation.take().unwrap(), u32::MAX - 1);
    assert_eq!(generation.take().unwrap(), u32::MAX);
    assert!(
        generation
            .take()
            .unwrap_err()
            .to_string()
            .contains("exhausted")
    );
}

#[test]
fn sparse_csa_multigroup_scratch_is_explicit_and_capacity_bounded() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut shallow = DeepSeekV4SparseCsaScratch::new(&ctx, 768).unwrap();
    assert_eq!(shallow.selector_mode, DeepSeekV4SparseSelectorMode::Radix4);
    assert!(shallow.multigroup.is_none());
    assert!(!shallow.multigroup_selector_telemetry().sealed());
    assert!(shallow.enable_multigroup_selector_experiment().is_err());

    let mut terminal = DeepSeekV4SparseCsaScratch::new(&ctx, 262_144).unwrap();
    assert_eq!(terminal.selector_mode, DeepSeekV4SparseSelectorMode::Radix4);
    assert!(terminal.multigroup.is_some());
    assert!(!terminal.multigroup_selector_telemetry().sealed());
    terminal.enable_multigroup_selector_experiment().unwrap();
    assert_eq!(
        terminal.selector_mode,
        DeepSeekV4SparseSelectorMode::MultigroupExperimental
    );
    assert!(terminal.multigroup_selector_telemetry().sealed());
    assert!(terminal.enable_multigroup_selector_experiment().is_err());
}

#[test]
fn session_phase_encodes_observation_and_poison_transitions() {
    let mut phase = DeepSeekV4SessionPhase::fresh();
    assert_eq!(phase.next_position(), 0);
    assert_eq!(phase.ready_position().unwrap(), 0);
    assert!(!phase.observation_valid());

    assert_eq!(phase.begin_mutation().unwrap(), 0);
    assert_eq!(phase, DeepSeekV4SessionPhase::Poisoned { next_position: 0 });
    assert!(!phase.observation_valid());
    assert!(
        phase
            .ready_position()
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );

    phase.complete_mutation(0, 1, true).unwrap();
    assert_eq!(phase.ready_position().unwrap(), 1);
    assert!(phase.observation_valid());
    assert_eq!(phase.begin_mutation().unwrap(), 1);
    assert!(!phase.observation_valid());
    phase.complete_mutation(1, 2, false).unwrap();
    assert_eq!(phase.ready_position().unwrap(), 2);
    assert!(!phase.observation_valid());

    let invalid_phase = phase.complete_mutation(2, 3, true).unwrap_err();
    assert!(invalid_phase.to_string().contains("invalid session phase"));

    assert_eq!(phase.begin_restore().unwrap(), 2);
    phase.complete_restore(2, 1).unwrap();
    assert_eq!(phase.ready_position().unwrap(), 1);
    assert!(!phase.observation_valid());
    assert_eq!(phase.begin_restore().unwrap(), 1);
    phase.complete_restore(1, 1).unwrap();
    assert_eq!(phase.ready_position().unwrap(), 1);

    assert_eq!(phase.begin_mutation().unwrap(), 1);
    let nonadvancing = phase.complete_mutation(1, 1, true).unwrap_err();
    assert!(nonadvancing.to_string().contains("did not advance"));
    assert_eq!(phase.next_position(), 1);
    assert!(!phase.observation_valid());
    assert!(
        phase
            .begin_mutation()
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset"]
fn current_asset_packed_gpu_route_kill_packet() {
    const CHUNK_TOKENS: usize = 128;
    const PREFIX_TOKENS: usize = 140;
    const CONTINUATION_TOKEN: u32 = 35;
    const FORWARD_LIMIT: usize = PREFIX_TOKENS + 1;

    struct Evidence {
        packed_logits: Vec<f32>,
        packed_hidden: Vec<f32>,
        packed_causal_digest: [u8; 32],
        packed_prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        continuation_logits: Vec<f32>,
        continuation_causal_digest: [u8; 32],
        committed_tokens: Vec<u32>,
        route_generations: u32,
        packed_wall_ms: f64,
    }

    fn metrics(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        assert_eq!(actual.len(), reference.len());
        let mut dot = 0.0_f64;
        let mut actual_norm = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        let mut squared_error = 0.0_f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            let actual = actual as f64;
            let reference = reference as f64;
            dot += actual * reference;
            actual_norm += actual * actual;
            reference_norm += reference * reference;
            let error = actual - reference;
            squared_error += error * error;
        }
        (
            dot / (actual_norm * reference_norm).sqrt(),
            (squared_error / reference_norm).sqrt(),
        )
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn digest_f32(values: &[f32]) -> [u8; 32] {
        Sha256::digest(bytemuck::cast_slice(values)).into()
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        gpu_route: bool,
        preserve_cpu_weights: bool,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct packed GPU route session");
        let generation_before = session.packed_route_generation_for_test();
        let started = std::time::Instant::now();
        session
            .execute_packed_tokens_with_route_policy_for_test(
                ctx,
                &prefix[..CHUNK_TOKENS],
                false,
                gpu_route,
                preserve_cpu_weights,
            )
            .expect("execute 128-token packed route chunk");
        session
            .execute_packed_tokens_with_route_policy_for_test(
                ctx,
                &prefix[CHUNK_TOKENS..],
                true,
                gpu_route,
                preserve_cpu_weights,
            )
            .expect("execute 12-token packed route tail");
        let packed_wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let route_generations = session
            .packed_route_generation_for_test()
            .checked_sub(generation_before)
            .expect("packed route generation is monotonic");
        let packed_logits = session.copy_logits_f32().expect("copy packed route logits");
        let packed_hidden = host_read_f32(
            session
                .final_normalized_hidden()
                .expect("packed final hidden is visible"),
            "packed GPU route final hidden",
        )
        .expect("copy packed route hidden");
        let packed = session
            .capture_causal_snapshot()
            .expect("capture packed GPU route state");
        let packed_causal_digest = *packed.causal_digest();
        let packed_prefix_digest = *packed.prefix_digest();
        let compatibility_digest = *packed.compatibility_digest().as_bytes();
        session
            .restore_causal_snapshot(&packed)
            .expect("restore packed GPU route state");
        session
            .forward_token(ctx, CONTINUATION_TOKEN)
            .expect("consume packed GPU route state");
        let continuation_logits = session
            .copy_logits_f32()
            .expect("copy packed route continuation logits");
        let continuation = session
            .capture_causal_snapshot()
            .expect("capture packed route continuation state");
        let evidence = Evidence {
            packed_logits,
            packed_hidden,
            packed_causal_digest,
            packed_prefix_digest,
            compatibility_digest,
            continuation_logits,
            continuation_causal_digest: *continuation.causal_digest(),
            committed_tokens: session.committed_tokens().to_vec(),
            route_generations,
            packed_wall_ms,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
        .expect("plan packed GPU route session");
    assert_eq!(plan.session_capacity().csa_physical_rows(), 768);
    assert_eq!(plan.session_capacity().hca_physical_rows(), 512);
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit packed GPU route session");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let residency = realized.into_residency();
    let prefix = (0..PREFIX_TOKENS)
        .map(|index| [35, 201, 200, 34][index % 4])
        .collect::<Vec<_>>();

    let (residency, current_before) =
        execute(&ctx, residency, model_content_id, &prefix, false, false);
    let (residency, candidate) = execute(&ctx, residency, model_content_id, &prefix, true, false);
    let (residency, cpu_weight_hybrid) =
        execute(&ctx, residency, model_content_id, &prefix, true, true);
    let (_residency, current_after) =
        execute(&ctx, residency, model_content_id, &prefix, false, false);

    assert_eq!(
        bits(&current_before.packed_logits),
        bits(&current_after.packed_logits)
    );
    assert_eq!(
        bits(&current_before.packed_hidden),
        bits(&current_after.packed_hidden)
    );
    assert_eq!(
        bits(&current_before.continuation_logits),
        bits(&current_after.continuation_logits)
    );
    assert_eq!(
        current_before.packed_causal_digest,
        current_after.packed_causal_digest
    );
    assert_eq!(
        current_before.continuation_causal_digest,
        current_after.continuation_causal_digest
    );
    assert_eq!(
        current_before.committed_tokens,
        current_after.committed_tokens
    );
    assert_eq!(candidate.committed_tokens, current_before.committed_tokens);
    assert_eq!(
        cpu_weight_hybrid.committed_tokens,
        current_before.committed_tokens
    );
    assert_eq!(
        candidate.packed_prefix_digest,
        current_before.packed_prefix_digest
    );
    assert_eq!(
        candidate.compatibility_digest,
        current_before.compatibility_digest
    );
    assert_eq!(current_before.route_generations, 0);
    assert_eq!(current_after.route_generations, 0);
    assert_eq!(
        candidate.route_generations,
        2 * DEEPSEEK_V4_LAYER_COUNT as u32
    );
    assert_eq!(
        cpu_weight_hybrid.route_generations,
        2 * DEEPSEEK_V4_LAYER_COUNT as u32
    );

    let packed_argmax = argmax(&current_before.packed_logits);
    let candidate_packed_argmax = argmax(&candidate.packed_logits);
    let continuation_argmax = argmax(&current_before.continuation_logits);
    let candidate_continuation_argmax = argmax(&candidate.continuation_logits);
    let (packed_cosine, packed_relative_rms) =
        metrics(&candidate.packed_logits, &current_before.packed_logits);
    let (hidden_cosine, hidden_relative_rms) =
        metrics(&candidate.packed_hidden, &current_before.packed_hidden);
    let (continuation_cosine, continuation_relative_rms) = metrics(
        &candidate.continuation_logits,
        &current_before.continuation_logits,
    );
    assert_eq!(candidate_packed_argmax, packed_argmax);
    assert_eq!(candidate_continuation_argmax, continuation_argmax);
    assert_ne!(
        bits(&candidate.packed_logits),
        bits(&current_before.packed_logits)
    );
    assert_ne!(
        bits(&candidate.packed_hidden),
        bits(&current_before.packed_hidden)
    );
    assert_ne!(
        bits(&candidate.continuation_logits),
        bits(&current_before.continuation_logits)
    );
    assert_ne!(
        candidate.packed_causal_digest,
        current_before.packed_causal_digest
    );
    assert_ne!(
        candidate.continuation_causal_digest,
        current_before.continuation_causal_digest
    );
    assert_eq!(
        digest_hex(&digest_f32(&current_before.packed_logits)),
        "688aecb312c9bff3469b4947e37086d9230efaa835c8afef529e7e7618ff886e"
    );
    assert_eq!(
        digest_hex(&digest_f32(&candidate.packed_logits)),
        "81cb6874e6ada8d905c9744105ee01622e01f4df8ddca575f5b4281fa2bfd94f"
    );
    assert_eq!(
        digest_hex(&current_before.packed_causal_digest),
        "0bc001b6045aa52f636db496628cad69b9bab8d7f1727e9e4f41b21075cf41bb"
    );
    assert_eq!(
        digest_hex(&candidate.packed_causal_digest),
        "067c8e5c191cbef2e599b7637e55cde545b32236a771499448c16fe102ffdb47"
    );
    assert_eq!(
        digest_hex(&candidate.continuation_causal_digest),
        "a7526bfb5c733cd311c30536fe10e7ef4b62e6a2866c22d661d42b1e0c0891ae"
    );

    assert_eq!(
        bits(&cpu_weight_hybrid.packed_logits),
        bits(&current_before.packed_logits)
    );
    assert_eq!(
        bits(&cpu_weight_hybrid.packed_hidden),
        bits(&current_before.packed_hidden)
    );
    assert_eq!(
        bits(&cpu_weight_hybrid.continuation_logits),
        bits(&current_before.continuation_logits)
    );
    assert_eq!(
        cpu_weight_hybrid.packed_causal_digest,
        current_before.packed_causal_digest
    );
    assert_eq!(
        cpu_weight_hybrid.packed_prefix_digest,
        current_before.packed_prefix_digest
    );
    assert_eq!(
        cpu_weight_hybrid.compatibility_digest,
        current_before.compatibility_digest
    );
    assert_eq!(
        cpu_weight_hybrid.continuation_causal_digest,
        current_before.continuation_causal_digest
    );

    eprintln!(
        "deepseek_v4 packed_gpu_route_kill prompt_tokens={PREFIX_TOKENS} chunks=128+12 packed_argmax={packed_argmax} continuation_argmax={continuation_argmax} packed_cosine={packed_cosine:.9} packed_rel_rms={packed_relative_rms:.9} hidden_cosine={hidden_cosine:.9} hidden_rel_rms={hidden_relative_rms:.9} continuation_cosine={continuation_cosine:.9} continuation_rel_rms={continuation_relative_rms:.9} current_before_wall_ms={:.3} candidate_wall_ms={:.3} cpu_weight_hybrid_wall_ms={:.3} current_after_wall_ms={:.3} candidate_generations={} cpu_weight_hybrid_generations={} current_before_logits_sha256={} candidate_logits_sha256={} cpu_weight_hybrid_logits_sha256={} current_after_logits_sha256={} current_before_causal={} candidate_causal={} cpu_weight_hybrid_causal={} current_after_causal={} candidate_continuation_causal={} cpu_weight_hybrid_continuation_causal={}",
        current_before.packed_wall_ms,
        candidate.packed_wall_ms,
        cpu_weight_hybrid.packed_wall_ms,
        current_after.packed_wall_ms,
        candidate.route_generations,
        cpu_weight_hybrid.route_generations,
        digest_hex(&digest_f32(&current_before.packed_logits)),
        digest_hex(&digest_f32(&candidate.packed_logits)),
        digest_hex(&digest_f32(&cpu_weight_hybrid.packed_logits)),
        digest_hex(&digest_f32(&current_after.packed_logits)),
        digest_hex(&current_before.packed_causal_digest),
        digest_hex(&candidate.packed_causal_digest),
        digest_hex(&cpu_weight_hybrid.packed_causal_digest),
        digest_hex(&current_after.packed_causal_digest),
        digest_hex(&candidate.continuation_causal_digest),
        digest_hex(&cpu_weight_hybrid.continuation_causal_digest),
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset"]
fn current_asset_packed_grouped_expert_integration_packet() {
    const CONTINUATION_TOKEN: u32 = 35;
    const IQ2_ELIGIBLE_LAYERS: u32 = 25;
    const IQ3_ELIGIBLE_LAYERS: u32 = 16;

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum ExpertArm {
        Current,
        Ordinary,
        Iq2Baseline,
        Iq2AndIq3Candidate,
    }

    struct Evidence {
        logits: Vec<f32>,
        hidden: Vec<f32>,
        causal_digest: [u8; 32],
        prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        continuation_logits: Vec<f32>,
        continuation_causal_digest: [u8; 32],
        committed_tokens: Vec<u32>,
        grouped_iq2_invocations: u32,
        grouped_iq3_invocations: u32,
        wall_ms: f64,
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn digest_hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        arm: ExpertArm,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct packed grouped expert session");
        let started = std::time::Instant::now();
        match arm {
            ExpertArm::Ordinary => {
                session
                    .prefill_tokens(ctx, prefix)
                    .expect("execute ordinary packed grouped expert prefix");
            }
            ExpertArm::Current | ExpertArm::Iq2Baseline | ExpertArm::Iq2AndIq3Candidate => {
                let (grouped_iq2, grouped_iq3) = match arm {
                    ExpertArm::Current => (false, false),
                    ExpertArm::Iq2Baseline => (true, false),
                    ExpertArm::Iq2AndIq3Candidate => (true, true),
                    ExpertArm::Ordinary => unreachable!(),
                };
                session
                    .execute_packed_tokens_with_expert_policy_for_test(
                        ctx,
                        prefix,
                        true,
                        grouped_iq2,
                        grouped_iq3,
                    )
                    .expect("execute forced packed expert policy");
            }
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let logits = session
            .copy_logits_f32()
            .expect("copy packed grouped logits");
        let hidden = host_read_f32(
            session
                .final_normalized_hidden()
                .expect("packed grouped final hidden is visible"),
            "packed grouped final hidden",
        )
        .expect("copy packed grouped hidden");
        let snapshot = session
            .capture_causal_snapshot()
            .expect("capture packed grouped state");
        let causal_digest = *snapshot.causal_digest();
        let prefix_digest = *snapshot.prefix_digest();
        let compatibility_digest = *snapshot.compatibility_digest().as_bytes();
        session
            .restore_causal_snapshot(&snapshot)
            .expect("restore packed grouped state");
        session
            .forward_token(ctx, CONTINUATION_TOKEN)
            .expect("continue packed grouped state");
        let continuation_logits = session
            .copy_logits_f32()
            .expect("copy packed grouped continuation logits");
        let continuation = session
            .capture_causal_snapshot()
            .expect("capture packed grouped continuation state");
        let evidence = Evidence {
            logits,
            hidden,
            causal_digest,
            prefix_digest,
            compatibility_digest,
            continuation_logits,
            continuation_causal_digest: *continuation.causal_digest(),
            committed_tokens: session.committed_tokens().to_vec(),
            grouped_iq2_invocations: session.packed_grouped_iq2_invocations_for_test(),
            grouped_iq3_invocations: session.packed_grouped_iq3_invocations_for_test(),
            wall_ms,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    fn assert_exact(label: &str, actual: &Evidence, expected: &Evidence) {
        assert_eq!(
            bits(&actual.logits),
            bits(&expected.logits),
            "{label} logits"
        );
        assert_eq!(
            bits(&actual.hidden),
            bits(&expected.hidden),
            "{label} hidden"
        );
        assert_eq!(
            actual.causal_digest, expected.causal_digest,
            "{label} causal"
        );
        assert_eq!(
            actual.prefix_digest, expected.prefix_digest,
            "{label} prefix"
        );
        assert_eq!(
            actual.compatibility_digest, expected.compatibility_digest,
            "{label} compatibility"
        );
        assert_eq!(
            bits(&actual.continuation_logits),
            bits(&expected.continuation_logits),
            "{label} continuation logits"
        );
        assert_eq!(
            actual.continuation_causal_digest, expected.continuation_causal_digest,
            "{label} continuation causal"
        );
        assert_eq!(
            actual.committed_tokens, expected.committed_tokens,
            "{label} committed tokens"
        );
    }

    fn execute_samples(
        ctx: &MetalContext,
        mut residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        arm: ExpertArm,
        samples: usize,
    ) -> (DeepSeekV4MetalResidency, Vec<Evidence>) {
        let mut evidence = Vec::with_capacity(samples);
        for _ in 0..samples {
            let (next, sample) = execute(ctx, residency, model_content_id, prefix, arm);
            residency = next;
            evidence.push(sample);
        }
        (residency, evidence)
    }

    fn median_wall_ms(samples: &[Evidence]) -> f64 {
        let mut values = samples
            .iter()
            .map(|sample| sample.wall_ms)
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
        } else {
            values[values.len() / 2]
        }
    }

    fn p95_wall_ms(samples: &[Evidence]) -> f64 {
        let mut values = samples
            .iter()
            .map(|sample| sample.wall_ms)
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values[(values.len() * 95).div_ceil(100) - 1]
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let grouped_enabled = prefill::packed_grouped_expert_enabled_for_test(&ctx);
    let iq3_experiment = std::env::var_os("QWEN_DSV4_PACKED_GROUPED_IQ3_EXPERIMENT").is_some();
    let (control_arm, candidate_arm) = if iq3_experiment {
        (ExpertArm::Iq2Baseline, ExpertArm::Iq2AndIq3Candidate)
    } else {
        (ExpertArm::Current, ExpertArm::Ordinary)
    };
    let expected_control_iq2 = if iq3_experiment {
        IQ2_ELIGIBLE_LAYERS
    } else {
        0
    };
    let expected_candidate_iq2 = if iq3_experiment || grouped_enabled {
        IQ2_ELIGIBLE_LAYERS
    } else {
        0
    };
    let expected_candidate_iq3 = if iq3_experiment {
        IQ3_ELIGIBLE_LAYERS
    } else {
        0
    };
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, 129)
        .expect("plan packed grouped session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit packed grouped session");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let mut residency = realized.into_residency();

    let samples = std::env::var("QWEN_DSV4_PACKED_GROUPED_SAMPLES")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("QWEN_DSV4_PACKED_GROUPED_SAMPLES must be an integer")
        })
        .unwrap_or(1);
    assert!((1..=8).contains(&samples), "samples must be in 1..=8");
    if iq3_experiment && samples > 1 {
        assert_eq!(samples, 8, "grouped IQ3 timing requires balanced R8");
    }
    let only_n = std::env::var("QWEN_DSV4_PACKED_GROUPED_ONLY_N")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("QWEN_DSV4_PACKED_GROUPED_ONLY_N must be an integer")
        });
    if let Some(n_tokens) = only_n {
        assert!(
            [12, 32, 128].contains(&n_tokens),
            "QWEN_DSV4_PACKED_GROUPED_ONLY_N must be 12, 32, or 128"
        );
    }

    for n_tokens in [12usize, 32, 128]
        .into_iter()
        .filter(|n_tokens| only_n.is_none_or(|only_n| only_n == *n_tokens))
    {
        let prefix = (0..n_tokens)
            .map(|index| [35, 201, 200, 34][index % 4])
            .collect::<Vec<_>>();

        if iq3_experiment && samples > 1 {
            let mut next = residency;
            let mut warm_reference = None;
            for arm in [
                control_arm,
                candidate_arm,
                candidate_arm,
                control_arm,
                candidate_arm,
                control_arm,
                control_arm,
                candidate_arm,
            ] {
                let (residency, sample) = execute(&ctx, next, model_content_id, &prefix, arm);
                next = residency;
                if let Some(reference) = &warm_reference {
                    assert_exact("balanced warm-up", &sample, reference);
                } else {
                    warm_reference = Some(sample);
                }
            }
            residency = next;
        } else if samples > 1 {
            let (next, warm_control) =
                execute(&ctx, residency, model_content_id, &prefix, control_arm);
            let (next, warm_candidate) =
                execute(&ctx, next, model_content_id, &prefix, candidate_arm);
            residency = next;
            assert_exact("warm candidate", &warm_candidate, &warm_control);
        }

        let (next, current_before, candidate, current_after) = if iq3_experiment && samples > 1 {
            let mut next = residency;
            let mut controls = Vec::with_capacity(samples);
            let mut candidate = Vec::with_capacity(samples);
            for _ in 0..samples / 4 {
                for arm in [
                    control_arm,
                    candidate_arm,
                    candidate_arm,
                    control_arm,
                    candidate_arm,
                    control_arm,
                    control_arm,
                    candidate_arm,
                ] {
                    let (residency, sample) = execute(&ctx, next, model_content_id, &prefix, arm);
                    next = residency;
                    if arm == control_arm {
                        controls.push(sample);
                    } else {
                        candidate.push(sample);
                    }
                }
            }
            assert_eq!(controls.len(), samples);
            assert_eq!(candidate.len(), samples);
            let current_after = controls.split_off(samples / 2);
            let current_before = controls;
            (next, current_before, candidate, current_after)
        } else {
            let (next, current_before) = execute_samples(
                &ctx,
                residency,
                model_content_id,
                &prefix,
                control_arm,
                samples,
            );
            let (next, candidate) = execute_samples(
                &ctx,
                next,
                model_content_id,
                &prefix,
                candidate_arm,
                samples,
            );
            let (next, current_after) =
                execute_samples(&ctx, next, model_content_id, &prefix, control_arm, samples);
            (next, current_before, candidate, current_after)
        };
        residency = next;

        let reference = &current_before[0];
        for sample in &current_before {
            assert_exact("leading control", sample, reference);
            assert_eq!(sample.grouped_iq2_invocations, expected_control_iq2);
            assert_eq!(sample.grouped_iq3_invocations, 0);
        }
        for sample in &candidate {
            assert_exact("grouped candidate", sample, reference);
            assert_eq!(sample.grouped_iq2_invocations, expected_candidate_iq2);
            assert_eq!(sample.grouped_iq3_invocations, expected_candidate_iq3);
        }
        for sample in &current_after {
            assert_exact("trailing control", sample, reference);
            assert_eq!(sample.grouped_iq2_invocations, expected_control_iq2);
            assert_eq!(sample.grouped_iq3_invocations, 0);
        }

        let current_before_median = median_wall_ms(&current_before);
        let candidate_median = median_wall_ms(&candidate);
        let current_after_median = median_wall_ms(&current_after);
        let faster_control = current_before_median.min(current_after_median);
        let wall_saving = (faster_control - candidate_median) / faster_control;
        let control_drift = 2.0 * (current_before_median - current_after_median).abs()
            / (current_before_median + current_after_median);
        let current_before_p95 = p95_wall_ms(&current_before);
        let candidate_p95 = p95_wall_ms(&candidate);
        let current_after_p95 = p95_wall_ms(&current_after);
        let (candidate_first_median, candidate_second_median) = if iq3_experiment && samples > 1 {
            (
                median_wall_ms(&candidate[..samples / 2]),
                median_wall_ms(&candidate[samples / 2..]),
            )
        } else {
            (candidate_median, candidate_median)
        };
        let candidate_drift = 2.0 * (candidate_first_median - candidate_second_median).abs()
            / (candidate_first_median + candidate_second_median);
        let first_half_saving = 1.0 - candidate_first_median / current_before_median;
        let second_half_saving = 1.0 - candidate_second_median / current_after_median;
        let current_before_ms = current_before
            .iter()
            .map(|sample| sample.wall_ms)
            .collect::<Vec<_>>();
        let candidate_ms = candidate
            .iter()
            .map(|sample| sample.wall_ms)
            .collect::<Vec<_>>();
        let current_after_ms = current_after
            .iter()
            .map(|sample| sample.wall_ms)
            .collect::<Vec<_>>();
        let candidate_dispatches =
            candidate[0].grouped_iq2_invocations * 2 + candidate[0].grouped_iq3_invocations * 4;
        eprintln!(
            "deepseek_v4 packed_grouped_expert_integration experiment={} n={n_tokens} samples={samples} grouped_enabled={grouped_enabled} current_before_wall_ms={current_before_ms:?} current_before_median_ms={current_before_median:.3} current_before_p95_ms={current_before_p95:.3} candidate_wall_ms={candidate_ms:?} candidate_median_ms={candidate_median:.3} candidate_p95_ms={candidate_p95:.3} candidate_first_median_ms={candidate_first_median:.3} candidate_second_median_ms={candidate_second_median:.3} current_after_wall_ms={current_after_ms:?} current_after_median_ms={current_after_median:.3} current_after_p95_ms={current_after_p95:.3} wall_saving={wall_saving:.6} first_half_saving={first_half_saving:.6} second_half_saving={second_half_saving:.6} control_drift={control_drift:.6} candidate_drift={candidate_drift:.6} candidate_iq2_invocations={} candidate_iq3_invocations={} candidate_dispatches={} logits_sha256={} causal_digest={} continuation_causal_digest={}",
            if iq3_experiment { "iq3" } else { "iq2" },
            candidate[0].grouped_iq2_invocations,
            candidate[0].grouped_iq3_invocations,
            candidate_dispatches,
            digest_hex(Sha256::digest(bytemuck::cast_slice(&reference.logits))),
            digest_hex(reference.causal_digest),
            digest_hex(reference.continuation_causal_digest),
        );
        if iq3_experiment && samples >= 8 {
            assert!(
                control_drift <= 0.05,
                "N={n_tokens} grouped IQ3 control drift {control_drift:.3} exceeded 5%"
            );
            assert!(
                candidate_drift <= 0.05,
                "N={n_tokens} grouped IQ3 candidate drift {candidate_drift:.3} exceeded 5%"
            );
            if n_tokens == 128 {
                assert!(
                    first_half_saving >= 0.05 && second_half_saving >= 0.05,
                    "N=128 grouped IQ3 half savings {first_half_saving:.3}/{second_half_saving:.3} missed 5% gate"
                );
                assert!(
                    candidate_p95 <= current_before_p95.max(current_after_p95),
                    "N=128 grouped IQ3 p95 {candidate_p95:.3} exceeded control p95"
                );
            } else {
                assert!(
                    first_half_saving >= -0.02 && second_half_saving >= -0.02,
                    "N={n_tokens} grouped IQ3 half regression exceeded 2%: {first_half_saving:.3}/{second_half_saving:.3}"
                );
            }
        } else if grouped_enabled && n_tokens == 128 && samples >= 3 {
            assert!(
                wall_saving >= 0.15,
                "N=128 grouped expert wall saving {wall_saving:.3} missed 15% gate"
            );
            assert!(
                control_drift <= 0.05,
                "N=128 grouped expert control drift {control_drift:.3} exceeded 5%"
            );
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset"]
fn current_asset_packed_attention_split_attribution_packet() {
    const PREFIX_TOKENS: usize = 128;
    const CONTINUATION_TOKEN: u32 = 35;
    const ACCEPTED_COMBINED_LOG_SHA256: &str =
        "e735c536b742bb4f24c7dc89eaa7ceb40cbfa5ffdf65cf3e852130bf8be87cc0";
    const ACCEPTED_COMBINED_SHARES: [f64; 2] = [0.468_973_667_511_767_6, 0.472_380_181_289_259_2];
    const ORDINARY_ENCODERS: u64 = (DEEPSEEK_V4_LAYER_COUNT * 2) as u64;
    const SAMPLED_ENCODERS: u64 =
        (DEEPSEEK_V4_LAYER_COUNT * (prefill::PACKED_PREFILL_STAGE_KINDS.len() + 1)) as u64;

    struct Evidence {
        logits: Vec<f32>,
        hidden: Vec<f32>,
        causal_digest: [u8; 32],
        prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        continuation_logits: Vec<f32>,
        continuation_causal_digest: [u8; 32],
        committed_tokens: Vec<u32>,
        profile: prefill::PackedPrefillStageProfile,
        counters: crate::metal::KernelTraceCounters,
        dispatch_count: usize,
        dispatch_digest: [u8; 32],
        wall_ms: f64,
    }

    #[derive(Clone, Debug)]
    struct Summary {
        stage_ms: [f64; 4],
        cohort_stage_ms: [[f64; 4]; 3],
        cohort_command_ms: [f64; 3],
        cohort_gap_ms: [f64; 3],
        cohort_overlap_ms: [f64; 3],
        command_gpu_ms: f64,
        raw_span_ms: f64,
        gap_ms: f64,
        overlap_ms: f64,
        max_layer_coverage_error: f64,
        max_layer_transition_ambiguity: f64,
        max_single_transition_ambiguity: f64,
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn profile_digest_hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn stage_index(kind: prefill::PackedPrefillStageKind) -> usize {
        match kind {
            prefill::PackedPrefillStageKind::BeforeAttentionBody => 0,
            prefill::PackedPrefillStageKind::SparseIndexerPrepare
            | prefill::PackedPrefillStageKind::SparseIndexerScore
            | prefill::PackedPrefillStageKind::SparseSelection
            | prefill::PackedPrefillStageKind::AttentionCore
            | prefill::PackedPrefillStageKind::InverseRope => 1,
            prefill::PackedPrefillStageKind::AttentionOutputProjections => 2,
            prefill::PackedPrefillStageKind::AfterAttentionOutput => 3,
        }
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        sampled: bool,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct packed stage-profile session");
        crate::metal::dispatch_census_begin();
        let trace_guard = crate::metal::kernel_trace_begin();
        let started = std::time::Instant::now();
        let profile = session
            .execute_packed_tokens_with_stage_profile_for_test(ctx, prefix, true, sampled)
            .expect("execute packed stage-profile prefix");
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let counters = crate::metal::kernel_trace_snapshot();
        let census = crate::metal::dispatch_census_take();
        drop(trace_guard);
        let mut dispatch_hasher = Sha256::new();
        for row in &census {
            dispatch_hasher.update((row.family.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.family.as_bytes());
            dispatch_hasher.update((row.kernel.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.kernel.as_bytes());
            for extent in [
                row.grid_width,
                row.grid_height,
                row.grid_depth,
                row.threads_width,
                row.threads_height,
                row.threads_depth,
            ] {
                dispatch_hasher.update(extent.to_le_bytes());
            }
        }
        let dispatch_digest = dispatch_hasher.finalize().into();
        let logits = session
            .copy_logits_f32()
            .expect("copy packed stage-profile logits");
        let hidden = host_read_f32(
            session
                .final_normalized_hidden()
                .expect("packed stage-profile final hidden is visible"),
            "packed stage-profile final hidden",
        )
        .expect("copy packed stage-profile hidden");
        let snapshot = session
            .capture_causal_snapshot()
            .expect("capture packed stage-profile state");
        let causal_digest = *snapshot.causal_digest();
        let prefix_digest = *snapshot.prefix_digest();
        let compatibility_digest = *snapshot.compatibility_digest().as_bytes();
        session
            .restore_causal_snapshot(&snapshot)
            .expect("restore packed stage-profile state");
        session
            .forward_token(ctx, CONTINUATION_TOKEN)
            .expect("continue packed stage-profile state");
        let continuation_logits = session
            .copy_logits_f32()
            .expect("copy packed stage-profile continuation logits");
        let continuation = session
            .capture_causal_snapshot()
            .expect("capture packed stage-profile continuation state");
        let evidence = Evidence {
            logits,
            hidden,
            causal_digest,
            prefix_digest,
            compatibility_digest,
            continuation_logits,
            continuation_causal_digest: *continuation.causal_digest(),
            committed_tokens: session.committed_tokens().to_vec(),
            profile,
            counters,
            dispatch_count: census.len(),
            dispatch_digest,
            wall_ms,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    fn assert_exact(label: &str, actual: &Evidence, expected: &Evidence) {
        assert_eq!(
            bits(&actual.logits),
            bits(&expected.logits),
            "{label} logits"
        );
        assert_eq!(
            bits(&actual.hidden),
            bits(&expected.hidden),
            "{label} hidden"
        );
        assert_eq!(
            actual.causal_digest, expected.causal_digest,
            "{label} causal"
        );
        assert_eq!(
            actual.prefix_digest, expected.prefix_digest,
            "{label} prefix"
        );
        assert_eq!(
            actual.compatibility_digest, expected.compatibility_digest,
            "{label} compatibility"
        );
        assert_eq!(
            bits(&actual.continuation_logits),
            bits(&expected.continuation_logits),
            "{label} continuation logits"
        );
        assert_eq!(
            actual.continuation_causal_digest, expected.continuation_causal_digest,
            "{label} continuation causal"
        );
        assert_eq!(
            actual.committed_tokens, expected.committed_tokens,
            "{label} committed tokens"
        );
        assert_eq!(
            actual.counters.dispatches, expected.counters.dispatches,
            "{label} dispatch count"
        );
        assert_eq!(
            actual.dispatch_count, expected.dispatch_count,
            "{label} dispatch census count"
        );
        assert_eq!(
            actual.dispatch_digest, expected.dispatch_digest,
            "{label} dispatch order/shape digest"
        );
    }

    fn assert_topology(evidence: &Evidence, sampled: bool) {
        assert_eq!(evidence.profile.sampled, sampled);
        assert_eq!(
            evidence.profile.command_gpu_ms.len(),
            DEEPSEEK_V4_LAYER_COUNT
        );
        assert_eq!(evidence.counters.concurrent_encoders, 0);
        assert_eq!(
            evidence.counters.encoders,
            if sampled {
                SAMPLED_ENCODERS
            } else {
                ORDINARY_ENCODERS
            }
        );
        assert_eq!(
            evidence.counters.dispatches as usize,
            evidence.dispatch_count
        );
        assert_eq!(
            evidence.profile.sampled_layers.len(),
            if sampled { DEEPSEEK_V4_LAYER_COUNT } else { 0 }
        );
    }

    fn summarize(profile: &prefill::PackedPrefillStageProfile) -> Summary {
        assert!(profile.sampled);
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let mut stage_ms = [0.0; 4];
        let mut cohort_stage_ms = [[0.0; 4]; 3];
        let mut cohort_command_ms = [0.0; 3];
        let mut cohort_gap_ms = [0.0; 3];
        let mut cohort_overlap_ms = [0.0; 3];
        let mut raw_span_ms = 0.0;
        let mut gap_ms = 0.0;
        let mut overlap_ms = 0.0;
        let mut max_layer_coverage_error = 0.0f64;
        let mut max_layer_transition_ambiguity = 0.0f64;
        let mut max_single_transition_ambiguity = 0.0f64;
        for (layer, sampled_layer) in profile.sampled_layers.iter().enumerate() {
            assert_eq!(sampled_layer.layer, layer);
            assert_eq!(
                sampled_layer.stages.len(),
                prefill::PACKED_PREFILL_STAGE_KINDS.len()
            );
            let physical_stage_count = sampled_layer
                .stages
                .iter()
                .filter(|stage| stage.start_timestamp.is_some())
                .count();
            assert_eq!(
                sampled_layer.transitions.len(),
                physical_stage_count.saturating_sub(1)
            );
            assert!((sampled_layer.command_gpu_ms - profile.command_gpu_ms[layer]).abs() < 1e-9);
            max_layer_coverage_error =
                max_layer_coverage_error.max((sampled_layer.raw_coverage_assuming_ns - 1.0).abs());
            let cohort = match config.attention_kinds[layer] {
                AttentionKind::SlidingWindow => 0,
                AttentionKind::CompressedSparse => 1,
                AttentionKind::HeavilyCompressed => 2,
            };
            cohort_command_ms[cohort] += sampled_layer.command_gpu_ms;
            let mut layer_stage_ms = 0.0;
            for stage in &sampled_layer.stages {
                let index = stage_index(stage.kind);
                match (stage.start_timestamp, stage.end_timestamp) {
                    (Some(start), Some(end)) => {
                        assert!(start <= end);
                        assert_eq!(stage.duration_ticks, end - start);
                    }
                    (None, None)
                        if matches!(
                            stage.kind,
                            prefill::PackedPrefillStageKind::SparseIndexerPrepare
                                | prefill::PackedPrefillStageKind::SparseIndexerScore
                                | prefill::PackedPrefillStageKind::SparseSelection
                        ) =>
                    {
                        assert_eq!(stage.duration_ticks, 0);
                        assert_eq!(stage.duration_ms_scaled, 0.0);
                    }
                    _ => panic!("layer {layer} stage {:?} is malformed", stage.kind),
                }
                stage_ms[index] += stage.duration_ms_scaled;
                cohort_stage_ms[cohort][index] += stage.duration_ms_scaled;
                layer_stage_ms += stage.duration_ms_scaled;
            }
            for transition in &sampled_layer.transitions {
                assert_eq!(
                    transition.delta_ticks,
                    transition.gap_ticks as i128 - transition.overlap_ticks as i128
                );
                assert!(transition.gap_ticks == 0 || transition.overlap_ticks == 0);
                max_single_transition_ambiguity = max_single_transition_ambiguity.max(
                    (transition.gap_ms_scaled + transition.overlap_ms_scaled)
                        / sampled_layer.command_gpu_ms,
                );
            }
            let layer_artifact =
                sampled_layer.encoder_gap_ms_scaled + sampled_layer.encoder_overlap_ms_scaled;
            max_layer_transition_ambiguity =
                max_layer_transition_ambiguity.max(layer_artifact / sampled_layer.command_gpu_ms);
            assert!(
                (layer_stage_ms + sampled_layer.encoder_gap_ms_scaled
                    - sampled_layer.encoder_overlap_ms_scaled
                    - sampled_layer.command_gpu_ms)
                    .abs()
                    < 1e-6
            );
            raw_span_ms += sampled_layer.raw_span_ms_assuming_ns;
            gap_ms += sampled_layer.encoder_gap_ms_scaled;
            overlap_ms += sampled_layer.encoder_overlap_ms_scaled;
            cohort_gap_ms[cohort] += sampled_layer.encoder_gap_ms_scaled;
            cohort_overlap_ms[cohort] += sampled_layer.encoder_overlap_ms_scaled;
        }
        let command_gpu_ms = profile.command_gpu_ms.iter().sum::<f64>();
        assert!((stage_ms.iter().sum::<f64>() + gap_ms - overlap_ms - command_gpu_ms).abs() < 1e-6);
        Summary {
            stage_ms,
            cohort_stage_ms,
            cohort_command_ms,
            cohort_gap_ms,
            cohort_overlap_ms,
            command_gpu_ms,
            raw_span_ms,
            gap_ms,
            overlap_ms,
            max_layer_coverage_error,
            max_layer_transition_ambiguity,
            max_single_transition_ambiguity,
        }
    }

    fn drift(values: &[f64]) -> f64 {
        let minimum = values.iter().copied().reduce(f64::min).unwrap();
        let maximum = values.iter().copied().reduce(f64::max).unwrap();
        2.0 * (maximum - minimum) / (maximum + minimum)
    }

    fn median(values: &[f64]) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
        } else {
            values[values.len() / 2]
        }
    }

    fn stage_share(summary: &Summary, stage: usize) -> f64 {
        summary.stage_ms[stage] / summary.command_gpu_ms
    }

    fn cohort_stage_shares(summary: &Summary, stage: usize) -> [f64; 3] {
        std::array::from_fn(|cohort| {
            summary.cohort_stage_ms[cohort][stage] / summary.cohort_command_ms[cohort]
        })
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let grouped_default = prefill::packed_grouped_expert_enabled_for_test(&ctx);
    assert!(
        grouped_default,
        "packed stage attribution requires the qualified grouped IQ2 default"
    );
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, 129)
        .expect("plan packed stage-profile session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit packed stage-profile session");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let mut residency = realized.into_residency();
    let prefix = (0..PREFIX_TOKENS)
        .map(|index| [35, 201, 200, 34][index % 4])
        .collect::<Vec<_>>();

    let (next, warm_control) = execute(&ctx, residency, model_content_id, &prefix, false);
    let (next, warm_sampled) = execute(&ctx, next, model_content_id, &prefix, true);
    residency = next;
    assert_exact("sampled warm-up", &warm_sampled, &warm_control);
    assert_topology(&warm_control, false);
    assert_topology(&warm_sampled, true);

    let mut controls = Vec::with_capacity(3);
    let mut sampled_runs = Vec::with_capacity(2);
    for sampled in [false, true, false, true, false] {
        let (next, evidence) = execute(&ctx, residency, model_content_id, &prefix, sampled);
        residency = next;
        assert_exact("timed stage-profile arm", &evidence, &warm_control);
        assert_topology(&evidence, sampled);
        if sampled {
            sampled_runs.push(evidence);
        } else {
            controls.push(evidence);
        }
    }
    drop(residency);

    let control_gpu_ms = controls
        .iter()
        .map(|evidence| evidence.profile.command_gpu_ms.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let sampled_gpu_ms = sampled_runs
        .iter()
        .map(|evidence| evidence.profile.command_gpu_ms.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let control_wall_ms = controls
        .iter()
        .map(|evidence| evidence.wall_ms)
        .collect::<Vec<_>>();
    let sampled_wall_ms = sampled_runs
        .iter()
        .map(|evidence| evidence.wall_ms)
        .collect::<Vec<_>>();
    let control_drift = drift(&control_gpu_ms);
    let sampled_drift = drift(&sampled_gpu_ms);
    let interpolated_control_gpu_ms = [
        (control_gpu_ms[0] + control_gpu_ms[1]) * 0.5,
        (control_gpu_ms[1] + control_gpu_ms[2]) * 0.5,
    ];
    let perturbation = std::array::from_fn::<_, 2, _>(|index| {
        sampled_gpu_ms[index] / interpolated_control_gpu_ms[index] - 1.0
    });
    let summaries = sampled_runs
        .iter()
        .map(|evidence| summarize(&evidence.profile))
        .collect::<Vec<_>>();
    let body_shares = [stage_share(&summaries[0], 1), stage_share(&summaries[1], 1)];
    let output_shares = [stage_share(&summaries[0], 2), stage_share(&summaries[1], 2)];
    let combined_shares: [f64; 2] =
        std::array::from_fn(|index| body_shares[index] + output_shares[index]);
    let body_repeat_delta = (body_shares[0] - body_shares[1]).abs();
    let output_repeat_delta = (output_shares[0] - output_shares[1]).abs();
    let combined_repeat_delta = (combined_shares[0] - combined_shares[1]).abs();
    let combined_reproduction_delta: [f64; 2] = std::array::from_fn(|index| {
        (combined_shares[index] - ACCEPTED_COMBINED_SHARES[index]).abs()
    });
    let cohort_body_shares = [
        cohort_stage_shares(&summaries[0], 1),
        cohort_stage_shares(&summaries[1], 1),
    ];
    let cohort_output_shares = [
        cohort_stage_shares(&summaries[0], 2),
        cohort_stage_shares(&summaries[1], 2),
    ];
    let cohort_body_share_delta: [f64; 3] = std::array::from_fn(|cohort| {
        (cohort_body_shares[0][cohort] - cohort_body_shares[1][cohort]).abs()
    });
    let cohort_output_share_delta: [f64; 3] = std::array::from_fn(|cohort| {
        (cohort_output_shares[0][cohort] - cohort_output_shares[1][cohort]).abs()
    });
    let transition_uncertainty = summaries
        .iter()
        .map(|summary| (summary.gap_ms + summary.overlap_ms) / summary.command_gpu_ms)
        .reduce(f64::max)
        .unwrap();
    let topology_uncertainty = perturbation
        .iter()
        .map(|value| value.abs())
        .reduce(f64::max)
        .unwrap();
    let coverage_uncertainty = summaries
        .iter()
        .map(|summary| (summary.raw_span_ms / summary.command_gpu_ms - 1.0).abs())
        .reduce(f64::max)
        .unwrap();
    let common_uncertainty = transition_uncertainty.max(topology_uncertainty);
    let body_observer_uncertainty = common_uncertainty.max(body_repeat_delta);
    let output_observer_uncertainty = common_uncertainty.max(output_repeat_delta);
    let mean_body_share = (body_shares[0] + body_shares[1]) * 0.5;
    let mean_output_share = (output_shares[0] + output_shares[1]) * 0.5;
    let lower_body_share = (mean_body_share - body_observer_uncertainty).max(0.0);
    let lower_output_share = (mean_output_share - output_observer_uncertainty).max(0.0);
    let normalized_body_ms = [
        body_shares[0] * interpolated_control_gpu_ms[0],
        body_shares[1] * interpolated_control_gpu_ms[1],
    ];
    let normalized_output_ms = [
        output_shares[0] * interpolated_control_gpu_ms[0],
        output_shares[1] * interpolated_control_gpu_ms[1],
    ];
    let ordinary_gpu_median = median(&control_gpu_ms);
    let normalized_body_median_ms = median(&normalized_body_ms);
    let normalized_output_median_ms = median(&normalized_output_ms);
    let lower_body_ms =
        (normalized_body_median_ms - body_observer_uncertainty * ordinary_gpu_median).max(0.0);
    let lower_output_ms =
        (normalized_output_median_ms - output_observer_uncertainty * ordinary_gpu_median).max(0.0);
    let mean_cohort_body_share: [f64; 3] = std::array::from_fn(|cohort| {
        (cohort_body_shares[0][cohort] + cohort_body_shares[1][cohort]) * 0.5
    });
    let mean_cohort_output_share: [f64; 3] = std::array::from_fn(|cohort| {
        (cohort_output_shares[0][cohort] + cohort_output_shares[1][cohort]) * 0.5
    });
    let body_authorized = lower_body_share >= 0.15
        && lower_body_ms >= 150.0
        && mean_cohort_body_share[1] >= 0.10
        && mean_cohort_body_share[2] >= 0.10;
    let output_authorized = lower_output_share >= 0.15
        && lower_output_ms >= 150.0
        && mean_cohort_output_share[1] >= 0.10
        && mean_cohort_output_share[2] >= 0.10;

    eprintln!(
        "deepseek_v4 packed_attention_split_profile n={PREFIX_TOKENS} grouped_default={grouped_default} control_gpu_ms={control_gpu_ms:?} interpolated_control_gpu_ms={interpolated_control_gpu_ms:?} sampled_gpu_ms={sampled_gpu_ms:?} audit_inclusive_control_wall_ms={control_wall_ms:?} audit_inclusive_sampled_wall_ms={sampled_wall_ms:?} control_drift={control_drift:.6} sampled_drift={sampled_drift:.6} sampled_perturbation={perturbation:?} body_shares={body_shares:?} output_shares={output_shares:?} combined_shares={combined_shares:?} accepted_combined_log_sha256={ACCEPTED_COMBINED_LOG_SHA256} accepted_combined_shares={ACCEPTED_COMBINED_SHARES:?} combined_reproduction_delta={combined_reproduction_delta:?} body_repeat_delta={body_repeat_delta:.6} output_repeat_delta={output_repeat_delta:.6} combined_repeat_delta={combined_repeat_delta:.6} cohort_body_shares={cohort_body_shares:?} cohort_output_shares={cohort_output_shares:?} cohort_body_share_delta={cohort_body_share_delta:?} cohort_output_share_delta={cohort_output_share_delta:?} normalized_body_ms={normalized_body_ms:?} normalized_output_ms={normalized_output_ms:?} normalized_body_median_ms={normalized_body_median_ms:.3} normalized_output_median_ms={normalized_output_median_ms:.3} lower_body_share={lower_body_share:.6} lower_output_share={lower_output_share:.6} lower_body_ms={lower_body_ms:.3} lower_output_ms={lower_output_ms:.3} transition_uncertainty={transition_uncertainty:.6} topology_uncertainty={topology_uncertainty:.6} coverage_uncertainty={coverage_uncertainty:.6} body_observer_uncertainty={body_observer_uncertainty:.6} output_observer_uncertainty={output_observer_uncertainty:.6} body_authorized={body_authorized} output_authorized={output_authorized} dispatches={} ordinary_encoders={ORDINARY_ENCODERS} sampled_encoders={SAMPLED_ENCODERS} dispatch_sha256={} model_content_id={} logits_sha256={} hidden_sha256={} causal_digest={} prefix_digest={} compatibility_digest={} continuation_logits_sha256={} continuation_causal_digest={} committed_tokens_sha256={}",
        warm_control.dispatch_count,
        profile_digest_hex(warm_control.dispatch_digest),
        profile_digest_hex(model_content_id.as_bytes()),
        profile_digest_hex(Sha256::digest(bytemuck::cast_slice(&warm_control.logits))),
        profile_digest_hex(Sha256::digest(bytemuck::cast_slice(&warm_control.hidden))),
        profile_digest_hex(warm_control.causal_digest),
        profile_digest_hex(warm_control.prefix_digest),
        profile_digest_hex(warm_control.compatibility_digest),
        profile_digest_hex(Sha256::digest(bytemuck::cast_slice(
            &warm_control.continuation_logits,
        ))),
        profile_digest_hex(warm_control.continuation_causal_digest),
        profile_digest_hex(Sha256::digest(bytemuck::cast_slice(
            &warm_control.committed_tokens,
        ))),
    );
    for (index, control) in controls.iter().enumerate() {
        eprintln!(
            "deepseek_v4 packed_attention_split_profile_control index={index} layer_command_gpu_ms={:?}",
            control.profile.command_gpu_ms
        );
    }
    for (index, summary) in summaries.iter().enumerate() {
        eprintln!(
            "deepseek_v4 packed_attention_split_profile_sample index={index} command_gpu_ms={:.3} raw_span_ms={:.3} raw_coverage={:.6} gap_ms={:.3} overlap_ms={:.3} transition_ambiguity={:.6} max_layer_coverage_error={:.6} max_layer_transition_ambiguity={:.6} max_single_transition_ambiguity={:.6} stage_ms={:?} body_share={:.6} output_share={:.6} combined_share={:.6} cohort_stage_ms={:?} cohort_command_ms={:?} cohort_body_share={:?} cohort_output_share={:?} cohort_gap_ms={:?} cohort_overlap_ms={:?}",
            summary.command_gpu_ms,
            summary.raw_span_ms,
            summary.raw_span_ms / summary.command_gpu_ms,
            summary.gap_ms,
            summary.overlap_ms,
            (summary.gap_ms + summary.overlap_ms) / summary.command_gpu_ms,
            summary.max_layer_coverage_error,
            summary.max_layer_transition_ambiguity,
            summary.max_single_transition_ambiguity,
            summary.stage_ms,
            body_shares[index],
            output_shares[index],
            combined_shares[index],
            summary.cohort_stage_ms,
            summary.cohort_command_ms,
            cohort_body_shares[index],
            cohort_output_shares[index],
            summary.cohort_gap_ms,
            summary.cohort_overlap_ms,
        );
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        for layer in &sampled_runs[index].profile.sampled_layers {
            let stage_ticks: [u64; 6] =
                std::array::from_fn(|stage| layer.stages[stage].duration_ticks);
            let stage_ms: [f64; 6] =
                std::array::from_fn(|stage| layer.stages[stage].duration_ms_scaled);
            let layer_body_ms = layer
                .stages
                .iter()
                .filter(|stage| stage_index(stage.kind) == 1)
                .map(|stage| stage.duration_ms_scaled)
                .sum::<f64>();
            let layer_output_ms = layer
                .stages
                .iter()
                .find(|stage| stage_index(stage.kind) == 2)
                .unwrap()
                .duration_ms_scaled;
            eprintln!(
                "deepseek_v4 packed_attention_split_profile_layer sample={index} layer={} attention_kind={:?} command_gpu_ms={:.6} raw_span_ticks={} raw_span_ms={:.6} raw_coverage={:.9} gap_ms={:.6} overlap_ms={:.6} transition_ambiguity={:.9} body_share={:.9} output_share={:.9} combined_share={:.9} stage_ticks={stage_ticks:?} stage_ms={stage_ms:?} transitions={:?}",
                layer.layer,
                config.attention_kinds[layer.layer],
                layer.command_gpu_ms,
                layer.sampled_span_ticks,
                layer.raw_span_ms_assuming_ns,
                layer.raw_coverage_assuming_ns,
                layer.encoder_gap_ms_scaled,
                layer.encoder_overlap_ms_scaled,
                (layer.encoder_gap_ms_scaled + layer.encoder_overlap_ms_scaled)
                    / layer.command_gpu_ms,
                layer_body_ms / layer.command_gpu_ms,
                layer_output_ms / layer.command_gpu_ms,
                (layer_body_ms + layer_output_ms) / layer.command_gpu_ms,
                layer.transitions,
            );
        }
    }

    assert!(control_drift <= 0.05, "control GPU drift exceeded 5%");
    assert!(sampled_drift <= 0.05, "sampled GPU drift exceeded 5%");
    assert!(
        perturbation.iter().all(|&value| value.abs() <= 0.10),
        "sampled GPU perturbation exceeded 10%: {perturbation:?}"
    );
    assert!(
        summaries
            .iter()
            .all(|summary| summary.max_layer_coverage_error <= 0.02),
        "per-layer raw timestamp coverage exceeded 2%"
    );
    assert!(
        coverage_uncertainty <= 0.005,
        "aggregate raw timestamp coverage exceeded 0.5%"
    );
    assert!(
        summaries
            .iter()
            .all(|summary| summary.max_single_transition_ambiguity <= 0.05),
        "single transition ambiguity exceeded 5%"
    );
    assert!(
        summaries
            .iter()
            .all(|summary| summary.max_layer_transition_ambiguity <= 0.10),
        "per-layer combined transition ambiguity exceeded 10%"
    );
    assert!(
        transition_uncertainty <= 0.025,
        "aggregate transition ambiguity exceeded 2.5%"
    );
    assert!(
        body_repeat_delta <= 0.02 && output_repeat_delta <= 0.02,
        "body or output share changed by more than two points"
    );
    assert!(
        combined_repeat_delta <= 0.02,
        "combined attention share changed by more than two points"
    );
    assert!(
        combined_reproduction_delta
            .iter()
            .all(|&delta| delta <= 0.02),
        "split attention share did not reproduce the accepted combined envelope"
    );
    assert!(
        cohort_body_share_delta[1] <= 0.03
            && cohort_body_share_delta[2] <= 0.03
            && cohort_output_share_delta[1] <= 0.03
            && cohort_output_share_delta[2] <= 0.03,
        "CSA/HCA body or output shares changed by more than three points"
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset"]
fn current_asset_packed_post_route_stage_attribution_packet() {
    const PREFIX_TOKENS: usize = 128;
    const CONTINUATION_TOKEN: u32 = 35;
    const STAGE_COUNT: usize = 4;
    const COHORT_COUNT: usize = 3;
    const ORDINARY_ENCODERS: u64 = (DEEPSEEK_V4_LAYER_COUNT * 2) as u64;
    const SAMPLED_ENCODERS: u64 = (DEEPSEEK_V4_LAYER_COUNT * 5) as u64;

    struct Evidence {
        logits: Vec<f32>,
        hidden: Vec<f32>,
        causal_digest: [u8; 32],
        prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        continuation_logits: Vec<f32>,
        continuation_causal_digest: [u8; 32],
        committed_tokens: Vec<u32>,
        profile: prefill::PackedPostRouteStageProfile,
        counters: crate::metal::KernelTraceCounters,
        dispatch_count: usize,
        dispatch_digest: [u8; 32],
        wall_ms: f64,
    }

    #[derive(Clone, Debug)]
    struct Summary {
        stage_ms: [f64; STAGE_COUNT],
        cohort_stage_ms: [[f64; STAGE_COUNT]; COHORT_COUNT],
        cohort_command_ms: [f64; COHORT_COUNT],
        cohort_bucket_count: [usize; COHORT_COUNT],
        cohort_layer_count: [usize; COHORT_COUNT],
        command_gpu_ms: f64,
        raw_span_ms: f64,
        gap_ms: f64,
        overlap_ms: f64,
        max_layer_coverage_error: f64,
        max_layer_transition_ambiguity: f64,
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn stage_index(kind: prefill::PackedPostRouteStageKind) -> usize {
        match kind {
            prefill::PackedPostRouteStageKind::RoutedExperts => 0,
            prefill::PackedPostRouteStageKind::SharedExpert => 1,
            prefill::PackedPostRouteStageKind::ExpertCombine => 2,
            prefill::PackedPostRouteStageKind::HyperPostAndHead => 3,
            prefill::PackedPostRouteStageKind::RoutedGateUp
            | prefill::PackedPostRouteStageKind::RoutedSwiGlu
            | prefill::PackedPostRouteStageKind::RoutedDown => {
                panic!("N=128 post-route packet unexpectedly selected BM16 stages")
            }
        }
    }

    fn cohort(metadata: &prefill::PackedPostRouteLayerMetadata) -> usize {
        if metadata.grouped_iq2 {
            assert_eq!(metadata.gate_dtype, GgmlType::IQ2_XS);
            assert_eq!(metadata.up_dtype, GgmlType::IQ2_XS);
            assert_eq!(metadata.down_dtype, GgmlType::IQ3_XXS);
            0
        } else if metadata.gate_dtype == GgmlType::IQ3_XXS
            && metadata.up_dtype == GgmlType::IQ3_XXS
            && metadata.down_dtype == GgmlType::IQ3_XXS
        {
            1
        } else {
            assert_eq!(metadata.down_dtype, GgmlType::MXFP4);
            2
        }
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        sampled: bool,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct packed post-route stage-profile session");
        crate::metal::dispatch_census_begin();
        let trace_guard = crate::metal::kernel_trace_begin();
        let started = std::time::Instant::now();
        let profile = session
            .execute_packed_tokens_with_post_route_stage_profile_for_test(
                ctx, prefix, true, sampled, false,
            )
            .expect("execute packed post-route stage-profile prefix");
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let counters = crate::metal::kernel_trace_snapshot();
        let census = crate::metal::dispatch_census_take();
        drop(trace_guard);
        let mut dispatch_hasher = Sha256::new();
        for row in &census {
            dispatch_hasher.update((row.family.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.family.as_bytes());
            dispatch_hasher.update((row.kernel.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.kernel.as_bytes());
            for extent in [
                row.grid_width,
                row.grid_height,
                row.grid_depth,
                row.threads_width,
                row.threads_height,
                row.threads_depth,
            ] {
                dispatch_hasher.update(extent.to_le_bytes());
            }
        }
        let dispatch_digest = dispatch_hasher.finalize().into();
        let logits = session
            .copy_logits_f32()
            .expect("copy packed post-route stage-profile logits");
        let hidden = host_read_f32(
            session
                .final_normalized_hidden()
                .expect("packed post-route stage-profile final hidden is visible"),
            "packed post-route stage-profile final hidden",
        )
        .expect("copy packed post-route stage-profile hidden");
        let snapshot = session
            .capture_causal_snapshot()
            .expect("capture packed post-route stage-profile state");
        let causal_digest = *snapshot.causal_digest();
        let prefix_digest = *snapshot.prefix_digest();
        let compatibility_digest = *snapshot.compatibility_digest().as_bytes();
        session
            .restore_causal_snapshot(&snapshot)
            .expect("restore packed post-route stage-profile state");
        session
            .forward_token(ctx, CONTINUATION_TOKEN)
            .expect("continue packed post-route stage-profile state");
        let continuation_logits = session
            .copy_logits_f32()
            .expect("copy packed post-route continuation logits");
        let continuation = session
            .capture_causal_snapshot()
            .expect("capture packed post-route continuation state");
        let evidence = Evidence {
            logits,
            hidden,
            causal_digest,
            prefix_digest,
            compatibility_digest,
            continuation_logits,
            continuation_causal_digest: *continuation.causal_digest(),
            committed_tokens: session.committed_tokens().to_vec(),
            profile,
            counters,
            dispatch_count: census.len(),
            dispatch_digest,
            wall_ms,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    fn assert_exact(label: &str, actual: &Evidence, expected: &Evidence) {
        assert_eq!(
            bits(&actual.logits),
            bits(&expected.logits),
            "{label} logits"
        );
        assert_eq!(
            bits(&actual.hidden),
            bits(&expected.hidden),
            "{label} hidden"
        );
        assert_eq!(
            actual.causal_digest, expected.causal_digest,
            "{label} causal"
        );
        assert_eq!(
            actual.prefix_digest, expected.prefix_digest,
            "{label} prefix"
        );
        assert_eq!(
            actual.compatibility_digest, expected.compatibility_digest,
            "{label} compatibility"
        );
        assert_eq!(
            bits(&actual.continuation_logits),
            bits(&expected.continuation_logits),
            "{label} continuation logits"
        );
        assert_eq!(
            actual.continuation_causal_digest, expected.continuation_causal_digest,
            "{label} continuation causal"
        );
        assert_eq!(
            actual.committed_tokens, expected.committed_tokens,
            "{label} committed tokens"
        );
        assert_eq!(
            actual.profile.metadata, expected.profile.metadata,
            "{label} metadata"
        );
        assert_eq!(
            actual.counters.dispatches, expected.counters.dispatches,
            "{label} dispatch count"
        );
        assert_eq!(
            actual.dispatch_count, expected.dispatch_count,
            "{label} dispatch census count"
        );
        assert_eq!(
            actual.dispatch_digest, expected.dispatch_digest,
            "{label} dispatch order/shape digest"
        );
    }

    fn assert_topology(evidence: &Evidence, sampled: bool) {
        assert_eq!(evidence.profile.sampled, sampled);
        assert_eq!(
            evidence.profile.command_gpu_ms.len(),
            DEEPSEEK_V4_LAYER_COUNT
        );
        assert_eq!(evidence.profile.metadata.len(), DEEPSEEK_V4_LAYER_COUNT);
        assert_eq!(evidence.counters.concurrent_encoders, 0);
        assert_eq!(
            evidence.counters.encoders,
            if sampled {
                SAMPLED_ENCODERS
            } else {
                ORDINARY_ENCODERS
            }
        );
        assert_eq!(
            evidence.counters.dispatches as usize,
            evidence.dispatch_count
        );
        assert_eq!(
            evidence.profile.sampled_layers.len(),
            if sampled { DEEPSEEK_V4_LAYER_COUNT } else { 0 }
        );
    }

    fn summarize(profile: &prefill::PackedPostRouteStageProfile) -> Summary {
        assert!(profile.sampled);
        let mut stage_ms = [0.0; STAGE_COUNT];
        let mut cohort_stage_ms = [[0.0; STAGE_COUNT]; COHORT_COUNT];
        let mut cohort_command_ms = [0.0; COHORT_COUNT];
        let mut cohort_bucket_count = [0usize; COHORT_COUNT];
        let mut cohort_layer_count = [0usize; COHORT_COUNT];
        let mut raw_span_ms = 0.0;
        let mut gap_ms = 0.0;
        let mut overlap_ms = 0.0;
        let mut max_layer_coverage_error = 0.0f64;
        let mut max_layer_transition_ambiguity = 0.0f64;
        for (layer, (sampled_layer, metadata)) in profile
            .sampled_layers
            .iter()
            .zip(&profile.metadata)
            .enumerate()
        {
            assert_eq!(sampled_layer.layer, layer);
            assert_eq!(metadata.layer, layer);
            assert_eq!(sampled_layer.stages.len(), STAGE_COUNT);
            assert!((sampled_layer.command_gpu_ms - profile.command_gpu_ms[layer]).abs() < 1e-9);
            let cohort = cohort(metadata);
            cohort_command_ms[cohort] += sampled_layer.command_gpu_ms;
            cohort_bucket_count[cohort] += metadata.bucket_count;
            cohort_layer_count[cohort] += 1;
            max_layer_coverage_error =
                max_layer_coverage_error.max((sampled_layer.raw_coverage_assuming_ns - 1.0).abs());
            max_layer_transition_ambiguity = max_layer_transition_ambiguity.max(
                (sampled_layer.encoder_gap_ms_scaled + sampled_layer.encoder_overlap_ms_scaled)
                    / sampled_layer.command_gpu_ms,
            );
            let mut layer_stage_ms = 0.0;
            for (expected_index, stage) in sampled_layer.stages.iter().enumerate() {
                let index = stage_index(stage.kind);
                assert_eq!(index, expected_index);
                assert!(stage.start_timestamp <= stage.end_timestamp);
                assert_eq!(
                    stage.duration_ticks,
                    stage.end_timestamp - stage.start_timestamp
                );
                stage_ms[index] += stage.duration_ms_scaled;
                cohort_stage_ms[cohort][index] += stage.duration_ms_scaled;
                layer_stage_ms += stage.duration_ms_scaled;
            }
            assert!(
                (layer_stage_ms + sampled_layer.encoder_gap_ms_scaled
                    - sampled_layer.encoder_overlap_ms_scaled
                    - sampled_layer.command_gpu_ms)
                    .abs()
                    < 1e-6
            );
            raw_span_ms += sampled_layer.raw_span_ms_assuming_ns;
            gap_ms += sampled_layer.encoder_gap_ms_scaled;
            overlap_ms += sampled_layer.encoder_overlap_ms_scaled;
        }
        let command_gpu_ms = profile.command_gpu_ms.iter().sum::<f64>();
        assert!((stage_ms.iter().sum::<f64>() + gap_ms - overlap_ms - command_gpu_ms).abs() < 1e-6);
        Summary {
            stage_ms,
            cohort_stage_ms,
            cohort_command_ms,
            cohort_bucket_count,
            cohort_layer_count,
            command_gpu_ms,
            raw_span_ms,
            gap_ms,
            overlap_ms,
            max_layer_coverage_error,
            max_layer_transition_ambiguity,
        }
    }

    fn drift(values: &[f64]) -> f64 {
        let minimum = values.iter().copied().reduce(f64::min).unwrap();
        let maximum = values.iter().copied().reduce(f64::max).unwrap();
        2.0 * (maximum - minimum) / (maximum + minimum)
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    assert!(
        prefill::packed_grouped_expert_enabled_for_test(&ctx),
        "post-route attribution requires the qualified grouped IQ2 default"
    );
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, 129)
        .expect("plan packed post-route stage-profile session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit packed post-route stage-profile session");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let mut residency = realized.into_residency();
    let prefix = (0..PREFIX_TOKENS)
        .map(|index| [35, 201, 200, 34][index % 4])
        .collect::<Vec<_>>();

    let (next, warm_control) = execute(&ctx, residency, model_content_id, &prefix, false);
    let (next, warm_sampled) = execute(&ctx, next, model_content_id, &prefix, true);
    residency = next;
    assert_exact("sampled warm-up", &warm_sampled, &warm_control);
    assert_topology(&warm_control, false);
    assert_topology(&warm_sampled, true);

    let mut controls = Vec::with_capacity(4);
    let mut sampled_runs = Vec::with_capacity(3);
    for sampled in [false, true, false, true, false, true, false] {
        let (next, evidence) = execute(&ctx, residency, model_content_id, &prefix, sampled);
        residency = next;
        assert_exact(
            "timed post-route stage-profile arm",
            &evidence,
            &warm_control,
        );
        assert_topology(&evidence, sampled);
        if sampled {
            sampled_runs.push(evidence);
        } else {
            controls.push(evidence);
        }
    }
    drop(residency);

    let control_gpu_ms = controls
        .iter()
        .map(|evidence| evidence.profile.command_gpu_ms.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let sampled_gpu_ms = sampled_runs
        .iter()
        .map(|evidence| evidence.profile.command_gpu_ms.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let control_wall_ms = controls
        .iter()
        .map(|evidence| evidence.wall_ms)
        .collect::<Vec<_>>();
    let sampled_wall_ms = sampled_runs
        .iter()
        .map(|evidence| evidence.wall_ms)
        .collect::<Vec<_>>();
    let interpolated_control_gpu_ms = control_gpu_ms
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect::<Vec<_>>();
    let perturbation = sampled_gpu_ms
        .iter()
        .zip(&interpolated_control_gpu_ms)
        .map(|(sampled, control)| sampled / control - 1.0)
        .collect::<Vec<_>>();
    let summaries = sampled_runs
        .iter()
        .map(|evidence| summarize(&evidence.profile))
        .collect::<Vec<_>>();
    let sample_is_valid = summaries
        .iter()
        .zip(&perturbation)
        .map(|(summary, perturbation)| {
            summary.max_layer_coverage_error <= 0.02
                && (summary.raw_span_ms / summary.command_gpu_ms - 1.0).abs() <= 0.005
                && summary.max_layer_transition_ambiguity <= 0.10
                && (summary.gap_ms + summary.overlap_ms) / summary.command_gpu_ms <= 0.025
                && perturbation.abs() <= 0.10
        })
        .collect::<Vec<_>>();
    let accepted_indices = sample_is_valid
        .iter()
        .enumerate()
        .filter_map(|(index, accepted)| accepted.then_some(index))
        .collect::<Vec<_>>();
    assert!(
        accepted_indices.len() >= 2 && accepted_indices.len() + 1 >= summaries.len(),
        "post-route profiler retained fewer than two stable samples: {sample_is_valid:?}"
    );
    let accepted_gpu_ms = accepted_indices
        .iter()
        .map(|&index| sampled_gpu_ms[index])
        .collect::<Vec<_>>();
    let stage_shares = summaries
        .iter()
        .map(|summary| {
            std::array::from_fn::<_, STAGE_COUNT, _>(|stage| {
                summary.stage_ms[stage] / summary.command_gpu_ms
            })
        })
        .collect::<Vec<_>>();
    let stage_repeat_delta: [f64; STAGE_COUNT] = std::array::from_fn(|stage| {
        let values = accepted_indices
            .iter()
            .map(|&index| stage_shares[index][stage]);
        let minimum = values.clone().reduce(f64::min).unwrap();
        let maximum = values.reduce(f64::max).unwrap();
        maximum - minimum
    });
    let cohort_stage_shares = summaries
        .iter()
        .map(|summary| {
            std::array::from_fn(|cohort| {
                std::array::from_fn(|stage| {
                    summary.cohort_stage_ms[cohort][stage] / summary.cohort_command_ms[cohort]
                })
            })
        })
        .collect::<Vec<[[f64; STAGE_COUNT]; COHORT_COUNT]>>();
    let cohort_stage_repeat_delta: [[f64; STAGE_COUNT]; COHORT_COUNT] =
        std::array::from_fn(|cohort| {
            std::array::from_fn(|stage| {
                let values = accepted_indices
                    .iter()
                    .map(|&index| cohort_stage_shares[index][cohort][stage]);
                let minimum = values.clone().reduce(f64::min).unwrap();
                let maximum = values.reduce(f64::max).unwrap();
                maximum - minimum
            })
        });
    let transition_uncertainty = accepted_indices
        .iter()
        .map(|&index| {
            let summary = &summaries[index];
            (summary.gap_ms + summary.overlap_ms) / summary.command_gpu_ms
        })
        .reduce(f64::max)
        .unwrap();
    let coverage_uncertainty = accepted_indices
        .iter()
        .map(|&index| {
            let summary = &summaries[index];
            (summary.raw_span_ms / summary.command_gpu_ms - 1.0).abs()
        })
        .reduce(f64::max)
        .unwrap();

    eprintln!(
        "deepseek_v4 packed_post_route_stage_profile n={PREFIX_TOKENS} control_gpu_ms={control_gpu_ms:?} interpolated_control_gpu_ms={interpolated_control_gpu_ms:?} sampled_gpu_ms={sampled_gpu_ms:?} accepted_sample_indices={accepted_indices:?} accepted_gpu_ms={accepted_gpu_ms:?} sample_is_valid={sample_is_valid:?} control_wall_ms={control_wall_ms:?} sampled_wall_ms={sampled_wall_ms:?} control_drift={:.6} accepted_sampled_drift={:.6} sampled_perturbation={perturbation:?} stage_shares={stage_shares:?} stage_repeat_delta={stage_repeat_delta:?} cohort_stage_shares={cohort_stage_shares:?} cohort_stage_repeat_delta={cohort_stage_repeat_delta:?} transition_uncertainty={transition_uncertainty:.6} coverage_uncertainty={coverage_uncertainty:.6} dispatches={} ordinary_encoders={ORDINARY_ENCODERS} sampled_encoders={SAMPLED_ENCODERS} dispatch_sha256={} model_content_id={} logits_sha256={} hidden_sha256={} causal_digest={} prefix_digest={} compatibility_digest={} continuation_logits_sha256={} continuation_causal_digest={} committed_tokens_sha256={}",
        drift(&control_gpu_ms),
        drift(&accepted_gpu_ms),
        warm_control.dispatch_count,
        hex(warm_control.dispatch_digest),
        hex(model_content_id.as_bytes()),
        hex(Sha256::digest(bytemuck::cast_slice(&warm_control.logits))),
        hex(Sha256::digest(bytemuck::cast_slice(&warm_control.hidden))),
        hex(warm_control.causal_digest),
        hex(warm_control.prefix_digest),
        hex(warm_control.compatibility_digest),
        hex(Sha256::digest(bytemuck::cast_slice(
            &warm_control.continuation_logits,
        ))),
        hex(warm_control.continuation_causal_digest),
        hex(Sha256::digest(bytemuck::cast_slice(
            &warm_control.committed_tokens,
        ))),
    );
    for (index, summary) in summaries.iter().enumerate() {
        eprintln!(
            "deepseek_v4 packed_post_route_stage_profile_sample index={index} accepted={} perturbation={:.6} command_gpu_ms={:.3} raw_span_ms={:.3} raw_coverage={:.6} gap_ms={:.3} overlap_ms={:.3} transition_ambiguity={:.6} max_layer_coverage_error={:.6} max_layer_transition_ambiguity={:.6} stage_ms={:?} cohort_stage_ms={:?} cohort_command_ms={:?} cohort_bucket_count={:?} cohort_layer_count={:?}",
            sample_is_valid[index],
            perturbation[index],
            summary.command_gpu_ms,
            summary.raw_span_ms,
            summary.raw_span_ms / summary.command_gpu_ms,
            summary.gap_ms,
            summary.overlap_ms,
            (summary.gap_ms + summary.overlap_ms) / summary.command_gpu_ms,
            summary.max_layer_coverage_error,
            summary.max_layer_transition_ambiguity,
            summary.stage_ms,
            summary.cohort_stage_ms,
            summary.cohort_command_ms,
            summary.cohort_bucket_count,
            summary.cohort_layer_count,
        );
        for (layer, metadata) in sampled_runs[index].profile.metadata.iter().enumerate() {
            let sampled_layer = &sampled_runs[index].profile.sampled_layers[layer];
            let stage_ms: [f64; STAGE_COUNT] =
                std::array::from_fn(|stage| sampled_layer.stages[stage].duration_ms_scaled);
            eprintln!(
                "deepseek_v4 packed_post_route_stage_profile_layer sample={index} layer={layer} cohort={} gate={:?} up={:?} down={:?} grouped_iq2={} grouped_iq3={} buckets={} command_gpu_ms={:.6} raw_coverage={:.9} gap_ms={:.6} overlap_ms={:.6} stage_ms={stage_ms:?}",
                cohort(metadata),
                metadata.gate_dtype,
                metadata.up_dtype,
                metadata.down_dtype,
                metadata.grouped_iq2,
                metadata.grouped_iq3,
                metadata.bucket_count,
                sampled_layer.command_gpu_ms,
                sampled_layer.raw_coverage_assuming_ns,
                sampled_layer.encoder_gap_ms_scaled,
                sampled_layer.encoder_overlap_ms_scaled,
            );
        }
    }

    assert!(
        summaries
            .iter()
            .all(|summary| summary.cohort_layer_count == [25, 16, 2])
    );
    assert!(
        drift(&control_gpu_ms) <= 0.05,
        "control GPU drift exceeded 5%"
    );
    assert!(
        drift(&accepted_gpu_ms) <= 0.05,
        "accepted sampled GPU drift exceeded 5%"
    );
    assert!(
        accepted_indices
            .iter()
            .all(|&index| perturbation[index].abs() <= 0.10),
        "accepted sampled GPU perturbation exceeded 10%: {perturbation:?}"
    );
    assert!(
        accepted_indices
            .iter()
            .all(|&index| summaries[index].max_layer_coverage_error <= 0.02),
        "per-layer raw timestamp coverage exceeded 2%"
    );
    assert!(
        coverage_uncertainty <= 0.005,
        "aggregate raw timestamp coverage exceeded 0.5%"
    );
    assert!(
        accepted_indices
            .iter()
            .all(|&index| summaries[index].max_layer_transition_ambiguity <= 0.10),
        "per-layer combined transition ambiguity exceeded 10%"
    );
    assert!(
        transition_uncertainty <= 0.025,
        "aggregate transition ambiguity exceeded 2.5%"
    );
    assert!(
        stage_repeat_delta.iter().all(|delta| *delta <= 0.02),
        "post-route stage share changed by more than two points"
    );
    assert!(
        cohort_stage_repeat_delta
            .iter()
            .flatten()
            .all(|delta| *delta <= 0.03),
        "post-route cohort stage share changed by more than three points"
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset"]
fn current_asset_packed_all_iq3_sealed_promotion_gate() {
    const PREFIX_TOKENS: usize = 128;
    const CONTINUATION_TOKEN: u32 = 35;
    const ENDPOINT_COUNT: usize = 3;
    const SAMPLED_ENCODERS: u64 = (DEEPSEEK_V4_LAYER_COUNT * 5) as u64;
    const BLOCK: [Arm; 8] = [
        Arm::Control,
        Arm::Candidate,
        Arm::Candidate,
        Arm::Control,
        Arm::Candidate,
        Arm::Control,
        Arm::Control,
        Arm::Candidate,
    ];

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Arm {
        Control,
        Candidate,
    }

    struct Evidence {
        logits: Vec<f32>,
        hidden: Vec<f32>,
        causal_digest: [u8; 32],
        prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        continuation_logits: Vec<f32>,
        continuation_causal_digest: [u8; 32],
        committed_tokens: Vec<u32>,
        grouped_iq2_invocations: u32,
        grouped_iq3_invocations: u32,
        profile: prefill::PackedPostRouteStageProfile,
        counters: crate::metal::KernelTraceCounters,
        dispatch_count: usize,
        dispatch_digest: [u8; 32],
        wall_ms: f64,
    }

    #[derive(Clone, Copy, Debug)]
    struct Metrics {
        endpoints: [f64; ENDPOINT_COUNT],
        aggregate_coverage_error: f64,
        max_layer_coverage_error: f64,
        aggregate_transition_ambiguity: f64,
        max_layer_transition_ambiguity: f64,
    }

    struct TimedSample {
        arm: Arm,
        half: usize,
        evidence: Evidence,
        metrics: Metrics,
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        prefix: &[u32],
        arm: Arm,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct sealed all-IQ3 session");
        crate::metal::dispatch_census_begin();
        let trace_guard = crate::metal::kernel_trace_begin();
        let started = std::time::Instant::now();
        let profile = session
            .execute_packed_tokens_with_post_route_stage_profile_for_test(
                ctx,
                prefix,
                true,
                true,
                arm == Arm::Candidate,
            )
            .expect("execute sealed all-IQ3 prefix");
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let counters = crate::metal::kernel_trace_snapshot();
        let census = crate::metal::dispatch_census_take();
        drop(trace_guard);
        let mut dispatch_hasher = Sha256::new();
        for row in &census {
            dispatch_hasher.update((row.family.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.family.as_bytes());
            dispatch_hasher.update((row.kernel.len() as u64).to_le_bytes());
            dispatch_hasher.update(row.kernel.as_bytes());
            for extent in [
                row.grid_width,
                row.grid_height,
                row.grid_depth,
                row.threads_width,
                row.threads_height,
                row.threads_depth,
            ] {
                dispatch_hasher.update(extent.to_le_bytes());
            }
        }
        let dispatch_digest = dispatch_hasher.finalize().into();
        let logits = session
            .copy_logits_f32()
            .expect("copy sealed all-IQ3 logits");
        let hidden = host_read_f32(
            session
                .final_normalized_hidden()
                .expect("sealed all-IQ3 final hidden is visible"),
            "sealed all-IQ3 final hidden",
        )
        .expect("copy sealed all-IQ3 hidden");
        let snapshot = session
            .capture_causal_snapshot()
            .expect("capture sealed all-IQ3 state");
        let causal_digest = *snapshot.causal_digest();
        let prefix_digest = *snapshot.prefix_digest();
        let compatibility_digest = *snapshot.compatibility_digest().as_bytes();
        session
            .restore_causal_snapshot(&snapshot)
            .expect("restore sealed all-IQ3 state");
        session
            .forward_token(ctx, CONTINUATION_TOKEN)
            .expect("continue sealed all-IQ3 state");
        let continuation_logits = session
            .copy_logits_f32()
            .expect("copy sealed all-IQ3 continuation logits");
        let continuation = session
            .capture_causal_snapshot()
            .expect("capture sealed all-IQ3 continuation state");
        let evidence = Evidence {
            logits,
            hidden,
            causal_digest,
            prefix_digest,
            compatibility_digest,
            continuation_logits,
            continuation_causal_digest: *continuation.causal_digest(),
            committed_tokens: session.committed_tokens().to_vec(),
            grouped_iq2_invocations: session.packed_grouped_iq2_invocations_for_test(),
            grouped_iq3_invocations: session.packed_grouped_iq3_invocations_for_test(),
            profile,
            counters,
            dispatch_count: census.len(),
            dispatch_digest,
            wall_ms,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    fn assert_exact(label: &str, actual: &Evidence, expected: &Evidence) {
        assert_eq!(
            bits(&actual.logits),
            bits(&expected.logits),
            "{label} logits"
        );
        assert_eq!(
            bits(&actual.hidden),
            bits(&expected.hidden),
            "{label} hidden"
        );
        assert_eq!(
            actual.causal_digest, expected.causal_digest,
            "{label} causal"
        );
        assert_eq!(
            actual.prefix_digest, expected.prefix_digest,
            "{label} prefix"
        );
        assert_eq!(
            actual.compatibility_digest, expected.compatibility_digest,
            "{label} compatibility"
        );
        assert_eq!(
            bits(&actual.continuation_logits),
            bits(&expected.continuation_logits),
            "{label} continuation logits"
        );
        assert_eq!(
            actual.continuation_causal_digest, expected.continuation_causal_digest,
            "{label} continuation causal"
        );
        assert_eq!(
            actual.committed_tokens, expected.committed_tokens,
            "{label} committed tokens"
        );
        assert_eq!(
            actual.profile.metadata, expected.profile.metadata,
            "{label} metadata"
        );
    }

    fn assert_topology(evidence: &Evidence, arm: Arm) {
        assert!(evidence.profile.sampled);
        assert_eq!(
            evidence.profile.command_gpu_ms.len(),
            DEEPSEEK_V4_LAYER_COUNT
        );
        assert_eq!(evidence.profile.metadata.len(), DEEPSEEK_V4_LAYER_COUNT);
        assert_eq!(
            evidence.profile.sampled_layers.len(),
            DEEPSEEK_V4_LAYER_COUNT
        );
        assert_eq!(evidence.counters.encoders, SAMPLED_ENCODERS);
        assert_eq!(evidence.counters.concurrent_encoders, 0);
        assert_eq!(
            evidence.counters.dispatches as usize,
            evidence.dispatch_count
        );
        assert_eq!(evidence.grouped_iq2_invocations, 25);
        assert_eq!(
            evidence.grouped_iq3_invocations,
            if arm == Arm::Candidate { 16 } else { 0 }
        );
    }

    fn metrics(evidence: &Evidence) -> Metrics {
        let mut post_route_gpu_ms = 0.0;
        let mut affected_routed_gpu_ms = 0.0;
        let mut affected_layers = 0usize;
        let mut raw_span_ms = 0.0;
        let mut gap_ms = 0.0;
        let mut overlap_ms = 0.0;
        let mut max_layer_coverage_error = 0.0f64;
        let mut max_layer_transition_ambiguity = 0.0f64;
        for (layer, (sampled_layer, metadata)) in evidence
            .profile
            .sampled_layers
            .iter()
            .zip(&evidence.profile.metadata)
            .enumerate()
        {
            assert_eq!(sampled_layer.layer, layer);
            assert_eq!(metadata.layer, layer);
            assert_eq!(sampled_layer.stages.len(), 4);
            assert_eq!(
                sampled_layer.stages[0].kind,
                prefill::PackedPostRouteStageKind::RoutedExperts
            );
            assert!(
                (sampled_layer.command_gpu_ms - evidence.profile.command_gpu_ms[layer]).abs()
                    < 1e-9
            );
            let stage_ms = sampled_layer
                .stages
                .iter()
                .map(|stage| stage.duration_ms_scaled)
                .sum::<f64>();
            assert!(
                (stage_ms + sampled_layer.encoder_gap_ms_scaled
                    - sampled_layer.encoder_overlap_ms_scaled
                    - sampled_layer.command_gpu_ms)
                    .abs()
                    < 1e-6
            );
            post_route_gpu_ms += sampled_layer.command_gpu_ms;
            raw_span_ms += sampled_layer.raw_span_ms_assuming_ns;
            gap_ms += sampled_layer.encoder_gap_ms_scaled;
            overlap_ms += sampled_layer.encoder_overlap_ms_scaled;
            max_layer_coverage_error =
                max_layer_coverage_error.max((sampled_layer.raw_coverage_assuming_ns - 1.0).abs());
            max_layer_transition_ambiguity = max_layer_transition_ambiguity.max(
                (sampled_layer.encoder_gap_ms_scaled + sampled_layer.encoder_overlap_ms_scaled)
                    / sampled_layer.command_gpu_ms,
            );
            if metadata.gate_dtype == GgmlType::IQ3_XXS
                && metadata.up_dtype == GgmlType::IQ3_XXS
                && metadata.down_dtype == GgmlType::IQ3_XXS
            {
                affected_layers += 1;
                affected_routed_gpu_ms += sampled_layer.stages[0].duration_ms_scaled;
            }
        }
        assert_eq!(affected_layers, 16);
        Metrics {
            endpoints: [evidence.wall_ms, post_route_gpu_ms, affected_routed_gpu_ms],
            aggregate_coverage_error: (raw_span_ms / post_route_gpu_ms - 1.0).abs(),
            max_layer_coverage_error,
            aggregate_transition_ambiguity: (gap_ms + overlap_ms) / post_route_gpu_ms,
            max_layer_transition_ambiguity,
        }
    }

    fn assert_observer_valid(label: &str, metrics: Metrics) {
        assert!(
            metrics.aggregate_coverage_error <= 0.005,
            "{label} aggregate coverage error {} exceeded 0.5%",
            metrics.aggregate_coverage_error
        );
        assert!(
            metrics.max_layer_coverage_error <= 0.02,
            "{label} layer coverage error {} exceeded 2%",
            metrics.max_layer_coverage_error
        );
        assert!(
            metrics.aggregate_transition_ambiguity <= 0.025,
            "{label} aggregate transition ambiguity {} exceeded 2.5%",
            metrics.aggregate_transition_ambiguity
        );
        assert!(
            metrics.max_layer_transition_ambiguity <= 0.10,
            "{label} layer transition ambiguity {} exceeded 10%",
            metrics.max_layer_transition_ambiguity
        );
    }

    fn even_median_four(mut values: Vec<f64>) -> f64 {
        assert_eq!(values.len(), 4);
        values.sort_by(f64::total_cmp);
        (values[1] + values[2]) * 0.5
    }

    fn endpoint_medians(samples: &[TimedSample], arm: Arm, half: usize) -> [f64; 3] {
        std::array::from_fn(|endpoint| {
            even_median_four(
                samples
                    .iter()
                    .filter(|sample| sample.arm == arm && sample.half == half)
                    .map(|sample| sample.metrics.endpoints[endpoint])
                    .collect(),
            )
        })
    }

    fn endpoint_p95(samples: &[TimedSample], arm: Arm) -> [f64; 3] {
        std::array::from_fn(|endpoint| {
            let mut values = samples
                .iter()
                .filter(|sample| sample.arm == arm)
                .map(|sample| sample.metrics.endpoints[endpoint])
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 8);
            values.sort_by(f64::total_cmp);
            values[7]
        })
    }

    fn endpoint_samples(samples: &[TimedSample], arm: Arm, endpoint: usize) -> Vec<f64> {
        samples
            .iter()
            .filter(|sample| sample.arm == arm)
            .map(|sample| sample.metrics.endpoints[endpoint])
            .collect()
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    assert!(
        prefill::packed_grouped_expert_enabled_for_test(&ctx),
        "sealed all-IQ3 gate requires the grouped IQ2 baseline"
    );
    assert!(
        prefill::packed_grouped_iq3_candidate_supported_for_test(&ctx),
        "sealed all-IQ3 candidate is unavailable on this configuration"
    );
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, 129)
        .expect("plan sealed all-IQ3 session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit sealed all-IQ3 session");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let mut residency = realized.into_residency();
    let prefix = (0..PREFIX_TOKENS)
        .map(|index| [35, 201, 200, 34][index % 4])
        .collect::<Vec<_>>();

    let mut warm = Vec::with_capacity(BLOCK.len());
    for arm in BLOCK {
        let (next, evidence) = execute(&ctx, residency, model_content_id, &prefix, arm);
        residency = next;
        warm.push((arm, evidence));
    }
    let reference = &warm[0].1;
    assert_eq!(warm[0].0, Arm::Control);
    for (arm, evidence) in &warm {
        assert_exact("sealed warm-up", evidence, reference);
        assert_topology(evidence, *arm);
    }

    let mut timed = Vec::with_capacity(BLOCK.len() * 2);
    for half in 0..2 {
        for arm in BLOCK {
            let (next, evidence) = execute(&ctx, residency, model_content_id, &prefix, arm);
            residency = next;
            let sample_metrics = metrics(&evidence);
            timed.push(TimedSample {
                arm,
                half,
                evidence,
                metrics: sample_metrics,
            });
        }
    }
    drop(residency);
    assert_eq!(timed.len(), 16);

    let mut control_digest = None;
    let mut candidate_digest = None;
    let mut control_dispatches = None;
    let mut candidate_dispatches = None;
    for sample in &timed {
        assert_exact("sealed timed arm", &sample.evidence, reference);
        assert_topology(&sample.evidence, sample.arm);
        assert_observer_valid("sealed timed arm", sample.metrics);
        let (digest, dispatches) = match sample.arm {
            Arm::Control => (&mut control_digest, &mut control_dispatches),
            Arm::Candidate => (&mut candidate_digest, &mut candidate_dispatches),
        };
        if let Some(expected) = digest {
            assert_eq!(*expected, sample.evidence.dispatch_digest);
        } else {
            *digest = Some(sample.evidence.dispatch_digest);
        }
        if let Some(expected) = dispatches {
            assert_eq!(*expected, sample.evidence.dispatch_count);
        } else {
            *dispatches = Some(sample.evidence.dispatch_count);
        }
    }

    let control_first = endpoint_medians(&timed, Arm::Control, 0);
    let control_second = endpoint_medians(&timed, Arm::Control, 1);
    let candidate_first = endpoint_medians(&timed, Arm::Candidate, 0);
    let candidate_second = endpoint_medians(&timed, Arm::Candidate, 1);
    let control_stationarity: [f64; ENDPOINT_COUNT] = std::array::from_fn(|endpoint| {
        2.0 * (control_first[endpoint] - control_second[endpoint]).abs()
            / (control_first[endpoint] + control_second[endpoint])
    });
    let candidate_stationarity: [f64; ENDPOINT_COUNT] = std::array::from_fn(|endpoint| {
        2.0 * (candidate_first[endpoint] - candidate_second[endpoint]).abs()
            / (candidate_first[endpoint] + candidate_second[endpoint])
    });
    let first_saving: [f64; ENDPOINT_COUNT] =
        std::array::from_fn(|endpoint| 1.0 - candidate_first[endpoint] / control_first[endpoint]);
    let second_saving: [f64; ENDPOINT_COUNT] =
        std::array::from_fn(|endpoint| 1.0 - candidate_second[endpoint] / control_second[endpoint]);
    let control_p95 = endpoint_p95(&timed, Arm::Control);
    let candidate_p95 = endpoint_p95(&timed, Arm::Candidate);
    let control_wall_ms = endpoint_samples(&timed, Arm::Control, 0);
    let candidate_wall_ms = endpoint_samples(&timed, Arm::Candidate, 0);
    let control_post_route_gpu_ms = endpoint_samples(&timed, Arm::Control, 1);
    let candidate_post_route_gpu_ms = endpoint_samples(&timed, Arm::Candidate, 1);
    let control_affected_gpu_ms = endpoint_samples(&timed, Arm::Control, 2);
    let candidate_affected_gpu_ms = endpoint_samples(&timed, Arm::Candidate, 2);

    eprintln!(
        "deepseek_v4 packed_all_iq3_sealed_gate n={PREFIX_TOKENS} schedule=ABBA_BAAB_x2 control_wall_ms={control_wall_ms:?} candidate_wall_ms={candidate_wall_ms:?} control_post_route_gpu_ms={control_post_route_gpu_ms:?} candidate_post_route_gpu_ms={candidate_post_route_gpu_ms:?} control_affected_gpu_ms={control_affected_gpu_ms:?} candidate_affected_gpu_ms={candidate_affected_gpu_ms:?} control_first_median={control_first:?} control_second_median={control_second:?} candidate_first_median={candidate_first:?} candidate_second_median={candidate_second:?} control_stationarity={control_stationarity:?} candidate_stationarity={candidate_stationarity:?} first_saving={first_saving:?} second_saving={second_saving:?} control_p95={control_p95:?} candidate_p95={candidate_p95:?} control_dispatches={} candidate_dispatches={} control_dispatch_sha256={} candidate_dispatch_sha256={} model_content_id={} logits_sha256={} hidden_sha256={} causal_digest={} prefix_digest={} compatibility_digest={} continuation_logits_sha256={} continuation_causal_digest={} committed_tokens_sha256={}",
        control_dispatches.unwrap(),
        candidate_dispatches.unwrap(),
        hex(control_digest.unwrap()),
        hex(candidate_digest.unwrap()),
        hex(model_content_id.as_bytes()),
        hex(Sha256::digest(bytemuck::cast_slice(&reference.logits))),
        hex(Sha256::digest(bytemuck::cast_slice(&reference.hidden))),
        hex(reference.causal_digest),
        hex(reference.prefix_digest),
        hex(reference.compatibility_digest),
        hex(Sha256::digest(bytemuck::cast_slice(
            &reference.continuation_logits,
        ))),
        hex(reference.continuation_causal_digest),
        hex(Sha256::digest(bytemuck::cast_slice(
            &reference.committed_tokens,
        ))),
    );

    const SAVING_GATES: [f64; ENDPOINT_COUNT] = [0.05, 0.10, 0.30];
    for endpoint in 0..ENDPOINT_COUNT {
        assert!(
            control_stationarity[endpoint] <= 0.05,
            "control endpoint {endpoint} stationarity {} exceeded 5%",
            control_stationarity[endpoint]
        );
        assert!(
            candidate_stationarity[endpoint] <= 0.05,
            "candidate endpoint {endpoint} stationarity {} exceeded 5%",
            candidate_stationarity[endpoint]
        );
        assert!(
            first_saving[endpoint] >= SAVING_GATES[endpoint]
                && second_saving[endpoint] >= SAVING_GATES[endpoint],
            "endpoint {endpoint} half savings {}/{} missed {}",
            first_saving[endpoint],
            second_saving[endpoint],
            SAVING_GATES[endpoint]
        );
        assert!(
            candidate_p95[endpoint] <= control_p95[endpoint],
            "candidate endpoint {endpoint} p95 {} exceeded control {}",
            candidate_p95[endpoint],
            control_p95[endpoint]
        );
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn fp4_score_plans_are_exhaustive_and_dispatch_ledgers_fail_closed() {
    let cases = [
        (
            DeepSeekV4Fp4SessionMode::F16Authoritative,
            false,
            DeepSeekV4Fp4ScorePlanKind::F16Only,
            DeepSeekV4Fp4SelectionSource::F16,
        ),
        (
            DeepSeekV4Fp4SessionMode::F16Authoritative,
            true,
            DeepSeekV4Fp4ScorePlanKind::Paired,
            DeepSeekV4Fp4SelectionSource::F16,
        ),
        (
            DeepSeekV4Fp4SessionMode::PairedCounterfactual,
            false,
            DeepSeekV4Fp4ScorePlanKind::Paired,
            DeepSeekV4Fp4SelectionSource::Fp4,
        ),
        (
            DeepSeekV4Fp4SessionMode::PairedCounterfactual,
            true,
            DeepSeekV4Fp4ScorePlanKind::Paired,
            DeepSeekV4Fp4SelectionSource::Fp4,
        ),
        (
            DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental,
            false,
            DeepSeekV4Fp4ScorePlanKind::Fp4Only,
            DeepSeekV4Fp4SelectionSource::Fp4,
        ),
        (
            DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental,
            true,
            DeepSeekV4Fp4ScorePlanKind::Paired,
            DeepSeekV4Fp4SelectionSource::Fp4,
        ),
    ];
    for (mode, audit_active, expected_kind, expected_source) in cases {
        let plan = mode.score_plan(audit_active);
        assert_eq!(plan.kind(), expected_kind);
        assert_eq!(plan.consumed_source(), expected_source);
        let mut ledger = DeepSeekV4Fp4ScoreDispatchLedger::new(
            DeepSeekV4Fp4ShadowExecution::Singleton,
            2_051,
            plan.kind(),
            plan.consumed_source(),
        );
        for _ in 0..diagnostics::CSA_LAYER_COUNT {
            ledger.record_common_prepare().unwrap();
            if plan.runs_f16() {
                ledger.record_f16_score_and_selector().unwrap();
            }
            if plan.runs_fp4() {
                ledger.record_fp4_pipeline().unwrap();
            }
        }
        ledger.validate_completed().unwrap();
        assert_eq!(ledger.sparse_layer_count, 21);
        assert_eq!(
            ledger.f16_score_selector_pipeline_invocations == 0,
            !plan.runs_f16()
        );
        assert_eq!(ledger.fp4_pipeline_invocations == 0, !plan.runs_fp4());
    }

    let mut invalid = DeepSeekV4Fp4ScoreDispatchLedger::new(
        DeepSeekV4Fp4ShadowExecution::Packed,
        2_051,
        DeepSeekV4Fp4ScorePlanKind::Fp4Only,
        DeepSeekV4Fp4SelectionSource::Fp4,
    );
    invalid.record_common_prepare().unwrap();
    invalid.record_fp4_pipeline().unwrap();
    invalid.f16_score_selector_pipeline_invocations = 1;
    assert!(invalid.validate_completed().is_err());
}

#[test]
fn session_memory_inventory_is_complete_and_unique() {
    let mut kinds = vec![AttentionKind::SlidingWindow; 2];
    kinds.extend(std::iter::repeat_n(AttentionKind::CompressedSparse, 21));
    kinds.extend(std::iter::repeat_n(AttentionKind::HeavilyCompressed, 20));
    let config = crate::deepseek_v4::flash_0731_config_fixture();
    let capacity =
        DeepSeekV4SessionCapacity::for_forward_limit(3_073, config.context_length).unwrap();
    let requests =
        deepseek_v4_session_allocation_requests_for_kinds(&kinds, 256, capacity).unwrap();
    let csa_layer_count = kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::CompressedSparse)
        .count();
    let diagnostics_allocations = if cfg!(feature = "dsv4-diagnostics") {
        csa_layer_count * 3 + 13
    } else {
        0
    };
    let diagnostics_logical = if cfg!(feature = "dsv4-diagnostics") {
        csa_layer_count as u64 * capacity.csa_physical_rows() as u64 * 72
            + 112_160
            + capacity.csa_physical_rows() as u64 * 8
    } else {
        0
    };
    let packed_route_allocations = if cfg!(feature = "dsv4-diagnostics") {
        8
    } else {
        4
    };
    let packed_route_logical = if cfg!(feature = "dsv4-diagnostics") {
        2_115_648
    } else {
        17_440
    };
    assert_eq!(
        requests.len(),
        551 + diagnostics_allocations + packed_route_allocations
    );
    assert_eq!(
        requests
            .iter()
            .map(|request| request.logical_bytes)
            .sum::<u64>(),
        4_353_932_228 + diagnostics_logical + packed_route_logical
    );
    let names = requests
        .iter()
        .map(|request| request.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), requests.len());
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.name.ends_with(".published"))
            .map(|request| request.logical_bytes)
            .sum::<u64>(),
        31_129_600
    );
    assert_eq!(
        requests
            .iter()
            .find(|request| request.name == "raw_cache")
            .unwrap()
            .logical_bytes,
        5_636_096
    );
    let packed_gpu_route = requests
        .iter()
        .filter(|request| request.name.starts_with("prefill.moe.gpu_route."))
        .collect::<Vec<_>>();
    assert_eq!(packed_gpu_route.len(), packed_route_allocations);
    assert_eq!(
        packed_gpu_route
            .iter()
            .map(|request| request.logical_bytes)
            .sum::<u64>(),
        packed_route_logical
    );
    assert_eq!(
        requests
            .iter()
            .find(|request| request.name == "prefill.moe.grouped_inner")
            .map(|request| request.logical_bytes),
        Some(201_326_592)
    );
    assert_eq!(
        requests
            .iter()
            .find(|request| request.name == "prefill.moe.grouped_tiles")
            .map(|request| request.logical_bytes),
        Some(12_192)
    );
    assert_eq!(
        requests
            .iter()
            .find(|request| request.name == "prefill.moe.grouped_iq2_mma16_tiles")
            .map(|request| request.logical_bytes),
        Some(21_312)
    );

    let promoted_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
        config.context_length,
    )
    .unwrap();
    let promoted =
        deepseek_v4_session_allocation_requests_for_kinds(&kinds, 256, promoted_capacity).unwrap();
    assert_eq!(promoted.len(), requests.len() + 5);
    for name in [
        "prefill.attention.sparse_csa.scores",
        "prefill.attention.sparse_csa.selected_mask",
    ] {
        assert_eq!(
            promoted
                .iter()
                .find(|request| request.name == name)
                .map(|request| request.logical_bytes),
            Some(4_294_967_296)
        );
    }
    let promoted_diagnostics_logical = if cfg!(feature = "dsv4-diagnostics") {
        csa_layer_count as u64 * promoted_capacity.csa_physical_rows() as u64 * 72
            + 112_160
            + promoted_capacity.csa_physical_rows() as u64 * 8
    } else {
        0
    };
    assert_eq!(
        promoted
            .iter()
            .map(|request| request.logical_bytes)
            .sum::<u64>(),
        20_104_132_716 + promoted_diagnostics_logical + packed_route_logical
    );
    assert_eq!(
        promoted
            .iter()
            .filter(|request| request.name.ends_with(".published"))
            .map(|request| request.logical_bytes)
            .sum::<u64>(),
        7_214_202_880
    );
    let promoted_names = promoted
        .iter()
        .map(|request| request.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(names.is_subset(&promoted_names));
    assert_eq!(
        promoted_names
            .difference(&names)
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [
            "sparse_csa.multigroup.partition_plan",
            "sparse_csa.multigroup.private_ids",
            "sparse_csa.multigroup.private_mask",
            "sparse_csa.multigroup.records",
            "sparse_csa.multigroup.state",
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn session_memory_inventory_tracks_expert_geometry() {
    let mut kinds = vec![AttentionKind::SlidingWindow; 2];
    kinds.extend(std::iter::repeat_n(AttentionKind::CompressedSparse, 21));
    kinds.extend(std::iter::repeat_n(AttentionKind::HeavilyCompressed, 20));
    let config = crate::deepseek_v4::flash_0731_config_fixture();
    let capacity =
        DeepSeekV4SessionCapacity::for_forward_limit(2_049, config.context_length).unwrap();
    let e256 = deepseek_v4_session_allocation_requests_for_kinds(&kinds, 256, capacity).unwrap();
    let e160 = deepseek_v4_session_allocation_requests_for_kinds(&kinds, 160, capacity).unwrap();
    let differences = e256
        .iter()
        .zip(&e160)
        .filter(|(left, right)| left != right)
        .collect::<Vec<_>>();
    assert_eq!(differences.len(), 1);
    assert_eq!(differences[0].0.name, "moe.logits");
    assert_eq!(differences[0].0.logical_bytes, 256 * 4);
    assert_eq!(differences[0].1.logical_bytes, 160 * 4);
}

#[test]
fn shared_buffer_pricing_includes_host_page_granularity() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let page = host_page_size_bytes().unwrap() as u64;
    let (priced, alignment) = price_shared_buffer(&ctx, 1, "one-byte probe").unwrap();
    assert!(alignment >= page);
    assert_eq!(priced, alignment);
    let (priced, alignment) = price_shared_buffer(&ctx, page + 1, "cross-page probe").unwrap();
    assert!(alignment >= page);
    let expected = (page + alignment) / alignment * alignment;
    assert_eq!(priced, expected);
    let device_max = ctx.max_buffer_length() as u64;
    let error = price_shared_buffer(&ctx, device_max + 1, "oversized probe").unwrap_err();
    assert!(error.to_string().contains("beyond device maximum"));
}

#[test]
#[ignore = "requires the archived 95.93 GiB DS4 fixture and executes a synthetic deep-context token"]
fn native_synthetic_state_crosses_the_first_tiled_hca_boundary_repeatably() {
    const POSITION: u32 = 65_663;
    const FORWARD_LIMIT: usize = POSITION as usize + 1;
    let model_path = std::env::var_os("DSV4_LEGACY_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731-old/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = open_pinned_legacy_gguf(&model_path);
    let run = || {
        let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
            .expect("plan first tiled-HCA boundary session");
        assert_eq!(plan.session_capacity().hca_physical_rows(), 768);
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit first tiled-HCA boundary session");
        let residency = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
            .expect("realize first tiled-HCA boundary session")
            .into_residency();
        let mut session = DeepSeekV4Session::new(&ctx, residency)
            .expect("construct first tiled-HCA boundary session");
        initialize_zero_synthetic_causal_state(&session);
        session.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
            next_position: POSITION,
        };
        session
            .committed_tokens
            .extend(std::iter::repeat_n(35, POSITION as usize));
        session
            .forward_token(&ctx, 35)
            .expect("execute synthetic first tiled-HCA boundary");
        assert_eq!(session.next_position(), POSITION + 1);
        let logits = session
            .copy_logits_f32()
            .expect("copy synthetic tiled-HCA boundary logits");
        assert!(logits.iter().all(|value| value.is_finite()));
        let error = session
            .forward_token(&ctx, 35)
            .err()
            .expect("bounded tiled-HCA session must reject its next position");
        assert!(error.to_string().contains("next position is 65664"));
        assert_eq!(
            session
                .copy_logits_f32()
                .expect("rejection preserves synthetic boundary logits"),
            logits
        );
        logits
    };
    let first = run();
    let second = run();
    assert!(
        first
            .iter()
            .zip(second)
            .all(|(first, second)| first.to_bits() == second.to_bits())
    );
    let mut hasher = Sha256::new();
    for value in &first {
        hasher.update(value.to_le_bytes());
    }
    let hash = format!("{:x}", hasher.finalize());
    eprintln!("deepseek_v4 synthetic_position_65663_sha256={hash}");
    assert_eq!(
        hash,
        "92895a69787cc972880626ef391517236e4ee6d984c52113645d1d0e42824938"
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current DS4 asset and a full-context session allocation"]
fn splitk_hca_profiles_current_synthetic_terminal_tokens() {
    const START_POSITION: u32 = DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY as u32 - 2;
    const TERMINAL_POSITION: u32 = START_POSITION + 1;
    const FORWARD_LIMIT: usize = DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY;
    const TOKEN_ID: u32 = 35;

    struct Evidence {
        preterminal: DeepSeekV4WholeTokenProfile,
        terminal: DeepSeekV4WholeTokenProfile,
        logits: Vec<f32>,
        logits_sha256: String,
    }

    fn vector_sha256(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0
    }

    fn logit_metrics(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut squared_error = 0.0f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            assert!(actual.is_finite());
            dot += f64::from(actual) * f64::from(reference);
            actual_norm += f64::from(actual).powi(2);
            reference_norm += f64::from(reference).powi(2);
            squared_error += f64::from(actual - reference).powi(2);
        }
        (
            dot / (actual_norm.sqrt() * reference_norm.sqrt()),
            (squared_error / reference_norm).sqrt(),
        )
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        score_policy: DeepSeekV4IndexerScoreTestPolicy,
        policy: DeepSeekV4HcaTestPolicy,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new(ctx, residency).expect("construct terminal synthetic session");
        session.sparse_csa.set_score_test_policy(score_policy);
        session.attention.set_hca_test_policy(policy);
        initialize_zero_synthetic_causal_state(&session);
        session.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
            next_position: START_POSITION,
        };
        session
            .committed_tokens
            .extend(std::iter::repeat_n(TOKEN_ID, START_POSITION as usize));
        let preterminal = session
            .forward_token_whole_profiled(ctx, TOKEN_ID)
            .expect("profile preterminal synthetic token");
        assert_eq!(preterminal.position, START_POSITION);
        let terminal = session
            .forward_token_whole_profiled(ctx, TOKEN_ID)
            .expect("profile terminal synthetic token");
        assert_eq!(terminal.position, TERMINAL_POSITION);
        assert_eq!(session.next_position(), TERMINAL_POSITION + 1);
        let logits = session
            .copy_logits_f32()
            .expect("copy terminal synthetic logits");
        assert!(logits.iter().all(|value| value.is_finite()));
        let evidence = Evidence {
            logits_sha256: vector_sha256(&logits),
            logits,
            preterminal,
            terminal,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, _) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
        .expect("plan terminal synthetic session");
    assert_eq!(plan.session_capacity().csa_physical_rows(), 262_144);
    assert_eq!(plan.session_capacity().hca_physical_rows(), 8_192);
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit terminal synthetic session");
    let residency = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize terminal synthetic residency")
        .into_residency();
    let mut residency = residency;
    let mut splitk_hash = None;
    for packet in 0..2 {
        let grouped_before;
        (residency, grouped_before) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::Production,
            DeepSeekV4HcaTestPolicy::GroupedOnline,
        );
        let splitk;
        (residency, splitk) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::Production,
            DeepSeekV4HcaTestPolicy::Production,
        );
        let grouped_after;
        (residency, grouped_after) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::Production,
            DeepSeekV4HcaTestPolicy::GroupedOnline,
        );
        assert_eq!(grouped_before.logits_sha256, grouped_after.logits_sha256);
        match &splitk_hash {
            Some(expected) => assert_eq!(&splitk.logits_sha256, expected),
            None => splitk_hash = Some(splitk.logits_sha256.clone()),
        }
        let (cosine, relative_rms) = logit_metrics(&splitk.logits, &grouped_before.logits);
        assert_eq!(argmax(&splitk.logits), argmax(&grouped_before.logits));
        assert!(cosine >= 0.999_99, "terminal cosine {cosine}");
        assert!(
            relative_rms <= 0.005,
            "terminal relative RMS {relative_rms}"
        );

        let grouped_gpu_midpoint =
            (grouped_before.terminal.command_gpu_ms + grouped_after.terminal.command_gpu_ms) * 0.5;
        let grouped_wall_midpoint = (grouped_before.terminal.forward_wall_ms
            + grouped_after.terminal.forward_wall_ms)
            * 0.5;
        let grouped_outside_midpoint = (grouped_before.terminal.outside_gpu_ms()
            + grouped_after.terminal.outside_gpu_ms())
            * 0.5;
        let gpu_savings = grouped_gpu_midpoint - splitk.terminal.command_gpu_ms;
        let wall_savings = grouped_wall_midpoint - splitk.terminal.forward_wall_ms;
        assert!(
            gpu_savings >= 35.0,
            "packet {packet} GPU savings {gpu_savings:.3} ms"
        );
        assert!(
            wall_savings >= 30.0,
            "packet {packet} wall savings {wall_savings:.3} ms"
        );
        assert!(
            splitk.terminal.outside_gpu_ms() <= grouped_outside_midpoint + 1.0,
            "packet {packet} outside-GPU regression"
        );
        eprintln!(
            "deepseek_v4 splitk_hca_terminal packet={packet} grouped_before_preterminal_gpu_ms={:.3} grouped_before_preterminal_wall_ms={:.3} splitk_preterminal_gpu_ms={:.3} splitk_preterminal_wall_ms={:.3} grouped_after_preterminal_gpu_ms={:.3} grouped_after_preterminal_wall_ms={:.3} grouped_before_terminal_gpu_ms={:.3} splitk_terminal_gpu_ms={:.3} grouped_after_terminal_gpu_ms={:.3} gpu_savings_ms={gpu_savings:.3} grouped_before_terminal_wall_ms={:.3} splitk_terminal_wall_ms={:.3} grouped_after_terminal_wall_ms={:.3} wall_savings_ms={wall_savings:.3} grouped_outside_midpoint_ms={grouped_outside_midpoint:.3} splitk_outside_gpu_ms={:.3} argmax={} cosine={cosine:.9} rel_rms={relative_rms:.9} grouped_logits_sha256={} splitk_logits_sha256={}",
            grouped_before.preterminal.command_gpu_ms,
            grouped_before.preterminal.forward_wall_ms,
            splitk.preterminal.command_gpu_ms,
            splitk.preterminal.forward_wall_ms,
            grouped_after.preterminal.command_gpu_ms,
            grouped_after.preterminal.forward_wall_ms,
            grouped_before.terminal.command_gpu_ms,
            splitk.terminal.command_gpu_ms,
            grouped_after.terminal.command_gpu_ms,
            grouped_before.terminal.forward_wall_ms,
            splitk.terminal.forward_wall_ms,
            grouped_after.terminal.forward_wall_ms,
            splitk.terminal.outside_gpu_ms(),
            argmax(&splitk.logits),
            grouped_before.logits_sha256,
            splitk.logits_sha256,
        );
    }

    let mut matrix_hash = None;
    for packet in 0..2 {
        let control_before;
        (residency, control_before) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::Production,
            DeepSeekV4HcaTestPolicy::Production,
        );
        let matrix;
        (residency, matrix) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::MatrixF16,
            DeepSeekV4HcaTestPolicy::Production,
        );
        let control_after;
        (residency, control_after) = execute(
            &ctx,
            residency,
            DeepSeekV4IndexerScoreTestPolicy::Production,
            DeepSeekV4HcaTestPolicy::Production,
        );
        assert_eq!(control_before.logits_sha256, control_after.logits_sha256);
        match &matrix_hash {
            Some(expected) => assert_eq!(&matrix.logits_sha256, expected),
            None => matrix_hash = Some(matrix.logits_sha256.clone()),
        }
        let (cosine, relative_rms) = logit_metrics(&matrix.logits, &control_before.logits);
        assert_eq!(argmax(&matrix.logits), argmax(&control_before.logits));
        assert!(cosine >= 0.999_99, "terminal score cosine {cosine}");
        assert!(
            relative_rms <= 0.005,
            "terminal score relative RMS {relative_rms}"
        );
        let control_gpu_midpoint =
            (control_before.terminal.command_gpu_ms + control_after.terminal.command_gpu_ms) * 0.5;
        let control_wall_midpoint = (control_before.terminal.forward_wall_ms
            + control_after.terminal.forward_wall_ms)
            * 0.5;
        let gpu_savings = control_gpu_midpoint - matrix.terminal.command_gpu_ms;
        let wall_savings = control_wall_midpoint - matrix.terminal.forward_wall_ms;
        assert!(
            gpu_savings >= 20.0,
            "packet {packet} F16 matrix scorer GPU savings {gpu_savings:.3} ms"
        );
        assert!(
            wall_savings >= 18.0,
            "packet {packet} F16 matrix scorer wall savings {wall_savings:.3} ms"
        );
        eprintln!(
            "deepseek_v4 lightning_f16_matrix_terminal packet={packet} control_before_preterminal_gpu_ms={:.3} matrix_preterminal_gpu_ms={:.3} control_after_preterminal_gpu_ms={:.3} control_before_terminal_gpu_ms={:.3} matrix_terminal_gpu_ms={:.3} control_after_terminal_gpu_ms={:.3} gpu_savings_ms={gpu_savings:.3} control_before_terminal_wall_ms={:.3} matrix_terminal_wall_ms={:.3} control_after_terminal_wall_ms={:.3} wall_savings_ms={wall_savings:.3} argmax={} cosine={cosine:.9} rel_rms={relative_rms:.9} control_logits_sha256={} matrix_logits_sha256={}",
            control_before.preterminal.command_gpu_ms,
            matrix.preterminal.command_gpu_ms,
            control_after.preterminal.command_gpu_ms,
            control_before.terminal.command_gpu_ms,
            matrix.terminal.command_gpu_ms,
            control_after.terminal.command_gpu_ms,
            control_before.terminal.forward_wall_ms,
            matrix.terminal.forward_wall_ms,
            control_after.terminal.forward_wall_ms,
            argmax(&matrix.logits),
            control_before.logits_sha256,
            matrix.logits_sha256,
        );
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the local 95.93 GiB DS4 fixture and captures terminal decisions"]
fn online_hca_preserves_terminal_decisions_repeatably() {
    const START_POSITION: u32 = DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY as u32 - 2;
    const TERMINAL_POSITION: u32 = START_POSITION + 1;
    const TOKEN_ID: u32 = 35;
    const LEGACY_LOGITS_SHA256: &str =
        "4c54019668cb815036bd823ddee2c4156f481a48138b35896899d758184587be";
    const ONLINE_LOGITS_SHA256: &str =
        "c6a6075667623b0127d3b18383c5f4e11b136502ab53163d0d8b9edde3cb244c";
    const LEGACY_CAUSAL_SHA256: &str =
        "359feb464d986d37f0556b88287582346075568119510fb4b97f000e8183122b";
    const ONLINE_CAUSAL_SHA256: &str =
        "adb621cb44f5ad957dc60568db9a346344d28995430518e0f9d641c7d7982317";

    struct Evidence {
        logits: Vec<f32>,
        logits_sha256: String,
        causal_digest: [u8; 32],
        transcript: DeepSeekV4DecisionTranscript,
    }

    fn vector_sha256(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").unwrap();
        }
        encoded
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0
    }

    fn logit_metrics(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut squared_error = 0.0f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            assert!(actual.is_finite());
            dot += f64::from(actual) * f64::from(reference);
            actual_norm += f64::from(actual).powi(2);
            reference_norm += f64::from(reference).powi(2);
            squared_error += f64::from(actual - reference).powi(2);
        }
        (
            dot / (actual_norm.sqrt() * reference_norm.sqrt()),
            (squared_error / reference_norm).sqrt(),
        )
    }

    fn decision_deltas(
        actual: &DeepSeekV4DecisionTranscript,
        reference: &DeepSeekV4DecisionTranscript,
    ) -> (f32, f32, f32) {
        assert_eq!(actual.position, reference.position);
        assert_eq!(actual.layers.len(), reference.layers.len());
        let mut max_score_delta = 0.0f32;
        let mut max_route_weight_delta = 0.0f32;
        let mut max_cutoff_margin_delta = 0.0f32;
        for (actual, reference) in actual.layers.iter().zip(&reference.layers) {
            assert_eq!(actual.layer, reference.layer);
            assert_eq!(actual.route.expert_ids, reference.route.expert_ids);
            for (&actual, &reference) in actual
                .route
                .normalized_scaled_weights
                .iter()
                .zip(&reference.route.normalized_scaled_weights)
            {
                max_route_weight_delta = max_route_weight_delta.max((actual - reference).abs());
            }
            match (&actual.csa, &reference.csa) {
                (Some(actual), Some(reference)) => {
                    assert_eq!(
                        actual.cache_order_selected_ids,
                        reference.cache_order_selected_ids
                    );
                    assert_eq!(actual.selected_count, reference.selected_count);
                    assert_eq!(actual.selection_status, reference.selection_status);
                    for (&actual, &reference) in
                        actual.visible_scores.iter().zip(&reference.visible_scores)
                    {
                        max_score_delta = max_score_delta.max((actual - reference).abs());
                    }
                    max_cutoff_margin_delta = max_cutoff_margin_delta
                        .max((actual.rank_512_margin - reference.rank_512_margin).abs());
                }
                (None, None) => {}
                _ => panic!("layer {} CSA decision shape changed", actual.layer),
            }
        }
        (
            max_score_delta,
            max_route_weight_delta,
            max_cutoff_margin_delta,
        )
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        policy: DeepSeekV4HcaTestPolicy,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session = DeepSeekV4Session::new_with_model_content_id(
            ctx,
            residency,
            DeepSeekV4ModelContentId::new([0x10; 32]),
        )
        .expect("construct terminal decision session");
        session.attention.set_hca_test_policy(policy);
        initialize_zero_synthetic_causal_state(&session);
        session.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
            next_position: START_POSITION,
        };
        session
            .committed_tokens
            .extend(std::iter::repeat_n(TOKEN_ID, START_POSITION as usize));
        session
            .forward_token(ctx, TOKEN_ID)
            .expect("execute preterminal decision warmup");
        session
            .arm_decision_transcript(TERMINAL_POSITION)
            .expect("arm terminal decision transcript");
        session
            .forward_token(ctx, TOKEN_ID)
            .expect("execute terminal decision token");
        let transcript = session
            .take_decision_transcript()
            .expect("take terminal decision transcript");
        let logits = session
            .copy_logits_f32()
            .expect("copy terminal decision logits");
        let causal_digest = *session
            .capture_causal_snapshot()
            .expect("capture terminal decision state")
            .causal_digest();
        let evidence = Evidence {
            logits_sha256: vector_sha256(&logits),
            logits,
            causal_digest,
            transcript,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_LEGACY_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731-old/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = open_pinned_legacy_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(
        &ctx,
        &gguf,
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    )
    .expect("plan terminal decision session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit terminal decision session");
    let mut residency = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize terminal decision residency")
        .into_residency();

    let legacy;
    (residency, legacy) = execute(&ctx, residency, DeepSeekV4HcaTestPolicy::LegacyTiled);
    let online_first;
    (residency, online_first) = execute(&ctx, residency, DeepSeekV4HcaTestPolicy::GroupedOnline);
    let online_second;
    (_, online_second) = execute(&ctx, residency, DeepSeekV4HcaTestPolicy::GroupedOnline);

    assert_eq!(legacy.logits_sha256, LEGACY_LOGITS_SHA256);
    assert_eq!(online_first.logits_sha256, ONLINE_LOGITS_SHA256);
    assert_eq!(online_second.logits_sha256, ONLINE_LOGITS_SHA256);
    assert_eq!(digest_hex(&legacy.causal_digest), LEGACY_CAUSAL_SHA256);
    assert_eq!(
        digest_hex(&online_first.causal_digest),
        ONLINE_CAUSAL_SHA256
    );
    assert_eq!(online_first.causal_digest, online_second.causal_digest);
    assert_eq!(online_first.transcript, online_second.transcript);
    let (cosine, relative_rms) = logit_metrics(&online_first.logits, &legacy.logits);
    assert_eq!(argmax(&online_first.logits), argmax(&legacy.logits));
    assert!(cosine >= 0.999_99, "terminal cosine {cosine}");
    assert!(
        relative_rms <= 0.005,
        "terminal relative RMS {relative_rms}"
    );
    let (max_score_delta, max_route_weight_delta, max_cutoff_margin_delta) =
        decision_deltas(&online_first.transcript, &legacy.transcript);

    eprintln!(
        "deepseek_v4 online_hca_terminal_decisions position={TERMINAL_POSITION} argmax={} cosine={cosine:.9} rel_rms={relative_rms:.9} max_csa_score_delta={max_score_delta:.9} max_route_weight_delta={max_route_weight_delta:.9} max_cutoff_margin_delta={max_cutoff_margin_delta:.9} legacy_logits_sha256={} online_logits_sha256={} legacy_causal_sha256={} online_causal_sha256={}",
        argmax(&online_first.logits),
        legacy.logits_sha256,
        online_first.logits_sha256,
        digest_hex(&legacy.causal_digest),
        digest_hex(&online_first.causal_digest),
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the local 95.93 GiB DS4 fixture and executes focused far-context differentials"]
fn sparse_csa_optimizations_preserve_far_context_token_and_causal_state() {
    const POSITION: u32 = 65_663;
    const FORWARD_LIMIT: usize = POSITION as usize + 1;
    const TOKEN_ID: u32 = 35;
    const PINNED_LOGITS_SHA256: &str =
        "1c0f5e0475314e693bfe0664b5454a2ece26d9a5913f9a5218cdf39da59582d4";
    const PINNED_CAUSAL_SHA256: &str =
        "03f15887db83e4baf0ad5ba66f95b3e2a7fe461858d92aebe9f0b3009f83254e";

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RunMode {
        Timed,
        Transcript,
    }

    struct Evidence {
        logits_sha256: String,
        command_gpu_ms: Option<f64>,
        wall_ms: Option<f64>,
        causal_digest: Option<[u8; 32]>,
        transcript: Option<DeepSeekV4DecisionTranscript>,
    }

    fn logits_sha256(logits: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in logits {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").unwrap();
        }
        encoded
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        score_policy: DeepSeekV4IndexerScoreTestPolicy,
        selector_policy: DeepSeekV4SelectorTestPolicy,
        hca_policy: DeepSeekV4HcaTestPolicy,
        mode: RunMode,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session = DeepSeekV4Session::new_with_model_content_id(
            ctx,
            residency,
            DeepSeekV4ModelContentId::new([0x65; 32]),
        )
        .expect("construct synthetic far-context session");
        session.sparse_csa.set_score_test_policy(score_policy);
        session.sparse_csa.set_selector_test_policy(selector_policy);
        session.attention.set_hca_test_policy(hca_policy);
        initialize_zero_synthetic_causal_state(&session);
        session.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
            next_position: POSITION,
        };
        session
            .committed_tokens
            .extend(std::iter::repeat_n(TOKEN_ID, POSITION as usize));

        let (command_gpu_ms, wall_ms, transcript) = match mode {
            RunMode::Timed => {
                let profile = session
                    .forward_token_whole_profiled(ctx, TOKEN_ID)
                    .expect("profile synthetic far-context token");
                (
                    Some(profile.command_gpu_ms),
                    Some(profile.forward_wall_ms),
                    None,
                )
            }
            RunMode::Transcript => {
                session
                    .arm_decision_transcript(POSITION)
                    .expect("arm far-context decision transcript");
                session
                    .forward_token(ctx, TOKEN_ID)
                    .expect("execute synthetic far-context transcript token");
                (
                    None,
                    None,
                    Some(
                        session
                            .take_decision_transcript()
                            .expect("take far-context decision transcript"),
                    ),
                )
            }
        };
        let logits = session
            .copy_logits_f32()
            .expect("copy synthetic far-context logits");
        let causal_digest = if mode == RunMode::Transcript {
            Some(
                *session
                    .capture_causal_snapshot()
                    .expect("capture synthetic far-context causal state")
                    .causal_digest(),
            )
        } else {
            None
        };
        let evidence = Evidence {
            logits_sha256: logits_sha256(&logits),
            command_gpu_ms,
            wall_ms,
            causal_digest,
            transcript,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_LEGACY_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731-old/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = open_pinned_legacy_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
        .expect("plan far-context differential session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit far-context differential session");
    let mut residency = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize far-context differential residency")
        .into_residency();

    (residency, _) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let scalar_first;
    (residency, scalar_first) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::ScalarOracle,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let cooperative;
    (residency, cooperative) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let scalar_second;
    (residency, scalar_second) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::ScalarOracle,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );

    for evidence in [&scalar_first, &cooperative, &scalar_second] {
        assert_eq!(evidence.logits_sha256, PINNED_LOGITS_SHA256);
    }
    let cooperative_gpu = cooperative.command_gpu_ms.unwrap();
    let cooperative_wall = cooperative.wall_ms.unwrap();
    for (label, scalar) in [
        ("scalar-before", &scalar_first),
        ("scalar-after", &scalar_second),
    ] {
        let gpu_savings = scalar.command_gpu_ms.unwrap() - cooperative_gpu;
        let wall_savings = scalar.wall_ms.unwrap() - cooperative_wall;
        assert!(
            gpu_savings >= 8.0,
            "{label} command-GPU savings were {gpu_savings:.3} ms, need at least 8.0 ms"
        );
        assert!(
            wall_savings >= 8.0,
            "{label} wall savings were {wall_savings:.3} ms, need at least 8.0 ms"
        );
    }

    let bitwise_first;
    (residency, bitwise_first) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::BitwiseOracle,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let radix4;
    (residency, radix4) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let bitwise_second;
    (residency, bitwise_second) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::BitwiseOracle,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    for evidence in [&bitwise_first, &radix4, &bitwise_second] {
        assert_eq!(evidence.logits_sha256, PINNED_LOGITS_SHA256);
    }
    let radix4_gpu = radix4.command_gpu_ms.unwrap();
    let radix4_wall = radix4.wall_ms.unwrap();
    let bitwise_gpu_midpoint =
        (bitwise_first.command_gpu_ms.unwrap() + bitwise_second.command_gpu_ms.unwrap()) * 0.5;
    let bitwise_wall_midpoint =
        (bitwise_first.wall_ms.unwrap() + bitwise_second.wall_ms.unwrap()) * 0.5;
    let gpu_savings = bitwise_gpu_midpoint - radix4_gpu;
    let wall_savings = bitwise_wall_midpoint - radix4_wall;
    assert!(
        gpu_savings >= 1.0,
        "bitwise-bracket command-GPU savings were {gpu_savings:.3} ms, need at least 1.0 ms"
    );
    assert!(
        wall_savings >= 1.0,
        "bitwise-bracket wall savings were {wall_savings:.3} ms, need at least 1.0 ms"
    );

    let scalar_transcript;
    (residency, scalar_transcript) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::ScalarOracle,
        DeepSeekV4SelectorTestPolicy::BitwiseOracle,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Transcript,
    );
    let cooperative_transcript;
    (_, cooperative_transcript) = execute(
        &ctx,
        residency,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4SelectorTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Transcript,
    );
    assert_eq!(scalar_transcript.logits_sha256, PINNED_LOGITS_SHA256);
    assert_eq!(cooperative_transcript.logits_sha256, PINNED_LOGITS_SHA256);
    assert_eq!(
        cooperative_transcript.transcript, scalar_transcript.transcript,
        "all CSA scores, selected IDs, statuses, and route decisions must remain exact"
    );
    assert_eq!(
        cooperative_transcript.causal_digest, scalar_transcript.causal_digest,
        "cooperative scoring cannot change causal state"
    );
    assert_eq!(
        digest_hex(&cooperative_transcript.causal_digest.unwrap()),
        PINNED_CAUSAL_SHA256,
        "canonical synthetic causal state drifted"
    );

    eprintln!(
        "deepseek_v4 far_score_position={POSITION} scalar_before_gpu_ms={:.3} cooperative_gpu_ms={cooperative_gpu:.3} scalar_after_gpu_ms={:.3} scalar_before_wall_ms={:.3} cooperative_wall_ms={cooperative_wall:.3} scalar_after_wall_ms={:.3} bitwise_before_gpu_ms={:.3} radix4_gpu_ms={radix4_gpu:.3} bitwise_after_gpu_ms={:.3} bitwise_before_wall_ms={:.3} radix4_wall_ms={radix4_wall:.3} bitwise_after_wall_ms={:.3} logits_sha256={} causal_sha256={}",
        scalar_first.command_gpu_ms.unwrap(),
        scalar_second.command_gpu_ms.unwrap(),
        scalar_first.wall_ms.unwrap(),
        scalar_second.wall_ms.unwrap(),
        bitwise_first.command_gpu_ms.unwrap(),
        bitwise_second.command_gpu_ms.unwrap(),
        bitwise_first.wall_ms.unwrap(),
        bitwise_second.wall_ms.unwrap(),
        radix4.logits_sha256,
        digest_hex(&cooperative_transcript.causal_digest.unwrap()),
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current DS4 asset and executes focused far-context differentials"]
fn current_weight_boundary_optimization_differentials() {
    const POSITION: u32 = 65_663;
    const FORWARD_LIMIT: usize = POSITION as usize + 1;
    const TOKEN_ID: u32 = 35;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RunMode {
        Timed,
        Transcript,
    }

    struct Evidence {
        logits: Vec<f32>,
        logits_sha256: String,
        command_gpu_ms: Option<f64>,
        wall_ms: Option<f64>,
        causal_digest: Option<[u8; 32]>,
        transcript: Option<DeepSeekV4DecisionTranscript>,
    }

    fn vector_sha256(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").unwrap();
        }
        encoded
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0
    }

    fn assert_logit_envelope(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        assert_eq!(actual.len(), reference.len());
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut squared_error = 0.0f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            assert!(actual.is_finite());
            dot += f64::from(actual) * f64::from(reference);
            actual_norm += f64::from(actual).powi(2);
            reference_norm += f64::from(reference).powi(2);
            squared_error += f64::from(actual - reference).powi(2);
        }
        let cosine = dot / (actual_norm.sqrt() * reference_norm.sqrt());
        let relative_rms = (squared_error / reference_norm).sqrt();
        assert_eq!(argmax(actual), argmax(reference));
        assert!(cosine >= 0.999_99, "logit cosine {cosine}");
        assert!(relative_rms <= 0.005, "logit relative RMS {relative_rms}");
        (cosine, relative_rms)
    }

    fn assert_consumed_decisions(
        actual: &DeepSeekV4DecisionTranscript,
        reference: &DeepSeekV4DecisionTranscript,
    ) {
        assert_eq!(actual.position, reference.position);
        assert_eq!(actual.layers.len(), reference.layers.len());
        for (actual, reference) in actual.layers.iter().zip(&reference.layers) {
            assert_eq!(actual.layer, reference.layer);
            assert_eq!(actual.route.expert_ids, reference.route.expert_ids);
            match (&actual.csa, &reference.csa) {
                (Some(actual), Some(reference)) => {
                    assert_eq!(
                        actual.cache_order_selected_ids,
                        reference.cache_order_selected_ids
                    );
                    assert_eq!(actual.selected_count, reference.selected_count);
                    assert_eq!(actual.selection_status, reference.selection_status);
                }
                (None, None) => {}
                _ => panic!("layer {} CSA decision shape changed", actual.layer),
            }
        }
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        score_policy: DeepSeekV4IndexerScoreTestPolicy,
        policy: DeepSeekV4HcaTestPolicy,
        mode: RunMode,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct online HCA boundary session");
        session.sparse_csa.set_score_test_policy(score_policy);
        session.attention.set_hca_test_policy(policy);
        initialize_zero_synthetic_causal_state(&session);
        session.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
            next_position: POSITION,
        };
        session
            .committed_tokens
            .extend(std::iter::repeat_n(TOKEN_ID, POSITION as usize));

        let (command_gpu_ms, wall_ms, transcript) = match mode {
            RunMode::Timed => {
                let profile = session
                    .forward_token_whole_profiled(ctx, TOKEN_ID)
                    .expect("profile online HCA boundary token");
                (
                    Some(profile.command_gpu_ms),
                    Some(profile.forward_wall_ms),
                    None,
                )
            }
            RunMode::Transcript => {
                session
                    .arm_decision_transcript(POSITION)
                    .expect("arm online HCA boundary transcript");
                session
                    .forward_token(ctx, TOKEN_ID)
                    .expect("execute online HCA boundary transcript");
                (
                    None,
                    None,
                    Some(
                        session
                            .take_decision_transcript()
                            .expect("take online HCA boundary transcript"),
                    ),
                )
            }
        };
        let logits = session
            .copy_logits_f32()
            .expect("copy online HCA boundary logits");
        let causal_digest = if mode == RunMode::Transcript {
            Some(
                *session
                    .capture_causal_snapshot()
                    .expect("capture online HCA boundary causal state")
                    .causal_digest(),
            )
        } else {
            None
        };
        let evidence = Evidence {
            logits_sha256: vector_sha256(&logits),
            logits,
            command_gpu_ms,
            wall_ms,
            causal_digest,
            transcript,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
        .expect("plan online HCA boundary session");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit online HCA boundary session");
    let mut residency = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize online HCA boundary residency")
        .into_residency();

    (residency, _) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let legacy_first;
    (residency, legacy_first) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    let online;
    (residency, online) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Timed,
    );
    let legacy_second;
    (residency, legacy_second) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Timed,
    );
    assert_eq!(legacy_first.logits_sha256, legacy_second.logits_sha256);
    let (cosine, relative_rms) = assert_logit_envelope(&online.logits, &legacy_first.logits);
    let legacy_gpu_midpoint =
        (legacy_first.command_gpu_ms.unwrap() + legacy_second.command_gpu_ms.unwrap()) * 0.5;
    let legacy_wall_midpoint =
        (legacy_first.wall_ms.unwrap() + legacy_second.wall_ms.unwrap()) * 0.5;
    let gpu_savings = legacy_gpu_midpoint - online.command_gpu_ms.unwrap();
    let wall_savings = legacy_wall_midpoint - online.wall_ms.unwrap();

    let legacy_transcript;
    (residency, legacy_transcript) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::LegacyTiled,
        RunMode::Transcript,
    );
    let online_transcript;
    (residency, online_transcript) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Transcript,
    );
    assert_eq!(legacy_transcript.logits_sha256, legacy_first.logits_sha256);
    assert_eq!(online_transcript.logits_sha256, online.logits_sha256);
    assert_logit_envelope(&online_transcript.logits, &legacy_transcript.logits);
    assert_consumed_decisions(
        online_transcript.transcript.as_ref().unwrap(),
        legacy_transcript.transcript.as_ref().unwrap(),
    );

    eprintln!(
        "deepseek_v4 online_hca_boundary position={POSITION} legacy_before_gpu_ms={:.3} online_gpu_ms={:.3} legacy_after_gpu_ms={:.3} gpu_savings_ms={gpu_savings:.3} legacy_before_wall_ms={:.3} online_wall_ms={:.3} legacy_after_wall_ms={:.3} wall_savings_ms={wall_savings:.3} argmax={} cosine={cosine:.9} rel_rms={relative_rms:.9} legacy_logits_sha256={} online_logits_sha256={} legacy_causal_sha256={} online_causal_sha256={}",
        legacy_first.command_gpu_ms.unwrap(),
        online.command_gpu_ms.unwrap(),
        legacy_second.command_gpu_ms.unwrap(),
        legacy_first.wall_ms.unwrap(),
        online.wall_ms.unwrap(),
        legacy_second.wall_ms.unwrap(),
        argmax(&online.logits),
        legacy_first.logits_sha256,
        online.logits_sha256,
        digest_hex(&legacy_transcript.causal_digest.unwrap()),
        digest_hex(&online_transcript.causal_digest.unwrap()),
    );

    (residency, _) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Timed,
    );
    let score_control_first;
    (residency, score_control_first) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Timed,
    );
    let score_matrix;
    (residency, score_matrix) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::MatrixF16,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Timed,
    );
    let score_control_second;
    (residency, score_control_second) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Timed,
    );
    assert_eq!(
        score_control_first.logits_sha256,
        score_control_second.logits_sha256
    );
    assert_eq!(
        score_matrix.logits_sha256, score_control_first.logits_sha256,
        "F16 score arithmetic is unconsumed when selected IDs are unchanged"
    );
    let (score_cosine, score_relative_rms) =
        assert_logit_envelope(&score_matrix.logits, &score_control_first.logits);
    let score_control_gpu_midpoint = (score_control_first.command_gpu_ms.unwrap()
        + score_control_second.command_gpu_ms.unwrap())
        * 0.5;
    let score_control_wall_midpoint =
        (score_control_first.wall_ms.unwrap() + score_control_second.wall_ms.unwrap()) * 0.5;
    let score_gpu_savings = score_control_gpu_midpoint - score_matrix.command_gpu_ms.unwrap();
    let score_wall_savings = score_control_wall_midpoint - score_matrix.wall_ms.unwrap();

    let score_control_transcript;
    (residency, score_control_transcript) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::Production,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Transcript,
    );
    let score_matrix_transcript;
    (_, score_matrix_transcript) = execute(
        &ctx,
        residency,
        model_content_id,
        DeepSeekV4IndexerScoreTestPolicy::MatrixF16,
        DeepSeekV4HcaTestPolicy::Production,
        RunMode::Transcript,
    );
    assert_consumed_decisions(
        score_matrix_transcript.transcript.as_ref().unwrap(),
        score_control_transcript.transcript.as_ref().unwrap(),
    );
    assert_eq!(
        score_matrix_transcript.logits_sha256,
        score_control_transcript.logits_sha256
    );
    assert_logit_envelope(
        &score_matrix_transcript.logits,
        &score_control_transcript.logits,
    );
    assert_eq!(
        score_matrix_transcript.causal_digest, score_control_transcript.causal_digest,
        "score arithmetic cannot change causal state when consumed IDs are unchanged"
    );
    assert!(
        score_gpu_savings >= 1.5,
        "F16 matrix scorer GPU savings {score_gpu_savings:.3} ms"
    );
    assert!(
        score_wall_savings >= 1.0,
        "F16 matrix scorer wall savings {score_wall_savings:.3} ms"
    );
    eprintln!(
        "deepseek_v4 lightning_f16_matrix_boundary position={POSITION} visible_rows={} control_before_gpu_ms={:.3} matrix_gpu_ms={:.3} control_after_gpu_ms={:.3} gpu_savings_ms={score_gpu_savings:.3} control_before_wall_ms={:.3} matrix_wall_ms={:.3} control_after_wall_ms={:.3} wall_savings_ms={score_wall_savings:.3} argmax={} cosine={score_cosine:.9} rel_rms={score_relative_rms:.9} control_logits_sha256={} matrix_logits_sha256={} control_causal_sha256={} matrix_causal_sha256={}",
        (POSITION as usize + 1) / 4,
        score_control_first.command_gpu_ms.unwrap(),
        score_matrix.command_gpu_ms.unwrap(),
        score_control_second.command_gpu_ms.unwrap(),
        score_control_first.wall_ms.unwrap(),
        score_matrix.wall_ms.unwrap(),
        score_control_second.wall_ms.unwrap(),
        argmax(&score_matrix.logits),
        score_control_first.logits_sha256,
        score_matrix.logits_sha256,
        digest_hex(&score_control_transcript.causal_digest.unwrap()),
        digest_hex(&score_matrix_transcript.causal_digest.unwrap()),
    );
}

#[test]
fn memory_plan_admission_and_reconciliation_fail_closed() {
    let plan = DeepSeekV4MemoryPlan {
        residency_buffer_count: 4,
        residency_logical_bytes: 900,
        residency_priced_upper_bytes: 1_000,
        session_logical_bytes: 400,
        session_priced_upper_bytes: 500,
        total_priced_upper_bytes: 1_500,
        session_allocations: Vec::new(),
    };
    let required = plan.required_with_reserve_bytes().unwrap();
    let baseline = 100_u64;
    let exact = plan.admission(MetalMemorySignals {
        recommended_max_bytes: baseline + required,
        current_allocated_bytes: baseline,
        process_limit_remaining_bytes: Some(0),
    });
    assert!(exact.admitted);
    assert_eq!(
        exact.reason,
        crate::metal::MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
    );
    let short = plan.admission(MetalMemorySignals {
        recommended_max_bytes: baseline + required - 1,
        current_allocated_bytes: baseline,
        process_limit_remaining_bytes: Some(0),
    });
    assert!(!short.admitted);
    assert_eq!(
        short.reason,
        crate::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
    );

    let three_session_priced = 1_000 + 3 * 500;
    assert_eq!(
        plan.priced_upper_bytes_for_sessions(3).unwrap(),
        three_session_priced
    );
    assert_eq!(
        plan.required_with_reserve_bytes_for_sessions(3).unwrap(),
        three_session_priced + DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES
    );
    let three_required = plan.required_with_reserve_bytes_for_sessions(3).unwrap();
    let three_exact = plan
        .admission_for_sessions(
            MetalMemorySignals {
                recommended_max_bytes: baseline + three_required,
                current_allocated_bytes: baseline,
                process_limit_remaining_bytes: Some(0),
            },
            3,
        )
        .unwrap();
    assert!(three_exact.admitted);
    let error = plan
        .admission_for_sessions(
            MetalMemorySignals {
                recommended_max_bytes: u64::MAX,
                current_allocated_bytes: 0,
                process_limit_remaining_bytes: Some(0),
            },
            0,
        )
        .unwrap_err();
    assert!(error.to_string().contains("at least one session"));

    let reconciliation = plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes: baseline,
            after_residency_bytes: baseline + 1_000,
            after_session_bytes: baseline + 1_500,
            after_first_forward_bytes: baseline + required,
        })
        .unwrap();
    assert_eq!(reconciliation.observed_residency_delta_bytes, 1_000);
    assert_eq!(reconciliation.observed_session_delta_bytes, 1_500);
    assert_eq!(reconciliation.sampled_peak_delta_bytes, required);
    let released = plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes: baseline,
            after_residency_bytes: baseline - 1,
            after_session_bytes: baseline - 2,
            after_first_forward_bytes: baseline - 3,
        })
        .unwrap();
    assert_eq!(released.observed_residency_delta_bytes, 0);
    assert_eq!(released.observed_session_delta_bytes, 0);
    assert_eq!(released.observed_first_forward_delta_bytes, 0);
    assert_eq!(released.sampled_peak_delta_bytes, 0);
    let error = plan
        .reconcile_session(baseline, baseline + 999, baseline + 1_500)
        .unwrap_err();
    assert!(error.to_string().contains("session increment"));
    let error = plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes: baseline,
            after_residency_bytes: baseline + 1_000,
            after_session_bytes: baseline + 1_500,
            after_first_forward_bytes: baseline + required + 1,
        })
        .unwrap_err();
    assert!(error.to_string().contains("first forward delta"));
}

#[test]
fn shared_residency_can_cross_dedicated_queue_threads() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<DeepSeekV4MetalResidency>();
    assert_sync::<DeepSeekV4MetalResidency>();
}

#[test]
fn retained_plan_preflight_rejects_descriptor_and_window_drift() {
    let tensors = vec![f32_desc("a", 0), f32_desc("b", 32)];
    let requests = tensors.iter().collect::<Vec<_>>();
    let plan = plan_retained_storage(&[131_072], &requests, 4_096, 65_536, 32).unwrap();
    validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &plan).unwrap();

    let mut descriptor_drift = plan.clone();
    descriptor_drift.entries[1].data_offset += 32;
    let error =
        validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &descriptor_drift)
            .unwrap_err();
    assert!(error.to_string().contains("descriptor drift"));

    let mut window_drift = plan;
    window_drift.windows[0].length = 135_168;
    let error = validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &window_drift)
        .unwrap_err();
    assert!(error.to_string().contains("window 0"));

    let plan = plan_retained_storage(&[131_072], &requests, 4_096, 65_536, 32).unwrap();
    let mut usable_window_drift = plan.clone();
    usable_window_drift.usable_window_length -= 4_096;
    let error =
        validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &usable_window_drift)
            .unwrap_err();
    assert!(error.to_string().contains("geometry changed"));

    let mut view_drift = plan;
    let RetainedStorageDisposition::View { buffer_offset, .. } =
        &mut view_drift.entries[0].disposition
    else {
        panic!("expected retained view");
    };
    *buffer_offset += 1;
    let error = validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &view_drift)
        .unwrap_err();
    assert!(error.to_string().contains("window binding"));

    let alias_tensors = vec![f32_desc("source", 0), f32_desc("alias", 0)];
    let alias_requests = alias_tensors.iter().collect::<Vec<_>>();
    let mut alias_plan =
        plan_retained_storage(&[131_072], &alias_requests, 4_096, 65_536, 32).unwrap();
    alias_plan.entries[1].disposition = RetainedStorageDisposition::Alias {
        source_request_index: 1,
    };
    let error =
        validate_retained_plan_geometry(4_096, 65_536, &[131_072], &alias_tensors, &alias_plan)
            .unwrap_err();
    assert!(error.to_string().contains("non-prior source"));

    let tail_tensors = vec![f32_desc("tail", 131_072)];
    let tail_requests = tail_tensors.iter().collect::<Vec<_>>();
    let tail_plan = plan_retained_storage(&[131_120], &tail_requests, 4_096, 65_536, 32).unwrap();
    assert!(matches!(
        tail_plan.entries[0].disposition,
        RetainedStorageDisposition::CopyFallback {
            reason: RetainedStorageFallback::FinalPartialPage
        }
    ));
    validate_retained_plan_geometry(4_096, 65_536, &[131_120], &tail_tensors, &tail_plan).unwrap();
    let error =
        validate_retained_plan_geometry(4_096, 65_536, &[135_168], &tail_tensors, &tail_plan)
            .unwrap_err();
    assert!(error.to_string().contains("final-partial-page fallback"));
}

#[test]
fn descriptor_fingerprints_cover_representation_not_just_storage_range() {
    let tensors = vec![f32_desc("a", 0), f32_desc("b", 32)];
    let fingerprints = tensors
        .iter()
        .map(DeepSeekV4DescriptorFingerprint::from)
        .collect::<Vec<_>>();
    validate_descriptor_fingerprints(&tensors, &fingerprints).unwrap();

    let mut dtype_drift = tensors.clone();
    dtype_drift[0].dtype = GgmlType::I32;
    let error = validate_descriptor_fingerprints(&dtype_drift, &fingerprints).unwrap_err();
    assert!(error.to_string().contains("fingerprint changed"));

    let mut shape_drift = tensors;
    shape_drift[1].shape = vec![4, 2];
    let error = validate_descriptor_fingerprints(&shape_drift, &fingerprints).unwrap_err();
    assert!(error.to_string().contains("fingerprint changed"));
}

#[test]
fn session_lookup_storage_rejects_unsupported_dtypes_before_residency() {
    validate_session_lookup_dtypes(GgmlType::Q6_K, GgmlType::Q6_K).unwrap();
    validate_session_lookup_dtypes(GgmlType::Q8_0, GgmlType::Q8_0).unwrap();
    validate_session_lookup_dtypes(GgmlType::Q6_K, GgmlType::Q8_0).unwrap();
    validate_session_lookup_dtypes(GgmlType::Q8_0, GgmlType::Q6_K).unwrap();
    let embedding = validate_session_lookup_dtypes(GgmlType::F32, GgmlType::Q6_K).unwrap_err();
    assert!(embedding.to_string().contains("token_embd.weight"));
    let output = validate_session_lookup_dtypes(GgmlType::Q6_K, GgmlType::F32).unwrap_err();
    assert!(output.to_string().contains("output.weight"));
}

#[test]
fn position_zero_attention_matches_operation_oracles_with_offsets_and_groups() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 7,
        q_lora_rank: 3,
        head_count: 3,
        head_dim: 192,
        rotary_dim: 64,
        group_count: 3,
        output_rank: 2,
    };
    let c = config;
    let query_width = c.head_count * c.head_dim;
    let group_width = query_width / c.group_count;
    let low_rank_width = c.group_count * c.output_rank;
    let rms_eps = 2.0e-5;
    let input_values = (0..c.hidden_size)
        .map(|i| (i as f32 - 2.6) * 0.31 + if i % 2 == 0 { 0.17 } else { -0.23 })
        .collect::<Vec<_>>();
    let attention_norm_values = (0..c.hidden_size)
        .map(|i| 0.61 + i as f32 * 0.083)
        .collect::<Vec<_>>();
    let q_a_values = (0..c.hidden_size * c.q_lora_rank)
        .map(|i| ((i * 11 + 3) % 23) as f32 * 0.037 - 0.39)
        .collect::<Vec<_>>();
    let q_a_norm_values = vec![0.73, 1.19, 0.52];
    let q_b_values = (0..c.q_lora_rank * query_width)
        .map(|i| ((i * 7 + i / 5 + 1) % 29) as f32 * 0.029 - 0.36)
        .collect::<Vec<_>>();
    let kv_weight_values = (0..c.hidden_size * c.head_dim)
        .map(|i| ((i * 13 + 5) % 31) as f32 * 0.021 - 0.28)
        .collect::<Vec<_>>();
    let kv_norm_values = (0..c.head_dim)
        .map(|i| 0.51 + ((i * 19 + i / 7) % 47) as f32 * 0.029)
        .collect::<Vec<_>>();
    let sink_values = vec![-0.83, 0.41, 1.27];
    let output_a_values = (0..group_width * low_rank_width)
        .map(|i| {
            let row = i / group_width;
            let column = i % group_width;
            (row as f32 - 2.1) * 0.17
                + (column as f32 - 1.7) * 0.09
                + if (row + column).is_multiple_of(2) {
                    0.14
                } else {
                    -0.08
                }
        })
        .collect::<Vec<_>>();
    let output_b_values = (0..low_rank_width * c.hidden_size)
        .map(|i| ((i * 17 + i / 4 + 2) % 37) as f32 * 0.018 - 0.31)
        .collect::<Vec<_>>();

    let expected_normalized =
        rms_norm(&input_values, Some(&attention_norm_values), rms_eps).unwrap();
    let expected_projection = shared_kv_projection(
        &expected_normalized,
        &q_a_values,
        &q_a_norm_values,
        &q_b_values,
        &kv_weight_values,
        &kv_norm_values,
        c.q_lora_rank,
        c.head_count,
        c.head_dim,
        rms_eps,
    )
    .unwrap();
    let mut expected_cached_kv = expected_projection.kv.clone();
    attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut expected_cached_kv, c.rotary_dim).unwrap();
    let expected_attention = shared_kv_attention(
        &expected_projection.queries,
        c.head_count,
        c.head_dim,
        &expected_cached_kv,
        &[],
        None,
        &sink_values,
    )
    .unwrap();
    let direct_f32_attention = shared_kv_attention(
        &expected_projection.queries,
        c.head_count,
        c.head_dim,
        &expected_projection.kv,
        &[],
        None,
        &sink_values,
    )
    .unwrap();
    assert!(
        direct_f32_attention
            .iter()
            .zip(&expected_attention)
            .any(|(old, expected)| (old - expected).abs() > 1.0e-4),
        "fixture must reject direct-F32 same-token KV"
    );
    let expected_low_rank = grouped_low_rank_projection(
        &expected_attention,
        c.head_count,
        c.head_dim,
        c.group_count,
        c.output_rank,
        &output_a_values,
    )
    .unwrap();
    let expected_output = mat_vec(
        &output_b_values,
        low_rank_width,
        c.hidden_size,
        &expected_low_rank,
    )
    .unwrap();

    let input = offset_f32(&ctx, &input_values, vec![c.hidden_size as u64]);
    let attention_norm = offset_f32(&ctx, &attention_norm_values, vec![c.hidden_size as u64]);
    let q_a = offset_f32(
        &ctx,
        &q_a_values,
        vec![c.hidden_size as u64, c.q_lora_rank as u64],
    );
    let q_a_norm = offset_f32(&ctx, &q_a_norm_values, vec![c.q_lora_rank as u64]);
    let q_b = offset_f32(
        &ctx,
        &q_b_values,
        vec![c.q_lora_rank as u64, query_width as u64],
    );
    let kv_weight = offset_f32(
        &ctx,
        &kv_weight_values,
        vec![c.hidden_size as u64, c.head_dim as u64],
    );
    let kv_norm = offset_f32(&ctx, &kv_norm_values, vec![c.head_dim as u64]);
    let sinks = offset_f32(&ctx, &sink_values, vec![c.head_count as u64]);
    let output_a = offset_f32(
        &ctx,
        &output_a_values,
        vec![group_width as u64, low_rank_width as u64],
    );
    let output_b = offset_f32(
        &ctx,
        &output_b_values,
        vec![low_rank_width as u64, c.hidden_size as u64],
    );
    let scratch =
        DeepSeekV4PositionZeroAttentionScratch::new(&ctx, config).expect("attention scratch");
    assert_eq!(read_f32(&scratch.head_norm_ones), vec![1.0; c.head_dim]);

    let mut cache_fixture = (0..c.head_dim)
        .map(|i| (i as f32 - 91.0) * 0.000_061_035_156)
        .collect::<Vec<_>>();
    cache_fixture[0] = 1.003_906_3;
    cache_fixture[1] = 0.004_150_390_6;
    cache_fixture[2] = 0.004_638_672;
    cache_fixture[3] = -0.004_150_390_6;
    cache_fixture[63] = 1.5;
    cache_fixture[64] = 1.0625;
    cache_fixture[65] = 1.1875;
    cache_fixture[66] = -1.0625;
    cache_fixture[127] = 448.0;
    cache_fixture[128] = 1.003_906_3;
    cache_fixture[129] = 1.011_718_8;
    cache_fixture[130] = -1.003_906_3;
    let mut expected_cache_fixture = cache_fixture.clone();
    attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut expected_cache_fixture, c.rotary_dim)
        .unwrap();
    let cache_fixture_input = offset_f32(&ctx, &cache_fixture, vec![c.head_dim as u64]);
    let cache_fixture_output = offset_f32(&ctx, &vec![0.0; c.head_dim], vec![c.head_dim as u64]);
    let cache_command = ctx.queue.commandBuffer().expect("cache command buffer");
    let cache_encoder = KernelEncoder::begin(&cache_command);
    encode_attention_cache_roundtrip(
        &ctx,
        &cache_encoder,
        &cache_fixture_input,
        &cache_fixture_output,
        c,
    )
    .expect("encode attention cache roundtrip");
    cache_encoder.end();
    cache_command.commit();
    cache_command.waitUntilCompleted();
    assert!(
        cache_command.error().is_none(),
        "cache command failed: {:?}",
        cache_command.error()
    );
    let actual_cache_fixture = read_f32(&cache_fixture_output);
    assert!(
        actual_cache_fixture
            .iter()
            .zip(&expected_cache_fixture)
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()),
        "cache fixture must match the oracle bit-for-bit"
    );

    let command = ctx.queue.commandBuffer().expect("attention command buffer");
    let encoder = KernelEncoder::begin(&command);
    let encoded_output = scratch
        .encode(
            &ctx,
            &encoder,
            &input,
            &attention_norm,
            &q_a,
            &q_a_norm,
            &q_b,
            &kv_weight,
            &kv_norm,
            &sinks,
            &output_a,
            &output_b,
            rms_eps,
        )
        .expect("encode position-zero attention");
    assert!(std::ptr::eq(encoded_output, scratch.output()));
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "attention command failed: {:?}",
        command.error()
    );

    assert_close(
        "attention input norm",
        &read_f32(scratch.normalized_input()),
        &expected_normalized,
        3e-5,
    );
    assert_close(
        "Q LoRA raw",
        &read_f32(scratch.q_lora_raw()),
        &expected_projection.q_lora_raw,
        4e-5,
    );
    assert_close(
        "Q LoRA",
        &read_f32(scratch.q_lora()),
        &expected_projection.q_lora,
        4e-5,
    );
    assert_close(
        "queries",
        &read_f32(scratch.queries()),
        &expected_projection.queries,
        5e-5,
    );
    assert_close(
        "KV raw",
        &read_f32(scratch.kv_raw()),
        &expected_projection.kv_raw,
        4e-5,
    );
    assert_close("KV", &read_f32(scratch.kv()), &expected_projection.kv, 4e-5);
    assert_close(
        "cached KV",
        &read_f32(scratch.cached_kv()),
        &expected_cached_kv,
        0.0,
    );
    assert_close(
        "sink attention",
        &read_f32(scratch.attention_heads()),
        &expected_attention,
        6e-5,
    );
    assert_close(
        "grouped low rank",
        &read_f32(scratch.low_rank()),
        &expected_low_rank,
        7e-5,
    );
    assert_close(
        "block output",
        &read_f32(scratch.output()),
        &expected_output,
        8e-5,
    );

    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let mut sink_as_value = vec![0.0; query_width];
    for head in 0..c.head_count {
        let query = &expected_projection.queries[head * c.head_dim..(head + 1) * c.head_dim];
        let score = query
            .iter()
            .zip(&expected_cached_kv)
            .map(|(q, k)| q * k)
            .sum::<f32>()
            * scale;
        let maximum = score.max(sink_values[head]);
        let kv_mass = (score - maximum).exp();
        let sink_mass = (sink_values[head] - maximum).exp();
        for dimension in 0..c.head_dim {
            sink_as_value[head * c.head_dim + dimension] =
                (expected_cached_kv[dimension] * kv_mass + sink_values[head] * sink_mass)
                    / (kv_mass + sink_mass);
        }
    }
    assert!(
        sink_as_value
            .iter()
            .zip(&expected_attention)
            .any(|(wrong, right)| (wrong - right).abs() > 1e-2),
        "fixture must reject treating sink denominator mass as a value"
    );

    let mut reused_first_group_rows = Vec::with_capacity(low_rank_width);
    for group in 0..c.group_count {
        reused_first_group_rows.extend(
            mat_vec(
                &output_a_values[..group_width * c.output_rank],
                group_width,
                c.output_rank,
                &expected_attention[group * group_width..(group + 1) * group_width],
            )
            .unwrap(),
        );
    }
    assert!(
        reused_first_group_rows
            .iter()
            .zip(&expected_low_rank)
            .any(|(wrong, right)| (wrong - right).abs() > 1e-2),
        "fixture must reject cross-group output A row indexing"
    );
}

#[test]
fn continuing_rope_f16_cache_and_local_attention_match_cpu() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const ROTARY: usize = 64;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: ROTARY,
        group_count: 1,
        output_rank: 1,
    };
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: ROTARY,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::yarn(
        ROTARY,
        rope.theta,
        rope.scaling_factor,
        rope.original_context_length,
        rope.beta_fast,
        rope.beta_slow,
    );
    let query_values = (0..HEADS * HEAD_DIM)
        .map(|index| {
            (index as f32 - 91.0) * 0.0037 + if index.is_multiple_of(5) { 0.19 } else { -0.07 }
        })
        .collect::<Vec<_>>();
    let kv0 = (0..HEAD_DIM)
        .map(|index| (index as f32 - 43.0) * 0.0051 + (index % 7) as f32 * 0.013)
        .collect::<Vec<_>>();
    let kv1 = (0..HEAD_DIM)
        .map(|index| (67.0 - index as f32) * 0.0043 - (index % 11) as f32 * 0.009)
        .collect::<Vec<_>>();
    let sinks = vec![-0.37, 0.82];

    let mut expected_queries = query_values.clone();
    rope_tail_in_place(
        &mut expected_queries,
        HEADS,
        HEAD_DIM,
        1,
        oracle_rope,
        RopeDirection::Forward,
    )
    .unwrap();
    let mut expected_kv1 = kv1.clone();
    rope_tail_in_place(
        &mut expected_kv1,
        1,
        HEAD_DIM,
        1,
        oracle_rope,
        RopeDirection::Forward,
    )
    .unwrap();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let cached_kv0 = kv0.iter().copied().map(round_f16).collect::<Vec<_>>();
    let cached_kv1 = expected_kv1
        .iter()
        .copied()
        .map(round_f16)
        .collect::<Vec<_>>();
    let mut raw_rows = cached_kv0.clone();
    raw_rows.extend_from_slice(&cached_kv1);
    let mut expected_output = shared_kv_attention(
        &expected_queries,
        HEADS,
        HEAD_DIM,
        &raw_rows,
        &[],
        None,
        &sinks,
    )
    .unwrap();
    rope_tail_in_place(
        &mut expected_output,
        HEADS,
        HEAD_DIM,
        1,
        oracle_rope,
        RopeDirection::Inverse,
    )
    .unwrap();

    let queries = offset_f32(&ctx, &query_values, vec![HEAD_DIM as u64, HEADS as u64]);
    let kv0_tensor = offset_f32(&ctx, &kv0, vec![HEAD_DIM as u64]);
    let kv1_tensor = offset_f32(&ctx, &kv1, vec![HEAD_DIM as u64]);
    let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
    let raw_cache =
        MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
            .unwrap();
    let output = offset_f32(
        &ctx,
        &vec![0.0; HEADS * HEAD_DIM],
        vec![HEAD_DIM as u64, HEADS as u64],
    );
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_scatter_offset_f32_to_f16(&ctx, &encoder, &kv0_tensor, &raw_cache, 0, HEAD_DIM).unwrap();
    encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &queries, 1, rope, false).unwrap();
    encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &kv1_tensor, 1, rope, false).unwrap();
    encode_scatter_offset_f32_to_f16(&ctx, &encoder, &kv1_tensor, &raw_cache, HEAD_DIM, HEAD_DIM)
        .unwrap();
    encode_local_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_cache,
        &sink_tensor,
        &output,
        1,
        config,
    )
    .unwrap();
    encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &output, 1, rope, true).unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );

    assert_close(
        "forward adjacent RoPE",
        &read_f32(&queries),
        &expected_queries,
        2e-6,
    );
    assert_close(
        "forward KV RoPE",
        &read_f32(&kv1_tensor),
        &expected_kv1,
        2e-6,
    );
    assert_close(
        "continuing local attention",
        &read_f32(&output),
        &expected_output,
        2e-5,
    );
}

#[test]
fn yarn_rope_pins_direct_power_through_the_full_context_regime() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const ROTARY: usize = 64;
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: ROTARY,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::yarn(
        ROTARY,
        rope.theta,
        rope.scaling_factor,
        rope.original_context_length,
        rope.beta_fast,
        rope.beta_slow,
    );
    let values = (0..HEADS * HEAD_DIM)
        .map(|index| {
            let head = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            (dimension as f32 - 61.0) * 0.0041 + head as f32 * 0.037
        })
        .collect::<Vec<_>>();
    let correction = |rotations: f32| {
        ROTARY as f32
            * (rope.original_context_length as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * rope.theta.ln())
    };
    let correction_low = correction(rope.beta_fast).floor().max(0.0);
    let correction_high = correction(rope.beta_slow).ceil().min((ROTARY - 1) as f32);
    let direct_power = |position: u32, inverse: bool| {
        let mut output = values.clone();
        for head in 0..HEADS {
            let tail = head * HEAD_DIM + HEAD_DIM - ROTARY;
            for pair in 0..ROTARY / 2 {
                let relative = pair * 2;
                let extrapolated =
                    position as f32 * rope.theta.powf(-(relative as f32) / rope.rotary_dim as f32);
                let interpolated = extrapolated / rope.scaling_factor;
                let ramp = 1.0
                    - ((pair as f32 - correction_low)
                        / (correction_high - correction_low).max(0.001))
                    .clamp(0.0, 1.0);
                let angle = interpolated * (1.0 - ramp) + extrapolated * ramp;
                let (mut sine, cosine) = angle.sin_cos();
                if inverse {
                    sine = -sine;
                }
                let first = tail + relative;
                let second = first + 1;
                let x = output[first];
                let y = output[second];
                output[first] = x * cosine - y * sine;
                output[second] = x * sine + y * cosine;
            }
        }
        output
    };
    for position in [65_535u32, 65_536, 1_048_448, 1_048_575] {
        for (direction, inverse) in [
            (RopeDirection::Forward, false),
            (RopeDirection::Inverse, true),
        ] {
            let mut iterative = values.clone();
            rope_tail_in_place(
                &mut iterative,
                HEADS,
                HEAD_DIM,
                position,
                oracle_rope,
                direction,
            )
            .unwrap();
            let actual = offset_f32(&ctx, &values, vec![HEAD_DIM as u64, HEADS as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_ds4_rope_tail_adjacent_in_place(
                &ctx, &encoder, &actual, position, rope, inverse,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            let actual = read_f32(&actual);
            let direct = direct_power(position, inverse);
            let metrics = |reference: &[f32]| {
                let mut squared_error = 0.0f64;
                let mut reference_norm = 0.0f64;
                let mut max_error = 0.0f32;
                for (&actual, &reference) in actual.iter().zip(reference) {
                    squared_error += f64::from(actual - reference).powi(2);
                    reference_norm += f64::from(reference).powi(2);
                    max_error = max_error.max((actual - reference).abs());
                }
                ((squared_error / reference_norm).sqrt(), max_error)
            };
            let (direct_relative_rms, direct_max_error) = metrics(&direct);
            let (iterative_relative_rms, iterative_max_error) = metrics(&iterative);
            eprintln!(
                "deepseek_v4 yarn position={position} inverse={inverse} direct_rel_rms={direct_relative_rms:.9} direct_max_abs={direct_max_error} iterative_rel_rms={iterative_relative_rms:.9} iterative_max_abs={iterative_max_error}"
            );
            assert!(direct_relative_rms <= 0.01);
            assert!(direct_max_error <= 0.02);
            assert!(iterative_relative_rms <= 0.01);
            assert!(iterative_max_error <= 0.02);
        }
    }
}

#[test]
fn batched_rope_matches_position_ordered_rows() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 3;
    const HEAD_DIM: usize = 128;
    const ROWS: usize = 17;
    let values = (0..ROWS * HEADS * HEAD_DIM)
        .map(|index| {
            let row = index / (HEADS * HEAD_DIM);
            let within = index % (HEADS * HEAD_DIM);
            let head = within / HEAD_DIM;
            let dimension = within % HEAD_DIM;
            (dimension as f32 - 61.0) * 0.0037 + head as f32 * 0.021 + row as f32 * 0.0043
        })
        .collect::<Vec<_>>();
    for (start_position, rope) in [
        (
            0_u32,
            DeepSeekV4RopeParameters {
                rotary_dim: 64,
                theta: 10_000.0,
                scaling_factor: 1.0,
                original_context_length: 0,
                beta_fast: 0.0,
                beta_slow: 0.0,
            },
        ),
        (
            65_531,
            DeepSeekV4RopeParameters {
                rotary_dim: 64,
                theta: 160_000.0,
                scaling_factor: 16.0,
                original_context_length: 65_536,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        ),
    ] {
        for inverse in [false, true] {
            let batched = offset_f32(
                &ctx,
                &values,
                vec![HEAD_DIM as u64, HEADS as u64, ROWS as u64],
            );
            let ordered = offset_f32(
                &ctx,
                &values,
                vec![HEAD_DIM as u64, HEADS as u64, ROWS as u64],
            );
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_ds4_rope_tail_adjacent_batch_in_place(
                &ctx,
                &encoder,
                &batched,
                start_position,
                ROWS,
                1,
                rope,
                inverse,
            )
            .unwrap();
            for row in 0..ROWS {
                let row_view = ordered.view_subrange(
                    (row * HEADS * HEAD_DIM) as u64,
                    vec![HEAD_DIM as u64, HEADS as u64],
                );
                encode_ds4_rope_tail_adjacent_in_place(
                    &ctx,
                    &encoder,
                    &row_view,
                    start_position + row as u32,
                    rope,
                    inverse,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            let batched = read_f32(&batched);
            let ordered = read_f32(&ordered);
            let differing = batched
                .iter()
                .zip(&ordered)
                .filter(|(batched, ordered)| batched.to_bits() != ordered.to_bits())
                .count();
            if start_position == 0 {
                assert_eq!(differing, 0, "unscaled batched RoPE changed lineage");
                continue;
            }
            let mut squared_error = 0.0_f64;
            let mut reference_norm = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (&batched, &ordered) in batched.iter().zip(&ordered) {
                squared_error += f64::from(batched - ordered).powi(2);
                reference_norm += f64::from(ordered).powi(2);
                max_abs = max_abs.max((batched - ordered).abs());
            }
            let relative_rms = (squared_error / reference_norm).sqrt();
            eprintln!(
                "deepseek_v4 batched_rope start={start_position} inverse={inverse} differing={differing}/{} max_abs={max_abs:.9} rel_rms={relative_rms:.9}",
                batched.len(),
            );
            assert!(max_abs <= 1.0e-6, "batched RoPE max abs {max_abs}");
            assert!(
                relative_rms <= 5.0e-7,
                "batched RoPE relative RMS {relative_rms}"
            );
        }
    }
}

#[test]
fn unscaled_rope_remains_bounded_through_the_full_context_regime() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 128;
    const ROTARY: usize = 64;
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: ROTARY,
        theta: 10_000.0,
        scaling_factor: 1.0,
        original_context_length: 0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };
    let oracle_rope = RopeParameters::local(ROTARY, rope.theta);
    let values = (0..HEAD_DIM)
        .map(|dimension| (dimension as f32 - 59.0) * 0.0037)
        .collect::<Vec<_>>();
    for position in [65_536u32, 1_048_575] {
        let mut expected = values.clone();
        rope_tail_in_place(
            &mut expected,
            1,
            HEAD_DIM,
            position,
            oracle_rope,
            RopeDirection::Forward,
        )
        .unwrap();
        let actual = offset_f32(&ctx, &values, vec![HEAD_DIM as u64, 1]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &actual, position, rope, false)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let actual = read_f32(&actual);
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for (&actual, &reference) in actual.iter().zip(&expected) {
            squared_error += f64::from(actual - reference).powi(2);
            reference_norm += f64::from(reference).powi(2);
            max_error = max_error.max((actual - reference).abs());
        }
        let relative_rms = (squared_error / reference_norm).sqrt();
        eprintln!(
            "deepseek_v4 unscaled_rope position={position} iterative_rel_rms={relative_rms:.9} iterative_max_abs={max_error}"
        );
        assert!(relative_rms <= 0.01);
        assert!(max_error <= 0.02);
    }
}

#[test]
fn compressor_frontier_projects_ape_into_the_position_one_lane() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HIDDEN: usize = 3;
    const HEAD_DIM: usize = 2;
    const WIDTH: usize = HEAD_DIM * 2;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        4,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::Attention,
        DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
    )
    .unwrap();
    let input_values = [0.7, -0.4, 1.1];
    let kv_weights = (0..HIDDEN * WIDTH)
        .map(|index| (index as f32 - 4.0) * 0.07)
        .collect::<Vec<_>>();
    let score_weights = (0..HIDDEN * WIDTH)
        .map(|index| (5.0 - index as f32) * 0.043)
        .collect::<Vec<_>>();
    let ape_values = (0..4 * WIDTH)
        .map(|index| (index as f32 - 6.0) * 0.019)
        .collect::<Vec<_>>();
    let expected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, &input_values).unwrap();
    let expected_score = mat_vec(&score_weights, HIDDEN, WIDTH, &input_values)
        .unwrap()
        .into_iter()
        .zip(&ape_values[WIDTH..2 * WIDTH])
        .map(|(score, ape)| score + ape)
        .collect::<Vec<_>>();

    let input = offset_f32(&ctx, &input_values, vec![HIDDEN as u64]);
    let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
    let norm = offset_f32(&ctx, &[1.0; HEAD_DIM], vec![HEAD_DIM as u64]);
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: HEAD_DIM,
        theta: 10_000.0,
        scaling_factor: 1.0,
        original_context_length: 0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode(
            &ctx,
            &encoder,
            &input,
            &kv_weight,
            &score_weight,
            &ape,
            &norm,
            1,
            HIDDEN,
            rope,
            1e-5,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );

    let kv_state = read_f32(&frontier.kv_state);
    let score_state = read_f32(&frontier.score_state);
    let row_start = 5 * WIDTH;
    assert_close(
        "position-one compressor KV",
        &kv_state[row_start..row_start + WIDTH],
        &expected_kv,
        2e-5,
    );
    assert_close(
        "position-one compressor score plus APE",
        &score_state[row_start..row_start + WIDTH],
        &expected_score,
        2e-5,
    );
    assert!(
        score_state[..row_start]
            .iter()
            .chain(&score_state[row_start + WIDTH..])
            .all(|value| *value == f32::NEG_INFINITY)
    );
}

#[test]
fn compressor_chunk_matches_ordered_state_and_publications() {
    let Some(ctx) = metal_context() else {
        return;
    };

    fn assert_bits(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            panic!(
                "{label} first mismatch at {index}: actual={} ({:08x}) expected={} ({:08x})",
                actual[index],
                actual[index].to_bits(),
                expected[index],
                expected[index].to_bits(),
            );
        }
    }

    let run = |ratio: usize,
               head_dim: usize,
               publication: DeepSeekV4CompressorPublication,
               start_position: usize,
               row_count: usize| {
        let width = if ratio == 4 { 2 * head_dim } else { head_dim };
        let history_capacity =
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS.max(DEEPSEEK_V4_PREFILL_MAX_TOKENS / 4);
        let ordered =
            DeepSeekV4CompressorFrontier::new(&ctx, ratio, head_dim, publication, history_capacity)
                .unwrap();
        let batched =
            DeepSeekV4CompressorFrontier::new(&ctx, ratio, head_dim, publication, history_capacity)
                .unwrap();
        #[cfg(feature = "dsv4-diagnostics")]
        let (mut ordered, mut batched) = (ordered, batched);
        #[cfg(feature = "dsv4-diagnostics")]
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            ordered.fp4_sidecar.as_mut().unwrap().enable();
            batched.fp4_sidecar.as_mut().unwrap().enable();
        }
        let total_rows = start_position + row_count;
        let kv_values = (0..total_rows * width)
            .map(|index| ((index * 17 + index / 11 + 3) % 251) as f32 * 0.0017 - 0.19)
            .collect::<Vec<_>>();
        let score_values = (0..total_rows * width)
            .map(|index| ((index * 29 + index / 7 + 5) % 239) as f32 * 0.0013 - 0.16)
            .collect::<Vec<_>>();
        let ape_values = (0..ratio * width)
            .map(|index| ((index * 13 + 7) % 127) as f32 * 0.0009 - 0.05)
            .collect::<Vec<_>>();
        let norm_values = (0..head_dim)
            .map(|index| 0.63 + (index % 19) as f32 * 0.021)
            .collect::<Vec<_>>();
        let kv = offset_f32(&ctx, &kv_values, vec![width as u64, total_rows as u64]);
        let score = offset_f32(&ctx, &score_values, vec![width as u64, total_rows as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![width as u64, ratio as u64]);
        let norm = offset_f32(&ctx, &norm_values, vec![head_dim as u64]);
        let publication_rows = ((start_position % ratio + row_count) / ratio).max(1);
        let pooled = MetalTensor::zeros_f32(&ctx, vec![512, publication_rows as u64]).unwrap();
        let normalized = MetalTensor::zeros_f32(&ctx, vec![512, publication_rows as u64]).unwrap();
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 10_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in 0..start_position {
            let kv_row = kv.view_subrange((position * width) as u64, vec![width as u64]);
            let score_row = score.view_subrange((position * width) as u64, vec![width as u64]);
            for frontier in [&ordered, &batched] {
                frontier
                    .encode_projected(
                        &ctx,
                        &encoder,
                        &kv_row,
                        &score_row,
                        &ape,
                        &norm,
                        position as u32,
                        rope,
                        1e-5,
                    )
                    .unwrap();
            }
        }
        for row in 0..row_count {
            let position = start_position + row;
            let kv_row = kv.view_subrange((position * width) as u64, vec![width as u64]);
            let score_row = score.view_subrange((position * width) as u64, vec![width as u64]);
            ordered
                .encode_projected(
                    &ctx,
                    &encoder,
                    &kv_row,
                    &score_row,
                    &ape,
                    &norm,
                    position as u32,
                    rope,
                    1e-5,
                )
                .unwrap();
        }
        let kv_chunk = kv.view_subrange(
            (start_position * width) as u64,
            vec![width as u64, row_count as u64],
        );
        let score_chunk = score.view_subrange(
            (start_position * width) as u64,
            vec![width as u64, row_count as u64],
        );
        batched
            .encode_projected_chunk(
                &ctx,
                &encoder,
                &kv_chunk,
                &score_chunk,
                &ape,
                &norm,
                &pooled,
                &normalized,
                start_position as u32,
                row_count,
                rope,
                1e-5,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());

        let label = format!("ratio={ratio} head={head_dim} start={start_position}");
        assert_bits(
            &format!("{label} KV state"),
            &read_f32(&batched.kv_state),
            &read_f32(&ordered.kv_state),
        );
        assert_bits(
            &format!("{label} score state"),
            &read_f32(&batched.score_state),
            &read_f32(&ordered.score_state),
        );
        assert_bits(
            &format!("{label} publication"),
            &read_f16(&batched.published),
            &read_f16(&ordered.published),
        );
        #[cfg(feature = "dsv4-diagnostics")]
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            let ordered = ordered.fp4_sidecar.as_ref().unwrap();
            let batched = batched.fp4_sidecar.as_ref().unwrap();
            assert_eq!(read_u8(&batched.values), read_u8(&ordered.values));
            assert_eq!(read_u8(&batched.scales), read_u8(&ordered.scales));
            assert_eq!(read_i32(&batched.status), read_i32(&ordered.status));
        }
    };

    run(4, 512, DeepSeekV4CompressorPublication::Attention, 0, 128);
    run(
        4,
        512,
        DeepSeekV4CompressorPublication::Attention,
        0,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    );
    run(4, 512, DeepSeekV4CompressorPublication::Attention, 1, 6);
    run(
        4,
        128,
        DeepSeekV4CompressorPublication::IndexerHadamard,
        3,
        125,
    );
    run(
        4,
        128,
        DeepSeekV4CompressorPublication::IndexerHadamard,
        3,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS - 3,
    );
    run(
        128,
        512,
        DeepSeekV4CompressorPublication::Attention,
        127,
        128,
    );
    run(
        128,
        512,
        DeepSeekV4CompressorPublication::Attention,
        127,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    );
    run(128, 512, DeepSeekV4CompressorPublication::Attention, 1, 12);
}

#[test]
fn ratio4_frontier_publishes_rolls_and_continues_at_second_boundary() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HIDDEN: usize = 3;
    const HEAD_DIM: usize = 4;
    const WIDTH: usize = HEAD_DIM * 2;
    let rms_eps = 1.0e-5;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        4,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::Attention,
        DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
    )
    .unwrap();
    let kv_weights = (0..HIDDEN * WIDTH)
        .map(|index| ((index * 7 + 3) % 19) as f32 * 0.041 - 0.37)
        .collect::<Vec<_>>();
    let score_weights = (0..HIDDEN * WIDTH)
        .map(|index| ((index * 11 + 5) % 23) as f32 * 0.033 - 0.31)
        .collect::<Vec<_>>();
    let ape_values = (0..4 * WIDTH)
        .map(|index| ((index * 5 + 2) % 17) as f32 * 0.027 - 0.19)
        .collect::<Vec<_>>();
    let norm_values = [0.71, 1.13, 0.58, 0.92];
    let input_values = (0..8)
        .map(|position| {
            (0..HIDDEN)
                .map(|dimension| {
                    (position as f32 - 2.4) * 0.17
                        + (dimension as f32 - 0.8) * 0.23
                        + if (position + dimension).is_multiple_of(2) {
                            0.11
                        } else {
                            -0.07
                        }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
    let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
    let inputs = input_values
        .iter()
        .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
        .collect::<Vec<_>>();
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: 2,
        theta: 10_000.0,
        scaling_factor: 1.0,
        original_context_length: 0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::local(2, rope.theta);
    let mut oracle = CompressorState::new(4, HEAD_DIM).unwrap();
    let mut expected_rows = Vec::new();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (position, (input, values)) in inputs.iter().zip(&input_values).enumerate() {
        let projected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, WIDTH, values).unwrap();
        if let Some(row) = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
        {
            expected_rows.extend(row.value);
        }
        frontier
            .encode(
                &ctx,
                &encoder,
                input,
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );

    assert_eq!(frontier.published_count(7), 2);
    assert_close(
        "ratio-4 rolled KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "ratio-4 rolled score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
    let expected_rows = expected_rows
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-4 published rows",
        &published[..2 * HEAD_DIM],
        &expected_rows,
        1e-3,
    );
    assert!(
        published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0),
        "unpublished rows must remain zero"
    );
}

#[test]
fn ratio4_indexer_publication_matches_the_integrated_oracle() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HIDDEN: usize = 3;
    const HEAD_DIM: usize = 128;
    const WIDTH: usize = HEAD_DIM * 2;
    let rms_eps = 1.0e-5;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        4,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::IndexerHadamard,
        DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
    )
    .unwrap();
    #[cfg(feature = "dsv4-diagnostics")]
    let mut frontier = frontier;
    #[cfg(not(feature = "dsv4-diagnostics"))]
    let frontier = frontier;
    #[cfg(feature = "dsv4-diagnostics")]
    frontier
        .fp4_sidecar
        .as_mut()
        .expect("indexer frontier sidecar")
        .enable();
    let kv_weights = (0..HIDDEN * WIDTH)
        .map(|index| ((index * 13 + index / 5 + 3) % 47) as f32 * 0.011 - 0.24)
        .collect::<Vec<_>>();
    let score_weights = (0..HIDDEN * WIDTH)
        .map(|index| ((index * 17 + index / 7 + 1) % 53) as f32 * 0.009 - 0.21)
        .collect::<Vec<_>>();
    let ape_values = (0..4 * WIDTH)
        .map(|index| ((index * 19 + index / 11 + 4) % 59) as f32 * 0.007 - 0.18)
        .collect::<Vec<_>>();
    let norm_values = (0..HEAD_DIM)
        .map(|index| 0.53 + (index % 23) as f32 * 0.027)
        .collect::<Vec<_>>();
    let input_values = (0..8)
        .map(|position| {
            (0..HIDDEN)
                .map(|dimension| {
                    (position as f32 - 3.1) * 0.13
                        + (dimension as f32 - 0.9) * 0.19
                        + if (position + dimension).is_multiple_of(3) {
                            0.08
                        } else {
                            -0.04
                        }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
    let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
    let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
    let inputs = input_values
        .iter()
        .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
        .collect::<Vec<_>>();
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: 64,
        theta: 160_000.0,
        scaling_factor: 1.0,
        original_context_length: 0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::local(rope.rotary_dim, rope.theta);
    let mut oracle = CompressorState::new(4, HEAD_DIM).unwrap();
    let mut expected_rows = Vec::new();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (position, (input, values)) in inputs.iter().zip(&input_values).enumerate() {
        let projected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, WIDTH, values).unwrap();
        if let Some(mut row) = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
        {
            hadamard_128_in_place(&mut row.value).unwrap();
            expected_rows.extend(row.value);
        }
        frontier
            .encode(
                &ctx,
                &encoder,
                input,
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );

    #[cfg(feature = "dsv4-diagnostics")]
    {
        let expected_packed = expected_rows
            .chunks_exact(HEAD_DIM)
            .map(|row| pack_indexer_fp4_row(row).unwrap())
            .collect::<Vec<_>>();
        let sidecar = frontier.fp4_sidecar.as_ref().unwrap();
        let actual_values = read_u8(&sidecar.values);
        let actual_scales = read_u8(&sidecar.scales);
        for (row, expected) in expected_packed.iter().enumerate() {
            assert_eq!(
                &actual_values[row * INDEXER_FP4_VALUE_BYTES..(row + 1) * INDEXER_FP4_VALUE_BYTES],
                &expected.as_bytes()[..INDEXER_FP4_VALUE_BYTES]
            );
            assert_eq!(
                &actual_scales[row * INDEXER_FP4_SCALE_BYTES..(row + 1) * INDEXER_FP4_SCALE_BYTES],
                &expected.as_bytes()[INDEXER_FP4_VALUE_BYTES..]
            );
        }
        let status = read_i32(&sidecar.status);
        assert_eq!(&status[..expected_packed.len()], &[0, 0]);
        assert!(
            status[expected_packed.len()..]
                .iter()
                .all(|&value| value == DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE)
        );
    }

    let expected_rows = expected_rows
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "integrated ratio-4 indexer publication",
        &published[..2 * HEAD_DIM],
        &expected_rows,
        2e-3,
    );
    assert!(
        published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0),
        "unpublished index rows must remain zero"
    );
}

#[test]
fn ratio128_attention_publications_match_the_integrated_oracle() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HIDDEN: usize = 3;
    const HEAD_DIM: usize = 512;
    const RATIO: usize = 128;
    let rms_eps = 1.0e-5;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        RATIO,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::Attention,
        if RATIO == 4 {
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS
        } else {
            DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS
        },
    )
    .unwrap();
    let kv_weights = (0..HIDDEN * HEAD_DIM)
        .map(|index| ((index * 13 + index / 7 + 5) % 61) as f32 * 0.008 - 0.23)
        .collect::<Vec<_>>();
    let score_weights = (0..HIDDEN * HEAD_DIM)
        .map(|index| ((index * 17 + index / 11 + 3) % 67) as f32 * 0.007 - 0.21)
        .collect::<Vec<_>>();
    let ape_values = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 19 + index / 13 + 1) % 71) as f32 * 0.006 - 0.19)
        .collect::<Vec<_>>();
    let norm_values = (0..HEAD_DIM)
        .map(|index| 0.49 + (index % 29) as f32 * 0.021)
        .collect::<Vec<_>>();
    let input_values = (0..4 * RATIO)
        .map(|position| {
            (0..HIDDEN)
                .map(|dimension| {
                    (position as f32 - 61.0) * 0.004
                        + (dimension as f32 - 0.7) * 0.17
                        + if (position + dimension).is_multiple_of(5) {
                            0.06
                        } else {
                            -0.03
                        }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, HEAD_DIM as u64]);
    let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, HEAD_DIM as u64]);
    let ape = offset_f32(&ctx, &ape_values, vec![HEAD_DIM as u64, RATIO as u64]);
    let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
    let inputs = input_values
        .iter()
        .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
        .collect::<Vec<_>>();
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: 64,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::yarn(
        rope.rotary_dim,
        rope.theta,
        rope.scaling_factor,
        rope.original_context_length,
        rope.beta_fast,
        rope.beta_slow,
    );
    let mut oracle = CompressorState::new(RATIO, HEAD_DIM).unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for position in 0..RATIO - 1 {
        let values = &input_values[position];
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let emitted = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap();
        assert!(emitted.is_none());
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "pre-boundary command failed: {:?}",
        command.error()
    );
    assert_eq!(frontier.published_count(126), 0);
    assert!(
        read_f16(&frontier.published)
            .iter()
            .all(|value| *value == 0.0)
    );

    let position = RATIO - 1;
    let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let projected_scores =
        mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let expected = oracle
        .push_projected(
            position as u32,
            &projected_kv,
            &projected_scores,
            &ape_values,
            &norm_values,
            rms_eps,
            oracle_rope,
        )
        .unwrap()
        .expect("position 127 must publish");
    assert_eq!(expected.start_position, 0);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode(
            &ctx,
            &encoder,
            &inputs[position],
            &kv_weight,
            &score_weight,
            &ape,
            &norm,
            position as u32,
            HIDDEN,
            rope,
            rms_eps,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "boundary command failed: {:?}",
        command.error()
    );

    assert_eq!(frontier.published_count(127), 1);
    assert_close(
        "ratio-128 KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "ratio-128 score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
    let expected_first = expected
        .value
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 published row 0",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert!(published[HEAD_DIM..].iter().all(|value| *value == 0.0));

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for position in RATIO..2 * RATIO - 1 {
        let values = &input_values[position];
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let emitted = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap();
        assert!(emitted.is_none());
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "second pre-boundary command failed: {:?}",
        command.error()
    );
    assert_eq!(frontier.published_count(254), 1);
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 retained row 0",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert!(published[HEAD_DIM..].iter().all(|value| *value == 0.0));

    let position = 2 * RATIO - 1;
    let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let projected_scores =
        mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let expected_second = oracle
        .push_projected(
            position as u32,
            &projected_kv,
            &projected_scores,
            &ape_values,
            &norm_values,
            rms_eps,
            oracle_rope,
        )
        .unwrap()
        .expect("position 255 must publish");
    assert_eq!(expected_second.start_position, 128);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode(
            &ctx,
            &encoder,
            &inputs[position],
            &kv_weight,
            &score_weight,
            &ape,
            &norm,
            position as u32,
            HIDDEN,
            rope,
            rms_eps,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "second boundary command failed: {:?}",
        command.error()
    );

    assert_eq!(frontier.published_count(255), 2);
    assert_close(
        "ratio-128 second-boundary KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "ratio-128 second-boundary score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
    let expected_second = expected_second
        .value
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 retained row 0 after second publication",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert_close(
        "ratio-128 published row 1",
        &published[HEAD_DIM..2 * HEAD_DIM],
        &expected_second,
        1e-3,
    );
    assert!(published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0));

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for position in 2 * RATIO..3 * RATIO - 1 {
        let values = &input_values[position];
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let emitted = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap();
        assert!(emitted.is_none());
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "third pre-boundary command failed: {:?}",
        command.error()
    );
    assert_eq!(frontier.published_count(382), 2);
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 row 0 before third publication",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert_close(
        "ratio-128 row 1 before third publication",
        &published[HEAD_DIM..2 * HEAD_DIM],
        &expected_second,
        1e-3,
    );
    assert!(published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0));

    let position = 3 * RATIO - 1;
    let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let projected_scores =
        mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let expected_third = oracle
        .push_projected(
            position as u32,
            &projected_kv,
            &projected_scores,
            &ape_values,
            &norm_values,
            rms_eps,
            oracle_rope,
        )
        .unwrap()
        .expect("position 383 must publish");
    assert_eq!(expected_third.start_position, 256);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode(
            &ctx,
            &encoder,
            &inputs[position],
            &kv_weight,
            &score_weight,
            &ape,
            &norm,
            position as u32,
            HIDDEN,
            rope,
            rms_eps,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "third boundary command failed: {:?}",
        command.error()
    );

    assert_eq!(frontier.published_count(383), 3);
    assert_close(
        "ratio-128 third-boundary KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "ratio-128 third-boundary score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
    let expected_third = expected_third
        .value
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 retained row 0 after third publication",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert_close(
        "ratio-128 retained row 1 after third publication",
        &published[HEAD_DIM..2 * HEAD_DIM],
        &expected_second,
        1e-3,
    );
    assert_close(
        "ratio-128 published row 2",
        &published[2 * HEAD_DIM..3 * HEAD_DIM],
        &expected_third,
        1e-3,
    );
    assert!(published[3 * HEAD_DIM..].iter().all(|value| *value == 0.0));

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for position in 3 * RATIO..4 * RATIO - 1 {
        let values = &input_values[position];
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
        let emitted = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap();
        assert!(emitted.is_none());
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "fourth pre-boundary command failed: {:?}",
        command.error()
    );
    assert_eq!(frontier.published_count(510), 3);
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 row 0 before fourth publication",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert_close(
        "ratio-128 row 1 before fourth publication",
        &published[HEAD_DIM..2 * HEAD_DIM],
        &expected_second,
        1e-3,
    );
    assert_close(
        "ratio-128 row 2 before fourth publication",
        &published[2 * HEAD_DIM..3 * HEAD_DIM],
        &expected_third,
        1e-3,
    );
    assert!(published[3 * HEAD_DIM..].iter().all(|value| *value == 0.0));

    let position = 4 * RATIO - 1;
    let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let projected_scores =
        mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
    let expected_fourth = oracle
        .push_projected(
            position as u32,
            &projected_kv,
            &projected_scores,
            &ape_values,
            &norm_values,
            rms_eps,
            oracle_rope,
        )
        .unwrap()
        .expect("position 511 must publish");
    assert_eq!(expected_fourth.start_position, 384);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode(
            &ctx,
            &encoder,
            &inputs[position],
            &kv_weight,
            &score_weight,
            &ape,
            &norm,
            position as u32,
            HIDDEN,
            rope,
            rms_eps,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "fourth boundary command failed: {:?}",
        command.error()
    );

    assert_eq!(frontier.published_count(511), 4);
    assert_close(
        "ratio-128 fourth-boundary KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "ratio-128 fourth-boundary score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
    let expected_fourth = expected_fourth
        .value
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let published = read_f16(&frontier.published);
    assert_close(
        "ratio-128 retained row 0 after fourth publication",
        &published[..HEAD_DIM],
        &expected_first,
        1e-3,
    );
    assert_close(
        "ratio-128 retained row 1 after fourth publication",
        &published[HEAD_DIM..2 * HEAD_DIM],
        &expected_second,
        1e-3,
    );
    assert_close(
        "ratio-128 retained row 2 after fourth publication",
        &published[2 * HEAD_DIM..3 * HEAD_DIM],
        &expected_third,
        1e-3,
    );
    assert_close(
        "ratio-128 published row 3",
        &published[3 * HEAD_DIM..4 * HEAD_DIM],
        &expected_fourth,
        1e-3,
    );
    assert!(published[4 * HEAD_DIM..].iter().all(|value| *value == 0.0));
}

#[test]
fn ratio128_frontier_preserves_twenty_four_rows_and_publishes_before_attention() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 512;
    const RATIO: usize = 128;
    const PUBLISHED_ROWS: usize = 24;
    const POSITIONS: usize = PUBLISHED_ROWS * RATIO + 1;
    let rms_eps = 1.0e-5;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        RATIO,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::Attention,
        if RATIO == 4 {
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS
        } else {
            DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS
        },
    )
    .unwrap();
    let projected_kv_values = (0..POSITIONS * HEAD_DIM)
        .map(|index| {
            let position = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (position * 17 + dimension * 7 + position / 11) % 131;
            (tag as f32 - 65.0) * 0.0031
                + if (position + dimension).is_multiple_of(29) {
                    0.037
                } else {
                    -0.011
                }
        })
        .collect::<Vec<_>>();
    let projected_score_values = (0..POSITIONS * HEAD_DIM)
        .map(|index| {
            let position = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (position * 23 + dimension * 13 + position / 7) % 137;
            (tag as f32 - 68.0) * 0.0027
                + if (position * 3 + dimension).is_multiple_of(31) {
                    0.043
                } else {
                    -0.009
                }
        })
        .collect::<Vec<_>>();
    let ape_values = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 19 + index / 17 + 5) % 149) as f32 * 0.0023 - 0.17)
        .collect::<Vec<_>>();
    let norm_values = (0..HEAD_DIM)
        .map(|index| 0.47 + (index % 37) as f32 * 0.019)
        .collect::<Vec<_>>();
    let projected_kv = offset_f32(
        &ctx,
        &projected_kv_values,
        vec![HEAD_DIM as u64, POSITIONS as u64],
    );
    let projected_score = offset_f32(
        &ctx,
        &projected_score_values,
        vec![HEAD_DIM as u64, POSITIONS as u64],
    );
    let ape = offset_f32(&ctx, &ape_values, vec![HEAD_DIM as u64, RATIO as u64]);
    let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: 64,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::yarn(
        rope.rotary_dim,
        rope.theta,
        rope.scaling_factor,
        rope.original_context_length,
        rope.beta_fast,
        rope.beta_slow,
    );
    let attention_config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let sinks = [-0.29, 0.17];
    let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
    let mut oracle = CompressorState::new(RATIO, HEAD_DIM).unwrap();
    let mut expected_rows = Vec::with_capacity(PUBLISHED_ROWS * HEAD_DIM);

    for row in 0..PUBLISHED_ROWS {
        let start = row * RATIO;
        let boundary = (row + 1) * RATIO - 1;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in start..boundary {
            let offset = position * HEAD_DIM;
            let emitted = oracle
                .push_projected(
                    position as u32,
                    &projected_kv_values[offset..offset + HEAD_DIM],
                    &projected_score_values[offset..offset + HEAD_DIM],
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap();
            assert!(emitted.is_none(), "position {position} published early");
            let kv_row = projected_kv.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
            let score_row = projected_score.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
            frontier
                .encode_projected(
                    &ctx,
                    &encoder,
                    &kv_row,
                    &score_row,
                    &ape,
                    &norm,
                    position as u32,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "row {row} pre-boundary command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count((boundary - 1) as u32), row);
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-128 rows retained before publication {row}"),
            &published[..expected_rows.len()],
            &expected_rows,
            1e-3,
        );
        assert!(
            published[expected_rows.len()..]
                .iter()
                .all(|value| *value == 0.0),
            "row {row} appeared before position {boundary}"
        );

        let offset = boundary * HEAD_DIM;
        let emitted = oracle
            .push_projected(
                boundary as u32,
                &projected_kv_values[offset..offset + HEAD_DIM],
                &projected_score_values[offset..offset + HEAD_DIM],
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .unwrap_or_else(|| panic!("position {boundary} did not publish row {row}"));
        assert_eq!(emitted.start_position as usize, row * RATIO);
        let expected_row = emitted
            .value
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        expected_rows.extend_from_slice(&expected_row);

        let integration = if matches!(boundary, 639 | 1023 | 2047 | 2175 | 3071) {
            let raw_start = boundary + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
            let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
            let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
            let mut raw_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
            for logical_position in raw_start..=boundary {
                let raw_row = (0..HEAD_DIM)
                    .map(|dimension| {
                        let tag =
                            (logical_position * 31 + dimension * 11 + logical_position / 9) % 139;
                        round_f16(
                            (tag as f32 - 69.0) * 0.0021
                                + if (logical_position + dimension).is_multiple_of(23) {
                                    0.039
                                } else {
                                    -0.007
                                },
                        )
                    })
                    .collect::<Vec<_>>();
                raw_rows.extend_from_slice(&raw_row);
                let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
                raw_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(&raw_row);
            }
            let queries = (0..HEADS * HEAD_DIM)
                .map(|index| {
                    let head = index / HEAD_DIM;
                    let dimension = index % HEAD_DIM;
                    let tag = (boundary * 13 + head * 17 + dimension * 5) % 103;
                    (tag as f32 - 51.0) * 0.0029
                })
                .collect::<Vec<_>>();
            let expected = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &expected_rows,
                None,
                &sinks,
            )
            .unwrap();
            let prior = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &expected_rows[..expected_rows.len() - HEAD_DIM],
                None,
                &sinks,
            )
            .unwrap();
            assert!(
                expected
                    .iter()
                    .zip(&prior)
                    .any(|(current, prior)| (current - prior).abs() > 1e-3),
                "position {boundary} must distinguish the newest HCA row"
            );
            Some((
                offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, HEADS as u64]),
                offset_f32(
                    &ctx,
                    &raw_ring,
                    vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                ),
                MetalTensor::zeros_f16(
                    &ctx,
                    vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                )
                .unwrap(),
                offset_f32(
                    &ctx,
                    &vec![0.0; HEADS * HEAD_DIM],
                    vec![HEAD_DIM as u64, HEADS as u64],
                ),
                expected,
            ))
        } else {
            None
        };

        let kv_row = projected_kv.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
        let score_row = projected_score.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode_projected(
                &ctx,
                &encoder,
                &kv_row,
                &score_row,
                &ape,
                &norm,
                boundary as u32,
                rope,
                rms_eps,
            )
            .unwrap();
        if let Some((queries, raw_source, raw_cache, output, _)) = &integration {
            encode_scatter_offset_f32_to_f16(
                &ctx,
                &encoder,
                raw_source,
                raw_cache,
                0,
                DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM,
            )
            .unwrap();
            encode_dense_sink_attention_f16(
                &ctx,
                &encoder,
                queries,
                raw_cache,
                Some(DeepSeekV4PublishedRows {
                    cache: &frontier.published,
                    count: row + 1,
                    capacity_rows: frontier.capacity_rows,
                }),
                &sink_tensor,
                output,
                boundary as u32,
                attention_config,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "row {row} publication command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(boundary as u32), row + 1);
        assert_close(
            &format!("ratio-128 row {row} boundary KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-128 row {row} boundary score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-128 rows through {row}"),
            &published[..expected_rows.len()],
            &expected_rows,
            1e-3,
        );
        assert!(
            published[expected_rows.len()..]
                .iter()
                .all(|value| *value == 0.0),
            "ratio-128 slab tail changed after row {row}"
        );
        if let Some((_, _, _, output, expected)) = &integration {
            assert_close(
                &format!("position {boundary} serial HCA publication visibility"),
                &read_f32(output),
                expected,
                8e-5,
            );
        }
    }

    let position = PUBLISHED_ROWS * RATIO;
    let offset = position * HEAD_DIM;
    assert!(
        oracle
            .push_projected(
                position as u32,
                &projected_kv_values[offset..offset + HEAD_DIM],
                &projected_score_values[offset..offset + HEAD_DIM],
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .is_none()
    );
    let before_continuation = read_f16(&frontier.published);
    let kv_row = projected_kv.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
    let score_row = projected_score.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    frontier
        .encode_projected(
            &ctx,
            &encoder,
            &kv_row,
            &score_row,
            &ape,
            &norm,
            position as u32,
            rope,
            rms_eps,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "position {position} continuation command failed: {:?}",
        command.error()
    );
    assert_eq!(frontier.published_count(position as u32), PUBLISHED_ROWS);
    assert_eq!(read_f16(&frontier.published), before_continuation);
    assert_close(
        &format!("position {position} HCA KV state"),
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        &format!("position {position} HCA score state"),
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
}

#[test]
fn ratio4_frontiers_enter_third_slab_and_reject_row_768() {
    let Some(ctx) = metal_context() else {
        return;
    };

    fn run_case(ctx: &MetalContext, head_dim: usize, publication: DeepSeekV4CompressorPublication) {
        const RATIO: usize = 4;
        const POSITIONS: usize = 3_073;
        let width = 2 * head_dim;
        let rms_eps = 1.0e-5;
        let label = match publication {
            DeepSeekV4CompressorPublication::Attention => "attention",
            DeepSeekV4CompressorPublication::IndexerHadamard => "indexer",
        };
        let frontier = DeepSeekV4CompressorFrontier::new(
            ctx,
            RATIO,
            head_dim,
            publication,
            if RATIO == 4 {
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS
            } else {
                DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS
            },
        )
        .unwrap();
        let projected_kv_values = (0..POSITIONS * width)
            .map(|index| {
                let position = index / width;
                let dimension = index % width;
                let tag = (position * 17 + dimension * 7 + position / 13) % 127;
                (tag as f32 - 63.0) * 0.0027
                    + if (position + dimension).is_multiple_of(31) {
                        0.041
                    } else {
                        -0.009
                    }
            })
            .collect::<Vec<_>>();
        let projected_score_values = (0..POSITIONS * width)
            .map(|index| {
                let position = index / width;
                let dimension = index % width;
                let tag = (position * 23 + dimension * 11 + position / 5) % 131;
                (tag as f32 - 65.0) * 0.0023
                    + if (position * 3 + dimension).is_multiple_of(29) {
                        0.037
                    } else {
                        -0.007
                    }
            })
            .collect::<Vec<_>>();
        let ape_values = (0..RATIO * width)
            .map(|index| ((index * 19 + index / 7 + 3) % 137) as f32 * 0.0021 - 0.14)
            .collect::<Vec<_>>();
        let norm_values = (0..head_dim)
            .map(|index| 0.51 + (index % 31) as f32 * 0.017)
            .collect::<Vec<_>>();
        let projected_kv = offset_f32(
            ctx,
            &projected_kv_values,
            vec![width as u64, POSITIONS as u64],
        );
        let projected_score = offset_f32(
            ctx,
            &projected_score_values,
            vec![width as u64, POSITIONS as u64],
        );
        let ape = offset_f32(ctx, &ape_values, vec![width as u64, RATIO as u64]);
        let norm = offset_f32(ctx, &norm_values, vec![head_dim as u64]);
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 160_000.0,
            scaling_factor: 16.0,
            original_context_length: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::yarn(
            rope.rotary_dim,
            rope.theta,
            rope.scaling_factor,
            rope.original_context_length,
            rope.beta_fast,
            rope.beta_slow,
        );
        let mut oracle = CompressorState::new(RATIO, head_dim).unwrap();
        let mut expected_rows =
            Vec::with_capacity(DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * head_dim);

        for chunk_start in (0..2_047).step_by(64) {
            let chunk_end = (chunk_start + 64).min(2_047);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for position in chunk_start..chunk_end {
                let offset = position * width;
                if let Some(mut emitted) = oracle
                    .push_projected(
                        position as u32,
                        &projected_kv_values[offset..offset + width],
                        &projected_score_values[offset..offset + width],
                        &ape_values,
                        &norm_values,
                        rms_eps,
                        oracle_rope,
                    )
                    .unwrap()
                {
                    if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
                        hadamard_128_in_place(&mut emitted.value).unwrap();
                    }
                    expected_rows.extend(
                        emitted
                            .value
                            .into_iter()
                            .map(|value| half::f16::from_f32(value).to_f32()),
                    );
                }
                let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
                let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
                frontier
                    .encode_projected(
                        ctx,
                        &encoder,
                        &kv_row,
                        &score_row,
                        &ape,
                        &norm,
                        position as u32,
                        rope,
                        rms_eps,
                    )
                    .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "ratio-4 {label} chunk {chunk_start} command failed: {:?}",
                command.error()
            );
        }

        assert_eq!(frontier.published_count(2_046), 511);
        assert_eq!(expected_rows.len(), 511 * head_dim);
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-4 {label} rows before second-slab end"),
            &published[..expected_rows.len()],
            &expected_rows,
            2e-3,
        );
        assert!(
            published[expected_rows.len()..]
                .iter()
                .all(|value| *value == 0.0),
            "ratio-4 {label} row 511 published before position 2047"
        );
        assert_close(
            &format!("ratio-4 {label} pre-second-slab-end KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} pre-second-slab-end score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        let position = 2_047usize;
        let offset = position * width;
        let mut emitted = oracle
            .push_projected(
                position as u32,
                &projected_kv_values[offset..offset + width],
                &projected_score_values[offset..offset + width],
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .expect("position 2047 must publish row 511");
        assert_eq!(emitted.start_position, 2_044);
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            hadamard_128_in_place(&mut emitted.value).unwrap();
        }
        expected_rows.extend(
            emitted
                .value
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32()),
        );
        let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
        let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode_projected(
                ctx,
                &encoder,
                &kv_row,
                &score_row,
                &ape,
                &norm,
                position as u32,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "ratio-4 {label} second-slab-end command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(position as u32), 512);
        assert_eq!(expected_rows.len(), 512 * head_dim);
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-4 {label} complete second slab"),
            &published[..expected_rows.len()],
            &expected_rows,
            2e-3,
        );
        assert!(
            published[expected_rows.len()..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_close(
            &format!("ratio-4 {label} second-slab-end KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} second-slab-end score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        let position = 2_048usize;
        let offset = position * width;
        assert!(
            oracle
                .push_projected(
                    position as u32,
                    &projected_kv_values[offset..offset + width],
                    &projected_score_values[offset..offset + width],
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
                .is_none()
        );
        let before_continuation = read_f16(&frontier.published);
        let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
        let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode_projected(
                ctx,
                &encoder,
                &kv_row,
                &score_row,
                &ape,
                &norm,
                position as u32,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "ratio-4 {label} position-2048 command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(position as u32), 512);
        assert_eq!(read_f16(&frontier.published), before_continuation);
        assert_close(
            &format!("ratio-4 {label} position-2048 KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} position-2048 score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in 2_049usize..=2_051 {
            let offset = position * width;
            if let Some(mut emitted) = oracle
                .push_projected(
                    position as u32,
                    &projected_kv_values[offset..offset + width],
                    &projected_score_values[offset..offset + width],
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
            {
                assert_eq!(position, 2_051);
                assert_eq!(emitted.start_position, 2_048);
                if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
                    hadamard_128_in_place(&mut emitted.value).unwrap();
                }
                expected_rows.extend(
                    emitted
                        .value
                        .into_iter()
                        .map(|value| half::f16::from_f32(value).to_f32()),
                );
            }
            let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
            let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
            frontier
                .encode_projected(
                    ctx,
                    &encoder,
                    &kv_row,
                    &score_row,
                    &ape,
                    &norm,
                    position as u32,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "ratio-4 {label} third-slab entry command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(2_051), 513);
        assert_eq!(expected_rows.len(), 513 * head_dim);
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-4 {label} first third-slab row"),
            &published[..expected_rows.len()],
            &expected_rows,
            2e-3,
        );
        assert!(
            published[expected_rows.len()..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_close(
            &format!("ratio-4 {label} third-slab-entry KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} third-slab-entry score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        for chunk_start in (2_052usize..3_072).step_by(64) {
            let chunk_end = (chunk_start + 64).min(3_072);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for position in chunk_start..chunk_end {
                let offset = position * width;
                if let Some(mut emitted) = oracle
                    .push_projected(
                        position as u32,
                        &projected_kv_values[offset..offset + width],
                        &projected_score_values[offset..offset + width],
                        &ape_values,
                        &norm_values,
                        rms_eps,
                        oracle_rope,
                    )
                    .unwrap()
                {
                    if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
                        hadamard_128_in_place(&mut emitted.value).unwrap();
                    }
                    expected_rows.extend(
                        emitted
                            .value
                            .into_iter()
                            .map(|value| half::f16::from_f32(value).to_f32()),
                    );
                }
                let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
                let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
                frontier
                    .encode_projected(
                        ctx,
                        &encoder,
                        &kv_row,
                        &score_row,
                        &ape,
                        &norm,
                        position as u32,
                        rope,
                        rms_eps,
                    )
                    .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "ratio-4 {label} full-third-slab chunk {chunk_start} failed: {:?}",
                command.error()
            );
        }

        assert_eq!(frontier.published_count(3_071), 768);
        assert_eq!(expected_rows.len(), 768 * head_dim);
        let published = read_f16(&frontier.published);
        assert_close(
            &format!("ratio-4 {label} complete third slab"),
            &published,
            &expected_rows,
            2e-3,
        );
        assert_close(
            &format!("ratio-4 {label} third-slab-end KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} third-slab-end score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        let position = 3_072usize;
        let offset = position * width;
        assert!(
            oracle
                .push_projected(
                    position as u32,
                    &projected_kv_values[offset..offset + width],
                    &projected_score_values[offset..offset + width],
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
                .is_none()
        );
        let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
        let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode_projected(
                ctx,
                &encoder,
                &kv_row,
                &score_row,
                &ape,
                &norm,
                position as u32,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "ratio-4 {label} position-3072 command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(position as u32), 768);
        assert_eq!(read_f16(&frontier.published), published);
        assert_close(
            &format!("ratio-4 {label} position-3072 KV state"),
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            &format!("ratio-4 {label} position-3072 score state"),
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );

        let before_rejection_kv = read_f32(&frontier.kv_state);
        let before_rejection_score = read_f32(&frontier.score_state);
        let before_rejection_rows = published;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let offset = 2_051 * width;
        let kv_row = projected_kv.view_subrange(offset as u64, vec![width as u64]);
        let score_row = projected_score.view_subrange(offset as u64, vec![width as u64]);
        let error = frontier
            .encode_projected(
                ctx, &encoder, &kv_row, &score_row, &ape, &norm, 3_075, rope, rms_eps,
            )
            .unwrap_err();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "ratio-4 {label} rejected-row command failed: {:?}",
            command.error()
        );
        assert!(error.to_string().contains("published row 768"));
        assert_eq!(read_f32(&frontier.kv_state), before_rejection_kv);
        assert_eq!(read_f32(&frontier.score_state), before_rejection_score);
        assert_eq!(read_f16(&frontier.published), before_rejection_rows);
    }

    run_case(&ctx, 512, DeepSeekV4CompressorPublication::Attention);
    run_case(&ctx, 128, DeepSeekV4CompressorPublication::IndexerHadamard);
}

#[test]
fn ratio4_frontiers_publish_the_final_model_context_row_without_prefix_replay() {
    let Some(ctx) = metal_context() else {
        return;
    };

    fn run_case(ctx: &MetalContext, head_dim: usize, publication: DeepSeekV4CompressorPublication) {
        const RATIO: usize = 4;
        const START_POSITION: u32 = 1_048_568;
        const CAPACITY_ROWS: usize = 262_144;
        let width = 2 * head_dim;
        let rms_eps = 1.0e-5;
        let frontier =
            DeepSeekV4CompressorFrontier::new(ctx, RATIO, head_dim, publication, CAPACITY_ROWS)
                .unwrap();
        let lower_kv = (0..RATIO * width)
            .map(|index| ((index * 17 + 5) % 131) as f32 * 0.0017 - 0.09)
            .collect::<Vec<_>>();
        let lower_scores = (0..RATIO * width)
            .map(|index| ((index * 23 + 7) % 137) as f32 * 0.0019 - 0.11)
            .collect::<Vec<_>>();
        let mut initial_kv = lower_kv.clone();
        initial_kv.extend_from_slice(&lower_kv);
        let mut initial_scores = lower_scores.clone();
        initial_scores.extend_from_slice(&lower_scores);
        host_write_f32(&frontier.kv_state, &initial_kv, "far CSA KV state").unwrap();
        host_write_f32(
            &frontier.score_state,
            &initial_scores,
            "far CSA score state",
        )
        .unwrap();
        let mut oracle = CompressorState::from_snapshot(
            RATIO,
            head_dim,
            u64::from(START_POSITION),
            initial_kv,
            initial_scores,
        )
        .unwrap();
        let projected_kv_values = (0..8 * width)
            .map(|index| {
                let row = index / width;
                let dimension = index % width;
                ((row * 29 + dimension * 11 + 3) % 149) as f32 * 0.0013 - 0.08
            })
            .collect::<Vec<_>>();
        let projected_score_values = (0..8 * width)
            .map(|index| {
                let row = index / width;
                let dimension = index % width;
                ((row * 31 + dimension * 13 + 9) % 151) as f32 * 0.0011 - 0.07
            })
            .collect::<Vec<_>>();
        let ape_values = (0..RATIO * width)
            .map(|index| ((index * 19 + 1) % 127) as f32 * 0.0015 - 0.1)
            .collect::<Vec<_>>();
        let norm_values = (0..head_dim)
            .map(|index| 0.63 + (index % 37) as f32 * 0.009)
            .collect::<Vec<_>>();
        let projected_kv = offset_f32(ctx, &projected_kv_values, vec![width as u64, 8]);
        let projected_score = offset_f32(ctx, &projected_score_values, vec![width as u64, 8]);
        let ape = offset_f32(ctx, &ape_values, vec![width as u64, RATIO as u64]);
        let norm = offset_f32(ctx, &norm_values, vec![head_dim as u64]);
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 160_000.0,
            scaling_factor: 16.0,
            original_context_length: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::yarn(
            rope.rotary_dim,
            rope.theta,
            rope.scaling_factor,
            rope.original_context_length,
            rope.beta_fast,
            rope.beta_slow,
        );
        let mut expected_rows = Vec::new();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for local in 0..8usize {
            let position = START_POSITION + local as u32;
            let offset = local * width;
            if let Some(mut emitted) = oracle
                .push_projected(
                    position,
                    &projected_kv_values[offset..offset + width],
                    &projected_score_values[offset..offset + width],
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
            {
                if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
                    hadamard_128_in_place(&mut emitted.value).unwrap();
                }
                expected_rows.push((
                    (position as usize + 1) / RATIO - 1,
                    emitted
                        .value
                        .into_iter()
                        .map(|value| half::f16::from_f32(value).to_f32())
                        .collect::<Vec<_>>(),
                ));
            }
            let kv = projected_kv.view_subrange(offset as u64, vec![width as u64]);
            let score = projected_score.view_subrange(offset as u64, vec![width as u64]);
            frontier
                .encode_projected(
                    ctx, &encoder, &kv, &score, &ape, &norm, position, rope, rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        assert_eq!(
            expected_rows
                .iter()
                .map(|(row, _)| *row)
                .collect::<Vec<_>>(),
            [262_142, 262_143]
        );
        assert_eq!(frontier.published_count(1_048_575), CAPACITY_ROWS);
        for (row, expected) in expected_rows {
            let published = frontier
                .published
                .view_subrange((row * head_dim) as u64, vec![head_dim as u64]);
            let actual = read_f16(&published);
            let mut dot = 0.0_f64;
            let mut actual_norm = 0.0_f64;
            let mut expected_norm = 0.0_f64;
            let mut squared_error = 0.0_f64;
            let mut max_error = 0.0_f32;
            for (&actual, &expected) in actual.iter().zip(&expected) {
                dot += f64::from(actual) * f64::from(expected);
                actual_norm += f64::from(actual).powi(2);
                expected_norm += f64::from(expected).powi(2);
                squared_error += f64::from(actual - expected).powi(2);
                max_error = max_error.max((actual - expected).abs());
            }
            let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
            let relative_rms = (squared_error / expected_norm).sqrt();
            eprintln!(
                "far CSA row {row} cosine={cosine:.9} rel_rms={relative_rms:.9} max_abs={max_error}"
            );
            assert!(cosine >= 0.999_9);
            assert!(relative_rms <= 0.015);
            assert!(max_error <= 0.05);
        }
        let first = frontier.published.view_subrange(0, vec![head_dim as u64]);
        assert!(read_f16(&first).iter().all(|&value| value == 0.0));
        assert_close(
            "far CSA KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "far CSA score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
    }

    run_case(&ctx, 512, DeepSeekV4CompressorPublication::Attention);
    run_case(&ctx, 128, DeepSeekV4CompressorPublication::IndexerHadamard);
}

#[test]
fn ratio128_frontier_publishes_the_final_model_context_row_without_prefix_replay() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const RATIO: usize = 128;
    const HEAD_DIM: usize = 512;
    const START_POSITION: u32 = 1_048_448;
    const CAPACITY_ROWS: usize = 8_192;
    let rms_eps = 1.0e-5;
    let frontier = DeepSeekV4CompressorFrontier::new(
        &ctx,
        RATIO,
        HEAD_DIM,
        DeepSeekV4CompressorPublication::Attention,
        CAPACITY_ROWS,
    )
    .unwrap();
    let initial_kv = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 17 + 3) % 139) as f32 * 0.0013 - 0.08)
        .collect::<Vec<_>>();
    let initial_scores = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 19 + 7) % 149) as f32 * 0.0011 - 0.07)
        .collect::<Vec<_>>();
    host_write_f32(&frontier.kv_state, &initial_kv, "far HCA KV state").unwrap();
    host_write_f32(
        &frontier.score_state,
        &initial_scores,
        "far HCA score state",
    )
    .unwrap();
    let mut oracle = CompressorState::from_snapshot(
        RATIO,
        HEAD_DIM,
        u64::from(START_POSITION),
        initial_kv,
        initial_scores,
    )
    .unwrap();
    let projected_kv_values = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 23 + 5) % 151) as f32 * 0.0012 - 0.075)
        .collect::<Vec<_>>();
    let projected_score_values = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 29 + 11) % 157) as f32 * 0.001 - 0.068)
        .collect::<Vec<_>>();
    let ape_values = (0..RATIO * HEAD_DIM)
        .map(|index| ((index * 31 + 13) % 163) as f32 * 0.0009 - 0.073)
        .collect::<Vec<_>>();
    let norm_values = (0..HEAD_DIM)
        .map(|index| 0.59 + (index % 41) as f32 * 0.008)
        .collect::<Vec<_>>();
    let projected_kv = offset_f32(
        &ctx,
        &projected_kv_values,
        vec![HEAD_DIM as u64, RATIO as u64],
    );
    let projected_score = offset_f32(
        &ctx,
        &projected_score_values,
        vec![HEAD_DIM as u64, RATIO as u64],
    );
    let ape = offset_f32(&ctx, &ape_values, vec![HEAD_DIM as u64, RATIO as u64]);
    let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
    let rope = DeepSeekV4RopeParameters {
        rotary_dim: 64,
        theta: 160_000.0,
        scaling_factor: 16.0,
        original_context_length: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let oracle_rope = RopeParameters::yarn(
        rope.rotary_dim,
        rope.theta,
        rope.scaling_factor,
        rope.original_context_length,
        rope.beta_fast,
        rope.beta_slow,
    );
    let mut expected = None;
    for local in 0..RATIO {
        let position = START_POSITION + local as u32;
        let offset = local * HEAD_DIM;
        let emitted = oracle
            .push_projected(
                position,
                &projected_kv_values[offset..offset + HEAD_DIM],
                &projected_score_values[offset..offset + HEAD_DIM],
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap();
        if local + 1 == RATIO {
            expected = Some(
                emitted
                    .expect("position 1048575 must publish HCA row 8191")
                    .value
                    .into_iter()
                    .map(|value| half::f16::from_f32(value).to_f32())
                    .collect::<Vec<_>>(),
            );
        } else {
            assert!(emitted.is_none());
        }
    }
    let expected = expected.unwrap();
    let attention_query_values = expected
        .iter()
        .map(|value| value * 12.0)
        .collect::<Vec<_>>();
    let raw_rows = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
    let mut compressed_rows = vec![0.0f32; CAPACITY_ROWS * HEAD_DIM];
    compressed_rows[(CAPACITY_ROWS - 1) * HEAD_DIM..].copy_from_slice(&expected);
    let sinks = [-0.21f32];
    let expected_attention = shared_kv_attention(
        &attention_query_values,
        1,
        HEAD_DIM,
        &raw_rows,
        &compressed_rows,
        None,
        &sinks,
    )
    .unwrap();
    let prior_attention = shared_kv_attention(
        &attention_query_values,
        1,
        HEAD_DIM,
        &raw_rows,
        &compressed_rows[..(CAPACITY_ROWS - 1) * HEAD_DIM],
        None,
        &sinks,
    )
    .unwrap();
    assert!(
        expected_attention
            .iter()
            .zip(&prior_attention)
            .any(|(current, prior)| (current - prior).abs() > 1e-3)
    );
    let attention_queries = offset_f32(&ctx, &attention_query_values, vec![HEAD_DIM as u64, 1]);
    let raw_cache =
        MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
            .unwrap();
    let sink_tensor = offset_f32(&ctx, &sinks, vec![1]);
    let attention_output = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, 1]).unwrap();
    let attention_config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: 1,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for local in 0..RATIO {
        let position = START_POSITION + local as u32;
        let offset = local * HEAD_DIM;
        let kv = projected_kv.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
        let score = projected_score.view_subrange(offset as u64, vec![HEAD_DIM as u64]);
        frontier
            .encode_projected(
                &ctx, &encoder, &kv, &score, &ape, &norm, position, rope, rms_eps,
            )
            .unwrap();
    }
    encode_tiled_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &attention_queries,
        &raw_cache,
        &raw_cache,
        DeepSeekV4RawCacheLayout::Ring,
        DeepSeekV4PublishedRows {
            cache: &frontier.published,
            count: CAPACITY_ROWS,
            capacity_rows: CAPACITY_ROWS,
        },
        &sink_tensor,
        &attention_output,
        1_048_575,
        0,
        1,
        128,
        attention_config,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(frontier.published_count(1_048_575), CAPACITY_ROWS);
    let published = frontier.published.view_subrange(
        ((CAPACITY_ROWS - 1) * HEAD_DIM) as u64,
        vec![HEAD_DIM as u64],
    );
    let actual_published = read_f16(&published);
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut max_error = 0.0f32;
    for (&actual, &reference) in actual_published.iter().zip(&expected) {
        dot += f64::from(actual) * f64::from(reference);
        actual_norm += f64::from(actual).powi(2);
        expected_norm += f64::from(reference).powi(2);
        squared_error += f64::from(actual - reference).powi(2);
        max_error = max_error.max((actual - reference).abs());
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    let relative_rms = (squared_error / expected_norm).sqrt();
    eprintln!("far HCA row 8191 cosine={cosine:.9} rel_rms={relative_rms:.9} max_abs={max_error}");
    assert!(cosine >= 0.999_9);
    assert!(relative_rms <= 0.015);
    assert!(max_error <= 0.05);
    compressed_rows[(CAPACITY_ROWS - 1) * HEAD_DIM..].copy_from_slice(&actual_published);
    let actual_expected_attention = shared_kv_attention(
        &attention_query_values,
        1,
        HEAD_DIM,
        &raw_rows,
        &compressed_rows,
        None,
        &sinks,
    )
    .unwrap();
    assert_close(
        "far HCA attention consumes row 8191",
        &read_f32(&attention_output),
        &actual_expected_attention,
        8e-5,
    );
    assert!(
        actual_expected_attention
            .iter()
            .zip(&prior_attention)
            .any(|(current, prior)| (current - prior).abs() > 1e-3)
    );
    let prior = frontier.published.view_subrange(
        ((CAPACITY_ROWS - 2) * HEAD_DIM) as u64,
        vec![HEAD_DIM as u64],
    );
    assert!(read_f16(&prior).iter().all(|&value| value == 0.0));
    assert_close(
        "far HCA KV state",
        &read_f32(&frontier.kv_state),
        oracle.kv_state(),
        4e-5,
    );
    assert_close(
        "far HCA score state",
        &read_f32(&frontier.score_state),
        oracle.score_state(),
        4e-5,
    );
}

#[test]
fn online_hca_geometry_fails_closed_before_encoding() {
    let config = deepseek_v4_session_attention_config();
    validate_deepseek_v4_online_hca_request_geometry(config, 128, 1, 0, 1).unwrap();
    validate_deepseek_v4_online_hca_launch_geometry(
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
    )
    .unwrap();
    validate_deepseek_v4_online_hca_launch_geometry(
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        0,
        0,
    )
    .unwrap();

    let width_error = validate_deepseek_v4_online_hca_launch_geometry(
        16,
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
    )
    .unwrap_err();
    assert!(width_error.to_string().contains("SIMD width 32"));
    let thread_error = validate_deepseek_v4_online_hca_launch_geometry(
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADS - 1,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
    )
    .unwrap_err();
    assert!(thread_error.to_string().contains("max 31"));
    let memory_error = validate_deepseek_v4_online_hca_launch_geometry(
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADS,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES - 1,
        DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES,
    )
    .unwrap_err();
    assert!(memory_error.to_string().contains("device allows 1023"));

    let non_singleton =
        validate_deepseek_v4_online_hca_request_geometry(config, 128, 2, 0, 2).unwrap_err();
    assert!(non_singleton.to_string().contains("singleton query"));
    let offset =
        validate_deepseek_v4_online_hca_request_geometry(config, 128, 2, 1, 1).unwrap_err();
    assert!(offset.to_string().contains("singleton query"));
}

#[test]
fn tiled_hca_is_bit_identical_to_legacy_through_512_rows() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 512;
    const CAPACITY: usize = 512;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: 1,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let queries = (0..HEAD_DIM)
        .map(|dimension| 0.031 - (dimension % 23) as f32 * 0.0009)
        .collect::<Vec<_>>();
    let compressed = (0..CAPACITY * HEAD_DIM)
        .map(|index| {
            let row = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (row * 37 + dimension * 11 + row / 5) % 131;
            round_f16((tag as f32 - 65.0) * 0.0017)
        })
        .collect::<Vec<_>>();
    let compressed_bits = compressed
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![HEAD_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, 1]);
    let sinks = offset_f32(&ctx, &[-0.29], vec![1]);

    for count in [1usize, 511, 512] {
        let position = count * 128 - 1;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut raw_ring = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
        for logical_position in raw_start..=position {
            let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
            for dimension in 0..HEAD_DIM {
                let tag = (logical_position * 19 + dimension * 7) % 113;
                raw_ring[slot * HEAD_DIM + dimension] = round_f16((tag as f32 - 56.0) * 0.0013);
            }
        }
        let raw_bits = raw_ring
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let raw_cache = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&raw_bits),
            vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let legacy = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, 1]).unwrap();
        let tiled = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, 1]).unwrap();
        let rows = DeepSeekV4PublishedRows {
            cache: &compressed_cache,
            count,
            capacity_rows: CAPACITY,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &raw_cache,
            Some(rows),
            &sinks,
            &legacy,
            position as u32,
            config,
        )
        .unwrap();
        encode_tiled_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            rows,
            &sinks,
            &tiled,
            position as u32,
            0,
            1,
            128,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        assert!(
            read_f32(&legacy)
                .iter()
                .zip(read_f32(&tiled))
                .all(|(legacy, tiled)| legacy.to_bits() == tiled.to_bits()),
            "tiled HCA changed the established {count}-row reduction"
        );
    }
}

#[test]
fn tiled_hca_matches_cpu_through_the_full_model_context() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 512;
    const CAPACITY: usize = 8_192;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: 1,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let queries = (0..HEAD_DIM)
        .map(|dimension| 0.024 - (dimension % 29) as f32 * 0.0007)
        .collect::<Vec<_>>();
    let mut compressed = (0..CAPACITY * HEAD_DIM)
        .map(|index| {
            let row = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (row * 41 + dimension * 13 + row / 7) % 139;
            round_f16((tag as f32 - 69.0) * 0.0011)
        })
        .collect::<Vec<_>>();
    for (strength, row) in [512usize, 526, 527, 894, 895, 896, 8_190, 8_191]
        .into_iter()
        .enumerate()
    {
        for dimension in 0..HEAD_DIM {
            compressed[row * HEAD_DIM + dimension] =
                round_f16(queries[dimension] * (512.0 + strength as f32 * 256.0));
        }
    }
    let compressed_bits = compressed
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![HEAD_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, 1]);
    let sink_values = [-0.41f32];
    let sinks = offset_f32(&ctx, &sink_values, vec![1]);

    for count in [513usize, 527, 528, 895, 896, 897, 8_191, 8_192] {
        let position = count * 128 - 1;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut raw_ring = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
        let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
        for logical_position in raw_start..=position {
            let row = (0..HEAD_DIM)
                .map(|dimension| {
                    let tag = (logical_position * 17 + dimension * 5 + logical_position / 11) % 127;
                    round_f16((tag as f32 - 63.0) * 0.0009)
                })
                .collect::<Vec<_>>();
            raw_rows.extend_from_slice(&row);
            let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
            raw_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(&row);
        }
        let expected = shared_kv_attention(
            &queries,
            1,
            HEAD_DIM,
            &raw_rows,
            &compressed[..count * HEAD_DIM],
            None,
            &sink_values,
        )
        .unwrap();
        let prior = shared_kv_attention(
            &queries,
            1,
            HEAD_DIM,
            &raw_rows,
            &compressed[..(count - 1) * HEAD_DIM],
            None,
            &sink_values,
        )
        .unwrap();
        assert!(
            expected
                .iter()
                .zip(&prior)
                .any(|(current, prior)| (current - prior).abs() > 1e-3),
            "{count}-row fixture does not expose its final HCA row"
        );
        let raw_bits = raw_ring
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let raw_cache = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&raw_bits),
            vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, 1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_tiled_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            DeepSeekV4PublishedRows {
                cache: &compressed_cache,
                count,
                capacity_rows: CAPACITY,
            },
            &sinks,
            &output,
            position as u32,
            0,
            1,
            128,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "{count}-row tiled HCA command failed: {:?}",
            command.error()
        );
        assert_close(
            &format!("{count}-row tiled HCA"),
            &read_f32(&output),
            &expected,
            8e-5,
        );
    }
}

#[test]
fn online_hca_matches_legacy_envelope_at_structural_boundaries() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const HEAD_DIM: usize = 512;
    const CAPACITY: usize = 8_192;

    fn assert_envelope(label: &str, actual: &[f32], reference: &[f32]) {
        assert_eq!(actual.len(), reference.len());
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut squared_error = 0.0f64;
        let mut max_scaled_error = 0.0f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            assert!(actual.is_finite(), "{label} produced nonfinite output");
            dot += f64::from(actual) * f64::from(reference);
            actual_norm += f64::from(actual).powi(2);
            reference_norm += f64::from(reference).powi(2);
            squared_error += f64::from(actual - reference).powi(2);
            max_scaled_error = max_scaled_error.max(f64::from(
                (actual - reference).abs() / reference.abs().max(1.0),
            ));
        }
        let relative_rms = (squared_error / reference_norm).sqrt();
        let cosine = dot / (actual_norm.sqrt() * reference_norm.sqrt());
        eprintln!(
            "deepseek_v4 {label} cosine={cosine:.9} rel_rms={relative_rms:.9} max_scaled={max_scaled_error:.9}"
        );
        assert!(max_scaled_error <= 8e-5, "{label} scaled error");
        assert!(relative_rms <= 1e-3, "{label} relative RMS");
        assert!(cosine >= 0.999_999, "{label} cosine");
    }

    let config = deepseek_v4_session_attention_config();
    assert_eq!((config.head_count, config.head_dim), (HEADS, HEAD_DIM));
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let queries = (0..HEADS * HEAD_DIM)
        .map(|index| {
            let head = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            0.024 + head as f32 * 0.00002 - (dimension % 29) as f32 * 0.0007
        })
        .collect::<Vec<_>>();
    let head_zero_query = &queries[..HEAD_DIM];
    let mut compressed = (0..CAPACITY * HEAD_DIM)
        .map(|index| {
            let row = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (row * 41 + dimension * 13 + row / 7) % 139;
            round_f16((tag as f32 - 69.0) * 0.0011)
        })
        .collect::<Vec<_>>();
    for (row, strength) in [
        (511usize, 64.0f32),
        (512, 64.01),
        (526, 96.0),
        (527, 96.01),
        (894, 128.0),
        (895, 127.99),
        (896, 128.01),
        (2_047, 160.0),
        (8_190, 192.0),
        (8_191, 192.01),
    ] {
        for dimension in 0..HEAD_DIM {
            compressed[row * HEAD_DIM + dimension] =
                round_f16(head_zero_query[dimension] * strength);
        }
    }
    let compressed_bits = compressed
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![HEAD_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let query_tensor = offset_f32(&ctx, &queries, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let mut sink_values = (0..HEADS)
        .map(|head| -0.41 + head as f32 * 0.002)
        .collect::<Vec<_>>();
    sink_values[HEADS - 1] = 4.0;
    let sinks = offset_f32(&ctx, &sink_values, vec![HEADS as u64]);

    for count in [512usize, 513, 527, 528, 895, 896, 897, 2_048, 8_191, 8_192] {
        let position = count * 128 - 1;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut preserved_raw = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
        let mut current_raw = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
        let mut chronological_raw = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
        for logical_position in raw_start..=position {
            let row = (0..HEAD_DIM)
                .map(|dimension| {
                    let tag = (logical_position * 17 + dimension * 5 + logical_position / 11) % 127;
                    round_f16((tag as f32 - 63.0) * 0.0009)
                })
                .collect::<Vec<_>>();
            chronological_raw.extend_from_slice(&row);
            let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
            let range = slot * HEAD_DIM..(slot + 1) * HEAD_DIM;
            current_raw[range.clone()].copy_from_slice(&row);
            if logical_position < position {
                preserved_raw[range].copy_from_slice(&row);
            } else {
                for (dimension, value) in preserved_raw[range].iter_mut().enumerate() {
                    let tag = (slot * 31 + dimension * 7 + count / 3) % 131;
                    *value = round_f16((tag as f32 - 65.0) * 0.0013);
                }
            }
        }
        let expected_head_zero = shared_kv_attention(
            head_zero_query,
            1,
            HEAD_DIM,
            &chronological_raw,
            &compressed[..count * HEAD_DIM],
            None,
            &sink_values[..1],
        )
        .unwrap();
        let prior_head_zero = shared_kv_attention(
            head_zero_query,
            1,
            HEAD_DIM,
            &chronological_raw,
            &compressed[..(count - 1) * HEAD_DIM],
            None,
            &sink_values[..1],
        )
        .unwrap();
        assert!(
            expected_head_zero
                .iter()
                .zip(&prior_head_zero)
                .any(|(current, prior)| (current - prior).abs() > 1e-3),
            "{count}-row fixture does not expose its final HCA row"
        );
        let to_f16_tensor = |values: &[f32], label: &str| {
            let bits = values
                .iter()
                .map(|value| half::f16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&bits),
                vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                GgmlType::F16,
            )
            .unwrap_or_else(|error| panic!("allocate {label}: {error}"))
        };
        let current_raw = to_f16_tensor(&current_raw, "current raw cache");
        let preserved_raw = to_f16_tensor(&preserved_raw, "preserved raw cache");
        let legacy = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let online = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let online_repeat =
            MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let direct = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let grouped = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let split8 = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let split8_repeat =
            MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1]).unwrap();
        let split8_partial =
            MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, HEADS as u64, 8]).unwrap();
        let split8_ml = MetalTensor::zeros_f32(&ctx, vec![2, HEADS as u64, 8]).unwrap();
        let rows = DeepSeekV4PublishedRows {
            cache: &compressed_cache,
            count,
            capacity_rows: CAPACITY,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_tiled_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &current_raw,
            &preserved_raw,
            DeepSeekV4RawCacheLayout::Ring,
            rows,
            &sinks,
            &legacy,
            position as u32,
            0,
            1,
            128,
            config,
        )
        .unwrap();
        for output in [&online, &online_repeat] {
            encode_online_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &query_tensor,
                &current_raw,
                &preserved_raw,
                DeepSeekV4RawCacheLayout::Ring,
                rows,
                &sinks,
                output,
                position as u32,
                0,
                1,
                128,
                false,
                config,
            )
            .unwrap();
        }
        encode_online_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &current_raw,
            &preserved_raw,
            DeepSeekV4RawCacheLayout::Ring,
            rows,
            &sinks,
            &direct,
            position as u32,
            0,
            1,
            128,
            true,
            config,
        )
        .unwrap();
        encode_grouped_online_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &current_raw,
            &preserved_raw,
            DeepSeekV4RawCacheLayout::Ring,
            Some(rows),
            &sinks,
            &grouped,
            AttentionKind::HeavilyCompressed,
            position as u32,
            1,
            config,
        )
        .unwrap();
        for output in [&split8, &split8_repeat] {
            encode_grouped_splitk_hca_f16(
                &ctx,
                &encoder,
                &query_tensor,
                &current_raw,
                &preserved_raw,
                DeepSeekV4RawCacheLayout::Ring,
                rows,
                &sinks,
                &split8_partial,
                &split8_ml,
                output,
                position as u32,
                8,
                config,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "{count}-row online HCA command failed: {:?}",
            command.error()
        );
        let legacy = read_f32(&legacy);
        let online = read_f32(&online);
        assert_eq!(
            online
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&online_repeat)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row online HCA is not bit-stable"
        );
        assert_eq!(
            online
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&direct)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row direct HCA changed the online recurrence"
        );
        assert_eq!(
            online
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&grouped)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row grouped HCA changed the online recurrence"
        );
        let split8 = read_f32(&split8);
        assert_eq!(
            split8
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&split8_repeat)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row split-K HCA is not bit-stable"
        );
        assert_close(
            &format!("{count}-row legacy HCA head zero"),
            &legacy[..HEAD_DIM],
            &expected_head_zero,
            8e-5,
        );
        assert_envelope(&format!("online HCA rows={count}"), &online, &legacy);
        assert_envelope(&format!("split8 HCA rows={count}"), &split8, &legacy);
        assert_envelope(
            &format!("online HCA CPU head zero rows={count}"),
            &online[..HEAD_DIM],
            &expected_head_zero,
        );
    }
}

#[test]
fn tiled_hca_terminal_pair_preserves_visibility_and_raw_causality() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 512;
    const CAPACITY: usize = 8_192;
    const START_POSITION: usize = 1_048_574;
    const QUERY_COUNT: usize = 2;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: 1,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let queries = (0..QUERY_COUNT * HEAD_DIM)
        .map(|index| {
            let token = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            0.021 + token as f32 * 0.003 - (dimension % 31) as f32 * 0.0005
        })
        .collect::<Vec<_>>();
    let raw_value = |position: usize, dimension: usize| {
        let tag = (position * 29 + dimension * 7 + position / 17) % 131;
        round_f16((tag as f32 - 65.0) * 0.001)
    };
    let final_raw = queries[HEAD_DIM..]
        .iter()
        .map(|value| round_f16(value * 640.0))
        .collect::<Vec<_>>();
    let mut raw_before = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
    for position in START_POSITION - DEEPSEEK_V4_LOCAL_WINDOW..START_POSITION {
        let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
        for dimension in 0..HEAD_DIM {
            raw_before[slot * HEAD_DIM + dimension] = raw_value(position, dimension);
        }
    }
    let mut raw_chunk = (0..HEAD_DIM)
        .map(|dimension| raw_value(START_POSITION, dimension))
        .collect::<Vec<_>>();
    raw_chunk.extend_from_slice(&final_raw);
    let mut compressed = (0..CAPACITY * HEAD_DIM)
        .map(|index| {
            let row = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let tag = (row * 47 + dimension * 11 + row / 5) % 157;
            round_f16((tag as f32 - 78.0) * 0.0009)
        })
        .collect::<Vec<_>>();
    for dimension in 0..HEAD_DIM {
        compressed[(CAPACITY - 1) * HEAD_DIM + dimension] =
            round_f16(queries[HEAD_DIM + dimension] * 768.0);
    }
    let sink_values = [-0.33f32];
    let mut expected = Vec::with_capacity(QUERY_COUNT * HEAD_DIM);
    for token in 0..QUERY_COUNT {
        let position = START_POSITION + token;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
        for logical_position in raw_start..=position {
            if logical_position == START_POSITION + 1 {
                raw_rows.extend_from_slice(&final_raw);
            } else {
                raw_rows
                    .extend((0..HEAD_DIM).map(|dimension| raw_value(logical_position, dimension)));
            }
        }
        let visible = (position + 1) / 128;
        expected.extend(
            shared_kv_attention(
                &queries[token * HEAD_DIM..(token + 1) * HEAD_DIM],
                1,
                HEAD_DIM,
                &raw_rows,
                &compressed[..visible * HEAD_DIM],
                None,
                &sink_values,
            )
            .unwrap(),
        );
    }
    let first_raw_start = START_POSITION + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
    let mut leaked_raw = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
    for logical_position in first_raw_start..=START_POSITION {
        if logical_position == first_raw_start {
            leaked_raw.extend_from_slice(&final_raw);
        } else {
            leaked_raw
                .extend((0..HEAD_DIM).map(|dimension| raw_value(logical_position, dimension)));
        }
    }
    let leaked = shared_kv_attention(
        &queries[..HEAD_DIM],
        1,
        HEAD_DIM,
        &leaked_raw,
        &compressed[..(CAPACITY - 1) * HEAD_DIM],
        None,
        &sink_values,
    )
    .unwrap();
    assert!(
        expected[..HEAD_DIM]
            .iter()
            .zip(leaked)
            .any(|(correct, leaked)| (correct - leaked).abs() > 1e-3)
    );

    let queries_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, QUERY_COUNT as u64]);
    let make_raw = |values: &[f32], rows: usize| {
        let bits = values
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bits),
            vec![HEAD_DIM as u64, rows as u64],
            GgmlType::F16,
        )
        .unwrap()
    };
    let raw_before = make_raw(&raw_before, DEEPSEEK_V4_LOCAL_WINDOW);
    let raw_current = make_raw(&raw_chunk, QUERY_COUNT);
    let compressed_bits = compressed
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_tensor = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![HEAD_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sinks = offset_f32(&ctx, &sink_values, vec![1]);
    let output = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, QUERY_COUNT as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_tiled_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries_tensor,
        &raw_current,
        &raw_before,
        DeepSeekV4RawCacheLayout::Chunk,
        DeepSeekV4PublishedRows {
            cache: &compressed_tensor,
            count: CAPACITY,
            capacity_rows: CAPACITY,
        },
        &sinks,
        &output,
        START_POSITION as u32,
        0,
        QUERY_COUNT,
        128,
        config,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_close(
        "terminal paired tiled HCA",
        &read_f32(&output),
        &expected,
        8e-5,
    );
}

#[test]
fn hadamard_128_matches_the_indexer_oracle() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let values = (0usize..128)
        .map(|index| {
            ((index * 17 + index / 3 + 5) % 61) as f32 * 0.031 - 0.83
                + if index.is_multiple_of(7) { 0.19 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let mut expected = values.clone();
    hadamard_128_in_place(&mut expected).unwrap();
    let tensor = offset_f32(&ctx, &values, vec![128]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_hadamard_128_in_place(&ctx, &encoder, &tensor).unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );
    assert_close("Hadamard-128", &read_f32(&tensor), &expected, 2e-6);
}

#[test]
fn multigroup_selector_threshold_matches_current_and_fails_closed() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 4_096;
    const TOP_K: usize = 512;
    const GROUPS: usize = 8;
    const GENERATION: u32 = 0x1357_2468;

    let run_case = |label: &str, values: &[f32], visible: i32| {
        let scores = offset_f32(&ctx, values, vec![CAPACITY as u64, 1]);
        let visible_counts = offset_i32(&ctx, &[visible], vec![1]);
        let records = offset_i32(
            &ctx,
            &vec![i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS * GROUPS],
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                GROUPS as u64,
            ],
        );
        let partition_plan = offset_i32(
            &ctx,
            &[i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS * GROUPS],
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                GROUPS as u64,
            ],
        );
        let state = offset_i32(
            &ctx,
            &[i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS],
            vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        );
        let execute = || {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_select_top_k_multigroup_threshold_f32(
                &ctx,
                &encoder,
                &scores,
                &visible_counts,
                &records,
                &partition_plan,
                &state,
                CAPACITY,
                TOP_K,
                GROUPS,
                GENERATION,
                None,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label}: {:?}", command.error());
            (
                read_i32(&records),
                read_i32(&partition_plan),
                read_i32(&state),
            )
        };
        let first = execute();
        let repeat = execute();
        assert_eq!(repeat, first, "{label}: repeat drift");

        let expected =
            multigroup_selector_threshold_oracle(values, CAPACITY, visible, TOP_K, GROUPS);
        let state_words = first.2.iter().map(|&word| word as u32).collect::<Vec<_>>();
        assert_eq!(state_words[0], GENERATION, "{label}: generation");
        assert_eq!(state_words[1], 7, "{label}: digit");
        assert_eq!(state_words[5], expected.status, "{label}: status");
        assert_eq!(
            state_words[6], expected.threshold_key,
            "{label}: threshold key"
        );
        assert_eq!(
            state_words[7], expected.threshold_take,
            "{label}: threshold take"
        );
        assert_eq!(
            state_words[8], expected.selected_count,
            "{label}: selected count"
        );
        assert_eq!(
            state_words[9],
            deepseek_v4_multigroup_selector_state_completion(GENERATION, 7),
            "{label}: state completion"
        );
        assert_eq!(
            first.1.iter().map(|&word| word as u32).collect::<Vec<_>>(),
            expected.partition_plan,
            "{label}: partition plan"
        );
        let expected_record_error = expected.status.min(2);
        for group in 0..GROUPS {
            let base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS;
            assert_eq!(
                first.0[base] as u32, GENERATION,
                "{label}: record generation"
            );
            assert_eq!(first.0[base + 1], 7, "{label}: record digit");
            assert_eq!(
                first.0[base + 2] as u32,
                expected_record_error,
                "{label}: record error"
            );
            assert_eq!(
                first.0[base + 3] as u32,
                deepseek_v4_multigroup_selector_record_completion(GENERATION, 7, group),
                "{label}: record completion"
            );
        }

        let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
        let ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
        let count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
        let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_select_top_k_f32(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &mask,
            None,
            &ids,
            &count,
            &status,
            CAPACITY,
            CAPACITY,
            TOP_K,
            1,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        assert_eq!(
            read_i32(&status),
            [expected.status as i32],
            "{label}: current status"
        );
        assert_eq!(
            read_i32(&count),
            [expected.selected_count as i32],
            "{label}: current count"
        );
        if expected.status == 0 {
            let selected_ids = read_i32(&ids);
            let selected_keys = selected_ids
                .iter()
                .map(|&id| deployed_selector_order_key(values[id as usize]))
                .collect::<Vec<_>>();
            let threshold = *selected_keys.iter().min().unwrap();
            let take = selected_keys
                .iter()
                .filter(|&&key| key == threshold)
                .count();
            assert_eq!(
                threshold, expected.threshold_key,
                "{label}: current threshold"
            );
            assert_eq!(
                take, expected.threshold_take as usize,
                "{label}: current take"
            );
        }
    };

    let mixed = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    run_case("mixed", &mixed, CAPACITY as i32);
    run_case("all tied", &vec![0.0; CAPACITY], CAPACITY as i32);

    let mut threshold_take_one = vec![-1.0; CAPACITY];
    for (row, value) in threshold_take_one.iter_mut().take(TOP_K - 1).enumerate() {
        *value = 2.0 + row as f32 * 0.001;
    }
    threshold_take_one[CAPACITY - 1] = f32::from_bits(1);
    run_case(
        "last eligible subnormal tie",
        &threshold_take_one,
        CAPACITY as i32,
    );

    let mut partition_boundary = vec![-1.0; CAPACITY];
    for (offset, value) in partition_boundary[256..768].iter_mut().enumerate() {
        *value = match offset % 4 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            _ => f32::from_bits(0x8000_0001),
        };
    }
    run_case(
        "partition boundary zero and subnormal tie",
        &partition_boundary,
        CAPACITY as i32,
    );

    let mut outside_nonfinite = mixed.clone();
    outside_nonfinite[CAPACITY - 1] = f32::INFINITY;
    run_case(
        "nonfinite outside visibility",
        &outside_nonfinite,
        CAPACITY as i32 - 1,
    );
    let mut visible_nonfinite = mixed.clone();
    visible_nonfinite[CAPACITY / 2] = f32::NAN;
    run_case("visible nonfinite", &visible_nonfinite, CAPACITY as i32);
    run_case("invalid visibility", &mixed, 0);

    let scores = offset_f32(&ctx, &mixed, vec![CAPACITY as u64, 1]);
    let visible = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
    let records = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let partition_plan = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let state = MetalTensor::zeros_i32(
        &ctx,
        vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
    )
    .unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_multigroup_threshold_f32(
        &ctx,
        &encoder,
        &scores,
        &visible,
        &records,
        &partition_plan,
        &state,
        CAPACITY,
        TOP_K,
        GROUPS,
        GENERATION,
        Some(3),
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    let stale_state = read_i32(&state)
        .into_iter()
        .map(|word| word as u32)
        .collect::<Vec<_>>();
    assert_eq!(stale_state[0], GENERATION, "stale state generation");
    assert_eq!(stale_state[1], 7, "stale state digit");
    assert_eq!(stale_state[5], 3, "stale record must produce status 3");
    assert_eq!(stale_state[6], 0, "stale state threshold key");
    assert_eq!(stale_state[7], 0, "stale state threshold take");
    assert_eq!(
        stale_state[9],
        deepseek_v4_multigroup_selector_state_completion(GENERATION, 7),
        "stale state completion"
    );
    let stale_records = read_i32(&records);
    for group in 0..GROUPS {
        let base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS;
        assert_eq!(
            stale_records[base] as u32, GENERATION,
            "stale record generation group={group}"
        );
        assert_eq!(
            stale_records[base + 1],
            7,
            "stale record digit group={group}"
        );
        assert_eq!(
            stale_records[base + 2],
            3,
            "stale record error group={group}"
        );
        assert_eq!(
            stale_records[base + 3] as u32,
            deepseek_v4_multigroup_selector_record_completion(GENERATION, 7, group),
            "stale record completion group={group}"
        );
    }
}

#[test]
fn multigroup_selector_full_matches_current_and_fails_closed() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 4_096;
    const TOP_K: usize = 512;
    const GROUPS: usize = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;

    #[derive(Debug, Eq, PartialEq)]
    struct PublishedSelection {
        mask: Vec<i32>,
        ids: Vec<i32>,
        count: Vec<i32>,
        status: Vec<i32>,
    }

    let run_case = |label: &str, values: &[f32], visible: i32, generation: u32| {
        let scores = offset_f32(&ctx, values, vec![CAPACITY as u64, 1]);
        let visible_counts = offset_i32(&ctx, &[visible], vec![1]);
        let current_mask = offset_i32(&ctx, &vec![-7; CAPACITY], vec![CAPACITY as u64, 1]);
        let current_ids = offset_i32(&ctx, &[-7; TOP_K], vec![TOP_K as u64, 1]);
        let current_count = offset_i32(&ctx, &[-7], vec![1]);
        let current_status = offset_i32(&ctx, &[-7], vec![1]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_select_top_k_f32(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &current_mask,
            None,
            &current_ids,
            &current_count,
            &current_status,
            CAPACITY,
            CAPACITY,
            TOP_K,
            1,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{label}: current command");
        let current = PublishedSelection {
            mask: read_i32(&current_mask),
            ids: read_i32(&current_ids),
            count: read_i32(&current_count),
            status: read_i32(&current_status),
        };

        let records = offset_i32(
            &ctx,
            &vec![i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS * GROUPS],
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                GROUPS as u64,
            ],
        );
        let partition_plan = offset_i32(
            &ctx,
            &vec![i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS * GROUPS],
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                GROUPS as u64,
            ],
        );
        let state = offset_i32(
            &ctx,
            &[i32::MIN; DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS],
            vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        );
        let private_mask = offset_i8(&ctx, &vec![0xa5; CAPACITY], vec![CAPACITY as u64, 1]);
        let private_ids = offset_i32(&ctx, &[-11; TOP_K], vec![TOP_K as u64, 1]);
        let selected_mask = offset_i32(&ctx, &vec![-13; CAPACITY], vec![CAPACITY as u64, 1]);
        let cache_order_ids = offset_i32(&ctx, &[-13; TOP_K], vec![TOP_K as u64, 1]);
        let selected_count = offset_i32(&ctx, &[-13], vec![1]);
        let status = offset_i32(&ctx, &[-13], vec![1]);
        let execute = |invocation_generation: u32| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_select_top_k_multigroup_full_f32(
                &ctx,
                &encoder,
                &scores,
                &visible_counts,
                &records,
                &partition_plan,
                &state,
                &private_mask,
                &private_ids,
                &selected_mask,
                &cache_order_ids,
                &selected_count,
                &status,
                CAPACITY,
                TOP_K,
                invocation_generation,
                None,
                false,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label}: candidate command");
            PublishedSelection {
                mask: read_i32(&selected_mask),
                ids: read_i32(&cache_order_ids),
                count: read_i32(&selected_count),
                status: read_i32(&status),
            }
        };
        let first = execute(generation);
        let repeat = execute(generation + 1);
        assert_eq!(first, current, "{label}: candidate/current output");
        assert_eq!(repeat, first, "{label}: generation-repeat output");

        let expected =
            multigroup_selector_threshold_oracle(values, CAPACITY, visible, TOP_K, GROUPS);
        assert_eq!(
            read_i32(&partition_plan)
                .into_iter()
                .map(|word| word as u32)
                .collect::<Vec<_>>(),
            expected.partition_plan,
            "{label}: partition plan"
        );
        let state_words = read_i32(&state)
            .into_iter()
            .map(|word| word as u32)
            .collect::<Vec<_>>();
        assert_eq!(state_words[0], generation + 1, "{label}: state generation");
        assert_eq!(state_words[5], expected.status, "{label}: state status");
        assert_eq!(
            state_words[8], expected.selected_count,
            "{label}: state selected count"
        );
        let record_words = read_i32(&records);
        for group in 0..GROUPS {
            let record_base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS;
            let plan_base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS;
            assert_eq!(
                record_words[record_base] as u32,
                generation + 1,
                "{label}: compact generation group={group}"
            );
            assert_eq!(
                record_words[record_base + 1],
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_COMPACT_PHASE,
                "{label}: compact phase group={group}"
            );
            assert_eq!(
                record_words[record_base + 2] as u32,
                expected.status,
                "{label}: compact status group={group}"
            );
            assert_eq!(
                record_words[record_base + 3] as u32,
                deepseek_v4_multigroup_selector_compact_completion(generation + 1, group,),
                "{label}: compact completion group={group}"
            );
            if expected.status == 0 {
                assert_eq!(
                    record_words[record_base + 4] as u32,
                    expected.partition_plan
                        [plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_GREATER],
                    "{label}: compact greater group={group}"
                );
                assert_eq!(
                    record_words[record_base + 5] as u32,
                    expected.partition_plan[plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_EQUAL],
                    "{label}: compact equal group={group}"
                );
                assert_eq!(
                    record_words[record_base + 6] as u32,
                    expected.partition_plan
                        [plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_SELECTED],
                    "{label}: compact selected group={group}"
                );
                assert_eq!(
                    record_words[record_base + 7] as u32,
                    expected.partition_plan
                        [plan_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_ID_OFFSET],
                    "{label}: compact offset group={group}"
                );
            } else {
                assert_eq!(
                    &record_words[record_base + 4..record_base + 8],
                    [0; 4],
                    "{label}: failed compact payload group={group}"
                );
            }
            assert_eq!(
                &record_words
                    [record_base + 8..record_base + DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS],
                [0; DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS - 8],
                "{label}: compact reserved words group={group}"
            );
        }
        if expected.status == 0 {
            assert_eq!(
                read_u8(&private_mask),
                current
                    .mask
                    .iter()
                    .map(|&value| value as u8)
                    .collect::<Vec<_>>(),
                "{label}: private mask"
            );
            assert_eq!(
                &read_i32(&private_ids)[..expected.selected_count as usize],
                &current.ids[..expected.selected_count as usize],
                "{label}: private IDs"
            );
        }
    };

    let mixed = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    run_case("mixed", &mixed, CAPACITY as i32, 0x3100_0001);
    run_case(
        "all tied",
        &vec![0.0; CAPACITY],
        CAPACITY as i32,
        0x3100_0011,
    );

    let mut crossing_tie = vec![-1.0; CAPACITY];
    crossing_tie[..480].fill(2.0);
    crossing_tie[500..541].fill(1.0);
    run_case(
        "cutoff tie crosses partition",
        &crossing_tie,
        CAPACITY as i32,
        0x3100_0021,
    );

    let mut canonical_tie = vec![-1.0; CAPACITY];
    canonical_tie[..500].fill(2.0);
    for (offset, value) in canonical_tie[500..701].iter_mut().enumerate() {
        *value = match offset % 4 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            _ => f32::from_bits(0x8000_0001),
        };
    }
    run_case(
        "canonical tie crosses partition",
        &canonical_tie,
        CAPACITY as i32,
        0x3100_0031,
    );
    run_case("visible below top-k", &mixed, 511, 0x3100_0041);
    let mut outside_nonfinite = mixed.clone();
    outside_nonfinite[CAPACITY - 1] = f32::INFINITY;
    run_case(
        "nonfinite outside visibility",
        &outside_nonfinite,
        CAPACITY as i32 - 1,
        0x3100_0051,
    );
    let mut visible_nonfinite = mixed.clone();
    visible_nonfinite[CAPACITY / 2] = f32::NAN;
    run_case(
        "visible nonfinite fallback",
        &visible_nonfinite,
        CAPACITY as i32,
        0x3100_0061,
    );
    run_case("invalid visibility fallback", &mixed, 0, 0x3100_0071);

    let run_internal_fault =
        |label: &str, generation: u32, fault_digit: Option<usize>, omit_last_compactor: bool| {
            let scores = offset_f32(&ctx, &mixed, vec![CAPACITY as u64, 1]);
            let visible = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
            let records = MetalTensor::zeros_i32(
                &ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                    GROUPS as u64,
                ],
            )
            .unwrap();
            let partition_plan = MetalTensor::zeros_i32(
                &ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                    GROUPS as u64,
                ],
            )
            .unwrap();
            let state = MetalTensor::zeros_i32(
                &ctx,
                vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
            )
            .unwrap();
            let private_mask =
                MetalTensor::zeros_dtype(&ctx, vec![CAPACITY as u64, 1], GgmlType::I8).unwrap();
            let private_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
            let mask = offset_i32(&ctx, &vec![-17; CAPACITY], vec![CAPACITY as u64, 1]);
            let ids = offset_i32(&ctx, &[-17; TOP_K], vec![TOP_K as u64, 1]);
            let count = offset_i32(&ctx, &[-17], vec![1]);
            let status = offset_i32(&ctx, &[-17], vec![1]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_select_top_k_multigroup_full_f32(
                &ctx,
                &encoder,
                &scores,
                &visible,
                &records,
                &partition_plan,
                &state,
                &private_mask,
                &private_ids,
                &mask,
                &ids,
                &count,
                &status,
                CAPACITY,
                TOP_K,
                generation,
                fault_digit,
                omit_last_compactor,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label}: command");
            assert_eq!(read_i32(&status), [3], "{label}: status");
            assert_eq!(read_i32(&count), [TOP_K as i32], "{label}: count");
            assert_eq!(
                read_i32(&mask),
                (0..CAPACITY)
                    .map(|row| i32::from(row < TOP_K))
                    .collect::<Vec<_>>(),
                "{label}: fallback mask"
            );
            assert_eq!(
                read_i32(&ids),
                (0..TOP_K as i32).collect::<Vec<_>>(),
                "{label}: fallback IDs"
            );
        };
    run_internal_fault("missing histogram producer", 0x3100_0081, Some(3), false);
    run_internal_fault("missing compactor", 0x3100_0091, None, true);
}

#[test]
fn multigroup_selector_publisher_rejects_corrupt_private_state() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 4_096;
    const TOP_K: usize = 512;
    const GROUPS: usize = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    const GENERATION: u32 = 0x3200_0001;

    let values = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let scores = offset_f32(&ctx, &values, vec![CAPACITY as u64, 1]);
    let visible = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
    let records = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let partition_plan = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let state = MetalTensor::zeros_i32(
        &ctx,
        vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
    )
    .unwrap();
    let private_mask =
        MetalTensor::zeros_dtype(&ctx, vec![CAPACITY as u64, 1], GgmlType::I8).unwrap();
    let private_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_multigroup_full_f32(
        &ctx,
        &encoder,
        &scores,
        &visible,
        &records,
        &partition_plan,
        &state,
        &private_mask,
        &private_ids,
        &mask,
        &ids,
        &count,
        &status,
        CAPACITY,
        TOP_K,
        GENERATION,
        None,
        false,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(read_i32(&status), [0]);

    let exact_mask = read_u8(&private_mask);
    let exact_ids = read_i32(&private_ids);
    let exact_plan = read_i32(&partition_plan);
    let exact_records = read_i32(&records);
    let assert_fault = |label: &str| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_select_top_k_multigroup_publish_only_f32(
            &ctx,
            &encoder,
            &visible,
            &records,
            &partition_plan,
            &state,
            &private_mask,
            &private_ids,
            &mask,
            &ids,
            &count,
            &status,
            CAPACITY,
            TOP_K,
            GENERATION,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{label}: command");
        assert_eq!(read_i32(&status), [3], "{label}: status");
        assert_eq!(read_i32(&count), [TOP_K as i32], "{label}: count");
        assert_eq!(
            read_i32(&mask),
            (0..CAPACITY)
                .map(|row| i32::from(row < TOP_K))
                .collect::<Vec<_>>(),
            "{label}: fallback mask"
        );
        assert_eq!(
            read_i32(&ids),
            (0..TOP_K as i32).collect::<Vec<_>>(),
            "{label}: fallback IDs"
        );
    };

    let selected_row = exact_ids[0] as usize;
    unsafe {
        let pointer = private_mask
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(private_mask.offset as usize + selected_row);
        pointer.write(2);
    }
    assert_fault("non-binary private mask");
    unsafe {
        let destination = private_mask
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(private_mask.offset as usize);
        std::ptr::copy_nonoverlapping(exact_mask.as_ptr(), destination, exact_mask.len());
    }

    let mut corrupt_ids = exact_ids.clone();
    corrupt_ids[1] = corrupt_ids[0];
    host_write_i32(&private_ids, &corrupt_ids, "corrupt private selector IDs").unwrap();
    assert_fault("duplicate private ID");
    host_write_i32(&private_ids, &exact_ids, "restore private selector IDs").unwrap();

    let mut corrupt_plan = exact_plan.clone();
    corrupt_plan[DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_ID_OFFSET] += 1;
    host_write_i32(&partition_plan, &corrupt_plan, "corrupt selector plan").unwrap();
    assert_fault("corrupt partition offset");
    host_write_i32(&partition_plan, &exact_plan, "restore selector plan").unwrap();

    let mut corrupt_records = exact_records.clone();
    corrupt_records[3] ^= 1;
    host_write_i32(&records, &corrupt_records, "corrupt compaction records").unwrap();
    assert_fault("corrupt compaction completion");
}

#[test]
fn sparse_csa_multigroup_route_is_exact_and_opt_in() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 262_144;

    #[derive(Debug, Eq, PartialEq)]
    struct PublishedSelection {
        mask: Vec<i32>,
        ids: Vec<i32>,
        count: Vec<i32>,
        status: Vec<i32>,
    }

    fn execute(
        ctx: &MetalContext,
        scratch: &DeepSeekV4SparseCsaScratch,
        values: &[f32],
        visible_rows: usize,
    ) -> PublishedSelection {
        host_write_f32(&scratch.scores, values, "integrated selector scores").unwrap();
        let record = scratch.default_record();
        host_write_i32(
            &record.visible_count,
            &[visible_rows as i32],
            "integrated selector visibility",
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_scored_rows(ctx, &encoder, CAPACITY, visible_rows, &record)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());
        scratch.validate_completed(&record).unwrap();
        PublishedSelection {
            mask: read_i32(&scratch.selected_mask),
            ids: read_i32(&scratch.cache_order_ids),
            count: read_i32(&record.selected_count),
            status: read_i32(&record.status),
        }
    }

    let mixed = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let tied = vec![0.0f32; CAPACITY];
    let current = DeepSeekV4SparseCsaScratch::new(&ctx, CAPACITY).unwrap();
    let mut candidate = DeepSeekV4SparseCsaScratch::new(&ctx, CAPACITY).unwrap();
    candidate.enable_multigroup_selector_experiment().unwrap();
    let generation = &candidate.multigroup.as_ref().unwrap().generation;

    let ineligible = execute(&ctx, &current, &mixed, 131_072);
    assert_eq!(execute(&ctx, &candidate, &mixed, 131_072), ineligible);
    assert_eq!(generation.next.get(), Some(NonZeroU32::MIN));

    for (label, values) in [("mixed", mixed.as_slice()), ("tied", tied.as_slice())] {
        let before = execute(&ctx, &current, values, 196_608);
        let selected = execute(&ctx, &candidate, values, 196_608);
        let after = execute(&ctx, &current, values, 196_608);
        assert_eq!(selected, before, "{label}: candidate output");
        assert_eq!(after, before, "{label}: repeated radix4 output");
    }
    assert_eq!(generation.next.get().unwrap().get(), 3);
}

#[test]
#[ignore = "focused terminal full multi-group selector ceiling; run explicitly with --nocapture"]
fn profile_multigroup_selector_full_ceiling() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 262_144;
    const TOP_K: usize = 512;
    const GROUPS: usize = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    const SAMPLES: usize = 24;
    const MAX_MEDIAN_MS: f64 = 1.35;
    const MAX_P95_MS: f64 = 1.40;
    const MIN_SAVING_MS: f64 = 0.50;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PublishedSelection {
        mask: Vec<i32>,
        ids: Vec<i32>,
        count: Vec<i32>,
        status: Vec<i32>,
    }

    fn timed_gpu_wall<F>(ctx: &MetalContext, encode: F) -> (f64, f64)
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(command.error().is_none(), "{:?}", command.error());
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
        assert!(wall_ms.is_finite() && wall_ms > 0.0);
        (gpu_ms, wall_ms)
    }

    fn median_and_p95(samples: &[f64]) -> (f64, f64) {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };
        let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    let mixed_values = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let tied_values = vec![0.0f32; CAPACITY];
    let mixed_scores = offset_f32(&ctx, &mixed_values, vec![CAPACITY as u64, 1]);
    let tied_scores = offset_f32(&ctx, &tied_values, vec![CAPACITY as u64, 1]);
    let visible = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);

    let current_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let current_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let current_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let current_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let records = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let partition_plan = MetalTensor::zeros_i32(
        &ctx,
        vec![
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            GROUPS as u64,
        ],
    )
    .unwrap();
    let state = MetalTensor::zeros_i32(
        &ctx,
        vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
    )
    .unwrap();
    let private_mask =
        MetalTensor::zeros_dtype(&ctx, vec![CAPACITY as u64, 1], GgmlType::I8).unwrap();
    let private_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let candidate_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let candidate_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let candidate_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let candidate_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let generation = std::cell::Cell::new(0x4200_0001u32);
    let next_generation = || {
        let current = generation.get();
        let next = current.wrapping_add(1);
        generation.set(if next == 0 { 1 } else { next });
        current
    };

    let read_current = || PublishedSelection {
        mask: read_i32(&current_mask),
        ids: read_i32(&current_ids),
        count: read_i32(&current_count),
        status: read_i32(&current_status),
    };
    let read_candidate = || PublishedSelection {
        mask: read_i32(&candidate_mask),
        ids: read_i32(&candidate_ids),
        count: read_i32(&candidate_count),
        status: read_i32(&candidate_status),
    };
    let time_current = |scores: &MetalTensor| {
        timed_gpu_wall(&ctx, |encoder| {
            encode_select_top_k_f32(
                &ctx,
                encoder,
                scores,
                &visible,
                &current_mask,
                None,
                &current_ids,
                &current_count,
                &current_status,
                CAPACITY,
                CAPACITY,
                TOP_K,
                1,
            )
        })
    };
    let time_candidate = |scores: &MetalTensor| {
        let invocation_generation = next_generation();
        timed_gpu_wall(&ctx, |encoder| {
            encode_select_top_k_multigroup_full_f32(
                &ctx,
                encoder,
                scores,
                &visible,
                &records,
                &partition_plan,
                &state,
                &private_mask,
                &private_ids,
                &candidate_mask,
                &candidate_ids,
                &candidate_count,
                &candidate_status,
                CAPACITY,
                TOP_K,
                invocation_generation,
                None,
                false,
            )
        })
    };

    for (label, values, scores) in [
        ("mixed", mixed_values.as_slice(), &mixed_scores),
        ("tied", tied_values.as_slice(), &tied_scores),
    ] {
        time_current(scores);
        let expected_output = read_current();
        time_candidate(scores);
        assert_eq!(read_candidate(), expected_output, "{label}: untimed output");
        for _ in 0..5 {
            time_current(scores);
            time_candidate(scores);
        }

        let before = (0..SAMPLES)
            .map(|_| time_current(scores))
            .collect::<Vec<_>>();
        assert_eq!(read_current(), expected_output, "{label}: before output");
        let candidate = (0..SAMPLES)
            .map(|_| time_candidate(scores))
            .collect::<Vec<_>>();
        assert_eq!(
            read_candidate(),
            expected_output,
            "{label}: candidate output"
        );
        let after = (0..SAMPLES)
            .map(|_| time_current(scores))
            .collect::<Vec<_>>();
        assert_eq!(read_current(), expected_output, "{label}: after output");

        let expected =
            multigroup_selector_threshold_oracle(values, CAPACITY, CAPACITY as i32, TOP_K, GROUPS);
        assert_eq!(expected.status, 0, "{label}: oracle status");
        assert_eq!(
            read_i32(&partition_plan)
                .into_iter()
                .map(|word| word as u32)
                .collect::<Vec<_>>(),
            expected.partition_plan,
            "{label}: candidate plan"
        );
        let state_words = read_i32(&state)
            .into_iter()
            .map(|word| word as u32)
            .collect::<Vec<_>>();
        let final_generation = generation.get().wrapping_sub(1);
        assert_eq!(state_words[0], final_generation, "{label}: generation");
        assert_eq!(state_words[5], 0, "{label}: status");
        assert_eq!(state_words[6], expected.threshold_key, "{label}: threshold");
        assert_eq!(state_words[7], expected.threshold_take, "{label}: take");
        assert_eq!(state_words[8], TOP_K as u32, "{label}: count");
        let record_words = read_i32(&records);
        for group in 0..GROUPS {
            let base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS;
            assert_eq!(
                record_words[base] as u32, final_generation,
                "{label}: record generation group={group}"
            );
            assert_eq!(
                record_words[base + 1],
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_COMPACT_PHASE,
                "{label}: record phase group={group}"
            );
            assert_eq!(
                record_words[base + 2],
                0,
                "{label}: record error group={group}"
            );
            assert_eq!(
                record_words[base + 3] as u32,
                deepseek_v4_multigroup_selector_compact_completion(final_generation, group,),
                "{label}: record completion group={group}"
            );
        }

        let before_gpu = before.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let before_wall = before.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let candidate_gpu = candidate.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let candidate_wall = candidate.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let after_gpu = after.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let after_wall = after.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let (before_gpu_median, _) = median_and_p95(&before_gpu);
        let (before_wall_median, _) = median_and_p95(&before_wall);
        let (candidate_gpu_median, candidate_gpu_p95) = median_and_p95(&candidate_gpu);
        let (candidate_wall_median, candidate_wall_p95) = median_and_p95(&candidate_wall);
        let (after_gpu_median, _) = median_and_p95(&after_gpu);
        let (after_wall_median, _) = median_and_p95(&after_wall);
        let gpu_drift = 2.0 * (before_gpu_median - after_gpu_median).abs()
            / (before_gpu_median + after_gpu_median);
        let wall_drift = 2.0 * (before_wall_median - after_wall_median).abs()
            / (before_wall_median + after_wall_median);
        let gpu_saving = before_gpu_median.min(after_gpu_median) - candidate_gpu_median;
        let wall_saving = before_wall_median.min(after_wall_median) - candidate_wall_median;
        let passed = gpu_drift <= 0.05
            && wall_drift <= 0.05
            && candidate_gpu_median <= MAX_MEDIAN_MS
            && candidate_gpu_p95 <= MAX_P95_MS
            && candidate_wall_median <= MAX_MEDIAN_MS
            && candidate_wall_p95 <= MAX_P95_MS
            && gpu_saving >= MIN_SAVING_MS
            && wall_saving >= MIN_SAVING_MS;
        eprintln!(
            "deepseek_v4 multigroup_full case={label} current_before_gpu_median_ms={before_gpu_median:.6} current_before_wall_median_ms={before_wall_median:.6} candidate_gpu_median_ms={candidate_gpu_median:.6} candidate_gpu_p95_ms={candidate_gpu_p95:.6} candidate_wall_median_ms={candidate_wall_median:.6} candidate_wall_p95_ms={candidate_wall_p95:.6} current_after_gpu_median_ms={after_gpu_median:.6} current_after_wall_median_ms={after_wall_median:.6} gpu_drift={gpu_drift:.6} wall_drift={wall_drift:.6} faster_control_gpu_saving_ms={gpu_saving:.6} faster_control_wall_saving_ms={wall_saving:.6} pass={passed}"
        );
        eprintln!(
            "deepseek_v4 multigroup_full case={label} current_before_gpu_ms={before_gpu:?} current_before_wall_ms={before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} current_after_gpu_ms={after_gpu:?} current_after_wall_ms={after_wall:?}"
        );
        assert!(passed, "{label}: full multi-group selector gate failed");
    }
}

#[test]
#[ignore = "focused integrated multi-group selector gate; run explicitly with --nocapture"]
fn profile_sparse_csa_multigroup_integrated_gate() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const SAMPLES: usize = 24;
    const WARM_PAIRS: usize = 40;
    const MAX_MEDIAN_MS: f64 = 1.35;
    const MAX_P95_MS: f64 = 1.40;
    const MIN_SAVING_MS: f64 = 0.50;
    const CELLS: &[(usize, usize)] = &[(196_608, 196_608), (250_112, 196_608), (262_144, 196_608)];

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PublishedSelection {
        mask: Vec<i32>,
        ids: Vec<i32>,
        count: Vec<i32>,
        status: Vec<i32>,
    }

    fn timed_gpu_wall<F>(ctx: &MetalContext, encode: F) -> (f64, f64)
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(command.error().is_none(), "{:?}", command.error());
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
        assert!(wall_ms.is_finite() && wall_ms > 0.0);
        (gpu_ms, wall_ms)
    }

    fn median_and_p95(samples: &[f64]) -> (f64, f64) {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };
        let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    fn output(
        scratch: &DeepSeekV4SparseCsaScratch,
        record: &DeepSeekV4SelectionRecord,
    ) -> PublishedSelection {
        PublishedSelection {
            mask: read_i32(&scratch.selected_mask),
            ids: read_i32(&scratch.cache_order_ids),
            count: read_i32(&record.selected_count),
            status: read_i32(&record.status),
        }
    }

    for &(capacity, visible_rows) in CELLS {
        assert!(deepseek_v4_multigroup_selector_eligible(
            capacity,
            visible_rows
        ));
        for tied in [false, true] {
            let case = if tied { "tied" } else { "mixed" };
            let values = if tied {
                vec![0.0f32; capacity]
            } else {
                (0..capacity)
                    .map(|row| {
                        let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
                        bucket as f32 * 0.0003 - 1.1
                    })
                    .collect::<Vec<_>>()
            };
            let current = DeepSeekV4SparseCsaScratch::new(&ctx, capacity).unwrap();
            let mut candidate = DeepSeekV4SparseCsaScratch::new(&ctx, capacity).unwrap();
            candidate.enable_multigroup_selector_experiment().unwrap();
            host_write_f32(&current.scores, &values, "integrated control scores").unwrap();
            host_write_f32(&candidate.scores, &values, "integrated candidate scores").unwrap();
            let current_record = current.default_record();
            let candidate_record = candidate.default_record();
            host_write_i32(
                &current_record.visible_count,
                &[visible_rows as i32],
                "integrated control visibility",
            )
            .unwrap();
            host_write_i32(
                &candidate_record.visible_count,
                &[visible_rows as i32],
                "integrated candidate visibility",
            )
            .unwrap();
            let time_current = || {
                timed_gpu_wall(&ctx, |encoder| {
                    current.encode_scored_rows(
                        &ctx,
                        encoder,
                        capacity,
                        visible_rows,
                        &current_record,
                    )
                })
            };
            let time_candidate = || {
                timed_gpu_wall(&ctx, |encoder| {
                    candidate.encode_scored_rows(
                        &ctx,
                        encoder,
                        capacity,
                        visible_rows,
                        &candidate_record,
                    )
                })
            };

            time_current();
            let exact = output(&current, &current_record);
            time_candidate();
            assert_eq!(
                output(&candidate, &candidate_record),
                exact,
                "capacity={capacity} visible={visible_rows} case={case}: untimed output"
            );
            for _ in 0..WARM_PAIRS {
                time_current();
                time_candidate();
            }
            let before = (0..SAMPLES).map(|_| time_current()).collect::<Vec<_>>();
            let candidate_samples = (0..SAMPLES).map(|_| time_candidate()).collect::<Vec<_>>();
            let after = (0..SAMPLES).map(|_| time_current()).collect::<Vec<_>>();
            assert_eq!(output(&current, &current_record), exact);
            assert_eq!(output(&candidate, &candidate_record), exact);
            candidate.validate_completed(&candidate_record).unwrap();
            let multigroup = candidate.multigroup.as_ref().unwrap();
            let candidate_invocations = 1 + WARM_PAIRS + SAMPLES;
            assert_eq!(
                multigroup.generation.next.get().unwrap().get(),
                candidate_invocations as u32 + 1
            );
            let state = read_i32(&multigroup.state);
            assert_eq!(
                state[0], candidate_invocations as i32,
                "last invocation generation"
            );
            assert_eq!(state[5], 0, "last invocation status");

            let before_gpu = before.iter().map(|sample| sample.0).collect::<Vec<_>>();
            let before_wall = before.iter().map(|sample| sample.1).collect::<Vec<_>>();
            let candidate_gpu = candidate_samples
                .iter()
                .map(|sample| sample.0)
                .collect::<Vec<_>>();
            let candidate_wall = candidate_samples
                .iter()
                .map(|sample| sample.1)
                .collect::<Vec<_>>();
            let after_gpu = after.iter().map(|sample| sample.0).collect::<Vec<_>>();
            let after_wall = after.iter().map(|sample| sample.1).collect::<Vec<_>>();
            let (before_gpu_median, _) = median_and_p95(&before_gpu);
            let (before_wall_median, _) = median_and_p95(&before_wall);
            let (candidate_gpu_median, candidate_gpu_p95) = median_and_p95(&candidate_gpu);
            let (candidate_wall_median, candidate_wall_p95) = median_and_p95(&candidate_wall);
            let (after_gpu_median, _) = median_and_p95(&after_gpu);
            let (after_wall_median, _) = median_and_p95(&after_wall);
            let gpu_drift = 2.0 * (before_gpu_median - after_gpu_median).abs()
                / (before_gpu_median + after_gpu_median);
            let wall_drift = 2.0 * (before_wall_median - after_wall_median).abs()
                / (before_wall_median + after_wall_median);
            let gpu_saving = before_gpu_median.min(after_gpu_median) - candidate_gpu_median;
            let wall_saving = before_wall_median.min(after_wall_median) - candidate_wall_median;
            let passed = gpu_drift <= 0.05
                && wall_drift <= 0.05
                && candidate_gpu_median <= MAX_MEDIAN_MS
                && candidate_gpu_p95 <= MAX_P95_MS
                && candidate_wall_median <= MAX_MEDIAN_MS
                && candidate_wall_p95 <= MAX_P95_MS
                && gpu_saving >= MIN_SAVING_MS
                && wall_saving >= MIN_SAVING_MS;
            eprintln!(
                "deepseek_v4 multigroup_integrated capacity={capacity} visible={visible_rows} case={case} current_before_gpu_median_ms={before_gpu_median:.6} current_before_wall_median_ms={before_wall_median:.6} candidate_gpu_median_ms={candidate_gpu_median:.6} candidate_gpu_p95_ms={candidate_gpu_p95:.6} candidate_wall_median_ms={candidate_wall_median:.6} candidate_wall_p95_ms={candidate_wall_p95:.6} current_after_gpu_median_ms={after_gpu_median:.6} current_after_wall_median_ms={after_wall_median:.6} gpu_drift={gpu_drift:.6} wall_drift={wall_drift:.6} faster_control_gpu_saving_ms={gpu_saving:.6} faster_control_wall_saving_ms={wall_saving:.6} pass={passed}"
            );
            eprintln!(
                "deepseek_v4 multigroup_integrated capacity={capacity} visible={visible_rows} case={case} current_before_gpu_ms={before_gpu:?} current_before_wall_ms={before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} current_after_gpu_ms={after_gpu:?} current_after_wall_ms={after_wall:?}"
            );
            assert!(
                passed,
                "capacity={capacity} visible={visible_rows} case={case}: integrated gate failed"
            );
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
#[ignore = "requires the current 97.05 GiB DS4 asset and a full-context session"]
fn current_asset_multigroup_selector_whole_token_gate() {
    const POSITION: u32 = 786_431;
    const FORWARD_LIMIT: usize = DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY;
    const TOKEN_ID: u32 = 35;
    const WARM_SAMPLES: usize = 2;
    const TIMED_SAMPLES: usize = 8;
    const EXPECTED_ELIGIBLE_LAYERS: u32 = 21;
    const MIN_WHOLE_TOKEN_SAVING_MS: f64 = 10.5;

    struct Evidence {
        profiles: Vec<DeepSeekV4WholeTokenProfile>,
        logits_bits: Vec<u32>,
        hidden_bits: Vec<u32>,
        causal_digest: [u8; 32],
        prefix_digest: [u8; 32],
        compatibility_digest: [u8; 32],
        committed_digest: [u8; 32],
        committed_len: usize,
        decision: DeepSeekV4DecisionTranscript,
        generation_deltas: Vec<u32>,
        after_session_bytes: u64,
        after_first_forward_bytes: u64,
    }

    fn digest_u32(values: &[u32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn median(samples: &[f64]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        }
    }

    fn selector_generation(session: &DeepSeekV4Session) -> u32 {
        session
            .sparse_csa
            .multigroup
            .as_ref()
            .expect("full-context session owns multi-group scratch")
            .generation
            .next
            .get()
            .expect("live gate does not exhaust selector generations")
            .get()
    }

    fn record_generation_delta(deltas: &mut Vec<u32>, before: u32, after: u32, experimental: bool) {
        let delta = after.checked_sub(before).expect("generation is monotonic");
        assert_eq!(
            delta,
            if experimental {
                EXPECTED_ELIGIBLE_LAYERS
            } else {
                0
            }
        );
        deltas.push(delta);
    }

    fn execute(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
        snapshot: &DeepSeekV4CausalSnapshot,
        experimental: bool,
    ) -> (DeepSeekV4MetalResidency, Evidence) {
        let mut session =
            DeepSeekV4Session::new_with_model_content_id(ctx, residency, model_content_id)
                .expect("construct live selector session");
        if experimental {
            session
                .enable_multigroup_selector_experiment()
                .expect("seal live selector experiment before restore");
        }
        session
            .restore_causal_snapshot(snapshot)
            .expect("restore live selector prefix");
        assert_eq!(session.next_position(), POSITION);
        let after_session_bytes = ctx.current_allocated_size();
        let mut after_first_forward_bytes = after_session_bytes;
        let mut generation_deltas = Vec::with_capacity(
            WARM_SAMPLES
                .checked_add(TIMED_SAMPLES)
                .and_then(|count| count.checked_add(1))
                .unwrap(),
        );

        for warm in 0..WARM_SAMPLES {
            session
                .restore_causal_snapshot(snapshot)
                .expect("restore warm live selector prefix");
            let before = selector_generation(&session);
            session
                .forward_token(ctx, TOKEN_ID)
                .expect("execute warm live selector token");
            let after = selector_generation(&session);
            record_generation_delta(&mut generation_deltas, before, after, experimental);
            if warm == 0 {
                after_first_forward_bytes = ctx.current_allocated_size();
            }
        }

        let mut profiles = Vec::with_capacity(TIMED_SAMPLES);
        let mut logits_bits = None;
        let mut hidden_bits = None;
        for _ in 0..TIMED_SAMPLES {
            session
                .restore_causal_snapshot(snapshot)
                .expect("restore timed live selector prefix");
            let before = selector_generation(&session);
            let profile = session
                .forward_token_whole_profiled(ctx, TOKEN_ID)
                .expect("profile live selector token");
            let after = selector_generation(&session);
            record_generation_delta(&mut generation_deltas, before, after, experimental);
            assert_eq!(profile.position, POSITION);
            let observed_logits = session
                .copy_logits_f32()
                .expect("copy live selector logits")
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let observed_hidden = host_read_f32(
                session
                    .final_normalized_hidden()
                    .expect("live selector final hidden is visible"),
                "live selector final normalized hidden",
            )
            .expect("copy live selector final hidden")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
            match &logits_bits {
                Some(expected) => assert_eq!(&observed_logits, expected),
                None => logits_bits = Some(observed_logits),
            }
            match &hidden_bits {
                Some(expected) => assert_eq!(&observed_hidden, expected),
                None => hidden_bits = Some(observed_hidden),
            }
            profiles.push(profile);
        }

        let committed_digest = digest_u32(session.committed_tokens());
        let committed_len = session.committed_tokens().len();
        let completed = session
            .capture_causal_snapshot()
            .expect("capture completed live selector state");
        assert_eq!(completed.next_position(), POSITION + 1);
        assert_eq!(
            completed.source_observation(),
            DeepSeekV4SnapshotObservation::Available
        );
        let causal_digest = *completed.causal_digest();
        let prefix_digest = *completed.prefix_digest();
        let compatibility_digest = *completed.compatibility_digest().as_bytes();
        drop(completed);

        session
            .restore_causal_snapshot(snapshot)
            .expect("restore decision-trace live selector prefix");
        session
            .arm_decision_transcript(POSITION)
            .expect("arm live selector decision trace");
        let before = selector_generation(&session);
        session
            .forward_token(ctx, TOKEN_ID)
            .expect("execute live selector decision trace");
        let after = selector_generation(&session);
        record_generation_delta(&mut generation_deltas, before, after, experimental);
        let decision = session
            .take_decision_transcript()
            .expect("take live selector decision trace");
        assert_eq!(decision.position, POSITION);
        assert_eq!(decision.csa_layer_count, EXPECTED_ELIGIBLE_LAYERS);

        let evidence = Evidence {
            profiles,
            logits_bits: logits_bits.unwrap(),
            hidden_bits: hidden_bits.unwrap(),
            causal_digest,
            prefix_digest,
            compatibility_digest,
            committed_digest,
            committed_len,
            decision,
            generation_deltas,
            after_session_bytes,
            after_first_forward_bytes,
        };
        (
            session
                .into_residency()
                .expect("recover exclusive DeepSeek V4 residency"),
            evidence,
        )
    }

    let model_path = std::env::var_os("DSV4_CURRENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
            )
        });
    assert!(model_path.exists(), "missing current DS4 model");
    let ctx = MetalContext::new().expect("create Metal context");
    let (gguf, model_content_id) = open_pinned_current_gguf(&model_path);
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, FORWARD_LIMIT)
        .expect("plan full-context current-asset session");
    assert_eq!(plan.session_capacity().csa_physical_rows(), 262_144);
    assert_eq!(plan.session_capacity().hca_physical_rows(), 8_192);
    let memory_plan = plan.memory_plan().clone();
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit full-context current-asset session");
    let before_residency_bytes = admitted.admission().signals.current_allocated_bytes;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .expect("realize current DS4 residency");
    let after_residency_bytes = realized.after_residency_bytes();
    let residency = realized.into_residency();

    let mut seed = DeepSeekV4Session::new_with_model_content_id(&ctx, residency, model_content_id)
        .expect("construct live selector snapshot seed");
    initialize_zero_synthetic_causal_state(&seed);
    seed.phase = DeepSeekV4SessionPhase::ReadyWithoutObservation {
        next_position: POSITION,
    };
    seed.committed_tokens
        .extend(std::iter::repeat_n(TOKEN_ID, POSITION as usize));
    let snapshot = seed
        .capture_causal_snapshot()
        .expect("capture live selector input snapshot");
    assert_eq!(snapshot.next_position(), POSITION);
    assert_eq!(snapshot.prefix_tokens().len(), POSITION as usize);
    let input_snapshot_payload_bytes = snapshot.payload_bytes();
    let input_snapshot_causal_digest = *snapshot.causal_digest();
    let input_snapshot_prefix_digest = *snapshot.prefix_digest();
    let residency = seed
        .into_residency()
        .expect("recover exclusive DeepSeek V4 residency");

    let (residency, current_before) = execute(&ctx, residency, model_content_id, &snapshot, false);
    let (residency, candidate) = execute(&ctx, residency, model_content_id, &snapshot, true);
    let (_residency, current_after) = execute(&ctx, residency, model_content_id, &snapshot, false);

    for control in [&current_before, &current_after] {
        assert_eq!(candidate.logits_bits, control.logits_bits);
        assert_eq!(candidate.hidden_bits, control.hidden_bits);
        assert_eq!(candidate.causal_digest, control.causal_digest);
        assert_eq!(candidate.prefix_digest, control.prefix_digest);
        assert_eq!(candidate.compatibility_digest, control.compatibility_digest);
        assert_eq!(candidate.committed_digest, control.committed_digest);
        assert_eq!(candidate.committed_len, control.committed_len);
        assert_eq!(candidate.decision, control.decision);
    }
    assert_eq!(current_before.logits_bits, current_after.logits_bits);
    assert_eq!(current_before.hidden_bits, current_after.hidden_bits);
    assert_eq!(current_before.causal_digest, current_after.causal_digest);
    assert_eq!(current_before.decision, current_after.decision);
    assert_eq!(candidate.committed_len, POSITION as usize + 1);
    assert!(
        candidate
            .generation_deltas
            .iter()
            .all(|&delta| delta == EXPECTED_ELIGIBLE_LAYERS)
    );
    for control in [&current_before, &current_after] {
        assert!(control.generation_deltas.iter().all(|&delta| delta == 0));
    }

    let current_before_gpu = current_before
        .profiles
        .iter()
        .map(|profile| profile.command_gpu_ms)
        .collect::<Vec<_>>();
    let current_before_wall = current_before
        .profiles
        .iter()
        .map(|profile| profile.forward_wall_ms)
        .collect::<Vec<_>>();
    let candidate_gpu = candidate
        .profiles
        .iter()
        .map(|profile| profile.command_gpu_ms)
        .collect::<Vec<_>>();
    let candidate_wall = candidate
        .profiles
        .iter()
        .map(|profile| profile.forward_wall_ms)
        .collect::<Vec<_>>();
    let current_after_gpu = current_after
        .profiles
        .iter()
        .map(|profile| profile.command_gpu_ms)
        .collect::<Vec<_>>();
    let current_after_wall = current_after
        .profiles
        .iter()
        .map(|profile| profile.forward_wall_ms)
        .collect::<Vec<_>>();
    let current_before_gpu_median = median(&current_before_gpu);
    let current_before_wall_median = median(&current_before_wall);
    let candidate_gpu_median = median(&candidate_gpu);
    let candidate_wall_median = median(&candidate_wall);
    let current_after_gpu_median = median(&current_after_gpu);
    let current_after_wall_median = median(&current_after_wall);
    let gpu_drift = 2.0 * (current_before_gpu_median - current_after_gpu_median).abs()
        / (current_before_gpu_median + current_after_gpu_median);
    let wall_drift = 2.0 * (current_before_wall_median - current_after_wall_median).abs()
        / (current_before_wall_median + current_after_wall_median);
    let gpu_saving = current_before_gpu_median.min(current_after_gpu_median) - candidate_gpu_median;
    let wall_saving =
        current_before_wall_median.min(current_after_wall_median) - candidate_wall_median;
    assert!(gpu_drift <= 0.05, "control GPU drift {gpu_drift:.6}");
    assert!(wall_drift <= 0.05, "control wall drift {wall_drift:.6}");
    assert!(
        gpu_saving >= MIN_WHOLE_TOKEN_SAVING_MS,
        "whole-token GPU saving {gpu_saving:.3} ms"
    );
    assert!(
        wall_saving >= MIN_WHOLE_TOKEN_SAVING_MS,
        "whole-token wall saving {wall_saving:.3} ms"
    );

    let memory = memory_plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes: current_before.after_session_bytes,
            after_first_forward_bytes: current_before.after_first_forward_bytes,
        })
        .expect("reconcile live selector memory");
    let logits_sha256 = digest_u32(&candidate.logits_bits);
    let hidden_sha256 = digest_u32(&candidate.hidden_bits);
    let decision_sha256: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&candidate.decision).unwrap()).into();
    let metallib_sha256 = deepseek_v4_diagnostics_metallib_sha256();
    eprintln!(
        "deepseek_v4 multigroup_live position={POSITION} visible_rows=196608 eligible_layers={EXPECTED_ELIGIBLE_LAYERS} current_before_gpu_median_ms={current_before_gpu_median:.6} candidate_gpu_median_ms={candidate_gpu_median:.6} current_after_gpu_median_ms={current_after_gpu_median:.6} gpu_saving_ms={gpu_saving:.6} gpu_drift={gpu_drift:.6} current_before_wall_median_ms={current_before_wall_median:.6} candidate_wall_median_ms={candidate_wall_median:.6} current_after_wall_median_ms={current_after_wall_median:.6} wall_saving_ms={wall_saving:.6} wall_drift={wall_drift:.6} pass=true"
    );
    eprintln!(
        "deepseek_v4 multigroup_live current_before_gpu_ms={current_before_gpu:?} current_before_wall_ms={current_before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} current_after_gpu_ms={current_after_gpu:?} current_after_wall_ms={current_after_wall:?}"
    );
    eprintln!(
        "deepseek_v4 multigroup_live model_path={} model_content_id={} current_census_sha256=cbfddbea4260cbaffb02429f0d9593d6e00ec08eb9a5b8d56ea83e8d22860889 metallib_sha256={} device_name={} device_registry_id={} input_snapshot_payload_bytes={input_snapshot_payload_bytes} input_snapshot_causal_digest={} input_snapshot_prefix_digest={} completed_causal_digest={} completed_prefix_digest={} compatibility_digest={} committed_sha256={} logits_sha256={} hidden_sha256={} decision_sha256={} candidate_generation_deltas={:?} memory_plan=({memory_plan}) memory_reconciliation=({memory})",
        model_path.display(),
        digest_hex(model_content_id.as_bytes()),
        digest_hex(&metallib_sha256),
        ctx.device.name(),
        ctx.device.registryID(),
        digest_hex(&input_snapshot_causal_digest),
        digest_hex(&input_snapshot_prefix_digest),
        digest_hex(&candidate.causal_digest),
        digest_hex(&candidate.prefix_digest),
        digest_hex(&candidate.compatibility_digest),
        digest_hex(&candidate.committed_digest),
        digest_hex(&logits_sha256),
        digest_hex(&hidden_sha256),
        digest_hex(&decision_sha256),
        candidate.generation_deltas,
    );
}

#[test]
#[ignore = "focused multi-group selector crossover map; run explicitly with --nocapture"]
fn profile_multigroup_selector_crossover() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const TOP_K: usize = 512;
    const GROUPS: usize = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    const SAMPLES: usize = 16;
    const CELLS: &[(usize, usize)] = &[
        (16_384, 16_384),
        (32_768, 32_768),
        (49_152, 49_152),
        (65_536, 65_536),
        (98_304, 98_304),
        (131_072, 131_072),
        (196_608, 196_608),
        (262_144, 262_144),
        (131_072, 65_536),
        (262_144, 65_536),
        (262_144, 131_072),
        (262_144, 196_608),
        (250_112, 196_608),
    ];

    fn timed_gpu_wall<F>(ctx: &MetalContext, encode: F) -> (f64, f64)
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(command.error().is_none(), "{:?}", command.error());
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
        assert!(wall_ms.is_finite() && wall_ms > 0.0);
        (gpu_ms, wall_ms)
    }

    fn median(samples: &[f64]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        }
    }

    for (cell_index, &(capacity, visible_rows)) in CELLS.iter().enumerate() {
        assert!(visible_rows <= capacity && visible_rows > TOP_K);
        for tied in [false, true] {
            let case = if tied { "tied" } else { "mixed" };
            let values = if tied {
                vec![0.0f32; capacity]
            } else {
                (0..capacity)
                    .map(|row| {
                        let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
                        bucket as f32 * 0.0003 - 1.1
                    })
                    .collect::<Vec<_>>()
            };
            let scores = offset_f32(&ctx, &values, vec![capacity as u64, 1]);
            let visible = offset_i32(&ctx, &[visible_rows as i32], vec![1]);
            let current_mask = MetalTensor::zeros_i32(&ctx, vec![capacity as u64, 1]).unwrap();
            let current_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
            let current_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let current_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let records = MetalTensor::zeros_i32(
                &ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                    GROUPS as u64,
                ],
            )
            .unwrap();
            let partition_plan = MetalTensor::zeros_i32(
                &ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                    GROUPS as u64,
                ],
            )
            .unwrap();
            let state = MetalTensor::zeros_i32(
                &ctx,
                vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
            )
            .unwrap();
            let private_mask =
                MetalTensor::zeros_dtype(&ctx, vec![capacity as u64, 1], GgmlType::I8).unwrap();
            let private_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
            let candidate_mask = MetalTensor::zeros_i32(&ctx, vec![capacity as u64, 1]).unwrap();
            let candidate_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
            let candidate_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let candidate_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
            let generation = std::cell::Cell::new(
                0x4300_0001u32
                    .wrapping_add((cell_index as u32) << 12)
                    .wrapping_add(u32::from(tied) << 11),
            );
            let next_generation = || {
                let current = generation.get();
                let next = current.wrapping_add(1);
                generation.set(if next == 0 { 1 } else { next });
                current
            };
            let time_current = || {
                timed_gpu_wall(&ctx, |encoder| {
                    encode_select_top_k_f32(
                        &ctx,
                        encoder,
                        &scores,
                        &visible,
                        &current_mask,
                        None,
                        &current_ids,
                        &current_count,
                        &current_status,
                        capacity,
                        visible_rows,
                        TOP_K,
                        1,
                    )
                })
            };
            let time_candidate = || {
                let invocation_generation = next_generation();
                timed_gpu_wall(&ctx, |encoder| {
                    encode_select_top_k_multigroup_full_f32(
                        &ctx,
                        encoder,
                        &scores,
                        &visible,
                        &records,
                        &partition_plan,
                        &state,
                        &private_mask,
                        &private_ids,
                        &candidate_mask,
                        &candidate_ids,
                        &candidate_count,
                        &candidate_status,
                        capacity,
                        TOP_K,
                        invocation_generation,
                        None,
                        false,
                    )
                })
            };

            time_current();
            let exact = (
                read_i32(&current_mask),
                read_i32(&current_ids),
                read_i32(&current_count),
                read_i32(&current_status),
            );
            time_candidate();
            assert_eq!(
                (
                    read_i32(&candidate_mask),
                    read_i32(&candidate_ids),
                    read_i32(&candidate_count),
                    read_i32(&candidate_status),
                ),
                exact,
                "capacity={capacity} visible={visible_rows} case={case}: output"
            );
            for _ in 0..4 {
                time_current();
                time_candidate();
            }
            let before = (0..SAMPLES).map(|_| time_current()).collect::<Vec<_>>();
            let candidate = (0..SAMPLES).map(|_| time_candidate()).collect::<Vec<_>>();
            let after = (0..SAMPLES).map(|_| time_current()).collect::<Vec<_>>();
            assert_eq!(
                (
                    read_i32(&candidate_mask),
                    read_i32(&candidate_ids),
                    read_i32(&candidate_count),
                    read_i32(&candidate_status),
                ),
                exact,
                "capacity={capacity} visible={visible_rows} case={case}: timed output"
            );
            let before_gpu = before.iter().map(|sample| sample.0).collect::<Vec<_>>();
            let before_wall = before.iter().map(|sample| sample.1).collect::<Vec<_>>();
            let candidate_gpu = candidate.iter().map(|sample| sample.0).collect::<Vec<_>>();
            let candidate_wall = candidate.iter().map(|sample| sample.1).collect::<Vec<_>>();
            let after_gpu = after.iter().map(|sample| sample.0).collect::<Vec<_>>();
            let after_wall = after.iter().map(|sample| sample.1).collect::<Vec<_>>();
            let before_gpu_median = median(&before_gpu);
            let before_wall_median = median(&before_wall);
            let candidate_gpu_median = median(&candidate_gpu);
            let candidate_wall_median = median(&candidate_wall);
            let after_gpu_median = median(&after_gpu);
            let after_wall_median = median(&after_wall);
            let gpu_drift = 2.0 * (before_gpu_median - after_gpu_median).abs()
                / (before_gpu_median + after_gpu_median);
            let wall_drift = 2.0 * (before_wall_median - after_wall_median).abs()
                / (before_wall_median + after_wall_median);
            let gpu_saving = before_gpu_median.min(after_gpu_median) - candidate_gpu_median;
            let wall_saving = before_wall_median.min(after_wall_median) - candidate_wall_median;
            let stable = gpu_drift <= 0.05 && wall_drift <= 0.05;
            eprintln!(
                "deepseek_v4 multigroup_crossover capacity={capacity} visible={visible_rows} case={case} current_gpu_ms={:.6} candidate_gpu_ms={candidate_gpu_median:.6} current_wall_ms={:.6} candidate_wall_ms={candidate_wall_median:.6} gpu_saving_ms={gpu_saving:.6} wall_saving_ms={wall_saving:.6} gpu_drift={gpu_drift:.6} wall_drift={wall_drift:.6} stable={stable} win={}",
                before_gpu_median.min(after_gpu_median),
                before_wall_median.min(after_wall_median),
                gpu_saving > 0.0 && wall_saving > 0.0,
            );
            eprintln!(
                "deepseek_v4 multigroup_crossover capacity={capacity} visible={visible_rows} case={case} current_before_gpu_ms={before_gpu:?} current_before_wall_ms={before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} current_after_gpu_ms={after_gpu:?} current_after_wall_ms={after_wall:?}"
            );
        }
    }
}

#[test]
#[ignore = "focused terminal multi-group threshold ceiling; run explicitly with --nocapture"]
fn profile_multigroup_selector_threshold_ceiling() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 262_144;
    const TOP_K: usize = 512;
    const SAMPLES: usize = 24;
    const MAX_MEDIAN_MS: f64 = 1.00;
    const MAX_P95_MS: f64 = 1.05;
    const MIN_SAVING_MS: f64 = 0.70;

    fn timed_gpu<F>(ctx: &MetalContext, encode: F) -> f64
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());
        let elapsed = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed.is_finite() && elapsed > 0.0);
        elapsed
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64, Vec<f64>) {
        let raw = samples.clone();
        samples.sort_by(f64::total_cmp);
        let median = if samples.len().is_multiple_of(2) {
            (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) * 0.5
        } else {
            samples[samples.len() / 2]
        };
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95, raw)
    }

    let mixed_values = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let tied_values = vec![0.0f32; CAPACITY];
    let mixed_scores = offset_f32(&ctx, &mixed_values, vec![CAPACITY as u64, 1]);
    let tied_scores = offset_f32(&ctx, &tied_values, vec![CAPACITY as u64, 1]);
    let visible = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();

    let time_current = |scores: &MetalTensor| {
        timed_gpu(&ctx, |encoder| {
            encode_select_top_k_f32(
                &ctx, encoder, scores, &visible, &mask, None, &ids, &count, &status, CAPACITY,
                CAPACITY, TOP_K, 1,
            )
        })
    };

    let mut any_geometry_passed = false;
    for groups in [32usize, 64, 80] {
        let generation = 0x2468_0000 | groups as u32;
        let records = MetalTensor::zeros_i32(
            &ctx,
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                groups as u64,
            ],
        )
        .unwrap();
        let partition_plan = MetalTensor::zeros_i32(
            &ctx,
            vec![
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                groups as u64,
            ],
        )
        .unwrap();
        let state = MetalTensor::zeros_i32(
            &ctx,
            vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        )
        .unwrap();
        let time_candidate = |scores: &MetalTensor| {
            timed_gpu(&ctx, |encoder| {
                encode_select_top_k_multigroup_threshold_f32(
                    &ctx,
                    encoder,
                    scores,
                    &visible,
                    &records,
                    &partition_plan,
                    &state,
                    CAPACITY,
                    TOP_K,
                    groups,
                    generation,
                    None,
                )
            })
        };

        let assert_candidate = |label: &str, values: &[f32], scores: &MetalTensor| {
            time_candidate(scores);
            let expected = multigroup_selector_threshold_oracle(
                values,
                CAPACITY,
                CAPACITY as i32,
                TOP_K,
                groups,
            );
            let state_words = read_i32(&state)
                .into_iter()
                .map(|word| word as u32)
                .collect::<Vec<_>>();
            assert_eq!(state_words[5], 0, "{label} status groups={groups}");
            assert_eq!(
                state_words[6], expected.threshold_key,
                "{label} key groups={groups}"
            );
            assert_eq!(
                state_words[7], expected.threshold_take,
                "{label} take groups={groups}"
            );
            assert_eq!(
                state_words[8], TOP_K as u32,
                "{label} count groups={groups}"
            );
            assert_eq!(
                read_i32(&partition_plan)
                    .into_iter()
                    .map(|word| word as u32)
                    .collect::<Vec<_>>(),
                expected.partition_plan,
                "{label} partitions groups={groups}"
            );
        };
        assert_candidate("mixed", &mixed_values, &mixed_scores);
        assert_candidate("tied", &tied_values, &tied_scores);

        let mut geometry_passed = true;
        for (label, values, scores) in [
            ("mixed", mixed_values.as_slice(), &mixed_scores),
            ("tied", tied_values.as_slice(), &tied_scores),
        ] {
            let expected = multigroup_selector_threshold_oracle(
                values,
                CAPACITY,
                CAPACITY as i32,
                TOP_K,
                groups,
            );
            assert_eq!(expected.status, 0, "{label} oracle status groups={groups}");
            let assert_candidate_snapshot =
                |arm: &str, snapshot: &(Vec<i32>, Vec<i32>, Vec<i32>)| {
                    let state_words = snapshot
                        .2
                        .iter()
                        .map(|&word| word as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        state_words[0], generation,
                        "{label} {arm} state generation groups={groups}"
                    );
                    assert_eq!(
                        state_words[1], 7,
                        "{label} {arm} state digit groups={groups}"
                    );
                    assert_eq!(
                        state_words[2], expected.threshold_key,
                        "{label} {arm} state prefix groups={groups}"
                    );
                    assert_eq!(
                        state_words[3],
                        u32::MAX,
                        "{label} {arm} state prefix mask groups={groups}"
                    );
                    assert_eq!(
                        state_words[4], expected.threshold_take,
                        "{label} {arm} state rank groups={groups}"
                    );
                    assert_eq!(
                        state_words[5], expected.status,
                        "{label} {arm} state status groups={groups}"
                    );
                    assert_eq!(
                        state_words[6], expected.threshold_key,
                        "{label} {arm} state threshold groups={groups}"
                    );
                    assert_eq!(
                        state_words[7], expected.threshold_take,
                        "{label} {arm} state take groups={groups}"
                    );
                    assert_eq!(
                        state_words[8], expected.selected_count,
                        "{label} {arm} state count groups={groups}"
                    );
                    assert_eq!(
                        state_words[9],
                        deepseek_v4_multigroup_selector_state_completion(generation, 7),
                        "{label} {arm} state completion groups={groups}"
                    );
                    assert_eq!(
                        snapshot
                            .1
                            .iter()
                            .map(|&word| word as u32)
                            .collect::<Vec<_>>(),
                        expected.partition_plan,
                        "{label} {arm} partition plan groups={groups}"
                    );
                    for group in 0..groups {
                        let base = group * DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS;
                        assert_eq!(
                            snapshot.0[base] as u32, generation,
                            "{label} {arm} record generation group={group} groups={groups}"
                        );
                        assert_eq!(
                            snapshot.0[base + 1],
                            7,
                            "{label} {arm} record digit group={group} groups={groups}"
                        );
                        assert_eq!(
                            snapshot.0[base + 2],
                            0,
                            "{label} {arm} record error group={group} groups={groups}"
                        );
                        assert_eq!(
                            snapshot.0[base + 3] as u32,
                            deepseek_v4_multigroup_selector_record_completion(generation, 7, group,),
                            "{label} {arm} record completion group={group} groups={groups}"
                        );
                    }
                };
            let assert_current_output = |arm: &str| {
                assert_eq!(
                    read_i32(&status),
                    [expected.status as i32],
                    "{label} {arm} current status groups={groups}"
                );
                assert_eq!(
                    read_i32(&count),
                    [expected.selected_count as i32],
                    "{label} {arm} current count groups={groups}"
                );
                let selected_ids = read_i32(&ids);
                let selected_keys = selected_ids
                    .iter()
                    .take(expected.selected_count as usize)
                    .map(|&id| {
                        assert!(
                            (0..CAPACITY as i32).contains(&id),
                            "{label} {arm} current id {id} groups={groups}"
                        );
                        deployed_selector_order_key(values[id as usize])
                    })
                    .collect::<Vec<_>>();
                let threshold = *selected_keys.iter().min().unwrap();
                let take = selected_keys
                    .iter()
                    .filter(|&&key| key == threshold)
                    .count();
                assert_eq!(
                    threshold, expected.threshold_key,
                    "{label} {arm} current threshold groups={groups}"
                );
                assert_eq!(
                    take, expected.threshold_take as usize,
                    "{label} {arm} current take groups={groups}"
                );
            };
            for _ in 0..5 {
                time_current(scores);
                time_candidate(scores);
            }
            let (before_median, _, before_samples) =
                median_and_p95((0..SAMPLES).map(|_| time_current(scores)).collect());
            assert_current_output("before");
            let exact_before = (
                read_i32(&records),
                read_i32(&partition_plan),
                read_i32(&state),
            );
            assert_candidate_snapshot("before", &exact_before);
            let (candidate_median, candidate_p95, candidate_samples) =
                median_and_p95((0..SAMPLES).map(|_| time_candidate(scores)).collect());
            let exact_after = (
                read_i32(&records),
                read_i32(&partition_plan),
                read_i32(&state),
            );
            assert_candidate_snapshot("after", &exact_after);
            assert_eq!(
                exact_after, exact_before,
                "{label} record drift groups={groups}"
            );
            let (after_median, _, after_samples) =
                median_and_p95((0..SAMPLES).map(|_| time_current(scores)).collect());
            assert_current_output("after");
            let control_drift =
                2.0 * (before_median - after_median).abs() / (before_median + after_median);
            let faster_control = before_median.min(after_median);
            let saving = faster_control - candidate_median;
            let passed = control_drift <= 0.05
                && candidate_median <= MAX_MEDIAN_MS
                && candidate_p95 <= MAX_P95_MS
                && saving >= MIN_SAVING_MS;
            geometry_passed &= passed;
            eprintln!(
                "deepseek_v4 multigroup_threshold groups={groups} case={label} current_before_median_ms={before_median:.6} candidate_median_ms={candidate_median:.6} candidate_p95_ms={candidate_p95:.6} current_after_median_ms={after_median:.6} control_drift={control_drift:.6} faster_control_saving_ms={saving:.6} pass={passed}"
            );
            eprintln!(
                "deepseek_v4 multigroup_threshold groups={groups} case={label} current_before_samples_ms={before_samples:?} candidate_samples_ms={candidate_samples:?} current_after_samples_ms={after_samples:?}"
            );
        }
        any_geometry_passed |= geometry_passed;
    }
    assert!(
        any_geometry_passed,
        "no frozen multi-group threshold geometry cleared both mixed and tied gates"
    );
}

#[test]
fn parallel_selector_pipeline_requires_32_lane_simdgroups() {
    validate_parallel_selector_pipeline(32, 256).unwrap();
    let error = validate_parallel_selector_pipeline(16, 256).unwrap_err();
    assert!(error.to_string().contains("requires 32-lane SIMD groups"));
    let error = validate_parallel_selector_pipeline(64, 1_024).unwrap_err();
    assert!(error.to_string().contains("pipeline reports 64"));
    let error = validate_parallel_selector_pipeline(32, 255).unwrap_err();
    assert!(error.to_string().contains("255 threads, requires 256"));
}

#[test]
fn selector_rejects_u32_buffer_offset_overflow_before_binding() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let f32_tensor = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let i32_tensor = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let error = encode_select_top_k_f32(
        &ctx,
        &encoder,
        &f32_tensor,
        &i32_tensor,
        &i32_tensor,
        None,
        &i32_tensor,
        &i32_tensor,
        &i32_tensor,
        u32::MAX as usize,
        1,
        1,
        2,
    )
    .unwrap_err();
    encoder.end();
    assert!(error.to_string().contains("buffer offsets exceed u32"));
}

#[test]
fn parallel_selector_flushes_subnormals_and_ties_signed_zero_by_row() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 1_025;
    const TOP_K: usize = 512;
    let mut score_values = vec![-1.0f32; CAPACITY];
    score_values[..510].fill(1.0);
    score_values[511] = f32::from_bits(0x8000_0001);
    score_values[512] = 0.0;
    score_values[513] = -0.0;
    score_values[600] = f32::from_bits(1);
    let ieee_ranked = top_k_indices(&score_values, TOP_K).unwrap();
    assert_eq!(&ieee_ranked[510..], &[600, 512]);
    let expected_ranked = deployed_selector_top_k_indices(&score_values, TOP_K);
    assert_eq!(&expected_ranked[510..], &[511, 512]);
    let mut expected_cache_order = expected_ranked.clone();
    expected_cache_order.sort_unstable();

    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, 1]);
    let visible_counts = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let ranked = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let cache_order = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &mask,
        Some(&ranked),
        &cache_order,
        &counts,
        &status,
        CAPACITY,
        CAPACITY,
        TOP_K,
        1,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(read_i32(&status), [0]);
    assert_eq!(read_i32(&counts), [TOP_K as i32]);
    assert_eq!(
        read_i32(&ranked),
        expected_ranked
            .iter()
            .map(|&row| row as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_i32(&cache_order),
        expected_cache_order
            .iter()
            .map(|&row| row as i32)
            .collect::<Vec<_>>()
    );
    let actual_mask = read_i32(&mask);
    for (row, &actual) in actual_mask.iter().enumerate().take(CAPACITY) {
        assert_eq!(actual, i32::from(expected_cache_order.contains(&row)));
    }
}

#[test]
fn batched_hadamard_and_stable_top512_match_cpu_contracts() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const ROWS: usize = 64;
    const CAPACITY: usize = 768;
    const VISIBLE: usize = 513;
    const TOP_K: usize = 512;
    const QUERIES: usize = 3;

    let values = (0..ROWS * 128)
        .map(|index| {
            let row = index / 128;
            let dimension = index % 128;
            ((row * 31 + dimension * 17 + row / 3) % 127) as f32 * 0.0061 - 0.37
        })
        .collect::<Vec<_>>();
    let mut expected = values.clone();
    for row in expected.chunks_exact_mut(128) {
        hadamard_128_in_place(row).unwrap();
    }
    let transformed = offset_f32(&ctx, &values, vec![128, ROWS as u64]);

    let mut score_values = vec![23.0f32; CAPACITY * QUERIES];
    score_values[..VISIBLE].fill(1.0);
    for row in 0..CAPACITY {
        score_values[CAPACITY + row] = row as f32;
    }
    score_values[2 * CAPACITY..2 * CAPACITY + VISIBLE].fill(1.0);
    score_values[2 * CAPACITY + 17] = f32::NAN;
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, QUERIES as u64]);
    let visible_counts = offset_i32(
        &ctx,
        &[VISIBLE as i32, CAPACITY as i32, VISIBLE as i32],
        vec![QUERIES as u64],
    );
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let ranked = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let cache_order = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_hadamard_128_rows_in_place(&ctx, &encoder, &transformed, ROWS).unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &mask,
        Some(&ranked),
        &cache_order,
        &counts,
        &status,
        CAPACITY,
        CAPACITY,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "batched indexer command failed: {:?}",
        command.error()
    );

    assert_close(
        "batched Hadamard-128",
        &read_f32(&transformed),
        &expected,
        2e-6,
    );
    assert_eq!(read_i32(&status), vec![0, 0, 2]);
    assert_eq!(read_i32(&counts), vec![TOP_K as i32; QUERIES]);
    let first_ids = (0..TOP_K as i32).collect::<Vec<_>>();
    let full_ranked = (CAPACITY - TOP_K..CAPACITY)
        .rev()
        .map(|row| row as i32)
        .collect::<Vec<_>>();
    let full_cache_order = (CAPACITY - TOP_K..CAPACITY)
        .map(|row| row as i32)
        .collect::<Vec<_>>();
    let ranked = read_i32(&ranked);
    let cache_order = read_i32(&cache_order);
    assert_eq!(&ranked[..TOP_K], &first_ids);
    assert_eq!(&cache_order[..TOP_K], &first_ids);
    assert_eq!(&ranked[TOP_K..2 * TOP_K], &full_ranked);
    assert_eq!(&cache_order[TOP_K..2 * TOP_K], &full_cache_order);
    assert_eq!(&ranked[2 * TOP_K..], &first_ids);
    assert_eq!(&cache_order[2 * TOP_K..], &first_ids);
    let mask = read_i32(&mask);
    let first = &mask[..CAPACITY];
    assert!(first[..TOP_K].iter().all(|&selected| selected == 1));
    assert!(first[TOP_K..].iter().all(|&selected| selected == 0));
    let full = &mask[CAPACITY..2 * CAPACITY];
    assert!(
        full[..CAPACITY - TOP_K]
            .iter()
            .all(|&selected| selected == 0)
    );
    assert!(
        full[CAPACITY - TOP_K..]
            .iter()
            .all(|&selected| selected == 1)
    );
    let invalid = &mask[2 * CAPACITY..];
    assert!(invalid[..TOP_K].iter().all(|&selected| selected == 1));
    assert!(invalid[TOP_K..].iter().all(|&selected| selected == 0));
}

#[test]
fn radix_top512_matches_mixed_packed_cpu_contracts_repeatably() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 1_537;
    const TOP_K: usize = 512;
    const QUERIES: usize = 16;
    let mut score_values = vec![0.0f32; CAPACITY * QUERIES];
    fn query_scores(scores: &mut [f32], query: usize, capacity: usize) -> &mut [f32] {
        &mut scores[query * capacity..(query + 1) * capacity]
    }
    query_scores(&mut score_values, 0, CAPACITY).fill(1.0);
    for row in 600..900 {
        query_scores(&mut score_values, 1, CAPACITY)[row] = 2.0;
    }
    for row in 0..600 {
        query_scores(&mut score_values, 1, CAPACITY)[row] = 1.0;
    }
    for row in 0..CAPACITY {
        let bucket = (row * 73 + row / 11) % 503;
        query_scores(&mut score_values, 2, CAPACITY)[row] = bucket as f32 * 0.003 - 0.7;
        query_scores(&mut score_values, 3, CAPACITY)[row] = row as f32;
        query_scores(&mut score_values, 4, CAPACITY)[row] = -(row as f32);
    }
    query_scores(&mut score_values, 5, CAPACITY).fill(0.5);
    query_scores(&mut score_values, 5, CAPACITY)[CAPACITY - 1] = f32::NAN;
    query_scores(&mut score_values, 6, CAPACITY).fill(0.5);
    query_scores(&mut score_values, 6, CAPACITY)[CAPACITY / 2] = f32::INFINITY;
    query_scores(&mut score_values, 7, CAPACITY).fill(0.5);
    query_scores(&mut score_values, 7, CAPACITY)[17] = f32::NEG_INFINITY;
    for row in 0..CAPACITY {
        query_scores(&mut score_values, 8, CAPACITY)[row] = (row % 97) as f32 - 48.0;
    }
    query_scores(&mut score_values, 8, CAPACITY)[1_024] = f32::NAN;
    query_scores(&mut score_values, 9, CAPACITY).fill(f32::NAN);
    query_scores(&mut score_values, 10, CAPACITY).fill(f32::INFINITY);
    query_scores(&mut score_values, 11, CAPACITY).fill(f32::NEG_INFINITY);
    for (query, tie_count, greater_count) in
        [(12, 2, 511), (13, 511, 256), (14, 512, 256), (15, 513, 0)]
    {
        query_scores(&mut score_values, query, CAPACITY).fill(-1.0);
        query_scores(&mut score_values, query, CAPACITY)[..tie_count].fill(1.0);
        query_scores(&mut score_values, query, CAPACITY)[CAPACITY - greater_count..].fill(2.0);
    }

    let visible_values = [
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        1_024,
        -1,
        0,
        CAPACITY as i32 + 1,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
        CAPACITY as i32,
    ];
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, QUERIES as u64]);
    let visible_counts = offset_i32(&ctx, &visible_values, vec![QUERIES as u64]);
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let ranked = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let cache_order = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let run = |radix4| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_select_top_k_f32_with_policy(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &mask,
            Some(&ranked),
            &cache_order,
            &counts,
            &status,
            CAPACITY,
            CAPACITY,
            TOP_K,
            QUERIES,
            DeepSeekV4SelectorDispatchPolicy::Production,
            radix4,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (
            read_i32(&mask),
            read_i32(&ranked),
            read_i32(&cache_order),
            read_i32(&counts),
            read_i32(&status),
        )
    };
    let first = run(false);
    let second = run(false);
    let radix4 = run(true);
    assert_eq!(second, first);
    assert_eq!(radix4, first);
    let maskless_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let maskless_counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let maskless_status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_radix4_ids_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &maskless_ids,
        &maskless_counts,
        &maskless_status,
        CAPACITY,
        CAPACITY,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(read_i32(&maskless_ids), radix4.2);
    assert_eq!(read_i32(&maskless_counts), radix4.3);
    assert_eq!(read_i32(&maskless_status), radix4.4);
    let (actual_mask, actual_ranked, actual_cache_order, actual_counts, actual_status) = first;

    for query in 0..QUERIES {
        let visible = visible_values[query];
        let expected_status = if visible <= 0 || visible as usize > CAPACITY {
            1
        } else if score_values[query * CAPACITY..query * CAPACITY + visible as usize]
            .iter()
            .any(|score| !score.is_finite())
        {
            2
        } else {
            0
        };
        let expected_count = if expected_status == 1 {
            0
        } else {
            TOP_K.min(visible as usize)
        };
        let expected_ranked = if expected_status == 0 {
            deployed_selector_top_k_indices(
                &score_values[query * CAPACITY..query * CAPACITY + visible as usize],
                TOP_K,
            )
        } else {
            (0..expected_count).collect::<Vec<_>>()
        };
        let mut expected_cache_order = expected_ranked.clone();
        expected_cache_order.sort_unstable();
        let ranked_slice = &actual_ranked[query * TOP_K..(query + 1) * TOP_K];
        let cache_slice = &actual_cache_order[query * TOP_K..(query + 1) * TOP_K];
        assert_eq!(
            actual_status[query], expected_status,
            "query {query} status"
        );
        assert_eq!(
            actual_counts[query], expected_count as i32,
            "query {query} count"
        );
        assert_eq!(
            &ranked_slice[..expected_count],
            &expected_ranked
                .iter()
                .map(|&row| row as i32)
                .collect::<Vec<_>>(),
            "query {query} ranked IDs"
        );
        assert_eq!(
            &cache_slice[..expected_count],
            &expected_cache_order
                .iter()
                .map(|&row| row as i32)
                .collect::<Vec<_>>(),
            "query {query} cache-order IDs"
        );
        assert!(ranked_slice[expected_count..].iter().all(|&id| id == -1));
        assert!(cache_slice[expected_count..].iter().all(|&id| id == -1));
        let mask_slice = &actual_mask[query * CAPACITY..(query + 1) * CAPACITY];
        for (row, &selected) in mask_slice.iter().enumerate() {
            assert_eq!(
                selected,
                i32::from(expected_cache_order.binary_search(&row).is_ok()),
                "query {query} row {row} mask"
            );
        }
    }
}

#[test]
fn shallow_scalar_parallel_and_radix_selectors_match_cpu_contracts() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 1_024;
    const TOP_K: usize = 512;
    const QUERIES: usize = 12;
    let visible_values = [
        513i32, 520, 544, 576, 640, 768, 896, 1_024, 513, 640, 896, 1_024,
    ];
    let mut score_values = vec![-3.0f32; CAPACITY * QUERIES];
    fn query_scores(scores: &mut [f32], query: usize) -> &mut [f32] {
        &mut scores[query * CAPACITY..(query + 1) * CAPACITY]
    }

    query_scores(&mut score_values, 0).fill(1.0);
    for row in 0..CAPACITY {
        query_scores(&mut score_values, 1)[row] = row as f32;
        query_scores(&mut score_values, 2)[row] = -(row as f32);
        let bucket = (row * 193 + row / 7 + row / 503) % 509;
        query_scores(&mut score_values, 3)[row] = bucket as f32 * 0.003 - 0.7;
    }
    query_scores(&mut score_values, 4)[..511].fill(2.0);
    query_scores(&mut score_values, 4)[511..].fill(1.0);
    query_scores(&mut score_values, 5)[..507].fill(2.0);
    for (row, bits) in [
        (507, 0x0080_0000),
        (508, 0x0000_0000),
        (509, 0x8000_0000),
        (510, 0x007f_ffff),
        (511, 0x807f_ffff),
        (512, 0x0000_0001),
        (513, 0x8000_0001),
    ] {
        query_scores(&mut score_values, 5)[row] = f32::from_bits(bits);
    }
    query_scores(&mut score_values, 6).fill(0.5);
    query_scores(&mut score_values, 6)[700] = f32::NAN;
    query_scores(&mut score_values, 7).fill(0.5);
    query_scores(&mut score_values, 7)[900] = f32::INFINITY;
    for row in 0..CAPACITY {
        query_scores(&mut score_values, 8)[row] = ((row * 17) % 257) as f32 - 128.0;
    }
    query_scores(&mut score_values, 8)[700] = f32::NAN;
    query_scores(&mut score_values, 9).fill(0.5);
    query_scores(&mut score_values, 9)[100] = f32::NEG_INFINITY;
    query_scores(&mut score_values, 10)[..600].fill(1.0);
    query_scores(&mut score_values, 10)[600..].fill(0.0);
    for row in 0..CAPACITY {
        let bucket = (row * 73 + row / 11) % 503;
        query_scores(&mut score_values, 11)[row] = bucket as f32 * 0.005 - 1.1;
    }

    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, QUERIES as u64]);
    let visible_counts = offset_i32(&ctx, &visible_values, vec![QUERIES as u64]);
    let allocate_outputs = || {
        (
            MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
        )
    };
    let scalar = allocate_outputs();
    let bitwise = allocate_outputs();
    let radix4 = allocate_outputs();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (policy, use_radix4, outputs) in [
        (
            DeepSeekV4SelectorDispatchPolicy::ScalarOracle,
            false,
            &scalar,
        ),
        (DeepSeekV4SelectorDispatchPolicy::Parallel, false, &bitwise),
        (DeepSeekV4SelectorDispatchPolicy::Parallel, true, &radix4),
    ] {
        encode_select_top_k_f32_with_policy(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &outputs.0,
            Some(&outputs.1),
            &outputs.2,
            &outputs.3,
            &outputs.4,
            CAPACITY,
            CAPACITY,
            TOP_K,
            QUERIES,
            policy,
            use_radix4,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let read_outputs = |outputs: &(
        MetalTensor,
        MetalTensor,
        MetalTensor,
        MetalTensor,
        MetalTensor,
    )| {
        (
            read_i32(&outputs.0),
            read_i32(&outputs.1),
            read_i32(&outputs.2),
            read_i32(&outputs.3),
            read_i32(&outputs.4),
        )
    };
    let scalar = read_outputs(&scalar);
    let bitwise = read_outputs(&bitwise);
    let radix4 = read_outputs(&radix4);
    assert_eq!(bitwise, scalar, "parallel bitwise selector");
    assert_eq!(radix4, scalar, "parallel radix-4 selector");

    for (query, &visible) in visible_values.iter().enumerate() {
        let visible = visible as usize;
        let scores = &score_values[query * CAPACITY..query * CAPACITY + visible];
        let non_finite = scores.iter().any(|score| !score.is_finite());
        let expected_ranked = if non_finite {
            (0..TOP_K).collect::<Vec<_>>()
        } else {
            deployed_selector_top_k_indices(scores, TOP_K)
        };
        let mut expected_cache_order = expected_ranked.clone();
        expected_cache_order.sort_unstable();
        assert_eq!(scalar.3[query], TOP_K as i32, "query {query} count");
        assert_eq!(
            scalar.4[query],
            i32::from(non_finite) * 2,
            "query {query} status"
        );
        assert_eq!(
            &scalar.1[query * TOP_K..(query + 1) * TOP_K],
            expected_ranked
                .iter()
                .map(|&row| row as i32)
                .collect::<Vec<_>>(),
            "query {query} ranked IDs"
        );
        assert_eq!(
            &scalar.2[query * TOP_K..(query + 1) * TOP_K],
            expected_cache_order
                .iter()
                .map(|&row| row as i32)
                .collect::<Vec<_>>(),
            "query {query} cache-order IDs"
        );
        let mask = &scalar.0[query * CAPACITY..(query + 1) * CAPACITY];
        for (row, &selected) in mask.iter().enumerate() {
            assert_eq!(
                selected,
                i32::from(expected_cache_order.binary_search(&row).is_ok()),
                "query {query} row {row} mask"
            );
        }
    }
}

#[test]
fn packed_publication_cadence_uses_the_same_scalar_and_production_selection() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 514;
    const TOP_K: usize = 512;
    const QUERIES: usize = 5;
    let visible_values = [513i32, 513, 513, 513, 514];
    let score_values = (0..CAPACITY * QUERIES)
        .map(|index| {
            let query = index / CAPACITY;
            let row = index % CAPACITY;
            ((row * 37 + query * 101 + row / 13) % 257) as f32 * 0.007 - 0.8
        })
        .collect::<Vec<_>>();
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, QUERIES as u64]);
    let visible_counts = offset_i32(&ctx, &visible_values, vec![QUERIES as u64]);
    let allocate_outputs = || {
        (
            MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
        )
    };
    let scalar = allocate_outputs();
    let production = allocate_outputs();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (policy, outputs) in [
        (DeepSeekV4SelectorDispatchPolicy::ScalarOracle, &scalar),
        (DeepSeekV4SelectorDispatchPolicy::Production, &production),
    ] {
        encode_select_top_k_f32_with_policy(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &outputs.0,
            None,
            &outputs.1,
            &outputs.2,
            &outputs.3,
            CAPACITY,
            CAPACITY,
            TOP_K,
            QUERIES,
            policy,
            true,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    for (label, scalar, production) in [
        ("mask", &scalar.0, &production.0),
        ("cache order", &scalar.1, &production.1),
        ("count", &scalar.2, &production.2),
        ("status", &scalar.3, &production.3),
    ] {
        assert_eq!(read_i32(production), read_i32(scalar), "{label}");
    }
}

#[test]
fn scalar_parallel_and_radix_top512_are_bit_identical_at_1024() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 1_025;
    const VISIBLE: usize = 1_024;
    const TOP_K: usize = 512;
    let mut score_values = vec![-1.0f32; CAPACITY];
    score_values[..507].fill(1.0);
    score_values[600] = 0.0;
    score_values[601] = -0.0;
    score_values[602] = f32::from_bits(0x007f_ffff);
    score_values[603] = f32::from_bits(0x807f_ffff);
    score_values[604] = f32::from_bits(1);
    score_values[605] = f32::from_bits(0x8000_0001);
    score_values[900] = f32::from_bits(0x0080_0000);
    score_values[901] = f32::from_bits(0x8080_0000);
    score_values[1_024] = f32::NAN;
    let expected_ranked = deployed_selector_top_k_indices(&score_values[..VISIBLE], TOP_K);
    assert_eq!(&expected_ranked[507..], &[900, 600, 601, 602, 603]);
    let mut expected_cache_order = expected_ranked.clone();
    expected_cache_order.sort_unstable();
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, 1]);
    let visible_counts = offset_i32(&ctx, &[VISIBLE as i32], vec![1]);
    let allocate_outputs = || {
        (
            MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![1]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![1]).unwrap(),
        )
    };
    let scalar = allocate_outputs();
    let radix = allocate_outputs();
    let radix4 = allocate_outputs();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (dispatch_policy, use_radix4, outputs) in [
        (
            DeepSeekV4SelectorDispatchPolicy::ScalarOracle,
            false,
            &scalar,
        ),
        (DeepSeekV4SelectorDispatchPolicy::Parallel, false, &radix),
        (DeepSeekV4SelectorDispatchPolicy::Parallel, true, &radix4),
    ] {
        encode_select_top_k_f32_with_policy(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &outputs.0,
            Some(&outputs.1),
            &outputs.2,
            &outputs.3,
            &outputs.4,
            CAPACITY,
            VISIBLE,
            TOP_K,
            1,
            dispatch_policy,
            use_radix4,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    for (label, left, right) in [
        ("mask", &scalar.0, &radix.0),
        ("ranked", &scalar.1, &radix.1),
        ("cache order", &scalar.2, &radix.2),
        ("count", &scalar.3, &radix.3),
        ("status", &scalar.4, &radix.4),
        ("radix4 mask", &scalar.0, &radix4.0),
        ("radix4 ranked", &scalar.1, &radix4.1),
        ("radix4 cache order", &scalar.2, &radix4.2),
        ("radix4 count", &scalar.3, &radix4.3),
        ("radix4 status", &scalar.4, &radix4.4),
    ] {
        assert_eq!(read_i32(left), read_i32(right), "{label}");
    }
    assert_eq!(
        read_i32(&radix.1),
        expected_ranked
            .iter()
            .map(|&row| row as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_i32(&radix.2),
        expected_cache_order
            .iter()
            .map(|&row| row as i32)
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "focused shallow sparse-selector crossover profiler; run explicitly with --nocapture"]
fn profile_shallow_sparse_selector_crossover() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 2_304;
    const TOP_K: usize = 512;
    const SAMPLES: usize = 12;

    fn timed_gpu<F>(ctx: &MetalContext, encode: F) -> f64
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx
            .queue
            .commandBuffer()
            .expect("selector profile command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).expect("encode selector profile phase");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "selector profile command failed: {:?}",
            command.error()
        );
        let elapsed_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed_ms.is_finite() && elapsed_ms > 0.0);
        elapsed_ms
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    let score_values = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, 1]);
    let scalar_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let scalar_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let scalar_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let scalar_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let radix_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let radix_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let radix_count = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let radix_status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();

    for visible in [513usize, 520, 528, 544, 576, 640, 768, 896, 1_024] {
        let visible_counts = offset_i32(&ctx, &[visible as i32], vec![1]);
        let scalar = |encoder: &KernelEncoder| {
            encode_select_top_k_f32_with_policy(
                &ctx,
                encoder,
                &scores,
                &visible_counts,
                &scalar_mask,
                None,
                &scalar_ids,
                &scalar_count,
                &scalar_status,
                CAPACITY,
                visible,
                TOP_K,
                1,
                DeepSeekV4SelectorDispatchPolicy::ScalarOracle,
                true,
            )
        };
        let radix4 = |encoder: &KernelEncoder| {
            encode_select_top_k_f32_with_policy(
                &ctx,
                encoder,
                &scores,
                &visible_counts,
                &radix_mask,
                None,
                &radix_ids,
                &radix_count,
                &radix_status,
                CAPACITY,
                visible,
                TOP_K,
                1,
                DeepSeekV4SelectorDispatchPolicy::Parallel,
                true,
            )
        };
        for _ in 0..3 {
            timed_gpu(&ctx, scalar);
            timed_gpu(&ctx, radix4);
        }
        let scalar_before_samples = (0..SAMPLES)
            .map(|_| timed_gpu(&ctx, scalar))
            .collect::<Vec<_>>();
        let radix_samples = (0..SAMPLES)
            .map(|_| timed_gpu(&ctx, radix4))
            .collect::<Vec<_>>();
        let scalar_after_samples = (0..SAMPLES)
            .map(|_| timed_gpu(&ctx, scalar))
            .collect::<Vec<_>>();
        let (scalar_before_ms, scalar_before_p95_ms) =
            median_and_p95(scalar_before_samples.clone());
        let (radix_ms, radix_p95_ms) = median_and_p95(radix_samples.clone());
        let (scalar_after_ms, scalar_after_p95_ms) = median_and_p95(scalar_after_samples.clone());
        let scalar_midpoint_ms = (scalar_before_ms + scalar_after_ms) * 0.5;

        timed_gpu(&ctx, scalar);
        let scalar_result = (
            read_i32(&scalar_mask),
            read_i32(&scalar_ids),
            read_i32(&scalar_count),
            read_i32(&scalar_status),
        );
        timed_gpu(&ctx, radix4);
        let radix_result = (
            read_i32(&radix_mask),
            read_i32(&radix_ids),
            read_i32(&radix_count),
            read_i32(&radix_status),
        );
        assert_eq!(
            radix_result, scalar_result,
            "selector drift at {visible} rows"
        );
        assert_eq!(radix_result.2, [TOP_K as i32]);
        assert_eq!(radix_result.3, [0]);

        eprintln!(
            "deepseek_v4 shallow_selector visible={visible} capacity={CAPACITY} scalar_before_ms={scalar_before_ms:.6} scalar_before_p95_ms={scalar_before_p95_ms:.6} radix4_ms={radix_ms:.6} radix4_p95_ms={radix_p95_ms:.6} scalar_after_ms={scalar_after_ms:.6} scalar_after_p95_ms={scalar_after_p95_ms:.6} midpoint_saving_ms={:.6} projected_21_layer_saving_ms={:.3}",
            scalar_midpoint_ms - radix_ms,
            (scalar_midpoint_ms - radix_ms) * 21.0,
        );
        eprintln!(
            "deepseek_v4 shallow_selector visible={visible} scalar_before_samples_ms={scalar_before_samples:?} radix4_samples_ms={radix_samples:?} scalar_after_samples_ms={scalar_after_samples:?}"
        );
    }
}

#[test]
fn stable_top512_scales_to_the_full_64k_csa_history() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 16_384;
    const TOP_K: usize = 512;
    let scores = (0..CAPACITY)
        .map(|row| {
            let bucket = (row * 73 + row / 11) % 2_003;
            bucket as f32 * 0.001 - 0.7
        })
        .collect::<Vec<_>>();
    let mut expected = deployed_selector_top_k_indices(&scores, TOP_K);
    expected.sort_unstable();

    let scores = offset_f32(&ctx, &scores, vec![CAPACITY as u64, 1]);
    let visible_counts = offset_i32(&ctx, &[CAPACITY as i32], vec![1]);
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let cache_order = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let started = std::time::Instant::now();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &mask,
        None,
        &cache_order,
        &counts,
        &status,
        CAPACITY,
        CAPACITY,
        TOP_K,
        1,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    let cold_elapsed = started.elapsed();
    assert!(
        command.error().is_none(),
        "64K selector command failed: {:?}",
        command.error()
    );
    assert_eq!(read_i32(&status), [0]);
    assert_eq!(read_i32(&counts), [TOP_K as i32]);
    let cold_ids = read_i32(&cache_order);
    assert_eq!(
        cold_ids,
        expected.iter().map(|&row| row as i32).collect::<Vec<_>>()
    );
    let cold_mask = read_i32(&mask);
    assert_eq!(cold_mask.iter().filter(|&&value| value == 1).count(), TOP_K);
    for row in expected {
        assert_eq!(cold_mask[row], 1);
    }
    let warm_started = std::time::Instant::now();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &mask,
        None,
        &cache_order,
        &counts,
        &status,
        CAPACITY,
        CAPACITY,
        TOP_K,
        1,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    let warm_elapsed = warm_started.elapsed();
    assert!(command.error().is_none());
    assert_eq!(read_i32(&status), [0]);
    assert_eq!(read_i32(&counts), [TOP_K as i32]);
    assert_eq!(read_i32(&cache_order), cold_ids);
    assert_eq!(read_i32(&mask), cold_mask);
    eprintln!(
        "deepseek_v4 full_64k_top512 cold_ms={:.3} warm_ms={:.3}",
        cold_elapsed.as_secs_f64() * 1e3,
        warm_elapsed.as_secs_f64() * 1e3,
    );
}

#[test]
fn packed_top512_scales_to_the_full_64k_csa_history() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 16_384;
    const TOP_K: usize = 512;
    const QUERIES: usize = 128;
    let mut score_values = Vec::with_capacity(CAPACITY * QUERIES);
    for query in 0..QUERIES {
        score_values.extend((0..CAPACITY).map(|row| {
            if query.is_multiple_of(2) {
                row as f32
            } else {
                -(row as f32)
            }
        }));
    }
    let scores = offset_f32(&ctx, &score_values, vec![CAPACITY as u64, QUERIES as u64]);
    let visible_counts = offset_i32(&ctx, &vec![CAPACITY as i32; QUERIES], vec![QUERIES as u64]);
    let mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let cache_order = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let run = |radix4| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_select_top_k_f32_with_policy(
            &ctx,
            &encoder,
            &scores,
            &visible_counts,
            &mask,
            None,
            &cache_order,
            &counts,
            &status,
            CAPACITY,
            CAPACITY,
            TOP_K,
            QUERIES,
            DeepSeekV4SelectorDispatchPolicy::Production,
            radix4,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (
            (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
            read_i32(&mask),
            read_i32(&cache_order),
            read_i32(&counts),
            read_i32(&status),
        )
    };
    run(false);
    run(true);
    let bitwise_before = run(false);
    let radix4 = run(true);
    let bitwise_after = run(false);
    for (label, candidate, baseline) in [
        ("mask before", &radix4.1, &bitwise_before.1),
        ("IDs before", &radix4.2, &bitwise_before.2),
        ("counts before", &radix4.3, &bitwise_before.3),
        ("status before", &radix4.4, &bitwise_before.4),
        ("mask after", &radix4.1, &bitwise_after.1),
        ("IDs after", &radix4.2, &bitwise_after.2),
        ("counts after", &radix4.3, &bitwise_after.3),
        ("status after", &radix4.4, &bitwise_after.4),
    ] {
        assert_eq!(candidate, baseline, "{label}");
    }
    assert_eq!(radix4.4, vec![0; QUERIES]);
    assert_eq!(radix4.3, vec![TOP_K as i32; QUERIES]);
    let ids = radix4.2;
    let high = (CAPACITY - TOP_K..CAPACITY)
        .map(|row| row as i32)
        .collect::<Vec<_>>();
    let low = (0..TOP_K as i32).collect::<Vec<_>>();
    for query in 0..QUERIES {
        let expected = if query.is_multiple_of(2) { &high } else { &low };
        assert_eq!(&ids[query * TOP_K..(query + 1) * TOP_K], expected);
    }
    let bitwise_midpoint = (bitwise_before.0 + bitwise_after.0) * 0.5;
    assert!(
        radix4.0 <= bitwise_midpoint * 1.1,
        "packed radix4 regressed from {bitwise_midpoint:.3} to {:.3} ms",
        radix4.0
    );
    eprintln!(
        "deepseek_v4 packed_full_64k_top512 bitwise_before_gpu_ms={:.3} radix4_gpu_ms={:.3} bitwise_after_gpu_ms={:.3}",
        bitwise_before.0, radix4.0, bitwise_after.0,
    );
}

#[test]
fn sparse_csa_indexer_and_selected_attention_match_cpu_oracles() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const INDEX_HEADS: usize = 64;
    const INDEX_DIM: usize = 128;
    const ATTENTION_HEADS: usize = 2;
    const ATTENTION_DIM: usize = 128;
    const CAPACITY: usize = 768;
    const VISIBLE: usize = 513;
    const TOP_K: usize = 512;
    const POSITION: usize = 2_051;

    let index_queries = (0..INDEX_HEADS * INDEX_DIM)
        .map(|index| {
            let head = index / INDEX_DIM;
            let dimension = index % INDEX_DIM;
            0.012 + head as f32 * 0.000_031 + (dimension % 17) as f32 * 0.000_019
        })
        .collect::<Vec<_>>();
    let raw_head_weights = (0..INDEX_HEADS)
        .map(|head| 0.31 + head as f32 * 0.0017)
        .collect::<Vec<_>>();
    let index_scale = 1.0 / ((INDEX_HEADS * INDEX_DIM) as f32).sqrt();
    let index_key_bits = (0..CAPACITY * INDEX_DIM)
        .map(|index| {
            let row = index / INDEX_DIM;
            let dimension = index % INDEX_DIM;
            let value = (row + 1) as f32 * 0.000_01 * (1.0 + (dimension % 11) as f32 * 0.013);
            half::f16::from_f32(value).to_bits()
        })
        .collect::<Vec<_>>();
    let index_keys_f32 = index_key_bits
        .iter()
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    let expected_scores = indexer_scores(
        &index_queries,
        &raw_head_weights,
        &index_keys_f32[..VISIBLE * INDEX_DIM],
        INDEX_HEADS,
        INDEX_DIM,
    )
    .unwrap();
    let expected_ranked = top_k_indices(&expected_scores, TOP_K).unwrap();
    assert_eq!(expected_ranked[0], VISIBLE - 1);
    assert_eq!(*expected_ranked.last().unwrap(), 1);
    assert!(!expected_ranked.contains(&0));

    let attention_queries = (0..ATTENTION_HEADS * ATTENTION_DIM)
        .map(|index| {
            let head = index / ATTENTION_DIM;
            let dimension = index % ATTENTION_DIM;
            0.041 + head as f32 * 0.003 - (dimension % 19) as f32 * 0.0007
        })
        .collect::<Vec<_>>();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let raw_start = POSITION + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
    let mut raw_ring = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * ATTENTION_DIM];
    let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * ATTENTION_DIM);
    for logical_position in raw_start..=POSITION {
        let row = (0..ATTENTION_DIM)
            .map(|dimension| {
                let tag = (logical_position * 29 + dimension * 7 + logical_position / 5) % 101;
                round_f16((tag as f32 - 50.0) * 0.0013)
            })
            .collect::<Vec<_>>();
        raw_rows.extend_from_slice(&row);
        let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
        raw_ring[slot * ATTENTION_DIM..(slot + 1) * ATTENTION_DIM].copy_from_slice(&row);
    }
    let mut compressed_rows = (0..CAPACITY * ATTENTION_DIM)
        .map(|index| {
            let row = index / ATTENTION_DIM;
            let dimension = index % ATTENTION_DIM;
            let tag = (row * 37 + dimension * 13 + row / 7) % 113;
            round_f16((tag as f32 - 56.0) * 0.0011)
        })
        .collect::<Vec<_>>();
    for dimension in 0..ATTENTION_DIM {
        compressed_rows[dimension] = round_f16(1_800.0 * attention_queries[dimension] + 25.0);
        compressed_rows[(VISIBLE - 1) * ATTENTION_DIM + dimension] =
            round_f16(1_800.0 * attention_queries[dimension]);
    }
    let mut selected_mask = vec![false; VISIBLE];
    for &row in &expected_ranked {
        selected_mask[row] = true;
    }
    let sinks = [-0.27, 0.19];
    let expected_attention = shared_kv_attention(
        &attention_queries,
        ATTENTION_HEADS,
        ATTENTION_DIM,
        &raw_rows,
        &compressed_rows[..VISIBLE * ATTENTION_DIM],
        Some(&selected_mask),
        &sinks,
    )
    .unwrap();
    let dense_attention = shared_kv_attention(
        &attention_queries,
        ATTENTION_HEADS,
        ATTENTION_DIM,
        &raw_rows,
        &compressed_rows[..VISIBLE * ATTENTION_DIM],
        None,
        &sinks,
    )
    .unwrap();
    let mut first_512 = vec![false; VISIBLE];
    first_512[..TOP_K].fill(true);
    let first_512_attention = shared_kv_attention(
        &attention_queries,
        ATTENTION_HEADS,
        ATTENTION_DIM,
        &raw_rows,
        &compressed_rows[..VISIBLE * ATTENTION_DIM],
        Some(&first_512),
        &sinks,
    )
    .unwrap();
    assert!(
        expected_attention
            .iter()
            .zip(&dense_attention)
            .any(|(selected, dense)| (selected - dense).abs() > 1e-3)
    );
    assert!(
        expected_attention
            .iter()
            .zip(&first_512_attention)
            .any(|(selected, first)| (selected - first).abs() > 1e-3)
    );

    let index_queries = offset_f32(
        &ctx,
        &index_queries,
        vec![INDEX_DIM as u64, INDEX_HEADS as u64, 1],
    );
    let head_weights = offset_f32(&ctx, &raw_head_weights, vec![INDEX_HEADS as u64, 1]);
    let index_keys = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&index_key_bits),
        vec![INDEX_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let visible_counts = offset_i32(&ctx, &[VISIBLE as i32], vec![1]);
    let scores = MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let gpu_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let ranked_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let cache_order_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let selected_counts = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let attention_queries_tensor = offset_f32(
        &ctx,
        &attention_queries,
        vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64],
    );
    let raw_bits = raw_ring
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let raw_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&raw_bits),
        vec![ATTENTION_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        GgmlType::F16,
    )
    .unwrap();
    let compressed_bits = compressed_rows
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![ATTENTION_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sink_tensor = offset_f32(&ctx, &sinks, vec![ATTENTION_HEADS as u64]);
    let attention_output =
        MetalTensor::zeros_f32(&ctx, vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64]).unwrap();
    let cooperative_queries = attention_queries_tensor
        .view_subrange(0, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1]);
    let cooperative_output =
        attention_output.view_subrange(0, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1]);
    let attention_config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: ATTENTION_HEADS,
        head_dim: ATTENTION_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_scale_f32_in_place(
        &ctx,
        &encoder,
        &head_weights,
        index_scale,
        "indexer head weights",
    )
    .unwrap();
    encode_lightning_indexer_scores_f16(
        &ctx,
        &encoder,
        &index_queries,
        &head_weights,
        &index_keys,
        &visible_counts,
        &scores,
        INDEX_HEADS,
        INDEX_DIM,
        CAPACITY,
        1,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &gpu_mask,
        Some(&ranked_ids),
        &cache_order_ids,
        &selected_counts,
        &status,
        CAPACITY,
        VISIBLE,
        TOP_K,
        1,
    )
    .unwrap();
    encode_cooperative_selected_sink_attention_f16(
        &ctx,
        &encoder,
        &cooperative_queries,
        &raw_cache,
        &raw_cache,
        DeepSeekV4RawCacheLayout::Ring,
        &compressed_cache,
        CAPACITY,
        &cache_order_ids,
        &selected_counts,
        &visible_counts,
        &sink_tensor,
        &cooperative_output,
        POSITION as u32,
        0,
        1,
        1,
        TOP_K,
        false,
        false,
        attention_config,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "sparse CSA command failed: {:?}",
        command.error()
    );

    assert_eq!(read_i32(&status), vec![0]);
    assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32]);
    let actual_scores = read_f32(&scores);
    assert_close(
        "Lightning Indexer scores",
        &actual_scores[..VISIBLE],
        &expected_scores,
        3e-5,
    );
    assert!(
        actual_scores[VISIBLE..]
            .iter()
            .all(|score| *score == f32::NEG_INFINITY)
    );
    let ranked_ids = read_i32(&ranked_ids);
    assert_eq!(
        ranked_ids,
        expected_ranked
            .iter()
            .map(|index| *index as i32)
            .collect::<Vec<_>>()
    );
    let cache_order_ids = read_i32(&cache_order_ids);
    assert_eq!(
        cache_order_ids,
        (1..=VISIBLE - 1)
            .map(|index| index as i32)
            .collect::<Vec<_>>()
    );
    assert_close(
        "selected sparse CSA attention",
        &read_f32(&attention_output),
        &expected_attention,
        8e-5,
    );
}

#[test]
fn sparse_csa_score_select_attention_reaches_the_full_context_frontier() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const INDEX_HEADS: usize = 1;
    const INDEX_DIM: usize = 128;
    const ATTENTION_HEADS: usize = 1;
    const ATTENTION_DIM: usize = 128;
    const CAPACITY: usize = 262_144;
    const VISIBLE: usize = CAPACITY;
    const TOP_K: usize = 512;
    const POSITION: usize = 1_048_575;

    let index_queries = (0..INDEX_DIM)
        .map(|dimension| 0.008 + (dimension % 17) as f32 * 0.0003)
        .collect::<Vec<_>>();
    let raw_head_weights = [0.73f32];
    let index_scale = 1.0 / ((INDEX_HEADS * INDEX_DIM) as f32).sqrt();
    let index_key_bits = (0..CAPACITY * INDEX_DIM)
        .map(|index| {
            let row = index / INDEX_DIM;
            let dimension = index % INDEX_DIM;
            let value = (row + 1) as f32 * 0.000_004 * (1.0 + (dimension % 13) as f32 * 0.007);
            half::f16::from_f32(value).to_bits()
        })
        .collect::<Vec<_>>();
    let index_keys_f32 = index_key_bits
        .iter()
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    let expected_scores = indexer_scores(
        &index_queries,
        &raw_head_weights,
        &index_keys_f32[..VISIBLE * INDEX_DIM],
        INDEX_HEADS,
        INDEX_DIM,
    )
    .unwrap();
    let expected_ranked = top_k_indices(&expected_scores, TOP_K).unwrap();
    let mut expected_cache_order = expected_ranked.clone();
    expected_cache_order.sort_unstable();
    assert!(expected_cache_order[0] > 260_000);
    assert!(expected_cache_order.contains(&(CAPACITY - 1)));

    let attention_queries = (0..ATTENTION_DIM)
        .map(|dimension| 0.037 + (dimension % 19) as f32 * 0.0006)
        .collect::<Vec<_>>();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let raw_start = POSITION + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
    let mut raw_ring = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * ATTENTION_DIM];
    let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * ATTENTION_DIM);
    for logical_position in raw_start..=POSITION {
        let row = (0..ATTENTION_DIM)
            .map(|dimension| {
                let tag = (logical_position * 17 + dimension * 11) % 89;
                round_f16((tag as f32 - 44.0) * 0.0004)
            })
            .collect::<Vec<_>>();
        raw_rows.extend_from_slice(&row);
        let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
        raw_ring[slot * ATTENTION_DIM..(slot + 1) * ATTENTION_DIM].copy_from_slice(&row);
    }
    let compressed_rows = (0..CAPACITY * ATTENTION_DIM)
        .map(|index| {
            let row = index / ATTENTION_DIM;
            let dimension = index % ATTENTION_DIM;
            let progression = row as f32 / (VISIBLE - 1) as f32;
            round_f16(-0.25 + progression * 0.5 + ((dimension % 11) as f32 - 5.0) * 0.0005)
        })
        .collect::<Vec<_>>();
    let mut selected_mask = vec![false; VISIBLE];
    for &row in &expected_ranked {
        selected_mask[row] = true;
    }
    let sinks = [-0.17f32];
    let expected_attention = shared_kv_attention(
        &attention_queries,
        ATTENTION_HEADS,
        ATTENTION_DIM,
        &raw_rows,
        &compressed_rows[..VISIBLE * ATTENTION_DIM],
        Some(&selected_mask),
        &sinks,
    )
    .unwrap();
    let mut first_512 = vec![false; VISIBLE];
    first_512[..TOP_K].fill(true);
    let first_512_attention = shared_kv_attention(
        &attention_queries,
        ATTENTION_HEADS,
        ATTENTION_DIM,
        &raw_rows,
        &compressed_rows[..VISIBLE * ATTENTION_DIM],
        Some(&first_512),
        &sinks,
    )
    .unwrap();
    assert!(
        expected_attention
            .iter()
            .zip(&first_512_attention)
            .any(|(selected, first)| (selected - first).abs() > 1e-3)
    );

    let index_queries = offset_f32(
        &ctx,
        &index_queries,
        vec![INDEX_DIM as u64, INDEX_HEADS as u64, 1],
    );
    let head_weights = offset_f32(&ctx, &raw_head_weights, vec![INDEX_HEADS as u64, 1]);
    let index_keys = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&index_key_bits),
        vec![INDEX_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let visible_counts = offset_i32(&ctx, &[VISIBLE as i32], vec![1]);
    let scores = MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let gpu_mask = MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, 1]).unwrap();
    let cache_order_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).unwrap();
    let selected_counts = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
    let attention_queries_tensor = offset_f32(
        &ctx,
        &attention_queries,
        vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64],
    );
    let raw_bits = raw_ring
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let raw_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&raw_bits),
        vec![ATTENTION_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        GgmlType::F16,
    )
    .unwrap();
    let compressed_bits = compressed_rows
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![ATTENTION_DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sink_tensor = offset_f32(&ctx, &sinks, vec![ATTENTION_HEADS as u64]);
    let attention_output =
        MetalTensor::zeros_f32(&ctx, vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64]).unwrap();
    let attention_config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: ATTENTION_HEADS,
        head_dim: ATTENTION_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_scale_f32_in_place(
        &ctx,
        &encoder,
        &head_weights,
        index_scale,
        "full-context indexer head weights",
    )
    .unwrap();
    encode_lightning_indexer_scores_f16(
        &ctx,
        &encoder,
        &index_queries,
        &head_weights,
        &index_keys,
        &visible_counts,
        &scores,
        INDEX_HEADS,
        INDEX_DIM,
        CAPACITY,
        1,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &gpu_mask,
        None,
        &cache_order_ids,
        &selected_counts,
        &status,
        CAPACITY,
        VISIBLE,
        TOP_K,
        1,
    )
    .unwrap();
    let selected_ids = cache_order_ids.view_subrange(0, vec![TOP_K as u64]);
    encode_selected_sink_attention_f16(
        &ctx,
        &encoder,
        &attention_queries_tensor,
        &raw_cache,
        &compressed_cache,
        &selected_ids,
        &sink_tensor,
        &attention_output,
        POSITION as u32,
        VISIBLE,
        TOP_K,
        CAPACITY,
        attention_config,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "far sparse CSA command failed: {:?}",
        command.error()
    );

    assert_eq!(read_i32(&status), vec![0]);
    assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32]);
    let actual_scores = read_f32(&scores);
    assert_close(
        "full-context Lightning Indexer scores",
        &actual_scores,
        &expected_scores,
        3e-5,
    );
    let actual_mask = read_i32(&gpu_mask);
    assert!(
        actual_mask
            .iter()
            .enumerate()
            .all(|(row, &selected)| { (selected != 0) == (row < VISIBLE && selected_mask[row]) })
    );
    assert_eq!(
        read_i32(&cache_order_ids),
        expected_cache_order
            .iter()
            .map(|row| *row as i32)
            .collect::<Vec<_>>()
    );
    assert_close(
        "full-context selected sparse CSA attention",
        &read_f32(&attention_output),
        &expected_attention,
        1.5e-4,
    );
}

#[test]
fn cooperative_dense_attention_matches_legacy_singleton_within_roundoff() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 512;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let query_values = (0..dims.query_width)
        .map(|index| ((index * 17 + 3) % 131) as f32 * 0.0007 - 0.043)
        .collect::<Vec<_>>();
    let queries = offset_f32(
        &ctx,
        &query_values,
        vec![config.head_dim as u64, config.head_count as u64],
    );
    let cooperative_queries = queries.view_subrange(0, vec![dims.query_width as u64, 1]);
    let raw_values = (0..DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim)
        .map(|index| {
            let slot = index / config.head_dim;
            let dimension = index % config.head_dim;
            let tag = (slot * 23 + dimension * 7 + slot / 5) % 137;
            round_f16((tag as f32 - 68.0) * 0.0009)
        })
        .collect::<Vec<_>>();
    let raw_bits = raw_values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let raw_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&raw_bits),
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        GgmlType::F16,
    )
    .unwrap();
    let compressed_values = (0..CAPACITY * config.head_dim)
        .map(|index| {
            let row = index / config.head_dim;
            let dimension = index % config.head_dim;
            let tag = (row * 31 + dimension * 11 + row / 5) % 149;
            round_f16((tag as f32 - 74.0) * 0.0008)
        })
        .collect::<Vec<_>>();
    let compressed_bits = compressed_values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![config.head_dim as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sinks = offset_f32(
        &ctx,
        &(0..config.head_count)
            .map(|head| -0.37 + head as f32 * 0.003)
            .collect::<Vec<_>>(),
        vec![config.head_count as u64],
    );

    for (label, kind, position, compressed_count) in [
        ("swa-127", AttentionKind::SlidingWindow, 127u32, 0usize),
        ("csa-127", AttentionKind::CompressedSparse, 127, 32),
        ("csa-2047", AttentionKind::CompressedSparse, 2_047, 512),
        ("hca-127", AttentionKind::HeavilyCompressed, 127, 1),
        ("hca-65535", AttentionKind::HeavilyCompressed, 65_535, 512),
    ] {
        let compressed = (compressed_count > 0).then_some(DeepSeekV4PublishedRows {
            cache: &compressed_cache,
            count: compressed_count,
            capacity_rows: CAPACITY,
        });
        let legacy =
            MetalTensor::zeros_f32(&ctx, vec![config.head_dim as u64, config.head_count as u64])
                .unwrap();
        let cooperative = MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap();
        let grouped = MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap();
        let online = (kind == AttentionKind::HeavilyCompressed)
            .then(|| MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap());
        let tiled = (kind == AttentionKind::HeavilyCompressed
            && compressed_count == DEEPSEEK_V4_HCA_TILE_ROWS)
            .then(|| MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap());
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_dense_sink_attention_f16(
            &ctx, &encoder, &queries, &raw_cache, compressed, &sinks, &legacy, position, config,
        )
        .unwrap();
        encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &cooperative_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            compressed,
            &sinks,
            &cooperative,
            kind,
            position,
            1,
            config,
        )
        .unwrap();
        encode_grouped_online_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &cooperative_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            compressed,
            &sinks,
            &grouped,
            kind,
            position,
            1,
            config,
        )
        .unwrap();
        if let Some(online) = &online {
            encode_online_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &cooperative_queries,
                &raw_cache,
                &raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                compressed.expect("online HCA has compressed rows"),
                &sinks,
                online,
                position,
                0,
                1,
                128,
                false,
                config,
            )
            .unwrap();
        }
        if let Some(tiled) = &tiled {
            encode_tiled_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &cooperative_queries,
                &raw_cache,
                &raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                compressed.expect("HCA handoff has compressed rows"),
                &sinks,
                tiled,
                position,
                0,
                1,
                128,
                config,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let legacy = read_f32(&legacy);
        let cooperative = read_f32(&cooperative);
        let differing = legacy
            .iter()
            .zip(&cooperative)
            .filter(|(legacy, cooperative)| legacy.to_bits() != cooperative.to_bits())
            .count();
        let max_abs = legacy
            .iter()
            .zip(&cooperative)
            .map(|(legacy, cooperative)| (legacy - cooperative).abs())
            .fold(0.0f32, f32::max);
        let squared_error = legacy
            .iter()
            .zip(&cooperative)
            .map(|(legacy, cooperative)| f64::from(legacy - cooperative).powi(2))
            .sum::<f64>();
        let reference_norm = legacy
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let relative_rms = (squared_error / reference_norm).sqrt();
        eprintln!(
            "cooperative dense {label} rows={} differing={differing}/{} max_abs={max_abs} rel_rms={relative_rms:.9}",
            DEEPSEEK_V4_LOCAL_WINDOW.min(position as usize + 1) + compressed_count,
            legacy.len(),
        );
        assert!(max_abs <= 5e-7, "{label} max abs {max_abs}");
        assert!(relative_rms <= 1e-6, "{label} relative RMS {relative_rms}");
        let grouped = read_f32(&grouped);
        let grouped_max_abs = legacy
            .iter()
            .zip(&grouped)
            .map(|(legacy, grouped)| (legacy - grouped).abs())
            .fold(0.0f32, f32::max);
        let grouped_squared_error = legacy
            .iter()
            .zip(&grouped)
            .map(|(legacy, grouped)| f64::from(legacy - grouped).powi(2))
            .sum::<f64>();
        let grouped_relative_rms = (grouped_squared_error / reference_norm).sqrt();
        eprintln!(
            "grouped online dense {label} rows={} max_abs={grouped_max_abs} rel_rms={grouped_relative_rms:.9}",
            DEEPSEEK_V4_LOCAL_WINDOW.min(position as usize + 1) + compressed_count,
        );
        assert!(
            grouped_max_abs <= 8e-5,
            "grouped {label} max abs {grouped_max_abs}"
        );
        assert!(
            grouped_relative_rms <= 1e-3,
            "grouped {label} relative RMS {grouped_relative_rms}"
        );
        if let Some(online) = online {
            let online = read_f32(&online);
            let differing = grouped
                .iter()
                .zip(&online)
                .filter(|(grouped, online)| grouped.to_bits() != online.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "grouped and single-head online HCA differ for {label}"
            );
        }
        if let Some(tiled) = tiled {
            let tiled = read_f32(&tiled);
            let tiled_differing = legacy
                .iter()
                .zip(&tiled)
                .filter(|(legacy, tiled)| legacy.to_bits() != tiled.to_bits())
                .count();
            let tiled_max_abs = legacy
                .iter()
                .zip(&tiled)
                .map(|(legacy, tiled)| (legacy - tiled).abs())
                .fold(0.0f32, f32::max);
            let tiled_squared_error = legacy
                .iter()
                .zip(&tiled)
                .map(|(legacy, tiled)| f64::from(legacy - tiled).powi(2))
                .sum::<f64>();
            let tiled_relative_rms = (tiled_squared_error / reference_norm).sqrt();
            eprintln!(
                "tiled dense {label} rows={} differing={tiled_differing}/{} max_abs={tiled_max_abs} rel_rms={tiled_relative_rms:.9}",
                DEEPSEEK_V4_LOCAL_WINDOW + compressed_count,
                legacy.len(),
            );
            assert!(
                tiled_max_abs <= 5e-7,
                "tiled {label} max abs {tiled_max_abs}"
            );
            assert!(
                tiled_relative_rms <= 1e-6,
                "tiled {label} relative RMS {tiled_relative_rms}"
            );
            let cooperative_tiled_differing = cooperative
                .iter()
                .zip(&tiled)
                .filter(|(cooperative, tiled)| cooperative.to_bits() != tiled.to_bits())
                .count();
            assert_eq!(
                cooperative_tiled_differing, 0,
                "cooperative and tiled HCA differ at the 512-row handoff"
            );
        }
    }
}

#[test]
fn cooperative_selected_attention_matches_legacy_singleton_within_roundoff() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 768;
    const TOP_K: usize = 512;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let query_values = (0..dims.query_width)
        .map(|index| ((index * 17 + 3) % 131) as f32 * 0.0007 - 0.043)
        .collect::<Vec<_>>();
    let queries = offset_f32(
        &ctx,
        &query_values,
        vec![config.head_dim as u64, config.head_count as u64],
    );
    let packed_queries = queries.view_subrange(0, vec![dims.query_width as u64, 1]);
    let compressed_values = (0..CAPACITY * config.head_dim)
        .map(|index| {
            let row = index / config.head_dim;
            let dimension = index % config.head_dim;
            let tag = (row * 31 + dimension * 11 + row / 5) % 149;
            round_f16((tag as f32 - 74.0) * 0.0008)
        })
        .collect::<Vec<_>>();
    let compressed_bits = compressed_values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![config.head_dim as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sink_values = (0..config.head_count)
        .map(|head| -0.37 + head as f32 * 0.003)
        .collect::<Vec<_>>();
    let sinks = offset_f32(&ctx, &sink_values, vec![config.head_count as u64]);

    for (position, visible, first_selected) in [
        (2_051u32, 513usize, 0usize),
        (2_052, 513, 1),
        (3_071, 768, 256),
    ] {
        let raw_values = (0..DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim)
            .map(|index| {
                let slot = index / config.head_dim;
                let dimension = index % config.head_dim;
                let tag = (slot * 23 + dimension * 7 + position as usize) % 137;
                round_f16((tag as f32 - 68.0) * 0.0009)
            })
            .collect::<Vec<_>>();
        let raw_bits = raw_values
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let raw_cache = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&raw_bits),
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let ids = (first_selected..first_selected + TOP_K)
            .map(|row| row as i32)
            .collect::<Vec<_>>();
        let selected_ids = offset_i32(&ctx, &ids, vec![TOP_K as u64, 1]);
        let legacy_ids = selected_ids.view_subrange(0, vec![TOP_K as u64]);
        let selected_counts = offset_i32(&ctx, &[TOP_K as i32], vec![1]);
        let visible_counts = offset_i32(&ctx, &[visible as i32], vec![1]);
        let legacy =
            MetalTensor::zeros_f32(&ctx, vec![config.head_dim as u64, config.head_count as u64])
                .unwrap();
        let cooperative = MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap();
        let online = MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap();
        let direct = MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, 1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_cache,
            &compressed,
            &legacy_ids,
            &sinks,
            &legacy,
            position,
            visible,
            TOP_K,
            CAPACITY,
            config,
        )
        .unwrap();
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &packed_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            &compressed,
            CAPACITY,
            &selected_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            &cooperative,
            position,
            0,
            1,
            1,
            TOP_K,
            false,
            false,
            config,
        )
        .unwrap();
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &packed_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            &compressed,
            CAPACITY,
            &selected_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            &online,
            position,
            0,
            1,
            1,
            TOP_K,
            true,
            false,
            config,
        )
        .unwrap();
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &packed_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            &compressed,
            CAPACITY,
            &selected_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            &direct,
            position,
            0,
            1,
            1,
            TOP_K,
            true,
            true,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let legacy = read_f32(&legacy);
        let reference_norm = legacy
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        for (label, candidate, max_allowed) in [
            ("cooperative", read_f32(&cooperative), 1e-8),
            ("online", read_f32(&online), 1e-8),
            ("direct", read_f32(&direct), 1e-8),
        ] {
            let differing = legacy
                .iter()
                .zip(&candidate)
                .filter(|(legacy, candidate)| legacy.to_bits() != candidate.to_bits())
                .count();
            let max_abs = legacy
                .iter()
                .zip(&candidate)
                .map(|(legacy, candidate)| (legacy - candidate).abs())
                .fold(0.0f32, f32::max);
            let squared_error = legacy
                .iter()
                .zip(&candidate)
                .map(|(legacy, candidate)| f64::from(legacy - candidate).powi(2))
                .sum::<f64>();
            let relative_rms = (squared_error / reference_norm).sqrt();
            eprintln!(
                "{label} selected position={position} visible={visible} differing={differing}/{} max_abs={max_abs} rel_rms={relative_rms:.9}",
                legacy.len(),
            );
            assert!(max_abs <= max_allowed, "{label} max abs {max_abs}");
            assert!(relative_rms <= 1e-6, "{label} relative RMS {relative_rms}");
        }
        assert_eq!(
            read_f32(&online)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&direct)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "direct selected attention changed online output at {position}"
        );
    }
}

#[test]
fn direct_selected_attention_matches_staged_for_packed_chunk_layout() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const TOKENS: usize = 4;
    const START_POSITION: u32 = 2_048;
    const CAPACITY: usize = 513;
    const TOP_K: usize = 512;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let queries = offset_f32(
        &ctx,
        &(0..TOKENS * dims.query_width)
            .map(|index| ((index * 17 + index / 31 + 5) % 257) as f32 * 0.0004 - 0.051)
            .collect::<Vec<_>>(),
        vec![dims.query_width as u64, TOKENS as u64],
    );
    let f16_tensor = |elements: usize, shape: Vec<u64>, seed: usize| {
        let bits = (0..elements)
            .map(|index| {
                half::f16::from_f32(
                    ((index * 29 + index / 13 + seed) % 251) as f32 * 0.0005 - 0.061,
                )
                .to_bits()
            })
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&bits), shape, GgmlType::F16).unwrap()
    };
    let raw_chunk = f16_tensor(
        TOKENS * config.head_dim,
        vec![config.head_dim as u64, TOKENS as u64],
        7,
    );
    let preserved_raw = f16_tensor(
        DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim,
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        11,
    );
    let compressed = f16_tensor(
        CAPACITY * config.head_dim,
        vec![config.head_dim as u64, CAPACITY as u64],
        19,
    );
    let selected_ids = offset_i32(
        &ctx,
        &(0..TOKENS)
            .flat_map(|_| 0..TOP_K as i32)
            .collect::<Vec<_>>(),
        vec![TOP_K as u64, TOKENS as u64],
    );
    let selected_counts = offset_i32(&ctx, &[TOP_K as i32; TOKENS], vec![TOKENS as u64]);
    let visible_counts = offset_i32(&ctx, &[512, 512, 512, 513], vec![TOKENS as u64]);
    let sinks = offset_f32(
        &ctx,
        &(0..config.head_count)
            .map(|head| -0.43 + head as f32 * 0.002)
            .collect::<Vec<_>>(),
        vec![config.head_count as u64],
    );
    let staged =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, TOKENS as u64]).unwrap();
    let direct =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, TOKENS as u64]).unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (output, direct_load) in [(&staged, false), (&direct, true)] {
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &preserved_raw,
            DeepSeekV4RawCacheLayout::Chunk,
            &compressed,
            CAPACITY,
            &selected_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            output,
            START_POSITION,
            0,
            TOKENS,
            TOKENS,
            TOP_K,
            true,
            direct_load,
            config,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "{:?}", command.error());
    assert_eq!(
        read_f32(&direct)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        read_f32(&staged)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "direct packed-chunk attention changed staged online output"
    );
}

#[test]
fn cooperative_selected_attention_bounds_corrupt_selector_metadata() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEAD_DIM: usize = 4;
    const CAPACITY: usize = 3;
    const SELECTED_SLOTS: usize = 3;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: 1,
        head_dim: HEAD_DIM,
        rotary_dim: 2,
        group_count: 1,
        output_rank: 1,
    };
    let queries = offset_f32(&ctx, &[0.17, -0.23, 0.31, 0.11], vec![HEAD_DIM as u64, 1]);
    let f16_tensor = |values: &[f32], shape: Vec<u64>| {
        let bits = values
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&bits), shape, GgmlType::F16).unwrap()
    };
    let raw_values = (0..DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM)
        .map(|index| ((index * 7 + 3) % 29) as f32 * 0.01 - 0.14)
        .collect::<Vec<_>>();
    let raw_cache = f16_tensor(
        &raw_values,
        vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
    );
    let compressed_cache = f16_tensor(
        &[
            0.8, -0.4, 0.2, 0.6, -0.7, 0.9, 0.5, -0.3, 0.4, 0.1, -0.8, 0.7,
        ],
        vec![HEAD_DIM as u64, CAPACITY as u64],
    );
    let sinks = offset_f32(&ctx, &[-0.2], vec![1]);

    let run = |ids: [i32; SELECTED_SLOTS], count: i32, visible: i32| {
        let selected_ids = offset_i32(&ctx, &ids, vec![SELECTED_SLOTS as u64, 1]);
        let selected_counts = offset_i32(&ctx, &[count], vec![1]);
        let visible_counts = offset_i32(&ctx, &[visible], vec![1]);
        let output = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, 1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            &compressed_cache,
            CAPACITY,
            &selected_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            &output,
            127,
            0,
            1,
            1,
            SELECTED_SLOTS,
            false,
            false,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "corrupt-selector command failed: {:?}",
            command.error()
        );
        read_f32(&output)
    };

    let raw_only = run([0, 1, 2], 0, CAPACITY as i32);
    let invalid_ids = run([-1, 4, 3], SELECTED_SLOTS as i32, 4);
    let negative_count = run([0, 1, 2], -7, CAPACITY as i32);
    let selected = run([0, 1, 2], SELECTED_SLOTS as i32, CAPACITY as i32);
    let oversized_count = run([0, 1, 2], i32::MAX, CAPACITY as i32);

    assert_eq!(invalid_ids, raw_only);
    assert_eq!(negative_count, raw_only);
    assert_eq!(oversized_count, selected);
    assert!(
        selected
            .iter()
            .zip(&raw_only)
            .any(|(selected, raw)| selected.to_bits() != raw.to_bits())
    );
}

#[test]
fn cooperative_lightning_scores_match_scalar_for_packed_visibility() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const CAPACITY: usize = 48;
    const MAX_VISIBLE: usize = 40;
    const QUERIES: usize = 9;
    const TOP_K: usize = 8;

    let query_values = (0..QUERIES * HEADS * DIM)
        .map(|index| {
            let tag = (index * 17 + index / DIM * 11 + 3) % 251;
            (tag as f32 - 125.0) * 0.0007
        })
        .collect::<Vec<_>>();
    let mut weight_values = (0..QUERIES * HEADS)
        .map(|index| 0.003 + (index * 13 % 29) as f32 * 0.0004)
        .collect::<Vec<_>>();
    weight_values[3 * HEADS] = f32::NAN;
    let key_bits = (0..CAPACITY * DIM)
        .map(|index| {
            let tag = (index * 23 + index / DIM * 19 + 7) % 257;
            half::f16::from_f32((tag as f32 - 128.0) * 0.0009).to_bits()
        })
        .collect::<Vec<_>>();
    let queries = offset_f32(
        &ctx,
        &query_values,
        vec![DIM as u64, HEADS as u64, QUERIES as u64],
    );
    let head_weights = offset_f32(&ctx, &weight_values, vec![HEADS as u64, QUERIES as u64]);
    let keys = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&key_bits),
        vec![DIM as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let visible_counts = offset_i32(
        &ctx,
        &[MAX_VISIBLE as i32, 9, -1, 40, 17, 33, 34, 39, 40],
        vec![QUERIES as u64],
    );
    let scalar_scores =
        MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let cooperative_scores =
        MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let bounded_scores =
        MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let tiled_scores = MetalTensor::zeros_f32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap();
    let make_selection = || {
        (
            MetalTensor::zeros_i32(&ctx, vec![CAPACITY as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
            MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap(),
        )
    };
    let (scalar_mask, scalar_ids, scalar_counts, scalar_status) = make_selection();
    let (cooperative_mask, cooperative_ids, cooperative_counts, cooperative_status) =
        make_selection();
    let (bounded_mask, bounded_ids, bounded_counts, bounded_status) = make_selection();
    let (tiled_mask, tiled_ids, tiled_counts, tiled_status) = make_selection();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_lightning_indexer_scores_f16_with_policy(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &scalar_scores,
        HEADS,
        DIM,
        CAPACITY,
        QUERIES,
        true,
    )
    .unwrap();
    encode_lightning_indexer_scores_f16_tiled_f32_with_limit(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &tiled_scores,
        HEADS,
        DIM,
        CAPACITY,
        MAX_VISIBLE,
        QUERIES,
    )
    .unwrap();
    encode_lightning_indexer_scores_f16(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &cooperative_scores,
        HEADS,
        DIM,
        CAPACITY,
        QUERIES,
    )
    .unwrap();
    encode_lightning_indexer_scores_f16_with_limit(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &bounded_scores,
        HEADS,
        DIM,
        CAPACITY,
        MAX_VISIBLE,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scalar_scores,
        &visible_counts,
        &scalar_mask,
        None,
        &scalar_ids,
        &scalar_counts,
        &scalar_status,
        CAPACITY,
        MAX_VISIBLE,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &cooperative_scores,
        &visible_counts,
        &cooperative_mask,
        None,
        &cooperative_ids,
        &cooperative_counts,
        &cooperative_status,
        CAPACITY,
        MAX_VISIBLE,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &bounded_scores,
        &visible_counts,
        &bounded_mask,
        None,
        &bounded_ids,
        &bounded_counts,
        &bounded_status,
        CAPACITY,
        MAX_VISIBLE,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &tiled_scores,
        &visible_counts,
        &tiled_mask,
        None,
        &tiled_ids,
        &tiled_counts,
        &tiled_status,
        CAPACITY,
        MAX_VISIBLE,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "cooperative Lightning score command failed: {:?}",
        command.error()
    );

    let scalar = read_f32(&scalar_scores);
    let cooperative = read_f32(&cooperative_scores);
    let bounded = read_f32(&bounded_scores);
    let tiled = read_f32(&tiled_scores);
    assert_eq!(
        cooperative
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        scalar
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    for query in 0..QUERIES {
        let start = query * CAPACITY;
        assert_eq!(
            bounded[start..start + MAX_VISIBLE]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            cooperative[start..start + MAX_VISIBLE]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(
            bounded[start + MAX_VISIBLE..start + CAPACITY]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
    assert!(cooperative[..MAX_VISIBLE].iter().any(|value| *value != 0.0));
    assert!(
        cooperative[CAPACITY + 9..2 * CAPACITY]
            .iter()
            .all(|value| *value == f32::NEG_INFINITY)
    );
    assert!(
        cooperative[2 * CAPACITY..3 * CAPACITY]
            .iter()
            .all(|value| *value == f32::NEG_INFINITY)
    );
    assert!(
        cooperative[3 * CAPACITY..4 * CAPACITY]
            .iter()
            .all(|value| !value.is_finite())
    );
    let mut tiled_max_abs = 0.0f32;
    let mut tiled_squared_error = 0.0f64;
    let mut tiled_reference_norm = 0.0f64;
    for (query, visible) in [
        (0usize, MAX_VISIBLE),
        (1, 9),
        (4, 17),
        (5, 33),
        (6, 34),
        (7, 39),
        (8, 40),
    ] {
        let start = query * CAPACITY;
        for row in 0..visible {
            let reference = cooperative[start + row];
            let candidate = tiled[start + row];
            tiled_max_abs = tiled_max_abs.max((reference - candidate).abs());
            tiled_squared_error += f64::from(reference - candidate).powi(2);
            tiled_reference_norm += f64::from(reference).powi(2);
        }
    }
    let tiled_relative_rms = (tiled_squared_error / tiled_reference_norm).sqrt();
    eprintln!(
        "tiled F32 Lightning scores max_abs={tiled_max_abs:.9} rel_rms={tiled_relative_rms:.9}"
    );
    assert!(tiled_max_abs <= 1e-5, "tiled score max abs {tiled_max_abs}");
    assert!(
        tiled_relative_rms <= 1e-3,
        "tiled score relative RMS {tiled_relative_rms}"
    );
    assert_eq!(read_i32(&cooperative_mask), read_i32(&scalar_mask));
    assert_eq!(read_i32(&cooperative_ids), read_i32(&scalar_ids));
    assert_eq!(read_i32(&cooperative_counts), read_i32(&scalar_counts));
    assert_eq!(read_i32(&cooperative_status), read_i32(&scalar_status));
    assert_eq!(read_i32(&bounded_mask), read_i32(&scalar_mask));
    assert_eq!(read_i32(&bounded_ids), read_i32(&scalar_ids));
    assert_eq!(read_i32(&bounded_counts), read_i32(&scalar_counts));
    assert_eq!(read_i32(&bounded_status), read_i32(&scalar_status));
    assert_eq!(read_i32(&tiled_mask), read_i32(&scalar_mask));
    assert_eq!(read_i32(&tiled_ids), read_i32(&scalar_ids));
    assert_eq!(read_i32(&tiled_counts), read_i32(&scalar_counts));
    assert_eq!(read_i32(&tiled_status), read_i32(&scalar_status));
    let mut expected_counts = vec![TOP_K as i32; QUERIES];
    expected_counts[2] = 0;
    assert_eq!(read_i32(&cooperative_counts), expected_counts);
    let mut expected_status = vec![0; QUERIES];
    expected_status[2] = 1;
    expected_status[3] = 2;
    assert_eq!(read_i32(&cooperative_status), expected_status);
    assert_eq!(
        &read_i32(&cooperative_ids)[3 * TOP_K..4 * TOP_K],
        &(0..TOP_K as i32).collect::<Vec<_>>()
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn fp4_shadow_preflight_carries_operand_validity_and_fails_closed() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 768;
    const VISIBLE: usize = 513;

    let run = |query_statuses: &[i32], key_statuses: &[i32], visible: i32| {
        let query_status = offset_i32(&ctx, query_statuses, vec![64]);
        let key_status = offset_i32(&ctx, key_statuses, vec![CAPACITY as u64]);
        let requested_visible = offset_i32(&ctx, &[visible], vec![1]);
        let eligible_visible = offset_i32(&ctx, &[99], vec![1]);
        let eligibility_record = offset_i32(&ctx, &[99, 99, 99], vec![3]);
        let query_units = offset_f16(&ctx, &[0.0; 128 * 64], vec![128, 64, 1]);
        let query_scales = offset_i8(&ctx, &[127; 4 * 64], vec![4, 64, 1]);
        let head_weights = offset_f32(&ctx, &[0.0; 64], vec![64, 1]);
        let key_values = offset_i8(&ctx, &[0; 64 * CAPACITY], vec![64, CAPACITY as u64]);
        let key_scales = offset_i8(&ctx, &[127; 4 * CAPACITY], vec![4, CAPACITY as u64]);
        let scores = offset_f32(&ctx, &[0.0; CAPACITY], vec![CAPACITY as u64, 1]);
        let selected_mask = offset_i32(&ctx, &[-1; CAPACITY], vec![CAPACITY as u64, 1]);
        let cache_order_ids = offset_i32(
            &ctx,
            &[-1; DEEPSEEK_V4_CSA_TOP_K],
            vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1],
        );
        let selected_count = offset_i32(&ctx, &[-1], vec![1]);
        let selection_status = offset_i32(&ctx, &[-1], vec![1]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_indexer_fp4_shadow_preflight(
            &ctx,
            &encoder,
            &query_status,
            &key_status,
            &requested_visible,
            &eligible_visible,
            &eligibility_record,
            CAPACITY,
            VISIBLE,
        )
        .unwrap();
        encode_lightning_indexer_scores_fp4_matrix_shadow(
            &ctx,
            &encoder,
            &query_units,
            &query_scales,
            &head_weights,
            &key_values,
            &key_scales,
            &eligible_visible,
            &scores,
            CAPACITY,
            1,
        )
        .unwrap();
        encode_select_top_k_f32(
            &ctx,
            &encoder,
            &scores,
            &eligible_visible,
            &selected_mask,
            None,
            &cache_order_ids,
            &selected_count,
            &selection_status,
            CAPACITY,
            VISIBLE,
            DEEPSEEK_V4_CSA_TOP_K,
            1,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "preflight command failed: {:?}",
            command.error()
        );
        (
            read_i32(&eligible_visible),
            read_i32(&eligibility_record),
            read_i32(&selected_count),
            read_i32(&selection_status),
            read_i32(&selected_mask)
                .into_iter()
                .filter(|&selected| selected != 0)
                .count(),
        )
    };

    let ready_queries = [0; 64];
    let mut ready_keys = [DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; CAPACITY];
    ready_keys[..VISIBLE].fill(0);
    assert_eq!(
        run(&ready_queries, &ready_keys, VISIBLE as i32),
        (vec![513], vec![0, -1, 0], vec![512], vec![0], 512)
    );

    let mut writing_query = ready_queries;
    writing_query[17] = i32::MIN + 1;
    assert_eq!(
        run(&writing_query, &ready_keys, VISIBLE as i32),
        (vec![-1], vec![2, 17, i32::MIN + 1], vec![0], vec![1], 0)
    );

    let mut unavailable_key = ready_keys;
    unavailable_key[VISIBLE - 1] = DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE;
    assert_eq!(
        run(&ready_queries, &unavailable_key, VISIBLE as i32),
        (
            vec![-1],
            vec![3, (VISIBLE - 1) as i32, DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE],
            vec![0],
            vec![1],
            0,
        )
    );
    assert_eq!(
        run(&ready_queries, &ready_keys, CAPACITY as i32 + 1),
        (
            vec![-1],
            vec![1, CAPACITY as i32 + 1, VISIBLE as i32],
            vec![0],
            vec![1],
            0,
        )
    );
    assert_eq!(
        run(&ready_queries, &ready_keys, VISIBLE as i32 - 1),
        (
            vec![-1],
            vec![1, VISIBLE as i32 - 1, VISIBLE as i32],
            vec![0],
            vec![1],
            0,
        )
    );
}

#[test]
fn fp4_metal_primitives_and_packer_match_the_frozen_contract() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let fixture = metal_fp4_fixture();
    assert_eq!(fixture.rounding_cases.len(), 42);
    assert_eq!(fixture.scale_cases.len(), 13);
    assert_eq!(fixture.rows.len(), 5);
    assert_eq!(fixture.pack_rejections.len(), 1);

    let e2m1_values = fixture
        .rounding_cases
        .iter()
        .map(|case| f32::from_bits(case.input_bits))
        .collect::<Vec<_>>();
    let scale_maxima = fixture
        .scale_cases
        .iter()
        .map(|case| f32::from_bits(case.maximum_bits))
        .collect::<Vec<_>>();
    let e2m1_inputs = offset_f32(&ctx, &e2m1_values, vec![e2m1_values.len() as u64]);
    let scale_inputs = offset_f32(&ctx, &scale_maxima, vec![scale_maxima.len() as u64]);
    let e2m1_codes = offset_i8(&ctx, &vec![0xA5; e2m1_values.len()], vec![42]);
    let scale_codes = offset_i8(&ctx, &vec![0xA5; scale_maxima.len()], vec![13]);
    let scale_status = offset_i32(&ctx, &[-1; 13], vec![13]);

    let valid_rows = fixture.rows.len();
    let rejection_rows = fixture.pack_rejections.len();
    let total_rows = valid_rows + rejection_rows + 2;
    let mut pack_inputs = Vec::with_capacity(total_rows * INDEXER_FP4_VALUES_PER_ROW);
    for case in &fixture.rows {
        assert_eq!(case.input_bits.len(), INDEXER_FP4_VALUES_PER_ROW);
        pack_inputs.extend(case.input_bits.iter().map(|&bits| f32::from_bits(bits)));
    }
    for case in &fixture.pack_rejections {
        assert_eq!(case.error_category, "decoded_f32_overflow");
        assert_eq!(case.name, "bf16_max_decodes_beyond_f32");
        let mut row = [0.0f32; INDEXER_FP4_VALUES_PER_ROW];
        row[case.dimension] = f32::from_bits(case.input_bits);
        pack_inputs.extend_from_slice(&row);
    }
    let mut source_nonfinite = [0.0f32; INDEXER_FP4_VALUES_PER_ROW];
    source_nonfinite[17] = f32::INFINITY;
    pack_inputs.extend_from_slice(&source_nonfinite);
    let mut bf16_nonfinite = [0.0f32; INDEXER_FP4_VALUES_PER_ROW];
    bf16_nonfinite[23] = f32::MAX;
    pack_inputs.extend_from_slice(&bf16_nonfinite);

    let pack_inputs = offset_f32(
        &ctx,
        &pack_inputs,
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, total_rows as u64],
    );
    let packed_values = offset_i8(
        &ctx,
        &vec![0xA5; total_rows * INDEXER_FP4_VALUE_BYTES],
        vec![INDEXER_FP4_VALUE_BYTES as u64, total_rows as u64],
    );
    let packed_scales = offset_i8(
        &ctx,
        &vec![0xA5; total_rows * INDEXER_FP4_SCALE_BYTES],
        vec![INDEXER_FP4_SCALE_BYTES as u64, total_rows as u64],
    );
    let pack_status = offset_i32(&ctx, &vec![-1; total_rows], vec![total_rows as u64]);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_indexer_fp4_contract_primitives(
        &ctx,
        &encoder,
        &e2m1_inputs,
        &scale_inputs,
        &e2m1_codes,
        &scale_codes,
        &scale_status,
        e2m1_values.len(),
        scale_maxima.len(),
    )
    .unwrap();
    encode_pack_indexer_fp4_rows_shadow(
        &ctx,
        &encoder,
        &pack_inputs,
        &packed_values,
        &packed_scales,
        &pack_status,
        total_rows,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "FP4 pack command failed: {:?}",
        command.error()
    );

    assert_eq!(
        read_u8(&e2m1_codes),
        fixture
            .rounding_cases
            .iter()
            .map(|case| case.code)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_u8(&scale_codes),
        fixture
            .scale_cases
            .iter()
            .map(|case| case.code)
            .collect::<Vec<_>>()
    );
    assert_eq!(read_i32(&scale_status), vec![0; fixture.scale_cases.len()]);

    let actual_values = read_u8(&packed_values);
    let actual_scales = read_u8(&packed_scales);
    let actual_status = read_i32(&pack_status);
    for (row, case) in fixture.rows.iter().enumerate() {
        assert_eq!(actual_status[row], 0, "{} status", case.name);
        assert_eq!(
            &actual_values[row * INDEXER_FP4_VALUE_BYTES..(row + 1) * INDEXER_FP4_VALUE_BYTES],
            &case.packed_bytes[..INDEXER_FP4_VALUE_BYTES],
            "{} values",
            case.name
        );
        assert_eq!(
            &actual_scales[row * INDEXER_FP4_SCALE_BYTES..(row + 1) * INDEXER_FP4_SCALE_BYTES],
            &case.packed_bytes[INDEXER_FP4_VALUE_BYTES..],
            "{} scales",
            case.name
        );
    }
    assert_eq!(
        &actual_status[valid_rows..],
        &[3, 1, 1],
        "overflow, source nonfinite, and BF16 nonfinite statuses"
    );
    for row in valid_rows..total_rows {
        assert!(
            actual_values[row * INDEXER_FP4_VALUE_BYTES..(row + 1) * INDEXER_FP4_VALUE_BYTES]
                .iter()
                .all(|&byte| byte == 0xA5),
            "rejected row {row} partially published values"
        );
        assert!(
            actual_scales[row * INDEXER_FP4_SCALE_BYTES..(row + 1) * INDEXER_FP4_SCALE_BYTES]
                .iter()
                .all(|&byte| byte == 0xA5),
            "rejected row {row} partially published scales"
        );
    }
}

#[test]
fn fp4_metal_raw_row_validation_classifies_the_frozen_invalid_domain() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let fixture = metal_fp4_fixture();
    assert_eq!(fixture.invalid_rows.len(), 20);
    let mut rows = vec![fixture.rows[0].packed_bytes.clone()];
    for case in &fixture.invalid_rows {
        let mut row = vec![0u8; INDEXER_FP4_ROW_BYTES];
        row[INDEXER_FP4_VALUE_BYTES..].fill(127);
        for mutation in &case.mutations {
            row[mutation.byte_index] = mutation.byte_value;
        }
        rows.push(row);
    }
    let (values, scales) = split_fp4_rows(&rows);
    let values = offset_i8(
        &ctx,
        &values,
        vec![INDEXER_FP4_VALUE_BYTES as u64, rows.len() as u64],
    );
    let scales = offset_i8(
        &ctx,
        &scales,
        vec![INDEXER_FP4_SCALE_BYTES as u64, rows.len() as u64],
    );
    let status = offset_i32(&ctx, &vec![-1; rows.len()], vec![rows.len() as u64]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_validate_indexer_fp4_rows_shadow(&ctx, &encoder, &values, &scales, &status, rows.len())
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    let actual = read_i32(&status);
    assert_eq!(actual[0], 0);
    for (index, case) in fixture.invalid_rows.iter().enumerate() {
        let expected = match case.error_category.as_str() {
            "noncanonical_scale_code" => 2,
            "decoded_f32_overflow" => 3,
            category => panic!("unexpected invalid-row category {category}"),
        };
        assert_eq!(actual[index + 1], expected, "{}", case.name);
    }
}

#[test]
fn fp4_query_unpack_is_exact_for_codes_offsets_tails_and_128_queries() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const QUERIES: usize = 128;

    fn expected_unit_bits(code: u8) -> u16 {
        const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let magnitude = MAGNITUDES[usize::from(code & 0x07)];
        let value = if code & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        };
        half::f16::from_f32(value).to_bits()
    }

    fn packed_pattern(rows: usize) -> Vec<u8> {
        (0..rows * INDEXER_FP4_VALUE_BYTES)
            .map(|index| {
                let row = index / INDEXER_FP4_VALUE_BYTES;
                let byte = index % INDEXER_FP4_VALUE_BYTES;
                let low = ((row * 5 + byte * 3) % 16) as u8;
                let high = ((row * 11 + byte * 7 + 8) % 16) as u8;
                low | (high << 4)
            })
            .collect()
    }

    fn expected_units(packed: &[u8]) -> Vec<u16> {
        packed
            .iter()
            .flat_map(|&byte| {
                [
                    expected_unit_bits(byte & 0x0f),
                    expected_unit_bits(byte >> 4),
                ]
            })
            .collect()
    }

    let tail_rows = 3;
    let tail_packed = packed_pattern(tail_rows);
    let tail_expected = expected_units(&tail_packed);
    assert!(tail_expected.contains(&0x8000));
    let tail_values = offset_i8(
        &ctx,
        &tail_packed,
        vec![INDEXER_FP4_VALUE_BYTES as u64, tail_rows as u64],
    );
    let tail_status = offset_i32(&ctx, &[0; 3], vec![tail_rows as u64]);
    let tail_units = offset_f16(
        &ctx,
        &vec![1.0; tail_rows * INDEXER_FP4_VALUES_PER_ROW],
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, tail_rows as u64],
    );

    let batched_rows = HEADS * QUERIES;
    let batched_packed = packed_pattern(batched_rows);
    let batched_expected = expected_units(&batched_packed);
    let batched_values = offset_i8(
        &ctx,
        &batched_packed,
        vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, QUERIES as u64],
    );
    let batched_status = offset_i32(&ctx, &vec![0; batched_rows], vec![batched_rows as u64]);
    let batched_first = offset_f16(
        &ctx,
        &vec![1.0; batched_rows * INDEXER_FP4_VALUES_PER_ROW],
        vec![
            INDEXER_FP4_VALUES_PER_ROW as u64,
            HEADS as u64,
            QUERIES as u64,
        ],
    );
    let batched_second = offset_f16(
        &ctx,
        &vec![1.0; batched_rows * INDEXER_FP4_VALUES_PER_ROW],
        vec![
            INDEXER_FP4_VALUES_PER_ROW as u64,
            HEADS as u64,
            QUERIES as u64,
        ],
    );
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_unpack_indexer_fp4_units_shadow(
        &ctx,
        &encoder,
        &tail_values,
        &tail_status,
        &tail_units,
        tail_rows,
    )
    .unwrap();
    for output in [&batched_first, &batched_second] {
        encode_unpack_indexer_fp4_units_shadow(
            &ctx,
            &encoder,
            &batched_values,
            &batched_status,
            output,
            batched_rows,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(read_f16_bits(&tail_units), tail_expected);
    assert_eq!(read_f16_bits(&batched_first), batched_expected);
    assert_eq!(
        read_f16_bits(&batched_second),
        read_f16_bits(&batched_first)
    );
}

#[test]
fn fp4_matrix_shadow_matches_frozen_score_vectors_and_top2() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    let fixture = metal_fp4_fixture();
    assert_eq!(fixture.score_cases.len(), 3);
    for case in &fixture.score_cases {
        assert_eq!(case.query_rows.len(), 2);
        assert_eq!(case.key_rows.len(), 3);
        let mut query_values = vec![0u8; HEADS * INDEXER_FP4_VALUE_BYTES];
        let mut query_scales = vec![127u8; HEADS * INDEXER_FP4_SCALE_BYTES];
        for (head, row) in case.query_rows.iter().enumerate() {
            query_values[head * INDEXER_FP4_VALUE_BYTES..(head + 1) * INDEXER_FP4_VALUE_BYTES]
                .copy_from_slice(&row[..INDEXER_FP4_VALUE_BYTES]);
            query_scales[head * INDEXER_FP4_SCALE_BYTES..(head + 1) * INDEXER_FP4_SCALE_BYTES]
                .copy_from_slice(&row[INDEXER_FP4_VALUE_BYTES..]);
        }
        let (key_values, key_scales) = split_fp4_rows(&case.key_rows);
        let mut weights = vec![0.0f32; HEADS];
        for (weight, &bits) in weights.iter_mut().zip(&case.scaled_head_weight_bits) {
            *weight = f32::from_bits(bits);
        }
        let query_values = offset_i8(
            &ctx,
            &query_values,
            vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, 1],
        );
        let query_scales = offset_i8(
            &ctx,
            &query_scales,
            vec![INDEXER_FP4_SCALE_BYTES as u64, HEADS as u64, 1],
        );
        let query_status = offset_i32(&ctx, &vec![0; HEADS], vec![HEADS as u64]);
        let query_units = offset_f16(
            &ctx,
            &vec![0.0; HEADS * INDEXER_FP4_VALUES_PER_ROW],
            vec![INDEXER_FP4_VALUES_PER_ROW as u64, HEADS as u64, 1],
        );
        let head_weights = offset_f32(&ctx, &weights, vec![HEADS as u64, 1]);
        let key_values = offset_i8(
            &ctx,
            &key_values,
            vec![INDEXER_FP4_VALUE_BYTES as u64, case.key_rows.len() as u64],
        );
        let key_scales = offset_i8(
            &ctx,
            &key_scales,
            vec![INDEXER_FP4_SCALE_BYTES as u64, case.key_rows.len() as u64],
        );
        let visible_counts = offset_i32(&ctx, &[case.key_rows.len() as i32], vec![1]);
        let first = offset_f32(&ctx, &[0.0; 3], vec![3, 1]);
        let second = offset_f32(&ctx, &[0.0; 3], vec![3, 1]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_unpack_indexer_fp4_units_shadow(
            &ctx,
            &encoder,
            &query_values,
            &query_status,
            &query_units,
            HEADS,
        )
        .unwrap();
        for output in [&first, &second] {
            encode_lightning_indexer_scores_fp4_matrix_shadow(
                &ctx,
                &encoder,
                &query_units,
                &query_scales,
                &head_weights,
                &key_values,
                &key_scales,
                &visible_counts,
                output,
                case.key_rows.len(),
                1,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "{} command failed: {:?}",
            case.name,
            command.error()
        );
        let actual = read_f32(&first);
        let repeat = read_f32(&second);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeat
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} repeat",
            case.name
        );
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            case.score_bits,
            "{} score bits",
            case.name
        );
        assert_eq!(
            top_k_indices(&actual, 2).unwrap(),
            case.top2,
            "{} top2",
            case.name
        );
        let query_rows = case
            .query_rows
            .iter()
            .map(|row| fp4_row_from_bytes(row))
            .collect::<Vec<_>>();
        let key_rows = case
            .key_rows
            .iter()
            .map(|row| fp4_row_from_bytes(row))
            .collect::<Vec<_>>();
        let oracle = packed_indexer_scores(
            &query_rows,
            &case
                .scaled_head_weight_bits
                .iter()
                .map(|&bits| f32::from_bits(bits))
                .collect::<Vec<_>>(),
            &key_rows,
        )
        .unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            oracle
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn fp4_shadow_runs_pack_unpack_and_score_without_host_reconstruction() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const ROWS: usize = 9;
    let query_inputs = (0..HEADS * INDEXER_FP4_VALUES_PER_ROW)
        .map(|index| ((index * 17 + index / 29 + 5) % 257) as f32 * 0.0011 - 0.14)
        .collect::<Vec<_>>();
    let key_inputs = (0..ROWS * INDEXER_FP4_VALUES_PER_ROW)
        .map(|index| {
            let row = index / INDEXER_FP4_VALUES_PER_ROW;
            ((index * 31 + row * 13 + 11) % 263) as f32 * 0.0013 - 0.17
        })
        .collect::<Vec<_>>();
    let weights = (0..HEADS)
        .map(|head| {
            let magnitude = 0.001 + (head * 7 % 19) as f32 * 0.0004;
            if head % 9 == 0 { -magnitude } else { magnitude }
        })
        .collect::<Vec<_>>();
    let queries = offset_f32(
        &ctx,
        &query_inputs,
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, HEADS as u64, 1],
    );
    let query_values = offset_i8(
        &ctx,
        &vec![0xA5; HEADS * INDEXER_FP4_VALUE_BYTES],
        vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, 1],
    );
    let query_scales = offset_i8(
        &ctx,
        &vec![0xA5; HEADS * INDEXER_FP4_SCALE_BYTES],
        vec![INDEXER_FP4_SCALE_BYTES as u64, HEADS as u64, 1],
    );
    let query_status = offset_i32(&ctx, &vec![-1; HEADS], vec![HEADS as u64]);
    let query_units = offset_f16(
        &ctx,
        &vec![1.0; HEADS * INDEXER_FP4_VALUES_PER_ROW],
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, HEADS as u64, 1],
    );
    let keys = offset_f32(
        &ctx,
        &key_inputs,
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, ROWS as u64],
    );
    let key_values = offset_i8(
        &ctx,
        &vec![0xA5; ROWS * INDEXER_FP4_VALUE_BYTES],
        vec![INDEXER_FP4_VALUE_BYTES as u64, ROWS as u64],
    );
    let key_scales = offset_i8(
        &ctx,
        &[0xA5; ROWS * INDEXER_FP4_SCALE_BYTES],
        vec![INDEXER_FP4_SCALE_BYTES as u64, ROWS as u64],
    );
    let key_status = offset_i32(&ctx, &[-1; ROWS], vec![ROWS as u64]);
    let head_weights = offset_f32(&ctx, &weights, vec![HEADS as u64, 1]);
    let visible_counts = offset_i32(&ctx, &[ROWS as i32], vec![1]);
    let scores = offset_f32(&ctx, &[0.0; ROWS], vec![ROWS as u64, 1]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_pack_indexer_fp4_rows_shadow(
        &ctx,
        &encoder,
        &queries,
        &query_values,
        &query_scales,
        &query_status,
        HEADS,
    )
    .unwrap();
    encode_unpack_indexer_fp4_units_shadow(
        &ctx,
        &encoder,
        &query_values,
        &query_status,
        &query_units,
        HEADS,
    )
    .unwrap();
    encode_pack_indexer_fp4_rows_shadow(
        &ctx,
        &encoder,
        &keys,
        &key_values,
        &key_scales,
        &key_status,
        ROWS,
    )
    .unwrap();
    encode_lightning_indexer_scores_fp4_matrix_shadow(
        &ctx,
        &encoder,
        &query_units,
        &query_scales,
        &head_weights,
        &key_values,
        &key_scales,
        &visible_counts,
        &scores,
        ROWS,
        1,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());
    assert_eq!(read_i32(&query_status), vec![0; HEADS]);
    assert_eq!(read_i32(&key_status), vec![0; ROWS]);

    let query_rows = query_inputs
        .chunks_exact(INDEXER_FP4_VALUES_PER_ROW)
        .map(|row| pack_indexer_fp4_row(row).unwrap())
        .collect::<Vec<_>>();
    let key_rows = key_inputs
        .chunks_exact(INDEXER_FP4_VALUES_PER_ROW)
        .map(|row| pack_indexer_fp4_row(row).unwrap())
        .collect::<Vec<_>>();
    let expected = packed_indexer_scores(&query_rows, &weights, &key_rows).unwrap();
    let actual = read_f32(&scores);
    let squared_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| f64::from(actual - expected).powi(2))
        .sum::<f64>();
    let reference_norm = expected
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    let relative_rms = (squared_error / reference_norm).sqrt();
    for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let allowed = 2.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "row {row}: {actual} vs {expected}"
        );
    }
    assert!(relative_rms <= 2.0e-5, "relative RMS {relative_rms}");
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn fp4_shadow_layer_addressed_output_matches_instrumented_output_exactly() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const CAPACITY: usize = 768;
    const FIRST_VISIBLE: usize = 513;
    const SECOND_VISIBLE: usize = 517;
    const FIRST_LAYER: usize = 2;
    const SECOND_LAYER: usize = 4;
    let first_queries = offset_f32(
        &ctx,
        &(0..64 * INDEXER_FP4_VALUES_PER_ROW)
            .map(|index| 0.01 + (index % 17) as f32 * 0.0001)
            .collect::<Vec<_>>(),
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, 64, 1],
    );
    let second_queries = offset_f32(
        &ctx,
        &(0..64 * INDEXER_FP4_VALUES_PER_ROW)
            .map(|index| 0.02 + (index % 23) as f32 * 0.0002)
            .collect::<Vec<_>>(),
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, 64, 1],
    );
    let head_weights = offset_f32(
        &ctx,
        &(0..64)
            .map(|head| 0.001 + (head % 13) as f32 * 0.0002)
            .collect::<Vec<_>>(),
        vec![64, 1],
    );
    let attention_cache = offset_f16(&ctx, &vec![0.0; 512 * CAPACITY], vec![512, 768]);
    let indexer_cache = offset_f16(&ctx, &vec![0.0; 128 * CAPACITY], vec![128, 768]);
    let key_rows = (0..CAPACITY)
        .map(|row| {
            let magnitude = if row >= FIRST_VISIBLE {
                0.25 + (row - FIRST_VISIBLE) as f32 * 0.01
            } else {
                0.001 + row as f32 * 0.0001
            };
            pack_indexer_fp4_row(&vec![magnitude; INDEXER_FP4_VALUES_PER_ROW])
                .unwrap()
                .as_bytes()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let (key_values, key_scales) = split_fp4_rows(&key_rows);
    let sidecar = DeepSeekV4IndexerFp4Sidecar {
        enabled: true,
        capacity_rows: CAPACITY,
        values: offset_i8(
            &ctx,
            &key_values,
            vec![INDEXER_FP4_VALUE_BYTES as u64, CAPACITY as u64],
        ),
        scales: offset_i8(
            &ctx,
            &key_scales,
            vec![INDEXER_FP4_SCALE_BYTES as u64, CAPACITY as u64],
        ),
        status: offset_i32(&ctx, &vec![0; CAPACITY], vec![CAPACITY as u64]),
    };
    let first_rows = DeepSeekV4CsaRows {
        attention_cache: &attention_cache,
        indexer_cache: &indexer_cache,
        indexer_fp4_sidecar: Some(&sidecar),
        count: FIRST_VISIBLE,
        capacity_rows: CAPACITY,
    };
    let second_rows = DeepSeekV4CsaRows {
        count: SECOND_VISIBLE,
        ..first_rows
    };
    let scratch = DeepSeekV4Fp4ShadowScratch::new(&ctx, CAPACITY).unwrap();
    let first_requested = offset_i32(&ctx, &[FIRST_VISIBLE as i32], vec![1]);
    let second_requested = offset_i32(&ctx, &[SECOND_VISIBLE as i32], vec![1]);
    let run_instrumented =
        |queries: &MetalTensor, rows: DeepSeekV4CsaRows<'_>, requested: &MetalTensor| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            scratch
                .encode(&ctx, &encoder, queries, &head_weights, rows, requested)
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            scratch.validate_completed().unwrap();
            read_i32(&scratch.cache_order_ids)
        };
    let first_instrumented = run_instrumented(&first_queries, first_rows, &first_requested);
    let second_instrumented = run_instrumented(&second_queries, second_rows, &second_requested);
    assert_ne!(first_instrumented, second_instrumented);

    let selection_records = DeepSeekV4LayerSelectionRecords::new(&ctx).unwrap();
    selection_records.reset_for_token().unwrap();
    let first_selection = selection_records.layer(FIRST_LAYER).unwrap();
    let second_selection = selection_records.layer(SECOND_LAYER).unwrap();
    host_write_i32(
        &first_selection.visible_count,
        &[FIRST_VISIBLE as i32],
        "first layer-addressed FP4 requested visibility",
    )
    .unwrap();
    host_write_i32(
        &second_selection.visible_count,
        &[SECOND_VISIBLE as i32],
        "second layer-addressed FP4 requested visibility",
    )
    .unwrap();
    let fp4_records = DeepSeekV4LayerFp4SelectionRecords::new(&ctx).unwrap();
    fp4_records
        .reset_for_token(
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    let first_fp4 = fp4_records.layer(FIRST_LAYER).unwrap();
    let second_fp4 = fp4_records.layer(SECOND_LAYER).unwrap();
    let collapsed_command = ctx.queue.commandBuffer().unwrap();
    let collapsed_encoder = KernelEncoder::begin(&collapsed_command);
    scratch
        .encode_into(
            &ctx,
            &collapsed_encoder,
            &first_queries,
            &head_weights,
            first_rows,
            &first_selection.visible_count,
            first_fp4.output(&first_selection),
        )
        .unwrap();
    scratch
        .encode_into(
            &ctx,
            &collapsed_encoder,
            &second_queries,
            &head_weights,
            second_rows,
            &second_selection.visible_count,
            second_fp4.output(&second_selection),
        )
        .unwrap();
    collapsed_encoder.end();
    collapsed_command.commit();
    collapsed_command.waitUntilCompleted();
    assert!(collapsed_command.error().is_none());

    selection_records
        .read_completed()
        .unwrap()
        .validate_layer(FIRST_LAYER, FIRST_VISIBLE)
        .unwrap();
    selection_records
        .read_completed()
        .unwrap()
        .validate_layer(SECOND_LAYER, SECOND_VISIBLE)
        .unwrap();
    let completed = fp4_records.read_completed().unwrap();
    let first_collapsed = completed
        .validate_layer(
            FIRST_LAYER,
            FIRST_VISIBLE,
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    let second_collapsed = completed
        .validate_layer(
            SECOND_LAYER,
            SECOND_VISIBLE,
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    assert_eq!(first_collapsed, first_instrumented);
    assert_eq!(second_collapsed, second_instrumented);
    for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
        if layer != FIRST_LAYER && layer != SECOND_LAYER {
            completed
                .validate_inactive_layer(
                    layer,
                    DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                    DeepSeekV4Fp4SelectionSource::Fp4,
                )
                .unwrap();
        }
    }

    let mut instrumented_trace = diagnostics::DeepSeekV4Fp4CounterfactualTrace::default();
    instrumented_trace
        .record(
            DeepSeekV4Fp4ShadowExecution::Singleton,
            DeepSeekV4Fp4SelectionSource::Fp4,
            2_051,
            FIRST_LAYER,
            FIRST_VISIBLE as i32,
            &first_instrumented,
        )
        .unwrap();
    let mut collapsed_trace = diagnostics::DeepSeekV4Fp4CounterfactualTrace::default();
    collapsed_trace
        .record(
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
            2_051,
            FIRST_LAYER,
            FIRST_VISIBLE as i32,
            first_collapsed,
        )
        .unwrap();
    assert_ne!(instrumented_trace.digest().0, collapsed_trace.digest().0);
    assert_eq!(instrumented_trace.digest().1, collapsed_trace.digest().1);
}

#[test]
fn fp4_matrix_shadow_covers_batched_queries_tails_offsets_and_envelope() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const ROWS: usize = 9;
    const QUERIES: usize = 128;

    let query_inputs = (0..QUERIES * HEADS * INDEXER_FP4_VALUES_PER_ROW)
        .map(|index| {
            let row = index / INDEXER_FP4_VALUES_PER_ROW;
            let dimension = index % INDEXER_FP4_VALUES_PER_ROW;
            let tag = (index * 19 + row * 7 + dimension * 11 + 3) % 257;
            (tag as f32 - 128.0) * 0.0011
        })
        .collect::<Vec<_>>();
    let query_rows = query_inputs
        .chunks_exact(INDEXER_FP4_VALUES_PER_ROW)
        .map(|row| pack_indexer_fp4_row(row).unwrap())
        .collect::<Vec<_>>();
    let key_inputs = (0..ROWS * INDEXER_FP4_VALUES_PER_ROW)
        .map(|index| {
            let row = index / INDEXER_FP4_VALUES_PER_ROW;
            let dimension = index % INDEXER_FP4_VALUES_PER_ROW;
            let tag = (index * 31 + row * 13 + dimension * 5 + 17) % 263;
            (tag as f32 - 131.0) * 0.0013
        })
        .collect::<Vec<_>>();
    let key_rows = key_inputs
        .chunks_exact(INDEXER_FP4_VALUES_PER_ROW)
        .map(|row| pack_indexer_fp4_row(row).unwrap())
        .collect::<Vec<_>>();
    let weights = (0..QUERIES * HEADS)
        .map(|index| {
            let magnitude = 0.001 + (index * 11 % 23) as f32 * 0.0003;
            if index % 7 == 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect::<Vec<_>>();
    let visible = (0..QUERIES)
        .map(|query| [1, 7, 8, 9][query % 4])
        .collect::<Vec<i32>>();

    let query_bytes = query_rows
        .iter()
        .map(|row| row.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let key_bytes = key_rows
        .iter()
        .map(|row| row.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let (query_values, query_scales) = split_fp4_rows(&query_bytes);
    let (key_values, key_scales) = split_fp4_rows(&key_bytes);
    let query_values = offset_i8(
        &ctx,
        &query_values,
        vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, QUERIES as u64],
    );
    let query_scales = offset_i8(
        &ctx,
        &query_scales,
        vec![INDEXER_FP4_SCALE_BYTES as u64, HEADS as u64, QUERIES as u64],
    );
    let query_status = offset_i32(
        &ctx,
        &vec![0; HEADS * QUERIES],
        vec![(HEADS * QUERIES) as u64],
    );
    let query_units = offset_f16(
        &ctx,
        &vec![0.0; HEADS * QUERIES * INDEXER_FP4_VALUES_PER_ROW],
        vec![
            INDEXER_FP4_VALUES_PER_ROW as u64,
            HEADS as u64,
            QUERIES as u64,
        ],
    );
    let head_weights = offset_f32(&ctx, &weights, vec![HEADS as u64, QUERIES as u64]);
    let key_values = offset_i8(
        &ctx,
        &key_values,
        vec![INDEXER_FP4_VALUE_BYTES as u64, ROWS as u64],
    );
    let key_scales = offset_i8(
        &ctx,
        &key_scales,
        vec![INDEXER_FP4_SCALE_BYTES as u64, ROWS as u64],
    );
    let visible_counts = offset_i32(&ctx, &visible, vec![QUERIES as u64]);
    let first = offset_f32(
        &ctx,
        &vec![0.0; ROWS * QUERIES],
        vec![ROWS as u64, QUERIES as u64],
    );
    let second = offset_f32(
        &ctx,
        &vec![0.0; ROWS * QUERIES],
        vec![ROWS as u64, QUERIES as u64],
    );
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_unpack_indexer_fp4_units_shadow(
        &ctx,
        &encoder,
        &query_values,
        &query_status,
        &query_units,
        HEADS * QUERIES,
    )
    .unwrap();
    for output in [&first, &second] {
        encode_lightning_indexer_scores_fp4_matrix_shadow(
            &ctx,
            &encoder,
            &query_units,
            &query_scales,
            &head_weights,
            &key_values,
            &key_scales,
            &visible_counts,
            output,
            ROWS,
            QUERIES,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let actual = read_f32(&first);
    let repeat = read_f32(&second);
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        repeat
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let mut squared_error = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for query in 0..QUERIES {
        let query_start = query * HEADS;
        let expected = packed_indexer_scores(
            &query_rows[query_start..query_start + HEADS],
            &weights[query_start..query_start + HEADS],
            &key_rows,
        )
        .unwrap();
        let visible_rows = visible[query] as usize;
        for row in 0..ROWS {
            let value = actual[query * ROWS + row];
            if row >= visible_rows {
                assert_eq!(value, f32::NEG_INFINITY, "query {query} row {row}");
                continue;
            }
            let reference = expected[row];
            let error = (value - reference).abs();
            let allowed = 2.0e-5 * reference.abs().max(1.0);
            assert!(
                error <= allowed,
                "query {query} row {row}: {value} vs {reference}, error {error} > {allowed}"
            );
            max_abs = max_abs.max(error);
            squared_error += f64::from(error).powi(2);
            reference_norm += f64::from(reference).powi(2);
        }
    }
    let relative_rms = if reference_norm == 0.0 {
        assert_eq!(squared_error, 0.0);
        0.0
    } else {
        (squared_error / reference_norm).sqrt()
    };
    eprintln!(
        "FP4 matrix shadow batched differential max_abs={max_abs:.9} rel_rms={relative_rms:.9}"
    );
    assert!(
        relative_rms <= 2.0e-5,
        "FP4 matrix shadow relative RMS {relative_rms}"
    );
}

#[test]
fn fp4_matrix_shadow_preserves_selector_decisions_and_fallback() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const ROWS: usize = 1_025;
    const QUERIES: usize = 3;
    const TOP_K: usize = 512;

    let mut query_values = vec![0u8; QUERIES * HEADS * INDEXER_FP4_VALUE_BYTES];
    let query_scales = vec![127u8; QUERIES * HEADS * INDEXER_FP4_SCALE_BYTES];
    query_values[0] = 2;
    query_values[2 * HEADS * INDEXER_FP4_VALUE_BYTES] = 2;
    let mut key_values = vec![0u8; ROWS * INDEXER_FP4_VALUE_BYTES];
    for row in 0..ROWS {
        key_values[row * INDEXER_FP4_VALUE_BYTES] = if row < 513 { 10 } else { 2 };
    }
    let key_scales = vec![127u8; ROWS * INDEXER_FP4_SCALE_BYTES];
    let mut weights = vec![0.0f32; QUERIES * HEADS];
    weights[0] = 1.0;
    weights[HEADS] = 1.0;
    weights[2 * HEADS] = f32::NAN;

    let query_values = offset_i8(
        &ctx,
        &query_values,
        vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, QUERIES as u64],
    );
    let query_scales = offset_i8(
        &ctx,
        &query_scales,
        vec![INDEXER_FP4_SCALE_BYTES as u64, HEADS as u64, QUERIES as u64],
    );
    let query_pack_status = offset_i32(
        &ctx,
        &vec![0; HEADS * QUERIES],
        vec![(HEADS * QUERIES) as u64],
    );
    let query_units = offset_f16(
        &ctx,
        &vec![0.0; HEADS * QUERIES * INDEXER_FP4_VALUES_PER_ROW],
        vec![
            INDEXER_FP4_VALUES_PER_ROW as u64,
            HEADS as u64,
            QUERIES as u64,
        ],
    );
    let head_weights = offset_f32(&ctx, &weights, vec![HEADS as u64, QUERIES as u64]);
    let key_values = offset_i8(
        &ctx,
        &key_values,
        vec![INDEXER_FP4_VALUE_BYTES as u64, ROWS as u64],
    );
    let key_scales = offset_i8(
        &ctx,
        &key_scales,
        vec![INDEXER_FP4_SCALE_BYTES as u64, ROWS as u64],
    );
    let visible_counts = offset_i32(&ctx, &[ROWS as i32; QUERIES], vec![QUERIES as u64]);
    let scores = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();
    let selected_mask = MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();
    let selected_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let selected_counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_unpack_indexer_fp4_units_shadow(
        &ctx,
        &encoder,
        &query_values,
        &query_pack_status,
        &query_units,
        HEADS * QUERIES,
    )
    .unwrap();
    encode_lightning_indexer_scores_fp4_matrix_shadow(
        &ctx,
        &encoder,
        &query_units,
        &query_scales,
        &head_weights,
        &key_values,
        &key_scales,
        &visible_counts,
        &scores,
        ROWS,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &selected_mask,
        None,
        &selected_ids,
        &selected_counts,
        &status,
        ROWS,
        ROWS,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let scores = read_f32(&scores);
    assert!(scores[..513].iter().all(|score| *score == 0.0));
    assert!(scores[513..ROWS].iter().all(|score| *score == 1.0));
    assert!(scores[ROWS..2 * ROWS].iter().all(|score| *score == 0.0));
    assert!(scores[2 * ROWS..].iter().all(|score| !score.is_finite()));
    assert_eq!(read_i32(&status), vec![0, 0, 2]);
    assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32; QUERIES]);
    let ids = read_i32(&selected_ids);
    assert_eq!(&ids[..TOP_K], &(513..=1_024).collect::<Vec<i32>>());
    let stable_tie = (0..TOP_K as i32).collect::<Vec<_>>();
    assert_eq!(&ids[TOP_K..2 * TOP_K], &stable_tie);
    assert_eq!(&ids[2 * TOP_K..], &stable_tie);
}

#[test]
fn fp4_matrix_shadow_geometry_is_checked_at_production_limits() {
    assert!(validate_fp4_lightning_score_offsets(262_144, 128).is_ok());
    assert!(validate_fp4_lightning_score_offsets(u32::MAX as usize, 1).is_err());
    assert!(validate_fp4_matrix_score_geometry("fp4", 32, 256, 2_560).is_ok());
    assert!(validate_fp4_matrix_score_geometry("fp4", 16, 256, 2_560).is_err());
    assert!(validate_fp4_matrix_score_geometry("fp4", 32, 255, 2_560).is_err());
    assert!(validate_fp4_matrix_score_geometry("fp4", 32, 256, 2_559).is_err());
    assert!(validate_fp4_pack_geometry(32, 128, 596).is_ok());
    assert!(validate_fp4_pack_geometry(16, 128, 596).is_err());
    assert!(validate_fp4_pack_geometry(32, 127, 596).is_err());
    assert!(validate_fp4_pack_geometry(32, 128, 595).is_err());
}

#[test]
fn matrix_ceiling_lightning_scores_match_f16_cpu_oracle() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const ROWS: usize = 9;
    const QUERIES: usize = 3;

    let query_values = (0..QUERIES * HEADS * DIM)
        .map(|index| {
            let head = (index / DIM) % HEADS;
            let dimension = index % DIM;
            let tag = (index * 17 + head * 11 + dimension * 3 + 5) % 257;
            half::f16::from_f32((tag as f32 - 128.0) * 0.0009).to_f32()
        })
        .collect::<Vec<_>>();
    let weight_values = (0..QUERIES * HEADS)
        .map(|index| 0.002 + (index * 13 % 31) as f32 * 0.0005)
        .collect::<Vec<_>>();
    let key_values = (0..ROWS * DIM)
        .map(|index| {
            let row = index / DIM;
            let dimension = index % DIM;
            let tag = (row * 29 + dimension * 7 + row * dimension + 3) % 263;
            half::f16::from_f32((tag as f32 - 131.0) * 0.0008).to_f32()
        })
        .collect::<Vec<_>>();
    let queries = offset_f16(
        &ctx,
        &query_values,
        vec![DIM as u64, HEADS as u64, QUERIES as u64],
    );
    let head_weights = offset_f32(&ctx, &weight_values, vec![HEADS as u64, QUERIES as u64]);
    let keys = offset_f16(&ctx, &key_values, vec![DIM as u64, ROWS as u64]);
    let visible = [ROWS as i32, 7, -1];
    let visible_counts = offset_i32(&ctx, &visible, vec![QUERIES as u64]);
    let first = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();
    let second = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for output in [&first, &second] {
        encode_lightning_indexer_scores_f16_matrix(
            &ctx,
            &encoder,
            &queries,
            &head_weights,
            &keys,
            &visible_counts,
            output,
            HEADS,
            DIM,
            ROWS,
            ROWS,
            QUERIES,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "matrix-ceiling Lightning score command failed: {:?}",
        command.error()
    );

    let actual = read_f32(&first);
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        read_f32(&second)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "matrix-ceiling scorer must repeat bit-for-bit"
    );
    let mut expected = vec![f32::NEG_INFINITY; ROWS * QUERIES];
    for query in 0..QUERIES {
        if visible[query] < 0 {
            continue;
        }
        for row in 0..usize::try_from(visible[query]).unwrap() {
            let mut score = 0.0f32;
            for head in 0..HEADS {
                let mut dot = 0.0f32;
                for dimension in 0..DIM {
                    dot += query_values[(query * HEADS + head) * DIM + dimension]
                        * key_values[row * DIM + dimension];
                }
                score += dot.max(0.0) * weight_values[query * HEADS + head];
            }
            expected[query * ROWS + row] = score;
        }
    }
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        if expected == f32::NEG_INFINITY {
            assert_eq!(actual, f32::NEG_INFINITY);
            continue;
        }
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        squared_error += f64::from(error).powi(2);
        reference_norm += f64::from(expected).powi(2);
    }
    let relative_rms = (squared_error / reference_norm).sqrt();
    eprintln!(
        "matrix-ceiling Lightning CPU differential max_abs={max_abs:.9} rel_rms={relative_rms:.9}"
    );
    assert!(max_abs <= 2.0e-5, "matrix-ceiling max abs {max_abs}");
    assert!(
        relative_rms <= 2.0e-4,
        "matrix-ceiling relative RMS {relative_rms}"
    );
}

#[test]
fn matrix_ceiling_lightning_scores_preserve_selector_contracts() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const ROWS: usize = 1_025;
    const QUERIES: usize = 3;
    const TOP_K: usize = 512;

    let mut query_values = vec![0.0f32; QUERIES * HEADS * DIM];
    query_values[0] = 1.0;
    query_values[2 * HEADS * DIM] = 1.0;
    let mut weight_values = vec![0.0f32; QUERIES * HEADS];
    weight_values[0] = 1.0;
    weight_values[2 * HEADS] = f32::NAN;
    let mut key_values = vec![0.0f32; ROWS * DIM];
    for row in 0..ROWS {
        key_values[row * DIM] = half::f16::from_f32((row as f32 - 512.0) * 0.001).to_f32();
    }

    let queries = offset_f16(
        &ctx,
        &query_values,
        vec![DIM as u64, HEADS as u64, QUERIES as u64],
    );
    let head_weights = offset_f32(&ctx, &weight_values, vec![HEADS as u64, QUERIES as u64]);
    let keys = offset_f16(&ctx, &key_values, vec![DIM as u64, ROWS as u64]);
    let visible_counts = offset_i32(&ctx, &[ROWS as i32; QUERIES], vec![QUERIES as u64]);
    let scores = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();
    let selected_mask = MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, QUERIES as u64]).unwrap();
    let selected_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, QUERIES as u64]).unwrap();
    let selected_counts = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();
    let status = MetalTensor::zeros_i32(&ctx, vec![QUERIES as u64]).unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_lightning_indexer_scores_f16_matrix(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &scores,
        HEADS,
        DIM,
        ROWS,
        ROWS,
        QUERIES,
    )
    .unwrap();
    encode_select_top_k_f32(
        &ctx,
        &encoder,
        &scores,
        &visible_counts,
        &selected_mask,
        None,
        &selected_ids,
        &selected_counts,
        &status,
        ROWS,
        ROWS,
        TOP_K,
        QUERIES,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let scores = read_f32(&scores);
    assert_eq!(scores[0], 0.0);
    assert_eq!(scores[512], 0.0);
    assert!(scores[513] > 0.0);
    assert!(scores[ROWS..2 * ROWS].iter().all(|score| *score == 0.0));
    assert!(scores[2 * ROWS..].iter().all(|score| !score.is_finite()));
    assert_eq!(read_i32(&status), vec![0, 0, 2]);
    assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32; QUERIES]);
    let ids = read_i32(&selected_ids);
    assert_eq!(
        &ids[..TOP_K],
        &(513..=1_024).collect::<Vec<i32>>(),
        "large-margin scores must select the 512 positive rows"
    );
    let stable_tie = (0..TOP_K as i32).collect::<Vec<_>>();
    assert_eq!(&ids[TOP_K..2 * TOP_K], &stable_tie);
    assert_eq!(&ids[2 * TOP_K..], &stable_tie);
}

#[test]
fn matrix_ceiling_lightning_scores_cover_batched_query_tail_geometry() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const ROWS: usize = 9;
    const QUERIES: usize = 128;

    let query_values = (0..QUERIES * HEADS * DIM)
        .map(|index| {
            let tag = (index * 19 + index / DIM * 7 + 11) % 127;
            half::f16::from_f32((tag as f32 - 63.0) * 0.0011).to_f32()
        })
        .collect::<Vec<_>>();
    let weight_values = (0..QUERIES * HEADS)
        .map(|index| 0.001 + (index * 11 % 23) as f32 * 0.0003)
        .collect::<Vec<_>>();
    let key_values = (0..ROWS * DIM)
        .map(|index| {
            let tag = (index * 31 + index / DIM * 5 + 13) % 131;
            half::f16::from_f32((tag as f32 - 65.0) * 0.0013).to_f32()
        })
        .collect::<Vec<_>>();
    let visible = (0..QUERIES)
        .map(|query| [1, 7, 8, 9][query % 4])
        .collect::<Vec<i32>>();
    let queries = offset_f16(
        &ctx,
        &query_values,
        vec![DIM as u64, HEADS as u64, QUERIES as u64],
    );
    let head_weights = offset_f32(&ctx, &weight_values, vec![HEADS as u64, QUERIES as u64]);
    let keys = offset_f16(&ctx, &key_values, vec![DIM as u64, ROWS as u64]);
    let visible_counts = offset_i32(&ctx, &visible, vec![QUERIES as u64]);
    let output_shape = vec![ROWS as u64, QUERIES as u64];
    let first = offset_f32(&ctx, &vec![0.0; ROWS * QUERIES], output_shape.clone());
    let second = offset_f32(&ctx, &vec![0.0; ROWS * QUERIES], output_shape);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for output in [&first, &second] {
        encode_lightning_indexer_scores_f16_matrix(
            &ctx,
            &encoder,
            &queries,
            &head_weights,
            &keys,
            &visible_counts,
            output,
            HEADS,
            DIM,
            ROWS,
            ROWS,
            QUERIES,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let actual = read_f32(&first);
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        read_f32(&second)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let mut max_abs = 0.0f32;
    for query in 0..QUERIES {
        let visible_rows = usize::try_from(visible[query]).unwrap();
        for row in 0..ROWS {
            let value = actual[query * ROWS + row];
            if row >= visible_rows {
                assert_eq!(value, f32::NEG_INFINITY);
                continue;
            }
            let mut expected = 0.0f32;
            for head in 0..HEADS {
                let mut dot = 0.0f32;
                for dimension in 0..DIM {
                    dot += query_values[(query * HEADS + head) * DIM + dimension]
                        * key_values[row * DIM + dimension];
                }
                expected += dot.max(0.0) * weight_values[query * HEADS + head];
            }
            max_abs = max_abs.max((value - expected).abs());
        }
    }
    assert!(max_abs <= 2.0e-5, "batched-query max abs {max_abs}");
}

#[test]
fn matrix_lightning_scorer_bounds_dispatch_to_visible_prefix() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const CAPACITY: usize = 17;
    const DISPATCHED: usize = 9;
    const PADDED_DISPATCH: usize = DISPATCHED.next_multiple_of(8);
    const SENTINEL: f32 = 123.0;

    let queries = offset_f16(
        &ctx,
        &vec![0.0; HEADS * DIM],
        vec![DIM as u64, HEADS as u64, 1],
    );
    let head_weights = offset_f32(&ctx, &vec![0.0; HEADS], vec![HEADS as u64, 1]);
    let keys = offset_f16(
        &ctx,
        &vec![0.0; CAPACITY * DIM],
        vec![DIM as u64, CAPACITY as u64],
    );
    let visible_counts = offset_i32(&ctx, &[DISPATCHED as i32], vec![1]);
    let scores = offset_f32(&ctx, &[SENTINEL; CAPACITY], vec![CAPACITY as u64, 1]);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_lightning_indexer_scores_f16_matrix(
        &ctx,
        &encoder,
        &queries,
        &head_weights,
        &keys,
        &visible_counts,
        &scores,
        HEADS,
        DIM,
        CAPACITY,
        DISPATCHED,
        1,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let scores = read_f32(&scores);
    assert!(scores[..DISPATCHED].iter().all(|&score| score == 0.0));
    assert!(
        scores[DISPATCHED..PADDED_DISPATCH]
            .iter()
            .all(|&score| score == f32::NEG_INFINITY)
    );
    assert!(
        scores[PADDED_DISPATCH..]
            .iter()
            .all(|&score| score == SENTINEL)
    );
}

#[test]
#[ignore = "focused production-shape matrix-ceiling profiler; run explicitly with --nocapture"]
fn profile_lightning_matrix_ceiling_at_far_context() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const MAX_ROWS: usize = 262_144;

    fn timed_gpu<F>(ctx: &MetalContext, repeats: usize, encode: F) -> f64
    where
        F: Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx.queue.commandBuffer().expect("profile command buffer");
        let encoder = KernelEncoder::begin(&command);
        for _ in 0..repeats {
            encode(&encoder).expect("encode profiled phase");
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "profile command failed: {:?}",
            command.error()
        );
        let elapsed_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed_ms.is_finite() && elapsed_ms > 0.0);
        elapsed_ms / repeats as f64
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    fn stable_top_k(scores: &[f32], top_k: usize) -> (Vec<usize>, f32) {
        let mut ranked = (0..scores.len()).collect::<Vec<_>>();
        ranked.sort_unstable_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        let margin = scores[ranked[top_k - 1]] - scores[ranked[top_k]];
        let mut selected = ranked[..top_k].to_vec();
        selected.sort_unstable();
        (selected, margin)
    }

    fn centered_splitmix(index: usize, seed: u64) -> f32 {
        let mut value = (index as u64).wrapping_add(seed);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let unit = (value >> 40) as f32 / (1u32 << 24) as f32;
        (unit - 0.5) * 0.08
    }

    let query_values = (0..HEADS * DIM)
        .map(|index| centered_splitmix(index, 0x1234_5678_9abc_def0))
        .collect::<Vec<_>>();
    let queries = offset_f32(&ctx, &query_values, vec![DIM as u64, HEADS as u64, 1]);
    let queries_f16 = MetalTensor::zeros_f16(&ctx, vec![DIM as u64, HEADS as u64, 1])
        .expect("allocate matrix-ceiling F16 queries");
    let scale = 1.0 / ((HEADS * DIM) as f32).sqrt();
    let head_weight_values = (0..HEADS)
        .map(|head| {
            let signed = (head * 17 % 29) as f32 - 14.0;
            signed * scale / 14.0
        })
        .collect::<Vec<_>>();
    let negative_head_weights = head_weight_values
        .iter()
        .filter(|&&weight| weight < 0.0)
        .count();
    let head_weights = offset_f32(&ctx, &head_weight_values, vec![HEADS as u64, 1]);
    let keys = {
        let bits = (0..MAX_ROWS * DIM)
            .map(|index| {
                half::f16::from_f32(centered_splitmix(index, 0xfedc_ba98_7654_3210)).to_bits()
            })
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bits),
            vec![DIM as u64, MAX_ROWS as u64],
            GgmlType::F16,
        )
        .expect("allocate matrix-ceiling profile keys")
    };
    let current_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate current profile scores");
    let matrix_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate matrix profile scores");

    let convert_query = |encoder: &KernelEncoder| {
        encode_scatter_offset_f32_to_f16(&ctx, encoder, &queries, &queries_f16, 0, HEADS * DIM)
            .map_err(DeepSeekV4MetalError::from)
    };
    timed_gpu(&ctx, 1, convert_query);
    let query_conversion_samples = (0..20)
        .map(|_| timed_gpu(&ctx, 1, convert_query))
        .collect::<Vec<_>>();
    let (query_conversion_ms, query_conversion_p95_ms) =
        median_and_p95(query_conversion_samples.clone());

    for row_count in [16_384usize, 65_536, MAX_ROWS] {
        let visible_counts = offset_i32(&ctx, &[row_count as i32], vec![1]);
        let key_rows = keys.view_subrange(0, vec![DIM as u64, row_count as u64]);
        let current_score_rows = current_scores.view_subrange(0, vec![row_count as u64, 1]);
        let matrix_score_rows = matrix_scores.view_subrange(0, vec![row_count as u64, 1]);
        let current = |encoder: &KernelEncoder| {
            encode_lightning_indexer_scores_f16(
                &ctx,
                encoder,
                &queries,
                &head_weights,
                &key_rows,
                &visible_counts,
                &current_score_rows,
                HEADS,
                DIM,
                row_count,
                1,
            )
        };
        let matrix = |encoder: &KernelEncoder| {
            encode_lightning_indexer_scores_f16_matrix(
                &ctx,
                encoder,
                &queries_f16,
                &head_weights,
                &key_rows,
                &visible_counts,
                &matrix_score_rows,
                HEADS,
                DIM,
                row_count,
                row_count,
                1,
            )
        };

        for _ in 0..5 {
            timed_gpu(&ctx, 1, current);
            timed_gpu(&ctx, 1, matrix);
        }
        let current_before_samples = (0..20)
            .map(|_| timed_gpu(&ctx, 1, current))
            .collect::<Vec<_>>();
        let matrix_samples = (0..20)
            .map(|_| timed_gpu(&ctx, 1, matrix))
            .collect::<Vec<_>>();
        let current_after_samples = (0..20)
            .map(|_| timed_gpu(&ctx, 1, current))
            .collect::<Vec<_>>();
        let (current_before_ms, current_before_p95_ms) =
            median_and_p95(current_before_samples.clone());
        let (matrix_ms, matrix_p95_ms) = median_and_p95(matrix_samples.clone());
        let (current_after_ms, current_after_p95_ms) =
            median_and_p95(current_after_samples.clone());
        let current_midpoint_ms = (current_before_ms + current_after_ms) * 0.5;
        let saving_ms = current_midpoint_ms - matrix_ms;

        timed_gpu(&ctx, 1, current);
        let current_values = read_f32(&current_score_rows);
        timed_gpu(&ctx, 1, matrix);
        let matrix_values = read_f32(&matrix_score_rows);
        let squared_error = current_values
            .iter()
            .zip(&matrix_values)
            .map(|(current, matrix)| f64::from(current - matrix).powi(2))
            .sum::<f64>();
        let reference_norm = current_values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let relative_rms = (squared_error / reference_norm).sqrt();
        let max_abs = current_values
            .iter()
            .zip(&matrix_values)
            .map(|(current, matrix)| (current - matrix).abs())
            .fold(0.0f32, f32::max);
        let (current_ids, current_cutoff_margin) = stable_top_k(&current_values, 512);
        let (matrix_ids, matrix_cutoff_margin) = stable_top_k(&matrix_values, 512);
        let selected_exchanges = current_ids
            .iter()
            .filter(|current| matrix_ids.binary_search(current).is_err())
            .count();

        eprintln!(
            "deepseek_v4 matrix_ceiling rows={row_count} token_equivalent={} current_before_ms={current_before_ms:.3} current_before_p95_ms={current_before_p95_ms:.3} matrix_ms={matrix_ms:.3} matrix_p95_ms={matrix_p95_ms:.3} current_after_ms={current_after_ms:.3} current_after_p95_ms={current_after_p95_ms:.3} saving_ms={saving_ms:.3} query_conversion_ms={query_conversion_ms:.4} query_conversion_p95_ms={query_conversion_p95_ms:.4} negative_head_weights={negative_head_weights} max_abs={max_abs:.9} rel_rms={relative_rms:.9} selected_exchanges={selected_exchanges} current_cutoff_margin={current_cutoff_margin:.9} matrix_cutoff_margin={matrix_cutoff_margin:.9}",
            row_count * 4,
        );
        eprintln!(
            "deepseek_v4 matrix_ceiling current_before_samples_ms={current_before_samples:?}"
        );
        eprintln!("deepseek_v4 matrix_ceiling candidate_samples_ms={matrix_samples:?}");
        eprintln!("deepseek_v4 matrix_ceiling current_after_samples_ms={current_after_samples:?}");

        match row_count {
            16_384 => assert!(
                matrix_ms - current_midpoint_ms <= 0.05,
                "16K matrix ceiling regressed by {:.3} ms",
                matrix_ms - current_midpoint_ms
            ),
            65_536 => assert!(
                saving_ms >= 0.15,
                "65K matrix ceiling saved only {saving_ms:.3} ms"
            ),
            MAX_ROWS => {
                assert!(
                    saving_ms >= 0.80,
                    "terminal matrix ceiling saved only {saving_ms:.3} ms"
                );
                assert!(
                    matrix_ms <= 1.30,
                    "terminal matrix ceiling median {matrix_ms:.3} ms"
                );
                assert!(
                    matrix_p95_ms < current_before_ms && matrix_p95_ms < current_after_ms,
                    "terminal matrix p95 {matrix_p95_ms:.3} is not below both current medians {current_before_ms:.3}/{current_after_ms:.3}"
                );
            }
            _ => unreachable!(),
        }
    }
    eprintln!(
        "deepseek_v4 matrix_ceiling query_conversion_samples_ms={query_conversion_samples:?}"
    );
}

#[test]
#[ignore = "focused production-shape FP4 shadow profiler; run explicitly with --nocapture"]
fn profile_lightning_fp4_matrix_shadow_at_far_context() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const DIM: usize = 128;
    const MAX_ROWS: usize = 262_144;

    fn timed_gpu<F>(ctx: &MetalContext, encode: F) -> f64
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx.queue.commandBuffer().expect("profile command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).expect("encode FP4 profiled phase");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "FP4 profile command failed: {:?}",
            command.error()
        );
        let elapsed_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed_ms.is_finite() && elapsed_ms > 0.0);
        elapsed_ms
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    fn e2m1_unit(code: u8) -> f32 {
        const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let magnitude = MAGNITUDES[usize::from(code & 0x07)];
        if code & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    let query_values = (0..HEADS * DIM)
        .map(|index| ((index * 17 + 3) % 113) as f32 * 0.0007 - 0.037)
        .collect::<Vec<_>>();
    let queries = offset_f32(&ctx, &query_values, vec![DIM as u64, HEADS as u64, 1]);
    let query_fp4_values = MetalTensor::zeros_dtype(
        &ctx,
        vec![INDEXER_FP4_VALUE_BYTES as u64, HEADS as u64, 1],
        GgmlType::I8,
    )
    .expect("allocate FP4 profile query values");
    let query_fp4_scales = MetalTensor::zeros_dtype(
        &ctx,
        vec![INDEXER_FP4_SCALE_BYTES as u64, HEADS as u64, 1],
        GgmlType::I8,
    )
    .expect("allocate FP4 profile query scales");
    let query_pack_status = MetalTensor::zeros_i32(&ctx, vec![HEADS as u64])
        .expect("allocate FP4 profile query status");
    let query_fp4_units = MetalTensor::zeros_f16(
        &ctx,
        vec![INDEXER_FP4_VALUES_PER_ROW as u64, HEADS as u64, 1],
    )
    .expect("allocate FP4 profile query units");
    let scale = 1.0 / ((HEADS * DIM) as f32).sqrt();
    let head_weights = offset_f32(&ctx, &vec![scale; HEADS], vec![HEADS as u64, 1]);

    let mut key_fp4_value_bytes = Vec::with_capacity(MAX_ROWS * INDEXER_FP4_VALUE_BYTES);
    let mut key_fp4_scale_bytes = Vec::with_capacity(MAX_ROWS * INDEXER_FP4_SCALE_BYTES);
    let mut key_f16_bits = Vec::with_capacity(MAX_ROWS * DIM);
    for row in 0..MAX_ROWS {
        for block in 0..4usize {
            let scale_code = 124 + ((row + block * 3 + row / 251) % 7) as u8;
            key_fp4_scale_bytes.push(scale_code);
            let block_scale = f32::from_bits(u32::from(scale_code) << 23);
            for pair in 0..16usize {
                let low = ((row * 13 + block * 7 + pair * 3 + row / 503) % 16) as u8;
                let high = ((row * 17 + block * 5 + pair * 11 + 1) % 16) as u8;
                key_fp4_value_bytes.push(low | (high << 4));
                key_f16_bits.push(half::f16::from_f32(e2m1_unit(low) * block_scale).to_bits());
                key_f16_bits.push(half::f16::from_f32(e2m1_unit(high) * block_scale).to_bits());
            }
        }
    }
    let current_keys = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&key_f16_bits),
        vec![DIM as u64, MAX_ROWS as u64],
        GgmlType::F16,
    )
    .expect("allocate FP4 profile decoded keys");
    let key_fp4_values = offset_i8(
        &ctx,
        &key_fp4_value_bytes,
        vec![INDEXER_FP4_VALUE_BYTES as u64, MAX_ROWS as u64],
    );
    let key_fp4_scales = offset_i8(
        &ctx,
        &key_fp4_scale_bytes,
        vec![INDEXER_FP4_SCALE_BYTES as u64, MAX_ROWS as u64],
    );
    let current_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate current FP4 profile scores");
    let fp4_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate FP4 shadow profile scores");

    let incremental_input = queries.view_subrange(0, vec![DIM as u64, 1]);
    let incremental_values = i8_prefix(&query_fp4_values, vec![INDEXER_FP4_VALUE_BYTES as u64, 1]);
    let incremental_scales = i8_prefix(&query_fp4_scales, vec![INDEXER_FP4_SCALE_BYTES as u64, 1]);
    let incremental_status = query_pack_status.view_subrange(0, vec![1]);
    let incremental_pack = |encoder: &KernelEncoder| {
        encode_pack_indexer_fp4_rows_shadow(
            &ctx,
            encoder,
            &incremental_input,
            &incremental_values,
            &incremental_scales,
            &incremental_status,
            1,
        )
    };
    for _ in 0..5 {
        timed_gpu(&ctx, incremental_pack);
    }
    let incremental_samples = (0..20)
        .map(|_| timed_gpu(&ctx, incremental_pack))
        .collect::<Vec<_>>();
    let (incremental_ms, incremental_p95_ms) = median_and_p95(incremental_samples.clone());
    assert!(
        incremental_ms <= 0.020,
        "incremental FP4 K pack median {incremental_ms:.6} ms"
    );
    assert!(
        incremental_p95_ms <= 0.030,
        "incremental FP4 K pack p95 {incremental_p95_ms:.6} ms"
    );

    for row_count in [16_384usize, 65_536, MAX_ROWS] {
        let visible_counts = offset_i32(&ctx, &[row_count as i32], vec![1]);
        let current_key_rows = current_keys.view_subrange(0, vec![DIM as u64, row_count as u64]);
        let fp4_key_value_rows = i8_prefix(
            &key_fp4_values,
            vec![INDEXER_FP4_VALUE_BYTES as u64, row_count as u64],
        );
        let fp4_key_scale_rows = i8_prefix(
            &key_fp4_scales,
            vec![INDEXER_FP4_SCALE_BYTES as u64, row_count as u64],
        );
        let current_score_rows = current_scores.view_subrange(0, vec![row_count as u64, 1]);
        let fp4_score_rows = fp4_scores.view_subrange(0, vec![row_count as u64, 1]);
        let current = |encoder: &KernelEncoder| {
            encode_lightning_indexer_scores_f16(
                &ctx,
                encoder,
                &queries,
                &head_weights,
                &current_key_rows,
                &visible_counts,
                &current_score_rows,
                HEADS,
                DIM,
                row_count,
                1,
            )
        };
        let fp4 = |encoder: &KernelEncoder| {
            encode_pack_indexer_fp4_rows_shadow(
                &ctx,
                encoder,
                &queries,
                &query_fp4_values,
                &query_fp4_scales,
                &query_pack_status,
                HEADS,
            )?;
            encode_unpack_indexer_fp4_units_shadow(
                &ctx,
                encoder,
                &query_fp4_values,
                &query_pack_status,
                &query_fp4_units,
                HEADS,
            )?;
            encode_lightning_indexer_scores_fp4_matrix_shadow(
                &ctx,
                encoder,
                &query_fp4_units,
                &query_fp4_scales,
                &head_weights,
                &fp4_key_value_rows,
                &fp4_key_scale_rows,
                &visible_counts,
                &fp4_score_rows,
                row_count,
                1,
            )
        };

        for _ in 0..5 {
            timed_gpu(&ctx, current);
            timed_gpu(&ctx, fp4);
        }
        let current_before_samples = (0..20)
            .map(|_| timed_gpu(&ctx, current))
            .collect::<Vec<_>>();
        let fp4_samples = (0..20).map(|_| timed_gpu(&ctx, fp4)).collect::<Vec<_>>();
        let current_after_samples = (0..20)
            .map(|_| timed_gpu(&ctx, current))
            .collect::<Vec<_>>();
        let (current_before_ms, current_before_p95_ms) =
            median_and_p95(current_before_samples.clone());
        let (fp4_ms, fp4_p95_ms) = median_and_p95(fp4_samples.clone());
        let (current_after_ms, current_after_p95_ms) =
            median_and_p95(current_after_samples.clone());
        let current_midpoint_ms = (current_before_ms + current_after_ms) * 0.5;
        let control_drift = 2.0 * (current_before_ms - current_after_ms).abs()
            / (current_before_ms + current_after_ms);
        assert!(
            control_drift <= 0.05,
            "invalid FP4 campaign: {row_count} row control drift {:.2}% exceeds 5%",
            control_drift * 100.0
        );
        let saving_ms = current_midpoint_ms - fp4_ms;
        assert_eq!(read_i32(&query_pack_status), vec![0; HEADS]);

        eprintln!(
            "deepseek_v4 fp4_shadow rows={row_count} token_equivalent={} current_before_ms={current_before_ms:.3} current_before_p95_ms={current_before_p95_ms:.3} fp4_pack_unpack_score_ms={fp4_ms:.3} fp4_pack_unpack_score_p95_ms={fp4_p95_ms:.3} current_after_ms={current_after_ms:.3} current_after_p95_ms={current_after_p95_ms:.3} control_drift={control_drift:.6} saving_ms={saving_ms:.3}",
            row_count * 4,
        );
        eprintln!("deepseek_v4 fp4_shadow current_before_samples_ms={current_before_samples:?}");
        eprintln!("deepseek_v4 fp4_shadow candidate_samples_ms={fp4_samples:?}");
        eprintln!("deepseek_v4 fp4_shadow current_after_samples_ms={current_after_samples:?}");

        match row_count {
            16_384 => assert!(
                fp4_ms - current_midpoint_ms <= 0.05,
                "16K FP4 shadow regressed by {:.3} ms",
                fp4_ms - current_midpoint_ms
            ),
            65_536 => assert!(
                saving_ms >= 0.15,
                "65K FP4 shadow saved only {saving_ms:.3} ms"
            ),
            MAX_ROWS => {
                assert!(
                    saving_ms >= 0.80,
                    "terminal FP4 shadow saved only {saving_ms:.3} ms"
                );
                assert!(fp4_ms <= 1.30, "terminal FP4 shadow median {fp4_ms:.3} ms");
                assert!(
                    fp4_p95_ms < current_before_ms && fp4_p95_ms < current_after_ms,
                    "terminal FP4 p95 {fp4_p95_ms:.3} is not below both current medians {current_before_ms:.3}/{current_after_ms:.3}"
                );
            }
            _ => unreachable!(),
        }
    }
    eprintln!(
        "deepseek_v4 fp4_shadow incremental_pack_ms={incremental_ms:.6} incremental_pack_p95_ms={incremental_p95_ms:.6} incremental_samples_ms={incremental_samples:?}"
    );
}

#[test]
#[ignore = "focused production-shape GPU profiler; run explicitly with --nocapture"]
fn profile_sparse_csa_decode_phases_at_far_context() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const INDEX_HEADS: usize = 64;
    const INDEX_DIM: usize = 128;
    const ATTENTION_HEADS: usize = 64;
    const ATTENTION_DIM: usize = 512;
    const TOP_K: usize = 512;
    const MAX_ROWS: usize = 262_144;

    fn timed_gpu<F>(ctx: &MetalContext, repeats: usize, encode: F) -> f64
    where
        F: Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx.queue.commandBuffer().expect("profile command buffer");
        let encoder = KernelEncoder::begin(&command);
        for _ in 0..repeats {
            encode(&encoder).expect("encode profiled phase");
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "profile command failed: {:?}",
            command.error()
        );
        let elapsed_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed_ms.is_finite() && elapsed_ms > 0.0);
        elapsed_ms / repeats as f64
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    let index_query_values = (0..INDEX_HEADS * INDEX_DIM)
        .map(|index| ((index * 17 + 3) % 113) as f32 * 0.0007 - 0.037)
        .collect::<Vec<_>>();
    let index_queries = offset_f32(
        &ctx,
        &index_query_values,
        vec![INDEX_DIM as u64, INDEX_HEADS as u64, 1],
    );
    let index_scale = 1.0 / ((INDEX_HEADS * INDEX_DIM) as f32).sqrt();
    let head_weights = offset_f32(
        &ctx,
        &vec![index_scale; INDEX_HEADS],
        vec![INDEX_HEADS as u64, 1],
    );
    let index_keys = {
        let bits = (0..MAX_ROWS * INDEX_DIM)
            .map(|index| {
                let row = index / INDEX_DIM;
                let dimension = index % INDEX_DIM;
                let tag = (row * 13 + dimension * 7 + row / 251) % 257;
                half::f16::from_f32((tag as f32 - 128.0) * 0.0002).to_bits()
            })
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bits),
            vec![INDEX_DIM as u64, MAX_ROWS as u64],
            GgmlType::F16,
        )
        .expect("allocate profile index keys")
    };
    let scores =
        MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1]).expect("allocate profile scores");
    let scalar_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate scalar profile scores");
    let tied_scores = MetalTensor::zeros_f32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate tied profile scores");
    let mixed_score_values = (0..MAX_ROWS)
        .map(|row| {
            let bucket = (row * 193 + row / 7 + row / 1_003) % 8_191;
            bucket as f32 * 0.0003 - 1.1
        })
        .collect::<Vec<_>>();
    let mixed_scores = offset_f32(&ctx, &mixed_score_values, vec![MAX_ROWS as u64, 1]);
    let selected_mask = MetalTensor::zeros_i32(&ctx, vec![MAX_ROWS as u64, 1])
        .expect("allocate profile selected mask");
    let cache_order_ids =
        MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, 1]).expect("allocate profile selected IDs");
    let selected_counts =
        MetalTensor::zeros_i32(&ctx, vec![1]).expect("allocate profile selected count");
    let status = MetalTensor::zeros_i32(&ctx, vec![1]).expect("allocate profile status");

    let attention_query_values = (0..ATTENTION_HEADS * ATTENTION_DIM)
        .map(|index| ((index * 19 + 5) % 127) as f32 * 0.0004 - 0.025)
        .collect::<Vec<_>>();
    let attention_queries = offset_f32(
        &ctx,
        &attention_query_values,
        vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64],
    );
    let cooperative_queries =
        attention_queries.view_subrange(0, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1]);
    let raw_cache = MetalTensor::zeros_f16(
        &ctx,
        vec![ATTENTION_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
    )
    .expect("allocate profile raw cache");
    let compressed_cache =
        MetalTensor::zeros_f16(&ctx, vec![ATTENTION_DIM as u64, MAX_ROWS as u64])
            .expect("allocate profile compressed cache");
    let sinks = offset_f32(
        &ctx,
        &vec![-0.2f32; ATTENTION_HEADS],
        vec![ATTENTION_HEADS as u64],
    );
    let attention_output =
        MetalTensor::zeros_f32(&ctx, vec![ATTENTION_DIM as u64, ATTENTION_HEADS as u64])
            .expect("allocate profile attention output");
    let cooperative_output =
        MetalTensor::zeros_f32(&ctx, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1])
            .expect("allocate profile cooperative attention output");
    let online_output =
        MetalTensor::zeros_f32(&ctx, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1])
            .expect("allocate profile online attention output");
    let direct_output =
        MetalTensor::zeros_f32(&ctx, vec![(ATTENTION_HEADS * ATTENTION_DIM) as u64, 1])
            .expect("allocate profile direct attention output");
    let attention_config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: ATTENTION_HEADS,
        head_dim: ATTENTION_DIM,
        rotary_dim: 64,
        group_count: 8,
        output_rank: 1,
    };

    for row_count in [16_384usize, 65_536, MAX_ROWS] {
        let visible_counts = offset_i32(&ctx, &[row_count as i32], vec![1]);
        let keys = index_keys.view_subrange(0, vec![INDEX_DIM as u64, row_count as u64]);
        let score_rows = scores.view_subrange(0, vec![row_count as u64, 1]);
        let scalar_score_rows = scalar_scores.view_subrange(0, vec![row_count as u64, 1]);
        let tied_score_rows = tied_scores.view_subrange(0, vec![row_count as u64, 1]);
        let mixed_score_rows = mixed_scores.view_subrange(0, vec![row_count as u64, 1]);
        let mask_rows = selected_mask.view_subrange(0, vec![row_count as u64, 1]);
        let compressed_rows =
            compressed_cache.view_subrange(0, vec![ATTENTION_DIM as u64, row_count as u64]);
        let selected_ids = cache_order_ids.view_subrange(0, vec![TOP_K as u64]);
        let position = u32::try_from(row_count * 4 - 1).unwrap();

        let warm = ctx.queue.commandBuffer().expect("profile warm command");
        let encoder = KernelEncoder::begin(&warm);
        encode_lightning_indexer_scores_f16_with_policy(
            &ctx,
            &encoder,
            &index_queries,
            &head_weights,
            &keys,
            &visible_counts,
            &scalar_score_rows,
            INDEX_HEADS,
            INDEX_DIM,
            row_count,
            1,
            true,
        )
        .unwrap();
        encode_lightning_indexer_scores_f16(
            &ctx,
            &encoder,
            &index_queries,
            &head_weights,
            &keys,
            &visible_counts,
            &score_rows,
            INDEX_HEADS,
            INDEX_DIM,
            row_count,
            1,
        )
        .unwrap();
        encode_select_top_k_f32(
            &ctx,
            &encoder,
            &tied_score_rows,
            &visible_counts,
            &mask_rows,
            None,
            &cache_order_ids,
            &selected_counts,
            &status,
            row_count,
            row_count,
            TOP_K,
            1,
        )
        .unwrap();
        encode_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &attention_queries,
            &raw_cache,
            &compressed_rows,
            &selected_ids,
            &sinks,
            &attention_output,
            position,
            row_count,
            TOP_K,
            row_count,
            attention_config,
        )
        .unwrap();
        encode_cooperative_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &cooperative_queries,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            &compressed_rows,
            row_count,
            &cache_order_ids,
            &selected_counts,
            &visible_counts,
            &sinks,
            &cooperative_output,
            position,
            0,
            1,
            1,
            TOP_K,
            false,
            false,
            attention_config,
        )
        .unwrap();
        for (candidate, direct_load) in [(&online_output, false), (&direct_output, true)] {
            encode_cooperative_selected_sink_attention_f16(
                &ctx,
                &encoder,
                &cooperative_queries,
                &raw_cache,
                &raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                &compressed_rows,
                row_count,
                &cache_order_ids,
                &selected_counts,
                &visible_counts,
                &sinks,
                candidate,
                position,
                0,
                1,
                1,
                TOP_K,
                true,
                direct_load,
                attention_config,
            )
            .unwrap();
        }
        encoder.end();
        warm.commit();
        warm.waitUntilCompleted();
        assert!(warm.error().is_none());
        assert_eq!(
            read_f32(&score_rows)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&scalar_score_rows)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "cooperative/scalar index scores differ at {row_count} rows"
        );

        let scalar_score_before_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 1, |encoder| {
                        encode_lightning_indexer_scores_f16_with_policy(
                            &ctx,
                            encoder,
                            &index_queries,
                            &head_weights,
                            &keys,
                            &visible_counts,
                            &scalar_score_rows,
                            INDEX_HEADS,
                            INDEX_DIM,
                            row_count,
                            1,
                            true,
                        )
                    })
                })
                .collect(),
        )
        .0;
        let (score_ms, score_p95_ms) = median_and_p95(
            (0..20)
                .map(|_| {
                    timed_gpu(&ctx, 1, |encoder| {
                        encode_lightning_indexer_scores_f16(
                            &ctx,
                            encoder,
                            &index_queries,
                            &head_weights,
                            &keys,
                            &visible_counts,
                            &score_rows,
                            INDEX_HEADS,
                            INDEX_DIM,
                            row_count,
                            1,
                        )
                    })
                })
                .collect(),
        );
        let scalar_score_after_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 1, |encoder| {
                        encode_lightning_indexer_scores_f16_with_policy(
                            &ctx,
                            encoder,
                            &index_queries,
                            &head_weights,
                            &keys,
                            &visible_counts,
                            &scalar_score_rows,
                            INDEX_HEADS,
                            INDEX_DIM,
                            row_count,
                            1,
                            true,
                        )
                    })
                })
                .collect(),
        )
        .0;
        let time_selector = |selector_scores: &MetalTensor, radix4: bool| {
            timed_gpu(&ctx, 1, |encoder| {
                encode_select_top_k_f32_with_policy(
                    &ctx,
                    encoder,
                    selector_scores,
                    &visible_counts,
                    &mask_rows,
                    None,
                    &cache_order_ids,
                    &selected_counts,
                    &status,
                    row_count,
                    row_count,
                    TOP_K,
                    1,
                    DeepSeekV4SelectorDispatchPolicy::Production,
                    radix4,
                )
            })
        };
        for _ in 0..5 {
            time_selector(&tied_score_rows, false);
            time_selector(&tied_score_rows, true);
        }
        let tied_select_before_ms = median_and_p95(
            (0..5)
                .map(|_| time_selector(&tied_score_rows, false))
                .collect(),
        )
        .0;
        let (tied_select_ms, tied_select_p95_ms) = median_and_p95(
            (0..20)
                .map(|_| time_selector(&tied_score_rows, true))
                .collect(),
        );
        let tied_select_after_ms = median_and_p95(
            (0..5)
                .map(|_| time_selector(&tied_score_rows, false))
                .collect(),
        )
        .0;
        assert_eq!(read_i32(&status), vec![0]);
        assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32]);
        assert_eq!(
            read_i32(&cache_order_ids),
            (0..TOP_K as i32).collect::<Vec<_>>()
        );
        for _ in 0..5 {
            time_selector(&mixed_score_rows, false);
            time_selector(&mixed_score_rows, true);
        }
        let mixed_select_before_ms = median_and_p95(
            (0..5)
                .map(|_| time_selector(&mixed_score_rows, false))
                .collect(),
        )
        .0;
        let (mixed_select_ms, mixed_select_p95_ms) = median_and_p95(
            (0..20)
                .map(|_| time_selector(&mixed_score_rows, true))
                .collect(),
        );
        let mixed_select_after_ms = median_and_p95(
            (0..5)
                .map(|_| time_selector(&mixed_score_rows, false))
                .collect(),
        )
        .0;
        let bitwise_selection = (
            read_i32(&mask_rows),
            read_i32(&cache_order_ids),
            read_i32(&selected_counts),
            read_i32(&status),
        );
        time_selector(&mixed_score_rows, true);
        let radix4_selection = (
            read_i32(&mask_rows),
            read_i32(&cache_order_ids),
            read_i32(&selected_counts),
            read_i32(&status),
        );
        assert_eq!(
            radix4_selection, bitwise_selection,
            "four-bit/bitwise selection differs at {row_count} rows"
        );
        let mut expected_mixed =
            deployed_selector_top_k_indices(&mixed_score_values[..row_count], TOP_K);
        expected_mixed.sort_unstable();
        assert_eq!(
            read_i32(&cache_order_ids),
            expected_mixed
                .iter()
                .map(|&row| row as i32)
                .collect::<Vec<_>>()
        );
        let legacy_attention_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 4, |encoder| {
                        encode_selected_sink_attention_f16(
                            &ctx,
                            encoder,
                            &attention_queries,
                            &raw_cache,
                            &compressed_rows,
                            &selected_ids,
                            &sinks,
                            &attention_output,
                            position,
                            row_count,
                            TOP_K,
                            row_count,
                            attention_config,
                        )
                    })
                })
                .collect(),
        )
        .0;
        let cooperative_attention_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 4, |encoder| {
                        encode_cooperative_selected_sink_attention_f16(
                            &ctx,
                            encoder,
                            &cooperative_queries,
                            &raw_cache,
                            &raw_cache,
                            DeepSeekV4RawCacheLayout::Ring,
                            &compressed_rows,
                            row_count,
                            &cache_order_ids,
                            &selected_counts,
                            &visible_counts,
                            &sinks,
                            &cooperative_output,
                            position,
                            0,
                            1,
                            1,
                            TOP_K,
                            false,
                            false,
                            attention_config,
                        )
                    })
                })
                .collect(),
        )
        .0;
        let online_attention_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 4, |encoder| {
                        encode_cooperative_selected_sink_attention_f16(
                            &ctx,
                            encoder,
                            &cooperative_queries,
                            &raw_cache,
                            &raw_cache,
                            DeepSeekV4RawCacheLayout::Ring,
                            &compressed_rows,
                            row_count,
                            &cache_order_ids,
                            &selected_counts,
                            &visible_counts,
                            &sinks,
                            &online_output,
                            position,
                            0,
                            1,
                            1,
                            TOP_K,
                            true,
                            false,
                            attention_config,
                        )
                    })
                })
                .collect(),
        )
        .0;
        let direct_attention_ms = median_and_p95(
            (0..3)
                .map(|_| {
                    timed_gpu(&ctx, 4, |encoder| {
                        encode_cooperative_selected_sink_attention_f16(
                            &ctx,
                            encoder,
                            &cooperative_queries,
                            &raw_cache,
                            &raw_cache,
                            DeepSeekV4RawCacheLayout::Ring,
                            &compressed_rows,
                            row_count,
                            &cache_order_ids,
                            &selected_counts,
                            &visible_counts,
                            &sinks,
                            &direct_output,
                            position,
                            0,
                            1,
                            1,
                            TOP_K,
                            true,
                            true,
                            attention_config,
                        )
                    })
                })
                .collect(),
        )
        .0;
        assert_eq!(read_i32(&status), vec![0]);
        assert_eq!(read_i32(&selected_counts), vec![TOP_K as i32]);
        assert!(
            read_f32(&attention_output)
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            read_f32(&cooperative_output)
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(
            read_f32(&direct_output)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&online_output)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "direct selected attention changed online output at {row_count} rows"
        );
        let conservative_select_ms = tied_select_ms.max(mixed_select_ms);
        eprintln!(
            "deepseek_v4 sparse_profile rows={row_count} token_equivalent={} scalar_score_before_ms={scalar_score_before_ms:.3} score_ms={score_ms:.3} score_p95_ms={score_p95_ms:.3} scalar_score_after_ms={scalar_score_after_ms:.3} score_speedup={:.2} tied_bit_before_ms={tied_select_before_ms:.3} tied_radix4_ms={tied_select_ms:.3} tied_radix4_p95_ms={tied_select_p95_ms:.3} tied_bit_after_ms={tied_select_after_ms:.3} tied_speedup={:.2} mixed_bit_before_ms={mixed_select_before_ms:.3} mixed_radix4_ms={mixed_select_ms:.3} mixed_radix4_p95_ms={mixed_select_p95_ms:.3} mixed_bit_after_ms={mixed_select_after_ms:.3} mixed_speedup={:.2} legacy_attention_ms={legacy_attention_ms:.3} cooperative_attention_ms={cooperative_attention_ms:.3} online_attention_ms={online_attention_ms:.3} direct_attention_ms={direct_attention_ms:.3} direct_attention_saving_ms={:.3} attention_speedup={:.2} conservative_projected_21_csa_ms={:.3}",
            row_count * 4,
            ((scalar_score_before_ms + scalar_score_after_ms) * 0.5) / score_ms,
            ((tied_select_before_ms + tied_select_after_ms) * 0.5) / tied_select_ms,
            ((mixed_select_before_ms + mixed_select_after_ms) * 0.5) / mixed_select_ms,
            online_attention_ms - direct_attention_ms,
            legacy_attention_ms / cooperative_attention_ms,
            (score_ms + conservative_select_ms + cooperative_attention_ms) * 21.0,
        );
    }
}

#[test]
#[ignore = "focused production-width tiled-HCA GPU profiler; run explicitly with --nocapture"]
fn profile_tiled_hca_at_far_context() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 64;
    const HEAD_DIM: usize = 512;
    const CAPACITY: usize = 8_192;

    fn timed_gpu<F>(ctx: &MetalContext, encode: F) -> f64
    where
        F: FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    {
        let command = ctx.queue.commandBuffer().expect("HCA profile command");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).expect("encode HCA profile phase");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "HCA profile command failed: {:?}",
            command.error()
        );
        let elapsed_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(elapsed_ms.is_finite() && elapsed_ms > 0.0);
        elapsed_ms
    }

    fn median_and_p95(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        (median, p95)
    }

    fn profile_arm<F>(ctx: &MetalContext, encode: F) -> (Vec<f64>, f64, f64)
    where
        F: Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError> + Copy,
    {
        for _ in 0..5 {
            let _ = timed_gpu(ctx, encode);
        }
        let samples = (0..20).map(|_| timed_gpu(ctx, encode)).collect::<Vec<_>>();
        let (median, p95) = median_and_p95(samples.clone());
        (samples, median, p95)
    }

    let config = deepseek_v4_session_attention_config();
    assert_eq!((config.head_count, config.head_dim), (HEADS, HEAD_DIM));
    let query_values = (0..HEADS * HEAD_DIM)
        .map(|index| ((index * 17 + index / 29 + 3) % 257) as f32 * 0.0003 - 0.038)
        .collect::<Vec<_>>();
    let queries = offset_f32(&ctx, &query_values, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let f16_tensor = |values: Vec<f32>, shape: Vec<u64>, label: &str| {
        let bits = values
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&bits), shape, GgmlType::F16)
            .unwrap_or_else(|error| panic!("allocate {label}: {error}"))
    };
    let raw_cache = f16_tensor(
        (0..DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM)
            .map(|index| {
                let slot = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                ((slot * 23 + dimension * 11 + slot / 7) % 251) as f32 * 0.0004 - 0.05
            })
            .collect(),
        vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        "current raw cache",
    );
    let preserved_raw_cache = f16_tensor(
        (0..DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM)
            .map(|index| {
                let slot = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                ((slot * 31 + dimension * 7 + dimension / 13) % 241) as f32 * 0.0005 - 0.06
            })
            .collect(),
        vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        "preserved raw cache",
    );
    let compressed_cache = f16_tensor(
        (0..CAPACITY * HEAD_DIM)
            .map(|index| {
                let row = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                ((row * 37 + dimension * 13 + row / 17) % 263) as f32 * 0.0003 - 0.039
            })
            .collect(),
        vec![HEAD_DIM as u64, CAPACITY as u64],
        "compressed cache",
    );
    let sinks = offset_f32(
        &ctx,
        &(0..HEADS)
            .map(|head| -0.41 + head as f32 * 0.002)
            .collect::<Vec<_>>(),
        vec![HEADS as u64],
    );
    let output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate HCA profile output");
    let online_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate online HCA profile output");
    let direct_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate direct HCA profile output");
    let grouped_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate grouped HCA profile output");
    let split4_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate split4 HCA profile output");
    let split4_partial = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, HEADS as u64, 4])
        .expect("allocate split4 HCA partial output");
    let split4_ml = MetalTensor::zeros_f32(&ctx, vec![2, HEADS as u64, 4])
        .expect("allocate split4 HCA partial max/mass");
    let split8_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate split8 HCA profile output");
    let split8_partial = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, HEADS as u64, 8])
        .expect("allocate split8 HCA partial output");
    let split8_ml = MetalTensor::zeros_f32(&ctx, vec![2, HEADS as u64, 8])
        .expect("allocate split8 HCA partial max/mass");
    let split16_output = MetalTensor::zeros_f32(&ctx, vec![(HEADS * HEAD_DIM) as u64, 1])
        .expect("allocate split16 HCA profile output");
    let split16_partial = MetalTensor::zeros_f32(&ctx, vec![HEAD_DIM as u64, HEADS as u64, 16])
        .expect("allocate split16 HCA partial output");
    let split16_ml = MetalTensor::zeros_f32(&ctx, vec![2, HEADS as u64, 16])
        .expect("allocate split16 HCA partial max/mass");

    for count in [513usize, 2_048, CAPACITY] {
        let position = u32::try_from(count * 128 - 1).unwrap();
        let legacy = |encoder: &KernelEncoder| {
            encode_tiled_dense_sink_attention_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &output,
                position,
                0,
                1,
                128,
                config,
            )
        };
        let online = |encoder: &KernelEncoder| {
            encode_online_dense_sink_attention_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &online_output,
                position,
                0,
                1,
                128,
                false,
                config,
            )
        };
        let direct = |encoder: &KernelEncoder| {
            encode_online_dense_sink_attention_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &direct_output,
                position,
                0,
                1,
                128,
                true,
                config,
            )
        };
        let grouped = |encoder: &KernelEncoder| {
            encode_grouped_online_dense_sink_attention_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                Some(DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                }),
                &sinks,
                &grouped_output,
                AttentionKind::HeavilyCompressed,
                position,
                1,
                config,
            )
        };
        let split4 = |encoder: &KernelEncoder| {
            encode_grouped_splitk_hca_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &split4_partial,
                &split4_ml,
                &split4_output,
                position,
                4,
                config,
            )
        };
        let split8 = |encoder: &KernelEncoder| {
            encode_grouped_splitk_hca_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &split8_partial,
                &split8_ml,
                &split8_output,
                position,
                8,
                config,
            )
        };
        let split16 = |encoder: &KernelEncoder| {
            encode_grouped_splitk_hca_f16(
                &ctx,
                encoder,
                &queries,
                &raw_cache,
                &preserved_raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count,
                    capacity_rows: CAPACITY,
                },
                &sinks,
                &split16_partial,
                &split16_ml,
                &split16_output,
                position,
                16,
                config,
            )
        };
        let (before_samples, before_ms, before_p95_ms) = profile_arm(&ctx, legacy);
        let (online_samples, online_ms, online_p95_ms) = profile_arm(&ctx, online);
        let (direct_samples, direct_ms, direct_p95_ms) = profile_arm(&ctx, direct);
        let (grouped_samples, grouped_ms, grouped_p95_ms) = profile_arm(&ctx, grouped);
        let (split4_samples, split4_ms, split4_p95_ms) = profile_arm(&ctx, split4);
        let (split8_samples, split8_ms, split8_p95_ms) = profile_arm(&ctx, split8);
        let (split16_samples, split16_ms, split16_p95_ms) = profile_arm(&ctx, split16);
        let (online_after_samples, online_after_ms, online_after_p95_ms) =
            profile_arm(&ctx, online);
        let (after_samples, after_ms, after_p95_ms) = profile_arm(&ctx, legacy);
        let output_values = read_f32(&output);
        assert!(output_values.iter().all(|value| value.is_finite()));
        let online_values = read_f32(&online_output);
        assert!(online_values.iter().all(|value| value.is_finite()));
        let direct_values = read_f32(&direct_output);
        let grouped_values = read_f32(&grouped_output);
        let split4_values = read_f32(&split4_output);
        let split8_values = read_f32(&split8_output);
        let split16_values = read_f32(&split16_output);
        assert_eq!(
            direct_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            online_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row direct HCA changed online output"
        );
        assert_eq!(
            grouped_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            online_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{count}-row grouped HCA changed online output"
        );
        assert_close(
            &format!("{count}-row split4 HCA"),
            &split4_values,
            &output_values,
            8e-5,
        );
        assert_close(
            &format!("{count}-row split8 HCA"),
            &split8_values,
            &output_values,
            8e-5,
        );
        assert_close(
            &format!("{count}-row split16 HCA"),
            &split16_values,
            &output_values,
            8e-5,
        );
        let mut dot = 0.0f64;
        let mut online_norm = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut squared_error = 0.0f64;
        let mut max_scaled_error = 0.0f64;
        for (&actual, &reference) in online_values.iter().zip(&output_values) {
            dot += f64::from(actual) * f64::from(reference);
            online_norm += f64::from(actual).powi(2);
            reference_norm += f64::from(reference).powi(2);
            squared_error += f64::from(actual - reference).powi(2);
            max_scaled_error = max_scaled_error.max(f64::from(
                (actual - reference).abs() / reference.abs().max(1.0),
            ));
        }
        let cosine = dot / (online_norm.sqrt() * reference_norm.sqrt());
        let relative_rms = (squared_error / reference_norm).sqrt();
        let baseline_midpoint_ms = (before_ms + after_ms) * 0.5;
        let saving_ms = baseline_midpoint_ms - online_ms;
        let direct_saving_ms = (online_ms + online_after_ms) * 0.5 - direct_ms;
        eprintln!(
            "deepseek_v4 tiled_hca_profile rows={count} token_equivalent={} legacy_before_ms={before_ms:.3} legacy_before_p95_ms={before_p95_ms:.3} online_before_ms={online_ms:.3} online_before_p95_ms={online_p95_ms:.3} direct_ms={direct_ms:.3} direct_p95_ms={direct_p95_ms:.3} grouped_ms={grouped_ms:.3} grouped_p95_ms={grouped_p95_ms:.3} split4_ms={split4_ms:.3} split4_p95_ms={split4_p95_ms:.3} split8_ms={split8_ms:.3} split8_p95_ms={split8_p95_ms:.3} split16_ms={split16_ms:.3} split16_p95_ms={split16_p95_ms:.3} online_after_ms={online_after_ms:.3} online_after_p95_ms={online_after_p95_ms:.3} direct_saving_ms={direct_saving_ms:.3} legacy_after_ms={after_ms:.3} legacy_after_p95_ms={after_p95_ms:.3} saving_ms={saving_ms:.3} online_cosine={cosine:.9} online_rel_rms={relative_rms:.9} online_max_scaled={max_scaled_error:.9} legacy_before_samples_ms={before_samples:?} online_before_samples_ms={online_samples:?} direct_samples_ms={direct_samples:?} grouped_samples_ms={grouped_samples:?} split4_samples_ms={split4_samples:?} split8_samples_ms={split8_samples:?} split16_samples_ms={split16_samples:?} online_after_samples_ms={online_after_samples:?} legacy_after_samples_ms={after_samples:?}",
            count * 128,
        );
        assert!(max_scaled_error <= 8e-5, "{count}-row scaled error");
        assert!(relative_rms <= 1e-3, "{count}-row relative RMS");
        assert!(cosine >= 0.999_999, "{count}-row cosine");
        match count {
            513 => assert!(online_ms <= baseline_midpoint_ms + 0.05),
            2_048 => assert!(saving_ms >= 0.20),
            CAPACITY => {
                assert!(saving_ms >= 1.50);
                assert!(online_p95_ms < before_ms.min(after_ms));
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn dense_compressed_attention_includes_the_newest_published_row() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let queries = (0..HEADS * HEAD_DIM)
        .map(|index| {
            let head = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            if head == 0 {
                0.09 + (dimension % 13) as f32 * 0.003
            } else {
                -0.07 + (dimension % 11) as f32 * 0.002
            }
        })
        .collect::<Vec<_>>();
    let raw_rows = (0..16)
        .flat_map(|row| {
            (0..HEAD_DIM)
                .map(move |dimension| -0.24 + row as f32 * 0.071 + (dimension % 9) as f32 * 0.006)
        })
        .collect::<Vec<_>>();
    let compressed_rows = (0..4)
        .flat_map(|row| {
            (0..HEAD_DIM).map(move |dimension| match row {
                0 => 0.52 - (dimension % 17) as f32 * 0.004,
                1 => -0.41 + (dimension % 19) as f32 * 0.005,
                2 => 0.33 - (dimension % 23) as f32 * 0.006,
                _ => -0.28 + (dimension % 29) as f32 * 0.004,
            })
        })
        .collect::<Vec<_>>();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let raw_rows = raw_rows.into_iter().map(round_f16).collect::<Vec<_>>();
    let compressed_rows = compressed_rows
        .into_iter()
        .map(round_f16)
        .collect::<Vec<_>>();
    let sinks = [-0.43, 0.18];
    let expected = shared_kv_attention(
        &queries,
        HEADS,
        HEAD_DIM,
        &raw_rows,
        &compressed_rows,
        None,
        &sinks,
    )
    .unwrap();
    let prior_rows = shared_kv_attention(
        &queries,
        HEADS,
        HEAD_DIM,
        &raw_rows,
        &compressed_rows[..3 * HEAD_DIM],
        None,
        &sinks,
    )
    .unwrap();
    let local_only =
        shared_kv_attention(&queries, HEADS, HEAD_DIM, &raw_rows, &[], None, &sinks).unwrap();
    assert!(
        expected
            .iter()
            .zip(&local_only)
            .any(|(dense, local)| (dense - local).abs() > 1e-2),
        "fixture must distinguish dense CSA from local-only attention"
    );
    assert!(
        expected
            .iter()
            .zip(&prior_rows)
            .any(|(four_rows, prior)| (four_rows - prior).abs() > 1e-2),
        "fixture must make the newest compressed row materially visible"
    );

    let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, HEADS as u64]);
    let cooperative_query = query_tensor.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
    let raw_cache =
        MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
            .unwrap();
    let compressed_cache = MetalTensor::zeros_f16(
        &ctx,
        vec![
            HEAD_DIM as u64,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
        ],
    )
    .unwrap();
    let raw_sources = raw_rows
        .chunks_exact(HEAD_DIM)
        .map(|row| offset_f32(&ctx, row, vec![HEAD_DIM as u64]))
        .collect::<Vec<_>>();
    let compressed_source = offset_f32(&ctx, &compressed_rows, vec![HEAD_DIM as u64, 4]);
    let output = offset_f32(
        &ctx,
        &vec![0.0; HEADS * HEAD_DIM],
        vec![HEAD_DIM as u64, HEADS as u64],
    );
    let cooperative_output = output.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for (row, source) in raw_sources.iter().enumerate() {
        encode_scatter_offset_f32_to_f16(
            &ctx,
            &encoder,
            source,
            &raw_cache,
            row * HEAD_DIM,
            HEAD_DIM,
        )
        .unwrap();
    }
    encode_scatter_offset_f32_to_f16(
        &ctx,
        &encoder,
        &compressed_source,
        &compressed_cache,
        0,
        4 * HEAD_DIM,
    )
    .unwrap();
    encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &cooperative_query,
        &raw_cache,
        &raw_cache,
        DeepSeekV4RawCacheLayout::Ring,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed_cache,
            count: 4,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sink_tensor,
        &cooperative_output,
        AttentionKind::CompressedSparse,
        15,
        1,
        config,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "command failed: {:?}",
        command.error()
    );
    assert_close(
        "four-row dense compressed attention",
        &read_f32(&output),
        &expected,
        4e-5,
    );
}

#[test]
fn dense_attention_matches_wrapped_geometries_through_two_csa_slabs() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 512;
    const COMPRESSED_ROWS: usize = DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS;
    let config = DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: 1,
        q_lora_rank: 1,
        head_count: HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: 64,
        group_count: 1,
        output_rank: 1,
    };
    let sinks = [-0.31, 0.22];
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();

    for (position, expected_hca_count, expected_csa_count) in [
        (638usize, 4usize, 159usize),
        (639, 5, 160),
        (640, 5, 160),
        (1022, 7, 255),
        (1023, 8, 256),
        (1024, 8, 256),
        (1026, 8, 256),
        (1027, 8, 257),
        (1028, 8, 257),
        (2046, 15, 511),
        (2047, 16, 512),
        (2048, 16, 512),
    ] {
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
        let mut raw_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
        for logical_position in raw_start..=position {
            let row = (0..HEAD_DIM)
                .map(|dimension| {
                    let tag = (logical_position * 29 + dimension * 11 + logical_position / 5) % 127;
                    round_f16(
                        (tag as f32 - 63.0) * 0.0025
                            + if (logical_position + dimension).is_multiple_of(31) {
                                0.037
                            } else {
                                -0.009
                            },
                    )
                })
                .collect::<Vec<_>>();
            raw_rows.extend_from_slice(&row);
            let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
            raw_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(&row);
        }
        let mut compressed_rows = (0..COMPRESSED_ROWS)
            .flat_map(|row| {
                (0..HEAD_DIM).map(move |dimension| {
                    let tag = (row * 37 + dimension * 7 + row / 3) % 113;
                    round_f16(
                        (tag as f32 - 56.0) * 0.0031
                            + if (row + dimension).is_multiple_of(23) {
                                0.041
                            } else {
                                -0.013
                            },
                    )
                })
            })
            .collect::<Vec<_>>();
        let queries = (0..HEADS * HEAD_DIM)
            .map(|index| {
                let head = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                let tag = (position * 13 + head * 17 + dimension * 5) % 97;
                (tag as f32 - 48.0) * 0.0027
            })
            .collect::<Vec<_>>();
        let hca_count = (position + 1) / 128;
        let csa_count = (position + 1) / 4;
        assert_eq!(hca_count, expected_hca_count);
        assert_eq!(csa_count, expected_csa_count);
        if matches!(position, 639 | 1023 | 1027 | 2047) {
            for row in [hca_count - 1, csa_count - 1] {
                for dimension in 0..HEAD_DIM {
                    compressed_rows[row * HEAD_DIM + dimension] =
                        round_f16(queries[dimension] * 12.0);
                }
            }
        }
        let expected_hca = shared_kv_attention(
            &queries,
            HEADS,
            HEAD_DIM,
            &raw_rows,
            &compressed_rows[..hca_count * HEAD_DIM],
            None,
            &sinks,
        )
        .unwrap();
        let expected_csa = shared_kv_attention(
            &queries,
            HEADS,
            HEAD_DIM,
            &raw_rows,
            &compressed_rows[..csa_count * HEAD_DIM],
            None,
            &sinks,
        )
        .unwrap();
        assert!(
            expected_hca
                .iter()
                .zip(&expected_csa)
                .any(|(hca, csa)| (hca - csa).abs() > 1e-3),
            "position {position} fixture must distinguish HCA and CSA row counts"
        );
        if matches!(position, 639 | 1023 | 1027 | 2047) {
            let prior_hca = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &compressed_rows[..(hca_count - 1) * HEAD_DIM],
                None,
                &sinks,
            )
            .unwrap();
            assert!(
                expected_hca
                    .iter()
                    .zip(&prior_hca)
                    .any(|(current, prior)| (current - prior).abs() > 1e-3),
                "position {position} HCA fixture must expose the newest row"
            );
            let prior_csa = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &compressed_rows[..(csa_count - 1) * HEAD_DIM],
                None,
                &sinks,
            )
            .unwrap();
            assert!(
                expected_csa
                    .iter()
                    .zip(&prior_csa)
                    .any(|(current, prior)| (current - prior).abs() > 1e-3),
                "position {position} CSA fixture must expose the newest row"
            );
        }

        let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, HEADS as u64]);
        let cooperative_query = query_tensor.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
        let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
        let raw_source = offset_f32(
            &ctx,
            &raw_ring,
            vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        );
        let compressed_source = offset_f32(
            &ctx,
            &compressed_rows,
            vec![HEAD_DIM as u64, COMPRESSED_ROWS as u64],
        );
        let raw_cache =
            MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
                .unwrap();
        let compressed_cache = MetalTensor::zeros_f16(
            &ctx,
            vec![
                HEAD_DIM as u64,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
            ],
        )
        .unwrap();
        let hca_output = offset_f32(
            &ctx,
            &vec![0.0; HEADS * HEAD_DIM],
            vec![HEAD_DIM as u64, HEADS as u64],
        );
        let csa_output = offset_f32(
            &ctx,
            &vec![0.0; HEADS * HEAD_DIM],
            vec![HEAD_DIM as u64, HEADS as u64],
        );
        let cooperative_hca_output =
            hca_output.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
        let cooperative_csa_output =
            csa_output.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_scatter_offset_f32_to_f16(
            &ctx,
            &encoder,
            &raw_source,
            &raw_cache,
            0,
            raw_ring.len(),
        )
        .unwrap();
        encode_scatter_offset_f32_to_f16(
            &ctx,
            &encoder,
            &compressed_source,
            &compressed_cache,
            0,
            compressed_rows.len(),
        )
        .unwrap();
        encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &cooperative_query,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed_cache,
                count: hca_count,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sink_tensor,
            &cooperative_hca_output,
            AttentionKind::HeavilyCompressed,
            position as u32,
            1,
            config,
        )
        .unwrap();
        encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &cooperative_query,
            &raw_cache,
            &raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed_cache,
                count: csa_count,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sink_tensor,
            &cooperative_csa_output,
            AttentionKind::CompressedSparse,
            position as u32,
            1,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "position {position} command failed: {:?}",
            command.error()
        );
        assert_close(
            &format!("position {position} wrapped HCA attention"),
            &read_f32(&hca_output),
            &expected_hca,
            7e-5,
        );
        assert_close(
            &format!("position {position} wrapped CSA attention"),
            &read_f32(&csa_output),
            &expected_csa,
            7e-5,
        );
    }

    let queries = offset_f32(
        &ctx,
        &vec![0.0; HEADS * HEAD_DIM],
        vec![HEAD_DIM as u64, HEADS as u64],
    );
    let raw_cache =
        MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
            .unwrap();
    let compressed_cache = MetalTensor::zeros_f16(
        &ctx,
        vec![
            HEAD_DIM as u64,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
        ],
    )
    .unwrap();
    let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
    let output = offset_f32(
        &ctx,
        &vec![0.0; HEADS * HEAD_DIM],
        vec![HEAD_DIM as u64, HEADS as u64],
    );
    let cooperative_queries = queries.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let cooperative_output = output.view_subrange(0, vec![(HEADS * HEAD_DIM) as u64, 1]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let error = encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &cooperative_queries,
        &raw_cache,
        &raw_cache,
        DeepSeekV4RawCacheLayout::Ring,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed_cache,
            count: DEEPSEEK_V4_CSA_TOP_K + 1,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sink_tensor,
        &cooperative_output,
        AttentionKind::CompressedSparse,
        2_051,
        1,
        config,
    )
    .unwrap_err();
    encoder.end();
    assert!(error.to_string().contains("cannot consume 513 rows"));
}

fn oracle_expert(
    normalized: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden: usize,
    ffn: usize,
    clamp: f32,
) -> Vec<f32> {
    let gate = mat_vec(gate, hidden, ffn, normalized).expect("gate oracle");
    let up = mat_vec(up, hidden, ffn, normalized).expect("up oracle");
    let inner =
        crate::deepseek_v4_oracle::clamped_swiglu(&gate, &up, clamp).expect("SwiGLU oracle");
    mat_vec(down, ffn, hidden, &inner).expect("down oracle")
}

#[test]
fn all_slot_q3q4_fast_scope_is_exactly_k160_m4() {
    let qualified = DeepSeekV4MoeConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        ffn_size: DEEPSEEK_V4_ALL_SLOTS_Q3Q4_FFN_SIZE,
        expert_count: DEEPSEEK_V4_ALL_SLOTS_Q3Q4_EXPERT_COUNT,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
        routed_scale: 1.0,
    };
    assert!(deepseek_v4_all_slots_q3q4_scope_qualified(
        DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE,
        qualified,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    for config in [
        DeepSeekV4MoeConfig {
            expert_count: 256,
            ..qualified
        },
        DeepSeekV4MoeConfig {
            ffn_size: 4_096,
            ..qualified
        },
        DeepSeekV4MoeConfig {
            top_k: 5,
            ..qualified
        },
    ] {
        assert!(!deepseek_v4_all_slots_q3q4_scope_qualified(
            DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE,
            config,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ));
    }
    assert!(!deepseek_v4_all_slots_q3q4_scope_qualified(
        "Apple M3 Max",
        qualified,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!deepseek_v4_all_slots_q3q4_scope_qualified(
        DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE,
        qualified,
        GgmlType::IQ3_XXS,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
}

#[test]
fn single_token_moe_learned_and_hash_match_oracles_with_exact_bank_slices() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 3;
    const F: usize = 3;
    const E: usize = 3;
    const K: usize = 2;
    let config = DeepSeekV4MoeConfig {
        hidden_size: H,
        ffn_size: F,
        expert_count: E,
        top_k: K,
        routed_scale: 1.7,
    };
    let input_values = [1.3, -0.8, 0.45];
    let norm_values = [1.1, 0.65, 1.35];
    let router_values = [0.31, -0.27, 0.18, -0.22, 0.43, 0.09, 0.16, 0.08, -0.51];
    let gate_bank_values = (0..E * F * H)
        .map(|index| {
            let expert = index / (F * H);
            let row = (index / H) % F;
            let column = index % H;
            let lead = [2.7, -1.4, 1.1][row] * (expert as f32 + 1.0);
            if column == 0 {
                lead
            } else {
                (column as f32 - 1.5) * 0.23 + expert as f32 * 0.17
            }
        })
        .collect::<Vec<_>>();
    let up_bank_values = (0..E * F * H)
        .map(|index| {
            let expert = index / (F * H);
            let row = (index / H) % F;
            let column = index % H;
            let lead = [2.4, 2.1, -2.8][row] * (expert as f32 + 0.7);
            if column == 1 {
                lead
            } else {
                (row as f32 - column as f32) * 0.19 - expert as f32 * 0.11
            }
        })
        .collect::<Vec<_>>();
    let down_bank_values = (0..E * H * F)
        .map(|index| {
            let expert = index / (H * F);
            let row = (index / F) % H;
            let column = index % F;
            (expert as f32 + 1.0) * 0.37 + row as f32 * 0.21 - column as f32 * 0.16
        })
        .collect::<Vec<_>>();
    let shared_gate_values = [1.9, -0.2, 0.1, -1.1, 0.4, 0.2, 0.8, -0.7, 0.3];
    let shared_up_values = [-0.3, -2.2, 0.4, 0.2, 2.5, -0.1, 0.6, -1.8, 0.7];
    let shared_down_values = [0.7, -0.2, 0.4, -0.3, 0.8, 0.1, 0.2, -0.5, 0.9];
    let clamp = 0.55;
    let rms_eps = 1.0e-5;

    let input = offset_f32(&ctx, &input_values, vec![H as u64]);
    let norm = offset_f32(&ctx, &norm_values, vec![H as u64]);
    let router = offset_f32(&ctx, &router_values, vec![H as u64, E as u64]);
    let gate_bank = offset_f32(&ctx, &gate_bank_values, vec![H as u64, F as u64, E as u64]);
    let up_bank = offset_f32(&ctx, &up_bank_values, vec![H as u64, F as u64, E as u64]);
    let down_bank = offset_f32(&ctx, &down_bank_values, vec![F as u64, H as u64, E as u64]);
    let shared_gate = offset_f32(&ctx, &shared_gate_values, vec![H as u64, F as u64]);
    let shared_up = offset_f32(&ctx, &shared_up_values, vec![H as u64, F as u64]);
    let shared_down = offset_f32(&ctx, &shared_down_values, vec![F as u64, H as u64]);
    let normalized = rms_norm(&input_values, Some(&norm_values), rms_eps).unwrap();
    let logits = mat_vec(&router_values, H, E, &normalized).unwrap();
    let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits).unwrap();
    let shared_gate_projection =
        mat_vec(&shared_gate_values, H, F, &normalized).expect("shared gate projection");
    let shared_up_projection =
        mat_vec(&shared_up_values, H, F, &normalized).expect("shared up projection");
    assert!(shared_gate_projection.iter().any(|&value| value > clamp));
    assert!(shared_up_projection.iter().any(|&value| value > clamp));
    assert!(shared_up_projection.iter().any(|&value| value < -clamp));
    let shared_expected = oracle_expert(
        &normalized,
        &shared_gate_values,
        &shared_up_values,
        &shared_down_values,
        H,
        F,
        clamp,
    );

    let routes = [
        (
            "learned",
            vec![0.14, -0.31, 0.47],
            vec![0i32, 1, 2, 0, 1, 2],
            0usize,
        ),
        ("hash", vec![0.0; E], vec![0i32, 1, 2, 0, 1, 2], 1usize),
    ];
    for (label, bias_values, map_values, token_id) in routes {
        let scratch = DeepSeekV4MoeScratch::new(&ctx, config).expect("MoE scratch");
        let command = ctx.queue.commandBuffer().expect("router command");
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_router(&ctx, &encoder, &input, &norm, &router, rms_eps)
            .expect("encode router");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{label} router command failed");
        assert_close(
            label,
            &read_f32(scratch.normalized_input()),
            &normalized,
            3e-5,
        );
        assert_close(label, &read_f32(scratch.logits()), &logits, 3e-5);

        let decision = if label == "learned" {
            let bias = offset_f32(&ctx, &bias_values, vec![E as u64]);
            scratch.route_learned(&bias).expect("learned route");
            crate::deepseek_v4_oracle::learned_route(&scores, &bias_values, K, config.routed_scale)
                .unwrap()
        } else {
            let map = offset_i32(&ctx, &map_values, vec![K as u64, 3]);
            scratch.route_hash(token_id, &map).expect("hash route");
            let selected = map_values[token_id * K..token_id * K + K]
                .iter()
                .map(|&id| id as usize)
                .collect::<Vec<_>>();
            crate::deepseek_v4_oracle::hash_route(&scores, &selected, config.routed_scale).unwrap()
        };
        assert_eq!(
            read_i32(scratch.expert_ids()),
            decision
                .expert_ids
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>()
        );
        assert_close(label, &read_f32(scratch.weights()), &decision.weights, 1e-6);

        let mut expected_slots = Vec::new();
        for &expert in &decision.expert_ids {
            let gate = &gate_bank_values[expert * H * F..(expert + 1) * H * F];
            let up = &up_bank_values[expert * H * F..(expert + 1) * H * F];
            let down = &down_bank_values[expert * F * H..(expert + 1) * F * H];
            expected_slots.extend(oracle_expert(&normalized, gate, up, down, H, F, clamp));
        }
        let mut routed_expected = vec![0.0f32; H];
        for slot in 0..K {
            for dimension in 0..H {
                routed_expected[dimension] +=
                    expected_slots[slot * H + dimension] * decision.weights[slot];
            }
        }
        let final_expected = routed_expected
            .iter()
            .zip(&shared_expected)
            .map(|(routed, shared)| routed + shared)
            .collect::<Vec<_>>();

        let command = ctx.queue.commandBuffer().expect("expert command");
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_experts(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &down_bank,
                &shared_gate,
                &shared_up,
                &shared_down,
                clamp,
                clamp,
            )
            .expect("encode experts");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{label} expert command failed");
        assert_close(
            label,
            &read_f32(scratch.expert_outputs()),
            &expected_slots,
            8e-5,
        );
        assert_close(
            label,
            &read_f32(scratch.routed_output()),
            &routed_expected,
            1e-4,
        );
        assert_close(
            label,
            &read_f32(scratch.shared_output()),
            &shared_expected,
            8e-5,
        );
        assert_close(
            label,
            &read_f32(scratch.final_output()),
            &final_expected,
            1e-4,
        );
    }
}

#[test]
fn learned_moe_route_breaks_exact_score_bias_ties_by_expert_id() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let config = DeepSeekV4MoeConfig {
        hidden_size: 2,
        ffn_size: 2,
        expert_count: 4,
        top_k: 3,
        routed_scale: 1.0,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();

    let input = offset_f32(&ctx, &[0.7, -0.2], vec![2]);
    let norm = offset_f32(&ctx, &[1.0, 1.0], vec![2]);
    let zero_router = offset_f32(&ctx, &[0.0; 8], vec![2, 4]);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    scratch
        .encode_router(&ctx, &encoder, &input, &norm, &zero_router, 1e-5)
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    let tied_bias = offset_f32(&ctx, &[0.5, 0.5, 0.5, -0.2], vec![4]);
    scratch.route_learned(&tied_bias).unwrap();
    assert_eq!(read_i32(scratch.expert_ids()), vec![0, 1, 2]);
    let expected_scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&[0.0; 4]).unwrap();
    let expected =
        crate::deepseek_v4_oracle::learned_route(&expected_scores, &[0.5, 0.5, 0.5, -0.2], 3, 1.0)
            .unwrap();
    assert_close(
        "tie weights",
        &read_f32(scratch.weights()),
        &expected.weights,
        1e-6,
    );
}

#[test]
fn stage_recorder_rejects_empty_duplicate_and_out_of_range_layers() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let reject = |layers: &[usize]| match DeepSeekV4StageRecorder::new(&ctx, layers) {
        Ok(_) => panic!("stage recorder unexpectedly accepted {layers:?}"),
        Err(error) => error.to_string(),
    };
    assert!(reject(&[]).contains("at least one sampled layer"));
    assert!(reject(&[4, 4]).contains("requested twice"));
    assert!(reject(&[DEEPSEEK_V4_LAYER_COUNT]).contains("outside"));
    let recorder = DeepSeekV4StageRecorder::new(&ctx, &[0, 4, 42]).unwrap();
    assert!(recorder.samples_layer(0));
    assert!(recorder.samples_layer(4));
    assert!(recorder.samples_layer(42));
    assert!(!recorder.samples_layer(3));
    assert_eq!(recorder.samples.sample_count(), 3 * 10 * 2);
}

#[test]
fn stage_sample_resolver_rejects_overlap_inversion_and_zero_duration() {
    let records = DEEPSEEK_V4_STAGE_KINDS
        .iter()
        .enumerate()
        .map(|(index, &kind)| DeepSeekV4PendingStageSample {
            layer: 7,
            kind,
            start_sample: index * 2,
            end_sample: index * 2 + 1,
        })
        .collect::<Vec<_>>();
    let timestamps = (0..DEEPSEEK_V4_STAGE_KINDS.len())
        .flat_map(|index| [index as u64 * 20, index as u64 * 20 + 10])
        .collect::<Vec<_>>();
    let valid = resolve_deepseek_v4_layer_stage_samples(7, &records, &timestamps, 1.9).unwrap();
    assert_eq!(valid.sampled_span_ticks, 190);
    assert_eq!(
        valid
            .stages
            .iter()
            .map(|stage| stage.duration_ticks)
            .sum::<u64>(),
        100
    );
    assert!((valid.encoder_boundary_ms_scaled - 0.9).abs() <= 1e-12);

    let mut overlap = timestamps.clone();
    overlap[2] = 9;
    assert!(
        resolve_deepseek_v4_layer_stage_samples(7, &records, &overlap, 1.9)
            .unwrap_err()
            .to_string()
            .contains("overlaps")
    );

    let mut compensated_overlap = timestamps.clone();
    compensated_overlap[2] = 5;
    compensated_overlap[3] = 15;
    assert!(
        resolve_deepseek_v4_layer_stage_samples(7, &records, &compensated_overlap, 1.9)
            .unwrap_err()
            .to_string()
            .contains("overlaps")
    );

    let mut inverted = timestamps.clone();
    inverted[5] = inverted[4] - 1;
    assert!(
        resolve_deepseek_v4_layer_stage_samples(7, &records, &inverted, 1.9)
            .unwrap_err()
            .to_string()
            .contains("inverted")
    );
    assert!(
        resolve_deepseek_v4_layer_stage_samples(7, &records, &timestamps, 0.0)
            .unwrap_err()
            .to_string()
            .contains("invalid command GPU duration")
    );
}

#[test]
fn whole_token_profile_accounts_for_wait_without_double_counting_commit() {
    let profile = DeepSeekV4WholeTokenProfile {
        forward_wall_ms: 100.0,
        guards_phase_cpu_ms: 0.5,
        record_reset_cpu_ms: 0.1,
        command_encoder_create_cpu_ms: 0.2,
        encode_cpu_ms: 5.0,
        commit_cpu_ms: 0.75,
        commit_wait_wall_ms: 92.0,
        command_gpu_ms: 90.0,
        command_status_cpu_ms: 0.1,
        record_read_cpu_ms: 0.2,
        record_validate_callback_cpu_ms: 1.0,
        causal_commit_cpu_ms: 0.9,
        ..DeepSeekV4WholeTokenProfile::default()
    };
    assert_eq!(profile.wait_residual_ms(), 2.0);
    assert_eq!(profile.outside_gpu_ms(), 10.0);
    assert_eq!(profile.accounted_outside_gpu_ms(), 10.0);
    assert_eq!(profile.reconstruction_residual_ms(), 0.0);
}

#[test]
fn learned_route_pipeline_rejects_non_32_lane_geometry() {
    validate_deepseek_v4_route_pipeline_geometry("learned route", 32, 256, 256).unwrap();
    let width_error =
        validate_deepseek_v4_route_pipeline_geometry("learned route", 16, 256, 256).unwrap_err();
    assert!(width_error.to_string().contains("32-lane simdgroups"));
    let capacity_error =
        validate_deepseek_v4_route_pipeline_geometry("learned route", 32, 128, 256).unwrap_err();
    assert!(capacity_error.to_string().contains("only 128 threads"));
    validate_deepseek_v4_route_pipeline_geometry("hash route", 16, 1, 1).unwrap();
}

#[test]
fn sparse_selector_parallelizes_at_the_first_pruned_row() {
    assert!(!use_parallel_selector(
        DeepSeekV4SelectorDispatchPolicy::Production,
        DEEPSEEK_V4_CSA_TOP_K,
        DEEPSEEK_V4_CSA_TOP_K,
    ));
    assert!(use_parallel_selector(
        DeepSeekV4SelectorDispatchPolicy::Production,
        DEEPSEEK_V4_CSA_TOP_K + 1,
        DEEPSEEK_V4_CSA_TOP_K,
    ));
    assert!(!use_parallel_selector(
        DeepSeekV4SelectorDispatchPolicy::ScalarOracle,
        DEEPSEEK_V4_CSA_TOP_K + 1,
        DEEPSEEK_V4_CSA_TOP_K,
    ));
    assert!(use_parallel_selector(
        DeepSeekV4SelectorDispatchPolicy::Parallel,
        DEEPSEEK_V4_CSA_TOP_K,
        DEEPSEEK_V4_CSA_TOP_K,
    ));
}

#[test]
fn cooperative_lightning_score_geometry_fails_closed() {
    const KERNEL: &str = "cooperative Lightning scorer";
    validate_cooperative_lightning_score_geometry(KERNEL, 32, 256, 4_096).unwrap();
    assert!(
        validate_cooperative_lightning_score_geometry(KERNEL, 16, 256, 4_096)
            .unwrap_err()
            .to_string()
            .contains("SIMD width 32")
    );
    assert!(
        validate_cooperative_lightning_score_geometry(KERNEL, 32, 255, 4_096)
            .unwrap_err()
            .to_string()
            .contains("256 threads")
    );
    assert!(
        validate_cooperative_lightning_score_geometry(KERNEL, 32, 256, 4_095)
            .unwrap_err()
            .to_string()
            .contains("4096 threadgroup bytes")
    );
}

#[test]
fn lightning_score_offsets_reject_u32_shader_overflow() {
    validate_lightning_indexer_score_offsets(64, 128, 262_144, 1).unwrap();
    let over_u32 = u32::MAX as usize + 1;
    assert!(
        validate_lightning_indexer_score_offsets(1, 1, over_u32, 1)
            .unwrap_err()
            .to_string()
            .contains("score elements exceed u32")
    );
    assert!(
        validate_lightning_indexer_score_offsets(65_536, 65_536, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("query elements exceed u32")
    );
    assert!(
        validate_lightning_indexer_score_offsets(65_536, 1, 1, 65_536)
            .unwrap_err()
            .to_string()
            .contains("weight elements exceed u32")
    );
    assert!(
        validate_lightning_indexer_score_offsets(1, 65_536, 65_536, 1)
            .unwrap_err()
            .to_string()
            .contains("key elements exceed u32")
    );
}

#[test]
fn layer_records_preserve_earlier_failures_across_later_success() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const E: usize = 8;
    let config = DeepSeekV4MoeConfig {
        hidden_size: 1,
        ffn_size: 1,
        expert_count: E,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    let routes = DeepSeekV4LayerRouteRecords::new(&ctx, config).unwrap();
    routes.reset_for_token().unwrap();
    assert!(
        routes
            .read_completed()
            .unwrap()
            .validate_layer(6)
            .unwrap_err()
            .to_string()
            .contains("pending")
    );
    host_write_f32(
        &scratch.logits,
        &[-3.0, 0.5, 1.25, -0.75, 2.0, 0.125, -1.5, 0.875],
        "layer-record route logits",
    )
    .unwrap();
    let hash_map = offset_i32(&ctx, &[0, 2, 4, 1, 7, 3], vec![6, 1]);
    let failed = routes.layer(7).unwrap();
    let ready = routes.layer(8).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let failed_encoder = KernelEncoder::begin(&command);
    scratch
        .encode_route_hash_gpu_into(&ctx, &failed_encoder, 1, &hash_map, &failed)
        .unwrap();
    failed_encoder.end();
    let ready_encoder = KernelEncoder::begin(&command);
    scratch
        .encode_route_hash_gpu_into(&ctx, &ready_encoder, 0, &hash_map, &ready)
        .unwrap();
    ready_encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "{:?}", command.error());

    let completed = routes.read_completed().unwrap();
    let error = completed.validate_layer(7).unwrap_err();
    assert!(error.to_string().contains("invalid hash token"));
    completed.validate_layer(8).unwrap();
    assert_eq!(
        read_i32(&failed.status),
        vec![DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN]
    );
    assert_eq!(
        read_i32(&ready.status),
        vec![DEEPSEEK_V4_ROUTE_STATUS_READY]
    );

    let selections = DeepSeekV4LayerSelectionRecords::new(&ctx).unwrap();
    selections.reset_for_token().unwrap();
    assert!(
        selections
            .read_completed()
            .unwrap()
            .validate_layer(19, 513)
            .unwrap_err()
            .to_string()
            .contains("visible=-1")
    );
    let failed = selections.layer(19).unwrap();
    let ready = selections.layer(21).unwrap();
    host_write_i32(&failed.visible_count, &[513], "failed visible count").unwrap();
    host_write_i32(&failed.selected_count, &[511], "failed selected count").unwrap();
    host_write_i32(&failed.status, &[3], "failed selector status").unwrap();
    host_write_i32(&ready.visible_count, &[513], "ready visible count").unwrap();
    host_write_i32(&ready.selected_count, &[512], "ready selected count").unwrap();
    host_write_i32(&ready.status, &[0], "ready selector status").unwrap();
    let completed = selections.read_completed().unwrap();
    assert!(
        completed
            .validate_layer(19, 513)
            .unwrap_err()
            .to_string()
            .contains("selected=511 status=3")
    );
    completed.validate_layer(21, 513).unwrap();
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn collapsed_fp4_layer_records_bind_status_ids_source_and_schedule() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const LAYER: usize = 2;
    const VISIBLE: usize = 513;
    let records = DeepSeekV4LayerFp4SelectionRecords::new(&ctx).unwrap();
    records
        .reset_for_token(
            DeepSeekV4Fp4ShadowExecution::Singleton,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    assert!(
        records
            .read_completed()
            .unwrap()
            .validate_layer(
                LAYER,
                VISIBLE,
                DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                DeepSeekV4Fp4SelectionSource::Fp4,
            )
            .unwrap_err()
            .to_string()
            .contains("completion record")
    );

    records
        .reset_for_token(
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    records
        .read_completed()
        .unwrap()
        .validate_inactive_layer(
            4,
            DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
            DeepSeekV4Fp4SelectionSource::Fp4,
        )
        .unwrap();
    let layer = records.layer(LAYER).unwrap();
    let ids = (0..DEEPSEEK_V4_CSA_TOP_K as i32).collect::<Vec<_>>();
    host_write_i32(&layer.cache_order_ids, &ids, "ready collapsed FP4 IDs").unwrap();
    host_write_i32(
        &layer.eligible_visible,
        &[VISIBLE as i32],
        "ready collapsed FP4 visibility",
    )
    .unwrap();
    host_write_i32(
        &layer.eligibility_record,
        &[0, -1, 0],
        "ready collapsed FP4 eligibility",
    )
    .unwrap();
    let completed = records.read_completed().unwrap();
    assert_eq!(
        completed
            .validate_layer(
                LAYER,
                VISIBLE,
                DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                DeepSeekV4Fp4SelectionSource::Fp4,
            )
            .unwrap(),
        ids
    );

    let mut duplicate = ids.clone();
    duplicate[511] = duplicate[510];
    host_write_i32(
        &layer.cache_order_ids,
        &duplicate,
        "duplicate collapsed FP4 IDs",
    )
    .unwrap();
    assert!(
        records
            .read_completed()
            .unwrap()
            .validate_layer(
                LAYER,
                VISIBLE,
                DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                DeepSeekV4Fp4SelectionSource::Fp4,
            )
            .unwrap_err()
            .to_string()
            .contains("not sorted, unique, and in range")
    );

    host_write_i32(&layer.cache_order_ids, &ids, "restore collapsed FP4 IDs").unwrap();
    host_write_i32(
        &layer.eligibility_record,
        &[3, 512, 2],
        "failed collapsed FP4 eligibility",
    )
    .unwrap();
    assert!(
        records
            .read_completed()
            .unwrap()
            .validate_layer(
                LAYER,
                VISIBLE,
                DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                DeepSeekV4Fp4SelectionSource::Fp4,
            )
            .unwrap_err()
            .to_string()
            .contains("completion record")
    );

    let inactive = records.layer(4).unwrap();
    let mut overwritten = vec![-1; DEEPSEEK_V4_CSA_TOP_K];
    overwritten[0] = 0;
    host_write_i32(
        &inactive.cache_order_ids,
        &overwritten,
        "overwritten inactive collapsed FP4 IDs",
    )
    .unwrap();
    assert!(
        records
            .read_completed()
            .unwrap()
            .validate_inactive_layer(
                4,
                DeepSeekV4Fp4ShadowExecution::SingletonCollapsed,
                DeepSeekV4Fp4SelectionSource::Fp4,
            )
            .unwrap_err()
            .to_string()
            .contains("was modified")
    );
}

#[test]
fn gpu_moe_route_records_match_oracle_and_fail_closed() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const E: usize = 8;
    const K: usize = 3;
    let config = DeepSeekV4MoeConfig {
        hidden_size: 2,
        ffn_size: 2,
        expert_count: E,
        top_k: K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();

    let write_raw_f32 = |values: &[f32]| {
        assert_eq!(values.len(), E);
        let destination = unsafe {
            scratch
                .logits
                .buffer
                .contents()
                .as_ptr()
                .add(usize::try_from(scratch.logits.offset).unwrap()) as *mut f32
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
    };

    let run = |encode: &dyn Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "GPU route command failed: {:?}",
            command.error()
        );
        scratch.capture_gpu_route_record().unwrap()
    };

    let logits = [-3.0, 0.5, 1.25, -0.75, 2.0, 0.125, -1.5, 0.875];
    write_raw_f32(&logits);
    let bias_values = [0.0, 0.3, -0.2, 0.1, -0.4, 0.0, 0.2, -0.1];
    let bias = offset_f32(&ctx, &bias_values, vec![E as u64]);
    let learned = run(&|encoder| scratch.encode_route_learned_gpu(&ctx, encoder, &bias));
    let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits).unwrap();
    let expected = crate::deepseek_v4_oracle::learned_route(&scores, &bias_values, K, 1.5).unwrap();
    assert_eq!(learned.status, DEEPSEEK_V4_ROUTE_STATUS_READY);
    assert_eq!(
        learned.expert_ids,
        expected
            .expert_ids
            .iter()
            .map(|&expert| expert as i32)
            .collect::<Vec<_>>()
    );
    assert_close(
        "GPU learned route",
        &learned.weights,
        &expected.weights,
        2e-6,
    );
    let learned_repeat = run(&|encoder| scratch.encode_route_learned_gpu(&ctx, encoder, &bias));
    assert_eq!(learned_repeat.expert_ids, learned.expert_ids);
    assert_eq!(
        learned_repeat
            .weights
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        learned
            .weights
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let hash_values = [0, 2, 4, 1, 7, 3, 6, 5, 4];
    let hash_map = offset_i32(&ctx, &hash_values, vec![K as u64, 3]);
    let hash = run(&|encoder| scratch.encode_route_hash_gpu(&ctx, encoder, 1, &hash_map));
    let expected = crate::deepseek_v4_oracle::hash_route(&scores, &[1, 7, 3], 1.5).unwrap();
    assert_eq!(hash.status, DEEPSEEK_V4_ROUTE_STATUS_READY);
    assert_eq!(hash.expert_ids, vec![1, 7, 3]);
    assert_close("GPU hash route", &hash.weights, &expected.weights, 2e-6);

    write_raw_f32(&[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let failed = run(&|encoder| scratch.encode_route_learned_gpu(&ctx, encoder, &bias));
    assert_eq!(failed.status, DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT);
    assert_eq!(failed.expert_ids, vec![-1; K]);
    assert_eq!(failed.weights, vec![0.0; K]);

    write_raw_f32(&logits);
    let nonfinite_bias = offset_f32(
        &ctx,
        &[0.0, 0.0, f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![E as u64],
    );
    assert_eq!(
        run(&|encoder| scratch.encode_route_learned_gpu(&ctx, encoder, &nonfinite_bias)).status,
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS
    );
    assert_eq!(
        run(&|encoder| scratch.encode_route_hash_gpu(&ctx, encoder, 3, &hash_map)).status,
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN
    );
    let invalid_map = offset_i32(&ctx, &[0, 2, 4, 1, 8, 3], vec![K as u64, 2]);
    assert_eq!(
        run(&|encoder| scratch.encode_route_hash_gpu(&ctx, encoder, 1, &invalid_map)).status,
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT
    );
    let duplicate_map = offset_i32(&ctx, &[0, 2, 4, 1, 1, 3], vec![K as u64, 2]);
    let duplicate = run(&|encoder| scratch.encode_route_hash_gpu(&ctx, encoder, 1, &duplicate_map));
    let expected_duplicate =
        crate::deepseek_v4_oracle::hash_route(&scores, &[1, 1, 3], 1.5).unwrap();
    assert_eq!(duplicate.status, DEEPSEEK_V4_ROUTE_STATUS_READY);
    assert_eq!(duplicate.expert_ids, vec![1, 1, 3]);
    assert_close(
        "GPU duplicate-slot hash route",
        &duplicate.weights,
        &expected_duplicate.weights,
        2e-6,
    );
}

#[test]
fn failed_gpu_route_zeros_all_indexed_experts_and_rejects_after_completion() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 256;
    const F: usize = 256;
    const E: usize = 3;
    const K: usize = 3;
    let config = DeepSeekV4MoeConfig {
        hidden_size: H,
        ffn_size: F,
        expert_count: E,
        top_k: K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    host_write_f32(
        &scratch.normalized_input,
        &vec![0.25; H],
        "failed-route normalized input",
    )
    .unwrap();
    host_write_f32(
        &scratch.expert_outputs,
        &vec![7.0; H * K],
        "stale expert outputs",
    )
    .unwrap();
    host_write_f32(&scratch.routed_output, &vec![9.0; H], "stale routed output").unwrap();
    let logits_destination = unsafe {
        scratch
            .logits
            .buffer
            .contents()
            .as_ptr()
            .add(usize::try_from(scratch.logits.offset).unwrap()) as *mut f32
    };
    let bad_logits = [f32::NAN, 0.0, 0.0];
    unsafe {
        std::ptr::copy_nonoverlapping(bad_logits.as_ptr(), logits_destination, E);
    }

    let zero_bank = |dtype: GgmlType, n_in: usize, n_out: usize| {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let bytes = n_in * n_out * E / block_elements as usize * block_bytes as usize;
        MetalTensor::from_bytes(
            &ctx,
            &vec![0u8; bytes],
            vec![n_in as u64, n_out as u64, E as u64],
            dtype,
        )
        .unwrap()
    };
    let gate_bank = zero_bank(GgmlType::IQ2_S, H, F);
    let up_bank = zero_bank(GgmlType::IQ2_S, H, F);
    let down_bank = zero_bank(GgmlType::IQ3_XXS, F, H);
    let shared_gate = offset_f32(&ctx, &vec![0.0; H * F], vec![H as u64, F as u64]);
    let shared_up = offset_f32(&ctx, &vec![0.0; H * F], vec![H as u64, F as u64]);
    let shared_down = offset_f32(&ctx, &vec![0.0; F * H], vec![F as u64, H as u64]);
    let bias = offset_f32(&ctx, &[0.0; E], vec![E as u64]);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    scratch
        .encode_route_learned_gpu(&ctx, &encoder, &bias)
        .unwrap();
    scratch
        .encode_experts_indexed(
            &ctx,
            &encoder,
            &gate_bank,
            &up_bank,
            &down_bank,
            &shared_gate,
            &shared_up,
            &shared_down,
            10.0,
            10.0,
        )
        .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "failed-route composition command failed: {:?}",
        command.error()
    );
    assert_eq!(read_f32(scratch.expert_outputs()), vec![0.0; H * K]);
    assert_eq!(read_f32(scratch.routed_output()), vec![0.0; H]);
    let error = scratch.validate_gpu_route_completed().unwrap_err();
    assert!(error.to_string().contains("non-finite logit"));
}

#[test]
#[ignore = "focused GPU route-dispatch profiler; run explicitly with --nocapture"]
fn profile_gpu_moe_route_dispatches() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const REPEATS: usize = 100;
    let config = DeepSeekV4MoeConfig {
        hidden_size: 1,
        ffn_size: 1,
        expert_count: 256,
        top_k: 6,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    let logits = (0..256)
        .map(|index| ((index * 37 + 11) % 257) as f32 * 0.03125 - 4.0)
        .collect::<Vec<_>>();
    host_write_f32(&scratch.logits, &logits, "profile route logits").unwrap();
    let bias = offset_f32(
        &ctx,
        &(0..256)
            .map(|index| ((index * 19 + 3) % 127) as f32 * 0.0005 - 0.03)
            .collect::<Vec<_>>(),
        vec![256],
    );
    let hash_map = offset_i32(&ctx, &[3, 17, 42, 91, 173, 255], vec![6, 1]);

    let measure = |encode: &dyn Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>| {
        let warm = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&warm);
        encode(&encoder).unwrap();
        encoder.end();
        warm.commit();
        warm.waitUntilCompleted();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for _ in 0..REPEATS {
            encode(&encoder).unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / REPEATS as f64
    };

    let learned = measure(&|encoder| scratch.encode_route_learned_gpu(&ctx, encoder, &bias));
    let hash = measure(&|encoder| scratch.encode_route_hash_gpu(&ctx, encoder, 0, &hash_map));
    eprintln!(
        "deepseek_v4 gpu_route_dispatch learned_ms={learned:.6} hash_ms={hash:.6} repeats={REPEATS}"
    );
}

#[test]
fn gpu_learned_routes_preserve_cpu_ids_across_cutoff_shapes() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const E: usize = 256;
    const K: usize = 6;
    let config = DeepSeekV4MoeConfig {
        hidden_size: 1,
        ffn_size: 1,
        expert_count: E,
        top_k: K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    let write_logits = |values: &[f32]| {
        assert_eq!(values.len(), E);
        let destination = unsafe {
            scratch
                .logits
                .buffer
                .contents()
                .as_ptr()
                .add(usize::try_from(scratch.logits.offset).unwrap()) as *mut f32
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
    };
    let run = |bias: &MetalTensor| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_route_learned_gpu(&ctx, &encoder, bias)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        scratch.capture_gpu_route_record().unwrap()
    };

    let mut cases = Vec::new();
    cases.push((vec![0.0; E], vec![0.0; E], "all tied"));
    let mut cutoff_bias = vec![-1.0; E];
    cutoff_bias[..7].fill(0.25);
    cutoff_bias[6] = f32::from_bits(0.25f32.to_bits() - 1);
    cases.push((vec![0.0; E], cutoff_bias, "one-ULP cutoff"));
    let branch_values = [
        f32::from_bits((-20.0f32).to_bits() + 1),
        -20.0,
        f32::from_bits((-20.0f32).to_bits() - 1),
        f32::from_bits(20.0f32.to_bits() - 1),
        20.0,
        f32::from_bits(20.0f32.to_bits() + 1),
        -f32::MAX,
        f32::MAX,
    ];
    cases.push((
        (0..E)
            .map(|expert| branch_values[expert % branch_values.len()])
            .collect(),
        (0..E)
            .map(|expert| (expert % 11) as f32 * 0.0001 - 0.0005)
            .collect(),
        "finite extremes and softplus branches",
    ));
    for case in 0..24usize {
        let logits = (0..E)
            .map(|expert| {
                let mixed = expert * 1_103 + case * 7_919 + (expert ^ case) * 53;
                (mixed % 8_191) as f32 * 0.0025 - 10.0
            })
            .collect::<Vec<_>>();
        let bias = (0..E)
            .map(|expert| {
                let mixed = expert * 193 + case * 389 + expert / 7;
                (mixed % 257) as f32 * 0.0002 - 0.0256
            })
            .collect::<Vec<_>>();
        cases.push((logits, bias, "mixed deterministic"));
    }

    for (case, (logits, bias_values, label)) in cases.into_iter().enumerate() {
        write_logits(&logits);
        let bias = offset_f32(&ctx, &bias_values, vec![E as u64]);
        let actual = run(&bias);
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits).unwrap();
        let expected =
            crate::deepseek_v4_oracle::learned_route(&scores, &bias_values, K, 1.5).unwrap();
        assert_eq!(
            actual.status, DEEPSEEK_V4_ROUTE_STATUS_READY,
            "{label} {case}"
        );
        assert_eq!(
            actual.expert_ids,
            expected
                .expert_ids
                .iter()
                .map(|&expert| expert as i32)
                .collect::<Vec<_>>(),
            "{label} {case}"
        );
        assert_close(
            &format!("{label} {case} weights"),
            &actual.weights,
            &expected.weights,
            1e-4,
        );
    }
}

#[test]
fn indexed_ds4_expert_projections_match_static_views_and_zero_failures() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const N_IN: usize = 256;
    const N_OUT: usize = 8;
    const EXPERTS: usize = 3;

    let make_bank = |dtype: GgmlType| {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let block_elements = block_elements as usize;
        let block_bytes = block_bytes as usize;
        let blocks = N_IN * N_OUT * EXPERTS / block_elements;
        let mut payload = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let start = block * block_bytes;
            if dtype == GgmlType::MXFP4 {
                payload[start] = 127 + (block % 3) as u8;
                for byte in 1..block_bytes {
                    payload[start + byte] = (block * 29 + byte * 17 + 11) as u8;
                }
            } else if dtype == GgmlType::Q3_K {
                for byte in 0..block_bytes - 2 {
                    payload[start + byte] = (block * 31 + byte * 13 + 7) as u8;
                }
                payload[start + block_bytes - 2..start + block_bytes].copy_from_slice(
                    &half::f16::from_f32(0.015625 * (1 + block % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
            } else if dtype == GgmlType::Q4_K {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.015625 * (1 + block % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                payload[start + 2..start + 4].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + block % 3) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 4..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + 7) as u8;
                }
            } else {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.015625 * (1 + block % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 2..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + 7) as u8;
                }
            }
        }
        let prefix = 32usize;
        let mut bytes = vec![0xA5; prefix];
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&[0x5A; 32]);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).unwrap(),
            offset: prefix as u64,
            shape: vec![N_IN as u64, N_OUT as u64, EXPERTS as u64],
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    };
    let input_values = (0..N_IN)
        .map(|index| ((index * 37 + 5) % 127) as f32 * 0.001 - 0.063)
        .collect::<Vec<_>>();
    let input = offset_f32(&ctx, &input_values, vec![N_IN as u64]);
    let expert_ids = offset_i32(&ctx, &[2, 0], vec![2]);
    let ready = offset_i32(&ctx, &[DEEPSEEK_V4_ROUTE_STATUS_READY], vec![1]);

    for dtype in [
        GgmlType::IQ2_XS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::MXFP4,
    ] {
        let bank = make_bank(dtype);
        let static_weight =
            expert_weight_view(&bank, N_IN, N_OUT, 2, "static indexed-projection oracle").unwrap();
        let expected = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);
        let actual = offset_f32(&ctx, &[0.0; N_OUT], vec![N_OUT as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_projection(
            &ctx,
            &encoder,
            &static_weight,
            &input,
            &expected,
            N_IN,
            N_OUT,
            "static indexed-projection oracle",
        )
        .unwrap();
        encode_ds4_indexed_expert_projection(
            &ctx,
            &encoder,
            &bank,
            &input,
            &expert_ids,
            &ready,
            &actual,
            N_IN,
            N_OUT,
            EXPERTS,
            0,
            "indexed expert projection",
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "{dtype:?} indexed command failed: {:?}",
            command.error()
        );
        let actual_values = read_f32(&actual);
        let expected_values = read_f32(&expected);
        if matches!(dtype, GgmlType::Q3_K | GgmlType::Q4_K) {
            assert_close(
                &format!("{dtype:?} indexed projection"),
                &actual_values,
                &expected_values,
                5e-4,
            );
        } else {
            assert_eq!(
                actual_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{dtype:?} indexed projection changed reduction lineage"
            );
        }

        for (label, ids, status) in [
            ("failed status", vec![2, 0], -1),
            ("negative ID", vec![-1, 0], DEEPSEEK_V4_ROUTE_STATUS_READY),
            (
                "oversized ID",
                vec![EXPERTS as i32, 0],
                DEEPSEEK_V4_ROUTE_STATUS_READY,
            ),
        ] {
            let invalid_ids = offset_i32(&ctx, &ids, vec![2]);
            let invalid_status = offset_i32(&ctx, &[status], vec![1]);
            let output = offset_f32(&ctx, &[7.0; N_OUT], vec![N_OUT as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_ds4_indexed_expert_projection(
                &ctx,
                &encoder,
                &bank,
                &input,
                &invalid_ids,
                &invalid_status,
                &output,
                N_IN,
                N_OUT,
                EXPERTS,
                0,
                "invalid indexed expert projection",
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{dtype:?} {label}");
            assert_eq!(read_f32(&output), vec![0.0; N_OUT], "{dtype:?} {label}");
        }
    }
}

#[test]
fn all_slot_routed_experts_match_serial_indexed_path_and_zero_failures() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 256;
    const F: usize = 256;
    const E: usize = 7;
    const K: usize = 6;
    const CLAMP: f32 = 0.25;

    let make_bank = |dtype: GgmlType, n_in: usize, n_out: usize, seed: usize| {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let block_elements = block_elements as usize;
        let block_bytes = block_bytes as usize;
        let blocks = n_in * n_out * E / block_elements;
        let mut payload = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let start = block * block_bytes;
            if dtype == GgmlType::MXFP4 {
                payload[start] = 126 + ((block + seed) % 4) as u8;
                for byte in 1..block_bytes {
                    payload[start + byte] = (block * 29 + byte * 17 + seed * 11 + 7) as u8;
                }
            } else if dtype == GgmlType::Q3_K {
                for byte in 0..block_bytes - 2 {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
                payload[start + block_bytes - 2..start + block_bytes].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
            } else if dtype == GgmlType::Q4_K {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                payload[start + 2..start + 4].copy_from_slice(
                    &half::f16::from_f32(0.00390625 * (1 + (block + seed) % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 4..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
            } else {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 2..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
            }
        }
        let prefix = 32usize;
        let mut bytes = vec![0xA5; prefix];
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&[0x5A; 32]);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).unwrap(),
            offset: prefix as u64,
            shape: vec![n_in as u64, n_out as u64, E as u64],
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    };
    let bits = |tensor: &MetalTensor| {
        read_f32(tensor)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let config = DeepSeekV4MoeConfig {
        hidden_size: H,
        ffn_size: F,
        expert_count: E,
        top_k: K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    let input = (0..H)
        .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
        .collect::<Vec<_>>();
    host_write_f32(&scratch.normalized_input, &input, "all-slot test input").unwrap();
    let ids = [6, 0, 5, 1, 4, 2];
    host_write_i32(&scratch.expert_ids, &ids, "all-slot test IDs").unwrap();
    host_write_i32(
        &scratch.route_status,
        &[DEEPSEEK_V4_ROUTE_STATUS_READY],
        "all-slot test status",
    )
    .unwrap();
    let gate_scratch = scratch.gate.view_subrange(0, vec![F as u64]);

    for (gate_dtype, down_dtype) in [
        (GgmlType::IQ2_XS, GgmlType::IQ3_XXS),
        (GgmlType::IQ2_XS, GgmlType::MXFP4),
        (GgmlType::IQ2_S, GgmlType::IQ3_XXS),
        (GgmlType::IQ2_S, GgmlType::MXFP4),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_XXS, GgmlType::MXFP4),
        (GgmlType::IQ3_S, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_S, GgmlType::MXFP4),
        (GgmlType::Q3_K, GgmlType::Q4_K),
    ] {
        let gate_bank = make_bank(gate_dtype, H, F, 1);
        let up_bank = make_bank(gate_dtype, H, F, 3);
        let down_bank = make_bank(down_dtype, F, H, 5);
        let expected_inner = MetalTensor::zeros_f32(&ctx, vec![F as u64, K as u64]).unwrap();
        let expected_output = MetalTensor::zeros_f32(&ctx, vec![H as u64, K as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for slot in 0..K {
            encode_ds4_indexed_expert_projection(
                &ctx,
                &encoder,
                &gate_bank,
                &scratch.normalized_input,
                &scratch.expert_ids,
                &scratch.route_status,
                &gate_scratch,
                H,
                F,
                E,
                slot,
                "serial all-slot gate",
            )
            .unwrap();
            encode_ds4_indexed_expert_projection(
                &ctx,
                &encoder,
                &up_bank,
                &scratch.normalized_input,
                &scratch.expert_ids,
                &scratch.route_status,
                &scratch.up,
                H,
                F,
                E,
                slot,
                "serial all-slot up",
            )
            .unwrap();
            let inner = expected_inner.view_subrange((slot * F) as u64, vec![F as u64]);
            encode_ds4_clamped_swiglu(&ctx, &encoder, &gate_scratch, &scratch.up, &inner, CLAMP)
                .unwrap();
            let output = expected_output.view_subrange((slot * H) as u64, vec![H as u64]);
            encode_ds4_indexed_expert_projection(
                &ctx,
                &encoder,
                &down_bank,
                &inner,
                &scratch.expert_ids,
                &scratch.route_status,
                &output,
                F,
                H,
                E,
                slot,
                "serial all-slot down",
            )
            .unwrap();
        }
        scratch
            .encode_routed_experts_all_slots(
                &ctx, &encoder, &gate_bank, &up_bank, &down_bank, CLAMP,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "{gate_dtype:?}/{down_dtype:?} all-slot command failed: {:?}",
            command.error()
        );
        if gate_dtype == GgmlType::Q3_K {
            assert_close(
                "Q3_K all-slot activation",
                &read_f32(scratch.routed_inner()),
                &read_f32(&expected_inner),
                5e-4,
            );
            assert_close(
                "Q3_K/Q4_K all-slot output",
                &read_f32(scratch.expert_outputs()),
                &read_f32(&expected_output),
                1e-3,
            );
        } else {
            assert_eq!(
                bits(scratch.routed_inner()),
                bits(&expected_inner),
                "{gate_dtype:?}/{down_dtype:?} fused activation changed bits"
            );
            assert_eq!(
                bits(scratch.expert_outputs()),
                bits(&expected_output),
                "{gate_dtype:?}/{down_dtype:?} all-slot down changed bits"
            );
        }
    }

    let gate_bank = make_bank(GgmlType::IQ2_S, H, F, 1);
    let up_bank = make_bank(GgmlType::IQ2_S, H, F, 3);
    let down_bank = make_bank(GgmlType::IQ3_XXS, F, H, 5);
    for (label, status, bad_slot, bad_id) in [
        ("failed status", -1, None, 0),
        ("negative ID", DEEPSEEK_V4_ROUTE_STATUS_READY, Some(0), -1),
        (
            "oversized ID",
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            Some(K - 1),
            E as i32,
        ),
    ] {
        let mut invalid_ids = ids;
        if let Some(slot) = bad_slot {
            invalid_ids[slot] = bad_id;
        }
        host_write_i32(&scratch.expert_ids, &invalid_ids, "invalid all-slot IDs").unwrap();
        host_write_i32(&scratch.route_status, &[status], "invalid all-slot status").unwrap();
        host_write_f32(
            scratch.routed_inner(),
            &vec![7.0; F * K],
            "stale all-slot inner",
        )
        .unwrap();
        host_write_f32(
            scratch.expert_outputs(),
            &vec![9.0; H * K],
            "stale all-slot output",
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_routed_experts_all_slots(
                &ctx, &encoder, &gate_bank, &up_bank, &down_bank, CLAMP,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{label}");
        let inner = read_f32(scratch.routed_inner());
        let output = read_f32(scratch.expert_outputs());
        if status != DEEPSEEK_V4_ROUTE_STATUS_READY {
            assert_eq!(inner, vec![0.0; F * K], "{label} inner");
            assert_eq!(output, vec![0.0; H * K], "{label} output");
        } else {
            let slot = bad_slot.unwrap();
            assert_eq!(&inner[slot * F..(slot + 1) * F], &vec![0.0; F], "{label}");
            assert_eq!(&output[slot * H..(slot + 1) * H], &vec![0.0; H], "{label}");
            for valid_slot in 0..K {
                if valid_slot != slot {
                    assert!(
                        inner[valid_slot * F..(valid_slot + 1) * F]
                            .iter()
                            .any(|value| *value != 0.0),
                        "{label} cleared valid inner slot {valid_slot}"
                    );
                }
            }
        }
    }

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let mixed_up = make_bank(GgmlType::IQ3_S, H, F, 9);
    let mixed_error = scratch
        .encode_routed_experts_all_slots(&ctx, &encoder, &gate_bank, &mixed_up, &down_bank, CLAMP)
        .unwrap_err();
    assert!(mixed_error.to_string().contains("require matching IQ2_XS"));
    let down_error = scratch
        .encode_routed_experts_all_slots(&ctx, &encoder, &gate_bank, &up_bank, &gate_bank, CLAMP)
        .unwrap_err();
    assert!(down_error.to_string().contains("IQ3_XXS, MXFP4, or Q4_K"));
    encoder.end();

    let short_scratch =
        DeepSeekV4MoeScratch::new(&ctx, DeepSeekV4MoeConfig { top_k: 3, ..config }).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let top_k_error = short_scratch
        .encode_routed_experts_all_slots(&ctx, &encoder, &gate_bank, &up_bank, &down_bank, CLAMP)
        .unwrap_err();
    assert!(top_k_error.to_string().contains("require top-k 6"));
    encoder.end();
}

#[test]
fn all_slot_iq2_xs_production_input_width_matches_cpu_codec() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 4_096;
    const F: usize = 8;
    const E: usize = 7;
    const K: usize = 6;
    const CLAMP: f32 = 0.25;

    let make_bank = |seed: usize| {
        let (_, block_bytes) = ggml_type_layout(GgmlType::IQ2_XS).unwrap();
        let block_bytes = block_bytes as usize;
        let blocks = H * F * E / 256;
        let mut payload = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let start = block * block_bytes;
            payload[start..start + 2].copy_from_slice(
                &half::f16::from_f32(0.000_244_140_63 * (1 + (block + seed) % 5) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
            for byte in 2..block_bytes {
                payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
            }
        }
        let desc = TensorDesc {
            name: format!("iq2_xs_bank_{seed}"),
            shape: vec![H as u64, F as u64, E as u64],
            dtype: GgmlType::IQ2_XS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: payload.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &payload).unwrap();
        let prefix = 32usize;
        let mut bytes = vec![0xA5; prefix];
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&[0x5A; 32]);
        let tensor = MetalTensor {
            buffer: ctx.buffer_from(&bytes).unwrap(),
            offset: prefix as u64,
            shape: desc.shape,
            dtype: desc.dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        };
        (tensor, decoded)
    };
    let (gate_bank, gate_decoded) = make_bank(1);
    let (up_bank, up_decoded) = make_bank(3);
    let input_values = (0..H)
        .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
        .collect::<Vec<_>>();
    let input = offset_f32(&ctx, &input_values, vec![H as u64]);
    let ids = [6, 0, 5, 1, 4, 2];
    let expert_ids = offset_i32(&ctx, &ids, vec![K as u64]);
    let route_status = offset_i32(&ctx, &[DEEPSEEK_V4_ROUTE_STATUS_READY], vec![1]);
    let output = offset_f32(&ctx, &[0.0; F * K], vec![F as u64, K as u64]);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_ds4_all_slots_gate_up_swiglu(
        &ctx,
        &encoder,
        &gate_bank,
        &up_bank,
        &input,
        &expert_ids,
        &route_status,
        &output,
        H,
        F,
        E,
        CLAMP,
    )
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none());

    let mut expected = vec![0.0f32; F * K];
    for (slot, &expert) in ids.iter().enumerate() {
        for row in 0..F {
            let start = (expert as usize * F + row) * H;
            let gate = gate_decoded[start..start + H]
                .iter()
                .zip(&input_values)
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let up = up_decoded[start..start + H]
                .iter()
                .zip(&input_values)
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let gate = gate.min(CLAMP);
            let up = up.clamp(-CLAMP, CLAMP);
            expected[slot * F + row] = gate / (1.0 + (-gate).exp()) * up;
        }
    }
    assert_close(
        "production-width IQ2_XS all-slot",
        &read_f32(&output),
        &expected,
        2e-5,
    );
}

#[test]
#[ignore = "focused production-shape indexed expert profiler; run explicitly"]
fn profile_indexed_ds4_expert_projections_against_static_views() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const EXPERTS: usize = 4;
    const REPEATS: usize = 40;
    let expert_ids = offset_i32(&ctx, &[3], vec![1]);
    let ready = offset_i32(&ctx, &[DEEPSEEK_V4_ROUTE_STATUS_READY], vec![1]);

    let measure = |encode: &dyn Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>| {
        let warm = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&warm);
        encode(&encoder).unwrap();
        encoder.end();
        warm.commit();
        warm.waitUntilCompleted();
        assert!(warm.error().is_none());

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for _ in 0..REPEATS {
            encode(&encoder).unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / REPEATS as f64
    };

    for (dtype, n_in, n_out, label) in [
        (GgmlType::IQ2_S, 4_096, 2_048, "gate_up_iq2_s"),
        (GgmlType::IQ3_S, 4_096, 2_048, "gate_up_iq3_s"),
        (GgmlType::IQ3_XXS, 2_048, 4_096, "down_iq3_xxs"),
        (GgmlType::MXFP4, 2_048, 4_096, "down_mxfp4"),
        (GgmlType::Q3_K, 4_096, 2_048, "gate_up_q3_k"),
        (GgmlType::Q4_K, 2_048, 4_096, "down_q4_k"),
    ] {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let bank_bytes = n_in * n_out * EXPERTS / block_elements as usize * block_bytes as usize;
        let bank = MetalTensor::from_bytes(
            &ctx,
            &vec![0u8; bank_bytes],
            vec![n_in as u64, n_out as u64, EXPERTS as u64],
            dtype,
        )
        .unwrap();
        let static_weight =
            expert_weight_view(&bank, n_in, n_out, 3, label).expect("static expert view");
        let input = offset_f32(
            &ctx,
            &(0..n_in)
                .map(|index| ((index * 17 + 3) % 127) as f32 * 0.0005 - 0.031)
                .collect::<Vec<_>>(),
            vec![n_in as u64],
        );
        let static_output = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let indexed_output = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let static_ms = measure(&|encoder| {
            encode_projection(
                &ctx,
                encoder,
                &static_weight,
                &input,
                &static_output,
                n_in,
                n_out,
                label,
            )
        });
        let indexed_ms = measure(&|encoder| {
            encode_ds4_indexed_expert_projection(
                &ctx,
                encoder,
                &bank,
                &input,
                &expert_ids,
                &ready,
                &indexed_output,
                n_in,
                n_out,
                EXPERTS,
                0,
                label,
            )
        });
        assert_eq!(
            read_f32(&indexed_output)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&static_output)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        eprintln!(
            "deepseek_v4 indexed_expert {label} static_ms={static_ms:.6} indexed_ms={indexed_ms:.6} ratio={:.3} repeats={REPEATS}",
            indexed_ms / static_ms
        );
    }
}

#[test]
#[ignore = "focused production-shape all-slot routed expert profiler"]
fn profile_all_slot_routed_experts_against_serial_indexed() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 4_096;
    const F: usize = 2_048;
    const E: usize = 7;
    const K: usize = 6;
    const REPEATS: usize = 20;
    let config = DeepSeekV4MoeConfig {
        hidden_size: H,
        ffn_size: F,
        expert_count: E,
        top_k: K,
        routed_scale: 1.5,
    };
    let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
    host_write_f32(
        &scratch.normalized_input,
        &(0..H)
            .map(|index| ((index * 17 + 3) % 127) as f32 * 0.0005 - 0.031)
            .collect::<Vec<_>>(),
        "all-slot profile input",
    )
    .unwrap();
    host_write_i32(
        &scratch.expert_ids,
        &[6, 0, 5, 1, 4, 2],
        "all-slot profile IDs",
    )
    .unwrap();
    host_write_i32(
        &scratch.route_status,
        &[DEEPSEEK_V4_ROUTE_STATUS_READY],
        "all-slot profile status",
    )
    .unwrap();

    let make_bank = |dtype: GgmlType, n_in: usize, n_out: usize, seed: usize| {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let block_elements = block_elements as usize;
        let block_bytes = block_bytes as usize;
        let blocks = n_in * n_out * E / block_elements;
        let mut payload = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let start = block * block_bytes;
            if dtype == GgmlType::MXFP4 {
                payload[start] = 126 + ((block + seed) % 4) as u8;
                for byte in 1..block_bytes {
                    payload[start + byte] = (block * 29 + byte * 17 + seed * 11 + 7) as u8;
                }
            } else if dtype == GgmlType::Q3_K {
                for byte in 0..block_bytes - 2 {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
                payload[start + block_bytes - 2..start + block_bytes].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
            } else if dtype == GgmlType::Q4_K {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                payload[start + 2..start + 4].copy_from_slice(
                    &half::f16::from_f32(0.00390625 * (1 + (block + seed) % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 4..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
            } else {
                payload[start..start + 2].copy_from_slice(
                    &half::f16::from_f32(0.0078125 * (1 + (block + seed) % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for byte in 2..block_bytes {
                    payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
                }
            }
        }
        MetalTensor::from_bytes(
            &ctx,
            &payload,
            vec![n_in as u64, n_out as u64, E as u64],
            dtype,
        )
        .unwrap()
    };
    let measure = |encode: &dyn Fn(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>| {
        let warm = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&warm);
        encode(&encoder).unwrap();
        encoder.end();
        warm.commit();
        warm.waitUntilCompleted();
        assert!(warm.error().is_none());

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for _ in 0..REPEATS {
            encode(&encoder).unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / REPEATS as f64
    };
    let bits = |tensor: &MetalTensor| {
        read_f32(tensor)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    for (gate_dtype, down_dtype, label) in [
        (GgmlType::IQ2_S, GgmlType::IQ3_XXS, "iq2_iq3xxs"),
        (GgmlType::IQ2_S, GgmlType::MXFP4, "iq2_mxfp4"),
        (GgmlType::IQ3_S, GgmlType::IQ3_XXS, "iq3s_iq3xxs"),
        (GgmlType::IQ3_S, GgmlType::MXFP4, "iq3s_mxfp4"),
        (GgmlType::Q3_K, GgmlType::Q4_K, "q3k_q4k"),
    ] {
        let gate_bank = make_bank(gate_dtype, H, F, 1);
        let up_bank = make_bank(gate_dtype, H, F, 3);
        let down_bank = make_bank(down_dtype, F, H, 5);
        let serial = || {
            measure(&|encoder| {
                scratch.encode_routed_experts_indexed(
                    &ctx, encoder, &gate_bank, &up_bank, &down_bank, 10.0,
                )
            })
        };
        let all_slot = || {
            measure(&|encoder| {
                scratch.encode_routed_experts_all_slots(
                    &ctx, encoder, &gate_bank, &up_bank, &down_bank, 10.0,
                )
            })
        };
        let gate_up = || {
            measure(&|encoder| {
                encode_ds4_all_slots_gate_up_swiglu(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &up_bank,
                    &scratch.normalized_input,
                    &scratch.expert_ids,
                    &scratch.route_status,
                    &scratch.routed_inner,
                    H,
                    F,
                    E,
                    10.0,
                )
            })
        };
        let down = || {
            measure(&|encoder| {
                encode_ds4_all_slots_down(
                    &ctx,
                    encoder,
                    &down_bank,
                    &scratch.routed_inner,
                    &scratch.expert_ids,
                    &scratch.route_status,
                    &scratch.expert_outputs,
                    F,
                    H,
                    E,
                )
            })
        };
        let serial_before_ms = serial();
        let serial_values = read_f32(scratch.expert_outputs());
        let serial_bits = bits(scratch.expert_outputs());
        let all_slot_ms = all_slot();
        if gate_dtype == GgmlType::Q3_K {
            assert_close(
                label,
                &read_f32(scratch.expert_outputs()),
                &serial_values,
                1e-3,
            );
        } else {
            assert_eq!(bits(scratch.expert_outputs()), serial_bits, "{label}");
        }
        let all_slot_repeat_ms = all_slot();
        let gate_up_ms = gate_up();
        let gate_up_repeat_ms = gate_up();
        let down_ms = down();
        let serial_after_ms = serial();
        assert_eq!(
            bits(scratch.expert_outputs()),
            serial_bits,
            "{label} repeat"
        );
        eprintln!(
            "deepseek_v4 all_slot_experts {label} serial_before_ms={serial_before_ms:.6} all_slot_ms={all_slot_ms:.6} all_slot_repeat_ms={all_slot_repeat_ms:.6} gate_up_ms={gate_up_ms:.6} gate_up_repeat_ms={gate_up_repeat_ms:.6} gate_up_median_ms={:.6} down_ms={down_ms:.6} isolated_warmed_sum_ms={:.6} serial_after_ms={serial_after_ms:.6} serial_median_ms={:.6} all_slot_median_ms={:.6} speedup={:.3} repeats={REPEATS}",
            (gate_up_ms + gate_up_repeat_ms) * 0.5,
            (gate_up_ms + gate_up_repeat_ms) * 0.5 + down_ms,
            (serial_before_ms + serial_after_ms) * 0.5,
            (all_slot_ms + all_slot_repeat_ms) * 0.5,
            (serial_before_ms + serial_after_ms) / (all_slot_ms + all_slot_repeat_ms),
        );
    }
}

#[test]
fn report_is_exact_for_views_aliases_and_final_page_fallback() {
    let view = f32_desc("view", 64);
    let alias = f32_desc("alias", 64);
    let tail = f32_desc("tail", 128);
    let plan = plan_retained_storage(&[160], &[&view, &alias, &tail], 64, 128, 32)
        .expect("plan retained synthetic storage");
    validate_fallback_policy(&plan).expect("only final-page fallback");
    let report = report_for_plan(&plan).expect("report");

    assert_eq!(report.tensor_count, 3);
    assert_eq!(report.source_bytes, 96);
    assert_eq!(report.window_count, 1);
    assert_eq!(report.window_bytes, 64);
    assert_eq!(report.view_count, 1);
    assert_eq!(report.unique_view_bytes, 32);
    assert_eq!(report.logical_view_bytes, 64);
    assert_eq!(report.alias_count, 1);
    assert_eq!(report.alias_bytes, 32);
    assert_eq!(report.fallback_count, 1);
    assert_eq!(report.fallback_bytes, 32);
    assert_eq!(report.resident_bytes, 96);
    assert_eq!(report.required_alignment, 32);
}

#[test]
fn non_final_page_fallback_is_rejected() {
    let misaligned = f32_desc("misaligned", 4);
    let plan = plan_retained_storage(&[128], &[&misaligned], 64, 128, 32)
        .expect("planner classifies misalignment");
    let error = validate_fallback_policy(&plan).expect_err("misalignment must fail closed");
    assert!(error.to_string().contains("BindingMisalignment"));
}

#[test]
fn native_hyper_connections_match_oracle_with_offsets_and_asymmetric_streams() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 7;
    const N: usize = H * DEEPSEEK_V4_CONNECTION_COUNT;
    let residual_values = (0..N)
        .map(|index| {
            let stream = index / H;
            let dimension = index % H;
            (stream as f32 - 1.25) * 0.73
                + (dimension as f32 - 2.4) * (0.11 + stream as f32 * 0.07)
                + if (index + stream).is_multiple_of(3) {
                    -0.19
                } else {
                    0.13
                }
        })
        .collect::<Vec<_>>();
    let function = (0..N * DEEPSEEK_V4_HC_PARAMETER_COUNT)
        .map(|index| ((index * 17 + index / 11) % 31) as f32 * 0.006 - 0.087)
        .collect::<Vec<_>>();
    let scale_values = [0.7, -0.4, 1.2];
    let base_values = (0..DEEPSEEK_V4_HC_PARAMETER_COUNT)
        .map(|index| ((index * 7 + 3) % 19) as f32 * 0.035 - 0.29)
        .collect::<Vec<_>>();
    let block_values = (0..H)
        .map(|index| (index as f32 - 2.2) * 0.31)
        .collect::<Vec<_>>();
    let head_function = (0..N * DEEPSEEK_V4_CONNECTION_COUNT)
        .map(|index| ((index * 13 + 5) % 23) as f32 * 0.009 - 0.091)
        .collect::<Vec<_>>();
    let head_scale_values = [-0.63];
    let head_base_values = [0.17, -0.31, 0.08, 0.27];
    let rms_eps = 1.0e-5;
    let hc_eps = 1.0e-6;

    let expected_pre = hyper_connection_pre(
        &residual_values,
        H,
        4,
        &function,
        &scale_values,
        &base_values,
        rms_eps,
        DEEPSEEK_V4_SINKHORN_ITERATIONS,
        hc_eps,
    )
    .expect("pre oracle");
    let expected_post = hyper_connection_post(
        &block_values,
        &residual_values,
        &expected_pre.controls,
        H,
        4,
    )
    .expect("post oracle");
    let expected_head = hyper_connection_head(
        &expected_post,
        H,
        4,
        &head_function,
        head_scale_values[0],
        &head_base_values,
        rms_eps,
        hc_eps,
    )
    .expect("head oracle");
    let expected_head_mixes = mat_vec(
        &head_function,
        N,
        4,
        &rms_norm(&expected_post, None, rms_eps).expect("head norm oracle"),
    )
    .expect("head mix oracle");
    let expected_head_gates = expected_head_mixes
        .iter()
        .zip(head_base_values)
        .map(|(&mix, base)| 1.0 / (1.0 + (-(mix * head_scale_values[0] + base)).exp()) + hc_eps)
        .collect::<Vec<_>>();

    let residual = offset_f32(&ctx, &residual_values, vec![H as u64, 4]);
    let function_tensor = offset_f32(&ctx, &function, vec![N as u64, 24]);
    let scale = offset_f32(&ctx, &scale_values, vec![3]);
    let base = offset_f32(&ctx, &base_values, vec![24]);
    let block = offset_f32(&ctx, &block_values, vec![H as u64]);
    let post_output = offset_f32(&ctx, &[0.0; N], vec![H as u64, 4]);
    let head_function_tensor = offset_f32(&ctx, &head_function, vec![N as u64, 4]);
    let head_scale = offset_f32(&ctx, &head_scale_values, vec![1]);
    let head_base = offset_f32(&ctx, &head_base_values, vec![4]);
    let head_output = offset_f32(&ctx, &[0.0; H], vec![H as u64]);
    let scratch = DeepSeekV4HyperConnectionScratch::new(&ctx, H).expect("HC scratch");
    assert_eq!(read_f32(&residual), residual_values);
    assert_eq!(read_f32(&scratch.ones), vec![1.0; N]);

    let command = ctx.queue.commandBuffer().expect("norm preflight buffer");
    let encoder = KernelEncoder::begin(&command);
    encode_rms_norm_mul_f32(
        &ctx,
        &encoder,
        &residual,
        &scratch.ones,
        &scratch.normalized,
        rms_eps,
    )
    .expect("norm preflight");
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_close(
        "offset norm preflight",
        &read_f32(scratch.normalized()),
        &rms_norm(&residual_values, None, rms_eps).unwrap(),
        2e-5,
    );

    let command = ctx.queue.commandBuffer().expect("HC command buffer");
    let encoder = KernelEncoder::begin(&command);
    scratch
        .encode_pre(
            &ctx,
            &encoder,
            &residual,
            &function_tensor,
            &scale,
            &base,
            rms_eps,
            hc_eps,
        )
        .expect("encode pre");
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "pre command failed: {:?}",
        command.error()
    );

    assert_close(
        "flattened norm",
        &read_f32(scratch.normalized()),
        &rms_norm(&residual_values, None, rms_eps).unwrap(),
        2e-5,
    );
    assert_close(
        "pre mixes",
        &read_f32(scratch.mixes()),
        &expected_pre.mixes,
        3e-5,
    );
    assert_close(
        "pre gates",
        &read_f32(scratch.pre_gates()),
        &expected_pre.controls.pre,
        2e-5,
    );
    assert_close(
        "post gates",
        &read_f32(scratch.post_gates()),
        &expected_pre.controls.post,
        2e-5,
    );
    assert_close(
        "combination source-major",
        &read_f32(scratch.combination()),
        &expected_pre.controls.combination,
        3e-5,
    );
    assert_close(
        "collapsed input",
        &read_f32(scratch.collapsed_input()),
        &expected_pre.input,
        3e-5,
    );

    let command = ctx.queue.commandBuffer().expect("post/head command buffer");
    let encoder = KernelEncoder::begin(&command);
    scratch
        .encode_post(&ctx, &encoder, &block, &residual, &post_output)
        .expect("encode post");
    scratch
        .encode_head(
            &ctx,
            &encoder,
            &post_output,
            &head_function_tensor,
            &head_scale,
            &head_base,
            &head_output,
            rms_eps,
            hc_eps,
        )
        .expect("encode head");
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(
        command.error().is_none(),
        "post/head command failed: {:?}",
        command.error()
    );

    assert_close(
        "post residual",
        &read_f32(&post_output),
        &expected_post,
        4e-5,
    );
    assert_close(
        "head mixes",
        &read_f32(scratch.head_mixes()),
        &expected_head_mixes,
        3e-5,
    );
    assert_close(
        "head gates",
        &read_f32(scratch.head_gates()),
        &expected_head_gates,
        3e-5,
    );
    assert_close("final head", &read_f32(&head_output), &expected_head, 4e-5);

    let mut transposed_post = [0.0; N];
    for destination in 0..4 {
        for dimension in 0..H {
            let mut value = block_values[dimension] * expected_pre.controls.post[destination];
            for source in 0..4 {
                value += expected_pre.controls.combination[destination * 4 + source]
                    * residual_values[source * H + dimension];
            }
            transposed_post[destination * H + dimension] = value;
        }
    }
    assert!(
        transposed_post
            .iter()
            .zip(&expected_post)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "fixture must detect transposed combination axes"
    );
    let per_stream_norm = residual_values
        .chunks_exact(H)
        .flat_map(|stream| rms_norm(stream, None, rms_eps).unwrap())
        .collect::<Vec<_>>();
    let wrong_mixes = mat_vec(
        &function,
        N,
        DEEPSEEK_V4_HC_PARAMETER_COUNT,
        &per_stream_norm,
    )
    .unwrap();
    assert!(
        wrong_mixes
            .iter()
            .zip(&expected_pre.mixes)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "fixture must distinguish flattened and per-stream RMSNorm"
    );
}

#[test]
fn initial_repeat_and_first_pre_match_equal_stream_oracle() {
    let Some(ctx) = metal_context() else {
        return;
    };
    const H: usize = 9;
    const N: usize = H * 4;
    let embedding_values = (0..H)
        .map(|index| (index as f32 - 3.7) * 0.23 + if index % 2 == 0 { 0.14 } else { -0.09 })
        .collect::<Vec<_>>();
    let repeated = (0..4)
        .flat_map(|_| embedding_values.iter().copied())
        .collect::<Vec<_>>();
    let function = (0..N * 24)
        .map(|index| ((index * 5 + 1) % 29) as f32 * 0.004 - 0.052)
        .collect::<Vec<_>>();
    let scale_values = [0.41, 0.78, -0.57];
    let base_values = (0..24)
        .map(|index| index as f32 * 0.013 - 0.11)
        .collect::<Vec<_>>();
    let expected = hyper_connection_pre(
        &repeated,
        H,
        4,
        &function,
        &scale_values,
        &base_values,
        1e-5,
        20,
        1e-6,
    )
    .expect("equal stream oracle");

    let embedding = offset_f32(&ctx, &embedding_values, vec![H as u64]);
    let residual = offset_f32(&ctx, &[0.0; N], vec![H as u64, 4]);
    let function = offset_f32(&ctx, &function, vec![N as u64, 24]);
    let scale = offset_f32(&ctx, &scale_values, vec![3]);
    let base = offset_f32(&ctx, &base_values, vec![24]);
    let scratch = DeepSeekV4HyperConnectionScratch::new(&ctx, H).expect("HC scratch");
    let command = ctx.queue.commandBuffer().expect("repeat command buffer");
    let encoder = KernelEncoder::begin(&command);
    scratch
        .encode_initial_repeat(&ctx, &encoder, &embedding, &residual)
        .expect("repeat embedding");
    scratch
        .encode_pre(
            &ctx, &encoder, &residual, &function, &scale, &base, 1e-5, 1e-6,
        )
        .expect("first pre");
    encoder.end();
    command.commit();
    command.waitUntilCompleted();

    assert_eq!(read_f32(&residual), repeated);
    assert_close(
        "equal-stream pre",
        &read_f32(scratch.pre_gates()),
        &expected.controls.pre,
        2e-5,
    );
    assert_close(
        "equal-stream combination",
        &read_f32(scratch.combination()),
        &expected.controls.combination,
        3e-5,
    );
    assert_close(
        "equal-stream collapse",
        &read_f32(scratch.collapsed_input()),
        &expected.input,
        3e-5,
    );
}
