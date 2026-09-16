use super::*;
use crate::gguf::GgufFile;
use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
use crate::tensor::TensorDesc;
use objc2_metal::MTLCommandQueue;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn packed_test_context() -> Option<MetalContext> {
    crate::test_fixtures::metal_context_or_skip()
}

const ROUTING_ORACLE_JSON: &str = include_str!("../../tests/fixtures/qwen4exp_moe_routing_v1.json");
const ROUTING_ORACLE_F32: &[u8] =
    include_bytes!("../../tests/fixtures/qwen4exp_moe_routing_v1.f32");

#[derive(Deserialize)]
struct RoutingOracle {
    schema_version: u32,
    generator_version: u32,
    geometry: OracleGeometry,
    semantics: OracleSemantics,
    topk_ids: Vec<i32>,
    sentinels: OracleSentinels,
    binary: OracleBinary,
}

#[derive(Deserialize)]
struct OracleGeometry {
    hidden_size: usize,
    expert_count: usize,
    experts_per_token: usize,
    routed_intermediate_size: usize,
    shared_intermediate_size: usize,
}

#[derive(Deserialize)]
struct OracleSentinels {
    exact_tie_ids: Vec<i32>,
    selected_weight_sum: f32,
    full_softmax_selected_sum_without_renormalization: f32,
}

#[derive(Deserialize)]
struct OracleSemantics {
    router: String,
    tie_break: String,
    normalization: String,
    shared_gate: String,
}

#[derive(Deserialize)]
struct OracleBinary {
    file: String,
    dtype: String,
    byte_order: String,
    sha256: String,
    sections: BTreeMap<String, OracleSection>,
}

#[derive(Deserialize)]
struct OracleSection {
    offset_f32: usize,
    count_f32: usize,
    shape: Vec<usize>,
}

struct CpuMoeResult {
    routed_inner: Vec<f32>,
    routed_expert_output: Vec<f32>,
    shared_inner: Vec<f32>,
    shared_output: Vec<f32>,
    output: Vec<f32>,
}

fn metal_context() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
        Err(error) => panic!("Metal initialization failed: {error}"),
    }
}

fn parse_oracle() -> (RoutingOracle, Vec<f32>) {
    let oracle: RoutingOracle = serde_json::from_str(ROUTING_ORACLE_JSON).unwrap();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.generator_version, 1);
    assert_eq!(oracle.binary.file, "qwen4exp_moe_routing_v1.f32");
    assert_eq!(oracle.binary.dtype, "f32");
    assert_eq!(oracle.binary.byte_order, "little");
    assert_eq!(oracle.semantics.router, "F32 matrix-vector product");
    assert_eq!(oracle.semantics.tie_break, "lower expert ID first");
    assert_eq!(
        oracle.semantics.normalization,
        "softmax over selected logits"
    );
    assert_eq!(
        oracle.semantics.shared_gate,
        "sigmoid(dot(shared_router, input))"
    );
    assert_eq!(oracle.sentinels.exact_tie_ids, [3, 14]);
    assert_eq!(oracle.sentinels.selected_weight_sum, 1.0);
    assert!(
        oracle
            .sentinels
            .full_softmax_selected_sum_without_renormalization
            < 0.9
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(ROUTING_ORACLE_F32)),
        oracle.binary.sha256
    );
    assert!(ROUTING_ORACLE_F32.len().is_multiple_of(4));
    let values = ROUTING_ORACLE_F32
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let required_sections = [
        "input",
        "router",
        "shared_router",
        "router_logits",
        "topk_weights",
        "shared_gate",
    ];
    assert_eq!(oracle.binary.sections.len(), required_sections.len());
    for name in required_sections {
        assert!(oracle.binary.sections.contains_key(name));
    }
    let mut sections = oracle.binary.sections.values().collect::<Vec<_>>();
    sections.sort_by_key(|section| section.offset_f32);
    let mut cursor = 0_usize;
    for section in sections {
        assert_eq!(section.offset_f32, cursor);
        assert_eq!(section.shape.iter().product::<usize>(), section.count_f32);
        cursor = cursor.checked_add(section.count_f32).unwrap();
        assert!(cursor <= values.len());
    }
    assert_eq!(cursor, values.len());
    (oracle, values)
}

fn oracle_section<'a>(oracle: &RoutingOracle, values: &'a [f32], name: &str) -> &'a [f32] {
    let section = &oracle.binary.sections[name];
    assert_eq!(section.shape.iter().product::<usize>(), section.count_f32);
    &values[section.offset_f32..section.offset_f32 + section.count_f32]
}

fn geometry(oracle: &RoutingOracle) -> Qwen4ExpMoeMetalGeometry {
    Qwen4ExpMoeMetalGeometry::new(
        oracle.geometry.hidden_size,
        oracle.geometry.expert_count,
        oracle.geometry.experts_per_token,
        oracle.geometry.routed_intermediate_size,
        oracle.geometry.shared_intermediate_size,
    )
    .unwrap()
}

fn tensor_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
}

fn tensor_i32(ctx: &MetalContext, values: &[i32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::I32).unwrap()
}

fn weight_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let mut tensor = tensor_f32(ctx, values, shape);
    tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    tensor
}

fn weight_bytes(ctx: &MetalContext, bytes: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let mut tensor = MetalTensor::from_bytes(ctx, bytes, shape, dtype).unwrap();
    tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    tensor
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, GgmlType::F32);
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
    }
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    assert_eq!(tensor.dtype, GgmlType::I32);
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
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

#[test]
fn packed_router_e8p32_strict_scope_is_exact() {
    let qualified = |device, hidden, experts, dtype, tokens| {
        packed_router_e8p32_strict_scope_qualified(device, hidden, experts, dtype, tokens)
    };
    for &tokens in &PACKED_ROUTER_E8P32_STRICT_TOKEN_COUNTS {
        assert!(tokens <= MAX_PACKED_TOKENS);
    }
    for tokens in 1..=MAX_PACKED_TOKENS + 1 {
        assert_eq!(
            qualified(
                PACKED_ROUTER_E8P32_STRICT_DEVICE,
                PACKED_ROUTER_E8P32_STRICT_HIDDEN,
                PACKED_ROUTER_E8P32_STRICT_EXPERTS,
                GgmlType::F32,
                tokens,
            ),
            PACKED_ROUTER_E8P32_STRICT_TOKEN_COUNTS.contains(&tokens),
            "strict E8P32 token scope at N={tokens}",
        );
    }
    assert!(!qualified(
        "Apple M3 Max",
        PACKED_ROUTER_E8P32_STRICT_HIDDEN,
        PACKED_ROUTER_E8P32_STRICT_EXPERTS,
        GgmlType::F32,
        PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS,
    ));
    assert!(!qualified(
        PACKED_ROUTER_E8P32_STRICT_DEVICE,
        PACKED_ROUTER_E8P32_STRICT_HIDDEN / 2,
        PACKED_ROUTER_E8P32_STRICT_EXPERTS,
        GgmlType::F32,
        PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS,
    ));
    assert!(!qualified(
        PACKED_ROUTER_E8P32_STRICT_DEVICE,
        PACKED_ROUTER_E8P32_STRICT_HIDDEN,
        160,
        GgmlType::F32,
        PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS,
    ));
    assert!(!qualified(
        PACKED_ROUTER_E8P32_STRICT_DEVICE,
        PACKED_ROUTER_E8P32_STRICT_HIDDEN,
        PACKED_ROUTER_E8P32_STRICT_EXPERTS,
        GgmlType::F16,
        PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS,
    ));
}

#[test]
fn packed_iq4_down_m128_n16_scope_is_exact_and_rollbackable() {
    let exact =
        Qwen4ExpMoeMetalGeometry::from_config(&Qwen4ExpConfig::flash_next_reference()).unwrap();
    assert_eq!(exact.hidden_size, PACKED_IQ4_DOWN_M128_N16_HIDDEN);
    assert_eq!(
        exact.routed_intermediate_size,
        PACKED_IQ4_DOWN_M128_N16_ROUTED
    );
    assert_eq!(exact.expert_count, PACKED_IQ4_DOWN_M128_N16_EXPERTS);
    assert_eq!(exact.experts_per_token, PACKED_IQ4_DOWN_M128_N16_TOP_K);
    let qualified = |device, geometry, dtype, tokens| {
        packed_iq4_down_m128_n16_scope_qualified(device, geometry, dtype, tokens)
    };
    for tokens in 1..=MAX_PACKED_TOKENS + 1 {
        assert_eq!(
            qualified(
                PACKED_IQ4_DOWN_M128_N16_DEVICE,
                exact,
                GgmlType::IQ4_NL,
                tokens,
            ),
            PACKED_IQ4_DOWN_M128_N16_TOKEN_COUNTS.contains(&tokens),
            "tokens={tokens}",
        );
    }

    for (label, device, geometry, dtype, tokens) in [
        (
            "device",
            "Apple M3 Max",
            exact,
            GgmlType::IQ4_NL,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
        (
            "hidden",
            PACKED_IQ4_DOWN_M128_N16_DEVICE,
            Qwen4ExpMoeMetalGeometry::new(
                2_304,
                exact.expert_count,
                exact.experts_per_token,
                exact.routed_intermediate_size,
                exact.shared_intermediate_size,
            )
            .unwrap(),
            GgmlType::IQ4_NL,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
        (
            "routed",
            PACKED_IQ4_DOWN_M128_N16_DEVICE,
            Qwen4ExpMoeMetalGeometry::new(
                exact.hidden_size,
                exact.expert_count,
                exact.experts_per_token,
                608,
                exact.shared_intermediate_size,
            )
            .unwrap(),
            GgmlType::IQ4_NL,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
        (
            "experts",
            PACKED_IQ4_DOWN_M128_N16_DEVICE,
            Qwen4ExpMoeMetalGeometry::new(
                exact.hidden_size,
                511,
                exact.experts_per_token,
                exact.routed_intermediate_size,
                exact.shared_intermediate_size,
            )
            .unwrap(),
            GgmlType::IQ4_NL,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
        (
            "top-k",
            PACKED_IQ4_DOWN_M128_N16_DEVICE,
            Qwen4ExpMoeMetalGeometry::new(
                exact.hidden_size,
                exact.expert_count,
                9,
                exact.routed_intermediate_size,
                exact.shared_intermediate_size,
            )
            .unwrap(),
            GgmlType::IQ4_NL,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
        (
            "dtype",
            PACKED_IQ4_DOWN_M128_N16_DEVICE,
            exact,
            GgmlType::Q8_0,
            PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
        ),
    ] {
        assert!(!qualified(device, geometry, dtype, tokens), "{label}");
    }

    let Some(ctx) = packed_test_context() else {
        return;
    };
    for &tokens in &PACKED_IQ4_DOWN_M128_N16_TOKEN_COUNTS {
        with_qwen4exp_moe_iq4_down_m128_n16_override(false, || {
            assert!(!packed_iq4_down_m128_n16_qualified(
                &ctx,
                exact,
                GgmlType::IQ4_NL,
                tokens,
            ));
        });
        with_qwen4exp_moe_iq4_down_m128_n16_override(true, || {
            assert_eq!(
                packed_iq4_down_m128_n16_qualified(&ctx, exact, GgmlType::IQ4_NL, tokens,),
                ctx.device.name().to_string() == PACKED_IQ4_DOWN_M128_N16_DEVICE,
            );
        });
    }
}

#[test]
fn packed_router_e8p32_strict_matches_generic_route_bits() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    if ctx.device.name().to_string() != PACKED_ROUTER_E8P32_STRICT_DEVICE {
        eprintln!(
            "strict E8P32 Qwen router differential skipped on {}",
            ctx.device.name()
        );
        return;
    }

    check_packed_router_e8p32_bits(ctx, &PACKED_ROUTER_E8P32_STRICT_TOKEN_COUNTS);
}

#[test]
#[ignore = "production lease; exact N1024 existing packed-router differential"]
fn packed_router_n1024_component_qualification() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required");
    assert_eq!(
        ctx.device.name().to_string(),
        PACKED_ROUTER_E8P32_STRICT_DEVICE
    );
    check_packed_router_e8p32_bits(ctx, &[1024]);
}

fn check_packed_router_e8p32_bits(ctx: MetalContext, token_counts: &[usize]) {
    const TOP_K: usize = 10;
    let geometry = Qwen4ExpMoeMetalGeometry::new(
        PACKED_ROUTER_E8P32_STRICT_HIDDEN,
        PACKED_ROUTER_E8P32_STRICT_EXPERTS,
        TOP_K,
        640,
        640,
    )
    .unwrap();
    let router_values = (0..geometry.hidden_size * geometry.expert_count)
        .map(|index| ((index * 37 + 11) % 509) as f32 * 0.000_4 - 0.101_6)
        .collect::<Vec<_>>();
    let router = weight_f32(
        &ctx,
        &router_values,
        vec![geometry.hidden_size as u64, geometry.expert_count as u64],
    );
    let shared_router_values = (0..geometry.hidden_size)
        .map(|index| ((index * 29 + 7) % 257) as f32 * 0.000_5 - 0.064)
        .collect::<Vec<_>>();
    let shared_router = weight_f32(
        &ctx,
        &shared_router_values,
        vec![geometry.hidden_size as u64],
    );

    for &tokens in token_counts {
        let input_values = (0..tokens * geometry.hidden_size)
            .map(|index| {
                let token = index / geometry.hidden_size;
                let lane = index % geometry.hidden_size;
                ((token * 43 + lane * 17 + 13) % 521) as f32 * 0.000_3 - 0.078
            })
            .collect::<Vec<_>>();
        let input = tensor_f32(
            &ctx,
            &input_values,
            vec![geometry.hidden_size as u64, tokens as u64],
        );
        let baseline_logits =
            MetalTensor::zeros_f32(&ctx, vec![geometry.expert_count as u64, tokens as u64])
                .unwrap();
        let candidate_logits =
            MetalTensor::zeros_f32(&ctx, vec![geometry.expert_count as u64, tokens as u64])
                .unwrap();
        let baseline_ids = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let candidate_ids =
            MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let baseline_weights =
            MetalTensor::zeros_f32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let candidate_weights =
            MetalTensor::zeros_f32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let baseline_shared = MetalTensor::zeros_f32(&ctx, vec![tokens as u64]).unwrap();
        let candidate_shared = MetalTensor::zeros_f32(&ctx, vec![tokens as u64]).unwrap();
        let baseline_counts =
            MetalTensor::zeros_i32(&ctx, vec![geometry.expert_count as u64]).unwrap();
        let candidate_counts =
            MetalTensor::zeros_i32(&ctx, vec![geometry.expert_count as u64]).unwrap();
        let baseline_slots =
            MetalTensor::zeros_i32(&ctx, vec![tokens as u64, geometry.expert_count as u64])
                .unwrap();
        let candidate_slots =
            MetalTensor::zeros_i32(&ctx, vec![tokens as u64, geometry.expert_count as u64])
                .unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        with_qwen4exp_packed_router_e8p32_strict_override(false, || {
            encode_packed_router_projection(
                &ctx,
                &encoder,
                &router,
                &input,
                &baseline_logits,
                geometry,
                tokens,
            )
        })
        .unwrap();
        with_qwen4exp_packed_router_e8p32_strict_override(true, || {
            encode_packed_router_projection(
                &ctx,
                &encoder,
                &router,
                &input,
                &candidate_logits,
                geometry,
                tokens,
            )
        })
        .unwrap();
        for (logits, ids, weights, shared) in [
            (
                &baseline_logits,
                &baseline_ids,
                &baseline_weights,
                &baseline_shared,
            ),
            (
                &candidate_logits,
                &candidate_ids,
                &candidate_weights,
                &candidate_shared,
            ),
        ] {
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                &encoder,
                logits,
                &shared_router,
                &input,
                ids,
                weights,
                shared,
                geometry.expert_count,
                TOP_K,
                geometry.hidden_size,
                tokens,
            )
            .unwrap();
        }
        for (ids, counts, slots) in [
            (&baseline_ids, &baseline_counts, &baseline_slots),
            (&candidate_ids, &candidate_counts, &candidate_slots),
        ] {
            encode_moe_route_bucket_slots_f32(
                &ctx,
                &encoder,
                ids,
                counts,
                slots,
                geometry.expert_count,
                tokens,
                TOP_K,
            )
            .unwrap();
        }
        let census = crate::metal::dispatch_census_take();
        assert_eq!(
            census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>(),
            [
                "kernel_mat_mat_f32_f32",
                "kernel_mat_mat_f32_f32_router_e8p32_strict",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_moe_route_bucket_slots_f32",
                "kernel_moe_route_bucket_slots_f32",
            ],
            "strict E8P32 Qwen route N={tokens}: {census:#?}"
        );
        assert_eq!(census[0].grid_width, geometry.expert_count as u64);
        assert_eq!(census[1].grid_width, (geometry.expert_count / 8) as u64);
        assert_eq!(census[0].grid_height, tokens.div_ceil(32) as u64);
        assert_eq!(census[1].grid_height, tokens.div_ceil(32) as u64);
        assert_eq!(census[0].threads_width, 32);
        assert_eq!(census[1].threads_width, 32);
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        assert_bits_eq(
            &format!("strict E8P32 Qwen router logits N={tokens}"),
            &read_f32(&candidate_logits),
            &read_f32(&baseline_logits),
        );
        assert_eq!(
            read_i32(&candidate_ids),
            read_i32(&baseline_ids),
            "strict E8P32 Qwen top-k IDs N={tokens}"
        );
        assert_bits_eq(
            &format!("strict E8P32 Qwen top-k weights N={tokens}"),
            &read_f32(&candidate_weights),
            &read_f32(&baseline_weights),
        );
        assert_bits_eq(
            &format!("strict E8P32 Qwen shared scale N={tokens}"),
            &read_f32(&candidate_shared),
            &read_f32(&baseline_shared),
        );
        assert_eq!(
            read_i32(&candidate_counts),
            read_i32(&baseline_counts),
            "strict E8P32 Qwen route counts N={tokens}"
        );
        assert_eq!(
            read_i32(&candidate_slots),
            read_i32(&baseline_slots),
            "strict E8P32 Qwen route slots N={tokens}"
        );
    }
}

const ACTIVE_F32_SENTINEL: u32 = 0x7fc0_1234;
const GUARD_F32_SENTINEL: u32 = 0x7fc0_5678;
const ACTIVE_I32_SENTINEL: i32 = 0x1234_5678;
const GUARD_I32_SENTINEL: i32 = 0x2345_6789;

fn fill_f32_bits(tensor: &MetalTensor, bits: u32) {
    assert_eq!(tensor.dtype, GgmlType::F32);
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u32>();
        std::slice::from_raw_parts_mut(destination, tensor.n_elements() as usize).fill(bits);
    }
}

fn fill_i32(tensor: &MetalTensor, value: i32) {
    assert_eq!(tensor.dtype, GgmlType::I32);
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts_mut(destination, tensor.n_elements() as usize).fill(value);
    }
}

fn seed_packed_scratch(scratch: &Qwen4ExpMoePackedMotorScratch, tokens: usize) {
    let g = scratch.geometry;
    for (tensor, width) in [
        (&scratch.router_logits, g.expert_count),
        (&scratch.topk_weights, g.experts_per_token),
        (&scratch.shared_scale, 1),
        (
            &scratch.routed_inner,
            g.experts_per_token * g.routed_intermediate_size,
        ),
        (
            &scratch.routed_expert_output,
            g.experts_per_token * g.hidden_size,
        ),
        (&scratch.shared_gate_projection, g.shared_intermediate_size),
        (&scratch.shared_up_projection, g.shared_intermediate_size),
        (&scratch.shared_inner, g.shared_intermediate_size),
        (&scratch.shared_output, g.hidden_size),
        (&scratch.output, g.hidden_size),
    ] {
        fill_f32_bits(tensor, GUARD_F32_SENTINEL);
        let mut values = read_f32(tensor);
        values[..tokens * width]
            .iter_mut()
            .for_each(|value| *value = f32::from_bits(ACTIVE_F32_SENTINEL));
        unsafe {
            let destination = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
    }
    fill_i32(&scratch.topk_ids, GUARD_I32_SENTINEL);
    let topk_ids = unsafe {
        let destination = scratch
            .topk_ids
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(scratch.topk_ids.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts_mut(destination, scratch.topk_ids.n_elements() as usize)
    };
    topk_ids[..tokens * g.experts_per_token].fill(ACTIVE_I32_SENTINEL);
    fill_i32(&scratch.route_counts, ACTIVE_I32_SENTINEL);
    fill_i32(&scratch.route_slots, GUARD_I32_SENTINEL);
    let route_slots = unsafe {
        let destination = scratch
            .route_slots
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(scratch.route_slots.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts_mut(destination, scratch.route_slots.n_elements() as usize)
    };
    route_slots[..tokens * g.expert_count].fill(ACTIVE_I32_SENTINEL);
}

fn assert_packed_scratch_guards(scratch: &Qwen4ExpMoePackedMotorScratch, tokens: usize) {
    let g = scratch.geometry;
    for (tensor, width) in [
        (&scratch.router_logits, g.expert_count),
        (&scratch.topk_weights, g.experts_per_token),
        (&scratch.shared_scale, 1),
        (
            &scratch.routed_inner,
            g.experts_per_token * g.routed_intermediate_size,
        ),
        (
            &scratch.routed_expert_output,
            g.experts_per_token * g.hidden_size,
        ),
        (&scratch.shared_gate_projection, g.shared_intermediate_size),
        (&scratch.shared_up_projection, g.shared_intermediate_size),
        (&scratch.shared_inner, g.shared_intermediate_size),
        (&scratch.shared_output, g.hidden_size),
        (&scratch.output, g.hidden_size),
    ] {
        assert!(
            read_f32(tensor)[tokens * width..]
                .iter()
                .all(|value| value.to_bits() == GUARD_F32_SENTINEL)
        );
    }
    assert!(
        read_i32(&scratch.topk_ids)[tokens * g.experts_per_token..]
            .iter()
            .all(|&value| value == GUARD_I32_SENTINEL)
    );
    assert!(
        read_i32(&scratch.route_slots)[tokens * g.expert_count..]
            .iter()
            .all(|&value| value == GUARD_I32_SENTINEL)
    );
}

fn assert_packed_scratch_bits_eq(
    label: &str,
    actual: &Qwen4ExpMoePackedMotorScratch,
    expected: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
) {
    let actual = actual.views(tokens).unwrap();
    let expected = expected.views(tokens).unwrap();
    for (name, actual, expected) in [
        (
            "router logits",
            &actual.router_logits,
            &expected.router_logits,
        ),
        (
            "top-k weights",
            &actual.topk_weights,
            &expected.topk_weights,
        ),
        ("shared scale", &actual.shared_scale, &expected.shared_scale),
        ("routed inner", &actual.routed_inner, &expected.routed_inner),
        (
            "routed expert output",
            &actual.routed_expert_output,
            &expected.routed_expert_output,
        ),
        (
            "shared gate projection",
            &actual.shared_gate_projection,
            &expected.shared_gate_projection,
        ),
        (
            "shared up projection",
            &actual.shared_up_projection,
            &expected.shared_up_projection,
        ),
        ("shared inner", &actual.shared_inner, &expected.shared_inner),
        (
            "shared output",
            &actual.shared_output,
            &expected.shared_output,
        ),
        ("output", &actual.output, &expected.output),
    ] {
        assert_bits_eq(
            &format!("{label} {name}"),
            &read_f32(actual),
            &read_f32(expected),
        );
    }
    for (name, actual, expected) in [
        ("top-k IDs", &actual.topk_ids, &expected.topk_ids),
        ("route counts", &actual.route_counts, &expected.route_counts),
        ("route slots", &actual.route_slots, &expected.route_slots),
    ] {
        assert_eq!(read_i32(actual), read_i32(expected), "{label} {name}");
    }
}

fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}]: expected {expected:?}, got {actual:?}"
        );
    }
}

fn assert_tokenwise_similarity(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    width: usize,
    tokens: usize,
    max_abs: f32,
    cosine: f64,
) {
    assert_eq!(actual.len(), width * tokens, "{label} actual shape");
    assert_eq!(expected.len(), width * tokens, "{label} expected shape");
    for token in 0..tokens {
        let start = token * width;
        assert_similarity(
            &format!("{label} token={token}"),
            &actual[start..start + width],
            &expected[start..start + width],
            max_abs,
            cosine,
        );
    }
}

struct SerialPackedMoeTrace {
    router_logits: Vec<f32>,
    topk_ids: Vec<i32>,
    topk_weights: Vec<f32>,
    shared_scale: Vec<f32>,
    routed_inner: Vec<f32>,
    routed_expert_output: Vec<f32>,
    shared_inner: Vec<f32>,
    shared_output: Vec<f32>,
    output: Vec<f32>,
}

fn serial_packed_moe_trace(
    ctx: &MetalContext,
    geometry: Qwen4ExpMoeMetalGeometry,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    inputs: &[f32],
    tokens: usize,
) -> SerialPackedMoeTrace {
    let mut workspace = Qwen4ExpMoeMetalWorkspace::new(ctx, geometry).unwrap();
    let mut trace = SerialPackedMoeTrace {
        router_logits: Vec::with_capacity(tokens * geometry.expert_count),
        topk_ids: Vec::with_capacity(tokens * geometry.experts_per_token),
        topk_weights: Vec::with_capacity(tokens * geometry.experts_per_token),
        shared_scale: Vec::with_capacity(tokens),
        routed_inner: Vec::with_capacity(
            tokens * geometry.experts_per_token * geometry.routed_intermediate_size,
        ),
        routed_expert_output: Vec::with_capacity(
            tokens * geometry.experts_per_token * geometry.hidden_size,
        ),
        shared_inner: Vec::with_capacity(tokens * geometry.shared_intermediate_size),
        shared_output: Vec::with_capacity(tokens * geometry.hidden_size),
        output: Vec::with_capacity(tokens * geometry.hidden_size),
    };
    for token in 0..tokens {
        let start = token * geometry.hidden_size;
        let input = tensor_f32(
            ctx,
            &inputs[start..start + geometry.hidden_size],
            vec![geometry.hidden_size as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_moe(ctx, &encoder, &input, weights, &mut workspace).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        trace
            .router_logits
            .extend(read_f32(&workspace.router_logits));
        trace.topk_ids.extend(read_i32(&workspace.topk_ids));
        trace.topk_weights.extend(read_f32(&workspace.topk_weights));
        trace.shared_scale.extend(read_f32(&workspace.shared_gate));
        trace.routed_inner.extend(read_f32(&workspace.routed_inner));
        trace
            .routed_expert_output
            .extend(read_f32(&workspace.routed_expert_output));
        trace.shared_inner.extend(read_f32(&workspace.shared_inner));
        trace
            .shared_output
            .extend(read_f32(&workspace.shared_output));
        trace.output.extend(read_f32(&workspace.output));
    }
    trace
}

#[test]
fn packed_router_supports_512_experts_with_stable_ties() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const HIDDEN: usize = 256;
    let shared_values = (0..HIDDEN)
        .map(|index| ((index * 17 + 3) % 41) as f32 * 0.002 - 0.04)
        .collect::<Vec<_>>();
    let shared_weight = weight_f32(&ctx, &shared_values, vec![HIDDEN as u64]);

    for tokens in [2_usize, 8, 33] {
        let logits = (0..tokens * EXPERTS)
            .map(|index| {
                let token = index / EXPERTS;
                let expert = index % EXPERTS;
                let target = 300 + (token * 17) % 180;
                -(expert.abs_diff(target) as f32)
            })
            .collect::<Vec<_>>();
        let input = (0..tokens * HIDDEN)
            .map(|index| {
                let token = index / HIDDEN;
                let lane = index % HIDDEN;
                ((lane * 13 + token * 19 + 5) % 97) as f32 * 0.001 - 0.048
            })
            .collect::<Vec<_>>();
        let logits_gpu = tensor_f32(&ctx, &logits, vec![EXPERTS as u64, tokens as u64]);
        let input_gpu = tensor_f32(&ctx, &input, vec![HIDDEN as u64, tokens as u64]);
        let ids_gpu = MetalTensor::zeros_i32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let weights_gpu = MetalTensor::zeros_f32(&ctx, vec![TOP_K as u64, tokens as u64]).unwrap();
        let shared_gpu = MetalTensor::zeros_f32(&ctx, vec![tokens as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        crate::metal::encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            &encoder,
            &logits_gpu,
            &shared_weight,
            &input_gpu,
            &ids_gpu,
            &weights_gpu,
            &shared_gpu,
            EXPERTS,
            TOP_K,
            HIDDEN,
            tokens,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 1);
        assert_eq!(
            census[0].kernel,
            "kernel_topk_logits_softmax_dot_sigmoid_packed_f32"
        );
        assert_eq!(census[0].grid_height, tokens as u64);
        assert_eq!(census[0].threads_width, EXPERTS as u64);
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let actual_ids = read_i32(&ids_gpu);
        let actual_weights = read_f32(&weights_gpu);
        let actual_shared = read_f32(&shared_gpu);
        for token in 0..tokens {
            let logits_row = &logits[token * EXPERTS..(token + 1) * EXPERTS];
            let (expected_ids, expected_weights) = stable_topk(logits_row, TOP_K);
            assert_eq!(
                &actual_ids[token * TOP_K..(token + 1) * TOP_K],
                expected_ids
            );
            assert!(expected_ids.iter().all(|&expert| expert >= 256));
            assert_close(
                &format!("packed 512-expert route weights N={tokens} token={token}"),
                &actual_weights[token * TOP_K..(token + 1) * TOP_K],
                &expected_weights,
                1e-7,
                1e-7,
            );
            let shared_dot = shared_values
                .iter()
                .zip(&input[token * HIDDEN..(token + 1) * HIDDEN])
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let expected_shared = 1.0 / (1.0 + (-shared_dot).exp());
            assert_close(
                &format!("packed 512-expert shared gate N={tokens} token={token}"),
                &[actual_shared[token]],
                &[expected_shared],
                2e-6,
                2e-6,
            );
        }
    }
}

#[test]
fn grouped_iq4_xs_swiglu_matches_slotwise_512_expert_execution() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const HIDDEN: usize = 512;
    const ROUTED: usize = 96;
    let gate_bytes = synthetic_iq4_xs_bank(HIDDEN, ROUTED, EXPERTS, 173);
    let up_bytes = synthetic_iq4_xs_bank(HIDDEN, ROUTED, EXPERTS, 197);
    let gate = weight_bytes(
        &ctx,
        &gate_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ4_XS,
    );
    let up = weight_bytes(
        &ctx,
        &up_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ4_XS,
    );

    let aligned_input = tensor_f32(&ctx, &[0.0; HIDDEN], vec![HIDDEN as u64]);
    let mut padded_input = vec![0.0_f32; HIDDEN + 1];
    padded_input[1..].copy_from_slice(&read_f32(&aligned_input));
    let padded_input = tensor_f32(&ctx, &padded_input, vec![(HIDDEN + 1) as u64]);
    let misaligned_input = padded_input.view_subrange(1, vec![HIDDEN as u64]);
    let validation_counts = MetalTensor::zeros_i32(&ctx, vec![EXPERTS as u64]).unwrap();
    let validation_buckets = MetalTensor::zeros_i32(&ctx, vec![1, EXPERTS as u64]).unwrap();
    let validation_inner = MetalTensor::zeros_f32(&ctx, vec![ROUTED as u64, TOP_K as u64]).unwrap();
    let validation_command = ctx.queue.commandBuffer().unwrap();
    let validation_encoder = KernelEncoder::begin(&validation_command);
    let alignment_error = encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
        &ctx,
        &validation_encoder,
        &gate,
        &up,
        &misaligned_input,
        &validation_counts,
        &validation_buckets,
        &validation_inner,
        HIDDEN,
        ROUTED,
        EXPERTS,
        TOP_K,
        1,
    )
    .unwrap_err()
    .to_string();
    assert!(alignment_error.contains("16-byte aligned"));
    let f32_counts = MetalTensor::zeros_f32(&ctx, vec![EXPERTS as u64]).unwrap();
    let metadata_error = encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
        &ctx,
        &validation_encoder,
        &gate,
        &up,
        &aligned_input,
        &f32_counts,
        &validation_buckets,
        &validation_inner,
        HIDDEN,
        ROUTED,
        EXPERTS,
        TOP_K,
        1,
    )
    .unwrap_err()
    .to_string();
    assert!(metadata_error.contains("expected [I32]"));
    validation_encoder.end();

    for (label, count, slot) in [
        ("oversized count", 2_i32, 0_i32),
        ("negative slot", 1, -1),
        ("high slot", 1, TOP_K as i32),
    ] {
        let mut counts = vec![0_i32; EXPERTS];
        counts[0] = count;
        let mut buckets = vec![0_i32; EXPERTS];
        buckets[0] = slot;
        let counts = tensor_i32(&ctx, &counts, vec![EXPERTS as u64]);
        let buckets = tensor_i32(&ctx, &buckets, vec![1, EXPERTS as u64]);
        let inner = MetalTensor::zeros_f32(&ctx, vec![ROUTED as u64, TOP_K as u64]).unwrap();
        fill_f32_bits(&inner, GUARD_F32_SENTINEL);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
            &ctx,
            &encoder,
            &gate,
            &up,
            &aligned_input,
            &counts,
            &buckets,
            &inner,
            HIDDEN,
            ROUTED,
            EXPERTS,
            TOP_K,
            1,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(
            command.status(),
            MTLCommandBufferStatus::Completed,
            "{label}"
        );
        assert!(command.error().is_none(), "{label}: {:?}", command.error());
        assert!(
            read_f32(&inner)
                .iter()
                .all(|value| value.to_bits() == GUARD_F32_SENTINEL),
            "{label} wrote output"
        );
    }

    for tokens in [1_usize, 2, 8, 16, 33] {
        let concentrated = [511_i32, 300, 301, 302, 303, 304, 305, 306, 307, 308];
        let mut topk_ids = if tokens == 33 {
            (0..tokens).flat_map(|_| concentrated).collect::<Vec<_>>()
        } else {
            (0..tokens * TOP_K)
                .map(|slot| {
                    let token = slot / TOP_K;
                    let route = slot % TOP_K;
                    ((257 + token * 37 + route * 53) % EXPERTS) as i32
                })
                .collect::<Vec<_>>()
        };
        topk_ids[0] = 511;
        let inputs = (0..tokens * HIDDEN)
            .map(|index| {
                let token = index / HIDDEN;
                let lane = index % HIDDEN;
                ((token * 41 + lane * 19 + 13) % 127) as f32 * 0.000_8 - 0.05
            })
            .collect::<Vec<_>>();
        let ids_gpu = tensor_i32(&ctx, &topk_ids, vec![TOP_K as u64, tokens as u64]);
        let input_gpu = tensor_f32(&ctx, &inputs, vec![HIDDEN as u64, tokens as u64]);
        let counts_gpu = MetalTensor::zeros_i32(&ctx, vec![EXPERTS as u64]).unwrap();
        let buckets_gpu =
            MetalTensor::zeros_i32(&ctx, vec![tokens as u64, EXPERTS as u64]).unwrap();
        let serial_inner =
            MetalTensor::zeros_f32(&ctx, vec![ROUTED as u64, TOP_K as u64, tokens as u64]).unwrap();
        let grouped_inner =
            MetalTensor::zeros_f32(&ctx, vec![ROUTED as u64, TOP_K as u64, tokens as u64]).unwrap();
        fill_i32(&counts_gpu, ACTIVE_I32_SENTINEL);
        fill_i32(&buckets_gpu, GUARD_I32_SENTINEL);
        fill_f32_bits(&serial_inner, 0x7fc0_3111);
        fill_f32_bits(&grouped_inner, 0x7fc0_3222);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        for token in 0..tokens {
            let input_row = input_gpu.view_subrange((token * HIDDEN) as u64, vec![HIDDEN as u64]);
            let ids_row = ids_gpu.view_subrange((token * TOP_K) as u64, vec![TOP_K as u64]);
            let inner_row = serial_inner.view_subrange(
                (token * TOP_K * ROUTED) as u64,
                vec![ROUTED as u64, TOP_K as u64],
            );
            encode_moe_swiglu_iq4_xs_f32(
                &ctx, &encoder, &gate, &up, &input_row, &ids_row, &inner_row, HIDDEN, ROUTED,
                EXPERTS, TOP_K,
            )
            .unwrap();
        }
        crate::metal::encode_moe_route_bucket_slots_f32(
            &ctx,
            &encoder,
            &ids_gpu,
            &counts_gpu,
            &buckets_gpu,
            EXPERTS,
            tokens,
            TOP_K,
        )
        .unwrap();
        encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
            &ctx,
            &encoder,
            &gate,
            &up,
            &input_gpu,
            &counts_gpu,
            &buckets_gpu,
            &grouped_inner,
            HIDDEN,
            ROUTED,
            EXPERTS,
            TOP_K,
            tokens,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(
            census[census.len() - 2].kernel,
            "kernel_moe_route_bucket_slots_f32"
        );
        assert_eq!(
            census[census.len() - 1].kernel,
            "kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let mut expected_counts = vec![0_i32; EXPERTS];
        let mut expected_buckets = vec![0_i32; EXPERTS * tokens];
        for (slot, &expert) in topk_ids.iter().enumerate() {
            let expert = expert as usize;
            let count = expected_counts[expert] as usize;
            expected_buckets[expert * tokens + count] = slot as i32;
            expected_counts[expert] += 1;
        }
        let actual_counts = read_i32(&counts_gpu);
        let actual_buckets = read_i32(&buckets_gpu);
        assert_eq!(actual_counts, expected_counts);
        assert_eq!(actual_counts.iter().sum::<i32>(), (tokens * TOP_K) as i32);
        for expert in 0..EXPERTS {
            let count = actual_counts[expert] as usize;
            assert_eq!(
                &actual_buckets[expert * tokens..expert * tokens + count],
                &expected_buckets[expert * tokens..expert * tokens + count]
            );
        }
        if tokens == 33 {
            for (route, &expert) in concentrated.iter().enumerate() {
                let expert = expert as usize;
                assert_eq!(actual_counts[expert], 33);
                assert_eq!(
                    actual_buckets[expert * tokens + 32],
                    (32 * TOP_K + route) as i32
                );
            }
        }
        assert_similarity(
            &format!("grouped IQ4_XS SwiGLU N={tokens}"),
            &read_f32(&grouped_inner),
            &read_f32(&serial_inner),
            3e-6,
            0.999_999_8,
        );
    }
}

#[test]
fn grouped_iq4_nl_down_matches_slotwise_512_expert_execution() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const N_IN: usize = 64;
    const N_OUT: usize = 256;
    let down_bytes = synthetic_iq4_nl_bank(N_IN, N_OUT, EXPERTS, 211);
    let weight = weight_bytes(
        &ctx,
        &down_bytes,
        vec![N_IN as u64, N_OUT as u64, EXPERTS as u64],
        GgmlType::IQ4_NL,
    );

    for tokens in [1_usize, 2, 8, 16, 33] {
        let concentrated = [511_i32, 300, 301, 302, 303, 304, 305, 306, 307, 308];
        let mut topk_ids = if tokens == 33 {
            (0..tokens).flat_map(|_| concentrated).collect::<Vec<_>>()
        } else {
            (0..tokens * TOP_K)
                .map(|slot| {
                    let token = slot / TOP_K;
                    let route = slot % TOP_K;
                    ((257 + token * 37 + route * 53) % EXPERTS) as i32
                })
                .collect::<Vec<_>>()
        };
        topk_ids[0] = 511;
        let inner = (0..tokens * TOP_K * N_IN)
            .map(|index| {
                let slot = index / N_IN;
                let lane = index % N_IN;
                ((slot * 29 + lane * 17 + 7) % 113) as f32 * 0.001 - 0.056
            })
            .collect::<Vec<_>>();
        let ids_gpu = tensor_i32(&ctx, &topk_ids, vec![TOP_K as u64, tokens as u64]);
        let inner_gpu = tensor_f32(&ctx, &inner, vec![N_IN as u64, TOP_K as u64, tokens as u64]);
        let counts_gpu = MetalTensor::zeros_i32(&ctx, vec![EXPERTS as u64]).unwrap();
        let buckets_gpu =
            MetalTensor::zeros_i32(&ctx, vec![tokens as u64, EXPERTS as u64]).unwrap();
        let serial_output =
            MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, TOP_K as u64, tokens as u64]).unwrap();
        let grouped_output =
            MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, TOP_K as u64, tokens as u64]).unwrap();
        fill_i32(&counts_gpu, ACTIVE_I32_SENTINEL);
        fill_i32(&buckets_gpu, GUARD_I32_SENTINEL);
        fill_f32_bits(&serial_output, 0x7fc0_1111);
        fill_f32_bits(&grouped_output, 0x7fc0_2222);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        for token in 0..tokens {
            let slot_start = token * TOP_K;
            let inner_row = inner_gpu
                .view_subrange((slot_start * N_IN) as u64, vec![N_IN as u64, TOP_K as u64]);
            let ids_row = ids_gpu.view_subrange(slot_start as u64, vec![TOP_K as u64]);
            let output_row = serial_output.view_subrange(
                (slot_start * N_OUT) as u64,
                vec![N_OUT as u64, TOP_K as u64],
            );
            encode_moe_down_iq4_nl_f32(
                &ctx,
                &encoder,
                &weight,
                &inner_row,
                &ids_row,
                &output_row,
                N_IN,
                N_OUT,
                EXPERTS,
                TOP_K,
            )
            .unwrap();
        }
        crate::metal::encode_moe_route_bucket_slots_f32(
            &ctx,
            &encoder,
            &ids_gpu,
            &counts_gpu,
            &buckets_gpu,
            EXPERTS,
            tokens,
            TOP_K,
        )
        .unwrap();
        crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots(
            &ctx,
            &encoder,
            &weight,
            &inner_gpu,
            &counts_gpu,
            &buckets_gpu,
            &grouped_output,
            N_IN,
            N_OUT,
            EXPERTS,
            tokens,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(
            census[census.len() - 2].kernel,
            "kernel_moe_route_bucket_slots_f32"
        );
        assert_eq!(
            census[census.len() - 1].kernel,
            "kernel_moe_down_iq4_nl_f32_grouped_slots"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let mut expected_counts = vec![0_i32; EXPERTS];
        let mut expected_buckets = vec![0_i32; EXPERTS * tokens];
        for (slot, &expert) in topk_ids.iter().enumerate() {
            let expert = expert as usize;
            let count = expected_counts[expert] as usize;
            expected_buckets[expert * tokens + count] = slot as i32;
            expected_counts[expert] += 1;
        }
        let actual_counts = read_i32(&counts_gpu);
        let actual_buckets = read_i32(&buckets_gpu);
        assert_eq!(actual_counts, expected_counts);
        assert_eq!(actual_counts.iter().sum::<i32>(), (tokens * TOP_K) as i32);
        for expert in 0..EXPERTS {
            let count = actual_counts[expert] as usize;
            assert_eq!(
                &actual_buckets[expert * tokens..expert * tokens + count],
                &expected_buckets[expert * tokens..expert * tokens + count]
            );
        }
        if tokens == 33 {
            for (route, &expert) in concentrated.iter().enumerate() {
                let expert = expert as usize;
                assert_eq!(actual_counts[expert], 33);
                assert_eq!(
                    actual_buckets[expert * tokens + 32],
                    (32 * TOP_K + route) as i32
                );
            }
        }
        assert_similarity(
            &format!("grouped IQ4_NL down N={tokens}"),
            &read_f32(&grouped_output),
            &read_f32(&serial_output),
            1e-5,
            0.999_999_9,
        );
    }
}

#[test]
fn grouped_iq4_nl_down_m128_n16_is_bit_exact_and_guarded() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    require_pipeline_capacity(
        &ctx,
        "kernel_moe_down_iq4_nl_f32_grouped_slots_m128_n16",
        128,
        9_216,
    )
    .unwrap();

    fn run_case(
        ctx: &MetalContext,
        label: &str,
        n_in: usize,
        n_out: usize,
        n_expert: usize,
        token_capacity: Option<usize>,
        assignments: &[usize],
    ) {
        const GUARD_ELEMENTS: usize = 257;
        const BASELINE_ACTIVE: u32 = 0x7fc0_4111;
        const CANDIDATE_ACTIVE: u32 = 0x7fc0_4222;
        let slot_count = assignments.len();
        let n_tokens = token_capacity.unwrap_or(slot_count);
        assert!(n_in.is_multiple_of(32));
        assert!(n_tokens > 0);
        assert!(slot_count.is_multiple_of(n_tokens));
        assert!((1..=16).contains(&(slot_count / n_tokens)));
        assert!(assignments.iter().all(|&expert| expert < n_expert));

        let mut counts = vec![0_i32; n_expert];
        let mut slots = vec![GUARD_I32_SENTINEL; n_expert * n_tokens];
        for (slot, &expert) in assignments.iter().enumerate() {
            let count = counts[expert] as usize;
            slots[expert * n_tokens + count] = slot as i32;
            counts[expert] += 1;
        }
        assert!(counts.iter().all(|&count| count <= n_tokens as i32));
        assert_eq!(counts.iter().sum::<i32>(), slot_count as i32);

        let weight = weight_bytes(
            ctx,
            &synthetic_iq4_nl_bank(n_in, n_out, n_expert, n_in ^ n_out ^ n_expert ^ n_tokens),
            vec![n_in as u64, n_out as u64, n_expert as u64],
            GgmlType::IQ4_NL,
        );
        let inner_values = (0..slot_count * n_in)
            .map(|index| ((index * 43 + 17) % 257) as f32 * 0.000_5 - 0.064)
            .collect::<Vec<_>>();
        let inner = tensor_f32(ctx, &inner_values, vec![n_in as u64, slot_count as u64]);
        let counts = tensor_i32(ctx, &counts, vec![n_expert as u64]);
        let slots = tensor_i32(ctx, &slots, vec![n_tokens as u64, n_expert as u64]);

        let active_elements = slot_count * n_out;
        let guarded_output = |active_sentinel| {
            let storage =
                MetalTensor::zeros_f32(ctx, vec![(active_elements + GUARD_ELEMENTS) as u64])
                    .unwrap();
            fill_f32_bits(&storage, GUARD_F32_SENTINEL);
            let view = storage.view_subrange(0, vec![n_out as u64, slot_count as u64]);
            fill_f32_bits(&view, active_sentinel);
            (storage, view)
        };
        let (baseline_storage, baseline) = guarded_output(BASELINE_ACTIVE);
        let (candidate_storage, candidate) = guarded_output(CANDIDATE_ACTIVE);
        let input_snapshots = [
            ("weight", &weight, read_tensor_bytes(&weight)),
            ("inner", &inner, read_tensor_bytes(&inner)),
            ("counts", &counts, read_tensor_bytes(&counts)),
            ("slots", &slots, read_tensor_bytes(&slots)),
        ];

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots(
            ctx, &encoder, &weight, &inner, &counts, &slots, &baseline, n_in, n_out, n_expert,
            n_tokens,
        )
        .unwrap();
        crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots_m128_n16(
            ctx, &encoder, &weight, &inner, &counts, &slots, &candidate, n_in, n_out, n_expert,
            n_tokens,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 2, "{label} census: {census:#?}");
        for (row, kernel, tile_m, tile_n) in [
            (
                &census[0],
                "kernel_moe_down_iq4_nl_f32_grouped_slots",
                64,
                32,
            ),
            (
                &census[1],
                "kernel_moe_down_iq4_nl_f32_grouped_slots_m128_n16",
                128,
                16,
            ),
        ] {
            assert_eq!(row.kernel, kernel, "{label}");
            assert_eq!(row.grid_width, n_tokens.div_ceil(tile_n) as u64, "{label}");
            assert_eq!(row.grid_height, n_out.div_ceil(tile_m) as u64, "{label}");
            assert_eq!(row.grid_depth, n_expert as u64, "{label}");
            assert_eq!(
                (row.threads_width, row.threads_height, row.threads_depth),
                (128, 1, 1),
                "{label}",
            );
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(
            command.status(),
            MTLCommandBufferStatus::Completed,
            "{label}"
        );
        assert!(command.error().is_none(), "{label}: {:?}", command.error());

        for (name, tensor, before) in input_snapshots {
            assert_eq!(read_tensor_bytes(tensor), before, "{label} mutated {name}");
        }
        let baseline_values = read_f32(&baseline);
        let candidate_values = read_f32(&candidate);
        assert_bits_eq(label, &candidate_values, &baseline_values);
        assert!(
            baseline_values.iter().all(|value| {
                value.is_finite()
                    && value.to_bits() != BASELINE_ACTIVE
                    && value.to_bits() != CANDIDATE_ACTIVE
            }),
            "{label} left an output slot unwritten",
        );
        for (name, storage) in [
            ("baseline", &baseline_storage),
            ("candidate", &candidate_storage),
        ] {
            assert!(
                read_f32(storage)[active_elements..]
                    .iter()
                    .all(|value| value.to_bits() == GUARD_F32_SENTINEL),
                "{label} overwrote the {name} guard",
            );
        }
    }

    let residue_counts = [1_usize, 15, 16, 17, 31, 32, 33, 511, 512];
    let residue_assignments = residue_counts
        .iter()
        .enumerate()
        .flat_map(|(expert, &count)| std::iter::repeat_n(expert, count))
        .collect::<Vec<_>>();
    run_case(
        &ctx,
        "count residues",
        32,
        129,
        10,
        None,
        &residue_assignments,
    );

    let dispersed = (0..512).collect::<Vec<_>>();
    run_case(
        &ctx,
        "dispersed experts 0..511",
        32,
        64,
        512,
        None,
        &dispersed,
    );

    let alternating = (0..33).map(|token| token % 2).collect::<Vec<_>>();
    for n_out in [64_usize, 65, 127, 128, 129, 2_560] {
        run_case(
            &ctx,
            &format!("M={n_out}"),
            32,
            n_out,
            2,
            None,
            &alternating,
        );
    }
    for n_in in [64_usize, 640] {
        run_case(&ctx, &format!("K={n_in}"), n_in, 129, 2, None, &alternating);
    }

    let n527_topk10 = (0..10)
        .flat_map(|expert| std::iter::repeat_n(expert, 527))
        .collect::<Vec<_>>();
    run_case(
        &ctx,
        "N=527 top-k=10 boundary",
        32,
        129,
        10,
        Some(527),
        &n527_topk10,
    );
}

#[test]
fn grouped_iq4_nl_down_rejects_signed_shader_overflow() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    let down_bytes = synthetic_iq4_nl_bank(32, 1, 1, 223);
    let weight = weight_bytes(&ctx, &down_bytes, vec![32, 1, 1], GgmlType::IQ4_NL);
    let inner = tensor_f32(&ctx, &[0.0; 32], vec![32, 1]);
    let counts = tensor_i32(&ctx, &[1], vec![1]);
    let ids = tensor_i32(&ctx, &[0], vec![1]);
    let output = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let too_large = i32::MAX as usize + 1;
    let output_error = crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots(
        &ctx, &encoder, &weight, &inner, &counts, &ids, &output, 32, too_large, 1, 1,
    )
    .unwrap_err()
    .to_string();
    assert!(output_error.contains("n_out") && output_error.contains("signed"));
    let expert_error = crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots(
        &ctx, &encoder, &weight, &inner, &counts, &ids, &output, 32, 1, too_large, 1,
    )
    .unwrap_err()
    .to_string();
    assert!(expert_error.contains("n_expert") && expert_error.contains("signed"));
    encoder.end();
}

#[test]
fn packed_common_moe_motor_matches_serial_rows_and_routes() {
    let Some(ctx) = packed_test_context() else {
        return;
    };
    const HIDDEN: usize = 256;
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const ROUTED: usize = 64;
    const SHARED: usize = 64;
    const MAX_TOKENS: usize = 33;
    const CAPACITY: usize = 34;
    let geometry = Qwen4ExpMoeMetalGeometry::new(HIDDEN, EXPERTS, TOP_K, ROUTED, SHARED).unwrap();

    let mut router_values = vec![0.0_f32; HIDDEN * EXPERTS];
    for expert in 0..EXPERTS {
        let normalized = expert as f32 / EXPERTS as f32;
        let row = expert * HIDDEN;
        router_values[row] = 200.0 * normalized;
        router_values[row + 1] = 100.0 * normalized * normalized;
        router_values[row + 2] = normalized;
    }
    let router = weight_f32(&ctx, &router_values, vec![HIDDEN as u64, EXPERTS as u64]);
    let shared_router_values = (0..HIDDEN)
        .map(|lane| ((lane * 31 + 7) % 101) as f32 * 0.000_7 - 0.035)
        .collect::<Vec<_>>();
    let shared_router = weight_f32(&ctx, &shared_router_values, vec![HIDDEN as u64]);

    let routed_gate_source = synthetic_f32_bank(HIDDEN, ROUTED, EXPERTS, 307);
    let routed_gate_bytes = quantize_rows(&routed_gate_source, GgmlType::IQ3_XXS, HIDDEN);
    drop(routed_gate_source);
    let routed_up_source = synthetic_f32_bank(HIDDEN, ROUTED, EXPERTS, 331);
    let routed_up_bytes = quantize_rows(&routed_up_source, GgmlType::IQ3_XXS, HIDDEN);
    drop(routed_up_source);
    let routed_gate_iq4_bytes = synthetic_iq4_xs_bank(HIDDEN, ROUTED, EXPERTS, 337);
    let routed_up_iq4_bytes = synthetic_iq4_xs_bank(HIDDEN, ROUTED, EXPERTS, 347);
    let routed_down_bytes = synthetic_iq4_nl_bank(ROUTED, HIDDEN, EXPERTS, 353);
    let routed_down_q8_bytes = synthetic_q8_0_bank(ROUTED, HIDDEN, EXPERTS, 367);
    let shared_gate_bytes = synthetic_q8_0_bank(HIDDEN, SHARED, 1, 379);
    let shared_up_bytes = synthetic_q8_0_bank(HIDDEN, SHARED, 1, 401);
    let shared_down_bytes = synthetic_q8_0_bank(SHARED, HIDDEN, 1, 433);
    let routed_gate = weight_bytes(
        &ctx,
        &routed_gate_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ3_XXS,
    );
    let routed_up = weight_bytes(
        &ctx,
        &routed_up_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ3_XXS,
    );
    let routed_gate_iq4 = weight_bytes(
        &ctx,
        &routed_gate_iq4_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ4_XS,
    );
    let routed_up_iq4 = weight_bytes(
        &ctx,
        &routed_up_iq4_bytes,
        vec![HIDDEN as u64, ROUTED as u64, EXPERTS as u64],
        GgmlType::IQ4_XS,
    );
    let routed_down = weight_bytes(
        &ctx,
        &routed_down_bytes,
        vec![ROUTED as u64, HIDDEN as u64, EXPERTS as u64],
        GgmlType::IQ4_NL,
    );
    let routed_down_q8 = weight_bytes(
        &ctx,
        &routed_down_q8_bytes,
        vec![ROUTED as u64, HIDDEN as u64, EXPERTS as u64],
        GgmlType::Q8_0,
    );
    let shared_gate = weight_bytes(
        &ctx,
        &shared_gate_bytes,
        vec![HIDDEN as u64, SHARED as u64],
        GgmlType::Q8_0,
    );
    let shared_up = weight_bytes(
        &ctx,
        &shared_up_bytes,
        vec![HIDDEN as u64, SHARED as u64],
        GgmlType::Q8_0,
    );
    let shared_down = weight_bytes(
        &ctx,
        &shared_down_bytes,
        vec![SHARED as u64, HIDDEN as u64],
        GgmlType::Q8_0,
    );
    let weights = Qwen4ExpMoeMetalWeights {
        geometry,
        router: &router,
        routed_gate: &routed_gate,
        routed_up: &routed_up,
        routed_down: &routed_down,
        shared_router: &shared_router,
        shared_gate: &shared_gate,
        shared_up: &shared_up,
        shared_down: &shared_down,
    };
    let q8_weights = Qwen4ExpMoeMetalWeights {
        routed_down: &routed_down_q8,
        ..weights
    };
    let iq4_xs_q8_weights = Qwen4ExpMoeMetalWeights {
        routed_gate: &routed_gate_iq4,
        routed_up: &routed_up_iq4,
        routed_down: &routed_down_q8,
        ..weights
    };

    let mut inputs = (0..MAX_TOKENS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let lane = index % HIDDEN;
            ((token * 43 + lane * 17 + 11) % 127) as f32 * 0.000_8 - 0.05
        })
        .collect::<Vec<_>>();
    for token in 0..MAX_TOKENS {
        let target = 300 + (token * 17) % 180;
        let row = token * HIDDEN;
        inputs[row] = target as f32 / EXPERTS as f32;
        inputs[row + 1] = -1.0;
        inputs[row + 2] = -0.1;
    }
    let serial = serial_packed_moe_trace(&ctx, geometry, weights, &inputs, MAX_TOKENS);
    assert!(serial.topk_ids.iter().all(|&expert| expert >= 256));
    let mut packed_n33 = None;

    for tokens in [1_usize, 2, 8, 16, 33] {
        let scratch = Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, CAPACITY).unwrap();
        seed_packed_scratch(&scratch, tokens);
        let input = tensor_f32(
            &ctx,
            &inputs[..tokens * HIDDEN],
            vec![HIDDEN as u64, tokens as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let output = unsafe {
            encode_qwen4exp_moe_packed_motor(&ctx, &encoder, &input, weights, &scratch, tokens)
        }
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        let f32_matvec = if crate::metal::mat_vec_f32_lcpp_r2_enabled_for_test() {
            "kernel_mat_vec_f32_f32_lcpp_r2"
        } else {
            "kernel_mat_vec_f32_f32"
        };
        let q8_matvec = if crate::metal::mat_vec_q8_0_lcpp_enabled() {
            "kernel_mat_vec_q8_0_f32_lcpp"
        } else {
            "kernel_mat_vec_q8_0_f32"
        };
        let iq3_singleton = if qwen4exp_moe_iq3_fast_enabled() {
            "kernel_moe_swiglu_iq3_xxs_f32_fast"
        } else {
            "kernel_moe_swiglu_iq3_xxs_f32"
        };
        let iq4_down_singleton = if qwen4exp_moe_iq4_down_fast_enabled() {
            "kernel_moe_down_iq4_nl_f32_fast"
        } else {
            "kernel_moe_down_iq4_nl_f32"
        };
        let q8_matmat = |n_in: usize, n_out: usize| {
            if tokens == 8
                && crate::metal_forward::matmat_smalln_table_enabled_for_test()
                && n_in.is_multiple_of(256)
                && n_out.is_multiple_of(8)
            {
                "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32"
            } else if tokens == 16 {
                "kernel_mat_mat_q8_0_f32_n16"
            } else {
                "kernel_mat_mat_q8_0_f32"
            }
        };
        let expected_kernels = if tokens == 1 {
            vec![
                f32_matvec,
                "kernel_topk_logits_softmax_f32",
                "kernel_dot_sigmoid_f32",
                iq3_singleton,
                iq4_down_singleton,
                "kernel_moe_weighted_sum_f32",
                "kernel_shared_swiglu_q8_0_f32_lcpp",
                q8_matvec,
                "kernel_axpy_scalar_f32",
            ]
        } else {
            vec![
                "kernel_mat_mat_f32_f32",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_moe_route_bucket_slots_f32",
                "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16",
                "kernel_moe_down_iq4_nl_f32_grouped_slots",
                "kernel_moe_weighted_sum_packed_f32",
                q8_matmat(HIDDEN, SHARED),
                q8_matmat(HIDDEN, SHARED),
                "kernel_silu_mul_f32",
                q8_matmat(SHARED, HIDDEN),
                "kernel_axpy_rowwise_f32",
            ]
        };
        let actual_kernels = census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_kernels, expected_kernels,
            "packed common MoE N={tokens} route: {census:#?}"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        if tokens == 33 {
            const GUARD_ELEMENTS: usize = 64;
            let active_elements = tokens * TOP_K * ROUTED;
            let guarded_output = || {
                let storage =
                    MetalTensor::zeros_f32(&ctx, vec![(active_elements + GUARD_ELEMENTS) as u64])
                        .unwrap();
                fill_f32_bits(&storage, GUARD_F32_SENTINEL);
                let view =
                    storage.view_subrange(0, vec![ROUTED as u64, TOP_K as u64, tokens as u64]);
                fill_f32_bits(&view, ACTIVE_F32_SENTINEL);
                (storage, view)
            };
            let (full_storage, full) = guarded_output();
            let (bands_storage, bands) = guarded_output();
            let (no_work_storage, no_work) = guarded_output();
            let views = scratch.views(tokens).unwrap();
            let range_command = ctx.queue.commandBuffer().unwrap();
            let range_encoder = KernelEncoder::begin(&range_command);
            crate::metal::dispatch_census_begin();
            for (destination, arm) in [
                (&full, Qwen4ExpIq3GateUpProbeArm::Full),
                (&bands, Qwen4ExpIq3GateUpProbeArm::Count1To8),
                (&bands, Qwen4ExpIq3GateUpProbeArm::Count9To16),
                (&bands, Qwen4ExpIq3GateUpProbeArm::Count17To32),
                (&bands, Qwen4ExpIq3GateUpProbeArm::Count33To64),
                (&bands, Qwen4ExpIq3GateUpProbeArm::Count65Plus),
                (&no_work, Qwen4ExpIq3GateUpProbeArm::NoWork),
            ] {
                let (min_count, max_count) = arm.bounds();
                encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
                    &ctx,
                    &range_encoder,
                    weights.routed_gate,
                    weights.routed_up,
                    &input,
                    &views.route_counts,
                    &views.route_slots,
                    destination,
                    HIDDEN,
                    ROUTED,
                    EXPERTS,
                    TOP_K,
                    tokens,
                    min_count,
                    max_count,
                )
                .unwrap();
            }
            let range_census = crate::metal::dispatch_census_take();
            assert_eq!(range_census.len(), 7);
            assert!(range_census.iter().all(|row| {
                row.kernel == "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16"
                    && row.grid_width == tokens.div_ceil(16) as u64
                    && row.grid_height == ROUTED.div_ceil(64) as u64
                    && row.grid_depth == EXPERTS as u64
                    && row.threads_width == 128
            }));
            range_encoder.end();
            range_command.commit();
            range_command.waitUntilCompleted();
            assert_eq!(range_command.status(), MTLCommandBufferStatus::Completed);
            assert!(range_command.error().is_none());

            let production = read_f32(&views.routed_inner);
            assert_bits_eq(
                "IQ3 N16 full-range differential",
                &read_f32(&full),
                &production,
            );
            assert_bits_eq(
                "IQ3 N16 disjoint count-band differential",
                &read_f32(&bands),
                &production,
            );
            assert!(
                read_f32(&no_work)
                    .iter()
                    .all(|value| value.to_bits() == ACTIVE_F32_SENTINEL)
            );
            for storage in [&full_storage, &bands_storage, &no_work_storage] {
                assert!(
                    read_f32(storage)[active_elements..]
                        .iter()
                        .all(|value| value.to_bits() == GUARD_F32_SENTINEL)
                );
            }
        }

        if tokens == 8 {
            if let Ok(samples) = ctx.timestamp_sample_buffer(QWEN4EXP_PACKED_PROFILE_MOE_STAGES * 2)
            {
                let split_scratch =
                    Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, CAPACITY).unwrap();
                seed_packed_scratch(&split_scratch, tokens);
                let split_command = ctx.queue.commandBuffer().unwrap();
                let (split_output, spans) = unsafe {
                    encode_qwen4exp_moe_packed_motor_stage_sampled(
                        &ctx,
                        &split_command,
                        &samples,
                        0,
                        &input,
                        weights,
                        &split_scratch,
                        tokens,
                        5,
                        MixerKind::GatedDeltaNet,
                    )
                }
                .unwrap();
                split_command.commit();
                split_command.waitUntilCompleted();
                assert_eq!(split_command.status(), MTLCommandBufferStatus::Completed);
                assert!(split_command.error().is_none());
                assert_eq!(spans.len(), QWEN4EXP_PACKED_PROFILE_MOE_SPANS);
                let expected_spans = [
                    ("moe.routing", 2, 0, 5),
                    ("moe.router", 3, 0, 1),
                    ("moe.topk", 3, 2, 3),
                    ("moe.bucket", 3, 4, 5),
                    ("moe.routed_gate_up", 2, 6, 7),
                    ("moe.routed_down", 2, 8, 9),
                    ("moe.routed_reduce", 2, 10, 11),
                    ("moe.shared_tail", 2, 12, 13),
                ];
                for (span, (name, depth, start, end)) in spans.iter().zip(expected_spans) {
                    assert_eq!(span.label.name, name);
                    assert_eq!(span.depth, depth);
                    assert_eq!(span.start_sample, start);
                    assert_eq!(span.end_sample, end);
                }
                assert_packed_scratch_bits_eq(
                    "packed common MoE monolithic/split",
                    &split_scratch,
                    &scratch,
                    tokens,
                );
                assert_bits_eq(
                    "packed common MoE returned split output",
                    &read_f32(&split_output),
                    &read_f32(&output),
                );
                assert_packed_scratch_guards(&split_scratch, tokens);
            } else {
                eprintln!(
                    "packed common MoE split equivalence skipped: stage counters unavailable"
                );
            }
        }

        let views = scratch.views(tokens).unwrap();
        let actual_ids = read_i32(&views.topk_ids);
        assert_eq!(
            actual_ids,
            serial.topk_ids[..tokens * TOP_K],
            "packed common MoE N={tokens} top-k IDs"
        );
        let packed_stages = [
            (
                "router logits",
                read_f32(&views.router_logits),
                &serial.router_logits[..tokens * EXPERTS],
                EXPERTS,
            ),
            (
                "top-k weights",
                read_f32(&views.topk_weights),
                &serial.topk_weights[..tokens * TOP_K],
                TOP_K,
            ),
            (
                "shared scale",
                read_f32(&views.shared_scale),
                &serial.shared_scale[..tokens],
                1,
            ),
            (
                "routed inner",
                read_f32(&views.routed_inner),
                &serial.routed_inner[..tokens * TOP_K * ROUTED],
                TOP_K * ROUTED,
            ),
            (
                "routed expert output",
                read_f32(&views.routed_expert_output),
                &serial.routed_expert_output[..tokens * TOP_K * HIDDEN],
                TOP_K * HIDDEN,
            ),
            (
                "shared inner",
                read_f32(&views.shared_inner),
                &serial.shared_inner[..tokens * SHARED],
                SHARED,
            ),
            (
                "shared output",
                read_f32(&views.shared_output),
                &serial.shared_output[..tokens * HIDDEN],
                HIDDEN,
            ),
            (
                "output",
                read_f32(&output),
                &serial.output[..tokens * HIDDEN],
                HIDDEN,
            ),
        ];
        if tokens == 1 {
            for (stage, actual, expected, _) in &packed_stages {
                assert_bits_eq(&format!("packed common MoE N=1 {stage}"), actual, expected);
            }
        } else {
            for (stage, actual, expected, width) in &packed_stages {
                let (max_abs, cosine) = match *stage {
                    "router logits" => (1e-6, 0.999_999_99),
                    "top-k weights" => (1e-6, 0.999_999_99),
                    "shared scale" => (1e-7, 0.999_999_99),
                    "routed inner" => (1.5e-5, 0.999_999_9),
                    "routed expert output" => (1.5e-6, 0.999_999_85),
                    "shared inner" => (1.5e-7, 0.999_999_9),
                    "shared output" => (3e-8, 0.999_999_8),
                    "output" => (3e-7, 0.999_999_5),
                    _ => unreachable!(),
                };
                assert_tokenwise_similarity(
                    &format!("packed common MoE N={tokens} {stage}"),
                    actual,
                    expected,
                    *width,
                    tokens,
                    max_abs,
                    cosine,
                );
            }

            let actual_counts = read_i32(&views.route_counts);
            let actual_slots = read_i32(&views.route_slots);
            assert_eq!(actual_counts.iter().sum::<i32>(), (tokens * TOP_K) as i32);
            let mut seen = vec![false; tokens * TOP_K];
            for expert in 0..EXPERTS {
                let count = actual_counts[expert] as usize;
                assert!(count <= tokens);
                for &slot in &actual_slots[expert * tokens..expert * tokens + count] {
                    let slot = slot as usize;
                    assert!(slot < tokens * TOP_K);
                    assert!(!seen[slot], "packed route slot {slot} appeared twice");
                    assert_eq!(actual_ids[slot], expert as i32);
                    seen[slot] = true;
                }
                assert!(
                    actual_slots[expert * tokens + count..expert * tokens + tokens]
                        .iter()
                        .all(|&slot| slot == ACTIVE_I32_SENTINEL),
                    "packed route expert {expert} wrote beyond count {count}"
                );
            }
            assert!(seen.into_iter().all(|present| present));
        }
        assert_packed_scratch_guards(&scratch, tokens);
        if tokens == 33 {
            packed_n33 = Some(read_f32(&output));
        }
    }

    {
        const TOKENS: usize = 65;
        const GUARD_ELEMENTS: usize = 64;
        let seeded_counts = [1, 8, 9, 16, 17, 32, 33, 64, 65, 65, 65, 65, 65, 65, 65, 15];
        assert_eq!(seeded_counts.iter().sum::<usize>(), TOKENS * TOP_K);
        let mut counts = vec![0_i32; EXPERTS];
        let mut slots = vec![ACTIVE_I32_SENTINEL; EXPERTS * TOKENS];
        let mut topk_ids = vec![ACTIVE_I32_SENTINEL; TOKENS * TOP_K];
        let mut next_slot = 0_usize;
        for (expert, &count) in seeded_counts.iter().enumerate() {
            counts[expert] = count as i32;
            for local in 0..count {
                slots[expert * TOKENS + local] = next_slot as i32;
                topk_ids[next_slot] = expert as i32;
                next_slot += 1;
            }
        }
        assert_eq!(next_slot, TOKENS * TOP_K);
        for &(min_count, max_count) in &[(1, 8), (9, 16), (17, 32), (33, 64), (65, 65)] {
            assert!(
                counts
                    .iter()
                    .copied()
                    .filter(|&count| (min_count..=max_count).contains(&count))
                    .sum::<i32>()
                    > 0,
                "explicit fixture must exercise count band {min_count}..={max_count}"
            );
        }
        let input_values = (0..TOKENS * HIDDEN)
            .map(|index| ((index * 37 + 19) % 257) as f32 * 0.000_4 - 0.051)
            .collect::<Vec<_>>();
        let input = tensor_f32(&ctx, &input_values, vec![HIDDEN as u64, TOKENS as u64]);
        let counts_tensor = tensor_i32(&ctx, &counts, vec![EXPERTS as u64]);
        let slots_tensor = tensor_i32(&ctx, &slots, vec![TOKENS as u64, EXPERTS as u64]);
        let active_elements = TOKENS * TOP_K * ROUTED;
        let guarded_output = || {
            let storage =
                MetalTensor::zeros_f32(&ctx, vec![(active_elements + GUARD_ELEMENTS) as u64])
                    .unwrap();
            fill_f32_bits(&storage, GUARD_F32_SENTINEL);
            let view = storage.view_subrange(0, vec![ROUTED as u64, TOP_K as u64, TOKENS as u64]);
            fill_f32_bits(&view, ACTIVE_F32_SENTINEL);
            (storage, view)
        };
        let (full_storage, full) = guarded_output();
        let (bands_storage, bands) = guarded_output();
        let (no_work_storage, no_work) = guarded_output();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for (destination, arm) in [
            (&full, Qwen4ExpIq3GateUpProbeArm::Full),
            (&bands, Qwen4ExpIq3GateUpProbeArm::Count1To8),
            (&bands, Qwen4ExpIq3GateUpProbeArm::Count9To16),
            (&bands, Qwen4ExpIq3GateUpProbeArm::Count17To32),
            (&bands, Qwen4ExpIq3GateUpProbeArm::Count33To64),
            (&bands, Qwen4ExpIq3GateUpProbeArm::Count65Plus),
            (&no_work, Qwen4ExpIq3GateUpProbeArm::NoWork),
        ] {
            let (min_count, max_count) = arm.bounds();
            encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
                &ctx,
                &encoder,
                weights.routed_gate,
                weights.routed_up,
                &input,
                &counts_tensor,
                &slots_tensor,
                destination,
                HIDDEN,
                ROUTED,
                EXPERTS,
                TOP_K,
                TOKENS,
                min_count,
                max_count,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        assert_bits_eq(
            "IQ3 N16 explicit all-band differential",
            &read_f32(&bands),
            &read_f32(&full),
        );
        assert!(
            read_f32(&full)
                .iter()
                .all(|value| value.to_bits() != ACTIVE_F32_SENTINEL)
        );
        assert!(
            read_f32(&no_work)
                .iter()
                .all(|value| value.to_bits() == ACTIVE_F32_SENTINEL)
        );
        for storage in [&full_storage, &bands_storage, &no_work_storage] {
            assert!(
                read_f32(storage)[active_elements..]
                    .iter()
                    .all(|value| value.to_bits() == GUARD_F32_SENTINEL)
            );
        }
        let mut seen = vec![false; TOKENS * TOP_K];
        for expert in 0..EXPERTS {
            let count = counts[expert] as usize;
            for &slot in &slots[expert * TOKENS..expert * TOKENS + count] {
                let slot = slot as usize;
                assert!(slot < seen.len());
                assert!(!seen[slot]);
                assert_eq!(topk_ids[slot], expert as i32);
                seen[slot] = true;
            }
            assert!(
                slots[expert * TOKENS + count..expert * TOKENS + TOKENS]
                    .iter()
                    .all(|&slot| slot == ACTIVE_I32_SENTINEL)
            );
        }
        assert!(seen.into_iter().all(|present| present));
    }

    let mut perturbed_inputs = inputs.clone();
    const CHANGED_TOKEN: usize = 17;
    perturbed_inputs[CHANGED_TOKEN * HIDDEN + 5] += 0.75;
    let perturbed_input = tensor_f32(
        &ctx,
        &perturbed_inputs,
        vec![HIDDEN as u64, MAX_TOKENS as u64],
    );
    let perturbed_scratch = Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, CAPACITY).unwrap();
    seed_packed_scratch(&perturbed_scratch, MAX_TOKENS);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let perturbed_output = unsafe {
        encode_qwen4exp_moe_packed_motor(
            &ctx,
            &encoder,
            &perturbed_input,
            weights,
            &perturbed_scratch,
            MAX_TOKENS,
        )
    }
    .unwrap();
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    assert_packed_scratch_guards(&perturbed_scratch, MAX_TOKENS);
    let baseline = packed_n33.unwrap();
    let perturbed = read_f32(&perturbed_output);
    assert!(
        perturbed
            .iter()
            .all(|value| { value.is_finite() && value.to_bits() != ACTIVE_F32_SENTINEL })
    );
    for token in 0..MAX_TOKENS {
        let start = token * HIDDEN;
        if token == CHANGED_TOKEN {
            assert!(
                baseline[start..start + HIDDEN]
                    .iter()
                    .zip(&perturbed[start..start + HIDDEN])
                    .any(|(baseline, perturbed)| baseline.to_bits() != perturbed.to_bits())
            );
        } else {
            assert_bits_eq(
                &format!("packed common MoE perturbation token={token}"),
                &perturbed[start..start + HIDDEN],
                &baseline[start..start + HIDDEN],
            );
        }
    }

    let q8_serial = serial_packed_moe_trace(&ctx, geometry, q8_weights, &inputs, 8);
    for tokens in [1_usize, 8] {
        let q8_scratch = Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, 9).unwrap();
        seed_packed_scratch(&q8_scratch, tokens);
        let input = tensor_f32(
            &ctx,
            &inputs[..tokens * HIDDEN],
            vec![HIDDEN as u64, tokens as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let output = unsafe {
            encode_qwen4exp_moe_packed_motor(
                &ctx,
                &encoder,
                &input,
                q8_weights,
                &q8_scratch,
                tokens,
            )
        }
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        let f32_matvec = if crate::metal::mat_vec_f32_lcpp_r2_enabled_for_test() {
            "kernel_mat_vec_f32_f32_lcpp_r2"
        } else {
            "kernel_mat_vec_f32_f32"
        };
        let q8_matvec = if crate::metal::mat_vec_q8_0_lcpp_enabled() {
            "kernel_mat_vec_q8_0_f32_lcpp"
        } else {
            "kernel_mat_vec_q8_0_f32"
        };
        let iq3_singleton = if qwen4exp_moe_iq3_fast_enabled() {
            "kernel_moe_swiglu_iq3_xxs_f32_fast"
        } else {
            "kernel_moe_swiglu_iq3_xxs_f32"
        };
        let q8_matmat = |n_in: usize, n_out: usize| {
            if crate::metal_forward::matmat_smalln_table_enabled_for_test()
                && n_in.is_multiple_of(256)
                && n_out.is_multiple_of(8)
            {
                "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32"
            } else {
                "kernel_mat_mat_q8_0_f32"
            }
        };
        let expected_kernels = if tokens == 1 {
            vec![
                f32_matvec,
                "kernel_topk_logits_softmax_f32",
                "kernel_dot_sigmoid_f32",
                iq3_singleton,
                "kernel_moe_down_weighted_sum_q8_0_f32",
                "kernel_shared_swiglu_q8_0_f32_lcpp",
                q8_matvec,
                "kernel_axpy_scalar_f32",
            ]
        } else {
            vec![
                "kernel_mat_mat_f32_f32",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_moe_route_bucket_slots_f32",
                "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16",
                "kernel_moe_down_q8_0_f32_grouped_slots",
                "kernel_moe_weighted_sum_packed_f32",
                q8_matmat(HIDDEN, SHARED),
                q8_matmat(HIDDEN, SHARED),
                "kernel_silu_mul_f32",
                q8_matmat(SHARED, HIDDEN),
                "kernel_axpy_rowwise_f32",
            ]
        };
        assert_eq!(
            census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>(),
            expected_kernels,
            "packed Q8-down MoE N={tokens} route: {census:#?}"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let views = q8_scratch.views(tokens).unwrap();
        assert_eq!(
            read_i32(&views.topk_ids),
            q8_serial.topk_ids[..tokens * TOP_K]
        );
        let stages = [
            (
                "router logits",
                read_f32(&views.router_logits),
                &q8_serial.router_logits[..tokens * EXPERTS],
                EXPERTS,
            ),
            (
                "top-k weights",
                read_f32(&views.topk_weights),
                &q8_serial.topk_weights[..tokens * TOP_K],
                TOP_K,
            ),
            (
                "shared scale",
                read_f32(&views.shared_scale),
                &q8_serial.shared_scale[..tokens],
                1,
            ),
            (
                "routed inner",
                read_f32(&views.routed_inner),
                &q8_serial.routed_inner[..tokens * TOP_K * ROUTED],
                TOP_K * ROUTED,
            ),
            (
                "shared inner",
                read_f32(&views.shared_inner),
                &q8_serial.shared_inner[..tokens * SHARED],
                SHARED,
            ),
            (
                "shared output",
                read_f32(&views.shared_output),
                &q8_serial.shared_output[..tokens * HIDDEN],
                HIDDEN,
            ),
            (
                "output",
                read_f32(&output),
                &q8_serial.output[..tokens * HIDDEN],
                HIDDEN,
            ),
        ];
        if tokens == 1 {
            for (stage, actual, expected, _) in &stages {
                assert_bits_eq(&format!("packed Q8-down MoE N=1 {stage}"), actual, expected);
            }
        } else {
            for (stage, actual, expected, width) in &stages {
                let (max_abs, cosine) = match *stage {
                    "router logits" => (1e-6, 0.999_999_99),
                    "top-k weights" => (1e-6, 0.999_999_99),
                    "shared scale" => (1e-7, 0.999_999_99),
                    "routed inner" => (1.5e-5, 0.999_999_9),
                    "shared inner" => (1.5e-7, 0.999_999_9),
                    "shared output" => (3e-8, 0.999_999_8),
                    "output" => (2.5e-7, 0.999_999_8),
                    _ => unreachable!(),
                };
                assert_tokenwise_similarity(
                    &format!("packed Q8-down MoE N={tokens} {stage}"),
                    actual,
                    expected,
                    *width,
                    tokens,
                    max_abs,
                    cosine,
                );
            }
        }
        assert_packed_scratch_guards(&q8_scratch, tokens);
    }

    let iq4_xs_q8_serial = serial_packed_moe_trace(&ctx, geometry, iq4_xs_q8_weights, &inputs, 8);
    for tokens in [1_usize, 8] {
        let scratch = Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, 9).unwrap();
        seed_packed_scratch(&scratch, tokens);
        let input = tensor_f32(
            &ctx,
            &inputs[..tokens * HIDDEN],
            vec![HIDDEN as u64, tokens as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let output = unsafe {
            encode_qwen4exp_moe_packed_motor(
                &ctx,
                &encoder,
                &input,
                iq4_xs_q8_weights,
                &scratch,
                tokens,
            )
        }
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        let f32_matvec = if crate::metal::mat_vec_f32_lcpp_r2_enabled_for_test() {
            "kernel_mat_vec_f32_f32_lcpp_r2"
        } else {
            "kernel_mat_vec_f32_f32"
        };
        let q8_matvec = if crate::metal::mat_vec_q8_0_lcpp_enabled() {
            "kernel_mat_vec_q8_0_f32_lcpp"
        } else {
            "kernel_mat_vec_q8_0_f32"
        };
        let q8_matmat = |n_in: usize, n_out: usize| {
            if crate::metal_forward::matmat_smalln_table_enabled_for_test()
                && n_in.is_multiple_of(256)
                && n_out.is_multiple_of(8)
            {
                "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32"
            } else {
                "kernel_mat_mat_q8_0_f32"
            }
        };
        let expected_kernels = if tokens == 1 {
            vec![
                f32_matvec,
                "kernel_topk_logits_softmax_f32",
                "kernel_dot_sigmoid_f32",
                "kernel_moe_swiglu_iq4_xs_f32",
                "kernel_moe_down_weighted_sum_q8_0_f32",
                "kernel_shared_swiglu_q8_0_f32_lcpp",
                q8_matvec,
                "kernel_axpy_scalar_f32",
            ]
        } else {
            vec![
                "kernel_mat_mat_f32_f32",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_moe_route_bucket_slots_f32",
                "kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16",
                "kernel_moe_down_q8_0_f32_grouped_slots",
                "kernel_moe_weighted_sum_packed_f32",
                q8_matmat(HIDDEN, SHARED),
                q8_matmat(HIDDEN, SHARED),
                "kernel_silu_mul_f32",
                q8_matmat(SHARED, HIDDEN),
                "kernel_axpy_rowwise_f32",
            ]
        };
        assert_eq!(
            census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>(),
            expected_kernels,
            "packed IQ4_XS/Q8 MoE N={tokens} route: {census:#?}"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let views = scratch.views(tokens).unwrap();
        assert_eq!(
            read_i32(&views.topk_ids),
            iq4_xs_q8_serial.topk_ids[..tokens * TOP_K]
        );
        let stages = [
            (
                "router logits",
                read_f32(&views.router_logits),
                &iq4_xs_q8_serial.router_logits[..tokens * EXPERTS],
                EXPERTS,
            ),
            (
                "top-k weights",
                read_f32(&views.topk_weights),
                &iq4_xs_q8_serial.topk_weights[..tokens * TOP_K],
                TOP_K,
            ),
            (
                "shared scale",
                read_f32(&views.shared_scale),
                &iq4_xs_q8_serial.shared_scale[..tokens],
                1,
            ),
            (
                "routed inner",
                read_f32(&views.routed_inner),
                &iq4_xs_q8_serial.routed_inner[..tokens * TOP_K * ROUTED],
                TOP_K * ROUTED,
            ),
            (
                "shared inner",
                read_f32(&views.shared_inner),
                &iq4_xs_q8_serial.shared_inner[..tokens * SHARED],
                SHARED,
            ),
            (
                "shared output",
                read_f32(&views.shared_output),
                &iq4_xs_q8_serial.shared_output[..tokens * HIDDEN],
                HIDDEN,
            ),
            (
                "output",
                read_f32(&output),
                &iq4_xs_q8_serial.output[..tokens * HIDDEN],
                HIDDEN,
            ),
        ];
        if tokens == 1 {
            for (stage, actual, expected, _) in &stages {
                assert_bits_eq(
                    &format!("packed IQ4_XS/Q8 MoE N=1 {stage}"),
                    actual,
                    expected,
                );
            }
        } else {
            for (stage, actual, expected, width) in &stages {
                let (max_abs, cosine) = match *stage {
                    "router logits" => (1e-6, 0.999_999_99),
                    "top-k weights" => (1e-6, 0.999_999_99),
                    "shared scale" => (1e-7, 0.999_999_99),
                    "routed inner" => (1.5e-5, 0.999_999_8),
                    "shared inner" => (1.5e-7, 0.999_999_9),
                    "shared output" => (3e-8, 0.999_999_8),
                    "output" => (3e-7, 0.999_999_8),
                    _ => unreachable!(),
                };
                assert_tokenwise_similarity(
                    &format!("packed IQ4_XS/Q8 MoE N={tokens} {stage}"),
                    actual,
                    expected,
                    *width,
                    tokens,
                    max_abs,
                    cosine,
                );
            }
        }
        assert_packed_scratch_guards(&scratch, tokens);
    }

    let capacity_scratch =
        Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, MAX_PACKED_TOKENS).unwrap();
    assert_eq!(
        capacity_scratch
            .views(MAX_PACKED_TOKENS)
            .unwrap()
            .output
            .n_elements(),
        (MAX_PACKED_TOKENS * HIDDEN) as u64
    );
    let released_geometry = Qwen4ExpMoeMetalGeometry::new(2_560, 512, 10, 640, 640).unwrap();
    assert_eq!(
        Qwen4ExpMoePackedMotorScratch::required_bytes(released_geometry, MAX_PACKED_TOKENS,)
            .unwrap(),
        328_378_368
    );
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}[{index}]: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

fn mat_vec(weight: &[f32], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    assert_eq!(weight.len(), n_in * n_out);
    assert_eq!(input.len(), n_in);
    weight
        .chunks_exact(n_in)
        .take(n_out)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn encode_iq4_xs_block(d: f32, seed: usize) -> [u8; 136] {
    let mut block = [0_u8; 136];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    let mut scales_h = 0_u16;
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

fn synthetic_iq4_xs_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
    assert!(n_in.is_multiple_of(256));
    let blocks_per_row = n_in / 256;
    let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 136);
    for expert in 0..experts {
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

fn encode_iq4_nl_block(d: f32, seed: usize) -> [u8; 18] {
    let mut block = [0_u8; 18];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for lane in 0..16 {
        let low = (seed + lane * 3) % 16;
        let high = (seed * 5 + lane * 7 + 1) % 16;
        block[2 + lane] = low as u8 | ((high as u8) << 4);
    }
    block
}

fn synthetic_iq4_nl_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
    assert!(n_in.is_multiple_of(32));
    let blocks_per_row = n_in / 32;
    let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 18);
    for expert in 0..experts {
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let d = sign * (ordinal % 7 + 1) as f32 / 16_384.0;
                bytes.extend_from_slice(&encode_iq4_nl_block(d, ordinal));
            }
        }
    }
    bytes
}

fn encode_q8_0_block(d: f32, seed: usize) -> [u8; 34] {
    let mut block = [0_u8; 34];
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for lane in 0..32 {
        let quant = ((seed * 13 + lane * 7 + 5) % 31) as i8 - 15;
        block[2 + lane] = quant as u8;
    }
    block
}

fn synthetic_q8_0_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
    assert!(n_in.is_multiple_of(32));
    let blocks_per_row = n_in / 32;
    let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 34);
    for expert in 0..experts {
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                let sign = if ordinal.is_multiple_of(3) { -1.0 } else { 1.0 };
                let d = sign * (ordinal % 5 + 1) as f32 / 2_048.0;
                bytes.extend_from_slice(&encode_q8_0_block(d, ordinal));
            }
        }
    }
    bytes
}

fn synthetic_f32_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<f32> {
    (0..n_in * n_out * experts)
        .map(|index| {
            let row = index / n_in;
            let column = index % n_in;
            let raw = (row * 29 + column * 17 + index / 13 * 7 + seed * 11 + 5) % 127;
            (raw as f32 - 63.0) * 0.0015
        })
        .collect()
}

fn quantize_rows(values: &[f32], dtype: GgmlType, n_per_row: usize) -> Vec<u8> {
    assert!(!values.is_empty());
    assert!(values.len().is_multiple_of(n_per_row));
    let (block_elements, block_bytes) = dtype.storage_layout().unwrap();
    assert!((n_per_row as u64).is_multiple_of(block_elements));
    let expected_bytes = values.len() / block_elements as usize * block_bytes as usize;
    let mut bytes = vec![0_u8; expected_bytes];
    unsafe {
        llama_cpp_sys_2::ggml_quantize_init(dtype as u32);
        let written = llama_cpp_sys_2::ggml_quantize_chunk(
            dtype as u32,
            values.as_ptr(),
            bytes.as_mut_ptr().cast(),
            0,
            (values.len() / n_per_row) as i64,
            n_per_row as i64,
            std::ptr::null(),
        );
        assert_eq!(written, expected_bytes);
    }
    bytes
}

fn dequant_matrix(bytes: &[u8], dtype: GgmlType, n_in: usize, n_out: usize) -> Vec<f32> {
    let desc = TensorDesc {
        name: format!("synthetic_{dtype:?}"),
        shape: vec![n_in as u64, n_out as u64],
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    crate::codec::dequant_to_f32(&desc, bytes).unwrap()
}

fn dequant_expert(
    bytes: &[u8],
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    expert: usize,
) -> Vec<f32> {
    let (block_elements, block_bytes) = dtype.storage_layout().unwrap();
    let row_bytes = n_in / block_elements as usize * block_bytes as usize;
    let expert_bytes = n_out * row_bytes;
    let start = expert * expert_bytes;
    dequant_matrix(&bytes[start..start + expert_bytes], dtype, n_in, n_out)
}

struct CpuQuantizedInputs<'a> {
    geometry: Qwen4ExpMoeMetalGeometry,
    input: &'a [f32],
    topk_ids: &'a [i32],
    topk_weights: &'a [f32],
    shared_gate_scalar: f32,
    routed_gate: &'a [u8],
    routed_up: &'a [u8],
    routed_dtype: GgmlType,
    routed_down: &'a [u8],
    routed_down_dtype: GgmlType,
    shared_gate: &'a [u8],
    shared_up: &'a [u8],
    shared_down: &'a [u8],
}

fn cpu_quantized_moe(inputs: CpuQuantizedInputs<'_>) -> CpuMoeResult {
    let g = inputs.geometry;
    let mut routed_inner = vec![0.0; g.experts_per_token * g.routed_intermediate_size];
    for (slot, &expert) in inputs.topk_ids.iter().enumerate() {
        let expert = expert as usize;
        let gate = dequant_expert(
            inputs.routed_gate,
            inputs.routed_dtype,
            g.hidden_size,
            g.routed_intermediate_size,
            expert,
        );
        let up = dequant_expert(
            inputs.routed_up,
            inputs.routed_dtype,
            g.hidden_size,
            g.routed_intermediate_size,
            expert,
        );
        let gate = mat_vec(
            &gate,
            inputs.input,
            g.hidden_size,
            g.routed_intermediate_size,
        );
        let up = mat_vec(&up, inputs.input, g.hidden_size, g.routed_intermediate_size);
        for row in 0..g.routed_intermediate_size {
            routed_inner[slot * g.routed_intermediate_size + row] = silu(gate[row]) * up[row];
        }
    }

    let mut routed_expert_output = vec![0.0; g.experts_per_token * g.hidden_size];
    let mut output = vec![0.0; g.hidden_size];
    for (slot, (&expert, &route_weight)) in
        inputs.topk_ids.iter().zip(inputs.topk_weights).enumerate()
    {
        let down = dequant_expert(
            inputs.routed_down,
            inputs.routed_down_dtype,
            g.routed_intermediate_size,
            g.hidden_size,
            expert as usize,
        );
        let expert_output = mat_vec(
            &down,
            &routed_inner
                [slot * g.routed_intermediate_size..(slot + 1) * g.routed_intermediate_size],
            g.routed_intermediate_size,
            g.hidden_size,
        );
        routed_expert_output[slot * g.hidden_size..(slot + 1) * g.hidden_size]
            .copy_from_slice(&expert_output);
        for (output, expert_value) in output.iter_mut().zip(expert_output) {
            *output += route_weight * expert_value;
        }
    }

    let shared_gate_weight = dequant_matrix(
        inputs.shared_gate,
        GgmlType::Q8_0,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_up_weight = dequant_matrix(
        inputs.shared_up,
        GgmlType::Q8_0,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_down_weight = dequant_matrix(
        inputs.shared_down,
        GgmlType::Q8_0,
        g.shared_intermediate_size,
        g.hidden_size,
    );
    let shared_gate = mat_vec(
        &shared_gate_weight,
        inputs.input,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_up = mat_vec(
        &shared_up_weight,
        inputs.input,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_inner = shared_gate
        .into_iter()
        .zip(shared_up)
        .map(|(gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let shared_output = mat_vec(
        &shared_down_weight,
        &shared_inner,
        g.shared_intermediate_size,
        g.hidden_size,
    );
    for (output, &shared) in output.iter_mut().zip(&shared_output) {
        *output += inputs.shared_gate_scalar * shared;
    }
    CpuMoeResult {
        routed_inner,
        routed_expert_output,
        shared_inner,
        shared_output,
        output,
    }
}

#[test]
fn released_geometry_and_kernel_constraints_are_explicit() {
    let reference = Qwen4ExpConfig::flash_next_reference();
    let geometry = Qwen4ExpMoeMetalGeometry::from_config(&reference).unwrap();
    assert_eq!(geometry.hidden_size(), 2_560);
    assert_eq!(geometry.expert_count(), 512);
    assert_eq!(geometry.experts_per_token(), 10);
    assert_eq!(geometry.routed_intermediate_size(), 640);
    assert_eq!(geometry.shared_intermediate_size(), 640);
    assert!(Qwen4ExpMoeMetalGeometry::new(255, 16, 10, 32, 32).is_err());
    assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 17, 32, 32).is_err());
    assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 10, 31, 32).is_err());
    assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 10, 32, 31).is_err());
}

#[test]
fn iq4_xs_q8_composition_matches_routing_fixture_and_cpu_dequant() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (oracle, values) = parse_oracle();
    let g = geometry(&oracle);
    let input = oracle_section(&oracle, &values, "input");
    let router = oracle_section(&oracle, &values, "router");
    let shared_router = oracle_section(&oracle, &values, "shared_router");
    let expected_logits = oracle_section(&oracle, &values, "router_logits");
    let expected_weights = oracle_section(&oracle, &values, "topk_weights");
    let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];

    let routed_gate_bytes = synthetic_iq4_xs_bank(
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        17,
    );
    let routed_up_bytes = synthetic_iq4_xs_bank(
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        10_003,
    );
    let routed_down_bytes = synthetic_q8_0_bank(
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        20_011,
    );
    let shared_gate_bytes =
        synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 30_007);
    let shared_up_bytes = synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 40_009);
    let shared_down_bytes =
        synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 50_021);
    let expected = cpu_quantized_moe(CpuQuantizedInputs {
        geometry: g,
        input,
        topk_ids: &oracle.topk_ids,
        topk_weights: expected_weights,
        shared_gate_scalar: expected_shared_gate,
        routed_gate: &routed_gate_bytes,
        routed_up: &routed_up_bytes,
        routed_dtype: GgmlType::IQ4_XS,
        routed_down: &routed_down_bytes,
        routed_down_dtype: GgmlType::Q8_0,
        shared_gate: &shared_gate_bytes,
        shared_up: &shared_up_bytes,
        shared_down: &shared_down_bytes,
    });

    let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
    let router_gpu = weight_f32(
        &ctx,
        router,
        vec![g.hidden_size as u64, g.expert_count as u64],
    );
    let routed_gate_gpu = weight_bytes(
        &ctx,
        &routed_gate_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ4_XS,
    );
    let routed_up_gpu = weight_bytes(
        &ctx,
        &routed_up_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ4_XS,
    );
    let routed_down_gpu = weight_bytes(
        &ctx,
        &routed_down_bytes,
        vec![
            g.routed_intermediate_size as u64,
            g.hidden_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::Q8_0,
    );
    let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
    let shared_gate_gpu = weight_bytes(
        &ctx,
        &shared_gate_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_up_gpu = weight_bytes(
        &ctx,
        &shared_up_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_down_gpu = weight_bytes(
        &ctx,
        &shared_down_bytes,
        vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
        GgmlType::Q8_0,
    );
    let weights = Qwen4ExpMoeMetalWeights {
        geometry: g,
        router: &router_gpu,
        routed_gate: &routed_gate_gpu,
        routed_up: &routed_up_gpu,
        routed_down: &routed_down_gpu,
        shared_router: &shared_router_gpu,
        shared_gate: &shared_gate_gpu,
        shared_up: &shared_up_gpu,
        shared_down: &shared_down_gpu,
    };
    let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
    let exported = MetalTensor::zeros_f32(&ctx, vec![g.hidden_size as u64]).unwrap();

    let mutable_router = tensor_f32(
        &ctx,
        router,
        vec![g.hidden_size as u64, g.expert_count as u64],
    );
    let bad_weights = Qwen4ExpMoeMetalWeights {
        router: &mutable_router,
        ..weights
    };
    let bad_command = ctx.queue.commandBuffer().unwrap();
    let bad_encoder = KernelEncoder::begin(&bad_command);
    let error = encode_qwen4exp_moe(&ctx, &bad_encoder, &input_gpu, bad_weights, &mut workspace)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("read-only weight provenance"));
    bad_encoder.end();

    let mut padded_input = vec![0.0_f32; g.hidden_size + 1];
    padded_input[1..].copy_from_slice(input);
    let padded_input_gpu = tensor_f32(&ctx, &padded_input, vec![padded_input.len() as u64]);
    let misaligned_input = padded_input_gpu.view_subrange(1, vec![g.hidden_size as u64]);
    let misaligned_command = ctx.queue.commandBuffer().unwrap();
    let misaligned_encoder = KernelEncoder::begin(&misaligned_command);
    let error = encode_qwen4exp_moe(
        &ctx,
        &misaligned_encoder,
        &misaligned_input,
        weights,
        &mut workspace,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("not 16-byte aligned"));
    misaligned_encoder.end();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let read = encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
    assert_eq!(read.output().dtype(), GgmlType::F32);
    assert_eq!(read.output().n_elements(), g.hidden_size as u64);

    let wrong_command = ctx.queue.commandBuffer().unwrap();
    let wrong_encoder = KernelEncoder::begin(&wrong_command);
    assert!(
        read.output()
            .encode_copy_to(&ctx, &wrong_encoder, &exported)
            .is_err()
    );
    wrong_encoder.end();
    read.output()
        .encode_copy_to(&ctx, &encoder, &exported)
        .unwrap();
    drop(read);
    encoder.end();
    assert!(workspace.release_after().is_err());
    command.commit();
    workspace.release_after().unwrap();

    assert_close(
        "router logits",
        &read_f32(&workspace.router_logits),
        expected_logits,
        2e-5,
        2e-5,
    );
    assert_eq!(read_i32(&workspace.topk_ids), oracle.topk_ids);
    assert_close(
        "top-k weights",
        &read_f32(&workspace.topk_weights),
        expected_weights,
        2e-6,
        2e-6,
    );
    assert_close(
        "shared gate",
        &read_f32(&workspace.shared_gate),
        &[expected_shared_gate],
        2e-6,
        2e-6,
    );
    assert_close(
        "routed inner",
        &read_f32(&workspace.routed_inner),
        &expected.routed_inner,
        2e-4,
        2e-4,
    );
    assert_close(
        "shared inner",
        &read_f32(&workspace.shared_inner),
        &expected.shared_inner,
        2e-5,
        2e-4,
    );
    assert_close(
        "shared output",
        &read_f32(&workspace.shared_output),
        &expected.shared_output,
        2e-5,
        2e-4,
    );
    assert_close(
        "MoE output",
        &read_f32(&workspace.output),
        &expected.output,
        3e-4,
        3e-4,
    );
    assert_close(
        "copied MoE output",
        &read_f32(&exported),
        &expected.output,
        3e-4,
        3e-4,
    );
    assert!(!workspace.is_poisoned());
}

#[test]
fn iq3_xxs_iq4_nl_path_is_zero_safe_and_abandonable() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (oracle, values) = parse_oracle();
    let g = geometry(&oracle);
    let input = oracle_section(&oracle, &values, "input");
    let router = oracle_section(&oracle, &values, "router");
    let shared_router = oracle_section(&oracle, &values, "shared_router");
    let expected_weights = oracle_section(&oracle, &values, "topk_weights");
    let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];
    let routed_elements = g.hidden_size * g.routed_intermediate_size * g.expert_count;
    let routed_bytes = vec![0_u8; routed_elements / 256 * 98];
    let routed_down_bytes = synthetic_iq4_nl_bank(
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        60_013,
    );
    let shared_gate_bytes =
        synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 70_001);
    let shared_up_bytes = synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 80_021);
    let shared_down_bytes =
        synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 90_007);
    let expected = cpu_quantized_moe(CpuQuantizedInputs {
        geometry: g,
        input,
        topk_ids: &oracle.topk_ids,
        topk_weights: expected_weights,
        shared_gate_scalar: expected_shared_gate,
        routed_gate: &routed_bytes,
        routed_up: &routed_bytes,
        routed_dtype: GgmlType::IQ3_XXS,
        routed_down: &routed_down_bytes,
        routed_down_dtype: GgmlType::IQ4_NL,
        shared_gate: &shared_gate_bytes,
        shared_up: &shared_up_bytes,
        shared_down: &shared_down_bytes,
    });
    assert!(expected.routed_inner.iter().all(|&value| value == 0.0));
    assert!(
        expected
            .routed_expert_output
            .iter()
            .all(|&value| value == 0.0)
    );

    let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
    let router_gpu = weight_f32(
        &ctx,
        router,
        vec![g.hidden_size as u64, g.expert_count as u64],
    );
    let routed_gate_gpu = weight_bytes(
        &ctx,
        &routed_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ3_XXS,
    );
    let routed_up_gpu = weight_bytes(
        &ctx,
        &routed_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ3_XXS,
    );
    let routed_down_gpu = weight_bytes(
        &ctx,
        &routed_down_bytes,
        vec![
            g.routed_intermediate_size as u64,
            g.hidden_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ4_NL,
    );
    let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
    let shared_gate_gpu = weight_bytes(
        &ctx,
        &shared_gate_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_up_gpu = weight_bytes(
        &ctx,
        &shared_up_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_down_gpu = weight_bytes(
        &ctx,
        &shared_down_bytes,
        vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
        GgmlType::Q8_0,
    );
    let weights = Qwen4ExpMoeMetalWeights {
        geometry: g,
        router: &router_gpu,
        routed_gate: &routed_gate_gpu,
        routed_up: &routed_up_gpu,
        routed_down: &routed_down_gpu,
        shared_router: &shared_router_gpu,
        shared_gate: &shared_gate_gpu,
        shared_up: &shared_up_gpu,
        shared_down: &shared_down_gpu,
    };
    let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();

    let concurrent_command = ctx.queue.commandBuffer().unwrap();
    let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
    assert!(
        encode_qwen4exp_moe(
            &ctx,
            &concurrent_encoder,
            &input_gpu,
            weights,
            &mut workspace,
        )
        .is_err()
    );
    concurrent_encoder.end();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let read = encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
    drop(read);
    encoder.end();
    command.commit();
    workspace.release_after().unwrap();
    assert!(
        read_f32(&workspace.routed_inner)
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        read_f32(&workspace.routed_expert_output)
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_close(
        "IQ3/IQ4 MoE output",
        &read_f32(&workspace.output),
        &expected.output,
        2e-5,
        2e-4,
    );

    let abandoned_command = ctx.queue.commandBuffer().unwrap();
    let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
    let read = encode_qwen4exp_moe(
        &ctx,
        &abandoned_encoder,
        &input_gpu,
        weights,
        &mut workspace,
    )
    .unwrap();
    drop(read);
    abandoned_encoder.end();
    unsafe { workspace.abandon_uncommitted() }.unwrap();
    drop(abandoned_command);
    workspace.reset().unwrap();

    let reuse_command = ctx.queue.commandBuffer().unwrap();
    let reuse_encoder = KernelEncoder::begin(&reuse_command);
    let read =
        encode_qwen4exp_moe(&ctx, &reuse_encoder, &input_gpu, weights, &mut workspace).unwrap();
    drop(read);
    reuse_encoder.end();
    reuse_command.commit();
    workspace.release_after().unwrap();
    assert_close(
        "reused IQ3/IQ4 MoE output",
        &read_f32(&workspace.output),
        &expected.output,
        2e-5,
        2e-4,
    );

    workspace.state_poisoned = true;
    let poisoned_command = ctx.queue.commandBuffer().unwrap();
    let poisoned_encoder = KernelEncoder::begin(&poisoned_command);
    assert!(
        encode_qwen4exp_moe(&ctx, &poisoned_encoder, &input_gpu, weights, &mut workspace,).is_err()
    );
    poisoned_encoder.end();
    workspace.reset().unwrap();
    assert!(!workspace.is_poisoned());
}

#[test]
fn iq3_xxs_iq4_nl_and_q8_paths_match_nonzero_cpu_dequant() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (oracle, values) = parse_oracle();
    let g = geometry(&oracle);
    let input = oracle_section(&oracle, &values, "input");
    let router = oracle_section(&oracle, &values, "router");
    let shared_router = oracle_section(&oracle, &values, "shared_router");
    let expected_weights = oracle_section(&oracle, &values, "topk_weights");
    let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];

    let gate_values = synthetic_f32_bank(
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        101,
    );
    let up_values = synthetic_f32_bank(
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        211,
    );
    let gate_bytes = quantize_rows(&gate_values, GgmlType::IQ3_XXS, g.hidden_size);
    let up_bytes = quantize_rows(&up_values, GgmlType::IQ3_XXS, g.hidden_size);
    let down_iq4_bytes = synthetic_iq4_nl_bank(
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        60_013,
    );
    let down_q8_bytes = synthetic_q8_0_bank(
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        65_017,
    );
    let shared_gate_bytes =
        synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 70_001);
    let shared_up_bytes = synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 80_021);
    let shared_down_bytes =
        synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 90_007);

    let expected_iq4 = cpu_quantized_moe(CpuQuantizedInputs {
        geometry: g,
        input,
        topk_ids: &oracle.topk_ids,
        topk_weights: expected_weights,
        shared_gate_scalar: expected_shared_gate,
        routed_gate: &gate_bytes,
        routed_up: &up_bytes,
        routed_dtype: GgmlType::IQ3_XXS,
        routed_down: &down_iq4_bytes,
        routed_down_dtype: GgmlType::IQ4_NL,
        shared_gate: &shared_gate_bytes,
        shared_up: &shared_up_bytes,
        shared_down: &shared_down_bytes,
    });
    let expected_q8 = cpu_quantized_moe(CpuQuantizedInputs {
        geometry: g,
        input,
        topk_ids: &oracle.topk_ids,
        topk_weights: expected_weights,
        shared_gate_scalar: expected_shared_gate,
        routed_gate: &gate_bytes,
        routed_up: &up_bytes,
        routed_dtype: GgmlType::IQ3_XXS,
        routed_down: &down_q8_bytes,
        routed_down_dtype: GgmlType::Q8_0,
        shared_gate: &shared_gate_bytes,
        shared_up: &shared_up_bytes,
        shared_down: &shared_down_bytes,
    });
    assert!(
        expected_iq4
            .routed_inner
            .iter()
            .any(|value| value.abs() > 1e-5)
    );
    assert!(
        expected_iq4
            .routed_expert_output
            .iter()
            .any(|value| value.abs() > 1e-5)
    );
    assert!(
        expected_q8
            .routed_expert_output
            .iter()
            .any(|value| value.abs() > 1e-5)
    );

    let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
    let router_gpu = weight_f32(
        &ctx,
        router,
        vec![g.hidden_size as u64, g.expert_count as u64],
    );
    let gate_gpu = weight_bytes(
        &ctx,
        &gate_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ3_XXS,
    );
    let up_gpu = weight_bytes(
        &ctx,
        &up_bytes,
        vec![
            g.hidden_size as u64,
            g.routed_intermediate_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ3_XXS,
    );
    let down_iq4_gpu = weight_bytes(
        &ctx,
        &down_iq4_bytes,
        vec![
            g.routed_intermediate_size as u64,
            g.hidden_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::IQ4_NL,
    );
    let down_q8_gpu = weight_bytes(
        &ctx,
        &down_q8_bytes,
        vec![
            g.routed_intermediate_size as u64,
            g.hidden_size as u64,
            g.expert_count as u64,
        ],
        GgmlType::Q8_0,
    );
    let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
    let shared_gate_gpu = weight_bytes(
        &ctx,
        &shared_gate_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_up_gpu = weight_bytes(
        &ctx,
        &shared_up_bytes,
        vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
        GgmlType::Q8_0,
    );
    let shared_down_gpu = weight_bytes(
        &ctx,
        &shared_down_bytes,
        vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
        GgmlType::Q8_0,
    );
    let iq4_weights = Qwen4ExpMoeMetalWeights {
        geometry: g,
        router: &router_gpu,
        routed_gate: &gate_gpu,
        routed_up: &up_gpu,
        routed_down: &down_iq4_gpu,
        shared_router: &shared_router_gpu,
        shared_gate: &shared_gate_gpu,
        shared_up: &shared_up_gpu,
        shared_down: &shared_down_gpu,
    };
    let q8_weights = Qwen4ExpMoeMetalWeights {
        routed_down: &down_q8_gpu,
        ..iq4_weights
    };

    let mut iq4_workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
    let iq4_command = ctx.queue.commandBuffer().unwrap();
    let iq4_encoder = KernelEncoder::begin(&iq4_command);
    let read = encode_qwen4exp_moe(
        &ctx,
        &iq4_encoder,
        &input_gpu,
        iq4_weights,
        &mut iq4_workspace,
    )
    .unwrap();
    drop(read);
    iq4_encoder.end();
    iq4_command.commit();
    iq4_workspace.release_after().unwrap();
    assert_close(
        "nonzero IQ3 routed inner",
        &read_f32(&iq4_workspace.routed_inner),
        &expected_iq4.routed_inner,
        3e-4,
        3e-4,
    );
    assert_close(
        "nonzero IQ4_NL expert output",
        &read_f32(&iq4_workspace.routed_expert_output),
        &expected_iq4.routed_expert_output,
        5e-4,
        5e-4,
    );
    assert_close(
        "nonzero IQ3/IQ4 MoE output",
        &read_f32(&iq4_workspace.output),
        &expected_iq4.output,
        7e-4,
        7e-4,
    );

    let mut q8_workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
    let q8_command = ctx.queue.commandBuffer().unwrap();
    let q8_encoder = KernelEncoder::begin(&q8_command);
    let read =
        encode_qwen4exp_moe(&ctx, &q8_encoder, &input_gpu, q8_weights, &mut q8_workspace).unwrap();
    drop(read);
    q8_encoder.end();
    q8_command.commit();
    q8_workspace.release_after().unwrap();
    assert_close(
        "nonzero IQ3/Q8 MoE output",
        &read_f32(&q8_workspace.output),
        &expected_q8.output,
        7e-4,
        7e-4,
    );
}

fn stable_topk(logits: &[f32], top_k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = (0..logits.len()).collect::<Vec<_>>();
    ids.sort_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    ids.truncate(top_k);
    let maximum = logits[ids[0]];
    let exponentials = ids
        .iter()
        .map(|&expert| (logits[expert] - maximum).exp())
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f32>();
    (
        ids.into_iter().map(|expert| expert as i32).collect(),
        exponentials.into_iter().map(|value| value / sum).collect(),
    )
}

fn gguf_dequant(gguf: &GgufFile, name: &str) -> Vec<f32> {
    let desc = gguf.find(name).unwrap();
    crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
}

fn gguf_dequant_expert(gguf: &GgufFile, name: &str, expert: usize) -> Vec<f32> {
    let desc = gguf.find(name).unwrap();
    assert_eq!(desc.shape.len(), 3);
    let n_in = desc.shape[0] as usize;
    let n_out = desc.shape[1] as usize;
    let (block_elements, block_bytes) = desc.dtype.storage_layout().unwrap();
    assert!((n_in as u64).is_multiple_of(block_elements));
    let row_bytes = n_in / block_elements as usize * block_bytes as usize;
    let expert_bytes = n_out * row_bytes;
    let start = expert * expert_bytes;
    let bytes = gguf.try_slice(desc).unwrap();
    let expert_desc = TensorDesc {
        name: format!("{name}.expert.{expert}"),
        shape: vec![n_in as u64, n_out as u64],
        dtype: desc.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_bytes as u64,
    };
    crate::codec::dequant_to_f32(&expert_desc, &bytes[start..start + expert_bytes]).unwrap()
}

fn real_input(router: &[f32], g: Qwen4ExpMoeMetalGeometry) -> (Vec<f32>, Vec<f32>) {
    for seed in 0..64 {
        let input = (0..g.hidden_size)
            .map(|index| {
                let raw = ((index * 37 + index / 11 * 5 + seed * 17 + 3) % 257) as f32;
                (raw - 128.0) * 0.00075
            })
            .collect::<Vec<_>>();
        let logits = mat_vec(router, &input, g.hidden_size, g.expert_count);
        let (ids, _) = stable_topk(&logits, g.experts_per_token);
        let mut sorted = logits.clone();
        sorted.sort_by(|left, right| right.total_cmp(left));
        let boundary_margin = sorted[g.experts_per_token - 1] - sorted[g.experts_per_token];
        if ids.iter().any(|&expert| expert >= 256) && boundary_margin > 1e-4 {
            return (input, logits);
        }
    }
    panic!("could not find a stable real-weight routing input with a high expert ID")
}

fn real_cpu_oracle(
    gguf: &GgufFile,
    layer: u32,
    g: Qwen4ExpMoeMetalGeometry,
) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>, f32, CpuMoeResult) {
    let prefix = format!("blk.{layer}");
    let router = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_inp.weight"));
    let (input, logits) = real_input(&router, g);
    let (ids, weights) = stable_topk(&logits, g.experts_per_token);
    let shared_router = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_inp_shexp.weight"));
    let shared_gate_scalar = 1.0
        / (1.0
            + (-shared_router
                .iter()
                .zip(&input)
                .map(|(weight, input)| weight * input)
                .sum::<f32>())
            .exp());

    let gate_desc = gguf
        .find(&format!("{prefix}.ffn_gate_exps.weight"))
        .unwrap();
    let down_desc = gguf
        .find(&format!("{prefix}.ffn_down_exps.weight"))
        .unwrap();
    let mut routed_inner = vec![0.0; g.experts_per_token * g.routed_intermediate_size];
    for (slot, &expert) in ids.iter().enumerate() {
        let gate = gguf_dequant_expert(
            gguf,
            &format!("{prefix}.ffn_gate_exps.weight"),
            expert as usize,
        );
        let up = gguf_dequant_expert(
            gguf,
            &format!("{prefix}.ffn_up_exps.weight"),
            expert as usize,
        );
        let gate = mat_vec(&gate, &input, g.hidden_size, g.routed_intermediate_size);
        let up = mat_vec(&up, &input, g.hidden_size, g.routed_intermediate_size);
        for row in 0..g.routed_intermediate_size {
            routed_inner[slot * g.routed_intermediate_size + row] = silu(gate[row]) * up[row];
        }
    }
    let mut routed_expert_output = vec![0.0; g.experts_per_token * g.hidden_size];
    let mut output = vec![0.0; g.hidden_size];
    for (slot, (&expert, &route_weight)) in ids.iter().zip(&weights).enumerate() {
        let down = gguf_dequant_expert(
            gguf,
            &format!("{prefix}.ffn_down_exps.weight"),
            expert as usize,
        );
        let expert_output = mat_vec(
            &down,
            &routed_inner
                [slot * g.routed_intermediate_size..(slot + 1) * g.routed_intermediate_size],
            g.routed_intermediate_size,
            g.hidden_size,
        );
        routed_expert_output[slot * g.hidden_size..(slot + 1) * g.hidden_size]
            .copy_from_slice(&expert_output);
        for (output, expert_value) in output.iter_mut().zip(expert_output) {
            *output += route_weight * expert_value;
        }
    }

    let shared_gate_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_shexp.weight"));
    let shared_up_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_up_shexp.weight"));
    let shared_down_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_down_shexp.weight"));
    let shared_gate = mat_vec(
        &shared_gate_weight,
        &input,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_up = mat_vec(
        &shared_up_weight,
        &input,
        g.hidden_size,
        g.shared_intermediate_size,
    );
    let shared_inner = shared_gate
        .into_iter()
        .zip(shared_up)
        .map(|(gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let shared_output = mat_vec(
        &shared_down_weight,
        &shared_inner,
        g.shared_intermediate_size,
        g.hidden_size,
    );
    for (output, &shared) in output.iter_mut().zip(&shared_output) {
        *output += shared_gate_scalar * shared;
    }
    assert!(matches!(
        gate_desc.dtype,
        GgmlType::IQ3_XXS | GgmlType::IQ4_XS
    ));
    assert!(matches!(down_desc.dtype, GgmlType::IQ4_NL | GgmlType::Q8_0));
    (
        input,
        logits,
        ids,
        weights,
        shared_gate_scalar,
        CpuMoeResult {
            routed_inner,
            routed_expert_output,
            shared_inner,
            shared_output,
            output,
        },
    )
}

fn assert_similarity(label: &str, actual: &[f32], expected: &[f32], max_abs: f32, cosine: f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    let observed_max = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| *actual as f64 * *expected as f64)
        .sum::<f64>();
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
    let observed_cosine = dot / (actual_norm * expected_norm).max(1e-30);
    eprintln!("[{label}] max_abs={observed_max:.3e} cosine={observed_cosine:.9}");
    assert!(observed_max <= max_abs, "{label} max_abs={observed_max}");
    assert!(
        observed_cosine >= cosine,
        "{label} cosine={observed_cosine}"
    );
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_MOE_GGUF to the pinned first release shard"]
fn released_layers_two_and_three_packed_motor_match_serial_rows() {
    const TOKENS: usize = 8;
    let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = MetalContext::new().expect("initialize Metal");
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
    let admitted = plan.admit(ctx.memory_signals()).unwrap();
    let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
    let metal_weights = realized.weights();
    let geometry = Qwen4ExpMoeMetalGeometry::from_config(metal_weights.config()).unwrap();
    for (layer, gate_dtype, down_dtype, gate_kernel, down_kernel) in [
        (
            2_u32,
            GgmlType::IQ4_XS,
            GgmlType::Q8_0,
            "kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16",
            "kernel_moe_down_q8_0_f32_grouped_slots",
        ),
        (
            3_u32,
            GgmlType::IQ3_XXS,
            GgmlType::IQ4_NL,
            "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16",
            "kernel_moe_down_iq4_nl_f32_grouped_slots",
        ),
    ] {
        let weights = Qwen4ExpMoeMetalWeights::bind(metal_weights, layer).unwrap();
        assert_eq!(weights.routed_gate.dtype, gate_dtype);
        assert_eq!(weights.routed_down.dtype, down_dtype);

        let router = gguf_dequant(&gguf, &format!("blk.{layer}.ffn_gate_inp.weight"));
        let mut inputs = Vec::with_capacity(TOKENS * geometry.hidden_size);
        for seed in 0..512 {
            let input = (0..geometry.hidden_size)
                .map(|index| {
                    let raw = ((index * 37 + index / 11 * 5 + seed * 17 + 3) % 257) as f32;
                    (raw - 128.0) * 0.000_75
                })
                .collect::<Vec<_>>();
            let logits = mat_vec(&router, &input, geometry.hidden_size, geometry.expert_count);
            let (ids, _) = stable_topk(&logits, geometry.experts_per_token);
            let mut sorted = logits.clone();
            sorted.sort_by(|left, right| right.total_cmp(left));
            let margin =
                sorted[geometry.experts_per_token - 1] - sorted[geometry.experts_per_token];
            if ids.iter().any(|&expert| expert >= 256) && margin > 1e-4 {
                inputs.extend(input);
                if inputs.len() == TOKENS * geometry.hidden_size {
                    break;
                }
            }
        }
        assert_eq!(inputs.len(), TOKENS * geometry.hidden_size);
        let serial = serial_packed_moe_trace(&ctx, geometry, weights, &inputs, TOKENS);
        assert!(serial.topk_ids.iter().any(|&expert| expert >= 256));
        let input = tensor_f32(
            &ctx,
            &inputs,
            vec![geometry.hidden_size as u64, TOKENS as u64],
        );
        let scratch = Qwen4ExpMoePackedMotorScratch::new(&ctx, geometry, TOKENS).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let output = unsafe {
            encode_qwen4exp_moe_packed_motor(&ctx, &encoder, &input, weights, &scratch, TOKENS)
        }
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(
            census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>(),
            vec![
                "kernel_mat_mat_f32_f32",
                "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
                "kernel_moe_route_bucket_slots_f32",
                gate_kernel,
                down_kernel,
                "kernel_moe_weighted_sum_packed_f32",
                "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
                "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
                "kernel_silu_mul_f32",
                "kernel_mat_mat_q8_0_f32",
                "kernel_axpy_rowwise_f32",
            ],
            "released packed layer-{layer} route: {census:#?}"
        );
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let views = scratch.views(TOKENS).unwrap();
        assert_eq!(read_i32(&views.topk_ids), serial.topk_ids);
        let (routed_inner_abs, routed_inner_cosine, shared_output_abs, output_abs, output_cosine) =
            if layer == 2 {
                (5e-6, 0.999_999_9, 8e-7, 5e-7, 0.999_999_9)
            } else {
                (4e-6, 0.999_999_9, 6e-7, 4e-7, 0.999_999_9)
            };
        let mut stages = vec![
            (
                format!("released packed layer-{layer} router logits"),
                read_f32(&views.router_logits),
                serial.router_logits.as_slice(),
                geometry.expert_count,
                6e-7,
                0.999_999_99,
            ),
            (
                format!("released packed layer-{layer} top-k weights"),
                read_f32(&views.topk_weights),
                serial.topk_weights.as_slice(),
                geometry.experts_per_token,
                7e-8,
                0.999_999_99,
            ),
            (
                format!("released packed layer-{layer} shared scale"),
                read_f32(&views.shared_scale),
                serial.shared_scale.as_slice(),
                1,
                1e-7,
                0.999_999_99,
            ),
            (
                format!("released packed layer-{layer} routed inner"),
                read_f32(&views.routed_inner),
                serial.routed_inner.as_slice(),
                geometry.experts_per_token * geometry.routed_intermediate_size,
                routed_inner_abs,
                routed_inner_cosine,
            ),
            (
                format!("released packed layer-{layer} shared inner"),
                read_f32(&views.shared_inner),
                serial.shared_inner.as_slice(),
                geometry.shared_intermediate_size,
                1.5e-6,
                0.999_999_9,
            ),
            (
                format!("released packed layer-{layer} shared output"),
                read_f32(&views.shared_output),
                serial.shared_output.as_slice(),
                geometry.hidden_size,
                shared_output_abs,
                0.999_999_9,
            ),
            (
                format!("released packed layer-{layer} output"),
                read_f32(&output),
                serial.output.as_slice(),
                geometry.hidden_size,
                output_abs,
                output_cosine,
            ),
        ];
        if down_dtype == GgmlType::IQ4_NL {
            stages.insert(
                4,
                (
                    format!("released packed layer-{layer} routed expert output"),
                    read_f32(&views.routed_expert_output),
                    serial.routed_expert_output.as_slice(),
                    geometry.experts_per_token * geometry.hidden_size,
                    8e-7,
                    0.999_999_85,
                ),
            );
        }
        for (stage, actual, expected, width, max_abs, cosine) in stages {
            assert_tokenwise_similarity(&stage, &actual, expected, width, TOKENS, max_abs, cosine);
        }
    }
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_MOE_GGUF to the pinned first release shard"]
fn released_layers_two_through_four_match_selected_expert_cpu_oracles() {
    let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = MetalContext::new().expect("initialize Metal");
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
    let admitted = plan.admit(ctx.memory_signals()).unwrap();
    let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
    let metal_weights = realized.weights();
    let g = Qwen4ExpMoeMetalGeometry::from_config(metal_weights.config()).unwrap();
    let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();

    for layer in [2_u32, 3, 4] {
        let weights = Qwen4ExpMoeMetalWeights::bind(metal_weights, layer).unwrap();
        if layer == 2 {
            assert_eq!(weights.routed_gate.dtype, GgmlType::IQ4_XS);
            assert_eq!(weights.routed_down.dtype, GgmlType::Q8_0);
        } else if layer == 3 {
            assert_eq!(weights.routed_gate.dtype, GgmlType::IQ3_XXS);
            assert_eq!(weights.routed_down.dtype, GgmlType::IQ4_NL);
        } else {
            assert_eq!(weights.routed_gate.dtype, GgmlType::IQ3_XXS);
            assert_eq!(weights.routed_down.dtype, GgmlType::Q8_0);
        }
        let (input, logits, ids, route_weights, shared_gate, expected) =
            real_cpu_oracle(&gguf, layer, g);
        assert!(ids.iter().any(|&expert| expert >= 256));
        let input_gpu = tensor_f32(&ctx, &input, vec![g.hidden_size as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();

        assert_close(
            &format!("layer {layer} router logits"),
            &read_f32(&workspace.router_logits),
            &logits,
            2e-4,
            2e-4,
        );
        assert_eq!(read_i32(&workspace.topk_ids), ids);
        assert_close(
            &format!("layer {layer} route weights"),
            &read_f32(&workspace.topk_weights),
            &route_weights,
            2e-5,
            2e-5,
        );
        assert_close(
            &format!("layer {layer} shared gate"),
            &read_f32(&workspace.shared_gate),
            &[shared_gate],
            2e-5,
            2e-5,
        );
        assert_similarity(
            &format!("layer {layer} routed inner"),
            &read_f32(&workspace.routed_inner),
            &expected.routed_inner,
            if layer == 2 { 2e-3 } else { 3e-2 },
            0.999,
        );
        if layer == 3 {
            assert_similarity(
                "layer 3 routed expert output",
                &read_f32(&workspace.routed_expert_output),
                &expected.routed_expert_output,
                5e-2,
                0.998,
            );
        }
        assert_similarity(
            &format!("layer {layer} shared inner"),
            &read_f32(&workspace.shared_inner),
            &expected.shared_inner,
            2e-3,
            0.999_99,
        );
        assert_similarity(
            &format!("layer {layer} shared output"),
            &read_f32(&workspace.shared_output),
            &expected.shared_output,
            5e-3,
            0.999_99,
        );
        assert_similarity(
            &format!("layer {layer} final output"),
            &read_f32(&workspace.output),
            &expected.output,
            if layer == 2 { 2e-2 } else { 8e-2 },
            0.998,
        );
    }
}
