use super::*;
use crate::qwen4exp_qsa::selected_value as product;

#[test]
fn selected_value_scope_is_released_full_selected_bands_only() {
    let config = Qwen4ExpConfig::flash_next_reference();
    let g = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 8256).unwrap();
    for start in [2051, 8192] {
        assert!(product::eligible(g, start, 32, true));
        assert!(!product::eligible(g, start, 32, false));
        for n in [0, 1, 7, 31, 33, 64] {
            assert!(!product::eligible(g, start, n, true));
        }
    }
    assert!(product::eligible(g, g.capacity - 32, 32, true));
    for start in [0, 1, 2050, 8240, usize::MAX] {
        assert!(!product::eligible(g, start, 32, true));
    }
    for (heads, kv, dim, budget, ratio) in [
        (16, 2, 256, 2048, 4),
        (24, 1, 256, 2048, 4),
        (24, 2, 128, 2048, 4),
        (24, 2, 256, 1024, 4),
        (24, 2, 256, 2048, 8),
    ] {
        let other = QwenSparseAttentionMetalGeometry {
            query_heads: heads,
            kv_heads: kv,
            head_dim: dim,
            token_budget: budget,
            ratio,
            ..g
        };
        assert!(!product::eligible(other, 2051, 32, true));
    }
    let oversized = QwenSparseAttentionMetalGeometry {
        capacity: i32::MAX as usize + 1,
        ..g
    };
    assert!(!product::eligible(oversized, 2051, 32, true));
}

#[test]
fn selected_value_allocation_and_capacity_contract() {
    let g = QwenSparseAttentionMetalGeometry::from_config(
        &Qwen4ExpConfig::flash_next_reference(),
        3,
        8256,
    )
    .unwrap();
    for capacity in [1, 7, 18, 31, 32, 2048] {
        let allocations = g
            .selected_packed_scratch_logical_allocations(capacity)
            .unwrap();
        assert_eq!(allocations.len(), 10);
        assert_eq!(allocations[9], 24 * capacity.min(32) * 4);
    }
    assert!(product::capacity_supported(32, 256, 0, 144, 32768));
    assert!(!product::capacity_supported(16, 256, 0, 144, 32768));
    assert!(!product::capacity_supported(32, 255, 0, 144, 32768));
    assert!(!product::capacity_supported(32, 256, 32768, 144, 32768));
    assert!(!product::capacity_supported(
        32,
        256,
        usize::MAX,
        144,
        usize::MAX
    ));
}

fn require_gpu_workflow() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_METAL_LEASE_WAIT"] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok("1"),
            "requires {name}=1 before startup"
        );
    }
}

fn admitted_fixture(ctx: &MetalContext) -> SelectedAttentionFixture {
    let g = selected_attention_test_geometry(4096);
    let _ = product::Pipelines::prepare(ctx).unwrap();
    for name in [
        "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
        "kernel_qwen4exp_qsa_attention_softmax_value_packed_f16",
    ] {
        ctx.pipeline(name).unwrap();
    }
    let mut logical = g.packed_scratch_logical_allocations(32).unwrap();
    logical.extend(g.selected_packed_scratch_logical_allocations(32).unwrap());
    logical.extend([
        g.index_head_dim * 4,
        g.index_head_dim * g.block_capacity() * 2,
        g.kv_width() * g.capacity * 2,
        g.kv_width() * g.capacity * 2,
        g.query_width() * 32 * 4,
    ]);
    let priced = logical
        .iter()
        .map(|&bytes| {
            ctx.price_shared_buffer_upper(bytes as u64)
                .unwrap()
                .priced_upper_bytes
        })
        .sum();
    let _allocation = ctx.begin_allocation_transaction();
    let before = ctx.memory_signals();
    let reserve = crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
    let cpu = logical.iter().map(|&b| b as u64).sum::<u64>() * 4;
    let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        priced, cpu, reserve, before, true,
    );
    assert!(admission.is_admitted(), "{admission:?}");
    let fixture = selected_attention_fixture(ctx, 32);
    let after = ctx.memory_signals();
    assert!(
        after
            .current_allocated_bytes
            .saturating_sub(before.current_allocated_bytes)
            <= priced
    );
    assert!(crate::metal::evaluate_metal_memory_admission(0, reserve, after, true).is_admitted());
    fixture
}

#[test]
#[ignore = "production lease; product N32/tail/flag-off/unsupported dispatch and scratch inventory"]
fn selected_value_product_dispatch_and_inventory() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = context() else { return };
    let fixture = admitted_fixture(&ctx);
    let g = fixture.geometry;
    let selected = fixture.scratch.selected.as_ref().unwrap();
    let prepared = product::Pipelines::prepare(&ctx).unwrap();
    assert_eq!(selected.attention_denominators.shape, [24, 32]);
    assert_eq!(selected.attention_denominators.n_bytes(), 3072);
    let inventory = dense_packed_scratch_tensors(&fixture.scratch);
    assert_eq!(inventory.len(), 19);
    let mut expected_bytes = g.packed_scratch_logical_allocations(32).unwrap();
    expected_bytes.extend(g.selected_packed_scratch_logical_allocations(32).unwrap());
    assert_eq!(
        inventory
            .iter()
            .map(|(_, tensor)| tensor.n_bytes() as usize)
            .collect::<Vec<_>>(),
        expected_bytes
    );
    assert_eq!(
        inventory
            .iter()
            .filter(|(_, tensor)| {
                Retained::as_ptr(&tensor.buffer)
                    == Retained::as_ptr(&selected.attention_denominators.buffer)
            })
            .count(),
        1
    );
    require_same_device(&ctx, &inventory).unwrap();
    require_disjoint(&inventory).unwrap();
    fill_i32_tensor(&selected.selected_count, 512).unwrap();
    fill_i32_tensor(&selected.selector_status, 0).unwrap();
    fill_i32_tensor(&selected.token_ids, 0).unwrap();
    for (n, enabled, available) in [
        (32, true, true),
        (31, true, true),
        (7, true, true),
        (1, true, true),
        (32, false, true),
        (32, true, false),
    ] {
        let packed = fixture.scratch.views(n).unwrap();
        let band = selected.views(g, 32, 32, n).unwrap();
        let candidate = if available {
            prepared
                .as_ref()
                .map(|p| (p, &selected.attention_denominators))
        } else {
            None
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        encode_attention_softmax_value_packed_route(
            &ctx,
            &encoder,
            &packed.query_gate_projection,
            &fixture.value_cache,
            &band.token_ids,
            &band.selected_count,
            &band.selector_status,
            &band.attention_logits,
            &packed.attention,
            g,
            2051,
            n,
            candidate,
            enabled,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).unwrap();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let expected = if !enabled {
            vec!["kernel_qwen4exp_qsa_attention_softmax_value_packed_f16"]
        } else if n == 32 && available && prepared.is_some() {
            vec![product::SOFTMAX, product::VALUE]
        } else {
            vec!["kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16"]
        };
        assert_eq!(
            census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
            expected
        );
        if expected.len() == 2 {
            assert_eq!(
                (
                    census[0].grid_width,
                    census[0].grid_height,
                    census[0].grid_depth
                ),
                (3, 2, 32)
            );
            assert_eq!(
                (
                    census[1].grid_width,
                    census[1].grid_height,
                    census[1].grid_depth
                ),
                (1, 2, 32)
            );
            assert!(census.iter().all(|r| r.tg_threads == 256));
        }
        assert!(read_f32(&packed.attention).iter().all(|v| v.is_finite()));
    }
}

#[test]
#[ignore = "production lease; malformed denominator refuses whole packet and value leaf before mutation"]
fn selected_value_product_rejects_before_packet_updates() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = context() else { return };
    let mut fixture = admitted_fixture(&ctx);
    let g = fixture.geometry;
    let packed = fixture.scratch.views(32).unwrap();
    let selected = fixture.scratch.selected.as_ref().unwrap();
    let saved = selected.attention_denominators.clone();
    let controls = (
        read_i32(&selected.visible_blocks),
        read_i32(&selected.selected_count),
        read_i32(&selected.selector_status),
    );
    let cache = read_tensor_bytes(&fixture.value_cache);
    let logits = read_tensor_bytes(&selected.attention_logits);
    let prepared = product::Pipelines::prepare(&ctx).unwrap();
    for fault in 0..5 {
        let mut bad = saved.clone();
        match fault {
            0 => bad.shape[0] = 23,
            1 => bad.offset += 1,
            2 => bad.offset = u64::MAX - 3,
            3 => bad.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
            _ => bad = packed.query_gate_projection.view_subrange(0, vec![24, 32]),
        }
        fixture
            .scratch
            .selected
            .as_mut()
            .unwrap()
            .attention_denominators = bad;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        assert!(
            encode_selected_attention_packet(
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
                2051,
                0,
                32
            )
            .is_err()
        );
        assert!(crate::metal::dispatch_census_take().is_empty());
        if let Some(pipelines) = prepared.as_ref() {
            let selected = fixture.scratch.selected.as_ref().unwrap();
            crate::metal::dispatch_census_begin();
            assert!(
                encode_attention_softmax_value_packed_route(
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
                    2051,
                    32,
                    Some((pipelines, &selected.attention_denominators)),
                    true
                )
                .is_err()
            );
            assert!(crate::metal::dispatch_census_take().is_empty());
        }
        encoder.end(); // Never submit a rejected packet.
        let selected = fixture.scratch.selected.as_ref().unwrap();
        assert_eq!(
            (
                read_i32(&selected.visible_blocks),
                read_i32(&selected.selected_count),
                read_i32(&selected.selector_status)
            ),
            controls
        );
        assert_eq!(read_tensor_bytes(&selected.attention_logits), logits);
        assert_eq!(read_tensor_bytes(&fixture.value_cache), cache);
    }
    fixture
        .scratch
        .selected
        .as_mut()
        .unwrap()
        .attention_denominators = saved;
}
