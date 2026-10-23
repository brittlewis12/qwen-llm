use super::*;

const PAGE: usize = 16_384;
const M5_MAX_BUFFER_LENGTH: usize = 179_357_827_072;

fn preflight_signals() -> metal::MetalMemorySignals {
    metal::MetalMemorySignals {
        recommended_max_bytes: 1 << 40,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
        host_budget: Some(metal::HostMemoryBudget {
            physical_bytes: 1 << 40,
            conservative_available_bytes: 1 << 40,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    }
}

#[test]
fn cold_preflight_uses_validated_small_metadata_without_import_or_new_maps() {
    use crate::deepseek_v41::forward::{PrefillExecution, preflight};
    for engram in [false, true] {
        let g = fixture(1, GgmlType::F32, 2, engram);
        assert!(g.shards[0].mmap_len() < 4 * 1024 * 1024);
        let refs = Arc::strong_count(&g.shards[0].mmap);
        let storage = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
        let cold = preflight::preflight_with_pricing(
            &g,
            storage.clone(),
            8,
            16,
            &[
                PrefillExecution::FullForward,
                PrefillExecution::ShrinkingCed,
            ],
            334_226_432,
            true,
            &Ok,
            preflight_signals(),
        )
        .unwrap();
        assert_eq!(cold.resident_storage, storage);
        assert_eq!(cold.caller_cpu_bytes, 334_226_432);
        assert_eq!(Arc::strong_count(&g.shards[0].mmap), refs);
        assert!(cold.ced.is_none()); // One-layer metadata has no decoder split.
        assert_eq!(
            cold.runtime_metal_bytes,
            cold.request.metal_bytes + cold.full_incremental_metal_bytes
        );
        assert_eq!(
            cold.runtime_cpu_bytes,
            cold.request.cpu_bytes + cold.full_incremental_cpu_bytes
        );
        assert_eq!(
            cold.required_metal_bytes,
            storage.metal_view_bytes + cold.runtime_metal_bytes
        );
        assert_eq!(
            cold.required_cpu_bytes,
            storage.private_shadow_bytes
                + cold.binding_cpu_bytes
                + cold.runtime_cpu_bytes
                + 334_226_432
        );
        let p = plan(&g).unwrap();
        assert_ne!(
            cold.request,
            p.config.estimate_request_memory_with_gpu_step(16).unwrap()
        );
    }
}

#[test]
fn combined_preflight_refuses_before_injected_import_with_typed_telemetry_errors() {
    use crate::deepseek_v41::forward::{PrefillExecution, preflight};
    let g = fixture(1, GgmlType::F32, 2, false);
    let storage = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
    let run = |signals| {
        preflight::preflight_with_pricing(
            &g,
            storage.clone(),
            8,
            16,
            &[PrefillExecution::FullForward],
            334_226_432,
            false,
            &Ok,
            signals,
        )
    };
    let admitted = run(preflight_signals()).unwrap();
    let total = admitted.required_metal_bytes + admitted.required_cpu_bytes;
    let mut short = preflight_signals();
    short
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = total - 1;
    // Each side fits alone, but the preweight combined host exposure does not.
    assert!(
        metal::evaluate_metal_memory_admission_with_cpu_bytes(
            admitted.required_metal_bytes,
            0,
            0,
            short,
            true
        )
        .is_admitted()
    );
    assert!(
        metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            admitted.required_cpu_bytes,
            0,
            short,
            true
        )
        .is_admitted()
    );
    let mut missing = preflight_signals();
    missing.host_budget = None;
    let mut invalid = preflight_signals();
    invalid.recommended_max_bytes = 0;
    let mut invalid_host = preflight_signals();
    invalid_host.host_budget.as_mut().unwrap().physical_bytes = 0;
    let mut missing_process = preflight_signals();
    missing_process.process_limit_remaining_bytes = None;
    for signals in [short, missing, invalid, invalid_host, missing_process] {
        let called = std::cell::Cell::new(false);
        let outcome = run(signals).map(|_| {
            called.set(true);
        });
        let error = outcome.unwrap_err();
        assert!(error.as_memory_admission().is_some(), "{error}");
        assert!(!called.get(), "weight import callback ran after refusal");
    }
}

#[test]
fn cold_preflight_rejects_context_overflow_and_invalid_projection_metadata() {
    use crate::deepseek_v41::forward::{PrefillExecution, preflight};
    let mut g = fixture(1, GgmlType::F32, 2, true);
    let storage = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
    for (prompt, capacity, caller) in [(0, 16, 0), (17, 16, 0), (8, 1048577, 0), (8, 16, u64::MAX)]
    {
        assert!(
            preflight::preflight_with_pricing(
                &g,
                storage.clone(),
                prompt,
                capacity,
                &[PrefillExecution::FullForward],
                caller,
                false,
                &Ok,
                preflight_signals(),
            )
            .is_err()
        );
    }
    let tensor = Arc::get_mut(&mut g)
        .unwrap()
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.engram_kv.weight")
        .unwrap();
    tensor.shape[0] = 32;
    assert!(
        preflight::preflight_with_pricing(
            &g,
            storage,
            8,
            16,
            &[PrefillExecution::FullForward],
            0,
            false,
            &Ok,
            preflight_signals(),
        )
        .is_err()
    );
}

#[test]
fn bounded_gpu_preflight_request_matches_loaded_metadata_estimator() {
    use crate::deepseek_v41::forward::{Forward, PrefillExecution, preflight_cold_prefill};
    let ctx = MetalContext::new().unwrap();
    for engram in [false, true] {
        let g = fixture(1, GgmlType::F32, 2, engram);
        let cold =
            preflight_cold_prefill(&ctx, &g, 8, 16, &[PrefillExecution::FullForward], 0).unwrap();
        let weights = ResidentWeights::load(&ctx, Arc::clone(&g)).unwrap();
        let model =
            Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
                .unwrap();
        assert_eq!(
            cold.request,
            model.estimate_request_memory_with_gpu_step(16).unwrap()
        );
    }
}

fn check_counts(g: &GgufFile, max_buffer_length: usize) -> ResidentStoragePlan {
    let p = plan(g).unwrap();
    let result = resident_storage_plan(g, PAGE, max_buffer_length).unwrap();
    // This is the helper called by load, without a Metal context or import.
    assert_eq!(
        result,
        super::super::resident_storage::for_plan(g, &p, PAGE, max_buffer_length).unwrap()
    );
    assert_eq!(result.bytes, p.bytes);
    assert_eq!(result.page_size, PAGE);
    assert_eq!(result.max_buffer_length, max_buffer_length);
    assert_eq!(
        result.metal_view_bytes,
        result
            .windows
            .iter()
            .map(|w| w.window.length as u64)
            .sum::<u64>()
    );
    assert_eq!(
        result.private_shadow_bytes,
        result
            .windows
            .iter()
            .map(|w| w.mapping_kind.private_bytes())
            .sum::<u64>()
    );
    assert_eq!(
        result.shared_eof_window_count,
        result
            .windows
            .iter()
            .filter(|w| w.mapping_kind == ResidentMappingKind::SharedEof)
            .count()
    );
    assert_eq!(
        result.private_window_count,
        result
            .windows
            .iter()
            .filter(|w| matches!(w.mapping_kind, ResidentMappingKind::PrivateBoundary { .. }))
            .count()
    );
    for w in &result.windows {
        assert_eq!(w.window.length % PAGE, 0);
        assert!(w.window.length <= max_buffer_length);
        assert_eq!(
            w.mapping_kind,
            ResidentMappingKind::from(storage::mapping_kind(g, &w.window, PAGE).unwrap())
        );
    }
    for t in g.tensors.iter().filter(|t| resident_tensor(t, p.layers)) {
        assert!(result.windows.iter().any(|w| {
            w.window.shard_idx == t.shard_idx
                && w.window.mmap_offset <= t.data_offset
                && t.data_offset + t.n_bytes <= w.window.mmap_offset + w.window.length as u64
        }));
    }
    result
}

#[test]
fn metadata_storage_plan_shared_and_eof_counts_match_loader_helper() {
    for pad_end in [true, false] {
        let (kv, tensors) = fixture_parts(1, GgmlType::F32, 2, false);
        let g = write_fixture_layout(kv, tensors, pad_end);
        let refs = Arc::strong_count(&g.shards[0].mmap);
        let result = check_counts(&g, M5_MAX_BUFFER_LENGTH);
        assert_eq!(Arc::strong_count(&g.shards[0].mmap), refs);
        assert_eq!(result.windows.len(), 1);
        assert_eq!(result.private_shadow_bytes, 0);
        assert_eq!(result.private_window_count, 0);
        assert_eq!(result.shared_eof_window_count, usize::from(!pad_end));
        assert_eq!(
            result.windows[0].mapping_kind,
            if pad_end {
                ResidentMappingKind::OriginalShared
            } else {
                ResidentMappingKind::SharedEof
            }
        );
        let start = g.tensors.iter().map(|t| t.data_offset).min().unwrap();
        assert_eq!(
            result.metal_view_bytes,
            g.shards[0].mmap_len().next_multiple_of(PAGE) as u64 - start
        );
        assert!(result.metal_view_bytes > result.bytes.resident_bytes());
    }
}

#[test]
fn metadata_storage_plan_uses_supplied_maximum_and_is_order_independent() {
    let mut g = fixture(1, GgmlType::F32, 2, false);
    assert!(g.tensors.iter().all(|t| t.n_bytes <= PAGE as u64));
    let result = check_counts(&g, 2 * PAGE + 15);
    assert_eq!(result.windows.len(), g.tensors.len().div_ceil(2));
    assert_eq!(result.metal_view_bytes, (g.tensors.len() * PAGE) as u64);
    assert_eq!(result.private_shadow_bytes, 0);
    Arc::get_mut(&mut g).unwrap().tensors.reverse();
    assert_eq!(check_counts(&g, 2 * PAGE + 15), result);
}

#[test]
fn metadata_storage_plan_separates_aligned_engram_and_unused_segments() {
    for excluded in [
        "blk.0.engram_embd.weight",
        "per_layer_token_embd.weight",
        "mtp.unused",
    ] {
        let engram = excluded != "mtp.unused";
        let (kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, engram);
        let tensor = if engram {
            let mut table = tensors.pop().unwrap();
            table.0 = excluded.into();
            table
        } else {
            (excluded.into(), vec![32], GgmlType::I8)
        };
        tensors.insert(tensors.len() / 2, tensor);
        let g = write_fixture(kv, tensors);
        let result = check_counts(&g, M5_MAX_BUFFER_LENGTH);
        let t = g.find(excluded).unwrap();
        assert_eq!(result.windows.len(), 2);
        assert_eq!(result.private_window_count, 0);
        assert_eq!(result.private_shadow_bytes, 0);
        assert_eq!(result.shared_eof_window_count, 0);
        assert_eq!(result.bytes.disk_engram, if engram { t.n_bytes } else { 0 });
        assert!(result.windows.iter().all(|w| {
            w.window.mmap_offset >= t.data_offset + t.n_bytes
                || w.window.mmap_offset + w.window.length as u64 <= t.data_offset
        }));
    }
}

#[test]
fn metadata_storage_plan_charges_entire_private_engram_boundary_windows() {
    for table_name in ["blk.0.engram_embd.weight", "per_layer_token_embd.weight"] {
        let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, true);
        kv.iter_mut()
            .find(|(key, _)| *key == "general.alignment")
            .unwrap()
            .1 = Value::U32(32);
        let mut table = tensors.pop().unwrap();
        table.0 = table_name.into();
        table.2 = GgmlType::F32;
        tensors.insert(tensors.len() - 3, table);
        let g = write_fixture_layout(kv, tensors, false);
        let result = check_counts(&g, M5_MAX_BUFFER_LENGTH);
        let t = g.find(table_name).unwrap();
        assert_ne!(t.data_offset % PAGE as u64, 0);
        assert_ne!((t.data_offset + t.n_bytes) % PAGE as u64, 0);
        assert_eq!(result.bytes.disk_engram, t.n_bytes);
        assert_eq!(result.windows.len(), 2);
        assert_eq!(result.private_window_count, 2);
        assert_eq!(result.shared_eof_window_count, 0);
        assert_eq!(result.private_shadow_bytes, result.metal_view_bytes);
        assert!(result.private_shadow_bytes > 2 * PAGE as u64);
        for w in &result.windows {
            assert_eq!(
                w.mapping_kind,
                ResidentMappingKind::PrivateBoundary {
                    bytes: w.window.length as u64
                }
            );
            let overlap = (w.window.mmap_offset + w.window.length as u64)
                .min(t.data_offset + t.n_bytes)
                - w.window.mmap_offset.max(t.data_offset);
            assert!(overlap > 0 && overlap <= PAGE as u64);
            assert!(
                w.window.mmap_offset > t.data_offset
                    || w.window.mmap_offset + (w.window.length as u64) < t.data_offset + t.n_bytes
            );
        }
    }
}

#[test]
fn metadata_storage_plan_rejects_invalid_geometry_without_import() {
    let g = fixture(1, GgmlType::F32, 2, false);
    for (page, max) in [
        (0, M5_MAX_BUFFER_LENGTH),
        (3, M5_MAX_BUFFER_LENGTH),
        (8, PAGE),
        (PAGE, 0),
        (PAGE, PAGE - 1),
    ] {
        assert!(resident_storage_plan(&g, page, max).is_err());
    }
    let wide = fixture(1, GgmlType::F32, 128, false);
    assert!(resident_storage_plan(&wide, PAGE, PAGE).is_err());
    for case in 0..5 {
        let mut g = fixture(1, GgmlType::F32, 2, false);
        let raw_len = g.shards[0].mmap_len() as u64;
        let g = Arc::get_mut(&mut g).unwrap();
        let t = g
            .tensors
            .iter_mut()
            .find(|t| names::bank(&t.name).is_some())
            .unwrap();
        match case {
            0 => t.shard_idx = 1,
            1 => t.data_offset += 1,
            2 => t.data_offset = raw_len,
            3 => t.data_offset = u64::MAX,
            4 => t.shape[0] += 1,
            _ => unreachable!(),
        }
        assert!(resident_storage_plan(g, PAGE, M5_MAX_BUFFER_LENGTH).is_err());
    }
}
