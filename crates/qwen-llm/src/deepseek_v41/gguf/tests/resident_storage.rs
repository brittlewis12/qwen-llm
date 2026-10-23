use super::*;

const PAGE: usize = 16_384;
const M5_MAX_BUFFER_LENGTH: usize = 179_357_827_072;

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
            assert!(w.window.mmap_offset > t.data_offset
                || w.window.mmap_offset + w.window.length as u64 < t.data_offset + t.n_bytes);
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
