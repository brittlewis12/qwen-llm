use super::*;
use resident_storage::{M5_MAX_BUFFER_LENGTH, PAGE};

fn targeted(g: &GgufFile, target: Option<usize>) -> ResidentStoragePlan {
    super::super::resident_storage::for_plan_with_target(
        g,
        &plan(g).unwrap(),
        PAGE,
        M5_MAX_BUFFER_LENGTH,
        target,
    )
    .unwrap()
}

fn first_containing(storage: &ResidentStoragePlan, t: &TensorDesc) -> usize {
    storage
        .windows
        .iter()
        .position(|w| {
            w.window.shard_idx == t.shard_idx
                && w.window.mmap_offset <= t.data_offset
                && t.data_offset + t.n_bytes <= w.window.mmap_offset + w.window.length as u64
        })
        .unwrap()
}

fn overlapping_segments() -> Arc<GgufFile> {
    let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, false);
    kv.iter_mut()
        .find(|(key, _)| *key == "general.alignment")
        .unwrap()
        .1 = Value::U32(32);
    tensors.insert(1, ("mtp.unused".into(), vec![4], GgmlType::F32));
    write_fixture_layout(kv, tensors, false)
}

#[test]
fn soft_window_none_preserves_incumbent_geometry_and_selected_metadata_loader_parity() {
    for pad_end in [false, true] {
        let (kv, tensors) = fixture_parts(1, GgmlType::F32, 2, false);
        let g = write_fixture_layout(kv, tensors, pad_end);
        let p = plan(&g).unwrap();
        let none = targeted(&g, None);
        let requests: Vec<_> = g
            .tensors
            .iter()
            .filter(|t| resident_tensor(t, p.layers))
            .collect();
        let incumbent = metal::plan_retained_storage(
            &storage::padded_lengths(&g, PAGE).unwrap(),
            &requests,
            PAGE,
            M5_MAX_BUFFER_LENGTH,
            16,
        )
        .unwrap();
        assert_eq!(
            none.windows.iter().map(|w| &w.window).collect::<Vec<_>>(),
            incumbent.windows.iter().collect::<Vec<_>>()
        );
        assert_eq!(none.windows.len(), 1);
        assert_eq!(none.windows[0].window.mmap_offset, requests[0].data_offset);
        assert_eq!(
            none.metal_view_bytes,
            (g.shards[0].mmap_len().next_multiple_of(PAGE) as u64) - requests[0].data_offset
        );
        let selected = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
        assert_eq!(
            selected,
            super::super::resident_storage::for_plan(&g, &p, PAGE, M5_MAX_BUFFER_LENGTH).unwrap()
        );
        let audit = resident_storage_audit(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
        assert_eq!(audit.control, none);
        assert_eq!(
            selected,
            targeted(&g, audit.selected_soft_coalescing_target)
        );
        let soft = targeted(&g, Some(PAGE));
        assert_eq!(soft.windows.len(), requests.len());
        assert_eq!(soft.bytes, none.bytes);
        assert_eq!(soft.private_shadow_bytes, 0);
        assert_eq!(soft.shared_eof_window_count, usize::from(!pad_end));
        assert_eq!(soft.metal_view_bytes, none.metal_view_bytes);
    }
}

#[test]
fn soft_window_audit_is_bounded_excludes_disk_and_unused_and_does_not_map() {
    let (kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, true);
    tensors.insert(2, ("mtp.unused".into(), vec![32], GgmlType::F32));
    let mut g = write_fixture_layout(kv, tensors, false);
    let refs = Arc::strong_count(&g.shards[0].mmap);
    let stamps = g.revalidate_retained_shard_stamps().unwrap();
    let audit = resident_storage_audit(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
    let p = plan(&g).unwrap();
    let resident: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| resident_tensor(t, p.layers))
        .collect();
    assert_eq!(audit.tensors.count, resident.len());
    assert_eq!(audit.tensors.payload_bytes, p.bytes.resident_bytes());
    assert_eq!(
        audit.tensors.largest_payload_bytes,
        resident.iter().map(|t| t.n_bytes).max().unwrap()
    );
    assert_eq!(audit.tensors.top.len(), 10);
    assert!(
        audit
            .tensors
            .top
            .windows(2)
            .all(|w| w[0].tensor.n_bytes >= w[1].tensor.n_bytes)
    );
    assert!(
        audit
            .tensors
            .top
            .iter()
            .all(|e| resident_tensor(&e.tensor, p.layers))
    );
    assert_eq!(audit.tensors.above_soft_target_count, 0);
    assert_eq!(audit.tensors.above_soft_target_payload_bytes, 0);
    assert_eq!(Arc::strong_count(&g.shards[0].mmap), refs);
    assert_eq!(g.revalidate_retained_shard_stamps().unwrap(), stamps);
    Arc::get_mut(&mut g).unwrap().tensors.reverse();
    let permuted = resident_storage_audit(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
    assert_eq!(permuted.control, audit.control);
    assert_eq!(permuted.candidate, audit.candidate);
    assert_eq!(
        permuted
            .tensors
            .top
            .iter()
            .map(|e| &e.tensor.name)
            .collect::<Vec<_>>(),
        audit
            .tensors
            .top
            .iter()
            .map(|e| &e.tensor.name)
            .collect::<Vec<_>>()
    );
}

#[test]
fn soft_window_private_boundaries_and_oversized_singletons_keep_hardware_eligibility() {
    let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 128, true);
    kv.iter_mut()
        .find(|(key, _)| *key == "general.alignment")
        .unwrap()
        .1 = Value::U32(32);
    let table = tensors.pop().unwrap();
    tensors.insert(tensors.len() / 2, table);
    let g = write_fixture_layout(kv, tensors, false);
    let soft = targeted(&g, Some(PAGE));
    assert!(soft.windows.iter().any(|w| w.window.length > PAGE));
    assert!(soft.private_window_count > 0);
    assert_eq!(
        soft.private_shadow_bytes,
        soft.windows
            .iter()
            .map(|w| w.mapping_kind.private_bytes())
            .sum::<u64>()
    );
    assert_eq!(soft.bytes, plan(&g).unwrap().bytes);
    let audit =
        super::super::resident_storage::audit_tensors(&g, &plan(&g).unwrap(), PAGE, PAGE as u64)
            .unwrap();
    let oversized: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| {
            resident_tensor(t, 1)
                && (t.data_offset + t.n_bytes).next_multiple_of(PAGE as u64)
                    - t.data_offset / PAGE as u64 * PAGE as u64
                    > PAGE as u64
        })
        .collect();
    assert!(!oversized.is_empty());
    assert_eq!(audit.above_soft_target_count, oversized.len());
    assert_eq!(
        audit.above_soft_target_payload_bytes,
        oversized.iter().map(|t| t.n_bytes).sum::<u64>()
    );
    for w in &soft.windows {
        assert_eq!(
            w.mapping_kind,
            ResidentMappingKind::from(storage::mapping_kind(&g, &w.window, PAGE).unwrap())
        );
        assert!(w.window.length <= M5_MAX_BUFFER_LENGTH);
    }
    assert!(
        super::super::resident_storage::for_plan_with_target(
            &g,
            &plan(&g).unwrap(),
            PAGE,
            PAGE,
            Some(PAGE)
        )
        .is_err()
    );
}

#[test]
fn soft_window_overlap_padding_requires_first_containing_not_segment_index() {
    let g = overlapping_segments();
    let soft = targeted(&g, Some(PAGE));
    assert_eq!(soft.windows.len(), 2);
    let excluded = g.find("mtp.unused").unwrap();
    let later = g
        .tensors
        .iter()
        .find(|t| {
            resident_tensor(t, 1)
                && t.data_offset > excluded.data_offset
                && first_containing(&soft, t) == 0
        })
        .unwrap();
    assert!(later.data_offset >= soft.windows[1].window.mmap_offset);
    assert!(
        later.data_offset + later.n_bytes
            <= soft.windows[1].window.mmap_offset + soft.windows[1].window.length as u64
    );
    assert_eq!(first_containing(&soft, later), 0);
    assert!(
        soft.windows
            .iter()
            .all(|w| w.mapping_kind.private_bytes() == 0)
    );
}

#[test]
#[ignore = "self-leased tiny realized soft-window association proof; debug validation required"]
fn soft_window_live_resources_follow_first_containing_and_survive_owner_drop() {
    use objc2::rc::Retained;
    use objc2_metal::MTLBuffer;

    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let page = metal::host_page_size_bytes().unwrap();
    assert_eq!(page, PAGE);
    for g in [
        overlapping_segments(),
        fixture(1, GgmlType::F32, 128, false),
    ] {
        let p = plan(&g).unwrap();
        let soft = targeted(&g, Some(PAGE));
        assert!(soft.metal_view_bytes < 16 << 20);
        let weights = ResidentWeights::load_planned(&ctx, Arc::clone(&g), p, soft.clone()).unwrap();
        let mut identities = BTreeMap::new();
        let mut actual = Vec::new();
        for t in g
            .tensors
            .iter()
            .filter(|t| resident_tensor(t, weights.schedule.layers))
        {
            let view = if let Some((layer, part)) = names::bank(&t.name) {
                weights.bank(layer, part).unwrap()
            } else {
                &weights.dense[&names::dense(&t.name, weights.schedule.layers).unwrap()]
            };
            let index = first_containing(&soft, t);
            let identity = Retained::as_ptr(&view.buffer) as *const () as usize;
            if let Some(previous) = identities.insert(index, identity) {
                assert_eq!(identity, previous);
            }
            let w = &soft.windows[index].window;
            assert_eq!(view.buffer.length(), w.length);
            assert_eq!(view.offset as u64, t.data_offset - w.mmap_offset);
            actual.push((index, identity));
        }
        for &(a, id_a) in &actual {
            for &(b, id_b) in &actual {
                assert_eq!(a == b, id_a == id_b);
            }
        }
        let summary = &weights.retained_resources;
        assert_eq!(summary.resources.len(), identities.len());
        assert_eq!(summary.tensor_view_count, actual.len());
        assert_eq!(
            summary.total_unique_length,
            identities
                .keys()
                .map(|&i| soft.windows[i].window.length as u64)
                .sum::<u64>()
        );
        assert_eq!(
            summary.max_unique_length,
            identities
                .keys()
                .map(|&i| soft.windows[i].window.length as u64)
                .max()
                .unwrap()
        );
        assert_eq!(
            *summary,
            super::super::retained_resources::summarize(&weights.banks, &weights.dense).unwrap()
        );
        assert!(serde_json::to_string(summary).unwrap().find("0x").is_none());
        let retained = weights.banks[0].tensor.view_bytes(0, vec![8]);
        drop(weights);
        drop(g);
        assert_eq!(
            retained.provenance(),
            metal::MetalTensorProvenance::RetainedGgufReadOnly
        );
        assert!(!retained.is_writable());
        assert!(retained.buffer.length() as u64 >= retained.offset + 32);
    }
}
