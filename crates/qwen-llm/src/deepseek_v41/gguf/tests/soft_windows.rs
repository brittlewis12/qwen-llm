use super::*;

const PAGE: usize = 16_384;
const M5_MAX_BUFFER_LENGTH: usize = 179_357_827_072;

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
    write_fixture_payload(kv, tensors, false, owner_bank_payload)
}

const OWNER_VIEW_OFFSET: u64 = 4;
const OWNER_VIEW_ELEMENTS: u64 = 3;
const OWNER_VIEW_BYTES: u64 = 12;
const OWNER_VIEW_END: u64 = 16;
const OWNER_VIEW_EXPECTED: [f32; 3] = [2.5, 3.75, 5.0];

fn owner_bank_payload(name: &str, shape: &[u64], dtype: GgmlType) -> Option<Vec<u8>> {
    names::bank(name)?;
    assert_eq!(dtype, GgmlType::F32);
    Some(
        (0..shape.iter().product::<u64>())
            .flat_map(|index| ((index % 16 + 1) as f32 * 1.25).to_le_bytes())
            .collect(),
    )
}

fn owner_fixtures() -> [Arc<GgufFile>; 2] {
    let (kv, tensors) = fixture_parts(1, GgmlType::F32, 128, false);
    [
        overlapping_segments(),
        write_fixture_payload(kv, tensors, true, owner_bank_payload),
    ]
}

fn assert_owner_view_fits(t: &TensorDesc, offset: u64, elements: u64) -> u64 {
    assert_eq!(t.dtype, GgmlType::F32);
    let bytes = elements.checked_mul(4).unwrap();
    assert!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= t.n_bytes),
        "owner subview exceeds fixture bank"
    );
    bytes
}

fn owner_view_expected(g: &GgufFile, t: &TensorDesc) -> [f32; 3] {
    assert_eq!(
        assert_owner_view_fits(t, OWNER_VIEW_OFFSET, OWNER_VIEW_ELEMENTS),
        OWNER_VIEW_BYTES
    );
    assert_eq!(OWNER_VIEW_OFFSET + OWNER_VIEW_BYTES, OWNER_VIEW_END);
    let mut bytes = [0u8; OWNER_VIEW_BYTES as usize];
    g.read_shard_exact_at(t.shard_idx, t.data_offset + OWNER_VIEW_OFFSET, &mut bytes)
        .unwrap();
    let expected =
        std::array::from_fn(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    assert_eq!(expected, OWNER_VIEW_EXPECTED);
    expected
}

#[test]
fn soft_window_owner_subview_fixture_contract_is_nonzero_and_bounded() {
    for (g, width) in owner_fixtures().into_iter().zip([2, 128]) {
        let stamps = g.revalidate_retained_shard_stamps().unwrap();
        let bank = g
            .tensors
            .iter()
            .find(|t| names::bank(&t.name).is_some())
            .unwrap();
        assert_eq!(bank.shape, [width, width]);
        assert_eq!(bank.n_bytes, width * width * 4);
        assert_eq!(OWNER_VIEW_OFFSET, 4);
        assert_eq!(OWNER_VIEW_ELEMENTS, 3);
        assert_eq!(owner_view_expected(&g, bank), [2.5, 3.75, 5.0]);
        assert!(OWNER_VIEW_EXPECTED.iter().all(|&value| value != 0.0));
        assert!(g.shards[0].mmap_len() < 16 << 20);
        let storage = targeted(&g, Some(PAGE));
        assert!(storage.metal_view_bytes < 16 << 20);
        let bound: std::collections::BTreeSet<_> = g
            .tensors
            .iter()
            .filter(|t| resident_tensor(t, 1))
            .map(|t| first_containing(&storage, t))
            .collect();
        let lengths: Vec<_> = bound
            .iter()
            .map(|&i| storage.windows[i].window.length as u64)
            .collect();
        eprintln!(
            "soft_window_fixture_prices width={width} raw_file_bytes={} planned_windows={} planned_metal_view_bytes={} private_shadow_bytes={} first_containing_resource_lengths={lengths:?} first_containing_unique_bytes={} guarded_output_bytes=28",
            g.shards[0].mmap_len(),
            storage.windows.len(),
            storage.metal_view_bytes,
            storage.private_shadow_bytes,
            lengths.iter().sum::<u64>(),
        );
        assert_eq!(g.revalidate_retained_shard_stamps().unwrap(), stamps);
    }
}

#[test]
#[should_panic(expected = "owner subview exceeds fixture bank")]
fn soft_window_owner_subview_fixture_contract_rejects_old_eight_float_request() {
    let g = overlapping_segments();
    let bank = g
        .tensors
        .iter()
        .find(|t| names::bank(&t.name).is_some())
        .unwrap();
    assert_eq!(bank.n_bytes, 16);
    assert_owner_view_fits(bank, 0, 8);
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
fn soft_window_prepared_loader_plan_validates_chosen_geometry_and_preserves_controls() {
    for g in owner_fixtures() {
        let reference = plan(&g).unwrap();
        for target in [None, Some(4_294_967_296), Some(PAGE)] {
            let (p, storage) = prepared_load_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH, target).unwrap();
            assert_eq!(storage, targeted(&g, target));
            assert_eq!(p.bytes, reference.bytes);
            assert_eq!(p.controls, reference.controls);
            assert_eq!(p.source_groups(), reference.source_groups());
            assert_eq!(p.token_map, reference.token_map);
            assert_eq!(p.engram_hash, reference.engram_hash);
            assert_eq!(p.config.hidden, reference.config.hidden);
            assert_eq!(p.config.intermediate, reference.config.intermediate);
            assert_eq!(p.config.experts, reference.config.experts);
            assert_eq!(p.config.vocab, reference.config.vocab);
            assert_eq!(p.schedule, reference.schedule);
        }
        assert!(prepared_load_plan(&g, 0, M5_MAX_BUFFER_LENGTH, Some(PAGE)).is_err());
        assert!(prepared_load_plan(&g, PAGE, PAGE - 1, Some(PAGE)).is_err());
    }
}

#[test]
fn soft_window_tensor_controls_keep_source_groups_and_cpu_accounting() {
    let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, true);
    let mut payloads = BTreeMap::new();
    for (key, name, dtype) in [
        (
            "deepseek41.engram.token_map",
            "engram_token_map.weight",
            GgmlType::I32,
        ),
        (
            "deepseek41.engram.primes",
            "engram_primes.weight",
            GgmlType::I64,
        ),
        (
            "deepseek41.engram.offsets",
            "engram_offsets.weight",
            GgmlType::I64,
        ),
        (
            "deepseek41.engram.multipliers",
            "engram_multipliers.weight",
            GgmlType::I64,
        ),
    ] {
        let index = kv.iter().position(|(k, _)| *k == key).unwrap();
        let (_, value) = kv.remove(index);
        let (count, bytes): (usize, Vec<u8>) = match value {
            Value::Ints(values) => (
                values.len(),
                values.into_iter().flat_map(i32::to_le_bytes).collect(),
            ),
            Value::U64s(values) => (
                values.len(),
                values.into_iter().flat_map(u64::to_le_bytes).collect(),
            ),
            _ => unreachable!(),
        };
        tensors.push((name.into(), vec![count as u64], dtype));
        payloads.insert(name.to_string(), bytes);
    }
    let g = write_fixture_payload(kv, tensors, false, |name, _, _| payloads.get(name).cloned());
    let reference = plan(&g).unwrap();
    let stamps = reference.source_groups().clone();
    assert_eq!(reference.controls.payload_bytes_read, 428);
    assert_eq!(reference.controls.cold_cpu_bytes, 1528);
    assert_eq!(controls::cold_bound(&g).unwrap(), 2568);
    assert_eq!(reference.source_groups().backbone.len(), 1);
    assert!(reference.source_groups().auxiliary.is_empty());
    let audit = resident_storage_audit(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
    assert_eq!(audit.control, targeted(&g, None));
    assert_eq!(
        resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap(),
        targeted(&g, audit.selected_soft_coalescing_target)
    );
    for target in [None, Some(4_294_967_296), Some(PAGE)] {
        let (p, storage) = prepared_load_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH, target).unwrap();
        assert_eq!(p.controls, reference.controls);
        assert_eq!(p.source_groups(), &stamps);
        assert_eq!(p.token_map, reference.token_map);
        assert_eq!(p.engram_hash, reference.engram_hash);
        assert_eq!(p.bytes, reference.bytes);
        assert_eq!(storage.bytes, reference.bytes);
        assert_eq!(storage, targeted(&g, target));
        assert_eq!(audit.tensors.payload_bytes, p.bytes.resident_bytes());
    }
    eprintln!(
        "soft_window_tensor_controls controls={:?} source_groups=1+0",
        reference.controls
    );
}

#[test]
fn soft_window_prepared_loader_plan_rejects_changed_retained_source() {
    let g = overlapping_segments();
    prepared_load_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH, Some(PAGE)).unwrap();
    let shard = &g.shards[0];
    shard.file.set_len(shard.mmap_len() as u64 + 1).unwrap();
    assert!(g.revalidate_retained_shard_stamps().is_err());
    for target in [None, Some(PAGE)] {
        assert!(prepared_load_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH, target).is_err());
    }
    assert!(resident_storage_audit(&g, PAGE, M5_MAX_BUFFER_LENGTH).is_err());
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
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let page = metal::host_page_size_bytes().unwrap();
    assert_eq!(page, PAGE);
    for g in owner_fixtures() {
        let soft = targeted(&g, Some(PAGE));
        assert!(soft.metal_view_bytes < 16 << 20);
        let weights = ResidentWeights::load_planned(&ctx, Arc::clone(&g), Some(PAGE)).unwrap();
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
        let bank = &weights.banks[0];
        let descriptor = g
            .tensors
            .iter()
            .find(|t| names::bank(&t.name) == Some((bank.layer, bank.projection)))
            .unwrap();
        let expected = owner_view_expected(&g, descriptor);
        assert_eq!(bank.tensor.n_bytes(), descriptor.n_bytes);
        let retained = bank
            .tensor
            .view_bytes(OWNER_VIEW_OFFSET, vec![OWNER_VIEW_ELEMENTS]);
        assert_eq!(retained.offset, bank.tensor.offset + OWNER_VIEW_OFFSET);
        assert_eq!(retained.n_bytes(), OWNER_VIEW_BYTES);
        drop(weights);
        drop(g);
        assert_eq!(
            retained.provenance(),
            metal::MetalTensorProvenance::RetainedGgufReadOnly
        );
        assert!(!retained.is_writable());
        assert!(retained.buffer.length() as u64 >= retained.offset + OWNER_VIEW_BYTES);

        let guards = [-101.0f32, -102.0, -103.0, -104.0, -105.0, -106.0, -107.0];
        let output =
            MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&guards), vec![7], GgmlType::F32)
                .unwrap();
        let destination = output.view_bytes(8, vec![OWNER_VIEW_ELEMENTS]);
        assert_eq!(destination.n_bytes(), OWNER_VIEW_BYTES);
        let command = ctx.queue.commandBuffer().unwrap();
        let blit = metal::BlitEncoder::begin(&command);
        blit.copy_tensor(&retained, &destination);
        blit.end();
        command.commit();
        metal::wait_completed(&command).unwrap();
        assert!(command.error().is_none());
        let expected_output = [
            guards[0],
            guards[1],
            expected[0],
            expected[1],
            expected[2],
            guards[5],
            guards[6],
        ];
        let got = unsafe {
            std::slice::from_raw_parts(
                output.buffer.contents().as_ptr().cast::<u8>(),
                guards.len() * 4,
            )
        };
        assert_eq!(got, bytemuck::cast_slice::<f32, u8>(&expected_output));
    }
}
