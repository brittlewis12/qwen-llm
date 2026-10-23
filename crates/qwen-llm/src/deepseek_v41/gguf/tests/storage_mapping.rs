use super::*;
use storage::MappingKind;

fn payload_fixture() -> Arc<GgufFile> {
    let page = metal::host_page_size_bytes().unwrap();
    write_fixture_payload(
        vec![("general.alignment", Value::U32(page as u32))],
        vec![("payload".into(), vec![(page / 4 + 3) as u64], GgmlType::F32)],
        false,
        |_, shape, _| Some(vec![0xa5; shape[0] as usize * 4]),
    )
}

fn tensor_window(g: &GgufFile, name: &str, page: usize) -> metal::RetainedStorageWindow {
    let t = g.find(name).unwrap();
    let start = t.data_offset / page as u64 * page as u64;
    let end = (t.data_offset + t.n_bytes).next_multiple_of(page as u64);
    metal::RetainedStorageWindow {
        shard_idx: t.shard_idx,
        mmap_offset: start,
        length: (end - start) as usize,
    }
}

fn snapshot(g: &GgufFile) -> (Vec<crate::gguf::GgufShardStamp>, Vec<u8>) {
    let stamps = g.revalidate_retained_shard_stamps().unwrap();
    let mut bytes = vec![0; g.shards[0].mmap_len()];
    g.read_shard_exact_at(0, 0, &mut bytes).unwrap();
    (stamps, bytes)
}

fn assert_read_only(_: &memmap2::Mmap) {}

#[test]
fn original_shared_reuses_arc_and_absolute_offsets() {
    let g = payload_fixture();
    let before = snapshot(&g);
    let page = metal::host_page_size_bytes().unwrap();
    let mut w = tensor_window(&g, "payload", page);
    w.length = page;
    let kind = storage::mapping_kind(&g, &w, page).unwrap();
    assert_eq!(kind, MappingKind::OriginalShared);
    assert_eq!(kind.private_bytes(), 0);
    let (mapped, local) = storage::mapping(&g, &w, page).unwrap();
    assert_read_only(&mapped);
    assert!(!local);
    assert!(Arc::ptr_eq(&mapped, &g.shards[0].mmap));
    assert_eq!(mapped.len(), before.1.len());
    let offset = g.find("payload").unwrap().data_offset as usize;
    assert!(mapped[offset..offset + page].iter().all(|&b| b == 0xa5));
    assert_eq!(snapshot(&g), before);
}

#[test]
fn partial_eof_is_shared_read_only_with_nonzero_payload_and_zero_tail() {
    let g = payload_fixture();
    let before = snapshot(&g);
    let page = metal::host_page_size_bytes().unwrap();
    let w = tensor_window(&g, "payload", page);
    let t = g.find("payload").unwrap();
    let raw_len = g.shards[0].mmap_len() as u64;
    assert_eq!(t.data_offset + t.n_bytes, raw_len);
    assert_eq!(raw_len % page as u64, 12);
    assert_eq!(w.length, 2 * page);
    assert_eq!(
        storage::padded_lengths(&g, page).unwrap()[0] as u64,
        w.mmap_offset + w.length as u64
    );
    let kind = storage::mapping_kind(&g, &w, page).unwrap();
    assert_eq!(kind, MappingKind::SharedEof);
    assert_eq!(kind.private_bytes(), 0);
    let (mapped, local) = storage::mapping(&g, &w, page).unwrap();
    assert_read_only(&mapped);
    assert!(local);
    assert!(!Arc::ptr_eq(&mapped, &g.shards[0].mmap));
    assert_eq!(mapped.len(), w.length);
    assert!(mapped[..t.n_bytes as usize].iter().all(|&b| b == 0xa5));
    assert!(mapped[t.n_bytes as usize..].iter().all(|&b| b == 0));
    assert_eq!(snapshot(&g), before);
    drop(g);
    assert!(mapped[page..page + 12].iter().all(|&b| b == 0xa5));
    assert!(mapped[page + 12..].iter().all(|&b| b == 0));
}

#[test]
fn invalid_geometry_is_rejected_before_any_mapping() {
    let g = payload_fixture();
    let before = snapshot(&g);
    let page = metal::host_page_size_bytes().unwrap();
    let valid = tensor_window(&g, "payload", page);
    let mut cases = vec![(valid.clone(), 0), (valid.clone(), 3)];
    for (offset, length) in [
        (valid.mmap_offset, 0),
        (valid.mmap_offset + 1, valid.length),
        (valid.mmap_offset, valid.length - 1),
        (valid.mmap_offset, valid.length + page),
        (valid.mmap_offset + valid.length as u64, page),
        (u64::MAX - (page as u64 - 1), page),
        (0, (isize::MAX as usize) + 1),
    ] {
        cases.push((
            metal::RetainedStorageWindow {
                shard_idx: 0,
                mmap_offset: offset,
                length,
            },
            page,
        ));
    }
    let mut missing_shard = valid.clone();
    missing_shard.shard_idx = g.shards.len();
    cases.push((missing_shard, page));
    for (w, page) in cases {
        assert!(storage::mapping_kind(&g, &w, page).is_err(), "{w:?}");
        assert!(storage::mapping(&g, &w, page).is_err(), "{w:?}");
    }
    assert!(storage::padded_lengths(&g, 0).is_err());
    assert!(storage::padded_lengths(&g, 3).is_err());
    assert!(storage::plan_mapping(u64::MAX, &[], &valid, page).is_err());
    assert!(storage::plan_mapping(0, &[], &valid, page).is_err());
    assert_eq!(snapshot(&g), before);
}

#[test]
fn page_aligned_eof_never_allows_an_extra_page() {
    let page = metal::host_page_size_bytes().unwrap();
    let g = write_fixture_layout(
        vec![("general.alignment", Value::U32(page as u32))],
        vec![("payload".into(), vec![(page / 4) as u64], GgmlType::F32)],
        false,
    );
    let before = snapshot(&g);
    let mut w = tensor_window(&g, "payload", page);
    assert_eq!(g.shards[0].mmap_len() % page, 0);
    assert_eq!(
        storage::mapping_kind(&g, &w, page).unwrap(),
        MappingKind::OriginalShared
    );
    w.length += page;
    assert!(storage::mapping_kind(&g, &w, page).is_err());
    assert!(storage::mapping(&g, &w, page).is_err());
    assert_eq!(snapshot(&g), before);
}

#[test]
fn large_eof_plan_is_metadata_only_and_fake_extent_is_rejected() {
    let page = 16384;
    let w = metal::RetainedStorageWindow {
        shard_idx: 0,
        mmap_offset: 16 << 30,
        length: 48usize << 30,
    };
    let p = storage::plan_mapping((64u64 << 30) - 37, &[], &w, page).unwrap();
    assert_eq!(p.kind, MappingKind::SharedEof);
    assert_eq!(p.kind.private_bytes(), 0);
    let g = payload_fixture();
    let before = snapshot(&g);
    assert!(storage::mapping_kind(&g, &w, page).is_err());
    assert!(storage::mapping(&g, &w, page).is_err());
    assert_eq!(snapshot(&g), before);
}

#[test]
fn missing_tensor_payload_still_fails_raw_eof_validation() {
    use std::io::{Seek, SeekFrom};

    let g = payload_fixture();
    let before = snapshot(&g);
    let mut file = fixture_file(vec![], vec![], false, |_, _, _| None);
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&before.1[..before.1.len() - 1]).unwrap();
    assert!(
        GgufFile::from_opened_file(file, "missing-payload.gguf").is_err(),
        "missing real payload must not be supplied by EOF padding"
    );
    assert_eq!(snapshot(&g), before);
}

#[test]
fn engram_boundary_clear_is_private_bounded_and_preserves_source() {
    let page = metal::host_page_size_bytes().unwrap();
    for table_name in ["blk.0.engram_embd.weight", "per_layer_token_embd.weight"] {
        let g = write_fixture_payload(
            vec![("general.alignment", Value::U32(32))],
            vec![
                ("prefix".into(), vec![(page / 8) as u64], GgmlType::F32),
                (table_name.into(), vec![(page / 4) as u64], GgmlType::F32),
                ("suffix".into(), vec![(page / 4 + 3) as u64], GgmlType::F32),
            ],
            false,
            |name, shape, _| {
                Some(vec![
                    if name == table_name { 0x55 } else { 0x12 };
                    shape[0] as usize * 4
                ])
            },
        );
        let before = snapshot(&g);
        let table = g.find(table_name).unwrap();
        for name in ["prefix", "suffix"] {
            let w = tensor_window(&g, name, page);
            let kind = storage::mapping_kind(&g, &w, page).unwrap();
            assert_eq!(
                kind,
                MappingKind::PrivateBoundary {
                    bytes: w.length as u64
                }
            );
            assert_eq!(kind.private_bytes(), w.length as u64);
            let (mapped, local) = storage::mapping(&g, &w, page).unwrap();
            assert_read_only(&mapped);
            assert!(local);
            assert_eq!(mapped.len(), w.length);
            let mut cleared = 0;
            for (i, &byte) in mapped.iter().enumerate() {
                let offset = w.mmap_offset + i as u64;
                if (table.data_offset..table.data_offset + table.n_bytes).contains(&offset) {
                    assert_eq!(byte, 0);
                    assert_eq!(before.1[offset as usize], 0x55);
                    cleared += 1;
                } else {
                    assert_eq!(byte, before.1.get(offset as usize).copied().unwrap_or(0));
                }
            }
            assert!(cleared > 0 && cleared <= page);
        }
        let interior = metal::RetainedStorageWindow {
            shard_idx: 0,
            mmap_offset: 0,
            length: g.shards[0].mmap_len().next_multiple_of(page),
        };
        let error = storage::mapping_kind(&g, &interior, page)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Engram occurs inside"), "{error}");
        assert!(storage::mapping(&g, &interior, page).is_err());
        assert_eq!(snapshot(&g), before);
    }
}
