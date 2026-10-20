use super::*;

const COMBINED: &str = "per_layer_token_embd.weight";
const ALIASES: [(&str, &str); 7] = [
    ("engram.layer_ids", "ple.layers"),
    ("engram.max_ngram_size", "ple.ngram_size"),
    ("engram.head_count", "ple.heads_per_ngram"),
    ("engram.key_length", "embedding_length_per_layer_input"),
    ("engram.primes", "ple.head_vocab_sizes"),
    ("engram.multipliers", "ple.layer_multipliers"),
    ("engram.offsets", "ple.head_offsets"),
];

pub(super) fn parts(
    ple: bool,
    combined: bool,
    dtype: GgmlType,
) -> (
    Vec<(&'static str, Value<'static>)>,
    Vec<(String, Vec<u64>, GgmlType)>,
) {
    let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, true);
    for (key, value) in &mut kv {
        *value = match *key {
            "general.alignment" => Value::U32(32),
            "deepseek41.block_count" => Value::U32(2),
            "deepseek41.compress_ratios" => Value::U64s(vec![0, 0]),
            "deepseek41.engram.layer_ids" => Value::U32s(vec![0, 1]),
            "deepseek41.engram.primes" => Value::U64s([vec![2; 24], vec![3; 24]].concat()),
            "deepseek41.engram.offsets" => Value::U64s(
                (0..24)
                    .map(|i| i * 2)
                    .chain((0..24).map(|i| i * 3))
                    .collect(),
            ),
            "deepseek41.engram.multipliers" => Value::U64s(vec![1, 3, 5, 7, 9, 11, 13, 15]),
            _ => value.clone(),
        };
    }
    tensors.extend(
        tensors
            .clone()
            .into_iter()
            .filter_map(|(name, mut shape, dtype)| {
                name.strip_prefix("blk.0.").map(|tail| {
                    if tail == "engram_embd.weight" {
                        shape[1] = 72;
                    }
                    (format!("blk.1.{tail}"), shape, dtype)
                })
            }),
    );
    if combined {
        tensors.retain(|(name, _, _)| !engram::is_table(name));
        tensors.push((COMBINED.into(), vec![256, 120], dtype));
    } else {
        for (name, _, kind) in &mut tensors {
            if engram::is_table(name) {
                *kind = dtype;
            }
        }
    }
    if ple {
        for (key, value) in &mut kv {
            for (legacy, native) in ALIASES {
                if key.strip_prefix("deepseek41.") == Some(legacy) {
                    *key = match native {
                        "ple.layers" => "deepseek41.ple.layers",
                        "ple.ngram_size" => "deepseek41.ple.ngram_size",
                        "ple.heads_per_ngram" => "deepseek41.ple.heads_per_ngram",
                        "embedding_length_per_layer_input" => {
                            "deepseek41.embedding_length_per_layer_input"
                        }
                        "ple.head_vocab_sizes" => "deepseek41.ple.head_vocab_sizes",
                        "ple.layer_multipliers" => "deepseek41.ple.layer_multipliers",
                        "ple.head_offsets" => "deepseek41.ple.head_offsets",
                        _ => unreachable!(),
                    };
                    if native == "ple.head_offsets" {
                        let Value::U64s(offsets) = value else {
                            unreachable!()
                        };
                        for offset in &mut offsets[24..] {
                            *offset += 48;
                        }
                    }
                    break;
                }
            }
        }
    }
    (kv, tensors)
}

pub(super) fn packed_rows(dtype: GgmlType, start: usize, rows: usize) -> Vec<u8> {
    let (_, bytes) = dtype.storage_layout().unwrap();
    let mut payload = vec![0; rows * bytes as usize];
    for (row, block) in payload.chunks_exact_mut(bytes as usize).enumerate() {
        let scale = half::f16::from_f32((start + row + 1) as f32 * 0.125).to_le_bytes();
        match dtype {
            GgmlType::Q5_K => {
                block[..2].copy_from_slice(&scale);
                block[4..16].fill(1);
                block[16..48].fill(0xff);
                block[48..].fill(0x11);
            }
            GgmlType::Q6_K => {
                block[..128].fill(0x11);
                block[128..192].fill(0xaa);
                block[192..208].fill(1);
                block[208..].copy_from_slice(&scale);
            }
            _ => unreachable!(),
        }
    }
    payload
}

fn populated(ple: bool, combined: bool, dtype: GgmlType) -> Arc<GgufFile> {
    let (kv, tensors) = parts(ple, combined, dtype);
    write_fixture_payload(kv, tensors, false, |name, shape, dtype| {
        engram::is_table(name).then(|| {
            packed_rows(
                dtype,
                if name.starts_with("blk.1.") { 48 } else { 0 },
                shape[1] as usize,
            )
        })
    })
}

#[test]
fn ple_and_legacy_normalize_to_the_same_layer_local_hashes() {
    use crate::deepseek_v41::engram::{AddressState, Token};
    let legacy = plan(&populated(false, false, GgmlType::Q6_K)).unwrap();
    for ple in [false, true] {
        for combined in [false, true] {
            let g = populated(ple, combined, GgmlType::Q6_K);
            let p = plan(&g).unwrap();
            assert_eq!(p.engram_hash, legacy.engram_hash);
            assert_eq!(p.engram_offsets[24], 0);
            assert_eq!(p.engram_hash.layers[1].rows, 72);
            assert_eq!(p.token_map, legacy.token_map);
            assert_eq!(p.raw_pad, legacy.raw_pad);
            assert_eq!(p.bytes.disk_engram, 120 * 210);
            assert_eq!(p.bytes.resident_bytes(), legacy.bytes.resident_bytes());
            let mut state = AddressState::new_with_config(1, &p.engram_hash).unwrap();
            let mut reference = AddressState::new_with_config(1, &legacy.engram_hash).unwrap();
            for token in [0, 1, 1, 0, 1] {
                let (rows, next) = state
                    .step_with_config(Token::Compressed(token), &p.engram_hash)
                    .unwrap();
                let (expected, next_reference) = reference
                    .step_with_config(Token::Compressed(token), &legacy.engram_hash)
                    .unwrap();
                assert_eq!(rows, expected);
                state = next;
                reference = next_reference;
            }
        }
    }
}

#[test]
fn dual_schema_offsets_compare_after_coordinate_normalization() {
    let (mut kv, tensors) = parts(true, true, GgmlType::Q6_K);
    let (legacy, _) = parts(false, false, GgmlType::Q6_K);
    kv.extend(legacy.into_iter().filter(|(key, _)| {
        ALIASES
            .iter()
            .any(|(old, _)| key.strip_prefix("deepseek41.") == Some(*old))
    }));
    assert!(plan(&write_fixture(kv.clone(), tensors.clone())).is_ok());
    for (legacy, _) in ALIASES {
        let mut conflicting = kv.clone();
        let (_, value) = conflicting
            .iter_mut()
            .find(|(key, _)| key.strip_prefix("deepseek41.") == Some(legacy))
            .unwrap();
        match value {
            Value::U32(n) => *n += 1,
            Value::U32s(v) => v[0] += 1,
            Value::U64s(v) => v[0] += 1,
            _ => unreachable!(),
        }
        assert!(
            plan(&write_fixture(conflicting, tensors.clone())).is_err(),
            "{legacy}"
        );
    }
}

#[test]
fn ple_missing_malformed_and_inconsistent_fields_are_rejected() {
    let (kv, tensors) = parts(true, true, GgmlType::Q6_K);
    for (_, native) in ALIASES {
        let mut missing = kv.clone();
        missing.retain(|(key, _)| key.strip_prefix("deepseek41.") != Some(native));
        assert!(
            plan(&write_fixture(missing, tensors.clone())).is_err(),
            "{native}"
        );
        for bad in [
            Value::Str("malformed"),
            Value::U64s(vec![]),
            Value::Ints(vec![-1]),
        ] {
            let mut malformed = kv.clone();
            malformed
                .iter_mut()
                .find(|(key, _)| key.strip_prefix("deepseek41.") == Some(native))
                .unwrap()
                .1 = bad;
            assert!(
                plan(&write_fixture(malformed, tensors.clone())).is_err(),
                "{native}"
            );
        }
    }
    for (index, value) in [
        (0, 1),
        (24, 0),
        (24, 47),
        (24, 49),
        (24, u64::MAX),
        (25, 50),
    ] {
        let mut bad = kv.clone();
        let Value::U64s(offsets) = &mut bad
            .iter_mut()
            .find(|(key, _)| *key == "deepseek41.ple.head_offsets")
            .unwrap()
            .1
        else {
            unreachable!()
        };
        offsets[index] = value;
        assert!(
            plan(&write_fixture(bad, tensors.clone())).is_err(),
            "offset[{index}]={value}"
        );
    }
    let mut bad = kv;
    let Value::U64s(primes) = &mut bad
        .iter_mut()
        .find(|(key, _)| *key == "deepseek41.ple.head_vocab_sizes")
        .unwrap()
        .1
    else {
        unreachable!()
    };
    primes[47] = u64::MAX;
    let error = plan(&write_fixture(bad, tensors)).unwrap_err().to_string();
    assert!(error.contains("overflow"), "{error}");
}

#[test]
fn disabled_engram_accepts_zero_geometry_without_binding_auxiliary_tables() {
    let (mut kv, mut tensors) = fixture_parts(1, GgmlType::F32, 2, false);
    kv.extend([
        ("deepseek41.engram.head_dim", Value::U32(0)),
        ("deepseek41.embedding_length_per_layer_input", Value::U32(0)),
        ("deepseek41.engram.head_count", Value::U32(0)),
        ("deepseek41.ple.heads_per_ngram", Value::U32(0)),
        ("deepseek41.engram.max_ngram_size", Value::U32(0)),
        ("deepseek41.ple.ngram_size", Value::U32(0)),
    ]);
    tensors.push(("blk.1.engram_embd.weight".into(), vec![16], GgmlType::I8));
    let g = write_fixture(kv, tensors);
    let p = plan(&g).unwrap();
    assert!(p.engram_hash.layers.is_empty());
    assert_eq!(p.bytes.disk_engram, 0);
    assert!(DiskEngram::new(g).unwrap().read_rows(1, &[0]).is_err());
}

#[test]
fn combined_main_table_does_not_bind_or_validate_auxiliary_table_payloads() {
    let (kv, mut tensors) = parts(true, true, GgmlType::Q6_K);
    tensors.push(("blk.2.engram_embd.weight".into(), vec![16], GgmlType::I8));
    let g = write_fixture(kv, tensors);
    let p = plan(&g).unwrap();
    assert_eq!(p.bytes.disk_engram, 120 * 210);
    assert_eq!(p.dtype_inventory["I8"], 16);
    let disk = DiskEngram::new(g).unwrap();
    assert!(disk.read_rows(1, &[0]).is_ok());
    assert!(disk.read_rows(2, &[0]).is_err());
}

#[test]
fn combined_table_geometry_and_ambiguous_bindings_are_rejected() {
    let (kv, tensors) = parts(true, true, GgmlType::Q6_K);
    for shape in [
        vec![256, 119],
        vec![256, 121],
        vec![512, 120],
        vec![256, 120, 2],
    ] {
        let mut bad = tensors.clone();
        bad.iter_mut()
            .find(|(name, _, _)| name == COMBINED)
            .unwrap()
            .1 = shape;
        assert!(plan(&write_fixture(kv.clone(), bad)).is_err());
    }
    let mut ambiguous = tensors.clone();
    ambiguous.push((
        "blk.1.engram_embd.weight".into(),
        vec![256, 72],
        GgmlType::Q6_K,
    ));
    assert!(plan(&write_fixture(kv.clone(), ambiguous)).is_err());
    let mut missing = tensors;
    missing.retain(|(name, _, _)| name != COMBINED);
    assert!(plan(&write_fixture(kv, missing)).is_err());
}

#[test]
fn q5_and_q6_combined_rows_translate_once_and_retain_handles() {
    for dtype in [GgmlType::Q5_K, GgmlType::Q6_K] {
        let g = populated(true, true, dtype);
        let p = plan(&g).unwrap();
        let desc = g.find(COMBINED).unwrap();
        let bytes = packed_rows(dtype, 0, 120);
        let expected = crate::codec::dequant_to_f32(desc, &bytes).unwrap();
        assert_eq!(
            expected[0],
            if dtype == GgmlType::Q5_K {
                17.0 / 8.0
            } else {
                1.0 / 8.0
            }
        );
        assert_eq!(p.bytes.disk_engram, bytes.len() as u64);
        let disk = DiskEngram::new(g.clone()).unwrap();
        let separate = DiskEngram::new(populated(false, false, dtype)).unwrap();
        drop(g);
        for (layer, base, rows) in [(0, 0, 48), (1, 48, 72)] {
            let ids = [rows - 1, 0, 13, 1, rows - 1];
            let actual = disk.read_rows(layer, &ids).unwrap();
            assert_eq!(actual, separate.read_rows(layer, &ids).unwrap());
            for (&id, row) in ids.iter().zip(actual.chunks_exact(256)) {
                let start = (base + id) as usize * 256;
                assert_eq!(row, &expected[start..start + 256]);
            }
            assert!(disk.read_rows(layer, &[]).unwrap().is_empty());
            assert!(disk.read_rows(layer, &[rows]).is_err());
            assert!(disk.read_rows(layer, &[u64::MAX]).is_err());
        }
        assert_ne!(
            disk.read_rows(0, &[0]).unwrap(),
            disk.read_rows(1, &[0]).unwrap()
        );
        assert!(disk.read_rows(2, &[0]).is_err());
        assert!(disk.read_rows(1, &vec![0; 32769]).is_err());
    }
}

#[test]
fn combined_table_encoding_is_inferred_but_explicit_conflicts_fail() {
    let (mut kv, tensors) = parts(true, true, GgmlType::Q6_K);
    kv.push(("deepseek41.engram.encoding", Value::Str("Q5_K")));
    let error = plan(&write_fixture(kv, tensors)).unwrap_err().to_string();
    assert!(error.contains("conflicts"), "{error}");
}

#[test]
fn combined_table_stays_out_of_metal_residency() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let g = populated(true, true, GgmlType::Q6_K);
    let p = plan(&g).unwrap();
    let weights = ResidentWeights::load(&ctx, g).unwrap();
    assert_eq!(weights.bytes, p.bytes);
    assert_eq!(weights.engram_hash, p.engram_hash);
    assert_eq!(weights.banks.len(), 6);
    assert!(
        !weights
            .dense
            .keys()
            .any(|name| name.contains("token_embd") || name.contains("engram_embd"))
    );
    assert_eq!(
        weights.engram.read_rows(1, &[0]).unwrap(),
        vec![49.0 / 8.0; 256]
    );
}
