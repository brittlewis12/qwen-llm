use super::*;
use std::io::Read;
use std::io::{Seek, SeekFrom};

type Metadata = Vec<(&'static str, Value<'static>)>;
type Tensors = Vec<(String, Vec<u64>, GgmlType)>;
type Payloads = BTreeMap<String, Vec<u8>>;

const CONTROLS: [(&str, &str, GgmlType, usize); 4] = [
    (
        "deepseek41.engram.token_map",
        "engram_token_map.weight",
        GgmlType::I32,
        3,
    ),
    (
        "deepseek41.engram.primes",
        "engram_primes.weight",
        GgmlType::I64,
        24,
    ),
    (
        "deepseek41.engram.offsets",
        "engram_offsets.weight",
        GgmlType::I64,
        24,
    ),
    (
        "deepseek41.engram.multipliers",
        "engram_multipliers.weight",
        GgmlType::I64,
        4,
    ),
];

fn parts(keep_inline: bool) -> (Metadata, Tensors, Payloads) {
    let (mut kv, mut tensors) = ple_engram::parts(false, false, GgmlType::IQ4_NL);
    kv.iter_mut()
        .find(|(k, _)| *k == "deepseek41.engram.multipliers")
        .unwrap()
        .1 = Value::U64s(vec![
        16_777_217,
        4_839_876_093_313,
        35_959_672_319_349,
        73_987_337_458_391,
        67_716_810_739_261,
        51_510_806_800_915,
        30_921_347_202_721,
        82_619_226_485_591,
    ]);
    kv.extend([
        ("deepseek41.engram.compressed_vocab_size", Value::U32(2)),
        ("deepseek41.engram.pad_token_id", Value::U32(2)),
        ("deepseek41.hash_layer_count", Value::U32(0)),
    ]);
    let mut payloads = Payloads::new();
    for (key, name, dtype, width) in CONTROLS {
        let value = &kv.iter().find(|(k, _)| *k == key).unwrap().1;
        let bytes = match value {
            Value::Ints(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
            Value::U64s(v) => v.iter().flat_map(|n| (*n as i64).to_le_bytes()).collect(),
            _ => unreachable!(),
        };
        payloads.insert(name.into(), bytes);
        tensors.push((
            name.into(),
            if dtype == GgmlType::I32 {
                vec![3]
            } else {
                vec![width as u64, 2]
            },
            dtype,
        ));
        if !keep_inline {
            kv.retain(|(k, _)| *k != key);
        }
    }
    (kv, tensors, payloads)
}

fn iq4_rows(rows: usize, base: usize) -> Vec<u8> {
    let mut raw = vec![0u8; rows * 144];
    for (i, row) in raw.as_chunks_mut::<144>().0.iter_mut().enumerate() {
        for block in row.as_chunks_mut::<18>().0 {
            block[..2]
                .copy_from_slice(&half::f16::from_f32((base + i + 1) as f32 / 128.0).to_le_bytes());
            block[2..].fill(0x98);
        }
    }
    raw
}

fn payload(name: &str, shape: &[u64], dtype: GgmlType, controls: &Payloads) -> Option<Vec<u8>> {
    Some(if let Some(bytes) = controls.get(name) {
        bytes.clone()
    } else if engram::is_table(name) {
        iq4_rows(
            shape[1] as usize,
            if name.starts_with("blk.1.") { 100 } else { 0 },
        )
    } else {
        let (block, bytes) = dtype.storage_layout().unwrap();
        vec![0; (shape.iter().product::<u64>() / block * bytes) as usize]
    })
}

fn populated(kv: Metadata, tensors: Tensors, controls: Payloads) -> Arc<GgufFile> {
    write_fixture_payload(kv, tensors, false, |n, s, d| payload(n, s, d, &controls))
}

#[test]
fn tensor_controls_equal_inline_and_ple_without_float_rounding() {
    let (kv, tensors, controls) = parts(true);
    let both = plan(&populated(kv.clone(), tensors.clone(), controls.clone())).unwrap();
    let inline = plan(&populated(
        kv.clone(),
        tensors
            .iter()
            .filter(|(n, _, _)| !controls.contains_key(n))
            .cloned()
            .collect(),
        Payloads::new(),
    ))
    .unwrap();
    let (mut kv, tensors, controls) = parts(false);
    let tensor = plan(&populated(kv.clone(), tensors.clone(), controls.clone())).unwrap();
    assert_eq!(both.engram_hash, inline.engram_hash);
    assert_eq!(tensor.engram_hash, inline.engram_hash);
    assert_eq!(tensor.token_map, inline.token_map);
    assert_eq!(tensor.raw_pad, 2);
    assert_eq!(tensor.engram_layers, vec![0, 1]);
    assert_eq!(tensor.engram_offsets[24], 0);
    assert_eq!(tensor.engram_multipliers[0], 16_777_217);
    assert_eq!(tensor.engram_multipliers[1], 4_839_876_093_313);
    assert_eq!(tensor.controls.payload_bytes_read, 12 + 384 + 384 + 64);
    assert_eq!(inline.controls.payload_bytes_read, 0);
    assert!(tensor.controls.cold_cpu_bytes > tensor.controls.payload_bytes_read);
    assert_eq!(tensor.bytes, inline.bytes);
    kv.push((
        "deepseek41.ple.head_offsets",
        Value::U64s(
            (0..24)
                .map(|i| i * 2)
                .chain((0..24).map(|i| 48 + i * 3))
                .collect(),
        ),
    ));
    assert_eq!(
        plan(&populated(kv.clone(), tensors.clone(), controls.clone()))
            .unwrap()
            .engram_hash,
        inline.engram_hash
    );
    let Value::U64s(offsets) = &mut kv.last_mut().unwrap().1 else {
        unreachable!()
    };
    offsets[24] = 0;
    assert!(plan(&populated(kv, tensors, controls)).is_err());
}

#[test]
fn exact_i32_and_i64_domains_and_malformed_controls() {
    let (kv, tensors, controls) = parts(false);
    for (_, name, _, _) in CONTROLS {
        for dtype in [GgmlType::F32, GgmlType::I16] {
            let mut bad = tensors.clone();
            bad.iter_mut().find(|(n, _, _)| n == name).unwrap().2 = dtype;
            // Wrong types are rejected before payload access.
            assert!(
                plan(&write_fixture(kv.clone(), bad)).is_err(),
                "{name} {dtype:?}"
            );
        }
        let mut bad = tensors.clone();
        bad.iter_mut().find(|(n, _, _)| n == name).unwrap().1[0] += 1;
        assert!(
            plan(&write_fixture(kv.clone(), bad)).is_err(),
            "{name} shape"
        );
        let mut duplicate = tensors.clone();
        duplicate.push(tensors.iter().find(|(n, _, _)| n == name).unwrap().clone());
        assert!(
            GgufFile::from_opened_file(
                fixture_file(kv.clone(), duplicate, false, |n, s, d| payload(
                    n, s, d, &controls
                )),
                "duplicate-control.gguf"
            )
            .is_err(),
            "{name} duplicate"
        );
        let mut bad = controls.clone();
        let width = if name == "engram_token_map.weight" {
            4
        } else {
            8
        };
        bad.get_mut(name).unwrap()[..width].fill(0xff);
        assert!(
            plan(&populated(kv.clone(), tensors.clone(), bad)).is_err(),
            "{name} negative"
        );
    }
    for (name, value) in [
        ("engram_token_map.weight", 2i64),
        ("engram_primes.weight", 1),
        ("engram_offsets.weight", 1),
        ("engram_multipliers.weight", 2),
    ] {
        let mut bad = controls.clone();
        let width = if name == "engram_token_map.weight" {
            4
        } else {
            8
        };
        bad.get_mut(name).unwrap()[..width].copy_from_slice(&value.to_le_bytes()[..width]);
        assert!(
            plan(&populated(kv.clone(), tensors.clone(), bad)).is_err(),
            "{name} {value}"
        );
    }
    let mut overflow_kv = kv.clone();
    overflow_kv
        .iter_mut()
        .find(|(k, _)| *k == "deepseek41.engram.compressed_vocab_size")
        .unwrap()
        .1 = Value::U32(3);
    let mut overflow = controls.clone();
    overflow.get_mut("engram_multipliers.weight").unwrap()[..8]
        .copy_from_slice(&i64::MAX.to_le_bytes());
    assert!(plan(&populated(overflow_kv, tensors.clone(), overflow)).is_err());
    let g = write_fixture_payload(
        vec![],
        vec![("engram_token_map.weight".into(), vec![3], GgmlType::I32)],
        false,
        |_, _, _| {
            Some(
                [16_777_217i32, i32::MAX, 0]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            )
        },
    );
    let values = controls::merge(
        &g,
        "engram_token_map.weight",
        GgmlType::I32,
        &[3],
        None,
        &mut ControlAccounting::default(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(values, vec![16_777_217, i32::MAX as u64, 0]);
}

#[test]
fn metadata_tensor_conflicts_are_rejected_for_every_control() {
    let (kv, tensors, controls) = parts(true);
    for (key, _, _, _) in CONTROLS {
        let mut bad = kv.clone();
        match &mut bad.iter_mut().find(|(k, _)| *k == key).unwrap().1 {
            Value::Ints(v) => v[0] += 1,
            Value::U64s(v) => v[0] += 1,
            _ => unreachable!(),
        }
        let error = plan(&populated(bad, tensors.clone(), controls.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting"), "{key}: {error}");
    }
}

#[test]
fn grouped_output_a_accepts_only_the_exact_serialized_axes() {
    let c = Config {
        heads: 4,
        head_dim: 2,
        groups: 2,
        output_rank: 3,
        ..Config::default()
    };
    for (shape, valid) in [
        (vec![4, 6], true),
        (vec![4, 3, 2], true),
        (vec![4, 2, 3], false),
        (vec![2, 6, 2], false),
        (vec![4, 3, 2, 2], false),
    ] {
        let t = TensorDesc {
            name: "blk.0.attn_output_a.weight".into(),
            shape,
            dtype: GgmlType::BF16,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 48,
        };
        assert_eq!(names::validate_shape(&t, &c).is_ok(), valid);
    }
    assert_eq!(
        names::expected_shape("blk.0.attn_output_a.weight", &c),
        Some(vec![4, 6])
    );
}

const GEOMETRY_ALIASES: [(&str, &str); 5] = [
    ("deepseek41.engram.key_length", "deepseek41.engram_head_dim"),
    ("deepseek41.engram.head_count", "deepseek41.engram_n_heads"),
    (
        "deepseek41.engram.max_ngram_size",
        "deepseek41.engram_max_ngram_size",
    ),
    (
        "deepseek41.engram.compressed_vocab_size",
        "deepseek41.engram_compressed_vocab_size",
    ),
    (
        "deepseek41.engram.pad_token_id",
        "deepseek41.engram_pad_token_id",
    ),
];

#[test]
fn gsq_geometry_aliases_and_conflicts() {
    let (kv, tensors, controls) = parts(false);
    let reference = plan(&populated(kv.clone(), tensors.clone(), controls.clone())).unwrap();
    let mut aliases = kv.clone();
    for (key, _) in &mut aliases {
        if let Some((_, alias)) = GEOMETRY_ALIASES.iter().find(|(old, _)| old == key) {
            *key = alias;
        }
    }
    aliases.push(("deepseek41.engram_vocab_size", Value::U32(16_000_000)));
    let p = plan(&populated(
        aliases.clone(),
        tensors.clone(),
        controls.clone(),
    ))
    .unwrap();
    assert_eq!(p.engram_hash, reference.engram_hash);
    assert_eq!(p.raw_pad, reference.raw_pad);
    for (old, alias) in GEOMETRY_ALIASES {
        let mut conflict = aliases.clone();
        let mut value = kv.iter().find(|(k, _)| *k == old).unwrap().1.clone();
        let Value::U32(n) = &mut value else {
            unreachable!()
        };
        *n += 1;
        conflict.push((old, value));
        let error = plan(&populated(conflict, tensors.clone(), controls.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting"), "{alias}: {error}");
        let mut malformed = aliases.clone();
        malformed.iter_mut().find(|(k, _)| *k == alias).unwrap().1 = Value::F32(f32::NAN);
        assert!(plan(&populated(malformed, tensors.clone(), controls.clone())).is_err());
    }
}

struct Package(std::path::PathBuf);
impl Package {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen-v41-gsq-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn write(&self, name: &str, mut file: std::fs::File) {
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.0.join(name))
            .unwrap();
        std::io::copy(&mut file, &mut out).unwrap();
    }
    fn backbone(&self) -> Arc<GgufFile> {
        Arc::new(GgufFile::open(self.0.join("model-00001-of-00004.gguf")).unwrap())
    }
}
impl Drop for Package {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn package(case: &str) -> Package {
    let package = Package::new();
    let (mut kv, mut tensors, controls) = parts(false);
    let mut tables = Vec::new();
    tensors.retain(|t| {
        if engram::is_table(&t.0) {
            tables.push(t.clone());
            case == "inline"
        } else {
            true
        }
    });
    if case == "geometry" {
        tables[0].1[1] += 1;
    }
    if case == "duplicate" {
        tables[1] = tables[0].clone();
    }
    kv.extend([
        (
            "deepseek41.engram.file",
            Value::Str(match case {
                "unsafe" => "../aux.gguf",
                "absolute" => "/aux.gguf",
                "empty" => "",
                "backslash" => "dir\\aux.gguf",
                _ => "aux.gguf",
            }),
        ),
        (
            "deepseek41.engram.source",
            Value::Str("opaque-declared-identity-not-a-hash"),
        ),
    ]);
    if case == "no-source" {
        kv.retain(|(k, _)| *k != "deepseek41.engram.source");
    }
    if case == "encoding" {
        kv.push(("deepseek41.engram.encoding", Value::Str("Q8_0")));
    }
    let total = tensors.len() as u32;
    for index in 0..4 {
        let mut metadata = if index == 0 { kv.clone() } else { vec![] };
        metadata.extend([
            ("split.no", Value::U32(index)),
            ("split.count", Value::U32(4)),
            ("split.tensors.count", Value::U32(total)),
        ]);
        let shard = tensors
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 == index as usize)
            .map(|(_, t)| t.clone())
            .collect();
        package.write(
            &format!("model-{:05}-of-00004.gguf", index + 1),
            fixture_file(metadata, shard, false, |n, s, d| {
                payload(n, s, d, &controls)
            }),
        );
    }
    if case != "missing" {
        let mut auxiliary_metadata = vec![
            (
                "general.architecture",
                Value::Str(if case == "arch" {
                    "deepseek41"
                } else {
                    "deepseek41-engram"
                }),
            ),
            (
                "deepseek41.engram.source",
                Value::Str(if case == "source" {
                    "different"
                } else {
                    "opaque-declared-identity-not-a-hash"
                }),
            ),
        ];
        if case == "aux-no-source" {
            auxiliary_metadata.pop();
        }
        if case == "aux-split" {
            auxiliary_metadata.push(("split.count", Value::U32(2)));
        }
        package.write(
            "aux.gguf",
            fixture_file(auxiliary_metadata, tables, false, |n, s, d| {
                payload(n, s, d, &controls)
            }),
        );
    }
    package
}

#[test]
fn separate_four_shard_companion_groups_iq4_rows_and_disk_only_storage() {
    let package = package("valid");
    let g = package.backbone();
    let p = plan(&g).unwrap();
    assert_eq!(g.get_u64("split.count"), Some(4));
    assert_eq!(
        g.get_u64("split.tensors.count"),
        Some(g.tensors.len() as u64)
    );
    assert_eq!(g.shard_count(), 4);
    assert_eq!(p.source_groups().backbone.len(), 4);
    assert_eq!(p.source_groups().auxiliary.len(), 1);
    assert_eq!(p.bytes.disk_engram, 120 * 144);
    let storage = resident_storage_plan(&g, 16384, 256 * 1024 * 1024).unwrap();
    assert!(storage.windows.iter().all(|w| w.window.shard_idx < 4));
    assert!(
        p.engram_context
            .auxiliary
            .as_ref()
            .unwrap()
            .tensors
            .iter()
            .all(|t| !resident_tensor(t, p.layers))
    );
    let disk = DiskEngram::new(g.clone()).unwrap();
    assert_eq!(disk.source_groups(), p.source_groups());
    drop(p);
    drop(g);
    for (layer, rows, base) in [(0, 48, 0), (1, 72, 100)] {
        let ids = [rows - 1, 0, 13, rows - 1, 1];
        let actual = disk.read_rows(layer, &ids).unwrap();
        let descriptor = TensorDesc {
            name: "row-oracle".into(),
            shape: vec![256, rows],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: rows * 144,
        };
        let expected =
            crate::codec::dequant_to_f32(&descriptor, &iq4_rows(rows as usize, base)).unwrap();
        for (&id, row) in ids.iter().zip(actual.as_chunks::<256>().0) {
            assert_eq!(row, &expected[id as usize * 256..(id as usize + 1) * 256]);
        }
        assert!(actual.iter().any(|v| *v != 0.0));
        assert!(disk.read_rows(layer, &[rows]).is_err());
        assert!(disk.read_rows(layer, &[u64::MAX]).is_err());
        assert!(disk.read_rows(layer, &[]).unwrap().is_empty());
    }
    assert!(disk.read_rows(2, &[0]).is_err());
    assert!(disk.read_rows(0, &vec![0; 32769]).is_err());
}

#[test]
fn companion_rejects_missing_source_arch_geometry_duplicates_and_ambiguity() {
    for case in [
        "missing",
        "source",
        "arch",
        "geometry",
        "duplicate",
        "inline",
        "unsafe",
        "absolute",
        "empty",
        "backslash",
        "no-source",
        "aux-no-source",
        "aux-split",
        "encoding",
    ] {
        let package = package(case);
        assert!(plan(&package.backbone()).is_err(), "{case}");
    }
    let package = package("valid");
    let aux = Arc::new(GgufFile::open(package.0.join("aux.gguf")).unwrap());
    assert_eq!(crate::model_family::ModelFamily::detect(&aux), None);
    assert!(plan(&aux).is_err());
    assert!(DiskEngram::new(aux).is_err());
}

#[test]
fn logical_snapshot_symlinks_and_retained_descriptor_stamps() {
    let package = package("valid");
    let blobs = Package::new();
    for index in 1..=4 {
        let name = format!("model-{index:05}-of-00004.gguf");
        let blob = blobs.0.join(format!("blob-{index}"));
        std::fs::rename(package.0.join(&name), &blob).unwrap();
        std::os::unix::fs::symlink(&blob, package.0.join(&name)).unwrap();
    }
    let g = package.backbone();
    let p = plan(&g).unwrap();
    assert_eq!(p.source_groups().auxiliary.len(), 1);
    let disk = DiskEngram::new(g).unwrap();
    let mut changed = std::fs::OpenOptions::new()
        .write(true)
        .open(package.0.join("aux.gguf"))
        .unwrap();
    changed.seek(SeekFrom::End(-1)).unwrap();
    changed.write_all(&[1]).unwrap();
    assert!(disk.revalidate_sources().is_err());
    assert!(disk.read_rows(0, &[0]).is_err());
}

#[test]
fn retained_auxiliary_fd_does_not_follow_replaced_path() {
    let package = package("valid");
    let g = package.backbone();
    let p = plan(&g).unwrap();
    let aux = p.engram_context.auxiliary.as_ref().unwrap();
    let t = aux.find("blk.0.engram_embd.weight").unwrap();
    let mut expected = [0u8; 144];
    aux.read_shard_exact_at(0, t.data_offset, &mut expected)
        .unwrap();
    std::fs::rename(package.0.join("aux.gguf"), package.0.join("retained.gguf")).unwrap();
    std::fs::write(package.0.join("aux.gguf"), b"not the retained source").unwrap();
    let mut actual = [0u8; 144];
    aux.read_shard_exact_at(0, t.data_offset, &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
    assert!(p.engram_context.revalidate(&g).is_err());
}

/// The private prefixes are NOT complete weights. Build temporary sparse files
/// with authentic headers and ONLY control payloads to exercise CPU contracts.
/// Never call ResidentWeights::load, DiskEngram::read_rows, or any GPU API here.
#[test]
#[ignore = "CPU-only sparse header/control replay; requires V41_GSQ_AUDIT_DIR"]
fn cpu_only_sparse_header_control_replay() {
    use sha2::{Digest, Sha256};
    let root = std::path::PathBuf::from(
        std::env::var_os("V41_GSQ_AUDIT_DIR").expect("V41_GSQ_AUDIT_DIR required"),
    );
    let mut reference = None;
    for quant in ["IQ2_XS", "IQ2_XXS"] {
        let dir = Package::new();
        let mut names: Vec<_> = (1..=4)
            .map(|i| format!("DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quant}-{i:05}-of-00004.gguf"))
            .collect();
        names.push("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf".into());
        let mut control_bytes = 0u64;
        for name in &names {
            let header: serde_json::Value = serde_json::from_slice(
                &std::fs::read(root.join("headers").join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(
                header["source_revision"],
                "cb02422c900df489cea5833e8a92171224059b3c"
            );
            assert_eq!(header["full_payload_hash_verified"], false);
            let mut prefix =
                std::fs::File::open(root.join("prefixes").join(format!("{name}.prefix.bin")))
                    .unwrap();
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.0.join(name))
                .unwrap();
            output
                .set_len(header["source_size"].as_u64().unwrap())
                .unwrap();
            std::io::copy(
                &mut (&mut prefix).take(header["data_offset"].as_u64().unwrap()),
                &mut output,
            )
            .unwrap();
            for t in header["tensors"].as_array().unwrap() {
                let tensor_name = t["name"].as_str().unwrap();
                if !CONTROLS.iter().any(|(_, n, _, _)| *n == tensor_name) {
                    continue;
                }
                let start = t["absolute_data_offset"].as_u64().unwrap();
                let length = t["payload_bytes"].as_u64().unwrap();
                let mut raw = vec![0; length as usize];
                prefix.seek(SeekFrom::Start(start)).unwrap();
                prefix.read_exact(&mut raw).unwrap();
                let saved = std::fs::read(
                    root.join("controls")
                        .join(quant)
                        .join(format!("{tensor_name}.bin")),
                )
                .unwrap();
                assert_eq!(raw, saved);
                if tensor_name == "engram_token_map.weight" {
                    assert_eq!(
                        format!("{:x}", Sha256::digest(&raw)),
                        "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
                    );
                }
                output.seek(SeekFrom::Start(start)).unwrap();
                output.write_all(&raw).unwrap();
                control_bytes += length;
            }
        }
        let g = GgufFile::open(dir.0.join(&names[0])).unwrap();
        assert_eq!(g.shard_count(), 4);
        assert_eq!(g.tensors.len(), 1008);
        let p = metadata::parse(&g).unwrap();
        assert_eq!(p.engram_hash, HashConfig::default());
        assert_eq!(p.token_map.len(), 129280);
        assert_eq!(p.token_map.iter().copied().max(), Some(99091));
        assert_eq!(p.raw_pad, 2);
        assert_eq!(p.engram_layers, vec![1, 14]);
        assert_eq!(p.controls.payload_bytes_read, 517952);
        assert_eq!(p.controls.payload_bytes_read, control_bytes);
        assert_eq!(p.controls.cold_cpu_bytes, 3105632);
        assert_eq!(p.controls.retained_cpu_bytes, 519648);
        assert_eq!(p.config.candidate_block, 8);
        assert_eq!(p.config.candidate_topk, 2048);
        assert_eq!(p.schedule, Schedule::default());
        assert_eq!(p.compression_ratios.len(), 43);
        assert_eq!(p.source_groups().backbone.len(), 4);
        assert_eq!(p.source_groups().auxiliary.len(), 1);
        assert!(
            p.engram_context
                .tables
                .iter()
                .all(|t| engram::row_stride(&t.tensor).unwrap() == 144)
        );
        let controls = (p.token_map.clone(), p.engram_hash.clone());
        if let Some(expected) = &reference {
            assert_eq!(&controls, expected);
        } else {
            reference = Some(controls);
        }
        let validated = plan(&g).unwrap();
        assert_eq!(validated.layers, 40);
        assert_eq!(validated.controls, p.controls);
        assert_eq!(validated.source_groups(), p.source_groups());
        assert_eq!(validated.token_map, p.token_map);
        assert_eq!(validated.engram_hash, p.engram_hash);
        let banks: Vec<_> = g
            .tensors
            .iter()
            .filter(|t| names::bank(&t.name).is_some_and(|(l, _)| l < validated.layers))
            .collect();
        assert_eq!(banks.len(), 120);
        assert!(
            banks
                .iter()
                .all(|t| metal::all_slots_down_native_supported(t.dtype))
        );
        assert_eq!(
            banks.iter().map(|t| t.n_bytes).sum::<u64>(),
            validated.bytes.packed_experts
        );
        assert_eq!(
            banks.iter().filter(|t| t.dtype == GgmlType::IQ1_M).count(),
            if quant == "IQ2_XXS" { 60 } else { 0 }
        );
        assert_eq!(validated.bytes.disk_engram, 110595290400);
        eprintln!(
            "CPU SPARSE CONTRACT ONLY {quant}: controls={:?} groups=4+1 banks=120 bytes={:?}; no weight/table payload qualification",
            validated.controls, validated.bytes
        );
    }
}
