use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[test]
fn ephemeral_backing_bounds_padding_and_cleanup() {
    use crate::{metal::MetalContext, tensor::GgmlType};
    use backing::BackingWriter;
    use objc2_metal::MTLBuffer;
    let f = Fixture::new();
    let ctx = MetalContext::new().unwrap();
    let mut writer = BackingWriter::new(&f.0, 4).unwrap();
    assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 0);
    assert!(writer.write(&[0; 5]).is_err());
    writer.write(&[1, 2]).unwrap();
    assert!(writer.tensor(&ctx, vec![1], GgmlType::F32).is_err());
    let mut writer = BackingWriter::new(&f.0, 4).unwrap();
    writer.write(&[1, 2, 3, 4]).unwrap();
    assert!(writer.write(&[5]).is_err());
    let tensor = writer.tensor(&ctx, vec![1], GgmlType::F32).unwrap();
    assert_eq!(
        tensor.buffer.length() as u64,
        backing::padded_bytes(4).unwrap()
    );
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<u8>(),
            tensor.buffer.length(),
        )
    };
    assert_eq!(&bytes[..4], &[1, 2, 3, 4]);
    assert!(bytes[4..].iter().all(|&b| b == 0));
    drop(tensor);
    assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 0);
    assert!(BackingWriter::new(&f.0.join("absent"), 4).is_err());
    assert!(BackingWriter::new(&f.0, u64::MAX).is_err());
    assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 0);
}

#[test]
fn mapped_dense_conversion_and_matvec_outlive_loader() {
    use crate::{metal::*, tensor::GgmlType};
    let f = Fixture::new();
    let out = Fixture::new();
    f.shard(
        "dense.safetensors",
        &[(
            "dense.weight",
            "BF16",
            &[2, 2],
            &[128, 63, 0, 64, 64, 64, 128, 64],
        )],
    );
    let source = SourceStore::open(&f.0).unwrap();
    let ctx = MetalContext::new().unwrap();
    let (weight, read) =
        resident::materialize_dense(&ctx, &source, "dense.weight", &out.0).unwrap();
    assert_eq!(read, 8);
    assert_eq!(
        weight.provenance(),
        MetalTensorProvenance::RetainedGgufReadOnly
    );
    drop(source);
    assert_eq!(std::fs::read_dir(&out.0).unwrap().count(), 0);
    assert_eq!(read_back_f32(&weight.buffer, 4), [1., 2., 3., 4.]);
    let x = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[2f32, 3.]),
        vec![2],
        GgmlType::F32,
    )
    .unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![2]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_vec_f32(&ctx, enc, &weight, &x, &y, 2, 2)
    })
    .unwrap();
    assert_eq!(read_back_f32(&y.buffer, 2), [8., 18.]);
}

#[test]
fn residency_accepts_omitted_budget_but_preserves_real_limits() {
    use crate::metal::{MetalMemoryAdmissionReason as R, MetalMemorySignals};
    let mut signals = MetalMemorySignals {
        recommended_max_bytes: 1000,
        current_allocated_bytes: 100,
        process_limit_remaining_bytes: Some(0),
    };
    assert_eq!(
        resident::residency_admission(800, 20, signals).reason,
        R::AdmittedProcessBudgetOmitted
    );
    assert_eq!(
        resident::residency_admission(901, 20, signals).reason,
        R::WorkingSetInsufficient
    );
    signals.process_limit_remaining_bytes = None;
    assert_eq!(
        resident::residency_admission(800, 20, signals).reason,
        R::ProcessSignalUnavailable
    );
    signals.process_limit_remaining_bytes = Some(819);
    assert_eq!(
        resident::residency_admission(800, 20, signals).reason,
        R::ProcessInsufficient
    );
}

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "qwen-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn shard(&self, file: &str, tensors: &[(&str, &str, &[u64], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert((*name).into(), serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().div_ceil(8) * 8 + 16, b' ');
        let mut out = File::create(self.0.join(file)).unwrap();
        out.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        out.write_all(&header).unwrap();
        out.write_all(&payload).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn split_shards_padded_headers_fp8_block_and_engram_rows() {
    let f = Fixture::new();
    f.shard(
        "a.safetensors",
        &[
            ("p.weight", "F8_E4M3", &[33, 33], &vec![0x38; 33 * 33]),
            ("e.weight", "F8_E4M3", &[2, 32], &[0x40; 64]),
        ],
    );
    f.shard(
        "b.safetensors",
        &[
            ("p.scale", "F8_E8M0", &[2, 2], &[127, 128, 129, 130]),
            ("e.scale", "F8_E8M0", &[2, 1], &[126, 129]),
        ],
    );
    let s = SourceStore::open(&f.0).unwrap();
    s.stream_without_cache().unwrap();
    s.stream_without_cache().unwrap();
    let values = s.decode_rows("p.weight", 31..33, false).unwrap();
    assert_eq!(&values[..32], &[1.0; 32]);
    assert_eq!(values[32], 2.0);
    assert_eq!(&values[33..65], &[4.0; 32]);
    assert_eq!(values[65], 8.0);
    assert_eq!(
        s.decode_rows("e.weight", 0..2, true).unwrap(),
        [vec![1.0; 32], vec![8.0; 32]].concat()
    );
    assert!(s.decode_rows("e.weight", 1..3, true).is_err());
    assert!(s.decode_rows("e.weight", 2..1, true).is_err());
    assert!(s.bounded_read("e.weight", 63..65).is_err());
    assert!(
        s.bounded_read("e.weight", 0..MAX_READ_BYTES as u64 + 1)
            .is_err()
    );
    assert!(s.decode_rows("e.weight", 2..2, true).unwrap().is_empty());
}

#[test]
fn bf16_f32_scale_specials_and_wrong_scale_geometry() {
    let f = Fixture::new();
    f.shard(
        "a.safetensors",
        &[
            ("b", "BF16", &[2], &[128, 63, 0, 192]),
            (
                "f",
                "F32",
                &[1, 2],
                &[1f32.to_le_bytes(), (-3f32).to_le_bytes()].concat(),
            ),
            ("x.weight", "F8_E4M3", &[2, 32], &[0x38; 64]),
            ("x.scale", "F8_E8M0", &[2, 1], &[127, 255]),
        ],
    );
    let s = SourceStore::open(&f.0).unwrap();
    assert_eq!(s.decode_rows("b", 0..2, false).unwrap(), vec![1.0, -2.0]);
    assert_eq!(s.decode_rows("f", 0..1, false).unwrap(), vec![1.0, -3.0]);
    assert!(s.decode_rows("x.weight", 0..2, false).is_err());
    assert!(s.decode_rows("x.weight", 1..2, true).is_err());
    assert_eq!(ue8m0(0), 2f32.powi(-127));
    assert_eq!(ue8m0(127), 1.0);
    assert_eq!(ue8m0(254), 2f32.powi(127));
    assert!(ue8m0(255).is_nan());
    assert_eq!(fp8(0x7e), 448.0);
    assert_eq!(fp8(1), 1.0 / 512.0);
    assert_eq!(fp8(0xb8), -1.0);
    assert!(fp8(0x7f).is_nan());
}

#[test]
fn packed_split_reads_repack_with_canaries_without_expansion() {
    let f = Fixture::new();
    f.shard(
        "w.safetensors",
        &[("p.weight", "I8", &[2, 16], &[0x21; 32])],
    );
    f.shard(
        "s.safetensors",
        &[("p.scale", "F8_E8M0", &[2, 1], &[127, 128])],
    );
    let store = SourceStore::open(&f.0).unwrap();
    let codes = store.bounded_read("p.weight", 0..32).unwrap();
    let scales = store.bounded_read("p.scale", 0..2).unwrap();
    let mut out = [0xa5; 36];
    super::super::mxfp4::repack_into(2, 32, &codes, &scales, &mut out[1..35]).unwrap();
    assert_eq!(out[0], 0xa5);
    assert_eq!(out[35], 0xa5);
    assert_eq!(out[1], 127);
    assert_eq!(out[18], 128);
    assert_eq!(&out[2..6], &[0x11, 0x22, 0x11, 0x22]);
    assert!(store.decode_rows("p.weight", 0..2, false).is_err());
}

#[test]
fn index_progress_traversal_collision_and_truncation() {
    let f = Fixture::new();
    let index = f.0.join("model.safetensors.index.json");
    std::fs::write(&index, r#"{"weight_map":{"a":"missing.safetensors"}}"#).unwrap();
    let s = SourceStore::open(&f.0).unwrap();
    assert_eq!(s.missing_shards, ["missing.safetensors"]);
    assert_eq!(s.descriptors().count(), 0);
    for path in [
        "../bad.safetensors",
        "/tmp/bad.safetensors",
        "sub/bad.safetensors",
    ] {
        std::fs::write(
            &index,
            serde_json::json!({"weight_map":{"a":path}}).to_string(),
        )
        .unwrap();
        assert!(SourceStore::open(&f.0).is_err());
    }
    std::fs::remove_file(index).unwrap();
    f.shard("a.safetensors", &[("x", "BF16", &[1], &[0, 0])]);
    f.shard("b.safetensors", &[("model.x", "BF16", &[1], &[0, 0])]);
    assert!(SourceStore::open(&f.0).is_err());
    std::fs::remove_file(f.0.join("b.safetensors")).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(f.0.join("a.safetensors"))
        .unwrap();
    file.set_len(9).unwrap();
    assert!(SourceStore::open(&f.0).is_err());
}

#[test]
fn old_profile_preflight_rejects_missing_experts_before_allocation() {
    let f = Fixture::new();
    let s = SourceStore::open(&f.0).unwrap();
    let config = serde_json::json!({"text_config":{"hidden_size":5120,"moe_intermediate_size":2304,"num_hidden_layers":40,"n_routed_experts":272,"vocab_size":129280},"quantization_config":{"weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}});
    let err = ResidentWeights::plan(&s, &s, &config).unwrap_err();
    assert!(err.contains("layers.0.ffn.experts.0.w1.weight"));
    let mut wrong = config;
    wrong["text_config"]["n_routed_experts"] = 384.into();
    assert!(
        ResidentWeights::plan(&s, &s, &wrong)
            .unwrap_err()
            .contains("not old REAP")
    );
}

#[test]
fn disk_engram_reads_only_requested_rows_and_preserves_order() {
    let f = Fixture::new();
    let mut values = vec![0x7f; 3 * 256];
    values[256..512].fill(0x38);
    values[512..].fill(0x40);
    f.shard(
        "model-00047-of-00048.safetensors",
        &[
            (
                "layers.1.engram.embed.weight",
                "F8_E4M3",
                &[3, 256],
                &values,
            ),
            (
                "layers.1.engram.embed.scale",
                "F8_E8M0",
                &[3, 8],
                &[127; 24],
            ),
        ],
    );
    // A REAP index does not need to list the separately downloaded base pair.
    std::fs::write(
        f.0.join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"absent.safetensors"}}"#,
    )
    .unwrap();
    let source = Arc::new(SourceStore::open_engram(&f.0).unwrap());
    assert!(DiskEngram::new(source.clone()).is_err());
    // Small test table exercises exactly the production row-only read path.
    let disk = DiskEngram { source };
    let result = disk.decode_rows(1, &[2, 1, 2]).unwrap();
    assert_eq!(&result[..256], &[2.0; 256]);
    assert_eq!(&result[256..512], &[1.0; 256]);
    assert_eq!(&result[512..], &[2.0; 256]);
    assert!(disk.decode_rows(1, &[3]).is_err());
    assert!(disk.decode_rows(1, &[0]).is_err());
    assert!(disk.decode_rows(2, &[1]).is_err());
}

#[test]
fn tiny_metal_bank_materializes_directly_in_native_layout() {
    use objc2_metal::MTLBuffer;
    let f = Fixture::new();
    f.shard(
        "w.safetensors",
        &[
            (
                "layers.0.ffn.experts.0.w1.weight",
                "I8",
                &[2, 16],
                &[0x21; 32],
            ),
            (
                "layers.0.ffn.experts.1.w1.weight",
                "I8",
                &[2, 16],
                &[0x43; 32],
            ),
        ],
    );
    f.shard(
        "s.safetensors",
        &[
            (
                "layers.0.ffn.experts.0.w1.scale",
                "F8_E8M0",
                &[2, 1],
                &[127, 128],
            ),
            (
                "layers.0.ffn.experts.1.w1.scale",
                "F8_E8M0",
                &[2, 1],
                &[129, 130],
            ),
        ],
    );
    let source = SourceStore::open(&f.0).unwrap();
    let ctx = crate::metal::MetalContext::new().unwrap();
    let tensor = resident::materialize_bank(&ctx, &source, 0, "w1", 2, 32, 2, &f.0).unwrap();
    assert_eq!(tensor.shape, [32, 2, 2]);
    assert_eq!(tensor.dtype, crate::tensor::GgmlType::MXFP4);
    assert_eq!(tensor.n_bytes(), 68);
    assert_eq!(
        tensor.provenance(),
        crate::metal::MetalTensorProvenance::RetainedGgufReadOnly
    );
    drop(source);
    assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 2);
    let bytes =
        unsafe { std::slice::from_raw_parts(tensor.buffer.contents().as_ptr().cast::<u8>(), 68) };
    assert_eq!(
        [bytes[0], bytes[17], bytes[34], bytes[51]],
        [127, 128, 129, 130]
    );
    assert_eq!(&bytes[1..5], &[0x11, 0x22, 0x11, 0x22]);
    assert_eq!(&bytes[35..39], &[0x33, 0x44, 0x33, 0x44]);
    let expected: Vec<u8> = [
        (127, 0x11, 0x22),
        (128, 0x11, 0x22),
        (129, 0x33, 0x44),
        (130, 0x33, 0x44),
    ]
    .into_iter()
    .flat_map(|(scale, lo, hi)| std::iter::once(scale).chain([lo, hi].into_iter().cycle().take(16)))
    .collect();
    assert_eq!(bytes, expected);
}

#[test]
fn complete_header_plan_counts_all_experts_and_rejects_last_bad_pair() {
    let f = Fixture::new();
    let file = Arc::new(File::create(f.0.join("metadata-only")).unwrap());
    let mut main = SourceStore::open(&f.0).unwrap();
    let mut base = SourceStore::open(&f.0).unwrap();
    let insert = |store: &mut SourceStore, name: String, dtype: SourceDtype, shape: Vec<u64>| {
        let bytes = shape.iter().product::<u64>() * dtype.byte_width();
        store.tensors.insert(
            name,
            StoredTensor {
                source_file: "metadata-only".into(),
                descriptor: TensorDescriptor {
                    dtype,
                    shape,
                    relative_bytes: 0..bytes,
                    file_bytes: 0..bytes,
                },
                file: file.clone(),
                stamp: crate::checkpoint_identity::source_stamp(&file).unwrap(),
            },
        );
    };
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            for part in ["w1", "w3", "w2"] {
                let (r, c) = if part == "w2" {
                    (5120, 2304)
                } else {
                    (2304, 5120)
                };
                insert(
                    &mut main,
                    format!("layers.{layer}.ffn.experts.{expert}.{part}.weight"),
                    SourceDtype::I8,
                    vec![r, c / 2],
                );
                insert(
                    &mut main,
                    format!("layers.{layer}.ffn.experts.{expert}.{part}.scale"),
                    SourceDtype::F8E8M0,
                    vec![r, c / 32],
                );
            }
        }
    }
    for key in ["embed.weight", "head.weight"] {
        insert(&mut main, key.into(), SourceDtype::Bf16, vec![129280, 5120]);
    }
    insert(
        &mut main,
        "norm.weight".into(),
        SourceDtype::Bf16,
        vec![5120],
    );
    insert(
        &mut main,
        "mtp.0.ignored.weight".into(),
        SourceDtype::I8,
        vec![7],
    );
    insert(
        &mut main,
        "vision.ignored.weight".into(),
        SourceDtype::I8,
        vec![5],
    );
    for (layer, rows) in [(1, 384006168), (14, 384016682)] {
        insert(
            &mut base,
            format!("layers.{layer}.engram.embed.weight"),
            SourceDtype::F8E4M3,
            vec![rows, 256],
        );
        insert(
            &mut base,
            format!("layers.{layer}.engram.embed.scale"),
            SourceDtype::F8E8M0,
            vec![rows, 8],
        );
        for suffix in ["q_weight", "k_weight"] {
            insert(
                &mut base,
                format!("layers.{layer}.engram.{suffix}"),
                SourceDtype::Bf16,
                vec![4, 5120],
            );
        }
        insert(
            &mut base,
            format!("layers.{layer}.engram.wkv.weight"),
            SourceDtype::F8E4M3,
            vec![25600, 6144],
        );
        insert(
            &mut base,
            format!("layers.{layer}.engram.wkv.scale"),
            SourceDtype::F8E8M0,
            vec![800, 192],
        );
    }
    let config = serde_json::json!({"text_config":{"hidden_size":5120,"moe_intermediate_size":2304,"num_hidden_layers":40,"n_routed_experts":272,"vocab_size":129280},"quantization_config":{"weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}});
    let bytes = ResidentWeights::plan(&main, &base, &config).unwrap();
    assert_eq!(bytes.packed_experts, 204_550_963_200);
    assert_eq!(
        bytes.dense_f32,
        (2 * 129280 * 5120 + 5120 + 2 * (2 * 4 * 5120 + 25600 * 6144)) * 4
    );
    assert_eq!(bytes.disk_engram, (384006168 + 384016682) * 264);
    assert_eq!(bytes.skipped_source, 12);
    assert_eq!(bytes.peak_staging, 12_533_760);
    assert_eq!(bytes.source_bytes_read, 0);
    main.tensors
        .get_mut("layers.39.ffn.experts.271.w2.scale")
        .unwrap()
        .descriptor
        .dtype = SourceDtype::I8;
    assert!(
        ResidentWeights::plan(&main, &base, &config)
            .unwrap_err()
            .contains("layers.39.ffn.experts.271.w2.scale")
    );
}

fn private_cache(f: &Fixture) -> cache::Cache {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&f.0, std::fs::Permissions::from_mode(0o700)).unwrap();
    cache::Cache::open(&f.0).unwrap()
}

#[test]
fn persistent_fp8_hashes_split_scales_and_reuses_without_reads() {
    use crate::metal::{MetalContext, read_back_f32};
    let f = Fixture::new();
    let out = Fixture::new();
    let cache = private_cache(&out);
    f.shard(
        "weight.safetensors",
        &[("p.weight", "F8_E4M3", &[33, 33], &[0x38; 1089])],
    );
    f.shard(
        "scale.safetensors",
        &[("p.scale", "F8_E8M0", &[2, 2], &[127, 128, 129, 130])],
    );
    let ctx = MetalContext::new().unwrap();
    let source = SourceStore::open(&f.0).unwrap();
    let (first, bytes) =
        resident::materialize_dense_impl(&ctx, &source, "p.weight", &out.0, Some(&cache)).unwrap();
    assert_eq!(bytes, 1093);
    let (hit, bytes) =
        resident::materialize_dense_impl(&ctx, &source, "p.weight", &out.0, Some(&cache)).unwrap();
    assert_eq!(bytes, 0);
    assert_eq!(source.payload_reads.load(Ordering::Relaxed), 1093);
    assert_eq!(
        read_back_f32(&hit.buffer, 1089),
        read_back_f32(&first.buffer, 1089)
    );
    let offset = source
        .tensor("p.scale")
        .unwrap()
        .descriptor
        .file_bytes
        .start;
    std::fs::OpenOptions::new()
        .write(true)
        .open(f.0.join("scale.safetensors"))
        .unwrap()
        .write_all_at(&[128], offset)
        .unwrap();
    assert!(
        resident::materialize_dense_impl(&ctx, &source, "p.weight", &out.0, Some(&cache)).is_err()
    );
    let changed = SourceStore::open(&f.0).unwrap();
    let (converted, bytes) =
        resident::materialize_dense_impl(&ctx, &changed, "p.weight", &out.0, Some(&cache)).unwrap();
    assert_eq!(bytes, 1093);
    assert_eq!(read_back_f32(&converted.buffer, 1), [2.]);
    assert_eq!(manifests(&out).len(), 2);
}

#[test]
fn persistent_cache_serializes_publishers_and_rejects_public_root() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::mpsc;
    use std::time::Duration;
    let out = Fixture::new();
    std::fs::set_permissions(&out.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(cache::Cache::open(&out.0).is_err());
    let cache = private_cache(&out);
    let path = out.0.clone();
    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        attempted_tx.send(()).unwrap();
        let _cache = cache::Cache::open(&path).unwrap();
        acquired_tx.send(()).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());
    drop(cache);
    acquired_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    waiter.join().unwrap();
}

fn manifests(f: &Fixture) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(&f.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect()
}

#[test]
fn persistent_dense_reuse_gpu_lifetime_and_content_invalidation() {
    use crate::{metal::*, tensor::GgmlType};
    use std::os::unix::fs::FileExt;
    let f = Fixture::new();
    let out = Fixture::new();
    let cache = private_cache(&out);
    f.shard(
        "dense.safetensors",
        &[
            (
                "dense.weight",
                "BF16",
                &[2, 2],
                &[128, 63, 0, 64, 64, 64, 128, 64],
            ),
            (
                "layers.1.engram.embed.weight",
                "F8_E4M3",
                &[4096],
                &[0x38; 4096],
            ),
        ],
    );
    let source = SourceStore::open(&f.0).unwrap();
    let ctx = MetalContext::new().unwrap();
    let (first, read) =
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .unwrap();
    assert_eq!(read, 8);
    assert_eq!(source.payload_reads.load(Ordering::Relaxed), 8);
    drop(first);
    drop(source);
    drop(cache);
    let cache = private_cache(&out);
    let source = SourceStore::open(&f.0).unwrap();
    let (weight, read) =
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .unwrap();
    assert_eq!(read, 0);
    assert_eq!(source.payload_reads.load(Ordering::Relaxed), 0);
    assert_eq!(
        weight.provenance(),
        MetalTensorProvenance::RetainedGgufReadOnly
    );
    let old_manifest = std::fs::read_to_string(&manifests(&out)[0]).unwrap();
    let tensor = source.tensor("dense.weight").unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(f.0.join("dense.safetensors"))
        .unwrap();
    file.write_all_at(&[0, 64], tensor.descriptor.file_bytes.start)
        .unwrap();
    file.sync_all().unwrap();
    assert!(
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .is_err()
    );
    drop(source);
    let source = SourceStore::open(&f.0).unwrap();
    let (changed, read) =
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .unwrap();
    assert_eq!(read, 8);
    assert_eq!(read_back_f32(&changed.buffer, 4), [2., 2., 3., 4.]);
    assert_eq!(manifests(&out).len(), 2);
    let old: serde_json::Value = serde_json::from_str(&old_manifest).unwrap();
    let old_body: serde_json::Value = serde_json::from_str(old["body"].as_str().unwrap()).unwrap();
    assert!(manifests(&out).iter().any(|p| {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let body: serde_json::Value = serde_json::from_str(m["body"].as_str().unwrap()).unwrap();
        body["content"] != old_body["content"]
    }));
    drop(changed);
    drop(source);
    drop(cache);
    // Removing names cannot invalidate retained mappings or replace their inodes.
    drop(out);
    let x = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[2f32, 3.]),
        vec![2],
        GgmlType::F32,
    )
    .unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![2]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_vec_f32(&ctx, enc, &weight, &x, &y, 2, 2)
    })
    .unwrap();
    assert_eq!(read_back_f32(&y.buffer, 2), [8., 18.]);
}

#[test]
fn persistent_expert_reuse_and_partial_writer_cleanup() {
    use crate::metal::{MetalContext, MetalTensorProvenance};
    use objc2_metal::MTLBuffer;
    let f = Fixture::new();
    let out = Fixture::new();
    let cache = private_cache(&out);
    f.shard(
        "codes.safetensors",
        &[(
            "layers.0.ffn.experts.0.w1.weight",
            "I8",
            &[2, 16],
            &[0x21; 32],
        )],
    );
    f.shard(
        "scales.safetensors",
        &[(
            "layers.0.ffn.experts.0.w1.scale",
            "F8_E8M0",
            &[2, 1],
            &[127, 128],
        )],
    );
    let source = SourceStore::open(&f.0).unwrap();
    let ctx = MetalContext::new().unwrap();
    let before = std::fs::read_dir(&out.0).unwrap().count();
    let mut partial = backing::BackingWriter::persistent(&out.0, 34).unwrap();
    partial.write(&[0; 3]).unwrap();
    assert!(partial.publish(&out.0.join("unfinished.buffer")).is_err());
    assert_eq!(std::fs::read_dir(&out.0).unwrap().count(), before);
    // An interrupted staging file is not a cache entry.
    std::fs::write(out.0.join("qwen-weights-interrupted"), [0; 17]).unwrap();
    let (first, n) =
        resident::materialize_bank_impl(&ctx, &source, 0, "w1", 2, 32, 1, &out.0, Some(&cache))
            .unwrap();
    assert_eq!(n, 34);
    let (hit, n) =
        resident::materialize_bank_impl(&ctx, &source, 0, "w1", 2, 32, 1, &out.0, Some(&cache))
            .unwrap();
    assert_eq!(n, 0);
    assert_eq!(source.payload_reads.load(Ordering::Relaxed), 34);
    assert_eq!(
        hit.provenance(),
        MetalTensorProvenance::RetainedGgufReadOnly
    );
    let raw = |t: &crate::metal::MetalTensor| unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), 34).to_vec()
    };
    assert_eq!(raw(&first), raw(&hit));
    assert_eq!(&raw(&hit)[..5], &[127, 0x11, 0x22, 0x11, 0x22]);
}

#[test]
fn persistent_rejects_manifest_version_shape_and_truncated_buffer() {
    use crate::{metal::MetalContext, tensor::GgmlType};
    let f = Fixture::new();
    let out = Fixture::new();
    let cache = private_cache(&out);
    f.shard(
        "dense.safetensors",
        &[(
            "dense.weight",
            "BF16",
            &[2, 2],
            &[128, 63, 0, 64, 64, 64, 128, 64],
        )],
    );
    let source = SourceStore::open(&f.0).unwrap();
    let ctx = MetalContext::new().unwrap();
    drop(
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .unwrap(),
    );
    let changed_shape = cache::Entry::new(
        &cache,
        vec![("dense.weight", source.tensor("dense.weight").unwrap())],
        16,
        vec![4],
        GgmlType::F32,
    )
    .unwrap();
    assert!(changed_shape.lookup(&ctx).unwrap().is_none());
    let path = manifests(&out).pop().unwrap();
    let original = std::fs::read(&path).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let body: serde_json::Value = serde_json::from_str(manifest["body"].as_str().unwrap()).unwrap();
    for bad_spec in ["old-converter-layout-v0", "wrong-output-shape"] {
        let mut bad = body.clone();
        bad["spec"] = bad_spec.into();
        let text = bad.to_string();
        std::fs::write(&path, serde_json::json!({"body":text,"checksum":blake3::hash(text.as_bytes()).to_hex().to_string()}).to_string()).unwrap();
        assert!(
            resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
                .is_err()
        );
    }
    std::fs::write(&path, b"{partial").unwrap();
    assert!(
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .is_err()
    );
    std::fs::write(&path, original).unwrap();
    let buffer = out.0.join(body["buffer"].as_str().unwrap());
    std::fs::OpenOptions::new()
        .write(true)
        .open(buffer)
        .unwrap()
        .set_len(4)
        .unwrap();
    assert!(
        resident::materialize_dense_impl(&ctx, &source, "dense.weight", &out.0, Some(&cache))
            .is_err()
    );
    assert_eq!(source.payload_reads.load(Ordering::Relaxed), 8);
}
