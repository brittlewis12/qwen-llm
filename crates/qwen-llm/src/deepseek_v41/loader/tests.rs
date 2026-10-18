use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

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
    let tensor = resident::materialize_bank(&ctx, &source, 0, "w1", 2, 32, 2).unwrap();
    assert_eq!(tensor.shape, [32, 2, 2]);
    assert_eq!(tensor.dtype, crate::tensor::GgmlType::MXFP4);
    assert_eq!(tensor.n_bytes(), 68);
    let bytes =
        unsafe { std::slice::from_raw_parts(tensor.buffer.contents().as_ptr().cast::<u8>(), 68) };
    assert_eq!(
        [bytes[0], bytes[17], bytes[34], bytes[51]],
        [127, 128, 129, 130]
    );
    assert_eq!(&bytes[1..5], &[0x11, 0x22, 0x11, 0x22]);
    assert_eq!(&bytes[35..39], &[0x33, 0x44, 0x33, 0x44]);
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
    assert_eq!(bytes.peak_staging, 6_266_880);
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
