//! CPU storage contracts plus ignored, tiny Metal loader-realization fixtures.
use super::*;
use crate::loader::{AttnBlock, GdnBlock};
use crate::model::ArchKind;

const DENSE_NATIVE_IQ_DTYPES: [GgmlType; 4] = [
    GgmlType::IQ2_XS,
    GgmlType::IQ2_XXS,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
];

fn desc(dtype: GgmlType, shape: &[u64]) -> TensorDesc {
    let (block, bytes) = dtype.storage_layout().unwrap_or((1, 1));
    TensorDesc {
        name: "same_source".into(),
        dtype,
        shape: shape.to_vec(),
        shard_idx: 0,
        data_offset: 32,
        n_bytes: shape.iter().product::<u64>() / block.max(1) * bytes,
    }
}

#[test]
fn dense_iq_role_storage_admits_only_dense_projections() {
    for raw in 0..=43 {
        let dtype = GgmlType::from_raw(raw);
        let tensor = desc(dtype, &[256, 3]);
        for role in [
            WeightRole::DenseProjection,
            WeightRole::MoeProjection,
            WeightRole::OutputHead,
            WeightRole::ExpertBank,
        ] {
            let legacy = weight_dtype_kept_native(dtype);
            let added =
                role == WeightRole::DenseProjection && DENSE_NATIVE_IQ_DTYPES.contains(&dtype);
            assert_eq!(weight_role_dtype_supported(role, dtype), legacy || added);
            for enabled in [false, true] {
                let actual = with_dense_iq_native_storage(enabled, || {
                    weight_storage_kind(role, &tensor).unwrap()
                });
                assert_eq!(
                    actual,
                    if legacy || (added && enabled) {
                        ModelWeightStorageKind::Direct
                    } else {
                        ModelWeightStorageKind::ConvertedF32
                    },
                    "{role:?}/{dtype:?}/{enabled}"
                );
            }
        }
    }
    assert_eq!(
        projection_weight_role(ArchKind::Dense),
        WeightRole::DenseProjection
    );
    assert_eq!(
        projection_weight_role(ArchKind::Moe),
        WeightRole::MoeProjection
    );
    assert!(!native_quant_embedding_supported(
        GgmlType::IQ2_XS,
        &[5120, 248320]
    ));
    assert!(!crate::workspace_lens::selected_readout_head_dtype_supported(GgmlType::IQ2_XS));
}

#[test]
fn dense_iq_role_storage_checks_native_geometry() {
    for k in [256, 5120, 6144, 17408] {
        for m in [1, 7, 8, 9] {
            assert_eq!(
                weight_storage_kind(
                    WeightRole::DenseProjection,
                    &desc(GgmlType::IQ2_XS, &[k, m])
                )
                .unwrap(),
                ModelWeightStorageKind::Direct
            );
        }
    }
    for shape in [
        vec![],
        vec![256],
        vec![256, 3, 1],
        vec![0, 3],
        vec![256, 0],
        vec![257, 3],
        vec![1u64 << 32, 1],
        vec![256, 1u64 << 32],
    ] {
        assert!(
            weight_storage_kind(WeightRole::DenseProjection, &desc(GgmlType::IQ2_XS, &shape))
                .is_err(),
            "{shape:?}"
        );
    }
    let mut bad = desc(GgmlType::IQ2_XS, &[5120, 3]);
    bad.n_bytes -= 1;
    assert!(weight_storage_kind(WeightRole::DenseProjection, &bad).is_err());
    bad = desc(GgmlType::IQ2_XS, &[5120, 3]);
    bad.data_offset = u64::MAX;
    assert!(weight_storage_kind(WeightRole::DenseProjection, &bad).is_err());
    bad.shape = vec![u64::MAX, 2];
    assert!(weight_storage_kind(WeightRole::DenseProjection, &bad).is_err());
    let bank = desc(GgmlType::IQ2_XS, &[256, 3, 2]);
    assert_eq!(
        weight_storage_kind(WeightRole::ExpertBank, &bank).unwrap(),
        ModelWeightStorageKind::ConvertedF32
    );
    // Weight row addresses are ulong, not uint. Do not impose an unrelated
    // 4 GiB bank cap: these descriptors require no allocation in this test.
    for shape in [[65_536, 262_144], [u32::MAX as u64 & !255, 1]] {
        let tensor = desc(GgmlType::IQ2_XS, &shape);
        assert_eq!(
            weight_storage_kind_with_policy(WeightRole::DenseProjection, &tensor, true).unwrap(),
            ModelWeightStorageKind::Direct
        );
    }
}

#[test]
fn dense_iq_role_storage_strict_codecs_require_i32_dimensions() {
    for dtype in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for shape in [[256, 3], [65_536, 262_144], [i32::MAX as u64 & !255, 1]] {
            let tensor = desc(dtype, &shape);
            assert_eq!(
                weight_storage_kind_with_policy(WeightRole::DenseProjection, &tensor, true)
                    .unwrap(),
                ModelWeightStorageKind::Direct
            );
        }
        for shape in [
            vec![256],
            vec![256, 3, 1],
            vec![0, 3],
            vec![256, 0],
            vec![257, 3],
            vec![1 << 31, 1],
            vec![256, 1 << 31],
        ] {
            let tensor = desc(dtype, &shape);
            assert!(
                weight_storage_kind_with_policy(WeightRole::DenseProjection, &tensor, true)
                    .is_err()
            );
        }
        let mut tensor = desc(dtype, &[256, 3]);
        tensor.n_bytes -= 1;
        assert!(
            weight_storage_kind_with_policy(WeightRole::DenseProjection, &tensor, true).is_err()
        );
    }
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        assert_eq!(
            native_quant_embedding_supported(dtype, &[5120, 248320]),
            dtype == GgmlType::IQ1_M
        );
        assert!(!crate::workspace_lens::selected_readout_head_dtype_supported(dtype));
        assert!(!weight_dtype_kept_native(dtype));
    }
}

#[test]
fn dense_iq_role_storage_scope_restores_and_is_thread_local() {
    let tensor = desc(GgmlType::IQ2_XS, &[256, 3]);
    let kind = || weight_storage_kind(WeightRole::DenseProjection, &tensor).unwrap();
    assert_eq!(kind(), ModelWeightStorageKind::Direct);
    with_dense_iq_native_storage(false, || {
        assert_eq!(kind(), ModelWeightStorageKind::ConvertedF32);
        with_dense_iq_native_storage(true, || assert_eq!(kind(), ModelWeightStorageKind::Direct));
        assert_eq!(kind(), ModelWeightStorageKind::ConvertedF32);
        assert!(
            std::panic::catch_unwind(|| {
                with_dense_iq_native_storage(true, || panic!("test restoration"));
            })
            .is_err()
        );
        assert_eq!(kind(), ModelWeightStorageKind::ConvertedF32);
        assert_eq!(
            std::thread::spawn(|| {
                weight_storage_kind(
                    WeightRole::DenseProjection,
                    &desc(GgmlType::IQ2_XS, &[256, 3]),
                )
                .unwrap()
            })
            .join()
            .unwrap(),
            ModelWeightStorageKind::Direct
        );
        // Storage rollback must not change execution of already-native tensors.
        assert!(crate::metal_dflash::prefill_mat_mat_dispatch_eligible(
            GgmlType::IQ2_XS
        ));
    });
    assert_eq!(kind(), ModelWeightStorageKind::Direct);
    assert!(
        std::panic::catch_unwind(|| {
            with_dense_iq_native_storage(false, || panic!("outer restoration"));
        })
        .is_err()
    );
    assert_eq!(kind(), ModelWeightStorageKind::Direct);
}

// This fixture tests storage-role assignment, not the architecture binder.
fn model<'a>(
    matrix: &'a TensorDesc,
    vector: &'a TensorDesc,
    bank: &'a TensorDesc,
    kind: ArchKind,
) -> Model<'a> {
    let moe = (kind == ArchKind::Moe).then_some(MoeFfn {
        gate_inp: vector,
        gate_exps: bank,
        up_exps: bank,
        down_exps: bank,
        gate_inp_shexp: vector,
    });
    Model {
        arch: crate::model::Arch {
            kind,
            ..crate::model::QWEN3_27B
        },
        token_embd: matrix,
        output_norm: vector,
        lm_head: matrix,
        tied_embeddings: true,
        mtp: None,
        blocks: vec![
            Block::Gdn(GdnBlock {
                attn_norm: vector,
                post_attention_norm: vector,
                ffn_gate: matrix,
                ffn_up: matrix,
                ffn_down: matrix,
                in_proj_qkv: matrix,
                in_proj_z: matrix,
                beta_proj: matrix,
                alpha_proj: matrix,
                a_log: vector,
                dt_bias: vector,
                conv1d: vector,
                norm: vector,
                out_proj: matrix,
                ffn_moe: moe.clone(),
            }),
            Block::Attn(AttnBlock {
                attn_norm: vector,
                post_attention_norm: vector,
                ffn_gate: matrix,
                ffn_up: matrix,
                ffn_down: matrix,
                q: matrix,
                k: matrix,
                v: matrix,
                o: matrix,
                q_norm: vector,
                k_norm: vector,
                ffn_moe: moe,
            }),
        ],
    }
}

#[test]
fn dense_iq_role_storage_complete_plans_preserve_excluded_roles() {
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let matrix = desc(dtype, &[256, 3]);
        let vector = desc(GgmlType::F32, &[256]);
        let bank = desc(dtype, &[256, 3, 2]);
        for kind in [ArchKind::Dense, ArchKind::Moe] {
            let model = model(&matrix, &vector, &bank, kind);
            let [old, native] =
                dense_iq_native_storage_comparison_plans(&model, false, false).unwrap();
            assert_eq!(old.len(), if kind == ArchKind::Dense { 28 } else { 38 });
            assert_eq!(old.len(), native.len());
            let mut changed = 0;
            for (a, b) in old.iter().zip(&native) {
                assert!(std::ptr::eq(a.desc, b.desc));
                if a.kind != b.kind {
                    changed += 1;
                    assert_eq!(a.kind, ModelWeightStorageKind::ConvertedF32);
                    assert_eq!(a.resident_bytes, 256 * 3 * 4);
                    assert_eq!(b.kind, ModelWeightStorageKind::Direct);
                    assert_eq!(b.resident_bytes, matrix.n_bytes);
                } else {
                    assert_eq!(a.resident_bytes, b.resident_bytes);
                }
            }
            assert_eq!(changed, if kind == ArchKind::Dense { 15 } else { 0 });
            // Tied embedding/head uses the same descriptor, but neither gains admission.
            assert_eq!(native[0].kind, ModelWeightStorageKind::ConvertedF32);
            assert_eq!(native[2].kind, ModelWeightStorageKind::ConvertedF32);
            let embedding_plan = model_weight_storage_requests(&model, true, false);
            if dtype == GgmlType::IQ1_M {
                let requests = embedding_plan.unwrap();
                assert_eq!(requests[0].kind, ModelWeightStorageKind::Direct);
                assert_eq!(requests[2].kind, ModelWeightStorageKind::ConvertedF32);
            } else {
                assert!(embedding_plan.is_err());
            }
            assert_eq!(
                model_weight_storage_inventory_digest(&old)
                    == model_weight_storage_inventory_digest(&native),
                kind == ArchKind::Moe
            );
        }
    }
}

#[test]
fn dense_iq_role_storage_ledger_rejects_different_policy_plan() {
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let tensor = desc(dtype, &[256, 3]);
        for native in [false, true] {
            with_dense_iq_native_storage(native, || {
                let mut plan = Vec::new();
                push_native_weight_request(&mut plan, WeightRole::DenseProjection, &tensor)
                    .unwrap();
                push_native_weight_request(&mut plan, WeightRole::OutputHead, &tensor).unwrap();
                let mut ledger = WeightLoadLedger::default();
                for role in [WeightRole::DenseProjection, WeightRole::OutputHead] {
                    let kind = weight_storage_kind(role, &tensor).unwrap();
                    ledger
                        .record_source(
                            &tensor,
                            if kind == ModelWeightStorageKind::Direct {
                                SourceMaterialization::DirectCopy
                            } else {
                                SourceMaterialization::ConvertedF32
                            },
                            if kind == ModelWeightStorageKind::Direct {
                                tensor.n_bytes
                            } else {
                                tensor.checked_n_elements().unwrap() * 4
                            },
                        )
                        .unwrap();
                }
                validate_model_weight_request_sequence(&ledger.requests, &plan).unwrap();
                assert_eq!(ledger.converted_descriptors, if native { 1 } else { 2 });
                assert_eq!(
                    ledger.converted_resident_bytes,
                    if native { 3072 } else { 6144 }
                );
                let opposite = with_dense_iq_native_storage(!native, || {
                    let mut requests = Vec::new();
                    push_native_weight_request(&mut requests, WeightRole::DenseProjection, &tensor)
                        .unwrap();
                    push_native_weight_request(&mut requests, WeightRole::OutputHead, &tensor)
                        .unwrap();
                    requests
                });
                assert!(
                    validate_model_weight_request_sequence(&ledger.requests, &opposite).is_err()
                );
            });
        }
    }
}

#[test]
fn dense_iq_role_storage_retained_plan_keeps_compressed_spans() {
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let tensor = desc(dtype, &[5120, 3]);
        let mut requests = Vec::new();
        push_native_weight_request(&mut requests, WeightRole::DenseProjection, &tensor).unwrap();
        push_native_weight_request(&mut requests, WeightRole::OutputHead, &tensor).unwrap();
        let direct = requests
            .iter()
            .filter(|r| r.kind == ModelWeightStorageKind::Direct)
            .map(|r| r.desc)
            .collect::<Vec<_>>();
        assert_eq!(direct.len(), 1);
        let retained =
            crate::metal::plan_retained_storage(&[16384], &direct, 16384, 16384, 32).unwrap();
        assert_eq!(retained.unique_view_bytes, tensor.n_bytes);
        assert_eq!(retained.unique_fallback_bytes, 0);
        assert_eq!(retained.entries.len(), 1);
        assert!(matches!(
            retained.entries[0].disposition,
            RetainedStorageDisposition::View { .. }
        ));
        assert_eq!(requests[1].resident_bytes, 5120 * 3 * 4);
    }
}

#[test]
fn dense_iq_role_storage_prefill_capability_matches_dense_roles() {
    for raw in 0..=43 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            crate::metal_dflash::prefill_mat_mat_dispatch_eligible(dtype),
            weight_role_dtype_supported(WeightRole::DenseProjection, dtype)
        );
    }
    assert!(crate::metal_dflash::prefill_mat_mat_dispatch_eligible(
        GgmlType::IQ2_XXS
    ));
}

#[test]
fn dense_iq_role_storage_resolved_choices_freeze_policy() {
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let matrix = desc(dtype, &[256, 3]);
        let vector = desc(GgmlType::F32, &[256]);
        let model = model(&matrix, &vector, &matrix, ArchKind::Dense);
        let [old, native] = dense_iq_native_storage_comparison_plans(&model, false, false).unwrap();
        for enabled in [false, true] {
            let choices = with_dense_iq_native_storage(enabled, || {
                ResolvedWeightLoadChoices::resolve(
                    NativeQuantEmbeddingSelection::AutoUnpromoted,
                    false,
                )
            });
            with_dense_iq_native_storage(!enabled, || {
                assert_eq!(choices.dense_iq_native, enabled);
                let frozen = choices.storage_requests(&model).unwrap();
                assert_eq!(
                    model_weight_storage_inventory_digest(&frozen),
                    model_weight_storage_inventory_digest(if enabled { &native } else { &old }),
                );
                assert_ne!(
                    model_weight_storage_inventory_digest(&frozen),
                    model_weight_storage_inventory_digest(
                        &model_weight_storage_requests(&model, false, false).unwrap()
                    ),
                );
                for role in [
                    WeightRole::DenseProjection,
                    WeightRole::MoeProjection,
                    WeightRole::OutputHead,
                    WeightRole::ExpertBank,
                ] {
                    assert_eq!(
                        weight_storage_kind_with_policy(role, &matrix, choices.dense_iq_native)
                            .unwrap(),
                        if enabled && role == WeightRole::DenseProjection {
                            ModelWeightStorageKind::Direct
                        } else {
                            ModelWeightStorageKind::ConvertedF32
                        },
                    );
                }
            });
        }
    }
}

#[test]
fn iq2_xs_addressing_checks_query_products_without_weight_bank_cap() {
    use crate::metal::validate_iq2_xs_mat_mat_addressing as check;
    check(5120, 17408, 4096).unwrap();
    check(65_536, 262_144, 1).expect("ulong compressed bank larger than 4 GiB");
    check(256, 65_536, 65_536).expect("last output index is exactly u32::MAX");
    assert!(check(256, 65_536, 65_537).is_err());
    check(256, 1, 1 << 24).expect("last input row start fits uint");
    assert!(check(256, 1, (1 << 24) + 1).is_err());
    // Pointer additions after q*K are ulong. The last row may extend beyond
    // UINT_MAX elements provided its uint row-start product still fits.
    check(768, 1, 5_592_406).unwrap();
    assert!(check(768, 1, 5_592_407).is_err());
    for (k, m, n) in [
        (0, 1, 1),
        (256, 0, 1),
        (256, 1, 0),
        (257, 1, 1),
        (usize::MAX, 1, 1),
        (256, usize::MAX, 1),
        (256, 1, usize::MAX),
    ] {
        assert!(check(k, m, n).is_err(), "K={k} M={m} N={n}");
    }
}

#[test]
#[ignore = "requires Metal lease; invalid encodes only, no command submitted"]
fn iq2_xs_direct_encoders_reject_invalid_physical_bindings() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    let weight =
        MetalTensor::from_bytes(&ctx, &[0; 3 * 74], vec![256, 3], GgmlType::IQ2_XS).unwrap();
    let x = MetalTensor::zeros_f32(&ctx, vec![256]).unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![3]).unwrap();
    let reject = |w: &MetalTensor, x: &MetalTensor, y: &MetalTensor| {
        for matrix in [false, true] {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            let result = if matrix {
                crate::metal::encode_mat_mat_iq2_xs_f32(&ctx, &enc, w, x, y, 256, 3, 1)
            } else {
                crate::metal::encode_mat_vec_iq2_xs_f32(&ctx, &enc, w, x, y, 256, 3)
            };
            enc.end();
            assert!(matches!(result, Err(MetalError::BadShape { .. })));
            // Drop, never commit: a rejected binding must not dispatch work.
        }
    };
    let mut invalid = weight.clone();
    invalid.shape = vec![3, 256];
    reject(&invalid, &x, &y);
    invalid = weight.clone();
    invalid.dtype = GgmlType::IQ2_XXS;
    reject(&invalid, &x, &y);
    for offset in [1, 2, u64::MAX - 1] {
        invalid = weight.clone();
        invalid.offset = offset;
        reject(&invalid, &x, &y);
    }
    let mut short_input = x.clone();
    short_input.offset = 4;
    reject(&weight, &short_input, &y);
    let mut short_output = y.clone();
    short_output.offset = 4;
    reject(&weight, &x, &short_output);
    let mut readonly = y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    reject(&weight, &x, &readonly);
    let alias = x.view_subrange(0, vec![3]);
    reject(&weight, &x, &alias);
    // N overflow must be rejected by direct GEMM, even with tiny real buffers.
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    let error =
        crate::metal::encode_mat_mat_iq2_xs_f32(&ctx, &enc, &weight, &x, &y, 256, 3, (1 << 24) + 1)
            .unwrap_err();
    enc.end();
    assert!(error.to_string().contains("input row start"));
}

// Same minimal GGUF v3 encoding used by gguf.rs quant_fixture (that helper is
// private to its test module). At most 1,468 payload bytes; no real model required.
struct TinyDenseIqFixture(std::path::PathBuf);

impl TinyDenseIqFixture {
    fn new(dtype: GgmlType) -> Self {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let (tag, block_bytes) = match dtype {
            GgmlType::IQ2_XS => (17, 74),
            GgmlType::IQ2_XXS => (16, 66),
            GgmlType::IQ1_S => (19, 50),
            GgmlType::IQ1_M => (29, 56),
            _ => panic!("unsupported fixture dtype"),
        };
        let mut quant = vec![0u8; 3 * block_bytes];
        for (row, block) in quant.chunks_exact_mut(block_bytes).enumerate() {
            block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // finite, nonzero d
            for (i, byte) in block[2..].iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(17).wrapping_add(row as u8);
            }
        }
        if dtype == GgmlType::IQ1_M {
            // IQ1_M stores the half scale across the high nibbles of four
            // scale words, not in the first two bytes. Pin d=1.0, retain
            // deterministic low scale bits and quant grid/sign metadata.
            for block in quant.chunks_exact_mut(56) {
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let low = u16::from_le_bytes([block[at], block[at + 1]]) & 0x0fff;
                    let bits = low | (((0x3c00u16 >> (4 * i)) & 0xf) << 12);
                    block[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
        let vector = vec![0u8; 256 * 4];
        let specs: [(&str, u32, &[u64], &[u8]); 3] = [
            ("matrix", tag, &[256, 3], &quant),
            ("bank", tag, &[256, 1, 3], &quant),
            ("vector", 0, &[256], &vector),
        ];
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let mut payload = Vec::new();
        for (name, dtype, shape, data) in specs {
            payload.resize(payload.len().next_multiple_of(32), 0);
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&dtype.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            payload.extend_from_slice(data);
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend_from_slice(&payload);
        let path = std::env::temp_dir().join(format!(
            "qwen-dense-iq-role-{}-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let fixture = Self(path);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&fixture.0)
            .unwrap();
        file.write_all(&bytes).unwrap();
        fixture
    }
}

impl Drop for TinyDenseIqFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn assert_materialized(tensor: &MetalTensor, gguf: &GgufFile, desc: &TensorDesc, native: bool) {
    assert_eq!(tensor.shape, desc.shape);
    assert_eq!(
        tensor.dtype,
        if native { desc.dtype } else { GgmlType::F32 }
    );
    let expected = if native {
        gguf.try_slice(desc).unwrap().to_vec()
    } else {
        let values = crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| *v != 0.0));
        bytemuck::cast_slice(&values).to_vec()
    };
    assert_eq!(tensor.n_bytes(), expected.len() as u64);
    assert!(tensor.offset as usize + expected.len() <= tensor.buffer.length());
    // These are synchronous CPU-populated shared buffers; no command submitted.
    let actual = unsafe {
        std::slice::from_raw_parts(
            (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize),
            expected.len(),
        )
    };
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "requires Metal lease; tiny loader allocations, no GPU dispatch"]
fn dense_iq_role_storage_private_loader_materializes_and_finishes() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let fixture = TinyDenseIqFixture::new(dtype);
        let gguf = GgufFile::open(&fixture.0).unwrap();
        let matrix = gguf.tensors.iter().find(|t| t.name == "matrix").unwrap();
        let bank = gguf.tensors.iter().find(|t| t.name == "bank").unwrap();
        for native in [false, true] {
            let choices = with_dense_iq_native_storage(native, || {
                ResolvedWeightLoadChoices::resolve(
                    NativeQuantEmbeddingSelection::AutoUnpromoted,
                    false,
                )
            });
            with_dense_iq_native_storage(!native, || {
                let mut loader =
                    MetalWeightLoader::new(&ctx, &gguf, DirectStorage::Copied, choices);
                let mut expected = Vec::new();
                for (role, desc) in [
                    (WeightRole::DenseProjection, matrix),
                    (WeightRole::MoeProjection, matrix),
                    (WeightRole::OutputHead, matrix),
                    (WeightRole::ExpertBank, bank),
                ] {
                    push_native_weight_request_with_policy(
                        &mut expected,
                        role,
                        desc,
                        choices.dense_iq_native,
                    )
                    .unwrap();
                    let tensor = if role == WeightRole::ExpertBank {
                        loader.load_moe_expert(desc).unwrap()
                    } else {
                        loader.load_weight(role, desc).unwrap()
                    };
                    assert_materialized(
                        &tensor,
                        &gguf,
                        desc,
                        native && role == WeightRole::DenseProjection,
                    );
                }
                push_model_weight_request(
                    &mut expected,
                    matrix,
                    ModelWeightStorageKind::ConvertedF32,
                )
                .unwrap();
                assert_materialized(
                    &loader.load_embedding(matrix, false).unwrap(),
                    &gguf,
                    matrix,
                    false,
                );
                let converted = if native { 4 } else { 5 };
                assert_eq!(loader.ledger.direct_copy_descriptors, usize::from(native));
                assert_eq!(loader.ledger.converted_descriptors, converted);
                assert_eq!(
                    loader.ledger.converted_resident_bytes,
                    converted as u64 * 256 * 3 * 4
                );
                loader
                    .finish(false, &expected)
                    .expect("real allocation ledger matches frozen plan");
            });
        }
        // Checked admission rejects malformed native geometry before recording any
        // materialization, even if the ambient scope would have selected F32.
        let choices = with_dense_iq_native_storage(true, || {
            ResolvedWeightLoadChoices::resolve(NativeQuantEmbeddingSelection::AutoUnpromoted, false)
        });
        with_dense_iq_native_storage(false, || {
            let mut loader = MetalWeightLoader::new(&ctx, &gguf, DirectStorage::Copied, choices);
            let mut invalid = matrix.clone();
            invalid.shape[0] = 257;
            assert!(
                loader
                    .load_weight(WeightRole::DenseProjection, &invalid)
                    .is_err()
            );
            assert_eq!(loader.ledger.source_descriptors, 0);
            loader.finish(false, &[]).unwrap();
            // The final ledger check still rejects an actual allocation paired
            // with the wrong policy; it is defense in depth, not policy resolution.
            let mut loader = MetalWeightLoader::new(&ctx, &gguf, DirectStorage::Copied, choices);
            let _tensor = loader
                .load_weight(WeightRole::DenseProjection, matrix)
                .unwrap();
            let mut wrong = Vec::new();
            push_native_weight_request_with_policy(
                &mut wrong,
                WeightRole::DenseProjection,
                matrix,
                false,
            )
            .unwrap();
            assert!(loader.finish(false, &wrong).is_err());
        });
    }
}

#[test]
#[ignore = "requires Metal lease; tiny prepared-model allocations, no GPU dispatch"]
fn dense_iq_role_storage_prepared_load_honors_policy_across_scopes() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    for dtype in DENSE_NATIVE_IQ_DTYPES {
        let fixture = TinyDenseIqFixture::new(dtype);
        let gguf = GgufFile::open(&fixture.0).unwrap();
        let matrix = gguf.tensors.iter().find(|t| t.name == "matrix").unwrap();
        let bank = gguf.tensors.iter().find(|t| t.name == "bank").unwrap();
        let vector = gguf.tensors.iter().find(|t| t.name == "vector").unwrap();
        let model = model(matrix, vector, bank, ArchKind::Dense);
        for native in [false, true] {
            let prepared = with_dense_iq_native_storage(native, || {
                MetalModel::prepare_load_with_options(
                    &ctx,
                    &gguf,
                    &model,
                    MetalModelLoadOptions::default(),
                )
                .unwrap()
            });
            assert_eq!(prepared.choices.dense_iq_native, native);
            assert_eq!(
                prepared
                    .expected
                    .iter()
                    .filter(|r| r.desc.dtype == dtype && r.kind == ModelWeightStorageKind::Direct)
                    .count(),
                if native { 15 } else { 0 }
            );
            let loaded = with_dense_iq_native_storage(!native, || {
                MetalModel::load_prepared(prepared)
                    .expect("prepared choice must survive opposite ambient scope")
            });
            assert_materialized(&loaded.token_embd, &gguf, matrix, false);
            assert_materialized(&loaded.lm_head, &gguf, matrix, false);
            for block in &loaded.blocks {
                let projections: &[&MetalTensor] = match block {
                    MetalBlock::Gdn(g) => &[
                        &g.ffn_gate,
                        &g.ffn_up,
                        &g.ffn_down,
                        &g.in_proj_qkv,
                        &g.in_proj_z,
                        &g.beta_proj,
                        &g.alpha_proj,
                        &g.out_proj,
                    ],
                    MetalBlock::Attn(a) => {
                        &[&a.q, &a.k, &a.v, &a.o, &a.ffn_gate, &a.ffn_up, &a.ffn_down]
                    }
                };
                for projection in projections {
                    assert_materialized(projection, &gguf, matrix, native);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Metal lease; view validation only, no command submitted"]
fn native_iq_dispatch_views_check_prefix_and_physical_source() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    let x = MetalTensor::zeros_f32(&ctx, vec![1024]).unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![32]).unwrap();
    for matrix in [false, true] {
        let (xv, yv) = native_iq_dispatch_views(&x, &y, 256, 3, 1, matrix).unwrap();
        assert_eq!(xv.shape, if matrix { vec![256, 1] } else { vec![256] });
        assert_eq!(yv.shape, if matrix { vec![3, 1] } else { vec![3] });
        assert_eq!(Retained::as_ptr(&xv.buffer), Retained::as_ptr(&x.buffer));
        assert_eq!(Retained::as_ptr(&yv.buffer), Retained::as_ptr(&y.buffer));
        assert_eq!(xv.offset, x.offset);
        assert_eq!(yv.offset, y.offset);
    }
    let offset_x = x.view_subrange(4, vec![768]);
    let (xv, _) = native_iq_dispatch_views(&offset_x, &y, 256, 3, 3, true).unwrap();
    assert_eq!(xv.offset, 16);
    assert_eq!(xv.shape, [256, 3]);
    let too_short = x.view_subrange(0, vec![255]);
    assert!(native_iq_dispatch_views(&too_short, &y, 256, 3, 1, true).is_err());
    let short_y = y.view_subrange(0, vec![2]);
    assert!(native_iq_dispatch_views(&x, &short_y, 256, 3, 1, true).is_err());
    for offset in [1, 4, u64::MAX - 3] {
        let mut invalid = x.clone();
        invalid.offset = offset;
        assert!(native_iq_dispatch_views(&invalid, &y, 256, 3, 1, true).is_err());
    }
    let mut invalid = x.clone();
    invalid.dtype = GgmlType::F16;
    assert!(native_iq_dispatch_views(&invalid, &y, 256, 3, 1, true).is_err());
    invalid = x.clone();
    invalid.shape = vec![u64::MAX, 2];
    assert!(native_iq_dispatch_views(&invalid, &y, 256, 3, 1, true).is_err());
    let mut readonly = y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(native_iq_dispatch_views(&x, &readonly, 256, 3, 1, true).is_err());
    for (k, m, n) in [
        (0, 3, 1),
        (257, 3, 1),
        (1 << 31, 3, 1),
        (256, 1 << 31, 1),
        (256, 3, 1 << 31),
    ] {
        assert!(native_iq_dispatch_views(&x, &y, k, m, n, true).is_err());
    }
    // Activations may be normalized; a flattened weight must still fail the
    // primitive's [K,M] contract rather than being silently reinterpreted.
    let flat_weight =
        MetalTensor::from_bytes(&ctx, &[0; 3 * 66], vec![768], GgmlType::IQ2_XXS).unwrap();
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    assert!(encode_mat_vec_dispatch(&ctx, &enc, &flat_weight, &x, &y, 256, 3).is_err());
    assert!(
        encode_mat_mat_dispatch_with_policy(&ctx, &enc, &flat_weight, &x, &y, 256, 3, 1, false)
            .is_err()
    );
    enc.end();
}

#[test]
#[ignore = "requires Metal lease; tiny IQ1/IQ2_XXS packed dispatch integration"]
fn native_iq_dispatch_flat_packed_n1_policy_and_concurrent() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    for dtype in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::IQ1_M] {
        let fixture = TinyDenseIqFixture::new(dtype);
        let gguf = GgufFile::open(&fixture.0).unwrap();
        let desc = gguf.tensors.iter().find(|t| t.name == "matrix").unwrap();
        let weight =
            MetalTensor::from_gguf_tensor(&ctx, desc, gguf.try_slice(desc).unwrap()).unwrap();
        let decoded = crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap();
        let (k, m) = (256usize, 3usize);
        const SENTINEL: f32 = 12345.0;
        for n in [1usize, 3, 17] {
            for flat in [false, true] {
                for concurrent in [false, true] {
                    for allow_n1 in [false, true] {
                        if n != 1 && allow_n1 {
                            continue;
                        }
                        // Offset, overprovisioned source views exercise the real
                        // packed scratch contract; adapter must touch only prefixes.
                        let values: Vec<f32> = (0..k * (n + 1) + 4)
                            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.002)
                            .collect();
                        let input = MetalTensor::from_bytes(
                            &ctx,
                            bytemuck::cast_slice(&values),
                            vec![values.len() as u64],
                            GgmlType::F32,
                        )
                        .unwrap();
                        let x = input.view_subrange(
                            4,
                            if flat {
                                vec![(k * (n + 1)) as u64]
                            } else {
                                vec![k as u64, (n + 1) as u64]
                            },
                        );
                        let initial = vec![SENTINEL; 2 * m * n + 7];
                        let output = MetalTensor::from_bytes(
                            &ctx,
                            bytemuck::cast_slice(&initial),
                            vec![initial.len() as u64],
                            GgmlType::F32,
                        )
                        .unwrap();
                        // Both source views overlap in spare capacity, but the
                        // actual write prefixes are disjoint. Hazard notes must
                        // use normalized views and occur exactly once per call.
                        let y0 = output.view_subrange(
                            4,
                            if flat {
                                vec![(2 * m * n + 3) as u64]
                            } else {
                                vec![m as u64, (2 * n + 1) as u64]
                            },
                        );
                        let y1 = output.view_subrange(
                            (4 + m * n) as u64,
                            if flat {
                                vec![(m * n + 3) as u64]
                            } else {
                                vec![m as u64, (n + 1) as u64]
                            },
                        );
                        let cmd = ctx.queue.commandBuffer().unwrap();
                        let enc = if concurrent {
                            KernelEncoder::begin_concurrent(&cmd)
                        } else {
                            KernelEncoder::begin(&cmd)
                        };
                        encode_mat_mat_dispatch_with_policy(
                            &ctx, &enc, &weight, &x, &y0, k, m, n, allow_n1,
                        )
                        .unwrap();
                        if n == 1 {
                            encode_mat_vec_dispatch(&ctx, &enc, &weight, &x, &y1, k, m).unwrap();
                        } else {
                            encode_mat_mat_dispatch_with_policy(
                                &ctx, &enc, &weight, &x, &y1, k, m, n, allow_n1,
                            )
                            .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd).unwrap();
                        let actual = unsafe {
                            std::slice::from_raw_parts(
                                output.buffer.contents().as_ptr().cast::<f32>(),
                                initial.len(),
                            )
                        };
                        assert!(
                            actual[..4]
                                .iter()
                                .chain(&actual[4 + 2 * m * n..])
                                .all(|v| *v == SENTINEL)
                        );
                        let mut max_error = 0.0f32;
                        let mut max_ref = 0.0f32;
                        for q in 0..n {
                            for row in 0..m {
                                let expected: f32 = (0..k)
                                    .map(|i| decoded[row * k + i] * values[4 + q * k + i])
                                    .sum();
                                max_ref = max_ref.max(expected.abs());
                                for base in [4, 4 + m * n] {
                                    let observed = actual[base + q * m + row];
                                    assert!(observed.is_finite());
                                    max_error = max_error.max((observed - expected).abs());
                                }
                            }
                        }
                        eprintln!(
                            "{dtype:?} dispatch N={n} flat={flat} concurrent={concurrent} allow_n1={allow_n1} max_abs={max_error} max_ref={max_ref}"
                        );
                        assert!(max_error <= 1e-4 * (1.0 + max_ref));
                        assert_eq!(weight.shape, [256, 3]);
                    }
                }
            }
        }
    }
}

#[test]
fn iq1_m_embedding_policy_is_role_and_fingerprint_scoped() {
    let arch = crate::model::QWEN3_27B;
    let shape = [5120, 248_320];
    assert!(native_quant_embedding_supported(GgmlType::IQ1_M, &shape));
    assert!(native_quant_embedding_default_promoted(
        &arch,
        false,
        GgmlType::IQ1_M,
        &shape
    ));
    assert!(!native_quant_embedding_default_promoted(
        &arch,
        true,
        GgmlType::IQ1_M,
        &shape
    ));
    assert!(!native_quant_embedding_default_promoted(
        &arch,
        false,
        GgmlType::IQ1_S,
        &shape
    ));
    for changed in [
        crate::model::Arch {
            kind: ArchKind::Moe,
            ..arch
        },
        crate::model::Arch {
            n_layer: 63,
            ..arch
        },
        crate::model::Arch {
            hidden_size: 4096,
            ..arch
        },
    ] {
        assert!(!native_quant_embedding_default_promoted(
            &changed,
            false,
            GgmlType::IQ1_M,
            &shape
        ));
    }
    for shape in [
        vec![256],
        vec![256, 3, 1],
        vec![0, 3],
        vec![256, 0],
        vec![257, 3],
        vec![1 << 31, 3],
        vec![256, 1 << 31],
    ] {
        assert!(!native_quant_embedding_supported(GgmlType::IQ1_M, &shape));
    }
    assert!(native_quant_embedding_supported(GgmlType::IQ1_M, &[256, 3]));
    assert!(!native_quant_embedding_default_promoted(
        &arch,
        false,
        GgmlType::IQ1_M,
        &[256, 3]
    ));
    use NativeQuantEmbeddingMode::{Auto, Disabled, Forced, Invalid};
    for (mode, tied, expected_native) in [
        (Auto, false, true),
        (Auto, true, false),
        (Disabled, false, false),
        (Forced, true, true),
        (Invalid, false, false),
    ] {
        let promoted =
            native_quant_embedding_default_promoted(&arch, tied, GgmlType::IQ1_M, &shape);
        let selection = resolve_native_quant_embedding(mode, true, promoted);
        let matrix = desc(GgmlType::IQ1_M, &shape);
        let vector = desc(GgmlType::F32, &[5120]);
        let mut model = model(&matrix, &vector, &matrix, ArchKind::Dense);
        model.tied_embeddings = tied;
        for dense_iq_native in [false, true] {
            let choices = with_dense_iq_native_storage(dense_iq_native, || {
                ResolvedWeightLoadChoices::resolve(selection, false)
            });
            let requests = with_dense_iq_native_storage(!dense_iq_native, || {
                choices.storage_requests(&model).unwrap()
            });
            assert_eq!(choices.embedding_selection.uses_native(), expected_native);
            assert_eq!(
                requests[0].kind,
                if expected_native {
                    ModelWeightStorageKind::Direct
                } else {
                    ModelWeightStorageKind::ConvertedF32
                }
            );
            assert_eq!(
                requests[0].resident_bytes,
                if expected_native {
                    matrix.n_bytes
                } else {
                    5120 * 248_320 * 4
                }
            );
            assert_eq!(requests[2].kind, ModelWeightStorageKind::ConvertedF32); // output head, even tied
        }
    }
    let mut bad_embedding = desc(GgmlType::IQ1_M, &[256, 3]);
    bad_embedding.n_bytes -= 1;
    let vector = desc(GgmlType::F32, &[256]);
    let malformed = model(&bad_embedding, &vector, &bad_embedding, ArchKind::Dense);
    // Disable native projections: this rejection must come from the selected
    // embedding contract, not from a later projection sharing its descriptor.
    assert!(model_weight_storage_requests_with_policy(&malformed, true, false, false).is_err());
}

#[test]
#[ignore = "requires Metal lease; tiny IQ1_M load and embedding gather"]
fn iq1_m_embedding_loader_and_gather_use_frozen_choice() {
    let ctx = MetalContext::new().expect("owner must hold normal Metal lease");
    let fixture = TinyDenseIqFixture::new(GgmlType::IQ1_M);
    let gguf = GgufFile::open(&fixture.0).unwrap();
    let desc = gguf.tensors.iter().find(|t| t.name == "matrix").unwrap();
    let decoded = crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap();
    let choices = ResolvedWeightLoadChoices::resolve(NativeQuantEmbeddingSelection::Forced, false);
    let mut loader = MetalWeightLoader::new(&ctx, &gguf, DirectStorage::Copied, choices);
    let mut malformed = desc.clone();
    malformed.n_bytes -= 1;
    assert!(loader.load_embedding(&malformed, true).is_err());
    assert_eq!(loader.ledger.source_descriptors, 0);
    loader.finish(false, &[]).unwrap();
    use NativeQuantEmbeddingMode::{Auto, Disabled, Forced, Invalid};
    for (mode, promoted) in [
        (Auto, true),
        (Auto, false),
        (Forced, false),
        (Disabled, true),
        (Invalid, true),
    ] {
        let choices = with_dense_iq_native_storage(false, || {
            ResolvedWeightLoadChoices::resolve(
                resolve_native_quant_embedding(mode, true, promoted),
                false,
            )
        });
        let native = choices.embedding_selection.uses_native();
        let embedding = with_dense_iq_native_storage(true, || {
            let mut loader = MetalWeightLoader::new(&ctx, &gguf, DirectStorage::Copied, choices);
            let embedding = loader
                .load_embedding(desc, choices.embedding_selection.uses_native())
                .unwrap();
            assert_materialized(&embedding, &gguf, desc, native);
            let head = loader.load_weight(WeightRole::OutputHead, desc).unwrap();
            assert_materialized(&head, &gguf, desc, false);
            let mut expected = Vec::new();
            push_model_weight_request(
                &mut expected,
                desc,
                if native {
                    ModelWeightStorageKind::Direct
                } else {
                    ModelWeightStorageKind::ConvertedF32
                },
            )
            .unwrap();
            push_native_weight_request_with_policy(
                &mut expected,
                WeightRole::OutputHead,
                desc,
                choices.dense_iq_native,
            )
            .unwrap();
            loader.finish(false, &expected).unwrap();
            embedding
        });
        for shaped in [false, true] {
            let ids_values = [2i32, 0, 2];
            let ids = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&ids_values),
                vec![3],
                GgmlType::I32,
            )
            .unwrap();
            let backing = MetalTensor::zeros_f32(&ctx, vec![772]).unwrap();
            let output = backing.view_subrange(4, if shaped { vec![256, 3] } else { vec![768] });
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            encode_get_rows_f32(&ctx, &enc, &embedding, &ids, &output, 3, 256).unwrap();
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    (output.buffer.contents().as_ptr() as *const u8)
                        .add(output.offset as usize)
                        .cast::<f32>(),
                    768,
                )
            };
            for (row, id) in ids_values.into_iter().enumerate() {
                for col in 0..256 {
                    let expected = decoded[id as usize * 256 + col];
                    assert!(actual[row * 256 + col].is_finite());
                    assert!(
                        (actual[row * 256 + col] - expected).abs() <= 1e-6 * (1.0 + expected.abs())
                    );
                }
            }
        }
    }
}
