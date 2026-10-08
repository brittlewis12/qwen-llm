use super::*;
use crate::tensor::GgmlType;
use serde_json::json;

const GIB: u64 = 1 << 30;

fn release_metadata(arch: &str) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("general.architecture", json!(arch));
    put("general.type", json!("model"));
    put("tokenizer.ggml.model", json!("gpt2"));
    put("tokenizer.ggml.pre", json!("glm4"));
    put(
        "tokenizer.ggml.tokens",
        Value::Array(vec![json!("t"); RELEASE_VOCAB_SIZE as usize]),
    );
    let kv: Vec<u64> = (0..46).map(|i| u64::from(i % 4 == 3 || i == 45)).collect();
    for (suffix, value) in [
        ("block_count", json!(46)),
        ("context_length", json!(1_048_576)),
        ("embedding_length", json!(4096)),
        ("feed_forward_length", json!(12288)),
        ("vocab_size", json!(RELEASE_VOCAB_SIZE)),
        ("attention.head_count", json!(64)),
        ("attention.head_count_kv", json!(kv)),
        ("attention.layer_norm_rms_epsilon", json!(1e-5_f32)),
        ("attention.layer_norm_epsilon", json!(1e-6_f32)),
        ("attention.q_lora_rank", json!(1536)),
        ("attention.kv_lora_rank", json!(512)),
        ("attention.key_length", json!(512)),
        ("attention.value_length", json!(512)),
        ("attention.key_length_mla", json!(256)),
        ("attention.value_length_mla", json!(256)),
        ("attention.indexer.head_count", json!(32)),
        ("attention.indexer.key_length", json!(128)),
        ("attention.indexer.top_k", json!(2048)),
        ("attention.indexer.kpool", json!(4)),
        ("attention.indexer.index_share_mtp", json!(true)),
        ("rope.dimension_count", json!(0)),
        ("expert_count", json!(288)),
        ("expert_used_count", json!(8)),
        ("expert_group_count", json!(1)),
        ("expert_group_used_count", json!(1)),
        ("expert_gating_func", json!(2)),
        ("expert_feed_forward_length", json!(2048)),
        ("expert_shared_feed_forward_length", json!(2048)),
        ("expert_shared_count", json!(1)),
        ("expert_weights_scale", json!(2.5)),
        ("expert_weights_norm", json!(true)),
        ("leading_dense_block_count", json!(3)),
        ("swiglu_clamp_exp", json!(vec![10.0; 46])),
        ("swiglu_clamp_shexp", json!(vec![10.0; 46])),
        ("ssm.conv_kernel", json!(4)),
        ("kda.head_dim", json!(128)),
        ("kda.gate_lower_bound", json!(-5.0)),
        ("hyper_connection.count", json!(4)),
        ("hyper_connection.sinkhorn_iterations", json!(20)),
        ("hyper_connection.epsilon", json!(1e-6_f32)),
        ("nextn_predict_layers", json!(1)),
    ] {
        put(&format!("{arch}.{suffix}"), value);
    }
    m
}

fn config() -> Glm5NextConfig {
    Glm5NextConfig::from_metadata(&release_metadata(ARCHITECTURE_NAME)).unwrap()
}

/// Release-shaped descriptors with the census dtypes; offsets are synthetic.
fn release_tensors() -> Vec<TensorDesc> {
    use GgmlType::*;
    let mut out = Vec::new();
    let mut offset = 0u64;
    let mut add = |name: String, shape: &[u64], dtype: GgmlType| {
        let (block, size) = dtype.storage_layout().unwrap();
        let n_bytes = shape.iter().product::<u64>() / block * size;
        out.push(TensorDesc {
            name,
            shape: shape.to_vec(),
            dtype,
            shard_idx: 0,
            data_offset: offset,
            n_bytes,
        });
        offset += n_bytes;
    };
    add("token_embd.weight".into(), &[4096, 154_880], Q6_K);
    add("output_norm.weight".into(), &[4096], F32);
    add("output.weight".into(), &[4096, 154_880], Q6_K);
    for i in 0..46u64 {
        let n = |s: &str| format!("blk.{i}.{s}");
        let nextn = i == 45;
        let mla = i % 4 == 3 || nextn;
        let attn_q8 = i == 11;
        add(n("attn_norm.weight"), &[4096], F32);
        add(n("ffn_norm.weight"), &[4096], F32);
        if !nextn {
            for site in ["attn", "ffn"] {
                add(n(&format!("hc_{site}_fn.weight")), &[16384, 24], Q8_0);
                add(n(&format!("hc_{site}_base.weight")), &[24], F32);
                add(n(&format!("hc_{site}_scale.weight")), &[3], F32);
            }
        }
        if mla {
            let wide = if attn_q8 { Q8_0 } else { Q6_K };
            add(n("attn_q_a.weight"), &[4096, 1536], wide);
            add(n("attn_q_a_norm.weight"), &[1536], F32);
            add(n("attn_q_b.weight"), &[1536, 16384], Q8_0);
            add(n("attn_kv_a_mqa.weight"), &[4096, 512], Q8_0);
            add(n("attn_kv_a_norm.weight"), &[512], F32);
            add(n("attn_k_b.weight"), &[256, 512, 64], Q8_0);
            add(n("attn_v_b.weight"), &[512, 256, 64], Q8_0);
            add(n("attn_output.weight"), &[16384, 4096], wide);
            add(n("indexer.attn_q_b.weight"), &[1536, 4096], Q8_0);
            add(n("indexer.attn_k.weight"), &[4096, 128], Q8_0);
            add(n("indexer.k_norm.weight"), &[128], F32);
            add(n("indexer.k_norm.bias"), &[128], F32);
            add(n("indexer.proj.weight"), &[4096, 32], F32);
            add(n("indexer_compressor_gate.weight"), &[4096, 128], Q8_0);
            add(n("indexer_compressor_ape.weight"), &[128, 4], F32);
        } else {
            for p in ["attn_q", "attn_k", "attn_v"] {
                add(n(&format!("{p}.weight")), &[4096, 8192], Q6_K);
            }
            for c in ["q", "k", "v"] {
                add(n(&format!("ssm_conv1d_{c}.weight")), &[4, 1, 8192], F32);
            }
            add(n("ssm_f_a.weight"), &[4096, 128], Q8_0);
            add(n("ssm_f_b.weight"), &[128, 8192], Q8_0);
            add(n("ssm_dt.bias"), &[8192], F32);
            add(n("ssm_a"), &[64], F32);
            add(n("ssm_beta.weight"), &[4096, 64], Q8_0);
            add(n("ssm_g_a.weight"), &[4096, 128], Q8_0);
            add(n("ssm_g_b.weight"), &[128, 8192], Q8_0);
            add(n("ssm_norm.weight"), &[128], F32);
            add(n("attn_output.weight"), &[8192, 4096], Q6_K);
        }
        if i < 3 {
            add(n("ffn_gate.weight"), &[4096, 12288], Q6_K);
            add(n("ffn_up.weight"), &[4096, 12288], Q6_K);
            add(n("ffn_down.weight"), &[12288, 4096], Q6_K);
        } else {
            let (gate_up, down) = match i {
                45 => (Q2_K, Q3_K),
                11 => (IQ3_S, IQ4_XS),
                12 | 44 => (IQ2_S, IQ4_XS),
                _ => (IQ2_S, IQ3_S),
            };
            let shared = if i == 11 { Q8_0 } else { Q6_K };
            add(n("ffn_gate_inp.weight"), &[4096, 288], F32);
            add(n("exp_probs_b.bias"), &[288], F32);
            add(n("ffn_gate_exps.weight"), &[4096, 2048, 288], gate_up);
            add(n("ffn_up_exps.weight"), &[4096, 2048, 288], gate_up);
            add(n("ffn_down_exps.weight"), &[2048, 4096, 288], down);
            add(n("ffn_gate_shexp.weight"), &[4096, 2048], shared);
            add(n("ffn_up_shexp.weight"), &[4096, 2048], shared);
            add(n("ffn_down_shexp.weight"), &[2048, 4096], shared);
        }
        if nextn {
            add(n("nextn.eh_proj.weight"), &[8192, 4096], Q8_0);
            for norm in ["enorm", "hnorm", "shared_head_norm"] {
                add(n(&format!("nextn.{norm}.weight")), &[4096], F32);
            }
        }
    }
    out
}

#[test]
fn release_metadata_binds_in_both_namespaces() {
    for arch in [ARCHITECTURE_NAME, LEGACY_ARCHITECTURE_NAME] {
        let c = Glm5NextConfig::from_metadata(&release_metadata(arch)).unwrap();
        assert_eq!(c.architecture, arch);
        assert_eq!(c.executed_block_count(), 45);
        assert_eq!(c.blocks.len(), 45);
        assert_eq!(c.block_count(MixerKind::Kda), 34);
        assert_eq!(c.block_count(MixerKind::Mla), 11);
        assert_eq!(
            c.blocks.iter().filter(|b| b.ffn == FfnKind::Dense).count(),
            3
        );
        assert_eq!(
            (c.kda_width(), c.mla_width(), c.hc_width(), c.hc_mix_count()),
            (8192, 16384, 16384, 24)
        );
        assert_eq!(c.selected_pool_count(), 512);
        assert_eq!(c.selection_width(), 2051);
        assert_eq!(c.sparse_frontier(), 2052);
    }
}

#[test]
fn metadata_fails_closed() {
    let mutate = |f: &dyn Fn(&mut BTreeMap<String, Value>)| {
        let mut m = release_metadata(ARCHITECTURE_NAME);
        f(&mut m);
        Glm5NextConfig::from_metadata(&m).unwrap_err().to_string()
    };
    type Mutation = Box<dyn Fn(&mut BTreeMap<String, Value>)>;
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "unrecognized",
            Box::new(|m| {
                m.insert("glm5-next.rope.freq_base".into(), json!(10000.0));
            }),
        ),
        (
            "no RoPE",
            Box::new(|m| {
                m.insert("glm5-next.rope.dimension_count".into(), json!(64));
            }),
        ),
        (
            "sigmoid",
            Box::new(|m| {
                m.insert("glm5-next.expert_gating_func".into(), json!(1));
            }),
        ),
        (
            "release expects",
            Box::new(|m| {
                let mut kv: Vec<u64> = (0..46).map(|i| u64::from(i % 4 == 3)).collect();
                kv[4] = 1;
                m.insert("glm5-next.attention.head_count_kv".into(), json!(kv));
            }),
        ),
        (
            "per-layer values differ",
            Box::new(|m| {
                let mut clamp = vec![10.0; 46];
                clamp[7] = 7.0;
                m.insert("glm5-next.swiglu_clamp_exp".into(), json!(clamp));
            }),
        ),
        (
            "adapter",
            Box::new(|m| {
                m.insert("general.type".into(), json!("adapter"));
            }),
        ),
        (
            "expected glm4",
            Box::new(|m| {
                m.insert("tokenizer.ggml.pre".into(), json!("chatglm-bpe"));
            }),
        ),
        (
            "release expects",
            Box::new(|m| {
                m.insert("glm5-next.expert_weights_scale".into(), json!(1.5));
            }),
        ),
        (
            "general.architecture",
            Box::new(|m| {
                m.insert("general.architecture".into(), json!("glm4moe"));
            }),
        ),
        (
            "without the incomplete pool's tail",
            Box::new(|m| {
                m.insert(
                    "glm5-next.attention.indexer.kpool_select_tail".into(),
                    json!(false),
                );
            }),
        ),
        (
            "expected bool",
            Box::new(|m| {
                m.insert(
                    "glm5-next.attention.indexer.kpool_select_tail".into(),
                    json!(1),
                );
            }),
        ),
        (
            "shared indexer layers",
            Box::new(|m| {
                let mut types = vec![1u64; 46];
                types[7] = 0;
                m.insert("glm5-next.attention.indexer.types".into(), json!(types));
            }),
        ),
        (
            "shared indexer layers",
            Box::new(|m| {
                m.insert("glm5-next.attention.indexer.types".into(), json!(0));
            }),
        ),
        (
            "expected 46 (stored) or 45 (executed)",
            Box::new(|m| {
                m.insert(
                    "glm5-next.attention.indexer.types".into(),
                    json!(vec![1u64; 11]),
                );
            }),
        ),
    ];
    for (needle, f) in cases {
        let err = mutate(&*f);
        assert!(err.contains(needle), "{needle:?} not in {err:?}");
    }
    // Explicitly stated defaults are the implemented semantics.
    for (key, value) in [
        ("glm5-next.attention.indexer.kpool_select_tail", json!(true)),
        ("glm5-next.attention.indexer.types", json!(1)),
        ("glm5-next.attention.indexer.types", json!(vec![1u64; 46])),
        ("glm5-next.attention.indexer.types", json!(vec![1u64; 45])),
    ] {
        let mut m = release_metadata(ARCHITECTURE_NAME);
        m.insert(key.into(), value.clone());
        Glm5NextConfig::from_metadata(&m)
            .unwrap_or_else(|e| panic!("{key} = {value} refused: {e}"));
    }
    // A key from the other spelling's namespace is not interpreted.
    let mut m = release_metadata(LEGACY_ARCHITECTURE_NAME);
    m.remove("glm5next.expert_count");
    m.insert("glm5-next.expert_count".into(), json!(288));
    assert!(matches!(
        Glm5NextConfig::from_metadata(&m),
        Err(Glm5NextError::MissingMetadata(key)) if key == "glm5next.expert_count"
    ));
}

#[test]
fn binder_classifies_every_tensor_exactly_once() {
    let tensors = release_tensors();
    assert_eq!(tensors.len(), RELEASE_TENSOR_COUNT);
    let model = Glm5NextModel::bind(config(), &tensors).unwrap();
    assert_eq!(model.trunk.len(), 1383);
    assert_eq!(model.nextn.len(), 29);
    assert_eq!(model.blocks.len(), 45);
    assert!(model.trunk.iter().all(|b| b.role != TensorRole::NextN));
    assert!(model.nextn.iter().all(|t| t.name.starts_with("blk.45.")));
    let total: u64 = tensors.iter().map(|t| t.n_bytes).sum();
    assert_eq!(model.trunk_bytes + model.nextn_bytes, total);
    assert_eq!(model.retained_tensors().len(), 1383);
    assert!(matches!(model.blocks[3].mixer, MixerTensors::Mla(_)));
    assert!(matches!(model.blocks[4].mixer, MixerTensors::Kda(_)));
    assert!(matches!(model.blocks[2].ffn, FfnTensors::Dense(_)));
    assert!(matches!(model.blocks[3].ffn, FfnTensors::Moe(_)));
}

#[test]
fn binder_refuses_structural_drift() {
    let refuse = |f: &dyn Fn(&mut Vec<TensorDesc>)| {
        let mut tensors = release_tensors();
        f(&mut tensors);
        Glm5NextModel::bind(config(), &tensors)
            .unwrap_err()
            .to_string()
    };
    let find = |t: &mut Vec<TensorDesc>, name: &str| t.iter().position(|d| d.name == name).unwrap();
    assert!(
        refuse(&|t| {
            let i = find(t, "blk.7.attn_k_b.weight");
            t.remove(i);
        })
        .contains("missing required tensor")
    );
    assert!(
        refuse(&|t| {
            let mut extra = t[0].clone();
            extra.name = "blk.3.attn_sinks.weight".into();
            t.push(extra);
        })
        .contains("unexpected tensor")
    );
    assert!(
        refuse(&|t| {
            let dup = t[1].clone();
            t.push(dup);
        })
        .contains("duplicate")
    );
    assert!(
        refuse(&|t| {
            let i = find(t, "blk.4.ssm_a");
            t[i].shape = vec![128];
            t[i].n_bytes = 512;
        })
        .contains("expected shape")
    );
    assert!(
        refuse(&|t| {
            let i = find(t, "blk.4.ffn_down_exps.weight");
            t[i].n_bytes += 1;
        })
        .contains("inconsistent")
    );
    // A trunk expert bank stored as Q2_K has no verified path; NextN may be.
    assert!(
        refuse(&|t| {
            let i = find(t, "blk.20.ffn_gate_exps.weight");
            t[i].dtype = GgmlType::Q2_K;
            t[i].n_bytes = 4096 * 2048 * 288 / 256 * 84;
        })
        .contains("no executable path")
    );
}

#[test]
fn coverage_matrix_names_remaining_adaptations() {
    let tensors = release_tensors();
    let model = Glm5NextModel::bind(config(), &tensors).unwrap();
    let rows = model.coverage();
    assert_eq!(rows.iter().map(|r| r.tensors).sum::<usize>(), 1383);
    let pending = model
        .pending_coverage()
        .into_iter()
        .map(|r| (r.role, r.dtype, r.tensors))
        .collect::<Vec<_>>();
    assert!(pending.is_empty(), "{pending:?}");
    // Trunk IQ2_S down has no verified all-slot path.
    assert!(coverage::coverage(TensorRole::ExpertDown, GgmlType::IQ2_S).is_none());
    // Absent pairs have no path rather than an implicit F32 expansion.
    assert!(coverage::coverage(TensorRole::ExpertGateUp, GgmlType::Q8_0).is_none());
    assert!(coverage::coverage(TensorRole::LatentAbsorb, GgmlType::Q6_K).is_none());
}

#[test]
fn execution_gate_is_phase_specific() {
    let tensors = release_tensors();
    let model = Glm5NextModel::bind(config(), &tensors).unwrap();
    model
        .validate_execution(ExecutionMode::SerialDecode)
        .unwrap();
    model
        .validate_execution(ExecutionMode::PackedPrefill)
        .unwrap();
}

fn term(l: &Glm5NextMemoryLedger, name: &str) -> u64 {
    l.terms()
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no ledger term {name}"))
        .1
}

/// 16 KiB pages of the device-free Apple silicon pricing profile.
const G: u64 = memory::APPLE_16K_GRANULE;

/// Ledger under the device-free Apple silicon profile.
fn ledger(
    c: &Glm5NextConfig,
    retained: u64,
    capacity: u64,
    rows: u64,
) -> Result<Glm5NextMemoryLedger> {
    Glm5NextMemoryLedger::new(c, retained, capacity, rows, &memory::apple_16k_price)
}

fn max_capacity(c: &Glm5NextConfig, retained: u64, rows: u64, budget: u64) -> Option<u64> {
    Glm5NextMemoryLedger::max_capacity(c, retained, rows, budget, &memory::apple_16k_price).unwrap()
}

/// Hand-derived from the release geometry (h 4096, 64 heads, KDA width 8192,
/// MLA width 16384, 288 experts, top-8, 34 KDA / 11 MLA / 42 MoE executed
/// blocks), one granule-rounded buffer at a time, independent of the spec
/// functions so a layout change must update these numbers deliberately.
#[test]
fn ledger_terms_match_release_geometry() {
    let c = config();
    let l = ledger(&c, 100 * GIB, 32_768, 512).unwrap();
    assert_eq!((l.capacity(), l.prefill_rows()), (32_768, 512));
    // KDA: conv [8192,3,3] F32 = 18 G, S [128,128,64] F32 = 256 G.
    let kda = 34 * (18 + 256) * G;
    // MLA: latents [512,32768] F16 = 2048 G, pending ring 1 G, pools
    // [128,8192] F16 = 128 G.
    let mla = 11 * (2048 + 1 + 128) * G;
    // 46 decode buffers; every sub-granule vector costs one granule; logits
    // [154880] F32 = 38 G. The fused mHC pre needs no [16384] ones or
    // normalized row (4 G each), only sub-granule partials [24, 64] and [64].
    let decode = 129 * G;
    // ids, weights, status: one granule each per MoE block.
    let routes = 42 * 3 * G;
    // 512-row packed activations (largest: query/output latents 4096 G each,
    // slot outputs 4096 G). The fused mHC pre replaces the [16384, 512]
    // normalized rows (2048 G) with partials [24, 64, 512] (192 G) and
    // [64, 512] (8 G).
    let packed = 36_193 * G;
    // Sparse decode (capacity reaches the frontier): eleven sub-granule
    // buffers plus [8192] F32 pool scores (2 G), and split selected
    // attention partials for 64 heads x 17 splits: [512, 1088] F32 136 G and
    // [2, 1088] F32 1 G.
    let sparse = (12 + 137) * G;
    // Packed sparse (512 rows, 64-query microbatches): queries 512 + 256 G,
    // weights 4 G, visibility 2 G, scores [8192, 64] 128 G, pools 8 + 1 G,
    // rows [2051, 64] 33 + 1 G, statuses [11 x 512] 2 G; split attention
    // partials for 16-query sub-batches: [512, 17408] F32 2176 G and
    // [2, 17408] F32 9 G.
    let packed_sparse = (947 + 2185) * G;
    for (name, bytes) in [
        ("retained_weights", 100 * GIB),
        ("kda_state", kda),
        ("mla_state", mla),
        ("decode_scratch", decode),
        ("decode_routes", routes),
        ("sparse_decode", sparse),
        ("packed_scratch", packed),
        ("packed_sparse", packed_sparse),
        ("packed_routes", routes),
        ("reserve", memory::DYNAMIC_RESERVE_BYTES),
        ("session_state", kda + mla),
        (
            "session_buffers",
            kda + mla + decode + routes + sparse + packed + packed_sparse + routes,
        ),
    ] {
        assert_eq!(term(&l, name), bytes, "{name}");
    }
    let buffers = kda + mla + decode + 2 * routes + sparse + packed + packed_sparse;
    assert_eq!(l.session_state_bytes(), kda + mla);
    assert_eq!(l.session_buffer_bytes(), buffers);
    let p = l.phase_peaks();
    assert_eq!(p.resident, 100 * GIB);
    assert_eq!(
        p.session,
        100 * GIB + buffers + memory::DYNAMIC_RESERVE_BYTES
    );
    assert_eq!(l.peak_bytes(), p.session);
    // Native compact cache: 11.69 KiB/token versus llama.cpp's 19.25 KiB.
    let per_token = (mla - 11 * G) as f64 / 32_768.0;
    assert!((per_token / 1024.0 - 11.6875).abs() < 1e-9, "{per_token}");

    // Decode-only prices no packed scratch and no packed routes.
    let d = ledger(&c, 100 * GIB, 32_768, 0).unwrap();
    assert_eq!(term(&d, "packed_scratch") + term(&d, "packed_routes"), 0);
    assert_eq!(term(&d, "packed_sparse"), 0);
    assert_eq!(
        d.session_buffer_bytes(),
        buffers - packed - packed_sparse - routes
    );
    assert_eq!(
        d.peak_bytes(),
        l.peak_bytes() - packed - packed_sparse - routes
    );
    // Sparse scratch exists exactly from the frontier on.
    assert_eq!(term(&ledger(&c, 0, 2051, 0).unwrap(), "sparse_decode"), 0);
    assert!(term(&ledger(&c, 0, 2052, 0).unwrap(), "sparse_decode") > 0);

    // 513 rows: ids and weights [8, 513] cross into a second granule.
    let r513 = ledger(&c, 100 * GIB, 32_768, 513).unwrap();
    assert_eq!(term(&r513, "packed_routes"), 42 * (2 + 2 + 1) * G);
    assert!(term(&r513, "packed_scratch") > packed);

    assert!(ledger(&c, 0, 0, 512).is_err());
    assert!(ledger(&c, 0, (1 << 20) + 1, 512).is_err());
    // A chunk never exceeds the session; absurd rows are refused, not
    // wrapped.
    assert!(matches!(
        ledger(&c, 0, 16, 17),
        Err(Glm5NextError::InvalidMetadata { .. })
    ));
    assert!(matches!(
        ledger(&c, 0, 16, 1 << 62),
        Err(Glm5NextError::InvalidMetadata { .. })
    ));
    assert!(ledger(&c, 0, 1 << 20, 1 << 20).is_ok());
    // Spec sizes and sums are checked even when called directly.
    let absurd = memory::packed_scratch_specs(&c, 1 << 62);
    assert!(absurd.iter().any(|s| s.bytes().is_none()));
    assert!(matches!(
        memory::priced(&absurd, &memory::apple_16k_price),
        Err(Glm5NextError::Overflow(_))
    ));
    // A device that cannot allocate a buffer refuses the session by name;
    // prices that cannot be summed overflow.
    let no_large = |bytes: u64| (bytes <= 1 << 20).then_some(bytes);
    let refused = Glm5NextMemoryLedger::new(&c, 0, 32_768, 0, &no_large).unwrap_err();
    assert!(refused.to_string().contains("buffer"), "{refused}");
    let huge = |_: u64| Some(u64::MAX / 2);
    assert!(matches!(
        Glm5NextMemoryLedger::new(&c, 0, 16, 0, &huge),
        Err(Glm5NextError::Overflow(_))
    ));
    // Unrepresentable totals are errors, not wrapped or panicking sums.
    assert!(matches!(
        ledger(&c, u64::MAX - 1, 16, 1),
        Err(Glm5NextError::Overflow(_))
    ));
}

#[test]
fn buffer_specs_are_unique_nonempty_and_granule_priced() {
    let c = config();
    for (owner, specs) in [
        ("decode", memory::decode_scratch_specs(&c)),
        ("packed", memory::packed_scratch_specs(&c, 1)),
        ("route", memory::route_specs(&c, None)),
        ("route_rows", memory::route_specs(&c, Some(3))),
        ("kda", memory::kda_state_specs(&c)),
        ("sparse", memory::sparse_decode_specs(&c, 2052)),
        ("packed_sparse", memory::packed_sparse_specs(&c, 2052, 3)),
        ("mla", memory::mla_state_specs(&c, 1)),
    ] {
        let mut names: Vec<_> = specs.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), specs.len(), "{owner}: duplicate names");
        for s in &specs {
            let bytes = s.bytes().unwrap();
            let priced = s.priced_bytes(&memory::apple_16k_price).unwrap();
            assert!(bytes > 0, "{owner}.{}", s.name);
            assert!(priced >= bytes);
            assert!(priced.is_multiple_of(G));
            assert!(priced - bytes < G);
        }
    }
    // Partial pools round up: 2051 visible positions publish 512 complete
    // pools plus one in progress.
    let mla = memory::mla_state_specs(&c, 2051);
    let pooled = mla.iter().find(|s| s.name == "pooled").unwrap();
    assert_eq!(pooled.shape, vec![128, 513]);
}

#[test]
fn max_capacity_is_the_largest_fitting_context() {
    let c = config();
    let retained = 109 * GIB + GIB / 2;
    let budget = 112 * GIB;
    for rows in [0, 512] {
        let cap = max_capacity(&c, retained, rows, budget).unwrap();
        let at = |n| ledger(&c, retained, n, rows).unwrap().peak_bytes();
        assert!(at(cap) <= budget);
        assert!(cap == u64::from(c.context_length) || at(cap + 1) > budget);
        assert!(cap >= 32_768, "{cap}");
        assert_eq!(max_capacity(&c, budget, rows, budget), None);
    }
    let cap = |rows| max_capacity(&c, retained, rows, budget).unwrap();
    assert!(cap(0) > cap(512));
}

/// Real artifact census: header-only, no weights read, no GPU. Requires
/// `GLM53_GGUF` (shard 1 of the UD-IQ3_XXS release).
#[test]
#[ignore = "CPU/header-only; requires GLM53_GGUF (GLM-5.3-Flash shard 1)"]
fn release_artifact_census_and_allocation_plan() {
    let Some(path) = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.path_or_skip() else {
        return;
    };
    let gguf = GgufFile::open(&path).unwrap();
    let model = Glm5NextModel::from_gguf(&gguf).unwrap();
    assert_eq!(gguf.tensors.len(), RELEASE_TENSOR_COUNT);
    assert_eq!(model.trunk.len() + model.nextn.len(), RELEASE_TENSOR_COUNT);
    assert_eq!(model.nextn.len(), 29);
    assert_eq!(model.trunk_bytes, 117_558_078_712, "{}", model.trunk_bytes);
    let last_shard = gguf.shard_count() - 1;
    assert!(model.nextn.iter().all(|t| t.shard_idx == last_shard));
    let nextn_start = model.nextn.iter().map(|t| t.data_offset).min().unwrap();

    for row in model.coverage() {
        eprintln!(
            "coverage {:?} {:?} tensors={} GiB={:.3} decode={:?} prefill={:?}",
            row.role,
            row.dtype,
            row.tensors,
            row.bytes as f64 / GIB as f64,
            row.coverage.decode,
            row.coverage.prefill
        );
    }
    assert!(model.pending_coverage().is_empty());
    model
        .validate_execution(ExecutionMode::PackedPrefill)
        .unwrap();
    model
        .validate_execution(ExecutionMode::SerialDecode)
        .unwrap();

    // M4 Max: 16 KiB pages. The per-buffer cap is a conservative assumption;
    // a real load uses the device's maxBufferLength.
    let page = 16 * 1024;
    let (plan, retained) = model.plan_retained(&gguf, page, 32 * GIB as usize).unwrap();
    let ceil = |v: u64| v.div_ceil(page as u64) * page as u64;
    for w in plan.windows.iter().filter(|w| w.shard_idx == last_shard) {
        assert!(
            w.mmap_offset + w.length as u64 <= ceil(nextn_start),
            "window {w:?} reaches NextN at {nextn_start}"
        );
    }
    let overhead = retained - model.trunk_bytes;
    eprintln!(
        "retained windows={} bytes={} overhead={} MiB nextn_start={nextn_start}",
        plan.windows.len(),
        retained,
        overhead >> 20
    );
    assert!(overhead < 64 << 20, "window overhead {overhead}");

    let budget = 112 * GIB;
    for rows in [0, 128, 256, 512] {
        let cap = max_capacity(&model.config, retained, rows, budget).unwrap();
        let l = ledger(&model.config, retained, cap.min(32_768), rows).unwrap();
        eprintln!(
            "rows={rows} max_capacity={cap} (planning bound) peaks@{}={:?} terms={:?}",
            l.capacity(),
            l.phase_peaks(),
            l.terms()
        );
        assert!(cap >= 4096);
    }
}

/// Map #15: snapshot bytes at the #13 prefix (11,129 tokens), from the
/// release geometry by hand: 34 KDA blocks x (8,192 x 9 conv + 128 x 128 x 64
/// state) x 4 B, plus 11 MLA blocks x (512 x 11,129 latent + 128 x 2 x 4
/// pending + 128 x 2,782 pooled) x 2 B.
#[test]
fn snapshot_bytes_at_the_agent_prefix() {
    let c = config();
    let kda = 34u64 * (8_192 * 9 + 128 * 128 * 64) * 4;
    let mla = 11u64 * (512 * 11_129 + 128 * 2 * 4 + 128 * (11_129 / 4)) * 2;
    assert_eq!(kda + mla, 285_847_040);
    assert_eq!(crate::glm5_next_metal::snapshot_bytes(&c, 11_129), Some(kda + mla));
    assert_eq!(crate::glm5_next_metal::snapshot_bytes(&c, u64::MAX), None);
}
