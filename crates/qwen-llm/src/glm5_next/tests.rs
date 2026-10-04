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
    let cases: Vec<(&str, Box<dyn Fn(&mut BTreeMap<String, Value>)>)> = vec![
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
    ];
    for (needle, f) in cases {
        let err = mutate(&*f);
        assert!(err.contains(needle), "{needle:?} not in {err:?}");
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

#[test]
fn ledger_terms_match_release_geometry() {
    let c = config();
    assert_eq!(memory::row_activation_floats(&c), 240_256);
    let l = Glm5NextMemoryLedger::new(&c, 100 * GIB, 32_768, 512).unwrap();
    assert_eq!((l.capacity(), l.prefill_rows()), (32_768, 512));
    for (name, bytes) in [
        ("recurrent_state", 136 << 20),
        ("conv_state", 34 * 3 * 3 * 8192 * 4),
        ("latent_cache", 352 << 20),
        ("pooled_keys", 22 << 20),
        ("pending_pool", 11 * 4 * 2 * 128 * 2),
        ("decode_activations", 240_256 * 4),
        ("decode_routing", (2 * 288 + 2 * 8 + 1) * 4),
        ("decode_attention_partials", 64 * 16 * 514 * 4),
        ("prefill_activations", 240_256 * 512 * 4),
        ("prefill_selection", (8192 + 2051) * 512 * 4),
        ("prefill_routing", (2 * 288 + 2 * 8 + 1) * 512 * 4),
        ("logits", 154_880 * 4),
        ("reserve", memory::DYNAMIC_RESERVE_BYTES),
    ] {
        assert_eq!(term(&l, name), bytes, "{name}");
    }
    let state: u64 = [
        "recurrent_state",
        "conv_state",
        "latent_cache",
        "pooled_keys",
        "pending_pool",
    ]
    .iter()
    .map(|n| term(&l, n))
    .sum();
    assert_eq!(l.session_state_bytes(), state);
    let p = l.phase_peaks();
    assert_eq!(p.resident, 100 * GIB);
    assert_eq!(p.session, 100 * GIB + state + memory::DYNAMIC_RESERVE_BYTES);
    assert!(p.session < p.decode && p.decode < p.prefill);
    assert_eq!(l.peak_bytes(), p.prefill);
    // Native compact cache: 11.69 KiB/token versus llama.cpp's 19.25 KiB.
    let per_token = (term(&l, "latent_cache") + term(&l, "pooled_keys")) as f64 / 32_768.0;
    assert!((per_token / 1024.0 - 11.6875).abs() < 1e-9, "{per_token}");
    assert!(Glm5NextMemoryLedger::new(&c, 0, 0, 512).is_err());
    assert!(Glm5NextMemoryLedger::new(&c, 0, 1 << 21, 512).is_err());
    assert!(Glm5NextMemoryLedger::new(&c, 0, 16, 0).is_err());
    // Unrepresentable totals are errors, not wrapped or panicking sums.
    assert!(matches!(
        Glm5NextMemoryLedger::new(&c, u64::MAX - 1, 16, 1),
        Err(Glm5NextError::Overflow(_))
    ));
}

#[test]
fn max_capacity_is_the_largest_fitting_context() {
    let c = config();
    let retained = 109 * GIB + GIB / 2;
    let budget = 112 * GIB;
    let cap = Glm5NextMemoryLedger::max_capacity(&c, retained, 512, budget)
        .unwrap()
        .unwrap();
    let at = |n| {
        Glm5NextMemoryLedger::new(&c, retained, n, 512)
            .unwrap()
            .peak_bytes()
    };
    assert!(at(cap) <= budget);
    assert!(cap == u64::from(c.context_length) || at(cap + 1) > budget);
    assert!(cap >= 32_768, "{cap}");
    assert_eq!(
        Glm5NextMemoryLedger::max_capacity(&c, budget, 512, budget).unwrap(),
        None
    );
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
    for rows in [128, 256, 512] {
        let cap = Glm5NextMemoryLedger::max_capacity(&model.config, retained, rows, budget)
            .unwrap()
            .unwrap();
        let l = Glm5NextMemoryLedger::new(&model.config, retained, cap.min(32_768), rows).unwrap();
        eprintln!(
            "rows={rows} max_capacity={cap} (planning bound) peaks@{}={:?}",
            l.capacity(),
            l.phase_peaks()
        );
        assert!(cap >= 4096);
    }
}
