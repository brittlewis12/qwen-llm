use super::*;
use std::io::Write;

mod ownership_metadata;
mod ple_engram;
mod ple_split;
mod storage_mapping;

#[derive(Clone)]
enum Value<'a> {
    F32(f32),
    Bool(bool),
    F32s(Vec<f32>),
    U32(u32),
    U32s(Vec<u32>),
    Str(&'a str),
    Ints(Vec<i32>),
    U64s(Vec<u64>),
}
fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
fn fixture(experts: u32, dtype: GgmlType, width: u64, engram: bool) -> Arc<GgufFile> {
    let (kv, tensors) = fixture_parts(experts, dtype, width, engram);
    write_fixture(kv, tensors)
}
fn fixture_parts(
    experts: u32,
    dtype: GgmlType,
    width: u64,
    engram: bool,
) -> (
    Vec<(&'static str, Value<'static>)>,
    Vec<(String, Vec<u64>, GgmlType)>,
) {
    let mut kv = vec![
        ("general.architecture", Value::Str("deepseek41")),
        ("general.alignment", Value::U32(16384)),
        ("deepseek41.expert_count", Value::U32(experts)),
        ("deepseek41.expert_used_count", Value::U32(1)),
        ("deepseek41.block_count", Value::U32(1)),
        ("deepseek41.context_length", Value::U32(1048576)),
        ("deepseek41.embedding_length", Value::U32(width as u32)),
        (
            "deepseek41.expert_feed_forward_length",
            Value::U32(width as u32),
        ),
        ("deepseek41.attention.head_count_kv", Value::U32(1)),
        ("deepseek41.expert_shared_count", Value::U32(1)),
        ("deepseek41.hc_mult", Value::U32(4)),
        ("deepseek41.hc_sinkhorn_iters", Value::U32(20)),
        ("deepseek41.index_topk", Value::U32(512)),
        ("deepseek41.sliding_window", Value::U32(128)),
        ("deepseek41.rms_norm_eps", Value::F32(1e-20)),
        ("deepseek41.hc_eps", Value::F32(1e-6)),
        ("deepseek41.swiglu_limit", Value::F32(10.0)),
        ("deepseek41.routed_scaling_factor", Value::F32(1.5)),
        ("deepseek41.rope_theta", Value::F32(10000.0)),
        ("deepseek41.compress_rope_theta", Value::F32(160000.0)),
        ("deepseek41.rope_scaling.factor", Value::F32(16.0)),
        ("deepseek41.rope_scaling.beta_fast", Value::F32(32.0)),
        ("deepseek41.rope_scaling.beta_slow", Value::F32(1.0)),
        (
            "deepseek41.rope_scaling.original_max_position_embeddings",
            Value::U32(65536),
        ),
        ("deepseek41.compress_ratios", Value::U64s(vec![0])),
        ("deepseek41.kv_source_layer_ids", Value::U64s(vec![])),
        ("deepseek41.index_source_layer_ids", Value::U64s(vec![])),
        ("deepseek41.vocab_size", Value::U32(3)),
        ("deepseek41.num_attention_heads", Value::U32(1)),
        ("deepseek41.head_dim", Value::U32(2)),
        ("deepseek41.qk_rope_head_dim", Value::U32(2)),
        ("deepseek41.q_lora_rank", Value::U32(2)),
        ("deepseek41.o_lora_rank", Value::U32(2)),
        ("deepseek41.o_groups", Value::U32(1)),
        ("deepseek41.index_n_heads", Value::U32(1)),
        ("deepseek41.index_head_dim", Value::U32(2)),
        ("deepseek41.engram.token_map", Value::Ints(vec![0, 1, 1])),
        ("deepseek41.engram.pad_id", Value::U32(1)),
        (
            "deepseek41.engram.layer_ids",
            Value::U64s(if engram { vec![0] } else { vec![] }),
        ),
    ];
    if engram {
        kv.extend([
            ("deepseek41.engram.head_count", Value::U32(8)),
            ("deepseek41.engram.key_length", Value::U32(256)),
            ("deepseek41.engram.max_ngram_size", Value::U32(4)),
            (
                "deepseek41.engram.offsets",
                Value::U64s((0..24).map(|i| i * 2).collect()),
            ),
        ]);
        kv.push(("deepseek41.engram.primes", Value::U64s(vec![2; 24])));
        kv.push((
            "deepseek41.engram.multipliers",
            Value::U64s(vec![1, 3, 5, 7]),
        ));
    }
    let config = Config {
        hidden: width as usize,
        intermediate: width as usize,
        vocab: 3,
        experts: experts as usize,
        heads: 1,
        head_dim: 2,
        query_rank: 2,
        output_rank: 2,
        groups: 1,
        index_heads: 1,
        index_dim: 2,
        ..Config::default()
    };
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    };
    let mut required = names::required_dense(&schedule, if engram { &[0] } else { &[] });
    if engram {
        required.extend([
            "blk.0.engram_q_norm.weight".into(),
            "blk.0.engram_k_norm.weight".into(),
        ]);
    }
    let mut tensors: Vec<_> = required
        .into_iter()
        .map(|name| {
            let shape = names::expected_shape(&name, &config).unwrap();
            (name, shape, GgmlType::F32)
        })
        .collect();
    for name in [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ] {
        tensors.push((name.into(), vec![width, width, experts as u64], dtype));
    }
    if engram {
        tensors.push((
            "blk.0.engram_embd.weight".into(),
            vec![256, 48],
            GgmlType::Q8_0,
        ));
    }
    (kv, tensors)
}
fn write_fixture(
    kv: Vec<(&str, Value<'_>)>,
    tensors: Vec<(String, Vec<u64>, GgmlType)>,
) -> Arc<GgufFile> {
    write_fixture_layout(kv, tensors, true)
}
fn write_fixture_layout(
    kv: Vec<(&str, Value<'_>)>,
    tensors: Vec<(String, Vec<u64>, GgmlType)>,
    pad_end: bool,
) -> Arc<GgufFile> {
    write_fixture_payload(kv, tensors, pad_end, |_, _, _| None)
}

fn write_fixture_payload(
    kv: Vec<(&str, Value<'_>)>,
    tensors: Vec<(String, Vec<u64>, GgmlType)>,
    pad_end: bool,
    payload: impl Fn(&str, &[u64], GgmlType) -> Option<Vec<u8>>,
) -> Arc<GgufFile> {
    let file = fixture_file(kv, tensors, pad_end, payload);
    Arc::new(GgufFile::from_opened_file(file, "synthetic-v41.gguf").unwrap())
}

fn fixture_file(
    kv: Vec<(&str, Value<'_>)>,
    tensors: Vec<(String, Vec<u64>, GgmlType)>,
    pad_end: bool,
    payload: impl Fn(&str, &[u64], GgmlType) -> Option<Vec<u8>>,
) -> std::fs::File {
    let alignment = kv
        .iter()
        .find_map(|(k, v)| {
            if *k == "general.alignment" {
                if let Value::U32(n) = v {
                    Some(*n as u64)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or(32);
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for (key, val) in kv {
        string(&mut out, key);
        match val {
            Value::F32(v) => {
                out.extend(6u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Value::F32s(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(6u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for n in v {
                    out.extend(n.to_le_bytes());
                }
            }
            Value::Bool(v) => {
                out.extend(7u32.to_le_bytes());
                out.push(u8::from(v));
            }
            Value::U32(v) => {
                out.extend(4u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Value::U32s(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(4u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for n in v {
                    out.extend(n.to_le_bytes());
                }
            }
            Value::Str(v) => {
                out.extend(8u32.to_le_bytes());
                string(&mut out, v);
            }
            Value::Ints(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(5u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for n in v {
                    out.extend(n.to_le_bytes());
                }
            }
            Value::U64s(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(10u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for n in v {
                    out.extend(n.to_le_bytes());
                }
            }
        }
    }
    let mut offset = 0u64;
    let mut payloads = Vec::new();
    for (i, (name, shape, dtype)) in tensors.iter().enumerate() {
        string(&mut out, name);
        out.extend((shape.len() as u32).to_le_bytes());
        for &n in shape {
            out.extend(n.to_le_bytes());
        }
        out.extend((*dtype as u32).to_le_bytes());
        out.extend(offset.to_le_bytes());
        let (block, bytes) = dtype.storage_layout().unwrap();
        if let Some(data) = payload(name, shape, *dtype) {
            assert_eq!(
                data.len() as u64,
                shape.iter().product::<u64>() / block * bytes
            );
            payloads.push((offset, data));
        }
        offset += shape.iter().product::<u64>() / block * bytes;
        if pad_end || i + 1 < tensors.len() {
            offset = offset.next_multiple_of(alignment);
        }
    }
    out.resize(out.len().next_multiple_of(alignment as usize), 0);
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "qwen-v41-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    file.write_all(&out).unwrap();
    file.set_len(out.len() as u64 + offset).unwrap();
    use std::io::{Seek, SeekFrom};
    for (offset, data) in payloads {
        file.seek(SeekFrom::Start(out.len() as u64 + offset))
            .unwrap();
        file.write_all(&data).unwrap();
    }
    file
}

#[test]
fn nonzero_bf16_load_forward_multiple_steps() {
    use super::super::forward::Forward;
    use objc2_metal::MTLBuffer;

    let ctx = MetalContext::new().unwrap();
    for experts in [272, 320, 384] {
        let (kv, tensors) = fixture_parts(experts, GgmlType::BF16, 2, false);
        let payload = |name: &str, shape: &[u64], dtype| {
            let mut values = vec![0.0f32; shape.iter().product::<u64>() as usize];
            match name {
                "token_embd.weight" => values.copy_from_slice(&[1., 2., -2., 1., 2., -1.]),
                "output.weight" => values.copy_from_slice(&[1., 0., 0., 1., 0.5, -0.25]),
                "output_norm.weight" | "blk.0.ffn_norm.weight" => values.fill(1.),
                "blk.0.exp_probs_b.bias" => *values.last_mut().unwrap() = 10.,
                _ if names::bank(name).is_some() => {
                    // Only the last expert is live: wrong expert strides cannot pass.
                    let end = values.len();
                    let diagonal = match names::bank(name).unwrap().1 {
                        "w1" => [0.5, 0.5],
                        "w3" => [0.25, 0.25],
                        "w2" => [0.75, -0.5],
                        _ => unreachable!(),
                    };
                    values[end - 4] = diagonal[0];
                    values[end - 1] = diagonal[1];
                }
                _ => (),
            }
            Some(if dtype == GgmlType::BF16 {
                values
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect()
            } else {
                values.iter().flat_map(|v| v.to_le_bytes()).collect()
            })
        };
        let g = write_fixture_payload(kv, tensors, true, payload);
        let weights = ResidentWeights::load(&ctx, g.clone()).unwrap();
        for (disk, part) in [("gate", "w1"), ("up", "w3"), ("down", "w2")] {
            let name = format!("blk.0.ffn_{disk}_exps.weight");
            let desc = g.find(&name).unwrap();
            let bank = weights.bank(0, part).unwrap();
            assert_eq!(bank.shape, vec![2, 2, experts as u64]);
            assert_eq!(bank.dtype, GgmlType::BF16);
            assert!(!bank.is_writable());
            let mapped = unsafe {
                std::slice::from_raw_parts(
                    bank.buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(bank.offset as usize),
                    desc.n_bytes as usize,
                )
            };
            assert_eq!(mapped, payload(&name, &desc.shape, desc.dtype).unwrap());
        }
        drop(g);
        let forward =
            Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
                .unwrap();
        let mut state = forward.new_state(4).unwrap();
        for (position, token) in [0, 1, 2, 0].into_iter().enumerate() {
            let embedding: [f32; 2] = match token {
                0 => [1., 2.],
                1 => [-2., 1.],
                _ => [2., -1.],
            };
            let inv = ((embedding[0].powi(2) + embedding[1].powi(2)) / 2.)
                .sqrt()
                .recip();
            let mut hidden = embedding;
            for j in 0..2 {
                let x = embedding[j] * inv;
                let gate = 0.5 * x;
                // Top-1 retains sqrt(softplus(0)); it does not normalize to one.
                hidden[j] += 1.5
                    * 2.0f32.ln().sqrt()
                    * [0.75, -0.5][j]
                    * (gate / (1. + (-gate).exp()))
                    * (0.25 * x);
            }
            // Equal mHC streams collapse to a common scale, removed by RMSNorm.
            // Sinkhorn epsilon perturbs the residual by less than the tolerance.
            let inv = ((hidden[0].powi(2) + hidden[1].powi(2)) / 2.)
                .sqrt()
                .recip();
            let expected = [
                hidden[0] * inv,
                hidden[1] * inv,
                (0.5 * hidden[0] - 0.25 * hidden[1]) * inv,
            ];
            let logits = forward.step(&mut state, token, None).unwrap();
            assert_eq!(logits.len(), 3);
            for (actual, expected) in logits.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 3e-5,
                    "experts={experts} position={position}: {logits:?}, expected {expected}"
                );
            }
            assert_eq!(state.position(), position + 1);
            assert!(!state.is_poisoned());
        }
    }
}
#[test]
fn metadata_expert_counts_and_native_dtypes() {
    for experts in [1, 272, 320, 384] {
        for (dtype, width) in [
            (GgmlType::F32, 2),
            (GgmlType::F16, 2),
            (GgmlType::BF16, 2),
            (GgmlType::MXFP4, 32),
            (GgmlType::Q8_0, 32),
            (GgmlType::IQ4_NL, 32),
            (GgmlType::Q2_K, 256),
        ] {
            let g = fixture(experts, dtype, width, false);
            let p = plan(&g).unwrap();
            assert_eq!(p.config.experts, experts as usize);
            assert_eq!(p.config.hidden, width as usize);
            assert_eq!(p.raw_pad, 1);
            assert_eq!(p.token_map, vec![0, 1, 1]);
            assert_eq!(p.bytes.disk_engram, 0);
        }
    }
}
#[test]
fn oversized_plan_is_metadata_only() {
    let mut config = plan(&fixture(384, GgmlType::F32, 2, false)).unwrap().config;
    config.hidden = 1 << 20;
    config.intermediate = 1 << 20;
    let (mut kv, tensors) = fixture_parts(384, GgmlType::F32, 2, false);
    for (key, value) in &mut kv {
        if matches!(
            *key,
            "deepseek41.embedding_length" | "deepseek41.expert_feed_forward_length"
        ) {
            *value = Value::U32(1 << 20);
        }
    }
    let mut g = write_fixture(kv, tensors);
    let g = Arc::get_mut(&mut g).unwrap();
    for t in &mut g.tensors {
        if names::bank(&t.name).is_some() {
            t.shape = vec![1 << 20, 1 << 20, 384];
        } else {
            t.shape = names::expected_shape(&t.name, &config).unwrap();
        }
        t.n_bytes = t.shape.iter().product::<u64>() * 4;
    }
    let p = plan(g).unwrap();
    assert!(p.bytes.resident_bytes() > 1u64 << 50);
}
#[test]
fn inconsistent_geometry_fails() {
    let mut g = fixture(384, GgmlType::MXFP4, 32, false);
    let g = Arc::get_mut(&mut g).unwrap();
    g.tensors
        .iter_mut()
        .find(|t| names::bank(&t.name).is_some())
        .unwrap()
        .shape[2] = 272;
    assert!(plan(g).is_err());
}
#[test]
fn disk_engram_retains_handle_and_bounds_reads() {
    let g = fixture(272, GgmlType::F32, 2, true);
    let disk = DiskEngram::new(g.clone()).unwrap();
    drop(g);
    assert_eq!(disk.decode_rows(0, &[1, 0, 1]).unwrap(), vec![0.0; 768]);
    assert!(disk.decode_rows(0, &[48]).is_err());
    assert!(disk.decode_rows(1, &[0]).is_err());
    assert!(disk.decode_rows(0, &vec![0; 32769]).is_err());
}
#[test]
fn direct_views_retain_mapping_and_preserve_offsets() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let g = fixture(320, GgmlType::F32, 2, false);
    let expected = g.find("blk.0.ffn_gate_exps.weight").unwrap().data_offset;
    let weights = ResidentWeights::load(&ctx, g).unwrap();
    let bank = weights.bank(0, "w1").unwrap().clone();
    drop(weights);
    assert_eq!(bank.shape, vec![2, 2, 320]);
    assert_eq!(bank.dtype, GgmlType::F32);
    assert!(!bank.is_writable());
    assert_eq!(bank.offset % 16384, expected % 16384);
    use objc2_metal::MTLBuffer;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            bank.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(bank.offset as usize),
            16,
        )
    };
    assert_eq!(bytes, [0; 16]);
}

#[test]
fn expert_count_is_mandatory_and_positive() {
    for count in [None, Some(0)] {
        let (mut kv, tensors) = fixture_parts(17, GgmlType::F32, 2, false);
        kv.retain(|(key, _)| *key != "deepseek41.expert_count");
        if let Some(count) = count {
            kv.push(("deepseek41.expert_count", Value::U32(count)));
        }
        let g = write_fixture(kv, tensors);
        let error = plan(&g).unwrap_err().to_string();
        assert!(error.contains("expert_count"), "{error}");
    }
}

#[test]
fn routed_expert_count_is_an_independent_alias() {
    let (mut kv, tensors) = fixture_parts(320, GgmlType::F32, 2, false);
    kv.iter_mut()
        .find(|(key, _)| *key == "deepseek41.expert_count")
        .unwrap()
        .0 = "deepseek41.n_routed_experts";
    assert_eq!(
        plan(&write_fixture(kv.clone(), tensors.clone()))
            .unwrap()
            .config
            .experts,
        320
    );
    kv.push(("deepseek41.expert_count", Value::U32(384)));
    assert!(
        plan(&write_fixture(kv, tensors))
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
}

#[test]
fn declared_math_must_match_implemented_operations() {
    let (kv, tensors) = fixture_parts(320, GgmlType::F32, 2, false);
    for (key, good, bad) in [
        (
            "deepseek41.expert_gating_func",
            Value::U32(4),
            Value::U32(3),
        ),
        (
            "deepseek41.expert_weights_norm",
            Value::Bool(true),
            Value::Bool(false),
        ),
        (
            "deepseek41.scoring_func",
            Value::Str("sqrtsoftplus"),
            Value::Str("softmax"),
        ),
        (
            "deepseek41.topk_method",
            Value::Str("noaux_tc"),
            Value::Str("greedy"),
        ),
        (
            "deepseek41.hidden_act",
            Value::Str("silu"),
            Value::Str("gelu"),
        ),
        (
            "deepseek41.rope.scaling.type",
            Value::Str("yarn"),
            Value::Str("linear"),
        ),
        (
            "deepseek41.attention.value_length",
            Value::U32(2),
            Value::U32(3),
        ),
        (
            "deepseek41.norm_topk_prob",
            Value::Bool(true),
            Value::Bool(false),
        ),
        (
            "deepseek41.num_key_value_heads",
            Value::U32(1),
            Value::U32(2),
        ),
        (
            "deepseek41.attention.head_count_kv",
            Value::U32(1),
            Value::U32(2),
        ),
    ] {
        let mut accepted = kv.clone();
        accepted.retain(|(k, _)| *k != key);
        accepted.push((key, good));
        plan(&write_fixture(accepted, tensors.clone())).unwrap();
        let mut rejected = kv.clone();
        rejected.retain(|(k, _)| *k != key);
        rejected.push((key, bad));
        assert!(
            plan(&write_fixture(rejected, tensors.clone())).is_err(),
            "ignored {key}"
        );
    }
    let mut conflicting = kv;
    conflicting.retain(|(key, _)| *key != "deepseek41.context_length");
    conflicting.extend([
        ("deepseek41.context_length", Value::U32(1024)),
        ("deepseek41.max_position_embeddings", Value::U32(2048)),
    ]);
    assert!(
        plan(&write_fixture(conflicting, tensors))
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
}

#[test]
fn pr28696_numeric_parameters_and_uniform_clamps_are_consumed() {
    let (mut kv, tensors) = fixture_parts(320, GgmlType::F32, 2, false);
    kv.retain(|(key, _)| {
        !matches!(
            *key,
            "deepseek41.hc_sinkhorn_iters"
                | "deepseek41.hc_eps"
                | "deepseek41.compress_rope_theta"
                | "deepseek41.swiglu_limit"
        )
    });
    for (key, _) in &mut kv {
        *key = match *key {
            "deepseek41.o_groups" => "deepseek41.attention.output_group_count",
            "deepseek41.o_lora_rank" => "deepseek41.attention.output_lora_rank",
            key => key,
        };
    }
    kv.extend([
        (
            "deepseek41.hyper_connection.sinkhorn_iterations",
            Value::U32(3),
        ),
        ("deepseek41.hyper_connection.epsilon", Value::U32(2)),
        (
            "deepseek41.attention.compress_rope_freq_base",
            Value::U32(12345),
        ),
        ("deepseek41.swiglu_clamp_exp", Value::F32s(vec![7.0, 9.0])),
        ("deepseek41.swiglu_clamp_shexp", Value::F32s(vec![7.0, 8.0])),
    ]);
    let p = plan(&write_fixture(kv.clone(), tensors.clone())).unwrap();
    assert_eq!((p.config.groups, p.config.output_rank), (1, 2));
    assert_eq!(p.config.hc_iterations, 3);
    assert_eq!(p.config.hc_eps, 2.0);
    assert_eq!(p.config.compress_theta, 12345.0);
    assert_eq!(p.config.clamp, 7.0);
    kv.iter_mut()
        .find(|(key, _)| *key == "deepseek41.swiglu_clamp_shexp")
        .unwrap()
        .1 = Value::F32s(vec![8.0]);
    assert!(
        plan(&write_fixture(kv, tensors))
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
}

#[test]
fn standard_q8_engram_wkv_alias_and_positional_rows() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let (mut kv, mut tensors) = fixture_parts(320, GgmlType::F32, 2, true);
    tensors
        .iter_mut()
        .find(|(name, _, _)| name == "blk.0.engram_kv.weight")
        .unwrap()
        .0 = "blk.0.engram_wkv.weight".into();
    kv.push(("deepseek41.engram.encoding", Value::Str("Q8_0")));
    let g = write_fixture_payload(kv, tensors, true, |name, _, _| {
        if name != "blk.0.engram_embd.weight" {
            return None;
        }
        let mut rows = vec![0; 48 * 272];
        for (row, bytes) in rows.chunks_exact_mut(272).enumerate() {
            for block in bytes.chunks_exact_mut(34) {
                block[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                block[2..].fill((-(row as i8) - 1) as u8);
            }
        }
        Some(rows)
    });
    let w = ResidentWeights::load(&ctx, g).unwrap();
    let rows = w.engram.decode_rows(0, &[2, 0, 47]).unwrap();
    for (row, expected) in rows.chunks_exact(256).zip([-1.5, -0.5, -24.0]) {
        assert!(row.iter().all(|&v| v == expected));
    }
    assert!(w.dense.contains_key("layers.0.engram.wkv.weight"));
    super::super::forward::Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .unwrap();
}

fn packed_engram_rows(dtype: GgmlType, rows: usize) -> Vec<u8> {
    let (elements, bytes) = dtype.storage_layout().unwrap();
    let stride = (256 / elements * bytes) as usize;
    let mut payload = vec![0; rows * stride];
    for (row, packed) in payload.chunks_exact_mut(stride).enumerate() {
        let scale = (row + 1) as f32 * 0.25;
        if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            for (col, value) in packed.chunks_exact_mut(bytes as usize).enumerate() {
                let f = (col as i32 % 7 - 3) as f32 * scale;
                match dtype {
                    GgmlType::F32 => value.copy_from_slice(&f.to_le_bytes()),
                    GgmlType::F16 => value.copy_from_slice(&half::f16::from_f32(f).to_le_bytes()),
                    GgmlType::BF16 => value.copy_from_slice(&half::bf16::from_f32(f).to_le_bytes()),
                    _ => unreachable!(),
                }
            }
            continue;
        }
        let scale = half::f16::from_f32(scale).to_le_bytes();
        for block in packed.chunks_exact_mut(bytes as usize) {
            match dtype {
                GgmlType::Q2_K => {
                    block[..16].fill(0x11);
                    block[80..82].copy_from_slice(&scale);
                    block[82..84].copy_from_slice(&scale);
                }
                GgmlType::Q3_K => {
                    block[..32].fill(0xff);
                    block[32..96].fill(0x55);
                    block[108..110].copy_from_slice(&scale);
                }
                GgmlType::Q8_0 => {
                    block[..2].copy_from_slice(&scale);
                    for (col, q) in block[2..].iter_mut().enumerate() {
                        *q = (col as i8 - 16) as u8;
                    }
                }
                GgmlType::IQ4_NL | GgmlType::IQ2_XXS | GgmlType::IQ2_XS => {
                    block.fill(0x11);
                    block[..2].copy_from_slice(&scale);
                }
                _ => panic!("no synthetic block fixture for {dtype:?}"),
            }
        }
    }
    payload
}

#[test]
fn engram_standard_codecs_gather_selected_rows_with_exact_strides() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    for dtype in [
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q8_0,
        GgmlType::IQ4_NL,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
    ] {
        let (mut kv, mut tensors) = fixture_parts(320, GgmlType::F32, 2, true);
        // Exercise explicit standard encoding declarations as well as inferred storage.
        kv.push(("deepseek41.engram.encoding", Value::Str(dtype.wire_name())));
        kv.iter_mut()
            .find(|(key, _)| *key == "general.alignment")
            .unwrap()
            .1 = Value::U32(32);
        tensors
            .iter_mut()
            .find(|(name, _, _)| name == "blk.0.engram_embd.weight")
            .unwrap()
            .2 = dtype;
        let payload = packed_engram_rows(dtype, 48);
        let g = write_fixture_payload(kv, tensors, false, |name, _, _| {
            (name == "blk.0.engram_embd.weight").then(|| payload.clone())
        });
        let t = g.find("blk.0.engram_embd.weight").unwrap();
        assert_ne!(t.data_offset, 0);
        assert_eq!(
            t.data_offset + t.n_bytes,
            g.shards[t.shard_idx].mmap_len() as u64
        );
        let expected = crate::codec::dequant_to_f32(t, &payload).unwrap();
        assert!(expected.iter().all(|v| v.is_finite()));
        assert_ne!(
            &expected[..256],
            &expected[47 * 256..],
            "fixture rows must differ for {dtype:?}"
        );
        let p = plan(&g).unwrap();
        assert_eq!(p.bytes.disk_engram, payload.len() as u64);
        let weights = ResidentWeights::load(&ctx, g).unwrap();
        assert!(!weights.dense.keys().any(|key| key.contains("engram_embd")));
        let ids = [47, 0, 13, 1, 47];
        let actual = weights.engram.read_rows(0, &ids).unwrap();
        assert_eq!(actual.len(), ids.len() * 256);
        for (&id, actual) in ids.iter().zip(actual.chunks_exact(256)) {
            let expected = &expected[id as usize * 256..(id as usize + 1) * 256];
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{dtype:?} row {id}"
            );
        }
        assert!(weights.engram.read_rows(0, &[]).unwrap().is_empty());
        assert!(weights.engram.read_rows(0, &[48]).is_err());
        assert!(weights.engram.read_rows(0, &[u64::MAX]).is_err());
        assert!(weights.engram.read_rows(1, &[0]).is_err());
        assert!(weights.engram.read_rows(0, &vec![0; 32769]).is_err());
    }
}

#[test]
fn engram_header_validation_checks_codecs_without_table_payloads() {
    let g = fixture(320, GgmlType::F32, 2, true);
    let mut t = g.find("blk.0.engram_embd.weight").unwrap().clone();
    t.dtype = GgmlType::Q2_K;
    t.shape = vec![256, 1 << 40];
    t.n_bytes = 84 * (1 << 40);
    // Header-only validation must not allocate/decode this enormous descriptor.
    engram::validate(&g, &t).unwrap();
    t.shape[0] = 255;
    assert!(engram::validate(&g, &t).is_err());
    t.shape = vec![256, 48];
    t.dtype = GgmlType::I8;
    t.n_bytes = 256 * 48;
    assert!(
        engram::validate(&g, &t)
            .unwrap_err()
            .to_string()
            .contains("Engram codec")
    );
    t.dtype = GgmlType::from_raw(67);
    assert!(engram::validate(&g, &t).is_err());
    t.dtype = GgmlType::Q2_K;
    t.n_bytes = 84 * 48 - 1;
    assert!(engram::validate(&g, &t).is_err());
    t.shape[1] = u64::MAX;
    assert!(engram::validate(&g, &t).is_err());
}

#[test]
fn engram_rejects_nonfinite_rows_and_mismatched_encoding() {
    let (kv, mut tensors) = fixture_parts(320, GgmlType::F32, 2, true);
    tensors
        .iter_mut()
        .find(|(name, _, _)| name == "blk.0.engram_embd.weight")
        .unwrap()
        .2 = GgmlType::F16;
    let mut bad_encoding = kv.clone();
    bad_encoding.push(("deepseek41.engram.encoding", Value::Str("Q8_0")));
    assert!(
        plan(&write_fixture(bad_encoding, tensors.clone()))
            .unwrap_err()
            .to_string()
            .contains("conflicts")
    );
    let g = write_fixture_payload(kv, tensors, false, |name, _, _| {
        if name != "blk.0.engram_embd.weight" {
            return None;
        }
        let mut bytes = packed_engram_rows(GgmlType::F16, 48);
        bytes[47 * 512..47 * 512 + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
        Some(bytes)
    });
    let disk = DiskEngram::new(g).unwrap();
    assert!(disk.read_rows(0, &[0]).is_ok());
    assert!(
        disk.read_rows(0, &[47])
            .unwrap_err()
            .to_string()
            .contains("non-finite")
    );
}

#[test]
fn custom_engram_dtype_and_encoding_are_explicitly_unsupported() {
    let (kv, tensors) = fixture_parts(320, GgmlType::F32, 2, true);
    for encoding in ["e4m3_e8m0_32_row264", "unknown"] {
        let mut rejected = kv.clone();
        rejected.push(("deepseek41.engram.encoding", Value::Str(encoding)));
        assert!(
            plan(&write_fixture(rejected, tensors.clone()))
                .unwrap_err()
                .to_string()
                .contains("unsupported deepseek41.engram.encoding")
        );
    }
    let mut rejected = tensors;
    let table = rejected
        .iter_mut()
        .find(|(name, _, _)| name == "blk.0.engram_embd.weight")
        .unwrap();
    table.1 = vec![264, 48];
    table.2 = GgmlType::I8;
    let error = plan(&write_fixture(kv, rejected)).unwrap_err().to_string();
    assert!(error.contains("blk.0.engram_embd.weight I8"), "{error}");
}

#[test]
fn unused_tensors_keep_inventory_without_binding_or_dtype_requirements() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let (kv, mut tensors) = fixture_parts(320, GgmlType::F32, 2, false);
    for name in [
        "mtp.unused",
        "vision.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.1.engram_embd.weight",
        "blk.0.exp_probs_b_vl.bias",
    ] {
        tensors.push((name.into(), vec![16], GgmlType::I8));
    }
    let mut g = write_fixture(kv, tensors);
    // Future parser-recognized auxiliary types must not acquire kernel requirements.
    Arc::get_mut(&mut g)
        .unwrap()
        .tensors
        .iter_mut()
        .find(|t| t.name == "vision.weight")
        .unwrap()
        .dtype = GgmlType::Unknown;
    let p = plan(&g).unwrap();
    assert_eq!(p.dtype_inventory["I8"], 64);
    assert_eq!(p.dtype_inventory["Unknown"], 16);
    let w = ResidentWeights::load(&ctx, g).unwrap();
    assert!(!w.dense.contains_key("layers.0.ffn.gate.bias_vl"));
    assert_eq!(w.banks.len(), 3);
    assert_eq!(w.bytes.disk_engram, 0);
    assert!(w.engram.decode_rows(1, &[0]).is_err());
}

#[test]
fn tensor_overflow_is_rejected() {
    let t = TensorDesc {
        name: "overflow".into(),
        shape: vec![u64::MAX, 2],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: 0,
    };
    assert!(validate_storage(&t).is_err());
}

#[test]
fn missing_dense_inventory_fails_before_loading() {
    let mut g = fixture(384, GgmlType::F32, 2, false);
    Arc::get_mut(&mut g)
        .unwrap()
        .tensors
        .retain(|t| t.name != "blk.0.attn_q_b.weight");
    assert!(
        plan(&g)
            .unwrap_err()
            .to_string()
            .contains("missing dense tensor blk.0.attn_q_b.weight")
    );
}

#[test]
fn metadata_hash_survives_planning_and_loading() {
    let g = fixture(320, GgmlType::F32, 2, true);
    let p = plan(&g).unwrap();
    assert_eq!(p.engram_hash.layers[0].layer, 0);
    assert_eq!(p.engram_hash.layers[0].primes, [2; 24]);
    assert_eq!(p.engram_hash.layers[0].multipliers, [1, 3, 5, 7]);
    assert_eq!(p.engram_offsets, (0..24).map(|i| i * 2).collect::<Vec<_>>());
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let w = ResidentWeights::load(&ctx, g).unwrap();
    assert_eq!(w.engram_hash, p.engram_hash);
    assert_eq!(w.schedule, p.schedule);
    super::super::forward::Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .unwrap();
}

#[test]
fn schedule_accepts_structural_variation_not_identity() {
    let schedule = Schedule {
        layers: 6,
        kv_source_layers: vec![1, 4],
        index_source_layers: vec![1, 3, 4, 5],
        compression_ratios: vec![0, 2, 2, 2, 1, 1],
        candidate_source_layer: Some(4),
    };
    schedule.validate().unwrap();
    assert_eq!(schedule.owner(3), Some(1));
    assert_eq!(schedule.owner_slot(5), Some(1));
    assert!(schedule.produces_index(3));
    let mut bad = schedule.clone();
    bad.compression_ratios[2] = 1;
    assert!(bad.validate().is_err());
    bad = schedule;
    bad.index_source_layers.remove(0);
    assert!(bad.validate().is_err());
}

#[test]
fn mandatory_numeric_metadata_cannot_be_inferred_or_defaulted() {
    let (kv, tensors) = fixture_parts(17, GgmlType::F32, 2, false);
    for (key, _) in &kv {
        if !key.starts_with("deepseek41.")
            || key.starts_with("deepseek41.engram.")
            || *key == "deepseek41.vocab_size"
        {
            continue;
        }
        let mut missing = kv.clone();
        missing.retain(|(k, _)| k != key);
        let error = plan(&write_fixture(missing, tensors.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing"), "{key}: {error}");
    }
}

#[test]
fn named_ownership_fallback_requires_known_layer_operations() {
    let ratios: Vec<_> = Schedule::default()
        .compression_ratios
        .iter()
        .map(|&n| n as u64)
        .collect();
    assert_eq!(
        Schedule::deepseek41_shared_kv_layout(40, &ratios),
        Some(Schedule::default())
    );
    for layers in [1, 20, 39] {
        assert!(Schedule::deepseek41_shared_kv_layout(layers, &ratios[..layers]).is_none());
    }
    let mut wrong = ratios.clone();
    wrong[19] = 1;
    assert!(Schedule::deepseek41_shared_kv_layout(40, &wrong).is_none());
    let mut extended = ratios;
    extended.push(1);
    assert!(Schedule::deepseek41_shared_kv_layout(41, &extended).is_none());
    extended.resize(48, 1);
    assert!(Schedule::deepseek41_shared_kv_layout(48, &extended).is_none());
}

#[test]
fn active_engram_metadata_is_required_without_static_hash_fallbacks() {
    let (kv, tensors) = fixture_parts(17, GgmlType::F32, 2, true);
    for key in [
        "deepseek41.context_length",
        "deepseek41.engram.layer_ids",
        "deepseek41.engram.pad_id",
        "deepseek41.engram.head_count",
        "deepseek41.engram.key_length",
        "deepseek41.engram.max_ngram_size",
        "deepseek41.engram.primes",
        "deepseek41.engram.multipliers",
        "deepseek41.engram.offsets",
        "deepseek41.engram.token_map",
    ] {
        let mut missing = kv.clone();
        missing.retain(|(k, _)| *k != key);
        let error = plan(&write_fixture(missing, tensors.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing"), "{key}: {error}");
    }
    let expected = plan(&write_fixture(kv.clone(), tensors.clone())).unwrap();
    let mut aliases = kv;
    for (key, value) in &mut aliases {
        *key = match *key {
            "deepseek41.context_length" => "deepseek41.max_position_embeddings",
            "deepseek41.engram.layer_ids" => "deepseek41.engram.layers",
            "deepseek41.engram.head_count" => "deepseek41.engram.n_heads",
            "deepseek41.engram.key_length" => "deepseek41.engram.head_dim",
            "deepseek41.engram.pad_id" => {
                *value = Value::U32(expected.raw_pad);
                "deepseek41.engram.pad_token_id"
            }
            key => key,
        };
    }
    let actual = plan(&write_fixture(aliases, tensors)).unwrap();
    assert_eq!(actual.engram_primes, expected.engram_primes);
    assert_eq!(actual.engram_offsets, expected.engram_offsets);
    assert_eq!(actual.engram_multipliers, expected.engram_multipliers);
    assert_eq!(
        actual.engram_hash.max_position,
        expected.engram_hash.max_position
    );
    assert_eq!(actual.raw_pad, expected.raw_pad);
}

#[test]
fn empty_engram_layers_need_no_hash_arrays_or_geometry() {
    let (kv, tensors) = fixture_parts(17, GgmlType::F32, 2, false);
    let p = plan(&write_fixture(kv.clone(), tensors.clone())).unwrap();
    assert!(p.engram_hash.layers.is_empty());
    for key in [
        "deepseek41.engram.layer_ids",
        "deepseek41.engram.pad_id",
        "deepseek41.context_length",
    ] {
        let mut missing = kv.clone();
        missing.retain(|(k, _)| *k != key);
        assert!(
            plan(&write_fixture(missing, tensors.clone())).is_err(),
            "{key}"
        );
    }
}

#[test]
fn metadata_schedules_and_auxiliary_compression_entries_are_retained() {
    let custom = Schedule {
        layers: 6,
        kv_source_layers: vec![1, 4],
        index_source_layers: vec![1, 3, 4, 5],
        compression_ratios: vec![0, 2, 2, 2, 1, 1],
        candidate_source_layer: Some(4),
    };
    let config = plan(&fixture(384, GgmlType::F32, 2, false)).unwrap().config;
    let mut extended = Schedule::default();
    extended.layers = 41;
    extended.compression_ratios.push(1);
    for schedule in [custom, Schedule::default(), extended] {
        let (mut kv, _) = fixture_parts(384, GgmlType::F32, 2, false);
        kv.retain(|(key, _)| {
            !matches!(
                *key,
                "deepseek41.compress_ratios"
                    | "deepseek41.kv_source_layer_ids"
                    | "deepseek41.index_source_layer_ids"
            )
        });
        kv.iter_mut()
            .find(|(k, _)| *k == "deepseek41.block_count")
            .unwrap()
            .1 = Value::U32(schedule.layers as u32);
        let mut ratios: Vec<_> = schedule
            .compression_ratios
            .iter()
            .map(|&n| n as u64)
            .collect();
        ratios.extend([0; 3]);
        kv.push(("deepseek41.compress_ratios", Value::U64s(ratios.clone())));
        kv.push((
            "deepseek41.kv_source_layer_ids",
            Value::U64s(
                schedule
                    .kv_source_layers
                    .iter()
                    .map(|&n| n as u64)
                    .collect(),
            ),
        ));
        kv.push((
            "deepseek41.index_source_layer_ids",
            Value::U64s(
                schedule
                    .index_source_layers
                    .iter()
                    .map(|&n| n as u64)
                    .collect(),
            ),
        ));
        let mut tensors: Vec<_> = names::required_dense(&schedule, &[])
            .into_iter()
            .map(|name| {
                let shape = names::expected_shape(&name, &config).unwrap();
                (name, shape, GgmlType::F32)
            })
            .collect();
        for layer in 0..schedule.layers {
            for part in ["gate", "up", "down"] {
                tensors.push((
                    format!("blk.{layer}.ffn_{part}_exps.weight"),
                    vec![2, 2, 384],
                    GgmlType::F32,
                ));
            }
        }
        kv.push((
            "deepseek41.candidate_source_layer_id",
            Value::U32(schedule.candidate_source_layer.unwrap() as u32),
        ));
        for published in [false, true] {
            let mut kv = kv.clone();
            if published {
                published_keys(&mut kv);
            }
            if schedule.layers == 41 {
                for key in [
                    "deepseek41.kv_source_layer_ids",
                    "deepseek41.index_source_layer_ids",
                    "deepseek41.candidate_source_layer_id",
                ] {
                    let mut missing = kv.clone();
                    missing.retain(|(k, _)| *k != key);
                    let error = plan(&write_fixture(missing, tensors.clone()))
                        .unwrap_err()
                        .to_string();
                    assert!(error.contains("missing") && error.contains(key), "{error}");
                }
            }
            if schedule == Schedule::default() {
                let mut implicit = kv.clone();
                implicit.retain(|(key, _)| {
                    !matches!(
                        *key,
                        "deepseek41.kv_source_layer_ids"
                            | "deepseek41.attention.kv_source_layers"
                            | "deepseek41.index_source_layer_ids"
                            | "deepseek41.attention.indexer.source_layers"
                            | "deepseek41.candidate_source_layer_id"
                            | "deepseek41.attention.indexer.candidate_source_layer"
                    )
                });
                for experts in [17, 272, 384] {
                    implicit
                        .iter_mut()
                        .find(|(key, _)| *key == "deepseek41.expert_count")
                        .unwrap()
                        .1 = Value::U32(experts);
                    let mut banks = tensors.clone();
                    for (name, shape, _) in &mut banks {
                        if name.ends_with("_exps.weight") {
                            shape[2] = experts as u64;
                        }
                        if name.contains("ffn_gate_inp") || name.contains("exp_probs_b") {
                            *shape.last_mut().unwrap() = experts as u64;
                        }
                    }
                    // Parsing isolates ownership from dense/expert binding validation.
                    assert_eq!(
                        metadata::parse(&write_fixture(implicit.clone(), banks))
                            .unwrap()
                            .schedule,
                        schedule
                    );
                }
            }
            let g = write_fixture(kv, tensors.clone());
            let p = plan(&g).unwrap();
            assert_eq!(p.schedule, schedule);
            assert_eq!(p.compression_ratios, ratios);
        }
    }
}

fn published_keys(kv: &mut [(&'static str, Value<'static>)]) {
    for (key, _) in kv {
        *key = match *key {
            "deepseek41.num_attention_heads" => "deepseek41.attention.head_count",
            "deepseek41.head_dim" => "deepseek41.attention.key_length",
            "deepseek41.qk_rope_head_dim" => "deepseek41.rope.dimension_count",
            "deepseek41.q_lora_rank" => "deepseek41.attention.q_lora_rank",
            "deepseek41.o_lora_rank" => "deepseek41.attention.output_lora_rank",
            "deepseek41.o_groups" => "deepseek41.attention.output_group_count",
            "deepseek41.sliding_window" => "deepseek41.attention.sliding_window",
            "deepseek41.rms_norm_eps" => "deepseek41.attention.layer_norm_rms_epsilon",
            "deepseek41.hc_eps" => "deepseek41.hyper_connection.epsilon",
            "deepseek41.hc_sinkhorn_iters" => "deepseek41.hyper_connection.sinkhorn_iterations",
            "deepseek41.routed_scaling_factor" => "deepseek41.expert_weights_scale",
            "deepseek41.rope_theta" => "deepseek41.rope.freq_base",
            "deepseek41.compress_rope_theta" => "deepseek41.attention.compress_rope_freq_base",
            "deepseek41.rope_scaling.factor" => "deepseek41.rope.scaling.factor",
            "deepseek41.rope_scaling.beta_fast" => "deepseek41.rope.scaling.yarn_beta_fast",
            "deepseek41.rope_scaling.beta_slow" => "deepseek41.rope.scaling.yarn_beta_slow",
            "deepseek41.rope_scaling.original_max_position_embeddings" => {
                "deepseek41.rope.scaling.original_context_length"
            }
            "deepseek41.index_n_heads" => "deepseek41.attention.indexer.head_count",
            "deepseek41.index_head_dim" => "deepseek41.attention.indexer.key_length",
            "deepseek41.index_topk" => "deepseek41.attention.indexer.top_k",
            "deepseek41.compress_ratios" => "deepseek41.attention.compress_ratios",
            "deepseek41.hc_mult" => "deepseek41.hyper_connection.count",
            "deepseek41.engram.rows" => "deepseek41.engram.num_embeddings",
            key => key,
        };
    }
}

#[test]
fn published_scalar_and_engram_aliases_are_consumed() {
    let (mut kv, tensors) = fixture_parts(384, GgmlType::F32, 2, true);
    kv.retain(|(key, _)| !matches!(*key, "deepseek41.index_topk" | "deepseek41.hc_mult"));
    kv.extend([
        ("deepseek41.index_topk", Value::U32(7)),
        ("deepseek41.candidate_topk_blocks", Value::U32(5)),
        ("deepseek41.candidate_block_size", Value::U32(3)),
        ("deepseek41.hc_mult", Value::U32(4)),
        ("deepseek41.engram.rows", Value::U64s(vec![48])),
    ]);
    published_keys(&mut kv);
    let g = write_fixture(kv.clone(), tensors.clone());
    let p = plan(&g).unwrap();
    assert_eq!(
        (
            p.config.index_heads,
            p.config.index_dim,
            p.config.index_topk
        ),
        (1, 2, 7)
    );
    assert_eq!((p.config.candidate_topk, p.config.candidate_block), (5, 3));
    assert_eq!(p.engram_hash.layers[0].rows, 48);
    for key in [
        "deepseek41.hyper_connection.count",
        "deepseek41.engram.num_embeddings",
    ] {
        let mut bad = kv.clone();
        bad.iter_mut().find(|(k, _)| *k == key).unwrap().1 = if key.ends_with("count") {
            Value::U32(3)
        } else {
            Value::U64s(vec![49])
        };
        assert!(
            plan(&write_fixture(bad, tensors.clone())).is_err(),
            "{key} was ignored"
        );
    }
}

#[test]
fn candidate_mask_aliases_preserve_values_and_reject_conflicts() {
    let (kv, tensors) = fixture_parts(17, GgmlType::F32, 2, false);
    let aliases = [
        (
            "deepseek41.candidate_block_size",
            "deepseek41.attention.indexer.candidate_block_size",
            3,
        ),
        (
            "deepseek41.candidate_topk_blocks",
            "deepseek41.attention.indexer.candidate_top_k_blocks",
            5,
        ),
    ];
    for spelling in 0..3 {
        let mut declared = kv.clone();
        for (source, native, value) in aliases {
            if spelling != 1 {
                declared.push((source, Value::U32(value)));
            }
            if spelling != 0 {
                declared.push((native, Value::U32(value)));
            }
        }
        let p = plan(&write_fixture(declared, tensors.clone())).unwrap();
        assert_eq!((p.config.candidate_block, p.config.candidate_topk), (3, 5));
    }
    for (source, native, value) in aliases {
        let mut conflicting = kv.clone();
        conflicting.extend([(source, Value::U32(value)), (native, Value::U32(value + 1))]);
        let error = plan(&write_fixture(conflicting, tensors.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting aliases"), "{native}: {error}");
    }
}

#[test]
fn conflicting_published_aliases_are_rejected() {
    let (kv, tensors) = fixture_parts(384, GgmlType::F32, 2, false);
    for (old, new, a, b) in [
        (
            "deepseek41.index_topk",
            "deepseek41.attention.indexer.top_k",
            Value::U32(7),
            Value::U32(8),
        ),
        (
            "deepseek41.compress_ratios",
            "deepseek41.attention.compress_ratios",
            Value::U64s(vec![0]),
            Value::U64s(vec![1]),
        ),
        (
            "deepseek41.engram.rows",
            "deepseek41.engram.num_embeddings",
            Value::U64s(vec![]),
            Value::U64s(vec![48]),
        ),
    ] {
        let mut bad = kv.clone();
        bad.retain(|(key, _)| *key != old && *key != new);
        bad.extend([(old, a), (new, b)]);
        let error = plan(&write_fixture(bad, tensors.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting"), "{new}: {error}");
    }
}

#[test]
fn dense_capabilities_match_linear_not_all_ggml_types() {
    for t in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::IQ1_M] {
        assert!(dense_supported(t));
    }
    for t in [GgmlType::Q5_0, GgmlType::Q5_1] {
        assert!(!dense_supported(t));
    }
}

#[test]
fn singleton_expert_axes_are_restored_on_direct_views() {
    let ctx = MetalContext::new().expect("Metal is required for GGUF validation");
    let g = fixture(1, GgmlType::MXFP4, 32, false);
    assert_eq!(
        g.find("blk.0.ffn_gate_exps.weight").unwrap().shape,
        vec![32, 32]
    );
    let w = ResidentWeights::load(&ctx, g).unwrap();
    assert_eq!(w.bank(0, "w1").unwrap().shape, vec![32, 32, 1]);
    assert_eq!(w.dense["layers.0.ffn.gate.weight"].shape, vec![32, 1]);
    super::super::forward::Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .unwrap();
}

#[test]
fn unpadded_and_engram_shared_pages_use_private_boundary_pages() {
    for name in ["blk.0.engram_embd.weight", "per_layer_token_embd.weight"] {
        assert_private_engram_boundary_pages(name);
    }
}

fn assert_private_engram_boundary_pages(table_name: &str) {
    use std::os::unix::fs::FileExt;
    let g = write_fixture_layout(
        vec![("general.alignment", Value::U32(32))],
        vec![
            ("prefix".into(), vec![32], GgmlType::F32),
            (table_name.into(), vec![256, 100], GgmlType::Q8_0),
            ("suffix".into(), vec![32, 1024], GgmlType::F32),
        ],
        false,
    );
    let table = g.find(table_name).unwrap().clone();
    let suffix = g.find("suffix").unwrap().clone();
    // Populate this tiny fixture before reopening its immutable parsed handle.
    g.shards[0]
        .file
        .write_all_at(&vec![0x55; table.n_bytes as usize], table.data_offset)
        .unwrap();
    g.shards[0]
        .file
        .write_all_at(&[0x12; 32], suffix.data_offset)
        .unwrap();
    let file = g.shards[0].file.try_clone().unwrap();
    drop(g);
    let g = GgufFile::from_opened_file(file, "private-edge-fixture.gguf").unwrap();
    let page = metal::host_page_size_bytes().unwrap();
    for name in ["prefix", "suffix"] {
        let t = g.find(name).unwrap();
        let start = t.data_offset / page as u64 * page as u64;
        let end = (t.data_offset + t.n_bytes).next_multiple_of(page as u64);
        let window = metal::RetainedStorageWindow {
            shard_idx: 0,
            mmap_offset: start,
            length: (end - start) as usize,
        };
        let (mapped, local) = storage::mapping(&g, &window, page).unwrap();
        assert!(local);
        let left = table.data_offset.max(start);
        let right = (table.data_offset + table.n_bytes).min(end);
        assert!(
            mapped[(left - start) as usize..(right - start) as usize]
                .iter()
                .all(|&v| v == 0)
        );
        if name == "suffix" {
            assert_eq!(
                &mapped[(t.data_offset - start) as usize..(t.data_offset - start) as usize + 32],
                &[0x12; 32]
            );
            assert!(
                mapped[(t.data_offset + t.n_bytes - start) as usize..]
                    .iter()
                    .all(|&v| v == 0)
            );
        }
    }
    let mut original = [0; 32];
    g.read_shard_exact_at(0, table.data_offset, &mut original)
        .unwrap();
    assert_eq!(original, [0x55; 32]);
    g.revalidate_retained_shard_stamps().unwrap();
}
