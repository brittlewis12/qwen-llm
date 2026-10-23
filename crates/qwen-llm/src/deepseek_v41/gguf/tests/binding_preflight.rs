use super::resident_storage::{M5_MAX_BUFFER_LENGTH, PAGE, preflight_signals};
use super::*;
use crate::deepseek_v41::forward::{PrefillExecution, preflight, preflight_cold_prefill};

type Metadata = Vec<(&'static str, Value<'static>)>;
type Tensors = Vec<(String, Vec<u64>, GgmlType)>;

fn nondefault_parts(engram_layers: &[usize], folded: bool) -> (Metadata, Tensors) {
    let schedule = Schedule {
        layers: 6,
        kv_source_layers: vec![1, 4],
        index_source_layers: vec![1, 3, 4, 5],
        compression_ratios: vec![0, 2, 2, 2, 1, 1],
        candidate_source_layer: Some(4),
    };
    let c = Config {
        hidden: 32,
        intermediate: 32,
        vocab: 3,
        experts: 1,
        topk: 1,
        heads: 4,
        head_dim: 32,
        rope_dim: 2,
        query_rank: 32,
        output_rank: 32,
        groups: 2,
        index_heads: 2,
        index_dim: 32,
        ..Config::default()
    };
    let (mut kv, _) = fixture_parts(1, GgmlType::Q8_0, 32, true);
    for (key, value) in &mut kv {
        match *key {
            "deepseek41.block_count" => *value = Value::U32(6),
            "deepseek41.num_attention_heads" => *value = Value::U32(4),
            "deepseek41.head_dim"
            | "deepseek41.q_lora_rank"
            | "deepseek41.o_lora_rank"
            | "deepseek41.index_head_dim" => *value = Value::U32(32),
            "deepseek41.o_groups" | "deepseek41.index_n_heads" => *value = Value::U32(2),
            "deepseek41.compress_ratios" => *value = Value::U64s(vec![0, 2, 2, 2, 1, 1]),
            "deepseek41.kv_source_layer_ids" => *value = Value::U64s(vec![1, 4]),
            "deepseek41.index_source_layer_ids" => *value = Value::U64s(vec![1, 3, 4, 5]),
            "deepseek41.engram.layer_ids" => {
                *value = Value::U64s(engram_layers.iter().map(|&l| l as u64).collect())
            }
            "deepseek41.engram.primes" => *value = Value::U64s(vec![2; 24 * engram_layers.len()]),
            "deepseek41.engram.offsets" => {
                *value = Value::U64s(
                    (0..engram_layers.len())
                        .flat_map(|_| (0..24).map(|i| i * 2))
                        .collect(),
                )
            }
            "deepseek41.engram.multipliers" => {
                *value = Value::U64s([1, 3, 5, 7].repeat(engram_layers.len()))
            }
            _ => (),
        }
    }
    kv.push(("deepseek41.candidate_source_layer_id", Value::U32(4)));
    let mut required = names::required_dense(&schedule, engram_layers);
    for &layer in engram_layers {
        for tail in if folded {
            vec!["engram_gate.weight"]
        } else {
            vec!["engram_q_norm.weight", "engram_k_norm.weight"]
        } {
            required.push(format!("blk.{layer}.{tail}"));
        }
    }
    // Valid but inactive matrices must not create binding-control charges.
    required.extend(
        [
            "blk.4.attn_compressor_gate.weight",
            "blk.2.attn_compressor_kv.weight",
            "blk.2.indexer.attn_q_b.weight",
            "blk.5.engram_kv.weight",
        ]
        .map(String::from),
    );
    let mut tensors: Tensors = required
        .into_iter()
        .map(|name| {
            let shape = names::expected_shape(&name, &c).unwrap();
            let dtype = if shape.len() == 2 {
                GgmlType::Q8_0
            } else {
                GgmlType::F32
            };
            (name, shape, dtype)
        })
        .collect();
    for layer in 0..6 {
        for part in ["gate", "up", "down"] {
            tensors.push((
                format!("blk.{layer}.ffn_{part}_exps.weight"),
                vec![32, 32, 1],
                GgmlType::Q8_0,
            ));
        }
    }
    for &layer in engram_layers {
        tensors.push((
            format!("blk.{layer}.engram_embd.weight"),
            vec![256, 48],
            GgmlType::Q8_0,
        ));
    }
    (kv, tensors)
}

#[test]
fn nondefault_gguf_preflight_prices_only_active_binding_controls_and_refuses_short_budget() {
    for engram_layers in [vec![], vec![0], vec![0, 2]] {
        for folded in [false, true] {
            let (kv, tensors) = nondefault_parts(&engram_layers, folded);
            let g = write_fixture(kv, tensors);
            let refs = Arc::strong_count(&g.shards[0].mmap);
            let p = plan(&g).unwrap();
            // 6*11 base linears + 3 ratio-two owner + 2 ratio-one owner
            // + 4*2 index-query/weight pairs + head + one per active Engram.
            let controls = 80 + engram_layers.len() as u64;
            assert_eq!(p.binding_linears.len() as u64, controls);
            assert!(p.binding_linears.values().all(|&t| t == GgmlType::Q8_0));
            assert_eq!(
                p.binding_linears["layers.1.attn.compressor.wgate.weight"],
                GgmlType::Q8_0
            );
            assert!(
                !p.binding_linears
                    .contains_key("layers.4.attn.compressor.wgate.weight")
            );
            assert!(
                !p.binding_linears
                    .contains_key("layers.2.attn.compressor.wkv.weight")
            );
            assert!(
                !p.binding_linears
                    .contains_key("layers.2.attn.indexer.wq_b.weight")
            );
            assert!(!p.binding_linears.contains_key("layers.5.engram.wkv.weight"));
            assert!(!p.binding_linears.contains_key("embed.weight"));
            let storage = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
            let run = |signals| {
                preflight::preflight_with_pricing(
                    &g,
                    storage.clone(),
                    8,
                    16,
                    &[
                        PrefillExecution::FullForward,
                        PrefillExecution::ShrinkingCed,
                    ],
                    1234,
                    false,
                    &|n| Ok(n.next_multiple_of(PAGE as u64)),
                    signals,
                )
            };
            let cold = run(preflight_signals()).unwrap();
            assert!(cold.ced.is_some());
            assert_eq!(cold.binding_metal_bytes, controls * 16_384);
            assert_eq!(
                cold.required_metal_bytes - cold.runtime_metal_bytes - storage.metal_view_bytes,
                controls * 16_384
            );
            assert_eq!(
                cold.required_cpu_bytes,
                storage.private_shadow_bytes
                    + cold.binding_cpu_bytes
                    + cold.runtime_cpu_bytes
                    + 1234
            );
            let mut short = preflight_signals();
            short.recommended_max_bytes = cold.required_metal_bytes - 1;
            assert!(
                metal::evaluate_metal_memory_admission_with_cpu_bytes(
                    cold.required_metal_bytes - controls * 16_384,
                    cold.required_cpu_bytes,
                    0,
                    short,
                    true,
                )
                .is_admitted()
            );
            assert!(run(short).unwrap_err().as_memory_admission().is_some());
            let mut short_host = preflight_signals();
            short_host
                .host_budget
                .as_mut()
                .unwrap()
                .conservative_available_bytes =
                cold.required_metal_bytes + cold.required_cpu_bytes - 1;
            assert!(run(short_host).unwrap_err().as_memory_admission().is_some());
            assert_eq!(Arc::strong_count(&g.shards[0].mmap), refs);
        }
    }
}

#[test]
fn mixed_gguf_binding_inventory_keeps_actual_dtypes_and_direct_projection_exclusions() {
    let (kv, mut tensors) = nondefault_parts(&[0, 2], false);
    for (name, _, dtype) in &mut tensors {
        *dtype = match name.as_str() {
            "output.weight" => GgmlType::BF16,
            "blk.0.hc_attn_fn.weight" => GgmlType::F16,
            "blk.0.engram_kv.weight" => GgmlType::IQ2_XXS,
            "blk.2.engram_kv.weight" => GgmlType::IQ1_S,
            _ => *dtype,
        };
    }
    let g = write_fixture(kv, tensors);
    let p = plan(&g).unwrap();
    assert_eq!(p.binding_linears.len(), 82);
    assert_eq!(p.binding_linears["head.weight"], GgmlType::BF16);
    assert_eq!(p.binding_linears["layers.0.hc_attn_fn"], GgmlType::F16);
    assert_eq!(
        p.binding_linears["layers.0.engram.wkv.weight"],
        GgmlType::IQ2_XXS
    );
    assert_eq!(
        p.binding_linears["layers.2.engram.wkv.weight"],
        GgmlType::IQ1_S
    );
    let cold = preflight::preflight_with_pricing(
        &g,
        resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap(),
        8,
        16,
        &[PrefillExecution::ShrinkingCed],
        0,
        false,
        &|n| Ok(n.next_multiple_of(PAGE as u64)),
        preflight_signals(),
    )
    .unwrap();
    assert_eq!(cold.binding_metal_bytes, 78 * 16_384);
}

#[test]
fn binding_preflight_rejects_invalid_linear_inventory_before_pricing() {
    for case in 0..4 {
        let (kv, tensors) = nondefault_parts(&[0, 2], false);
        let mut g = write_fixture(kv, tensors);
        let storage = resident_storage_plan(&g, PAGE, M5_MAX_BUFFER_LENGTH).unwrap();
        let tensors = &mut Arc::get_mut(&mut g).unwrap().tensors;
        match case {
            0 => tensors.retain(|t| t.name != "blk.1.attn_compressor_gate.weight"),
            1 => {
                tensors
                    .iter_mut()
                    .find(|t| t.name == "output.weight")
                    .unwrap()
                    .dtype = GgmlType::I32
            }
            2 => {
                let t = tensors
                    .iter_mut()
                    .find(|t| t.name == "blk.0.engram_kv.weight")
                    .unwrap();
                t.shape[0] -= 32;
                t.n_bytes = t.shape.iter().product::<u64>() / 32 * 34;
            }
            3 => {
                let mut alias = tensors
                    .iter()
                    .find(|t| t.name == "blk.0.engram_kv.weight")
                    .unwrap()
                    .clone();
                alias.name = "blk.0.engram_wkv.weight".into();
                tensors.push(alias);
            }
            _ => unreachable!(),
        }
        let prices = std::cell::Cell::new(0);
        let error = preflight::preflight_with_pricing(
            &g,
            storage,
            8,
            16,
            &[PrefillExecution::FullForward],
            0,
            false,
            &|n| {
                prices.set(prices.get() + 1);
                Ok(n)
            },
            preflight_signals(),
        )
        .unwrap_err();
        assert!(matches!(error, LoadError::Invalid(_)), "{error}");
        assert_eq!(prices.get(), 0);
    }
}

#[test]
fn bounded_gpu_public_preflight_accepts_nondefault_schedule_and_engram_without_import() {
    let ctx = MetalContext::new().unwrap();
    let (kv, tensors) = nondefault_parts(&[0, 2], true);
    let g = write_fixture(kv, tensors);
    let refs = Arc::strong_count(&g.shards[0].mmap);
    let cold = preflight_cold_prefill(
        &ctx,
        &g,
        8,
        16,
        &[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ],
        0,
    )
    .unwrap();
    assert!(cold.ced.is_some());
    assert_eq!(
        cold.binding_metal_bytes,
        82 * ctx.price_shared_buffer_upper(8).unwrap().priced_upper_bytes
    );
    assert_eq!(
        cold.required_metal_bytes,
        cold.resident_storage.metal_view_bytes
            + cold.runtime_metal_bytes
            + cold.binding_metal_bytes
    );
    assert_eq!(Arc::strong_count(&g.shards[0].mmap), refs);
}
