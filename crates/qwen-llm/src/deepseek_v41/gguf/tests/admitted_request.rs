use super::*;

pub(in crate::deepseek_v41) const RECIPE_CPU_BYTES: u64 = 1 << 20;
// Unlike 2049, this does not add a scalar decoder unit only at the 2K width.
pub(in crate::deepseek_v41) const REQUEST_TOKENS: usize = 2050;

pub(in crate::deepseek_v41) fn fixture() -> (Arc<GgufFile>, u64) {
    let (mut kv, _) = fixture_parts(4, GgmlType::F32, 32, false);
    for (key, value) in &mut kv {
        *value = match *key {
            "deepseek41.block_count" => Value::U32(4),
            "deepseek41.expert_used_count"
            | "deepseek41.num_attention_heads"
            | "deepseek41.index_n_heads" => Value::U32(2),
            "deepseek41.head_dim" | "deepseek41.o_lora_rank" | "deepseek41.index_head_dim" => {
                Value::U32(4)
            }
            "deepseek41.q_lora_rank" => Value::U32(8),
            "deepseek41.index_topk" => Value::U32(2),
            "deepseek41.sliding_window" => Value::U32(3),
            "deepseek41.compress_ratios" => Value::U64s(vec![0, 1, 1, 1]),
            "deepseek41.kv_source_layer_ids" => Value::U64s(vec![1]),
            "deepseek41.index_source_layer_ids" => Value::U64s(vec![1, 2]),
            _ => continue,
        };
    }
    kv.extend([
        ("deepseek41.candidate_source_layer_id", Value::U32(1)),
        ("deepseek41.candidate_block_size", Value::U32(8)),
        ("deepseek41.candidate_topk_blocks", Value::U32(2)),
    ]);
    let c = Config {
        hidden: 32,
        intermediate: 32,
        vocab: 3,
        experts: 4,
        topk: 2,
        heads: 2,
        head_dim: 4,
        rope_dim: 2,
        query_rank: 8,
        output_rank: 4,
        groups: 1,
        index_heads: 2,
        index_dim: 4,
        index_topk: 2,
        window: 3,
        candidate_block: 8,
        candidate_topk: 2,
        ..Config::default()
    };
    let s = Schedule {
        layers: 4,
        kv_source_layers: vec![1],
        index_source_layers: vec![1, 2],
        compression_ratios: vec![0, 1, 1, 1],
        candidate_source_layer: Some(1),
    };
    let mut tensors: Vec<_> = names::required_dense(&s, &[])
        .into_iter()
        .map(|name| {
            let shape = names::expected_shape(&name, &c).unwrap();
            (name, shape, GgmlType::F32)
        })
        .collect();
    for layer in 0..s.layers {
        for part in ["gate", "up", "down"] {
            tensors.push((
                format!("blk.{layer}.ffn_{part}_exps.weight"),
                vec![32, 32, 4],
                GgmlType::F32,
            ));
        }
    }
    // Writer retains payloads and a growing serialization vector concurrently.
    // This exact recipe has <256 entries and <1 MiB payload; 32 MiB covers
    // their capacities, page padding, parsed metadata/mappings and name copies.
    assert!(tensors.len() < 256);
    let payload_bytes: u64 = tensors
        .iter()
        .map(|(_, shape, _)| shape.iter().product::<u64>() * 4)
        .sum();
    assert!(payload_bytes < 1 << 20);
    let g = write_fixture_payload(kv, tensors, true, |name, shape, dtype| {
        assert_eq!(dtype, GgmlType::F32);
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let values: Vec<f32> = (0..shape.iter().product::<u64>() as usize)
            .map(|i| {
                if name.contains("norm.weight") {
                    1.0
                } else if name.contains("_scale.weight") {
                    [0.25, 0.5, 0.75][i]
                } else if name.contains("_base.weight") {
                    ((i % 7) as f32 - 3.0) / 16.0
                } else if name.contains("sinks") {
                    0.0
                } else {
                    (((i * 17 + seed * 13 + i / shape[0] as usize * 7) % 61) as f32 - 30.0)
                        / (20.0 * (shape[0] as f32).sqrt())
                }
            })
            .collect();
        Some(bytemuck::cast_slice(&values).to_vec())
    });
    let p = plan(&g).unwrap();
    p.validate_execution().unwrap();
    assert_eq!(p.schedule, s);
    assert_eq!(p.config.hidden, c.hidden);
    assert!(p.engram_hash.layers.is_empty());
    (g, 32 << 20)
}

#[test]
fn small_ced_request_fixture_is_executable_metadata_not_a_mock_step() {
    let (g, cpu) = fixture();
    let p = plan(&g).unwrap();
    assert_eq!(p.layers, 4);
    assert_eq!(p.config.window, 3);
    assert_eq!(p.config.topk, 2);
    assert_eq!(p.schedule.candidate_source_layer, Some(1));
    assert_eq!(p.controls.payload_bytes_read, 0);
    assert!(p.bytes.resident_bytes() < 1 << 20);
    assert_eq!(cpu, 32 << 20);
}

#[test]
fn small_request_cpu_quote_preserves_optional_intent_and_required_pair_refusal() {
    use crate::deepseek_v41::forward::{PrefillExecution as E, PrefillIntent, preflight};
    let (g, _) = fixture();
    let storage = resident_storage_plan(&g, 16384, 1 << 30).unwrap();
    let observed = super::resident_storage::preflight_signals();
    for half in [false, true] {
        let quote = |executions: &[E], signals| {
            preflight::preflight_with_pricing(
                &g,
                storage.clone(),
                REQUEST_TOKENS,
                REQUEST_TOKENS + 1,
                executions,
                40 << 20,
                half,
                &|n| Ok(n.next_multiple_of(16384)),
                signals,
            )
        };
        let ced = quote(&[E::ShrinkingCed], observed).unwrap();
        let full = quote(&[E::FullForward], observed).unwrap();
        assert!(ced.required_metal_bytes > full.required_metal_bytes);
        assert_eq!(ced.prefill_admission.intent, PrefillIntent::OptionalCed);
        assert!(ced.prefill_admission.fallback.is_none());
        eprintln!(
            "CPU_ONLY_16K_PAGE_QUOTE tokens={REQUEST_TOKENS} capacity={} half={half} metal={} cpu={} binding_cpu={} runtime_metal={} runtime_cpu={} caller={} resident={} private={}",
            REQUEST_TOKENS + 1,
            ced.required_metal_bytes,
            ced.required_cpu_bytes,
            ced.binding_cpu_bytes,
            ced.runtime_metal_bytes,
            ced.runtime_cpu_bytes,
            ced.caller_cpu_bytes,
            ced.resident_storage.metal_view_bytes,
            ced.resident_storage.private_shadow_bytes
        );
        assert!(
            ced.required_metal_bytes + ced.required_cpu_bytes <= 512 << 20,
            "combined fixture domains: Metal={} CPU={}",
            ced.required_metal_bytes,
            ced.required_cpu_bytes,
        );
        let short = crate::metal::MetalMemorySignals {
            recommended_max_bytes: full.required_metal_bytes,
            ..observed
        };
        let fallback = quote(&[E::ShrinkingCed], short).unwrap();
        let repeated = quote(&[E::ShrinkingCed, E::ShrinkingCed], short).unwrap();
        assert_eq!(repeated.prefill_admission, fallback.prefill_admission);
        assert_eq!(repeated.required_metal_bytes, fallback.required_metal_bytes);
        assert_eq!(repeated.required_cpu_bytes, fallback.required_cpu_bytes);
        assert_eq!(
            fallback.prefill_admission.intent,
            PrefillIntent::OptionalCed
        );
        assert!(fallback.prefill_admission.fallback.is_some());
        assert_eq!(fallback.executions, [E::FullForward]);
        assert!(fallback.ced.is_none());
        assert!(quote(&[E::FullForward, E::ShrinkingCed], short).is_err());
    }
}

#[test]
fn unsupported_metadata_schedule_is_optional_full_not_a_required_ced_arm() {
    use crate::deepseek_v41::forward::{PrefillExecution as E, PrefillIntent, preflight};
    let g = super::fixture(4, GgmlType::F32, 32, false);
    let storage = resident_storage_plan(&g, 16384, 1 << 30).unwrap();
    let quote = |executions: &[E]| {
        preflight::preflight_with_pricing(
            &g,
            storage.clone(),
            3,
            4,
            executions,
            0,
            false,
            &Ok,
            super::resident_storage::preflight_signals(),
        )
    };
    let optional = quote(&[E::ShrinkingCed]).unwrap();
    assert_eq!(
        optional.prefill_admission.intent,
        PrefillIntent::OptionalCed
    );
    assert_eq!(
        optional.prefill_admission.fallback.as_deref(),
        Some("cold planning: CED unsupported")
    );
    assert!(
        quote(&[E::FullForward, E::ShrinkingCed])
            .unwrap_err()
            .to_string()
            .contains("required CED")
    );
}
