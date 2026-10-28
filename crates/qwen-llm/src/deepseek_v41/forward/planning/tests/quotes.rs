use super::*;
use crate::deepseek_v41::gguf::{self, tests::admitted_request};

pub(super) fn runtime_quote(
    c: &Config,
    schedule: &gguf::Schedule,
    hash: &engram::HashConfig,
    projections: &[(usize, usize)],
    tokens: usize,
    capacity: usize,
    rows: usize,
    half: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<(RequestMemoryEstimate, serde_json::Value)> {
    let (base, full, ced) = preflight::runtime_from_parts(
        c,
        schedule,
        hash,
        projections,
        tokens,
        capacity,
        &[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ],
        rows,
        half,
        price,
    )?;
    let ced = ced.ok_or("fixture requires CED")?;
    let full_runtime = runtime_memory(base, full, None)?;
    let pair = runtime_memory(base, full, Some(&ced))?;
    let geometry = PrefillGeometry {
        start: 0,
        tokens,
        selected_rows: rows,
    };
    let plan =
        ced::plan_for_geometry(c, schedule, hash, geometry, capacity).map_err(|e| e.to_string())?;
    let mut groups = vec![];
    if ced.decoder_batch_layers > 0 {
        for group in plan.decoder.chunks(ced.decoder_batch_layers) {
            let layers = group[0].layer..group.last().unwrap().layer + 1;
            let units =
                ced_plan::batches::unit_count(&plan, layers.clone()).map_err(|e| e.to_string())?;
            groups.push(
                serde_json::json!({"layers": [layers.start, layers.end], "physical_units": units}),
            );
        }
    }
    Ok((
        pair,
        serde_json::json!({
            "tokens": tokens, "capacity": capacity, "selected_rows": rows,
            "half_scratch_selected": half,
            "scalar_tail": plan.scalar_tail(), "decoder_groups": groups,
            "input_ranges": geometry.chunk_ranges().map(|r| [r.start, r.end]).collect::<Vec<_>>(),
            "base": {"metal_bytes": base.metal_bytes, "cpu_bytes": base.cpu_bytes},
            "full_incremental": {"metal_bytes": full.0, "cpu_bytes": full.1},
            "full_runtime": {"metal_bytes": full_runtime.metal_bytes, "cpu_bytes": full_runtime.cpu_bytes},
            "ced": ced,
            "required_pair_runtime": {"metal_bytes": pair.metal_bytes, "cpu_bytes": pair.cpu_bytes},
        }),
    ))
}

#[test]
fn admitted_fixture_width_quotes_separate_strict_pair_metal_costs_for_both_half_modes() {
    let (g, _) = admitted_request::fixture();
    let p = gguf::plan(&g).unwrap();
    assert!(p.engram_hash.layers.is_empty());
    assert_eq!(admitted_request::REQUEST_TOKENS, 2050);
    for tokens in [2049, admitted_request::REQUEST_TOKENS, 4097] {
        for half in [false, true] {
            let quote = |rows| {
                let (memory, details) = runtime_quote(
                    &p.config,
                    &p.schedule,
                    &p.engram_hash,
                    &[],
                    tokens,
                    tokens + 1,
                    rows,
                    half,
                    &|n| Ok(n.next_multiple_of(16384)),
                )
                .unwrap();
                eprintln!("CPU_ONLY_REQUEST_WIDTH_QUOTE {details}");
                (memory, details)
            };
            let (narrow, n) = quote(2048);
            let (wide, w) = quote(4096);
            assert_eq!(n["base"], w["base"]);
            assert!(
                w["full_incremental"]["metal_bytes"].as_u64().unwrap()
                    > n["full_incremental"]["metal_bytes"].as_u64().unwrap()
            );
            for (memory, details) in [(narrow, &n), (wide, &w)] {
                assert_eq!(
                    memory.metal_bytes,
                    details["base"]["metal_bytes"].as_u64().unwrap()
                        + details["ced"]["incremental_metal_bytes"].as_u64().unwrap(),
                    "CED, not Full, dominates the strict pair for this fixture"
                );
                assert_eq!(details["ced"]["decoder_batch_layers"], 2);
            }
            if tokens == 2049 {
                assert!(
                    narrow.metal_bytes > wide.metal_bytes,
                    "historical fixture cannot induce 2K selection with MWSS"
                );
                assert_eq!(n["scalar_tail"], 2048);
                assert!(w["scalar_tail"].is_null());
                assert_eq!(n["decoder_groups"][0]["physical_units"], 6);
                assert_eq!(w["decoder_groups"][0]["physical_units"], 4);
                assert_eq!(n["ced"]["batch_encoding_reserve_bytes"], 6 << 20);
                assert_eq!(w["ced"]["batch_encoding_reserve_bytes"], 4 << 20);
                continue;
            }
            let pipeline = |d: &serde_json::Value| {
                (
                    d["ced"]["priced_pipeline_chunk_bytes"].as_u64().unwrap(),
                    d["ced"]["priced_pipeline_cpu_bytes"].as_u64().unwrap(),
                )
            };
            let (narrow_pipe_metal, narrow_pipe_cpu) = pipeline(&n);
            let (wide_pipe_metal, wide_pipe_cpu) = pipeline(&w);
            assert!(
                wide.metal_bytes - wide_pipe_metal > narrow.metal_bytes - narrow_pipe_metal,
                "serialized chunk width must cost more in the wider plan"
            );
            assert!(
                wide.cpu_bytes - wide_pipe_cpu > narrow.cpu_bytes - narrow_pipe_cpu,
                "serialized upload staging must cost more in the wider plan"
            );
            assert_eq!(
                n["ced"]["batch_encoding_reserve_bytes"],
                w["ced"]["batch_encoding_reserve_bytes"]
            );
            if tokens == admitted_request::REQUEST_TOKENS {
                // Serialized width cost is pinned; the pipeline term is the
                // second chunk scratch plus its publication and reserve.
                let serial = if half { 786_432 } else { 737_280 };
                let expected = serial as i64 + wide_pipe_metal as i64 - narrow_pipe_metal as i64;
                assert_eq!(
                    wide.metal_bytes as i64 - narrow.metal_bytes as i64,
                    expected
                );
                assert_eq!(
                    n["ced"]["priced_pipeline_chunk_bytes"],
                    serde_json::json!(
                        n["ced"]["priced_chunk_bytes"].as_u64().unwrap()
                            + n["ced"]["priced_publication_bytes"].as_u64().unwrap()
                            + n["ced"]["command_reserve_bytes"].as_u64().unwrap()
                    )
                );
                assert_eq!(
                    w["ced"]["priced_pipeline_chunk_bytes"],
                    serde_json::json!(0)
                );
                assert!(n["scalar_tail"].is_null() && w["scalar_tail"].is_null());
                assert_eq!(
                    n["input_ranges"],
                    serde_json::json!([[0, 2048], [2048, 2050]])
                );
            }
            let mut observed = signals();
            observed.current_allocated_bytes = 0;
            observed.recommended_max_bytes = 1 << 40;
            observed.process_limit_remaining_bytes = Some(0);
            observed.host_budget = Some(HostMemoryBudget {
                physical_bytes: 1 << 40,
                conservative_available_bytes: 1 << 40,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            });
            let cold = preflight::preflight_with_pricing(
                &g,
                gguf::resident_storage_plan(&g, 16384, 1 << 30).unwrap(),
                tokens,
                tokens + 1,
                &[
                    PrefillExecution::FullForward,
                    PrefillExecution::ShrinkingCed,
                ],
                40 << 20,
                half,
                &|n| Ok(n.next_multiple_of(16384)),
                observed,
            )
            .unwrap();
            assert_eq!(wide.metal_bytes, cold.runtime_metal_bytes);
            assert_eq!(wide.cpu_bytes, cold.runtime_cpu_bytes);
            let reserve = cold
                .binding_cpu_bytes
                .checked_add(cold.caller_cpu_bytes)
                .unwrap()
                .checked_add(cold.resident_storage.private_shadow_bytes)
                .unwrap();
            // The pipeline charge can make the two-chunk plan larger than
            // the one-chunk plan; budget the smaller one and refuse the larger.
            let (smaller, larger) = if wide.metal_bytes < narrow.metal_bytes {
                (wide, narrow)
            } else {
                (narrow, wide)
            };
            observed.recommended_max_bytes = smaller.metal_bytes;
            let cost = |memory: RequestMemoryEstimate| Cost {
                metal: memory.metal_bytes,
                cpu: memory.cpu_bytes.checked_add(reserve).unwrap(),
            };
            assert_eq!(
                cost(larger)
                    .admit(observed)
                    .unwrap_err()
                    .memory_admission()
                    .unwrap()
                    .reason(),
                Reason::WorkingSetInsufficient
            );
            cost(smaller).admit(observed).unwrap();
            assert!(narrow.total_bytes + reserve <= 512 << 20);
            assert!(wide.total_bytes + reserve <= 512 << 20);
            assert!(cold.required_metal_bytes + cold.required_cpu_bytes <= 512 << 20);
        }
    }
}
