//! Model-shaped CPU feasibility only: no header read or Metal resources.
use super::*;
use prefill::with_full_chunk_ceiling_for_test as scoped;

const N: usize = 8192;
const CAPACITY: usize = N + 256;

fn projections(c: &Config, hash: &engram::HashConfig) -> Vec<(usize, usize)> {
    vec![(6144, 5 * c.hidden); hash.layers.len()]
}

// Assumed heap quote and per-allocation allowance, not a measured device price.
// Exercise the actual pure page/alignment upper-bound arithmetic.
fn price_with(bytes: u64, max_buffer: u64, extra: u64) -> Result<u64> {
    if bytes == 0 {
        return Ok(0);
    }
    metal::price_shared_buffer_upper(
        bytes,
        metal::MetalBufferSizeAndAlign {
            size: bytes,
            alignment: 256,
        },
        16384,
        max_buffer,
    )
    .map_err(|e| e.to_string())?
    .priced_upper_bytes
    .checked_add(extra)
    .ok_or_else(|| "fixture price overflow".into())
}

fn price(bytes: u64) -> Result<u64> {
    price_with(bytes, u64::MAX, 1024)
}

fn plan(width: usize, n: usize, capacity: usize) -> ced_plan::CedPlan {
    CedPlan::shrinking(
        &Config::default(),
        &gguf::Schedule::default(),
        &engram::HashConfig::default(),
        &CedRequest::cold(n, capacity).with_full_chunk_rows(width),
    )
    .unwrap()
}

#[test]
fn scope_is_validated_nested_panic_safe_thread_local_and_default_off() {
    assert_eq!(prefill::full_chunk_ceiling(), 4096);
    for invalid in [0, 2048, 4097, 8191, 8193, usize::MAX] {
        assert!(scoped(invalid, || panic!("invalid scope entered")).is_err());
    }
    scoped(N, || {
        assert_eq!(prefill::full_chunk_ceiling(), N);
        assert!(scoped(N + 1, || panic!("invalid nested scope entered")).is_err());
        scoped(N, || assert_eq!(prefill::full_chunk_ceiling(), N)).unwrap();
        let panic = std::panic::catch_unwind(|| {
            scoped(4096, || {
                assert_eq!(prefill::full_chunk_ceiling(), 4096);
                panic!("restore nested ceiling");
            })
            .unwrap();
        });
        assert!(panic.is_err());
        assert_eq!(prefill::full_chunk_ceiling(), N);
        std::thread::spawn(|| {
            assert_eq!(prefill::full_chunk_ceiling(), 4096);
            assert!(chunk_views(0..N, N).is_err());
            scoped(N, || assert_eq!(prefill::full_chunk_ceiling(), N)).unwrap();
            assert_eq!(prefill::full_chunk_ceiling(), 4096);
        })
        .join()
        .unwrap();
        assert_eq!(prefill::full_chunk_ceiling(), N);
    })
    .unwrap();
    assert!(std::panic::catch_unwind(|| scoped(N, || panic!("outer restore"))).is_err());
    assert_eq!(prefill::full_chunk_ceiling(), 4096);
}

#[test]
fn one_8k_plan_preserves_work_retention_and_2k_decoder_replay() {
    assert!(chunk_views(0..N, N).is_err());
    assert!(ced_plan::row_view(&(0..N), 0..N).is_err());
    scoped(N, || {
        let baseline = plan(4096, N, CAPACITY);
        let candidate = plan(N, N, CAPACITY);
        assert_eq!(candidate.encoder_layers, 0..20);
        assert_eq!(
            candidate.work.encoder_output_rows,
            candidate.encoder_layers.len() * N
        );
        assert_eq!(
            candidate.encoder_layers.end + candidate.decoder.len(),
            gguf::Schedule::default().layers
        );
        assert_eq!(candidate.publication_layer, 20);
        assert_eq!(candidate.boundary_bound_rows, 2541);
        assert_eq!(candidate.boundary_rows, 5651..N);
        assert_eq!(candidate.candidate_rows, 5778..N);
        assert_eq!(candidate.work.encoder_output_rows, 163840);
        assert_eq!(candidate.work.publication_rows, N);
        assert_eq!(candidate.work.decoder_output_rows, 24150);
        assert_eq!(candidate.work.decoder_seed_rows, 2540);
        assert_eq!(candidate.work.total_output_rows, 187990);
        assert_eq!(candidate.work.total_decoder_kv_rows, 26690);
        assert_eq!(candidate.work.total_execution_rows, 198722);
        assert_eq!(candidate.work.full_forward_output_rows, 327680);
        assert_eq!(candidate.memory.candidate_i32_per_row, CAPACITY / 8);
        assert_eq!(candidate.memory, baseline.memory);
        assert_eq!(candidate.work, baseline.work);
        assert_eq!(candidate.decoder, baseline.decoder);
        let ranges = |width| {
            chunk_views(0..N, width)
                .unwrap()
                .map(|v| v.absolute_rows)
                .collect::<Vec<_>>()
        };
        assert_eq!(ranges(4096), [0..4096, 4096..N]);
        assert_eq!(ranges(N), [0..N]);
        let mut only_width = candidate.clone();
        only_width.full_chunk_rows = baseline.full_chunk_rows;
        assert_eq!(only_width, baseline);
        for n in [8191, N, 8193] {
            let p = plan(N, n, CAPACITY);
            assert_eq!(p.scalar_tail(), (n == 8193).then_some(N));
            let end = p.scalar_tail().unwrap_or(n);
            for layer in &p.decoder {
                for rows in [layer.seed_rows.clone(), layer.output_rows.start..end] {
                    let views = chunk_views(rows.clone(), prefill::DEFAULT_CHUNK_ROWS)
                        .unwrap()
                        .collect::<Vec<_>>();
                    assert!(views.iter().all(|v| v.absolute_rows.len() <= 2048));
                    assert_eq!(
                        views.iter().map(|v| v.absolute_rows.len()).sum::<usize>(),
                        rows.len()
                    );
                }
            }
            for group in p.decoder.chunks(2) {
                let units =
                    ced_plan::batches::units(&p, group[0].layer..group.last().unwrap().layer + 1)
                        .unwrap();
                assert!(units.iter().all(|unit| unit.rows.len() <= 2048));
            }
        }
        assert_eq!(
            ced_plan::row_view(&(4096..16384), 8192..16384)
                .unwrap()
                .row_offset,
            4096
        );
    })
    .unwrap();
    assert!(chunk_views(0..N, N).is_err());
}

#[test]
fn pricing_compares_8k_with_two_4k_pipeline_at_full_request_capacity() {
    scoped(N, || {
        let c = Config::default();
        let schedule = gguf::Schedule::default();
        let hash = engram::HashConfig::default();
        let projections = projections(&c, &hash);
        for capacity in [CAPACITY, 16384] {
            let mut full_costs = Vec::new();
            let mut baseline_estimates = Vec::<CedMemoryEstimate>::new();
            for width in [4096, N] {
                let p = plan(width, N, capacity);
                assert_eq!(attention_gpu::Rows::tile_widths(&c, width).unwrap(), [
                    ((c.window + width) * c.head_dim).div_ceil(width),
                    (512 * c.heads * (c.window + c.index_topk + 1)).div_ceil(width),
                    (512 * c.heads * c.head_dim).div_ceil(width),
                ], "pricing fixture requires the existing default 512-row query tile");
                let (request, full, absent) = preflight::runtime_from_parts(
                    &c, &schedule, &hash, &projections, N, capacity,
                    &[PrefillExecution::FullForward], width, true, &price,
                )
                .unwrap();
                assert!(absent.is_none());
                assert_eq!(request, memory::estimate_gpu_step(&c, capacity, &schedule, hash.layers.len()).unwrap());
                let without_half = prefill::full_scratch_budget_with(
                    &c, schedule.layers, &projections, capacity, width, false, &price,
                ).unwrap();
                let half = prefill::half_scratch_budget_with(&c, width, true, &price).unwrap();
                assert_eq!(full.0, without_half.0 + half);
                assert_eq!(full.1, without_half.1);
                assert_eq!(full.1, prefill::upload_cpu_bytes(&c, &projections, width).unwrap()
                    + prefill::chunk_hash_cpu_bytes(hash.layers.len(), width).unwrap());
                full_costs.push(full);
                for half_selected in [false, true] {
                    let mut estimates = Vec::new();
                    for pipelined in [false, true] {
                        let ced = ced::estimate_from_parts(
                            &c, schedule.layers, &projections, hash.layers.len(), capacity,
                            &p, half_selected, pipelined, &price,
                        ).unwrap();
                        assert_eq!(ced.decoder_batch_layers, 2);
                        assert_eq!(ced.decoder_schedule_reason, None);
                        if width == 4096 {
                            baseline_estimates.push(ced.clone());
                        } else {
                            let a = &baseline_estimates[usize::from(half_selected) * 2 + usize::from(pipelined)];
                            let common = |e: &CedMemoryEstimate| [
                                e.retained_residual_bytes, e.retained_mix_bytes,
                                e.retained_selected_bytes, e.retained_candidate_bytes,
                                e.priced_retained_bytes, e.priced_seed_bytes,
                                e.priced_batch_status_bytes, e.batch_cpu_bytes,
                                e.priced_batch_diagnostic_bytes, e.batch_encoding_reserve_bytes,
                                e.command_reserve_bytes, e.retained_control_cpu_bytes,
                            ];
                            assert_eq!(common(&ced), common(a), "common A/B terms changed");
                            assert_eq!((ced.decoder_batch_layers, ced.decoder_schedule_reason),
                                (a.decoder_batch_layers, a.decoder_schedule_reason));
                        }
                        let half = prefill::half_scratch_budget_with(&c, width, half_selected, &price).unwrap();
                        assert_eq!(ced.priced_half_chunk_bytes, half);
                        let overlap = pipelined && width == 4096;
                        assert_eq!(ced.priced_pipeline_chunk_bytes, if overlap {
                            ced.priced_chunk_bytes + ced.priced_publication_bytes + ced.command_reserve_bytes
                        } else { 0 });
                        assert_eq!(ced.priced_pipeline_cpu_bytes, if overlap {
                            prefill::upload_cpu_bytes(&c, &projections, width).unwrap()
                        } else { 0 });
                        let full = if half_selected { full } else { without_half };
                        let runtime = planning::runtime_memory(request, full, Some(&ced)).unwrap();
                        assert_eq!(runtime.metal_bytes, request.metal_bytes + full.0.max(ced.incremental_metal_bytes));
                        assert_eq!(runtime.cpu_bytes, request.cpu_bytes + full.1.max(ced.incremental_cpu_bytes));
                        println!("{}", serde_json::json!({
                            "fixture":"Config/Schedule/HashConfig defaults; assumed Engram projection geometry; no header or model",
                            "pricing":"assumed heap logical, alignment256, host page16384; pure upper bound plus1024 per allocation",
                            "prompt_rows":N, "capacity":capacity, "encoder_chunks":N/width,
                            "full_chunk_rows":width, "half_selected":half_selected, "pipelined":pipelined,
                            "request_metal_bytes":request.metal_bytes, "request_cpu_bytes":request.cpu_bytes,
                            "full_metal_bytes":full.0, "full_cpu_bytes":full.1, "ced":ced,
                            "runtime_full_ced_metal_bytes":runtime.metal_bytes, "runtime_full_ced_cpu_bytes":runtime.cpu_bytes
                        }));
                        estimates.push(ced);
                    }
                    if width == N {
                        assert_eq!(estimates[0], estimates[1]);
                    } else {
                        assert_eq!(estimates[1].incremental_metal_bytes,
                            estimates[0].incremental_metal_bytes + estimates[1].priced_pipeline_chunk_bytes);
                        assert_eq!(estimates[1].incremental_cpu_bytes,
                            estimates[0].incremental_cpu_bytes + estimates[1].priced_pipeline_cpu_bytes);
                    }
                }
            }
            assert!(full_costs[1].0 > full_costs[0].0);
            assert!(full_costs[1].1 > full_costs[0].1);
        }
        let at_prompt = memory::estimate_gpu_step(&c, N, &schedule, hash.layers.len()).unwrap();
        let at_capacity = memory::estimate_gpu_step(&c, CAPACITY, &schedule, hash.layers.len()).unwrap();
        assert!(at_capacity.metal_bytes > at_prompt.metal_bytes);
        assert!(plan(N, N, CAPACITY).memory.candidate_bytes > plan(N, N, N).memory.candidate_bytes);
    }).unwrap();
}

#[test]
fn prices_are_monotone_conservative_and_reject_max_buffer_and_overflow() {
    scoped(N, || {
        let c = Config::default();
        let schedule = gguf::Schedule::default();
        let hash = engram::HashConfig::default();
        let projections = projections(&c, &hash);
        let full = |capacity, price: &dyn Fn(u64) -> Result<u64>| {
            prefill::full_scratch_budget_with(
                &c,
                schedule.layers,
                &projections,
                capacity,
                N,
                true,
                &|b| price(b),
            )
        };
        let logical = full(CAPACITY, &Ok).unwrap();
        let upper = full(CAPACITY, &price).unwrap();
        let dearer = full(CAPACITY, &|b| price_with(b, u64::MAX, 32768)).unwrap();
        assert!(upper.0 >= logical.0);
        assert!(dearer.0 >= upper.0);
        assert_eq!(logical.1, upper.1);
        assert!(full(16384, &price).unwrap().0 >= upper.0);
        let largest =
            gpu_step::chunk_widths_from_geometry(&c, schedule.layers, &projections, CAPACITY, N)
                .unwrap()
                .into_iter()
                .map(|w| w as u64 * N as u64 * 4)
                .max()
                .unwrap();
        assert!(full(CAPACITY, &|b| price_with(b, largest - 1, 1024)).is_err());
        assert!(full(CAPACITY, &|_| Ok(u64::MAX)).is_err());
        assert!(price(u64::MAX).is_err());
        assert!(price_with(u64::MAX, u64::MAX, 0).is_err());
        assert!(prefill::half_scratch_budget_with(&c, usize::MAX, true, &price).is_err());
        assert!(prefill::upload_cpu_bytes(&c, &projections, usize::MAX).is_err());
        assert!(prefill::chunk_hash_cpu_bytes(usize::MAX, N).is_err());
        let p = plan(N, N, CAPACITY);
        assert!(
            ced::estimate_from_parts(
                &c,
                schedule.layers,
                &projections,
                hash.layers.len(),
                CAPACITY,
                &p,
                true,
                true,
                &|_| Ok(u64::MAX)
            )
            .is_err()
        );
        let request =
            memory::estimate_gpu_step(&c, CAPACITY, &schedule, hash.layers.len()).unwrap();
        assert!(planning::runtime_memory(request, (u64::MAX, 0), None).is_err());
        assert!(planning::runtime_memory(request, (0, u64::MAX), None).is_err());
    })
    .unwrap();
}

#[test]
fn model_shaped_extents_fit_existing_integer_domains_without_kernel_changes() {
    scoped(N, || {
        let c = Config::default();
        let schedule = gguf::Schedule::default();
        let hash = engram::HashConfig::default();
        let projections = projections(&c, &hash);
        let ffn = ffn_gpu::Rows::widths(&c, schedule.layers)
            .into_iter()
            .map(|w| w.checked_mul(N).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(N * c.topk, 49152);
        for extent in [
            N * c.experts,
            N * c.topk * c.intermediate,
            N * c.topk * c.hidden,
        ] {
            assert!(ffn.contains(&extent));
            assert!(u32::try_from(extent).is_ok());
            assert!(extent.checked_mul(4).is_some());
        }
        assert_eq!(N * c.experts, 3145728);
        assert_eq!(N * c.topk * c.intermediate, 113246208);
        assert_eq!(N * c.topk * c.hidden, 251658240);
        for rows in [4096, N] {
            let widths = gpu_step::chunk_widths_from_geometry(
                &c,
                schedule.layers,
                &projections,
                CAPACITY,
                rows,
            )
            .unwrap();
            let largest = widths
                .into_iter()
                .map(|w| w.checked_mul(rows).unwrap())
                .max()
                .unwrap();
            assert_eq!(largest, rows * c.heads * c.head_dim);
            assert!(largest <= u32::MAX as usize / 4);
            assert_eq!(largest * 4, if rows == N { 1 << 30 } else { 1 << 29 });
        }
        assert_eq!(
            prefill::upload_cpu_bytes(&c, &projections, 4096).unwrap() * 2,
            prefill::upload_cpu_bytes(&c, &projections, N).unwrap()
        );
        assert_eq!(
            prefill::chunk_hash_cpu_bytes(hash.layers.len(), 4096).unwrap() * 2,
            prefill::chunk_hash_cpu_bytes(hash.layers.len(), N).unwrap()
        );
    })
    .unwrap();
}

#[test]
fn conditional_resident_headroom_and_runtime_gates_use_cpu_plus_metal() {
    scoped(N, || {
        let c = Config::default();
        let schedule = gguf::Schedule::default();
        let hash = engram::HashConfig::default();
        let projections = projections(&c, &hash);
        for width in [4096, N] {
            let request = memory::estimate_gpu_step(&c, CAPACITY, &schedule, hash.layers.len()).unwrap();
            let full = prefill::full_scratch_budget_with(
                &c, schedule.layers, &projections, CAPACITY, width, true, &price,
            ).unwrap();
            let ced = ced::estimate_from_parts(
                &c, schedule.layers, &projections, hash.layers.len(), CAPACITY,
                &plan(width, N, CAPACITY), true, true, &price,
            ).unwrap();
            let runtime = planning::runtime_memory(request, full, Some(&ced)).unwrap();
            let physical = 1u64 << 40;
            let device_budget = 1u64 << 39;
            let maximum_resident = device_budget.checked_sub(runtime.metal_bytes).unwrap()
                .min(physical.checked_sub(runtime.total_bytes).unwrap());
            let mut signals = metal::MetalMemorySignals {
                recommended_max_bytes: device_budget,
                current_allocated_bytes: maximum_resident,
                process_limit_remaining_bytes: Some(0),
                host_budget: Some(metal::HostMemoryBudget {
                    physical_bytes: physical, conservative_available_bytes: physical,
                    external_gap_bytes: 0, private_shadow_bytes: 0,
                }),
            };
            let gate = |s| metal::evaluate_metal_memory_capacity_with_cpu_bytes(
                runtime.metal_bytes, runtime.cpu_bytes, s,
            );
            assert!(gate(signals).is_ok());
            assert!(metal::evaluate_metal_memory_admission_with_cpu_bytes(
                runtime.metal_bytes, runtime.cpu_bytes, 0, signals, true,
            ).is_admitted());
            signals.current_allocated_bytes += 1;
            assert!(gate(signals).is_err());
            if width == N {
                let fixed = |s, calls: &mut Vec<usize>| planning::select(
                    4096, Some(N), N, planning::Cost::default(), s,
                    |rows| {
                        calls.push(rows);
                        Ok(planning::Cost { metal: runtime.metal_bytes, cpu: runtime.cpu_bytes })
                    },
                );
                let mut calls = Vec::new();
                assert!(fixed(signals, &mut calls).is_err());
                assert_eq!(calls, [N], "fixed 8K pressure must not retry 4K/2K");
                signals.current_allocated_bytes -= 1;
                calls.clear();
                let (selected, cost) = fixed(signals, &mut calls).unwrap();
                assert_eq!(selected, N);
                assert_eq!(cost, planning::Cost { metal: runtime.metal_bytes, cpu: runtime.cpu_bytes });
                assert_eq!(calls, [N], "fixed 8K success prices exactly once");
            }
            signals.recommended_max_bytes = physical;
            signals.current_allocated_bytes = physical - runtime.total_bytes;
            assert!(gate(signals).is_ok());
            signals.current_allocated_bytes += 1;
            assert!(gate(signals).is_err());
            signals.current_allocated_bytes = 0;
            signals.host_budget.as_mut().unwrap().conservative_available_bytes = runtime.cpu_bytes - 1;
            assert!(gate(signals).is_ok());
            assert!(!metal::evaluate_metal_memory_admission_with_cpu_bytes(
                runtime.metal_bytes, runtime.cpu_bytes, 0, signals, true,
            ).is_admitted());
            signals.host_budget = None;
            assert!(gate(signals).is_err());
            println!("{}", serde_json::json!({
                "fixture":"synthetic capacity boundary, not actual device or model admission",
                "full_chunk_rows":width, "capacity":CAPACITY,
                "assumed_physical_bytes":physical, "assumed_device_budget_bytes":device_budget,
                "conditional_max_resident_metal_bytes":maximum_resident,
                "excludes":"cold loader, binding, caller, external gap and private shadow demands; require real header/plan later"
            }));
        }
    }).unwrap();
}

#[test]
fn scope_does_not_admit_invalid_requests_or_change_defaults_and_execution_guards() {
    let c = Config::default();
    let schedule = gguf::Schedule::default();
    let hash = engram::HashConfig::default();
    let projections = projections(&c, &hash);
    let full = |c: &Config, capacity, width| {
        preflight::runtime_from_parts(
            c,
            &schedule,
            &hash,
            &projections,
            N,
            capacity,
            &[PrefillExecution::FullForward],
            width,
            true,
            &price,
        )
    };
    assert!(full(&c, CAPACITY, N).is_err());
    assert!(
        CedPlan::shrinking(
            &c,
            &schedule,
            &hash,
            &CedRequest::cold(N, CAPACITY).with_full_chunk_rows(N)
        )
        .is_err()
    );
    scoped(N, || {
        for width in [0, 4097, 8191, 8193, usize::MAX] {
            assert!(full(&c, CAPACITY, width).is_err());
            assert!(
                CedPlan::shrinking(
                    &c,
                    &schedule,
                    &hash,
                    &CedRequest::cold(N, CAPACITY).with_full_chunk_rows(width)
                )
                .is_err()
            );
        }
        for capacity in [0, N - 1, hash.max_position as usize + 1, usize::MAX] {
            assert!(full(&c, capacity, N).is_err());
        }
        for c in [
            Config {
                topk: 17,
                ..c.clone()
            },
            Config {
                window: 0,
                ..c.clone()
            },
            Config {
                heads: 17,
                ..c.clone()
            },
            Config {
                hidden: usize::MAX,
                ..c.clone()
            },
        ] {
            assert!(full(&c, CAPACITY, N).is_err());
        }
        let mut bad = CedRequest::cold(N, CAPACITY).with_full_chunk_rows(N);
        bad.start_position = 1;
        assert!(CedPlan::shrinking(&c, &schedule, &hash, &bad).is_err());
        let mut overflowing = c.clone();
        overflowing.hidden = usize::MAX;
        assert!(
            CedPlan::shrinking(
                &overflowing,
                &schedule,
                &hash,
                &CedRequest::cold(N, CAPACITY).with_full_chunk_rows(N)
            )
            .is_err()
        );
        assert!(chunk_views(0..N, N + 1).is_err());
        assert!(ced_plan::row_view(&(0..N), 1..N + 1).is_err());
        let near_max = chunk_views(usize::MAX - N..usize::MAX, N)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(near_max.len(), 1);
        assert_eq!(near_max[0].absolute_rows, usize::MAX - N..usize::MAX);
        assert_eq!(
            (prefill::DEFAULT_CHUNK_ROWS, prefill::MAX_CHUNK_ROWS),
            (2048, 4096)
        );
        assert_eq!(CedRequest::cold(N, CAPACITY).full_chunk_rows, 2048);
        assert_eq!(prefill_rows_for_resident(240 << 30), 4096);
        assert_eq!(prefill_rows_for_resident((240 << 30) + 1), 2048);
        assert_eq!(
            planning::candidates(4096, None).collect::<Vec<_>>(),
            [4096, 2048]
        );
        assert_eq!(planning::candidates(4096, Some(N)).collect::<Vec<_>>(), [N]);
        assert!(prefill::checked_scratch_rows(N, 1).is_err());
        assert!(attention_gpu::PublicationRows::shapes(&c, N).is_err());
        let source = include_str!("../prefill.rs");
        for entry in ["fn run_hybrid(", "fn run_captured("] {
            assert!(
                source
                    .split(entry)
                    .nth(1)
                    .unwrap()
                    .split("invalid prefill chunk rows")
                    .next()
                    .unwrap()
                    .contains("(1..=MAX_CHUNK_ROWS).contains(&chunk_rows)")
            );
        }
        let ced_source = include_str!("../attention_gpu/ced.rs");
        assert!(
            ced_source.contains("(1..=prefill::MAX_CHUNK_ROWS).contains(&plan.full_chunk_rows)")
        );
        assert!(
            ced_source
                .split("fn valid_chunk(")
                .nth(1)
                .unwrap()
                .split("CED chunk rows")
                .next()
                .unwrap()
                .contains("(1..=prefill::MAX_CHUNK_ROWS).contains(&rows)")
        );
    })
    .unwrap();
    assert!(full(&c, CAPACITY, N).is_err());
}
