use super::*;
use crate::deepseek_v41::gguf;
use ced_plan::{CedPlan, CedRequest, chunk_views};

const EDGES: [usize; 6] = [2047, 2048, 2049, 4095, 4096, 4097];

#[test]
fn released_shape_prices_actual_4k_encoder_and_half_scratch_per_allocation() {
    let c = Config::default();
    let s = gguf::Schedule::default();
    let h = engram::HashConfig::default();
    let projections = vec![(6144, 5 * c.hidden); h.layers.len()];
    let price = |bytes: u64| -> Result<u64> {
        Ok(if bytes == 0 {
            0
        } else {
            bytes.div_ceil(16384) * 16384 + 1024
        })
    };
    let mut costs = Vec::new();
    for width in [2048, 4096] {
        let (_, full, ced) = preflight::runtime_from_parts(
            &c,
            &s,
            &h,
            &projections,
            8192,
            8448,
            &[
                PrefillExecution::FullForward,
                PrefillExecution::ShrinkingCed,
            ],
            width,
            true,
            &price,
        )
        .unwrap();
        let ced = ced.unwrap();
        let half = prefill::half_scratch_budget_with(&c, width, true, &price).unwrap();
        assert_eq!(ced.priced_half_chunk_bytes, half);
        println!(
            "{}",
            serde_json::json!({"geometry":"Config::default; no weights loaded","per_allocation_price":"ceil_16KiB_plus_1KiB_nonempty","full_chunk_rows":width,"full_metal_bytes":full.0,"full_cpu_bytes":full.1,"ced":ced})
        );
        costs.push((full, half));
    }
    assert!(costs[1].0.0 > costs[0].0.0);
    assert!(costs[1].0.1 > costs[0].0.1);
    assert!(costs[1].1 > costs[0].1);
}

#[test]
fn selected_full_geometry_keeps_encoder_schedule_and_scalar_native_identity() {
    for width in [prefill::DEFAULT_CHUNK_ROWS, prefill::MAX_CHUNK_ROWS] {
        for n in EDGES {
            let plan = CedPlan::shrinking(
                &Config::default(),
                &gguf::Schedule::default(),
                &engram::HashConfig::default(),
                &CedRequest::cold(n, n).with_full_chunk_rows(width),
            )
            .unwrap();
            assert_eq!(plan.full_chunk_rows, width);
            let chunks: Vec<_> = chunk_views(0..n, plan.full_chunk_rows)
                .unwrap()
                .map(|c| c.absolute_rows)
                .collect();
            let expected: Vec<_> = (0..n)
                .step_by(width)
                .map(|start| start..(start + width).min(n))
                .collect();
            assert_eq!(chunks, expected);
            assert_eq!(plan.scalar_tail(), (n % width == 1).then_some(n - 1));
            for layer in &plan.decoder {
                let end = plan.scalar_tail().unwrap_or(n);
                let replay: Vec<_> =
                    chunk_views(layer.output_rows.start..end, prefill::DEFAULT_CHUNK_ROWS)
                        .unwrap()
                        .collect();
                assert!(replay.iter().all(|c| c.absolute_rows.len() <= 2048));
                assert_eq!(
                    replay.iter().map(|c| c.absolute_rows.len()).sum::<usize>()
                        + usize::from(plan.scalar_tail().is_some()),
                    layer.output_rows.len()
                );
            }
            if n == 2049 && width == 4096 {
                assert_eq!(chunks, [0..2049]);
                assert_eq!(plan.scalar_tail(), None);
                assert!(prefill::RowsArithmetic::Packed.packed_operands(1));
            }
        }
    }
    for width in [0, 1, 2049, 4097, usize::MAX] {
        assert!(
            CedPlan::shrinking(
                &Config::default(),
                &gguf::Schedule::default(),
                &engram::HashConfig::default(),
                &CedRequest::cold(4097, 4097).with_full_chunk_rows(width)
            )
            .is_err()
        );
    }
}

#[test]
fn full_and_ced_metadata_price_the_selected_runtime_width_not_a_fixed_2k_cap() {
    let c = toy::config();
    let schedule = gguf::Schedule::default();
    let hash = engram::HashConfig::default();
    let projections = vec![(6144, 5 * c.hidden); hash.layers.len()];
    let price = |bytes: u64| -> Result<u64> {
        Ok(if bytes == 0 {
            0
        } else {
            bytes.div_ceil(16384) * 16384 + 1024
        })
    };
    for half in [false, true] {
        for n in EDGES {
            let mut full_costs = Vec::new();
            for width in [2048, 4096] {
                let (_, full, estimate) = preflight::runtime_from_parts(
                    &c,
                    &schedule,
                    &hash,
                    &projections,
                    n,
                    n,
                    &[
                        PrefillExecution::FullForward,
                        PrefillExecution::ShrinkingCed,
                    ],
                    width,
                    half,
                    &price,
                )
                .unwrap();
                let rows = n.min(width);
                let runtime = prefill::full_scratch_budget_with(
                    &c,
                    schedule.layers,
                    &projections,
                    n,
                    rows,
                    half,
                    &price,
                )
                .unwrap();
                assert_eq!(full, runtime);
                let plan = CedPlan::shrinking(
                    &c,
                    &schedule,
                    &hash,
                    &CedRequest::cold(n, n).with_full_chunk_rows(width),
                )
                .unwrap();
                let ced = estimate.unwrap();
                assert_eq!(ced.full_chunk_rows, width);
                assert_eq!(
                    ced,
                    ced::estimate_from_parts(
                        &c,
                        schedule.layers,
                        &projections,
                        hash.layers.len(),
                        n,
                        &plan,
                        half,
                        gpu_step::ced::encoder_pipeline_priced(),
                        &price
                    )
                    .unwrap()
                );
                assert_eq!(
                    ced.chunk_hash_cpu_bytes,
                    prefill::chunk_hash_cpu_bytes(hash.layers.len(), rows).unwrap()
                );
                assert_eq!(
                    ced.priced_half_chunk_bytes,
                    prefill::half_scratch_budget_with(&c, rows, half, &price).unwrap()
                );
                assert_eq!(
                    full.1,
                    prefill::upload_cpu_bytes(&c, &projections, rows).unwrap()
                        + prefill::chunk_hash_cpu_bytes(hash.layers.len(), rows).unwrap()
                );
                full_costs.push(full);
            }
            if n > 2048 {
                assert!(
                    full_costs[1].0 > full_costs[0].0,
                    "4K Metal scratch underpriced at N={n}"
                );
                assert!(
                    full_costs[1].1 > full_costs[0].1,
                    "4K CPU uploads underpriced at N={n}"
                );
            } else {
                assert_eq!(full_costs[0], full_costs[1]);
            }
        }
    }
    assert_eq!(prefill_rows_for_resident(240 << 30), 4096);
    assert_eq!(prefill_rows_for_resident((240 << 30) + 1), 2048);
}
