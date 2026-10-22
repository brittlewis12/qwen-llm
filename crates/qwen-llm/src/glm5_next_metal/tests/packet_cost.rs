use super::*;
use std::time::Instant;

fn bits(values: &[f32]) -> Vec<u32> {
    assert!(values.iter().all(|value| value.is_finite()));
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
#[ignore = "local GLM weights and QWEN_PACKET_TEXT; exclusive production GPU lease"]
fn native_short_packet_cost_and_stage_attribution() {
    use super::super::packed::{
        ExactStages, F32Stages, F32Tile, PacketHalfExperts, PacketPolicy, PacketQ8Fewrow, Stage,
    };
    let diagnostic =
        std::env::var("QWEN_PACKET_DIAGNOSTIC").unwrap_or_else(|_| "production".into());
    assert!(matches!(
        diagnostic.as_str(),
        "baseline"
            | "production"
            | "native_experts_router"
            | "narrow_projections"
            | "combined"
            | "q8_fewrow"
            | "native_experts_router_q8"
            | "half_experts_router"
            | "half_experts_router_q8"
    ));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let text = std::fs::read_to_string(std::env::var_os("QWEN_PACKET_TEXT").unwrap()).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|token| u32::try_from(token).unwrap())
        .collect();
    let depths: Vec<usize> = std::env::var("QWEN_PACKET_DEPTHS")
        .unwrap_or_else(|_| "64,2179,8192".into())
        .split(',')
        .map(|depth| depth.parse().unwrap())
        .collect();
    assert!(!depths.is_empty());
    let capacity = depths.iter().max().unwrap() + 9;
    assert!(tokens.len() >= capacity);
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, 512).unwrap();
    assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
    let taps = [5, 14, 24, 33, 42];
    let mut logits_bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[], 0, 8).unwrap(),
    )
    .unwrap();
    let mut capture_bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &taps, 8, 8).unwrap(),
    )
    .unwrap();
    let vocab = weights.config.vocab_size as usize;
    for depth in depths {
        // All arms start from the same legacy prefix, including its final
        // short chunk. References must not silently inherit production policy.
        let legacy = PacketPolicy::legacy();
        session.reset().unwrap();
        session.prefill_packed(&ctx, &tokens[..depth]).unwrap();
        let base = session.capture_snapshot().unwrap();
        let mut baseline = BTreeMap::new();
        for width in [1, 2, 4, 8] {
            session.restore_snapshot(&base).unwrap();
            let rows = session
                .verify_packed_with_captures(&ctx, &tokens[depth..depth + width], &mut logits_bank)
                .unwrap();
            let logits = read_f32(rows.logits().unwrap()).unwrap();
            let future = session.forward(&ctx, tokens[depth + width]).unwrap();
            baseline.insert(width, (logits, future));
        }
        drop(legacy);
        // Named experiments retain their original baseline plus substitutions;
        // production clears overrides and exercises the real default selectors.
        let _policy = if diagnostic == "production" {
            PacketPolicy::production()
        } else {
            PacketPolicy::legacy()
        };
        let _half = matches!(
            diagnostic.as_str(),
            "half_experts_router" | "half_experts_router_q8"
        )
        .then(PacketHalfExperts::set);
        let _router = matches!(
            diagnostic.as_str(),
            "half_experts_router" | "half_experts_router_q8"
        )
        .then(|| ExactStages::set(&[Stage::Router]));
        let _native = matches!(
            diagnostic.as_str(),
            "native_experts_router" | "combined" | "native_experts_router_q8"
        )
        .then(|| ExactStages::set(&[Stage::RoutedExperts, Stage::Router]));
        let _q8 = matches!(
            diagnostic.as_str(),
            "q8_fewrow" | "native_experts_router_q8" | "half_experts_router_q8"
        )
        .then(PacketQ8Fewrow::set);
        let _projections =
            matches!(diagnostic.as_str(), "narrow_projections" | "combined").then(|| {
                F32Stages::with_tiles(
                    &[
                        Stage::DenseFfn,
                        Stage::SharedExpert,
                        Stage::KdaProjection,
                        Stage::KdaExpand,
                        Stage::MlaProjection,
                        Stage::MlaAbsorb,
                        Stage::IndexerProjection,
                    ],
                    &[F32Tile::Q8_0, F32Tile::Q6K],
                )
            });
        for width in [1, 2, 4, 8] {
            let suffix = &tokens[depth..depth + width];
            let next = tokens[depth + width];
            let mut expected = BTreeMap::<&str, Vec<u32>>::new();
            let mut expected_future = BTreeMap::<&str, Vec<u32>>::new();
            // Warm each arm, then balance forward/reverse order. Restore, checks,
            // continuation witnesses and profiling are outside ordinary timings.
            for (iteration, order) in [
                ["serial", "packed_last", "verify_all", "verify_features"],
                ["serial", "packed_last", "verify_all", "verify_features"],
                ["verify_features", "verify_all", "packed_last", "serial"],
            ]
            .into_iter()
            .enumerate()
            {
                for arm in order {
                    session.restore_snapshot(&base).unwrap();
                    let start = Instant::now();
                    let logits = match arm {
                        "serial" => suffix
                            .iter()
                            .flat_map(|&token| session.forward(&ctx, token).unwrap())
                            .collect::<Vec<_>>(),
                        "packed_last" => session.prefill_packed(&ctx, suffix).unwrap(),
                        "verify_all" => {
                            let rows = session
                                .verify_packed_with_captures(&ctx, suffix, &mut logits_bank)
                                .unwrap();
                            read_f32(rows.logits().unwrap()).unwrap()
                        }
                        "verify_features" => {
                            let rows = session
                                .verify_packed_with_captures(&ctx, suffix, &mut capture_bank)
                                .unwrap();
                            read_f32(rows.logits().unwrap()).unwrap()
                        }
                        _ => unreachable!(),
                    };
                    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                    assert_eq!(session.position(), depth + width);
                    let actual = bits(&logits);
                    let future = bits(&session.forward(&ctx, next).unwrap());
                    if iteration == 0 {
                        expected.insert(arm, actual);
                        expected_future.insert(arm, future);
                    } else {
                        assert_eq!(actual, expected[arm], "{depth}/{width}/{arm}");
                        assert_eq!(future, expected_future[arm]);
                    }
                    eprintln!(
                        "packet_cost {}",
                        serde_json::json!({
                            "family": "glm5_next", "depth": depth, "width": width,
                            "diagnostic": diagnostic,
                            "arm": arm, "warmup": iteration == 0, "iteration": iteration,
                            "wall_ms": wall_ms,
                            "prefill_rows": session.packed.as_ref().map(|packed| packed.rows()),
                            "lineage": format!("{:?}", session.packed_lineage()),
                            "precision": format!("{:?}", session.effective_fast_precision()),
                        })
                    );
                }
            }
            for arm in ["verify_all", "verify_features"] {
                assert_eq!(
                    expected[arm][(width - 1) * vocab..],
                    expected["packed_last"]
                );
                assert_eq!(expected_future[arm], expected_future["packed_last"]);
            }
            assert_eq!(expected["verify_all"], expected["verify_features"]);
            if diagnostic == "baseline" {
                assert_eq!(expected["verify_all"], bits(&baseline[&width].0));
                assert_eq!(expected_future["verify_all"], bits(&baseline[&width].1));
            }
            let rows = |values: &[u32]| {
                values
                    .chunks_exact(vocab)
                    .map(|row| row.iter().copied().map(f32::from_bits).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            };
            let reference = baseline[&width]
                .0
                .chunks_exact(vocab)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            let candidate = rows(&expected["verify_all"]);
            let future = rows(&expected_future["verify_all"]);
            eprintln!(
                "packet_comparison {}",
                serde_json::json!({
                    "family": "glm5_next", "depth": depth, "width": width, "diagnostic": diagnostic,
                    "reference": "same-width legacy Fast/Half at identical legacy prefix",
                    "reference_to_candidate": Drift::measure("packet", &reference, &candidate).json(),
                    "candidate_to_reference": Drift::measure("packet reverse", &candidate, &reference).json(),
                    "continuation": Drift::measure("continuation", &[baseline[&width].1.clone()], &future).json(),
                })
            );
            session.restore_snapshot(&base).unwrap();
            let start = Instant::now();
            let (logits, report) = session.prefill_chunk_stage_profiled(&ctx, suffix).unwrap();
            let wall_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(bits(&logits), expected["packed_last"]);
            let mut stages = BTreeMap::<String, f64>::new();
            for span in report.spans {
                *stages.entry(format!("{:?}", span.stage)).or_default() += span.gpu_ms;
            }
            eprintln!(
                "packet_profile {}",
                serde_json::json!({
                    "family": "glm5_next", "depth": depth, "width": width,
                    "diagnostic": diagnostic,
                    "wall_ms": wall_ms, "gpu_ms": report.command_gpu_ms,
                    "span_sum_ms": report.span_sum_ms, "stages_ms": stages,
                    "scope": "packed_last; encoder boundaries perturb timing",
                })
            );
        }
    }
}
