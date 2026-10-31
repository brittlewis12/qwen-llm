use super::*;
use crate::deepseek_v41::gguf::{self, ResidentWeights, tests::admitted_request};
use objc2::rc::{Weak, autoreleasepool};

const TOKENS: usize = admitted_request::REQUEST_TOKENS;
const CAPACITY: usize = TOKENS + 1;
const FIXTURE_CPU: u64 = 32 << 20;
// Token/logit vectors, saved causal-region bits and report serialization.
const COMPARISON_CPU: u64 = 8 << 20;
const BOUND: u64 = 512 << 20;

fn bits(values: &[f32]) -> Vec<u32> {
    assert!(values.iter().all(|v| v.is_finite()));
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "self-leased <=512 MiB combined CPU+Metal ledger; admitted 2K state with unchanged 4K model; MTL debug required; no model"]
fn admitted_2k_request_on_4k_model_executes_ced_public_api_and_matches_same_width_full() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
    ));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let resident_probes = autoreleasepool(|_| {
        let ctx = MetalContext::new().unwrap();
        Cost {
            metal: 0,
            cpu: admitted_request::RECIPE_CPU_BYTES + FIXTURE_CPU,
        }
        .admit(ctx.memory_signals())
        .unwrap();
        let (g, fixture_cpu) = admitted_request::fixture();
        assert_eq!(fixture_cpu, FIXTURE_CPU);
        let cold = preflight::plan_cold_prefill(
            &ctx,
            &g,
            TOKENS,
            CAPACITY,
            &[
                PrefillExecution::FullForward,
                PrefillExecution::ShrinkingCed,
            ],
            FIXTURE_CPU + COMPARISON_CPU,
        )
        .unwrap();
        let priced_total = cold
            .required_metal_bytes
            .checked_add(cold.required_cpu_bytes)
            .unwrap();
        eprintln!(
            "admitted_request_fixture_prices {}",
            serde_json::json!({
                "tokens": TOKENS, "capacity": CAPACITY,
                "bound_bytes": BOUND, "resident_metal_bytes": cold.load_metal_bytes,
                "private_shadow_bytes": cold.resident_storage.private_shadow_bytes,
                "binding_cpu_bytes": cold.binding_cpu_bytes,
                "caller_cpu_bytes": cold.caller_cpu_bytes,
                "runtime_metal_bytes": cold.runtime_metal_bytes,
                "runtime_cpu_bytes": cold.runtime_cpu_bytes,
                "required_metal_bytes": cold.required_metal_bytes,
                "required_cpu_bytes": cold.required_cpu_bytes,
                "total_bytes": priced_total,
            })
        );
        let p = gguf::plan(&g).unwrap();
        let half = crate::metal::moe_grouped_tensor_selected(&ctx);
        let quote = |rows| {
            quotes::runtime_quote(
                &p.config,
                &p.schedule,
                &p.engram_hash,
                &[],
                TOKENS,
                CAPACITY,
                rows,
                half,
                &|bytes| {
                    ctx.price_shared_buffer_upper(bytes)
                        .map(|p| p.priced_upper_bytes)
                        .map_err(|e| e.to_string())
                },
            )
            .unwrap()
        };
        let (quoted_wide, wide_details) = quote(4096);
        let (quoted_narrow, narrow_details) = quote(2048);
        eprintln!(
            "admitted_request_runtime_quotes {}",
            serde_json::json!({
                "schema": "v36_actual_device_width_quotes_v1",
                "wide": wide_details, "narrow": narrow_details,
                "binding_cpu_bytes": cold.binding_cpu_bytes,
                "caller_cpu_bytes": cold.caller_cpu_bytes,
                "private_shadow_bytes": cold.resident_storage.private_shadow_bytes,
            })
        );
        assert!(
            priced_total <= BOUND,
            "actual-pricer CPU+Metal fixture ledger {priced_total} exceeds {BOUND}"
        );
        assert!(
            quoted_wide.metal_bytes > quoted_narrow.metal_bytes,
            "fixture must admit genuine MWSS-induced narrowing before import"
        );
        // The complete real-signal peak is admitted before importing any weight.
        Cost {
            metal: cold.required_metal_bytes,
            cpu: cold.required_cpu_bytes,
        }
        .admit(ctx.memory_signals())
        .unwrap();
        cold.assess_load(ctx.memory_signals()).unwrap();
        let narrow_cold = cold
            .alternatives
            .iter()
            .find(|p| p.full_chunk_rows == 2048)
            .unwrap()
            .clone();
        let resident = ResidentWeights::load(&ctx, g).unwrap();
        let probes: Vec<_> = resident
            .dense
            .values()
            .chain(resident.banks.iter().map(|b| &b.tensor))
            .map(|t| Weak::from_retained(&t.buffer))
            .collect();
        let model = Forward::from_resident_with_cold_plan(
            &ctx,
            &resident,
            resident.token_map.clone(),
            resident.raw_pad,
            &cold,
        )
        .unwrap();
        assert_eq!(model.prefill_chunk_rows(), 4096);
        let (wide_price, _) = model.price_request(CAPACITY, TOKENS, 4096, true).unwrap();
        let (narrow_price, _) = model.price_request(CAPACITY, TOKENS, 2048, true).unwrap();
        assert_eq!(wide_price, quoted_wide);
        assert_eq!(narrow_price, quoted_narrow);
        assert!(wide_price.metal_bytes > narrow_price.metal_bytes);
        let mut controlled = ctx.memory_signals();
        controlled.recommended_max_bytes = controlled
            .effective_device_usage()
            .unwrap()
            .checked_add(narrow_price.metal_bytes)
            .unwrap();
        eprintln!(
            "admitted_request_selection_signals {}",
            serde_json::json!({
                "model_preference_rows": model.prefill_chunk_rows(),
                "effective_device_usage_bytes": controlled.effective_device_usage().unwrap(),
                "controlled_mwss_bytes": controlled.recommended_max_bytes,
                "wide_minus_narrow_metal_bytes": wide_price.metal_bytes - narrow_price.metal_bytes,
                "binding_and_caller_cpu_reserve_bytes": cold.binding_cpu_bytes + cold.caller_cpu_bytes,
            })
        );
        // Evaluation-only snapshots, never an in-process environment or hardware limit change.
        let mut plan = with_request_signals(vec![controlled, controlled], || {
            model.plan_request_for_executions(
                CAPACITY,
                TOKENS,
                cold.caller_cpu_bytes + cold.binding_cpu_bytes,
                &[
                    PrefillExecution::FullForward,
                    PrefillExecution::ShrinkingCed,
                ],
            )
        })
        .unwrap();
        assert_eq!(plan.prefill().selected_rows, 2048);
        assert_eq!(plan.memory(), narrow_price);
        assert_eq!(plan.admission().intent, PrefillIntent::RequiredFullCed);
        assert!(plan.admission().fallback.is_none());
        let post_bind = model.plan_request_from_cold(&narrow_cold).unwrap();
        assert_eq!(post_bind.prefill(), plan.prefill());
        assert_eq!(post_bind.memory(), plan.memory());
        assert_eq!(post_bind.binding_cpu_reserve(), cold.binding_cpu_bytes);
        let original = plan.identity.clone();
        plan.identity = Arc::new(());
        assert!(
            model.new_state_from_plan(&plan).is_err(),
            "equal geometry is not Arc identity"
        );
        plan.identity = original;
        let plan = *Box::new(plan);
        let model = *Box::new(model);
        let ids: Vec<_> = (0..TOKENS).map(|i| ((i * 7 + i / 5) % 3) as u32).collect();
        let mut reference = None;
        for execution in [
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ] {
            let state_probes = autoreleasepool(|_| {
                let mut state = model.new_state_from_plan(&plan).unwrap();
                assert_eq!(state.selected_prefill_rows(), Some(2048));
                assert_eq!(model.prefill_chunk_rows(), 4096);
                let (logits, report) = model
                    .prefill_with_execution(&mut state, &ids, None, execution)
                    .unwrap();
                assert_eq!(report.execution, execution);
                assert!(report.fallback.is_none());
                assert_eq!(
                    report.admission.as_ref().unwrap().intent,
                    PrefillIntent::RequiredFullCed
                );
                assert_eq!(report.geometry, plan.prefill());
                assert_eq!(
                    report.geometry.chunk_ranges().collect::<Vec<_>>(),
                    [0..2048, 2048..TOKENS]
                );
                assert!(state.is_ready() && !state.is_poisoned());
                assert_eq!(state.position(), TOKENS);
                assert_eq!(state.prefill_origin(), execution);
                let causal: Vec<_> = ced::tests::causal_bits(&state)
                    .iter()
                    .map(|v| bits(v))
                    .collect();
                let got = (bits(&logits), causal, state.addresses);
                if let Some(expected) = &reference {
                    assert_eq!(&got, expected);
                } else {
                    reference = Some(got);
                }
                if execution == PrefillExecution::ShrinkingCed {
                    let counts = report.counters.as_ref().unwrap();
                    assert_eq!(counts.encoder_output_rows, TOKENS);
                    assert_eq!(counts.encoded_commands, counts.completed_commands);
                    assert_eq!(report.memory.as_ref().unwrap().full_chunk_rows, 2048);
                    assert_eq!(report.memory.as_ref().unwrap().decoder_batch_layers, 2);
                }
                let probes: Vec<_> = state
                    .gpu_step
                    .as_ref()
                    .unwrap()
                    .attention()
                    .causal_regions(TOKENS)
                    .unwrap()
                    .iter()
                    .map(|(t, _)| Weak::from_retained(&t.buffer))
                    .collect();
                assert!(!probes.is_empty());
                assert!(probes.iter().all(|p| p.load().is_some()));
                drop(state);
                probes
            });
            assert!(
                state_probes.iter().all(|p| p.load().is_none()),
                "state owners survive pool drain"
            );
        }
        assert!(probes.iter().all(|p| p.load().is_some()));
        drop(model);
        drop(resident);
        probes
    });
    assert!(!resident_probes.is_empty());
    assert!(
        resident_probes.iter().all(|p| p.load().is_none()),
        "resident owners survive outer pool drain"
    );
}
