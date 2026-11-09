# Ignored-test cleanup, batch 3 — 2026-11-09

Fixed base: `a954568650492fe43e66bd200a3a9b89789caf87` (landed batches 1–2). Branch: `chore/ignored-sweep-3`. Main is not chased during this batch.

Retire **19 ignored tests** (eight model/artifact cases, three other-GPU cases and eight recorded timing drivers), plus **four ordinary tests** exclusive to the retired experiments/protocol helpers. Remove the neutral GLM scalar8 selector/TLS and kernel instantiation, rejected selected-value MMA variant and shader, native research bindings/banks, their exclusive helpers and obsolete printouts. Shared primitives and shipped rollbacks remain.

No selected entry point has a model-witness annotation, qualification registry entry, acceptance-inventory record, maintained-script caller or new reference in current active-lane diffs. Source, current registrations/callers and active worktree diffs were checked on top of the original cx census `01a1bfba-826e-7542-8c92-6e3e2e87d090`. Frozen dated packet data and all prior PERF-LOG entries remain unchanged.

The 889-entry reviewed inventory re-renders unchanged and is checked by exact compiled discovery. ENV is regenerated. Two retained ordinary tests are adjusted only for the retired scalar8 kernel count and the removed MMA launch metadata; shared decoder/F64 and GQA12 assertions remain. Other retained test bodies and all 129 witness annotations remain unchanged.

The source census falls **815 -> 796 ignore sites**, with **2,468 net Rust lines removed**. There are 172 byte-identical retained test bodies/attributes in touched files, plus the two explicit retired-kernel metadata updates described above.

## Decisions

| Family | Tests removed | Evidence / reason |
| --- | ---: | --- |
| Artifact inventory printout | 2 | dense_iq_role_storage_* tests; descriptor plan/identity unit tests; qwen4exp_metadata and packed MoE shape/quant witnesses. old/native dense-IQ plan report prints totals and checks source stamps; actual storage policies and model parity witnesses retained; print-only tensor dtype census without an assertion; typed live-artifact binding and numeric witnesses retained. |
| Duplicate GQA12 prototype gate | 1 | selected_value_product_correctness and selected_value_split_product_matches_f64 in same file. product gate runs the identical 18 fixture cases and checks prototype/product/baseline/F64; retain the stronger product gate. |
| Duplicate embedding metadata | 1 | dense_q8_embedding_residency_and_state_witness in same file. same metadata() assertions also run in retained default-on residency/gather/causal-state witness. |
| Duplicate embedding residency | 1 | native_embedding_pilot::dense_q8_embedding_residency_and_state_witness; native_quant_embedding_default_promoted tests. old allocation/size probe is covered by retained stronger default Q8 byte/gather/state witness and general selector/load tests. |
| Finite top-k prototype | 1 | PERF-LOG 2026-09-15 Parallel Singleton Routing Native Research PASS; 2026-09-16 Guarded Singleton Selector Product And CLI PASS. test-only finite selector substitution superseded by shipped guarded top-k and its compatibility/default closure. |
| GLM latent characterizations | 5 | PERF-LOG 2026-10-11 Grouped Native Latent Matrix Execution; docs/bench/2026-10-11-glm-latent-mma/README.md. recorded Half/F32/production A/B and precision packet, including the failed forced-Half experiment; shipped primitive/F64 and model quality checks remain. |
| GLM scalar8 prototype | 2 | PERF-LOG 2026-10-11 scalar8 ratios 0.996/1.006/1.001/1.002; test-only host selection. exclusive tests for neutral, unshipped scalar8 kernel and TLS override; scalar4/MMA decoder/F64 guards retained. |
| Host memory characterization | 1 | metal_memory_probes_are_available_on_product_host and admission decision tests. hard-coded one-host working-set/process-budget values; portable product-host probe and safety admission checks retained. |
| Newly ignored load timing | 1 | lease fix 08b8ddc7d; public native loading/benchmark load telemetry. print-only model-scale load stopwatch, no maintained caller or behavior assertion; default loader witnesses remain. |
| Render-boundary diagnostic | 1 | docs/bench/2026-09-06-consumed-boundary-reuse/RESULT.md. canonical re-tokenization diagnostic explicitly cannot witness actual emitted IDs/checkpoints; stronger recorded_completed_checkpoint_witness retained. |
| Selected-value research | 6 | PERF-LOG 2026-10-22 Flash Chronological Value Reuse; negative MMA KL 0.0244; PERF-LOG 2026-10-22 Flash Chronological Value Reuse; docs/bench/2026-10-22-flash-selected-value/README.md. MMA failed model quality and was never a product route; recorded timing wrappers retired; product/F64 and split-value checks retained; recorded parent/ABBA drivers and exclusive distribution helper test; shipped cancellation/retry, product/F64 and boundary gates retained. |
| Validation ceiling | 1 | PERF-LOG 2026-09-16 child attribution, validation-ceiling40224. 0.1718545ms misses 1ms design gate; recorded no-commit ceiling, not a correctness witness. |

## Exact deletion ledger

Names refer to the fixed base. Frozen source is the reproduction archive for retired probes.

| Source | Entry point | Census kind |
| --- | --- | --- |
| `crates/qwen-cli/src/serve/restored_tail_pilot.rs` | `recorded_render_boundary_diagnostic` | M |
| `crates/qwen-llm/src/glm5_next_metal/tests/latent_mma_ab.rs` | `grouped_latent_mma_ab` | P |
| `crates/qwen-llm/src/glm5_next_metal/tests/latent_mma_ab.rs` | `grouped_latent_precision_probe` | M |
| `crates/qwen-llm/src/glm5_next_metal/tests/latent_mma_ab.rs` | `grouped_latent_production_ab` | P |
| `crates/qwen-llm/src/glm5_next_metal/tests/latent_mma_ab.rs` | `grouped_latent_scalar8_ab` | P |
| `crates/qwen-llm/src/glm5_next_metal/tests/latent_mma_ab.rs` | `precision_probe_metrics_follow_prefill_width` | ordinary |
| `crates/qwen-llm/src/metal/context.rs` | `metal_memory_probes_match_local_m4_max` | G |
| `crates/qwen-llm/src/metal/latent/grouped_native_tests.rs` | `grouped_q6_scalar8_policy_is_bitwise_scalar4_at_real_shapes` | ordinary |
| `crates/qwen-llm/src/metal/latent/grouped_native_tests.rs` | `grouped_scalar8_override_is_nested_unwind_safe_and_thread_local` | ordinary |
| `crates/qwen-llm/src/metal_forward/native_embedding_pilot.rs` | `dense_q8_embedding_metadata_witness` | M |
| `crates/qwen-llm/src/metal_forward/tests.rs` | `metal_27b_byte_ledger` | M |
| `crates/qwen-llm/src/metal_forward/tests.rs` | `quantized_embedding_residency_load_probe` | M |
| `crates/qwen-llm/src/qwen4exp_moe_down_index_probe.rs` | `artifact_packed_expert_dtype_inventory` | M |
| `crates/qwen-llm/src/qwen4exp_qsa/value_mma_probe.rs` | `selected_value_gqa12_correctness` | G |
| `crates/qwen-llm/src/qwen4exp_qsa/value_mma_probe.rs` | `selected_value_gqa12_timing_n32` | P |
| `crates/qwen-llm/src/qwen4exp_qsa/value_mma_probe.rs` | `selected_value_mma_correctness` | G |
| `crates/qwen-llm/src/qwen4exp_qsa/value_mma_probe.rs` | `selected_value_mma_timing_n32` | P |
| `crates/qwen-llm/src/qwen4exp_runtime/selected_value_probe.rs` | `distribution_screen_checks` | ordinary |
| `crates/qwen-llm/src/qwen4exp_runtime/selected_value_probe.rs` | `ordinary_selected_parent_cost` | P |
| `crates/qwen-llm/src/qwen4exp_runtime/selected_value_probe.rs` | `ordinary_selected_value_abba` | P |
| `crates/qwen-llm/src/qwen4exp_runtime/tests/topk_native.rs` | `native_parallel_topk_shared_prefix` | M |
| `crates/qwen-llm/src/qwen4exp_runtime/tests/validation_ceiling.rs` | `scalar_validation_cpu_ceiling` | M |
| `crates/qwen-llm/tests/load_wall.rs` | `a3b_load_wall_ms` | P |

## Lease-fix review: all 13 tests

One deletion and twelve runner-migration candidates. The eight retained model readers preserve actual artifact/loader/numerical/safety checks. The four V4.1 timing probes have explicit inventory protection. This batch lists migration work and does not introduce a runner or change their annotation/lease requirements.

| Entry point | Decision | Reason and runner needs |
| --- | --- | --- |
| `cpu_k2_chat_artifact_structure_and_native_tokens` | MIGRATE | Real published K2 tokenizer/template/profile check; synthetic render fixtures do not bind the actual artifact. K2 Horizon 7B Q8_0 or Q4_K_M headers/tokenizer; CPU. |
| `fewrow_gen_real_weights_logit_and_margin` | MIGRATE | Real quantized K/V projection/logit-margin check for the supported few-row operator; keep its actual-weight coverage. 27B Q4_K_M artifact and Metal. |
| `load_plan_matches_previous_schedules_when_fixtures_are_present` | MIGRATE | Actual descriptor schedules, byte balance and loader admission planning; retain public-loader policy coverage. A3B/dense27B/A10B GGUF headers; optional Metal pricing. |
| `gguf_parallel_copied_a3b_q4_is_bit_exact` | MIGRATE | Complete loaded bytes, prefill/decode logits and persistent state against serial copy; preserve supported loader choices. A3B or dense27B Q4 fixture; canonical Metal lease. |
| `gguf_parallel_pread_a3b_q4_is_bit_exact` | MIGRATE | Complete loaded bytes, prefill/decode logits and persistent state against serial copy; preserve supported loader choices. A3B or dense27B Q4 fixture; canonical Metal lease. |
| `gguf_parallel_copied_dense27b_q4_is_bit_exact` | MIGRATE | Complete loaded bytes, prefill/decode logits and persistent state against serial copy; preserve supported loader choices. A3B or dense27B Q4 fixture; canonical Metal lease. |
| `gguf_parallel_pread_dense27b_q4_is_bit_exact` | MIGRATE | Complete loaded bytes, prefill/decode logits and persistent state against serial copy; preserve supported loader choices. A3B or dense27B Q4 fixture; canonical Metal lease. |
| `public_metal_loaders_refuse_an_unsafe_half_weight` | MIGRATE | Only direct public-loader regression for an unsafe half-conversion weight; safety check remains. Small Q8 artifact copied/mutated by witness; canonical Metal lease. |
| `a3b_load_wall_ms` | DELETE | Print-only load stopwatch, no functional assertion or maintained caller. No runner migration. |
| `select_kernels_timing` | MIGRATE | Explicitly protected timing entry in acceptance_inventory.json; leave available for the V4.1 lane. Existing exact inventory selector, timing env and canonical Metal lease; M5 where required. |
| `mma_sink_tile_dispatch_timing` | MIGRATE | Explicitly protected timing entry in acceptance_inventory.json; leave available for the V4.1 lane. Existing exact inventory selector, timing env and canonical Metal lease; M5 where required. |
| `flash_decode_kill_test_timing` | MIGRATE | Explicitly protected timing entry in acceptance_inventory.json; leave available for the V4.1 lane. Existing exact inventory selector, timing env and canonical Metal lease; M5 where required. |
| `hc_rows_kernel_timing` | MIGRATE | Explicitly protected timing entry in acceptance_inventory.json; leave available for the V4.1 lane. Existing exact inventory selector, timing env and canonical Metal lease; M5 where required. |

## Retained rather than inferred obsolete

- The largest remaining DS4 position/boundary/snapshot suite checks live causal state and continuation. Absence of registration is not evidence that those unique model boundaries are redundant. Keep for runner migration.
- Current GLM numerical/quality cohorts, grouped native scalar4/MMA decoded-F64 gates, BF16/F16 paths, default embedding/MTP banks and safety checks remain. The removed GLM file is the frozen Oct-11 timing/one-prompt characterization packet, not the current model quality runner.
- `selected_value_product_correctness` runs every ordinary/masked/invalid-ID/metadata/cancellation fixture from the removed GQA12 prototype gate, plus product-kernel/F64 checks. It and `selected_value_split_product_matches_f64` remain; shared stats/GQA12 reference shader bodies are unchanged. `product_selected_cancellation_retry` and the active-state hash helper remain.
- Guarded top-k compatibility, final default closure, router capture and current decode/serving tests remain. The removed finite-only native substitution predates the actual guarded product and has no nonfinite fallback. Its shared production selector primitive remains.
- Dense-IQ descriptor comparison planning remains because both role/storage tests use it. The old byte-ledger printout is removed without deleting that shared planner. The stronger native-Q8 embedding gather/residency/causal-state witness remains.
- Ordinary product-host memory signal availability and admission safety decision tests remain; only the hard-coded M4 host-value characterization is removed.
- Current split-decode work was re-promoted after the old September retirement. Its current primitive/model/product checks remain; the historical KILL alone is not used to delete reopened shipped coverage.
- Active FN, GLM, V4.1 and lens work, all registered witnesses, and the twelve newly ignored migration candidates above remain.

## Validation and build-time method

Use the fixed base and final committed tree. Build release lib and CLI test targets, check diagnostics-enabled library/integration targets, run formatting and `scripts/check-contracts.sh`. Compiled-name differences must exactly match this deletion ledger. No GPU test, inference, GPU timing, custom build lock or Metal lease is used.

The initial base library build was a Cargo cache hit (0.46s), so it is excluded from the comparison. Both reported before/after measurements remove only the two worktree-local release package artifact sets (`cargo clean --release -p qwen-llm -p qwen-cli`), keep dependencies warm, and invoke the Rustup Cargo launcher directly without the action-cache wrapper. Commands are `cargo test --offline --locked --release -p qwen-llm --lib --no-run` then `cargo test --offline --locked --release -p qwen-cli --no-run`. Cleanup is outside the timed intervals. These are single unlocked shared-machine observations, not isolated causal estimates.

Local receipts and exact elapsed times: `bench/results/ignored-sweep-batch3/{baseline.json,after.json,final.json}`. The base source is validated before editing; all final checks use the committed fixed-base branch.
