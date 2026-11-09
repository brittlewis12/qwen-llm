# Ignored-test cleanup, batch 2 — 2026-11-09

Base: `bc05a78ebf325b8efa891a16aaa29c7da3d13568` (batch 1 rebased onto main `fdd087b74`). Census: cx `01a1bfba-826e-7542-8c92-6e3e2e87d090`, with current source and caller checks.

Retire **27 ignored tests** (16 originally classified model/artifact, six other-GPU and five timing) plus **four ordinary unit tests** that exercise only the retired snapshot/HC prototypes. Delete eight exclusive prototype modules, a strict-HC test binding, their private helper chains and two exclusive research shaders. Registered witnesses and shipped math, kernels, selectors, rollbacks, snapshots and safety guards are retained. Dispatcher edits remove only `cfg(test)` hooks; the normal causal-row fallback remains.

No selected test is annotated for model-witnesses.py, registered for device qualification, in the acceptance inventory or selected by a maintained script. Current active-lane source diffs do not intersect these deletions. Dated packet scripts and raw results are historical evidence and are retained; their README recipes point to frozen source. Earlier PERF-LOG entries remain byte-for-byte intact.

The reviewed acceptance inventory re-renders unchanged at 889 entries; exact compiled discovery and CPU contracts verify its names, classes and ignore states. Main’s ordinary-test lease changes and all its newly ignored model-reading/timing cases are retained. ENV is regenerated. The two identity exceptions for the retired strict-HC fixture/LCG are removed from the guard allowlist and its live-source self-check; the other artifact-identity tripwires remain.

The source census falls **842 -> 815 ignore sites**, with **6,209 net Rust lines removed**. All **289 surviving test bodies and attributes** in touched files match the base byte for byte.

## Decisions

| Family | Tests removed | Evidence and disposition |
| --- | ---: | --- |
| Artifact inventory | 1 | docs/archive/H4-MTP-2026-05-03.md; metal_mtp::tests native-bank witnesses. print-only initial MTP weight census; actual default native-bank logits/bytes/guards witnesses stay. |
| DS4 FP4 selector | 2 | PERF-LOG 2026-08-05 DeepSeek V4 Collapsed FP4 Selector KILL. diagnostics-only direct FP4-Q/K selector failed deep quality and speed gates; current F16 selector witnesses stay. |
| Duplicate snapshot spike | 1 | h2_packed_arena_snapshot_matches_cold; runtime_snapshot_tests. manual raw-byte H2.0 proof superseded by production snapshot API and alias/continuation witnesses. |
| MoE CPU grouped down | 2 | PERF-LOG 2026-05-19 Post-v0.100 MoE Preload Plateau And Search-Space Elimination. old CPU group-builder/scatter-accum alternative; no shipped selector and only prints output deltas. |
| MoE CPU hot split | 4 | PERF-LOG 2026-05-19 Fix Grouped Q4 Sentinel, Validate Fused Route, And Allowlist Hot Routed SwiGLU. historical CPU-planned split/threshold characterization; product is GPU-owned with no split buffers. |
| MoE all-N32 | 4 | PERF-LOG 2026-05-23 llama.cpp MoE Win Is Not Metal Tensor API On This Box / All-n32 Recheck. recorded all-N32 negative; production hot-expert correctness oracles stay. |
| N16 verifier | 3 | PERF-LOG 2026-09-07 N16 Long-Context Online Attention Primitive PASS; Full-Verifier Phase PASS / Product HOLD; PERF-LOG 2026-09-07 N16 Online Attention Full-Verifier Phase PASS / Product HOLD. test-only N16 primitive; installed calibrated DFlash2 is physical N8; private N16 override, outside calibrated N8 and default context cutoff. |
| Snapshot prototypes | 5 | PERF-LOG 2026-09-07 Incremental CPU Snapshot Lifecycle PASS; PERF-LOG 2026-09-07 Snapshot Lifecycle 32K PASS / Two-Cell HOLD. synthetic incremental publication is not a production path; closed-world representation prototype; escaped-alias blocker; product full-copy/alias witnesses retained; synthetic GPU snapshot implementation never used by product. |
| Strict HC projection | 5 | PERF-LOG 2026-10-22 Flash Strict HC Projection: Kernel Gain Does Not Survive Native Prefill. no useful native gain; closed test-only kernel and injection; closed negative native timing; shared state() helper is retained for selected-value product witnesses. |
| Tiled V transpose | 4 | docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/README.md; PERF-LOG 2026-09-07 Exact Tiled VT Restored Phase HOLD / Bitwise PASS. print-only environment companion for a recorded closed transpose experiment; recorded unpromoted transpose copy experiment; default transpose coverage stays; unpromoted test-only transpose hook; charged timing gate failed. |

The H2.0 raw-byte snapshot spike is redundant with retained `h2_packed_arena_snapshot_matches_cold`, the actual production snapshot API, and `runtime_snapshot_tests` alias/capture/continuation checks. The N16, tiled-copy and synthetic snapshot implementations were never production selectors. The strict-HC four-output injection lost its native gate; its shared active-state hashing helper stays because the selected-value product witness still uses it.

## Exact deletion ledger

Names below are at the frozen base above. Reproduction must use that source tree and its original fixture/hardware/lease requirements.

| Source | Test | Census class |
| --- | --- | --- |
| `crates/qwen-llm/src/metal/attn.rs` | `attn_matrix_vt_environment_probe` | G |
| `crates/qwen-llm/src/metal/attn_verifier_online_pilot.rs` | `verifier_online_n16_attention_screen` | P |
| `crates/qwen-llm/src/metal/attn_vt_tiled_pilot.rs` | `vt_tiled_layer_bank_screen` | P |
| `crates/qwen-llm/src/metal/attn_vt_tiled_pilot.rs` | `vt_tiled_raw_bits_and_subspans` | G |
| `crates/qwen-llm/src/metal/moe/hc4_strict.rs` | `hc4_strict_cpu_contracts` | ordinary |
| `crates/qwen-llm/src/metal/moe/hc4_strict.rs` | `hc4_strict_gpu_timing` | P |
| `crates/qwen-llm/src/metal/moe/hc4_strict.rs` | `hc4_strict_matches_generic_bits` | G |
| `crates/qwen-llm/src/metal/moe/hc4_strict.rs` | `hc4_strict_rejects_invalid_bindings` | G |
| `crates/qwen-llm/src/metal_dflash/online_verify_pilot.rs` | `online_n16_fixture_identity` | M |
| `crates/qwen-llm/src/metal_dflash/online_verify_pilot.rs` | `online_n16_full_verifier_witness` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_122b_a10b_grouped_q4_hot_n32_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_122b_a10b_grouped_q4_hot_n32_threshold_scan_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_122b_a10b_grouped_q4_n32_proof_1024` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_122b_a10b_grouped_q4_n32_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_122b_a10b_grouped_routed_down_accum_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_35b_a3b_grouped_q4_hot_n32_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_35b_a3b_grouped_q4_hot_n32_threshold_scan_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_35b_a3b_grouped_q4_n32_proof_1024` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_35b_a3b_grouped_q4_n32_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tests.rs` | `metal_35b_a3b_grouped_routed_down_accum_proof_512` | M |
| `crates/qwen-llm/src/metal_dflash/tiled_vt_prefill_pilot.rs` | `tiled_vt_restored_prefill_phase` | P |
| `crates/qwen-llm/src/metal_forward/snapshot_segments_lifecycle_pilot.rs` | `snapshot_segmented_publication_full_restore` | G |
| `crates/qwen-llm/src/metal_forward/snapshot_segments_pilot.rs` | `segmented_snapshot_forks_do_not_infer_provenance_or_retain_ancestors` | ordinary |
| `crates/qwen-llm/src/metal_forward/snapshot_segments_pilot.rs` | `segmented_snapshot_mutation_rewind_and_bad_inputs_fail_closed` | ordinary |
| `crates/qwen-llm/src/metal_forward/snapshot_segments_pilot.rs` | `segmented_snapshot_restores_and_shares_only_immutable_prefixes` | ordinary |
| `crates/qwen-llm/src/metal_forward/snapshot_transfer_pilot.rs` | `snapshot_cpu_vs_gpu_publication_restore` | G |
| `crates/qwen-llm/src/metal_forward/tests.rs` | `h2_prefix_cache_correctness_spike` | M |
| `crates/qwen-llm/src/metal_forward/tests.rs` | `mtp_tensor_inventory` | M |
| `crates/qwen-llm/src/qwen4exp_runtime/strict_inject_probe.rs` | `ordinary_prefill_exact_and_abba` | P |
| `crates/qwen-llm/tests/deepseek_v4_position_zero_live.rs` | `current_deepseek_v4_fp4_collapsed_position_3070_packet` | M |
| `crates/qwen-llm/tests/deepseek_v4_position_zero_live.rs` | `current_deepseek_v4_fp4_selection_counterfactual_packet` | M |

## Retained / deferred

- BF16/F16 and legacy-Q4 model witnesses stay: these are default-on precision paths, despite old test names.
- Native quantized embedding and MTP native-bank checks stay; both protect defaults shipped by the knob sweep. The print-only first MTP tensor census is removed, not a bank/math/guard witness.
- Selected-value MMA/GQA12/product probes stay in this batch: GQA12 is now shipped, and the older module contains the product/F64 oracle. The shared state helper is not deleted along with strict-HC timing.
- MoE 511/512-expert indexing witnesses stay: they may be the only explicit last-expert regression checks for shipped grouped kernels.
- DS4 current-asset dtype/dispatch, exact F16 selection, current all-IQ3 promotion and distant-context causal-state checks stay. The two removed FP4 packets characterize a diagnostic-only direct selector with a recorded KILL.
- Active FN, GLM, V4.1 and lens coverage stays. Uncertain snapshot/runtime, model-import and artifact-identity cases are deferred rather than inferred dead from naming or age.

## Validation

Release lib and CLI test-target builds use the same commands as batch 1: `cargo test --offline --locked --release -p qwen-llm --lib --no-run` and `cargo test --offline --locked --release -p qwen-cli --no-run`. The rebased batch-1 builds are the before measurements. The diagnostics-enabled library/integration targets are also checked with `cargo check --offline --locked -p qwen-llm --tests --features dsv4-diagnostics`. `cargo fmt --all --check` and `scripts/check-contracts.sh` run on the final committed tree. No GPU test, inference, GPU timing, build lock or Metal lease is needed.

All before/after commands, durations and exit statuses are retained locally under `bench/results/ignored-sweep-batch1-rebased/` and `bench/results/ignored-sweep-batch2/`. They are single cached unlocked rebuild observations on a machine also running other lanes, not an isolated estimate of deletion-only speedup. Retained test bodies and attributes in touched files are compared against the base; compiled test-name differences are checked against the deletion ledger.
