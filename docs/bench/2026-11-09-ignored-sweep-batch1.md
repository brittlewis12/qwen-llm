# Ignored-test cleanup, batch 1 — 2026-11-09

Base: `d3087f8d3c7c874d84559e666d541769a3642dca`. Census: cx `01a1bfba-826e-7542-8c92-6e3e2e87d090`, all five ledger outputs including its final corrections.

Rebased onto main `fdd087b7431f6251ce36eb5411550ba0f4259650` after the ordinary-test lease fix. The 61 retired entry points remain absent; main’s retained test bodies and nine newly ignored model-reading cases remain intact. ENV is regenerated and the reviewed acceptance inventory is re-rendered and checked by exact discovery, preserving main’s classifications. Release lib/CLI test-target builds and contracts are rerun on this rebased tree; the original source base above still identifies the historical ledger locations.

Retire **61 ignored test entry points**: 59 old timing/attribution probes and two closed-research candidates. Remove their exclusive helpers (17 functions and one grouping struct). Production math, kernels, policies and safety guards are unchanged; one now-unused production import is removed.

All model-witness annotations, qualification recipes, acceptance-inventory records, maintained script callers and six live CPU subprocess helpers are retained. The existing inventory and contract rules need no entry removal: no selected case belongs to them. Generated ENV.md is refreshed for retired test-only reads. Dated recipes are marked historical; frozen artifacts/scripts and the old PERF-LOG remain intact.

The retained-source check compares every surviving test body in touched files byte for byte. Registered/active witnesses are not inferred obsolete from lack of execution logs.

## Decisions and reproduction

Names below identify the source cases at the base commit. Check out that commit to reproduce a retired case with its original fixture, hardware, feature and canonical lease requirements. No replacement runner is added in this deletion batch.

The two explicitly closed candidates are the 2026-08-05 **Fused All-IQ3 Falsifier KILL** and the 2026-07-25 **v0.636–v0.637 Deferred-Restore Publication Floor** (archive-only callers). The current-asset GPU route packet reproduces the 2026-08-05 **Packed GPU Route Integration KILL**. Other DS4 rows are completed routing/selector/attention cost or instrumentation attribution; the named qualification and correctness witnesses remain. Historical MoE-tail sweeps are recorded throughout May–July; default correctness witnesses remain.

| Source file | Retired test | Historical evidence |
| --- | --- | --- |
| `crates/qwen-llm/src/metal/expert.rs:1584` | `routed_expert_block_dispatch_costs` | 2026-10-05 - GLM Bitwise Dispatch Fusion (KDA q/k/v, Shared SwiGLU): tg128 ~+2%; Routed-Expert Screen |
| `crates/qwen-llm/src/checkpoint_store.rs:1759` | `checkpoint_deferred_restore_exact_size_floor` | PERF-LOG 2026-07-25 v0.636–v0.637 floor; frozen archive runners |
| `crates/qwen-llm/src/metal/attn.rs:4722` | `attn_matrix_online_vs_sidecar_microbench` | 2026-07-02 - v0.439 Two-Pass Online-Softmax Matrix Prefill Attention |
| `crates/qwen-llm/src/metal/attn.rs:5519` | `attn_decode_v4_bandwidth_audit_130k` | 2026-08-22 — Attn-V4 Split-K Under-Partitioning: Root Cause, Retuned, Committed |
| `crates/qwen-llm/src/deepseek_v4_metal/prefill/tests.rs:4829` | `profile_packed_grouped_fused_all_iq3_production_shape` | PERF-LOG 2026-08-05 fused all-IQ3 KILL; sealed ignore reason |
| `crates/qwen-llm/src/deepseek_v4_metal/prefill/tests.rs:6715` | `profile_exact_packed_gpu_route_and_schedule_packet` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:2462` | `current_asset_packed_gpu_route_kill_packet` | PERF-LOG 2026-08-05 packed GPU route integration KILL |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:3406` | `current_asset_packed_attention_split_attribution_packet` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:4072` | `current_asset_packed_post_route_stage_attribution_packet` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:11943` | `profile_multigroup_selector_full_ceiling` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:12215` | `profile_sparse_csa_multigroup_integrated_gate` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:12859` | `profile_multigroup_selector_crossover` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:13084` | `profile_multigroup_selector_threshold_ceiling` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:14178` | `profile_shallow_sparse_selector_crossover` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:14542` | `profile_packed_top512_radix4_against_bitwise` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:17745` | `profile_sparse_csa_decode_phases_at_far_context` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:18346` | `profile_tiled_hca_at_far_context` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:20395` | `profile_gpu_moe_route_dispatches` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/deepseek_v4_metal/tests.rs:21391` | `profile_indexed_ds4_expert_projections_against_static_views` | recorded DS4 routing/selector/attention attribution (2026-08-04..2026-09-24) |
| `crates/qwen-llm/src/metal_dflash/tests.rs:3940` | `metal_35b_a3b_packed_moe_tail_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:3956` | `metal_122b_a10b_packed_moe_tail_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:4402` | `metal_35b_a3b_packed_moe_tail_ab_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:4418` | `metal_122b_a10b_packed_moe_tail_ab_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5042` | `metal_35b_a3b_live_packed_moe_tail_phase_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5058` | `metal_122b_a10b_live_packed_moe_tail_phase_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5447` | `metal_35b_a3b_live_grouped_moe_tail_phase_profile_320` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5463` | `metal_122b_a10b_live_grouped_moe_tail_phase_profile_320` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5479` | `metal_35b_a3b_live_grouped_moe_tail_phase_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5495` | `metal_122b_a10b_live_grouped_moe_tail_phase_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5511` | `metal_35b_a3b_live_grouped_moe_tail_phase_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:5527` | `metal_122b_a10b_live_grouped_moe_tail_phase_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:6516` | `metal_122b_a10b_gpu_compacted_grouped_down_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:6532` | `metal_35b_a3b_gpu_compacted_grouped_down_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8182` | `metal_122b_a10b_grouped_swiglu_down_backend_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8198` | `metal_35b_a3b_grouped_swiglu_down_backend_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8214` | `metal_35b_a3b_grouped_swiglu_down_backend_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8230` | `metal_122b_a10b_grouped_swiglu_down_backend_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8246` | `metal_35b_a3b_grouped_swiglu_down_backend_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8262` | `metal_122b_a10b_grouped_swiglu_down_backend_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8525` | `metal_35b_a3b_grouped_swiglu_fused_bank_profile_320` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8541` | `metal_122b_a10b_grouped_swiglu_fused_bank_profile_320` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8557` | `metal_35b_a3b_grouped_swiglu_fused_bank_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8573` | `metal_122b_a10b_grouped_swiglu_fused_bank_profile_512` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8589` | `metal_35b_a3b_grouped_swiglu_fused_bank_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:8605` | `metal_122b_a10b_grouped_swiglu_fused_bank_profile_1024` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:10279` | `metal_122b_a10b_gpu_owned_grouped_routed_backend_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:10295` | `metal_35b_a3b_gpu_owned_grouped_routed_backend_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:10872` | `metal_122b_a10b_fused_routed_tail_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:10888` | `metal_35b_a3b_fused_routed_tail_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:11141` | `metal_122b_a10b_grouped_routed_down_experiment_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:11157` | `metal_35b_a3b_grouped_routed_down_experiment_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:11431` | `metal_122b_a10b_hot_expert_routed_down_hybrid_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:11450` | `metal_35b_a3b_hot_expert_routed_down_hybrid_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:12248` | `metal_27b_packed_prefill_phase_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal_dflash/tests.rs:12264` | `metal_27b_packed_gdn_tail_profile` | 2026-05-14..2026-07-03: recorded packed MoE-tail backend sweeps and dense prefill phase attribution |
| `crates/qwen-llm/src/metal/moe.rs:8637` | `attn_v4_nwg_sweep_moe_shapes` | 2026-05-21 — Real Rollout Prompt Lane And A3B Group-8 Long-Prefill Attention Breakthrough |
| `crates/qwen-llm/src/metal/moe.rs:8737` | `attn_v4_tile_c_sweep_moe_shapes` | 2026-05-21 — Real Rollout Prompt Lane And A3B Group-8 Long-Prefill Attention Breakthrough |
| `crates/qwen-llm/src/metal/moe.rs:8850` | `attn_v4_main_reduce_breakdown_moe_shapes` | 2026-06-23 — v0.313 Coop4 Long-Attention Falsifier; 2026-05-21 — Real Rollout Prompt Lane And A3B Group-8 Long-Prefill Attention Breakthrough |
| `crates/qwen-llm/src/metal/moe.rs:9005` | `attn_v4_head_major_main_reduce_breakdown_moe_shapes` | 2026-06-25 - v0.335 Head-Major KV Attention Falsifier |
| `crates/qwen-llm/src/metal/moe.rs:9201` | `attn_prefill_v4_main_reduce_breakdown_moe_shapes` | 2026-05-21 — Default Long-Prefill Packed Attention For A3B/A10B, And Re-Rank The Residual Gap |
| `crates/qwen-llm/src/metal/kda.rs:1305` | `kda_block_dispatch_costs` | 2026-10-05 - GLM Short-K Q8_0 Mat-Vec for KDA: tg128 +4.7% (30.0 -> 31.4 tok/s) |

## Deferred rather than guessed

- All seven old `tests/dflash_correctness.rs` timing cases are retained for now. Their timing bodies mix numerical assertions with cost measurements; `multicol_gemv_micro_27b` is still cited as the E0 check for shipped multi-column GEMV. Separate the accuracy witness before dropping its timing.
- `current_asset_packed_all_iq3_sealed_promotion_gate` is retained: its relation to current default grouped-IQ3 qualification needs a separate review.
- `profile_lightning_fp4_matrix_shadow_at_far_context`, `run_short_append_probe` and `m5_kda_half_complete_eight_op_fanout_bitwise_and_full_copy_cost` are retained for review rather than treating modification dates as liveness proof.
- Active Flash-Next, GLM and V4.1 worktree files are left alone, including unregistered timing probes. Registered model/state/distribution witnesses remain for later migration.
- The census’s pre-existing MTP “(MTP-aware variant)” fixture annotation remains an unresolved runner-metadata issue; this batch does not change any witness annotation.

## Validation and build-time comparison

Identical unlocked release-build commands are timed before and after: `cargo test --offline --locked --release -p qwen-llm --lib --no-run` and `cargo test --offline --locked --release -p qwen-cli --no-run`. Full commands, exit codes and elapsed times are retained locally in `bench/results/ignored-sweep-batch1/baseline.json` and `after.json` and reported with the batch. These are single cached rebuild observations on a shared machine, not an isolated causal speedup estimate. No GPU tests or model inference are required for this deletion batch.

Validation also checks the diagnostics-enabled library, default integration-test compilation, generated references, compiled test-name removal and `scripts/check-contracts.sh` (without a build lock or GPU lease).
