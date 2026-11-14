# Prefill waste audit: GLM-5.3 Flash and DeepSeek V4.1

**Correction (2026-11-14 UTC):** the three V4.1 append index-head matches
below are false positives. The original recorder omitted inline constants;
successive query sub-tiles use `row0=612` and `row0=816` despite identical
rounded grids and buffer spans. Their outputs differ. Do not reuse these
results. The constant-aware audit and strict GPU witness are recorded in
[the serial-attention follow-up](2026-11-13-v41-attention-serial.md).
The original timing tables remain as historical evidence.

**Scope.** GPU dispatch graph only. Two prefill cells per model: a 2,048-row
chunk at 48K and a 1,024-row append at 80K. Flash-Next capture code is kept
but not run (focus/licence change). One model per lease; both cells of a model
share one load.

## Method

- Branch `perf/waste-audit`, based on main `c2935fa36`. The causal recorder now
  exports read-write bindings as both a read and a write span (previously a
  single write flag hid every in-place update), and the three prefill paths
  open a causal scope when node capture is active.
- Probes: `target/iter/examples/{glm_solo_bounds,v41_batch_depth} ... cells`
  under `QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task
  waste-audit --for 15m`. Every capture is bitwise against the same chunk run
  without instrumentation; `waste_audit.py` rejects non-bitwise captures.
- `scripts/bench/waste_audit.py --top 10 --markdown ...` ranks isolated node
  GPU time. Nodes are serialized for sampling, so "GPU service" is the sum of
  isolated dispatch times, not a wall time. The table's wall share uses the
  uninstrumented per-cell wall measured beside the capture.
- Fusion gain is priced as a memory round trip at 800 GB/s
  (`2 * intermediate bytes`), capped at the producer's GPU time. It is not a
  whole-kernel saving.
- Repeated per-layer occurrences are aggregated by `(category, kernel pair)`;
  the count is in the row.
- Host load on the shared M5 was about 16-22 (1/5/15-minute averages) around
  the runs. Builds and loads ran at nice 10 under the lease, which serializes
  GPU access. This is a service-time ranking, not a timing gate.

## Captures

| model | cell | rows | GPU service | uninstrumented wall | dispatches | bitwise |
|---|---|---:|---:|---:|---:|---|
| GLM-5.3 Flash UD-IQ3_XXS | chunk @48K | 2048 | 2015.9 ms | 2005.0 ms | 3725 | yes |
| GLM-5.3 Flash UD-IQ3_XXS | append @80K | 1024 | 1167.4 ms | 1163.3 ms | 2669 | yes |
| V4.1 Flash GSQ-RCO IQ2_XS (file A) | chunk @48K | 2048 | 1696.7 ms | 2646.3 ms | 3435 | yes |
| V4.1 Flash GSQ-RCO IQ2_XS (file A) | append @80K | 1024 | 988.1 ms | 1102.4 ms | 2816 | yes |

V4.1's chunk reference run is the first touch of the resident weights after
the CED prefix, so its wall includes a cold read; use the GPU-service share as
the comparable number for that cell. Raw captures and generated tables:
`metagross:~/qwen-runs/2026-11-12-waste-audit/` (`glm-cells.jsonl`,
`v41-cells.jsonl`, `glm-tables.md`, `v41-tables.md`).

## Findings

Dead writes and recomputation are almost absent: 0 in either GLM cell, 0 in
the V4.1 chunk, and 3 initially suspected index-head MMAs in the V4.1 append
(7.53 ms, 0.68% of wall), subsequently disproven by the inline arguments.
The dominant waste is memory round trips inside attention and
the routed-expert chain. That is consistent with the earlier two-queue finding
that these tiers are bandwidth/serialization-bound rather than launch-bound.

### GLM-5.3 Flash

1. **Latent score -> softmax, 352 (chunk) / 176 (append) pairs.** 177.9 /
   89.6 ms of producer time, 12.0 / 6.0 GB of scores written and re-read;
   priced round-trip saving 30.0 / 15.0 ms (1.5% / 1.3% of cell wall).
2. **Routed-expert gate/up epilogue.** `moe_wide_probe -> moe_up_clamped`
   plus the reverse edge writes 5.5 / 2.8 GB; 180.5 / 118.2 ms of producer
   time, priced saving 27.5 / 13.8 ms.
3. **Expert down -> packed weighted sum.** 13.4 / 8.6 ms (268 / 134 MB
   intermediate), priced 0.67 / 0.34 ms. Small alone but one node in the
   expert chain above.

**Build first:** fuse score and softmax into one online-attention pass. It is
the largest single chain, repeats once per MLA row group, and the same change
pattern applies to V4.1.

### DeepSeek V4.1 Flash (file A)

1. **PV -> normalize rows, 160 / 80 pairs.** 130.8 / 63.5 ms, 10.7 / 5.4 GB,
   priced 26.8 / 13.4 ms (1.0% / 1.2% of wall).
2. **Score -> softmax, 160 / 80 pairs.** 56.5 / 26.4 ms, 13.4 / 6.7 GB,
   priced 33.6 / 16.8 ms (1.3% / 1.5% of wall).
3. **Format movement.** `scatter_offset_f32_to_f16` 280 calls (39.2 / 19.2 ms,
   14.3 / 7.2 GB) and `v41_sorted_pack` 40 calls (32.3 / 15.1 ms, 5.0 /
   2.5 GB). A producer that writes F16/packed directly removes the auxiliary
   dispatch; check the producer's rounding contract.
4. **Disproven duplicate index heads (append only).** Three
   `attention_index_heads_rows_mma` dispatches share the same 75 MB buffer
   spans and rounded grids, but their inline `row0` arguments differ.
   Their 7.53 ms is required work for different query rows, not recoverable
   recomputation. See the correction above.

**Build first:** the two attention fusions (score->softmax and PV->normalize)
as one online-attention epilogue change. Together they are the largest
repeated chain in both V4.1 cells and mirror the GLM item.

## Cross-check against the Opus code audit

- **V4.1 full appends compute unneeded rows (Opus #6):** not contradicted, but
  not visible here. The waste is a row-partition/mask change, not a dead
  dispatch; our top list is orthogonal. Opus agrees that last-row-only logits
  are already implemented, which the head spans confirm.
- **V4.1 host sampler (#5) and full-vocabulary copy/re-check (#9):** out of
  scope (decode/host); no overlap with these prefill captures.
- **GLM latent score->softmax and routed-expert epilogue fusions:** the Opus
  audit has no kernel-graph item for either; complementary, not conflicting.
- **Flash-Next items (#2-#4):** not run, per the focus change.
- **Opus "GLM GPU loop allocates a buffer per token" and pipeline lookups:**
  host-side; our node service times exclude them.
- **Opus "RoPE and mask recompute in-kernel and negligible":** agrees; the
  suspected repeated V4.1 index-head pairs above were later disproven.

## Limits

- These original captures omit inline constants. Equal spans and grids
  cannot prove identical computations; the updated tool reports such legacy
  matches as `unverified_recomputation`. New captures include constants,
  threadgroup memory and pipeline identity in the comparison key.

- The movement classifier is name-based. An early pass counted
  `kda_prefill_*` as copies because "prefill" contains "fill"; the final
  regex requires a token boundary. Dead-write and recomputation checks also
  reject any candidate whose input span is overwritten between the two nodes,
  so scratch reuse across layers is not reported as duplication.
- Real outputs (logits, KV, recurrent state) are excluded by the same-round
  overwrite rule; cross-round accesses are never treated as proof of a dead
  write.
- Node sampling serializes dispatches. These are service-time rankings, not
  wall-time claims; the wall columns are the uninstrumented reference only.

## Reproduction

```sh
nice -n 10 cargo build --profile iter -p qwen-llm --features perf-ablations \
  --example v41_batch_depth --example glm_solo_bounds

# Under the lease, one load per model:
target/iter/examples/glm_solo_bounds "$GLM_MODEL" 49152 1 cells
target/iter/examples/v41_batch_depth "$V41_MODEL" 49152 1 1 cells

uv run --python 3.13 scripts/bench/waste_audit.py --top 10 \
  --markdown glm-tables.md glm-cells.jsonl
uv run --python 3.13 scripts/bench/waste_audit.py --top 10 \
  --markdown v41-tables.md v41-cells.jsonl
```

## Generated tables


### GLM-5.3 Flash


## `glm-cells.jsonl`: prefill, width 1, 2048 rows

GPU service 2015.871 ms (2015.871 ms/chunk); uninstrumented wall 2005.015083 ms/chunk

| category | node / kernel | ms/call | share of cell wall | bytes | proposed fix | expected gain |
|---|---|---:|---:|---:|---|---|
| fusable_chain | `[1:90] kernel_glm_latent_score_tensor -> kernel_glm_latent_softmax (x352)` | 177.8604 | 8.87% | 11995709440 | fuse the producer into its single consumer | 29.9893 ms/call (1.50% wall) |
| fusable_chain | `[1:289] kernel_moe_wide_probe_iq2_s_f16act -> kernel_moe_up_clamped_iq2_s_f16act_grouped_slots_tensor (x41)` | 137.5813 | 6.86% | 5502926848 | fuse the producer into its single consumer | 13.7573 ms/call (0.69% wall) |
| fusable_chain | `[1:290] kernel_moe_up_clamped_iq2_s_f16act_grouped_slots_tensor -> kernel_moe_wide_probe_iq2_s_f16act (x41)` | 42.9320 | 2.14% | 5502926848 | fuse the producer into its single consumer | 13.7573 ms/call (0.69% wall) |
| fusable_chain | `[1:20] kernel_mat_mat_q6_K_f32_tensor_wide -> kernel_deepseek_v4_clamped_swiglu (x46)` | 30.7160 | 1.53% | 1107296256 | fuse the producer into its single consumer | 2.7682 ms/call (0.14% wall) |
| fusable_chain | `[1:3714] kernel_moe_down_iq4_xs_f32_grouped_slots -> kernel_moe_weighted_sum_packed_f32 (x1)` | 13.3780 | 0.67% | 268435456 | fuse the producer into its single consumer | 0.6711 ms/call (0.03% wall) |
| fusable_chain | `[1:85] kernel_mat_mat_f32_f32 -> kernel_deepseek_v4_scale_f32_in_place (x11)` | 4.2710 | 0.21% | 2883584 | fuse the producer into its single consumer | 0.0072 ms/call (0.00% wall) |
| fusable_chain | `[1:967] kernel_moe_wide_probe_iq3_s_f16act -> kernel_moe_up_clamped_iq3_s_f16act_grouped_slots_tensor (x1)` | 3.7970 | 0.19% | 134217728 | fuse the producer into its single consumer | 0.3355 ms/call (0.02% wall) |
| data_movement | `[1:4] kernel_scatter_offset_f32_to_f16 (x98)` | 2.8644 | 0.14% | 2405433344 | have the producer write the consumer format / bind the source | 2.8644 ms/call (0.14% wall) |
| fusable_chain | `[1:968] kernel_moe_up_clamped_iq3_s_f16act_grouped_slots_tensor -> kernel_moe_wide_probe_iq3_s_f16act (x1)` | 1.0718 | 0.05% | 134217728 | fuse the producer into its single consumer | 0.3355 ms/call (0.02% wall) |
| fusable_chain | `[1:3719] kernel_mat_mat_q6_K_f32_tensor_wide -> kernel_add_f32 (x1)` | 0.4600 | 0.02% | 33554432 | fuse the producer into its single consumer | 0.0839 ms/call (0.00% wall) |

## `glm-cells.jsonl`: append, width 1, 1024 rows

GPU service 1167.407 ms (1167.407 ms/chunk); uninstrumented wall 1163.327 ms/chunk

| category | node / kernel | ms/call | share of cell wall | bytes | proposed fix | expected gain |
|---|---|---:|---:|---:|---|---|
| fusable_chain | `[1:90] kernel_glm_latent_score_tensor -> kernel_glm_latent_softmax (x176)` | 89.6068 | 7.70% | 5997854720 | fuse the producer into its single consumer | 14.9946 ms/call (1.29% wall) |
| fusable_chain | `[1:193] kernel_moe_wide_probe_iq2_s_f16act -> kernel_moe_up_clamped_iq2_s_f16act_grouped_slots_tensor (x41)` | 59.3212 | 5.10% | 2751463424 | fuse the producer into its single consumer | 6.8787 ms/call (0.59% wall) |
| fusable_chain | `[1:194] kernel_moe_up_clamped_iq2_s_f16act_grouped_slots_tensor -> kernel_moe_wide_probe_iq2_s_f16act (x41)` | 58.8878 | 5.06% | 2751463424 | fuse the producer into its single consumer | 6.8787 ms/call (0.59% wall) |
| fusable_chain | `[1:20] kernel_mat_mat_q6_K_f32_tensor_wide -> kernel_deepseek_v4_clamped_swiglu (x46)` | 15.6655 | 1.35% | 553648128 | fuse the producer into its single consumer | 1.3841 ms/call (0.12% wall) |
| fusable_chain | `[1:2658] kernel_moe_down_iq4_xs_f32_grouped_slots -> kernel_moe_weighted_sum_packed_f32 (x1)` | 8.5877 | 0.74% | 134217728 | fuse the producer into its single consumer | 0.3355 ms/call (0.03% wall) |
| fusable_chain | `[1:85] kernel_mat_mat_f32_f32 -> kernel_deepseek_v4_scale_f32_in_place (x11)` | 2.9815 | 0.26% | 1441792 | fuse the producer into its single consumer | 0.0036 ms/call (0.00% wall) |
| fusable_chain | `[1:680] kernel_moe_up_clamped_iq3_s_f16act_grouped_slots_tensor -> kernel_moe_wide_probe_iq3_s_f16act (x1)` | 1.7861 | 0.15% | 67108864 | fuse the producer into its single consumer | 0.1678 ms/call (0.01% wall) |
| fusable_chain | `[1:679] kernel_moe_wide_probe_iq3_s_f16act -> kernel_moe_up_clamped_iq3_s_f16act_grouped_slots_tensor (x1)` | 1.4882 | 0.13% | 67108864 | fuse the producer into its single consumer | 0.1678 ms/call (0.01% wall) |
| data_movement | `[1:4] kernel_scatter_offset_f32_to_f16 (x98)` | 1.3613 | 0.12% | 1767899136 | have the producer write the consumer format / bind the source | 1.3613 ms/call (0.12% wall) |
| fusable_chain | `[1:2663] kernel_mat_mat_q6_K_f32_tensor_wide -> kernel_add_f32 (x1)` | 0.2541 | 0.02% | 16777216 | fuse the producer into its single consumer | 0.0419 ms/call (0.00% wall) |


### DeepSeek V4.1 Flash (file A)


## `v41-cells.jsonl`: prefill, width 1, 2048 rows

GPU service 1696.711 ms (1696.711 ms/chunk); uninstrumented wall 2646.3377499999997 ms/chunk

| category | node / kernel | ms/call | share of cell wall | bytes | proposed fix | expected gain |
|---|---|---:|---:|---:|---|---|
| fusable_chain | `[1:24] kernel_deepseek_v41_attention_pv_tensor64 -> kernel_deepseek_v41_attention_normalize_rows (x160)` | 130.7672 | 4.94% | 10737418240 | fuse the producer into its single consumer | 26.8435 ms/call (1.01% wall) |
| fusable_chain | `[1:10] kernel_mat_mat_f16_f16act_tensor_wide -> kernel_deepseek_v41_attention_rope_rows (x40)` | 72.2366 | 2.73% | 10737418240 | fuse the producer into its single consumer | 26.8435 ms/call (1.01% wall) |
| fusable_chain | `[1:22] kernel_deepseek_v41_attention_score_simdgroup_qk -> kernel_deepseek_v41_attention_softmax_rows (x160)` | 56.5033 | 2.14% | 13442744320 | fuse the producer into its single consumer | 33.6069 ms/call (1.27% wall) |
| data_movement | `[1:4] kernel_scatter_offset_f32_to_f16 (x280)` | 39.1577 | 1.48% | 14344519680 | have the producer write the consumer format / bind the source | 39.1577 ms/call (1.48% wall) |
| data_movement | `[1:71] v41_sorted_pack (x40)` | 32.2980 | 1.22% | 5033164800 | have the producer write the consumer format / bind the source | 32.2980 ms/call (1.22% wall) |
| fusable_chain | `[1:5] kernel_mat_mat_q8_0_f16act_tensor_wide -> kernel_rms_norm_mul_rows_f32 (x80)` | 19.4154 | 0.73% | 587202560 | fuse the producer into its single consumer | 1.4680 ms/call (0.06% wall) |
| fusable_chain | `[3:2288] v41_sorted_down_iq2_s -> kernel_deepseek_v41_sum (x1)` | 5.9917 | 0.23% | 251658240 | fuse the producer into its single consumer | 0.6291 ms/call (0.02% wall) |
| fusable_chain | `[2:93] kernel_mat_mat_bf16_f16act_tensor -> kernel_deepseek_v41_attention_rope_rows (x8)` | 2.4572 | 0.09% | 268435456 | fuse the producer into its single consumer | 0.6711 ms/call (0.03% wall) |
| fusable_chain | `[3:2278] kernel_mat_mat_f16_f16act_tensor_wide -> kernel_deepseek_v4_clamped_swiglu (x2)` | 1.0420 | 0.04% | 37748736 | fuse the producer into its single consumer | 0.0944 ms/call (0.00% wall) |
| fusable_chain | `[3:2257] kernel_deepseek_v41_group_rows -> kernel_scatter_offset_f32_to_f16 (x2)` | 0.6772 | 0.03% | 335544320 | fuse the producer into its single consumer | 0.6772 ms/call (0.03% wall) |

## `v41-cells.jsonl`: append, width 1, 1024 rows

GPU service 988.081 ms (988.081 ms/chunk); uninstrumented wall 1102.3846660000002 ms/chunk

| category | node / kernel | ms/call | share of cell wall | bytes | proposed fix | expected gain |
|---|---|---:|---:|---:|---|---|
| fusable_chain | `[1:20] kernel_deepseek_v41_attention_pv_tensor64 -> kernel_deepseek_v41_attention_normalize_rows (x80)` | 63.4711 | 5.76% | 5368709120 | fuse the producer into its single consumer | 13.4218 ms/call (1.22% wall) |
| fusable_chain | `[1:10] kernel_mat_mat_q5_K_f16act_tensor_wide -> kernel_deepseek_v41_attention_rope_rows (x40)` | 41.2491 | 3.74% | 5368709120 | fuse the producer into its single consumer | 13.4218 ms/call (1.22% wall) |
| fusable_chain | `[1:18] kernel_deepseek_v41_attention_score_simdgroup_qk -> kernel_deepseek_v41_attention_softmax_rows (x80)` | 26.3540 | 2.39% | 6721372160 | fuse the producer into its single consumer | 16.8034 ms/call (1.52% wall) |
| data_movement | `[1:4] kernel_scatter_offset_f32_to_f16 (x280)` | 19.2433 | 1.75% | 7172259840 | have the producer write the consumer format / bind the source | 19.2433 ms/call (1.75% wall) |
| data_movement | `[1:57] v41_sorted_pack (x40)` | 15.0799 | 1.37% | 2516582400 | have the producer write the consumer format / bind the source | 15.0799 ms/call (1.37% wall) |
| fusable_chain | `[1:5] kernel_mat_mat_q8_0_f16act_tensor_wide -> kernel_rms_norm_mul_rows_f32 (x80)` | 10.3437 | 0.94% | 293601280 | fuse the producer into its single consumer | 0.7340 ms/call (0.07% wall) |
| unverified_recomputation (disproven) | `[2:103] kernel_deepseek_v41_attention_index_heads_rows_mma (x3)` | 7.5339 | 0.68% | 0 | retain: inline row0 differs between query sub-tiles | 0.0000 ms/call (0.00% wall) |
| fusable_chain | `[3:1875] v41_sorted_down_iq2_s -> kernel_deepseek_v41_sum (x1)` | 4.0983 | 0.37% | 125829120 | fuse the producer into its single consumer | 0.3146 ms/call (0.03% wall) |
| fusable_chain | `[2:79] kernel_mat_mat_bf16_f16act_tensor -> kernel_deepseek_v41_attention_rope_rows (x8)` | 1.2682 | 0.12% | 134217728 | fuse the producer into its single consumer | 0.3355 ms/call (0.03% wall) |
| fusable_chain | `[3:1865] kernel_mat_mat_q5_K_f16act_tensor_wide -> kernel_deepseek_v4_clamped_swiglu (x2)` | 0.7305 | 0.07% | 18874368 | fuse the producer into its single consumer | 0.0472 ms/call (0.00% wall) |

