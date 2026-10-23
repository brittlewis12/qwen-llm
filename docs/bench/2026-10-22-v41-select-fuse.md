# V4.1 prefill: fused selection sort and chronological output

Candidate `perf/v41-select-fuse` (`43737744`) on main `49ff1a6e` (batched
selection landing). M5 Ultra 80-core, Unsloth DeepSeek-V4.1-Flash-1017
UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 2048,4096 --runs 2 --seed 1`, A/B/C/C/B/A, same
binary.

Three arms:

- **incumbent**: `QWEN_V41_SELECT_PER_ROW=1`, the original per-row selector.
- **split**: `QWEN_V41_SELECT_SPLIT_SORT=1`, batched prepare/index with the
  incumbent per-row sort/chronological (the previous landing).
- **fused**: default, one threadgroup per top-k row.

Raw: `metagross:~/qwen-runs/2026-10-22-select-fuse/`.

## Throughput

| shape | incumbent | split | fused | fused vs split | fused vs incumbent |
|---|---:|---:|---:|---:|---:|
| Q2 pp2048 | 545.8 | 565.6 | **653.2** | +15.5% | +19.7% |
| Q2 pp4096 | 496.7 | 514.2 | **572.1** | +11.3% | +15.2% |
| IQ3 pp2048 | 541.5 | 559.2 | **644.9** | +15.3% | +19.1% |
| IQ3 pp4096 | 492.9 | 510.6 | **565.7** | +10.8% | +14.8% |

tok/s, arm means. Both fused arms sit far above both split arms at every
shape and quant.

Rebased confirmation on the final tree (`6fec78d4`): Q2 PP2048 fused 652.4
vs split 563.4; IQ3 PP2048 fused 647.4 vs split 557.9, same direction and
magnitude as the three-arm table.

## Dispatch census

Q2 PP2048, one run:

- split: **88,133** dispatches
- fused: **45,982** dispatches
- delta: **-42,151 (-47.8%)**

The removed families are `sort_init`, `sort_tile`, `sort_step` and
`chronological`. One fused dispatch per 32-row tile replaces the per-row
global-scratch round trip.

## Producer-stage split

`QWEN_V41_PREFILL_STAGES=1`, Q2 PP2048, shares only (sampling inflates wall
about 2.4x):

| span | split | fused |
|---|---:|---:|
| attention HC | 2.5% | 2.9% |
| attention prepare | 5.5% | 6.4% |
| **owner/producer serial** | **18.7%** | **5.2%** |
| attention tiles | 29.8% | 34.7% |
| attention finish/post | 8.4% | 9.8% |
| FFN | 35.0% | 40.9% |

Serial GPU time fell from **1,895.8 ms to 452.5 ms (-76.1%)**; the sampled
total fell 10,111.1 -> 8,673.5 ms (-14.2%), matching the throughput gain.
IQ3 PP2048 post-fused shares are FFN 42.9%, tiles 34.0%, serial 5.1%,
finish/post 9.2%, prepare 6.0%, HC 2.9%.

The selection sort was not just a dispatch-floor problem: the per-row global
scratch round trip and repeated launches dominated the serial span, which is
why the fused one-threadgroup form beat the expected 10-15% of that stage by
a wide margin.

## What changed

`kernel_deepseek_v41_attention_select_fused_rows` runs one 256-thread
threadgroup per top-k row. It loads the row's index-score slab into
threadgroup memory, runs the incumbent bitonic network and tie-break
(`values`/`order`, 2048 entries each), and emits chronological top-k IDs
directly to the row's `selected` buffer. All-selected rows (visible length
<= `index_topk`) are left to the batched prepare; rows with padded length
above 2048 keep the per-row sort fallback. The kernel reports the same NaN
status bit as `sort_init`.

The fused path uses 16 KiB of static threadgroup memory. At PP4096 the
ratio-one producer layers reach visible length 4096, so they still take the
per-row fallback; the ratio-two layers use the fused path. Extending the
fused selector to padded 4096 (32 KiB threadgroup, or a two-pass merge) is
the next selection step.

## Correctness

- `query_batch_batched_selection_matches_incumbent_across_tiles` now also
  asserts that the fused kernel is dispatched for the 65-row,
  second-tile-transition cases; it compares outputs, selected IDs, masks,
  causal state and the first decode against the scalar path.
- 31 attention tests and 14 prefill tests pass.
- Same-binary held-text witness, IQ3, 384 targets: incumbent, split and
  fused all give identical NLL and top-1 at the packet precision:

| document | NLL | top-1 |
|---|---:|---:|
| prose_1 | 2.604260127 | 0.594 |
| prose_2 | 4.204815708 | 0.188 |
| prose_3 | 2.260082474 | 0.453 |
| code_4 | 1.252495368 | 0.719 |
| code_5 | 1.570008374 | 0.688 |
| code_6 | 0.986541110 | 0.766 |
| mean | 2.146367194 | 0.568 |

## Limits

- One artifact pair, one M5 Ultra. The fused selector is exact on the
  held-text and synthetic selection protocols, but it is a rewritten sort,
  so keep the regression and witness checks with any future change.
- PP4096 late ratio-one tiles fell back above padded 2048; the 4K follow-up
  covers them (bench/2026-10-22-v41-select-fuse4k.md).
- The 16 KiB common-path threadgroup footprint is unchanged; the 4K follow-up
  adds the 32 KiB late-tile variant.
