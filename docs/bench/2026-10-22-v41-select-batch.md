# V4.1 prefill: batched unpruned selection

Candidate `perf/v41-select-batch` (`efbbb233`) on main `545318d3` (tensor
dense projections). M5 Ultra 80-core, Unsloth DeepSeek-V4.1-Flash-1017
UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 2048,4096 --runs 2 --seed 1`, A/B/B/A, same binary.
`QWEN_V41_SELECT_PER_ROW=1` keeps the incumbent per-row selection for the
control. Raw: `metagross:~/qwen-runs/2026-10-22-select-batch/`.

## Throughput

| shape | per-row | batched | change |
|---|---:|---:|---:|
| Q2 pp2048 | 545.3 (543.9-546.6) | 561.9 (561.4-562.4) | **+3.05%** |
| Q2 pp4096 | 495.4 (494.0-496.8) | 513.1 (511.4-514.8) | **+3.57%** |
| IQ3 pp2048 | 540.3 (540.3-540.3) | 559.0 (558.5-559.5) | **+3.46%** |
| IQ3 pp4096 | 490.8 (490.6-490.9) | 509.4 (509.1-509.7) | **+3.80%** |

tok/s, arm means with ranges. Both batched arms sit above both per-row arms
at every shape and quant.

## Dispatch census

Q2 PP2048, one run, `QWEN_V41_CENSUS=1`:

- per-row: **105,202** dispatches
- batched: **88,133** dispatches
- delta: **-17,069 (-16.2%)**

The removed families are the per-row `selection_prepare` and `index_heads`
launches. The per-row `sort_init`, `sort_tile`, `sort_step` and
`chronological` passes remain and are now the largest non-sink families.
At PP2048 the visible lengths stay below 16,385 keys, so the pruning path
(`blocks > candidate_topk`) is not reached.

## Post-selection span split

Fresh `QWEN_V41_PREFILL_STAGES=1` sample at PP2048, shares only (sampling
inflates wall about 2.8x):

| span | Q2 | IQ3 |
|---|---:|---:|
| FFN | 35.1% | 36.6% |
| attention tiles (QK^T/softmax/PV) | 29.8% | 29.4% |
| owner/producer serial | 18.7% | 18.5% |
| attention finish/post | 8.4% | 7.9% |
| attention prepare | 5.5% | 5.2% |
| attention HC | 2.5% | 2.5% |

The serial share still includes compressor, index-key publication, selection
and sorting; selection batching removed about 3.5% of wall, not the whole
serial span.

## What changed

Compressor and index-key publication remain position-ordered, one row at a
time. For unpruned layers, selection then moves in 32-row tiles:

- one `selection_prepare_rows` dispatch writes all-selected IDs and the
  source-layer candidate mask for the tile;
- one `index_heads_rows` dispatch computes index scores for rows whose
  visible length exceeds `index_topk`, with the same 32-lane head-sum order;
- the incumbent per-row `sort_init`/`sort_tile`/`sort_step` and
  `chronological` passes consume each row's tile score slab unchanged.

The pruning path (`>16,385` visible keys) keeps the old per-row selector.
The tiled score scratch is bounded by
`min(capacity, candidate_topk * candidate_block)` and priced in admission.

## Correctness

Two bugs were caught by the new regression before landing:

- The first implementation used tile-local row indices for the
  selected/candidates/index-query/index-weight buffers, so only the first
  tile was addressed correctly. A held-text witness showed a large NLL
  shift; the kernels now take an explicit tile row offset.
- The prepare grid width was bounded by the last row's all-selected extent.
  A tile ending in top-k rows could under-cover earlier all-selected rows
  (row 32 wrote 256 of 512 IDs). The x bound is now tile-wide
  `min(last_length, top_k)`.

`query_batch_batched_selection_matches_incumbent_across_tiles` runs 65 rows
at `(start=479, ratio=1/2)` and `(start=991, ratio=1/2)`, putting the
`512 -> 513` transition inside the second tile; it compares outputs,
selected IDs, masks, causal state and the first decode against the scalar
path. All 31 attention tests and 14 prefill tests pass.

Same-binary held-text witness (IQ3, 384 targets, batched versus
`QWEN_V41_SELECT_PER_ROW=1`): all six document NLL values and all top-1
values are identical at the reported precision:

| document | NLL | top-1 |
|---|---:|---:|
| prose_1 | 2.604260127 | 0.594 |
| prose_2 | 4.204815708 | 0.188 |
| prose_3 | 2.260082474 | 0.453 |
| code_4 | 1.252495368 | 0.719 |
| code_5 | 1.570008374 | 0.688 |
| code_6 | 0.986541110 | 0.766 |
| mean | 2.146367194 | 0.568 |

The selection change is numerically identical on this protocol; the absolute
NLL differs from the pre-tensor-dense packet because the tensor tiles landed
in between.

## Remaining

The per-row sort/chronological pass is now the main selection cost. A fused
one-threadgroup-per-row selector for padded lengths up to 2048 is the next
step, but it needs the three-arm producer replay cx recommended:
incumbent, current batching, and fused selector, measuring selection agreement
and producer-stage time, not census alone. The pruning path above 16,385
keys remains on the old selector.

## Limits

- One artifact pair, one M5 Ultra, one protocol. The gain is in the unpruned
  path; the pruning fallback is unchanged and unmeasured here.
- The batched index-score kernel preserves the scalar head-sum order and
  per-key dot order on the same inputs; the same-binary witness is
  identical.
- `QWEN_V41_SELECT_PER_ROW=1` is a timing/control hook, not a product mode.
