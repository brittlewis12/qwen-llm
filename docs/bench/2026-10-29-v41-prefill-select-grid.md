# V4.1 Prefill Wide Radix Selector: Exact But Sub-Bar On 128K Append

2026-10-29. Probe branch `perf/v41-prefill-select-grid`, code `142e03b1`
(based on main `4c1f956d`). Clean build, production lease. Raw:
`~/qwen-runs/2026-10-29-v41-encoder-depth/append-128k-*.json`.

## Decision

Park the wide-selector promotion. It is exact and the isolated kernel is
5-6x faster, but the 128K 1K append moves by only **1.8% mean / about 3% on
the warm samples**, inside the run spread and below the 10% bar. The long
append's cost is not primarily the single-threadgroup top-k; the index score
and the per-row chain are the next suspects.

## Mechanism

The 128K append selects 512 keys from a 131,072-key owner bank. The long
prefill path used `select_radix_rows` (one 1,024-thread threadgroup per row)
for every long-selection tile. Decode already used the multi-threadgroup
`select_radix_grid_hist`/`_place` selector, which produces the same IDs given
the same scores. The probe gives the reusable scratch 64 rows above 32K
context and calls the wide selector when the long tile fits; shallow
workspaces keep one row, so the 8K/32K fixture memory caps are unchanged.
`QWEN_V41_PREFILL_SELECT_GRID=0` restores the single-threadgroup path.

## Isolated kernel

`QWEN_V41_SELECT_TIMING=1`, one row of 131,072 keys, top-512:

| kernel | us/row |
| --- | ---: |
| index scores, MMA | 45.9 |
| index scores, ILP | 209.9 |
| top-k, single-threadgroup radix | 121.6 |
| top-k, wide grid (1,024-key chunks) | 20.4 |

The wide top-k is 5.97x faster in isolation for the index-score distribution.

## End-to-end A/B

`qwen-bench suite --pp 1024 -d 131072 --runs 2`, CED fill, 180 s between
arms, ABBA order. Values are tok/s for the 1K append; samples in parentheses.

| arm | select grid | append tok/s |
| --- | --- | ---: |
| on1 | 1 | 570.49 (588.93, 552.05) |
| off1 | 0 | 578.67 (579.47, 577.88) |
| off2 | 0 | 570.07 (561.46, 578.68) |
| on2 | 1 | 557.23 (554.21, 560.25) |

Mean on 563.86, off 574.37: **+1.83%**. The on arm's first sample is a cold
outlier; the remaining on samples are 552-560 versus 561-579 off, about +3%.
The 3-6% run spread is the same order as the effect.

## Reading

The isolated selector saves about 100 us per row at 128K. If the selection
were the append bottleneck and the saving reached the wall, the 1K append
would improve much more than 3%. It does not, so either the selection is a
small share of the serial span, the batched single-threadgroup selector is
already efficient, or the saved time is overlapped by the index score and
per-row chain. The cold-prefill path is unaffected: the probe only changes
the >32K long-tile path, and the 8K pair is not in this A/B.

## Next

Split the 128K append serial span into index-score, per-row transition and
top-k before another selector or dataflow change. If the index score
dominates, the production-shaped MMA GEMM (45.9 us/row isolated) is the next
probe; if the per-row chain dominates, the fix is batching or fusing those
transitions. Do not promote the wide selector alone.
