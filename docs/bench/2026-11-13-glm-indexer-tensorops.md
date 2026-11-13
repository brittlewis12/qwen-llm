# GLM TensorOps Indexer Scores: Rejected

## Result

On the M5 Ultra, the 49,152-token prefix plus 2,048-row full chunk produced
three interleaved sparse-score spans per arm. The incumbent matrix path was
377.200 / 377.102 / 377.082 ms; the TensorOps candidate was 377.393 / 377.381
/ 377.010 ms. Median speedup was **0.999x**, below the required **1.25x**
isolated-stage gate. The candidate does not reduce the 377 ms indexer-score
span at this chunk size.

The stage-profile wall medians were 1,932.48 ms for the incumbent and 1,950.47
ms for TensorOps. One TensorOps sample was 2,440.31 ms; these are supporting
stage-profile observations, not the required ordinary-wall A/B. Since the
isolated-stage gate failed, no ordinary-wall pairs, exactness/non-inferiority
check, or serve campaign was run.

## Run conditions

- M5 worktree commit `6ddaef946`, `iter` profile, normal-priority measurement.
- Root depth 49,152 and 2,048 new rows; the baseline mean was 56.89 routed
  rows/expert.
- Pressure was 0 throughout, with 60 seconds of pressure-0 settling before
  model load. The before-load and preload snapshots found no resident model.
- Load average changed from `10.91 12.12 16.24` before the cell to
  `11.18 11.96 15.84` after it. This was substantially above the baseline
  census load; the interleaved stage spans remain close, but the run does not
  support an end-to-end claim.
- The candidate requested the MPP TensorOps score path for Fast/Half, 64-head,
  128-wide indexer queries. The incumbent and candidate reused the same
  restored prefix snapshot and score slab.

## Decision

The candidate missed the first gate, so the paired wall, quality, and serve
gates were skipped. The TensorOps score code and probe are removed. The next
lever is a lossless in-memory IQ2_S/IQ3_S repack measured on the 2,048-row
full-chunk regime. The earlier B=4 repack result is not treated as a proxy for
this ALU-bound regime.

No external code was copied or adapted for this candidate.

## Evidence

Raw stage, process, load, thermal, status, and corpus-hash files are in
[`indexer-tensorops/`](2026-11-13-glm-prefill/indexer-tensorops/). The M5
campaign output is retained under
`metagross:~/qwen-runs/2026-11-12-glm-prefill/indexer-tensor-stage/`.
