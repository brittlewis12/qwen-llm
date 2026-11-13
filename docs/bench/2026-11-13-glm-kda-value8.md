# GLM KDA Value8 Recurrence: Rejected

## Result

On the M5 Ultra, the 49,152-token prefix plus 2,048-row full chunk produced
three interleaved `packed_kda_recurrence` spans per arm. The existing value4
path measured **75.395 / 75.341 / 75.052 ms**; value8 measured **78.633 /
78.691 / 78.643 ms**. Median speedup was **0.958x**, below the required
**1.25x** isolated-stage gate.

The stage-profile wall medians were 1,951.95 ms for value4 and 2,361.33 ms for
value8. These are not ordinary-wall A/B results; load average was high, so
they remain supporting context only. The isolated gate failed, and no paired
ordinary-wall, exactness/non-inferiority, or serve campaign was run.

## Run conditions

- M5 worktree built in the `iter` profile; measurement ran at normal priority.
- The production value4 route was the incumbent. The candidate interleaved
  eight independent value rows per SIMD group, reusing each q/k/decay load
  across them while keeping each row's F32 operation and reduction order.
- Pressure stayed at 0, with 60 seconds of pressure-at-most-1 settling before
  model load. The pre-load residency snapshots were empty.
- Load average changed from `20.13 19.60 18.74` at the header to
  `18.94 19.57 18.82` after the run.
- No session buffers, state formats, or model weights were added.

## Decision

The 1.25x recurrence-stage gate failed, so paired wall, quality, and serve
gates were skipped. The default-off value8 kernel and probe were removed. The
next lever is a parallel-row KDA preparation kernel, aimed at the 83 ms KDA
prepare span in the 49K + 2,048 baseline.

## Evidence

Raw candidate, thermal, process, load, corpus, build, and status files are in
[`kda-value8/attempt-1-stage-gate/`](2026-11-13-glm-prefill/kda-value8/attempt-1-stage-gate/).
The M5 campaign output remains under
`metagross:~/qwen-runs/2026-11-12-glm-prefill/kda-value8-stage/`.
