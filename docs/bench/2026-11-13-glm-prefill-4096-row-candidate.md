# GLM prefill: 4,096-row / 128-query candidate rejected

Status: closed after failing both the isolated-stage and paired-wall gates.
Candidate source commit: `f65c48b69` (default-off while evaluated). The code
has been removed from the current worktree; this note and the raw evidence are
retained.

## Candidate and conditions

The candidate raised the packed chunk ceiling to 4,096 rows and enabled
128-query sparse-index/attention microbatches for those sessions. The paired
root cell used the same 4,096 real tokens at depth 49,152: baseline processed
two 2,048-row chunks; candidate processed one 4,096-row chunk. Both sessions
restored the same captured 48K state. Three interleaved pairs ran under the
same `iter` binary and one M5 model load.

The candidate passed the residency check. Thermal pressure was 1 through most
of the measurement and reached 2 at the end, within the allowed ceiling. The
preload process snapshot shows background `~/ghost` headless work, including
two `shadps4` workers at about 448% and 453% CPU. Load averages were about
18–22 during the candidate cells, versus about 4 during the baseline census.
The paired wall and routed-stage samples were stable within each arm; all
three pairs lost by a wide margin under this shared-host load.

## Gates

| Measure | Baseline: 2K + 64 queries | Candidate: 4K + 128 queries | Candidate / baseline |
| --- | ---: | ---: | ---: |
| Routed-expert stage, same 4,096-token window | 1,441.33 ms | 2,619.44 ms | 0.550x |
| Ordinary root-cell wall, p50 of three paired samples | 4,055.51 ms | 5,752.75 ms | 0.705x |

The three ordinary pairs were 4,043.42 → 5,753.78 ms, 4,055.51 → 5,703.44
ms, and 4,095.24 → 5,752.75 ms. Routed-expert samples were about 1,440–1,442
ms for baseline and 2,618–2,620 ms for candidate. The required 1.25x
isolated-stage improvement failed: candidate expert throughput was 0.55x
baseline. The paired wall also regressed, so the serve gate was not run.

Fast logits and captured persistent state were not bitwise equal in any of the
three stage-profile pairs or ordinary-wall pairs. The non-inferiority protocol
was not run because the performance gates had already failed.

The 4K route distribution reached mean 113.8 rows per expert (p50 85, p90
239), up from 56.9 (p50 41, p90 125) at 2K. Filling larger tiles did not
make the current route/attention schedule faster: in the paired stage samples,
sparse index scores rose from about 761 ms to 1,282 ms, and sparse attention
from about 791 ms to 1,443 ms. The 80K append cells also regressed in the
4K-session candidate: 256 rows took 489.91 ms versus the 453.11 ms baseline;
1,024 rows took 1,323.06 ms versus 1,154.81 ms.

**Decision: remove the 4K/128 code path and move to the resident Q6 F16 cache
candidate.** The cache targets Q6 projections across KDA, MLA, dense FFN, and
the shared expert; the baseline census measures those stages separately. Its
14–15 GiB residency and three-session memory fit remain required gates.

## Evidence

- Paired candidate run and process/thermal snapshots:
  [`candidate-4096/`](2026-11-13-glm-prefill/candidate-4096/).
- Baseline stage census: [GLM prefill baseline](2026-11-13-glm-prefill-baseline.md).
- The independent MLX/MLX-serve source comparison remains in
  [the comparison note](2026-11-13-glm-prefill-mlx-comparison.md).
