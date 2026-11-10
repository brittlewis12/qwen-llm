# V4.1 batched dense census, split floor, shared-pair rejection — 2026-11-10

M5 Ultra, four-shard `GSQ-RCO IQ2_XS` DeepSeek V4.1 Flash artifact.
Screening uses `--profile iter` (thin LTO) in both arms of every A/B. GPU
work held the canonical Metal lease; raw logs on
`metagross:~/qwen-runs/2026-11-10-dense-census/`.

## Method

`examples/dense_band_census` (perf-ablations) times one projection at a time
from a set of rotating weight copies totalling at least 384 MiB, so reads
stream from DRAM. For each shape it reports medians of per-dispatch GPU time
for:

1. the production per-column native GEMV path (one dispatch per column),
2. the production skinny dispatch with the automatic K split, and
3. the same dispatch with each valid explicit split (`1..=8` divisor).

`cols` is 2, 3 and 4, matching the batched decode band. The full-model A/B
uses `examples/v41_batch_depth` at 32K and 64K, 64 rounds, with the split
floor off (`QWEN_V41_DENSE_SPLIT_MIN_BLOCKS=1`) and on (default 3).

## Census (isolated, B=3, GB/s)

| projection | `n_in -> n_out` | bytes/round | auto split | auto | best split | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q_b | 1280 -> 32768 | 1.153 GB | 5 | 646 | 5 | 653 |
| output_b | 8192 -> 5120 | 1.153 GB | 8 | 686 | 4 | 694 |
| output_a | 8 x (4096 -> 1024) | 0.923 GB | 8 | 558 | 2 | 638 |
| shared gate/up | 5120 -> 2304 | 0.649 GB | 5 | 390 | 5 | 390 |
| shared down | 2304 -> 5120 | 0.324 GB | 3 | 382 | 3 | 382 |
| Engram | 6144 -> 25600 | 0.216 GB | 2 | 770 | 3 | 815 |

Per-shape split detail at B=2/3/4 (selected rows, GB/s):

| shape | split 1 | 2 | 3 | 4 | 5 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared gate/up B=3 | 122 | 231 | - | 344 | **390** | - |
| shared down B=3 | 217 | - | **382** | - | - | - |
| q_b B=3 | **653** | - | - | - | 646 | - |
| q_b B=2 | **343** | - | - | - | 318 | - |
| output_b B=3 | 311 | 531 | - | **694** | - | 686 |
| output_a B=3 | 422 | **638** | - | 622 | - | 558 |
| output_a B=2 | 254 | **582** | - | 590 | - | 269 |
| Engram B=3 | 789 | 770 | **815** | 806 | - | 799 |

Re-derived band budget with these rates totals ~8.9 ms against the measured
8.51 ms `DenseHold`, within 4%, so the isolated rates explain the band.

## Split floor

`skinny_k_split` now narrows the split only when a twofold subdivision still
remains under `DENSE_SKINNY_MIN_BLOCKS` (3). At the V4.1 shapes this changes
only the grouped output_a bank (16 blocks: split 8 -> 4). Leaving q_b and
shared_down alone is deliberate: the census showed that forcing q_b to split
1 is neutral in isolation and preferred on the overlapped wall, while
shared_down split 1 is 1.76x slower.

Same-binary A/B (`QWEN_V41_DENSE_SPLIT_MIN_BLOCKS=1` vs default), 64 rounds:

| depth | B | off ms/round | on ms/round | delta |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 2 | 22.365 | 22.120 | -0.245 |
| 32K | 3 | 26.340 / 26.584 | 26.005 / 26.396 | -0.26 (ABBA) |
| 32K | 4 | 30.809 | 30.531 | -0.278 |
| 64K | 3 | 26.594 / 26.715 | 26.355 / 26.410 | -0.27 (ABBA) |

Least-squares at 32K, `round = F + m*B`: **F 14.028 -> 13.806 ms (-0.22)**,
**m 4.222 -> 4.206 ms/row (-0.02, unchanged within noise)**. The dense band
reads each weight once per round, so the saving is a fixed cost, not a
marginal one.

Numerics, same binary, B=3 32K teacher-forced, 192 targets: signed
**dNLL -2.47e-7**, **mean KL 2.27e-10**, max KL 2.25e-8, **192/192 argmax**,
gate qualified. (An earlier arm that also changed q_b's split measured
dNLL -4.1e-8 with the same KL class; the shipping floor is this file's.)

## Shared gate+up fused pair (rejected)

A Q5_K twin of the existing Q8_0 pair kernel reads both banks in one
dispatch. Isolated at the shared shape, B=3:

| arm | us/layer | GB/s |
| --- | ---: | ---: |
| two skinny dispatches | 72.4 | 224 |
| fused pair | 56.5 | 287 |

It is **bitwise equal** to the two dispatches (identical full-vocabulary
hashes in the wall A/B). But at B=3 32K, thermal 0, ABBA:

| arm | ms/round |
| --- | ---: |
| off | 26.428 |
| fused | 26.502 / 26.502 |

No wall gain: the causal executor already overlaps the two dispatches. The
kernel, host entry, call site and knob were removed.

## Limits

- Isolated rates are service-like and do not include the causal executor's
  overlap; the wall A/B is the authority for shipping decisions.
- The full-model F/m fit comes from single runs per B (thermal level 1); the
  B=3 rows are thermal-0 ABBA and the more reliable deltas.
- The Q8 rank pair rate is not re-measured here; the production path uses the
  split-K pair kernel, not the skinny pair.
- Design D (small-M tensor Q5_K tile) is not built yet. The census says the
  remaining band is q_b/output_b/output_a at 620-690 GB/s and shared at 380,
  so lifting the medium shapes is the tensor tile's job.
