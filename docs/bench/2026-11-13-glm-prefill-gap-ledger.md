# GLM-5.3 Flash prefill gap ledger

Status: local synthesis of the M5 census and closed swings; no GPU work was
run for this note. The cited shares divide sampled stage spans by ordinary
wall p50. They are rankings, not an additive partition: span sums exceed
command GPU time by 303 / 42 / 191 ms in these cells. No same-machine GLM
MLX-prefill result exists, so the absolute MLX gap remains unmeasured.

## Cells and wall

| Prefix + rows | Ordinary wall p50 | Command GPU p50 | Host encode/readback estimate | Rows/expert, mean (p50/p90/p99/max) |
| --- | ---: | ---: | ---: | --- |
| 49,152 + 2,048 | 1,999.71 ms | 1,992.21 ms | 6.31 ms (0.32%) | 56.89 (41/125/311/1,144) |
| 81,920 + 256 | 453.11 ms | 441.35 ms | 3.40 ms (0.75%) | 7.11 (5/19/64/153) |
| 81,920 + 1,024 | 1,154.81 ms | 1,096.90 ms | 4.57 ms (0.40%) | 28.44 (21/63/168/467) |

The host estimate is the median paired stage-profile wall minus command GPU;
it includes encoding and logits readback. The profiler span sums are 114.7%,
106.5%, and 111.5% of ordinary wall because sampled spans overlap/scale; do
not sum these shares as wall fractions.

## Stage time and share of ordinary wall

| Stage | 49K + 2,048 | 80K + 256 | 80K + 1,024 |
| --- | ---: | ---: | ---: |
| Routed experts | 721.28 ms (36.1%) | 244.90 ms (54.0%) | 446.66 ms (38.7%) |
| Dense FFN | 24.79 ms (1.2%) | 3.40 ms (0.8%) | 12.64 ms (1.1%) |
| Shared expert | 64.61 ms (3.2%) | 12.21 ms (2.7%) | 32.32 ms (2.8%) |
| KDA, all projections + recurrence | 408.08 ms (20.4%) | 64.04 ms (14.1%) | 207.23 ms (17.9%) |
| mHC | 85.78 ms (4.3%) | 10.76 ms (2.4%) | 40.43 ms (3.5%) |
| MLA projection | 42.88 ms (2.1%) | 7.09 ms (1.6%) | 20.82 ms (1.8%) |
| Indexer key/cache projection | 23.28 ms (1.2%) | 4.20 ms (0.9%) | 11.50 ms (1.0%) |
| Sparse query projection | 8.64 ms (0.4%) | 3.47 ms (0.8%) | 5.28 ms (0.5%) |
| Sparse index scores | 376.71 ms (18.8%) | 55.18 ms (12.2%) | 231.24 ms (20.0%) |
| Sparse selection + expansion | 30.44 ms (1.5%) | 5.27 ms (1.2%) | 21.08 ms (1.8%) |
| Sparse attention | 396.36 ms (19.8%) | 49.38 ms (10.9%) | 197.27 ms (17.1%) |
| MLA output projections | 74.44 ms (3.7%) | 9.36 ms (2.1%) | 38.31 ms (3.3%) |
| Router | 36.98 ms (1.8%) | 13.29 ms (2.9%) | 22.55 ms (2.0%) |

KDA subspans (input projections / prepare / recurrence / output norm / output
projection) are 177.92 / 83.01 / 78.37 / 5.90 / 62.88 ms at 2,048 rows;
34.29 / 9.83 / 10.20 / 0.73 / 8.99 ms at 256; and 92.74 / 39.33 / 40.07 /
2.61 / 32.48 ms at 1,024.

## Modeled roof and tensor-peak gap

Cells show achieved logical TFLOP/s / logical GB/s and percentage-point gap
from the 131.7 TFLOP/s tensor peak. FLOPs follow logical tensor shapes;
logical bytes are unique-weight plus modeled scan bytes, not measured bus
traffic. There is no physical memory-bandwidth counter/roof in this census.

| Stage | 49K + 2,048 | 80K + 256 | 80K + 1,024 |
| --- | ---: | ---: | ---: |
| Routed experts | 48.02 TF/s / 147.89 GB/s; gap 63.5 pp | 17.68 / 360.32; 86.6 pp | 38.77 / 236.37; 70.6 pp |
| Dense FFN | 74.85 / 14.99; 43.2 pp | 68.18 / 109.23; 48.2 pp | 73.41 / 29.40; 44.3 pp |
| Shared expert | 67.01 / 13.51; 49.1 pp | 44.31 / 71.49; 66.4 pp | 66.98 / 27.02; 49.1 pp |
| KDA | 47.45 / 9.51; 64.0 pp | 37.80 / 60.61; 71.3 pp | 46.72 / 18.73; 64.5 pp |
| mHC | 0.282 / 0.456; 99.8 pp | 0.281 / 3.633; 99.8 pp | 0.299 / 0.967; 99.8 pp |
| MLA projection | 44.07 / 11.08; 66.5 pp | 33.30 / 66.96; 74.7 pp | 45.39 / 22.82; 65.5 pp |
| Indexer key/cache projection | 2.03 / 0.526; 98.5 pp | 1.40 / 2.915; 98.9 pp | 2.05 / 1.065; 98.4 pp |
| Sparse query projection | 33.48 / 9.17; 74.6 pp | 10.43 / 22.87; 92.1 pp | 27.40 / 15.02; 79.2 pp |
| Sparse index scores | 0.559 / 17.46; 99.6 pp | 0.780 / 24.36; 99.4 pp | 0.748 / 23.36; 99.4 pp |
| Sparse selection + expansion | n/a / 3.376; n/a | n/a / 3.988; n/a | n/a / 4.004; n/a |
| Sparse attention | 1.040 / 16.25; 99.2 pp | 1.044 / 16.31; 99.2 pp | 1.045 / 16.33; 99.2 pp |
| MLA output projections | 45.69 / 9.67; 65.3 pp | 45.42 / 76.90; 65.5 pp | 44.39 / 18.79; 66.3 pp |
| Router | 5.49 / 5.36; 95.8 pp | 1.91 / 14.91; 98.6 pp | 4.50 / 8.79; 96.6 pp |

Dense/shared Q6 / Q8 logical bandwidths (GB/s) are: dense FFN 14.99/0,
109.23/0, 29.40/0; shared expert 13.10/0.41, 69.30/2.19, 26.19/0.83 for
the same three cells. The largest spans by wall are routed experts at every
cell; at the 2K root, KDA, sparse scores, and sparse attention are the next
largest. mHC/indexer/attention have the largest tensor-roof gaps but are not
large dense-MMA workloads, so tensor-peak distance alone is not a speedup
forecast.

## Closed swings

- **KDA value8, 0.958x:** sharing q/k/decay loads across more value rows did
  not repay the likely register/resource cost of the larger independent
  accumulator set; live register occupancy was not measured.
- **Full-chunk IQ repack, 0.976x:** padding the banks did not remove the
  active-tile codebook/scale ALU, shared staging, or barriers; it added 4.8–5.5%
  bank bytes.
- **TensorOps indexer scores, 0.999x:** replacing the score matrix path did
  not accelerate the sparse indexed reduction; the same 377 ms score span
  remained.
- **Sorted expert-major, 1.023x:** row reordering saved little while each
  active tile still paid the dominant IQ decode, scale, staging, and barrier
  work; the required 1.25x stage gate failed.
- **Q6 F16 cache:** F16 expansion trades Q6 decode for wider weight reads and
  a slower selected tile; dense/shared and several KDA/MLA spans regressed,
  despite the cache fitting the memory preflight.

The separate 4K/128-row candidate also remains closed: it regressed ordinary
wall to 0.705x and was not bitwise equal. Do not treat the MLX tile transition
as evidence that this chunk/attention schedule works here.

## Remaining probes, ranked

The branch already contains a default-off parallel KDA-prepare candidate: its
2K-root prepare span improved 80.40 to 31.87 ms (2.52x), and the three
interleaved ordinary-wall pairs improved 2,001.86 to 1,957.42 ms (1.023x),
bitwise logits/state. This is about a 2.3% root-wall gain; the 80K append and
serve gates were not run. It is not a demonstrated >5% wall lever.

1. **Decode IQ weights once into a reusable expert tile for the chunk.** The
   2K routed stage is 36.1% of ordinary wall, so a 1.25x stage win gives about
   7% nominal root-wall headroom before overlap; the 80K/1K stage has similar
   nominal leverage, while 256 rows have only 7.1 rows/expert and may not
   amortize staging. F32 decoded values can target bitwise equality; Half
   staging needs the paired non-inferiority protocol. First probe: one real
   2K routed layer with captured routes, decode each active expert once and
   reuse it over its rows; require ≥1.25x routed-stage speed and pass the
   three-session memory ledger before expanding scope.
2. **Tensorize the selected sparse MLA attention core.** Attention is 19.8%
   of root wall and 17.1% at 80K/1K. A 1.25x attention-only win is only about
   4% nominal root wall; it takes roughly 1.5x, or a coupled score/attention
   win, to cross 5%. Preserve the exact selected IDs and F32 accumulation for
   a bitwise probe; if the tensor path changes reductions, use the paired
   non-inferiority protocol. First probe: one real MLA block at 2K rows with
   frozen production-selected IDs, timing only the selected-attention stage.

Neither lever has evidence yet for a >5% ordinary-wall gain. Expert reuse is
the only credible route to that threshold from the current census; sparse
attention is conditional on a substantially faster isolated kernel. The
already measured KDA prepare win is smaller at whole-cell scale. “Distance to
131.7 TFLOP/s” is a modeled compute comparison, not an MLX speed ratio or a
physical bandwidth roof.

## Sources

- [Baseline census](2026-11-13-glm-prefill-baseline.md) and
  [MLX implementation comparison](2026-11-13-glm-prefill-mlx-comparison.md).
- [KDA value8](2026-11-13-glm-kda-value8.md),
  [IQ repack](2026-11-13-glm-iq-repack.md),
  [TensorOps indexer](2026-11-13-glm-indexer-tensorops.md),
  [sorted expert-major](2026-11-13-glm-sorted-expert-major.md), and
  [Q6 F16 cache](2026-11-13-glm-q6-f16-cache.md) records.
- KDA parallel-prep local test and M5 stage/wall evidence are retained under
  `metagross:~/qwen-runs/2026-11-12-glm-prefill/kda-parallel-prep/`.
