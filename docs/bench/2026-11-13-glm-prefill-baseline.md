# GLM-5.3 Flash prefill census

Status: baseline complete; no optimization candidate has run yet.

## Run

- Artifact: stock GLM-5.3-Flash UD-IQ3_XXS, model shard
  `/Users/tito/models/glm-stock-iq3-download/glm5-next/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf`.
- M5 worktree commit: `090ad87365f3dd7fb1f601b9fca6712805d2a73e`, `iter`
  profile, Apple M5 Ultra, pressure 0 throughout measurement.
- Cells: prefix depth 49,152 with 2,048 new rows; depth 81,920 with 256 and
  1,024 new rows. Five ordinary wall samples and three stage profiles per
  cell. Routes were production-selected.
- Corpus SHA256:
  `117c207146a43d7e7c97de4a4bfc36d883d2550db6bb849ef983f1ba24131b72`.
- Load average during the cells was 3.88–4.14. The before/preload snapshots
  had no `~/ghost` process and no other model process. This run completed at
  2026-11-13 09:41:45 UTC; no process from this census remained afterward.
- The first attribution attempt used 8,192 counter samples and was rejected by
  Metal's 32 KiB sample-buffer limit. It passed the residency check and loaded
  GLM, then failed before producing a cell result; its process exited. The
  failed run is retained under `stage-attribution/failed-sample-buffer-cap/`;
  the successful run uses 4,096 samples.

Raw records, including the process and thermal snapshots, are in
[`stage-attribution/`](2026-11-13-glm-prefill/stage-attribution/). The earlier
run with merged MLA attribution is retained as
[`baseline-census.jsonl`](2026-11-13-glm-prefill/baseline-census.jsonl).

## Wall and route occupancy

| Cell | Ordinary wall p50 | Stage-profile wall p50 | Command GPU p50 | Host, encode, and readback estimate | Mean rows/expert; p50 / p90 / p99 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 + 2,048 | 1,999.71 ms | 1,998.35 ms | 1,992.21 ms | 6.31 ms | 56.89; 41 / 125 / 311 / 1,144 |
| 81,920 + 256 | 453.11 ms | 445.08 ms | 441.35 ms | 3.40 ms | 7.11; 5 / 19 / 64 / 153 |
| 81,920 + 1,024 | 1,154.81 ms | 1,101.48 ms | 1,096.90 ms | 4.57 ms | 28.44; 21 / 63 / 168 / 467 |

The ordinary-wall p50s came from five samples. The host figure is the median of
paired stage-profile wall minus command-GPU samples. It includes host encoding
and logits readback, so it is not a CPU-encode-only timer. Stage profiling was
0.999x / 0.982x / 0.954x the ordinary wall; the small append differences are
measurement variation, not speedups.

The 2,048-row chunk has 56.89 routed rows per expert on average. Its p50 is 41
because routing is skewed. At the append widths the means fall to 7.11 and
28.44. The append measurements from the existing small-row work do not
represent the full-chunk expert regime.

## Stage times and modeled roofline

Each cell below is `milliseconds / estimated TFLOP/s / logical GB/s / percent
of the 131.7 TFLOP/s tensor peak`. FLOPs are estimated from logical tensor
shapes. Logical bytes are unique-weight bytes plus modeled scan bytes, not
physical memory-bus traffic. The roof percentage is therefore a cost-model
estimate, not a hardware counter. Sparse selection has no modeled FLOP count.

| Stage | 49,152 + 2,048 | 81,920 + 256 | 81,920 + 1,024 |
| --- | --- | --- | --- |
| Routed experts | 721.28 / 48.02 / 147.89 / 36.46% | 244.90 / 17.68 / 360.32 / 13.42% | 446.66 / 38.77 / 236.37 / 29.44% |
| Dense FFN | 24.79 / 74.85 / 14.99 / 56.83% | 3.40 / 68.18 / 109.23 / 51.77% | 12.64 / 73.41 / 29.40 / 55.74% |
| Shared expert | 64.61 / 67.01 / 13.51 / 50.88% | 12.21 / 44.31 / 71.49 / 33.64% | 32.32 / 66.98 / 27.02 / 50.86% |
| KDA, all projections and recurrence | 408.08 / 47.45 / 9.51 / 36.03% | 64.04 / 37.80 / 60.61 / 28.70% | 207.23 / 46.72 / 18.73 / 35.47% |
| mHC | 85.78 / 0.282 / 0.456 / 0.214% | 10.76 / 0.281 / 3.633 / 0.213% | 40.43 / 0.299 / 0.967 / 0.227% |
| MLA projection | 42.88 / 44.07 / 11.08 / 33.46% | 7.09 / 33.30 / 66.96 / 25.29% | 20.82 / 45.39 / 22.82 / 34.46% |
| Indexer key/cache projection | 23.28 / 2.03 / 0.526 / 1.54% | 4.20 / 1.40 / 2.915 / 1.07% | 11.50 / 2.05 / 1.065 / 1.56% |
| Sparse query projection | 8.64 / 33.48 / 9.17 / 25.42% | 3.47 / 10.43 / 22.87 / 7.92% | 5.28 / 27.40 / 15.02 / 20.81% |
| Sparse index scores | 376.71 / 0.559 / 17.46 / 0.424% | 55.18 / 0.780 / 24.36 / 0.592% | 231.24 / 0.748 / 23.36 / 0.568% |
| Sparse selection and expansion | 30.44 / n/a / 3.376 / n/a | 5.27 / n/a / 3.988 / n/a | 21.08 / n/a / 4.004 / n/a |
| Sparse attention | 396.36 / 1.040 / 16.25 / 0.790% | 49.38 / 1.044 / 16.31 / 0.793% | 197.27 / 1.045 / 16.33 / 0.794% |
| MLA output projections | 74.44 / 45.69 / 9.67 / 34.69% | 9.36 / 45.42 / 76.90 / 34.49% | 38.31 / 44.39 / 18.79 / 33.71% |
| Router | 36.98 / 5.49 / 5.36 / 4.17% | 13.29 / 1.91 / 14.91 / 1.45% | 22.55 / 4.50 / 8.79 / 3.42% |

KDA's 49K-root profile breaks down to 177.92 ms input projections, 83.01 ms
prepare, 78.37 ms recurrence, 5.90 ms output norm, and 62.88 ms output
projection. At 80K, the 256-row breakdown is 34.29 / 9.83 / 10.20 / 0.73 /
8.99 ms; the 1,024-row breakdown is 92.74 / 39.33 / 40.07 / 2.61 / 32.48 ms.

Q6/Q8 logical projection bandwidths are shown as `Q6 / Q8 GB/s`:

| Cell | Dense FFN | Shared expert | KDA | MLA projection | MLA output |
| --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 + 2,048 | 14.99 / 0.00 | 13.10 / 0.41 | 9.17 / 0.30 | 1.20 / 9.87 | 7.40 / 2.28 |
| 81,920 + 256 | 109.23 / 0.00 | 69.30 / 2.19 | 58.46 / 1.92 | 7.28 / 59.68 | 58.81 / 18.09 |
| 81,920 + 1,024 | 29.40 / 0.00 | 26.19 / 0.83 | 18.06 / 0.59 | 2.48 / 20.34 | 14.37 / 4.42 |

The modeled Q6 bytes are 371.6 MB in dense FFN, 846.4 MB in the shared
expert, 3,743.4 MB in KDA, 51.6 MB in MLA projection, and 550.5 MB in MLA
output per cell. The corresponding Q8 bytes are 0, 26.7, 123.1, 423.4, and
169.3 MB. These figures describe logical weight shapes, not bus reads.

### Distance from roof and cell ranking

By modeled tensor utilization, mHC is farthest from the tensor roof at about
0.21%, followed by sparse index scoring at 0.42–0.59%, sparse attention at
0.79%, and the indexer key/cache path at 1.07–1.56%. These stages are small in
modeled FLOPs; mHC also takes under 4% of any cell wall. Dense FFN/shared
projections are closest at 34–57%, KDA is around 29–36%, and routed experts
range from 13% to 36%. There is no physical memory-bandwidth counter or
bandwidth roof in this probe.

By absolute sampled span in the 49K-root cell, routed experts (721 ms), sparse
attention (396 ms), sparse index scores (377 ms), and KDA (408 ms) dominate.
The 80K/256 cell is expert-heavy (245 ms); at 80K/1,024, experts (447 ms),
sparse index scores (231 ms), sparse attention (197 ms), and KDA (207 ms) are
the largest stages.

The stage intervals are not an additive wall partition: their per-sample sum
exceeds the sampled command-GPU duration by 303 ms (15.2%) at 49K/2,048, 42
ms (9.4%) at 80K/256, and 191 ms (17.4%) at 80K/1,024. This likely reflects
overlap or timestamp scaling between sampled encoders. The row times above
are useful for ranking, but summing them overstates the cell wall. Use the
ordinary wall for end-to-end comparisons.

## First candidate

The first candidate is a 4,096-row chunk with 128-row sparse-query tiles. At
top-8 over 288 experts, the mean occupancy rises from 56.9 to 113.8 rows per
expert, matching the MLX tile-plan transition discussed in the
[MLX comparison](2026-11-13-glm-prefill-mlx-comparison.md). This tests both
expert tile fill and the repeated sparse-index/attention work that dominates
the root and 1,024-row cells.

The Q6 F16 cache is a later candidate: the visible dense FFN and shared-expert
spans alone are about 89 ms in the 49K-root cell. A cache could also affect
Q6 projections inside KDA and MLA, but the 14–15 GiB residency cost and
three-session memory gate remain. No candidate has passed the required 1.25x
isolated-stage gate, paired wall gate, quality gate, or three-agent serve
gate yet.

## Artifacts

- Successful raw census and run metadata: `stage-attribution/`.
- Initial raw census before the MLA split: `baseline-census.jsonl`.
- Metal sample-cap failure: `stage-attribution/failed-sample-buffer-cap/`.
- MLX implementation and license comparison:
  [`2026-11-13-glm-prefill-mlx-comparison.md`](2026-11-13-glm-prefill-mlx-comparison.md).
