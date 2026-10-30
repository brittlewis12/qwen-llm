# GLM Q8 MLA Tail Batch: Limited Native Conversion

## Decision

**Park; no promotion and no further large quality run.** Ordinary model
conversion is not compelling despite promising tiny debug leaf timing.
Candidate `d1b897cb` remains off main/default-off, not installed; retain the
experimental branch for replay. This is a limited/inconclusive conversion
result and negative promotion decision, not a proven regression or a verdict
that all tail paths are worthless. It is separate from the core GLM defaults.

## Frozen Run

- M5 Ultra, actual six-shard GLM-5.3-Flash UD-Q4_K_XL, SDK 26.2,
  Metal compiler 32023.864; frozen optimized release source `d1b897cb`.
- Executable SHA256:
  `a22d6f2a7ad1d4f54b4463a46d847f6f76962c7949f4a1b398cc6553b0641ce9`.
- Build/runtime `git-source-sha256-v2`:
  `3697d2cd11d86dc7acf280e9454c3612378e38c466e132b857664e1f7b1c8296`.
- Order **A0, T0, T1, A1**, one fresh process each, 15-second pauses.
  Wide and Q5 down are fixed **on** throughout; only
  `QWEN_GLM53_Q8_TAIL_BATCH` changes: A0/A1=0, T0/T1=1.
- N2/127/128/129 at depths 0/4096, seed 1, two runs per shape, Fast/Half,
  admitted chunk 2048 for every row. Depth is live, untimed fill, not RAM
  snapshot restore; native inputs are synthetic, not saved requests.
- Ordinary prefill wall includes encode/wait/final-logit readback, excludes
  fresh-session allocation/model load. No Metal debug, census, profiler or
  capture; not network/cache/restore timing or serving TTFT.

All four arms succeed: **32 clean matching report rows / 64 samples**, no
identity overrides/problems; report hashes match events. Completion is
**412.196776125 s**, within the 600 s driver / 660 s outer timeout. Recorded
inter-arm gaps are about 15.010 s. No sample is excluded or replaced.

## All-Sample Medians

Each median uses all four observations per policy/shape. Positive reduction
means less wall (`100 * (1 - candidate / baseline)`). These are descriptive,
**not robust speedup estimates**; all samples remain in
[native-samples.csv](native-samples.csv).

| Depth | N | Baseline ms | Tail batch ms | Wall reduction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2 | 119.608 | 119.872 | -0.2210% |
| 0 | 127 | 717.326 | 708.309 | +1.2571% |
| 0 | 128 | 666.964 | 667.412 | -0.0672% |
| 0 | 129 | 678.420 | 677.718 | +0.1035% |
| 4096 | 2 | 87.737 | 86.658 | +1.2297% |
| 4096 | 127 | 746.053 | 753.773 | -1.0348% |
| 4096 | 128 | 718.286 | 733.340 | -2.0957% |
| 4096 | 129 | 751.075 | 790.118 | -5.1983% |

## Process Means And Limits

Means in ms retain both observations per process/shape, including drift:

| Depth | Arm | N2 | N127 | N128 | N129 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | A0 | 119.759 | 681.843 | 640.654 | 649.113 |
| 0 | T0 | 120.044 | 673.009 | 640.753 | 650.884 |
| 0 | T1 | 119.852 | 745.663 | 697.824 | 711.830 |
| 0 | A1 | 119.400 | 756.132 | 699.711 | 707.260 |
| 4096 | A0 | 82.906 | 696.069 | 653.275 | 663.261 |
| 4096 | T0 | 82.331 | 733.947 | 733.340 | 794.939 |
| 4096 | T1 | 91.944 | 773.169 | 741.497 | 788.475 |
| 4096 | A1 | 93.283 | 822.084 | 792.246 | 846.301 |

Early fresh A0/T0 N127 means fall 681.843 -> 673.009 ms (1.2956%), while
N128 stays 640.654 -> 640.753 ms. N128/N129 do not select tail batching
(remainders 0/1), yet drift materially. Later/depth state drift weakens
attribution; no frequency trace identifies its cause or proves constant clocks.
Neither R2 nor sparse-frontier N127 has a demonstrated robust model gain.

Tiny M4/M5 bitwise/guard witnesses passed on synthetic fixtures only. The
debug complete 255-row pair's approximately 9.4 -> 4 ms (about 2.4x) does
**not** convert to a high-leverage ordinary model win. Do not pool leaf debug
timing with these native results. The native benchmark retains neither full
causal state nor logits for a quality comparison: no new full-model numerical
quality or equal-state proof follows. See the
[leverage map](../2026-10-30-glm-m5-leverage.md) for the separate tiny witnesses.
