# GLM-5.3-Flash M5 Transfers: Native Performance Checkpoint

## Status

**Performance only; strict numerical qualification remains pending.** Actual-M5
native A/B measurements support further qualification of independent wide-Q8
dense and Q5-down candidates on the current UD-Q4_K_XL artifact. Neither flag
is promoted, no default changes, and these results alone do not qualify source
integration. This packet adds documentation only.

At 2K/4K, median native wall falls 3.3526%/3.1844% for wide-only and
8.3784%/7.5065% for Q5-down-only. N512 does not improve. Late-run drift limits
the precision of the down estimate; these are not universal speedups or
full-model quality results.

## Frozen Identity And Artifact

- `qwen-bench` source: `0473b95c2e299bd39794d0350c5e18d60718d146`.
- Executable SHA256:
  `dbd6532a39444d7ef9a821636da8012fee49ccb501ad2015b6c3b54f4ab62ea6`.
- Build/runtime source state, `git-source-sha256-v2`:
  `7e7c8c0334c840c383ef31290cf12663ea598a59780b40f15501a0bb850e5456`.
  All 24 report rows have matching clean build/runtime identities, with no
  overrides, identity problems or stamp errors. Report-file hashes also match
  the campaign events; this is not verification of model payload hashes.
- M5 Ultra, 80 GPU cores, 256 GiB, macOS 27.0.1. Metal SDK **26.2**, Apple
  Metal compiler **32023.864**. This is not the V4.1 SDK 27 release cohort.
- Actual six-shard GLM-5.3-Flash UD-Q4_K_XL; reported `model_size` is
  **199697800568 bytes**. Payload hashes are **not verified**. Detailed model
  pin data belongs to the pending numerical screen; no artifact-equivalence
  claim is made to the older UD-IQ3_XXS landing packets.

The current artifact's main routed roles differ from those older packets:

| Role | Main layers 0-44 | MTP layer 45, unused here |
| --- | --- | --- |
| Down | 39 Q5_K; 3 Q6_K at layers 11, 12, 44 | Q5_K |
| Gate/up | 82 Q4_K; 2 Q5_K at layer 11 | Q4_K |
| MLA K/V | All 22 main K/V tensors Q8_0, at layers 3, 7, ..., 43 | Not exercised |

This inventory describes candidate coverage, not a completed numerical or
payload-integrity gate. Do not substitute the old IQ3 role inventory.

## Native Measurement Boundary

Each fresh process runs N512, N2048 and N4096 at depth zero, seed 1, two timed
runs per shape. Inputs are synthetic token sequences, **not an actual saved
request**. The admitted chunk is 2048 for every report row, including N512;
the CSV's `prefill_rows` means admitted chunk size, not prompt length.
Production Fast/Half policy is used, with reported mode
`packed_fast+last_logits`. There is no Metal debug layer, GPU capture,
profiler or dispatch census in this timed cohort.

Wall includes ordinary full-model encoding, wait and final-logit readback.
It excludes fresh-session allocation, model load, network, cache and restore.
It is not serving TTFT, whole-response latency, a cached turn, or a stage-only
GPU interval. No GPU-time decomposition or decode benefit is inferred.

The independent blocks are never combined:

| Block | Process order | Baseline flags (wide, Q5 down) | Candidate flags |
| --- | --- | --- | --- |
| Wide | A0, W0, W1, A1 | A0/A1: (0, 0) | W0/W1: (1, 0) |
| Down | A2, D0, D1, A3 | A2/A3: (0, 0) | D0/D1: (0, 1) |

Flag names are `QWEN_GLM53_DENSE_Q8_WIDE` and
`QWEN_GLM53_TENSOR_Q5_DOWN`. Each arm is one fresh process, not a replicate
within one long-lived model session. There are four observations per
policy/shape within each block, 24 report rows and **48 timed samples** total.
Keep the two baseline blocks separate; do not add the candidate percentages
or claim a measured combined configuration.

Arms are separated by a requested 15-second pause (recorded gaps about
15.010 seconds). The driver budget is 600 seconds, with a real 660-second
outer timeout. The cohort completes in **409.656 seconds**, with every arm
successful and no forced kill. No timed sample is dropped or replaced.

## All-Sample Medians

Medians below use all four observations per policy/shape. Wall change is
`100 * (candidate / baseline - 1)`; negative means less wall time.

| Block | Prompt tokens | Baseline median ms | Candidate median ms | Wall change |
| --- | ---: | ---: | ---: | ---: |
| Wide | 512 | 1238.142 | 1238.465 | +0.0261% |
| Wide | 2048 | 3053.629 | 2951.253 | -3.3526% |
| Wide | 4096 | 6482.452 | 6276.026 | -3.1844% |
| Down | 512 | 1257.496 | 1259.810 | +0.1840% |
| Down | 2048 | 3115.793 | 2854.739 | -8.3784% |
| Down | 4096 | 6606.264 | 6110.363 | -7.5065% |

Both candidate processes, and all their individual samples, remain faster
than every baseline sample in their respective block at 2K and 4K. That
separation is useful performance evidence, not a confidence interval or
evidence that 8.3784% is a precise universal Q5-down gain.

## Drift And Observer Sensitivity

Per-process means in ms retain the order-dependent shift; each cell averages
two samples. [native-samples.csv](native-samples.csv) retains every individual
wall/rate value, policy, shape, identity status and observer marker.

| Arm | N512 mean ms | N2048 mean ms | N4096 mean ms |
| --- | ---: | ---: | ---: |
| A0 | 1236.581 | 3051.366 | 6482.452 |
| W0 | 1238.752 | 2951.780 | 6274.226 |
| W1 | 1238.465 | 2951.169 | 6276.393 |
| A1 | 1240.002 | 3058.084 | 6483.757 |
| A2 | 1238.455 | 3055.659 | 6487.223 |
| D0 | 1239.232 | 2780.313 | 5941.534 |
| D1 | 1289.995 | 2937.866 | 6272.852 |
| A3 | 1286.020 | 3175.829 | 6739.205 |

D1 and A3 are slower even at N512, where neither candidate improves wall:
about 4.1% versus D0 and 3.8% versus A2, respectively. The late down-block
shift is therefore not hidden behind the pooled median. Recorded thermal
and performance warning fields are false in all reports; **that does not
establish constant frequency**, power, thermal state or absence of drift.
No frequency/power trace or GPU capture is available to attribute the shift.

During A0's N4096 shape, one one-second read-only CPU process sample checked
initial quiet progress. The CSV marks both samples in that shape with
`external_cpu_sample_in_shape=1`; it does not assert which timed sample
overlapped the observer. All samples remain in the primary result. Using
only A1 as the wide 4K control still gives a **3.2039%** median wall reduction,
so the observed wide benefit is not dependent on retaining A0.

## Qualification Boundary

The shared tiny M5 wide/Q5 leaf witnesses passed with Metal debugging enabled.
They are separate from these debug-off full-model timings and are **not a
full-model quality witness**. An earlier identity refusal and two actual-Q4
pin refusals occurred before model load, outside this completed cohort; their
records remain private. They are not timed failures or passing qualification.

The corrected numerical screen plus existing 22-document/tool extension builds
successfully from clean `5081fcc0`: 23 focused CPU tests pass from its release
binary, and independent source review has no remaining P1/P2 findings. Its
separate test-executable SHA256 is
`801872a93e3e1e6c440555cb2a07b010e43822b00759c56b8c2fd1bfb1e7ead4`.
GPU execution has not yet occurred because the GPU is occupied by parallel
work. Do not infer passing full-model quality from the build, CPU checks or
earlier leaf witnesses. A later numerical qualification update must close the
gate before flag/default promotion or source integration. This performance
checkpoint does not close it.

Depth/ragged append behavior, saved-request continuation/tool behavior,
full-model quality, combined candidates and serving/decode conversion remain
unmeasured here. See the [leverage map](../2026-10-30-glm-m5-leverage.md) for
historical anchors and distinct remaining candidates.
