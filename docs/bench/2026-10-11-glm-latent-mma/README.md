# Grouped Native Latent Matrix Execution

Base: main `27b49476`. The refreshed leverage map initially ranked this bounded
GLM prefill optimization before a wider expert-down experiment and deep attention.
Maya's 11 executed MLA blocks use Q6_K absorption/expansion; UD uses Q8_0 and
provides an unchanged-path control.

## Mechanism

Reuse the existing Q8 grouped F32 matrix geometry with the shared native Q6_K
decoder: 16 output channels by 128 tokens, stepping through K in blocks of 64.
Both real shapes (K256/M512 and K512/M256, 64 groups) are eligible. Weights stay
compressed; operands and accumulation remain F32. Only 4096 bytes of transient
threadgroup memory are used, with no persistent allocation or repacking.

Complete 128-token prefixes use the matrix kernel and remaining rows use the
existing scalar kernel. Other formats, unsupported geometry/device capacity,
and single-token calls retain their existing paths. Q8 dispatch is unchanged.
Support remains general: this optional optimization is not an admission gate.

## Numerical Development

The first matrix reduction exceeded the existing pointwise tolerance at
K512/M256/groups64/N128, last row, channel index 241:

- Scalar decode: 0.4223144.
- Initial matrix output: 0.42233974.
- Independent decoded F64 dot: 0.4223148670280934.

The original sampled oracle missed this channel. The oracle now checks every
channel of the last row in addition to boundary-token samples. Matrix reduction
now accumulates a fresh partial per K64 block, then adds it to the F32 total.
All thresholds are unchanged. All eight grouped-latent tests passed under Metal
validation, including real shapes, offsets, guards, group isolation, ragged
prefix/tail selection and large F32 activations.

## Whole-Model Measurement

Same binary, one model load, default Half prefill (the latent operands remain
F32), fresh sessions, warmup A/B then measured A-B-B-A per cell. A forces the
scalar Q6 path with a scoped test-only override; B permits matrix execution.
Untimed warmup census verifies the selected paths. Timed runs exclude loading,
session allocation, prefix preparation and census overhead.

Cells: full 512 and 2048-token prefill, 129-token suffix after 512 tokens, and
512-token suffix after 4096 tokens. Identical tokens and three common decode
steps compare finite full-vocabulary logits against A1. No performance threshold
is embedded in the diagnostic. The original forced Half/MMA test recorded all
cells before failing its 1e-3 numerical bound; that failure remains retained.

| Forced Half/MMA cell | Scalar mean ms | Matrix mean ms | Throughput ratio |
| --- | ---: | ---: | ---: |
| Full 512 | 2613.30 | 2396.86 | 1.090 |
| Full 2048 | 9956.72 | 9110.11 | 1.093 |
| Suffix 129 after 512 | 1222.25 | 1179.99 | 1.036 |
| Suffix 512 after 4096 | 3002.11 | 2761.52 | 1.087 |

The largest old/new KL was 0.01287. The deep suffix stayed below 1.69e-5, but
both arms started from an identical scalar-built prefix; this does not establish
the behavior of a complete matrix-built 4K trajectory. UD Q8 control outputs
were bit-identical; its ratios were 0.997/0.997/0.998/1.009 respectively.

## Precision Probe and Production Decision

One fixed prompt at each of 512 and 2048 tokens compared five configurations:
native Exact forced scalar; Half scalar/matrix; and full-F32 scalar/matrix.
The native Exact reference also explicitly disabled matrix selection, preventing
reference contamination. F32 timing used warmups then ABBA. Raw JSON includes
KL, top-token changes, Exact-side regret and common-token log probabilities.

| Full-F32 cell | Scalar mean ms | Matrix mean ms | Throughput ratio | Max scalar/matrix KL |
| --- | ---: | ---: | ---: | ---: |
| 512 | 3269.76 | 3091.02 | 1.058 | 2.99e-10 |
| 2048 | 10872.43 | 10021.55 | 1.085 | 4.80e-5 |

At 512, both F32 arms stayed within 5.27e-10 KL of scalar Exact. At 2048, the
matrix arm stayed within 7.37e-5. No top-token changes occurred in either probe.
Half comparisons against Exact were mixed: neither scalar nor matrix was
uniformly closer. This supports precision-dependent downstream amplification,
not isolated routing causality or a task-quality claim. The original 1e-3
old/new diagnostic bound is not the historical Exact/Fast investigation trigger
and is not a model-admission criterion.

GLM production therefore permits this matrix path only for Fast lineage with
explicit full-F32 precision. Half, DenseF32, Exact and decode retain scalar4;
Q8 remains unchanged. The shared primitive exposes an explicit caller policy
and retains its full native baseline coverage. Test-only overrides reproduce
the original forced experiment without adding production environment switches.

## Arithmetic-Preserving Alternative

A separate scalar8 experiment doubled reuse of decoded weights while keeping
each output's lane assignment, FMA sequence and SIMD reduction unchanged.
Both synthetic outputs and complete model logits/continuations were bit-identical.
Whole-phase ABBA ratios were 0.996/1.006/1.001/1.002 on the four cells above:
neutral, not a demonstrated speedup. Scalar8 host selection is test-only and
defaults off. Its kernel and diagnostic remain available to reproduce this
falsifier; it is not a production optimization.

## Final Checks and Next Leverage

Required Metal execution with validation: 10 grouped-latent tests, 86 GLM tests,
and 15 HC tests passed. Fifty opt-in model/timing tests in those suites remained
ignored; the model experiments described above were separately enabled. The
failed Half experiment is not included in the passing-test claim. Independent
review found no concrete integration blocker.

The next implementation bet is high-count grouped expert down at the actual
serving width. Do not keep iterating scalar token counts after a neutral result,
or turn a kernel-level numerical comparison into an unsupported quality claim.
