# General F32 Grouped Expert Down

Follow-up to `eea15653`. Extends the existing grouped F32-operand down template
from IQ3_S/IQ4_XS to all 17 native formats with shared F32 decoders. The original
two down instantiations and fused gate/up arithmetic remain unchanged. Kernel
capabilities drive the GLM precision selector; no extra scratch or expanded
weights are needed.

This removes Maya's IQ3_XXS down-projection row fallback in explicit F32 prefill.
It does not change default Half prefill or serial decode. Mixed gate/up pairs
without a fused F32 tile continue to use the native row baseline, not half
operands disguised as F32.

## Paired Model Result

One resident Maya model, release binary, 512 identical prompt tokens, explicit
`FastPrecision::F32`, fresh sessions, warmup of each arm, then A-B-B-A. A is the
previous down-support selection (IQ3_S/IQ4_XS only); B uses the new capability.
The override is scoped and test-only, not a production environment switch.
GPU ownership was serialized; Metal validation was disabled for timing.

| Arm | Prefill milliseconds |
| --- | ---: |
| A1 | 5687.96 |
| B1 | 3086.29 |
| B2 | 3026.30 |
| A2 | 6262.36 |
| A mean | 5975.16 |
| B mean | 3056.29 |

The paired means show 1.955x throughput (48.9% less wall time) for this workload.
Both B runs beat both A runs. This is a short paired diagnostic, not a confidence
interval or a claim about every context length or quantization.

Final-logit KL from A1 to B was 3.813e-11; worst KL including three common
teacher-forced continuation steps was 2.219e-10. Warmup A included cold weight
faults (79.7 seconds) and is excluded from measured means. Raw measurements and
correctness results are in `maya-f32-down-ab.json` and its log.

## Validation

- 11 F32 expert tests passed with required GPU execution and Metal validation.
  Tests cover all formats, independent CPU decoding, K=2048/M=4096 IQ3_XXS,
  token/bucket tails, nonzero offsets, invalid IDs/counts, and large F32 values.
- Targeted GLM suite: 82 passed, 45 opt-in tests ignored; the paired model test
  was explicitly run separately.
- Complete expert suite: 16 passed, one timing test ignored (includes the 11
  F32 tests above). Existing fused/grouped paths remain covered.
- Workspace release checking passed, with the existing unrelated CLI dead-code
  warning. Independent source review found no concrete correctness issues.

## Retrospective

The performance gap was missing dispatch coverage, not an inherent cost of the
smaller artifact. Reuse the existing template and shared decoders, verify the
actual precision path, then measure a same-binary control instead of inferring
speed from weight size or comparing unrelated cold loads.
