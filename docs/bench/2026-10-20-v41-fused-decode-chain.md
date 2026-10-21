# V4.1 fused decode expert chain

Candidate `55b0157e` reduces routed decode from five passes to two. Gate and up
decode in one loop so each activation load feeds both banks, and the epilogue
applies the existing upper-only gate clamp, up clamp, SwiGLU and route weight.
A second kernel runs the down projection per slot, sums slots in order and adds
the shared branch, writing the final FFN output directly. Intermediate gate and
down-slot buffers are no longer written or read in decode.

The lane partition, per-lane FMA order, SIMD reduction and slot order are
unchanged. MXFP4 keeps the native chain because its subnormal rescaling is per
bank. Scalar and grouped prefill paths are untouched.

## Numerical witnesses

`metal::expert_native::tests::fused_chain` compares each fused kernel against
the existing GPU passes for every non-MXFP4 storage type at K=256 and K=5120,
including invalid IDs and signed-zero route weights. The fused gate/up/weight
output equals the native up output times the route weight bitwise; the fused
down/sum output equals native down followed by the production ordered-sum
kernel bitwise. All 18 native expert tests and both V4.1 native FFN composition
tests pass on M4 Max with Metal validation. The existing baseline arithmetic
source hash is unchanged.

## M5 ABBA

- M5 Ultra, Metal compiler `32023.921`, SDK 27.0, validation off, production
  lease, one fresh model load per arm.
- Baseline `4b1fca04` (`~/code/qwen-llm-v41-base`), candidate `55b0157e`
  (`~/code/qwen-llm-ours`), both `build-info` status `match`.
- Q2_K_XL and IQ3_XXS unsloth artifacts, `suite --tg 32 -d 2048 --runs 3
  --seed 1`, arm order base/candidate/candidate/base.

| Quant | base 1 | cand 2 | cand 3 | base 4 | base mean | cand mean | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 21.512 | 21.833 | 22.166 | 22.024 | 21.768 | 22.000 | **+1.06%** |
| IQ3 | 21.250 | 21.155 | 21.556 | 21.426 | 21.338 | 21.356 | **+0.08%** |

Within-arm SD is 0.03-0.08 tok/s except IQ3 base 1 at 0.33. The box drifted
upward during each pair (Q2 base +2.4%, candidate +1.5%; IQ3 +0.8%/+1.9%), so
the adjacent pairs are the useful comparisons: Q2 +1.49% and +0.64%, IQ3 -0.44%
and +0.61%. The later pair is conservative because it runs the candidate first.

## Interpretation

Dispatch count and intermediate traffic were not the dominant decode cost. The
change is kept because it is bitwise, consistently non-negative on Q2 and a
platform for tuning the fused kernel's work per thread. Reaching the remaining
decode gap needs a GEMV geometry with more independent work per SIMD group and
fewer activation loads, not fewer dispatches.
