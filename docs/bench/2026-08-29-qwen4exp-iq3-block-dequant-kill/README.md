# Flash-Next IQ3 Block-Local Dequantization KILL

Decision: **KILL** the block-local IQ3_XXS packed gate/up rewrite. It is exact,
but the first valid candidate row makes the preregistered 10% leaf gate
mathematically unreachable, so the remaining bracket arms did not run.

## Candidate

Only `dequantize_iq3_xxs_half_grouped` changed. The candidate hoisted the IQ3
block scale, auxiliary word, four grid words, and two sign words out of the
16-value scalar loop while retaining the grouped kernel, dispatch geometry,
threadgroup layout, barriers, weight format, and arithmetic association.

The candidate remained uncommitted. Its release executable has SHA-256
`170e32e4a84afc22574e01101fa8e5c0edc6b5e463b972a5bde24222614254f6`; its
embedded metallib has SHA-256
`4ce2816683104ecde78a95c92ed5aa7bd687473756d3acc0baeb078aeea3e3eb`.
It was built from `4e8fcc0670e49b11f8b334cf7f8111902b044e05` plus that helper change.
The preserved A1 executable came from `e938a250fada78bd8b071619c521ec66547747a8`;
its metallib matches the post-KILL `4e8fcc0` rebuild exactly, isolating the GPU
kernel difference from the intervening inactive prefetch cleanup.

## Correctness

- A temporary compiled Metal probe compared scalar and block-local output as raw
  half bits for every `il=0..15` and all 16 values per residue.
- Six hostile blocks covered zero, subnormal, ordinary, maximum finite, and
  negative scales; zero/255 and mixed grid/sign patterns. Six released weight
  blocks covered beginning, interior, and final tensor positions.
- All 3,072 half values were bit-identical.
- Focused packed common-MoE serial/route coverage and the nonzero
  IQ3_XXS/IQ4_NL/Q8 CPU-dequant gate passed.
- The temporary probe kernel and Rust test were removed before performance
  timing; the candidate helper was then the only kernel-source change.

## Screen

The planned order was baseline/candidate/candidate/baseline at natural N=512.
The gate used the worst candidate row, so one valid candidate miss was already a
decisive futility stop.

| Arm | Layer-5 gate/up (ms) | Warm command GPU (ms) | Profiled command GPU (ms) |
|:---|---:|---:|---:|
| A1 baseline | 5.425000 | 991.498750 | 992.866333 |
| B1 candidate | 5.306417 | 987.057042 | 988.515208 |

Both processes used the tracked natural N=512 prompt, one packed command, zero
scalar tail, encoder-stage sampling with 128 samples, complete GPU timing,
accepted observers, and raw coverage 1.0. Their stdout digest is
`e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.

The candidate saves `0.118583 ms/layer`, or 2.19%, and exceeds the
`4.857600 ms/layer` ceiling by `0.448817 ms`. Its worst-case value can only stay
at or above B1 after another candidate arm, so the 10% gate cannot recover.
Warm/profiled command GPU improves by only 0.45%/0.44%, also below the 1% gate.
B2 and A2 therefore could not change the decision and did not run.

This is a preregistered futility KILL, not a completed drift bracket or a precise
effect estimate. The candidate, temporary baseline executable, and diagnostic
code were removed. The restored release metallib matches the baseline SHA-256
`0f3c521ee202872694171c4e8c5859609163cfcf98f379174b0013cf780c8097`.

Raw logs and executable hashes remain under
`target/profiles/qwen4exp-iq3-block-dequant-screen-20260829/`.

Adversarial review: `01a04de7-d66e-73f1-b3de-14ba60527c89`.
