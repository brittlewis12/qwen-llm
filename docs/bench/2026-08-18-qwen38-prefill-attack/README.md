# Qwen3.8 27B Prefill Attack

Status: completed benchmark packet. This records the clean baseline and four
bounded follow-ups; it is historical evidence at the named source identities,
not a current-HEAD throughput claim.

## Contents

- `baseline.md`: clean Qwen3.8-27B Q4_K_M sweep from pp512 through pp32768,
  with matching llama.cpp rows and raw JSON.
- `f01-fused-swiglu-falsifier.md`: existing fused Q4_K SwiGLU loses outside its
  small-hidden eligibility; no source change retained.
- `f02-padding-cliff.md`: the N=321 tile cliff is real in isolation but bounded
  to about 1-2% on a production residual chunk; no source change retained.
- `f03-production-phase-trace.md`: production pp2048 attribution puts about
  87% of sampled GPU work in Q4_K matmul and only 7.6% in attention.
- `f04-n64-gate-inversion.md`: dirty-tree signal only. It is explicitly
  non-authoritative and does not justify a gate change without a clean rerun.
- `run.sh` and JSON: acquisition commands and raw observations.

## Authority

The baseline and F01-F03 are retained because they close or rank concrete
prefill hypotheses. F04 is retained only as an excluded attempt so its dirty
result is not rediscovered as evidence. The packet does not authorize a generic
fusion campaign, a Q4_K N64 policy change, or any residency mechanism.
