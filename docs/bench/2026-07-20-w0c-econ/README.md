# W0c-econ: A3B verifier economics after the W0b decomposition

Status: PREREGISTERED (frozen before any timed run). Date: 2026-07-20.
Worktree: ~/code/qwen-llm-wy @ wy/w1-chunked-gdn. Follows W0b
(../2026-07-20-w0b-a3b-first-divergence/, D1/D2/D3 decomposition;
bit-exact verifier configuration exists via flags).

## Questions (pricing only — no promotion, no contract claims)

- Q1: cost of the D1 interim organization switch on plain serial decode
  (QWEN_DECODE_MOE_CONCURRENT_SHARED=1 vs 0), A3B tg128.
- Q2: verify-packet cost in serial-transition units for (a) the
  production batched path (contract-broken; ceiling pricing) and (b) the
  bit-exact configuration (CONCURRENT_SHARED=0 + BATCHED_MIXER=0;
  contract-clean per W0b). Zero-overhead speculation floors for both.
- Q3: end-to-end decode movement of the real-draft MTP probe (normal
  probe, spec 7, physical N8) under (a) and (b), on a code prompt and a
  chat prompt (alpha differs by prompt class).
- Q4 (qualitative only): is the packet cost mixer/FFN-dominated or
  GDN-dominated — sanity input to W1c's Amdahl ceiling. No new
  instrumentation; verify_ms decomposition beyond stats fields is out of
  scope.

## Cells (all A3B = Qwen3.6-35B-A3B-UD-Q4_K_M MTP GGUF, exclusive GPU,
fresh process per run, sequential, pmset therm before/after batch)

Decode-side (qwen-bench tg, llama-bench tg128 semantics), x5 each:
- T1: default flags
- T2: QWEN_DECODE_MOE_CONCURRENT_SHARED=0

Verifier-side (qwen-bench mtp --mtp-probe normal --spec-tokens 7
--mtp-physical-n 8 --tokens 128), x5 each:
- M1: default flags, code prompt (W0 F-code fixture prompt)
- M2: bit-exact config (QWEN_DECODE_MOE_CONCURRENT_SHARED=0 +
  QWEN_MTP_MOE_VERIFY_BATCHED_MIXER=0), code prompt
- M3: default flags, chat prompt (W0 F-chat fixture prompt, --qwen-chat
  --disable-thinking)
- M4: bit-exact config, chat prompt

The normal probe computes its own MTP=off reference serially in the SAME
process/invocation — every speedup is a same-invocation pair (the
trellis-era nonstationarity rule is satisfied by construction for the
headline ratios; cross-cell comparisons use medians of 5 with spread).

## Metrics per run (harvested from stderr results block)

Decode-only t/s (ref and spec), total speedup, alpha, steps, verify_ms,
draft_ms, restore_ms; derived: serial ms/transition = ref_decode_ms /
ref_emitted; packet cost in transitions = (verify_ms/steps) / serial
ms/transition; emitted tokens/packet = (1 + alpha*7-ish, from stats);
zero-overhead floor comparison per the v0.556 method.

## Reporting rules

Median of 5 + min-max spread per cell; no cell-to-cell claim under 5%
unless spreads are disjoint; alpha reported alongside every speedup (a
speed ratio without its alpha is meaningless); M1/M3 numbers carry a
permanent "CONTRACT-BROKEN PATH (state-invalidated, ceiling pricing
only)" label; M2/M4 carry "bit-exact config (W0b)".

## Decision-rule outputs (informational)

- M2/M4 median end-to-end >= 1.0x => bit-exact speculation is
  free-or-better TODAY; recommend opening the integration lane and
  treating W0c-parity as pure upside.
- M2/M4 < 1.0x => the M1-M2 / M3-M4 gaps price the W0c-parity kernel
  program; report the gap as "the parity prize".
- T2-T1 delta prices the interim org switch for oracle/test use only
  (the production D1 fix is call-site alignment, not the global flag).
- W1c input: if verify packets are dominated by non-GDN work, chunked
  GDN's verify-side ceiling is small and W1 remains prefill-justified
  only (qualitative, from draft/verify/restore splits + known layer
  composition).

## Non-claims

No contract/promotion claims; no product-default changes; M1/M3 do not
revalidate the killed A3B lane; numbers are M4-Max/this-build specific.
