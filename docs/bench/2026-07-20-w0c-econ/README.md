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

---

# RESULTS (2026-07-20; medians of 5 fresh processes; same-invocation
ref/spec pairs; therm files in dir; one protocol note: original M1 cell
(pure defaults) could not emit timing because the normal probe bailed at
the state audit before printing — fixed by a committed harness change
(deferred gate bails, exit semantics unchanged), then M1 rerun; the five
pre-fix failing M1 runs are retained as m1-run*.err evidence of the
production path failing its own audit with REAL drafts on the code
prompt, kv 0.999699 / gdn 0.129 / conv 0.555, deterministic)

## T-cells: decode-side organization pricing (tg128, A3B)

| Cell | Config | t/s (median of 5) | Spread |
|---|---|---|---|
| T1 | default (concurrent-shared) | 110.57 | 110.33-111.17 |
| T2 | CONCURRENT_SHARED=0 | 101.01 | 100.66-102.01 |

Q1: the global interim org switch costs 8.6% decode (disjoint spreads).
=> production D1 fix must be verifier-prefill call-site alignment (adopt
concurrent-shared there), NOT the global flag. The flag remains the
bit-exact ORACLE config only.

## M-cells: verifier economics (mtp normal probe, spec 7, physical N8,
128 tokens; alpha from real MTP drafts; deterministic within cell)

| Cell | Config | Prompt | Total speedup (med of 5) | alpha | Verify ms/packet | Packet cost (serial transitions) | Emitted/packet |
|---|---|---|---|---|---|---|---|
| M1 | batched (CONTRACT-BROKEN) | code | 0.866x (0.865-0.872) | 0.771 | 53.2 | 5.78 | 6.35 |
| M2 | bit-exact (W0b config) | code | 0.711x (0.701-0.718) | 0.771 | 76.2 | 7.64 | 6.35 |
| M3 | batched (CONTRACT-BROKEN) | chat | 0.399x (0.395-0.406) | 0.246 | 52.6 | 5.73 | 2.72 |
| M4 | bit-exact (W0b config) | chat | 0.340x (0.338-0.342) | 0.263 | 76.0 | 7.60 | 2.84 |

(Serial denominators 9.2-10.0 ms/transition from each run's own MTP=off
reference. M4's alpha slightly exceeds M3's — consistent with D3 head
flips costing the batched path acceptances on near-ties.)

## Verdict

- ALL FOUR CELLS LOSE. A3B N8/D7 speculation is net-negative TODAY even
  on the contract-broken batched path (0.87x code / 0.40x chat). The
  correctness question is moot until the economics exist.
- Structural cause: the verify packet's expert FFN runs PER TOKEN in
  both configs (grouped verify FFN exists but is default_off), so the
  packet floor is ~5.8 serial transitions (batched projections only
  shave the mixer); bit-exact per-token mixer raises it to ~7.6 (+32%,
  the current price of correctness). Break-even at the current shape
  needs alpha ~0.89 sustained; the real drafter delivers 0.77 on code,
  0.25 on chat prose.
- Ceiling arithmetic: perfect-alpha at current shape = 8/7.2 ~ 1.11x.
  If grouped verify-FFN reached dense-class packet cost (~4-5 tr), the
  realistic prize at measured alpha is ~1.1-1.2x on code-like prompts
  and dead on prose without a better proposer. The handoff-era
  "A3B unlock ~1.29-1.36x" (v0.520) is STALE against today's 110 t/s
  serial decode — the denominator improved, compressing the margin.
- W1c Amdahl input (Q4): A3B verify is expert-FFN-bound; chunked GDN's
  verify-side ceiling on A3B is small. W1's remaining case is prefill
  (A3B GDN share of prefill unmeasured — one phase check would size it)
  and the derivation/oracle assets stand regardless.
- D1 production fix (call-site alignment) remains correct and cheap but
  no longer urgent on its own: it serves the oracle/testing lane and
  future promoted paths, not a currently-winning product path.

## Recommended A3B-lane priority (post-pricing)

1. Verify expert-FFN batching at N=8 (validate the existing grouped
   path; measure packet-cost movement) — the only lever on the 5.8-tr
   floor. Gate: packet cost <= ~4.5 tr before anything else matters.
2. Proposer quality on prose (PLD/DFlash-class or better MTP usage) —
   alpha 0.25 kills chat regardless of kernels.
3. Parity kernels (D2+D3) + D1 call-site fix — required for ANY
   promoted lane; sized by (1)'s outcome.
4. Chunked GDN (W1c) for VERIFY: deprioritized on A3B by Amdahl;
   prefill case pending a phase measurement.

No promotion claims; all numbers M4-Max, this build, these fixtures.
