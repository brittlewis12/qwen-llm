# W1 program close: chunked/WY GDN killed at its own gates; A3B lane repriced

Date: 2026-07-20. Branch wy/w1-chunked-gdn (W0/W0b/W0c packets + W1A
derivation, 14 commits). Verdict: KILL the chunked-GDN throughput swing
(W1b oracle NOT built, W1c kernel NOT started — by measurement, not
opinion). The program's diagnostic packets replaced the original mission
with a sharper, fully-priced map.

## The three-front measurement against W1's own roadmap gates

Gate (PERF-ROADMAP, GDN recurrence): one-layer all-in >= 20%, projected
prefill >= 5%, exact state or explicit numerical contract.

1. VERIFY/DECODE: A3B decode phase — GDN recurrence tail = 6.0% of
   token time (phase-a3b-ctx1024); A3B verify packets are expert-FFN-
   bound (W0c-econ). Amdahl kills the verify-side case.
2. PREFILL: A3B pp1024 GDN-split ladder — recurrence step 4.6-7.5% of
   prefill; FREE-step ceiling 8.1%; realistic 2x-step gain ~3.9% <
   the 5% gate. Dense-27B was already 2.6%-class. No family member
   measured today clears the gate.
3. CONTRACT: the A3B state-contract failure W1 was framed to fix is NOT
   a recurrence-formulation problem: W0b decomposed it to D1 (serial-
   organization mismatch at the verifier-prefill call site — diagnosed
   to encode_moe_ffn_apply_gpu_concurrent_shared vs plain, not a
   hazard), D2 (batched mma8 packet projections + route), D3 (batched
   output-path decision arithmetic). A BIT-EXACT A3B verifier
   configuration exists today (kv cosine 1.0000000000:
   QWEN_DECODE_MOE_CONCURRENT_SHARED=0 + QWEN_MTP_MOE_VERIFY_BATCHED_
   MIXER=0) and is the regression oracle for any parity work.

## The repriced A3B speculation lane (W0c-econ, medians of 5)

- Batched (contract-broken): 0.866x code / 0.399x chat.
- Bit-exact (today's flags): 0.711x / 0.340x.
- Packet floor 5.8 serial transitions (7.6 bit-exact) — per-token
  expert FFN; grouped verify-FFN LOSES at N=8 (76.3 vs 53.2 ms/packet);
  alpha 0.771 code / 0.246 chat (proposer-limited on prose).
- Perfect-alpha ceiling at current shape 1.11x; with dense-class
  packets ~1.2x at measured code alpha. The v0.520-era 1.29-1.36x prize
  is STALE (serial decode reached ~110 t/s; margin compressed).

## What survives (shelf assets, all committed)

- docs/archive/wy/W1A-DERIVATION.md: exact chunked WY/UT algebra in engine
  conventions, adversarially reviewed. Correct regardless of economics.
- The bit-exact verifier oracle configuration + per-packet state probe
  (--mtp-state-trace) + deferred-bail timing harness.
- D1 production fix, scoped: align single_token_argmax_with_hidden's
  MoE FFN organization with production decode (call-site change; the
  global flag costs 8.6% decode and is oracle-only).
- The union-expert batched-shape GEMV convergence: ONE kernel family
  (stream each active expert's weights once per packet, accumulate per
  token in serial-identical order) attacks BOTH the parity defect
  (D2/D3) and the packet-cost floor. It is the sole recommended entry
  point if the A3B verifier lane reopens, gated on: projected packet
  cost <= ~4.5 transitions AND a prose-capable proposer plan
  (alpha >= ~0.5) — otherwise the lane stays economically dead.

## Reopen conditions for chunked GDN specifically

- A family member whose GDN-step share of prefill or decode is
  structurally >= 15-20% (none measured today: dense 2.6% pp, A3B
  4.6-7.5% pp / 6% decode), or
- a workload requiring one-dispatch multi-token GDN with GEMM shape for
  reasons other than raw share (e.g. a future training/finetune-on-
  device lane, where the chunked form is the standard parallelization
  and W1A-DERIVATION.md is the starting document).

## Method note

Every verdict above is from preregistered packets closed same-day
(W0 v3 after two adversarial redesigns; W0b SOUND-WITH-CAVEATS with
caveats adopted; W0c-econ with a declared harness amendment), with
deterministic reruns where claimed and artifacts in docs/bench/. The
original mission statement ("chunked GDN unlocks A3B speculation") was
falsified in its causal premise and its economic sizing — the honest
outcome of packet zero being a diagnostic instead of a derivation
sprint.
