# W0b: A3B packed-verify first-divergence trace (bug-hunt diagnostic)

Status: PREREGISTERED (protocol; a bug hunt has evidence categories and
close conditions, not promotion gates). Date: 2026-07-20.
Worktree: ~/code/qwen-llm-wy @ wy/w1-chunked-gdn. Follows W0
(../2026-07-20-w0-a3b-route-bisect/, verdict NOT-FIXTURE-SUFFICIENT).

## Question

W0 established: A3B packed-verify state diverges from serial even with
fully-serial per-token arithmetic (C-SER: gdn 6.2e-2 / conv 7.0e-1 vs
gates 1e-2/1e-1), while dense-27B passes fully batched. Where does the
divergence FIRST appear, and does its onset correlate with packet
orchestration events (partial-accept restores, checkpoint slots,
terminal handling) or accumulate uniformly per packet?

## Instrumentation (bench-only, this branch)

1. `PackedStepProbe` callback in `decode_packed_n_planned`
   (metal_mtp.rs): fires once after prefill (phase=Prefill) and once per
   verify packet after accept/restore (phase=Packet), carrying step
   metadata (start_position, n_eff, n_accepted, n_keep, restore_fired,
   stop_now, committed tokens) and `&MetalSession` (candidate). Fires
   only when a probe is installed; production paths untouched.
2. `--mtp-state-trace` on `qwen-bench mtp` (oracle probe only):
   maintains a SHADOW serial session in-process — prefilled with
   `mf.single_token` over the prompt, advanced at each packet by that
   packet's committed tokens at their positions — and prints per-packet
   per-layer max-abs deltas (all GDN state tensors, all conv tensors)
   candidate-vs-shadow, plus kv_n_pos equality. All fp32 host reads on
   unified memory; verify/restore paths waitUntilCompleted before
   return (verified metal_dflash.rs:11991-11992), so reads are
   coherent.

The oracle-draft structure gives a second free signal: oracle drafts ARE
the serial stream, so any rejection (n_accepted < n_draft) marks a
packed-vs-serial argmax flip at that position.

## Preregistered evidence categories

- E-PREFILL: nonzero state delta already at phase=Prefill => the defect
  is upstream of packets entirely (single_token_argmax_with_hidden vs
  single_token, or session init) — the packet machinery is exonerated
  for onset (not necessarily for growth).
- E-RESTORE: deltas are ~0 for full-accept packets and jump on the
  first restore_fired packet => checkpoint/restore bookkeeping
  implicated (slot indexing, conv window, KV rollback width).
- E-UNIFORM: deltas grow on every packet including full-accepts =>
  a per-packet op is not serial-identical under C-SER (inventory gap)
  — candidates: hidden bridge capture, get_rows/copy plumbing, the
  n_eff padding tokens' side effects.
- E-TERMINAL: divergence only at the stop_now packet => terminal
  n_keep/pending-token handling.
- E-MIXED: combinations; report the earliest onset class first.

## Protocol

Fixture F-code16 (W0's), config C-SER (BATCHED_MIXER=0 — maximal
serialization = minimal confound), plus one C0 run for contrast (does
batching change the ONSET class or only magnitudes?). One run each
first; the trace is deterministic per W0 (byte-identical reruns), so
single runs suffice for localization; any surprising row gets a
confirm rerun before being cited.

Runs write to target/profiles/w0b-a3b-first-divergence/, copied here at
close. Same manifest discipline as W0 (same model file; commit recorded
per run by the harness identity guard).

## Close conditions

Close with: (a) the earliest divergence class (E-*) with the trace
excerpt, (b) the implicated code path narrowed as far as the evidence
supports WITHOUT fix attempts in this packet, (c) the recommended fix
packet scope (W0c) or, if E-UNIFORM with no candidate, the escalation
plan (per-layer hidden capture diff inside one packet). No fix is
implemented under this packet's paperwork — localization only; fixing
under diagnostic momentum is how bugs get half-fixed.

## Non-claims

No timing evidence. No contract claims. The probe changes host-side
pacing (extra readbacks + shadow decode between packets) — irrelevant
to correctness comparisons, fatal to any timing quotation.

---

# RESULTS (2026-07-20; runs at commit a2e58f4-class HEAD, deterministic;
adversarial interpretation review: cx 019f7ff3-8ef0-7100-bca4-6d8f16e15f7d,
verdict SOUND-WITH-CAVEATS, caveats adopted throughout)

## Trace evidence (all in this directory)

1. trace-cser-code16: C-SER, default decode. Divergence seed ALREADY AT
   PREFILL (gdn 4.1e-5 / conv 1.8e-4, 27-29/30 layers) before any packet;
   packets grow it mildly (1.2e-4 by step 1). => E-PREFILL.
2. Root cause of the seed (D1): the engine has two serial MoE single-token
   organizations — "concurrent" (production decode default; used by the
   v0.55x audits' reference and by our shadow) and "plain"
   (single_token_argmax_with_hidden; used by the verifier's internal
   prefill). They differ ~1e-5/token on A3B.
3. trace-cser-noconc-code16 / -code128: with one organization
   (QWEN_DECODE_MOE_CONCURRENT_GDN=0) + per-token verifier mixer
   (BATCHED_MIXER=0), the audit is BIT-EXACT END TO END: kv cosine
   1.0000000000, every gdn/conv/kv/continuation delta exactly 0.0, on both
   16- and 128-token code fixtures. PASS.
4. trace-cser-noconc-chat128: state stays bit-exact through all 55 packets
   (acceptance patterns 0..7, restore-bearing and terminal packets
   included) — but the STREAM still diverges: with state bit-exact, the
   only remaining packed-vs-serial arithmetic is the batched
   final-norm/output-head/argmax path (D3), and chat-prose near-ties flip
   verify argmax (acceptance collapses to 0 before divergence).
5. trace-c0-noconc-code16: production batched mixer vs plain reference
   fails in ONE packet (gdn 2.7e-2, conv 2.4e-1, all 30 layers): batched
   mma8 half-staged projections + batched route (D2, bundled — which
   component dominates is untested; W0's C1 result weakly suggests the
   route KERNEL swap alone is not the driver).

## Findings (caveat-tightened language, binding for citations)

- F1. No orchestration defect was observed on the exercised restore paths
  (full accepts, partial accepts with restore, zero-accept packets,
  terminal packets); the v0.556 failure is explained WITHOUT invoking
  one. This is conditional on the verifier's committed token history
  (the contract's own shape) and does NOT prove every orchestration
  branch (e.g. terminal-with-rejection depth combinations not all
  exercised).
- F2. D1 (organization mismatch): concurrent-vs-plain serial MoE decode
  differ ~1e-5/token. BENIGN-VS-HAZARD IS OPEN: deterministic reruns rule
  out visible nondeterminism, not a deterministic missing-dependency/
  aliasing hazard. First-differing-KERNEL isolation (not layer) decides;
  if private-scratch/explicit-sync eliminates the difference, D1 is a
  bug to fix, not an organization to standardize.
- F3. D2 (batched mixer packet arithmetic): unacceptable for parity on
  A3B as-is; bundles half-staged mma8 and batched route; crossed
  interventions not yet run.
- F4. D3 is localized to the batched final-norm + output-head + argmax
  path as a WHOLE ("output-path decision arithmetic"); norm-vs-head-vs-
  tie-breaking attribution requires split experiments with top-2 margin
  logging.
- F5. v0.556's A3B kill was confounded by D1+D2+D3 (sequential,
  nonlinear — not an additive decomposition). Its "packed-MoE/GDN
  verifier" framing over-implicated the packed machinery.
- F6. A bit-exact A3B verifier configuration exists TODAY (per-token
  mixer branch + plain-organization decode). Its wall-time economics are
  UNMEASURED (approximately packet-width x serial mathematical work, but
  encoder overlap/weight-streaming effects unknown). Do not retire or
  promote it before measurement.
- F7. Dense-27B is not currently blocking under the frozen contract and
  tested fixtures — but the same mechanisms exist there sub-threshold
  (kv max-abs 3.1e-2 with passing cosine); shared fixes should cover it;
  "below today's threshold" is not "mechanism absent".

## Recommended follow-ups (not under this packet)

- W0c-econ: measure the bit-exact configuration's verify economics
  (packet cost in serial-transition units, acceptance-rate-weighted;
  plus plain-vs-concurrent serial decode cost) — this doubles as the
  post-W0 verifier-economics packet gating W1c.
- W0c-hazard: first-differing-kernel isolation of D1 (hazard vs
  reduction-order); Metal API validation + private-scratch A/B.
- W0c-parity: batched-shape GEMV kernel family (per-token serial-
  identical accumulation, single weight stream) for projections, router,
  and output head in verify packets — bit-parity is an EMPIRICAL kernel
  invariant requiring byte-equality tests (quant formats x widths x
  tails), with register-pressure/occupancy risk at vocab-wide 8-way
  accumulation. This is the plausible production fix for D2+D3.
- D3 split experiment with per-packet draft/target argmax + top-2 margin
  logging (extend the step probe).

Bottom line: the A3B speculation unlock is a NUMERICS-PARITY ENGINEERING
program with a working bit-exact oracle, not a mystery. Chunked GDN (W1)
remains a throughput play gated on W0c-econ.
