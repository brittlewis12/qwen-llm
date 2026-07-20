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
