# W1 HANDOFF: chunked/WY-form GDN — the MoE speculation unlock

You are a fresh session opening the second front of a two-swing program.
This document is your complete inheritance from the session that ran the
first swing (trellis quantization, T1-T8b, 2026-07-19). Read it fully
before acting. Work ONLY in this worktree (`~/code/qwen-llm-wy`, branch
`wy/w1-chunked-gdn`). A second live agent works in the main checkout
(`~/code/qwen-llm`) — never touch it.

## Mission

Derive and validate a chunked (WY/UT-transform) formulation of the Gated
DeltaNet recurrence that is GEMM-shaped or one-dispatch-per-layer, under
an explicit numerical contract, such that a packed multi-token verifier
can reproduce serial recurrent state on the MoE models (A3B first).

Two payoffs, honestly sized:
1. PRIMARY — A3B state-preserving packed verifier: unlocks ALL
   speculation (PLD / DFlash / drafters / MTP-class) on the MoE side of
   the family, currently embargoed. Honest ceiling from the old (later
   state-invalidated) A3B oracle: perfect-oracle 1.364x, replay-current
   1.294x decode (PERF-ROADMAP.md, v0.520 entry). NOT dense-27B's 2.60x
   — expert-union inflation across packed tokens is intrinsic to MoE
   verify economics.
2. SECONDARY — GDN prefill work unit on MoE/small-dense. Do NOT claim
   dense-27B prefill wins: gdn_step is ~2.6% of dense-27B pp4096
   (measured: 482 ms of ~18.4 s).

## The admissible design (the roadmap's own frozen words)

From docs/PERF-ROADMAP.md, force-ranked entry 7 + "Decisive gates":
- "Any new candidate must name a state organization that reproduces
  serial recurrent, convolution, KV, logits, and multi-transition
  continuation state before timing."
- "WY must be one-dispatch-per-layer or materially matmul-shaped under
  a numerical contract."
- "Serial-order packing needs a kernel-level dispatch/memory mechanism;
  checkpoint fusion and sparse suffix replay remain closed."
- GDN recurrence gate: "complete one-layer all-in gain >= 20%,
  projected prefill gain >= 5%, and exact state or an explicit
  numerical contract."
- "No charged product work follows without state passage and at least
  5% projected warm-decode movement."

## Prior kills that bind (verify citations before relying on them)

- v0.483 (PERF-LOG.md ~line 4245): host-loop chunk16 GDN kernel —
  killed on EXECUTION SHAPE; the log explicitly says it does "not
  falsify a real chunked delta-rule formulation."
- Carry-product chunk16 with Gram precomputation (PERF-ROADMAP.md
  ~line 2620): correctness PASSED but 27B pp1024 collapsed
  238.75 -> 124.34 t/s. So one real chunked implementation already
  lost on throughput. The unfalsified premise is SPECIFICALLY the
  one-dispatch / GEMM-shaped organization.
- v0.556 (PERF-ROADMAP.md ~line 204): the current A3B physical-N8
  verifier FAILS the state contract — 12-token code prompt keeps the
  128-token stream but fails resume numerics: KV cosine 0.999698864,
  continuation-logit cosine 0.999197009, max logit delta 0.437913 at
  token 128. This is the defect W1 exists to fix. Dense-27B PASSES
  the same contract (KV cosine 0.9999999124, v0.554) — dense is not
  blocked; this swing is about A3B/MoE.
- v0.532/v0.533 closed MTP N8 row-wave scheduling and duplicated-L2
  removal as sub-gate; v0.587 closed fixed committed-tail MTP history.
  None of these bar a WY formulation.

## Architecture facts (docs/PLAN.md ~line 122-140; verify in code)

- 27B: 64 layers, pattern [GDN x3, full-attn x1] x16 -> 48 GDN layers.
  GDN: 48 V heads, 16 K heads (V/K = 3), head_dim 128; conv1d k=4
  depthwise + SiLU front; scalar gate per head
  g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b);
  L2-norm Q/K INSIDE the kernel (eps 1e-6); state fp32
  [n_v_heads, 128, 128] — fp32 is non-negotiable (drift evidence).
- Projections are SEPARATED (in_proj_qkv, in_proj_z, in_proj_b,
  in_proj_a) — common loader-bug source.
- A3B (35B MoE) dims differ — read crates/qwen-llm/src/model.rs and
  loader.rs for exact A3B GDN head counts/layers before deriving.
- Serial reference implementations to trust: forward.rs (CPU oracle,
  bit-tight vs llama.cpp) and kernels/gated_delta_net.metal (GPU,
  production). The delta rule per token t, per V head:
    q,k L2-normed; S <- alpha_t * S + beta_t * (v_t - S k_t) k_t^T
  (verify exact order/gating in forward.rs — do not trust this line).

## The math to derive (W1a)

Chunked gated delta rule with WY/UT-transform structure. The training
literature computes this exactly:
- Gated DeltaNet paper (Yang et al., 2024/2025) trains with a chunked
  parallel form — the algebra exists and is exact.
- The canonical implementation is `chunk_gated_delta_rule` in the
  flash-linear-attention repo (github.com/fla-org/flash-linear-attention,
  Songlin Yang) — UT transform: within a chunk of C tokens, the
  Householder-like products (I - beta_i k_i k_i^T) with decay compose
  into T = (I + tril-strict A)^{-1}-shaped small matrices; per-chunk
  work becomes [C x d] GEMMs + one [d x d] state update per chunk.
- YOUR derivation must handle THIS parameterization exactly: per-head
  scalar decay alpha_t (gating), beta_t, L2-normed k, head_dim 128,
  fp32. Write it as docs/wy/W1A-DERIVATION.md with explicit shapes and
  a mapping to available kernel primitives (simdgroup_matrix mat-mat
  tiles exist in-repo; see kernels/mat_mat_mm_tile.h).
- cx (`cx ask`, defaults read-only sandbox; it HAS web search) is
  excellent for this: jam the derivation, then adversarially review it
  in a FRESH cx session before implementing anything.

## The validation ladder (W1b -> W1c), oracle-before-kernel ALWAYS

The first swing's decisive lesson: a cheap CPU oracle, validated
against the production implementation, predicted real behavior to
~0.5% three consecutive times and killed three would-be kernel efforts
before they wasted GPU time. Replicate that pattern:

- W1b — numerical-contract oracle (CPU, no GPU): capture REAL per-layer
  GDN input streams (q, k, v, alpha, beta after all projections/conv/
  norms) for a real prompt — either instrument forward.rs (CPU path,
  0.8B F32 exact) or add a bench-only readback. Implement serial fp32
  reference (forward.rs already is one) and chunked fp32 (C in
  {8, 16, 32, 64}); compare terminal state + per-chunk-boundary state +
  a multi-token CONTINUATION after the chunked segment, per the v0.556
  contract shape. The bar class: dense passed at KV cosine
  ~0.9999999; A3B failed at 0.9997. Preregister tolerances BEFORE
  running. Edge cases to include deliberately: alpha near 1 and near 0,
  beta near 0/1, long runs (128+ tokens) for drift accumulation.
- W1c — only if W1b passes: one-layer, ONE-DISPATCH GPU kernel oracle
  at production A3B shapes; roadmap gate >= 20% one-layer all-in vs the
  serial per-token kernel path, >= 5% projected prefill (A3B), state
  contract green. THEN the N=8 verifier integration question (packed
  verify lives in metal_dflash.rs / metal_mtp.rs).

## Operational inheritance (hard-won today — do not relearn)

1. WORKTREES: one per workstream. You are in yours. The main checkout
   has a live co-tenant agent; NEVER edit, commit, or `git add -A`
   there. Trellis swing lives in ~/code/qwen-llm-trellis (parked:
   LDLQ pilot next; branch trellis/t7-real-weight-oracle).
2. GPU: Britt granted this program exclusive GPU access (2026-07-19,
   trellis context). Re-confirm scope with Britt before your first
   timed GPU run, and check whether the co-tenant still runs timed
   work. CPU-heavy work: nice -n 19 + thread caps (env knob pattern)
   as courtesy regardless.
3. MEASUREMENT: near-memory-wall GPU ratios carry ±4-5% ACROSS
   invocations even on a quiet box (three evidence packets:
   T6/T7a-part1/T8b). Same-invocation pairs only; >= 5 invocations +
   spread rule for any 3%-scale gate; `pmset -g therm` before/after;
   record every tuning iteration, no silent best-of.
4. HOUSE METHOD: preregister (frozen gates, artifact README in
   docs/bench/YYYY-MM-DD-name/) -> run -> close with a verdict, every
   packet, even kills. Tuning budgets declared up front. Commit style:
   short prefix ("w1: ..."), body explains why, close-commits carry
   the verdict.
5. TOOLING TRAPS: python-heredoc edits must assert every replacement
   and write once at the end; commands on a new line after a heredoc
   run regardless of the heredoc's exit code (chain with && on ONE
   line); criterion overwrites new/ each invocation (harvest JSON or
   capture stdout per invocation); `-ffast-math` is on for kernels;
   build.rs recompiles kernels on any kernels/ change.
6. cx SESSIONS OF RECORD (resumable via `cx resume <id>`): 019f7bbc
   (portfolio collab), 019f7bc6 (portfolio adversarial), 019f7c2d
   (physics adversarial), 019f7c4b (Apple-GPU/trellis research incl.
   ALU rates: imul32 quarter-rate, fp16=fp32 rate, Apple9 cross-
   simdgroup pipe overlap), 019f7c67 (T1 review), 019f7ce2 (post-T7a
   jam incl. force-ranking that put THIS swing at #4).
7. The other program queue (context, not your job): trellis LDLQ
   pilot; attention/GDN fidelity coverage; measurement control plane
   (three evidence packets waiting for a fix packet).

## Suggested first hour

1. Read PLAN.md GDN section + forward.rs GDN serial code + 
   gated_delta_net.metal (the packed prompt variant especially) +
   model.rs/loader.rs A3B dims.
2. Launch cx research: exact chunked-gated-delta-rule algebra from the
   fla repo for THIS parameterization (per-head scalar gate, L2 k,
   beta), plus any published numerical-stability analysis of the
   chunked form at fp32.
3. Write W1A-DERIVATION.md; adversarial-cx it.
4. Preregister W1b (tolerances, chunk sizes, capture protocol, edge
   cases) before writing the oracle.

Iron sharpens iron. Preregister, let the gates decide, and have fun.
