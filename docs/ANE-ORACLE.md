# ANE Prefill Co-Processor Oracle (P0-P4)

Status: **P0 preregistered** (this commit). Branch `ane-oracle`; integrate to
main only through the gates below. Every rung is killable; a kill merges this
doc + artifacts + a PERF-LOG entry and removes experiment source.

## Objective lane and boundary

Approximate/concurrency lane, prefill only, serial BS=1. The hypothesis is
NOT "ANE matmul is faster than GPU" (it is not: ~3.2-7.3 TFLOP/s vs our 12.8
mat-mat anchor). It is:

> Statically-shaped, stateless projection classes can execute on ANE
> concurrently with dependent-path GPU work during chunked prefill, producing
> a faster complete phase after charging staging, synchronization, precision
> conversion, compilation, and unified-memory contention.

Decode is out of scope (blocking ~95-125 us dispatch; bandwidth-bound on a
weaker fabric). GDN recurrence/conv, attention bodies, and dynamically
bucketed routed experts are out of scope (stateful or dynamic shapes).
Drafter-on-ANE remains a separate lane behind speculative economics
(PERF-ROADMAP.md "ANE/AMX drafting" boundary), untouched by this program.

## Prior evidence (external, commit-pinned)

Source: rustane @ c422447 + ncdrone/ane @ 016b754 (fp16-only static graphs,
conv1x1/matmul, NEON f32<->f16 staging, `_ANEInMemoryModel` direct dispatch).

- ANE conv1x1 768->3072 w=512: 7.315 TFLOP/s; 3072->768: 3.238 TFLOP/s
  (expansion vs reduction asymmetry) [rustane results/ane_tflops_m4max.md].
- Dual-load: ANE -1.5% / GPU -2.6% mutual degradation [results/dual_load_m4max.md].
- Staging: ~90 GB/s flat, ~10 GB/s interleaved; locks ~0.5-0.7 us
  [results/iosurface_staging_m4max.md].
- Compile ~50-90 ms/kernel; no cross-process plan cache (upstream: fixed-path
  override "not viable").
- Efficiency cliff reported at model dim 5120 (4.7x/layer) — measured on
  model-forward, NOT on isolated matmul; P1 must decide which.
- Perf-stats regression (`hwExecutionTime`=0) reported on macOS 26; this host
  is macOS 15.6.1 — expected functional, verify in P1.

## Environment

M4 Max 128GB (zekrom), macOS 15.6.1, worktree @ 81481491c (clean stamp).
Anchors: stream 474.0 GB/s, Q4_K mat-mat 12.80 TFLOP/s.
Models: Qwen3.6-35B-A3B-UD-Q4_K_M (sentinel), Qwen3.6-27B-Q4_K_M (dense
guardrail), Qwen3.5-122B-A10B-UD-Q4_K_XL (heavy, if P0 warrants).

## Offloadable phase classes (fixed vocabulary for this program)

From `QWEN_PREFILL_TRACE_LAYER_PHASES` labels:

| Class | Phases | ANE shape class | Rate prior r (vs 12.8) |
|---|---|---|---|
| OFF-EXP | `gdn_qkv`, `gdn_z` | expansion | 0.57 (7.3/12.8) |
| OFF-RED | `gdn_back` (= out_proj) | reduction | 0.25 (3.2/12.8) |
| OFF-SMALL | `gdn_beta_alpha`, `shared_packed` | small/skinny | 0.25 (conservative) |
| OFF-ATTN | attn qkv/o projections (decomposed via `QWEN_PREFILL_TRACE_ATTN_PHASES`) | mixed | 0.40 |
| DENSE-FFN | 27B `ffn_*` gate/up (expansion), down (reduction) | mixed | 0.57 / 0.25, **contingent on P1 5120-cliff test** |
| NOT-OFF | everything else (recurrence, conv, attention body, routed experts, norms, scatter, residuals) | — | — |

## P0 — Amdahl ceiling from measured phase shares (this rung)

**Design.** For each model (A3B, 27B; A10B optional): `qwen-bench pp` at
pp1024 and pp4096, `--runs 3`, layer+attn+ffn phase traces on, stderr logs to
/tmp, summarized by `scripts/profile/prefill_phase_summary.py --last-pass`.
Cross-check: untraced `pp` wall at the same shapes to record the serialized
trace-mode distortion factor (traced GPU sum / untraced wall).

**Known bias, declared:** serialized flush attribution removes existing
concurrent-encoder overlap (e.g. v0.340 GDN front concurrency), so measured
shares overstate the marginal win of removing that work from the GPU. P0's
output is therefore an **upper bound**. The bound is:

- Per class i: serialized share x_i, ANE relative rate r_i (table above).
- Choose offload set S maximizing sum(x_i) subject to ANE keep-up:
  `sum_{i in S}(x_i / r_i) <= 1 - sum_{i in S}(x_i)`.
- Ceiling = `1 / (1 - sum_{i in S} x_i)`, reported per model per shape.
- Stress variant: same computation with every r_i halved (staging/contention
  penalty proxy).

**Preregistered gates (decided before the runs below):**

- PROCEED to P1 iff, on at least one model, ceiling(pp4096) >= **1.15x** with
  prior rates AND >= **1.08x** under the halved-rate stress.
- KILL the lane (record + merge) iff no model clears; reopen condition: new
  measured ANE rates materially above priors, a new offloadable class, or a
  fabric-level API change (public stateful ANE, Metal-IOSurface zero-copy).
- Host-model selection: highest stressed ceiling wins P1's real-shape list.

**P0 does not authorize:** any qwen-llm production code, any ANE dependency
in the workspace, any quality claim.

## P1 — Shape truth (pinned sibling harness, no qwen-llm coupling)

Standalone crate `spikes/ane-oracle` (worktree-only; NOT a workspace member;
own lockfile; `ane` pinned by rev to 016b754). Random fp16 data.
Shapes: the P0-selected model's real projection shapes, e.g. A3B
qkv X[1024,2048]xW[2048,8192], z [1024,2048]x[2048,4096], out_proj
[1024,4096]x[4096,2048], shared expert; 27B contingent shapes incl. isolated
5120-dim matmuls (**cliff test**: model-forward artifact vs intrinsic).
Measure: compute-only TFLOP/s (wall clock; `hwExecutionTime` if live;
`powermetrics` ANE residency corroboration if sudo available), compile
ms/kernel, dispatch overhead amortization at w in {512, 1024}.
**Gates:** measured rate >= 0.8x of the class prior on >= half the target
shapes AND compile <= 150 ms/kernel. Below-prior rates feed back into the P0
formula; if the recomputed stressed ceiling < 1.08x, KILL.

## P2 — Fully charged pipeline + numerics (real weights/activations)

Weights: Q4_K -> fp16 offline via the existing codec seam (tooling lane, not
GPU hot path). Activations: captured from the CPU reference forward at the
probe layer (existing oracle infrastructure).
Charge everything: f32->fp16 staging (flat AND the conv-layout interleave
question decides which rate applies), ANE exec, readback fp16->f32, locks.
Comparators: (a) production Q4_K Metal time for the same projection
(`gdn-proj-micro` exact-shape bench), (b) Metal F16 same-weights control.
Numerics: projection cosine vs f32 oracle; one offline logit-replay with the
substituted projection on the prompt triad (favorable short / canonical
real-long / adversarial witness + generic-narrative guardrail).
**Gates:** charged ANE projection >= **1.20x** vs production Metal, OR
staging demonstrated pipelineable with compute such that the P3 concurrency
premise stands (staging_overlap_fraction >= 0.8). Fidelity: logit cosine
within the lane's recorded bounds on all triad rows. Else KILL.

## P3 — Interference oracle (zero integration)

Two processes: real `qwen-bench pp` (P0 host model, pp4096, quiet box) vs the
sibling harness hammering the real shapes in a loop. Both directions measured
vs solo baselines, 5 paired repetitions.
**Gates:** qwen prefill degradation <= **5%** AND ANE sustained >= **80%** of
solo AND projected whole-phase (P0 formula with P1/P2-measured rates and
P3-measured contention) >= **5%**. Else KILL with the contention numbers as
the recorded boundary.

## P4 — Conditional in-situ pilot (first production seam)

One projection class (P0-selected; prior: A3B `gdn_qkv`), flag
`QWEN_PREFILL_ANE_QKV` (default off, rollback documented), wavefront overlap
across prefill chunks; ragged tails stay on GPU; ANE weights staged once at
load; kernel compile overlapped with GGUF materialization (charge residual).
**Gates:** whole-prefill >= **1.05x** product-measured at pp4096/pp8192 on
the host model; process-cold TTFT >= **1.10x** including all charges; logit
fidelity green on the triad; `Result`-based dispatch with silent GPU fallback
verified by fault injection; no warm-decode or memory regression
(peak RSS delta accounted; fp16 mirrors ~1-1.5 GB budgeted).
Only a green P4 authorizes merging the seam to main (default-off).

## Review checkpoints (fresh cx sessions, read-only)

1. This preregistration (before P0 runs). 2. P0 artifact + gate decision.
3. P1 harness design before first ANE dispatch. 4. P2 numerics artifact.
5. P3 artifact + P4 go/no-go. 6. P4 diff review before merge.

## Artifacts

`docs/bench/2026-07-14-ane-p0/` (committed at the P0 checkpoint): raw phase
logs, summary JSON, ceiling table, gate decision. Later rungs follow the same
dated pattern.
