# ANE Prefill Co-Processor Oracle (P0-P4)

Status: **P0/P0b preregistered, rev 2** (amended after adversarial review 1,
cx session 019f61ea; review verdict NO-GO on rev 1's formula — this revision
adopts its corrections in full). Branch `ane-oracle`; integrate to main only
through the gates below. Every rung is killable; a kill merges this doc +
artifacts + a PERF-LOG entry and removes experiment source.

## Objective lane and boundary

Approximate/concurrency lane, prefill only, serial BS=1. The hypothesis is
NOT "ANE matmul is faster than GPU" (it is not: ~3-7 TFLOP/s vs our 12.8
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

**Product objective contract (fixed now, per review):** this lane's primary
prize is a scoped **loaded-model prefill win** (auto-prefill-lane style,
cf. the promoted 1.0641x A3B cell); process-cold TTFT must be a
**non-regression**, not the prize. P4 promotes on warm prefill with cold
parity; it does not require a 1.10x cold win.

## Prior evidence (external, commit-pinned; reliability annotated)

Source: rustane @ c422447 + ncdrone/ane @ 016b754. Corrections from review:

- ANE conv1x1 768->3072 w=512: 7.315 TFLOP/s; 3072->768: 3.238 TFLOP/s.
  **Single shape family; possibly fp32-I/O era** (bench builds f32 tensors;
  the fork's fp16 TensorData postdates parts of the results). Treat as
  weak priors, replaced by P1 measurements on our shapes.
- Dual-load ANE -1.5% / GPU -2.6%: **unpaired, interval-diluted** (5s solo vs
  30s dual, ANE active only part of the window). Suggestive only; P3 replaces
  it with paired, duty-cycle-matched measurement.
- Staging: 90 GB/s is flat f32 memcpy. The relevant fused f32->fp16 path is
  **13.55 GB/s** (rustane results/f16_convert_m4max.md); interleaved ~10 GB/s.
  Charged staging uses 13.55 GB/s until P2 measures better.
- Compile ~50-90 ms/kernel; no cross-process plan cache. **Charged as
  aggregate critical-path compile for the full executable set**, not
  per-kernel (see P1 gate).
- Dim-5120 cliff (4.7x/layer): model-forward evidence only; P1 isolates.
- macOS 15.6.1 here; the hwExecutionTime=0 regression was reported on 26.

## Environment

M4 Max 128GB (zekrom), macOS 15.6.1, worktree @ 81481491c (clean stamp).
Anchors: stream 474.0 GB/s, Q4_K mat-mat 12.80 TFLOP/s.
Models: Qwen3.6-35B-A3B-UD-Q4_K_M (sentinel), Qwen3.6-27B-Q4_K_M (dense
guardrail), Qwen3.5-122B-A10B-UD-Q4_K_XL (heavy, only if P0 warrants).
Thermal protocol for every timed rung: AC power, high-power mode, quiet box,
>=60s idle between model-scale runs, alternating A/B order where paired,
`pmset -g thermlog` clean or the run is discarded and rerun.

## Offloadable job classes (fixed vocabulary)

Jobs are (phase, layer, chunk) instances from `QWEN_PREFILL_TRACE_LAYER_PHASES`
/ `QWEN_PREFILL_TRACE_ATTN_PHASES`. Classes carry **absolute** ANE
service-rate priors A_i (TFLOP/s), replaced by P1 measurements:

| Class | Phases | Shape class | A_i prior |
|---|---|---|---|
| OFF-EXP | `gdn_qkv`, `gdn_z` | expansion | 7.3 |
| OFF-RED | `gdn_back` (= out_proj) | reduction | 3.2 |
| OFF-SHARED | `shared_packed` | small expansion+reduction pair | 3.2 |
| OFF-SKINNY | `gdn_beta_alpha` | skinny | 3.2 |
| OFF-ATTN-E | attn qkv (decomposed) | expansion | 7.3 |
| OFF-ATTN-R | attn o_proj (decomposed) | reduction | 3.2 |
| DENSE-FFN-E / -R | 27B gate/up | down | 7.3 / 3.2, **contingent on P1 cliff test** |
| NOT-OFF | everything else | — | — |

Per-job FLOPs F_j are exact from shapes; per-job ANE service time
a_j = F_j / A_class + staging charge
(bytes_in(f32->fp16 at 13.55 GB/s) + bytes_out(fp16->f32 at 13.55 GB/s),
overlappable fraction 0 in the pessimistic row, 1 in the optimistic row).
GPU time per job t_j is measured, not derived from any anchor.

## P0 — traces + P0b DAG makespan oracle

**P0 traces.** For A3B and 27B: `qwen-bench pp` at pp1024 and pp4096,
`--runs 3`, layer+attn+ffn traces on, logs to /tmp, committed to the artifact
dir. Untraced `pp --runs 5` wall at the same shapes = the **baseline W**
(median). Serialized-trace distortion factor recorded and used only to scale
shares onto W, with the scaling declared per row.

**Estimator freeze:** per-job t_j = median across the 3 traced runs' matching
(chunk, layer, phase) records; W = median of 5 untraced runs; report min/max
spread; a gate decision within +/-2% of its threshold at either spread bound
is INDETERMINATE and triggers one preregistered re-run set, not a redefinition.

**P0b DAG oracle (document-only formula is retired; this is the decision
instrument).** A script (`scripts/profile/ane_dag_oracle.py`, committed with
the artifact) replays the traced execution order per chunk/layer/phase and
computes makespans under an explicit dependency model:

- GPU is one serial resource executing traced phases in traced order, minus
  jobs moved to ANE; ANE is one serial queue.
- Dependency edges (current engine order): within a layer,
  `gdn_qkv -> gdn_prep_conv -> gdn_step -> gdn_gated -> gdn_back -> resid`;
  `gdn_z -> gdn_gated`; `attn-proj -> attn-body -> o_proj`; MoE:
  mixer output -> {`route_fused` -> routed chain} and -> `shared_packed`,
  both joining at the layer output; layers sequential within a chunk;
  chunks sequential (**current-order model**).
- An ANE job's release = completion of its producer on either resource;
  its consumer cannot start before the ANE job + its readback charge finish.
- Sync cost per ANE job: two scenarios, 0 (optimistic) and 250 us
  (pessimistic CPU rendezvous + encoder split proxy, revisited in P2).
- Search: exhaustive over class subsets x {all layers, alternating layers}
  x {optimistic, pessimistic} staging/sync rows; report
  ceiling(S) = W / makespan(S) for the best S per row.
- **Wavefront variant (documented, not authorizing):** same DAG with
  chunk c+1 layer l released after (chunk c, layer l) state commit and
  (chunk c+1, layer l-1) — the "new GDN work unit" premise. Reported to size
  the prize behind engine restructuring; P1-P4 as scoped CANNOT claim it.

**Preregistered gates (frozen before running):**

- PROCEED to P1 iff, on at least one model at pp4096, **current-order**
  ceiling >= **1.10x** in the pessimistic row (measured-t_j, staging charged,
  250 us sync, A_i priors) AND >= **1.15x** in the optimistic row.
- KILL the lane iff no model clears. Reopen conditions: measured ANE rates
  materially above priors (P1 run anyway as a 1-day salvage is NOT permitted
  — kill means kill), a public stateful-ANE or Metal-IOSurface zero-copy API,
  or a promoted wavefront work unit changing the DAG (which reopens P0b with
  the same script and gates, no new preregistration needed).
- Host model = highest pessimistic-row ceiling.

**P0 does not authorize:** production code, workspace ANE deps, quality claims.

## P1 — Shape truth (pinned sibling harness, no qwen-llm coupling)

Standalone crate `spikes/ane-oracle` (not a workspace member; own lockfile;
`ane` pinned by rev 016b754). Random fp16 data. Shapes: every job class the
P0b-winning subset uses, at the host model's exact dims; plus isolated
5120-dim matmuls (cliff test) if DENSE-FFN is in any winning subset.
Measure per shape: compute-only TFLOP/s (wall clock; hwExecutionTime if
live; powermetrics ANE-power corroboration if sudo available), compile wall
per executable, executable count for the winning subset, dispatch overhead
at w in {512, 1024}.
**Gates:** every scheduled job class's measured rate feeds P0b re-run; the
recomputed pessimistic ceiling must hold >= **1.10x** with **measured** rates
(jobs whose class underperforms are dropped from S, not averaged over) AND
aggregate critical-path compile + weight-blob creation for the winning
executable set <= **50%** of one median model-load wall (so load-overlap can
plausibly hide it; measured against the host model's load). Else KILL.

## P2 — Fully charged pipeline + numerics (real weights/activations)

Weights: Q4_K -> fp16 offline via the codec seam (tooling lane). Activations:
captured from the CPU reference forward at probe layers. Charge everything:
staging in/out at measured rates (flat vs interleaved resolved by the actual
layout the conv path requires), ANE exec, locks, rendezvous latency with a
GPU command stream actually running (same-process measurement of encoder
starvation while a CPU thread blocks on ANE — two-thread probe, not
two-process).
Comparators: production Q4_K Metal (`gdn-proj-micro`), Metal F16 same-weights.
**Gate (replaces rev 1's unsafe OR):** dependency-aware projected
whole-prefill gain from the P0b DAG, re-run with all P2-measured charges
(ANE service, non-overlapped staging measured not assumed, measured sync),
>= **5%** on the host model at pp4096. Else KILL.
**Numeric contract (frozen now):** per-projection cosine vs f32 oracle
>= 0.9995 and max-abs within 4x the Q4_K-vs-f32 envelope on the same tensor;
**full-depth replay before P3** — every proposed offloaded occurrence
substituted simultaneously across all layers on the prompt triad (favorable
short / canonical real-long / adversarial witness + generic-narrative
guardrail): final-logit cosine >= 0.999, greedy argmax parity over 64
continuation tokens, and (MoE) route-set stability >= 99.5% of tokens with
all discrepancies enumerated. Any miss = KILL (approximate-lane rules).

## P3 — Interference oracle (zero integration)

Same-process two-thread probe from P2 extended to full duty cycle, plus the
two-process hammer as a secondary check. The ANE load replays the winning
subset's real burst pattern (duty cycle from the P0b schedule, not a steady
loop). Paired A/B (ANE-active vs ANE-idle), 5 repetitions, alternating order.
Also measured: **ANE-resident-but-inactive** engine throughput (fp16 mirrors
+ compiled models resident, no dispatch) vs clean baseline — the
memory-topology charge (v0.591-0.596 lesson: provenance/topology alone can
cost 3-13%); loaded prefill AND warm decode rows; peak RSS, Metal allocated
bytes, pageins/pageouts.
**Gate:** net projected whole-prefill gain — P0b DAG with P1/P2 rates, P2
sync, and **both-direction measured interference applied** (GPU phases
inflated by measured degradation while ANE active; ANE rates deflated
likewise) — >= **5%** at pp4096, AND warm decode + ANE-resident-inactive
rows are >= **0.99x** baseline. Else KILL.

## P4 — Conditional in-situ pilot (first production seam)

One projection class (P0b-selected), flag `QWEN_PREFILL_ANE_*` default off,
current-order overlap only (no wavefront restructuring in this program);
ragged tails on GPU; ANE weights staged once at load; kernel compile
overlapped with GGUF materialization.
**Gates:** loaded-model whole-prefill >= **1.05x** at pp4096 and pp8192 on
the host model (paired, 10 pairs, median); process-cold first-byte >=
**0.99x** (non-regression, all compile/materialization charges in);
warm decode >= 0.995x; full-depth numeric contract green (same thresholds as
P2); fault-injection verified silent GPU fallback (kill ANE mid-run, output
exactness preserved); peak-memory delta within the budgeted fp16 mirror set
+ compiled models, enumerated.
Only a green P4 authorizes merging the seam to main (default-off).

## Review checkpoints (fresh cx sessions, read-only)

1. Rev 1 preregistration — DONE (verdict NO-GO; this rev 2 is the response).
2. Rev 2 + P0/P0b artifact + gate decision (before any P1 code).
3. P1 harness design before first ANE dispatch. 4. P2 numerics artifact.
5. P3 artifact + P4 go/no-go. 6. P4 diff review before merge.

## Artifacts

`docs/bench/2026-07-14-ane-p0/`: raw phase logs, untraced baselines, DAG
oracle script output (all subsets, all rows), ceiling table, gate decision,
thermal log notes. Later rungs follow the same dated pattern.
