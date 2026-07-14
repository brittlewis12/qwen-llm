# P0/P0b: ANE prefill co-processor Amdahl + DAG oracle — KILL

Program: docs/ANE-ORACLE.md rev 2 (preregistered at 18684c7, before any run).
Identity: zekrom M4 Max 128GB, macOS 15.6.1, binary+source 18684c75c (clean),
AC power, quiet box, `pmset -g therm` clean at start and end
(thermal-start.txt / thermal-end.txt).

## Data

| Cell | Baseline W (untraced, median of 5) | Traced passes |
|---|---|---|
| A3B pp1024 | 1741.7 t/s (0.588 s) | warmup + 3 |
| A3B pp4096 | 1650.7 t/s (2.489 s) | warmup + 3 |
| 27B pp1024 | 245.3 t/s (4.182 s) | warmup + 3 |
| 27B pp4096 | 232.3 t/s (17.505 s) | warmup + 3 |

Models: Qwen3.6-35B-A3B-UD-Q4_K_M, Qwen3.6-27B-Q4_K_M. Traces:
`QWEN_PREFILL_TRACE_{LAYER,ATTN,FFN}_PHASES=1`, `qwen-bench pp --runs 3`.
Baselines: `qwen-bench pp --runs 5 -o json`. Raw logs in this directory.
Estimator: per-job median across the 3 timed passes; W = median of samples.

## Instrument

`scripts/profile/ane_dag_oracle.py` (committed with this artifact): replays
traced execution order; offloaded classes move to a serial ANE queue with
producer/consumer edges (gdn_qkv->conv chain, gdn_z->gdn_gated,
shared_packed parallel to routed block, ffn gate/up->swiglu->down, attn
proj->rope, o_proj->resid); ANE service = FLOPs / class prior
(expansion 7.3, reduction 3.2 TFLOP/s); pessimistic row adds 250 us/job sync
+ serial f32<->fp16 staging at 13.55 GB/s (input staging deduped per
producer); optimistic row = zero sync, fully overlapped staging; exhaustive
subset x {all,alt} layer coverage search; ceilings projected onto W via the
recorded distortion factor (0.97-1.00 across cells).

## Results (best subset per cell)

| Cell | Best subset | cov | pess | opt | Gate (pess>=1.10 AND opt>=1.15) |
|---|---|---|---|---|---|
| A3B pp1024 | OFF-SHARED+OFF-SKINNY | all | 1.0552 | 1.0552 | FAIL |
| A3B pp4096 | OFF-SHARED+OFF-SKINNY | all | 1.0533 | 1.0533 | FAIL |
| 27B pp1024 | OFF-SKINNY | all | 1.0055 | 1.0055 | FAIL |
| 27B pp4096 | OFF-SKINNY | all | 1.0054 | 1.0054 | FAIL |

Full top-10 tables: oracle-*-pp*.txt. Estimator spread (per-slot min /
median / max across the 3 timed passes):

| Cell | min | median | max |
|---|---|---|---|
| A3B pp1024 | 1.0550 | 1.0552 | 1.0552 |
| A3B pp4096 | 1.0533 | 1.0533 | 1.0533 |
| 27B pp1024 | 1.0055 | 1.0055 | 1.0055 |
| 27B pp4096 | 1.0054 | 1.0054 | 1.0054 |

Margins are not within +/-2% of any gate threshold at spread bounds ->
determinate, no re-run set triggered. These ceilings are **modeled
serialized-trace upper bounds**, not measured residual product wins (see
mechanism note 2).

## Mechanism (why the lane dies)

1. **The big classes never schedule.** OFF-EXP (gdn_qkv/z) and DENSE-FFN
   run on GPU at ~13 TFLOP/s (measured t_j vs exact FLOPs). ANE priors are
   7.3/3.2 TFLOP/s. Offloading them stalls their own consumer
   (qkv -> conv -> step; gate/up -> swiglu) longer than just running them on
   GPU. No subset containing them beats its own parent without them, in any
   coverage, in either scenario row. Aggregate idle time is not schedulable
   overlap — the dependency chain is binding, exactly as review 1 predicted.
2. **The only real slack is already small — and already taken.** The winning
   jobs (shared_packed under the routed-expert block; skinny beta_alpha under
   the qkv chain) are fully hidden in both scenario rows (pess == opt), worth
   5.3-5.5% on A3B and 0.5% on 27B *against a serialized baseline*. The
   production engine already overlaps shared-expert and GDN-front work with
   concurrent encoders (v0.340 lineage, +8.8% A3B), so part of this modeled
   win is already banked in W. The true residual is smaller than the number
   shown; the kill does not depend on this correction but is strengthened
   by it.
3. **Wavefront was not tested.** The preregistered wavefront variant was
   not implemented; nothing here bounds a restructured-execution-order
   schedule. Wavefront remains open strictly as a reopen condition, and a
   wavefront reopen must first implement that variant in the retained
   oracle script.

## Decision

**KILL the ANE prefill co-processor lane** per the preregistered gate.
P1-P4 do not run. No workspace ANE dependency is introduced. The pinned
sibling-harness design and rustane survey remain recorded in
docs/ANE-ORACLE.md for reopen use.

## Boundary (what this does and does not close)

- CLOSED: ANE offload of statically-shaped prefill projections under the
  current chunk-sequential execution order, at fp16 IOSurface staging costs,
  with graph-matmul ANE rates in the 3.2-7.3 TFLOP/s band (measured
  independently by rustane on M4 Max; not re-measured here — P1 was not
  reached, and per prereg a P1 salvage run is not permitted).
- NOT closed: drafter-on-ANE (separate lane, still gated behind speculative
  economics per PERF-ROADMAP.md); ANE for a *different work unit* (e.g.
  fp16 whole-block compute where GPU is not the comparator).
- REOPEN conditions (amended from prereg after review 2): measured ANE
  rates materially above priors from credible external evidence; a public
  stateful-ANE API; or a promoted wavefront/new-GDN work unit changing the
  DAG — which requires implementing the preregistered wavefront variant in
  the retained oracle script before any hardware work. A Metal<->IOSurface
  zero-copy API **alone is no longer a sufficient reopen**: the optimistic
  row already prices staging and sync at zero and still fails both gates.

## Instrument corrections after certification (review 2, cx 019f6218)

Review 2 CERTIFIED the kill and identified safe-side instrument defects:
double-charged sync, layer-join chunk-boundary leakage, inconsistent staging
queue occupancy, two optimistic consumer mappings (beta_alpha ->
gdn_alpha_beta; proj -> norm), and even-only "alt" coverage (odd parity
unsearched). All were fixed in scripts/profile/ane_dag_oracle.py; outputs
regenerated from the same raw logs are identical at 4 decimals (tables
above). Review 2 independently verified singleton OFF-EXP at odd parity:
0.912-0.983x (below 1.0) — attention-layer coverage cannot rescue the gate.
The oracle script is retained as the reopen instrument; this amends the
program's default "removes experiment source" contract.
