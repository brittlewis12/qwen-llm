# Charged Fixed-Cohort Refill Screen

Date: 2026-08-12
Status: bounded dense B8 mechanism GO; synchronous MoE B16 refill killed

## Question

Does replacing finished fixed-cohort lanes between committed decode steps retain
a meaningful whole-file advantage once replacement prefill and the already
available B2 executor are charged?

The screen intentionally precedes scheduler code. Commit `01d9b06` establishes
independent per-lane frontiers, but that mechanism alone does not prove that
pausing active decode to prefill a replacement is useful.

## Inputs

The refill oracle assigns requests longest-transition-first to the currently
least-loaded slot. This is optimistic list scheduling: it charges every prompt
prefill serially, assumes zero scheduler/reallocation overhead, and uses measured
fixed-width transition wall. It therefore upper-bounds a synchronous refill
implementation rather than predicting one conservatively.

A 32-request skew trace repeats generation limits
`1,2,3,4,5,6,7,8,9,12,16,20,24,28,32,40` twice over short heterogeneous
prompts. Actual B2 and serial controls use the same release binary and emit
byte-identical JSONL.

## Results

| Family | Serial | Measured B2 | Optimistic refill | Refill vs B2 | Decision |
|---|---:|---:|---:|---:|---|
| Dense 0.8B Q8, B8 refill | 2.76 s | 2.29 s | 1.85 s | 1.239x | spike |
| Qwen A3B Q4, B16 refill | 10.08 s | 9.06 s | 10.06 s | 0.900x | kill |

The measured A3B B2 path spends 3.264 seconds in prefill and 3.043 seconds in
decode over 224 pair-equivalent steps. Ideal B16 list scheduling reduces the
402 productive transitions to 39 physical B16 steps, but each measured B16 step
costs about 103.75 ms. After charging unchanged prefill and process residual,
the optimistic endpoint is 10.06 seconds—already slower than B2 before any
replacement allocation or scheduler overhead.

Dense economics differ. Measured B2 spends 0.517 seconds in prefill and 1.340
seconds over 224 pair-equivalent steps. Ideal B8 list scheduling needs 51 B8
steps at about 17.62 ms each, yielding a charged 1.85-second upper-bound endpoint
against measured B2 at 2.29 seconds.

## Decision

Authorize one bounded dense-only mechanism spike:

- at most two waves, or 16 requests, per B8 arena;
- longest-transition-first deterministic slot scheduling;
- synchronous replacement prefill between completed B8 steps;
- exact greedy output and input-ordered publication;
- explicit default-off gate;
- require at least 1.10x whole-file improvement over measured B2 and no memory
  growth beyond eight live sessions plus persistent prefill scratch.

Do not implement synchronous MoE B16 refill. Reopen it only when replacement
prefill can overlap decode, or a new transition organization materially changes
B16-versus-B2 economics. Keep generic scheduler arithmetic reusable, but do not
broaden the product executor merely for symmetry.


## Dense Mechanism Result

The heterogeneous dense spike was measured with the refill gate plus the
existing ragged-prompt gate:

```text
QWEN_FIXED_COHORT_RAGGED_PROMPTS=1
QWEN_DENSE_BATCH8_REFILL=1
```

It rewrites the exact refill-disabled planner output only after prefix and ragged
policy have run, groups at most 16 requests, schedules longest requested decode
first, and replaces finished sequences between completed B8 commands. Admission
prices exactly eight live sessions, persistent prefill scratch, executor scratch,
and reserve. Refill denial restores the captured static plan before ordinary
cohort admission. Automatic mode and MoE never see the rewrite. Planner schema 6 makes refill
configuration, planned arenas, realized arenas, and admission fallback explicit.

Final release measurements on the 32-request dense skew trace are:

The durable fixture is `requests.jsonl`; `validation.json` records its digest,
base commit, abbreviated command matrix, wall times, complete per-request output
digests, and both refill-arena telemetry records. The measured binary contained
the refill implementation over base commit `01d9b06`; the final source-only
cleanup changes planner schema/accounting and docs, not execution.

| Organization | Wall | Result |
|---|---:|---|
| Serial | 2.76 s | exact control |
| B2 pair planner | 2.29 s | byte-identical |
| Static ragged B8 | 2.27 s | byte-identical |
| Bounded B8 refill | 1.83 s | byte-identical |

Refill improves `1.251x` over measured B2 and `1.240x` over static ragged B8,
clearing the preregistered `1.10x` whole-file gate. Two refill arenas execute
7 and 47 B8 steps, respectively. The second records 346 productive and 30
padding transitions (`0.9202` utilization); replacement prefill remains
synchronous and is included in whole-arena timing.

This is a bounded mechanism result, not default admission authority. Refill
requires `QWEN_DENSE_BATCH8_REFILL=1`; heterogeneous prompt lengths additionally
require the ragged-prompt gate. It excludes cohorts with selected prefix fanout,
honors explicit context limits, and remains absent from automatic execution.
