# Root-Aware Dense Refill Screen

Date: 2026-08-12

Status: HOLD before implementation; transition-only projected gain is weak.

## Question

Does composing the new file-scoped root checkpoint with bounded dense B8 refill
clear the existing `1.10x` whole-file gate against the best current executor?

The fixture uses 16 realistic prompts sharing a 6,482-token Current/system root.
Each has a distinct post-root user task, so B2 cannot receive an accidental
deeper identical-prompt checkpoint. Generation limits are
`8,9,10,11,12,13,14,15,40,44,48,52,56,60,64,68`; every request reaches the token
limit. Runs are warm-mmap, greedy, and use chunk 1,024.

## Measured Incumbents

| Organization | Wall | Physical B8 steps |
|---|---:|---:|
| B2 with one file root | `7.27 s` | n/a |
| Static B8 with one file root | `7.42 s` | `81` |

Every B2 pair matches only the shared root (`6,482` tokens) and selects the same
chunk-aligned 6,144-token checkpoint. No pair reports `selected_identical`.

B2 and B8 agree on 15 of 16 greedy completions. This packet does not localize
or classify the one cross-schedule mismatch; it records each per-request digest
and makes no exactness claim across different execution schedules. All requests
in both arms reach `token_limit`.

## Transition-Only Projection

Longest-first list scheduling assigns 508 productive transitions to eight slots
in 67 steps (`0.948` utilization). Static B8 uses 81 steps. Its measured
transition wall is `1,473.334 ms`, or `18.189 ms/step`. Crediting the candidate
all 14 eliminated steps at that full measured rate, while charging zero
replacement allocation, restore, or scheduler overhead, gives:

```text
7.42 s - 14 * 18.189 ms = 7.165 s
7.27 / 7.165 = 1.015x versus B2
```

This transition-only projection misses the `1.10x` gate. It is not a formal
upper bound: a refill implementation could also remove some per-cohort setup and
allocation work, while adding replacement restore and scheduler work. The result
therefore supplies a weak prior against implementation rather than a proof of
impossibility.

## Decision

Hold root-aware synchronous refill rather than implementing it now. Its best
measured incumbent is B2 at `7.27 s`, while eliminating every avoidable B8 step
projects only `1.015x`. The remaining non-transition opportunity is unmeasured and too uncertain to
justify more scheduler work at current leverage. Reconsider only if
profiles expose material duplicate setup beyond the retained root, B8 transition
cost falls relative to B2, or replacement prefill can overlap decode safely.

The fixture and machine-readable arithmetic are retained in this directory.
