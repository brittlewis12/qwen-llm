# v0.600 Query-Capped Auto-Prefill Admission

Status: **implementation complete; confirmation inconclusive**.

Implementation source: `c4e4c4ae085b7e794dcb2cedbe900797d47aeeb0`.
P1/P2 artifacts:

- `target/profiles/v0600-auto-prefill-admission-p1/`
- `target/profiles/v0600-auto-prefill-admission-p2/`

## Delivered Path

The CLI now admits the measured 8K-16K wide-prefill profiles only when the exact
query-capped scratch plan fits conservative Metal headroom:

- A3B selects outer/query `2048/1024`;
- A10B selects outer/query `4096/1024`;
- every eager and deferred allocation is planned, Metal-priced, reconciled, and
  revalidated against the model before allocation;
- sequence growth is double-charged with a 512 MiB transient reserve;
- missing, invalid, overflowing, or insufficient signals fail closed to the
  untouched outer-1024 constructor;
- numeric overrides and cache-bearing JSONL requests retain legacy behavior;
- automatic rows use schema 5 with recomputable plan and admission telemetry;
- an admitted allocation failure is a hard error, never a baseline retry.

The full CLI suite passes 31 tests, including exact profile rejection, topology,
pricing, overflow, memory-boundary, ordering, schema, environment, and cache gates.
A latest-source A3B smoke admitted the exact 41-allocation plan and executed the
expected `2048/1024`, `60/120` topology with zero decode transitions.

## Confirmation Outcome

Neither canonical attempt launched a child, so neither observed A nor B timing:

- P1 stopped before A3B pair 1 arm A when its cache interval changed an unknown
  global pageout or swap term. Completion identity passed, but the original runner
  did not retain the two VM endpoints.
- A committed observability-only addendum authorized one complete P2 with all
  conditioning, ordering, gates, thresholds, and zero-growth predicates unchanged.
- P2 stopped at the same boundary. It records Pageouts `1,247,112 -> 1,247,466`,
  a 354-page (`5.53 MiB`) increase over the 22.13 GB cache read and 30.010-second
  cooldown. Swap occupancy stayed `1,737,689,661` bytes; cumulative Swapouts and
  Compressions were unchanged; memory availability stayed 96%; AC and thermal
  state were valid.

The repeated failure falsifies the packet's zero-global-Pageouts feasibility on
this otherwise pressure-neutral host interval. It says nothing about candidate
speed. P1 and P2 remain immutable and unpooled; no P3 is authorized.

## Implication

The product path remains exact and opt-in. Existing v0.567/v0.568 evidence still
projects about `1.060x` A3B and `1.206x` A10B fresh TTFT after query-cap overhead,
but v0.600 grants no default-on authority.

Any final confirmation must be a new preregistered experiment with a mechanistic
VM-pressure predicate. Global Pageouts should remain recorded, but the veto should
track actual anonymous-memory pressure: swap occupancy, cumulative Swapouts and
Compressions, plus the existing memory-availability, AC, thermal, block-input, and
paired-order controls. This is a protocol change, not a v0.600 rescue.
