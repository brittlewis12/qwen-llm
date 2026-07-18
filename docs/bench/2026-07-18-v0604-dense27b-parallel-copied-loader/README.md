# v0.604 Dense-27B Topology-Preserving Parallel-Copied Loader

Status: **sealed prelaunch inconclusive** with no product authority. The exact
dense implementation passes the frozen full-state correctness gate, but the
packet launches zero product children because a unilateral compressor-gauge
predicate reacts to a two-page host-wide change without corroborating pressure.

Frozen source, build, and runtime:
`040e4d77e8f048e3599d58ffa94d5408b4a5721a`.
Canonical packet:
`target/profiles/v0604-dense27b-parallel-copied-loader-p1/`.

## Correctness And Integrity

- The release correctness gate authenticates all `16,806,250,496` candidate
  bytes and 851 independent exact-sized offset-zero
  Shared/DefaultCache/Tracked resources.
- Resource identity, modes, bindings, frozen four-worker schedule, marker,
  copied ledger, checked-write rejection, packed-prefill logits, complete
  KV/GDN/conv state, argmax, forced transition, continuation logits, and
  continuation state pass exactly.
- The correctness candidate records `619.408 ms` ready wall and
  `2369.549 ms` endpoint CPU. The latter is encouraging but only `80.451 ms`
  below the product packet's `2.45 s` median cap and is not a product sample.
- The manifest binds clean source/build/runtime identity, macOS product/build,
  both release binaries, the exact model and prompt, the preregistration, the
  immutable runner, and all v0.603 authorization evidence.
- The decision, six-entry artifact inventory, final model hash, and completion
  seal verify. No attempt ledger or launch seal exists because conditioning
  stops before the first product launch.

## Pressure Stop

The exact 16.817 GB cache read and SHA-256 completes in `5472.877 ms`, followed
by the frozen cooldown and host sample. Across that host-wide interval:

| Signal | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Compressor occupied pages | `89,274` | `89,276` | `+2` |
| Compressor stored pages | `763,612` | `763,598` | `-14` |
| Compressions | `6,657,756,124` | same | `0` |
| Pageouts | `1,275,053` | same | `0` |
| Swapouts | `182,564,128` | same | `0` |
| Swap occupancy | `2,335,703,040 B` | same | `0` |

Decompressions rise by exactly 14. Memory remains 95% available, AC power is
valid, and no thermal or performance warning is present. The `+2` occupied-page
change is only 32 KiB and is not corroborated by compression, paging, swap, or
availability evidence. It is compatible with host-wide compressor
allocation/repacking or snapshot granularity, but this packet cannot localize
the cause.

The frozen contract nevertheless says any positive stored **or** occupied gauge
delta is invalid. The runner therefore correctly stops `inconclusive` before
launching `loaded-p01-ab-r1-a`. This is a falsification of that unilateral
predicate's specificity and operational feasibility, not a post-hoc pass and
not a failure of the packet's sealing, identity, correctness, or decision logic.

## Interpretation

The packet establishes production implementation correctness only. It provides
no loaded stability, 1% prefill/decode/request noninferiority, product-prompt
output identity, paired cold endpoint, model-ready, memory, or complete-process
CPU evidence. The correctness invocation cannot substitute for any product row.

No v0.604 rerun is authorized. Because zero product children launched, one new
v0.605 successor may prospectively revise only the pressure classifier while
freezing the implementation, correctness scope, order, commands, all loaded and
fresh gates, the `2.45/2.55 s` endpoint-CPU caps, sole-attempt behavior, and
force-only authority.

The successor should hard-fail capture/counter regression, positive
Compressions, Swapouts, swap-occupancy growth, child block input, fresh major
faults, or invalid host state. Pageouts and both compressor gauges remain fully
recorded diagnostics but do not independently veto an interval. This change
tests a new protocol; it does not rescue or reuse v0.604 observations.

Adversarial runner review used `cx ask` session
`019f773a-69f3-7ee1-8c4d-d2e084467eb3`. Sealed-result interpretation used
session `019f7760-bfb6-7310-bec5-a393609ec6c3`.
