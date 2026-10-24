# Bounded CED Decoder Retirement

Candidate branch: `perf/v41-ced-batches`, based on `cada5864`. Executable proof
source: clean `09e5f88255e04fbfe53ccab166a75da38a04cacf` on Apple M4 Max.
CED remains opt-in. This candidate does not incorporate or qualify the parallel
4K Full/encoder geometry integration. No full-model, native latency or M5 result
is claimed.

## Execution Contract

The encoder retains its existing chunk completion boundaries. Production CED
groups two consecutive decoder layers into one command buffer with one serial
compute encoder. Every actual seed chunk, packed output chunk and native scalar
tail is encoded in original layer-major order. Scoped tests select the old
per-unit incumbent or one/two/four-layer groups; there is no production knob or
generic queue scheduler.

One request-local maximum decoder activation pool and one maximum seed backing
are reused under ordered GPU dependencies. Gate/up remain serial, shared half
staging is overwritten before reuse, local ring writes precede their consumers,
and absolute control origins are unchanged. Native scalar tails still call the
original scalar layer operator. Compact packed singletons retain packed Full
operator lineage rather than being reclassified as native tails.

## Transaction And Errors

A CED-local ledger retains encoded seed/output progress and exact ordered units.
These are not completed/public frontiers. Selected IDs and candidate masks are
authorized only by completed coverage or an earlier fully encoded matching
producer in the same group, including its absolute range and producer epoch.
Generation and layer/range checks precede mutation. Retirement validates every
unit before atomically publishing completed frontiers and control coverage.

Packed FFN units bind disjoint row-status ranges in one priced ledger. Batch
views never CPU-clear the ordinary compact `[layer * current_rows]` status slab.
Production scalar tails execute at most once per layer, so their existing
disjoint scalar status words remain valid witnesses. An ordered one-word copy
records sticky attention status after every unit; later successful operations
cannot erase an earlier failure.

Inspection preserves incumbent logical failure precedence: standalone multichunk
seed attention first, and output FFN before output attention. A seed that the
incumbent fused with its first packed output retains that compound's FFN-before-
sticky-seed/attention precedence. Every command result is checked; only one
submitted decoder-group command is owned at a time.

Cancellation during encoding discards the entire unsubmitted group. After
submit, synchronous completion drains owned work before status inspection,
cancellation observation or source release. Failure stops future groups. The
existing provisional guard poisons failures and unwinds; no post-mutation Full
fallback, Ready state, snapshot or decode access is permitted.

## Count And Memory Plans

Counts derive from actual encoder geometry and decoder groups. On this pinned
2048-row geometry, the 40-layer default schedule plans:

| Prompt Rows | Incumbent | Group 1 | Group 2 | Group 4 |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 21 | 21 | 11 | 6 |
| 8192 | 27 | 24 | 14 | 9 |

Group 1 also co-encodes a layer's multiple chunks. Useful output/seed work and
ordered arithmetic dispatches do not change. These are command-count plans, not
latency multipliers or current-main 4K compatibility evidence.

The single shared status allocation is the maximum over actual groups of four
bytes times packed output rows plus one attention word per unit. Scalar FFN
statuses and global sticky attention status already belong to the workspace.
CPU pricing adds frontier/Pending sizes and the established 1 KiB descriptor
allowance for retained views. Default two-layer metadata witnesses at capacity
equal to prompt length are:

| Prompt Rows | Logical Status Bytes | Batch CPU Bytes | Additional Encoding Allowance Per Domain |
| --- | ---: | ---: | ---: |
| 128 | 1032 | 30736 | 1051648 |
| 2048 | 16392 | 35184 | 16780288 |
| 8192 | 18828 | 52400 | 20301824 |

Status buffers use normal device pricing, including page slack. The incumbent
16 MiB command allowance remains; the additional explicit allowance charges
4 KiB per encoded seed/output operation-row plus 1 KiB per unit to both
CPU/private and Metal admission. This is a conservative allowance, not measured
driver high-water, a percentage-based capacity gate or a waiver.

Strict test qualification GPU-copies each unit's selected IDs and candidate masks
before later reindexing overwrites them. Bounded per-group diagnostic views and
the finite-status fault source are priced in test-build runtime, metadata and
frozen-reference preweight plans. Production allocates none of these diagnostic
buffers. Arithmetic parity removes only the exact separately captured ledger /
diagnostic-copy interval and asserts every excluded dispatch.

## Bounded Evidence

Release workspace all-target checking, formatting and optimized test `--no-run`
pass. Executables are frozen under the same build lease before releasing the
shared target. Focused pure planner/cursor/coverage/preflight/status tests and
five production-self-leased GPU tests pass with `MTL_DEBUG_LAYER=1`:

- Old versus group 1/2/4: exact complete heads, all causal state, all absolute
  query controls, ordered arithmetic PSOs/grids, counters and observed admission.
- Rows 1, 127/128/129, 2047/2048/2049, plus N=640/W=257 and N=2050/W=2050.
  The latter has two decoder layers and crosses both output and seed boundaries;
  other fixtures retain four decoder layers. All new complete ledgers are at
  most 128 MiB, including reserves; no capacity exception is used.
- Early FFN and attention faults survive later valid writes; simultaneous faults
  preserve precedence. Cancellation, completion errors and unwind stop later
  submissions, retain public frontiers, poison state and recover allocation
  baseline after draining.
- Invalid scoped status prefixes preserve compact statuses and trailing canaries.
  Loaded runtime and metadata cold ledgers match exactly.

The private artifact root ends in
`v41-leverage-2026-10-24.dc1jxYdL/ced-batches-candidate1`. The final build manifest
is `09e5f882-proof4/frozen.json`, results are `09e5f882-tests2/results.json`, and
the aggregate packet is `09e5f882-proof-summary.json`. Frozen core-test SHA-256:
`a76d3936550bfed2b39fd980a7fc46ee6bd710766302122431205034e9959d2c`.
Earlier rejected build/fixture attempts remain in that private chronology.

M5 small/wide tensor-lineage fixtures compile but are not run on M4. Parent
review, rebase of only these new commits onto integrated scratch, joined 4K
boundary tests, M5 tensor qualification, full-model/native qualification and
uninstrumented latency measurement remain separate required work.
