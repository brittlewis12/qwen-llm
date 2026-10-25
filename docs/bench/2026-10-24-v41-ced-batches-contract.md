# Bounded CED Decoder Retirement

Candidate branch: `perf/v41-ced-batches`, based on `cada5864`. Executable proof
source: clean `24c0f41188cfb6e3b31c7a836d4618a43ee2ff39` on Apple M4 Max.
CED remains opt-in. This candidate does not incorporate or qualify the parallel
4K Full/encoder geometry integration. No full-model, native latency or M5 result
is claimed.

## Execution Contract

The encoder retains its existing chunk completion boundaries. Production CED
groups two consecutive decoder layers into one command buffer with one serial
compute encoder only when every requested group has at most 16 physical units.
Each seed/output unit is bounded by the actual decoder chunk geometry (2048 rows
here); scalar tails have one row. Every actual seed chunk, packed output chunk
and native scalar tail is encoded in original layer-major order. Scoped tests select the old
per-unit incumbent or one/two/four-layer groups; there is no production knob or
generic queue scheduler.

If any requested group exceeds that unit budget, selection uses the existing
bounded per-unit CED decoder schedule for the entire decoder. It does not use
one huge one-layer group, reject a model/context, shrink rows or restart Full.
Selection precedes decoder encoding and batch pricing. The memory report records
`decoder_batch_layers=0` and a `physical_unit_budget` reason in that case; new
batch-only status/descriptor/encoding allowances are zero. Counters still count
actual completed commands. This policy bounds a grouped command to at most
32768 operation-rows on the pinned geometry, independently of window/request
length. Pure 32768/1M-row and window tests also check retained-control extents.

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

After drain, the guard also clears provisional attention cursor/coverage owners.
It does not reset mutated causal storage or unpoison the State. This releases
borrowed control buffers on failure while a caller continues holding the
poisoned State, including failures in the incumbent decoder path.

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
CPU pricing adds frontier/Pending sizes and the established descriptor allowance
for retained views. Test builds additionally price GPU-written guard regions and
their views around every status slice and the unused ledger tail. Default
two-layer witnesses at capacity equal to prompt length are:

| Prompt Rows | Production Logical Status Bytes | Test Guarded Status Bytes | Additional Encoding Allowance Per Domain |
| --- | ---: | ---: | ---: |
| 128 | 1032 | 1160 | 3145728 |
| 2048 | 16392 | 16520 | 4194304 |
| 8192 | 18828 | 19148 | 6291456 |

Status buffers use normal device pricing, including page slack. The incumbent
16 MiB command allowance remains. The additional allowance is now 1 MiB per
physical unit to both CPU/private and Metal admission, at most 16 MiB/domain.
This supersedes the unbounded 4 KiB/operation-row plus 1 KiB/unit allowance from
candidate1. It is explicitly unmeasured and is not a formal driver upper bound,
a percentage-based capacity gate or a waiver. Over-budget schedules price the
actual incumbent commands, not an enormous unsubmitted batch.

Strict test qualification GPU-copies each unit's selected IDs and candidate masks
before later reindexing overwrites them. Bounded per-group diagnostic views and
the finite/status-word fault sources are priced in test-build runtime, metadata and
frozen-reference preweight plans. Production allocates none of these diagnostic
buffers. Arithmetic parity removes only the exact separately captured ledger /
diagnostic-copy interval and asserts every excluded dispatch.

## Source Freshness

Frozen-reference and current-pair qualification use the same additional compiled
source inventory: the batch planner, runtime, test diagnostics, existing scoped
runtime hooks, and Rust/Metal elementwise primitives used by the new proofs.
Both actual freshness gates have per-file negative checks: changing only any
one registered body rejects the binary before import. No maintained source is
edited by those tests. These gates specifically cover the three new batching
modules omitted by candidate1's inventory.

## Bounded Evidence

Release workspace all-target checking, formatting and optimized test `--no-run`
pass. Executables are frozen under the same build lease before releasing the
shared target. Focused pure planner/cursor/coverage/preflight/status tests and
eight production-self-leased GPU tests pass with `MTL_DEBUG_LAYER=1`:

- Old versus group 1/2/4: exact complete heads, all causal state, all absolute
  query controls, ordered arithmetic PSOs/grids, counters and observed admission.
- Rows 1, 127/128/129, 2047/2048/2049, plus N=640/W=257 and N=2050/W=2050.
  The latter has two decoder layers and crosses both output and seed boundaries;
  other fixtures retain four decoder layers. All executed GPU fixture ledgers are at
  most 128 MiB, including reserves; no capacity exception is used.
- One FFN word is GPU-overwritten after healthy unit math: layer 2, row 2047,
  status -7. That exact coordinate fails, while every other ledger word is READY,
  including the same layer's second two-row chunk and a later one-row unit.
- One-shot attention failure leaves every FFN word READY. Separate competing
  error cases prove same-unit FFN-before-attention and earlier-unit attention
  before later FFN precedence, without corrupting biases or activations.
- Typed NoBuffer injections cover ledger allocation, partially allocated seed
  storage and lazy decoder backing. Cancellation/error/unwind also run after one
  successful group retirement and in the incumbent path. All release checks use
  the pre-existing causal-state baseline while poisoned State remains alive;
  public step/snapshot/growth rejection and stopped future submits are asserted.
- GPU-written status canaries survive every successful completion; a deliberate
  GPU boundary clobber is detected after completion. Invalid CPU status prefixes
  preserve trailing canaries. Loaded runtime and metadata cold ledgers match.

The private artifact root ends in
`v41-leverage-2026-10-24.dc1jxYdL/ced-batches-candidate2`. The final build manifest
is `controls-extents-proof5/frozen.json`, results are `24c0f411-tests4/results.json`,
and the aggregate packet is `candidate2-proof-summary.json`. Frozen core-test
SHA-256: `6398b49a18bf3aa8767a6fd74621e56bf7a2740d0810a698479ae54d906c57de`.
Candidate1 artifacts remain immutable; rejected candidate2 build/fixture attempts
remain in the new packet's chronology. Test process durations include lease
waiting and are not native latency measurements.

M5 small/wide tensor-lineage fixtures compile but are not run on M4. Parent
review, rebase of only these new commits onto integrated scratch, joined 4K
boundary tests, M5 tensor qualification, full-model/native qualification and
uninstrumented latency measurement remain separate required work.
Inclusive normal-production CPU/device transient peaks without test control
copies remain for parent diagnostics; allocation samples and release-to-baseline
checks do not establish those peaks or formally qualify the encoding allowance.
