# Qwen Restored-Prefix V_T Product A/B

Date: 2026-08-17

Status: **PREREGISTERED**. No model timing has run.

Authority commit: `26e3f8e` (`perf(qwen): delete redundant V_T suffix rebuild`).

## Question

On the pinned Qwen3.8 27B Q4_K_M model, after an exact canonical snapshot
restore, does compact V_T transpose dispatch materially reduce packed-suffix
latency without changing routing, allocation, logits, recurrent state, or
greedy continuation?

The model-free authority packet at
`docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/` found that the retained host
encoder launches one 256-thread threadgroup per element when compact dispatch
is disabled. D1 reduced isolated V_T rebuild GPU time by 197--217x and passed
every preregistered gate. This campaign is the separately required product
attribution. It does not compare cache against no cache and cannot authorize a
persistent V_T cache or snapshot ABI change.

## Pinned Asset And Route

- model: `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`;
- bytes: `17,106,773,984`;
- SHA-256: `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`;
- expected attention topology: 16 G6 layers, 24 Q heads, 4 KV heads, head
  dimension 256; and
- F16 canonical K/V plus the scratch-owned F16 V_T sidecar.

The runner must hash and stat the asset before building the chronology. Model
loading uses ordinary pageable loading with prefetch disabled. It must not use
`MTLResidencySet`, `requestResidency`, pre-wire, `mlock`, cache-bypass reads,
uncached reads, or a residency-coupled pread path.

## Prerequisite Repairs

The existing `qwen-bench prefix-cache` helper takes
`(prefill_chunk, prompt_len)`, but its three long-prompt call sites pass
`(total_len, prefill_chunk)`. That accidentally makes the whole request the
scratch block size above 1K. Correct those calls to `(prefill_chunk, total_len)`
and add a pure regression test before any product timing. This repair applies
identically to control and candidate and is not itself measured by this packet.

The treatment flag is cached in a `OnceLock`, so mutating the process
environment between arms is invalid. Add a scoped, thread-local explicit
override around the existing resolver. Ordinary callers retain exactly the
environment-selected behavior; the dedicated harness selects legacy or compact
for one synchronous packed-prefill call. Add scoped dispatch telemetry that
records only calls made inside that invocation.

The override and capture use non-Send RAII guards, reject nesting, restore on
unwind, record the owner thread ID, and reject any dispatch observed from
another thread. The synchronous packed-prefill call, all 16 resolver decisions,
and every telemetry write must occur on that owner thread.

Normal packed prefill currently checks final command-buffer status only for a
special supplied-tail evidence path. Before timing, make every ordinary packed
chunk fail closed after `waitUntilCompleted` and before GPU timestamps or CPU
readback: status must be `Completed` and `error()` must be absent. Diagnostic
phase-splitting and oracle environment modes are removed by the collector and
are forbidden in this packet. The harness attributes any returned failure to
the active A/B arm.

## Arms And Geometry

Both arms include the retained prefix-only D2 rebuild and differ only in the
number of V_T transpose threadgroups:

- **A / legacy:** compact dispatch `false`; for each attention layer, launch
  `4 * 256 * P` threadgroups of 256.
- **B / compact:** compact dispatch `true`; for each attention layer, launch
  `ceil((4 * 256 * P) / 256) = 4P` threadgroups of 256.

Shader code, admitted global IDs, tensor arguments, canonical snapshot, suffix
tokens, session capacity `P+C+8`, scratch plan, command-buffer boundaries, and all
other environment and argv values remain identical.

Frozen cells, in order:

```text
P8192/C128
P16384/C128
P16384/C1024
```

P16K is the largest legal legacy cell: its padded maximum global thread ID is
exactly `u32::MAX`. P32K legacy would exceed the shader's uint global-ID range
and is forbidden; do not weaken that source check. The earlier model-free
packet's P32K compact-only row remains scalability evidence, not a product A/B.

Prefix and suffix IDs are tokenizer-produced deterministic filler sequences,
truncated to the exact lengths above. Record the full i32-LE token hashes. The
prefix is built once per cell with a 1024-row packed scratch and snapshotted once.
Prefix scratch has matrix capacity exactly P and query rows exactly 1024. Each
suffix uses one packed call with block size `C` and matrix capacity `P+C`.
Two independently allocated suffix scratch banks X/Y must have equal plans,
logical bytes, maximum bytes including deferred allocations, allocation counts,
deferred-allocation counts, matrix capacity, and query rows. They must also
prove that no mutable Metal buffer aliases across banks.

## In-Process Protocol

Each cell runs in one fresh process and loads the model once. Before measured
arms it:

1. allocates only prefix scratch and one builder sequence;
2. runs one ordinary one-token warmup;
3. builds the canonical prefix once and creates one immutable typed
   `SessionSnapshot` with a section-and-length-delimited SHA-256 fingerprint;
4. explicitly drops builder sequence, prefix logits, and prefix scratch, then
   verifies the snapshot fingerprint is unchanged;
5. performs checked peak-memory admission and only then allocates disjoint
   suffix banks X and Y;
6. runs one correctness A/B from that same snapshot object; and
7. runs untimed warmups `A_X, B_Y, A_Y, B_X`.

Builder, correctness, warmup, and measured sequences all have capacity exactly
`P+C+8`. Every arm creates a fresh sequence, calls the production `MetalSession`
snapshot-restore primitive with the same immutable snapshot and identity,
advances the sequence position by exactly P, runs the exact suffix once,
advances the sequence position by exactly C immediately after successful packed
prefill, selects the first greedy token, and drops the sequence. No arm reuses a
mutated session. Lookup policy and cache indexing are deliberately excluded
because they are common to both arms and are not the changed work.

Peak admission uses checked arithmetic after model load and snapshot creation,
but before X/Y allocation. Its conservative bound includes current Metal
allocation, the model file bytes, both scratch-plan maximums, one measured
session allocation delta, the retained prefix snapshot, one temporary
post-suffix correctness snapshot, a fixed 2 GiB allocator allowance, and a
fixed 16 GiB reserve. The collector supplies physical memory; the cell fails preflight if
the bound exceeds it, memory pressure is not normal, swap is already growing,
any single planned buffer exceeds `maxBufferLength`, or X/Y alias. Record Metal
allocation size before X, after X, and after Y.

Six measured pairs use the frozen balanced schedule:

```text
A_X B_Y
B_X A_Y
B_Y A_X
A_Y B_X
A_X B_Y
B_X A_Y
```

No measured arm may be retried, replaced, excluded, or reordered. GPU work and
children are serialized. No sleep, cooling wait, or discretionary delay may be
inserted after chronology creation. A running child is never killed.

## Measurements

Each warmup and measured arm emits one strict JSON record with:

- cell, role, bank, pair, order, and sequence;
- fresh-sequence creation wall time;
- exact typed-snapshot restore wall time;
- direct packed-suffix wall time;
- packed-prefill total GPU timestamp sum;
- post-restore TTFT through first-token selection;
- full request wall from sequence creation through first-token selection;
- first token and the complete F32 vocabulary-vector SHA-256 for the final
  suffix token;
- snapshot identity, section lengths/hashes, fingerprint, and bytes;
- scratch-plan identity; and
- captured V_T dispatch calls, rows, elements, threadgroups, compact/legacy
  calls, base-position sum, and `n_pos` sum.

No timing is computed by subtracting independently measured intervals.
Sequence creation and restore are supporting product endpoints; direct suffix
wall is primary. The dispatch capture is topology evidence, not a second timing
clock.

Timing boundaries are exact. Sequence creation ends before restore begins.
Restore timing covers only `restore_from`. Post-restore TTFT begins after a
successful restore and sequence-position advance, immediately before the
scoped override/capture setup, and ends after argmax. Direct suffix wall begins
immediately before the profiled packed call inside the established scope and
ends on its successful return. The required `advance_by(C)` is outside direct
suffix wall but inside post-restore TTFT and full request wall. GPU time is the sum of completed command-buffer
`GPUEndTime-GPUStartTime` deltas returned by that call. Hashing, state snapshot,
continuation, record serialization, and teardown are outside every endpoint.
All command-buffer-splitting trace modes are off.

## Exactness Gates

Before warmups, the correctness A/B must match exactly on:

- restored prefix position P, snapshot identity/fingerprint, and snapshot bytes;
- the complete final-suffix-token F32 vocabulary vector and selected first token;
- post-suffix `kv_n_pos`, canonical K arena, canonical V arena, GDN convolution
  arena, and GDN recurrent-state arena;
- eight correctly advanced greedy continuation token IDs and every continuation
  logits digest; and
- scratch plan and V_T dispatch topology.

The state oracle is a post-suffix `SessionSnapshot` from each fresh arm. Hash
each typed section with its name and length, compare all section hashes and
lengths, then drop it before the next arm. Measured-arm logits must equal the
correctness digest. Any exactness mismatch is a global **KILL**, not a noisy
performance loss.

The continuation starts from the final-suffix logits. For steps 0 through 7,
select argmax, record that token, consume it once at position `P+C+step`,
advance the sequence by one, and retain the returned logits for the next step.
Compare all eight IDs and all eight returned-logits digests. The suffix argmax
is continuation token 1; no token is consumed twice.

Every suffix arm must capture exactly 16 V_T rebuild calls with:

- base-position sum zero;
- row sum `16P`;
- element sum `16 * 4 * 256 * P`;
- `n_pos` sum `16(P+C)`;
- A: 16 legacy calls, zero compact calls, and threadgroup sum equal to the
  element sum; and
- B: 16 compact calls, zero legacy calls, and threadgroup sum `16 * 4P`.

Any route or topology mismatch invalidates the cell. It cannot be scored as a
candidate loss.

## Evidence And Safety

The collector follows the immutable-packet discipline of the model-free
campaign:

- require a tracked-clean committed implementation tree;
- record full `HEAD`, `HEAD^{tree}`, build identity, and preregistration hash;
- build one release `qwen-bench`, copy it to a campaign-owned immutable path,
  resolve it, and record SHA-256 before and after every child;
- remove inherited `QWEN*`, `MTL*`, `METAL*`, and `RUST_LOG`, then allow only
  the collector's explicit non-treatment environment;
- record exact argv, child PID, start/completion timestamps, return status, and
  byte-exact stdout/stderr with SHA-256;
- atomically flush, fsync, rename, and directory-fsync the chronology after
  every state transition; and
- record device registry ID/name, OS, physical memory, AC state, memory
  pressure, thermal state, swap, and protected PID 8770 before and after.

PID 8770 is user-owned: only query it, abort if active, and never signal it.
The collector may not use name-based process cleanup. Existing packet roots are
never overwritten.

Preflight build, model hash, memory admission, correctness, or provenance
failure before the first measured arm creates no authoritative chronology and
may be repaired.
After the first measured arm, malformed output, nonzero child status, identity
drift, environmental invalidation, or missing records consumes and invalidates
the packet; no retry is permitted.

The P16K/C128 cell runs only if the completed P8K cell is valid, no A suffix
wall sample exceeds 10 seconds, and `2 * median(A suffix wall) <= 8 seconds`.
P16K/C1024 then requires a valid P16K/C128 cell, no A suffix wall sample there
above 10 seconds, `8 * median(A suffix wall) <= 40 seconds`, fresh successful
memory admission, normal pressure, and no swap increase. Otherwise record the
preregistered safety stop and return **INCONCLUSIVE**. Do not terminate a
running command.

## Decision Rule

For every endpoint, paired saving is `A-B`; positive favors compact dispatch.
The six-value median is the arithmetic mean of the middle two sorted values.
Each AB/BA stratum has three values and uses its middle sorted value. Report all
raw values, paired savings, wins, ratios, and strata.

Immediate **KILL**:

- any exactness mismatch;
- any B command-buffer or process failure attributable to compact dispatch; or
- any cell where `median(B_i-A_i)` exceeds
  `max(1 ms, 0.02 * median(A_i))` in suffix wall, suffix GPU time, or
  post-restore TTFT. Ties are not wins.

Promote compact dispatch to the product default only if all of these hold:

1. B wins all six pairs in both suffix wall and suffix GPU time in every cell.
2. Every AB and BA suffix-wall and suffix-GPU stratum median is positive.
3. P8K/C128 median saving is at least 500 ms in suffix wall and suffix GPU.
4. Each P16K cell saves at least 1,000 ms median in suffix wall and suffix GPU.
5. At each P16K cell, B wins at least five of six post-restore TTFT pairs, both
   order strata are positive, and median saving is at least 750 ms.
6. Full request wall has at least five wins and positive AB/BA strata at both
   P16K cells. At P8K, `median(B_i-A_i)` must not exceed
   `max(100 ms, 0.02 * median(A_i))`.
7. All identity, dispatch-topology, environment, and evidence gates pass.

If exactness passes but any performance gate misses, retain compact dispatch
default-off and bank **KEEP_DEFAULT_OFF**. A promotion keeps
`QWEN_ATTN_MATRIX_VT_COMPACT_DISPATCH=0` as the exact rollback.

## Required Validation Before Timing

- focused tests for scoped override restoration and dispatch capture;
- ordinary packed-prefill command completion checks before timing/readback;
- existing legacy/compact V_T bit-equality tests;
- existing suffix-preservation and integrated nonzero-prefix G6 tests;
- pure scratch-argument-order regression test;
- `cargo fmt --all -- --check` and `cargo check --workspace --all-targets`;
- collector/analyzer syntax checks; and
- analyzer mutation tests for order, identity, topology, exactness, safety, and
  every decision boundary.
