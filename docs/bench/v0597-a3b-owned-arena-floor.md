# v0.597 A3B Anonymous Owned-Arena Floor

Status: completed at source `1b7a897`. Four-worker C clears every gate; serial B
fails. The packet authorizes one force-only A3B loader pilot and nothing broader.

## Intent

Determine whether planner-shaped anonymous shared Metal buffers can remove a
material fraction of A3B's process-cold weight-copy wall while preserving the
physical byte representation used by copied storage. The current path issues one
`newBufferWithBytes` call for each direct request. v0.595-v0.596 attribute about
`1.99 s` to materializing `22.12 GB` through 733 buffers and show that file-backed
retained views do not preserve warm decode speed.

This is a warm-filesystem-cache, fresh-process host-materialization floor. It is
not a model-load, first-byte, storage-cold, warm-decode, loader, policy, persistent,
server, or broad-family result.

## Frozen Asset And Geometry

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Production-auto native Q8 embedding must be active.
- Direct requests: 733 over `22,123,538,944` logical copied bytes.
- Unique planned views: 732 over `22,123,530,752` bytes.
- Aliases: zero.
- Final-partial-page fallback: one request over 8,192 bytes.
- Planner window: one 16,384-byte-aligned window over `22,123,544,576` bytes.
- Planner gaps: 13,824 bytes.
- Arena physical copy bytes: `22,123,552,768`, including the fallback.
- Resource options: `MTLResourceStorageModeShared` for every arm.
- Descriptor-layout digest: `0x5ae645df5cf7d568`.
- Ordered inventory digest:
  `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`.
- Planner digest:
  `fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af`.

Use the production ordered request inventory and `plan_retained_storage`. Hash the
ordered tensor name, shard, source offset, byte length, dtype, shape, and
materialization kind. Separately hash page size, device `maxBufferLength`, required
alignment, windows, entries, dispositions, offsets, and fallback reasons. Freeze
the GGUF descriptor-layout digest and executable/source identity in every packet.

Any geometry, identity, policy, digest, count, or byte drift terminates the packet.
The clean runner executes one metadata-only geometry control before timing. It
hashes the complete model in the parent; the child never hashes or prefaults model
payload before its authoritative interval.

## Arms

- **A — current copied primitive**: for every ordered direct request, resolve the
  source with `GgufFile::try_slice`, then call production
  `MetalContext::buffer_from` exactly once. Retain all 733 buffers and create a
  bindable `(buffer, offset=0, length)` resolution entry for every request.
- **B — serial owned arena**: allocate one anonymous buffer for every planner
  window and fallback. Copy each complete contiguous source window, including
  gaps, with one serial host copy. Copy the fallback serially. Build all 733
  bindable resolution entries from planner offsets.
- **C — four-worker owned arena**: allocate the same resources as B. Partition
  every complete window into four disjoint page-aligned ranges and copy them with
  exactly four scoped host threads. The fallback remains serial. Thread creation,
  synchronization, and join are charged. Build the same resolution table as B.

Do not add a requested-ranges arena arm. It would restore hundreds of scattered
copies and test a different premise. No Objective-C calls run on C's workers. The
parent retains the source mapping and destination buffer through all joins.

For a window containing `N` pages, C uses boundaries
`page_size * floor(k*N/4)` for `k in 0..=4`. Require exactly four complete,
nonoverlapping ranges whose union is the window and exactly one write per byte.

## Timing And Memory Boundary

GGUF open, model binding, request inventory, planning, Metal context creation,
and file-cache conditioning occur before the timed interval. Do not resolve any
payload source slice, hash payload bytes, touch destination pages, or verify bytes
before timing.

The authoritative `ready_wall_ms` begins immediately before the first source
resolution or Metal allocation. It ends only after all allocation, source
resolution, copying, worker work, output retention, and construction of the full
733-entry bindable resolution table. Every buffer remains alive through the
endpoint and correctness pass.

A reports only fused ready wall because `buffer_from` combines allocation and
copy. B/C also report allocation, source-resolution, copy, and resolution-table
subintervals. Their sum plus timer bookkeeping must reconcile with ready wall.

Capture `getrusage(RUSAGE_SELF)` immediately before and after the authoritative
interval. Timer-local major faults must be zero. Timer-local minor faults are
causal, charged, and reported. Record Metal `currentAllocatedSize` before, at the
ready endpoint, and after explicit resource drop. Record explicit teardown wall.

The runner wraps every child in `/usr/bin/time -l`. Its maximum RSS and peak
memory footprint are conservative process-level gates: every destination page is
written and every buffer remains live through post-timing verification. Also
record process page reclaims, hard faults, block input, and host pageout/swap
state. `currentAllocatedSize` is a resource ledger, not a residency claim.

## Correctness

Correctness runs after the authoritative endpoint and cannot improve its score.
Require:

- exact byte equality for every full copied planner window, including gaps;
- exact fallback equality;
- all 733 resolution entries in bounds, correctly aligned, and byte-equal to
  their source request;
- exact resource lengths and storage modes;
- all logical, unique, fallback, gap, physical-copy, and allocation ledgers to
  reconcile;
- `alias_count == 0`; this packet makes no alias-realization claim.

A mismatch is a terminal packet failure, not a retryable row.

## Process Packet

Run six fresh-process blocks in this fixed order:

```text
ABC
BCA
CAB
CBA
ACB
BAC
```

This places each arm twice in every position and balances every directed
within-block predecessor twice. Processes never overlap. A complete model-file
read and at least 30 seconds of cooldown precede every child. Before cache
conditioning and child launch require AC power, no thermal/performance warning,
and at least 50% memory availability.

Bracket cache conditioning and each child with pageout/swap state. Cache-interval
growth terminates before launch. Child block input or pageout/swap growth
invalidates the complete three-arm block. A host/thermal/I/O-invalid block may be
repeated in the same order at most twice. Failure to obtain six valid blocks is
`inconclusive`. No performance-based optional stopping is allowed.

Hash the model, executable, runner, this contract, and imported helpers. Require
clean matching runtime/build source identity. Normalize QWEN and Metal environment
controls before every arm. Record raw stdout, stderr, resource output, host state,
VM state, command, environment, and parsed row for every attempt.

## Decision

For candidate `X` in `{B, C}` and valid block `i`, calculate:

```text
D_X = median_i(A_i - X_i)
R_X = median_i(X_i / A_i)
W_X = count_i(X_i < A_i)
P_X = 2463.13 / (2463.13 - D_X)
M_X,rss = max_i(peak_rss_X,i) / max_i(peak_rss_A,i)
M_X,foot = max_i(peak_foot_X,i) / max_i(peak_foot_A,i)
```

`2463.13 ms` is the frozen v0.595 copied A3B output-1 spawn-to-first-byte
median. `P_X` is arithmetic projection only; this floor does not measure first
byte.

Candidate X qualifies only if all conditions hold:

- `D_X >= 500 ms`;
- `R_X <= 0.85`;
- `W_X >= 5/6`;
- `P_X >= 1.20x`;
- `M_X,rss <= 1.05` and `M_X,foot <= 1.05`;
- every scored row has zero timer-local major faults, zero block input, and zero
  child pageout/swap growth;
- every correctness and provenance gate passes.

Report A fused throughput as `22,123,538,944 / ready_wall`. Report B/C copy
throughput as `22,123,552,768 / copy_wall` and ready throughput as the same bytes
divided by ready wall. Use decimal GB/s.

Evaluate B and C independently. B passing authorizes only a force-only serial
A3B loader pilot. C passing authorizes only a force-only four-worker A3B loader
pilot. If both pass, select the lower median ready wall. If neither passes, kill
owned-arena loader work under this premise.

## Authority

A passing arm authorizes implementation of one force-only A3B loader pilot using
the exact planner and copy strategy measured here. The pilot must independently
prove bit-exact full state, `>=1.20x` process-cold first byte, `>=0.99x` late
decode and warm prefill, and no loaded output-128 request regression.

No floor result authorizes default-on behavior, asynchronous promotion, mixed
storage, other model families, aliases, converted weights, MTP, split shards,
storage-cold claims, warm-throughput claims, or broad loader integration.
