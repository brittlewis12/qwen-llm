# v0.599 A3B Topology-Preserving Parallel-Copy Floor

Status: preregistration. No implementation or timing result exists yet.

## Intent

Determine whether a composite topology-preserving parallel-copy primitive can
remove a material fraction of A3B's process-cold weight-materialization wall
without changing the final Metal resource topology that preserves production
decode performance.

v0.597 measured one four-worker, one-window copy at `702.167 ms`, versus
`2062.791 ms` for 733 production `newBufferWithBytes` calls. v0.598 then proved
that both anonymous and file-backed one-window storage incur the same stable
decode tax. This packet tests a composite topology-preserving parallel-copy
primitive without reintroducing that killed topology change. It does not isolate
copy concurrency from batched allocation, source resolution, or manual copy.

This is a warm-filesystem-cache, fresh-process host-materialization floor. It is
not a model-load, first-byte, storage-cold, loaded-request, default-policy, or
broad-family result.

## Frozen Asset And Geometry

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Production-auto native Q8 embedding must be active.
- Direct requests: 733 over `22,123,538,944` bytes.
- Every request is direct, unique, and materialized into one exact-sized buffer.
- Every binding has offset zero and the production request byte length.
- Resource options are exactly those used by production copied storage.
- Page size: 16,384 bytes. Required alignment: 32 bytes.
- Metal `maxBufferLength`: `77,309,411,328` bytes.
- Architecture: `qwen35moe`.
- Descriptor-layout digest: `0x5ae645df5cf7d568`.
- Ordered inventory digest:
  `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`.
- Planner digest, retained only as a metadata identity control:
  `fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af`.

Freeze the GGUF, executable, source, runner, this contract, descriptor layout,
ordered request inventory, request count, byte total, architecture, page size,
and Metal device geometry. Any drift terminates the packet.

## Arms

- **A - production copied**: in exact production request order, resolve one source
  slice and call `MetalContext::buffer_from` once per request. Retain all 733
  resources and construct one `(buffer, offset=0, length)` binding per request.
- **B - parallel copied**: in the same request order, allocate 733 exact-sized
  anonymous shared buffers with `buffer_uninit`. On the parent thread, resolve all
  immutable source slices and destination spans. Copy whole-tensor tasks through
  exactly four scoped workers, join them, then construct the same request-ordered,
  offset-zero binding table.

B must not use planner windows, aliases, fallback resources, nonzero offsets,
file-backed buffers, GPU blits, asynchronous work, or new dependencies. Workers
must not call Objective-C, resolve GGUF slices, allocate buffers, construct
tensors, retain resources, or report errors. All resources remain alive through
post-endpoint verification. Parent validation makes worker bodies infallible; a
worker panic or failed join terminates the packet. The retained planner digest is
an identity control only and does not drive B materialization.

## Worker Schedule

Sort immutable tasks by `(shard_idx, data_offset, request_index)`. Partition that
order into exactly four nonempty contiguous ranges that minimize the maximum
assigned byte total. Among equally optimal complete three-cut tuples, select the
lexicographically smallest tuple. Each worker traverses its complete sorted tuple
range in order.

Record each cut, task count, byte total, first and last source offsets,
`max_bytes / ideal_bytes`, and `max_bytes / min_bytes`. Require complete task
union, no duplicate request, no split tensor, and one destination writer per
request. Freeze prefix-length cuts `[155, 359, 539]`, task counts
`[155, 204, 180, 194]`, and worker bytes
`[5,532,746,240, 5,462,315,776, 5,595,522,304, 5,532,954,624]`.
These give `max/min = 1.024386` and `max/ideal = 1.011687`. Schedule drift is
terminal.

This schedule gives up at most about 1.2% theoretical balance versus LPT on the
frozen inventory while preserving four sequential source regions. LPT is not an
experimental arm.

## Timing And Correctness

GGUF open, model binding, request inventory, Metal context creation, and file-cache
conditioning occur before the authoritative interval. No payload source is
resolved, destination allocated or touched, or payload byte verified beforehand.

`ready_wall_ms` starts immediately before A's first source resolution or B's
first destination allocation. It ends after every allocation, source resolution,
worker spawn/copy/join, resource retention, and binding-table construction. A
reports authoritative fused ready wall and a nested binding diagnostic that is not
added again. B reports allocation, source resolution, worker spawn-to-join,
binding, and unattributed wall. B's components must reconcile with ready wall;
ready wall is authoritative for both arms.

Verification occurs after the endpoint for both arms. Require:

- exact bytes for all 733 complete resources;
- 733 unique resources with exact request lengths and offset-zero bindings;
- creation options Shared/DefaultCache/Default and observed buffer modes
  Shared/DefaultCache/Tracked;
- identical resource modes between paired arms;
- exact logical and physical byte ledgers;
- exact binding-to-request identity and one write per candidate destination;
- zero timer-local major faults and no child block input or pageout/swap growth.

Capture timer-local `getrusage`, Metal allocation before/ready/after drop, explicit
teardown wall, `/usr/bin/time -l`, host state, and VM state as in v0.597. Byte
verification cannot improve the score.

Hash the model, executable, runner, contract, and imported helpers. Require clean,
matching source/build/runtime identity. Normalize QWEN and Metal environment
controls before every child. Retain raw command, normalized environment, stdout,
stderr, resource output, host state, VM state, and parsed rows.

## Process Packet

Run six fresh-process two-arm blocks in this fixed order:

```text
AB
BA
BA
AB
AB
BA
```

Processes never overlap. A complete model read and at least 30 seconds of cooldown
precede every child. Require AC power, no thermal or performance warning, and at
least 50% memory availability before cache conditioning and launch.

Cache-interval pageout/swap growth terminates before launch. Host, thermal, or I/O
invalidity, including timer-local major faults, child block input, or child
pageout/swap growth, invalidates the complete pair. Such a pair may repeat in the
same order at most twice. Geometry, identity, schema, schedule, topology, byte,
mode, or timing-reconciliation mismatch terminates the packet. A valid but slow
pair is never retried. Failure to obtain six valid pairs is `inconclusive`.
Performance cannot trigger retries or optional stopping.

## Decision

For valid pair `i`, calculate:

```text
D = median_i(A_i - B_i)
R = median_i(B_i / A_i)
W = count_i(B_i < A_i)
D_o = median_{i: order_i=o}(A_i - B_i), o in {AB, BA}
P = 2463.65 / (2463.65 - 0.96 * D)
M_rss = max_i(peak_rss_B,i / peak_rss_A,i)
M_foot = max_i(peak_foot_B,i / peak_foot_A,i)
```

`2463.65 ms` is the accepted v0.598 copied output-1 first-byte median. The `0.96`
transfer haircut is the observed ratio between v0.598 model-load saving and
v0.597 floor saving. `P` is arithmetic only. The 500 ms economic gate dominates
the ratio and projection gates at current anchors; the latter remain drift guards.

B qualifies only if:

- `D >= 500 ms`;
- `R <= 0.85`;
- `W >= 5/6`;
- `D_AB >= 500 ms`, `D_BA >= 500 ms`, and at least 2/3 wins in each stratum;
- `P >= 1.20x`;
- both paired maximum memory ratios are `<=1.05`;
- every schedule, topology, byte, mode, timing, and validity gate passes.

Report fused A throughput and B copy/ready throughput using exactly
`22,123,538,944` bytes as the decimal-GB/s numerator. Report both order strata
separately. Require checked finite arithmetic and a positive projection
denominator.

## Authority And Sequence

A passing floor authorizes only implementation of one force-only A3B
parallel-copied loader pilot using the exact resource topology and worker schedule
measured here. It does not authorize immediate product timing: finish the already
measured query-capped auto-prefill admission path first, then return to the pilot.

The loader pilot requires a separate preregistration. It must independently prove
bit-exact full state, 733 unique offset-zero copied resources, and loaded
prefill/decode/complete-request noninferiority before any fresh-process product
packet. This floor does not define or authorize product promotion gates.

No result authorizes default-on behavior, one-window owned storage, asynchronous
promotion, split shards, aliases, converted tensors, MTP, other assets,
storage-cold claims, or broad loader integration.
