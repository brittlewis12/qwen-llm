# Qwen Restored-Prefix V_T Rebuild Screen

Date: 2026-08-17

Status: **PREREGISTERED, AMENDED BEFORE TIMING**. No candidate timing has run.

Baseline commit: `dda7a5e7e8f0cb63c4b8e29981d991ecf65dd9fe`.

## Question

On every packed-prefill invocation,
`attn_matrix_vt_valid_until` starts at zero. If a restored or otherwise resumed
Qwen prefix begins at position `P > 0`, each attention layer first scatters the
new `C` rows into canonical V and V_T, then transposes canonical V over
`0..P+C`. This reconstructs the entire transposed-V sidecar and rewrites the
current chunk a second time.

Is that exact derived-cache rebuild large enough at agentic 8K--32K prefixes to
justify a product-shaped attribution campaign? Separately, is the local no-risk
cleanup of transposing only `0..P` measurable enough to retain?

This model-free experiment measures isolated steady-state kernel-body cost. It
is not a strict bound on product savings: production interleaves each layer's
transpose with substantial compute, has different cache pressure, and still
needs some representation transfer under any cache design. Phase 0 can reject a
low-ceiling idea; it cannot authorize cache ownership or claim TTFT savings.

## Current Cost Model

For Qwen3.8 27B G6:

- 16 attention layers;
- 4 KV heads;
- head dimension 256;
- F16 canonical V and V_T; and
- 2 KiB of V per token per attention layer.

The exact read-plus-write traffic is
`L * n_kv * head_dim * rows * 2 bytes * 2 directions`. Across all layers this
is 64 KiB per row. A prefix-only B arm moves 512 MiB, 1 GiB, and 2 GiB at 8K,
16K, and 32K. A moves an additional 8 MiB for `C=128` or 64 MiB for `C=1024`.
The redundant current-chunk overlap is therefore proportional to `C`, not
`P+C`.

The current implementation is at
`crates/qwen-llm/src/metal_dflash.rs:7568`,
`crates/qwen-llm/src/metal_dflash.rs:8732`, and
`crates/qwen-llm/src/metal_dflash.rs:9225`. The fused current-chunk V_T scatter
is at `crates/qwen-llm/src/metal_dflash.rs:8758`.

## Amendment A1: Compact The 256x V_T Dispatch Grid

An adversarial review of the unmeasured A0 harness found a stronger defect in
the retained host encoder. `KernelEncoder::dispatch` calls Metal's
`dispatchThreadgroups`, but `encode_attn_matrix_transpose_v_f16` passes the total
element count as the threadgroup count and also requests 256 threads per group.
The shader receives `thread_position_in_grid`, computes the same total, and
returns whenever `tid >= total`.

For `T = n_kv * head_dim * rows`, current code therefore launches `T`
threadgroups and `256T` thread slots. Only `T` slots copy an element; the other
`255T` return. The compact grid removes `T - ceil(T/256)` threadgroups and
`256 * (T - ceil(T/256))` rejected slots; its final partial group can retain up
to 255 rejected slots. Its exact Metal width is `T.div_ceil(256)` threadgroups.
This is a host-dispatch repair, not a new kernel or numerical regime.

This was discovered before candidate code or timing. One small, untimed
correctness run exercised the existing dispatch while validating the harness's
layout premise; it supplied no performance sample. A0's large-cell timing would
primarily measure this 256x dispatch error and is superseded by A1 below.

### A1 Arms

Add a default-off experiment flag,
`QWEN_ATTN_MATRIX_VT_COMPACT_DISPATCH=1`, and one internal explicit-mode encoder
so the same release binary can exercise both grids:

- **D0 / legacy:** full `0..P+C` transpose with `T` threadgroups of 256.
- **D1 / compact:** the identical full transpose with `ceil(T/256)`
  threadgroups of 256.
- **D2 / compact prefix-only:** compact dispatch over `0..P`, preserving the
  current chunk already written by fused scatter.

All tensor arguments, element ownership, storage modes, command-buffer
boundaries, and shader code remain identical between D0 and D1. D1 changes only
the number of groups whose threads are outside the shader's admitted range.
Checked host arithmetic must prove every shader `uint` argument and product fits
before either dispatch, including the selected grid's padded maximum global
thread ID. If promoted, compact becomes default and `=0` becomes the rollback.

### A1 Correctness Gates

Before timing:

1. A pure host test pins `T=1,255,256,257` and proves legacy/compact slot
   coverage, with compact groups exactly `ceil(T/256)`.
2. Independent legacy and compact destinations match an explicit CPU transpose
   bit-for-bit across nonzero `base_pos`, edge totals, multiple KV heads, and
   sentinel padding.
3. The restored-prefix scatter fixture constructs its CPU oracle solely from
   the original prefix plus CPU-rounded current F32 values. Both canonical
   caches and both V_T outputs must match it; GPU output may not define expected
   data.
4. Every timed command buffer must complete successfully before timestamps are
   published.

Any mismatch or D1/D2 command-buffer failure kills the corresponding candidate.
A D0 failure makes the dispatch-comparison packet invalid and inconclusive; it
does not count against D1.

### A1 Model-Free Protocol

The A1 protocol explicitly retains A0's per-layer ordinary buffers,
deterministic first-touch, fail-closed per-layer `maxBufferLength` check,
independent X/Y banks, one serial encoder, 16 forward-order layer dispatches,
commit-to-completion wall timer, Metal GPU timestamps, median definition, and
balanced schedule. Warmups, measured arms, and all GPU work are serialized. No
arm may be retried, discarded, or replaced. No cooling wait, sleep, or other
discretionary delay may be inserted; only command completion and ordinary fresh
process startup separate arms. The authoritative fresh-process chronology is
now:

```text
dispatch D0/D1: P512/C128, P2048/C128, conditional P8192/C128
compact D1 only: P16384/C128, P32768/C128
overlap D1/D2: P32768/C1024
```

Paired cells use:

```text
A_X B_Y, B_X A_Y, B_Y A_X, A_Y B_X, A_X B_Y, B_X A_Y
```

where A/B mean D0/D1 in `dispatch` mode and D1/D2 in `overlap` mode. Compact-only
cells run six measured D1 arms with frozen banks `X,Y,Y,X,X,Y`. Frozen warmups
are `D0_X,D1_Y,D0_Y,D1_X` for dispatch, `D1_X,D1_Y` for compact-only, and
`D1_X,D2_Y,D1_Y,D2_X` for overlap.

Before every overlap warmup and measured arm, an untimed, completed compact
transpose of exactly `P..P+C` populates that bank's suffix from its canonical
source. This common preparation is outside both timed arms and prevents D2 from
preserving a sentinel or stale output.

The collector requires a tracked-clean worktree, records build-time `HEAD` and
`HEAD^{tree}`, builds the release libtest artifact once, copies it to an
immutable campaign-owned path, resolves that path, and hashes it. Every child
must retain the same `HEAD`, tree, clean status, and binary hash. Each geometry
then runs that exact copied binary in one fresh process with this invocation,
plus the cell-specific mode/prefix/chunk environment:

```text
TEST_BINARY 'metal::tests::attn_matrix_vt_rebuild_screen' \
  --ignored --exact --nocapture --test-threads=1
```

Cargo-mediated execution, substring filtering, or invoking all ignored tests is
forbidden. Every child records its PID, exact argv, resolved executable, binary
SHA-256 immediately before and after execution, and `HEAD` immediately before
and after execution. Any drift invalidates the packet.

Before measured children, the same binary runs these three full test names
individually with `--exact --nocapture --test-threads=1`:

```text
metal::tests::attn_matrix_vt_dispatch_groups_cover_exact_thread_range
metal::tests::attn_matrix_vt_compact_dispatch_matches_legacy_nonzero_span
metal::tests::attn_matrix_vt_prefix_rebuild_preserves_scattered_suffix
```

Each must report exactly its named test and one pass. A separate exact,
untimed `metal::tests::attn_matrix_vt_environment_probe` runs before and after
the measured chronology, must likewise report one named pass, and records the
Metal device registry ID, device name, and `maxBufferLength` without allocating
benchmark banks.

The P8192 D0 cell runs only if no measured P2048 D0 arm wall time exceeds
1,000 ms and `4 * median(measured D0 wall) <= 2,000 ms`; warmups are excluded.
Otherwise the collector records the preregistered D0 safety skip and runs the
six frozen D1 compact-only samples at P8192 instead. No running command is
killed. A safety skip is not a D1 failure and does not authorize a P8192
dispatch speedup claim.

The collector must fail on missing cells, duplicate arms, nonzero child status,
command-buffer failure, identity drift, or any schema/order mismatch. It emits
the preregistered paired values, wins, medians, strata, exact bytes, and gates;
the test emits only authenticated raw records. The only permitted absent cell
is the P8192 D0 arm with the exact safety predicate above recorded as true; P8192
D1 remains mandatory.
Environment provenance comes from the collector, not the test body. Before and
after the chronology it records binary SHA-256, implementation commit, device,
`maxBufferLength`, OS, physical memory, power source, memory pressure, thermal
state, and protected-PID status.

Preflight build, correctness, or provenance failure creates no campaign
chronology and may be repaired and rerun. Immediately before launching the first
measured child, the collector creates the immutable campaign packet. For every
child it atomically records command, identity, timestamp, and PID before waiting;
after completion it atomically records status/stdout/stderr before attempting
JSON parsing. A malformed record remains evidence and ends the campaign; no
attempt may be discarded or retried. Existing campaign evidence is never
overwritten.

Stdout and stderr are retained as exact base64-encoded bytes with SHA-256,
alongside PID, return code or terminating signal, and spawn/completion
timestamps. Packet writes use write-flush-fsync-rename plus parent-directory
fsync. A post-chronology identity, environment, protected-PID, or Metal-probe
failure invalidates the existing packet and is not covered by the repairable
preflight exception.

If a command-buffer failure identifies D0, the dispatch result is
**INCONCLUSIVE** rather than a D1 failure. A D1 failure is a D1 **KILL**; a D2
or `PREP_D2` overlap-preparation failure is a D2 **KILL**. Unattributed process
or infrastructure failure invalidates the packet. The harness must include the
active `D0`, `D1`, `D2`, or `PREP_D2` label in any command-failure record.

All bandwidth values are algorithmic logical bytes,
`4 * L * n_kv * head_dim * rows`, divided by GPU time. They are not claims about
physical DRAM traffic.

### A1 Decision Gates

D1 proceeds to a separately preregistered restored-prefix product A/B only if:

- all correctness gates pass;
- at the largest completed paired dispatch cell, D1 wins all six pairs on GPU
  time, where a strict `D0_gpu > D1_gpu` is a win and a tie is not;
- GPU paired saving is `D0_gpu - D1_gpu`; each three-pair AB/BA stratum median
  is its middle sorted saving and must be positive;
- GPU speedup is `median(D0_gpu) / median(D1_gpu)` and must be at least 8x both
  overall and separately within the AB and BA strata;
- D1 median GPU time is nondecreasing from 8K through 32K; and
- the median of the six per-arm D1 exact-byte rates is 50--800 decimal GB/s at
  16K and 32K. Each rate uses that arm's GPU delta; the six-value median is the
  arithmetic mean of the middle two sorted values.

The local D2 source change remains separate. Retain it only if D2 wins at least
five of six P32768/C1024 pairs under the same strict GPU rule, each order
stratum's middle paired `D1_gpu - D2_gpu` saving is positive, the median of all
six paired GPU savings is at least 0.25 ms, and no correctness gate changes.
Wall results are supporting and never replace GPU gates. Otherwise remove D2
even if D1 is promoted.

No isolated result authorizes V_T cache duplication, snapshot ABI changes, or a
TTFT claim. It authorizes only the product campaign for the exact dispatch fix.

## Superseded A0: Model-Free Screen

Add one ignored release-test harness around the retained
`encode_attn_matrix_transpose_v_f16` kernel. It must not load a GGUF or execute
model code.

Exact geometry:

- `L=16`, `n_kv=4`, `head_dim=256`;
- restored prefixes `P={8192,16384,32768}`;
- suffix chunks `C={128,1024}`; and
- `n_pos=vt_stride=P+C`.

Allocate two independent source/destination bank sets X and Y, with one ordinary
Metal buffer per layer. Every dispatch must therefore use a distinct layer
buffer, and the two arms in a pair may not alias. Per-layer allocation avoids
assuming a buffer larger than the device's `maxBufferLength`; the harness must
record that limit and fail rather than skip if one layer does not fit.

Fully initialize and first-touch every source and destination page before
warmup. Sources use a deterministic nonzero pattern and destinations use
arm-specific sentinels. The resulting 32K working set is several GiB and uses
16 distinct source/destination pairs per arm; it must not replay one layer's
SLC-resident bytes.

Arms:

- **A / current:** call the kernel with `base_pos=0`, `n_rows=P+C`,
  `n_pos=P+C`, and `vt_stride=P+C` for all 16 layers.
- **B / overlap deletion:** call it with `base_pos=0`, `n_rows=P`,
  `n_pos=P+C`, and `vt_stride=P+C`; the common fused scatter of `P..P+C` is
  excluded from both timed arms.
- **Z / optimistic body estimate:** zero rebuild time. This is arithmetic, not
  a timed implementation or product bound; A's raw time is only the isolated
  body potentially removable.

Each arm uses one command buffer, one serial compute encoder, and 16
forward-order dispatches. The frozen warmup is `A_X, B_Y, A_Y, B_X`. The harness
then runs six fresh command-buffer pairs in frozen order with balanced physical
bank ownership:

```text
A_X B_Y, B_X A_Y, B_Y A_X, A_Y B_X, A_X B_Y, B_X A_Y
```

No pair may be retried or discarded. Wall timing starts immediately before
`commit` and ends after `waitUntilCompleted`; GPU time is the command buffer's
start/end timestamp delta. The median of six is the arithmetic mean of the
middle two sorted values. Report every raw arm, paired `A-B`, AB and BA medians,
wins, exact per-arm logical bytes, and per-arm decimal GB/s (`bytes / 10^9`). GPU
workloads are serialized.

Run one geometry per fresh test process in this frozen order:

```text
P8192/C128, P8192/C1024, P16384/C128,
P16384/C1024, P32768/C128, P32768/C1024
```

Record test-binary SHA-256, implementation commit, device name,
`maxBufferLength`, macOS version, physical memory, power source, memory pressure,
and thermal state before and after the chronology. No cell order changes,
retries, exclusions, or post-hoc cooling waits are allowed.

## Superseded A0 Correctness

Before timing, a small deterministic nonzero-prefix fixture must use independent
A and B canonical/V_T destinations and establish that:

1. fused scatter writes current rows to canonical V and V_T with identical F16
   bits;
2. after a fresh scatter into each arm, full `0..P+C` transpose and prefix-only
   `0..P` transpose produce bit-identical V_T over every visible element and
   match an explicit CPU layout oracle; and
3. prefix-only transpose does not overwrite current rows or capacity padding.

The fixture must use patterned prefix/current rows, multiple KV heads and
dimensions, arm-specific sentinel padding, and `vt_stride > n_pos`. Existing
matrix-attention CPU oracles remain unchanged. Any mismatch kills and removes
the overlap-deletion candidate before timing. A retained source change also
requires an integrated nonzero-prefix matrix-attention test.

## Superseded A0 Decision Gates

The isolated screen permits, but does not authorize, a product attribution only
if:

- median A GPU time is at least 5.0 ms at `P=32768` in both suffix cells;
- the median is nondecreasing across 8K, 16K, and 32K for each `C`;
- A's exact-byte effective rate lies between 50 and 800 GB/s in every 16K/32K
  cell; values outside that preregistered sanity range invalidate the packet
  rather than prove or kill the optimization.

The local prefix-only source change is retained only if:

- B wins at least five of six pairs at `P=32768,C=1024`, with positive median
  paired saving in both three-pair AB and BA strata;
- median paired GPU saving at that cell is at least 0.25 ms; and
- no cell's B median exceeds A by more than
  `max(0.05 ms, 0.02 * A_median)`.

Otherwise remove the prototype and bank a **KILL**. Exact byte deletion alone is
not sufficient to retain an unmeasurable branch or source complication.

## Superseded A0 Phase 1

If the 32K screen passes, preregister a separate product A/B before changing
snapshot or cache ownership. Its control is the enabled online matrix path after
canonical prefix restore, not forced V4. It must report warm TTFT, packed suffix
wall, rebuild attribution, extra cache bytes, cache-capacity loss, and exact
continuation agreement at suffixes 128 and 1024.

A cache-owned V_T duplicate costs 1 GiB at 32K for this geometry. It is not a
no-lose optimization. Pointer-plus-position validity is also insufficient:
scratch can be reused with another session, and the same session buffers can be
overwritten by a different equal-length restore. Any persistence design needs a
session mutation epoch, typed lease, or cache-entry ownership that fails closed
on restore, rewind, and cross-session reuse.

## Safety Boundary

Phase 0 is model-free and uses ordinary Metal buffers only. It must not load a
GGUF or exercise a residency set, `requestResidency`, pre-wire, `mlock`,
cache-bypass read, uncached read, or residency-coupled pread path. PID 8770 is
user-owned and must remain untouched.
