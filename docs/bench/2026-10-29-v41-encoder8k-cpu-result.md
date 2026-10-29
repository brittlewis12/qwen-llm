# Fixed-8K encoder: CPU feasibility passes, execution still closed

## Decision

Continue to explicit fixed-width admission and encoder execution proof. This
checkpoint establishes CPU plan, geometry, bounds and synthetic resource-ledger
feasibility, not an 8K runtime, numerical, device-admission or performance result.
Production still selects at most 4096 encoder rows; decoder replay stays at most
2048 rows. The test-only 8192 planning scope does not open execution guards.

At source `96325bc952277e9248871bf88ba5593ba9438705`, tree
`bc18747347d0d4f3d3ffd4917bb0db25a10cd0e3`:

- The offline/locked Rust 1.99.1 release library-test binary build succeeds on
  `aarch64-apple-darwin`; the separate non-test release library typecheck passes.
- Actual discovery finds 2919 library tests and reconciles the maintained scope
  and ignore set. All 350 explicitly classified library CPU tests pass, with zero
  failures or ignored tests executed. This includes the eight new feasibility
  tests; the two fixture ledgers also pass in a separate focused run.
- All 26 focused fixture quotes satisfy the unchanged 128 MiB cap. Compiler,
  discovery, CPU checks and typecheck run without a GPU lease. No Metal context,
  kernel, artifact import or model execution is part of this packet.

The recorded scope is the library CPU inventory, not acceptance of the oracle,
benchmark or CLI test targets.

## Executed Synthetic Ledger

The fixture uses default `Config`, `Schedule` and `HashConfig`, prompt length 8192
and capacity 8448, with two assumed Engram projection geometries `(6144,25600)`.
Pricing also checks capacity 16384. These are model-shaped source inputs, not
metadata read from a real artifact. The supplied heap size equals logical bytes,
with assumed alignment 256, page 16384 and 1024 extra per nonempty allocation.
The existing checked upper-bound arithmetic runs over these assumptions; they
are not measured device heap prices or a proven device upper bound.

With half staging and encoder prefetch/pipeline enabled in both configurations:

| Runtime Fixture Term | Two 4096-Row Chunks | One 8192-Row Chunk |
| --- | ---: | ---: |
| Metal bytes | 11,399,540,168 | 11,199,335,880 |
| CPU bytes | 1,815,174,906 | 1,840,340,730 |
| Largest logical allocation | 512 MiB | 1 GiB |

One 8K chunk replaces two co-live pipelined 4K scratch sets rather than doubling
their combined storage. In this fixture, Metal demand falls 200,204,288 bytes
(190.93 MiB), while CPU demand rises 25,165,824 bytes (24 MiB). Shared retained,
seed, status and reserve terms compare equal. Runtime pricing adds the maximum
of Full and CED incremental storage to the base request, not their sum.

These totals exclude actual resident-model load, binding and caller demands.
They establish neither real cold peak nor current host/device availability.
Admission remains mandatory and is not a reservation. The larger first chunk
can also expose preparation that the two-chunk pipeline overlaps; memory
feasibility predicts no speedup.

## Geometry And Fixture Repair

The executed plan checks preserve 20 encoder layers and 163840 encoder output
rows, 8192 publication rows, 24150 decoder output rows and 2540 seed rows.
Boundary rows remain `5651..8192`, candidate-query rows `5778..8192`; the decoder
plan and sixteen 512-row query tiles match. These are geometry/lineage checks,
not proof that every projection, routed expert or GPU output has equal bits.
Maximum-buffer refusal, integer overflow, scope restoration/thread isolation,
and fixed-8192 admission success/refusal without a smaller-width retry pass.

Two earlier attempts remain preserved. The first stops before test execution on
six missing inventory entries: two host-only cursor tests and four ignored
Metal fault/drain tests. Explicit classification repairs the registry without
running those GPU tests. The next passes all eight new tests but fails two old
128 MiB fixture ledgers after default-on pipelining adds a second live chunk.

The source-fixed fixtures retain pipelining, every original row/window/chunk
boundary, numerical oracle and cap. Only the large-window cohorts W1025/W2050
use block 8 candidate slabs; a shared helper configures both CPU quotes and GPU
model/layer metadata. Short-window block 2, wide and separate P4 block 128 fixtures
remain intact. Quote helpers use the existing pipeline-plus-prefetch predicate.
The maximum of all 26 executed synthetic fixture quotes is 129,317,612 bytes,
below 134,217,728. This CPU result does not transfer historical GPU parity to the
changed fixture geometry.

## Evidence And Next Gate

- Frozen release library-test binary SHA256:
  `c62a8c6d84b2937c26788c8298a7e15821933d093a060934ae65f4a034ff18d7`.
- Accepted 33-file CPU evidence manifest SHA256:
  `da4d6f50dd3ce70432b88c0af1b44f931744e13208b318759d1de232536cee8c`.
- The non-test typecheck is recorded separately from that CPU-test manifest.
- The prior pre-execution refusal and partial CPU run remain failed, sealed and
  separate; no failed result is replaced or pooled into acceptance.

The pricing selector is
`deepseek_v41::forward::chunk_geometry::encoder8k::pricing_compares_8k_with_two_4k_pipeline_at_full_request_capacity`.
Method and proposed later gates are in
`docs/bench/2026-10-29-v41-encoder8k-feasibility-protocol.md`.

Next, carry explicit 8192 width through normal cold/request/state/admitted plans,
without overwriting a 4K admission or silently falling back. Encoder-only guard
changes need fresh review and qualification; decoder/query-tile limits and
defaults stay unchanged. Real-header admission, checked completion/cancellation,
exact Full/4K-CED/8K-CED vocabulary and causal-state comparison, actual route
populations and inclusive timing all remain ahead. No GPU run, default promotion
or performance claim follows from this CPU checkpoint.
