# DSV4 Typed Snapshot Decode

Date: 2026-08-17

Status: **PREREGISTERED**. No candidate timing result has been admitted.

## Question

`decode_causal_snapshot` currently reads four complete wire sections into
`Vec<u8>`, verifies the record digest, then allocates equally large typed arenas
and converts the bytes. Can it instead hash and parse bounded little-endian
chunks directly into the final typed arenas while preserving the v1 wire ABI,
error taxonomy, validation order, and failure atomicity?

This is a host snapshot-load representation deletion. It does not change model
state, Metal restore, checkpoint publication, wire bytes, or GPU execution.

## Allocation Model

Let `B` be the sum of prefix, raw, compressor, and published section bytes.
Current successful decode requests `B` bytes of raw vectors plus `B` bytes of
typed vectors, for a logical payload peak of `2B`. The candidate retains only
the final typed `B` plus a fixed 4 KiB read buffer.

Production Flash-0731 fixtures:

| Position | Payload bytes | Record bytes | Deleted logical bytes |
|---:|---:|---:|---:|
| 6,144 | 60,137,472 | 60,153,888 | 60,137,472 |
| 97,040 | 685,862,976 | 685,879,392 | 685,862,976 |

The 6,144 cell represents the repository's common 60 MB snapshots. The 97,040
cell is a decisive memory/wall falsifier below the explicit 1 GiB file cap.

## Candidate

- Control: `QWEN_DSV4_STREAMING_TYPED_DECODE=0`, retaining the byte-vector then
  typed-vector decoder.
- Candidate: `QWEN_DSV4_STREAMING_TYPED_DECODE=1`, reserving the final typed
  section once, reading aligned chunks, hashing the exact wire bytes, and
  extending with `from_le_bytes` words.
- Public decode selects the implementation once per process; unit tests call
  both implementations explicitly.
- F32/F16 bit patterns remain integer words. The final snapshot fields remain
  `Box<[u16]>` and `Box<[u32]>`.
- The existing decoder remains as a rollback path until qualification.

## Correctness Gates

Before timing, both implementations must pass:

1. Equal decoded snapshot fields for canonical production-geometry fixtures.
2. Byte-identical `encode(decode(record))` output and equal record, prefix, and
   causal digests.
3. Exact existing errors for budget rejection, unsealed corruption,
   truncation, trailing bytes, header drift, and resealed semantic corruption.
4. Arbitrary short reads and `Interrupted` reads without digest or output drift.
5. Missing fixture/output paths fail the ignored floor rather than skip.

Any semantic, wire, digest, or error-class mismatch kills and removes the
candidate regardless of timing.

## Fixture Protocol

An unmeasured release-test child creates each immutable fixture from the
production Flash-0731 geometry, writes mode `0600`, and emits a machine packet
containing:

- exact position, section lengths, payload and record bytes;
- file SHA-256 and record BLAKE3;
- prefix and causal digests;
- test-binary SHA-256 and the stable benchmarked commit.

Fixture generation and re-encoding never occur in measured children. Before
each pair, the runner authenticates the immutable fixture and deliberately
warms the same bytes for both arms.

## Timing Protocol

For each fixture, run six fresh-process pairs in frozen order:

```text
AB BA BA AB AB BA
```

A is legacy and B is typed streaming. Each child opens and decodes exactly once,
retains the decoded snapshot until process exit, records decode microseconds and
decoded digests, and is wrapped by `/usr/bin/time -l`. No arm is retried or
discarded. All events are serialized in one chronology with build identity
before and after the campaign.

Primary 97,040 gates:

- candidate wins decode wall in at least five of six pairs;
- median paired wall saving is positive in both AB and BA strata;
- marginal median decode-time reduction is at least 5%;
- median maximum-RSS deletion is at least 548,690,381 bytes (`0.80B`); and
- peak-footprint direction agrees with RSS.

Supporting 6,144 gates:

- no more than 2% marginal median wall regression;
- positive paired wall saving in both order strata; and
- median maximum-RSS deletion is at least 48,109,978 bytes (`0.80B`).

## Decision Rule

- **GO**: all correctness gates and both fixture gates pass. Promote streaming
  decode default-on with the environment variable as rollback.
- **SIZE-GATED GO**: the 97,040 cell passes but the 6,144 wall gate fails;
  retain legacy below a preregistered 128 MiB payload threshold and rerun the
  exact boundary cells before promotion.
- **KILL**: any correctness failure, negative AB/BA median at 97,040, less than
  5% large-record reduction, or less than `0.80B` large-record RSS deletion.

Whole CLI/model-load effects are out of scope. A passing result authorizes only
the snapshot decoder and does not imply inference or GPU speedup.

## Safety Boundary

All work is CPU-only and model-free. No GGUF, Metal context, residency set,
`requestResidency`, pre-wire, `mlock`, cache-bypass read, uncached read, or
residency-coupled pread path may run. PID 8770 remains user-owned and untouched.
