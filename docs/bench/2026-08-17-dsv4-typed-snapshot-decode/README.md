# DSV4 Typed Snapshot Decode

Date: 2026-08-17

Status: **PREREGISTERED AMENDMENT A1** after a decisive parsed-streaming
**KILL**. No direct-word-fill timing result has been admitted.

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

## Parsed Streaming Result: KILL

The first implementation followed the literal candidate above: 4 KiB wire
chunks were hashed and converted word-by-word with `from_le_bytes`. Its
representation deletion was real, but its large-record wall result failed every
preregistered performance gate:

| Position | Legacy median | Parsed median | Reduction | Wins | RSS deletion |
|---:|---:|---:|---:|---:|---:|
| 6,144 | 216.375 ms | 216.138 ms | +0.11% | 4/6 | 60,194,816 B |
| 97,040 | 2,604.258 ms | 2,861.602 ms | **-9.88%** | 0/6 | 685,588,480 B |

At 97,040, AB and BA median savings were `-256.952 ms` and `-251.710 ms`.
Peak-footprint deletion was 686,080,528 bytes. This is a clean **KILL** for
per-word parsed streaming; `parsed-streaming-results.json` retains every arm and
gate. No arm was retried or excluded.

## Amendment A1: Native Direct Word Fill

The codec already rejects non-little-endian hosts. A `u16`/`u32` typed arena on
the admitted host therefore has the exact wire byte order, and both word types
accept every bit pattern. Before any A1 timing, the candidate changes to:

1. fallibly reserve the final `Vec<T>` with length zero;
2. read into one fixed aligned wire buffer;
3. hash those exact bytes;
4. copy bytes into checked spare typed capacity; and
5. set typed length only after the complete section arrives.

This preserves truncation failure atomicity and the one-payload allocation
model, but removes hundreds of millions of `from_le_bytes` conversions. The
unsafe boundary is limited to `T: bytemuck::Pod`, checked byte/count arithmetic,
and writes within reserved capacity. All original correctness and performance
gates remain unchanged. A1 restarts fixture generation and every pair from zero;
the parsed-streaming campaign contributes no A1 timing sample.

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
