# v0.624 A3B Direct-Pread Prefetch Suppression

Status: preregistration. No v0.624 child or performance observation exists.

## Intent

Price one changed storage-cold contract: when the already-authenticated
disposable A3B Auto path will populate its final independent weight resources by
direct `pread`, suppress the complete preceding `ColdOnly` phase rather than
first reading the same shard into the unified buffer cache.

This estimates removal of the configured `ColdOnly` phase, including its
residency probe, memory-headroom telemetry, read loop, and outcome construction.
It does not isolate only the read loop.

## Identity And Scope

Import and verify the complete sealed v0.621 GO. Pin packet parent
`bd0c0495bff030b34b022cf22f5e3ef671d5c2be`; this packet commit may add only
this contract and `scripts/profile/v0624_a3b_pread_prefetch_suppression.py`.

Use the exact v0.621 model, host, release `first_byte_spike`, disposable intent,
five-token prompt, output-1 shape, and in-child `MS_SYNC | MS_INVALIDATE` cache
reset. Processes never overlap. No production selector is changed by this
packet.

## Arms And Order

Run six fresh pairs in fixed order:

```text
AB BA BA AB AB BA
```

- A passes `--policy cold-only`.
- B passes `--policy off`.
- Both leave `QWEN_GGUF_PARALLEL_COPY` and every Auto-suppressing environment
  variable absent.
- Both must select native embedding, exact Auto profile `a3b-q4km-v1`, direct
  pread population, and the copied-storage ledger in that order.
- No retries, substitutions, pooled rows, or post-observation gate changes are
  permitted.

## Per-Child Contract

Every child must satisfy all of the following:

- pre-arm residency is exactly `1,350,985/1,350,985` pages;
- invalidation is exactly `1,350,985/1,350,985 -> 0/1,350,985`;
- post-arm residency is at least 99%;
- process-attributed physical reads are `20.50..=20.75 GiB`;
- the first token is exactly `id=11751 piece=" Paris"`;
- A reports one complete `20.61 GiB` prefetch, with one shard prefetched and
  zero skipped;
- B reports policy `Off` and no prefetch phase;
- launch, completion, source/build/model, marker, host, pressure, swap, process
  resource, and artifact evidence are complete and valid.

Invalid host, pressure, process, or artifact evidence stops inconclusive. A
valid performance miss is a KILL.

## Measurements And Gates

Record internal load and first-byte wall, spawn-to-exit wall, A prefetch wall,
parallel-pread allocation/source/copy/binding/ready times, pageins, physical
reads, user/system/total CPU, RSS, and physical footprint.

For each pair compute A minus B load and first-byte savings, physical-read B/A,
total-CPU B/A, physical-footprint B/A, and the absolute difference between load
and first-byte savings.

GO requires:

- B wins load and first byte in all 6/6 pairs and 3/3 in each order stratum;
- overall median load and first-byte savings are each at least `400 ms`;
- AB and BA median load and first-byte savings are each at least `250 ms`;
- every-pair physical-read B/A is within `[0.98, 1.02]`;
- every-pair total-CPU B/A is at most `1.10`;
- every-pair physical-footprint B/A is at most `1.05`;
- the overall median absolute load-versus-first-byte saving difference is at
  most `100 ms`, with no pair above `250 ms`.

## Authority

A complete GO authorizes implementation work on one selector that suppresses
`ColdOnly` only after the existing exact plan has authenticated disposable A3B
Auto direct pread. It does not certify or default-enable the resulting code;
that requires selector-table tests and a post-edit exact-marker confirmation.

The implementation must prepare and consume one Metal load plan rather than
coarsely duplicate A3B recognition before prefetch. Configured `Off` remains
Off, explicit `Always` still runs, and observability must distinguish configured
`ColdOnly` from authenticated suppression.

No authority extends to ForceOnly or reusable loads, JSONL, forced pread or
copy, overrides, other assets or hosts, serving, concurrency, energy,
partial-residency policy, or untouched-media claims.
