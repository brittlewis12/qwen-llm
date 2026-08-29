# Flash-Next Full-Shard Prefetch N=512

Decision: **KEEP** an explicit default-off whole-file prefetch option for the
released three-shard asset. This packet does not authorize default-on warming or
a range-specific successor.

## Protocol

- Source: `e938a250fada78bd8b071619c521ec66547747a8`, rebuilt before acquisition.
- Device: Apple M4 Max with unified memory.
- Model: the checksum-verified internal UD-Q3_K_XL release, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: the tracked natural N=512 prompt with no special tokens, one
  generated token, exact 512-forward capacity, strict router enabled, and
  first/warm/profile packed passes.
- Order: internal demand paging A1, forced full-shard prefetch B, internal
  demand paging A2, with five seconds between arms.
- Cache state: immediately before every process, targeted
  `msync(MS_SYNC|MS_INVALIDATE)` evicted all three mappings and `mincore`
  verified zero resident pages. No machine-wide purge ran.
- Candidate: `QWEN4EXP_FULL_SHARD_PREFETCH=1` used the retained descriptors and
  `PrefetchPolicy::Always` once before load. The charged candidate wall is its
  complete feature timer plus the first packed-pass runtime wall.
- KEEP: improve against the A-bracket mean by at least 20% and 5 seconds, retain
  complete GPU timing and accepted observers, keep first/warm/profile GPU within
  1.5%, and preserve one output digest.

## Results

| Arm | Prefetch (ms) | First runtime (ms) | Charged wall (ms) | First GPU (ms) | Warm GPU (ms) | Profiled GPU (ms) |
|:---|---:|---:|---:|---:|---:|---:|
| A1 demand | 0.000 | 22,922.262 | 22,922.262 | 1,002.564 | 991.739 | 993.005 |
| B prefetch | 13,241.761 | 1,414.408 | 14,656.168 | 1,001.280 | 993.270 | 994.392 |
| A2 demand | 0.000 | 22,541.024 | 22,541.024 | 1,002.941 | 992.793 | 993.787 |

The A mean is `22,731.643 ms`, with 1.68% bracket spread. The candidate saves
`8,075.475 ms`, or 35.53%, and clears the stricter `17,731.643 ms` gate by
`3,075.475 ms`. The displayed prefetch and first-pass values round separately;
the emitted `14,656.168 ms` charge uses their unrounded values.

Candidate first/warm/profile GPU differs from the A means by
`-0.147%/+0.101%/+0.100%`. Every command has complete GPU timing, every observer
and raw-coverage gate passed, and all three stdout files have SHA-256
`e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.

## Full-File Authority

| Shard | Bytes returned | Wall (ms) |
|:---|---:|---:|
| 1 | 10,946,624 | 0.660 |
| 2 | 49,983,253,824 | 7,332.459 |
| 3 | 39,992,153,376 | 5,908.380 |

All `89,986,353,824` mapped bytes returned across 3/3 shards with zero skips.
Process telemetry reports `89,975,398,400` physical-read bytes; returned bytes
are the exact authority because metadata was touched before the timed phase.
The read intentionally includes the 28.8 GB CPU PLE region and reaches
6.80 effective decimal GB/s.

Post-prefetch admission remained unchanged: aggregate required bytes were
`62,362,648,576`, observed weight bytes `61,175,349,248`, observed session bytes
`638,681,088`, and required session bytes `1,187,299,328`. This proves only the
measured M4 Max N=512 configuration survived the page-cache pressure.

## Disposition

Retain `QWEN4EXP_FULL_SHARD_PREFETCH=1` as an explicit option for ordinary
Flash-Next single-turn requests against the released three-shard asset. Remove
the N=512/profile-only harness and charged-wall plumbing after measurement while
keeping exact read validation and diagnostics. Do not enable it automatically:
already-warm files, larger sessions, smaller-memory systems, and alternate
sharding were not qualified. Do not build a PLE-excluding or range-specific
warmer from this result.

Raw logs remain under
`target/profiles/qwen4exp-full-shard-prefetch-n512-20260829/`; their hashes are
recorded in `results.json`.

Adversarial review: `01a04de7-d66e-73f1-b3de-14ba60527c89`.
