# V4.1 CED Engram Parallel Decode: Neutral Negative

2026-10-28. Source: experiment `2abcd480` on
`perf/v41-engram-prefetch-main` (rebased onto main `c34ddceb`), reverted after
measurement. Artifact: GSQ-RCO IQ2_XS, 40 layers, 4,096-row chunks, M5 Ultra.
Raw: `~/qwen-runs/2026-10-28-v41-engram-decode/`.

## Decision

The first encoder chunk's Engram layers were decoded by a bounded worker set
instead of one serial stream. The 8K pair stays bitwise, but the same-binary
CED A/B is **neutral**: off 1,864.15, on 1,863.48 tok/s; **-0.04% / -1.6 ms**.
The experiment is reverted; the serial stream is retained.

## What Changed

The fresh 8K timeline put **73.5 ms of encoder encode recv stalls** plus
**71.0 ms of Engram prepare** on the first chunk. Each Engram layer owns a
channel; a bounded worker set (default 4, overridden by
`QWEN_V41_ENGRAM_DECODE_THREADS`) decoded layers in parallel, while the
encoder still wrote them in layer order. Reads use positioned `read_at`, so
the concurrency is safe, and each layer's values are deterministic.

## Evidence

8K pair at the production default: Full ~1,000, CED ~1,571; all 14 endpoints
shared the same hash. The machine was in a slower state than the earlier
packets (Full ~1,000 vs 1,120), so only the same-binary A/B is claimed.

CED-only A/B, `O S S O`, two runs per arm, 180 s apart:

| arm | samples (tok/s) | mean | wall |
| --- | --- | ---: | ---: |
| off (1 worker) | 1,869.88 / 1,863.87 | 1,866.9 | 4.3898 |
| on (4 workers) | 1,866.28 / 1,856.91 | 1,861.6 | 4.4009 |
| on (4 workers) | 1,861.21 / 1,869.51 | 1,865.4 | 4.3914 |
| off (1 worker) | 1,863.52 / 1,859.32 | 1,861.4 | 4.3993 |

Off mean 1,864.15; on mean 1,863.48. The predeclared bar was at least 1% or
50 ms; the experiment does not cross it.

## Why It Does Not Pay

The serial `decode_rows_batched` already issues read-ahead for the block, and
the chunk-1 prefetch thread is already running during the first chunk. Four
concurrent layer decoders contend for the same SSD/page cache and CPU
bandwidth; the work is not reduced, only interleaved. The pair staying
bitwise confirms the negative is economic, not numerical.

## Limits

One prompt, one artifact, one session. Only 1 versus 4 workers was compared;
a different staging (for example a dedicated staged buffer with a single I/O
queue) is a different proposal, and the negative does not close it.
