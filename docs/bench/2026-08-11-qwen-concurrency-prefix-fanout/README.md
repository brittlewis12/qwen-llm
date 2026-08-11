# Qwen Concurrent Prefix Fanout

Date: 2026-08-11

Status: product `GO` for dense and MoE Qwen with fixed prefill chunks.

## Scope

`--concurrency 2` previously prefetched both prompts completely before
overlapping decode. The candidate computes the exact pairwise token LCP,
prefills one stable boundary, captures a process-local `PreparedCheckpoint`,
restores the second fresh sequence, and evaluates only private suffixes.

The policy defaults on at 256 shared tokens. Partial prefixes align down to the
fixed prefill chunk so restored execution retains the cold chunk schedule.
Identical prompts may use their complete unaligned boundary. Automatic chunk
selection and failed snapshot-memory admission retain two serial prefills.

Rollback:

```bash
QWEN_CONCURRENCY_PREFIX_FANOUT=0
```

## Fixture

- Model: `/Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Device: Apple M4 Max
- Decode: greedy, eight requested tokens per lane
- Prefill chunk: 1,024
- Residency: cache-warm GGUF
- Control/candidate are separate processes using one identical release binary.

The identical pair uses the 6,469-token tokenization of
`the_current_ring0_v0.2.md`. The partial pair appends distinct red/blue notebook
suffixes; its exact LCP is 6,475 tokens.

## Results

| Pair | Arm | Selected prefix | Model prefill, ms | Preparation, ms |
|---|---|---:|---:|---:|
| Identical | rollback | 0 | 8,246.567 | 8,251.823 |
| Identical | candidate | 6,469 | 4,146.937 | 4,199.627 |
| Partial | rollback | 0 | 8,240.966 | 8,244.651 |
| Partial | candidate | 6,144 | 5,241.881 | 5,293.907 |

The identical snapshot is 198,374,756 bytes and costs 39.064/10.049 ms to
capture/restore. The partial snapshot is 191,717,456 bytes and costs
38.147/10.398 ms. Candidate preparation improves 1.965x and 1.557x.

A dense Qwen3.5 0.8B guard over the identical fixture moves preparation
`1,657.922 -> 870.028 ms`, or 1.905x. Its 99,718,468-byte snapshot costs
27.153/4.789 ms to capture/restore, and both output rows remain byte-identical.

Decode is unchanged independent-queue execution. Identical-pair decode is
112.680/115.780 ms for control/candidate; partial-pair decode is
113.222/110.555 ms. These short windows carry no decode-speed claim.

## Correctness

Each request's complete output object is byte-identical between rollback and
candidate. This includes generated-token SHA-256, decoded text, stop reason,
token counts, and the unconsumed-terminal-token marker. The partial pair also
retains distinct red/blue continuations correctly.

The runtime validates the checkpoint owner, exact request-prefix match,
destination capacity, fresh position, and state geometry before restore.
Snapshot size must equal its pre-capture estimate. The snapshot is never
published to the RAM or durable prefix stores and is dropped before decode.

## Decision

Promote for qualifying fixed-chunk Qwen concurrency pairs. The mechanism reuses
causal work rather than trying to overlap two bandwidth-intensive prefills, and
it directly addresses
the long-prompt regime where decode-only concurrency was otherwise a weak
whole-request win. DeepSeek now implements the same policy through its distinct
transient identity and restored-logit contract, recorded in
`docs/bench/2026-08-11-dsv4-concurrency-prefix-fanout/README.md`.
