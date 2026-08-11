# Dense B=8 cohort planner and underfill fallback

## Question

Can `--batch-size 8` accept an ordinary heterogeneous JSONL file rather than
requiring callers to hand-order a request count divisible by eight?

## Planner

The file-mode planner validates every request independently, then stable-buckets
it by the execution constraints the fixed backend actually needs:

- tokenized prompt length; and
- requested generation limit.

Each bucket contributes complete B=8 cohorts in input-index order. Remaining
requests use the established serial request path rather than a smaller static
batch; the measured dense crossover remains near B=6-8, so padding or a generic
B<8 kernel would spend work without evidence of a gain.

Work units are ordered by their earliest input index. Completed rows enter an
indexed reorder buffer and stdout advances only through the contiguous ready
prefix. Execution may group compatible requests, but observable JSONL order
remains exact input order. The buffer flushes after every work unit rather than
withholding the complete file.

This is a seekable-file planner, not an arrival queue. It does not add stdin
batching, deadlines, fairness, a daemon, lane replacement, or ragged frontiers.

## Validation

An 18-request Qwen3.5 0.8B Q4_K_M fixture deliberately interleaved:

- eight 1,029-token requests with a four-token generation limit;
- eight 1,112-token requests with a six-token generation limit;
- one additional request compatible with the first group; and
- one short incompatible request.

The planner reported three compatibility buckets, formed two full cohorts, and
routed two requests through serial fallback. Both cohorts retained automatic
shared-prefix fanout. The candidate emitted all 18 rows in original interleaved
order.

Candidate and cache-disabled serial outputs were byte-identical with SHA-256
`935285b8763ce02fccaa983465ed4281b2c65053c3128857ef2f5d3dec724467`.
This checks both reordered batch completion and the serial underfill seam; an
indexing or output-order error would change the complete file.

The two fixed cohorts sustained 617.88 and 598.61 aggregate generated tok/s.
This packet is primarily a semantic/product gate, not a fresh throughput claim:
the underlying executor and fanout mechanisms retain their own performance
evidence.

CPU-only tests cover mixed compatibility buckets, two complete cohorts,
underfill-only files, request accounting, and the reorder buffer's contiguous
flush rule. The final binary passed clippy with warnings denied.

## Decision

- Promote compatibility-based cohort formation and serial underfill for
  arbitrary non-empty dense-Qwen file JSONL inputs.
- Keep prompt length and generation limit in the compatibility key. Per-lane
  generation limits are mechanically possible but remain unqualified here.
- Preserve exact input-order output and expose planner counts on stderr.
- Keep independent command queues out of this file path; serialized underfill is
  simpler and avoids the measured singleton contention tax.
- Treat a bounded resident arrival queue as a separate product step. The file
  planner supplies its compatibility vocabulary but not its fairness policy.

## Commands

```bash
target/release/qwen \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-planner-mixed.jsonl \
  --batch-size 8

target/release/qwen \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-planner-mixed.jsonl \
  --prefix-cache-max-mib 0 \
  --cache-prefix-auto-min-tokens 0
```
