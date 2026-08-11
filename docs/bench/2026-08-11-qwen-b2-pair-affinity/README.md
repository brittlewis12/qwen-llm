# Qwen B=2 Prefix-Affinity And Depth Pairing

Date: 2026-08-11

Status: product `GO` for seekable Qwen `--concurrency 2` files. DeepSeek V4
uses the same planner implementation but remains default-off pending a safe
model-backed validation under its much larger residency envelope.

## Question

Can offline request ordering recover more value from the existing independent
B=2 queues before building dynamic refill or continuous batching?

Input-order pairing can place two unrelated long prompts together, preventing
prefix fanout even when their natural partners appear later in the file. It can
also pair short and long generations, leaving most transitions on the slower
single-lane tail.

## Mechanism

The planner operates independently within 16-request windows, preserving a
strict bound on completed output held for input-order emission.

1. If a window is odd, retain its final input request as the serial tail.
2. Score every remaining pair by the prefix boundary the existing family
   fanout planner can actually restore.
3. Greedily select disjoint edges by descending reusable tokens, then ascending
   requested generation-depth difference and stable input indices.
4. Sort unmatched requests by requested generation limit and pair adjacent
   depths.
5. Execute planned work by earliest input index and buffer at most 15 completed
   outputs while preserving stdout input order.

This is a bounded deterministic heuristic, not a maximum-weight matching claim
and not dynamic lane refill. Pair-local memory admission, prefix snapshots,
session ownership, failure handling, and decode executors are unchanged.

Qwen defaults the planner on. Set `QWEN_CONCURRENCY_PAIR_PLANNER=0` to restore
input-order pairs. Invalid values fail closed. DeepSeek defaults off and can be
probed explicitly with `QWEN_CONCURRENCY_PAIR_PLANNER=1` after memory admission
is known safe.

## Validation

Base: `0c11cff0b77099734f169a56197f4f69dc9725bc` plus this candidate. One
release binary served every arm on Apple M4 Max. GPU/model processes were
serialized, whole-model DeepSeek residency was not used, and stdout was hashed
without normalization.

### Interleaved long prefixes

Four requests arrive as `A0, B0, A1, B1`. The A requests share 5,400 prompt
tokens and the B requests share 7,200, but input-order pairs share zero. Every
request asks for 32 tokens.

| Organization | Pair indices | Prefix tokens | Pair prepare + decode | Warm process wall |
|---|---|---:|---:|---:|
| Input order | `[0,1]`, `[2,3]` | `0 + 0` | `15.752 s` | `18.48 s` |
| Affinity planner | `[0,2]`, `[1,3]` | `5,120 + 7,168` | `10.182 s` | `12.81 s` |

The candidate is `1.547x` on the summed model-serving interval and `1.443x` on
warm process wall. Its two prefills total `9.176 s` versus `14.816 s` for the
closing control. All arms emit stdout SHA-256
`85c5e619fa9e5835c8a6002a427ea44dbd111f5c538508d1dde8df8ebb603d33`.

### No-prefix generation skew

Four unrelated short prompts request `8,64,10,62` tokens. Input order forms
`[8,64]` and `[10,62]`; the planner forms `[8,10]` and `[64,62]`.

| Organization | Paired transitions | Serial-tail transitions | Decode wall | Process wall |
|---|---:|---:|---:|---:|
| Input order | `16` | `108` | `1.218 s` | `4.32 s` |
| Depth paired | `68` | `4` | `0.982 s` | `4.11 s` |

Decode improves `1.241x`; process wall improves `1.051x` despite model load and
short serial prefills dominating this fixture. Both arms emit stdout SHA-256
`0b9ddb84481804eaeb98ae759c1c6e7fc08e444443d9e552f54f60b68ae4dc4a`.

## Decision

Promote bounded affinity/depth planning for Qwen B=2. It composes two already
qualified mechanisms and removes avoidable work without changing kernels,
session state, or numerical organization. Keep the rollback while request
ordering gains field coverage.

Do not infer authority for dynamic refill, ragged B=8/B=16 positions, or
cross-sequence prefill. Keep DeepSeek default-off until a memory-safe K160-class
control/candidate run confirms that queue assignment and sampled output remain
equivalent.

Representative commands:

```text
QWEN_CONCURRENCY_PAIR_PLANNER=0 qwen --model MODEL \
  --requests-jsonl FIXTURE --concurrency 2 --temp 0 --prefill-chunk 1024

qwen --model MODEL --requests-jsonl FIXTURE \
  --concurrency 2 --temp 0 --prefill-chunk 1024
```
