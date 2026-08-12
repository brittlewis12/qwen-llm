# DeepSeek Pair-Affinity Planner

Date: 2026-08-12

Status: product GO for bounded DeepSeek V4 B2 scheduling.

## Question

Should the existing 16-request pair planner become the default for DeepSeek V4
concurrency, now that pair-local prefix fanout can reuse a causal snapshot?

The four-request fixture is deliberately interleaved as `A0,B0,A1,B1`. Requests
within each family share 6,268 tokens; cross-family requests share only three.
Input-order scheduling therefore forms two useless pairs, while bounded affinity
scheduling can form `A0,A1` and `B0,B1`. Each request generates 32 greedy tokens.

## Command

Run each arm serially, changing only the planner value between `0` and `1`:

```bash
/usr/bin/time -l env \
  QWEN_DSV4_RESIDENCY_SET=0 \
  QWEN_DSV4_PREFILL_CHUNK_TOKENS=4096 \
  QWEN_CONCURRENCY_PAIR_PLANNER=0 \
  QWEN_CONCURRENCY_PREFIX_FANOUT=1 \
  target/release/qwen \
  --model /Users/tito/models/deepseek-v4-flash-0731-reap-k160/DeepSeek-V4-Flash-0731-REAP-K160-Q3_K_Q4_K-00001-of-00004.gguf \
  --requests-jsonl requests.jsonl \
  --concurrency 2
```

## Result

Both arms use K160, a 4,096-token packed-prefill cap, pair-local fanout, and
`QWEN_DSV4_RESIDENCY_SET=0`.

| Organization | Wall | Evaluated prompt tokens | Pair wall sum |
|---|---:|---:|---:|
| Input-order pairs | `114.71 s` | `25,080` | `112.782 s` |
| Prefix-affinity pairs | `71.57 s` | `12,792` | `70.079 s` |

The planner improves process wall by `1.603x`, pair wall by `1.609x`, and avoids
12,288 prompt-token evaluations. It selects two 6,144-token checkpoints. Their
snapshot captures take `382.138/394.027 ms`, restores take
`192.835/196.202 ms`, and the two decode phases remain effectively flat at
`2.532/2.482 s` versus `2.540/2.502 s` in the control.

Complete input-ordered JSONL is byte-identical between arms, with SHA-256
`eb72076dd34ad4f2fa860bc68b72c6aab9c7fb7fec578c623215633ce9cf6acb`.
Both runs complete without process swaps or memory fallback. The candidate's
fanout admission prices about 9.38 GB, dominated by the second live session;
each causal snapshot payload is about 60.1 MB.

## Decision

Enable the bounded planner by default for DeepSeek V4 B2. Keep
`QWEN_CONCURRENCY_PAIR_PLANNER=0` as strict input-order rollback, preserve the
16-request window and input-ordered publication, and leave pair-local memory
admission authoritative. This changes physical request organization, not model
math or the result order.

The measurement binary identifies commit `47eb2a1` with a dirty worktree because
two unrelated user-owned documents were untracked. The candidate path itself is
the reviewed source plus explicit `QWEN_CONCURRENCY_PAIR_PLANNER=1`; promotion
changes only that policy default.

Machine-readable measurements and the complete fixture are in this directory.
The fixture SHA-256 is
`6cdd8b110834d1ca3398fcc88a42cf0a3c8261c29c764e68572f67654c5a3fb2`.
