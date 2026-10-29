# V4.1 Engram Row Dedup: Closed Negative

2026-10-29. Discriminator branch `perf/v41-engram-dedup`, based on main
`17972f22`. CPU-only replay of the production plan, token map and
`AddressState`; no GPU, no lease, no runtime override.
Raw: `~/qwen-runs/2026-10-29-v41-engram-dedup/`.

## Decision

Dedup is closed for the 8K benchmark and does not clear the bar on the real
frozen prompt. The synthetic suite prompt is uniformly novel (xorshift over
the model vocabulary): **98,296 unique / 98,304 ids per layer per chunk**
(0.008% duplicates). The real 8,192-token frozen prompt is **81.7% unique**
across the whole prompt (321,404 / 393,216) and **87.5% unique per 2,048-row
chunk**. Even if every 18.3% duplicate were removable read work, the 108 ms
Engram exposure would yield at most ~19 ms / 0.43% wall, below the
predeclared 50 ms / 1% bar; page-miss dominance makes the real saving smaller,
because a repeated row is usually already cached after its first occurrence.

## Evidence

Synthetic suite (`shrinking_ced`, `--pp 8192`, `--runs 1`, seed 1; two chunks
by two layers):

| layer | chunk | total | unique |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 98,304 | 98,296 |
| 14 | 0 | 98,304 | 98,296 |
| 1 | 1 | 98,304 | 98,292 |
| 14 | 1 | 98,304 | 98,287 |

Real frozen prompt, CPU replay with 2,048-row chunks (the CED encoder uses
4,096 rows; the whole-prompt share bounds the larger chunk):

| chunk | layer | total | unique |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 49,152 | 43,007 |
| 0 | 14 | 49,152 | 43,005 |
| 1 | 1 | 49,152 | 41,885 |
| 1 | 14 | 49,152 | 41,886 |
| 2 | 1 | 49,152 | 43,623 |
| 2 | 14 | 49,152 | 43,621 |
| 3 | 1 | 49,152 | 43,851 |
| 3 | 14 | 49,152 | 43,854 |
| whole | both | 393,216 | 321,404 |

The replay test is `real_prompt_engram_duplication` (CPU-only, ignored; needs
`V41_CED_GGUF` and `V41_CED_TOKENS`). It uses the same plan, token map and
`AddressState` path as prefill and does not override the runtime.

## Why

The 24 ids per token per layer are n-gram hashes. The suite prompt is xorshift
over the full vocabulary, so its n-grams are uniformly novel; the earlier
small-periodic-prompt premise was wrong (cx review, 2026-10-29). The real
prompt's repetition is only ~18%, and the read is I/O/page-bound: removing
duplicate rows does not remove first-use page misses. The neutral per-layer
worker experiment points the same way.

## Limits

One real prompt. The replay chunks at 2,048 rows, so the 4,096-row CED share
is bounded by the whole-prompt 18.3%, not measured directly. No GPU A/B was
run because the ceiling is below the declared bar. Real traffic with much
heavier n-gram repetition could still benefit; that is a product question,
not this benchmark.

## Next

The exact cold-prefill host/queue budget is now characterized and its cheap
levers are closed. The remaining work is GPU-side attribution (attention
operand reuse) or the exact-prefix-reuse product bet.
