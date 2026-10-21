> Historical measurement on the pre-merge base. Candidate `2c51322f` is on
> `backup/v41-chunk-1024-966bcae8`, not in the landed branch. The landed base
> and the merged-main numbers are in
> [2026-10-21-v41-prefill-merged-main.md](2026-10-21-v41-prefill-merged-main.md).

# V4.1 prefill: grouped IQ2_XXS/IQ2_XS experts

Candidate `2c51322f` adds F32-operand grouped tiles for IQ2_XXS/IQ2_XS gate/up
and down on top of the 1024-row chunk (`69318883`). At 1024 rows the 384-expert,
top-6 geometry averages 16 slots per expert; the 16-slot tile is full, and every
routed layer in both artifacts now uses it. Before this change Q2 grouped 25/40
layers and IQ3 35/40; the rest looped native rows, rereading each expert bank
per slot.

## M5 ABBA

- M5 Ultra, Metal compiler `32023.921`, SDK 27.0, validation off, production
  lease, one fresh model load per arm, no competing benchmark.
- Chunk-only `69318883` (`~/code/qwen-llm-ours`), grouped candidate `2c51322f`
  (`~/code/qwen-llm-fat-m5`), both `build-info` status `match`, both reporting
  `prefill_chunk=1024`.
- Q2_K_XL and IQ3_XXS, `qwen-bench suite --pp 1024 --runs 3 --seed 1`, arm
  order chunk/candidate/candidate/chunk, unique per-arm files.

| Quant | chunk 1 | cand 2 | cand 3 | chunk 4 | chunk mean | cand mean | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 204.068 | 303.945 | 304.963 | 204.286 | 204.177 | 304.454 | **+49.11%** |
| IQ3 | 257.487 | 303.151 | 302.924 | 257.957 | 257.722 | 303.037 | **+17.58%** |

Within-arm SD is 0.16-0.77 tok/s. Both candidate samples sit above both
chunk-only samples on each artifact.

The same pair at PP2048 (`--pp 2048 --runs 2`, chunk-only first and last):

| Quant | chunk 1 | cand 2 | cand 3 | chunk 4 | chunk mean | cand mean | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 192.817 | 280.441 | 280.365 | 192.737 | 192.777 | 280.403 | **+45.45%** |
| IQ3 | 239.709 | 279.064 | 278.781 | 240.140 | 239.924 | 278.922 | **+16.25%** |

Combined with the chunk change, against main `4b1fca04` (chunk 256, native IQ2
rows):

| Quant | main PP1024 | chunk+IQ2 | delta |
| --- | ---: | ---: | ---: |
| Q2 | 179.539 | 304.454 | **+69.6% (1.70x)** |
| IQ3 | 204.797 | 303.037 | **+48.0% (1.48x)** |

## Numerical status

The grouped F32x tile decodes weights to F32 and keeps F32 accumulation, but
its reduction order is not the native row order. New CPU-codec/F64 witnesses
cover both IQ2 decoders at K=256/512 and real 5120x2304 widths; all 24 F32x
tests and 16 V4.1 FFN tests pass. Full-model NLL/task screening and the ignored
257-token ring-dispatch witness are not yet requalified. Do not treat this as
bitwise.

## Next

Attention is the next dominant stage (37.9% Q2 / 42.9% IQ3 in the accepted
chunk-256 profile, and a larger share after this change), then grouped-launch
grid compaction and a 2048-row/n32-tile test for long prompts.
