# V4.1 Attention Half-KV Reuse: Probe Negative

2026-10-29. Probe source `d967d7ee` on `perf/v41-attn-half-kv` (design
`49832549`), based on main `dde217a1`. Resident synthetic tensor64 tile:
H=64, M=512, window 128, topk 512, K=256/512/640, n=1/8/32, ABBA with three
repeats and four launches per command, GPU command timestamps. Raw:
the probe output captured with the build.

## Decision

The half-KV mirrors are **bitwise-exact** on every probe case, but the tile
kernel only gains **1.0-5.4%** (best 5.4% at K=256, n=32; 2.7-3.4% at the
production-like K=512/640, n=32). The predeclared bar was a 10% tile-kernel
reduction. The projected complete-attention gain is below 0.6% wall, so the
experiment is **closed and not integrated**; production still selects the
F32 tile.

## Evidence

n=32 (best amortization of per-command overhead):

| K | F32 us | half us | speedup |
| ---: | ---: | ---: | ---: |
| 256 | 370.667 | 351.583 | 1.054 |
| 512 | 710.141 | 688.214 | 1.032 |
| 512 ragged | 706.203 | 683.292 | 1.034 |
| 640 | 888.719 | 865.234 | 1.027 |
| 640 ragged | 879.922 | 854.552 | 1.030 |

n=1 cases gain 1.1-2.5%. In every case the F32-input tile and the half-mirror
tile produce identical output bits and identical used score/probability
lanes.

## Why

The K/V rows are shared across queries and mostly cache-resident; the
in-kernel `half4(f32 row)` conversion is not the dominant cost. Halving the
K/V bytes saves only the fraction of tile time spent on those reads, while
the tensor matmuls, softmax, barriers and command overhead dominate. The
earlier worker-concurrency negative and the gathered-slab negative pointed
the same way; this probe rules out the bounded half-mirror variant for this
geometry.

## Limits

Synthetic fixture; window 128 rather than the production window; the probe
times the tile kernels only (score, softmax, PV, normalize), not the full
attention core or the mirror construction and memory cost. A different slab
granularity or a fused score+PV kernel could behave differently, but this
bar is not met.

## Update: production-shaped replay

The original probe used at most 32 queries per dispatch. At production
512-query tiles (K=640, window 128, topk 512): K/V half is 1.022x, Q half is
1.113-1.115x, and Q+K/V is 1.166-1.170x, all bitwise-identical on finite
lanes. The K/V-only negative stands for that operand pair, but the Q mirror
is a real win and was integrated (see
`2026-10-29-v41-attention-qhalf.md`). The broader conclusion that the
attention operand/dataflow family is closed is withdrawn; query staging and
SIMD ownership remain open.

## Next

The exact cold-prefill lane is now exhausted: host/queue, Engram dedup,
decoder preparation, per-layer concurrency, and half-KV tile reuse are all
characterized and closed. The remaining work is exact prefix reuse (product)
or a new GPU scheduling premise with fresh attribution.
