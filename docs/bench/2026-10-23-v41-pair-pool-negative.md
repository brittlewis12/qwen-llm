# Ratio-2 compressor pair pool: negative/mixed

Probe branch `perf/v41-attn-tensor` (local; pair-pool commit `e7856f4b`),
M5 Ultra, same-binary ABCCBA with `QWEN_V41_PAIR_POOL=0` as the per-row
control. Batch prefill has both rows of each ratio-2 pair resident, so one
threadgroup per pair mixes the compressor values with the gate softmax,
applies the compressor RMSNorm (1024-thread reduction matching the per-row
norm), and writes the pending even row. It replaces the per-pair pending
copies, pool and latent norm. The full batched-vs-scalar state test and
the standalone kernel test pass.

## Throughput

| artifact | shape | control | pair pool | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 848.9 | 802.5 | **-5.5%** |
| IQ3_XXS | pp4096 | 815.2 | 825.8 | +1.3% |
| Q2_K_XL | pp2048 | 850.9 | 862.8 | +1.4% |
| Q2_K_XL | pp4096 | 813.7 | 825.4 | +1.4% |

IQ3 pp2048 recheck: control 846.8 / 850.0, batch 803.0. The regression is
reproducible at the model/shape that matters most.

## Interpretation

The pair kernel is correct and its dispatch count is lower, but the
1024-thread-per-pair grid (one full core per pair) lands between the
attention prepare and the FFN stream; on IQ3 its cache/occupancy
footprint costs more than the ~9k removed small dispatches save. Q2's
smaller working set and pp4096's second chunk absorb it. Not landed; the
serial lane needs a lower-footprint fusion (e.g., copies only, or a
publish-rows kernel that keeps the existing norm threadgroup), not a
1024-thread pair kernel. Main/origin stay at `402cf628`.
