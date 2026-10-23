# V4.1 expert decoder attribution: not the lever

Production expert shape (384 experts, 2048 tokens, top-6, 32 slots per
expert, 5120 hidden, 2304 FFN), `tensor_tile_production_shape_timing`,
`QWEN_V41_TILE_TIMING=1`, f16act activations. M5 Ultra. Raw:
`metagross:~/qwen-runs/2026-10-23-f16act-takeover/decoder-attrib.log`.

The tensor tile is staged as half operands either way; the question was
whether a cheaper IQ3_XXS/Q3_K decoder would recover the gap to the
130-TFLOPS resident-operand ceiling. The Q8_0 control has the simplest
decoder and was added to the same harness to bound the decoder share.

| dtype | gate+up TFLOPS | gate+up weight GB/s | down TFLOPS | down weight GB/s |
|---|---:|---:|---:|---:|
| Q3_K | 44.1 | 296 | 41.1 | 276 |
| IQ3_XXS | 54.1 | 324 | 49.5 | 296 |
| IQ2_XS | 56.9 | 257 | 52.4 | 237 |
| MXFP4 | 51.2 | 425 | 47.3 | 392 |
| Q8_0 | 39.1 | 649 | 39.7 | 659 |

Q8_0 is the **slowest** tile even though its decoder is the simplest; it
streams 2-2.5x the weight bytes per output. The quantized tiles reach
higher TFLOPS at lower byte rates, so the tile is bound by weight-byte
staging/bandwidth, not by grid/sign/scale decoder ALU. A vectorized
IQ3_XXS/Q3_K decoder therefore does not have the headroom that the
previous "47-57 vs 75-79 TFLOPS" comparison implied (that comparison
used the 64-slot dense Q8_0 tile, not the same-shape expert tile).

Stop criterion from the checkpoint review: an optimistic decoder-free
control must save at least 10% of combined gate/up+down to justify the
rewrite. The same-shape Q8_0 control is 22-28% *slower*, so the decoder
route is closed. The remaining routed-expert lever is fewer staged weight
bytes (native MXFP4 or a lower-bit artifact) or staging traffic, not a
faster decoder.

## Limits

- One M5 Ultra, one synthetic routing distribution (~32 slots per
  expert), f16act operands. The conclusion is about the decoder's share,
  not a claim that native FP4 would pay; that needs its own probe.
- Weight GB/s is the effective staged weight rate from the harness, not a
  DRAM peak; it is used only to compare the dtypes at one shape.
