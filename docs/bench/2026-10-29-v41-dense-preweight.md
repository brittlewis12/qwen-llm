# V4.1 Dense Preweight Probe: Q5_K Is Bitwise And Faster In The Tile

2026-10-29. Probe branch `test/v41-dense-preweight`, test code `d534832f`,
based on main `cfb27cbe`. M5 Ultra 80-core, 256 GiB, Metal 4 tensor tier,
self-leased production GPU. Raw:
`~/qwen-runs/2026-10-29-v41-encoder-depth/dense-preweight-*.log`.

## Decision

The MLX-Serve `DENSE_ROWS=1024` lesson transfers to our **Q5_K** dense
linears, not our Q8_0 ones. At the V4.1 production shapes and 2,048 rows, a
Q5_K bank dequantized once to F16 and run through the dense F16 tensor tile
is **1.27-1.73x faster** than the current Q5_K tensor tile, and the output is
**bitwise identical** with the same F16 activation. The Q8_0 dense preweight
loses; the existing Q8_0 tensor tile is already the wide 64-slot form.

## Probe

Synthetic GGUF banks at the model's dense shapes, 2,048 query rows, F16
activation, `encode_mat_mat_tensor_f32` (same tile). The dense arm builds an
F16 weight from `dequant_to_f32` + F16 rounding once, outside the timed
region, then runs the dense F16 tensor tile. Median of six launches per arm,
two rounds; self-leased, opt-in `QWEN_V41_TILE_TIMING=1`.

Q5_K shapes:

| shape | dense F16 W | Q5_K tile, F16 act | Q5_K tile, F32 act | dense / tile-F16 |
| --- | ---: | ---: | ---: | ---: |
| q_b 1280->32768 | 2,252 us | 2,952 us | 3,330 us | 1.31x |
| output_a 4096->8192 | 1,550 us | 2,335 us | 2,676 us | 1.51x |
| output_b 8192->5120 | 2,375 us | 3,010 us | 3,502 us | 1.27x |
| shexp gate 5120->2304 | 589 us | 875 us | 1,010 us | 1.49x |
| shexp down 2304->5120 | 553 us | 816 us | 922 us | 1.48x |

Q8_0 controls:

| shape | dense F16 W | Q8_0 tile, F16 act | dense / tile-F16 |
| --- | ---: | ---: | ---: |
| q_a 5120->1280 | 322 us | 290 us | 0.90x |
| kv 5120->512 | 148 us | 137 us | 0.93x |

## Exactness

`tensor_dense_q5_k_preweight_matches_tile` runs a synthetic 5120->2304 Q5_K
bank at 2,048 rows through both paths with the same F16 activation:

```
q5k_preweight exact=true max_abs=0e0 rms=0e0
```

The dense arm uses the CPU `dequant_to_f32` reference rounded to F16. The
new GPU kernel `v41_dequant_q5_k_f16` (tensor module) dequantizes a synthetic
5120x2304 Q5_K bank with the same `qt_dequantize_q5_K` helper and writes the
natural row-major `block*256 + il*16` order. Its half bits match the CPU
reference on **0 mismatches of 11,796,480 values**
(`v41_q5_k_f16_dequant_matches_cpu`).

## Reading

The Q5_K tile dequantizes its weight on the fly and runs the 32-slot dense
tile. The dense preweight path dequantizes once per chunk, reads an F16
weight (2 bytes/weight) and runs the same 32-slot tile. The extra weight
traffic is small beside the saved dequant work: the Q5_K tile is 49-59
TFLOPS while the dense F16 weight reaches 72-88 TFLOPS. The Q8_0 tile, by
contrast, is the 64-slot wide form and already reaches 78-102 TFLOPS; the
F16 dense path is the 32-slot form and loses.

## GPU dequant

`v41_dequant_q5_k_f16` is a scalar natural-order kernel in
`kernels/tensor/moe_grouped_slots_tensor.metal`: one thread per 16-element
sub-tile, the same `qt_dequantize_q5_K` helper as the quantized tile, and a
row-major `block*256 + il*16` write. On a synthetic 5120x2304 Q5_K bank it
matches the CPU `dequant_to_f32` + F16 reference exactly (0 / 11,796,480
half values).

## End-to-end A/B: 8K CED regression

The full routing (q_b, index.query, output_a groups, output_b, shared
gate/up/down) was integrated on `test/v41-q5k-dequant` (`e7051796`) and
measured at 8K CED with `--runs 2`, 180 s between arms, ABBA:

| arm | dense preweight | tok/s |
| --- | ---: | ---: |
| off1 | 0 | 2,050.96 |
| on1 | 1 | 2,001.69 |
| on2 | 1 | 1,999.36 |
| off2 | 0 | 2,039.30 |

The on arm is **2.2% slower**. The kernel-level 1.27-1.73x did not survive
the production stream: the per-chunk dequant and the F16 weight read/write
cost more than the denser GEMM saves. The likely hot spot is the eight
per-group `output_a` linears (4096->1024 each), unlike the single
4096->8192 bank used in the kernel probe; per-group dequant overhead
dominates its small GEMM.

The integration is **not promoted and not on main**. The per-chunk route
needs either the output_a groups removed (one more A/B) or a persistent F16
cache that removes the dequant; if neither wins, close it and keep the
kernel probe as evidence.

## Next

1. ~~Add a GPU Q5_K->F16 dequant kernel~~ done, bitwise above.
2. Add a reusable F16 dense-weight scratch for the largest dense linear
   (output_b/q_b: 41,943,040 elements / 83.9 MB) and price it in the
   prefill/CED ledger.
3. Route Q5_K dense linears through the preweight path at rows >= 2,048
   (production chunks are 2,048/4,096); keep Q8_0 on the current tile. The
   scratch is per chunk and shared across layers; a two-chunk CED pipeline
   prices two copies.
4. Run the 8K Full/CED pair endpoint gate and the 8K/32K CED wall A/B. The
   probe predicts a bitwise pair if the GPU dequant matches.
5. Only then consider lowering the threshold toward 1,024 rows.

## Limits

Synthetic banks, kernel-level, no model quality or end-to-end timing. The
probe's dense weight is built on the CPU; the production dequant is a GPU
kernel and its cost is not in the table. The Q5_K tile and dense F16 tile use
different pipelines, so the bitwise result is empirical, not a proof for
every shape; re-run the equality test at each production shape before the
pair gate.
