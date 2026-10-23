# V4.1 attention sink: tensor-ops score/PV, +1.7-2.3%

Candidate `perf/v41-attn-tensor` (`00b6d408`) on the 512-row sink tile.
The score and PV kernels are the Metal 4 tensor-ops forms (`matmul2d` per
16-key block) from the stalled attention-tensor lane, ported onto our
512-row tiling (not its superseded 256-row tiling). M5 Ultra, macOS
27.0.1, production lease, same binary, `qwen-bench suite --pp 2048,4096
--runs 2 --seed 1`, ABCCBA with `QWEN_V41_SINK_TENSOR=0` as the
simdgroup-MMA control. Raw: `metagross:~/qwen-runs/2026-10-23-attn-tensor/`.

## What changed

`score_tensor` and `pv_tensor` run the same half-staged Q/K/V operands
and F32 accumulators as `score_mma`/`pv_mma`, with one `matmul2d` per
16-key block; the shared softmax and normalize passes are unchanged.
Selected only where the Metal 4 tensor tier exists and the sink tile has
at least 128 rows. The tensor PV dispatch uses 128 threads (the MMA PV
keeps 256); the tensor kernels stage in `e += 128` loops.

## Throughput

Means of four samples per artifact/shape:

| artifact | shape | MMA | tensor | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 837.9 | **852.3** | **+1.72%** |
| IQ3_XXS | pp4096 | 796.9 | **814.9** | **+2.26%** |
| Q2_K_XL | pp2048 | 838.2 | **852.6** | **+1.72%** |
| Q2_K_XL | pp4096 | 797.6 | **816.0** | **+2.31%** |

Fixture, full sink, 640 keys: 32 rows 437 -> 518 us (tensor 18% slower),
64 rows tie, 128 rows tie, 256 rows 2,051 -> 1,927 us (-6.1%), 512 rows
3,418 -> 3,303 us (-3.4%). Hence the 128-row gate.

## Quality

Six frozen held-text documents, IQ3_XXS, same clean build, 384 targets.
The tensor kernels change the reduction order of the half-staged products;
the tile width changes nothing.

| document | control NLL | tensor NLL | delta | control top-1 | tensor top-1 |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.604260 | 2.610929 | +0.006669 | 0.594 | 0.594 |
| prose_2 | 4.204816 | 4.199976 | -0.004840 | 0.188 | 0.188 |
| prose_3 | 2.260082 | 2.288914 | +0.028832 | 0.453 | 0.453 |
| code_4 | 1.252495 | 1.255010 | +0.002515 | 0.719 | 0.719 |
| code_5 | 1.570008 | 1.585876 | +0.015868 | 0.688 | 0.688 |
| code_6 | 0.986541 | 0.985045 | -0.001496 | 0.766 | 0.766 |
| mean | 2.146367 | **2.154291** | **+0.007924 (+0.37%)** | 0.568 | 0.568 |

The top-1 count is unchanged (218/384) and the per-document deltas are in
the same band as the tensor expert tile (+0.0052 mean). This is a
reduction-order change, not a bitwise one; `QWEN_V41_SINK_TENSOR=0`
restores the MMA kernels.

## Correctness

- `tensor_sink_tile_matches_mma_sink` compares the tensor tile against
  the scalar sink (half tolerance) and the MMA tile (1e-4 + 2e-3 relative)
  across the fixture cases, including invalid-selection and finite guards.
- 46 V4.1 tests pass (31 attention, 14 prefill, 1 tensor sink), including
  census tests that count either sink form per tile.

## Limits

- M5-class tensor tier only; macOS 26/M4 devices keep the MMA kernels.
  Tiles below 128 rows keep MMA (the tensor form is slower there).
- One artifact pair, one probe corpus; the +0.0079 mean NLL is a
  descriptive document mean, not a task threshold.
- Decode does not run the sink tile.
