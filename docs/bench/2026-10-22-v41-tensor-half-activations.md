# V4.1 Prefill: Half Activation Copies For The Tensor Tiles

Candidate `perf/v41-tensor-f16act` (measured at `41a9bc20`; the landed
series is rebased and kernel-identical) against the dense-projections
landing `545318d3`. M5 Ultra 80-core, macOS 27.0.1, Unsloth
DeepSeek-V4.1-Flash-1017, default wired limit, production lease, clean
binaries (`build_identity.status = match` on every arm), A/B/B/A, 2 runs
per arm, seed 1. Raw: `metagross:~/qwen-runs/2026-10-22-tensor-tile/`
(`abba-f16act/`, `f16act.log`, `quality-f16act/`).

## 2026-10-23 takeover on the 64-slot dense tile

The series was rebased onto `e75cce0a` and integrated with the 64-slot
dense tile: the wide Q8_0 form has an `_f16act` twin
(`kernel_mat_mat_q8_0_f16act_tensor_wide`), so the dense re-read removal
and the half activation copies stack. Same build-vs-build ABCCBA
(main `e75cce0a` vs this branch, both source-matched, build identity
`match`), 2 runs per arm, seed 1:

| artifact | shape | main e75cce0a | f16act | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 790.2 | **836.9** | **+5.91%** |
| IQ3_XXS | pp4096 | 757.7 | **799.3** | **+5.49%** |
| Q2_K_XL | pp2048 | 788.3 | **838.9** | **+6.42%** |
| Q2_K_XL | pp4096 | 754.5 | **805.4** | **+6.75%** |

The six-document IQ3 witness is bitwise identical to the baseline
(concatenated SHA-256
`b90633d032d9eb12616ff970ea47b16a0730c4a267bb02b2c24b36ef60d853cd`,
mean NLL 2.146367194, top-1 0.5677). 173 tests pass across the tensor,
V4.1 attention/prefill and GLM tensor-prefill filters. Decode rechecked:
tg32 Q2 43.2 / IQ3 41.0. Raw:
`metagross:~/qwen-runs/2026-10-23-f16act-takeover/`.

## What changed

Both tensor tile forms read F32 activations and cast each 32 x 32 block
to half while staging it, once per 64-column weight tile: 4.5x the
weight bytes re-read and converted (`q_b` re-reads its 2048 x 1280
activation block 512 times). The `_f16act` kernels (`HALF_ACT` template
flag, one instantiation per role and dtype) read half activations
instead; the host converts each distinct input once
(`kernel_scatter_offset_f32_to_f16` into a per-row-set F16 scratch,
`prefill::HalfScratch`) and every tensor-routed consumer of that input
reads the copy:

- attention prepare: the layer input (q_a, kv, compressor kv/gate, index
  weights) and, after the query norm, the rank (q_b, index query); the
  projections were reordered so each input's consumers are adjacent,
  which the dependency graph allows;
- attention finish: the grouped output_a input (one copy, per-group
  views) and the output_b input;
- FFN: the normalized input (shared gate/up and the routed gate/up tile),
  the shared down input and the routed down slots.

The tile rounds its operand to half either way, so the output is bitwise
identical to the F32-input form (`tensor_dense_*` asserts equality); the
pass only removes the re-read and cast. Scratch: at a 2048-row chunk
~134 MB for attention (the 32768-wide grouped input), 21 + 57 MB for
FFN, allocated only where the tier is selected.

## Per dispatch

`tensor_dense_production_shape_timing` / `tensor_tile_production_shape_timing`
(heated, interleaved, two agreeing rounds; `f16act.log`):

| dispatch | F32 activations | half activations | gain |
|---|---:|---:|---:|
| q_b 1280->32768 (Q8_0) | 2.52 ms (68.2 TFLOPS) | 2.17 ms (79.0) | 1.16x |
| output_a 4096->8192 | 2.12 (64.9) | 1.77 (77.5) | 1.19x |
| output_b 8192->5120 | 2.97 (57.8) | 2.24 (76.7) | 1.33x |
| shexp gate 5120->2304 | 0.83 (58.0) | 0.65 (74.8) | 1.29x |
| shexp down 2304->5120 | 0.72 (67.0) | 0.61 (79.0) | 1.18x |
| q_a 5120->1280 | 0.49 (54.3) | 0.35 (76.7) | 1.41x |
| kv 5120->512 | 0.18 (58.2) | 0.16 (68.2) | 1.17x |
| IQ3_XXS gate+up (routed) | 13.06 (44.4) | 10.72 (54.1) | 1.22x |
| IQ3_XXS down (routed) | 6.74 (43.0) | 5.85 (49.5) | 1.15x |
| Q3_K gate+up / down | 14.88 / 7.89 | 13.16 / 7.02 | 1.13x / 1.12x |
| IQ2_XS gate+up / down | 12.55 / 6.44 | 10.15 / 5.53 | 1.24x / 1.17x |
| MXFP4 gate+up / down | 13.04 / 6.70 | 11.32 / 6.14 | 1.15x / 1.09x |

The dense Q8_0 tiles reach 75-79 TFLOPS; the routed tiles 47-57, their
K/I-quant decoders now the clear remainder.

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32`, depth 0:

| artifact | shape | dense landing `545318d3` | candidate | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 540.0 (539.9-540.2) | 567.7 (567.6-567.8) | +5.1% |
| IQ3_XXS | pp4096 | 491.3 (490.9-491.6) | 513.1 (512.6-513.7) | +4.4% |
| IQ3_XXS | tg32 | 40.7 (40.3-41.1) | 40.4 (40.2-40.7) | -0.6% |
| Q2_K_XL | pp2048 | 544.9 (544.4-545.4) | 571.9 (571.2-572.6) | +5.0% |
| Q2_K_XL | pp4096 | 494.8 (494.1-495.6) | 517.1 (516.2-517.9) | +4.5% |
| Q2_K_XL | tg32 | 42.5 (42.4-42.6) | 43.4 (43.3-43.5) | +2.2% |

From this morning's main `14377c46`: pp2048 383 -> 572 tok/s (+49%),
pp4096 358 -> 517 (+44%). Cumulative from `4b1fca04`: pp2048 192 -> 572
(3.0x).

Decode is single-row and untouched.

## Quality

The same six-document witness as the previous landings (IQ3_XXS,
`MTL_DEBUG_LAYER=1`, clean builds): every per-document NLL, target rank
and top-1 value is identical to the dense landing's (mean NLL 2.1464,
218/384 top-1), and the six teacher-forced logit files are byte-identical
between the arms (`cmp`), as the construction promises. The 22-task
screen has the identical per-case pattern (code 8/10, reasoning 1/8 with
one truncation, JSON 0/4). Raw: `quality-f16act/`.

## Limits

- M5-class only (the scratch is not allocated elsewhere).
- Serial encoders: each conversion is consumed before the next one
  overwrites the scratch; the FFN keeps two scratches so the routed
  gate/up can read the normalized input after the shared down converted
  its own.
- The routed `_f16act` tiles are selected by the activation tensor's dtype;
  a caller passing F16 activations to the generic (simdgroup) tile is
  rejected by its F32 contract, as before.
