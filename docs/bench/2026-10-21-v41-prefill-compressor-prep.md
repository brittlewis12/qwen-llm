# V4.1 prefill: fused compressor norm/RoPE/cache publish

Candidate `a7de76ab` on `perf/v41-compressor-prep`, based on landed main
`5b358b86` (code `c4125c6b`). M5 Ultra 80-core, Unsloth
DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 1024,2048,4096 --runs 2 --seed 1`, A/B/B/A.

The measurement binary carried a `QWEN_V41_COMPRESS_SPLIT=1` override that
kept the old five-dispatch path; the landed candidate always takes the fused
path for supported geometries. Raw:
`metagross:~/qwen-runs/2026-10-21-merge/compress-ab/`.

## Throughput (same binary, fused versus split control)

| shape | split | fused | change |
|---|---:|---:|---:|
| Q2 pp1024 | 341.9 (341.7-342.0) | 343.5 (343.1-343.8) | **+0.47%** |
| Q2 pp2048 | 338.8 (338.3-339.2) | 341.5 (341.3-341.6) | **+0.80%** |
| Q2 pp4096 | 318.2 (318.0-318.4) | 320.4 (320.1-320.7) | **+0.69%** |
| IQ3 pp1024 | 343.1 (342.5-343.6) | 344.8 (343.8-345.7) | **+0.50%** |
| IQ3 pp2048 | 337.4 (337.3-337.5) | 339.1 (339.0-339.1) | **+0.49%** |
| IQ3 pp4096 | 316.8 (316.5-317.0) | 318.6 (318.2-319.0) | **+0.58%** |

tok/s, arm means with ranges. Both fused arms sit above both split arms at
every shape and quant.

## Dispatch census

`QWEN_V41_CENSUS=1`, IQ3 PP1024, one run:

- split: **50,710** dispatches
- fused: **40,470** dispatches
- delta: **-10,240 (-20.2%)**

The removed dispatches are exactly the five per-group elementwise kernels the
fusion replaces (key RMS norm, key RoPE, latent RoPE, key cache copy, latent
cache copy). `selection_prepare` (6,141) is now the largest single non-tiled
family.

## What changed

The fused kernel runs one 256-thread threadgroup per owner group end. Threads
0-127 reduce and normalize the index-width key, then rotate its final
`2*pairs` values and write the cache key row. Threads 128-255 copy the latent
row, rotate its final `2*pairs` values and write the compressed cache row. The
latent RMS norm and key projection remain separate because the projection
consumes the normalized latent and produces the key that is normalized here.
Geometries with `index_dim > 128` or `pairs > 128` keep the old five-dispatch
path.

## Numerical status

Same six-document held-text protocol as the merged-base packet (512-token
prefix, 64 teacher-forced IQ3 targets per document). All six NLL values and
all top-1 values are identical to the landed candidate at the packet's
precision:

| document | landed NLL | fused NLL | top-1 |
|---|---:|---:|---:|
| prose_1 | 2.620457 | 2.620457 | 0.578 |
| prose_2 | 4.199603 | 4.199603 | 0.203 |
| prose_3 | 2.215482 | 2.215482 | 0.469 |
| code_4 | 1.261914 | 1.261914 | 0.719 |
| code_5 | 1.589632 | 1.589632 | 0.703 |
| code_6 | 1.001854 | 1.001854 | 0.766 |

The fused key norm uses 128 threads where the old norm used the pipeline's
1024-thread default, but the witness does not distinguish them. This is a
held-text check, not a proof of bitwise identity beyond these targets.

## Limits

- The gain is small (+0.5-0.8%) and measured on one artifact pair and one
  machine. The dispatch reduction is the structural part and should transfer
  wherever the five kernels were separate.
- The fallback path is unchanged from the landed code but is now only taken
  for larger index/rope geometries; the production V4.1 geometry takes the
  fused path.
- Selection remains per-row (6,141 `selection_prepare` dispatches at PP1024);
  batching that loop is the next owner/producer bet.
