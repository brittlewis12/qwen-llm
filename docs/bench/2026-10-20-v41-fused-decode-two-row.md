# V4.1 fused decode: two rows per SIMD group (rejected)

Candidate `21ed03fa` added a two-output-row variant of the fused gate/up/weight
and down/sum kernels. Each lane still owns 16 contiguous values per row, and
each row keeps the native lane partition, per-lane FMA order and SIMD
reduction; the pair shares activation loads. It is bitwise against the native
passes for every non-MXFP4 storage type at K=256 and K=5120. Production decode
selected it for one M5 ABBA.

## M5 ABBA

- M5 Ultra, Metal compiler `32023.921`, SDK 27.0, validation off, production
  lease, one fresh model load per arm, no competing benchmark.
- R1 baseline `55b0157e` (`~/code/qwen-llm-ours`), R2 candidate `21ed03fa`
  (`~/code/qwen-llm-fat-m5`), both `build-info` status `match` after bypassing
  the build-cache wrapper with the rustup cargo.
- Q2_K_XL and IQ3_XXS, `suite --tg 32 -d 2048 --runs 3 --seed 1`, arm order
  R1/R2/R2/R1, unique per-arm files.

| Quant | R1 1 | R2 2 | R2 3 | R1 4 | R1 mean | R2 mean | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 21.749 | 21.432 | 21.406 | 21.698 | 21.724 | 21.419 | **-1.40%** |
| IQ3 | 21.142 | 21.015 | 20.991 | 20.966 | 21.054 | 21.003 | **-0.24%** |

Within-arm SD is 0.02-0.06 tok/s. Both Q2 R2 samples are below both Q2 R1
samples; the adjacent pairs agree in sign (+1.46% and +1.37% for R1). IQ3 is
mixed but not positive. An earlier attempt interleaved with another lane's
benchmark and was discarded; this table is from the clean rerun.

## Interpretation

The two-row kernel doubles per-thread register pressure (four decoded tiles and
four accumulators) without reducing the dequant work, which dominates. Sharing
activation reads does not pay for the occupancy loss. Expert decode is not
activation-load-bound; more rows per SIMD group is rejected. The one-row fused
chain stays. Further serial gains need less dequant arithmetic per weight or a
form that batches several tokens per weight pass, not another GEMV geometry.
