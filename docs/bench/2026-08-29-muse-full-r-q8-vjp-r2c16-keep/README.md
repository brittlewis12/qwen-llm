# Muse Full-R Q8 Transpose VJP KEEP

Decision: **KEEP** the native-Q8 R2C16K64 transpose as the bank-specific
activation-VJP path. Keep the scalar dispatcher unchanged as the exact oracle.

## Mechanism

The kernel computes `[Q,n_out] * W[n_out,n_in] -> [Q,n_in]` without
materializing transposed or dequantized weights. Four SIMDgroups retain 128-way
query parallelism while sharing a dequantized `[16,64]` F32 weight tile through
4 KiB of threadgroup memory. Aligned Q128 prefixes use the matrix path; query
tails and output widths not divisible by 64 use the incumbent.

## Gates

The preregistered packet required:

- finite CPU-adjoint and incumbent differentials with offset and tail coverage;
- positive savings in both B-C-C-B comparisons for every Muse FFN shape;
- at least 4x on every shape and 5.5x geometric mean;
- immediate KILL if gate/up Q128 exceeded 23.579 ms.

Small release cases `(32,64,128)`, `(96,128,257)`, and `(96,65,129)` passed.
CPU relative L2 was at most `2.07e-7`; candidate-versus-incumbent relative L2
was at most `1.13e-7`. Q=257 exercised two matrix tiles plus one incumbent tail;
the 65-output case preserved whole-call scalar fallback bits.

## Production Shapes

All rows use synthetic resident Q8 weights and standalone fresh commands after
one warmup per implementation. No model asset was opened.

| Shape | B1 (ms) | C1 (ms) | C2 (ms) | B2 (ms) | Mean B -> C (ms) | Speedup |
|:--|--:|--:|--:|--:|--:|--:|
| gate/up Q128 | 94.381750 | 3.714750 | 3.710625 | 93.954875 | 94.168313 -> 3.712688 | 25.364x |
| gate/up Q512 | 351.073250 | 13.378125 | 13.462875 | 351.940125 | 351.506687 -> 13.420500 | 26.192x |
| down Q128 | 88.199125 | 3.395250 | 3.388417 | 88.243792 | 88.221458 -> 3.391833 | 26.010x |
| down Q512 | 345.511833 | 12.946375 | 12.945208 | 345.520875 | 345.516354 -> 12.945792 | 26.689x |

Geometric-mean speedup is `26.059x`. Production relative L2 is at most
`4.27e-7`, normalized max error is at most `7.52e-7`, and cosine is at least
`0.999999999998` apart from benign F64 reporting roundoff above one.

The model-free correctness and production packets complete in `0.04 s` and
`2.92 s` after the release build.

## Disposition

Expose the mechanism through the bank-specific frozen-linear VJP dispatcher.
Do not change the scalar dispatcher. The next product-bearing seam is a
shared-primal Muse one-block VJP bank with Metal causal-GQA backward and fixed
scratch; no full-R fit should run before that seam has a bounded projection.

Adversarial design and integration review:
`01a05072-6951-7782-978a-2f274d90f474`.
