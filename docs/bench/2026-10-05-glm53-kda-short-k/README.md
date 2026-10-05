# GLM-5.3-Flash short-K Q8_0 mat-vec for KDA (2026-10-05)

Leverage map 2026-10-05 #2. UD-IQ3_XXS, M4 Max 128 GB, AC power. Parent
`344da7e7` vs `dc2d5ada` (short-K from `b7655f61`). Same session, back to
back, untraced, `QWEN_METAL_LEASE_WAIT=1`.

```sh
qwen-bench suite -m <shard 1> --pp 32,128,512 --tg 128 -d 0,4096 --runs 3
qwen-bench suite -m <shard 1> --pp 4096 -d 0 --runs 2
```

## Screen (`metal::kda::tests::kda_block_dispatch_costs`)

Synthetic GLM shapes. 34 chained KDA blocks per command, cycling 8 weight
and state sets so weights stream from DRAM, warm GPU. The whole block
measures 352.5 µs, against 352 µs per block in decode attribution v3, so
the screen reproduces the model.

| Group (µs per block) | µs | GB/s |
|---|---:|---:|
| whole block | 352.5 | 346 |
| q, k, v (3 × Q6_K 4096 -> 8192) | 187.2 | 441 |
| proxy: one Q6_K 4096 -> 24576 | 169.5 | 487 |
| five small Q8_0 projections | 74.1 | 49 |
| - Q8_0 128 -> 8192, generic kernel | 28.8 | 39 |
| - Q8_0 4096 -> 128 | 3.2 | 174 |
| recurrence | 20.9 | 401 (state read and write) |
| output (Q6_K 8192 -> 4096) | 62.4 | 441 |
| **short-K: Q8_0 128 -> 8192** | **4.7** | 238 |
| **short-K: five small projections** | **22.7** | 160 |
| **short-K: whole block** | **302.6** | 404 |

The generic Q8_0 kernel (`_lcpp`: 2 rows per 128-thread threadgroup, K
split across lanes and simdgroups) keeps only 16 of 128 lanes busy on a
4-block row. `kernel_mat_vec_q8_0_f32_short_k` gives each row `2 * nb`
lanes and a simdgroup `16 / nb` rows.

A one-dispatch proxy for q/k/v would save ~18 µs per block (~0.6 ms per
token). That is left for later.

## Change and exactness

GLM decode and packed Exact rows use the short-K kernel for `ssm_f_b` and
`ssm_g_b`; packed Fast keeps mat-mat. Other families never call it: a
census of the local GGUFs found Q8_0 with K <= 256 only in GLM and in the
DFlash2 selectors. Exactness is **Numerical (requalified)**. Live GLM gates
under `MTL_DEBUG_LAYER=1` pass 14/14:

- Exact packed == serial decode (max|dlogit| 0, 9 cases);
- near-4096 Exact prompt-end KL 2.95e-8, worst 3.5e-5, top-1 33/33;
- packed sparse 1.05e-5 at 2092 (gate 1e-2);
- frontier top-1 46/46.

## Results (tok/s, mean ± stddev of 3; pp4096 2 runs)

| Test | Depth | 344da7e7 | dc2d5ada | Change |
|---|---:|---:|---:|---:|
| tg128 | 0 | 30.03 ± 0.22 | **31.43 ± 0.14** | **+4.7%** (33.30 → 31.81 ms) |
| tg128 | 4096 | 28.46 ± 0.04 | **29.72 ± 0.03** | **+4.4%** (35.14 → 33.65 ms) |
| pp32 | 0 | 61.35 ± 0.05 | 60.84 ± 0.59 | -0.8% |
| pp128 | 0 | 127.90 ± 0.72 | 127.45 ± 0.64 | -0.4% |
| pp512 | 0 | 206.23 ± 1.79 | 206.15 ± 3.72 | 0.0% |
| pp32 | 4096 | 55.87 ± 1.04 | 55.74 ± 0.32 | -0.2% |
| pp128 | 4096 | 108.18 ± 0.33 | 108.29 ± 0.08 | +0.1% |
| pp512 | 4096 | 175.47 ± 0.96 | 175.46 ± 0.17 | 0.0% |
| pp4096 | 0 | 182.30 ± 0.13 | 181.77 ± 1.34 | -0.3% |

Decode saves 1.49 ms per token, close to the screen's 1.7 ms. Prefill is
unchanged: packed Fast does not use the kernel.

## Decision

Promote. Since P6 started, GLM tg128 has gone from 27.6 to 31.4 tok/s at
depth 0 and from 20.8 to 29.7 at depth 4096. Placement-matched llama.cpp
`e1425c0be` measured 23.7.
