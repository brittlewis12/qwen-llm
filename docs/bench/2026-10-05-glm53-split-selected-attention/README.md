# GLM-5.3-Flash split selected attention (2026-10-05)

Leverage map 2026-10-05 #2. UD-IQ3_XXS, M4 Max 128 GB, AC power. Parent
`479fa9c1` (serial selected attention) vs `a99792ba` (split from
`ea49595f`, hardened in `02d3ccdf`). Same session, back to back, untraced,
`QWEN_METAL_LEASE_WAIT=1`, no other GPU work.

Reproduce (each arm checked out and rebuilt; the bench refuses a commit
mismatch):

```sh
qwen-bench suite -m <shard 1> --pp 32,512 --tg 128 -d 0,4096,8192 --runs 3
qwen-bench suite -m <shard 1> --pp 4096 -d 0 --runs 2
```

## Change

The serial kernel runs one simdgroup per (query, head) through all 2,051
selected slots. The split kernel cuts each query's rows (raw window, then
selected slots) into 128-row ranges: one simdgroup per (query, head, split)
writes an unnormalized partial, then one simdgroup per (query, head) folds the
partials in order into the sink's initial state. The partition depends only
on the query's own geometry. Packed sparse rows run it in 16-query
sub-batches that reuse one partial scratch; decode is one query.

Exactness: **Numerical (requalified)**. The reduction order differs from the
serial kernel's. Exact packed prefill still equals serial decode bitwise,
because both use the split. Live GLM gates under `MTL_DEBUG_LAYER=1` (14/14):

- near-4096 vs llama.cpp: prompt-end KL 2.402e-8, worst KL 3.343e-5, top-1
  33/33;
- packed sparse vs llama.cpp at 2092: KL 1.040e-6;
- frontier decode vs llama.cpp: top-1 46/46;
- selection replays: score error 0.

DS4 still runs the serial kernel and adopts the split only after its own
qualification.

## Results (tok/s, mean ± stddev of 3; pp4096 2 runs)

| Test | Depth | 479fa9c1 | a99792ba | Change |
|---|---:|---:|---:|---:|
| tg128 | 0 | 27.47 ± 0.06 | 28.09 ± 0.17 | +2.2% (no sparse rows; noise) |
| tg128 | 4096 | 20.81 ± 0.04 | **26.62 ± 0.15** | **+27.9%** (48.06 → 37.57 ms) |
| tg128 | 8192 | 20.92 ± 0.21 | **26.41 ± 0.03** | **+26.3%** (47.81 → 37.86 ms) |
| pp32 | 0 | 61.58 ± 0.09 | 60.24 ± 0.25 | -2.2% (no sparse rows; noise) |
| pp32 | 4096 | 55.47 ± 1.22 | 56.02 ± 0.13 | +1.0% |
| pp32 | 8192 | 54.69 ± 1.64 | 55.84 ± 0.06 | +2.1% |
| pp512 | 0 | 209.98 ± 3.55 | 211.08 ± 2.98 | +0.5% |
| pp512 | 4096 | 175.47 ± 0.57 | 170.62 ± 6.64 | -2.8% (within noise) |
| pp512 | 8192 | 174.15 ± 0.83 | 173.95 ± 0.59 | -0.1% |
| pp4096 | 0 | 181.38 ± 0.09 | 181.68 ± 0.14 | +0.2% (crosses the frontier) |

Decode attribution v2 (`../2026-10-05-glm53-decode-attribution/`): selected
attention goes from 11.60 to 1.05 ms per token at depth 4096, about 95 µs per
MLA block. The unprofiled step goes from 51.21 to 40.60 ms; depth 64 measures
38.58 ms.

## Decision

Promote. Decode at depth now costs ~2.0 ms per token over depth 0, against
~11.6 ms before. Packed prefill is neutral, so the 16-query sub-batches cost
nothing measurable at these shapes. The cx review's packed-prefill gate
(fresh pp4096, pp32 and pp512 at depth) is met.
