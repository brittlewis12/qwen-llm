# GLM-5.3-Flash bitwise dispatch fusion: KDA q/k/v and shared SwiGLU (2026-10-05)

Leverage map 2026-10-05 #2. UD-IQ3_XXS, M4 Max 128 GB. Parent `a9fafb50` vs
`d786a9f5`, same session, back to back, untraced.

```sh
qwen-bench suite -m <shard 1> --pp 512 --tg 128 -d 0,4096 --runs 3
qwen-bench suite -m <shard 1> --tg 128 -d 0 --runs 3   # repeat
```

## Change

- **KDA q, k, v:** these are three Q6_K [4096 -> 8192] mat-vecs from one
  input. They now run as one dispatch (`kernel_mat_vec_q6_K_f32_x3`), which
  maps grid rows to one of three (weight, output) pairs and runs the single
  kernel's body verbatim. On the KDA screen, the one-dispatch proxy took
  169.5 µs per block against 187.2 µs.
- **Shared expert:** gate and up (Q6_K [4096 -> 2048]) plus the clamped
  SwiGLU now run as DS4's fused kernel (`encode_ds4_shared_swiglu_q6_k_f32`)
  instead of three dispatches.

Exactness: **Bitwise**.
- Unit tests compare each output with its separate-dispatch path, bit for
  bit.
- On real weights, a greedy 96-token GLM chat decode gives fingerprint
  `199e752f5e0d926c…` both before (`dc2d5ada`) and after
  (`greedy-*.jsonl`).
- Packed prefill keeps separate dispatches, which equal the fused ones
  bitwise, so the Exact lineage holds.

## Results (tok/s, mean ± stddev of 3)

| Test | Depth | a9fafb50 | d786a9f5 | Change |
|---|---:|---:|---:|---:|
| tg128 | 0 | 31.33 ± 0.13 | 32.16 ± 0.24 | +2.6% |
| tg128 (repeat) | 0 | 32.12 ± 0.01 | 32.78 ± 0.00 | +2.1% |
| tg128 | 4096 | 28.38 ± 0.09 | 30.10 ± 0.04 | +6.1% (see note) |
| pp512 | 0 | 209.84 ± 3.67 | 212.61 ± 3.75 | +1.3% (path unchanged) |
| pp512 | 4096 | 170.91 ± 4.88 | 175.20 ± 0.43 | +2.5% (path unchanged) |

Note: the parent's depth-4096 tg came in 1.3 tok/s below the same GPU
code's earlier measurement (29.72, `dc2d5ada` in the short-K packet), so
that cell overstates the change. At depth 0 the two measurements agree on
~0.6-0.8 ms per token, the size the screen predicts for q/k/v (~0.6 ms)
plus the shared-expert fusion.

## Decision

Keep. The change is bitwise and gains ~2% of decode. GLM tg128 is now
~32.2-32.8 tok/s at depth 0, against 27.6 at the P6 baseline and 23.7 for
llama.cpp.
