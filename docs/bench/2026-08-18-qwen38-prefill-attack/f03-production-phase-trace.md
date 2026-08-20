# F03 — Production phase trace: prefill is ~87% Q4_K matmul

Date: 2026-08-18. HEAD `307007d` (allow-dirty; another agent's work in tree, no
prefill code paths touched). Model: `Qwen3.8-27B-Q4_K_M.gguf`. Workload: pp2048
single chunk-p=512, warmup on, 1 timed run.
`QWEN_PREFILL_TRACE_LAYER_PHASES=1 QWEN_PREFILL_TRACE_FFN_SUBPHASES=1`.

## Aggregated across 64 layers × 4 chunks

Total GPU-ms (inclusive per-phase, sums to ~1.8× wall due to overlap between
command buffers — this attributes cost, not wall):

| Phase | ms | % | Kernel |
| --- | ---: | ---: | --- |
| ffn_down | 3757.4 | 22.2 | Q4_K matmul (f→h) |
| ffn_up | 3712.6 | 21.9 | Q4_K matmul (h→f) |
| ffn_gate | 3696.7 | 21.8 | Q4_K matmul (h→f) |
| gdn_qkv | 1652.0 | 9.7 | Q4_K matmul (h→conv_dim) |
| attn | 1283.0 | 7.6 | SDPA |
| gdn_back | 1003.0 | 5.9 | Q4_K matmul (v_dim→h) |
| gdn_z | 998.2 | 5.9 | Q4_K matmul (h→v_dim) |
| gdn_step | 454.3 | 2.7 | recurrence (algorithmic bottleneck) |
| ffn_swiglu | 105.0 | 0.6 | activation |
| gdn_beta_alpha | 92.4 | 0.5 | small matmul |
| gdn_prep_conv | 83.4 | 0.5 | conv + silu packed |
| Other | ~90 | 0.5 | norms, residuals, gated, prep_l2 |
| **Total** | **16952** | **100%** | |

**Buckets**:
- **Q4_K matmul (FFN + GDN front/back/z): 87.3%** of all prefill GPU-ms
- Attention (SDPA): 7.6%
- GDN state recurrence: 2.7% (algorithmic; can't parallelize positions)
- Everything else: <2%

## What corrects vs the debug phase profile

The prior `metal_dflash.rs:22073` test profile (F02 baseline analysis) said
FFN=50.4%, GDN=35.9%, Attn=13.6% at pp321 chunk_p=321. Production at pp2048
chunk_p=512 shows FFN=66.7%, GDN=25.7%, Attn=7.6%. Delta drivers:

- Test profile un-packs GDN prep to per-token launches (F01 map called this
  out mistakenly as production behavior); production uses
  `encode_gdn_prep_packed_f32` + `encode_l2_norm_pair_batched_f32` in a single
  launch per layer, driving GDN share down.
- Larger chunk_p means more FFN compute per prompt, so FFN share rises.
- These match production defaults — trust these, not the debug profile.

## What kernel production actually dispatches for FFN

`encode_mat_mat_q4_k_f32` at `metal.rs:7137` picks kernel by shape:

```rust
let use_n64 = mat_mat_q4_k_use_n64(n_in, n_out, n_query)
    && n_query.is_multiple_of(64)
    && n_out.is_multiple_of(64);
let kernel_name = if use_n64 {
    "kernel_mat_mat_q4_K_f32_n64"
} else if use_n16_v2 { "..._n16_v2" }
else if n_query == 16 { "..._n16" }
else { "kernel_mat_mat_q4_K_f32" };  // generic 32-column tile
```

The n64 gate at `metal.rs:7100`:
```rust
Auto => !(n_query <= 512 && (n_in <= 2048 || n_out <= 2048))
```

For our FFN gate at chunk_p=512, K=5120, M=17408: `n_query=512, n_in=5120,
n_out=17408` → **n64 fires**. Confirmed via inspection.

MMA variants exist (`kernel_mat_mat_q4_K_mma8_f32`, `mma8v_{r,c,k}` family in
`kernels/mat_mat_mma8.metal`) but the encode wrapper at `metal.rs:6770`
requires `y.n_elements == 8*n_out` — **fixed N=8 shape check**. Wired for
verify only, not prefill.

## Effective throughput vs known peak

FFN gate matmul: M=17408, N=512, K=5120.
- FLOPs per matmul: 2·M·N·K = **91.4 GFLOP**
- Weight bytes (Q4_K, 9 bits/elem): M·K·9/8 = **100.4 MB**
- Bandwidth floor at measured 474 GB/s stream: 100.4/474 = **212 μs/matmul**
- Compute floor at perflog `:17473` peak 12.53 TFLOP/s: 91.4/12530 = **7.29
  ms/matmul**
- **Observed 14.2 ms/matmul** (11.3 s FFN / 65 layers / 4 chunks / 3 matmuls)
- **51% of the kernel's known peak** at these shapes → ~2× headroom without
  new kernel work.

## Two structural gap sources vs MLX

1. **Tile shape at prefill N** (2× available). Precedent
   `PERF-LOG.md:163-166`: Ridge N=16 32-column tile fix moved 5240.6 →
   2255.4 ms (2.32×). Same class of shape cliff, different N. This is real
   kernel work but bounded scope. Anchored to n64 at prefill shapes; try
   MMA-based variants at variable N (currently N=8-locked), or a tile
   optimized for (M=17408, N=512, K=5120) specifically.
2. **F16 activations + accumulator** (2× on top). Q4_K peak is 12.53
   TFLOP/s at F32, ~26 TFLOP/s FP16 on M4 Max. No F16-activation Q4_K
   variant exists in the tree (grepped
   `mat_mat_q4_K_f16 / q4.*f16.*mat` — zero matches). New kernel needed to
   close the "MLX must be running lower-precision matmul" gap.

**Combined ceiling**: (1) × (2) = ~4×, which matches the observed MLX gap.

## Non-attack: attention

7.6% of prefill. Even halving attention only saves 3.8% total. Not the story.

## Next step

**Q4_K prefill tile sweep** at production FFN shapes (M=17408, N=512, K=5120;
M=5120, N=512, K=17408 for down). Microbench-style: run current `n64` at
these exact shapes as baseline, then measure MMA8 with a variable-N dispatch,
plus one or two hand-tuned candidates. This is the "tile-shape cliff repair"
precedented at v0.500 Ridge N=16 (`PERF-LOG.md:163-166`); expected win 1.5-2×
FFN alone = 10-13% of total prefill wall. Bounded scope, correctness-cheap
(compare against existing n64 output), maps directly onto the observed
half-of-peak throughput gap.

If that yields the expected 2×, revisit for the F16-activation kernel work
which requires larger investment. If not, the anchor was wrong somewhere and
we need to re-instrument before committing more.

## Artifacts

- Raw trace: `/tmp/pp2048-traced.err` (transient; regenerate via `qwen-bench
  pp --allow-dirty --n-prompt 2048 --runs 1 --output json` with the two
  QWEN_PREFILL_TRACE_* env vars set).
- Summary: this file.
