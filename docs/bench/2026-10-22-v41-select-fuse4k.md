# V4.1 prefill: 4K fused selection for late tiles

Candidate `perf/v41-select-fuse4k` (`742d6b75`) on main `df6d1451` (fused
selection for padded <=2048). M5 Ultra 80-core, Unsloth
DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 4096 --runs 2 --seed 1`, A/B/B/A, same binary.

The only difference at PP4096 is the last tiles of ratio-one producer
layers: their visible length exceeds 2048, so the 2K fused selector fell
back to the per-row sort. `QWEN_V41_SELECT_FUSE_2048=1` is the 2K cutoff
control. Raw: `metagross:~/qwen-runs/2026-10-22-select-fuse4k/`.

## Throughput

| shape | 2K cutoff | 4K fused | change |
|---|---:|---:|---:|
| Q2 pp4096 | 570.2 (569.7-570.7) | **622.2 (620.5-623.8)** | **+9.1%** |
| IQ3 pp4096 | 567.2 (566.5-567.9) | **616.8 (616.3-617.3)** | **+8.7%** |

PP2048 is unchanged: every tile there fits the 2048-entry selector, so the
4K kernel is not dispatched.

## Dispatch census

Q2 PP4096, one run:

- 2K cutoff: **173,913** dispatches
- 4K fused: **92,313** dispatches
- delta: **-81,600 (-46.9%)**

The 2K cutoff left the late ratio-one tiles on the per-row
`sort_init`/`sort_tile`/`sort_step`/`chronological` families; the 4K
threadgroup covers them in one dispatch per tile.

## What changed

The fused selector now has two static-threadgroup forms:

- `kernel_deepseek_v41_attention_select_fused_rows` uses 2048-entry
  `values`/`order` arrays (16 KiB) for visible lengths up to 2048.
- `kernel_deepseek_v41_attention_select_fused_rows_4k` uses 4096-entry
  arrays (32 KiB) only when the tile's last visible length exceeds 2048 and
  is at most 4096.

Splitting the kernels keeps the common path at 16 KiB instead of
allocating 32 KiB for every row. Above padded 4096 the per-row fallback
remains. Both kernels use the incumbent bitonic tie-break and emit
chronological IDs directly.

## Correctness

- The 65-row batch-vs-incumbent regression now includes `(start=3000, n=65)`,
  which crosses into the 4K selector, and asserts that a fused kernel is
  dispatched.
- 31 attention tests and 14 prefill tests pass.
- Same-binary held-text witness, IQ3, 384 targets: 4K cutoff and 2K cutoff
  give identical NLL and top-1 values:

| document | NLL | top-1 |
|---|---:|---:|
| prose_1 | 2.604260127 | 0.594 |
| prose_2 | 4.204815708 | 0.188 |
| prose_3 | 2.260082474 | 0.453 |
| code_4 | 1.252495368 | 0.719 |
| code_5 | 1.570008374 | 0.688 |
| code_6 | 0.986541110 | 0.766 |
| mean | 2.146367194 | 0.568 |

## Limits

- One artifact pair, one M5 Ultra. The 4K kernel's 32 KiB threadgroup
  footprint reduces occupancy; the measured pp4096 gain already includes
  that cost on this machine.
- Above padded 4096 (very long single prefill chunks) the per-row fallback
  remains.
- PP2048 is unaffected, which is why the header's pp2048 numbers do not
  change.
