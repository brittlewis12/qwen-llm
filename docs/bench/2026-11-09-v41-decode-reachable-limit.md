# V4.1 decode reachable limit and per-dispatch census, 2026-11-09

Read-only model; two candidate changes were built, measured and rejected.
M5 Ultra, production four-shard
`GSQ-RCO IQ2_XS` artifact, 32K prefix. "Isolated service" below is the sum of
per-dispatch GPU durations from `profile::Scope::begin_dispatches` (every
dispatch in its own sampled encoder): it removes concurrency and so is a
service budget, not a production wall; compare it with an unprofiled round to
see how much overlap the causal executor already recovers.

## Bottom line

- **B=3 is the workload.** This session at 32K: B=1 **14.02 ms** native /
  **13.76 ms** causal; B=3 **28.09 ms** native / **25.09 ms** causal. The
  causal executor is already default-on and worth +1.9% / +10.7%.
- **Dead time is 1.3-1.5 us per dispatch, not 4.7 us.** The frozen-graph
  4.65 us/dispatch is a *cost per removed dependency* (it also overlaps real
  per-sequence work), not launch dead time. The reachable launch budget at B=3
  is ~2,189 dispatches x ~1.4 us ~= **3.1 ms**, and the causal executor already
  hides most of it (isolated serial wall 37.8 ms vs 25.1 ms production).
- **At B=3 the isolated service is 34.3 ms and splits routed 30%, attention
  path 45%, shared 9%, small per-sequence rest.** The closable shared-band work
  is the batched dense Q5_K occupancy and the per-sequence small kernels;
  routed is mixed-instruction bound and mostly not closable.
- **Wall pricing (`decode_bounds`, B=3 32K): batched dense 8.51 ms, routed
  8.36 ms (2.60 compute), attention core 3.45 ms, posts 1.68 ms, attention
  staging 1.00 ms, HC pre 0.38 ms.** The two largest are the dense band
  (618 GB/s, ~3.75 ms to streaming) and the attention core (~4 TFLOP/s,
  ~2.4-3.1 ms with half MMA).
- **Reachable B=3 is ~20-22 ms** (from ~25-28 ms) by first building the
  3-column dense Q5_K path (call: largest bound, lowest effort), then the
  attention-core half-MMA path; staging is a smaller follow-on. Grid-filling
  and per-sequence batching/rescheduling were measured neutral and are not
  levers. Routed read caps any further swing.

## 1. Byte and math floors

Active weight bytes from the artifact census (`role` summed over `active`):

| role | GB |
| --- | ---: |
| attention Q5_K/Q8_0/BF16/F32 | 3.823 |
| routed experts IQ2/IQ3 | 2.329 |
| FFN Q5_K/BF16/F32 | 1.131 |
| head Q5_K | 0.455 |
| Engram Q5_K | 0.216 |
| HC F32 | 0.079 |
| **total** | **8.034** |

- Measured streaming is **1.106 TB/s**; the census one-read floor at 1.0 TB/s
  is 8.193 ms at 32K, so B=1 floor at 1.106 TB/s is **7.41 ms**.
- B=3 floor: shared weights once, plus three sequences' routed expert reads.
  Top-6 of 384 per sequence gives ~17.8 distinct experts for three sequences
  (~3.0x the one-sequence routed bytes) and three KV/index rings, about
  **13.1 GB -> 11.8 ms**. The marginal cost is genuinely per-sequence.
- Math floor: about 31 useful GFLOP/token (16.99 routed, 3.22 attention,
  ~2.8 shared, ~6.7 attention dense, ~1.5 head) -> **1.3 ms** at the measured
  24.1 TFLOP/s sixteen-chain FMA rate. Bytes bind, not FLOPs, except in the
  routed kernels' integer/table work.

## 2. Dependent-dispatch dead time (probe)

`examples/dispatch_deadtime.rs` on `measure/fn-widths-deadtime`, one command
buffer, 64 dispatches, per-dispatch slopes in us:

| shape | serial-dependent | serial-independent | concurrent | recover by concurrency |
| --- | ---: | ---: | ---: | ---: |
| tg8x128 | 1.19 | 1.06 | 0.15 | 1.04 |
| tg64x128 | 1.36 | 1.29 | 0.54 | 0.83 |
| tg384x128 | 1.60 | 1.54 | 0.57 | 1.03 |
| tg1536x128 | 2.27 | 2.11 | 1.26 | 1.01 |
| small-matvec (4x128) | 1.97 | 1.69 | 0.12 | 1.85 |
| norm (32x128) | 2.74 | 2.21 | 0.48 | 2.26 |
| reduction (1x256) | 2.67 | 4.03 | 0.38 | 2.30 |
| expert-wide (384x64) | 1.89 | 1.82 | 0.76 | 1.13 |

Serial-dependent and serial-independent agree to within noise: the cost is
launch/encoder overhead, not dependency. A concurrent encoder removes
0.8-2.3 us per dispatch. Payload beyond ~128 KB adds *service* time at the
probe's tiny grid (524 KB took 3.35 us), not dead time. The earlier 4.7 us is
therefore an upper bound that bundles real serialized work.

## 3. Per-dispatch census (this session)

Solo: **903 dispatches/round**, isolated service **16.79 ms** vs 13.76 ms
production wall. Batched B=3: **2,189 dispatches/round**, isolated service
**34.33 ms** vs 25.09 ms production wall. Per-family isolated ms (share of
service):

| family | B=1 | B=3 | B=3 share |
| --- | ---: | ---: | ---: |
| routed (gate/up, down, route, staging) | 4.41 | 10.32 | 30.1% |
| attention core (flash, combine/RoPE) | 1.28 | 4.64 | 13.5% |
| attention qkv (q_a/kv, q_b, norms, staging) | 2.56 | 4.19 | 12.2% |
| attention output (output_a/b, staging) | 2.54 | 3.91 | 11.4% |
| shared expert | 2.20 | 3.10 | 9.0% |
| attention index + owner | 0.89 | 2.87 | 8.4% |
| ffn post (combine + HC post) | 0.43 | 1.51 | 4.4% |
| hc post | 0.36 | 1.18 | 3.4% |
| ffn router | 0.37 | 1.11 | 3.2% |
| head | 0.41 | 0.53 | 1.5% |
| engram projection | 0.42 | 0.34 | 1.0% |
| hc pre | 0.32 | 0.62 | 1.8% |

The dense Q5_K kernels never reach streaming for medium `n_out`:

| projection | n_out | isolated GB/s |
| --- | ---: | ---: |
| head | ~150k | 1,123 |
| attentional output_b (B=3 skinny) | 5,120 | ~600 |
| shared expert (B=3 skinny) | 2,304 | 369 |
| shared expert (B=1 split) | 2,304 | 468 |
| q_b (B=1 split) | 32,768 | 693 |

The trend is isolated grid occupancy: the B=3 skinny launch covers
`ceil(n_out/16)` threadgroups x K-split, and for `n_out = 2304` the split
stopped at a divisor of the block count (5), leaving only 720 of the 2,560
target simdgroups. **Do not read this as recoverable headroom:** filling the
small-`n_out` grid was measured and is neutral-to-slower on the overlapped
schedule (section 5). The isolated deficit reflects a kernel alone on the GPU,
not the production schedule.

## 4. What is closable

| bucket | isolated gap vs byte floor | realistic recovery | how |
| --- | ---: | ---: | --- |
| dense Q5_K (shared/attention) | ~2-4 ms | 1-2 ms | narrower column tiles (grid fill is neutral, measured) |
| per-sequence posts (2.7 ms isolated) | 1.55 ms wall | 0.5-1 ms | shorten the dependency tail, not batch (batching neutral, measured) |
| attention staging (gather/scatter/rms) | 0.86 ms wall | 0.5-0.8 ms | column-native state; no gather |
| attention core/index | ~3-4 ms | 0.5-1 ms | per-sequence KV; not occupancy-bound at 32K |
| routed | 8.27 ms wall (5.6 read + 2.66 compute) | 0.3-0.5 ms | read at ~1.23 TB/s; mixed-instruction; V41-VY3X ceiling |
| host/command gaps | ~3.5 ms | 0.5-1.5 ms | fewer command buffers / shorter encode |
| **B=3 total** | **~13 ms** | **~2.5-4 ms (10-16%)** | |

### 4b. Wall pricing (`decode_bounds`, B=3, 32K)

Counterfactuals remove work rather than reschedule it, so these are wall
deltas, not isolated service. Two controls bracketed each other to 0.1%
(27.943 / 27.916 ms):

| mode | ms/round | saving | what it removes |
| --- | ---: | ---: | --- |
| Reference | 27.943 | - | - |
| AttentionIo | 27.052 | **0.86** | batched gather/scatter columns + rank-norm phase |
| PostControls | 26.369 | **1.55** | `hc_post_controls` + `ffn_post_controls` |
| AllHc | 25.995 | 1.93 | also `hc_pre_batch` (so HC pre alone 0.38) |
| RoutedRead | 25.284 | 2.66 | routed compute, keeping its read |
| RoutedHold | 19.645 | 8.27 | routed read + compute |

Two readings matter:

- **Routed is not a read problem.** The read-only arm keeps ~5.6 ms for the
  ~6.9 GB of per-sequence expert bytes (~1.23 TB/s, at/above the streaming
  reference); the removable part is 2.66 ms of mixed-instruction compute.
- **The batched-path "fixed overhead" is not one removable term.** It is the
  per-sequence post work (1.55 ms) and attention staging (0.86 ms) sitting on
  the dependency chain. The posts are already co-scheduled across sequences by
  the causal executor, so their cost is the tail of the layer, not launch
  count (confirmed by the neutral batching experiment below).

### 4c. Dense and attention-core bounds (B=3, both depths)

New `decode_bounds` modes `DenseHold` (skip the batched dense Q5_K/Q8_0
projections) and `AttentionCoreHold` (skip the flash partial + combine/RoPE).
Controls mean 28.465 ms (32K) / 28.743 ms (64K):

| mode | 32K saving | 64K saving |
| --- | ---: | ---: |
| AttentionIo (staging) | 1.00 | 0.95 |
| PostControls | 1.68 | 1.48 |
| RoutedRead (compute) | 2.60 | 2.49 |
| RoutedHold | 8.36 | 8.25 |
| **DenseHold (dense projections)** | **8.51** | **8.43** |
| **AttentionCoreHold (flash core)** | **3.45** | **3.38** |

The dense bound is much larger than the isolated framing suggested. The
batched dense band reads ~5.26 GB/round (4.87 GB Q5_K + 0.39 GB Q8_0 pair)
and costs 8.51 ms on the wall: **618 GB/s, 56% of the 1.106 TB/s streaming
reference.** The headroom to streaming is **~3.75 ms**, all on the critical
path (the hold saving equals the isolated service, so it is not overlapped).
This is the single largest credible lever: a 3-column path that reads each
weight once and does three FMAs per element (no 8-wide tile waste) targets
~900-1,100 GB/s. The B=1 split kernel already reaches ~859 GB/s for one
column, so the 618 GB/s at B=3 is a batched-path artifact, not a floor.

The attention core is 3.45 ms of wall at ~9.66 GFLOP/round (~4 TFLOP/s). It is
not occupancy-bound. Half-operand simdgroup MMA (or Metal 4 tensor ops) at
10/20/40 TFLOP/s would run the same FLOPs in 0.97/0.48/0.24 ms, i.e. **~2.5-3.2
ms recoverable**; the KV cache is already 16-bit, so half operands are a
numerical question (dNLL/KL gate), not a new policy.

### 4d. Ranking by credible wall bound (per unit of effort)

| # | lever | wall bound | credible recovery | effort |
| ---: | --- | ---: | ---: | --- |
| 1 | **3-column dense Q5_K** | 8.51 ms exposed | **~2.5-3.7 ms** | one kernel family |
| 2 | attention-core MMA/half | 3.45 ms exposed | ~2.4-3.1 ms | new attention core |
| 3 | attention staging | 1.00 ms exposed | ~0.3-0.4 ms (Flash-Next wash) | sequence-axis kernels |
| 4 | routed compute | 2.60 ms exposed | ~0.25 ms (V41-VY3X ceiling) | IQ2 kernel |

**Call: build the 3-column dense path first** — largest bound, smallest effort.
Attention-core MMA is the next-largest and the right second swing.

### 4f. B=3 encode breakdown (host timers, per round)

`hostgap` with section timers (`batch_embedding`, `decode_engram_rows`,
`batch_metal_encode` incl the per-layer causal replay, `batch_cmd_alloc`,
`batch_encoder_end`, `batch_commit`):

| section | 32K | 64K |
| --- | ---: | ---: |
| total encode | 4.42 ms | 4.44 ms |
| embedding decode/write | 0.002 | 0.004 |
| Engram row decode/write (CPU) | 0.90 | 0.76 |
| Metal command encoding (incl causal replay) | 3.53 | 3.47 |
| cmd alloc / encoder end / commit | 0.084 | 0.078 |

**~80% is Metal command encoding, ~20% CPU staging** (Engram rows dominate
staging; embedding is negligible). So the encode-side fix is pipelining the
next round's encoding under the current GPU work (or fewer dispatches /
pre-encoded command replay), not a non-pipelined GPU-inputs step.

## 5. Changes evaluated (eight negative, all deleted)

None is shipped. The lesson is that the isolated underutilization is not
recoverable on the production schedule: the causal executor already fills the
idle SIMD slots with other work, so reducing a single kernel's dispatch or
raising its grid adds overhead without shortening the wall.

- **Two-row Q5_K matvec** (`kernel_mat_vec_q5_K_f32_r2`, `NR0=2`).
  Off-path for V4.1: the census shows `kernel_expert_native_q5_K_f32_split`
  at B=1 and `kernel_dense_skinny_mma_q5_K_f32` at B=3, never the matvec that
  this changed. Paired same-binary off/on: **21.630 vs 21.719 ms** at B=2 and
  **26.429 vs 26.583 ms** at B=3 (level-1 thermal), no win.
- **Ragged skinny K-split fill** (`QWEN_V41_SKINNY_FILL`). Both skinny
  kernels walk `b = sz; b < nb; b += nz` and reduce over `nz`, so a slice
  count that does not divide the block count is exact (witness: max relative
  error 9.09e-7 / 7.26e-7 against the per-column kernels; 5 passed, 2 ignored).
  It raised the shared expert from 720 to 1,152 simdgroups and output_a from
  1,024 to 2,048. Paired same-binary ABBA at 32K, all retained arms thermal 0
  except one B=1 arm at level 1 (flagged):

  | width | fill off | fill on | delta |
  | ---: | ---: | ---: | ---: |
  | 3 | 26.058 ms | 26.245 ms | **+0.71%** |
  | 1 | 14.199 ms | 14.516 ms | +2.24% (one level-1 arm; noisy) |

  The extra K slices add a partial reduction per threadgroup; the GPU was
  already busy, so the fill only adds work. Grid occupancy of an isolated
  kernel is not a production lever here.
- **Sequence-batched attention mHC post** (`kernel_deepseek_v41_hc_post_controls_batch`).
  One dispatch with a sequence depth row instead of one post per sequence;
  bitwise per sequence (witness: 1 passed). Paired same-binary ABBA at 32K,
  every arm bitwise-identical full-vocabulary hashes:

  | width | batch off | batch on | delta |
  | ---: | ---: | ---: | ---: |
  | 2 | 21.675 ms | 21.719 ms | +0.20% |
  | 3 | 26.223 ms | 26.238 ms | +0.06% |
  | 4 (falls back; noise floor) | 30.329 ms | 30.360 ms | +0.10% |

  The B=4 row is a same-code control, so **<~0.2% is the noise floor**. The
  causal-executor waves already co-schedule the per-sequence posts; merging
  them changes neither the arithmetic nor the critical path. Deleted.
- **Controls computed after HC pre, post apply-only** (`QWEN_V41_HC_CONTROLS_EARLY`).
  Emit the sigmoid/Sinkhorn prologue as a 1-threadgroup kernel right after HC
  pre (so it can overlap the attention core / routed experts), and have the
  post load `pre`/`post`/`combination` and only apply. Bitwise per sequence
  (witness `controls_then_apply_matches_fused_post_bitwise`; every arm's
  full-vocabulary hashes identical). Paired same-binary ABBA at 32K:

  | width | controls off | controls on | delta |
  | ---: | ---: | ---: | ---: |
  | 1 | 14.067 ms | 14.129 ms | +0.44% |
  | 2 | 21.982 ms | 21.957 ms | -0.11% |
  | 3 | 25.909 ms | 26.095 ms | +0.72% |
  | 4 | 30.358 ms | 31.274 ms | **+3.02%** |

  Splitting the prologue costs more than it hides: the fused post computes the
  controls in threadgroup memory with the apply, whereas the split adds a
  global write/read round-trip and a dispatch the apply must wait for. **The
  post tail is not the Sinkhorn prologue**; it is the apply's own work and the
  layer dependency. Deleted.
- **Multi-column FMA dense Q5_K** (`kernel_dense_multicol_q5_K_f32`,
  `QWEN_V41_DENSE_MULTICOL`). Each simdgroup owns one output channel and keeps
  one accumulator per column, so each weight block is read and dequantized once
  and every element feeds `cols` FMAs. Bitwise the per-column split kernel
  (witness `dense_multicol_q5_k_is_bitwise_per_column`), and signed dNLL/KL vs
  the MMA path at B=3 32K was clean: **dNLL -2.11e-7, mean KL 7.35e-11, 192/192
  argmax equal, qualified**. But it is much slower:

  | width | MMA | multi-column FMA | delta |
  | ---: | ---: | ---: | ---: |
  | 2 | 22.071 ms | 23.187 ms | +5.06% |
  | 3 | 26.382 ms | 30.450 ms | **+15.42%** |
  | 4 | 30.542 ms | 37.446 ms | **+22.60%** |
  | 3 @64K | 26.089 ms | 30.234 ms | +15.89% |

  The 8x8 simdgroup MMA is far more efficient per weight byte than scalar FMA
  even with 5/8 of the tile wasted at 3 columns. **The dense 618 GB/s is not an
  FMA-vs-MMA problem.** The MMA is the right primitive; the remaining dense
  levers are half-operand MMA (double MMA throughput; dNLL/KL gate) or a better
  MMA tiling/occupancy, not an FMA rewrite. Deleted.
- **Decode attention core on the existing half simdgroup-MMA kernels**
  (`score_mma` -> `softmax_rows` -> `pv_mma` -> `normalize_rows` ->
  inverse RoPE; `QWEN_V41_ATTN_MMA`). These reuse the prefill sink-tile
  history-tile convention with decode's F32 ring (no KV conversion), so it is
  host-only wiring. Numerically qualified but catastrophically slower:

  | width | flash (F32) | MMA | delta |
  | ---: | ---: | ---: | ---: |
  | 1 | 14.098 ms | 36.384 ms | **+158.1%** |
  | 2 | 22.297 ms | 43.916 ms | **+97.0%** |

  Teacher-forced at B=3 32K: dNLL +0.0026, mean KL 3.6e-4, 192/192 argmax
  equal, qualified — so **half operands are quality-viable**; the problem is
  occupancy. `score_mma`/`pv_mma` are row-batched: a decode step is one query
  per sequence, so they launch **4 threadgroups** (heads/16) and loop ~40 key
  blocks serially, versus the flash kernel's ~160 threadgroups (20 key splits
  x 8 head groups). **The existing tensor kernels cannot serve single-query
  decode.** Deleted.
- **Key-split half-MMA flash partial** (`kernel_deepseek_v41_attention_flash_mma`,
  `QWEN_V41_FLASH_MMA`). Kept the flash split/softmax/combine structure but did
  QK (M = 8 heads) and PV (value dim split across 8 simdgroups) with
  `simdgroup_half8x8`, half rows staged from the F32 ring/compressed on load,
  reusing `v41_flash_split_softmax` and the combine. Two split sizes tried:
  KB=16 (40 splits) was +0.96% at B=3 32K because it doubled the combine; KB=32
  (20 splits, chunked 128-dim Q/K staging) gave the mixed table below.
  Numerically qualified at B=3 32K (dNLL +0.0017, mean KL 3.4e-4, argmax equal).

  | width | flash | MMA | delta |
  | ---: | ---: | ---: | ---: |
  | 1 (32K) | 14.065 ms | 14.358 ms | +2.08% |
  | 2 (32K) | 21.731 ms | 21.724 ms | -0.03% |
  | 3 (32K) | 26.659 ms | 26.392 ms | -1.00% |
  | 4 (32K) | 30.278 ms | 30.476 ms | +0.66% |
  | 3 (64K) | 26.163 ms | 26.454 ms | +1.11% |

  **Limiter:** the decode core is not QK/PV-compute-bound enough for MMA to
  pay. The combine is ~1.2 ms (unchanged, ~35% of the 3.45 ms core), the split
  softmax and the F32->half staging add work MMA cannot remove, QK uses only
  4 of 8 simdgroups, and chunked staging adds barriers. Deleting the kernel.
  The 4-TFLOP/s / 2.4-3.1 ms ceiling over-attributed the core to QK/PV; most
  of it is combine, softmax and staging. The work is preserved in branch
  commits `71e2b8834` (KB=32) and `fd7faef3f` (KB=16) for a future tune.
- **K64 attention splits to halve the combine** (`flash_h4_k64_d512`,
  `QWEN_V41_FLASH_K64`). KB=64 (10 splits, HG=4 keeps 160 threadgroups) should
  halve the combine's `used` loop. It regressed: **B=3 32K 26.470 -> 26.879
  (+1.55%)**, **64K 25.997 -> 26.491 (+1.90%)**. The 64-key partial loses more
  (fewer head groups, more keys per simdgroup) than the halved combine saves.
  Deleted. **Attention work stops here**: the combine bound was 1.06 ms, but no
  tried restructuring beats it net.

### 4e. Attention sub-item pricing (B=3, this session)

New `decode_bounds` modes `FlashPartialHold` (QK + split softmax + PV + KV
reads) and `FlashCombineHold`; controls 28.327 (32K) / 28.627 (64K):

| item | 32K saving | 64K saving |
| --- | ---: | ---: |
| flash partial (QK + split softmax + PV + KV) | 2.38 ms | 2.33 ms |
| flash combine (+ inverse RoPE) | 1.06 ms | 1.03 ms |
| whole core (`AttentionCoreHold`) | 3.32 ms | 3.28 ms |

The split softmax is inlined in the partial kernel, so it cannot be isolated
by a dispatch-level counterfactual; it is inside the 2.38 ms partial together
with QK/PV and the KV reads.

**Half KV cache bound.** The core reads `count = window 128 + selected 512`
keys x 512 dim x 4 B, twice (QK and PV), across 40 layers x 3 sequences = ~314
MB/round -> **~0.29 ms floor** at 1.106 TB/s. The current F32 flash reads F32
directly (no staging conversion). A persistent half ring/compressed cache would
halve the read (~**0.15 ms** saved) and remove a staging step that does not
exist on this path; the append-side write is ~0.24 MB/round (negligible). **The
half-KV bound is ~0.15 ms, below the 0.5 ms bar.**

**Host bubble (not in these GPU bounds).** `hostgap` at B=3: wall **27.43 ms**
(32K) / 26.61 ms (64K), commit/wait **20.84 / 21.31 ms**, encode **4.53 / 4.74
ms**. The GPU-span field in this probe is miscomputed (read 1.55 ms), so use a
conservative proxy of wall - commit_wait = **~6.6 ms (32K) / ~5.3 ms (64K)** as
an upper bound on the non-GPU bubble; it is consistent with the separate
`perf/batch-gpu-loop` lane's ~7 ms. It does not appear in any GPU
counterfactual. V4.1's `gpu_step::batch` decodes embedding/Engram rows on the
CPU; that lane owns those inputs, so this lane does not edit `gpu_step/batch.rs`.

**Decision (0.5 ms bar):** the combine (**1.06 ms**) is the only attention
sub-item that clears it; the half-KV cache (~0.15 ms) does not, and the split
softmax is not isolatable. Next build: a cheaper combine (better occupancy,
split-reduction, or fused into the following output projection) against the
1.06 ms bound.

## 6. Next (ranked by the 4c/4d bounds)

1. **Attention: stopped.** The combine (1.06 ms bound) did not survive
   restructuring (K64 +1.55%/+1.90%); the half-KV cache (~0.15 ms) and split
   softmax (not isolatable) are below the bar. Do not reopen without a new
   idea (fusing the combine into the following output projection).
2. **Batch host encode (~4.4 ms/round) is ~80% Metal command encoding, not
   staging** (see 4f): the fix is pipelining the next round's encode under the
   GPU or fewer dispatches/pre-encoded replay, owned by the batch-gpu-loop lane.
2. **Dense half-operand MMA** (revised dense plan): the FMA rewrite lost 5-23%,
   so keep the 8x8 MMA and cut its throughput cost with `simdgroup_half8x8`
   operands (dNLL/KL gate). Ceiling up to ~3.75 ms if the dense is MMA-bound.
3. **Attention staging** (~1.0 ms exposed): column-native state or sequence-axis
   projections. Realized likely ~0.3-0.4 ms (Flash-Next's equivalent cut was a
   wash because strided reads got slower).
4. Do **not** reschedule per-sequence work again (batching, controls-early,
   grid fill all neutral/slower), and do **not** replace the 8-wide MMA with
   scalar FMA (measured +15% at B=3). Re-price routed only against a kernel
   that beats the V41-VY3X +9.4% ceiling.
