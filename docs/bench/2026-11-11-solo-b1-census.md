# Solo B=1 decode census and reachable limit, 2026-11-11

M5 Ultra (Apple M5 Ultra, 80-core, SDK 27.0), branch `perf/solo-decode`, base
`7c6062f18`. Real solo paths: Flash-Next and GLM GPU token loops, V4.1
`Forward::step` (the `decode_serial` forward). 32K context. `--profile iter`
for the counterfactual and graph probes; release CLI for the serve baseline.
No production optimization shipped: the three tested kernel levers below
failed the >=1.25x isolated gate.

## Standings and byte floors

| Family | Path | Measured | Floor | Gap | Roof |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Next | B=1 GPU loop | 12.03 ms/token | 4.73 ms | 7.30 ms | 39% |
| V4.1 | `Forward::step` + host sampling | 14.02 ms/token | 7.41 ms | 6.61 ms | 53% |
| GLM | B=1 GPU loop | 17.62 ms/token | 9.56 ms | 8.06 ms | 54% |

Floors are one logical read of active weights plus the depth's KV/state at the
measured streaming reference (FN 1.184 TB/s, V4.1 1.106, GLM 1.106). They are
optimistic: cache reuse, repeated activation traffic and dispatch latency
remain in the gap.

## V4.1 B=1, causal schedule on

`v41_batch_depth MODEL 32768 1 32 bounds` (width parameterized for this run).
Reference is the mean of the first and last controls.

| Mode | ms/token | Saving vs ref | Removes |
| --- | ---: | ---: | --- |
| Reference | 14.019 | — | — |
| DenseHold | 8.746 | **5.273** | Q5_K/Q8_0 dense projections |
| RoutedHold | 10.339 | **3.680** | routed read + compute |
| AttentionCoreHold | 12.983 | 1.036 | flash partial + combine/RoPE |
| FlashPartialHold | 13.312 | 0.707 | QK + split softmax + PV + KV reads |
| AllHc | 12.944 | 1.075 | HC pre + post |
| PostControls | 13.547 | 0.472 | HC post + FFN post controls |
| FlashCombineHold | 13.659 | 0.360 | combine + inverse RoPE |
| IndexHold | 13.722 | 0.297 | index query/scores/sort/select |
| RoutedRead | 13.438 | 0.581 | routed compute only, read kept |
| AttentionIo | 13.989 | 0.030 | gather/scatter columns + rank norm |

Expert-stream witness: 3.63 ms/round for the captured routed read at
**896 GB/s** logical; the routed byte floor is 2.94 ms. Causal graph: 701
waves, 41 scopes. Per-layer 0.31-0.52 ms; the critical path is dense Q5_K ->
routed gate/up -> routed down, with the attention core and index mostly
co-scheduled inside those waves.

## Flash-Next B=1

`fn_solo_bounds MODEL 32768 32 bounds` (thermal level 1, flagged). Reference
12.03 ms/token; the order stopped at the `copies` hold because that group
breaks QSA status, so the remaining arms are controls.

| Hold | ms/token | Saving | Note |
| --- | ---: | ---: | --- |
| HC | 9.398 | **2.632** | up-mix + down-silu-inject + norms/inject |
| Q6 | 9.912 | **2.118** | Q6 body + head |
| Routed | 9.974 | **2.056** | gate/up + down + weighted sum |
| QSA core | 11.058 | 0.972 | split partial + decode + expand |
| GDN | 11.041 | 0.989 | controls prep + step decay + norm |
| Norms | 11.433 | 0.597 | overlaps HC norms |
| Routing | 11.692 | 0.338 | guarded top-k |
| Transport | 11.896 | 0.134 | no-op control |

Family byte floors from the 2026-11-06 census: HC 1.09 ms, Q6 1.59, routed
0.83, QSA 0.05, GDN 0.20. The corresponding exposed-minus-floor gaps are HC
**1.54**, routed **1.23**, QSA 0.92, GDN 0.79, Q6 0.53. Causal graph: 49
scopes, 898 waves for two tokens; per-layer 0.20-0.58 ms, six to eight HC
up/down waves and one Q6+routed chain per layer.

## GLM B=1

`glm_solo_bounds MODEL 32768 32 bounds` and `... 1 profile` (thermal level 1,
flagged). Reference 17.62 ms/token, GPU 16.72, encode 0.70. The routed hold
only matched `moe_down_iq4_xs` and `weighted_sum`; GLM's gate/up name is
`kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast_r1`, so the hold result
(0.345 ms) is not a routed-band price.

| Measurement | ms/token | Byte floor | Gap |
| --- | ---: | ---: | ---: |
| Q6 hold (body + head + shared) | 7.019 | 5.50 | **1.52** |
| Q8 hold (includes mHC pre-mix) | 2.013 | 0.78 | 0.7-1.2 |
| mHC (census service) | 1.11 | ~0 | **1.08** |
| MLA selected attention (census) | 0.76 | 0.02 | 0.74 |
| Routed gate/up + down (profile) | 4.33 | 2.66 | 0.88 |
| Index/selection (profile) | 0.76 | ~0 | 0.51 |
| KDA (profile) | 0.71 | 0.30 | 0.37 |

B=1 dispatch profile: 1,060 dispatches. Q6 kernels dominate the GPU sum
(`mat_vec_q6_K_f32_r1` 4.33 ms, `_x3` 2.95, shared 1.11, head 0.48 under
sampled encoders). The width-1 batch path (which has causal waves) measured
30.3 ms wall under profile and is not a faster solo path.

## Ranking by closable ms

Hold minus byte floor, largest first. Feasibility notes are from the probes
below and the 2026-11-06/10/11 entries.

| Rank | Family / band | Closable | Why it did not ship now |
| ---: | --- | ---: | --- |
| 1 | Flash-Next HC | 1.54 | up/down are globally separated; re-grid neutral |
| 2 | GLM Q6 | 1.52 | KDA QKV/head already at roof; shared/other needs a new Q6 mapping |
| 3 | Flash-Next routed | 1.23 | mixed IQ decode math; incumbent grid at read ceiling |
| 4 | GLM mHC | 1.08 | global reduction; needs a fused small-band design |
| 5 | V4.1 HC + posts | 1.07 | controls are only a small part; apply remains |
| 6 | V4.1 attention core | 1.04 | K64 and restructures already lost at B=3 |
| 7 | V4.1 dense | 0.95 | split-cols/MMA mapping closed at B=1 (below) |
| 8 | Flash-Next QSA core | 0.92 | selection status makes holds fragile; no isolated lever |
| 9 | GLM routed | 0.88 | same row/QB limits as V4.1 |
| 10 | Flash-Next GDN | 0.79 | small projections; fusion is the 1-2 week item |
| 11 | V4.1 routed | 0.74 | IQ2_XXS rows sweep neutral (below) |
| 12 | GLM Q8 | 0.7 | includes mHC pre-mix; no separate lever |
| 13 | Flash-Next Q6 | 0.53 | body/head near streaming |
| 14 | GLM index | 0.51 | small dependent sort/select waves |
| 15 | V4.1 index | 0.30 | small |

## Tested and rejected levers

### B=1 dense split-cols / MMA

`dense_band_census --cols 1 --reps 12 --target-mib 384`. The production
per-column kernel already beats the padded 8-row MMA and the split-cols FMA
at one activation column:

| Shape | Per-column us (GB/s) | MMA auto us | split_cols us |
| --- | ---: | ---: | ---: |
| q_b 1280->32768 | **87.1 (331)** | 89.6 | 92.3 |
| output_b 8192->5120 | **39.5 (730)** | 57.1 | 44.0 |
| output_a 8x4096->1024 | 32.5 (710) | 49.6 | **29.4 (786)** |
| engram 6144->25600 | **98.4 (1098)** | 135.4 | 102.6 |
| q8_pair 5120->1280+512 | 19.0 (514) | **18.9 (516)** | — |

The only positive point is output_a split-cols at **1.11x**, below the gate.
The dense mapping is closed at B=1.

### V4.1 IQ2_XXS channels-per-simdgroup

The shipped split kernel uses two channels per simdgroup. The existing IQ2
probe was extended temporarily to instantiate rows 1/2/4/8 for IQ2_XXS
(bitwise rows-axis variants), at the real K=5120 gate/up and K=2304 down
shapes, 128-expert rotating banks, top-6:

| Projection | Generic | Shipped r2 | r1 | r4 | r8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | 56.7 us / 1.000x | 49.7 / 1.140x | 61.6 / 0.920x | 49.4 / 1.147x | 54.9 / 1.033x |
| down | 37.1 us / 1.000x | 32.6 / 1.140x | 39.8 / 0.933x | 32.1 / 1.155x | 36.4 / 1.019x |

r4 is **1.006x** gate/up and **1.016x** down versus the shipped r2. The
dominant routed dtype is already at its best rows mapping; the temporary
instantiations were reverted.

### Flash-Next HC up re-grid

`fn_hc_up_regrid` (research metallib) times the same BF16 up-mix arithmetic
with 2/4/8/16 hidden rows per threadgroup (64/128/256/512 threads):

| Rows/tg | us | GB/s | vs r4 |
| ---: | ---: | ---: | ---: |
| 2 | 13.750 | 476.6 | 0.991x |
| 4 (shipped) | 13.625 | 481.0 | 1.000x |
| 8 | 13.583 | 482.5 | **1.003x** |
| 16 | 13.667 | 479.5 | 0.997x |

No re-grid. The HC gap needs a different BF16 GEMV mapping or the mixed-format
multi-output dispatches already scoped in the 2026-11-06 census.

## Single-agent serve baseline (V4.1, real sampling)

One request, `QWEN_SERVE_V41_MAX_ACTIVE=1`, 31,918 prompt tokens, 64 generated
tokens, released sampling defaults (no sampling fields), release CLI:

- TTFT **33.58 s** (server `prefill_ms=33529.0`; 9 prefill chunks, shrinking
  CED + full).
- Decode **902.1 ms / 63 transitions = 14.32 ms/token**; `decode_tps=66.94`.
  Total 34.53 s.
- Coherent text: the model correctly described the repeated `qwen serve`
  contract and asked what to do with it.
- The V4.1 serve output partition buffered the SSE deltas until completion, so
  the client-observed inter-delta p50/p99 (0.015/0.018 ms) is a transport
  artifact, not token latency. The server has no per-token percentile field;
  only the aggregate `transition_ms`/`transitions` is available here.
- Flash-Next, same 32K prompt: TTFT **18.61 s**, total 19.54 s. Its SSE path
  also buffered (127 deltas, 64 tokens after intra-token splitting), and the
  released sampling produced a repetitive completion on the repeated-spec
  prompt, so only V4.1 is used as the coherence check. GLM was not re-run
  after the census; its decode standing remains the 17.62 ms/token GPU-loop
  number above.

## Added tools

- `v41_batch_depth` bounds/graph now accept width 1..4 (B=1 uses `Forward::step`).
- `fn_solo_bounds` (`perf-ablations`): Flash-Next B=1 family holds through the
  real GPU token loop plus a causal wave graph.
- `glm_solo_bounds` (`perf-ablations`): GLM B=1 family holds through the real
  GPU token loop plus a per-dispatch profile.
- `fn_hc_up_regrid` (`perf-ablations`): isolated HC up-mix rows-per-threadgroup
  probe; research kernel in `kernels/research/`.

## Evidence

M5 `~/qwen-runs/2026-11-11-solo-b1/`: `v41-bounds.log`, `v41-graph.log`,
`fn-bounds.log`, `fn-graph.log`, `glm-bounds.log`, `glm-profile.log`,
`dense-col1.log`, `iq2-rows.log`, `fn-hc-regrid.log`, and the V4.1 serve
`v41-32k-stats.json{,.server.log}`. Local worktree `/Users/tito/code/qwen-llm-solo`;
M5 worktree removed after the run. No merge.
