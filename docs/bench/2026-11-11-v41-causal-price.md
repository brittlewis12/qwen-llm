# V4.1 causal-schedule re-pricing, dense split-cols probe, member-axis price — 2026-11-11

M5 Ultra, four-shard `GSQ-RCO IQ2_XS` DeepSeek V4.1 Flash artifact, B=3.
Read-only pricing and probes; no production build was changed. Thermal
pressure was **0** at every probe start and end, and every lease acquisition
waited **0 s**. Worktree `perf/v41-causal-price`, no merge.

## Why this probe

The 2026-11-09 wall bounds (`DenseHold` 8.51, `RoutedRead` 2.60, core 3.45,
posts 1.68, staging 1.00 ms) were measured with the causal executor **off**:
`decode_bounds` forced the native path (`batch.rs`, `gpu_step.rs`). Production
runs causal waves (~25 ms against the bounds' ~28.4 ms controls), so those
marginal costs were native-schedule numbers. Two gaps follow:

1. Re-price the holds with causal waves enabled, and price the index chain,
   which had no counterfactual.
2. Dump one causal round's dispatch graph (kernel names, waves, per-wave GPU
   time) so the critical path is auditable.

Both are now implemented behind `perf-ablations`:

- `decode_bounds::IndexHold` skips the labelled `attn_index` chain plus the
  index kernel names, and leaves a bitwise-invalid but **valid** sequential
  `selected` repair (`kernel_deepseek_v41_attention_selection_repair`) so the
  core's key check stays at status 0. The bound therefore prices scoring +
  selection minus that tiny fill.
- `batch_causal_scope(true)` now wins over the counterfactual's native
  default, so the same holds run on causal waves.
- The causal recorder stores each node's `decode_bounds` label. Without this,
  label-based holds were evaluated with the scope's *last* label at replay
  time, not the label of the recorded dispatch.
- `examples/v41_batch_depth ... graph` records one causal round. M5 rejects
  `sampleCountersInBuffer:` inside a compute encoder, so the dump suspends the
  production encoder and encodes each wave in its own stage-boundary-sampled
  concurrent encoder. Waves were already separated by buffer barriers, so the
  wave graph and per-wave overlap are unchanged; the encoder transitions are
  diagnostic overhead outside the samples.

## Causal-on bounds (B=3, 32 rounds, `bounds`)

Controls are the first and last `Reference` rows. The 32K control drifted
0.29 ms across the run; the 48K control drifted 0.03 ms.

| mode | 32K saving ms | 48K saving ms | what it removes |
| --- | ---: | ---: | --- |
| Reference | 26.896 | 25.686 | — |
| AttentionIo | 0.51 | 0.73 | gather/scatter columns + rank-norm phase |
| PostControls | 0.76 | 0.61 | HC post + FFN post controls |
| AllHc | 1.27 | 1.45 | also HC pre |
| RoutedRead | 1.22 | 1.82 | routed compute, keeping its read |
| RoutedHold | 7.67 | 7.63 | routed read + compute |
| DenseHold | 6.92 | 6.51 | batched dense Q5_K/Q8_0 projections |
| AttentionCoreHold | 2.47 | 2.30 | flash partial + combine/RoPE |
| FlashPartialHold | 1.55 | 1.64 | QK + split softmax + PV + KV reads |
| FlashCombineHold | 0.59 | 0.60 | combine + inverse RoPE |
| **IndexHold** | **0.52** | **0.74** | index query/weight + scores + block/sort/select |

Reads:

- **Causal waves recover a lot of the native bound.** Dense falls 8.51 -> 6.92,
  routed compute 2.60 -> 1.22, core 3.45 -> 2.47, posts 1.68 -> 0.76, staging
  1.00 -> 0.51 ms. The native controls were ~28.4 ms; these are ~26.9 ms.
  Dense is still the largest single exposed band, but the earlier 8.51 was not
  a causal number.
- **Flash partial + combine are nearly additive** (1.55 + 0.59 = 2.14 vs core
  2.47 at 32K; 1.64 + 0.60 = 2.24 vs 2.30 at 48K), so the core bound is not
  hiding much overlap between its two halves.
- **The index is real but small.** 0.52 ms at 32K and 0.74 ms at 48K. Its
  exposed part is spread over 8 index layers (~0.06-0.09 ms/layer), not a
  single visible block.
- **Routed is a read band.** Removing only its compute saves 1.22/1.82 ms;
  removing the read as well saves 7.67/7.63.
- Controls and absolute walls are higher than the 2026-11-09 session (25.1 ms
  causal). Treat the *savings* as the comparable quantity; the 32K control
  drift bounds the noise at ~0.3 ms.

## Causal dispatch graph (one B=3 32K round)

41 scopes (40 layers plus the head segment), **1,062 waves**, sampled wave
spans sum to **24.33 ms**. The diagnostic wall (40.1 ms) is inflated by the
per-wave encoder transitions and is not a production number.

Representative per-scope totals: most layers **0.55-0.60 ms / 25 waves**; the
dense/index owner layers with extra columns run 0.67-0.82 ms / 29-43 waves;
the head scope is 0.54 ms / 5 waves.

Largest single waves:

| scope/wave | ms | kernels |
| --- | ---: | --- |
| 43/3 | 0.521 | `kernel_dense_skinny_mma_q5_K_f32` (head) |
| 3/1 | 0.264 | two `kernel_dense_skinny_mma_q5_K_f32` |
| 28/21 | 0.160 | routed `gate_up_iq2_s` + 3 `scatter_columns` |
| 15/21 | 0.152 | routed `gate_up_iq2_xs` + 3 `scatter_columns` |
| 32/21 | 0.141 | routed `gate_up_iq2_xxs_split` + 3 `scatter_columns` |

Wave families (a wave may contain more than one family, so the totals overlap):

| family | waves | summed ms | shape |
| --- | ---: | ---: | --- |
| dense skinny MMA | 202 | 8.16 | q_b/output_b/shared/head; dominant chain |
| routed gate/up | 29-41 | ~6 | one ~0.14-0.16 ms wave per routed layer |
| flash partial | 41 | 1.67 | ~0.041 ms/wave, 3 per-sequence dispatches in each wave |
| flash combine | 43 | 0.47 | ~0.011 ms/wave, 3 per-sequence dispatches |
| index (`index_heads`/select) | 47 | 0.84 | often shares a wave with RoPE or a flash partial |

The critical path is dense -> routed gate/up -> scatter/gather columns, with
the attention core and index almost entirely co-scheduled inside those waves.
That is why the causal core/index holds are much smaller than their isolated
service.

## Dense split-cols access-pattern probe

`kernel_expert_native_q5_K_f32_split_cols` keeps the work-split lane map
(four channels per 128-thread threadgroup, `yl`/`yh` registers,
scale-after-dot) and adds 1..4 activation accumulators; the weight block is
decoded once and feeds one FMA chain per column. `dense_band_census` gained a
`split_cols` arm and a witness against the production per-column split path.
All five shapes at cols=3 are **bitwise** (0 differing values, max abs 0.0).

Isolated rates at the real B=3 width, M5, thermal 0, 12 reps, 384 MiB rotating
banks (GB/s; higher is better):

| shape | per-column | MMA (auto) | best explicit split | split_cols | vs MMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_b 1280->32768 | 273.3 | 631.3 | 632 | **312.8** | 0.50x |
| output_b 8192->5120 | 352.6 | 662.9 | 672 | **438.4** | 0.66x |
| output_a 8x(4096->1024) | 348.7 | 603.1 | 611 | **452.6** | 0.75x |
| shared gate/up 5120->2304 | 172.6 | 314.7 | 284 | **221.8** | 0.70x |
| shared down 2304->5120 | 228.9 | 490.9 | 275 | **299.9** | 0.61x |

At cols=2 the best split_cols point is output_a at **621.4 GB/s** (MMA 605.7);
q_b is 441.5 vs 621.8, output_b 606.9 vs 666.1, shared 292.6 vs 319.6. At
cols=4 it is worse everywhere. No point reaches the **800 GB/s gate**, and at
the production width every medium shape is below the same-run MMA.

The work-split mapping does amortize the weight read: three columns cost
**1.52x** one column instead of 3x. But the result is ALU-bound (three FMA/
dequant chains per weight byte), and the 8x8 MMA remains faster per weight
byte even with 5/8 of the tile wasted. This is the same verdict as the earlier
multi-column FMA, now with the split-kernel lane map rather than a different
mapping.

**Gate: failed. The dense band is closed on this artifact.** The measured
headroom that remains is in the weight bytes themselves (a lower-bit dense
artifact, dNLL/KL-gated) or an address-only MMA tile/repack, not in this
access pattern.

## Member-axis attention and index price

GLM's today-shipped member-axis selected attention (+2.4% bitwise, B=4) and
index-chain batching (+0.8% bitwise, B=4) are the cross-family precedent.
V4.1's per-sequence flash partial/combine and index chain already sit in the
same causal wave as their siblings, so the relevant question is only what
merging the dispatches buys beyond the overlap already realized.

- **Flash partial:** 41 waves, 1.67 ms; each wave already contains the three
  per-sequence partial dispatches and lasts ~0.041 ms. Member-axis batching
  removes two dispatches per layer (~1.3-1.5 us each) but not the tensor work.
  Ceiling ~0.10-0.15 ms.
- **Combine:** 43 waves, 0.47 ms; each ~0.011 ms with three dispatches.
  Ceiling ~0.05-0.10 ms.
- **Index:** the whole chain is 0.52 ms (32K) / 0.74 ms (48K) exposed. Index
  layers are 0.55-0.57 ms total, the same as non-index layers, and index waves
  share RoPE/flash waves. Member-axis batching removes dispatch overhead, not
  the scoring/sort work; the GLM +0.8% anchor is ~0.22 ms at B=4, so expect
  ~0.1-0.2 ms at B=3 32K and maybe 0.2-0.3 ms at 48K.

Combined attention + index member-axis price: **~0.2-0.4 ms at 32K, ~0.3-0.5
ms at 48K**. That is below the 0.5 ms bar for either change individually and
only marginal for the pair.

## Recommendation

1. **Do not build the split-cols dense path.** It is bitwise and it amortizes
   the read, but it is 0.50-0.75x the MMA at B=3 and misses the 800 GB/s gate.
   Reopen dense only for a lower-byte artifact or an address-only tile repack,
   not another FMA mapping.
2. **Do not build attention member-axis now.** The partial and combine are
   already 3-wide in one causal wave; the ceiling is ~0.1-0.2 ms.
3. **Index-chain member-axis is the only member-axis lever with a measured
   exposed cost** (0.52/0.74 ms). If a small bitwise win is wanted, batch the
   index chain first and re-price at 48K; expect ~0.1-0.3 ms, not the full
   hold.
4. The causal re-pricing moves the ranked levers to: dense **6.9** exposed
   (closed on this artifact), routed **7.7** hold / **1.2** exposed compute,
   core **2.5**, posts **0.8**, index **0.5-0.7**, staging **0.5**. Perfect
   scheduling still has only a ~1-2 ms ceiling; the remaining large move is
   fewer dense weight bytes.

## Evidence

- Raw M5 logs: `~/qwen-runs/2026-11-11-v41-price/` on `metagross`
  (`bounds-causal-32k-final.jsonl`, `bounds-causal-48k.jsonl`,
  `graph-causal-32k.jsonl`, `dense-split-cols-census.jsonl`, stderr files).
- Commands: `v41_batch_depth <model> {32768,49152} 3 32 bounds`,
  `v41_batch_depth <model> 32768 3 1 graph`,
  `dense_band_census --cols 1,2,3,4 --reps 12 --target-mib 384
  --shapes q_b,output_a,output_b,shared_gate_up,shared_down`.
- Branch `perf/v41-causal-price`; no merge. M5 worktree removed.
