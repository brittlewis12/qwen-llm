# 2026-11-11 — GLM small-append price: the floor is the routed-expert union, and the middle path misses 180 ms

Status: probe + decision. **No production change and no middle path built.**
GLM-5.3-Flash UD-IQ3_XXS on the M5 Ultra, 48K depth, `--profile iter`, real
three-agent corpus (`~/qwen-runs/2026-11-07-serve-agents/code.txt`, 150,243
tokens). Thermal pressure 0 on every run; lease held < 15 min per arm. GPU
clocks remain unavailable unprivileged (same limitation as the chunk-price
probe).

Probes:

- `crates/qwen-llm/examples/glm_small_append_price.rs` (`perf-ablations`):
  ordinary wall, sampled stage spans, and a per-dispatch GPU census for packed
  prefill widths 1..128, plus the independent decode batch at B=1..4.
- `scripts/bench/serve_idle_append.py`: real `qwen serve`, one 48K root,
  refresh → exact idle → 44-token append, default vs disabled idle residency.

Raw: `metagross:~/qwen-runs/2026-11-11-glm-small-append/{census.jsonl,
census.err,idle.log,idle-60/,idle-0/}`.

## 1. The floor is the routed-expert union, not the indexer or row padding

One 48K session (capacity 65,536, 2,048 prefill rows), snapshot before every
width, warm median of three ordinary `prefill_packed` calls, then the
stage-profiled and per-dispatch runs. Stage spans and the dispatch census agree
to a few percent. The dispatch profiler serializes each dispatch, so its sums
are attribution, not wall; ordinary wall is the timing number.

| rows | wall ms | routed experts ms | non-expert ms | attention-family ms |
|---:|---:|---:|---:|---:|
| 1 | 20.7 | 5.6 | 15.1 | 0.9 |
| 2 | 31.4 | 10.1 | 21.3 | 1.2 |
| 4 | 47.9 | 19.2 | 28.7 | 1.7 |
| 8 | 115.8 | 37.4 | 78.4 | 2.8 |
| 16 | 256.4 | 144.6 | 111.8 | 5.0 |
| 32 | **300.2** | **187.9** | 112.3 | 9.9 |
| 48 | 366.0 | 228.7 | 137.3 | 14.9 |
| 64 | 412.6 | 262.5 | 150.1 | 19.8 |
| 128 | **315.9** | 195.4 | 120.5 | ~20 (tensor names) |

At 32 rows the `routed_experts` stage is **188 of 300 ms (63%)**. The dominant
kernels are the generic grouped-slot MoE pair:

| kernel | calls | 16 rows | 32 rows | 64 rows |
|---|---:|---:|---:|---:|
| `moe_swiglu_clamped_iq2_s_f32_grouped_slots_generic` (gate/up) | 41 | 78.1 | **111.7** | 146.0 |
| `moe_down_iq3_s_f32_grouped_slots_m128_n16_retile` | 39 | 59.8 | **57.2** | 75.0 |

The cliff is a **kernel switch**, not the row count. At 8 rows the path uses
per-slot `expert_native_iq2_s/iq3_s` kernels (36.3 ms of experts total); at 16
rows it switches to the grouped-bucket kernels above (138 ms) and also lands on
a worse dense tile (`mat_mat_q6_K_mma8v_r2c2k64_f32`, 109 ms at 16 rows vs
47 ms at 32). At 128 rows the tensor up/down pair is used and the whole forward
drops below the 64-row cost, but the expert stage is still **195 ms**.

Ruled out:

- **Indexer over the full 48K context.** Attention-family kernels are 9.9 ms at
  32 rows (selected attention 9.4 + index scores 2.3 + selection/cache), ~3%
  of the forward. Not the floor.
- **Tiles padded to 64+ rows.** The 32-row dispatch grids are exact-row
  `mat_mat_q6_K_f32`/`mat_mat_q8_0_f32` calls, not a 64-row tensor tile. The
  non-expert cost is ~112 ms at both 16 and 32 rows and ~120 ms at 128 rows, so
  it is a weight-stream/launch floor, not row padding.
- **Host bookkeeping.** The earlier chunk probe already showed prepare/finish
  < 1 µs and GPU at 99.9% of the kernel wall; this run reproduces the same
  ordinary walls.

**Cause:** the 16–64-row regime visits nearly the whole routed-expert bank with
a grouped-bucket kernel that is inefficient at small buckets. The same union is
what any middle path must also read.

## 2. Middle-path price: the 4.4 ms/row marginal does not survive the expert union

The shipped independent batch is capped at 4 members
(`GLM5_NEXT_MAX_BATCH_SIZE = 4`; member-axis selection is 2..=4), so B=5..64
cannot be run through it. All four members were restored to the **same 48K
prefix** (so attention is 48K on every member), warmed, and timed with one
dispatch profile.

| B | wall ms | profile ms | marginal vs B−1 |
|---:|---:|---:|---:|
| 1 | 22.9 | 21.7 | — |
| 2 | 29.6 | 32.4 | +6.7 |
| 3 | 34.0 | 40.1 | +4.4 |
| 4 | 38.9 | 46.1 | +4.9 |

The wall marginal is 4.4–4.9 ms/extra row, matching the 4.4 ms anchor. But the
B=1 arm is serial decode and B≥2 is one token per member, so at B=4 the batch
touches 32 expert slots (≤32 distinct experts). A 32-row append touches
32 × top_k 8 = **256 slots (~170 distinct experts)**. The expert union grows
~8× while the measured marginal was taken at ≤32 slots.

Measured batch expert kernels at B=4 give a per-slot rate that does extrapolate:

| kernel | calls | B=4 ms | ms/slot |
|---|---:|---:|---:|
| `glm_batch_iq2_s_gate_up_k4096_tg64` | 41 | 8.13 | 0.254 |
| `glm_batch_iq3_s_down_k2048_tg64` | 39 | 4.54 | 0.142 |

At 256 slots that is **~101 ms of expert work** for a 32-row append, versus
188 ms in the grouped prefill path. The saving is real but bounded.

Component projection for 32 rows (measured pieces only):

| component | best measured / lower bound |
|---|---:|
| non-expert row-independent work | **112 ms** (prefill 32 rows); batch few-row extrapolation 118 ms; the cheapest measured non-expert is the 8-row path at 78 ms, whose 8-row tile would cost ~4× if tiled to 32 |
| routed experts, batch all-slots at 0.396 ms/slot × 256 | **~101 ms** |
| causal chunk attention (same sparse kernels) | **~15 ms** |
| total | **~213–234 ms** |

Even the most generous combination (8-row non-expert floor 78 ms + 101 ms
experts + 15 ms attention) is **194 ms**. The gate is **≤180 ms**, so it does
not clearly pass; the realistic build is **~210–235 ms (1.2–1.3× the gate)**.

Why the naive `22 + (r−1)×4.4 ≈ 159 ms` is wrong: that marginal was measured
while the expert union was small. The prefill census prices the same union
directly — routed experts grow 19.2 ms at 4 rows → 187.9 ms at 32 rows while
rows grow 8×, and the best batch per-slot model still pays ~101 ms at 32 rows.
The non-expert ~112 ms is a weight-stream/tile floor that the batch few-row
kernels do not beat (they cost ~3–4 ms/row at B=4).

Exactness was not exercised because no path was built. The proposed parts have
existing exactness evidence (batch FFN/MoE/head is bitwise to serial decode
under the Fast lineage; packed sparse attention is the same per-row kernel
family), so exactness would not be the blocker — the price is.

## 3. Keep-alive side check: the 60 s window covers ≤60 s idle; 90 s leaks

Real `qwen serve` (iter `qwen`, 65,536 context, default GLM backend), one 48K
root, then for each gap: re-send the root (reopens the residency window on a
completed compute request), sleep exactly the gap, send a 44-token append that
reuses 49,178 tokens. `QWEN_SERVE_IDLE_RESIDENCY_SECS` unset (60 s) vs `0`.

| keep-alive | idle | TTFT ms | prefill ms | reused tokens |
|---:|---:|---:|---:|---:|
| 60 s | 15 s | **417** | 407 | 49,178 |
| 60 s | 45 s | **416** | 407 | 49,178 |
| 60 s | 90 s | **1000** | 990 | 49,178 |
| 0 | 15 s | **981** | 973 | 49,178 |
| 0 | 45 s | **968** | 960 | 49,178 |
| 0 | 90 s | **979** | 971 | 49,178 |

Thermal 0 on every trial; decode is 130–131 ms in all six. The 60 s keep-alive
holds the weights wired for 15 s and 45 s gaps and lapses by 90 s; with
keep-alive 0 every ≥15 s gap pays the cold first-touch.

**Interpretation:** the ~1.5 s "first append" is partly a harness artifact
(an idle gap longer than the residency window, or the first append straight
after the root prefill). In a real serve process the cold penalty here is
**+0.56 s** (990 vs 407 ms prefill), and it does **not** leak through for
interactive gaps ≤60 s. The earlier 1.5–1.9 s figures included first-append
conditions that the default window normally covers. Do not change scheduling
for this; leave the default 60 s and, if a first turn after a >60 s pause
matters, warm/retain rather than re-price prefill.

## 4. Decision: do not build the middle path

- The 200 ms floor is the routed-expert union in the small-bucket grouped
  kernels (63% at 32 rows, 195 ms even on the 128-row tensor island). It is not
  the indexer and not 64-row padding.
- The middle path saves expert time (188 → ~101 ms at 32 rows) but pays a
  ~112 ms non-expert floor plus ~15 ms attention, landing at **~210–235 ms**
  against a **180 ms** gate. The B≤4 batch marginal cannot be extrapolated to
  32 rows because the expert union grows ~8×.
- The existing coarse-chunk / overlap recommendation
  (`docs/bench/2026-11-11-serve-chunk-price.md`) still stands for large
  appends; the keep-alive result says the cold-start term is a retention
  issue, not a scheduling one.
- The only remaining small-append lever not falsified here is a cheaper
  small-bucket expert kernel/tile. The 128-row island bounds its upside: even
  its tensor expert path costs ~195 ms at 128 rows and would still leave the
  32-row total above 180 ms. Do not build without a kernel-level reason to
  expect ≤70 ms combined expert work at 32 rows.

## 5. Follow-up: the expert-packet row threshold fails the 32-row gate

The prefill expert choice is `packed.rs::packet_half_experts_selected`: rows
1..=8 take the half-operand per-slot packet (`expert_native_iq2_s_f32` /
`expert_native_iq3_s_f32`), rows above take grouped buckets. The cutoff is now
`QWEN_GLM53_EXPERT_PACKET_MAX_ROWS` (default 8, hard cap 128); the packet's
validation bound was widened from 8 to 128 so a probe can A/B it. The probe's
`threshold` mode swaps only that cutoff on the same session, snapshot and
tokens, alternates arm order, and reports full-vocab KL / signed dNLL.

Warm median of three, 48K, same corpus. "grouped" is the shipped default; at
8 rows grouped was forced (cutoff 0) because the default already selects the
packet there.

| rows | grouped ms | packet ms | packet saving | KL(g‖p) | signed dNLL |
|---:|---:|---:|---:|---:|---:|
| 8 | 176.1 | 114.8 | 34.8% | 1.1e-4 | −0.0034 |
| 12 | 220.2 | 150.9 | 31.5% | 3.1e-3 | +0.0298 |
| 16 | 256.1 | 184.9 | 27.8% | 5.1e-4 | −0.0337 |
| 24 | 296.8 | 219.7 | 26.0% | 3.9e-2 | +0.1933 |
| **32** | **301.2** | **264.8** | **12.1%** | **2.4e-2** | **+0.3053** |
| 48 | 366.0 | 368.4 | −0.7% | 1.7e-5 | −0.0000 |
| 64 | 412.9 | 461.1 | −11.7% | 2.7e-3 | −0.0048 |
| 96 | 495.3 | 661.2 | −33.5% | 2.0e-3 | +0.0073 |

The selected-kernel profiles confirm the swap: at 32 rows grouped runs
`moe_swiglu...generic` 111.5 + `moe_down...m128_n16_retile` 57.1 ms, while the
packet runs `expert_native_iq2_s_f32` 92.2 + `expert_native_iq3_s_f32` 55.0 ms.
At 8 rows the same swap saves 34.8%, which is why the default already stops at
8; the packet's per-slot rate is ~0.575 ms/slot, not the batch kernel's 0.396.

**Gate result: fail.** The 32-row wall drops 12.1%, below the required 15%.
The real crossover is **≈46–48 rows** (32 still wins 12%, 48 is a wash, 64
loses 12%, 96 loses 33%). The 12–24-row window is a real 26–31% win, but it is
not the gate as written.

**Exactness is not bitwise** (different reduction), and the full-vocab
single-token KL at 24/32 rows (3.9e-2 / 2.4e-2) and signed dNLL (+0.193 /
+0.305) exceed the strict same-binary mean thresholds (1e-2 / 0.005). The
8-row forced-grouped control is the packet's inherent path delta (KL 1.1e-4,
dNLL −0.0034); top-1 agrees at every width. A speed-passing version would still
need a proper cohort, but the speed gate already fails.

**Decision: keep the default at 8; ship nothing.** The env knob and the widened
validation bound remain (default behavior unchanged) so the crossover can be
re-measured without a code edit. The only measured lever that could clear 32
rows is the batch tile (`glm_batch_iq2_s_gate_up_k4096_tg64` at ~0.396 ms/slot,
1.45× the prefill packet), which is a separate selection/validation packet, not
a threshold change.
