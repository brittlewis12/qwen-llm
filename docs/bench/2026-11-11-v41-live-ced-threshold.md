# 2026-11-11 — V4.1 live-CED threshold: the lever is below the dependency halo

Status: measurement + decision. **No production change, nothing shipped.** The
live-CED threshold cannot make V4.1's real warm appends yield: 53 of 54 recorded
appends are shorter than the 2541-row decoder dependency halo, where the native
planner declines CED before any serve threshold is consulted. The one longer
append (20650) is already above the 8192 gate and already uses CED.

M5 Ultra, four-shard **DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XS + IQ4_NL
Engram**, all 384 experts/top-6, 48K prefix, `--profile iter`, real three-agent
corpus (`~/qwen-runs/2026-11-07-serve-agents/code.txt`), one canonical Metal
lease per run, all holds under 10 minutes, thermal pressure **0–1** (recorded
per arm; exactness is thermal-independent, timing is not). Probe:
`crates/qwen-llm/examples/v41_live_ced_probe.rs` (`perf-ablations`). Raw:
`metagross:~/qwen-runs/2026-11-11-v41-live-ced/{run1.jsonl,run2-state.jsonl,
run3-large.jsonl}`.

## 1. What live CED does, and where the thresholds are

A warm request enters `Settings::begin` with a restored live prefix (`matched`
tokens). `execution.rs:289-293` sets:

```
ced_live    = matched > 0 && Some(matched) != shared_cut
ced_enabled = live_ced && ids.len() - matched >= 2 * V41_PREFILL_ROWS   // 8192
```

`V41_PREFILL_ROWS` is `MAX_PREFILL_CHUNK_ROWS` = **4096**
(`family_options.rs:14`, `prefill.rs:16-18`), and this model's
`selected_prefill_rows` is 4096 (117 GB resident < the 240 GB full-chunk
limit), so Full chunks its append at 4096 rows. `live_ced` is default-on with
`QWEN_SERVE_V41_LIVE_CED=0` as the existing rollback.

`prefill_slice` then asks `ced_episode_end` (`execution.rs:863-879`) for an
endpoint. For a live request the minimum span is `1 * 4096`, and with a ladder
cut the endpoint rounds **down** to a 4096 multiple. Finally the native
planner, `CedPlan::from_prefix` (`ced_plan.rs:159-193`), refuses any episode
whose decoder reconstruction would reach into the committed prefix:

```
boundary_bound_rows = 1 + decoder_layers * (window - 1)
                    = 1 + 20 * 127 = 2541
if start > end - 2541  ->  "CED prefix overlaps decoder dependency halo"
```

So a live append needs **N ≥ 2541** before CED can even be planned. The serve
gate (8192) and episode minimum (4096) are layered on top of that hard native
bound. Inside an episode each `cold_prefill_episode_slice` encodes
`full_chunk_rows.min(max_rows)` rows (`ced/streaming.rs:135`); under fair load
the scheduler passes 4096, and the plan's `full_chunk_rows` is 4096, so an
episode yields once per 4096 encoder rows. Only the final slice runs
`finish_ced_stream`, i.e. the whole decoder reconstruction, as one
non-yieldable block.

## 2. The real append distribution never reaches the bound

54 recorded V4.1 primary live-reuse calls (the plan population used by the
serve harness), `new` tokens:

| statistic | rows |
|---|---:|
| n | 54 |
| min / p50 / mean / max | 15 / 427 / 934 / 20650 |
| ≥ 2541 (native halo) | **1 / 54 (1.9%)** |
| ≥ 4096 (episode minimum) | 1 / 54 |
| ≥ 8192 (serve gate) | 1 / 54 |

The single long call is 20650. Every other call (15–2260 rows) is below the
dependency halo, so:

- thresholds **2048, 1024, 512** are below the halo; lowering the serve gate to
  any of them changes **nothing** for any of the 54 calls;
- an append at 4096 rows would still form no episode under the ladder (the
  endpoint rounds down to the start); 4097–8191 would start an episode at a
  4096 gate and yield once at the CED→Full boundary, but that band contains
  **zero** recorded appends;
- the only call that can yield today, 20650, is already ≥ 8192 and already
  does.

This is why the threshold is 8192 rather than a numerical crossover: it is
`2 * 4096`, the "at least two full chunks" scheduling minimum, with the live
planner retaining one 4096 interior. It is conservative, but the halo makes
the region between 512 and 2541 unreachable in principle, not just unreached by
this sample.

## 3. What a forced episode would cost (native probe)

Each arm restores the same 48K snapshot and appends the corpus continuation.
`CED(k)` caps the encoder submission at `k` rows; the plan keeps
`full_chunk_rows = 4096`. Wall includes all encoder slices plus the endpoint
decoder reconstruction; exactness compares final logits against Full, and a
separate run at 2600/5000/8191 also compares one common post-append decode
step. Two timing runs (thermal 0/1) are shown where both exist; exactness is
identical across runs.

| new | Full ms | CED(4096) ms / slices | bitwise | CED(2048) ms / slices | bitwise | CED(512) ms / slices | bitwise |
|---:|---:|---:|:--:|---:|:--:|---:|:--:|
| 405 | 708 | declined | — | declined | — | declined | — |
| 1499 | 1448 | declined | — | declined | — | declined | — |
| 2260 | 2741 | declined | — | declined | — | declined | — |
| 2600 | 2470 / 3553 | 2422 / 2111, 1 | **yes** | 2195 / 2050, 2 | no | 2837, 6 | no |
| 3000 | 4359 | 2254, 1 | **yes** | 2261, 2 | no | 2800, 6 | no |
| 5000 | 5335 / 4713 | 3602 / 3116, 2 | **yes** | 4717 / 3596, 3 | no | 4089, 10 | no |
| 8191 | 10720 / 8432 | 5858 / 5836, 2 | **yes** | 4981 / 5681, 4 | no | 6139, 16 | no |
| 20650 | 25198 | 13127, 6 | **yes** | 14642, 11 | no | 20945, 41 | no |

- **Declines:** every append below 2541 is refused by the native planner at
  every slice width, before mutation. This is the measured form of §2.
- **Partition:** CED at Full's 4096 chunk width is **bitwise** in logits at
  all sizes, and in the post-append state at 2600/5000/8191. Any smaller slice
  width changes
  the encoder and decoder partitions and is **not** bitwise. At 2600 the mean
  KL is 5.8e-4, signed dNLL +7.8e-3, max |Δ| 0.93, and the state KL is
  2.8e-3; at 5000 state KL 4.1e-5, max |Δ| 3.9; at 8191 mean KL 4.0e-6, dNLL
  +2.8e-5, state KL 2.6e-6; at 20650 mean KL 1.2e-5, dNLL +2.6e-4, max |Δ|
  2.2. The sub-quantum arms are therefore failures under the exact-CED
  contract, independent of how small the numbers look.
- **TTFT:** at the matching width, CED is break-even near the 2541-row
  halo and 1.3–1.9× faster for larger appends (20650: 13.1 s vs 25.2 s), which
  is the existing 2026-11-08 live-CED result. Full is measured first in each
  size and the two timing runs differ by up to ~30% (thermal/load and the
  post-restore first touch), so use the direction and the exactness verdicts,
  not the small-N ratios. Sub-quantum slicing does not fix
  the worst stall: the final slice still runs the whole decoder reconstruction
  (1.2–3.7 s at 4096; 1.0–1.9 s at 512), and at 20650 the first slice pays the
  same streaming cost. The maximum slice stays above ~1 s in every arm.
- **No yield:** for 2600–4096 there is exactly one slice. The scheduler serves
  decoders only *after* a slice returns, so a one-slice episode yields nothing.
  The 3000/5000/8191 arms show 2–16 slices only because the probe ignores the
  serve gate and episode minimum; the serve path would not enter them at the
  candidate thresholds.

## 4. Decode service during a real append

Unchanged from the standing analysis: a warm append under 4096 rows is one
non-preemptible Full call, and the other agents get ~3% service inside it
(0.81 s→6/224 and 0.65 s→6/178 in the V4.1 arms). The candidate thresholds do
not alter that call at all because CED declines below 2541 rows; the
`QWEN_SERVE_V41_DECODE_PERCENT` credit is earned only after the call returns.
The 20650 call already yields 4 times at 4096 (5 slices + a 170-row tail) with
the current default.

## 5. Decision

**Do not lower the threshold; no A/B.** The ship gate asks for a three-agent
delivered-tok/s or gap-p99 improvement without material TTFT harm. Lowering to
2048/1024/512 changes no phase line for any of the 54 recorded appends (CED is
refused below 2541 rows), so the A/B has nothing to compare. Lowering to 4096
would add an exact, faster CED path (one 4096-row slice plus the Full tail, one
yield at that boundary) for 4097–8191-row appends, but that band contains zero
recorded calls; it is a reasonable future default if the workload grows
appends in that band, not a measured win today. Sub-quantum
slicing (512) is the only way to make a sub-4096 append yield, and it is not
bitwise and does not bound the tail below ~1 s.

The real V4.1 responsiveness lever for these appends is **not** CED: it is
either splitting the 4096-row Full forward at existing layer-scope boundaries
(which preserves the row partition and can be bitwise, unlike a row split), or
overlapping the endpoint decoder reconstruction with the final encoder chunk.
Both are engine changes outside this threshold turn.

Branch `perf/v41-live-ced-threshold`; probe only, not merged; local and M5
worktrees removed.
