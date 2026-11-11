# 2026-11-11 — Serve chunk price: the append floor is per forward

Status: probe + decision. No production scheduling change (the only code change
is the cancelled-turn prefill accounting fix). GLM-5.3-Flash UD-IQ3_XXS on the
M5 Ultra, 48K depth, `--profile iter`, real three-agent corpus
(`~/qwen-runs/2026-11-07-serve-agents/code.txt`). Thermal pressure 0–1, host
load 10–58, GPU clocks unavailable unprivileged (`sudo -n powermetrics` needs a
password; `ioreg` exposes no clock).

Probe: `crates/qwen-llm/examples/serve_chunk_probe.rs` (`perf-ablations`).
Raw: `metagross:~/qwen-runs/2026-11-11-serve-chunk/{probe-idle.jsonl,
stage-loaded-65536.jsonl,stage-idle-196608.jsonl,edges.jsonl,edges2.jsonl,
stage-idle15.jsonl,stage-idle60.jsonl}`. Lease waits 0 s; every hold < 12 min.

## 1. The steady-state append is GPU work, not host bookkeeping

At 48K, warm medians over five restores of the same 48K snapshot. The three
buckets are an explicit `prepare_prefill`, the whole `prefill_packed`, and
`prepare_decode`.

| new tokens | prepare ms | kernel ms | finish ms | total ms | profiled GPU ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 20.590 | 0.000 | 20.591 | 21.69 |
| 32 | 0.000 | 301.191 | 0.000 | 301.192 | 300.76 |
| 85 | 0.000 | 474.197 | 0.000 | 474.197 | 475.12 |
| 342 | 0.000 | 579.018 | 0.000 | 579.019 | 580.85 |

- Prepare/admission and finish/release are **< 1 µs**. `released_bytes = 0` on
  every run: independent batching is supported, so `prepare_decode` never
  releases the prompt workspace.
- The GPU command is 99.9% of the kernel wall (`wall − gpu` is 3.6–6.3 ms).
- These match the *later* appends in the standing serve phases line to within
  1–2%: serve 32→304.6 ms, 45→373.7, 67→468.4 vs probe 32→301, 85→474.
  The kernel is not the source of the 1.5 s first-turn outlier.

## 2. The first append is a cold-idle cost, not a kernel or pipeline cost

The first 32-token append after the 48K prefill, under four host/session
variants (all otherwise identical):

| variant | first 32 ms | later 32 ms |
|---|---:|---:|
| hot, idle host | 315.7 | 301.2 |
| hot, 48 CPU-load threads | 312.8 | 312.3 |
| capacity 196608 + 8 decode steps before append | 312.0 | 305.0 |
| 15 s idle before the first append | **1024.9** | 306.0 |
| 60 s idle before the first append | **995.7** | 304.9 |

- Host load, capacity (and therefore workspace policy), post-decode state, and
  first-touch of the append path are ruled out: the first append is 4.8%
  above warm when the GPU is hot.
- An idle gap alone reproduces the penalty, in the GPU kernel, at both 15 s and
  60 s. This is cold-GPU first-touch/residency (no-copy weights unwired after
  ~2 s idle; the serve keep-alive pulses only within its window), with a clock
  ramp possible but not separable without privileged sampling.
- Serve confirms the shape: the first 32-token turn is **1892.4 ms**
  (`reused_tokens=48360`, `reuse_source=live`, `restore_ms=0`), then later
  32-token turns are 304.6 ms. Pipeline compile cannot explain it (it is
  global; the probe's fresh session is hot after the 48K prefill), nor can
  thermal pressure (0–1) or a workspace re-admission (`released_bytes=0`).

**Interpretation:** the ~1.5 s first-append term is a per-idle-period cold
start, not steady-state prefill. In a 30 s window that begins after the
harness's 15 s ramp it is counted as prefill for each agent's first turn; in a
busy loop (turn gaps < the residency window) it does not recur. It should be
fixed by retention/warming or excluded from the steady budget, not by
scheduling.

## 3. The chunk-cost sweep: the floor comes back every chunk

Single forward at 48K, warm median (edges2 run). Widths are not a single line:

| rows | ms | regime |
|---:|---:|---|
| 1 / 2 / 4 / 8 | 22.0 / 32.5 / 48.8 / 120.5 | near-decode |
| 16 / 24 / 32 / 48 / 64 / 80 / 96 / 112 | 260.9 / 299.5 / 303.2 / 368.1 / 415.7 / 463.2 / 498.5 / 552.2 | slow: ≈212 + 3.0 ms/row |
| **128** | **319.6** | fast island |
| 160 / 192 / 224 / 256 / 320 / 384 / 448 | 378.7 / 418.7 / 467.3 / 462.6 / 537.8 / 582.5 / 658.8 | ≈224 + 0.97 ms/row |
| **512** | **696.7** | fast island, knife-edge |
| 640 / 768 / 896 | 1436.7 / 1652.2 / 1868.4 | bad: ≈362 + 1.68 ms/row |
| **1024** | **1121.2** | fast island |
| 1280 / 1536 / 1792 / 2048 | 1347.8 / 1585.0 / 1824.9 / 2054.4 | ≈204 + 0.90 ms/row |
| 1025 / 1032 | 1134.9 / 1146.5 | 1024 tolerates +1..+32 |

Fit over the fast islands (128/1024/1280…2048): **≈204 ms + 0.90 ms/row**.
The ~200 ms intercept is the per-forward fixed cost. The most consistent
mechanism is the routed-expert weight stream: 1–8 rows route to a small expert
subset and cost near a decode round, while 16+ rows cover most of the expert
bank. Whatever the split between expert streaming and batched sparse
attention, it is GPU work that every forward pays again.

Splitting the same 1509-row append:

| chunk | calls | total ms | vs one call (~1676 ms) |
|---:|---:|---:|---:|
| 1024 | 2 | 1809.5 | +8% |
| 512 | 3 | 2086.2 | +24% |
| 256 | 6 | 2758.9 | +65% |
| 128 | 12 | 4188.0 | +150% |
| 64 | 24 | 10255.1 | +512% |
| 32 | 48 | 14672.6 | +775% |

So a 512-row chunk costs ≈ floor + 512×per_row **per chunk**: the floor is
repaid, not amortized. There is no chunk size that gives a 50–150 ms stall;
16 rows already costs 261 ms, and 128 rows (the cheapest useful island) costs
320 ms.

Marginal rows inside one forward:

| base | +1 / +2 / +3 rows |
|---:|---|
| 1024 | +12.4 / +12.4 / +14.4 ms (1.1–1.3%) |
| 128 | +3 rows: +11.3 ms (3.6%) |
| 512 | +1 row: **+560 ms** (513 falls into the 513–1023 bad regime) |

1024 is the only split that both halves the current 2048-row stall and
tolerates the extra decode rows. 512 must not be made ragged.

## 4. Mixed forward feasibility (GLM)

The prefill chunk is single-sequence by construction
(`glm5_next_metal/packed.rs`):

- `encode_mla_rows` scatters every row's latent KV at `self.position * kv`
  and runs `encode_latent_attention(..., self.position, dense, ...)` with one
  base position; `encode_indexer_append_rows` uses `self.position`; the
  sparse visibility is `visible[r] = self.position + r + 1`.
- `encode_kda_rows` runs one sequence's recurrent conv/state over the chunk.
- `step_packed_with_row_output` computes one sparse selector status per
  `[MLA block][chunk row]`.

The independent decode batch (`glm5_next_metal/batch.rs::encode_independent_batch`)
already has the per-sequence machinery for one token per member: `seqs: &[&Self]`,
per-member sparse visibility (`encode_glm_sparse_shared` with `visibles`),
per-member MLA/KDA (`encode_batch_mla`, `encode_batch_kda`), and a shared
row-independent FFN/MoE grid on the first member's packed scratch.

To merge C prefill rows with D decode rows in one forward:

1. A merged row grid with a row→(sequence, absolute position) map.
2. Attention/KDA/conv/MLA/KV writes and sparse selection routed per row to its
   sequence (extend the per-member batch path from one row to `r_i` rows).
3. Keep the prefill rows' attention on the exact fast geometry (128/1024/2048)
   and run the D decode rows as a separate attention group in the same command
   buffer. A ragged 513–1023 prefill attention group costs +560 ms to +900 ms.
4. The row-independent dense/MoE/head can span all C+D rows; that is where the
   shared weight stream is won.

Exactness: changing the prefill rows' partition changes the F32 schedule for
appends above the quantum. Gate any merged path on bitwise output.

Price: 3 extra rows cost ~12–14 ms at 1024 (1.1–1.3%) and ~11 ms at 128
(3.6%). Cross-sequence attention needs the per-row bases, but the marginal
compute is the same order.

## 5. Partial-call accounting

`serve prefill:` records are now emitted at prefill completion and at both
cancellation points (mid-prefill chunk, and the finish tick), carrying the
same `prefill_ms` the phases line would have reported. `serve_realistic.py`
parses them, counts only records whose turn never completed (matched by prompt
length and near-first-token time to avoid double counting), overlaps them with
the window, and reports `partial_prefill_s` / `partial_prefill_calls`
(included in `prefill_s`). Unit-tested; no scheduling or arithmetic change.

## 6. Decision

**The fine-grained-chunk gate fails.** A 32-row chunk costs 303 ms and a
128-row chunk 320 ms; the 512-row chunk costs 697 ms and becomes 1258 ms with
one extra row. Chunking re-pays the ~200 ms per-forward floor, so a
1509-token append in 128-row chunks costs 4.19 s instead of 1.68 s (+150%),
and a 32-row target costs 14.7 s (+775%). The 50–150 ms target is not
reachable by chunking this kernel.

**Recommendation: coarse chunking of only the large appends.** While another
decoder is pending, cap the prefill call at 1024 rows (today it is the
session's selected width, up to 2048); keep 2048 when the owner is otherwise
idle; keep the decode budget between chunks. Optionally merge the pending
decode rows into the 1024 forward (14 ms) after the per-sequence attention
path exists. Use exact 1024 (or 128) widths; never make 512 ragged.

Expected, GLM:

- Delivered: **~75–77 tok/s** vs 76.3–79.4 today (−3% to +1%). Large-append
  prefill grows ~8–11% (1024 + remainder; the remainder should be chosen to
  avoid the 513–1023 regime), and the steady prefill share (excluding the
  cold-start) rises from ~15–19% to ~17–21%. Merged rows recover <1%.
- Gap max: **1.76–2.8 s → ~1.1–1.2 s** (one 1024 chunk).
- Gap p99: large-append stalls move from ~0.5 s to ~1.1 s; small-append p99
  (~30–55 ms) is unchanged. Worst case halves, but p99 can worsen when
  >1024-token appends are frequent. If p99 is the priority, 128-row chunks
  give ~0.32 s but cost ~+150% prefill on large appends — reject.

V4.1: **do not apply the coarse split to its warm appends.** Its appends
(173–1230 tokens) are already one Full call under the 4096 cap; a 1024 split
adds a floor per call. Its lever is the live-CED threshold
(`ids.len() − matched ≥ 2 × 4096` today), which would let warm appends yield
at chunk boundaries without changing the row partition; verify bitwise.

The cheapest real throughput/responsiveness win is outside the three paths:
remove the cold-idle first-append penalty (retention/warm pulse, or exclude
the harness ramp from the steady standing). It is ~1.5 s of TTFT and ~10% of
delivered tokens in a window that starts after idle, at no scheduling cost.

Do not build fine-grained chunking. Do not chase "removing the floor" as host
bookkeeping: prepare and finish are < 1 µs and a 48-thread host load adds
< 10 ms; the floor is GPU expert work.
