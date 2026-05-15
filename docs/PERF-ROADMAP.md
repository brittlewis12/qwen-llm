# Performance Roadmap

Living cross-session performance plan for qwen-llm. Keep this file concise and
current: update the ranking when new measurements change expected value, risk,
or dependencies. Treat `docs/PLAN.md` as the architecture/history plan; this
file is the active optimization queue.

For append-only checkpoint history and exact current handoff state, see
`docs/PERF-LOG.md`.

## Current North Star

Beat llama.cpp across dense and MoE Qwen 3.5/3.6 workloads, ideally by more
than a little, without taking shortcuts that fail at long context or larger
model shapes.

Primary guardrails:

- Dense: `Qwen3.6-27B-Q4_K_M.gguf`
- MoE A3B: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- MoE A10B: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`
- Never run performance benchmarks in parallel.
- Always keep dense 27B in perf analysis while optimizing MoE.

## Latest Baseline Snapshot

M4 Max, release `qwen-bench`, sequential runs.

| Model | Context | Total ms/token | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| 27B dense | 4K | 42.80 | 23.4 | after dense group6 NWG64 |
| 27B dense | 16K | 46.11 | 21.7 | attention ~14.3 ms, GDN/FFN dominates |
| 27B dense | 32K | 51.25 | 19.5 | attention ~19.3 ms |
| 35B A3B | 4K | 14.65 | 68.2 | NWG64 guardrail held |
| 35B A3B | 16K | 17.12 | 58.4 | NWG64 guardrail held |
| 35B A3B | 32K | 20.45 | 48.9 | NWG64 guardrail held |
| 122B A10B | 4K | 31.42 | 31.8 | group16 tile4 + NWG64 |
| 122B A10B | 16K | 33.47 | 29.9 | group16 tile4 + NWG64 |
| 122B A10B | 32K | 35.14 | 28.5 | group16 tile4 + NWG64 |

Recent confirmed wins:

- Prompt-prefill scratch now skips the unused `[P, V]` logits pack on no-spec
  prompt paths, removing a large dead allocation from timed prefill and nudging
  dense 27B prompt throughput to ~173.3 t/s.
- Packed dense `gdn_step_decay` over prompt tokens is now live and materially
  improves dense prompt processing: the same-prompt 27B plateau rises from
  ~165.0 t/s to ~172.9 t/s while decode stays unchanged.
- Dense packed GDN `alpha/beta` batching was a major prompt win: same-prompt 27B
  prefill rose from ~141.9 t/s plateau to ~165.0 t/s plateau, with decode
  unchanged.
- Dense packed-prefill GDN-tail attribution is now in place and points at the
  true `step_decay` recurrence as the sharpest next dense tail target.
- Dense packed prefill chunk tuning was a major win: on the same 321-token 27B
  prompt, moving from inherited `P=16` to dense default `P=256` improved prompt
  throughput from ~77.9 t/s to ~140.7 t/s with decode unchanged; one-chunk
  saturation is ~141.9 t/s once `P >= 321`.
- Dense packed prefill is now the default no-spec path in `qwen-bench decode`
  for dense models; on a 321-token 27B prompt it improved prefill from 24.3 t/s
  to 78.1 t/s (~3.22x) with decode unchanged.
- MoE packed prefill stage 1 is now live in `qwen-bench decode`; on a 321-token
  prompt it improves prefill from 74.8 -> 91.3 t/s on 35B A3B and 32.1 -> 36.6
  t/s on 122B A10B, with decode essentially unchanged.
- MoE packed prefill chunk tuning also matters: tuned default `P=128` lifts the
  same prompt to ~95.3 t/s on 35B A3B and ~37.6 t/s on 122B A10B.
- No-spec GPU argmax decode path landed. Dense decode is neutral within noise;
  MoE decode improves modestly by avoiding full logits readback (~1.1-1.5% on
  A3B / 122B in current 64-token runs).
- Dense KV-Q8 prototype is currently a negative result on M4 for the existing
  v4 main-kernel structure; attention gets slower, not faster.
- Attention v4 now supports `group=4`, unlocking the small dense family
  (0.8B / 2B / 4B / 9B) as real long-context canaries instead of failing back to
  the old threadgroup-memory-limited attention path. The local 9B sweep now runs
  cleanly through 32K: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, and `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

## Force-Ranked Next Bets

### 1. Grouped Expert-Major MoE Packed Prefill

Optimizes: MoE prompt throughput, TTFT, `pp512` for 35B A3B and 122B A10B.

Why it stays first:

- MoE packed prefill stage 1 already landed and is a real win.
- Packed-MoE tail attribution still says routed FFN is the dominant remaining
  bucket: about `47%` of the A3B tail and `57%` of the 122B tail.
- The `ds4` close read reinforces the same conclusion: expert-major grouped work
  is the real MoE prefill play, not more per-token tail gardening.

Current design rule:

- Do not optimize "packed slots" first.
- Build an explicit ledger of `(token_idx, topk_rank, expert_id, weight)`.
- Gather by expert, run grouped routed FFN, then scatter/reduce back in fixed
  top-k order.
- Treat F16 routed-mid storage as a follow-on once the grouped F32 path is green.

Acceptance gates:

- A bench-only grouped path with frozen router outputs beats the current per-token
  routed FFN by at least `10-15%` on both A3B and 122B once gather/scatter cost
  is included.
- Production grouped path preserves logits, hidden capture, KV, and GDN state on
  the existing A3B hard gates before it becomes default.

### 2. Fast-Path Validation On One Production Path

Optimizes: iteration speed, correctness confidence, and willingness to take
larger performance swings without wasting days on silent divergence.

Why it moves up:

- The first grouped-MoE packed-slot attempt failed a hard correctness gate and
  had to be rolled back.
- `ds4`'s "one production path, diagnostics validate it" philosophy is worth
  stealing whole here.
- `cx` agrees this is performance leverage now, not paperwork.

Immediate focus:

- Keep candidate fast paths behind the production dispatch surface, not in a
  permanent fork.
- Add differential harnesses that can compare current vs candidate paths on the
  same routed decisions and fail at the first divergent layer/subphase.

Acceptance gates:

- Grouped-MoE experiments can localize divergence faster than full end-to-end
  cosine hunting.
- New packed fast paths clear logits + hidden + KV + GDN state gates before they
  are allowed to influence benchmark defaults.

### 3. Dense Prompt: Paired Same-Input Projection Fusion First

Optimizes: dense prompt throughput and TTFT on 27B, with fast iteration on 9B.

Why it refines the dense branch:

- Dense packed-prefill attribution now says prompt time is mostly FFN / projection
  work: `ffn ~53.2%`, `gdn_front ~15.6%`, `gdn_tail ~12.5%`, `attn_decode ~7.8%`.
- Exact-shape chained prompt mat-mat audit at `N=321` shows the current backend is
  only around `9-13 GiB/s` on the real 27B FFN / projection surfaces.
- The first easy large-`N` Q4 tile tweak was a loser, so another broad tile sweep
  is lower-signal than stealing `ds4`'s paired same-input fusion pattern.

Immediate target order:

1. Prototype fused packed `gate_proj + up_proj` on the same activation stream.
2. If it wins end-to-end, revisit other paired front-end projections where the
   same activation stream and layout line up cleanly.
3. Only then return to broader mat-mat backend gardening.

Acceptance gates:

- A fused dense `gate+up` prototype improves 27B packed prefill by at least
  `3-5%` end-to-end, not only in a microbench.
- Correctness matches the current FFN path on the existing dense packed-prefill
  gates.

### 4. Use 9B As The Fast Dense Long-Context Canary

Optimizes: experiment throughput and long-context turnaround while preserving the
27B guardrail.

Why it is now active:

- `group=4` attention v4 is now enabled, which unlocks the whole small dense line
  (0.8B / 2B / 4B / 9B) for realistic long-context decode.
- Local 9B now reaches 32K cleanly and is dramatically faster to iterate on than
  27B: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.

Usage rule:

- Use 9B for fast falsification of long-context attention / dense prompt ideas.
- Keep 27B in the analysis loop before claiming a real win.

Acceptance gates:

- Long-context experiments should be reproducible first on 9B, then confirmed on
  27B before the roadmap moves.

### 5. No-Copy GGUF Views And Residency Warmup

Optimizes: TTFT, cold-start variance, load-time memory pressure, and possible VM
object overhead.

Why it enters the roadmap now:

- The `ds4` close read makes this the strongest non-kernel structural crib.
- Current `qwen-llm` still copies weights tensor-by-tensor into fresh shared
  buffers; `ds4` instead wraps a few large GGUF-backed no-copy Metal views and
  warms residency up front.

Why it is not above the current prompt work:

- This is more likely a load / first-token / memory-cleanliness lever than the
  next steady-state prompt-throughput unlock.
- It still looks high-EV enough to prototype once the current MoE and dense
  prompt branches have a stable checkpoint.

Acceptance gates:

- Prototype shows materially better load time, first measured token stability, or
  memory / VM-object behavior without regressing steady-state throughput.

### 6. Frontier Benchmark Harness With Snapshot / Restore

Optimizes: benchmark quality and long-context decision speed.

Why it matters:

- `ds4-bench`'s frontier measurement style is a better mental model for prompt vs
  decode frontiers than one blended tokens/sec number.
- This would sharpen long-context dense/MoE comparisons and future speculative
  work without changing model semantics.

Acceptance gates:

- Add exact frontier prompt/decode probes that can restore from snapshots and
  measure a fixed local window.
- Use it to compare qwen vs llama phase-for-phase, not on blended totals.

### 7. Mid-Graph Flush / Overlap Before ICB / MTL4

Optimizes: decode and prompt wall only if CPU encode / driver gaps are real.

Why it stays behind the others:

- Recent data still says GPU time dominates and CPU encode is only about
  `0.15-0.25 ms/token` on the current hot paths.
- `ds4`'s mid-graph flush is a cheaper falsification path than full ICB / MTL4,
  but it still needs evidence of queue gaps first.

Acceptance gates:

- `xctrace` / Metal timeline shows enough idle gap or CPU encode overlap headroom
  to justify the plumbing.
- Prototype improves total wall, not only encode time.

### 8. Speculative Path: Attack Repeated Long-Context Attention Cost

Optimizes: DFlash / MTP viability at realistic context lengths.

Priority rule:

- Keep this behind the current dense/MoE prompt push.
- When returning to speculative work, do not lead with policy/schedule tuning;
  lead with kernel work that removes repeated long-context attention cost.

What the latest analysis says:

- Current MTP shape is structurally weak at long context because lazy verify does
  not amortize enough base work and MTP has its own growing KV attention cost.
- Current DFlash is made safe by adaptive verify `N`, but not fast, because the
  drafter and target still pay too much long-context attention work.

Highest-EV speculative kernel targets:

1. Target packed-verify multi-query attention so consecutive verify queries share
   KV reads.
2. DFlash two-range attention reading ctx-cache and noise directly, without
   `k_full` / `v_full` materialization.
3. Adaptive draft compute width, not only adaptive verify width.

### 9. KV-Q8 / Quantized KV Cache For Long Context

Optimizes: long-context decode, DFlash usefulness at long context, memory.

Current read:

- The first dense KV-Q8 prototype is a negative result on M4: despite exact
  append quantization and good output similarity, the current Q8 v4 main path
  makes attention slower than the tuned F16 path.
- Codex-wrap review says the likely cause is structural: scalar Q8 dequant/load
  overhead is overpowering stored-byte savings against an already-strong F16
  vectorized path.

Expected payoff: still potentially large in theory, but only if a materially
different reader structure wins. Do not spend more blind sweep time on the
current implementation.

Risks and constraints:

- Easy time sink.
- Needs a fundamentally better Q8 read path or a different compression format to
  be worth revisiting.

Acceptance gates:

- Revisit only with a concrete new kernel structure and a fast feedback plan.
- Cut again quickly if attention does not beat F16 at 32K or 64K.

### 10. Dense Decode Surgery And Small Decode Hygiene

Optimizes: dense decode throughput and measurement integrity.

Why it stays late:

- Dense decode is already competitive enough that prompt work dominates the
  scoreboard.
- Prior FFN mega-fusion had weak payoff and GDN recurrence semantics are
  correctness-sensitive.
- GPU argmax is already landed; dense gain is neutral within noise and MoE gain
  is modest but real.

Acceptance gates:

- Any decode surgery must be driven by a fresh dense phase profile identifying a
  specific waste pocket.
- Exact-token A/B paths (`--full-logits-decode`) and argmax regression tests stay
  green while decode work proceeds.

## Deprioritized For Now

- FFN mega-fusion as a first move: prior layer-major fusion produced too little
  gain for the complexity.
- Giant GDN recurrence rewrite: correctness risk is too high without a sharper
  measured target.
- Synthetic-only attention tuning: attention is much improved; further tuning
  should be driven by full phase/ctx sweeps.
- Vocab pruning or approximate lm_head shortcuts without exact-token gates.

## How To Update This File

When a session changes performance direction, update only the smallest relevant
section:

1. Add new measured baseline rows or replace stale ones.
2. Move ranked bets only when measurements change expected value or risk.
3. Record accepted wins in "Recent confirmed wins".
4. Record failed experiments in "Deprioritized" or in the relevant bet's risks.
5. Keep benchmark notes sequential and reproducible; do not mix parallel runs.
6. At each improved checkpoint, make the diff tell one optimization story:
   short `v0.xx:` subject, detailed wrapped body with measurement + validation.
7. When a win changes the broader lowering or bottleneck picture, update
   `docs/INFERENCE-GRAPH.md` alongside this file so the semantic map stays in
   sync with the engine and the current performance story.

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```
