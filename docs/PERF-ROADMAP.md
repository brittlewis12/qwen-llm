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
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, and `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

## Force-Ranked Next Bets

### 1. Grouped Routed-Expert Execution For MoE Packed Prefill

Optimizes: MoE prompt processing, TTFT, `pp512` for 35B A3B and 122B A10B.

Why it is first now:

- MoE packed prefill stage 1 already landed and is a real win.
- The remaining structural waste is still token-by-token routed expert execution.
- Grouping tokens by selected expert is the next step that can unlock another
  meaningful chunk of MoE prompt throughput.
- Packed-MoE tail attribution now confirms this directly: routed FFN is ~47% of
  the A3B tail and ~57% of the 122B tail.

Expected payoff: medium-to-large additional MoE prefill gain; no direct
decode-token speedup.

Risks and constraints:

- Dynamic top-k routing can destroy batching if grouped naively.
- Must preserve router softmax/top-k tie semantics and shared expert gate.
- First packed-slot routed execution attempt failed the hard correctness gate;
  keep future attempts behind the existing tests until they are green.

Acceptance gates:

- Packed-MoE profiler identifies routed expert execution as the remaining large
  bucket.
- Grouped execution preserves logits / state equivalence on A3B production gates.

### 2. MoE Packed Prefill Hardening

Optimizes: correctness confidence and trustworthy iteration speed.

Why it is second:

- Codex-wrap found no major blocker, but the MoE hidden-capture path still lacks
  a dedicated packed-vs-oracle gate.
- We also want packed-MoE phase timing so the grouped-expert follow-up targets
  the actual remaining cost, not our guesses.

Acceptance gates:

- Add a MoE packed hidden-capture / chunk-boundary correctness gate.
- Add packed-MoE phase timing or equivalent attribution for prompt processing.

Status:

- The hidden-capture / chunk-boundary gate is now in place for 35B A3B.
- Packed-MoE tail attribution is now available and points at routed expert
  execution as the dominant bucket.

### 3. Dense Prompt Attribution Beyond Chunk Size

Optimizes: dense prompt processing and TTFT.

Why it changed:

- We found the chunk-size win and then removed the batched `alpha/beta` blind
  spot.
- Dense prompt throughput is now ~165.0 t/s on the same prompt where llama.cpp
  reports ~186.8 t/s, so the remaining gap is much smaller.
- Packed-prefill attribution now says the remaining dense gap is dominated by
  FFN mat-mat and the true GDN tail, not prompt-attention.

Acceptance gates:

- Keep dense prompt attribution current after each major packed-prefill change.
- Use it to choose between GDN-tail work and any broader mat-mat backend work.

Status:

- Dense packed-prefill phase profile (`P=321`) now shows:
  - `ffn`: ~53.2%
  - `gdn_front`: ~15.6%
  - `gdn_tail`: ~12.5%
  - `attn_decode`: ~7.8%
- Representative one-layer GDN tail split shows `step_decay` as the largest true
  tail sub-bucket once `out_proj` is excluded to `gdn_back`.
- A quick same-prompt llama.cpp tensor on/off falsification on M4 Max showed no
  meaningful prompt-rate delta, so a broad Metal tensor port is not the first
  assumption to chase.
- Packed `gdn_step_decay` did buy a real end-to-end reduction, and it is now the
  active dense prompt path.
- The next dense attack is therefore the broad FFN / projection mat-mat surface
  unless a sharper bandwidth indictment changes that conclusion.

### 4. Measure Command Overhead, Then Decide ICB / MTL4

Optimizes: MoE prompt processing, TTFT, `pp512` for 35B A3B and 122B A10B.

Why it matters:

- MoE decode is already strong; prompt replay is the likely structural gap.
- Current packed/no-tail helpers reject MoE.
- Real MoE prefill needs routing P tokens, grouping selected experts, mat-mat by
  expert, scatter/sum, and shared expert handling.

Expected payoff: potentially very large for MoE prefill; no direct decode-token
speedup.

Risks and constraints:

- Much harder than dense packed prefill.
- Naive per-token routed kernels do not amortize enough; expert grouping is the
  point.
- Must preserve router softmax/top-k tie semantics and shared expert gate.

Acceptance gates:

- MoE packed prefill matches sequential MoE loop on logits, KV, GDN state, and
  router decisions for small and production models.
- A3B and 122B `pp512` improve materially against sequential baseline.

### 5. Prefill Mat-Mat Quality

Optimizes: decode throughput, all models if host/driver overhead is real.

Why it is not higher:

- Current production path already uses one command buffer per token.
- Recent sweeps show CPU encode around ~0.15-0.25 ms/token, while GPU kernels
  dominate the measured wall.
- ICB/MTL4 has high plumbing cost and API churn risk.

Expected payoff: uncertain; could matter most for smaller dense models and MoE
tiny-kernel-heavy paths.

Risks and constraints:

- Easy to spend a lot of time for sub-5% if GPU time dominates.
- Dynamic scalar args and per-session buffers must stay correct.

Acceptance gates:

- Before implementation, produce production-shaped CPU encode / GPU / wall gap
  evidence that justifies the work.
- Prototype must improve total wall, not only encode time.

### 6. KV-Q8 / Quantized KV Cache For Long Context

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

### 7. Dense GDN / FFN Decode Surgery

Optimizes: decode throughput, all models if host/driver overhead is real.

Why it is not higher:

- Current production path already uses one command buffer per token.
- Recent sweeps show CPU encode around ~0.15-0.25 ms/token, while GPU kernels
  dominate the measured wall.
- ICB/MTL4 has high plumbing cost and API churn risk.

Expected payoff: uncertain; could matter most for smaller dense models and MoE
tiny-kernel-heavy paths.

Risks and constraints:

- Easy to spend a lot of time for sub-5% if GPU time dominates.
- Dynamic scalar args and per-session buffers must stay correct.

Acceptance gates:

- Before implementation, produce production-shaped CPU encode / GPU / wall gap
  evidence that justifies the work.
- Prototype must improve total wall, not only encode time.

### 8. Keep GPU Argmax / Full-Logits Decode Honest

Optimizes: prompt processing after packed prefill is wired everywhere.

Why it waits:

- First we need the packed prefill path in the main benchmark/product path.
- If `pp512` still trails llama.cpp afterward, mat-mat quality becomes the next
  obvious kernel-side target.

Expected payoff: medium to large for prefill, dependent on post-wiring pp data.

Risks and constraints:

- Kernel complexity and maintainability.
- Tune against end-to-end pp, not isolated synthetic mat-mat alone.

Acceptance gates:

- Same-harness pp comparison identifies mat-mat as the remaining bottleneck.
- Kernel changes improve dense and/or MoE packed prefill end-to-end.

### 9. Dense-Specific Long-Context Compression Revisit

Optimizes: future dense long-context decode if a better compression path exists.

Why it stays on the horizon:

- The negative result applies to the current Q8 main-body structure, not to all
  possible KV compression ideas forever.
- If future evidence suggests F16 is saturating memory harder at very high ctx,
  revisit with a better vectorized or block-shared design, not the current one.

Optimizes: benchmark integrity and small decode wins, especially on MoE.

Why it remains tracked:

- Current data says dense is neutral within noise while A3B / 122B gain
  modestly.
- Keep the `--full-logits-decode` A/B path and targeted regression tests so
  future decode or sampling work stays measurable and exact.

Expected payoff: small; mostly a measurement and architecture hygiene item.

Risks and constraints:

- Do not overstate the gain on dense before more repeated measurements.
- Preserve exact-token equivalence and oracle semantics.

Acceptance gates:

- Dense + MoE argmax wrapper tests stay green.
- CLI continues to support direct full-logits A/B when needed.

Optimizes: dense decode throughput.

Why it is deferred:

- The bucket is large (~29-31 ms on 27B), but prior FFN mega-fusion had weak
  end-to-end payoff and GDN recurrence semantics are correctness-sensitive.
- Narrow, measured surgery is better than a broad rewrite.

Expected payoff: possible low-to-mid single-digit ms if a concrete waste pocket
is identified; high uncertainty.

Risks and constraints:

- Do not lower GDN state precision or reorder recurrence semantics.
- Avoid another large FFN mega-fusion without a new profile showing it will pay.

Acceptance gates:

- Fresh intra-profile identifies a specific dominant subphase and mechanism.
- Correctness gates cover dense 27B and small F32 oracle models.

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

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```
