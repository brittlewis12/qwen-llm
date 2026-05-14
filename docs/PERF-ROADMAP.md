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

- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, and `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

## Force-Ranked Next Bets

### 1. Wire Dense Packed Prefill Into Normal No-Spec Decode

Optimizes: prompt processing, TTFT, `pp512`, total wall.

Why it is first:

- `qwen-bench decode` still sequentially replays prompt tokens through
  `single_token`.
- Packed dense prefill already exists in the DFlash path via
  `prefill_tokens_with_multi_hidden` and has measured ~2.7-3.4x prompt speedups.
- This is the most direct path to improving plain no-spec benchmark parity
  against llama.cpp prompt processing.

Expected payoff: large for prefill/TTFT; no direct decode-token speedup.

Risks and constraints:

- Dense-only until MoE semantics are implemented.
- Do not route MoE through dense FFN packed prefill.
- Preserve last-token tail semantics: final norm + lm_head + bootstrap logits.

Acceptance gates:

- Dense 27B packed-prefill logits/session state match sequential loop.
- `qwen-bench decode` uses packed prefill for dense by default.
- Compare `pp512`/TTFT against llama.cpp and old sequential path.

### 2. Add No-Spec GPU Argmax / Avoid Full Logits Readback

Optimizes: decode throughput, all greedy paths, readback overhead.

Why it is second:

- Plain decode still copies the full vocab logits row to CPU and argmaxes there.
- GPU argmax already exists and is used by MTP/DFlash paths.
- Low-risk, broad cleanup before heavier command-system work.

Expected payoff: small but cross-model; likely low single-digit percent, with
larger relative benefit on fast MoE decode.

Risks and constraints:

- Greedy argmax is not a full sampling implementation.
- Keep a full-logits/debug path for validation and API users that need logits.
- Tie-breaking must match existing CPU argmax tests.

Acceptance gates:

- Greedy generated token stream matches full-logits CPU argmax path.
- Readback bytes/token drop for no-spec greedy decode.
- Dense 27B, 35B A3B, and 122B A10B decode guardrails do not regress.

### 3. KV-Q8 / Quantized KV Cache For Long Context

Optimizes: long-context decode, DFlash usefulness at long context, memory.

Why it is third:

- After attention NWG64, attention remains the main context-scaling term.
- Dense 27B attention is ~14.3 ms at 16K and ~19.3 ms at 32K.
- Halving KV bandwidth should compound with attention v4 improvements and push
  speculative decode thresholds higher.

Expected payoff: several ms/token on dense 27B at 16K-32K; smaller but still
meaningful on 122B because attention is a smaller share.

Risks and constraints:

- Higher correctness and quality burden than prefill/readback work.
- Requires KV append conversion, new attention-v4 read path, and cache restore /
  prefix-cache compatibility.
- Needs quality/perplexity or long-context oracle checks, not just cosine smoke.

Acceptance gates:

- F16-KV and Q8-KV logits stay within agreed tolerance on dense and MoE shapes.
- Long-context ctx sweeps improve at 16K/32K without short-context regression.
- Prefix cache / restore tests cover the new KV dtype.

### 4. MoE Packed Routed-Expert Prefill

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

### 5. Measure Command Overhead, Then Decide ICB / MTL4

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

### 6. Prefill Mat-Mat Quality

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

### 7. Dense GDN / FFN Decode Surgery

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

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```
