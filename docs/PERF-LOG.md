# Performance Log

Append-only checkpoint log for qwen-llm performance work. Use this to answer
"where are we right now?" without re-running audits or reconstructing context
from chat history. Keep entries short, factual, and tied to measurements.

See also: `docs/PERF-ROADMAP.md` for the active force-ranked queue.

## 2026-05-14 — Attention Parity Push + Roadmap Reset

Status: improved checkpoint reached, not yet committed in git.

### Current Performance State

Sequential release `qwen-bench` runs on M4 Max:

| Model | Context | Total ms/token | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| 27B dense | 4K | 42.80 | 23.4 | dense group6 `NWG=64` |
| 27B dense | 16K | 46.11 | 21.7 | attention ~14.3 ms |
| 27B dense | 32K | 51.25 | 19.5 | attention ~19.3 ms |
| 35B A3B | 4K | 14.65 | 68.2 | MoE guardrail |
| 35B A3B | 16K | 17.12 | 58.4 | MoE guardrail |
| 35B A3B | 32K | 20.45 | 48.9 | MoE guardrail |
| 122B A10B | 4K | 31.42 | 31.8 | group16 tile4 + `NWG=64` |
| 122B A10B | 16K | 33.47 | 29.9 | group16 tile4 + `NWG=64` |
| 122B A10B | 32K | 35.14 | 28.5 | group16 tile4 + `NWG=64` |

### Confirmed Changes / Wins

- Promoted group16 tile4 default for long-context 122B A10B attention.
- Promoted dense group6 attention to `NWG=64` for `n_pos >= 4096`.
- Added attention A/B knobs: `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`,
  `QWEN_ATTN_V4_G16_TILE`.
- Expanded attention correctness to cover production-style `NWG=64` cases.
- Fixed stale attention/GDN intra-profilers so they time production paths.
- Added MoE intra-block profiler to split mixer, route, routed FFN, shared FFN,
  and residual pieces.
- Added `docs/PERF-ROADMAP.md` as the active force-ranked optimization queue.

### Key Measurement Deltas

Dense 27B long-context attention improved materially:

- 16K phase: old total ~50.42 ms / attention ~18.79 ms; `NWG=64` total
  ~46.19 ms / attention ~14.28 ms.
- 32K phase: old total ~60.32 ms / attention ~28.62 ms; `NWG=64` total
  ~51.12 ms / attention ~19.28 ms.

MoE `NWG=64` guardrails beat `NWG=32`:

- 35B A3B `NWG=32`: 4K 15.30 ms, 16K 20.20 ms, 32K 26.81 ms.
- 35B A3B default `NWG=64`: 4K 14.65 ms, 16K 17.12 ms, 32K 20.45 ms.
- 122B A10B `NWG=32`: 4K 31.91 ms, 16K 35.66 ms, 32K 40.54 ms.
- 122B A10B default `NWG=64`: 4K 31.42 ms, 16K 33.47 ms, 32K 35.14 ms.

Fresh subphase read:

- Dense 27B one GDN layer: ~0.646 ms; largest pieces are FFN gate/up/silu
  (~0.209 ms), FFN down (~0.162 ms), then GDN projections.
- 35B A3B one MoE block: ~0.299 ms; mixer prep dominates (~0.166 ms).
- 122B A10B one MoE block: ~0.594 ms; mixer prep dominates (~0.365 ms),
  shared FFN totals only ~0.074 ms.

### Validation Run

- `cargo fmt --all`
- `cargo check -p qwen-llm`
- `cargo test -p qwen-llm --no-run`
- `cargo test --release -p qwen-llm attn_v4_matches_naive_f16kv -- --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`

Notes:

- Existing warnings remain from upstream `llama-cpp-sys-2` and two ignored-test
  unused variables in `metal.rs`; no new functional failures observed.
- All performance runs above were sequential, not parallel.

### Current Force-Ranked Next Work

1. Wire dense packed prefill into normal no-spec `qwen-bench decode`.
2. Add no-spec GPU argmax / avoid full logits readback for greedy decode.
3. Prototype/design KV-Q8 for long-context attention.
4. Build MoE packed routed-expert prefill.
5. Measure command overhead before deciding on ICB / MTL4.
6. Tune prefill mat-mat quality after main packed prefill is wired.
7. Revisit dense GDN/FFN decode only with sharper subphase evidence.

### Workspace / Commit State

Current repo state is intentionally dirty and includes broad uncommitted work
from this performance arc. Do not assume all modified files belong to one small
change set.

Known currently uncommitted new docs from this checkpoint:

- `docs/PERF-ROADMAP.md`
- `docs/PERF-LOG.md`

Checkpoint commit style going forward:

```text
v0.xx: concise optimization headline

Explain the story in the body: what changed, why it matters, measured
impact, validation, risks, and next step. Keep the subject short enough
to scan cleanly; put numbers in the body unless they are essential to
the headline.
```

Before committing, inspect staged/unstaged diff carefully and include only the
intended checkpoint files/changes. Do not commit secrets or unrelated scratch.
Each commit should represent one measured optimization or one deliberate
workflow/documentation checkpoint.

### Next Handoff Instruction

Start by reading:

1. `docs/PERF-LOG.md`
2. `docs/PERF-ROADMAP.md`
3. `git status --short`

Then continue with the ranked item #1 unless fresh measurements or user
direction change priority.

## 2026-05-14 — Dense Packed Prefill Defaulted In No-Spec Decode

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `qwen-bench decode` now defaults dense models to the existing packed prefill
  path (`prefill_tokens_with_multi_hidden` with no hidden capture).
- Added `--sequential-prefill` for explicit A/B against the legacy prompt replay
  loop.
- Kept MoE models on sequential prefill until routed-expert packed prefill lands.
- Warmup now uses the selected prefill mode, so packed vs sequential A/B is not
  confounded by one-token-only sequential warmup.
- Added `--oracle-phase {prefill,final}` so oracle comparisons can target either
  the last prompt-token logits or the final decode-step logits.
- Tightened CLI behavior for empty prompts and zero-decode reporting.

### Measured Impact

27B dense, 321-token prompt, 4 decode tokens, release build, sequential runs:

- Packed prefill default: 4109.2 ms prefill = 12.80 ms/token = 78.1 t/s.
- Legacy sequential prefill: 13235.5 ms prefill = 41.23 ms/token = 24.3 t/s.
- Prefill speedup: ~3.22x.
- Decode unchanged within noise: ~41 ms/token on both paths.
- Generated text matched on the measured A/B run.

### Validation

- `cargo fmt --all`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`

Key correctness signal:

- `prefill_tokens_matches_single_token_loop_27b` passed again after wiring the
  CLI path; correctness test still shows ~2.78x standalone oracle-vs-packed
  speedup with cosine agreement on logits, hidden capture, GDN state, conv, and
  KV cache.

### Current Next Step

Roadmap item #1 is now no-spec GPU argmax / avoid full logits readback.

### Suggested Checkpoint Commit

```text
v0.77: packed dense prefill in no-spec decode
```

## 2026-05-14 — GPU Argmax Decode Path + A/B Harness

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added `single_token_argmax` / `single_token_argmax_profiled` to the Metal
  forward path for dense and MoE decode.
- `qwen-bench decode` now defaults to GPU-argmax decode and avoids full logits
  readback on greedy steps unless `--full-logits-decode` is set.
- Added `--full-logits-decode` for direct A/B measurement.
- Warmup now exercises the selected decode mode, including the argmax path.
- Added `single_token_argmax*` regression tests for dense and MoE short chains.

### Measured Impact

Current sequential A/B runs, 321-token prompt, 64 decode tokens:

- 27B dense:
  - full logits: 41.01 ms/token
  - gpu argmax: 41.09 ms/token
  - result: neutral within noise on dense; no convincing dense decode win yet.
- 35B A3B:
  - full logits: 13.90 ms/token
  - gpu argmax: 13.74 ms/token
  - result: ~1.1% decode win.
- 122B A10B:
  - full logits: 31.62 ms/token
  - gpu argmax: 31.30 ms/token
  - result: ~1.0% decode win.

Interpretation:

- GPU argmax is not a major dense ceiling breaker; dense benefit is neutral in
  current measurements.
- It is still a modest positive for MoE decode and reduces readback volume for
  greedy generation.

### Validation

- `cargo fmt --all`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo check -p qwen-llm`
- `cargo test -p qwen-llm --no-run`
- `cargo test --release -p qwen-llm metal_argmax_chain_matches_full_logits_dense -- --nocapture`
- `cargo test --release -p qwen-llm metal_argmax_chain_matches_full_logits_moe -- --nocapture`

Regression tests passed:

- dense 0.8B chain: cos = 1.000000, argmax path matches full-logits path.
- MoE A3B chain: cos = 1.000000, argmax path matches full-logits path.

### Current Next Step

Shift main pressure to KV-Q8 for long-context decode, while keeping the new
GPU-argmax path honest via `--full-logits-decode` A/B and the new regression
tests.

### Suggested Checkpoint Commit

```text
v0.78: gpu argmax decode path
```

## 2026-05-14 — KV-Q8 Dense Prototype: Negative Result

Status: negative result reached; do not spend more blind sweep time on the
current dense KV-Q8 main-kernel shape.

### What Changed

- Added an experimental dense-only `QWEN_KV_Q8=1` path:
  - KV cache allocates as Q8_0 instead of F16 for dense group6 / head_dim=256.
  - Fused K+V append quantizes with exact ggml Q8_0 rules.
  - Dense v4 attention main pass can read Q8 KV; reduce path unchanged.
  - Snapshot identity now records KV byte width, so prefix/snapshot state
    cannot silently alias F16 and Q8 layouts.
- Added two gates:
  - `scatter_kv_q8_matches_ref_quant` exact-byte test
  - `attn_v4_q8_kv_close_to_f16_kv` similarity gate (`cos=0.999994`)

### Measured Impact

Dense 27B, sequential runs:

Baseline (current best F16 KV path):

- 4K: 42.80 ms/token
- 16K: 46.11 ms/token
- 32K: 51.25 ms/token
- 64K: 61.26 ms/token

KV-Q8 prototype:

- 4K: 43.11 ms/token
- 16K: 47.58 ms/token
- 32K: 53.49 ms/token

Phase evidence says the loss is in the attention read path itself, not append:

- F16 KV attention bucket:
  - 16K: ~14.28 ms
  - 32K: ~19.28 ms
- Q8 KV attention bucket:
  - 16K: ~15.36 ms
  - 32K: ~21.55 ms

Codex-wrap hypothesis: our current Q8 reader loses to a very good F16 path on
Apple/M4 because the scalar Q8 dequant/load structure outweighs the stored-byte
savings, especially since v4 already amortizes KV reads across the dense GQA
group.

### Kill Test

- Tried a fast scale-broadcast style rescue on the Q8 v4 path.
- Result: worse, not better.
- Decision: cut bait on this Q8 main-body shape for now.

### Conclusion

- Keep the experimental Q8 path as evidence / future reference only if useful.
- Do NOT invest more time in broad KV-Q8 sweeps on this implementation.
- Pivot the main optimization pressure to MoE packed routed-expert prefill.

### Suggested Checkpoint Commit

```text
v0.79: KV-Q8 negative result on dense v4
```

## 2026-05-14 — MoE Packed Prefill, Stage 1

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `prefill_tokens_with_multi_hidden` now supports MoE target models.
- The new MoE packed path batches mixer prep and post-norm across the chunk,
  then runs routed FFN per token with the existing exact MoE route/apply path,
  scattering the updated `session.x` back into the packed chunk state.
- `qwen-bench decode` now defaults to packed prefill for MoE too; the legacy
  path remains available via `--sequential-prefill`.

### Validation

- New correctness gate:
  `prefill_tokens_matches_single_token_loop_35b_a3b_moe`
- Result:
  - `cos(final logits)=1.000000`
  - `GDN state cos_min=1.000000`
  - `KV K/V cos_min=1.000000`
  - small-gate speedup in correctness harness: `~2.69x`

Codex-wrap review:

- No obvious correctness blocker in the packed MoE shape.
- Main remaining test gap: MoE hidden-capture path still lacks a dedicated gate.
- Highest-EV next increment: grouped routed-expert execution, but add packed-MoE
  phase profiling first so the next step is aimed at the actual residual waste.

### Measured Impact

321-token prompt, 64 decode tokens, sequential runs:

- 35B A3B:
  - sequential prefill: 4290.1 ms = 74.8 t/s
  - packed prefill: 3514.4 ms = 91.3 t/s
  - improvement: `~22%` faster prefill
  - decode: essentially unchanged within noise
- 122B A10B:
  - sequential prefill: 10001.4 ms = 32.1 t/s
  - packed prefill: 8760.1 ms = 36.6 t/s
  - improvement: `~14%` faster prefill
  - decode: essentially unchanged within noise

### Current Next Step

Add packed-MoE phase timing and a MoE hidden-capture/chunk-boundary gate, then
attack grouped routed-expert execution for the routed branch.

### Suggested Checkpoint Commit

```text
v0.80: packed prefill for MoE no-spec decode
```

## 2026-05-14 — Dense Packed Prefill Chunk Tuning

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added `--prefill-chunk` to `qwen-bench decode` for packed prefill A/B work.
- `qwen-bench decode` now chooses a model-aware packed prefill chunk by default:
  - dense: `256`
  - MoE: `16`

This was driven by `cx` review: our inherited `P=16` came from DFlash, not from
any dense prompt-time evidence.

### Dense 27B Prompt Sweep (same 321-token prompt, `--tokens 0`)

- `P=8`: `35.0 t/s`
- `P=16`: `77.9 t/s`
- `P=32`: `103.6 t/s`
- `P=64`: `127.2 t/s`
- `P=128`: `136.1 t/s`
- `P=256`: `139.9 t/s`

Best measured point so far is `P=256`.

### Product-Shaped Dense 27B Result (same prompt, 64 decode tokens)

- Packed prefill with dense default `P=256`:
  - prefill: `2281.3 ms` = `140.7 t/s`
  - decode: `40.96 ms/token` = `24.4 t/s`

Comparison to earlier dense packed prefill default (`P=16`):

- old prefill: `~79.9 t/s`
- new prefill: `~140.7 t/s`
- improvement: `~1.76x` over the prior packed default

Comparison to same-prompt llama.cpp data the user supplied:

- llama.cpp prompt: `186.8 t/s`
- qwen-llm prompt after tuning: `140.7 t/s`

This does not close the full prompt gap, but it narrows it substantially.

### Conclusion

- Dense prompt processing was being artificially capped by a bad inherited chunk
  size, not just by deep kernel limitations.
- Dense prefill remains the biggest remaining dense gap vs llama.cpp, but the
  gap is now materially smaller.
- MoE stays on a conservative default chunk size until it gets its own sweep.

### Suggested Checkpoint Commit

```text
v0.81: tune dense packed prefill chunk size
```

### Follow-on State (same checkpoint arc)

- Added a MoE hidden-capture/chunk-boundary gate using a `P=1` packed oracle
  against `P=8` packed prefill; it passes with `cos(final logits)=1.0` and
  `hidden cos_min=1.0` on 35B A3B.
- Added packed-MoE tail attribution helpers.

Packed-MoE tail attribution (chunk_p=8):

- 35B A3B:
  - postnorm: `0.01 ms` (`0.4%`)
  - route+copy: `0.81 ms` (`20.5%`)
  - routed_ffn: `1.84 ms` (`47.0%`)
  - shared+resid+copy: `1.26 ms` (`32.1%`)
- 122B A10B:
  - postnorm: `0.02 ms` (`0.2%`)
  - route+copy: `0.91 ms` (`13.7%`)
  - routed_ffn: `3.78 ms` (`56.9%`)
  - shared+resid+copy: `1.94 ms` (`29.2%`)

Interpretation:

- Routed expert execution is clearly the largest remaining packed-MoE tail
  bucket on both A3B and 122B.
- Shared branch is still meaningful, but secondary.

Attempted next step:

- Tried a first packed-slot routed-expert execution path.
- Hard correctness gate failed immediately (`cos(final logits) ~ 0.97465`), so
  the active execution path was reverted to the last known-correct stage-1 MoE
  implementation.
- Result: keep the profiler/test scaffolding, but do not keep a broken fast path
  live in the tree.
