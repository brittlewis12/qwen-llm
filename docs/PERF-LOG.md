# Performance Log

Append-only checkpoint log for qwen-llm performance work. Use this to answer
"where are we right now?" without re-running audits or reconstructing context
from chat history. Keep entries short, factual, and tied to measurements.

See also: `docs/PERF-ROADMAP.md` for the active force-ranked queue.

## 2026-05-16 — Packed GDN Prep Over Prompt Tokens

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Used `cx` to adversarially review the post-v0.93 dense prompt plan before the
  next implementation step.
- Added `QWEN_PREFILL_GDN_SPLIT={skip_all,out_only,prep_out,prep_step_out}` on
  the real packed-prefill graph to split packed GDN body cost without falling
  back to a cloned profiler path.
- Replaced the old packed GDN prep train in `crates/qwen-llm/src/metal_dflash.rs`
  with a new packed kernel in `kernels/ssm_conv.metal` plus two in-place batched
  L2 norms:
  - old shape: `P` launches of `ssm_conv_silu`, two L2 norms, and three scatters
    before the packed recurrence
  - new shape: one `kernel_gdn_prep_packed_f32` over the chunk, then two batched
    in-place L2 norms

### Why This Was The Right Move

The split ladder on the repeated 320-token 27B prompt, before the rewrite:

- baseline: `1718.3 ms` wall, `1691.5 ms` GPU
- `skip_all`: `1425.6 ms` wall, `1415.5 ms` GPU
- `out_only`: `1518.3 ms` wall, `1506.6 ms` GPU
- `prep_out`: `1664.4 ms` wall, `1637.2 ms` GPU
- `prep_step_out`: `1708.4 ms` wall, `1679.7 ms` GPU

That implies the old packed GDN prep loop was the largest GDN sub-bucket:

- out-proj tail: `~92.7 ms` wall / `~91.1 ms` GPU
- prep loop: `~146.1 ms` wall / `~130.6 ms` GPU
- packed recurrence: `~44.0 ms` wall / `~42.5 ms` GPU

So the prep loop, not the recurrence kernel itself, was the sharpest next dense
prompt target.

### Measured Impact

Repeated 320-token quick-brown-fox prompt, 27B dense, packed prefill chunk 512,
release build, sequential runs:

- Before packed prep rewrite: `~1715-1720 ms`, `186.0-186.5 t/s`,
  `~1687-1693 ms` GPU
- After packed prep rewrite: `~1582-1588 ms`, `201.5-202.3 t/s`,
  `~1570-1576 ms` GPU

Net:

- about `7.8-8.2%` faster prompt prefill on the repeated 27B prompt
- dense same-prompt gap versus current `llama.cpp` (`212.44 t/s`) is now down to
  roughly five percent

Fresh current dense prompt read after the rewrite:

- baseline: `1581.8 ms` wall, `1570.6 ms` GPU, `202.3 t/s`
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1420.0 ms` wall, `1414.3 ms` GPU,
  `225.3 t/s`

That makes packed attention body the next largest non-FFN dense prompt bucket at
about `~162 ms` wall / `~156 ms` GPU.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`

Correctness stayed green:

- `cos(final logits)=1.000000`
- hidden / GDN state / conv / KV cache gates all remained effectively exact.

### Current Next Step

1. Packed attention body cleanup: batched consecutive-position RoPE and chunk-wise
   KV scatter / glue removal before considering a more invasive packed attention
   rewrite.
2. Then return to the remaining GDN out-proj / recurrence tail only if attention
   cleanup does not move the prompt enough.

## 2026-05-16 — Packed Prompt Differential Profiling + Batched GDN Gating

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `qwen-bench decode` now reports total packed-prefill GPU time via
  `prefill_tokens_with_multi_hidden_profiled`.
- Added dense prompt differential profiling flags on the real packed prefill
  graph:
  - `QWEN_PREFILL_NOOP_FFN=1`
  - `QWEN_PREFILL_NOOP_GDN_BODY=1`
  - `QWEN_PREFILL_NOOP_ATTN_BODY=1`
- Batched packed GDN `rmsnorm_gated` over the whole prompt chunk instead of one
  dispatch per token.

### Fresh Dense Prompt Read

Repeated 320-token quick-brown-fox prompt, 27B dense, packed prefill chunk 512,
release build, sequential runs:

- Latest packed prefill plateau after batched GDN gating: `~1718 ms` wall,
  `~1691 ms` GPU, `~186.2 t/s`.
- Fresh current `llama.cpp` prompt baseline on the same machine/model family:
  `212.44 t/s`.
- Prompt remains behind, but the gap is now about `~14%`, not the earlier
  `~16%` pre-win read.

### Production-Shape Differential Prompt Deltas

Representative no-op runs after the new batching change:

- `QWEN_PREFILL_NOOP_FFN=1`: `793.0 ms` wall, `766.2 ms` GPU.
- `QWEN_PREFILL_NOOP_GDN_BODY=1`: `1432.1 ms` wall, `1422.2 ms` GPU.
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1557.8 ms` wall, `1535.7 ms` GPU.

Against the new `~1718 ms` / `~1691 ms` baseline, that says the live packed
prompt graph is approximately:

- FFN: `~925 ms`
- GDN body: `~286 ms` wall, `~269 ms` GPU
- attention body: `~160 ms` wall, `~155 ms` GPU

Interpretation:

- Prompt is still overwhelmingly real GPU work (`~98.3%` GPU / wall), not outer
  orchestration.
- The old "FFN is the hidden prompt mystery" theory is now dead: direct
  production-shape FFN delta matches the broad bucket story, and isolated exact
  shape FFN mat-mat refs already match or beat exported llama.cpp prompt ops.
- The highest-EV remaining dense prompt work is now the non-FFN prompt path:
  first packed GDN body staging, then attention body cleanup.

### Measured Win

- Before batched packed `rmsnorm_gated`: `~1747.5 ms`, `183.1 t/s`,
  `~1717.1 ms` GPU.
- After batching it over the full chunk: `~1715-1720 ms`, `186.0-186.5 t/s`,
  `~1687-1693 ms` GPU.
- Net: about `1.7-1.9%` faster prompt prefill on the repeated 27B prompt.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`

Correctness stayed green:

- `cos(final logits)=1.000000`
- hidden / GDN state / conv / KV cache gates all remained effectively exact.

### Current Next Step

1. Packed GDN body cleanup: attack SSM-conv prep and Q/K norm + V-pack staging
   around the existing packed recurrence.
2. Packed attention body cleanup: batched consecutive-position RoPE and chunk-wise
   KV scatter / glue removal before considering a more invasive packed attention
   rewrite.

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
3. `docs/INFERENCE-GRAPH.md`
4. `git status --short`

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
- `P=321`: `141.8 t/s`
- `P=384`: `141.9 t/s`
- `P=512`: `141.7 t/s`

Dense prefill effectively saturates once the whole 321-token prompt fits in a
single packed chunk. We keep the dense default at `256` as a conservative
near-optimal point with materially smaller scratch than `512`.

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
- MoE chunk-size sweep is next; do not assume the dense result transfers.

### Suggested Checkpoint Commit

```text
v0.81: tune dense packed prefill chunk size
```

## 2026-05-15 — Batched Dense GDN Alpha/Beta Prompt Path

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added F32 packed mat-mat support for prompt-style `[N, H] x [H, n_out]` in the
  narrow form we need for small output widths.
- Dense packed prefill now batches GDN `beta_proj` and `alpha_proj` across the
  whole packed chunk instead of re-running them token-by-token.
- Added a batched GDN decay-chain kernel so `[N, n_v]` alpha activations can be
  turned into per-token decay values in one pass.

### Dense Prompt Result

Same repeated 321-token 27B prompt:

- before this change: prompt plateau ~`141.9 t/s`
- after this change: prompt plateau ~`165.0 t/s`

Product-shaped run (`64` decode tokens):

- packed prefill with dense default `P=512`: `1947.4 ms` = `164.8 t/s`
- decode unchanged: `41.04 ms/token` = `24.4 t/s`

This moves the same-prompt dense prompt gap versus llama.cpp from roughly
`186.8 / 141.9 = 1.32x` behind to about `186.8 / 165.0 = 1.13x` behind.

### Updated Dense Packed-Prefill Attribution (`P=321`)

- total: `2016.30 ms`
- `ffn`: `1008.40 ms` (`50.0%`)
- `gdn_front`: `291.92 ms` (`14.5%`)
- `gdn_alpha_beta`: `0.26 ms` (effectively gone)
- `gdn_tail`: `359.69 ms` (`17.8%`)
- `gdn_back`: `100.05 ms` (`5.0%`)
- `attn_front`: `72.89 ms` (`3.6%`)
- `attn_decode`: `148.82 ms` (`7.4%`)
- `attn_back`: `31.73 ms` (`1.6%`)

Interpretation:

- The old dense prompt bottleneck from `alpha/beta` projections is no longer
  relevant.
- Dense prompt time is now dominated by FFN mat-mat and the true GDN tail.

### Supporting Fast-Feedback Probe

Exact production-shape prompt probes at `N=321` still show mat-mat strongly
beating repeated mat-vec on the real 27B weights:

- `ffn_gate` Q4_K: `34.45 ms` vec321 vs `5.32 ms` mat-mat (`6.47x`)
- `ffn_up` Q4_K: `32.24 ms` vec321 vs `5.27 ms` mat-mat (`6.12x`)
- `ffn_down` Q6_K: `51.59 ms` vec321 vs `5.68 ms` mat-mat (`9.08x`)
- `attn_qkv` Q6_K: `27.26 ms` vec321 vs `3.21 ms` mat-mat (`8.50x`)

No-code falsification check from `cx` recommendation:

- llama.cpp same-prompt run with Metal tensor path forced on/off on M4 Max
  showed no meaningful delta (`~207.0` vs `~207.2 t/s` prompt in the
  one-token probe), so the Metal tensor path is not obviously the missing trick
  on this hardware.

### Current Next Step

Per `cx`, the sharpest next dense attack is now the GDN tail bucket, not a
broad mat-mat backend port. The best fast-feedback next step is to split the
`gdn_tail` bucket further and/or prototype a packed `gdn_step_decay` time-loop
kernel before attempting a larger rewrite.

### Suggested Checkpoint Commit

```text
v0.83: batch dense GDN alpha and beta prompt path
```

### Follow-on State (same checkpoint arc)

- Added a denser packed-prefill profiler for the real 27B prompt and a
  representative one-layer GDN tail subprofile.

Updated dense packed-prefill attribution (`P=321`):

- total: `2016.30 ms`
- `ffn`: `1008.40 ms` (`50.0%`)
- `gdn_front`: `291.92 ms` (`14.5%`)
- `gdn_tail`: `359.69 ms` (`17.8%`)
- `gdn_back`: `100.05 ms` (`5.0%`)
- `attn_decode`: `148.82 ms` (`7.4%`)

Representative one-layer GDN tail split over the same 321-token prompt:

- total: `42.63 ms`
- `conv`: `1.56 ms`
- `l2`: `2.10 ms`
- `step_decay`: `6.16 ms`
- `rmsnorm_gated`: `1.67 ms`
- `out_proj`: `31.13 ms`

Interpretation:

- In the full dense prompt profile, `out_proj` already belongs to `gdn_back`, so
  the true `gdn_tail` bucket is the sum of `conv + l2 + step_decay + rmsnorm`.
- Within that true tail, `step_decay` is the largest sub-bucket.
- `cx` review says the sharpest next checkpoint is a bounded packed
  `gdn_step_decay` time-loop falsification kernel over prompt tokens; if it does
  not buy roughly `80-100 ms` end-to-end, pivot back toward broader FFN/backend
  work.

### Suggested Checkpoint Commit

```text
v0.84: profile dense GDN tail prompt bucket
```

## 2026-05-15 — Packed Dense GDN Step Time-Loop

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added an experimental packed `gdn_step_decay` kernel that loops over prompt
  tokens inside the kernel while keeping each GDN state row resident across the
  whole packed chunk.
- Dense packed prefill now uses this packed step path by default; the old path is
  still available as a kill switch via `QWEN_DENSE_GDN_STEP_PACKED=0`.
- Updated the dense packed-prefill profiler so its phase numbers reflect the new
  active path.

### Validation

- `prefill_tokens_matches_single_token_loop_27b` passes on the default path:
  - `cos(final logits)=1.000000`
  - `hidden cos_min=0.999999`
  - `GDN state cos_min=0.999999`
  - `KV K/V cos_min=1.000000`

### Dense Prompt Result

Same repeated 321-token 27B prompt:

- before this change: prompt plateau ~`165.0 t/s`
- after this change: prompt plateau ~`172.9 t/s`

Product-shaped run (`64` decode tokens):

- packed prefill with dense default `P=512`: `1861.1 ms` = `172.5 t/s`
- decode unchanged: `41.12 ms/token` = `24.3 t/s`

This closes the same-prompt dense prompt gap vs the earlier llama.cpp reading
(`186.8 t/s`) to roughly `1.08x`.

### Updated Dense Packed-Prefill Attribution (`P=321`)

- total: `1896.62 ms`
- `ffn`: `1008.61 ms` (`53.2%`)
- `gdn_front`: `295.71 ms` (`15.6%`)
- `gdn_tail`: `236.87 ms` (`12.5%`)
- `gdn_back`: `100.12 ms` (`5.3%`)
- `attn_front`: `72.86 ms` (`3.8%`)
- `attn_decode`: `148.16 ms` (`7.8%`)
- `attn_back`: `31.73 ms` (`1.7%`)

Interpretation:

- Packed `gdn_step_decay` reduced the true dense `gdn_tail` bucket from roughly
  `359.69 ms` to `236.87 ms` in the same profiler.
- Dense prompt time is now even more dominated by the FFN mat-mat surface.

### Current Next Step

- The next dense lever is likely the broad FFN / projection mat-mat surface,
  unless a sharper bandwidth indictment says otherwise.
- MoE still wants grouped routed-expert execution as the next structural win.

### Suggested Checkpoint Commit

```text
v0.85: pack dense gdn step over prompt tokens
```

## 2026-05-15 — Drop Unused Prompt Logits Scratch

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added a lighter `MetalDFlashLayerMajorScratch::fresh_prefill` constructor that
  skips the huge `[P, V]` `final_logits_pack` allocation when the caller only
  needs `prefill_tokens_with_multi_hidden`.
- Switched packed prompt-prefill call sites in `qwen-bench decode` and related
  no-spec prompt paths to use the lighter scratch.

Rationale:

- With dense `P=512`, the old scratch shape allocated a very large unused
  `[P, V]` buffer during timed prompt prefill. That was both unnecessary memory
  pressure and unnecessary timed wall.

### Measured Impact

Same repeated 321-token prompt, product-shaped runs:

- Dense 27B (`P=512`, packed step path already on):
  - before: `172.5 t/s` prefill
  - after: `173.3 t/s` prefill
  - decode unchanged / slightly better within noise (`~24.5 t/s`)
- 35B A3B (`P=128` default): `95.1 t/s` prefill, no regression.
- 122B A10B (`P=128` default): `37.6 t/s` prefill, no regression.

Interpretation:

- This is a small but real cleanup checkpoint, not a giant algorithmic leap.
- It removes a bad allocation pattern from the hot prompt path and shaves a bit
  more prompt wall on the dense target while keeping the MoE path clean.

### Current Dense Prompt State

- Same-prompt dense 27B prompt throughput is now about `173.3 t/s`.
- That is very close to the earlier same-prompt llama.cpp reading of `186.8 t/s`.

### Suggested Checkpoint Commit

```text
v0.86: drop unused prompt logits scratch
```

## 2026-05-15 — Dense Prompt Mat-Mat Audit + Direction Check

Status: measurement checkpoint reached, no production fast path changed.

### What Changed

- Added an exact-shape chained prompt mat-mat audit for real 27B production
  surfaces at `N=321`.
- Tried a Q4 large-`N` (`NR1=64`) prompt mat-mat specialization and measured it.
- It got worse, so it was reverted immediately.

### Exact-Shape Prompt Mat-Mat Audit

Chained prompt-shape numbers (`N=321`, `64` chained dispatches) on real 27B
weights:

- `blk.0.ffn_gate.weight` Q4_K:
  - `~5.09 ms / dispatch`
  - `~9.2 GiB/s` weight throughput
- `blk.0.ffn_up.weight` Q4_K:
  - `~5.09 ms / dispatch`
  - `~9.2 GiB/s` weight throughput
- `blk.0.ffn_down.weight` Q6_K:
  - `~5.45 ms / dispatch`
  - `~12.5 GiB/s` weight throughput
- `blk.0.attn_qkv.weight` Q6_K:
  - `~3.03 ms / dispatch`
  - `~13.2 GiB/s` weight throughput

Interpretation:

- Prompt mat-mat is already the right algorithmic shape, but backend quality is
  still very low versus the hardware envelope and versus our decode mat-vecs.
- The broad dense FFN / projection mat-mat surface is still a legitimate next
  dense lever, but the first easy Q4 large-`N` specialization was not the win.

### Direction Check

- `cx` review says the highest-EV branch overall is still grouped routed-expert
  execution for MoE packed prefill.
- For dense, the next checkpoint should be chosen carefully: either a sharper
  backend-quality experiment or a more principled mat-mat rewrite, not another
  casual tile tweak.

### Suggested Checkpoint Commit

```text
v0.87: audit dense prompt mat-mat backend
```

### Follow-on State (same checkpoint arc)

- MoE packed prefill chunk sweep on the same 321-token prompt (`--tokens 0`):
  - 35B A3B: `P=8 85.9`, `16 87.0`, `32 89.8`, `64 92.1`, `128 94.9`,
    `256 95.0`, `321 95.2` t/s
  - 122B A10B: `P=8 35.8`, `16 36.2`, `32 37.2`, `64 37.7`, `128 37.7`,
    `256 37.8`, `321 37.7` t/s
- Decision: move the default MoE packed prefill chunk from `16` to `128`.
- Product-shaped check with new default (`64` decode tokens):
  - 35B A3B prefill: `95.3 t/s`
  - 122B A10B prefill: `37.6 t/s`

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

## 2026-05-15 — Group-4 Attention v4 And 9B Long-Context Canary

Status: enablement checkpoint reached; small dense family can now use the v4
long-context attention path.

### What Changed

- Added `attn_v4` F16 kernels for `group=4` in `kernels/attn_v4.metal`, including
  the reduce path.
- Wired host dispatch selection through `crates/qwen-llm/src/metal.rs`,
  `crates/qwen-llm/src/metal_forward.rs`, `crates/qwen-llm/src/metal_dflash.rs`,
  and the packed correctness plumbing so `group=4` shapes stop falling back to
  the old threadgroup-memory-limited `attn_decode_f16kv` path.
- Extended the v4-vs-naive correctness test to cover the small dense shape
  (`n_q=8`, `n_kv=2`, `group=4`).

### Validation

- `cargo check -p qwen-llm`
- `cargo test --release -p qwen-llm attn_v4_matches_naive_f16kv -- --nocapture`
  passes with exact-style agreement for the new `group=4` shape across
  `n_pos ∈ {1, 32, 64, 256, 1024, 4096}`, `NWG ∈ {1, 2, 4, 8, 16, 64}` where
  applicable, and `C ∈ {16, 32, 64, 128}`.

### 9B Long-Context Canary Result

Model: `/Users/tito/models/Qwen3.5-9B-Q4_K_M.gguf`

`qwen-bench ctx-sweep --checkpoints 1,4096,8192,16384,32768 --window 2`:

- `1`: `14.75 ms` / `67.8 t/s`
- `4096`: `15.63 ms` / `64.0 t/s`
- `8192`: `15.95 ms` / `62.7 t/s`
- `16384`: `16.82 ms` / `59.5 t/s`
- `32768`: `18.57 ms` / `53.8 t/s`

Interpretation:

- The 9B no longer hits the old `~7K` long-context cliff.
- Group `4` is the one immediate unlock for the whole small dense family
  (`0.8B / 2B / 4B / 9B`), so we now have a much faster dense long-context
  canary without giving up the 27B guardrail.

### Direction Check

- The new `cx` review on the `ds4` close read reinforces the current ordering:
  grouped expert-major MoE prefill remains first, fast-path validation moves up
  beside it, and the dense branch should try paired same-input projection fusion
  before broader mat-mat gardening.
- `ds4` also surfaces two later but promising structural ideas to keep on deck:
  no-copy GGUF-backed Metal views with residency warmup, and a frontier
  snapshot/restore benchmark harness.

## 2026-05-15 — MoE Follow-On Falsifications + 27B 4K Trace Harness

Status: no new performance checkpoint; several important branches were cleanly
falsified and the command-model picture is now sharper.

### MoE Follow-On Results

Grouped expert-major routed FFN, implemented as CPU ledger + gather/scatter +
generic per-expert mat-mat, was semantically correct but strongly negative:

- 35B A3B:
  - `chunk=8`: `0.50 ms -> 8.64 ms` (`0.06x`)
  - `chunk=128`: `7.42 ms -> 20.13 ms` (`0.37x`)
- 122B A10B:
  - `chunk=8`: `0.99 ms -> 8.93 ms` (`0.11x`)
  - `chunk=128`: `15.19 ms -> 26.59 ms` (`0.57x`)

Interpretation:

- Generic grouped GEMM is the wrong organization here.
- Average expert groups are too small, and gather/scatter overhead dominates.

Two more MoE follow-ons also failed to earn a checkpoint:

- Batched shared-expert stage-2 rewrite: correct, but slower end-to-end on A3B.
- F16 routed-inner traffic reduction on the live Q5-down path: correct, but a
  wash-to-slight loser end-to-end.

Current MoE read after the falsifications:

- Stage-1 packed MoE prefill remains the live baseline.
- Further MoE upside likely needs either smaller token-major cleanup or a truly
  custom persistent grouped kernel, not another generic grouped experiment.

### Dense Prompt Follow-On Results

Dense paired prompt fusion was also pushed to a real go/no-go point and failed
to clear the bar:

- Shared-X paired `gate+up` Q4 prompt kernel at exact-shape 27B `N=321`:
  - `1.04x` microbench speedup over two separate mat-mats (`64` FFN layers)
  - exact-correct numerically
- Narrower `NR1=16` paired kernel: worse (`0.86x`)
- Forcing single Q4 prompt mat-mat itself to `NR1=16` at `N=321` also regressed:
  - `5.09 ms -> 6.37 ms` per dispatch

Interpretation:

- The easy paired-fusion / narrower-tile branch is mostly tapped out.
- Dense prompt should pivot toward less-staged Q4 mat-mat traversal/locality,
  not another fusion-first attempt.

### New Trace Tooling

- Added `qwen-bench decode-window`, an attach-friendly helper that warms to a
  target context, writes a ready file, waits for a go file, then runs a fixed
  decode window.
- Added `scripts/profile/trace-metal.py`, a repo-local Metal System Trace
  summarizer that reports command-buffer cadence and related stats without raw
  XML spelunking.

These exist specifically to keep Metal timeline work aligned with
`docs/PERF-TOOLS.md` instead of ad hoc one-off commands.

### 27B 4K Decode Trace

Using the new helper and parser, a real 27B decode window at `ctx=4096` now has
command-model evidence instead of guesswork:

- direct decode-window `TokenProfile` run (`128` tokens at `ctx=4096`):
  - `avg_total=42.61 ms`, `avg_gpu=42.08 ms`, `avg_cpu_enc=0.24 ms`
  - `med_total=42.69 ms`, `med_gpu=42.14 ms`, `med_cpu_enc=0.20 ms`
  - `p95_total=43.13 ms`, `p95_gpu=42.64 ms`, `p95_cpu_enc=0.27 ms`
  - GPU / total ratio: `~98.7-98.8%`
- Metal trace summary:
  - `128` decode tokens -> `128` command buffers -> `128` encoders
  - encoder duration median: `0.699 ms`, p95 `1.553 ms`
  - submission cadence median: `43.883 ms`, p95 `45.501 ms`
  - previous completion -> next submit median: `0.538 ms`, p95 `0.754 ms`
  - process-scoped compute intervals: `201`
  - process compute total: `2377.224 ms`, process gap total: `4696.175 ms`
  - process gap split:
    - `<= 10 ms`: `76` gaps, `158.166 ms` total
    - `> 10 ms`: `124` gaps, `4538.009 ms` total
  - compute-intervals-per-CB histogram: `1:75, 2:36, 3:15, 4:1, 5:1`

Interpretation:

- Decode is fully serialized token-by-token today.
- The model's own per-token profiler is the decisive source here: dense 27B 4K
  decode is overwhelmingly GPU-busy, not a giant hidden CPU/driver bubble.
- There is still a real but modest host/command-model gap at 4K, not a giant
  hidden bubble.
- The alarming raw process-gap median was a mixed population. Most of the large
  gaps are simply token-to-token cadence; the short intra-CB gaps total only
  about `158 ms / 128 tokens ≈ 1.2 ms/token` at 4K.
- Double-buffered decode submission remains a legitimate low-single-digit decode
  candidate, but not a miracle lever.
- Heavier encoder/fence restructuring should wait for more context-shape traces
  or a stronger kernel-side reason.

### Bench-Only Pipelined Decode Follow-Up

After settling the 4K decode question, I added a dense-only bench harness path:

- `qwen-bench decode-window --pipelined`

This ping-pongs only `ids_buf` and `argmax_tok`, pre-encodes the next token's
command buffer while the current token is running, and keeps it bench-only.

Measured result so far:

- 27B dense at `ctx=4096`, `window=128`
  - serial: `avg_total=43.08 ms`, `med_gpu=42.73 ms`
  - pipelined: `avg_total=42.54 ms`, `med_gpu=42.27 ms`
  - effect: about `0.54 ms/token` on the first A/B, but only `~0.14-0.16 ms`
    (`~0.3%`) across alternating repeats
- 27B dense at `ctx=32768`, `window=64`
  - serial: `avg_total=51.42 ms`, `med_gpu=50.97 ms`
  - pipelined: `avg_total=51.26 ms`, `med_gpu=50.96 ms`
  - effect: again about `~0.16 ms` (`~0.3%`)

Interpretation:

- The branch is real but tiny, exactly in line with the small completion -> next
  submit gap we saw in the trace.
- It is worth keeping behind the bench-only flag for future context checks, but
  it is not a production checkpoint on its own.

## 2026-05-15 — Concurrent GDN Front Projections At 4K

Status: improved checkpoint reached, bench-only / opt-in branch.

### What Changed

- Added a dense-only bench path that splits each GDN block across multiple
  encoders and runs the four independent front projections (`qkv`, `z`, `beta`,
  `alpha`) in a concurrent compute encoder.
- Left attention and the rest of the dense block logic unchanged.
- Exposed the branch through `qwen-bench ctx-sweep --concurrent-gdn-proj`.

### Validation

- Added a dense correctness gate on `Qwen3.5-0.8B.F32.gguf` comparing the new
  path against the serial path:
  - argmax identical
  - `cos = 1.000000`

### 27B 4K Result

Same harness, same context, same window (`ctx-sweep --checkpoints 4096 --window 64`):

- serial:
  - `43.87 ms/token` total
  - `43.23 ms/token` GPU
  - `0.31 ms/token` CPU encode
  - `22.8 t/s`
- concurrent GDN projections:
  - `42.14 ms/token` total
  - `41.53 ms/token` GPU
  - `0.38 ms/token` CPU encode
  - `23.7 t/s`

Interpretation:

- This is a real GPU-side decode win, not a CPU noise artifact.
- The branch improves total decode by about `1.73 ms/token` at 4K, about `4%`
  throughput.
- CPU encode rises slightly, which is fine because the gain is in GPU time.

### Current Next Step

- Keep this as a checkpoint-worthy experimental branch.
- 27B dense at `16K` now confirms the gain survives as attention cost grows:
  - serial: `47.67 ms/token`, `47.04 ms` GPU, `21.0 t/s`
  - concurrent GDN projections: `46.51 ms/token`, `45.94 ms` GPU, `21.5 t/s`
  - effect: about `1.16 ms/token`, roughly `2.5%`

Interpretation:

- The concurrent-GDN branch is not just a 4K-local artifact.
- The gain compresses somewhat as attention grows, but still holds at realistic
  longer context.

## 2026-05-15 — Concurrent GDN + Attention Front Projections

Status: improved checkpoint reached, still bench-only / opt-in.

### What Changed

- Added a second dense-only branch that applies the same concurrent compute
  encoder pattern to attention front projections (`q`, `k`, `v`).
- Exposed it through `qwen-bench ctx-sweep --concurrent-attn-proj`.
- Added support for running both projection-overlap branches together via
  `--concurrent-gdn-proj --concurrent-attn-proj`.

### Validation

- Added dense correctness gates on `Qwen3.5-0.8B.F32.gguf`:
  - concurrent attention vs serial: argmax matches, `cos = 1.000000`
  - concurrent GDN + attention vs serial: argmax matches, `cos = 1.000000`

### Bounded A/B Results

27B dense, `ctx=4096`, `window=64`, same `ctx-sweep` harness:

- serial: `43.64 ms/token`, `43.08 ms` GPU, `22.9 t/s`
- both branches on: `42.31 ms/token`, `41.77 ms` GPU, `23.6 t/s`
- effect: about `1.33 ms/token`, roughly `3.1%`

27B dense, `ctx=16384`, `window=64`, same harness:

- serial: `47.23 ms/token`, `46.65 ms` GPU, `21.2 t/s`
- both branches on: `46.14 ms/token`, `45.61 ms` GPU, `21.7 t/s`
- effect: about `1.09 ms/token`, roughly `2.3%`

Attention-only by itself was smaller:

- `ctx=4096`: `43.89 -> 43.33 ms/token` (`~1.3%`)
- `ctx=16384`: `47.26 -> 47.22 ms/token` (effectively flat)

Interpretation:

- The combined branch is real and positive at both 4K and 16K.
- It is not additive with GDN-only overlap; attention overlap helps at 4K, but
  contributes little by 16K.
- The combined branch is still a stronger overall decode checkpoint than either
  attention-only or pipelined submission.
