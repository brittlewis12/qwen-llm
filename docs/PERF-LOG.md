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

Recommended next checkpoint commit, if requested explicitly:

```text
perf: record roadmap and promote long-context attention policy
```

Before committing, inspect staged/unstaged diff carefully and include only the
intended checkpoint files/changes. Do not commit secrets or unrelated scratch.

### Next Handoff Instruction

Start by reading:

1. `docs/PERF-LOG.md`
2. `docs/PERF-ROADMAP.md`
3. `git status --short`

Then continue with the ranked item #1 unless fresh measurements or user
direction change priority.
