# Bench Notes

For commands: `scripts/bench/family.py --help`.
For history: `docs/bench/<stamp>-family/`.
For priorities: `docs/PERF-ROADMAP.md`.

## What we copy from llama-bench

- `build_commit` on every row, so a comparison is decodable later.
- `pp<N>` / `tg<N>` as the shape vocabulary.
- JSON output as the machine-readable surface; stderr stays human-only.

## llama.cpp comparator provenance

- Scoreboard scripts default to `scripts/bench/llama-cpp.lock.json`, built by
  `scripts/bench/ensure_llama_cpp.py` into `~/.cache/qwen-llm/llama.cpp`.
- Ambient local llama.cpp binaries are allowed only as explicit one-offs with
  `--allow-unpinned-lcpp`; they are not canonical family baselines.
- In `scripts/bench/family.py`, `--shapes` is exact: `--shapes tg128` runs only
  `tg128`; omit `--shapes` for the default family grid.

## What we don't

- `MTL,BLAS` in lcpp's `backends` is a registration artifact — BLAS does
  no work on Qwen3.5/3.6 at any shape we measure (audit in
  `docs/PERF-LOG.md`). Treat it as informational.
- `--runs 3` not `-r 5`. The M4 Max stays stable enough that the extra
  reps mostly pay in elapsed time.

## Opinionated choices

- **No model-size bandwidth row for MoE models.** Naive `model_size x t/s`
  overstates by ~10x for A3B / A10B. A MoE packet may instead report
  census-derived `active_bytes_per_token`, effective active-byte bandwidth, and
  percent of the maintained stream anchor when it pins the dense/shared/routed
  role decomposition and quant byte widths. These fields belong in the JSON
  schema before becoming scoreboard authority.
- **One JSON file per (engine, model, shape).** Lets you re-run a single
  cell of the matrix without disturbing the rest.
- **Sanity flags inline in the digest.** `pp1024 < pp512` and friends
  appear at the bottom of the README, not in a separate file.

## Known gaps

- No cross-day / thermal variance tracking — we mitigate by running
  lcpp and qwen in the same sweep so drift cancels in the ratio.
- No auto-diff against the previous baseline.
- No first-token / TTFT measurement.
- No census-to-active-bytes exporter in the bench tooling yet.
