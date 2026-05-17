# Bench Notes

For commands: `scripts/bench/family.py --help`.
For history: `docs/bench/<stamp>-family/`.
For priorities: `docs/PERF-ROADMAP.md`.

## What we copy from llama-bench

- `build_commit` on every row, so a comparison is decodable later.
- `pp<N>` / `tg<N>` as the shape vocabulary.
- JSON output as the machine-readable surface; stderr stays human-only.

## What we don't

- `MTL,BLAS` in lcpp's `backends` is a registration artifact — BLAS does
  no work on Qwen3.5/3.6 at any shape we measure (audit in
  `docs/PERF-LOG.md`). Treat it as informational.
- `--runs 3` not `-r 5`. The M4 Max stays stable enough that the extra
  reps mostly pay in elapsed time.

## Opinionated choices

- **No bandwidth row for MoE models.** Naive `model_size × t/s`
  overstates by ~10x for A3B / A10B; honest accounting needs active-param
  tracking in the JSON schema first.
- **One JSON file per (engine, model, shape).** Lets you re-run a single
  cell of the matrix without disturbing the rest.
- **Sanity flags inline in the digest.** `pp1024 < pp512` and friends
  appear at the bottom of the README, not in a separate file.

## Known gaps

- No cross-day / thermal variance tracking — we mitigate by running
  lcpp and qwen in the same sweep so drift cancels in the ratio.
- No auto-diff against the previous baseline.
- No first-token / TTFT measurement.
