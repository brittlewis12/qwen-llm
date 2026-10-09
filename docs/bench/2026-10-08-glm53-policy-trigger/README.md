# GLM-5.3-Flash Fast policy: first run of the investigation trigger (map #12, 2026-10-08)

The adopted Fast policy (PERF-LOG 2026-10-08) keeps one investigation
trigger: worst-position KL(Exact || Fast) above 0.33 on the frozen natural
cases (`scripts/reference/glm53/reuse-natural-v1.json`), Fast at 512 rows,
cold and warm. `reuse_natural_evaluate` (report schema v2) ran it at
`d65a09a2` (rebased as `9148817d`; the interleaved main commits touched
only qwen-cli code, not the harness) with Metal API validation (`natural-v2.json`).

- **Not tripped.** Worst at 512 rows: 0.155 (H2 warm), 0.146 (H5 warm),
  0.113 (H2 cold); every other case below 0.10. At 97 rows (diagnostic)
  the worst is 0.181 (H2 cold).
- **No change from the baseline:** every arm's worst KL equals the
  committed 2026-10-07 report's (`../2026-10-07-glm53-natural-reuse/`,
  hash-pinned) exactly, so nothing between those runs (the router change,
  the sparse IQ3_S down retile, the drift-test migration) moved Fast on
  these cases.
- **Exact stays bitwise:** Exact warm (512 and 97 rows) and Exact cold at
  97 rows equal Exact cold at 512 in logits and persistent state on every
  case; no failures recorded.
- The migrated tests also pass at `d65a09a2`:
  `fast_reuse_schedule_sensitivity_across_reuse_boundaries` (cancel and
  resume bitwise; drift reported: ordinary warm/cold 2.13e-2, tool loop
  5.63e-2, as on 2026-10-06) and
  `packed_fast_chunkings_keep_bit_identities_with_drift_alarms` (oracle
  gates, the 512==128 and 64==97 bit identities, state alarms quiet).

The trigger is an explicitly chosen, provisional investigation limit, not a
quality verdict; quality is the preregistered cohort's
(`../2026-10-08-glm53-fast-quality/`, `../2026-10-08-glm53-fast-split-quality/`).
