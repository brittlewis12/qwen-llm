# Result (acquired 2026-09-03)

Disposition: **SEMANTIC_GO**. See `result.json` (schema
`qwen4exp.selected_semantic_result` v1) for every per-token NLL and both
known-answer generations.

- Source: `refactor/consolidate` at `9bdb0a54` plus a temporary ignored
  evaluator in `qwen4exp_runtime.rs` following PROTOCOL.md exactly (removed
  after acquisition, as the protocol requires). Environment scrubbed to only
  `QWEN4EXP_Q3_K_XL_RUNTIME_GGUF`; selection arm chosen by the test-scoped
  override, not the public switch.
- Natural gate: pooled `mean_nll(selected) - mean_nll(safe)` =
  `0.005498` <= `0.009950`. Per context: ssh
  `-0.000183`, zsh `+0.011179`.
- Known answer: safe and selected both parse to
  `{"code":"amber-lattice-2049","record":"K-17"}`.
- Prefill wall: ssh `30070 -> 4585 ms`,
  zsh `26372 -> 4570 ms` (2,179 tokens).
- Run order and single attempt as frozen: ssh safe, ssh selected, zsh safe,
  zsh selected, known-answer safe, known-answer selected.
