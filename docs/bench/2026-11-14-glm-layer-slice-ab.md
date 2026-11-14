# GLM layer-slice target A/B

## Frozen protocol (before first arm)

Base: current main `50d5397ac`; release binary built once and pinned for all
arms. No source or runtime-default changes. A is the current default
`QWEN_SERVE_GLM_LAYER_SLICE_MS=200`; B is `400`. Keep
`QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS=500` fixed in both arms. Keep the other
serve settings fixed at max active 4, prefill fairness enabled, decode budget
10%, 2,048 loaded rows and 3 sample workers.

Use the existing GLM three-agent 180-second plan and corpus at 48K roots,
`--sampling greedy`, three agents, the same stock UD-IQ3_XXS artifact, and symmetric warm-up
(`--ramp 15`, 48,000 warm tokens, 8 warm reply tokens). Run one arm per
Metal-lease acquisition with a separate model load; each arm must settle at
thermal pressure ≤1 for 60 seconds, allow 2, and abort at 3. Both the driver
and server must measure at nice 0. Record each arm's process priorities, load,
thermal trace, compiler overlap, call counts, and binary/plan hashes.

Pair order is **A/B, B/A, A/B**, for three pairs. For each completed call,
`new_rows = input_tokens - cached_tokens`. Report p50/p95 client TTFT and
server prefill by class: short `<512`, about 1K `512–2,048`, long `>2,048`,
and 48K roots separately. Also report the residual `client_TTFT -
server_prefill_ms` as a diagnostic, not as a named queue or transport stage.
Keep call counts and censoring beside every percentile; missing samples stay
missing.

### Frozen pass rule

- For each of the long and about-1K classes, the geometric mean of the three
  paired B/A p50 client-TTFT ratios must be **<1.00**.
- Short-call p95 client TTFT's paired geometric-mean B/A ratio must be
  **≤1.05**.
- Aggregate TPS's paired geometric-mean B/A ratio must be **≥0.98**.

If a required class lacks completed samples in any pair, report the gate as
unevaluable rather than filling or reclassifying the missing values. Roots
are reported for context, not included in the append pass rule.

Raw outputs go under
`metagross:~/qwen-runs/2026-11-14-glm-layer-slice-ab/`; retain the release
binary until the gate closes. No pushes.
