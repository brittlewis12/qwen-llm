# GLM layer-slice target A/B

## Frozen protocol (before first arm)

Base: current main `50d5397ac`; release binary built once and pinned for all
arms. No source or runtime-default changes. A is the current default
`QWEN_SERVE_GLM_LAYER_SLICE_MS=200`; B is `400`. Keep
`QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS=500` fixed in both arms. Keep the other
serve settings fixed at max active 4, prefill fairness enabled, decode budget
10%, 2,048 loaded rows and 3 sample workers.

Use the existing plan `serve-plan-180.json` (seed `20261109`) and corpus
`metal-code-corpus.txt` at 48K roots, `--sampling greedy`, three agents, the
same stock UD-IQ3_XXS artifact, and symmetric warm-up (`--ramp 15`, 48,000
warm tokens, 8 warm reply tokens). Run one arm per
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

## Results

The frozen three-pair gate is complete. The measured default was 200 ms;
candidate B was 400 ms, with `QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS=500` fixed.
All six arms used the release binary built from `50d5397ac` (qwen SHA-256
`1f574dcf768ddcbd4417ec6eb1f48e0e42f3a78fad7af432cc614772c8d9343f`). Arms
ran in the preregistered A/B, B/A, A/B order, one arm per lease. Both driver
and server ran at nice 0. Each arm had three 48K warm roots, then a 180 s
measurement window. Thermal settled at 0 or 1 for at least 60 s before load
and again after warm-up; peak during arms was at most 2, never 3. Shader
compilation overlapped measurement in all six arms.

The following are pooled across the three agents. Percentiles use nearest
rank. Each cell is p50/p95 in milliseconds; residual is computed per call as
client TTFT minus server `prefill_ms`, then percentiled separately. `n` is
completed calls with a known row count and TTFT. Three offered calls per arm
had no final row count at capture; one or two remained pending at the window
cutoff. They are not assigned to a class or imputed.

### Short appends (<512 new rows)

| Arm | n | Client TTFT | Server prefill | Residual |
|---|---:|---:|---:|---:|
| Pair 0 A200 | 72 | 1,150/1,796 | 347/609 | 810/1,262 |
| Pair 0 B400 | 73 | 1,179/1,659 | 352/693 | 830/1,069 |
| Pair 1 A200 | 83 | 1,234/3,327 | 424/1,813 | 696/2,257 |
| Pair 1 B400 | 73 | 1,151/1,631 | 347/692 | 804/1,069 |
| Pair 2 A200 | 84 | 1,187/3,313 | 406/1,886 | 661/2,301 |
| Pair 2 B400 | 88 | 1,190/3,179 | 387/1,605 | 748/2,166 |

### About 1K appends (512–2,048 new rows)

| Arm | n | Client TTFT | Server prefill | Residual |
|---|---:|---:|---:|---:|
| Pair 0 A200 | 13 | 2,578/2,781 | 1,575/1,709 | 977/1,184 |
| Pair 0 B400 | 13 | 2,564/2,940 | 1,590/1,851 | 914/1,090 |
| Pair 1 A200 | 14 | 2,875/8,507 | 1,540/1,707 | 1,361/6,931 |
| Pair 1 B400 | 13 | 2,505/2,746 | 1,576/1,740 | 883/1,006 |
| Pair 2 A200 | 14 | 2,935/7,891 | 1,566/1,713 | 1,417/6,241 |
| Pair 2 B400 | 16 | 2,494/7,352 | 1,534/1,674 | 866/5,817 |

### Long appends (>2,048 new rows)

| Arm | n | Client TTFT | Server prefill | Residual |
|---|---:|---:|---:|---:|
| Pair 0 A200 | 6 | 6,571/9,554 | 4,058/7,352 | 2,176/2,609 |
| Pair 0 B400 | 6 | 6,491/9,076 | 4,139/7,475 | 1,595/2,517 |
| Pair 1 A200 | 9 | 9,244/12,209 | 7,158/7,509 | 2,973/5,056 |
| Pair 1 B400 | 7 | 8,612/9,203 | 7,073/7,595 | 1,572/2,421 |
| Pair 2 A200 | 9 | 9,117/12,019 | 7,101/7,540 | 2,889/4,919 |
| Pair 2 B400 | 9 | 8,919/9,856 | 7,155/7,369 | 1,861/2,869 |

Root warm-up client TTFT over three roots per arm was p50/p95: pair 0 A/B
45.947/47.865 s and 44.495/44.845 s; pair 1 44.847/44.904 s and
44.418/44.811 s; pair 2 44.458/44.865 s and 44.376/44.845 s. The harness
retained these warm-root TTFT events but not matching per-request root
`prefill_ms`, so root server-prefill percentiles are unavailable.

### Frozen gates and decision

| Measure | B/A by pair | Geomean | Gate |
|---|---|---:|---|
| Long client TTFT p50 | 0.988, 0.932, 0.978 | 0.966 | Pass (<1.00) |
| About-1K client TTFT p50 | 0.995, 0.871, 0.850 | 0.903 | Pass (<1.00) |
| Short client TTFT p95 | 0.924, 0.490, 0.959 | 0.757 | Pass (≤1.05) |
| Aggregate TPS | 0.931, 1.601, 0.985 | 1.137 | Pass (≥0.98) |

Aggregate TPS was 46.739/43.500, 27.994/44.828, and 28.228/27.806 for
A/B in pairs 0–2. Pair 1 is a large order/noise swing; removing it would make
the remaining two-pair TPS geomean 0.958, below the frozen gate. The three-pair
rule nevertheless passes as preregistered. Recommend changing the default
slice from 200 ms to 400 ms with a `QWEN_SERVE_GLM_LAYER_SLICE_MS=200`
rollback. No product default was changed in this stage. The steady p50 wins
for long and about-1K appends support the direction; the throughput margin
should be treated as noisy, not as a reliable 13.7% gain.

The residual remains an aggregate of all time outside native server prefill;
it is not a direct measurement of admission, peer wait, first-token work, or
transport. Per-request lifecycle instrumentation is needed to split it.
