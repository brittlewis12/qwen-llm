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
should be treated as noisy, not as a reliable 13.7% gain. The first A/B
stage left the product default untouched; the completion-time follow-up below
met its additional rule and this branch now changes the default to 400 ms.

The residual remains an aggregate of all time outside native server prefill;
it is not a direct measurement of admission, peer wait, first-token work, or
transport. Per-request lifecycle instrumentation is needed to split it.

## Completion-time follow-up from the same arms

Using only the retained call records, request completion time is `end - sent`
from HTTP request start to stream completion. We include calls offered in the
180 s window that completed with a known row count. Pairwise comparisons below
use the **intersection of labels** present in both arms; every matched label
has identical input rows, prompt, request spec, and generated-token count on A
and B. This makes the timing distributions like-for-like. Percentiles use the
same nearest-rank method as the serve harness.

The first table gives the raw completed sample counts and generated output
tokens per class in each arm. Pair 1's raw A/B totals differ because the arms
completed different numbers of calls; matched-set output counts are shown in
the next table and are equal on A/B.

| Arm | Short n/tokens | About-1K n/tokens | Long n/tokens |
|---|---:|---:|---:|
| Pair 0 A200 | 72 / 2,212 | 13 / 250 | 6 / 114 |
| Pair 0 B400 | 73 / 2,231 | 13 / 250 | 6 / 114 |
| Pair 1 A200 | 83 / 3,047 | 14 / 379 | 9 / 237 |
| Pair 1 B400 | 73 / 2,231 | 13 / 250 | 7 / 133 |
| Pair 2 A200 | 84 / 3,066 | 14 / 379 | 9 / 237 |
| Pair 2 B400 | 88 / 3,155 | 16 / 417 | 9 / 237 |

Matched completion percentiles are milliseconds. The output column is the
same A and B output-token total for the matched calls.

| Pair | Class | Matched n / output tokens | A200 p50/p95 | B400 p50/p95 | B/A p50/p95 |
|---|---|---:|---:|---:|---:|
| 0 | Short | 72 / 2,212 | 1,173/2,660 | 1,187/2,756 | 1.012/1.036 |
| 0 | About 1K | 13 / 250 | 2,596/3,723 | 2,571/3,052 | 0.990/0.820 |
| 0 | Long | 6 / 114 | 6,575/9,559 | 6,495/9,080 | 0.988/0.950 |
| 0 | Pooled | 91 / 2,576 | 1,276/9,236 | 1,262/8,784 | 0.989/0.951 |
| 1 | Short | 52 / 1,832 | 1,493/6,982 | 1,159/5,140 | 0.776/0.736 |
| 1 | About 1K | 9 / 174 | 3,086/8,088 | 2,508/2,918 | 0.813/0.361 |
| 1 | Long | 6 / 114 | 7,758/11,161 | 6,337/8,913 | 0.817/0.799 |
| 1 | Pooled | 67 / 2,120 | 1,790/9,678 | 1,220/8,817 | 0.681/0.911 |
| 2 | Short | 84 / 3,066 | 1,974/7,425 | 1,912/7,279 | 0.968/0.980 |
| 2 | About 1K | 14 / 379 | 3,235/7,912 | 3,399/7,355 | 1.051/0.930 |
| 2 | Long | 9 / 237 | 9,476/14,727 | 9,106/12,732 | 0.961/0.865 |
| 2 | Pooled | 107 / 3,682 | 2,344/10,245 | 2,459/9,232 | 1.049/0.901 |

| Completion time | Pair 0 B/A | Pair 1 B/A | Pair 2 B/A | Geomean |
|---|---:|---:|---:|---:|
| Short p50 | 1.012 | 0.776 | 0.968 | 0.913 |
| Short p95 | 1.036 | 0.736 | 0.980 | 0.908 |
| About-1K p50 | 0.990 | 0.813 | 1.051 | 0.946 |
| About-1K p95 | 0.820 | 0.361 | 0.930 | 0.650 |
| Long p50 | 0.988 | 0.817 | 0.961 | 0.919 |
| Long p95 | 0.950 | 0.799 | 0.865 | 0.869 |
| Pooled p50 | 0.989 | 0.681 | 1.049 | **0.891** |
| Pooled p95 | 0.951 | 0.911 | 0.901 | 0.921 |

The harness's saved `gap_*` values are gaps between consecutive timestamped
SSE delta events in each response, which are the available client-visible
stall measure. They are not exact token timestamps: the six arms captured
0.40–0.52 delta events per generated token, so a delta can contain multiple
tokens. Values below are p50/p99/max event-gap milliseconds per arm.

| Arm | Event gap p50/p99/max (ms) |
|---|---:|
| Pair 0 A200 | 25.4/297.8/463.3 |
| Pair 0 B400 | 26.1/422.4/751.6 |
| Pair 1 A200 | 25.4/461.5/2,768.8 |
| Pair 1 B400 | 25.2/415.2/774.4 |
| Pair 2 A200 | 25.4/456.6/3,089.1 |
| Pair 2 B400 | 23.6/648.6/4,485.8 |

| Event gap | Pair 0 B/A | Pair 1 B/A | Pair 2 B/A | Geomean |
|---|---:|---:|---:|---:|
| p50 | 1.028 | 0.993 | 0.928 | 0.982 |
| p99 | 1.419 | 0.900 | 1.421 | **1.219** |
| Maximum | 1.622 | 0.280 | 1.452 | 0.870 |

Pooled matched completion p50 passes the ≤1.00 rule at 0.891×. The p99
client-visible delta-gap geomean is 1.219×, within the suggested 1.25× bound,
though pairs 0 and 2 individually rise to 1.419× and 1.421× and the maximum
gap reaches 4.49 s in pair 2. Applying the three-pair geomean rule, recommend
400 ms default with the exact rollback `QWEN_SERVE_GLM_LAYER_SLICE_MS=200`.
True token-level ITL needs per-token or token-batched timestamps; this commit
uses the measured client-visible delta-gap proxy and records that limit.

The default change and regenerated ENV reference are recorded in the latest
commit; this is a same-binary A/B interpretation, not a new GPU run.
