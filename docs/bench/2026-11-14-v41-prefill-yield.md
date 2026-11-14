# V4.1 row-preserving prefill yielding

Stage start2026-11-14 14:54:10 UTC; stop/report by16:54:10 UTC. Fresh branch
`perf/v41-prefill-yield` from main `1522261bc`. Default-off prototype;
`QWEN_SERVE_V41_PREFILL_YIELD=1` enables it, `=0` restores the incumbent.

## Design and exactness

V4.1 already has drained-command layer callbacks, but they service decoders
only and do not return to HTTP intake. The new hook collects already-prepared
HTTP jobs at those boundaries, admits at most one within the existing active
limit, and executes one waiting short prefill call (remaining rows<512).
The original parent is suspended on the same stack/thread, excluded from
decode/snapshot scans; its active slot and output reserve remain charged.
Peer prefill does not recursively interleave. Raw HTTP/control events and
worker joins stay in the ordinary loop. Native work excludes peer service;
`peer_prefills` is retained with phase timings.

The scheduling switch selects existing400ms layer targets (minimum predicted
work500ms); explicit `QWEN_SERVE_V41_LAYER_SLICE_MS=0` disables boundaries.
Rows, layer order, kernels, precision, CED encoder/replay geometry, Engram
representation and checkpoint rules are unchanged. Shared tracing stays opt-in.
The GLM resumable scheduler informed the policy; the V4.1 implementation uses
its existing stack-preserving callbacks instead of moving native state.

CPU tests cover short service before the parent finishes, identical scripted
outputs, suspended admission/reserves, cancellation, and actual loopback
prepared-job submission before the parent pump returns. The native witness
uses production layer APIs at48K CED root and1K@80K, comparing logits, full
causal-payload digests, next-token logits and a nested150-row peer's logits/state.
`perf-ablations` is needed by the witness only for the existing snapshot digest;
serve uses the ordinary release build. Both metallib identities are recorded.

## Frozen serve protocol, before the first arm

One pinned release binary; A=`PREFILL_YIELD=0`, B=`PREFILL_YIELD=1`, each
without a layer-MS override (A default0, B default400). Other settings fixed:
FAIR1, MAX_ACTIVE4, SAMPLE_WORKERS3, greedy; file A/corpus and seed20261112;
three sequential49,152-token roots/32-token replies,15s ramp,120s window,
context196608. Run order **A1 B1 B2 A2 A3 B3** (AB/BA/AB). Each arm gets its
own waiting lease<=15min. Engram companion purged to0 resident pages before
each load. Actual serve nice0; build nice10; no resident model before loading.
Settle thermal<=1 continuously60s, allow2, abort3. Background load/compiler
presence and process/window disk-read counters are retained for every arm.
Identical CPU boundary tests plus three-root/ramp warmup in both arms; all GPU
pipeline shapes are incumbent shapes. No arm exclusions based on speed/load.

Nearest-rank percentiles, geometric means of three paired ratios, with
matching by label, request spec, exact input-token IDs/prompt hash and actual
new-row count. Calls must be offered inside both windows. Short means1..511
new rows, about1K means512..2000. TTFT includes every matched observed first
token, even if the response is partial; unobserved/censored counts are reported.
Completion matching additionally requires completion in both arms and equal
generated-token counts; completion is end-sent. Raw counts and exclusions remain.

Frozen scheduling rule:

- Short client TTFT p95: geomean(candidate/baseline)<1.00.
- Pooled matched-call completion p50: candidate/baseline<=1.00, pooling the
  three pairs' matched completed samples before taking each median.
- Aggregate TPS: geomean(candidate/baseline)>=0.98.
- Root client TTFT, root native prefill, short native prefill, and about1K
  native prefill: each geomean(baseline/candidate)>=0.98. Native append ratios
  use sums over matched completed calls, preserving the previous reduction.
- Client-visible SSE delta-event gap p99: geomean(candidate/baseline)<=1.25,
  using each arm's harness window gaps. This is an event-gap proxy for ITL.

All three pairs and the bitwise witness are required for promotion. If the
queue prevents completion by the cutoff, report incomplete and keep pins.
Pass: default-on with0 rollback. Fail: default-off, record retained. The prior
served A/A control already demonstrated batch-composition variation; this
stage's exactness claim is same-input/rows/order native logits and state.

## Results

Pending. No promotion or performance claim before the witness and full gate.
