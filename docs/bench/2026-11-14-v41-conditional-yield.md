# V4.1 conditional prefill yielding

Bounded stage:2026-11-14 16:52:42–19:22:42 UTC. Branch
`perf/v41-prefill-yield-conditional`, based on `cabbb8a24` / main `1522261bc`.
The previous unconditional experiment stays recorded separately.

## Policy frozen before the first arm

Default-off `QWEN_SERVE_V41_PREFILL_YIELD=1`, independent `=0` rollback.
Conditional mode uses **only incumbent completed-command drains**: Full
appends poll after original Engram-overlap segments; CED polls after its
original encoder quantum drains, leaving the decoder pipeline untouched.
There are no imposed timed GPU cuts and no decoder rounds at empty polls.
Positive `LAYER_SLICE_MS` enables polls;0 disables them. Its400 default in
conditional mode is a legacy enable value, not a claimed time bound.
`LAYER_SLICE_MIN_MS` defaults700 in this mode; all calls below512 rows stay
unsliced even if the minimum is lowered.

At a drain, collect prepared HTTP work. A read-only compatible-live-prefix
hint identifies short pending suffixes before allocating/restoring a state;
unknown long prompts wait for ordinary admission. A waiting short request
finishes its original checkpoint/header calls as **one** service event,
without recursive yields. Suspended slots/reserves remain charged. Ready
short work also takes priority at ordinary pumps. Normal post-quantum decode
credit includes actual peer prefill service; empty polls retain incumbent
credit. No numerical kernels, precision, row partition, layer order, Engram
representation or checkpoint capture rules change.

Per-request polls, actual conditional yields and native-call counts are
recorded; explicit yield-event lines retain parent/child request IDs. Report
full-arm and measured-window fire counts separately from ordinary short-priority
dispatches. No work is called a yield solely because a poll occurred.

## Exactness and checks

CPU cases cover no decode inside empty polls, preserved post-quantum credit,
no short slicing, no long admission at an empty poll, read-only prefix hints,
bounded suspended admission/reserves and cancellation. No new ignored tests.
The M5 release witness compares baseline, empty polls and actual nested short
service at48K CED root and1K@80K: identical logits, causal payload digest and
next logits, plus peer logits/state. Empty polls must also preserve the exact
GPU command-operation/dimension/count topology. `perf-ablations` is used by the
witness for existing diagnostics only; serve is the ordinary release build,
and both metallib identities must match.

## Frozen serve rule

Same file-A/corpus and plan as the preceding gate: seed20261112, greedy,
three49,152-token sequential roots /32-token replies,15s ramp,120s window,
context196608, FAIR1 / MAX_ACTIVE4 / SAMPLE_WORKERS3. Same pinned release binary
for both switches. Run order **A1 B1 B2 A2 A3 B3**, each arm its own waiting
lease<=15min. Actual nice0 checked by `ps`; builds nice10. Engram companion
purged to0 resident pages before every arm; no other resident model. Settle
thermal<=1 continuously60s, allow2, abort3. Symmetric CPU boundary tests and
root/ramp warmup in both arms. Host load, disk reads and compiler presence stay
in the record; no speed/load-based arm exclusions. Latest arm start19:00 UTC
leaves cleanup/reporting time before19:22:42.

The **unchanged** previous reduction uses label/spec/target, exact prompt/token
hashes and actual new-row equality for calls offered in both windows. Short
means1..511 new rows; about1K512..2000. Matched observed TTFT includes partial
responses. Completion additionally requires both responses complete and equal
generated-token counts; pool all three matched pair samples before p50.
Percentiles are nearest rank. Report all exclusions and censored counts.

- Short client TTFT p95: geomean(candidate/baseline)<1.00.
- Pooled matched completion p50: candidate/baseline<=1.00.
- Aggregate TPS: geomean(candidate/baseline)>=0.98.
- Root client TTFT, root native, short native and about1K native: each
  geomean(baseline/candidate)>=0.98. Root uses each arm's mean of three roots;
  append-native ratios use sums over matched completed calls.
- SSE delta-event gap p99: geomean(candidate/baseline)<=1.25; both consecutive
  event endpoints inside the window, as in the harness. This is an event-gap
  proxy, not exact token-level ITL.

All three pairs and bitwise checks are required. Pass=>default-on with0
rollback; fail=>default-off and record whether the line merits more work.
If the queue prevents completion, report the completed measurements and keep
the pins. No pushes.

## Results

Pending. No promotion before the complete frozen gate.
