# V4.1 intake + exact tokenization reuse: served qualification

Stage began 2026-11-14 12:36:17 UTC, stop/report by 14:36:17 UTC. Rebased `perf/serve-ttft-instrumentation` onto main `e70da3f8a`. Only generated ENV documentation conflicted and was regenerated. Shared tracing method signatures, stages and snapshots remain stable for the GLM hook lane.

## Frozen protocol before the first arm

One release binary, candidate defaults on for `QWEN_SERVE_ASYNC_PREPARE` and `QWEN_SERVE_TOKEN_PREFIX_CACHE`; each has its independent `=0` rollback. All baseline arms explicitly set both to 0; candidate arms exercise the new defaults without those overrides. No shader, numerical kernel or checkpoint change.

Run order is **A0, A1, B1, B2, A2, A3, B3**. A0/A1 are the two baseline controls. A1 is also preregistered as pair1's baseline, with no selection based on its speed; pairs are A1/B1 (AB), B2/A2 (BA), A3/B3 (AB). This avoids one redundant baseline arm while preserving the requested interleaving. Every arm is its own waiting <=15-minute lease, actual nice0, inherited FD, Engram purge to 0 resident pages before loading, no other resident model, thermal <=1 continuously60s before load/window, allow2/abort3. Build nice10; release serve and test-profile tests. File-A artifact/corpus/plan remain unchanged: greedy, seed20261112, 3 agents, 49,152-token sequential roots / 32-token root replies, 15s ramp, 120s window, context196608, V41_MAX_ACTIVE4 / SAMPLE_WORKERS3 / PREFILL_FAIR1.

Symmetric warmup before every arm: identical CPU path tests (including the cache and blocked-pump paths even for baseline), released-tokenizer full/incremental witness, then the same three roots and three-agent ramp. No new GPU pipelines exist; metallib identity is recorded. The cache's per-conversation serving state warms only where the feature is enabled, as intended.

Control: compare every common response's text/reasoning/argument bytes; completed pairs require equality, partial pairs require common-prefix equality, no-output cases are unobserved. Report response mismatch rates plus first divergence per conversation, matched-input hashes/cuts, and decode-round cohort histories. If A0/A1 have no genuine observed divergence while the prior candidate did, stop for attribution and do not promote. A positive A/A control supports the existing batch-numerics hypothesis only where request inputs/cuts match and cohort histories differ; it is not proof of every possible candidate difference.

Frozen six-metric gate, preserving `v41_prefill_serve_compare.py`: geometric mean over three pair ratios, candidate/reference TPS >=0.98, reference/candidate root **native prefill** >=0.98; short and ~1K native prefill and client TTFT >=1.00. Match labels and exact new rows, both completed/offered inside the window, requested append150–2000, no CED; short native rows128–511, ~1K512–2000. Retain all excluded/unmatched/censored counts. Root client TTFT is reported separately, including all three roots per arm and host load, without substituting it for native prefill or changing the frozen rule. Outside-native p50/p95 is reported for every arm using all eligible observed TTFTs.

## Diagnostics added without changing the shared API

The existing timing knob now additionally emits full generation round records on round completion, after releasing the history mutex. Request/round IDs, ordered participants, times and shape are recorded, with no token content. The prior snapshots still stop at first visible text. Opt-in input witness lines carry exact rendered-prompt/token-ID SHA256, byte boundaries, resolved token cuts, and the initial prefill-logit SHA256. These do not change arithmetic; every new arm uses the same probes. Raw request/response payloads are retained on M5 for attribution rather than being discarded after a failed or cancelled turn. No tracing API signatures changed.

## Results

Pending. No promotion before controls and all three pairs complete and pass. Queue/deadline exhaustion is incomplete, not a pass. Pinned executables remain in the run directory until this qualification closes.
