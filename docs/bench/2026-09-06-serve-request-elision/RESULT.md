# Dense prefill head elision and HTTP readiness: KEEP

Two independently measured mechanisms. Ordinary loading on Apple M4 Max; no
whole-model residency or parallel GPU work. All attempts, including failures,
remain in the request-work-elision worktree under
`target/profiles/request-elision/`. The reusable acquisition instrument is
`scripts/serve/request_probe.py`; its request bodies, raw SSE events, server
logs, and before/after power/thermal/VM observations accompany each attempt.
Wall ends at the SSE `[DONE]` marker, not TCP EOF. First content excludes HTTP
headers and heartbeats. These are resident-server requests, not process-cold CLI
results. No new all-context, all-family, or distribution-exact claim is made.

## Dense serial-prefill head elision

Baseline `ab5e3d2f`, candidate `ae9115d2`. The baseline was refreshed after main's
pinned-template integration. Two release processes per arm, A-B-B-A, with a short
warmup followed by distinct code/prose/sampled requests and cache/continuation
controls. Requests have 32/48/37 prompt tokens and 128 generated tokens. Sampler
is greedy except the third request (temperature 0.7, seed 1729).

Serve now skips final norm, full-vocabulary head, and logits readback on non-final
dense serial-prefill rows. The last row still returns logits; capture rows remain
complete. The plain no-tail API now follows the same default concurrent-GDN
schedule as ordinary dense forwarding instead of accidentally changing topology.
MoE and the 48-token serial/packed threshold remain unchanged.

| Model/lane and request | A TTFT ms | B TTFT ms | TTFT saving | Request-wall saving |
| --- | ---: | ---: | ---: | ---: |
| 27B Q4 no-spec, prose | 1927.239 | 1843.712 | 4.334% | 0.774% |
| 27B Q4 no-spec, sampled | 1523.398 | 1459.952 | 4.165% | 1.078% |
| 27B Q8 + DFlash2 Q8, code | 1940.887 | 1891.466 | 2.546% | 1.021% |
| 27B Q8 + DFlash2 Q8, prose | 2921.146 | 2796.166 | 4.278% | 1.223% |
| 27B Q8 + DFlash2 Q8, sampled | 2239.126 | 2145.563 | 4.179% | 1.455% |

Both ordering pairs improve TTFT in every listed cell. Q4 code TTFT is excluded
from promotion: control spread 5.67% and its first pair is flat. Q4's completed
EOS followup restores 21 of 41 tokens and improves TTFT/wall 2.885%/2.619%.
The 128-output wall gains are deliberately not presented as 4% request gains.

All output hashes and usage/cache counts match within each model/lane. Engine
tests additionally compare final logit and captured-hidden bits, K/V and GDN
snapshot bytes, and a continuation on 0.8B F32, 27B Q4_K_M, and 27B Q8_0. Default
concurrent and explicit serial scheduling pass in separate processes. The CLI
tail regression covers widths 1/2/16/48 at fresh and nonzero positions on 0.8B Q4.

Tiny exact-hit rows were not clean no-regression evidence: Q8 EOS-hit wall
regressed 29.534 ms / 16.8%, with 6.71% control spread; Q4 exact-hit controls were
also noisy. Rather than treating those rows as a kernel effect or discarding
them, source inspection found the 50 ms acceptor sleep. This prompted the
independent transport experiment below. The head-elision claim remains scoped
to engaged serial-prefill TTFT, not cache-hit improvement or decode throughput.

## Readiness wait and inherited socket mode

Baseline `ae9115d2` already includes head elision. Final candidate `f7061e86`
adds bounded `poll(POLLIN)` instead of sleeping 50 ms after `WouldBlock`, and
clears inherited `O_NONBLOCK` on accepted sockets. Inference, HTTP wire bytes,
single-flight ownership, and the bounded shutdown wait remain unchanged.

The initial readiness-only `2f0dc10b` had strong latency results but a fast
0.8B client exposed a connection reset. A completion-readiness hypothesis led
to `cb452471`; the next probe still failed, so that prototype was removed at
`5ac3dec0`. The actual inherited socket flag is reproduced by a focused macOS
test (`accepted inherited O_NONBLOCK=true`). Without clearing it, accepting
before body arrival can return EAGAIN and misclassify a request as timed out.
The new delayed-byte regression verifies blocking reads under the existing
absolute deadline. No early-readiness callback or new queue survives.

The first small-model request also correctly rejected an unsupported
`x_qwen.no_thinking` control before acquisition. Its repaired fixture omits that
control in both arms. No failed attempts were deleted or scored as successes.

Final A-B-B-A repeats the full Q8/DFlash panel plus 16 consecutive exact EOS hits
per process; the dissimilar 0.8B Q4 no-spec guard uses warmup plus 16 exact hits.
All 160 responses across the eight final processes complete, retaining every
output hash and usage/cache count. The process, not each cache hit, is the
replication unit. Values below are medians of the two process medians.

| Lane | A cached TTFT ms | B cached TTFT ms | A request wall ms | B request wall ms |
| --- | ---: | ---: | ---: | ---: |
| 27B Q8 + DFlash2 Q8 | 56.448 | 15.237 | 232.862 | 174.151 |
| 0.8B Q4 no-spec | 43.256 | 2.895 | 58.591 | 8.005 |

Both pairs save more than 39 ms TTFT in each lane, above the 10 ms screen.
Q8 cached wall saves 25.21%; small-model cached wall saves 86.34%. Per-process
nearest-rank p95 TTFT (16 rows, hence the maximum) moves Q8
`76.238/77.078 -> 17.788/17.025 ms` and 0.8B
`52.639/53.897 -> 3.537/3.271 ms`.

Do not transfer these percentages to long outputs: Q8 128-output code/prose/
sampled wall medians regress 1.47%/2.15%/1.42% in the final packet, below the
3% guard but not strict Pareto evidence. Exact-code 128-output wall is +0.69%.
There is visible decode drift in the second candidate process. The release
decision is the large, repeated cached-response latency reduction and corrected
request-read contract, not improved steady-state generation. Wall savings larger
than the deleted sleep are not attributed solely to kernel wakeup time.

99 focused serve tests pass, with the separately executed GPU test normally
ignored. Coverage includes delayed body arrival, busy 503/Retry-After, first
admission, disconnect handling, SSE/non-stream conformance, and cooperative
shutdown. The first new busy test assumed header case and was corrected before
timing; HTTP header matching is case-insensitive.

Disposition: keep both scoped improvements. Preserve all negative/noisy rows.
No further same-cell retuning, broad benchmark grid, or cold-load claim follows.
