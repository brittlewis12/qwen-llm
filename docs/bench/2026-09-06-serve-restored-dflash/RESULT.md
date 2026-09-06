# Restored DFlash admission: HOLD

## Decision

Do not promote unconditional pending-checkpoint admission. Candidate `f12bc2f9`
corrects a real consumed-versus-matched length mismatch, but exposes a losing
generation policy on prose and short EOS turns. Reverted at `01d24ccd`.
No new runtime behavior survives. This is not a rejection of restored
speculation itself: the code continuation shows a substantial scoped win.

## Mechanism and scope

A completed checkpoint matches its emitted pending token without having consumed
or captured it. For the first continuation here, 8,812 matched tokens restore
8,811 consumed tokens. A capture window starting at 6,784 therefore contains
2,027 restored rows, not 2,028. Existing admission compares with matched length
and rejects the complete consumed tail. The candidate uses restored position;
prefill then captures pending/suffix rows before the existing final capture gate.
The 16K hard stop, verifier margin, sampling, and backoff policies are unchanged.

Below 16K the adaptive controller uses accepted-token yield without charging
margin-fallback serial replay. This is consequential, not a kernel microbench:
both candidate processes execute 19 speculative code steps with 109/133 accepted
drafts and two fallbacks, versus 39 prose steps with 89/273 accepted drafts and
21 fallbacks. Neither turns speculation off. Prose replay alone costs
4,477.8/4,565.2 ms, in addition to 4,700.3/5,401.9 ms verification. High draft
acceptance is not equivalent to avoiding target forwards when fallback runs.

## Acquisition

- M4 Max, Qwen3.8-27B Q8_0 target + DFlash2 Q8_0, release baseline `75e30353`
  versus candidate `f12bc2f9`. Both include the prior scratch/HTTP improvements.
- Four disposable server processes, A1-B1-B2-A2, eight sequential SSE requests
  each. Fresh 8,810-token natural seed, exact hit, pending-checkpoint blue and
  sampled-green turns, code/prose 128-output turns, packed suffix, fresh short.
- Real client first string delta and full wall through `[DONE]`; model loading
  precedes the endpoint clock. This is neither process-cold first byte nor an
  OS-cold I/O claim. Each process exits cooperatively via SIGINT.
- The reverse pair waited for another worktree's Metal tests to exit. No timing
  was intentionally overlapped with that suite. The later controls nevertheless
  drift: retain every attempt and withhold stable effect authority above 5%.
- Frozen protocol required a named 128-output gain of at least 10% in both
  orders, equal outputs/cache accounting, and a Q4 dissimilar replay before
  promotion. Code passes the Q8 gain screen. The product regressions decide HOLD
  before spending GPU time on Q4; that promotion requirement remains unmet.

## Endpoint results

Two observations per arm, median milliseconds. Positive wall change is slower.
These are descriptive estimates, not confidence intervals.

| Request | First delta A / B | Full wall A / B | Wall change | A wall spread |
| --- | ---: | ---: | ---: | ---: |
| Fresh long, 2 output | 40961.585 / 40518.843 | 41265.164 / 40845.282 | -1.02% | 6.92% |
| Exact long hit, 2 | 87.990 / 94.948 | 379.114 / 397.959 | +4.97% | 11.90% |
| Warm blue, 2 | 1405.830 / 1457.439 | 1539.379 / 1780.900 | +15.69% | 1.97% |
| Warm sampled green, 2 | 1437.193 / 1440.013 | 1594.148 / 1791.899 | +12.40% | 5.97% |
| Warm code, 128 | 2241.950 / 2313.924 | 10502.141 / 6304.742 | -39.97% | 4.44% |
| Warm prose, 128 | 2487.299 / 2462.504 | 10867.391 / 12913.906 | +18.83% | 5.22% |
| Packed suffix, 2 | 1155.954 / 1065.562 | 1312.957 / 1407.087 | +7.17% | 17.60% |
| Fresh short, 3 | 1133.629 / 1127.710 | 1309.375 / 1307.893 | -0.11% | 0.36% |

Code wall saves 43.151% in A1/B1 and 36.919% in B2/A2. Its first-delta median
instead regresses 3.210%; this is a generation win, not a prefill win. Blue wall
loses 11.755%/19.548%, despite stable controls. Prose loses 16.909%/20.659%; its
5.225% control spread bars a stable percentage claim, but does not establish
safety or justify promotion. Sampled/packed/exact noisy rows carry no stable
effect authority. The packed suffix is also newly speculative, not an unchanged
decode guard. Fresh paths are unchanged; their fluctuations are not causal wins.

## Correctness and follow-up boundary

All 32 response texts/hashes, usage including cached tokens, completion statuses,
incomplete reasons, and errors agree by corresponding case. Subsequent turns
restore and finish successfully after both EOS and token-limit boundaries.
The candidate passes 101 serve CPU tests (two GPU tests remain ignored in that
command), including consumed-tail missing/excess/pending boundary cases.
This is a greedy endpoint regression witness, not proof of bitwise persistent
state equivalence. The forced-green sampled case is not distributional proof.

Any reopening needs charged fallback/probe economics, not weaker margins or
prompt-content classification fitted to these two outputs. Price discovery and
recovery on short EOS, preserve the code win, and test dissimilar quantization
before admission promotion. Do not transplant the explicit-long fallback
controller merely because it already exists: it has a different cost envelope.

Raw requests, manifests, SSE events, server logs, host reports, frozen protocol,
and metric-only `score.py`/`summary.json` remain in the dedicated worktree at
`target/profiles/restored-dflash/`. Acquisition uses the tracked reusable
`scripts/serve/request_probe.py`; no new benchmark framework is added.
