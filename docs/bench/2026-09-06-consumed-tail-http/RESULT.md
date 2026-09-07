# Consumed-tail HTTP: INCONCLUSIVE / HOLD

## Decision

The complete A-B-B-A packet fails its frozen primary TTFT stability gate:
control spread 6.919% exceeds 5%. No retuning, repeat packet or retrospective
gate change. Full-request wall measurements are promising and independently
stable, but do not override the failed packet gate. Serving opt-in and its unused
runtime API are removed in separate commits; the prior test-only phase evidence
and landed Q8 optimization remain.

## Implementation and validation

- `6ed6032a` adds a separate consumed-extension lookup/restore type. Original
  logical pending-token matching is untouched. Lookup shares existing snapshot
  Arcs without new index entries or LRU changes; strict consumed extension and
  mismatched pending token are required. Restore validates model owner, fresh
  sequence, capacity, snapshot identity/ABI and vocabulary before mutation. Its
  report is consumed/consumed, non-exact, with no final logits or capture tail.
- `0c58d830` adds `QWEN_SERVE_CONSUMED_TAIL=1`, default off. Qwen36 dense27
  geometry/Q6_K output only, no drafter, greedy, prompt 8192-16384, suffix 7-32,
  at least 64 more consumed tokens than the pinned original lookup, original
  work >48 forwards, priced scratch <=128 MiB. Candidate admission, allocation
  and restore must all succeed before selection. Any failure drops candidate
  objects before the original admission/allocation/restore block executes.
- Twenty cache CPU tests pass, including unchanged logical matching, strict
  extension, identity, longest state, non-mutating lookup and shared retention.
  Six filtered CLI tests pass, including scope and injected transaction failures.
  Mock allocation/restore errors prove drop order, not real driver OOM behavior.
- The real API release oracle repeats prompt logits, per-layer numerical KV/GDN
  and 128-token greedy agreement from the earlier phase screen. Wrong consumed
  prefix, consumed-only exact request and same-pending request reject before
  position mutation. Old KV before 8860 is bitwise; minimum remaining state cosine
  0.9999990571. No bitwise full-state or distributional claim.
- Independent reviews cover API separation, serving transaction ordering,
  downstream exact/capture/speculation gates, and final disposition.

## Frozen endpoint packet

Same release binary `0c58d830` on both arms: A env=0, B env=1. Four disposable
processes, A1/B1/B2/A2, serial GPU. Installed Qwen3.6-27B-Q4_K_M, no drafter,
existing ten-request `single-chunk-vt/endpoint-requests.json` fixture. First token
is the first real SSE string delta; request wall ends at `[DONE]`. Model readiness
precedes request timing: these are not process-cold measurements.

Primary compact-prose requires >=25% TTFT and >=5% complete-wall savings in
both orders, <=5% control spreads. Fresh-long/fresh-short wall median regressions
must be <=3%, control spread <=5%. Gates are frozen before launch.

| Metric | A1 | B1 | B2 | A2 | Median A -> B | Paired savings | Control spread |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| TTFT ms | 1078.878 | 349.654 | 388.840 | 1009.057 | 1043.967 -> 369.247 | 67.591% / 61.465% | **6.919% FAIL** |
| Wall ms | 6741.648 | 5688.415 | 5913.744 | 6504.116 | 6622.882 -> 5801.080 | 15.623% / 9.077% | 3.652% |

Observed median wall savings: 12.409% / 821.802 ms. TTFT percentages are
directional, not stable release authority. Both candidate processes select only
compact-prose: restored 8987 versus baseline 8860, suffix 26, priced 46,502,992
bytes. No candidate denial, allocation failure, restore failure or fallback occurs.

All 40 responses match text/hash, input/output usage, status and terminal reason.
The sole usage difference is expected: compact-prose cached tokens 8860 -> 8987,
exactly 127 consumed tokens, excluding the unmatched pending newline. Later
sampled output agreement is a fixture witness, not distributional equivalence.

### All unchanged rows

Positive savings means faster. These controls are reported, not attributed wins.

| Row | TTFT A -> B ms | Wall A -> B ms | Wall savings | Wall control spread |
| --- | --- | --- | ---: | ---: |
| Fresh long | 38654.435 -> 38237.344 | 38746.081 -> 38326.229 | 1.084% | 1.292% |
| Exact hit | 41.660 -> 41.394 | 137.807 -> 136.623 | 0.859% | 2.085% |
| Blue | 913.144 -> 912.960 | 1002.665 -> 1008.735 | -0.605% | 0.039% |
| Compact code | 1143.903 -> 1147.275 | 6753.632 -> 6603.700 | 2.220% | 0.704% |
| Sampled green | 976.896 -> 972.409 | 1081.050 -> 1067.796 | 1.226% | 1.951% |
| Longer code | 1451.287 -> 1467.191 | 6967.885 -> 7074.601 | -1.532% | 2.785% |
| Longer prose | 1116.743 -> 1135.544 | 6627.806 -> 6590.621 | 0.561% | 1.619% |
| Packed guard | 980.210 -> 953.772 | 1084.522 -> 1049.748 | 3.206% | 1.182% |
| Fresh short | 735.834 -> 743.975 | 833.338 -> 842.298 | -1.075% | 2.391% |

Fresh guards pass. Longer-code forward pair regresses 3.019% (reverse 0.003%);
longer-prose TTFT forward regresses 7.027% (reverse improves 3.570%). These
unchanged-row movements are retained rather than hidden behind medians.

## Removal and artifacts

`359e9bf8` removes the serving candidate; `03b49394` removes the unused API.
Final production code returns to `8b605db8`. No default or opt-in consumed reuse
ships. No same-cell repeat follows this packet; return to the fresh/cold cost
frontier rather than letting a warm fixture dictate the global queue.

Raw: `target/profiles/consumed-tail-http/{PROTOCOL.md,score.py,summary.json,A1,B1,B2,A2}`.
The scorer writes all rows before rejecting the TTFT spread. Earlier API protocol
and oracle: `target/profiles/consumed-prefix-reuse/{API-PROTOCOL.md,q4-consumed-api-oracle.log}`.
Owned servers 57771,58035,58293,58554 all exit cooperatively with130. No remote push.
