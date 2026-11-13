# Serve GPU-thread QoS coverage: every blocking feeding path

**Decision: ship the new scopes.** V4.1's multi-agent serve now raises the
thread that feeds the GPU for every blocking prefill, CED, slice, solo step and
batched step, not only the opt-in sliced prefill. The shared GLM / Flash-Next
driver covers admission and plain prefill beside its existing decode scopes,
and their serial and DFlash paths get one burst scope each. A three-pair
candidate-vs-main campaign at 48K under real host load measured **+10.72%
tok/s**, 95% paired CI **[+10.45%, +10.99%]**, with ITL p99 and the worst gap
not worse. `QWEN_GPU_THREAD_QOS=0` remains the global rollback.

The mechanism is the one recorded for Flash-Next in PERF-LOG 2026-11-10
("Batched One-Ahead Loop Was Host-Scheduling Starved") and priced for V4.1 in
`2026-11-12-v41-pipelined-batch-price.md`: the command buffer finishes, then the
waiting thread is not scheduled promptly while unrelated host CPU work
saturates the machine. The guard is nested-safe and never lowers a thread
already at user-initiated.

Measured candidate `68559c950` (`perf/serve-qos-coverage`) against its branch
point `9c0dc6013` on M5 Ultra. Both are release builds of `qwen` and `qwen-tok`
from the same worktree layout. The branch point is the comparison arm because
other lanes advanced the shared `main` while this packet ran; the candidate
diff is the QoS coverage above. Artifact: four IQ2_XS GSQ-RCO shards plus the
disk-streamed IQ4_NL Engram companion.

## Audit: GPU-feeding serve paths

"Main" is the branch point; "candidate" is this commit. `yes` means one
`GpuThreadQos` scope encloses the blocking call, directly or through an outer
round/burst scope.

| Family | Path | Main | Candidate |
| --- | --- | --- | --- |
| V4.1 | Startup batched decode warm-up (`step`, `step_batch`) | no | yes |
| V4.1 | Serial plain prefill (`Forward::prefill` / `advance_prefill`) | no | yes (request burst) |
| V4.1 | Serial cold CED, non-layered (`cold_prefill_episode_slice`) | no | yes (request burst) |
| V4.1 | Serial layer-sliced prefill / cold CED (`prefill_layered`, `cold_prefill_layered`) | yes | yes (nested) |
| V4.1 | Serial solo decode (`Forward::step`) | no | yes (request burst) |
| V4.1 | Batched plain prefill and CED (`prefill_slice`) | no | yes (scheduler round) |
| V4.1 | Batched layer-sliced prefill / cold CED | yes | yes (nested) |
| V4.1 | Batched solo step (`step_one`) | no | yes (scheduler round) |
| V4.1 | Batched `step_batch` | no | yes (scheduler round) |
| V4.1 | Batched `step_batch_sampled` (device sampling) | no | yes (scheduler round) |
| V4.1 | Checkpoint capture / restore inside admission and finish | no | yes (round/burst) |
| GLM | Startup batch warm-up (`batch.step`, `forward`) | no | yes |
| GLM | Batched admission / live-session restore | no | yes (driver round) |
| GLM | Batched plain prefill (`prefill_run`) | no | yes (driver round) |
| GLM | Batched layer-sliced prefill | yes | yes (nested) |
| GLM | Batched `step_batch` / sampled step | yes (driver decode) | yes (nested) |
| GLM | Lone GPU token loop (`generate_gpu_tokens`) and `advance_group` | yes (driver advance/burst) | yes (nested) |
| GLM | Serial prefill, GPU token loop and single `forward` | no | yes (request burst) |
| GLM | DFlash capture prefill and speculative step loop | no | yes (packet burst) |
| Flash-Next | Startup native batch warm step | no | yes |
| Flash-Next | Batched admission / live-state restore | no | yes (driver round) |
| Flash-Next | Batched plain prefill | no | yes (driver round) |
| Flash-Next | Batched layer-sliced prefill | yes | yes (nested) |
| Flash-Next | Batched `step_batch`, `decode_group`, sampled step | yes (driver decode) | yes (nested) |
| Flash-Next | Lone/group GPU token loops (`generate_gpu_tokens`) | yes (driver advance/burst) | yes (nested) |
| Flash-Next | Serial prefill and `ordinary_step` loop | no | yes (request burst) |
| Flash-Next | DFlash cold packet prefill and step loop | no | yes (packet burst) |

Solo decode, batch rounds and the GPU token loops are the host-scheduling
cases the guard was built for. Plain prefill and cold/CED prefill are also
blocking command-buffer waits even though they are less frequent; they are
now covered by the same round or request scope. One scope per scheduler round
(for batched work) or per request/packet burst (for serial work) avoids a
syscall pair at every token, and an inner scope is a no-op when an outer one
already raised the thread.

## Implementation

- `backend_v41::scheduler::pump` enters one guard per round. That round runs
  plain prefill, CED, sliced prefill, solo steps, batched steps, device
  sampling, and the checkpoint work between them on the same thread.
- `backend_v41::execution::execute_with_engine` enters one guard per serial
  request (max-active 1), covering plain/sliced/CED prefill and the
  token-at-a-time decode.
- `batching::driver::pump` enters one guard per shared-driver round, covering
  admission and both plain and sliced prefill; its existing `decode` and
  `advance` guards become nested no-ops.
- GLM and Flash-Next serial `generate` bodies and their DFlash packet loops get
  one burst guard each. The two startup warm-ups are covered as well.
- Existing V4.1 layered, GLM sliced/group, Flash-Next sliced/group and the GLM
  library batch-loop guards remain in place as inner no-ops.
- No new environment knob. `GpuThreadQos::enter()` is default-on through
  `QWEN_GPU_THREAD_QOS`; `0` still disables every raise.

Two CPU tests record the thread's current QoS class inside scripted GPU
phases: a V4.1 scheduler round with plain and sampled batch steps, and a shared
driver round with plain prefill plus batch decode. Removing either outer scope
fails the test under the default-on flag; with the rollback set they observe no
raise.

## V4.1 gate

Three warm agents, production `max_active=4` with three sampler workers, 49,152
warm tokens per client, context 196,608, agent-default sampling, fixed plan
seed 20261109, 15 s ramp and 180 s measurement window. Each arm settled to
thermal pressure <= 1 for 60 s inside its own lease after warming; pressure 2
was allowed during the window and 3 aborted. Every arm loaded under the
canonical Metal lease, and the arm script checked that no other `qwen serve`
was resident after acquiring it. Order **A1 B1 B2 A2 A3 B3** (ABBAAB), pairs
`(A1,B1)`, `(A2,B2)`, `(A3,B3)`; A = main, B = candidate.

| arm | build | tok/s | ITL p50 / p99 / max (s) | append TTFT p50 / p95 (s) | thermal 0/1/2 s | max | load before -> after |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| A1 | main | 77.861 | 0.0310 / 0.0350 / 4.610 | 1.096 / 4.250 | 57.8 / 122.2 / 0.0 | 1 | 32.18 -> 29.22 |
| B1 | cand | 86.289 | 0.0281 / 0.0347 / 6.510 | 0.993 / 2.360 | 121.1 / 1.0 / 57.8 | 2 | 29.22 -> 24.37 |
| B2 | cand | 85.528 | 0.0283 / 0.0347 / 4.036 | 1.040 / 3.634 | 98.7 / 2.0 / 79.3 | 2 | 24.37 -> 19.27 |
| A2 | main | 77.244 | 0.0306 / 0.0420 / 6.037 | 1.003 / 3.268 | 127.3 / 52.7 / 0.0 | 1 | 19.27 -> 23.47 |
| A3 | main | 82.283 | 0.0289 / 0.0390 / 2.477 | 1.000 / 2.712 | 0.0 / 180.0 / 0.0 | 1 | 23.47 -> 9.21 |
| B3 | cand | 91.011 | 0.0279 / 0.0347 / 2.365 | 0.943 / 1.958 | 139.8 / 1.0 / 39.2 | 2 | 9.21 -> 24.52 |

Load is the 1-minute load average before -> after each arm from
`host-before.txt` / `host-after.txt`; `~/ghost` shadPS4/pack-store/merge
workers were active throughout (loads reached 32-36 in A1). Thermal seconds are
sample-and-hold inside the 180 s window.

| pair | order | tok/s change | ITL p99 ratio | worst-gap change | TTFT p95 change |
| --- | --- | ---: | ---: | ---: | ---: |
| A1/B1 | AB | +10.82% | 0.991 | +1.900 s | -44.5% |
| A2/B2 | BA | +10.72% | 0.827 | -2.001 s | +11.2% |
| A3/B3 | AB | +10.61% | 0.891 | -0.113 s | -27.8% |

Paired Student-t over three arm pairs (df 2): tok/s relative mean **+10.72%**,
95% CI **[+10.45%, +10.99%]**; absolute mean **+8.48 tok/s**, CI
**[+7.92, +9.04]**. The interval excludes zero by a wide margin, so the ship
gate passes on throughput alone. ITL p99 ratio mean is 0.903 (CI [0.699,
1.107]); worst-gap mean change is -0.071 s (CI [-4.917, +4.774]); TTFT p95
relative mean is -20.4% (CI [-91.4%, +50.6%]) with one pair +11.2%. Neither
latency metric regresses beyond noise.

Host load and thermal state moved between arms. Candidate arms B1-B3 ran at
lower or similar before-loads than their preceding main arms and touched
thermal pressure 2 for 39-79 s, while all main arms stayed at pressure 1. The
effect direction is therefore not a quiet-machine artifact, but the shared
host was not controlled. The three pair changes are within 0.22 percentage
points of each other, which is much tighter than the arm-to-arm load movement.

Output coherence: retained delta streams for every arm contain no replacement
characters and no repeated 8-word shingle above twice (30-42 files and 56-67K
characters per arm).

## Other family check: GLM

Same three-pair order and thermal protocol, 48,000 warm tokens, agent-default
sampling, 180 s window, release build, `QWEN_SERVE_GLM_MAX_ACTIVE=4`. The GLM
plan is `plan-glm.json` (seed 20261109, family `glm`).

| arm | build | tok/s | ITL p50 / p99 / max (s) | append TTFT p50 / p95 (s) | thermal 0/1/2 s | max | load before -> after |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| A1 | main | 84.561 | 0.0277 / 0.1734 / 0.6309 | 0.923 / 5.141 | 118.5 / 61.5 / 0.0 | 1 | 8.65 -> 9.98 |
| B1 | cand | 84.694 | 0.0278 / 0.1741 / 0.6174 | 0.869 / 5.164 | 110.3 / 26.2 / 43.5 | 2 | 9.98 -> 9.95 |
| B2 | cand | 92.133 | 0.0278 / 0.1396 / 0.5583 | 0.735 / 2.222 | 63.4 / 116.6 / 0.0 | 1 | 9.95 -> 14.41 |
| A2 | main | 84.311 | 0.0279 / 0.1750 / 0.6361 | 0.931 / 5.139 | 95.6 / 1.0 / 83.3 | 2 | 13.90 -> 16.70 |
| A3 | main | 86.994 | 0.0281 / 0.1732 / 0.7373 | 0.928 / 5.092 | 123.5 / 1.0 / 55.5 | 2 | 16.70 -> 23.23 |
| B3 | cand | 84.228 | 0.0278 / 0.1752 / 0.6316 | 0.947 / 5.216 | 103.0 / 1.0 / 76.0 | 2 | 23.23 -> 13.96 |

| pair | order | tok/s change | ITL p99 ratio | worst-gap change | TTFT p95 change |
| --- | --- | ---: | ---: | ---: | ---: |
| A1/B1 | AB | +0.16% | 1.004 | -0.013 s | +0.43% |
| A2/B2 | BA | +9.28% | 0.798 | -0.078 s | -56.8% |
| A3/B3 | AB | -3.18% | 1.012 | -0.106 s | +2.43% |

Paired Student-t over three arm pairs (df 2): tok/s relative mean **+2.09%**,
95% CI **[-13.93%, +18.10%]**; absolute mean **+1.73 tok/s**,
**[-11.86, +15.32]**. ITL p99 ratio mean is 0.938 (CI [0.636, 1.239]); worst-gap
mean change is -0.066 s (CI [-0.183, +0.052]); TTFT p95 relative mean is
-18.0% (CI [-101.5%, +65.5%]).

**Reading:** GLM does not reproduce the V4.1 effect consistently. Its worst
gap improves slightly in all three pairs and its p99 is neutral-to-better on
average, but the throughput interval includes zero and one pair regressed 3.2%
while another gained 9.3%. The host load was lower than the V4.1 campaign
(8.7-23.2) and moved between arms. The added GLM scopes are retained because
they close the plain-prefill/serial/DFlash coverage gap and the paired latency
pointers do not regress; no GLM throughput claim is made from this check.
Coherence was clean in every arm (no replacement characters, no 8-word shingle
above twice). Flash-Next received the same coverage but was not separately
gated; its paths use the same shared driver round and the same serial/DFlash
burst scope.

## Records

- Raw packet: `metagross:~/qwen-runs/2026-11-12-serve-qos/`, including
  `serve/{A1,B1,B2,A2,A3,B3}/summary.json`, `calls.jsonl`, retained event and
  request files, per-arm `host-before.txt` / `host-after.txt`, `gate.log`,
  `analyze_gate.py`, and the same layout under `serve_glm/`.
- Candidate commit `68559c950`; reference `9c0dc6013`.
- Builds and every model load ran under
  `QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease`; no lease was held
  longer than one arm.
- The aven note on `SRV-2K0Y` is `B4DV97X43WQZEEPE`.
