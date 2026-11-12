# Two-queue causal branches: ceiling passes, isolation does not

**Close this prototype at the isolation gate.** GLM has the largest
demonstrated scheduling ceiling, **1.111 ms/token / 6.65% of token wall**,
including device-event costs. The implemented partition preserves **bitwise
logits and persistent state**, but 48 affected fork groups improve only
**1.112x** in isolation (**4.813 -> 4.328 ms GPU**), below **1.25x**.
No token-wall saving is claimed. The conditional three-pair wall A/B,
default-on rollout and serve validation were not entered.

Branch `research/two-queue-causal` was rebased onto main `932f3b729` after the
barrier record landed as `76ad233a5`; its final prototype code is `25a35a756`.
The record and reusable measurement tooling are landed on main. The two-queue
execution prototype itself is not: it remains only on that branch. Execution
changes were research-only under `perf-ablations`; ordinary builds retain
their policy.

## Event handoff and return join

`dispatch_deadtime --event-sync REPS [event|shared|both]` sends actual dependent
data from queue 0 to queue 1 and back. Each round has two tiny 128-thread
kernels, a signal/wait in each direction, and a final queue-0 consumer that
checks the joined result bitwise. Controls execute the same work on one queue,
in one serial encoder or in separate encoders. Both command buffers are
committed before waiting. GPU time is the envelope across their
`GPUStartTime`/`GPUEndTime` intervals; CPU allocation/encoding is reported
separately. All outputs are bitwise.

Fits over 1/4/16/32/64 roundtrips remove fixed startup. The event-only campaign
has 121 interleaved repetitions per depth and a 500-ms GPU warmup.

| Device | `MTLEvent`, added us/handoff | Added us/roundtrip | `MTLSharedEvent`, added us/handoff |
|---|---:|---:|---:|
| M5 Ultra | 4.63 | 9.26 | 222.03 |
| M4 Max | 4.39 | 8.78 | 71.03 |

These are amortized additions versus the single-encoder control, including
encoder transitions and the return join. Short campaigns also expose startup
and clock effects: the paired 64-round estimate is 4.60 us/handoff on M5 and
4.52 on M4. The schedule model charges **4.6 us per cross-queue handoff**, also
at the final join. Secondary-buffer submission and host encoding still need
a real wall-time gate. Shared-event numbers come from the initial 61-rep
interleaved campaign. Apple recommends device-local events for queues on one
device in [About synchronization events](https://developer.apple.com/documentation/metal/about-synchronization-events).

## Node capture and schedule price

Existing M4/M5 run directories and bench notes contain wave membership and
dispatch profiles, but not the exact span-derived graph paired with
unambiguous node service times. New `nodes` modes export kernel/grid/group,
buffer spans, all RAW/WAR/WAW predecessors, original ASAP waves, and GPU
start/end timestamps. Final captures cover **four consecutive forward rounds
near 32K** and match ordinary logits bitwise. Reference wall is three
uninstrumented runs from the same prefix: 32 emitted-token GPU loops for
Flash-Next/GLM, divided by actual forwarded tokens; eight rounds for V4.1.

**The first capture was not isolated:** independent encoder passes overlapped,
and sampled windows could include dependency drain. Those runs are provisional
and excluded. Final node mode encodes a same-queue `MTLEvent` signal/wait
outside every sampled encoder. An M4 GPU regression checks a long independent
dispatch plus a dependent chain, nonoverlapping windows and exact output words.
The analyzer requires the isolation marker and rejects overlapping windows.
Node durations exclude intervening event commands. Legacy wave-mode spans
remain diagnostic windows, not ordinary wall attribution.

The analyzer computes `sum(max(node duration))` over today's ASAP waves. It
also reconstructs cross-scope conflicts from actual spans for a global DAG
critical path, resetting between token rounds. Two-queue partitions use
concurrent batches: a batch lasts for its slowest node; each queue orders its
batches; cross-queue consumers wait for the producing batch plus 4.6 us.
Queue 0 supplies entry and consumes the join. Partitions retain existing
causal scopes. Every solution is independently checked against the unreduced
graph; original waves are a feasible fallback. The integer solver is
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver),
version 9.15.6755.

All table entries are **ms per output token**. B=3 batch-round times are three
times its row. DAG includes event costs at the selected placement but relaxes
artificial scope joins; the partition preserves those joins.

| Family / path | Reference wall | Wave sum | Global DAG + sync | Two-queue partition | Saving vs waves | % wall |
|---|---:|---:|---:|---:|---:|---:|
| Flash-Next B=1 GPU loop | 12.009 | 12.024 | 11.616 | 11.616 | 0.408 | 3.39% |
| GLM B=1 solo kernels, hypothetical causal retrofit | 16.721 | 16.145 | 15.034 | 15.034 | 1.111 | 6.65% |
| V4.1 B=1 `Forward::step` | 13.990 | 13.211 | 12.630 | 12.737 | 0.474 | 3.39% |
| V4.1 B=3 | 9.200 | 7.601 | 7.088 | 7.149 | 0.452 | 4.91% |

The scope partitions are proved optimal for all 196/184/172 captured
Flash-Next/GLM/V4.1 B=1 scopes. B=3 has 168/172 scopes proved; the remainder
leaves **0.0134 ms/batch round**, under **0.05% of round wall**, between the
best solution and bound. Its scope-preserving ceiling is **4.91%..4.96%**.
These are constant-node-duration scheduling prices, not measured concurrent
model times; wave sums need not equal uninstrumented GPU or wall time.

Even with *free* sync and scope boundaries removed, global ceilings are
Flash-Next **0.408 ms / 3.39%**, V4.1 B=1 **0.585 ms / 4.18%**, and V4.1 B=3
**0.521 ms/output token / 5.67%**. GLM's constructible 1.111 ms/token ceiling is
larger than those bounds and passes stage 1.

**GLM path conflict:** production B=1 uses serial `forward` and the GPU token
loop, not the causal batch executor. Its wave column is explicitly a retrofit
of the same solo kernels. The prototype therefore retains separate incumbent
serial, single-queue causal and two-queue controls, so a causal retrofit gain
cannot be counted as a two-queue win.

## Independent branches

| Family | Work that is independent before the join |
|---|---|
| Flash-Next | GDN controls versus Q/K/V; QSA index scoring/selection versus query/KV preparation; shared versus routed FFN after the common projections. Attention consumes query plus selection. |
| GLM | KDA controls versus Q6 Q/K/V; MLA index key/query/score/selection versus query/latent preparation; shared versus router/routed experts. Selected attention and FFN addition are joins. |
| V4.1 | Query projection/RoPE versus index heads and radix selection; shared gate/up/down versus router/routed gate/up/down. Flash attention and FFN post controls join the paths. |

HC pre-reduction, finish, mixer result and post-update mostly form the serial
spine. Different HC kernel names do not by themselves imply independent
streams; actual span conflicts decide every legal move.

## Prototype gate

`glm_solo_bounds MODEL 32768 32 twoqueue PLAN.json`, with
`QWEN_CAUSAL_TWO_QUEUE=1`, loads an emitted plan. Actual current spans certify
node coverage, local ordering and absence of cross-queue cycles before any
branch is encoded. Invalid/unmatched scopes use normal waves. Events and the
second queue persist. Local batches keep buffer barriers; encoders end at
event boundaries. Secondary buffers remain uncommitted until their parent is
committed; pending parents are weak references. All submitted branches finish
before CPU readback.

The checked step applies **45 scopes**, with zero rejected scopes; the head
has no parallel plan. Both waves and two queues are bitwise against serial
logits and the complete persistent snapshot. An M4 GPU fork/join test and CPU
coverage/cycle tests pass.
The prototype attaches to synchronous B=1 `forward`; GPU-loop submission
integration belongs to the subsequent wall gate, which was not reached.

The isolation harness rotates layer weights across **48 fork groups**,
alternates A/B order for **nine sweeps**, and restores every written range
outside GPU timing. All written bytes, including cache and status regions,
match. The explicit entry and return join place secondary work inside the
timed primary command interval.

| Affected cohort | Full-barrier us | Two-queue us | Speedup |
|---|---:|---:|---:|
| All 48 groups | 4812.71 | 4327.58 | **1.112x** |
| KDA controls | 2970.08 | 2379.71 | 1.248x |
| MLA query/index | 1405.67 | 1448.42 | 0.970x |
| Shared/routed FFN | 438.50 | 447.87 | 0.979x |

Rows are separate medians, so components need not add to the aggregate median.
The static-duration ceiling does not survive real queue execution well enough
to clear 1.25x. This probe does not separate resource competition from
scheduling/cache synchronization costs. **Stop at this gate:** no full-wall
A/B, default change or serve test is warranted for this candidate.

## Reproduction and evidence

Build only the needed examples as below. GPU work uses the canonical queued Metal
lease; every M5 build was leased too, with declared holds at most ten minutes.
Final preflights reject other model executables regardless of RSS and defer
thermal pressure >=2. An earlier RSS-only check missed large Metal allocations;
the provisional B=3 run with high wired memory is excluded and replaced by a
clean four-round capture.

```sh
nice -n 10 cargo build --profile iter -p qwen-llm --features perf-ablations \
  --example dispatch_deadtime --example fn_solo_bounds \
  --example glm_solo_bounds --example v41_batch_depth

# Under scripts/bench/with-metal-lease --task two-queue --for 10m:
target/iter/examples/dispatch_deadtime --event-sync 121 event
target/iter/examples/fn_solo_bounds "$FN_MODEL" 32768 5 nodes
target/iter/examples/glm_solo_bounds "$GLM_MODEL" 32768 4 nodes
target/iter/examples/v41_batch_depth "$V41_MODEL" 32768 1 4 nodes
target/iter/examples/v41_batch_depth "$V41_MODEL" 32768 3 4 nodes

# CPU-only:
uv run --python 3.13 scripts/bench/two_queue_ceiling.py --self-test
uv run --python 3.13 scripts/bench/two_queue_ceiling.py \
  --sync-us 4.6 --solve-seconds 10 --emit-plan plan.json glm-isolated-nodes.jsonl

# Prototype-only, from research/two-queue-causal @ 25a35a756:
# Under the Metal lease; research flag only:
QWEN_CAUSAL_TWO_QUEUE=1 target/iter/examples/glm_solo_bounds \
  "$GLM_MODEL" 32768 32 twoqueue plan.json
```

Raw: `metagross:~/qwen-runs/2026-11-11-two-queue/` and the matching M4 run
directory. Use only `*-isolated-nodes.jsonl` for node timing;
`*-final-schedule.json` contains analyzed partitions. The prototype result is
`glm-twoqueue-isolated.jsonl`. Event captures, source patches/manifests, memory
preflights, build/test logs and the plan are retained. Lane bundles, M5
worktree and build output are removed after verification.
