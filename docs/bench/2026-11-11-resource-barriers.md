# Probe 3a: resource-scoped barriers do not release the causal-wave floor

**Decision:** close the direct `memoryBarrierWithScope(.buffers)` to
`memoryBarrierWithResources:` replacement class. On both M5 Ultra and M4 Max,
resource-scoped barriers still make the short dependent chain wait for the
independent long kernel. Do not add a causal-executor flag or load a model for
the conditional Flash-Next A/B: the prerequisite overlap gate failed.

**What does overlap:** two command queues, with the long dispatch in one
command buffer and the correctly ordered chain in a serial encoder on the
other queue. Both are committed before either is waited for. This is a
microbenchmark result, not a measured model speedup.

## Workload and measurement

`dispatch_deadtime --resource-barriers [reps] [long_us]` adds an opt-in mode to
the existing `perf-ablations` example. It uses two research kernels with the
same dependent wrapping-u32 recurrence, one 128-thread group each. The long
kernel has separate input/output Metal buffers; the chain alternates two
other Metal buffers for **20 dispatches**. The long dispatch and link zero are
encoded together, then the remaining 19 links are separated by:

- full buffer-scope barriers;
- resource barriers naming exactly the chain's two buffers;
- no barriers, deliberately racing as an overlap-ceiling control.

Additional controls time the long kernel and the full-barrier chain alone,
and the safe two-queue arrangement. The long kernel's low occupancy leaves
capacity available for overlap. Every output word is compared bitwise with
a cached CPU integer oracle after completion. Only the no-barrier control
may fail that check. Initialization, oracle construction and validation are
outside the GPU timestamp window.

All durations use `MTLCommandBuffer.GPUStartTime/GPUEndTime`. There are no
counter samples between dispatches and no profiler-induced encoder splits.
For two queues, total GPU time is `max(end) - min(start)` across the two
command buffers; raw rows also contain each command duration, their start
skew and timestamp-interval intersection. This avoids claiming overlap from
the duration of just one queue.

Each process sustains GPU work for 500 ms before calibration, warms every
mode, then runs **121 repetitions per mode**, with a rotating forward/reverse
order. Targets are 350/400/450 us for the long kernel and approximately 180 us
for the entire chain. The program fails if the measured long-kernel median
falls outside 300–500 us. An initial M5 pass without sustained warmup drifted
below 300 us as clocks rose; it is retained as exploratory evidence and
excluded from the tables below. Cached oracles also remove CPU calculation
gaps between measured submissions.

Both builds use `nice -n 10 cargo build --profile iter`, targeting only this
example. M5 Ultra: macOS 27.0.1, Metal compiler 32023.921; M4 Max: macOS 15.7.9,
Metal compiler 32023.864. Both use Rust 1.99.1. Neither machine reported a
thermal/performance warning. M5 disk and memory were checked before building;
about 1.3 TiB was available. No model was loaded. GPU work used the queued
Metal lease with five-minute declared holds; actual campaigns lasted seconds.

## Results

GPU command time, median us, at the 400 us target:

| Mode | M5 Ultra | M4 Max | Bitwise repetitions, each device |
|---|---:|---:|---:|
| Long alone | 387.38 | 395.87 | 121/121 |
| 20-link chain alone, full barriers | 164.12 | 162.63 | 121/121 |
| Long + chain, full buffer barriers | 545.83 | 549.75 | 121/121 |
| Long + chain, chain-resource barriers | 545.75 | 549.75 | 121/121 |
| Long + chain, no barriers (unsafe) | 388.00 | 395.87 | **0/121** |
| Two queues, serial chain (safe) | 388.50 | 400.88 | **121/121** |

Resource barriers leave the same exposed chain tail as full barriers. The
paired per-repetition percentage improvement `(full - resource) / full` is
**-0.023%** on M5 (95% bootstrap interval **-0.100%..+0.038%**) and **+0.008%**
on M4 (**-0.023%..+0.053%**). The ratio of separate medians can differ slightly
from the median paired percentage. Bootstrap resamples 121 repetition pairs
4,000 times with seed 311; these intervals describe within-campaign variation,
not independent-machine reproducibility.

The timing benefit from removing barriers comes with incorrect chain values
on every repetition. Two queues retain exact chain semantics and achieve
**28.83%** lower microbenchmark GPU time on M5 and **27.10%** on M4 versus full
barriers. On M5, the second command's median duration is **150.50 us**, the
interval intersection is **149.88 us**, and start skew is **3.25 us**. On M4,
those values are **153.29/148.37/5.00 us**. The median per-pair fraction of the
chain command interval overlapping the long command is **100% / 96.74%**
(M5/M4). Medians of separate quantities need not form an exact identity.

The other long-kernel targets reach the same decision:

| Device | Target us | Actual long us | Full us | Resource us | Safe two-queue us | Paired resource improvement |
|---|---:|---:|---:|---:|---:|---:|
| M5 Ultra | 350 | 341.13 | 503.96 | 503.79 | 342.12 | -0.033% |
| M5 Ultra | 450 | 438.62 | 601.58 | 601.63 | 439.50 | -0.021% |
| M4 Max | 350 | 346.88 | 502.71 | 502.67 | 351.50 | -0.008% |
| M4 Max | 450 | 445.63 | 597.12 | 597.25 | 450.63 | 0.000% |

All ordered modes are bitwise in every sweep; all no-barrier chains fail.

## Executor consequence

This result agrees with the SDK contract for
`memoryBarrierWithResources:count:`: previous dispatches must complete, while
the resource list identifies the memory whose side effects are made visible.
Apple also describes a memory barrier as making subsequent commands in the
pass wait for previous commands to finish in
[Go bindless with Metal 3](https://developer.apple.com/videos/play/wwdc2022/10101/).
Narrowing that list does not make this API a per-resource dispatch scheduler.

Changing `dependency_waves` into per-node resource barriers would therefore
keep the long-kernel stall and could sacrifice existing within-wave overlap.
Wave-local resource lists do not remove the demonstrated synchronization
floor either. There is no supported performance case here for that executor
prototype, so no production executor code, environment flag, or model A/B was
added. The bitwise and 3% **model wall-time** gate remains unattempted.

The measured alternative is to place independent work on separate command
queues. Applying that to a real graph would need explicit inter-queue ordering
of all overlapping RAW, WAR and WAW spans, resource lifetimes through both
completions, and exact causal-scope entry/exit semantics. Keep disjoint spans
distinct in the dependency analysis even though this barrier API takes whole
Metal resources. Queue/submission/synchronization costs and actual independent
model work must be priced before predicting a model win. Kernel fusion that
removes dispatch boundaries is another architectural direction, unmeasured
by this probe. Reachable-limit prices under the current wave executor remain
conditional on that execution model.

## Reproduction and evidence

```sh
nice -n 10 cargo build --profile iter -p qwen-llm \
  --features perf-ablations --example dispatch_deadtime
QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease \
  --task resource-barriers --for 5m -- \
  nice -n 10 target/iter/examples/dispatch_deadtime --resource-barriers 121 400
```

Use 350 and 450 in the last argument for the companion sweeps. The default
example invocation retains the existing dispatch-deadtime shape sweep.

Raw evidence: `metagross:~/qwen-runs/2026-11-11-resource-barriers/` and the same
run-directory name on the M4. Final files are `m{4,5}-{350,400,450}-final.jsonl`;
`analysis.jsonl` and `analyze.py` contain paired analysis and validation. Source
SHA256 manifests match between devices; the original base commit plus
`measurement-source.patch` preserve the measured source. Only the source,
logs and measurements are retained; lane build directories and worktrees are
removed after completion. The record is landed on `main`; the measurement
branch `research/resource-barriers` is kept at `b3c1d0628`, based on
`0b4b75094`.
