# v0.389 Metal Counter Capability Probe

Added a lightweight `qwen-bench metal-counters` probe so future agents can check
Metal counter availability before planning a counter-driven branch. This was
triggered by the v0.388/cx recommendation to run a saturation audit before any
new route, attention, MoE, or GDN rewrite.

## Commands

```bash
xcrun xctrace list templates

xcrun xctrace record --no-prompt \
  --template "Metal System Trace" \
  --instrument "Metal GPU Counters" \
  --time-limit 10s \
  --output target/profiles/v0389-counter-smoke2.trace \
  --launch -- target/release/qwen-bench roofline \
    --stream-mib 64 \
    --fma-elements 262144 \
    --fma-iters 512 \
    --runs 2

xcrun xctrace export \
  --input target/profiles/v0389-counter-smoke2.trace \
  --toc \
  > target/profiles/v0389-counter-smoke2-toc.xml

xcrun xctrace export \
  --input target/profiles/v0389-counter-smoke2.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]' \
  > target/profiles/v0389-counter-smoke2-gpu-counter-value.xml

cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench
target/release/qwen-bench metal-counters \
  > target/profiles/v0389-metal-counters-release.out
```

## Results

`xctrace` still cannot provide useful hardware counters on this target:

```text
Run issues were detected (trace is still ready to be viewed):
* [Warning] GPU Service reported error: Selected counter profile is not supported on target device
```

The exported `gpu-counter-value` and `metal-gpu-counter-intervals` tables contain
schemas but no rows.

The in-process capability probe reports only timestamp sampling:

```text
device	Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes
sampling	stage=true	dispatch=false	blit=false
counter_sets	1
set	timestamp	counters=1	sample_buffer=ok
counter	timestamp	GPUTimestamp
```

## Decision

Do not plan the next performance branch around autonomous hardware counters from
`xctrace` or `MTLCounterSampleBuffer` on this M4 Max. The only in-process counter
set exposed here is `timestamp`, and dispatch-boundary sampling is unsupported.
This can support timing probes, not bandwidth, occupancy, stall, register-spill,
or hidden-traffic claims.

Keep `qwen-bench metal-counters` as a cheap capability check. If the next branch
truly requires hardware counters, it needs a manual Xcode GPU capture path or a
different external profiler; otherwise use the repo's phase, no-op, microbench,
trace-count, and roofline gates.
