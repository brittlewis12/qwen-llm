# Xcode/Instruments decode capture packet (long-context attention + MoE routed)

Purpose: convert the two standing byte-PROXY claims into MEASURED limiter data.
This is the twice-endorsed, never-scheduled epistemics unlock (see
PERF-ROADMAP "Current North Star" notes). One interactive ~15-minute session
answers questions that three falsified branches (read-once v0.405, coop4
v0.313, KV-Q8 v0.437) each spent days guessing at.

## The three questions

1. **Decode attention KV-read limiter** (A3B/27B at ctx >= 16384): the v4
   main body sustains only `~282-296 GB/s` of KV traffic vs `474 GB/s`
   stream (v0.403), and attention is `26-38%` of the decode phase slope.
   Is the kernel limited by: memory-system latency (occupancy too low to
   cover DRAM), memory bandwidth actually saturated at the SLC/DRAM level
   (proxy wrong), ALU (Phase A reduction chain), or threadgroup sync?
2. **MoE routed projection limiter** (A3B decode): routed gate/up/down
   mat-vecs put MoE decode at `~51%` of stream — the biggest raw gap on
   the board. Occupancy? Expert-scatter latency? Row underfill?
3. **Proxy calibration**: per-kernel achieved DRAM bytes vs our byte-proxy
   models (PERF-LOG entries assume proxies; one measured table retires the
   standing caveat).

## Launch (terminal side)

```bash
scripts/profile/xcode_capture_decode_window.sh a3b 16384   # or: 27b 16384
```

The script starts `qwen-bench decode-window`, ramps to the target context,
prints the PID, and BLOCKS at the ready-file. Attach the profiler, then:

```bash
touch /tmp/qwen-capture.go   # releases a 256-token decode window
```

## Capture (Instruments path, primary)

1. Instruments > "Metal System Trace" template; add the "GPU" (GPU
   Counters) instrument if not present.
2. Attach to the printed PID (All Processes -> filter `qwen-bench`).
3. Start recording, `touch /tmp/qwen-capture.go`, stop after ~3-5 s
   (window prints stats and exits).
4. In the GPU track, select a steady-state slice (skip the first ~10
   tokens).

## Capture (Xcode GPU capture, secondary)

`MTL_CAPTURE_ENABLED=1` is exported by the launch script; from Xcode:
Debug > Attach to Process by PID, then Metal frame capture (camera icon)
with "Capture Scope: Command Queue", trigger, and `touch` the go-file.
One capture holds a handful of decode iterations — enough for per-kernel
counters; use Instruments for timeline-level questions.

## What to record (per kernel, steady-state)

| kernel | occupancy (simdgroups/core) | limiter (top 2, %) | DRAM GB/s | ALU % | notes |
| --- | --- | --- | --- | --- | --- |
| `kernel_attn_decode_v4_*` (active variant at this ctx) | | | | | Phase A vs B split if visible |
| `kernel_gdn_step_decay*` | | | | | state r/w efficiency |
| routed `kernel_moe_*swiglu*` (A3B) | | | | | |
| routed down / `mat_vec_q6_K` (A3B) | | | | | |
| `kernel_mat_vec_q4_K*` (dense FFN, 27B) | | | | | reference row: known-good ~78% stream |

Also grab: whole-window achieved DRAM bandwidth (GPU Counters summary) and
the occupancy timeline dip pattern (does attention UNDERFILL between
denser phases?).

## Interpretation guide (pre-registered, so we do not rationalize after)

- Attention limiter = "memory latency"/low occupancy => the live branch is
  occupancy-shape work (smaller partials footprint, more concurrent
  threadgroups), NOT byte reduction; KV-Q8-class ideas stay dead.
- Attention limiter = bandwidth saturated at SLC/DRAM >= ~90% => the
  282-296 GB/s proxy was mis-modeled; update the proxy and CLOSE the
  attention-slope branch (it is already at the wall).
- Attention limiter = ALU => the Phase-A reduction-shape fence
  (byte-reduction rule) gets re-litigated with evidence.
- MoE routed limiter = latency/occupancy => batching economics (v0.401)
  get re-priced with the measured occupancy headroom; if bandwidth-bound
  at the wall => the 51% figure was proxy error, close the "biggest raw
  gap" framing.

## Results — autonomous xctrace pass (2026-07-03, A3B ctx16384 w400)

What the CLI path could and could not get:

- `xcrun xctrace record --template 'Metal System Trace' --attach PID`
  works end-to-end with the launcher (note: touch the go-file AFTER the
  ready-file appears; a pre-created go-file is ignored by the waiter).
- Per-kernel shader profiling and `Metal GPU Counters` tables export
  EMPTY under both --attach and --launch (the instrument records no
  samples without an Instruments-UI counter-set configuration). The
  per-kernel limiter cell (q1/q2) still needs the interactive session.

Timeline-level findings (from metal-gpu-execution-points +
application-command-buffer-submissions, id-interned XML export):

- Each decode token = ONE command buffer executing as EXACTLY 3 serial
  GPU kicks (400/400 tokens): med 2.89 / 3.63 / 3.40 ms = 9.9 ms GPU.
- Intra-token kick gaps are ZERO (3 us/token total): no scheduling
  stalls inside a token at kick granularity. The GPU held Maximum
  performance state through the window (no downclock story).
- ALL idle is inter-token. Clean (untraced) run: 0.68 ms/token gap =
  ~6% of the 10.94 ms period (gpu/total 93.8-95.7%, 91.4 t/s). Under
  active tracing the gap inflates to ~1.95 ms (84-87% busy) — tracing
  overhead lives in exactly this window; do not read traced busy
  fractions as production numbers.
- `--pipelined` prototype A/B at ctx16384: REGRESSES (10.94 -> 12.11
  ms/token; GPU time itself 10.26 -> 11.29 ms). The 6%-ceiling
  inter-token gap is not recoverable with this overlap shape.

Interpretation per the pre-registered guide: decode at long context is
GPU-kernel-bound with a clean pipeline at kick granularity; the
282-296 GB/s attention question is intra-kernel (occupancy/latency
inside attn_v4), so only the counter cell can crack it. The bench-loop
idle (~6%) is real but small and the cheap overlap idea is falsified.


## Results — headless limiter capture (2026-07-03, metal-counters template)

The one-time UI-saved template (`metal-counters.tracetemplate`: Metal GPU
Counters, Counter Set = Performance Limiters, Performance State = Maximum,
shader profiler on) unlocks fully headless counter capture:
`xcrun xctrace record --template 'metal-counters' --attach PID`. The
`gpu-counter-value` table exports 24M timestamped samples; join to the
exact kick timeline from `metal-gpu-execution-points`.

A3B ctx16384 decode, per-kick means (5.9-7.4M samples per kick):

| counter | kick0 | kick1 | kick2 |
| --- | ---: | ---: | ---: |
| Kernel Occupancy (%) | 28.8 | 28.3 | 29.4 |
| Compute SIMD Groups Inflight | 27.7 | 27.2 | 28.2 |
| GPU Read Bandwidth (GB/s) | 265.6 | 271.0 | 309.6 |
| Instruction Throughput Limiter (%) | 44.9 | 45.0 | 56.2 |
| ALU Utilization (%) | 13.9 | 14.3 | 19.0 |
| F32 / Int+Complex Limiter (%) | 13.6 / 21.0 | 14.1 / 21.2 | 14.0 / 30.9 |
| L1 Cache Limiter (%) | 7.8 | 7.9 | 8.4 |
| Buffer L1 Miss Rate (%) | 25.6 | 26.2 | 24.0 |

Device-level over the window: 268 GB/s (253 R + 15.5 W), Kernel Occupancy
25.4% vs Occupancy Manager Target 72.5%.

ANSWER to q1/q2 (pre-registered, wording per cx review): long-context
decode is primarily limited by POOR LATENCY HIDING / LOW EFFECTIVE
RESIDENCY, not by DRAM bandwidth or ALU saturation. All three kicks show
the same first-order symptom (occupancy ~28% vs manager target ~72%,
bandwidth 57-68% of stream, ALU pipes <= 31%, L1 <= 8.4%) — though the
root residency cap may still differ by kernel family. Instruction
Throughput Limiter (45-56%) and Int+Complex (up to 31%) are nonzero but
not dominant. Byte reduction is NOT FIRST-ORDER under this measurement
(it can still help via latency exposure / cache pressure, but bandwidth
is not the binding limiter).

DISCRIMINATOR RUN (cx-prescribed): default vs QWEN_ATTN_V4_NWG=192, two
independent captures, exact per-kick joins over 664/682 tokens. NWG192
changes NOTHING: Kernel Occupancy 28.3/27.9/29.0 -> 28.3/28.0/29.0,
SIMD Inflight 27.2/26.8/27.8 -> 27.2/26.9/27.8, kick durations
2.98/3.76/3.58 -> 2.92/3.66/3.54 ms (within run noise), t/s 85.1 -> 86.0.
More logical partitions do not raise resident work => the residency cap
is PER-KERNEL (registers / threadgroup-memory / occupancy shape), not
launch starvation. This also retro-explains the v0.404 NWG shelf.
Remaining secondary discriminator (not yet run): a two-stream /
batch-2 concurrency probe to bound how much independent work the device
would absorb.

Caveat: the shader-profiler per-kernel table is kick-sampling-biased
(routed-down mat-vec reads 52% of samples vs ~13% of known wall; GDN
kernels nearly absent) — use it for kernel NAMES, not shares. Per-kick
counter joins are exact; per-KERNEL counter attribution needs
finer-grained encoder labels (planned) or per-dispatch timestamp
sampling via the app-accessible GPUTimestamp counter set.

## Results — phase-noop limiter packet (v0.457)

Purpose: test whether the v0.455 low-residency signature is caused by one
family of work, and harden the analyzer for graph variants whose command-buffer
topology changes. These are attribution runs only; they are not throughput
claims because xctrace overhead is large.

Commands:

```sh
scripts/profile/gpu_limiter_capture.py capture --model a3b --ctx 16384 \
  --window 4000 --seconds 8 --label v0457-a3b-full

scripts/profile/gpu_limiter_capture.py capture --model a3b --ctx 16384 \
  --window 4000 --seconds 8 --label v0457-a3b-no-gdn \
  --env QWEN_DECODE_GDN_NOOP_FRONT=1 \
  --env QWEN_DECODE_GDN_NOOP_OUT=1

scripts/profile/gpu_limiter_capture.py capture --model a3b --ctx 16384 \
  --window 4000 --seconds 8 --label v0457-a3b-no-moe \
  --env QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP=1 \
  --env QWEN_DECODE_MOE_NOOP_ROUTED_DOWN=1

scripts/profile/gpu_limiter_capture.py capture --model a3b --ctx 16384 \
  --window 4000 --seconds 8 --label v0457-a3b-attn-residual \
  --env QWEN_DECODE_GDN_NOOP_FRONT=1 \
  --env QWEN_DECODE_GDN_NOOP_OUT=1 \
  --env QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP=1 \
  --env QWEN_DECODE_MOE_NOOP_ROUTED_DOWN=1
```

Counter summary:

| variant | tokens | kicks/token | kick medians (ms) | occupancy | read BW (GB/s) |
| --- | ---: | ---: | --- | --- | --- |
| full | 675 | 3 | `2.94/3.70/3.49` | `28.1/27.9/29.0` | `265/270/310` |
| no-GDN | 845 | 3 | `2.14/2.80/2.69` | `17.6/18.0/23.3` | `208/217/288` |
| no-MoE routed | 776 | 1 | `8.86` | `28.1` | `261` |
| no-GDN + no-MoE | 1051 | 1 | `6.36` | `17.5` | `203` |

Read:

- The noops give an approximate traced-token budget: full `10.13 ms`, no-GDN
  `7.63 ms`, no-MoE `8.86 ms`, no-GDN+no-MoE `6.36 ms`. The combined delta
  is roughly additive, but kick identity is not stable across variants.
- Removing MoE collapses to one large kick without improving occupancy; removing
  both GDN and MoE also leaves low occupancy and lower bandwidth. This argues
  against launch starvation and bandwidth saturation as the first-order limiter.
- Noop counter levels are not production family counters. They perturb topology
  and execution mix; compare total medians and broad limiter signatures, not
  kick0-to-kick0 values.
- The v0.455 conclusion stands: the live limiter is per-kernel residency /
  occupancy shape. Next steps are PSO/resource audit, batch-2/two-stream
  concurrency discriminator, then one surgical occupancy retune with a counter
  gate.


## Provenance (per cx review)

- machine: zekrom, M4 Max MacBook Pro, macOS 15.6.1 (24G90); xctrace
  26.0 (17C52); template `metal-counters.tracetemplate` (user templates
  dir; Metal GPU Counters, Counter Set = Performance Limiters,
  Performance State = Maximum, shader profiler enabled)
- workload: `qwen-bench decode-window -m Qwen3.5-35B-A3B-Q4_K_M.gguf
  --target-ctx 16384 --window 1200`, engine at main v0.454-era; window
  released post-ready; recording 8 s attach-mode ending BEFORE process
  exit (truncated-bundle pitfall otherwise)
- join: per-kick means over `gpu-counter-value` samples bucketed into
  exact kick windows from `metal-gpu-execution-points` (cmdbuf-id begin/
  end pairs, >2 ms intervals, 3-kick tokens only); SAMPLE-weighted
  (cadence approximately uniform; time-weighting is the upgrade path)
- shader-profiler names are METADATA ONLY (kick-sampling bias measured:
  routed-down mat-vec reads 52% of samples vs ~13% known wall share)
- raw traces on zekrom: /tmp/qwen-a3b-limiters.trace,
  /tmp/qwen-nwg-default.trace, /tmp/qwen-nwg192.trace (regenerate:
  launcher + the join scripts in this session's PERF-LOG entry)
