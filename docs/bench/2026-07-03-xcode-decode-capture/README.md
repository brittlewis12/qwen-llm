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
