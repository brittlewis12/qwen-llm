# Flash-Next HC persistent phase-sync and site price

**Decision: close the persistent HC site experiment at the site gate.** Device-
scope phase handoff is cheaper than a dependent dispatch on both chips, and the
M5 contention torture remained bitwise exact. The real-weight site kernel is
far slower than the incumbent on both devices. No production kernel or rollback
environment switch was added; the whole-path B=1 A/B, three-agent serve, and
GLM/V4.1 ports were not entered.

The bench-only code is on `research/hc-persistent` at `050ceeb15`; the phase
sync probe is at `92e608d43`. Both examples require `perf-ablations`. Builds and
GPU runs used `nice -n 10`, `--profile iter`, and queued Metal leases. M5 source
was transferred by bundle into its own worktree; `main` was not changed.

## Stage 1: device-scope phase handoff

`dispatch_deadtime --phase-sync` compares 1,024 task-ordered threadgroup
handoffs inside one dispatch with 1,024 dependent dispatches. Each task reads
the prior task's payload. It uses coherent device buffers, relaxed atomics, and
device-scope sequentially consistent fences. The contention arm runs 512 groups
of 128 threads with 8,192 integer rounds per lane. MSL 3.2 runtime compilation
succeeded on both devices; all outputs were checked bitwise over 31 repetitions.

| Device | Handoff us, unloaded | Dispatch us, unloaded | Handoff us, loaded | Dispatch us, loaded |
| --- | ---: | ---: | ---: | ---: |
| M4 Max | 0.808 | 1.619 | 1.049 | 2.190 |
| M5 Ultra | 1.012 | 1.439 | 1.026 | 1.568 |

The M5 contention pass checked 63,426 cross-threadgroup edges with exact
payloads. The M4 pass checked the same count. The M5 Metal device does not
expose a physical die ID per threadgroup, so the test confirms cross-threadgroup
handoffs under Ultra contention but cannot tag individual edges as crossing
dies. GPU power telemetry was unavailable without elevated privileges; the
unprivileged `powermetrics` request was denied.

## Stage 2: real block-0 site

The candidate runs norm, BF16 down/SiLU, F32 injection preparation, and BF16
up/gate-mix in one task-ordered MSL 3.2 dispatch. A separate incumbent
residual-injection dispatch remains after the site. Norm branches and down
rows are independent tasks; a device counter releases the up phase when all 320
down rows finish. Each up worker preloads BF16 weight fragments into registers
while down is active, then handles multiple output tiles. The run uses the
actual block-0 GSQ-RCO IQ3_S GGUF tensors: 40,960-byte F32 norm, 6,553,600-byte
BF16 down, 6,553,600-byte BF16 up, and 163,840-byte BF16 injection widened to
F32 exactly as at runtime.

The M5 run passed the probe's pre-load check for any other `qwen` model process.
Its 256-worker paired medians were:

| Measurement | Incumbent | Persistent candidate | Candidate speedup |
| --- | ---: | ---: | ---: |
| Site, us | 18.875 | 51.625 | 0.366x |
| 47-site chain, ms | 0.786 | 2.355 | 0.334x |

The requested limits were at most 25 us per site and at least 1.5x for the
47-site chain. Both fail decisively. The 47-site probe repeats the real block-0
weights at each site; it is a dependency/dispatch price, not a full-model run
with 47 distinct layer tensors. All compared intermediates, mixed outputs, and
residuals were bitwise equal. Thermal pressure was 0 at the start and end.

M4 also missed with both worker counts. With 64 up workers, site medians were
24.375 -> 60.458 us and the 47-site chain was 1.068 -> 2.251 ms. With 256,
they were 32.791 -> 50.625 us and 1.355 -> 2.029 ms. Outputs stayed bitwise
equal in both paired runs.

The primitive wins in isolation do not carry through this site mapping: the
task dispatch and work distribution outweigh the removed dispatch boundaries.
That points the remaining work toward a better node mapping or less work, not
more synchronization tuning. This is an inference from these probes; no full
model wall-time result is claimed. Since the Flash-Next site gate failed, no
whole-path A/B, three-agent serve, or GLM mHC/V4.1 HC port was run.

## Run conditions and artifacts

M5 had concurrent `~/ghost` activity, left untouched. A pre-run snapshot at
thermal pressure 0 recorded two `shadps4` workers at 390% / 412% CPU and about
6.5 / 2.9 GB RSS; the full snapshot is retained with the M5 raw output. The
site result itself reports thermal 0. The M5 Qwen process preflight succeeded
before its GGUF load.

- M4 raw phase and site runs: `qwen-runs/2026-11-12-hc-persistent/m4/`
- M5 raw runs and host-load snapshot: `metagross:~/qwen-runs/2026-11-12-hc-persistent/m5/`
- M5 build, site probe, and GGUF load were inside the queued 12-minute `hc-persistent` lease.
- No per-sequence scheduling changes, production kernel, environment switch, or roadmap edit was made.
