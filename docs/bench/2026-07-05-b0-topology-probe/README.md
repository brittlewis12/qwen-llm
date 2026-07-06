# B0: Residency + Forward-Progress + Boundary-Drain Probe

Status: RUN COMPLETE. Design cx-signed (session `019f347b-c...`, two rounds;
four required changes applied verbatim). Program B gate 0 per the cx-signed
topology program. Results and verdicts below; full artifacts in
`target/profiles/topology-probe/topology-probe-dwellext.json` (run 2, with
dwell extension) and `/tmp/b0-full-run{,2}.log`.

## RESULTS (two full runs, quiet box, runs=5 medians)

Calibration (stable across runs): dependent-FMA 13.3 ns/iter; relaxed
device-atomic poll 96.7 ns/poll.

### Arm R - PASS (kill gate cleared in every reading)

- W32/lo/no-traffic: max_alive 1191-2311 across dwells/runs = 29.8-57.8
  TGs/core, ALWAYS >= 16/core kill line (worst reading has 86% margin).
  max_alive is churn-noisy run-to-run; `steady_p10` (alive count observed at
  entry by post-ramp TGs) is the dwell-stable censor: 717-722 TGs (~18/core)
  across 200 us - 25 ms dwells and both runs. Use p10 ~ 720 as the
  conservative co-residency planning figure for 32-wide kernels.
- Fill ceilings: W32 compute-bound ~1764 threads/core; W32 MEMORY-STALLED
  reaches ~2950 threads/core (~92 TGs/core) - the scheduler deep-fills only
  when TGs stall. W256 compute-bound reaches ~2630 threads/core. Hard caps
  at high width + register pressure: 960 TGs at W128, 480-482 at W256 =
  ~96 simdgroups/core both ways (architectural).
- Register pressure did NOT reduce W32 residency (TG-slot/scheduling limits
  bind first at one-simdgroup width); pressure effects only appear as the
  ~96 sg/core cap at W128/W256.

### Arm S - PASS (decisive), plus a binding memory-model finding

- Delivery 100% at 40/160/720 consumers x {tight, 25 us, 100 us} x
  {traffic off, on}; zero timeouts, zero corrupt reads, zero detected
  global-stall slots; filtered excess-over-cadence p99 <= 0.1 us (one poll
  unit) - ~250x inside the 25 us gate. Raw p99 tracks cadence exactly as
  predicted by the amended metric definition.
- Cross-object reorder probe: 0 stale reads in 6.4M no-traffic
  observations, but 2.2e-3 / 6.0e-3 stale rates under device traffic
  (runs 2/1; always nonzero under load). Relaxed data-then-flag publication
  is EMPIRICALLY UNSAFE on M4 Max under load; any future cross-TG protocol
  must be self-validating (payload-in-flag / checksummed slots).

### Arm D - P-D1 fires; P-D2 fails BOTH variants => B1a KILLED pre-build

- Uniform ladders (m in {4, 64, low-water} x W {32, 64} x K {8, 32, 110} x
  work {0, 5, 15, 30 us}): per-boundary cost spans 1.0-5.1 us with median
  ~2.5 us; the BINDING decode-realistic uniform rows sit at ~2-3 us, far
  below the pre-registered 8-25 us prediction band and the A0-derived
  ~18 us anchor (a few edge rows graze 5 us; the gate reads the binding
  rows, per the cx results review). P-D1 fires.
- Mixed narrow<->wide ladders (4 alternating with low-water cap): ~11
  us/boundary at K110/15 us - heterogeneity multiplies boundary cost ~3-5x
  but still stays under the anchor.
- Persistence recovery at decode-realistic stages (K110, 15-30 us):
  local_mem 3.8-18.3%, local_reg lower (in-register fusion does not help at
  these stage sizes), global -4.6..14.3%, ALL far below the 30% kill line
  for both dependency classes. `global_wide` - the actual B1a persistent-
  host shape (full low-water grid hosting narrow stages, idle TGs paying
  every barrier) - has NEGATIVE recovery everywhere (-2..-9%): the
  persistent host is slower than the serial ladder it would replace.
- Zero barrier aborts anywhere; all checksums bit-exact. The persistence
  MECHANICS work (co-residency, bounded barriers ~0.6-5.8 us, forward
  progress under compositor preemption); the ECONOMICS do not.

### Verdict and re-attribution (for the PERF-LOG and cx review)

Pre-registered kill semantics applied: P-D1 kill signal (uniform boundary
< 5 us) + P-D2 both-variant fail => Program B's persistence thesis is
falsified at the glue-ladder scale and B1a is killed BEFORE touching
production kernels. Dispatch boundaries cost ~2-4 us x ~330/token =
~0.7-1.3 ms/token (7-13% of the token), not the ~5.9 ms interior valley
mass. The valley majority is INTRA-dispatch under-parallelism: narrow
one-simdgroup dispatches structurally cannot fill 40 cores (a 4-TG glue
dispatch idles ~36 cores while it runs), and D-local proves fusion cannot
recover that (perfect in-register fusion of a 110-stage ladder recovers
<= 18%). The recoverable lane is WIDTH/PARALLELISM restructuring (wider
glue dispatches; concurrent encoding of independent narrow stages - the
hazard-tracked concurrent-encoder machinery already exists), not
persistence. This also closes the DFlash reopen condition recorded at
v0.443 (persistent-kernel verify structure), which was contingent on
Program B economics.

BUILD-TIME FINDING (memory-model smoke, cx-required first step): the Metal
toolchain (v17.3.7003, macosx SDK) accepts ONLY `memory_order_relaxed` on
device atomics. `memory_order_release/acquire/seq_cst/acq_rel` do not exist
as identifiers and `atomic_thread_fence` does not exist in any signature.
Consequences, per the signed contingency ("treat S/D-global as memory-model
probes before interpreting latency"):
  - Arm S epochs use a per-epoch slot ring where the PAYLOAD IS THE EPOCH
    VALUE (self-validating; no cross-object ordering assumed), plus a
    dedicated reordering sub-probe: producer writes atomic object A then
    atomic object B (both relaxed); consumers poll B then read A; the
    stale-A-after-fresh-B count is a measured cross-object reordering rate.
  - Arm D-global stage barriers are sense-reversing on relaxed atomics with
    `threadgroup_barrier(mem_flags::mem_device)` for intra-TG ordering of
    each TG's device writes; the pre-existing checksum-equality gate is the
    correctness detector for any visibility violation.
  - Arm D-local has no cross-TG waits and is immune - the cx-required
    variant split is what keeps a relaxed-only memory model from blocking
    the whole arm.

## Why this exists (evidence base, do not re-derive)

- v0.455: A3B ctx16384 decode is latency-bound: Kernel Occupancy ~28% vs
  manager 72% target, 57-68% of stream BW, ALU pipes <= 31%.
- v0.491 (A0): the 28% is a duty-cycle artifact. Kicks start at 56-58%
  occupancy, burst to 60-70%, and spend ~64/64/43% of interior 25 us bins
  below half their own p90 in RECURRING valleys, no tail decay. Verdict H3:
  inter-dispatch drain/serialization between ~110 short dependent dispatches
  per kick. Program A (TG packaging) demoted to confirmatory; Program B
  (persistence/fusion across dispatch boundaries) evidenced.
- v0.492: the drain is not compositor interference (enrichment 1.15x < 2x
  gate; ~3% of shortfall), and trace-based valley-to-dispatch alignment is
  triply impossible on M4 Max (opaque compute channel; one interval per kick
  since each kick is a single encoder; v0.460 label/perturbation falsifiers).
  The drain question must be answered CAUSALLY inside a controlled workload.
  Ambient fact: WindowServer preempts ~once per 8.3 ms for ~400 us of
  depressed occupancy - headless does not mean exclusive.
- Metal dependency dichotomy (metal.rs:869-908): serial encoders barrier
  every dispatch against the previous one (full drain+fill per boundary);
  concurrent encoders offer only all-or-nothing encoder-level barriers.
  There is no finer-grained cross-dispatch dependency primitive, so
  persistence/fusion is the only mechanism that can overlap dependent work
  across what are dispatch boundaries today.

Arithmetic anchor for what is at stake: interior valley time is ~5.9 ms of a
~10.3 ms token (56.3% pooled of 2.98+3.76+3.58 ms kicks). At ~330 dependent
dispatches/token that is ~18 us per boundary. If valleys filled to burst-level
occupancy, the recoverable ceiling is roughly 3-3.5 ms/token (~30%). B1a's
pre-registered gate (>= 25% reduction on the ~2.6 ms glue/route family) is
consistent with this anchor.

## Questions B0 must answer (and B1a may not proceed without)

1. RESIDENCY: how many TGs are simultaneously resident, as a function of
   thread width and register pressure - is a persistent grid covering real
   work shapes even schedulable? (We have no register counts from any tool,
   v0.458; the probe is the oracle.)
2. SIGNALING: is bounded one-way cross-TG signaling through device memory
   reliable and fast enough to replace dispatch boundaries, under device
   traffic and ambient compositor preemption?
3. BOUNDARY DRAIN (new arm, from v0.492): what does a serial-encoder
   dispatch boundary actually cost at decode-realistic shapes, and how much
   does a persistent equivalent recover? This is the causal replacement for
   the impossible trace alignment.

## Probe arms

All kernels live in a bench-only Metal file (`kernels/topology_probe.metal`),
never referenced by production PSO tables. Vehicle: new `qwen-bench
topology-probe` subcommand (JSON per config to
`target/profiles/topology-probe/`, summary table to stdout). Quiet-box rules
apply to all runs; counter-loop attach (`gpu_limiter_capture.py hold`) is the
external verification instrument.

### Arm R: residency census

Kernel: TG entry does `alive = atomic_fetch_add(g_alive, 1) + 1;
atomic_max(g_max_alive, alive)`, spins a calibrated dependent-FMA loop,
decrements on exit. Each TG also records its entry `alive` value into an
output array indexed by TG id (post-hoc distribution, not just max).

Matrix: thread width W in {32, 64, 128, 256} x register pressure in
{low, high} x device traffic in {off, on} x dwell in {200 us, 1 ms, 5 ms}.

- Dwell sweep (cx-required): short dwells can undercount residency if
  launch/ramp or atomic-entry contention dominates. The residency kill gate
  is binding only after max-alive PLATEAUS across the dwell sweep (5 ms vs
  1 ms within ~10%).
- Register pressure high = per-thread live accumulator array sized to force
  spills/low occupancy (compile-time constant N_ACC swept 8/32/64 as needed;
  verified by its EFFECT on residency, since no tool reports register
  counts). Accumulators are SEEDED FROM DEVICE MEMORY and their final
  reduction is written to a host-checked output buffer, so the compiler
  cannot fold the pressure away (cx-required).
- Device traffic on = each TG additionally streams a strided read over a
  >= 2 GB buffer (defeats caching) at ~decode-like intensity; checks whether
  residency or scheduling changes under bandwidth load.
- Grid size G >> plausible capacity (e.g. 40 cores x 256 TGs) so the census
  saturates.

Outputs: max-alive (global), max-alive/40 (per-core estimate), alive-at-entry
distribution, wall time. Cross-check one configuration under the counter
capture: Kernel Occupancy should track (max_alive x W) / device thread
capacity qualitatively.

KILL GATE (pre-registered, binding from cx session): if max_alive/40 < 16 at
W=32/low-pressure/no-traffic AFTER the dwell plateau condition is met,
Program B dies here (persistent grids cannot cover the 32-thread
one-simdgroup kernel family that dominates decode, v0.458).

### Arm S: bounded one-way signaling

Kernels: producer TG writes a monotonically increasing epoch to device memory
(device-scope atomic store, release); consumer TGs spin-read (acquire) with a
BOUNDED spin budget per epoch, recording (a) epochs observed, (b) spin
iterations until observation (latency proxy, calibrated against the measured
spin-loop iteration cost), (c) timeouts. One-way only: producers never wait
on consumers; consumers time out and exit cleanly. CANNOT deadlock by
construction - every wait is bounded, no circular waits, kernel always
terminates within its spin budget.

Matrix: consumer count {1 TG/core-est, 4/core, max-resident} x traffic
{off, on} x producer cadence {tight, 25 us, 100 us}.

Spin budgets sized >= 10 ms equivalent so ambient compositor preemption
(~400 us) cannot produce false timeouts; epochs make missed-then-caught-up
signals distinguishable from lost ones.

Lost-signal semantics (cx-required; skipped intermediate epochs at tight
cadence are NOT losses unless a per-epoch ring makes each epoch individually
observable). Failure means any of:
  - final epoch not observed before the spin budget expires;
  - observed-epoch monotonicity violation (memory-ordering failure);
  - a stale read persisting beyond budget when producer cadence leaves
    enough time for observation;
  - timeout between co-resident producer/consumer AFTER preemption
    filtering.
The {25 us, 100 us} cadence rows use a per-epoch slot ring so every epoch is
individually observable; the tight-cadence row measures throughput/latency
only and cannot generate loss verdicts.

Outputs: delivery success fraction (per the semantics above), latency
distribution (p50/p90/p95/p99 in calibrated us) both RAW and
PREEMPTION-FILTERED (epochs overlapping detected global-stall windows -
all-consumer simultaneous latency spikes - reported separately), timeout
count, sensitivity to traffic and cadence.

KILL GATE (preemption-aware per cx): any lost signal per the semantics
above between co-resident producer/consumer, or PREEMPTION-FILTERED p99
latency > 25 us under no-traffic (raw p99 including ambient compositor
preemption is reported but does NOT kill; p50 target <= 5 us; a signaling
boundary must decisively beat the ~18 us dispatch boundary it replaces).

### Arm D: boundary drain vs persistence (causal H3 quantifier)

Reference ladder: ONE serial encoder, K dependent dispatches (production
semantics - each dispatch full-barriers on the previous). Each stage is a
tiny kernel shaped like the decode glue ladder: M TGs x W threads, per-stage
work ~5-30 us, stage i reads stage i-1's output buffer (real dependency).

Persistent equivalents - TWO variants (cx-required; they answer different
B1a scopings):

- `D-global`: ONE dispatch of a grid-stride kernel executing the same K
  stages internally; stage boundary = device-scope release/acquire on an
  arrive-counter (all-arrive then proceed), waits BOUNDED with timeout ->
  abort-and-record (an abort is itself a forward-progress data point, and
  the run is discarded from timing). Measures "dispatch boundary replaced
  by global GPU barrier".
- `D-local`: same K stages and the same per-slice data dependency, but each
  TG owns a slice end-to-end and carries its dependencies locally with NO
  cross-TG wait. Measures the fusion shape B1a's glue-ladder scoping
  actually needs when stage dataflow is slice-parallel.

Barrier-only calibration row (cx-required): zero/near-zero per-stage work
with K swept - isolates persistent barrier cost from useful-work fusion so
D-global's barrier price is a measured number, not a residual.

Matrix: {ladder, D-global, D-local} x K in {8, 32, 110} x per-stage work
{~0 (calibration), 5, 15, 30 us} x W in {32, 64} x grid sized from Arm R
(persistent variants REQUIRE arm R numbers; this ordering is mandatory).
Grid sizing is CONSERVATIVE per cx: a low-water value from the plateaued
arm-R alive distribution (e.g. p10), never max-alive - oversubscribing the
global-barrier variant mostly measures self-inflicted starvation.

Outputs: per-boundary cost from the ladder = (T_ladder - T_persistent) / K
at matched total work (plus T_ladder scaling vs K at fixed work as a
consistency check against the ~18 us/boundary anchor); recovery fraction =
1 - T_persistent/T_ladder; occupancy time-series under counter capture for
one ladder and one persistent config (the persistent capture should show the
sawtooth REPLACED by sustained occupancy - direct visual confirmation of the
A0 mechanism, or its refutation).

PRE-REGISTERED PREDICTIONS AND KILL SEMANTICS (per cx review):
- P-D1: ladder per-boundary cost lands in 8-25 us at decode-realistic shapes
  (consistent with the A0-derived ~18 us anchor). If < 5 us, H3's magnitude
  was misattributed and Program B's ceiling collapses - B kill signal
  regardless of other arms.
- P-D2: recovery >= 60% of ladder boundary cost at 15-30 us stages for the
  variant matching B1a's dependency structure. The < 30% recovery kill
  applies PER VARIANT: if only D-global fails, B1a is re-scoped to
  no-cross-TG fusion (D-local shapes); if BOTH variants fail, kill B1a.
- P-D3: signaling-based stage barriers cost <= 10 us each once resident
  (ties Arm S to Arm D; checked against the barrier-only calibration row).

## Safety invariants (binding)

- Every spin/wait in every kernel is bounded and records timeouts; kernels
  always terminate within their budget; no kernel runs > ~1 s wall (GPU
  watchdog margin).
- Probe never loads model weights, never touches production PSO tables, and
  runs only on the quiet box (no concurrent GPU processes; v0.433/439
  corruption class).
- Persistent-variant correctness: each stage writes a checksum; host
  verifies ladder and persistent variants produce identical checksums before
  any timing is admitted.

## Deliverables

- `kernels/topology_probe.metal` + `qwen-bench topology-probe` (bench-only).
- JSON artifacts under `target/profiles/topology-probe/` + one counter
  capture pair (ladder vs persistent).
- PERF-LOG checkpoint with the three arms' verdicts against the gates above.
- GO/NO-GO recommendation for B1a with measured numbers filled into its
  pre-registered gate (>= 25% reduction on the ~2.6 ms glue/route family).

## Amendments during build (pre-run; gates unchanged - flag at results review)

Smoke passes (`--quick`) exposed three methodology bugs and one early
finding; fixes extend the signed matrix without weakening any gate:

1. Arm R dwell sizing now uses EMPIRICAL per-variant iteration calibration
   (1 TG, min of 3): the NACC accumulator chains pipeline (ILP), so scaling
   the nacc=1 chain cost by NACC over-sized hi-pressure dwells by up to ~8x
   (visible as hi64 "beating" lo residency in the first smoke). Traffic-
   variant rows cap the dwell sweep at {200 us, 1 ms} (qualitative rows;
   memory-dominated cost).
2. Arm S gate metric is EXCESS-over-cadence (`filtered_excess_p99_us`):
   the sampled wait spans a full inter-epoch interval, so raw waits cluster
   near the producer cadence and a 100 us row would false-kill a 25 us raw
   gate. Raw and filtered percentiles are still recorded.
3. Arm D adds: a TG-count axis m in {4, 64, low-water} (production glue
   dispatches are NARROW on a mostly-idle machine - the uniform low-water
   ladder was too friendly and backfilled instantly); PSO alternation
   between consecutive ladder stages (tp_chain_stage/_b) to match
   production per-dispatch state changes; a `global_wide` variant (full
   low-water persistent grid hosting narrow stages, idle TGs paying every
   barrier - the actual B1a host shape); and mixed-shape timing-only
   ladders (m alternating 4/64 and 4/cap) to model narrow->wide drain
   asymmetry. P-D1/P-D2 read at the decode-realistic (narrow/mixed) shapes.
4. Early finding banked from smoke: the cross-object reorder probe measures
   ZERO stale reads with no traffic but a REAL nonzero rate under device
   traffic (6e-4 to 2e-3 across smokes) on relaxed atomics. Self-validating
   protocols (payload-in-flag) are mandatory for any B1a signaling design;
   a bare data-then-flag publication is empirically unsafe on this GPU.

## Out of scope (binding, from the cx gate)

- No production-kernel persistence in B0 (that is B1a, gated on this).
- B1a remains scoped to the GDN glue/route descriptor family ONLY; folding
  projections/mat-vecs in is forbidden ("must not become
  rewrite-mat-vecs-in-a-megakernel").
- No scheduler/multi-slot work (demoted v0.459); no TGM axis in the first
  pass (add {8 KB, 32 KB} only if arm R results are ambiguous AND cheap to
  extend).
