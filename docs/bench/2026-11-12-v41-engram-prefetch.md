# V4.1 Engram admission prefetch: I/O is real but not the stall; negative

**Do not ship the prototype.** The Engram row reads are a real source of
process disk traffic, but they are not the dominant component of the
multi-second serve gaps in this packet. A read-only admission prefetch removed
most prefill-time Engram disk reads yet did not improve append TTFT or the
worst gap consistently. The prototype was deleted; the measurement
instrumentation and the paired harness controls remain.

Measured source `d8602c89`, branch `perf/v41-engram-prefetch`, release build,
three clients, production `max_active=4` with three sampler workers, 49,152
warm tokens per client, context 196,608, agent-default sampling, fixed plan
seed 20261109, 15 s ramp and 180 s window. Artifact: the four IQ2_XS GSQ-RCO
shards plus the disk-streamed `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`
companion. Every arm loaded under the canonical Metal lease; no other qwen
model was resident at load.

## Step 1: per-append attribution

`QWEN_SERVE_V41_PREFILL_TIMING=1` now reports, for each native prefill call,
the existing whole-process page-in and disk-read counters plus four new
source counters from the Engram row path:

- `engram_read_rows`, `engram_read_bytes`: exact rows requested and logical
  `pread` bytes;
- `engram_read_page_bytes`: an upper bound on the distinct 16 KiB host pages
  those rows touch (one or two per row, deduplicated within a read call);
- `engram_read_wall_ms`, `engram_advise_wall_ms`: inclusive host time in the
  row-read loop and in issuing `F_RDADVISE` (advice is asynchronous, so this is
  syscall issue time, not device transfer).

The counters are active only inside an existing profile scope, so the
default-off timing knob is still the only switch and normal execution pays one
TLS check per Engram read.

Attribution arm (`S1-off`, prefetch off, timing on), 71 native prefill calls:

| measure | value |
| --- | ---: |
| active prefill wall | 43.88 s |
| GPU active (completed command timestamps) | 39.59 s |
| commit/wait (includes GPU and driver queue) | 39.48 s |
| process disk reads | 1.804 GB |
| Engram row-read host time | 1.899 s |
| Engram advice issue time | 1.630 s |
| Engram logical bytes | 235.5 MB |
| Engram page-touch upper bound | 23.86 GB |
| process major page-ins | 0 |

Correlations with per-call prefill wall: rows 0.958, GPU active 0.969,
process disk reads 0.356, Engram row-read time 0.487. Row count and GPU work
track the wall; disk traffic does not. The 23.86 GB page upper bound versus
1.80 GB of physical reads also shows that most of the append rows were already
cache-resident from the 48K warm roots.

Three representative calls:

| position | rows | wall ms | GPU ms | disk bytes | Engram read ms | advice ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64,720 | 1,959 | 1,956.6 | 1,721.7 | 233,193,472 | 76.8 | 157.4 |
| 56,517 | 1,249 | 1,241.4 | 1,125.3 | 494,747,648 | 58.0 | 169.8 |
| 49,517 | 1,106 | 2,150.3 | 1,050.2 | 20,807,680 | 42.3 | 43.9 |

The third call is the first large append after the 60 s settle and has the
largest wall/GPU divergence (about 1.1 s outside both GPU and Engram spans);
its disk traffic is negligible. That is queue/background/thermal exposure, not
Engram I/O. The 494 MB call is not a stall on the Engram path: its row-read
time is 58 ms and the 494 MB is asynchronous advice completing in the interval.

The gate's largest gap points the same way. A1 recorded a 3.621 s worst gap,
but its largest native prefill was 1.897 s (1,959 rows) with 32 KB of process
disk reads, and the rest of its calls were smaller. The gap therefore contains
queue/service exposure that no single append's Engram read explains; it
coincided with 72.9 s at thermal pressure 2 and host load average near 14.

**Which file?** The large prefill-time bursts coincide with Engram advice
windows and the only file the prefetch experiment touched was the auxiliary
IQ4_NL Engram table. With that table's rows warmed at admission (`S2b-advise`,
timing on), 68 native calls recorded 8.14 MB of process disk reads total, so
the residual attributable to weight shards or anything else during those calls
is at most about 8 MB. Weight-shard reads are not separately instrumented:
`proc_pid_rusage` is whole-process, and backend weight pages are consumed by
Metal from mapped shards. The measured residual and the lack of a wall/disk
correlation bound their contribution in this packet, but this is not an exact
per-file split.

**Answer:** Engram host read plus advice is roughly 4-8% of active prefill
wall; process disk reads are not on the critical path in these arms. The
multi-second gaps are dominated by GPU/queue time and, in this packet, by
external load and thermal pressure. The prior 1,798-row / 883 MB event remains
real, but it is a cold-cache outlier, not the steady explanation.

## Step 2: admission prefetch prototype and gate

The prototype computed the append suffix's Engram row IDs from the request's
address state at admission and started one detached read-only worker before
prefill was scheduled. It capped at 8,192 new tokens and skipped cold prompts
above that bound. Default mode issued `F_RDADVISE` for every row; a background
`pread` mode was also implemented. It changed only when bytes were read, so
decoded values were unchanged by construction. Rollback was
`QWEN_V41_ENGRAM_ADMISSION_PREFETCH=0`.

Prefetch activity was visible: 43 jobs in the timing probe, up to 393,216 rows
(8,192 tokens x 48 rows) and 6-2,149 ms of advice issue time depending on host
load. In the quiet probe, prefill-time disk reads fell from 1.80 GB to 8.14 MB.
Under the gate's background load, advice issue for large jobs took 0.7-2.1 s,
and some prefill calls still read 122-222 MB because the advice had not
finished. The worker competes with the owner thread and does not make the
critical path shorter when the rows are already cache-resident.

### Gate

Order **A1 B1 B2 A2 A3 B3** (ABBAAB), pairs `(A1,B1)`, `(A2,B2)`,
`(A3,B3)`; A = prefetch off, B = admission advice. Each arm settled to thermal
pressure <= 1 for 60 s inside its own lease before loading, then allowed
pressure 2 during warm-up and the window, aborting at 3. Pressure seconds are
sample-and-hold within the 180 s window.

| arm | mode | tok/s | ITL p95 / max (s) | append TTFT p50 / p95 (s) | thermal 0/1/2 s | max |
| --- | --- | ---: | --- | --- | --- | ---: |
| A1 | off | 89.517 | 0.0303 / 3.621 | 0.960 / 1.957 | 107.1 / 0.0 / 72.9 | 2 |
| B1 | advice | 87.922 | 0.0300 / 2.683 | 0.951 / 1.983 | 104.3 / 75.7 / 0.0 | 1 |
| B2 | advice | 93.417 | 0.0281 / 1.859 | 0.890 / 1.792 | 180.0 / 0.0 / 0.0 | 0 |
| A2 | off | 92.072 | 0.0284 / 2.599 | 0.902 / 1.899 | 86.5 / 30.3 / 63.1 | 2 |
| A3 | off | 89.500 | 0.0293 / 3.321 | 0.948 / 1.904 | 149.5 / 30.5 / 0.0 | 1 |
| B3 | advice | 85.206 | 0.0304 / 5.527 | 1.041 / 2.071 | 109.7 / 28.8 / 41.5 | 2 |

| pair | order | tok/s change | worst-gap change | TTFT p95 change |
| --- | --- | ---: | ---: | ---: |
| A1/B1 | AB | -1.78% | -0.939 s | +1.34% |
| A2/B2 | BA | +1.46% | -0.740 s | -5.66% |
| A3/B3 | AB | -4.80% | +2.206 s | +8.78% |

Paired Student-t over three arm pairs (df 2): tok/s relative mean **-1.71%**,
95% CI **[-9.48%, +6.07%]**; absolute mean **-1.515 tok/s**, CI
**[-8.52, +5.49]**. Throughput is not worse beyond noise, but there is no gain.
Worst gap improves in two pairs and regresses by 2.206 s in the third; mean
change is +0.176 s worse. Append TTFT p95 relative mean is +1.49%, with
changes from -5.66% to +8.78%. Neither primary metric improves materially or
consistently, so the gate fails.

Thermal state is a confound but not a rescue: the two off arms that hit
pressure 2 worst gaps of 3.62 s and 2.60 s, while the on arm B2 at pressure 0
recorded 1.86 s; however, on arm B3 at pressure 2 recorded the packet's worst
5.53 s gap, and the on arms did not improve TTFT p95. Every arm observed
compiler processes and the host ran unrelated ghost/merge work (load average
about 13-14 at A1 load); those are recorded in `host-before.txt` and
`host-after.txt` per arm, not controlled.

Output sample: retained delta streams for B1/B2/B3/A2/A3 contain about 62-68K
characters each, with no replacement characters and no repeated 8-word shingle
above twice. A1 was retained without bulk response files after a driver edit
during its arm; its summary and per-call stats remain. This is a qualitative
coherence check, not a bitwise qualification. The prototype did not alter the
data path, so the existing native bitwise evidence still applies.

## Decision and records

- **Delete** the admission prefetch path and its environment knob. Keep the
  opt-in Engram source counters from Step 1, the harness settle-limit/pressure
  trace controls, and the three-pair throughput interval helper.
- No sampler, routing, artifact, residency, or default serve policy changed.
- Raw packet: `metagross:~/qwen-runs/2026-11-12-v41-engram/`, including
  `serve/{A1,B1,B2,A2,A3,B3}/summary.json`, `calls.jsonl`, `thermal.json`,
  `settle.json`, `native-lines.json`, host snapshots and retained deltas;
  timing probes `serve/{S1-off,S2-advise,S2b-advise}/`; driver
  `run_one.sh`, `gate.sh` and `analyze.py`.
- The prototype's measured commits are `473ef4a4e` and `d8602c894`; the final
  branch tree removes the prefetch path. Twenty harness tests and the CPU
  contracts cover the retained instrumentation and controls.

## Integration note

Main advanced from `379c8ea88` to `f691af6e1` while this packet ran: the
layer-sliced prefill and equivalent settle/CI harness controls landed. This
branch is based on `379c8ea88` and was not rebased. Its harness hunks predate
the landed versions (they do not carry the later `events` row field or
`gap_p99_s`), so **do not take the harness hunks over the newer main**. For a
landing, take the profile/Engram instrumentation, the CED fixture fields, this
note and the PERF-LOG entry, then regenerate `docs/ENV.md`; the serve numbers
remain measurements of the `379c8ea88` / `d8602c89` tree before the layer-slice
landing.
