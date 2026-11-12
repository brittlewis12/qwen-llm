# V4.1 layer-sliced prefill: exact native port, serve gate unmet

The port is staged on `perf/v41-layer-sliced-prefill`, from main `932f3b729`,
with **75/75 bitwise pairs** and five native cancellation/resume cases.
**Default remains off.** Both incumbent serve controls were rejected by the
harness at thermal pressure 2. There is no accepted ABBA packet and no claim
that delivered throughput meets the 2% gate. Work stopped at this gate;
Flash-Next is unmodified.

The artifact remains DeepSeek V4.1 Flash 1017 **GSQ-RCO IQ2_XS with IQ4_NL
Engram**. No artifact selection or roadmap change is included.

## Incumbent price

The fresh same-artifact standing packet is
`2026-11-11-artifact-choice/serve-v2/A2`, commit `bba889ccf0a0`, release,
180 measured seconds, pressure 1:

| delivered tok/s | ITL p50 ms | ITL p99 ms | worst ms | intervals |
| ---: | ---: | ---: | ---: | ---: |
| 84.006 | 27.921 | 55.938 | 2097.924 | 14973 |

P99 is nearest rank over content-delta intervals whose two endpoints are in
the window. Carry-in requests count; warm-up requests do not. The raw-event
manifest and summary hash are frozen in `price-standing.json` in the lane's
run directory. The maximum intersects a **1959-row Full append at position
63471**, with **2703360 disk-read bytes** in that native call.
The previous 30-second standings had maxima 2038.088/1718.843 ms and p99
31.904/29.488 ms at 89.433/96.633 tok/s. These packets are pricing evidence,
not paired throughput controls.

The [live-CED analysis](2026-11-11-v41-live-ced-threshold.md) and
[chunk-price analysis](2026-11-11-serve-chunk-price.md) explain the schedule:
most warm appends use Full. Live CED requires an eligible episode beyond
its 2541-row decoder halo and the serve threshold is 8192 rows. CED yields
between encoder submissions, normally 4096 rows, but its final decoder
reconstruction is synchronous. Smaller row quanta change arithmetic and
still leave the reconstruction stall. The fresh Full stall clears the price
gate for layer slicing.

## Execution and policy

Opt in with `QWEN_SERVE_V41_LAYER_SLICE_MS=200`; the minimum predicted native
prefill wall is `QWEN_SERVE_V41_LAYER_SLICE_MIN_MS=500`. The current default
for MS is **0**. Zero selects the incumbent native calls, original row
geometry and decode-percentage budget. Small appends also use the incumbent
native forward. The conservative M5 chunk estimate is `275 + 0.8 * rows` ms;
layer boundaries divide its predicted work without repartitioning rows.
This is a nominal GPU slice target, not a hard host-wall deadline.

V4.1 differs from GLM:

- HC carries four residual streams plus incoming pre-mix. Full parks its
  existing chunk scratch, including `x` and `incoming`, rather than retaining
  all layer outputs. At hidden 5120 the live residual/mix is
  `rows * (4 * hidden + 4) * 4` bytes: about 320 MiB at 4096 rows. There is no
  additional GPU activation allocation for parking.
- Engram hash/address history is private per sequence. The original reads,
  projections and injections at layers 1/14, including Full's existing
  Engram/GPU overlap boundaries, retain their order.
- CED retains boundary residuals, HC mix and selection controls. Encoder
  cuts keep the original rows and publication. Decoder cuts preserve the
  original two-layer replay groups, tiles and scalar-tail lineage. The
  submit-ahead queue drains before peer service; existing status validation
  and final cache commit still gate Ready state.

This family has its own scheduler. It temporarily removes the appender from
peer scans while the native call is on the stack. One ready-peer decode round
runs at each drained boundary and at sliced native-call completion; it earns
no additional decode-budget credit. Existing sampling, fallback, output flush
and backpressure paths are reused. Decode's causal executor uses each peer's
private workspace. No appender command remains pending across peer service.
The QoS guard spans the sliced call and unwinds on every exit.

Cancellation disposes of provisional/poisoned state. Snapshot and decode
refuse that state; a fresh state can restore the prior immutable checkpoint.
Checkpoint cuts, ladder capture and header folding remain outside the
incomplete native call. `prefill_ms` now sums active native calls in both arms,
excluding peer service. Complete and cancelled partial records support
`partial_prefill_s`; peer-round counts are logged separately.

## Native regression and price

`v41_layer_slice_probe` requires `perf-ablations`, uses an iter build and a
48 Ki-token root, and alternates paired orders where there are two repeats.
Two ready peers decode between slices; the control executes the same extra
rounds afterward. Corrected CED probes also give both arms a peer opportunity
at each nonfinal encoder return. The original CED gap trace omitted those
opportunities and is retained only as preliminary evidence.

| cases | exact pairs |
| --- | ---: |
| warm Full: 32/342/1509 rows, 1/2/4/8 slices, two repeats | 24 |
| cold Full: same rows/slices, one repeat | 12 |
| initial warm/cold CED: 8192 rows, 1/2/4/8 slices | 12 |
| corrected warm CED: 1/2/4/8/18 slices, two repeats | 10 |
| corrected cold CED: 1/2/4/8/18 slices | 5 |
| warm Full actual policy cuts: 342/1509 rows, 3/8 slices | 8 |
| cold CED scalar tail: 8193 rows, 1/2/4/8 slices | 4 |
| **total** | **75** |

Every pair checks finite full logits, complete causal snapshot digests,
post-append decode logits, peer logits and peer causal-state digests. The
canonical warm-root digest is also identical across the original and refactored
CED implementations. Five native cancellation cases cover warm Full at
342/1509 rows, warm CED at 8192, and cold CED at 8192/8193. They check refusal
of snapshot/decode after cancellation and exact checkpoint restore/replay.

The initial Full price was measured at pressure 0. At 1509 rows/eight slices,
paired worst peer gaps were **1387.0/1393.9 ms off** and **217.7/216.6 ms
sliced**, with equal-round completed-command GPU-duration ratios
**0.9900/0.9902**. These are controlled probe gaps, not serve ITL statistics.
Small appends show a TTFT tax when sliced unconditionally.

CED command-duration ratios varied with cut placement and run. Initial
coarse cuts were 12–19% higher; eight-way initial pairs were near neutral.
These are sums of completed command durations, not a proof of equal shader
work. The sustained actual-policy probe reached pressure 2, so its timing is
excluded. A short attempt with 45-second idle intervals still reached 2 after
its first control arm and produced no usable timing pair. Its exactness and
cancellation results remain valid. End-to-end throughput must decide the port.

## Serve gate

Both attempts used the same release binary at `25cde3c75`, the production
artifact, `scripts/bench/serve_realistic.py`, real agent sampling with no
request sampling overrides, 49152 warm tokens per client, context 196608,
seeded call draws and a requested 180-second window. MS was **0** in both:

| retained attempt | scheduler / sample workers | cooling / settle seconds | result |
| --- | --- | --- | --- |
| `serve/A1-thermal-max3` | 3 / 2 | 45 before; 60 between warm-ups; 15 settle | thermal abort at 2 |
| `serve/A1-thermal-max4` | production 4 / 3, three clients | 60 before; 90 between warm-ups; 60 settle | thermal abort at 2 |

The first setup unintentionally reduced sampler parallelism; it was reported
and corrected before accepting any data. The second retained production
defaults. Both started timing after settling, with a measured pressure limit
of 1 (warm-up alone allowed 2). Both have `failed.json` and **no accepted
summary**. The harness's `window_done.seconds` prints the requested duration
even after early abort and is not evidence that the whole window completed.

No sliced serve arm was run after the rejected controls. Delivered tok/s,
ITL p50/p99/max and `partial_prefill_s` for a valid on/off comparison therefore
remain **unmeasured**. Sampled control outputs were coherent Rust/Metal reviews;
there is no sliced-serve coherence acceptance. The requested two interleaved
pairs and throughput gate are unmet, so default-on is not justified. No thermal
limit, row arithmetic or replay schedule was changed to obtain acceptance.

## Validation and records

- V4.1 model-free suite: **95 passed**, including Full/CED cancellation at
  every modeled seam, provisional exclusion and active-time accounting.
- Shared Driver: **23 passed**; final complete serve suite: **695 passed**,
  **42 pre-existing ignored**, no new ignored tests.
- Slice-policy geometry/zero-rollback unit passes; CPU acceptance inventory
  includes the three new model-free tests.
- `scripts/check-contracts.sh` passes; ENV regenerated. No roadmap edits.

All M5 builds/model loads used queued canonical `v41-layer-slice` leases,
nice 10, with no other model resident at load. The sustained build/probe lease
held 10m05s; no lane lease exceeded 15 minutes. Disk had 1.8 TiB free before
building. Raw evidence and scripts remain under
`metagross:~/qwen-runs/2026-11-12-v41-layer-slice/`.
M5 build output, bundles and the lane worktree are removed after recording.

Reproduce the native probe with
`cargo build --profile iter -p qwen-llm --features perf-ablations --example v41_layer_slice_probe`.
Arguments are model, corpus, repeats, row sizes, `full|ced`, `warm|cold`, slice
counts and optional idle seconds. Build serve with release and use the same
binary for both arms. The retained `serve-arm.sh` records the corrected setup.

Flash-Next pricing stops here under the gate rule. Older 40-second standings
at `487d0b57` show 136.1–141.3 tok/s, ITL p50 19.63–19.69 ms and maxima
1.352–1.397 s, but raw events are not retained for p99. Shorter loop packets
are not enough to price rare appends. A fresh real-schedule baseline is needed
before any Flash-Next port; no build/no-build conclusion is asserted for it.
