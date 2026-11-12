# V4.1 and Flash-Next layer-slice pricing

Base: main `932f3b729`. Research branch: `perf/v41-layer-sliced-prefill`.
The production artifact remains GSQ-RCO IQ2_XS with IQ4_NL Engram.

## Price gate

The latest same-artifact three-agent V4.1 packet is
`2026-11-11-artifact-choice/serve-v2/A2`, commit `bba889ccf0a0`,
release profile, 180 measured seconds. Raw content events within the
measurement window yield 14,973 within-request inter-token intervals:

| delivered tok/s | ITL p50 ms | ITL p99 ms | worst ms | thermal pressure |
| ---: | ---: | ---: | ---: | ---: |
| 84.006 | 27.921 | 55.938 | 2097.924 | 1 |

P99 uses nearest rank and requires both interval endpoints in the window.
Carry-in requests count; warm-up requests do not. The previous 30-second
standing packets had maxima 2038.088 and 1718.843 ms, p99 31.904 and
29.488 ms, at 89.433 and 96.633 tok/s. These are schedule-dependent packets,
not paired throughput comparisons. The fresh 180-second packet clears the
price gate for a layer-slice experiment.

The existing chunk-price and live-CED notes explain why row subdivision is
unsuitable. Live CED requires a sufficiently long episode beyond its
2541-row decoder halo; the 8192-row serve threshold leaves most warm appends
on Full. Full uses the request's fixed admitted geometry, up to 4096 rows.
CED yields between encoder submissions but its final decoder reconstruction
is synchronous. Smaller encoder submissions change arithmetic and still
leave that final stall.

V4.1 differs from GLM in three relevant ways:

- HC carries four residual streams plus the incoming pre-mix between layers.
  Full can park the existing chunk scratch without retaining every layer's
  activation: the live residual and mix are `rows * (4 * hidden + 4) * 4`
  bytes, about 320 MiB for 4096 rows at hidden 5120, within existing scratch.
- Engram hash/address history is causal per sequence. Engram reads at layers
  1 and 14 may overlap a submitted GPU prefix. A layer cut must retain the
  original hashes, injection order and overlap boundaries.
- CED retains a boundary residual, HC mix and selection controls and replays
  the planned decoder dependency cone. Its original row tiles, scalar tail,
  publication, grouped status validation and final cache commit must survive
  a layer cut. Decode's causal wave executor uses each peer's private state;
  it cannot include a provisional appender.

## Probe

`v41_layer_slice_probe` is behind `perf-ablations`. It compares original-row
Full appends with 1/2/4/8 layer slices at a 48K prefix, alternating paired
orders. Two ready peers decode between slices; the control does the same
rounds afterward. A round before time zero and a common final round bound
the gaps. It compares finite logits, complete causal snapshot digests,
post-append decode logits, peer logits and peer snapshot digests. Profiling
sums completed command GPU durations without counter sampling.

Results and the production decision will be recorded after the probe.

### Full preliminary result

M5 iter, two paired repetitions at each of 32/342/1509 rows and 1/2/4/8
slices: **24/24 bitwise**, including complete persistent-state digests and
post-append decode. Pressure fell from 1 at load to 0 before measurements
and remained 0. For 1509 rows and eight slices, the paired worst gaps were
1387.0/1393.9 ms off and 217.7/216.6 ms sliced; equal-round GPU ratios were
0.9900/0.9902. These are controlled probe gaps, not serve ITL statistics.
The 32-row case confirms that unconditional slicing taxes TTFT. CED and
cold cases are still under test; no serving policy has been enabled.

### CED placement sensitivity

The initial CED probe produced **8/8 warm** and **4/4 cold** exact pairs;
cold Full added **12/12**. Together with warm Full this is **48/48** exact
pairs. CED coarse placements cost 12–19% more completed-command GPU time;
eight slices were neutral (warm ratios 0.9951/0.9967, cold 0.9933).
Pressure was 1. These results contradict an unconditional equal-GPU-cost
assumption for every CED placement; this was reported before integration.
The actual target policy therefore still needs measurement and the serve gate.

The initial CED gap trace also omitted the incumbent peer opportunity at
nonfinal encoder returns. It is retained as `ced-probe.jsonl`, not used as
an incumbent serve-gap estimate. The corrected probe gives both arms that
opportunity, and preserves every original encoder row partition and decoder
group. No alternate replay arithmetic is substituted.

The staged serve policy is opt-in pending A/B: nominal 200 ms slices,
minimum predicted 500 ms, fixed row geometry, one peer round per drained
boundary and native-call completion. A conservative M5 estimate is
`275 + 0.8 * rows` ms per original chunk; tiny chunks stay on the original
call. The family-specific scheduler temporarily removes the appender from
peer scans and does not double-credit decode service. Completed and partial
prefill accounting sums active native work, excluding peer decode time.
