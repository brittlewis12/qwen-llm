# Paired IQ2_S Down: Whole-FFN Gate Not Met

Sharing decoded down-weight panels improves this production-routing FFN by
about 2.2%, but does not meet the predeclared continuation gate. Both primary
blocks also have a small inclusive-host regression. The paired leaf is closed;
there is no default change or model-throughput claim.

## Workload And Identity

- M5 Ultra, 256 GiB unified memory; macOS/aarch64, Rust 1.99.1, SDK 27.0,
  canonical release build and native TensorOps tier.
- Research source `fdf6b3241416590a9a4c4e34e10d54c1707cd186`, tree
  `0d81d481ecd7bd58e07da64ef9414812c9d5e13f`, on
  `test/v41-paired-ffn-replay`; inference defaults remain unchanged.
- Frozen executable SHA-256:
  `1265272f524eabacf1710207790bb38196fc570d069fb459e1d89891f0739482`.
- Full DeepSeek V4.1 GSQ IQ2_XS artifact, mixed IQ2_XXS gate/up and IQ2_S down.
  Metadata selects layer 8, absolute start 0, 4,096 ordinary-arithmetic rows;
  hidden 5,120, intermediate 2,304, 384 experts, top-6 routing.
- Frozen source IDs contain 8,192 prompt and 256 continuation IDs; only the first
  4,096 prompt IDs execute. Selected u32 little-endian ID SHA-256:
  `10227cfb882eae3b0af8bf3b80c9d495bffe07297a4e5bc533209ac6b64f0c87`.
  IDs are retained privately, not published here; original text/template
  provenance is not established, so this is not a fully public input fixture.

## Method And Correctness

One completed ordinary Full prefill freezes the production FFN boundary inputs,
output and defined scratch regions. Replay recomputes mHC, normalization, live
routing, shared expert, bucketing, gate/up, weighting, half staging, down,
sum and post. Saved routes and intermediate values are references, not replay
inputs. A and B use the same work buffers and addresses.

The only candidate substitution is
`kernel_moe_down_iq2_s_f16act_grouped_slots_tensor_paired_probe` for the incumbent
IQ2_S down pipeline. Arguments, grid, threadgroup, bindings and rounding remain
unchanged. Warm census verifies one down substitution, two unchanged wide gate/up
calls and identical remaining ordered dispatches.

The sequence is sampled A, warm A/B, then primary `A B B A`, `A B B A`.
All eleven completed calls match every defined production reference region:
15 FFN stages, six mHC stages, both final half backings and output. Status,
guards, unused work capacity, live slot ownership and all immutable capture
regions pass. Source, artifact, frozen IDs and borrowed banks pass native
postflight. There is no Full/CED-logit, causal-state or sustained-model claim.

The live population contains 24,576 routed slots, 282 paired panels, 403 residual
panels and 66 zero-count experts. Pair coverage is 73.4375%:
`64 * sum(floor(count[e] / 64)) / sum(count[e])`. This is not a time fraction.

Primary GPU latency is command `GPUEndTime - GPUStartTime`; inclusive host latency
includes encoding, commit and completion wait. Reset, comparison, hashing and
reporting stay outside both timed intervals. Primary calls have no census,
stage sampling or validation layer. No samples are filtered or replaced.

Each block must independently reduce mean whole-FFN GPU latency by at least 3%
and have no mean host-latency regression. The gate is not an aggregate eight-call
estimator and is not inferred from libtest success.

## Primary Results

| Block | A GPU Mean, ms | B GPU Mean, ms | GPU Reduction | A Host Mean, ms | B Host Mean, ms | Host Regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.074354 | 44.070458 | 2.227200% | 395.678854 | 396.206833 | 0.133436% |
| 2 | 45.070458 | 44.055750 | 2.251382% | 391.755750 | 394.161209 | 0.614020% |

Both blocks fail both continuation conditions. B saves about 1 ms of GPU command
latency; the earlier isolated down operator's roughly 10% gain does not carry
through to the whole FFN. This window does not establish applicability to 1K
agent-append chunks or to other down formats.

## Execution And Analysis Status

The frozen binary passes 374 classified CPU checks and the actual-artifact
CPU-only header fixture. The native libtest completes successfully in 115.10 s;
its report records all eleven verified calls and `decision: close_leaf`.
The fixed phase caps are 30/30/70/20 s with a 150 s child cancellation trigger
and a separate 170 s supervisor. Recorded phase durations are
25.490/21.867/57.624/10.056 s. Build, freeze, CPU and header checks hold no GPU
lease; one continuously inherited canonical lease covers the native packet.

The outer controller subsequently fails while parsing a stale execution-field
name: it expects `no_rerun_or_sweep`, while the source emits
`no_in_packet_retry_or_sweep`. Its original failure and closed latch remain
recorded. The raw completed native packet is retained unchanged; its SHA-256 is
`1607ec6777e6970ac4ab93a92ed33db59af8ccf5859ed788337a1abab3c6915a`.
Separate CPU-only reanalysis corrects that strict field expectation, independently
recomputes both block means and confirms `close_leaf`. All 94 offline controls
pass, including four new stale/missing/false/non-boolean field controls. Frozen
source, binary, config, inputs, artifact and CPU/header receipt checks pass again;
all 91 original retained files keep their hashes. No native rerun or original
acceptance record is created. The complete outer-controller elapsed time is not
available, so original 170-second protocol acceptance is not synthesized. Native
completion must not be mislabeled as original controller acceptance.

Three earlier incomplete packets remain retained: unsupported global stage
ordering; confusion between the SDK's `timestamp` set and `GPUTimestamp` counter;
and insufficient host-verification budget before primaries. None provides a
primary performance result. The final packet keeps the original workload,
primary order, equality checks and comparative gates.

Stage counters remain nonexclusive diagnostic envelopes, not shader-active time,
occupancy, exclusive down cost or inputs to a model-gain projection. See
`docs/bench/2026-10-28-v41-stage-counter-intervals.md` for that distinction.
