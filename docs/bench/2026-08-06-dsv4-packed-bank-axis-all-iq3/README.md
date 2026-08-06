# DeepSeek V4 Packed Bank-Axis All-IQ3 Gate

Status: `KILL_BANK_AXIS_MAPPED_ALL_IQ3`. The exact candidate is removed from
live source. It authorizes no current-asset run or production change.

## Question

The current N=128 route census assigns 355.062 ms of accepted post-route GPU
time to 16 layers whose routed gate, up, and down banks are all IQ3_XXS. Their
production path executes one gather, gate, up, SwiGLU, down, and scatter chain
for each of 1,036 active layer/expert buckets: 6,216 dispatches.

Test whether the proven mapped IQ3 matrix lineage can process the same exact
expert-major routes economically when gate and up become one bank-axis grid:

- one depth-two mapped gate/up dispatch per layer;
- one exact clamped SwiGLU dispatch per layer; and
- one existing mapped down dispatch per layer.

The candidate therefore executes 48 dispatches over all 16 layers. It retains
the deployed IQ3 dequantization, K traversal, eight F32 matrix accumulators,
MMA order, destination ownership, clamp expression, and mapped-down lineage.

## Correctness

Focused release tests cover N=1/12/31/32/33/64/128, experts 0 and 255,
partial and continuation panels, both clamp sides, malformed banks/maps/
schedules/arenas, repeat identity, exact gate/up/inner/final bits, guards, and
the three-dispatch topology.

The frozen profiler reconstructs all 16 current-asset schedules from the
canonical route fixture:

- model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`;
- route payload:
  `ee106712f42aed80cc559414140327537dd9f256110b6acffee6a1b893319582`;
- 12,288 routes, 1,036 active buckets, 1,187 width-32 panels, and 25,696
  padded assignment columns.

Before and after timing, the direct production control and candidate preserve
the same 16-layer output payload:
`2919195b47f880923fa37c041bde287c7e237e046e5047d542b45b80a81a90eb`.
All weight banks, input values, row/slot maps, schedules, and serialized fixture
also retain one exact digest:
`17b744cecf85e940601c2cb6a37fcb29fc2be511e718e959be4ca959b60eeab6`.

## Frozen Gate

After one exact control/candidate preflight and one untimed `ABBA BAAB` block,
retain `(ABBA BAAB) x 3`: 12 samples per arm. Metal command GPU intervals are
authoritative; wall intervals are diagnostic. Keep every sample.

- Median is sorted index 6; p95 is the maximum.
- Each arm requires `(max - min) / max <= 5%`.
- `Q = 5 * max(control_range, candidate_range)`.
- Median and p95 lower savings are `control - candidate - Q`.
- Both lower savings must reach 158.3 ms and 15% of their control endpoint.

Invalid evidence is `INCONCLUSIVE_HOLD`; a stable miss is the named KILL. A
pass would authorize only a separately frozen current-asset A/B.

## Result

| Endpoint | Production | Bank-axis | Raw saving | Conservative lower saving |
|---|---:|---:|---:|---:|
| GPU upper median | 348.206875 ms | 144.260875 ms | 203.946000 ms (58.57%) | 155.959125 ms |
| GPU p95 | 353.384000 ms | 144.285083 ms | 209.098917 ms (59.17%) | 161.112041 ms |

Production/candidate GPU drift is 2.7158%/0.0624%, so the packet is stable.
The production range is 9.597375 ms, making the frozen uncertainty charge
47.986875 ms. The p95 gate passes, as do both relative gates. The median lower
saving misses the 158.3 ms absolute gate by 2.340875 ms. The formal result is
therefore `KILL_BANK_AXIS_MAPPED_ALL_IQ3`.

Wall diagnostics independently point the same way, with a 159.064872 ms median
lower saving, but have no classification authority.

## Decision

Do not rerun, relax the uncertainty charge, or spend a current-asset campaign
on this exact bank-axis geometry. Archive the executed source and remove it.

The bounded lesson is not that expert-major all-IQ3 execution lacks leverage:
the candidate removes 58.57% of this measured cohort and 99.23% of its
dispatches. Rather, dispatch collapse alone leaves a 144 ms matrix/data-movement
core and narrowly misses the deliberately conservative whole-prefill floor.
Reopen only for a materially different work unit, such as a true multi-row
matrix-friendly low-bit representation, a larger effective token superchunk,
or relevant compiler/device drift--not another launch-axis retune.

## Provenance

- Base revision: `b24e992af4544f0865f3497b22b0e473e356b40c`.
- Device: Apple M4 Max, 128 GB unified memory.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`.
- Metal: Apple metal 32023.864, `air64-apple-darwin24.6.0`.
- In-test elapsed: 19.48 seconds.
- Maximum allocated Metal bytes reported by the fixture: 2,979,528,704.
- Executed release test binary:
  `043cc04d90cce094468046e0b85cb9be62f784d3dd5c172f120161d5c357ae8b`.
- Archived executed-source diff:
  `e87fed35143c2a6ec466af5e63677e0414b9260b754793c71f78bb808adb88ed`.
- Sole raw log:
  `2fd07d9a3d0aa4a9ed8d777c3f6d8413972e96bf146da0197f5f58e004d845a8`.
- CX review: `019fd685-acb3-7890-83c9-192cbea48c6e`, final static verdict GO.

Executed command:

```bash
cargo test -p qwen-llm --release --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::profile_packed_grouped_bank_axis_all_iq3_production_routes \
  -- --ignored --exact --nocapture
```
