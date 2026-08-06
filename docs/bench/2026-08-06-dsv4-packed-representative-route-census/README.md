# DeepSeek V4 Representative Packed Route Census

Status: canonical current-asset-derived schedule fixture with limited capture
provenance. It is route geometry for model-free gates, not accepted model-output
or performance evidence.

## Question

The first grouped-IQ2 route fixture uses `[35,201,200,34]` repeated to 128
tokens. That boundary prompt activates only 1,542 layer/expert buckets across
the 25 grouped layers and may overstate occupancy for a BM16 expert matrix.

Capture one real mixed prose/code prompt from the product path and retain both
schedules in the BM16 gate. A candidate must not promote on boundary-prompt
geometry alone.

## Capture

The ignored release-diagnostics harness used:

- current model content ID
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`;
- the current strategy document rendered as an ordinary 0731 chat request;
- the first 128 native JoyAI token IDs, whose little-endian digest is
  `ee09a95c18d0231d195a88cc4d96c34ae27df5b3fd2fecb0c5e909807e4be8da`;
- ordinary CPU routing and the qualified grouped-IQ2 packed policy; and
- one unsampled execution with route-ID capture enabled.

The run completed in 39.19 seconds. The retained canonical fixture contains the
prompt IDs, route IDs, and per-layer geometry needed for model-free replay; the
single-use capture harness and duplicate raw payload were removed.

No raw command transcript was retained. Therefore this packet deliberately
makes no accepted logits, hidden-state, causal-state, or timing claim. The
fixture is independently checked from its model, prompt, route, and schedule
identities.

## Geometry

| N=128 schedule | Active layer/expert buckets | BM16 tiles | Useful occupancy | Padding columns |
|---|---:|---:|---:|---:|
| Repeated boundary | 1,542 | 2,234 | 53.7153% | 16,544 |
| Representative strategy chat | 3,154 | 3,558 | 33.7268% | 37,728 |

The representative layers range from 90 to 235 active experts, with median
117. The repeated-boundary layers range from 24 to 85, with median 63. The
schedule concern is real and large: the representative fixture launches 59.3%
more BM16 tiles for the same 19,200 useful assignments.

The representative count payload is
`0486f37c39a37cab0cbb1cfe41d8d4fca401059b7abbe872901e841fdd4d394a`.
The route payload is
`7454b2692359464c0d932e1c2fffe53d345e0ea969db9de67458ed992ddd539c`.

## Decision

Carry both schedules as co-primary cells in the BM16 production-shape packet.
Neither schedule may rescue the other. Use the representative fixture as the
primary estimate of diverse-prompt occupancy; retain the boundary fixture as a
stress case for routing concentration and large buckets.

Do not rerun this asset capture merely to manufacture missing binary or raw-log
provenance. Reopen only if the current asset, routing policy, chunk size, or
prompt-representativeness premise changes materially.

The reusable fixture lives at
`crates/qwen-llm/tests/fixtures/deepseek_v4_packed_grouped_iq2_route_census_representative_v1.json`.
