# v0.605 Dense-27B Topology-Preserving Parallel-Copied Loader

Status: **sealed loaded-stage inconclusive** with no product authority. Exact
correctness and all 12 loaded children complete, but three frozen decode-stability
gates miss. Fresh output-128 never runs.

Frozen source, build, and runtime:
`6114ea4f7fbd6db93048368276420a6c7a37ba4b`.
Canonical packet:
`target/profiles/v0605-dense27b-parallel-copied-loader-p1/`.

## Integrity And Correctness

- All 12 loaded children are unique sole attempts in the frozen
  `AB/BA/BA/AB/AB/BA` order. Every child exits zero with exact command,
  environment, load identity, global output hash, and five repetitions.
- The release gate authenticates all `16,806,250,496` candidate bytes, 851
  independent exact-sized offset-zero resources, modes, bindings, schedule,
  marker, ledger, checked-write guards, logits, complete KV/GDN/conv state,
  argmax, forced transition, continuation logits, and continuation state.
- All 55 inventory members, 24 launch/completion rows, 12 attempts, final model
  identity, decision, inventory, and completion seal verify.
- Every cache and child interval records zero Pageouts, Compressions, and
  Swapouts. Swap occupancy never grows, block input is zero, and all host checks
  pass. One cache interval has a `+5` compressor-occupied-page diagnostic with no
  corroborating activity, independently confirming why v0.604's gauge veto was
  not a usable pressure classifier.

## Loaded Result

The candidate materialization endpoint transfers cleanly:

- B ready wall median is `559.370 ms`, range `555.951-563.556 ms`;
- v0.603's isolated floor median was `557.7565 ms`;
- B endpoint total-CPU median is `2212.887 ms`, with AB/BA medians
  `2220.070/2211.615 ms`; all `2.45/2.55 s` caps pass;
- maximum paired RSS/footprint B/A is `1.000303/1.000712x`;
- B complete-process CPU is descriptively `+610-700 ms`, or
  `1.245-1.297x` A.

The frozen stability gate fails in three pairs:

- B pair 2 repetition-5/repetition-3 decode TPS: `0.978522868x`;
- B pair 4: `0.975774593x`;
- A pair 6: `0.974869414x`.

Instability has higher decision precedence than loaded performance. Pairs 1-2
also miss B prefill/request gates, while pairs 3-6 pass and remain near parity;
those early losses therefore cannot support a candidate regression claim.

## Nonstationarity Diagnosis

The loaded cell has strong upward within-process drift in both arms. Median
prefill by repetition is:

| Repetition | A | B | B/A |
| ---: | ---: | ---: | ---: |
| 1 | `1836.20 ms` | `1836.65 ms` | `1.00025x` |
| 2 | `1837.60 ms` | `1837.75 ms` | `1.00008x` |
| 3 | `1985.70 ms` | `1986.70 ms` | `1.00050x` |
| 4 | `2038.75 ms` | `2046.40 ms` | `1.00375x` |
| 5 | `2089.65 ms` | `2087.25 ms` | `0.99885x` |

Every child ends 11.9-25.8% above repetition 1. Decode medians also rise from
`5014.35/4993.30 ms` to `5107.30/5141.60 ms` for A/B. This pattern is consistent
with heat or another temporal state change, but the packet has no clocks,
temperatures, or power evidence and does not prove thermal throttling. The drift
simply dwarfs the intended 1% causal resolution.

Across repetition index, aggregate A/B behavior remains near parity. That argues
against a stable systematic B warm tax, but the unresolved first-two-pair
candidate interaction prevents exonerating B. The supported statement is no
evidence of a systematic candidate tax, not evidence that no tax exists.

## Authority And Closure

The mechanical result is `inconclusive`, stopped after loaded instability.
Fresh-process first-byte, exit, runtime/load, model-ready, and complete-process
CPU gates are entirely unobserved. No force-path or default authority follows.

v0.603 remains valid mechanism-floor evidence, and v0.605 shows that its
materialization endpoint transfers into the product loader. The frozen protocol
forbids another successor after valid v0.605 children, so this dense product
validation lane closes without authority. Do not rerun, pool, or rescore it.

Adversarial sealed-result review used `cx ask` session
`019f7790-a6ad-7601-8a4b-1b9b3e3e13fc`.
