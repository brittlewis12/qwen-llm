# Qwen MoE B=8 whole-model gate

## Question

Does fixed-cohort layer-synchronous execution create enough aggregate throughput
on the 35B-A3B responsiveness anchor to justify a second production batch
executor after dense B=8, and can it preserve the deterministic continuation
contract?

This packet deliberately stops before scheduler or JSONL integration. It compares
the complete 40-layer graph against serialized singleton execution, the existing
GDN-projection replay, and the already measured independent-queue fallback.

## Contract

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Width: eight independent sequence-private sessions at one logical position.
- Cell: the native repeating three-GDN plus one-attention topology, then all 40
  blocks and the LM head.
- Routing: singleton router, top-k, and shared-gate arithmetic only.
- Candidate work: batched GDN projections, packed routed-expert Q4 gate/up and
  Q5 down, optional shared-expert mat-mat, and optional batched LM head.
- Correctness tiers: bitwise routed-stage comparison, numerical state/KV/logit
  comparison, and generated greedy continuation against production singleton.
- Performance gate: at least `1.15x` over serialization and at least `1.10x`
  over the simpler independent-queue fallback before product work.
- Source identity for the final exploratory binary:
  `git-source-sha256-v2:8491420833fa5fe08e5a609eee5597c8913968f5e71dc2ab95ecf7f35c87295a`.
  The dirty identity is non-authoritative for promotion but sufficient to kill
  an implementation whose corrected ceiling misses the crossover.

## Initial mechanism result

The first production-shaped four-block cell cleared its local gate:

| Cell | Position | Serialized GPU | Packed GPU | Speedup | Route IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| blocks `0..4` | `1,024` | `7.2323 ms` | `5.2769 ms` | `1.3705x` | `32/32` equal |
| blocks `20..24` | `32,768` | `8.7135 ms` | `6.7451 ms` | `1.2918x` | `32/32` equal |

Across the complete model and batched LM head at position 1,024, the aggressive
candidate moved `78.3039 -> 54.8739 ms` GPU for eight transitions (`1.4270x`,
about `145.8` aggregate token/s). One-step argmax matched in all eight lanes and
minimum candidate/control logit cosine was `0.999999941`.

At position 32,768 it moved `92.9091 -> 69.6190 ms` (`1.3345x`, about `114.9`
aggregate token/s). One of 320 route decisions crossed a near tie, but all eight
final argmax IDs still matched.

The independent-queue control at position 1,024 reached `125.81` aggregate
token/s at B=8. The aggressive static arm therefore had a real `~1.159x`
crossover, enough to justify the continuation gate.

## Continuation falsifier

The aggressive packed-all arm diverged one of eight generated lanes at step 4.
The routed-only and shared-only ablations each survived eight steps, but
routed-only diverged at step 15. Serializing the LM head did not change that
result, locating the difference before the head.

A routed-stage oracle then separated the arithmetic:

- Packed Q4 gate/up was bit-identical to eight singleton rows.
- The B=8 K512-R2 Q5 down differed from the replay control's generic fused Q5
  down by at most `9.095e-13`; the distinction was alternate-kernel rounding,
  not token-grid corruption.
- Keeping packed Q4 gate/up while copying its exact inner rows into the existing
  singleton generic Q5 down made both routed inner and routed output bitwise
  equal to the replay control.

That corrected candidate remained greedy-identical to production for 32
generated steps. At step 38, replay and candidate changed the same lane together,
while remaining equal to one another. The remaining divergence is therefore the
pre-existing GDN mat-mat schedule, not incremental MoE packing.

## Corrected economics

The bitwise-incremental candidate retains only Q4 expert gate/up batching and
keeps Q5 down, shared expert, and LM head on the replay/singleton schedules:

| Arm | GPU wall for 8 | Per transition | Speedup vs serial | Speedup vs replay |
| --- | ---: | ---: | ---: | ---: |
| serialized | `78.0269 ms` | `9.7534 ms` | `1.0000x` | — |
| GDN replay, serial FFN | `66.8357 ms` | `8.3545 ms` | `1.1674x` | `1.0000x` |
| replay + packed Q4 gate/up | `64.6661 ms` | `8.0833 ms` | `1.2066x` | `1.0335x` |

The corrected arm reaches about `123.7` aggregate token/s, slightly below the
existing B=8 independent-queue result (`125.8`). Its incremental value over the
already-known replay is only `3.35%`, while productization would still require a
second executor, admission, cohorting, cancellation, and long-context policy.

## Decision

**KILL the current Qwen MoE static-B=8 product lane.** Delete the large spike and
do not build a scheduler around it.

- Packed-all and packed-routed variants clear throughput but fail deterministic
  generated continuation.
- The corrected exact-incremental variant no longer beats independent queues.
- The existing queue-overlap mechanism remains the simpler cross-family serving
  fallback; dense B=8 remains the promoted true-reuse backend.

Reopen only if one of these changes the economics:

1. A production-shaped, realistically prefilled 64-token fixture demonstrates
   that a functional-equivalence contract is acceptable and a packed arm stays
   at least `10%` ahead of independent queues.
2. A new exact routed/shared organization removes at least another `8-10%` of
   whole-token wall without changing the continuation stream.
3. A shared cross-family executor makes the marginal MoE product cost small
   enough that a `3-5%` incremental win is worth carrying.
