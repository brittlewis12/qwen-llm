# Flash-Next Selected GQA4 Attention

Decision: **KEEP** both GQA4 kernels as defaults inside the experimental
selected packed-QSA path. Selected-range packed QSA itself remains default-off.

## Mechanisms

- Gathered QK assigns one SIMDgroup several selected slots and computes four
  query heads sharing one KV head. It removes the incumbent per-slot
  threadgroup barriers without changing each head's dimension accumulation or
  SIMD reduction order.
- Softmax/value keeps four independent maxima, denominators, chronological
  accumulators, and gates while loading each shared KV-head value once for four
  query heads.

Both candidates retain one dispatch per incumbent dispatch. Roll back
independently with `QWEN4EXP_QSA_GQA4_LOGITS=0` or
`QWEN4EXP_QSA_GQA4_VALUE=0`.

## Component Falsifiers

The production-shaped model-free fixture uses 32 queries at position 2,051,
24 query heads, two KV heads, dimension 256, ratio four, 512 selected blocks,
and a 2,051-token row. Complete output buffers match the incumbents bytewise.
Each timing command contains 12 layer-equivalent dispatches after B/C warmups.

| Leaf | B1 (ms) | C1 (ms) | C2 (ms) | B2 (ms) | Mean B -> C | Saving |
|:--|--:|--:|--:|--:|--:|--:|
| gathered QK | 24.311208 | 6.930084 | 6.932167 | 24.659708 | 24.485458 -> 6.931125 | 71.69% |
| softmax/value | 32.683083 | 10.453500 | 10.405125 | 33.006250 | 32.844667 -> 10.429312 | 68.25% |

Both clear the preregistered 20% leaf gate. The focused packet, fault, and
two-band tests pass with the promoted defaults; the packet remains equivalent
to repeated scalar attention at N=1 and N=32.

## Command Gate

Four fresh release-CLI processes used the frozen 2,578-token known-answer
prompt, selected packed QSA, greedy decoding, and a 64-token cap. B disabled
both kernels; C enabled both. Aggregate prefill GPU time was the decision
metric; process wall time was descriptive only.

| Arm | Prefill GPU (ms) | Prefill wall (ms) | Output |
|:--|--:|--:|:--|
| B1 | 5,848.967792 | 27,896.8 | expected JSON, EOS |
| C1 | 5,298.185292 | 5,750.0 | expected JSON, EOS |
| C2 | 5,495.908750 | 6,032.2 | expected JSON, EOS |
| B2 | 6,758.866667 | 7,252.1 | expected JSON, EOS |

Mean GPU time moves `6,303.917229 -> 5,397.047021 ms`, saving
`906.870208 ms` or 14.39%. Both balanced comparisons are positive
(`550.782500` and `1,262.957917 ms`). Every arm emits exactly
`FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}`, generates 23
tokens over 22 transitions, and stops on EOS.

This packet establishes performance and internal arithmetic preservation for
the two kernels. It is not held-out NLL authority and does not change the
default-off status of selected packed QSA.

Adversarial leverage and source review:
`01a0500f-6313-7563-b258-b1664fc1320b`.
