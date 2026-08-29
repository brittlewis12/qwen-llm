# Flash-Next Complete GDN Middle Falsifier

Decision: **KILL** the default-off experiment and remove its implementation.

## Candidate

The candidate adapted MTPLX's singleton GDN-middle organization to the local
released geometry while preserving the incumbent F32 arithmetic and modulo-16
Q/K ownership. It left all four input projections and the quantized output
projection unchanged.

One 256-thread threadgroup per value head performed convolution/SiLU, paired
Q/K L2 normalization, decay, delta recurrence, and gated RMS output. Because
three modulo-mapped value heads read each Q/K history row, a second ordered
kernel rolled only the 4,096 Q/K channels after every fused group had finished.
The V tail remained uniquely owned and advanced in the fused dispatch. This
race-free organization reduced the complete singleton route from 10 dispatches
to 7 and eliminated global decay, convolved-QKV, normalized-Q/K, and recurrent
intermediates.

Focused gates established:

- fused-only execution left all Q/K convolution history bit-exact and advanced
  the V tail exactly;
- roll-only execution advanced Q/K history exactly and left the V tail
  bit-exact;
- hostile N=1 and two-step rows matched staged normalized output, convolution
  state, and delta state bit-for-bit; and
- the complete Q8 route matched final output and both states bit-for-bit while
  substituting the declared 10-to-7 topology.

## Protocol

- Device: Apple M4 Max with unified memory.
- Source: committed base `393f9fb` plus an uncommitted, default-off falsifier.
  The kernel, host route, flag, tests, and probe were removed after disposition.
- Workload: 36 independent released-shape singleton middles per command, each
  with distinct weights, activations, convolution state, and delta state.
- Arms: B encoded the five staged middle kernels; C encoded the fused middle
  followed immediately by the Q/K history roll.
- State: B and C used separate, identically initialized banks. One untimed
  command warmed each arm; `B-C-C-B` repeated three times then advanced both
  banks equally within every block.
- Endpoint: complete command-buffer GPU time for all 36 middles, equivalent to
  one token across the model's 36 GDN layers. No model or weight asset was
  loaded.
- KEEP: median saving at least `0.75 ms/token` and positive mean saving in every
  balanced block. No model run was permitted before this gate.

## Results

| Run | Block | Arm | GPU ms/token |
|---:|---:|:---|---:|
| 1 | 1 | B | 2.697583 |
| 2 | 1 | C | 2.069458 |
| 3 | 1 | C | 2.064625 |
| 4 | 1 | B | 1.775250 |
| 5 | 2 | B | 1.154917 |
| 6 | 2 | C | 0.971250 |
| 7 | 2 | C | 0.947625 |
| 8 | 2 | B | 1.055542 |
| 9 | 3 | B | 1.052250 |
| 10 | 3 | C | 0.909875 |
| 11 | 3 | C | 0.953375 |
| 12 | 3 | B | 1.082958 |

Median GPU time moved `1.118938 -> 0.962312 ms/token`, saving `0.156625 ms`.
Balanced-block mean savings were `0.169375`, `0.145792`, and `0.135979 ms`.
All directions agree despite substantial first-block first-touch drift.

## Decision

The consistent saving is real but reaches only 20.9% of the required floor and
misses it by `0.593375 ms/token`. The result prices complete middle fusion at
roughly `0.14-0.17 ms/token` on this target, not at actionable decode leverage.
No model run was made, and no implementation remains.

Adversarial design and disposition review:
`01a04e59-3a82-7de1-ac63-1f7c0b5763fe`.
