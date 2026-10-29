# V4.1 Encoder Depth Ablation: Training-Free Skip Is a Quality Negative

2026-10-29. Discriminator branch `perf/v41-encoder-depth`, based on main
`ad7e3b9a`; clean build `cf82e551`. Raw:
`~/qwen-runs/2026-10-29-v41-encoder-depth/`.

## Decision

Training-free encoder depth reduction is closed. Keeping only selected CED
encoder blocks and bypassing the rest costs **0.68-1.79 nats** on the paired
witness, against a predeclared screen gate of 0.10 nats and 5 top-1 points.
The best schedule (ten blocks including the final block) still costs 0.68
nats, 6.8x the gate. The encoder's hidden-state transformation is not
redundant at this scale; a 1.6-2.2x cold-prefill budget is not available for
free. Any further encoder
depth bet needs distillation or calibration with a stronger quality
instrument, and is a new project rather than a cheap continuation of this one.

## Protocol

The private six-document / 384-target likelihood witness used a 512-token
prefix and 63 continuation steps at chunk 2,048. The unskipped shrinking-CED
arm is **bitwise identical** to full forward (all logits; corpus mean NLL
2.139579, rank 47.753, top-1 55.73%), so the skip comparison is not confounded
by the CED arithmetic.

The experimental hook executes a listed subset of the 20 encoder blocks.
Skipped blocks advance the CED attention cursor without encoding HC, attention
or FFN; they own no KV that the decoder reuses (CED reuses only the
publication layer). Each layer's Engram injection is retained, so the measured
loss is the transformer block, not a missing Engram read. The default path is
unchanged; the hook is active only when the keep list is set.

## Results

| encoder blocks kept | mean NLL | delta NLL | mean rank | top-1 | delta top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 (baseline CED) | 2.139579 | - | 47.753 | 55.73% | - |
| 10: `0,2,...,18` | 3.569097 | +1.429519 | 445.073 | 37.24% | -18.49 pts |
| 10 final: `0,2,...,16,19` | 2.821327 | +0.681748 | 127.906 | 45.57% | -10.16 pts |
| 6: `0,4,8,12,16,19` | 3.364224 | +1.224645 | 205.333 | 40.10% | -15.63 pts |
| 3: `0,9,19` | 3.932431 | +1.792852 | 439.094 | 33.59% | -22.14 pts |

The final encoder block carries a large share of the loss: swapping block 18
for block 19 at the same ten-block budget recovers 0.75 nats, and the
six-block arm retained it too. The witness is too small to rank the other
blocks, but the publication-adjacent transformation is plainly not removable
by this method, and even retaining it leaves the arm far outside the gate.

## Limits

The witness is a screening instrument: six documents, 384 targets, no task
acceptance threshold. The result is far outside the gate, so the negative does
not depend on a fine-grained quality comparison. No latency arm was run
because quality failed the gate; the hook would have saved most of the encoder
block work, but that number is not promotion evidence. The arm kept Engram
injections and skipped whole transformer blocks only; attention-only or
FFN-only reductions, layer merging and publication-state calibration are
different experiments and are not measured here.

## Next

The exact attention dataflow work remains the live prefill lever, and decode
still has a large distance to its byte roof. Before another approximate
encoder arm, require a concrete distillation/calibration proposal and a
quality instrument stronger than this witness.
