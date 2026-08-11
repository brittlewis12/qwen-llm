# Dense static B=8 complete-block gate

## Question

Does the B=8 projection crossover survive one complete dense transformer
block when every sequence retains private recurrent state and the candidate
pays real row packing, elementwise work, and residual updates?

## Candidate

The diagnostic executes one GDN block over independent `MetalSession`s:

1. normalize each private residual;
2. pack rows and issue one batched qkv projection, z projection, and output
   projection;
3. run convolution and GDN recurrence against each session's private state;
4. apply the mixer residual and post-attention norm per session;
5. pack the normalized rows and issue one batched dense gate, up, and down
   projection;
6. apply SiLU and the final residual without sharing mutable state.

The serialized control uses the same separate residual/norm topology and the
production dense FFN policy. The candidate therefore contains six real
shared-weight projection dispatches rather than overlapping singleton graphs.

Every timed repetition restores a deterministic slot-and-element-patterned
residual, convolution buffer, and recurrent state. Each batch size has an
independent correctness comparison, command errors and timestamps fail closed,
and timing uses ABBA/BAAB order after a 500 ms B=8 device ramp.

The dirty probe binary and checkout matched at acquisition under source ID
`git-source-sha256-v2:ad21d150907f611555b6351908db97228e80ea5b475c9fe1ff78a02786107b83`.

## Results

### Qwen3.6 27B Q4_K_M

| batch | serialized GPU ms | static GPU ms | aggregate speedup | saving |
|---:|---:|---:|---:|---:|
| 2 | 1.2942 | 1.2772 | 1.013x | 1.3% |
| 4 | 2.5154 | 2.1339 | 1.179x | 15.2% |
| 6 | 4.0650 | 3.4205 | 1.188x | 15.9% |
| 8 | 5.3701 | 1.9821 | **2.709x** | **63.1%** |

B=8 is both the fastest and the cleaner numerical schedule. Its worst slot has
`cos(x)=0.999999983`, `max_abs(x)=0.016182`, relative RMS `0.000220441`,
`cos(state)=0.999999895`, and `cos(conv)=0.999999978`. The generic B=6 path
has a larger relative RMS (`0.000870737`) despite its smaller speedup.

### Qwen3.5 0.8B Q4_K_M

| batch | serialized GPU ms | static GPU ms | aggregate speedup | saving |
|---:|---:|---:|---:|---:|
| 2 | 0.1356 | 0.2368 | 0.573x | -74.6% |
| 4 | 0.2666 | 0.3187 | 0.837x | -19.5% |
| 6 | 0.3980 | 0.5315 | 0.749x | -33.6% |
| 8 | 0.5344 | 0.4238 | **1.261x** | **20.7%** |

The small model remains latency/dispatch-shaped. B=8 is real but not large
enough to lead the implementation order, especially against the already
measured B=2 independent-queue fallback.

## Decision

- Promote dense 27B B=8 from projection primitive to complete-GDN-block proof.
- Keep B=2 closed and do not build a generic small-batch scheduler.
- Price one complete attention block next, retaining private KV and batching
  only its large projections.
- Require a whole-model fixed-B=8 candidate to beat both serialized execution
  and independent queues before any resident scheduler or ragged work.
- Treat the current numerical envelope as diagnostic, not yet as a full-model
  continuation contract; accumulation across 64 blocks remains unmeasured.

## Artifacts

- `build-info.json`
- `qwen35-0p8b.txt`
- `qwen36-27b.txt`
