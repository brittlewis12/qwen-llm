# Dense static B=8 complete-attention-block gate

## Question

Does the B=8 projection crossover survive one complete dense attention block
when every sequence retains private KV state and the candidate pays real row
packing, attention work, residual updates, and a singleton output projection?

## Candidate

The diagnostic executes one attention block over independent `MetalSession`s:

1. normalize each private residual and pack the rows;
2. issue one batched Q, K, and V projection;
3. copy each projected row into its sequence-private attention scratch;
4. run RoPE, KV insertion, single-key attention, and the output projection per
   sequence;
5. apply the mixer residual and post-attention norm per sequence;
6. pack the normalized rows and issue one batched dense gate, up, and down
   projection;
7. apply SiLU and the final residual without sharing mutable state.

The serialized control performs the same graph one sequence at a time. The
candidate therefore contains six real shared-weight projection dispatches; the
attention output projection remains singleton because its input rows are still
produced by sequence-private attention bodies.

Every timed repetition restores deterministic slot-patterned residuals and
resets the selected layer's logical KV position. Correctness checks cover the
final residual, direct Q/K/V projection outputs, the written F16 K/V cache row,
and the exact `kv_n_pos=1` frontier. Command errors and timestamps fail closed,
and timing uses ABBA/BAAB order after a 500 ms B=8 device ramp.

This gate deliberately runs at position zero with one available key. It proves
the block topology, projection batching, and private-KV contract. It does not
price long-context attention or nonzero-position RoPE.

The dirty probe binary and checkout matched at acquisition under source ID
`git-source-sha256-v2:322cbf2aa10bf0bd01ff91715f97d64b9d40826bb680df62eb5e34c944892f2e`.

## Results

### Qwen3.6 27B Q4_K_M

| batch | serialized GPU ms | static GPU ms | aggregate speedup | saving |
|---:|---:|---:|---:|---:|
| 2 | 1.1133 | 0.9226 | 1.207x | 17.1% |
| 4 | 2.2170 | 1.7438 | 1.271x | 21.3% |
| 6 | 3.3332 | 3.1452 | 1.060x | 5.6% |
| 8 | 4.4377 | 1.6029 | **2.769x** | **63.9%** |

B=8 is both the fastest schedule and numerically well inside the diagnostic
gate. Its worst slot has `cos(x)=0.999999946`, `max_abs(x)=0.008286`, relative
RMS `0.000329244`, direct-projection cosine `0.999999954`, and cached-KV cosine
`0.999999914`. B=2 and B=4 are bit-exact. The generic B=6 path again misses the
specialized B=8 crossover and is not a useful implementation target.

### Qwen3.5 0.8B Q4_K_M

| batch | serialized GPU ms | static GPU ms | aggregate speedup | saving |
|---:|---:|---:|---:|---:|
| 2 | 0.1267 | 0.1338 | 0.947x | -5.6% |
| 4 | 0.2452 | 0.2324 | 1.055x | 5.2% |
| 6 | 0.3672 | 0.5269 | 0.697x | -43.5% |
| 8 | 0.4886 | 0.4005 | **1.220x** | **18.0%** |

The small model remains latency- and dispatch-shaped. B=8 is positive but too
small to lead the implementation order, especially against independent queue
overlap for heterogeneous low-latency work.

## Decision

- Promote dense 27B B=8 from complete GDN-block proof to both block families.
- The independent block gates agree: complete GDN and attention blocks each
  retain about `2.7x` aggregate throughput at B=8.
- Keep static B=2 and generic B=6 closed; target the fixed B=8 kernels already
  responsible for the measured crossover.
- Build one whole-model fixed-B=8 dense decode candidate before any scheduler,
  ragged-batch, or resident-service product work.
- First prove the full graph at a short, equal frontier; then add one
  representative nonzero/long-context attention gate.
- Require the whole-model candidate to beat both serialized execution and the
  measured independent-queue fallback with per-sequence continuation evidence.

## Artifacts

- `build-info.json`
- `qwen35-0p8b.txt`
- `qwen36-27b.txt`
