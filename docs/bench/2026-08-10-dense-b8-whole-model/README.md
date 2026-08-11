# Dense fixed-B=8 whole-model decode gate

## Question

Do the independently measured B=8 GDN- and attention-block crossovers survive
the complete dense model, including embedding, all 64 blocks, private recurrent
and KV state, final normalization, the full LM head, and row-wise token
selection?

The decision baseline is not only serialized production execution. A useful
fixed cohort must also beat the already available independent-command-queue
fallback, which overlaps complete singleton graphs but cannot reuse a weight
read across sequences.

## Candidate

The fixed cohort owns eight independent `MetalSession`s and one shared batch
arena. For every teacher-forced transition it:

1. gathers eight token embeddings and scatters one residual row per session;
2. walks the model in layer order;
3. batches the large QKV, Z, mixer-output, Q/K/V, and dense-FFN projections;
4. retains private GDN convolution/recurrent state and private attention KV;
5. executes per-session GDN and attention bodies at one equal causal frontier;
6. applies the production residual/norm policy and private residual updates;
7. batches the final LM head as `[8, hidden] -> [8, vocab]`; and
8. performs row-wise GPU argmax over all eight complete vocabulary rows.

The serialized arm calls the production dense decode path eight times. It uses
the default concurrent-GDN organization and the same explicit residual/norm
policy. Arm order alternates at every position. A 500 ms static-B=8 ramp and
two untimed transitions precede six measured positions.

Every position requires exact per-slot argmax agreement and bounded full-logit
and residual drift. The final state audit checks every GDN recurrent tensor,
every GDN convolution tensor, and all written F16 K/V rows. This establishes an
eight-transition causal continuation, not just a stateless token-zero result.

The queue comparison uses the existing exact-evidence probe with eight clients,
a two-token context, six transitions per client, and both AB/BA orders. Its
workload is not byte-identical to the static probe, but the serialized anchors
agree within about 2.1% (`25.72` versus `25.19 tok/s`).

The dirty probe binary and checkout matched at acquisition under source ID
`git-source-sha256-v2:6f64e665e53374e11c14bb513ce3952190fe4781381000f0bcf7066a7ead62d0`.

## Results

### Qwen3.6 27B Q4_K_M

| static-probe execution | aggregate tok/s | median wall for 8 | relative |
|---|---:|---:|---:|
| production serialized | 25.72 | 311.076 ms | 1.000x |
| fixed static B=8 | **66.12** | **120.997 ms** | **2.570x** |

| queue-probe execution | aggregate tok/s | wall for 8 transitions | relative |
|---|---:|---:|---:|
| monolithic serialized | 25.19 | 317.535 ms equivalent | 1.000x |
| independent queues | 28.03 | 285.415 ms equivalent | 1.113x |

Across the two probes, static B=8 is an estimated `2.359x` faster than the
independent-queue fallback. This is directional rather than a matched
head-to-head claim: the source, model, device, cohort width, and short-context
regime match, while token streams and graph organizations differ. The margin is
large enough to choose the next implementation target, not to close the final
product promotion gate.

Within the matched static probe, median GPU wall is 118.554 ms versus 307.350 ms
summed across serialized commands, a `2.593x` GPU-service reduction.

All 64 argmax decisions across eight causal positions agree. Worst observed
whole-model evidence is:

- logits cosine `0.999999479`, relative RMS `0.001027374`, max abs `0.024388`;
- residual cosine `0.999998495`, relative RMS `0.001735626`;
- recurrent-state cosine `0.999999658`, max abs `0.002332`;
- convolution cosine `0.999999638`, max abs `0.009260`; and
- written-KV cosine `0.999999759`, max abs `0.062500`.

The independent-queue probe retains exact per-client argmax and final-logit
hashes in both orders. Its `3.77x` command-interval concurrency produces only a
`1.113x` aggregate gain because each singleton graph stretches under resource
contention.

### Qwen3.5 0.8B Q4_K_M

| execution | aggregate tok/s | median wall for 8 | relative to serialized |
|---|---:|---:|---:|
| production serialized | 381.34 | 20.979 ms | 1.000x |
| fixed static B=8 | **604.21** | **13.241 ms** | **1.586x** |

The small dense model also benefits end to end, but its dispatch-shaped graph
leaves much less reuse headroom than 27B. All argmaxes agree; worst logits
cosine is `0.999998268`, and all persistent-state cosines exceed `0.99999958`.

## Decision

- Promote fixed B=8 dense decode from a diagnostic primitive to a complete-model
  backend candidate. Both block families and the complete model independently
  retain the same large 27B crossover.
- The candidate clears the matched production-serialization gate at `2.570x`.
  The separate queue probe strongly indicates a `2.359x` advantage over
  independent queues, but retain one matched queue/static check before product
  promotion.
- Keep B=2 and generic B=6 closed. The measured win depends on the specialized
  B=8 matrix kernels and a full cohort.
- Preserve independent queues as the capability fallback for underfilled or
  heterogeneous work; do not use them as the full-cohort policy.
- Extract a family-neutral fixed-cohort scheduler contract, but keep this first
  execution backend dense-Qwen-specific. A3B/MoE and DeepSeek should qualify
  their own layer-synchronous backends against the same contract.
- Before a product default, validate one representative long-context frontier
  and move the executor out of the benchmark with checked admission, graceful
  cancellation, per-slot outputs, and an explicit underfill policy.
- Do not build ragged or continuous batching yet. First land the simple fixed
  cohort that the complete-model evidence already supports.

## Commands

```bash
target/release/qwen-bench --allow-dirty decode-dense-whole-batch \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --warmup-steps 2 --steps 6 --ramp-ms 500 --seed 1

target/release/qwen-bench --allow-dirty queue-overlap-probe \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --clients 8 --target-ctx 2 --window 6 --runs 2 --seed 1
```

## Artifacts

- `build-info.json`
- `qwen35-0p8b.txt`
- `qwen36-27b.txt`
- `qwen36-27b-queues.json`
