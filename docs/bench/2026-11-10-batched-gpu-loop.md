# Batched GPU token loop, 2026-11-10

Lane: `perf/batch-gpu-loop` (worktree `~/code/qwen-llm-batch-loop`). Task
SRV-R2D8, related to SRV-8F18. No merge; this note is the decision record.

## Bottom line

- The B>=2 serve round pays a host bubble much larger than the ~1 ms build
  gate. Direct B=3 measurement (32K/48K) puts it at **6.6-7.2 ms/round
  (24-26% of wall)**: ~4.3-4.7 ms CPU encode, ~1.2-1.5 ms commit + full-logit
  readback, and ~1.0 ms host sampling. The existing B=4 diagnostics show the
  same shape at 5.5-7.5 ms (17-21%) with a larger encode term. Almost all of
  the encode is CPU PLE row staging.
- The host encode is the same work the B=1 loop already moved to the GPU:
  `prepare_qwen4exp_text_token` -> `stage_ple_rows` gathers PLE rows on the CPU
  and copies them into a shared tensor for every member every round.
- Native Flash-Next device-feedback batch loop landed: per-member `GpuInputs`,
  per-member `GpuTokenSampler`, one command prepared ahead, tokens read back
  only for streaming and stop decisions. Exactness arm: greedy logits must be
  bitwise equal to the host-staged batch forward.
- The **non-pipelined device-input step is shipped default on**: +7.4-8.7%
  per round at B=2..4, bitwise exact, rollback `QWEN_FN_BATCH_GPU_INPUTS=0`.
- The one-ahead device-feedback loop is bitwise exact but measured **slower**
  than the host path on the loaded M5, so it stays **off** behind
  `QWEN_SERVE_FN_BATCH_GPU_LOOP`. V4.1 and GLM need their token-dependent host
  staging moved to the device; see the plan.

## 1. Measured bubble

Source files (no new run needed for the split): `~/qwen-runs/2026-11-09-fn-batch-next/`
and `~/qwen-runs/2026-11-09-fn-qsa-prep/` on M5, written by
`fn_batch_counterfactual` through the same `Qwen4ExpDecodeBatch::step` that
serve uses. All rows are B=4.

| packet / arm | encode ms | GPU ms | wall ms | wall - GPU | wall - GPU - encode |
| --- | ---: | ---: | ---: | ---: | ---: |
| fn-batch-next 8K arm0 | 5.85 | 28.60 | 36.06 | 7.45 | 1.61 |
| fn-batch-next 8K arm1 | 5.78 | 27.30 | 34.67 | 7.37 | 1.59 |
| fn-batch-next 32K arm0 | 5.06 | 28.58 | 35.09 | 6.51 | 1.46 |
| fn-batch-next 32K arm1 | 5.06 | 27.28 | 33.80 | 6.51 | 1.45 |
| fn-qsa-prep 32K candidate | 4.25 | 26.57 | 32.12 | 5.55 | 1.30 |

Reading:

- `wall - GPU` is the host critical path while the GPU is idle between rounds:
  encode + commit + readback. It is 5.5-7.5 ms.
- `encode` is dominated by the CPU PLE gather in `stage_ple_rows`
  (`table.gather_packed_into` allocates a host `Vec`, gathers, then
  `write_tensor_bytes` for each member). It scales with members.
- The remaining 1.3-1.6 ms is the blocking commit/wait plus copying every
  member's full-vocabulary logits to the host.
- Host sampling is not in these diagnostics; the probe below measures it.
- The direct B=3 probe below measures the same decomposition and confirms the
  host share; its GPU busy time is ~20.7-20.9 ms and its round wall
  ~27.6-27.7 ms.

Direct B=3 probe: `crates/qwen-llm/examples/fn_batch_loop_price.rs`
(`MODEL CODE PROSE 32768|48000 64 3`), same binary, released sampling preset,
after eight excluded warm rounds. Medians (mean in parentheses), ms:

| depth | host sample | step wall | encode | GPU | step - GPU | round - GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 0.96 (0.96) | 26.60 (26.28) | 4.73 (4.26) | 20.67 (20.62) | 5.93 (5.66) | **7.18 (6.62)** |
| 48000 | 1.02 (0.99) | 26.71 (26.59) | 4.57 (4.31) | 20.90 (20.88) | 5.81 (5.72) | **6.83 (6.71)** |

So the B=3 host bubble is **~6.6-7.2 ms/round (24-26% of wall)**: ~4.3-4.7 ms
CPU encode, ~1.2-1.5 ms commit/readback, and ~1.0 ms host sampling. This is
much larger than the ~2 ms working estimate and confirms the build gate by a
wide margin. The 32K/48K thermal start was level 0; the campaign ended at
level 1.

## 2. Design

### Device feedback instead of host staging

`Qwen4ExpDecodeBatch::generate_gpu_tokens` (native) runs a fixed member set:

- Each member keeps its own `GpuInputs` (PLE bank views, control argument
  buffer, GPU PLE history, token-embedding selection). The prefix kernels read
  the sampled token from device storage, gather PLE rows on the GPU, write the
  control scalars, update the PLE history and write the token embedding.
- `decode_batch::encode_gpu` uses the same layer-major batched kernels as
  `step`, with the session encoder marked `gpu_owned` so no child stages CPU
  inputs or records command ownership. The forward arithmetic is unchanged.
- The initial frame samples from the completed logits. Each later frame is a
  forward for the previous frame's sampled tokens plus a new sample, encoded
  while the previous command executes (same command queue ordering as the B=1
  loop). Only the token readback is host-visible, and it overlaps the next
  encode.
- A member stop (EOS, token budget, backpressure) discards the prepared frame
  before commit, so no member state advances past an unaccepted token. A burst
  is all-or-nothing across the member set; the serve driver re-forms the set
  after the burst.

### Sampling and exactness

- One `GpuTokenSampler` per member, using the existing token-feedback kernels
  (`kernel_token_sort_tiles`, `kernel_token_sample_sorted`,
  `kernel_argmax_f32_greedy`).
- RNG follows the B=1 loop: the first frame peeks `gpu_uniform(0)`, later
  frames peek `gpu_uniform(1)`, and `commit_gpu_draw` advances exactly one draw
  per accepted token. A discarded frame commits no draw.
- Standard: the sampler-v1 exactness test in `metal/token_feedback.rs` pins the
  GPU categorical distribution to `Sampler::diagnose` for the same uniform.
  Batch exactness additionally checks that, at temperature 0, the device-input
  forward produces bitwise-equal logits and the same tokens as the host-staged
  batch forward over many steps.

## 3. Native Flash-Next proof

`crates/qwen-llm/examples/fn_batch_loop_ab.rs` restores the same prefills for
both arms, warms both paths, then:

1. greedy exactness: host-staged `step` vs the device loop, comparing every
   logit vector and token;
2. released preset (T=1, top_k=20, top_p=0.95, seed 1) timing: delivered
   tokens/s, round cost, and the loop's GPU/encode/wait split.

Result (three same-binary runs; the last has direct wait and preallocated
token sets):

- **Exactness passed at both depths, every run.** Greedy 48 steps x 3 members:
  `nonbitwise_logit_vectors=0`, `max_abs_logit_diff=0.0`, no token mismatch.
  The device-input forward is bitwise identical to the host-staged batch
  forward. Released-preset token identity also passed (0 mismatches over 144
  tokens), so the per-member RNG draw sequence and GPU sampler match the host.
- **Timing does not beat the host on this contended M5.** Best observed 48K
  run was +2-3%; the final run was **32K -11.7%** (host 108.8 vs loop
  96.1 tok/s) and **48K -10.9%** (102.0 vs 90.9). The loop's GPU span is
  ~20.6-21.2 ms/round, but its command wall from commit to completion is
  ~28-30 ms. The ~8-10 ms gap does not come from the completion-handler
  channel (a direct `wait_completed` is the same) or from per-round token
  allocation (two preallocated sets are the same). It behaves like the GPU
  driver stalling on the concurrently encoded next command while the host CPU
  is loaded by other lanes' builds: the host arm encodes, commits, then waits
  serially, while the loop encodes ahead.
- **Decision: do not enable.** The serve flag stays default off. The loop is
  exact and the bubble is real, but this implementation does not yet convert
  it into delivered tokens/s. The next probe should capture a GPU timeline and
  test on an idle host before changing the design.

Raw: `~/qwen-runs/2026-11-10-batch-loop/ab-summary.json`.

## 3b. Shipped: non-pipelined device-input step

`Qwen4ExpDecodeBatch::step` now uses `step_gpu_inputs` for B>=2, default on
(`QWEN_FN_BATCH_GPU_INPUTS=0` rolls back). The host still samples; it writes
the one-token device input, and the GPU input prefix gathers PLE rows, selects
the embedding and writes the controls. Paired alternating A/B with two
independent restored state sets and identical teacher-forced tokens:

| Depth | B=2 | B=3 | B=4 | F CPU -> GPU | m CPU -> GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 21.40 -> 19.92 (+7.4%) | 25.91 -> 23.95 (+8.2%) | 30.50 -> 28.08 (+8.6%) | 12.28 -> 11.74 | 4.55 -> 4.08 |
| 48K | 21.71 -> 20.22 (+7.4%) | 26.25 -> 24.22 (+8.4%) | 30.86 -> 28.40 (+8.7%) | 12.55 -> 12.01 | 4.57 -> 4.09 |

Every cell is bitwise exact (`max_abs_logit_diff=0.0`,
`nonbitwise_logit_vectors=0`). B=3 host breakdown after (ms/round): 32K encode
3.10, GPU 19.75, commit+readback 1.03, sample 1.01; 48K 3.09 / 20.04 / 1.02 /
1.00. The CPU PLE gather is removed; the device prefix adds ~2 ms of encode but
removes ~3.5-3.8 ms of host work. Thermal started at level 1 and ended at level
2; the paired alternating design bounds the comparison, but the absolute ms is
a hot-host number. Probe `fn_batch_gpu_inputs_ab` (`--profile iter`); raw
`~/qwen-runs/2026-11-10-batch-loop/gpu-inputs-ab.jsonl`.

## 4. Serve integration

**Landing split:** this branch (`perf/fn-gpu-inputs`) lands only the
non-pipelined device-input step and the QSA/validation fixes. The one-ahead
loop and the `Runtime::advance_group` hook stay on `perf/batch-gpu-loop`,
default off.

- `batching::driver::Runtime::advance_group` is additive and defaults to
  `None`; `Driver::decode` calls it only when every active member is in the
  batch and no member needs a single step. The ordinary step + host-sample path
  is unchanged and remains the fallback.
- Flash-Next's `Owner::advance_group` finishes the pending round with the
  normal batch forward, then runs up to 8 device-feedback tokens per member,
  streams each token through that member's sink, and completes each cursor.
  The burst is bounded to 8 because peers are already active at B>=2.
- Rollback: `QWEN_SERVE_FN_BATCH_GPU_LOOP=0`. Default stays off until the serve
  A/B is measured; flip to default-on with the measured pass-through.

## 5. V4.1 and GLM: measured, and the premise does not hold

Measured B=3 (v41 lane `hostgap` plus local `wave_native`; GLM production
causal waves on):

| family | wall | GPU | encode | commit+readback | CPU input decode | host sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 32K | 25.1-27.1 | 23.7-24.0 | 4.5 | 20.8 | 0.001 embed + 0.30-0.41 Engram | 2.5 |
| V4.1 64K | 26.1 | ~24 | 4.8 | 21.3 | 0.30 Engram | 2.5 |
| GLM 32K | 27.9 | 25.0 | 2.1 | 0.8 | inside encode | 2.3 |

The Flash-Next 4.3 ms PLE gather is family-specific. V4.1's token embedding and
Engram row decode are **sub-millisecond**; GLM's host embedding gather is
inside a 2.1 ms encode. A device-inputs step would save **<0.5 ms** in either
family, so it is not worth building. The remaining shared host bubble is
**host sampling (~2.3-2.5 ms)** plus ~1 ms commit/readback.

The relevant next lever is therefore GPU sampling (the deferred loop) or a
cheaper/pooled host sampler, not moving the embedding/Engram gather. This
conflicts with the instruction to build the GPU-inputs step for whichever
family is bigger; the measurement says neither is.

Caveat: the GLM 49152 and V4.1 49152 cells did not complete in the packet
(V4.1's engine depth is 49152; its 64K hostgap is the long-context
corroboration). Thermal was level 1 at both ends.

## 6. Limits

- The A/B was measured at `fb4d81cb` (pre-rebase). The branch was then rebased
  onto the landed serve-owner-host `main` (`7cdebc1f`); a diff of every
  Flash-Next loop/probe file against the measured commit is empty, so the
  numbers transfer. The serve path was not re-measured after the rebase.
- The B=3 direct split is now measured (table above); the B=4 table is the
  independent reference.
- The native A/B and serve A/B are same-binary and same-prefill, but the GPU is
  shared with other lanes; thermal and lease state are recorded with each run.
- The serve burst still pays one host-staged forward per burst (the pending
  round). Moving that first frame to device inputs is the next increment and is
  already available as `encode_gpu`.
