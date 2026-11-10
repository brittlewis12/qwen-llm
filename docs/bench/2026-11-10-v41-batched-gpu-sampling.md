# V4.1 non-pipelined batched GPU sampling, 2026-11-10

Ships device sampling inside the existing V4.1 batch step (no pipelined loop).
M5 Ultra, production four-shard `GSQ-RCO IQ2_XS` artifact, 32K prefix.
Branch `perf/v41-batch-loop`, built `--profile iter`. Raw runs:
`~/qwen-runs/2026-11-10-v41-nps/` on the M5 host.

## Bottom line

- **Native, same binary, released preset (T=1, top_p=1, top_k=0, min_p=0),
  serial host sampling vs fast GPU sampling:** B=2 **22.55 -> 21.08 ms
  (+6.98%)**, B=3 **27.62 -> 25.43 ms (+8.58%)**, B=4 **35.08 -> 32.26 ms
  (+8.73%)**. The win is almost exactly the removed host-sampling term
  (1.46 / 2.03 / 3.19 ms); the logit readback is a smaller part on this
  shared-buffer artifact.
- **Exact policy (explicit seed) stays host-identical:** the exact GPU
  sampler (uniform drawn from the request's host stream) produced **0 token
  mismatches** against the host arm at every width, 32 steps x width. Greedy
  through the exact GPU sampler is bitwise equal to host greedy (0
  mismatches at every width).
- **Serve is neutral at the wall.** Three-agent ABBA at avg width 2.97, agent
  defaults, fast vs `QWEN_FAST_SAMPLER=0` (both through the sampled step):
  AB **103.32 -> 104.72 tps (+1.36%)**, BA **105.32 -> 103.64 (-1.59%)**, order
  spread 2.95%, all arms thermal 0. `gap_p50` 0.0260-0.0264 s; transition
  25.85-26.25 ms; residual host sampling 0.168/0.175 -> 0.139/0.159 ms. The
  sampling pool already overlaps host sampling and the forward dominates, so
  fast device sampling removes CPU work but not wall time at width ~3. The
  earlier +7.6% serve number was a bug (see the Serve A/B correction below).
- **The exact GPU sampler does not win the released preset** (B=2 22.81,
  B=3 27.75, B=4 36.86 ms; -0.5 to -5%) even though it skips the readback:
  its ~0.74 ms/member cost replaces host sampling. It is a correctness arm,
  not a throughput arm. Only the fast Gumbel sampler ships on the hot path.

## Native A/B method

`crates/qwen-llm/examples/v41_batch_gpu_sampling_ab.rs` restores two
independent state sets from one 32K snapshot and feeds identical
teacher-forced tokens (warm 16, 32 measured steps, width 2-4). Three arms
share one forward:

- **host:** `step_batch` reads full logits and the host `Sampler` samples.
- **fast:** `step_batch_sampled` with `BatchSampleMode::Fast` samples each
  member in a follow-up command (Gumbel-max over the whole vocabulary) and
  reads only tokens.
- **exact:** `step_batch_sampled` with `BatchSampleMode::Exact` runs the exact
  GPU sampler and reads only tokens.

Model depth 32768, 4096-token terminal window per sequence.

| B | host round ms | fast round ms | fast gain | exact round ms | exact mism. | greedy exact ms | greedy gain |
| -: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 22.55 | 21.08 | +6.98% | 22.81 | 0 | 21.38 | +7.04% |
| 3 | 27.62 | 25.43 | +8.58% | 27.75 | 0 | 25.83 | +5.03% |
| 4 | 35.08 | 32.26 | +8.73% | 36.86 | 0 | 33.53 | +1.52% |

`fast mismatches` is non-zero by design (53 / 79 / 104 of 64 / 96 / 128 token
vectors): the fast sampler is distribution-exact, not token-identical, to the
host at the same seed. Its correctness is pinned separately by the
chi-square/determinism tests in `metal/token_feedback.rs`.

## What ships

- `Forward::step_batch_sampled`: one forward with a per-member read mask, then
  one sampling command (fast and/or exact members) and a token-only readback.
  A `Host` member keeps its logits. A sampling failure after the forward
  poisons every state, so a caller cannot re-forward the same token.
- `Sampler::gpu_seed`: peeks one whole draw for the fast path, sharing the
  RNG stream with `gpu_uniform`/`commit_gpu_draw`.
- Serve: `QWEN_SERVE_V41_GPU_SAMPLE` (default on) routes batches of width >= 2
  through the sampled step. Per member: fast when
  `fast_sampler_allowed` (T>0, top_k=0, top_p=1, min_p=0) and the request
  named no seed; otherwise the exact host sampler. Width 1 stays serial.
  `Run.explicit_seed = request.seed.is_some()`, so the server's default seed
  is not treated as explicit.
- Rollbacks: `QWEN_SERVE_V41_GPU_SAMPLE=0` restores the host batch path;
  `QWEN_FAST_SAMPLER=0` (or `QWEN_GPU_SAMPLER_EXACT=1`) forces exact sampling
  for fast-eligible requests. The fast-sampler pipelines are warmed at start.

## Serve A/B

**Correction (2026-11-10, branch `perf/seed-intent`).** The first serve A/B
below compared `QWEN_SERVE_V41_GPU_SAMPLE=0` against on, but a bug made every
normalized request look explicitly seeded, so the "on" arm also used host
sampling. It was the host path plus noise, not the feature. The corrected A/B
follows; the original rows are kept only to explain the earlier record.

`scripts/bench/serve_realistic_pair.py`, three warm agents, agent defaults
(no sampling fields, no seed), 25 s windows, ABBA, `--warm-tokens 24000`,
`--context 196608`, `--max-thermal-pressure 1`. Plan
`serve-pair/plan-v41-25s.json`. Corrected pair: `serve-seed-pair/pair`,
`--flag QWEN_FAST_SAMPLER --reference 0 --candidate unset` (both arms keep the
sampled step; reference forces Host mode).

| order | arm | tps | gap p50 s | transition ms/round | host ms/round | thermal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AB | reference (`FAST_SAMPLER=0`) | 103.32 | 0.0264 | 26.254 | 0.168 | 0 |
| AB | candidate (fast) | 104.72 | 0.0262 | 26.023 | 0.139 | 0 |
| BA | reference (`FAST_SAMPLER=0`) | 105.32 | 0.0260 | 25.845 | 0.175 | 0 |
| BA | candidate (fast) | 103.64 | 0.0264 | 26.212 | 0.159 | 0 |

All four arms thermal 0, compile overlapped, order spread 2.95%. Directions are
opposite: **AB +1.36%, BA -1.59%**. The honest serve-level result is
**neutral at the wall** (within ±1.6%), with the residual host-sampling term
~0.02-0.03 ms lower. The sampling pool already overlaps host sampling and the
forward dominates a ~26 ms round, so fast device sampling removes CPU work but
not wall time at width ~3.

Superseded (pre-fix) rows, kept for the record:

| order | arm | tps | transition ms/round | host ms/round | thermal |
| --- | --- | ---: | ---: | ---: | ---: |
| AB | reference (`GPU_SAMPLE=0`) | 95.80 | 27.29 | 1.092 | 1 |
| AB | candidate (on, but host path) | 103.12 | 26.96 | 0.000 | 1 |
| BA | reference (`GPU_SAMPLE=0`) | 106.40 | 24.58 | 0.768 | 0 |
| BA | candidate (on, but host path) | 100.56 | 26.98 | 0.028 | 1 |

The 1.09 -> 0.00 "bubble removal" there was an artifact of *where* sampling
ran (scheduler after the round vs inside the step), not a wall win, and the
candidate was not using the fast sampler at all.

## Limits

- 32K only; 48K was not run this turn (thermal/lease budget).
- The native A/B samples serially on the host. That is the clean measurement of
  the sampler mechanism, but it is not the shipped serve cost: serve's pool
  already overlaps host sampling, so the serve wall win is ~neutral.
- Fast-sampler exactness is statistical (chi-square, 4/4 tests) and
  determinism-per-seed; it is not token-identical to the host by design.

