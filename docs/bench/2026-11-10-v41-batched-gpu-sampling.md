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
- **Serve A/B is small and thermal-confounded; the sampling pool already
  hides most of the host bubble.** Three-agent ABBA at avg width 2.97, agent
  defaults: AB **95.80 -> 103.12 tps (+7.6%)**, BA **106.40 -> 100.56
  (-5.5%)**, order spread 13.1% with opposite directions and thermal 1 on
  three arms. The clean signal is the bubble itself: `host_ms_per_round`
  **1.092 -> 0.000** (AB) and **0.768 -> 0.028** (BA), transition **27.29 ->
  26.96 ms** in the same-thermal AB pair. In serve the released path samples
  on the pool's worker threads, so removing host sampling buys ~0.3-0.6 ms
  rather than the native ~2 ms.
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

`scripts/bench/serve_realistic_pair.py`, three warm agents, agent defaults
(no sampling fields, no seed), 25 s windows, ABBA, `--warm-tokens 24000`,
`--context 196608`, `--max-thermal-pressure 1`. Plan
`serve-pair/plan-v41-25s.json`.

| order | arm | tps | transition ms/round | host ms/round | thermal |
| --- | --- | ---: | ---: | ---: | ---: |
| AB | reference (`GPU_SAMPLE=0`) | 95.80 | 27.29 | 1.092 | 1 |
| AB | candidate (on) | 103.12 | 26.96 | 0.000 | 1 |
| BA | reference (`GPU_SAMPLE=0`) | 106.40 | 24.58 | 0.768 | 0 |
| BA | candidate (on) | 100.56 | 26.98 | 0.028 | 1 |

The AB pair holds thermal constant and shows the intended effect (bubble
removed, transition 0.33 ms lower, +7.6% throughput). The BA pair is
thermal-confounded: its reference ran at thermal 0 and its candidate at
thermal 1. Read this as "the bubble is gone; the serve wall win is inside
thermal noise", not as a throughput claim. The native A/B above is the clean
measurement of the mechanism.

## Limits

- 32K only; 48K was not run this turn (thermal/lease budget).
- The native A/B samples serially on the host, which is the pre-pool cost. A
  serve comparison against a *serial-sampling* serve build would show the
  larger win but that build is not the shipped default.
- Fast-sampler exactness is statistical (chi-square, 4/4 tests) and
  determinism-per-seed; it is not token-identical to the host by design.
