# Measured router transfers: Flash-Next and GLM

Decision: enable the existing strict E8P32 F32 router at Flash-Next N2045,
and for GLM Fast chunks of N128/N512. No new shader, planner change, cache
approximation, or allocation is introduced. Exact GLM and other GLM widths
retain their previous routes; Flash's global router rollback remains active.
Both families retain geometry, dtype and device/pipeline capability checks.

This implements the cheap transfers identified in
`../2026-10-07-glm-flash-prefill-map/`. Larger expert-density work remains next.

## Flash: approximately 7% less whole-4K prefill time

The unchanged planner is 2048 + 3 + 2045. The final command previously used
the generic router despite having almost the same shape as the optimized
2048-row command. Candidate B changes only the N2045 eligibility decision;
A explicitly pins that width to the historical generic path, including after
promotion. The other commands and all attention/expert arithmetic are unchanged.

One resident native session (capacity8196), two token streams (prose and a
whole-text-repeated retained SSH fixture), separately warmed ABBA. Ordinary
fresh4096 calls include all three commands; suffix tests restore at2051
outside timing. Warm census witnesses are excluded from timing. Four following
teacher-forced tokens check the handoff; an 8192 witness is unscored for speed.

| Artifact / stream | Whole4096 GPU, A -> B | GPU latency reduction | Ordinary-call wall reduction |
| --- | --- | ---: | ---: |
| UD prose, final v2 repeat | 8799.41 -> 8164.28 ms | 7.22% | 7.07% |
| UD repeated SSH, final v2 repeat | 8821.19 -> 8164.48 ms | 7.44% | 7.29% |
| GSQ prose | 9028.11 -> 8388.22 ms | 7.09% | 6.96% |
| GSQ repeated SSH | 9182.50 -> 8395.98 ms | 8.57% | 8.45% |

GSQ SSH has first-arm drift; its two GPU pairs save9.99% and7.09%. The initial
UD prose packet likewise has a drifty A1 and an inflated12.16% mean; retain it
but do not use that as the headline. Stable pairs and the final UD repeat
support the conservative **about7% latency reduction**, roughly0.63 seconds.
Suffix savings (~12% in the stable UD cells) are already inside whole4096
savings, not an additional win. Fresh means reset state with resident weights,
not cold storage startup or HTTP TTFT. Repeated text is not an unseen agent
quality corpus. Smaller or otherwise shaped prompts may not reach N2045.

Predeclared performance screen: each stream must save at least3% mean
whole4096 GPU time, improve both GPU pairs and avoid a regression in either
ordinary-call wall pair. Both streams pass on both artifacts. The v2 UD
repeat also passes. We inspect drift separately rather than letting the
process exit status decide promotion.

Every artifact packet has five valid dispatch witnesses: exactly48 generic
router calls become strict calls, solely in the final N2045 command. All85
logit comparisons in each packet match, with KL0, finite values and matching
causal metadata through the continuations. All121 persistent tensor hashes
match. These observations support this existing-kernel transfer; they are
not a new blanket requirement for unrelated numerical optimizations.

### Diagnostic correction retained in evidence

The first v1 packets report six composite `state_digests_equal=false` results
per artifact. These differ **only** in `hyper_sha256_f32_le`, not in any
persistent tensor or causal metadata. The accessor read singleton scratch
after packed execution. Each later arm's observed hyper hash is its preceding
arm's continuation3 scratch; even A1 versus A2 differs. Packed execution uses
the final row of a separate packed hyper buffer, and the next scalar command
overwrites singleton scratch before use.

The raw v1 observations remain unchanged. Actual packed final-hyper equality
was not captured there and cannot be recovered retroactively. The test-only
observer now takes the actual last packed command's row count, or explicitly
selects singleton scratch after a scalar command. In the v2 UD repeat, all85
full endpoint/state comparisons, including the corrected hyper observation,
match. This fixes instrumentation, not production model execution.

## GLM: small, narrow and repeatable

The existing strict kernel is selected for Fast, F32, H4096/E288, rows128 or512,
and only if pipeline SIMD/threadgroup requirements are supported. Unsupported
devices and all other widths fall back. Exact lineage does not change.
The diagnostic's A arm forces generic even after promotion; B goes through
the same guarded encoder as production (and probes N32 diagnostically only).

| Rows | Final packet GPU, A -> B | GPU / wall reduction | Production decision |
| ---: | --- | ---: | --- |
| 32 | 544.37 -> 547.25 ms | -0.53% / -0.52% | Keep incumbent |
| 128 | 1049.42 -> 1034.37 ms | 1.43% / 1.42% | Enable |
| 512 | 2527.16 -> 2485.26 ms | 1.66% / 1.68% | Enable |

Both measured GPU/wall pairs improve at128/512. Earlier independent packets
in the prior map found1.51% and2.85% GPU savings; precise effects vary. This
is a roughly1-3% cheap transfer, not GLM's dominant bottleneck. No new3% GLM
gate is invented after seeing results. Routes, route weights and endpoint
logits match on the retained natural-token prefixes. N32's attractive
cache-hot leaf timing does not outweigh its whole-model regression.

## Verification and scope

- Workspace check with tests compiles without warnings; formatting and diff
  checks pass.
- Ten focused CPU tests cover scope boundaries, geometry/device guards,
  planner reachability, forced-arm restoration, timestamp validity and KL
  diagnostics.
- The existing all-enabled-width router differential, now including2045,
  passes under the normal lease with `MTL_DEBUG_LAYER=1`.
- Timing packets use release builds, no API validation, production lease and
  normal memory admission. Checkpoint/diagnostic CPU memory is admitted.
- Flash timing uses ordinary prefill, not a profiled or traced command. Reset,
  checkpoint work, hashing and continuation are excluded and labelled.
- No continuous benchmarking was run alongside another lease owner; queued
  work waited rather than bypassing the lease.

Source/metallib and shard bindings, prompts, every attempt and failure status
are in the raw JSONLs. `flash-*-router-2045.jsonl` are v1 before promotion;
`flash-ud-router-2045-v2.jsonl` tests the promoted width with the corrected
observer. `glm-router-production-abba.jsonl` is GLM v3, the guarded production
call site versus forced incumbent. Diagnostic completion and performance
screen results remain separate from this owner promotion decision.

Reproduce Flash using the ignored test
`qwen4exp_runtime::tests::prefill_map::router_2045::native_router_n2045` with
`FLASH_PREFILL_MODEL` and a new `FLASH_PREFILL_OUT`. Reproduce GLM with
`glm5_next_metal::packed::router_prefill::router_prefill_abba`, optional
`GLM53_GGUF`, and a new `GLM53_ROUTER_OUT`. Both take the production lease.
Compile with `cargo test --release -p qwen-llm --lib --no-run`, then select
only the named test using `--exact --ignored --nocapture --test-threads=1`.

`summarize.py` summarizes Flash; the prior map's `summarize.py` handles GLM.
Use `uv run` for either. Original raw evidence is never rewritten.

Adversarial review: `cx` `01a118ca-c0f6-7c00-a0dd-23ae962883a5`; implementation
jams used `01a11893-3352-75b1-9ec7-2ed940154a10` and
`01a11893-3352-79d3-8d8d-8343c0d8f293`. Review caught a summary conflating
greedy choices with performance and an omitted suffix summary; both were
fixed before execution. The stale hyper observer was localized and corrected
without hiding the initial observations.

Next: actual-route expert-down replay (GLM first, Flash by dtype cohort), and
measure any remaining Flash scheduling benefit against the now-efficient
router baseline. Do not add last packet's mixed-schedule savings to this win.
