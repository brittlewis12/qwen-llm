# GLM Ordinary Workspace Selection

Ordinary `qwen run` and family benchmarks now select the largest admitted GLM
prefill workspace from the same bounded range already used by serving, rather
than imposing a 512-row cap. This improves ordinary single-turn requests, not
speculation. Existing resident serving policy is unchanged.

## Implementation

- Share candidates 2048/1024/512/256/.../1/0, capped to the useful prompt or
  benchmark prefill endpoint. Zero means serial execution.
- Before prefetch or weight loading, price the complete retained weights,
  session, and reserve with the existing model preflight.
- After loading, create the session with resident-aware admission. Only typed
  memory-pressure refusals permit a smaller workspace; geometry, telemetry,
  overflow, allocation, GPU and cancellation errors remain terminal.
- Never shrink requested context. Freeze the successful width for the session;
  benchmark repetitions freeze after their first successful creation, including
  warmup. Report the actual width, including a pressure fallback.
- Keep the low-level 512-row default, precision, rendering, sampling, serving
  prefix preference, and DFlash's existing width/admission policy unchanged.

The implementation is `glm5_next_metal::prefill_plan`, used by the ordinary CLI,
family benchmark adapter and serving's existing candidate wrapper. There is no
new shader, persistent weight representation, memory override, or runtime knob.

## Saved-Rollout Requests

M4 Max, 128 GiB, Metal API validation on. GLM-5.3-Flash UD-IQ3_XXS, the same
14 saved messages used in the native DFlash packet, rendered natively to 11,105
tokens. Temperature 1, top-p 0.95, top-k disabled, seed 42, 256-token cap.
Every run retains context capacity 11,360. Fresh processes use the production
Metal lease and an allowlisted environment, with no concurrent builds.

Both frozen binaries receive excluded 16-token warmups before A/B/B/A. There
is a five-second untimed pause after each measured request. All measured runs
report zero cold prefetch windows and zero prefetch bytes. Request time excludes
model loading and prefetch; process wall remains in [results.json](results.json).
Baseline source is `545318d3` with documentation-only changes; the candidate
adds this implementation. Both builds report dirty, not clean-build identities.
Binary and source hashes are retained in JSON; private inputs and logs are not.

| Run | Prefill rows | Prefill s | Decode s | Request s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 512 | 66.21 | 13.47 | 79.69 |
| B1 | 2048 | 52.56 | 9.87 | 62.44 |
| B2 | 2048 | 47.08 | 9.80 | 56.88 |
| A2 | 512 | 58.80 | 9.88 | 68.68 |
| Baseline mean | 512 | 62.51 | 11.68 | 74.19 |
| Candidate mean | 2048 | 49.82 | 9.84 | 59.66 |

Prefill latency falls **20.3%** (throughput +25.5%). Both paired prefill
comparisons improve by about 20%. Observed whole-request latency falls **19.6%**
on average, with paired reductions of 17.2% and 21.6%. A1 has a decode-time
outlier; **no decode improvement is claimed**, and not all observed request
savings are attributed to workspace selection.

All four runs produce the same 256-token output fingerprint and reach the cap.
This is a bounded saved-rollout continuation, not completed-game quality or
universal throughput evidence. The change does not promise faster cache hits in
serving, which already had adaptive workspaces.

## Chunk Behavior And Quality

Changing chunk boundaries can change Fast floating-point behavior even though
precision is unchanged. Reuse of the deployed serving range is not a claim of
general 512/2048 bit identity.

The fixed quality-v1 screen selects the first prose and code documents at 2048
tokens and the first long document at 4352, with 64 held true tokens each.
Selection is fixed before evaluation. Exact512, Fast512 and Fast2048 use the
same weights, context and continuation, with normal CPU/GPU memory admission.

| Stratum | Exact NLL | Fast512 NLL | Fast2048 NLL | Exact / Fast hits |
| --- | ---: | ---: | ---: | ---: |
| Prose | 1.738891 | 1.730009 | 1.730009 | 39 / 38 |
| Code | 1.233664 | 1.239073 | 1.239073 | 46 / 46 |
| Long | 2.964874 | 2.957709 | 2.957709 | 25 / 26 |

Fast512 and Fast2048 have identical per-token NLL and hit vectors over all 192
held positions; their measured KL is zero. Their differences from Exact are
also unchanged. This is descriptive evidence, not formal noninferiority or a
raw recurrent-state/logit bit-identity claim.

The existing real-model adaptive serving witness passes actual 2048-row
allocation, same-schedule cold/snapshot-hit logits and output equality,
cancellation at a committed 2048-token boundary, identical retry, and a
three-token suffix with live prefix reuse. At its 4096-token capacity, session
buffers are 853,688,320 bytes at 512 rows and 2,674,622,464 at 2048: about
**1.70 GiB more**. This is buffer pricing, not process peak memory. Its cancelled
2048-row command takes about 8.1 seconds; larger chunks have longer cancellation
intervals. No memory limit was increased to make either witness run.

## Verification And Reproduction

- Shared planner CPU tests: 4 pass, including full-weight/resident pricing,
  fixed context, zero/short bounds, pressure-only retry and final refusal advice.
- CLI library / `qwen` / `qwen-bench` unit suites: 119 / 957 / 90 pass.
- Header-only sparse/generic-HC ledger, cross-width quality, and real adaptive
  cancellation/reuse witnesses each pass.
- A real family `pp2048` smoke run reports `prefill_chunk=2048` and
  `packed_fast+last_logits`; this smoke is not a separate comparative speedup.

Ordinary request command, with fresh output files:

```sh
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$QWEN" run \
  --model "$GLM_TARGET" --messages "$PRIVATE_MESSAGES" \
  --max-tokens 256 --max-context-tokens 11360 --temp 1 --seed 42 \
  --request-stats-jsonl "$NEW_STATS"
```

Quality witness: ignored library test
`glm5_next_metal::tests::quality::workspace::workspace_quality_screen`, with
`GLM53_GGUF`, a new `GLM53_WORKSPACE_REPORT`, Metal validation and lease waiting.
State witness: ignored `qwen` binary test
`serve::backend_glm5_next::tests::gpu_adaptive_prefill_cold_hit_and_continuation`.
