# GLM FFN Encoder Split: Bounded Negative Result

## Decision

Close this candidate without promotion or another geometry/encoder sweep.
Actual-Q4 M5 validation passed the bitwise contract, but canonical native
decode showed no ordinary latency win. Control drift prevents a precise
effect estimate; the measurements do not establish a causal regression.
Keep the implementation default OFF and off main production. This standalone
document can be cherry-picked onto `093ea5dd` without the candidate code or its
preparation document.

## Source and Mechanism

Candidate source: `37b79547b246a9b9a7aa414a651552166b43329e`, based on
`093ea5ddcf418c0c9a0f8bc36184dd7733b8f72c`.

`QWEN_GLM53_FFN_ENCODERS_SPLIT=1` separates the ordinary single-token MoE FFN
using three serial-encoder boundaries on the same command buffer:

1. After FFN HC pre, before the router and routed experts.
2. Before shared-expert gate/up and down.
3. Before the routed/shared add and HC post join.

Operator order remains HC pre, router/routed/weighted sum, shared, then join.
Read-only normalized input feeds independent routed/shared branches with
separate existing scratch. Resource tracking may overlap independent work
across these boundaries; encoder counts alone do not prove GPU overlap.
There are no new shaders, arithmetic, buffers, repacking, precision changes,
command-buffer commits or waits. Packed prefill is unchanged. Stage recording,
probes, module hooks/captures, row output and per-dispatch profiling decline
the candidate; census remains eligible to verify its boundaries. Existing
poisoning, checked completion, route/sparse validation and position publication
remain in place.

## Actual-Model Validation

The six-shard GLM-5.3-Flash UD-Q4_K_XL smoke used one weight load with Metal
validation ON (`MTL_DEBUG_LAYER=1`). Elapsed time was 89.658818 seconds:

- 20 sessions and 660 full-vocabulary logit rows, all bit-exact.
- 40 common-prefix/final causal-state captures, all exact.
- 512 continuous plain decode steps and eight separate warmup censuses.
- Census confirmed one to 127 serial encoders per token; kernel inventory
  remained unchanged. This verifies the intended boundaries, not GPU overlap.

Both arms used common CURRENT packed prefill with the core Q8-wide and Q5-down
policies forced ON and the split OFF. Only the decode split scope differed.
At depths 512 and 4096, 32-step trajectories ran ordinary forwards continuously,
retaining returned logits for comparison after the trajectory. State captures
occurred only after the common prefix and final step, not between timed steps.
Comparisons covered recurrent state and valid visible KV/indexer history using
the existing capture API, not undefined scratch or capacity padding. Normal
admission, equal capacity 4128/R2048, ledger/position checks and route/sparse
validation remained enabled.

Validation-ON timings are diagnostic only, not canonical native performance:

| Depth | Baseline ms/token | Split ms/token |
| --- | ---: | ---: |
| 512 | 24.8644 | 26.1543 |
| 4096 | 25.5662 | 25.9497 |

The fixture identity check covers pinned shard names/sizes, metadata and
retained file stamps. It is not a full weight-payload hash verification.

## Canonical Native Campaign

Native execution used Metal validation OFF, the canonical `qwen-bench` build
from the candidate source, and an M5 Ultra with 80 GPU cores and 256 GiB on
macOS 27. The build used SDK 26.2; source/build identity was MATCH, the source
tree was clean, there were no build overrides, and features were `[]`.

Canonical benchmark binary SHA256:

```text
ba519250629289a8f1aaddfec3c4dfb7578fe8aa2be7c78037dc3b9cadd271e1
```

The core Q8-wide and Q5-down policies were ON in both arms. For each depth,
the heated bracket order was A0, S0, S1, A1, where A is split OFF and S is split
ON. There were 15-second gaps, two runs per row, an unchanged `pp512` control
and `tg128` decode at depths 512/4096. The 330.204-second campaign retained all
32 samples across 16 rows. Raw reports remain private; no private report paths,
prompts or token IDs are needed for this public closeout.

All `tg128` elapsed totals, in milliseconds (two samples per cell):

| Depth | A0 | S0 | S1 | A1 |
| --- | --- | --- | --- | --- |
| 512 | 2794.136, 2795.456 | 2843.373, 2840.921 | 2851.485, 2847.899 | 2866.127, 2889.363 |
| 4096 | 2710.129, 2711.752 | 2761.845, 2757.437 | 2760.589, 2762.114 | 2756.488, 2753.652 |

Pooling all four baseline samples (A0/A1) and all four split samples (S0/S1),
then dividing each elapsed total by 128 tokens, gives:

| Depth | Baseline ms/token | Split ms/token | Observed latency change |
| --- | ---: | ---: | ---: |
| 512 | 22.1584 | 22.2337 | +0.340% |
| 4096 | 21.3516 | 21.5664 | +1.006% |

Unchanged control drift from A0 to A1:

| Depth | `pp512` elapsed-time drift | Decode elapsed-time drift |
| --- | ---: | ---: |
| 512 | +11.182% | +2.968% |
| 4096 | +10.241% | +1.628% |

These pooled point estimates show no ordinary decode win in this bounded
campaign. The control movement is substantial relative to the candidate
difference, so neither a precise causal slowdown nor a precise near-zero
effect is established. Diagnostic stage spans are not recoverable exclusive
wall-time shares, and neither the validation result nor encoder census proves
that useful GPU overlap occurred.

## Closeout and Next Lever

Park this scheduling-only candidate default OFF and off main production. Do
not promote it, repeat the native campaign, or start another encoder-order or
geometry sweep on the strength of these results. The negative decision is
bounded to this candidate and campaign, not a claim that independent FFN work
can never overlap.

The next stronger lever is the existing actual-Q4 decode-fusion work: reduce
work/dispatch or intermediate traffic in the decode path rather than spending
another campaign on encoder partitioning alone. That is separate work with
its own correctness, admission and canonical native-conversion evidence;
this closeout authorizes no new implementation or benchmark run.
