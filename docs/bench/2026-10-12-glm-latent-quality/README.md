# Half latent MMA: fixed task subset

Reoriented on `73d13835`, with the Oct 11 GLM arithmetic and the updated
MSL 3.1/macOS 14 target. The initial task screen and forced-arm timing preceded
the Half policy promotion; production verification followed. Retired
down/gate-up/KDA experiments remain archived.

Before viewing outputs, fix the following subset of the committed
`scripts/reference/glm53/quality-v1.json`:

- First prose and code documents, prefix 2048; first long document, prefix 4352.
  Score each document's 64 real next tokens (NLL and top-1 hits).
- Tool tasks T03, T10, T11, T12, greedy and seed 11, existing 256-token cap.
  Use the existing strict `quality_analysis.tool_correct` contract unchanged.
- Maya weights, Fast Half, workspace 2048, fresh sessions and identical chunk
  boundaries. Force scalar or permit MMA throughout every prefill chunk;
  decode is unchanged. Verify frozen-token compatibility against all complete
  tokenizer fields in the pinned metadata-only UD shard.

This is a descriptive paired subset, not the historical full-cohort verdict
and not a quality non-inferiority claim. Old Fast output is not ground truth.
The task packet itself does not change production selection; the subsequent
integration enables eligible Q6 MMA in Fast Half alongside existing Fast F32.

The known approximately 9% Half MMA gain motivates this quality comparison
before additional profiling. Another parent-stage profile cannot answer the
remaining task-quality question. Refresh performance separately without the
task packet's per-token recording or dispatch census.

## Results and narrow promotion

The [fixed task analysis](maya-analysis.json) reports lower mean NLL on all
three documents: overall MMA-minus-scalar NLL is -0.009979935, with two fewer
top-1 hits out of 192. Both arms score 8/8 tool generations, with identical
emitted token sequences. This remains descriptive evidence from a small fixed
subset, not a full-cohort or non-inferiority verdict.

The refreshed [same-binary forced-MMA ABBA](maya-current-abba.json) completed
all three cells: A/B ratios 1.0708 (512), 1.1056 (2048), and 1.0333 (129-row
suffix after 512). Its historical 1e-3 KL assertion still fails; the complete
measurements and failure records are retained. These are forced-arm results,
not a completed post-promotion production verification.

GLM now permits eligible Q6 grouped latent MMA for **Fast Half and Fast F32**.
DenseF32 and Exact retain scalar4; Q8 and decode are unchanged. Existing
geometry and pipeline/capacity fallback remains scalar4, including the ragged
MMA remainder. There is no artifact whitelist, production environment toggle,
or new quality gate. The task packet retains its explicit scalar/MMA scopes.

`glm5_next_metal::packed::latent_mma_ab::grouped_latent_production_ab` reuses
`GLM53_LATENT_MMA_AB_OUT` (new file) and optional `GLM53_LATENT_MMA_AB_DEPTH=4096`.
Its schema is `glm53.grouped_latent_production_ab.v1`: A forces scalar4; B uses
production selection with no override, including any untimed prefix. It
records finite full-vocabulary KL at the join and three common teacher-forced
decode steps without a numeric bound. The historical test and its bound remain
separate. Scalar8 remains a dormant test-only experiment; its scope suppresses
MMA, while a forced scalar4 baseline still wins.

## Production Verification

M4 Max 128 GiB, release, no Metal validation during timing, internal production
lease. A1/B1/B2/A2 after warmups; ratio is A wall / B wall. In
`maya-production-abba.json`, B has no override, including prefix construction.
A forces scalar throughout. Prefix construction is outside the suffix timer.

| Cell | Scalar A ms | Production B ms | Ratio |
| --- | ---: | ---: | ---: |
| Full512 | 2883.07 | 2700.67 | 1.068 |
| Full2048 | 10761.84 | 9676.86 | 1.112 |
| Suffix129 after512 | 1448.25 | 1396.72 | 1.037 |
| Suffix512 after4096 | 3538.78 | 3165.50 | 1.118 |

Warm census confirms 22 Q6 substitutions and the ragged scalar remainder.
Unlike historical suffix probes, these arms need not start from identical
prefix states. These are paired observations, not a precise universal speedup;
the deep A samples vary materially. The historical forced-policy refresh still
fails its unchanged KL assertion (maximum 0.012874). That failure is retained,
not relabeled as a pass or used as a task-quality veto.

`ud-production-control.json` is **incomplete**. Full512 and full2048 cells
complete with unchanged Q8 dispatch and bit-identical join/continuation
outputs, but N2048 times vary from 30.8 to 61.3 seconds. The process times out
before the third cell. Other owner tests/builds and heavy host paging were
observed afterward; no unique cause is established. Exclude its ratios from
performance conclusions. No manual process termination, memory-setting change
or cache purge was performed. Existing Q8 primitive/production tests also pass.

## Validation And Decision

- Ten grouped-latent and 87 GLM tests pass with Metal validation, including
  unoverridden production selection, scalar modes, ragged and fallback cases.
  Six analyzer CPU tests and all-target release checks pass.
- Maya adaptive serving cold/hit/cancellation/retry/short-suffix test passes
  with Metal validation: actual workspace2048, shared prefix2051, prompt2068.
  Cold/hit equality and continuation assertions exercise production dispatch.
  Validation-enabled durations are not performance evidence.
- Snapshot compatibility needs no new namespace: GLM snapshots are RAM-only
  and bound to the loaded weights instance, lineage, precision and schedule.
  They do not survive a binary replacement.
- No new production knob, artifact allowlist, weight expansion or persistent
  GPU memory. Q6 MMA retains F32 operands/accumulation; this changes reduction
  order, not stored weights or requested operand precision.

The task packet records complete tokenizer field, fixture, binary and source
hashes. It ran before formatting-only changes to its source and promotion of
the default. NLL deltas per document are -0.018024/-0.005903/-0.006013; top-1
hits are 111 scalar versus 109 MMA. Both metrics and the identical correct tool
outputs matter; the small screen does not prove general quality improvement.

Promote the reversible optimization on independent primitive checks, bounded
task evidence and repeated whole-prefill gains. Do not equate old Fast with
ground truth or demand a general non-inferiority proof for every changed
reduction. Preserve caveats and investigate actual task regressions if they arise.
