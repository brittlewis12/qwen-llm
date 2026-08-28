# Flash-Next Selected-Prefill Quality Preregistration

Status: **PREREGISTERED / NOT ACQUIRED**. Freeze this protocol before running
any candidate arm. Changes after observing outputs create a new packet.

## Decision

Choose whether generic selected-packed prefill can replace the default-safe
packed-dense plus scalar-selected path. Evaluate F32 HC down only as a
hierarchical challenger. Upstream BF16 is optional later calibration, not this
packet's missing authority.

Evidence authority, in order:

1. held-out teacher-forced NLL;
2. frozen known-answer correctness and answer probability;
3. observed-token top-1 and open greedy sentinels;
4. same-token, same-quant llama.cpp triangulation; and
5. component oracles, selector overlap, and margins for localization only.

Neither default-safe nor llama.cpp is numerical truth.

## Frozen Inputs

Before acquisition, record the clean source commit, model revision and every
shard hash, tokenizer artifact identity, llama.cpp lock, token manifests,
scoring-source digest, bootstrap seed, and this document's digest.

Natural corpus:

- three document-disjoint prompts at each context length
  `2179/2563/3075/4099`;
- selected suffixes exactly `128/512/1024/2048` tokens;
- 96 held-out continuation tokens per prompt, for 12 documents and 1,152 scored
  tokens; and
- no repository text, roadmap text, benchmark or tuning prompt, prior diagnostic
  prompt, or near-duplicate document.

Freeze token IDs before any arm output is observed. Score token one from prefill
logits, then feed each observed token teacher-forced to score the next. Feed the
last observed token once more only for a terminal state/logit digest.

Additional fixtures:

- one N=2,051 scope control with no selected rows;
- eight N=4,099 nonce retrieval tasks, four single-hop and four two-hop;
- single-hop evidence beginning near token indices 256, 1,280, 2,304, and 3,584;
- two-hop evidence pairs `(256,2304)`, `(512,3584)`, `(1280,3072)`, and
  `(1792,3584)`; and
- counterbalanced distractors with frozen one- or two-token answers, scored by
  answer-sequence NLL and at most eight greedy answer tokens.

Open 32-token greedy replay on one natural prompt at each shape is descriptive
only.

## Arms And Order

- **A — default-safe incumbent:** packed through position 2,050, then scalar
  selected overflow.
- **B — generic candidate:** selected packed execution through the prompt.
- **C — F32-HC-down challenger:** B plus exactly two selected-command HC down
  substitutions per layer.
- **D — llama.cpp triangulation:** the same GGUF and exact manifest token IDs.

The four selected suffix sizes are multiples of 128, satisfying C's current
test-only kernel contract. At N=2,051, A/B/C must be bit-identical and C must
record zero substitutions.

Across the 12 natural prompts, use `ABC`, `BCA`, `CAB`, `ACB`, `CBA`, and `BAC`
exactly twice. Reset the runner and explicitly zero persistent state before every
local arm. Repeat one N=4,099 prompt in reverse arm order after the cohort;
same-arm logits and state must be bit-identical. Run D separately in a
manifest-derived prompt permutation and never compare its timing with local
timing.

Semantic runs keep composition and QSA captures off. Separate selector-support
runs may enable them only after capture-off/on equality at that shape.

## Metrics

Compute observed-token NLL as F64 `logsumexp` over finite F32 logits. Report
token-weighted, document-level, and shape-level means. Primary paired deltas are
`NLL(B)-NLL(A)` and `NLL(C)-NLL(A)`.

Use 100,000 paired document-cluster bootstrap draws stratified by shape, with a
frozen seed. Because there are two candidates, use one-sided 97.5% percentile
confidence bounds.

Also report:

- observed-token top-1 hits and paired net change by document and shape;
- retrieval exact passes and correct-answer-sequence NLL;
- open greedy token IDs;
- endpoint and terminal state/logit digests; and
- selector status, counts, ID validity/order, overlap, first-flip position, and
  margin quantiles on support runs.

Local RMS, cosine, selector overlap, and margins have no semantic threshold.

## Gates

NLL noninferiority for each candidate requires all of:

- one-sided 97.5% upper bound at most `+0.010 nats/token`;
- every shape point estimate at most `+0.020`; and
- no document delta above `+0.050`.

A point-estimate pass with a confidence-bound miss is `HOLD`; extend the frozen
cohort without retuning or replacing its documents. Any extension is a new
preregistration with fresh document-disjoint inputs and an explicit
multiplicity/sequential-error rule; do not append and retest under this packet.

Known-answer gates:

- candidate total exact passes may not be lower than A;
- a candidate failure where both A and D pass is a hard kill; and
- token-weighted mean correct-answer NLL across all frozen answer tokens may
  worsen against A by at most
  `0.050 nats/answer-token`.

Report the unweighted per-task answer-NLL mean as a secondary statistic.

Each retrieval prompt instructs the model to emit only its frozen nonce answer.
An exact pass requires the complete one- or two-token answer sequence followed
by a producer-declared stop token within eight generated tokens. No text, case,
or whitespace normalization is allowed.

Observed-token top-1 and open greedy divergence are descriptive only and cannot
override NLL or known-answer gates. Token predictions within a continuation are
clustered and must not be treated as independent significance units.

Structural hard kills are any nonfinite logits, runtime failure, invalid token,
selector status, selected count other than 512 where selection is active,
out-of-range or non-cache-ordered IDs, wrong committed length, persistent-state
inconsistency, capture observer failure, or treatment-topology mismatch.

## Disposition

- B passes every gate: generic selected-packed receives correctness `GO`.
- C passes while B fails: C advances only to a separate performance/cost gate.
- B and C both pass: choose C only if the 95% upper bound for
  `NLL(C)-NLL(B)` is below `-0.005 nats/token` under the same paired,
  shape-stratified percentile document bootstrap, C passes every retrieval task
  B passes, and its later cost gate passes; otherwise choose simpler B.
- Neither passes: selected-packed remains default-off. Diagnose frozen worst
  cases without tuning on this corpus.

Any passing arithmetic policy still needs an uninstrumented performance packet
before the default changes. The known selected suffix leverage is large enough
that correctness, not another kernel optimization, remains the first decision.
