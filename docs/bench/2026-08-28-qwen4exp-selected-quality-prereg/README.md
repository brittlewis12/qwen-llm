# Flash-Next Selected-Prefill Quality Preregistration

Status: **PREREGISTERED V2 / NOT ACQUIRED**. Freeze this protocol before
running any candidate arm. Changes after observing outputs create a new packet.

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
scoring-source digest, bootstrap seed, and this document's digest. Hash every
local model shard against the frozen release lock before the first model call;
retained file stamps alone are not semantic-evidence identity.

The clean-source contract includes every dynamically discovered Metal source
and the pinned Git commit/tree plus clean relevant scope for both workspace path
dependencies (`gguf-rs` and `llama-cpp-sys-2`). Bind those source manifests,
the exact test executable, and the embedded metallib into the evidence root.

Natural corpus:

- three document-disjoint prompts at each context length
  `2179/2563/3075/4099`;
- selected suffixes exactly `128/512/1024/2048` tokens;
- 96 held-out continuation tokens per prompt, for 12 documents and 1,152 scored
  tokens; and
- no repository text, roadmap text, benchmark or tuning prompt, prior diagnostic
  prompt, or near-duplicate document. Enforce this against the frozen pre-packet
  repository using whole-file and 256-word-window text audits plus exact token-ID
  five-gram comparison with recognized prior token fixtures.

Freeze token IDs before any arm output is observed. Prefill the prompt. For each
of the 96 continuation IDs, score that ID from the current logits and then feed
it exactly once. Logits after feeding continuation token 96 are terminal-only
and are not scored. Thus every arm performs exactly 96 continuation forwards,
not 97.

Additional fixtures:

- one N=2,051 scope control with no selected rows;
- eight N=4,099 nonce retrieval tasks, four single-hop and four two-hop;
- single-hop evidence beginning near token indices 256, 1,280, 2,304, and 3,584;
- two-hop evidence pairs `(256,2304)`, `(512,3584)`, `(1280,3072)`, and
  `(1792,3584)`; and
- matched decoy records or chains with target/decoy order counterbalanced across
  tasks, and frozen opaque one- or two-token answers scored by answer-sequence
  NLL and at most eight greedy answer tokens; and
- an exclusion audit proving every generated key, relay, and answer is absent
  from filler source and occurs only at its declared generated locations.

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

Force the production-default fast IQ3 gate/up and strict packed-router policies
through test-scoped overrides for every local operation. Reject all undeclared
`QWEN*` environment switches before loading the model, and record the effective
arithmetic policy in the global evidence binding.

The frozen operation list runs scope control first, then all natural semantic
arms, four open-greedy sentinels, all retrieval semantic arms, the reverse
replay, and finally D in its independent order. Selector-support captures are
not part of this packet. D pins llama.cpp merge commit
`6c84c7d5d8833c6e0df69628f75a0f599797934e` from support PR 27742.

Semantic runs keep composition and QSA captures off. Separate selector-support
runs may enable them only after capture-off/on equality at that shape.

Each run binding covers its canonical semantic payload, including scored rows,
aggregates, generated IDs, exact-pass result, topology, logits/state identity,
and treatment records. Bind the ordered 78-run sequence at the report root.
Reserve the destination before the first model call and publish a fully synced
temporary report by a no-clobber atomic operation; a failed run must not leave a
partial official report.

## Metrics

Require exactly 248,320 finite F32 logits in every scored row. Compute
observed-token NLL with a max-subtracted F64 `logsumexp` over the complete row;
never filter values. Report token-weighted, document-level, and shape-level
means. Primary paired deltas are `NLL(B)-NLL(A)` and `NLL(C)-NLL(A)`.

Use 100,000 paired document-cluster bootstrap draws stratified by shape, with a
frozen seed and one shared resample matrix for every contrast. For each draw,
visit shapes in ascending order and draw three document indices with replacement
using SplitMix64 rejection mapping into `[0,3)`, then average the resulting 12
document mean deltas. Because there are two candidates, use one-sided 97.5%
percentile bounds for B-A and C-A. Use a one-sided 95% bound for C-B. The upper
percentile is `sorted[ceil(p * 100000) - 1]`, without interpolation.

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

Each retrieval prompt instructs the model to emit only its frozen opaque answer.
An exact pass requires generation to begin with the complete one- or two-token
answer sequence and the immediately following generated ID to be a
producer-declared stop token. This is necessarily within eight generated tokens.
No leading or intervening token and no text, case, or whitespace normalization
is allowed.

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
