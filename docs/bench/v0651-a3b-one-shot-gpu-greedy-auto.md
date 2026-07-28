# v0.651 A3B one-shot exact-GPU-greedy admission

Status: preregistration. No v0.651 implementation or observation exists. No
v0.648-v0.650 row has authority in this packet.

## Question and authority

May exact GPU greedy become automatic when `QWEN_GREEDY_GPU_ARGMAX` is absent
for the authenticated disposable A3B single-turn CLI profile?

A GO may authorize only:

- the exact versioned disposable profile defined below;
- ordinary fresh single-turn CLI request index zero;
- temperature-zero, non-prompt-lookup generation requesting `128..=512` tokens;
- no durable checkpoint restore or publication and no warm follow-up;
- the current M4 Max host/profile constraints; and
- explicit `QWEN_GREEDY_GPU_ARGMAX=0` rollback.

It grants no JSONL, reusable-model, warm-follow-up, durable-cache, sampling,
prompt-lookup, speculative, concurrent, dense, A10B, other architecture,
quantization, tensor inventory, storage profile, or provider-serving authority.
The marker does not authenticate model-weight or tokenizer-content bytes. The
measured checkpoint hash is evidence-cell identity, not selector scope.
Explicit `=1` remains a force-only mechanism for otherwise eligible exact-greedy
requests; it does not override sampling or prompt-lookup ineligibility.

Any non-GO requires absent-environment automatic selection to return to
default-off before further product use. The candidate implementation commit
itself grants no authority.

## Candidate contract

The implementation must not repeat the prior `metadata_compatibility_v1`
tokenizer/metadata walk to decide policy. That walk cost about 18.4-19.3 ms in
v0.649 and was paid by both arms, so it did not charge the proposed default.

Instead, propagate a versioned realized-load marker from the existing exact
disposable loader decision:

```text
DisposableA3bQ4kmV1PreadLogicalExactRetainedPlanV1
```

The marker candidate exists only for `PreparedAutoSelection::Selected` with:

- profile `A3bQ4kmV1`;
- population `Pread`;
- destination `LogicalExact`; and
- proof `A3bRetainedPlan`.

Store it in `LoadedModel` only after `MetalModel::load_prepared` succeeds. A
forced physically similar storage path, a future profile variant, a prepared
but failed load, or any Auto miss must expose no marker. The marker is a narrow
performance-profile attestation, not a reducer correctness claim or a content
security digest.

Absent-environment automatic selection requires all of:

1. ordinary single-turn scope, request index zero;
2. no `--request-timing-warm-followup`;
3. no configured durable checkpoint store;
4. requested output tokens in the inclusive envelope `128..=512`;
5. temperature zero and no prompt lookup; and
6. the exact realized marker above.

Decision precedence is:

1. falsy, invalid, or non-Unicode environment value: explicit rollback;
2. sampling or prompt lookup: ineligible;
3. truthy value: explicit force;
4. absent value plus every narrow Auto condition: enabled; and
5. absent value otherwise: default off with a scoped reason.

JSONL must receive explicit reusable scope and remain off when the environment
is absent. The existing `greedy_gpu_selection_reason`, `decode_policy`, and
loader markers must make every decision legible. The candidate may add a
versioned ordinary stderr policy marker, but it may not silently change timing
schema 7 solely for this packet.

Freeze these scored or conformance reason strings:

- `disabled_by_explicit_rollback`;
- `auto_disposable_a3b_q4km_v1`;
- `default_off_requested_tokens_outside_128_512`;
- `default_off_reusable_path`;
- `default_off_no_disposable_profile`; and
- `force_enabled`.

Pure tests must cover rollback/force/absent parsing, the 127/128 and 512/513
token boundaries, single-turn/JSONL/warm-follow-up/durable scopes, request
index, missing and future profile variants, sampling, prompt lookup, and exact
marker matching.

## Frozen fixture and environment

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,134,528,992
  bytes, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  1,891 bytes, 419 tokens, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Warm-filesystem-cache process-cold contract; one fresh process and one request
  per scored observation.
- Fixed context 1024, prefill chunk 1024, temperature zero, zero RAM/durable
  prefix-cache admission, no prompt lookup, and no warm follow-up.
- Pin `QWEN_DECODE_GDN_FUSED_BETA_PROJ=1`, `QWEN_DECODE_ROPE_PAIR=1`,
  `QWEN_DECODE_MOE_GROUPED_FINALIZER=1`, and
  `QWEN_DECODE_MOE_FUSED_FINALIZER=1` in both arms.
- Remove every other inherited `QWEN_*` variable before applying frozen values.
- Archive the complete inherited `QWEN_*` environment before normalization.
  Scored A and B environments must differ only in
  `QWEN_GREEDY_GPU_ARGMAX`: A contains `0`, while B has no entry.
- A sets `QWEN_GREEDY_GPU_ARGMAX=0`; B leaves it absent. No other environment,
  argument, fixture, load policy, or process rule may differ.

Each scored command is the direct release `qwen` binary with:

```text
--model MODEL --prompt-file PROMPT --tokens N --temp 0
--prefill-chunk 1024 --max-context-tokens 1024
--prefix-cache-max-mib 0 --cache-prefix-auto-min-tokens 0
--request-timings UNIQUE_PATH
```

## Pre-acquisition gates

Before scored acquisition:

1. run the exact A3B model-backed greedy-chain state test and require its
   positive marker, exact token IDs, bit-identical continuation logits, and
   byte-identical active KV, GDN convolution, and GDN recurrent state;
2. run one fresh 128-token A/B single-turn conformance pair and require identical
   stdout, generated-token digest, stop reason, terminal semantics, and exact
   realized profile, with A reporting rollback and B reporting automatic GPU
   greedy;
3. run one absent-environment 128-token JSONL smoke and require reusable-path
   default-off;
4. run one `QWEN_GREEDY_GPU_ARGMAX=1` 128-token JSONL smoke and require the
   reusable explicit-force path;
5. run one absent-environment 128-token disposable smoke with
   `QWEN_GGUF_PARALLEL_COPY=0` and require the exact no-profile fallback reason;
6. run one explicit-rollback N512 reference and require exactly 512 selected
   tokens, 511 transitions, `token_limit`, no consumed terminal transition, and
   a sealed stdout/token digest used by every scored N512 row; and
7. pass the complete pure policy and realized-profile test matrix.

These are conformance gates, not timing observations. Any failure is `INVALID`.

## Scored acquisition

Run four fixed cells in this order:

1. `N=1`: exactly two pairs, `AB, BA`. B must remain CPU greedy because the
   requested output is below 128. This is a structural/fallback guardrail.
2. `N=16`: exactly two pairs, `BA, AB`. B must also remain CPU greedy. This
   guards a nontrivial but non-admitted short generation.
3. `N=128`: exactly eight pairs, `AB, BA, AB, BA, AB, BA, AB, BA`. B must
   automatically use GPU greedy and execute 127 transitions. This is the fixed-n
   inferential efficacy cell and the lower admission boundary.
4. `N=512`: exactly four pairs, `AB, BA, AB, BA`. B must automatically use GPU
   greedy and execute 511 transitions. This is the upper-envelope guardrail.

The packet therefore has 16 pairs and 32 fresh scored model processes. Cool down
five seconds between arm processes. There is no retry, replacement, adaptive
count, continuation, trimming, early stop, or extra scored row.

Before packet creation and every charged child, require at least 85% free
memory, no structurally identified inference competitor, and no thermal or
performance warning. Capture a second host snapshot after every child. Archive
each raw pre/post snapshot before validating it; a competitor or thermal or
performance warning arising during execution invalidates the row. Capture exact
child rusage. Initial and final complete model SHA-256 plus per-child
device/inode/size/mtime protect model identity; do not reread the complete 22 GB
file before every child. Every scored child must report zero block-input
operations and zero major faults; otherwise the warm-filesystem-cache contract
is invalid.

## External and internal endpoints

Launch `qwen` directly with piped stdout. Timestamp from immediately before
spawn through:

- `F`: first nonempty stdout read;
- `L`: monotonic timestamp of the last nonempty stdout byte-read;
- `E`: successful wait for that exact PID; and
- `X=E-L`: post-response teardown.

Use one monotonic clock for spawn, F, L, and E. Drain stdout and stderr
concurrently. Timestamp every nonempty stdout read; EOF is not L. After EOF,
concatenated stdout must equal the exact expected generated response bytes
followed by exactly one CLI-added newline. Archive stdout/stderr bytes and
hashes. Do not wrap the child in `/usr/bin/time`.

The internal timing row supplies `runtime_and_model_load_ms`, `prefill_ms`,
`ttft_ms`, `generation_ms`, `total_request_ms`, transition time/count, PSO
metrics, and Metal allocation samples. The current timing-only
`metadata_compatibility_v1` walk occurs after the final response flush; it may
affect `E` and `X`, but not `F`, `L`, complete generation, or internal total
request. E and X are diagnostic only and cannot reject admission.

For every endpoint, pair ratio is `A/B`; values above one favor B.

## Correctness and validity

Every scored row must preserve:

- clean matching source/build/runtime identity and unchanged binaries;
- the exact realized disposable profile and complete pread/logical-exact loader
  markers in both arms;
- 419 prompt tokens and requested/generated counts for its fixed cell;
- transitions `0`, `15`, `127`, or `511` for N `1`, `16`, `128`, or `512`;
- exact output bytes and generated-token digest across all arms/processes within
  each cell;
- `token_limit` and
  `terminal_token_target_transition_consumed=false`;
- A rollback telemetry in every cell;
- B short-output default-off at N1/N16 and exact Auto GPU telemetry at N128/N512;
- no cache, durable restore/publication, lookup, sampling, warm follow-up, or
  extra request surface; and
- finite ordered external/internal milestones and complete host evidence.

Any source, fixture, build, state, profile, process, host, row, output, cache,
terminal, timing, identity, inventory, or completion failure seals `INVALID`
without retry.

## Statistics and gates

For the fixed-n N128 cell, define `y_i=log(M_A/M_B)` for each paired endpoint.
With exactly eight pairs and seven degrees of freedom:

```text
S   = exp(mean(y_i))
LCB = exp(mean(y_i) - 1.894579 * sd(y_i) / sqrt(8))
UCB = exp(mean(y_i) + 1.894579 * sd(y_i) / sqrt(8))
```

Report every pair ratio, log SD, separate one-sided 95% bounds, raw endpoints,
order strata, rusage, and policy/load telemetry. Admission requires all of:

- external response-complete `L`: `S>=1.02` and `LCB>1.00`;
- internal complete generation per selected token: `S>=1.05` and `LCB>1.03`;
- internal total request: `S>=1.02` and `LCB>1.00`;
- process-cold first byte `F`: `LCB>=0.97`;
- runtime/load, prefill, and model-ready TTFT point estimates each inside the
  inclusive band `[0.97,1.03]`; and
- all correctness and validity gates above.

N1 and N16 are structural only: B must remain CPU greedy and every
output/policy/profile gate must pass. They carry no performance inference and
cannot authorize or rescue N128 efficacy.

N512 uses four paired logs with `t3=2.353363`. Its complete-generation and
external-L estimates must each be `>=1.02` with one-sided lower bounds `>1.00`;
its internal-total-request estimate must be `>=1.02` with lower bound `>1.00`,
and its F lower bound must be `>=0.97`. For each N512 endpoint:

```text
S512   = exp(mean(y_i))
LCB512 = exp(mean(y_i) - 2.353363 * sd(y_i) / sqrt(4))
```

This guard may reject the whole requested envelope but cannot rescue N128.
N128 and N512 empirically bracket the requested-length envelope; they do not
prove monotonic performance at every intermediate requested length.

Earlier packets inform the fixed sample counts only. No prior row contributes to
an estimate, bound, variance, MDE, gate, or decision. The packet does not claim
prospective power for every conjunctive endpoint; unresolved fixed-n bounds are
non-GO by construction.

## Mechanical decision

Decision precedence:

1. `INVALID` for any validity failure;
2. `INCONCLUSIVE_CONTAMINATION` for an N128 load/prefill/TTFT band miss;
3. `KEEP_DEFAULT_OFF_COLD_REGRESSION` for an N128 F noninferiority miss;
4. `KEEP_DEFAULT_OFF_ENVELOPE_GUARD_MISS` for an N512 gate miss;
5. `KEEP_DEFAULT_OFF_EFFECT_MISS` for any N128 L, generation, or total-request
   efficacy miss;
6. `ADMIT_A3B_ONE_SHOT_AUTO_V1` only if every prior gate passes.

Every non-GO has `authority=none`. A GO has only the narrow authority stated at
the start of this document. No observation from v0.648-v0.650 may be pooled,
substituted, trimmed, or used to relax a v0.651 gate.

Any verdict other than GO, including `INVALID` after packet creation, requires a
committed restoration of absent-environment default-off before any further GPU
or product experiment or candidate use. The sealed packet is never retried,
extended, or repaired.

Planned artifact root:
`target/profiles/v0651-a3b-one-shot-gpu-greedy-auto-p1/`.
