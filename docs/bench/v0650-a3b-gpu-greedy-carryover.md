# v0.650 A3B GPU-greedy immediate-carryover diagnostic

Status: preregistration. No v0.650 observation exists. This is a diagnostic
packet with no product, default, performance-admission, or benchmark authority.

## Question

Does one exact GPU-greedy 128-token generation cause more than 3% relative
degradation in the immediately following identical-prompt prefill, compared
with ordinary CPU full-logit greedy generation?

v0.649 showed strong complete-decode efficacy but failed its steady-prefill
contamination control during a bilateral middle-window slowdown. Request-zero
prefill was stable, while later prefill followed arm-specific generation. This
packet changes the measurement organization to isolate that carryover question.
It may not reinterpret, pool, trim, replace, or rescue any v0.648/v0.649 row.

Current product policy is default-off at
`15a7092779e81e190442fe89e6d59d6ee7e6d3f7`. A diagnostic result cannot change
that policy. Exact GPU greedy remains available only through explicit
`QWEN_GREEDY_GPU_ARGMAX=1`.

## Frozen fixture and policies

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  1,891 bytes, 419 tokens, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Fixed prefill chunk 1024, context capacity 1024, temperature zero, no prompt
  lookup, and zero RAM/durable prefix-cache admission.
- Both arms pin `QWEN_DECODE_GDN_FUSED_BETA_PROJ=1`,
  `QWEN_DECODE_ROPE_PAIR=1`, `QWEN_DECODE_MOE_GROUPED_FINALIZER=1`, and
  `QWEN_DECODE_MOE_FUSED_FINALIZER=1`.
- A sets `QWEN_GREEDY_GPU_ARGMAX=0` and must report
  `greedy_argmax/disabled_by_explicit_rollback`.
- B sets `QWEN_GREEDY_GPU_ARGMAX=1` and must report
  `greedy_gpu_argmax/force_enabled`.

Inherited `QWEN_*` variables are archived and removed before applying the
frozen values. No other environment, argument, fixture, or process policy may
differ.

## One-cycle process contract

Each arm observation is one fresh process and one loaded model. It executes
exactly two JSONL requests with the identical prompt:

1. `treatment`: select 128 tokens and execute 127 target transitions;
2. `sentinel`: select one token and execute zero target transitions.

The treatment prefill is causally pre-treatment. Its generation applies the A
or B policy. The sentinel prefill follows immediately and is the post-treatment
observation. The sentinel CPU-selects its only token from prompt logits and
stops before any target transition, so it never invokes the GPU reducer even
though B still reports the force-enabled policy.

Run exactly 16 independent pairs, 32 fresh processes, in order:

```text
AB, BA, AB, BA, AB, BA, AB, BA,
AB, BA, AB, BA, AB, BA, AB, BA
```

Cool down five seconds between arm processes. There is no retry, replacement,
adaptive count, continuation, early stop, or additional cycle. Requests are
within-process reduction inputs, not inferential replicates.

## Correctness and observability

Before acquisition, rerun the exact A3B model-backed greedy-chain state test and
require its positive marker, exact token IDs, bit-identical continuation logits,
and byte-identical active KV, GDN, and convolution state. Then run one fresh
16-token A/B conformance pair with identical stdout and generated-token digest.

Every treatment and sentinel row/output must enforce schema 7, exact prompt and
request IDs, policy/reason, token digest/text by request type, token-limit stop,
`terminal_token_target_transition_consumed=false`, and zero cache, sampling, or
prompt-lookup surface. Treatments require 128/127 selected/transition counts;
sentinels require 1/0. Digest/text equality is compared across arms and sessions
separately for treatment and sentinel.

Capture quiet-host evidence before and after each arm process. Post-creation
host evidence must be archived before validation. Record child user/system CPU
rusage and elapsed wall around only the Qwen child; these values are descriptive
and cannot trim, condition, invalidate, or classify an otherwise valid row.

Before packet creation and every charged child, require at least 85% free
memory, no structurally identified inference competitor, and no thermal or
performance warning. Any post-creation failure seals `INVALID` without retry.

## Endpoint and statistics

For arm session `X`, define:

```text
carry_X = sentinel_prefill_ms_X / treatment_prefill_ms_X
```

For pair `i`:

```text
d_i = log(carry_B_i / carry_A_i)
q_i = log(treatment_prefill_A_i / treatment_prefill_B_i)
```

Values above one on the carry scale mean GPU greedy leaves the following
prefill relatively slower. With exactly 16 pairs and 15 degrees of freedom:

```text
C   = exp(mean(d_i))
LCB = exp(mean(d_i) - 1.75305 * sd(d_i) / sqrt(16))
UCB = exp(mean(d_i) + 1.75305 * sd(d_i) / sqrt(16))
Q   = exp(mean(q_i))
```

Report every ratio, log SD, bound, raw prefill, treatment decode/token, command
wall, and child rusage. No metric other than `C`, its bounds, and baseline `Q`
may affect the decision.

## Mechanical decision

Decision precedence:

1. `INVALID` for any source, fixture, exact-state, conformance, process, row,
   output, cache, policy, host, identity, completion, or finite-value failure;
2. `INCONCLUSIVE_BASELINE_IMBALANCE` if `Q` is outside inclusive
   `[0.97,1.03]`;
3. `CARRYOVER_HARM_SIGNAL` if `LCB > 1.03`;
4. `NO_MATERIAL_CARRYOVER_AT_3_PERCENT` if `UCB < 1.03`;
5. `INCONCLUSIVE` otherwise, including equality at `1.03`.

Every valid verdict has `authority=diagnostic-only`. It cannot admit the
automatic policy. `NO_MATERIAL_CARRYOVER_AT_3_PERCENT` may authorize only a new
independent admission design with a structurally pre-treatment contamination
control. `CARRYOVER_HARM_SIGNAL` authorizes only attribution work while the
exact path remains explicit.

Artifact root: `target/profiles/v0650-a3b-gpu-greedy-carryover-p1/`.
