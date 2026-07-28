# v0.648 A3B automatic GPU-greedy admission

Status: preregistration. No v0.648 implementation or scored observation exists.
The v0.645 observations are a prior only and may not enter this packet's
estimates.

## Question and authority

Can the exact GPU greedy reducer become the automatic serial-greedy policy for
one measured A3B compatibility identity while preserving product semantics and
improving the complete generation-loop wall?

A pass may authorize automatic GPU-greedy selection only for
`metadata_compatibility_v1` model/tokenizer IDs
`e6024ce53109fdf7/a4b0b26f8a8c9917`, temperature-zero serial generation, and
the exact implementation measured by this packet. `QWEN_GREEDY_GPU_ARGMAX=0`
must remain an unconditional rollback. The result grants no authority for
dense, A10B, another model or quant, positive-temperature sampling, prompt
lookup, MTP/speculation, prefix-cache policy, or any v0.646 algebraic leaf.

The packet's SHA-256 authenticates the evidence fixture. Runtime selection uses
a conservative local metadata compatibility fingerprint, not a content digest
or security boundary. The reducer is semantics-exact for arbitrary logits; an
identity alias can mis-scope performance, not greedy correctness. Do not add a
full-model content hash to this cold path.

## Runtime policy under test

Implementation must replace the process-global opt-in decision with two
separate decisions:

1. parse the environment mode once per process;
2. resolve automatic eligibility separately for each loaded model and request.

The frozen modes and reasons are:

| Environment | Model/request | Result | Reason |
| --- | --- | --- | --- |
| explicit falsy | any | CPU full logits | `disabled_by_explicit_rollback` |
| present invalid/non-Unicode | any | CPU full logits | `disabled_by_explicit_rollback` |
| explicit truthy | supported greedy request | exact GPU reducer | `force_enabled` |
| absent | admitted identity, temp 0, no prompt lookup | exact GPU reducer | `auto_metadata_a3b_v1` |
| absent | other identity | CPU full logits | `auto_identity_miss` |
| any | sampling or prompt lookup | existing non-GPU policy | `ineligible_request` |

Explicit rollback takes telemetry precedence over request ineligibility. Only
true absence selects Auto; every unrecognized present value fails closed.

An automatic decision must never inherit eligibility from the first model
examined in a process. Unit tests must cover every mode/identity/request cell
without mutating a latched environment.

The implementation must expose the effective decode policy, selection reason,
identity kind, model ID, and tokenizer ID. Resolving the metadata identity may
walk already parsed metadata and tensor descriptors, but must not hash model
contents or add source I/O.

## Frozen fixture and request

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  1,891 bytes, 419 tokens, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Generation: temperature zero, 128 selected tokens, 127 target transitions,
  fixed prefill chunk 1024, context capacity 1024, and token-limit termination.
- RAM insertion, automatic prefix admission, durable caching, prompt lookup,
  sampling, and speculative paths are disabled.

Both arms must pin these current-source defaults to `1`:

```text
QWEN_DECODE_GDN_FUSED_BETA_PROJ
QWEN_DECODE_ROPE_PAIR
QWEN_DECODE_MOE_GROUPED_FINALIZER
QWEN_DECODE_MOE_FUSED_FINALIZER
```

The deleted raw-Q flag must remain absent. Unflagged output-scale folding and
other current-source changes are bound by clean source state plus binary hashes.
This packet compares greedy policies with the same current stack; it cannot
certify or promote those shared transforms independently.

## Arms

Every arm-session is a fresh process. Inherited `QWEN_*` variables are archived
and then removed before the frozen values are applied.

- A sets `QWEN_GREEDY_GPU_ARGMAX=0`. It must report
  `greedy_argmax/disabled_by_explicit_rollback`.
- B leaves `QWEN_GREEDY_GPU_ARGMAX` absent. It must report
  `greedy_gpu_argmax/auto_metadata_a3b_v1`.

No other environment value, argument, model byte, prompt byte, or process
policy may differ.

## Semantic and identity gates

Before powered acquisition, run the exact A3B model-backed greedy-chain test on
the final measured source. It must emit the positive A3B execution marker and
preserve every selected token ID, bit-identical continuation logits, and
byte-identical active KV, GDN state, and convolution state. A skipped body,
missing Metal device, missing fixture, or missing marker is failure.

Then run one fresh 16-token A/B product-conformance pair. Require identical
stdout, canonical generated-token SHA-256, token counts, token-limit stop,
15 transitions, no consumed terminal transition, the frozen policy/reason, and
zero cache/prompt-lookup/sampling surface.

The generated-token digest is SHA-256 over this canonical byte stream:

```text
"qwen-generated-token-ids-v1\0" || count_u64_le || each_token_i32_le
```

Every JSONL output and stats row must include the digest, stop reason, and
`terminal_token_target_transition_consumed=false`. Byte-identical decoded text
alone is insufficient.

## Fixed acquisition

Run exactly eight paired arm-sessions in this order:

```text
AB, BA, AB, BA, AB, BA, AB, BA
```

There are 16 fresh processes, no replacement child, no retry, no adaptive
sample count, and no early efficacy or futility stop. Cool down five seconds
between arm-sessions.

Each arm-session executes five identical JSONL requests against the loaded
model:

1. request zero is the separately scored first-use safety observation;
2. requests one through four form the steady-state session median.

Every request must preserve 419 prompt tokens, 128 selected tokens, 127
transitions, token-limit stop, exact token digest/text, fresh sequence state,
and zero cache state. Request zero includes first use of the selected reducer
pipeline. The steady primary intentionally measures loaded repeated-request
generation; the first-use guard prevents that organization from hiding a cold
regression.

Before packet creation and every charged model/GPU child, require at least 85%
system-wide free memory, no competing inference process, and no thermal or
performance warning. A preflight failure after packet creation invalidates the
one-shot packet rather than authorizing a retry.

## Endpoints and statistics

For each session and each steady metric, reduce requests one through four to
their median. The primary is:

```text
decode_ms / generated_tokens
```

`decode_ms` is generation-loop wall including token decoding into the output
string. It excludes prefill, model load, JSON serialization, and stdout flush.
`transition_ms/decode_transitions` is descriptive only and cannot affect the
decision.

For pair `i`, compute `d_i = log(A_i/B_i)`. With exactly eight pairs:

```text
S   = exp(mean(d_i))
LCB = exp(mean(d_i) - 1.894579 * sd(d_i) / sqrt(8))
```

Use the same paired calculation for steady TTFT and request-zero first-use
decode/token. Report point estimates, one-sided bounds, log SD, and all pair
ratios for primary, first-use, TTFT, prefill, and diagnostic transition time.
Do not trim, pool, condition, or extend acquisition.

## Mechanical decision

Hard validity gates:

- clean source/build/runtime identity and exact model/prompt hashes;
- implementation, runner, preregistration, binaries, and model file identity
  unchanged from opening through sealing;
- exact-state and conformance gates pass with positive execution markers;
- every expected child/row/output exists exactly once and exits zero;
- exact token digest/text, count, terminal, policy, reason, cache, and request
  contracts pass within and across all sessions;
- every timing value is finite and positive;
- the steady paired prefill point estimate lies in `[0.97, 1.03]`.

Performance gates:

- steady primary `S >= 1.010` and one-sided 95% `LCB > 1.000`;
- request-zero decode/token one-sided 95% `LCB >= 0.970`;
- steady model-TTFT one-sided 95% `LCB >= 0.970`.

Decision precedence:

- `INVALID` for any identity, correctness, environment, completion, or row
  failure;
- `INCONCLUSIVE_CONTAMINATION` for a valid prefill-band miss;
- `KEEP_DEFAULT_OFF_REGRESSION` for a valid first-use or TTFT noninferiority
  miss;
- `KEEP_DEFAULT_OFF_EFFECT_MISS` for any valid primary efficacy miss;
- `ADMIT_A3B_AUTO_METADATA_V1` only when every hard and performance gate passes.

There is no resolution continuation. Invalid and completed packets must write a
decision, failure where applicable, complete artifact inventory, and final
one-shot seal.

Artifact root: `target/profiles/v0648-a3b-gpu-greedy-auto-p1/`.
