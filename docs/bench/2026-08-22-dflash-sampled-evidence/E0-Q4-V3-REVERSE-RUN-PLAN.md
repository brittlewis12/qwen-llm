# Prospective E0 Q4 v3 Reverse-Order Run Plan

Status: frozen before v3 acquisition. Arm order is the sole intended change to
the computational treatment from the passed v2 sentinel. Administrative
identities necessarily change: fixture and binding identity plus trace, state,
reduction, source-commit, and executable identities. V3 is a development
interference diagnostic, not an independent replicate, untouched hypothesis
test, product gate, or replacement for any prior artifact.

## Objective And Inputs

The sole new question is whether running the multi-hidden capture arm before the
ordinary serial arm reveals shared Metal queue, allocation, cache, or order
interference hidden by v2's `serial_then_capture` order.

- Fixture ID: `e0-q4-token-major-reverse-v3`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V3.json`, SHA-256
  `b22496e0fe483e2b515537d81f2682f5c80ba1ddfa348dddab2b8d9e7ef8874a`.
- Reducer: `scripts/profile/dflash_e0_evidence.py`, SHA-256
  `2db8c4d2df795a771be035493f3c032e945ab5234e03693b84942607f43d6eb5`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Prompt: UTF-8 `Hello`, length `5`, SHA-256
  `185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969`,
  one token with i32le SHA-256
  `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`.
- Sampling: one requested token, sampler-v1, seed `0`, temperature `0.7`
  (`0x3f333333`), top-k `200`, top-p `1.0` (`0x3f800000`), min-p `0.05`
  (`0x3d4ccccd`), default resolved stop token `[248046]`, warmup enabled.
- Arm order: `capture_then_serial`, required by the schema-v2 binding manifest
  and independently checked by producer and reducer. No other computational
  input differs intentionally.

The frozen working directory is
`/Users/tito/code/qwen-llm-dflash-sampled-evidence`. The binary must be built
with `cargo build --locked` from the clean committed HEAD containing this plan.
Its build/runtime commit and source identities must match, both dirty flags must
be false, and problems/overrides must be empty. No source change is permitted
between build, acquisition, and reduction.

## Exclusive Artifact Names

- Trace: `/tmp/qwen-dflash-e0-q4-v3-reverse-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v3-reverse-20260823.state.bin`.
- Reduction: `/tmp/qwen-dflash-e0-q4-v3-reverse-20260823.reduction.json`.

The trace, state, and reduction paths must be absent before acquisition. Their
exclusive-create reservations and the canonical model, drafter, binding, trace,
state, reduction, and executable paths must be pairwise distinct.

Exactly one producer invocation and one reducer invocation are authorized.
Every outcome and artifact is preserved without deletion, overwrite, or retry.
Any further invocation requires a new incident review, fixture identity, frozen
binding, and frozen artifact names.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V3.json --prompt Hello --tokens 1 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order capture-then-serial --output /tmp/qwen-dflash-e0-q4-v3-reverse-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v3-reverse-20260823.state.bin --fixture-id e0-q4-token-major-reverse-v3 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v3-reverse-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v3-reverse-20260823.reduction.json
```

## Decision Rule

Only an exit-zero strict reduction with exactly one trace, exactly one run, and
`development_gate=development_lockstep_passed` establishes within-run arm parity
under reverse execution order on this narrow fixture. Together with v2 it would
show no observed within-run parity failure under either order; it does not rule
out order effects generally. A reducer mismatch is retained as
`development_failed`. Any other producer or reducer outcome stops acquisition
and has only the authority assigned by the strict reducer; a malformed or
nonreducible artifact has zero E0 authority.

Even a pass does not add generated-transition, longer-context, multi-seed,
quantization-wide, hidden-semantic, packed-verifier, sparse-q, acceptance,
economics, performance, serving, held-out, or product authority.
