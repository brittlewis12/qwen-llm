# Prospective E0 Q4 v5 Generated Reverse-Order Run Plan

Status: frozen before v5 acquisition. Arm order is the sole intended change to
the v4 computational treatment. Fixture, binding, source/executable, and artifact
identities necessarily change. V5 is a paired development order diagnostic, not
an independent replicate or untouched hypothesis test.

## Objective And Binding

V4 passed three intermediate generated target transitions under
`serial_then_capture`. V5 asks whether the same prompt, generated trajectory
contract, sampler, seed, request length, and full-state checks remain lockstep
when the multi-hidden arm executes first.

Binding-manifest schema v3 requires the complete prompt/config object and
`capture_then_serial` order in both producer and reducer.

- Fixture ID: `e0-q4-generated-reverse-v5`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V5.json`, SHA-256
  `fe7c1e0c6550a151626a07b34bfc57241de8fb81d95c1a20c12c99077a960884`.
- Reducer: `scripts/profile/dflash_e0_evidence.py`, SHA-256
  `43b9fa9216fabcff8f573152939be01f0d1b7980000a89f500fbf83de304c059`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Prompt: UTF-8 `Hello`, one token, token i32le SHA-256
  `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`.
- Sampling: four requested tokens, sampler-v1, seed `0`, temperature `0.7`,
  top-k `200`, top-p `1.0`, min-p `0.05`, default stop token `[248046]`,
  warmup enabled.
- Arm order: `capture_then_serial`.

The frozen working directory is
`/Users/tito/code/qwen-llm-dflash-sampled-evidence`. Build with
`cargo build --locked` from the clean committed HEAD containing this plan.
Build/runtime commit and source state must match, dirty flags must be false, and
problems/overrides must be empty. No source change is permitted through
reduction.

## Bounded Evidence And Paths

The successful four-token path again requires one prompt, three generated
transitions, one boundary, one continuation, six snapshot pairs, and exact
`1,885,208,576`-byte sidecar coverage. Full raw evidence remains below all v4
storage/RAM redesign thresholds.

- Trace: `/tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.state.bin`.
- Reduction:
  `/tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.reduction.json`.

All output paths must be absent, exclusive-create, and canonically distinct from
each other and all inputs/executable. Exactly one producer and one reducer
invocation are authorized. Preserve every outcome without overwrite or retry.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V5.json --prompt Hello --tokens 4 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order capture-then-serial --output /tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.state.bin --fixture-id e0-q4-generated-reverse-v5 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v5-generated-reverse-20260823.reduction.json
```

## Decision Rule

V5 passes only if producer/reducer exit zero, the reduction reports exactly one
passing run with `emitted=4` and `transitions=5`, the trace contains exactly
three `target_transition` rows, and the fully covered sidecar is exactly
`1,885,208,576` bytes. Earlier EOS fails this objective even if narrow parity
holds. Any mismatch or invalid artifact is preserved and stops acquisition.

A v5 pass plus v4 would show no observed within-run generated-history parity
failure under either order on this same exposed fixture. It would not generally
exclude order effects or establish prompt/seed/quantization diversity, hidden
residual semantics, sparse-q/K0, verifier, economics, performance, serving,
held-out, or product authority.
