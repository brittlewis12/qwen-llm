# Prospective E0 Q4 v6 Distinct-Prompt Run Plan

Status: frozen before v6 acquisition. Prompt is the sole intended change to the
v4 computational treatment; administrative fixture, binding, source/executable,
and artifact identities necessarily change. V6 is development prompt-diversity
evidence only at the minimal one-token level, not a meaningful code task,
independent replicate, or held-out test.

## Objective And Binding

V4/v5 passed four-token generated history on the exposed `Hello` prompt under
both orders. V6 holds serial-first order, model/drafter, sampler, seed, request
length, and full-state checks fixed while changing the one-token prompt to
UTF-8 `def`.

Development fixture preparation recorded token ID `[727]`, canonical i32le SHA-256
`110a3fffafc92b01e8c967241e5bdf4651c93128160ae079e6b0a34a2a78f0f4`,
and UTF-8 SHA-256
`cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34`.
Binding-manifest schema v3 requires that exact prompt plus the complete config
and `serial_then_capture` order.

- Fixture ID: `e0-q4-def-generated-v6`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V6.json`, SHA-256
  `ee469af5bec9e1086bd4cc8c9997efeef97dca22638f504f912731640f32f4e0`.
- Reducer: `scripts/profile/dflash_e0_evidence.py`, SHA-256
  `43b9fa9216fabcff8f573152939be01f0d1b7980000a89f500fbf83de304c059`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Sampling: four requested tokens, sampler-v1, seed `0`, temperature `0.7`,
  top-k `200`, top-p `1.0`, min-p `0.05`, default stop token `[248046]`,
  warmup enabled.
- Arm order: `serial_then_capture`.

The frozen working directory is
`/Users/tito/code/qwen-llm-dflash-sampled-evidence`. Build with
`cargo build --locked` from the clean committed HEAD containing this plan.
Build/runtime commit and source state must match, dirty flags must be false, and
problems/overrides must be empty. No source change is permitted through
reduction.

## Bounded Evidence And Paths

With one prompt and four emitted tokens, the successful path again requires
three generated transitions, five reduced transitions, six snapshot pairs, and
exact `1,885,208,576`-byte full-raw sidecar coverage.

- Trace: `/tmp/qwen-dflash-e0-q4-v6-def-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v6-def-20260823.state.bin`.
- Reduction: `/tmp/qwen-dflash-e0-q4-v6-def-20260823.reduction.json`.

All outputs must be absent, exclusive-create, and canonically distinct from one
another and all inputs/executable. Exactly one producer and one reducer
invocation are authorized. Preserve every outcome without overwrite or retry.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V6.json --prompt def --tokens 4 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order serial-then-capture --output /tmp/qwen-dflash-e0-q4-v6-def-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v6-def-20260823.state.bin --fixture-id e0-q4-def-generated-v6 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v6-def-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v6-def-20260823.reduction.json
```

## Decision Rule

V6 passes only if producer/reducer exit zero, the reduction reports exactly one
passing run with `emitted=4` and `transitions=5`, the trace contains exactly
three `target_transition` rows, and the fully covered sidecar is exactly
`1,885,208,576` bytes. Earlier EOS fails this objective even if narrow parity
holds. Any mismatch or invalid artifact is preserved and stops acquisition.

A pass adds one alternate `def`-seeded one-token development trajectory at
generated-history depth. It does not establish meaningful code behavior,
reverse order for this prompt, prompt/seed breadth, sustained contexts, hidden
residual semantics, Q8/BF16, sparse-q/K0, verifier, economics, performance,
serving, held-out, or product authority.
