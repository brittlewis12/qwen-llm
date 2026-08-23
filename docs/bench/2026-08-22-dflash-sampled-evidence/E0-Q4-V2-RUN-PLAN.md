# Prospective E0 Q4 v2 Development Run Plan

Status: frozen before v2 acquisition. This is a distinct post-incident
development sentinel governed by `E0-INCIDENT-001.md`; it cannot replace or
repair v1 and grants no product or held-out authority.

## Fixture And Inputs

- Fixture ID: `e0-q4-token-major-smoke-v2`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V2.json`, SHA-256
  `9997f1bcbfbed4034e688c0c8f1eacbcdcd1336bb281fb501f53771fa860e066`.
- Reducer: `scripts/profile/dflash_e0_evidence.py`, SHA-256
  `319e0a34452fccf2d2461067dc60418e2c79c1f2c22a6bdb68ab82df56ecd09b`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Prompt bytes: UTF-8 `Hello`, length `5`, SHA-256
  `185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969`.
- Prompt tokens: count `1`, i32le SHA-256
  `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`.
- Sampling: one requested token, sampler-v1, seed `0`, temperature `0.7`
  (`0x3f333333`), top-k `200`, top-p `1.0` (`0x3f800000`), min-p `0.05`
  (`0x3d4ccccd`), default resolved stop token `[248046]`, warmup enabled.
- Arm order: `serial_then_capture` only for this fixture.

The acquisition binary must be built with `cargo build --locked` from the clean
committed HEAD containing this plan and the incident repair. Its recorded build
and runtime commit/source identities must exactly match with both dirty flags
false, status `match`, and no problems or overrides. The trace independently
records and hashes that executable; no source change is permitted between build,
acquisition, and reduction.

The frozen working directory is
`/Users/tito/code/qwen-llm-dflash-sampled-evidence`; relative executable,
binding, and reducer paths are interpreted only from that directory.

## Exclusive Artifact Names

- Trace: `/tmp/qwen-dflash-e0-q4-v2-serial-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v2-serial-20260823.state.bin`.
- Reduction: `/tmp/qwen-dflash-e0-q4-v2-serial-20260823.reduction.json`.

All three paths must be absent before acquisition. Reservation is exclusive;
the model, drafter, binding, trace, sidecar, and executable canonical paths must
be pairwise distinct.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V2.json --prompt Hello --tokens 1 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order serial-then-capture --output /tmp/qwen-dflash-e0-q4-v2-serial-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v2-serial-20260823.state.bin --fixture-id e0-q4-token-major-smoke-v2 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v2-serial-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v2-serial-20260823.reduction.json
```

## Decision Rule

Only an exit-zero strict reduction with
`development_gate=development_lockstep_passed`, exactly one input, and exactly
one run supports a development-only E0 sentinel claim. Producer status or text
alone never passes the gate. A reducer mismatch is retained as
`development_failed`. A malformed or nonreducible artifact has zero E0
authority, is quarantined without modification, and stops acquisition pending a
new incident review.

Even a passing v2 result says nothing about packed verification, rollback,
sparse-q correctness, acceptance, economics, performance, serving, product
integration, model-wide equivalence, held-out closure, or replication.
