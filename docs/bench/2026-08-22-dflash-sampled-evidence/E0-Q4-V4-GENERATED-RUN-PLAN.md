# Prospective E0 Q4 v4 Generated-History Run Plan

Status: frozen before v4 acquisition. V4 is a development depth diagnostic on
the already exposed `Hello` fixture, not an independent replicate or untouched
hypothesis test. It adds generated-token consumption while retaining the full
raw state-evidence format.

## Objective And Machine Binding

V2/v3 contained only a prompt transition and pending-token continuation. V4
requests four emitted tokens so that three sampled tokens are subsequently
consumed by ordinary and multi-hidden target sessions. This exercises growing
KV/GDN/conv state and DFlash hidden-context history at generated transitions.

Binding-manifest schema v3 machine-binds the entire prompt and config object in
addition to assets, geometry, fixture labels, snapshot ABI, and exact arm order.
A one-token invocation, wrong seed, wrong sampler setting, alternate prompt, or
reverse order must fail before Metal initialization or in the strict reducer.

- Fixture ID: `e0-q4-generated-history-v4`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V4.json`, SHA-256
  `64a09cfda9bd8944f236208f95d55c3ad9e5662750ddaabf38cd0a86b5545c6f`.
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
- Arm order: `serial_then_capture`.

The frozen working directory is
`/Users/tito/code/qwen-llm-dflash-sampled-evidence`. Build with
`cargo build --locked` from the clean committed HEAD containing this plan.
Build/runtime commit and source state must match, dirty flags must be false, and
problems/overrides must be empty. No source change is permitted through
reduction.

## Bounded Evidence Cost

For one prompt token and four emitted tokens, the successful token-limit path
contains one prompt transition, three intermediate generated target transitions,
one terminal boundary, and one continuation. It writes six serial/capture
snapshot pairs with exact expected sidecar size `1,885,208,576` bytes. Estimated
trace size is about 25 MB and reducer peak memory about 2.5 GB.

The full-raw format remains below the predeclared redesign thresholds: 8 GiB per
sidecar, 20 GiB per fixture batch, 25% of free evidence-volume storage, and 25%
of physical RAM for reduction. No compression, checkpoint substitution, or
hash-only weakening is authorized.

## Exclusive Artifact Names

- Trace: `/tmp/qwen-dflash-e0-q4-v4-generated-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v4-generated-20260823.state.bin`.
- Reduction: `/tmp/qwen-dflash-e0-q4-v4-generated-20260823.reduction.json`.

The trace, state, and reduction paths must be absent. Their exclusive-create
reservations and all canonical input/output/executable paths must be pairwise
distinct. Exactly one producer and one reducer invocation are authorized; every
outcome is preserved without deletion, overwrite, or retry.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V4.json --prompt Hello --tokens 4 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order serial-then-capture --output /tmp/qwen-dflash-e0-q4-v4-generated-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v4-generated-20260823.state.bin --fixture-id e0-q4-generated-history-v4 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v4-generated-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v4-generated-20260823.reduction.json
```

## Decision Rule

The generated-history objective passes only if all conditions hold:

- producer and reducer each exit zero;
- the strict reduction reports exactly one run,
  `development_gate=development_lockstep_passed`, `emitted=4`, and
  `transitions=5`;
- the trace reports three `target_transition` rows and exact parity at every
  prompt/generated/boundary/continuation event;
- the state sidecar is exactly `1,885,208,576` bytes with full authenticated
  coverage.

An earlier EOS may still produce valid narrow parity evidence, but it fails this
generated-history objective and is never retried under v4. A reducer mismatch is
retained as `development_failed`; malformed/nonreducible evidence has zero E0
authority. Any outcome stops v4 acquisition.

A pass establishes only generated-history development parity on this exact
serial-first fixture. Reverse order, prompt/seed diversity, sustained histories,
hidden residual semantics, Q8/BF16, sparse-q/K0, packed verification, rollback,
acceptance, economics, performance, serving, held-out, and product claims remain
open.
