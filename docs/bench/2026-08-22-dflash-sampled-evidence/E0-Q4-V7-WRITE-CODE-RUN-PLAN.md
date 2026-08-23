# Prospective E0 Q4 v7 Two-Token Prompt Run Plan

Status: frozen before v7 acquisition. V7 is the final incremental development
E0 row in this phase. It changes the prompt from v4 to the short two-token
instruction `Write code`, while holding serial order, sampler, seed, request
length, assets, and full-state evidence fixed. It is not held-out or independent.

## Objective And Binding

Unlike the one-token development prompts, v7 exercises two consecutive prompt
transitions before three generated target transitions. The four-token output is
too short to establish code quality or realistic instruction-following; the
only claim is a short two-token string with repeated prompt-state growth.

Fixture preparation recorded token IDs `[7734, 1970]`, canonical i32le SHA-256
`39400964f33473f82888817289bca3bba220e3e9f51856ce1e5f88dc8b411f2c`,
and UTF-8 SHA-256
`b365a7d68fd699d7938042031965dac164cac0696dab1fb9be239b62c31e2734`.
The retained clean CPU fixture `E0-Q4-V7-TOKENIZER-FIXTURE.json`, SHA-256
`479dd9439abe9138aea99f886dac02fcabd4cc019edc5e87edb2ce4966874b00`,
binds those values to exact native/llama.cpp encode/decode parity, target asset,
build/source identity, executable, command, and `add_special=false`.
Binding-manifest schema v3 requires the exact prompt, complete config, and
`serial_then_capture` order.

- Fixture ID: `e0-q4-write-code-v7`.
- Fixture role: `development-sentinel`.
- Binding: `E0-Q4-DEV-BINDING-V7.json`, SHA-256
  `58a7922d78d8974bf25011790704a75424fb77aa28472dc1d73a85eee11f7b8a`.
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

The successful path requires two prompt transitions, three generated target
transitions, one continuation, and a separately authenticated boundary. Seven
serial/capture snapshot pairs at prefix lengths `1, 2, 3, 4, 5, 5, 6` produce
exact sidecar size `2,199,912,448` bytes, below all frozen redesign thresholds.
At `2026-08-23T04:54:12Z`, the shared evidence volume had `68,445,863,936`
bytes available. Preserving both original and quarantine sidecars requires
`4,399,824,896` bytes, 6.43% of that free space and below the 25% limit.

- Trace: `/tmp/qwen-dflash-e0-q4-v7-write-code-20260823.jsonl`.
- State: `/tmp/qwen-dflash-e0-q4-v7-write-code-20260823.state.bin`.
- Reduction: `/tmp/qwen-dflash-e0-q4-v7-write-code-20260823.reduction.json`.

All outputs must be absent, exclusive-create, and canonically distinct from one
another and all inputs/executable. Exactly one producer and one reducer
invocation are authorized. Preserve every outcome without overwrite or retry.
No further development E0 acquisition is authorized by this phase after any v7
outcome.

## Frozen Commands

```sh
QWEN_METAL_LEASE_WAIT=1 target/debug/qwen-bench dflash-e0-lockstep --model /Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf --drafter /Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --binding-manifest docs/bench/2026-08-22-dflash-sampled-evidence/E0-Q4-DEV-BINDING-V7.json --prompt "Write code" --tokens 4 --temperature 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 0 --arm-order serial-then-capture --output /tmp/qwen-dflash-e0-q4-v7-write-code-20260823.jsonl --state-sidecar /tmp/qwen-dflash-e0-q4-v7-write-code-20260823.state.bin --fixture-id e0-q4-write-code-v7 --fixture-role development-sentinel --target-arm qwen3.8-27b-q4_k_m --drafter-arm dflash2-q4_k_m-post-prereg-development
```

```sh
uv run --no-project scripts/profile/dflash_e0_evidence.py --input /tmp/qwen-dflash-e0-q4-v7-write-code-20260823.jsonl --output /tmp/qwen-dflash-e0-q4-v7-write-code-20260823.reduction.json
```

## Decision Rule

V7 passes only if producer/reducer exit zero, the reduction reports exactly one
passing run with `emitted=4` and `transitions=6`, the trace contains exactly two
`prompt_step` and three `target_transition` rows, and the fully covered sidecar
is exactly `2,199,912,448` bytes. Earlier EOS fails this objective even if narrow
parity holds. Any mismatch or invalid artifact is preserved and stops
acquisition.

A pass adds one short two-token trajectory at generated-history depth. It does
not establish meaningful code behavior, code coverage, instruction-following,
reverse order for this prompt, realistic prompt breadth, sustained contexts,
hidden residual semantics, Q8/BF16, sparse-q/K0, verifier, economics,
performance, serving, held-out, or product authority. The next phase is
lane-separated K0/economic problem framing, not another easy E0 sentinel.
