# v0.391 Captured MoE Down Microbench

Extended the captured MoE replay harness from routed gate/up into routed down.
`capture_moe_gateup_replay_for_token` now records top-k weights as well as ids
and hidden states, and `moe-down-micro --route-capture-ctx` replays per-layer
route ids/weights from a real decode context.

The down microbench also defaults the `f_exp=512` Q5 path to the production R2
kernel, with `--legacy-k512` as the old-kernel override. This keeps the microbench
aligned with the decode path instead of timing a stale legacy default.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0391-a3b-q4-moe-down-micro-captured-default-r2.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0391-a10b-q4xl-moe-down-micro-captured-default.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0391-a3b-q4-moe-gateup-micro-captured-ctx1024.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0391-a10b-q4xl-moe-gateup-micro-captured-ctx1024.out
```

## Results

| Model | Harness | Captured micro | Matching phase row | Read |
| --- | --- | ---: | ---: | --- |
| A3B Q4 | Q4 gate/up | `1.0794 ms` | `1.08 ms` | phase-faithful |
| A3B Q4 | Q5 down R2 | `0.8144 ms` | `0.82 ms` | phase-faithful |
| A10B Q4_XL | Q4 gate/up | `3.1068 ms` | `3.21 ms` | phase-faithful |
| A10B Q4_XL | Q5 down | `2.5946 ms` | `2.56 ms` | phase-faithful |

The stale A3B down micro shape explains why this fix matters: the old synthetic
legacy path timed at `1.4286 ms` GPU, while captured production R2 is `0.8144 ms`
and matches the full phase row.

## Decision

Use captured MoE microbenching as the gate for the next compute branch. A
candidate must first move captured gate/up or down by at least `5-10%`, then
survive a full phase row with route included and route replay excluded. Synthetic
MoE micro wins are not promotion evidence unless captured replay agrees.
