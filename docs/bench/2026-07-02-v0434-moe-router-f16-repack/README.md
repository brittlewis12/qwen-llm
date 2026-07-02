# v0.434 MoE Router F16 Repack

Goal: test the do-less hypothesis that MoE router `gate_inp` is a realized
decode/precompute byte bottleneck. The experiment keeps CPU F32 router weights
for reference, stores the Metal `gate_inp` tensor as F16 when
`QWEN_MOE_ROUTER_F16=1`, and checks GPU route top-k against CPU F32 logits on
real post-mixer hidden states.

Commands:

```sh
QWEN_MOE_ROUTER_F16=1 target/release/qwen-bench decode-moe-router-repack-check \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/qwen-llm/crates/qwen-cli/src/bench.rs \
  --file /Users/tito/code/qwen-llm/docs/PERF-ROADMAP.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 4 --context 128,512,2048 \
  > target/profiles/v0434-a3b-router-f16-check-mixed4-c128-512-2048.out

QWEN_MOE_ROUTER_F16=1 target/release/qwen-bench decode-moe-router-repack-check \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/qwen-llm/crates/qwen-cli/src/bench.rs \
  --file /Users/tito/code/qwen-llm/docs/PERF-ROADMAP.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 4 --context 128,512 \
  > target/profiles/v0434-a10b-router-f16-check-mixed4-c128-512.out

target/release/qwen-bench tg \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --n-gen 128 --runs 3 -o json \
  > target/profiles/v0434-a3b-tg128-router-f16-base-a.json

QWEN_MOE_ROUTER_F16=1 target/release/qwen-bench tg \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --n-gen 128 --runs 3 -o json \
  > target/profiles/v0434-a3b-tg128-router-f16-on-a.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --n-prompt 4096 --runs 1 --no-warmup --cooldown-seconds 10 \
  --repeat-blocks 2 --shuffle-seed 435 \
  --variant base --variant router-f16:QWEN_MOE_ROUTER_F16=1 \
  --output target/profiles/v0434-a3b-pp4096-router-f16-sweep.json

QWEN_MOE_ROUTER_F16=1 cargo test --release -p qwen-llm \
  prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --ignored --nocapture
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B and A10B real-prompt router top-k equivalence checks
- A3B ignored MoE prefill-vs-single gate with `QWEN_MOE_ROUTER_F16=1`
- A3B/A10B decode and prefill spot measurements
- cx adversarial review session `019f2399-2e38-7c73-adb1-6e4ace1b29d3`

## Results

Correctness/top-k signal:

| Model | Contexts | Route checks | Route order mismatches | Route set mismatches | Max logit abs | Min F32 margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A3B | `128,512,2048` | `480` | `0` | `0` | `0.000020` | `0.000196` |
| A10B | `128,512` | `384` | `0` | `0` | `0.000052` | `0.000057` |

The ignored A3B MoE prefill-vs-single gate passes with the F16 router and F16
admitted to the packed route path. This avoids the token-loop prefill hazard cx
flagged during review.

Performance signal:

| Model / Shape | Base | Router F16 | Read |
| --- | ---: | ---: | --- |
| A3B `tg128` ABBA | `107.91/108.03` | `108.24/108.02` | flat/noise |
| A10B `tg128` AB | `45.10` | `45.21` | flat/noise |
| A3B `pp512` | `1406/1381` | `1412/1416` | small positive/noisy |
| A3B `pp4096` | `1616/1619` | `1633/1633` | `~+1%` |
| A10B `pp1024` | `129.23` | `114.53` | regresses hard |

Phase split at `ctx128` confirms decode did not move: A3B route logits were
`0.31 ms` base vs `0.32 ms` F16; A10B route logits were `0.51 ms` base vs
`0.50 ms` F16.

## Decision

Do not default. Keep `QWEN_MOE_ROUTER_F16=1` and
`decode-moe-router-repack-check` as an opt-in diagnostic / exact-route harness,
but demote router repack as a performance branch. The route top-k safety signal
is encouraging, yet decode is flat and A10B prefill regresses because the F16
router loses the specialized F32 E8xP32 route-logits path.

Do not widen to BF16 unless a future branch has a different router kernel shape.
Next highest-leverage do-less item remains the fused online-softmax matrix
attention body.
