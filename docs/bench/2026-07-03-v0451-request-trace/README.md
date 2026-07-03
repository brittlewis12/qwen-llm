# v0.451 Request-Trace Seam for Replay Economics

Goal: add the smallest trace-only bridge between real `qwen` invocations and the
existing replay economics model, without building a scheduler or changing decode
policy.

## What changed

- `qwen -p ... --trace-request PATH` appends one TSV row after successful
  generation: `arrival_ms`, generated `tokens`, request `id`, and `prompt_tokens`.
- `scripts/profile/replay_economics.py --request-trace` now normalizes request
  arrivals by subtracting the first row, so traces can use either relative
  arrivals or epoch milliseconds from real CLI invocations.

## Commands

```sh
cargo check -p qwen-cli --bin qwen
cargo build --release -p qwen-cli --bin qwen

rm -f target/profiles/v0451-qwen-request-trace.tsv
target/release/qwen \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -p "Hello" -n 2 \
  --trace-request target/profiles/v0451-qwen-request-trace.tsv \
  > target/profiles/v0451-qwen-trace-smoke.out \
  2> target/profiles/v0451-qwen-trace-smoke.err

uv run scripts/profile/replay_economics.py \
  --real-margin target/profiles/v0449-a3b-real-economics-s8-c8192-16384-b0-b20.out \
  --request-trace target/profiles/v0451-qwen-request-trace.tsv \
  --interpolate-slots \
  > target/profiles/v0451-qwen-request-trace-economics.tsv
```

## Result

Trace smoke row:

```tsv
arrival_ms	tokens	id	prompt_tokens
1783046071211	2	37568-1783046071211	1
```

The economics parser now normalizes that epoch timestamp and reports the expected
single-request shape: occupancy `1:2`, `0.00%` replay save because the S8 policy
cannot fire on a one-slot trace.

## Decision

This is measurement plumbing only. It does not justify a replay scheduler, and it
does not change the v0.449 result. The next replay decision still needs a real
multi-request trace with enough overlapping decode work to test whether S8 replay
is deployable outside the harness.
