# v0.440 Replay Timing Reset and Policy Model

Goal: remove the remaining timing confound from the real-window replay economics
harness and make S8 replay policy decisions trace/model driven rather than based
on a single full-occupancy packet.

## What changed

- Timed real-window replay rows now reset each working session from an immutable
  prepared seed before every warmup and timed repetition.
- The reset copies `x`, GDN conv/state, KV K/V, and `kv_n_pos` so every repetition
  starts from the same block-slice input state.
- `scripts/profile/replay_economics.py` now parses timed
  `decode-block-slice-real-margin` rows, applies fallback packets, and can model
  active-slot occupancy traces or FIFO request traces for p95 shadow-policy
  checks.

This corrects v0.438: those timing rows reused sessions after prior repetitions
had mutated the state, so they should be treated as shape smoke only.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --file /Users/tito/code/llm/game/v02_3.6_run.json \
  --file /Users/tito/code/llm/game/marcus_full.json \
  --tokens 8 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0439-a3b-real-economics-s8-c512-2048-b0-b20.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 6 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0439-a3b-real-economics-s6-c512-2048-b0-b20.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --tokens 4 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0439-a3b-real-economics-s4-c512-2048-b0-b20.out

uv run python scripts/profile/replay_economics.py \
  --margin-summary target/profiles/v0438-a3b-real-margin-blocks2-s8-summary.tsv \
  --real-margin target/profiles/v0439-a3b-real-economics-s4-c512-2048-b0-b20.out \
  --real-margin target/profiles/v0439-a3b-real-economics-s6-c512-2048-b0-b20.out \
  --real-margin target/profiles/v0439-a3b-real-economics-s8-c512-2048-b0-b20.out \
  --occupancy '4=0.2,6=0.3,8=0.5' \
  > target/profiles/v0439-replay-economics-policy-example.tsv

uv run python scripts/profile/replay_economics.py \
  --margin-summary target/profiles/v0438-a3b-real-margin-blocks2-s8-summary.tsv \
  --real-margin target/profiles/v0439-a3b-real-economics-s4-c512-2048-b0-b20.out \
  --real-margin target/profiles/v0439-a3b-real-economics-s6-c512-2048-b0-b20.out \
  --real-margin target/profiles/v0439-a3b-real-economics-s8-c512-2048-b0-b20.out \
  --request-trace - --interpolate-slots \
  > target/profiles/v0441-replay-request-sim-smoke.tsv <<'EOF'
arrival_ms tokens id
0 8 a
0 8 b
0 8 c
0 8 d
0 8 e
0 8 f
0 8 g
0 8 h
0 8 i
0 8 j
0 8 k
0 8 l
0 8 m
0 8 n
0 8 o
0 8 p
EOF
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- smoke row: S1 context 16 with corrected reset path
- corrected S4/S6/S8 real-prompt rows at contexts 512 and 2048
- replay policy script occupancy and request-simulation smoke on corrected rows

## Results

S8, timing rows, no observed fallback slots at threshold `3e-4` in these four
rows:

| Window | Context | Gross save | Validated net | Net after 6.67% fallback |
| --- | ---: | ---: | ---: | ---: |
| `block0..2` | `512` | `21.69%` | `10.75%` | `4.08%` |
| `block20..22` | `512` | `21.42%` | `11.21%` | `4.54%` |
| `block0..2` | `2048` | `21.59%` | `12.80%` | `6.13%` |
| `block20..22` | `2048` | `20.94%` | `11.31%` | `4.64%` |

S6 remains too thin:

| Window | Context | Gross save | Validated net | Net after 6.67% fallback |
| --- | ---: | ---: | ---: | ---: |
| `block0..2` | `512` | `14.99%` | `3.02%` | `-3.65%` |
| `block20..22` | `512` | `16.27%` | `5.05%` | `-1.62%` |
| `block0..2` | `2048` | `15.83%` | `3.28%` | `-3.39%` |
| `block20..22` | `2048` | `14.73%` | `4.57%` | `-2.10%` |

S4 stays negative before any broader fallback charge:

| Window | Context | Gross save | Validated net |
| --- | ---: | ---: | ---: |
| `block0..2` | `512` | `2.57%` | `-12.76%` |
| `block20..22` | `512` | `2.44%` | `-11.07%` |
| `block0..2` | `2048` | `4.37%` | `-12.04%` |
| `block20..22` | `2048` | `4.71%` | `-9.33%` |

The policy-model smoke is intentionally illustrative, not a measured workload
trace: with token-step occupancy weights `4=0.2,6=0.3,8=0.5`, the S8-only policy
blends to `2.38%` after applying the `3e-4` fallback packet. This shows why the
next gate must be a real occupancy trace rather than another full-S8 micro row.

A FIFO request-simulation smoke with 16 simultaneous 8-token requests and capacity
8 reproduces the full-S8 fallback-adjusted save (`4.85%`) and improves p95 by the
same amount because occupancy is always 8. That is a harness check, not product
evidence.

## Decision

Replay is still alive only at full S8 occupancy, and even there the conservative
margin guard narrows the win to the lower edge of the required gate. The next
replay branch must be a shadow S8-only policy model over real occupancy and p95
latency. Do not build S4/S6 replay, and do not build production replay until the
shadow model clears `>=5-8%` blended net after fallback.
